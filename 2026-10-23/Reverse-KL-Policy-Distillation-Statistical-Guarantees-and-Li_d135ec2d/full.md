# Reverse-KL Policy Distillation: Statistical Guarantees and Limits

Haolin Liu University of Virginia srs8rh@virginia.edu

Chen-Yu Wei University of Virginia chenyu.wei@virginia.edu

Julian Zimmert   
Google Research /   
National Taiwan University   
zimmert@google.com

## Abstract

Reverse Kullback–Leibler (KL) distillation has shown empirical promise for training language models, but its statistical guarantees and limits remain poorly understood. In particular, the effects of two design choices remain unclear: Does access to the teacher’s full action distribution provide an advantage over observing only the probability it assigns to the sampled action? Must training trajectories come from the student (online distillation), or can teacher-generated trajectories (offline distillation) suffice? Our work takes a first step toward understanding the theory underlying these questions.

We study the sample complexity of online reverse-KL distillation under two feedback models: the teacher reveals either its full action distribution or only the probability of the student-generated action. We establish minimax bounds for the former and show that substantially more data may be required for the latter. In contrast, offline reverse-KL distillation can attain the same minimax rate whether the teacher reveals its full distribution or only its probability of the teacher-generated action. The key distinction is that teachergenerated actions themselves reveal the teacher’s preferences, while queries at student-selected actions can leave those preferences unobserved. This highlights an additional statistical challenge for online distillation compared to offline distillation.

These statistical guarantees do not follow simply from optimizing standard distillation objectives: our lower bounds show that common online objectives can fail when candidates are evaluated only on past studentgenerated trajectories. We develop trajectory-loss and importance-ratio truncation techniques together with policy-space learning to achieve optimal online distillation guarantees.

## 1 Introduction

Policy distillation compresses a teacher into a student that may lack the capacity to reproduce the teacher’s behavior in full. The choice of divergence therefore matters. Forward Kullback–Leibler (KL) divergence (i.e., KL(teacher∥student)) encourages the student to cover the teacher’s full range of responses. Reverse KL (i.e., KL(student∥teacher)) instead allows a limited-capacity student to concentrate on teacher behaviors it can represent, while discouraging responses that are unlikely under the teacher. Reverse KL has shown empirical promise in on-policy distillation for large language models (LLMs), where the student generates responses and learns from the teacher’s action probability at the observed state [Gu et al., 2024, Agarwal et al., 2024, Lu and Thinking Machines Lab, 2025].

Although the empirical results are promising, the optimal statistical guarantees for reverse-KL learning remain unclear. The statistical question depends first on what information the teacher provides. Some distillation methods only use the teacher’s probability on student-generated actions [Lu and Thinking Machines Lab, 2025], whereas recent studies report benefits from access to the teacher’s full action distribution [Li et al., 2026b, Yu et al., 2026, Fu et al., 2026]. Whether this richer form of teacher feedback improves statistical efficiency, and by how much, remains open. Moreover, reverse KL is evaluated under the student’s distribution, so training typically uses student-generated responses. It is therefore unclear whether learning from offline teacher-generated data can match or even improve on the statistical guarantees attainable from online student rollouts.

To answer these questions, we study sequence-level reverse-KL learning with a student policy class that may not contain the teacher, allowing for misspecification. Our framework distinguishes two aspects of the learning process: who generates the training trajectories and what information the teacher provides along them. The trajectories come either from student policies chosen adaptively by the learner in the online setting, or from the teacher in the offline setting. With either data source, the teacher may reveal its entire action distribution at each visited state (fullfeedback) or only the probability of the observed action (actionfeedback). Across these four settings, we study how closely a learner can match the best student in its policy class for a given number of training trajectories.

We give a rather complete picture for this problem. With full feedback, we establish the minimax-optimal guarantees for reverse-KL learning under a relative second-moment assumption. For the online setting, we first show that common local distillation losses can be suboptimal. These objectives, motivated by DAgger and on-policy distillation [Ross et al., 2011, Lu and Thinking Machines Lab, 2025], evaluate candidate policy on states generated by the rollout policies, including state that a candidate itself would rarely visit, which can distort policy rankings. Our algorithm (Algorithm 1) corrects this mismatch by reweighting the observed prefix to account for each policy’s own generation behavior, attaining the optimal guarantee (Section 3). For the offline setting, we develop a new algorithm (Equation (12)) that attains the same optimal guarantee from teacher-generated trajectories.

With action feedback only, online learning can require substantially more samples than with full feedback (Section 3.4). Yet under a trajectory likelihood-ratio bound, offline learning with action feedback can still obtain a comparable guarantee to full feedback learning (Sections 3.4 and 4). These results show that richer feedback can improve online sample efficiency, while surprisingly optimal reverse-KL learning need not require online interaction.

Formally, let $\pi ^ { \star }$ be the teacher policy, Π be the student policy class, and $P ^ { \pi }$ the sequence distribution generated by π. Our guarantees take the form

$$
\mathrm { K L } ( P ^ { \widehat { \pi } } \| P ^ { \pi ^ { \star } } ) \leq C _ { \mathrm { a p p } } \operatorname* { m i n } _ { \pi \in \Pi } \mathrm { K L } ( P ^ { \pi } \| P ^ { \pi ^ { \star } } ) + C _ { \mathrm { e s t } } \frac { \log ( | \Pi | / \delta ) } { T }
$$

with probability at least $1 - \delta .$ , using T complete episodes and returning $\widehat { \pi } \in \Pi$ . Here $C _ { \mathrm { a p p } }$ (approximation factor) measures the approximation cost of competing with the best available student, and $C _ { \mathrm { e s t } }$ (estimation factor) controls the estimation error. At the minimax scale of $C _ { \mathrm { e s t } }$ , we characterize the smallest attainable $C _ { \mathrm { a p p } }$ Table 1 summarizes the guarantees and separations.

Table 1: Reverse-KL learning by data source and teacher feedback. Approximation coefficients are compared at the optimal estimation scale.
<table><tr><td>Data source</td><td>Feedback Assumption</td><td> $C _ { \mathrm { a p p } }$ </td><td> $C _ { \mathrm { e s t } }$ </td></tr><tr><td></td><td>Relative second moment</td><td> $\Theta ( B )$ </td><td> $\Theta ( B )$ </td></tr><tr><td>Online / Offline Full</td><td></td><td> $\mathbb { E } _ { P ^ { \pi } } [ K _ { \pi } ^ { 2 } ] \le B \mathbb { E } _ { P ^ { \pi } } [ K _ { \pi } ]$ </td><td></td></tr><tr><td>Offline</td><td>Action</td><td> $P ^ { \pi } ( \tau ) \leq e ^ { B _ { \infty } } P ^ { \pi ^ { \star } } ( \tau )$ </td><td> $\Theta ( B _ { \infty } )$   $\Theta ( B _ { \infty } )$ </td></tr><tr><td>Online Action</td><td> $P ^ { \pi } ( \tau ) \leq e ^ { B _ { \infty } } P ^ { \pi ^ { \star } } ( \tau )$ </td><td></td><td> $\begin{array} { r l } { - } & { { } \Omega { \Big ( } { \frac { B _ { \infty } m } { \log ( e m ) } } { \Big ) } } \end{array}$ </td></tr><tr><td colspan="4">Here  $\begin{array} { r } { K _ { \pi } ( \tau ) \ = \ \sum _ { h = 1 } ^ { H } \mathrm { K L } ( \pi _ { h } ( \cdot \ \vert \ s _ { h } ) \vert \vert \pi _ { h } ^ { \star } ( \cdot \ \vert \ s _ { h } ) ) } \end{array}$  for every feasible trajectory  $\begin{array} { l c l } { { \tau } } & { { = } } & { { \left( s _ { 1 : H } , a _ { 1 : H } \right) } } \end{array}$  , and  $\begin{array} { r l } { m } & { { } = } \end{array}$  min  $\{ | \mathcal { A } | , \dot { | \Pi | } , e ^ { B _ { \infty } } \overline  { \} }$  . The assumptions hold for every π and feasible trajectory τ. The trajectory likelihood-ratio condition</td></tr></table>

Related work. Our closest theoretical predecessors Rohatgi et al. [2025] and Xu et al. [2026b] also study imitation learning under misspecification. Rohatgi et al. [2025] obtain constant approximation and estimation coefficients for learning trajectory Hellinger distance with computationally inefficient algorithms, and show that standard next-token prediction approaches (e.g. cross-entropy loss) must suffer an $\Omega ( H )$ approximation factor where H is the learning horizon. Xu et al. [2026b] establish a constant approximation factor for learning forward KL divergence under standard cross-entropy loss, but the estimation error scales with a boundedness factor. Both papers only study offline learning from teacher trajectories with action feedback. We study reverse-KL learning in both online and offline settings and characterize the statistical advantage of full feedback over action feedback. Appendix A provides further discussion of related work.

## 2 Problem formulation

Notation. For a positive integer n, write $[ n ] = \{ 1 , \dots , n \}$ . All logarithms are natural unless a base is displayed. We use x $\wedge y = \operatorname* { m i n } \{ x , y \} , x _ { + } = \operatorname* { m a x } \{ x , 0 \}$ , and $\mathbf { 1 } \{ E \}$ for the indicator of an event E. The notation $O _ { \mathrm { { r } } }$ Ω, and Θ denotes upper, lower, and two-sided bounds up to positive numerical factors, and $\lesssim$ suppresses such an upper-bound factor. We write $x \asymp y$ when both $x \lesssim y$ and $y \lesssim x .$ . For distributions $p , q .$ , write $\begin{array} { r } { \mathrm { K L } ( p \| q ) = \sum _ { z } p ( z ) \log ( p ( z ) / q ( z ) ) } \end{array}$ , where terms with $p ( z ) = 0$ equal zero and $\mathrm { K L } ( p \| q ) = + \infty \operatorname { i f } p ( z ) > 0$ and $q ( z ) = 0$ for some z.

Setting. We consider a reward-free Markov decision process with horizon $H \geq 1$ , discrete state space ${ \mathcal { S } } _ { : }$ , action set A, initial state distribution $\mu ,$ and transition kernel $\mathsf { P } _ { h } ( \cdot \mid s , a )$ . A Markov policy $\pi = ( \pi _ { h } ) _ { h = 1 } ^ { H }$ consists of maps $\pi _ { h } : S \to \Delta ( { \mathcal { A } } )$ , where $\begin{array} { r } { \Delta ( \mathcal { A } ) = \{ p \in [ 0 , 1 ] ^ { | \mathcal { A } | } : \sum _ { a \in \mathcal { A } } p ( a ) = 1 \} } \end{array}$ is the probability simplex over actions. We write $\pi _ { h } ( \boldsymbol { a } \mid \boldsymbol { s } )$ for the probability assigned to action a at state s. Let $s _ { h } \in S$ and $a _ { h } \in { \mathcal { A } }$ denote the state and action at step h. Under $\pi _ { \mathfrak { c } }$ , an episode generates a trajectory $\tau = ( s _ { 1 } , a _ { 1 } , \ldots , s _ { H } , a _ { H } ) \in \mathcal { T } = ( S \times \mathcal { A } ) ^ { H }$ by sampling $s _ { 1 } \sim \mu , a _ { h } \sim \pi _ { h } ( \cdot \mid s _ { h } )$ , and, for $h < H , s _ { h + 1 } \sim \mathsf { P } _ { h } ( \cdot \mid s _ { h } , a _ { h } )$ . We write $P ^ { \pi }$ for the trajectory distribution under policy π, so $\begin{array} { r } { P ^ { \pi } ( \tau ) \stackrel { } { = } \mu ( s _ { 1 } ) \prod _ { h = 1 } ^ { H } \pi _ { h } ( a _ { h } \ \vert \ s _ { h } ) \prod _ { h = 1 } ^ { H - 1 } \mathsf { P } _ { h } ( s _ { h + 1 } \ \vert \ s _ { h } , a _ { h } ) } \end{array}$ . Let $\xi _ { h } = ( s _ { 1 } , a _ { 1 } , \ldots , s _ { h - 1 } , a _ { h - 1 } , s _ { h } )$ denote the prefix before action $a _ { h }$ . Its probability under π is $P ^ { \pi } ( \xi _ { h } ) =$ $\begin{array} { r } { \mu ( s _ { 1 } ) \prod _ { j = 1 } ^ { h - 1 } \pi _ { j } ( a _ { j } \mid s _ { j } ) \mathsf { P } _ { j } ( s _ { j + 1 } \mid s _ { j } , a _ { j } ) } \end{array}$ . The empty product equals one when $h = 1$

The learner is given a finite policy class Π and seeks to imitate a fixed teacher $\pi ^ { \star }$ . Define the local reverse KL between policy π and $\pi ^ { \star }$ at state s and step h by $k _ { h } ( \boldsymbol { s } ; \pi ) = \mathrm { K L } ( \pi _ { h } ( \cdot \mid \boldsymbol { s } ) \| \pi _ { h } ^ { \star } ( \cdot \mid \boldsymbol { s } ) )$ ). Our target is the trajectory reverse-KL risk, whose chain-rule decomposition is $\begin{array} { r } { L ( \pi ) : = \mathrm { K L } ( P ^ { \pi } | | P ^ { \pi ^ { \star } } ) = \mathbb { E } _ { P ^ { \pi } } \big [ \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { h } ; \pi ) \big ] } \end{array}$ . Let

$L ^ { \star } = \operatorname* { m i n } _ { \pi \in \Pi } L ( \pi )$ . The setting is realizable when $\pi ^ { \star } \in$ Π, giving $L ^ { \star } = 0$ , and agnostic when the teacher need not belong to the class.

Observation models. With a budget of $T$ complete episodes, we consider four observation models (see Table 1) that differ in their trajectory source and teacher feedback:

• Trajectory source: In the online setting, at each round $t \in [ T ]$ , the learner selects $\pi _ { t } \in \Pi$ based on past observations and executes it for a complete episode, obtaining $\tau _ { t } \sim P ^ { \pi _ { t } }$ . In the offline setting, the learner receives independent teacher trajectories $\tau _ { t } ^ { \star } \sim P ^ { \pi ^ { \star } }$ for $t \in [ T ]$

• Teacher feedback: At each visited state $s _ { h }$ , the learner receives one of two forms of teacher feedback. Full feedback reveals the teacher’s entire action distribution $\pi _ { h } ^ { \star } ( \cdot \mid s _ { h } )$ , whereas actionfeedback reveals only the probability $\pi _ { h } ^ { \star } ( a _ { h } \mid s _ { h } )$ assigned to the observed action $a _ { h }$ . Either feedback model can accompany online or offline trajectories, and no feedback is available at unvisited states.

Boundedness assumptions. Our full-feedback guarantees hold under a relative second-moment condition. For action feedback, we use a stronger trajectory likelihood-ratio bound.

Assumption 2.1 (Relative second-moment bound). There exists $B \geq 1$ such that, for every $\pi \in \Pi$

$$
\mathbb { E } _ { \tau \sim P ^ { \pi } } \left[ \left( \sum _ { h = 1 } ^ { H } \mathrm { K L } ( \pi _ { h } ( \cdot \mid s _ { h } ) \parallel \pi _ { h } ^ { \star } ( \cdot \mid s _ { h } ) ) \right) ^ { 2 } \right] \le B \mathrm { ~ K L } ( P ^ { \pi } \| P ^ { \pi ^ { \star } } ) < \infty .
$$

By the KL chain rule, the average cumulative local KL under $P ^ { \pi }$ equals the trajectory divergence $\mathrm { K L } ( P ^ { \pi } \| P ^ { \pi ^ { \star } } )$ Assumption 2.1 thus imposes a Bernstein-type moment condition, the second moment is controlled linearly by the mean under each candidate policy’s own trajectory distribution. Intuitively, trajectories with cumulative local KL much larger than B can contribute only a small fraction of the average reverse KL.

This condition is weaker than several uniform boundedness assumptions. For example, a uniform local KL bound $k _ { h } ( s ; \pi ) \le G$ for all $h , s ,$ , π implies Assumption 2.1 with $B = \operatorname* { m a x } \{ 1 , H G \}$ . The trajectory likelihood-ratio bound in Assumption 2.2 below also implies Assumption 2.1, with $B = 4 ( B _ { \infty } + 1 )$ . Appendix D.1 provides the precise implications and proofs.

Let $\tau _ { \mathrm { e n v } }$ denote the set of environment-feasible trajectories $\tau = ( s _ { 1 : H } , a _ { 1 : H } )$ satisfying $\begin{array} { r } { \mu ( s _ { 1 } ) \prod _ { h = 1 } ^ { H - 1 } \mathsf { P } _ { h } ( s _ { h + 1 } \mid } \end{array}$ $s _ { h } , a _ { h } ) > 0$ . Assumption 2.2 gives trajectory likelihood-ratio bound.

Assumption 2.2 (Trajectory likelihood-ratio bound). There is $B _ { \infty } > 1$ such that $P ^ { \pi } ( \tau ) \leq e ^ { B \infty } P ^ { \pi ^ { \star } } ( \tau )$ for every $\pi \in \Pi$ and $\tau \in \mathcal { T } _ { \mathrm { e n v } }$

The local bound $\pi _ { h } ( a \mid s ) \leq e ^ { G } \pi _ { h } ^ { \star } ( a \mid s )$ implies trajectory-level bound $P ^ { \pi } ( \tau ) \leq e ^ { H G } P ^ { \pi ^ { \star } } ( \tau )$ . Thus, B and $B _ { \infty }$ may themselves grow with H, so the absence of an explicit horizon factor in our guarantees does not imply horizon independence.

Like our Assumption 2.2, Assumption 1 of Xu et al. [2026b] also bounds likelihood ratios, but in the opposite direction $P ^ { \pi ^ { \star } } ( \tau ) / P ^ { \pi } ( \tau )$ rather than our $P ^ { \pi } ( \tau ) / { P ^ { \pi ^ { \star } } ( \tau ) }$ . Their assumption requires every student to cover the teacher for a forward-KL objective, whereas our Assumption 2.2 requires the teacher to cover every student for a reverse KL objective. Teacher coverage is more natural in many cases because a limited-capacity student may concentrate on teacher-supported behaviors without representing the teacher’s entire range of behaviors.

Approximation error and estimation error. Following Rohatgi et al. [2025] and Xu et al. [2026b], we seek a learner that returns ${ \widehat { \pi } } \in \Pi$ such that, with probability at least $1 - \delta .$

$$
L ( \widehat \pi ) \leq \underbrace { C _ { \mathrm { a p p } } L ^ { \star } } _ { \mathrm { a p p r o x i m a t i o n ~ e r r o r } } + \underbrace { C _ { \mathrm { e s t } } \frac { \log ( | \Pi | / \delta ) } { T } } _ { \mathrm { e s t i m a t i o n ~ e r r o r } } .
$$

The coefficients may depend on $B , B _ { \infty } , H , | \Pi | , | A |$ and the observation model. Here $L ^ { \star }$ is the smallest risk achievable by any student in Π. The coefficient $C _ { \mathrm { a p p } }$ measures how closely the learner competes with this best-in-class risk. The coefficient $C _ { \mathrm { e s t } }$ scales the estimation term log $( | \Pi | / \delta ) / T$ , determining how many episodes are needed for accurate learning. We fix $C _ { \mathrm { e s t } }$ at its minimax-optimal order and characterize the smallest attainable $C _ { \mathrm { a p p } }$

## 3 Online learning from student trajectories

In this section, we study online reverse-KL learning, where student policies generate trajectories and the teacher reveals its full action distribution at each visited state. We first show that exact refitting with common distillation losses can fail because it evaluates candidates under mismatched visitation distributions. We then introduce an algorithm that corrects this mismatch and attains the optimal guarantee.

## 3.1 Why common objectives can fail

A natural approach fits each new student to the teacher’s predictions on states visited during training, as in DAgger-style refitting and the local objectives used in on-policy distillation [Ross et al., 2011, Agarwal et al., 2024]. At round t, the current student $\pi _ { t }$ generates a batch of complete trajectories with full teacher feedback. Let $\mathcal { D } _ { t }$ contain all trajectories observed so far. The learner chooses the next student by

$$
\pi _ { t + 1 } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } Q _ { t } ( \pi ; \mathcal { D } _ { t } ) .\tag{1}
$$

The objective may use any nonempty subset of $\mathcal { D } _ { t }$ , including only the latest batch, or assign arbitrary nonnegative weights to complete trajectories. These choices may depend on the observed data but must be the same for all candidates. If $\pi _ { t }$ is already an empirical minimizer, the learner retains it.

For example, reverse-KL refitting minimizes a weighted sum of $\begin{array} { r } { \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { h } ; \pi ) } \end{array}$ over the selected trajectories. Such local reverse-KL losses appear in generalized knowledge distillation and recent on-policy distillation studies [Agarwal et al., 2024, Li et al., 2026a]. Other choices include forward KL, mixtures of the two KL directions, and clipped KL losses [Lin et al., 2020, Wu et al., 2025, Zhao et al., 2026]. Appendix B.1 specifies the allowed losses and transformations. The result concerns exact refitting with these objectives, not every gradient update used in the cited methods.

Changing which trajectories are retained does not necessarily correct the visitation mismatch. A candidate π is still evaluated at states visited by the rollout policies, rather than according to how often π itself would visit those states. In the construction below, a constant-probability event makes every observed trajectory favor a high-risk policy, so selecting a different subset cannot improve the empirical ranking.

Proposition 3.1. For every sufficiently large B, there is an $H = 2 , \left| \Pi \right| = 2$ instance satisfying Assumptions 2.1 and 2.2 with $B _ { \infty } = 2 B$ and $L ^ { \star } \leq e ^ { - \Omega ( B ) }$ on which the refitting rules above fail. Under uniform random initialization, after $T = \Theta ( B ^ { 3 } )$ episodes, the policy returned by the final refit has risk $\Omega ( B )$ with constant probability, even with adaptive selection and weighting ofpast trajectories.

The construction has a high-risk policy $\pi _ { \mathrm { b a d } }$ that frequently visits a state where the near-optimal policy $\pi _ { \mathrm { g o o d } }$ has a large local loss. Although $\pi _ { \mathrm { g o o d } }$ rarely reaches this state itself, its loss on trajectories from $\pi _ { \mathrm { b a d } }$ makes it look worse. With constant probability, random initialization selects $\pi _ { \mathrm { b a d } }$ and every subsequent refit retains it, even if past data are discarded. Our algorithm reduces the influence of prefixes that the evaluated policy would rarely reach.

## 3.2 Correcting visitation mismatch

Suppose $\pi _ { t }$ generates a trajectory $\left( s _ { 1 : H } , a _ { 1 : H } \right)$ at round $t \in [ T ]$ . Full teacher feedback reveals the local loss $k _ { h } ( s _ { h } ; \pi )$ for every candidate π at each visited state, but the visited prefixes are sampled according to $\pi _ { t }$ rather than π. Writing $\xi _ { h } = ( s _ { 1 } , a _ { 1 } , \ldots , s _ { h - 1 } , a _ { h - 1 } , s _ { h } )$ for the prefix before action $a _ { h }$ , standard importance weighting multiplies a loss at $\xi _ { h }$ by $P ^ { \pi } ( \xi _ { h } ) / P ^ { \pi _ { t } } ( \xi _ { h } )$ . Under rollouts from $\pi _ { t }$ , this changes its expected coefficient from $P ^ { \pi _ { t } } ( \xi _ { h } )$ to $P ^ { \pi } ( \xi _ { h } )$ . However, the importance ratio can be arbitrarily large when a prefix is rare under $\pi _ { t } ,$ making the resulting estimate highly sensitive to rare observations.

We therefore clip the weight at one, giving min $\left\{ 1 , P ^ { \pi } ( \xi _ { h } ) / P ^ { \pi _ { t } } ( \xi _ { h } ) \right\}$ . Any loss attached to $\xi _ { h }$ then enters the expectation with coefficient $P ^ { \pi _ { t } } ( \xi _ { h } )$ min $\{ 1 , P ^ { \pi } ( \xi _ { h } ) / P ^ { \pi _ { t } } ( \xi _ { h } ) \} = \operatorname* { m i n } \{ P ^ { \pi } ( \xi _ { h } ) , P ^ { \pi _ { t } } ( \xi _ { h } ) \}$ . Thus, a prefix $\xi _ { h }$ that π visits less often than $\pi _ { t }$ is downweighted to $P ^ { \pi } ( \xi _ { h } )$ , whereas a prefix that π visits more often is not amplified beyond $P ^ { \pi _ { t } } ( \xi _ { h } )$ . The ratio can be computed as $\begin{array} { r } { P ^ { \pi } ( \xi _ { h } ) / P ^ { \pi _ { t } } ( \xi _ { h } ) = \prod _ { u < h } \pi _ { u } ( a _ { u } | s _ { u } ) / \pi _ { t , u } ( a _ { u } | s _ { u } ) } \end{array}$ because the transition probabilities cancel.

Although each importance ratio is clipped, the local KL losses along a trajectory can still sum to a large value. We therefore consider the clipped cumulative local loss min $\left\{ 1 , \sum _ { h = 1 } ^ { \bar { H } } k _ { h } \big ( s _ { h } ; \pi \big ) \right\}$ . To combine this trajectorylevel clipping with the step-dependent prefix weights, we decompose the clipped total loss into increments. Specifically, let $\boldsymbol { e } _ { h } ( \xi _ { h } ; \pi )$ be the increase caused by adding the local loss at step $h _ { ; }$

$$
e _ { h } ( \xi _ { h } ; \pi ) = \operatorname* { m i n } \biggl \{ 1 , \sum _ { u = 1 } ^ { h } k _ { u } ( s _ { u } ; \pi ) \biggr \} - \operatorname* { m i n } \biggl \{ 1 , \sum _ { u = 1 } ^ { h - 1 } k _ { u } ( s _ { u } ; \pi ) \biggr \} .\tag{2}
$$

The increments $\boldsymbol { e } _ { h } ( \xi _ { h } ; \pi )$ are nonnegative and $\begin{array} { r } { \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) = \operatorname* { m i n } \{ 1 , \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { h } ; \pi ) \} \le 1 } \end{array}$

The two clipping operations reduce the influence of rare, high-loss observations, but they also discard information about the original reverse-KL risk. To quantify what remains, let $\begin{array} { r } { \overline { { L } } ( \pi ) : = \mathbb { E } _ { P ^ { \pi } } \big [ \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \big ] } \end{array}$ denote policy $\pi ^ { \prime } \boldsymbol { \mathrm { s } }$ expected clipped loss.

First, under Assumption 2.1, clipping incurs a factor $O ( B )$

$$
\begin{array} { r } { \overline { { L } } ( { \boldsymbol \pi } ) \leq L ( { \boldsymbol \pi } ) \lesssim B \overline { { L } } ( { \boldsymbol \pi } ) . } \end{array}\tag{3}
$$

Thus, controlling the clipped loss $\overline { { L } } ( \pi )$ suffices to control sequence risk $L ( \pi )$ , at a cost of B (Lemma D.2).

Second, the rollout policy $\pi _ { t }$ may rarely visit prefixes that are important to $\pi .$ . Since the importance weight is clipped, trajectories from $\pi _ { t }$ cannot fully recover $\pi ^ { \prime } \boldsymbol { \mathrm { s } }$ loss on prefixes $\xi _ { h }$ with $P ^ { \pi _ { t } } ( \xi _ { h } ) < P ^ { \pi } ( \xi _ { h } )$ . The clipped importance sampling ratios retain each increment $\boldsymbol { e } _ { h } ( \xi _ { h } ; \pi )$ with probability mass min $\{ P ^ { \pi } ( \xi _ { h } ) , P ^ { \pi _ { t } } ( \xi _ { h } ) \}$ rather than its full mass $P ^ { \pi } ( \xi _ { h } )$ in $\overline { { L } } ( \pi )$ . More generally, for any $\pi , \pi ^ { \prime } \in \Pi$ , we define

$$
\mathrm { T r u m } _ { \pi ^ { \prime } } \left[ \overline { { L } } ( \pi ) \right] : = \mathbb { E } _ { P ^ { \pi ^ { \prime } } } \left[ \sum _ { h = 1 } ^ { H } \operatorname* { m i n } \left\{ 1 , \frac { P ^ { \pi } ( \xi _ { h } ) } { P ^ { \pi ^ { \prime } } ( \xi _ { h } ) } \right\} e _ { h } ( \xi _ { h } ; \pi ) \right] = \sum _ { h , \xi _ { h } } \operatorname* { m i n } \{ P ^ { \pi } ( \xi _ { h } ) , P ^ { \pi ^ { \prime } } ( \xi _ { h } ) \} e _ { h } ( \xi _ { h } ; \pi ) .\tag{4}
$$

Here, $\pi ^ { \prime }$ in the subscript is the rollout policy, while π inside $\overline { { L } } ( \pi )$ is the policy being evaluated. In episode t, we take $\pi ^ { \prime } = \pi _ { t }$ . Since $\mathrm { T r u n } _ { \pi _ { t } } [ \overline { { L } } ( \pi ) ]$ retains only the prefix mass shared by $\pi$ and $\pi _ { t }$ , it may underestimate

$\overline { { L } } ( \pi )$ . Indeed, $0 \leq \mathrm { T r u n } _ { \pi _ { t } } [ \overline { { L } } ( \pi ) ] \leq \overline { { L } } ( \pi )$ . The key fact is that when $\overline { { L } } ( \pi _ { t } )$ is small, the truncated quantity still controls $\overline { { L } } ( \pi )$ . Lemma B.1 gives

$$
\begin{array} { r } { \overline { { L } } ( \pi ) \leq 6 4 \left( \operatorname { T r u n } _ { \pi _ { t } } \left[ \overline { { L } } ( \pi ) \right] + \overline { { L } } ( \pi _ { t } ) \right) . } \end{array}\tag{5}
$$

Equation (5) shows why the quality of the rollout policy matters. When $\overline { { L } } ( \pi _ { t } )$ is small, a candidate π with substantially larger $\overline { { L } } ( \pi )$ must also have a large $\mathrm { T r u n } _ { \pi _ { t } } \left[ \dot { \overline { { L } } } ( \pi ) \right]$ . Thus, even after truncation, losses observed on rollouts from $\pi _ { t }$ provide evidence against π. Since the learner does not know in advance how to choose a rollout policy $\pi _ { t }$ with small $\overline { { L } } ( \pi _ { t } )$ , the observed losses must guide both candidate evaluation and the choice of future rollout policies.

## 3.3 Online learning for reverse KL minimization

To let the observed losses guide future rollouts, our algorithm assigns each candidate $\pi \textbf { a }$ weight $w _ { t } ( \pi )$ and samples $\pi _ { t } \sim w _ { t }$ in episode t. After executing $\pi _ { t } .$ the learner observes the trajectory $\tau _ { t } = ( s _ { t , 1 : H } , a _ { t , 1 : H } )$ and full teacher feedback $\pi ^ { \star } ( \cdot | s _ { t , h } )$ for every $h \in [ H ]$ . These observations are used to compare every $\pi \in \Pi$ with $\pi _ { t } .$ . At the observed prefix $\xi _ { t , h } = ( s _ { t , 1 } , a _ { t , 1 } , \cdot \cdot \cdot , s _ { t , h } )$ , the loss $\ell _ { t } ( \pi )$ subtracts $e _ { h } ( \xi _ { t , h } ; \pi _ { t } )$ from $e _ { h } ( \xi _ { t , h } ; \pi )$ and applies the same clipped prefix importance sampling weight to both terms, yielding

$$
\ell _ { t } ( \pi ) = \sum _ { h = 1 } ^ { H } \operatorname* { m i n } \biggr \{ 1 , \frac { P ^ { \pi } ( \xi _ { t , h } ) } { P ^ { \pi _ { t } } ( \xi _ { t , h } ) } \biggr \} \left( e _ { h } ( \xi _ { t , h } ; \pi ) - e _ { h } ( \xi _ { t , h } ; \pi _ { t } ) \right) .\tag{6}
$$

Let $\mathbb { E } _ { t }$ denote expectation conditioned on the history before round $t ,$ averaging over both $\pi _ { t }$ and its trajectory. We have

$$
\begin{array} { r } { \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \right] = \mathbb { E } _ { \pi _ { t } \sim w _ { t } } \left[ \mathrm { T r u n } _ { \pi _ { t } } \left[ \overline { { L } } ( \pi ) \right] - \mathrm { T r u n } _ { \pi } \left[ \overline { { L } } ( \pi _ { t } ) \right] \right] . } \end{array}\tag{7}
$$

Thus in expectation, $\ell _ { t } ( \pi )$ compares the losses of $\pi$ and $\pi _ { t }$ under prefix probability mass min $\{ P ^ { \pi } ( \xi _ { t , h } )$ $P ^ { \pi _ { t } } ( \xi _ { t , h } ) \}$ . From clipping, each of the two weighted sums in Equation (6) lies in [0, 1], so $\ell _ { t } ( \pi ) \in [ - 1 , 1 ]$

Our algorithm is Algorithm 1. The exponential-weights update turns $\ell _ { t } ( \pi )$ into a change in sampling probability. Since $\ell _ { t } ( \pi _ { t } ) = 0$ , a positive $\ell _ { t } ( \pi )$ decreases the weight of π relative to $\pi _ { t }$ , while a negative value increases the weight of π. Candidates with smaller accumulated losses $\scriptstyle \sum _ { u = 1 } ^ { t } \ell _ { u } ( \pi )$ receive larger $w _ { t + 1 } ( \pi )$ and are therefore more likely to generate subsequent trajectories. The algorithm outputs the policy with the largest final weight $w _ { T + 1 }$

```latex
Algorithm 1 Online learning by bounded prefix comparisons
Require: Policy class Π, episode budget T.
1: Set $w _ { 1 } ( \pi ) = 1 / | \Pi |$ for all $\pi \in \Pi$ and $\eta = 1 / 2 5 6 .$
2: for $t = 1 , \dots , T$ do
3: Sample $\pi _ { t } \sim w _ { t }$ and execute it for the entire episode.
4: Observe the trajectory $\tau _ { t } = ( s _ { t , 1 : H } , a _ { t , 1 : H } )$ and the teacher distribution $\pi ^ { \star } ( \cdot | s _ { t , h } )$ for every $h \in [ H ]$
5: For every $\pi \in \Pi ,$ compute the loss $\ell _ { t } ( \pi )$ using Equation (6).
6: $w _ { t + 1 } ( \pi ) = \frac { w _ { t } ( \pi ) \exp ( - \eta \ell _ { t } ( \pi ) ) } { \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \exp ( - \eta \ell _ { t } ( \pi ^ { \prime } ) ) }$ for all $\pi \in \Pi .$
7: end for
8: Return $\widehat { \pi } \in \arg \operatorname* { m a x } _ { \pi \in \Pi } w _ { T + 1 } ( \pi )$ , breaking ties by the smallest candidate index.
```

The following theorem shows that Algorithm 1 achieves the optimal estimation scale while competing with $L ^ { \star }$

Theorem 3.2 (Online full-feedback guarantee). Under Assumption 2.1, for every $\delta \in ( 0 , 1 )$ , Algorithm 1 returns a candidate satisfying, with probability at least $1 - \delta ,$

$$
L ( \widehat \pi ) \leq O \bigg ( B \left[ L ^ { \star } + \frac { \log ( | \Pi | / \delta ) } { T } \right] \bigg ) .\tag{8}
$$

The $O ( \cdot )$ notation hides only universal numerical constants.

Under Assumption 2.2, Lemma D.1 gives Assumption 2.1 with $B = 4 ( B _ { \infty } + 1 )$ , so Equation (8) holds with $B _ { \infty }$ replacing B.

Theorem 3.3 establishes the tightness of both coefficients in Equation (8).

Theorem 3.3 (Optimality of full-feedback learning). For both online and offline learning with full feedback, the minimax estimation coefficient is $C _ { \mathrm { e s t } } = \Theta ( B )$ under Assumption 2.1 in the realizable setting. Any learner achieving $C _ { \mathrm { e s t } } = O ( B )$ on all instances must incur $C _ { \mathrm { a p p } } = \Omega ( B )$ in the worst case. Under Assumption 2.2, the same conclusions hold with $B _ { \infty }$ replacing B.

Together, Theorems 3.2 and 3.3 show that Algorithm 1 attains the optimal estimation rate and the smallest possible order of approximation coefficient at that rate. The estimation and approximation lower bounds are proved in Appendices B.4.1 and B.4.2, respectively. Below, we give a proof sketch for the upper bound.

Proofsketch ofTheorem 3.2. By the standard second-order exponential-weights analysis [Lattimore and Szepesvári, 2020, Eq. (27.2)–(27.3)], for a fixed minimizer $\pi ^ { \dagger } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } L ( \pi ) , \ell _ { t } ( \pi ) \in [ - 1 , 1 ]$ , and $\eta \leq 1$ we have

$$
- \underbrace { \sum _ { t = 1 } ^ { T } \ell _ { t } ( \pi ^ { \dagger } ) } _ { \mathrm { b e n c h m a r k l o s s } } \le \frac { \log | \Pi | } { \eta } - \underbrace { \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) } _ { \mathrm { w e i g h t e d ~ l o s s } } + \underbrace { \eta \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) ^ { 2 } } _ { \mathrm { q u a d r a t i c t e r m } } .\tag{9}
$$

We further bound the conditional expectation of the three terms above. For the benchmark loss,

$$
\begin{array} { r l r } { \mathbb { E } _ { t } \Big [ \ell _ { t } ( \boldsymbol { \pi } ^ { \dagger } ) \Big ] } & { = \mathbb { E } _ { \boldsymbol { \pi } _ { t } \sim w _ { t } } \left[ \mathrm { T r u n } _ { \boldsymbol { \pi } _ { t } } \left[ \overline { { L } } ( \boldsymbol { \pi } ^ { \dagger } ) \right] - \mathrm { T r u n } _ { \boldsymbol { \pi } ^ { \dagger } } \left[ \overline { { L } } ( \boldsymbol { \pi } _ { t } ) \right] \right] } & { { \scriptstyle ( \mathrm { E q u a t i o n ~ } ( 7 ) ) } } \\ & { \leq L ^ { \star } + \overline { { L } } ( \boldsymbol { \pi } ^ { \dagger } ) - \frac { 1 } { 6 4 } \mathbb { E } _ { \boldsymbol { \pi } _ { t } \sim w _ { t } } \left[ \overline { { L } } ( \boldsymbol { \pi } _ { t } ) \right] } & { { \scriptstyle ( \mathrm { T r u n } _ { \boldsymbol { \pi } _ { t } } \left[ \overline { { L } } ( \boldsymbol { \pi } ^ { \dagger } ) \right] \leq \overline { { L } } ( \boldsymbol { \pi } ^ { \dagger } ) \leq L ^ { \star } \mathrm { ~ a n d ~ E q u a t i o n ~ } ( 5 ) ) } } \\ & { \leq 2 L ^ { \star } - \frac { 1 } { 6 4 } \displaystyle \sum _ { \boldsymbol { \pi } \in \Pi } w _ { t } ( \boldsymbol { \pi } ) \overline { { L } } ( \boldsymbol { \pi } ) . } & { { \scriptstyle ( \overline { { L } } ( \boldsymbol { \pi } ^ { \dagger } ) \leq L ^ { \star } ) } } \end{array}
$$

The weighted loss has zero conditional expectation:

$$
\mathbb { E } _ { t } \left[ \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) \right] = \sum _ { \pi , \pi _ { t } \in \Pi } w _ { t } ( \pi ) w _ { t } ( \pi _ { t } ) \Big ( \mathrm { T r u n } _ { \pi _ { t } } \big [ \overline { { L } } ( \pi ) \big ] - \mathrm { T r u n } _ { \pi } \big [ \overline { { L } } ( \pi _ { t } ) \big ] \Big ) = 0 .\tag{10}
$$

For the quadratic term, since $e _ { h } ( \xi _ { t , h } ; \pi ) \in [ 0 , 1 ]$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \big [ \ell _ { t } ( \pi ) ^ { 2 } \big ] \leq \mathbb { E } _ { t } \Bigg [ \displaystyle \sum _ { h = 1 } ^ { H } \operatorname* { m i n } \Bigg \{ 1 , \frac { P ^ { \pi } \left( \xi _ { t , h } \right) } { P ^ { \pi } \left( \xi _ { t , h } \right) } \Bigg \} \left( e _ { h } ( \xi _ { t , h } ; \pi ) + e _ { h } ( \xi _ { t , h } ; \pi _ { t } ) \right) \Bigg ] } \\ & { \quad \quad \quad = \mathbb { E } _ { \pi _ { t } \sim w _ { t } } \left[ \mathrm { T r u n } _ { \pi _ { t } } \left[ \overline { { L } } ( \pi ) \right] + \mathrm { T r u n } _ { \pi } \left[ \overline { { L } } ( \pi _ { t } ) \right] \right] \leq \overline { { L } } ( \pi ) + \mathbb { E } _ { \pi _ { t } \sim w _ { t } } \left[ \overline { { L } } ( \pi _ { t } ) \right] . } \end{array}
$$

Weighting by $w _ { t } ( \pi )$ yields $\begin{array} { r } { \mathbb { E } _ { t } \left[ \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) ^ { 2 } \right] \leq 2 \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \overline { { L } } ( \pi ) } \end{array}$

Combining everything above with Equation (9) gives

$$
\frac { 1 } { 6 4 } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \overline { { L } } ( \pi ) \right] \leq 2 T L ^ { \star } + \frac { \log | \Pi | } { \eta } + 2 \eta \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \overline { { L } } ( \pi ) \right] .
$$

With $\eta = 1 / 2 5 6$ as in Algorithm 1, self-bounding yields E $\begin{array} { r } { \mathfrak { \backslash } \big [ \frac { 1 } { T } \sum _ { t } \sum _ { \pi } w _ { t } ( \pi ) \overline { { L } } ( \pi ) \big ] \leq O \big ( L ^ { \star } + \frac { \log | \Pi | } { T } \big ) } \end{array}$ , which further implies $\begin{array} { r } { \mathbb { E } \big [ \frac { 1 } { T } \sum _ { t } \sum _ { \pi } w _ { t } ( \pi ) L ( \pi ) \big ] \leq O \big ( B \big [ L ^ { \star } + \frac { \log | \Pi | } { T } \big ] \big ) } \end{array}$ by Equation (3) under Assumption 2.1.

Since $\pi _ { t } ~ \sim ~ w _ { t }$ , this gives an expected-risk bound when the output policy πb is uniformly sampled from $\{ \pi _ { 1 } , \cdots , \pi _ { T } \}$ . Our Algorithm 1 instead returns $\widehat { \pi } \in \arg \operatorname* { m a x } _ { \pi \in \Pi } w _ { T + 1 } ( \pi )$ for a high-probability bound. Its high-probability analysis uses the same second-moment bounds that control the quadratic term to bound accumulated sampling fluctuations. The full high-probability analysis appears in Appendix B.2. □

Extension to Hellinger distance. The same algorithmic approach also applies to Hellinger distance $\begin{array} { l } { { D _ { \mathrm { H } } ^ { 2 } ( P , Q ) ~ = ~ \sum _ { z } ( \sqrt { P ( z ) } - \sqrt { Q ( z ) } ) ^ { 2 } } } \end{array}$ We define policy $\pi ^ { \prime } \boldsymbol { \mathrm { s } }$ local loss on prefix $\xi _ { h }$ as $g _ { h } ( \xi _ { h } ; \pi ) \ =$ $\begin{array} { r l } { \sqrt { \frac { P ^ { \pi ^ { \star } } ( \xi _ { h } ) } { P ^ { \pi } ( \xi _ { h } ) } } D _ { \mathrm { H } } ^ { 2 } ( \pi _ { h } ( \cdot | s _ { h } ) , \pi _ { h } ^ { \star } ( \cdot | s _ { h } ) ) } & { { } } \end{array}$ with $\begin{array} { r l } { \mathbb { E } _ { P ^ { \pi } } \big [ \sum _ { h = 1 } ^ { H } g _ { h } ( \xi _ { h } ; \pi ) \big ] } & { { } = D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) } \end{array}$ . We also clip the cumulative local loss by letting $e _ { h } ^ { \mathrm { H } } ( \xi _ { h } ; \pi ) =$ min $\begin{array} { r } { \{ 1 , \sum _ { u = 1 } ^ { h } g _ { u } ( \xi _ { u } ; \pi ) \} } \end{array}$ − min $\begin{array} { r } { \{ 1 , \sum _ { u = 1 } ^ { h - 1 } g _ { u } ( \xi _ { u } ; \pi ) \} } \end{array}$ . The algorithm then uses losses $\ell _ { t } ( \pi ) = \sum _ { h = 1 } ^ { H }$ min $\begin{array} { r l } {  { \bigl \{ 1 , \frac { P ^ { \pi } ( \xi _ { t , h } ) } { P ^ { \pi _ { t } } ( \xi _ { t , h } ) } \bigr \} \ \bigl ( e _ { h } ^ { \mathrm { H } } ( \xi _ { t , h } ; \pi ) - e _ { h } ^ { \mathrm { H } } ( \xi _ { t , h } ; \pi _ { t } ) \bigr ) } \qquad } & { { } } \end{array}$ in the same exponential-weights update as Algorithm 1. The full algorithm for online Hellinger distance learning is provided in Algorithm 2. Theorem 3.4 establishes constant approximation and estimation coefficients for online learning with full feedback, matching the guarantee of Rohatgi et al. [2025] for offline learning from teacher trajectories.

Theorem 3.4. Let ${ \widehat { \pi } } \in \Pi$ be the output policy of Algorithm 2. With probability at least $1 - \delta , D _ { \mathrm { H } } ^ { 2 } ( P ^ { \widehat { \pi } } , P ^ { \pi ^ { \star } } ) \lesssim$ min $\mathfrak { l } _ { \pi \in \Pi } D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) + \log ( | \Pi | / \delta ) / T$

Under Assumption 2.2, every policy $\pi \in$ Π satisfies $D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) \leq \mathrm { K L } \left( P ^ { \pi } \| P ^ { \pi ^ { \star } } \right) \lesssim B _ { \infty } D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } )$ because $( { \sqrt { x } } - 1 ) ^ { 2 } \leq$ x log $x - x + 1 \lesssim B _ { \infty } ( \sqrt { x } - 1 ) ^ { 2 }$ for $0 \leq x \leq e ^ { B _ { \infty } }$ . Thus, the guarantee of Hellinger distance can be transferred to reverse KL divergence at the cost of $B _ { \infty }$ . Thus, Theorem 3.4 directly implies $L ( \widehat { \pi } ) \leq O ( B _ { \infty } \left[ L ^ { \star } + \log ( | \Pi | / \delta ) / T \right] )$ . However, a similar reduction does not hold under Assumption 2.1 (see the construction in Proposition C.1), so Algorithm 1 provides more general guarantees.

## 3.4 The Limitation of Action Feedback in the Online Setting

Algorithm 1 uses the teacher’s full action distribution to evaluate the local KL losses of every $\pi \in \Pi$ . With action feedback, the learner only observes the teacher’s probability on its chosen action, while the probabilities of other actions remain unobserved. The following lower bound shows that this missing information can significantly increase the sample requirement of any learner, even in the realizable setting.

Theorem 3.5 (Online action-feedback lower bound). For a sufficiently large T, every online learner returning a policy $\widehat { \pi } \in$ Π with action feedback from the teacher has a realizable instance satisfying Assumption 2.2, on which $\begin{array} { r } { L ( \widehat { \pi } ) = \widetilde { \Omega } \Big ( \operatorname* { m i n } \{ | \boldsymbol { A } | , | \Pi | , e ^ { B _ { \infty } } \} \frac { B _ { \infty } \log | \Pi | } { T } \Big ) } \end{array}$ with probability at least $1 / 4 .$

Compared with the full-feedback guarantee in Theorem 3.2, this lower bound requires an additional factor of min $\{ | \mathcal { A } | , | \Pi | , e ^ { B _ { \infty } } \}$ in the estimation error even in the realizable setting. Full feedback therefore provides a statistical advantage over action feedback in the online learning setting.

## 4 Offline learning from teacher trajectories

In Section 3, the learner chooses a rollout policy in each round and uses the observed losses to guide future rollouts. In the offline setting, the learner instead receives $T$ independent teacher trajectories $\tau _ { t } ^ { \star } \sim P ^ { \pi ^ { \star } }$ with $\tau _ { t } ^ { \star } = ( s _ { t , 1 } ^ { \star } , a _ { t , 1 } ^ { \star } , \cdot \cdot \cdot , s _ { t , H } ^ { \star } , a _ { t , H } ^ { \star } )$ . We will discuss algorithms to deal with both full feedback and action feedback for the offline setting in this section.

## 4.1 Offline learning with full feedback

Full feedback reveals $k _ { h } ( s _ { t , h } ^ { \star } ; \pi )$ for every candidate $\pi$ at each teacher-visited state. A natural estimator minimizes the average accumulated local KL over the teacher trajectories,

$$
\widehat { \pi } _ { \mathrm { r a w } } \in \mathop { \arg \operatorname* { m i n } } _ { \pi \in \Pi } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { t , h } ^ { \star } ; \pi ) .\tag{11}
$$

However, Proposition 4.1 shows that although this algorithm can obtain the optimal estimation rate in the realizable setting, it suffers $C _ { \mathrm { a p p } } = \widetilde \Omega ( B ^ { 2 } )$ approximation error rate under misspecification even under the stronger pathwise bound $\begin{array} { r } { \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { h } ; \pi ) \le B } \end{array}$ , which is suboptimal compared with the lower bound in Theorem 3.3.

Proposition 4.1 (Unclipped offline ERM). Under Assumption $2 . I , i f \pi ^ { \star } \in \Pi$ , then for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta , L ( \widehat { \pi } _ { \mathrm { r a w } } ) \leq O ( B \log ( | \Pi | / \delta ) / T )$ . Under misspecification, even under the stronger pathwise bound $\begin{array} { r } { \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { h } ; \pi ) \le B } \end{array}$ as in Lemma $D . I ,$ , a uniform guarantee for Equation (11) with an estimation term vanishing as $T \to \infty$ requires $C _ { \mathrm { a p p } } = \widetilde \Omega ( B ^ { 2 } )$

The failure under misspecification arises because teacher trajectories may visit states that a candidate $\pi$ never reaches. Large local KL losses at these states do not contribute to $L ( \pi )$ , but can dominate the objective in Equation (11) and cause it to reject π even when $L ( \pi )$ is small. Similar to the online setting, our estimator limits each trajectory’s contribution by clipping the accumulated loss, yielding

$$
\widehat { \pi } \in \underset { \pi \in \Pi } { \arg \operatorname* { m i n } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \left\{ 1 , \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { t , h } ^ { \star } ; \pi ) \right\} .\tag{12}
$$

Theorem 4.2 (Offline full-feedback guarantee). The estimator in Equation (12) satisfies the high-probability guarantee ofTheorem 3.2 under Assumption 2.1.

Under Assumption 2.1, clipping preserves the optimal estimation rate in the realizable setting while improving the approximation coefficient from the $\widetilde \Omega ( B ^ { 2 } )$ required by the unclipped estimator to $O ( B )$ in the pathwise special case, matching the lower bound in Theorem 3.3.

We now provide a proof sketch for Theorem 4.2. Let $\begin{array} { r } { \widehat { \mu } ( \pi ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } } \end{array}$ min $\left\{ 1 , \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { t , h } ^ { \star } ; \pi ) \right\}$ and $\mu ( \pi ) =$ $\mathbb { E } _ { P ^ { \pi ^ { \star } } }$ - min $\left\{ 1 , \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { t , h } ^ { \star } ; \pi ) \right\} \Big ]$ . With probability at least $1 - \delta .$ , standard Bernstein concentration for bounded nonnegative losses gives for all $\pi \in \Pi$

$$
{ \widehat { \mu } } ( \pi ) \leq 3 \mu ( \pi ) + O \left( { \frac { \log ( | \Pi | / \delta ) } { T } } \right) , \qquad \mu ( \pi ) \leq 3 { \widehat { \mu } } ( \pi ) + O \left( { \frac { \log ( | \Pi | / \delta ) } { T } } \right) .
$$

Since $\widehat { \pi }$ minimizes $\widehat { \mu }$ in Equation (12), these inequalities give

$$
\mu ( \widehat \pi ) \leq 3 \widehat \mu ( \widehat \pi ) + O \left( \frac { \log ( | \Pi | / \delta ) } { T } \right) = 3 \operatorname* { m i n } _ { \pi \in \Pi } \widehat \mu ( \pi ) + O \left( \frac { \log ( | \Pi | / \delta ) } { T } \right) \leq 9 \operatorname* { m i n } _ { \pi \in \Pi } \mu ( \pi ) + O \left( \frac { \log ( | \Pi | / \delta ) } { T } \right) .
$$

To get the target risk bound on $L ( \pi )$ , we need to connect $\mu ( \pi )$ with $\overline { { L } } ( \pi )$ , and bound $L ( \pi )$ through Equation (3). The challenge is that $\mu ( \pi )$ averages losses under $P ^ { \pi ^ { \star } }$ , whereas $\overline { { L } } ( \pi )$ averages under $P ^ { \pi }$ . After clipping, this change of measure costs only a constant factor. Lemma B.1 establishes $\mu ( \pi ) \asymp \overline { { L } } ( \pi )$ . Thus, with probability at least $1 - \delta .$

$$
\overline { { L } } ( \widehat { \pi } ) \lesssim \mu ( \widehat { \pi } ) \lesssim \operatorname* { m i n } _ { \pi \in \Pi } \mu ( \pi ) + \frac { \log ( | \Pi | / \delta ) } { T } \lesssim \operatorname* { m i n } _ { \pi \in \Pi } \overline { { L } } ( \pi ) + \frac { \log ( | \Pi | / \delta ) } { T } \leq L ^ { \star } + \frac { \log ( | \Pi | / \delta ) } { T } .
$$

Equation (3) converts this bound to $L ( \widehat { \pi } )$ , introducing a factor B under Assumption 2.1. Appendices B.2.1 and C give the full change-of-measure and concentration arguments.

## 4.2 Offline learning with action feedback

Action feedback reveals the teacher’s probability only for the observed action, so the local KL losses in Equation (12) are generally unavailable. Even under realizability, Assumption 2.1 alone does not permit a uniformly vanishing risk bound for offline learners with action feedback (Proposition C.1). We instead only consider Assumption 2.2 and use the offline Hellinger estimator of Rohatgi et al. [2025], which returns $\widehat { \pi } \in \Pi$ satisfying, with probability at least $\begin{array} { r } { 1 - \delta , D _ { \mathrm { H } } ^ { 2 } ( \hat { P ^ { \pi } } , P ^ { \pi ^ { \star } } ) \lesssim \operatorname* { m i n } _ { \pi \in \Pi } \bar { D _ { \mathrm { H } } ^ { 2 } } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) + \log ( | \Pi | / \delta ) / T } \end{array}$ Under Assumption 2.2, the Hellinger-to-KL conversion in Section 3.3 gives $L ( \widehat \pi ) \lesssim B _ { \infty } D _ { \mathrm { H } } ^ { 2 } ( P ^ { \widehat \pi } , P ^ { \pi ^ { \star } } )$ and $\begin{array} { r } { \operatorname* { m i n } _ { \pi \in \Pi } D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) \le L ^ { \star } } \end{array}$ . Substituting these inequalities gives the following guarantee.

Proposition 4.3 (Offline action-feedback guarantee). Under Assumption 2.2, let $\widehat { \pi } \in \Pi$ be the output of the estimator in Equation (11) ofRohatgi et al. [2025]. For every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
L ( \widehat \pi ) \leq O \biggl ( B _ { \infty } \left[ L ^ { \star } + \frac { \log ( | \Pi | / \delta ) } { T } \right] \biggr ) .\tag{13}
$$

The lower bounds in Theorem 3.3 also hold for offline learning with action feedback under Assumption 2.2. Thus, the reduction from Hellinger to reverse KL learning is minimax optimal in this setting. Proposition 4.3 and Theorem 3.5 show that, under Assumption 2.2, offline action-feedback learning can require exponentially fewer episodes than online action-feedback learning in the worst case.

## 5 Conclusion

We characterize how data source and teacher feedback shape reverse-KL learning under misspecification. For the online learning setting, we show the failure of common empirical algorithms and provide the first algorithm achieving the optimal guarantee under full feedback and a relative second-moment condition. We further show that full feedback can significantly improve online sample efficiency over action feedback. For the offline setting, we also establish algorithms with tight bounds, demonstrating that under a trajectory likelihood ratio bound, teacher trajectories can avoid the additional sample cost of online learning with action feedback. Future work includes developing and empirically evaluating practical distillation algorithms based on our prefix reweighting and trajectory clipping mechanism, and determining the optimal statistical guarantees achievable by computationally efficient learners.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html.

Jonathan Chang, Masatoshi Uehara, Dhruv Sreenivas, Rahul Kidambi, and Wen Sun. Mitigating covariate shift in imitation learning via offline data with partial coverage. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://papers.nips.cc/paper/2021/hash/ 07d5938693cc3903b261e1a3844590ed-Abstract.html.

Nicolas Espinosa-Dice, Sanjiban Choudhury, Wen Sun, and Gokul Swamy. Efficient imitation under misspecification. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ ca96d9b87059a0b408ec1a1cd0ab46ec-Abstract-Conference.html.

Dylan J. Foster, Adam Block, and Dipendra Misra. Is behavior cloning all you need? understanding horizon in imitation learning. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ da84e39ae51fd26bb5110d9659c06e13-Abstract-Conference.html.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes. arXiv preprint arXiv:2603.25562v2, 2026. URL https://arxiv.org/abs/2603.25562v2.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 8ac015d409635f196f9e3e9dcfb9a94e-Abstract-Conference.html.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531, 2015. URL https://arxiv.org/abs/1503.02531.

Yoon Kim and Alexander M. Rush. Sequence-Level Knowledge Distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, 2016. doi: 10.18653/ v1/D16-1139. URL https://aclanthology.org/D16-1139/.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. DistiLLM: Towards streamlined distillation for large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 24872–24895. PMLR, 2024. URL https://proceedings.mlr.press/v235/ko24c.html.

Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and Se-Young Yun. DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 31044–31062. PMLR, 2025. URL https://proceedings.mlr.press/v267/ko25a.html.

Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020. doi: 10.1017/ 9781108571401. URL https://tor-lattimore.com/downloads/book/book.pdf.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016v2, 2026a. URL https: //arxiv.org/abs/2604.13016v2.

Yichen Li and Chicheng Zhang. Agnostic interactive imitation learning: New theory and practical algorithms. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research. PMLR, 2024. URL https://proceedings.mlr.press/v235/ li24ck.html.

Yijiang Li, Bingyang Wang, Yijun Liang, Yunjie Tian, Di Fu, and Nuno Vasconcelos. On-policy self-distillation without any supervision. arXiv preprint arXiv:2608.06296, 2026b. URL https://arxiv.org/abs/ 2608.06296.

Alexander Lin, Jeremy Wohlwend, Howard Chen, and Tao Lei. Autoregressive knowledge distillation through imitation learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 6121–6133, 2020. doi: 10.18653/v1/2020.emnlp-main.494. URL https://aclanthology.org/2020.emnlp-main.494/.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, October 2025. URL https://thinkingmachines.ai/blog/on-policy-distillation/. Technical blog, October 27, 2025.

Yury Polyanskiy and Yihong Wu. Information theory: From coding to learning, 2024. URL https://people. lids.mit.edu/yp/homepage/data/itbook-export.pdf. Author manuscript, August 16, 2024.

Nived Rajaraman, Lin F. Yang, Jiantao Jiao, and Kannan Ramchandran. Toward the Fundamental Limits of Imitation Learning. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1e7875cf32d306989d80c14308f3a099-Abstract.html.

Nived Rajaraman, Yanjun Han, Lin F. Yang, Kannan Ramchandran, and Jiantao Jiao. Provably breaking the quadratic error compounding barrier in imitation learning, optimally. arXiv preprint arXiv:2102.12948, 2021. URL https://arxiv.org/abs/2102.12948.

Dhruv Rohatgi, Adam Block, Audrey Huang, Akshay Krishnamurthy, and Dylan J. Foster. Computationalstatistical tradeoffs at the next-token prediction barrier: Autoregressive and imitation learning under misspecification. arXiv preprint arXiv:2502.12465v1, 2025. URL https://arxiv.org/abs/2502.12465v1. Full version. Extended abstract in Proceedings of Thirty Eighth Conference on Learning Theory, PMLR 291:4831–4837, 2025.

Stephane Ross and Drew Bagnell. Efficient reductions for imitation learning. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9, pages 661–668. PMLR, 2010. URL https://proceedings.mlr.press/v9/ross10a.html.

Stéphane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings ofthe Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings ofMachine Learning Research, pages 627–635. PMLR, 2011. URL https://proceedings.mlr.press/v15/ross11a.html.

Andrei A. Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins, James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia Hadsell. Policy Distillation. In International Conference on Learning Representations, 2016. URL https://arxiv.org/abs/1511.06295.

Igal Sason and Sergio Verdú. f-Divergence Inequalities. IEEE Transactions on Information Theory, 62(11): 5973–6006, 2016. doi: 10.1109/TIT.2016.2603151. URL https://arxiv.org/abs/1508.00335v7.

Ved Sriraman, Peihan Liu, Daniel Hsu, and Adam Block. Behavior cloning is not all you need: The optimality of on-policy distillation for noisy expert feedback. arXiv preprint arXiv:2606.30923, 2026. URL https: //arxiv.org/abs/2606.30923.

Luca Viano, Antoine Moulin, Audrey Huang, Volkan Cevher, Philip Amortila, and Dylan J. Foster. When does on-policy interaction help? representational tradeoffs in value-based imitation learning. arXiv preprint arXiv:2607.29617, 2026. URL https://arxiv.org/abs/2607.29617.

Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, and Ngai Wong. Rethinking Kullback– Leibler divergence in knowledge distillation for large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 5737–5755, 2025. URL https://aclanthology. org/2025.coling-main.383/.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026a.

Yunbei Xu, Yuzhe Yuan, and Ruohan Zhan. Autoregressive learning in joint KL: Sharp oracle bounds and lower bounds. arXiv preprint arXiv:2605.12316, 2026b. URL https://arxiv.org/abs/2605.12316.

Zichao Yu, Chengzhi Yu, Shengze Xu, Yujin Han, Bingqing Jiang, Xu Wang, and Difan Zou. Mismatch matters: On-policy distillation beyond token agreement. arXiv preprint arXiv:2608.09836, 2026. URL https://arxiv.org/abs/2608.09836.

Huaqing Zhang, Jingchu Gai, Juno Kim, Bingbin Liu, and Andrej Risteski. When Does Online Imitation Learning Help in LLM Post-Training? The Role of (Non-)Realizability Beyond Horizon. arXiv preprint arXiv:2606.30445, 2026. URL https://arxiv.org/abs/2606.30445.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734v3, 2026. URL https://arxiv.org/abs/2601.18734v3.

## Appendices

A Related work 16   
B Online learning 17   
B.1 Failure of exact empirical fitting 17   
B.2 Online reverse-KL learning . 19   
B.2.1 Clipping and prefix truncation 19   
B.2.2 Exponential-weights analysis . 24   
B.3 Online Hellinger learning from full feedback 28   
B.4 Estimation and approximation lower bounds . 32   
B.4.1 Estimation lower bound 32   
B.4.2 Approximation lower bound 34   
B.5 The additional cost of online action feedback 36   
C Offline learning 37   
C.1 Full feedback 37   
C.2 Action feedback . 39   
D Auxiliary lemmas 41   
D.1 Second moments and clipping 41   
D.2 KL and Hellinger inequalities . 43   
D.3 Concentration and testing bounds 44

## A Related work

Offline imitation learning theory Offline imitation learning asks what can be learned from a fixed set of expert demonstrations without further interaction. In the realizable setting, classical analyses of zero-one-loss behavior cloning relate action-prediction errors to compounding errors in task performance [Ross and Bagnell, 2010]. Rajaraman et al. [2020] characterize minimax task-performance guarantees in finite-state MDPs, showing that error compounding can be unavoidable when dynamics are unknown. With known dynamics and a deterministic expert, improved dependence on the horizon is possible [Rajaraman et al., 2021]. For general policy classes under realizability, Foster et al. [2024] show that log-loss behavior cloning admits horizon-free guarantees when the cumulative reward range and policy-class complexity are controlled. Additional offline data provide another source of information: MILO learns a dynamics model from auxiliary transition data and mitigates covariate shift when those data sufficiently cover the expert’s behavior [Chang et al., 2021]. Our offline learners instead use teacher trajectories and the specified probability feedback, without auxiliary transition data.

When the student policy class is not expressive enough to cover the teacher, the question becomes how closely a learner can compete with the best available policy. A common target is an oracle bound of the form $D ( \widehat { \pi } ; \pi ^ { \star } ) \leq$ $\begin{array} { r l } { { \cal { C } } _ { \mathrm { a p p } } \operatorname* { m i n } _ { \pi \in \Pi } D ( \pi ; \pi ^ { \star } ) + { \cal { C } } _ { \mathrm { e s t } } \log ( | \Pi | / \delta ) / T } \end{array}$ where $D ( \cdot ; \cdot )$ can be any distance measure. For trajectory-level Hellinger distance, Rohatgi et al. [2025] give an offline estimator with constant approximation and estimation coefficients, while establishing an $\Omega ( H )$ approximation barrier for a broad class of next-token methods. Xu et al. [2026b] study trajectory-level forward KL divergence, obtaining constant-factor approximation error with Θ(HG) estimation error where ${ \mathrm { s u p } } _ { \pi \in \Pi } { \mathrm { s u p } } _ { h \in [ H ] } { \mathrm { s u p } } _ { s _ { h } }$ log ${ \frac { \pi _ { h } ^ { \star } ( x _ { h } | s _ { h } ) } { \pi _ { h } ( x _ { h } | s _ { h } ) } } \leq G$ . Both Rohatgi et al. [2025] and Xu et al. [2026b] study offline learning with action feedback. In the same setting, we study reverse KL learning and show that $C _ { \mathrm { a p p } } , C _ { \mathrm { e s t } } = \Theta ( B _ { \infty } )$ under Assumption 2.2. Due to the different directions in forward and reverse KL, we assume the teacher can cover any student, whereas Xu et al. [2026b] assumes any student can cover the teacher. Teacher coverage is more natural for distillation because a limited-capacity student may concentrate on teacher-supported behaviors without representing the teacher’s entire range of behaviors.

Interactive imitation learning A student may visit states rarely represented in teacher demonstrations, especially after making a mistake [Ross and Bagnell, 2010]. DAgger obtains expert supervision on these states and reduces imitation to no-regret online learning [Ross et al., 2011]. Li and Zhang [2024] give an oracle-efficient agnostic algorithm with access to exploratory state samples. Its comparator measures local imitation error on states visited during training; ours evaluates each candidate under its own trajectory law. Thus a small online regret alone does not imply our reverse-KL oracle bound.

Several works identify conditions under which interaction improves imitation-learning performance. Espinosa-Dice et al. [2025] avoid quadratic error compounding under a completeness condition on the policy class. Zhang et al. [2026] identify an offline information bottleneck under misspecification even at horizon one. Sriraman et al. [2026] establish an exponential benefit from interaction with a noisy expert when performance is measured against a clean expert. Viano et al. [2026] obtain efficient on-policy imitation when the expert’s Q-function, rather than its policy, is realizable. These guarantees concern different targets and assumptions from our comparison of online and offline reverse-KL learning rates.

Language-model distillation Building on knowledge and policy distillation [Hinton et al., 2015, Rusu et al., 2016], language-model distillation can use teacher-generated responses [Kim and Rush, 2016] or studentgenerated responses [Lin et al., 2020, Agarwal et al., 2024, Li et al., 2026a], with forward KL and reverse KL as common losses. Beyond the standard losses, Wu et al. [2025] propose an adaptive combination of these two losses. Ko et al. [2024] propose skew KL losses that compare a distribution with a mixture of teacher and student distributions, while Ko et al. [2025] use a contrastive objective to increase the likelihood of teacher responses and decrease that of student responses. Gu et al. [2024] develop policy-gradient updates for sequence-level reverse KL, using returns based on accumulated future teacher-to-student log likelihood ratios. These returns allow later predictions to influence updates to earlier actions.

The feedback available at each training recipe is a separate choice. Lu and Thinking Machines Lab [2025] use teacher probabilities for sampled tokens, Xu et al. [2026a] use full next-token distributions, and Fu et al. [2026], Yu et al. [2026] incorporate information about teacher-preferred alternatives. Our full-versus-action-feedback comparison isolates the statistical value of the available information by comparing the best attainable rates under each feedback model.

## B Online learning

## B.1 Failure of exact empirical fitting

At round t, the current policy $\pi _ { t }$ generates a nonempty batch of H-step trajectories with full teacher feedback. Let $\mathcal { D } _ { t }$ contain all trajectories collected up to and including round t. The learner assigns weights $\omega _ { t , \tau } \geq 0$ with $\sum _ { \tau \in \mathcal { D } _ { t } } \omega _ { t , \tau } > 0$ and refits by

$$
\begin{array} { r l } & { \pi _ { t + 1 } \in \underset { \pi \in \Pi } { \arg \operatorname* { m i n } } Q _ { t } ( \pi ; \mathcal { D } _ { t } ) , } \\ & { Q _ { t } ( \pi ; \mathcal { D } _ { t } ) = \displaystyle \sum _ { \tau \in \mathcal { D } _ { t } } \omega _ { t , \tau } \phi _ { t } \left( \displaystyle \sum _ { h = 1 } ^ { H } d _ { t } ( \pi _ { h } ( \cdot \mid s _ { h } ) , \pi _ { h } ^ { \star } ( \cdot \mid s _ { h } ) ) \right) . } \end{array}\tag{14}
$$

The weights may depend on the observed history but are the same for all candidates. Assigning weight zero discards a trajectory, so this rule includes fitting any nonempty subset of the collected trajectories, using only the latest batch, or reusing all past data. If the current policy is already an empirical minimizer, it is retained: $\pi _ { t } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } Q _ { t } ( \pi ; \mathcal { D } _ { t } ) \Longrightarrow \pi _ { t + 1 } = \pi _ { t }$ . Thus a change of policy requires a strict decrease in the empirical objective. The output πb is the policy returned by the final step.

For student and teacher distributions $p , q$ , the local loss $d _ { t } ( p , q )$ is finite, nonnegative, zero exactly when $p = q$ and convex in $q .$ The function $\phi _ { t }$ is nondecreasing, with $\phi _ { t } ( 0 ) = 0$ and $\phi _ { t } ( u ) > 0$ for $u > 0$ . With $\phi _ { t } ( u ) = u$ forward KL uses $d _ { t } ( p , q ) = \operatorname { K L } ( q \| p )$ and has the same minimizers as the teacher-weighted cross-entropy $- \textstyle \sum _ { a } q ( a )$ log $p ( a )$ used by Lin et al. [2020]. Reverse KL uses $d _ { t } ( p , q ) = \operatorname { K L } ( p \| q )$ , as in the local distillation objective of Agarwal et al. [2024]. Skew reverse KL uses $d _ { t } ( p , q ) = \mathrm { K L } ( p \| \alpha _ { t } p + ( 1 - \alpha _ { t } ) q )$ for $\alpha _ { t } \in [ 0 , 1 )$ [Ko et al., 2024]. These losses are convex in q by joint convexity of KL. The class also includes mixtures of the two KL directions and clipping with $\phi _ { t } ( u ) = \operatorname* { m i n } \{ c _ { t } , u \}$ for $c _ { t } > 0$

Proposition 3.1 (detailed statement). For every sufficiently large λ, there is an $H = 2 , \left| \Pi \right| = 2$ instance satisfying Assumptions 2.1 and 2.2 with $B = 4 \lambda , B _ { \infty } = 8 \lambda$ , and $L ^ { \star } \lesssim \lambda e ^ { - \lambda }$ . Every update rule defined in Equation (14), initialized uniformly over Π and using $T = \lceil \lambda ^ { 3 } \rceil$ episodes, returns πb satisfying

$$
\mathbb { P } \{ L ( \widehat { \pi } ) \geq \lambda / 4 \} \geq \frac { 1 } { 4 } .
$$

ProofofProposition 3.1. MDP and policy class. The two-step MDP has

$$
\begin{array} { r } { \boldsymbol { A } = \{ a _ { 1 } , a _ { 2 } \} , \qquad \boldsymbol { S } _ { 1 } = \{ s _ { 1 , 1 } , s _ { 1 , 2 } \} , \qquad \boldsymbol { S } _ { 2 } = \{ s _ { 2 , 1 } , s _ { 2 , 2 } \} , \qquad \mu ( s _ { 1 , 1 } ) = \mu ( s _ { 1 , 2 } ) = \frac { 1 } { 2 } . } \end{array}
$$

Its transitions are

$$
\begin{array} { r l } { \mathsf { P } _ { 1 } ( s _ { 2 , 1 } \mid s _ { 1 , 1 } , a _ { 1 } ) = 1 , } & { } \\ { \mathsf { P } _ { 1 } ( s _ { 2 , 2 } \mid s , a ) = 1 } & { \big ( ( s , a ) \in ( S _ { 1 } \times \mathcal { A } ) \setminus \{ ( s _ { 1 , 1 } , a _ { 1 } ) \} \big ) . } \end{array}
$$

The episode ends after the second action. Set $\epsilon = { e ^ { - \lambda } , \bar { p } = 1 - \epsilon ^ { 2 } }$ , and $v = e ^ { - 4 \lambda }$ , and let $\Pi = \{ \pi _ { \mathrm { g o o d } } , \pi _ { \mathrm { b a d } } \}$ The table gives the probability of $a _ { 1 }$ at each state; the probability of $a _ { 2 }$ is its complement.

<table><tr><td>Policy</td><td> $s _ { 1 , 1 }$ </td><td> $s _ { 1 , 2 }$ </td><td> $s _ { 2 , 1 }$ </td><td> $s _ { 2 , 2 }$ </td></tr><tr><td> $\pi ^ { \star }$ </td><td>€</td><td>€</td><td>v</td><td> $1 / 2$ </td></tr><tr><td>πgood</td><td>€</td><td> $\epsilon ^ { 2 }$ </td><td>p</td><td> $1 / 2$ </td></tr><tr><td>πbad</td><td>p</td><td>€</td><td>v</td><td> $1 / 2$ </td></tr></table>

Write $\operatorname { B e r } ( p ) = ( p , 1 - p ) \operatorname { o n } \left( a _ { 1 } , a _ { 2 } \right)$ . All action probabilities are positive.

Boundedness and trajectory risks. The only candidate–state pairs at which the policies differ are listed below. For $\lambda \geq 1$ ，

$$
\begin{array} { r l r }  \left.  \frac { \pi } { \pi } \left. \begin{array} { c } { s _ { h } } \end{array} \right. \begin{array} { c } { \operatorname* { m a x } \frac { \pi _ { h } ( a  { \left| \begin{array} { l } { s _ { h } } \end{array} \right| } s _ { h } ) } { \pi _ { h } ^ { \star } ( a  { \left| \begin{array} { l } { s _ { h } } \end{array} \right| } s _ { h } ) } \qquad k _ { h } ( s _ { h } ; \pi ) } \\ { \pi _ { \mathrm { b a d } } \left. \begin{array} { l } { s _ { 1 , 1 } } \\ { s _ { 1 , 2 } } \\ { \pi _ { \mathrm { g o o d } } } \end{array} \right. \begin{array} { l } { ( 1 - \epsilon ^ { 2 } ) / ( 1 - \epsilon ) = 1 + \epsilon } \\ { \bar { p } / v \leq e ^ { 4 \lambda } \qquad \leq 4 \lambda } \end{array} \right. \begin{array} { l } { \leq \infty } \\ { \leq 4 \lambda } \end{array} } \end{array} \end{array}
$$

A feasible trajectory cannot contain both $s _ { 1 , 2 }$ and $s _ { 2 , 1 }$ . For each $\pi \in \Pi$ and $\tau \in \mathcal { T } _ { \mathrm { e n v } }$

$$
\begin{array} { r l r } {  { \sum _ { h = 1 } ^ { 2 } k _ { h } ( \boldsymbol { s } _ { h } ; \boldsymbol { \pi } ) \le 4 \lambda } } \\ & { } & { \log \frac { P ^ { \boldsymbol { \pi } } ( \boldsymbol { \tau } ) } { P ^ { \boldsymbol { \pi } \star } ( \boldsymbol { \tau } ) } = \displaystyle \sum _ { h = 1 } ^ { 2 } \log \frac { \pi _ { h } ( \boldsymbol { a } _ { h } \mid \boldsymbol { s } _ { h } ) } { \pi _ { h } ^ { \star } ( \boldsymbol { a } _ { h } \mid \boldsymbol { s } _ { h } ) } \le 4 \lambda \le 8 \lambda . } \end{array}
$$

Thus Assumptions 2.1 and 2.2 hold with the stated budgets, using Lemma D.1 for the first assumption. The trajectory risks satisfy

$$
\begin{array} { r l } & { L ( \pi _ { \mathrm { g o o d } } ) = \frac 1 2 \left[ k _ { 1 } ( s _ { 1 , 2 } ; \pi _ { \mathrm { g o o d } } ) + \epsilon k _ { 2 } ( s _ { 2 , 1 } ; \pi _ { \mathrm { g o o d } } ) \right] \le \frac 1 2 ( 1 + 4 \lambda ) \epsilon , } \\ & { L ( \pi _ { \mathrm { b a d } } ) = \frac 1 2 \operatorname { K L } ( \mathrm { B e r } ( \bar { p } ) | | \mathrm { B e r } ( \epsilon ) ) \ge \frac 1 2 ( \bar { p } \lambda - \log 2 ) \ge \lambda / 4 . } \end{array}\tag{15}
$$

The first equality in each line is the KL chain rule; the second line uses binary entropy at most log 2 and sufficiently large λ. Hence $L ^ { \star } = L ( \pi _ { \mathrm { g o o d } } ) \lesssim \lambda e ^ { - \lambda }$

Empirical loss on each observed trajectory. For $\theta = ( \bar { p } - \epsilon ) / ( \bar { p } - v ) \in ( 0 , 1 ) , \mathrm { B e r } ( \epsilon ) = \theta \mathrm { B e r } ( v ) + ( 1 - \theta ) \mathrm { B e r } ( \bar { p } )$ On a trajectory with $( s _ { 1 } , s _ { 2 } ) = ( s _ { 1 , 1 } , s _ { 2 , 1 } )$

$$
\begin{array} { r l r l } {  { \sum _ { h = 1 } ^ { 2 } d _ { t } ( \pi _ { \mathrm { b a d } , h } ( \cdot \mid s _ { h } ) , \pi _ { h } ^ { * } ( \cdot \mid s _ { h } ) ) } } \\ & { = \textstyle d _ { t } ( \mathrm { B e r } ( \bar { p } ) , \mathrm { B e r } ( \epsilon ) ) } \\ & { \le \theta d _ { t } ( \mathrm { B e r } ( \bar { p } ) , \mathrm { B e r } ( v ) ) + ( 1 - \theta ) d _ { t } ( \mathrm { B e r } ( \bar { p } ) , \mathrm { B e r } ( \bar { p } ) ) } & & { \textstyle ( \mathrm { C o n v e x i t y ~ i n ~ t h e ~ s e c o n d ~ p a r a m e t e r } ) } \\ & { = \theta d _ { t } ( \mathrm { B e r } ( \bar { p } ) , \mathrm { B e r } ( v ) ) < d _ { t } ( \mathrm { B e r } ( \bar { p } ) , \mathrm { B e r } ( v ) ) } & & { \textstyle ( d _ { t } ( p , p ) = 0 , 0 < \theta < 1 , v \neq \bar { p } ) } \\ & { = \displaystyle \sum _ { h = 1 } ^ { 2 } d _ { t } ( \pi _ { \mathrm { g o o d } , h } ( \cdot \mid s _ { h } ) , \pi _ { h } ^ { * } ( \cdot \mid s _ { h } ) ) . } & & { \textstyle ( d _ { t } ( \mathrm { B e r } ( \epsilon ) , \mathrm { B e r } ( \epsilon ) ) = 0 ) } & & { \textstyle ( 1 6 ) } \end{array}
$$

On a trajectory with $( s _ { 1 } , s _ { 2 } ) = ( s _ { 1 , 2 } , s _ { 2 , 2 } )$

$$
\begin{array} { l } { { \displaystyle \sum _ { h = 1 } ^ { 2 } d _ { t } ( \pi _ { \mathrm { b a d } , h } ( \cdot \mid s _ { h } ) , \pi _ { h } ^ { \star } ( \cdot \mid s _ { h } ) ) = 0 } \qquad \displaystyle ( \pi _ { \mathrm { b a d } } = \pi ^ { \star } \mathrm { ~ a t ~ b o t h ~ s t a t e s } ) }  \\ { { \displaystyle \quad < d _ { t } ( \mathrm { B e r } ( \epsilon ^ { 2 } ) , \mathrm { B e r } ( \epsilon ) ) } \qquad \displaystyle ( d _ { t } ( p , q ) > 0 \mathrm { ~ f o r ~ } p \neq q , \epsilon ^ { 2 } \neq \epsilon ) }  \\ { { \displaystyle = \sum _ { h = 1 } ^ { 2 } d _ { t } ( \pi _ { \mathrm { g o o d } , h } ( \cdot \mid s _ { h } ) , \pi _ { h } ^ { \star } ( \cdot \mid s _ { h } ) ) . \qquad \quad ( d _ { t } ( \mathrm { B e r } ( 1 / 2 ) , \mathrm { B e r } ( 1 / 2 ) ) = 0 ) \qquad ( 1 7 ) } } \end{array}
$$

Whenever every trajectory in $\mathcal { D } _ { t }$ follows one of these two state sequences,

$$
\begin{array} { r l r } {  { Q _ { t } \big ( \pi _ { \mathrm { b a d } } ; \mathcal { D } _ { t } \big ) = \sum _ { \tau \in \mathcal { D } _ { t } } \omega _ { t , \tau } \phi _ { t } \Bigg ( \sum _ { h = 1 } ^ { 2 } d _ { t } \big ( \pi _ { \mathrm { b a d } , h } ( \cdot \vert \ s _ { h } ) , \pi _ { h } ^ { \star } ( \cdot \vert s _ { h } ) \big ) \Bigg ) } } \\ & { } & { \leq \sum _ { \tau \in \mathcal { D } _ { t } } \omega _ { t , \tau } \phi _ { t } \Bigg ( \displaystyle \sum _ { h = 1 } ^ { 2 } d _ { t } \big ( \pi _ { \mathrm { g o o d } , h } ( \cdot \vert \ s _ { h } ) , \pi _ { h } ^ { \star } ( \cdot \vert s _ { h } ) \big ) \Bigg ) } \\ & { } & { \quad \qquad \mathrm { ( E q u a t i o n s ~ ( 1 6 ) ~ a n d ~ ( 1 7 ) } , \omega _ { t , \tau } \geq 0 , \phi _ { t } \mathrm { ~ n o n d e c r e a s i n g ~ } } \\ & { } & { \quad \qquad \mathrm { ( E q u a t i o n ~ ( 1 4 ) ) } , \qquad ( \mathrm { E q u a t i o n ~ ( 1 4 ) ) } ) \qquad ( 1 8 ) } \end{array}
$$

Since the update keeps the current policy whenever it is an empirical minimizer, Equation (18) gives

$$
\pi _ { t } = \pi _ { \mathrm { b a d } } \quad \implies \quad \pi _ { t + 1 } = \pi _ { \mathrm { b a d } } ,
$$

including ties caused by clipping.

Persistence andfailure probability. The only remaining state sequence is $s _ { 1 , 1 } \to s _ { 2 , 2 }$ . Under $\pi _ { \mathrm { b a d } }$ , its probability is

$$
P ^ { \pi _ { \mathrm { b a d } } } \{ s _ { 1 } = s _ { 1 , 1 } , s _ { 2 } = s _ { 2 , 2 } \} = { \textstyle { \frac { 1 } { 2 } } } \pi _ { \mathrm { b a d , 1 } } ( a _ { 2 } \mid s _ { 1 , 1 } ) = { \cfrac { \epsilon ^ { 2 } } { 2 } } .
$$

Let ${ \mathcal { E } } _ { m }$ be the event that $\pi _ { 1 } = \pi _ { \mathrm { b a d } }$ and none of the first m episodes takes $s _ { 1 , 1 } \to s _ { 2 , 2 }$ . On ${ \mathcal { E } } _ { m - 1 }$ , every completed refit keeps $\pi _ { \mathrm { b a d } }$ by Equation (18), so episode m also uses $\pi _ { \mathrm { b a d } }$ . Therefore,

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { E } _ { 0 } ) = \frac { 1 } { 2 } , \qquad \mathbb { P } ( \mathcal { E } _ { m } \mid \mathcal { E } _ { m - 1 } ) = 1 - \epsilon ^ { 2 } / 2 \quad ( m \in [ T ] ) , \qquad \widehat { \pi } = \pi _ { \mathrm { b a d } } \operatorname { o n } \mathcal { E } _ { T } . } \end{array}\tag{19}
$$

For arbitrary batch sizes and allowed trajectory weights,

$$
\begin{array} { l } { \mathbb { P } \{ L ( \widehat { \pi } ) \geq \lambda / 4 \} \geq \mathbb { P } ( \mathcal { E } _ { T } ) } \\ { \displaystyle = \frac { 1 } { 2 } \left( 1 - \frac { \epsilon ^ { 2 } } { 2 } \right) ^ { T } \geq \frac { 1 } { 2 } \left( 1 - \frac { T \epsilon ^ { 2 } } { 2 } \right) } \\ { \displaystyle = \frac { 1 } { 2 } \left( 1 - \frac { \lceil \lambda ^ { 3 } \rceil e ^ { - 2 \lambda } } { 2 } \right) \geq \frac { 1 } { 4 } . } \end{array}\tag{Equations (15) and (19)}
$$

(Bernoulli’s inequality)

(Sufficiently large λ)

## B.2 Online reverse-KL learning

## B.2.1 Clipping and prefix truncation

Lemma B.1 (Clipped objectives and truncation). $I f L ( \pi ) < \infty$ for every candidate, then

$$
\mathbb { E } _ { P ^ { \pi ^ { \star } } } \left[ \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \right] \asymp \overline { { L } } ( \pi ) , \qquad D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) \le 6 \overline { { L } } ( \pi ) .\tag{20}
$$

For every $\pi , \pi ^ { \prime } \in \Pi , 0 \leq \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ] \leq \overline { { L } } ( \pi )$ and

$$
\mathrm { T r u n } _ { \pi ^ { \prime } } \bigl [ \overline { { L } } ( \pi ) \bigr ] \geq \frac { 1 } { 1 6 } \overline { { L } } ( \pi ) - D _ { \mathrm { H } } ^ { 2 } \bigl ( P ^ { \pi ^ { \star } } , P ^ { \pi ^ { \prime } } \bigr ) .\tag{21}
$$

Consequently,

$$
\begin{array} { r } { \overline { { L } } ( \pi ) \leq 6 4 \left( \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ] + \overline { { L } } ( \pi ^ { \prime } ) \right) . } \end{array}
$$

To prove Lemma B.1, we first introduce several auxiliary lemmas to control the behaviour of the clipped loss. Lemma B.2 (Change of measure for a bounded loss). For probability laws $R , S$ and a measurable $f \in [ 0 , 1 ]$ KL $\iota ( R \| S ) \leq \mathbb { E } _ { R } \left[ f \right]$ implies $\mathbb { E } _ { S } \left[ f \right] \asymp \mathbb { E } _ { R } \left[ f \right]$

Proof. Lemma D.3, applied to $g = \lambda f$ , gives for every $\lambda \in \mathbb { R }$

$$
\begin{array} { r } { \mathrm { K L } ( R \| S ) \geq \lambda \mathbb { E } _ { R } \left[ f \right] - \log \mathbb { E } _ { S } \left[ e ^ { \lambda f } \right] \geq \lambda \mathbb { E } _ { R } \left[ f \right] - ( e ^ { \lambda } - 1 ) \mathbb { E } _ { S } \left[ f \right] . } \end{array}
$$

The second inequality uses convexity of the exponential on $[ 0 , 1 ]$ . The choices $\lambda = 2$ and $\lambda = - 1$ , respectively, give $( e ^ { 2 } - 1 ) \mathbb { E } _ { S } \left[ f \right] \geq \mathbb { E } _ { R } \left[ f \right]$ and $( 1 - e ^ { - 1 } ) \mathbb { E } _ { S } \left[ f \right] \leq 2 \mathbb { E } _ { R } \left[ f \right]$ □

Each term in $\mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ]$ can be written as min $\{ e _ { h } ( \xi _ { h } ; \pi ) P ^ { \pi } ( \xi _ { h } ) , e _ { h } ( \xi _ { h } ; \pi ) P ^ { \pi ^ { \prime } } ( \xi _ { h } ) \}$ . For a fixed candidate $\pi$ and any trajectory distribution $P _ { \mathrm { : } }$ , we therefore define an auxiliary distribution $\mathcal { R } _ { \pi } [ P ]$ that assigns probability $e _ { h } ( \xi _ { h } ; \pi ) P ( \xi _ { h } )$ to each prefix outcome $( h , \xi _ { h } ) , h \in [ H ]$ . The distribution also includes complete-trajectory outcomes $( \infty , \tau )$ for the remaining probability, and is defined by

$$
\begin{array} { l } { { \displaystyle \mathcal { R } _ { \pi } [ P ] ( h , \xi _ { h } ) = e _ { h } ( \xi _ { h } ; \pi ) P ( \xi _ { h } ) , } } \\ { ~ } \\ { { \displaystyle \mathcal { R } _ { \pi } [ P ] ( \infty , \tau ) = \left( 1 - \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \right) P ( \tau ) . } } \end{array}\tag{22}
$$

The label $\infty$ distinguishes a complete trajectory from a prefix. The difference between $\tau$ and the last prefix $\xi _ { H }$ is that $\xi _ { H }$ do not have $a _ { H }$ but $\tau$ has. Given $\tau \sim P$ , these probabilities correspond to retaining its prefix $( h , \xi _ { h } )$ with probability $\boldsymbol { e } _ { h } ( \xi _ { h } ; \pi )$ and retaining $( \infty , \tau )$ with the remaining probability. Since $\begin{array} { r l } { ~ } & { { } \sum _ { h = 1 } ^ { H } \bar { e _ { h } } ( \xi _ { h } ; \pi ) = } \end{array}$ min $\left\{ 1 , \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { h } ; \pi ) \right\}$ , all probabilities are nonnegative and sum to one.

Lemma B.3 (KL between auxiliary distributions). For $L ( \pi ) < \infty$

$$
\mathrm { K L } ( \mathcal { R } _ { \pi } [ P ^ { \pi } ] \| \mathcal { R } _ { \pi } [ P ^ { \pi ^ { \star } } ] ) = \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { h = 1 } ^ { H } \left( 1 - \sum _ { v = 1 } ^ { h } k _ { v } ( s _ { v } ; \pi ) \right) _ { + } k _ { h } ( s _ { h } ; \pi ) \right] \le \overline { { L } } ( \pi ) .\tag{23}
$$

Proof. Fix $\pi \in \Pi$ . For each ${ \widetilde { \pi } } \in \Pi \cup \{ \pi ^ { \star } \}$ , define the joint law $\mathbb { P } _ { \widetilde { \pi } }$ by $\tau \sim P ^ { \widetilde { \pi } }$ and $\sigma \in [ H ] \cup \{ \infty \}$ with conditional distribution

$$
\mathbb { P } _ { \widetilde { \pi } } ( \sigma = h \mid \tau ) = e _ { h } ( \xi _ { h } ; \pi ) , \qquad h \in [ H ] ,
$$

$$
\mathbb { P } _ { \widetilde { \pi } } ( \sigma = \infty \mid \tau ) = 1 - \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) .
$$

By Equation (2), $e _ { h } ( \xi _ { h } ; \pi ) \ge 0$ and $\begin{array} { r } { \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \le 1 } \end{array}$ , so these probabilities sum to one. For $h \in [ H ]$

$$
\mathbb { P } _ { \widetilde { \pi } } ( \sigma \geq h \mid \tau ) = 1 - \sum _ { v = 1 } ^ { h - 1 } e _ { v } ( \xi _ { v } ; \pi ) ,\tag{24}
$$

$$
\mathbb { P } _ { \widetilde { \pi } } \left( \boldsymbol { \sigma } > h \mid \tau \right) = 1 - \sum _ { v = 1 } ^ { h } e _ { v } ( \xi _ { v } ; \pi ) = \left( 1 - \sum _ { v = 1 } ^ { h } k _ { v } ( s _ { v } ; \pi ) \right) _ { + } .\tag{25}
$$

where the second equation uses $1 - \operatorname* { m i n } \{ 1 , x \} = ( 1 - x ) _ { + }$ . Consequently, whenever $\textstyle \sum _ { v = 1 } ^ { h - 1 } e _ { v } ( \xi _ { v } ; \pi ) < 1$

$$
\mathbb { P } _ { \widetilde { \pi } } ( \sigma = h \mid \sigma \geq h , \tau ) = \frac { e _ { h } ( \xi _ { h } ; \pi ) } { 1 - \sum _ { v = 1 } ^ { h - 1 } e _ { v } ( \xi _ { v } ; \pi ) } = \mathbb { P } _ { \widetilde { \pi } } ( \sigma = h \mid \sigma \geq h , \xi _ { h } ) .
$$

The last equality holds because the ratio depends only on $\xi _ { h }$

Define the retained record by

$$
Y = \left\{ \begin{array} { l l } { ( \sigma , \xi _ { \sigma } ) , } & { \sigma \in [ H ] , } \\ { ( \infty , \tau ) , } & { \sigma = \infty . } \end{array} \right.
$$

Since $\boldsymbol { e } _ { h } ( \xi _ { h } ; \pi )$ depends only on $\xi _ { h }$ ,

$$
\begin{array} { l } { { \mathbb { P } _ { \widetilde { \pi } } ( Y = ( h , \xi _ { h } ) ) = P ^ { \widetilde { \pi } } ( \xi _ { h } ) \mathbb { P } _ { \widetilde { \pi } } ( \sigma = h \mid \xi _ { h } ) } } \\ { { \displaystyle \quad = P ^ { \widetilde { \pi } } ( \xi _ { h } ) e _ { h } ( \xi _ { h } ; \pi ) = \mathcal { R } _ { \pi } [ P ^ { \widetilde { \pi } } ] ( h , \xi _ { h } ) , } } \\ { { \displaystyle \mathbb { P } _ { \widetilde { \pi } } ( Y = ( \infty , \tau ) ) = P ^ { \widetilde { \pi } } ( \tau ) \left( 1 - \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \right) = \mathcal { R } _ { \pi } [ P ^ { \widetilde { \pi } } ] ( \infty , \tau ) . } } \end{array}
$$

Thus $Y \sim \mathcal { R } _ { \pi } [ P ^ { \widetilde { \pi } } ]$ . From Equation (22), for any outcome with $\mathcal { R } _ { \pi } [ P ^ { \pi } ] ( Y ) > 0$ , the common stopping factor cancels from the likelihood ratio. When $\sigma \in [ H ]$

$$
\frac { \mathcal { R } _ { \pi } [ P ^ { \pi } ] ( Y ) } { \mathcal { R } _ { \pi } [ P ^ { \pi ^ { \star } } ] ( Y ) } = \frac { e _ { \sigma } ( \xi _ { \sigma } ; \pi ) P ^ { \pi } ( \xi _ { \sigma } ) } { e _ { \sigma } ( \xi _ { \sigma } ; \pi ) P ^ { \pi ^ { \star } } ( \xi _ { \sigma } ) } = \frac { P ^ { \pi } ( \xi _ { \sigma } ) } { P ^ { \pi ^ { \star } } ( \xi _ { \sigma } ) } .
$$

When $\sigma = \infty$ , the common factor $\begin{array} { r } { 1 - \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) } \end{array}$ cancels in the same way, leaving $P ^ { \pi } ( \tau ) / { P ^ { \pi ^ { \star } } ( \tau ) }$ . The two policies share the initial-state distribution and environment transitions, so

$$
\log \frac { \mathcal { R } _ { \pi } [ P ^ { \pi } ] ( Y ) } { \mathcal { R } _ { \pi } [ P ^ { \pi ^ { \star } } ] ( Y ) } = \sum _ { h = 1 } ^ { H } \mathbf { 1 } \{ \sigma > h \} \log \frac { \pi _ { h } ( a _ { h } \mid s _ { h } ) } { \pi _ { h } ^ { \star } ( a _ { h } \mid s _ { h } ) } .
$$

Thus, we have

$$
\begin{array} { r l } & { \mathrm { K L } ( \mathcal { R } _ { \pi } [ P ^ { \kappa } ] | | \mathcal { R } _ { \kappa } [ P ^ { \kappa } ] | ) } \\ & { \quad = \mathbb { E } _ { \mathbb { P } _ { \kappa } } \Bigg [ \displaystyle \sum _ { k = 1 } ^ { H } \mathbf { 1 } \{ \sigma > h \} \log \frac { \pi _ { k } ( a _ { k } | \ s _ { k } ) } { \pi _ { k } ^ { k } ( a _ { k } | \ s _ { k } ) } \Bigg ] } \\ & { \quad = \mathbb { E } _ { P ^ { \kappa } } \Bigg [ \displaystyle \sum _ { k = 1 } ^ { H } \mathbb { P } _ { \pi } ( \sigma > h \mid \tau ) \log \frac { \pi _ { k } ( a _ { k } | \ s _ { k } ) } { \pi _ { k } ^ { k } ( a _ { k } | \ s _ { k } ) } \Bigg ] } \\ & { \quad = \mathbb { E } _ { P ^ { \kappa } } \Bigg [ \displaystyle \sum _ { k = 1 } ^ { H } \Bigg ( 1 - \displaystyle \sum _ { \nu = 1 } ^ { h } k _ { \nu } ( s _ { \nu } ; \pi ) \Bigg ) _ { + } \log \frac { \pi _ { k } ( a _ { k } | \ s _ { k } ) } { \pi _ { k } ^ { k } ( a _ { k } | \ s _ { k } ) } \Bigg ] } \\ & { \quad = \mathbb { E } _ { P ^ { \kappa } } \Bigg [ \displaystyle \sum _ { k = 1 } ^ { H } \Bigg ( 1 - \displaystyle \sum _ { \nu = 1 } ^ { h } k _ { \nu } ( s _ { \nu } ; \pi ) \Bigg ) _ { + } k _ { h } ( s _ { k } | \sigma _ { k } ) \Bigg ] . } \end{array}\tag{Equation (25)}
$$

By Equation (2), $\boldsymbol { e } _ { h } ( \xi _ { h } ; \pi ) = \boldsymbol { k } _ { h } ( s _ { h } ; \pi )$ when $\textstyle \sum _ { v = 1 } ^ { h } k _ { v } ( s _ { v } ; \pi ) < 1$ , otherwise, $\begin{array} { r } { ( 1 - \sum _ { v = 1 } ^ { h } k _ { v } ( s _ { v } ; \pi ) ) _ { + } = 0 } \end{array}$ Thus,

$$
\begin{array} { r l } { \displaystyle \left( 1 - \sum _ { v = 1 } ^ { h } k _ { v } ( s _ { v } ; \pi ) \right) _ { + } k _ { h } ( s _ { h } ; \pi ) \leq 1 \left\{ \displaystyle \sum _ { v = 1 } ^ { h } k _ { v } ( s _ { v } ; \pi ) < 1 \right\} k _ { h } ( s _ { h } ; \pi ) } & { } \\ { \displaystyle } & { = 1 \left\{ \displaystyle \sum _ { v = 1 } ^ { h } k _ { v } ( s _ { v } ; \pi ) < 1 \right\} e _ { h } ( \xi _ { h } ; \pi ) \leq e _ { h } ( \xi _ { h } ; \pi ) . } \end{array}
$$

Hence

$$
\mathrm { K L } ( \mathcal { R } _ { \pi } [ P ^ { \pi } ] \| \mathcal { R } _ { \pi } [ P ^ { \pi ^ { \star } } ] ) \leq \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \right] = \overline { { L } } ( \pi ) .
$$

The construction also permits infinite local KL under a source ${ \widetilde { \pi } } ,$ because $\begin{array} { r } { k _ { h } ( s _ { h } ; \pi ) = \infty , \sum _ { v = 1 } ^ { h } e _ { v } ( \xi _ { v } ; \pi ) = } \end{array}$ $1 , \mathbb { P } _ { \widetilde { \pi } } ( \sigma > h \mid \tau ) = 0$ □

Lemma B.4 (Shared mass on an event). Let $R , S$ be probability distributions on a countable space $\mathcal { X }$ with $\mathrm { K L } ( R \| S ) < \infty ,$ , and let $D \subseteq { \mathcal { X } } .$ . Write $z = R ( D )$ and $v = \textstyle \sum _ { x \in D }$ min $\{ R ( x ) , S ( x ) \}$ . For $z > 0$

$$
\mathrm { K L } ( R \| S ) \geq z \log ( z / v ) - z .\tag{26}
$$

Consequently, $\mathrm { K L } ( R \| S ) \leq z$ implies $v \gtrsim z$

Proof. For $z = 0 , v = 0$ . For $z > 0 , 0 < v \leq z$ . Define

$$
g ( x ) = \left\{ { \begin{array} { l l } { \log { \frac { R ( x ) } { \operatorname* { m i n } \{ R ( x ) , S ( x ) \} } } , } & { x \in D , R ( x ) > 0 , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.
$$

Then

$$
\begin{array} { r l } & { \mathbb { E } _ { S } \left[ e ^ { g } \right] = 1 + \displaystyle \sum _ { \stackrel { R \in D } { R ( x ) \leq S ( x ) } } \left( R ( x ) - S ( x ) \right) } \\ & { \qquad R ( x ) > S ( x ) } \\ & { \qquad = 1 + \displaystyle \sum _ { x \in D } \left( R ( x ) - \operatorname* { m i n } \left\{ R ( x ) , S ( x ) \right\} \right) = 1 + z - v , } \\ & { \mathbb { E } _ { R } \left[ g \right] = - z \displaystyle \sum _ { \stackrel { R \in D } { R ( x ) > 0 } } \frac { R ( x ) } { z } \log \frac { \operatorname* { m i n } \left\{ R ( x ) , S ( x ) \right\} } { R ( x ) } } \\ & { \qquad \geq - z \log \left( \displaystyle \sum _ { \stackrel { R \in D } { R ( x ) \leq 0 } } \frac { R ( x ) } { z } \frac { \operatorname* { m i n } \left\{ R ( x ) , S ( x ) \right\} } { R ( x ) } \right) = z \log ( z / v ) . } \end{array}
$$

The inequality follows from Jensen’s inequality. Since $\mathbb { E } _ { S } \left[ e ^ { g } \right] = 1 + z - v \in [ 1 , 2 ]$ , Lemma D.3 gives

$$
\begin{array} { r l } & { \mathrm { K L } ( R \| S ) \geq \mathbb { E } _ { R } \left[ g \right] - \log \mathbb { E } _ { S } \left[ e ^ { g } \right] } \\ & { \qquad \geq z \log ( z / v ) - \log ( 1 + z - v ) } \\ & { \qquad \geq z \log ( z / v ) - ( z - v ) \geq z \log ( z / v ) - z . } \end{array}
$$

Consequently, $\mathrm { K L } ( R \| S ) \leq z \Longrightarrow z \log ( z / v ) - z \leq z \Longrightarrow v \geq e ^ { - 2 } z .$

Proof of Lemma B.1. Fix $\pi \in \Pi$ , and let $D = \{ ( h , \xi _ { h } ) : h \in [ H ] \}$ be the set of prefix outcomes in Equation (22). For every ${ \widetilde { \pi } } \in \Pi \cup \{ \pi ^ { \star } \}$ , Equation (22) gives

$$
\mathcal { R } _ { \pi } [ P ^ { \widetilde { \pi } } ] ( D ) = \sum _ { h = 1 } ^ { H } \sum _ { \xi _ { h } } e _ { h } ( \xi _ { h } ; \pi ) P ^ { \widetilde { \pi } } ( \xi _ { h } ) = \mathbb { E } _ { P ^ { \widetilde { \pi } } } \left[ \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \right] .
$$

Lemma B.3 therefore implies

$$
\begin{array} { r } { \mathrm { K L } ( \mathcal { R } _ { \pi } [ P ^ { \pi } ] \| \mathcal { R } _ { \pi } [ P ^ { \pi ^ { \star } } ] ) \leq \overline { { L } } ( \pi ) = \mathcal { R } _ { \pi } [ P ^ { \pi } ] ( D ) . } \end{array}
$$

Applying Lemma B.2 to $\mathcal { R } _ { \pi } [ P ^ { \pi } ]$ and $\mathcal { R } _ { \pi } [ P ^ { \pi ^ { \star } } ]$ and $f ( y ) = { \bf 1 } \{ y \in D \}$ as the indicator of D yields

$$
\mathbb { E } _ { P ^ { \pi ^ { \star } } } \left[ \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \right] = { \mathcal { R } _ { \pi } } [ P ^ { \pi ^ { \star } } ] ( D ) \asymp { \mathcal { R } _ { \pi } } [ P ^ { \pi } ] ( D ) = \overline { { L } } ( \pi ) .\tag{27}
$$

Now we control the trajectory Hellinger distance.

$$
\begin{array} { r l r } { T _ { \mathrm { H } } ^ { \prime } ( F ^ { \prime } , P ^ { \prime \prime } ) } & { = \displaystyle \sum _ { j = 1 } ^ { N } ( 1 - \frac { \sum _ { k = 1 } ^ { N } \alpha _ { k } ( j \leq \nu + \nu ) } { \sum _ { k = 1 } ^ { N } \alpha _ { k } ( j \leq \nu + \nu ) } ) ( \sqrt { \nu ^ { \top \prime } \nu ^ { \top } \nu ^ { \top } } - \sqrt { \nu ^ { \top \prime } ( \nu ) } ) ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { = \displaystyle \sum _ { j = 1 } ^ { N } ( \frac { \sum _ { k = 1 } ^ { N } \alpha _ { k } ( j \leq \nu + \nu ) } { \sum _ { k = 1 } ^ { N } \alpha _ { k } ( j \leq \nu + \nu ) } ) ( \sqrt { \nu ^ { \top \prime \prime } \nu ^ { \top } } ) \cdot \ \ \sqrt { \nu ^ { \top \prime } \nu ( \nu ) } ) ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

The last line uses Lemma B.3 and the explicit bound $\begin{array} { r } { \mathbb { E } _ { P ^ { \pi ^ { \star } } } \Big [ \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \Big ] \le \frac { 2 } { 1 - e ^ { - 1 } } \overline { { L } } ( \pi ) \le 4 \overline { { L } } ( \pi ) } \end{array}$ from the proof of Lemma B.2.

For the same event $D = \{ ( h , \xi _ { h } ) : h \in [ H ] \}$ , Equation (22) gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { z \in D } \operatorname* { m i n } \{ \mathcal { R } _ { \pi } [ P ^ { \pi } ] ( z ) , \mathcal { R } _ { \pi } [ P ^ { \pi ^ { \prime } } ] ( z ) \} = \displaystyle \sum _ { h = 1 } ^ { H } \sum _ { \xi _ { h } } e _ { h } ( \xi _ { h } ; \pi ) \operatorname* { m i n } \{ P ^ { \pi } ( \xi _ { h } ) , P ^ { \pi ^ { \prime } } ( \xi _ { h } ) \} } \\ & { \quad \quad \quad \quad = \mathrm { T r u n } _ { \pi ^ { \prime } } \big [ \overline { { L } } ( \pi ) \big ] \in [ 0 , \overline { { L } } ( \pi ) ] . } \end{array}
$$

The interval bound uses min $\{ P ^ { \pi } ( \xi _ { h } ) , P ^ { \pi ^ { \prime } } ( \xi _ { h } ) \} \le P ^ { \pi } ( \xi _ { h } )$ . By Equation (23), Lemma B.4 applied to D gives

$$
\sum _ { z \in D } \operatorname* { m i n } \{ \mathcal { R } _ { \pi } [ P ^ { \pi } ] ( z ) , \mathcal { R } _ { \pi } [ P ^ { \pi ^ { \star } } ] ( z ) \} \geq e ^ { - 2 } \overline { { L } } ( \pi ) .
$$

Equation (68) and Equation (70) in Lemma D.4 give

$$
\begin{array} { r l } & { e ^ { - 1 } \sqrt { \overline { { L } } ( \pi ) } \leq \sqrt { \displaystyle \sum _ { z \in D } \operatorname* { m i n } \{ \mathcal R _ { \pi } [ P ^ { \pi } ] ( z ) , \mathcal R _ { \pi } [ P ^ { \pi \star } ] ( z ) \} } } \\ & { \qquad \leq \sqrt { \displaystyle \sum _ { z \in D } \operatorname* { m i n } \{ \mathcal R _ { \pi } [ P ^ { \pi } ] ( z ) , \mathcal R _ { \pi } [ P ^ { \pi } ] ( z ) \} } + D _ { \mathrm { H } } ( \mathcal R _ { \pi } [ P ^ { \pi ^ { \star } } ] , \mathcal R _ { \pi } [ P ^ { \pi ^ { \prime } } ] ) } \\ & { \qquad \leq \sqrt { \mathrm { T r u n } _ { \pi ^ { \prime } } \left[ \overline { { L } } ( \pi ) \right] } + D _ { \mathrm { H } } ( P ^ { \pi ^ { \star } } , P ^ { \pi ^ { \prime } } ) . } \end{array}
$$

Squaring and using $( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$ and $e ^ { - 2 } > 1 / 8$ gives

$$
\begin{array} { r l r } {  { e ^ { - 2 } \overline { { L } } ( \pi ) \leq 2 \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ] + 2 D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi ^ { \star } } , P ^ { \pi ^ { \prime } } ) , } } \\ & { } & { \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ] \geq \frac { 1 } { 1 6 } \overline { { L } } ( \pi ) - D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi ^ { \star } } , P ^ { \pi ^ { \prime } } ) , } \\ & { } & { \overline { { L } } ( \pi ) \leq e ^ { 2 } ( \sqrt { \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ] } + \sqrt { 6 \overline { { L } } ( \pi ^ { \prime } ) } ) ^ { 2 } } \\ & { } & { \leq 7 e ^ { 2 } ( \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ] + \overline { { L } } ( \pi ^ { \prime } ) ) } \\ & { } & { \leq 6 4 ( \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } ( \pi ) ] + \overline { { L } } ( \pi ^ { \prime } ) ) . } \end{array}
$$

The last three inequalities use Equation (28), Cauchy–Schwarz, and $7 e ^ { 2 } < 6 4 $ , respectively.

## B.2.2 Exponential-weights analysis

Throughout this subsection, $\mathbb { E } _ { t }$ conditions on the history before episode t and averages over both $\pi _ { t } \sim w _ { t }$ and its trajectory. For uniform initialization, $\ell _ { t } ( \pi ) \in [ - 1 , 1 ]$ , and $0 < \eta \le \log 2$ , standard exponential-weights analysis [Lattimore and Szepesvári, 2020, Eqs. (27.2)–(27.3)] gives

$$
- \sum _ { t = 1 } ^ { T } \ell _ { t } ( \pi ^ { \dagger } ) \le \frac { \log | \Pi | } { \eta } - \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) + \eta \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) ^ { 2 }\tag{29}
$$

for every fixed $\pi ^ { \dagger } \in \Pi$ and every realized loss sequence. The following lemma relates the loss terms to ${ \overline { { L } } } .$

Lemma B.5 (Conditional loss moments). For Algorithm 1, $| \ell _ { t } ( \pi ) | \leq 1$ and

$$
\mathbb { E } _ { t } \left[ \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) \right] = 0 .\tag{30}
$$

With $a = 1 / 1 6$ and a numerical $C > 0$

$$
\begin{array} { l } { { \displaystyle - \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \right] \geq a \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \overline { { L } } ( \pi ^ { \prime } ) - 2 L ( \pi ) , } } \\ { { \displaystyle \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \right] \geq a \overline { { L } } ( \pi ) - C \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \overline { { L } } ( \pi ^ { \prime } ) . } } \end{array}\tag{31}
$$

The second moments satisfy

$$
\mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) ^ { 2 } \right] \leq \overline { { L } } ( \pi ) + \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \overline { { L } } ( \pi ^ { \prime } ) ,
$$

$$
\mathbb { E } _ { t } \left[ \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) ^ { 2 } \right] \leq 2 \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \overline { { L } } ( \pi ^ { \prime } ) .\tag{32}
$$

Proof. The definition of $\ell _ { t }$ gives the conditional mean and cancellation

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \mid \pi _ { t } = \pi ^ { \prime } \right] = \operatorname { T r u n } _ { \pi ^ { \prime } } \left[ \overline { { L } } ( \pi ) \right] - \operatorname { T r u n } _ { \pi } \left[ \overline { { L } } ( \pi ^ { \prime } ) \right] , } \\ & { \mathbb { E } _ { t } \left[ \displaystyle \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) \right] = \displaystyle \sum _ { \pi , \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ) w _ { t } ( \pi ^ { \prime } ) \Big ( \operatorname { T r u n } _ { \pi ^ { \prime } } \big [ \overline { { L } } ( \pi ) \big ] - \operatorname { T r u n } _ { \pi } \big [ \overline { { L } } ( \pi ^ { \prime } ) \big ] \Big ) = 0 . } \end{array}\tag{33}
$$

The truncation bound in Lemma B.1, together with ${ \overline { { L } } } ( \pi ) \leq L ( \pi )$ and $D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) \leq L ( \pi )$ , yields

$$
\begin{array} { r l r } & { } & { - \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \mid \pi _ { t } = \pi ^ { \prime } \right] \ge a \overline { { L } } ( \pi ^ { \prime } ) - D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) - \overline { { L } } ( \pi ) \ge a \overline { { L } } ( \pi ^ { \prime } ) - 2 L ( \pi ) , } \\ & { } & { \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \mid \pi _ { t } = \pi ^ { \prime } \right] \ge a \overline { { L } } ( \pi ) - D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi ^ { \prime } } , P ^ { \pi ^ { \star } } ) - \overline { { L } } ( \pi ^ { \prime } ) \ge a \overline { { L } } ( \pi ) - C \overline { { L } } ( \pi ^ { \prime } ) . } \end{array}
$$

Here the last inequality uses $D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi ^ { \prime } } , P ^ { \pi ^ { \star } } ) \lesssim \overline { { L } } ( \pi ^ { \prime } )$ from the same lemma. Averaging over $\pi ^ { \prime } \sim w _ { t }$ gives the two risk bounds.

Each weighted sum in Equation (6) lies in [0, 1], so $| \ell _ { t } ( \pi ) | \leq 1$ . The inequality $( x - y ) ^ { 2 } \leq x + y \mathrm { { f o r } } x , y \in [ 0 , 1 ]$ gives the remaining bounds in succession,

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) ^ { 2 } \mid \pi _ { t } = \pi ^ { \prime } \right] \leq \mathrm { T r u n } _ { \pi ^ { \prime } } \left[ \overline { { L } } ( \pi ) \right] + \mathrm { T r u n } _ { \pi } \left[ \overline { { L } } ( \pi ^ { \prime } ) \right] \leq \overline { { L } } ( \pi ) + \overline { { L } } ( \pi ^ { \prime } ) , } \\ & { \qquad \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) ^ { 2 } \right] \leq \overline { { L } } ( \pi ) + \displaystyle \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \overline { { L } } ( \pi ^ { \prime } ) , } \\ & { \mathbb { E } _ { t } \left[ \displaystyle \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) ^ { 2 } \right] \leq \displaystyle \sum _ { \pi , \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ) w _ { t } ( \pi ^ { \prime } ) \big ( \overline { { L } } ( \pi ) + \overline { { L } } ( \pi ^ { \prime } ) \big ) = 2 \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \overline { { L } } ( \pi ) . } \end{array}
$$

ProofofTheorem 3.2. The case $| \Pi | = 1$ is immediate. For $| \Pi | \geq 2 ,$ , fix $\pi ^ { \dagger } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } L ( \pi )$ and write $\begin{array} { r } { S _ { T } = \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \overline { { L } } ( \pi ) } \end{array}$ . The cumulative weighted clipped risk $S _ { T }$ is random because $w _ { t }$ depends on the training history.

Cumulative training risk. The first inequality in Equation (31), summed for $\pi ^ { \dagger }$ , bounds $a S _ { T }$ by $2 T L ^ { \star } -$ $\begin{array} { r l } { ~ } & { { } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ^ { \dagger } ) \right] } \end{array}$ . Separating the observed losses from their conditional expectations and applying Equation (29) gives

$$
\begin{array} { r l } & { a S _ { T } \leq 2 T L ^ { \star } - \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \hat { \varepsilon } _ { t } ( \mathfrak { z } ^ { t } ) \right] } \\ & { \quad = 2 T L ^ { \star } - \displaystyle \sum _ { t = 1 } ^ { T } \hat { \varepsilon } _ { t } ( \pi ^ { \sf t } ) + \displaystyle \sum _ { t = 1 } ^ { T } \left( \hat { \varepsilon } _ { t } ( \mathfrak { z } ^ { t } ) - \mathbb { E } _ { t } \left[ \hat { \varepsilon } _ { t } ( \mathfrak { z } ^ { t } ) \right] \right) } \\ & { \quad \leq 2 T L ^ { \star } + \displaystyle \frac { \log | \mathfrak { z } | | } { \eta } + \underbrace { \left( - \sum _ { t = 1 } ^ { T } \sum _ { \mathfrak { c } \in \Pi } u _ { t } ( \pi ) \hat { \varepsilon } _ { t } ( \pi ) \right) } _ { \mathrm { w i g h \ : \ : \mathrm { e x p t } = 1 \ : \log \ : \mathrm { e x p } } } + \underbrace { \eta \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { \mathfrak { c } \in \Pi } u _ { t } ( \pi ) \hat { \varepsilon } _ { t } ( \pi ) ^ { 2 } } _ { \mathrm { g u l s i a r ~ t e r m } } } \\ & { \quad \quad \quad + \displaystyle \sum _ { t = 1 } ^ { T } \left( \hat { \varepsilon } _ { t } ( \mathfrak { z } ^ { t } ) - \mathbb { E } _ { t } \left[ \hat { \varepsilon } _ { t } ( \mathfrak { z } ^ { t } ) \right] \right) . } \end{array}\tag{34}
$$

For any fixed $0 < \lambda \leq 1 / 2$ , the following bounds hold simultaneously with probability at least $1 - \delta / 2 .$ , by Lemma D.6 and a union bound.

For the weighted loss,

$$
\begin{array} { r l } & { \displaystyle \left. \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) \right. } \\ & { \displaystyle \quad \le \lambda \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \left( \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) \right) ^ { 2 } \right] + \frac { \log ( 8 / \delta ) } { \lambda } } \\ & { \displaystyle \le \lambda \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) ^ { 2 } \right] + \frac { \log ( 8 / \delta ) } { \lambda } } \\ & { \displaystyle \le 2 \lambda S _ { T } + \frac { \log ( 8 / \delta ) } { \lambda } . } \end{array}\tag{Equations (30) and (71)}
$$

(Jensen’s inequality)

(Equation (32))

(35)

For the quadratic term, Lemma B.5 gives $\begin{array} { r } { 0 \leq \sum _ { \pi \in \Pi } w _ { t } ( \pi ) \ell _ { t } ( \pi ) ^ { 2 } \leq 1 } \end{array}$ , so

$$
\begin{array} { r l } & { \eta \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { \ell = 1 } ^ { m } \mu _ { i } ( \overline { { \boldsymbol \alpha } } ) \ell _ { i } ( \boldsymbol \pi ) ^ { 2 } } \\ & { \quad \le \eta \displaystyle \sum _ { \ell = 1 } ^ { T } \mathbb { E } _ { t } \left[ \sum _ { \boldsymbol \sigma \in \Pi } w _ { i } ( \boldsymbol \pi ) \ell _ { i } ( \boldsymbol \pi ) ^ { 2 } \right] + \eta \displaystyle \frac { \log [ 8 / \delta ] } { \lambda } } \\ & { \qquad + \eta \displaystyle \lambda \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \left( \sum _ { \boldsymbol \sigma \in \Pi } w _ { i } ( \boldsymbol \pi ) \ell _ { i } ( \boldsymbol \pi ) ^ { 2 } \right) ^ { 2 } \right] } \\ & { \le \eta ( 1 + \lambda ) \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \sum _ { \operatorname* { s u p } } w _ { i } ( \boldsymbol \pi ) \ell _ { i } ( \boldsymbol \pi ) ^ { 2 } \right] + \eta \displaystyle \frac { \log ( 8 / \delta ) } { \lambda } } \\ & { \quad \le 2 \eta ( 1 + \lambda ) S _ { T } + \eta \displaystyle \frac { \log ( \delta / \delta ) } { \lambda } . } \end{array}\tag{Equation (71)}
$$

(x<sup>2</sup> ≤ x for x ∈ [0, 1])

(Equation (32))

(36)

For the comparator sampling error,

$$
\begin{array} { r l } {  { \sum _ { t = 1 } ^ { T } \Bigl ( \ell _ { t } ( \boldsymbol { \pi } ^ { \dagger } ) - \mathbb { E } _ { t } [ \ell _ { t } ( \boldsymbol { \pi } ^ { \dagger } ) ] \Bigr ) } \quad } & { } \\ & { \leq \lambda \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } [ \ell _ { t } ( \boldsymbol { \pi } ^ { \dagger } ) ^ { 2 } ] + \frac { \log ( 8 / \delta ) } { \lambda } } \\ & { \leq \lambda \bigl ( T \bar { L } ( \boldsymbol { \pi } ^ { \dagger } ) + S _ { T } \bigr ) + \frac { \log ( 8 / \delta ) } { \lambda } } \\ & { \leq \lambda ( T L ^ { \star } + S _ { T } ) + \frac { \log ( 8 / \delta ) } { \lambda } . } \end{array}\tag{Equation (71)}
$$

(Equation (32))

$$
( \overline { { L } } ( \pi ^ { \dagger } ) \leq L ^ { \star } )\tag{37}
$$

Substituting Equation (35), Equation (36), and Equation (37) for the weighted loss, quadratic term, and comparator sampling error in Equation (34), respectively, gives

$$
\begin{array} { c } { { a S _ { T } \leq 2 T L ^ { \star } + \displaystyle \frac { \log | \Pi | } \eta + 2 \lambda S _ { T } + \displaystyle \frac { \log ( 8 / \delta ) } \lambda } } \\ { { \displaystyle + 2 \eta ( 1 + \lambda ) S _ { T } + \eta \displaystyle \frac { \log ( 8 / \delta ) } \lambda + \lambda ( T L ^ { \star } + S _ { T } ) + \displaystyle \frac { \log ( 8 / \delta ) } \lambda } } \\ { { \displaystyle \left[ a - 3 \lambda - 2 \eta ( 1 + \lambda ) \right] S _ { T } \leq ( 2 + \lambda ) T L ^ { \star } + \displaystyle \frac { \log | \Pi | } \eta + ( 2 + \eta ) \displaystyle \frac { \log ( 8 / \delta ) } \lambda . } } \end{array}\tag{38}
$$

For $\eta = 1 / 2 5 6 , a - 2 \eta = 7 / 1 2 8 > 0 .$ . A sufficiently small fixed $\lambda > 0$ therefore makes the coefficient on $S _ { T }$ positive, yielding $S _ { T } \lesssim T L ^ { \star } + \log ( | \Pi | / \delta )$ .

Selection by final weight. The maximum-weight output satisfies $w _ { T + 1 } ( \widehat { \pi } ) \geq 1 / | \Pi | = w _ { 1 } ( \widehat { \pi } )$ . For every candidate with $w _ { T + 1 } ( \pi ) \geq w _ { 1 } ( \pi )$ , the weight update and Jensen’s inequality bound its cumulative observed loss by the weighted loss,

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ( \pi ) = - \frac { 1 } { \eta } \log \frac { w _ { T + 1 } ( \pi ) } { w _ { 1 } ( \pi ) } - \frac { 1 } { \eta } \sum _ { t = 1 } ^ { T } \log \left( \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) e ^ { - \eta \ell _ { t } ( \pi ^ { \prime } ) } \right) } \\ & { \qquad \le - \frac { 1 } { \eta } \log \frac { w _ { T + 1 } ( \pi ) } { w _ { 1 } ( \pi ) } + \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \ell _ { t } ( \pi ^ { \prime } ) } \\ & { \qquad \le \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \ell _ { t } ( \pi ^ { \prime } ) . } \end{array}\tag{39}
$$

The last inequality uses $w _ { T + 1 } ( \pi ) \geq w _ { 1 } ( \pi )$ . To convert this observed-loss bound to a risk bound, we apply Equation (72) with failure probability $\delta / ( 2 | \Pi | )$ to each candidate and take a union bound over Π. The uniform bound over Π and the preceding concentration bounds hold jointly with probability at least $1 - \delta$ . On this event, every candidate with $w _ { T + 1 } ( \pi ) \geq 1 / | \Pi |$ satisfies

$$
a T \overline { { L } } ( \pi ) \leq \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \right] + C S _ { T }\tag{Equation (31)}
$$

$$
\leq \sum _ { t = 1 } ^ { T } \ell _ { t } ( \pi ) + \lambda \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) ^ { 2 } \right] + C S _ { T } + \frac { \log ( 2 | \Pi | / \delta ) } { \lambda }\tag{Equation (72)}
$$

$$
\le \sum _ { t = 1 } ^ { T } \ell _ { t } ( \pi ) + \lambda T \overline { { L } } ( \pi ) + ( C + \lambda ) S _ { T } + \frac { \log ( 2 | \Pi | / \delta ) } { \lambda }\tag{Equation (32)}
$$

$$
\leq \sum _ { t = 1 } ^ { T } \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \ell _ { t } ( \pi ^ { \prime } ) + \lambda T \overline { { L } } ( \pi ) + ( C + \lambda ) S _ { T } + \frac { \log ( 2 | \Pi | / \delta ) } { \lambda }\tag{Equation (39)}
$$

$$
\leq \lambda T \overline { { { L } } } ( \pi ) + ( C + 3 \lambda ) S _ { T } + \frac { \log ( 2 | \Pi | / \delta ) + \log ( 8 / \delta ) } { \lambda } .\tag{Equation (35)}
$$

Rearranging with $\lambda < a$ and using Equation (38) gives, simultaneously for every $\pi$ with $w _ { T + 1 } ( \pi ) \geq 1 / | \Pi |$

$$
\begin{array} { c } { { \overline { { L } } ( \pi ) \leq \displaystyle \frac { ( C + 3 \lambda ) S _ { T } + \lambda ^ { - 1 } \big ( \log ( 2 | \Pi | / \delta ) + \log ( 8 / \delta ) \big ) } { ( a - \lambda ) T } } } \\ { { \displaystyle \lesssim L ^ { \star } + \frac { \log ( | \Pi | / \delta ) } { T } . } } \end{array}\tag{40}
$$

In particular, $\widehat { \pi }$ satisfies this bound. Under Assumption 2.1, Lemma D.2 then yields

$$
L ( \widehat \pi ) \leq ( 1 + B ) \overline { { L } } ( \widehat \pi ) \lesssim B \left( L ^ { \star } + \frac { \log ( | \Pi | / \delta ) } { T } \right) .
$$

## B.3 Online Hellinger learning from full feedback

Algorithm 2 replaces the local-KL score in Algorithm 1 by a prefix-weighted local Hellinger score before clipping.

Algorithm 2 Online Hellinger learning with full teacher feedback   
Require: Finite candidate class Π, episode budget T.   
1: Set $w _ { 1 } ( \pi ) = 1 / | \Pi |$ for every $\pi \in \Pi$ and $\eta = 1 / 1 2 8 .$   
2: for $t = 1 , \dots , T$ do   
3: Sample a policy $\pi _ { t } \sim w _ { t }$ and generate a complete trajectory $\tau _ { t } = ( s _ { t , 1 } , a _ { t , 1 } , \ldots , s _ { t , H } , a _ { t , H } ) \sim P ^ { \pi _ { t } }$ by executing   
$\pi _ { t }$ for the entire episode.   
4: Observe $\pi _ { h } ^ { \star } ( \cdot \mid s _ { t , h } )$ for every $h = 1 , \ldots , H .$ Write $\xi _ { t , h } = ( s _ { t , 1 } , a _ { t , 1 } , \ldots , s _ { t , h } )$   
5: for each $\pi \in$ Π do   
6: For every $h = 1 , \ldots , H$ with $\begin{array} { r } { P ^ { \pi } ( \xi _ { t , h } ) > 0 , } \end{array}$ compute   
$g _ { h } ( \xi _ { t , h } ; \pi ) = \sqrt { \prod _ { v = 1 } ^ { h - 1 } \frac { \pi _ { v } ^ { \star } ( a _ { t , v } \mid s _ { t , v } ) } { \pi _ { v } ( a _ { t , v } \mid s _ { t , v } ) } D _ { \mathrm { H } } ^ { 2 } ( \pi _ { h } ( \cdot \mid s _ { t , h } ) , \pi _ { h } ^ { \star } ( \cdot \mid s _ { t , h } ) ) }$   
Set $g _ { h } ( \xi _ { t , h } ; \pi ) = 0$ when $\begin{array} { r } { P ^ { \pi } ( \xi _ { t , h } ) = 0 . } \end{array}$ . The empty product equals one.   
7: For every $h = 1 , \ldots , H$ , compute the clipped increment   
$e _ { h } ^ { \mathrm { H } } ( \xi _ { t , h } ; \pi ) = \operatorname* { m i n } \Biggl \{ 1 , \sum _ { v = 1 } ^ { h } g _ { v } ( \xi _ { t , v } ; \pi ) \Biggr \} - \operatorname* { m i n } \Biggl \{ 1 , \sum _ { v = 1 } ^ { h - 1 } g _ { v } ( \xi _ { t , v } ; \pi ) \Biggr \}$   
8: end for   
9: For every $\pi \in \Pi ,$ compute   
$\ell _ { t } ( \pi ) = \sum _ { h = 1 } ^ { H } \operatorname* { m i n } \left\{ 1 , \prod _ { v = 1 } ^ { h - 1 } \frac { \pi _ { v } ( a _ { t , v } \mid s _ { t , v } ) } { \pi _ { t , v } ( a _ { t , v } \mid s _ { t , v } ) } \right\} \left( e _ { h } ^ { \mathrm { H } } ( \xi _ { t , h } ; \pi ) - e _ { h } ^ { \mathrm { H } } ( \xi _ { t , h } ; \pi _ { t } ) \right)$   
where $\pi _ { t , v }$ is the action distribution of the sampled policy $\pi _ { t }$ at step v.   
10: Update all candidate weights by $w _ { t + 1 } ( \pi ) = \frac { \begin{array} { l } { { w _ { t } ( \pi ) \exp ( - \eta \ell _ { t } ( \pi ) ) } } \\ { { \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \exp ( - \eta \ell _ { t } ( \pi ^ { \prime } ) ) } } \end{array} } { \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \exp ( - \eta \ell _ { t } ( \pi ^ { \prime } ) ) } . $   
11: end for   
12: Return $\widehat { \pi } \in \arg \operatorname* { m a x } _ { \pi \in \Pi } w _ { T + 1 } ( \pi )$   
Write ${ \cal L } _ { \mathrm { H } } ( \pi ) = { \cal D } _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } )$ and $\begin{array} { r } { \overline { { L } } _ { \mathrm { H } } ( \pi ) = \mathbb { E } _ { P ^ { \pi } } \left[ \sum _ { h = 1 } ^ { H } e _ { h } ^ { \mathrm { H } } ( \xi _ { h } ; \pi ) \right] } \end{array}$   
Theorem B.6 (Online Hellinger guarantee). For a finite class Π, Algorithm 2 returns $\widehat { \pi } \in \Pi$ such that, with   
probability at least $1 - \delta ,$   
$L _ { \mathrm { H } } ( \widehat { \pi } ) \lesssim \operatorname* { m i n } _ { \pi \in \Pi } L _ { \mathrm { H } } ( \pi ) + \frac { \log ( | \Pi | / \delta ) } { T } .$ (41)   
Clipping and prefix truncation. For a fixed candidate $\pi ,$ , write $P = P ^ { \pi } , Q = P ^ { \pi ^ { \star } }$ , and $\begin{array} { r } { G _ { h } = \sum _ { v = 1 } ^ { h } g _ { v } ( \xi _ { v } ; \pi ) } \end{array}$   
with $G _ { 0 } = 0$ . For $c > 0 ,$ , define   
$e _ { h } ^ { ( c ) } ( \xi _ { h } ) = \operatorname* { m i n } \{ c , G _ { h } \} - \operatorname* { m i n } \{ c , G _ { h - 1 } \} , \qquad \overline { { L } } _ { c } = \mathbb { E } _ { P } \left[ \operatorname* { m i n } \{ c , G _ { H } \} \right]$   
The proof uses $c = 1$ to compare $\overline { { L } } _ { \mathrm { H } } ( \pi )$ with $L _ { \mathrm { H } } ( \pi )$ , and $c = 1 / 4$ to bound prefix truncation. For any trajectory

distribution U, the auxiliary distribution $\mathcal { R } _ { c } [ U ]$ assigns probabilities

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { c } [ U ] ( h , \xi _ { h } ) = \frac { e _ { h } ^ { ( c ) } ( \xi _ { h } ) } { c } U ( \xi _ { h } ) , \qquad h \in [ H ] , } \\ { \displaystyle \mathcal { R } _ { c } [ U ] ( \infty , \tau ) = \left( 1 - \frac { G _ { H } } { c } \right) _ { + } U ( \tau ) . } \end{array}\tag{42}
$$

Equivalently, conditional on $\tau \sim U$ , the outcome is $( h , \xi _ { h } )$ with probability $e _ { h } ^ { ( c ) } ( \xi _ { h } ) / c$ and $( \infty , \tau )$ with probability $( 1 - G _ { H } / c ) _ { + }$ . These probabilities sum to one because $\scriptstyle \sum _ { h = 1 } ^ { H } e _ { h } ^ { ( c ) } / c =$ min $\{ 1 , G _ { H } / c \}$ . The fixed candidate π determines $G _ { h }$ and $e _ { h } ^ { ( c ) }$ for all input distributions $U$ , including $U = P ^ { \pi ^ { \prime } }$

Lemma B.7 (Hellinger identities for auxiliary distributions). For every candidate π and $c > 0 ,$

$$
\begin{array} { r l r } { \displaystyle } & { } & { \displaystyle { L _ { \mathrm { H } } ( \boldsymbol { \pi } ) = \mathbb { E } _ { P } \left[ \sum _ { h = 1 } ^ { H } g _ { h } ( \xi _ { h } ; \boldsymbol { \pi } ) \right] , } } \\ { \displaystyle } & { } & { \displaystyle { D _ { \mathrm { H } } ^ { 2 } ( \mathcal { R } _ { c } [ P ] , \mathcal { R } _ { c } [ Q ] ) = \mathbb { E } _ { P } \left[ \sum _ { h = 1 } ^ { H } \left( 1 - \frac { G _ { h } } { c } \right) _ { + } g _ { h } ( \xi _ { h } ; \boldsymbol { \pi } ) \right] \le \overline { { L } } _ { c } . } } \end{array}\tag{43}
$$

Proof. Define the prefix affinities $\begin{array} { r } { A _ { h } = \sum _ { \xi _ { h } } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } } \end{array}$ for $h \in [ H ]$ and $\begin{array} { r } { A _ { H + 1 } = \sum _ { \tau } \sqrt { P ( \tau ) Q ( \tau ) } } \end{array}$ . The common initial law gives $A _ { 1 } = 1$ . For $h \in [ \ddot { H } ]$

$$
\begin{array} { r l r } { 2 ( A _ { h } - A _ { h + 1 } ) = \displaystyle \sum _ { \xi _ { h } } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } \left( 2 - 2 \sum _ { a } \sqrt { \pi _ { h } ( a \mid s _ { h } ) \pi _ { h } ^ { \star } ( a \mid s _ { h } ) } \right) } & { \quad \mathrm { ( C o m m o n ~ t r a n s i t i o n ~ l a w s ) } } \\ { = \displaystyle \sum _ { \xi _ { h } } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } D _ { \mathbb H } ^ { 2 } ( \pi _ { h } ( \cdot \mid s _ { h } ) , \pi _ { h } ^ { \star } ( \cdot \mid s _ { h } ) ) } & { \quad \mathrm { ( D e f i n i t i o n ~ o f ~ } D _ { \mathbb H } ^ { 2 } ) } \\ & { \quad \quad \quad = \mathbb E _ { P } \left[ g _ { h } ( \xi _ { h } ; \pi ) \right] . } & { \quad \mathrm { ( A l g o r i t h m 2 ) } } \end{array}
$$

Prefixes with $P ( \xi _ { h } ) = 0$ contribute zero to both sides. Consequently,

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { \mathrm { H } } ( \pi ) = 2 ( 1 - A _ { H + 1 } ) } } \\ { { \displaystyle ~ = 2 \sum _ { h = 1 } ^ { H } ( A _ { h } - A _ { h + 1 } ) } } \\ { { \displaystyle ~ = \mathbb { E } _ { P } \left[ \sum _ { h = 1 } ^ { H } g _ { h } ( \xi _ { h } ; \pi ) \right] . } } \end{array}
$$

(Definition of $D _ { \mathrm { H } } ^ { 2 } )$

(Equation (44))

For the auxiliary distributions, the corresponding affinities before each stopping decision are

$$
A _ { h } ^ { ( c ) } = \sum _ { \xi _ { h } } \left( 1 - \frac { G _ { h - 1 } } { c } \right) _ { + } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } , \qquad A _ { H + 1 } ^ { ( c ) } = \sum _ { \tau } \left( 1 - \frac { G _ { H } } { c } \right) _ { + } \sqrt { P ( \tau ) Q ( \tau ) } .
$$

$$
A _ { 1 } ^ { ( c ) } = 1
$$

$$
\begin{array} { r l } & { = 1 . \operatorname { S i n c e } \left( 1 - G _ { h - 1 } / c \right) + \operatorname { - } \operatorname { e } _ { h } ^ { \left( c \right) } / c + \left( 1 - G _ { h } / c \right) , } \\ & { A _ { h } ^ { \left( c \right) } - A _ { h + 1 } ^ { \left( c \right) } = \displaystyle \sum _ { \xi _ { h } } \frac { c _ { h } ^ { \left( c \right) } \left( \xi _ { h } \right) } { c } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } } \\ & { \qquad + \displaystyle \sum _ { \xi _ { h } } \left( 1 - \frac { G _ { h } } { c } \right) _ { + } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } \left( 1 - \sum _ { a } \sqrt { \pi _ { h } \left( a \left| \xi _ { h } \right| \right) x _ { h } ^ { \ast } \left( a \left| \ x _ { h } \right| \right) } \right) } \\ & { \qquad = \displaystyle \sum _ { \xi _ { h } } \frac { c _ { h } ^ { \left( c \right) } \left( \xi _ { h } \right) } { c } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathrm { ( D e f i n i t i o n ~ o f ~ } g _ { h } ) } \\ & { \qquad + \displaystyle \frac { 1 } { 2 } \mathbb { E } _ { P } \left[ \left( 1 - \frac { G _ { h } } { c } \right) _ { + } \mathcal { B _ { h } } \{ G _ { h } : \pi \} \right] . } \end{array}
$$

sition laws)

(45)

$$
\begin{array} { r l } & { D _ { \mathrm { H } } ^ { \mathrm { 2 } } ( { \widetilde R } _ { \mathrm { c } } [ P ] , { \widetilde R } _ { \mathrm { c } } [ Q ] ) } \\ & { \quad = 2 \left( 1 - \varDelta _ { H + 1 } ^ { ( \phi ) } - \displaystyle \sum _ { b = 1 } ^ { H } \sum _ { \delta _ { h } } \frac { e _ { b } ^ { ( \xi ) } ( \xi _ { h } ) } { c } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } \right) } \\ & { \quad = 2 \displaystyle \sum _ { h = 1 } ^ { H } \left( { \cal A } _ { h } ^ { ( \phi ) } - { \cal A } _ { h + 1 } ^ { ( \phi ) } - \sum _ { b _ { h } } \frac { e _ { h } ^ { ( \xi ) } ( \xi _ { h } ) } { c } \sqrt { P ( \xi _ { h } ) Q ( \xi _ { h } ) } \right) } \\ & { \quad = { \mathbb E } _ { P } \left[ \displaystyle \sum _ { h = 1 } ^ { H } \left( 1 - \frac { G _ { h } } { c } \right) _ { 1 } g h ( \xi _ { h } ; \pi ) \right] } \\ & { \quad \le { \mathbb E } _ { P } \left[ \displaystyle \sum _ { b = 1 } ^ { H } e _ { h } ^ { ( \xi ) } ( \xi _ { h } ) \right] = { \overline { { { \cal L } } } _ { c } } . } \end{array}\tag{Equation (42)}
$$

$$
( A _ { 1 } ^ { ( c ) } = 1 )\tag{Equation (45)}
$$

$$
( \left( 1 - G _ { h } / c \right) _ { + } g _ { h } \leq e _ { h } ^ { ( c ) } )
$$

The pointwise inequality holds because $G _ { h } < c$ implies $e _ { h } ^ { ( c ) } = g _ { h }$ , whereas $G _ { h } \ge c$ makes its left-hand side zero. □

The definition Equation (4), with $e _ { h } ^ { \mathrm { H } }$ in place of $e _ { h }$ , defines $\mathrm { T r u n } _ { \pi ^ { \prime } } \left[ \overline { { L } } _ { \mathrm { H } } ( \pi ) \right]$

Lemma B.8 (Hellinger clipping and truncation). For every $\pi , \pi ^ { \prime } \in \Pi ,$

$$
\begin{array} { r } { \overline { { L } } _ { \mathrm { H } } ( \pi ) \asymp L _ { \mathrm { H } } ( \pi ) , \qquad \overline { { L } } _ { \mathrm { H } } ( \pi ) \lesssim \mathrm { T r u n } _ { \pi ^ { \prime } } \big [ \overline { { L } } _ { \mathrm { H } } ( \pi ) \big ] + \overline { { L } } _ { \mathrm { H } } ( \pi ^ { \prime } ) . } \end{array}\tag{46}
$$

More precisely,

$$
\begin{array} { r } { \mathrm { T r u n } _ { \pi ^ { \prime } } \left[ \overline { { L } } _ { \mathrm { H } } ( \pi ) \right] \geq \overline { { L } } _ { \mathrm { H } } ( \pi ) / 3 2 - L _ { \mathrm { H } } ( \pi ^ { \prime } ) / 4 . } \end{array}\tag{47}
$$

$$
A l s o \overline { { L } } _ { \mathrm { H } } ( \pi ) \leq L _ { \mathrm { H } } ( \pi ) a n d 0 \leq \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } _ { \mathrm { H } } ( \pi ) ] \leq \overline { { L } } _ { \mathrm { H } } ( \pi ) .
$$

Proof. Clipping and Equation (43) give $\overline { { L } } _ { \mathrm { H } } ( \pi ) \leq L _ { \mathrm { H } } ( \pi )$ . The bounds $0 \leq \mathrm { T r u n } _ { \pi ^ { \prime } } [ \overline { { L } } _ { \mathrm { H } } ( \pi ) ] \leq \overline { { L } } _ { \mathrm { H } } ( \pi )$ follow from Equation (4). For the reverse inequality, let $D = \{ ( h , \xi _ { h } ) : h \in [ H ] \}$ be the set of prefix outcomes. Then $\mathcal { R } _ { 1 } [ P ] ( D ) = \overline { { L } } _ { \mathrm { H } } ( \pi )$ , and

$$
\begin{array} { r l } & { \sqrt { \mathbb { E } _ { Q } \left[ \operatorname* { m i n } \{ 1 , G _ { H } \} \right] } = \sqrt { \mathcal { R } _ { 1 } [ Q ] ( D ) } } \\ & { \qquad \leq \sqrt { \mathcal { R } _ { 1 } [ P ] ( D ) } + D _ { \mathrm { H } } ( \mathcal { R } _ { 1 } [ P ] , \mathcal { R } _ { 1 } [ Q ] ) } \\ & { \qquad \leq 2 \sqrt { \overline { { L } } _ { \mathrm { H } } ( \pi ) } . } \end{array}\tag{Equation (42)}
$$

(Equation (69), F = 1<sub>D</sub>)

(Equation (43))

(48)

Splitting each trajectory’s contribution with $1 = ( 1 - G _ { H } ) _ { + } + \mathrm { { m i n } } \{ 1 , G _ { H } \}$ gives

$$
\begin{array} { r l r } { L _ { \mathrm { H } } ( \pi ) = \displaystyle \sum _ { \tau } \left( \sqrt { \mathcal { R } _ { 1 } [ P ] ( \infty , \tau ) } - \sqrt { \mathcal { R } _ { 1 } [ Q ] ( \infty , \tau ) } \right) ^ { 2 } } & \\ { \displaystyle + \sum _ { \tau } \operatorname* { m i n } \{ 1 , G _ { H } \} ( \sqrt { P ( \tau ) } - \sqrt { Q ( \tau ) } ) ^ { 2 } } & { \qquad \mathrm { ( E q u a t i o n ~ ( 4 2 ) ) } } \\ { \displaystyle \le D _ { \mathrm { H } } ^ { 2 } ( \mathcal { R } _ { 1 } [ P ] , \mathcal { R } _ { 1 } [ Q ] ) + \overline { { L } } _ { \mathrm { H } } ( \pi ) + \mathbb { E } _ { Q } \left[ \operatorname* { m i n } \{ 1 , G _ { H } \} \right] } & { \qquad ( ( \sqrt { p } - \sqrt { q } ) ^ { 2 } \le p + q ) } \\ { \displaystyle \le 6 \overline { { L } } _ { \mathrm { H } } ( \pi ) . } & { \qquad } & { \mathrm { ( E q u a t i o n s ~ ( 4 3 ) ~ a n d ~ ( 4 8 ) ) } } \end{array}
$$

For truncation, set $c = 1 / 4 , R = \mathcal { R } _ { c } [ P ] , S = \mathcal { R } _ { c } [ Q ] , S ^ { \prime } = \mathcal { R } _ { c } [ P ^ { \pi ^ { \prime } } ]$ , and $z = R ( D ) = \overline { { L } } _ { c } / c$ . The overlap inequality first compares $R$ with S, and then $S$ with $S ^ { \prime } \mathrm { { : } }$

$$
\begin{array} { r l r } {  { \frac { 1 } { 2 } \sqrt { z } \le \sqrt { \displaystyle \sum _ { y \in D } \operatorname* { m i n } \{ R ( y ) , S ( y ) \} } } } \\ & { } & { \le \sqrt { \displaystyle \sum _ { y \in D } \operatorname* { m i n } \{ R ( y ) , S ^ { \prime } ( y ) \} } + D _ { \mathrm { H } } ( S , S ^ { \prime } ) } \\ & { } & { \le \sqrt { \displaystyle \sum _ { y \in D } \operatorname* { m i n } \{ R ( y ) , S ^ { \prime } ( y ) \} } + \sqrt { L _ { \mathrm { H } } ( \pi ^ { \prime } ) } . } \end{array}
$$

(Equations (43) and (70), c = 1/4)

(Equation (70))

(Equation (68))

(50)

Since $e _ { h } ^ { ( c ) } \leq e _ { h } ^ { \mathrm { H } }$ and min $\{ c , G _ { H } \} \ge c \operatorname* { m i n } \{ 1 , G _ { H } \}$

$$
\begin{array} { r l r } {  { \mathrm { T r u n } _ { \pi ^ { \prime } } [ \bar { L } _ { \mathrm { H } } ( \pi ) ] } } \\ & { \geq \displaystyle \sum _ { h = 1 } ^ { H } \sum _ { \xi _ { h } } \operatorname* { m i n } \{ P ( \xi _ { h } ) , P ^ { \pi ^ { \prime } } ( \xi _ { h } ) \} e _ { h } ^ { ( c ) } ( \xi _ { h } ) } \\ & { = c \sum _ { y \in D } \operatorname* { m i n } \{ R ( y ) , S ^ { \prime } ( y ) \} } \\ & { \geq c ( \frac { \underline { { z } } } { 8 } - L _ { \mathrm { H } } ( \pi ^ { \prime } ) ) } & { \quad \scriptstyle ( \mathrm { E q u a t i o n ~ } ( 5 0 ) , ( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 } ) } \\ & { = \displaystyle \frac { \overline { { L } } _ { c } } { 8 } - \frac { L _ { \mathrm { H } } ( \pi ^ { \prime } ) } { 4 } \geq \frac { \overline { { L } } _ { \mathrm { H } } ( \pi ) } { 3 2 } - \frac { L _ { \mathrm { H } } ( \pi ^ { \prime } ) } { 4 } . } & { \quad \scriptstyle ( c z = \overline { { L } } _ { c } , \overline { { L } } _ { c } \geq c \overline { { L } } _ { \mathrm { H } } ( \pi ) ) } \end{array}
$$

This proves Equation (47). Rearranging Equation (47) and using $L _ { \mathrm { H } } ( \pi ^ { \prime } ) \le 6 \overline { { L } } _ { \mathrm { H } } ( \pi ^ { \prime } )$ from Equation (49) gives

$$
\begin{array} { r } { \overline { { L } } _ { \mathrm { H } } ( \pi ) \leq 3 2 \mathrm { T r u n } _ { \pi ^ { \prime } } \left[ \overline { { L } } _ { \mathrm { H } } ( \pi ) \right] + 4 8 \overline { { L } } _ { \mathrm { H } } ( \pi ^ { \prime } ) , } \end{array}
$$

which proves the second relation in Equation (46).

ProofofTheorem B.6. The conditional-mean, cancellation, and second-moment calculations in Lemma B.5 use only the form of the loss and nonnegative increments satisfying $\textstyle \sum _ { h = 1 } ^ { H } e _ { h } \leq 1$ . Replacing $e _ { h }$ by $e _ { h } ^ { \mathrm { H } }$ therefore gives $| \ell _ { t } ( \pi ) | \leq 1$ and Equation (33), Equation (30), and Equation (32) with $\overline { { L } } _ { \mathrm { H } }$ in place of $\overline { { L } }$

Applying Equation (47) to each direction of the conditional mean in Equation (33), and using $\overline { { L } } _ { \mathrm { H } } \leq L _ { \mathrm { H } } \leq 6 \overline { { L } } _ { \mathrm { H } }$ ， gives

$$
- \mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \right] \ge \frac { 1 } { 3 2 } \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \overline { { L } } _ { \mathrm { H } } ( \pi ^ { \prime } ) - \frac { 5 } { 4 } L _ { \mathrm { H } } ( \pi ) ,
$$

$$
\mathbb { E } _ { t } \left[ \ell _ { t } ( \pi ) \right] \geq \frac { 1 } { 3 2 } \overline { { L } } _ { \mathrm { H } } ( \pi ) - \frac { 5 } { 2 } \sum _ { \pi ^ { \prime } \in \Pi } w _ { t } ( \pi ^ { \prime } ) \overline { { L } } _ { \mathrm { H } } ( \pi ^ { \prime } ) .
$$

Thus Equation (31) holds with $a = 1 / 3 2 , C = 5 / 2$ , and $L _ { \mathrm { H } } , \overline { { L } } _ { \mathrm { H } }$ in place of $L , { \overline { { L } } } ,$ , since $5 / 4 \le 2$ . The stepsize $\eta = 1 / 1 2 8$ satisfies $a - 2 \eta = 1 / 6 4 > 0$ . The analysis of concentration and selection by final weight in Appendix B.2.2, from Equation (29) through Equation (40), therefore gives, with probability at least $1 - \delta$

$$
L _ { \mathrm { H } } ( \widehat { \pi } ) \leq 6 \overline { { L } } _ { \mathrm { H } } ( \widehat { \pi } ) \lesssim \operatorname* { m i n } _ { \pi \in \Pi } L _ { \mathrm { H } } ( \pi ) + \frac { \log ( | \Pi | / \delta ) } { T } ,
$$

where the first inequality is Equation (49).

Corollary B.9 (Reverse KL via online Hellinger learning). Under Assumption 2.2, Algorithm 2 satisfies, with probability at least $1 - \delta _ { : }$

$$
L ( \widehat \pi ) \lesssim B _ { \infty } \left( \operatorname* { m i n } _ { \pi \in \Pi } D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) + \frac { \log ( | \Pi | / \delta ) } { T } \right) .\tag{51}
$$

Proof. Theorem B.6 and Lemma D.5 give Equation (51). Moreover, $\begin{array} { r } { \operatorname* { m i n } _ { \pi \in \Pi } L _ { \mathrm { H } } ( \pi ) \le L ^ { \star } } \end{array}$ by Equation (67), recovering the reverse-KL guarantee with approximation term $B _ { \infty } L ^ { \star }$ □

## B.4 Estimation and approximation lower bounds

Lower bounds for Theorem 3.3 and Section 4. Estimation. For every integer $d \geq 1 , T \geq 1 , B \geq 1$ , and $\delta \in ( 0 , 1 / 8 ]$ , every online or offlinefull-feedback learner using T episodes has a realizable instance with $H = 1$ $| { \mathcal { A } } | = 2 ,$ , and $| \Pi | = 2 ^ { d }$ satisfying Assumption 2.1 on which, with probability greater than δ,

$$
L ( \widehat { \pi } ) = \Omega \bigg ( B \operatorname* { m i n } \bigg \{ 1 , \frac { \log ( | \Pi | / \delta ) } { T } \bigg \} \bigg ) .\tag{52}
$$

Under Assumption 2.2, the same lower bound holds with $B _ { \infty } > 1$ replacing B, including for offline action feedback.

Approximation at optimal estimation. Fix κ $> 0$ . There is an integer $k = k ( \kappa ) \geq 8$ such that,for $B \geq 2$ log k and $T \geq k ,$ , any learner returning $\widehat { \pi } \in \Pi$ whose guarantee at confidence $3 / 4$ has $C _ { \mathrm { e s t } } \le \kappa B$ on all $H = 2$ instances satisfying Assumption 2.1 with $| { \mathcal { A } } | = | \Pi | = k$ must have $C _ { \mathrm { a p p } } = \Omega ( B )$ . This holdsfor both online and offline full feedback. Under Assumption 2.2, the same conclusion holds with $B _ { \infty }$ replacing B, includingfor offline actionfeedback.

## B.4.1 Estimation lower bound

Proof of the estimation lower bound. MDP and policy class. The one-step MDP has

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } = \{ a _ { 1 } , a _ { 2 } \} , \qquad \boldsymbol { \mathcal { S } } _ { 1 } = \{ s _ { 1 , 0 } , s _ { 1 , 1 } , \ldots , s _ { 1 , d } \} , \qquad \boldsymbol { \Pi } = \{ \pi _ { f } : f \in \{ 0 , 1 \} ^ { d } \} . } \end{array}
$$

For $\rho \in ( 0 , 1 ]$ , the initial law is $\mu ( { s } _ { 1 , x } ) = \rho / d$ for $x \in [ d ]$ and $\mu ( s _ { 1 , 0 } ) = 1 - \rho$ . The episode ends after the first action. For $\beta \geq 1$ , define distributions on $( a _ { 1 } , a _ { 2 } )$ by

$$
R _ { 0 } ( \beta ) = \left( \frac { e ^ { \beta } } { 1 + e ^ { \beta } } , \frac { 1 } { 1 + e ^ { \beta } } \right) , \qquad R _ { 1 } ( \beta ) = \left( \frac { 1 } { 1 + e ^ { \beta } } , \frac { e ^ { \beta } } { 1 + e ^ { \beta } } \right) .
$$

For $f \in \{ 0 , 1 \} ^ { d }$ and $x \in [ d ]$

$$
\pi _ { f , 1 } ( \cdot \mid s _ { 1 , x } ) = R _ { f _ { x } } ( \beta ) , \qquad \pi _ { f , 1 } ( \cdot \mid s _ { 1 , 0 } ) = ( 1 / 2 , 1 / 2 ) .
$$

The teacher is $\pi _ { \boldsymbol { \theta } } ^ { \star } = \pi _ { \boldsymbol { \theta } }$ for $\theta \in \{ 0 , 1 \} ^ { d } , \mathtt { s o } L ^ { \star } = 0$

Boundedness. For $x \in [ d ]$ and each feasible ${ \boldsymbol { \tau } } = ( s _ { 1 } , a _ { 1 } )$

$$
\begin{array} { r l r } & { k _ { 1 } ( s _ { 1 , x } ; \pi _ { f } ) = \mathrm { K L } ( R _ { f _ { \pi } } ( \beta ) \| R _ { \theta _ { \pi } } ( \beta ) ) = \beta \operatorname { t a n h } ( \beta / 2 ) { \bf 1 } \{ f _ { x } \neq \theta _ { x } \} \leq \beta , \quad \quad } & { \mathrm { ( B e r m o u l i t ~ K L ~ f o r m u l a ) } } \\ & { \quad \frac { P ^ { \pi _ { f } } ( \tau ) } { P ^ { \pi _ { \theta } ^ { * } } ( \tau ) } = \frac { \pi _ { f , 1 } ( a _ { 1 } \mid s _ { 1 } ) } { \pi _ { \theta , 1 } ^ { * } ( a _ { 1 } \mid s _ { 1 } ) } \in \{ 1 , e ^ { - \beta } , e ^ { \beta } \} . \quad \quad } & { \mathrm { ( C o m m o n ~ i n i t i a l ~ l a w ) } } \end{array}
$$

Thus $\beta = B$ verifies Assumption 2.1 by Lemma D.1, and $\beta = B _ { \infty }$ verifies Assumption 2.2, for either initial law used below.

Risk at an unvisited state. Write $\begin{array} { r } { L _ { \theta } ( \pi ) = \mathrm { K L } ( P ^ { \pi } \| P ^ { \pi _ { \theta } ^ { \star } } ) } \end{array}$ . Let θ be uniform on $\{ 0 , 1 \} ^ { d }$ , and let $M _ { x }$ count training episodes starting at $s _ { 1 , x } .$ For any binary distribution v and $U _ { 2 } = ( 1 / 2 , 1 / 2 )$

$$
{ \frac { 1 } { 2 } } \sum _ { z = 0 } ^ { 1 } \operatorname { K L } ( v \| R _ { z } ( \beta ) ) = \operatorname { K L } ( v \| U _ { 2 } ) + b ( \beta ) \geq b ( \beta ) , \qquad b ( \beta ) : = \log \cosh ( \beta / 2 ) \asymp \beta .\tag{53}
$$

Let $v _ { x } = \widehat { \pi } _ { 1 } ( \cdot \mid s _ { 1 , x } )$ . Conditional on $M _ { x } = 0$ , the training observations and $v _ { x }$ are independent of $\theta _ { x }$ for either trajectory source. Equation (53) gives

$$
\mathbb { P } \{ \mathrm { K L } ( v _ { x } \| R _ { \theta _ { x } } ( \beta ) ) \geq b ( \beta ) \ | \ M _ { x } = 0 \} \geq \frac { 1 } { 2 } .
$$

Class-size term. Take $\rho = \operatorname* { m i n } \{ 1 , d / ( 8 T ) \}$ and set

$$
W = \frac { 1 } { d } \sum _ { x = 1 } ^ { d } \mathbf { 1 } \{ M _ { x } = 0 , \mathrm { K L } ( v _ { x } \| R _ { \theta _ { x } } ( \beta ) ) \geq b ( \beta ) \} .
$$

Then

$$
\mathbb { E } \left[ W \right] \geq \frac { 1 } { 2 d } \sum _ { x = 1 } ^ { d } \mathbb { P } \{ M _ { x } = 0 \} = \frac { 1 } { 2 } \left( 1 - \frac { \rho } { d } \right) ^ { T }
$$

(Independent initial-state draws)

$$
\geq { \frac { 1 } { 2 } } \left( 1 - { \frac { T \rho } { d } } \right) \geq { \frac { 7 } { 1 6 } } ,\tag{Bernoulli’s inequality}
$$

$$
\begin{array} { r } { L _ { \theta } ( \widehat \pi ) = \displaystyle \sum _ { s \in { \mathcal S } _ { 1 } } \mu ( s ) \operatorname { K L } ( \widehat \pi _ { 1 } ( \cdot \mid s ) \| \pi _ { \theta , 1 } ^ { \star } ( \cdot \mid s ) ) } \\ { \displaystyle \geq \frac \rho d \displaystyle \sum _ { x = 1 } ^ { d } \operatorname { K L } ( v _ { x } \| R _ { \theta _ { x } } ( \beta ) ) \geq \rho b ( \beta ) W , } \end{array}\tag{H = 1}
$$

(Definition of W)

$$
\mathbb { P } \Bigg \{ L _ { \theta } ( \widehat { \pi } ) > \frac { \rho b ( \beta ) } { 8 } \Bigg \} \geq \mathbb { P } \{ W > 1 / 8 \} \geq \frac { \mathbb { E } \left[ W \right] - 1 / 8 } { 1 - 1 / 8 } \geq \frac { 5 } { 1 4 } > \frac { 1 } { 4 } .\tag{0 ≤ W ≤ 1}
$$

Here $\rho b ( \beta ) \asymp \beta \operatorname* { m i n } \{ 1 , \log | \Pi | / T \}$ . Averaging over the finite teacher prior gives a fixed teacher with this failure probability.

Confidence term. Keep the same policy class and take $\rho = \operatorname* { m i n } \{ 1 / 2 , \log ( 1 / ( 2 \delta ) ) / ( 4 T ) \}$ . Replace the initial law by

$$
\mu ( s _ { 1 , 1 } ) = \rho , \qquad \left\{ \begin{array} { l l } { \mu ( s _ { 1 , 0 } ) = 1 - \rho , } & { d = 1 , } \\ { \mu ( s _ { 1 , 0 } ) = 0 , \quad \mu ( s _ { 1 , x } ) = ( 1 - \rho ) / ( d - 1 ) } & { ( 2 \leq x \leq d ) , d \geq 2 . } \end{array} \right.
$$

Only $\theta _ { 1 }$ is unknown: it is uniform on $\{ 0 , 1 \}$ , and $\theta _ { x } = 0$ for $x \ge 2$ . Thus observations away from $s _ { 1 , 1 }$ reveal no information about $\theta _ { 1 }$ , and

$$
\begin{array} { r l r } { \mathbb { P } \{ L _ { \theta } ( \widehat \pi ) \geq \rho b ( \beta ) \} \geq \frac { 1 } { 2 } \mathbb { P } \{ M _ { 1 } = 0 \} = \frac { 1 } { 2 } ( 1 - \rho ) ^ { T } } & { \qquad } & { \mathrm { ( E q u a t i o n ~ ( 5 3 ) ) } } \\ { \geq \frac { 1 } { 2 } e ^ { - 2 T \rho } \geq \frac { 1 } { 2 } \sqrt { 2 \delta } > \delta . } & { \qquad } & { \mathrm { ( l o g ( 1 - \rho ) \geq - 2 \rho , \delta \leq 1 / 8 ) } } \end{array}
$$

Here $\rho b ( \beta ) \asymp \beta$ min $\{ 1 , \log ( 1 / \delta ) / T \}$ , and prior averaging again fixes a teacher. Choosing the larger of the two risk thresholds yields

$$
\beta \operatorname* { m a x } \biggr \{ \operatorname* { m i n } \biggr \{ 1 , \frac { \log | \Pi | } { T } \biggr \} , \operatorname* { m i n } \biggr \{ 1 , \frac { \log ( 1 / \delta ) } { T } \biggr \} \biggr \} \asymp \beta \operatorname* { m i n } \biggr \{ 1 , \frac { \log ( | \Pi | / \delta ) } { T } \biggr \} .
$$

Both arguments allow full feedback from either trajectory source. Restricting feedback gives the offline actionfeedback bound. □

## B.4.2 Approximation lower bound

Proof of the approximation lower bound. We use a construction parameter $B \geq 2$ log k. MDP and policy class. Let $k \geq 8$ be chosen below in terms of $\kappa ,$ and set $\rho = { ( k - 1 ) } / { ( 8 T ) }$ . The two-step MDP has

$$
\begin{array} { c c } { { \mathcal { A } = \{ a _ { 1 } , \dotsc , a _ { k } \} , } } & { { S _ { 1 } = \{ s _ { 1 , 0 } , s _ { 1 , 1 } \} , } } \\ { { } } & { { } } \\ { { \mu ( s _ { 1 , 1 } ) = \rho , } } & { { \mu ( s _ { 1 , 0 } ) = 1 - \rho , } } \\ { { } } & { { } } \\ { { \ P _ { 1 } ( s _ { 2 , j } \mid s _ { 1 , 1 } , a _ { j } ) = 1 , } } & { { \ P _ { 1 } ( s _ { 2 , 0 } \mid s _ { 1 , 0 } , a _ { j } ) = 1 } } & { { ( j \in [ k ] ) . } } \end{array}
$$

The episode ends after the second action. The class is $\Pi = \{ \pi _ { i } : i \in [ k ] \}$ , and the teacher is indexed by $\theta \in [ k ]$ Their first-step distributions at $s _ { 1 , 1 }$ are

$$
\pi _ { i , 1 } ( a _ { j } \mid s _ { 1 , 1 } ) = \left\{ { 3 / 4 } , \begin{array} { l l } { { j = i , } } & { { } } \\ { { 1 / [ 4 ( k - 1 ) ] , } } & { { j \neq i , } } \end{array} \right. \pi _ { \theta , 1 } ^ { \star } ( a _ { j } \mid s _ { 1 , 1 } ) = 1 / k .
$$

For $z \in \{ 0 , 1 \}$ , define distributions on A by

$$
R _ { z } ( a _ { 1 } ) = \displaystyle \frac { \exp ( ( B / 2 ) { \bf 1 } \{ z = 0 \} ) } { 1 + e ^ { B / 2 } } , \qquad R _ { z } ( a _ { j } ) = \displaystyle \frac { \exp ( ( B / 2 ) { \bf 1 } \{ z = 1 \} ) } { ( 1 + e ^ { B / 2 } ) ( k - 1 ) } \quad ( 2 \leq j \leq k ) .
$$

At the second layer,

$$
\pi _ { i , 2 } ( \cdot \mid s _ { 2 , j } ) = R _ { \mathbf 1 \{ j = i \} } , \qquad \pi _ { \theta , 2 } ^ { \star } ( \cdot \mid s _ { 2 , j } ) = R _ { \mathbf 1 \{ j = \theta \} } \quad ( j \in [ k ] ) .
$$

Every policy is uniform at $s _ { 1 , 0 }$ and $s _ { 2 , 0 }$ . The learner may know the initial law and transitions and explore adaptively, but cannot reset within an episode or query unvisited states. Its output belongs to Π.

Boundedness and trajectory risks. Write $\begin{array} { r } { L _ { \theta } ( \pi ) = \mathrm { K L } ( P ^ { \pi } \| P ^ { \pi _ { \theta } ^ { \star } } ) } \end{array}$ and $d _ { k } = \mathrm { K L } ( \mathrm { B e r } ( 3 / 4 ) \| \mathrm { B e r } ( 1 / k ) ) \leq \log k .$ For $B \geq 2 \log k$

$$
\begin{array} { l } { { \displaystyle \mathrm { K L } ( R _ { 0 } | | R _ { 1 } ) = \mathrm { K L } ( R _ { 1 } | | R _ { 0 } ) = \frac { B } { 2 } \operatorname { t a n h } ( B / 4 ) \in [ B / 4 , B / 2 ] , \qquad ( \mathrm { B e r n o u l i ~ K L ~ f o r m u l a , } k \ge 8 ) } } \\ { { \displaystyle k _ { 1 } ( s _ { 1 , 1 } ; \pi _ { i } ) = d _ { k } , \qquad k _ { 2 } ( s _ { 2 , j } ; \pi _ { i } ) = \mathrm { K L } ( R _ { 0 } | | R _ { 1 } ) { \bf 1 } \{ { \bf 1 } \{ j = i \} \ne { \bf 1 } \{ j = \theta \} \} } . } \end{array}
$$

(The specified action distributions)

All action probabilities are positive, and every feasible trajectory satisfies

$$
\begin{array} { r l r } {  { \sum _ { h = 1 } ^ { 2 } k _ { h } ( s _ { h } ; \pi _ { i } ) \le \log { k } + B / 2 \le B } , }  \\ & { } & { \log { \frac { p \pi _ { i } } { P ^ { \pi _ { \delta } } ( \tau ) } } = \displaystyle \sum _ { h = 1 } ^ { 2 } \log { \frac { \pi _ { i , h } ( a _ { h } \mid s _ { h } ) } { \pi _ { \delta , h } ^ { * } ( a _ { h } \mid s _ { h } ) } } \le \log { k } + B / 2 \le B . \mathrm { ( F i r s t . \ s t e p \ r a t i o \le } k \mathrm { , \ s e c o n d \ l { \ l { \times } } t e p \ r a t i o \le } e ^ { B / 2 } ) } \end{array}
$$

Lemma D.1 verifies Assumption 2.1 from the first bound. The second verifies Assumption 2.2 with $B _ { \infty } = B$ . If $i \neq \theta ,$ , only branches $j = i$ and $j = \theta$ have nonzero second-step KL. The KL chain rule gives

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { \theta } ( \pi _ { i } ) = \rho d _ { k } + \rho \left( \frac { 3 } { 4 } + \frac { 1 } { 4 ( k - 1 ) } \right) \mathrm { K L } ( R _ { 0 } \| R _ { 1 } ) { \bf 1 } \{ i \neq \theta \} , } } \\ { { \displaystyle ~ { \cal L } ^ { \star } = { \cal L } _ { \theta } ( \pi _ { \theta } ) = \rho d _ { k } . } } \end{array}\tag{54}
$$

Identification error. Let θ be uniform on $[ k ]$ . The teacher’s first-step distribution is independent of θ. A trajectory taking $a _ { j }$ at $s _ { 1 , 1 }$ reaches $s _ { 2 , j }$ , where full feedback reveals exactly ${ \bf 1 } \{ j = \theta \}$ . One equality test also simulates an offline teacher trajectory: its first action is uniform, and its second follows $R _ { 1 \{ j = \theta \} }$ . Let $\begin{array} { r } { M = \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ s _ { t , 1 } = s _ { 1 , 1 } \} } \end{array}$ and write $\widehat { \pi } = \pi _ { \widehat { i } } .$ . Conditional on the initial-state sequence, M is fixed and θ remains uniform on [k]. Therefore,

$$
\begin{array} { r l r } & { } & { \mathbb { P } \{ \hat { i } \neq \theta \} \ge \mathbb { E } \left[ 1 - \frac { M + 1 } { k } \right] } \\ & { } & { = \frac { 7 ( k - 1 ) } { 8 k } \ge \frac { 3 } { 4 } , } \\ & { } & { L _ { \theta } ( \widehat { \pi } ) - L ^ { \star } \ge \frac { 3 \rho B } { 1 6 } \quad \mathrm { o n } \ \{ \hat { i } \neq \theta \} . } \end{array}
$$

(Lemma D.8, conditionally then averaged)

$$
( \mathbb { E } \left[ M \right] = T \rho = ( k - 1 ) / 8 , k \geq 8 )
$$

$$
( \mathrm { E q u a t i o n } \ ( 5 4 ) , \mathrm { K L } ( R _ { 0 } \| R _ { 1 } ) \geq B / 4 )
$$

Prior averaging fixes a teacher for which this event has probability at least $3 / 4$

Approximation coefficient. The event above intersects the success event of any guarantee at confidence $3 / 4 ,$ so

$$
\begin{array} { l l } { \displaystyle \frac { 3 \rho B } { 1 6 } \leq ( C _ { \mathrm { a p p } } - 1 ) L ^ { \star } + C _ { \mathrm { e s t } } \frac { \log ( 4 k ) } { T } } & { \qquad \mathrm { ( G u a r a n t e ~ a t ~ c o n f i d e n c e ~ 3 / 4 ) } } \\ { \displaystyle \frac { 3 B } { 1 6 } \leq ( C _ { \mathrm { a p p } } - 1 ) d _ { k } + \frac { 8 C _ { \mathrm { e s t } } \log ( 4 k ) } { k - 1 } . } & { \qquad ( L ^ { \star } = \rho d _ { k } , \rho = ( k - 1 ) / ( 8 T ) ) ( 5 T ) } \end{array}
$$

Choose the fixed integer $k = k ( \kappa ) \geq 8$ so that $8 \kappa \log ( 4 k ) / ( k - 1 ) \leq 3 / 3 2$ . Then

$$
\begin{array} { c } { \displaystyle \frac { 8 C _ { \mathrm { e s t } } \log ( 4 k ) } { k - 1 } \leq \frac { 8 \kappa B \log ( 4 k ) } { k - 1 } \leq \frac { 3 B } { 3 2 } , } \\ { ( C _ { \mathrm { a p p } } - 1 ) d _ { k } \geq \frac { 3 B } { 3 2 } , } \\ { C _ { \mathrm { a p p } } \geq 1 + \frac { 3 B } { 3 2 \log k } = \Omega ( B ) . } \end{array}
$$

$$
( C _ { \mathrm { e s t } } \le \kappa B )\tag{Equation (55)}
$$

$$
( d _ { k } \leq \log k , \operatorname { f i x e d } k = k ( \kappa ) )
$$

This proves the relative second-moment result. Replacing B by $B _ { \infty }$ gives the ratio-bound result. Restricting the offline observations to action feedback preserves the lower bound. □

## B.5 The additional cost of online action feedback

Theorem 3.5 (detailed statement). For $T \geq 1 , B _ { \infty } > 1$ , and $| \Pi | , | { \cal A } | \ge 2 ,$ , let $m =$ min $\{ | \mathcal { A } | , | \Pi | , e ^ { B _ { \infty } } \}$ Every online action-feedback learner with arbitrary causal exploration and output in Π has a realizable $H = 1$ instance with afull-support candidate class satisfying Assumption 2.2 such that, with probability at least $1 / 4$

$$
L ( \widehat { \pi } ) = \Omega \bigg ( B _ { \infty } \operatorname* { m i n } \bigg \{ 1 , \frac { m \log | \Pi | } { T \log ( e m ) } \bigg \} \bigg ) .\tag{56}
$$

ProofofTheorem 3.5. MDP and policy class. Write $A = | \mathcal { A } | , N = | \Pi |$ , and set

$$
m = \operatorname* { m i n } \{ A , N , e ^ { B _ { \infty } } \} , \qquad k = \lfloor m \rfloor , \qquad d = \lfloor \log N / \log k \rfloor , \qquad \rho = \operatorname* { m i n } \biggl \{ 1 , \frac { ( k - 1 ) d } { 8 T } \biggr \} .
$$

The one-step MDP has

$$
\begin{array} { r } { \mathcal { A } = \{ a _ { 1 } , \dotsc , a _ { A } \} , \qquad S _ { 1 } = \{ s _ { 1 , 0 } , s _ { 1 , 1 } , \dotsc , s _ { 1 , d } , s _ { 1 , d + 1 } \} , } \end{array}
$$

$$
\mu ( s _ { 1 , x } ) = \rho / d \quad ( x \in [ d ] ) , \qquad \mu ( s _ { 1 , 0 } ) = 1 - \rho , \qquad \mu ( s _ { 1 , d + 1 } ) = 0 .
$$

The episode ends after the first action. The state $s _ { 1 , d + 1 }$ will distinguish additional candidates if $k ^ { d } < N$ . For $f \in [ k ] ^ { d }$ and $x \in [ d ]$ , define

$$
\pi _ { f , 1 } ( a _ { j } \mid s _ { 1 , x } ) = \left\{ \begin{array} { l l } { \displaystyle \frac { A + k } { 2 A } \frac { \exp ( B _ { \infty } { \bf 1 } \{ j = f _ { x } \} ) } { e ^ { B _ { \infty } } + k - 1 } , } & { 1 \leq j \leq k , } \\ { \displaystyle \frac { 1 } { 2 A } , } & { k < j \leq A . } \end{array} \right.\tag{57}
$$

The last $A - k$ actions have total probability $( A - k ) / ( 2 A )$ ; the first k have the remaining probability $( A +$ $k ) / ( 2 A ) \geq 1 / 2$ . At $s _ { 1 , 0 }$ and $s _ { 1 , d + 1 }$ , every $\pi _ { f }$ is uniform. The class contains all $k ^ { d }$ policies $\pi _ { f } .$ . The remaining $N - k ^ { d }$ candidates agree with $\pi _ { ( 1 , \ldots , 1 ) }$ except at $s _ { 1 , d + 1 }$ , where they use distinct, nonuniform, strictly positive action distributions. The teacher is $\pi _ { \theta } ^ { \star } = \pi _ { \theta } , \theta \in [ k ] ^ { d }$ , so $L ^ { \star } = 0$

Boundedness and trajectory risks. Write $\begin{array} { r } { L _ { \theta } ( \pi ) = \mathrm { K L } ( P ^ { \pi } \| P ^ { \pi _ { \theta } ^ { \star } } ) } \end{array}$ ). All action probabilities are positive. On a trajectory starting at $s _ { 1 , x }$ with action $a _ { j }$ ,

$$
\frac { P ^ { \pi _ { f } } ( \tau ) } { P ^ { \pi _ { \theta } ^ { * } } ( \tau ) } = \left\{ \begin{array} { l l } { \exp ( B _ { \infty } [ \mathbf { 1 } \{ j = f _ { x } \} - \mathbf { 1 } \{ j = \theta _ { x } \} ] ) , } & { x \in [ d ] , \ j \leq k , } \\ { 1 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. \leq e ^ { B _ { \infty } } .
$$

This also holds for the additional candidates, whose trajectory law is $P ^ { \pi _ { ( 1 , \ldots , 1 ) } }$ . For $f _ { x } \neq \theta _ { x }$

$$
\begin{array} { r l r } & { } & { k _ { 1 } ( s _ { 1 , x } ; \pi _ { f } ) = B _ { \infty } \big ( \pi _ { f , 1 } ( a _ { f _ { x } } \mid s _ { 1 , x } ) - \pi _ { f , 1 } ( a _ { \theta _ { x } } \mid s _ { 1 , x } ) \big ) \qquad \mathrm { ( O n l y ~ } a _ { f _ { x } } \mathrm { ~ a n d ~ } a _ { \theta _ { x } } \mathrm { ~ h a v e ~ n o n u n i t ~ r a t i o s ) } } \\ & { } & { \qquad = \displaystyle \frac { A + k } { 2 A } \frac { B _ { \infty } \big ( e ^ { B _ { \infty } } - 1 \big ) } { e ^ { B _ { \infty } } + k - 1 } \asymp { B _ { \infty } } . \qquad ( k \leq e ^ { B _ { \infty } } , { B _ { \infty } } > 1 , 1 / 2 \leq ( A + k ) / ( 2 A ) \leq 1 ) } \end{array}
$$

Consequently,

$$
L _ { \theta } ( \pi _ { f } ) = \frac { \rho ( A + k ) } { 2 A d } \frac { B _ { \infty } ( e ^ { B _ { \infty } } - 1 ) } { e ^ { B _ { \infty } } + k - 1 } \sum _ { x = 1 } ^ { d } \mathbf { 1 } \{ f _ { x } \neq \theta _ { x } \} \asymp \frac { \rho B _ { \infty } } { d } \sum _ { x = 1 } ^ { d } \mathbf { 1 } \{ f _ { x } \neq \theta _ { x } \} .\tag{58}
$$

Identification error and risk. Let θ be uniform on $[ k ] ^ { d }$ . By Equation (57), querying $a _ { j }$ at $s _ { 1 , x }$ reveals exactly $\mathbf { 1 } \{ j = \theta _ { x } \}$ for $j \le k$ , and reveals no information about $\theta _ { x }$ for $j > k$ . Feedback at $s _ { 1 , 0 }$ is independent of

θ. Let $\begin{array} { r } { M _ { x } = \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ s _ { t , 1 } = s _ { 1 , x } \} } \end{array}$ and write the output as $\pi _ { \widehat { f } } ,$ assigning label $( 1 , \ldots , 1 )$ to the additional candidates. Conditional on the initial-state sequence and all other labels, $M _ { x }$ is fixed and $\theta _ { x }$ remains uniform on [k]. Lemma D.8, followed by averaging, gives

$$
\begin{array} { r l } & { \mathbb { P } \{ \widehat { f } _ { x } \neq \theta _ { x } \} \geq \mathbb { E } \left[ 1 - \frac { M _ { x } + 1 } { k } \right] } \\ & { \qquad \geq 1 - \frac { ( k - 1 ) / 8 + 1 } { k } = \frac { 7 ( k - 1 ) } { 8 k } \geq \frac { 7 } { 1 6 } . \qquad ( \mathbb { E } \left[ M _ { x } \right] = T \rho / d \leq ( k - 1 ) / 8 , k \geq 2 ) } \end{array}
$$

For $\begin{array} { r } { W = d ^ { - 1 } \sum _ { x = 1 } ^ { d } \mathbf { 1 } \{ \widehat { f } _ { x } \neq \theta _ { x } \} } \end{array}$

$$
\mathbb { E } \left[ W \right] \geq 7 / 1 6 , \qquad \mathbb { P } \{ W > 1 / 8 \} \geq \frac { \mathbb { E } \left[ W \right] - 1 / 8 } { 1 - 1 / 8 } \geq \frac { 5 } { 1 4 } ,
$$

$$
( 0 \leq W \leq 1 )
$$

$$
L _ { \theta } ( { \widehat \pi } ) \gtrsim \rho B _ { \infty } \quad \mathrm { o n } ~ \{ W > 1 / 8 \} ,\tag{Equation (58)}
$$

$$
\begin{array} { c c } { \displaystyle { \rho B _ { \infty } \asymp B _ { \infty } \operatorname* { m i n } \Biggl \{ 1 , \frac { k d } { T } \Biggr \} } } & { \qquad ( k \geq 2 \mathrm { ~ a n d ~ t h e ~ d e f i n i t i o n ~ o f ~ } \rho ) } \\ { \displaystyle { \qquad \asymp B _ { \infty } \operatorname* { m i n } \Biggl \{ 1 , \frac { m \log N } { T \log ( e m ) } \Biggr \} . } } & { \qquad ( k \asymp m , d \asymp \log N / \log k , \log k \asymp \log ( e m ) ) } \end{array}
$$

Averaging over the finite teacher prior gives a fixed teacher with failure probability at least $5 / 1 4 > 1 / 4$ □

## C Offline learning

## C.1 Full feedback

ProofofProposition 4.1. If $\pi ^ { \star } \in \Pi$ , both empirical objectives have minimum zero. Nonnegativity of the local KL losses gives, for every $\pi \in \Pi$

$$
\sum _ { t = 1 } ^ { T } \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { t , h } ^ { \star } ; \pi ) = 0 \quad \Longleftrightarrow \quad \sum _ { t = 1 } ^ { T } \operatorname* { m i n } \biggr \{ 1 , \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { t , h } ^ { \star } ; \pi ) \biggr \} = 0 .
$$

Thus Equation (11) and Equation (12) have the same minimizers, and Theorem 4.2 with $L ^ { \star } = 0$ proves the first claim. The lower bound Equation (52) shows that this estimation rate is minimax optimal.

MDP and policy class. For the misspecified instance, fix $B \geq 3 2$ and set

$$
\epsilon = \frac { 1 } { B } , \qquad p = \frac { 1 } { 1 6 \log B } , \qquad q = \frac { p } { B } .
$$

The $H = 2 { \bf M } { \bf D } \mathrm { P }$ has action set $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } \}$ and state layers $S _ { 1 } = \{ s _ { 1 , 1 } , s _ { 1 , 2 } \}$ and $S _ { 2 } = \{ s _ { 2 , 0 } , s _ { 2 , 1 } , s _ { 2 , 2 } \}$ . The initial law is $\mu ( s _ { 1 , 1 } ) = \mu ( s _ { 1 , 2 } ) = 1 / 2$ , and

$$
\begin{array} { r } { \mathsf { P } _ { 1 } ( s _ { 2 , j } \mid s _ { 1 , j } , a _ { 1 } ) = 1 , \qquad \mathsf { P } _ { 1 } ( s _ { 2 , 0 } \mid s _ { 1 , j } , a _ { 2 } ) = 1 \quad ( j \in \{ 1 , 2 \} ) . } \end{array}
$$

The episode ends after the second action. The table gives the probability of $a _ { 1 }$ at each state, with its complement assigned to $a _ { 2 }$ . The candidate class is $\Pi = \{ \pi _ { \mathrm { g o o d } } , \pi _ { \mathrm { b a d } } \}$

$$
\begin{array} { r } { \overbrace { \boldsymbol { \pi } ^ { \star } \quad \overleftarrow { \boldsymbol { \epsilon } } \quad \overleftarrow { \boldsymbol { \epsilon } } \quad \begin{array} { c } { \boldsymbol { \it { q } } } \\ { \boldsymbol { \mu } } \\ { \boldsymbol { \pi } _ { \mathrm { b a d } } } \end{array} } ^ { \mathrm { ~ ~ } \left| \begin{array} { c c } { \boldsymbol { s } _ { 1 , 1 } } & { \boldsymbol { s } _ { 1 , 2 } } \end{array} \right| \boldsymbol { s } _ { 2 , 0 } } \quad \begin{array} { c c } { \boldsymbol { s } _ { 2 , 1 } } & { \boldsymbol { s } _ { 2 , 2 } } \\ { \boldsymbol { e } ^ { - B / 2 } } & { \boldsymbol { e } ^ { - B / 2 } } \end{array} } \\ { \pi _ { \mathrm { b a d } } \quad \left| \begin{array} { c c } { \boldsymbol { 0 } } & { \boldsymbol { q } } \\ { \boldsymbol { \epsilon } } & { \boldsymbol { p } } \end{array} \right| \quad \mathrm { 0 } \quad \begin{array} { c c } { 1 } & { \boldsymbol { e } ^ { - B / 2 } } \\ { \boldsymbol { e } ^ { - B / 2 } } & { 1 } \end{array} } \end{array}
$$

Boundedness. The only nonzero local KL losses are

$$
\begin{array} { r l r } & { k _ { 1 } ( s _ { 1 , 1 } ; \pi _ { \mathrm { g o o d } } ) = \ell : = - \log ( 1 - \epsilon ) , } & { k _ { 2 } ( s _ { 2 , 1 } ; \pi _ { \mathrm { g o o d } } ) = B / 2 , } \\ & { k _ { 1 } ( s _ { 1 , 2 } ; \pi _ { \mathrm { b a d } } ) = r : = p \log \displaystyle \frac { p } { q } + ( 1 - p ) \log \displaystyle \frac { 1 - p } { 1 - q } , } & { k _ { 2 } ( s _ { 2 , 2 } ; \pi _ { \mathrm { b a d } } ) = B / 2 . } \end{array}
$$

Since $0 < q < p < 1$

$$
\begin{array} { c c c } { { \displaystyle \epsilon \leq \ell \leq 2 \epsilon , } } & { { } } & { { 0 \leq r \leq p \log ( p / q ) = \displaystyle \frac { 1 } { 1 6 } , \qquad ( \epsilon \leq 1 / 2 , p / q = B ) \smallskip } } \\ { { \displaystyle \sum _ { h = 1 } ^ { 2 } k _ { h } ( s _ { h } ; \pi ) \leq \displaystyle \frac { B } { 2 } + \operatorname* { m a x } \{ \ell , r \} \leq B ~ } } & { { } } & { { \mathrm { f o r ~ e v e r y ~ } \pi \in \mathrm { I I ~ a n d ~ } \tau \in \mathcal T _ { \mathrm { e n v } } . } } \end{array}
$$

This verifies the pathwise condition of Lemma D.1, including trajectories that a candidate never generates, and hence Assumption 2.1.

Population objectives and empirical selection. Under the teacher, $s _ { 2 , 1 }$ and $s _ { 2 , 2 }$ have probabilities $\epsilon / 2$ and $q / 2$ . Under $\pi _ { \mathrm { g o o d } } , s _ { 2 , 1 }$ is never visited, whereas $\pi _ { \mathrm { b a d } }$ visits $s _ { 2 , 2 }$ with probability $p / 2$ . Consequently,

$$
\frac { \displaystyle { \frac { \mathbb { E } _ { P ^ { \pi ^ { \star } } } \left[ \sum _ { h = 1 } ^ { 2 } k _ { h } ( s _ { h } ; \pi ) \right] } { \pi _ { \mathrm { g o o d } } } \frac { \displaystyle { L ( \pi ) } } { \displaystyle { ( \ell + \epsilon B / 2 ) / 2 } } } } { \pi _ { \mathrm { b a d } } } { ( r + q B / 2 ) / 2 } \quad \quad \quad { ( r + p B / 2 ) / 2 } \quad
$$

The untruncated objective ranks the candidates in the wrong order,

$$
\begin{array} { r l r } {  { \mathbb { E } _ { P ^ { \pi ^ { \star } } } [ \sum _ { h = 1 } ^ { 2 } k _ { h } ( s _ { h } ; \pi _ { \mathrm { b a d } } ) ] \leq \frac { 1 } { 3 2 } + \frac { p } { 4 } \leq \frac { 1 } { 1 6 } , } } & { \quad } & { \quad ( r \leq 1 / 1 6 , q = p / B ) } \\ & { } & { \mathbb { E } _ { P ^ { \pi ^ { \star } } } [ \sum _ { h = 1 } ^ { 2 } k _ { h } ( s _ { h } ; \pi _ { \mathrm { g o o d } } ) ] \geq \frac { \epsilon B } { 4 } = \frac { 1 } { 4 } , } & { \quad } & { \quad ( \ell \geq 0 , \epsilon = 1 / B ) } \\ & { } & { \frac { L ( \pi _ { \mathrm { b a d } } ) } { L ( \pi _ { \mathrm { g o o d } } ) } = \frac { r + p B / 2 } { \ell } \geq \frac { p B } { 4 \epsilon } = \frac { B ^ { 2 } } { 6 4 \log B } > 1 . } & { \quad } & { \quad ( r \geq 0 , \ell \leq 2 \epsilon , B \geq 3 2 ) } \end{array}
$$

In particular, $L ^ { \star } = L ( \pi _ { \mathrm { g o o d } } ) > 0$ . For each $\pi \in \Pi$ , the strong law of large numbers gives, with probability one,

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { h = 1 } ^ { 2 } k _ { h } ( s _ { t , h } ^ { \star } ; \pi ) \longrightarrow \mathbb { E } _ { P ^ { \pi ^ { \star } } } \left[ \sum _ { h = 1 } ^ { 2 } k _ { h } ( s _ { h } ; \pi ) \right] .
$$

The population objective of $\pi _ { \mathrm { b a d } }$ is smaller by at least $3 / 1 6$ , so

$$
\mathbb { P } \{ \widehat { \pi } _ { \mathrm { r a w } } = \pi _ { \mathrm { b a d } } \} \longrightarrow 1 .
$$

Consequently, any guarantee with an estimation term vanishing as $T \to \infty$ requires

$$
C _ { \mathrm { a p p } } \geq \frac { L ( \pi _ { \mathrm { b a d } } ) } { L ^ { \star } } \geq \frac { B ^ { 2 } } { 6 4 \log B } = \Omega \left( \frac { B ^ { 2 } } { \log ( e B ) } \right) .
$$

Proof of Theorem 4.2. Each teacher trajectory contributes $\begin{array} { r } { \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { t , h } ^ { \star } ; \pi ) \in [ 0 , 1 ] } \end{array}$ to Equation (12), and the teacher trajectories are independent. Let $\pi ^ { \dagger } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } L ( \pi )$ . With probability at least $1 - \delta ,$

$$
\begin{array} { r l } & { \overline { { L } } ( \widehat { \pi } ) \lesssim \mathbb { E } _ { P r ^ { \star } } \left[ \displaystyle \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \widehat { \pi } ) \right] } \\ & { \qquad \lesssim \displaystyle \operatorname* { m i n } _ { \alpha \in \Pi } \mathbb { E } _ { P r ^ { \star } } \left[ \displaystyle \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ) \right] + \displaystyle \frac { \log ( | \Pi | / \delta ) } { T } } \\ & { \qquad \leq \mathbb { E } _ { P r ^ { \star } } \left[ \displaystyle \sum _ { h = 1 } ^ { H } e _ { h } ( \xi _ { h } ; \pi ^ { \star } ) \right] + \displaystyle \frac { \log ( | \Pi | / \delta ) } { T } } \\ & { \qquad \lesssim \overline { { L } } ( \pi ^ { \dagger } ) + \displaystyle \frac { \log ( | \Pi | / \delta ) } { T } } \\ & { \qquad \leq L ^ { \star } + \displaystyle \frac { \log ( | \Pi | / \delta ) } { T } . } \end{array}\tag{Equation (20)}
$$

(Lemma D.7)

(π<sup>†</sup> ∈ Π)

(Equation (20))

$$
( \overline { { L } } ( \pi ^ { \dagger } ) \leq L ( \pi ^ { \dagger } ) = L ^ { \star } )\tag{59}
$$

The expectations under $P ^ { \pi ^ { \star } }$ evaluate each displayed candidate on an independent teacher trajectory. Under Assumption 2.1,

$$
\begin{array} { l } { { \displaystyle { \cal L } ( \widehat { \pi } ) \leq ( 1 + B ) { \cal L } ( \widehat { \pi } ) } } \\ { { \displaystyle ~ \lesssim B \left[ { \cal L } ^ { \star } + \frac { \log ( | \Pi | / \delta ) } { T } \right] } . } \end{array}\tag{Lemma D.2}
$$

(Equation (59))

## C.2 Action feedback

Proposition C.1 (Insufficiency of the relative second-moment bound). For every $B \geq 1$ and integer $T \geq 1$ there exist an H = 1 MDP with one state and three actions and a class Π oftwo policies such that every possibly randomized offline action-feedback learner returning $\widehat { \pi } \in$ Π has a teacher $\pi ^ { \star } \in \Pi$ for which Assumption 2.1 holds and

$$
L ^ { \star } = 0 , \qquad \mathbb { P } \{ L ( \widehat { \pi } ) \geq B / 2 \} \geq \frac { 1 } { 4 } .
$$

Proof. Fix $B \geq 1$ and $T \geq 1$

MDP and policy class. The state and action sets are $\begin{array} { r } { \mathcal { S } = \{ s _ { 1 , 1 } \} } \end{array}$ and $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , a _ { 3 } \}$ , with $\mu ( s _ { 1 , 1 } ) = 1$ . The episode ends after the first action. Let

$$
\epsilon = \frac { 1 } { 4 T } , \qquad r = \epsilon e ^ { - B / \epsilon } , \qquad \Pi = \{ \pi _ { 0 } , \pi _ { 1 } \} ,
$$

where the action probabilities at $s _ { 1 , 1 }$ are

$$
\begin{array} { c } { { \begin{array} { c c c } { { } } & { { a _ { 1 } } } & { { a _ { 2 } } } & { { a _ { 3 } } } \\ { { \pi _ { 0 } } } & { { 1 - \epsilon - r } } & { { \epsilon } } & { { r } } \\ { { \pi _ { 1 } } } & { { 1 - \epsilon - r } } & { { r } } & { { \epsilon } } \end{array} } } \end{array}
$$

The teacher is $\pi ^ { \star } = \pi _ { \theta }$ for an unknown $\theta \in \{ 0 , 1 \}$

□

Boundedness and candidate risks. All action probabilities are positive, and for either teacher,

$$
\begin{array} { r l r } { k _ { 1 } ( s _ { 1 , 1 } ; \pi _ { 1 - \theta } ) = \mathrm { K L } ( P ^ { \pi _ { 1 - \theta } } \| P ^ { \pi _ { \theta } } ) = \epsilon \log \frac { \epsilon } { r } + r \log \frac { r } { \epsilon } \qquad } & { \quad } & { \mathrm { ( T h e ~ t w o ~ p o l i c i e s ~ a g r e e ~ o n ~ } a _ { 1 } ) } \\ { = ( \epsilon - r ) \frac { B } { \epsilon } = B \bigl ( 1 - e ^ { - B / \epsilon } \bigr ) \in [ B / 2 , B ] . } & { \quad } & { \quad ( B / \epsilon = 4 B T \geq 4 ) \qquad ( 6 0 ) } \end{array}
$$

Since $k _ { 1 } ( s _ { 1 , 1 } ; \pi _ { \theta } ) = 0$ , both candidates satisfy $K _ { \pi } \leq B$ pointwise. Lemma D.1 verifies Assumption 2.1, and $L ^ { \star } = L ( \pi _ { \theta } ) = 0$

Indistinguishable observations. Write $\mathbb { P } _ { \theta }$ for the law of the training data and the learner’s output when $\pi ^ { \star } = \pi _ { \theta }$ . For $E = \{ a _ { t , 1 } ^ { \star } = a _ { 1 }$ for all $t \in [ T ] \}$ ,

$$
\begin{array} { l } { \displaystyle \mathbb { P } _ { \theta } ( E ) = ( 1 - \epsilon - r ) ^ { T } } \\ { \displaystyle \quad \geq 1 - T ( \epsilon + r ) \geq 1 - 2 T \epsilon = \frac 1 2 . } \end{array}
$$

(Independent teacher trajectories)

(Bernoulli’s inequality, $r \leq \epsilon )$

On $E ,$ both teachers produce the same $T$ observations $( s _ { 1 , 1 } , a _ { 1 } , 1 - \epsilon - r )$ , including the reported action probabilities. Thus the conditional output distributions coincide, and $\widehat { \pi } \in \{ \pi _ { 0 } , \pi _ { 1 } \}$ implies

$$
\mathbb { P } _ { 0 } ( \widehat { \pi } \neq \pi _ { 0 } \mid E ) + \mathbb { P } _ { 1 } ( \widehat { \pi } \neq \pi _ { 1 } \mid E ) = 1 .
$$

Consequently,

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \theta \in \{ 0 , 1 \} } \mathbb { P } _ { \theta } \Big \{ \mathrm { K L } ( P ^ { \widehat { \pi } } \| P ^ { \pi _ { \theta } } ) \geq B / 2 \Big \} = \displaystyle \operatorname* { m a x } _ { \theta \in \{ 0 , 1 \} } \mathbb { P } _ { \theta } \{ \widehat { \pi } \neq \pi _ { \theta } \} } & { } \\ { \displaystyle \geq \frac { 1 } { 2 } \displaystyle \sum _ { \theta = 0 } ^ { 1 } \mathbb { P } _ { \theta } \big ( \{ \widehat { \pi } \neq \pi _ { \theta } \} \cap E \big ) } & { } \\ { \displaystyle } & { = \frac { \big ( 1 - \epsilon - r \big ) ^ { T } } { 2 } \geq \frac { 1 } { 4 } . } \end{array}\tag{Equation (60)}
$$

(Maximum bounds the average)

(Identical observations on E)

ProofofProposition 4.3. The estimator in Rohatgi et al. [2025, Appendix C, Eq. (11)] maximizes the empirical log-likelihood of $( P ^ { \pi } + P ^ { \pi ^ { \star } } ) / 2$ over $\pi \in \Pi$ . Removing terms independent of π gives the equivalent objective

$$
\widehat { \pi } \in \underset { \pi \in \Pi } { \arg \operatorname* { m i n } } - \sum _ { t = 1 } ^ { T } \log \biggr ( 1 + \frac { P ^ { \pi } ( \tau _ { t } ^ { \star } ) } { P ^ { \pi ^ { \star } } ( \tau _ { t } ^ { \star } ) } \biggr ) .\tag{61}
$$

For every observed teacher trajectory, the common initial-state and transition probabilities cancel, so

$$
\frac { P ^ { \pi } ( \tau _ { t } ^ { \star } ) } { P ^ { \pi ^ { \star } } ( \tau _ { t } ^ { \star } ) } = \prod _ { h = 1 } ^ { H } \frac { \pi _ { h } ( a _ { t , h } ^ { \star } \mid s _ { t , h } ^ { \star } ) } { \pi _ { h } ^ { \star } ( a _ { t , h } ^ { \star } \mid s _ { t , h } ^ { \star } ) } .\tag{62}
$$

Thus action feedback suffices to evaluate Equation (61) without knowing the dynamics.

Under Assumption 2.2, with probability at least $1 - \delta .$

$$
\begin{array} { r l } & { L ( \widehat \pi ) \lesssim B _ { \infty } D _ { \mathrm { H } } ^ { 2 } ( P ^ { \widehat \pi } , P ^ { \pi ^ { \star } } ) } \\ & { \qquad \lesssim B _ { \infty } \left[ \underset { \pi \in \Pi } { \operatorname* { m i n } } D _ { \mathrm { H } } ^ { 2 } ( P ^ { \pi } , P ^ { \pi ^ { \star } } ) + \frac { \log ( | \Pi | / \delta ) } { T } \right] } \\ & { \qquad \leq B _ { \infty } \left[ L ^ { \star } + \frac { \log ( | \Pi | / \delta ) } { T } \right] . } \end{array}\tag{Lemma D.5, B<sub>∞</sub> > 1}
$$

(Rohatgi et al., 2025, Proposition C.1)

(Equation (67))

## D Auxiliary lemmas

## D.1 Second moments and clipping

Throughout this subsection, write $\begin{array} { r } { K _ { \pi } ( \tau ) : = \sum _ { h = 1 } ^ { H } k _ { h } ( s _ { h } ; \pi ) } \end{array}$

Lemma D.1 (Sufficient conditions for the second-moment bound). Each ofthefollowing conditions is sufficient for Assumption 2.1.

1. Pathwise accumulated KL bound. If $B \geq 1$ and, for every $\pi \in \Pi$ and $\tau \in \mathcal { T } _ { \mathrm { e n v } } , K _ { \pi } ( \tau ) \leq B ,$ , then Assumption 2.1 holds.

2. Trajectory likelihood-ratio bound. Assumption 2.2 implies Assumption 2.1 with $B = 4 ( B _ { \infty } + 1 )$

Proof of Lemma $D . l .$ . For the first claim, nonnegativity and the pathwise bound give $K _ { \pi } ^ { 2 } \le B K _ { \pi }$ . Taking expectation under $P ^ { \pi }$ yields

$$
\begin{array} { r } { \mathbb { E } _ { P ^ { \pi } } [ K _ { \pi } ^ { 2 } ] \leq B \mathbb { E } _ { P ^ { \pi } } [ K _ { \pi } ] = B L ( \pi ) . } \end{array}
$$

For the second claim, fix $\pi \in \Pi$ and write

$$
P = P ^ { \pi } , \qquad Q = P ^ { \pi ^ { \star } } , \qquad b = B _ { \infty } , \qquad K = K _ { \pi } .
$$

Abbreviate $k _ { h } = k _ { h } ( s _ { h } ; \pi )$ . For $h = 1 , \ldots , H$ , let $\xi _ { h } = ( s _ { 1 } , a _ { 1 } , \dots , s _ { h } )$ denote the history immediately before action $a _ { h }$ . Let $\mathcal { F } _ { h - 1 } = \sigma ( \xi _ { h } )$ and let $\mathcal { F } _ { H } = \sigma ( \tau )$ . For $h = 0 , \ldots , H$ , define the prefix likelihood ratio and its logarithm by

$$
R _ { h } = \prod _ { j = 1 } ^ { h } { \frac { \pi _ { j } ( a _ { j } \mid s _ { j } ) } { \pi _ { j } ^ { \star } ( a _ { j } \mid s _ { j } ) } } , \qquad X _ { h } = \log R _ { h } ,
$$

where logarithms are evaluated P-almost surely. The common initial-state distribution gives $R _ { 0 } = 1$ and $X _ { 0 } = 0$ Moreover,

$$
R _ { h } = \mathbb { E } _ { Q } [ R _ { H } ~ | ~ \mathcal { F } _ { h } ] \leq e ^ { b } , \qquad \mathrm { s o } \qquad X _ { h } \leq b .
$$

Let $Y _ { h } = X _ { h } - X _ { h - 1 }$ . Since the common transition probabilities cancel,

$$
Y _ { h } = \log \frac { \pi _ { h } ( a _ { h } \mid s _ { h } ) } { \pi _ { h } ^ { \star } ( a _ { h } \mid s _ { h } ) } , \qquad \mathbb { E } _ { P } [ Y _ { h } \mid \mathcal { F } _ { h - 1 } ] = k _ { h } .
$$

All subsequent quantities are integrable. Indeed, the likelihood-ratio bound implies $P \ll Q$

$$
\mathbb { E } _ { P } [ X _ { h } ^ { 2 } ] = \mathbb { E } _ { Q } [ R _ { h } ( \log R _ { h } ) ^ { 2 } ] \le b ^ { 2 } + \frac { 4 } { e ^ { 2 } } ,
$$

and $0 \leq k _ { h } \leq b - X _ { h - 1 }$ . Since H is finite, K is square-integrable.

By the tower property,

$$
\begin{array} { r } { \mathbb { E } _ { P } \left[ \left. \sum _ { j = h + 1 } ^ { H } k _ { j } \right| \mathcal { F } _ { h - 1 } \right] = \mathbb { E } _ { P } [ X _ { H } - X _ { h - 1 } \mid \mathcal { F } _ { h - 1 } ] - k _ { h } } \\ { \le b - X _ { h - 1 } - k _ { h } . } \end{array}\tag{63}
$$

Expanding $K ^ { 2 }$ and applying Equation (63) therefore gives

$$
\begin{array} { r l } {  { \mathbb { E } _ { P } [ K ^ { 2 } ] = \mathbb { E } _ { P } [ \sum _ { h = 1 } ^ { H } ( k _ { h } ^ { 2 } + 2 k _ { h } \sum _ { j = h + 1 } ^ { H } k _ { j } ) ] } \quad } & { } \\ & { \leq 2 \mathbb { E } _ { P } [ \sum _ { h = 1 } ^ { H } k _ { h } ( b - X _ { h - 1 } ) ] - \mathbb { E } _ { P } [ \sum _ { h = 1 } ^ { H } k _ { h } ^ { 2 } ] } \\ & { \leq 2 \mathbb { E } _ { P } [ \sum _ { h = 1 } ^ { H } k _ { h } ( b - X _ { h - 1 } ) ] . } \end{array}\tag{64}
$$

We next use the scalar inequality

$$
u ( \log u ) ^ { 2 } \leq ( c + 2 ) ( u \log u - u + 1 ) , \qquad 0 \leq u \leq e ^ { c } , \quad c \geq 0 .\tag{65}
$$

At $u = 0$ , the left-hand side is defined as zero. For $0 < u \le 1$ , the stronger bound with coefficient 2 follows from $e ^ { t } \geq 1 + t + t ^ { 2 } / 2$ , with $t = -$ log u. For u $\geq 1$ , the stronger bound with coefficient log $u + 2$ follows from

$$
( \log u + 2 ) ( u \log u - u + 1 ) - u ( \log u ) ^ { 2 } = ( u + 1 ) \log u - 2 ( u - 1 ) \geq 0 .
$$

This proves Equation (65).

Conditional on $\mathcal { F } _ { h - 1 }$ , the action likelihood ratio is bounded above by $e ^ { b - X _ { h - 1 } }$ . Applying Equation (65) to this ratio, with $c = b - X _ { h - 1 }$ , and averaging under the teacher’s conditional action distribution yields

$$
\mathbb { E } _ { P } [ Y _ { h } ^ { 2 } \mid \mathcal { F } _ { h - 1 } ] \le ( b - X _ { h - 1 } + 2 ) k _ { h } .\tag{66}
$$

Telescoping $X _ { h } ^ { 2 }$ and using Equation (66), we obtain

$$
\begin{array} { r l } {  { \mathbb { E } _ { P } [ X _ { H } ^ { 2 } ] = \mathbb { E } _ { P } [ \sum _ { h = 1 } ^ { H } \big ( 2 X _ { h - 1 } k _ { h } + \mathbb { E } _ { P } [ Y _ { h } ^ { 2 } \mid \mathcal { F } _ { h - 1 } ] \big ) ] } } \\ & { \leq \mathbb { E } _ { P } [ \sum _ { h = 1 } ^ { H } ( b + X _ { h - 1 } + 2 ) k _ { h } ] } \\ & { = 2 ( b + 1 ) L ( \pi ) - \mathbb { E } _ { P } [ \sum _ { h = 1 } ^ { H } k _ { h } ( b - X _ { h - 1 } ) ] . } \end{array}
$$

Since $\mathbb { E } _ { P } [ X _ { H } ^ { 2 } ] \ge 0$ , combining this inequality with Equation (64) proves

$$
\begin{array} { r } { \mathbb { E } _ { P ^ { \pi } } [ K _ { \pi } ^ { 2 } ] \le 4 ( b + 1 ) L ( \pi ) = 4 ( B _ { \infty } + 1 ) L ( \pi ) . } \end{array}
$$

Lemma D.2 (Clipping under a relative second-moment bound). Under Assumption $2 . l ,$ , every $\pi \in$ Π satisfies $L ( \pi ) \leq B$ and

$$
\begin{array} { r } { L ( \pi ) : = \mathbb { E } _ { P ^ { \pi } } [ \operatorname* { m i n } \lbrace 1 , K _ { \pi } \rbrace ] \leq L ( \pi ) \leq ( 1 + B ) \overline { { L } } ( \pi ) \leq 2 B \overline { { L } } ( \pi ) . } \end{array}
$$

Proof. Fix π and write $K = K _ { \pi }$ and $f = \operatorname* { m i n } \{ 1 , K \}$ . Jensen’s inequality gives $L ( \pi ) ^ { 2 } \leq \mathbb { E } _ { P ^ { \pi } } [ K ^ { 2 } ] \leq B L ( \pi )$ hence $L ( \pi ) \leq B$ . The clipping inequality is immediate if $L ( \pi ) = 0$ . Otherwise, set $K / \sqrt { f } = K ^ { 2 } / f = 0$ when $K = 0$ . Cauchy–Schwarz and $K ^ { 2 } / f \le K + K ^ { 2 }$ give

$$
\begin{array} { r l r } & {  { L ( \pi ) ^ { 2 } = \bigg ( \mathbb { E } _ { P ^ { \pi } } \bigg [ \sqrt { f } \frac { K } { \sqrt { f } } \bigg ] \bigg ) ^ { 2 } \leq \mathbb { E } _ { P ^ { \pi } } [ f ] \mathbb { E } _ { P ^ { \pi } } [ K ^ { 2 } / f ] } } \\ & { } & { \leq \overline { { L } } ( \pi ) \big ( L ( \pi ) + \mathbb { E } _ { P ^ { \pi } } [ K ^ { 2 } ] \big ) \leq ( 1 + B ) \overline { { L } } ( \pi ) L ( \pi ) . } \end{array}
$$

Dividing by $L ( \pi ) > 0$ proves the upper bound. The lower bound follows from $f \leq K$ , and $1 + B \le 2 B$ since $B \geq 1$ □

## D.2 KL and Hellinger inequalities

All distributions below are on a common discrete space. We write $P \ll Q$ when $Q ( z ) = 0$ implies $P ( z ) = 0$ Ratios $P ( z ) / Q ( z )$ are evaluated where $Q ( z ) > 0$ , and equal zero when $P ( z ) = 0$

Lemma D.3 (KL variational inequality [Polyanskiy and Wu, 2024, Theorem 4.6]). Let $R , S$ be probability distributions with $\mathrm { K L } ( R \| S ) < \infty .$ . For every real-valuedfunction g satisfying $0 < \mathbb { E } _ { S } \left[ e ^ { g } \right] < \infty .$

$$
\mathrm { K L } ( R \| S ) \geq \mathbb { E } _ { R } \left[ g \right] - \log \mathbb { E } _ { S } \left[ e ^ { g } \right] .
$$

Lemma D.4 (Elementary Hellinger inequalities). For probability distributions $P , Q ,$

$$
D _ { \mathrm { H } } ^ { 2 } ( P , Q ) = \sum _ { z } ( \sqrt { P ( z ) } - \sqrt { Q ( z ) } ) ^ { 2 } \leq \mathrm { K L } ( P \| Q ) .\tag{67}
$$

For conditional probabilities $r ( y \mid x ) \geq 0$ with $\textstyle \sum _ { y } r ( y \mid x ) = 1$ , define $\begin{array} { r } { \widetilde { P } ( y ) = \sum _ { x } r ( y \mid x ) P ( x ) } \end{array}$ and $\begin{array} { r } { \widetilde Q ( y ) = \sum _ { x } r ( y \mid x ) Q ( x ) } \end{array}$ . Then

$$
D _ { \mathrm { H } } ( \widetilde { P } , \widetilde { Q } ) \leq D _ { \mathrm { H } } ( P , Q ) .\tag{68}
$$

For afunction F with $0 \leq F ( z ) \leq M$

$$
\sqrt { \mathbb { E } _ { P } \left[ F \right] } \leq \sqrt { \mathbb { E } _ { Q } \left[ F \right] } + \sqrt { M } D _ { \mathrm { H } } ( P , Q ) .\tag{69}
$$

For distributions $R , S , S ^ { \prime }$ and a subset D of their common space,

$$
\sqrt { \sum _ { z \in D } \operatorname* { m i n } \{ R ( z ) , S ( z ) \} } \leq \sqrt { \sum _ { z \in D } \operatorname* { m i n } \{ R ( z ) , S ^ { \prime } ( z ) \} } + D _ { \mathrm { H } } ( S , S ^ { \prime } ) .\tag{70}
$$

Proof. For Equation (67), only $\mathrm { K L } ( P \| Q ) < \infty$ needs proof. The scalar inequality $( { \sqrt { x } } - 1 ) ^ { 2 } \leq x \log x - x + 1$ for $x \geq 0$ gives

$$
\begin{array} { r l } & { D _ { \mathrm { H } } ^ { 2 } ( P , Q ) = \mathbb { E } _ { Q } \left[ \left( \sqrt { P / Q } - 1 \right) ^ { 2 } \right] } \\ & { \qquad \leq \mathbb { E } _ { Q } \left[ \left( P / Q \right) \log ( P / Q ) - P / Q + 1 \right] } \\ & { \qquad = \mathrm { K L } ( P \| Q ) . } \end{array}\tag{P ≪ Q}
$$

$$
\begin{array} { r l r } & { } & { ( \mathrm { S c a l a r i n e q u a l i t y } ) } \\ & { } & { ( \sum _ { z } P ( z ) = \sum _ { z } Q ( z ) = 1 ) } \end{array}
$$

For Equation (68),

$$
\begin{array} { r l } & { D _ { \mathrm { H } } ^ { 2 } ( \widetilde { P } , \widetilde { Q } ) = 2 - 2 \displaystyle \sum _ { y } \sqrt { \left( \displaystyle \sum _ { x } r ( y \mid x ) P ( x ) \right) \left( \displaystyle \sum _ { x } r ( y \mid x ) Q ( x ) \right) } } \\ & { \qquad \le 2 - 2 \displaystyle \sum _ { y } \displaystyle \sum _ { x } r ( y \mid x ) \sqrt { P ( x ) Q ( x ) } } \\ & { \qquad = 2 - 2 \displaystyle \sum _ { x } \sqrt { P ( x ) Q ( x ) } = D _ { \mathrm { H } } ^ { 2 } ( P , Q ) . } \end{array}
$$

(Definition of $\widetilde { P } , \widetilde { Q } )$

(Cauchy–Schwarz)

$$
\begin{array} { r } { ( \sum _ { y } r ( y \mid x ) = 1 ) } \end{array}
$$

For Equation (69),

$$
\begin{array} { r l } & { \sqrt { \mathbb { E } _ { P } \left[ F \right] } = \| \sqrt { F } \sqrt { P } \| _ { 2 } } \\ & { \qquad \leq \| \sqrt { F } \sqrt { Q } \| _ { 2 } + \| \sqrt { F } ( \sqrt { P } - \sqrt { Q } ) \| _ { 2 } } \\ & { \qquad \leq \sqrt { \mathbb { E } _ { Q } \left[ F \right] } + \sqrt { M } D _ { \mathrm { H } } ( P , Q ) . } \end{array}
$$

$$
\Vert \cdot \Vert _ { 2 } )
$$

For Equation (70), the minimum is taken pointwise, and

$$
\begin{array} { r l r } {  { \sqrt { \sum _ { z \in D } \operatorname* { m i n } \{ R ( z ) , S ( z ) \} } = \| \mathbf { 1 } _ { D } \operatorname* { m i n } \{ \sqrt { R } , \sqrt { S } \} \| _ { 2 } } } & { ( \mathrm { D e f i n i t i o n ~ o f } \parallel \cdot \parallel _ { 2 } ) } \\ & { } & { \leq \| \mathbf { 1 } _ { D } \operatorname* { m i n } \{ \sqrt { R } , \sqrt { S ^ { \prime } } \} \| _ { 2 } + \| \mathbf { 1 } _ { D } ( \sqrt { S } - \sqrt { S ^ { \prime } } ) \| _ { 2 } } \\ & { } & { ( \mathrm { T r i a n g l e ~ i n e q u a l i t y ~ a n d ~ 1 - L i p s c h i t z ~ m i n i m m } ) } \\ & { } & { \leq \sqrt { \sum _ { z \in D } \operatorname* { m i n } \{ R ( z ) , S ^ { \prime } ( z ) \} } + D _ { \mathrm { H } } ( S , S ^ { \prime } ) . \qquad ( \mathbf { 1 } _ { D } \leq 1 ) } \end{array}
$$

Lemma D.5 (Hellinger-to-KL conversion). If $P \ll Q$ and $P ( z ) / Q ( z ) \leq e ^ { b } f o r b \geq 0 ,$ , then

$$
D _ { \mathrm { H } } ^ { 2 } ( P , Q ) \leq \mathrm { K L } ( P \| Q ) \lesssim ( 1 + b ) D _ { \mathrm { H } } ^ { 2 } ( P , Q ) , \qquad \mathrm { K L } ( P \| Q ) \leq b .
$$

Proof. For $0 \leq x \leq e ^ { b } .$ , the scalar comparison underlying Sason and Verdú [2016, Theorem 9], at Hellinger order $1 / 2 .$ , gives x log x $- x + 1 \lesssim ( 1 + b ) ( \sqrt { x } - 1 ) ^ { 2 }$ . Hence

$$
\begin{array} { r l } & { D _ { \mathrm { H } } ^ { 2 } ( P , Q ) \leq \mathrm { K L } ( P \Vert Q ) } \\ & { \qquad = \mathbb { E } _ { Q } \left[ ( P / Q ) \log ( P / Q ) - P / Q + 1 \right] } \\ & { \qquad \lesssim ( 1 + b ) \mathbb { E } _ { Q } \left[ \left( \sqrt { P / Q } - 1 \right) ^ { 2 } \right] } \\ & { \qquad = ( 1 + b ) D _ { \mathrm { H } } ^ { 2 } ( P , Q ) . } \end{array}
$$

(Equation (67))

$$
D _ { \mathrm { H } } ^ { 2 } )
$$

Also $\mathrm { K L } ( P \| Q ) = \mathbb { E } _ { P } \left[ \log ( P / Q ) \right] \leq b .$

## D.3 Concentration and testing bounds

Let $( \mathcal { F } _ { t } ) _ { t = 0 } ^ { T }$ be the training-history filtration. The history $\mathcal { F } _ { t - 1 }$ includes all observations and randomization before episode $t ,$ and $\mathbb { E } _ { t } \left[ \cdot \right] = \mathbb { E } \left[ \cdot \mid \mathcal { F } _ { t - 1 } \right]$ . In particular, $\pi _ { t }$ and its trajectory are both averaged over unless $\pi _ { t }$ is explicitly included in the conditioning.

Lemma D.6 (A fixed-time exponential inequality). Let $X _ { t }$ be $\mathcal { F } _ { t }$ -measurable with $| X _ { t } | \le 2$ . For fixed $T _ { i }$ $0 < \lambda \leq 1 / 2 ,$ , and $\epsilon \in ( 0 , 1 )$ , each of

$$
\sum _ { t = 1 } ^ { T } X _ { t } \leq \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ X _ { t } \right] + \lambda \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ X _ { t } ^ { 2 } \right] + \frac { \log ( 1 / \epsilon ) } { \lambda } ,\tag{71}
$$

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ X _ { t } \right] \leq \sum _ { t = 1 } ^ { T } X _ { t } + \lambda \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ X _ { t } ^ { 2 } \right] + { \frac { \log ( 1 / \epsilon ) } { \lambda } }\tag{72}
$$

holds with probability at least $1 - \epsilon .$

Proof. The exponential inequality of Foster et al. [2024, Lemma D.1], applied to $- \lambda X _ { t }$ , gives, with probability at least $1 - \epsilon$

$$
\begin{array} { r l r } { \displaystyle \sum _ { t = 1 } ^ { T } X _ { t } \leq \frac { 1 } { \lambda } \displaystyle \sum _ { t = 1 } ^ { T } \log \mathbb { E } _ { t } \left[ e ^ { \lambda X _ { t } } \right] + \frac { \log ( 1 / \epsilon ) } { \lambda } } & { \quad } & { \mathrm { ( F o s t e r ~ e t ~ a l . , 2 0 2 4 , L e m m a ~ D . 1 ) } } \\ { \displaystyle } & { \leq \frac { 1 } { \lambda } \displaystyle \sum _ { t = 1 } ^ { T } \log \bigl ( 1 + \lambda \mathbb { E } _ { t } \left[ X _ { t } \right] + \lambda ^ { 2 } \mathbb { E } _ { t } \left[ X _ { t } ^ { 2 } \right] \bigr ) } \\ { \displaystyle } & { \quad } & { + \frac { \log ( 1 / \epsilon ) } { \lambda } } \\ { \displaystyle } & { \leq \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ X _ { t } \right] + \lambda \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ X _ { t } ^ { 2 } \right] + \frac { \log ( 1 / \epsilon ) } { \lambda } . } & { \quad } & { \mathrm { ( l o g } ( 1 + x ) \leq x ) } \end{array}
$$

This is Equation (71). Applying the same inequality to $- X _ { t }$ gives Equation (72).

Lemma D.7 (Empirical minimization of bounded scores). Let the vectors $( Z _ { t } ( \pi ) ) _ { \pi \in \Pi }$ be independent and identically distributed across $t ,$ with $0 \leq Z _ { t } ( \pi ) \leq 1$ and $\mu ( \pi ) = \mathbb { E } \left[ Z _ { t } ( \pi ) \right]$ . Dependence across candidates within an episode is allowed. Every empirical minimizer $\widehat { \pi } \in$ arg min<sub>π∈Π</sub> $\scriptstyle \sum _ { t = 1 } ^ { T } Z _ { t } ( \pi )$ satisfies, with probability at least $1 - \delta ,$

$$
\mu ( \widehat { \pi } ) \lesssim \operatorname* { m i n } _ { \pi \in \Pi } \mu ( \pi ) + \frac { \log ( | \Pi | / \delta ) } { T } .
$$

Proof. For every $\pi ,$ independence across episodes and boundedness give $\mathbb { E } _ { t } \left[ Z _ { t } ( \pi ) \right] = \mu ( \pi )$ and $\mathbb { E } _ { t } \left\lceil Z _ { t } ( \pi ) ^ { 2 } \right\rceil \leq$ $\mu ( \pi )$ . With $\lambda = 1 / 2$ and $\epsilon = \delta / ( 2 | \boldsymbol { \Pi } | )$ , Equation (71) and Equation (72) hold simultaneously for all $\pi \in \Pi$ with probability at least $1 - \delta$ . On this event,

$$
\begin{array} { r l } & { \displaystyle \left. \frac { 1 } { T } \sum _ { t = 1 } ^ { T } Z _ { t } ( \pi ) - \mu ( \pi ) \right. \leq \frac { 1 } { 2 T } \sum _ { t = 1 } ^ { T } \mathbb { E } _ { t } \left[ Z _ { t } ( \pi ) ^ { 2 } \right] + \frac { 2 \log ( 2 | \Pi | / \delta ) } { T } } \\ & { \quad \quad \quad \leq \displaystyle \frac { 1 } { 2 } \mu ( \pi ) + \frac { 2 \log ( 2 | \Pi | / \delta ) } { T } . } \end{array}\tag{Equations (71) and (72)}
$$

$$
( Z _ { t } ( \pi ) ^ { 2 } \leq Z _ { t } ( \pi ) )\tag{73}
$$

For $\pi _ { \mu } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } \mu ( \pi )$

$$
\begin{array} { l } { \displaystyle \frac 1 2 \mu ( \widehat \pi ) \le \frac 1 T \sum _ { t = 1 } ^ { T } Z _ { t } ( \widehat \pi ) + \frac { 2 \log ( 2 | \Pi | / \delta ) } { T } } \\ { \displaystyle \qquad \le \frac 1 T \sum _ { t = 1 } ^ { T } Z _ { t } ( \pi _ { \mu } ) + \frac { 2 \log ( 2 | \Pi | / \delta ) } { T } } \\ { \displaystyle \qquad \le \frac 3 2 \mu ( \pi _ { \mu } ) + \frac { 4 \log ( 2 | \Pi | / \delta ) } { T } . } \end{array}\tag{Equation (73)}
$$

(Definition of πb)

(Equation (73))

$$
\begin{array} { r } { \mathrm { F o r } | \Pi | \ge 2 , \log ( 2 | \Pi | / \delta ) \lesssim \log ( | \Pi | / \delta ) . \mathrm { F o r } | \Pi | = 1 , \mu ( \widehat { \pi } ) = \operatorname* { m i n } _ { \pi \in \Pi } \mu ( \pi ) . } \end{array}
$$

Lemma D.8 (Adaptive equality tests). Let Θ be uniform on $[ k ]$ and $n \geq 0$ be an integer. A possibly randomized learner obtains information about Θ only through at most n tests. Each test selects $j \in [ k ]$ using previous answers and reveals $\mathbf { 1 } \{ j = \Theta \}$ . Everyfinal estimate ${ \widehat { \Theta } } \in [ k ]$ satisfies

$$
\mathbb { P } \{ \widehat { \Theta } \neq \Theta \} \geq 1 - \frac { n + 1 } { k } .
$$

Proof. The claim is trivial for $n \geq k - 1$ . For a deterministic learner, let $j _ { 1 } , \ldots , j _ { q } , q \le n$ , and b be its tests and final estimate when every answer is zero. Then

$$
\begin{array} { c } { { \Theta \not \in \{ j _ { 1 } , \dots , j _ { q } \} \quad \Longrightarrow \quad \widehat { \Theta } = b , } } \\ { { \{ \widehat { \Theta } = \Theta \} \subseteq \{ \Theta \in \{ j _ { 1 } , \dots , j _ { q } , b \} \} , } } \\ { { \mathbb { P } \{ \widehat { \Theta } = \Theta \} \leq \displaystyle \frac { \left| \left\{ j _ { 1 } , \dots , j _ { q } , b \right\} \right| } { k } \leq \displaystyle \frac { q + 1 } { k } \leq \displaystyle \frac { n + 1 } { k } . } } \end{array}
$$

(Every test returns zero)

(Uniform Θ)

For a randomized learner, the same bound holds conditional on its internal random choices and hence also after averaging. □