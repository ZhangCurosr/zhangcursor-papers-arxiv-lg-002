# On the Computational Tractability of Robust Bandits

Vanessa Kosoy<sup>1,2</sup> Vinayak Pathak<sup>2,∗</sup>

<sup>1</sup>Faculty of Mathematics, Technion, Haifa, Israel <sup>2</sup>CORAL<sup>†</sup>

## Abstract

Learning when the environment does not belong to the learner’s hypothesis class is typically handled using agnostic learning guarantees. However, for anything beyond supervised learning, agnostic guarantees are dificult to come by. Recently, imprecise bandits [51] (later renamed to robust bandits in Appel and Kosoy [3]) were introduced as another approach to unrealizable learning in the bandits setting and a Θ( T) regret learner was shown for a large class. However, no computational guarantees were provided. In this paper we identify a special case that admits a polynomial-time learner with O<sup>˜</sup>( T) regret. We also show that several small generalizations of this special case are NP-hard thus indicating that the special case is at the boundary of what is tractable. It has been recently suggested [48] that computationally eficient learners for unrealizable learning problems are crucial for solving the AI alignment problem. This work is a small step in that direction.

## 1 Introduction

Making sure that an artificial superintelligence does not pursue goals that are unaligned with humans is a challenging problem, and arguably one of the most important of our times [65, 10]. Several attempts have been made to get a handle on it [72, 2, 26, 42] and a recurring theme is that at its core it is a learning problem [24, 13, 77, 71, 64, 34, 35, 60, 41, 17, 55, 68, 6, 75, 4, 38, 16]. Indeed, imagine several agents coexisting in the universe and interacting with each other. Suppose that some of these agents (the AIs) are trying to act in the interests of other groups of agents (their users). The agents only have partial information about each other’s behaviours and goals as well as about the universe, and must ensure the universe evolves towards desirable states. Given the role of learning, it is not unreasonable to imagine that learning theory might help us pave the way towards solving the problem [48].

However, the currently known results in learning theory are likely insuficient to solve alignment. In particular, it has been suggested that designing better learners for unrealizable learning is crucial for solving the problem [48, 23]. Realizability is a strong assumption in general, but it becomes even more questionable in the multiagent learning scenarios described above.

One concrete issue with realizable learning is the grain of truth problem [39, 54, 81, 49]. Suppose Alice and Bob are playing a known game. Neither knows the other’s strategy but assumes that they lie in a class H of strategies [44, 66]. Can each player simultaneously achieve a low regret? The issue is that the strategy that achieves a low regret against all strategies in H may not itself be a member of H [62, 63]. In that case, if Alice’s strategy is in H, then Bob’s strategy will have to be outside of H and at least one of the players needs to solve an unrealizable learning problem. Finding a general and natural class H whose learner lies within H is called the grain of truth problem that remains open despite some recent progress using reflective oracles [28, 54, 81].

More generally, realizability works only when the agent is more expressive than its environment. However, if the environment contains other agents that are similar or more powerful than the agent doing the learning then realizability fails. This can happen, for example, when Alice and Bob are running similar source codes [79, 7], or when Bob is able to predict Alice’s actions [18, 67].

The usual response to unrealizability is to provide agnostic learning guarantees. Here, one makes no assumption about the environment but guarantees that the learner’s performance is competitive among a given class of strategies [46, 9]. However, designing learners with agnostic learning guarantees has been dificult. In supervised learning, there are many natural examples where agnostic learning is computationally intractable even though the realizable version is easy [21, 22]. In more sophisticated learning frameworks, agnostic learning is even statistically hard [53, 74]. In online learning, one can generally get a much better regret for the realizable case than for agnostic [59, 9]. In reinforcement learning, the agnostic case is usually statistically intractable [58, 43, 74] and only very simple problems are known to have good sample complexity bounds.

Closer to our current work, in an “agnostic” version of the bandits problem, one makes no assumptions about the mapping from arms to reward, but competes against the best policy in a given class. This can work if the class contains a small number of policies [5], however, impossibility results are easy to prove for infinite policy classes [14, 15]. Moreover, even for finite classes, since the regret grows as the square root of the size of the policy class, the situation is hopeless if the class is finite but large.

Our results In this paper, we build on the framework of robust bandits as introduced in Kosoy [51], which can be seen as an alternative to agnostic learning in the unrealizable case. The general idea is to say that we still have a known hypothesis class that contains a ground truth hypothesis, however, each hypothesis in our class provides only a partial specification of the environment. Given such a hypothesis, an adversary can pick any environment that is consistent with the specifications. The task of the learner is to compete with the optimal strategy that knows the true hypothesis but does not know the choices made by the adversary. Details can be found in Section 2. In Kosoy [51], a class of instances of the robust bandits problem, which can be seen as a generalization of stochastic linear bandits, was shown to have a $\Theta ( { \sqrt { T } } )$ regret. Here, we investigate the computational complexity of the learner. We demonstrate a special case of the setting of Kosoy [51] in which a polynomial time learner achieves $O ( \sqrt { T } )$ regret. We also show that generalizing this special instance in minor ways quickly leads to computational intractability, thus demonstrating that it is at th boundary of what is tractable.

## 2 Setting

A credal set over a domain D is a nonempty closed convex set of probability distributions over D, where closedness is with respect to the weak topology. We denote by D the set of all credal sets over D.

An instance of the robust bandits problem is a tuple $\mathcal { M } = ( X , D , H , r )$ , where X is a set of arms, D is a set of possible outcomes, H is a set of mappings of type $X  \bigsqcup D$ , and $r : X \times D $ R is a reward function.

The robust bandits game is played between a learner and an adversary for T rounds (both adversary and learner know T). At the beginning of the game, the adversary picks some $h ^ { \star } \in H$

The learner knows $X , D , H , r .$ , and $T ,$ but does not have knowledge of $h ^ { \star }$ . Then, at each round t:

1. Learner picks an arm $x _ { t } \in X$

2. Adversary picks a distribution $P _ { t } \in h ^ { \star } \left( x _ { t } \right)$

3. An outcome $y _ { t } \sim P _ { t }$ is drawn and shown to the learner as feedback.

4. Learner gets reward $r \left( x _ { t } , y _ { t } \right)$

Given a hypothesis $h \in H$ , we define the value functions $\begin{array} { r } { v _ { h } \left( x \right) : = \operatorname* { m i n } _ { P \in h \left( x \right) } \mathbb { E } _ { y \sim P } \left[ r \left( x , y \right) \right] } \end{array}$ and $V _ { h } : =$ $\operatorname* { m a x } _ { x \in X } v _ { h } \left( x \right)$ . We use the shorthand $v ^ { \ast } \left( x \right)$ and $V ^ { \star }$ to denote the value functions corresponding to the true hypothesis $h ^ { \star }$

The regret over horizon $T$ is

$$
R _ { T } : = T V ^ { \star } - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } r \left( x _ { t } , y _ { t } \right) \right] .
$$

Here the expectation in the second term is taken wrt the randomness in the learner’s and the adversary’s strategies as well as the randomness arising from sampling $y _ { t } \sim P _ { t }$

We are interested in learner strategies that can achieve a sublinear (in T) regret for all true hypotheses $h ^ { \star } \in H$ . We say that a robust bandits instance $\mathcal { M } = ( X , D , H , r )$ has a statistically eficient learner if such a learner strategy exists.

Computational tractability. In this paper, we are interested in studying the computational complexity of the learner, and thus we need to precisely specify the computational problem being solved. We assume that the learner is specialized to a parameterized family $\{ \mathcal { M } _ { \phi } \}$ of robust bandits instances. Then, a learner policy is a randomized algorithm that takes, as input, a history $h _ { t } = ( ( x _ { i } , y _ { i } ) ) _ { i = 1 } ^ { t }$ and an encoding of the parameters $\phi ,$ and outputs the next arm $x _ { t + 1 }$ . We use |ϕ| to denote the total encoding length of $\phi .$ We will say that a learner is computationally eficient if its policy runs in time polynomial in $| \phi |$ and $T$ and achieves a regret $R _ { T } \leq \mathrm { p o l y } \left( \left| \phi \right| \right) T ^ { 1 - \alpha }$ for some fixed constant $\alpha > 0 . { } ^ { 1 }$

## 2.1 Linear Robust Bandits

For most of this work, we study the following linear specialization of the instance $\mathcal { M } = ( X , D , H , r )$ defined above. Let $X \subseteq \mathbb { R } ^ { d _ { X } }$ be the arm set, and let $D \subseteq \mathbb { R } ^ { d _ { D } }$ be the outcome set. We assume that X and D are Euclidean balls of radii $R _ { X }$ and R (not necessarily centred at the origin). The reward is linear in the arm and outcome, $r \left( x , y \right) : = a ^ { T } x + b ^ { T } y$ . Write $\begin{array} { r } { C _ { r } : = \operatorname* { m a x } _ { X \times D } r - \operatorname* { m i n } _ { X \times D } r } \end{array}$ for the range of the reward.

Fix $d _ { C } \in \mathbb { N }$ . Each hypothesis h uses a set of $d _ { C }$ afine constraints as follows. Let $B \in \mathbb { R } ^ { d _ { C } \times d _ { X } }$ $C \in \mathbb { R } ^ { d _ { C } \times d _ { D } }$ be matrices, and $d \in \mathbb { R } ^ { d _ { C } }$ a vector. Define

$$
K _ { h } ( x ) : = \{ y \in D : C y + B x + d = 0 \} ,\tag{1}
$$

and $h ( x ) : = \{ P \in \Delta D : \mathbb { E } _ { y \sim P } [ y ] \in K _ { h } ( x ) \}$ . Let H consist of all hypotheses of this form for which $K _ { h } ( x )$ is nonempty for every $x \in X$ and which satisfy the transversality condition of Kosoy [51] with a cutof $S \in ( 0 , 1 ]$ . That is, h admits a representation $B , C , d$ such that, for every $x \in X$ and every $p \in \mathbb { R } ^ { d _ { D } }$ satisfying $C p + B x + d = 0$ with $p \notin D$ , we have dist $( p , D ) \geq S$ dist $\left( { p , K _ { h } \left( x \right) } \right)$ ). For a Euclidean ball, this says that the afine solution spaces defined by the constraints remain uniformly bounded away from tangency to $D$

We write $B ^ { \star } , C ^ { \star } , d ^ { \star }$ for such a representation of the true hypothesis $h ^ { \star }$ , and define $K ^ { \star } ( x ) : =$ $K _ { h ^ { \star } } ( x )$

Because r is linear in its outcome argument, the value function $v ^ { * }$ defined above satisfies $v ^ { * } ( x ) = \mathrm { m i n } _ { y \in K ^ { \star } ( x ) } r ( x , y )$

For the computational problem, the parameters ϕ of Section 2 are the dimensions $d _ { X } , d _ { D } , d _ { C }$ , the centres and radii of $X$ and $D _ { i }$ , the reward coeficients $a , b .$ , the cutof S, and an integer upper bound $C _ { \mathrm { m a x } } \geq C _ { r }$ on the reward range. They determine $X , D , r .$ , and H. We assume these numerical data are rational and given explicitly. Our regret bounds grow polynomially with $S ^ { - 1 }$ and $C _ { r }$ , so we want |ϕ| to grow at least as fast. We therefore require $S ^ { - 1 }$ to be an integer and give $S ^ { - 1 }$ and $C _ { \mathrm { m a x } }$ in unary. Then $S ^ { - 1 } \leq | \phi |$ and $C _ { r } \leq C _ { \mathrm { m a x } } \leq | \phi |$ . Requiring $S ^ { - 1 }$ to be an integer loses nothing, since lowering S only weakens the transversality condition.

Running time is measured in bit operations.

Now we state our main result.

Theorem 1 (eficient $\sqrt { T }$ learning). There exists a policy for linear robust bandits as defined above that, for every time horizon $T$ and every true hypothesis $h ^ { \star } \in H$ , runs in time polynomial in $| \phi |$ and $T$ and achieves regret $R _ { T } \leq \tilde { O } \left( \sqrt { T } \right)$ . The hidden factor depends polynomially on the dimensions, $S ^ { - 1 }$ , and $C _ { r }$ . The precise dependence is given in Theorem 21.

The dimensions, $S ^ { - 1 }$ , and $C _ { r } \leq C _ { \operatorname* { m a x } }$ are all at most $| \phi |$ . So this policy is computationally eficient in the sense of Section 2.

This dependence on $T$ is tight up to logarithmic factors. We first observe that linear robust bandits contain stochastic linear bandits as a special case, and then transfer the known lower bound for stochastic linear bandits.

Remark 2. Linear robust bandits contain stochastic linear bandits as a special case. In a stochastic linear bandit, the learner picks arms $x _ { t }$ from the unit ball $X \subseteq \mathbb { R } ^ { d }$ and observes a reward in $[ - 1 , 1 ]$ whose mean is $\theta ^ { T } x _ { t }$ , for an unknown θ in the unit ball $o f \mathbb { R } ^ { d }$ To write this as an instance of linear robust bandits, we let the outcome be the reward itself: take $D = [ - 1 , 1 ] , \ r ( x , y ) = y$ , and for each θ in the unit ball of $\mathbb { R } ^ { d }$ , define $h _ { \theta }$ by $K _ { h _ { \theta } } \left( x \right) : = \left\{ y \in D : y - \theta ^ { T } x = 0 \right\} = \left\{ \theta ^ { T } x \right\}$ . This has the required form with $C _ { \theta } = 1 , B _ { \theta } = - \theta ^ { T }$ , and $d _ { \theta } = 0$ . Since $\left| \theta ^ { T } x \right| \leq 1$ , the only solution of the constraint already lies in D, so transversality holds with $S = 1$ . The hypothesis $h _ { \theta } \left( x \right)$ now contains every distribution on $[ - 1 , 1 ]$ with mean $\theta ^ { T } x$ . In particular, it contains the reward distribution of the stochastic linear bandit at $x _ { i }$ , and the adversary may choose that distribution at every round. Against this adversary, the learner sees exactly what it would see in the stochastic linear bandit. The regret is also the same in both problems, since $v ^ { * } \left( x \right) = \theta ^ { T } x$ and $V ^ { \star } = \operatorname* { m a x } _ { x \in X } \theta ^ { T } x$ . So any learner for linear robust bandits is also a learner for stochastic linear bandits with the same regret.

Theorem 3 (lower bound, informal). Every learner for linear robust bandits has worst-case regret $\Omega \left( { \sqrt { T } } \right)$

Proof. Stochastic linear bandits have a worst-case regret lower bound of $\Omega \left( { \sqrt { T } } \right)$ [20]. By Remark 2, the same bound holds for linear robust bandits. □

## 3 Related work

Imprecise probabilities and credal sets While there is substantial theory to support the idea that a rational agent’s uncertain beliefs can be encoded using probabilities [73, 40], there are situations where probabilities aren’t enough (see Dalrymple [19] for examples). Encoding one’s belief using a convex set of probabilities [25] and making decisions by maximizing the worst case expected utility can fix many of the issues and this decision rule can be shown to be implied by natural axioms on preferences [33]. Levi called a closed convex set of distributions that represents a belief a credal set [56, 57], and Walley built a general theory of imprecise probability around such sets [80]. Robust bandits can be thought of as generalizing the notion of stochastic bandits using imprecise probabilities.

Partial specifications of the environment The idea that a hypothesis in the learner’s hypothesis class should be a partial specification of the environment has been considered in previous work. The most notable is the work on learning partial concept classes [1], where a hypothesis is allowed to output a ‘don’t know’ label. Closer to our work in spirit is the framework introduced by Pour, Mansouri, and Ben-David [69] (see also Kosoy [52]) where each hypothesis outputs a set of labels for each instance, and the learner’s job is to guess any label from that set. Or in other words, the regret of the learner is the diference in loss compared to the learner that knows the true hypothesis but has no idea what precise label from the set of labels prescribed by the hypothesis is the adversary going to pick. This is exactly what robust bandits model, but in the bandits setting.

Outcome feedback In our model the learner does not just see its reward. It sees the outcome $y _ { t } ,$ and the reward $r \left( x _ { t } , y _ { t } \right)$ is a known function of it. Similar feedback appears in partial monitoring, where the learner sees a signal that need not reveal its reward [8, 47], and in the framework of Foster et al. [30], where the learner receives an arbitrary observation along with its reward. However, these models define regret diferently. They compare the learner with the best action for the true environment, or with the best fixed action in hindsight, rather than with the worst-case value of the true hypothesis.

Smoothed analysis and smoothed adversaries Another way to restrict the adversary is to require it to pick a smooth distribution, i.e., one whose density with respect to a fixed base measure is at most $1 / \sigma .$ . This is the idea behind smoothed analysis [78], and in online learning it goes by the name of smoothed adversaries [70, 37]. The set of σ-smooth distributions is a credal set, so a smoothed adversary is a special case of our adversary. It turns out that this restriction makes online learning as easy as statistical learning [12], and, closer to our concerns, it makes computationally eficient learning possible given an optimization oracle for the hypothesis class [36]. There are two diferences from our setting though. First, in smoothed online learning the credal set is known to the learner, whereas we need to learn it. Second, the regret there is measured against the best hypothesis in hindsight, whereas we compete with the worst-case value of the true hypothesis, which is easier.

## 4 Warmup: A $T ^ { 2 / 3 }$ Learner

For an arm x and outcome y, consider the vector w $( x , y ) = ( 1 , x , y )$ . This vector lies in $\mathbb { R } ^ { 1 + d _ { X } + d _ { D } }$ and the constraints defining the feasible set of outcomes, $C ^ { \star } y + B ^ { \star } x + d ^ { \star } = 0$ , determine a linear subspace of this vector space. Define ${ \mathcal { L } } : = \ker \left( d ^ { \star } \quad B ^ { \star } \quad C ^ { \star } \right)$ . A distribution is feasible for arm x precisely when its expectation $m$ satisfies $m \in D$ and $w \left( x , m \right) \in \mathcal { L }$ . The learner’s task is to learn enough about $\mathcal { L }$ to get low regret. Of course, we may never learn all of $\mathcal { L } .$ For example, if the adversary keeps choosing distributions with expected values in a strictly smaller subspace, then we cannot distinguish that subspace from the true ${ \mathcal { L } } .$ But this is fine, since we do not need to learn directions that the adversary never uses.

The fact that we never get to observe the true conditional mean, but only a noisy observation of it, adds an extra complication. In a noiseless world, a new observation tells us a new direction of ${ \mathcal { L } } .$ . Thus we can simply maintain a linear subspace of $\mathbb { R } ^ { 1 + d _ { X } + d _ { D } }$ , and incrementally grow it every time we get new information. However, in the presence of noise, no observation can be used to add an entire direction with certainty. Instead, we encode our knowledge of $\mathcal { L }$ with an ellipsoid, and incrementally grow it as we get new information. The details are written below.

We first translate and rescale the balls so that $X = \{ x : \| x \| _ { 2 } \leq 1 \}$ and $D = \{ y : \| y \| _ { 2 } \leq 1 \}$ . We continue to write the reward as $\boldsymbol { r } \left( x , y \right) = a ^ { T } \boldsymbol { x } + b ^ { T } \boldsymbol { y }$ and the true subspace as $\mathcal { L }$ in these coordinates. The new $\left\| \boldsymbol { b } \right\| _ { 2 }$ equals $R _ { D } \| \boldsymbol { b } \| _ { 2 }$ in the original coordinates, and the non-tangency constant $S$ stays the same. The translation also adds a constant to the reward. This constant shifts every reward and $V ^ { \star }$ by the same amount, so it changes neither the regret nor any choice of the learner, and we drop it. Write $q : = 1 + d _ { X } + d _ { D }$ for the number of coordinates of w $( x , y )$ . For a positive-definite matrix $V .$ , write $\| w \| _ { V ^ { - 1 } } : = \sqrt { w ^ { T } V ^ { - 1 } w }$

Define $p _ { V } \left( x , y \right) : = \left\| w \left( x , y \right) \right\| _ { V ^ { - 1 } } ^ { 2 }$ , and $E _ { V } : = \left\{ w \in \mathbb { R } ^ { q } : \| w \| _ { V ^ { - 1 } } ^ { 2 } \leq 1 \right\}$

Here $E _ { V }$ is the ellipsoid we described above, which we incrementally expand as we get new information about ${ \mathcal { L } } .$ . The algorithm works in blocks of length $n$ (we will choose n later to get a small regret). At the beginning of each block, we choose an arm x and play the same arm throughout the block. The ellipsoid $E _ { V }$ stays fixed during these rounds.

To choose an arm, we use our current guess for the feasible expected values. For each $x ,$ this consists of the $m \in D$ with $( 1 , x , m ) \in E _ { V }$ . If some arm has no such $m .$ , we choose it to learn more about it. Otherwise, we choose the arm with the largest worst-case reward over its guessed feasible means. This is an example of the principle of “optimism in the face of uncertainty” because our set of feasible means is conservative, and therefore our arm values are optimistic.

At the end of a full block, we average its n observations to get ${ \overline { { y } } } .$ . We then check whether $( 1 , x , { \overline { { y } } } )$ lies in $E _ { V }$ . If it does not, we enlarge the ellipsoid to include it. Otherwise, we leave the ellipsoid unchanged.

For the time being, we assume that we can carry out the rule for choosing an arm exactly. We will consider the computational aspects of the algorithm later.

Theorem 4 (Informal). Algorithm 1 has expected regret $\begin{array} { r } { R _ { T } \leq \tilde { O } \left( n + \frac { T } { \sqrt { n } } \right) } \end{array}$ . In particular, choosing $n = \left\lceil T ^ { 2 / 3 } \right\rceil$ gives $R _ { T } \leq \tilde { O } \left( T ^ { 2 / 3 } \right)$

Proof sketch. The sketch relies on three lemmas, which we state here without proof. Their precise versions, with explicit constants, and their proofs are in Appendix $\mathrm { A } { : }$ see, respectively, Lemma 14, Lemma 12, and the proof of Theorem 15.

Lemma 5. With probability at least $1 - \delta$ , every ellipsoid $E _ { V }$ that Algorithm 1 maintains during the $T$ rounds satisfies dist $\begin{array} { r } { ( w , \mathcal { L } ) \leq \tilde { O } \left( \frac { 1 } { \sqrt { n } } \right) } \end{array}$ for all $w \in E _ { V }$ , where $\tilde { O }$ hides a factor polynomial in the dimensions and logarithmic in $T$ and $\frac { 1 } { \delta }$

Lemma 6. There is a constant $G ,$ depending only on $S$ and $\left\| \boldsymbol { b } \right\| _ { 2 }$ , such that for every $x \in X$ and $y \in D , r \left( x , y \right) \geq v ^ { \star } \left( x \right) - G$ dist $\left( w \left( x , y \right) , \mathcal { L } \right)$

Lemma 7. Algorithm 1 updates V at most $O \left( \log n \right)$ times.

```latex
Algorithm 1.
Input: Horizon T and block length $n \in \{ 1 , \ldots , T \}$
$V  n ^ { - 1 } I _ { q }$
for $j  1$ to $\lceil T / n \rceil$ do
Define $\hat { K } _ { V } ( x ) : = \{ y \in D : p _ { V } ( x , y ) \leq 1 \}$ for every $x \in X$
if there exists $x \in X$ with $\hat { K } _ { V } ( x ) = \varnothing$ then
Choose an arm $x \in X$ with $\hat { K } _ { V } ( x ) = \varnothing$
else
Define $\begin{array} { r } { H _ { V } ( x ) : = \operatorname* { m i n } _ { y \in \hat { K } _ { V } ( x ) } r ( x , y ) } \end{array}$ for every $x \in X$
Choose x ∈ argmax<sub>x</sub>′<sub>∈X</sub> $H _ { V } ( x ^ { \prime } )$
end
$n _ { j } $ min $\{ n , T - ( j - 1 ) n \}$
for $i \gets 1$ to $n _ { j }$ do
Play x and observe $y _ { i }$
end
$\begin{array} { r } { \overline { { y } }  \frac { 1 } { n _ { j } } \sum _ { i = 1 } ^ { n _ { j } } y _ { i } , \overline { { w } }  w ( x , \overline { { y } } ) } \end{array}$
if $n _ { j } = n$ and $p _ { V } ( x , \overline { { y } } ) > 1$ then
$\dot { V }  V + \overline { { w } } \overline { { w } } ^ { T }$
end
end
```

Lemma 6 is a straightforward consequence of the fact that r is linear. Lemmas 5 and 7 are due to the specific way we update V. Adding $\overline { { w } } \overline { { w } } ^ { T }$ to V has the efect that it expands $E _ { V }$ in the correct direction. This ensures that the distance between $E _ { V }$ and the true $\mathcal { L }$ grows only by the noise in w, which is about $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ . Also, one can show that each update increases the volume of $E _ { V }$ by a constant factor, which explains why we can have at most a logarithmic number of updates.

Now consider an uninformative block, meaning a full block in which we do not update V. The arm x that we play during this block is picked by maximizing $H _ { V }$ . Since the block is uninformative, its empirical mean outcome satisfies w $( x , \overline { { y } } ) \in E _ { V }$ . These two facts, together with Lemmas 5 and $6 ,$ imply that the average reward is at least $V ^ { \star }$ minus about $\scriptstyle { \frac { 1 } { \sqrt { n } } }$ . Thus the block’s regret (the best arm’s guaranteed reward $n V ^ { \star }$ minus the reward we actually achieved) is at most about ${ \frac { n } { \sqrt { n } } } = { \sqrt { n } }$

On the other hand, for an informative block, the empirical mean can lie outside the ellipsoid, so we only have the trivial regret bound of $O \left( n \right)$

Finally, by Lemma 7 there are at most $O ( \log n )$ informative and $O \left( { \frac { T } { n } } \right)$ uninformative blocks, so the total regret is about n log $n + { \sqrt { n } } { \frac { T } { n } }$

Ignoring logarithmic factors, this bound is minimized by choosing $n = \left\lceil T ^ { 2 / 3 } \right\rceil$ , giving an upper bound of $\tilde { O } \left( T ^ { 2 / 3 } \right)$ on the regret. □

The precise statement and proof are in Appendix A.

## 4.1 Computational tractability

In this section we describe a polytime implementation of Algorithm 1.

The matrix updates $V \gets V + \overline { { w w } } ^ { T }$ and the test $p _ { V } \left( x , { \overline { { y } } } \right) > 1$ can be computed in polynomial time. The main issue is choosing an arm, which involves solving the following two computational

problems:

1. Decide whether there exists an arm $x \in X$ with $\hat { K } _ { V } ( x ) = \varnothing$ , and if so, find such an arm.

2. If ${ \hat { K } } _ { V } ( x )$ is nonempty for every $x \in X$ , find an arm $x \in \mathrm { a r g m a x } _ { x ^ { \prime } \in X } H _ { V } ( x ^ { \prime } )$ , where $H _ { V } ( x ) =$ min $\mathsf { 1 } _ { y \in \hat { K } _ { V } ( x ) } r ( x , y )$

One dificulty is that this is a max-min problem, since the score $H _ { V }$ itself is a result of a minimization problem. However, the minimization problem is convex, and thus using convex duality, can be written as a maximization problem. We use the following lemma.

Lemma 8. Fix $x \in X$ and write $z = ( z _ { 0 } , z _ { X } , z _ { D } ) \in \mathbb { R } ^ { q }$

(a) $\hat { K } _ { V } ( x ) = \varnothing$ if and only if

$$
\operatorname* { m a x } _ { \| \boldsymbol { z } \| _ { 2 } \leq 1 } \left[ z _ { 0 } + z _ { X } ^ { T } \boldsymbol { x } - \sqrt { z ^ { T } V \boldsymbol { z } } - \| \boldsymbol { z } _ { D } \| _ { 2 } \right] > 0 .
$$

(b) If $\hat { K } _ { V } ( x ) \neq \varnothing$ , then

$$
H _ { V } ( x ) = a ^ { T } x + \operatorname* { s u p } _ { z \in \mathbb { R } ^ { q } } \left[ z _ { 0 } + z _ { X } ^ { T } x - \sqrt { z ^ { T } V z } - \| b + z _ { D } \| _ { 2 } \right] .
$$

Proof. Fix x. We start with (b). Note that ${ \hat { K } } _ { V } ( x )$ is an intersection of two ellipsoids. If it was just one ellipsoid, then the inner minimization problem would have an easy closed-form solution. To handle the intersection of two ellipsoids, we can use Lagrangian duality.

We first show how this works in general, and then apply it to our case. Let $A , B \subseteq \mathbb { R } ^ { q }$ be compact convex sets with a nonempty intersection, and suppose we want to minimize $g ^ { T } v$ over $v \in A \cap B$ . Introduce two copies of v: one vector $s \in A$ and another $u \in B$ , with the equality $s = u$ requiring them to represent the same point. Let $z \in \mathbb { R } ^ { q }$ be the Lagrangian multipliers for the constraints $s = u$ , and consider the objective function $L ( s , u , z ) : = g ^ { T } s + z ^ { T } ( s - u ) = ( g + z ) ^ { T } s - z ^ { T } u$ Whenever $s = u ,$ , the added term vanishes and $L ( s , u , z ) = g ^ { T } s$ . Thus, for every fixed z, dropping the equality and minimizing L over all $( s , u ) \in A \times B$ gives a lower bound on the original minimum. This minimization separates into one linear optimization over A and another over B. We then choose z to make the lower bound as large as possible. This is the dual problem:

$$
\operatorname* { s u p } _ { z \in \mathbb { R } ^ { q } } \operatorname* { m i n } _ { ( s , u ) \in A \times B } L ( s , u , z ) = \operatorname* { s u p } _ { z \in \mathbb { R } ^ { q } } \left[ \operatorname* { m i n } _ { s \in A } ( g + z ) ^ { T } s + \operatorname* { m i n } _ { u \in B } ( - z ) ^ { T } u \right] .
$$

We still need to show that the best lower bound equals the original minimum. First note that it’s clear that min<sub>v∈A∩B</sub> $\begin{array} { r } { g ^ { T } v = \operatorname* { m i n } _ { ( s , u ) \in A \times B } \operatorname* { s u p } _ { z \in \mathbb { R } ^ { q } } L ( s , u , z ) } \end{array}$ . Indeed, for fixed $s , u$ , the supremum of $L ( s , u , z )$ over z is $g ^ { T } s$ if $s = u$ and +∞ otherwise: when $s \neq u ,$ take $z = t ( s - u )$ and let $t \to \infty$ Next, since $A \times B$ is compact and convex and L is continuous and afine in each of $( s , u )$ and $z ,$ Sion’s minimax theorem [76] lets us exchange the minimum and supremum.

In our problem, take $A = S _ { x } : = \{ w ( x , y ) : y \in D \} , B = E _ { V }$ , and $g \ : = \ : ( 0 , 0 , b )$ , so that $g ^ { T } w ( x , y ) = b ^ { T } y$ . The equality $s = u$ above is now $u = w ( x , y ) = ( 1 , x , y ) \colon$ : the vector u in the ellipsoid must be the same point as the one specified by $y \in D$ . For fixed z, Cauchy–Schwarz gives the two separate minima explicitly:

$$
\operatorname* { m i n } _ { s \in S _ { x } } ( g + z ) ^ { T } s = z _ { 0 } + z _ { X } ^ { T } x + \operatorname* { m i n } _ { y \in D } ( b + z _ { D } ) ^ { T } y = z _ { 0 } + z _ { X } ^ { T } x - \| b + z _ { D } \| _ { 2 } ,
$$

$$
\operatorname* { m i n } _ { u \in E _ { V } } ( - z ) ^ { T } u = - { \sqrt { z ^ { T } V z } } .
$$

Substituting these formulas into the general identity proves (b). No strictly feasible point is needed, and the supremum need not be attained.

For (a), the set ${ \hat { K } } _ { V } ( x )$ is empty exactly when the compact convex sets $S _ { x }$ and $E _ { V }$ are disjoint. By the strict separating hyperplane theorem, this is equivalent to the existence of z with

$$
\operatorname* { m i n } _ { s \in S _ { x } } z ^ { T } s - \operatorname* { m a x } _ { u \in E _ { V } } z ^ { T } u = z _ { 0 } + z _ { X } ^ { T } x - \| z _ { D } \| _ { 2 } - \sqrt { z ^ { T } V z } > 0 .
$$

The expression is positively homogeneous in z, so we may restrict to $\| z \| _ { 2 } \leq 1$ . It is continuous, so its maximum over that ball is attained. This proves (a). □

Thus having converted the inner problem into a maximization<sup>2</sup> problem, the entire optimization becomes a maximization. However it is not convex, and it’s still unclear if it can be solved eficiently. Fortunately, it is a quadratically constrained quadratic program (QCQP) with some nice properties (see Appendix A.1 for details) and thus has a polynomial time algorithm. We use the following result of Bienstock [11, Theorem 1.3].

Theorem 9 (Bienstock [11]). Consider a maximization problem with a nonempty feasible set that has: a) a fixed number of quadratic constraints with at least one of them being a bounded ellipsoid, b) a quadratic objective function, and c) rational coeficients for both the objective function and the constraints. Then, there exists an algorithm that, for every $\varepsilon \in ( 0 , 1 )$ , returns rational assignments to the variables that violate each constraint by at most ε and achieve an objective value that is at least the optimum minus ε. Its running time is polynomial in the input size and log(1/ε).

Of course, Theorem 9 only gives us an approximate solution. This is fine because one can choose the ε as an appropriate function of $T$ such that the approximate solution is still good enough that it does not change the regret rate. Since the runtime depends logarithmically on $1 / \varepsilon$ , this only costs us an extra log T factor in the runtime.

## 5 A $\sqrt { T }$ learner

The previous section attempts to apply the principle of “optimism in the face of uncertainty,” but does not fully embrace it. Indeed, the arm for each block is picked optimistically, but each block still looks like a standard exploration block. The learner does not always need to wait for n steps before identifying that the arm it is playing is problematic. If the outcomes collected so far have an empirical mean y with $w ( x , { \overline { { y } } } )$ already far outside the current ellipsoid, then the block can be cut short more quickly. However, the current arm score $H _ { V } ( x )$ does not account for outcomes outside the ellipsoid. First, we design a score function that takes every outcome in D into account, not just those represented inside our chosen ellipsoid. The score function will be linked to the regret in a way that lets us decide when to move on to the next block.

We construct this score by replacing the hard constraint $p _ { V } \left( x , y \right) \leq 1$ in $H _ { V }$ with a penalty, so that we can minimize over all of D. For a parameter $\lambda > 0$ , which we will choose later to get a small regret, define $\begin{array} { r } { U _ { V , \lambda } \left( x \right) : = \operatorname* { m i n } _ { y \in D } \left[ r \left( x , y \right) + \lambda p _ { V } \left( x , y \right) \right] } \end{array}$ . We pick the arm x by maximizing $U _ { V , \lambda }$ . For now, assume that we can maximize the score exactly.

The following lemma gives us a way to control regret when the outcome y appears anywhere in D.

Lemma 10. Fix $\delta \in ( 0 , 1 )$ . With probability at least $1 - \delta$ , simultaneously for every matrix V that Algorithm 2 considers, every $\lambda > 0$ , and every arm x maximizing $\begin{array} { r } { U _ { V , \lambda } , \ V ^ { \star } - r ( x , y ) \ \le } \end{array}$ $\lambda p _ { V } ( x , y ) + \tilde { O } ( 1 / \lambda )$ for every $y \in D$ . Here O<sup>˜</sup> hides logarithmic factors in $T / \delta$

Proof. We first claim that, with probability at least $1 - \delta ,$ for every matrix V that Algorithm 2 considers, every arm $x ^ { \prime } \in X$ , and every outcome $y \in D , v ^ { \star } ( x ^ { \prime } ) - r ( x ^ { \prime } , y ) \leq c \sqrt { p _ { V } ( x ^ { \prime } , y ) }$ , where c does not depend on $x ^ { \prime } , y ,$ or $\lambda ;$ it is polynomial in the dimensions and logarithmic in $T / \delta$ , so $c = { \tilde { O } } ( 1 )$ . We prove this in Appendix B, as Equation 9.

Now fix any arm $x ^ { \prime }$ and outcome $y \in D$ . Rearranging the claim and adding the penalty $\lambda p _ { V } ( x ^ { \prime } , y )$ to both sides gives $r ( x ^ { \prime } , y ) + \lambda p _ { V } ( x ^ { \prime } , y ) \geq v ^ { \star } ( x ^ { \prime } ) + \lambda p _ { V } ( x ^ { \prime } , y ) - c \sqrt { p _ { V } ( x ^ { \prime } , y ) }$ . By the AM–GM inequality, $\begin{array} { r } { \lambda p + \frac { c ^ { 2 } } { 4 \lambda } \geq 2 \sqrt { \lambda p \cdot \frac { c ^ { 2 } } { 4 \lambda } } = c \sqrt { p } } \end{array}$ for every $p \geq 0$ , so the right-hand side is at least $\begin{array} { r } { v ^ { \star } ( x ^ { \prime } ) - \frac { c ^ { 2 } } { 4 \lambda } } \end{array}$ . Minimizing over y gives $\begin{array} { r } { U _ { V , \lambda } ( x ^ { \prime } ) \geq v ^ { \star } ( x ^ { \prime } ) - \frac { c ^ { 2 } } { 4 \lambda } } \end{array}$ , and since x maximizes the score, $\begin{array} { r } { U _ { V , \lambda } ( x ) \geq \dot { V } ^ { \star } - \frac { c ^ { 2 } } { 4 \lambda } } \end{array}$ . Finally, the definition of the score gives $U _ { V , \lambda } ( x ) \le r ( x , y ) + \lambda p _ { V } ( x , y )$ for every $y \in D$ . Combining the last two bounds, $\begin{array} { r } { V ^ { \star } - r ( x , y ) \le \lambda p _ { V } ( x , y ) + \frac { c ^ { 2 } } { 4 \lambda } } \end{array}$ , which is the claim with $\begin{array} { r } { \tilde { O } ( 1 / \lambda ) = \frac { c ^ { 2 } } { 4 \lambda } } \end{array}$ . Since c does not depend on λ, the argument holds for all $\lambda > 0$ on the same event.

To use this bound during a block, we keep V fixed and play the same arm x repeatedly, but recompute the empirical mean after every observation. Let $\overline { { y } } _ { n }$ be the mean of the first n outcomes in the block. Since the reward is linear, the reward accumulated so far is $n r ( x , { \overline { { y } } } _ { n } )$ . Applying Lemma 10 to this mean gives

$$
n V ^ { \star } - \sum _ { i = 1 } ^ { n } r ( x , y _ { i } ) \leq \lambda n p _ { V } ( x , \overline { { y } } _ { n } ) + \tilde { O } ( n / \lambda ) .
$$

The first term is observable. We can compute it after every observation, and we end the block and update V the first time it reaches λ, or equivalently, when $n p _ { V } ( x , \overline { { y } } _ { n } ) \geq 1$ . Before this happens, the block’s regret is bounded by $\lambda + \tilde { O } ( n / \lambda )$

A large empirical penalty can therefore make us update after only a few observations. A small one lets us keep playing the arm for longer. Thus we choose when to update V by monitoring this bound throughout the block, rather than waiting for a fixed number of rounds.

Theorem 11 (Informal). Algorithm 2, with $\lambda = \lceil { \sqrt { T } } \rceil$ , has expected regret $R _ { T } \leq \tilde { O } \left( \sqrt { T } \right)$

Proof sketch. As before, each completed block at least doubles the determinant of V , so there are only ${ \cal O } ( \log T )$ blocks.

By Lemma 10, with high probability, a block of length n has regret at most $\lambda n p _ { V } ( x , \overline { { y } } ) + \tilde { O } ( n / \lambda )$ The stopping rule keeps $n p _ { V } ( x , { \overline { { y } } } )$ bounded by a constant, so the first term costs $O ( \lambda )$ per block, regardless of its length. The remaining term costs about $1 / \lambda$ per round. Summing both costs gives regret at most about λ log $T + T / \lambda$ . Ignoring logarithmic factors, choosing $\lambda = \lceil { \sqrt { T } } \rceil$ balances the two terms. □

To run the algorithm in polynomial time we use an almost identical approach as the previous section. The main diferences are that there is no emptiness test any more and the inner minimization problem has an extra quadratic term. But the same techniques can be applied to convert the entire optimization problem into a maximization problem and the Bienstock’s result can be used to solve it in polytime. The precise regret bound and detailed proofs of the algorithm are in Appendix B.

Algorithm 2.   
Input: Horizon T and penalty parameter $\lambda > 0 .$   
$V  I _ { q } , \quad t  0$   
while $t < T$ do   
Define $\begin{array} { r } { U _ { V , \lambda } ( x ) : = \operatorname* { m i n } _ { y \in D } [ r ( x , y ) + \lambda p _ { V } ( x , y ) ] } \end{array}$ for every $x \in X$   
Choose x ∈ argmax<sub>x</sub>′<sub>∈X</sub> U<sub>V,λ</sub>(x<sup>′</sup>)   
$n \gets 0$   
repeat   
Play x and observe $y _ { n + 1 }$   
$n  n + 1 , ~ t  t + 1$   
$\textstyle { \overline { { y } } } _ { n } \gets { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } y _ { i }$   
until $n p _ { V } ( x , \overline { { y } } _ { n } ) \geq 1 ~ o r ~ t = T$   
if $n p _ { V } ( x , \overline { { y } } _ { n } ) \geq 1$ then   
${ \overline { { w } } } _ { n } \gets w ( x , { \overline { { y } } } _ { n } )$   
$V \gets V + n \overline { { w } } _ { n } \overline { { w } } _ { n } ^ { T }$   
end   
end

## 6 NP-hardness

The precise statements and proofs of the results below are in Appendix C.

What other instances of robust bandits have eficient, sublinear regret learners? Interestingly, the linear instance studied in the previous sections appears to be at the boundary of what’s tractable. Generalizing it in natural ways quickly gives us computationally intractable problems.

First, we show that if D is a simplex instead of a Euclidean ball, the problem is computationally hard (Appendix C.2). If D remains a ball, but we change X to be a simplex, then the problem is essentially the same as having $d _ { X } + 1$ arms, one for each vertex of X, since a non-vertex arm is never better than randomizing over the vertices. With finitely many arms, robust bandits reduce to adversarial bandits [50], so running Exp3 [5] on the vertices gives $\tilde { O } ( \sqrt { d _ { X } T } )$ regret in polynomial time. However, if X is a convex polytope with k faces, then the problem once again becomes computationally hard (Appendix C.3). Next, we analyze other kinds of hypothesis classes H than the ones defined by the linear constraints of Section 2.1. In Kosoy [51], it was assumed that for a given hypothesis z and a given arm x, the set of feasible expected outcomes was given by constraints $F ( x , y , z ) = 0$ where F was bilinear in y and z. We ask what happens if we impose trilinearity on F while keeping X and D as Euclidean balls. We show once again, that this small generalization leads to a computationally hard problem (Appendix C.4).

All the above hardness proofs rely on proving that the planning problem is already computationally hard. That is, given the true hypothesis, finding an arm that maximizes $v ^ { * } ( x )$ is hard. This, along with the fact that planning can be reduced to learning (Appendix C.1) shows that learning is also hard. This leads to a natural question: if we were given a polynomial time planning oracle, would learning become easy? We answer this in the negative by showing an example where planning is easy but learning is computationally hard (Appendix C.5).

## 7 Discussion and open problems

Robust bandits give a new way of dealing with unrealizability. It will be interesting to see if similar definitions can be made for other frameworks of learning, and if computationally eficient learners can be designed for them.

For robust bandits themselves, it will be nice to see which other hypothesis classes admit computationally eficient learners. Our hardness results show that small changes to the linear instance already make planning hard, so such classes will need some structure that keeps planning easy, and the example in Appendix C.5 shows that easy planning alone is not enough. It would be nice to understand computational eficiency in the presence of certain oracles. For example, is there a set of natural oracles, in addition to the planning oracle, whose existence guarantees a computationally eficient learner?

Appel and Kosoy [3] generalize robust bandits to online decision making with structured observations, which includes robust linear bandits and tabular reinforcement learning, and prove regret bounds in this setting, but they do not address computational eficiency. It will be nice to see if the techniques of this paper could be used to design eficient learners for their setting.

## Acknowledgments

This work was supported by the Advanced Research+Invention Agency (ARIA) of the United Kingdom and Coeficient Giving in San Francisco, California.

## AI Disclosure

This work grew out of a few months of research involving many brainstorming sessions with GPT 5.6 and later GPT 6. The NP-hardness results and the ${ \tilde { O } } ( T ^ { 2 / 3 } )$ upper bound were proved mostly by the authors, with some help from GPT in finding relevant literature. Given the accumulated notes and observations as context, GPT 6 Astra was able to improve the upper bound to $\tilde { O } ( \sqrt { T } )$ .

The writing also involved help from GPT 6, Fable 5.1, and Opus 5.5. The main body is largely human-written. The appendix proofs were written over many iterations with AI. We supplied the models with notes containing proofs and detailed writing guidelines, which they used to draft the appendix. Over several iterations of feedback, we refined the guidelines and revised the drafts.

## References

[1] Noga Alon, Steve Hanneke, Ron Holzman, and Shay Moran. A theory of PAC learnability of partial concept classes. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pages 658–671, 2022. doi: 10.1109/FOCS52979.2021.00070.

[2] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. arXiv preprint arXiv:1606.06565, 2016.

[3] Alexander Appel and Vanessa Kosoy. Regret bounds for robust online decision making. In Proceedings of the 38th Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 64–146, 2025. URL https://proceedings.mlr.press/ v291/appel25a.html.

[4] Stuart Armstrong and Sören Mindermann. Occam’s razor is insuficient to infer the preferences of irrational agents. In Advances in Neural Information Processing Systems, volume 31, 2018. URL https://arxiv.org/abs/1712.05812.

[5] Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002. doi: 10.1137/ S0097539701398375.

[6] Yuntao Bai et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

[7] Mihály Bárász, Paul Christiano, Benja Fallenstein, Marcello Herreshof, Patrick LaVictoire, and Eliezer Yudkowsky. Robust cooperation in the Prisoner’s Dilemma: Program equilibrium via provability logic. arXiv preprint arXiv:1401.5577, 2014.

[8] Gábor Bartók, Dean P. Foster, Dávid Pál, Alexander Rakhlin, and Csaba Szepesvári. Partial monitoring—classification, regret bounds, and algorithms. Mathematics of Operations Research, 39(4):967–997, 2014. doi: 10.1287/moor.2014.0663.

[9] Shai Ben-David, Dávid Pál, and Shai Shalev-Shwartz. Agnostic online learning. In Proceedings of the 22nd Conference on Learning Theory, 2009. URL https://home.ttic.edu/\~shai/ papers/Ben-DavidPalShalev09.pdf.

[10] Yoshua Bengio et al. Managing extreme AI risks amid rapid progress. Science, 384(6698): 842–845, 2024. doi: 10.1126/science.adn0117.

[11] Daniel Bienstock. A note on polynomial solvability of the CDT problem. SIAM Journal on Optimization, 26(1):488–498, 2016. doi: 10.1137/15M1009871.

[12] Adam Block, Yuval Dagan, Noah Golowich, and Alexander Rakhlin. Smoothed online learning is as easy as statistical learning. In Proceedings of the 35th Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 1716–1786, 2022. URL https://proceedings.mlr.press/v178/block22a.html.

[13] Nick Bostrom. Superintelligence: Paths, Dangers, Strategies. Oxford University Press, 2014.

[14] Sébastien Bubeck, Rémi Munos, and Gilles Stoltz. Pure exploration in finitely-armed and continuous-armed bandits. Theoretical Computer Science, 412(19):1832–1852, 2011. doi: 10.1016/j.tcs.2010.12.059.

[15] Sébastien Bubeck, Rémi Munos, Gilles Stoltz, and Csaba Szepesvári. X-armed bandits. Journal of Machine Learning Research, 12(46):1655–1695, 2011. URL https://www.jmlr.org/papers/ v12/bubeck11a.html.

[16] Stephen Casper et al. Open problems and fundamental limitations of reinforcement learning from human feedback. Transactions on Machine Learning Research, 2023. URL https: //arxiv.org/abs/2307.15217.

[17] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://arxiv.org/abs/1706.03741.

[18] Andrew Critch. A parametric, resource-bounded generalization of Löb’s theorem, and a robust cooperation criterion for open-source game theory. The Journal of Symbolic Logic, 84(4): 1368–1381, 2019. doi: 10.1017/jsl.2017.42.

[19] David A. Dalrymple. Imprecise beliefs: A tiny introduction. AI Alignment Forum, July 2026. URL https://www.alignmentforum.org/posts/e7Pd4Q9TF7jFdmPgz/ imprecise-beliefs-a-tiny-introduction. Published as davidad.

[20] Varsha Dani, Thomas P. Hayes, and Sham M. Kakade. Stochastic linear optimization under bandit feedback. In Proceedings of the 21st Conference on Learning Theory, pages 355–366, 2008.

[21] Amit Daniely. Complexity theoretic limitations on learning halfspaces. In Proceedings of the 48th Annual ACM Symposium on Theory of Computing, pages 105–117, 2016. doi: 10.1145/ 2897518.2897520.

[22] Amit Daniely and Shai Shalev-Shwartz. Complexity theoretic limitations on learning DNF’s. In Proceedings of the 29th Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 815–830, 2016. URL https://proceedings.mlr.press/ v49/daniely16.html.

[23] Abram Demski and Scott Garrabrant. Embedded agency. arXiv preprint arXiv:1902.09469, 2019.

[24] Daniel Dewey. Learning what to value. In Artificial General Intelligence, volume 6830 of Lecture Notes in Computer Science, pages 309–314, Berlin, 2011. Springer. doi: 10.1007/ 978-3-642-22887-2\_35.

[25] Daniel Ellsberg. Risk, ambiguity, and the Savage axioms. The Quarterly Journal of Economics, 75(4):643–669, 1961. doi: 10.2307/1884324.

[26] Tom Everitt, Gary Lea, and Marcus Hutter. AGI safety literature review. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, pages 5441–5449, 2018. doi: 10.24963/ijcai.2018/768.

[27] Rémi Eyraud, Jefrey Heinz, and Ryo Yoshinaka. Eficiency in the identification in the limit learning paradigm. In Jefrey Heinz and José M. Sempere, editors, Topics in Grammatical Inference, pages 25–46. Springer, Berlin, Heidelberg, 2016. doi: 10.1007/978-3-662-48395-4\_2.

[28] Benja Fallenstein, Jessica Taylor, and Paul F. Christiano. Reflective oracles: A foundation for classical game theory. arXiv preprint arXiv:1508.04145, 2015.

[29] Dean Foster, Satyen Kale, and Howard Karlof. Online sparse linear regression. In Proceedings of the 29th Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 960–970, 2016. URL https://proceedings.mlr.press/v49/foster16.html.

[30] Dylan J. Foster, Sham M. Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.

[31] Michael R. Garey and David S. Johnson. Computers and Intractability: A Guide to the Theory of NP-Completeness. W. H. Freeman, San Francisco, 1979.

[32] Michael R. Garey, David S. Johnson, and Larry Stockmeyer. Some simplified NP-complete graph problems. Theoretical Computer Science, 1(3):237–267, 1976. doi: 10.1016/0304-3975(76) 90059-1.

[33] Itzhak Gilboa and David Schmeidler. Maxmin expected utility with non-unique prior. Journal of Mathematical Economics, 18(2):141–153, 1989. doi: 10.1016/0304-4068(89)90018-9.

[34] Dylan Hadfield-Menell, Anca Dragan, Pieter Abbeel, and Stuart Russell. Cooperative inverse reinforcement learning. In Advances in Neural Information Processing Systems, volume 29, pages 3909–3917, 2016. URL https://arxiv.org/abs/1606.03137.

[35] Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart Russell, and Anca Dragan. Inverse reward design. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://arxiv.org/abs/1711.02827.

[36] Nika Haghtalab, Yanjun Han, Abhishek Shetty, and Kunhe Yang. Oracle-eficient online learning for smoothed adversaries. In Advances in Neural Information Processing Systems, volume 35, 2022.

[37] Nika Haghtalab, Tim Roughgarden, and Abhishek Shetty. Smoothed analysis with adaptive adversaries. Journal of the ACM, 71(3):19:1–19:34, 2024. doi: 10.1145/3656638.

[38] Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse, and Scott Garrabrant. Risks from learned optimization in advanced machine learning systems. arXiv preprint arXiv:1906.01820, 2019.

[39] Marcus Hutter. Open problems in universal induction & intelligence. Algorithms, 2(3):879–906, 2009. doi: 10.3390/a2030879.

[40] Edwin T. Jaynes. Probability Theory: The Logic of Science. Cambridge University Press, 2003. Edited by G. Larry Bretthorst.

[41] Hong Jun Jeon, Smitha Milli, and Anca Dragan. Reward-rational (implicit) choice: A unifying formalism for reward learning. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://arxiv.org/abs/2002.04833.

[42] Jiaming Ji et al. AI alignment: A contemporary survey. ACM Computing Surveys, 58(5): 132:1–132:38, 2026. doi: 10.1145/3770749.

[43] Zeyu Jia, Gene Li, Alexander Rakhlin, Ayush Sekhari, and Nathan Srebro. When is agnostic reinforcement learning statistically tractable? In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://arxiv.org/abs/2310.06113.

[44] Ehud Kalai and Ehud Lehrer. Rational learning leads to Nash equilibrium. Econometrica, 61 (5):1019–1045, 1993. doi: 10.2307/2951492.

[45] Richard M. Karp. Reducibility among combinatorial problems. In Raymond E. Miller, James W. Thatcher, and Jean D. Bohlinger, editors, Complexity of Computer Computations, pages 85–103. Plenum Press, New York, 1972. doi: 10.1007/978-1-4684-2001-2\_9.

[46] Michael J. Kearns, Robert E. Schapire, and Linda M. Sellie. Toward eficient agnostic learning. Machine Learning, 17(2–3):115–141, 1994. doi: 10.1007/BF00993468.

[47] Johannes Kirschner, Tor Lattimore, and Andreas Krause. Information directed sampling for linear partial monitoring. In Proceedings of the 33rd Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 2328–2369, 2020. URL https://proceedings.mlr.press/v125/kirschner20a.html.

[48] Vanessa Kosoy. The learning-theoretic AI alignment research agenda. AI Alignment Forum, July 2018. URL https://www.alignmentforum.org/posts/5bd75cc58225bf0670375575/ the-learning-theoretic-ai-alignment-research-agenda.

[49] Vanessa Kosoy. The learning-theoretic agenda: Status 2023. AI Alignment Forum, April 2023. URL https://www.alignmentforum.org/posts/ZwshvqiqCvXPsZEct/ the-learning-theoretic-agenda-status-2023.

[50] Vanessa Kosoy. Imprecise multi-armed bandits. M.sc. thesis, The Rachel and Selim Benin School of Computer Science, The Hebrew University of Jerusalem, 2024. URL https://arxiv. org/abs/2405.05673v1.

[51] Vanessa Kosoy. Imprecise multi-armed bandits: Representing irreducible uncertainty as a zero-sum game. Journal of Machine Learning Research, 26(184):1–75, 2025. URL https: //www.jmlr.org/papers/v26/24-2001.html.

[52] Vanessa Kosoy. Ambiguous online learning. In Proceedings of the 39th Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 4229–4266, 2026. URL https://proceedings.mlr.press/v336/kosoy26a.html.

[53] Akshay Krishnamurthy, Alekh Agarwal, and John Langford. PAC reinforcement learning with rich observations. In Advances in Neural Information Processing Systems, volume 29, 2016. URL https://proceedings.neurips.cc/paper/2016/hash/ 2387337ba1e0b0249ba90f55b2ba2521-Abstract.html.

[54] Jan Leike, Jessica Taylor, and Benya Fallenstein. A formal solution to the grain of truth problem. In Proceedings of the 32nd Conference on Uncertainty in Artificial Intelligence, pages 427–436, 2016. URL https://arxiv.org/abs/1609.05058.

[55] Jan Leike, David Krueger, Tom Everitt, Miljan Martic, Vishal Maini, and Shane Legg. Scalable agent alignment via reward modeling: A research direction. arXiv preprint arXiv:1811.07871, 2018.

[56] Isaac Levi. On indeterminate probabilities. The Journal of Philosophy, 71(13):391–418, 1974. doi: 10.2307/2025161.

[57] Isaac Levi. The Enterprise of Knowledge: An Essay on Knowledge, Credal Probability, and Chance. MIT Press, Cambridge, MA, 1980.

[58] Gene X. Li. Agnostic Reinforcement Learning: Foundations and Algorithms. PhD thesis, Toyota Technological Institute at Chicago, Chicago, IL, 2025. URL https://arxiv.org/abs/2506. 01884.

[59] Nick Littlestone. Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm. Machine Learning, 2(4):285–318, 1988. doi: 10.1007/BF00116827.

[60] Smitha Milli, Dylan Hadfield-Menell, Anca Dragan, and Stuart Russell. Should robots be obedient? In Proceedings of the 26th International Joint Conference on Artificial Intelligence, pages 4754–4760, 2017. doi: 10.24963/ijcai.2017/662.

[61] Theodore S. Motzkin and Ernst G. Straus. Maxima for graphs and a new proof of a theorem of Turán. Canadian Journal of Mathematics, 17:533–540, 1965. doi: 10.4153/CJM-1965-053-6.

[62] John H. Nachbar. Prediction, optimization, and learning in repeated games. Econometrica, 65 (2):275–309, 1997. doi: 10.2307/2171894.

[63] John H. Nachbar. Beliefs in repeated games. Econometrica, 73(2):459–480, 2005. doi: 10.1111/ j.1468-0262.2005.00585.x.

[64] Andrew Y. Ng and Stuart Russell. Algorithms for inverse reinforcement learning. In Proceedings of the 17th International Conference on Machine Learning, pages 663–670, 2000.

[65] Richard Ngo, Lawrence Chan, and Sören Mindermann. The alignment problem from a deep learning perspective. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2209.00626v6.

[66] Yuichi Noguchi. Bayesian learning, smooth approximate optimal behavior, and convergence to ε-Nash equilibrium. Econometrica, 83(1):353–373, 2015. doi: 10.3982/ECTA9332.

[67] Caspar Oesterheld. Robust program equilibrium. Theory and Decision, 86(1):143–159, 2019. doi: 10.1007/s11238-018-9679-3.

[68] Long Ouyang et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://arxiv. org/abs/2203.02155.

[69] Alireza F. Pour, Farnam Mansouri, and Shai Ben-David. Learning with multiple correct answers – regret bounds under diferent feedback models. arXiv preprint arXiv:2602.09402, 2026.

[70] Alexander Rakhlin, Karthik Sridharan, and Ambuj Tewari. Online learning: Stochastic, constrained, and smoothed adversaries. In Advances in Neural Information Processing Systems, volume 24, 2011.

[71] Stuart Russell. Human Compatible: Artificial Intelligence and the Problem of Control. Viking, New York, 2019.

[72] Stuart Russell, Daniel Dewey, and Max Tegmark. Research priorities for robust and beneficial artificial intelligence. AI Magazine, 36(4):105–114, 2015. doi: 10.1609/aimag.v36i4.2577.

[73] Leonard J. Savage. The Foundations of Statistics. Wiley, New York, 1954.

[74] Ayush Sekhari, Christoph Dann, Mehryar Mohri, Yishay Mansour, and Karthik Sridharan. Agnostic reinforcement learning with low-rank MDPs and rich observations. In Advances in Neural Information Processing Systems, volume 34, pages 19033–19045, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 9eed867b73ab1eab60583c9d4a789b1b-Abstract.html.

[75] Rohin Shah, Dmitrii Krasheninnikov, Jordan Alexander, Pieter Abbeel, and Anca Dragan. Preferences implicit in the state of the world. In International Conference on Learning Representations, 2019. URL https://arxiv.org/abs/1902.04198.

[76] Maurice Sion. On general minimax theorems. Pacific Journal of Mathematics, 8(1):171–176, 1958. doi: 10.2140/pjm.1958.8.171.

[77] Nate Soares. The value learning problem. In Ethics for Artificial Intelligence Workshop at IJCAI-16, 2016. URL https://intelligence.org/files/ValueLearningProblem.pdf.

[78] Daniel A. Spielman and Shang-Hua Teng. Smoothed analysis of algorithms: Why the simplex algorithm usually takes polynomial time. Journal of the ACM, 51(3):385–463, 2004. doi: 10.1145/990308.990310.

[79] Moshe Tennenholtz. Program equilibrium. Games and Economic Behavior, 49(2):363–373, 2004. doi: 10.1016/j.geb.2004.02.002.

[80] Peter Walley. Statistical Reasoning with Imprecise Probabilities. Chapman and Hall, London, 1991.

[81] Cole Wyeth, Marcus Hutter, Jan Leike, and Jessica Taylor. Limit-computable grains of truth for arbitrary computable extensive-form (un)known games. arXiv preprint arXiv:2508.16245, 2025.

## A Regret analysis of the $T ^ { 2 / 3 }$ learner

We give the precise statement and proof of Theorem 4. We use the coordinates and notation from Section 4, and assume that Algorithm 1 carries out its rule for choosing an arm exactly. A full block is informative if it causes an update, and uninformative otherwise.

For the analysis, fix $\delta \in ( 0 , 1 )$ and define

$$
\begin{array} { c } { { J _ { \operatorname* { m a x } } : = \left\lceil q \left( 4 + 2 \log _ { 2 } n \right) \right\rceil , } } \\ { { \displaystyle h _ { T } ^ { 2 } : = 8 d _ { D } \log \left( \frac { 2 d _ { D } T ^ { 2 } } { \delta } \right) , } } \\ { { \displaystyle B _ { T } ^ { 2 } : = 1 + J _ { \operatorname* { m a x } } h _ { T } ^ { 2 } } , } \\ { { \displaystyle G : = \sqrt { 3 } \left( 1 + S ^ { - 1 } \right) \| b \| _ { 2 } } . } \end{array}
$$

We write J for the number of updates actually made. We will show that $J \le J _ { \mathrm { m a x } }$

If w $( x , y )$ is close to $\mathcal { L } .$ , then $r \left( x , y \right)$ cannot be much smaller than the worst feasible reward $v ^ { \star } \left( x \right)$ . We first prove this.

Lemma 12. For every $x \in X$ and $y \in D$

$$
v ^ { \star } \left( x \right) \leq r \left( x , y \right) + G \operatorname { d i s t } \left( w \left( x , y \right) , \mathcal { L } \right) .
$$

Proof. Let $N : = \ker C ^ { \star }$ and let $Q$ be the orthogonal projector onto $N ^ { \perp }$ . Every arm has a feasible outcome, so we can write the minimum-norm solution of the afine constraints as $f \left( x \right) = A x + e \in N ^ { \bot }$ Thus

$$
K ^ { \star } \left( x \right) = \left( A x + e + N \right) \cap D .
$$

Since there is a feasible outcome in the unit ball for every arm, $\| A x + e \| _ { 2 } \leq 1$ on $X$ . Taking $x = 0$ gives $\| e \| _ { 2 } \leq 1$ , and comparing opposite unit vectors gives $\| A \| _ { \mathrm { o p } } \leq 1$ . Now define

$$
\Phi \left( s , x , y \right) : = Q y - A x - s e .
$$

This map has operator norm at most $\sqrt { 3 }$ and kernel ${ \mathcal { L } } .$ Indeed, the original constraints are equivalent to $Q y = A x + e $ , and homogenizing either equation gives the same subspace. Therefore

$$
\operatorname { d i s t } \left( y , A x + e + N \right) = \left\| \Phi \left( 1 , x , y \right) \right\| _ { 2 } \leq { \sqrt { 3 } } \operatorname { d i s t } \left( w \left( x , y \right) , { \mathcal { L } } \right) .
$$

Let $p$ be the point in $A x + e + N$ closest to y. Since $y \in D$ , the transversality condition gives

$$
\mathrm { d i s t } \left( p , K ^ { \star } \left( x \right) \right) \leq S ^ { - 1 } \mathrm { d i s t } \left( p , D \right) \leq S ^ { - 1 } \| p - y \| _ { 2 } ,
$$

where the first inequality is immediate if $p \in D$ , since then $p \in K ^ { \star } \left( x \right)$ . The triangle inequality now gives

$$
\operatorname { d i s t } \left( y , K ^ { \star } \left( x \right) \right) \leq \left( 1 + S ^ { - 1 } \right) \operatorname { d i s t } \left( y , A x + e + N \right) .
$$

Moving y to the nearest point in $K ^ { \star } \left( x \right)$ changes the reward by at most $\left\| \boldsymbol { b } \right\| _ { 2 }$ times this distance, which proves the claim. □

We also need to know how close a block average is to the average of the true conditional means.

Lemma 13. Let $m _ { t }$ be the conditional mean of y<sub>t</sub> given the history and the chosen arm on round t. For each interval in the first T rounds, write n for its length and $\overline { y }$ and m for the averages of y<sub>t</sub> and $m _ { t }$ over that interval. With probability at least $1 - \delta$ , the bound

$$
n \| \overline { { y } } - \overline { { m } } \| _ { 2 } ^ { 2 } \leq h _ { T } ^ { 2 }
$$

holds simultaneously on all these intervals.

Proof. Each coordinate of $y _ { t } - m _ { t }$ is a martingale diference bounded in absolute value by two. For any fixed interval of length n and coordinate $i ,$ Azuma–Hoefding gives

$$
\operatorname* { P r } \left( \lvert \overline { { y } } _ { i } - \overline { { m } } _ { i } \rvert > \frac { h _ { T } } { \sqrt { n d _ { D } } } \right) \leq 2 \exp \left( - \frac { h _ { T } ^ { 2 } } { 8 d _ { D } } \right) = \frac { \delta } { d _ { D } T ^ { 2 } } .
$$

There are $d _ { D }$ coordinates and at most $T ^ { 2 }$ intervals. A union bound and summing the squared coordinate bounds give the claim. □

We need to know how far a point in $E _ { V }$ can be from L. The next lemma bounds this distance using the noise bound we just proved. We can then apply Lemma 12 to the outcomes the learner treats as feasible.

Lemma 14. With probability at least $1 - \delta$ , after any number J of updates in Algorithm 1, the matrix V satisfies, for every $w \in \mathbb { R } ^ { q }$ 2

$$
\mathrm { d i s t } \left( w , \mathcal { L } \right) \leq \sqrt { \frac { 1 + J h _ { T } ^ { 2 } } { n } } \| w \| _ { V ^ { - 1 } } .
$$

Proof. Number the informative blocks $j = 1 , \dots , J$ . Write $x _ { j }$ for the arm played in block j, and $\overline { { y } } _ { j }$ and $\overline { { m } } _ { j }$ for its average observation and average conditional mean. With $\overline { { w } } _ { j } : = w \left( x _ { j } , \overline { { y } } _ { j } \right)$ , the update rule gives

$$
V = n ^ { - 1 } I _ { q } + \sum _ { j = 1 } ^ { J } \overline { { w } } _ { j } \overline { { w } } _ { j } ^ { T } .
$$

Fix $u \in \mathcal { L } ^ { \perp }$ with $\left. u \right. _ { 2 } \leq 1$ . Since the arm is held fixed within each block, $w \left( x _ { j } , \overline { { m } } _ { j } \right) \in \mathcal { L }$ . Each informative block has length n. By Lemma 13, with probability at least $1 - \delta ,$ , the bound

$$
\left( u ^ { T } \overline { { w } } _ { j } \right) ^ { 2 } = \left( u ^ { T } \left( 0 , 0 , \overline { { y } } _ { j } - \overline { { m } } _ { j } \right) \right) ^ { 2 } \leq \frac { h _ { T } ^ { 2 } } { n }
$$

holds simultaneously for all informative blocks and all such u. On this event, $\begin{array} { r } { u ^ { T } V u \leq \frac { 1 + J h _ { T } ^ { 2 } } { n } } \end{array}$ For any $w ,$ Cauchy–Schwarz gives $u ^ { T } w \leq \sqrt { u ^ { T } V u } \| w \| _ { V ^ { - 1 } }$ . Taking the supremum over the unit ball of $\mathcal { L } ^ { \perp }$ proves the claim. □

Theorem 15. Algorithm 1 satisfies, with probability at least $1 - \delta$

$$
T V ^ { \star } - \sum _ { t = 1 } ^ { T } r \left( x _ { t } , y _ { t } \right) \leq C _ { r } n \left( J _ { \operatorname* { m a x } } + 1 \right) + \frac { G B _ { T } T } { \sqrt { n } } .
$$

Its expected regret is at most the same expression plus $C _ { r } T \delta$ . Taking $n = \left\lceil T ^ { 2 / 3 } \right\rceil$ and $\delta =$ $( T + 1 ) ^ { - 2 }$ give $R _ { T } \leq \tilde { O } \left( T ^ { 2 / 3 } \right)$ for fixed problem parameters.

Proof. First, there cannot be very many informative blocks. Each one more than doubles the determinant:

$$
\operatorname* { d e t } \left( V + { \overline { { w w } } } ^ { T } \right) = \operatorname* { d e t } \left( V \right) ( 1 + p _ { V } \left( x , { \overline { { y } } } \right) ) > 2 \operatorname* { d e t } \left( V \right) .
$$

Each update adds at most three to the trace, since $\| \overline { { w } } \| _ { 2 } ^ { 2 } \leq 3 .$ , so after J updates tr $V \leq \frac { q } { n } + 3 J$ The initial determinant is $n ^ { - q }$ . Using the arithmetic–geometric mean inequality on the eigenvalues, we get

$$
2 ^ { J } \leq { \frac { \operatorname* { d e t } V } { \operatorname* { d e t } \left( n ^ { - 1 } I _ { q } \right) } } \leq \left( 1 + 3 { \frac { n J } { q } } \right) ^ { q } .
$$

Write $\begin{array} { r } { \xi : = \frac { J } { q } . } \end{array}$ , so that $\xi \leq \log _ { 2 } \left( 1 + 3 n \xi \right)$ . If $\xi < 4$ , then $\xi \leq 4 + 2 \log _ { 2 } n$ holds trivially. If $\xi \ge 4$ then $\begin{array} { r } { \log _ { 2 } { ( 1 + 3 n \xi ) } \leq \log _ { 2 } { ( 4 n \xi ) } = 2 + \log _ { 2 } { n } + \log _ { 2 } { \xi } \leq 2 + \log _ { 2 } { n } + \frac { \xi } { 2 } } \end{array}$ , using $\begin{array} { r } { \log _ { 2 } \xi \le \frac { \xi } { 2 } } \end{array}$ for $\xi \ge 4$ and hence again $\xi \leq 4 + 2 \log _ { 2 } n$ . In either case $J \le q ( 4 + 2 \log _ { 2 } n ) \le \bar { J } _ { \mathrm { m a x } }$ . Note that the horizon $T$ does not enter. The count is the logarithm of the ratio between the final volume of $E _ { V }$ , which is bounded in terms of J alone, and its initial volume, which is set by the block length n.

Since $J \le J _ { \mathrm { m a x } }$ , Lemma 14 gives, with probability at least $1 - \delta$ , simultaneously for every matrix V that the learner maintains and every $( x , y ) \in X \times D$ 2

$$
\mathrm { d i s t } \left( w \left( x , y \right) , \mathcal { L } \right) \leq \frac { B _ { T } } { \sqrt { n } } \cdot \sqrt { p _ { V } \left( x , y \right) } .\tag{2}
$$

On this event, if $y \in \hat { K } _ { V } \left( x \right)$ , then $p _ { V } \left( x , y \right) \leq 1$ , so Lemma 12 gives $\begin{array} { r } { r \left( x , y \right) \geq v ^ { \star } \left( x \right) - G \frac { B _ { T } } { \sqrt { n } } } \end{array}$ Taking the minimum over ${ \hat { K } } _ { V } \left( x \right)$ , whenever this set is nonempty, gives

$$
H _ { V } \left( x \right) \geq v ^ { \star } \left( x \right) - G \frac { B _ { T } } { \sqrt { n } } .
$$

Consider a full block in which we do not update the ellipsoid. Its empirical mean belongs to ${ \hat { K } } _ { V } \left( x \right)$ . Such a block could not have been played at an arm x with ${ \hat { K } } _ { V } \left( x \right)$ empty, since then its mean would lie outside the ellipsoid. Thus all the sets $\hat { K } _ { V } \left( x ^ { \prime } \right)$ were nonempty, and we chose x by maximizing $H _ { V }$ . It follows that

$$
r \left( x , \overline { { y } } \right) \geq H _ { V } \left( x \right) = \operatorname* { m a x } _ { x ^ { \prime } \in X } H _ { V } \left( x ^ { \prime } \right) \geq V ^ { \star } - G \frac { B _ { T } } { \sqrt { n } } .
$$

Since the reward is linear, $r \left( x , { \overline { { y } } } \right)$ is the average reward in the block. The block’s regret is therefore at most $G B _ { T } { \sqrt { n } }$ . There are at most $\textstyle { \frac { T } { n } }$ such blocks, for a total of $G B _ { T } \frac { T } { \sqrt { n } }$ . An informative block may cost as much as $C _ { r } n$ , but there are at most $J _ { \mathrm { m a x } }$ of them. A final incomplete block also costs at most $C _ { r } n$ . Adding these bounds proves the high-probability claim. Outside this event, regret is still at most $C _ { r } T$ , so taking expectations adds at most $C _ { r } T \delta$ □

## A.1 Computational tractability and finite-precision implementation

In Section 4.1, Lemma 8 turned problems 1 and 2 into maximization problems over an arm x and a dual vector z. In this section, we first show how each of them can be written as a QCQP that fits Theorem 9, and then we explain how to use an approximate solution. For the second step, we will modify the rule for choosing an arm and show that the changes preserve the ${ \tilde { O } } ( T ^ { 2 / 3 } )$ regret rate.

We start with problem 1, which asks whether there is an arm x with $\hat { K } _ { V } ( x ) = \varnothing$ (recall that ${ \hat { K } } _ { V } ( x )$ is the set of outcomes y in the Euclidean ball D with $w ( x , y )$ in the ellipsoid $E _ { V } )$ and, if so, returns such an arm. By Lemma $8 ( \mathrm { a } )$ , such an arm exists exactly when

$$
\operatorname* { m a x } _ { \substack { { x \in X , \| z \| _ { 2 } \leq 1 } } } \left[ z _ { 0 } + z _ { X } ^ { T } x - \sqrt { z ^ { T } V z } - \| z _ { D } \| _ { 2 } \right] > 0 ,\tag{3}
$$

where, as in Lemma 8, we split the dual vector as $\boldsymbol { z } = ( z _ { 0 } , z _ { X } , z _ { D } )$ with $z _ { 0 } \in \mathbb { R } , z _ { X } \in \mathbb { R } ^ { d _ { X } }$ , and $z _ { D } \in \mathbb { R } ^ { d _ { D } }$ , so that $z ^ { T } w ( x , y ) = z _ { 0 } + z _ { X } ^ { T } x + z _ { D } ^ { T } y$ . If the maximum is positive, the arm of every maximizing pair $( x , z )$ is such an arm. The next lemma shows that Theorem 9 can compute this maximum.

Lemma 16. The maximum in Equation 3 is the optimal value of a QCQP in x, z, and two scalar variables, with nine quadratic constraints, that satisfies the hypotheses of Theorem 9.

Proof. The maximization in Equation 3 is not yet a QCQP, because its objective contains the square roots $\sqrt { z ^ { T } V z }$ and $\| z _ { D } \| _ { 2 }$ . We remove them with two new scalar variables that bound them from above. We require $t \geq 0$ and $z ^ { T } V z \leq t ^ { 2 }$ , so that $t \geq \sqrt { z ^ { T } V z }$ , and $s \geq 0$ and $\| z _ { D } \| _ { 2 } ^ { 2 } \le s ^ { 2 }$ , so that $s \geq \| z _ { D } \| _ { 2 }$ . These constraints are quadratic. In the objective, we write $- t - s$ in place of the two square roots. For fixed x and z, maximizing −t − s pushes t and s down to the two square roots, so the optimal value does not change.

Theorem 9 also needs a constraint that defines a bounded ellipsoid in all the variables. The arm x and the dual vector z already lie in unit balls, so it remains to bound t and s. Since V is positive semidefinite, its largest eigenvalue is at most its trace tr V. Hence, for $\| z \| _ { 2 } \leq 1$ , the two square roots are at most ${ \sqrt { \operatorname { t r } V } } \leq 1 + \operatorname { t r } V$ and 1. We bound t and s by $2 + \operatorname { t r } V$ and 2. These bounds exceed the values above by at least one, so they do not change the optimal value. We use this extra room when we round the solver’s output below. With these bounds, the last constraint below is implied by the others. We add it because it defines a bounded ellipsoid in all the variables. This gives the QCQP

$$
\begin{array} { r l } { \underset { x \in \mathbb { R } ^ { d } } { \mathrm { m a x } } } & { z _ { 0 } + z _ { X } ^ { T } x - t - s } \\ { \underset { t , s \in \mathbb { R } } { \mathrm { m a x } } } & { } \\ { \mathrm { s u b j e c t ~ t o } } & { \| x \| _ { 2 } ^ { 2 } \leq 1 , \qquad \| z \| _ { 2 } ^ { 2 } \leq 1 , } \\ & { z ^ { T } V z \leq t ^ { 2 } , \qquad \| z _ { D } \| _ { 2 } ^ { 2 } \leq s ^ { 2 } , } \\ & { 0 \leq t \leq 2 + \mathrm { t r } V , \qquad 0 \leq s \leq 2 , } \\ & { \| x \| _ { 2 } ^ { 2 } + \| z \| _ { 2 } ^ { 2 } + \displaystyle \frac { t ^ { 2 } } { ( 2 + \mathrm { t r } V ) ^ { 2 } } + \displaystyle \frac { s ^ { 2 } } { 4 } \leq 4 . } \end{array}\tag{4}
$$

This program fits Theorem 9. It has nine quadratic constraints, however large $d _ { X }$ and $d _ { D }$ are, and a nonempty feasible set. The objective is quadratic, and the coeficients are rational when V has rational entries, which the rounding below ensures. The feasible set is convex, but the objective is not concave because of the bilinear term $z _ { X } ^ { T } x$

At an optimal solution, t and s equal the two square roots, so the pair $( x , z )$ attains the maximum in Equation 3, and the two optimal values agree. If this value is positive, Lemma $8 ( \mathrm { a ) }$ gives $\hat { K } _ { V } ( x ) = \varnothing$ at the arm x of every optimal solution. If it is not positive, the same lemma gives $\hat { K } _ { V } ( x ^ { \prime } ) \neq \emptyset$ at every arm $x ^ { \prime }$ □

Problem 2 asks for an arm maximizing $H _ { V }$ when ${ \hat { K } } _ { V } ( x )$ is nonempty for every arm, and it has the extra complication that in Lemma $8 ( \mathrm { b } ) , H _ { V } ( x )$ is a supremum over all $z \in \mathbb { R } ^ { q }$ instead of a maximum, and this supremum need not be attained. For now, we make the simplifying assumption that, for a known rational R, the supremum is attained at some z with $\| z \| _ { 2 } \leq R$ at every arm. We show later how to remove this assumption by enlarging the ellipsoid. Under this assumption, it is easy to write problem 2 as a QCQP as well.

Lemma 17. Suppose that ${ \hat { K } } _ { V } ( x )$ is nonempty for every arm and that, for a rational R, the supremum in Lemma $g ( b )$ is attained at some z with $\| z \| _ { 2 } \leq R$ at every arm. Then $\operatorname* { m a x } _ { x \in X } H _ { V } ( x )$ is the optimal value of a $Q C Q P$ in $x , z ,$ and two scalar variables, with nine quadratic constraints, that satisfies the hypotheses of Theorem 9 whenever V has rational entries. The arm of every optimal solution maximizes $H _ { V }$

Proof. By the assumption and Lemma $8 ( \mathrm { b } )$

$$
\operatorname* { m a x } _ { x \in X } H _ { V } ( x ) = \operatorname* { m a x } _ { x \in X , \ \| | z | \| _ { 2 } \leq R } \left[ a ^ { T } x + z _ { 0 } + z _ { X } ^ { T } x - \sqrt { z ^ { T } V z } - \| b + z _ { D } \| _ { 2 } \right] .
$$

Using the same trick as in the proof of Lemma 16, we get the following QCQP, where $U : = { }$ $1 + ( 1 + \operatorname { t r } V )$ R and $W : = 1 + \| b \| _ { 1 } + R$ exceed the largest values of $\sqrt { z ^ { T } V z }$ and $\| b + z _ { D } \| _ { 2 }$ on the ball $\| z \| _ { 2 } \leq R$ by at least one:

$$
\begin{array} { r l } { \underset { x \in \mathbb { R } ^ { d _ { X } } , z \in \mathbb { R } ^ { q } } { \operatorname* { m a x } } } & { a ^ { T } x + z _ { 0 } + z _ { X } ^ { T } x - t - s } \\ { \underset { t , s \in \mathbb { R } } { \operatorname* { m a x } } } & { } \\ { \mathrm { s u b j e c t ~ t o } } & { \| x \| _ { 2 } ^ { 2 } \leq 1 , \quad \| z \| _ { 2 } ^ { 2 } \leq R ^ { 2 } , } \\ & { z ^ { T } V z \leq t ^ { 2 } , \quad \| b + z _ { D } \| _ { 2 } ^ { 2 } \leq s ^ { 2 } , } \\ & { 0 \leq t \leq U , \quad 0 \leq s \leq W , } \\ & { \| x \| _ { 2 } ^ { 2 } + \frac { \| z \| _ { 2 } ^ { 2 } } { R ^ { 2 } } + \frac { t ^ { 2 } } { U ^ { 2 } } + \frac { s ^ { 2 } } { W ^ { 2 } } \leq 4 . } \end{array}\tag{5}
$$

For the same reasons as in the proof of Lemma 16, this program has the same optimal value as the maximization above and satisfies the hypotheses of Theorem 9, and the arm of every optimal solution maximizes $H _ { V }$ □

Theorem 9 only gives an approximate solution, allowing errors in both the objective and the constraints. We now explain how to use such a solution, starting again with problem 1. An approximate solution cannot tell exactly whether some ${ \hat { K } } _ { V } ( x )$ is empty. The next lemma shows that it either finds an arm with $\hat { K } _ { V } ( x ) = \varnothing$ or certifies that every arm has a nonempty set once we enlarge the ellipsoid slightly.

Lemma 18. Let V be a matrix with rational entries maintained by the learner, let $\eta \in ( 0 , 1 )$ be rational, and write $\hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x ) : = \{ y \in D : p _ { V } ( x , y ) \leq ( 1 + \eta \sqrt { n } ) ^ { 2 } \}$ , (recall that n is the block length of Algorithm 1 and appears in the definition of V). There is an algorithm that computes a rational arm $x \in X$ and either certifies that $\hat { K } _ { V } ( x ) = \varnothing$ or certifies that $\hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x ^ { \prime } ) \neq \varnothing$ for every arm $x ^ { \prime } \in X$ Its running time is polynomial in $d _ { X } , d _ { D } , \log ( 1 / \eta )$ , and the bit length of V.

Proof. If $\eta ^ { \prime } \leq \eta .$ , then $\hat { K } _ { V , \eta ^ { \prime } } ^ { \mathrm { o u t } } ( x ) \subseteq \hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x )$ for every arm x. So we may replace η by the largest power of 2 that is at most η. This changes $\log _ { 2 } ( 1 / \eta )$ by less than one and gives η a bit length of $O ( \log ( 1 / \eta ) )$

Write $S _ { x } : = \{ w ( x , y ) : y \in D \}$ , and let dist denote the Euclidean distance between two sets. We first show that, for every arm x,

$$
\operatorname* { m a x } _ { \| \boldsymbol { z } \| _ { 2 } \leq 1 } \left[ z _ { 0 } + z _ { X } ^ { T } \boldsymbol { x } - \sqrt { \boldsymbol { z } ^ { T } V \boldsymbol { z } } - \| \boldsymbol { z } _ { D } \| _ { 2 } \right] = \operatorname { d i s t } ( E _ { V } , S _ { x } ) .\tag{6}
$$

By Lemma 16 and its proof, the optimal value of Equation 4 is then ma $\complement _ { x \in X }$ dist $( E _ { V } , S _ { x } )$ . The Euclidean distance between u and v is $\begin{array} { r } { \operatorname* { m a x } _ { \| \boldsymbol { z } \| _ { 2 } \leq 1 } \boldsymbol { z } ^ { T } ( \boldsymbol { v } - \boldsymbol { u } ) } \end{array}$ . Both $E _ { V } \times S _ { x }$ and the unit ball for z are compact and convex, and $z ^ { T } ( v - u )$ is afine in each variable. Sion’s minimax theorem [76] therefore gives

$$
\mathrm { d i s t } ( E _ { V } , S _ { x } ) = \operatorname* { m a x } _ { \| z \| _ { 2 } \leq 1 } \left[ \operatorname* { m i n } _ { v \in S _ { x } } z ^ { T } v - \operatorname* { m a x } _ { u \in E _ { V } } z ^ { T } u \right] ,
$$

and the linear optimization formulas from the proof of Lemma 8 turn the right-hand side into the left-hand side of Equation 6.

Next, we run the algorithm of Theorem 9 on Equation 4 with tolerance $\varepsilon : = ( \eta / ( 2 1 + 2 \operatorname { t r } V ) ) ^ { 2 }$ It returns a rational point $( \hat { x } , \hat { z } , \hat { t } , \hat { s } )$ that violates each constraint by at most ε and whose objective value is at least the optimum minus ε. We repair this point so that it becomes feasible. Since $\| \hat { x } \| _ { 2 } ^ { 2 } \leq 1 + \varepsilon$ and $\begin{array} { r } { \| \hat { z } \| _ { 2 } ^ { 2 } \leq 1 + \varepsilon . } \end{array}$ , we can scale xˆ and zˆ into the unit ball and round their coordinates toward zero to the grid whose spacing is the largest power of 2 that is at most $\sqrt { \varepsilon } / q$ . This gives rational x and z in the unit ball with $\| x - { \hat { x } } \| _ { 2 } \leq 2 { \sqrt { \varepsilon } }$ and $\| z - { \hat { z } } \| _ { 2 } \leq 2 { \sqrt { \varepsilon } }$ . Since the spacing is a power of 2, the grids used in diferent calls are nested. We then choose rational t and s with $\sqrt { z ^ { T } V z } \leq t \leq \sqrt { z ^ { T } V z } + \sqrt { \varepsilon }$ and $\| z _ { D } \| _ { 2 } \leq s \leq \| z _ { D } \| _ { 2 } + \sqrt { \varepsilon }$ . Since $\sqrt { z ^ { T } V z } \leq \sqrt { \mathrm { t r } V } \leq 1 + \mathrm { t r } V$ $\| z _ { D } \| _ { 2 } \leq 1$ , and $\sqrt { \varepsilon } \leq 1$ , the point $( x , z , t , s )$ satisfies every constraint of Equation 4.

The repair lowers the objective by at most $( 2 0 + 2 \operatorname { t r } V ) { \sqrt { \varepsilon } }$ . Indeed, the solver’s constraints give $\hat { t } , \hat { s } \geq - \varepsilon , \sqrt { \hat { z } ^ { T } V \hat { z } } \leq | \hat { t } | + \sqrt { \varepsilon } .$ , and $\| \hat { z } _ { D } \| _ { 2 } \leq | \hat { s } | + \sqrt { \varepsilon }$ . Since $\| V ^ { 1 / 2 } v \| _ { 2 } \leq \sqrt { \mathrm { t r } V } \| v \| _ { 2 }$ for every v, the triangle inequality gives

$$
t - { \hat { t } } \leq { \bigl ( } 4 + 2 { \sqrt { \operatorname { t r } V } } { \bigr ) } { \sqrt { \varepsilon } } , \qquad s - { \hat { s } } \leq 6 { \sqrt { \varepsilon } } .
$$

Also $\| \hat { z } \| _ { 2 } \leq 2 , \mathrm { s o } \| \hat { z } _ { 0 } - z _ { 0 } \| + | \hat { z } _ { X } ^ { T } \hat { x } - z _ { X } ^ { T } x | \leq 8 \sqrt { \varepsilon }$ . Adding these bounds proves the claim. Write ℓ for the objective value of $( x , z , t , s )$ , which we compute exactly. Then ℓ is at least the optimum minus

$$
\varepsilon + ( 2 0 + 2 \operatorname { t r } V ) \sqrt { \varepsilon } \leq \eta , \mathrm { s o }
$$

$$
\operatorname* { m a x } _ { x ^ { \prime } \in X } \mathrm { d i s t } ( E _ { V } , S _ { x ^ { \prime } } ) \leq \ell + \eta .
$$

On the other hand, $t \geq \sqrt { z ^ { T } V z }$ and $s \geq \| z _ { D } \| _ { 2 }$ , so Equation 6 gives $\ell \leq \mathrm { d i s t } ( E _ { V } , S _ { x } )$

If $\ell > 0$ , then $S _ { x }$ and $E _ { V }$ are disjoint, so no $y \in D$ has $p _ { V } ( x , y ) \leq 1$ , and we certify that $\hat { K } _ { V } ( x ) = \varnothing$ . Otherwise, every arm $x ^ { \prime }$ satisfies dist $( E _ { V } , S _ { x ^ { \prime } } ) \leq \eta$ . Since both sets are compact, there are $y \in D$ and $u \in E _ { V }$ with $\| w ( x ^ { \prime } , y ) - u \| _ { 2 } \leq \eta$ . Since V starts at $n ^ { - 1 } I _ { q }$ and each update adds a positive semidefinite matrix, we have $V \succeq n ^ { - 1 } I _ { q }$ . Hence $\| v \| _ { V ^ { - 1 } } \leq \sqrt { n } \| v \| _ { 2 }$ for every v, and the triangle inequality gives $\| w ( x ^ { \prime } , y ) \| _ { V ^ { - 1 } } \leq \| u \| _ { V ^ { - 1 } } + \eta \sqrt { n } \leq 1 + \eta \sqrt { n }$ . Thus $y \in \hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x ^ { \prime } )$ , and we certify the second case.

For the running time, the program in Equation 4 has $2 d _ { X } + d _ { D } + 3$ variables and nine constraints. Its coeficients are the entries of V and numbers computed from tr $V ,$ so their total bit length is polynomial in $q$ and the bit length of V. By Theorem 9, the solver runs in time polynomial in this input size and $\log ( 1 / \varepsilon ) = 2 \log ( ( 2 1 + 2 \operatorname { t r } V ) / \eta )$ , which is polynomial in $\log ( 1 / \eta ) , q ;$ , and the bit length of $V ,$ , since tr $V$ is a sum of $q$ entries of V. The repair and the computation of ℓ use rational arithmetic and square roots to precision ${ \sqrt { \varepsilon } } .$ They take time polynomial in the same quantities and in the bit length of the solver’s output, which is bounded by the solver’s running time. □

We now use Lemma 18 to choose the arm, with a value of η that we pick later. At the start of each block, we run its algorithm. If it certifies that $\hat { K } _ { V } ( x ) = \varnothing$ , we play the arm x that it returns, as Algorithm 1 would. Otherwise, it certifies that $\hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x )$ is nonempty for every arm, and we solve problem 2 with these sets in place of ${ \hat { K } } _ { V } ( x )$ . Recall, however, that Lemma $8 ( \mathrm { b ) }$ writes the score as a supremum over all $z \in \mathbb { R } ^ { q }$ , which need not be attained at a bounded z. We fix this with the following lemma.

Lemma 19. Let V be a positive definite matrix with $V \succeq n ^ { - 1 } I _ { q }$ , and let $x \in X$ . Suppose that some outcome $y _ { 0 } \in D$ satisfies $\| w ( x , y _ { 0 } ) \| _ { V ^ { - 1 } } \leq 1 - \mu$ for some $\mu \in ( 0 , 1 ]$ . Then the supremum in Lemma $g ( b )$ is attained at some z with $\| z \| _ { 2 } \leq 2 { \sqrt { n } } \| b \| _ { 2 } / \mu$

Proof. The outcome $y _ { 0 }$ belongs to ${ \hat { K } } _ { V } ( x )$ , so Lemma 8(b) applies. Write $w _ { 0 } : = w ( x , y _ { 0 } )$ , and let

$$
F ( z ) : = z _ { 0 } + z _ { X } ^ { T } x - \sqrt { z ^ { T } V z } - \| b + z _ { D } \| _ { 2 }
$$

be the expression in its supremum. Since $\| y _ { 0 } \| _ { 2 } \leq 1$ , we have $- \| b + z _ { D } \| _ { 2 } \le ( b + z _ { D } ) ^ { T } y _ { 0 } ,$ so $F ( z ) \leq \bar { b ^ { T } } y _ { 0 } + z ^ { T } w _ { 0 } - \sqrt { z ^ { T } \bar { V z } }$ . Cauchy–Schwarz in the V norm gives $z ^ { T } w _ { 0 } \leq \| w _ { 0 } \| _ { V ^ { - 1 } } \sqrt { z ^ { T } V z } \leq$ $( 1 - \mu ) \sqrt { z ^ { T } V z }$ . Hence

$$
F ( z ) \leq \| b \| _ { 2 } - \mu { \sqrt { z ^ { T } V z } } .
$$

Since $F ( 0 ) = - \| b \| _ { 2 } .$ , every z with $\sqrt { z ^ { T } V z } > 2 \| b \| _ { 2 } / \mu$ gives a smaller value than $z = 0$ . The set of z with $\sqrt { z ^ { T } V z } \leq 2 \| b \| _ { 2 } / \mu$ is compact and F is continuous, so $F$ attains its supremum on this set. Finally, $V \succeq n ^ { - 1 } I _ { q }$ gives $\| z \| _ { 2 } \leq \sqrt { n } \sqrt { z ^ { T } V z } \leq 2 \sqrt { n } \| b \| _ { 2 } / \mu$ □

Lemma 19 needs an outcome strictly inside the ellipsoid, with a known margin. The sets $\hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x )$ do not provide one, since the outcomes that Lemma 18 certifies may lie on the boundary of their ellipsoid. We therefore enlarge the ellipsoid a little further and maximize $H _ { 4 V }$ , the score of Algorithm 1 for the matrix 4V . Its sets $\hat { K } _ { 4 V } ( x ) = \{ y \in D : p _ { V } ( x , y ) \leq 4 \}$ use the ellipsoid $E _ { 4 V } = 2 E _ { V }$ , and when $\eta \sqrt { n } \leq 1 / 2$ , they contain the sets $\hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x )$ with room to spare. Like the program for problem 1, the program for this score can only be solved approximately. The next lemma shows that we can still find an arm whose score is within any $\kappa \in ( 0 , 1 )$ of the maximum.

Lemma 20. Let V be a matrix with rational entries maintained by the learner, and let $\eta , \kappa \in ( 0 , 1 )$ be rational with $\eta \sqrt { n } \leq 1 / 2$ . Suppose that $\hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x )$ is nonempty for every arm x. There is an algorithm that computes a rational arm $x \in X$ with $H _ { 4 V } ( x ) \geq \operatorname* { m a x } _ { x ^ { \prime } \in X } H _ { 4 V } ( x ^ { \prime } ) - \kappa$ . Its running time is polynomial in $d _ { X } , d _ { D }$ , log $n , \log ( 1 / \kappa )$ , and the bit lengths of $V , a ,$ and b.

Proof. We first write the problem as a program that fits Theorem 9. Fix an arm x. Since $\hat { K } _ { V , \eta } ^ { \mathrm { o u t } } ( x )$ is nonempty, some outcome $y _ { 0 } \in D$ satisfies $\| w ( x , y _ { 0 } ) \| _ { V ^ { - 1 } } \leq 1 + \eta \sqrt { n } \leq 3 / 2$ . Since $\| \cdot \| _ { ( 4 V ) ^ { - 1 } } = \frac { 1 } { 2 } \| \cdot \| _ { V ^ { - 1 } }$ , this gives $\| w ( x , y _ { 0 } ) \| _ { ( 4 V ) ^ { - 1 } } \leq 3 / 4$ . So $\hat { K } _ { 4 V } ( x )$ is nonempty, and Lemma 19, applied to the matrix $4 V \succeq n ^ { - 1 } I _ { q }$ with $\mu = 1 / 4$ , shows that the supremum in Lemma $8 ( \mathrm { b ) }$ for $4 V$ is attained at some z with $\| z \| _ { 2 } \leq 8 { \sqrt { n } } \| b \| _ { 2 } \leq M$ , where $M : = 8 n ( 1 + \| b \| _ { 1 } )$ . Hence the assumption of Lemma 17 holds for the matrix 4V with $R = M$ , and $\operatorname* { m a x } _ { x \in X } H _ { 4 V } ( x )$ is the optimal value of the program in Equation 5 with V replaced by 4V and $R = M$

A smaller κ gives a stronger guarantee, so, as in the proof of Lemma 18, we may assume that κ is a power of 2. We run the algorithm of Theorem 9 on this program, using the tolerance $\varepsilon : = ( \kappa / ( K + 1 ) ) ^ { 2 }$ , where $K : = 2 0 + 2 \| a \| _ { 1 } + 2 M + 4 \operatorname { t r } V$ . We repair its output as in the proof of Lemma 18, now scaling zˆ into the ball of radius M. The repaired point is feasible, and the computation in that proof shows that the repair lowers the objective by at most $K \sqrt \varepsilon$ , with three changes. The bound $\| \hat { z } \| _ { 2 } \leq M + 1$ replaces $\| \hat { z } \| _ { 2 } \leq 2 \qquad $ , so the terms $z _ { \mathrm { 0 } }$ and $z _ { X } ^ { T } x$ change by at most $( 2 M + 6 ) \sqrt { \varepsilon }$ in place of $8 \sqrt \varepsilon$ . The matrix 4V replaces V, which turns $\sqrt { \mathrm { t r } V }$ into $2 \sqrt { \mathrm { t r } V }$ in the bound on $t - { \widehat t } .$ Finally, the term $a ^ { T } x$ changes by at most $2 \| a \| _ { 1 } \sqrt { \varepsilon }$

Let $( x , z , t , s )$ be the repaired point. Since $t \ge \sqrt { z ^ { T } ( 4 V ) z }$ and $s \geq \| b + z _ { D } \| _ { 2 }$ , its objective value is at most $a ^ { T } x + F _ { x } ( z )$ , where $F _ { x } ( z ) : = z _ { 0 } + z _ { X } ^ { T } x - 2 \sqrt { z ^ { T } V z } - \| b + z _ { D } \| _ { 2 }$ is the expression in Lemma $8 ( \mathrm { b ) }$ for the matrix 4V. The proof of Lemma $8 ( \mathrm { b } )$ shows that $a ^ { T } x + F _ { x } ( z ) \leq H _ { 4 V } ( x )$ for every z. Hence

$$
H _ { 4 V } ( x ) \geq \operatorname* { m a x } _ { x ^ { \prime } \in X } H _ { 4 V } ( x ^ { \prime } ) - \varepsilon - K \sqrt { \varepsilon } \geq \operatorname* { m a x } _ { x ^ { \prime } \in X } H _ { 4 V } ( x ^ { \prime } ) - \kappa .
$$

The running time follows as in the proof of Lemma 18. The coeficients now also include $a , b ,$ and M, and $\log ( 1 / \varepsilon ) = 2 \log ( ( K + 1 ) / \kappa )$ is polynomial in $\log ( 1 / \kappa )$ , log n, q, and the bit lengths of $V , a ,$ and b. □

It remains to pick η and κ so that these approximations do not change the regret rate. We take

$$
\eta : = \frac { 1 } { 2 \lceil \sqrt { n } \rceil } , \qquad \kappa : = \frac { 1 } { T } .
$$

The choice of η gives $\eta \sqrt { n } \leq 1 / 2$ , so in the second case of Lemma 18 the hypotheses of Lemma 20 hold, and we choose the arm with the algorithm of that lemma. The choice of κ puts the score of this arm within $1 / T$ of the maximum of $H _ { 4 V }$ , which adds at most 1 to the regret over all T rounds, as we show next. Both choices keep the running times polynomial, since $\log ( 1 / \eta ) = \log ( 2 \lceil \sqrt { n } \rceil )$ and $\log ( 1 / \kappa ) = \log T$

We now bound the regret, following the proof of Theorem 15. That proof uses the rule for choosing arms only to bound the regret of blocks that do not update V. Its other steps, namely the event of Lemma 13, the bound $J \le J _ { \mathrm { m a x } }$ on the number of updates, and Equation 2, depend only on the update rule, which we have not changed. A full block played at an arm with $\hat { K } _ { V } ( x ) = \varnothing$ always updates V, since its average lies in D and therefore has $p _ { V } > 1$ . So every full block that does not update V was chosen by Lemma 20. In such a block, the average $\bar { y }$ has $p _ { V } ( x , \bar { y } ) \leq 1$ , so $\bar { y } \in \hat { K } _ { 4 V } ( x )$ Every $y \in \hat { K } _ { 4 V } ( x ^ { \prime } )$ has $p _ { V } ( x ^ { \prime } , y ) \leq 4$ , so Equation 2 and Lemma 12 give $r ( x ^ { \prime } , y ) \geq v ^ { \star } ( x ^ { \prime } ) - 2 G B _ { T } / \sqrt { n }$ Taking $x ^ { \prime }$ to be an arm with $v ^ { \star } ( x ^ { \prime } ) = V ^ { \star }$ , we get

$$
r ( x , \bar { y } ) \geq H _ { 4 V } ( x ) \geq \operatorname* { m a x } _ { x ^ { \prime } \in X } H _ { 4 V } ( x ^ { \prime } ) - \kappa \geq V ^ { \star } - \frac { 2 G B _ { T } } { \sqrt { n } } - \kappa .
$$

Since $r$ is linear in $y ,$ the reward of the block is $n r ( x , { \bar { y } } )$ . There are at most $J _ { \mathrm { m a x } }$ blocks that update $V ,$ , each costing at most $C _ { r } n$ , and the final incomplete block costs at most another $C _ { r } n$ . Hence, with probability at least $1 - \delta$

$$
T V ^ { \star } - \sum _ { t = 1 } ^ { T } r ( x _ { t } , y _ { t } ) \leq C _ { r } n ( J _ { \operatorname* { m a x } } + 1 ) + \frac { 2 G B _ { T } T } { \sqrt { n } } + T \kappa .
$$

This is the bound of Theorem 15 with two changes. The middle term doubles, because we score arms with $2 E _ { V }$ instead of $E _ { V }$ , and the new term Tκ equals 1. The expected regret is at most this bound plus $C _ { r } T \delta$ , so taking $n = \lceil T ^ { 2 / 3 } \rceil$ and $\delta = ( T + 1 ) ^ { - 2 }$ , as in that theorem, gives the same ${ \tilde { O } } ( T ^ { 2 / 3 } )$ rate.

Finally, we bound the running time of the whole learner. We assume that the observations are rational, with bit length polynomial in |ϕ| and T. Every matrix maintained by the learner satisfies $V \preceq ( n ^ { - 1 } + 3 T ) I _ { q } .$ , since $V$ starts at $n ^ { - 1 } I _ { q }$ and each of the at most T updates adds a positive semidefinite matrix of operator norm at most three. So tr $V \leq q ( n ^ { - 1 } + 3 T )$ , and the grid spacings used in Lemmas 18 and 20 are powers of 2 whose exponents are polynomial in q, log $T$ , and the bit lengths of $a$ and b. Hence every stored matrix has rational entries of polynomial bit length, and matrix inversion and the test $p _ { V } > 1$ can be performed exactly in polynomial time. Each block calls the algorithm of Lemma 18 once and, in its second case, the algorithm of Lemma 20 once, and there are at most $\lceil T / n \rceil$ blocks. With our choices of $\eta$ and $\kappa ,$ each call takes time polynomial in |ϕ| and $T ,$ as required.

## B Regret analysis of the $\sqrt { T }$ learner

We give the precise statement and proof of Theorem 11. We use the coordinates and notation from Section 5, including $\lambda = \lceil { \sqrt { T } } \rceil$ , and assume that Algorithm 2 maximizes $U _ { V , \lambda }$ exactly. Appendix B.1 shows how to maximize it approximately in polynomial time without changing the regret rate.

For the analysis, fix $\delta \in ( 0 , 1 )$ , use $h _ { T }$ and G from Appendix A, and define

$$
J _ { \mathrm { m a x } } ^ { \prime } : = \left\lceil q \log _ { 2 } \left( 1 + \frac { 3 T } { q } \right) \right\rceil , \qquad B _ { T } ^ { \prime } : = 1 + J _ { \mathrm { m a x } } ^ { \prime } h _ { T } ^ { 2 } .
$$

These play the roles of $J _ { \mathrm { m a x } }$ and $B _ { T }$ in Appendix A. The horizon T takes the place of the block length n because this learner starts from $V = I _ { q }$ and weights each block by its length, so the trace of $V$ can grow to $q + 3 T$ . We use the concentration bound from Lemma 13 and adapt the proof of Lemma 14 to the new matrix update.

Theorem 21. Algorithm 2 satisfies, with probability at least $1 - \delta$

$$
T V ^ { \star } - \sum _ { t = 1 } ^ { T } r ( x _ { t } , y _ { t } ) \leq 4 \lambda \left( J _ { \operatorname* { m a x } } ^ { \prime } + 1 \right) + \frac { G ^ { 2 } { B _ { T } ^ { \prime } } ^ { 2 } T } { 4 \lambda } .
$$

Its expected regret is at most the same expression plus $C _ { r } T \delta$ , and taking $\delta = ( T + 1 ) ^ { - 2 }$ gives

$$
R _ { T } \leq \tilde { O } \left( \left[ q + \left( 1 + S ^ { - 1 } \right) ^ { 2 } \| b \| _ { 2 } ^ { 2 } q d _ { D } \right] \sqrt { T } \right) + O \left( \frac { C _ { r } } { T } \right) .
$$

Note that the learner does not need to know $S$ or the quantities $J _ { \operatorname* { m a x } } ^ { \prime } , h _ { T } , B _ { T } ^ { \prime } , G$

Proof. As in the proof of Theorem 15, each completed block at least doubles the determinant of $V .$ The stopping rule gives

$$
\operatorname* { d e t } \left( V + n \overline { { w } } \overline { { w } } ^ { T } \right) = \operatorname* { d e t } ( V ) \left( 1 + n p _ { V } ( x , \overline { { y } } ) \right) \geq 2 \operatorname* { d e t } ( V ) .
$$

Since $\| \overline { { w } } \| _ { 2 } ^ { 2 } \leq 3$ and the block lengths add up to at most $T ,$ the final trace is at most $q + 3 T$ . Thus, if we complete J blocks, the arithmetic–geometric mean inequality on the eigenvalues of V gives $2 ^ { J } \leq$ det $V \leq ( 1 + 3 T / q ) ^ { q }$ , so $J \leq J _ { \mathrm { m a x } } ^ { \prime }$

We also need to bound how far $n p _ { V } ( x , { \overline { { y } } } _ { n } )$ can jump above 1 on the last observation. For $n \geq 2$ convexity gives

$$
n p _ { V } ( x , \overline { { y } } _ { n } ) \leq ( n - 1 ) p _ { V } ( x , \overline { { y } } _ { n - 1 } ) + p _ { V } ( x , y _ { n } ) < 1 + 3 = 4 .
$$

The first term is less than one because we did not stop at $n - 1$ . The second is at most three because $V \succeq I _ { q }$ and $p _ { V } ( x , y _ { n } ) \leq \| w ( x , y _ { n } ) \| _ { 2 } ^ { 2 } \leq 3$ . If we stop at $n = 1$ , the bound is at most three. If the horizon ends before $n p _ { V } ( x , { \overline { { y } } } _ { n } )$ reaches 1, it is less than one. Thus every block satisfies

$$
n p _ { V } ( x , { \overline { { y } } } ) \leq 4 .\tag{7}
$$

We now adapt the proof of Lemma 14. Number the completed blocks $j = 1 , \dots , J ,$ and let $n _ { j }$ be the length of block j. Write $x _ { j }$ for its arm, $\overline { { y } } _ { j }$ for its average observation, $\overline { { m } } _ { j }$ for the average over the block of $m _ { t } = \mathbb { E } _ { y \sim P _ { t } } [ y ]$ , the expected value of the distribution chosen on round t, and $\overline { { w } } _ { j } : = w ( x _ { j } , \overline { { y } } _ { j } )$ . After J completed blocks, the matrix has the form

$$
V = I _ { q } + \sum _ { j = 1 } ^ { J } n _ { j } { \overline { { w } } } _ { j } { \overline { { w } } } _ { j } ^ { T } .
$$

Fix $u \in \mathcal { L } ^ { \perp }$ with $\| u \| _ { 2 } \leq 1$ . Since the arm is held fixed within each block, $w ( x _ { j } , \overline { { m } } _ { j } ) \in \mathcal { L }$ , so $u ^ { T } \overline { { w } } _ { j } = u ^ { T } ( 0 , 0 , \overline { { y } } _ { j } - \overline { { m } } _ { j } )$ . By Lemma 13, with probability at least $1 - \delta .$ , the bound $( u ^ { \bar { T } } \overline { { w } } _ { j } ) ^ { 2 } \leq h _ { T } ^ { 2 } / n _ { j }$ holds simultaneously for all completed blocks and all such u. On this event,

$$
u ^ { T } V u \leq 1 + \sum _ { j = 1 } ^ { J } n _ { j } \cdot \frac { h _ { T } ^ { 2 } } { n _ { j } } = 1 + J h _ { T } ^ { 2 } \leq { B _ { T } ^ { \prime } } ^ { 2 } .
$$

This is where the weight $n _ { j }$ matters. It cancels the $1 / n _ { j }$ in the squared noise bound, so each block contributes at most $h _ { T } ^ { 2 }$ , however long it is. Applying Cauchy–Schwarz and taking the supremum over these u gives

$$
\mathrm { d i s t } ( w , { \mathcal { L } } ) \leq B _ { T } ^ { \prime } \| w \| _ { V ^ { - 1 } } .\tag{8}
$$

Combining Lemma 12 with Equation 8 gives, for every x $\in X$ and $y \in D$

$$
v ^ { \star } ( x ) \leq r ( x , y ) + G B _ { T } ^ { \prime } \sqrt { p _ { V } ( x , y ) } .\tag{9}
$$

This is the claim in the proof of Lemma 10, with $c = G B _ { T } ^ { \prime }$ . That proof then shows that, on the same event, the arm x of every block satisfies

$$
V ^ { \star } - r ( x , y ) \leq \lambda p _ { V } ( x , y ) + \frac { G ^ { 2 } { B _ { T } ^ { \prime } } ^ { 2 } } { 4 \lambda } \qquad \mathrm { f o r ~ e v e r y ~ } y \in D .
$$

We apply this to the block’s average ${ \overline { { y } } } ,$ which lies in D because $D$ is convex. Since the reward is linear, the block’s reward is $n r ( x , { \overline { { y } } } )$ , and Equation 7 bounds its regret by

$$
n \left( V ^ { \star } - r ( x , \overline { { y } } ) \right) \leq \lambda n p _ { V } ( x , \overline { { y } } ) + \frac { n G ^ { 2 } { B _ { T } ^ { \prime } } ^ { 2 } } { 4 \lambda } \leq 4 \lambda + \frac { n G ^ { 2 } { B _ { T } ^ { \prime } } ^ { 2 } } { 4 \lambda } .
$$

There are at most $J _ { \operatorname* { m a x } } ^ { \prime } + 1$ blocks, namely the completed ones and possibly a final block cut short by the horizon, and their lengths add up to T. Adding the bounds proves the high-probability claim. Outside this event, regret is still at most $C _ { r } T _ { \ l }$ , so taking expectations adds at most $C _ { r } T \delta$

Finally, with $\lambda = \lceil \sqrt { T } \rceil$ , the first term of the bound is $\tilde { O } ( q \sqrt { T } )$ . Since $G ^ { 2 } = 3 ( 1 + S ^ { - 1 } ) ^ { 2 } \| b \| _ { 2 } ^ { 2 }$ and ${ B _ { T } ^ { \prime } } ^ { 2 } = \tilde { O } ( q d _ { D } )$ , the second is $\tilde { O } ( ( 1 + S ^ { - 1 } ) ^ { 2 } \| b \| _ { 2 } ^ { 2 } q d _ { D } \sqrt { T } )$ . With $\delta = ( T + 1 ) ^ { - 2 }$ , we have $C _ { r } T \delta \leq C _ { r } / T$ □

## B.1 Computational tractability and finite-precision implementation

Algorithm 2 has a single computational problem, which is to find an arm maximizing $U _ { V , \lambda }$ . Unlike Algorithm 1, it has no emptiness test. We follow Appendix A.1. We first write the maximization as a QCQP that fits Theorem 9, and then we explain how to use an approximate solution and show that this preserves the $\tilde { O } ( \sqrt { T } )$ rate.

Recall that X and D are unit balls, that $\begin{array} { r } { r ( x , y ) = a ^ { T } x + b ^ { T } y . } \end{array}$ , that $w ( x , y ) = ( 1 , x , y )$ , and that $p _ { V } ( x , y ) = \| w ( x , y ) \| _ { V ^ { - 1 } } ^ { 2 }$ . As in Lemma 8, we split a vector $z \in \mathbb { R } ^ { q }$ as $\boldsymbol { z } = ( z _ { 0 } , z _ { X } , z _ { D } )$ , so that $z ^ { T } w ( x , y ) = z _ { 0 } + z _ { X } ^ { T } \dot { x } + z _ { D } ^ { T } y$ . Every matrix maintained by the learner satisfies $V \succeq I _ { q }$ , since $V$ starts at $I _ { q }$ and each update adds a positive semidefinite matrix.

Lemma 8(b) turned the minimum over outcomes in $H _ { V }$ into a maximum over a dual vector $z .$ The next lemma does the same for $U _ { V , \lambda }$ . The diference is where z comes from. In Lemma 8(b), z was the Lagrange multiplier that tied $w ( x , y )$ to the ellipsoid $E _ { V }$ . Here y ranges over all of $D ,$ and z comes from writing the penalty $\lambda p _ { V } ( x , y )$ itself as a maximum.

Lemma 22. Let $V \succeq I _ { q }$ and $\lambda > 0$ . For every $x \in X$ ，

$$
U _ { V , \lambda } ( \boldsymbol { x } ) = \boldsymbol { a } ^ { T } \boldsymbol { x } + \operatorname* { m a x } _ { \| \boldsymbol { z } \| _ { 2 } \leq 4 \lambda } \left[ \boldsymbol { z } _ { 0 } + \boldsymbol { z } _ { X } ^ { T } \boldsymbol { x } - \| \boldsymbol { b } + \boldsymbol { z } _ { D } \| _ { 2 } - \frac { \boldsymbol { z } ^ { T } V \boldsymbol { z } } { 4 \lambda } \right] .
$$

Proof. For every $w \in \mathbb { R } ^ { q }$

$$
\lambda w ^ { T } V ^ { - 1 } w = \operatorname* { m a x } _ { z \in \mathbb { R } ^ { q } } \left[ z ^ { T } w - \frac { z ^ { T } V z } { 4 \lambda } \right] .
$$

Indeed, the expression in brackets is concave in $z ,$ and its gradient $w - V z / ( 2 \lambda )$ vanishes at $z = 2 \lambda V ^ { - 1 } w$ , where the expression equals $2 \lambda w ^ { T } V ^ { - 1 } w - \lambda w ^ { T } \bar { V } ^ { - 1 } w$ . For $w = w ( x , y )$ with $x \in X$ and $y \in D ,$ , we have $\| w \| _ { 2 } ^ { 2 } = 1 + \| x \| _ { 2 } ^ { 2 } + \| y \| _ { 2 } ^ { 2 } \leq 3$ , and $V \succeq I _ { q }$ gives $\| 2 \lambda V ^ { - 1 } w \| _ { 2 } \leq 2 \lambda \| w \| _ { 2 } \leq 4 \lambda$ . So we may restrict the maximum to the ball $\| z \| _ { 2 } \leq 4 \lambda$ , and

$$
U _ { V , \lambda } ( x ) = \operatorname* { m i n } _ { y \in D } \operatorname* { m a x } _ { \| z \| _ { 2 } \leq 4 \lambda } \left[ r ( x , y ) + z ^ { T } w ( x , y ) - \frac { z ^ { T } V z } { 4 \lambda } \right] .
$$

For fixed x, the expression in brackets is afine in y and concave in $z ,$ and both D and the ball are compact and convex. Sion’s minimax theorem [76] therefore lets us exchange the minimum and the maximum. For fixed z, the minimum over $y \in D$ is $a ^ { T } x + z _ { 0 } + z _ { X } ^ { T } x - \| b + z _ { D } \| _ { 2 }$ , by the same linear optimization over D as in the proof of Lemma 8. The maximum over z is attained because the resulting expression is continuous in z and the ball is compact. □

Unlike the supremum in Lemma $8 ( \mathrm { b ) }$ , this maximum is over a bounded set, so we do not need a margin argument such as Lemma 19. The penalty term $z ^ { T } V z / ( 4 \lambda )$ is already quadratic, so writing the maximization over arms as a QCQP only needs one of the two substitutions from the proof of Lemma 16.

Lemma 23. Let $V \succeq I _ { q }$ have rational entries, and let $\lambda \geq 1$ be an integer. Then $\mathrm { m a x } _ { x \in X } U _ { V , \lambda } ( x )$ is the optimal value of a QCQP in $x , z ,$ and one scalar variable, with six quadratic constraints, that satisfies the hypotheses of Theorem 9. The arm of every optimal solution maximizes $U _ { V , \lambda }$

Proof. By Lemma 22,

$$
\operatorname* { m a x } _ { x \in X } U _ { V , \lambda } ( x ) = \operatorname* { m a x } _ { x \in X , \ \| | z | | _ { 2 } \leq 4 \lambda } \left[ a ^ { T } x + z _ { 0 } + z _ { X } ^ { T } x - \| b + z _ { D } \| _ { 2 } - \frac { z ^ { T } V z } { 4 \lambda } \right] .
$$

As in the proof of Lemma 16, we replace the norm by a scalar variable s with $s \geq 0$ and $\| b + z _ { D } \| _ { 2 } ^ { 2 } \leq s ^ { 2 }$ and write −s in its place. Let $W : = 1 + \| b \| _ { 1 } + 4 \lambda$ , which exceeds the largest value of $\| b + z _ { D } \| .$ 2 on the ball $\| z \| _ { 2 } \leq 4 \lambda$ by at least one. We use this extra room when we repair the solver’s output below. This gives the QCQP

$$
\begin{array} { r l r } { \underset { x \in \mathbb { R } ^ { d _ { X } } , \ z \in \mathbb { R } ^ { q } } { \operatorname* { m a x } } } & { a ^ { T } x + z _ { 0 } + z _ { X } ^ { T } x - s - \frac { z ^ { T } V z } { 4 \lambda } } \\ { \mathrm { s u b j e c t ~ t o } } & { \| x \| _ { 2 } ^ { 2 } \leq 1 , \quad } & { \| z \| _ { 2 } ^ { 2 } \leq 1 6 \lambda ^ { 2 } , \quad } & { \| b + z _ { D } \| _ { 2 } ^ { 2 } \leq s ^ { 2 } , } \\ & { 0 \leq s \leq W , \quad } & { \| x \| _ { 2 } ^ { 2 } + \frac { \| z \| _ { 2 } ^ { 2 } } { 1 6 \lambda ^ { 2 } } + \frac { s ^ { 2 } } { W ^ { 2 } } \leq 3 . } \end{array}\tag{10}
$$

For the same reasons as in the proof of Lemma 16, this program has the same optimal value as the maximization above, and the arm of every optimal solution maximizes $U _ { V , \lambda }$ . It has six quadratic constraints and a nonempty feasible set. Its last constraint is implied by the others and defines a bounded ellipsoid in all the variables, and its coeficients are rational because the entries of $V , a ,$ and b and the number λ are. So it satisfies the hypotheses of Theorem 9. □

Theorem 9 only gives an approximate solution, allowing errors in both the objective and the constraints. As Lemma 20 did for the score of Algorithm 1, the next lemma shows that we can still find an arm whose score is within any $\kappa \in ( 0 , 1 )$ of the maximum.

Lemma 24. Let $V \succeq I _ { q }$ have rational entries, let $\lambda \geq 1$ be an integer, and let $\kappa \in ( 0 , 1 )$ be rational. There is an algorithm that computes an $a r m \textit { x } \in \textit { X }$ with rational coordinates and $U _ { V , \lambda } ( x ) \ge \operatorname* { m a x } _ { x ^ { \prime } \in X } U _ { V , \lambda } ( x ^ { \prime } ) - \kappa$ . Its running time is polynomial in $d _ { X } , d _ { D } ,$ log $\lambda , \log ( 1 / \kappa )$ , and the bit lengths of V, a, and b.

Proof. As in the proof of Lemma 20, we may assume that κ is a power of 2. Let $K : = 1 2 + 2 \| a \| _ { 1 } +$ $8 \lambda + 6 \operatorname { t r } V$ . We run the algorithm of Theorem 9 on the program in Equation 10 with tolerance $\varepsilon : = ( \kappa / ( K + 1 ) ) ^ { 2 }$ . It returns a point $( \boldsymbol { \hat { x } } , \boldsymbol { \hat { z } } , \boldsymbol { \hat { s } } )$ with rational coordinates that violates each constraint by at most ε and whose objective value is at least the optimum minus ε. We repair this point as in the proof of Lemma 18. We scale xˆ into the unit ball and zˆ into the ball of radius 4λ, round their coordinates toward zero to the grid whose spacing is the largest power of 2 that is at most $\sqrt { \varepsilon } / q$ and choose a rational s with $\| b + z _ { D } \| _ { 2 } \leq s \leq \| b + z _ { D } \| _ { 2 } + \sqrt { \varepsilon }$ . This gives x and z with rational coordinates, $\| x - { \hat { x } } \| _ { 2 } \leq 2 { \sqrt { \varepsilon } } .$ , and $\| z - \hat { z } \| _ { 2 } \leq 2 \sqrt { \varepsilon }$ . Since $\| b + z _ { D } \| _ { 2 } \leq \| b \| _ { 1 } + 4 \lambda$ and $\sqrt { \varepsilon } \leq 1$ , the point $( x , z , s )$ satisfies every constraint of Equation 10.

The repair lowers the objective by at most $K { \sqrt { \varepsilon } }$ . The terms $a ^ { T } x$ and $z _ { \mathrm { 0 } }$ change by at most $2 \| a \| _ { 1 } \sqrt { \varepsilon }$ and $2 \sqrt \varepsilon$ . Since $\| z \| _ { 2 } \leq 4 \lambda$ and $\| \hat { x } \| _ { 2 } \leq 2$ , the term $z _ { X } ^ { T } x$ changes by at most $( 8 \lambda + 4 ) \sqrt \varepsilon$ As in the proof of Lemma 18, $s - \hat { s } \leq 6 \sqrt { \varepsilon }$ . Finally, $\| \hat { z } \| _ { 2 } \leq 4 \lambda + 1$ and the largest eigenvalue of V is at most tr V, so

$$
\frac { \left| z ^ { T } V z - \hat { z } ^ { T } V \hat { z } \right| } { 4 \lambda } = \frac { \left| ( z - \hat { z } ) ^ { T } V ( z + \hat { z } ) \right| } { 4 \lambda } \leq \frac { 2 \sqrt { \varepsilon } \operatorname { t r } V \left( 8 \lambda + 1 \right) } { 4 \lambda } \leq 6 \operatorname { t r } V \sqrt { \varepsilon } .
$$

Adding these bounds proves the claim.

Write ℓ for the objective value of $( x , z , s )$ , which we compute exactly. By Lemma 23, the optimum of the program is ma $\mathrm { x } _ { x ^ { \prime } \in X } U _ { V , \lambda } ( x ^ { \prime } )$ , so

$$
\ell \geq \operatorname* { m a x } _ { x ^ { \prime } \in X } U _ { V , \lambda } ( x ^ { \prime } ) - \varepsilon - K \sqrt { \varepsilon } \geq \operatorname* { m a x } _ { x ^ { \prime } \in X } U _ { V , \lambda } ( x ^ { \prime } ) - \kappa .
$$

On the other hand, $s \geq \| b + z _ { D } \| _ { 2 }$ and $\| z \| _ { 2 } \leq 4 \lambda$ , so Lemma 22 gives $\ell \leq U _ { V , \lambda } ( x )$

For the running time, the program has $2 d _ { X } + d _ { D } + 2$ variables and six constraints. Its coeficients are the entries of $V , a ,$ and b and numbers computed from λ and $\| \boldsymbol { b } \| _ { 1 }$ . By Theorem 9, the solver runs in time polynomial in this input size and $\log ( 1 / \varepsilon ) = 2 \log ( ( K + 1 ) / \kappa )$ , which is polynomial in $\log ( 1 / \kappa )$ , log $\lambda , q .$ and the bit lengths of V and a. As in the proof of Lemma 18, the repair and the computation of ℓ take time polynomial in the same quantities and in the bit length of the solver’s output. □

At the start of each block, we choose the arm with the algorithm of Lemma 24, with $\kappa : = 1 / T$ This choice makes $\log ( 1 / \kappa ) = \log T$ in the running time and, as we show next, adds at most 1 to the regret.

The proof of Theorem 21 uses the rule for choosing arms only through the proof of Lemma 10, and that proof uses it only in the step $U _ { V , \lambda } ( x ) \geq V ^ { \star } - c ^ { 2 } / ( 4 \lambda )$ . For an arm whose score is within κ of the maximum, the same step gives $\begin{array} { r } { U _ { V , \lambda } ( x ) \geq V ^ { \star } - c ^ { 2 } / ( 4 \lambda ) - \kappa . } \end{array}$ . The other steps, namely the event of Lemma 13, the bound $J \le J _ { \mathrm { m a x } } ^ { \prime } .$ , and Equation 7, depend only on the rules for updating V and ending a block, which we have not changed. So the regret of a block of length n grows by at most nκ, and the high-probability bound of Theorem 21 grows by $T \kappa = 1$ . The expected regret therefore has the same rate.

Finally, we bound the running time of the whole learner. As in Appendix A.1, we assume that the observations are rational, with bit length polynomial in |ϕ| and $T .$ . Every matrix maintained by the learner satisfies $I _ { q } \preceq V \preceq ( 1 + 3 T ) I _ { q }$ , since $V$ starts at $I _ { q } ,$ , a completed block of length n adds the positive semidefinite matrix nw $\overline { { w } } ^ { T }$ of operator norm at most $3 n .$ , and the block lengths add up to at most T. So tr $V \leq q ( 1 + 3 T )$ , and the grid spacings used in Lemma 24 are powers of 2 whose exponents are polynomial in $q ,$ log $T .$ , and the bit length of $a .$ . Hence every stored matrix has rational entries of polynomial bit length, and matrix inversion and the test $n p _ { V } ( x , \overline { { y } } _ { n } ) \geq 1$ after each observation can be performed exactly in polynomial time. Each block calls the algorithm of Lemma 24 once, and there are at most $J _ { \operatorname* { m a x } } ^ { \prime } + 1$ blocks. With $\lambda = \lceil \sqrt { T } \rceil$ and $\kappa = 1 / T .$ , each call takes time polynomial in |ϕ| and T. Together with the regret bound above, this proves Theorem 1.

## C NP-hardness results

In this section we show that several natural variations of the linear robust bandits problem are NP-hard. Thus in some sense, the problem is the hardest problem that’s still tractable.

## C.1 Planning from learning

We first show that eficient planning can be reduced to eficient learning. Most proofs below show that planning is already NP-hard. Due to the reduction from planning to learning, this shows that learning is also NP-hard.

Lemma 25. Fix a robust bandits instance $\mathcal { M } _ { \phi } = ( X , D , H , r )$ from a parameterized family and a known hypothesis $h \in H$ whose description has encoding length $| h |$ . Suppose that, for every arm x and every $\eta \in ( 0 , 1 )$ , there is a distribution $Q _ { \eta } \left( x \right) \in h \left( x \right)$ satisfying $\begin{array} { r } { { \mathbb E } _ { y \sim Q _ { \eta } ( x ) } \left[ r \left( x , y \right) \right] \le v _ { h } \left( x \right) + \eta } \end{array}$ from which one can sample in time polynomial in $| \phi | , \ | h |$ , the encoding length of x, and $1 / \eta$ Suppose also that, for some fixed $\alpha > 0$ and polynomial $P ,$ a polynomial-time learner guarantees $R _ { T } \leq P \left( | \phi | \right) T ^ { 1 - \alpha }$ . Then, for every $\varepsilon \in ( 0 , 1 )$ , one can compute in time poly $\begin{array} { r } { \left( | \phi | , | h | , \frac { 1 } { \varepsilon } \right) } \end{array}$ an arm xˆ such that with probability at least $2 / 3 , V _ { h } - v _ { h } \left( \hat { x } \right) \le \varepsilon$

Proof. Fix $\varepsilon \in ( 0 , 1 )$ , set $\begin{array} { r } { \eta : = \frac { \varepsilon } { 6 } } \end{array}$ , and simulate the learner on $\mathcal { M } _ { \phi }$ for

$$
T : = \left\lceil \operatorname* { m a x } \left( 1 , \left( 6 \frac { P \left( \left| \phi \right| \right) } { \varepsilon } \right) ^ { \frac { 1 } { \alpha } } \right) \right\rceil
$$

rounds, with true hypothesis h. Whenever the learner plays $x _ { t }$ , let the adversary choose $P _ { t } : = Q _ { \eta } \left( x _ { t } \right)$ , and draw the outcome $y _ { t }$ shown to the learner using the sampler. This is allowed because $Q _ { \eta } \left( x _ { t } \right) \in h \left( x _ { t } \right)$ . Finally, choose t<sup>ˆ</sup> uniformly from $\{ 1 , \ldots , T \}$ and return $\hat { x } : = x _ { \hat { t } }$

The guarantee on $Q _ { \eta }$ gives $V _ { h } - v _ { h } \left( x _ { t } \right) \leq V _ { h } - \mathbb { E } _ { y \sim P _ { t } } \left[ r \left( x _ { t } , y \right) \right] .$ + η in every round. The expected reward of round t is $\mathbb { E } \left[ r \left( x _ { t } , y _ { t } \right) \right] = \mathbb { E } \left[ \mathbb { E } _ { y \sim P _ { t } } \left[ r \left( x _ { t } , y \right) \right] \right]$ , so averaging over $\hat { t }$ and using the definition of $R _ { T }$ for this adversary policy gives

$$
\mathbb { E } \left[ V _ { h } - v _ { h } \left( \hat { x } \right) \right] = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ V _ { h } - v _ { h } \left( x _ { t } \right) \right] \leq \frac { R _ { T } } { T } + \eta \leq P \left( \left| \phi \right| \right) T ^ { - \alpha } + \eta \leq \frac { \varepsilon } { 3 } .
$$

The gap $V _ { h } - v _ { h } \left( \hat { x } \right)$ is nonnegative, so Markov’s inequality shows that it exceeds ε with probability at most $\textstyle { \frac { 1 } { 3 } }$ . The horizon and the learner’s output lengths are polynomial in |ϕ| and $1 / \varepsilon ,$ since α is fixed. The assumed sampler therefore makes the simulation polynomial in $| \phi | , | h |$ , and $1 / \varepsilon . \qquad \sqcup$

In all our applications, the reward is linear and the hypothesis constrains only the expected outcome. That is, $h ( x ) = \{ P \in \Delta D : \mathbb { E } _ { y \sim P } [ y ] \in K _ { h } ( x ) \}$ for some set $K _ { h } ( x ) \subseteq D$ . Then it sufices to compute an outcome $y \in K _ { h } ( x )$ with $r ( x , y ) \leq v _ { h } ( x ) + \eta .$ , since the point mass at $y$ lies in $h ( x )$ and has expected reward $r ( x , y )$

Below we show that the “smoothness” of X and D are crucial for eficiency. It is easy to land in the NP-hard territory if we make either X or D a convex shape with edges and corners.

## C.2 D = simplex, X = ball

First we show that if, instead of a Euclidean ball, we allow D to be a simplex, then even known hypothesis planning becomes NP-hard. Then, using the planning to learning reduction (Lemma 25), we conclude that learning is NP-hard.

We take D to be the probability simplex $\{ y \in \mathbb { R } ^ { d _ { D } } : y \ge 0 , ~ \textstyle \sum _ { i } y _ { i } = 1 \}$ , which is determined by $d _ { D }$ . So the parameters $\phi$ are those of Section 2.1 without the centre and radius of D.

Theorem 26. Consider the setting of Section 2.1 with D a simplex instead of a ball. Suppose that, for some fixed $\alpha > 0$ and polynomial $P ,$ one randomized polynomial-time learner guarantees $R _ { T } \leq P ( | \phi | ) T ^ { 1 - \alpha }$ for all parameters ϕ and every true hypothesis $h ^ { \star } \in H$ . Then $\mathrm { N P \subseteq B P P }$ . If a deterministic learner achieves the same guarantee, then $\mathrm { P } = \mathrm { N P }$

Proof. We reduce from MAX-CUT, which asks, given a graph G and an integer $k ,$ whether some partition of the vertices into two sides has at least k edges between the sides. It is NP-complete, even for unweighted graphs [45, 32]. Let $( G , k )$ be an instance, where $G = ( V , E )$ has n vertices and $m \geq 1$ edges and $k \in \{ 1 , \ldots , m \}$ . Orient the edges arbitrarily and let $B _ { G } \in \mathbb { R } ^ { m \times n }$ be the resulting edge-vertex incidence matrix, i.e., if $e = \left\{ i , j \right\}$ is oriented from $j$ to $i ,$ then row e has 1 in column $i ,$ −1 in column $j ,$ , and zeros elsewhere. Put

$$
X : = \left\{ x \in \mathbb { R } ^ { m } : \left\| x \right\| _ { 2 } \leq 1 \right\} .
$$

Write an outcome as $\boldsymbol { y } = ( p , q , s ) \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { n } \times \mathbf { \mathbb { R } } ^ { n } \times \mathbf { \mathbb { \Lambda } }$ R and take

$$
D : = \left\{ ( p , q , s ) : p _ { i } \geq 0 , q _ { i } \geq 0 \mathrm { f o r } i = 1 , \ldots , n , s \geq 0 , \sum _ { i = 1 } ^ { n } ( p _ { i } + q _ { i } ) + s = 1 \right\} .
$$

Take $d _ { C } = n$ , the cutof $S : = 1 / ( 1 + 4 \lceil \sqrt { n } \rceil )$ , and $C _ { \mathrm { m a x } } : = 1$ , and let H be the hypothesis class of Section 2.1 with these sets X and D. For a scale $\gamma > 0$ to be chosen later, define the true hypothesis $h _ { G }$ by

$$
K _ { h _ { G } } \left( x \right) : = \left\{ \left( p , q , s \right) \in D : p - q = \gamma B _ { G } ^ { T } x \right\} .
$$

Its coeficients are $B _ { h _ { G } } = - \gamma B _ { G } ^ { T } , C _ { h _ { G } } = ( I _ { n } ~ - I _ { n } ~ 0 )$ , and $d _ { h _ { G } } = 0$ , as in Equation 1. Let the reward be

$$
r \left( x , ( p , q , s ) \right) : = \sum _ { i = 1 } ^ { n } \left( p _ { i } + q _ { i } \right) .
$$

This reward takes values in [0, 1] on $D ,$ so $C _ { r } \leq C _ { \operatorname* { m a x } }$ . For fixed $n , m .$ , the parameters $\phi ,$ and hence the sets X and $D ,$ the reward, and the class $H ,$ do not depend on the graph. Only the true hypothesis $h _ { G }$ does. We next show that $( G , k )$ is a yes-instance of MAX-CUT if and only if there is an arm $x \in X$ satisfying

$$
v _ { h _ { G } } \left( x \right) \geq 2 \gamma \sqrt { k } .
$$

A vector $\sigma \in \{ - 1 , 1 \} ^ { n }$ represents a partition of the vertices of G into the two sides $\{ i : \sigma _ { i } = 1 \}$ and $\{ i : \sigma _ { i } = - 1 \}$ . Let $c _ { G } \left( \sigma \right)$ denote the number of edges between these sides.

Lemma 27. For every $\sigma \in \{ - 1 , 1 \} ^ { n }$

$$
\| B _ { G } \sigma \| _ { 2 } = 2 \sqrt { c _ { G } \left( \sigma \right) } .
$$

Proof. Fix $\sigma \in \{ - 1 , 1 \} ^ { n }$ . For every edge $e = \{ i , j \}$

$$
( B _ { G } \sigma ) _ { e } ^ { 2 } = ( \sigma _ { i } - \sigma _ { j } ) ^ { 2 } = \left\{ \begin{array} { l l } { { 4 } } & { { \mathrm { i f } \ \sigma _ { i } \neq \sigma _ { j } } } \\ { { 0 } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. .
$$

Summing over $e \in E$ and taking the square root proves the identity.

For the forward implication, suppose $( G , k )$ is a yes-instance. Choose σ such that $c _ { G } \left( \sigma \right) \geq k$ and define the arm

$$
x _ { \sigma } : = B _ { G } \frac { \sigma } { \| B _ { G } \sigma \| _ { 2 } } .
$$

Since $c _ { G } \left( \sigma \right) \geq k \geq 1$ , Lemma 27 shows that $x _ { \sigma }$ is well-defined. Moreover, we get the following inequality.

Lemma 28. For every $\sigma \in \{ - 1 , 1 \} ^ { n }$ such that $B _ { G } \sigma \neq 0$ , we have $\left\| B _ { G } ^ { T } x _ { \sigma } \right\| _ { 1 } \geq \| B _ { G } \sigma \| _ { 2 }$ . Moreover, equality holds when σ maximizes $\| B _ { G } \sigma \| _ { 2 }$ over $\{ - 1 , 1 \} ^ { n }$ ; in particular, equality holds for at least one σ.

Proof. Choosing the sign of each coordinate gives

$$
\left\| B _ { G } ^ { T } x _ { \sigma } \right\| _ { 1 } = \operatorname* { m a x } _ { \tau \in \{ - 1 , 1 \} ^ { n } } \tau ^ { T } B _ { G } ^ { T } x _ { \sigma } = \operatorname* { m a x } _ { \tau \in \{ - 1 , 1 \} ^ { n } } x _ { \sigma } ^ { T } B _ { G } \tau .
$$

Taking $\tau = \sigma$ yields $\left. B _ { G } ^ { T } x _ { \sigma } \right. _ { 1 } \geq x _ { \sigma } ^ { T } B _ { G } \sigma = \left. B _ { G } \sigma \right. _ { 2 }$ . Now let $\sigma ^ { \star }$ maximize $\| B _ { G } \sigma \| _ { 2 }$ . Since $m \geq 1$ this maximum is positive, so $x _ { \sigma ^ { \star } }$ is well-defined. For every $\tau \in \{ - 1 , 1 \} ^ { n }$ , Cauchy–Schwarz gives

$$
x _ { \sigma ^ { \star } } ^ { T } B _ { G } \tau \leq \| x _ { \sigma ^ { \star } } \| _ { 2 } \| B _ { G } \tau \| _ { 2 } \leq \| B _ { G } \sigma ^ { \star } \| _ { 2 } .
$$

Taking the maximum over τ gives the reverse inequality, proving equality.

The lemma above gives us a way to link properties of cuts with properties of arms. The missing ingredient is to show that the value of an arm has something to do with $\left\| B _ { G } ^ { T } x \right\| _ { 1 }$ . We show this in the next two lemmas.

By construction, $x _ { \sigma } \in X$ . Write $K ( x ) : = K _ { h _ { G } } ( x )$ for the feasible outcomes under this true hypothesis.

Lemma 29. For every $x \in X$ , the vector $t : = \gamma B _ { G } ^ { T } x$ satisfies $\| t \| _ { 1 } \leq 2 \gamma \sqrt { m }$ . Therefore, $\begin{array} { r } { i f \gamma \leq \frac { 1 } { 2 \sqrt { m } } } \end{array}$ then $\| t \| _ { 1 } \leq 1$

Proof. Each edge coordinate $x _ { e }$ contributes $x _ { e }$ and $- { \boldsymbol { x } } _ { e }$ to two coordinates of $B _ { G } ^ { T } x$ before contributions from diferent edges are added. The triangle inequality therefore gives

$$
\left\| t \right\| _ { 1 } = \gamma \Big \| B _ { G } ^ { T } x \Big \| _ { 1 } \leq 2 \gamma \| x \| _ { 1 } \leq 2 \gamma \sqrt { m } \| x \| _ { 2 } \leq 2 \gamma \sqrt { m } ,
$$

where the second inequality is Cauchy–Schwarz and the third uses $\| x \| _ { 2 } \leq 1$

Lemma 30. Suppose $\begin{array} { r } { \gamma \leq \frac { 1 } { 2 \sqrt { m } } } \end{array}$ . For each arm $x \in X$ , set $t : = \gamma B _ { G } ^ { T } x$ . Then $K \left( x \right)$ is nonempty, and $\begin{array} { r } { v _ { h _ { G } } \left( x \right) = \operatorname* { m i n } _ { y \in K \left( x \right) } r \left( x , y \right) = \left\| t \right\| _ { 1 } = \gamma \left\| B _ { G } ^ { T } x \right\| _ { 1 } } \end{array}$ . Moreover, the minimum is attained at the outcome $\boldsymbol { y } = \left( p , q , s \right)$ given by p<sub>i</sub> := max (t<sub>i</sub>, 0), q<sub>i</sub> := max $\left( - t _ { i } , 0 \right)$ , and $s : = 1 - \| t \| _ { 1 }$

Proof. Fix $x \in X$ and set $t : = \gamma B _ { G } ^ { T } x$ . Lemma 29 gives $\| t \| _ { 1 } \leq 1$ . For the outcome in the statement, $p , q , s$ are nonnegative, $p - q = t ,$ and

$$
\sum _ { i = 1 } ^ { n } { ( p _ { i } + q _ { i } ) } + s = \left\| t \right\| _ { 1 } + 1 - \left\| t \right\| _ { 1 } = 1 .
$$

Thus this outcome belongs to $K \left( x \right)$ . For any $( p , q , s ) \in K \left( x \right)$ , nonnegativity gives $p _ { i } + q _ { i } \geq$ $\left| p _ { i } - q _ { i } \right| = \left| t _ { i } \right|$ for every $i ,$ and hence $r \left( x , ( p , q , s ) \right) \geq \left\| t \right\| _ { 1 }$ . The outcome in the statement attains equality in every coordinate, so its reward is the minimum. □

Given G and an arm x, we can therefore compute both its value and a minimizing outcome, as needed for Lemma 25.

For the final step, apply Lemma 30, Lemma 28, and Lemma 27 to obtain

$$
v _ { h _ { G } } \left( x _ { \sigma } \right) = \gamma \Big \| B _ { G } ^ { T } x _ { \sigma } \Big \| _ { 1 } \geq \gamma \| B _ { G } \sigma \| _ { 2 } = 2 \gamma \sqrt { c _ { G } \left( \sigma \right) } \geq 2 \gamma \sqrt { k } .
$$

For the reverse implication, suppose there is an arm $x \in X$ satisfying $v _ { h _ { G } } \left( x \right) \geq 2 \gamma \sqrt { k }$ . By $\ell _ { 1 } - \ell _ { \infty }$ duality, choose $\sigma \in \{ - 1 , 1 \} ^ { n }$ such that

$$
\left\| B _ { G } ^ { T } x \right\| _ { 1 } = \sigma ^ { T } B _ { G } ^ { T } x .
$$

Using Lemma 30 and Lemma 27,

$$
2 \gamma \sqrt { k } \leq v _ { h _ { G } } \left( x \right) = \gamma \Big \| B _ { G } ^ { T } x \Big \| _ { 1 } = \gamma x ^ { T } B _ { G } \sigma \leq \gamma \| x \| _ { 2 } \| B _ { G } \sigma \| _ { 2 } \leq 2 \gamma \sqrt { c _ { G } \left( \sigma \right) } .
$$

Since $\gamma > 0$ , this implies $c _ { G } \left( \sigma \right) \geq k .$ , so $( G , k )$ is a yes-instance. This proves the claimed equivalence.

To apply Lemma 25, we need to show that approximate planning is hard. Although the lemma returns only an approximate optimizer, this is suficient because the maximum cut size $M _ { G }$ is an integer between 0 and $m ,$ , and $V _ { h _ { G } } = 2 \gamma \sqrt { M _ { G } }$ . Taking $\begin{array} { r } { \gamma : = \frac { 1 } { 2 m n } } \end{array}$ ensures that every $K _ { h _ { G } } \left( x \right)$ is nonempty and that the values of $V _ { h _ { G } }$ for consecutive values of $M _ { G }$ are separated by at least ${ \frac { 1 } { 2 m ^ { 2 } n } } .$ Thus an additive approximation with error $\begin{array} { r } { \varepsilon : = \frac { 1 } { 6 m ^ { 2 } n } } \end{array}$ distinguishes $M _ { G } \geq k$ from $M _ { G } \leq k - 1$

Both |ϕ| and $| h _ { G } |$ are polynomial in $n , m$ . Run the learner on this instance, using Lemma 30 to supply minimizing outcomes for $h _ { G }$ . Lemma 25 then returns an approximately optimal arm in polynomial time. We can therefore decide MAX-CUT by comparing the returned arm’s value with the midpoint between $2 \gamma \sqrt { k - 1 }$ and $2 \gamma \sqrt { k }$ , computed to suficient precision. Hence Lemma 25 implies that a randomized polynomial-time learner would place MAX-CUT in BPP, and therefore that NP ⊆ BPP. A deterministic learner would instead imply P = NP. Finally, the constructed hypothesis satisfies the transversality condition with constant $( 1 + 4 { \sqrt { n } } ) ^ { - 1 } \geq S$ , so $h _ { G } \in H$ . To see this, fix x and put $t = \gamma B _ { G } ^ { T } x , \mathrm { s o } \| t \| _ { 1 } \leq 2 \gamma \sqrt { m } \leq 1 / 2$ by Lemma 29 and the choice of γ. For any u with $C _ { h _ { G } } u = t _ { \colon }$ , let v be a nearest point in $D ,$ and set $e = t - C _ { h _ { G } } v$ and $a = \| e \| _ { 1 } \leq \sqrt { 2 n } \| u - v \| _ { 2 }$ If $a = 0$ , then $v \in K ( x )$ . Otherwise, $\| t + e / ( 2 a ) \| _ { 1 } \leq 1$ , so there is a $w \in D$ with $C _ { h _ { G } } w = t + e / ( 2 a )$ The point $v ^ { \prime } = ( v + 2 a w ) / ( 1 + 2 a )$ lies in $K ( x )$ . Since D has diameter $\sqrt { 2 }$ , we obtain

$$
\mathrm { d i s t } ( u , K ( x ) ) \leq \| u - v \| _ { 2 } + \frac { 2 a \sqrt { 2 } } { 1 + 2 a } \leq ( 1 + 4 \sqrt { n } ) \mathrm { d i s t } ( u , D ) ,
$$

which proves the condition.

## C.3 A Euclidean outcome ball and a polytope of arms

We now keep the Euclidean outcome ball and the afine constraints of Section 2.1, but allow the arm set to be a polytope given by inequalities with rational coeficients. The parameters $\phi$ are those of Section 2.1, with these inequalities in place of the centre and radius of X. Planning is NP-hard even when the hypothesis is known and the arm set is the cube. The cube has only 2n defining inequalities, but $2 ^ { n }$ vertices.

Theorem 31. Consider the setting of Section 2.1 with X a polytope instead of a ball. If, for some fixed $\alpha > 0$ and polynomial P, one randomized polynomial-time learner guarantees $R _ { T } \leq P \left( \left| \phi \right| \right) T ^ { 1 - }$ α for all parameters ϕ and every true hypothesis $h ^ { \star } \in H$ , then $\mathrm { N P \subseteq B P P }$ . If a deterministic learner achieves the same regret guarantee, then $\mathrm { P } = \mathrm { N P }$

Proof. We again reduce from MAX-CUT. Let $G = ( V , E )$ have n vertices and $m \geq 1$ edges, and let $B _ { G } \in \mathbb { R } ^ { m \times n }$ be its edge-vertex incidence matrix, as in Appendix C.2. Put

$$
\gamma : = \frac { 1 } { 4 m } , \quad X : = [ - 1 , 1 ] ^ { n } .
$$

Write outcomes as $y = ( u , s ) \in \mathbb { R } ^ { m } \times \mathbb { R }$ and take

$$
D : = \left\{ ( u , s ) : \| u \| _ { 2 } ^ { 2 } + s ^ { 2 } \leq 1 \right\} .
$$

Take $d _ { X } = n , d _ { D } = m + 1 , d _ { C } = m$ , the cutof $\begin{array} { r } { S : = \frac { 1 } { 2 } } \end{array}$ , and $C _ { \mathrm { m a x } } : = 2$ , let H be the hypothesis class of Section 2.1 with these sets X and $D ,$ and let $r \left( x , ( u , s ) \right) : = s .$ , so that $C _ { r } ~ = ~ 2$ . The parameters $\phi ,$ and hence the sets X and $D ,$ the reward $^ { r , }$ and the class $H _ { \cdot }$ , depend only on $n , m$ , not on G. The graph determines only the true hypothesis. We let $h _ { G }$ be the hypothesis with coeficients $C _ { h _ { G } } = [ I _ { m } 0 ] , B _ { h _ { G } } = - \gamma B _ { G }$ , and $d _ { h _ { G } } = 0$ . These coeficients have total encoding length polynomial in $n , m ,$ and

$$
K _ { h _ { G } } \left( x \right) = \left\{ \left( u , s \right) \in D : u = \gamma B _ { G } x \right\} .
$$

Every $x \in X$ satisfies

$$
\| B _ { G } x \| _ { 2 } ^ { 2 } = \sum _ { \{ i , j \} \in E } \left( x _ { i } - x _ { j } \right) ^ { 2 } \leq 4 m ,
$$

so

$$
\| \gamma B _ { G } x \| _ { 2 } ^ { 2 } \leq \frac { 1 } { 4 m } \leq \frac { 1 } { 4 } .
$$

Thus $K _ { h _ { G } } \left( x \right)$ is nonempty for every $x \in X$ , so $h _ { G }$ belongs to H. Minimizing s over this set gives

$$
v _ { h _ { G } } \left( x \right) = - \sqrt { 1 - \gamma ^ { 2 } \| B _ { G } x \| _ { 2 } ^ { 2 } } .
$$

Set $q \left( x \right) : = \| B _ { G } x \| _ { 2 } ^ { 2 }$ . With every coordinate except $x _ { i }$ fixed, $q \left( x \right)$ is a convex quadratic in $x _ { i } ,$ so replacing $x _ { i }$ by one of the endpoints of [−1, 1] does not decrease q. Applying this replacement one coordinate at a time produces a sign vector $\sigma \in \{ - 1 , 1 \} ^ { n }$ with $q \left( \sigma \right) \geq q \left( x \right)$ . By Lemma 27, $q \left( \sigma \right) = 4 c _ { G } \left( \sigma \right)$ , where $c _ { G } \left( \sigma \right)$ is the number of edges between the two sides of $\sigma .$ . Consequently, the maximum cut size $M _ { G }$ satisfies

$$
\operatorname* { m a x } _ { x \in [ - 1 , 1 ] ^ { n } } q \left( x \right) = 4 M _ { G } .
$$

Since $v _ { h _ { G } } \left( x \right)$ is strictly increasing in $q \left( x \right)$ , if the maximum cut has size $k ,$ then $V _ { h _ { G } } = V _ { k }$ , where

$$
V _ { k } = - \sqrt { 1 - \frac { k } { 4 m ^ { 2 } } } .
$$

For $k = 1 , \ldots , m$ , consecutive possible values satisfy

$$
V _ { k } - V _ { k - 1 } = { \frac { \frac { 1 } { 4 m ^ { 2 } } } { \sqrt { 1 - { \frac { k - 1 } { 4 m ^ { 2 } } } } + \sqrt { 1 - { \frac { k } { 4 m ^ { 2 } } } } } } \geq { \frac { 1 } { 8 m ^ { 2 } } } .
$$

Hence an estimate of $V _ { h _ { G } }$ within

$$
\Delta : = \frac { 1 } { 3 2 m ^ { 2 } }
$$

identifies k after we compute the $m + 1$ candidate values to polynomially many bits. Thus approximating $V _ { h _ { G } }$ is NP-hard.

We now run the same learner for all graphs with n vertices and m edges, using $h _ { G }$ only to simulate its observations. To apply Lemma 25, we need, for every arm x and every $\eta \in ( 0 , 1 )$ , an outcome in $K _ { h _ { G } } \left( x \right)$ that has rational coordinates and whose reward is within $\eta$ of $v _ { h _ { G } } \left( x \right)$ . The outcome attaining $v _ { h _ { G } } \left( x \right)$ need not have rational coordinates, so we compute an approximation that does. For an arm x with rational coordinates, binary search gives, in time polynomial in $| \phi |$ $| h _ { G } |$ , the encoding length of $x ,$ and log $\left( { \frac { 1 } { \eta } } \right)$ , a rational $a \geq 0$ such that

$$
0 \leq \sqrt { 1 - \gamma ^ { 2 } q \left( x \right) } - a \leq \eta , \quad a ^ { 2 } \leq 1 - \gamma ^ { 2 } q \left( x \right) .
$$

The point $y : = ( \gamma B _ { G } x , - a )$ then lies in $K _ { h _ { G } } \left( x \right)$ , and its reward satisfies $0 \leq r \left( x , y \right) - v _ { h _ { G } } \left( x \right) \leq \eta$ So Lemma 25, applied with accuracy $\Delta$ , returns with probability at least $\frac { 2 } { 3 }$ an arm $\hat { x }$ with ${ { V } _ { { { h } _ { G } } } } - { { v } _ { { { h } _ { G } } } } \left( \hat { x } \right) \le \Delta$

Round the coordinates of $\hat { x }$ to endpoints without decreasing $q .$ If the maximum cut has size $k ,$ the resulting sign vector σ satisfies $v _ { h _ { G } } \left( \sigma \right) \geq V _ { k } - \Delta . \mathrm { ~ I f ~ } c _ { G } \left( \sigma \right) \leq k - 1$ , it would instead satisfy $\begin{array} { r } { v _ { h _ { G } } \left( \sigma \right) \leq V _ { k - 1 } \leq V _ { k } - \frac { 1 } { 8 m ^ { 2 } } < V _ { k } - \Delta } \end{array}$ . The rounded sign vector therefore encodes a maximum cut. The binary search, the simulation, and the rounding all take polynomial time, proving ${ \mathrm { N P } } \subseteq { \mathrm { B P P } }$ If the learner is deterministic, the same argument succeeds with certainty and gives $\mathrm { P } = \mathrm { N P }$

It remains to verify the transversality condition. For a fixed $x ,$ put

$$
a : = \| \gamma B _ { G } x \| _ { 2 } , \quad \rho : = \sqrt { 1 - a ^ { 2 } } \geq \frac { \sqrt { 3 } } { 2 } .
$$

The outcomes satisfying the afine constraint form the line $L _ { h _ { G } } \left( x \right)$ , whose intersection with D is $K _ { h _ { G } } \left( x \right)$ :

$$
L _ { h _ { G } } \left( x \right) = \left\{ \left( \gamma B _ { G } x , s \right) : s \in \mathbb { R } \right\} , \quad K _ { h _ { G } } \left( x \right) = \left\{ \left( \gamma B _ { G } x , s \right) : \left| s \right| \le \rho \right\} .
$$

If $p = ( \gamma B _ { G } x , s ) \in L _ { h _ { G } }$ (x) lies outside D and $t : = | s | > \rho$ , then

$$
\operatorname { d i s t } \left( p , K _ { h _ { G } } \left( x \right) \right) = t - \rho , \quad \operatorname { d i s t } \left( p , D \right) = \sqrt { a ^ { 2 } + t ^ { 2 } } - 1 .
$$

Factoring the second expression gives

$$
\frac { \operatorname { d i s t } \left( p , D \right) } { \operatorname { d i s t } \left( p , K _ { h _ { G } } \left( x \right) \right) } = \frac { t + \rho } { \sqrt { a ^ { 2 } + t ^ { 2 } } + 1 } \geq \rho \geq \frac { \sqrt { 3 } } { 2 } .
$$

To check the first inequality, multiply by the positive denominator, subtract $\rho$ from both sides, and square. This gives the equivalent inequality $\left( 1 - \rho ^ { 2 } \right) \left( t ^ { 2 } - \rho ^ { 2 } \right) \geq 0$ . Thus the transversality condition holds with constant ${ \frac { \sqrt { 3 } } { 2 } } \geq S$ , so $h _ { G } \in H$ □

## C.4 Trilinear constraints with Euclidean arm and outcome balls

In this section, we keep X and D Euclidean balls and the reward linear, but make the constraints more general. We now allow the coeficients of the outcome in the constraints to depend on the arm. The resulting constraints are still a special case of those in the linear robust bandits of Kosoy [51, Section 2.2]. We show that this change already makes planning NP-hard, even approximately.

We first recall Kosoy’s constraints. There, outcomes are vectors in a space $Y ,$ and D lies in a hyperplane $\{ y \in Y : y _ { 0 } = 1 \}$ given by one of the outcome coordinates. A hypothesis is specified by a parameter $z$ in a compact set $Z \subseteq \mathbb { R } ^ { d _ { Z } }$ , through a map

$$
F : \mathbb { R } ^ { d _ { X } } \times \mathbb { R } ^ { d _ { Z } } \times Y  W
$$

into a constraint space W. This map must be bilinear in $( z , y )$ , but it may depend on the arm in any continuous way. Writing $F _ { x , z } \left( y \right) : = F \left( x , z , y \right)$ , the hypothesis $h _ { z }$ allows the distributions whose expected value lies in the kernel of $F _ { x , z } \mathrm { : }$

$$
L _ { z } \left( x \right) : = \ker F _ { x , z } , \quad K _ { z } \left( x \right) : = L _ { z } \left( x \right) \cap D , \quad h _ { z } \left( x \right) : = \left\{ P \in \Delta D : \mathbb { E } _ { y \sim P } \left[ y \right] \in K _ { z } \left( x \right) \right\} .
$$

The constraints of Section 2.1 are of this form. To see this, add an outcome coordinate $y _ { 0 }$ that is fixed to one, write $z = ( B , C , d )$ , and set $F ( x , z , ( y _ { 0 } , y ) ) : = C y + ( B x + d ) y _ { 0 }$ . Then $K _ { z } ( x )$ is the set in Equation 1. In this map, the arm enters only through the term $( B x + d ) y _ { 0 }$ , so the coeficients C of the outcome do not depend on x.

We now let these coeficients depend on x by allowing $F$ to be linear in each of $x , z ,$ and $y$ separately, that is, trilinear. Such a map is still allowed in Kosoy’s framework. An instance now consists of a Euclidean ball X, a Euclidean ball $D$ in the hyperplane $y _ { 0 } = 1$ , a linear reward $^ { r , }$ a Euclidean ball $Z$ of parameters, and a trilinear map $F .$ . As in Section 2.1, the parameters $\phi$ of the family are rational data, namely the centres and radii of $X , D$ , and $Z ,$ , the coeficients of $r$ and $F ,$ , the cutof $S ,$ and the bound $C _ { \mathrm { m a x } } \geq C _ { r } ,$ with $S ^ { - 1 }$ and $C _ { \mathrm { m a x } }$ integers given in unary. Since the reward is linear, the value functions of $h _ { z }$ are

$$
v _ { z } \left( x \right) : = v _ { h _ { z } } \left( x \right) = \operatorname* { m i n } _ { y \in K _ { z } \left( x \right) } r \left( x , y \right) , \quad V _ { z } : = \operatorname* { m a x } _ { x \in X } v _ { z } \left( x \right) .
$$

The hypothesis class H consists of the $h _ { z }$ with $z \in Z$ that satisfy the transversality condition of Section 2.1 with the cutof S. For these constraints, the condition asks that dist $( p , D ) \geq$ $S \operatorname { d i s t } ( p , K _ { z } ( x ) )$ for every arm x and every point $p \in L _ { z } ( x )$ outside D with $p _ { 0 } = 1$ . The proof also checks Kosoy’s requirements that every $K _ { z } ( x )$ is nonempty and every $F _ { x , z }$ is onto.

Theorem 32. Consider the setting of Section 2.1 with the afine constraints replaced by the trilinear constraints described above. If, for some fixed $\alpha > 0$ and polynomial $P _ { \varepsilon }$ one randomized polynomialtime learner guarantees $R _ { T } \leq P \left( | \phi | \right) T ^ { 1 - \alpha }$ for all parameters ϕ and every true hypothesis $h ^ { \star } \in H$ then ${ \mathrm { N P } } \subseteq { \mathrm { B P P } }$ . If a deterministic learner achieves the same regret guarantee, then $\mathrm { P } = \mathrm { N P }$

Let $G = ( V , E )$ be an undirected graph with $V = \{ 1 , \ldots , n \}$ and $E = \{ e _ { 1 } , \ldots , e _ { m } \}$ , where $m \geq 1$ and $e _ { \ell } = \{ i _ { \ell } , j _ { \ell } \}$ . Put $d : = n + m$ . For $q = ( u , w ) \in \mathbb R ^ { n } \times \mathbb R ^ { m }$ , define the homogeneous cubic

$$
p _ { G } \left( q \right) : = \sum _ { \ell = 1 } ^ { m } u _ { i _ { \ell } } u _ { j _ { \ell } } w _ { \ell } .
$$

Lemma 33. $I f \omega \left( G \right)$ is the clique number of G, then

$$
\operatorname* { m a x } _ { \| q \| _ { 2 } = 1 } p _ { G } ( q ) ^ { 2 } = \frac { 2 } { 2 7 } \left( 1 - \frac { 1 } { \omega ( G ) } \right) .
$$

Proof. Write $\rho : = \| u \| _ { 2 }$ and $\sigma : = \| w \| _ { 2 }$ . For fixed u and $\sigma ,$ Cauchy–Schwarz gives

$$
\operatorname* { m a x } _ { \| w \| _ { 2 } = \sigma } p _ { G } \left( u , w \right) = \sigma \sqrt { \sum _ { \ell = 1 } ^ { m } u _ { i _ { \ell } } ^ { 2 } u _ { j _ { \ell } } ^ { 2 } } .
$$

Write $u = \rho s$ with $\| s \| _ { 2 } = 1$ and set $\pi _ { i } : = s _ { i } ^ { 2 }$ . The Motzkin–Straus theorem [61] gives

$$
\operatorname* { m a x } _ { \| s \| _ { 2 } = 1 } \sum _ { \substack { \{ i , j \} \in E } } s _ { i } ^ { 2 } s _ { j } ^ { 2 } = \operatorname* { m a x } _ { \substack { \pi _ { i } \geq 0 , \sum _ { i } \pi _ { i } = 1 } } \sum _ { \substack { \{ i , j \} \in E } } \pi _ { i } \pi _ { j } = \frac { 1 } { 2 } \left( 1 - \frac { 1 } { \omega \left( G \right) } \right) .
$$

Finally,

$$
\operatorname* { m a x } _ { \rho ^ { 2 } + \sigma ^ { 2 } = 1 , \rho \geq 0 , \sigma \geq 0 } \rho ^ { 2 } \sigma = \frac { 2 } { 3 \sqrt { 3 } } .
$$

Substituting these bounds into the first display and squaring proves the claim. Since $p _ { G }$ is odd, its maximum equals its maximum absolute value. □

Proof of Theorem 32. We reduce from CLIQUE. Given a graph $G$ and an integer $k ,$ , we construct an instance in which X is a full-dimensional Euclidean ball, D is a Euclidean ball in the hyperplane $y _ { 0 } = 1$ , and $Z = [ 1 , 2 ]$ . Every set $K _ { z } ( x )$ is a single point that does not depend on $z ,$ and the inequality of the transversality condition holds with a constant greater than 0.88, even for the points of $L _ { z } ( x )$ with $p _ { 0 } \neq 1$ . So we can take the cutof $\begin{array} { r } { S : = \frac { 1 } { 2 } } \end{array}$ , and every $h _ { z }$ belongs to H. We also take $C _ { \mathrm { m a x } } : = 2$ . Nevertheless, approximating $V _ { z }$ to inverse-polynomial additive accuracy decides whether G has a k-clique, and Lemma 25 turns a learner with the assumed regret into such an approximation.

We may restrict CLIQUE to instances with $m \geq 1$ and $3 ~ \leq ~ k ~ \leq ~ n$ , since we can decide the other cases in polynomial time. Given $( G , k )$ , use the cubic $p _ { G }$ above and write an arm as $\boldsymbol { x } = ( x _ { 0 } , \xi ) \in \mathbb { R } \times \mathbb { R } ^ { d }$ . Take

$$
X : = \left\{ ( x _ { 0 } , \xi ) : \left( x _ { 0 } - { \frac { 5 } { 3 } } \right) ^ { 2 } + \| \xi \| _ { 2 } ^ { 2 } \leq 1 \right\} .
$$

This is a full-dimensional Euclidean ball and $\begin{array} { r } { x _ { 0 } \geq \frac { 2 } { 3 } } \end{array}$ throughout X. The ratio $\begin{array} { r } { q : = \frac { \xi } { x _ { 0 } } } \end{array}$ ranges over exactly the ball $\begin{array} { r } { \| q \| _ { 2 } \le \frac { 3 } { 4 } } \end{array}$ . Indeed,

$$
1 - \left( x _ { 0 } - \frac { 5 } { 3 } \right) ^ { 2 } - \frac { 9 } { 1 6 } x _ { 0 } ^ { 2 } = - \frac { \left( 1 5 x _ { 0 } - 1 6 \right) ^ { 2 } } { 1 4 4 } \leq 0 ,
$$

so every ratio has norm at most ${ \frac { 3 } { 4 } } .$ . Conversely, for any such $q ,$ the choice $\textstyle x _ { 0 } = { \frac { 1 6 } { 1 5 } }$ and $\xi = x _ { 0 } q$ belongs to $X$

Let $Z : = [ 1 , 2 ]$ , and take the outcome and constraint spaces to be

$$
Y : = \mathbb { R } \times \mathbb { R } ^ { d } \times \mathbb { R } ^ { d \times d } \times \mathbb { R } ^ { d \times d \times d } ,
$$

$$
W : = \mathbb { R } ^ { d } \times \mathbb { R } ^ { d \times d } \times \mathbb { R } ^ { d \times d \times d } ,
$$

and write $y = ( y _ { 0 } , \eta , \Theta , \Xi ) \in \boldsymbol { Y }$ . We fix $y _ { 0 } = 1$ and take the remaining coordinates to lie in the Euclidean unit ball:

$$
D : = \left\{ ( 1 , \eta , \Theta , \Xi ) : \| \eta \| _ { 2 } ^ { 2 } + \| \Theta \| _ { F } ^ { 2 } + \| \Xi \| _ { F } ^ { 2 } \leq 1 \right\} .
$$

For $x = ( x _ { 0 } , \xi ) , z \in \mathbb { R }$ , and $y \in Y$ , define $F \left( x , z , y \right)$ coordinatewise by

$$
F _ { i } \left( x , z , y \right) : = z \left( x _ { 0 } \eta _ { i } - \frac { 1 } { 2 } \xi _ { i } y _ { 0 } \right) ,
$$

$$
F _ { i j } \left( x , z , y \right) : = z \left( x _ { 0 } \Theta _ { i j } - \xi _ { i } \eta _ { j } \right) ,
$$

$$
F _ { i j \ell } \left( x , z , y \right) : = z \left( x _ { 0 } \Xi _ { i j \ell } - \xi _ { i } \Theta _ { j \ell } \right) .
$$

Each term contains one coordinate from each of $x , z ,$ and $y ,$ so $F$ is trilinear. Define the reward by

$$
r \left( x , y \right) : = \frac { 1 } { m } \sum _ { \ell = 1 } ^ { m } \Xi _ { i _ { \ell } , j _ { \ell } , n + \ell } .
$$

It is linear and independent of $x ;$ its coeficient vector has Euclidean norm $\textstyle { \frac { 1 } { \sqrt { m } } } \leq 1$ , so $\begin{array} { r } { C _ { r } = \frac { 2 } { \sqrt { m } } \leq 2 } \end{array}$

Fix $x \in X$ and $z \in Z .$ . Since $z x _ { 0 } \neq 0$ and $y _ { 0 } = 1$ on $D$ , we can solve $F \left( x , z , y \right) = 0$ first for $\eta _ { ; }$ then for Θ, and then for Ξ. This gives

$$
\eta _ { i } = \frac { 1 } { 2 } q _ { i } , \quad \Theta _ { i j } = \frac { 1 } { 2 } q _ { i } q _ { j } , \quad \Xi _ { i j \ell } = \frac { 1 } { 2 } q _ { i } q _ { j } q _ { \ell } .
$$

Writing $\begin{array} { r } { t : = \| q \| _ { 2 } \leq \frac { 3 } { 4 } } \end{array}$ , the squared norm of $( \eta , \Theta , \Xi )$ is

$$
{ \frac { 1 } { 4 } } \left( t ^ { 2 } + t ^ { 4 } + t ^ { 6 } \right) \leq { \frac { 4 3 2 9 } { 1 6 3 8 4 } } < 1 .
$$

Thus this point lies in $D ,$ so $K _ { z } \left( x \right)$ is a singleton independent of z. The map $F _ { x , z } : Y \to W$ is also onto. For any right-hand side, we can set $y _ { 0 } = 0$ and solve successively for $\eta , \Theta$ , and Ξ.

Within the hyperplane $y _ { 0 } = 1$ , the equations have just one solution, and that solution lies in $D .$ So there are no solutions outside D to check in the transversality condition. We can also prove the distance inequality for points in the full kernel $L _ { z } \left( x \right)$ . Since $F _ { x , z }$ is onto and the dimension of $Y$ is one more than that of $W _ { i }$ , this kernel is the line spanned by $y \left( x \right) = \left( 1 , g \left( q \right) \right)$ , where $g \left( q \right) : = \left( \eta , \Theta , \Xi \right)$ is given by the equations above. If $p = \tau y \left( x \right) \in L _ { z } \left( x \right)$ lies outside D, then

$$
\operatorname { d i s t } \left( p , D \right) \geq \left| \tau - 1 \right| , \quad \operatorname { d i s t } \left( p , K _ { z } \left( x \right) \right) = \left| \tau - 1 \right| \sqrt { 1 + \| g \left( q \right) \| _ { 2 } ^ { 2 } } .
$$

Thus, for all these points, dist $( p , D )$ is at least dist $\left( p , K _ { z } \left( x \right) \right)$ times

$$
{ \frac { 1 } { \sqrt { 1 + { \frac { 4 3 2 9 } { 1 6 3 8 4 } } } } } = { \frac { 1 2 8 } { \sqrt { 2 0 7 1 3 } } } > 0 . 8 8 \geq S .
$$

Evaluating the reward at the unique point in $K _ { z } \left( x \right) \mathrm { g i v e }$ s

$$
v _ { z } \left( x \right) = \frac { 1 } { 2 m } p _ { G } \left( q \right) .
$$

Since q ranges over the radius- 3 ball, homogeneity and Lemma 33 imply, for every $z \in Z$ ，

$$
V _ { z } = \frac { 2 7 } { 1 2 8 m } \operatorname* { m a x } _ { \left\| h \right\| _ { 2 } = 1 } p _ { G } \left( h \right) = \frac { 2 7 } { 1 2 8 m } \sqrt { \frac { 2 } { 2 7 } } \sqrt { 1 - \frac { 1 } { \omega \left( G \right) } } .
$$

For $j = 2 , \dots , n$ , let

$$
U _ { j } : = \frac { 2 7 } { 1 2 8 m } \sqrt { \frac { 2 } { 2 7 } } \sqrt { 1 - \frac { 1 } { j } } .
$$

If $\omega \left( G \right) \geq k$ , then $V _ { z } \geq U _ { k } ; { \mathrm { i f } } \ \omega \left( G \right) \leq k - 1$ , then $V _ { z } \leq U _ { k - 1 }$ . The gap between these bounds satisfies

$$
U _ { k } - U _ { k - 1 } \geq \frac { 2 7 } { 1 2 8 m } \sqrt { \frac { 2 } { 2 7 } } \frac { 1 } { 2 k \left( k - 1 \right) } = \Omega \left( \frac { 1 } { m k ^ { 2 } } \right) .
$$

Since ${ \sqrt { \frac { 2 } { 2 7 } } } > { \frac { 1 } { 4 } }$ and $k \leq n$ , the gap $\Delta _ { k } : = U _ { k } - U _ { k - 1 }$ satisfies

$$
\Delta _ { k } > \frac { 1 } { 4 0 m k \left( k - 1 \right) } \geq \frac { 1 } { 4 0 m n ^ { 2 } } .
$$

Set $\begin{array} { r } { \varepsilon : = \frac { 1 } { 5 0 0 m n ^ { 2 } } } \end{array}$ . Using polynomially many bits, we can compute a rational $\tau _ { k }$ such that

$$
\left| \tau _ { k } - \frac { 1 } { 2 } \left( U _ { k } + U _ { k - 1 } \right) \right| \leq \varepsilon .
$$

An estimate of $V _ { z }$ with additive error at most $\varepsilon$ lies above $\tau _ { k }$ when $\omega \left( G \right) \geq k$ and below it when $\omega \left( G \right) \leq k - 1$ , because $\begin{array} { r } { 2 \varepsilon < \frac { \Delta _ { k } } { 2 } } \end{array}$ . It therefore decides whether G contains a k-clique. The constructed instance has dimension $O \left( d ^ { 3 } \right)$ , and all its defining coeficients are rational with polynomial encoding length. This proves that approximating $V _ { z }$ is NP-hard even when $z$ is known.

Given any arm x with rational coordinates, we can compute the unique point in $K _ { z } \left( x \right)$ exactly in polynomial time using the equations above; $\textstyle x _ { 0 } \geq { \frac { 2 } { 3 } }$ ensures that we can divide by x . We can therefore simulate the adversary by returning this point. Applying Lemma 25 with the chosen $\varepsilon$ gives, with probability at least ${ \frac { 2 } { 3 } } ,$ , an arm xˆ with $V _ { z } - v _ { z } \left( \hat { x } \right) \le \varepsilon$ . We compute $v _ { z } \left( \hat { x } \right)$ exactly from the unique point in $K _ { z } \left( \hat { x } \right)$ , which gives an estimate of $V _ { z }$ with additive error at most ε. Comparing it with $\tau _ { k }$ decides CLIQUE, so ${ \mathrm { N P } } \subseteq { \mathrm { B P P } }$ . For a deterministic learner, the estimate is deterministic and gives $\mathrm { P } = \mathrm { N P }$ □

## C.5 Eficient planning does not imply eficient learning

An eficient planner for every hypothesis does not always give an eficient learner, so the converse of Lemma 25 fails. We show this with instances outside the setting of Section 2.1. They have finite arm and hypothesis sets with short descriptions, and an outcome set that is a polytope. We also allow the reward to be linear in the arm and in the outcome separately, so it can contain products of arm and outcome coordinates.

Theorem 34. There is a family of instances with parameters $\phi = ( n , k )$ , for integers $2 \leq k \leq n$ given in unary, in which planning for a known hypothesis takes $O ( n ^ { 2 } )$ time, however, if for some fixed $\alpha > 0$ and polynomial P, one randomized polynomial-time learner guarantees $R _ { T } \leq P \left( | \phi | \right) T ^ { 1 - \alpha }$ for all parameters $\phi$ and every true hypothesis $h ^ { \star } \in H$ , then $\mathrm { N P = R P }$ . If a deterministic learner achieves the same regret guarantee, then $\mathrm { P } = \mathrm { N P }$

Proof. We reduce from CLIQUE. Both the arms and the hypotheses will correspond to the $k -$ element subsets of $[ n ]$ , and the optimal arm for the hypothesis corresponding to a set U is the arm corresponding to $U$ . So planning is easy once the hypothesis is known. But an adversary can make every observation consistent with every k-clique of a graph $G ,$ so a learner with low regret must find a k-clique of G on its own.

Fix $2 \leq k \leq n$ , and let

$$
E _ { n } : = \{ \{ i , j \} : 1 \leq i < j \leq n \} , \quad m : = { \binom { n } { 2 } } , \quad q : = { \binom { k } { 2 } } .
$$

For every k-element set $A \subseteq [ n ]$ , define $\boldsymbol { x } ^ { A } \in \mathbb { R } ^ { m }$ by

$$
x _ { e } ^ { A } : = { \left\{ \begin{array} { l l } { { \frac { 1 } { q } } } & { { \mathrm { i f } } \ e \subseteq A } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. } , \quad e \in E _ { n } ,
$$

and take

$$
X : = \left\{ x ^ { A } : A \subseteq \left[ n \right] , \left| A \right| = k \right\} .
$$

An arm is represented succinctly by the vertex set A. For every k-element set $U \subseteq [ n ]$ , define $z ^ { U } \in \mathbb { R } ^ { 1 + m }$ by

$$
z _ { 0 } ^ { U } : = 1 , \quad z _ { e } ^ { U } : = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } e \subseteq U } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} , \right.
$$

and let $Z : = \left\{ z ^ { U } : U \subseteq \left[ n \right] , \left| U \right| = k \right\}$

Write an outcome as $y = ( y _ { 0 } , g , s ) \\\bar { \in } \mathbb { R } ^ { 1 + 2 m }$ and set

$$
D : = \{ ( 1 , g , s ) : g \in [ 0 , 1 ] ^ { m } , s \in [ - 1 , 1 ] ^ { m } \} .
$$

With constraint space $W = \mathbb { R } ^ { m }$ , define the linear map $C _ { z } : \mathbb { R } ^ { 1 + 2 m }  W$ by

$$
( C _ { z } y ) _ { e } : = z _ { e } ( g _ { e } - y _ { 0 } - s _ { e } ) + z _ { 0 } s _ { e } , \quad e \in E _ { n } .
$$

For $z \in Z$ , set

$$
K _ { z } \left( x \right) : = \left\{ y \in D : C _ { z } y = 0 \right\} , \quad h _ { z } \left( x \right) : = \left\{ Q \in \Delta D : \mathbb { E } _ { y \sim Q } \left[ y \right] \in K _ { z } \left( x \right) \right\} ,
$$

and let $H : = \{ h _ { z } : z \in Z \}$

The map $( y , z ) \mapsto C _ { z } y$ is bilinear. Since $y _ { 0 }$ is an outcome coordinate, the constraint has the form of Equation 1 with $B _ { z } = 0 , d _ { z } = 0$ , and $z \mapsto C _ { z }$ linear. For $z = z ^ { U }$

$$
( C _ { z } v y ) _ { e } = { \left\{ \begin{array} { l l } { g _ { e } - y _ { 0 } } & { { \mathrm { i f ~ } } e \subseteq U } \\ { s _ { e } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

Thus, inside $D ,$ the constraints force $g _ { e } = 1$ for every $e \subseteq U$ and $s _ { e } = 0$ for every e $\nsubseteq U$ . The remaining $g _ { e }$ coordinates are free to vary in $[ 0 , 1 ]$ . The feasible set is nonempty. It contains the outcome with $g _ { e } = 1$ on the edges of $U , g _ { e } = 0$ elsewhere, and $s = 0$ . Moreover, $C _ { z } { \upsilon }$ is onto $W .$ Given $w \in W$ , we can set $y _ { 0 } = 0$ , use $g _ { e } = w _ { e }$ and $s _ { e } = 0$ on the edges of $U _ { ; }$ , and use $g _ { e } = 0$ and $s _ { e } = w _ { e }$ elsewhere.

Define the reward

$$
r \left( x , ( y _ { 0 } , g , s ) \right) : = \sum _ { e \in E _ { n } } x _ { e } g _ { e } .
$$

The parameters $\phi = ( n , k )$ determine $X , D , r ,$ , and $H _ { ; }$ , with the maps $C _ { z }$ built into the learner. Only the true $U$ is hidden. The reward lies in $[ 0 , 1 ]$ on $X \times D$ . For an arm $x ^ { A }$ , the adversary minimizes it by setting every coordinate $g _ { e }$ not forced by U to zero, so

$$
v _ { z ^ { U } } \left( x ^ { A } \right) = \frac { \binom { | U \cap A | } { 2 } } { q } .
$$

Thus $V _ { z } \sigma = 1$ , uniquely attained by $x ^ { U }$ . Given $z ^ { U }$ , an exact planner returns $x ^ { U }$ by setting $\begin{array} { r } { x _ { e } ^ { U } = \frac { z _ { e } ^ { U } } { q } } \end{array}$ for every $e \in E _ { n }$ , which takes $O \left( m \right) = O \left( n ^ { 2 } \right)$ time.

We now show that a learner with the stated regret bound would solve CLIQUE. Given a graph $G = ( [ n ] , E \left( G \right) )$ and target clique size k [31], let

$$
g _ { e } ^ { G } : = \left\{ { 1 \atop 0 } \right. \mathrm { i f } e \in E \left( G \right)  , \quad y ^ { G } : = \left( 1 , g ^ { G } , 0 \right) \in D .
$$

If U is a k-clique of $G ,$ the adversary can return $y ^ { G }$ on every round under hypothesis $z ^ { U }$ . Indeed, every edge inside $U$ has $g _ { e } ^ { G } = 1$ , and all the $s _ { e }$ coordinates are zero. For every candidate A,

$$
r \left( x ^ { A } , y ^ { G } \right) = { \frac { | E \left( G \left[ A \right] \right) | } { q } } .
$$

This reward is 1 exactly when A is a k-clique, and is at most $\textstyle 1 - { \frac { 1 } { q } }$ otherwise.

Assume the learner in the theorem exists, and give it the outcome $y ^ { G }$ on every round. If G contains a k-clique $U$ , these outcomes are compatible with $z ^ { U }$ , whose optimal value is 1. If the learner plays $x ^ { A _ { 1 } } , \ldots , x ^ { A _ { T } }$ , the total reward it loses relative to this value is

$$
\mathcal { L } _ { T } : = T - \sum _ { t = 1 } ^ { T } r \left( x ^ { A _ { t } } , y ^ { G } \right) .
$$

This quantity is nonnegative. Let $\mathcal { E }$ be the event that none of $A _ { 1 } , \ldots , A _ { T }$ is a clique. On this event, the learner loses at least $\textstyle { \frac { 1 } { q } }$ on every round, so $\begin{array} { r } { \mathcal { L } _ { T } \geq \frac { T } { q } } \end{array}$ . Markov’s inequality and the assumed regret bound give

$$
\operatorname* { P r } \left( { \mathcal { E } } \right) \leq q { \frac { E \left[ { \mathcal { L } } _ { T } \right] } { T } } \leq q P \left( \left| \phi \right| \right) T ^ { - \alpha } .
$$

Choose

$$
T : = \left\lceil ( 4 q P \left( \left| \phi \right| \right) ) ^ { \frac { 1 } { \alpha } } \right\rceil .
$$

Since $| \phi | = { \cal { O } } ( n )$ , both T and the time needed to simulate the learner are polynomial in the CLIQUE input length. Check each k-set proposed by the learner and accept as soon as one is a clique in G. If G has a k-clique, the preceding bound gives acceptance probability at least $\frac 3 4$ . If G has none, none of the proposed sets can pass this check. In that case, the outcomes may be incompatible with every hypothesis, but the learner still runs in polynomial time on every history of arms in X and outcomes in D. Thus the simulation also takes polynomial time when G has no k-clique.

This is an RP algorithm for CLIQUE. Since CLIQUE is NP-complete, it implies NP = RP. If the learner is deterministic, the same verified search gives $\mathrm { P } = \mathrm { N P }$ . The adversary in this argument returns the same outcome $y ^ { G }$ on every round, so the conclusion holds even if the regret guarantee is required only against such adversaries. □

This example shows that eficient planning alone does not sufice for eficient learning. It uses exponentially large finite sets X and Z represented by k-subsets, a polytope $D ,$ and the bilinear reward $\begin{array} { r } { r \left( x , y \right) \ = \ \sum _ { e } x _ { e } g _ { e } } \end{array}$ . It therefore falls outside our setting of Euclidean balls and linear rewards.