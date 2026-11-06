# Online Learning with Ordered Experts (via two-point feedback)

Shiliang Zuo<sup>1</sup>

<sup>1</sup>Meta, szuo.rs@gmail.com

## Abstract

We study adversarial online learning with expert recommendations in a totally ordered action space, where the experts’ recommendations may change and be reordered arbitrarily. Our focus is two-point feedback: on each round the learner plays two current recommendations, observes both, and pays their average. We study two problems. In the separated monotone problem, the loss is the sum of a hidden nondecreasing component and a hidden nonincreasing component, and a query reveals the two components separately. In the BV problem, the losses are bounded and have bounded variation (BV) along the order, and a query reveals only the loss.

In the separated monotone problem, two paid observations achieve

$$
O \big ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } \big )
$$

regret against the best of |F| experts, within a logarithmic factor of the classical fullinformation bound. For the BV problem, we obtain $\widetilde { O } ( \sqrt { T } + | \mathcal { F } | )$ regret. We give an application in dynamic pricing with expert advice, showing that two-option menus matches the full information rate up to log factors.

The main technical ingredient is a marginal-preserving endpoint flow estimator. Two observations reveal a loss increment along the order; random paths supply telescoping identities that turn this local information into unbiased loss estimates for all current recommendations. A reversible pair sampler preserves the learner’s intended marginal at both played actions. The estimator’s second-moment and stability bounds support online updates for both feedback models.

## 1 Introduction

In online learning, a learner makes a decision on every round, an adversary chooses the loss of every possible decision, and the learner aims to keep its cumulative loss close to that of the best strategy in a comparison class. We study the expert-advice version of this problem. On every round, each expert in a finite class $\mathcal { F }$ recommends an action, the learner chooses an action, and its regret is measured against the best expert in hindsight. In the classical experts and bandit problems, the actions form an unstructured set, and nothing relates the losses of diferent actions. In many applications, however, the actions carry a natural total order.

For example, in pricing, a seller chooses a price. In inventory management, a retailer chooses a stock level. In classification, a decision maker chooses a threshold and flags every score above it. In each case the experts’ recommendations are points on a line, and they may change and be reordered arbitrarily across rounds.

The order brings additional structure that a learner may exploit. Many problems involve a natural tradeof between two monotone losses: moving the action up increases one cost and decreases the other. The loss then decomposes as

$$
\ell _ { t } = \ell _ { t } ^ { \uparrow } + \ell _ { t } ^ { \downarrow } ,
$$

with a nondecreasing component $\ell _ { t } ^ { \uparrow }$ and a nonincreasing component $\ell _ { t } ^ { \downarrow }$

We formulate the problem of online learning with ordered experts and focus on two-point feedback. The learner simultaneously plays two current recommendations, observes the feedback at both, and pays the average of their losses; there are no free probes. We study two problems. In the separated monotone problem, each loss is a monotone sum as above, and a query at an action reveals $\ell _ { t } ^ { \uparrow }$ and $\ell _ { t } ^ { \downarrow }$ there separately. In the BV problem, each loss has total variation at most one along the order, and a query reveals only the loss. On a finite chain, the Jordan decomposition expresses every loss of bounded variation (BV) as such a monotone sum, up to an action-independent ofset and a constant rescaling, so the two problems involve essentially the same losses and difer only in what a query reveals. To calibrate the results, we compare with one-point feedback, where the learner plays and observes a single recommendation, and with full information.

At a high level, two observations on the same round reveal an increment of the loss along the order, which no single observation can, and this is what separates two-point from one-point feedback. In the separated monotone problem, two observations match full information up to a logarithmic factor, whereas one observation costs a polynomial factor. In the BV problem, two observations replace the multiplicative $\sqrt { | \mathcal { F } | }$ cost of bandit feedback by an additive |F|.

An important application is dynamic pricing, with or without expert advice. Menus with two purchase options and an outside option for one risk-neutral buyer yield revenue-regret guarantees against the best fixed posted price and against the best price-recommending expert.

## 1.1 Results

The separated monotone problem. Against $| \mathcal F |$ experts with arbitrary current recommendations, two paid observations achieve

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) = O \big ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } \big ) .
$$

The key ingredient is the endpoint flow estimator described below: from the two observed values it builds unbiased and nonnegative estimates of both monotone parts at every active recommendation, so exponential weights can be run on them. Conversely, no two-point learner can beat $\Omega ( { \sqrt { T } }$ log min $\{ | { \mathcal { F } } | , { \sqrt { T } } \} )$ ), even when the experts never move; this lower bound holds in the BV problem as well.

<table><tr><td>Problem</td><td>two paid observations</td><td>one observation</td><td>full information</td></tr><tr><td>Separated</td><td> $O ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } )$   $\scriptstyle ( { \mathrm { T h e o r e m ~ 2 } } ) ;$ </td><td> $\Omega ( \operatorname* { m i n } \{ \sqrt { | \mathcal { F } | T } , T ^ { 2 / 3 } \} )$  (Theorem 7)</td><td> $\Theta ( \sqrt { T \log | \mathcal { F } | } )$  (Theorem 9)</td></tr><tr><td>monotone problem</td><td> $\overset { \cdot } { \Omega } ( \sqrt { T } \log \operatorname* { m i n } \{ | \mathcal { F } | , \sqrt { T } \} )$ </td><td></td><td></td></tr><tr><td>BV problem</td><td>(Theorem 3)  $\widetilde { \Theta } ( \sqrt { T } + | \mathcal { F } | )$   $\left( \mathrm { C o r o l l a r y \ 1 } \right)$ </td><td> $\Theta ( \sqrt { | \mathcal { F } | T } )$  (Theorem 8)</td><td> $\Theta ( \sqrt { T \log | \mathcal { F } | } )$  (Theorem 9)</td></tr></table>

Table 1: Regret against the best of $| \mathcal F |$ experts. The BV two-point and one-point Θ rates are stated for $| { \mathcal { F } } | \leq T ;$ the full-information Θ rate is stated for $\log _ { 2 } | \mathcal { F } | \leq T$ . All lower bounds hold even when every expert recommends one fixed action on every round.

The BV problem. When a query reveals only the total loss and its variation is at most one, two paid observations achieve $\widetilde { O } ( \sqrt { T } + | \mathcal { F } | )$ regret. Signed endpoint-flow estimates are controlled by an entropy–log-barrier update using their coordinatewise bound. Matching lower bounds up to logarithmic factors show that the two-point rate is $\widetilde { \Theta } ( \sqrt { T } + | \mathcal { F } | )$ when $| \mathcal { F } | \le T$

Dynamic pricing with expert advice. We apply our framework to adversarial dynamic pricing through an incentive-compatible menu mechanism. On each round, the two prices sampled by our algorithm are converted into a menu with two purchase options and an outside option. The buyer’s choice reveals the acceptance decisions at both prices, while the seller’s expected revenue equals the average revenue of the two posted prices. This implementation yields $\mathsf { \bar { O } } ( \sqrt { T } \log ^ { 3 / 2 } T )$ expected revenue regret against the best fixed posted price. More generally, given a finite class $\mathcal { F }$ of price-recommending experts, we achieve $O ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } )$ regret against the best expert, even when expert recommendations change and reorder arbitrarily across rounds. This matches the full information up to log factors. For comparison, the one-point bandit (posted-price auction) feedback gives regret $\bar { \Theta } ( T ^ { 2 / 3 } )$ against the best fixed price setting.

Comparison with one point and full information. As summarized in Table 1, one observation gives regret $\Omega ( \operatorname* { m i n } \{ \sqrt { | \mathcal { F } | T } , T ^ { 2 / 3 } \} )$ in the separated monotone problem and $\Theta ( { \sqrt { | { \mathcal { F } } | T } } )$ in the BV problem for $| { \mathcal { F } } | \leq T$ . Full information gives $\Theta ( \sqrt { T \log | \mathcal { F } | } )$ in both problems for $\log _ { 2 } | \mathcal { F } | \leq T$ even with fixed recommendations. For $| \mathcal { F } | \le T ^ { 1 / 3 }$ , two-point feedback removes the one-point rate’s multiplicative $\sqrt { | \mathcal { F } | }$ dependence up to logarithms; full information also removes the additive |F| term in our BV upper bound.

## 1.2 The endpoint flow estimator

All of our two-point upper bounds rest on a single estimator. Standard one-point importance weighting preserves unbiasedness by dividing the observed loss by its sampling probability, but its distribution-weighted second moment can grow linearly with the number of distinct recommendations.

Under two-point feedback, our key technical tool is the endpoint flow estimator (Theorem 1). For a target vector g with values in [0, 1] and total variation at most one, let $\pi$ be the distribution over distinct current recommendations after the $1 / T$ smoothing used by our algorithms. Three properties are central:

1. Both played actions have marginal π.

2. It is unbiased at every coordinate: $\mathbb { E } \widehat { g } ( j ) = g ( j )$

3. Its distribution-weighted second moment satisfies $\begin{array} { r } { \mathbb { E } \sum _ { j } { \pi ( j ) } \widehat { g } ( j ) ^ { 2 } = O ( \log ^ { 2 } ( | \mathcal { F } | T ) ) } \end{array}$ Marginal preservation aligns the expected paid loss with the intended mixture, while unbiasedness and the second-moment bound control estimation error. Section 3 gives the construction and full analysis.

## 1.3 Related work

Expert advice and ordered actions. Exponential weights gives the full-information benchmark [Littlestone and Warmuth, 1994, Freund and Schapire, 1997]; see Cesa-Bianchi and Lugosi [2006]. One-point expert advice uses the bandit framework [Auer, Cesa-Bianchi, Freund, and Schapire, 2002, McMahan and Streeter, 2009], with ordinary bandit rates providing the comparison for our BV result [Audibert and Bubeck, 2009, Bubeck and Cesa-Bianchi, 2012, Lattimore and Szepesv´ari, 2020]. Metric bandits [Kleinberg, Slivkins, and Upfal, 2008], unimodal bandits [Yu and Mannor, 2011, Combes and Prouti\`ere, 2014], posted pricing [Kleinberg and Leighton, 2003], and censored-demand inventory control [Huh and Rusmevichientong, 2009] exploit structure over a fixed action set. Here experts may change and reorder their recommendations arbitrarily.

Multi-point feedback. Single-point convex-bandit estimates [Flaxman, Kalai, and McMahan, 2005] can be improved using two-point queries [Agarwal, Dekel, and Xiao, 2010, Duchi, Jordan, Wainwright, and Wibisono, 2015, Shamir, 2017]. Feedback graphs [Mannor and Shamir, 2011, Alon, Cesa-Bianchi, Dekel, and Koren, 2015] and paid side observations [Seldin, Bartlett, Crammer, and Abbasi-Yadkori, 2014] provide information beyond the played action; dueling bandits observe comparisons [Yue, Broder, Kleinberg, and Joachims, 2012]. Our two played actions supply numerical interval increments for reconstructing the ordered loss vector, with their intended marginals preserved.

Regularization and reversible sampling. Log-barrier and hybrid entropy–log-barrier updates stabilize importance-weighted estimates [Foster, Li, Lykouris, Sridharan, and Tardos, 2016, Agarwal, Luo, Neyshabur, and Schapire, 2017, Wei and Luo, 2018, Bubeck, Cohen, and Li, 2018]; endpoint flow supplies the required coordinatewise control. The pair sampler uses a reversible kernel with prescribed stationary distribution [Hastings, 1970], with conductances chosen to control reconstruction variance.

Organization. Section 2 defines the ordered-expert model, the two-point protocol, and the two problems. Section 3 develops the marginal-preserving endpoint flow estimator. Section 4 presents and analyzes a two-point algorithm for the separated monotone problem and proves a lower bound. Section 5 presents and analyzes a two-point algorithm for the BV problem and proves lower bounds that match it up to logarithmic factors. Section 6 contains a mechanism-design native application in dynamic pricing. Section 7 contains the one-point and full-information comparisons. Proofs of the general results are deferred to the appendices.

## 2 Model and preliminaries

## 2.1 Ordered experts

Fix a horizon $T \geq 2$ . Let $( A , \preceq )$ be a totally ordered action space with a least element $\perp$ and a greatest element ⊤. For example, $\mathcal { A } = [ 0 , 1 ]$ with the usual order, so that $\perp = 0$ and ${ \top = 1 }$ where an action may be a normalized price, a threshold, or a stock level. Let $\mathcal { F }$ be a finite class of $| \mathcal { F } | \ge 2$ experts. On round $t ,$ before the learner acts, every expert $f \in { \mathcal { F } }$ recommends an action $a _ { t , f } \in \mathcal { A }$ . The recommendations may change arbitrarily across rounds. The oblivious adversary fixes all recommendations and all loss functions $\ell _ { t } : \mathcal { A }  [ 0 , 1 ]$ before the interaction begins. Let

$$
X _ { t } = \{ a _ { t , f } : f \in \mathcal { F } \} = \{ x _ { t , 1 } \prec \cdots \prec x _ { t , m _ { t } } \} , \qquad m _ { t } \leq | \mathcal { F } | ,
$$

be the distinct active actions at round t. We call $j$ the rank of $x _ { t , j }$ , write $j _ { t } ( f )$ for the rank of the recommendation $a _ { t , f } .$ and identify each active action with its rank when convenient; in particular, a distribution over $X _ { t }$ is a distribution over the ranks $\{ 1 , \ldots , m _ { t } \}$

We study two problems, which difer in what a query reveals. Before that, we define the notion of total variation. For a function g on a finite chain $x _ { 1 } \prec \dots \prec x _ { m } .$

$$
\mathrm { T V } ( g ) = \sum _ { j = 2 } ^ { m } | g ( x _ { j } ) - g ( x _ { j - 1 } ) |
$$

is its total variation. Unless stated otherwise, $\mathrm { T V } ( \ell _ { t } )$ denotes the total variation of $\ell _ { t }$ restricted to the active chain $X _ { t }$

The separated monotone problem: monotone-sum losses, separated observations. On each round the adversary chooses a nondecreasing component $\ell _ { t } ^ { \uparrow } : \mathcal { A } \to [ 0 , 1 / 2 ]$ with $\ell _ { t } ^ { \uparrow } ( \bot ) = 0$ and a nonincreasing component $\ell _ { t } ^ { \downarrow } : \mathcal { A }  [ 0 , 1 / 2 ]$ with $\ell _ { t } ^ { \downarrow } ( \top ) = 0$ , and the loss is their sum,

$$
\ell _ { t } ( a ) = \ell _ { t } ^ { \uparrow } ( a ) + \ell _ { t } ^ { \downarrow } ( a ) .
$$

Querying an action a reveals $\ell _ { t } ^ { \uparrow } ( a )$ and $\ell _ { t } ^ { \downarrow } ( a )$ separately.

The BV problem: BV losses, aggregate observations. On every round the loss $\ell _ { t } : \mathcal { A } \to [ 0 , 1 ]$ has bounded variation, $\mathrm { T V } ( \ell _ { t } ) \leq 1$ , and querying an action a reveals only $\ell _ { t } ( a )$

Relation between the two problems. The two problems involve essentially the same losses and difer in what a query reveals. Every loss of the separated monotone problem is a BV loss, since $\mathrm { T V } ( \ell _ { t } ) \leq \mathrm { T V } ( \ell _ { t } ^ { \uparrow } ) + \mathrm { T V } ( \ell _ { t } ^ { \downarrow } ) \leq 1 / 2 + 1 / 2$ . Conversely, by the Jordan decomposition, every loss with $\mathrm { T V } ( \ell _ { t } ) \leq 1$ can be written as

$$
\ell _ { t } ( x ) = c _ { t } + \ell _ { t } ^ { \uparrow } ( x ) + \ell _ { t } ^ { \downarrow } ( x ) , \qquad x \in X _ { t } ,
$$

where the ofset $c _ { t }$ is a constant that does not depend on x. On the active chain, take

$$
\ell _ { t } ^ { \uparrow } ( x _ { t , j } ) = \sum _ { r = 2 } ^ { j } \bigl ( \ell _ { t } ( x _ { t , r } ) - \ell _ { t } ( x _ { t , r - 1 } ) \bigr ) ^ { + } , \qquad \ell _ { t } ^ { \downarrow } ( x _ { t , j } ) = \sum _ { r = j + 1 } ^ { m _ { t } } \bigl ( \ell _ { t } ( x _ { t , r - 1 } ) - \ell _ { t } ( x _ { t , r } ) \bigr ) ^ { + } ,
$$

where $z ^ { + } = \operatorname* { m a x } \{ z , 0 \}$ , and set $c _ { t } = \ell _ { t } ( x _ { t , 1 } ) - \ell _ { t } ^ { \downarrow } ( x _ { t , 1 } )$ . Extend the components to $\mathcal { A }$ as step functions, keeping the upward component zero below the first active action and the downward component zero above the last. Thus $\ell _ { t } ^ { \uparrow }$ is nondecreasing with $\ell _ { t } ^ { \uparrow } ( \bot ) = 0 , \ell _ { t } ^ { \downarrow }$ is nonincreasing with $\ell _ { t } ^ { \downarrow } ( \top ) = 0$ and $\ell _ { t } ^ { \uparrow } ( \top ) + \ell _ { t } ^ { \downarrow } ( \bot ) = \mathrm { T V } ( \ell _ { t } ) \leq 1$ . The ofset $c _ { t }$ shifts every expert’s loss equally and does not afect regret. Hence, after removing the ofset and dividing both components by two, the active loss vector belongs to the separated monotone problem. This is a relation between loss classes; aggregate feedback does not reveal the decomposition or its ofset.

## 2.2 Two-point feedback and regret

Our main protocol gives the learner two paid observations per round. The learner simultaneously chooses $A _ { t } , B _ { t } \in X _ { t }$ and incurs

$$
\frac { 1 } { 2 } \big ( \ell _ { t } ( A _ { t } ) + \ell _ { t } ( B _ { t } ) \big ) .
$$

For the separated monotone problem, the learner then observes the four values $\ell _ { t } ^ { \uparrow } ( A _ { t } ) , \ell _ { t } ^ { \downarrow } ( A _ { t } ) , \ell _ { t } ^ { \uparrow } ( B _ { t } )$ , and $\ell _ { t } ^ { \downarrow } ( B _ { t } )$ . For the BV problem, it observes only the two values $\ell _ { t } ( A _ { t } )$ and $\ell _ { t } ( B _ { t } )$ . The regret to the best expert is

$$
\mathrm { R e g } _ { T } ( \mathcal { F } ) = \sum _ { t = 1 } ^ { T } \frac { \ell _ { t } ( A _ { t } ) + \ell _ { t } ( B _ { t } ) } { 2 } - \operatorname* { m i n } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } \ell _ { t } ( a _ { t , f } ) .
$$

We bound the expected regret $\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } )$ , where the expectation is over learner randomness. We write $\mathbb { E } _ { t }$ for expectation conditional on the history before the learner draws its pair on round t. Section $7$ compares two-point feedback with one-point feedback, where the learner plays and observes a single action, and with full information.

## 3 The endpoint flow estimator

This section develops the estimator behind our main results. Throughout this section, we fix one round and suppress the time index. The active chain is $x _ { 1 } \prec \cdots \prec x _ { m }$ , and $g$ is a hidden vector on it with entries $g ( x _ { 1 } ) , \dots , g ( x _ { m } )$ . For the separated monotone problem, $g$ is one of the two monotone components, $\ell ^ { \uparrow }$ or $\ell ^ { \downarrow }$ . For the BV problem, g is the loss vector ℓ. Finally, π is the distribution over ranks $\{ 1 , \ldots , m \}$ with which the learner intends to play.

## 3.1 Main result and overview

Main result. We assume a floor on the intended distribution,

$$
\pi ( j ) \geq \frac { \xi } { m } , \qquad j = 1 , \ldots , m ,
$$

where $0 < \xi \le 1$ , and write

$$
\Lambda : = 1 + \log \frac { m } { \xi } .
$$

Any distribution $\pi ^ { \circ }$ on the active ranks can be brought into this form by mixing it with $\xi$ times the uniform distribution on active actions, at a cost of at most ξ per round in expected loss. Our algorithms do this, and the final choice will be $\xi = 1 / T$ , so that $\Lambda \leq 1 + \log ( | \mathcal { F } | T )$ and the mixing costs at most one in total loss over $T$ rounds. The anchor $x _ { 0 }$ is an auxiliary node below the active chain with assigned value $g ( x _ { 0 } ) = 0$ , even if $x _ { 1 } = \bot ;$ it is never played and serves only as the starting point of the reconstruction.

Theorem 1 (Endpoint flow estimator). Let π satisfy the floor $\pi ( j ) \geq \xi / m$ . There exist a random pair of ranks $( A , B )$ , whose law depends only on π, and an estimator ${ \widehat { g } } ,$ , which depends on g only through the observed values $g ( x _ { A } )$ and $g ( x _ { B } )$ , such that for every hidden vector g with values in [0, 1] and $\mathrm { T V } ( g ) \leq 1$

1. marginals: $A \sim \pi$ and $B \sim \pi$

2. unbiasedness: $\mathbb { E } [ \widehat { g } ( j ) ] = g ( x _ { j } )$ for every target $j ;$

3. second moment:

$$
\mathbb { E } \langle \pi , { \widehat { g } } ^ { 2 } \rangle = O ( \Lambda ^ { 2 } ) ;
$$

4. self-bound: for every realization,

$$
\operatorname* { m a x } _ { j } \pi ( j ) | \widehat { g } ( j ) | \leq 3 \Lambda ;
$$

5. monotone sign: if g is nondecreasing, then ${ \widehat { g } } ( j ) \geq 0$ for every j and every realization.

High-level overview. Fix a target rank $j .$ . Imagine a particle that starts at the anchor $x _ { 0 }$ and moves up the chain one jump at a time until it exits at $\boldsymbol { x } _ { j } ;$ from $x _ { a }$ it jumps to a rank $b \in \{ a + 1 , \ldots , j \}$ with a probability, specified later, that depends only on π. Let $\phi _ { j } ( a , b )$ be the probability that the particle jumps directly from $x _ { a }$ to $x _ { b }$ . Along every trajectory the increments $g ( x _ { b } ) - g ( x _ { a } )$ of the jumps telescope to $g ( x _ { j } ) - g ( x _ { 0 } ) = g ( x _ { j } )$ , so averaging over trajectories gives the decomposition

$$
g ( x _ { j } ) = \sum _ { 0 \leq a < b \leq j } \phi _ { j } ( a , b ) { \big ( } g ( x _ { b } ) - g ( x _ { a } ) { \big ) } .
$$

Thus $g ( x _ { j } )$ is a fixed linear combination of increments, with coeficients determined by $\pi$ alone.

Now suppose the queried pair is sampled from some probability law. The value $g ( x _ { A } )$ alone reveals the anchor increment $g ( x _ { A } ) - g ( x _ { 0 } )$ on the edge $( 0 , A )$ , and the two values together reveal the increment on the edge between A and B. Write $q ( a , b )$ for the probability that the increment on the edge $( a , b )$ is revealed. If $q ( a , b ) > 0$ whenever $\phi _ { j } ( a , b ) > 0$ , then weighting each revealed increment by $\phi _ { j } ( a , b ) / q ( a , b )$ and summing gives an unbiased estimate of $g ( x _ { j } )$ , simultaneously for every target $j$ , from only two observations.

Two design questions remain. First, the pair cannot follow an arbitrary law: both queried actions are played, so both must have marginal $\pi$ . We draw $A \sim \pi$ and then B by one step of a reversible Markov kernel with stationary distribution $\pi ,$ so that detailed balance gives $B \sim \pi$ as well. Second, the importance weights $\phi _ { j } ( a , b ) / q ( a , b )$ must not inflate the second moment. We therefore match the kernel to the particle: pairs of likely actions separated by little π-mass, which carry most of the flow, are sampled most often. The next subsection states this importance-weighting step in general, and the one after it gives the specific paths and kernel.

## 3.2 Telescoping flows and importance weighting

This subsection builds an unbiased estimate of every value $g ( x _ { j } )$ from the two observed values $g ( x _ { A } )$ and $g ( x _ { B } )$ . It works for any law of the queried pair; the specific law is chosen in the next subsection.

Edges. Index the anchor by rank 0, so that $g ( x _ { 0 } ) = 0$ . An edge is a pair of ranks $e = ( a , b )$ with $0 \leq a < b \leq m$ . It is an anchor edge if $a = 0$ and an interval edge if $a \geq 1$ . Its increment is

$$
H ( e ) = g ( x _ { b } ) - g ( x _ { a } ) .
$$

Flows. For a target rank $j ,$ let a random increasing path run from the anchor to $j ,$ jumping from each rank $a < j$ to some rank $b \in \{ a + 1 , \ldots , j \}$ , and let $\phi _ { j } ( e )$ be the probability that it traverses the edge e. The law of the path depends only on π and is specified in the next subsection. Along every realization the increments telescope to $g ( x _ { j } ) - g ( x _ { 0 } ) = g ( x _ { j } )$ , so taking expectations gives

$$
g ( x _ { j } ) = \sum _ { e } \phi _ { j } ( e ) H ( e ) .
$$

Revealed edges. Write $A \land B = \operatorname* { m i n } \{ A , B \}$ and $A \lor B = \operatorname* { m a x } \{ A , B \}$ . Since $g ( x _ { 0 } ) = 0$ , the value $g ( x _ { A } )$ alone reveals the increment of the anchor edge $E _ { 1 } = ( 0 , A ) . { \mathrm { ~ I f ~ } } A \neq B$ , the two values together also reveal the increment of the interval edge $E _ { 2 } = ( A \land B , A \lor B ) ; { \mathrm { i f ~ } } A = B$ , no interval edge is revealed. The revelation probability $q ( e )$ of an edge is the probability that its increment is revealed:

$$
q ( 0 , a ) = \mathbb { P } ( E _ { 1 } = ( 0 , a ) ) , \qquad q ( a , b ) = \mathbb { P } ( E _ { 2 } = ( a , b ) ) \quad ( 1 \leq a < b ) .
$$

Lemma 1 (Unbiased estimator). Suppose that $q ( e ) > 0$ whenever $\phi _ { j } ( e ) > 0$ , and define

$$
\widehat { g } ( j ) = \sum _ { a } \mathbf { 1 } \{ E _ { 1 } = ( 0 , a ) \} \frac { \phi _ { j } ( 0 , a ) } { q ( 0 , a ) } H ( 0 , a ) + \sum _ { 1 \leq a < b } \mathbf { 1 } \{ E _ { 2 } = ( a , b ) \} \frac { \phi _ { j } ( a , b ) } { q ( a , b ) } H ( a , b ) ,
$$

where the sums run over edges with $q > 0$ . Then $\mathbb { E } \widehat { g } ( j ) = g ( x _ { j } )$ for every j. If g is nonnegative and nondecreasing, then ${ \widehat { g } } ( j ) \geq 0$ for every j and every realization.

Each revealed increment is weighted by its flow divided by its revelation probability, so in expectation every edge contributes $\phi _ { j } ( e ) H ( e )$ and the identity above is recovered; the proof is in Section A. The first sum handles the anchor edges and the second the interval edges, so $E _ { 1 }$ and $E _ { 2 }$ need not lie on a common path. Two choices remain: the jump probabilities of the path, which determine the flows $\phi _ { j }$ , and the law of the queried pair, which determines the revelation probabilities $q .$ Both queried actions are played, so both must have marginal $\pi ,$ and the weights $\phi _ { j } / q$ must keep the second moment of $\widehat g$ small.

## 3.3 The construction

This subsection makes these two choices and bounds the second moment of the resulting estimator. Throughout, write $\begin{array} { r } { \Pi [ a , b ] = \sum _ { r = a } ^ { b } \pi ( r ) } \end{array}$ and $\begin{array} { r } { \Pi ( a , b ] = \sum _ { r = a + 1 } ^ { b } \pi ( r ) } \end{array}$ for the π-mass of an interval of ranks.

Random paths. For a target rank $j ,$ the particle of the overview starts at the anchor 0, and from a node $a < j$ it jumps to $b \in \{ a + 1 , \ldots , j \}$ with probability proportional to $\pi ( b )$

$$
\mathbb { P } ( b \mid a , j ) = \frac { \pi ( b ) } { \Pi ( a , j ] } .
$$

The particle thus steps preferentially onto actions that the learner is likely to play. Its edge-use probabilities, computed in Section $\mathrm { A }$ , are

$$
\phi _ { j } ( 0 , a ) = \frac { \pi ( a ) } { \Pi [ 1 , j ] } \quad ( a \leq j ) , \qquad \phi _ { j } ( a , b ) = \frac { \pi ( a ) \pi ( b ) } { \Pi [ a , j ] \Pi ( a , j ] } \quad ( 1 \leq a < b \leq j ) .
$$

Set $\phi _ { j } ( a , b ) = 0$ for all edges outside $0 \leq a < b \leq j$

Reversible pair sampler. Now we define how the pair of points is actually sampled. For $1 \leq a < b \leq m$ , define the symmetric conductance

$$
\Gamma ( a , b ) = \frac { 1 } { 4 \Lambda } \frac { \pi ( a ) \pi ( b ) } { \Pi [ a , b ] } ,
$$

set $\Gamma ( b , a ) = \Gamma ( a , b )$ , and define the kernel

$$
\kappa ( b , a ) = \frac { \Gamma ( a , b ) } { \pi ( b ) } \quad ( a \neq b ) , \qquad \kappa ( b , b ) = 1 - \sum _ { a \neq b } \kappa ( b , a ) .
$$

The conductance is largest for pairs of likely actions separated by little π-mass. These are the edges on which the random paths place the most flow, so the kernel observes each edge roughly in proportion to its use; the factor $1 / ( 4 \Lambda )$ keeps every row of κ a probability distribution.

Algorithm 1: Endpoint flow   
1 Procedure $\mathtt { P a i r } ( \pi )$ :   
2 Draw $A \sim \pi$ and then $B \sim \kappa ( A , \cdot )$ , where $\kappa ( b , a ) = \Gamma ( a , b ) / \pi ( b )$ for $a \neq b$ and   
$\begin{array} { r } { \kappa ( b , b ) = 1 - \sum _ { a \neq b } \kappa ( b , a ) ; } \end{array}$   
3 return $( A , B ) ;$   
4 Procedure Estimate $( \pi , A , B , g ( x _ { A } ) , g ( x _ { B } ) )$ :   
5 for $j = 1 , \ldots , m$ do   
6 $\begin{array} { r } { \widehat { g } ( j )  \frac { \phi _ { j } ( 0 , A ) } { \pi ( A ) } g ( x _ { A } ) ; } \end{array}$   
7 if $A \neq B$ then   
8 $\begin{array} { r } { \Big \lfloor \begin{array} { c c } { \widehat { g } ( j ) \gets \widehat { g } ( j ) + \frac { \phi _ { j } ( A \wedge B , A \vee B ) } { q ( A \wedge B , A \vee B ) } \big ( g ( x _ { A \vee B } ) - g ( x _ { A \wedge B } ) \big ) } \end{array} } \end{array}$   
9 return ${ \widehat { g } } ;$

Lemma 2 (Kernel validity). The kernel κ is valid, in particular $\kappa ( b , b ) \geq 1 / 2$ , and satisfies detailed balance

$$
\pi ( b ) \kappa ( b , a ) = \pi ( a ) \kappa ( a , b ) = \Gamma ( a , b ) .
$$

Hence π is stationary.

Draw $A \sim \pi$ and then $B \sim \kappa ( A , \cdot )$ . By stationarity, $B \sim \pi { \mathrm { ~ a s } }$ well, which is the marginal claim of the theorem; in particular, the expected paid average is $\langle \pi , g \rangle$ . For this pair, the revelation probabilities are

$$
q ( 0 , a ) = \pi ( a ) , \qquad q ( a , b ) = \pi ( a ) \kappa ( a , b ) + \pi ( b ) \kappa ( b , a ) = 2 \Gamma ( a , b ) \quad ( 1 \leq a < b ) .
$$

Estimator. The endpoint flow estimate is the estimator of Lemma 1 for this pair and the flows above. Since $H ( 0 , A ) = g ( x _ { A } )$ and $q ( 0 , A ) = \pi ( A )$ , it reads

$$
{ \widehat { g } } ( j ) = { \frac { \phi _ { j } ( 0 , A ) } { \pi ( A ) } } g ( x _ { A } ) + \mathbf { 1 } \{ A \neq B \} { \frac { \phi _ { j } ( A \land B , A \lor B ) } { q ( A \land B , A \lor B ) } } { \big ( } g ( x _ { A \lor B } ) - g ( x _ { A \land B } ) { \big ) } .
$$

When $A = B$ , the interval term is omitted; no flow or revelation probability is evaluated on a diagonal edge. The estimate uses only the two observed values and the known distribution $\pi .$ . By that lemma, $\widehat g$ is unbiased. The flows are nonnegative, and a nondecreasing g with values in $[ 0 , 1 ]$ has nonnegative increments, including the anchor increments $g ( x _ { a } ) - g ( x _ { 0 } ) = g ( x _ { a } )$ , so $\widehat g \geq 0$ in that case. Algorithm 1 summarizes the sampler and the estimator; our learning algorithms call its two procedures.

Second moment. The first lemma holds for the general estimator of the previous subsection.

Lemma 3 (Second moment). Under the assumptions of Lemma 1,

$$
\mathbb { E } \langle \pi , \widehat { g } ^ { 2 } \rangle \leq 2 \sum _ { e } \frac { H ( e ) ^ { 2 } } { q ( e ) } \sum _ { j } \pi ( j ) \phi _ { j } ( e ) ^ { 2 } ,
$$

where the outer sum runs over edges with $q ( e ) > 0$

The right side pairs two factors for each edge: the load it carries relative to its revelation probability, which depends only on π, and its squared increment. The next lemma bounds the per-edge loads; it is a discrete-integral computation, and the factor Λ in the interval bound is the price of the kernel normalization.

Lemma 4 (Edge loads). For every rank $a ,$

$$
\frac { \sum _ { j \geq a } \pi ( j ) \phi _ { j } ( 0 , a ) ^ { 2 } } { q ( 0 , a ) } = O \Bigl ( \frac { \pi ( a ) } { \Pi [ 1 , a ] } \Bigr ) ,
$$

and for every pair of ranks $1 \leq a < b _ { ; }$

$$
\frac { \sum _ { j \geq b } \pi ( j ) \phi _ { j } ( a , b ) ^ { 2 } } { q ( a , b ) } = O \Big ( \Lambda \frac { \pi ( a ) \pi ( b ) } { \Pi ( a , b ] \Pi [ a , b ] } \Big ) .
$$

The last lemma sums these weights against the squared increments. For monotone g, a layer-cake decomposition reduces the sum to edges crossing a single cut of the chain, whose total weight is a harmonic sum of order $\Lambda ;$ general g follows from the Jordan decomposition, which is where the variation of g enters.

Lemma 5 (Ordered cut bound). Let $H ( a , b ) = g ( x _ { b } ) - g ( x _ { a } )$ with $g ( x _ { 0 } ) = 0 \quad$ , and let $V =$ $| g ( x _ { 1 } ) | + \mathrm { T V } ( g )$ be the total variation $o f g$ on the chain extended by the anchor. Then

$$
\sum _ { a } H ( 0 , a ) ^ { 2 } { \frac { \pi ( a ) } { \Pi [ 1 , a ] } } + \sum _ { 1 \leq a < b } H ( a , b ) ^ { 2 } { \frac { \pi ( a ) \pi ( b ) } { \Pi ( a , b ] \Pi [ a , b ] } } = O ( \Lambda V ^ { 2 } ) .
$$

Under the hypotheses of the theorem, $V \leq 2$ , so combining the last three lemmas gives the second-moment bound of Theorem 1.

Self-bound. The self-bound is a direct computation: for every target $j \geq a , \pi ( j ) \phi _ { j } ( 0 , a ) / q ( 0 , a ) \leq$ 1, and for every target $j \ge b , \pi ( j ) \phi _ { j } ( a , b ) / q ( a , b ) \le 2 \Lambda$ . Thus the anchor and interval terms contribute at most 1 and 2Λ times their respective increment magnitudes. For $g$ with values in $[ 0 , 1 ] .$ every increment, including the anchor increments, has magnitude at most one, giving $1 + 2 \Lambda \leq 3 \Lambda$ The proofs of the lemmas of this section are in Section $\mathrm { A }$

## 4 The separated monotone problem

We now show how the endpoint flow estimator solves the separated monotone problem. Querying an action reveals $\ell _ { t } ^ { \uparrow }$ and $\ell _ { t } ^ { \downarrow }$ there separately, so the estimator can be applied to each component in its own orientation, and by the monotone-sign property both estimates are nonnegative. Let $R _ { T } ^ { ( 2 ) }$ denote the minimax expected regret of two-point learners over oblivious loss sequences of the separated monotone problem and expert recommendations. This section proves that $R _ { T } ^ { ( 2 ) }$ is $O ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } )$ and $\Omega ( { \sqrt { T } }$ log min $\{ | { \mathcal { F } } | , { \sqrt { T } } \} )$ .

## 4.1 Algorithm and upper bound

Let $u _ { t }$ be the current expert mixture, and let $\begin{array} { r } { p _ { t } ( j ) = \sum _ { f : j _ { t } ( f ) = j } u _ { t } ( f ) } \end{array}$ be the induced distribution on active ranks, which aggregates experts recommending the same action. Smooth directly on active actions:

$$
\pi _ { t } = { \Big ( } 1 - { \frac { 1 } { T } } { \Big ) } p _ { t } + { \frac { 1 } { T } } \operatorname { U n i f } ( X _ { t } ) .
$$

Algorithm 2: Endpoint flow with exponential weights for the separated monotone problem   
Input: learning rate $\eta > 0$   
1 Set $L _ { 0 } ( f ) = 0$ for every $f \in { \mathcal { F } } ;$   
2 for $t = 1 , \dots , T$ do   
3 Receive the recommendations and form the active chain $X _ { t } ;$   
4 Set $\begin{array} { r } { u _ { t } ( f ) \propto \exp ( - \eta L _ { t - 1 } ( f ) ) , p _ { t } ( j ) = \sum _ { f : j _ { t } ( f ) = j } u _ { t } ( f ) } \end{array}$ , and $\begin{array} { r } { \pi _ { t } = ( 1 - \frac { 1 } { T } ) p _ { t } + \frac { 1 } { T } } \end{array}$ Unif(X<sub>t</sub>);   
5 $( A _ { t } , B _ { t } ) \gets \mathtt { P a i r } ( \pi _ { t } ) ;$   
6 Play ${ \boldsymbol { x } } _ { t , A _ { t } }$ and $x _ { t , B _ { t } }$ , and observe $\ell _ { t } ^ { \uparrow }$ and $\ell _ { t } ^ { \downarrow }$ at both;   
7 $\widehat { \ell } _ { t } ^ { \dagger } \gets$ Estimate $( \pi _ { t } , A _ { t } , B _ { t } , \ell _ { t } ^ { \uparrow } ( x _ { t , A _ { t } } ) , \ell _ { t } ^ { \uparrow } ( x _ { t , B _ { t } } ) )$ on the chain read forward;   
8 $\widehat { \ell _ { t } ^ { \downarrow } } \gets$ Estimate $( \pi _ { t } , A _ { t } , B _ { t } , \ell _ { t } ^ { \downarrow } ( x _ { t , A _ { t } } ) , \ell _ { t } ^ { \downarrow } ( x _ { t , B _ { t } } ) )$ on the chain read backward;   
9 Set $L _ { t } ( f ) = L _ { t - 1 } ( f ) + \widehat { \ell } _ { t } ^ { \uparrow } ( j _ { t } ( f ) ) + \widehat { \ell } _ { t } ^ { \downarrow } ( j _ { t } ( f ) )$ for every $f \in { \mathcal { F } } ;$

Run the endpoint flow sampler using $\pi _ { t } .$ . Read the active chain forward for $\ell _ { t } ^ { \uparrow }$ and backward for $\ell _ { t } ^ { \downarrow }$ reading the chain backward replaces each rank $j$ by $m _ { t } + 1 - j$ in $\pi _ { t } ,$ in the pair, and in the output of the estimator. Reversing the order preserves the unordered interval and its $\pi _ { t } { \mathrm { - } } \operatorname* { m a s s } ,$ so the same paid pair can be inserted into both oriented estimators. Each estimate is nonnegative, and their sum is unbiased for the total loss. Algorithm 2 summarizes the procedure.

Because the estimates are nonnegative, the standard second-order exponential-weights bound applies.

Lemma 6 (Exponential weights for nonnegative estimates). Let $u _ { t }$ be the exponential-weights distribution generated by nonnegative loss estimates $\widehat { \ell } _ { t } ( f ) \geq 0$

$$
u _ { t } ( f ) = \frac { \exp ( - \eta L _ { t - 1 } ( f ) ) } { \sum _ { f ^ { \prime } } \exp ( - \eta L _ { t - 1 } ( f ^ { \prime } ) ) } , \qquad L _ { t } ( f ) = L _ { t - 1 } ( f ) + \widehat { \ell } _ { t } ( f ) .
$$

Then for every f<sup>⋆</sup>, $f ^ { \star }$

$$
\sum _ { t = 1 } ^ { T } \langle u _ { t } , \widehat { \ell } _ { t } \rangle - \sum _ { t = 1 } ^ { T } \widehat { \ell } _ { t } ( f ^ { \star } ) \leq \frac { \log | \mathcal { F } | } { \eta } + \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } \langle u _ { t } , \widehat { \ell } _ { t } ^ { 2 } \rangle .
$$

Theorem 2 (Separated two-point upper bound). Let $\Lambda = 1 + \log ( | \mathcal { F } | T )$ . For every oblivious loss sequence of the separated monotone problem and arbitrary current recommendations, Algorithm 2 with $\eta = \sqrt { \log | \mathcal { F } | / ( \Lambda ^ { 2 } T ) }$ satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) = O \big ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } \big ) .
$$

Each component takes values in $[ 0 , 1 / 2 ]$ and has total variation at most $1 / 2$ , so the theorem applies to it in its own orientation, and the sum of the two reconstructions has expert-weighted second moment $O ( \Lambda _ { t } ^ { 2 } )$ , where $\Lambda _ { t } = 1 + \log ( m _ { t } T ) \leq 1 + \log ( | \mathcal { F } | T )$ . Pathwise nonnegativity permits the second-order exponential-weights bound of the lemma above, and the stated learning rate makes both of its terms $O ( \Lambda { \sqrt { T \log | { \mathcal { F } } | } } )$ . The smoothing itself costs at most one in cumulative loss.

## 4.2 Lower bound: a hidden dyadic path

We now complement the upper bound with a lower bound that holds even for fixed recommendations.

Theorem 3 (Two-point lower bound). For all $| \mathcal { F } | , T \ge 2$ and every two-point learner, some oblivious instance of the separated monotone problem forces

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) = \Omega \big ( \sqrt { T } \log \operatorname* { m i n } \{ \vert \mathcal { F } \vert , \sqrt { T } \} \big ) .
$$

In the instance every expert recommends one fixed action on every round. Every loss has total variation at most one, and observing only the losses at the two queried actions reveals the same information as observing both components, so the bound also holds for the BV problem.

The hard instance uses threshold losses $\begin{array} { r } { \ell _ { t } ( x ) = \frac { 1 - x } { 2 } + \frac 1 2 { \bf 1 } \{ Z _ { t } < x \} } \end{array}$ on $[ 0 , 1 ]$ , so a query at $x$ reveals only whether the random threshold $Z _ { t }$ lies below $x ,$ and a simultaneous pair reveals only which of three bins contains $Z _ { t }$ . Experts sit at the centers of the $2 ^ { k }$ dyadic cells of depth $k \approx \log _ { 2 } \operatorname* { m i n } \{ | \mathcal { F } | , \sqrt { T } \}$ . The adversary hides a leaf $\theta$ and tilts the law of $Z _ { t }$ by $\delta \asymp 1 / \sqrt { T }$ times the sum of the Haar functions along the dyadic path to $\theta ;$ this lowers the mean loss of the expert at $\theta$ by order $\delta k$ relative to any action chosen without knowledge of $\theta .$

Averaged over $\theta ,$ the tilt moves the probability of a bin of length r by only $O ( \delta \sqrt { r } )$ , so each simultaneous pair carries $O ( \delta ^ { 2 } )$ relative entropy about $\theta .$ Over $T$ rounds the learner gathers $O ( 1 )$ information, cannot locate $\theta ,$ and loses $\Omega ( \delta k )$ per round, a total of $\Omega ( { \sqrt { T } }$ log min $\{ | { \mathcal { F } } | , { \sqrt { T } } \}$ ). Proofs are in Section B.

## 5 The BV problem

We now turn to the BV problem, where querying x reveals only the loss $\ell _ { t } ( x )$ . Endpoint flow remains unbiased for the loss vector, but the estimate is signed, so the exponential-weights analysis of the separated monotone problem no longer applies.

## 5.1 Algorithm and upper bound

For signed estimates we use mirror descent with a regularizer that combines negative entropy with a log barrier. Its regret bound requires only a coordinatewise bound on $w _ { t } ( f ) | \widehat { \ell _ { t } } ( f ) |$ , which is exactly the self-bound of endpoint flow in Theorem 1. Write $\Delta ^ { \circ } ( { \mathcal { F } } )$ for the set of distributions on $\mathcal { F }$ with full support, and let

$$
D _ { \Psi } ( u , w ) = \Psi ( u ) - \Psi ( w ) - \langle \nabla \Psi ( w ) , u - w \rangle
$$

be the Bregman divergence generated by a diferentiable convex regularizer $\Psi$

Proposition 1 (Robust mirror descent). Let $\widehat { \ell } _ { 1 } , \dots , \widehat { \ell } _ { T }$ be signed loss estimates on $\mathcal { F } _ { \mathbf { \Phi } }$ , let $\mu > 0$ and run the update

$$
w _ { t + 1 } = \arg \operatorname* { m i n } _ { w \in \Delta ^ { \circ } ( \mathcal { F } ) } \left\{ D _ { \Psi } ( w , w _ { t } ) + \eta \langle w , \widehat { \ell } _ { t } \rangle \right\}
$$

with the regularizer

$$
\Psi ( w ) = \sum _ { f } w ( f ) \log w ( f ) - \mu \sum _ { f } \log w ( f ) .
$$

Suppose that max<sub>f</sub> $w _ { t } ( f ) | \widehat { \ell _ { t } } ( f ) | \leq \beta$ on every round. $I f \mu \geq 2 \eta \beta$ , then for every $u \in \Delta ^ { \circ } ( \mathcal { F } )$

$$
\sum _ { t = 1 } ^ { T } \langle w _ { t } - u , \widehat { \ell } _ { t } \rangle \leq \frac { D _ { \Psi } ( u , w _ { 1 } ) } { \eta } + \eta \sum _ { t = 1 } ^ { T } \sum _ { f } w _ { t } ( f ) \widehat { \ell } _ { t } ( f ) ^ { 2 } .
$$

Algorithm 3 runs the endpoint flow sampler and feeds the signed loss estimate to this update.

Algorithm 3: Endpoint flow with robust mirror descent for the BV problem   
Input: learning rate $\eta > 0$ , barrier weight $\mu > 0$   
1 Set $w _ { 1 } ( f ) = 1 / | \mathcal { F } |$ for every $f \in { \mathcal { F } } ,$ , and let $\begin{array} { r } { \Psi ( w ) = \sum _ { f } w ( f ) \log w ( f ) - \mu \sum _ { f } \log w ( f ) ; } \end{array}$   
2 for $t = 1 , \dots , T$ do   
3 Receive the recommendations and form the active chain $X _ { t } ;$   
4 Set $\begin{array} { r } { p _ { t } ( j ) = \sum _ { f : j _ { t } ( f ) = j } w _ { t } ( f ) \mathrm { ~ a n d ~ } \pi _ { t } = ( 1 - \frac 1 T ) p _ { t } + \frac { 1 } { T } \operatorname { U n i f } ( X _ { t } ) ; } \end{array}$   
5 $( A _ { t } , B _ { t } ) \gets \mathtt { P a i r } ( \pi _ { t } ) ;$   
6 Play $x _ { t , A _ { t } }$ and $x _ { t , B _ { t } }$ , and observe $\ell _ { t }$ at both;   
7 $\widehat { \ell _ { t } } \gets \mathtt { E s t i m a t e } ( \pi _ { t } , A _ { t } , B _ { t } , \ell _ { t } \big ( x _ { t , A _ { t } } \big ) , \ell _ { t } \big ( x _ { t , B _ { t } } \big ) )$   
8 Set $\begin{array} { r } { w _ { t + 1 } = \arg \operatorname* { m i n } _ { w \in \Delta ^ { \circ } ( \mathcal { F } ) } \big \{ D _ { \Psi } ( w , w _ { t } ) + \eta \sum _ { f } w ( f ) \widehat \ell _ { t } ( j _ { t } ( f ) ) \big \} } \end{array}$

Theorem 4 (BV two-point upper bound). Let $\Lambda = 1 + \log ( | \mathcal { F } | T )$ . For every oblivious loss sequence of the BV problem and arbitrary current recommendations, Algorithm 3 with $\eta = \sqrt { \log | \mathcal { F } | / ( \Lambda ^ { 2 } T ) }$ and $\mu = 1 2 \eta \Lambda$ satisfies

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) = O \big ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } + | \mathcal { F } | \log ( | \mathcal { F } | T ) \log T \big ) = \widetilde { O } \big ( \sqrt { T } + | \mathcal { F } | \big ) .
$$

The additive $| \mathcal F |$ comes from the barrier diameter required to stabilize a signed importanceweighted estimator; it is not merely an artifact of the analysis, as the next lower bound shows.

## 5.2 Lower bounds

Theorem 5 (Needle lower bound). For every two-point algorithm for the BV problem, there is an oblivious loss sequence of the BV problem such that

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) = \Omega \big ( \operatorname* { m i n } \{ | \mathcal { F } | , T \} \big ) .
$$

The instance is literal search. Let $M = \operatorname* { m i n } \{ | \mathcal { F } | , T \}$ and hide one action $i ^ { \star }$ with loss zero; every other candidate has loss $1 / 2$ on every round. Until $i ^ { \star }$ is queried, every feedback history is identical. Two simultaneous queries inspect at most two new candidates, so the expected discovery time is Ω(M). Every loss has total variation at most one.

The two-point lower bound of Theorem 3 also applies to the BV problem, and together with the needle it determines the two-point BV rate up to logarithmic factors.

Corollary 1 (Two-point BV rate). For all $| \mathcal { F } | , T \geq 2$ and every two-point algorithm for the BV problem, some oblivious loss sequence of the BV problem forces

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) = \Omega \Big ( \sqrt { T } \log \operatorname* { m i n } \{ | \mathcal { F } | , \sqrt { T } \} + \operatorname* { m i n } \{ | \mathcal { F } | , T \} \Big ) .
$$

Together with the upper bound above, the two-point minimax regret for the BV problem is $\widetilde { \Theta } ( \sqrt { T } + | \mathcal { F } | )$ when $| \mathcal { F } | \le T$

The remaining proofs are in Section C.

## 6 Dynamic pricing with expert advice

In this section, we give a pricing-native application of online learning with ordered experts with two point feedback.

## 6.1 Model and the single-price benchmark

On round t, a seller faces a fresh risk-neutral buyer for one item. An oblivious adversary fixes the private valuations $v _ { 1 } , \ldots , v _ { T } \in [ 0 , 1 ]$ before the interaction; valuations may vary arbitrarily across rounds. Before observing $v _ { t }$ , the seller ofers a finite menu

$$
\mathcal { M } _ { t } \subseteq [ 0 , 1 ] \times [ 0 , 1 ] , \qquad ( 0 , 0 ) \in \mathcal { M } _ { t } .
$$

An entry $( \alpha , \tau )$ specifies an allocation probability α and an expected payment τ . The buyer selects an entry maximizing $v _ { t } \alpha - \tau$ , with ties broken in favor of the larger allocation probability. This is the menu interpretation of an incentive-compatible mechanism for a risk-neutral buyer; the outside option gives nonnegative expected utility. The seller observes the selected entry $( \alpha _ { t } , \tau _ { t } )$ , which gives expected revenue $\tau _ { t }$

A posted price $p$ is the menu $\{ ( 0 , 0 ) , ( 1 , p ) \}$ , with revenue

$$
r _ { t } ( p ) = p { \bf 1 } \{ v _ { t } \geq p \} .
$$

We retain the standard benchmark of the best single posted price, fixed across the entire horizon [Kleinberg and Leighton, 2003]. Revenue regret is

$$
\mathrm { R e g } _ { T } ^ { \mathrm { p r i c e } } = \operatorname* { m a x } _ { p \in [ 0 , 1 ] } \sum _ { t = 1 } ^ { T } r _ { t } ( p ) - \sum _ { t = 1 } ^ { T } \tau _ { t } ,
$$

and we bound $\mathbb { E } \mathrm { R e g } _ { T } ^ { \mathrm { p r i c e } }$ . The learner’s menu may change each round, while the benchmark uses the same price throughout.

In fact, allowing a fixed menu in the benchmark would give the same optimal revenue. For any fixed finite menu M, its selected allocation probability is nondecreasing in the valuation. At a threshold $p _ { j }$ , a jump of size $w _ { j }$ in allocation probability raises the payment by $w _ { j } p _ { j }$ , by indiference between the adjacent selected entries. Starting from zero payment at valuation zero, this gives

$$
\tau _ { \mathcal { M } } ( v ) = \sum _ { j } w _ { j } p _ { j } \mathbf { 1 } \{ v \geq p _ { j } \} , \qquad w _ { j } \geq 0 , \qquad \sum _ { j } w _ { j } \leq 1 .
$$

Consequently, $\begin{array} { r } { \sum _ { t } \tau _ { \mathcal { M } } ( v _ { t } ) \leq \operatorname* { m a x } _ { p } \sum _ { t } r _ { t } ( p ) } \end{array}$ . Since every posted price is itself a menu, the best fixed menu and the best fixed posted price have equal revenue.

## 6.2 From ordered experts to a two-option menu

The ordered-expert problem. Fix a finite price grid ${ \mathcal { P } } \subseteq [ 0 , 1 ]$ . Each price $p \in \mathcal P$ defines an expert $f _ { p }$ that recommends p on every round, so the best expert is exactly the best fixed grid price. The actions are these prices in their usual order. Normalize revenue shortfall as

$$
\ell _ { t } ( p ) = \frac { 1 - r _ { t } ( p ) } { 2 } , \qquad \ell _ { t } ^ { \uparrow } ( p ) = \frac { p } { 2 } { \bf 1 } \{ v _ { t } < p \} , \qquad \ell _ { t } ^ { \downarrow } ( p ) = \frac { 1 - p } { 2 } .
$$

For every valuation $v _ { t } ,$ the first component is nondecreasing in price and the second is nonincreasing, both lie in $[ 0 , 1 / 2 ]$ , and $\ell _ { t } = \ell _ { t } ^ { \uparrow } + \ell _ { t } ^ { \downarrow }$ . They also satisfy the endpoint normalizations. Thus the losses have exactly the structure of the separated monotone problem. An acceptance decision at p reveals both components there; to run Algorithm 2, we need these observations at two prices on the same valuation.

The endpoint-flow pair becomes a menu. The endpoint-flow sampler used by Algorithm 2 draws two prices $P _ { t } , Q _ { t }$ , each with the learner’s intended smoothed marginal $\pi _ { t }$ . Write

$$
\begin{array} { r } { p _ { t } ^ { - } = \operatorname* { m i n } \{ P _ { t } , Q _ { t } \} , \qquad p _ { t } ^ { + } = \operatorname* { m a x } \{ P _ { t } , Q _ { t } \} . } \end{array}
$$

This pair naturally gives two purchase options: a half chance of allocation for expected payment $p _ { t } ^ { - } / 2$ , or certain allocation for expected payment $( p _ { t } ^ { - } + p _ { t } ^ { + } ) / 2$ . Together with the outside option, the menu is

$$
\mathcal { M } _ { t } = \left\{ ( 0 , 0 ) , \left( \frac { 1 } { 2 } , \frac { p _ { t } ^ { - } } { 2 } \right) , \left( 1 , \frac { p _ { t } ^ { - } + p _ { t } ^ { + } } { 2 } \right) \right\} .
$$

The buyer’s utility from the first purchase option is $( v _ { t } - p _ { t } ^ { - } ) / 2$ . Upgrading to the second adds another half chance of allocation for an additional expected payment $p _ { t } ^ { + } / 2$ , so its incremental utility is $( v _ { t } - p _ { t } ^ { + } ) / 2$ . Consequently, under our tie-breaking rule,

$$
\left( \alpha _ { t } , \tau _ { t } \right) = \left\{ \begin{array} { l l } { ( 0 , 0 ) , } & { v _ { t } < p _ { t } ^ { - } , } \\ { ( 1 / 2 , p _ { t } ^ { - } / 2 ) , } & { p _ { t } ^ { - } \le v _ { t } < p _ { t } ^ { + } , } \\ { ( 1 , ( p _ { t } ^ { - } + p _ { t } ^ { + } ) / 2 ) , } & { v _ { t } \ge p _ { t } ^ { + } . } \end{array} \right.
$$

The menu choice therefore reveals whether the buyer accepts neither price, only the lower price, or both. This gives the two revenues $r _ { t } ( P _ { t } ) , r _ { t } ( Q _ { t } )$ and hence the separated loss observations required by the endpoint flow estimator. If the sampled prices coincide, the middle case is empty and the menu behaves as that single posted price.

The same feedback and payof. The expected payment and normalized loss obey

$$
\tau _ { t } = \frac { r _ { t } ( P _ { t } ) + r _ { t } ( Q _ { t } ) } { 2 } , \qquad \frac { \ell _ { t } ( P _ { t } ) + \ell _ { t } ( Q _ { t } ) } { 2 } = \frac { 1 - \tau _ { t } } { 2 } .
$$

Thus the two purchase options implement the two paid observations supplied to the forward and backward endpoint-flow estimates in Algorithm 2. Since both sampled prices have marginal $\pi _ { t } .$ , the menu’s expected revenue is the revenue of the learner’s intended price mixture. Revenue regret against the best grid price is twice the algorithm’s loss regret against the constant-price experts.

Pricing with expert advice. Expert $f$ may recommend a diferent price $a _ { t , f }$ on each round, for example by mapping an exogenous public context $z _ { t }$ to $f ( z _ { t } )$ . The valuation and recommendation sequences are oblivious. The benchmark follows the best single expert throughout; constant recommendations give the fixed-price comparison.

Theorem 6 (Pricing with menus). Online mechanisms ofering two purchase options and an outside option can achieve the following expected revenue-regret guarantees:

(i) No expert advice. Against the best fixed posted price,

$$
\operatorname* { m a x } _ { p \in [ 0 , 1 ] } \sum _ { t = 1 } ^ { T } r _ { t } ( p ) - \mathbb { E } \sum _ { t = 1 } ^ { T } \tau _ { t } \leq O \big ( \sqrt { T } \log ^ { 3 / 2 } T \big ) .
$$

(ii) Expert advice. Given a finite class $\mathcal { F }$ with oblivious price recommendations $a _ { t , f } \in [ 0 , 1 ]$ supplied before each menu,

$$
\operatorname* { m a x } _ { f \in \mathcal { F } } \sum _ { t = 1 } ^ { T } r _ { t } \big ( a _ { t , f } \big ) - \mathbb { E } \sum _ { t = 1 } ^ { T } \tau _ { t } \leq O \big ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } \big ) .
$$

Both mechanisms are incentive compatible and individually rational in expected utility.

The proof of both guarantees appear in Section E.

## 7 One-point and full-information feedback

To calibrate the two-point results, we compare them with two other protocols. Under one-point feedback, the learner chooses one active action $A _ { t } ,$ incurs $\ell _ { t } ( A _ { t } )$ , and receives the problem-specific observation at that point; regret is defined as for two points, with the paid average replaced by $\ell _ { t } ( A _ { t } )$ . Under full information, the learner chooses its action before seeing the current loss and then observes the full loss vector on the active recommendations.

## 7.1 One-point feedback in the separated monotone problem

Theorem 7 (Separated one-point lower bound). For every randomized one-point learner and all suficiently large T, there is a deterministic oblivious instance of the separated monotone problem with constant recommendations such that

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) = \Omega \big ( \operatorname* { m i n } \{ \sqrt { | \mathcal { F } | T } , T ^ { 2 / 3 } \} \big ) .
$$

The instance hides one slightly better action inside a monotone component. Keeping the component monotone requires a compensating slope, which limits the number of hidden candidates to $T ^ { 1 / 3 }$

## 7.2 One-point feedback in the BV problem

Theorem 8 (One-point BV feedback). For $2 \leq | \mathcal { F } | \leq T$ , the minimax one-point regret in the BV problem satisfies $R _ { T } ^ { ( 1 ) } = \Theta ( \sqrt { | \mathcal { F } | T } )$ , and the lower bound holds with losses that also belong to the separated monotone problem.

The upper bound is the standard adversarial bandit guarantee with $| \mathcal F |$ arms. The lower bound hides one slightly better action among |F| candidates; all actions share a common random baseline on each round, so every loss has small total variation.

## 7.3 Full information: order alone does not help

Theorem 9 (Full-information benchmark). For log $| \mathcal { F } | \le T$ , the minimax full-information regret in both problems satisfies $R _ { T } ^ { \mathrm { f u l l } } = \Theta ( \sqrt { T \log | \mathcal { F } | } )$ . The lower bound holds even when every expert recommends one fixed action on every round, and its argument does not use what the learner observes.

Exponential weights gives the upper bound, and a nested-block construction gives the lower bound. Proofs are in Section D.

## A Proofs for endpoint flow

This appendix proves the lemmas used in the proof of Theorem 1.

## A.1 Unbiasedness and second moment

Write $\widehat g ( j ) = z _ { 1 } ( j ) + z _ { 2 } ( j )$ , where

$$
z _ { 1 } ( j ) = \sum _ { a } \mathbf { 1 } \{ E _ { 1 } = ( 0 , a ) \} \frac { \phi _ { j } ( 0 , a ) } { q ( 0 , a ) } H ( 0 , a ) , \qquad z _ { 2 } ( j ) = \sum _ { 1 < a < b } \mathbf { 1 } \{ E _ { 2 } = ( a , b ) \} \frac { \phi _ { j } ( a , b ) } { q ( a , b ) } H ( a , b )
$$

are the two sums in the definition of ${ \widehat { g } } .$ At most one indicator in each sum equals one.

Proof of Lemma 1. Since $E _ { 1 } = ( 0 , a )$ with probability $q ( 0 , a )$

$$
\mathbb { E } z _ { 1 } ( j ) = \sum _ { a : q ( 0 , a ) > 0 } q ( 0 , a ) \frac { \phi _ { j } ( 0 , a ) } { q ( 0 , a ) } H ( 0 , a ) = \sum _ { a } \phi _ { j } ( 0 , a ) H ( 0 , a ) ,
$$

where the last step uses that $q ( 0 , a ) > 0$ whenever $\phi _ { j } ( 0 , a ) > 0$ . In the same way, $\mathbb { E } z _ { 2 } ( j ) =$ $\begin{array} { r } { \sum _ { 1 < a < b } \phi _ { j } ( a , b ) H ( a , b ) } \end{array}$ . Adding the two and using the flow identity $\begin{array} { r } { g ( x _ { j } ) = \sum _ { e } \phi _ { j } ( e ) H ( e ) } \end{array}$ gives $\mathbb { E } \widehat { g } ( \bar { j } ) \ : = \ : g ( x _ { j } )$ . If $g$ is nonnegative and nondecreasing, then every increment is nonnegative, including the anchor increments $H ( 0 , a ) = g ( x _ { a } )$ ; since the flows are nonnegative, $z _ { 1 } ( j )$ and $z _ { 2 } ( j )$ are nonnegative for every realization. □

Proof of Lemma 3. By $( z _ { 1 } + z _ { 2 } ) ^ { 2 } \leq 2 z _ { 1 } ^ { 2 } + 2 z _ { 2 } ^ { 2 }$

$$
\mathbb { E } \widehat { g } ( j ) ^ { 2 } \leq 2 \mathbb { E } z _ { 1 } ( j ) ^ { 2 } + 2 \mathbb { E } z _ { 2 } ( j ) ^ { 2 } = 2 \sum _ { a } \frac { \phi _ { j } ( 0 , a ) ^ { 2 } } { q ( 0 , a ) } H ( 0 , a ) ^ { 2 } + 2 \sum _ { 1 \leq a < b } \frac { \phi _ { j } ( a , b ) ^ { 2 } } { q ( a , b ) } H ( a , b ) ^ { 2 } ,
$$

with both sums over edges of positive revelation probability. Multiplying by $\pi ( j )$ , summing over $j ,$ and exchanging the target and edge sums proves the lemma. No independence between $E _ { 1 }$ and $E _ { 2 }$ is needed. □

## A.2 Kernel validity

Proof of Lemma 2. Symmetry of Γ gives detailed balance. For a fixed $b ,$ the of-diagonal row mass is

$$
\frac { 1 } { 4 \Lambda } \left( \sum _ { a < b } \frac { \pi ( a ) } { \Pi [ a , b ] } + \sum _ { a > b } \frac { \pi ( a ) } { \Pi [ b , a ] } \right) .
$$

Both sums are harmonic. For the first, set $S _ { a } = \Pi [ a , b ]$ . Since

$$
{ \frac { \pi ( a ) } { S _ { a } } } = { \frac { S _ { a } - S _ { a + 1 } } { S _ { a } } } \leq \log S _ { a } - \log S _ { a + 1 } ,
$$

we get

$$
\sum _ { a < b } \frac { \pi ( a ) } { \Pi [ a , b ] } \leq \log \frac { \Pi [ 1 , b ] } { \pi ( b ) } \leq \log \frac { m } { \xi } \leq \Lambda .
$$

The second sum is identical by symmetry. Thus the of-diagonal row mass is at most $1 / 2$ □

## A.3 Random-path edge probabilities

Fix target $j .$ . Starting from the anchor 0, at node $a < j$ the path jumps to $b \in \{ a + 1 , \ldots , j \}$ with probability $\pi ( b ) / \Pi ( a , j ]$

Lemma 7 (Edge-use probabilities). The probabilities that the path uses the anchor edge $( 0 , a )$ and the interval edge $( a , b )$ are

$$
\phi _ { j } ( 0 , a ) = \frac { \pi ( a ) } { \Pi [ 1 , j ] } \quad ( a \leq j ) , \qquad \phi _ { j } ( a , b ) = \frac { \pi ( a ) \pi ( b ) } { \Pi [ a , j ] \Pi ( a , j ] } \quad ( 1 \leq a < b \leq j ) .
$$

The edge-use probability is zero outside $0 \leq a < b \leq j$

Proof. The path reaches $a \leq j$ with probability $\pi ( a ) / \Pi [ a , j ]$ . This is proved inductively using

$$
\frac { \pi ( r ) } { \Pi [ r , j ] \Pi ( r , j ] } = \frac { 1 } { \Pi ( r , j ] } - \frac { 1 } { \Pi [ r , j ] } .
$$

Multiplying the reach probability of a by the next-step probability $\pi ( b ) / \Pi ( a , j ]$ gives the formula. The anchor case is the same calculation starting at 0. □

## A.4 Discrete integral and edge-load estimates

We use the following elementary estimate.

Lemma 8 (Discrete integral bounds). Let $0 \leq s _ { b - 1 } < s _ { b } < s _ { b + 1 } < \cdot \cdot \cdot$ and $\delta _ { j } = s _ { j } - s _ { j - 1 }$ . Then

$$
\sum _ { j \geq b } { \frac { \delta _ { j } } { s _ { j } ^ { 2 } } } \leq { \frac { 2 } { s _ { b } } } .
$$

For every $\alpha > 0$

$$
\sum _ { j \geq b } \frac { \delta _ { j } } { s _ { j } ^ { 2 } ( \alpha + s _ { j } ) ^ { 2 } } = O \Bigl ( \frac { 1 } { s _ { b } ( \alpha + s _ { b } ) ^ { 2 } } \Bigr ) .
$$

Proof. For the first claim, $\delta _ { b } \leq s _ { b }$ and the function $s \mapsto s ^ { - 2 }$ is decreasing, so

$$
\sum _ { j \ge b } \frac { \delta _ { j } } { s _ { j } ^ { 2 } } \le \frac { 1 } { s _ { b } } + \int _ { s _ { b } } ^ { \infty } \frac { d s } { s ^ { 2 } } = \frac { 2 } { s _ { b } } .
$$

For the second claim, apply the same comparison to $s \mapsto s ^ { - 2 } ( \alpha + s ) ^ { - 2 }$ and use $( \alpha + s ) ^ { - 2 } \leq ( \alpha + s _ { b } ) ^ { - 2 }$ for $s \geq s _ { b }$ . This gives an upper bound of $2 / [ s _ { b } ( \alpha + s _ { b } ) ^ { 2 } ]$ □

Proof of Lemma $\it 4 .$ For the anchor edge $( 0 , a )$ , substitute $\phi _ { j } ( 0 , a ) = \pi ( a ) / \Pi [ 1 , j ]$ and $q ( 0 , a ) = \pi ( a )$ and apply the first part of Lemma $8 ,$ starting at index $a ,$ to the cumulative masses $s _ { j } = \Pi [ 1 , j ]$ For an interval edge, set $s _ { j } = \Pi ( a , j ]$ and $\alpha = \pi ( a )$ , apply the second discrete-integral bound, then divide by $q ( a , b ) = 2 \Gamma ( a , b ) = \pi ( a ) \pi ( b ) / ( 2 \Lambda \Pi [ a , b ] )$ □

## A.5 Ordered cut bound

Proof of Lemma 5. First suppose g is nondecreasing on the chain including the anchor, with $g ( x _ { 0 } ) = 0$ and $g ( x _ { m } ) = V$ . Since $0 \leq H ( a , b ) \leq V$

$$
H ( a , b ) ^ { 2 } \leq V H ( a , b ) = V \int _ { 0 } ^ { V } \mathbf { 1 } \{ g ( x _ { a } ) \leq z < g ( x _ { b } ) \} d z .
$$

For fixed $z \in [ 0 , V )$ , monotonicity makes the lower-level set an initial segment $\{ 0 , \ldots , k \}$ with $k < m$ Thus the contributing edges cross a single ordered cut. For the anchor edges,

$$
\sum _ { b > k } \frac { \pi ( b ) } { \Pi [ 1 , b ] } \leq \sum _ { b = 1 } ^ { m } \frac { \pi ( b ) } { \Pi [ 1 , b ] } \leq 1 + \log \frac { 1 } { \pi ( 1 ) } \leq \Lambda .
$$

For an interval edge, the identity

$$
\frac { \pi ( a ) } { \Pi ( a , b ] \Pi [ a , b ] } = \frac { 1 } { \Pi ( a , b ] } - \frac { 1 } { \Pi [ a , b ] }
$$

gives, for $k \geq 1$ and $b > k$

$$
\sum _ { a = 1 } ^ { k } \frac { \pi ( a ) \pi ( b ) } { \Pi ( a , b ] \Pi [ a , b ] } = \pi ( b ) \left( \frac { 1 } { \Pi ( k , b ] } - \frac { 1 } { \Pi [ 1 , b ] } \right) \leq \frac { \pi ( b ) } { \Pi ( k , b ] } .
$$

Summing over $b > k$ gives at most $1 + \log ( 1 / \pi ( k + 1 ) ) \leq \Lambda$ by the same harmonic bound. When $k = 0$ there are no interval edges crossing the cut. The total cut weight is therefore at most $2 \Lambda$ , and integrating over an interval of length V yields $O ( \Lambda V ^ { 2 } )$

For a general g, take the Jordan decomposition $g = g ^ { + } - g ^ { - }$ with $g ^ { \pm } ( x _ { 0 } ) = 0$ and both nondecreasing. Then

$$
H ^ { 2 } \leq 2 ( H ^ { + } ) ^ { 2 } + 2 ( H ^ { - } ) ^ { 2 } ,
$$

and $V ^ { + } + V ^ { - } = | g ( x _ { 1 } ) | + \mathrm { T V } ( g ) = V$ , the total variation of $g$ on the chain extended by the anchor. Applying the monotone bound to both pieces proves the lemma. □

## A.6 Coordinatewise self-bound

Lemma 9 (Self-bound). For every realization and every active rank $j$

$$
\pi ( j ) | \widehat { g } ( j ) | \leq 3 \Lambda \operatorname* { m a x } _ { 0 \leq a < b \leq m } | g ( x _ { b } ) - g ( x _ { a } ) | ,
$$

where $g ( x _ { 0 } ) = 0$ . In particular, the right side is at most 3Λ when g takes values in [0, 1].

Proof. For the anchor term, if $A = a \le j$ , then

$$
z _ { 1 } ( j ) = \frac { g ( x _ { a } ) - g ( x _ { 0 } ) } { \Pi [ 1 , j ] } ,
$$

so multiplication by $\pi ( j )$ gives at most the increment magnitude; if $A > j$ , then $z _ { 1 } ( j ) = 0$ . For the interval term, when $a < b \leq j$

$$
\frac { \phi _ { j } ( a , b ) } { 2 \Gamma ( a , b ) } = \frac { 2 \Lambda \Pi [ a , b ] } { \Pi [ a , j ] \Pi ( a , j ] } .
$$

Because $\Pi [ a , b ] \leq \Pi [ a , j ]$ and $\pi ( j ) \leq \Pi ( a , j ]$ , multiplying by $\pi ( j )$ leaves at most 2Λ times the observed increment. Adding the two terms and using $\Lambda \geq 1$ proves the claim. □

## B Proofs for the separated monotone problem

This appendix contains the proofs for Section 4.

## B.1 Exponential weights for nonnegative estimates

Proof of Lemma 6. Let

$$
\Phi ( L ) = - \frac { 1 } { \eta } \log \left( \frac { 1 } { | \mathcal { F } | } \sum _ { f } e ^ { - \eta L ( f ) } \right) .
$$

For the exponential-weights distribution $u _ { t } = \nabla \Phi ( L _ { t - 1 } )$ and a nonnegative estimate $\widehat { \ell } _ { t } .$ , the inequality $e ^ { - x } \leq 1 - x + x ^ { 2 } / 2$ for $x \geq 0$ gives

$$
\langle u _ { t } , \widehat { \ell } _ { t } \rangle \leq \Phi ( L _ { t } ) - \Phi ( L _ { t - 1 } ) + \frac { \eta } { 2 } \langle u _ { t } , \widehat { \ell } _ { t } ^ { 2 } \rangle .
$$

Summing over t and using

$$
\Phi ( L _ { T } ) \leq L _ { T } ( f ^ { \star } ) + \frac { \log | \mathcal { F } | } { \eta }
$$

proves the lemma.

## B.2 Separated two-point upper bound

Proof of Theorem 2. Let $u _ { t }$ be the expert mixture before round t, and let $p _ { t }$ be its induced distri  
bution on the distinct active actions. Smooth with $\xi = 1 / T$ 2

$$
\pi _ { t } = ( 1 - \xi ) p _ { t } + \xi \operatorname { U n i f } ( X _ { t } ) ,
$$

and write

$$
\Lambda _ { t } = 1 + \log \frac { m _ { t } } { \xi } \leq 1 + \log ( | \mathcal { F } | T ) .
$$

Run the endpoint flow sampler with target marginal $\pi _ { t }$

Read $\ell _ { t } ^ { \uparrow }$ in the forward order from the anchor below the chain and $\ell _ { t } ^ { \downarrow }$ in the reversed order from an anchor above the chain. Reversal preserves every unordered interval and its $\pi _ { t ^ { - } } { \mathrm { m a s s } }$ , so the conductances and hence the sampled pair are identical in the two orientations. The same paid pair therefore feeds both estimators.

Let $\widehat { \ell } _ { t } ^ { \dagger }$ and $\widehat { \ell } _ { t } ^ { \downarrow }$ denote the two reconstructed vectors on active actions, and set

$$
\widehat { \ell } _ { t } = \widehat { \ell } _ { t } ^ { \uparrow } + \widehat { \ell } _ { t } ^ { \downarrow } .
$$

Each component takes values in $[ 0 , 1 / 2 ]$ and has total variation at most $1 / 2$ , and read in its own orientation it is nondecreasing. By Theorem 1, both reconstructed vectors are pathwise nonnegative and unbiased. Therefore $\widehat { \ell } _ { t } \geq 0$ coordinatewise and

$$
\mathbb { E } _ { t } [ \widehat { \ell } _ { t } ( j ) ] = \ell _ { t } ( x _ { t , j } ) .
$$

Moreover, by $( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$ and the second-moment bound of Theorem 1,

$$
\mathbb { E } _ { t } \sum _ { j } \pi _ { t } ( j ) \widehat { \ell } _ { t } ( j ) ^ { 2 } = O ( \Lambda _ { t } ^ { 2 } ) .
$$

Since $p _ { t } ( j ) \leq \pi _ { t } ( j ) / ( 1 - \xi ) \leq 2 \pi _ { t } ( j )$ for $\xi \le 1 / 2$ , aggregation from actions back to experts gives

$$
\mathbb { E } _ { t } \sum _ { f } u _ { t } ( f ) \widehat { \ell } _ { t } ( f ) ^ { 2 } = O ( \Lambda _ { t } ^ { 2 } ) .
$$

Apply Lemma $^ { 6 , }$ take conditional expectations, and use unbiasedness. Because both paid endpoints have marginal $\pi _ { t } .$ , their expected paid average equals $\langle \pi _ { t } , \ell _ { t } \rangle$ . Replacing $p _ { t }$ by $\pi _ { t }$ costs at most $\xi$ per round since losses lie in $[ 0 , 1 ]$ . Hence, for every comparator expert $f ^ { \star }$

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) \leq \frac { \log | \mathcal { F } | } { \eta } + O \Big ( \eta \sum _ { t = 1 } ^ { T } \Lambda _ { t } ^ { 2 } \Big ) + \xi T .
$$

The smoothing cost is $\xi T = 1$ . Since $\Lambda _ { t } ~ \leq ~ \Lambda$ , the middle term is $O ( \eta \Lambda ^ { 2 } T )$ , and the choice $\eta = \sqrt { \log | \mathcal { F } | / ( \Lambda ^ { 2 } T ) }$ makes both remaining terms equal to $O ( \Lambda { \sqrt { T \log | { \mathcal { F } } | } } )$ . This yields

$$
\mathbb { E } { \mathrm { R e g } } _ { T } ( \mathcal { F } ) = O \big ( \log ( | \mathcal { F } | T ) \sqrt { T \log | \mathcal { F } | } \big ) .
$$

## B.3 Two-point lower bound: a hidden dyadic path

This subsection proves Theorem 3. Throughout, the action space is [0, 1], so $\perp = 0$ and $\top = 1$

Threshold losses. For $z \in [ 0 , 1 ]$ let

$$
\ell _ { z } ( x ) = { \frac { 1 - x } { 2 } } + { \frac { 1 } { 2 } } \mathbf { 1 } \{ z < x \} , \qquad \ell ^ { \downarrow } ( x ) = { \frac { 1 - x } { 2 } } , \qquad \ell _ { z } ^ { \uparrow } ( x ) = { \frac { 1 } { 2 } } \mathbf { 1 } \{ z < x \} .
$$

The component $\ell ^ { \downarrow }$ is nonincreasing with values in $[ 0 , 1 / 2 ]$ and $\ell ^ { \downarrow } ( 1 ) = 0 ;$ the component $\ell _ { z } ^ { \uparrow }$ is nondecreasing with values in $\{ 0 , 1 / 2 \}$ and $\ell _ { z } ^ { \uparrow } ( 0 ) = 0$ . Hence $\ell _ { z }$ is a loss of the separated monotone problem, and its total variation on any chain is at most one, so it is also a BV loss. Because $\ell ^ { \downarrow }$ does not depend on $z ,$ observing both components at $x ,$ or only $\ell _ { z } ( x )$ , is equivalent to observing the bit $\mathbf { 1 } \{ z < x \}$ . A pair $0 \leq x \leq y \leq 1$ therefore reveals exactly which of the bins $[ 0 , x ) , [ x , y ) , [ y , 1 ]$ contains z, the middle bin being empty when $x = y$

Small depth. Let $M \ = \ \operatorname* { m i n } \{ | { \mathcal F } | , \lfloor \sqrt { T } \rfloor \}$ If $M \ < \ 2 ^ { 1 1 }$ , then min $\{ | { \mathcal { F } } | , { \sqrt { T } } \} ~ < ~ 2 ^ { 1 1 }$ , so log min $\{ | \mathcal { F } | , \sqrt { T } \} < 8$ , and it sufices to force regret $\Omega ( { \sqrt { T } } )$ . Place one expert at $0 ,$ one at 1, and the remaining experts anywhere. On each round, independently, draw a Rademacher sign $\sigma _ { t }$ and use $\ell _ { t } ( x ) = x / 2 { \mathrm { ~ i f ~ } } \sigma _ { t } = + 1$ and $\ell _ { t } ( x ) = ( 1 - x ) / 2 { \mathrm { ~ i f ~ } } \sigma _ { t } = - 1$ . These are losses of the separated monotone problem with total variation $1 / 2$ . Every action has conditional expected loss $1 / 4$ given the past, so every learner, whatever it observes, has expected loss $T / 4$ , while the better of the experts at 0 and 1 has loss $\begin{array} { r } { T / 4 - \frac { 1 } { 4 } \big | \sum _ { t } \sigma _ { t } \big | } \end{array}$ . Since $\begin{array} { r } { \mathbb { E } | \sum _ { t } \sigma _ { t } | \ge c \sqrt { T } } \end{array}$ , the regret is $\Omega ( { \sqrt { T } } )$

Parameters. Assume from now on that $M \geq 2 ^ { 1 1 }$ , and set

$$
k = \lfloor \log _ { 2 } M \rfloor \ge 1 1 , \qquad K = 2 ^ { k } \le M , \qquad \delta = \frac { c _ { 0 } } { \sqrt { T } } , \qquad c _ { 0 } = \frac { 1 } { 1 2 8 \sqrt { 2 } } .
$$

Since $\lfloor { \sqrt { T } } \rfloor \geq { \sqrt { T } } / 2$ , we have $k \geq \log _ { 2 }$ min $\begin{array} { r } { \{ | \mathcal { F } | , \sqrt { T } \} - 2 \ge \frac { 1 1 } { 1 3 } } \end{array}$ log<sub>2</sub> min $\{ | \mathcal { F } | , \sqrt { T } \}$ . Place one expert at each point of

$$
\mathcal { C } = \Big \{ \frac { i + 1 / 2 } { K } : ~ i = 0 , 1 , \ldots , K - 1 \Big \} ,
$$

the centers of the dyadic cells of depth $k ,$ and place any remaining experts at other points of $[ 0 , 1 ]$ The argument below allows the learner to query arbitrary points of [0, 1].

Haar functions and tents. For a dyadic interval $\boldsymbol { I } = [ a , a + w ]$ with left half $I _ { \mathrm { L } }$ and right half $I _ { \mathrm { R } }$ , let

$$
\psi _ { I } = \frac { \mathbf { 1 } _ { I _ { \mathrm { L } } } - \mathbf { 1 } _ { I _ { \mathrm { R } } } } { w } , \qquad \tau _ { I } ( x ) = \int _ { 0 } ^ { x } \psi _ { I } ( z ) d z = \mathbf { 1 } \{ x \in I \} \frac { \operatorname* { m i n } \{ x - a , a + w - x \} } { w } .
$$

Thus $\tau _ { I }$ is a tent of height $1 / 2$ supported on $I ,$ it is $1 / w \mathrm { - L i p s c h i t z }$ , and it vanishes at 0 and 1. For $\theta \in { \mathcal { C } }$ and $0 \leq d < k$ , let ${ \cal I } _ { d } ( \theta )$ be the dyadic interval of width $2 ^ { - d }$ containing $\theta ,$ and set

$$
G _ { \theta } = \sum _ { d = 0 } ^ { k - 1 } \psi _ { I _ { d } ( \theta ) } , \qquad H _ { \theta } = \sum _ { d = 0 } ^ { k - 1 } \tau _ { I _ { d } ( \theta ) } ,
$$

so that $\begin{array} { r } { H _ { \theta } ( x ) = \int _ { 0 } ^ { x } G _ { \theta } ( z ) d z } \end{array}$

The random instance. Draw Θ uniformly from C. Given $\Theta = \theta .$ , draw $Z _ { 1 } , \dots , Z _ { T }$ independently from the density $\rho _ { \theta } = 1 - \delta G _ { \theta }$ , and use the losses $\ell _ { t } = \ell _ { Z _ { t } }$ . Each Haar function integrates to zero, and $\begin{array} { r } { | G _ { \theta } | \le \sum _ { d < k } 2 ^ { d } < K } \end{array}$ with $\delta K \le c _ { 0 } < 1 / 4$ , so $\rho _ { \theta }$ is a density with values in $[ 3 / 4 , 5 / 4 ]$ . Its distribution function is $x - \delta H _ { \theta } ( x )$ , so the mean loss is

$$
\mathbb { E } _ { \theta } \ell _ { Z _ { t } } ( x ) = \frac { 1 - x } { 2 } + \frac { 1 } { 2 } \big ( x - \delta H _ { \theta } ( x ) \big ) = \frac { 1 } { 2 } - \frac { \delta } { 2 } H _ { \theta } ( x ) .
$$

The null environment draws $Z _ { 1 } , \dots , Z _ { T }$ independently and uniformly from $[ 0 , 1 ]$

Lemma 10 (Tents along a random path). Let Θ be uniform on C. Then

1. $0 \leq H _ { \theta } ( x ) \leq k / 2$ for every θ and x;

2. $\mathbb { E } H _ { \Theta } ( x ) < 1$ for every fixed $x ;$

3. E ${ \mathrm { : } } H _ { \Theta } ( \Theta ) = k / 4 .$

4. ${  { \mathbb E } } \big ( H _ { \Theta } ( y ) - H _ { \Theta } ( x ) \big ) ^ { 2 } \le 6 4 ( y - x ) f o r 0 \le x < y \le 1$ , and in particular $\begin{array} { r } { \mathfrak { T } H _ { \Theta } ( x ) ^ { 2 } \le 6 4 \operatorname* { m i n } \{ x , 1 } \end{array}$ x}.

Proof. The first claim holds because $H _ { \theta }$ is a sum of k tents with values in $[ 0 , 1 / 2 ]$

For the second claim, fix x and a depth $d ,$ and let $w = 2 ^ { - d }$ . Only the depth-d dyadic interval containing x can contribute a nonzero tent at $x ,$ and it is the ancestor $I _ { d } ( \Theta )$ with probability w, because it contains a fraction w of the leaf centers. Its contribution is at most $1 / 2 .$ so the expected contribution of depth d is at most $w / 2$ , and summing over the k depths gives $\mathbb { E } H _ { \Theta } ( x ) \le$ $\begin{array} { r } { \sum _ { d = 0 } ^ { \operatorname { \dot { k } } - 1 } 2 ^ { - d - 1 } = 1 - 2 ^ { - k } < 1 } \end{array}$

For the third claim, condition on the depth-d ancestor $\boldsymbol { I } = [ a , a + w ]$ of Θ. Then Θ is uniform over the $2 ^ { k - d } \geq 2$ leaf centers in I, half of them in each half of I. The leaf centers in the left half are symmetric about the quarter point $a + w / 4$ , and on the left half $\tau _ { I } ( u ) = ( u - a ) / w$ , so a center and its reflection about $a + w / 4$ have tent heights summing to $1 / 2 ;$ ; a center at the quarter point itself has height $1 / 4$ . The same holds in the right half. Hence the conditional mean of $\tau _ { I } ( \Theta )$ is $1 / 4$ and summing over the k depths gives $k / 4$

For the last claim, let $r = y - x$ and $X _ { d } = \tau _ { I _ { d } ( \Theta ) } ( y ) - \tau _ { I _ { d } ( \Theta ) } ( x )$ , and let $w = 2 ^ { - d }$ . The diference $X _ { d }$ is nonzero only if $I _ { d } ( \Theta )$ contains x or $y .$ , which has probability at most $2 w .$ , and $| X _ { d } | \le \operatorname* { m i n } \{ r / w , 1 / 2 \}$ because the tent is 1/w-Lipschitz with values in $[ 0 , 1 / 2 ]$ . Hence $\| X _ { d } \| _ { 2 } \le \sqrt { 2 w } \operatorname* { m i n } \{ r / w , 1 / 2 \}$ , where $\| \cdot \| _ { 2 }$ is the $L ^ { 2 }$ norm over Θ. By Minkowski’s inequality, splitting the dyadic widths at $r ,$

$$
\big \| \boldsymbol { H } _ { \boldsymbol { \Theta } } ( y ) - \boldsymbol { H } _ { \boldsymbol { \Theta } } ( x ) \big \| _ { 2 } \leq \sqrt { 2 } r \sum _ { w \geq r } w ^ { - 1 / 2 } + \sum _ { w < r } \sqrt { w / 2 } \leq 5 \sqrt { r } + 3 \sqrt { r } ,
$$

since both are geometric sums with ratio $\sqrt { 2 }$ whose largest terms are at most $\sqrt { 2 } r \cdot r ^ { - 1 / 2 }$ and $\sqrt { r / 2 }$ respectively. Squaring gives the bound 64r. Since every tent vanishes at 0 and 1, applying it to the intervals [0, x] and [x, 1] gives the final inequality. □

Lemma 11 (Three-bin information). For a pair $0 \leq x \leq y \leq 1$ , let $Q _ { 0 }$ and $Q _ { \theta }$ be the laws of the bin $o f Z _ { t }$ among $[ 0 , x ) , [ x , y ) , [ y , 1 ]$ under the null environment and under the hidden leaf θ. Then

$$
\frac { 1 } { K } \sum _ { \theta \in \mathcal { C } } D _ { \mathrm { K L } } ( Q _ { 0 } \Vert Q _ { \theta } ) \leq 2 5 6 \delta ^ { 2 } .
$$

Proof. The bin probabilities are $( x , y - x , 1 - y )$ under the null and

$$
\big ( x - \delta H _ { \theta } ( x ) , y - x + \delta H _ { \theta } ( x ) - \delta H _ { \theta } ( y ) , 1 - y + \delta H _ { \theta } ( y ) \big )
$$

under θ. Since $\rho _ { \theta } \ge 3 / 4$ , every bin probability under θ is at least $3 / 4$ of its null probability. Bounding the relative entropy by the $\chi ^ { 2 }$ divergence,

$$
D _ { \mathrm { K L } } ( Q _ { 0 } \| Q _ { \theta } ) \leq \frac { 4 \delta ^ { 2 } } { 3 } \left( \frac { H _ { \theta } ( x ) ^ { 2 } } { x } + \frac { \left( H _ { \theta } ( y ) - H _ { \theta } ( x ) \right) ^ { 2 } } { y - x } + \frac { H _ { \theta } ( y ) ^ { 2 } } { 1 - y } \right) ,
$$

where a term with a vanishing denominator is omitted, since its numerator vanishes as well. By Lemma 10, the average over θ of each of the three quotients is at most 64, which gives ${ \begin{array} { l } { { \frac { 4 } { 3 } } \cdot 3 \cdot 6 4 \delta ^ { 2 } = 2 5 6 \delta ^ { 2 } } \end{array} }$ □

Proof of Theorem 3. Fix a two-point learner. For the analysis only, after the learner chooses its pair $( A _ { t } , B _ { t } )$ , let $Y _ { t }$ equal $A _ { t }$ or $B _ { t }$ with probability $1 / 2$ each, independently of everything else. Then the conditional expectation of $\ell _ { t } ( Y _ { t } )$ given the pair and $Z _ { t }$ is the paid average $\begin{array} { r } { \frac { 1 } { 2 } ( \ell _ { t } ( A _ { t } ) + \ell _ { t } ( B _ { t } ) ) } \end{array}$ . Let $P _ { 0 }$ and $P _ { \theta }$ be the laws of the transcript, consisting of the learner’s internal randomness, its pairs, the observed bins, and $Y _ { 1 } , \dots , Y _ { T }$ , under the null environment and under the hidden leaf θ.

Information. On every round the learner chooses its pair from the past transcript in the same way under every environment, and the auxiliary choice of $Y _ { t }$ does not depend on the environment. By the chain rule for relative entropy and Lemma 11,

$$
{ \frac { 1 } { K } } \sum _ { \theta } D _ { \mathrm { K L } } ( P _ { 0 } \| P _ { \theta } ) \leq 2 5 6 T \delta ^ { 2 } = 2 5 6 c _ { 0 } ^ { 2 } = { \frac { 1 } { 1 2 8 } } .
$$

By Pinsker’s inequality and Jensen’s inequality,

$$
\frac { 1 } { K } \sum _ { \theta } d _ { \mathrm { T V } } ( P _ { 0 } , P _ { \theta } ) \leq \sqrt { \frac { 1 } { 2 } \cdot \frac { 1 } { 1 2 8 } } = \frac { 1 } { 1 6 } .
$$

Tent height collected by the learner. For each θ let $\begin{array} { r } { S _ { \theta } = \sum _ { t = 1 } ^ { T } H _ { \theta } ( Y _ { t } ) } \end{array}$ , a function of the transcript with $0 \leq S _ { \theta } \leq T k / 2$ by Lemma 10. Under $P _ { 0 }$ the transcript does not depend on $\theta ,$ so averaging over θ and using the second claim of Lemma 10 gives $\begin{array} { r } { \frac { 1 } { K } \sum _ { \theta } \mathbb { E } _ { 0 } S _ { \theta } \leq T } \end{array}$ . Changing measure,

$$
{ \frac { 1 } { K } } \sum _ { \theta } \mathbb { E } _ { \theta } S _ { \theta } \leq { \frac { 1 } { K } } \sum _ { \theta } { \Big ( } \mathbb { E } _ { 0 } S _ { \theta } + { \frac { T k } { 2 } } d _ { \mathrm { T V } } ( P _ { 0 } , P _ { \theta } ) { \Big ) } \leq T + { \frac { T k } { 3 2 } } .
$$

Regret. The best expert in hindsight has loss at most that of the expert at θ. Since $Y _ { t }$ is chosen before $Z _ { t }$ is drawn, the mean-loss formula gives

$$
\mathbb { E } _ { \theta } \mathrm { R e g } _ { T } \geq \mathbb { E } _ { \theta } \sum _ { t = 1 } ^ { T } \bigl ( \ell _ { t } ( Y _ { t } ) - \ell _ { t } ( \theta ) \bigr ) = \frac { \delta } { 2 } \bigl ( T H _ { \theta } ( \theta ) - \mathbb { E } _ { \theta } S _ { \theta } \bigr ) .
$$

Averaging over $\theta$ and using the third claim of Lemma 10,

$$
\frac { 1 } { K } \sum _ { \theta } \mathbb { E } _ { \theta } \mathrm { R e g } _ { T } \geq \frac { \delta T } { 2 } \Big ( \frac { k } { 4 } - 1 - \frac { k } { 3 2 } \Big ) \geq \frac { \delta T k } { 1 6 } ,
$$

where the last step uses $k \geq 1 1$ . Since $\delta = c _ { 0 } / \sqrt { T }$ and $\begin{array} { r } { k \geq \frac { 1 1 } { 1 3 } \log _ { 2 } } \end{array}$ min $\{ | \mathcal { F } | , \sqrt { T } \} \ge$ log min $\{ | \mathcal { F } | , \sqrt { T } \}$ the average regret is $\Omega ( { \sqrt { T } }$ log min $\{ | { \mathcal { F } } | , { \sqrt { T } } \} )$ . All randomness of the instance, including Θ, is drawn before play, so some realization is an oblivious instance with at least this regret. □

## C Proofs for the BV problem

This appendix contains the proofs for Section 5.

## C.1 Robust mirror descent for signed estimates

Proof of Proposition 1. The barrier term makes Ψ a Legendre function on $\Delta ^ { \circ } ( { \mathcal { F } } )$ , so the minimizer $w _ { t + 1 }$ exists, has full support, and satisfies the first-order condition

$$
\nabla \Psi ( w _ { t + 1 } ) = \nabla \Psi ( w _ { t } ) - \eta \widehat { \ell } _ { t } - \lambda _ { t } { \bf 1 }
$$

for some scalar $\lambda _ { t }$ . Since $\langle u - w _ { t + 1 } , \mathbf { 1 } \rangle = 0$ , the three-point identity for Bregman divergences gives

$$
\eta \langle w _ { t + 1 } - u , \widehat { \ell } _ { t } \rangle = D _ { \Psi } ( u , w _ { t } ) - D _ { \Psi } ( u , w _ { t + 1 } ) - D _ { \Psi } ( w _ { t + 1 } , w _ { t } ) .
$$

Adding $\eta \langle w _ { t } - w _ { t + 1 } , \widehat { \ell } _ { t } \rangle$ to both sides,

$$
\begin{array} { r } { \eta \langle w _ { t } - u , \widehat { \ell } _ { t } \rangle \leq D _ { \Psi } ( u , w _ { t } ) - D _ { \Psi } ( u , w _ { t + 1 } ) + S _ { t } , } \end{array}
$$

where

$$
S _ { t } = \operatorname* { s u p } _ { v \in \mathbb { R } _ { > 0 } ^ { \mathcal { F } } } \{ \eta \langle w _ { t } - v , \widehat { \ell } _ { t } \rangle - D _ { \Psi } ( v , w _ { t } ) \}
$$

and we have relaxed the simplex constraint. Because $\Psi$ is separable, writing $y _ { f } = v ( f ) / w _ { t } ( f )$ gives

$$
D _ { \Psi } ( v , w _ { t } ) = \sum _ { f } w _ { t } ( f ) \varphi ( y _ { f } ) + \mu \sum _ { f } ( y _ { f } - 1 - \log y _ { f } ) , \qquad \varphi ( y ) = y \log y - y + 1 ,
$$

so $\begin{array} { r } { S _ { t } \le \sum _ { f } \operatorname* { s u p } _ { y > 0 } F _ { f } ( y ) } \end{array}$ with

$$
F _ { f } ( y ) = w _ { t } ( f ) { \big ( } c _ { f } ( y - 1 ) - \varphi ( y ) { \big ) } - \mu ( y - 1 - \log y ) , \qquad c _ { f } = - \eta { \widehat { \ell } } _ { t } ( f ) .
$$

We show su $) _ { y > 0 } F _ { f } ( y ) \leq w _ { t } ( f ) c _ { f } ^ { 2 }$

Case $0 < y \le 2$ . Since $\varphi ( 1 ) ^ { \prime } = \varphi ^ { \prime } ( 1 ) = 0$ and $\varphi ^ { \prime \prime } ( s ) = 1 / s \geq 1 / 2$ for $s \leq 2$ , Taylor’s theorem gives $\varphi ( y ) \geq ( y - 1 ) ^ { 2 } / 4$ . Dropping the nonnegative barrier term,

$$
F _ { f } ( y ) \leq w _ { t } ( f ) \Big ( c _ { f } ( y - 1 ) - \frac { ( y - 1 ) ^ { 2 } } { 4 } \Big ) \leq w _ { t } ( f ) c _ { f } ^ { 2 } .
$$

Case $y > 2$ . Here

$$
F _ { f } ^ { \prime } ( y ) = w _ { t } ( f ) c _ { f } - w _ { t } ( f ) \log y - \mu \Big ( 1 - \frac { 1 } { y } \Big ) < w _ { t } ( f ) | c _ { f } | - \frac { \mu } { 2 } \le 0 ,
$$

because log $y > 0 , 1 - 1 / y \geq 1 / 2$ , and $\mu \geq 2 \eta \beta \geq 2 w _ { t } ( f ) | c _ { f } |$ . Hence $F _ { f }$ is decreasing on $[ 2 , \infty )$ and $\begin{array} { r } { \operatorname* { s u p } _ { y > 2 } F _ { f } ( y ) \le F _ { f } ( 2 ) \le w _ { t } ( f ) c _ { f } ^ { 2 } } \end{array}$ by the first case.

Therefore

$$
S _ { t } \leq \eta ^ { 2 } \sum _ { f } w _ { t } ( f ) \widehat { \ell } _ { t } ( f ) ^ { 2 } .
$$

Summing the three-point inequality over $t ,$ discarding $- D _ { \Psi } ( u , w _ { T + 1 } ) \leq 0$ , and dividing by $\eta$ proves the proposition. □

## C.2 BV two-point upper bound

Proof of Theorem $\it 4 .$ Run endpoint flow over the loss vector $\ell _ { t } .$ , anchored below the chain. Let $w _ { t }$ be the expert mixture produced by the robust mirror-descent update, aggregate it on active actions as

$$
p _ { t } ( j ) = \sum _ { f : j _ { t } ( f ) = j } w _ { t } ( f ) ,
$$

and smooth

$$
\pi _ { t } = ( 1 - \xi ) p _ { t } + \xi \operatorname { U n i f } ( X _ { t } ) .
$$

Here $\xi = 1 / T$ . Let $\Lambda _ { t } = 1 + \log ( m _ { t } / \xi )$ ; since $m _ { t } \leq | \mathcal { F } |$ , we have $\Lambda _ { t } \leq \Lambda$ . Write $\widehat { \ell } _ { t } ( f ) = \widehat { \ell } _ { t } ( j _ { t } ( f ) )$ for the estimate at the recommendation of $f .$ Since $\ell _ { t }$ takes values in [0, 1] and $\mathrm { T V } ( \ell _ { t } ) \leq 1$ , Theorem 1 gives

$$
\mathbb { E } _ { t } \sum _ { j } \pi _ { t } ( j ) \widehat { \ell } _ { t } ( j ) ^ { 2 } = O ( \Lambda ^ { 2 } ) .
$$

Since $p _ { t } ( j ) \leq \pi _ { t } ( j ) / ( 1 - \xi ) \leq 2 \pi _ { t } ( j )$

$$
\mathbb { E } _ { t } \sum _ { f } w _ { t } ( f ) \widehat { \ell } _ { t } ( f ) ^ { 2 } = O ( \Lambda ^ { 2 } ) .
$$

Likewise the self-bound and $w _ { t } ( f ) \leq 2 \pi _ { t } ( j _ { t } ( f ) )$ give

$$
\operatorname* { m a x } _ { f } w _ { t } ( f ) | \widehat { \ell } _ { t } ( f ) | \leq 6 \Lambda .
$$

Thus Proposition 1 applies with $\beta = 6 \Lambda$ and $\mu = 1 2 \eta \Lambda$

Use the interior comparator

$$
\boldsymbol { u } = ( 1 - \varepsilon ) \boldsymbol { e } _ { f ^ { \star } } + \varepsilon \mathbf { 1 } / | \mathcal { F } | .
$$

Comparator smoothing costs at most $\varepsilon T$ and action smoothing costs at most $\xi T .$ . For the regularizer diameter, the entropy part of $D _ { \Psi } { \left( u , w _ { 1 } \right) }$ is $D _ { \mathrm { K L } } ( u \| w _ { 1 } ) \leq \log | \mathcal { F } |$ , and since $\textstyle \sum _ { f } u ( f ) / w _ { 1 } ( f ) = | { \mathcal { F } } |$ the barrier part equals

$$
\mu \sum _ { f } \left( \frac { u ( f ) } { w _ { 1 } ( f ) } - 1 - \log \frac { u ( f ) } { w _ { 1 } ( f ) } \right) = - \mu \sum _ { f } \log \left( | \mathcal { F } | u ( f ) \right) \leq \mu | \mathcal { F } | \log ( 1 / \varepsilon ) ,
$$

because $| { \mathcal { F } } | u ( f ) = \varepsilon$ for $f \neq f ^ { \star }$ and $| \mathcal { F } | u ( f ^ { \star } ) \geq 1$ . Hence

$$
D _ { \Psi } ( u , w _ { 1 } ) \leq \log | \mathcal { F } | + 1 2 \eta \Lambda | \mathcal { F } | \log ( 1 / \varepsilon ) .
$$

Taking expectations and using unbiasedness gives

$$
\mathbb { E } \mathrm { R e g } _ { T } ( \mathcal { F } ) \le \varepsilon T + \xi T + 1 2 \Lambda | \mathcal { F } | \log ( 1 / \varepsilon ) + \frac { \log | \mathcal { F } | } { \eta } + O ( \eta \Lambda ^ { 2 } T ) .
$$

Setting $\varepsilon = \xi = 1 / T$ and $\eta = \sqrt { \log | \mathcal { F } | / ( \Lambda ^ { 2 } T ) }$ gives the claimed bound.

## C.3 Needle lower bound

Proof of Theorem 5. Let $M = \operatorname* { m i n } \{ | \mathcal { F } | , T \}$ . Place one expert on each action $1 \prec \cdots \prec M$ and duplicate arbitrary experts if $| { \mathcal { F } } | > M$ . Draw $i ^ { \star }$ uniformly from $[ M ]$ and set, for every round,

$$
\ell _ { t } ( i ) = \frac { 1 } { 2 } { \mathbf 1 } \{ i \neq i ^ { \star } \} .
$$

This is realizable in the separated monotone problem: take

$$
\ell _ { t } ^ { \uparrow } ( i ) = \frac { 1 } { 2 } \mathbf { 1 } \{ i > i ^ { \star } \} , \qquad \ell _ { t } ^ { \downarrow } ( i ) = \frac { 1 } { 2 } \mathbf { 1 } \{ i < i ^ { \star } \} .
$$

Both components satisfy the endpoint normalizations and take values in $[ 0 , 1 / 2 ]$

Let τ be the first round on which $i ^ { \star }$ is among the two queried actions, with $\tau = T + 1$ if this never occurs. Fix the learner’s internal randomness and run it counterfactually on the all-1/2 feedback history. The first s query pairs touch a deterministic set $S _ { s }$ of at most 2s candidate actions. The real interaction agrees with this counterfactual trajectory until $i ^ { \star }$ is queried, so

$$
\mathbb { P } ( \tau \leq s ) = \mathbb { P } ( i ^ { \star } \in S _ { s } ) \leq \frac { 2 s } { M } .
$$

Therefore

$$
\mathbb { E } \tau \geq \sum _ { s = 0 } ^ { \lceil M / 2 \rceil - 1 } \left( 1 - \frac { 2 s } { M } \right) \geq \frac { M } { 4 } .
$$

Before discovery the learner cannot play the zero-loss action. Hence its expected cumulative loss, equal to its regret because the best expert incurs zero, is at least

$$
\frac 1 2 ( \mathbb { E } \tau - 1 ) \geq \frac { M } { 8 } - \frac 1 2 .
$$

Moreover, on the first round each played action has expected loss $( 1 - 1 / M ) / 2 \geq 1 / 4$ , so the regret is also at least $1 / 4$ . Since max $\{ 1 / 4 , M / 8 - 1 / 2 \} \ge M / 2 4$ for every $M \geq 2$ , the regret is at least $M / 2 4$ . Averaging over $i ^ { \star }$ leaves at least one fixed needle location with the same lower bound for the randomized learner. □

## C.4 The two-point BV rate

Proof of Corollary 1. Let $M = \operatorname* { m i n } \{ | \mathcal { F } | , T \}$ . Theorem 3 gives a BV instance with $\mathbb { E } { \mathrm { R e g } _ { T } } \geq \rho : =$ $c _ { 1 } \sqrt { T }$ log min $\{ | { \mathcal { F } } | , { \sqrt { T } } \}$ for a universal constant $c _ { 1 } > 0$ , and the proof of Theorem 5 gives one with $\mathbb { E } \mathrm { { R e g } } _ { T } \geq M / 2 4$ . The worse of the two instances forces at least max $\{ \rho , M / 2 4 \} \ge \rho / 2 + M / 4 8$ , which proves the corollary. □

## D Proofs for one-point and full-information feedback

This appendix contains the proofs for Section 7. We begin with the full-information benchmark.

## D.1 Full-information benchmark

Proof of Theorem 9. The upper bound is the standard exponential-weights guarantee $O ( \sqrt { T \log | \mathcal { F } | } )$ for losses in [0, 1]. For the lower bound, by Yao’s principle it sufices to construct a random oblivious instance on which every learner has large expected regret; averaging then yields a fixed instance for every randomized learner. Let

$$
k = \lfloor \log _ { 2 } | \mathcal { F } | \rfloor ,
$$

so that $1 \leq k \leq T$ by the assumption $\log _ { 2 } | \mathcal { F } | \leq T$ . Use the actions $1 < \cdots < 2 ^ { k }$ , so that $\perp = 1$ and ${ \mathsf { T } } = 2 ^ { k }$ , place one expert permanently on each action, and place any remaining experts on action 1; extra experts can only lower the comparator’s loss. Partition the first nk rounds into $k$ consecutive temporal blocks $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { k }$ of equal length

$$
n = \left\lfloor { \frac { T } { k } } \right\rfloor \geq 1 .
$$

Unused rounds, if any, have zero loss. The adversary samples all randomness below before the interaction.

Initially let $I _ { 0 } = [ 2 ^ { k } ]$ . In block $d ,$ the interval $I _ { d - }$ <sub>−</sub> has two equal consecutive children $I _ { d - 1 } ^ { \mathrm { L } }$ and $I _ { d - 1 } ^ { \mathrm { R } }$ separated by a threshold $q _ { d }$ , with $1 \leq q _ { d } < 2 ^ { k }$ . On every round $t \in \mathcal { T } _ { d }$ , draw an independent Rademacher sign $\sigma _ { t }$ . If $\sigma _ { t } = + 1$ , use the increasing step

$$
\ell _ { t } ( i ) = \frac { 1 } { 2 } { \bf 1 } \{ i > q _ { d } \} ;
$$

if $\sigma _ { t } = - 1$ , use the decreasing step

$$
\ell _ { t } ( i ) = \frac { 1 } { 2 } { \bf 1 } \{ i \leq q _ { d } \} .
$$

The increasing step is a component $\ell _ { t } ^ { \uparrow }$ with $\ell _ { t } ^ { \uparrow } ( \bot ) = 0$ , the decreasing step is a component $\ell _ { t } ^ { \downarrow }$ with $\ell _ { t } ^ { \downarrow } ( \top ) = 0$ , and each step has total variation $1 / 2$ . Hence the instance belongs to both problems.

Let $\mathcal { G } _ { t - 1 }$ be generated by the learner’s internal randomness and $\sigma _ { 1 } , \ldots , \sigma _ { t - 1 }$ . Everything the learner has observed before round $t ,$ under any feedback model, is $\mathcal { G } _ { t - 1 }$ -measurable, so its queries on round t are $\mathcal { G } _ { t - 1 }$ -measurable, while $\sigma _ { t }$ is independent of $\mathcal { G } _ { t - 1 }$ . Conditional on $\mathcal { G } _ { t - 1 }$ , every action has loss $1 / 2$ with probability $1 / 2$ and 0 otherwise. Consequently the conditional expected paid average of a two-point learner is $1 / 4$ on every round, and so is the conditional expected loss of a one-point or full-information learner.

Let

$$
S _ { d } = \sum _ { t \in \mathcal { T } _ { d } } \sigma _ { t } .
$$

After block $d ,$ define $I _ { d }$ to be whichever child of $I _ { d - 1 }$ has smaller cumulative loss in that block, breaking ties arbitrarily. Every expert in $I _ { d }$ has block-d loss

$$
\frac { n } { 4 } - \frac { | S _ { d } | } { 4 } .
$$

The intervals are nested,

$$
I _ { 0 } \supset I _ { 1 } \supset \dots \supset I _ { k } ,
$$

and $I _ { k }$ contains a single expert $i ^ { \star }$ . Since this same expert belongs to the winning child at every level,

$$
\sum _ { t = 1 } ^ { n k } \ell _ { t } ( i ^ { \star } ) = \frac { n k } { 4 } - \frac { 1 } { 4 } \sum _ { d = 1 } ^ { k } | S _ { d } | .
$$

The learner, meanwhile, has expected loss $n k / 4$ . Therefore

$$
\mathbb { E } { \mathrm { R e g } _ { T } } \geq \frac { 1 } { 4 } \sum _ { d = 1 } ^ { k } \mathbb { E } | S _ { d } | .
$$

A length-n Rademacher sum satisfies $\mathbb { E } | S _ { d } | \geq c \sqrt { n }$ for a universal constant $c > 0$ . Hence

$$
\mathbb { E } \mathrm { { R e g } } _ { T } \geq c ^ { \prime } k \sqrt { n } \geq c ^ { \prime \prime } \sqrt { T k } ,
$$

using $n = \lfloor T / k \rfloor \geq T / ( 2 k )$ . Since $k \geq \textstyle { \frac { 1 } { 2 } } \log _ { 2 } { | \mathcal { F } | }$ , this is $\Omega ( \sqrt { T \log | \mathcal { F } | } )$

The two one-point constructions share a scalar common-noise perturbation and a hiddencoordinate testing lemma, which we record first. Both proofs use Yao’s principle: it sufices to lower bound the average regret of deterministic learners under a random oblivious instance, and averaging then yields a fixed oblivious instance for every randomized learner.

## D.2 Smooth common-noise perturbation

Lemma 12 (Smooth perturbation). For every $\varepsilon \in ( 0 , 1 / 4 ]$ there exist coupled random variables $Z , Z _ { \varepsilon } \in [ 1 / 4 , 1 / 2 ]$ with laws $P , Q _ { \varepsilon }$ such that $0 \leq Z - Z _ { \varepsilon } \leq \varepsilon / 1 6$ almost surely,

$$
\begin{array} { r } { \mathbb { E } Z - \mathbb { E } Z _ { \varepsilon } = \varepsilon / 2 4 , \qquad D _ { \mathrm { K L } } ( P \| Q _ { \varepsilon } ) \leq 3 \varepsilon ^ { 2 } . } \end{array}
$$

Proof. Let $U \sim \mathrm { U n i f } [ 0 , 1 ]$ and $\tau _ { \varepsilon } ( u ) = u - \varepsilon u ( 1 - u )$ . Since $\tau _ { \varepsilon } ^ { \prime } ( u ) = 1 - \varepsilon + 2 \varepsilon u \in [ 1 - \varepsilon , 1 + \varepsilon ] , \tau _ { \varepsilon }$ is an increasing bijection. Define

$$
Z = \frac { 1 } { 4 } + \frac { U } { 4 } , \qquad Z _ { \varepsilon } = \frac { 1 } { 4 } + \frac { \tau _ { \varepsilon } ( U ) } { 4 } .
$$

The pointwise bound follows from $u ( 1 - u ) \leq 1 / 4$ , and

$$
\mathbb { E } ( Z - Z _ { \varepsilon } ) = { \frac { \varepsilon } { 4 } } \mathbb { E } [ U ( 1 - U ) ] = { \frac { \varepsilon } { 2 4 } } .
$$

After removing the common afine transformation, compare the uniform density on $[ 0 , 1 ]$ with the density $q _ { \varepsilon } ( y ) = 1 / \tau _ { \varepsilon } ^ { \prime } ( \tau _ { \varepsilon } ^ { - 1 } ( y ) )$ . For $\varepsilon \le 1 / 4$

$$
\frac { 1 } { 1 + \varepsilon } \leq q _ { \varepsilon } ( y ) \leq \frac { 1 } { 1 - \varepsilon } , \qquad | q _ { \varepsilon } ( y ) - 1 | \leq \frac { \varepsilon } { 1 - \varepsilon } .
$$

Therefore $\chi ^ { 2 } ( P \| Q _ { \varepsilon } ) \leq 3 \varepsilon ^ { 2 }$ , and $D _ { \mathrm { K L } } ( P \| Q _ { \varepsilon } ) \leq \chi ^ { 2 } ( P \| Q _ { \varepsilon } )$

## D.3 One hidden favorable coordinate

Lemma 13 (Testing lemma). Fix $K \ge 2 , T \ge K$ , and $\varepsilon \in ( 0 , 1 / 4 ]$ , and use the laws $P , Q _ { \varepsilon }$ from Lemma 12. On each round, independently of the past, every action under the null has observation law P and the same mean loss. Under alternative $\theta \in [ K ]$ , only action θ changes to observation law $Q _ { \varepsilon }$ and has mean loss lower by $\Delta = \varepsilon / 2 4$ . Then every one-point learner satisfies

$$
\frac { 1 } { K } \sum _ { \theta = 1 } ^ { K } \mathbb { E } _ { \theta } \mathrm { R e g } _ { T } \geq \Delta T \left( 1 - \frac { 1 } { K } - \varepsilon \sqrt { \frac { 3 T } { 2 K } } \right) .
$$

In particular, for suficiently small universal $a > 0 , \varepsilon = a \sqrt { K / T }$ gives $\Omega ( \sqrt { K T } )$ average regret.

Proof. Let

$$
N _ { \theta } = \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ A _ { t } = \theta \} , \qquad n _ { \theta } = \mathbb { E } _ { 0 } N _ { \theta } .
$$

The adaptive chain rule gives

$$
D _ { \mathrm { K L } } ( P _ { 0 } \Vert P _ { \theta } ) = n _ { \theta } D _ { \mathrm { K L } } ( P \Vert Q _ { \varepsilon } ) \leq 3 \varepsilon ^ { 2 } n _ { \theta } ,
$$

because the observation law difers only when the learner selects θ. Pinsker and $0 \leq N _ { \theta } \leq T$ imply

$$
\mathbb { E } _ { \theta } N _ { \theta } \leq n _ { \theta } + T \varepsilon \sqrt { \frac { 3 n _ { \theta } } { 2 } } .
$$

Since $\textstyle \sum _ { \theta } n _ { \theta } = T$ , Jensen yields

$$
\frac { 1 } { K } \sum _ { \theta } \mathbb { E } _ { \theta } N _ { \theta } \leq \frac { T } { K } + \varepsilon T \sqrt { \frac { 3 T } { 2 K } } .
$$

Since the best action in hindsight has cumulative loss at most that of action $\theta ,$ independence of the current losses from the learner’s past gives

$$
\mathbb { E } _ { \theta } \mathrm { R e g } _ { T } \geq \Delta \big ( T - \mathbb { E } _ { \theta } { \cal N } _ { \theta } \big ) .
$$

Averaging proves the lemma.

## D.4 Separated one-point lower bound

Proof of Theorem 7. By Yao’s principle it sufices to lower bound deterministic learners under a random oblivious instance. Let

$$
K = \operatorname* { m i n } \{ | \mathcal { F } | , { } [ T ^ { 1 / 3 } ] \} , \qquad \varepsilon = a \sqrt { K / T } , \qquad s = \varepsilon / 1 6 ,
$$

where $a$ is the constant from Lemma 13. Use actions $\perp \prec 1 \prec \cdots \prec K$ and one constant expert at each positive action; place any remaining experts on action 1.

Reveal before play the fixed nonincreasing component

$$
\ell ^ { \downarrow } ( \bot ) = s K , \qquad \ell ^ { \downarrow } ( i ) = s ( K - i ) \quad ( i \in [ K ] ) .
$$

Since $K \le T ^ { 1 / 3 }$

$$
s K = \frac { a K ^ { 3 / 2 } } { 1 6 \sqrt { T } } \leq \frac { a } { 1 6 } ,
$$

so $\ell ^ { \downarrow }$ takes values in $[ 0 , 1 / 2 ]$ for small $^ { a , }$ and $\ell ^ { \downarrow } ( K ) = 0$

Under the null, draw $Z _ { t }$ independently and set

$$
\ell _ { t } ^ { \uparrow , 0 } ( \perp ) = 0 , \qquad \ell _ { t } ^ { \uparrow , 0 } ( i ) = Z _ { t } - s ( K - i ) .
$$

Under alternative θ, use the coupled pair $( Z _ { t } , Z _ { t , \varepsilon } )$ and set

$$
\ell _ { t } ^ { \uparrow , \theta } ( \perp ) = 0 , \qquad \ell _ { t } ^ { \uparrow , \theta } ( i ) = \left\{ Z _ { t , \varepsilon } - s ( K - \theta ) , \quad i = \theta , \right.
$$

The pointwise bound $0 \leq Z _ { t } - Z _ { t , \varepsilon } \leq \varepsilon / 1 6$ and $s = \varepsilon / 1 6$ ensure that the increments into and out of θ remain nonnegative; all other positive-action increments equal s. Thus $\ell _ { t } ^ { \uparrow , \theta }$ is nonnegative and nondecreasing, with values in $[ 0 , 1 / 2 ]$ after choosing a small enough.

For $i \neq \theta$ the total loss is

$$
\ell ^ { \downarrow } ( i ) + \ell _ { t } ^ { \uparrow , \theta } ( i ) = Z _ { t } ,
$$

while at θ it is $Z _ { t , \varepsilon } .$ . After adding the known downward component to the observed upward component, the observation law is P of the favorable action and $Q _ { \varepsilon }$ at the favorable action, with gap $\Delta = \varepsilon / 2 4$ . This invertible transformation preserves relative entropy. Even revealing $\ell ^ { \downarrow }$ before play only strengthens the learner. Lemma 13 therefore gives $\Omega ( \sqrt { K T } )$ regret.

If $| \mathcal { F } | \le T ^ { 1 / 3 }$ then $K = | { \mathcal { F } } |$ and this is $\Omega ( { \sqrt { | { \mathcal { F } } | T } } )$ ; otherwise $K = \Theta ( T ^ { 1 / 3 } )$ and it is $\Omega ( T ^ { 2 / 3 } )$ Averaging over the random choice of θ and common-noise sequence yields a deterministic oblivious sequence for every randomized learner. □

## D.5 BV one-point lower bound

We use Lemmas 12 and 13 with $K = | { \mathcal { F } } | \leq T$ and $\varepsilon = a \sqrt { K / T }$

Proof of Theorem 8. The standard adversarial K-armed bandit upper bound gives the matching upper rate. It remains to prove the lower bound.

Use fixed actions [K] and one expert per action. Under the null, draw $Z _ { t }$ independently and set

$$
\ell _ { t } ^ { 0 } ( i ) = Z _ { t }
$$

for every action i. Under alternative θ, use the coupled pair $( Z _ { t } , Z _ { t , \varepsilon } )$ and set

$$
\ell _ { t } ^ { \theta } ( i ) = \left\{ { Z } _ { t , \varepsilon } , \ : \ : i = \theta , \right.
$$

Every coordinate lies in $[ 1 / 4 , 1 / 2 ]$ . The realized vector is constant except for one downward dip, so its total variation satisfies

$$
\mathrm { T V } ( \ell _ { t } ^ { \theta } ) \leq 2 | Z _ { t } - Z _ { t , \varepsilon } | \leq \frac { \varepsilon } { 8 } \leq 1 .
$$

Thus the hard family lies in the BV model with variation at most one on every round.

We also verify explicit realizability in the separated monotone problem. In the null environment, take

$$
\ell _ { t } ^ { \uparrow , 0 } ( i ) = Z _ { t } { \bf 1 } \{ i = K \} , \qquad \ell _ { t } ^ { \downarrow , 0 } ( i ) = Z _ { t } { \bf 1 } \{ i < K \} .
$$

For an alternative $\theta < K$ , take

$$
\begin{array} { c } { { \ell _ { t } ^ { \uparrow , \theta } ( i ) = Z _ { t } { \bf 1 } \{ i > \theta \} , } } \\ { { \ell _ { t } ^ { \downarrow , \theta } ( i ) = Z _ { t } { \bf 1 } \{ i < \theta \} + Z _ { t , \varepsilon } { \bf 1 } \{ i = \theta \} . } } \end{array}
$$

For $\theta = K$ , take

$$
\ell _ { t } ^ { \uparrow , K } ( i ) = Z _ { t , \varepsilon } \mathbf { 1 } \{ i = K \} , \qquad \ell _ { t } ^ { \downarrow , K } ( i ) = Z _ { t } \mathbf { 1 } \{ i < K \} .
$$

In every case the first component is nondecreasing and zero at the least action, the second is nonincreasing and zero at the greatest action, both are in $[ 0 , 1 / 2 ]$ , and their sum is exactly $\ell _ { t } ^ { \theta }$

A one-point learner sees law P at every ordinary action and law $Q _ { \varepsilon }$ at the hidden favorable action, with mean loss gap $\Delta = \varepsilon / 2 4$ . Lemma 13 gives

$$
\frac { 1 } { K } \sum _ { \theta } \mathbb { E } _ { \theta } \mathrm { R e g } _ { T } = \Omega \big ( \sqrt { | \mathcal { F } | T } \big ) .
$$

Yao’s principle and averaging yield a deterministic oblivious loss sequence for every randomized learner. □

## E Proof of Dynamic Pricing Using Menus

Proof of Theorem 6. (i) No expert advice. Use $\mathcal { P } = \{ j / T : j = 0 , \dots , T \}$ and run Algorithm 2 on its $T + 1$ constant-price experts. These experts are constructed by the learner and require no external advice. The reduction above and Theorem 2 give revenue regret against the best grid price at most

$$
O \big ( \log ( | \mathcal { P } | T ) \sqrt { T \log | \mathcal { P } | } \big ) = O \big ( \sqrt { T } \log ^ { 3 / 2 } T \big ) .
$$

For any price $p ,$ round down to $p ^ { \prime } = \lfloor { T p } \rfloor / T$ . Whenever a buyer accepts $p ,$ it also accepts $p ^ { \prime } ,$ , and $r _ { t } ( p ^ { \prime } ) \geq r _ { t } ( p ) - 1 / T$ on every round. Thus the best grid price loses at most one in total revenue relative to the best price in [0, 1].

(ii) Expert advice. Run the same algorithm on the current recommendations $a _ { t , f }$ . The menu choice reveals both sampled prices’ acceptance decisions and supplies the separated component observations. Its expected payment is the two revenues’ average, so revenue regret against the best expert is twice the algorithm’s loss regret. Theorem 2 gives the stated bound, even when the recommendations change and reorder arbitrarily.

In both cases the outside option and utility-maximizing menu choice give incentive compatibility and individual rationality in expected utility. □

## References

Alekh Agarwal, Ofer Dekel, and Lin Xiao. Optimal algorithms for online convex optimization with multi-point bandit feedback. In Proceedings of the 23rd Conference on Learning Theory (COLT), 2010.

Alekh Agarwal, Haipeng Luo, Behnam Neyshabur, and Robert E. Schapire. Corralling a band of bandit algorithms. In Proceedings of the 30th Conference on Learning Theory (COLT), 2017.

Noga Alon, Nicol\`o Cesa-Bianchi, Ofer Dekel, and Tomer Koren. Online learning with feedback graphs: Beyond bandits. In Proceedings of the 28th Conference on Learning Theory (COLT), 2015.

Jean-Yves Audibert and S´ebastien Bubeck. Minimax policies for adversarial and stochastic bandits. In Proceedings of the 22nd Conference on Learning Theory (COLT), 2009.

Peter Auer, Nicol\`o Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002.

S´ebastien Bubeck and Nicol\`o Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multiarmed bandit problems. Foundations and Trends in Machine Learning, 5(1):1–122, 2012.

S´ebastien Bubeck, Michael B. Cohen, and Yuanzhi Li. Sparsity, variance and curvature in multiarmed bandits. In Proceedings of Algorithmic Learning Theory (ALT), 2018.

Nicol\`o Cesa-Bianchi and G´abor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

Richard Combes and Alexandre Prouti\`ere. Unimodal bandits: Regret lower bounds and optimal algorithms. In Proceedings of the 31st International Conference on Machine Learning (ICML), 2014.

John C. Duchi, Michael I. Jordan, Martin J. Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61(5):2788–2806, 2015.

Abraham D. Flaxman, Adam Tauman Kalai, and H. Brendan McMahan. Online convex optimization in the bandit setting: Gradient descent without a gradient. In Proceedings of the 16th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), 2005.

Dylan J. Foster, Zhiyuan Li, Thodoris Lykouris, Karthik Sridharan, and Eva Tardos. Learning in<sup>´</sup> games: Robustness of fast convergence. In Advances in Neural Information Processing Systems 29 (NeurIPS), 2016.

Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997.

W. Keith Hastings. Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1):97–109, 1970.

Woonghee Tim Huh and Paat Rusmevichientong. A nonparametric asymptotic analysis of inventory planning with censored demand. Mathematics of Operations Research, 34(1):103–123, 2009.

Robert Kleinberg and Tom Leighton. The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In Proceedings of the 44th Annual IEEE Symposium on Foundations of Computer Science (FOCS), 2003.

Robert Kleinberg, Aleksandrs Slivkins, and Eli Upfal. Multi-armed bandits in metric spaces. In Proceedings of the 40th Annual ACM Symposium on Theory of Computing (STOC), 2008.

Tor Lattimore and Csaba Szepesv´ari. Bandit Algorithms. Cambridge University Press, 2020.

Nick Littlestone and Manfred K. Warmuth. The weighted majority algorithm. Information and Computation, 108(2):212–261, 1994.

Shie Mannor and Ohad Shamir. From bandits to experts: On the value of side-observations. In Advances in Neural Information Processing Systems 24 (NeurIPS), 2011.

H. Brendan McMahan and Matthew Streeter. Tighter bounds for multi-armed bandits with expert advice. In Proceedings of the 22nd Conference on Learning Theory (COLT), 2009.

Yevgeny Seldin, Peter L. Bartlett, Koby Crammer, and Yasin Abbasi-Yadkori. Prediction with limited advice and multiarmed bandits with paid observations. In Proceedings of the 31st International Conference on Machine Learning (ICML), 2014.

Ohad Shamir. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. Journal of Machine Learning Research, 18(52):1–11, 2017.

Chen-Yu Wei and Haipeng Luo. More adaptive algorithms for adversarial bandits. In Proceedings of the 31st Conference on Learning Theory (COLT), 2018.

Jia Yuan Yu and Shie Mannor. Unimodal bandits. In Proceedings of the 28th International Conference on Machine Learning (ICML), 2011.

Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The K-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012.