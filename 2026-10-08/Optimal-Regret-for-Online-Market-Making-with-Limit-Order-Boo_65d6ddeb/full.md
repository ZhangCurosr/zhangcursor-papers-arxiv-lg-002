# Optimal Regret for Online Market Making with Limit Order Book

Maria Elena Vischi mariaelena.vischi@polimi.it Politecnico di Milano

Francesco Emanuele Stradi francescoemanuele.stradi@polimi.it Politecnico di Milano

Alberto Marchesi alberto.marchesi@polimi.it Politecnico di Milano

October 8, 2026

## Abstract

We study online learning in market making, where, at each round, a market maker posts bid and ask prices before observing the market price and the private valuation of an incoming trader. In this setting, Maran and Restelli [2026] introduce a feedback model motivated by limit order books, in which the trader’s valuation is revealed only if no transaction occurs. Assuming that trader valuations are drawn i.i.d. from an unknown distribution while market prices are chosen adversarially, they establish an expected regret bound of Oe(T<sup>2/3</sup>). In this work, we improve upon this guarantee by establishing a high-probability regret bound of Oe( T). As a warm-up, we first consider the full-feedback setting. We introduce a discretization of the bid–ask space based on two coupled grids and combine it with Hedge to achieve the desired regret rate. Building on these ideas, we then address the substantially weaker feedback induced by a limit order book and develop an algorithm that achieves the same guarantee. Finally, we investigate the limits of learnability in fully adversarial environments, where the valuations may vary arbitrarily as well. Perhaps surprisingly, we show that when both market prices and trader valuations are chosen adversarially, sublinear regret is impossible even under full feedback, thereby motivating our stochastic assumption on the valuations.

## 1 Introduction

Market makers provide liquidity to financial markets by posting prices at which they are willing to buy and sell an asset [Amihud and Mendelson, 1986]. They aim to profit from the bid–ask spread while managing the risks of price changes and trading with better-informed participants, a central theme in the economics literature on market making. Today, market making is a core activity of quantitative trading firms such as Jane Street and Citadel Securities, which use mathematical models and automated trading systems to provide liquidity across financial markets [Biais et al., 2005]. Designing efective strategies for choosing and updating these quotes is therefore an important problem in algorithmic trading [Glosten and Harris, 1988, Madhavan, 2000]. A central challenge in designing market-making strategies is that quotes must be chosen under uncertainty about both the market price of the asset and the willingness of incoming traders to transact. Moreover, the available information depends on the quotes themselves, linking current trading decisions to future learning opportunities. Online learning provides a framework for studying this interaction through a sequential decision-making problem. At each round, a market maker posts a bid and an ask before observing the market price and the incoming trader’s private valuation. Thus, the performance is measured by regret against the best fixed bid–ask pair in hindsight

[Cesa-Bianchi et al., 2024] study this problem under two-bit feedback, where the learner observes the market price and whether the trader buys, sells, or does not trade, while the trader’s valuation remains hidden. They establish tight regret guarantees under several assumptions on market prices and valuations. In particular, the minimax regret is of order $T ^ { 2 / 3 }$ when market prices and trader valuations are i.i.d. and independent of each other. Their lower bound holds even under Lipschitz assumptions on the underlying distributions, showing that the dificulty persists in regular stochastic environments. They also establish impossibility results when suitable assumptions on regularity and independence are absent. These results highlight the role of the information available to the market maker in determining learnability.

More recently, Maran and Restelli [2026] introduce a richer feedback model motivated by limit order books. In this model, the trader’s valuation remains hidden when a transaction occurs but is revealed when no transaction takes place. Thus, the posted quotes determine both the reward and the information available for future decisions. Under i.i.d. trader valuations, they obtain high-probability regret bounds of $\widetilde { \mathcal { O } } ( \sqrt { T } )$ for independent stochastic market prices and extend these guarantees to certain mean-reverting price processes, without imposing smoothness on the valuation distribution. Diferently, when the market prices are chosen by an oblivious adversary, while the valuations remain stochastic and stationary, they establish an expected regret bound of $\widetilde { \mathcal { O } } ( T ^ { 2 / 3 } )$

This leaves a gap between the guarantees available for stochastic and adversarial market prices under limit order book feedback. Is the slower rate an unavoidable consequence of arbitrary market prices, or can the information revealed by the order book still support a $\widetilde { \mathcal { O } } ( \sqrt { T } )$ guarantee? Moreover, both guarantees rely on the assumption that trader valuations are drawn from a fixed distribution. It is therefore natural to ask whether learning remains possible when this assumption is removed. Specifically, in this work we address the following two research questions:

(i) Can we achieve $\widetilde { \mathcal { O } } ( \sqrt { T } )$ regret under limit order book feedback with adversarial market

prices?

(ii) Can sublinear regret also be achieved when both market prices and trader valuations are adversarial?

We answer the first question afirmatively and the second negatively. When trader valuations are drawn i.i.d. from an unknown distribution, we establish a high-probability regret bound of $\widetilde { \mathcal { O } } ( \sqrt { T } )$ while allowing market prices to remain arbitrary. Finally, we show that sublinear regret is impossible in fully adversarial environments, even under full feedback.

## 1.1 Original Contributions

We study online market making under diferent assumptions on the environment and the feedback available to the learner. Our main focus is the setting in which market prices are chosen by an adaptive adversary, while private valuations are drawn from an arbitrary distribution. We establish optimal regret rates under both full feedback and limit order book feedback, and show that learning becomes impossible when private valuations are also chosen adversarially. Our contributions can be summarized as follows:

• Optimal regret under full feedback. As a warm-up, we establish a $\widetilde { \mathcal { O } } ( \sqrt { T } )$ regret bound under full feedback. Our algorithm combines two Hedge instances, one for bid prices and one for ask prices. A key observation is that swapping the sampled quotes whenever they violate the feasibility constraint never decreases utility, allowing us to analyze the two instances separately and transfer their guarantees to the actions actually played.

• Optimal high-probability regret under limit order book. For the feedback model introduced by Maran and Restelli [2026], we establish a $\widetilde { \mathcal { O } } ( \sqrt { T } )$ regret bound holding with high probability, even against an adaptive adversary. This improves upon the previous $\mathcal { \widetilde { O } } ( T ^ { 2 / 3 } )$ bound, which held in expectation and assumed an oblivious adversary. Our analysis exploits the feedback structure. Specifically, each round allows the learner to reconstruct losses for the interval between the sampled quotes. Using this additional information, we achieve the same regret rate as under full feedback.

• Impossibility in a fully adversarial environment. When both market prices and private valuations are chosen adversarially, we prove a linear regret lower bound. This impossibility result holds even under full feedback, showing that observing all parameters at the end of each round is insuficient to guarantee sublinear regret.

Table 1: Comparison between our results and the state-of-the-art.
<table><tr><td>Feedback</td><td>(Mt, Vt)</td><td>[Maran and Restelli, 2026]</td><td>Our Work</td></tr><tr><td>Full</td><td> $( \mathbf { a d v } ) + ( \mathbf { s t o c } )$  1</td><td>x</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } )$ </td></tr><tr><td></td><td> $( \mathbf { a d v } ) + ( \mathbf { a d v } )$  </td><td>x</td><td>Ω(T)</td></tr><tr><td>LOB</td><td> $( \mathbf { a d v } ) + ( \mathbf { s t o c } )$ </td><td> $\mathcal { \widetilde { O } } ( T ^ { 2 / 3 } )$ </td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } )$ </td></tr></table>

The $\tilde { \mathcal { O } } ( { \sqrt { T } } )$ upper bounds are optimal up to logarithmic factors, matching the standard $\Omega ( { \sqrt { T } } )$ lower bound for the multi-armed bandit problem [Lattimore and Szepesv´ari, 2020]. Together, our results clarify how the assumptions on the environment and the available feedback afect the learnability of market making.

Table 1 summarizes our guarantees and compares them with previous work.

## 1.2 Related Works

Market making has long been studied in market microstructure [Bouchaud et al., 2008], with traditional approaches often formulating the choice of bid and ask prices as a stochastic control problem [Gu´eant, 2017]. An alternative approach treats market making as an online learning problem, following the work of Abernethy and Kale [2013] and its more recent developments by Cesa-Bianchi et al. [2024] and Maran and Restelli [2026].

Within this line of research, Cesa-Bianchi et al. [2024] study two-bit feedback about private valuation, where the learner observes only whether a purchase and a sale occur, and derive regret guarantees under regularity assumptions on the distributions describing the environment. Although these assumptions support the analysis, they may exclude behavior concentrated around particular prices. For example, the tendency of bidders to bid round numbers documented by van Oldeniel et al. [2025] motivates considering distributions that need not be smooth.

Alongside the assumptions on the environment, the information available to the learner is another key aspect of the problem. Since two-bit feedback may omit information available in a limit order book, Maran and Restelli [2026] introduce a richer feedback model, for which they establish an expected regret bound of $\widetilde { \mathcal { O } } ( T ^ { 2 / 3 } )$ against an oblivious adversary. Their guarantees are limited to expected regret in the adversarialprice case and do not cover the fully adversarial setting in which both market prices and private valuations are chosen adversarially.

More broadly, market making is related to other online learning problems in economics, such as dynamic pricing [Kleinberg and Leighton, 2003], first-price auctions [Cesa-Bianchi et al., 2026, Myerson, 1981] and bilateral trade [Cesa-Bianchi et al., 2023, Myerson and Satterthwaite, 1983], where the feedback makes it dificult to infer how rewards vary across actions.

## 2 Preliminaries

We study online market making over $T \in \mathbb { N } _ { \geq 0 }$ rounds. At each round $t \in \{ 1 , \ldots , T \}$ , the learner $( i . e .$ the market maker) posts a bid quote $B _ { t }$ and an ask quote $A _ { t } .$ , where $A _ { t } \ \geq B _ { t } .$ We normalize prices to the interval [0, 1], which can always be done through an afine transformation of the original price space. Accordingly, the feasible decision set is:

$$
\mathcal { U } = \{ ( a , b ) \in [ 0 , 1 ] ^ { 2 } : a \geq b \} .
$$

At the beginning of round $t \in \{ 1 , \ldots , T \}$ , a trader with private valuation $V _ { t } \in [ 0 , 1 ]$ arrives. The trader buys the asset from the market maker whenever $V _ { t } > A _ { t }$ , and sells the asset to the market maker whenever $V _ { t } \ \leq \ B _ { t }$ . If $V _ { t } \in ( B _ { t } , A _ { t } ]$ , no trade occurs. We assume that, whenever a trade takes place, the market maker immediately ofloads its position on a perfectly liquid secondary market at price $M _ { t } \in [ 0 , 1 ]$ , which is observed at the end of the round only, incurring no additional transaction costs or inventory constraints. The resulting utility function of the market maker is:

$$
u ( a , b , v , m ) = ( a - m ) \mathbb { I } \{ v > a \} + ( m - b ) \mathbb { I } \{ v \leq b \} .
$$

Throughout the paper, we will use the shorthand notation of $u _ { t } ( a , b ) = u ( a , b , V _ { t } , M _ { t } )$

The online market making interaction is summarized in Protocol 1. We consider two possible types of feedback: full feedback (Line 10) and limit order book feedback (Lines 4–9). In both cases, the learner observes the market price $M _ { t }$ at the end of each round. The realized trader valuation $V _ { t }$ is always observed at the end of each round under full feedback, while it is observed only if no trade occurs in the limit order book case.

Protocol 1 Online Market Making   
1: for $\overline { { t = 1 , \ldots , T } }$ do   
2: A new trader with valuation $V _ { t }$ arrives   
3: The learner posts ask and bid prices $( A _ { t } , B _ { t } ) \in \mathcal { U }$   
4: $M _ { t } \in [ 0 , 1 ]$ is observed Limit order book   
<sup>5:</sup> <sub>6:</sub> if $B _ { t } < V _ { t } \leq A _ { t }$ $V _ { t } \in [ 0 , 1 ]$ <sup>then</sup>observed   
7: else   
8: The learner observes $\mathbb { 1 } \{ B _ { t } \geq V _ { t } \}$ and $\mathbb { 1 } \{ A _ { t } < V _ { t } \}$   
9: end if   
10: $( V _ { t } , M _ { t } ) \in [ 0 , 1 ] ^ { 2 }$ is observed Full feedback   
11: The learner gains utility $u _ { t } ( A _ { t } , B _ { t } )$   
12: end for

We consider two diferent models for the sequence $\{ ( V _ { t } , M _ { t } ) \} _ { t = 1 } ^ { T } \colon$

$( \mathbf { a d v } ) + ( \mathbf { s t o c } ) \ : \ : \{ V _ { t } \} _ { t = 1 } ^ { T }$ is an i.i.d. sequence, while $\{ M _ { t } \} _ { t = 1 } ^ { T }$ is chosen by an adaptive adversary.

(adv) + (adv) Fully adversarial setting, in which the sequence $\{ ( V _ { t } , M _ { t } ) \} _ { t = 1 } ^ { T }$ is generated by an oblivious adversary.

The first case (adv) + (stoc) is the main focus of this work, while we provide a lower bound, which naturally extends to the case of an adaptive adversary, for the second case $( \mathbf { a d v } ) + ( \mathbf { a d v } )$

Let us remark that we make no assumption on the regularity of the sequence of $\{ V _ { t } \} _ { t = 1 } ^ { T }$ . Indeed, diferently from Cesa-Bianchi et al. [2024], we do not assume that the c.d.f. of the trader valuations distribution is Lipschitz–continuous in any part of the work.

## 2.1 Performance Metric

We use as performance metric the notion of cumulative regret, which is defined as follows:

$$
R _ { T } = \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } \mathbb { E } _ { V } \left[ \sum _ { t = 1 } ^ { T } ( u _ { t } ( a , b ) - u _ { t } ( A _ { t } , B _ { t } ) ) \right] ,
$$

where the expectation is taken over the randomization of the environment. We define the c.d.f. for the random variable V underlying the trader valuations in the (adv)+(stoc) as $F ( x ) = \mathbb { P } ( V \leq x )$ and, accordingly, we

define the expected utility as:

$$
U _ { t } ( a , b ) : = \mathbb { E } _ { V } [ u _ { t } ( a , b ) ] = ( M _ { t } - b ) F ( b ) + ( a - M _ { t } ) ( 1 - F ( a ) ) .
$$

We notice that the utility can be decomposed into two components, one for the bid and one for the ask price. Thus, we denote:

$$
u _ { t } ^ { A } ( a ) : = ( a - M _ { t } ) \mathbb { 1 } \{ V _ { t } > a \} , \qquad u _ { t } ^ { B } ( b ) : = ( M _ { t } - b ) \mathbb { 1 } \{ V _ { t } \leq b \} ,
$$

so that $u _ { t } ( a , b ) = u _ { t } ^ { B } ( b ) + u _ { t } ^ { A } ( a )$ . Defining $\begin{array} { r } { \mu _ { T } = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } M _ { t } } \end{array}$ , we also notice that:

$$
\operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } U _ { t } ( a , b ) = \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } \bigg [ ( \mu _ { T } - b ) F ( b ) + ( a - \mu _ { T } ) ( 1 - F ( a ) ) \bigg ] .
$$

Finally, defining $U ( a , b ) : = ( \mu _ { T } - b ) F ( b ) + ( a - \mu _ { T } ) ( 1 - F ( a ) )$ , we can rewrite the regret as:

$$
R _ { T } = \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } \sum _ { t = 1 } ^ { T } \left[ U _ { t } ( a , b ) - U _ { t } ( A _ { t } , B _ { t } ) \right] = T \cdot \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } U ( a , b ) - \sum _ { t = 1 } ^ { T } U _ { t } ( A _ { t } , B _ { t } ) .\tag{1}
$$

Notice that, in case of adversarial trader valuations, where no statistical assumption is made on the sequence $\{ ( M _ { t } , V _ { t } ) \} _ { t = 1 } ^ { T }$ , no expectation is taken in the regret definition.

In the next section, we start with assuming full feedback and stochastic trader valuations.

## 3 Warm up: The Full Feedback Case

In this section, we start by considering the (adv)+(stoc) scenario.

This setting is the one already studied in [Maran and Restelli, 2026], with the diference that we assume, throughout this section, to have full feedback, while we let the adversary to be adaptive. Our approach will employ two instances of the Hedge algorithm [Freund and Schapire, 1997], thus significantly departing from the existing technique of Maran and Restelli [2026], where the algorithm is mainly based on Follow the Perturbed Leader (FTPL) [Hutter and Poland, 2005].

## 3.1 Algorithm

Algorithm 2 Market Making with Experts   
Input: Time horizon $\overline { { \boldsymbol { T } , \boldsymbol { K } , \boldsymbol { \eta } } }$   
1: Initialize $A _ { 1 }  1 , B _ { 1 }  0 ,$ and grid $\mathcal { G }$   
2: Initialize $W _ { 1 } ^ { A } , W _ { 1 } ^ { B }$ to 1 on $\mathcal { G }$   
3: for $t = 1 , \dots , T$ do   
4: Form $w _ { t } ^ { A } , w _ { t } ^ { B }$ by normalizing $W _ { t } ^ { A } , W _ { t } ^ { B }$   
5: Draw $Y _ { t } \sim w _ { t } ^ { A }$ and $X _ { t } \sim w _ { t } ^ { B }$   
6: Post prices $( A _ { t } , B _ { t } ) \gets ( \operatorname* { m a x } \{ X _ { t } , Y _ { t } \}$ , min $\{ X _ { t } , Y _ { t } \} )$   
7: Gain utility $u _ { t } ( A _ { t } , B _ { t } )$   
8: Observe $( M _ { t } , V _ { t } )$ and reconstruct $u _ { t } ^ { A } ( a ) , u _ { t } ^ { B } ( b )$ for all $a , b \in { \mathcal { G } }$   
9: Update weights for all $a , b \in { \mathcal { G } } \colon$   
$W _ { t + 1 } ^ { A } ( a )  W _ { t } ^ { A } ( a ) \exp ( \eta \ u _ { t } ^ { A } ( a ) ) , \quad W _ { t + 1 } ^ { B } ( b )  W _ { t } ^ { B } ( b ) \exp ( \eta \ u _ { t } ^ { B } ( b ) )$   
10: end for

Our algorithm, whose pseudocode is provided in Algorithm $2 ,$ discretizes the bid and ask prices using the uniform grid $\begin{array} { r } { \mathcal { G } = \left\{ \frac { i } { K } : i = 0 , . . . , K \right\} } \end{array}$ . The resulting Cartesian product ${ \mathcal { G } } \times { \mathcal { G } }$ consists of $( K + 1 ) ^ { 2 }$ points in $[ 0 , 1 ] ^ { 2 }$ and serves as the discrete action set. We then run two independent instances of Hedge [Orabona, 2019] with the same learning rate $\eta > 0$ , one for bid prices and one for ask prices. Each instance maintains weights $\{ W _ { t } ^ { B } \} _ { t = 1 } ^ { T }$ and $\{ W _ { t } ^ { A } \} _ { t = 1 } ^ { T }$ , respectively, and the corresponding probability distributions $\{ w _ { t } ^ { B } \} _ { t = 1 } ^ { T }$ and $\{ w _ { t } ^ { \bar { A } } \} _ { t = 1 } ^ { T }$ over the one-dimensional grid G (Line 2). At each round, the algorithm samples a bid and an ask price from the corresponding distributions and swaps them if the resulting pair violates the constraint defining U (Line 6). It then posts the resulting pair of prices and updates the weights of both Hedge instances (Line 9). We emphasize that the swap only afects the prices played, not the weight updates. Since feedback is fully observable, at the end of each round we can evaluate the utility associated with every grid price in its respective role as a bid or an ask. Each Hedge instance is therefore updated using these utilities, regardless of whether the sampled prices were swapped (Lines 8–9).

## 3.2 Theoretical Results

In this section, we analyze Algorithm 2 and establish the associated $\tilde { \mathcal { O } } ( \sqrt { T } )$ regret bound.

We start by stating the following lemma, which shows how enforcing feasibility by swapping the prices predicted by the two Hedge instances never decreases the utility. More precisely, the utility remains unchanged when the sampled prices already satisfy the constraint, and increases by $X _ { t } - Y _ { t }$ otherwise. Thus, the swap in Line 6 of Algorithm 2 guarantees a pair of prices whose utility is at least as large as that of the original predictions. This property will allow us to transfer the utility guarantees of the two Hedge instances to the actions actually played by the algorithm.<sup>2</sup>

Lemma 3.1. Consider $X _ { t } \sim w _ { t } ^ { B }$ and $Y _ { t } \sim w _ { t } ^ { A }$ . Then it holds that:

$$
\begin{array} { r } { u _ { t } ( \operatorname* { m a x } \{ X _ { t } , Y _ { t } \} , \operatorname* { m i n } \{ X _ { t } , Y _ { t } \} ) = u _ { t } ( Y _ { t } , X _ { t } ) + ( X _ { t } - Y _ { t } ) _ { + } , } \end{array}
$$

where we denote $( x ) _ { + } = \operatorname* { m a x } \{ 0 , x \}$ . As a consequence, for any $t \in \{ 1 , \ldots , T \}$ it holds that ${ U _ { t } } ( A _ { t } , B _ { t } ) \ \geq$ $U _ { t } ( Y _ { t } , X _ { t } )$ and $U ( A _ { t } , B _ { t } ) \geq U ( Y _ { t } , X _ { t } )$

Next, we show that restricting the action set to the grid incurs only a bounded loss in expected utility. Specifically, there exists a grid point whose expected utility is within $5 / { _ { 2 K } }$ of the supremum over the continuous feasible set U. This allows us to relate performance on the grid to the continuous optimum.

Lemma 3.2. There exists a point $( A ^ { * } , B ^ { * } ) \in \mathcal { G } \times \mathcal { G }$ such that:

$$
0 \le \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } U ( a , b ) - U ( A ^ { * } , B ^ { * } ) \le \frac { 5 } { 2 K } .
$$

Figure 1 illustrates the idea behind the proof of Lemma 3.2. We start from a nearly optimal point and choose a nearby point on the grid, taking care not to cross the boundary defined by $\mu _ { T }$ , as doing so could reduce the expected utility. This choice keeps the discretization loss small. We provide the full argument in Appendix A.

We are now ready to provide the final regret bound attained by Algorithm 2.

Theorem 3.3. Let $\delta \in ( 0 , 1 )$ and $T \geq 2$ . Algorithm 2 with $\begin{array} { r } { K = \lceil \sqrt { T } \rceil , \eta = \sqrt { \frac { \log ( T ) } { T } } } \end{array}$ attains regret:

$$
R _ { T } \leq \mathcal { O } \left( \sqrt { T \log ( T ) } + \sqrt { T \log ( 1 / \delta ) } + \log ( 1 / \delta ) \right) ,
$$

with probability at least $1 - \delta$

In Theorem 3.3, we derive a regret bound by adding the standard Hedge guarantees for the two onedimensional instances. We analyze each instance using its original sampled quote, before any swap takes place. Although the swap couples the bid and ask prices actually posted, Lemma 3.1 shows that it never decreases utility. We can therefore analyze the two Hedge instances separately and carry their combined guarantee over to the actions played by the algorithm. The regret bound in Theorem 3.3 depends only logarithmically on the grid size $K ,$ , as a consequence of the Hedge guarantee. This dependence is optimal in the standard expert setting. We would expect a diferent dependence on $K$ under limit order book feedback, which provides less information than full feedback.

![](images/05f0692a31ad1ebcc92f6d32a6e5d88e9cf92054ae4b2036da4edddc7bf86215.jpg)  
Figure 1: Discretization around the optimal action. $L e f t { : }$ inward rounding. Right: correction when both coordinates would cross $\mu _ { T }$

## 4 Optimal Bounds For The Limit Order Book Case

In this section, we study the more interesting setting of the limit order book feedback, where the information available to the learner depends on whether a trade occurs. If no exchange takes place, the learner receives full feedback; otherwise, only the utility is observed with the type of trade.

## 4.1 Algorithm

We now present Algorithm 3 for the limit order book feedback and analyze its regret.

Similarly to the previous approach, Algorithm 3 treats the bid and ask prices separately, maintaining disjoint grids $\mathcal { G } _ { B }$ and $\mathcal { G } _ { A }$ (Line 1), weights $W _ { t } ^ { B }$ and $W _ { t } ^ { A }$ (Line 2), and sampling distributions $w _ { t } ^ { \lambda }$ and $w _ { t } ^ { B }$ (Lines 4–6). Unlike in the full-feedback setting, we must account for the possibility that grid points coincide with atoms of $F ^ { 3 }$ , which might afect the observed feedback. This issue is addressed in Lemma 4.1 in Appendix B, where we show that, independently of the underlying distribution of $\{ V _ { t } \} _ { t = 1 } ^ { T }$ , we can still collect the necessary information for our estimation.

For an integer $K \geq 1$ , we define the two grids as:

$$
\mathcal G _ { B } = \left\{ \frac { i } { K } + \frac { 1 } { 4 K } : i = 0 , \ldots , K - 1 \right\} \cup \{ 1 \} , \ \mathcal G _ { A } = \left\{ \frac { i } { K } + \frac { 1 } { 2 K } : i = 0 , \ldots , K - 1 \right\} \cup \{ 0 \} .
$$

Algorithm 3 Market Making with Limit Order Book   
Input: Time horizon T, parameters $\overline { { \eta , \beta , K } }$   
1: Initialize ${ \mathcal { G } } _ { A } , { \mathcal { G } } _ { B }$   
2: Set $W _ { 1 } ^ { A } , W _ { 1 } ^ { B }$ to 1   
3: for $t = 1 , \dots , T$ do   
4: Form probabilities $w _ { t } ^ { A } , w _ { t } ^ { B }$ normalizing $W _ { t } ^ { A } , W _ { t } ^ { B }$   
5: Compute $Q _ { t }$ with Equation (3) in internal grid points, $Q _ { t } ( 0 ) = Q _ { t } ( 1 ) = 1$   
6: Draw $X _ { t } \sim w _ { t } ^ { B }$ and $\bar { Y } _ { t } \sim w _ { t } ^ { A }$   
7: Play prices $( A _ { t } , B _ { t } ) = ( \operatorname* { m a x } \{ X _ { t } , Y _ { t } \}$ , min $\{ X _ { t } , Y _ { t } \} )$   
8: Observe $M _ { t } ,$ observe $V _ { t }$ if no trade occurred   
9: $\forall z \in ( \mathcal { G } _ { A } \cup \mathcal { G } _ { B } ) \cap [ B _ { t } , A _ { t } ] .$ , compute $\ell _ { t } ^ { s } ( z )$ , for $s \in \{ A , B \}$ respectively   
10: Form the loss estimator $\hat { \ell } _ { t } ^ { s } ,$ with $s \in \{ A , B \}$ , as in Equation (2)   
11: Update weights $W _ { t + 1 } ^ { A } , W _ { t + 1 } ^ { \tilde { B } }$   
12: end for

Each grid contains K internal points and one endpoint. We run two parallel instances of an EXP3-based algorithm [Auer et al., 2002]. For convenience, we express the utility contributions as losses in [0, 1] by defining $\mathring { \ell _ { t } ^ { A } } ( a ) : = \mathring { ( 1 - u _ { t } ^ { A } ( a ) ) } \big / 2 , \ell _ { t } ^ { B } ( b ) \ \backslash = \ \mathring { ( 1 - u _ { t } ^ { B } ( b ) ) } \big / 2$ . Let $\beta > 0$ be an implicit exploration parameter [Neu, 2015]. For each internal grid point $x \in \mathcal G _ { s }$ , with $s \in \{ A , B \}$ , let $Q _ { t } ( x ) : = \mathbb { P } \big ( x \in [ \bar { B } _ { t } , A _ { t } ] \cup \{ 0 , 1 \} \big )$  denote the probability that its loss is observed at round t. We set $Q _ { t } ( 1 ) = Q _ { t } ( 0 ) = 1$ , since losses at the endpoints can always be computed from $M _ { t } .$ , which is observed at every round. We further denote $O _ { t } ( x ) : = \mathbb { 1 } \{ x \in [ B _ { t } , A _ { t } ] \cup \{ 0 , 1 \} \}$ as the event that a point $x \in \mathcal { G } _ { s } , \ s \in \{ A , B \}$ has deducible utility. The limit order book feedback allows us to recover the exact losses at all grid points in $[ B _ { t } , A _ { t } ]$ , as we will show in Lemma 4.1. We use these observed losses to construct the estimates:

$$
\widehat { \ell } _ { t } ^ { A } ( x ) = \frac { O _ { t } ( x ) \ell _ { t } ^ { A } ( x ) } { Q _ { t } ( x ) + \beta } , \qquad \widehat { \ell } _ { t } ^ { B } ( x ) = \frac { O _ { t } ( x ) \ell _ { t } ^ { B } ( x ) } { Q _ { t } ( x ) + \beta } .\tag{2}
$$

The implicit exploration parameter $\beta$ prevents the estimates from becoming too large when a grid point is rarely observed, helping to control their variance.

For each component $s \in \{ A , B \}$ , we update the weights according to:

$$
\begin{array} { r } { W _ { t + 1 } ^ { s } ( x ) = W _ { t } ^ { s } ( x ) \exp \bigl ( - \eta \hat { \ell } _ { t } ^ { s } ( x ) \bigr ) , } \end{array}
$$

and denote the corresponding normalized weights by $w _ { t } ^ { s } , s \in \{ A , B \}$ , which represent the probability of sampling for the two problems (Line 6).

Finally, we notice that for any internal grid point $x \in \mathcal G _ { s }$ , with $s \in \{ A , B \}$ , we can compute the observation probability directly from the sampling distributions:

$$
Q _ { t } ( x ) = 1 - \left( \sum _ { y \in \mathcal { Q } _ { A } : y < x } w _ { t } ^ { A } ( y ) \right) \left( \sum _ { y \in \mathcal { Q } _ { B } : y < x } w _ { t } ^ { B } ( y ) \right) - \left( \sum _ { y \in \mathcal { Q } _ { A } : y > x } w _ { t } ^ { A } ( y ) \right) \left( \sum _ { y \in \mathcal { Q } _ { B } : y > x } w _ { t } ^ { B } ( y ) \right) .\tag{3}
$$

Indeed, x lies outside the played interva $[ B _ { t } , A _ { t } ]$ precisely when both sampled prices are strictly below x or both are strictly above it. Since the prices are sampled independently, the probabilities of these two events are given by the products above.

## 4.2 Theoretical results

In this section, we establish the regret bound for Algorithm 3.

We begin by showing that the available feedback is suficient to compute the loss estimates used by the algorithm, and then use these estimates to analyze its performance. Specifically, the first lemma shows that, despite the limited feedback, playing a price pair $( A _ { t } , B _ { t } )$ allows us to recover the exact losses at every grid point between the two quotes. Thus, a single round provides information about an entire interval of grid points, which we use to construct the estimates in Equation (2).

![](images/fc3f08297148e2c41f6fe68414d2743af4483093fe81ee6977e2c80acc9324d5.jpg)  
Figure 2: The two panels illustrate sampling under a narrower (left) and wider (right) bid–ask spread. The bars $w _ { t } ^ { B }$ and $w _ { t } ^ { A }$ denote the sampling probabilities over prices, while $Q _ { t } ( z _ { i } )$ is the probability of observing an internal point $z _ { i } \in [ B _ { t } , A _ { t } ]$ . Concentrating the sampling mass induces greater utility, but weakens observability of internal points. Consequently, although $Q _ { t } ( z _ { i } )$ may be small, the aggregate importance-weight ratios $w _ { t } ^ { B } ( z _ { i } ) / Q _ { t } ( z _ { i } )$ and $w _ { t } ^ { A } ( z _ { i } ) / Q _ { t } ( z _ { i } )$ remain bounded, yielding the O(log K) dependence in Lemma 4.2, provided that $\beta \propto 1 / K$

Lemma 4.1. Let $( A _ { t } , B _ { t } )$ be a price pair with $B _ { t } \le A _ { t }$ , played at time step t. Then, for any point $x \in$ $\left( { \mathcal { G } } _ { A } \cup { \mathcal { G } } _ { B } \right) \cap \left[ B _ { t } , A _ { t } \right]$ , it is possible to reconstruct $\ell _ { t } ^ { s } ( x ) , \ s \in \{ A , B \}$ , after playing $( A _ { t } , B _ { t } )$

Lemma 4.1 follows from the structure of the limit order book feedback. If no trade occurs, the learner receives full feedback. Otherwise, observing whether a purchase or a sale occurred reveals on which side of the bid–ask spread the private value lies. In either case, the learner can determine how the private value compares with each grid point within the spread and, since $M _ { t }$ is always observed, reconstruct the corresponding utility.

Next, Lemma 4.2 bounds the ratios between the probability of sampling each grid point and the probability of observing its loss. These ratios measure how much information the feedback provides relative to how often each point is selected. Since each round reveals losses throughout the interval between the sampled quotes, the sum of these ratios grows only logarithmically with the grid size.

Lemma 4.2. For any time step $t \in \{ 1 , \ldots , T \}$ , with $T \geq 2$ , it holds:

$$
\sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } \frac { w _ { t } ^ { s } ( x ) } { Q _ { t } ( x ) + \beta } \leq 4 \left( \log \left( \frac { 1 + \beta } { \beta } \right) \right) .\tag{4}
$$

Figure 2 illustrates the result of Lemma 4.2. Specifically, reconstructing the losses over the interval $[ B _ { t } , A _ { t } ]$ allows us to bound the term measuring the cost of the limit order book feedback by ${ \mathcal { O } } ( \log K )$ . This lies between the full-feedback setting, where every loss is observed and the corresponding term is $\mathcal { O } ( 1 )$ , and the bandit setting, where only the loss of the sampled action is observed and the term scales as $\mathcal O ( K )$

We are now ready to show the regret guarantees of Algorithm 3.

Theorem 4.3. Let $\delta \in ( 0 , 1 )$ and $T \geq 2$ . By selecting $\eta = \beta = \sqrt { 1 / T }$ , Algorithm 3 instantiated with $K = \lceil { \sqrt { T } } \rceil$ attains:

$$
R _ { T } = \mathcal { O } \left( \sqrt { T } \log \left( \frac { T } { \delta } \right) \right) ,
$$

with probability at least $1 - \delta$

Theorem 4.3 improves upon the bound of Maran and Restelli [2026] by establishing a high-probability regret bound of $\widetilde { \mathcal { O } } ( \sqrt { T } )$ , which is optimal up to logarithmic factors and holds even against an adaptive adversary.

## 5 An impossibility result

We finally consider the full-feedback setting under the (adv)+(adv) case. We establish a linear regret lower bound, showing that market making is unlearnable in a fully adversarial environment, even when the entire valuations are observed at the end of each round.

This is done in the following theorem.

Theorem 5.1. Consider the market making problem with adversarial sequence of $\{ M _ { t } \} _ { t }$ and $\{ V _ { t } \} _ { \ i }$ and full feedback. Then, for any algorithm A, there exists a sequence $\{ ( M _ { t } , V _ { t } ) \} _ { t = 1 } ^ { T }$ where expected regret $\begin{array} { r } { \dot { R } _ { T } \geq \frac { T } { 6 } } \end{array}$

The proof of Theorem 5.1 employs the well-known Yao’s minimax principle [Yao, 1977]. Specifically, we build a distribution over problem instances in which the relevant price interval shrinks from either the left or the right at random in each round. The learner cannot anticipate which part of the interval will remain, whereas a single fixed action chosen in hindsight can be selected using the final interval. This uncertainty creates a constant gap between the learner’s expected utility and that of the hindsight benchmark in each round, yielding a linear regret lower bound. The full proof is given in Appendix C.

## References

Jacob Abernethy and Satyen Kale. Adaptive market making via online learning. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013. URL https://proceedings.neurips.cc/paper\_ files/paper/2013/file/995e1fda4a2b5f55ef0df50868bf2a8f-Paper.pdf.

Yakov Amihud and Haim Mendelson. Asset pricing and the bid-ask spread. Journal of financial Economics, 17(2):223–249, 1986.

Peter Auer, Nicol\`o Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002. doi: 10.1137/S0097539701398375. URL https://doi.org/10.1137/S0097539701398375.

Kazuoki Azuma. Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, 19 (3):357 – 367, 1967. doi: 10.2748/tmj/1178243286. URL https://doi.org/10.2748/tmj/1178243286.

Bruno Biais, Larry Glosten, and Chester Spatt. Market microstructure: A survey of microfoundations, empirical results, and policy implications. Journal of Financial Markets, 8(2):217–264, 2005. ISSN 1386-4181. doi: https://doi.org/10.1016/j.finmar.2004.11.001. URL https://www.sciencedirect.com/ science/article/pii/S1386418104000382.

Jean-Philippe Bouchaud, J. Doyne Farmer, and Fabrizio Lillo. How markets slowly digest changes in supply and demand, 2008. URL https://arxiv.org/abs/0809.0822.

Nicol\`o Cesa-Bianchi, Tommaso R. Cesari, Roberto Colomboni, Federico Fusco, and Stefano Leonardi. Repeated bilateral trade against a smoothed adversary. In Gergely Neu and Lorenzo Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 1095–1130. PMLR, 12–15 Jul 2023. URL https://proceedings.mlr.press/v195/ cesa-bianchi23a.html

Nicol\`o Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Luigi Foscari, and Vinayak Pathak. Market making without regret. ArXiv, abs/2411.13993, 2024. URL https://api.semanticscholar.org/CorpusID: 274165544.

Nicol\`o Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Federico Fusco, and Stefano Leonardi. The role of transparency in repeated first-price auctions with unknown valuations. SIAM Journal on Computing, 55(5):903–938, 2026. doi: 10.1137/24M1712308. URL https://doi.org/10.1137/24M1712308.

Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997. ISSN 0022-0000. doi: https://doi.org/10.1006/jcss.1997.1504. URL https://www.sciencedirect.com/science/article/ pii/S002200009791504X.

Lawrence R Glosten and Lawrence E Harris. Estimating the components of the bid/ask spread. Journal of financial Economics, 21(1):123–142, 1988.

Olivier Gu´eant. Optimal market making, 2017. URL https://arxiv.org/abs/1605.01862.

Marcus Hutter and Jan Poland. Adaptive online prediction by following the perturbed leader. Journal of Machine Learning Research, 6(22):639–660, 2005. URL http://jmlr.org/papers/v6/hutter05a.html.

Robert Kleinberg and Tom Leighton. The value of knowing a demand curve: Bounds on regret for online posted-price auctions. In Proceedings of the 44th Annual IEEE Symposium on Foundations of Computer Science, FOCS ’03, page 594, USA, 2003. IEEE Computer Society. ISBN 0769520405.

Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University Press, 2020.

Haipeng Luo and Robert Schapire. A drifting-games analysis for online learning and applications to boosting. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014. URL https://proceedings. neurips.cc/paper\_files/paper/2014/file/24402144990624b417229a96ad7fa7bc-Paper.pdf.

Ananth Madhavan. Market microstructure: A survey. Journal of financial markets, 3(3):205–258, 2000.

Davide Maran and Marcello Restelli. Online market making and the value of observing the order book. In Steve Hanneke and Tor Lattimore, editors, The Thirty Ninth Annual Conference on Learning Theory, June 29 - July 3, 2026, San Diego, California, USA, volume 336 of Proceedings of Machine Learning Research, pages 4969–4998. PMLR, 2026. URL https://proceedings.mlr.press/v336/maran26a.html.

Roger B. Myerson. Optimal auction design. Mathematics of Operations Research, 6(1):58–73, 1981. doi: 10.1287/moor.6.1.58. URL https://doi.org/10.1287/moor.6.1.58.

Roger B Myerson and Mark A Satterthwaite. Eficient mechanisms for bilateral trading. Journal of Economic Theory, 29(2):265–281, 1983. ISSN 0022-0531. doi: https://doi.org/10.1016/0022-0531(83)90048-0. URL https://www.sciencedirect.com/science/article/pii/0022053183900480.

Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/ paper\_files/paper/2015/file/e5a4d6bf330f23a8707bb0d6001dfbe8-Paper.pdf.

Francesco Orabona. A modern introduction to online learning. CoRR, abs/1912.13213, 2019. URL http: //arxiv.org/abs/1912.13213.

Mark van Oldeniel, Christopher M Snyder, and Adriaan R Soetevent. Round bidding in auctions. Working Paper 34564, National Bureau of Economic Research, December 2025. URL http://www.nber.org/ papers/w34564.

Andrew Chi-Chin Yao. Probabilistic computations: Toward a unified measure of complexity. In 18th Annual Symposium on Foundations of Computer Science (sfcs 1977), pages 222–227, 1977. doi: 10.1109/SFCS. 1977.24.

## Appendix

The Appendix is organized as follows:

• In Appendix A, we provide the omitted proofs and lemmas of Section 3.

• In Appendix B, we provide the omitted proofs and lemmas of Section 4.

• In Appendix C, we prove the lower bound stated in Section 5.

## A Warm up

We start by proving that, by ordering the bid and ask prices, the mean utility cannot reduce.

Lemma A.1. Consider $X _ { t } \sim w _ { t } ^ { B }$ and $Y _ { t } \sim w _ { t } ^ { A }$ . Then it holds that:

$$
\begin{array} { r } { u _ { t } ( \operatorname* { m a x } \{ X _ { t } , Y _ { t } \} , \operatorname* { m i n } \{ X _ { t } , Y _ { t } \} ) = u _ { t } ( Y _ { t } , X _ { t } ) + ( X _ { t } - Y _ { t } ) _ { + } , } \end{array}
$$

where we denote $( x ) _ { + } = \operatorname* { m a x } \{ 0 , x \}$ . As a consequence, for any $t \in \{ 1 , \ldots , T \}$ it holds that ${ U _ { t } } ( A _ { t } , B _ { t } ) \ \geq$ $U _ { t } ( Y _ { t } , X _ { t } )$ and $U ( A _ { t } , B _ { t } ) \geq U ( Y _ { t } , X _ { t } )$

Proof. If $X _ { t } = Y _ { t }$ then the statement is trivially proved, so we assume $X _ { t } \neq Y _ { t } . { \mathrm { ~ I f ~ } } X _ { t } < Y _ { t }$ , then $u _ { t } ( Y _ { t } , X _ { t } ) =$ $u _ { t } ^ { B } ( B _ { t } ) + u _ { t } ^ { A } ( A _ { t } )$ and $( X _ { t } - Y _ { t } ) _ { + } = 0$ the claim is true.

If instead $X _ { t } ~ > ~ Y _ { t }$ and $V _ { t } \neq \{ Y _ { t } , X _ { t } \}$ , then we have that for each case $V _ { t } < Y _ { t } < X _ { t } , V _ { t } \in ( Y _ { t } , X _ { t } )$ and $Y _ { t } < X _ { t } < V _ { t }$ we have an improvement of $( X _ { t } - Y _ { t } ) .$ in the utility by swapping $X _ { t }$ with $Y _ { t } .$ . For example, if $V _ { t } < Y _ { t } < X _ { t } .$ then $u _ { t } ( X _ { t } , Y _ { t } ) = u _ { t } ^ { A } ( X _ { t } ) ^ { \cdot } + u _ { t } ^ { B } ( Y _ { t } ) ^ { \cdot } = ( M _ { t } - Y _ { t } ) ^ { \cdot } = u _ { t } ( Y _ { t } , X _ { t } ) ^ { \cdot } + ( X _ { t } - Y _ { t } ) _ { + }$ and a similar reasoning can be applied to the other cases.

In the cases where $V _ { t } \in \{ A _ { t } , B _ { t } \} , { \mathrm { i f ~ } } V _ { t } = X _ { t }$ and $X _ { t } > Y _ { t }$ , then $u _ { t } ( Y _ { t } , X _ { t } ) = \left( Y _ { t } - X _ { t } \right)$ and $u _ { t } ( X _ { t } , Y _ { t } ) = 0 .$ , so the statement still holds. Lastly, if $V _ { t } = Y _ { t }$ and $X _ { t } > Y _ { t } , u _ { t } ( Y _ { t } , X _ { t } ) = ( M _ { t } - X _ { t } )$ and $u _ { t } ( X _ { t } , Y _ { t } ) = \left( M _ { t } - Y _ { t } \right)$ by which we conclude the first part of the proof.

By observing that $( X _ { t } - Y _ { t } ) _ { + } \geq 0$ , we take the expected value of the previously proved equality, yielding $U _ { t } ( A _ { t } , B _ { t } ) \geq U _ { t } ( Y _ { t } , X _ { t } )$ . Finally, by taking the empirical mean over the sequence $\{ M _ { t } \} _ { t = 1 } ^ { T }$ , we get $U ( A _ { t } , B _ { t } ) \geq U ( Y _ { t } , X _ { t } )$ □

We now turn to the intermediate Lemma 3.2, which justifies the reduction to a finite set of actions in Algorithm 2, by showing explicitly that there exists an almost optimal point on the grid, even though the utility is not regular.

Lemma A.2. There exists a point $( A ^ { * } , B ^ { * } ) \in \mathcal { G } \times \mathcal { G }$ such that:

$$
0 \le \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } U ( a , b ) - U ( A ^ { * } , B ^ { * } ) \le \frac { 5 } { 2 K } .
$$

Proof. We define a point on $( A ^ { \ast } , B ^ { \ast } ) \in \mathcal { G } \times \mathcal { G }$ and then prove that such point has utility $2 / K$ close to the optimal action $( a ^ { \ast } , b ^ { \ast } ) \in \mathcal { U }$ . While the supremum is not necessarily attained for the utility, we select $\textstyle 0 < \varepsilon < { \frac { 1 } { 2 K } }$ and a point such that $U ( a ^ { * } , b ^ { * } ) \geq \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } U ( a , b ) - \varepsilon$

Let us first notice that we can reduce to the case where $b ^ { * } \leq \mu _ { T } \leq a ^ { * }$ . This is because of the structure of our utility function: if $a ^ { * } < \mu _ { T }$ , we have $( a ^ { * } - \mu _ { T } ) ( 1 - F ( a ^ { * } ) ) \leq 0 ,$ so we would weakly improve our revenue by substituting $a ^ { * }$ with $\mu _ { T } \mathbf { : }$ : analogously, we can argue that $b ^ { * } \leq \mu _ { T }$ We select the point $( A ^ { * } , B ^ { * } )$ following the same order, using the same logic.

Let $( A ^ { * } , B ^ { * } ) \in \mathcal { G } \times \mathcal { G }$ be defined as:

$$
B ^ { * } = \left\{ \begin{array} { l l } { \displaystyle \frac { \left[ K b ^ { * } \right] } { K } , } & { \mathrm { i f } \ \frac { \left[ K b ^ { * } \right] } { K } \le \mu _ { T } , } \\ { \displaystyle \frac { \left[ K \mu _ { T } \right] } { K } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. \qquad A ^ { * } = \left\{ \begin{array} { l l } { \displaystyle \frac { \left\lfloor K a ^ { * } \right\rfloor } { K } , } & { \mathrm { i f } \ \frac { \left\lfloor K a ^ { * } \right\rfloor } { K } \ge \mu _ { T } , } \\ { \displaystyle \left[ K \mu _ { T } \right] } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{5}
$$

We decouple the contributions from the ask and the bid components, only studying the bid case, as the other follows from a similar argument: we then we sum up the additional contribution of ε to measure the diference with the supremum. Suppose first that the first case holds and $\begin{array} { r } { B ^ { * } = \frac { \lceil K b ^ { * } \rceil } { K } \leq \mu _ { T } } \end{array}$ . Since F is non decreasing and bounded in $[ 0 , 1 ] , F ( B ^ { * } ) \geq F ( b ^ { * } )$ . Therefore, $( \mu _ { T } - B ^ { * } ) F ( B ^ { * } ) \ { \stackrel { \cdot \cdot } { \geq } } \ ( \mu _ { T } - B ^ { * } ) F ( b ^ { * } ) =$ $\begin{array} { r } { ( \mu _ { T } - b ^ { * } ) F ( b ^ { * } ) - ( B ^ { * } - b ^ { * } ) F ( b ^ { * } ) \geq ( \mu _ { T } - b ^ { * } ) F ( b ^ { * } ) - \frac { 1 } { K } } \end{array}$

Suppose instead that $\begin{array} { r } { B ^ { * } = \frac { \lfloor K \mu _ { T } \rfloor } { K } } \end{array}$ . In this case the first rounding point lies above $\mu _ { T }$ , which implies $0 \leq$ $\textstyle \mu _ { T } - b ^ { * } < { \frac { 1 } { K } }$ . Since $F ( b ^ { * } ) \leq 1 , 0 \leq ( \mu _ { T } - b ^ { * } ) F ( b ^ { * } ) < \frac { 1 } { K }$ . On the other hand, $B ^ { * } \leq \mu _ { T } , \mathrm { s o } \left( \mu _ { T } - B ^ { * } \right) F ( B ^ { * } ) \geq 0 .$ Thus again $\begin{array} { r } { ( \mu _ { T } - b ^ { * } ) F ( b ^ { * } ) - ( \mu _ { T } - B ^ { * } ) F ( B ^ { * } ) \le \frac { 1 } { K } } \end{array}$

Hence, in all cases, $\begin{array} { r } { ( \mu _ { T } - b ^ { * } ) F ( b ^ { * } ) - ( \mu _ { T } - B ^ { * } ) F ( B ^ { * } ) \le \frac { 1 } { K } } \end{array}$ , which concludes the argument, by summing up the two contributions from bid and ask prices. We conclude the argument by observing that:

$$
\operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } U ( a , b ) - U ( A ^ { * } , B ^ { * } ) = \operatorname* { s u p } _ { ( a , b ) \in \mathcal { U } } U ( a , b ) - U ( a ^ { * } , b ^ { * } ) + U ( a ^ { * } , b ^ { * } ) - U ( A ^ { * } , B ^ { * } ) \leq \frac { 5 } { 2 K } .
$$

This concludes the proof.

After proving that there exists an almost-optimal point with respect to K on the grid $\mathcal { G } \times \mathcal { G }$ , we finally turn to proving our regret bound.

Theorem 3.3. Let $\delta \in ( 0 , 1 )$ and $T \geq 2$ . Algorithm 2 with $\begin{array} { r } { K = \lceil \sqrt { T } \rceil , \eta = \sqrt { \frac { \log ( T ) } { T } } } \end{array}$ attains regret:

$$
R _ { T } \leq \mathcal { O } \left( \sqrt { T \log ( T ) } + \sqrt { T \log ( 1 / \delta ) } + \log ( 1 / \delta ) \right) ,
$$

with probability at least $1 - \delta$

Proof. Using Lemma 3.2 and the regret decomposition in Equation (1), we have:

$$
R _ { T } \leq \frac { 5 T } { 2 K } + \operatorname* { m a x } _ { ( A ^ { * } , B ^ { * } ) \in { \mathcal { G } } \times { \mathcal { G } } } \sum _ { t = 1 } ^ { T } \left( U _ { t } ( A ^ { * } , B ^ { * } ) - U _ { t } ( A _ { t } , B _ { t } ) \right) .
$$

We first bound the diference between the expected utility for a fixed action $( A ^ { * } , B ^ { * } )$ and the corresponding observed utility. We observe that:

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } ( U _ { t } ( A ^ { * } , B ^ { * } ) - u _ { t } ( A ^ { * } , B ^ { * } ) ) } \\ { \displaystyle \qquad = \displaystyle \sum _ { t = 1 } ^ { T } [ ( M _ { t } - B ^ { * } ) ( F ( B ^ { * } ) - \mathbb { 1 } \{ V _ { t } \leq B ^ { * } \} ) + ( A ^ { * } - M _ { t } ) ( 1 - F ( A ^ { * } ) - \mathbb { 1 } \{ V _ { t } > A ^ { * } \} ) ] . } \end{array}
$$

The two terms of the sum have conditional range $| M _ { t } - B ^ { * } |$ and $\left| A ^ { * } - M _ { t } \right|$ bounded in 1 and form a martingale sum. Thus, using the conditional Azuma-Hoefding inequality [Azuma, 1967] and a union bound for the two terms, we find that with probability $1 - \delta / 3$ the term is bounded by $\scriptstyle { \sqrt { 2 T \log \left( { \frac { 3 ( K + 1 ) } { \delta } } \right) } }$ . As it holds for every action $( A , B ) \in \mathcal { G } \times \mathcal { G }$ , we can select an optimal action in hindsight $( A ^ { * } , B ^ { * } ) \in \operatorname { a r g m a x } _ { ( A , B ) \in { \mathcal { G } } \times { \mathcal { G } } } \sum _ { t = 1 } ^ { T } U _ { t } ( A , B )$ and obtain the same bound. Let us now consider the action $( Y _ { t } , X _ { t } )$ selected at each time step t by the Hedge algorithms. By Azuma-Hoefding inequality [Azuma, 1967] we have that with probability $1 - \delta / 3$

$$
\sum _ { t = 1 } ^ { T } \left( u _ { t } ( Y _ { t } , X _ { t } ) - U _ { t } ( Y _ { t } , X _ { t } ) \right) \leq \sqrt { 2 T \log \left( \frac { 3 } { \delta } \right) } .
$$

We can rewrite the regret on the grid actions as:

$$
\sum _ { t = 1 } ^ { T } \Bigl ( U _ { t } ( A ^ { * } , B ^ { * } ) - U _ { t } ( A _ { t } , B _ { t } ) \Bigr ) \le \sum _ { t = 1 } ^ { T } \Bigl ( U _ { t } ( A ^ { * } , B ^ { * } ) - U _ { t } ( Y _ { t } , X _ { t } ) \Bigr )
$$

$$
= \sum _ { t = 1 } ^ { T } \Big [ u _ { t } ( A ^ { * } , B ^ { * } ) - u _ { t } ( Y _ { t } , X _ { t } ) \Big ] + \sum _ { t = 1 } ^ { T } \Big [ U _ { t } ( A ^ { * } , B ^ { * } ) - u _ { t } ( A ^ { * } , B ^ { * } ) \Big ] + \sum _ { t = 1 } ^ { T } \Big [ u _ { t } ( Y _ { t } , X _ { t } ) - U _ { t } ( Y _ { t } , X _ { t } ) \Big ] ,
$$

where the inequality holds by applying Lemma 3.1, while the equality holds because the utility can be decomposed into separate components. We notice here that we use in the proof the points $\{ ( X _ { t } , Y _ { t } ) \} _ { t = 1 } ^ { T }$ selected in the original instances of Hedge for the bid and ask, so that the standard concentration argument can be applied, for example using Luo and Schapire [2014]. Thus, observing that both instances of Hedge have the same learning rate $\eta > 0$ , we have that with probability $1 - \delta / 3$ , using the decomposition into two separate problems:

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \Big [ u _ { t } ( \boldsymbol { A } ^ { * } , \boldsymbol { B } ^ { * } ) - u _ { t } ( Y _ { t } , X _ { t } ) \Big ] = \sum _ { t = 1 } ^ { T } \Big [ u _ { t } ^ { A } ( \boldsymbol { A } ^ { * } ) - u _ { t } ^ { A } ( Y _ { t } ) \Big ] + \sum _ { t = 1 } ^ { T } \Big [ u _ { t } ^ { B } ( \boldsymbol { B } ^ { * } ) - u _ { t } ^ { B } ( X _ { t } ) \Big ] } } \\ & { } & { \leq \frac { 2 \log ( K + 1 ) } { \eta } + \eta T + 2 \sqrt { T \log \Big ( \frac { 3 } { \delta } \Big ) } , } \end{array}
$$

Building on that, by choosing $K = \lceil { \sqrt { T } } \rceil$ and $\eta = { \sqrt { \frac { \log ( T ) } { T } } }$ , we get the second result $R _ { T } = \mathcal { O } ( \sqrt { T \log ( T ) } +$ $\sqrt { T \log ( 1 / \delta ) } + \log ( 1 / \delta ) )$ with probability $1 - \delta$ by summing the failure probabilities, which concludes the argument. □

## B Optimal bounds for LOB

Lemma B.1. Let $( A _ { t } , B _ { t } )$ be a price pair with $B _ { t } \ \leq \ A _ { t }$ , played at time step t. Then, for any point $x \in ( \mathcal G _ { A } \cup \mathcal G _ { B } ) \cap [ B _ { t } , A _ { t } ]$ , it is possible to reconstruct $\ell _ { t } ^ { s } ( x ) , \ s \in \{ A , B \}$ , after playing $( A _ { t } , B _ { t } )$

Proof. If $V _ { t } \in ( B _ { t } , A _ { t } ]$ the statement follows trivially by the fact that we observe $( M _ { t } , V _ { t } )$ entirely.

If a purchase occurred and $x \in { \mathcal { G } } _ { B }$ , then $V _ { t } \leq B _ { t } \leq x \colon$ therefore $u _ { t } ^ { B } ( x ) = M _ { t } - x$ . If instead we consider a point on the other grid $y \in \mathcal { G } _ { A } \cap [ B _ { t } , A _ { t } ]$ , then $V _ { t } \ \leq \ B _ { t } \ \leq \ y$ and $u _ { t } ^ { A } ( y ) = 0 :$ therefore we can again reconstruct the feedback. The same reasoning can be mirrored in case of a sale, thus we conclude the proof. We notice here that we can reconstruct the revenue independently of the $V _ { t }$ distribution, as it only depends on the structure of the utility and the feedback. □

After proving the consistency of our estimators, we now move on to the proof of Lemma 4.2. This lemma presents the first step for bounding in the regret analysis

Lemma B.2. For any time step $t \in \{ 1 , \ldots , T \}$ , with $T \geq 2$ , it holds:

$$
\sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } \frac { w _ { t } ^ { s } ( x ) } { Q _ { t } ( x ) + \beta } \leq 4 \left( \log \left( \frac { 1 + \beta } { \beta } \right) \right) .\tag{4}
$$

Proof. For the sake of this proof, we fix a unique indexing on the union of the two grids $\mathcal { G } _ { A } \cup \mathcal { G } _ { B }$ , with a total of $2 ( K + 1 )$ points, denoted as $z _ { 1 } \le \cdots \le z _ { N }$ . We can therefore call the probability weights with the unique name of w<sub>t</sub> for simplicity of notation, where $w _ { t } ( z _ { i } ) = w _ { t } ^ { B } ( z _ { i } ) \mathrm { ~ i f ~ } z _ { i } \in \mathcal G _ { B }$ and $w _ { t } ( z _ { i } ) = w _ { t } ^ { A } ( z _ { i } ) \mathrm { ~ i f ~ } z _ { i } \in \mathcal { G } _ { A }$

Let $\begin{array} { r } { S _ { t } ( z _ { i } ) ~ = ~ \frac { 1 } { 2 } \sum _ { i < i } w _ { t } ( z _ { j } ) } \end{array}$ be the normalized sum of the probabilities in each point on the grid until point $z _ { i } \in \mathcal { G } _ { A } \cup \mathcal { G } _ { B } ^ { - } .$ Recall Equation (3): we observe that, selecting a point $z _ { i } \in \mathcal { G } _ { A } \cup \mathcal { G } _ { B }$ , it holds that $\begin{array} { r } { \sum _ { j < i } w _ { t } ( z _ { j } ) = 2 S _ { t } ( z _ { i - 1 } ) } \end{array}$ and $\begin{array} { r } { \sum _ { j > i } w _ { t } ( z _ { j } ) = 2 ( 1 - S _ { t } ( z _ { i } ) ) } \end{array}$ with our new notation. Thus we have that, from Identity (3):

$$
\begin{array} { r l } & { Q _ { t } ( z _ { i } ) \geq 1 - S _ { t } ^ { 2 } ( z _ { i - 1 } ) - ( 1 - S _ { t } ( z _ { i } ) ) ^ { 2 } } \\ & { \qquad = 2 S _ { t } ( z _ { i } ) ( 1 - S _ { t } ( z _ { i - 1 } ) ) - \bigg ( \frac { w _ { t } ( z _ { i } ) } { 2 } \bigg ) ^ { 2 } } \end{array}
$$

$$
\geq S _ { t } ( z _ { i } ) ( 1 - S _ { t } ( z _ { i - 1 } ) ) .
$$

By using the standard analysis inequality $a b \leq ( a + b ) ^ { 2 } / 4$ and the fact that:

$$
S _ { t } ( z _ { i } ) \left( 1 - S _ { t } ( z _ { i - 1 } ) \right) \geq \left( \frac { w _ { t } ( z _ { i } ) } { 2 } \right) ^ { 2 } ,
$$

since $\begin{array} { r } { \left( \frac { w _ { t } ( z _ { i } ) } { 2 } \right) = S _ { t } ( z _ { i } ) - S _ { t } ( z _ { i - 1 } ) } \end{array}$ is no larger than both $S _ { t } ( z _ { i } )$ and $1 - S _ { t } ( z _ { i - 1 } )$

We use these inequalities to construct our bound on the importance sampling term:

$$
\begin{array} { l } { \displaystyle \sum _ { i = 1 } ^ { N } \displaystyle \frac { w _ { t } ( z _ { i } ) } { Q _ { t } ( z _ { i } ) + \beta } \leq \sum _ { i = 1 } ^ { N } \displaystyle \frac { w _ { t } ( z _ { i } ) } { S _ { t } ( z _ { i } ) ( 1 - S _ { t } ( z _ { i - 1 } ) ) + \beta } } \\ { \displaystyle \leq \sum _ { i = 1 } ^ { N } \displaystyle \frac { w _ { t } ( z _ { i } ) } { S _ { t } ( z _ { i } ) + \beta } + \sum _ { i = 1 } ^ { N } \displaystyle \frac { w _ { t } ( z _ { i } ) } { 1 - S _ { t } ( z _ { i - 1 } ) + \beta } } \\ { \displaystyle = 2 \sum _ { i = 1 } ^ { N } \displaystyle \frac { S _ { t } ( z _ { i } ) - S _ { t } ( z _ { i - 1 } ) } { S _ { t } ( z _ { i } ) + \beta } + 2 \sum _ { i = 1 } ^ { N } \displaystyle \frac { S _ { t } ( z _ { i } ) - S _ { t } ( z _ { i - 1 } ) } { 1 - S _ { t } ( z _ { i - 1 } ) + \beta } } \\ { \displaystyle \leq 4 \int _ { 0 } ^ { 1 } \displaystyle \frac { d S } { S + \beta } = 4 \left( \log \left( \displaystyle \frac { 1 + \beta } { \beta } \right) \right) . } \end{array}
$$

Which holds by applying definitions and using that $1 / ( a b ) \leq 1 / a + 1 / b$ with $a + b \geq 1$ and standard inequalities from analysis.

We define for the following proofs the factor in Equation (4) as:

$$
D : = 4 \left( \log \left( \frac { 1 + \beta } { \beta } \right) \right) ,
$$

to simplify later derivations.

Before proving Theorem 4.3, let us state some intermediate steps Lemma B.3, Lemma B.4 and Lemma B $. 5 ,$ providing high probability bounds on the conditional mean and variance of the estimated losses at each time step, given the data from the previous rounds $\mathcal { H } _ { t - 1 } = \{ ( A _ { 1 } , B _ { 1 } ) , ( V _ { 1 } , M _ { 1 } ) , \dots , ( V _ { t - 1 } , M _ { t - 1 } ) \}$

In the following Lemma B.3, we bound the cumulative diference between the estimated loss of the two current algorithms, denoted as $Z _ { t } .$ , and that of any fixed bid–ask pair $( a , b ) \in \mathcal { G } _ { A } \times \mathcal { G } _ { B }$ , up to the usual logarithmic and second-order terms, denoted as $Y _ { t }$

Lemma B.3. Let $Z _ { t } , \ Y _ { t }$ be defined as:

$$
Z _ { t } = \sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } w _ { t } ^ { s } ( x ) \hat { \ell } _ { t } ^ { s } ( x ) , \qquad Y _ { t } = \sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } w _ { t } ^ { s } ( x ) ( \hat { \ell } _ { t } ^ { s } ( x ) ) ^ { 2 } .
$$

Then we have that, for any $b \in \mathcal { G } _ { B } , a \in \mathcal { G } _ { A }$

$$
\sum _ { t = 1 } ^ { T } \Big ( Z _ { t } - \Big ( \hat { \ell } _ { t } ^ { B } ( b ) + \hat { \ell } _ { t } ^ { A } ( a ) \Big ) \Big ) \leq \frac { 2 \log ( K + 1 ) } { \eta } + \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } Y _ { t } .
$$

Proof. The proof follows by manipulation of the weights $W _ { t } ^ { s }$ . Fix a time step $t \leq T$ and let $s \in \{ A , B \}$ First, consider that:

$$
\frac { W _ { t + 1 } ^ { s } ( x ) } { \sum _ { y \in \mathcal { G } _ { s } } W _ { t } ^ { s } ( y ) } = \frac { W _ { t } ^ { s } ( x ) } { \sum _ { y \in \mathcal { G } _ { s } } W _ { t } ^ { s } ( y ) } \exp \Bigl ( - \eta \hat { \ell } _ { t } ^ { s } ( x ) \Bigr ) = w _ { t } ^ { s } ( x ) \exp \Bigl ( - \eta \hat { \ell } _ { t } ^ { s } ( x ) \Bigr ) .
$$

Therefore, if we sum the equality above over the grid points $x \in \mathcal G _ { s }$ , we have:

$$
\begin{array} { r l r } {  { \frac { \sum _ { x \in \mathcal { G } _ { s } } W _ { t + 1 } ^ { s } ( x ) } { \sum _ { y \in \mathcal { G } _ { s } } W _ { t } ^ { s } ( y ) } = \sum _ { x \in \mathcal { G } _ { s } } w _ { t } ^ { s } ( x ) \exp \Bigl ( - \eta \hat { \ell } _ { t } ^ { s } ( x ) \Bigr ) } } \\ & { } & { \leq \sum _ { x \in \mathcal { G } _ { s } } w _ { t } ^ { s } ( x ) ( 1 - \eta \hat { \ell } _ { t } ^ { s } ( x ) + \frac { \eta ^ { 2 } } { 2 } ( \hat { \ell } _ { t } ^ { s } ( x ) ) ^ { 2 } ) } \\ & { } & { = 1 - \eta \sum _ { x \in \mathcal { G } _ { s } } w _ { t } ^ { s } ( x ) \hat { \ell } _ { t } ^ { s } ( x ) + \frac { \eta ^ { 2 } } { 2 } \sum _ { x \in \mathcal { G } _ { s } } w _ { t } ^ { s } ( x ) ( \hat { \ell } _ { t } ^ { s } ( x ) ) ^ { 2 } , } \end{array}
$$

where we used that $\textstyle \exp ( - x ) \leq 1 - x + { \frac { x ^ { 2 } } { 2 } }$ . After taking the logarithm on both sides and employing that log $( 1 + x ) \leq x \colon$

$$
\log \left( \frac { \sum _ { x \in \mathcal { G } _ { s } } W _ { t + 1 } ^ { s } ( x ) } { \sum _ { y \in \mathcal { G } _ { s } } W _ { t } ^ { s } ( y ) } \right) \leq - \eta \sum _ { x \in \mathcal { G } _ { s } } w _ { t } ^ { s } ( x ) \hat { \ell } _ { t } ^ { s } ( x ) + \frac { \eta ^ { 2 } } { 2 } \sum _ { x \in \mathcal { G } _ { s } } w _ { t } ^ { s } ( x ) ( \hat { \ell } _ { t } ^ { s } ( x ) ) ^ { 2 } ,
$$

and by summing over the time steps $t = 1 , \dots , T$ and $s \in \{ A , B \}$ , and telescoping, we deduce:

$$
\log \left( \sum _ { y \in \mathcal { G } _ { A } } W _ { T + 1 } ^ { A } ( y ) \right) + \log \left( \sum _ { y \in \mathcal { G } _ { B } } W _ { T + 1 } ^ { B } ( y ) \right) - 2 \log ( K + 1 ) \leq \sum _ { t = 1 } ^ { T } \left( - \eta Z _ { t } + \frac { \eta ^ { 2 } } { 2 } Y _ { t } \right) .
$$

where we used $\textstyle \sum _ { y \in { \mathcal { G } } _ { s } } W _ { 1 } ^ { s } ( y ) = K + 1 , \ s \in \{ A , B \}$ and the definitions of $Z _ { t }$ and $Y _ { t }$ . We conclude the argument observing that, fixing an $s \in \{ A , B \}$ and $\begin{array} { r } { x \in \mathcal { G } _ { s } , \sum _ { y \in \mathcal { G } _ { s } } W _ { T + 1 } ^ { s } ( y ) \geq W _ { T + 1 } ^ { s } ( x ) = \exp \left( - \eta \sum _ { t = 1 } ^ { T } \hat { \ell } _ { t } ^ { s } ( x ) \right) } \end{array}$ , since it represents the last update, and substituting this inequality into the last step of the previous chain of inequalities. □

We now provide a high probability bound on the diference between the estimated losses $\hat { \ell }$ and the real ℓ.

Lemma B.4. Consider the realized mean loss a time t and point $x \in { \mathcal { G } } _ { B }$ , defined as $\begin{array} { r } { \bar { \ell } _ { t } ^ { B } ( x ) : = \frac { 1 - ( M _ { t } - x ) F ( x ) } { 2 } } \end{array}$ and for $y \in { \mathcal { G } } _ { A }$ as $\begin{array} { r } { \bar { \ell } _ { t } ^ { A } ( y ) : = \frac { 1 - ( y - M _ { t } ) ( 1 - F ( y ) ) } { 2 } } \end{array}$

For any $x \in \mathcal G _ { s }$ , with $s \in \{ A , B \}$ and for all $u > 0$ it holds that:

$$
\mathbb { P } \left( \sum _ { { t } = 1 } ^ { T } \left( \hat { \ell } _ { t } ^ { s } ( x ) - \bar { \ell } _ { t } ^ { s } ( x ) \right) \geq \frac { u } { \beta } \right) \leq \exp ( - u ) .
$$

Hence, setting $\delta \in ( 0 , 1 )$ , with probability $1 - \delta / 3$ we have that for any $z \in \mathcal { G } _ { s }$

$$
\sum _ { t = 1 } ^ { T } \Big ( \hat { \ell } _ { t } ^ { s } ( x ) - \bar { \ell } _ { t } ^ { s } ( x ) \Big ) \leq \frac { 1 } { \beta } \log \left( \frac { 6 ( K + 1 ) } { \delta } \right) .
$$

Proof. Let $x \in \mathcal G _ { s }$ and fix a time step t. We observe that $O _ { t } ( x )$ is distributed as a Bernoulli with parameter $Q _ { t } ( x )$ , given the history $\mathcal { H } _ { t - 1 }$ and loss $\ell _ { t } ^ { s } ( x )$ . By taking the exponential of the estimator $\hat { \ell } _ { t } ^ { s } ( x )$ , we can therefore bound:

$$
\begin{array} { r l } & { \mathbb { E } _ { O _ { t } ( x ) } \left[ \exp \left( \displaystyle \frac { \beta O _ { t } ( x ) \ell _ { t } ^ { s } ( x ) } { Q _ { t } ( x ) + \beta } \right) \mid \mathcal { H } _ { t - 1 } , \ell _ { t } ^ { s } ( x ) \right] = 1 - Q _ { t } ( x ) + Q _ { t } ( x ) \exp \left( \displaystyle \frac { \beta \ell _ { t } ^ { s } ( x ) } { Q _ { t } ( x ) + \beta } \right) } \\ & { \qquad \leq 1 + \displaystyle \frac { \beta Q _ { t } ( x ) \ell _ { t } ^ { s } ( x ) } { Q _ { t } ( x ) + \beta - \beta \ell _ { t } ^ { s } ( x ) } } \\ & { \qquad \leq 1 + \beta \ell _ { t } ^ { s } ( x ) . } \end{array}
$$

Where we used that, for $\begin{array} { r } { x \in [ 0 , 1 ) , \exp ( x ) \leq 1 + \frac { x } { 1 - x } } \end{array}$ , then that $\begin{array} { r } { \frac { \displaystyle \mathrm { , } Q _ { t } ( x ) } { \displaystyle Q _ { t } ( x ) + \beta - \beta \ell _ { t } ^ { s } ( x ) } \le 1 } \end{array}$

By the tower rule on $\ell _ { t } ^ { s } ( z )$ , we can rewrite the expected value $\mathbb { E } \left[ \exp \left( \hat { \ell } _ { t } ^ { s } ( x ) \right) \mid \mathcal { H } _ { t - 1 } \right]$

$$
\begin{array} { r l r } & { } & { \mathbb { E } \left[ \mathbb { E } _ { O _ { t } ( x ) } \left[ \exp \left( \beta \hat { \ell } _ { t } ^ { s } ( x ) \right) \bigm | \mathcal { H } _ { t - 1 } , \ell _ { t } ^ { s } ( x ) \right] \bigm | \mathcal { H } _ { t - 1 } \right] \leq 1 + \beta \bar { \ell } _ { t } ^ { s } ( x ) } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \leq \exp \left( \beta \bar { \ell } _ { t } ^ { s } ( x ) \right) , } \end{array}
$$

where the last inequality follows by $1 + x \leq \exp ( x )$

Hence, the process exp $\begin{array} { r } { \left( \beta \sum _ { \tau = 1 } ^ { t } \Big ( \hat { \ell } _ { \tau } ^ { s } ( x ) - \bar { \ell } _ { \tau } ^ { s } ( x ) \Big ) \right) } \end{array}$ is a nonnegative supermartingale that has an initial value of 1.

The first part of the thesis is deduced by applying Markov’s inequality to the diference of the sums $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \Big ( \hat { \ell } _ { t } ^ { s } ( x ) - \bar { \ell } _ { t } ^ { s } ( x ) \Big ) } \end{array}$ and applying the deduced inequality in terms of the exponential. That is,

$$
\begin{array} { r l } & { \mathbb { P } \left( \displaystyle \sum _ { t = 1 } ^ { T } \left( \hat { \ell } _ { t } ^ { s } ( x ) - \bar { \ell } _ { t } ^ { s } ( x ) \right) \geq \frac { u } { \beta } \right) = } \\ & { \displaystyle = \mathbb { P } \left( \exp \left( \beta \displaystyle \sum _ { t = 1 } ^ { T } \left( \hat { \ell } _ { t } ^ { s } ( x ) - \bar { \ell } _ { t } ^ { s } ( x ) \right) \right) \geq \exp ( u ) \right) \leq \frac { 1 } { \exp ( u ) } . } \end{array}
$$

The second part of the claim is obtained building on the first, by a union bound on the $2 ( K + 1 )$ total grid points and where we select $\begin{array} { r } { u = \log \left( \frac { 6 ( K + 1 ) } { \delta } \right) } \end{array}$ □

Lemma B.5. Consider $\eta \in ( 0 , \beta ]$ and recall the definition of $Z _ { t }$ and $Y _ { t }$ given in Lemma B.3. Then, it holds that $\begin{array} { r } { 0 \le Z _ { t } - \frac { \eta } { 2 } Y _ { t } \le D } \end{array}$ , and $\begin{array} { r } { \mathbb { E } [ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \mid \mathcal { H } _ { t - 1 } ] \leq 2 } \end{array}$ . Moreover:

$$
\sum _ { \substack { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } } w _ { t } ^ { s } ( x ) \bar { \ell } _ { t } ^ { s } ( x ) - \mathbb { E } \left[ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \mid \mathcal { H } _ { t - 1 } \right] \leq \left( \beta + \frac { \eta } { 2 } \right) D .
$$

Furthermore, for any $u > 0 .$

$$
\mathbb { P } \left( \sum _ { t = 1 } ^ { T } \left( \mathbb { E } \left[ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \mid \mathcal { H } _ { t - 1 } \right] - \left( Z _ { t } - \frac { \eta } { 2 } Y _ { t } \right) \right) > 2 \sqrt { D T u } \right) \le \exp ( - u ) ,
$$

and for every $x \in { \mathcal { G } } _ { B }$ and $y \in { \mathcal { G } } _ { A }$ it holds that:

$$
\sum _ { t = 1 } ^ { T } \Big ( Z _ { t } - \frac { \eta } { 2 } Y _ { t } \Big ) - \sum _ { t = 1 } ^ { T } \Big ( \hat { \ell } _ { t } ^ { B } ( x ) + \hat { \ell } _ { t } ^ { A } ( y ) \Big ) \leq \frac { 2 \log ( K + 1 ) } { \eta } .
$$

Proof. We first notice that:

$$
\mathbb { E } [ \bar { \ell } _ { t } ^ { A } ( A _ { t } ) + \bar { \ell } _ { t } ^ { B } ( B _ { t } ) \mid \mathcal { H } _ { t - 1 } ] \le \mathbb { E } [ \bar { \ell } _ { t } ^ { A } ( Y _ { t } ) + \bar { \ell } _ { t } ^ { B } ( X _ { t } ) \mid \mathcal { H } _ { t - 1 } ] = \sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } w _ { t } ^ { s } ( x ) \bar { \ell } _ { t } ^ { s } ( x ) .
$$

The inequality derives from the fact that we might swap the prices $( Y _ { t } , X _ { t } )$ initially selected by Algorithm 3 using Lemma 3.1, for which we weakly decrease the loss if in the step t we had swapped the prices.

We consider the expected value for the losses estimates. It holds, by definition:

$$
\begin{array} { r l } & { \mathbb { E } [ \widehat \ell _ { t } ^ { s } ( x ) \mid \mathcal { H } _ { t - 1 } ] = \displaystyle \frac { Q _ { t } ( x ) } { Q _ { t } ( x ) + \beta } \bar { \ell } _ { t } ^ { s } ( x ) , } \\ & { \mathbb { E } [ ( \widehat \ell _ { t } ^ { s } ( x ) ) ^ { 2 } \mid \mathcal { H } _ { t - 1 } ] = \displaystyle \frac { Q _ { t } ( x ) } { ( Q _ { t } ( x ) + \beta ) ^ { 2 } } \mathbb { E } [ \ell _ { t } ^ { s } ( x ) ^ { 2 } \mid \mathcal { H } _ { t - 1 } ] \leq \displaystyle \frac { 1 } { Q _ { t } ( x ) + \beta } . } \end{array}
$$

Thus, follows that $\begin{array} { r } { \sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } w _ { t } ^ { s } ( x ) \bar { \ell } _ { t } ^ { s } ( x ) - \mathbb { E } [ Z _ { t } \mid \mathcal { H } _ { t - 1 } ] \leq \beta D } \end{array}$ and $\mathbb { E } [ Y _ { t } \mid \mathcal { H } _ { t - 1 } ] \leq D$ , from Lemma 4.2.

Notice that, for any time step t and $\begin{array} { r } { x \in \mathcal { G } _ { s } , \hat { \ell } _ { t } ^ { s } ( x ) \in [ 0 , \frac { 1 } { \beta } ] \colon } \end{array}$ : so $\begin{array} { r } { 0 \leq \hat { \ell } _ { t } ^ { s } ( x ) \left( 1 - \frac { \eta } { 2 } \hat { \ell } _ { t } ^ { s } ( x ) \right) \leq \hat { \ell } _ { t } ^ { s } ( x ) } \end{array}$ . Therefore, by recalling Lemma 4.2, we have that $\begin{array} { r } { 0 \leq Z _ { t } - \frac { \eta } { 2 } Y _ { t } \leq Z _ { t } \leq D } \end{array}$ . By taking the expectation:

$$
\mathbb { E } \left[ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \mid \mathcal { H } _ { t - 1 } \right] \leq \mathbb { E } \left[ Z _ { t } \mid \mathcal { H } _ { t - 1 } \right] \leq \sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } w _ { t } ^ { s } ( x ) \bar { \ell } _ { t } ^ { s } ( x ) \leq 2 .
$$

We deduce the second claim from Lemma B.3. We also notice that $\begin{array} { r } { \mathbb { E } \big [ \big ( Z _ { t } - \frac { \eta } { 2 } Y _ { t } \big ) ^ { 2 } \ \big | \ \mathcal { H } _ { t - 1 } \big ] \leq D \mathbb { E } \big [ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \ \big | } \end{array}$ $\mathcal { H } _ { t - 1 } ] \le 2 D$

Therefore, $\forall \lambda \geq 0$ and using $\textstyle \exp ( - x ) \leq 1 - x + { \frac { x ^ { 2 } } { 2 } }$ and $1 + x \leq \exp ( x )$ , we have that:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \exp \left( \lambda \left( \mathbb { E } \left[ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \mid \mathcal { H } _ { t - 1 } \right] - \left( Z _ { t } - \frac { \eta } { 2 } Y _ { t } \right) \right) \right) \mid \mathcal { H } _ { t - 1 } \right] } \\ & { \leq \exp \left( \lambda \mathbb { E } \left[ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \mid \mathcal { H } _ { t - 1 } \right] \right) \left( 1 - \lambda \mathbb { E } \left[ Z _ { t } - \frac { \eta } { 2 } Y _ { t } \mid \mathcal { H } _ { t - 1 } \right] + \frac { \lambda ^ { 2 } } { 2 } \mathbb { E } \left[ \left( Z _ { t } - \frac { \eta } { 2 } Y _ { t } \right) ^ { 2 } \mid \mathcal { H } _ { t - 1 } \right] \right) } \\ & { \leq \exp \left( \frac { \lambda ^ { 2 } } { 2 } \mathbb { E } \left[ \left( Z _ { t } - \frac { \eta } { 2 } Y _ { t } \right) ^ { 2 } \mid \mathcal { H } _ { t - 1 } \right] \right) } \\ & { \leq \exp \left( \lambda ^ { 2 } D \right) . } \end{array}
$$

By summing over the $t ,$ as the sequence is a supermartingale with expectation bounded by 1 and applying Markov’s inequality with $\lambda = \sqrt { u / T D }$ we deduce the fourth claim, while the last claim is a reordering of Lemma B.3. □

We finally prove the full Theorem 4.3 statement, using the results from the previous Lemma.

Theorem 4.3. Let $\delta \in ( 0 , 1 )$ and $T \geq 2$ $B y$ selecting $\eta = \beta = \sqrt { 1 / T }$ , Algorithm $\mathcal { B }$ instantiated with $K = \lceil { \sqrt { T } } \rceil$ attains:

$$
R _ { T } = \mathcal { O } \left( \sqrt { T } \log \left( \frac { T } { \delta } \right) \right) ,
$$

with probability at least $1 - \delta .$

Proof. Recall $Z _ { t } , Y _ { t }$ as defined in Lemma B.3 and the observations regarding their conditional expected value in Lemma B.5. We first observe that by conditional Hoefding’s bound for all time steps t:

$$
{ \mathbb E } \left[ \exp \left( \lambda ( \bar { \ell } _ { t } ^ { A } ( Y _ { t } ) + \bar { \ell } _ { t } ^ { B } ( X _ { t } ) - \sum _ { x \in { \mathcal G } _ { s } , s \in \{ A , B \} } w _ { t } ^ { s } ( x ) \bar { \ell } _ { t } ^ { s } ( x ) \right) | \mathcal { H } _ { t - 1 } \right] \leq \exp ( \lambda ^ { 2 } / 2 ) .
$$

Then with a union bound and Markov inequality:

$$
\mathbb { P } \left( \sum _ { t = 1 } ^ { T } \left( \bar { \ell } _ { t } ^ { A } ( Y _ { t } ) + \bar { \ell } _ { t } ^ { B } ( X _ { t } ) - \sum _ { x \in \mathcal { G } _ { s } , s \in \{ A , B \} } w _ { t } ^ { s } ( x ) \bar { \ell } _ { t } ^ { s } ( x ) \right) \geq \sqrt { 2 T u } \right) \leq \exp ( - u ) .
$$

To get the bound, we use the results in the previous Lemma B.3, Lemma B.4 and Lemma B.5 by adding and subtracting the terms defined there.

We consider as a comparator action $( a , b ) \in \mathcal { G } _ { A } \times \mathcal { G } _ { B }$ . This is justified as we notice that a similar construction to the one in Lemma 3.2 repeated in the case of $\mathcal { G } _ { A } \times \mathcal { G } _ { B }$ establishes that we can reduce the problem to our discrete set at the total cost of $\frac { 5 T } { K }$ , accounting for the additional constants in the loss.

In particular fix any pair $( a , b ) \in \mathcal { G } _ { A } \times \mathcal { G } _ { B } \colon$ with probability at least $1 - \delta ,$ which derives from summing up the failure probabilities from the previous lemmas:

$$
\left[ \sum _ { t = 1 } ^ { T } \biggl ( \bar { \ell } _ { t } ^ { A } ( A _ { t } ) + \bar { \ell } _ { t } ^ { B } ( B _ { t } ) - \bar { \ell } _ { t } ^ { A } ( a ) - \bar { \ell } _ { t } ^ { B } ( b ) \biggr ) \right]
$$

Table 2: Upper bound on $u _ { t } ( A _ { t } , B _ { t } )$ for each case of the random sequence
<table><tr><td> $\sum \mathrm { ~ { ~ ( ~ } } V _ { t } , M _ { t } { ) }$ </td><td rowspan="3"> $( L _ { t - 1 } , 1 )$ </td><td rowspan="3"> $( y , 1 )$ </td><td rowspan="3"> $( x , 0 )$ </td><td rowspan="3"> $( U _ { t - 1 } , 0 )$ </td><td rowspan="3">Upper bound</td></tr><tr><td> $( A _ { t } , B _ { t } ) \frown$ </td><td></td></tr><tr><td> $A _ { t } \in [ L _ { t - 1 } , x ) , B _ { t } \in [ L _ { t - 1 } , x ) ,$ </td><td> $A _ { t } - 1$   $A _ { t }$ </td></tr><tr><td> $A _ { t } \in [ x , y ) , B _ { t } \in [ L _ { t - 1 } , x )$ </td><td> $1 - B _ { t }$   $1 - B _ { t }$ </td><td> $A _ { t } - 1$ </td><td>0</td><td> $A _ { t }$   $A _ { t }$ </td><td>13 1-4</td></tr><tr><td> $A _ { t } \in [ y , U _ { t - 1 } ] , B _ { t } \in [ L _ { t - 1 } , x )$ </td><td> $1 - B _ { t }$ </td><td>0</td><td>0</td><td> $A _ { t }$ </td><td>113</td></tr><tr><td> $A _ { t } \in [ x , y ) , B _ { t } \in [ x , y )$ </td><td> $1 - B _ { t }$ </td><td></td><td> $A _ { t } - 1 \mathrm { ~  ~ { ~ \tau ~ } ~ } - B _ { t }$ </td><td> $A _ { t }$ </td><td>1</td></tr><tr><td> $A _ { t } \in [ y , U _ { t - 1 } ] , B _ { t } \in [ x , y )$ </td><td> $1 - B _ { t }$ </td><td>0</td><td>-Bt</td><td> $A _ { t }$ </td><td>18</td></tr><tr><td> $A _ { t } \in [ y , U _ { t - 1 } ] , B _ { t } \in [ y , U _ { t - 1 } ]$ </td><td></td><td> $1 - B _ { t }$   $1 - B _ { t }$ </td><td> $- B _ { t }$ </td><td> $A _ { t }$ </td><td>1-3 113</td></tr></table>

$$
\begin{array} { r l } & { \leq \displaystyle \sum _ { t = 1 } ^ { T } ( \tau _ { t } ^ { 4 } ( Y _ { t } ) + \hat { r } _ { t } ^ { B _ { t } } ( X _ { t } ) - \displaystyle \sum _ { x \leq t } \displaystyle \sum _ { x \leq t } w _ { t } ^ { \varepsilon } ( x ^ { \varepsilon } ) \hat { r } _ { t } ^ { B _ { t } } ( x ) ) } \\ & { \quad + \displaystyle \sum _ { t = 1 } ^ { T } ( \displaystyle \sum _ { x \leq t } \displaystyle \sum _ { x \leq t } w _ { t } ^ { \varepsilon } ( X _ { t } ) \hat { r } _ { t } ^ { B _ { t } } ( x ^ { \varepsilon } ) - \mathbb { E } [ 2 \nu _ { t } - \frac { \eta } { 2 } Y _ { t } ] W _ { t - 1 } ] ) } \\ & { \quad + \displaystyle \sum _ { t = 1 } ^ { T } ( E [ \xi _ { t } - \frac { \eta } { 2 } Y _ { t } ] \hat { r } _ { t - 1 } \Big \} - \xi _ { t } - \frac { \eta } { 2 } Y _ { t } ) } \\ & { \quad + \displaystyle \sum _ { t = 1 } ^ { T } ( E _ { t } - \frac { \eta } { 2 } Y _ { t } - \hat { r } _ { t } \hat { r } _ { t } ( x ) - \hat { r } _ { t } ^ { B _ { t } } ( y ) + \displaystyle \sum _ { t = 1 } ^ { T } ( \hat { r } _ { t } ^ { A } ( x ) + \hat { r } _ { t } ^ { B _ { t } } ( y ) - \hat { r } _ { t } ^ { A } ( x ) - \hat { r } _ { t } ^ { B _ { t } } ( y ) )  } \\ &  \quad  \leq \displaystyle \sum _ { t = 1 } ^ { T } ( E _ { t } - \frac { \eta } { 2 } Y _ { t } - \hat { r } _ { t } ^ { B _ { t } } ( x ) - \hat { r } _ { t } ^ { B _ { t } } ( y ) ) D + \displaystyle \sum _ { t = 1 } ^ { T } ( \hat { r } _ { t } ^ { B _ { t } } ( x ) + \hat { r } _  t  \end{array}
$$

By applying this bound to the regret in the utility with factor 2 due to the loss rescaling, we obtain the general claim. We also used Lemma 3.1 in the first inequality.

Consequently, by selecting $K = \lceil \sqrt { T } \rceil , \eta = \beta = \sqrt { 1 / T }$ we have $D \propto \log ( T )$ , thus we deduce the optimal rate $R _ { T } \le \tilde { \mathcal { O } } ( \sqrt { T } )$ . In particular, with probability $1 - \delta$ it holds that $\begin{array} { r } { R _ { T } = \mathcal { O } \left( \sqrt { T } \log \left( \frac { T } { \delta } \right) \right) } \end{array}$ , which concludes the proof. □

## C Proof for the impossibility result

In this final section, we give a proof of the unlearnability result for market making, under the favorable condition of full feedback and oblivious adversary.

Theorem 5.1. Consider the market making problem with adversarial sequence of $\{ M _ { t } \}$ <sub>t</sub> and $\{ V _ { t } \} _ { t }$ and full feedback. Then, for any algorithm A, there exists a sequence $\{ ( M _ { t } , V _ { t } ) \} _ { t = 1 } ^ { T }$ where expected regret $\begin{array} { r } { \dot { R } _ { T } \geq \frac { T } { 6 } } \end{array}$

Proof. To prove the statement, we rely on Yao’s minimax principle [Yao, 1977]: we construct a random instance of the problem and prove that any deterministic bid-ask sequence sufers linear regret with such sequence.

Consider the following construction for a sequence bounded in $[ 0 , 1 ] \ ( L _ { t } , U _ { t } ) _ { t }$ . Let $D _ { t } = U _ { t } - L _ { t }$ at any given

time step t. We initialize $( L _ { 0 } , U _ { 0 } ) = ( 1 / 3 , 2 / 3 )$ , then for any $t = 1 , \dots , T \colon$

$$
\left\{ \begin{array} { l l } { ( L _ { t } , U _ { t } ) = ( L _ { t - 1 } , U _ { t - 1 } - \frac { 2 D _ { t - 1 } } { 3 } ) } & { \quad \mathrm { w i t h ~ p r o b a b i l i t y } \quad \frac { 1 } { 2 } } \\ { ( L _ { t } , U _ { t } ) = ( L _ { t - 1 } + \frac { 2 D _ { t - 1 } } { 3 } , U _ { t - 1 } ) } & { \quad \mathrm { w i t h ~ p r o b a b i l i t y } \quad \frac { 1 } { 2 } } \end{array} \right.
$$

Given this sequence, we construct the instance of prices $\{ ( V _ { t } , M _ { t } ) \} _ { t }$ as follows:

$$
\left\{ \begin{array} { l l } { ( V _ { t } , M _ { t } ) = ( L _ { t } , 1 ) , } & { \quad \mathrm { w i t h ~ p r o b a b i l i t y } \quad \frac { 1 } { 2 } , } \\ { ( V _ { t } , M _ { t } ) = ( U _ { t } , 0 ) , } & { \quad \mathrm { w i t h ~ p r o b a b i l i t y } \quad \frac { 1 } { 2 } , } \end{array} \right.
$$

where the sequence is drawn independently of the previous one.

We now consider an optimal in hindsight action for such instance of the problem. We notice that there exists a $( a ^ { * } , b ^ { * } )$ where $a ^ { * } , b ^ { * } \in [ L _ { T } , U _ { T } ]$ such that the learner always wins. Defining a constant $\begin{array} { r } { \varepsilon \le \frac { D _ { T } } { 2 } } \end{array}$ , we can take such action as $( a ^ { * } , b ^ { * } ) = ( U _ { T } - \varepsilon , L _ { T } )$ . The total expected revenue with respect to the randomization of the environment is:

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } u _ { t } ( a ^ { * } , b ^ { * } ) \right] = T \frac { 1 - L _ { T } + U _ { T } - \varepsilon } { 2 } = T \frac { 1 + D _ { T } - \varepsilon } { 2 } \geq \frac { T } { 2 } + \frac { T D _ { T } } { 4 }
$$

where we imposed $\begin{array} { r } { \varepsilon = \frac { D _ { T } } { 2 } } \end{array}$ and $\begin{array} { r } { D _ { T } = \frac { 1 } { 3 ^ { T + } } } \end{array}$ <sub>1</sub> and vanishes as $T$ grows.

We now turn to the expected revenue for any deterministic sequence. For any time step $A _ { t } > U _ { t }$ or $B _ { t } < L _ { t }$ gains 0 revenue for the corresponding utility, while having $A _ { t } \ \leq \ L _ { t }$ or $B _ { t } \ \geq \ U _ { t }$ doesn’t improve utility compared to choosing $A _ { t } = L _ { t }$ and $B _ { t } = U _ { t }$ . With this observation, we can restrict to only considering the cases of bid-ask prices $A _ { t } \in ( L _ { t - 1 } , U _ { t - 1 } ]$ and $B _ { t } \in [ L _ { t - 1 } , U _ { t - 1 } )$ . We have six cases to consider, depending on where the bid ask prices lie in the three intervals $\begin{array} { r l } { ( L _ { t - 1 } , L _ { t - 1 } + \frac { D _ { t - 1 } } { 3 } ) , ( L _ { t - 1 } + \frac { D _ { t - 1 } } { 3 } , L _ { t - 1 } + \frac { 2 D _ { t - 1 } } { 3 } ) } & { { } } \end{array}$ and $\begin{array} { r } { ( L _ { t - 1 } + \frac { 2 D _ { t - 1 } } { 3 } , U _ { t - 1 } ) } \end{array}$ and given our general constraint $A _ { t } \ \geq \ B _ { t }$ . For simplicity, we call $\begin{array} { r } { x = L _ { t - 1 } + \frac { D _ { t - 1 } } { 3 } } \end{array}$ and $\begin{array} { r } { y = L _ { t - 1 } + \frac { 2 D _ { t - 1 } } { 3 } } \end{array}$ in Table 2. We proceed by computing the expected revenue given each event in the original sequence and summing up the resulting revenue for each case, thus obtaining the general expected revenue for a deterministic sequence of prices. We summarize the results for each case with Table 2.

Overall, the expected utility for a deterministic algorithm playing $\begin{array} { r } { ( A _ { t } , B _ { t } ) \in \mathcal { U } \mathrm { ~ i s ~ } \mathbb { E } \left[ u _ { t } ( A _ { t } , B _ { t } ) \right] \leq \frac { 1 } { 3 } } \end{array}$ , given our construction.

Therefore by Yao’s principle that for any probabilistic algorithm $\{ ( A _ { t } , B _ { t } ) \} _ { t = 1 } ^ { T }$ :

$$
R _ { T } \geq \frac { T } { 2 } - \frac { T } { 3 } = \frac { T } { 6 } ,
$$

which proves the statement, by Yao’s principle.