# Toward Optimal Second-Order Path-Length Guarantee for Adversarial Multi-Armed Bandits

Mengxiao Zhang University of Iowa mengxiao-zhang@uiowa.edu

## Abstract

We study second-order path-length regret in adversarial K-armed bandits against oblivious loss sequences. Bubeck et al. [4] designed an algorithm that achieves $\widetilde { \mathcal { O } } ( K + \sqrt { K Q _ { \infty , 1 } } )$ regret, where $Q _ { \infty , 1 }$ is the first-order path length, and left open whether $ { \widetilde { \mathcal { O } } } (  { \operatorname { p o l y } } ( K ) \sqrt { 1 + Q _ { \infty , 2 } } )$ regret is achievable under bandit feedback, where $Q _ { \infty , 2 }$ is the second-order path length. Somewhat surprisingly, we resolve this question positively by showing that with a more involved analysis, the exact same algorithm of Bubeck et al. [4] achieves $\mathcal { O } \left( K \log ( K T ) + \sqrt { K \log ( K T ) \big ( 1 + Q _ { \infty , 2 } \big ) } \right)$ expected regret when $Q _ { \infty , 2 }$ is known, where T is the horizon. This matches the $\Omega ( \sqrt { K Q _ { \infty , 2 } } )$ lower bound up to logarithmic factors and additive terms. We further remove the knowledge of $Q _ { \infty , 2 }$ using an adaptive restart scheme whose path-length estimator has uniformly bounded increments.

## 1 Introduction

The adversarial multi-armed bandit problem is a canonical model of sequential decision making under partial feedback. At the beginning of the game, the environment secretly decides a loss sequence $\ell _ { 1 } , \ell _ { 2 } , \dots , \ell _ { T } \in [ 0 , 1 ] ^ { K }$ for T rounds where K is the number of actions.<sup>1</sup> In each of $T$ rounds, a learner chooses one of K actions, or arms. Then, she incurs and observes only the selected coordinate $\ell _ { t , I _ { t } } ;$ the other coordinates remain hidden. Performance is measured against the best single arm in hindsight. This elementary protocol captures the central tension between exploration and exploitation and has consequently served as a basic model for online recommendation, routing, allocation, and repeated decision making. Without further structure on the loss sequence, the minimax regret is $\Theta ( \sqrt { K T } )$ [1, 26].

The minimax guarantee is derived for an environment that may change arbitrarily at every round, and can therefore be pessimistic on more regular data. A substantial literature develops adaptive guarantees that replace the horizon by an observable or instance-dependent measure of dificulty. Examples include first-order bounds governed by the loss of a good comparator [19, 24], best-of-both-worlds guarantees that are minimax in adversarial environments and logarithmic in stochastic ones [2, 26], and bounds depending on empirical variance or related second-order quantities [3, 7, 9]. These results express the same broad principle: the regret should depend on the complexity actually present in the loss sequence rather than automaticall scaling as the worst-case rate $\sqrt { K T }$

This paper concerns adaptivity to temporal variation, also called the path length of the loss sequence. For loss vectors $\ell _ { 1 } , \ldots , \ell _ { T } \in [ 0 , 1 ] ^ { \bar { K } }$ , define the first- and second-order path lengths by

$$
Q _ { \infty , 1 } \triangleq \sum _ { t = 2 } ^ { T } \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } , \qquad Q _ { \infty , 2 } \triangleq \sum _ { t = 2 } ^ { T } \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 } .
$$

The first quantity measures movement linearly, whereas the second measures it quadratically. This distinction is substantial when the environment moves through many small increments: if each increment has infinity norm $T ^ { - 1 / 2 }$ , then $Q _ { \infty , 2 } = \mathcal { O } ( 1 )$ although the corresponding first-order movement can be $\mathcal { O } ( \sqrt { T } )$ .

For adversarial MAB, Wei and Luo [24] proved the guarantee $\widetilde { \mathcal { O } } ( K + \sqrt { K Q _ { 1 , 1 } } )$ , where $\begin{array} { r } { Q _ { 1 , 1 } = \sum _ { t = 2 } ^ { T } \lVert \ell _ { t } - } \end{array}$ $\ell _ { t - 1 } \Vert _ { 1 } . \ ^ { 2 }$ Bubeck et al. [4] subsequently introduced a recent-arm-biased optimistic mirror-descent algorithm and proved the sharper guarantee of $\widetilde { \mathcal { O } } ( K + \sqrt { K Q _ { \infty , 1 } } )$ , where $\begin{array} { r } { Q _ { \infty , 1 } = \sum _ { t = 2 } ^ { T } \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } \leq Q _ { 1 , 1 } . ^ { 3 } } \end{array}$ Both guarantees nevertheless depend linearly on the magnitude of each temporal increment. This is diferent from the full-information experts setting, in which the complete vector $\ell _ { t }$ is observed after every round. In that setting, Steinhardt and Liang [21] showed that second-order path-length dependence is achievable. More recently, Chen et al. [5] resolved the associated “impossible tuning” problem: a single expert algorithm can adapt simultaneously to comparator-specific second-order prediction errors without knowing their scales in advance. Thus full information permits such second-order path-length guarantees, while the corresponding possibility under bandit feedback remained open. This leads to the following question left by Bubeck et al. [4], Wei and Luo [24]:

## $I s \widetilde { \mathcal { O } } \bigl ( \mathrm { p o l y } ( K ) ( 1 + \sqrt { Q _ { \infty , 2 } } ) \bigr )$ regret achievable for adversarial multi-armed bandits?

Contributions. We answer the question afirmatively for oblivious loss sequences. Somewhat surprisingly, when $Q _ { \infty , 2 }$ is known, we prove that the exact same algorithm proposed by Bubeck et al. [4], without any change to its sampler, estimator, or mirror-descent update, achieves the regret of

$$
\mathcal { O } \left( K \log ( K T ) + \sqrt { K \log ( K T ) ( 1 + Q _ { \infty , 2 } ) } \right) ,
$$

under oracle tuning. The contribution in Section 3 is therefore a new analysis, not a new algorithm. Standard OMD reduces the proof to a stability divergence plus the bias caused by recent-arm sampling. Instead of taking an absolute value and bounding it by a first-order movement term, as in Bubeck et al. [4], we combine the bias exactly with the prediction-error square and retain a signed diference term. This refined analysis leads to the desired second-order path-length guarantee. We then remove knowledge of $Q _ { \infty , 2 }$ in Section 4 using a diferent, adaptively tuned algorithm with a modified sampler and a bounded-increment path-length estimator. The resulting parameter-free regret is $\mathcal { O } ( K \log ( K T ) + \log T + \sqrt { K Q _ { \infty , 2 } \log ( K T ) \log T } )$ ), which is a factor of $\sqrt { \log T }$ worse than the one with oracle tuning.

Related work. Under full-information feedback, path-length guarantees were developed through gradualvariation and predictable-sequence analyses where optimistic online learning can compare the current loss vector with a predictable hint and obtain regret whose data-dependent term is governed by the prediction error [6, 20]. Using the previous loss vector as the hint turns the squared prediction error into squared temporal variation. Steinhardt and Liang [21] developed an adaptive optimistic exponentiated-gradient algorithm and obtained variance and squared-path regret guarantees for the best expert. Chen et al. [5] later solved the impossible-tuning problem for experts by combining mirror descent, a correction term, and a weighted entropy regularizer. Their algorithm adapts simultaneously to the second-order prediction error of every expert. Both results use the full loss vector. They therefore do not directly tune the hidden squared path length in ordinary MAB, where only one coordinate is observed per round and the squared increments are themselves hidden.

The bandit path-length literature replaces the missing vector observation by importance weighting. Wei and Luo [24] developed the BROAD-OMD framework and obtained bounds involving the first-order movement of the best arm or of all arms. The latter comes with a negative stability term that can cancel the opponents’ movement in repeated games. This gives faster convergence of the average play to equilibrium in two-player zero-sum games with bandit feedback, continuing the full-information connection between no-regret learning and game dynamics [22]. Bubeck et al. [4] improved the MAB dependence from the $\ell _ { 1 }$ path measure to the smaller $\ell _ { \infty }$ path measure by using a common optimistic prediction and biasing play toward the most recently selected arm. Our oracle-tuned result uses precisely that algorithm. What changes is the analysis: their proof bounds the bias by a first-order selected-coordinate movement, whereas our signed identity and contraction reduce the same algorithm to $Q _ { \infty , 2 }$

Beyond temporal variation, adaptive adversarial-bandit guarantees have been developed for small comparator loss [19, 24], loss variance [3, 7, 9], sparse loss vectors [3, 13], and best-of-both-worlds adaptation [2, 12, 16, 26]. These guarantees are generally incomparable because they exploit diferent structure, including cumulative magnitude, dispersion, support size, or stochastic separation. None of these is characterized by temporal smoothness alone. Related instance-adaptive guarantees also extend beyond ordinary MABs, including bandits with feedback graphs [15, 17], adversarial MDPs [10, 11, 14], linear bandits [4, 8], combinatorial semi-bandits [18, 24], and bandit convex optimization [23, 25].

## 2 Notations and Problem Setting

Notation. For $K \geq 2$ , write $[ K ] = \{ 1 , \ldots , K \}$ . Let $e _ { i }$ be the ith standard basis vector and for a vector $v \in \mathbb { R } ^ { K }$ , denote its ith coordinate as $v _ { i }$ . Let $\begin{array} { r } { \Delta _ { K } = \{ p \in \mathbb { R } ^ { K } : \sum _ { i = 1 } ^ { K } p _ { i } = 1 , \ p _ { i } \geq 0 \ \forall i \in [ K ] \} } \end{array}$ be the probability simplex. Inner products and Euclidean norms are denoted by $\langle \cdot , \cdot \rangle$ and $\| \cdot \| _ { 2 }$ . For a diferentiable convex function $\Psi ,$ , its Bregman divergence is $D _ { \Psi } ( x , y ) \triangleq \Psi ( x ) - \Psi ( y ) - \langle \nabla \Psi ( y ) , x - y \rangle$ ⟩. For $z \in \mathbb { R } ^ { K }$ , the notation $z ^ { \odot 2 }$ means coordinatewise squaring: $( z ^ { \odot 2 } ) _ { i } = z _ { i } ^ { 2 }$ for all $i \in [ K ]$ . By convention, we use $0 / 0 = 0$

Problem setting. Before interaction, an oblivious adversary fixes loss vectors $\ell _ { 1 } , \ldots , \ell _ { T } \in [ 0 , 1 ] ^ { K }$ . At round $t ,$ the learner selects $I _ { t } \in [ K ]$ using only its internal randomness and the history $\mathcal { F } _ { t - 1 } = \sigma ( I _ { 1 } , c _ { 1 } , \ldots , I _ { t - 1 } , c _ { t - 1 } )$ incurs loss $c _ { t } = \ell _ { t , I _ { t } }$ , and observes only $c _ { t }$ . The performance criterion is expected static (pseudo) regret

$$
\mathsf { R e g } _ { T } \triangleq \operatorname* { m a x } _ { i \in [ K ] } \mathsf { R e g } _ { T } ( i ) \triangleq \operatorname* { m a x } _ { i \in [ K ] } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \ell _ { t , I _ { t } } - \sum _ { t = 1 } ^ { T } \ell _ { t , i } \right] .\tag{2.1}
$$

All expectations in the upper bounds are over the learner’s internal randomization (and over the adversary’s initial randomization if it draws the oblivious loss table from a distribution). The goal is to obtain an adaptive regret bound depending on the second-order path length $\begin{array} { r } { Q _ { \infty , 2 } \triangleq \sum _ { t = 2 } ^ { T } \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 } } \end{array}$ . For conciseness, we also write $\mathbb { E } _ { t } [ \cdot ] = \mathbb { E } [ \cdot \mid \mathcal { F } _ { t - 1 } ]$ when conditioning on all the randomness before $I _ { t }$ is drawn.

Lower bound. Adapting the binary-loss construction used in Bubeck et al. [4], a lower bound of $\Omega ( \sqrt { K Q _ { \infty , 2 } } )$ can be directly derived by the standard lower bound instance in adversarial MAB. We include this lower bound in the following for completeness.

Proposition 2.1. For every $K \geq 2 , T \geq 2$ , and $2 K \le q \le T$ , every algorithm has to sufer $\Omega ( { \sqrt { K q } } )$ regret under certain oblivious loss sequence $\ell _ { 1 } , \ell _ { 2 } , \dots , \ell _ { T } \in [ 0 , 1 ] ^ { K }$ where $Q _ { \infty , 2 } \leq q$

Proof. Let $n = \lfloor q \rfloor$ . By the standard adversarial MAB lower bound [1, Theorem 5.1], for any algorithm, one can find an oblivious binary loss sequence of length n on which the algorithm sufers $\Omega ( \sqrt { K n } )$ expected regret. Following Bubeck et al. [4, Section 2], use this sequence during the first n rounds and set all losses to zero afterward. Since the losses are binary, every $\| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 }$ is at most one. There are at most n transitions before the loss sequence becomes identically zero, and hence $Q _ { \infty , 2 } \leq n \leq q$ . Finally, $q \ge 2 K \ge 2$ implies $n = \lfloor q \rfloor \ge q / 2$ , so the regret is $\Omega ( \sqrt { K n } ) = \Omega ( \sqrt { K q } )$ □

## 3 Known $Q _ { \infty , 2 } \colon$ a Refined Analysis of Bubeck et al. [4]

In this section, we first consider the case in which the learner knows the value of $Q _ { \infty , 2 }$ . In Section 4, we will remove this restriction using a more adaptive algorithm. Somewhat surprisingly, no new algorithm is needed when $Q _ { \infty , 2 }$ is known, and we show that Algorithm 1 of Bubeck et al. [4] in fact already achieves the desired guarantee when its learning rate is chosen using $Q _ { \infty , 2 }$ . Our contribution is then a more refined analysis of that algorithm.

For completeness, here we briefly introduce Algorithm 1 of Bubeck et al. [4]. The algorithm maintains two distributions $p _ { t }$ and $x _ { t }$ at each round t. Specifically, the algorithm first selects an action $I _ { t }$ from distribution $p _ { t }$ . After observing the loss $c _ { t } \triangleq \ell _ { t , I _ { t } }$ , the algorithm constructs an unbiased loss estimator $\widehat { \ell } _ { t , i }$ using both $c _ { t - 1 }$ , the loss sufered in the previous round, and $c _ { t }$ . A direct calculation shows that $\mathbb { E } _ { t } [ \widehat { \ell } _ { t , i } ] = \ell _ { t , i }$ for every $i \in [ K ]$ and more importantly, the variance of this estimator can be much smaller when the path length of the loss vectors is small. Next, the algorithm computes $x _ { t + 1 }$ by applying a step of online mirror descent with log-barrier regularizer using $\widehat { \ell } _ { t }$ . Finally, the next-round strategy $p _ { t + 1 }$ is obtained by biasing $x _ { t + 1 }$ toward the arm $I _ { t }$ just selected, with weight determined by $\lambda _ { t + 1 } = \alpha ( 1 - c _ { t } )$ . Algorithm 1 shows the full pseudocode for completeness.

```latex
Algorithm 1: Oracle-tuned Algorithm 1 of Bubeck et al. [4]
Input: $\overline { { K , T , \eta > 0 } }$
1 Set $\begin{array} { r } { x _ { 1 } = p _ { 1 } = \frac { 1 } { K } \mathbf { 1 } , \alpha = 8 \eta . } \end{array}$ , and $c _ { 0 } = 0$
2 Set $\begin{array} { r } { \Psi ( x )  \eta ^ { - 1 } \sum _ { i = 1 } ^ { K } \log ( 1 / x _ { i } ) } \end{array}$
3 for $t = 1 , \dots , T$ do
4 Draw $I _ { t } \sim p _ { t }$ and observe $c _ { t } \gets \ell _ { t , I _ { t } }$
5 Set $\widehat { \ell } _ { t , i } \gets c _ { t - 1 } + \mathbf { 1 } \{ I _ { t } = i \} ( c _ { t } - c _ { t - 1 } ) / p _ { t , i }$ for every $i \in [ K ]$
6 Set $\begin{array} { r } { x _ { t + 1 } \gets \arg \operatorname* { m i n } _ { x \in \Delta _ { K } } \Big ( \Big \langle x , \widehat { \ell } _ { t } \Big \rangle + D _ { \Psi } ( x , x _ { t } ) \Big ) } \end{array}$
7 Set $\lambda _ { t + 1 }  \alpha ( 1 - c _ { t } )$ and $p _ { t + 1 }  ( x _ { t + 1 } + \lambda _ { t + 1 } e _ { I _ { t } } ) / ( 1 + \lambda _ { t + 1 } )$
```

Bubeck et al. [4] shows that Algorithm 1 achieves $\widetilde { \mathcal { O } } ( K + \sqrt { K Q _ { \infty , 1 } } )$ regret where $\begin{array} { r } { Q _ { \infty , 1 } = \sum _ { t = 2 } ^ { T } \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } } \end{array}$ is the first-order path-length that can be much larger than $Q _ { \infty , 2 }$ . However, we show in the following theorem that when η and α are chosen properly based on the knowledge of $Q _ { \infty , 2 }$ , Algorithm 1 in fact achieves the desired $\widetilde { \mathcal { O } } ( K + \sqrt { K Q _ { \infty , 2 } } )$ regret bound.

Theorem 3.1. For every $K \ge 2 , T \ge 2$ , and oblivious loss sequence $\ell _ { 1 } , \ldots , \ell _ { T } \in [ 0 , 1 ] ^ { K }$ , Algorithm 1 with $\begin{array} { r } { \eta = \operatorname* { m i n } \left\{ \frac { 1 } { 1 6 2 } , \sqrt { \frac { K \log ( K T ) } { 1 + Q _ { \infty , 2 } } } \right\} } \end{array}$ guarantees that

$$
\mathsf { R e g } _ { T } \leq \mathcal { O } \left( K \log ( K T ) + \sqrt { K \log ( K T ) ( 1 + Q _ { \infty , 2 } ) } \right) .\tag{3.1}
$$

To our knowledge, Theorem 3.1 gives the first $\widetilde { \mathcal { O } } ( \mathrm { p o l y } ( K ) ( 1 + \sqrt { Q _ { \infty , 2 } } ) )$ guarantee for adversarial MAB, resolving the open problems proposed in Bubeck et al. [4], Wei and Luo [24]. In the following, we will show the proof of Theorem 3.1, which is our key novelty.

## 3.1 Analysis of Algorithm 1

We begin with the standard OMD analysis and highlight how our analysis is diferent from Bubeck et al. [4]. Define $D _ { t } \triangleq D _ { \Psi } ( x _ { t } , x _ { t + 1 } )$ as the Bregman divergence between consecutive OMD updates $x _ { t }$ and $x _ { t + 1 } .$ , and $B _ { t } \triangleq \langle p _ { t } - x _ { t } , \ell _ { t } \rangle$ as the loss diference between $x _ { t }$ and $p _ { t }$ at round t. For each arm $i \in [ K ]$ , set $\gamma = 1 / ( K T )$ and define the interior comparator $\begin{array} { r } { u ^ { ( i ) } = ( 1 - ( K - 1 ) \gamma ) e _ { i } + \gamma \sum _ { j \neq i } e _ { j } } \end{array}$ . The OMD first-order condition and the three-point identity (e.g. Wei and Luo [24, Lemma 6]) imply that

$$
\left. x _ { t } - u ^ { ( i ) } , \widehat { \ell } _ { t } \right. \leq D _ { \Psi } ( u ^ { ( i ) } , x _ { t } ) - D _ { \Psi } ( u ^ { ( i ) } , x _ { t + 1 } ) + D _ { t } .\tag{3.2}
$$

Summing (3.2), taking expectations, using $\mathbb { E } _ { t } [ \widehat { \ell } _ { t } ] = \ell _ { t }$ , and adding the sampling bias gives, for every $i \in [ K ]$

$$
\begin{array} { r l } & { \mathsf { R e g } _ { T } ( i ) \leq 1 + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \left. p _ { t } - u ^ { ( i ) } , \ell _ { t } \right. \right] } \\ & { \qquad = 1 + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } B _ { t } \right] + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \left. x _ { t } - u ^ { ( i ) } , \widehat { \ell } _ { t } \right. \right] } \\ & { \qquad \leq 1 + D _ { \Psi } ( u ^ { ( i ) } , x _ { 1 } ) + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } ( D _ { t } + B _ { t } ) \right] . } \end{array}\tag{3.3}
$$

A direct calculation from $\textstyle x _ { 1 } = { \frac { 1 } { K } } \mathbf { 1 }$ gives $D _ { \Psi } ( u ^ { ( i ) } , x _ { 1 } ) \le \mathcal { O } ( K \log ( K T ) / \eta )$ , uniformly over $i \in [ K ]$ . Therefore, it remains to prove that

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( D _ { t } + B _ { t } ) \right] \leq \mathcal { O } ( 1 + \eta Q _ { \infty , 2 } ) .\tag{3.4}
$$

This is the exact point at which our analysis departs from Bubeck et al. [4, proof of Theorem $2 ]$ . We briefly discuss how they handle this term. Specifically, their OMD calculation first upper-bounds $\mathbb { E } [ \sum _ { t = 1 } ^ { T } D _ { t } ]$ by $4 \eta \mathbb { E } [ \sum _ { t = 1 } ^ { T } ( c _ { t } - c _ { t - 1 } ) ^ { 2 } ]$ . Then, their subsequent control of the sampling bias $\mathbb { E } [ \sum _ { t = 1 } ^ { T } B _ { t } ]$ contains the negative of the same term shown above and an additional first-order quantity

$$
\alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \left| \ell _ { t , I _ { t - 1 } } - \ell _ { t - 1 , I _ { t - 1 } } \right| \right] .\tag{3.5}
$$

The two residual-square terms cancel, leaving (3.5). This is precisely why their argument produces a first-order path length.

However, we do not make that relaxation in bounding $\mathbb { E } [ \sum _ { t = 1 } ^ { T } ( D _ { t } + B _ { t } ) ]$ . Instead, we keep the exact divergence $D _ { t } ,$ whose coeficient is strictly smaller than the available negative residual square as we will see later, and rewrite $B _ { t }$ by an equality which replaces the first-order term in (3.5) by a squared movement plus a signed diference term.

The next four lemmas implement this argument in three steps: an exact expansion of $B _ { t }$ , the stability control of the OMD movement $x _ { t }$ , and control of the recent-arm correction. For the remainder of the section, we define $r _ { t } \triangleq c _ { t } - c _ { t - 1 } , v _ { t } \triangleq \ell _ { t , I _ { t - 1 } } - \ell _ { t - 1 , I _ { t - 1 } } = \ell _ { t , I _ { t - 1 } } - c _ { t - 1 }$ , and $R \triangleq \mathbb { E } [ \sum _ { t = 2 } ^ { T } r _ { t } ^ { 2 } ]$ . Also, for notational convenience, we define $\varphi ( z ) \triangleq z - z ^ { 2 } / 2$ and set $\phi _ { t } \triangleq ( \varphi ( \ell _ { t , 1 } ) , \dots , \varphi ( \ell _ { t , K } ) )$

Step 1: expand the sampling bias exactly. The first lemma replaces the relaxation leading to (3.5) by an identity. It is a scalar calculation based only on the explicit diference between $p _ { t }$ and $x _ { t }$

Lemma 3.2. Algorithm 1 satisfies that

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } B _ { t } \right] = 4 \eta \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } v _ { t } ^ { 2 } \right] - 4 \eta \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } r _ { t } ^ { 2 } \right] + \alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \langle \phi _ { t } , p _ { t - 1 } - p _ { t } \rangle \right] .\tag{3.6}
$$

Proof. Fix any $t \geq 2$ and condition on $\mathcal { F } _ { t - 1 }$ . Since $I _ { t } \sim p _ { t }$ , we know that $\begin{array} { r } { \mathbb E _ { t } [ r _ { t } ^ { 2 } ] = \sum _ { i = 1 } ^ { K } p _ { t , i } ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } \end{array}$ while $( \ell _ { t , I _ { t - 1 } } - c _ { t - 1 } ) ^ { 2 } = v _ { t } ^ { 2 }$ . Moreover, the definition of $p _ { t }$ gives $p _ { t } = ( x _ { t } + \alpha ( 1 - c _ { t - 1 } ) e _ { I _ { t - 1 } } ) / ( 1 + \alpha ( 1 - c _ { t - 1 } ) )$ Consequently, $p _ { t } - x _ { t } = \alpha ( 1 - c _ { t - 1 } ) ( e _ { I _ { t - 1 } } - p _ { t } )$ , and hence

$$
B _ { t } = \alpha ( 1 - c _ { t - 1 } ) \sum _ { i = 1 } ^ { K } p _ { t , i } \mathopen { } \mathclose \bgroup \left( \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } \aftergroup \egroup \right) .\tag{3.7}
$$

We next separate the residual-square diference $4 \eta ( v _ { t } ^ { 2 } - \mathbb { E } _ { t } [ r _ { t } ^ { 2 } ] )$ from (3.7). This is the quantity whose negative part will later be combined with the OMD stability term $D _ { t }$ . For each $i \in [ K ]$ , add and subtract the corresponding residual-square diference:

$$
\begin{array} { r l } & { \alpha ( 1 - c _ { t - 1 } ) ( \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } ) } \\ & { = 4 \eta \left( v _ { t } ^ { 2 } - ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } \right) + \left[ 4 \eta \left( ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } - v _ { t } ^ { 2 } \right) + \alpha ( 1 - c _ { t - 1 } ) ( \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } ) \right] } \\ & { = 4 \eta \left( v _ { t } ^ { 2 } - ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } \right) + 4 \eta ( \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } ) ( 2 - \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } ) } \\ & { = 4 \eta \left( v _ { t } ^ { 2 } - ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } \right) + \alpha \left( \varphi ( \ell _ { t , I _ { t - 1 } } ) - \varphi ( \ell _ { t , i } ) \right) . } \end{array}\tag{3.8}
$$

Multiplying (3.8) by $p _ { t , i }$ and summing over $i \in [ K ]$ now yields

$$
B _ { t } = 4 \eta v _ { t } ^ { 2 } - 4 \eta \mathbb { E } _ { t } \left[ r _ { t } ^ { 2 } \right] + \alpha \left( \varphi ( \ell _ { t , I _ { t - 1 } } ) - \mathbb { E } _ { t } \left[ \varphi ( \ell _ { t , I _ { t } } ) \right] \right) .\tag{3.9}
$$

It remains to identify the expectation of the final term. As the loss vectors $\ell _ { 1 } , \ldots , \ell _ { T }$ are oblivious, the tower property gives

$$
\begin{array} { r } { \mathbb { E } \left[ \varphi ( \ell _ { t , I _ { t - 1 } } ) - \mathbb { E } _ { t } \left[ \varphi ( \ell _ { t , I _ { t } } ) \right] \right] = \mathbb { E } \left[ \left. \phi _ { t } , p _ { t - 1 } \right. - \left. \phi _ { t } , p _ { t } \right. \right] = \mathbb { E } \left[ \left. \phi _ { t } , p _ { t - 1 } - p _ { t } \right. \right] . } \end{array}\tag{3.10}
$$

Taking expectations in (3.9), combining (3.10), and summing over $t = 2 , \ldots , T$ proves (3.6).

Step 2: control the stability of OMD updates $x _ { t }$ . We next control the two terms involving the OMD updates. The first bound in the following lemma compares the stability term $D _ { t }$ with the negative residual square in Lemma 3.2, and the second controls the movement $x _ { t + 1 } - x _ { t }$ in a weighted Euclidean norm. Both are standard in the analysis of OMD; see, for example, Wei and Luo [24, Definition 11 and Lemmas 12–14]. We include their specialization to our update because the orientation of $D _ { t }$ and the strict constant in (3.11) are useful later. The third bound is not inherited from standard OMD analysis and controls the inner product between $\phi _ { t }$ and the OMD update $x _ { t } - x _ { t + 1 }$ . This turns out to be important to bound the part of the final term in Lemma 3.2 arising from the movement of $x _ { t }$

Lemma 3.3 (Log-barrier stability). Suppose $\eta \leq 1 / 1 6 2$ . For every $t ,$ Algorithm 1 satisfies

$$
D _ { t } \leq 0 . 6 \eta r _ { t } ^ { 2 } ,\tag{3.11}
$$

$$
\begin{array} { r l } { | \langle z , x _ { t + 1 } - x _ { t } \rangle | \leq 1 . 5 \sqrt { \eta D _ { t } } \sqrt { \langle x _ { t } , z ^ { \odot 2 } \rangle } } & { { } f o r \ e v e r y \ z \in \mathbb { R } ^ { K } , } \end{array}\tag{3.12}
$$

$$
\mathbb { E } _ { t } \left[ \langle \phi _ { t } , x _ { t } - x _ { t + 1 } \rangle \right] \leq 8 \mathbb { E } _ { t } \left[ D _ { t } \right] .\tag{3.13}
$$

In addition, $x _ { t + 1 , i } / x _ { t , i } \in [ 0 . 9 9 , 1 . 0 1 ]$ for every $i \in [ K ]$ , and $D _ { \Psi } ( x _ { t + 1 } , x _ { t } ) \le 2 D _ { t }$

Proof. Fix the history before round t. For every fixed $i \in [ K ]$ and $s \in \ [ 0 , 1 ]$ , let $x ^ { ( i ) } ( s )$ be the OMD update obtained when the estimator is $c _ { t - 1 } \mathbf { 1 } + s ( \boldsymbol { \ell } _ { t , i } - c _ { t - 1 } ) e _ { i } / p _ { t , i }$ . Thus $x ^ { ( i ) } ( 0 ) = x _ { t } , x _ { t + 1 } = x ^ { ( I _ { t } ) } ( 1 )$ , and $\begin{array} { r } { r _ { t } = \sum _ { i = 1 } ^ { K } \mathbf { 1 } \{ I _ { t } = i \} ( \ell _ { t , i } - c _ { t - 1 } ) } \end{array}$ . We first prove that $x _ { j } ^ { ( i ) } ( 1 ) / x _ { t , j } \in [ 0 . 9 9 , 1 . 0 1 ]$ for every fixed $i , j \in [ K ]$ The common vector $_ { c _ { t - 1 } 1 }$ has zero inner product with every simplex-tangent direction. Thus $x ^ { ( i ) } ( 1 )$ can be equivalently written as arg $\begin{array} { r } { \operatorname* { m i n } _ { x \in \Delta _ { K } } \left( \langle x , ( \ell _ { t , i } - c _ { t - 1 } ) e _ { i } / p _ { t , i } \rangle + D _ { \Psi } ( x , x _ { t } ) \right) } \end{array}$ . The dual local norm of $( \ell _ { t , i } - c _ { t - 1 } ) e _ { i } / p _ { t , i }$ at $x _ { t }$ is bounded as

$$
\eta x _ { t , i } ^ { 2 } \frac { ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } { p _ { t , i } ^ { 2 } } \leq \eta ( 1 + 8 \eta ) ^ { 2 } ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } \leq \eta ( 1 + 8 \eta ) ^ { 2 } < \frac { 1 } { 9 } ,\tag{3.14}
$$

where we used $p _ { t , i } \geq x _ { t , i } / ( 1 + 8 \eta )$ and $| \ell _ { t , i } - c _ { t - 1 } | \leq 1$ . We next modify the Dikin calculation in Lemmas 12–14 of Wei and Luo [24] only by retaining the exact curvature of the log-barrier. Define the local displacement associated with the fixed index i by $\begin{array} { r } { \Delta _ { t , i } ^ { \mathrm { l o c } } = \left( \sum _ { j = 1 } ^ { K } ( x _ { j } ^ { ( i ) } ( 1 ) - x _ { t , j } ) ^ { 2 } / x _ { t , j } ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ . Instead of replacing the Hessian along the update segment by the relaxed factor $1 / 3$ in Wei and Luo [24, Eq. (15)], direct integration gives

$$
\begin{array} { r l } & { \Big \langle \nabla ( \eta \Psi ) ( x ^ { ( i ) } ( 1 ) ) - \nabla ( \eta \Psi ) ( x _ { t } ) , x ^ { ( i ) } ( 1 ) - x _ { t } \Big \rangle } \\ & { = \displaystyle \int _ { 0 } ^ { 1 } \Big \langle \nabla ^ { 2 } ( \eta \Psi ) ( x _ { t } + s ( x ^ { ( i ) } ( 1 ) - x _ { t } ) ) ( x ^ { ( i ) } ( 1 ) - x _ { t } ) , x ^ { ( i ) } ( 1 ) - x _ { t } \Big \rangle d s } \\ & { = \displaystyle \int _ { 0 } ^ { 1 } \sum _ { j = 1 } ^ { K } \frac { ( x _ { j } ^ { ( i ) } ( 1 ) - x _ { t , j } ) ^ { 2 } } { ( x _ { t , j } + s ( x _ { j } ^ { ( i ) } ( 1 ) - x _ { t , j } ) ) ^ { 2 } } d s \geq \frac { ( \Delta _ { t , i } ^ { \mathrm { l o c } } ) ^ { 2 } } { 1 + \Delta _ { t , i } ^ { \mathrm { l o c } } } . } \end{array}\tag{3.15}
$$

The first equality is the fundamental theorem of calculus applied to $\nabla ( \eta \Psi )$ along the segment from $x _ { t }$ to $x ^ { ( i ) } ( 1 )$ , and the second uses $\nabla ^ { 2 } ( \eta \Psi ) ( x ) = \mathrm { d i a g } ( 1 / x _ { j } ^ { 2 } )$ . To prove the last inequality, the definition of $\Delta _ { t , i } ^ { \mathrm { l o c } }$ gives $| x _ { j } ^ { ( i ) } ( 1 ) - x _ { t , j } | / x _ { t , j } \leq \Delta _ { t , i } ^ { \mathrm { l o c } }$ for every j. Thus the denominator in the integrand is at most $x _ { t , j } ^ { 2 } ( 1 + s \Delta _ { t , i } ^ { \mathrm { l o c } } ) ^ { 2 }$ and hence the integral is at least $\begin{array} { r } { \sum _ { j = 1 } ^ { K } \frac { ( x _ { j } ^ { ( i ) } ( 1 ) - x _ { t , j } ) ^ { 2 } } { x _ { t , i } ^ { 2 } } \int _ { 0 } ^ { 1 } \frac { 1 } { ( 1 + s \Delta _ { t , i } ^ { \mathrm { l o c } } ) ^ { 2 } } d s = \frac { ( \Delta _ { t , i } ^ { \mathrm { l o c } } ) ^ { 2 } } { 1 + \Delta _ { t , i } ^ { \mathrm { l o c } } } } \end{array}$

On the other hand, the KKT equation defining $x ^ { ( i ) } ( 1 )$ is $\nabla \Psi ( x ^ { ( i ) } ( 1 ) ) - \nabla \Psi ( x _ { t } ) + ( ( \ell _ { t , i } - c _ { t - 1 } ) / p _ { t , i } ) e _ { i } +$ $\lambda _ { t , i } \mathbf { 1 } = 0$ for a scalar $\lambda _ { t , i }$ . Taking its inner product with $x ^ { ( i ) } ( 1 ) - x _ { t }$ eliminates the multiplier because $\langle \mathbf { 1 } , x ^ { ( i ) } ( 1 ) - x _ { t } \rangle = 0$ . Therefore,

$$
\begin{array} { r l } & { \Big \langle \nabla ( \eta \Psi ) ( x ^ { ( i ) } ( 1 ) ) - \nabla ( \eta \Psi ) ( x _ { t } ) , x ^ { ( i ) } ( 1 ) - x _ { t } \Big \rangle } \\ & { = - \eta \frac { \ell _ { t , i } - c _ { t - 1 } } { p _ { t , i } } ( x _ { i } ^ { ( i ) } ( 1 ) - x _ { t , i } ) \leq \eta \frac { | \ell _ { t , i } - c _ { t - 1 } | } { p _ { t , i } } | x _ { i } ^ { ( i ) } ( 1 ) - x _ { t , i } | \leq \eta x _ { t , i } \frac { | \ell _ { t , i } - c _ { t - 1 } | } { p _ { t , i } } \Delta _ { t , i } ^ { \mathrm { l o c } } . } \end{array}\tag{3.16}
$$

Combining (3.15) and (3.16) gives

$$
\operatorname* { m a x } _ { j \in [ K ] } \left| \frac { x _ { j } ^ { ( i ) } ( 1 ) - x _ { t , j } } { x _ { t , j } } \right| \leq \Delta _ { t , i } ^ { \mathrm { l o c } } \leq \frac { \eta x _ { t , i } | \ell _ { t , i } - c _ { t - 1 } | / p _ { t , i } } { 1 - \eta x _ { t , i } | \ell _ { t , i } - c _ { t - 1 } | / p _ { t , i } } \leq \frac { \eta ( 1 + 8 \eta ) } { 1 - \eta ( 1 + 8 \eta ) } < 0 . 0 1 .\tag{3.17}
$$

The first inequality follows from the definition of $\Delta _ { t , i } ^ { \mathrm { l o c } }$ . For the second, if $\Delta _ { t , i } ^ { \mathrm { l o c } } = 0$ there is nothing to prove; otherwise, divide (3.15)–(3.16) by $\Delta _ { t , i } ^ { \mathrm { l o c } }$ and solve the inequality. The third inequality uses $p _ { t , i } \geq x _ { t , i } / ( 1 + 8 \eta )$ and $| \ell _ { t , i } - c _ { t - 1 } | \leq 1$ . The fourth uses $\eta \leq 1 / 1 6 2$ . Therefore, $x _ { j } ^ { ( i ) } ( 1 ) / x _ { t , j } \in [ 0 . 9 9 , 1 . 0 1 ]$ for every fixed $i , j \in [ K ]$ The representation of the realized update above therefore implies $x _ { t + 1 , j } / x _ { t , j } \in [ 0 . 9 9 , 1 . 0 1 ]$ for every $j \in [ K ]$

Next, we prove (3.11). We retain the orientation of the Bregman divergence to obtain the strict constant needed below. The self-concordant conjugate inequality used in the proof of Wei and Luo [24, Lemma 14], applied with local decrement $\eta x _ { t , i } \vert \ell _ { t , i } - c _ { t - 1 } \vert / p _ { t , i } .$ , bounds $\eta D _ { \Psi } ( x _ { t } , x ^ { ( i ) } ( 1 ) )$ by $- z - \log ( 1 - z )$ at $z = \eta x _ { t , i } \vert \ell _ { t , i } - c _ { t - 1 } \vert / p _ { t , i }$ . Since $- z - \log ( 1 - z ) \leq z ^ { 2 } / ( 2 ( 1 - z ) )$ for $0 \leq z < 1$ , and $p _ { t , i } \geq x _ { t , i } / ( 1 + 8 \eta )$ , we obtain that

$$
D _ { \Psi } ( x _ { t } , x ^ { ( i ) } ( 1 ) ) \leq \frac { 1 } { 2 \eta ( 1 - \eta ( 1 + 8 \eta ) ) } \left( \eta x _ { t , i } \frac { | \ell _ { t , i } - c _ { t - 1 } | } { p _ { t , i } } \right) ^ { 2 } \leq \frac { ( 1 + 8 \eta ) ^ { 2 } } { 2 ( 1 - \eta ( 1 + 8 \eta ) ) } \eta ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } .\tag{3.18}
$$

For $\eta \leq 1 / 1 6 2$ , the final coeficient is no more than 0.6, proving (3.11) after substituting the realized index $I _ { t } .$ We next derive the two consequences of coordinatewise stability used below. By (3.17), $x _ { t + 1 , i } / x _ { t , i } \in$ [0.99, 1.01] for every $i \in [ K ]$ . For every $z \in [ 0 . 9 9 , 1 . 0 1 ]$ , direct diferentiation gives

$$
\begin{array} { l } { { z - 1 - \ln z \leq 2 \left( \displaystyle \frac { 1 } { z } - 1 + \ln z \right) , } } \\ { { \frac { 1 } { z } - 1 + \ln z \geq \displaystyle \frac { ( z - 1 ) ^ { 2 } } { 2 . 1 } . } } \end{array}
$$

Applying the first inequality coordinatewise gives

$$
\begin{array} { r l } & { D _ { \Psi } ( x _ { t + 1 } , x _ { t } ) = \displaystyle \frac { 1 } { \eta } \sum _ { i = 1 } ^ { K } \left( \frac { x _ { t + 1 , i } } { x _ { t , i } } - 1 - \ln \frac { x _ { t + 1 , i } } { x _ { t , i } } \right) } \\ & { \qquad \le \displaystyle \frac { 2 } { \eta } \sum _ { i = 1 } ^ { K } \left( \frac { x _ { t , i } } { x _ { t + 1 , i } } - 1 + \ln \frac { x _ { t + 1 , i } } { x _ { t , i } } \right) } \\ & { \qquad = 2 D _ { \Psi } ( x _ { t } , x _ { t + 1 } ) = 2 D _ { t } , } \end{array}
$$

proving the last argument in the statement. Similarly, the second inequality implies

$$
D _ { t } = \frac { 1 } { \eta } \sum _ { i = 1 } ^ { K } \left( \frac { x _ { t , i } } { x _ { t + 1 , i } } - 1 + \ln \frac { x _ { t + 1 , i } } { x _ { t , i } } \right) \geq \frac { 1 } { 2 . 1 \eta } \sum _ { i = 1 } ^ { K } \left( \frac { x _ { t + 1 , i } - x _ { t , i } } { x _ { t , i } } \right) ^ { 2 } .
$$

Rearranging the terms gives the following inequality

$$
\sum _ { i = 1 } ^ { K } \frac { ( x _ { t + 1 , i } - x _ { t , i } ) ^ { 2 } } { x _ { t , i } ^ { 2 } } \leq 2 . 1 \eta D _ { t } .\tag{3.19}
$$

Therefore, for every $z \in \mathbb { R } ^ { K }$ , applying Cauchy–Schwarz in the log-barrier local norm gives

$$
| \langle z , x _ { t + 1 } - x _ { t } \rangle | \leq \sqrt { \sum _ { j = 1 } ^ { K } x _ { t , j } ^ { 2 } z _ { j } ^ { 2 } } \sqrt { \sum _ { j = 1 } ^ { K } \frac { ( x _ { t + 1 , j } - x _ { t , j } ) ^ { 2 } } { x _ { t , j } ^ { 2 } } } \leq 1 . 5 \sqrt { \eta D _ { t } } \sqrt { \langle x _ { t } ^ { \ G 2 } , z ^ { \ G 2 } \rangle } \leq 1 . 5 \sqrt { \eta D _ { t } } \sqrt { \langle x _ { t } , z ^ { \ G 2 } \rangle } ,
$$

where the last inequality uses $x _ { t , j } ^ { 2 } \leq x _ { t , j }$ . Therefore, we prove (3.12).

It remains to prove (3.13), which does not follow from the generic local-norm argument. We prove it by a direct calculation from the KKT equation. Recall that $x ^ { ( i ) } ( s )$ is defined for each fixed $i \in [ K ]$ and $s \in [ 0 , 1 ]$ and write $x _ { j } ^ { ( i ) } ( s )$ for its jth coordinate.

The KKT equation of $x ^ { ( i ) } ( s )$ is $\nabla \Psi ( x ^ { ( i ) } ( s ) ) + s ( \ell _ { t , i } - c _ { t - 1 } ) e _ { i } / p _ { t , i } + \lambda ^ { ( i ) } ( s ) \mathbf { 1 } = \nabla \Psi ( x _ { t } )$ for a scalar $\lambda ^ { ( i ) } ( s )$ . Since $\nabla ^ { 2 } \Psi ( x ) = \eta ^ { - 1 } \mathrm { d i a g } ( 1 / x _ { 1 } ^ { 2 } , \ldots , 1 / x _ { K } ^ { 2 } )$ , diferentiating the jth coordinate gives $\begin{array} { r } { \frac { d } { d s } x _ { j } ^ { ( i ) } ( s ) = } \end{array}$ $- \eta ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } ( ( \ell _ { t , i } - c _ { t - 1 } ) \mathbf { 1 } ( j = i ) / p _ { t , i } + ( \lambda ^ { ( i ) } ) ^ { \prime } ( s ) )$ . Summing this identity over $j$ and using $\begin{array} { r } { \sum _ { j = 1 } ^ { K } \frac { d } { d s } x _ { j } ^ { ( i ) } ( s ) = 0 } \end{array}$ determines $( \lambda ^ { ( i ) } ) ^ { \prime } ( s )$ . Substitution gives

$$
- \frac { d } { d s } x _ { j } ^ { ( i ) } ( s ) = \eta \frac { \ell _ { t , i } - c _ { t - 1 } } { p _ { t , i } } \left( ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } \mathbf { 1 } ( j = i ) - \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \right) .\tag{3.20}
$$

Multiplying (3.20) by $\varphi ( \ell _ { t , j } )$ , summing over $j \in [ K ]$ , and integrating over $s \in [ 0 , 1 ]$ gives

$$
\begin{array} { r l } { \left. \phi _ { t } , x _ { t } - x ^ { ( i ) } ( 1 ) \right. = - \displaystyle \int _ { 0 } ^ { 1 } \left. \phi _ { t } , \frac { d } { d x } x ^ { ( i ) } ( s ) \right. d s } & { } \\ & { = \displaystyle \eta \frac { \hat { \varepsilon } _ { t , i } - c _ { t - 1 } } { p _ { t , i } } \int _ { 0 } ^ { 1 } \sum _ { j = 1 } ^ { K } \varphi ( \hat { \varepsilon } _ { t , i } ) \left( ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } \mathbf { 1 } ( j = i ) - \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \right) d s } \\ & { = \displaystyle \eta \frac { \hat { \varepsilon } _ { t , i } - c _ { t - 1 } } { p _ { t , i } } \int _ { 0 } ^ { 1 } \left( \varphi ( \hat { \varepsilon } _ { t , i } ) ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } - \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \sum _ { j = 1 } ^ { K } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } \varphi ( \hat { \varepsilon } _ { t , i } ) \right) d s } \\ &  = \displaystyle \eta \frac { \hat { \varepsilon } _ { t , i } - c _ { t - 1 } } { p _ { t , i } } \int _ { 0 } ^ { 1 } \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \sum _ { j = 1 } ^ { K } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } \varphi ( \hat { \varepsilon } _ { t , i } ) - \varphi ( \hat { \varepsilon } _  t , \end{array}\tag{3.21}
$$

We first establish coordinatewise stability along the entire scaled update path. Fix $i \in [ K ]$ and $s \in [ 0 , 1 ]$ Since $\boldsymbol { x } ^ { ( i ) } ( s )$ is the OMD update driven by $s ( \ell _ { t , i } - c _ { t - 1 } ) e _ { i } / p _ { t , i }$ , repeating the calculation leading to (3.17) gives

$$
\operatorname* { m a x } _ { j \in [ K ] } \left| \frac { x _ { j } ^ { ( i ) } ( s ) - x _ { t , j } } { x _ { t , j } } \right| \leq \frac { \eta x _ { t , i } s | \ell _ { t , i } - c _ { t - 1 } | / p _ { t , i } } { 1 - \eta x _ { t , i } s | \ell _ { t , i } - c _ { t - 1 } | / p _ { t , i } } \leq \frac { \eta ( 1 + 8 \eta ) } { 1 - \eta ( 1 + 8 \eta ) } < 0 . 0 1 .\tag{3.22}
$$

Therefore, 0.99 $x _ { t , j } \leq x _ { j } ^ { ( i ) } ( s ) \leq 1 . 0 1 x _ { t , j }$ for every $j \in [ K ]$ and $s \in [ 0 , 1 ]$ . Consequently,

$$
0 . 9 4 \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \leq \frac { 0 . 9 9 ^ { 4 } } { 1 . 0 1 ^ { 2 } } \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \leq \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \leq \frac { 1 . 0 1 ^ { 4 } } { 0 . 9 9 ^ { 2 } } \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \leq 1 . 0 7 \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } ,\tag{3.23}
$$

which implies that

$$
\left| \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } - \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \right| \leq 0 . 0 7 \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } .\tag{3.24}
$$

Since $x _ { t + 1 } = x ^ { ( I _ { t } ) } ( 1 )$ and the conditional distribution of $I _ { t }$ is $p _ { t }$ , the definition of conditional expectation gives $\begin{array} { r } { \mathbb { E } _ { t } [ \langle \phi _ { t } , x _ { t } - x _ { t + 1 } \rangle ] = \sum _ { i = 1 } ^ { K } p _ { t , i } \left. \phi _ { t } , x _ { t } - x ^ { ( i ) } ( 1 ) \right. } \end{array}$ . Substituting (3.21) into this finite sum cancels each factor $p _ { t , i }$ and gives

$$
= \eta \sum _ { 1 \leq i < j \leq K } ( \varphi ( \ell _ { t , i } ) - \varphi ( \ell _ { t , j } ) ) \int _ { 0 } ^ { 1 } \left( ( \ell _ { t , i } - c _ { t - 1 } ) \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } - ( \ell _ { t , j } - c _ { t - 1 } ) \frac { ( x _ { i } ^ { ( j ) } ( s ) ) ^ { 2 } ( x _ { j } ^ { ( j ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( j ) } ( s ) ) ^ { 2 } } \right) d s .\tag{3.25}
$$

Adding and subtracting $x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } / \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 }$ inside the integral and applying (3.24) yield

$$
\begin{array} { r l } & { \mathbb { E } _ { t } [  \phi _ { t } , x _ { t } - x _ { t + 1 }  ] \leq \eta \displaystyle \sum _ { 1 \leq i < j \leq K } \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \big ( ( \varphi ( \ell _ { t , i } ) - \varphi ( \ell _ { t , j } ) ) ( \ell _ { t , i } - \ell _ { t , j } ) \big . } \\ & { \qquad +  0 . 0 7 | \varphi ( \ell _ { t , i } ) - \varphi ( \ell _ { t , j } ) | ( | \ell _ { t , i } - c _ { t - 1 } | + | \ell _ { t , j } - c _ { t - 1 } | ) ) . } \end{array}\tag{3.26}
$$

Since $\varphi ( x )$ is 1-Lipschitz on [0, 1] and direct calculation shows that

$$
\begin{array} { r } { ( \ell _ { t , i } - \ell _ { t , j } ) ^ { 2 } \leq 2 \left( ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } + ( \ell _ { t , j } - c _ { t - 1 } ) ^ { 2 } \right) , } \\ { | \ell _ { t , i } - \ell _ { t , j } | \left( | \ell _ { t , i } - c _ { t - 1 } | + | \ell _ { t , j } - c _ { t - 1 } | \right) \leq 2 \left( ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } + ( \ell _ { t , j } - c _ { t - 1 } ) ^ { 2 } \right) , } \end{array}
$$

substituting these inequalities into (3.26) gives

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \left[ \langle \phi _ { t } , x _ { t } - x _ { t + 1 } \rangle \right] \leq 2 . 1 4 \eta \displaystyle \sum _ { 1 \leq i < j \leq K } \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \left( ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } + ( \ell _ { t , j } - c _ { t - 1 } ) ^ { 2 } \right) } \\ & { \qquad = 2 . 1 4 \eta \displaystyle \sum _ { i = 1 } ^ { K } \left( x _ { t , i } ^ { 2 } - \frac { x _ { t , i } ^ { 4 } } { \sum _ { j = 1 } ^ { K } x _ { t , j } ^ { 2 } } \right) \left( \ell _ { t , i } - c _ { t - 1 } \right) ^ { 2 } } \\ & { \qquad \leq 2 . 2 \eta \displaystyle \sum _ { i = 1 } ^ { K } \left( x _ { t , i } ^ { 2 } - \frac { x _ { t , i } ^ { 4 } } { \sum _ { j = 1 } ^ { K } x _ { t , j } ^ { 2 } } \right) \left( \ell _ { t , i } - c _ { t - 1 } \right) ^ { 2 } . } \end{array}\tag{3.27}
$$

It remains to compare the right-hand side of (3.27) with $D _ { t }$ . For each fixed $i \in [ K ]$ , the KKT condition defining $x ^ { ( i ) } ( 1 )$ gives

$$
\nabla \Psi ( x _ { t } ) - \nabla \Psi ( x ^ { ( i ) } ( 1 ) ) = c _ { t - 1 } \mathbf { 1 } + \frac { \ell _ { t , i } - c _ { t - 1 } } { p _ { t , i } } e _ { i } + \lambda _ { t , i } \mathbf { 1 }
$$

for some scalar $\lambda _ { t , i }$ . By the definition of the Bregman divergence, we know that

$$
D _ { \Psi } ( x _ { t } , x ^ { ( i ) } ( 1 ) ) + D _ { \Psi } ( x ^ { ( i ) } ( 1 ) , x _ { t } ) = \left. \nabla \Psi ( x _ { t } ) - \nabla \Psi ( x ^ { ( i ) } ( 1 ) ) , x _ { t } - x ^ { ( i ) } ( 1 ) \right. = \frac { \ell _ { t , i } - c _ { t - 1 } } { p _ { t , i } } ( x _ { t , i } - x _ { i } ^ { ( i ) } ( 1 ) ) .\tag{3.28}
$$

Next, setting $j = i$ in (3.20) and integrating over $s \in [ 0 , 1 ]$ gives

$$
x _ { t , i } - x _ { i } ^ { ( i ) } ( 1 ) = \eta \frac { \ell _ { t , i } - c _ { t - 1 } } { p _ { t , i } } \int _ { 0 } ^ { 1 } \left( ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } - \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 4 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \right) d s .\tag{3.29}
$$

Substituting (3.29) into (3.28) yields

$$
\begin{array} { r l } & { D _ { \Psi } ( x _ { t } , x ^ { ( i ) } ( 1 ) ) + D _ { \Psi } ( x ^ { ( i ) } ( 1 ) , x _ { t } ) = \eta \frac { ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } { p _ { t , i } ^ { 2 } } \displaystyle \int _ { 0 } ^ { 1 } \left( ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } - \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 4 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \right) d s } \\ & { \quad \quad \quad \quad = \eta \frac { ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } { p _ { t , i } ^ { 2 } } \displaystyle \int _ { 0 } ^ { 1 } \left( \sum _ { j \neq i } \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } ( x _ { j } ^ { ( i ) } ( s ) ) ^ { 2 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \right) d s } \\ & { \quad \quad \quad \geq \eta \cdot \frac { 0 . 9 4 ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } { p _ { t , i } ^ { 2 } } \left( x _ { t , i } ^ { 2 } - \frac { x _ { t , i } ^ { 4 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \right) } \end{array}\tag{3.30}
$$

where the last inequality uses (3.23).

Finally, as $D _ { t } = D _ { \Psi } ( x _ { t } , x ^ { ( I _ { t } ) } ( 1 ) )$ , applying the definition of conditional expectation to both divergences and then using (3.30) gives

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \left[ D _ { t } + D _ { \Psi } ( x _ { t + 1 } , x _ { t } ) \right] = \eta \displaystyle \sum _ { i = 1 } ^ { K } \frac { ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } { p _ { t , i } } \int _ { 0 } ^ { 1 } \left( ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 2 } - \frac { ( x _ { i } ^ { ( i ) } ( s ) ) ^ { 4 } } { \sum _ { k = 1 } ^ { K } ( x _ { k } ^ { ( i ) } ( s ) ) ^ { 2 } } \right) d s } \\ & { \qquad \ge 0 . 9 4 \eta \displaystyle \sum _ { i = 1 } ^ { K } \frac { ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } { p _ { t , i } } \left( x _ { t , i } ^ { 2 } - \frac { x _ { t , i } ^ { 4 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \right) } \\ & { \qquad \ge 0 . 9 4 \eta \displaystyle \sum _ { i = 1 } ^ { K } \left( x _ { t , i } ^ { 2 } - \frac { x _ { t , i } ^ { 4 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \right) ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } . } \end{array}\tag{3.31}
$$

The last inequality uses $p _ { t , i } \leq 1$ and its non-negativity: $\begin{array} { r } { x _ { t , i } ^ { 2 } - \frac { x _ { t , i } ^ { 4 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } = \sum _ { j \neq i } \frac { x _ { t , i } ^ { 2 } x _ { t , j } ^ { 2 } } { \sum _ { k = 1 } ^ { K } x _ { t , k } ^ { 2 } } \geq 0 . } \end{array}$

Combining (3.27) and (3.31), and then using $D _ { \Psi } ( x _ { t + 1 } , x _ { t } ) \le 2 D _ { t }$ , gives

$$
{  { \mathbb E } } _ { t } \left[ \langle \phi _ { t } , x _ { t } - x _ { t + 1 } \rangle \right] \le \frac { 2 . 2 } { 0 . 9 4 } {  { \mathbb E } } _ { t } \left[ D _ { t } + D _ { \Psi } ( x _ { t + 1 } , x _ { t } ) \right] \le \frac { 6 . 6 } { 0 . 9 4 } {  { \mathbb E } } _ { t } \left[ D _ { t } \right] \le 8 {  { \mathbb E } } _ { t } \left[ D _ { t } \right] .
$$

We now apply the stability lemma to the part of the final term in Lemma 3.2 that contains the OMD iterates $x _ { t }$

Lemma 3.4. Under the conditions of Lemma 3.3, Algorithm 1 guarantees that

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \left. \phi _ { t } , x _ { t - 1 } - x _ { t } \right. \right] \leq 8 \eta \left( 1 + R + \sqrt { R Q _ { \infty , 2 } } \right) .\tag{3.32}
$$

Proof. For every $t \geq 2$ , write $\phi _ { t } = \phi _ { t - 1 } + \left( \phi _ { t } - \phi _ { t - 1 } \right)$ . The left-hand side of (3.32) is therefore equal to $S _ { 1 } + S _ { 2 }$ , where

$$
S _ { 1 } = \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \langle \phi _ { t - 1 } , x _ { t - 1 } - x _ { t } \rangle \right] ,\tag{3.33}
$$

$$
S _ { 2 } = \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \langle \phi _ { t } - \phi _ { t - 1 } , x _ { t - 1 } - x _ { t } \rangle \right] .\tag{3.34}
$$

The term $S _ { 1 }$ pairs $\phi _ { t - 1 }$ with the OMD update performed on round $t - 1$ . Applying (3.13) on round $t - 1$ and summing over $t = 2 , \ldots , T$ gives

$$
S _ { 1 } \leq 8 \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } D _ { t - 1 } \right] = 8 \mathbb { E } \left[ \sum _ { s = 1 } ^ { T - 1 } D _ { s } \right] \leq 4 . 8 \eta \mathbb { E } \left[ \sum _ { s = 1 } ^ { T - 1 } r _ { s } ^ { 2 } \right] \leq 4 . 8 \eta ( 1 + R ) ,
$$

where the second inequality is due to (3.11) and the last inequality follows from $r _ { 1 } ^ { 2 } \leq 1$ and $R = \mathbb { E } [ \sum _ { s = 2 } ^ { T } r _ { s } ^ { 2 } ]$ We next control $S _ { 2 }$ . We treat its first summand separately and write

$$
S _ { 2 } = \mathbb { E } \left[ \langle \phi _ { 2 } - \phi _ { 1 } , x _ { 1 } - x _ { 2 } \rangle \right] + \mathbb { E } \left[ \sum _ { t = 3 } ^ { T } \langle \phi _ { t } - \phi _ { t - 1 } , x _ { t - 1 } - x _ { t } \rangle \right] .
$$

Applying (3.12) on round 1 with $z = \phi _ { 2 } - \phi _ { 1 }$ gives

$$
\left| \mathbb { E } \left[ \langle \phi _ { 2 } - \phi _ { 1 } , x _ { 1 } - x _ { 2 } \rangle \right] \right| \leq 1 . 5 \mathbb { E } \left[ \sqrt { \eta D _ { 1 } } \sqrt { \langle x _ { 1 } , ( \phi _ { 2 } - \phi _ { 1 } ) ^ { \odot 2 } \rangle } \right] \leq 1 . 5 \sqrt { 0 . 6 } \eta \leq 1 . 2 \eta .
$$

Here (3.11) and $r _ { 1 } ^ { 2 } \leq 1$ imply $D _ { 1 } \leq 0 . 6 \eta$ , while the one-Lipschitz property of $\varphi$ and $\ell _ { 1 } , \ell _ { 2 } \in [ 0 , 1 ] ^ { K }$ imply $\left. x _ { 1 } , ( \dot { \phi } _ { 2 } - \dot { \phi } _ { 1 } ) ^ { \odot 2 } \right. \stackrel { \ } { \ \leq \ } 1$

For the remaining summands, applying (3.12) on round $t - 1$ and then Cauchy–Schwarz over time and expectation gives

$$
\left| \mathbb { E } \left[ \sum _ { t = 3 } ^ { T } \langle \phi _ { t } - \phi _ { t - 1 } , x _ { t - 1 } - x _ { t } \rangle \right] \right| \leq 1 . 5 \sqrt { \eta \mathbb { E } \left[ \sum _ { t = 3 } ^ { T } D _ { t - 1 } \right] } \sqrt { \mathbb { E } \left[ \sum _ { t = 3 } ^ { T } \langle x _ { t - 1 } , ( \phi _ { t } - \phi _ { t - 1 } ) ^ { \odot 2 } \rangle \right] } .\tag{3.35}
$$

By (3.11) and the definition of $R .$

$$
\mathbb { E } \left[ \sum _ { t = 3 } ^ { T } D _ { t - 1 } \right] = \mathbb { E } \left[ \sum _ { s = 2 } ^ { T - 1 } D _ { s } \right] \leq 0 . 6 \eta \mathbb { E } \left[ \sum _ { s = 2 } ^ { T - 1 } r _ { s } ^ { 2 } \right] \leq 0 . 6 \eta R .
$$

Moreover, since $\varphi ^ { \prime } ( z ) = 1 - z \in [ 0 , 1 ]$ for $z \in [ 0 , 1 ]$ , φ is 1-Lipschitz. Hence

$$
\left. x _ { t - 1 } , ( \phi _ { t } - \phi _ { t - 1 } ) ^ { \odot 2 } \right. = \sum _ { i = 1 } ^ { K } x _ { t - 1 , i } ( \varphi ( \ell _ { t , i } ) - \varphi ( \ell _ { t - 1 , i } ) ) ^ { 2 } \leq \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 } ,
$$

and therefore

$$
\mathbb { E } \left[ \sum _ { t = 3 } ^ { T } \left. x _ { t - 1 } , ( \phi _ { t } - \phi _ { t - 1 } ) ^ { \odot 2 } \right. \right] \leq Q _ { \infty , 2 } .
$$

Substituting these two bounds into (3.35) gives

$$
\left| S _ { 2 } \right| \leq 1 . 2 \eta + 1 . 5 \sqrt { 0 . 6 } \eta \sqrt { R Q _ { \infty , 2 } } \leq 1 . 2 \eta \left( 1 + \sqrt { R Q _ { \infty , 2 } } \right) .
$$

Combining the bounds on $S _ { 1 }$ and $S _ { 2 }$ yields

$$
S _ { 1 } + S _ { 2 } \leq 4 . 8 \eta ( 1 + R ) + 1 . 2 \eta \left( 1 + \sqrt { R Q _ { \infty , 2 } } \right) \leq 8 \eta \left( 1 + R + \sqrt { R Q _ { \infty , 2 } } \right) ,
$$

which proves (3.32).

Step 3: control the diference between $p _ { t }$ and $x _ { t }$ . Define $b _ { t } \triangleq p _ { t } - x _ { t }$ . Then $p _ { t - 1 } - p _ { t } = ( x _ { t - 1 } - x _ { t } ) +$ $\left( b _ { t - 1 } - b _ { t } \right)$ . The first diference is handled by Lemma $3 . 4$ . It remains to control the part containing $b _ { t } .$ , which arises because $p _ { t }$ places additional mass on the arm selected on the preceding round.

Lemma 3.5. Let $b _ { t } = p _ { t } - x _ { t }$ . Algorithm $\mathit { 1 }$ with $\eta \leq 1 / 1 6 2$ guarantees that

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \left. \phi _ { t } , b _ { t - 1 } - b _ { t } \right. \right] \leq 4 0 \eta \left( 1 + Q _ { \infty , 2 } + \sqrt { R Q _ { \infty , 2 } } \right) .\tag{3.36}
$$

Proof. We first explain why it sufices to control the squared norm of $\mathbb { E } [ b _ { t } ]$ . Since the loss table is oblivious, $\phi _ { t }$ is deterministic, and rearranging the summation terms gives

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \left. \phi _ { t } , b _ { t - 1 } - b _ { t } \right. \right] = \left. \phi _ { 2 } , \mathbb { E } \left[ b _ { 1 } \right] \right. - \left. \phi _ { T } , \mathbb { E } \left[ b _ { T } \right] \right. + \sum _ { t = 2 } ^ { T - 1 } \left. \phi _ { t + 1 } - \phi _ { t } , \mathbb { E } \left[ b _ { t } \right] \right. .\tag{3.37}
$$

Moreover, Cauchy–Schwarz inequality and the 1-Lipschitz property of $\varphi ( \cdot )$ imply that

$$
\sum _ { t = 2 } ^ { T - 1 } \left. \phi _ { t + 1 } - \phi _ { t } , \mathbb { E } \left[ b _ { t } \right] \right. \leq \left( \sum _ { t = 2 } ^ { T - 1 } \left. \phi _ { t + 1 } - \phi _ { t } \right. _ { \infty } ^ { 2 } \right) ^ { 1 / 2 } \left( \sum _ { t = 2 } ^ { T - 1 } \left. \mathbb { E } \left[ b _ { t } \right] \right. _ { 1 } ^ { 2 } \right) ^ { 1 / 2 } \leq \sqrt { Q _ { \infty , 2 } } \left( \sum _ { t = 2 } ^ { T - 1 } \left. \mathbb { E } \left[ b _ { t } \right] \right. _ { 1 } ^ { 2 } \right) ^ { 1 / 2 } ,\tag{3.38}
$$

Therefore, it remains only to prove the following

$$
\sum _ { t = 2 } ^ { T } \left\| \mathbb { E } \left[ b _ { t } \right] \right\| _ { 1 } ^ { 2 } \leq 1 6 \alpha ^ { 2 } ( 1 + R + Q _ { \infty , 2 } ) .\tag{3.39}
$$

For $i \in [ K ]$ , define $\beta _ { t , i } \triangleq \alpha ( 1 - \ell _ { t - 1 , i } ) / ( 1 + \alpha ( 1 - \ell _ { t - 1 , i } ) )$ . The sampling rule gives $b _ { t } = \beta _ { t , I _ { t - 1 } } ( e _ { I _ { t - 1 } } - x _ { t } )$ Since the loss table is oblivious, $\beta _ { t , i }$ is also deterministic. Therefore, direct calculation shows that for each $i \in [ K ]$ ,

$$
\begin{array} { r l r } {  { \mathbb { E } [ b _ { t , i } ] = \mathbb { E } [ p _ { t , i } - x _ { t , i } ] } } & { \quad \mathrm { ( b y ~ d e f n i t i o n ) } } \\ & { = \mathbb { E } [ \beta _ { t , I _ { t - 1 } } ( \mathbf { 1 } \{ I _ { t - 1 } = i \} - x _ { t , i } ) ] } & { \quad \mathrm { ( b y ~ d e f i n i t i o n ~ o f ~ } p _ { t } \mathrm { ) } } \\ & { = \beta _ { t , i } \mathbb { E } [ p _ { t - 1 , i } ] - \mathbb { E } [ \beta _ { t , I _ { t - 1 } } x _ { t , i } ] } & { \quad \mathrm { ( s i n c e ~ } \beta _ { t , i } \mathrm { ~ i s ~ d e t e r m i n i s t i c ) } } \\ & { = \beta _ { t , i } ( \mathbb { E } [ p _ { t - 1 , i } ] - \mathbb { E } [ x _ { t , i } ] ) + \mathbb { E } [ ( \beta _ { t , i } - \beta _ { t , I _ { t - 1 } } ) x _ { t , i } ] } \\ & { = \beta _ { t , i } ( \mathbb { E } [ b _ { t - 1 , i } ] - ( \mathbb { E } [ x _ { t , i } ] - \mathbb { E } [ x _ { t - 1 , i } ] ) ) + \mathbb { E } [ ( \beta _ { t , i } - \beta _ { t , I _ { t - 1 } } ) x _ { t , i } ] , } & { \quad \mathrm { ( 3 . 4 0 ) } } \end{array}
$$

where the last equality uses the fact that $p _ { t - 1 } = x _ { t - 1 } + b _ { t - 1 }$ . Since $0 \leq \beta _ { t , i } \leq \alpha$ for every $i \in [ K ]$ , taking the $\ell _ { 1 }$ norm in (3.40) gives

$$
\begin{array} { r } { \left\| \mathbb { E } \left[ b _ { t } \right] \right\| _ { 1 } \leq \alpha \| \mathbb { E } \left[ b _ { t - 1 } \right] \| _ { 1 } + \alpha \| \mathbb { E } \left[ x _ { t } \right] - \mathbb { E } \left[ x _ { t - 1 } \right] \| _ { 1 } + \left\| \big ( \mathbb { E } \left[ ( \beta _ { t , i } - \beta _ { t , I _ { t - 1 } } ) x _ { t , i } \right] \big ) _ { i = 1 } ^ { K } \right\| _ { 1 } . } \end{array}
$$

We next bound the final term. For notational convenience, we draw an auxiliary arm $J$ from $x _ { t }$ conditionally on $\mathcal { F } _ { t - 1 }$ according to $x _ { t }$ . Since the mapping $z \mapsto \alpha ( 1 - z ) / ( 1 + \alpha ( 1 - z ) )$ is α-Lipschitz, we have $| \beta _ { t , i } - \beta _ { t , j } | \leq$ $\alpha | \ell _ { t - 1 , i } - \ell _ { t - 1 , j } |$ . The triangle inequality followed by the conditional law of J and Jensen’s inequality therefore gives

$$
\begin{array} { r l } {  { \Big \| \big ( \mathbb { E } [ ( \beta _ { t , i } - \beta _ { t , I _ { t - 1 } } ) x _ { t , i } ] \big ) _ { i = 1 } ^ { K } \Big \| _ { 1 } \leq \mathbb { E } [ \sum _ { i = 1 } ^ { K } x _ { t , i } | \beta _ { t , i } - \beta _ { t , I _ { t - 1 } } | ] } \quad } & { } \\ & { = \mathbb { E } [ | \beta _ { t , J } - \beta _ { t , I _ { t - 1 } } | ] } \\ & { \leq \alpha \mathbb { E } [ \big | \ell _ { t - 1 , J } - \ell _ { t - 1 , I _ { t - 1 } } \big | ] } \\ & { \leq \alpha \sqrt { \mathbb { E } [ \big ( \ell _ { t - 1 , J } - \ell _ { t - 1 , I _ { t - 1 } } \big ) ^ { 2 } ] } . } \end{array}
$$

To control the last expectation, observe directly from the definition of $\beta _ { t , I _ { t - 1 } }$ that $p _ { t } = ( 1 - \beta _ { t , I _ { t - 1 } } ) x _ { t } +$ $\beta _ { t , I _ { t - 1 } } e _ { I _ { t - 1 } }$ . Consequently, conditionally on $\mathcal { F } _ { t - 1 }$ , the draw $I _ { t } \sim p _ { t }$ can be generated using a Bernoulli coin that is independent of $J$ conditionally on $\mathcal { F } _ { t - 1 } \colon$ set $I _ { t } = I _ { t - 1 }$ with probability $\beta _ { t , I _ { t - 1 } }$ , and set $I _ { t } = J$ otherwise. Under this equivalent coupling,

$$
\begin{array} { r l } & { \mathbb { E } _ { t } \left[ r _ { t } ^ { 2 } \right] = \beta _ { t , I _ { t - 1 } } ( \ell _ { t , I _ { t - 1 } } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } + ( 1 - \beta _ { t , I _ { t - 1 } } ) \mathbb { E } \left[ ( \ell _ { t , J } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } \mid \mathcal { F } _ { t - 1 } \right] } \\ & { \qquad \geq ( 1 - \alpha ) \mathbb { E } \left[ ( \ell _ { t , J } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } \mid \mathcal { F } _ { t - 1 } \right] . } \end{array}
$$

Taking the full expectation and using $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ now gives

$$
\begin{array} { r l } { \left. { \left\| \left( \mathbb { E } \left[ ( \beta _ { t , i } - \beta _ { t , I _ { t - 1 } } ) x _ { t , i } \right] \right) _ { i = 1 } ^ { K } \right\| _ { 1 } ^ { 2 } \leq \alpha ^ { 2 } \mathbb { E } \left[ ( \ell _ { t - 1 , J } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } \right] } \right. }  \\ & { \leq \frac { 2 \alpha ^ { 2 } } { 1 - \alpha } \mathbb { E } \left[ r _ { t } ^ { 2 } \right] + 2 \alpha ^ { 2 } \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 } } \\ & { \leq 3 \alpha ^ { 2 } \left( \mathbb { E } \left[ r _ { t } ^ { 2 } \right] + \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 } \right) , } \end{array}
$$

where the last inequality uses $\alpha \leq 8 / 1 6 2$ . Summing over $t = 2 , \ldots , T$ yields

$$
\sum _ { t = 2 } ^ { T } \Big \| \big ( \mathbb { E } \left[ ( \beta _ { t , i } - \beta _ { t , I _ { t - 1 } } ) x _ { t , i } \right] \big ) _ { i = 1 } ^ { K } \Big \| _ { 1 } ^ { 2 } \leq 3 \alpha ^ { 2 } ( R + Q _ { \infty , 2 } ) .
$$

It remains to bound the movement of $x _ { t }$ in (3.40). For every $t \geq 2 .$ , Jensen’s and Cauchy–Schwarz

inequality give

$$
\begin{array} { r l r } {  { \| \mathbb { E } [ \boldsymbol { x } _ { t } ] - \mathbb { E } [ \boldsymbol { x } _ { t - 1 } ] \| _ { 1 } ^ { 2 } \leq \mathbb { E } [ \| \boldsymbol { x } _ { t } - \boldsymbol { x } _ { t - 1 } \| _ { 1 } ^ { 2 } ] } } \\ & { } & { \leq \mathbb { E } [ \displaystyle \sum _ { i = 1 } ^ { K } \frac { ( \boldsymbol { x } _ { t , i } - \boldsymbol { x } _ { t - 1 , i } ) ^ { 2 } } { \boldsymbol { x } _ { t - 1 , i } } ] } \\ & { } & { \leq \mathbb { E } [ \displaystyle \sum _ { i = 1 } ^ { K } \frac { ( \boldsymbol { x } _ { t , i } - \boldsymbol { x } _ { t - 1 , i } ) ^ { 2 } } { \boldsymbol { x } _ { t - 1 , i } ^ { 2 } } ] \leq 2 . 1 \eta \mathbb { E } [ \boldsymbol { D } _ { t - 1 } ] , } \end{array}
$$

where the third inequality uses $x _ { t - 1 , i } \leq 1$ , and the last inequality is (3.19) applied on round $t - 1$ . Summing this display and then applying (3.11) gives

$$
\sum _ { t = 2 } ^ { T } \left. \mathbb { E } \left[ x _ { t } \right] - \mathbb { E } \left[ x _ { t - 1 } \right] \right. _ { 1 } ^ { 2 } \leq 2 . 1 \eta \mathbb { E } \left[ \sum _ { t = 1 } ^ { T - 1 } D _ { t } \right] \leq 1 . 2 6 \eta ^ { 2 } \mathbb { E } \left[ \sum _ { t = 1 } ^ { T - 1 } r _ { t } ^ { 2 } \right] \leq 2 \eta ^ { 2 } ( 1 + R ) .
$$

Finally, we square the preceding bound on $\lVert \mathbb { E } [ b _ { t } ] \rVert _ { 1 }$ , use $( a + b + c ) ^ { 2 } \leq 3 a ^ { 2 } + 3 b ^ { 2 } + 3 c ^ { 2 }$ , and sum over $t = 2 , \ldots , T$ . The two bounds just proved yield

$$
\sum _ { t = 2 } ^ { T } \| \mathbb { E } \left[ b _ { t } \right] \| _ { 1 } ^ { 2 } \leq 3 \alpha ^ { 2 } \sum _ { t = 2 } ^ { T } \| \mathbb { E } \left[ b _ { t - 1 } \right] \| _ { 1 } ^ { 2 } + 6 \alpha ^ { 2 } \eta ^ { 2 } ( 1 + R ) + 9 \alpha ^ { 2 } ( R + Q _ { \infty , 2 } ) .
$$

Rearranging the terms and using the fact that $6 \eta ^ { 2 } < 1$ , we know that

$$
\big ( 1 - 3 \alpha ^ { 2 } \big ) \sum _ { t = 2 } ^ { T } \| \mathbb { E } \left[ b _ { t } \right] \| _ { 1 } ^ { 2 } \leq 1 0 \alpha ^ { 2 } ( 1 + R + Q _ { \infty , 2 } ) ,
$$

and $1 - 3 \alpha ^ { 2 } > 9 9 / 1 0 0$ proves (3.39).

Finally, the endpoint terms in (3.37) are at most α because $b _ { 1 } = 0 , \Vert \mathbb { E } [ b _ { T } ] \Vert _ { 1 } \leq 2 \alpha$ , and $0 \le \varphi \le 1 / 2$ Combining (3.38) and (3.39) gives

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \langle \phi _ { t } , b _ { t - 1 } - b _ { t } \rangle \right] \leq \alpha + 4 \alpha \sqrt { Q _ { \infty , 2 } ( 1 + R + Q _ { \infty , 2 } ) } \leq 5 \alpha \left( 1 + Q _ { \infty , 2 } + \sqrt { R Q _ { \infty , 2 } } \right) .
$$

Since $\alpha = 8 \eta$ , this proves (3.36).

## 3.2 Proof of Theorem 3.1

Proof. Fix an arbitrary arm $i \in [ K ]$ and the corresponding comparator $\boldsymbol { u } ^ { ( i ) }$ defined above. The OMD decomposition (3.3) reduces the proof to bounding $D _ { t } + B _ { t }$ . We now combine the three steps above in the same order in which they were proved.

Lemma 3.2 and the identity $p _ { t } = x _ { t } + b _ { t }$ give, up to the O(1) first-round boundary,

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } ( D _ { t } + B _ { t } ) \right] = \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } D _ { t } \right] + 4 \eta \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } v _ { t } ^ { 2 } \right] - 4 \eta R } \\ & { \quad \quad \quad \quad + \alpha \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \left. \phi _ { t } , x _ { t - 1 } - x _ { t } \right. \right] + \alpha \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \left. \phi _ { t } , b _ { t - 1 } - b _ { t } \right. \right] + { \mathcal O } ( 1 ) . } \end{array}\tag{3.41}
$$

We bound the four terms on the right-hand side separately. First, Lemma 3.3 gives $\begin{array} { r } { \mathbb { E } [ \sum _ { t = 2 } ^ { T } D _ { t } ] \le 0 . 6 \eta R } \end{array}$ Second, $v _ { t } ^ { 2 } \leq \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 }$ pathwise, and hence $\begin{array} { r } { \mathbb { E } [ \sum _ { t = 2 } ^ { T } v _ { t } ^ { 2 } ] \leq Q _ { \infty , 2 } . } \end{array}$ Third, Lemma 3.4, together with $\alpha = 8 \eta$ bounds the term containing $x _ { t - 1 } - x _ { t }$ by $6 4 \eta ^ { 2 } ( 1 + R + \sqrt { R Q _ { \infty , 2 } } )$ . Finally, Lemma 3.5 bounds the term containing $b _ { t - 1 } - b _ { t }$ by $3 2 0 \eta ^ { 2 } ( 1 + Q _ { \infty , 2 } + \sqrt { R Q _ { \infty , 2 } } )$

It remains to absorb the terms depending positively on $R .$ Since $\eta \leq 1 / 1 6 2$ , we have $6 4 \eta ^ { 2 } R \le 0 . 4 \eta R$ and $3 8 4 \eta ^ { 2 } \sqrt { R Q _ { \infty , 2 } } \leq 0 . 8 \eta R + 1 . 8 \eta Q _ { \infty , 2 }$ . Therefore, the entire positive R-dependent contribution is at most $1 . 2 \eta R .$ . Substituting the four bounds into (3.41) therefore gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( D _ { t } + B _ { t } ) \right] \leq ( 0 . 6 - 4 + 1 . 2 ) \eta R + \mathcal { O } ( \eta Q _ { \infty , 2 } + 1 ) \leq \mathcal { O } ( \eta Q _ { \infty , 2 } + 1 ) .\tag{3.42}
$$

This proves (3.4). Finally, $D _ { \Psi } ( u ^ { ( i ) } , x _ { 1 } ) \le \mathcal { O } ( K \log ( K T ) / \eta )$ . Substituting (3.42) into (3.3) and maximizing over $i \in [ K ]$ give the fixed-rate inequality

$$
\mathsf { R e g } _ { T } \leq \mathcal { O } \left( \frac { K \log ( K T ) } { \eta } + \eta Q _ { \infty , 2 } + 1 \right) .\tag{3.43}
$$

Tuning η optimally leads to (3.1).

## 4 Unknown $Q _ { \infty , 2 } \colon$ Adaptive Tuning Via Bounded-Scale Estimator

Section 3 tunes Algorithm 1 using the value of $Q _ { \infty , 2 }$ . To remove this knowledge, a standard approach is the doubling trick: start from an initial guess, run the algorithm with the learning rate corresponding to that guess, and restart with a larger guess once the cumulative variation exceeds the current threshold. Such restart constructions are common in adaptive bandit algorithms (e.g., Lee et al. [15], Wei and Luo [24]). The dificulty here is that the second-order path-length $Q _ { \infty , 2 }$ is not observed under bandit feedback.

Recall that $r _ { t } \triangleq c _ { t } - c _ { t - 1 }$ and $v _ { t } \triangleq \ell _ { t , I _ { t - 1 } } - \ell _ { t - 1 , I _ { t - 1 } }$ . The quantity $v _ { t } ^ { 2 }$ is observed whenever $I _ { t } = I _ { t - 1 }$ because $r _ { t } = v _ { t }$ on this event. This suggests the following single-round path-length estimator

$$
\widehat { v } _ { t } ^ { 2 } \triangleq \frac { \mathbf { 1 } ( I _ { t } = I _ { t - 1 } ) r _ { t } ^ { 2 } } { p _ { t , I _ { t } } } .\tag{4.1}
$$

A direct calculation gives $\mathbb { E } _ { t } [ \widehat { v } _ { t } ^ { 2 } ] = v _ { t } ^ { 2 }$ . It is therefore natural to use the cumulative sum of $\widehat { v } _ { t } ^ { 2 }$ to estimate the total second-order path length. Under Algorithm 1, however, the denominator in (4.1) can be arbitrarily small. Indeed, if $c _ { t - 1 } = 1$ , then the additional mass on $I _ { t - 1 }$ vanishes and, on the event $I _ { t } = I _ { t - 1 } , p _ { t , I _ { t } } = x _ { t , I _ { t - 1 } } .$ Although the log-barrier guarantees that this probability is positive, it does not provide a uniform lower bound. Consequently, $\widehat { v } _ { t } ^ { 2 }$ can be arbitrarily large, and the cumulative estimate can overshoot a doubling threshold by an uncontrolled amount.

We therefore make one modification to Algorithm 1: we replace $\lambda _ { t } = \alpha ( 1 - c _ { t - 1 } )$ by $\lambda _ { t } = \alpha ( 2 - c _ { t - 1 } )$ Equivalently, we add an α floor to the additional mass assigned to $I _ { t - 1 } .$ thereby keeping the denominator bounded away from zero. The loss estimator $\widehat { \ell } _ { t }$ and the OMD update remain unchanged. Since $c _ { t - 1 } \in [ 0 , 1 ]$ this modification gives $p _ { t , I _ { t - 1 } } \geq \alpha / ( 1 + \alpha )$ and makes (4.1) uniformly bounded. We later show that this uniform bound controls the overshoot of the cumulative estimator at every phase transition. The following lemma records the two properties of the estimator that are needed by the doubling argument.

Lemma 4.1. Suppose $p _ { t } = ( x _ { t } + \lambda _ { t } e _ { I _ { t - 1 } } ) / ( 1 + \lambda _ { t } )$ with $\lambda _ { t } = \alpha ( 2 - c _ { t - 1 } )$ . Then the estimator in (4.1) satisfies

$$
\mathbb { E } _ { t } \left[ \widehat { v } _ { t } ^ { 2 } \right] = v _ { t } ^ { 2 } \quad a n d \quad 0 \leq \widehat { v } _ { t } ^ { 2 } \leq \frac { 1 + \alpha } { \alpha } .\tag{4.2}
$$

Proof. The definition of $p _ { t }$ gives $p _ { t , I _ { t - 1 } } \geq \lambda _ { t } / ( 1 + \lambda _ { t } ) \geq \alpha / ( 1 + \alpha )$ . Moreover, $r _ { t } = v _ { t }$ on the event $I _ { t } = I _ { t - 1 }$ Therefore,

$$
\mathbb { E } _ { t } \left[ \widehat { v } _ { t } ^ { 2 } \right] = \sum _ { i = 1 } ^ { K } p _ { t , i } \frac { \mathbf { 1 } ( i = I _ { t - 1 } ) ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } { p _ { t , i } } = { v } _ { t } ^ { 2 } .
$$

For the upper bound, the numerator is nonzero only when $I _ { t } = I _ { t - 1 }$ , in which case $p _ { t , I _ { t } } = p _ { t , I _ { t - 1 } } \geq \alpha / ( 1 + \alpha )$ The result now follows from $| r _ { t } | \le 1$ □

We are now ready to present our adaptive algorithm. Define $A _ { K } = 6 4 K \log ( K T )$ and $\eta _ { 0 } = 1 0 ^ { - 5 }$ . The algorithm starts with the threshold $H _ { 0 } \stackrel { - } { = } 4 A _ { K } \stackrel { - } { / } \eta _ { 0 } ^ { 2 }$ . During phase $j ,$ it uses $\eta _ { j } \triangleq 2 ^ { - j } \eta _ { 0 } , \alpha _ { j } \triangleq 8 \eta _ { j }$ , and $H _ { j } \triangleq 4 A _ { K } / \eta _ { j } ^ { 2 }$ . It samples from $p _ { t }$ and updates $x _ { t + 1 }$ exactly as in Algorithm 1, except for the $\alpha _ { j }$ floor in Line 16. After every noninitial round of the phase, it adds $\widehat { v } _ { t } ^ { 2 }$ to its estimate $\widehat { H } _ { j } . \mathrm { ~ I f ~ } \widehat { H } _ { j }$ exceeds $H _ { j }$ , phase $j$ ends and the outer loop begins phase $j + 1$ on the next round with a uniform initial distribution. The scalar $\widehat { H } _ { j }$ is initialized to zero when phase j begins. The following theorem shows that Algorithm 2 achieves $\widetilde { \mathcal { O } } ( K + \sqrt { K Q _ { \infty , 2 } } )$ without knowing $Q _ { \infty , 2 }$ . The overhead is a $\sqrt { \log T }$ factor compared to the guarantee in Theorem 3.1.

Algorithm 2: Adaptive tuning without knowledge of $Q _ { \infty , 2 }$   
Input: $\overline { { K > 0 , T > 0 } }$   
1 Set $A _ { K } = 6 4 K \log ( K T ) , \eta _ { 0 } = 1 0 ^ { - 5 } , j = 0 , s _ { 0 } = t = 1 , \mathrm { a n d } c _ { 0 } = 0$   
2 while $t \leq T$ do   
3 Set $\eta _ { j } = 2 ^ { - j } \eta _ { 0 } , \alpha _ { j } = 8 \eta _ { j } , H _ { j } = 4 A _ { K } / \eta _ { j } ^ { 2 }$ , and $\widehat { H } _ { j } = 0$   
4 Phase j starts at round $s _ { j } = t$   
5 Set $\begin{array} { r } { x _ { t } = p _ { t } = \frac { 1 } { K } \mathbf { 1 } . } \end{array}$ , and $\widehat { v } _ { s _ { j } } ^ { 2 } = 0$   
6 while $t \leq T$ do   
7 Draw $I _ { t } \sim p _ { t }$ and observe $c _ { t } = \ell _ { t , I _ { t } }$   
8 Set $\widehat { \ell } _ { t , i } = c _ { t - 1 } + \mathbf { 1 } ( I _ { t } = i ) ( c _ { t } - c _ { t - 1 } ) / p _ { t , i }$ for every $i \in [ K ]$   
9 Set $\begin{array} { r } { x _ { t + 1 } = \arg \operatorname* { m i n } _ { x \in \Delta _ { K } } \left( \left. x , \widehat { \ell } _ { t } \right. + D _ { \Psi _ { j } } ( x , x _ { t } ) \right) } \end{array}$ , where $\begin{array} { r } { \Psi _ { j } ( x ) = \eta _ { j } ^ { - 1 } \sum _ { i = 1 } ^ { K } } \end{array}$ <sub>1</sub> log(1/x<sub>i</sub>)   
10 if $t > s _ { j }$ then   
11 Set $r _ { t } = c _ { t } - c _ { t - 1 } , \widehat { v } _ { t } ^ { 2 } = { \bf 1 } ( I _ { t } = I _ { t - 1 } ) r _ { t } ^ { 2 } / p _ { t , I _ { t } }$ , and $\widehat { H } _ { j } \gets \widehat { H } _ { j } + \widehat { v } _ { t } ^ { 2 }$   
12 if $\widehat { H } _ { j } > H _ { j }$ and $t < T$ then   
13 Set $t  t + 1 , j  j + 1$   
14 break   
15 else if $t < T$ then   
16 Set $\lambda _ { t + 1 } = \alpha _ { j } ( 2 - c _ { t } ) { \mathrm { ~ a n d ~ } } p _ { t + 1 } = ( x _ { t + 1 } + \lambda _ { t + 1 } e _ { I _ { t } } ) / ( 1 + \lambda _ { t + 1 } )$   
17 Set $t \gets t + 1$

Theorem 4.2. For every $K \ge 2 , T \ge 2$ , and oblivious loss sequence $\ell _ { 1 } , \ldots , \ell _ { T } \in [ 0 , 1 ] ^ { K }$ , Algorithm $\mathcal { Q }$ requires no knowledge of $Q _ { \infty , 2 }$ and guarantees that

$$
\mathsf { R e g } _ { T } \leq \mathcal { O } \left( K \log ( K T ) + \sqrt { K Q _ { \infty , 2 } \log ( K T ) \log T } \right) .\tag{4.3}
$$

## 4.1 Analysis of Algorithm 2

We first relate the cumulative estimate $\widehat { H } _ { j }$ to the two squared quantities in the analysis of Section 3. Let $\mathsf { J } _ { t }$ denote the value of the algorithm’s phase counter immediately before $I _ { t }$ is drawn. Thus $\mathsf { J } _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable. In the analysis below, j is always a deterministic phase label; the randomness of the active phase is carried by $\mathsf { J } _ { t } .$ Consistently with Algorithm 2, define the first round of phase $j$ by $s _ { j } \triangleq \operatorname* { i n f } \{ t \in [ T ] : \mathsf { J } _ { t } = j \}$ , with the convention inf $\mathcal { D } = \infty ,$ , and define $\chi _ { j , t } \triangleq \mathbf { 1 } ( \mathsf { J } _ { t } = j , \ t > s _ { j } )$ . Hence $\chi _ { j , t }$ is the indicator that round t is a noninitial round of phase $j ,$ , and it is $\mathcal { F } _ { t - 1 }$ -measurable. Let $\rho _ { j } \triangleq \operatorname* { P r } ( s _ { j } \leq T )$ be the probability that phase j is reached. For notational convenience, we define the following two quantities:

$$
R _ { j } \triangleq \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } r _ { t } ^ { 2 } \right] ,\tag{4.4}
$$

$$
P _ { j } \triangleq \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } v _ { t } ^ { 2 } \right] .\tag{4.5}
$$

Specifically, the quantity $R _ { j }$ is the expected cumulative residual square over the noninitial rounds of phase j and the quantity $P _ { j }$ is the expected cumulative squared loss movement along the previously selected arm. By the unbiasedness of $\widehat { v } _ { t } ^ { 2 }$ , it is also the expected value of the doubling statistic accumulated during phase $j$ The next lemma shows that the doubling statistic has expectation $P _ { j } .$ , has controlled overshoot, and that the sum of these phase-wise expectations is at most the true squared path-length.

Lemma 4.3. For every deterministic phase label $j \in \{ 0 , \ldots , T - 1 \}$ , Algorithm 2 guarantees that $H _ { j } \rho _ { j + 1 } \leq$ $P _ { j } \leq 2 H _ { j } \rho _ { j }$ and $\textstyle \sum _ { j = 0 } ^ { T - 1 } P _ { j } \leq Q _ { \infty , 2 }$

Proof. For a deterministic $j ,$ let $\begin{array} { r } { \widehat { H } _ { j } ^ { \mathrm { f i n } } \triangleq \sum _ { t = 2 } ^ { T } \chi _ { j , t } \widehat { v } _ { t } ^ { 2 } } \end{array}$ denote the final value accumulated by phase $j ;$ it is zero on the event $s _ { j } = \infty$ . Since $\chi _ { j , t }$ is $\mathcal { F } _ { t - } .$ -measurable, Lemma 4.1 and the tower property give $\mathbb { E } [ \chi _ { j , t } \widehat { v } _ { t } ^ { 2 } ] = \mathbb { E } [ \chi _ { j , t } v _ { t } ^ { 2 } ]$ Summing over the deterministic set $t = 2 , \ldots , T$ therefore yields $\mathbb { E } [ \widehat { H } _ { j } ^ { \mathrm { f i n } } ] = P _ { j }$

Pathwise, reaching phase $j + 1$ requires phase $j$ to cross its threshold, so $H _ { j } \mathbf { 1 } ( s _ { j + 1 } \leq T ) \leq \widehat { H } _ { j } ^ { \mathrm { f i n } }$ . Before the triggering round the accumulated value is at most $H _ { j }$ , while Lemma 4.1 bounds the triggering increment by $( 1 + \alpha _ { j } ) / \alpha _ { j }$ . Since $\alpha _ { j } = 8 \eta _ { j }$ and $\eta _ { j } \leq 1 0 ^ { - 5 }$ , this increment is at most $1 / ( 4 \eta _ { j } )$ as

$$
\frac { 1 + \alpha _ { j } } { \alpha _ { j } } = \frac { 1 } { 8 \eta _ { j } } + 1 \leq \frac { 1 } { 4 \eta _ { j } } .
$$

Moreover, $H _ { j } = 4 A _ { K } / \eta _ { j } ^ { 2 } \ge 1 / ( 4 \eta _ { j } )$ . Consequently, $\widehat { H } _ { j } ^ { \mathrm { f i n } } \leq 2 H _ { j } \mathbf { 1 } ( s _ { j } \leq T )$ pathwise. Taking expectations in these two pathwise inequalities proves $H _ { j } \rho _ { j + 1 } \leq P _ { j } \leq 2 H _ { j } \rho _ { j }$

Finally, for every $t ,$ at most one indicator $\chi _ { j , t }$ equals one. Since the loss table is fixed and $v _ { t } ^ { 2 } \leq \| \ell _ { t } - \ell _ { t - 1 } \| _ { \infty } ^ { 2 }$ we have

$$
\sum _ { j = 0 } ^ { T - 1 } P _ { j } = \mathbb { E } \left[ \sum _ { { t = 2 } } ^ { T } \sum _ { j = 0 } ^ { T - 1 } \chi _ { j , t } v _ { t } ^ { 2 } \right] \leq \sum _ { { t = 2 } } ^ { T } \lVert \ell _ { t } - \ell _ { { t - 1 } } \rVert _ { \infty } ^ { 2 } = Q _ { \infty , 2 } .
$$

This proves the second claim.

We next show in the following lemma how to control the contribution of the noninitial rounds of each phase. Specifically, as shown in (3.3), in order to bound the regret, the relevant quantity after the standard OMD decomposition is the sum of $D _ { t } + B _ { t }$ , where we recall that $B _ { t } \triangleq \langle p _ { t } - x _ { t } , \ell _ { t } \rangle$ and with a slight abuse of notation, we define $D _ { t } \triangleq D _ { \Psi _ { \star } } ( x _ { t } , x _ { t + 1 } )$ . Here and throughout the phasewise analysis, if round t ends a phase, $x _ { t + 1 }$ denotes the OMD iterate computed in Line 9 before the subsequent reset. The initial round of each phase is excluded from $\chi _ { j , t }$ because its sampling distribution is reset to the uniform distribution and we do not count the path-length quantity across phase boundaries. We will handle its contribution separately in the proof of Theorem 4.2.

Lemma 4.4. For every deterministic phase label $j \in \{ 0 , \ldots , T - 1 \}$ ,

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } ( D _ { t } + B _ { t } ) \right] \leq \mathcal { O } ( \eta _ { j } P _ { j } + \rho _ { j } ) .\tag{4.6}
$$

Proof. Fix a phase label $j \in \{ 0 , 1 , \ldots , T - 1 \}$ . The additional $\alpha _ { j }$ in the sampling weight $\lambda _ { t } = \alpha _ { j } ( 2 - c _ { t - 1 } )$ changes the scalar function used in Lemma 3.2. Therefore, we define $\varphi _ { 2 } ( z ) \triangleq 2 z - z ^ { 2 } / 2$ , so that $\varphi _ { 2 } ( z ) = \varphi ( z ) + z$ and set

$$
\phi _ { 2 , t } \triangleq \left( \varphi _ { 2 } ( \ell _ { t , 1 } ) , \hdots , \varphi _ { 2 } ( \ell _ { t , K } ) \right) .
$$

We decompose $B _ { t }$ at every noninitial round of phase $j$ by following the same add-and-subtract calculation as in Lemma 3.2. Specifically, the definition of the sampling distribution gives

$$
B _ { t } = \langle p _ { t } - x _ { t } , \ell _ { t } \rangle = \alpha _ { j } ( 2 - c _ { t - 1 } ) \sum _ { i = 1 } ^ { K } p _ { t , i } \left( \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } \right) .\tag{4.7}
$$

Direct calculation shows that

$$
\begin{array} { r l } & { 4 \eta _ { j } \left( ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } - ( \ell _ { t , I _ { t - 1 } } - c _ { t - 1 } ) ^ { 2 } \right) + \alpha _ { j } ( 2 - c _ { t - 1 } ) ( \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } ) } \\ & { = 8 \eta _ { j } ( \ell _ { t , I _ { t - 1 } } - \ell _ { t , i } ) \left( 2 - \frac { \ell _ { t , I _ { t - 1 } } + \ell _ { t , i } } { 2 } \right) = \alpha _ { j } \left( \varphi _ { 2 } ( \ell _ { t , I _ { t - 1 } } ) - \varphi _ { 2 } ( \ell _ { t , i } ) \right) . } \end{array}\tag{4.8}
$$

Multiplying (4.8) by $p _ { t , i }$ , summing over $i \in [ K ]$ , and using the fact that $\begin{array} { r } { \mathbb { E } _ { t } \left[ r _ { t } ^ { 2 } \right] = \sum _ { i = 1 } ^ { K } p _ { t , i } ( \ell _ { t , i } - c _ { t - 1 } ) ^ { 2 } } \end{array}$ and $v _ { t } ^ { 2 } = ( \ell _ { t , I _ { t - 1 } } - c _ { t - 1 } ) ^ { 2 }$ yield

$$
B _ { t } = 4 \eta _ { j } v _ { t } ^ { 2 } - 4 \eta _ { j } \mathbb { E } _ { t } \left[ r _ { t } ^ { 2 } \right] + \alpha _ { j } \left( \varphi _ { 2 } ( \ell _ { t , I _ { t - 1 } } ) - \mathbb { E } _ { t } \left[ \varphi _ { 2 } ( \ell _ { t , I _ { t } } ) \right] \right) .\tag{4.9}
$$

Since $\chi _ { j , t }$ is $\mathcal { F } _ { t - 1 }$ -measurable, the tower property gives $\mathbb { E } \left[ \chi _ { j , t } \mathbb { E } _ { t } \left[ \varphi _ { 2 } ( \ell _ { t , I _ { t } } ) \right] \right] = \mathbb { E } \left[ \chi _ { j , t } \varphi _ { 2 } ( \ell _ { t , I _ { t } } ) \right]$ . It remains to bound

$$
\alpha _ { j } \mathbb { E } \left[ \sum _ { { t } = 2 } ^ { T } \chi _ { j , t } \left( \varphi _ { 2 } ( \ell _ { t , I _ { t - 1 } } ) - \varphi _ { 2 } ( \ell _ { t , I _ { t } } ) \right) \right] .
$$

Two deterministic-horizon steps from Section 3 cannot be obtained by simply multiplying each summand by $\chi _ { j , t }$ . First, $\chi _ { j , t }$ is $\mathcal { F } _ { t - 1 }$ -measurable but need not be $\mathcal { F } _ { t }$ <sub>−2</sub>-measurable. Although

$$
\mathbb { E } [ \mathbf { 1 } \{ I _ { t - 1 } = i \} \mid \mathcal { F } _ { t - 2 } ] = p _ { t - 1 , i } ,
$$

the indicator $\chi _ { j , t }$ cannot in general be taken outside this conditional expectation. Consequently, in general, $\mathbb { E } [ \chi _ { j , t } \mathbf { 1 } \{ I _ { t - 1 } = i \} ] \neq \mathbb { E } [ \chi _ { j , t } p _ { t - 1 , i } ]$

The second issue concerns the phasewise tuning of $\eta _ { j }$ using the path-length estimator. With $d _ { t , i } =$ $\ell _ { t , i } - \ell _ { t - 1 , i }$ , define

$$
Q _ { j } ^ { \operatorname* { m a x } } \triangleq \mathbb { E } \left[ \sum _ { { t = 2 } } ^ { T } \chi _ { j , { t } } \operatorname* { m a x } _ { i \in [ K ] } d _ { { t } , i } ^ { 2 } \right] .\tag{4.10}
$$

The phase indicators are disjoint, and hence $\textstyle \sum _ { j } Q _ { j } ^ { \mathrm { m a x } } \leq Q _ { \infty , 2 }$ . Even after handling the first issue through stopping-boundary corrections, a direct phasewise application of the maximum-norm argument from Section 3 would give a bound of the form

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } ( D _ { t } + B _ { t } ) \right] \leq \mathcal { O } \left( \eta _ { j } P _ { j } + \eta _ { j } ^ { 2 } Q _ { j } ^ { \operatorname* { m a x } } + \rho _ { j } \right) .\tag{4.11}
$$

Summing the additional term over phases yields only $\begin{array} { r } { \sum _ { j } \eta _ { j } ^ { 2 } Q _ { j } ^ { \operatorname* { m a x } } \le \eta _ { 0 } ^ { 2 } \sum _ { j } Q _ { j } ^ { \operatorname* { m a x } } \le \eta _ { 0 } ^ { 2 } Q _ { \infty , 2 } } \end{array}$ , which depends linearly on $Q _ { \infty , 2 }$ and is therefore insuficient for the desired bound. The phasewise estimator, however, controls

$$
P _ { j } \triangleq \sum _ { t = 2 } ^ { T } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { j , t } \mathbf { 1 } \{ I _ { t - 1 } = i \} ] d _ { t , i } ^ { 2 } .\tag{4.12}
$$

Therefore, to exploit the phasewise choice of $\eta _ { j }$ , the analysis must replace $Q _ { j } ^ { \mathrm { m a x } }$ by $P _ { j }$

The first issue can be handled by a one-round correction. If $A _ { j , t } = \mathbf { 1 } \{ \mathsf { J } _ { t } \overset { \cdot } { = } j \}$ , then

$$
A _ { j , t - 1 } = \chi _ { j , t } + \delta _ { j , t } , \qquad { \mathbb E } \left[ \sum _ { t = 2 } ^ { T } \delta _ { j , t } \right] \leq \rho _ { j } ,\tag{4.13}
$$

where $A _ { j , t - 1 }$ is measurable before $I _ { t - 1 }$ is drawn. At the boundary we use the OMD update computed before the subsequent reset. This extension is suficient for the OMD part of Lemma 3.4: if $g _ { t } = \phi _ { 2 , t } - \phi _ { 2 , t - 1 }$ , then

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } A _ { j , t - 1 } \left. x _ { t - 1 } , g _ { t } ^ { \odot 2 } \right. \right] \leq 5 ( P _ { j } + \rho _ { j } ) .\tag{4.14}
$$

Here we use $| g _ { t , i } | \leq 2 | d _ { t , i } |$ by the 2-Lipschitzness of $\varphi _ { 2 } ( \cdot )$ and $p _ { t - 1 , i } \geq x _ { t - 1 , i } / ( 1 + 2 \alpha _ { j } )$ .

The second issue requires a self-normalized replacement for the unavailable maximum-coordinate bound. Following the notation in Section $^ { 3 , }$ we set $b _ { t } \triangleq p _ { t } - x _ { t }$ . With $q _ { j , t , i } \triangleq \mathbb { E } [ \chi _ { j , t } ( { \bf 1 } \{ I _ { t } = i \} + x _ { t , i } ) ]$ , Cauchy–Schwarz inequality gives that

$$
\left| \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } g _ { t + 1 , i } \mathbb { E } [ \chi _ { j , t } b _ { t , i } ] \right| \leq \left( \sum _ { t , i } \frac { \left( \mathbb { E } [ \chi _ { j , t } b _ { t , i } ] \right) ^ { 2 } } { q _ { j , t , i } } \right) ^ { 1 / 2 } \left( \sum _ { t , i } q _ { j , t , i } g _ { t + 1 , i } ^ { 2 } \right) ^ { 1 / 2 } .\tag{4.15}
$$

With a more careful calculation, we bound the right-hand side terms as follows

$$
\begin{array} { r l r } {  { \sum _ { t , i } \frac { ( \mathbb { E } [ \chi _ { j , t } b _ { t , i } ] ) ^ { 2 } } { q _ { j , t , i } } \le \mathcal { O } ( \eta _ { j } ^ { 2 } R _ { j } + \alpha _ { j } ^ { 2 } P _ { j } + \alpha _ { j } \rho _ { j } ) , } } \\ & { } & { \sum _ { t , i } q _ { j , t , i } g _ { t + 1 , i } ^ { 2 } \le \mathcal { O } ( \eta _ { j } ^ { 2 } R _ { j } + P _ { j } + \rho _ { j } ) . \quad } \end{array}\tag{4.16}
$$

The rest of the proof uses $p _ { t } = x _ { t } + b _ { t }$ in the same order as Lemmas 3.4 and 3.5. It gives, for every $j ,$

$$
\alpha _ { j } \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } \left( \varphi _ { 2 } ( \ell _ { t , I _ { t - 1 } } ) - \varphi _ { 2 } ( \ell _ { t , I _ { t } } ) \right) \right] \leq 2 0 0 0 \eta _ { j } ^ { 2 } R _ { j } + \mathcal { O } \left( \eta _ { j } ^ { 2 } P _ { j } + \rho _ { j } \right) .\tag{4.17}
$$

The full proof of (4.17) is deferred to Appendix A. The $\mathcal { O } ( \rho _ { j } )$ additional term bounds the initialization term and the single stopping-boundary term of phase j.

It remains to control $D _ { t }$ . Since $\lambda _ { t } = \alpha _ { j } ( 2 - c _ { t - 1 } ) \leq 2 \alpha _ { j } .$ the modified sampling rule satisfies $p _ { t , i } \geq$ $x _ { t , i } / ( 1 + 2 \alpha _ { j } )$ . Applying the direct calculation in (3.18) with this lower bound gives

$$
D _ { t } \leq \frac { ( 1 + 2 \alpha _ { j } ) ^ { 2 } } { 2 \left( 1 - \eta _ { j } ( 1 + 2 \alpha _ { j } ) \right) } \eta _ { j } r _ { t } ^ { 2 } \leq 0 . 6 \eta _ { j } r _ { t } ^ { 2 } ,\tag{4.18}
$$

where the last inequality follows from $\alpha _ { j } = 8 \eta _ { j }$ and $\eta _ { j } \leq 1 0 ^ { - 5 }$

Multiplying (4.9) by $\chi _ { j , t }$ , taking expectations, and summing over t now give

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } ( D _ { t } + B _ { t } ) \right] \leq \left( - 3 . 4 \eta _ { j } + 2 0 0 0 \eta _ { j } ^ { 2 } \right) R _ { j } + \mathcal { O } \left( \eta _ { j } P _ { j } + \rho _ { j } \right) .
$$

Since $2 0 0 0 \eta _ { j } \le 0 . 0 2$ , the coeficient of $R _ { j }$ is negative. Dropping this term proves (4.6), which is exactly the phasewise counterpart of (3.42). □

We are now ready to prove Theorem 4.2.

Proof of Theorem 4.2. Fix an arbitrary arm $i \in [ K ]$ . On every realization for which phase j is reached, its rounds form a contiguous interval. Sum (3.2) over this interval and compare with the smoothed version of $e _ { i }$ used in Section 3. The OMD divergences telescope pathwise. Since the phase starts from the uniform distribution, the comparator term is at most $A _ { K } / \eta _ { j }$ on the event that the phase is reached. Moreover, $p _ { s _ { j } } = x _ { s _ { j } }$ , so the initial round has no sampling bias, and its remaining OMD and smoothing terms are at most 3. Lemma 4.4 therefore gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbf { 1 } ( \mathsf { J } _ { t } = j ) ( \ell _ { t , { I _ { i } } } - \ell _ { t , i } ) \right] \leq \underbrace { \frac { A _ { K } \rho _ { j } } { \eta _ { j } } } _ { \mathrm { c o m p a r a t o r ~ t e r m } } + \underbrace { \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } ( D _ { t } + B _ { t } ) \right] } _ { \mathrm { n o n i n i t i a l ~ r o n m o l s } } + 3 \rho _ { j } \leq \frac { A _ { K } \rho _ { j } } { \eta _ { j } } + \mathcal { O } \left( \eta _ { j } P _ { j } + \rho _ { j } \right) .\tag{4.19}
$$

It remains to sum this bound over the phases. From Lemma 4.3 and $H _ { j } = 4 A _ { K } / \eta _ { j } ^ { 2 }$ , we know that

$$
\eta _ { j } P _ { j } \leq \sqrt { 8 A _ { K } \rho _ { j } P _ { j } } .\tag{4.20}
$$

The lower bound in the same lemma also gives $4 A _ { K } \rho _ { j + 1 } / \eta _ { j } ^ { 2 } \ \leq \ P _ { j }$ Since $\eta _ { j + 1 } ~ = ~ \eta _ { j } / 2$ , we obtain $A _ { K } \rho _ { j + 1 } / \eta _ { j + 1 } \leq \eta _ { j } P _ { j } / 2$ . Thus the movement term of phase j also controls the comparator term of phase $j + 1$

Next, we show that the number of phases is logarithmic. Indeed, Lemma 4.1 gives $\widehat { v } _ { t } ^ { 2 } \leq 1 / ( 4 \eta _ { j } )$ during phase j. A completed phase must cross $H _ { j } = 4 A _ { K } / \eta _ { j } ^ { 2 }$ and hence contains at least $1 6 A _ { K } / \eta _ { j }$ noninitial rounds. Let M denote the random number of phases reached. Pathwise, summing this lower bound over the first $M - 1$ completed phases gives $T \geq 1 6 A _ { K } ( 2 ^ { M - 1 } - 1 ) / \eta _ { 0 }$ , and therefore $M \leq 2 \log _ { 2 } ( 2 T )$ . Since $\begin{array} { r } { M = \sum _ { j = 0 } ^ { T - 1 } \mathbf { 1 } ( s _ { j } \leq T ) } \end{array}$ taking expectations yields

$$
\sum _ { j = 0 } ^ { T - 1 } \rho _ { j } = \mathbb { E } [ M ] \leq 2 \log _ { 2 } ( 2 T ) .\tag{4.21}
$$

Finally, the phase indicators partition the horizon. Summing (4.19), bounding every comparator term except the first one by the preceding-phase movement term as above, and using (4.20) yield

$$
\begin{array} { r l } & { \mathsf { R e } _ { T } ( i ) \le \displaystyle \frac { A _ { K } } { \eta _ { 0 } } + \mathcal { O } \left( \sum _ { j = 0 } ^ { T - 1 } \sqrt { A _ { K } \rho _ { j } P _ { j } } + \sum _ { j = 0 } ^ { T - 1 } \rho _ { j } \right) } \\ & { \qquad \le \displaystyle \frac { A _ { K } } { \eta _ { 0 } } + \mathcal { O } \left( \sqrt { A _ { K } \log ( T ) \sum _ { j = 0 } ^ { T - 1 } P _ { j } } + \log T \right) } \\ & { \qquad \le \mathcal { O } \left( K \log ( K T ) + \sqrt { K Q _ { \infty , 2 } \log ( K T ) \log T } \right) , } \end{array}
$$

where the second inequality uses Cauchy–Schwarz and the last one uses Lemma 4.3, $A _ { K } = 6 4 K \log ( K T )$ 2 and $\eta _ { 0 } = 1 0 ^ { - 5 }$ . The bound holds for every $i \in [ K ]$ . Taking the maximum over i proves (4.3). □

## 5 Conclusion

We study second-order path-length guarantees for adversarial multi-armed bandits under ordinary bandit feedback. Our main result shows that, when $Q _ { \infty , 2 }$ is known, the recent-arm optimistic log-barrier algorithm of Bubeck et al. [4] already achieves $O \left( K \log ( K T ) + \sqrt { K \log ( K T ) ( 1 + Q _ { \infty , 2 } ) } \right)$ expected regret against oblivious loss sequences, but requires a more refined analysis. We further remove the knowledge of $Q _ { \infty , 2 }$ through an adaptive restart scheme. Together with the $\Omega ( \sqrt { K Q _ { \infty , 2 } } )$ lower bound, these results establish the optimal dependence on the second-order path length up to logarithmic factors.

Several questions remain open. First, the additional $\sqrt { \log T }$ factor in the adaptive guarantee arises from the phase-based tuning argument, and it would be interesting to determine whether the oracle rate can be achieved without knowing $Q _ { \infty , 2 }$ . Second, our analysis relies essentially on the loss sequence being oblivious. Extending the second-order guarantee to adaptive adversaries would require a diferent treatment of the signed diference terms used in the proof. More broadly, it would be interesting to understand whether similar second-order path-length guarantees can be obtained in richer partial-information models, such as linear bandits or MDPs.

Acknowledgments. The authors used GPT-5.6 for assistance with writing and to explore proof strategies.   
The authors have checked the arguments and take full responsibility for the content of the paper.

## References

[1] Peter Auer, Nicol\`o Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1), 2002.

[2] S´ebastien Bubeck and Aleksandrs Slivkins. The best of both worlds: Stochastic and adversarial bandits. In Conference on Learning Theory, 2012.

[3] S´ebastien Bubeck, Michael B. Cohen, and Yuanzhi Li. Sparsity, variance and curvature in multi-armed bandits. In Algorithmic Learning Theory, 2018.

[4] S´ebastien Bubeck, Yuanzhi Li, Haipeng Luo, and Chen-Yu Wei. Improved path-length regret bounds for bandits. In Conference on Learning Theory, 2019.

[5] Liyu Chen, Haipeng Luo, and Chen-Yu Wei. Impossible tuning made possible: A new expert algorithm and its applications. In Conference on Learning Theory, 2021.

[6] Chao-Kai Chiang, Tianbao Yang, Chia-Jung Lee, Mehrdad Mahdavi, Chi-Jen Lu, Rong Jin, and Shenghuo Zhu. Online optimization with gradual variations. In Conference on Learning Theory, 2012.

[7] Elad Hazan and Satyen Kale. Better algorithms for benign bandits. Journal of Machine Learning Research, 12, 2011.

[8] Shinji Ito and Kei Takemura. Best-of-three-worlds linear bandit algorithm with variance-adaptive regret bounds. In Conference on Learning Theory, 2023.

[9] Shinji Ito, Taira Tsuchiya, and Junya Honda. Adversarially robust multi-armed bandit algorithm with variance-dependent regret bounds. In Conference on Learning Theory, 2022.

[10] Tiancheng Jin and Haipeng Luo. Simultaneously learning stochastic and adversarial episodic MDPs with known transition. In Advances in Neural Information Processing Systems, volume 33, 2020.

[11] Tiancheng Jin, Longbo Huang, and Haipeng Luo. The best of both worlds: stochastic and adversarial episodic mdps with unknown transition. Advances in Neural Information Processing Systems, 34, 2021.

[12] Tiancheng Jin, Junyan Liu, and Haipeng Luo. Improved best-of-both-worlds guarantees for multi-armed bandits: Ftrl with general regularizers and multiple optimal arms. Advances in Neural Information Processing Systems, 36, 2023.

[13] Joon Kwon and Vianney Perchet. Gains and losses are fundamentally diferent in regret minimization: The sparse case. Journal of Machine Learning Research, 17, 2016.

[14] Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, and Mengxiao Zhang. Bias no more: High-probability data-dependent regret bounds for adversarial bandits and MDPs. In Advances in Neural Information Processing Systems, volume 33, 2020.

[15] Chung-Wei Lee, Haipeng Luo, and Mengxiao Zhang. A closer look at small-loss bounds for bandits with graph feedback. In Conference on Learning Theory, 2020.

[16] Jongyeong Lee, Junya Honda, Shinji Ito, and Min-hwan Oh. Follow-the-perturbed-leader with fr´echettype tail distributions: Optimality in adversarial bandits and best-of-both-worlds. In Conference on Learning Theory, 2024.

[17] Thodoris Lykouris, Karthik Sridharan, and Eva Tardos. Small-loss bounds for online learning with<sup>´</sup> partial information. In Conference on Learning Theory, 2018.

[18] Gergely Neu. First-order regret bounds for combinatorial semi-bandits. In Conference on Learning Theory, 2015.

[19] Roman Pogodin and Tor Lattimore. On first-order bounds, variance and gap-dependent bounds for adversarial bandits. In Uncertainty in Artificial Intelligence Conference, volume 115 of Proceedings of Machine Learning Research. PMLR, 2020.

[20] Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Conference on Learning Theory, 2013.

[21] Jacob Steinhardt and Percy Liang. Adaptivity and optimism: An improved exponentiated gradient algorithm. In International Conference on Machine Learning, 2014.

[22] Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E. Schapire. Fast convergence of regularized learning in games. In Advances in Neural Information Processing Systems, volume 28, 2015.

[23] Dirk van der Hoeven, Ashok Cutkosky, and Haipeng Luo. Comparator-adaptive convex bandits. In Advances in Neural Information Processing Systems, volume 33, 2020.

[24] Chen-Yu Wei and Haipeng Luo. More adaptive algorithms for adversarial bandits. In Conference on Learning Theory, 2018.

[25] Hang Yu, Yu-Hu Yan, and Peng Zhao. Improved dimension dependence for bandit convex optimization with gradient variation. In International Conference on Machine Learning, 2026.

[26] Julian Zimmert and Yevgeny Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research, 22(28), 2021.

## A Omitted Details in Section 4

In this section, we show the omitted details in Section 4. Specifically, we provide the proof of (4.17), which is the key to control the regret in each phase $j .$ For notational convenience, since we will consider each phase index $j$ separately, in the proofs in this section, we will fix a deterministic phase label $j \in \{ 0 , \ldots , T - 1 \}$ and abbreviate

$$
\begin{array} { r l } & { \chi _ { t } = \chi _ { j , t } , \qquad \rho = \rho _ { j } , \qquad \eta = \eta _ { j } , \qquad \alpha = \alpha _ { j } , \qquad R = R _ { j } , \qquad P = P _ { j } , } \\ & { s = s _ { j } , \qquad \tau = \operatorname* { m a x } \{ u \in [ T ] : \mathrm { J } _ { u } = j \} , \qquad \bar { \chi } _ { t } = \mathbf { 1 } \{ \mathbf { J } _ { t - 1 } = j \} , \qquad \delta _ { t } = \bar { \chi } _ { t } - \chi _ { t } . } \end{array}
$$

Here $\tau = 0$ if phase $j$ is not reached. We also recall that $\phi _ { 2 , t , i } = \varphi _ { 2 } ( \ell _ { t , i } ) , b _ { t } = p _ { t } - x _ { t } , \alpha = 8 \eta$ , and $\eta \leq 1 0 ^ { - 5 }$ For the theorem statement, we still keep the index $j$ . The following lemma shows the bound on (4.17).

Lemma A.1. For every deterministic phase label $j \in \{ 0 , \ldots , T - 1 \}$ , Algorithm 2 satisfies

$$
\alpha _ { j } \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } \left( \varphi _ { 2 } ( \ell _ { t , I _ { t - 1 } } ) - \varphi _ { 2 } ( \ell _ { t , I _ { t } } ) \right) \right] \leq 2 0 0 0 \eta _ { j } ^ { 2 } R _ { j } + \mathcal { O } \left( \eta _ { j } ^ { 2 } P _ { j } + \rho _ { j } \right) .\tag{A.1}
$$

To prove this, following the analysis in Section $^ { 3 , }$ we need to first bound the summation of $\left. \phi _ { 2 , t } , b _ { t - 1 } - b _ { t } \right.$ within each phase. The following lemma shows that this term is bounded by

Lemma A.2. For every deterministic phase label $j \in \{ 0 , \ldots , T - 1 \}$ , Algorithm 2 satisfies

$$
\alpha _ { j } \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } \left. \phi _ { 2 , t } , b _ { t - 1 } - b _ { t } \right. \right] \leq 1 7 0 0 \eta _ { j } ^ { 2 } R _ { j } + 2 5 0 0 \eta _ { j } ^ { 2 } P _ { j } + 2 0 \rho _ { j } .\tag{A.2}
$$

Proof. If phase $j$ is not reached, the left-hand side of (A.2) is zero. On the event that it is reached, s and $\tau$ are its first and last rounds. Since $b _ { s } = 0$ , the left-hand side can be written as

$$
\sum _ { t = 2 } ^ { T } \chi _ { t } \left. \phi _ { 2 , t } , b _ { t - 1 } - b _ { t } \right. = - \left. \phi _ { 2 , \tau } , b _ { \tau } \right. + \sum _ { t = 2 } ^ { T - 1 } \chi _ { t + 1 } \left. \phi _ { 2 , t + 1 } - \phi _ { 2 , t } , b _ { t } \right. .
$$

Since $0 \leq \varphi _ { 2 } ( z ) \leq 3 / 2$ on [0, 1], we have $\| \phi _ { 2 , \tau } \| _ { \infty } \leq 3 / 2$ . Moreover, the sampling rule gives $\| b _ { \tau } \| _ { 1 } \leq 4 \alpha$ . Since phase $j$ is reached with probability $\rho ,$ we have

$$
\begin{array} { r l } & { \alpha \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \chi _ { t } \left. \phi _ { 2 , t } , b _ { t - 1 } - b _ { t } \right. \right] } \\ & { \le 6 \alpha ^ { 2 } \rho + \alpha \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T - 1 } \left( \chi _ { t + 1 } - \chi _ { t } \right) \left. \phi _ { 2 , t + 1 } - \phi _ { 2 , t } , b _ { t } \right. \right] + \alpha \left| \displaystyle \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } ( \phi _ { 2 , t + 1 , i } - \phi _ { 2 , t , i } ) \mathbb { E } [ \chi _ { t } b _ { t , i } ] \right| } \\ & { \le 1 2 \alpha ^ { 2 } \rho + \alpha \left| \displaystyle \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } ( \phi _ { 2 , t + 1 , i } - \phi _ { 2 , t , i } ) \mathbb { E } [ \chi _ { t } b _ { t , i } ] \right| , } \end{array}
$$

where the last inequality follows from the fact that $| \langle \phi _ { 2 , t + 1 } - \phi _ { 2 , t } , b _ { t } \rangle | \leq \| \phi _ { 2 , t + 1 } - \phi _ { 2 , t } \| _ { \infty } \| b _ { t } \| _ { 1 } \leq \frac { 3 } { 2 } \cdot 4 \alpha = 6 \alpha$ and

$$
\begin{array} { r l } { \displaystyle \left. \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } ( \chi _ { t + 1 } - \chi _ { t } ) \left. \phi _ { 2 , t + 1 } - \phi _ { 2 , t } , b _ { t } \right. \right] \right. = \displaystyle \left. \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \left. \phi _ { 2 , t + 1 } - \phi _ { 2 , t } , b _ { t } \right. \right] \right. } & { } \\ { \leq 6 \alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \right] \leq 6 \alpha \rho , } & { } \end{array}
$$

where the equality uses $b _ { s } = 0$ , and the last inequality follows from $\textstyle \sum _ { t = 2 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \leq \mathbf { 1 } \{ s \leq T \}$ . Further applying Cauchy–Schwarz inequality gives

$$
\begin{array} { r l } & { \displaystyle \alpha \left| \displaystyle \sum _ { t = 2 } ^ { | T - 1 } \sum _ { i = 1 } ^ { K } ( \phi _ { 2 , t + 1 , i } - \phi _ { 2 , t , i } ) \mathbb { E } [ \chi _ { t } b _ { t , i } ] \right| } \\ & { \le \alpha \left( \displaystyle \sum _ { t = 2 } ^ { T - 1 } \displaystyle \sum _ { i = 1 } ^ { K } \frac { \big ( \mathbb { E } [ \chi _ { t } b _ { t , i } ] \big ) ^ { 2 } } { \mathbb { E } \big [ \chi _ { t } \big ( \mathbf { 1 } \{ I _ { t } = i \} + \chi _ { t , i } \big ) \big ] } \right) ^ { 1 / 2 } \left( \displaystyle \sum _ { t = 2 } ^ { T - 1 } \displaystyle \sum _ { i = 1 } ^ { K } \mathbb { E } \big [ \chi _ { t } \big ( \mathbf { 1 } \{ I _ { t } = i \} + \chi _ { t , i } \big ) \big ] ( \phi _ { 2 , t + 1 , i } - \phi _ { 2 , t , i } ) ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \le 2 \displaystyle \sum _ { t = 2 } ^ { T - 1 } \displaystyle \sum _ { i = 1 } ^ { K } \frac { \big ( \mathbb { E } [ \chi _ { t } b _ { t , i } ] \big ) ^ { 2 } } { \mathbb { E } \big [ \chi _ { t } \big ( \mathbf { 1 } \{ I _ { t } = i \} + \chi _ { t , i } \big ) \big ] } + \displaystyle \frac { \alpha ^ { 2 } } { 8 } \displaystyle \sum _ { t = 2 } ^ { T - 1 } \displaystyle \sum _ { i = 1 } ^ { K } \mathbb { E } \big [ \chi _ { t } \big ( \mathbf { 1 } \{ I _ { t } = i \} + \chi _ { t , i } \big ) \big ] \big ( \phi _ { 2 , t + 1 , i } - \phi _ { 2 , t , i } \big ) ^ { 2 } . } \end{array}
$$

Because $\chi _ { t }$ is measurable before $I _ { t }$ is drawn, we have $\mathbb { E } [ \chi _ { t } b _ { t , i } ] = \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} - x _ { t , i } ) ]$ . Therefore, using Lemma $\mathrm { A . 6 } ,$ , the first term is bounded as

$$
2 \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \frac { \left( \mathbb { E } [ \chi _ { t } b _ { t , i } ] \right) ^ { 2 } } { \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ T _ { t } = i \} + x _ { t , i } ) ] } = 2 \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \frac { \left( \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} - x _ { t , i } ) ] \right) ^ { 2 } } { \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} + x _ { t , i } ) ] } \le 1 0 3 0 \eta ^ { 2 } R + 1 8 \alpha ^ { 2 } P + 6 8 0 \alpha \rho .
$$

Lemma $\mathrm { A . 7 }$ further bounds the second term as follows:

$$
\frac { \alpha ^ { 2 } } { 8 } \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t } ( { \mathbf { 1 } \{ I _ { t } = i \} } + x _ { t , i } ) ] ( \phi _ { 2 , t + 1 , i } - \phi _ { 2 , t , i } ) ^ { 2 } \leq \frac { \alpha ^ { 2 } } { 8 } \big ( 1 2 3 6 0 \eta ^ { 2 } R + 1 3 P + 2 0 0 \rho \big ) .
$$

Combining these two bounds, we have

$$
\begin{array} { r l } & { \alpha \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \chi _ { t } \left. \phi _ { 2 , t } , b _ { t - 1 } - b _ { t } \right. \right] \leq 1 2 \alpha ^ { 2 } \rho + 2 ( 5 1 5 \eta ^ { 2 } R + 9 \alpha ^ { 2 } P + 3 4 0 \alpha \rho ) + \displaystyle \frac { \alpha ^ { 2 } } { 8 } \left( 1 2 3 6 0 \eta ^ { 2 } R + 1 3 P + 2 0 0 \rho \right) } \\ & { \qquad \leq 1 7 0 0 \eta ^ { 2 } R + 2 5 0 0 \eta ^ { 2 } P + 2 0 \rho , } \end{array}
$$

where the last inequality uses $\alpha = 8 \eta$ and $\eta \leq 1 0 ^ { - 5 }$

Next, we control the summation of $\langle \phi _ { 2 , t } , x _ { t - 1 } - x _ { t } \rangle$ ⟩ within the phase.

Lemma A.3. For every deterministic phase label $j \in \{ 0 , \ldots , T - 1 \}$ , Algorithm 2 satisfies

$$
\alpha _ { j } \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { j , t } \left. \phi _ { 2 , t } , x _ { t - 1 } - x _ { t } \right. \right] \leq 1 0 0 \eta _ { j } ^ { 2 } R _ { j } + 2 0 \eta _ { j } ^ { 2 } P _ { j } + 4 \rho _ { j } .\tag{A.3}
$$

Proof. If phase $j$ is not reached, the left-hand side of (A.3) is zero. On $\{ \bar { \chi } _ { t } = 1 \}$ , let $\bar { x } _ { t }$ be the OMD update computed at the end of round $t - 1$ , before a possible phase reset; when $\bar { \chi } _ { t } = 0$ , set $\bar { x } _ { t } = x _ { t - 1 }$ . Thus ${ \bar { x } } _ { t } = x _ { t }$ whenever $\chi _ { t } = 1$ . Define

$$
S _ { x } \triangleq \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { t } \left. \phi _ { 2 , t } , x _ { t - 1 } - x _ { t } \right. \right]
$$

and

$$
\bar { S } _ { x } \triangleq \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } \left. \phi _ { 2 , t } , x _ { t - 1 } - \bar { x } _ { t } \right. \right] .
$$

Since $\bar { \chi } _ { t } = \chi _ { t } + \delta _ { t }$ and ${ \bar { x } } _ { t } = x _ { t }$ on $\{ \chi _ { t } = 1 \}$ , their exact diference is

$$
S _ { x } - \bar { S } _ { x } = - \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \delta _ { t } \left. \phi _ { 2 , t } , x _ { t - 1 } - \bar { x } _ { t } \right. \right] .
$$

Using $\begin{array} { r } { \| \phi _ { 2 , t } \| _ { \infty } \leq 3 / 2 , \mathbb { E } [ \sum _ { t = 2 } ^ { T } \delta _ { t } ] \leq \rho } \end{array}$ , we have

$$
| S _ { x } - \bar { S } _ { x } | \leq \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \delta _ { t } \| \phi _ { 2 , t } \| _ { \infty } \| x _ { t - 1 } - \bar { x } _ { t } \| _ { 1 } \right] \leq 3 \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \delta _ { t } \right] \leq 3 \rho .
$$

Following the decomposition (3.33)–(3.34), write $g _ { t } = \phi _ { 2 , t } - \phi _ { 2 , t - 1 }$ and define

$$
\bar { S } _ { x , 1 } \triangleq \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } \left. \phi _ { 2 , t - 1 } , x _ { t - 1 } - \bar { x } _ { t } \right. \right]
$$

and

$$
\bar { S } _ { x , 2 } \triangleq \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } \left. g _ { t } , x _ { t - 1 } - \bar { x } _ { t } \right. \right] .
$$

We have $\bar { S } _ { x } = \bar { S } _ { x , 1 } + \bar { S } _ { x , 2 }$

We first control $\hat { S } _ { x , 1 }$ by making the comparison with Section 3 explicit. Equation (3.20) is unchanged. In the passage from (3.26) to (3.27), replacing the 1-Lipschitz function $\varphi$ by the 2-Lipschitz function $\varphi _ { 2 }$ doubles the coeficient 2.2 to 4.4. Equation (3.31) is unchanged. The phase-sampler bound $p _ { t - 1 , i } \geq x _ { t - 1 , i } / ( 1 + 2 \alpha )$ and $\eta \leq 1 0 ^ { - 5 }$ also preserve the 0.01 coordinatewise stability used in (3.23). Therefore the final comparison used to prove (3.13) becomes

$$
\mathbb { E } _ { t - 1 } \big [ \langle \phi _ { 2 , t - 1 } , x _ { t - 1 } - \bar { x } _ { t } \rangle \big ] \leq \frac { 4 . 4 } { 0 . 9 4 } \mathbb { E } _ { t - 1 } \big [ D _ { t - 1 } + D _ { \Psi } ( \bar { x } _ { t } , x _ { t - 1 } ) \big ] \leq 1 6 \mathbb { E } _ { t - 1 } \big [ D _ { t - 1 } \big ] .
$$

Here the second inequality uses $D _ { \Psi } \big ( \bar { x } _ { t } , x _ { t - 1 } \big ) \leq 2 D _ { t - 1 }$ , as shown in the last step of the proof of (3.13). Further, since $\bar { \chi } _ { t } = \mathbf { 1 } \{ \mathsf { J } _ { t - 1 } = j \}$ is $\mathcal { F } _ { t - 2 }$ -measurable, the tower property and (4.18) give

$$
\begin{array} { r l } & { \tilde { S } _ { z , 1 } = \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \tilde { \chi } _ { t } \mathbb { E } _ { t - 1 } \left[ \langle \phi _ { 2 , t - 1 } , x _ { t - 1 } - \bar { x } _ { t } \rangle \right] \right] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad  \end{array}\tag{using (4.18)}
$$

We next control $\bar { S } _ { x , 2 }$ . Set $\boldsymbol { q } _ { t } \triangleq \left. \boldsymbol { x } _ { t - 1 } , g _ { t } ^ { \odot 2 } \right.$ . Applying (3.12) on round t − 1 with $z = g _ { t }$ , and then applying (4.18), gives

$$
| \langle g _ { t } , x _ { t - 1 } - \bar { x } _ { t } \rangle | \leq 1 . 5 \sqrt { \eta D _ { t - 1 } } \sqrt { q _ { t } } \leq 1 . 5 \sqrt { 0 . 6 } \eta | r _ { t - 1 } | \sqrt { q _ { t } } \leq 1 . 2 \eta | r _ { t - 1 } | \sqrt { q _ { t } } .
$$

Therefore, we have

$$
\begin{array} { r l r } {  { | \bar { S } _ { x , 2 } | \le 1 . 2 \eta \mathbb { E } [ \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } | r _ { t - 1 } | \sqrt { q _ { t } } ] } } \\ & { } & { \le 1 . 2 \eta \sqrt { \mathbb { E } [ \displaystyle \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } r _ { t - 1 } ^ { 2 } ] \mathbb { E } [ \displaystyle \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } q _ { t } ] } } \\ & { } & { \le 1 . 2 \eta \sqrt { ( R + \rho ) \mathbb { E } [ \displaystyle \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } q _ { t } ] } . } \end{array}\tag{Cauchy–Schwarz}
$$

(A.4)

The 2-Lipschitz property of $\varphi _ { 2 }$ gives $| g _ { t , i } | = | \varphi _ { 2 } ( \ell _ { t , i } ) - \varphi _ { 2 } ( \ell _ { t - 1 , i } ) | \leq 2 | \ell _ { t , i } - \ell _ { t - 1 , i } |$ and $g _ { t , i } ^ { 2 } \leq 4 ( \ell _ { t , i } -$ $\ell _ { t - 1 , i } ) ^ { 2 }$ . Together with $x _ { t - 1 , i } \leq ( 1 + 2 \alpha ) p _ { t - 1 , i }$ , this yields

$$
q _ { t } = \langle x _ { t - 1 } , g _ { t } ^ { \odot 2 } \rangle = \sum _ { i = 1 } ^ { K } x _ { t - 1 } , i g _ { t , i } ^ { 2 } \le 4 \sum _ { i = 1 } ^ { K } x _ { t - 1 , i } ( \ell _ { t , i } - \ell _ { t - 1 , i } ) ^ { 2 } \le 4 ( 1 + 2 \alpha ) \sum _ { i = 1 } ^ { K } p _ { t - 1 , i } ( \ell _ { t , i } - \ell _ { t - 1 , i } ) ^ { 2 } .
$$

Since $\bar { \chi } _ { t }$ is $\mathcal { F } _ { t - 2 } .$ -measurable, we have

$$
\begin{array} { r l r } { \mathbb { E } \left[ \displaystyle \sum _ { t = - 2 } ^ { T } \bar { \chi } \tau \theta t \right] \leq 4 ( 1 + 2 \alpha ) \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t - 1 } ^ { K } , \boldsymbol { \mathcal { E } } ( t _ { t , t } - \ell _ { t - 1 , t - 1 } ) ^ { 2 } \right] } \\ & { = 4 ( 1 + 2 \alpha ) \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } \mathbb { E } _ { t - 1 } \left[ ( \ell _ { t , t _ { t - 1 } } - \ell _ { t - 1 , t _ { t - 1 } } ) ^ { 2 } \right] \right] } \\ & { = 4 ( 1 + 2 \alpha ) \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \bar { \chi } _ { t } ( \ell _ { t , t _ { t - 1 } } - \ell _ { t - 1 , t _ { t - 1 } } ) ^ { 2 } \right] } & { \quad \mathrm { ( ~ \bar { \chi } _ { t } ~ i s ~ \mathcal { F } _ { t - 2 } - m e a s u r a b l e ) } } \\ & { = 4 ( 1 + 2 \alpha ) \left( P + \mathbb { E } \left[ \displaystyle \sum _ { t = 2 } ^ { T } \delta _ { t } ( \ell _ { t , t _ { t - 1 } } - \ell _ { t - 1 , t _ { t - 1 } } ) ^ { 2 } \right] \right) } \\ & { \leq 4 ( 1 + 2 \alpha ) ( P + \rho ) \left( \rho + \rho \right) } & { \quad \mathrm { ( ~ \bar { \chi } _ { t } ~ > \rho ~ \bar { \chi } _ { t - 1 } ~ > \rho ~ \bar { \chi } _ { t - 1 } ~ ) ^ { 2 } } } \\ & { \leq 5 ( P + \rho ) . } \end{array}
$$

Combining the above inequalities together, we obtain that

$$
S _ { x } \le \bar { S } _ { x } + 3 \rho = \bar { S } _ { x , 1 } + \bar { S } _ { x , 2 } + 3 \rho \le 9 . 6 \eta ( R + \rho ) + 1 . 2 \eta \sqrt { 5 ( R + \rho ) ( P + \rho ) } + 3 \rho .\tag{A.6}
$$

Since $\alpha = 8 \eta$ , the AM-GM inequality gives

$$
9 . 6 \sqrt { 5 } \eta ^ { 2 } \sqrt { ( R + \rho ) ( P + \rho ) } \leq 4 . 8 \sqrt { 5 } \eta ^ { 2 } ( R + P + 2 \rho ) \leq 1 0 . 8 \eta ^ { 2 } ( R + P + 2 \rho ) .
$$

Further using the fact that $\eta \leq 1 0 ^ { - 5 }$ gives

$$
\alpha S _ { x } \leq 7 6 . 8 \eta ^ { 2 } ( R + \rho ) + 1 0 . 8 \eta ^ { 2 } ( R + P + 2 \rho ) + 2 4 \eta \rho \leq 1 0 0 \eta ^ { 2 } R + 2 0 \eta ^ { 2 } P + 4 \rho ,
$$

Now we are ready to prove Lemma A.1.

Proof of Lemma A.1. By the definitions of $\bar { \chi } _ { t }$ and $\delta _ { t } , { \bar { \chi } } _ { t } = \chi _ { t } + \delta _ { t }$ . The indicator $\delta _ { t }$ marks the unique round immediately after phase $j$ ends. Therefore we have E $\left\lceil \sum _ { t = 2 } ^ { T } \delta _ { t } \right\rceil \leq \rho$

Next, as $\bar { \chi } _ { t }$ is measurable before $I _ { t - 1 }$ is drawn, but $\chi _ { t }$ is not, the two conditional-expectation identities are $\mathbb { E } [ \bar { \chi } _ { t } \varphi _ { 2 } ( \ell _ { t , I _ { t - 1 } } ) ] = \mathbb { E } [ \bar { \chi } _ { t } \left. \phi _ { 2 , t } , p _ { t - 1 } \right. ]$ ] and $\mathbb { E } [ \chi _ { t } \varphi _ { 2 } ( \ell _ { t , I _ { t } } ) ] = \mathbb { E } [ \chi _ { t } \left. \phi _ { 2 , t } , p _ { t } \right. ]$ . Further using $\bar { \chi } _ { t } = \chi _ { t } + \delta _ { t }$ gives

$$
\alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { t } \left( \varphi _ { 2 } ( \ell _ { t , t _ { i - 1 } } ) - \varphi _ { 2 } ( \ell _ { t , t _ { i } } ) \right) \right] = \alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { t } \left. \phi _ { 2 , t } , p _ { t - 1 } - p _ { t } \right. \right] + \alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \delta _ { t } \left( \left. \phi _ { 2 , t } , p _ { t - 1 } \right. - \varphi _ { 2 } ( \ell _ { t , t _ { i - 1 } } ) \right) \right] .\tag{A.7}
$$

Thus (A.7) is the stopped counterpart of (3.10): the second term is precisely the boundary residual created when the predictable indicator $\bar { \chi } _ { t }$ is replaced by χ<sub>t</sub>. Since $0 \leq \varphi _ { 2 } ( z ) \leq 3 / 2$ on [0, 1], the absolute value of its summand is at most $3 \delta _ { t } / 2$ , and $\mathbb { E } [ \sum _ { t = 2 } ^ { T } \delta _ { t } ] \le \rho$ bounds the entire residual by 3αρ/2. Substituting $p _ { t } = x _ { t } + b _ { t }$ therefore gives

$$
\alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { t } \left( \varphi _ { 2 } ( \ell _ { t , t _ { t - 1 } } ) - \varphi _ { 2 } ( \ell _ { t , t _ { t } } ) \right) \right] \leq \alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { t } \left. \phi _ { 2 , t } , x _ { t - 1 } - x _ { t } \right. \right] + \alpha \mathbb { E } \left[ \sum _ { t = 2 } ^ { T } \chi _ { t } \left. \phi _ { 2 , t } , b _ { t - 1 } - b _ { t } \right. \right] + \frac { 3 } { 2 } \alpha \rho .\tag{A.8}
$$

Lemma A.3 bounds the first term on the right-hand side of (A.8), and Lemma A.2 bounds the second. Their sum is at most $1 8 0 0 \eta ^ { 2 } R + 2 5 2 0 \eta ^ { 2 } P + ( 2 4 + 3 \alpha / 2 ) \rho$ , which is bounded by the right-hand side of (A.1). This proves the lemma. □

## A.1 Auxiliary Lemmas

The following auxiliary lemmas are used in the proof of Lemma A.1. Throughout this subsection, E denotes expectation with respect to all randomness generated by Algorithm 2, with the oblivious loss sequence fixed.

The first lemma records the properties of the ratio functional used in the proof of Lemma A.2.

Lemma A.4. For any jointly distributed random arm $I \in [ K ]$ , random probability vector $x \in \Delta _ { K }$ , and random weight $w \in [ 0 , 1 ]$ , define

$$
\mathcal { N } ( w ; I , x ) \triangleq \sum _ { i = 1 } ^ { K } \frac { \big ( \mathbb { E } [ w ( \mathbf { 1 } \{ I = i \} - x _ { i } ) ] \big ) ^ { 2 } } { \mathbb { E } [ w ( \mathbf { 1 } \{ I = i \} + x _ { i } ) ] } .\tag{A.9}
$$

$I f 0 \leq w ^ { \prime } \leq w \leq 1$ , then we have $\mathcal { N } ( w ^ { \prime } ; I , x ) \leq \mathcal { N } ( w ; I , x ) + 6 \mathbb { E } [ w - w ^ { \prime } ]$ and $\mathcal { N } ( w ; I , x ) \leq 2 \mathbb { E } [ w ]$ . Moreover, whenever $w _ { 1 } + w _ { 2 } \leq 1$ , we have $\mathcal { N } ( w _ { 1 } + w _ { 2 } ; I , x ) \le \mathcal { N } ( w _ { 1 } ; I , x ) + \mathcal { N } ( w _ { 2 } ; I , x )$

Proof. For every nonnegative random variable u, define $\mu _ { i } ( u ) \triangleq \mathbb { E } [ u ( { \bf 1 } \{ I = { \} } \mathrm { - } x _ { i } ) ]$ and $\nu _ { i } ( u ) \triangleq \mathbb { E } [ u ( \mathbf { 1 } \{ I =$ $i \} + x _ { i } ) ]$ . The pointwise inequality $| \mathbf { 1 } \{ I = i \} - x _ { i } | \leq \mathbf { 1 } \{ I = i \} + x _ { i }$ gives

$$
| \mu _ { i } ( u ) | \leq \mathbb { E } [ u | \mathbf { 1 } \{ I = i \} - x _ { i } | ] \leq \nu _ { i } ( u ) .
$$

Direct calculation shows that $\mu _ { i } ( w ) = \mu _ { i } ( w ^ { \prime } ) + \mu _ { i } ( w - w ^ { \prime } )$ and $\nu _ { i } ( w ) = \nu _ { i } ( w ^ { \prime } ) + \nu _ { i } ( w - w ^ { \prime } ) . { \mathrm { ~ I f ~ } } \nu _ { i } ( w ^ { \prime } ) = 0$ then $\mu _ { i } ( w ^ { \prime } ) = 0$ and for any $w \geq w ^ { \prime }$

$$
\frac { \mu _ { i } ( w ^ { \prime } ) ^ { 2 } } { \nu _ { i } ( w ^ { \prime } ) } - \frac { \mu _ { i } ( w ) ^ { 2 } } { \nu _ { i } ( w ) } = - \frac { \mu _ { i } ( w ) ^ { 2 } } { \nu _ { i } ( w ) } \leq 0 \leq 3 \nu _ { i } ( w - w ^ { \prime } ) .
$$

If $\nu _ { i } ( w ^ { \prime } ) > 0$ , then we have

$$
\begin{array} { c l } { \displaystyle \frac { \mu _ { i } ( w ^ { \prime } ) ^ { 2 } } { \nu _ { i } ( w ^ { \prime } ) } - \frac { \mu _ { i } ( w ) ^ { 2 } } { \nu _ { i } ( w ) } = \frac { \nu _ { i } ( w - w ^ { \prime } ) \mu _ { i } ( w ^ { \prime } ) ^ { 2 } - 2 \nu _ { i } ( w ^ { \prime } ) \mu _ { i } ( w ^ { \prime } ) \mu _ { i } ( w - w ^ { \prime } ) - \nu _ { i } ( w ^ { \prime } ) \mu _ { i } ( w - w ^ { \prime } ) ^ { 2 } } { \nu _ { i } ( w ^ { \prime } ) \nu _ { i } ( w ) } } \\ { \displaystyle \qquad \leq \frac { \nu _ { i } ( w - w ^ { \prime } ) | \mu _ { i } ( w ^ { \prime } ) | ^ { 2 } + 2 \nu _ { i } ( w ^ { \prime } ) | \mu _ { i } ( w ^ { \prime } ) | | \mu _ { i } ( w - w ^ { \prime } ) | } { \nu _ { i } ( w ^ { \prime } ) \nu _ { i } ( w ) } } \\ { \displaystyle \qquad \leq \frac { 3 \nu _ { i } ( w ^ { \prime } ) ^ { 2 } \nu _ { i } ( w - w ^ { \prime } ) } { \nu _ { i } ( w ^ { \prime } ) \nu _ { i } ( w ) } \leq \frac { 3 \nu _ { i } ( w ^ { \prime } ) ^ { 2 } \nu _ { i } ( w - w ^ { \prime } ) } { \nu _ { i } ( w ^ { \prime } ) ^ { 2 } } = 3 \nu _ { i } ( w - w ^ { \prime } ) . } \end{array}
$$

Summing over i gives $\begin{array} { r } {  { \mathcal { N } } ( w ^ { \prime } ; I , x ) -  { \mathcal { N } } ( w ; I , x ) \leq 3 \sum _ { i = 1 } ^ { K } \nu _ { i } ( w - w ^ { \prime } ) = 6  { \mathbb { E } } [ w - w ^ { \prime } ] } \end{array}$ . The second inequality follows from

$$
\mathcal { N } ( w ; I , x ) = \sum _ { i = 1 } ^ { K } \frac { \mu _ { i } ( w ) ^ { 2 } } { \nu _ { i } ( w ) } \leq \sum _ { i = 1 } ^ { K } \nu _ { i } ( w ) = 2 \mathbb { E } [ w ] ,
$$

where we use the fact that $x \in \Delta _ { K }$ . Finally, Cauchy-Schwarz inequality leads to the following

$$
\mathcal { N } ( w _ { 1 } + w _ { 2 } ; I , x ) = \sum _ { i = 1 } ^ { K } \frac { ( \mu _ { i } ( w _ { 1 } ) + \mu _ { i } ( w _ { 2 } ) ) ^ { 2 } } { \nu _ { i } ( w _ { 1 } ) + \nu _ { i } ( w _ { 2 } ) } \leq \sum _ { i = 1 } ^ { K } \left( \frac { \mu _ { i } ( w _ { 1 } ) ^ { 2 } } { \nu _ { i } ( w _ { 1 } ) } + \frac { \mu _ { i } ( w _ { 2 } ) ^ { 2 } } { \nu _ { i } ( w _ { 2 } ) } \right) = \mathcal { N } ( w _ { 1 } ; I , x ) + \mathcal { N } ( w _ { 2 } ; I , x ) .
$$

Define $\mathcal { N } _ { t } ^ { \mathrm { c u r } } \triangleq \mathcal { N } ( \chi _ { t } ; I _ { t } , x _ { t } )$ and $\mathcal { N } _ { t } ^ { \mathrm { p r e v } } \triangleq \mathcal { N } ( \chi _ { t } ; I _ { t - 1 } , x _ { t } )$ . Since $\chi _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable, we have

$$
\mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} - x _ { t , i } ) ] = \mathbb { E } \left[ \chi _ { t } \mathbb { E } _ { t } [ { \bf 1 } \{ I _ { t } = i \} - x _ { t , i } ] \right] = \mathbb { E } [ \chi _ { t } ( p _ { t , i } - x _ { t , i } ) ] = \mathbb { E } [ \chi _ { t } b _ { t , i } ] .
$$

The following lemma bounds the current squared mean norm in terms of its previous-arm counterpart. Lemma A.5. For every fixed round $t \in \{ 2 , \ldots , T \}$ ,

$$
\mathcal { N } _ { t } ^ { \mathrm { c u r } } \leq 1 3 \alpha \mathcal { N } _ { t } ^ { \mathrm { p r e v } } + 8 \alpha ^ { 2 } \mathbb { E } [ \chi _ { t } ( r _ { t } ^ { 2 } + v _ { t } ^ { 2 } ) ] .\tag{A.10}
$$

Proof. On the event $\{ \chi _ { t } = 1 \}$ , the sampling rule in Algorithm 2 gives

$$
\begin{array} { r l } & { \lambda _ { t } = \alpha ( 2 - c _ { t - 1 } ) = \alpha ( 2 - \ell _ { t - 1 , I _ { t - 1 } } ) , } \\ & { p _ { t , i } = \frac { x _ { t , i } + \lambda _ { t } \mathbf { 1 } \left\{ I _ { t - 1 } = i \right\} } { 1 + \lambda _ { t } } = x _ { t , i } + \frac { \alpha ( 2 - \ell _ { t - 1 , I _ { t - 1 } } ) } { 1 + \alpha ( 2 - \ell _ { t - 1 , I _ { t - 1 } } ) } \bigl ( \mathbf { 1 } \{ I _ { t - 1 } = i \} - x _ { t , i } \bigr ) . } \end{array}
$$

Define $\gamma _ { t , i } \triangleq \alpha ( 2 - \ell _ { t - 1 , i } ) / ( 1 + \alpha ( 2 - \ell _ { t - 1 , i } ) )$ . Thus,

$$
\begin{array} { r l } & { p _ { t , i } - x _ { t , i } = \gamma _ { t , I _ { t - 1 } } \big ( \mathbf { 1 } \{ I _ { t - 1 } = i \} - x _ { t , i } \big ) , } \\ & { p _ { t , i } + x _ { t , i } = ( 2 - \gamma _ { t , I _ { t - 1 } } ) x _ { t , i } + \gamma _ { t , I _ { t - 1 } } \mathbf { 1 } \{ I _ { t - 1 } = i \} . } \end{array}
$$

Since $\chi _ { t }$ is measurable before $I _ { t }$ is drawn, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} - x _ { t , i } ) ] = \mathbb { E } [ \chi _ { t } ( p _ { t , i } - x _ { t , i } ) ] = \mathbb { E } \left[ \chi _ { t } \gamma _ { t , I _ { t - 1 } } \big ( { \bf 1 } \{ I _ { t - 1 } = i \} - x _ { t , i } \big ) \right] , } \\ & { \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} + x _ { t , i } ) ] = \mathbb { E } [ \chi _ { t } ( p _ { t , i } + x _ { t , i } ) ] = \mathbb { E } \left[ \chi _ { t } \left( ( 2 - \gamma _ { t , I _ { t - 1 } } ) x _ { t , i } + \gamma _ { t , I _ { t - 1 } } { \bf 1 } \{ I _ { t - 1 } = i \} \right) \right] . } \end{array}
$$

Therefore, by the definition of $\mathcal { N } _ { t } ^ { \mathrm { c u r } }$

$$
\mathcal { N } _ { t } ^ { \mathrm { c u r } } = \sum _ { i = 1 } ^ { K } \frac { \left( \mathbb { E } \left[ \chi _ { t } \gamma _ { t , I _ { t - 1 } } ( \mathbf { 1 } \{ I _ { t - 1 } = i \} - x _ { t , i } ) \right] \right) ^ { 2 } } { \mathbb { E } \left[ \chi _ { t } \left( ( 2 - \gamma _ { t , I _ { t - 1 } } ) x _ { t , i } + \gamma _ { t , I _ { t - 1 } } \mathbf { 1 } \{ I _ { t - 1 } = i \} \right) \right] } .
$$

We then decompose $\mathcal { N } _ { t } ^ { \mathrm { c u r } }$ using $( u + v ) ^ { 2 } \leq 2 u ^ { 2 } + 2 v ^ { 2 } $

$$
\mathcal { N } _ { t } ^ { \mathrm { { e u r } } } \leq 2 \underbrace { \sum _ { i = 1 } ^ { K } \frac { \gamma _ { t , i } ^ { 2 } \left( \mathbb { E } [ \chi _ { t } ( \mathbf { 1 } \{ I _ { t - 1 } = i \} - x _ { t , i } ) ] \right) ^ { 2 } } { \mathbb { E } \left[ \chi _ { t } \left( \left( 2 - \gamma _ { t , t - 1 } \right) x _ { t , i } + \gamma _ { t , t _ { - 1 } } \mathbf { 1 } \{ I _ { t - 1 } = i \} \right) \right] } } _ { \mathrm { p r e s t o n s - a r m ~ c o n t r i h u t i o n } } + 2 \underbrace { \sum _ { i = 1 } ^ { K } \frac { \left( \mathbb { E } [ \chi _ { t } x _ { t , i } ( \gamma _ { t , i } - \gamma _ { t , t - 1 } ) ] \right) ^ { 2 } } { \mathbb { E } \left[ \chi _ { t } \left( \left( 2 - \gamma _ { t , t _ { - 1 } } \right) x _ { t , i } + \gamma _ { t , t _ { - 1 } } \mathbf { 1 } \{ I _ { t - 1 } = i \} \right) \right] } } _ { \mathrm { r e m a n i n d e r } } .
$$

We first bound the previous-arm contribution. Since $\begin{array} { r } { \mathbb { E } \left[ \chi _ { t } \left( ( 2 - \gamma _ { t , I _ { t - 1 } } ) x _ { t , i } + \gamma _ { t , I _ { t - 1 } } \mathbf { 1 } \{ I _ { t - 1 } = i \} \right) \right] \geq } \end{array}$ $\begin{array} { r } { \frac { \alpha } { 1 + \alpha } \mathbb { E } \big [ \chi _ { t } \big ( \mathbf { 1 } \{ I _ { t - 1 } = i \} + x _ { t , i } \big ) \big ] } \end{array}$ , and $\gamma _ { t , i } \leq 2 \alpha / ( 1 + 2 \alpha )$ , we have

$$
2 \sum _ { i = 1 } ^ { K } \frac { \gamma _ { t , i } ^ { 2 } \left( \mathbb { E } \left[ \chi _ { t } \left( 1 \{ I _ { t - 1 } = i \} - x _ { t , i } \right) \right] \right) ^ { 2 } } { \mathbb { E } \left[ \chi _ { t } \left( \left( 2 - \gamma _ { t , t - 1 } \right) x _ { t , i } + \gamma _ { t , I _ { t - 1 } } \mathbf { 1 } \{ I _ { t - 1 } = i \} \right) \right] } \leq 2 \frac { ( 2 \alpha / ( 1 + 2 \alpha ) ) ^ { 2 } } { \alpha / ( 1 + \alpha ) } \mathcal { N } _ { t } ^ { \mathrm { p r e v } } \leq 1 2 \alpha \mathcal { N } _ { t } ^ { \mathrm { p r e v } } .
$$

It remains to control the remainder. Since $\mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} + x _ { t , i } ) ] \ge \mathbb { E } [ \chi _ { t } x _ { t , i } ]$ , Cauchy–Schwarz inequality gives

$$
\begin{array} { r l } { 2 \displaystyle \sum _ { i = 1 } ^ { K } \frac { \big ( \mathbb { E } [ \chi _ { t } x _ { t , i } ( \gamma _ { t , i } - \gamma _ { t , I _ { t - 1 } } ) ] \big ) ^ { 2 } } { \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} + x _ { t , i } ) ] } \leq 2 \displaystyle \sum _ { i = 1 } ^ { K } \frac { \big ( \mathbb { E } [ \chi _ { t } x _ { t , i } ( \gamma _ { t , i } - \gamma _ { t , I _ { t - 1 } } ) ] \big ) ^ { 2 } } { \mathbb { E } [ \chi _ { t } x _ { t , i } ] } } & { } \\ & { \leq 2 \mathbb { E } \left[ \chi _ { t } \displaystyle \sum _ { i = 1 } ^ { K } x _ { t , i } ( \gamma _ { t , i } - \gamma _ { t , I _ { t - 1 } } ) ^ { 2 } \right] } \\ & { \leq 2 \alpha ^ { 2 } \mathbb { E } \left[ \chi _ { t } \displaystyle \sum _ { i = 1 } ^ { K } x _ { t , i } ( \ell _ { t - 1 , i } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } \right] . } \end{array}
$$

Combining the two parts yields

$$
\mathcal { N } _ { t } ^ { \mathrm { c u r } } \leq 1 2 \alpha \mathcal { N } _ { t } ^ { \mathrm { p r e v } } + 2 \alpha ^ { 2 } \mathbb { E } \left[ \chi _ { t } \sum _ { i = 1 } ^ { K } x _ { t , i } ( \ell _ { t - 1 , i } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } \right] .\tag{A.11}
$$

To bound the remaining loss discrepancy, conditionally on $\mathcal { F } _ { t - 1 }$ draw $I _ { t } ^ { \prime } \sim x _ { t }$ and realize the sampling rule by setting $I _ { t } = I _ { t - 1 }$ with probability $\gamma _ { t , I _ { t - } }$ and $I _ { t } = I _ { t } ^ { \prime }$ otherwise. Since $\gamma _ { t , I _ { t - 1 } } \leq 2 \alpha$ , we have

$$
\begin{array} { r } { \mathbb { E } [ \chi _ { t } r _ { t } ^ { 2 } ] = \mathbb { E } \left[ \chi _ { t } \gamma _ { t , I _ { t - 1 } } v _ { t } ^ { 2 } + \chi _ { t } ( 1 - \gamma _ { t , I _ { t - 1 } } ) ( \ell _ { t , I _ { t } ^ { \prime } } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } \right] \geq ( 1 - 2 \alpha ) \mathbb { E } [ \chi _ { t } ( \ell _ { t , I _ { t } ^ { \prime } } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } ] . } \end{array}
$$

Applying Proposition A.8 with $A = \mathbb { E } [ \chi _ { t } x _ { t , i } ]$ and $B = \mathbb { E } [ \chi _ { t } { \bf 1 } \{ I _ { t - 1 } = i \} ]$ ], multiplying the resulting inequality by $( \ell _ { t , i } - \ell _ { t - 1 , i } ) ^ { 2 }$ , and summing over i give

$$
\begin{array} { r l } {  { \mathbb { E } [ \chi _ { t } ( \ell _ { t , I _ { t } ^ { \prime } } - \ell _ { t - 1 , I _ { t } ^ { \prime } } ) ^ { 2 } ] = \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t } x x _ { t , i } ] ( \ell _ { t , i } - \ell _ { t - 1 , i } ) ^ { 2 } } } \\ & { \leq \sum _ { i = 1 } ^ { K } \Bigg ( 2 \mathbb { E } [ \chi _ { t } \mathbf { 1 } \{ I _ { t - 1 } = i \} ] + 6 \frac { ( \mathbb { E } [ \chi _ { t } ( x _ { t , i } - \mathbf { 1 } \{ I _ { t - 1 } = i \} ) ] ) ^ { 2 } } { \mathbb { E } [ \chi _ { t } ( x _ { t , i } + \mathbf { 1 } \{ I _ { t - 1 } = i \} ) ] } \Bigg ) ( \ell _ { t , i } - \ell _ { t - 1 , i } ) ^ { 2 } } \\ & { \leq 2 \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t } \mathbf { 1 } \{ I _ { t - 1 } = i \} ] ( \ell _ { t , i } - \ell _ { t - 1 , i } ) ^ { 2 } + 6 \sum _ { i = 1 } ^ { K } \frac { ( \mathbb { E } [ \chi _ { t } ( \mathbf { 1 } \{ I _ { t - 1 } = i \} - x _ { t , i } ) ] ) ^ { 2 } } { \mathbb { E } [ \chi _ { t } ( \mathbf { 1 } \{ I _ { t - 1 } = i \} + x _ { t , i } ) ] } } \\ & { = 2 \mathbb { E } [ \chi _ { t } v _ { t } ^ { 2 } ] + 6 \mathcal { N } _ { t } ^ { \mathrm { p r e v } } , } \end{array}
$$

where the second inequality uses $( \ell _ { t , i } - \ell _ { t - 1 , i } ) ^ { 2 } \leq 1$ . Consequently, we can obtain that

$$
\begin{array} { r l r } {  { \mathbb E [ \chi _ { t } \sum _ { i = 1 } ^ { K } x _ { t , i } ( \ell _ { t - 1 , i } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } ] = \mathbb E [ \chi _ { t } ( \ell _ { t - 1 , I _ { t } ^ { \prime } } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } ] } } \\ & { } & { \leq 2 \mathbb E [ \chi _ { t } ( \ell _ { t , I _ { t } ^ { \prime } } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } ] + 2 \mathbb E [ \chi _ { t } ( \ell _ { t , I _ { t } ^ { \prime } } - \ell _ { t - 1 , I _ { t } ^ { \prime } } ) ^ { 2 } ] } \\ & { } & { \leq 3 \mathbb E [ \chi _ { t } r _ { t } ^ { 2 } ] + 4 \mathbb E [ \chi _ { t } v _ { t } ^ { 2 } ] + 1 2 N _ { t } ^ { \mathrm { p r e v } } . } \end{array}\tag{A.12}
$$

Here the last inequality uses $2 / ( 1 - 2 \alpha ) \leq 3$ . Substituting (A.12) into (A.11) gives

$$
\begin{array} { r } { \mathcal { N } _ { t } ^ { \mathrm { c u r } } \leq ( 1 2 \alpha + 2 4 \alpha ^ { 2 } ) \mathcal { N } _ { t } ^ { \mathrm { p r e v } } + 6 \alpha ^ { 2 } \mathbb { E } [ \chi _ { t } r _ { t } ^ { 2 } ] + 8 \alpha ^ { 2 } \mathbb { E } [ \chi _ { t } v _ { t } ^ { 2 } ] \leq 1 3 \alpha \mathcal { N } _ { t } ^ { \mathrm { p r e v } } + 8 \alpha ^ { 2 } \mathbb { E } [ \chi _ { t } ( r _ { t } ^ { 2 } + v _ { t } ^ { 2 } ) ] , } \end{array}
$$

where the last inequality uses $2 4 \alpha \leq 1$

The next lemma propagates the squared mean norm by one round and sums the result over the phase.

Lemma A.6. Algorithm 2 satisfies

$$
\sum _ { t = 2 } ^ { T } \mathcal { N } _ { t } ^ { \mathrm { p r e v } } \leq 3 \sum _ { t = 2 } ^ { T } \mathcal { N } _ { t } ^ { \mathrm { c u r } } + 3 \eta ^ { 2 } R + 2 5 \rho ,\tag{A.13}
$$

$$
\sum _ { t = 2 } ^ { T } \mathcal { N } _ { t } ^ { \mathrm { c u r } } \leq 5 1 5 \eta ^ { 2 } R + 9 \alpha ^ { 2 } P + 3 4 0 \alpha \rho .\tag{A.14}
$$

Proof. By definition, $\chi _ { t } = 1 \{ s < t \leq \tau \}$ if the phase is entered and otherwise, $\chi _ { t } = 0$ for every t. Therefore, pathwise, we have $\textstyle \sum _ { t = 1 } ^ { T - 1 } ( 1 - \chi _ { t } ) \chi _ { t + 1 } \leq \mathbf { 1 } \{ s \leq T \}$ and $\textstyle \sum _ { t = 1 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \leq \mathbf { 1 } \{ s \leq T \}$

Using $\chi _ { t + 1 } = \chi _ { t } \chi _ { t + 1 } + ( 1 - \chi _ { t } ) \chi _ { t + 1 }$ and $\chi _ { t } \chi _ { t + 1 } = \chi _ { t } - \chi _ { t } \big ( 1 - \chi _ { t + 1 } \big )$ , Lemma A.4 gives that

$$
\begin{array} { r l } & { \mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) \leq \mathcal { N } ( \chi _ { t } \chi _ { t + 1 } ; I _ { t } , x _ { t } ) + \mathcal { N } ( ( 1 - \chi _ { t } ) \chi _ { t + 1 } ; I _ { t } , x _ { t } ) } \\ & { \qquad \leq \mathcal { N } _ { t } ^ { \mathrm { c u r } } + 6 \mathbb { E } [ \chi _ { t } ( 1 - \chi _ { t + 1 } ) ] + 2 \mathbb { E } [ ( 1 - \chi _ { t } ) \chi _ { t + 1 } ] . } \end{array}\tag{A.15}
$$

We next compare $\mathscr { N } _ { t + 1 } ^ { \mathrm { p r e v } } = \mathscr { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t + 1 } )$ with $\mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } )$ . For every $i ,$

$$
\mathbb { E } [ \chi _ { t + 1 } ( \mathbf { 1 } \{ I _ { t } = i \} - x _ { t + 1 , i } ) ] = \mathbb { E } [ \chi _ { t + 1 } ( \mathbf { 1 } \{ I _ { t } = i \} - x _ { t , i } ) ] - \mathbb { E } [ \chi _ { t + 1 } ( x _ { t + 1 , i } - x _ { t , i } ) ] .
$$

On $\{ \chi _ { t + 1 } = 1 \}$ , applying (3.17) on round t gives $x _ { t + 1 , i } \geq 0 . 9 9 x _ { t , i }$ and hence

$$
\begin{array} { r } { \mathbb { E } \big [ \chi _ { t + 1 } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t + 1 , i } ) \big ] \geq 0 . 9 9 \mathbb { E } \big [ \chi _ { t + 1 } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t , i } ) \big ] . } \end{array}
$$

Therefore, applying AM-GM inequality and the above inequality give the following

$$
\mathcal { N } _ { t + 1 } ^ { \mathrm { p r e v } } \leq \frac { 2 } { 0 . 9 9 } \mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) + \frac { 2 } { 0 . 9 9 } \sum _ { i = 1 } ^ { K } \frac { \left( \mathbb { E } [ \chi _ { t + 1 } ( x _ { t + 1 , i } - x _ { t , i } ) ] \right) ^ { 2 } } { \mathbb { E } \left[ \chi _ { t + 1 } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t , i } ) \right] } .
$$

To further bound the second term, (3.19) applied on round t and Cauchy-Schwarz inequality yield

$$
\begin{array} { r l r } { \displaystyle \sum _ { i = 1 } ^ { K } \frac { \big ( \mathbb { E } \big [ \chi _ { t + 1 } ( x _ { t + 1 , i } - x _ { t , i } ) \big ] \big ) ^ { 2 } } { \mathbb { E } \big [ \chi _ { t + 1 } \big ( \mathbf { 1 } \big \{ I _ { t } = i \big \} + x _ { t , i } \big ) \big ] } \leq \displaystyle \sum _ { i = 1 } ^ { K } \frac { \big ( \mathbb { E } \big [ \chi _ { t + 1 } | x _ { t + 1 , i } - x _ { t , i } | \big ] \big ) ^ { 2 } } { \mathbb { E } \big [ \chi _ { t + 1 } x _ { t , i } \big ] } } \\ { \leq \displaystyle \sum _ { i = 1 } ^ { K } \mathbb { E } \bigg [ \chi _ { t + 1 } \frac { \big ( x _ { t + 1 , i } - x _ { t , i } \big ) ^ { 2 } } { x _ { t , i } } \bigg ] } & { } & { \mathrm { ( u s i n g ~ C a u c h y - S c h w a r z ~ i n e q u a l i t y ) } } \\ { \leq \displaystyle \mathbb { E } \bigg [ \chi _ { t + 1 } \sum _ { i = 1 } ^ { K } \frac { \big ( x _ { t + 1 , i } - x _ { t , i } \big ) ^ { 2 } } { x _ { t , i } ^ { 2 } } \bigg ] } \\ { \leq 2 . 1 \eta \mathbb { E } \big [ \chi _ { t + 1 } D _ { t } \big ] . } \end{array}
$$

Combining the above two inequalities gives

$$
\begin{array} { r } { \mathcal { N } _ { t + 1 } ^ { \mathrm { p r e v } } \leq 2 . 0 3 \mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) + 4 . 3 \eta \mathbb { E } [ \chi _ { t + 1 } D _ { t } ] . } \end{array}\tag{A.16}
$$

Moreover, since $\textstyle \sum _ { t = 1 } ^ { T - 1 } \chi _ { t + 1 } r _ { t } ^ { 2 } \leq { \bf 1 } \{ s \leq T \} + \sum _ { t = 2 } ^ { T } \chi _ { t } r _ { t } ^ { 2 } .$ , (4.18) applied on round t gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T - 1 } \chi _ { t + 1 } D _ { t } \right] \leq 0 . 6 \eta \mathbb { E } \left[ \sum _ { t = 1 } ^ { T - 1 } \chi _ { t + 1 } r _ { t } ^ { 2 } \right] \leq 0 . 6 \eta ( R + \rho ) .
$$

Summing (A.15) and (A.16) over $t = 1 , \dots , T - 1$ , using $\chi _ { 1 } = 0$ and the preceding bounds, yields

$$
\begin{array} { l } { \displaystyle \sum _ { t = 2 } ^ { T } { \mathcal { N } _ { t } ^ { \mathrm { p r e v } } \le 2 . 0 3 \sum _ { t = 1 } ^ { T - 1 } \mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) + 4 . 3 \eta \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T - 1 } \chi _ { t + 1 } D _ { t } \right] } } \\ { \displaystyle \qquad \le 2 . 0 3 \displaystyle \sum _ { t = 2 } ^ { T } \mathcal { N } _ { t } ^ { \mathrm { c u r } } + 1 2 . 1 8 \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \right] + 4 . 0 6 \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T - 1 } ( 1 - \chi _ { t } ) \chi _ { t + 1 } \right] + 4 . 3 \eta \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T - 1 } \chi _ { t + 1 } D _ { t } \right] } \\ { \displaystyle \qquad \le 2 . 0 3 \displaystyle \sum _ { t = 2 } ^ { T } \mathcal { N } _ { t } ^ { \mathrm { c u r } } + 1 6 . 2 4 \rho + 2 . 5 8 \eta ^ { 2 } ( R + \rho ) } \\ { \displaystyle \qquad \le 3 \displaystyle \sum _ { t = 2 } ^ { T } \mathcal { N } _ { t } ^ { \mathrm { c u r } } + 3 \eta ^ { 2 } R + 2 5 \rho , } \end{array}
$$

which proves (A.13).

Finally, summing (A.10) over t and applying (A.13) give

$$
\sum _ { t = 2 } ^ { T } { N _ { t } ^ { \mathrm { { c u r } } } } \le 1 3 \alpha \sum _ { t = 2 } ^ { T } { N _ { t } ^ { \mathrm { { p r e v } } } } + 8 \alpha ^ { 2 } ( R + P ) \le 3 9 \alpha \sum _ { t = 2 } ^ { T } { N _ { t } ^ { \mathrm { { c u r } } } } + ( 3 9 \alpha \eta ^ { 2 } + 8 \alpha ^ { 2 } ) R + 8 \alpha ^ { 2 } P + 3 2 5 \alpha \rho .
$$

Rearranging and using α = 8η and $\eta \leq 1 0 ^ { - 5 }$ yield

$$
\sum _ { t = 2 } ^ { T } { \mathcal { N } } _ { t } ^ { \mathrm { c u r } } \leq \frac { 3 9 \alpha \eta ^ { 2 } + 8 \alpha ^ { 2 } } { 1 - 3 9 \alpha } R + \frac { 8 \alpha ^ { 2 } } { 1 - 3 9 \alpha } P + \frac { 3 2 5 \alpha } { 1 - 3 9 \alpha } \rho \leq 5 1 5 \eta ^ { 2 } R + 9 \alpha ^ { 2 } P + 3 4 0 \alpha \rho ,
$$

which proves (A.14).

The following lemma bounds the weighted sum of squared diferences used after summation by parts. Lemma A.7. Let $g _ { t + 1 , i } \triangleq \phi _ { 2 , t + 1 , i } - \phi _ { 2 , t , i }$ and $q _ { t , i } \triangleq \mathbb { E } [ \chi _ { t } ( { \bf 1 } \{ I _ { t } = i \} + x _ { t , i } ) ]$ . Then

$$
\sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } q _ { t , i } g _ { t + 1 , i } ^ { 2 } \leq 1 2 3 6 0 \eta ^ { 2 } R + 1 3 P + 2 0 0 \rho .\tag{A.17}
$$

Proof. The inequalities $\chi _ { t } \leq \chi _ { t + 1 } + \chi _ { t } ( 1 - \chi _ { t + 1 } ) , | g _ { t + 1 , i } | \leq 3 / 2$ , and $\begin{array} { r } { \sum _ { i = 1 } ^ { K } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t , i } ) = 2 } \end{array}$ give

$$
\mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \sum _ { i = 1 } ^ { K } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t , i } ) g _ { t + 1 , i } ^ { 2 } \right] \leq \frac { 9 } { 2 } \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \right] \leq 7 \rho .
$$

By the definition of $q _ { t , i }$ and $\chi _ { t } \leq \chi _ { t + 1 } + \chi _ { t } \bigl ( 1 - \chi _ { t + 1 } \bigr )$ ，

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } q _ { t , i } q _ { t + 1 , i } ^ { 2 } = \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t , i } ) ] g _ { t + 1 , i } ^ { 2 } } \\ & { \displaystyle \qquad \leq \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t + 1 } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t , i } ) ] g _ { t + 1 , i } ^ { 2 } + \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } \chi _ { t } ( 1 - \chi _ { t + 1 } ) \sum _ { i = 1 } ^ { K } ( \mathbf { 1 } \{ I _ { t } = i \} + x _ { t , i } ) g _ { t + 1 , i } ^ { 2 } \right] } \\ & { \displaystyle \qquad \quad \overset { T - 1 } { \leq } \tau \rho + \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t + 1 } \mathbf { 1 } \{ I _ { t } = i \} ] g _ { t + 1 , i } ^ { 2 } + \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t + 1 } x _ { t , i } ] g _ { t + 1 , i } ^ { 2 } , } \end{array}
$$

Since $\varphi _ { 2 }$ is 2-Lipschitz, $g _ { t + 1 , i } ^ { 2 } \leq 4 ( \ell _ { t + 1 , i } - \ell _ { t , i } ) ^ { 2 }$ . Hence, we have

$$
\sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t + 1 } { \mathbf { 1 } \{ I _ { t } = i \} } ] g _ { t + 1 , i } ^ { 2 } \leq 4 \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } \chi _ { t + 1 } ( \ell _ { t + 1 , I _ { t } } - \ell _ { t , I _ { t } } ) ^ { 2 } \right] = 4 \mathbb { E } \left[ \sum _ { u = 3 } ^ { T } \chi _ { u } ( \ell _ { u , I _ { u - 1 } } - \ell _ { u - 1 , I _ { u - 1 } } ) ^ { 2 } \right] \leq 4 P .
$$

Applying Proposition A.8 with $A = \mathbb { E } [ \chi _ { t + 1 } x _ { t , i } ]$ and $B = \mathbb { E } [ \chi _ { t + 1 } \mathbf { 1 } \{ I _ { t } = i \} ]$ , and using $g _ { t + 1 , i } ^ { 2 } \leq 4 ( \ell _ { t + 1 , i } -$ $\ell _ { t , i } ) ^ { 2 } \leq 4$ , give

$$
\mathbb { E } [ \chi _ { t + 1 } x _ { t , i } ] g _ { t + 1 , i } ^ { 2 } \le 8 \mathbb { E } [ \chi _ { t + 1 } { \bf 1 } \{ I _ { t } = i \} ] ( \ell _ { t + 1 , i } - \ell _ { t , i } ) ^ { 2 } + 2 4 \frac { \big ( \mathbb { E } [ \chi _ { t + 1 } ( { \bf 1 } \{ I _ { t } = i \} - x _ { t , i } ) ] \big ) ^ { 2 } } { \mathbb { E } [ \chi _ { t + 1 } ( { \bf 1 } \{ I _ { t } = i \} + x _ { t , i } ) ] } .
$$

Summing over $t \in \{ 2 , \ldots , T - 1 \}$ and $i \in [ K ]$ yields

$$
\begin{array} { l } { \displaystyle \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } \mathbb { E } [ \chi _ { t + 1 } x _ { t , i } ] g _ { t + 1 , i } ^ { 2 } \leq 8 \mathbb { E } \left[ \displaystyle \sum _ { t = 3 } ^ { T } \chi _ { t } ( \ell _ { t , I _ { t - 1 } } - \ell _ { t - 1 , I _ { t - 1 } } ) ^ { 2 } \right] + 2 4 \displaystyle \sum _ { t = 2 } ^ { T - 1 } \mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) } \\ { \leq 8 P + 2 4 \displaystyle \sum _ { t = 2 } ^ { T - 1 } \mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) . } \end{array}
$$

Summing (A.15) over $t = 2 , \ldots , T - 1$ gives

$$
\sum _ { t = 2 } ^ { T - 1 } { N ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) \leq \sum _ { t = 2 } ^ { T - 1 } { N _ { t } ^ { \mathrm { c u r } } + 6 \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } { \chi _ { t } ( 1 - \chi _ { t + 1 } ) } \right] } } + 2 \mathbb { E } \left[ \sum _ { t = 2 } ^ { T - 1 } ( 1 - \chi _ { t } ) \chi _ { t + 1 } \right] \leq \sum _ { t = 2 } ^ { T - 1 } { N _ { t } ^ { \mathrm { c u r } } + 8 \rho } .
$$

Combining the preceding bounds and applying (A.14) yield

$$
\begin{array} { r l r } {  { \sum _ { t = 2 } ^ { T - 1 } \sum _ { i = 1 } ^ { K } q _ { t , i } g _ { t + 1 , i } ^ { 2 } \le 7 \rho + 4 P + 8 P + 2 4 \sum _ { t = 2 } ^ { T - 1 } \mathcal { N } ( \chi _ { t + 1 } ; I _ { t } , x _ { t } ) } } \\ & { } & \\ & { } & { \le 1 2 P + 2 4 \sum _ { t = 2 } ^ { T - 1 } \mathcal { N } _ { t } ^ { \mathrm { c u r } } + 1 9 9 \rho } \\ & { } & \\ & { } & { \le 1 2 3 6 0 \eta ^ { 2 } R + ( 1 2 + 2 1 6 \alpha ^ { 2 } ) P + ( 1 9 9 + 8 1 6 0 \alpha ) \rho } \\ & { } & { \le 1 2 3 6 0 \eta ^ { 2 } R + 1 3 P + 2 0 0 \rho , } \end{array}
$$

where $\alpha = 8 \eta \leq 8 \cdot 1 0 ^ { - 5 }$ implies $2 1 6 \alpha ^ { 2 } \leq 1$ and $8 1 6 0 \alpha \leq 1$

Proposition A.8. For any $A , B \geq 0$ , we have $\begin{array} { r } { A \le 2 B + 6 \frac { ( A - B ) ^ { 2 } } { A + B } } \end{array}$

Proof. The inequality is immediate when $A \leq 2 B$ . When $A > 2 B ,$ , we have $\begin{array} { r } { 6 \frac { ( A - B ) ^ { 2 } } { A + B } \geq 6 \frac { ( A / 2 ) ^ { 2 } } { 3 A / 2 } = A . } \end{array}$ □