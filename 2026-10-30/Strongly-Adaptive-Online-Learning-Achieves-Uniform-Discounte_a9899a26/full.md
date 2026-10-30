# Strongly Adaptive Online Learning Achieves Uniform Discounted Regret

Sifan Yang Wenhao Yang Lijun Zhang

yangsf@lamda.nju.edu.cn

yangwh@lamda.nju.edu.cn

zhanglj@lamda.nju.edu.cn

State Key Laboratory of Novel Software Technology, Nanjing University, Nanjing, China School of Artificial Intelligence, Nanjing University, Nanjing, China

## Abstract

Discounted regret measures the performance of an online learner by assigning exponentially decaying weights to past losses, reflecting the greater importance of recent losses in nonstationary online learning. While previous work obtains uniform discounted regret bounds over a continuous range of discount factors using a specialized meta-algorithm, its results are limited to convex functions. In this paper, we demonstrate that strongly adaptive regret bounds can directly imply uniform discounted regret guarantees for convex, strongly convex, and exp-concave loss functions. Our key observation is that discounted regret against a fixed comparator admits an exact decomposition into a convex combination of interval regrets. For convex functions, this reduction yields an $O ( \sqrt { \log T / ( 1 - \lambda ) } )$ ) discounted regret bound for all discount factors $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ , matching the previous rate over a broader range of discount factors. For strongly convex and exp-concave functions, we provide regret bounds of $\begin{array} { r } { O \left( \log T \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) \right) } \end{array}$ .

Keywords: online convex optimization, discounted regret, strongly adaptive regret, adap tive regret, exp-concave functions

## 1 Introduction

Online convex optimization (OCO) is a classic framework for sequential decision making (Zinkevich, 2003; Hazan, 2016), where a learner chooses a decision $\mathbf { w } _ { t }$ from a convex domain $\mathcal { W } \subseteq \mathbb { R } ^ { d }$ in each round $t ~ \in ~ [ T ]$ and then sufers a convex loss $f _ { t } ( \mathbf { w } _ { t } )$ selected by the environment. The performance of the learner is commonly measured by the static regret (Zinkevich, 2003), which compares its cumulative loss with that of the best fixed decision in hindsight. In non-stationary environments, recent losses are often more relevant than earlier ones, whereas static regret assigns equal weight to all rounds. To account for this diference, the λ-discounted regret (Cesa-Bianchi and Lugosi, 2006; Freund and Hsu, 2008) assigns a weight of $\lambda ^ { T - t }$ to the loss incurred at round t and is defined as

$$
\mathrm { D - R e g r e t } ( T , \lambda ) = \sum _ { t = 1 } ^ { T } \lambda ^ { T - t } f _ { t } ( \mathbf { w } _ { t } ) - \operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } } \sum _ { t = 1 } ^ { T } \lambda ^ { T - t } f _ { t } ( \mathbf { w } ) ,
$$

where $\lambda \in ( 0 , 1 )$ controls the rate of discounting. When $\lambda = 1 ,$ , discounted regret reduces to static regret. Discounted regret has been studied in adaptive regularization (Zhang et al., 2024), the analysis of Adam (Ahn et al., 2024), and online linear regression (Jacobsen and

Cutkosky, 2024). For a given discount factor, online gradient descent (OGD) with a step size proportional to $\sqrt { 1 - \lambda }$ achieves an $O ( 1 / \sqrt { 1 - \lambda } )$ discounted regret bound. Yang et al. (2026) further establish an $O ( \sqrt { \log T / ( 1 - \lambda ) } )$ discounted regret bound simultaneously for all discount factors in a continuous interval using a single algorithm.

The appropriate discount factor depends on how fast the environment changes, which is rarely known in advance. It is desirable to obtain discounted regret bounds simultaneously for all discount factors using a single algorithm. Directly aggregating OGD instances tuned to diferent discount factors is challenging, since each instance is analyzed under its own discounted regret, whereas standard meta-algorithms such as Hedge compare experts under a common performance measure. SOGD (Yang et al., 2026) addresses this challenge by aggregating OGD instances tuned to a geometric grid of discount factors through the Discounted-Normal-Predictor with conservative updating (DNP-cu) (Kapralov and Panigrahy, 2010), a carefully designed meta-algorithm. Discounted regret assigns larger weights to more recent rounds, and regret over sufix intervals captures the learner’s performance on these rounds. For any interval $[ r , s ] \subseteq [ T ]$ , the interval regret is defined as

$$
\mathrm { R e g r e t } ( [ r , s ] ) = \sum _ { t = r } ^ { s } f _ { t } ( \mathbf { w } _ { t } ) - \operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } } \sum _ { t = r } ^ { s } f _ { t } ( \mathbf { w } ) .
$$

Adaptive regret (Hazan and Seshadhri, 2007) measures the maximum regret over all contiguous intervals, while strongly adaptive regret (Daniely et al., 2015) additionally accounts for the interval length τ and is defined as

$$
\operatorname { S A - R e g r e t } ( T , \tau ) = \operatorname* { m a x } _ { 1 \leq r \leq T - \tau + 1 } \operatorname { R e g r e t } ( [ r , r + \tau - 1 ] ) .
$$

Discounted regret assigns larger weights to more recent rounds, while strongly adaptive regret controls regret over any contiguous interval, including intervals ending at the final round T. This paper investigates whether interval regret guarantees can establish uniform discounted regret bounds.

For any nonincreasing discount sequence, the discounted regret against a fixed comparator can be expressed as a convex combination of regrets over intervals of $[ r , T ]$ . Since strongly adaptive regret controls the regret over every interval, this decomposition provides a direct reduction from strongly adaptive regret to discounted regret. In particular, if an algorithm admits a strongly adaptive regret bound $\psi _ { T } ( \tau )$ that is concave in the interval length τ , it achieves

$$
\begin{array} { r } { \mathrm { D - R e g r e t } ( T , \lambda ) \le \psi _ { T } ( \tau _ { \lambda } ) , } \end{array}
$$

simultaneously for all $\lambda \in ( 0 , 1 )$ , where $\textstyle \tau _ { \lambda } = \sum _ { k = 0 } ^ { T - 1 } \lambda ^ { k } \leq 1 / ( 1 - \lambda )$

Existing strongly adaptive algorithms achieve uniform discounted regret bounds through this reduction, as summarized in Table 1. For convex functions, CBCE (Jun et al., 2017) achieves an $O ( \sqrt { \log T / ( 1 - \lambda ) } )$ discounted regret bound for all $\lambda \in ( 0 , 1 )$ , matching the rate of SOGD (Yang et al., 2026) over a broader range of discount factors. For strongly convex and exp-concave functions, IFLH (Zhang et al., 2018) and AFLH (Hazan and Seshadhri, 2007) achieve $\begin{array} { r } { O \left( \log T \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) \right) } \end{array}$ discounted regret bounds. These algorithms maintain only $O ( \log t )$ experts at round t, while FLH (Hazan and Seshadhri, 2007) achieves sharper regret bounds at the cost of t active experts.

Table 1: Uniform discounted regret bounds for strongly adaptive algorithms. All bounds hold simultaneously for $\lambda \in ( 0 , 1 )$
<table><tr><td>Functions</td><td>Algorithm</td><td> $\mathrm { D - R e g r e t } ( T , \lambda )$ </td><td>Active experts</td></tr><tr><td>Convex</td><td>CBCE (Jun et al., 2017)</td><td> $O \left( { \sqrt { \frac { \log T } { 1 - \lambda } } } \right)$ </td><td> $O ( \log ( t + 1 ) )$ </td></tr><tr><td>Strongly convex</td><td>IFLH (K = 2) (Zhang et al., 2018)</td><td> $\begin{array} { r } { O \left( \frac { \log T } { \mu } \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) \right) } \end{array}$ </td><td> $O ( \log ( t + 1 ) )$ </td></tr><tr><td>Strongly convex</td><td> $\mathrm { I F L H } \ ( K = \lceil T ^ { 1 / q } \rceil )$   $\mathrm { ( Z h a n g e t a l . , 2 0 1 8 ) }$ </td><td> $\begin{array} { r } { O \left( \frac { q \log T } { \mu } \right) } \end{array}$ </td><td> $O ( q T ^ { 1 / q } )$ </td></tr><tr><td>Exp-concave</td><td>AFLH (Hazan and Seshadhri, 2007)</td><td> $\begin{array} { r } { O \left( \frac { d \log T } { \alpha } \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) \right) } \end{array}$ </td><td> $O ( \log ( t + 1 ) )$ </td></tr><tr><td>Exp-concave</td><td>FLH (Hazan and Seshadhri, 2007)</td><td> $\begin{array} { r } { O \left( \frac { 1 } { \alpha } \left( d \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) + \log T \right) \right) } \end{array}$ </td><td>t</td></tr></table>

Here $q \geq 2$ is a fixed integer, and $T \geq 2 .$ . For exp-concave functions, we use a valid exp-concavity parameter satisfying $\begin{array} { r } { x G D \le 1 . } \end{array}$ . The last column reports the number of active experts at round t.

Organization. In the rest of the paper, Section 2 reviews the related work, Section 3 introduces the necessary assumptions, Section 4 establishes the reduction from strongly adaptive regret to discounted regret, Section 5 derives uniform discounted regret bounds for existing strongly adaptive methods, and Section 6 concludes this paper.

## 2 Related Work

## 2.1 Adaptive and Strongly Adaptive Regret

Static regret evaluates the performance of an online learner against a fixed decision over all rounds, which may fail to capture their performance in changing environments. To address this issue, Hazan and Seshadhri (2007) introduce adaptive regret, which measures the maximum static regret over all contiguous intervals $[ r , s ] \subseteq [ T ]$ . They establish an ${ \cal O } ( \log T )$ adaptive regret bound for exp-concave functions, but their algorithm requires T active experts. They further develop a more eficient algorithm at the cost of a worse regret bound.

Daniely et al. (2015) introduce strongly adaptive regret, which measures the maximum regret over intervals of a given length τ , rather than over all intervals as in adaptive regret. For convex functions, their algorithm achieves an $O ( { \sqrt { \tau } } \log T )$ regret bound. Jun et al. (2017) improve this bound to $O ( { \sqrt { \tau \log T } } )$ by using coin betting. For strongly convex and exp-concave functions, Zhang et al. (2018) establish logarithmic strongly adaptive regret bounds with explicit dependence on the interval length τ .

## 2.2 Discounted Online Learning

Discounted regret has been studied in prediction with expert advice. Cesa-Bianchi and Lugosi (2006) consider general nonincreasing discount sequences, while Freund and Hsu (2008) propose NormalHedge, which achieves an $O ( \sqrt { \log N / ( 1 - \lambda ) } )$ discounted regret bound against

N experts. In the field of OCO, Zhang et al. (2024) develop discounted FTRL with instanceadaptive regularization for time-varying discount factors, which are assumed to be known to the learner in advance.

Zhang et al. (2022) propose smoothed OGD (SOGD), which aggregates OGD experts using the Discounted-Normal-Predictor (DNP) (Kapralov and Panigrahy, 2010). Building on SOGD, Yang et al. (2026) obtain an $O ( \sqrt { \log T / ( 1 - \lambda ) } )$ discounted regret bound simultaneously over a continuous interval of discount factors without knowing λ in advance. However, their approach relies on a specialized meta-algorithm, and the regret guarantee is limited to convex functions.

## 3 Preliminaries

Notation. For integers $1 \leq r \leq s \leq T$ , we denote $[ r , s ] = \{ r , r + 1 , \ldots , s \}$ and $[ T ] = [ 1 , T ]$ We use $\| \cdot \| \ \mathrm { a n d } \ \langle \cdot , \cdot \rangle$ to denote the Euclidean norm and inner product, respectively.

We introduce the following standard assumptions (Zhang et al., 2022; Yang et al., 2026).

Assumption 1 Each loss function $f _ { t } ( \cdot )$ is convex on the feasible domain W.

Assumption 2 Each loss function $f _ { t } ( \cdot )$ is µ-strongly convex on the feasible domain W for some $\mu > 0$ , i.e., for any w, $\mathbf { w } ^ { \prime } \in \mathcal { W }$

$$
f _ { t } ( \mathbf { w } ^ { \prime } ) \geq f _ { t } ( \mathbf { w } ) + \langle \nabla f _ { t } ( \mathbf { w } ) , \mathbf { w } ^ { \prime } - \mathbf { w } \rangle + \frac { \mu } { 2 } \| \mathbf { w } ^ { \prime } - \mathbf { w } \| ^ { 2 } .
$$

Assumption 3 Each loss function $f _ { t } ( \cdot )$ is α-exp-concave on the feasible domain W for some $\alpha > 0 , i . e . , \exp ( - \alpha f _ { t } )$ is concave on the feasible domain W.

Assumption 4 The gradients are bounded by $G > 0$ , i.e., $\| \nabla f _ { t } ( \mathbf { w } ) \| \leq G$ for all $t \in [ T ]$ and $\mathbf { w } \in \mathcal { W }$

Assumption 5 The feasible domain $\mathcal { W } \subseteq \mathbb { R } ^ { d }$ is convex with diameter at most $D > 0$ , i.e., $\| \mathbf { w } - \mathbf { w } ^ { \prime } \| \leq D$ for all $\mathbf { w } , \mathbf { w } ^ { \prime } \in \mathcal { W }$

Without loss of generality, we further assume that the losses satisfy

$$
0 \leq f _ { t } ( \mathbf { w } ) \leq G D , \quad \forall t \in [ T ] , \ \mathbf { w } \in \mathcal { W } .
$$

Then we introduce the definitions of regrets. For any interval $[ r , s ] \subseteq [ T ]$ and comparator $\mathbf { w } \in \mathcal { W }$ , we define the interval regret as

$$
\operatorname { R e g r e t } _ { [ r , s ] } ( \mathbf { w } ) = \sum _ { t = r } ^ { s } { \big ( } f _ { t } ( \mathbf { w } _ { t } ) - f _ { t } ( \mathbf { w } ) { \big ) } , \qquad \operatorname { R e g r e t } ( [ r , s ] ) = \operatorname* { m a x } _ { \mathbf { w } \in \mathcal { W } } \operatorname { R e g r e t } _ { [ r , s ] } ( \mathbf { w } ) ,
$$

where the second quantity compares the learner with the best fixed decision on $[ r , s ]$ . Adaptive regret is the maximum of ${ \mathrm { R e g r e t } } ( [ r , s ] )$ over all intervals (Hazan and Seshadhri, 2007), which is defined as

$$
\operatorname { A - R e g r e t } ( T ) = \operatorname* { m a x } _ { 1 \leq r \leq s \leq T } \operatorname { R e g r e t } ( [ r , s ] ) .
$$

For a given interval length τ , strongly adaptive regret (Daniely et al., 2015) is defined as

$$
\operatorname { S A - R e g r e t } ( T , \tau ) = \operatorname* { m a x } _ { 1 \leq r \leq T - \tau + 1 } \operatorname { R e g r e t } ( [ r , r + \tau - 1 ] ) , \qquad \tau \in [ T ] .
$$

For discounted regret, we consider a nonincreasing discount sequence $\beta = ( \beta _ { 0 } , \dots , \beta _ { T - 1 } )$ satisfying

$$
1 = \beta _ { 0 } \geq \beta _ { 1 } \geq \cdot \cdot \cdot \geq \beta _ { T - 1 } \geq \beta _ { T } = 0 .\tag{1}
$$

The discounted regret against a fixed comparator $\mathbf { w } \in \mathcal { W }$ and the discounted regret of the learner are defined as

$$
\mathrm { D - R e g r e t } _ { T } ^ { \beta } ( \mathbf { w } ) = \sum _ { t = 1 } ^ { T } \beta _ { T - t } \big ( f _ { t } ( \mathbf { w } _ { t } ) - f _ { t } ( \mathbf { w } ) \big ) , \qquad \mathrm { D - R e g r e t } ( T , \beta ) = \operatorname* { m a x } _ { \mathbf { w } \in \mathcal { W } } \mathrm { D - R e g r e t } _ { T } ^ { \beta } ( \mathbf { w } ) .
$$

This definition covers the general discounted regret (Cesa-Bianchi and Lugosi, 2006). The special case $\beta _ { k } = \lambda ^ { k }$ implies the λ-discounted regret (Freund and Hsu, 2008), that is,

$$
\mathrm { D - R e g r e t } ( T , \lambda ) = \sum _ { t = 1 } ^ { T } \lambda ^ { T - t } f _ { t } ( \mathbf { w } _ { t } ) - \operatorname* { m i n } _ { \mathbf { w } \in \mathcal { W } } \sum _ { t = 1 } ^ { T } \lambda ^ { T - t } f _ { t } ( \mathbf { w } ) ,
$$

where $\lambda \in ( 0 , 1 )$ . In particular, $\lambda = 0$ corresponds to the regret at the last round, while $\lambda = 1$ recovers static regret.

We define the efective window size as the sum of the discount weights, i.e.,

$$
\tau _ { \beta } = \sum _ { k = 0 } ^ { T - 1 } \beta _ { k } , \qquad \tau _ { \lambda } = \sum _ { k = 0 } ^ { T - 1 } \lambda ^ { k } = \{ \frac { 1 - \lambda ^ { T } } { 1 - \lambda } , \ : \ : \lambda \in ( 0 , 1 ) , \ :
$$

By using (1), we have $1 \leq \tau _ { \beta } \leq T$ . For exponential discounting, we further have

$$
1 \leq \tau _ { \lambda } \leq \operatorname* { m i n } \left\{ T , { \frac { 1 } { 1 - \lambda } } \right\} , \qquad \lambda \in ( 0 , 1 ) .
$$

For subsequent analysis, suppose an online algorithm admits a strongly adaptive regret bound $\psi _ { T } ( \cdot ) : [ T ] \to \mathbb { R }$ satisfying

$$
\mathrm { S A - R e g r e t } ( T , \tau ) \le \psi _ { T } ( \tau ) , \qquad \forall \tau \in [ T ] .\tag{2}
$$

The algorithm may depend on $T _ { i }$ , but does not require the discount sequence $\beta$ as input.

## 4 From Strongly Adaptive Regret to Discounted Regret

Strongly adaptive regret controls the cumulative regret over every contiguous interval, whereas discounted regret assigns diferent weights to individual rounds. To establish a connection between these two measures, the key challenge is to relate the weighted regret to the interval regrets controlled by strongly adaptive regret. An important property of discounted regret is that more recent rounds receive larger weights, which motivates considering intervals of the form $[ r , T ] , r \in [ T ]$ . For any nonincreasing discount sequence, the discounted regret against a fixed comparator can be expressed as a convex combination of regrets over these intervals. Since strongly adaptive regret controls the regret on every interval, this decomposition provides a direct reduction from strongly adaptive regret to discounted regret.

Lemma 1 For any discount sequence $\beta$ satisfying (1) with $\beta _ { T } = 0$ , the discounted regret against any fixed comparator $\mathbf { w } \in \mathcal { W }$ satisfies

$$
\mathrm { D } \mathrm { - R e g r e t } _ { T } ^ { \beta } ( \mathbf { w } ) = \sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) \mathrm { R e g r e t } _ { [ T - \tau + 1 , T ] } ( \mathbf { w } ) .\tag{3}
$$

Moreover,

$$
\beta _ { \tau - 1 } - \beta _ { \tau } \geq 0 , \forall \tau \in [ T ] , \qquad \sum _ { \tau = 1 } ^ { T } \tau ( \beta _ { \tau - 1 } - \beta _ { \tau } ) = \tau _ { \beta } .\tag{4}
$$

Proof Since $\beta _ { T } = 0$ , for every $t \in [ T ]$ , we have

$$
\beta _ { T - t } = \beta _ { T - t } - \beta _ { T } = \sum _ { \tau = T - t + 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) .
$$

Substituting this identity into $\mathrm { D - R e g r e t } _ { T } ^ { \beta } ( \mathbf { w } )$ , we obtain

$$
\begin{array} { l } { \displaystyle \mathrm { D } \mathrm { - } \mathrm { R e g r e t } _ { T } ^ { \beta } ( \mathbf { w } ) = \displaystyle \sum _ { t = 1 } ^ { T } \beta _ { T - t } \big ( f _ { t } ( \mathbf { w } _ { t } ) - f _ { t } ( \mathbf { w } ) \big ) } \\ { = \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { \tau = T - t + 1 } ^ { T } \big ( \beta _ { \tau - 1 } - \beta _ { \tau } \big ) \big ( f _ { t } ( \mathbf { w } _ { t } ) - f _ { t } ( \mathbf { w } ) \big ) } \\ { = \displaystyle \sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) \sum _ { t = T - \tau + 1 } ^ { T } \big ( f _ { t } ( \mathbf { w } _ { t } ) - f _ { t } ( \mathbf { w } ) \big ) } \\ { = \displaystyle \sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) \mathrm { R e g r e t } _ { [ T - \tau + 1 , T ] } ( \mathbf { w } ) . } \end{array}
$$

Furthermore, we have

$$
\sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) = \beta _ { 0 } - \beta _ { T } = 1 .
$$

Finally, we obtain

$$
\begin{array} { l } { \displaystyle \sum _ { \tau = 1 } ^ { T } \tau ( \beta _ { \tau - 1 } - \beta _ { \tau } ) = \sum _ { \tau = 1 } ^ { T } \tau \beta _ { \tau - 1 } - \sum _ { \tau = 1 } ^ { T } \tau \beta _ { \tau } } \\ { = \beta _ { 0 } + \sum _ { \tau = 1 } ^ { T - 1 } ( \tau + 1 ) \beta _ { \tau } - \sum _ { \tau = 1 } ^ { T - 1 } \tau \beta _ { \tau } - T \beta _ { T } } \\ { = \sum _ { \tau = 0 } ^ { T - 1 } \beta _ { \tau } = \tau \beta . } \end{array}
$$

Lemma 1 establishes a connection between discounted regret and interval regret. Since each interval regret is bounded by strongly adaptive regret at the corresponding length, the decomposition allows us to bound discounted regret using strongly adaptive regret guarantees. Moreover, the weighted average of the interval lengths equals $\tau _ { \beta }$ , which enables a tighter bound when the strongly adaptive regret bound is concave. The following theorem formalizes this reduction.

Theorem 2 Suppose an online algorithm satisfies SA-Regret $( T , \tau ) \le \psi _ { T } ( \tau )$ for all $\tau \in [ T ]$ For any discount sequence $\beta$ satisfying (1), we have

$$
\operatorname { D - R e g r e t } ( T , \beta ) \leq \sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) \psi _ { T } ( \tau ) .\tag{5}
$$

If $\psi _ { T }$ admits a concave extension to $[ 1 , T ]$ , then

$$
\operatorname { D - R e g r e t } ( T , \beta ) \leq \psi _ { T } ( \tau _ { \beta } ) .\tag{6}
$$

Both bounds hold simultaneously for all discount sequences satisfying (1).

Proof For any $\mathbf { w } \in \mathcal { W }$ and $\tau \in [ T ]$ , we have

$$
\mathrm { R e g r e t } _ { [ T - \tau + 1 , T ] } ( \mathbf { w } ) \leq \mathrm { R e g r e t } ( [ T - \tau + 1 , T ] ) \leq \mathrm { S A } { \mathrm { - R e g r e t } } ( T , \tau ) \leq \psi _ { T } ( \tau ) .
$$

Combining Lemma 1 with the nonnegativity of $\beta _ { \tau - 1 } - \beta _ { \tau }$ , we can derive

$$
\begin{array} { r l } { \mathrm { D . R e g r e t } ( T , \beta ) = } & { \underset { \mathcal { C } = \mathcal { C } } { \operatorname* { m a x } } \frac { T } { \gamma } ( \beta _ { \mathcal { C } - 1 } - \beta _ { \mathcal { T } } ) \mathrm { R e g r e t } [ \eta _ { \mathcal { C } - \tau + 1 , \mathcal { T } } | \mathbf { w } ] } \\ & { \leq \underset { \mathcal { T } = 1 } { \overset { T } { \sum } } ( \beta _ { \mathcal { T } - 1 } - \beta _ { \mathcal { T } } ) \underset { \mathbf { w } \in \mathbb { Z } } { \operatorname* { m a x } } \mathrm { R e g r e t } [ \eta _ { \mathcal { C } - \tau + 1 , \mathcal { T } } | \mathbf { w } ] } \\ & { \leq \underset { \mathcal { T } = 1 } { \overset { T } { \sum } } ( \beta _ { \mathcal { T } - 1 } - \beta _ { \mathcal { T } } ) \underset { \mathbf { w } \in \mathbb { Z } } { \operatorname* { m a x } } \mathrm { R e g r e t } [ T - \tau + 1 , T ] [ \mathbf { w } ] } \\ & { = \underset { \mathcal { T } = 1 } { \overset { T } { \sum } } ( \beta _ { \mathcal { T } - 1 } - \beta _ { \mathcal { T } } ) \mathrm { R e g r e t } [ ( T - \tau + 1 , T ] ) } \\ & { \leq \underset { \mathcal { T } = 1 } { \overset { T } { \sum } } ( \beta _ { \mathcal { T } - 1 } - \beta _ { \mathcal { T } } ) \mathrm { S A . T e g r e t } ( T , \tau ) } \\ & { \leq \underset { \mathcal { T } = 1 } { \overset { T } { \sum } } ( \beta _ { \mathcal { T } - 1 } - \beta _ { \mathcal { T } } ) \forall \tau ( \tau ) . } \end{array}
$$

If $\psi _ { T }$ is concave on [1, T], Jensen’s inequality implies

$$
\begin{array} { r l } { \displaystyle \mathrm { D } \mathrm { - } \mathrm { R e g r e t } ( T , \beta ) \leq \sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) \psi _ { T } ( \tau ) } \\ { \leq \psi _ { T } \left( \displaystyle \sum _ { \tau = 1 } ^ { T } \tau ( \beta _ { \tau - 1 } - \beta _ { \tau } ) \right) } \\ { = \psi _ { T } ( \tau _ { \beta } ) , } \end{array}
$$

where the second inequality uses $\begin{array} { r } { \sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) = 1 } \end{array}$

Remark 3 The decomposition in Lemma 1 holds for a fixed comparator, whereas the maximization over comparators in discounted regret leads to the inequality in (5). More importantly, Theorem 2 transfers the strongly adaptive regret guarantee to all nonincreasing discount sequences simultaneously. This reduction does not require convexity or other assumptions on the loss functions, which are only needed to establish specific strongly adaptive regret bounds.

For exponential discounting, the discount sequence takes the form $\beta _ { k } = \lambda ^ { k }$ and we can derive the following corollary.

Corollary 4 Suppose an online algorithm satisfies SA-Regret $( T , \tau ) \le \psi _ { T } ( \tau )$ for all $\tau \in [ T ]$ Then, for all $\lambda \in ( 0 , 1 )$ simultaneously,

$$
\operatorname { D - R e g r e t } ( T , \lambda ) \leq ( 1 - \lambda ) \sum _ { \tau = 1 } ^ { T - 1 } \lambda ^ { \tau - 1 } \psi _ { T } ( \tau ) + \lambda ^ { T - 1 } \psi _ { T } ( T ) .
$$

If ψ<sub>T</sub> admits a concave extension to [1, T], then

$$
\begin{array} { r } { \mathrm { D - R e g r e t } ( T , \lambda ) \leq \psi _ { T } ( \tau _ { \lambda } ) . } \end{array}
$$

Proof For any $\lambda \in ( 0 , 1 )$ , setting $\beta _ { k } = \lambda ^ { k }$ for $k = 0 , \ldots , T - 1$ and $\beta _ { T } = 0$ , we have

$$
\begin{array} { r l } & { \beta _ { \tau - 1 } - \beta _ { \tau } = ( 1 - \lambda ) \lambda ^ { \tau - 1 } , \quad \tau \in [ T - 1 ] , } \\ & { } \\ & { \beta _ { T - 1 } - \beta _ { T } = \lambda ^ { T - 1 } . } \end{array}
$$

Substituting these expressions into (5), we obtain

$$
\begin{array} { r l r } & { } & { \mathrm { D - R e g r e t } ( T , \lambda ) = \mathrm { D - R e g r e t } ( T , \beta ) \leq \displaystyle \sum _ { \tau = 1 } ^ { T } ( \beta _ { \tau - 1 } - \beta _ { \tau } ) \psi _ { T } ( \tau ) } \\ & { } & { = ( 1 - \lambda ) \displaystyle \sum _ { \tau = 1 } ^ { T - 1 } \lambda ^ { \tau - 1 } \psi _ { T } ( \tau ) + \lambda ^ { T - 1 } \psi _ { T } ( T ) . } \end{array}
$$

If ψ<sub>T</sub> admits a concave extension to [1, T], we further have

$$
\operatorname { D - R e g r e t } ( T , \lambda ) \leq \psi _ { T } ( \tau _ { \beta } ) = \psi _ { T } \left( \sum _ { k = 0 } ^ { T - 1 } \lambda ^ { k } \right) = \psi _ { T } ( \tau _ { \lambda } ) .
$$

Theorem 2 and Corollary 4 establish a general reduction from strongly adaptive regret to discounted regret for nonincreasing discount sequences, including exponential discounting with any $\lambda \in \mathsf { \Gamma } ( 0 , 1 )$ . Since this reduction does not depend on the curvature of the loss functions, it can be applied to strongly adaptive regret bounds for diferent function classes. In the next section, we derive uniform discounted regret bounds for convex, strongly convex, and exp-concave functions.

## 5 Discounted Regret of Strongly Adaptive Methods

In this section, we apply our reduction to existing strongly adaptive algorithms and derive uniform discounted regret bounds for convex, strongly convex, and exp-concave functions, respectively.

## 5.1 Convex Functions

For convex functions, Jun et al. (2017) propose CBCE, which uses OGD as the base learner and maintains $O ( \log t )$ active experts at round t through geometric covering intervals. Under Assumptions 1, 4 and 5, their method achieves the following strongly adaptive regret bound for any interval $[ r , s ] \subseteq [ T ]$

$$
\mathrm { R e g r e t } ( [ r , s ] ) \leq G D \left( { \frac { 6 } { { \sqrt { 2 } } - 1 } } + 8 { \sqrt { 7 \log T + 5 } } \right) { \sqrt { s - r + 1 } } .
$$

Since the regret bound is concave in the interval length, Corollary 4 implies that, simultaneously for all $\lambda \in ( 0 , 1 )$ ,

$$
\mathrm { D - R e g r e t } ( T , \lambda ) \leq G D \left( \frac { 6 } { \sqrt { 2 } - 1 } + 8 \sqrt { 7 \log T + 5 } \right) \sqrt { \tau _ { \lambda } } = O \left( \sqrt { \tau _ { \lambda } \log T } \right) = O \left( G D \sqrt { \frac { \log T } { 1 - \lambda } } \right) ,
$$

which matches the regret bound of Yang et al. (2026) over the range of discount factors they consider and extends the guarantee to all $\lambda \in ( 0 , 1 )$

## 5.2 Strongly Convex Functions

For strongly convex functions, Zhang et al. (2018) propose improved following the leading history (IFLH), which uses OGD as the base learner and adopts a base-K lifetime schedule to control the number of active experts. For any integer $K \geq 2$ , IFLH maintains at most $( K - 1 ) ( \lfloor \log _ { K } t \rfloor + 1 )$ active experts at round t. Under Assumptions 2, 4, and 5, their algorithm achieves the following strongly adaptive regret bound:

$$
\mathrm { S A - R e g r e t } ( T , \tau ) \leq \frac { G ^ { 2 } } { 2 \mu } \left[ \left( 1 + 3 \log T \right) \left( \left\lceil \log _ { K } \tau \right\rceil + 1 \right) + 4 \log T \right] , \quad \forall \tau \in [ T ] .
$$

Since $\lceil \log _ { K } \tau \rceil + 1 \leq 2 + \log \tau / \log K$ , the above bound admits a concave upper bound in τ . Combining Corollary 4 with $\begin{array} { r } { \sum _ { k = 0 } ^ { \tilde { T } - 1 } \lambda ^ { k } \le 1 / ( 1 - \lambda ) } \end{array}$ , we obtain

$$
\mathrm { D \mathrm { - } R e g r e t } ( T , \lambda ) \leq \frac { G ^ { 2 } } { 2 \mu } \left[ 2 + 1 0 \log T + \frac { 1 + 3 \log T } { \log K } \log \frac { 1 } { 1 - \lambda } \right] ,
$$

simultaneously for all $\lambda \in ( 0 , 1 )$

The parameter K controls the trade-of between the regret bound and the number of active experts. When $K = 2$ , IFLH maintains $O ( \log t )$ active experts and achieves

$$
\mathrm { D - R e g r e t } ( T , \lambda ) = O \left( \frac { G ^ { 2 } } { \mu } \log T \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) \right) .
$$

Alternatively, for any fixed integer $q \geq 2 .$ , choosing $K = \lceil T ^ { 1 / q } \rceil$ ensures $\lceil \log _ { K } \tau \rceil \leq q$ for all $\tau \in [ T ]$ . The strongly adaptive regret bound of IFLH and Corollary 4 imply

$$
\mathrm { D - R e g r e t } ( T , \lambda ) \leq \frac { G ^ { 2 } } { 2 \mu } \left[ q + 1 + ( 3 q + 7 ) \log T \right] = O \left( \frac { q G ^ { 2 } } { \mu } \log T \right) ,
$$

with $O ( q T ^ { 1 / q } )$ active experts.

## 5.3 Exp-Concave Functions

For exp-concave functions, Hazan and Seshadhri (2007) develop FLH, which uses online Newton step (ONS) as the base learner and maintains t active experts at round t. Under Assumptions $3 , 4 ,$ and $5 ,$ let $\begin{array} { r } { \gamma = \frac { 1 } { 2 } \operatorname* { m i n } \{ \alpha , 1 / ( 4 G D ) \} } \end{array}$ . Combining the ONS regret bound with the FLH aggregation guarantee, we have

$$
\mathrm { S A - R e g r e t } ( T , \tau ) \leq \frac { \log T } { \alpha } + \frac { 1 } { 2 \gamma } \left[ 1 + d \log \left( 1 + \frac { \gamma ^ { 2 } G ^ { 2 } D ^ { 2 } } { d } \tau \right) \right] , \quad \forall \tau \in [ T ] .
$$

Since the right-hand side is concave in τ , Corollary 4 implies

$$
\begin{array} { r l } & { \mathrm { D } { \cdot } \mathrm { R e g r e t } ( T , \lambda ) \leq \displaystyle \frac { \log T } { \alpha } + \frac { 1 } { 2 \gamma } \left[ 1 + d \log \left( 1 + \frac { \gamma ^ { 2 } G ^ { 2 } D ^ { 2 } } { d ( 1 - \lambda ) } \right) \right] } \\ & { \quad \quad \quad = O \left( d \left( \frac { 1 } { \alpha } + G D \right) \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) + \frac { \log T } { \alpha } \right) , } \end{array}
$$

simultaneously for all $\lambda \in ( 0 , 1 )$

To reduce the computational cost of FLH, Hazan and Seshadhri (2007) further develop AFLH, which maintains $O ( \log t )$ active experts at the cost of a worse regret bound. With ONS as the base learner, AFLH achieves

$$
\mathrm { S A - R e g r e t } ( T , \tau ) = O \left( d \left( { \frac { 1 } { \alpha } } + G D \right) \log T ( 1 + \log \tau ) \right) .
$$

Applying Corollary 4, we obtain

$$
\mathrm { D - R e g r e t } ( T , \lambda ) = O \left( d \left( \frac { 1 } { \alpha } + G D \right) \log T \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) \right) ,
$$

simultaneously for all $\lambda \in ( 0 , 1 )$

To further control the trade-of between regret and computational cost, Zhang et al. (2018) develop IFLH with a base-K lifetime schedule. Using ONS as the base learner, IFLH maintains at most $( K - 1 ) ( \lfloor \log _ { K } t \rfloor + 1 )$ active experts at round t and achieves

$$
\mathrm { S A - R e g r e t } ( T , \tau ) \leq \left[ \frac { ( 5 d + 1 ) ( \lceil \log _ { K } \tau \rceil + 1 ) + 2 } { \alpha } + 5 d G D ( \lceil \log _ { K } \tau \rceil + 1 ) \right] \log T , \quad \forall \tau \in [ T ] .
$$

Since $\lceil \log _ { K } \tau \rceil + 1 \leq 2 + \log \tau / \log K$ , the above bound admits a concave upper bound in $\tau .$ Corollary 4 implies

$$
\mathrm { D - R e g r e t } ( T , \lambda ) \leq \left[ \left( \frac { 5 d + 1 } { \alpha } + 5 d G D \right) \left( 2 + \frac { \log \frac { 1 } { 1 - \lambda } } { \log K } \right) + \frac { 2 } { \alpha } \right] \log T ,
$$

simultaneously for all $\lambda \in ( 0 , 1 )$

When $K = 2$ , IFLH maintains $O ( \log t )$ active experts and achieves

$$
\mathrm { D - R e g r e t } ( T , \lambda ) = O \left( d \left( \frac { 1 } { \alpha } + G D \right) \log T \left( 1 + \log \frac { 1 } { 1 - \lambda } \right) \right) ,
$$

which matches the regret bound of AFLH. Alternatively, for any fixed integer $q \geq 2$ , choosing $K = \lceil T ^ { 1 / q } \rceil$ ensures $\lceil \log _ { K } \tau \rceil \leq q$ for all $\tau \in [ T ]$ . Combining the strongly adaptive regret bound of IFLH with Corollary 4, we obtain

$$
\begin{array} { r l } & { \mathrm { D - R e g r e t } ( T , \lambda ) \leq \left[ \frac { ( 5 d + 1 ) ( q + 1 ) + 2 } { \alpha } + 5 d ( q + 1 ) G D \right] \log T } \\ & { ~ = O \left( q d \left( \frac { 1 } { \alpha } + G D \right) \log T \right) , } \end{array}
$$

with $O ( q T ^ { 1 / q } )$ active experts.

Remark 5 For convex functions, CBCE matches the discounted regret bound of Yang et al. (2026) while extending the guarantee to all $\lambda \in ( 0 , 1 )$ . For strongly convex and exp-concave functions, IFLH with $K = 2$ and AFLH achieve uniform discounted regret bounds with $O ( \log t )$ active experts, whereas FLH improves the multiplicative logarithmic dependence to an additive dependence at the cost of t active experts. Moreover, choosing $K = \lceil T ^ { 1 / q } \rceil$ in IFLH yields regret bounds independent of λ, with $O ( q T ^ { 1 / q } )$ active experts.

## 6 Conclusion

This paper demonstrates that strongly adaptive regret implies discounted regret uniformly over all discount factors. The reduction rests on an exact decomposition of the discounted regret against a fixed comparator into its regrets over sufixes of the horizon, and the mean sufix length under exponential discounting coincides with the efective window size. The decomposition applies to the decisions of any strongly adaptive method without modification, and existing methods thereby achieve uniform discounted regret bounds for convex, strongly convex, and exp-concave functions with their original computational cost. For convex functions, the resulting order matches that of the dedicated construction of Yang et al. (2026) on the interval considered there and extends to all discount factors in (0, 1).

## Use of AI Tools

During the development of the results, GPT-6 Pro was employed to explore proof approaches and refine the paper, including its writing and organization. The authors have verified all statements and proofs and take full responsibility for the content of this paper.

## References

Kwangjun Ahn, Zhiyu Zhang, Yunbum Kook, and Yan Dai. Understanding Adam optimizer via online learning of updates: Adam is FTRL in disguise. In Proceedings of the 41st International Conference on Machine Learning, pages 619–640, 2024.

Nicol\`o Cesa-Bianchi and G´abor Lugosi. Prediction, Learning, and Games. Cambridge University Press, 2006.

Amit Daniely, Alon Gonen, and Shai Shalev-Shwartz. Strongly adaptive online learning. In Proceedings of the 32nd International Conference on Machine Learning, pages 1405–1411, 2015.

Yoav Freund and Daniel Hsu. A new Hedging algorithm and its application to inferring latent random variables. arXiv preprint arXiv:0806.4802, 2008.

Elad Hazan. Introduction to online convex optimization. Foundations and Trends in Optimization, 2(3-4):157–325, 2016.

Elad Hazan and C. Seshadhri. Adaptive algorithms for online decision problems. Technical Report TR07-088, Electronic Colloquium on Computational Complexity, 2007. The revised version is titled Adaptive Algorithms for Online Optimization.

Andrew Jacobsen and Ashok Cutkosky. Online linear regression in dynamic environments via discounting. In Proceedings of the 41st International Conference on Machine Learning, pages 21083–21120, 2024.

Kwang-Sung Jun, Francesco Orabona, Stephen Wright, and Rebecca Willett. Improved strongly adaptive online learning using coin betting. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pages 943–951, 2017.

Michael Kapralov and Rina Panigrahy. Prediction strategies without loss. arXiv preprint arXiv:1008.3672, 2010.

Wenhao Yang, Sifan Yang, and Lijun Zhang. Discounted online convex optimization: Uniform regret across a continuous interval. In The Fourteenth International Conference on Learning Representations, 2026.

Lijun Zhang, Tianbao Yang, Rong Jin, and Zhi-Hua Zhou. Dynamic regret of strongly adaptive methods. In Proceedings of the 35th International Conference on Machine Learning, pages 5882–5891, 2018.

Lijun Zhang, Wei Jiang, Jinfeng Yi, and Tianbao Yang. Smoothed online convex optimization based on Discounted-Normal-Predictor. In Advances in Neural Information Processing Systems 35, pages 4928–4942, 2022.

Zhiyu Zhang, David Bombara, and Heng Yang. Discounted adaptive online learning: Towards better regularization. In Proceedings of the 41st International Conference on Machine Learning, pages 58631–58661, 2024.

Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th International Conference on Machine Learning, pages 928–936, 2003.