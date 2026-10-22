# Private Online Prediction from Experts with Small Losses

Bo Li<sup>1,2</sup> Peng Ye<sup>2</sup>

<sup>1</sup>Guangzhou HKUST Fok Ying Tung Research Institute <sup>2</sup>Department of Computer Science and Engineering, HKUST bli@ust.hk, pyeac@connect.ust.hk

## Abstract

We consider online prediction from experts, a fundamental problem in machine learning, under differential privacy constraints. Existing private algorithms achieve near-optimal regret $\tilde { O } ( \sqrt { T } )$ , where T is the time horizon. However, this bound becomes suboptimal when $L ^ { \star }$ , the cumulative loss of the best expert, is significantly less than T. In this work, we present the first differentially private algorithm with regret $\tilde { O } ( \sqrt { L ^ { \star } } )$ , offering a substantial improvement in the low-loss regime. Our regret bound matches the non-private lower bound up to poly-logarithmic factors for constant privacy parameters, demonstrating that privacy incurs only a small cost in this setting. Our result assumes that the input sequence is oblivious, i.e., it is fixed in advance and independent of the algorithm’s outputs. We also prove that an $\tilde { \Omega } ( \sqrt { T } )$ regret is inevitable for adaptively generated inputs, even when $L ^ { \star } = 0$

## 1 Introduction

Online prediction from experts (OPE) is a fundamental problem in machine learning. In this problem, the learner receives a stream of $T$ loss functions $\ell _ { 1 } , \ldots , \ell _ { T }$ . In each round t, the learner chooses an expert $i _ { t } \in [ N ]$ before seeing $\ell _ { t }$ and suffers loss $\ell _ { t } ( i _ { t } )$ . The learner aims to minimize its total loss, which is compared to that of the best expert in hindsight, denoted by $L ^ { \star }$ . The difference between these two quantities is called the regret, which measures the learner’s performance for this problem.

A critical concern when deploying machine learning algorithms on sensitive data (e.g., financial or medical records) is privacy leakage. To address this, differential privacy (DP) [9, 8] has emerged as the de facto standard for formalizing the privacy guarantee provided by an algorithm. Informally, an algorithm is said to be differentially private if changing a single input data point alters its output distribution only negligibly. As a result, such a change in an individual’s information is unlikely to be detected, thereby protecting personal privacy.

Given the importance of OPE, studying it under DP has received significant attention. Previous work [3, 4] has developed algorithms that achieve regret $\tilde { O } ( \sqrt { T } )$ , which is nearly optimal because it matches the $\Omega ( { \sqrt { T } }$ log N + log N) lower bound [6]—a bound that holds even without any privacy constraints. These results suggest that the cost of privacy in OPE is negligible in the worst case.

In many practical settings—such as financial forecasting, weather prediction, or personalized recommendations—domain experts are specifically selected for their accuracy, so their cumulative loss tends to be low over time. However, the $\tilde { O } ( \sqrt { T } )$ regret becomes less meaningful when the loss of the best expert $L ^ { \star }$ is small because the regret itself can be substantially greater than $L ^ { \star }$ rendering the learner’s loss considerably higher than $L ^ { \star }$ . As an illustration, suppose $L ^ { \star } = T ^ { 1 / 4 }$ . An algorithm with regret $\sqrt { T }$ may incur a loss of $\sqrt { T }$ , which is $T ^ { 1 / 4 }$ times larger than $L ^ { \star }$ . In contrast, when $L ^ { \star } = T ^ { 3 / 4 }$ the total loss is only $( 1 + o ( 1 ) ) L ^ { \star }$ . Thus, worst-case regret bounds are not suitable for OPE with small losses.

In the non-private setting, a natural remedy is to run the multiplicative weights method with a larger learning rate: increasing the learning rate from $\sqrt { \ln { N / T } } \mathrm { t o } \sqrt { \ln { N / L ^ { \star } } }$ yields a regret bound of $O ( \sqrt { L ^ { \star } \log N } + \log N )$ [19, 13]. Unfortunately, such a strategy cannot be directly applied under DP. This is because a large learning rate amplifies the algorithm’s sensitivity to individual data points, thereby violating privacy constraints. Indeed, all existing private OPE algorithms impose a strict upper bound on the learning rate to ensure DP, which prevents them from adapting to small $L ^ { \star }$ Consequently, it remains unclear whether an $\tilde { O } ( \sqrt { L ^ { \star } } )$ regret bound is achievable in the private setting.

## 1.1 Our contribution

In this work, we assume that $\ell _ { 1 } , \ldots , \ell _ { T }$ are chosen by an oblivious adversary. That ${ \mathrm { i s } } ,$ they are fixed in advance and independent of the outputs of the learner.

We propose the first differentially private algorithm that attains regret $\tilde { O } ( \sqrt { L ^ { \star } } )$ for OPE. This nearly matches the optimal regret without privacy constraints, suggesting that the cost of privacy remains small in the low-loss regime. The results are formally stated in the theorems below.

Theorem 1.1. Let $\ell _ { 1 } , \ldots , \ell _ { T }$ be chosen by an oblivious adversary. Given $\varepsilon \leq 1$ , there exists an ε-differentially private algorithm for OPE with expected regret

$$
O \left( \left( \sqrt { \log N \log T } + \log N \right) \cdot \sqrt { \frac { L ^ { \star } } { \varepsilon } } \right) + ( 1 + \ln ( 1 + L ^ { \star } ) ) \cdot \iota _ { 1 } ,
$$

where $\iota _ { 1 } = { \cal O } ( \log ^ { 2 } N / \varepsilon +$ log N log $T / \varepsilon )$

Theorem 1.2. Let $\ell _ { 1 } , \ldots , \ell _ { T }$ be chosen by an oblivious adversary. Given $\varepsilon \leq 1$ and $0 < \delta \le 1 / 2$ there exists an $( \varepsilon , \delta ) { - } d i f f$ erentially private algorithmfor OPE with expected regret

$$
O \left( \sqrt { \frac { L ^ { \star } \log N \log T } { \varepsilon } } + \log N \cdot \operatorname* { m i n } \left( \frac { ( L ^ { \star } \log ( 1 / \delta ) ) ^ { 1 / 3 } } { \varepsilon ^ { 2 / 3 } } , \sqrt { \frac { L ^ { \star } } { \varepsilon } } \right) \right) + ( 1 + \ln ( 1 + L ^ { \star } ) ) \cdot \iota _ { 2 } ,
$$

$$
\iota _ { 2 } = O \left( \log ^ { 3 / 2 } N \sqrt { \log ( \operatorname* { m i n } ( 1 / \delta , N ) ) } / \varepsilon + \log N \log T / \varepsilon \right) \}
$$

Comparison with previous results. When $L ^ { \star } = 0$ , our results recover the bounds obtained by Asi et al. [2] for the realizable setting. When $L ^ { \star }$ grows, our algorithm yields a quadratic improvement over their $\tilde { O } ( L ^ { \star } )$ regret bound. Compared to the $\tilde { O } ( \sqrt { T } )$ regret obtained by Asi et al. $[ 3 , 4 ]$ , our results are significantly better when $L ^ { \star }$ is much smaller than $\bar { T }$

As shown by Asi et al. [3], the above improvement cannot be achieved against an adaptive adversary under pure DP: any OPE algorithm must incur an $\Omega ( T )$ regret even if $L ^ { \star } = 0$ . We additionally prove that it remains impossible under approximate DP by providing an $\tilde { \Omega } ( \sqrt { T } )$ lower bound, which also holds when $L ^ { \star } = 0$

Theorem 1.3. Let T be sufficiently large and $N \geq 2 T$ . Any adaptively $( \varepsilon , \delta )$ -differentially private OPE algorithm with $\varepsilon \leq 1$ and $\delta \leq 1 / T ^ { 3 }$ must incur an $\Omega ( \operatorname* { m i n } ( T , \sqrt { T } / ( \varepsilon \log T ) ) )$ expected regret when $\ell _ { 1 } , \ldots , \ell _ { T }$ are chosen by an adaptive adversary, even ifit is promised that $L ^ { \star } = 0$

## 1.2 Related work

In the non-private setting, the OPE problem has been extensively studied. The multiplicative weights algorithm [19, 23] was shown to achieve a data-dependent regret bound of $O ( \sqrt { L ^ { \star } \log N } + \log N )$ ) with an appropriately tuned learning rate [13]. There are also other methods with the same regret bound, such as follow-the-perturbed-leader [15, 18, 16]. It was formally proved by Cesa-Bianchi et al. [6] that such a bound is asymptotically optimal.

The private version of OPE was first considered by Dwork et al. [11]. It is a folklore result that the multiplicative weights method preserves differential privacy, which follows from the exponential mechanism [20] and advanced composition [12]. A series of subsequent works [21, 17, 1–4] further investigated the problem. In the oblivious adversarial setting considered in this paper, Asi et al. [3] proposed algorithms with regret $\tilde { O } ( \sqrt { T } )$ under pure DP and $O ( \sqrt { T \log N } ) + \tilde { O } ( T ^ { 1 / 3 } )$ under approximate DP, demonstrating that the cost of privacy is negligible. The bound for approximate DP was later refined by Asi et al. [4], who further optimized the dependence on the privacy parameter ε. However, these results are unfavorable when $\hat { L ^ { \star } }$ is much smaller than T. Asi et al. [2] studied the realizable setting, where $L ^ { \star } = 0$ , and gave algorithms with ${ \tilde { O } } ( 1 )$ regret. Their algorithms can be extended to the agnostic setting where $L ^ { \star } > 0 .$ . Nevertheless, the resulting regret bound is $\tilde { O } ( L ^ { \star } )$ which is far from optimal.

## 2 Preliminaries

Online prediction from experts can be formulated as a $T \cdot$ -round game played between a learner and an adversary. In this work, we assume an oblivious adversary who determines a sequence of loss functions $\ell _ { 1 } , \ldots , \ell _ { T }$ in advance and releases them to the learner one by one, where every $\ell _ { t } : [ N ] \to [ 0 , 1 ]$ assigns each of the $N$ experts a loss in [0, 1]. In round $t ,$ the learner chooses an expert $i _ { t } \in [ N ]$ before receiving $\ell _ { t }$ and incurs loss $\ell _ { t } ( i _ { t } )$ . The performance of the learner is assessed by the expected regret, defined as

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \ell _ { t } ( i _ { t } ) - \operatorname* { m i n } _ { i \in [ N ] } \sum _ { t = 1 } ^ { T } \ell _ { t } ( i ) \right] .
$$

For ease of notation, we write $L ^ { \star }$ to denote mi $\begin{array} { r } { \Omega _ { i \in [ N ] } \sum _ { t = 1 } ^ { T } \ell _ { t } ( i ) } \end{array}$ . Note that $L ^ { \star }$ is determined in advance because the adversary is oblivious. However, it is unknown to the learner until the end.

One common approach to solving OPE is the celebrated multiplicative weights method, which in each round t samples $i _ { t }$ such that

$$
\mathrm { P r } [ i _ { t } = i ] = \frac { e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ) } } { \sum _ { i ^ { \prime } \in [ N ] } e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ^ { \prime } ) } } ,
$$

where $\eta$ is the learning rate. The following lemma provides an upper bound on its expected regret. Lemma 2.1 ([13, 5]). For $0 < \eta \leq 1$ , we have

$$
\sum _ { t = 1 } ^ { T } \sum _ { i \in [ N ] } P _ { t } ( i ) \ell _ { t } ( i ) - L ^ { \star } = O \left( \frac { \log N } { \eta } + \eta L ^ { \star } \right) ,
$$

where $\begin{array} { r } { P _ { t }  { \left( i \right) } = \frac { e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } \left( i \right) } } { \sum _ { i ^ { \prime } \in \left[ N \right] } e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } \left( i ^ { \prime } \right) } } . } \end{array}$

We then introduce the notion of differential privacy in the context of OPE.

Definition 2.2 (Differential Privacy [9, 8]). An algorithm A is $( \varepsilon , \delta )$ -differentially private if, for any pair of sequences $S = ( \ell _ { 1 } , \dots , \ell _ { T } )$ and $\bar { S ^ { \prime } } = ( \ell _ { 1 } ^ { \prime } , \cdot \cdot \cdot , \ell _ { T } ^ { \prime } )$ that differ in a single function, and for any subset $O$ of the output range, we have

$$
\operatorname* { P r } [ A ( S ) \in O ] \leq e ^ { \varepsilon } \operatorname* { P r } [ A ( S ^ { \prime } ) \in O ] + \delta .
$$

When $\delta = 0 ,$ , we usually omit δ and say A is ε-differentially private. This case is referred to as pure DP, while the case where $\delta > 0$ is referred to as approximate DP.

A key ingredient in our design is the AboveThreshold mechanism, which receives a sequence of queries and privately identifies the first one that exceeds a prespecified threshold.

Theorem 2.3 (AboveThreshold [10, 7]). We say a query q has sensitivity 1 $i f \left| q ( S ) - q ( S ^ { \prime } ) \right| \leq 1$ for any pair of sequences $S$ and $S ^ { \prime }$ that differ in a single function. Given a threshold τ , there exists an ε-differentially private mechanism AboveThreshold that at each round t receives a sensitivity-1 query q<sub>t</sub> and outputs $\sigma _ { t } \in \{ \top , \bot \}$ , continuing until it either produces the first ⊤ or has processed all queries. Ifthere are at most T queries, then with probability $1 - \beta _ { ; }$ , the AboveThreshold mechanism is α-accuratefor

$$
\alpha = \frac { 8 ( \ln T + \ln ( 2 / \beta ) ) } { \varepsilon } .
$$

Namely, if the mechanism halts after k rounds, we have $q _ { t } \ge \tau - \alpha$ for all $t \in [ k ]$ with $\sigma _ { t } = \top$ and $q _ { t } \leq \tau + \alpha$ for all $t \in [ k ]$ with $\sigma _ { t } = \bot$

Every step of the multiplicative weights method is exactly the exponential mechanism with privacy parameter $\varepsilon = 2 \eta$ . This property, combined with advanced composition, has been widely used in the literature to develop private OPE algorithms.

Lemma 2.4 (Exponential Mechanism [20]). Let $P _ { t }$ be defined as in Lemma 2.1. The process of sampling $i _ { t . }$ from $P _ { t }$ is 2η-differentially private.

## 3 Algorithm and analysis

In this section, we present our algorithm and analyze its privacy and regret guarantees.

Our algorithm is built on the shrinking dartboard algorithm [14], which was also leveraged by Asi et al. [3] to develop algorithms with $\tilde { O } ( \sqrt { T } )$ regret. The key insight is that the marginal distribution of the multiplicative weights method changes only slightly between consecutive rounds. Consequently, it suffices to update the output with only a small probability after each round. By reducing the number of updates, one can allocate a larger privacy budget for each update (i.e., use a larger learning rate η), thereby achieving smaller regret.

However, the shrinking dartboard algorithm is not inherently private because the probability of making an update depends on the underlying data (specifically, $\ell _ { t } ( i _ { t } )$ , i.e., the loss associated with the currently selected expert $i _ { t } )$ . Asi et al. [3, 4] addressed this by forcing the algorithm to perform an additional update at each round with a probability independent of the data. However, to satisfy even approximate DP, this probability must be at least $\tilde { \Omega } ( T ^ { - 1 / 3 } )$ ). By advanced composition, this imposes an ${ \tilde { O } } ( 1 / T ^ { 1 / 3 } )$ upper bound on the learning rate $\eta ,$ which in turn yields an ${ \tilde { O } } ( T ^ { 1 / 3 } )$ regret bound according to Lemma 2.1. This exceeds our desired $\tilde { O } ( \sqrt { L ^ { \star } } )$ regret whenever $L ^ { \star } < T ^ { 2 / 3 }$

We address this by delaying the update until the currently selected expert has incurred a sufficiently large loss, and then making an update with a probability that depends on the cumulative loss accumulated during the delay. Our key consideration here is that, while such a probability is datadependent, performing an update with this probability preserves privacy. Moreover, the delay before an update can be implemented using the AboveThreshold mechanism, which is also differentially private. In addition, we impose a lower bound on the probability of updating and an upper bound on the number of updates, which are crucial for satisfying DP in the worst case. The overall algorithm is described in Algorithm 1. We formally prove its privacy guarantee in Section 3.1 and regret bound in Section 3.2.

## 3.1 Privacy analysis

Let $\mathsf { C o m p } _ { \delta } ( k , \varepsilon _ { 0 } )$ be the smallest ε such that the composition of k $\varepsilon _ { \boldsymbol { 0 } } \cdot$ -differentially private algorithms is $( \varepsilon , \delta )$ -differentially private. We prove the privacy guarantee of Algorithm 1 in the following lemma.

Lemma 3.1. Algorithm 1 is $( \varepsilon , \delta )$ -differentially private for $\varepsilon = \varepsilon _ { 1 } + \operatorname* { m a x } ( \eta , \varepsilon _ { 2 } ) + \mathsf { C o m p } _ { \delta } ( V , 2 \eta )$

Proof. We analyze the augmented transcript consisting of the initial expert $i _ { 1 } ,$ the outputs of all AboveThreshold instances, the $Z _ { u } \mathrm { ' s } ,$ , and the experts generated at the (at most $V )$ resampling steps. The learner’s output sequence is a post-processing of this transcript.

We view the algorithm as the concurrent composition of three types of interactive mechanisms: the collection of AboveThreshold instances, the mechanism generating the $Z _ { u } \mathrm { ' s } ,$ , and the at most V resampling mechanisms. Throughout the proof, when comparing two neighboring input sequences, we consider an arbitrary common prefix of the interaction transcript. Thus, all previously generated AboveThreshold outputs, $Z _ { u } \mathbf { \bar { s } }$ , resampled experts, and the resulting termination times are fixed and identical on the two executions.

We first show that the collection of AboveThreshold instances is ε -differentially private. Let the two neighboring sequences $S$ and $S ^ { \prime }$ differ at round t<sup>˜</sup>. Along any fixed common interaction transcript, there is at most one AboveThreshold instance whose query stream contains the differing loss. All earlier instances receive identical query streams and therefore have identical output distributions on S and $S ^ { \prime }$

Consider the potentially affected instance. Conditioned on the common transcript up to its launch, its starting time and selected expert are fixed and identical on the two executions. Hence, every query issued by this instance has sensitivity at most 1. By the privacy guarantee of AboveThreshold, the transcript generated by this instance is ε -differentially private. If this instance terminates, then conditioned on a common transcript through its termination, every subsequent AboveThreshold instance starts after round $\tilde { t } .$ Consequently, all subsequent instances receive identical query streams on S and $S ^ { \prime }$ . If the affected instance never terminates, there are no subsequent instances. Applying the chain rule to the conditional output distributions therefore shows that the collection of all AboveThreshold instances, viewed as one interactive mechanism, is $\varepsilon _ { 1 }$ -differentially private.

Algorithm 1: Private online prediction from experts   
Input: privacy parameters $\varepsilon _ { 1 }$ and $\varepsilon _ { 2 } .$ learning rate $\eta ,$ threshold τ, limit on updates $\overline { V }$   
1 $u  1 , v  0 ,$ , and $t _ { 0 } \gets 1$   
2 Sample $i _ { 1 }$ uniformly from $[ N ]$   
3 Initiate an instance of AboveThreshold with privacy parameter $\varepsilon _ { 1 }$ and threshold $\tau$   
4 Observe $\ell _ { 1 }$ and suffer loss $\ell _ { 1 } ( i _ { 1 } )$   
5 for $t = 2 , \ldots , T$ do   
6 if $v < V$ then   
7 Define query $\begin{array} { r } { q _ { t } \equiv \sum _ { s = t _ { u - 1 } } ^ { t - 1 } \ell _ { s } ( i _ { t - 1 } ) } \end{array}$   
8 Feed $q _ { t }$ to AboveThreshold and obtain outcome $\sigma _ { t }$   
9 if $\sigma _ { t } = \top$ then   
10 $r _ { u } \gets \operatorname* { m a x } ( 1 / ( e ^ { \varepsilon _ { 2 } } - 1 ) , \sum _ { s = t _ { u - 1 } } ^ { t - 1 } \ell _ { s } ( i _ { t - 1 } ) )$   
11 Sample $Z _ { u }$ from Ber $\left( e ^ { - \eta r _ { u } } \right)$   
12 if $\bar { Z _ { u } _ { u } } = 1$ then   
13 $\textit { \textbf { i } } \{ \textit { \textbf { i } } \{ \textit { \textbf { i } } _ { t - 1 }$   
14 else   
$e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ) }$   
15 Sample $i _ { t } \gets i$ with probability $\frac { - } { \sum _ { i ^ { \prime } \in [ N ] } e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ^ { \prime } ) } }$ for $i \in [ N ]$   
16 $v  v + 1$   
17 end   
18 $t _ { u } \gets t$   
19 $u \gets u + 1$   
20 Initiate an instance of AboveThreshold with privacy parameter $\varepsilon _ { 1 }$ and threshold   
τ   
21 else   
22 $\textit { \textbf { | } }  { 2 } \{ \textit { \textbf { \xi } } \} \longleftarrow { \textit { i } } _ { t - 1 }$   
23 end   
24 else   
25 $i _ { t } \gets i _ { t - 1 }$   
26 end   
27 Observe $\ell _ { t }$ and suffer loss $\ell _ { t } ( i _ { t } )$   
28 end

We next analyze the privacy loss from generating the $Z _ { u } \mathbf { \bar { s } }$ . Let $Z _ { 1 } , Z _ { 2 } , \ldots$ and $Z _ { 1 } ^ { \prime } , Z _ { 2 } ^ { \prime } , \ldots$ . denote the $Z _ { u } \mathbf { \bar { s } }$ generated on neighboring sequences $S$ and $S ^ { \prime }$ , respectively. Similarly, let $r _ { u } \mathrm { { ^ { s } } }$ and $r _ { u } ^ { \prime } \mathrm { { s } }$ denote the $r _ { u }$ -values computed on $S$ and $S ^ { \prime }$ , respectively. As above, all probabilities involving $Z _ { u } { \mathrm { : } } { \mathrm { : } }$ are understood to be conditioned on the relevant fixed common transcript prefix.

Along any common transcript path, the intervals $[ t _ { u - 1 } , t _ { u } - 1 ]$ corresponding to generated $Z _ { u } \mathrm {  ~ s ~ }$ are disjoint. Hence, there is at most one index $j$ such that S and $S ^ { \prime }$ differ at round $\tilde { t } \in [ t _ { j - 1 } , t _ { j } - 1 ]$ . If such $j$ does not exist, then $r _ { u } = r _ { u } ^ { \prime }$ for every $Z _ { u }$ , which implies that the two $Z _ { u }$ -transcripts have identical conditional distributions.

Otherwise, we have $r _ { u } = r _ { u } ^ { \prime }$ for all $u < j$ and $| r _ { j } - r _ { j } ^ { \prime } | \le 1$ . Therefore,

$$
\frac { \mathrm { P r } [ Z _ { j } = 1 ] } { \mathrm { P r } [ Z _ { j } ^ { \prime } = 1 ] } = \frac { e ^ { - \eta r _ { j } } } { e ^ { - \eta r _ { j } ^ { \prime } } } \le e ^ { \eta } .
$$

Moreover, since min $( r _ { j } , r _ { j } ^ { \prime } ) \ge 1 / ( e ^ { \varepsilon _ { 2 } } - 1 )$ , we have

$$
\frac { \operatorname* { P r } [ Z _ { j } = 0 ] } { \operatorname* { P r } [ Z _ { j } ^ { \prime } = 0 ] } = \frac { 1 - e ^ { - \eta r _ { j } } } { 1 - e ^ { - \eta r _ { j } ^ { \prime } } } \leq \frac { \operatorname* { m i n } ( r _ { j } , r _ { j } ^ { \prime } ) + | r _ { j } - r _ { j } ^ { \prime } | } { \operatorname* { m i n } ( r _ { j } , r _ { j } ^ { \prime } ) } \leq e ^ { \varepsilon _ { 2 } } ,
$$

where we use the fact that ${ \frac { 1 - e ^ { - y } } { 1 - e ^ { - x } } } \leq { \frac { y } { x } }$ for $0 < x \leq y .$ . For every later $u > j ,$ , condition on an arbitrary common transcript prefix up to the generation of $Z _ { u }$ . The selected expert and the interval $[ t _ { u - 1 } , t _ { u } - 1 ]$ are then fixed and identical on the two neighboring sequences. Since this interval occurs after the differing round, we have $r _ { u } = r _ { u } ^ { \prime }$ . Thus, $Z _ { u }$ and $Z _ { u } ^ { \prime }$ have the same conditional distribution on the two executions. Therefore, by the chain rule, only $Z _ { j }$ contributes privacy loss, and the generation of the $Z _ { u } { \mathrm { : } } { \mathrm { : } }$ s is max $( \eta , \varepsilon _ { 2 } )$ )-differentially private.

Finally, by Lemma 2.4, each of the at most V resampling steps is 2η-differentially private conditioned on an arbitrary common transcript prefix. By the concurrent composition theorem [22], the adaptive interleaving of these mechanisms has the same composition guarantee as their non-interactive composition. Hence the augmented transcript, and therefore the learner’s output sequence by postprocessing, is $( \varepsilon , \delta )$ -differentially private. □

## 3.2 Regret analysis

We proceed to analyze the regret. The following key lemma provides lower and upper bounds on the cumulative loss from round $t _ { u - 1 }$ through round $t \sb u - 1$

Lemma 3.2. With probability $1 - \beta _ { 1 }$ , we have

$$
\tau - \frac { 8 ( \ln T + \ln ( 2 T / \beta _ { 1 } ) ) } { \varepsilon _ { 1 } } \leq \sum _ { s = t _ { s - 1 } } ^ { t _ { u } - 1 } \ell _ { s } ( i _ { t _ { u } - 1 } ) \leq \operatorname* { m a x } \bigg ( \tau + \frac { 8 ( \ln T + \ln ( 2 T / \beta _ { 1 } ) ) } { \varepsilon _ { 1 } } , 0 \bigg ) + 1
$$

for all $u \geq 1$ . The probability is taken solely over the randomness of the AboveThreshold instances.

Proof. By Theorem 2.3 and the union bound, with probability $1 - \beta _ { 1 }$ we have $q _ { t } \geq \tau - \alpha \mathrm { i f } \sigma _ { t } = \top$ and $q _ { t } \le \tau + \alpha$ if $\sigma _ { t } = \perp$ for all $t \in \left\lceil T \right\rceil$ , where $\alpha = 8 ( \ln T +$ ln $( 2 T / \beta _ { 1 } ) ) / \varepsilon _ { 1 }$ . The lower bound follows directly from the fact that $\sigma _ { t _ { u } } = \top$ . For the upper bound, the inequality trivially holds if $t _ { u - 1 } = t _ { u } - 1$ . When $t _ { u - 1 } < t _ { u } - 1$ , we have $\sigma _ { t _ { u } - 1 } = \perp$ and $i _ { t _ { u } - 2 } = i _ { t _ { u } - 1 }$ . Hence,

$$
\sum _ { s = t _ { u - 1 } } ^ { t _ { u } - 1 } \ell _ { s } ( i _ { t _ { u } - 1 } ) \leq \sum _ { s = t _ { u - 1 } } ^ { t _ { u } - 2 } \ell _ { s } ( i _ { t _ { u } - 2 } ) + 1 \leq \tau + \alpha + 1 .
$$

Next, we show that the number of updates is $\tilde { O } ( \eta L ^ { \star } )$ with high probability. As a consequence, the algorithm works normally (i.e., it won’t reach the update limit) if $V = \tilde { \cal O } ( \eta L ^ { \star } )$ . By the composition theorems, this allows us to set $\eta = \tilde { \Theta } ( 1 / \sqrt { L ^ { \star } } )$ , which minimizes the regret bound in Lemma 2.1.

Lemma 3.3. $\begin{array} { r } { L e t \tau \geq \frac { 8 ( \ln T + \ln ( 2 T / \beta _ { 1 } ) ) } { \varepsilon _ { 1 } } + \frac { 1 } { e ^ { \varepsilon _ { 2 } } - 1 } } \end{array}$ . Conditioned on the event in Lemma 3.2, we have

$$
\operatorname* { P r } [ v _ { T } < 7 1 ( \ln N + \eta L ^ { \star } + 1 ) + 2 8 0 \ln ( 1 / \beta _ { 2 } ) ] \geq 1 - \beta _ { 2 } ,
$$

where $v _ { T }$ is the value ofv when Algorithm 1 terminates.

Proof. Note that v can increase by at most 1 in each round. Therefore, the condition $v < V$ is always met given that $V \geq T$ . In what follows, we will prove the result assuming $V \geq T$ because v<sub>T</sub> can only be smaller if we decrease V.

Define a potential function

$$
\Phi _ { t } = \sum _ { i = 1 } ^ { N } e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ) } .
$$

We have $N = \Phi _ { 1 } \geq \cdots \geq \Phi _ { T + 1 } \geq e ^ { - \eta L ^ { \star } }$ . Let $s _ { j } \in [ T ]$ be the largest index such that $\Phi _ { s _ { i } } \geq \Phi _ { 1 } / e ^ { j }$ and let m be the smallest positive integer such that $s _ { m } = T$ . Then $m \leq \ln ( \Phi _ { 1 } / \Phi _ { T + 1 } ) + 1 \leq$ ln $N + \eta L ^ { \star } + 1$

Suppose an update happens at round $t \in \left[ s _ { j - 1 } + 1 , s _ { j } \right] ( \mathrm { s e t } s _ { 0 } = 0$ for simplicity). Let E denote the event that another update happens at some round $t ^ { \prime } \in [ t + 1 , s _ { j } ]$ . Let

$$
S = \left\{ i \in [ N ] : \sum _ { s = t } ^ { s _ { j } - 1 } \ell _ { s } ( i ) \le 2 / \eta \right\} .
$$

By Lemma 3.2, we have $\begin{array} { r } { \sum _ { s = t _ { u - 1 } } ^ { t _ { u } - 1 } \ell _ { s } ( i _ { t _ { u } - 1 } ) \geq 1 / ( e ^ { \varepsilon _ { 2 } } - 1 ) } \end{array}$ for any $u \geq 1$ . This implies $r _ { u } =$ $\begin{array} { r } { \sum _ { s = t _ { u - 1 } } ^ { t _ { u } - 1 } \ell _ { s } ( i _ { t _ { u } - 1 } ) } \end{array}$ . Thus, for any $i \in S ,$ we have

$$
\begin{array} { r } { \operatorname* { P r } [ E \mid i _ { t } = i ] \le 1 - e ^ { - \eta \sum _ { s = t } ^ { s _ { j } - 1 } \ell _ { s } ( i ) } \le 1 - e ^ { - 2 } . } \end{array}
$$

Note that

$$
\begin{array} { r l } & { \Phi _ { s _ { j } } = \displaystyle \sum _ { i \in S } e ^ { - \eta \sum _ { s = 1 } ^ { s _ { j } - 1 } \ell _ { s } ( i ) } + \sum _ { i \notin S } e ^ { - \eta \sum _ { s = 1 } ^ { s _ { j } - 1 } \ell _ { s } ( i ) } } \\ & { \quad \le \displaystyle \sum _ { i \in S } e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ) } + e ^ { - 2 } \sum _ { i \notin S } e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ) } } \\ & { \quad = \Phi _ { t } { \operatorname* { P r } [ i _ { t } \in S ] } + e ^ { - 2 } \Phi _ { t } { \operatorname* { P r } [ i _ { t } \notin S ] } } \\ & { \quad = \Phi _ { t } ( 1 - \operatorname* { P r } [ i _ { t } \notin S ] ) + e ^ { - 2 } \Phi _ { t } { \operatorname* { P r } [ i _ { t } \notin S ] } . } \end{array}
$$

Rearranging this yields

$$
\operatorname* { P r } [ i _ { t } \notin S ] \leq \frac { 1 - \Phi _ { s _ { j } } / \Phi _ { t } } { 1 - e ^ { - 2 } } \leq \frac { 1 - e ^ { - 1 } } { 1 - e ^ { - 2 } } .
$$

Therefore,

$$
{ \begin{array} { r l } & { \operatorname* { P r } [ E ] \leq \operatorname* { P r } [ i _ { t } \in S ] \cdot \operatorname* { P r } [ E \mid i _ { t } \in S ] + \operatorname* { P r } [ i _ { t } \not \in S ] } \\ & { \qquad = ( 1 - \operatorname* { P r } [ i _ { t } \not \in S ] ) \cdot ( 1 - e ^ { - 2 } ) + \operatorname* { P r } [ i _ { t } \not \in S ] } \\ & { \qquad \leq 1 - e ^ { - 2 } + { \frac { 1 - e ^ { - 1 } } { 1 - e ^ { - 2 } } } \cdot e ^ { - 2 } } \\ & { \qquad < 0 . 9 7 . } \end{array} }
$$

As a consequence, the probability that $v _ { T } \ge n + m$ can be bounded by the probability that the sum of n Bernoulli random variables with mean 0.03 is less than m. By the Chernoff bound, this is at most $\exp ( - 0 . 0 3 n / 8 )$ given that $0 . 0 3 n \geq 2 m$ . Thus, it suffices to set $n = 7 0 m + 2 8 0 \ln ( 1 / \beta _ { 2 } )$ □

The most challenging task in our proof is to compare the output distribution of Algorithm 1 with that of the multiplicative weights method. To this end, we introduce Algorithm 2, which makes the following modifications to Algorithm 1:

• The limit V on the number of updates and the lower bound on $r _ { u }$ are removed.

• The AboveThreshold mechanisms are replaced by a delay function d.

• The learning process can start from an arbitrary round $T _ { s }$

• The update procedure at round $T _ { e }$ is forced.

We first prove that the output distributions of Algorithm 2 and the multiplicative weights method are the same at round $T _ { e }$

Lemma 3.4. For any $T _ { s } \in [ T _ { e } ]$ and $d : \mathbb { Z } ^ { 3 } \to \mathbb { N } ,$ , the output distribution ofAlgorithm 2 at round $T _ { e }$ satisfies

$$
\operatorname* { P r } [ i _ { T _ { e } } = i ] = \frac { e ^ { - \eta \sum _ { s = 1 } ^ { T _ { e } - 1 } \ell _ { s } ( i ) } } { \sum _ { i ^ { \prime } \in [ N ] } e ^ { - \eta \sum _ { s = 1 } ^ { T _ { e } - 1 } \ell _ { s } ( i ^ { \prime } ) } } .
$$

Proof. We prove by backward induction on $T _ { s }$ . When $T _ { s } = T _ { e }$ , the conclusion holds since we sample $i _ { T _ { e } }$ exactly according to the desired distribution. Now consider ${ T _ { s } = t ^ { \prime } \in [ T _ { e } - 1 ] }$ and suppose that the result holds for all $T _ { s } \geq t ^ { \prime } + 1$ . Fix some $i _ { t ^ { \prime } } = k \in [ N ]$ . Define $t _ { 0 } = \bar { t ^ { \prime } }$ and

$$
t _ { j } = \operatorname* { m i n } ( T _ { e } , \operatorname* { m a x } ( t _ { j - 1 } + 1 , d ( t _ { j - 1 } , i _ { t ^ { \prime } } , j ) ) ) .
$$

Algorithm 2: An auxiliary algorithm   
Input: starting point $T _ { s }$ , ending point $T _ { e }$ , learning rate η, delay function d : $\mathbb { Z } ^ { 3 } \to \mathbb { N }$   
1 $u \gets 1$ and $t _ { 0 } \gets T _ { s }$   
2 Sample $i _ { T _ { s } } \gets i$ with probability $\frac { e ^ { - \eta \sum _ { s = 1 } ^ { T _ { s } - 1 } \ell _ { s } ( i ) } } { \sum _ { j \in [ N ] } e ^ { - \eta \sum _ { s = 1 } ^ { T _ { s } - 1 } \ell _ { s } ( j ) } }$ for $i \in [ N ]$   
3 Observe $\ell _ { T _ { s } }$ and suffer loss $\ell _ { T _ { s } } ( i _ { T _ { s } } )$   
4 for $t = T _ { s } + 1 , \dots , T _ { e }$ do   
5 i $\mathbf { f } t \geq d ( t _ { u - 1 } , i _ { t _ { u - 1 } } , u ) o r t = T _ { e }$ then   
6 $\begin{array} { r } { r _ { u } \gets \sum _ { s = t _ { u - 1 } } ^ { t - 1 } \ell _ { s } ( i _ { t - 1 } ) } \end{array}$   
7 Sample $Z _ { u }$ from Ber $\left( e ^ { - \eta r _ { u } } \right)$   
8 if $Z _ { u } ^ { ' } = 1$ then   
9 $\mid \mathrm { ~ \textit ~ { ~ i ~ } ~ }  \dot { \boldsymbol { \imath } } _ { t - 1 }$   
10 else   
$\boldsymbol { \cdot } \boldsymbol { \eta } \sum _ { s = 1 } ^ { t - 1 } \boldsymbol { \ell } _ { s } ( \boldsymbol { i } )$   
11 Sample $i _ { t } \gets i$ with probability $\begin{array} { r } { \overbrace { \sum _ { j \in [ N ] } e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( j ) } } ^ { - } } \end{array}$ e<sup>−</sup> for $i \in [ N ]$   
12 end   
13 $t _ { u } \gets t$   
14 $u \gets u + 1$   
15 else   
16 $i _ { t } \gets i _ { t - 1 }$   
17 end   
18 Observe $\ell _ { t }$ and suffer loss $\ell _ { t } ( i _ { t } )$   
19 end

These are exactly the $t _ { u } \mathrm { \Delta } _ { \mathrm { s } }$ in Algorithm 2 on the event $i _ { t ^ { \prime } } = \cdot \cdot \cdot = i _ { T _ { e } - 1 } = k$ . Let m be the first index such that $t _ { m } = T _ { e }$ . Note that

$$
\mathrm { P r } [ Z _ { 1 } = \cdots \cdot \cdot = Z _ { m } = 1 \mid i _ { t ^ { \prime } } = k ] = e ^ { - \eta \sum _ { s = t ^ { \prime } } ^ { T _ { e } - 1 } \ell _ { s } ( k ) } .
$$

Moreover, by the induction hypothesis, we have

$$
\operatorname* { P r } [ i _ { T _ { e } } = i \mid Z _ { 1 } = \dots = Z _ { j - 1 } = 1 , Z _ { j } = 0 , i _ { t ^ { \prime } } = k ] = { \frac { e ^ { - \eta \sum _ { s = 1 } ^ { T _ { e } - 1 } \ell _ { s } ( i ) } } { \sum _ { i ^ { \prime } \in [ N ] } e ^ { - \eta \sum _ { s = 1 } ^ { T _ { e } - 1 } \ell _ { s } ( i ^ { \prime } ) } } }
$$

for any $j \in [ m ]$ since the learning process starting from $t _ { j }$ can be seen as running Algorithm 2 with d shifted by j along the third dimension and $T _ { s } = t _ { j }$ . For simplicity, write $w _ { t } ( i ) = e ^ { - \eta \sum _ { s = 1 } ^ { t - 1 } \ell _ { s } ( i ) }$ and $\begin{array} { r } { \Phi _ { t } = \sum _ { i \in [ N ] } w _ { t } \bar { ( i ) } } \end{array}$ for $t \in [ T _ { e } ]$ . Note that $i _ { T _ { e } } \neq i _ { t ^ { \prime } }$ only $\mathrm { i f } \ \bar { Z } _ { j } = 0$ for some $j .$ . Therefore, we have

$$
\operatorname* { P r } [ i _ { T _ { e } } = i \mid i _ { t ^ { \prime } } = k ] = \left( 1 - e ^ { - \eta \sum _ { s = t ^ { \prime } } ^ { T _ { e } - 1 } \ell _ { s } ( k ) } \right) \cdot \frac { w _ { T _ { e } } ( i ) } { \Phi _ { T _ { e } } }
$$

for $i \neq k$ and

$$
\operatorname* { P r } [ i _ { T _ { e } } = k \mid i _ { t ^ { \prime } } = k ] = e ^ { - \eta \sum _ { s = t ^ { \prime } } ^ { T _ { e } - 1 } \ell _ { s } ( k ) } + \Bigl ( 1 - e ^ { - \eta \sum _ { s = t ^ { \prime } } ^ { T _ { e } - 1 } \ell _ { s } ( k ) } \Bigr ) \cdot \frac { w _ { T _ { e } } ( k ) } { \Phi _ { T _ { e } } } .
$$

As a consequence, we have

$$
\begin{array} { r l } & { \operatorname* { P r } [ i _ { T _ { c } } = i ] = \displaystyle \sum _ { k \in [ N ] } \operatorname* { P r } [ i _ { T _ { c } } = i \mid i _ { t ^ { \prime } } = k ] \cdot \operatorname* { P r } [ i _ { t ^ { \prime } } = k ] } \\ & { \quad \quad \quad = \displaystyle \sum _ { k \in [ N ] } \Big ( 1 - e ^ { - \eta \sum _ { s = t ^ { \prime } } ^ { T _ { c } - 1 } \ell _ { s } ( k ) } \Big ) \cdot \frac { w _ { T _ { c } } ( i ) } { \Phi _ { T _ { c } } } \cdot \frac { w _ { t ^ { \prime } } ( k ) } { \Phi _ { t ^ { \prime } } } + e ^ { - \eta \sum _ { s = t ^ { \prime } } ^ { T _ { c } - 1 } \ell _ { s } ( i ) } \cdot \frac { w _ { t ^ { \prime } } ( i ) } { \Phi _ { t ^ { \prime } } } } \\ & { \quad \quad \quad = \frac { w _ { T _ { c } } ( i ) } { \Phi _ { T _ { c } } } \cdot \frac { \Phi _ { t ^ { \prime } } - \Phi _ { T _ { c } } } { \Phi _ { t ^ { \prime } } } + \frac { w _ { T _ { c } } ( i ) } { \Phi _ { t ^ { \prime } } } } \\ & { \quad \quad \quad = \frac { w _ { T _ { c } } ( i ) } { \Phi _ { T _ { c } } } } \end{array}
$$

as desired.

We leverage the fact that Algorithm 2 behaves similarly to Algorithm 1 before round $T _ { e }$ when $T _ { s } = 1 \colon$ Lemma 3.2 and Lemma 3.3 suggest that the first modification has little impact, while the second modification can be seen as fixing the random coins used by the AboveThreshold mechanisms (see Appendix B for a more detailed explanation). By analyzing their behavior at round $T _ { e }$ and applying the above lemma, we can argue that with high probability, the output distribution of Algorithm 1 at each round is pointwise at most $e ^ { O ( \eta ( \tau + \mathrm { l o g } \hat { T / \varepsilon } _ { 1 } ) ) } \approx \dot { 1 } + O ( \eta \tau + \eta \log T / \varepsilon _ { 1 } )$ times that of the multiplicative weights mechanism. This leads to the following regret bound.

Lemma 3.5. Suppose $\begin{array} { r } { \tau \geq \frac { 8 ( \ln T + 2 \ln ( 2 T ) ) } { \varepsilon _ { 1 } } + \frac { 1 } { e ^ { \varepsilon _ { 2 } } - 1 } \geq 1 , 0 < \eta \tau \leq 1 } \end{array}$ , and $V \geq 7 1 ( \ln N + \eta L ^ { \star } +$ $1 ) + 2 8 0 \ln T$ . The expected regret ofAlgorithm 1 is at most

$$
{ \cal O } \left( \frac { \log N } { \eta } + \tau \eta L ^ { \star } \right) .
$$

Proof. Let $\beta _ { 1 } = 1 / T$ . Since the regret cannot be larger than T, it suffices to prove the desired conclusion conditioned on the event in Lemma 3.2. As argued in the proof of Lemma 3.3, for any $u \geq 1$ we have

$$
r _ { u } = \sum _ { s = t _ { u - 1 } } ^ { t _ { u } - 1 } \ell _ { s } ( i _ { t _ { u } - 1 } ) .
$$

By Lemma 3.2, there exists $\Delta = \tau + O ( \log T / \varepsilon _ { 1 } ) + 1 = O ( \tau )$ such that $r _ { u } \le \Delta$ for all $u \geq 1$

We first prove the result under the condition that $V \geq T$ . Fix a realization of the random coins used by the AboveThreshold mechanisms and let $\hat { Q } _ { t } ( i )$ be the probability that Algorithm 1 outputs $i _ { t } = i$ Since $V \geq T$ , there is no restriction on the number of updates. Hence, this realization induces a delay function d so that Algorithm 1 and Algorithm 2 (with this delay function d and $T _ { s } = 1 )$ behave identically in the first $t - 1$ rounds.

Let $P _ { t }$ be as defined in Lemma 2.1 and consider any $i \in [ N ]$ . By Lemma 3.4 with $1 = T _ { s } < T _ { e } = t ,$ Algorithm 2 outputs $i _ { t } = i$ with probability $P _ { t } ( \dot { i } )$ . Since it produces $i _ { t - 1 } = i$ with probability $\hat { Q } _ { t - 1 } ( i )$ , we have $P _ { t } ( i ) \geq \hat { Q } _ { t - 1 } ( i ) \cdot e ^ { - \eta \Delta }$ . Now consider Algorithm 1. If it outputs $i _ { t - 1 } \neq i$ , then the probability that it outputs $i _ { t } = i$ is at most $( 1 - e ^ { - \eta \Delta } ) P _ { t } ( \bar { i } )$ . As a consequence, we have

$$
\begin{array} { r l } & { \hat { Q } _ { t } ( i ) \leq \hat { Q } _ { t - 1 } ( i ) + ( 1 - \hat { Q } _ { t - 1 } ( i ) ) \cdot ( 1 - e ^ { - \eta \Delta } ) P _ { t } ( i ) } \\ & { \qquad \leq e ^ { \eta \Delta } P _ { t } ( i ) + ( 1 - e ^ { - \eta \Delta } ) P _ { t } ( i ) } \\ & { \qquad \leq e ^ { 2 \eta \Delta } P _ { t } ( i ) , } \end{array}
$$

where the last inequality is due to the fact that $1 + x - 1 / x \leq x ^ { 2 }$ for $x \geq 1$ . Moreover, we have $\hat { Q } _ { 1 } ( i ) = P _ { 1 } ( i )$ since we sample $i _ { 1 }$ uniformly at random. Thus, $\hat { Q } _ { t } ( i ) \leq e ^ { 2 \eta \Delta } P _ { t } ( i )$ for all $t \in [ T ]$ By Lemma 2.1, the expected regret of Algorithm 1 with $V \geq T$ is

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { i \in [ N ] } \hat { Q } _ { t } ( i ) \ell _ { t } ( i ) - L ^ { \star } \leq e ^ { 2 \eta \Delta } \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { i \in [ N ] } P _ { t } ( i ) \ell _ { t } ( i ) - L ^ { \star } } & { } \\ { = e ^ { 2 \eta \Delta } \cdot O \left( \displaystyle \frac { \log N } { \eta } + \eta L ^ { \star } \right) + ( e ^ { 2 \eta \Delta } - 1 ) L ^ { \star } } & { } \\ { } & { = O \left( \displaystyle \frac { \log N } { \eta } + \eta \tau L ^ { \star } \right) } \end{array}
$$

because $\eta \Delta = O ( \eta \tau ) = O ( 1 )$

Let $Q _ { t } ( i )$ be the probability that Algorithm 1 outputs $i _ { t } = i$ with V as in the lemma statement. Applying Lemma 3.3 with $\dot { \beta } _ { 2 } = 1 / \check { T }$ , we obtain $\dot { \mathrm { P r } } [ v _ { T } < V ] \geq 1 - 1 / T$ . We can decompose $Q _ { t } ( i )$ as $\mathrm { P r } [ v _ { T } < V ] Q _ { t } ^ { 0 } ( i ) \stackrel { - } { + } \mathrm { P r } [ \stackrel { . } { v } _ { T } \geq V ] Q _ { t } ^ { 1 } ( i )$ , where $Q _ { t } ^ { 0 } ( i )$ and $Q _ { t } ^ { 1 } ( i )$ are the probabilities of outputting $i _ { t } \ = \ i$ conditioned on $v _ { T } ~ < ~ V$ and $v _ { T } ~ \ge ~ \dot { V }$ , respectively. We can also write $\hat { Q } _ { t } ( i ) = \operatorname* { P r } [ v _ { T } < V ] \hat { Q } _ { t } ^ { 0 } ( i ) + \operatorname* { P r } [ v _ { T } \geq V ] \hat { Q } _ { t } ^ { 1 } ( i )$ in the same way. For any $t \in [ T ]$ , we have

$$
\begin{array} { r l } {  { \sum _ { i \in [ N ] } | Q _ { t } ( i ) - \hat { Q } _ { t } ( i ) | \le \sum _ { i \in [ N ] } \operatorname* { P r } [ v _ { T } < V ] | Q _ { t } ^ { 0 } ( i ) - \hat { Q } _ { t } ^ { 0 } ( i ) | + \operatorname* { P r } [ v _ { T } \ge V ] | Q _ { t } ^ { 1 } ( i ) - \hat { Q } _ { t } ^ { 1 } ( i ) | } } \\ & { \le 2 \operatorname* { P r } [ v _ { T } \ge V ] } \\ & { \le 2 / T } \end{array}
$$

since $Q _ { t } ^ { 0 } ( i ) = \hat { Q } _ { t } ^ { 0 } ( i )$ . Consequently,

$$
\sum _ { t = 1 } ^ { T } \sum _ { i \in [ N ] } Q _ { t } ( i ) \ell _ { t } ( i ) - \sum _ { t = 1 } ^ { T } \sum _ { i \in [ N ] } \hat { Q } _ { t } ( i ) \ell _ { t } ( i ) \leq \sum _ { t = 1 } ^ { T } \sum _ { i \in [ N ] } | Q _ { t } ( i ) - \hat { Q } _ { t } ( i ) | \leq 2 .
$$

Applying the triangle inequality leads to the desired result.

Choosing $\eta = \tilde { \Theta } ( 1 / \sqrt { L ^ { \star } } )$ minimizes the regret bound in Lemma 3.5. However, we cannot do this directly since $L ^ { \star }$ is unknown at the beginning. This issue can be circumvented by applying the standard doubling trick, which starts by assuming that $L ^ { \star }$ is small and doubles the estimated value once the estimate becomes inaccurate. We defer the detailed implementation to Appendix A.

## 4 Lower bound for adaptive adversaries

In this section, we briefly discuss how we establish the lower bound for adaptive inputs. We utilize a lower bound in [3], which states that every adaptively $( \varepsilon , \delta )$ -differentially private OPE algorithm with $N \geq 2 T$ experts must incur $\Omega ( T )$ regret in the high-privacy regime where $\varepsilon = \tilde { O } ( 1 / \sqrt { T } )$ even if $L ^ { \star } = 0$ . Now suppose A is an adaptively $( \varepsilon , \delta )$ )-differentially private OPE algorithm with $\varepsilon \leq 1$ and regret at most $R _ { A } ( T )$ . We create an algorithm $A _ { q }$ that simulates ${ \mathcal { A } } ,$ forwards A’s outputs as its own, and feeds the loss function $\mathsf { B e r } ( q ) \bar { \ell } _ { t } \mathrm { t o } \mathcal { A } .$ . This can be seen as an extension of the canonical “privacy amplification by subsampling” technique to the online setting. We prove that $A _ { q }$ has privacy parameter $O ( q \varepsilon )$ and regret at most $R _ { \bf \bar { A } } ( T ) / q$ . Applying the lower bound in [3] to $A _ { q }$ with $q = \tilde { \Theta } ( \operatorname* { m i n } ( 1 , 1 / ( \varepsilon \sqrt { T } ) ) )$ yields $R _ { A } ( T ) / q = \Omega ( T )$ . Rearranging this gives $R _ { A } ( T ) = \Omega ( q T ) = \tilde { \Omega } ( \operatorname* { m i n } ( T , \sqrt { T } / \varepsilon ) )$

The main technical challenge lies in proving the privacy amplification result in the adaptive setting, which requires a more sophisticated analysis than in the oblivious setting. We defer the formal proof to Appendix C.

## 5 Discussion and future work

In this work, we propose the first differentially private algorithm for OPE that achieves a datadependent regret bound of $\tilde { O } ( \sqrt { L ^ { \star } } )$ , extending the previous $\tilde { O } ( \sqrt { T } )$ bound to the low-loss regime. Our result matches the non-private lower bound $\hat { \Omega } ( \sqrt { L ^ { \star } }$ log $\overline { { N } } +$ log N) up to poly-logarithmic factors for constant privacy parameters, suggesting that the cost of privacy is mild for OPE with small losses. Our algorithm works against any oblivious adversary. We also show that $\tilde { \Omega } ( \sqrt { T } )$ regret is necessary for adaptive inputs, demonstrating that such an improvement cannot be made against an adaptive adversary.

When $L ^ { \star }$ is large, our regret bounds are slightly worse than those in [3, 4] in certain parameter ranges. For example, suppose $L ^ { \star } = \Theta ( T )$ . Under pure DP, Asi et al. [3] gave an upper bound of $O ( \sqrt { T } \log N / \varepsilon )$ . Our bound in Theorem 1.1 is larger than theirs by a factor of $\sqrt { \varepsilon \log T / \log N }$ when this factor is greater than 1 (when it is less than 1, our bound is better). Under approximate DP, Asi et al. [4] proved an upper bound of $O ( \sqrt { T \log N } + T ^ { 1 / 3 } \log N \log ( T / \delta ) / \varepsilon ^ { 2 / 3 } )$ . The first term in our bound in Theorem 1.2 is larger than that in their bound by a factor of $O ( { \sqrt { \log T / \varepsilon } } )$ . An interesting future direction is to investigate whether our bounds can be further improved to match these results.

## Acknowledgments and Disclosure of Funding

The research was supported in part by an NSFC grant 62432008, an RGC TRS grant T43-513/23N-2, RGC CRF grants C1029-22G and C6015-23G, an NSFC/RGC grant CRS\_HKUST601/24 and RGC GRF grants 16207423, 16203824 and 16216725.

## References

[1] Naman Agarwal and Karan Singh. The price of differential privacy for online learning. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pages 32–40, 2017.

[2] Hilal Asi, Vitaly Feldman, Tomer Koren, and Kunal Talwar. Near-optimal algorithms for private online optimization in the realizable regime. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 1107–1120, 2023.

[3] Hilal Asi, Vitaly Feldman, Tomer Koren, and Kunal Talwar. Private online prediction from experts: Separations and faster rates. In Proceedings of the 36th Conference on Learning Theory, volume 195, pages 674–699, 2023.

[4] Hilal Asi, Tomer Koren, Daogao Liu, and Kunal Talwar. Private online learning via lazy algorithms. In Advances in Neural Information Processing Systems, volume 37, pages 112158– 112183, 2024.

[5] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge University Press, 2006.

[6] Nicolo Cesa-Bianchi, Yoav Freund, David Haussler, David P Helmbold, Robert E Schapire, and Manfred K Warmuth. How to use expert advice. Journal ofthe ACM, 44(3):427–485, 1997.

[7] Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy. Foundations and Trends® in Theoretical Computer Science, 9(3–4):211–407, 2014.

[8] Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Proceedings of the 25th Annual International Conference on the Theory and Applications ofCryptographic Techniques, pages 486–503, 2006.

[9] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Proceedings of the 3rd Conference on Theory of Cryptography, pages 265–284, 2006.

[10] Cynthia Dwork, Moni Naor, Omer Reingold, Guy N Rothblum, and Salil Vadhan. On the complexity of differentially private data release: Efficient algorithms and hardness results. In Proceedings of the 41st Annual ACM Symposium on Theory of Computing, pages 381–390, 2009.

[11] Cynthia Dwork, Moni Naor, Toniann Pitassi, and Guy N Rothblum. Differential privacy under continual observation. In Proceedings of the 42nd Annual ACM Symposium on Theory of Computing, pages 715–724, 2010.

[12] Cynthia Dwork, Guy N Rothblum, and Salil Vadhan. Boosting and differential privacy. In Proceedings of the 51st Annual IEEE Symposium on Foundations of Computer Science, pages 51–60, 2010.

[13] Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal ofComputer and System Sciences, 55(1):119–139, 1997.

[14] Sascha Geulen, Berthold Vöcking, and Melanie Winkler. Regret minimization for online buffering problems using the weighted majority algorithm. In Proceedings ofthe 23rd Conference on Learning Theory, pages 132–143, 2010.

[15] James Hannan. Approximation to Bayes risk in repeated play. In Contributions to the Theory of Games, Volume III, volume 39 of Annals of Mathematics Studies, pages 97–139. Princeton University Press, 1957.

[16] Marcus Hutter and Jan Poland. Adaptive online prediction by following the perturbed leader. Journal ofMachine Learning Research, 6:639–660, 2005.

[17] Prateek Jain and Abhradeep Guha Thakurta. (Near) dimension independent risk bounds for differentially private learning. In Proceedings of the 31st International Conference on Machine Learning, volume 32, pages 476–484, 2014.

[18] Adam Kalai and Santosh Vempala. Efficient algorithms for online decision problems. Journal of Computer and System Sciences, 71(3):291–307, 2005.

[19] Nick Littlestone and Manfred K Warmuth. The weighted majority algorithm. Information and Computation, 108(2):212–261, 1994.

[20] Frank McSherry and Kunal Talwar. Mechanism design via differential privacy. In Proceedings of the 48th Annual IEEE Symposium on Foundations of Computer Science, pages 94–103, 2007.

[21] Adam Smith and Abhradeep Thakurta. (Nearly) optimal algorithms for private online learning in full-information and bandit settings. In Advances in Neural Information Processing Systems, volume 26, pages 2733–2741, 2013.

[22] Salil Vadhan and Tianhao Wang. Concurrent composition of differential privacy. In Theory of Cryptography: 19th International Conference, pages 582–604. Springer, 2021.

[23] Volodimir G Vovk. Aggregating strategies. In Proceedings of the Third Annual Workshop on Computational Learning Theory, pages 371–383, 1990.

## A Algorithm without prior knowledge of $L ^ { \star }$

In this section, we show how to construct an algorithm without prior knowledge of $L ^ { \star }$ . We first prove regret bounds when some estimate $L ^ { \prime } \geq L ^ { \star }$ is given.

Theorem A.1. $L e t \varepsilon \leq 1$ and $L ^ { \prime } > 0$ . There exists an ε-differentially private algorithmfor OPE with expected regret

$$
O \left( \left( \log N + { \sqrt { \log N \log T } } \right) \cdot { \sqrt { \frac { L ^ { \prime } } { \varepsilon } } } \right) + \iota _ { 1 }
$$

whenever $L ^ { \prime } \geq L ^ { \star }$ , where $\iota _ { 1 } = O ( \log ^ { 2 } N / \varepsilon + \log N \log T / \varepsilon )$

Proof. We run Algorithm 1 with $\varepsilon _ { 1 } = \varepsilon _ { 2 } = \varepsilon / 3 , \tau = 8 ( \ln T + 2 \ln ( 2 T ) ) / \varepsilon _ { 1 } + 1 / ( e ^ { \varepsilon _ { 2 } } - 1 )$ $\eta ~ = ~ \operatorname* { m i n } ( { \varepsilon } / ( 1 2 7 8 \ln N ~ + ~ 1 2 9 6 ) , { \varepsilon } / ( 5 0 4 0 \ln T ) , \sqrt { { \varepsilon } / ( 1 2 7 8 L ^ { \prime } ) } , \sqrt { { \varepsilon } \ln N / ( L ^ { \prime } \ln T ) } )$ , and $V =$ $\left\lceil 7 1 ( \ln N \dot { + } \dot { \eta } L ^ { \prime } + 1 ) + 2 8 0 \ln T \right\rceil$ . One can verify that $\eta \leq \varepsilon / 3 , 2 \eta V \leq \varepsilon / 3 ,$ and $\eta \tau \ \leq \ 1$ By Lemma 3.1 and basic composition, the overall algorithm is ε-differentially private. By Lemma 3.5 and the fact that $L ^ { \star } \leq L ^ { \prime }$ , the expected regret is bounded by

$$
O \left( { \frac { \log N } { \eta } } + { \frac { \log T } { \varepsilon } } \cdot \eta L ^ { \prime } \right) .
$$

Plugging in the value of η yields the desired result.

Theorem A.2. Let $\varepsilon \le 1 , 0 < \delta \le 1 / 2 ,$ , and $L ^ { \prime } > 0 .$ . There exists an (ε, δ)-differentially private algorithmfor OPE with expected regret

$$
O \left( { \sqrt { \frac { L ^ { \prime } \log N \log T } { \varepsilon } } } + \log N \cdot \operatorname* { m i n } \left( { \frac { ( L ^ { \prime } \log ( 1 / \delta ) ) ^ { 1 / 3 } } { \varepsilon ^ { 2 / 3 } } } , { \sqrt { \frac { L ^ { \prime } } { \varepsilon } } } \right) \right) + \iota _ { 2 }
$$

whenever $L ^ { \prime } \geq L ^ { \star }$ , where $\iota _ { 2 } = O \left( \log ^ { 3 / 2 } N \sqrt { \log ( \operatorname* { m i n } ( 1 / \delta , N ) ) } / \varepsilon + \log N \log T / \varepsilon \right) \rvert$

Proof. We run Algorithm 1 with $\varepsilon _ { 1 } = \varepsilon _ { 2 } = \varepsilon / 3 , \tau = 8 ( \ln T + 2 \ln ( 2 T ) ) / \varepsilon _ { 1 } + 1 / ( e ^ { \varepsilon _ { 2 } } - 1 )$ $\eta = \operatorname* { m i n } ( \eta _ { 1 } , \eta _ { 2 } , \eta _ { 3 } , \eta _ { 4 } , 1 / \tau )$ , and $V = \left\lceil 7 1 ( \ln N + \eta L ^ { \prime } + 1 ) + 2 8 0 \ln T \right\rceil ,$ , where

$$
\begin{array} { r l } & { \bullet \ \eta _ { 1 } = \operatorname* { m a x } ( \varepsilon / ( 1 8 \cdot \lceil 7 1 \ln N + 7 2 \rceil ) , \varepsilon / ( 3 6 \sqrt { 2 \cdot \lceil 7 1 \ln N + 7 2 \rceil \ln ( 3 / \delta ) } ) ) . } \\ & { \bullet \ \eta _ { 2 } = \operatorname* { m a x } ( \varepsilon / ( 1 8 \cdot \lceil 2 8 0 \ln T \rceil ) , \varepsilon / ( 3 6 \sqrt { 2 \cdot \lceil 2 8 0 \ln T \rceil \ln ( 3 / \delta ) } ) ) . } \end{array}
$$

$$
\begin{array} { r l } & { \bullet \ \eta _ { 3 } = \operatorname* { m a x } ( \sqrt { \varepsilon / ( 1 2 7 8 L ^ { \prime } ) } , \varepsilon ^ { 2 / 3 } / ( 1 8 4 0 3 2 L ^ { \prime } \ln ( 3 / \delta ) ) ^ { 1 / 3 } ) . } \\ & { \bullet \ \eta _ { 4 } = \sqrt { \varepsilon \ln { N / ( L ^ { \prime } \ln { T } ) } } . } \end{array}
$$

Note that $\left\lceil 7 1 ( \ln N + \eta L ^ { \prime } + 1 ) + 2 8 0 \ln T \right\rceil \leq \left\lceil 7 1 \ln N + 7 2 \right\rceil + \left\lfloor 7 1 \eta L ^ { \prime } \right\rfloor + \left\lceil 2 8 0 \ln T \right\rceil$ . By basic and advanced composition, we have

$$
\begin{array} { r l } & { \quad \mathsf { C o m p } _ { \delta } ( V , 2 \eta ) } \\ & { \le \mathsf { C o m p } _ { \delta / 3 } \bigl ( \lceil 7 1 \ln N + 7 2 \rceil , 2 \eta \bigr ) + \mathsf { C o m p } _ { \delta / 3 } \bigl ( \lfloor 7 1 \eta L ^ { \prime } \rfloor , 2 \eta \bigr ) + \mathsf { C o m p } _ { \delta / 3 } \bigl ( \lceil 2 8 0 \ln T \rceil , 2 \eta \bigr ) } \\ & { \le \varepsilon / 9 + \varepsilon / 9 + \varepsilon / 9 } \\ & { = \varepsilon / 3 . } \end{array}
$$

Applying Lemma 3.1 proves the privacy guarantee. Since $\eta \tau \leq 1$ , by Lemma 3.5 and the fact that $L ^ { \star } \leq L ^ { \prime } .$ , the expected regret is bounded by

$$
O \left( { \frac { \log N } { \eta } } + { \frac { \log T } { \varepsilon } } \cdot \eta L ^ { \prime } \right) .
$$

Plugging in the value of η yields the desired result.

□

```latex
Algorithm 3: Algorithm without prior knowledge of $\overline { { L ^ { \star } } }$
1 $k  1 , b _ { 1 }  1$ , and α ← 16(ln T +2 ln(2T))
ε
2 Initiate an instance of AboveThreshold with privacy parameter $\varepsilon / 2$ and threshold 4α
3 Initiate an instance of Alg with $L ^ { \prime } = 6 \alpha$
4 for $t = 1 , \dots , T$ do
5 Obtain $i _ { t }$ from Alg
6 Observe $\ell _ { t }$ and suffer loss $\ell _ { t } ( i _ { t } )$
7 Define query $\begin{array} { r } { q _ { t } \equiv \operatorname* { m i n } _ { i \in [ N ] } \sum _ { s = b _ { k } } ^ { t } \ell _ { s } ( i ) } \end{array}$
8 Feed $q _ { t }$ to AboveThreshold and obtain outcome $\sigma _ { t }$
9 if $\cdot \sigma _ { t } = \top$ then
10 $k \gets k + 1$
11 $b _ { k } \gets t + 1$
12 Terminate the existing instance of $\mathsf { A l g }$ and initiate a new one with $L ^ { \prime } = 3 \cdot 2 ^ { k } \alpha$
13 Initiate an instance of AboveThreshold with privacy parameter $\varepsilon / 2$ and threshold
$( 3 \cdot 2 ^ { k } - 2 ) \alpha$
14 end
15 end
```

The overall algorithm works as follows. We start with $L ^ { \prime } = 6 \alpha$ and employ an AboveThreshold mechanism to test if the current cumulative loss of the best expert exceeds $L ^ { \prime } .$ . Once this happens, we double the value of $L ^ { \prime }$ and initiate new instances of both the learning algorithm and the AboveThreshold mechanism. This procedure is formalized in Algorithm 3 and the regret guarantee is established in the lemma below. Applying this framework to the two preceding theorems yields our main results in Section 1.1.

Lemma A.3. IfAlg is an $( \varepsilon / 2 , \delta )$ -differentially private OPE algorithm for every $L ^ { \prime } > 0$ and has expected regret at most $R ( L ^ { \prime } )$ whenever $L ^ { \prime } \geq L ^ { \star }$ , then Algorithm 3 is (ε, δ)-differentially private and has expected regret at most

$$
1 + \sum _ { k = 1 } ^ { K } R ( 3 \cdot 2 ^ { k } \alpha ) ,
$$

$$
\begin{array} { r } { \alpha = \frac { 1 6 ( \ln T + 2 \ln ( 2 T ) ) } { \varepsilon } } \end{array}
$$

Proof. We first prove the privacy guarantee. Consider the augmented transcript consisting of the termination times of all AboveThreshold instances together with the outputs of all instances of Alg.

We first claim that the sequence of AboveThreshold termination times is $\varepsilon / 2$ -differentially private. Consider two neighboring loss sequences that differ at round t<sup>˜</sup>, and condition on an arbitrary common prefix of the termination-time transcript. There is at most one AboveThreshold instance whose queries contain the differing loss. All earlier instances receive identical query streams. For the affected instance, every query has sensitivity at most 1, so its termination time is $\varepsilon / 2 .$ -differentially private. Conditioned on a common termination time for this instance, every subsequent instance starts strictly after round $\tilde { t } ,$ and therefore receives identical query streams on the two neighboring sequences. By the chain rule, the entire sequence of termination times is $\varepsilon / 2$ -differentially private.

Now fix any sequence of termination times. This also fixes a partition of the loss sequence into disjoint subsequences. The instances of Alg are run independently on these disjoint subsequences, and at most one of them receives the differing loss. By parallel composition, their joint output is $( \varepsilon / 2 , \delta )$ -differentially private. Applying basic composition to the termination-time transcript and the outputs of $\mathsf { A l g } .$ , followed by post-processing, shows that Algorithm 3 is $( \varepsilon , \delta )$ -differentially private.

We then prove utility. By Theorem 2.3 and the union bound, with probability $1 - 1 / T$ all applications of AboveThreshold are α-accurate. We first prove a regret bound conditioned on this event. Suppose $\sigma _ { t } = \top$ at round t. Before increasing the value of $k ,$ we have

$$
( 3 \cdot 2 ^ { k } - 3 ) \alpha \leq q _ { t } \leq q _ { t - 1 } + 1 \leq ( 3 \cdot 2 ^ { k } - 1 ) \alpha + 1 < 3 \cdot 2 ^ { k } \alpha .
$$

By our assumption, the algorithm $\mathsf { A l g }$ with $L ^ { \prime } = 3 { \cdot } 2 ^ { k } \alpha$ incurs an expected regret of at most $R ( 3 \cdot 2 ^ { k } \alpha )$ during $[ b _ { k } , t ] \dot { = } [ b _ { k } , b _ { k + 1 } - 1 ]$ . Moreover, we have

$$
L ^ { \star } = \operatorname* { m i n } _ { i \in [ N ] } \sum _ { t = 1 } ^ { T } \ell _ { t } ( i ) \geq q _ { t } \geq ( 3 \cdot 2 ^ { k } - 3 ) \alpha \geq 3 \cdot 2 ^ { k - 1 } \alpha .
$$

This implies that

$$
3 \cdot 2 ^ { k + 1 } \alpha \leq 4 L ^ { \star } .
$$

Hence, whenever a new bucket $k + 1$ is initiated, we have $k + 1 \leq K$ . Since the algorithm starts with $k = 1$ , the total number of buckets is therefore at most $K$ . For the final bucket, if no $\top$ is returned, the accuracy of AboveThreshold implies $q _ { T } \leq ( 3 \cdot 2 ^ { k } - 1 ) \alpha < 3 \cdot 2 ^ { k } \alpha .$ , so the same regret bound applies. Summing the regret over all buckets gives $\textstyle \sum _ { k = 1 } ^ { K } R ( 3 \cdot 2 ^ { k } \alpha )$

Since the regret can be at most $T ,$ , the unconditional expected regret is at most

$$
( 1 - 1 / T ) \sum _ { k = 1 } ^ { K } R ( 3 \cdot 2 ^ { k } \alpha ) + 1 / T \cdot T \leq 1 + \sum _ { k = 1 } ^ { K } R ( 3 \cdot 2 ^ { k } \alpha ) .
$$

## B Construction of the delay function

In this section, we show how to construct the delay function when the randomness of the AboveThreshold mechanisms is fixed. Consider an AboveThreshold mechanism that is initiated at round s. It is implemented as in Algorithm 4.

Note that in each round $t \in \left\lceil T \right\rceil$ we always draw a random variable $\nu _ { t }$ . Hence, we can fix the values of $\nu _ { 1 } , \dots , \nu _ { T }$ at the beginning. Additionally, we draw $\xi _ { 1 } , \dots , \xi _ { T }$ , where $\xi _ { u }$ denotes the variable $\xi$ used by the u-th AboveThreshold mechanism.

Now suppose that in Algorithm 1 we initiate the u-th AboveThreshold mechanism at round $t _ { u - 1 }$ Then it will return $\sigma _ { t } = \top$ for the first t satisfying

$$
\sum _ { s = t _ { u - 1 } } ^ { t - 1 } \ell _ { s } ( i _ { t _ { u - 1 } } ) + \nu _ { t } \geq \tau + \xi _ { u } .
$$

As a consequence, we can define $d ( t _ { u - 1 } , i _ { t _ { u - 1 } } , u )$ as this t if such a t exists, and as $T + 1$ otherwise.   
Doing this for every tuple $( t _ { u - 1 } , i _ { t _ { u - 1 } } , u ) \stackrel {  } { \in } [ T ] \times [ N ] \times [ T ]$ constructs the delay function d.

Algorithm 4: AboveThreshold   
Input: privacy parameter $\varepsilon ,$ threshold τ   
1 $\xi \bar {  } \bar { \mathrm { L a p } } ( 2 / \bar { \varepsilon } )$   
2 for $t = s + 1 , \ldots , T$ do   
3 $\nu _ { t } \gets \mathrm { L a p } ( 4 / \varepsilon )$   
4 Receive $q _ { t }$   
5 if $q _ { t } + \nu _ { t } \ge \tau + \xi$ then   
6 $\sigma _ { t } \gets \top$   
7 Halt.   
8 else   
9 $\sigma _ { t } \gets \perp$   
10 end   
11 end

## C Lower bound for adaptive adversaries

Before presenting the proof, we first formally define differential privacy against an adaptive adversary. Let A be an online learner, D be an adversary, and $b \in \{ 0 , 1 \}$ be a hidden bit that is unknown to both A and D. Define the privacy game played between $\mathcal { A }$ and $\dot { \mathcal { D } }$ as follows. At each round $t \in [ T ]$ , the adversary D generates two loss functions $\ell _ { t } ^ { 0 }$ and $\ell _ { t } ^ { 1 }$ and the algorithm produces an output $i _ { t } ,$ both based on the interaction history up to round t − 1. After these choices are made, $\mathcal { A }$ receives $\ell _ { t } ^ { b }$ while D observes $i _ { t }$ . During the entire game, the adversary $\mathcal { D }$ ensures that $\ell _ { t } ^ { 0 } = \ell _ { t } ^ { 1 }$ for all $t \in [ T ] \setminus \{ \tilde { t } \}$ where t<sup>˜</sup> is a special round that is chosen adaptively by $\mathcal { D } \left( \mathrm { i . e . } \right.$ , the choice of $\tilde { t }$ can depend on the interaction history before round $\tilde { t } )$ . Let $\mathsf { V i e w } _ { \boldsymbol { A } , \mathcal { D } } \big ( \boldsymbol { b } \big )$ denote the view of $\mathcal { D }$ in this privacy game, including the internal randomness of D and $( i _ { 1 } , \dots , i _ { T } )$ produced by A. Differential privacy against adaptive adversaries is defined as follows.

Definition C.1 (Differential Privacy against Adaptive Adversaries [3]). An OPE algorithm A is adaptively (ε, δ)-differentially private if for any adversary D and any set O of views, we have

$$
\begin{array} { r } { \operatorname* { P r } [ \mathsf { V i e w } _ { \boldsymbol { A } , \mathcal { D } } ( 0 ) \in O ] \leq e ^ { \varepsilon } \operatorname* { P r } [ \mathsf { V i e w } _ { \boldsymbol { A } , \mathcal { D } } ( 1 ) \in O ] + \delta . } \end{array}
$$

Our proof is based on the following lower bound from [3] that holds in the high-privacy regime.

Lemma C.2 (High-Privacy Lower Bound [3]). Let T be sufficiently large and $N \geq 2 T$ . For any adaptively (ε, δ)-differentially private OPE algorithm with $\varepsilon \leq 1 / ( { \sqrt { T } } \ln T )$ and $\delta \leq T ^ { - 3 }$ , there exists an adaptive adversary generating $\ell _ { 1 } , \ldots , \ell _ { T }$ such that $L ^ { \star } = 0$ and

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \ell _ { t } ( i _ { t } ) \right] = \Omega ( T ) .
$$

Remark. The above lemma can be obtained by setting $\varepsilon \leq 1 / ( \sqrt { T } \ln T )$ and $\delta \leq T ^ { - 3 }$ in Theorem 9 of [3]. While the theorem statement does not explicitly state that $L ^ { \star } = 0$ , the proof actually constructs a realizable input sequence.

Our idea can be seen as extending the privacy amplification technique to the online setting.

Definition C.3 (Poisson Sampling). Given an OPE algorithm A and $0 \textless q \textless 1$ , define a new algorithm $A _ { q }$ as follows. It runs a copy of A. At each round t, it first draws $Z _ { t } \sim \mathsf { B e r } ( q )$ . Then, it forwards that copy’s output $i _ { t }$ as its own. After observing $\ell _ { t } ,$ , it feeds $\tilde { \ell } _ { t } = Z _ { t } \ell _ { t }$ to the copy.

We first establish the privacy guarantee of $\mathcal { A } _ { \boldsymbol { q } }$ . In the proof, we write $P \approx _ { ( \varepsilon , \delta ) } Q$ to denote that distributions P and Q are $( \varepsilon , \delta )$ -close, i.e., they satisfy the inequalities defining DP:

$$
P ( O ) \leq e ^ { \varepsilon } Q ( O ) + \delta \qquad { \mathrm { a n d } } \qquad Q ( O ) \leq e ^ { \varepsilon } P ( O ) + \delta
$$

for any event O.

Lemma C.4. Let A and $\mathcal { A } _ { \boldsymbol { q } }$ be defined as in Definition C.3. If A is adaptively $( \varepsilon , \delta )$ -differentially private, then $A _ { q }$ is adaptively $( \ln ( \bar { 1 } + q ( e ^ { \varepsilon } - 1 ) ) , q \delta )$ -differentially private.

Proof. Let D be any adversary that interacts with $A _ { q }$ . Write

$$
V _ { b } = \mathsf { V i e w } _ { \mathcal { A } _ { q } , \mathcal { D } } ( b )
$$

for $b \in \{ 0 , 1 \}$ . Let $P _ { b }$ be the distribution of $V _ { b }$ . Introduce the following three auxiliary view distributions:

$P ^ { - } \colon$ the distribution of $\mathcal { D } \mathrm { { s } }$ view in an auxiliary privacy game where $Z _ { \tilde { t } }$ is set deterministically to 0, while all other sampling coins are drawn as usual.

$P _ { 0 } ^ { + }$ : the distribution of $\mathcal { D } \mathrm { { s } }$ view in an auxiliary privacy game with $b = 0$ where $Z _ { \tilde { t } }$ is set deterministically to 1, while all other sampling coins are drawn as usual.

$P _ { 1 } ^ { + }$ : the distribution of $\mathcal { D } \mathrm { { s } }$ view in an auxiliary privacy game with $b = 1$ where $Z _ { \tilde { t } }$ is set deterministically to 1, while all other sampling coins are drawn as usual.

Since $\tilde { t }$ is chosen based on the interaction history before round $\tilde { t } ,$ while $Z _ { \tilde { t } }$ is sampled independently at round $\tilde { t } ,$ we have

$$
P _ { b } = ( 1 - q ) P ^ { - } + q P _ { b } ^ { + } ,
$$

where the first term is due to the fact that A receives $\tilde { \ell } _ { \tilde { t } } = Z _ { \tilde { t } } \ell _ { \tilde { t } } ^ { b } = 0$ whenever $Z _ { \tilde { t } } = 0$ , regardless of the value of b.

Now we derive privacy relations among $P ^ { - } , P _ { 0 } ^ { + }$ , and $P _ { 1 } ^ { + }$ . We first consider $P _ { 0 } ^ { + }$ and $P _ { 1 } ^ { + }$ . Construct an adversary $\mathcal { D } ^ { + }$ that simulates D and at each round $t \neq \hat { t }$ draws $Z _ { t } \sim \mathsf { B e r } ( q )$ and generates

$$
\begin{array} { r } { ( \tilde { \ell } _ { t } ^ { 0 } , \tilde { \ell } _ { t } ^ { 1 } ) = ( Z _ { t } \ell _ { t } ^ { 0 } , Z _ { t } \ell _ { t } ^ { 1 } ) . } \end{array}
$$

At round $\tilde { t } ,$ it generates

$$
( \tilde { \ell } _ { \tilde { t } } ^ { 0 } , \tilde { \ell } _ { \tilde { t } } ^ { 1 } ) = ( \ell _ { \tilde { t } } ^ { 0 } , \ell _ { \tilde { t } } ^ { 1 } ) .
$$

It is easy to check that $\mathcal { D } ^ { + }$ is a valid adversary in the privacy game with $\mathcal { A } \mathrm { : }$ it only generates different loss functions at round t<sup>˜</sup>. For $b \in \{ 0 , 1 \}$ , the resulting interaction is exactly the branch defining $P _ { b } ^ { + }$ By the privacy guarantee of A and the fact that the view of $\mathcal { D }$ can be seen as a post-processing of $\mathcal { D } ^ { + } \mathrm { \overline { { ~ s ~ } } v i e w } .$ , we have $P _ { 0 } ^ { + } \approx _ { ( \varepsilon , \delta ) } P _ { 1 } ^ { + }$

Next we compare $P _ { 0 } ^ { + }$ with $P ^ { - }$ . Construct an adversary $\mathcal { D } ^ { - }$ that performs the same simulation as $\mathcal { D } ^ { + }$ at round $t \neq \tilde { t } .$ . At round $\tilde { t } ,$ it generates

$$
( \tilde { \ell } _ { \tilde { t } } ^ { 0 } , \tilde { \ell } _ { \tilde { t } } ^ { 1 } ) = ( \ell _ { \tilde { t } } ^ { 0 } , 0 ) .
$$

For $b = 0 .$ , the interaction is again the branch defining $P _ { 0 } ^ { + }$ . For $b = 1$ , the interaction becomes the branch defining $P ^ { - }$ . Therefore, we have $P _ { 0 } ^ { + } \approx _ { ( \varepsilon , \delta ) } P ^ { - }$ . We can similarly prove that $P _ { 1 } ^ { + } \approx _ { ( \varepsilon , \delta ) } P ^ { - }$ We now prove the final conclusion. For any measurable set $O$ , write $x = P _ { 0 } ^ { + } ( O ) , y = P _ { 1 } ^ { + } ( O )$ and $z = P ^ { - } ( O )$ . By the pairwise $( \varepsilon , \delta )$ -closeness of $P _ { 0 } ^ { + } , P _ { 1 } ^ { + }$ , and $P ^ { - }$ , we have $x \leq e ^ { \varepsilon } y + \delta$ and $x \leq e ^ { \varepsilon } z + \delta .$ . Hence,

$$
\begin{array} { r l } & { P _ { 0 } ( O ) = ( 1 - q ) z + q x } \\ & { \qquad \le ( 1 - q ) z + q e ^ { \varepsilon } \operatorname* { m i n } ( y , z ) + q \delta } \\ & { \qquad \le ( 1 + q ( e ^ { \varepsilon } - 1 ) ) ( ( 1 - q ) z + q y ) + q \delta } \\ & { \qquad = ( 1 + q ( e ^ { \varepsilon } - 1 ) ) P _ { 1 } ( O ) + q \delta , } \end{array}
$$

where the second inequality follows because

$$
\begin{array} { r l } & { \quad ( 1 + q ( e ^ { \varepsilon } - 1 ) ) ( ( 1 - q ) z + q y ) - ( ( 1 - q ) z + q e ^ { \varepsilon } \operatorname* { m i n } ( y , z ) ) } \\ & { \geq q ( 1 - q ) ( e ^ { \varepsilon } - 1 ) ( z - \operatorname* { m i n } ( y , z ) ) + q ( 1 + q ( e ^ { \varepsilon } - 1 ) ) ( y - \operatorname* { m i n } ( y , z ) ) } \\ & { \geq 0 . } \end{array}
$$

By symmetry, $P _ { 1 } ( O ) \leq ( 1 + q ( e ^ { \varepsilon } - 1 ) ) P _ { 0 } ( O ) + q \delta$ . This proves the desired result.

Lemma C.5. Let A and $A _ { q }$ be defined as in Definition C.3. If A has expected cumulative loss at most $R _ { A } ( T )$ for any adaptively generated realizable input sequence, then $A _ { q }$ has expected cumulative loss at most $\bar { R _ { \cal A } } ( T ) \bar { / { q } }$ for any adaptively generated realizable input sequence.

Proof. Let $\ell _ { 1 } , \cdot \cdot \cdot , \ell _ { T }$ be a realizable sequence generated by an adaptive adversary. There exists $i ^ { \star } \in [ N ]$ such that ${ \ell _ { t } } ( i ^ { \star } ) = 0$ for all $t \in [ T ]$ . By definition, $\tilde { \ell } _ { t } ( i ^ { \star } ) = Z _ { t } \ell _ { t } ( i ^ { \star } ) = 0$ . This implies that the sequence $\tilde { \ell } _ { 1 } , \cdots , \tilde { \ell } _ { T }$ is also realizable. Therefore,

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } Z _ { t } \ell _ { t } ( i _ { t } ) \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \tilde { \ell } _ { t } ( i _ { t } ) \right] \leq R _ { A } ( T ) .
$$

On the other hand, since $Z _ { t }$ is independent of $i _ { t }$ and $\ell _ { t } ,$ we have

$$
\mathbb { E } \left[ Z _ { t } \ell _ { t } ( i _ { t } ) \right] = \mathbb { E } \left[ Z _ { t } \right] \mathbb { E } \left[ \ell _ { t } ( i _ { t } ) \right] = q \mathbb { E } \left[ \ell _ { t } ( i _ { t } ) \right] .\tag{1}
$$

Summing over t gives

$$
q \mathbb { E } \left[ \sum _ { \ell = 1 } ^ { T } \ell _ { t } ( i _ { t } ) \right] = \mathbb { E } \left[ \sum _ { \ell = 1 } ^ { T } Z _ { t } \ell _ { t } ( i _ { t } ) \right] \leq R _ { A } ( T ) .
$$

Rearranging the inequality yields the desired conclusion.

Theorem C.6. Let T be sufficiently large and $N \geq 2 T$ . For any adaptively $( \varepsilon , \delta )$ -differentially private OPE algorithm with $\varepsilon \leq 1$ and $\bar { \delta } \leq T ^ { - 3 }$ , there exists an adaptive adversary generating $\ell _ { 1 } , \ldots , \ell _ { T }$ such that $L ^ { \star } = 0$ and

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \ell _ { t } ( i _ { t } ) \right] = \Omega \left( \operatorname* { m i n } \left\{ T , \frac { \sqrt { T } } { \varepsilon \log T } \right\} \right) .
$$

Proof. Suppose $\mathcal { A }$ is adaptively $( \varepsilon , \delta ) \mathrm { - D P }$ and has expected cumulative loss at most $R _ { A } ( T )$ on realizable input sequences. Consider $A _ { q }$ in Definition C.3 with

$$
q = \operatorname* { m i n } \left\{ 1 , \frac { 1 } { 2 \varepsilon \sqrt { T } \ln { T } } \right\} .
$$

Lemma C.4 establishes that $A _ { q }$ is adaptively $( \ln ( 1 + q ( e ^ { \varepsilon } - 1 ) ) , q \delta )$ -differentially private. Since $\varepsilon \leq 1$ , we have

$$
\ln ( 1 + q ( e ^ { \varepsilon } - 1 ) ) \leq 2 q \varepsilon \leq 1 / ( \sqrt { T } \ln T ) .
$$

Moreover,

$$
q \delta \leq \delta \leq T ^ { - 3 } .
$$

Therefore, $\mathcal { A } _ { \boldsymbol { q } }$ is adaptively $( 1 / ( \sqrt { T } \ln T ) , T ^ { - 3 } )$ -differentially private. By Lemma $\mathrm { C } . 2 , A _ { q }$ must incur an expected cumulative loss of $\Omega ( T )$ . Lemma C.5 then yields

$$
R _ { A } ( T ) = \Omega ( q T ) = \Omega \left( \operatorname* { m i n } \left\{ T , { \frac { \sqrt { T } } { \varepsilon \log T } } \right\} \right) .
$$