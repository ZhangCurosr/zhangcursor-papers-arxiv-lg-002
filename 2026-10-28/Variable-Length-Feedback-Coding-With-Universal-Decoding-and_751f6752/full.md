# Variable-Length Feedback Coding With Universal Decoding and Bandit Channel Selection

Recep Can Yavas , Member, IEEE

## Abstract

We study variable-length feedback (VLF) coding over K additive white Gaussian noise (AWGN) channels whose noise variances are unknown to both terminals but lie in a known interval. The receiver selects one channel at each time, conveys the selection over noiseless feedback, and must learn the channel qualities while decoding, to avoid overusing suboptimal channels. This setting, which we call the VLF-Bandit problem, combines VLF coding, universal decoding, and bandit channel selection. We develop two joint coding and channel-selection schemes. The VLSF-Bandit scheme is a single-phase variable-length stop-feedback (VLSF) scheme based on a Gaussian random codebook, a score formed from a ratio of likelihood mixtures, and the asymptotically optimal kl- $\mathrm { { . U C B ^ { + + } } }$ bandit algorithm applied to the received power. The VLF-Bandit scheme sends the message in packets by Schalkwijk–Kailath transmission, uses its normalized outputs as noise samples to learn the variances, and lets its exploration packets carry message bits. We prove second-order achievability bounds for both schemes, which are first-order optimal. In the logarithm of the achievable codebook size, the magnitude of the log N coefficient, with N the average decoding time, is a baseline of one plus a channel-exploration term for VLSF-Bandit. For VLF-Bandit, this magnitude is smaller, with a universal-decoding term replacing the baseline and a smaller channel-exploration term. The VLF-Bandit bound improves on the universal VLF bound of Yavas and Tan (2025) for $K = 1$ and on the bound of an explore-then-commit strategy whenever $K \geq 2$

## Index Terms

Variable-length feedback codes, universal channel coding, multi-armed bandits, AWGN channel, Schalkwijk–Kailath scheme, KL-UCB algorithms.

## I. INTRODUCTION

Feedback does not increase the capacity of memoryless channels [1]. Yet it can simplify capacity-achieving schemes [2], [3]. The benefits of feedback are more pronounced for variable-length feedback (VLF) codes, in which transmission stops at a random time determined by the noise realization. Burnashev [4] establishes the optimal error exponent of VLF coding over a discrete memoryless channel (DMC), given by

$$
\operatorname* { l i m } _ { \epsilon \to 0 } { \frac { \log { \frac { 1 } { \epsilon } } } { \mathbb { E } \left[ \tau \right] } } = C _ { 1 } \left( 1 - { \frac { R } { C } } \right) ,\tag{1}
$$

where $C$ is the capacity of the DMC, $R \in ( 0 , C )$ is the rate, ϵ is the average error probability, E [τ ] is the average decoding time, and

$$
C _ { 1 } \triangleq \operatorname* { m a x } _ { x , x ^ { \prime } \in \mathcal { X } } D ( P _ { Y | X = x } | | P _ { Y | X = x ^ { \prime } } )\tag{2}
$$

is the Kullback–Leibler (KL) divergence between the conditional output distributions induced by the two most distinguishable input symbols. For every rate below capacity, the exponent in (1) exceeds the random-coding error exponent for fixed-length codes without feedback [5].

## A. VLF Literature

Yamamoto and Itoh [6] propose a simple alternative to Burnashev’s scheme that still achieves the optimal exponent in (1). It alternates fixed-length communication and confirmation phases, and in each confirmation phase, the transmitter sends one of two control sequences, built from the two input symbols that attain (2), to tell the receiver whether to accept its estimate. Chen et al. [7] derive non-asymptotic achievability bounds for a variant with a finite number of feedback instances, in which the phases may have different lengths and the outputs of a rejected phase are kept.

Error-exponent analysis says little about the maximum achievable rate in the finite-blocklength regime, where the target rate is close to capacity. To address this, Polyanskiy et al. [8] extend Burnashev’s work to non-vanishing error probabilities and derive achievability and converse bounds on the logarithm of the maximum achievable codebook size log $M _ { \mathrm { V L F } } ^ { * } ( N , \epsilon )$ under an average decoding-time constraint N and an average error probability constraint ϵ. They show that

$$
\frac { N C } { 1 - \epsilon } - \log N + O ( 1 ) \leq \log M _ { \mathrm { V L F } } ^ { * } ( N , \epsilon ) \leq \frac { N C } { 1 - \epsilon } + \frac { h ( \epsilon ) } { 1 - \epsilon } ,\tag{3}
$$

where $h ( \epsilon ) \triangleq - \epsilon \log \epsilon - ( 1 - \epsilon ) \log ( 1 - \epsilon )$ is the binary entropy function. The bounds in (3) imply that the ϵ-capacity of VLF codes is $\frac { C } { 1 - \epsilon }$ and that the achievable second-order term in the rate scales as $O ( { \textstyle { \frac { \log N } { N } } } )$ . The lower bound in (3) is achieved by variable-length stop-feedback (VLSF) codes, in which a single feedback bit indicates whether the transmitter should stop or continue. VLSF codes, a subset of VLF codes, use codewords that are fixed before communication starts. In [9, Th. 1], Truong and Tan establish the corresponding bounds for VLSF codes over the additive white Gaussian noise (AWGN) channel unde an average power constraint. Variants of VLSF codes with a finite number of feedback instances appear in [10]–[15], and multi-transmitter extensions in [14], [16]. For VLF codes, Yavas and Tan [17] improve the coefficient of the log N term in the l b d i f $- 1 \mathrm { t o } - \frac { C } { C _ { 1 } }$ by modifying the Yamamoto–Itoh construction to let the communication and confirmation phases have variable lengths. Their scheme does not belong to the VLSF subset.

A complementary line of work studies deterministic schemes built on posterior-matching partitions. For symmetric binaryinput channels, Naghshvar et al. [18] achieve Burnashev’s exponent with a single-phase small-enough-difference (SED) encoder, which splits the message set at each time into two subsets of nearly equal posterior probability, and they extend the idea to arbitrary DMCs in [19]. Yang et al. [20] extend the SED encoder to binary asymmetric channels, Antonini et al. [21] propose a posterior-matching encoder of polynomial complexity for the binary symmetric channel (BSC), and Yavas [22] extends both the non-asymptotic analysis and the low-complexity encoder to binary-input memoryless symmetric channels, including the binary-input AWGN channel after output quantization. With the refined analyses in [20]–[22], the SED encoder attains the same second-order term $- \frac { C } { C _ { 1 } }$ log N as in [17], with sharper non-asymptotic bounds. Yavas [23] has recently shown that no VLF code can do better over a DMC with $C > 0$ and $C _ { 1 } < \infty$ , so that

$$
\log M _ { \mathrm { V L F } } ^ { * } ( N , \epsilon ) = \frac { N C } { 1 - \epsilon } - \frac { C } { C _ { 1 } } \log N + o ( \log N ) .\tag{4}
$$

For the AWGN channel with an average power constraint, the best VLF converse known to us still has the form of the upper bound in (3), with no log N term [9].

For the AWGN channel with noiseless feedback, the Schalkwijk–Kailath (SK) scheme [3], [24] achieves every rate below capacity, with an error probability that decays doubly exponentially in the blocklength. The transmitter first sends the message as a point of a pulse-amplitude constellation. At each later time, the transmitter learns the receiver’s current estimation error through the feedback and sends this error multiplied by a gain that depends on the noise variance. Gallager and Nakiboglu [˘ 25] derive the scheme from Elias’s transmission of a Gaussian random variable over a Gaussian channel with feedback. Shayevitz and Feder [26] identify the SK scheme as the posterior-matching scheme of the AWGN channel. Truong et al. [27] combine the scheme with power control to determine the ϵ-capacity of fixed-length feedback codes under an expected power constraint.

The preceding achievability results assume that the channel law is known to the code designer. Universal codes require only that the channel belong to a prescribed family. For DMCs, Goppa’s maximum mutual information decoder [28] attain capacity universally, and in [29, Th. 10.2], Csiszar and K´ orner show that the same decoder also achieves the random-coding¨ error exponent of constant-composition codes.

There are also several works on universal coding with feedback. Tchamkerten and Telatar [30] achieve Burnashev’s exponent universally over families of BSCs and Z channels through a universal Yamamoto–Itoh scheme. Lomnitz and Feder adapt the rate to the empirical mutual information over a fixed block without a channel model [31], and vary the transmit power of the SK scheme with the empirical power of an unknown individual noise sequence so that the rate and the signal-to-noise ratio asymptotically satisfy the Gaussian capacity formula [32]. Yavas and Tan [17] analyze universal VLF codes for DMCs and for AWGN channels with unknown noise variance. For both the BSC and the AWGN channel, replacing the channel-dependent information density of a known-channel scheme by a universal score incurs an additional $\textstyle { \frac { 1 } { 2 } }$ log N cost in the second-order term of their achievability bounds. All of these universal results operate over a single unknown channel; we now consider settings in which several channels are available simultaneously.

## B. Motivation and the VLF-Bandit Problem

Modern wireless systems often operate over multiple orthogonal resources of varying quality. A user may communicate through several frequency bands, spatial streams, or subcarriers, each modeled as a memoryless channel, while hardware and scheduling constraints allow only one of these resources per channel use. The quality of each resource depends on physical parameters such as noise power, fading, path loss, and interference, which may be unknown to both terminals before transmission starts. The receiver, however, observes the output statistics and knows which resources are available, so it is natural to let the receiver act as the channel scheduler: after observing the past outputs, it selects the next channel and sends this selection to the transmitter through the noiseless feedback link.

There is a second reason to let the receiver select the channels. If the transmitter chose the channel index as an additional message-dependent input, the index sequence would itself carry information, since the receiver observes which channel is in use. With $K$ distinguishable resources, it could carry up to log $K$ nats per channel use, even when the individual channels have zero capacity. The problem would then become the universal version of Problem 7.28 in Cover and Thomas [33]. We therefore require the channel-selection rule to reside at the receiver, depending only on the common randomness, the past selections, and the past outputs, but not directly on the message.

![](images/2d7b1b506a6e437ff56008041beecf31bdc740c5e27c6fc67fee51b480b3e9a2.jpg)  
Fig. 1. VLF-Bandit system model, with the selected channel $A _ { t } = 2$ in solid black.

Under this convention, the receiver has two coupled tasks: it schedules the channels so that high-capacity channels are used more often, and it decodes the message from observations collected under an adaptively chosen channel sequence. We call this adaptive multi-channel setting the VLF-Bandit problem and consider two code classes, VLSF-Bandit and VLF-Bandit, which differ in how the transmitter uses the feedback. In the VLSF-Bandit setting, the transmitted codeword is fixed before communication starts, and the feedback link carries only the channel selections and the stop signal. In the VLF-Bandit setting, the transmitter also observes the past outputs through noiseless feedback and may adapt the transmitted symbols to the feedback history.

A natural modular baseline is an explore-then-commit (ETC) strategy, which first sends pilot symbols to estimate the channel qualities and then runs a single-channel VLF code over the declared best channel. This separation has a cost, since the pilot observations probe the channels but carry no information about the message. A joint scheme instead lets exploration and communication share the channel uses, so that a use of a channel refines the receiver’s knowledge of that channel and also carries information about the message. See Figure 1 for our system model

## C. Multi-Armed Bandit Literature

The channel-selection component of the VLF-Bandit problem connects to the multi-armed bandit (MAB) literature; see [34] for a textbook treatment. In the stochastic MAB model, a learner repeatedly pulls one of K arms and observes a sample from that arm’s unknown distribution. In regret minimization [35], [36], the learner maximizes its expected cumulative reward over a fixed time horizon; in best-arm identification [37], it seeks to identify the arm with the largest mean as quickly as possible while satisfying an error probability criterion.

The scheduling problem considered here is closer in spirit to regret minimization than to best-arm identification. The receive does not identify the best channel in a separate sensing phase; it keeps selecting channels while it decodes the message. Selecting a suboptimal channel therefore creates an opportunity cost analogous to regret, even though the performance criterion is the average decoding time rather than cumulative reward.

A common approach to regret minimization is optimism in the face of uncertainty: each arm receives an index equal to the largest mean that remains statistically plausible given its observations, and the arm with the largest index is pulled. The regret of such upper confidence bound (UCB) algorithms [36] grows logarithmically in the horizon $T$ for bounded rewards, and Lai and Robbins [35] lower-bound the coefficient of log T through the KL divergences between the arms’ reward distributions. For one-parameter exponential families, the KL-UCB rule of Garivier and Cappe [ ´ 38], [39] attains this lower bound asymptotically, and the $\mathrm { k l - U C B ^ { + + } }$ variant of Menard and Garivier [´ 40] attains it and is also minimax optimal.

Bandit methods have been applied to channel access and selection in wireless networks [41]. Closer to the multi-channel coding setting, Egger et al. [42] bound the number of samples needed to identify the maximal-capacity DMC, and Weinberger and Yemini [43] study MABs with self-information rewards, in which the expected reward of an arm is the entropy of its source. Active sequential hypothesis testing [44], in which a decision maker adaptively selects sensing actions to discriminate among finitely many hypotheses, is another related framework; VLF coding over a known DMC is an instance of it.

The setting studied here falls outside the standard VLF, universal-coding, channel-identification, and bandit literature Universal VLF coding treats a single unknown channel, bandit formulations do not include a variable-length code, and channelidentification methods learn the best channel from samples that carry no coded information. Extending a universal VLF code with a bandit channel-selection rule meets three obstacles. The decoder must aggregate evidence collected under different unknown parameters into one message score. Adaptive scheduling makes each score increment depend on the history, so the analyses of previous VLF codes that rely on independent and identically distributed (i.i.d.) observations do not apply directly. Regret bounds do not immediately translate into bounds on the average decoding time, the expectation of a stopping time. A full-feedback design based on the SK scheme needs no message score, but its gains depend on the noise variance, which must be learned from the same outputs that refine the message estimate, and the estimation error enters the second-order term. We address these obstacles through the contributions listed below.

## D. Contributions

We study the VLF-Bandit problem for K AWGN channels whose noise variances are unknown but lie in a known interval. The output of channel k is $\bar { Y ^ { ( k ) } } = X + Z ^ { ( k ) }$ with $Z ^ { ( k ) } \sim { \mathcal { N } } ( 0 , \sigma _ { k } ^ { 2 } )$ , where $\sigma _ { k } ^ { 2 } \in [ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } ]$ is unknown, and the inputs $X _ { t }$ satisfy the average power constraint E $\begin{array} { r } { \left[ \sum _ { t = 1 } ^ { \tau } X _ { t } ^ { 2 } \right] \leq \mathbb { E } \left[ \tau \right] } \end{array}$ . Dividing the output by $\sigma _ { k }$ gives the form of a scalar quasi-static fading AWGN channel [45] with gain $\scriptstyle { \frac { 1 } { \sigma _ { k } } }$ . Feedback lets the decoding time adapt to this fixed gain, so the first-order term of our results is governed by the capacity of the best channel rather than by an outage capacity.

We propose two schemes, one for each code class. Our main results are second-order achievability bounds for them, with explicit coefficients of log N.

1) We formulate the VLF-Bandit problem, a model that couples VLF coding, universal decoding over unknown noise variances, and adaptive channel selection. The formulation includes both VLSF-Bandit and VLF-Bandit codes.

2) For the VLSF-Bandit setting, we develop a single-phase scheme based on a Gaussian random codebook, a score formed from a ratio of likelihood mixtures for the candidate residuals and the outputs, and an output-power $\mathrm { k l - U C B ^ { + + } }$ channelselection rule. In Theorem 1, the magnitude of the coefficient of log N consists of a baseline of one for the single-phase architecture, as in the lower bound in (3), and a channel-exploration term that weights the capacity gap of each suboptima channel by the reciprocal of the KL divergence between its output distribution and that of the best channel.

3) For the VLF-Bandit setting, we develop a scheme that divides the message into packets and sends each by SK transmission over one channel selected by the receiver. After the first symbol of a packet, each output carries the receiver’s estimation error, which depends only on past noise, so the normalized outputs are distributed as i.i.d. noise samples of the channel. Both terminals use them to estimate the noise variance, which sets the next gain, and to eliminate channels across packets. Exploration packets carry message bits at reduced power, and the final packet, sent over the selected channel, spends the energy they save. In Theorem 2, the magnitude of the coefficient of log $N$ has no baseline, and its universal-decoding term, due to the variance estimates, is $\frac { 1 } { 2 ( 1 + \sigma ^ { 2 } ) ^ { 2 } }$ for the noise variance $\sigma ^ { 2 }$ of the best channel, smaller than the baseline of one in item 2. The channel-exploration term is also smaller than in item 2, because of the energy saved in exploration and because noise samples separate the channels better than output powers do. For $K = 1$ and $\sigma ^ { 2 } = 1$ , this magnitude drops from about 0.673 in [17, Th. 5] to 0.125.

4) We further compare the joint design with an ETC strategy that identifies a channel from zero-input pilot observations by an asymptotically optimal fixed-confidence identification rule [46] and then sends the whole message over the declared channel by the SK transmission of item 3. In every problem instance with $K \geq 2 .$ , the magnitude of the coefficient of log N in Theorem 2 is strictly smaller than in the resulting ETC bound.

## E. Paper Organization

Section II introduces the notation, and Section III formulates the VLF-Bandit problem and defines the two code classes. Section IV describes the coding schemes, a single-phase VLSF-Bandit scheme and a packetized SK VLF-Bandit scheme. Section V states the main achievability results. Sections VI and VII prove the results for the two schemes. Section VIII compares the VLF-Bandit scheme with the ETC strategy. Section IX concludes the paper. Appendices A and B contain deferred proofs.

## II. NOTATION AND PRELIMINARIES

For $n \in \mathbb { N } ,$ , let $[ n ] \triangleq \{ 1 , \dots , n \}$ and $x ^ { n } \triangleq ( x _ { 1 } , \ldots , x _ { n } )$ . For real numbers a and $b ,$ we write $a \wedge b \triangleq \operatorname* { m i n } \{ a , b \}$ and $a \vee b \triangleq \operatorname* { m a x } \{ a , b \}$ , and we let $x ^ { + } \triangleq \operatorname* { m a x } \{ 0 , x \}$ . For $a \leq b , \Pi _ { [ a , b ] } ( x ) \triangleq b \wedge ( a \vee x )$ denotes the projection of $x \in \mathbb { R }$ onto $[ a , b ]$ . Ties in arg min and arg max are broken in favor of the smallest index.

The distribution of a random variable X is denoted by $P _ { X }$ . The density of the Gaussian distribution $\mathcal { N } ( 0 , v ) , v > 0$ , is denoted by $\phi _ { v } ( \cdot )$ . We use the standard asymptotic notation $o ( \cdot )$ and $O ( \cdot )$

All logarithms have base e. We use log $\mathbf { \Psi } _ { - } ( x ) \triangleq \operatorname* { m a x } \{ \log x , 0 \}$ for $x > 0$ . The binary entropy function $h ( \cdot )$ is defined as

$$
h ( \epsilon ) \triangleq - \epsilon \log ( \epsilon ) - ( 1 - \epsilon ) \log ( 1 - \epsilon ) .\tag{5}
$$

For distributions $P$ and $Q$ with $P$ absolutely continuous with respect to $Q ,$ , the KL divergence is denoted by

$$
D ( P \| Q ) \triangleq \int \log { \frac { d P } { d Q } } d P .\tag{6}
$$

An AWGN channel with noise variance $\sigma ^ { 2 } > 0$ , denoted by $\operatorname { A W G N } ( \sigma ^ { 2 } )$ , has input and output in $\mathbb { R } ,$ and its output is

$$
Y = X + Z ,\tag{7}
$$

where $Z \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ is independent of the input X. Throughout the paper, the available average power is $P = 1 ;$ an average power constraint $P > 0$ reduces to this case after dividing inputs and outputs by $\sqrt { P }$ and all noise variances by $P .$ For a signal-to-noise ratio x, define

$$
C ( x ) \triangleq { \frac { 1 } { 2 } } \log \left( 1 + x \right) , \qquad x \geq 0 ,\tag{8}
$$

$$
C _ { 1 } ( x ) \triangleq D \left( \mathcal { N } \left( 1 , \frac { 1 } { x } \right) \Vert \mathcal { N } \left( - 1 , \frac { 1 } { x } \right) \right) = 2 x ,\tag{9}
$$

where $x \ > \ 0$ . The capacity of $\operatorname { A W G N } ( \sigma ^ { 2 } )$ under this constraint is $\textstyle C \left( { \frac { 1 } { \sigma ^ { 2 } } } \right)$ , achieved by the Gaussian input distribution $X \sim \mathcal { N } ( 0 , 1 )$ . For the antipodal inputs ±1, the KL divergence between the corresponding conditional output distributions is $\textstyle C _ { 1 } \left( { \frac { 1 } { \sigma ^ { 2 } } } \right)$

The KL divergence between $\mathcal { N } ( 0 , u )$ and $\mathcal { N } ( 0 , v ) , u , v > 0 .$ , is denoted by

$$
d ( u \| v ) \triangleq D \big ( \mathcal { N } ( 0 , u ) \| \mathcal { N } ( 0 , v ) \big ) = \frac { 1 } { 2 } \Big ( \frac { u } { v } - 1 - \log \frac { u } { v } \Big ) .\tag{10}
$$

## III. PROBLEM FORMULATION

## A. Channel Models and Code Definitions

Consider K memoryless AWGN channels $( \mathrm { A W G N } ( \sigma _ { 1 } ^ { 2 } ) , \dots , \mathrm { A W G N } ( \sigma _ { K } ^ { 2 } ) )$ . Channel $k \in [ K ]$ has noise variance $\sigma _ { k } ^ { 2 }$ , unknown to both terminals but known to lie in the interval $[ \sigma _ { \mathrm { m i n } } ^ { 2 } , \sigma _ { \mathrm { m a x } } ^ { 2 } ]$ , where $0 < \sigma _ { \mathrm { m i n } } ^ { 2 } < \sigma _ { \mathrm { m a x } } ^ { 2 } <$ ∞ are known to the code designer. Under the precision parameterization $\begin{array} { r } { \eta = \frac { 1 } { \sigma ^ { 2 } } } \end{array}$ , this interval corresponds to

$$
\eta _ { \mathrm { m i n } } \triangleq \frac { 1 } { \sigma _ { \mathrm { m a x } } ^ { 2 } } , \qquad \eta _ { \mathrm { m a x } } \triangleq \frac { 1 } { \sigma _ { \mathrm { m i n } } ^ { 2 } } .\tag{11}
$$

At each time $t \in \mathbb { N } .$ , the receiver selects a channel $A _ { t } \in [ K ]$ as a function of the common randomness $U ,$ the past selections $A ^ { t - 1 }$ , and the past channel outputs $Y ^ { t - 1 }$ , and communicates this selection to the transmitter over a noiseless feedback link. The transmitter then sends a symbol $X _ { t } \in \mathbb { R }$ over channel $A _ { t } .$ . In the VLSF setting, $X _ { t }$ is a preselected function of the message and the common randomness alone. In the VLF setting, the transmitter additionally observes the past outputs $Y ^ { t - 1 }$ through full noiseless feedback and may adapt $X _ { t }$ to $Y ^ { t - 1 }$ and $A ^ { t }$ . Given $X _ { t } = x$ and $A _ { t } = k$ , the receiver observes $Y _ { t } \sim \mathcal { N } ( x , \sigma _ { k } ^ { 2 } )$ The following definitions adapt the VLF and VLSF codes in [8], [16], [17] to bandit channel selection over K memoryless The following definitions adapt the VLF and VLSF codes in [8], [16], [17] to bandit channel selection over K memoryless channels.

Definition 1 (VLF-Bandit code with an average power constraint): Fix $\epsilon \in ( 0 , 1 ) , N > 0$ , and positive integers K and $M .$ and let the message W be uniformly distributed on [M]. An (N, M, ϵ, K)-VLF-Bandit code for K unknown AWGN channels consists of the following components.

1) Common randomness: A random variable U, independent of W, taking values on an alphabet $u ,$ with distribution $P _ { U }$ whose realization is revealed to both the transmitter and the receiver before the start of transmission.

2) Channel selection functions: $\mathsf { a } _ { t } : \mathcal { U } \times [ K ] ^ { t - 1 } \times \mathbb { R } ^ { t - 1 } \to [ K ]$ such that

$$
A _ { t } = \mathsf { a } _ { t } ( U , A ^ { t - 1 } , Y ^ { t - 1 } ) , \quad \forall t \geq 1 .\tag{12}
$$

3) Encoding functions: $\mathsf { f } _ { t } \colon \mathcal { U } \times [ M ] \times [ K ] ^ { t } \times \mathbb { R } ^ { t - 1 } \to$ R such that

$$
X _ { t } = \mathsf { f } _ { t } ( U , W , A ^ { t } , Y ^ { t - 1 } ) , \quad \forall t \geq 1 .\tag{13}
$$

4) Stopping time: A random time $\tau \in \mathbb { Z } _ { > 0 }$ that is a stopping time with respect to the filtration $\{ \sigma ( U , A ^ { t } , Y ^ { t } ) \} _ { t \geq 0 }$ , satisfying $\bar { \mathfrak { L } } \left[ \tau \right] \leq N$

5) Decoding function: A function $\begin{array} { r } { \mathsf { g } _ { \tau } : \mathcal { U } \times [ K ] ^ { \tau } \times \mathbb { R } ^ { \tau } \to [ M ] } \end{array}$ producing the estimate $\widehat { W } = \mathtt { g } _ { \tau } ( U , A ^ { \tau } , Y ^ { \tau } )$ , which satisfies P $\left\lceil { \widehat { W } } \neq W \right\rceil \leq \epsilon$

6) Average power constraint: The transmitted symbols satisfy

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { \tau } X _ { t } ^ { 2 } \right] \leq \mathbb { E } \left[ \tau \right] .\tag{14}
$$

Definition 2 (VLSF-Bandit code with an average power constraint): An $( N , M , \epsilon , K ) – \nabla \mathrm { L S F }$ -Bandit code, where $\epsilon \in ( 0 , 1 )$ $N > 0$ , and $K$ and M are positive integers, is defined as in Definition 1, except that the encoding functions in item 3) are replaced by

3’) Encoding functions: $\mathsf { f } _ { t } \colon \mathcal { U } \times [ M ] \to \mathbb { R }$ such that

$$
X _ { t } = \mathsf { f } _ { t } ( U , W ) , \quad \forall t \geq 1 .\tag{15}
$$

The VLSF restriction (15) fixes the input sequence before transmission. Every $( N , M , \epsilon , K ) – \mathbf { V } \mathrm { L S F - } ]$ Bandit code is therefore an (N, M, ϵ, K)-VLF-Bandit code.

In both code classes, the common-randomness distribution $P _ { U }$ and the channel-selection, encoding, stopping, and decoding rules are specified without knowledge of $( \sigma _ { 1 } ^ { 2 } , \dots , \sigma _ { K } ^ { 2 } )$ . They may depend on N, M, ϵ, K, and the known variance bounds $\sigma _ { \mathrm { m i n } } ^ { 2 }$ and $\sigma _ { \mathrm { m a x } } ^ { 2 }$

## B. Performance Metrics

The problem instance is the noise-variance vector $\pmb { \sigma } ^ { 2 } \triangleq ( \sigma _ { 1 } ^ { 2 } , \dots , \sigma _ { K } ^ { 2 } )$ . We require the error-probability constraint in item 5) of Definition 1 and the average power constraint (14) at every noise-variance vector in $[ \sigma _ { \mathrm { m i n } } ^ { 2 } , \sigma _ { \mathrm { m a x } } ^ { 2 } ] ^ { K }$ with a unique smallest entry, and we evaluate the average decoding time at each such instance. Thus, one code must perform well at every instance, as one bandit algorithm must in regret minimization [35].

## C. Instance-Dependent Quantities

We let $v _ { k } \triangleq 1 + \sigma _ { k } ^ { 2 }$ denote the output variance on channel k under a Gaussian $\mathcal { N } ( 0 , 1 )$ input and define

$$
C ^ { ( k ) } \triangleq C \left( { \frac { 1 } { \sigma _ { k } ^ { 2 } } } \right) ,\tag{16}
$$

$$
k ^ { \star } \triangleq \underset { k \in [ K ] } { \arg \operatorname* { m i n } } \sigma _ { k } ^ { 2 } = \underset { k \in [ K ] } { \arg \operatorname* { m a x } } C ^ { ( k ) } ,\tag{17}
$$

$$
{ \cal C } _ { \star } \triangleq { \cal C } ^ { ( k ^ { \star } ) } , \qquad v _ { \star } \triangleq v _ { k ^ { \star } } = 1 + \sigma _ { k ^ { \star } } ^ { 2 } ,\tag{18}
$$

$$
\Delta _ { k } \triangleq C _ { \star } - C ^ { ( k ) } , \quad k \neq k ^ { \star } ,\tag{19}
$$

$$
D _ { k , k ^ { \star } } \triangleq d ( v _ { k } \| v _ { \star } ) , \quad k \neq k ^ { \star } ,\tag{20}
$$

$$
C _ { 1 } ^ { ( k ) } \triangleq C _ { 1 } \left( \frac { 1 } { \sigma _ { k } ^ { 2 } } \right) ,\tag{21}
$$

$$
C _ { 1 , \star } \triangleq C _ { 1 } ^ { ( k ^ { \star } ) } .\tag{22}
$$

We assume throughout the paper that the best channel $k ^ { \star }$ is unique, which implies $\Delta _ { k } > 0$ and $D _ { k , k ^ { \star } } > 0$ for $k \neq k ^ { \star }$

## IV. CODING SCHEMES

We describe a VLSF-Bandit scheme and a VLF-Bandit scheme, which instantiate the code definitions in Section III.

## A. VLSF-Bandit Components

The VLSF-Bandit scheme uses a random Gaussian codebook, a score formed from the ratio of two likelihood mixtures, and an output-power-based channel-selection rule.

1) Random Gaussian codebook: The common randomness U includes M independent infinite-length codewords $\mathbf { c } ( m ) \triangleq$ $( \mathbf { c } _ { 1 } ( m ) , \mathbf { c } _ { 2 } ( m ) , \ldots )$ $m \in \ [ M ]$ , whose entries $\mathbf { c } _ { t } ( m ) , t \ \geq \ 1$ , are independent $\mathcal { N } ( 0 , 1 )$ random variables. At time $t ,$ the transmitter sends $X _ { t } = \mathbf { c } _ { t } ( W )$ over channel $A _ { t }$ . Since the channel-selection rule uses only past selections and outputs, under the random-coding ensemble the output satisfies $Y _ { t } \sim \mathcal { N } ( 0 , v _ { k } )$ conditioned on $( A ^ { t - 1 } , Y ^ { t - \bar { 1 } } )$ and $A _ { t } = k$

2) Per-channel statistics and exploration budget: For each channel k, define its set and number of uses up to time n by

$$
\begin{array} { r } { \mathcal { T } _ { k } ( n ) \overset { \Delta } { = } \{ i \in [ n ] : A _ { i } = k \} , \qquad N _ { k } ( n ) \overset { \Delta } { = } | \mathcal { T } _ { k } ( n ) | . } \end{array}\tag{23}
$$

For a candidate message $m \in [ M ]$ , a channel $k ,$ and a time $n ,$ define

$$
\zeta _ { k , m } ( n ) \triangleq \sum _ { t \in { \mathcal T } _ { k } ( n ) } \bigl ( Y _ { t } - \mathbf c _ { t } ( m ) \bigr ) ^ { 2 } , \qquad \Xi _ { k } ( n ) \triangleq \sum _ { t \in { \mathcal T } _ { k } ( n ) } Y _ { t } ^ { 2 } .\tag{24}
$$

The channel-selection rule uses the exploration budget function in [40, eq. (3)], which is defined for a design horizon $T \in \mathbb { N }$ with $T \geq K$ and an integer $s \geq 1$ by

$$
g _ { T } ( s ) \triangleq \log _ { + } \left( \frac { T } { K s } \left( 1 + \log _ { + } ^ { 2 } \left( \frac { T } { K s } \right) \right) \right) , \qquad G _ { K } ( T ) \triangleq g _ { T } ( 1 ) .\tag{25}
$$

3) Ratio-of-mixtures score: For a candidate message m and a channel $k ,$ the receiver forms two Gaussian likelihoods from the uses $t \in \mathcal { T } _ { k } ( n )$ , one of the residuals $Y _ { t } - \mathbf { c } _ { t } ( m )$ with variance s and one of the outputs $Y _ { t }$ with variance $1 + s ,$ and mixes each over the unknown noise variance s. Both mixtures use the log-uniform prior on the enlarged interval $\Big [ \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } , 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } \Big ]$ , which is the Jeffreys prior of the AWGN channel family restricted to this interval. The per-channel and total scores are

$$
\begin{array} { l } { S _ { k , n } ( m ) \triangleq \log \frac { { \displaystyle \int _ { \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } } { 2 } } ^ { 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } } } s ^ { - \frac { N _ { k } ( n ) } { 2 } } \exp \left\{ - \frac { \zeta _ { k , m } ( n ) } { 2 s } \right\} \frac { d s } { s } } { { \displaystyle \int _ { \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } } { 2 } } ^ { 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } } } ( 1 + s ) ^ { - \frac { N _ { k } ( n ) } { 2 } } \exp \left\{ - \frac { \Xi _ { k } ( n ) } { 2 ( 1 + s ) } \right\} \frac { d s } { s } } , } \\ { S _ { n } ( m ) \triangleq \displaystyle \sum _ { k = 1 } ^ { K } S _ { k , n } ( m ) . } \end{array}\tag{26}
$$

(27)

The normalizing factors of the prior and of the Gaussian densities cancel in (26). If the prior were a point mass at the true noise variance $\sigma _ { k } ^ { 2 } , S _ { k , n } ( m )$ would be the information density between the candidate codeword and the outputs, accumulated over the uses of channel k.

4) Output-power $k l - U C B ^ { + + }$ channel selection: Because $\textstyle C \left( { \frac { 1 } { \sigma ^ { 2 } } } \right)$ is strictly decreasing in $\sigma ^ { 2 } ,$ , the largest-capacity channel is the one with the smallest output variance $v _ { k } = 1 + \sigma _ { k } ^ { 2 }$ under the random-coding ensemble. For $N _ { k } ( t ) \geq 1$ , define the constrained empirical output variance

$$
\widehat { v } _ { k } ( t ) \triangleq \Pi _ { [ 1 + \sigma _ { \operatorname* { m i n } } ^ { 2 } , 1 + \sigma _ { \operatorname* { m a x } } ^ { 2 } ] } \left( \frac { \Xi _ { k } ( t ) } { N _ { k } ( t ) } \right) .\tag{28}
$$

Set $b _ { k } ( t ) = \sigma _ { \operatorname* { m i n } } ^ { 2 }$ for $N _ { k } ( t ) = 0$ . For $N _ { k } ( t ) \geq 1$ and the design horizon $T$ in (25), let $b _ { k } ( t )$ be the smallest noise variance whose induced output variance remains plausible under the Gaussian KL confidence constraint, given by

$$
b _ { k } ( t ) \triangleq \operatorname* { i n f } \left\{ \widetilde { \sigma } ^ { 2 } \in [ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } ] \colon N _ { k } ( t ) d \big ( \widehat { v } _ { k } ( t ) \big \| 1 + \widetilde { \sigma } ^ { 2 } \big ) \leq g _ { T } \left( N _ { k } ( t ) \right) \right\} .\tag{29}
$$

After the round-robin initialization $A _ { t } = t$ for $t \in [ K ]$ , the receiver selects, for $t \geq K + 1$ and until the decoder stops,

$$
A _ { t } \triangleq \arg \operatorname* { m a x } _ { k \in [ K ] } C \left( \frac { 1 } { b _ { k } ( t - 1 ) } \right) .\tag{30}
$$

The rule (30) selects the channel with the largest optimistic capacity estimate, which is the $\mathrm { k l - U C B ^ { + + } }$ index in [40] applied to the output power.

## B. VLSF-Bandit Scheme

The VLSF-Bandit scheme realizes Definition 2. The receiver forms the scores $S _ { n } ( m )$ in (27) for each candidate message $m \in [ M ]$ and selects channels according to (30). Fix $M \geq 2 , \epsilon \in ( 0 , 1 )$ , and $\epsilon _ { 0 } \in [ 0 , \epsilon )$ . Set

$$
\epsilon _ { \mathrm { d e c } } \triangleq \frac { \epsilon - \epsilon _ { 0 } } { 1 - \epsilon _ { 0 } } ,\tag{31}
$$

$$
\gamma \triangleq \log { \frac { M - 1 } { \epsilon _ { \mathrm { d e c } } } } + \sqrt { \log \left( 1 + \log \frac { M - 1 } { \epsilon _ { \mathrm { d e c } } } \right) } ,\tag{32}
$$

$$
T \triangleq K \vee \lceil \gamma \log ^ { 2 } \gamma \rceil .\tag{33}
$$

The decoder stops at

$$
\tau \triangleq \operatorname* { i n f } \left\{ n \geq 1 \colon \operatorname* { m a x } _ { m \in [ M ] } S _ { n } ( m ) \geq \gamma \right\}\tag{34}
$$

and decodes to

$$
\widehat { W } \triangleq \underset { m \in [ M ] } { \arg \operatorname* { m a x } } S _ { \tau } ( m ) .\tag{35}
$$

Following the stop-at-time-zero strategy in [8, Proof of Th. 2], the decoder uses a component of the common randomness, independent of the codebook, the message, and the channel noises, to stop at time zero and output message 1 with probability $\epsilon _ { \mathrm { 0 } } .$ With probability $1 - \epsilon _ { 0 }$ , the terminals run the procedure above.

![](images/2133ac7e2b1856918e1aa477e8647c1b8f1598ee9c38139ae85a4696043b86f6.jpg)  
Fig. 2. The VLF-Bandit scheme. With probability $\epsilon _ { 0 } ,$ the decoder stops at time zero; otherwise, $\ell _ { 0 }$ zero inputs on each channel precede a preliminary packet, $\bar { B { \mathbf { \alpha } } } \geq 0$ exploration packets, and a final packet. Top: the bits of $W - \bar { 1 }$ that each packet carries. Heights: the nominal powers, with the saved allocation Ω in green. Bottom: the number of channel uses of each phase, for the final packet on the best channel. Not to scale. The quantities are defined in Sections IV-C to IV-C5.

## C. VLF-Bandit Scheme

The VLF-Bandit scheme realizes Definition 1. It divides the message into successive segments, called packets, and sends each packet by SK feedback transmission [3], [24] over one channel, which the receiver selects. After the first symbol of a packet, every use is a refinement of the receiver’s message estimate. In a refinement, the transmitter sends a scaled version of the receiver’s current estimation error, and the receiver normalizes the output by a factor computed from the earlier outputs of the packet. Both terminals store these normalized outputs and use them to estimate the noise precision within a packet and to compare the channels across packets. After zero-input observations on every channel, the scheme sends a preliminary packet, zero or more short exploration packets, which carry message bits at powers of at most one, and a final packet, which carries the remaining bits with the power saved in exploration (Figure 2).

1) Message packets: Fix $\epsilon \in ( 0 , 1 )$ and a large enough message size $M ,$ , a power of two, and define the construction scale

$$
\psi \triangleq \log \log M .\tag{36}
$$

Since log M is of order N for the message sizes of interest, ψ is of order log $N .$ , as is the second-order term. As in Section IV-B, the decoder stops at time zero and outputs message 1 with probability

$$
\epsilon _ { 0 } \triangleq \epsilon - \frac { 1 } { \log M } .\tag{37}
$$

With probability $1 - \epsilon _ { 0 } .$ , the terminals execute the transmission procedure below, and the margin $\frac { 1 } { \log M }$ is reserved for decoding errors on this branch.

The message $W - 1$ is represented by a binary string of length $\log _ { 2 } M$ . Packet $j = 0 , 1 , . . .$ . has alphabet size $M _ { j }$ and carries the next $\log _ { 2 } M _ { j }$ unused bits, whose binary value plus one defines its message $W _ { j } \in [ M _ { j } ]$ . Its channel is denoted by $J _ { j }$

2) Feedback transmission of a packet: Fix packet $j .$ Its channel $J _ { j } { \mathrm { : } }$ , alphabet size $M _ { j }$ , and nominal powers $P _ { j , 0 } , P _ { j , 1 } , . . .$ are prescribed before it starts. The refinement powers $P _ { j , n } , n \geq 1$ , are positive, non-increasing, and bounded away from zero, and the initial power $P _ { j , 0 } > 0$ is specified separately. With the uses within the packet indexed by $n = 0 , 1 , \ldots$ , the outputs are $Y _ { j , n } = X _ { j , n } + Z _ { j , n }$ with $Z _ { j , n } \sim \mathcal { N } ( 0 , \sigma _ { J _ { i } } ^ { 2 } )$ .

The initial symbol uses pulse-amplitude modulation, as in [25, Sec. III]. The transmitter sends $X _ { j , 0 }$ , and the receiver initializes its real-valued message estimate to $\widehat { W } _ { j , 0 }$ , where

$$
X _ { j , 0 } \triangleq \frac { \sqrt { 1 2 P _ { j , 0 } } } { M _ { j } } \left( W _ { j } - \frac { M _ { j } + 1 } { 2 } \right) , \qquad \widehat { W } _ { j , 0 } \triangleq \frac { M _ { j } + 1 } { 2 } + \frac { M _ { j } Y _ { j , 0 } } { \sqrt { 1 2 P _ { j , 0 } } } .\tag{38}
$$

Let $F _ { j , n }$ denote the accumulated logarithmic error reduction after refinement $n ,$ updated in (42) below, with $F _ { j , 0 } \triangleq { 0 } .$ , and define the normalized estimation error

$$
E _ { j , n } \triangleq \frac { \sqrt { 1 2 P _ { j , 0 } } } { M _ { j } } \exp \{ F _ { j , n } \} \left( W _ { j } - \widehat { W } _ { j , n } \right) , \qquad n \geq 0 ,\tag{39}
$$

so that $E _ { j , 0 } = - Z _ { j , 0 }$ by (38). The transmitter computes $E _ { j , n - 1 }$ from $W _ { j }$ and the feedback. At refinement $n \geq 1$ , both terminals compute the precision estimate from the normalized outputs $\widetilde { Z } _ { j , 1 } , \dotsc , \widetilde { Z } _ { j , n - 1 }$ of the earlier refinements, defined in (41),

$$
\widehat { \eta } _ { j , n - 1 } \triangleq \left\{ \begin{array} { l l } { \frac { \eta _ { \operatorname* { m i n } } } { 2 } , } & { 1 \leq n \leq 9 , } \\ { \prod _ { \left[ \frac { \eta _ { \operatorname* { m i n } } } { 2 } , 2 \eta _ { \operatorname* { m a x } } \right] } \left( \frac { n - 3 } { \sum _ { i = 1 } ^ { n - 1 } \widetilde { Z } _ { j , i } ^ { 2 } } \right) , } & { n \geq 1 0 , } \end{array} \right.\tag{40}
$$

![](images/58e3f4600080dad9f77aa87dd9a9794977d74d59905a01ff14a774d25034fd1b.jpg)  
Fig. 3. Refinement $n \geq 1$ of packet j over channel $J _ { j } .$ Both terminals compute the precision estimate (dashed box), the constant $\textstyle { \frac { \eta _ { \mathrm { m i n } } } { 2 } }$ for $n \leq 9 .$

where the precision bounds are given in (11). The transmitter sends $X _ { j , n } ,$ and upon receiving $Y _ { j , n } ,$ the receiver stores the normalized output $\widetilde { Z } _ { j , n }$ , where

$$
X _ { j , n } \triangleq \sqrt { P _ { j , n } \widehat { \eta } _ { j , n - 1 } } E _ { j , n - 1 } , \qquad \widetilde { Z } _ { j , n } \triangleq \frac { Y _ { j , n } } { \sqrt { 1 + P _ { j , n } \widehat { \eta } _ { j , n - 1 } } } .\tag{41}
$$

Given the selected channel, the normalized outputs within a packet are i.i.d. $\mathcal { N } ( 0 , \sigma _ { J _ { j } } ^ { 2 } )$ ) samples (Lemma 9). The transmitter learns them through feedback, and both terminals update

$$
F _ { j , n } \triangleq F _ { j , n - 1 } + C \left( P _ { j , n } \widehat { \eta } _ { j , n - 1 } \right) ,\tag{42}
$$

$$
\widehat { W } _ { j , n } \triangleq \widehat { W } _ { j , n - 1 } + \frac { M _ { j } \exp \{ - F _ { j , n - 1 } \} \sqrt { P _ { j , n } \widehat { \eta } _ { j , n - 1 } } } { \sqrt { 1 2 P _ { j , 0 } } ( 1 + P _ { j , n } \widehat { \eta } _ { j , n - 1 } ) } Y _ { j , n } .\tag{43}
$$

Figure 3 shows one refinement. If the true precision $\frac { 1 } { \sigma _ { J _ { i } } ^ { 2 } }$ replaced $\widehat { \eta } _ { j , n - 1 }$ , then (41)–(43) would be the refinements of the SK scheme [3], [25] with powers $P _ { j , n }$ and a known noise variance. Each packet therefore runs a universal version of the SK scheme. The initial output $Y _ { j , 0 }$ is used only for message estimation.

Each packet has the error target exp{−b}, where

$$
b \triangleq \psi + \log \left( \lceil \psi ^ { 2 } \rceil + 2 \right) ,\tag{44}
$$

so that the margin $\frac { 1 } { \log M }$ in (37) is split equally among $\lceil \psi ^ { 2 } \rceil + 2$ packets, the most that the scheme transmits. Packet j stops at

$$
\tau ^ { ( j ) } \triangleq \operatorname* { i n f } \left\{ n \geq 1 \colon F _ { j , n } \geq \gamma _ { j } \right\} , \qquad \gamma _ { j } \triangleq \log M _ { j } + \frac { 1 } { 2 } \log \left( \frac { 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } b } { 3 P _ { j , 0 } } \right) .\tag{45}
$$

The receiver rounds $\widehat { W } _ { j , \tau ^ { ( j ) } }$ to the nearest element of $[ M _ { j } ]$ and expresses that index minus one as a binary string of length log $M _ { j }$ , the decoded packet. Packet $j$ takes $\tau ^ { ( j ) } + 1$ channel uses.

3) Initial observations and preliminary packet: Let $\ell _ { 0 } \triangleq \left\lceil \psi ^ { \frac { 1 } { 3 } } \right\rceil$ . The receiver selects the channels in increasing order, taking $\ell _ { 0 }$ zero-input observations on each. For channel k, the sample count and the empirical variance are initialized to

$$
\widetilde L _ { k } \triangleq \ell _ { 0 } , \qquad \widehat s _ { k } \triangleq \frac { 1 } { \ell _ { 0 } } \sum _ { n = 1 } ^ { \ell _ { 0 } } Y _ { ( k - 1 ) \ell _ { 0 } + n } ^ { 2 } .\tag{46}
$$

Packet zero, the preliminary packet, uses the channel

$$
J _ { 0 } \triangleq \underset { k \in [ K ] } { \arg \operatorname* { m i n } } \Pi _ { \left[ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } \right] } \left( \widehat { s } _ { k } \right)\tag{47}
$$

and carries the first $\lceil \psi ^ { 2 } \rceil$ bits, with

$$
M _ { 0 } \triangleq 2 ^ { \lceil \psi ^ { 2 } \rceil } , \qquad P _ { 0 , n } \triangleq 1 , \quad n \geq 0 .\tag{48}
$$

With its order- $- \psi ^ { 2 }$ uses, this packet gives an accurate variance estimate of channel $J _ { 0 }$ , with which the other channels are compared during exploration.

After each completed packet $j$ on channel $k = J _ { j }$ , the empirical variance and then the sample count of channel k are updated by

$$
\widehat s _ { k } \gets \frac { \widetilde L _ { k } \widehat s _ { k } + \sum _ { n = 1 } ^ { \tau ^ { ( j ) } } \widetilde Z _ { j , n } ^ { 2 } } { \widetilde L _ { k } + \tau ^ { ( j ) } } , \qquad \widetilde L _ { k } \gets \widetilde L _ { k } + \tau ^ { ( j ) } ,\tag{49}
$$

and the statistics of the other channels are unchanged. These statistics retain observations across packets, whereas the precision estimate (40) uses only those of its own packet.

4) Exploration packets: Each exploration packet carries $\ell _ { 0 }$ bits, so that the channels are compared often.

$$
A = [ K ]
$$

$$
j = 0
$$

1) Form the confidence intervals

$$
\begin{array} { r } { \mathcal { T } _ { k } \triangleq \left\{ \widetilde { \sigma } ^ { 2 } \in \left[ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } \right] : \widetilde { L } _ { k } d \left( \widehat { s } _ { k } \| \widetilde { \sigma } ^ { 2 } \right) \leq b \right\} , \qquad k \in \mathcal { A } . } \end{array}\tag{50}
$$

If any interval is empty, end exploration and select channel 1 for the final packet.

2) Remove the channels whose entire interval lies above another active channel’s interval, by the simultaneous assignment

$$
\mathcal { A }  \{ k \in \mathcal { A } \colon \operatorname* { i n f } \mathcal { T } _ { k } \leq \operatorname* { m i n } _ { k ^ { \prime } \in \mathcal { A } } \operatorname* { s u p } \mathcal { T } _ { k ^ { \prime } } \} ,\tag{51}
$$

which retains a channel attaining the smallest upper endpoint. If one channel remains, end exploration and select it for the final packet. Otherwise, if the completed exploration packets have reached the exploration cap of $\lceil \psi ^ { 2 } \rceil$ channel uses, end exploration and select channel 1 for the final packet.

3) Select the active channel with the fewest retained observations and transmit packet $j + 1$ using

$$
J _ { j + 1 } \triangleq \underset { k \in \cal { A } } { \arg \operatorname* { m i n } } \widetilde { L } _ { k } , \qquad M _ { j + 1 } \triangleq 2 ^ { \ell _ { 0 } } ,\tag{52}
$$

$$
\begin{array} { r } { P _ { j + 1 , n } \triangleq \Pi _ { \left[ \frac { 1 } { \ell _ { 0 } } , 1 \right] } \left( 1 + \Pi _ { \left[ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } \right] } \left( \widehat { s } _ { J _ { 0 } } \right) - \Pi _ { \left[ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } \right] } \left( \widehat { s } _ { J _ { j + 1 } } \right) \right) , \qquad n \ge 0 . } \end{array}\tag{53}
$$

Complete this packet even if its length crosses the exploration cap. Apply (49), set $j  j + 1$ , and return to step 1.

At termination, the number of exploration packets is $B = j ,$ , and the final channel is denoted by $J _ { B + 1 }$ . The last exploration packet starts before the exploration cap is reached, and each packet takes at least one use, so $B \le \lceil \psi ^ { 2 } \rceil$ and at most $\bar { \lceil \psi ^ { 2 } \rceil } + 2$ packets are transmitted.

5) Final packet and decoding: Packet $B + 1$ uses channel $J _ { B + 1 }$ and carries all remaining bits, so that

$$
\log M _ { B + 1 } \triangleq \log M - \log M _ { 0 } - B \ell _ { 0 } \log 2 .\tag{54}
$$

Exploration saves the allocation Ω, which the final packet spreads over its first $n _ { \mathrm { b } }$ refinements, where

$$
\Omega \triangleq \sum _ { j = 1 } ^ { B } \left( 1 - P _ { j , 0 } \right) \left( \tau ^ { ( j ) } + 1 \right) , \qquad n _ { \mathrm { b } } \triangleq \left\lceil 1 + \Omega ^ { 2 } \right\rceil ,\tag{55}
$$

$$
P _ { B + 1 , 0 } \triangleq 1 , \qquad P _ { B + 1 , n } \triangleq \left\{ 1 + \frac { \Omega } { n _ { \mathrm { b } } } , \quad 1 \leq n \leq n _ { \mathrm { b } } , \right.\tag{56}
$$

The decoder concatenates the decoded binary strings of packets $0 , 1 , \ldots , B + 1$ in this order and adds one to the resulting binary value to obtain $\widehat { W } \in [ M ]$ . The transmission branch uses $B + 2$ packets and has decoding time

$$
\tau = K \ell _ { 0 } + \sum _ { j = 0 } ^ { B + 1 } \left( \tau ^ { \left( j \right) } + 1 \right) .\tag{57}
$$

Packet parameters and channel selections use only the initial zero-input observations and the normalized outputs of completed packets, which both terminals know through feedback. Decoded packets are used only for final decoding. The parameters of the scheme depend only on $M , \epsilon , K$ , and the known variance bounds.

## V. MAIN RESULTS

For the schemes in Section IV, we state second-order achievability bounds for the VLSF-Bandit scheme (Theorem 1) and the VLF-Bandit scheme (Theorem 2).

## A. VLSF-Bandit Scheme

We first state the achievability bound for the single-phase scheme.

Theorem 1 (Second-order achievability bound for the VLSF-Bandit scheme): Fix $\epsilon \in \mathsf { \Gamma } ( 0 , 1 )$ and $K \geq 1$ , and fix the VLSF-Bandit scheme in Section IV-B with the stop-at-time-zero strategy at $\begin{array} { r } { \epsilon _ { 0 } = \epsilon - \frac { 1 } { \log ( M - 1 ) } } \end{array}$ . For every problem instance ${ \pmb \sigma } ^ { 2 } \in [ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } ] ^ { K }$ with a unique best channel $k ^ { \star }$ , this scheme is, for all sufficiently large M, an (N, M, ϵ, K)-VLSF-Bandit code with N satisfying, as $M \to \infty$

$$
\log M \geq \frac { N C _ { \star } } { 1 - \epsilon } - \left( 1 + \sum _ { k \neq k ^ { \star } } \frac { \Delta _ { k } } { D _ { k , k ^ { \star } } } \right) \log N + o ( \log N ) .\tag{58}
$$

Proof: See Section VI.

The magnitude of the coefficient of log N in (58) decomposes as

$$
\underbrace { \vphantom { \biggl ( } \frac { 1 } { \sqrt { \sum _ { k = 1 } ^ { 3 } } } + \sum _ { k \neq k ^ { \star } } \frac { \Delta _ { k } } { D _ { k , k ^ { \star } } } \mathrm { ~ . ~ } } _ { \mathrm { c h a n n e l ~ e x p l o r a t i o n } } + \underbrace { \vphantom { \biggl ( } \frac { 1 } { \sqrt { \sum _ { k = 1 } ^ { 3 } } } \biggr ) } _ { \mathrm { c h a n n e l ~ e x p l o r a t i o n } } .\tag{59}
$$

The first term is the single-phase VLSF baseline. It is the magnitude of the coefficient in the achievability part of (3) and arises from the threshold decoder and the stop-at-time-zero strategy. For the transmitted message, the logarithm of each likelihood mixture in (26) falls below that of the likelihood at the true noise variance by about $\frac { 1 } { 2 }$ log $N _ { k } ( n )$ , and the $\frac { 1 } { 2 }$ log $N _ { k } ( n )$ terms cancel in the ratio. The remaining universal-decoding cost, from score fluctuations and the square-root term in (32), is absorbed in the $o ( \log N )$ remainder. For $K = 1$ , the coefficient is one for every admissible noise variance, without the cost $\frac { 1 } { 2 }$ of the universal score in [17].

The last term is the time cost of suboptimal channel selections. Under the Gaussian input distribution, each use of channel $k \neq k ^ { \star }$ incurs an expected information loss of $\Delta _ { k }$ relative to the best channel. The Lai–Robbins lower bound [35] states that, for any bandit algorithm whose expected number of suboptimal arm pulls is $o ( n ^ { \alpha } )$ for every $\alpha > 0$ at every admissible problem instance, the expected number of uses of each suboptimal channel $k \neq k ^ { \star }$ satisfies

$$
\operatorname* { l i m } _ { n \to \infty } \frac { \mathbb { E } \left[ N _ { k } ( n ) \right] } { \log n } \geq \frac { 1 } { D _ { k , k ^ { \star } } } .\tag{60}
$$

The kl- ${ \mathrm { . U C B ^ { + + } } }$ rule attains the right-hand side of (60) [40, Th. 2]. Our bound retains $\frac { 1 } { D _ { k , k ^ { \star } } }$ at the random decoding time and weights it by the capacity gap $\Delta _ { k }$

## B. VLF-Bandit Scheme

We next state the achievability bound for the VLF-Bandit scheme in Section IV-C. For each suboptimal channel $k \neq k ^ { \star }$ define the limiting exploration power

$$
P _ { k } ^ { \star } \triangleq \left( v _ { \star } - \sigma _ { k } ^ { 2 } \right) ^ { + } ,\tag{61}
$$

which the power (53) of an exploration packet on channel k approaches as $M  \infty .$

Theorem 2 (Second-order achievability boundfor the VLF-Bandit scheme): Fix $\epsilon \in ( 0 , 1 )$ and $K \geq 1$ , and fix the VLF-Bandit scheme in Section IV-C. For every problem instance ${ \pmb \sigma } ^ { 2 } \in [ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } ] ^ { K }$ with a unique best channel $k ^ { \star }$ , this scheme is, for all sufficiently large M, an $( N , M , \epsilon , K ) – \mathrm { V L F } .$ -Bandit code with N satisfying, as $M \to \infty$ through powers of two,

$$
\log M \geq \frac { N C _ { \star } } { 1 - \epsilon } - \left( \frac { 1 } { 2 v _ { \star } ^ { 2 } } + \sum _ { k \neq k ^ { \star } } \frac { C _ { \star } - C \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) - \frac { 1 - P _ { k } ^ { \star } } { 2 v _ { \star } } } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } \right) \log N + o ( \log N ) .\tag{62}
$$

Proof: See Section VII.

The magnitude of the coefficient of log N in (62) decomposes as

$$
\underbrace { \frac { 1 } { 2 v _ { \star } ^ { 2 } } } _ { \mathrm { u n i v e r s a l ~ d e c o d i n g } } + \underbrace { \sum _ { k \ne k ^ { \star } } \frac { C _ { \star } - C \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) - \frac { 1 - P _ { k } ^ { \star } } { 2 v _ { \star } } } _ { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } } _ { \mathrm { c h a n n e l ~ e x p l o r a t i o n } } .\tag{63}
$$

The decomposition (63) has no baseline, because the packet error targets in Section IV-C affect only the $o ( \log N )$ remainder. The universal-decoding term is the cost of estimating the noise precision within the final packet. After m refinements, the precision estimate is nearly unbiased with variance of order $\textstyle { \frac { 1 } { m } } ,$ , so the concavity of the error reduction (42) gives an expected loss of the same order at the next refinement. These losses accumulate over the order-log M refinements of the packet to a cost of order log log M. For a single AWGN channel with unknown noise variance $( K = 1 )$ , the bound improves the magnitude $\begin{array} { r } { \frac { 1 } { 2 } + \frac { C _ { \star } } { C _ { 1 , \star } } } \end{array}$ in [17, Th. 5] to $\frac { 1 } { 2 v _ { \star } ^ { 2 } }$ . With known noise variance, the SK scheme can be shown to achieve the second-order term −O(log log N), improving the coefficient of log N in [17, Th. 4] from $- \frac { C _ { \star } } { C _ { 1 , \star } }$ to zero.

The channel-exploration term accounts for both communication during exploration and reuse of the saved allocation. At the limiting power $P _ { k } ^ { \star }$ , an exploration use of channel k carries $\begin{array} { r l } { C } & { { } \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) } \end{array}$ nats instead of $C _ { \star }$ , and the final packet converts the saved allocation $1 - P _ { k } ^ { \star }$ of the use into $\frac { 1 - P _ { k } ^ { \star } } { 2 v _ { \star } }$ nats, since $\frac { 1 } { 2 v _ { \star } }$ is the derivative of the capacity of channel $k ^ { \star }$ with respect to the input power at power one. The numerator is the remaining information loss of the use, and $P _ { k } ^ { \star }$ minimizes it among powers in [0, 1].

Every VLSF-Bandit code is a VLF-Bandit code, and the magnitude of the coefficient of log N is strictly smaller in Theorem 2 than in Theorem 1. Indeed, $\begin{array} { r } { \frac { 1 } { 2 v ^ { 2 } } < 1 } \end{array}$ , and each exploration summand is smaller than $\frac { \Delta _ { k } } { D _ { k , k ^ { \star } } }$ , since its numerator is at most $\Delta _ { k }$ and its denominator exceeds $\hat { D _ { k , k ^ { \star } } }$

The bounds in Theorems 1 and 2 both attain the optimal first-order term $\frac { N C _ { \star } } { 1 - \epsilon }$ . To see this, reveal the noise variances to both terminals and simulate each selected channel on channel $k ^ { \star }$ by adding independent Gaussian noise of variance $\sigma _ { A _ { t } } ^ { 2 } - \sigma _ { k } ^ { 2 }$ from common randomness, which preserves the error probability, expected decoding time, and expected energy. The converse argument in [9, Sec. II-E] extends to the simulated full-feedback code by summing conditional AWGN information over the uses up to τ and applying concavity under (14), and gives $\left( 1 - \epsilon \right) \log M \leq C _ { \star } \mathbb { E } \left[ \tau \right] + h ( \epsilon )$ for every VLF-Bandit code, which is the upper bound in (3) with $C = C _ { \star }$

## C. Numerical Illustration

For the known variance interval [0.5, 2.5] and a single channel with $\sigma _ { 1 } ^ { 2 } = 1$ , the magnitudes of the log N coefficients in the VLSF-Bandit and VLF-Bandit bounds are 1 and 0.125, respectively. For $K = 2$ with $\sigma ^ { 2 } = ( 1 , 2 )$ , these magnitudes are approximately 4.043 and 0.754, respectively.

Figure 4 compares numerical rate evaluations of the VLF-Bandit, VLSF-Bandit, and ETC schemes at $\epsilon = 1 0 ^ { - 3 }$ and $\epsilon _ { 0 } = 0 $ together with the converse in [9]. We choose the free parameters of VLF-Bandit numerically. The plotted rate is $\frac { \log _ { 2 } M } { \widehat { N } }$ , where $\widehat { N }$ is the sample average duration. For VLSF-Bandit, this duration is the first time the transmitted message reaches the threshold γ, which upper-bounds the decoder’s stopping time.

Reading the curves at $N = 5 0 0$ , the VLF-Bandit and VLSF-Bandit rates are approximately 0.481 and 0.470 bits per channel use for $K = 1$ , and 0.450 and 0.438 for $K = 2 ,$ , respectively. At $N = 2 0 0 0$ and $K = 2$ , VLF-Bandit achieves about 0.483 bits per channel use, compared with 0.179 for ETC.

## VI. PROOF OF THE VLSF-BANDIT RESULTS

This section proves Theorem 1. Wrong-message threshold crossings and the tail of the true-message stopping time give the error bound. A comparison with the true information density, an overshoot bound, and the suboptimal channel-use counts give the expected decoding time. We then apply the stop-at-time-zero strategy and expand the bound as $M \to \infty$

For the analysis, we continue the channel-selection rule and the scores beyond τ using the infinite codebook and fresh channel noises. Let

$$
{ \mathcal { F } } _ { t } \triangleq \sigma ( A ^ { t } , X ^ { t } , Y ^ { t } ) , \qquad t \geq 0 .\tag{64}
$$

The channel choice $A _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable. On $\left\{ A _ { t } = k \right\}$ , the current transmitted symbol and channel noise, conditioned on $\mathcal { F } _ { t - 1 }$ , are independent zero-mean Gaussian variables with variances 1 and $\sigma _ { k } ^ { 2 } .$ , respectively.

## A. Supporting Lemmas

We first present three tools for the VLSF-Bandit proof.

Lemma 1 (Doob’s maximal inequalities [47, Th. $\mathbf { \xi } _ { l . { l j } . { \bf { \cdot } } }$ Let $\{ X _ { n } \} _ { n \geq 0 }$ be a non-negative submartingale. For every integer $H \geq 0 ,$

$$
\mathbb { E } \left[ \left( \operatorname* { m a x } _ { 0 \leq n \leq H } X _ { n } \right) ^ { 4 } \right] \leq \left( { \frac { 4 } { 3 } } \right) ^ { 4 } \mathbb { E } \left[ X _ { H } ^ { 4 } \right] .\tag{65}
$$

If $\{ X _ { n } \} _ { n \geq 0 }$ is a martingale with $X _ { 0 } = 1$ , then

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { 0 \leq n \leq H } X _ { n } \right] \leq \frac { e } { e - 1 } \big ( 1 + \mathbb { E } \left[ X _ { H } \log X _ { H } \right] \big ) .\tag{66}
$$

Lemma 2 (Overshoot bound): In this lemma and its proof, all notation is local. Let $\{ X _ { n } \} _ { n \geq 1 }$ be an integrable process adapted to a filtration $\{ { \mathcal F } _ { n } \} _ { n \ge 0 }$ . Fix $c \in [ 0 , \infty )$ such that, for every $n \geq 1$ and every non-negative $\mathcal { F } _ { n - 1 }$ -measurable random variable a,

$$
\begin{array} { r } { \mathbb { E } \left[ ( X _ { n } - a ) ^ { + } \mid { \mathcal F } _ { n - 1 } \right] \leq c \mathbb { P } \left[ X _ { n } > a \mid { \mathcal F } _ { n - 1 } \right] \quad \mathrm { a . s . } } \end{array}\tag{67}
$$

Fix $\omega \in \mathbb { R }$ . Let $S _ { 0 }$ be an integrable $\mathcal { F } _ { 0 }$ -measurable random variable satisfying $S _ { 0 } < \omega$ almost surely, and define

$$
S _ { n } \triangleq S _ { 0 } + \sum _ { t = 1 } ^ { n } X _ { t } , \qquad \tau \triangleq \operatorname* { i n f } \{ n \geq 1 \colon S _ { n } \geq \omega \} .\tag{68}
$$

Then, for every ${ \overline { { n } } } \in \mathbb { N } .$

$$
\mathbb { E } \left[ S _ { \tau \wedge \overline { { n } } } \right] \leq \omega + c .\tag{69}
$$

![](images/db1b4c5c00302d56796562b9889f91a3a563816c2ee84fcee6b67e5b93fb22f9.jpg)

![](images/b50d11d3cc848d3727b3290f0f212e9036d3b7e73f312bd00a8e0edd621bcee2.jpg)

![](images/f1efba7621c30d7e701be58baa5592ec91839ed707e0592dd52d3bc44cc724d2.jpg)  
Fig. 4. Rate evaluations for the known variance interval $[ 0 . 5 , 2 . 5 ] , \epsilon = 1 0 ^ { - 3 }$ , and $\epsilon _ { 0 } = 0 .$ . The instances are $\sigma _ { 1 } ^ { 2 } = 1$ for $K = 1$ and $\sigma ^ { 2 } = ( 1 , 2 )$ for $K = 2 .$ Thin and thick curves denote $K = 1$ and $K = 2 ,$ respectively; ETC uses $K = 2 .$ The VLSF-Bandit curves report simulated true-message threshold-crossing rates. Each VLF-Bandit and VLSF-Bandit rate evaluation uses at least 100 000 independent trials; each ETC evaluation uses 50 000. The dashed red curve is the converse.

Proof: For $n \geq 1$ , let $a _ { n } = ( \omega - S _ { n - 1 } ) ^ { + }$ . On $\{ \tau \geq n \}$ , we have $S _ { n - 1 } < \omega$ and $a _ { n } = \omega - S _ { n - 1 }$ . Therefore,

$$
\mathbb { E } \left[ ( S _ { \tau } - \omega ) 1 \left\{ \tau \leq \overline { { n } } \right\} \right] = \sum _ { n = 1 } ^ { \overline { { n } } } \mathbb { E } \left[ 1 \left\{ \tau \geq n \right\} \left( X _ { n } - a _ { n } \right) ^ { + } \right] \leq c \sum _ { n = 1 } ^ { \overline { { n } } } \mathbb { P } \left[ \tau \geq n , \ X _ { n } > a _ { n } \right] \leq c ,\tag{70}
$$

where the first inequality applies (67) with $a = a _ { n }$ on the event $\{ \tau \geq n \} \in \mathcal { F } _ { n - 1 }$ , and the last inequality follows because the events in the sum are disjoint. On $\{ \tau > \overline { { n } } \}$ , we have $S _ { \overline { { { n } } } } < \omega$ . Splitting $\mathbb { E } \left[ S _ { \tau \wedge \overline { { n } } } \right]$ over the events $\{ \tau \leq \overline { { n } } \}$ and $\{ \tau > \overline { { n } } \}$ and applying (70) proves (69).

Lemma 3 (Fixed-horizon $k l - U C B ^ { + + }$ pull count, adaptedfrom [40, Th. 2]): Consider a K-armed bandit whose rewards belong to a one-parameter canonical exponential family parameterized by its mean, with KL divergence $d ( \cdot \| \cdot )$ , whose raw empirical means lie in the mean domain almost surely. Fix a closed interval $[ \mu ^ { - } , \mu ^ { + } ]$ inside the mean domain, containing all arm means, on which the family variance is at most $V < \infty$ . Suppose that the best arm $k ^ { \star }$ is unique, with mean $\mu _ { \star } = \mathrm { m a x } _ { j \in [ K ] } \mu _ { j }$ Run $\mathrm { k l - U C B ^ { + + } }$ with horizon $T \geq K$ and exploration budget $g _ { T }$ in (25), projecting each empirical mean onto $[ \mu ^ { - } , \mu ^ { + } ]$ and restricting the index candidates to that interval. If $k \neq k ^ { \star }$ and

$$
\sqrt { \frac { 2 2 V K } { T } } \leq \delta \leq \frac { \mu _ { \star } - \mu _ { k } } { 3 } ,\tag{71}
$$

then

$$
\mathbb { E } \left[ N _ { k } ( T ) \right] \leq \frac { G _ { K } ( T ) } { d ( \mu _ { k } + \delta \| \mu _ { \star } - \delta ) } + \frac { ( 1 6 e ^ { 2 } + 2 ) 2 V K } { \delta ^ { 2 } } + 1 .\tag{72}
$$

Proof: Apply the decomposition in [40, eqs. (24)–(30)]. Projection toward an interval containing $\mu _ { \star }$ can only reduce the lower-tail KL deviation of the best arm’s empirical mean, and once $\begin{array} { r } { n > \frac { G _ { K } ( T ) } { d ( \mu _ { k } + \delta \Vert \mu _ { \star } - \delta ) } } \end{array}$ , an index of arm k above $\mu _ { \star } - \delta$ requires its raw empirical mean to exceed $\mu _ { k } + \delta ,$ so the same tail bounds apply. The variance bound is needed only between the cutoff means and the true means, all in $[ \mu ^ { - } , \mu ^ { + } ]$ , and the maximal inequality [40, Lemma 4] needs no variance bound outside this interval. The estimates in [40, eqs. (27) and (30)], together with the initial pull, give (72). 厂

## B. Wrong-Message Error Control

The following lemma bounds the probability that the score of a wrong message reaches a given level within a given number of uses.

Lemma 4 (Finite-horizon wrong-message control): Let $\begin{array} { r } { \beta \triangleq \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } } { \sigma _ { \operatorname* { m i n } } ^ { 2 } } } \end{array}$ and

$$
c _ { 0 } \triangleq \frac { e } { e - 1 } \left( \frac { 5 } { 4 } + K \log \left( 2 \beta \log ( 4 \beta ) \right) + \frac { K } { 2 } \right) .\tag{73}
$$

For every $u \geq 0$ and $H \in \mathbb { N } ,$ , the VLSF-Bandit process satisfies

$$
\mathbb { P } \left[ \exists m \neq W : \operatorname* { m a x } _ { 1 \leq n \leq H } S _ { n } ( m ) \geq u \right] \leq ( M - 1 ) c _ { 0 } ( 1 + \log H ) \exp \{ - u \} .\tag{74}
$$

Proof: See Appendix A.I.

## C. True-Message Score

We compare the score of the transmitted message with the information density at the true noise variances. Define

$$
I _ { n } \triangleq \sum _ { t = 1 } ^ { n } \log \frac { \phi _ { \sigma _ { A _ { t } } ^ { 2 } } ( Y _ { t } - X _ { t } ) } { \phi _ { v _ { A _ { t } } } ( Y _ { t } ) } , \qquad I _ { 0 } = 0 ,\tag{75}
$$

$$
\tau _ { W } \triangleq \operatorname* { i n f } \{ n \geq 1 \colon S _ { n } ( W ) \geq \gamma \} .\tag{76}
$$

Conditioned on $\mathcal { F } _ { t - 1 }$ , the increment $I _ { t } - I _ { t - 1 }$ has mean $C ^ { ( A _ { t } ) }$ , and $\tau \leq \tau _ { W }$ by the stopping rule. The next lemma bounds, at the time $\tau _ { W } \wedge H$ , the expected gap between $I _ { n }$ and $S _ { n } ( W )$ and the expected score itself.

Lemma 5 (True-message score): Let

$$
c _ { 1 } \triangleq 2 \beta + \frac 1 2 \log ( 4 \beta + 2 \eta _ { \mathrm { m a x } } ) ,\tag{77}
$$

$$
c _ { 2 } \triangleq 4 K ( c _ { 1 } + 8 \beta ^ { 2 } ) ,\tag{78}
$$

where $\begin{array} { r } { \beta = \frac { \sigma _ { \mathrm { m a x } } ^ { 2 } } { \sigma _ { \mathrm { m i n } } ^ { 2 } } } \end{array}$ and $\begin{array} { r } { \eta _ { \mathrm { m a x } } = \frac { 1 } { \sigma _ { \mathrm { m i n } } ^ { 2 } } } \end{array}$ . For every $\gamma > 0$ and $H \in \mathbb { N } ,$

$$
\begin{array} { r } { \mathbb { E } \left[ I _ { \tau _ { W } \wedge H } - S _ { \tau _ { W } \wedge H } ( W ) \right] \leq c _ { 2 } \sqrt { 1 + \log _ { 2 } H } , } \end{array}\tag{79}
$$

$$
\mathbb { E } \left[ S _ { \tau _ { W } \wedge H } ( W ) \right] \leq \gamma + c _ { 1 } .\tag{80}
$$

Proof: The two bounds are proved in Appendix A.II and Appendix A.III, respectively.

## D. Output-Power kl- $U C B ^ { + + }$ Pull Counts

The expected decoding time also depends on how often each suboptimal channel is used. The channel-selection rule acts on the output powers. Their negatives form the exponential-family rewards to which we apply Lemma 3.

Lemma 6 (Exponential-family reduction): For channel $k \in [ K ]$ , under the Gaussian random-coding ensemble, the reward $- Y ^ { 2 }$ , with $Y \sim { \mathcal { N } } ( 0 , v _ { k } )$ and $v _ { k } = 1 + \sigma _ { k } ^ { 2 } .$ , has a one-parameter canonical exponential-family law with mean $- v _ { k }$ . The variance of this family is at most $V _ { \mathrm { s q } } \triangleq 2 ( \overset { \cdot } { 1 } + \sigma _ { \operatorname* { m a x } } ^ { 2 } ) ^ { 2 }$ over the output-variance interval $[ 1 + \sigma _ { \mathrm { m i n } } ^ { 2 } , 1 + \sigma _ { \mathrm { m a x } } ^ { 2 } ]$ , and the KL divergence between the laws with output variances u and v is $d ( u \parallel v )$

Proof: See Appendix A.IV.

By Lemma 6, the projected empirical reward mean is $- \widehat { v } _ { k } ( t )$ . Maximizing a plausible reward mean $- ( 1 + s )$ is equivalent to minimizing s, so the selection rule (30) is the $\mathrm { k l - U C B ^ { + + } }$ rule for the rewards $- Y _ { t } ^ { 2 }$ , with $\mu _ { k } = - v _ { k }$ and $\mu _ { \star } = - v _ { \star }$ . The reward-family KL divergence satisfies

$$
d ( \mu _ { k } + \delta \| \mu _ { \star } - \delta ) = d ( v _ { k } - \delta \| v _ { \star } + \delta ) .\tag{81}
$$

The following tail bound lets us use the fixed-horizon estimates without restricting the decoding time.

Lemma 7 (Tail of the true-message stopping time): For every $\gamma > 0$ , the true-message stopping time has finite expectation. Let

$$
c _ { 3 } \triangleq \frac { 2 8 8 } { C ( \eta _ { \mathrm { m i n } } ) ^ { 4 } } + \frac { 3 8 4 c _ { 2 } ^ { 2 } } { C ( \eta _ { \mathrm { m i n } } ) ^ { 2 } } .\tag{82}
$$

If log $\gamma \geq 1 \vee \frac { 2 } { \sqrt { C ( \eta _ { \mathrm { m i n } } ) } }$ , then, at the design horizon $T = K \vee \left\lceil \gamma \log ^ { 2 } \gamma \right\rceil$

$$
\mathbb { P } \left[ \tau _ { W } > T \right] \le \frac { c _ { 3 } } { \gamma ^ { 2 } \log ^ { 3 } \gamma } ,\tag{83}
$$

$$
\mathbb { E } \left[ ( \tau _ { W } - T ) ^ { + } \right] \leq \frac { c _ { 3 } } { \gamma \log \gamma } .\tag{84}
$$

Proof: See Appendix A.V.

The next lemma bounds the expected number of uses of each suboptimal channel up to the design horizon T.

Lemma $8 \ ( k l - U C B ^ { + + }$ pull count): Under the VLSF-Bandit scheme in Section IV-B, fix a suboptimal channel k $\neq k ^ { \star }$ . Then, as $\gamma \to \infty$

$$
\mathbb { E } \left[ N _ { k } ( T ) \right] \leq \frac { \log { \gamma } } { D _ { k , k ^ { \star } } } + O \big ( ( \log \gamma ) ^ { \frac { 3 } { 4 } } \big ) .\tag{85}
$$

Proof: See Appendix A.VI.

## E. Proof of Theorem 1

We use the threshold and design horizon in Section IV-B. Until the stop-at-time-zero strategy is applied below, all probabilities and expectations are conditioned on the transmission branch.

1) Error probability bound: By Lemma $\begin{array} { r } { 7 , \mathbb { E } \left[ \tau _ { W } \right] < \infty , \mathrm { s o } \tau \leq \tau _ { W } < \infty } \end{array}$ almost surely and the decoder stops. An error by time $T$ requires a wrong-message threshold crossing by then, and an error after time $T$ requires $\tau _ { W } > T$ . For all sufficiently large $M ,$ Lemma 4 and (83) give

$$
\begin{array} { r } { \mathbb { P } \left[ \widehat { W } \neq W \right] \leq c _ { 0 } ( M - 1 ) ( 1 + \log T ) \exp \{ - \gamma \} + \frac { c _ { 3 } } { \gamma ^ { 2 } \log ^ { 3 } \gamma } . } \end{array}\tag{86}
$$

With $\begin{array} { r } { g \triangleq \log \frac { M - 1 } { \epsilon _ { \mathrm { d e c } } } } \end{array}$ , the threshold (32) is $\gamma = g + \sqrt { \log ( 1 + g ) }$ , and the choice of $\epsilon _ { \mathrm { 0 } }$ in Theorem 1 gives

$$
\frac { 1 } { \epsilon _ { \mathrm { d e c } } } = ( 1 - \epsilon _ { 0 } ) \log ( M - 1 ) \leq \gamma .\tag{87}
$$

Dividing (86) by $\epsilon _ { \mathrm { d e c } }$ and using (87) in its last term yields

$$
\frac { \mathbb { P } \left[ \widehat { W } \neq W \right] } { \epsilon _ { \mathrm { d e c } } } \leq c _ { 0 } ( 1 + \log T ) \exp \{ - \sqrt { \log ( 1 + g ) } \} + \frac { c _ { 3 } } { \gamma \log ^ { 3 } \gamma } .\tag{88}
$$

The right-hand side of (88) decays to zero as $\gamma \to \infty$ . Hence, for all sufficiently large M,

$$
\mathbb { P } \left[ \widehat { W } \neq W \right] \leq \epsilon _ { \mathrm { d e c } } ,\tag{89}
$$

and the condition in Lemma 7 and $\epsilon _ { 0 } \in [ 0 , \epsilon )$ hold as well. The value of M beyond which these hold depends only on $\epsilon , K ,$ and the known variance bounds, and is therefore the same at every admissible instance.

2) Average decoding time bound: Since $\begin{array} { r } { I _ { n } - \sum _ { t = 1 } ^ { n } C ^ { ( A _ { t } ) } } \end{array}$ is a martingale, the optional stopping theorem for bounded stopping times [48, Th. 10.10] and $C ^ { ( k ) } = C _ { \star } - \Delta _ { k } \ \mathrm { \bar { f o r } } \ k \ne k ^ { \star }$ give

$$
C _ { \star } \mathbb { E } \left[ \tau _ { W } \wedge T \right] = \mathbb { E } \left[ I _ { \tau _ { W } \wedge T } \right] + \sum _ { k \neq k ^ { \star } } \Delta _ { k } \mathbb { E } \left[ N _ { k } ( \tau _ { W } \wedge T ) \right] .\tag{90}
$$

Bounding $\mathbb { E } \left[ I _ { \tau _ { W } \wedge T } \right]$ by Lemma $5 ,$ each $N _ { k } ( \tau _ { W } \wedge T )$ by $N _ { k } ( T )$ , and $\mathbb { E } \left[ \tau _ { W } \right] - \mathbb { E } \left[ \tau _ { W } \wedge T \right] = \mathbb { E } \left[ ( \tau _ { W } - T ) ^ { + } \right]$ by (84) gives

$$
C _ { \star } \mathbb { E } \left[ \tau _ { W } \right] \leq \gamma + c _ { 1 } + c _ { 2 } \sqrt { 1 + \log _ { 2 } T } + \sum _ { k \neq k ^ { \star } } \Delta _ { k } \mathbb { E } \left[ N _ { k } ( T ) \right] + \frac { C _ { \star } c _ { 3 } } { \gamma \log \gamma } .\tag{91}
$$

Since log $T = \log \gamma + O ( \log \log \gamma )$ , the term $c _ { 2 } \sqrt { 1 + \log _ { 2 } T }$ is $O ( \sqrt { \log { \gamma } } )$ , and substituting (85) gives

$$
\mathbb { E } \left[ \tau \right] \leq \mathbb { E } \left[ \tau _ { W } \right] \leq \frac { 1 } { C _ { \star } } \left( \gamma + \sum _ { k \neq k ^ { \star } } \frac { \Delta _ { k } } { D _ { k , k ^ { \star } } } \log \gamma \right) + O \big ( ( \log \gamma ) ^ { \frac { 3 } { 4 } } \big ) .\tag{92}
$$

3) Power constraint: It remains to verify (14). For $t \geq 1$ , conditioned on $W ,$ , the outputs $Y ^ { t - 1 }$ , and the codeword symbols $\{ \mathbf { c } _ { s } ( m ) \colon s \in [ t - 1 ] , m \in [ M ] \}$ , the input $X _ { t }$ has distribution $\mathcal { N } ( 0 , 1 )$ , while $\{ \tau \geq t \}$ is determined by these variables. Thus, $\begin{array} { r } { \mathbb { E } \left[ X _ { t } ^ { 2 } \mathbf { 1 } \left\{ \tau \geq \dot { t } \right\} \right] = \mathbb { P } \left[ \tau \geq \dot { t } \right] } \end{array}$ . Interchanging the non-negative sum and expectation gives

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { \tau } X _ { t } ^ { 2 } \right] = \sum _ { t = 1 } ^ { \infty } \mathbb { E } \left[ X _ { t } ^ { 2 } 1 \left\{ \tau \geq t \right\} \right] = \sum _ { t = 1 } ^ { \infty } \mathbb { P } \left[ \tau \geq t \right] = \mathbb { E } \left[ \tau \right] .\tag{93}
$$

Hence, the power constraint (14) holds with equality. For all sufficiently large M, the error and power bounds hold at every variance vector in $[ \sigma _ { \mathrm { m i n } } ^ { 2 } , \sigma _ { \mathrm { m a x } } ^ { 2 } ] ^ { K }$

4) Second-order expansion: By (32) and (87), the threshold satisfies

$$
\gamma = \log M + \log \log M + O ( { \sqrt { \log \log M } } ) .\tag{94}
$$

The independent stop at time zero keeps the error probability at most $\epsilon _ { 0 } + ( 1 - \epsilon _ { 0 } ) \epsilon _ { \mathrm { d e c } } = \epsilon$ and multiplies the expected decoding time and energy by $1 - \epsilon _ { 0 }$ . Thus, the scheme is an $( N , M , \epsilon , K ) – \nabla \mathrm { L S F } .$ -Bandit code with $N = ( 1 - \epsilon _ { 0 } ) \mathbb { E } \left[ \tau \right]$ . Substituting (94) into (92) gives

$$
N \leq \frac { 1 - \epsilon } { C _ { \star } } \log M + \frac { 1 - \epsilon } { C _ { \star } } \left( 1 + \sum _ { k \neq k ^ { \star } } \frac { \Delta _ { k } } { D _ { k , k \star } } \right) \log \log M + O \big ( ( \log \log M ) ^ { \frac { 3 } { 4 } } \big ) .\tag{95}
$$

The converse in Section V-B and (95) imply log $N = \log \log M + O ( 1 )$ . Replacing log log M by log N costs $O ( 1 )$ , and inversion yields

$$
\log M \geq \frac { N C _ { \star } } { 1 - \epsilon } - \left( 1 + \sum _ { k \neq k ^ { \star } } \frac { \Delta _ { k } } { D _ { k , k ^ { \star } } } \right) \log N + o ( \log N ) ,\tag{96}
$$

where the remainder is $O ( ( \log N ) ^ { \frac { 3 } { 4 } } )$ . This proves Theorem 1.

## VII. PROOF OF THE VLF-BANDIT RESULTS

This section proves Theorem 2. The conditional Gaussian law of the normalized outputs gives the error and energy bounds. We then derive the packet durations, combine them with the channel-selection bounds, and apply the stop-at-time-zero strategy. Figure 5 shows how these steps depend on each other.

![](images/6d839a1572bbbafe1c3e0480dd8d330ac4165351a09f56e6f87e9a66eee44242.jpg)  
Fig. 5. Steps of the proof of Theorem 2 and the subsections that contain them. An arrow points from a result to a step that uses it, and Lemma 9 also enters the proofs of Lemmas 10–12.

Fix the problem instance and let $M \to \infty$ through powers of two. Until the stop-at-time-zero strategy is applied in Sec tion VII-F, all probabilities and expectations are conditioned on the transmission branch.

## A. Packet Lengths

Every precision estimate in (40) lies in $\left[ \frac { \eta _ { \mathrm { m i n } } } { 2 } , 2 \eta _ { \mathrm { m a x } } \right]$ . Hence, for a packet with constant power $P _ { j , n } = P _ { j , 0 }$ , the stopping rule (45) gives, for $\gamma _ { j } > 0$ , the pathwise bounds

$$
\begin{array} { r } { \frac { \gamma _ { j } } { C \left( 2 P _ { j , 0 } \eta _ { \mathrm { m a x } } \right) } \le \tau ^ { ( j ) } \le \left\lceil \frac { \gamma _ { j } } { C \left( \frac { P _ { j , 0 } \eta _ { \mathrm { m i n } } } { 2 } \right) } \right\rceil . } \end{array}\tag{97}
$$

For the preliminary and exploration packets, $\begin{array} { r } { \frac { 1 } { \ell _ { 0 } } \leq P _ { j , 0 } \leq 1 } \end{array}$ , so that

$$
0 \leq \gamma _ { j } - \log M _ { j } \leq \frac { 1 } { 2 } \log \left( \frac { 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } b \ell _ { 0 } } { 3 } \right) = O \left( \log \psi \right)\tag{98}
$$

for sufficiently large M. Since log $\psi = o \left( \ell _ { 0 } \right)$ , it follows that log $M _ { j } \leq \gamma _ { j } \leq 2$ log $M _ { j }$ for $0 \le j \le B$ . Substituting these bounds into (97), and using $C ( p a ) \geq p C ( a )$ for $0 \leq p \leq 1$ , log $M _ { j } = \ell _ { 0 }$ log 2 for $1 \leq j \leq B$ , and $\begin{array} { r } { P _ { j , 0 } \geq \frac { 1 } { \ell _ { 0 } } } \end{array}$ for the upper bound on $\tau ^ { ( j ) }$ , we obtain

$$
\frac { \log M _ { 0 } } { C \left( 2 \eta _ { \mathrm { m a x } } \right) } \le \tau ^ { ( 0 ) } \le \frac { 2 \log M _ { 0 } } { C \left( \frac { \eta _ { \mathrm { m i n } } } { 2 } \right) } + 1 ,\tag{99}
$$

$$
\frac { \ell _ { 0 } \log 2 } { C \left( 2 \eta _ { \mathrm { m a x } } \right) } \le \tau ^ { \left( j \right) } ,\tag{100}
$$

$$
\tau ^ { ( j ) } + 1 \le \frac { 2 \log 2 } { C \left( \frac { \eta _ { \mathrm { m i n } } } { 2 } \right) } \ell _ { 0 } ^ { 2 } + 2 = O \left( \psi ^ { \frac { 2 } { 3 } } \right) , \qquad 1 \le j \le B .\tag{101}
$$

The last exploration packet starts before the completed exploration packets reach $\lceil \psi ^ { 2 } \rceil$ channel uses, so (100)–(101) give

$$
\sum _ { j = 1 } ^ { B } \left( \tau ^ { \left( j \right) } + 1 \right) \le \lceil \psi ^ { 2 } \rceil + \frac { 2 \log 2 } { C \left( \frac { \eta _ { \operatorname* { m i n } } } { 2 } \right) } \ell _ { 0 } ^ { 2 } + 2 = O \left( \psi ^ { 2 } \right) ,\tag{102}
$$

$$
B \ell _ { 0 } \log 2 \leq C \left( 2 \eta _ { \mathrm { m a x } } \right) \sum _ { j = 1 } ^ { B } \tau ^ { \left( j \right) } = O \left( \psi ^ { 2 } \right) .\tag{103}
$$

By (54) and (55), the final packet therefore satisfies

$$
0 \leq \Omega = O \left( \psi ^ { 2 } \right) ,\tag{104}
$$

$$
\log M _ { B + 1 } = \log M - O \left( \psi ^ { 2 } \right) > 0 ,\tag{105}
$$

$$
n _ { \mathrm { b } } = O \left( 1 + \psi ^ { 4 } \right) = o \left( \log M \right) ,\tag{106}
$$

$$
0 \leq \frac { \Omega } { n _ { \mathrm { b } } } \leq \frac { \Omega } { 1 + \Omega ^ { 2 } } \leq \frac { 1 } { 2 } .\tag{107}
$$

Thus, the final packet has a positive integer number of bits and refinement powers in $[ 1 , \textstyle { \frac { 3 } { 2 } } ]$ , its threshold is $\gamma _ { B + 1 } =$ log $M _ { B + 1 } + O \left( \log \psi \right)$ , and, since each increment of $F _ { B + 1 , n }$ is at least $C \left( { \frac { \eta _ { \mathrm { m i n } } } { 2 } } \right)$

$$
\tau ^ { ( B + 1 ) } + 1 = O \left( \log M \right) .\tag{108}
$$

Finally, every threshold is at least $\ell _ { 0 }$ log 2 for sufficiently large M, whereas the first nine refinements of every packet use the precision $\frac { \eta _ { \mathrm { m i n } } } { 2 }$ and powers at most ${ \frac { 3 } { 2 } } .$ , so that $\begin{array} { r } { F _ { j , \mathfrak { s } } \le \mathfrak { s } C \left( \frac { 3 \eta _ { \mathrm { m i n } } } { 4 } \right) < \ell _ { 0 } } \end{array}$ log 2. Hence, $\tau ^ { ( j ) } \geq 1 0$ , and the first nine refinements of every packet are transmitted.<sup>1</sup>

## B. Normalized Outputs and Decoding Error

The analysis of a packet conditions on the information retained at its start. For $j \geq 0$ , define the retained history

$$
\begin{array} { r } { \mathcal { G } _ { j , 0 } \triangleq \sigma \left( U , Y ^ { K \ell _ { 0 } } , \left( \tau ^ { ( i ) } , \left( \widetilde { Z } _ { i , n } \right) _ { n = 1 } ^ { \tau ^ { ( i ) } } \right) _ { i = 0 } ^ { j - 1 } \right) , } \end{array}\tag{109}
$$

where $\tau ^ { ( i ) } \triangleq 0$ for a packet that is not transmitted. This history excludes the initial outputs $Y _ { i , 0 }$ and the decoded packets, which depend on the message and enter only the message estimates. It determines whether packet $j$ is transmitted and, if so, its channel, alphabet size, threshold, powers, and first use. For the analysis, continue its recursion beyond $\tau ^ { ( j ) }$ using fresh $\mathcal { N } ( 0 , \sigma _ { J _ { j } } ^ { 2 } )$ ) noises, independent of the actual transmission and of each other given $\mathcal { G } _ { j , 0 }$ . Let

$$
\mathcal { G } _ { j , n } \triangleq \mathcal { G } _ { j , 0 } \vee \sigma \left( \widetilde { Z } _ { j , 1 } , \ldots , \widetilde { Z } _ { j , n } \right) , \qquad n \ge 1 ,\tag{110}
$$

where $\vee$ denotes the join of sigma-fields, and let $\mathscr { G } _ { j , \tau ^ { ( j ) } }$ be the stopped sigma-field. Since $\{ B + 1 = j \} \in \mathcal { G } _ { j , 0 }$ and $\mathcal { G } _ { j , 0 } \subseteq \mathcal { G } _ { j + 1 , 0 } .$ the index $B + 1$ of the final packet is a stopping time for $\left( \mathcal { G } _ { j , 0 } \right) _ { i > 0 } ; \mathcal { G } _ { B + 1 , 0 }$ denotes its stopped sigma-field.

The next lemma gives the conditional law of the normalized outputs and of the normalized error.

Lemma 9 (Conditional Gaussian law): Fix a transmitted packet j. For every $n \geq 0$ , conditioned on $\mathcal { G } _ { j , 0 }$ and $W$ , the variables $\widetilde { Z } _ { j , 1 } , \ldots , \widetilde { Z } _ { j , n } , E _ { j , n }$ are independent with distribution $\mathcal { N } \left( 0 , \sigma _ { J _ { j } } ^ { 2 } \right)$ , and

$$
\mathbb { P } \left[ E _ { j , \tau ^ { ( j ) } } \leq z \mid \mathcal { G } _ { j , \tau ^ { ( j ) } } , W \right] = \int _ { - \infty } ^ { z } \phi _ { \sigma _ { J _ { j } } ^ { 2 } } ( y ) d y , \qquad z \in \mathbb { R } .\tag{111}
$$

Moreover, $\mathcal { G } _ { j , \tau ^ { ( j ) } }$ is independent of W, and, conditioned on $\mathcal { G } _ { j , 0 } ^ { } .$ , the packet message $W _ { j }$ is uniform on $[ M _ { j } ]$

Proof: Write $s = \sigma _ { J _ { i } } ^ { 2 }$ . On $\{ \tau ^ { ( j ) } \ge n \} \in \mathcal { G } _ { j , n - 1 }$ , the next refinement uses a channel and a time determined by the retained history. Since $\mathcal { G } _ { j , n - 1 } , \mathrm { ~ W ~ }$ , and $E _ { j , n - 1 }$ depend only on the common randomness, the message, and earlier channel noises, memorylessness gives $Z _ { j , n } \sim { \mathcal { N } } ( { \bar { 0 } } , s )$ conditionally on these variables. The continuation noises have the same conditional law on $\{ \tau ^ { ( \dot { j } ) } < n \}$ . The initial noise also has this law given $\mathcal { G } _ { j , 0 }$ and W, so $E _ { j , 0 } = - Z _ { j , 0 }$ proves the case $n = 0$ Suppose that $E _ { j , n - 1 }$ has distribution $\mathcal { N } ( 0 , s )$ conditioned on $\mathcal { G } _ { j , n - 1 }$ and W. By (39)–(43), whose squared gain $P _ { j , n } \widehat { \eta } _ { j , n - 1 }$ is $\mathcal { G } _ { j , n - 1 }$ -measurable,

$$
\left( \begin{array} { c } { E _ { j , n } } \\ { \widetilde { Z } _ { j , n } } \end{array} \right) = \frac { 1 } { \sqrt { 1 + P _ { j , n } \widehat { \eta } _ { j , n - 1 } } } \left( \begin{array} { c c } { 1 } & { - \sqrt { P _ { j , n } \widehat { \eta } _ { j , n - 1 } } } \\ { \sqrt { P _ { j , n } \widehat { \eta } _ { j , n - 1 } } } & { 1 } \end{array} \right) \left( \begin{array} { c } { E _ { j , n - 1 } } \\ { Z _ { j , n } } \end{array} \right) .\tag{112}
$$

Since the matrix is orthogonal, $E _ { j , n }$ and $\widetilde { Z } _ { j , n }$ are independent $\mathcal { N } ( 0 , s )$ variables conditioned on $\mathcal { G } _ { j , n - 1 }$ and $W .$ . In particular, $E _ { j , n }$ has distribution $\mathcal { N } ( 0 , s )$ conditioned on $\mathcal { G } _ { j , n } = \mathcal { G } _ { j , n - 1 } \vee \sigma \left( \widetilde { Z } _ { j , n } \right)$ and W. This completes an induction over n and proves the first assertion.

The packet length is bounded given $\mathcal { G } _ { j , 0 }$ by Section VII-A. Partitioning over $\{ \tau ^ { ( j ) } = n \}$ therefore gives (111).

Since $E _ { j , 0 } = - Z _ { j , 0 } { \mathrm { . } }$ , the rotation (112) makes the normalized outputs functions of the retained history and fresh packet noises alone. Starting with the zero-input observations, induction over packets therefore makes every retained history independent of W. Given $\mathcal { G } _ { j , 0 }$ , the packet’s bit positions are fixed, so the independent uniform bits of $W - 1$ make $W _ { j }$ uniform on $[ M _ { j } ]$ .7

An incorrect packet decision implies $\begin{array} { r } { \left| W _ { j } - \widehat { W } _ { j , \tau ^ { ( j ) } } \right| \geq \frac { 1 } { 2 } } \end{array}$ . Since $F _ { j , \tau ^ { ( j ) } } \geq \gamma _ { j }$ , the definitions (39) and (45) give

$$
\{ \{ \mathrm { p a c k e t ~ } j \mathrm { ~ i s ~ d e c o d e d ~ i n c o r r e c t l y } \} \subseteq \left\{ \left| E _ { j , \tau ^ { ( j ) } } \right| \geq \frac { \sqrt { 3 P _ { j , 0 } } \exp \{ \gamma _ { j } \} } { M _ { j } } \right\} = \left\{ E _ { j , \tau ^ { ( j ) } } ^ { 2 } \geq 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } b \right\} .\tag{113}
$$

The bound $\begin{array} { r } { \mathbb { P } \left[ | Z | \geq a \right] \leq \exp \left\{ - \frac { a ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} } \end{array}$ for $Z \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ and $a \geq 0$ , together with (111) and $\sigma _ { J _ { j } } ^ { 2 } \leq \sigma _ { \operatorname* { m a x } } ^ { 2 } .$ , gives

$$
\mathbb { P } \left[ { \mathrm { p a c k e t ~ } } j { \mathrm { ~ i s ~ d e c o d e d ~ i n c o r r e c t l y ~ } } | \mathcal { G } _ { j , 0 } \right] \leq \exp \left\{ - { \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } b } { \sigma _ { J _ { j } } ^ { 2 } } } \right\} \leq \exp \{ - b \} .\tag{114}
$$

Since whether packet j is transmitted is determined by $\mathcal { G } _ { j , 0 }$ and at most $\lceil \psi ^ { 2 } \rceil + 2$ packets are transmitted, the union bound and (44) give

$$
\mathbb { P } \left[ \widehat { W } \neq W \right] \leq \left( \lceil \psi ^ { 2 } \rceil + 2 \right) \exp \{ - b \} = \frac { 1 } { \log M } .\tag{115}
$$

## C. Power Constraint

The energy of a packet is random, because its gains depend on the precision estimates. The next lemma bounds its expected energy by its nominal allocation.

Lemma 10 (Packet energy): For every transmitted packet $j ,$

$$
\mathbb { E } \left[ \sum _ { n = 0 } ^ { \tau ^ { ( j ) } } X _ { j , n } ^ { 2 } \Bigg | \mathcal { G } _ { j , 0 } \right] \leq P _ { j , 0 } + \mathbb { E } \left[ \sum _ { n = 1 } ^ { \tau ^ { ( j ) } } P _ { j , n } \Bigg | \mathcal { G } _ { j , 0 } \right] .\tag{116}
$$

Proof: See Appendix B.I.

The preliminary and exploration packets have total nominal allocation $\begin{array} { r } { \sum _ { j = 0 } ^ { B } ( \tau ^ { ( j ) } + 1 ) - \Omega } \end{array}$ by (55), and the final packet has allocation at most $\tau ^ { ( B + 1 ) } + 1 + \Omega$ by (56). Summing (116) by the tower property, and noting that the $K \ell _ { 0 }$ initial uses carry no energy, gives

$$
\mathbb E \left[ \sum _ { t = 1 } ^ { \tau } X _ { t } ^ { 2 } \right] \leq \mathbb E \left[ \sum _ { j = 0 } ^ { B } \left( \tau ^ { ( j ) } + 1 \right) \right] - \mathbb E \left[ \Omega \right] + \mathbb E \left[ \tau ^ { ( B + 1 ) } + 1 \right] + \mathbb E \left[ \Omega \right] = \mathbb E \left[ \tau \right] - K \ell _ { 0 } \leq \mathbb E \left[ \tau \right] .\tag{117}
$$

## D. Expected Packet Duration

The duration of a packet is governed by the true-precision reduction

$$
{ \overline { { F } } } _ { j , n } \triangleq \sum _ { i = 1 } ^ { n } C \left( { \frac { P _ { j , i } } { \sigma _ { J _ { j } } ^ { 2 } } } \right) , \qquad n \geq 0 ,\tag{118}
$$

which the packet would accumulate if the precision were known. We compare $\overline { { F } } _ { j , n }$ with $F _ { j , n }$ for a transmitted packet $j$ whose refinement powers satisfy

$$
q \leq P _ { j , n } \leq 2 q , \qquad n \geq 1 ,\tag{119}
$$

for a $\mathcal { G } _ { j , 0 }$ -measurable $q \in ( 0 , 1 ]$ . By Section VII-A, this holds with $q = P _ { j , 0 }$ for the preliminary and exploration packets and with $q = 1$ for the final packet. Write $s = \sigma _ { J _ { \acute { \prime } } } ^ { 2 }$ and condition on $\mathcal { G } _ { j , 0 } ;$ by Lemma 9, the normalized outputs are then i.i.d. $\mathcal { N } ( 0 , s )$ , and the alphabet size, the threshold, and the powers are fixed. For $m \geq 9$ , refinement $m + 1$ uses the estimate $\widehat { \eta } _ { j , m }$ in place of $\textstyle { \frac { 1 } { s } }$ , and concavity of $C$ gives the non-negative Taylor remainder

$$
R _ { j , m } \triangleq C \left( \frac { P _ { j , m + 1 } } { s } \right) + \frac { P _ { j , m + 1 } s } { 2 \left( P _ { j , m + 1 } + s \right) } \left( \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right) - C \left( P _ { j , m + 1 } \widehat { \eta } _ { j , m } \right) \geq 0 .\tag{120}
$$

Since the first nine refinements use the precision $\frac { \eta _ { \mathrm { m i n } } } { 2 }$ and $\tau ^ { ( j ) } \geq 1 0 .$ , summing (120) over the refinements of the packet gives

$$
\overline { { F } } _ { j , \tau ^ { ( j ) } } = F _ { j , \tau ^ { ( j ) } } + \sum _ { n = 1 } ^ { 9 } \left[ C \left( \frac { P _ { j , n } } { s } \right) - C \left( \frac { P _ { j , n } \eta _ { \mathrm { m i n } } } { 2 } \right) \right] - \sum _ { m = 9 } ^ { \tau ^ { ( j ) } - 1 } \frac { P _ { j , m + 1 } s } { 2 ( P _ { j , m + 1 } + s ) } \left( \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right) + \sum _ { m = 9 } ^ { \tau ^ { ( j ) } - 1 } R _ { j , m } .\tag{121}
$$

In Appendix B.II, we calculate

$$
\mathbb { E } \left[ R _ { j , m } \right] = \frac { P _ { j , m + 1 } ^ { 2 } } { 2 \left( P _ { j , m + 1 } + s \right) ^ { 2 } m } + O \left( q ^ { 2 } m ^ { - \frac { 3 } { 2 } } \right) , \qquad m \ge 9 .\tag{122}
$$

The linear sum stops at $\tau ^ { ( j ) }$ , and the event $\{ \tau ^ { ( j ) } > m \}$ is determined by $\widetilde { Z } _ { j , 1 } , \dotsc , \widetilde { Z } _ { j , m - 1 }$ , which also enter $\widehat { \eta } _ { j , m }$ . We show in Appendix B.II that its expectation nevertheless satisfies

$$
\left| \mathbb { E } \left[ \sum _ { m = 9 } ^ { \tau ^ { \left( j \right) } - 1 } \frac { P _ { j , m + 1 } s } { 2 \left( P _ { j , m + 1 } + s \right) } \left( \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right) \right] \right| = O \left( q \right) .\tag{123}
$$

In (121), the first sum is at most $9 q \eta _ { \mathrm { m a x } } .$ , the overshoot $F _ { j , \tau ^ { ( j ) } } - \gamma _ { j }$ is at most $2 q \eta _ { \mathrm { m a x } }$ , and $\begin{array} { r } { \tau ^ { ( j ) } \le \left\lceil \frac { \gamma _ { j } } { C \left( \frac { q \eta _ { \mathrm { m i n } } } { 2 } \right) } \right\rceil } \end{array}$ because every increment of $F _ { j , n }$ is at least $C \left( \begin{array} { c } { { { \frac { q \eta _ { \mathrm { m i n } } } { 2 } } } } \end{array} \right)$ . As $R _ { j , m } \geq 0$ , its stopped sum is at most its sum up to this deterministic bound. Taking expectations in (121) and applying (122) and (123) yields

$$
\mathbb { E } [ \overline { { F } } _ { j , \tau ^ { ( j ) } }  \mathcal { G } _ { j , 0 }  \leq \gamma _ { j } + \sum _ { m = 9 } ^ { \lceil \frac { \gamma _ { j } } { C ( \frac { q \eta _ { \operatorname* { m i n } } } { 2 } ) } \rceil - 1 } \frac { P _ { j , m + 1 } ^ { 2 } } { 2 ( P _ { j , m + 1 } + s ) ^ { 2 } m } + O ( q ) .\tag{124}
$$

The summation in (124) produces the universal-decoding term. We now convert the capacity-sum bound into packet durations. Lemma 11 (Expected packet duration): For every transmitted preliminary or exploration packet $j ,$ as $M  \infty$ through powers of two,

$$
C \left( \frac { P _ { j , 0 } } { \sigma _ { J _ { j } } ^ { 2 } } \right) \mathbb { E } \left[ \tau ^ { \left( j \right) } + 1 \mid \mathcal { G } _ { j , 0 } \right] \leq \log M _ { j } + O \left( \log \psi \right) .\tag{125}
$$

For the final packet,

$$
C \left( \frac { 1 } { \sigma _ { _ { J _ { B + 1 } } } ^ { 2 } } \right) \mathbb { E } \left[ \tau ^ { ( B + 1 ) } + 1 \mid \mathcal { G } _ { B + 1 , 0 } \right] \leq \log M _ { B + 1 } - \frac { \Omega } { 2 \left( 1 + \sigma _ { _ { J _ { B + 1 } } } ^ { 2 } \right) } + \frac { \psi } { 2 \left( 1 + \sigma _ { _ { J _ { B + 1 } } } ^ { 2 } \right) ^ { 2 } } + o \left( \psi \right) .\tag{126}
$$

Proof: See Appendix B.III.

The remainders are uniform over the retained histories and the variances in $\left[ \sigma _ { \operatorname* { m i n } } ^ { 2 } , \sigma _ { \operatorname* { m a x } } ^ { 2 } \right]$ , and in (125), also over the exploration powers in $\begin{array} { r } { \left\lceil { \frac { 1 } { \ell _ { 0 } } } , 1 \right\rceil } \end{array}$

## E. Channel Selection

The expected decoding time also depends on how often each channel is explored. Define the number of exploration uses of channel $k ,$ including the initial use of each packet,

$$
T _ { k } \triangleq \sum _ { j = 1 } ^ { B } \left( \tau ^ { ( j ) } + 1 \right) 1 \left\{ J _ { j } = k \right\} , \qquad k \in [ K ] .\tag{127}
$$

We use the following bounds on the exploration uses, powers, and channel-selection errors.

Lemma $^ { 1 2 }$ (Channel-selection bounds): As $M \to \infty$ through powers of two,

$$
\mathbb { E } \left[ T _ { k } \right] \leq \frac { \psi } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } + o ( \psi ) , \qquad k \neq k ^ { \star } ,\tag{128}
$$

$$
\mathbb { E } [ T _ { k ^ { \star } } ] = o ( \psi ) ,
$$

$$
\mathbb { E } \left[ B \right] = O \left( \psi ^ { \frac { 2 } { 3 } } \right) ,\tag{129}
$$

(130)

$$
\mathbb { E } \left[ \sum _ { j = 1 } ^ { B } \lvert P _ { j , 0 } - P _ { k } ^ { \star } \rvert \left( \tau ^ { ( j ) } + 1 \right) 1 \{ J _ { j } = k \} \right] = o ( \psi ) , \qquad k \neq k ^ { \star } .\tag{131}
$$

Moreover,

$$
\mathbb { P } \left[ J _ { 0 } \neq k ^ { \star } \right] = o \left( \psi ^ { - 1 } \right) ,\tag{132}
$$

$$
\mathbb { P } \left[ J _ { B + 1 } \neq k ^ { \star } \right] = O \left( { \frac { 1 } { \log M } } \right) .\tag{133}
$$

Proof: See Appendix B.IV.

## F. Proof of Theorem 2

By (57), the decoding time consists of the $K \ell _ { 0 } = o ( \psi )$ initial uses and the packet durations. We bound $C _ { \star }$ times each expected packet duration by the number of nats that the packet carries plus its excess cost.

For the preliminary packet, $\begin{array} { r } { C _ { \star } - C \left( \frac { 1 } { \sigma _ { J _ { 0 } } ^ { 2 } } \right) } \end{array}$ vanishes on $\{ J _ { 0 } = k ^ { \star } \}$ and is bounded otherwise. Averaging (125) and using (99) and (132) gives

$$
\begin{array} { r } { C _ { \star } \mathbb { E } \left[ \tau ^ { ( 0 ) } + 1 \right] \leq \log M _ { 0 } + O \left( \log \psi \right) + O \left( \psi ^ { 2 } \right) \mathbb { P } \left[ J _ { 0 } \neq k ^ { \star } \right] = \log M _ { 0 } + o ( \psi ) . } \end{array}\tag{134}
$$

Summing (125) over the exploration packets with the tower property, and using $\mathbb { E } \left[ B \right] O ( \log \psi ) = o ( \psi )$ by (130), gives

$$
C _ { \star } \mathbb { E } \left[ \sum _ { j = 1 } ^ { B } \left( \tau ^ { ( j ) } + 1 \right) \right] \leq \mathbb { E } \left[ B \right] \ell _ { 0 } \log 2 + \mathbb { E } \left[ \sum _ { j = 1 } ^ { B } \left[ C _ { \star } - C \left( \frac { P _ { j , 0 } } { \sigma _ { \mathcal { I } _ { j } } ^ { 2 } } \right) \right] \left( \tau ^ { ( j ) } + 1 \right) \right] + o ( \psi ) .\tag{135}
$$

For the final packet, replacing $\sigma _ { J _ { B + 1 } } ^ { 2 }$ by $\sigma _ { k } ^ { 2 } ,$ <sub>⋆</sub> in (126) costs $O ( 1 )$ in expectation: the affected terms are at most ${ \cal O } ( \log M )$ by (104) and (108), and channel misselection has probability $\begin{array} { r } { O \left( \frac { 1 } { \log M } \right) } \end{array}$ by (133). Hence,

$$
C _ { \star } \mathbb { E } \left[ \tau ^ { ( B + 1 ) } + 1 \right] \leq \mathbb { E } \left[ \log M _ { B + 1 } \right] - \frac { \mathbb { E } \left[ \Omega \right] } { 2 v _ { \star } } + \frac { \psi } { 2 v _ { \star } ^ { 2 } } + o ( \psi ) .\tag{136}
$$

The payloads satisfy log $M _ { 0 } + B \ell _ { 0 } \log 2 + \log M _ { B + 1 } = \log M$ by (54). Adding (134)–(136) and substituting Ω from (55) yields

$$
C _ { \star } \mathbb { E } \left[ \tau \right] \leq \log M + \frac { \psi } { 2 v _ { \star } ^ { 2 } } + \mathbb { E } \left[ \sum _ { j = 1 } ^ { B } \left[ C _ { \star } - C \left( \frac { P _ { j , 0 } } { \sigma _ { J _ { j } } ^ { 2 } } \right) - \frac { 1 - P _ { j , 0 } } { 2 v _ { \star } } \right] \left( \tau ^ { ( j ) } + 1 \right) \right] + o ( \psi ) .\tag{137}
$$

The bracket is uniformly bounded and Lipschitz in $P _ { j , 0 } \in [ 0 , 1 ]$ . Hence, packets on the best channel contribute $o ( \psi )$ by (129), and (131) replaces the suboptimal-channel powers by $P _ { k } ^ { \star }$ at a cost $o ( \psi )$ . Using (127) gives

$$
C _ { \star } \mathbb { E } \left[ \tau \right] \leq \log M + \frac { \psi } { 2 v _ { \star } ^ { 2 } } + \sum _ { k \neq k ^ { \star } } \left[ C _ { \star } - C \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) - \frac { 1 - P _ { k } ^ { \star } } { 2 v _ { \star } } \right] \mathbb { E } \left[ T _ { k } \right] + o ( \psi ) .\tag{138}
$$

The channel ordering and the tangent bound for C give $\begin{array} { r } { C \left( \frac { p } { \sigma _ { k } ^ { 2 } } \right) \leq C \left( \frac { p } { \sigma _ { k ^ { \star } } ^ { 2 } } \right) \leq C _ { \star } - \frac { 1 - p } { 2 v _ { \star } } } \end{array}$ for $0 \leq p \leq 1$ . The multipliers of E $\left[ T _ { k } \right]$ in (138) are therefore non-negative, and (128) yields

$$
C _ { \star } \mathbb { E } \left[ \tau \right] \leq \log M + \left( \frac { 1 } { 2 v _ { \star } ^ { 2 } } + \sum _ { k \neq k ^ { \star } } \frac { C _ { \star } - C \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) - \frac { 1 - P _ { k } ^ { \star } } { 2 v _ { \star } } } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } \right) \psi + o ( \psi ) .\tag{139}
$$

Stop-at-time-zero strategy: We now include the initial randomization, so probabilities and expectations below are uncondi tional. For $\begin{array} { r } { \epsilon _ { 0 } = \epsilon - \frac { 1 } { \log M } } \end{array}$ in (37), the bounds (115) and (117) give

$$
\mathbb { P } \left[ \widehat { W } \neq W \right] \leq \epsilon _ { 0 } + \frac { 1 - \epsilon _ { 0 } } { \log M } \leq \epsilon ,\tag{140}
$$

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { \tau } X _ { t } ^ { 2 } \right] = ( 1 - \epsilon _ { 0 } ) \mathbb { E } \left[ \sum _ { t = 1 } ^ { \tau } X _ { t } ^ { 2 } \Bigg | \tau > 0 \right] \leq ( 1 - \epsilon _ { 0 } ) \mathbb { E } \left[ \tau \mid \tau > 0 \right] = \mathbb { E } \left[ \tau \right] .\tag{141}
$$

The error and power bounds (140)–(141) hold throughout the prescribed variance interval.

The duration in (139) is multiplied by $\begin{array} { r } { 1 - \epsilon _ { 0 } = 1 - \epsilon + \frac { 1 } { \log M } } \end{array}$ . With $N = \mathbb { E } \left[ \tau \right]$ , the converse in Section V-B and this bound give $\psi = \log N + O ( 1 )$ . Substituting this relation into (139) and inverting after the randomization yields, as $M \to \infty$ through powers of two at every instance,

$$
\log M \geq \frac { N C _ { \star } } { 1 - \epsilon } - \left( \frac { 1 } { 2 v _ { \star } ^ { 2 } } + \sum _ { k \neq k ^ { \star } } \frac { C _ { \star } - C \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) - \frac { 1 - P _ { k } ^ { \star } } { 2 v _ { \star } } } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } \right) \log N + o ( \log N ) .\tag{142}
$$

This proves Theorem 2.

## VIII. COMPARISON WITH THE EXPLORE-THEN-COMMIT STRATEGY

An ETC strategy separates the two tasks that the VLF-Bandit scheme performs jointly. It first identifies a channel from zero-input pilot observations and then sends the entire message over the declared channel as one packet (Section IV-C), which spends the energy saved during identification. The ETC strategy is a VLF-Bandit code.

Let $K \geq 2 .$ , and use the stop-at-time-zero randomization (37). Until it is applied below, all probabilities and expectations in this section are conditioned on the transmission branch. During identification, the transmitter sends zero, so a pilot observation on channel k satisfies $\begin{array} { r } { \frac { Y _ { t } ^ { 2 } } { \sigma _ { k } ^ { 2 } } \sim \chi _ { 1 } ^ { 2 } } \end{array}$ . The computation in Appendix A.IV, with $v _ { k }$ replaced by $\sigma _ { k } ^ { 2 }$ , shows that the reward $- Y _ { t } ^ { 2 }$ on channel k form a one-parameter exponential family with mean $- \sigma _ { k } ^ { 2 } .$ , in which the divergence between the members with means −u and −v is $d ( u \Vert v )$ . The results of Kaufmann and Koolen [46] on best-arm identification in such families therefore apply. The characteristic time of the problem is $\Upsilon _ { \mathrm { E T C } } \left( \sigma ^ { 2 } \right)$ , where

$$
\left( \Upsilon _ { \mathrm { E T C } } \left( \pmb { \sigma } ^ { 2 } \right) \right) ^ { - 1 } \triangleq \operatorname* { s u p } _ { \substack { \lambda \in \left[ 0 , 1 \right] ^ { K } : \quad \widetilde { \pmb { \sigma } } ^ { 2 } \in \left( 0 , \infty \right) ^ { K } : \quad } } \sum _ { i = 1 } ^ { K } \lambda _ { i } d \left( \sigma _ { i } ^ { 2 } \| \widetilde { \sigma } _ { i } ^ { 2 } \right) .\tag{143}
$$

We use the asymptotically optimal Track-and-Stop rule in [46, Sec. 5.1] at error level $\begin{array} { r } { \delta _ { \mathrm { p } } = \frac { 1 } { \log M } } \end{array}$ . Let $\tau _ { \mathrm { p } } \geq K$ be the number of pilot uses and $J _ { 0 }$ the declared channel, so that P ${ } ^ { \mathsf { p } } [ J _ { 0 } \neq k ^ { \star } ] \leq \delta _ { \mathrm { p } }$ . The best channel is unique, and the optimal allocation is unique and continuous [37, Th. 5 and Prop. 6]. The expected-time argument in [46, Th. 17] and its tail bound [46, App. D] give, for $\psi = \log \log M$

$$
\begin{array} { r } { \mathbb { E } \left[ \tau _ { \mathrm { p } } \right] \leq \Upsilon _ { \mathrm { E T C } } ( \pmb { \sigma } ^ { 2 } ) \psi + o ( \psi ) , } \end{array}\tag{144}
$$

$$
\mathbb { E } \left[ \tau _ { \mathrm { p } } ^ { 2 } \right] = O ( \psi ^ { 2 } ) .\tag{145}
$$

After identification, the transmitter sends the entire message as packet zero on channel $J _ { 0 }$ , with $M _ { 0 } = M$ , the threshold in (45) at $b = \psi , P _ { 0 , 0 } = 1$ , and

$$
P _ { 0 , n } \triangleq \left\{ { 1 + \frac { 1 } { \tau _ { \mathrm { p } } } } , 1 \leq n \leq \tau _ { \mathrm { p } } ^ { 2 } , \right.\tag{146}
$$

The extra allocation is at most $\tau _ { \mathrm { p } } ,$ the energy saved by the zero pilots. Since the pilot history is independent of the message, Lemmas 9 and 10 apply to the subsequent packet. Its error probability is at most $\textstyle \exp \{ - \psi \} = { \frac { 1 } { \log M } }$ , and the saved allocation covers its excess energy.

The packet-duration calculation in (124), with $q = 1$ , gives a capacity credit of $\frac { \mathbb { E } [ \tau _ { \mathrm { p } } ] } { 2 v _ { \star } } + o ( \psi )$ for the saved allocation. The second-moment bound in (145) makes the loss from unused allocation negligible, while a wrongly declared channel contribute only $O ( 1 )$ to the expected duration. Adding the pilot duration yields

$$
C _ { \star } \mathbb { E } \left[ \tau _ { \mathrm { p } } + \tau ^ { ( 0 ) } + 1 \right] \leq \log M + \frac { \psi } { 2 v _ { \star } ^ { 2 } } + \left( C _ { \star } - \frac { 1 } { 2 v _ { \star } } \right) \mathbb { E } \left[ \tau _ { \mathrm { p } } \right] + o \left( \psi \right) \leq \log M + \rho _ { \mathrm { E T C } } \psi + o ( \psi ) ,\tag{147}
$$

$$
\rho _ { \mathrm { E T C } } \triangleq \frac { 1 } { 2 v _ { \star } ^ { 2 } } + \left( C _ { \star } - \frac { 1 } { 2 v _ { \star } } \right) \Upsilon _ { \mathrm { E T C } } \left( \pmb { \sigma } ^ { 2 } \right) .\tag{148}
$$

The stop-at-time-zero strategy and the inversion in Section VII-F then show that this ETC strategy achieves the expansion (62) with the coefficient of log N replaced by $- \rho _ { \mathrm { E T C } }$

Denote the magnitude of the coefficient of log N in (62) by

$$
\rho _ { \mathrm { j o i n t } } \triangleq \frac { 1 } { 2 v _ { \star } ^ { 2 } } + \sum _ { k \neq k \star } \frac { C _ { \star } - C \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) - \frac { 1 - P _ { k } ^ { \star } } { 2 v _ { \star } } } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k \star } ^ { 2 } \right) } .\tag{149}
$$

We show that $\rho _ { \mathrm { j o i n t } } < \rho _ { \mathrm { E T C } }$ for every instance with $K \geq 2$ . Let $\sigma _ { ( 2 ) } ^ { 2 } \triangleq$ min $k \neq k ^ { \star } \ \sigma _ { k } ^ { 2 }$ , and fix weights λ in (143). Alternatives that make channels k and $k ^ { \star }$ nearly tied at variance $\sigma _ { k } ^ { 2 }$ bound the inner infimum by $\lambda _ { k } d ( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } )$ . A near tie between the best and second-best channels at variance $\sigma _ { ( 2 ) } ^ { 2 }$ bounds it by $\lambda _ { k ^ { \star } } d ( \sigma _ { k ^ { \star } } ^ { 2 } \| \sigma _ { ( 2 ) } ^ { 2 } )$ . Summing the resulting lower bounds on the weights gives

$$
\Upsilon _ { \mathrm { E T C } } \left( \pmb { \sigma } ^ { 2 } \right) \geq \frac { 1 } { d \left( \sigma _ { k ^ { \star } } ^ { 2 } \| \sigma _ { \left( 2 \right) } ^ { 2 } \right) } + \sum _ { k \neq k ^ { \star } } \frac { 1 } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } .\tag{150}
$$

Also, $P _ { k } ^ { \star }$ maximizes $\begin{array} { r } { C \left( \frac { p } { \sigma _ { k } ^ { 2 } } \right) - \frac { p } { 2 v _ { \star } } } \end{array}$ over $0 ~ \leq ~ p ~ \leq ~ 1$ , and the maximum is non-negative because $p = ~ 0$ is feasible. Subtracting (149) from (148) now gives

$$
\rho _ { \mathrm { E T C } } - \rho _ { \mathrm { j o i n t } } \geq \frac { C _ { \star } - \frac { 1 } { 2 v _ { \star } } } { d \left( \sigma _ { k ^ { \star } } ^ { 2 } \| \sigma _ { ( 2 ) } ^ { 2 } \right) } + \sum _ { k \neq k ^ { \star } } \frac { C \left( \frac { P _ { k } ^ { \star } } { \sigma _ { k } ^ { 2 } } \right) - \frac { P _ { k } ^ { \star } } { 2 v _ { \star } } } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } > 0 .\tag{151}
$$

The first term is strictly positive, and every summand is non-negative. ETC must spend zero-input observations on the best channel itself, whereas the joint scheme learns its noise variance while transmitting message bits. Communication on suboptimal channels with $P _ { k } ^ { \star } > 0$ provides the additional non-negative terms in (151).

Figure 6 illustrates the coefficient comparison for $\sigma _ { 1 } ^ { 2 } = 1$ and $\sigma _ { 2 } ^ { 2 } \in [ 1 . 5 , 2 . 5 ]$ . At $( \sigma _ { 1 } ^ { 2 } , \sigma _ { 2 } ^ { 2 } ) = ( 1 , 2 )$ , the costs in the two achievability bounds are $\rho _ { \mathrm { E T C } } \approx 3 . 3 6 2$ and $\rho _ { \mathrm { j o i n t } } \approx 0 . 7 5 4$ , a difference of about 2.608.

## IX. CONCLUSION

This paper couples VLF coding with bandit channel selection over K AWGN channels whose noise variances are unknown. The VLSF-Bandit bound has a baseline of one and a channel-exploration term at second order. The VLF-Bandit bound has no baseline and a smaller magnitude of the total coefficient, consisting of a universal-decoding term and a smaller channel exploration term. It improves on the ETC bound in every instance with $K \geq 2$

Several problems remain open. The most important is a tight second-order converse. It is unknown whether every sequence of VLF-Bandit codes whose expected number of uses of each suboptimal channel is $o ( N ^ { \alpha } )$ for every $\alpha > 0$ at every admissible problem instance must pay a channel-exploration cost of the Lai–Robbins order, since the change-of-measure arguments behind [35] constrain bandit policies at deterministic horizons and do not directly apply to codes, whose error criterion involves only the message estimate. A second direction is a joint scheme for channel families whose outputs do not reveal the unknown parameter under the capacity-achieving input, such as the BSC.

## APPENDIX A

## PROOF OF THE VLSF-BANDIT LEMMAS

## I. Proof of Lemma 4

Fix $w _ { 0 } \in [ M ]$ and $m \neq w _ { 0 }$ , and condition on $W = w _ { 0 }$ . The output density at a variance vector s and its average over the product of the priors in (26) are

$$
p _ { \pmb { s } , n } ( y ^ { n } ) \triangleq \prod _ { t = 1 } ^ { n } \phi _ { 1 + s _ { A _ { t } } } ( y _ { t } ) ,\tag{152}
$$

$$
q _ { n } ( y ^ { n } ) \triangleq \frac { 1 } { [ \log ( 4 \beta ) ] ^ { K } } \int _ { [ \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } , 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } ] ^ { K } } p _ { s , n } ( y ^ { n } ) \prod _ { k = 1 } ^ { K } \frac { d s _ { k } } { s _ { k } } ,\tag{153}
$$

![](images/bb432f10fcb6b66cf0ff1ce71955b0408d2f41c3402a6c40af2bfa36234a0418.jpg)  
Fig. 6. Second-order coefficients for $K = 2 ,$ unit average power, $\sigma _ { 1 } ^ { 2 } = 1 ,$ , and $\sigma _ { 2 } ^ { 2 } \in [ 1 . 5 , 2 . 5 ]$ . The shaded region shows $\rho _ { \mathrm { E T C } } - \rho _ { \mathrm { j o i n t } }$ , evaluated from (148) and (149).

where the selections $A _ { t }$ are determined by the past of the output path. Under the true law, the candidate codeword $\mathbf { c } ( m )$ is independent of the outputs, which have the density $p _ { \pmb { \sigma } ^ { 2 } , n }$ once the transmitted codeword is integrated out. Let $\mathbb { Q }$ be the auxiliary law under which the candidate $\mathbf { c } ( m )$ is transmitted and the noise variance of each channel is drawn independently from the prior before transmission; by Gaussian convolution, the outputs have the density $q _ { n }$ under $\mathbb { Q } .$ . Write $\mathbb { P } _ { n }$ and $\mathbb { Q } _ { n }$ for the joint laws of the candidate prefix $( \mathbf { c } _ { 1 } ( m ) , \hdots , \mathbf { c } _ { n } ( m ) )$ ) and $Y ^ { n }$ under the true law and under $\mathbb { Q } .$ The candidate prefix has the same Gaussian density under both laws. Its remaining density factor is $p _ { \sigma ^ { 2 } , n }$ under $\mathbb { P } _ { n }$ and the product of the normalized residual mixtures under $\mathbb { Q } _ { n }$ . Since $q _ { n }$ is the product of the output mixtures, (26) gives

$$
L _ { n } \triangleq { \frac { p _ { \pmb { \sigma } ^ { 2 } , n } ( Y ^ { n } ) } { q _ { n } ( Y ^ { n } ) } } ,\tag{154}
$$

$$
\frac { d \mathbb { P } _ { n } } { d \mathbb { Q } _ { n } } = L _ { n } \exp \{ - S _ { n } ( m ) \} .\tag{155}
$$

As the likelihood ratio of the two output laws, $\{ L _ { n } \} _ { n \geq 0 }$ is a martingale under $\mathbb { Q }$ in the output filtration, with $L _ { 0 } = 1$ Let $\tau ^ { \prime }$ be the first $n \geq 1$ with $S _ { n } ( m ) \geq u$ . Then

$$
\begin{array} { r l r } {  { \mathbb { P } [ \tau ^ { \prime } \leq H ] = \sum _ { n = 1 } ^ { H } \mathbb { E } _ { \mathbb { Q } } \big [ L _ { n } \exp \{ - S _ { n } ( m ) \} 1 \{ \tau ^ { \prime } = n \} \big ] \leq \exp \{ - u \} \mathbb { E } _ { \mathbb { Q } } \bigg [ \operatorname* { m a x } _ { 0 \leq n \leq H } L _ { n } \bigg ] } } \\ & { } & { \leq \frac { e } { e - 1 } ( 1 + D ( \mathbb { P } _ { H } ^ { Y } \| \mathbb { Q } _ { H } ^ { Y } ) ) \exp \{ - u \} , } \end{array}\tag{156}
$$

(157)

where $\mathbb { P } _ { H } ^ { Y }$ and $\mathbb { Q } _ { H } ^ { Y }$ are the laws of $Y ^ { H }$ . The equality in (156) sums (155) over the disjoint events $\{ \tau ^ { \prime } = n \}$ , each of which is determined by the candidate prefix and $Y ^ { n }$ , and the inequality uses $S _ { n } ( m ) \geq u$ on $\{ \tau ^ { \prime } = n \}$ . The maximum depends only on the outputs, so (66) applies in the output filtration and, with $\mathbb { E } _ { \mathbb { Q } } [ L _ { H }$ log $L _ { H } ] = D ( \mathbb { P } _ { H } ^ { Y } \| \mathbb { Q } _ { H } ^ { Y } )$ , gives (157).

It remains to bound this divergence. We restrict the prior to a box around the true variances,

$$
r _ { H } \triangleq \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } \wedge \frac { 1 + \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } } { \sqrt { H } } , \qquad \mathcal { B } \triangleq \prod _ { k = 1 } ^ { K } [ \sigma _ { k } ^ { 2 } - r _ { H } , \sigma _ { k } ^ { 2 } + r _ { H } ] ,\tag{158}
$$

and let $\pi _ { B }$ be the product of the priors restricted to $\boldsymbol { B }$ and normalized. Each factor of $\boldsymbol { B }$ lies inside the enlarged prior interval and has prior mass at least $\frac { r _ { H } } { \sigma _ { \operatorname* { m a x } } ^ { 2 } \log ( 4 \beta ) }$ . Restricting the integral in (153) to $B ,$ applying Jensen’s inequality to $\pi _ { B } .$ and conditioning each log-likelihood increment on the selected channel and the past, under which the output has the law $\mathcal { N } ( 0 , v _ { A _ { t } } )$ , we obtain

$$
D ( \mathbb { P } _ { H } ^ { Y } \| \mathbb { Q } _ { H } ^ { Y } ) = \mathbb { E } \left[ \log \frac { p _ { \pmb { \sigma } ^ { 2 } , H } ( Y ^ { H } ) } { q _ { H } ( Y ^ { H } ) } \right]\tag{159}
$$

$$
\leq K \log \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } \log ( 4 \beta ) } { r _ { H } } - \mathbb { E } \left[ \log \int \frac { p _ { s , H } ( Y ^ { H } ) } { p _ { \sigma ^ { 2 } , H } ( Y ^ { H } ) } \pi _ { \mathcal { B } } ( d s ) \right]\tag{160}
$$

$$
\leq K \log { \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } \log ( 4 \beta ) } { r _ { H } } } + \int \mathbb { E } \left[ \log { \frac { p _ { \sigma ^ { 2 } , H } ( Y ^ { H } ) } { p _ { s , H } ( Y ^ { H } ) } } \right] \pi _ { B } ( d s )\tag{161}
$$

$$
\leq K \log \frac { \sigma _ { \operatorname* { m a x } } ^ { 2 } \log ( 4 \beta ) } { r _ { H } } + \operatorname* { s u p } _ { s \in \mathcal { B } } \sum _ { k = 1 } ^ { K } \mathbb { E } \left[ N _ { k } ( H ) \right] d ( v _ { k } \| 1 + s _ { k } )\tag{162}
$$

$$
\leq K \log \bigl ( 2 \beta \log ( 4 \beta ) \sqrt { H } \bigr ) + \frac { H r _ { H } ^ { 2 } } { 4 \left( 1 + \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } \right) ^ { 2 } }\tag{163}
$$

$$
\leq \frac { K } { 2 } \log H + K \log \bigl ( 2 \beta \log ( 4 \beta ) \bigr ) + \frac { 1 } { 4 } .\tag{164}
$$

In (163), the first term comes from $\begin{array} { r } { r _ { H } \ge \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 \sqrt { H } } } \end{array}$ , and the second from the fact that the counts add up to H and from the bound $\begin{array} { r } { d ( v _ { k } \| 1 + s _ { k } ) \le \frac { ( v _ { k } - 1 - s _ { k } ) ^ { 2 } } { 4 ( v _ { k } \wedge ( 1 + s _ { k } ) ) ^ { 2 } } } \end{array}$ . For $s \in B$ , these two variances are both at least $1 + \textstyle { \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } }$ and differ by at most $r _ { H }$

Since $\begin{array} { r } { \frac { 5 } { 4 } + K \log ( 2 \dot { \beta } \log ( 4 \beta ) ) \geq 0 } \end{array}$ , substituting (164) into (157) and using the definition of $c _ { 0 }$ give

$$
\begin{array} { r } { \mathbb { P } \left[ \tau ^ { \prime } \le H \right] \le c _ { 0 } ( 1 + \log H ) \exp \{ - u \} . } \end{array}\tag{165}
$$

As $\{ \tau ^ { \prime } \leq H \} = \{ \mathrm { m a x } _ { 1 \leq n \leq H } S _ { n } ( m ) \geq u \}$ , a union bound over m $\neq w _ { 0 }$ and an average over $W$ prove (74).

## II. Proof of (79)

We use the reward-stack representation in [34, Sec. 4.6]. For each channel k, let $\breve { Y } _ { k , 1 } , \breve { Y } _ { k , 2 } , \ldots$ . be i.i.d. $\mathcal { N } ( 0 , v _ { k } )$ , independently over $k ,$ and let $\breve { Y } _ { k , j }$ be the output of the j-th use of channel k. Since the selection rule uses only past outputs, and the next output has the law $\mathcal { N } ( 0 , v _ { k } )$ conditionally on the past and on the selection of channel k, the selections and outputs have the same law as in the continued process. Define

$$
\chi _ { k , j } ^ { 2 } \triangleq \sum _ { i = 1 } ^ { j } \frac { \check { Y } _ { k , i } ^ { 2 } } { v _ { k } } .\tag{166}
$$

For the first j uses of channel $k ,$ let $\Lambda _ { k , j } ^ { \mathrm { n u m } }$ be the logarithm of the likelihood of the residuals at the true noise variance divided by the residual mixture in (26), and let $\mathrm { \bar { A } } _ { k , j } ^ { \mathrm { d e n } }$ be the same quantity for the outputs and the output mixture, both mixtures with their normalizing factors. These redundancies vanish at $j = 0$ . Grouping the factors of $I _ { n }$ in (75) and of $S _ { n } ( W )$ by channel gives, for every $n ,$

$$
I _ { n } - S _ { n } ( W ) = \sum _ { k = 1 } ^ { K } \left( \Lambda _ { k , N _ { k } ( n ) } ^ { \mathrm { n u m } } - \Lambda _ { k , N _ { k } ( n ) } ^ { \mathrm { d e n } } \right) .\tag{167}
$$

We show that the first redundancy is at most $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ log j plus a constant and that the second is at least ${ \frac { 1 } { 2 } } \log j$ minus a constant and a chi-square fluctuation, so that the leading logarithms cancel in (167).

For the residual mixture, we write the noise variance through its precision $\begin{array} { r } { \eta = \frac { 1 } { s } . } \end{array}$ , so that $\begin{array} { r } { \frac { d s } { s } = \frac { d \eta } { \eta } } \end{array}$ , the enlarged interval becomes $\begin{array} { r } { [ \frac { \eta _ { \mathrm { m i n } } } { 2 } , 2 \eta _ { \mathrm { m a x } } ] } \end{array}$ , and the true precision is $\begin{array} { r } { \eta _ { k } = \frac { 1 } { \sigma _ { k } ^ { 2 } } } \end{array}$ . Fix n with $j = N _ { k } ( n ) \ge 1$ , so that the first $j$ residuals of channel $k$ have the squared sum $\zeta _ { k , W } ( n )$ . We keep only the precisions in the segment $\begin{array} { r } { | \eta - \eta _ { k } | \le \frac { \eta _ { \mathrm { m i n } } } { 2 \sqrt { j } } } \end{array}$ , which lies in the enlarged interval because $\eta _ { \mathrm { m i n } } \le \eta _ { k } \le \eta _ { \mathrm { m a x } }$ . On this segment, $\begin{array} { r } { \frac { 1 } { \eta } \geq \frac { \sigma _ { \mathrm { m i n } } ^ { 2 } } { 2 } } \end{array}$ and the second derivative of ${ \frac { j } { 2 } } \log \eta$ is at least $- 2 j \sigma _ { \mathrm { m a x } } ^ { 4 }$ Taylor’s theorem gives

$$
\exp \{ - \Lambda _ { k , j } ^ { \mathrm { n u m } } \} = \frac { 1 } { \log ( 4 \beta ) } \int _ { \frac { \eta _ { \mathrm { m i n } } } { 2 } } ^ { 2 \eta _ { \mathrm { m a x } } } \left( \frac { \eta } { \eta _ { k } } \right) ^ { \frac { \eta } { 2 } } \exp \left\{ - \frac { ( \eta - \eta _ { k } ) \zeta _ { k , W } ( n ) } { 2 } \right\} \frac { d \eta } { \eta }\tag{168}
$$

$$
\geq \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 \log ( 4 \beta ) } \int _ { \eta _ { k } - \frac { \eta _ { \operatorname* { m i n } } } { 2 \sqrt { j } } } ^ { \eta _ { k } + \frac { \eta _ { \operatorname* { m i n } } } { 2 \sqrt { j } } } \exp \left\{ \frac { ( \eta - \eta _ { k } ) \big ( j \sigma _ { k } ^ { 2 } - \zeta _ { k , W } ( n ) \big ) } { 2 } - j \sigma _ { \operatorname* { m a x } } ^ { 4 } ( \eta - \eta _ { k } ) ^ { 2 } \right\} d \eta\tag{169}
$$

$$
\ge \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } e ^ { - \frac { 1 } { 4 } } } { \log ( 4 \beta ) } \int _ { 0 } ^ { \frac { \eta _ { \operatorname* { m i n } } } { 2 \sqrt { j } } } \cosh \left( \frac { x \left( j \sigma _ { k } ^ { 2 } - \zeta _ { k , W } ( n ) \right) } { 2 } \right) d x\tag{170}
$$

$$
\geq { \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } \eta _ { \operatorname* { m i n } } e ^ { - { \frac { 1 } { 4 } } } } { 2 \log ( 4 \beta ) \sqrt { j } } } = { \frac { e ^ { - { \frac { 1 } { 4 } } } } { 2 \beta \log ( 4 \beta ) \sqrt { j } } } .\tag{171}
$$

Pairing the points $\eta _ { k } \pm x$ gives the hyperbolic cosine in (170), which is at least one. Since $j \sigma _ { \mathrm { m a x } } ^ { 4 } ( \eta - \eta _ { k } ) ^ { 2 } \leq \frac { 1 } { 4 }$ on the segment, the last two rows follow. Taking logarithms gives, for every residual sequence,

$$
\Lambda _ { k , j } ^ { \mathrm { n u m } } \leq \frac 1 2 \log j + \frac 1 4 + \log \bigl ( 2 \beta \log ( 4 \beta ) \bigr ) , \qquad j \geq 1 .\tag{172}
$$

The output mixture is bounded from above by a similar argument, in the natural parameter $\begin{array} { r } { \theta = \frac { 1 } { 2 ( 1 + s ) } } \end{array}$ of the output family. Then $\begin{array} { r } { \frac { d s } { s } = \frac { 2 ( 1 + s ) ^ { 2 } } { s } d \theta } \end{array}$ , the enlarged interval becomes $\begin{array} { r } { \big [ \frac { 1 } { 2 + 4 \sigma _ { \mathrm { m a x } } ^ { 2 } } , \frac { 1 } { 2 + \sigma _ { \mathrm { m i n } } ^ { 2 } } \big ] } \end{array}$ , and the true output variance corresponds to $\begin{array} { r } { \theta _ { k } = \frac { 1 } { 2 v _ { k } } } \end{array}$ Since $\frac { ( 1 + s ) ^ { 2 } } { s }$ is convex, the prior density in θ is largest at an endpoint of the enlarged interval, and both endpoints give

$$
\frac { 2 ( 1 + s ) ^ { 2 } } { s \log ( 4 \beta ) } \leq \Bigl ( 1 + \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } \Bigr ) \frac { 8 \beta + 4 \eta _ { \operatorname* { m a x } } } { \log ( 4 \beta ) } , \qquad s \in \Bigl [ \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } , 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } \Bigr ] .\tag{173}
$$

In the reward-stack representation, the first $j$ outputs of channel k have the squared sum $v _ { k } \chi _ { k , j } ^ { 2 }$ , and, for $j \geq 1$

$$
\exp \{ - \Lambda _ { k , j } ^ { \mathrm { d e n } } \} = \frac { 1 } { \log ( 4 \beta ) } \int _ { \frac { 1 } { 2 + 4 \sigma _ { \operatorname* { m i n } } ^ { 2 } } } ^ { \frac { 1 } { 2 + \sigma _ { \operatorname* { m i n } } ^ { 2 } } } \left( \frac { \theta } { \theta _ { k } } \right) ^ { \frac { j } { 2 } } \exp \left\{ - ( \theta - \theta _ { k } ) v _ { k } \chi _ { k , j } ^ { 2 } \right\} \frac { 2 ( 1 + s ) ^ { 2 } } { s } d \theta\tag{174}
$$

$$
\leq \left( 1 + \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } \right) \frac { 8 \beta + 4 \eta _ { \operatorname* { m a x } } } { \log ( 4 \beta ) } \int _ { - \infty } ^ { \infty } \exp \left\{ ( \theta - \theta _ { k } ) v _ { k } \big ( j - \chi _ { k , j } ^ { 2 } \big ) - j \Big ( 1 + \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } \Big ) ^ { 2 } ( \theta - \theta _ { k } ) ^ { 2 } \right\} d \theta\tag{175}
$$

$$
= \frac { ( 8 \beta + 4 \eta _ { \mathrm { m a x } } ) \sqrt { \pi } } { \log ( 4 \beta ) \sqrt { j } } \exp \left\{ \frac { v _ { k } ^ { 2 } \big ( \chi _ { k , j } ^ { 2 } - j \big ) ^ { 2 } } { 4 j \left( 1 + \frac { \sigma _ { \mathrm { m i n } } ^ { 2 } } { 2 } \right) ^ { 2 } } \right\}\tag{176}
$$

$$
\leq \frac { 4 \sqrt { \pi } ( 2 \beta + \eta _ { \operatorname* { m a x } } ) } { \log ( 4 \beta ) \sqrt { j } } \exp \left\{ \frac { \beta ^ { 2 } \left( \chi _ { k , j } ^ { 2 } - j \right) ^ { 2 } } { j } \right\} ,\tag{177}
$$

$$
\Lambda _ { k , j } ^ { \mathrm { d e n } } \geq \frac { 1 } { 2 } \log j - \log \frac { 4 \sqrt { \pi } ( 2 \beta + \eta _ { \operatorname* { m a x } } ) } { \log ( 4 \beta ) } - \frac { \beta ^ { 2 } \left( \chi _ { k , j } ^ { 2 } - j \right) ^ { 2 } } { j } .\tag{178}
$$

Inequality (175) uses (173) and Taylor’s theorem with second derivative at most $\begin{array} { r } { - 2 j ( 1 + \frac { \sigma _ { \mathrm { m i n } } ^ { 2 } } { 2 } ) ^ { 2 } } \end{array}$ , then extends the integral to the real line. The remaining rows use $v _ { k } \le ( 2 + \sigma _ { \operatorname* { m i n } } ^ { 2 } ) \beta$ and take logarithms.

Subtracting (178) from (172) cancels the leading logarithms and gives

$$
\begin{array} { r l } & { \Lambda _ { k , j } ^ { \mathrm { n u m } } - \Lambda _ { k , j } ^ { \mathrm { d e n } } \leq \displaystyle \frac { 1 } { 4 } + \log \left( 8 \sqrt { \pi } \beta ( 2 \beta + \eta _ { \mathrm { m a x } } ) \right) + \frac { \beta ^ { 2 } \left( \chi _ { k , j } ^ { 2 } - j \right) ^ { 2 } } { j } } \\ & { \qquad \leq 4 ( c _ { 1 } + \beta ^ { 2 } ) + \displaystyle \frac { \beta ^ { 2 } \left( \chi _ { k , j } ^ { 2 } - j \right) ^ { 2 } } { j } , } \end{array}\tag{179}
$$

(180)

where (180) follows from the definition of $c _ { 1 }$ in Lemma $5 , \beta \geq 1$ , and $\textstyle { \frac { 1 } { 4 } } + { \frac { 1 } { 2 } }$ log $\pi < 1$

Since the redundancies of an unused channel vanish, (167) and (180) give, for every integer $H \geq 1$ and on every path,

$$
\operatorname* { m a x } _ { 0 \leq n \leq H } ( I _ { n } - S _ { n } ( W ) ) ^ { + } \leq 4 K ( c _ { 1 } + \beta ^ { 2 } ) + \beta ^ { 2 } \sum _ { k = 1 } ^ { K } \operatorname* { m a x } _ { 1 \leq j \leq H } \frac { \left( \chi _ { k , j } ^ { 2 } - j \right) ^ { 2 } } { j } .\tag{181}
$$

The right-hand side depends on the outputs alone, whose law is the same in the reward-stack representation. There, $\chi _ { k , j } ^ { 2 } - j$ is a sum of $j$ i.i.d. centered variables, so $| \chi _ { k , j } ^ { 2 } - j |$ is a non-negative submartingale in $j ,$ whose fourth moments are those of a centered chi-square variable,

$$
\begin{array} { r } { \mathbb { E } \left[ \left( \chi _ { k , n } ^ { 2 } - n \right) ^ { 4 } \right] = 1 2 n ^ { 2 } + 4 8 n \leq 6 0 n ^ { 2 } , \qquad n \geq 1 . } \end{array}\tag{182}
$$

We apply Minkowski’s inequality to (181), split the indices into the dyadic blocks $2 ^ { \ell } \leq j < 2 ^ { \ell + 1 }$ , and apply (65) and (182) on each block. This gives

$$
\left( \mathbb { E } \left[ \left[ \operatorname* { m a x } _ { 0 \leq n \leq H } ( I _ { n } - S _ { n } ( W ) ) ^ { + } \right] ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } \leq 4 K ( c _ { 1 } + \beta ^ { 2 } ) + \beta ^ { 2 } \sum _ { k = 1 } ^ { K } \left( \mathbb { E } \left[ \operatorname* { m a x } _ { 1 \leq j \leq H } \frac { \left( \chi _ { k , j } ^ { 2 } - j \right) ^ { 4 } } { j ^ { 2 } } \right] \right) ^ { \frac { 1 } { 2 } }\tag{183}
$$

$$
\leq 4 K ( c _ { 1 } + \beta ^ { 2 } ) + \beta ^ { 2 } \sum _ { k = 1 } ^ { K } \left( \sum _ { \ell = 0 } ^ { \lfloor \log _ { 2 } H \rfloor } 2 ^ { - 2 \ell } \mathbb { E } \left[ \operatorname* { m a x } _ { 1 \leq j \leq 2 ^ { \ell + 1 } } \left( \chi _ { k , j } ^ { 2 } - j \right) ^ { 4 } \right] \right) ^ { \frac { 1 } { 2 } }\tag{184}
$$

$$
\leq 4 K ( c _ { 1 } + \beta ^ { 2 } ) + 2 8 K \beta ^ { 2 } \sqrt { 1 + \log _ { 2 } { H } }\tag{185}
$$

$$
\leq c _ { 2 } { \sqrt { 1 + \log _ { 2 } H } } ,\tag{186}
$$

since $2 4 0 ( \textstyle { \frac { 4 } { 3 } } ) ^ { 4 } < 2 8 ^ { 2 } , \sqrt { 1 + \log _ { 2 } { H } } \ge 1$ , and $c _ { 2 } = 4 K ( c _ { 1 } + 8 \beta ^ { 2 } )$ . Finally, $I _ { \tau _ { W } \wedge H } - S _ { \tau _ { W } \wedge H } ( W )$ is at most the maximum on the left-hand side of (183), so the Cauchy–Schwarz inequality gives

$$
\begin{array} { r } { \mathbb { E } \left[ I _ { \tau _ { W } \wedge H } - S _ { \tau _ { W } \wedge H } ( W ) \right] \leq c _ { 2 } \sqrt { 1 + \log _ { 2 } H } , } \end{array}\tag{187}
$$

which proves (79).

## III. Proof of (80)

We verify the mean-excess condition of Lemma 2, with $c = c _ { 1 }$ , for the score increments

$$
\xi _ { t } \triangleq S _ { t } ( W ) - S _ { t - 1 } ( W ) .\tag{188}
$$

Condition on $\mathcal { F } _ { t - 1 }$ and $A _ { t } = k$ . Then $\xi _ { t }$ is the logarithm of the ratio of two predictive densities, that of the residual $Y _ { t } - X _ { t }$ under the residual mixture and that of the output $Y _ { t }$ under the output mixture. These are mixtures of $\mathcal { N } ( 0 , s )$ and $\mathcal { N } ( 0 , 1 + s )$ densities, respectively, over $s \in [ \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } , 2 \sigma _ { \operatorname* { m a x } } ^ { 2 } ]$ , with mixing weights fixed by $\mathcal { F } _ { t - 1 }$ . Write the independent standard Gaussian variables $X _ { t }$ and $\frac { Y _ { t } \dot { - } X _ { t } } { \sigma _ { k } }$ in polar coordinates as ${ \sqrt { 2 \rho } } \cos \varphi$ and ${ \sqrt { 2 \rho } } \sin \varphi ,$ . In these coordinates, $\rho$ is exponential with mean one and independent of $\varphi _ { : }$ , and the residual and the output are $\sigma _ { k } \sqrt { 2 \rho }$ sin $\varphi$ and ${ \sqrt { 2 \rho } } ( \cos \varphi + \sigma _ { k } \sin \varphi )$

For a fixed angle, regard the increment as a function $\xi _ { t } ( \rho )$ of $\rho .$ Each Gaussian log-density in the two mixtures is affine in $\rho ,$ with slope $- { \frac { \sigma _ { k } ^ { 2 } \sin ^ { 2 } \varphi } { s } }$ for the residual and $- \frac { ( \cos \varphi + \sigma _ { k } \sin \varphi ) ^ { 2 } } { 1 + s }$ for the output, so $\xi _ { t } ^ { \prime } ( \rho )$ is a weighted average of the first slopes minus a weighted average of the second. On the prior interval, $\sigma _ { k } ^ { 2 } \le 2 \beta s$ and, by the Cauchy–Schwarz inequality, (cos $\varphi + \sigma _ { k }$ sin $\varphi ) ^ { 2 } \leq v _ { k } \leq 2 \beta ( 1 + s )$ , so all these slopes lie in $[ - 2 \beta , 0 ] . \mathrm { A t } \ \rho = 0$ , the two predictive densities are averages of $( 2 \pi s ) ^ { - { \frac { 1 } { 2 } } }$ and of $( 2 \pi ( 1 + s ) ) ^ { - { \frac { 1 } { 2 } } }$ over the prior interval, and the extreme values of these two functions bound their ratio. Therefore,

$$
| \xi _ { t } ^ { \prime } ( \rho ) | \leq 2 \beta , \qquad \rho \geq 0 ,\tag{189}
$$

$$
| \xi _ { t } ( 0 ) | \leq \frac { 1 } { 2 } \log ( 4 \beta + 2 \eta _ { \operatorname* { m a x } } ) = c _ { 1 } - 2 \beta .\tag{190}
$$

Fix a non-negative $\mathcal { F } _ { t - 1 }$ <sub>1</sub>-measurable threshold u. The set of values $r \geq 0$ with $\xi _ { t } ( r ) > u$ is open in $[ 0 , \infty )$ and is a disjoint union of intervals with endpoints $r _ { - } , r _ { + }$ . On such an interval, $\xi _ { t } ( r ) - u \le 2 \beta ( r - r _ { - } ) + c _ { 1 } - 2 \beta$ , by (189) if $r _ { - } > 0$ , since then $\xi _ { t } ( \boldsymbol { r } _ { - } ) = u ,$ and by (189) and (190) if $r _ { - } = 0$ . Integration by parts applied to this bound gives

$$
\int _ { r _ { - } } ^ { r _ { + } } \left( \xi _ { t } ( r ) - u \right) \exp \{ - r \} d r \leq c _ { 1 } \int _ { r _ { - } } ^ { r _ { + } } \exp \{ - r \} d r ,\tag{191}
$$

since the upper boundary term is non-positive and vanishes when $r _ { + } = \infty$ . Summing over the intervals and averaging over the angle, we obtain

$$
\begin{array} { r } { \mathbb { E } \left[ ( \xi _ { t } - u ) ^ { + } \mid \mathcal { F } _ { t - 1 } \right] \leq c _ { 1 } \mathbb { P } \left[ \xi _ { t } > u \mid \mathcal { F } _ { t - 1 } \right] . } \end{array}\tag{192}
$$

The increments are also integrable, since $| \xi _ { t } ( \rho ) | \leq | \xi _ { t } ( 0 ) | + 2 \beta \rho$ and E $[ \rho ] = 1$ give $\mathbb { E } [ | \xi _ { t } |  \big | \mathcal { F } _ { t - 1 } ] \leq c _ { 1 }$ . Since $S _ { 0 } ( W ) = 0 < \gamma$ Lemma 2 with $\omega = \gamma$ now gives (80).

## IV. Proof of Lemma 6

Conditioned on selecting channel k, the output Y has distribution $\mathcal { N } ( 0 , v _ { k } )$ with $v _ { k } = 1 + \sigma _ { k } ^ { 2 }$ under the random-coding ensemble. A change of variables gives the density of the reward $- Y ^ { 2 }$ , and, with the natural parameter $\begin{array} { r } { \theta = \frac { 1 } { 2 v _ { k } } > 0 } \end{array}$ , its density with respect to a dominating measure $\mu _ { \mathrm { s q } }$ takes the canonical form,

$$
\frac { d P _ { - Y ^ { 2 } } } { d s } ( s ) = \frac { 1 } { \sqrt { 2 \pi v _ { k } ( - s ) } } \exp \biggl \{ \frac { s } { 2 v _ { k } } \biggr \} 1 \{ s < 0 \} ,\tag{193}
$$

$$
\mu _ { \mathrm { s q } } ( d r ) \triangleq { \frac { 1 \left\{ r < 0 \right\} } { \sqrt { 2 \pi ( - r ) } } } d r ,
$$

$$
\frac { d \nu _ { \theta } } { d \mu _ { \mathrm { s q } } } ( s ) = \exp \{ \theta s - \kappa ( \theta ) \} ,\tag{194}
$$

(195)

$$
\kappa ( \theta ) \triangleq - \frac { 1 } { 2 } \log ( 2 \theta ) .\tag{196}
$$

Thus, $\{ \nu _ { \theta } \} _ { \theta > 0 }$ is a one-parameter canonical exponential family. Its mean, its variance, and the divergence between two of its members follow from the cumulant (196),

$$
\kappa ^ { \prime } ( \theta ) = - \frac { 1 } { 2 \theta } = - v _ { k } ,\tag{197}
$$

$$
\kappa ^ { \prime \prime } ( \theta ) = \frac { 1 } { 2 \theta ^ { 2 } } = 2 v _ { k } ^ { 2 } \leq 2 ( 1 + \sigma _ { \mathrm { m a x } } ^ { 2 } ) ^ { 2 } = V _ { \mathrm { s q } } ,\tag{198}
$$

$$
D ( \nu _ { \frac { 1 } { 2 u } }  \nu _ { \frac { 1 } { 2 v } } ) = \kappa ( \frac { 1 } { 2 v } ) - \kappa ( \frac { 1 } { 2 u } ) - \kappa ^ { \prime } ( \frac { 1 } { 2 u } ) ( \frac { 1 } { 2 v } - \frac { 1 } { 2 u } ) = \frac { 1 } { 2 } ( \frac { u } { v } - 1 - \log \frac { u } { v } ) = d ( u \Vert v ) .\tag{199}
$$

This proves the lemma.

V. Proof of Lemma 7

Define the centered information density

$$
\widetilde { I } _ { n } \triangleq I _ { n } - \sum _ { t = 1 } ^ { n } C ^ { ( A _ { t } ) } , \qquad n \geq 0 .\tag{200}
$$

By (75), on $\left\{ A _ { t } = k \right\}$ and conditioned on $\mathcal { F } _ { t - 1 }$ , the increment $\widetilde { I } _ { t } - \widetilde { I } _ { t - 1 }$ is a quadratic form in the independent standard Gaussian variables $X _ { t }$ and $\frac { Y _ { t } - X _ { t } } { \sigma _ { k } }$ , with eigenvalues $\pm \frac { 1 } { 2 \sqrt { v _ { k } } }$ . An orthogonal rotation gives

$$
\widetilde { I } _ { t } - \widetilde { I } _ { t - 1 } \overset { \mathrm { d } } { = } \frac { G _ { 1 } G _ { 2 } } { \sqrt { v _ { A _ { t } } } } ,\tag{201}
$$

where $G _ { 1 }$ and $G _ { 2 }$ are independent standard Gaussian variables under this conditional law. The conditional second and fourth moments of the increment are thus at most one and nine, respectively, and its conditional odd moments vanish. Orthogonality gives $\mathbb { E } \left\lceil \widetilde { I } _ { t - 1 } ^ { 2 } \right\rceil \leq t - 1$ . Expanding the fourth power conditionally and summing the resulting bounds gives

$$
\mathbb { E } \left[ \widetilde { I } _ { n } ^ { 4 } \right] \leq \sum _ { t = 1 } ^ { n } { \bigl ( } 6 ( t - 1 ) + 9 { \bigr ) } = 3 n ^ { 2 } + 6 n \leq 9 n ^ { 2 } , \qquad n \geq 1 .\tag{202}
$$

Let $\begin{array} { r } { n \ge \frac { 4 \gamma } { C ( \eta _ { \mathrm { m i n } } ) } } \end{array}$ be an integer. As every selected-channel capacity is at least $C ( \eta _ { \mathrm { m i n } } )$ , the score satisfies $S _ { n } ( W ) \geq$ $\widetilde { I } _ { n } + n C ( \eta _ { \mathrm { m i n } } ) ^ { + } - ( I _ { n } - S _ { n } ( W ) ) ^ { + }$ , and therefore

$$
\left\{ \tau _ { W } > n \right\} \subseteq \left\{ \widetilde { I } _ { n } < - \frac { n C ( \eta _ { \operatorname* { m i n } } ) } { 2 } \right\} \cup \left\{ ( I _ { n } - S _ { n } ( W ) ) ^ { + } > \frac { n C ( \eta _ { \operatorname* { m i n } } ) } { 4 } \right\} .\tag{203}
$$

The union bound and Markov’s inequality for the fourth and the second powers, followed by (202) and (186), give

$$
\mathbb { P } \left[ \tau _ { W } > n \right] \le \frac { 1 6 \mathbb { E } \left[ \widetilde { I } _ { n } ^ { 4 } \right] } { C ( \eta _ { \operatorname* { m i n } } ) ^ { 4 } n ^ { 4 } } + \frac { 1 6 \mathbb { E } \left[ [ ( I _ { n } - S _ { n } ( W ) ) ^ { + } ] ^ { 2 } \right] } { C ( \eta _ { \operatorname* { m i n } } ) ^ { 2 } n ^ { 2 } }\tag{204}
$$

$$
\leq \frac { 1 4 4 } { C ( \eta _ { \mathrm { m i n } } ) ^ { 4 } n ^ { 2 } } + \frac { 1 6 c _ { 2 } ^ { 2 } ( 1 + \log _ { 2 } n ) } { C ( \eta _ { \mathrm { m i n } } ) ^ { 2 } n ^ { 2 } } .\tag{205}
$$

This bound is summable over n, so $\mathbb { E } \left[ \tau _ { W } \right] < \infty$ . For every integer $\begin{array} { r } { H \ge \frac { 4 \gamma } { C ( \eta _ { \mathrm { m i n } } ) } } \end{array}$ , it also gives

$$
\mathbb { E } \left[ ( \tau _ { W } - H ) ^ { + } \right] = \sum _ { i = H } ^ { \infty } \mathbb { P } \left[ \tau _ { W } > i \right] \le \frac { 2 8 8 } { C ( \eta _ { \operatorname* { m i n } } ) ^ { 4 } H } + \frac { 6 4 c _ { 2 } ^ { 2 } ( 1 + \log _ { 2 } H ) } { C ( \eta _ { \operatorname* { m i n } } ) ^ { 2 } H } ,\tag{206}
$$

since $\textstyle \sum _ { i = H } ^ { \infty } { \frac { 1 } { i ^ { 2 } } } \leq { \frac { 2 } { H } }$ and, by comparison with an integral, $\begin{array} { r } { \sum _ { i = H } ^ { \infty } { \frac { 1 + \log _ { 2 } i } { i ^ { 2 } } } \leq { \frac { 4 ( 1 + \log _ { 2 } H ) } { H } } } \end{array}$

Under the condition on γ in Lemma 7, the design horizon satisfies $\begin{array} { r } { T \geq \gamma \log ^ { \frac { \mu } { 2 } } \gamma \geq e \vee \frac { 4 \gamma } { C ( \eta _ { \mathrm { m i n } } ) } } \end{array}$ , and the right-hand sides of (205) and (206) decrease in n and H beyond e. Taking $n = H = T$ , replacing $T$ by $\gamma \log ^ { 2 } \gamma$ , and using $1 + \log _ { 2 } ( \gamma \log ^ { 2 } \gamma ) \leq$ 6 log γ, we arrive at

$$
\mathbb { P } \left[ \tau _ { W } > T \right] \le \frac { 1 4 4 } { C ( \eta _ { \mathrm { m i n } } ) ^ { 4 } \gamma ^ { 2 } \log ^ { 4 } \gamma } + \frac { 9 6 c _ { 2 } ^ { 2 } } { C ( \eta _ { \mathrm { m i n } } ) ^ { 2 } \gamma ^ { 2 } \log ^ { 3 } \gamma } \le \frac { c _ { 3 } } { \gamma ^ { 2 } \log ^ { 3 } \gamma } ,\tag{207}
$$

$$
\mathbb { E } \left[ ( \tau _ { W } - T ) ^ { + } \right] \leq \frac { 2 8 8 } { C ( \eta _ { \mathrm { m i n } } ) ^ { 4 } \gamma \log ^ { 2 } \gamma } + \frac { 3 8 4 c _ { 2 } ^ { 2 } } { C ( \eta _ { \mathrm { m i n } } ) ^ { 2 } \gamma \log \gamma } \leq \frac { c _ { 3 } } { \gamma \log \gamma } ,\tag{208}
$$

where both final inequalities use log $\gamma \geq 1$ and the definition of $c _ { 3 }$ . Together with the finiteness of $\mathbb { E } \left[ \tau _ { W } \right]$ , this is Lemma 7.

## VI. Proof of Lemma 8

We apply Lemma 3 at the horizon $T ,$ in the reward-stack representation in Appendix A.II. By Lemma $^ { 6 , }$ the rewards $- \breve { Y } _ { k , s } ^ { 2 }$ are negative almost surely and have means $\mu _ { k } = - v _ { k }$ and variance bounded by $V _ { \mathrm { s q } } \ \mathrm { o n } \ \bigl [ - ( 1 + \sigma _ { \mathrm { m a x } } ^ { 2 } ) , - ( 1 + \sigma _ { \mathrm { m i n } } ^ { 2 } ) \bigr ]$ , an interval inside the mean domain $( - \infty , 0 )$ . The projected means and restricted indices are those in Lemma 3. Let $\delta = ( \log T ) ^ { - \frac { 1 } { 4 } }$ . For all sufficiently large $\gamma ,$ this $\delta$ satisfies (71) with $V = V _ { \mathrm { s q } }$ . Hence, Lemma 3 and (81) give

$$
\mathbb { E } \left[ N _ { k } ( T ) \right] \leq \frac { G _ { K } ( T ) } { d ( v _ { k } - \delta \| v _ { \star } + \delta ) } + 2 ( 1 6 e ^ { 2 } + 2 ) V _ { \mathrm { s q } } K \sqrt { \log T } + 1\tag{209}
$$

$$
\leq \frac { \log T + \log ( 1 + \log ^ { 2 } T ) } { D _ { k , k ^ { \star } } - O ( \delta ) } + O ( \sqrt { \log T } )\tag{210}
$$

$$
= \frac { \log \gamma } { D _ { k , k ^ { \star } } } + O \bigl ( \delta \log \gamma + \log \log \gamma + \sqrt { \log \gamma } \bigr )\tag{211}
$$

$$
\leq \frac { \log \gamma } { D _ { k , k ^ { \star } } } + O \bigl ( ( \log \gamma ) ^ { \frac { 3 } { 4 } } \bigr ) ,\tag{212}
$$

where (210) uses $G _ { K } ( T ) \leq$ log $T + \log ( 1 + \log ^ { 2 } T )$ , which follows from (25), and $d ( v _ { k } - \delta \| v _ { \star } + \delta ) \geq D _ { k , k ^ { \star } } - O ( \delta )$ , a consequence of the continuous differentiability of d and of $d ( v _ { k } \| v _ { \star } ) = D _ { k , k ^ { \star } } > 0 ;$ (211) follows from (33), which gives log $T = \log \gamma + O ( \log \log \gamma )$ . The last row is (85).

## APPENDIX B

## PROOF OF THE VLF-BANDIT LEMMAS

All probabilities and expectations retain the transmission-branch convention in Section VII. Throughout this appendix, $\chi _ { n } ^ { 2 }$ denotes a variable with the chi-square distribution with n degrees of freedom.

## I. Proof of Lemma 10

Fix a transmitted packet $j ,$ condition on $\mathcal { G } _ { j , 0 } ^ { }$ , and write $s = \sigma _ { J , } ^ { 2 }$ ; all expectations and probabilities below use this conditional law. By Lemma $9 , W _ { j }$ is uniform on $[ M _ { j } ]$ , so the initial symbol (38) has energy

$$
\mathbb { E } \left[ X _ { j , 0 } ^ { 2 } \right] = \frac { 1 2 P _ { j , 0 } } { M _ { j } ^ { 2 } } \cdot \frac { M _ { j } ^ { 2 } - 1 } { 1 2 } \le P _ { j , 0 } .\tag{213}
$$

For $n \geq 1$ , the estimate $\widehat { \eta } _ { j , n - 1 }$ and the indicator $1 \left\{ \tau ^ { ( j ) } \geq n \right\}$ are $\mathcal { G } _ { j , n - 1 }$ -measurable, and $E _ { j , n - 1 }$ has conditional second moment s given $\mathcal { G } _ { j , n - 1 }$ by Lemma 9. Hence, by (41),

$$
\mathbb { E } \left[ X _ { j , n } ^ { 2 } \mathbf { 1 } \left\{ { \tau } ^ { ( j ) } \geq n \right\} \right] = s P _ { j , n } \mathbb { E } \left[ \widehat { \eta } _ { j , n - 1 } \mathbf { 1 } \left\{ { \tau } ^ { ( j ) } \geq n \right\} \right] , \qquad n \geq 1 ,\tag{214}
$$

$$
\leq \frac { 1 } { 2 } P _ { j , n } \mathbb { P } \left[ \tau ^ { ( j ) } \geq n \right] , \quad \quad 1 \leq n \leq 9 ,\tag{215}
$$

where the inequality holds because the first nine refinements use the estimate $\frac { \eta _ { \mathrm { m i n } } } { 2 }$ , which is at most $\scriptstyle { \frac { 1 } { 2 s } }$ since $s \leq \sigma _ { \operatorname* { m a x } } ^ { 2 }$ . These refinements spend at most half of their nominal energy, and the other half is a reserve.

For $n \geq 1 0$ , refinement n uses the estimate $\widehat { \eta } _ { j , m }$ with $m = n - 1 \geq 9$ which by (40) is the projection onto $\begin{array} { r } { \left[ \frac { \eta _ { \mathrm { m i n } } } { 2 } , 2 \eta _ { \mathrm { m a x } } \right] } \end{array}$ of the unprojected estimate $\begin{array} { r } { \eta _ { j , m } ^ { \circ } = \frac { m - 2 } { \sum _ { i = 1 } ^ { m } \widetilde { Z } _ { i , i } ^ { 2 } } } \end{array}$ . By Lemma $9 , \textstyle { \frac { 1 } { s } } \sum _ { i = 1 } ^ { m } { \widetilde { Z } } _ { j , i } ^ { 2 }$ has the law of $\chi _ { m } ^ { 2 }$ . Projection at the upper end decreases the estimate, whereas projection at the lower end increases it by at most <sup>ηmin</sup> and happens only if $\begin{array} { r } { \eta _ { j , m } ^ { \circ } < \frac { \eta _ { \mathrm { m i n } } } { 2 } } \end{array}$ , so that

$$
s \mathbb { E } \left[ \widehat { \eta } _ { j , m } \right] \leq s \mathbb { E } \left[ \eta _ { j , m } ^ { \circ } \right] + \frac { s \eta _ { \operatorname* { m i n } } } { 2 } \mathbb { P } \left[ \eta _ { j , m } ^ { \circ } < \frac { \eta _ { \operatorname* { m i n } } } { 2 } \right]\tag{216}
$$

$$
\leq 1 + \frac { 1 } { 2 } \mathbb { P } \left[ \chi _ { m } ^ { 2 } > 2 \left( m - 2 \right) \right]\tag{217}
$$

$$
\leq 1 + \frac { 1 } { 2 } \exp \left\{ 1 - \frac { m } { 2 } \left( 1 - \log 2 \right) \right\} ,\tag{218}
$$

where (217) uses $\begin{array} { r } { \mathbb { E } \left\lceil \left( \chi _ { m } ^ { 2 } \right) ^ { - 1 } \right\rceil = \frac { 1 } { m - 2 } } \end{array}$ and $s \eta _ { \mathrm { m i n } } \leq 1$ , which also shows that $\begin{array} { r } { \eta _ { j , m } ^ { \circ } < \frac { \eta _ { \mathrm { m i n } } } { 2 } } \end{array}$ forces $\begin{array} { r } { \frac { 1 } { s } \sum _ { i = 1 } ^ { m } \widetilde { Z } _ { j , i } ^ { 2 } > 2 ( m - 2 ) } \end{array}$ Markov’s inequality, applied to $\mathrm { e x p } \{ \frac { \chi _ { m } ^ { 2 } } { 4 } \}$ , whose mean is $2 ^ { \frac { m } { 2 } }$ , gives (218).

By (40) and (42), the quantities $\widehat { \eta } _ { j , n - 1 }$ and $F _ { j , n - 1 }$ are non-increasing in the squared normalized outputs $\widetilde { Z } _ { j , 1 } ^ { 2 } , \ldots , \widetilde { Z } _ { j , n - 1 } ^ { 2 } .$ which are independent by Lemma 9. Hence, 1 $\left\{ \tau ^ { ( j ) } \ge n \right\} = 1 \left\{ F _ { j , n - 1 } < \gamma _ { j } \right\}$ is non-decreasing, and association [49, Th. 2.1] gives non-positive covariance with $\widehat { \eta } _ { j , n - 1 }$ , so

$$
\begin{array} { r } { \mathbb { E } \left[ \widehat { \eta } _ { j , n - 1 } 1 \left\{ \tau ^ { ( j ) } \geq n \right\} \right] \leq \mathbb { E } \left[ \widehat { \eta } _ { j , n - 1 } \right] \mathbb { P } \left[ \tau ^ { ( j ) } \geq n \right] , \qquad n \geq 1 0 . } \end{array}\tag{219}
$$

Summing (214) over $n \geq 1$ , and bounding its terms by (215) for $n \leq 9$ and by (219) and (218) for $n \geq 1 0$ , we get

$$
\begin{array} { r l } { \mathbb { E } \left[ \displaystyle \sum _ { n = 1 } ^ { \infty } X _ { j , n } ^ { 2 ( n ) } \right] } & { = \displaystyle \sum _ { n = 1 } ^ { \infty } s P _ { j , n } \mathbb { E } \left[ \widehat { \eta } _ { \delta , n - 1 } \left\{ \tau ^ { ( i ) } \ge n \right\} \right] } \\ & { \le \mathbb { E } \left[ \displaystyle \sum _ { n = 1 } ^ { \lfloor \xi \Delta \tau \rfloor } P _ { j , n } \right] - \displaystyle \frac { 1 } { 2 } \sum _ { n = 1 } ^ { \infty } P _ { \omega , n } \mathbb { E } \left[ \sigma ^ { ( i ) } \ge n \right] } \\ & { \qquad + \displaystyle \frac { 1 } { 2 } \sum _ { n = 1 } ^ { \infty } P _ { j , n } \mathbb { E } \left[ \sigma ^ { ( i ) } \ge n \right] \exp \left\{ 1 - \displaystyle \frac { n - 1 } { 2 } \left( 1 - \log 2 \right) \right\} } \\ & { \le \mathbb { E } \left[ \displaystyle \sum _ { n = 1 } ^ { \lfloor \xi \Delta \tau \rfloor } P _ { j , n } \right] - \left( \displaystyle \frac { 9 } { 2 } - \displaystyle \frac { \exp \left\{ 1 - \displaystyle \frac { 9 } { 2 } \left( 1 - \log 2 \right) \right\} } { 2 \left( 1 - \exp \left\{ - \displaystyle \frac { 1 } { 2 } \left( 1 - \log 2 \right) \right\} \right) } \right) P _ { j , n } \mathbb { E } \left[ \tau ^ { ( i ) } \ge 1 0 \right] } \\ & { \le \mathbb { E } \left[ \displaystyle \sum _ { n = 1 } ^ { \lfloor \xi \Delta \tau \rfloor } P _ { j , n } \right] . } \end{array}\tag{220}
$$

(221)

(222)

(223)

Since $P _ { j , n }$ and P $\lceil \tau ^ { ( j ) } \geq n \rceil$ are non-increasing in n, the subtracted sum in (221) is at least $\frac { 9 } { 2 } P _ { j , 1 0 } \mathbb { P } \left[ \tau ^ { ( j ) } \geq 1 0 \right]$ , whereas the last sum is at most $\begin{array} { r } { P _ { j , 1 0 } ^ {  } \mathbb { P } [ \tau ^ { ( j ) } \geq 1 0 ] } \end{array}$ times a geometric series; this gives (222). The coefficient in (222) is positive, giving (223). The reserve of the first nine refinements thus covers the bias of the later estimates, and adding (213) proves (116).

## II. Proof of (122) and (123)

Let $j$ be a transmitted packet that satisfies (119). We condition on $\mathcal { G } _ { j , 0 }$ and write $s = \sigma _ { J _ { i } } ^ { 2 }$ ; by Lemma 9 and the continued recursion, the normalized outputs are then i.i.d. $\mathcal { N } ( 0 , s )$ at every deterministic index. The constants below depend only on the known variance interval.

1) Moments of the precision estimate: For $m \geq 9$ , the unprojected estimate $\eta _ { j , m } ^ { \circ }$ has the law of $\frac { m - 2 } { s \chi _ { m } ^ { 2 } }$ , and the inverse moments of $\chi _ { m } ^ { 2 }$ give

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left[ \left( \eta _ { j , m } ^ { \circ } - \frac { 1 } { s } \right) ^ { 2 } \right] = \frac { 2 } { s ^ { 2 } \left( m - 4 \right) } , } \\ { \displaystyle \mathbb { E } \left[ \left( \eta _ { j , m } ^ { \circ } - \frac { 1 } { s } \right) ^ { 4 } \right] = \frac { 1 2 \left( m + 1 0 \right) } { s ^ { 4 } \left( m - 4 \right) \left( m - 6 \right) \left( m - 8 \right) } = O \left( m ^ { - 2 } \right) . } \end{array}\tag{224}
$$

(225)

The projection interval $\begin{array} { r } { \left\lceil \frac { \eta _ { \mathrm { m i n } } } { 2 } , 2 \eta _ { \mathrm { m a x } } \right\rceil } \end{array}$ contains $^ { \underline { { { 1 } } } } _ { s } ,$ so projection cannot increase the distance to ${ \frac { 1 } { s } } .$ . It is active only if $\textstyle \chi _ { m } ^ { 2 } < { \frac { m } { 2 } }$ or $\begin{array} { r } { \chi _ { m } ^ { 2 } > 2 ( m - 2 ) \geq \frac { 3 \bar { m } } { 2 } } \end{array}$ , an event whose probability is at most $2 \exp \left\{ - \frac { m } { 2 4 } \right\}$ by the Chernoff bound. The Cauchy–Schwarz inequality, applied on this event with (224) and (225), gives the first two of the bounds

$$
\left| \mathbb { E } \left[ \widehat { \eta } _ { j , m } \right] - \frac { 1 } { s } \right| = O \left( \exp \left\{ - \frac { m } { 4 8 } \right\} \right) ,\tag{226}
$$

$$
\mathbb { E } \left[ \left( \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right) ^ { 2 } \right] = \frac { 2 } { s ^ { 2 } m } + O \left( m ^ { - 2 } \right) ,\tag{227}
$$

$$
\mathbb { E } \left[ \left| \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right| ^ { 4 } \right] = O \left( m ^ { - 2 } \right) ,\tag{228}
$$

and (225) gives the third directly.

2) Proof of (122): Write $p = P _ { j , m + 1 } \leq 2 q$ . Taylor’s theorem at $\begin{array} { r } { \eta = \frac { 1 } { s } } \end{array}$ , with the third derivative of $C ( p \eta )$ bounded by $p ^ { 3 }$ gives

$$
\left| R _ { j , m } - \frac { p ^ { 2 } s ^ { 2 } } { 4 \left( p + s \right) ^ { 2 } } \left( \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right) ^ { 2 } \right| \leq \frac { p ^ { 3 } } { 6 } \left| \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right| ^ { 3 } .\tag{229}
$$

We take expectations, using (227) for the quadratic term and (228) with Lyapunov’s inequality for the cubic term. Since $q \leq 1$ this yields

$$
\operatorname { \mathbb { E } } \left[ R _ { j , m } \right] = { \frac { p ^ { 2 } } { 2 \left( p + s \right) ^ { 2 } m } } + O \left( q ^ { 2 } m ^ { - 2 } \right) + O \left( q ^ { 3 } m ^ { - { \frac { 3 } { 2 } } } \right) = { \frac { p ^ { 2 } } { 2 \left( p + s \right) ^ { 2 } m } } + O \left( q ^ { 2 } m ^ { - { \frac { 3 } { 2 } } } \right) ,\tag{230}
$$

which is (122).

3) Proofof (123): We compare the stopped sum in (123) with the sum up to the deterministic time at which the true-precision reduction crosses the threshold,

$$
\begin{array} { r } { \overline { { \tau } } ^ { \left( j \right) } \triangleq \operatorname* { i n f } \left\{ n \geq 0 colon \overline { { F } } _ { j , n } \geq \gamma _ { j } \right\} . } \end{array}\tag{231}
$$

Since $\gamma _ { j } > 0$ for sufficiently large M by $( 4 5 ) , \overline { { \tau } } ^ { ( j ) } \geq 1$ . Every increment of $F _ { j , n }$ and of $\overline { { F } } _ { j , n }$ is at least $C \left( \begin{array} { c } { { { \frac { q \eta _ { \mathrm { m i n } } } { 2 } } } } \end{array} \right)$ , so both $\tau ^ { ( j ) }$ and $\overline { { \tau } } ^ { \left( j \right) }$ are at most $\left\lceil \frac { \gamma _ { j } } { C \left( \frac { q \eta _ { \mathrm { m i n } } } { 2 } \right) } \right\rceil$ . As the two reductions increase in $n , \tau ^ { ( j ) } > m$ if and only if $F _ { j , m } < \gamma _ { j }$ , and $\overline { { \tau } } ^ { \left( j \right) } > m$ if and only if $\overline { { F } } _ { j , m } < \gamma _ { j }$

The two reductions stay close. The derivative of $C \left( P _ { j , n } \eta \right)$ in $\eta > 0$ is at most ${ \frac { P _ { j , n } } { 2 } } \leq q ,$ and the first nine refinements use the precision $\frac { \eta _ { \mathrm { m i n } } } { 2 }$ . Writing $\lVert \cdot \rVert _ { 4 }$ for the fourth-moment norm, we get from Minkowski’s inequality and (228), for $m \geq 9$

$$
\left\| F _ { j , m } - \overline { { F } } _ { j , m } \right\| _ { 4 } \leq 9 q \eta _ { \operatorname* { m a x } } + q \sum _ { n = 9 } ^ { m - 1 } \left\| \widehat { \eta } _ { j , n } - \frac 1 s \right\| _ { 4 } = O \left( q \sqrt { m } \right) .\tag{232}
$$

As a consequence, the two stopping indicators rarely differ. Every increment of $\overline { { F } } _ { j , n }$ is at least $C \left( q \eta _ { \mathrm { m i n } } \right) \ge q C \left( \eta _ { \mathrm { m i n } } \right)$ , so $\overline { { F } } _ { j , m }$ is at distance at least $\frac { q C ( \overline { { \eta _ { \mathrm { m i n } } } } ) } { 2 } \left| m - \overline { { \tau } } ^ { ( j ) } \right|$ from $\gamma _ { j }$ whenever $\left| m - \overline { { \tau } } ^ { ( j ) } \right| \geq 2$ , and for such $m .$

$$
\left\{ 1 \left\{ \tau ^ { ( j ) } > m \right\} \neq 1 \left\{ \overline { { \tau } } ^ { ( j ) } > m \right\} \right\} \subseteq \left\{ \left| F _ { j , m } - \overline { { F } } _ { j , m } \right| \geq \frac { q C \left( \eta _ { \operatorname* { m i n } } \right) } { 2 } \left| m - \overline { { \tau } } ^ { ( j ) } \right| \right\} .\tag{233}
$$

Markov’s inequality for the fourth power and (232) bound the probability of the right-hand side, and the bound one covers $\left| m - \overline { { \tau } } ^ { ( j ) } \right| \leq 1$ , so that

$$
\mathbb { P } \left[ 1 \left\{ \tau ^ { ( j ) } > m \right\} \neq 1 \left\{ \tau ^ { ( j ) } > m \right\} \right] = O \left( \operatorname* { m i n } \left\{ 1 , \frac { m ^ { 2 } } { \left( 1 + \left| m - \overline { { \tau } } ^ { ( j ) } \right| \right) ^ { 4 } } \right\} \right) , \qquad m \geq 9 .\tag{234}
$$

The coefficient $\frac { P _ { j , m + 1 } s } { 2 ( P _ { j , m + 1 } + s ) }$ in (123) is at most $q .$ The stopped sum and the sum up to $\overline { { \tau } } ^ { ( j ) } - 1$ differ only in the terms with $1 \left\{ \tau ^ { ( j ) } > m \right\} \neq 1 \left\{ \overline { { \tau } } ^ { ( j ) } > m \right\}$ , so

$$
\left| \mathbb { E } \left[ \sum _ { m = 9 } ^ { \tau ^ { ( j ) } - 1 } \frac { P _ { j , m + 1 } s } { 2 \left( P _ { j , m + 1 } + s \right) } \left( \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right) \right] - \sum _ { m = 9 } ^ { \tau ^ { ( j ) } - 1 } \frac { P _ { j , m + 1 } s } { 2 \left( P _ { j , m + 1 } + s \right) } \mathbb { E } \left[ \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right] \right|
$$

$$
\leq q \sum _ { m = 9 } ^ { \infty } \mathbb { E } \left[ \left| \widehat { \eta } _ { j , m } - \frac { 1 } { s } \right| \left| 1 \left\{ \tau ^ { ( j ) } > m \right\} - 1 \left\{ \overline { { \tau } } ^ { ( j ) } > m \right\} \right| \right]\tag{235}
$$

$$
= O \left( q \sum _ { m = 9 } ^ { \infty } m ^ { - \frac { 1 } { 2 } } \operatorname* { m i n } \left\{ 1 , \frac { m } { \left( 1 + \left| m - \overline { { \tau } } ^ { ( j ) } \right| \right) ^ { 2 } } \right\} \right)\tag{236}
$$

$$
= { \cal O } \left( \displaystyle q \left( \frac { 1 } { \left( \overline { { { \tau } } } ^ { ( j ) } \right) ^ { 2 } } \sum _ { m \leq \frac { \overline { { { \tau } } } ^ { ( j ) } } { 2 } } \sqrt { m } + \sum _ { m \geq 2 \overline { { { \tau } } } ^ { ( j ) } } m ^ { - \frac { 3 } { 2 } } + \frac { 1 } { \sqrt { \overline { { { \tau } } } ^ { ( j ) } } } \sum _ { i = 0 } ^ { \infty } \operatorname* { m i n } \left\{ 1 , \frac { 2 \overline { { { \tau } } } ^ { ( j ) } } { \left( 1 + i \right) ^ { 2 } } \right\} \right) \right)  \\  = { \cal O } ( q ) .\tag{237}
$$

(238)

The Cauchy–Schwarz inequality with (227) and (234) gives (236). For (237), split at $\begin{array} { r } { m = \frac { \overline { { \tau } } ^ { ( j ) } } { 2 } } \end{array}$ and $m = 2 \overline { { \tau } } ^ { ( j ) }$ . The two outer sums contribute $O \left( \left( \overline { { \tau } } ^ { \left( j \right) } \right) ^ { - \frac { 1 } { 2 } } \right)$ . In the middle range, each distance $i = \left| m - \overline { { \tau } } ^ { \left( j \right) } \right|$ occurs at most twice; splitting the last sum at $i = \sqrt { \overline { { \tau } } ^ { \left( j \right) } }$ gives $O ( 1 )$ , proving (238). The deterministic sum in (235) is $O ( q )$ because $\begin{array} { c } { { P _ { j , m + 1 } s } } \\ { { 2 ( P _ { j , m + 1 } + s ) } } \end{array} \leq q$ and (226) is summable. The triangle inequality then gives (123).

## III. Proof of Lemma 11

Both bounds follow from (124). As in Section VII-D, write $s = \sigma _ { J _ { i } } ^ { 2 }$ and condition on $\mathcal { G } _ { j , 0 }$

1) Preliminary and exploration packets, proof of (125): For a preliminary or exploration packet, $P _ { j , n } = q = P _ { j , 0 }$ for every $n ,$ so $\begin{array} { r } { \overline { { F } } _ { j , \tau ^ { ( j ) } } = \dot { \tau } ^ { ( j ) } C \left( \frac { q } { s } \right) } \end{array}$ . Concavity of $C$ and $\begin{array} { r } { \frac { 1 } { \ell _ { 0 } } \leq q \leq 1 } \end{array}$ give

$$
\frac { \gamma _ { j } } { C \left( \frac { q \eta _ { \mathrm { m i n } } } { 2 } \right) } \le \frac { \ell _ { 0 } \gamma _ { j } } { C \left( \frac { \eta _ { \mathrm { m i n } } } { 2 } \right) } = O \left( \psi ^ { 2 } \ell _ { 0 } \right) ,\tag{239}
$$

where $\gamma _ { j } = O ( \psi ^ { 2 } )$ by Section VII-A. The summation in (124) is therefore $O ( \log \psi )$ , since its coefficient is at most $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ and $\ell _ { 0 } = \left\lceil \psi ^ { \frac { 1 } { 3 } } \right\rceil$ . Counting the initial use and applying (124), (98), and $\begin{array} { r } { C \left( \frac { q } { s } \right) \le \frac { \eta _ { \operatorname* { m a x } } } { 2 } } \end{array}$ gives

$$
C \left( \frac { q } { s } \right) \mathbb { E } \left[ \tau ^ { \left( j \right) } + 1 \mid \mathcal { G } _ { j , 0 } \right] = \mathbb { E } \left[ \overline { { F } } _ { j , \tau ^ { \left( j \right) } } \middle \vert \mathcal { G } _ { j , 0 } \right] + C \left( \frac { q } { s } \right) \leq \log M _ { j } + O \left( \log \psi \right) ,\tag{240}
$$

which proves (125).

2) Final packet, proof of (126): Set $j = B + 1$ and $q = 1$ . The first $n _ { \mathrm { b } }$ refinements use the power $\begin{array} { r } { { 1 } + \frac { \Omega } { n _ { \mathrm { b } } } } \end{array}$ , and the later ones use power one. By (106)–(107), each increment of $F _ { j , n }$ is at most $C \left( 3 \eta _ { \mathrm { m a x } } \right)$ and $n _ { \mathrm { b } } C \left( 3 \eta _ { \mathrm { m a x } } \right) = o ( \mathrm { l o g } M ) < \gamma _ { j }$ , so $\tau ^ { ( j ) } > n _ { \mathrm { b } }$ . The true-precision reduction therefore splits into the reduction at power one and the gain from the larger powers,

$$
\overline { { F } } _ { j , \tau ^ { ( j ) } } = \tau ^ { ( j ) } C \left( \frac { 1 } { s } \right) + n _ { \mathrm { b } } \left[ C \left( \frac { 1 + \frac { \Omega } { n _ { \mathrm { b } } } } { s } \right) - C \left( \frac { 1 } { s } \right) \right] .\tag{241}
$$

The gain satisfies

$$
n _ { \mathrm { b } } \left[ C \left( \frac { 1 + \frac { \Omega } { n _ { \mathrm { b } } } } { s } \right) - C \left( \frac { 1 } { s } \right) \right] = \frac { n _ { \mathrm { b } } } { 2 } \log \left( 1 + \frac { \Omega } { n _ { \mathrm { b } } \left( 1 + s \right) } \right)\tag{242}
$$

$$
\geq \frac { \Omega } { 2 \left( 1 + s \right) } - \frac { \Omega ^ { 2 } } { 4 n _ { \mathrm { b } } \left( 1 + s \right) ^ { 2 } } \geq \frac { \Omega } { 2 \left( 1 + s \right) } - \frac { 1 } { 4 \left( 1 + s \right) ^ { 2 } } ,\tag{243}
$$

by the definition of C, log $\textstyle ( 1 + x ) \geq x - { \frac { x ^ { 2 } } { 2 } }$ for $x \geq 0 .$ , and $n _ { \mathrm { b } } \geq 1 + \Omega ^ { 2 }$ from (55). For the true-precision reduction, (124) with $q = 1$ gives

$$
\mathbb { E } [ \overline { { F } } _ { j , \tau ^ { ( j ) } }  \mathcal { G } _ { j , 0 }  \leq \gamma _ { j } + \frac { 1 } { 2 ( 1 + s ) ^ { 2 } } \overbrace { \sum _ { m = 9 } ^ { \tau _ { ( } \frac { \gamma _ { ) } } { 2 } } } ^ { \lceil \frac { \gamma _ { j } } { C ( \frac { \eta _ { \mathrm { m i n } } } { 2 } ) } \rceil - 1 } \frac { 1 } { m } + \frac { \Omega } { 4 n _ { \mathrm { b } } } \sum _ { m = 1 } ^ { n _ { \mathrm { b } } } \frac { 1 } { m } + O ( 1 )\tag{244}
$$

$$
\leq \log M _ { j } + \frac { \psi } { 2 \left( 1 + s \right) ^ { 2 } } + O \left( \log \psi \right) .\tag{245}
$$

For (244), only $m < n _ { \mathrm { b } }$ has power larger than one. The derivative of $\frac { p ^ { 2 } } { 2 ( p + s ) ^ { 2 } }$ is at most $\begin{array} { r } { \frac { 4 } { 2 7 p } < \frac { 1 } { 4 } } \end{array}$ for $p \geq 1$ , so the coefficient increases by at most $\frac { \Omega } { 4 n _ { \mathrm { b } } }$ . In (245), the first harmonic sum is at most $\psi + O ( 1 )$ because its upper limit is ${ \cal O } \left( \log M \right)$ . The second contributes at most $\begin{array} { r } { \frac { \Omega } { 2 \sqrt { n _ { \mathrm { b } } } } \leq \frac { 1 } { 2 } } \end{array}$ by $\textstyle \sum _ { m = 1 } ^ { n } { \frac { 1 } { m } } \leq 2 { \sqrt { n } }$ , and $\gamma _ { j } = \log M _ { j } + O \left( \log \psi \right)$ . Taking expectations in (241), using (243) and (245), and adding $\begin{array} { r } { C \left( \frac { 1 } { s } \right) \leq \frac { \eta _ { \operatorname* { m a x } } } { 2 } } \end{array}$ for the initial use gives

$$
C \left( \frac { 1 } { s } \right) \mathbb { E } \left[ \tau ^ { ( j ) } + 1 \mid \mathcal { G } _ { j , 0 } \right] \leq \log M _ { j } - \frac { \Omega } { 2 \left( 1 + s \right) } + \frac { \psi } { 2 \left( 1 + s \right) ^ { 2 } } + O \left( \log \psi \right) ,\tag{246}
$$

which gives (126) and completes the proof of Lemma 11.

## IV. Proof of Lemma 12

If $K = 1$ , then $B = 0$ and $J _ { 0 } = J _ { B + 1 } = 1$ , and the lemma holds. Assume that $K \geq 2$ . The implicit constants below may depend on K, the instance, the variance bounds, and the parameters u, δ, and $\delta ^ { \prime }$ introduced below, but not on M or on the history. Part 1 shows that the variance estimates are accurate with high probability, and parts 2 to 5 prove (132), (128)–(130), (133), and (131), in this order.

1) Variance estimates: The retained observations consist of the initial zero-input outputs and the normalized outputs of packets $0 , 1 , \ldots , B$ . By Lemma 9, each next observation on channel k has conditional law $\mathcal { N } ( 0 , \sigma _ { k } ^ { 2 } )$ , and its channel depends only on $U$ and preceding retained observations. The reward-stack representation in Appendix A.II therefore applies, with stacks independent of one another and of $U .$ . Let $\widehat { s } _ { k , n }$ be the empirical second moment of channel $k \mathrm { { : } }$ first n stack entries, so $\widehat { s } _ { k } = \widehat { s } _ { k , \widetilde { L } _ { k } }$ at packet boundaries. For every integer $n \geq 1 , x > 0 .$ , and $\begin{array} { r } { 0 < \delta \le \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } } \end{array}$

$$
\begin{array} { r } { \mathbb { P } \left[ n d \left( \widehat { s } _ { k , n } \Vert \sigma _ { k } ^ { 2 } \right) \geq x \right] \leq 2 \exp \{ - x \} , } \end{array}\tag{247}
$$

$$
\mathbb { P } \left[ \exists k \in [ K ] , \ m \geq n \colon \left| { \widehat { s } } _ { k , m } - \sigma _ { k } ^ { 2 } \right| > \delta \right] \leq { \frac { 2 K \exp \left\{ - { \frac { n \delta ^ { 2 } } { 6 \sigma _ { \operatorname* { m a x } } ^ { 4 } } } \right\} } { 1 - \exp \left\{ - { \frac { \delta ^ { 2 } } { 6 \sigma _ { \operatorname* { m a x } } ^ { 4 } } } \right\} } } .\tag{248}
$$

Equation (247) follows from [40, Lemma $^ { 4 ] , }$ applied to both tails at fixed n. The bound $\begin{array} { r } { d \left( \sigma _ { k } ^ { 2 } \pm \delta \| \sigma _ { k } ^ { 2 } \right) \ge \frac { \delta ^ { 2 } } { 6 \sigma _ { \operatorname* { m a x } } ^ { 4 } } } \end{array}$ together with (247) at $\begin{array} { r } { x = \frac { m \delta ^ { 2 } } { 6 \sigma _ { \mathrm { m a x } } ^ { 4 } } } \end{array}$ and a union bound over $k$ and $m \geq n$ then gives (248).

Before the final packet, every count $\widetilde { L } _ { k }$ is at most the deterministic number $\overline { { { L } } } = O \left( \psi ^ { 2 } \right)$ obtained by adding $K \ell _ { 0 }$ and the upper bounds in (99) and (102). Let $\mathcal { E }$ be the event that n d $\left( \widehat { s } _ { k , n } \big | \big | \sigma _ { k } ^ { 2 } \right) \leq b$ for every $\dot { k } \in [ K ]$ and $1 \leq n \leq { \overline { { L } } }$ . On $\mathcal { E } ,$ , every queried interval (50) contains its channel’s true variance. If $k ^ { \star }$ is active, then inf $\begin{array} { r } { \mathcal { T } _ { k ^ { \star } } \leq \sigma _ { k ^ { \star } } ^ { 2 } \leq } \end{array}$ min<sub>k∈A</sub> σ<sup>2</sup><sub>k</sub> ≤ min $\boldsymbol { \mathrm { l } } \boldsymbol { k } \in \mathcal { A }$ sup $\mathcal { T } _ { k }$ so the elimination rule (51) retains $k ^ { \star }$ . The best channel therefore stays active throughout exploration on $\mathcal { E } ,$ and a singleton active set consists of $k ^ { \star }$

Since the best channel is unique, we can fix u and $\delta$ with

$$
0 < \delta \leq \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } , \qquad \sigma _ { k ^ { \star } } ^ { 2 } + \delta < u < \operatorname* { m i n } _ { k \neq k ^ { \star } } \left( \sigma _ { k } ^ { 2 } - \delta \right) .\tag{249}
$$

Let $\nu _ { \delta }$ be the event that $\left| \widehat { s } _ { k , m } - \sigma _ { k } ^ { 2 } \right| \leq \delta$ for every $k \in [ K ]$ and $m \geq \ell _ { 0 }$ . For the simultaneous confidence event $\mathcal { E } ,$ the union bound with (44) and $( 2 4 7 )$ at $x = b ,$ and then (248) with $n = \ell _ { 0 }$ and $\ell _ { 0 } \geq \psi ^ { \frac { 1 } { 3 } }$ , give

$$
\mathbb { P } \left[ \mathscr { E } ^ { \mathrm { c } } \right] \leq 2 K L \exp \{ - b \} = O \left( \frac { 1 } { \log M } \right) ,\tag{250}
$$

$$
\mathbb { P } \left[ \left( \mathcal { E } \cap \mathcal { V } _ { \delta } \right) ^ { \mathrm { c } } \right] = O \left( \frac { 1 } { \log M } \right) + O \left( \exp \left\{ - \frac { \ell _ { 0 } \delta ^ { 2 } } { 6 \sigma _ { \mathrm { m a x } } ^ { 4 } } \right\} \right) = o \left( \psi ^ { - 1 } \right) .\tag{251}
$$

2) Preliminary channel, proof of (132): On $\nu _ { \delta }$ , the initial estimates $\widehat { s } _ { k , \ell _ { 0 } }$ are within $\delta$ of the true variances, and projection onto $\left[ \sigma _ { \mathrm { m i n } } ^ { 2 } , \sigma _ { \mathrm { m a x } } ^ { 2 } \right]$ preserves this accuracy. By (249), the projected estimate of the best channel is then the smallest, so $J _ { 0 } = k ^ { \star }$ by (47). Consequently, by (248) with $n = \ell _ { 0 }$ and $\ell _ { 0 } \geq \psi ^ { \frac { 1 } { 3 } }$

$$
\mathbb { P } \left[ J _ { 0 } \neq k ^ { \star } \right] \leq \mathbb { P } \left[ \mathcal { V } _ { \delta } ^ { \mathrm { c } } \right] = O \left( \exp \left\{ - \frac { \delta ^ { 2 } } { 6 \sigma _ { \operatorname* { m a x } } ^ { 4 } } \psi ^ { \frac { 1 } { 3 } } \right\} \right) ,\tag{252}
$$

Since $\psi \exp \{ - \frac { \delta ^ { 2 } } { 6 \sigma _ { \operatorname* { m a x } } ^ { 4 } } \psi ^ { \frac { 1 } { 3 } } \}  0$ , this proves (132).

3) Exploration uses, proof of (128)–(130): For $k \neq k ^ { \star }$ , let $\begin{array} { r } { n _ { k } \triangleq 1 + \left\lceil \frac { b } { d \left( \sigma _ { k } ^ { 2 } - \delta \| u \right) } \right\rceil } \end{array}$ . Since $\textstyle { \frac { b } { \psi } } \to 1$ , each $n _ { k }$ is of order $\psi$ and exceeds $\ell _ { 0 }$ for sufficiently large M. We first show that, on $\mathcal { E } \cap \mathcal { V } _ { \delta }$ , the best channel is never explored, and a suboptimal channel k is explored only until its count reaches $n _ { k }$

Consider a history in $\mathcal { E } \cap \mathcal { V } _ { \delta }$ . After the preliminary packet, which uses the channel $J _ { 0 } = k ^ { \star }$ by part 2, the best channel has the count $\begin{array} { r } { \widetilde L _ { k ^ { \star } } \ge \tau ^ { ( \bar { 0 } ) } \ge \frac { \log \bar { M } _ { 0 } } { C ( 2 \eta _ { \operatorname* { m a x } } ) } } \end{array}$ by (99), and its estimate satisfies $\widehat { s } _ { k ^ { \star } } \leq \sigma _ { k ^ { \star } } ^ { 2 } + \delta < u . \mathrm { ~ A ~ }$ suboptimal channel k with $\widetilde { L } _ { k } \ge n _ { k }$ has the estimate $\widehat { s } _ { k } \geq \sigma _ { k } ^ { 2 } - \bar { \delta } > u$ . Since $d \left( a \Vert u \right)$ decreases in $a < u$ and increases in $a > u$ , for sufficiently large M,

$$
\widetilde { L } _ { k ^ { \star } } d \left( \widehat { s } _ { k ^ { \star } } \| u \right) \geq \frac { \log M _ { 0 } } { C \left( 2 \eta _ { \operatorname* { m a x } } \right) } d \left( \sigma _ { k ^ { \star } } ^ { 2 } + \delta \| u \right) > b ,\tag{253}
$$

$$
\widetilde L _ { k } d \left( \widehat s _ { k } \| u \right) \ge n _ { k } d \left( \sigma _ { k } ^ { 2 } - \delta \| u \right) > b , \qquad \widetilde L _ { k } \ge n _ { k } .\tag{254}
$$

The divergence d $\scriptstyle \left( a \| \widetilde { \sigma } ^ { 2 } \right)$ increases when $\widetilde { \sigma } ^ { 2 }$ moves away from $a .$ Thus, (253)–(254) give sup $\smash { \mathcal { T } _ { k ^ { \star } } ~ < \ u < }$ inf $\mathcal { T } _ { k }$ , and the elimination rule (51) removes channel k at the next decision.

Two consequences follow. First, a suboptimal channel starts an exploration packet only while $\widetilde { L } _ { k } < n _ { k } .$ , and its last packet adds $O \left( \ell _ { 0 } ^ { 2 } \right)$ observations by (101). Thus, its exploration packets contain at most $n _ { k } + O \left( \ell _ { 0 } ^ { 2 } \right)$ refinements. Each packet has at least $\frac { \ell _ { 0 } \mathrm { ` l o g ~ 2 ~ } } { C ( 2 \eta _ { \mathrm { m a x } } ) }$ refinements by (100), so including one initial use per packet gives

$$
T _ { k } \le \left( 1 + \frac { C \left( 2 \eta _ { \mathrm { m a x } } \right) } { \ell _ { 0 } \log 2 } \right) \left( n _ { k } + O \left( \ell _ { 0 } ^ { 2 } \right) \right) = n _ { k } + O \left( \psi ^ { \frac { 2 } { 3 } } \right) , \qquad k \neq k ^ { \star } ,\tag{255}
$$

where $n _ { k } = O ( \psi )$ and $\ell _ { 0 } = \left\lceil \psi ^ { \frac { 1 } { 3 } } \right\rceil$ . Second, at every decision, an active suboptimal channel has fewer than $n _ { k } = O ( \psi )$ observations, whereas $\widetilde { L } _ { k }$ ⋆ is of order $\psi ^ { 2 }$ , so the minimum-count rule never selects the best channel, and $T _ { k ^ { \star } } = 0$

The event $\mathcal { E } ^ { \mathrm { c } } \cup \mathcal { V } _ { \delta } ^ { \mathrm { c } }$ contributes o(ψ) to $\mathbb { E } \left[ T _ { k } \right]$ by (102) and (251). Taking expectations in (255) and using $\textstyle { \frac { b } { \psi } } \to 1$ , we get

$$
\operatorname* { l i m } _ { M \to \infty } \operatorname* { s u p } _ { \psi } \frac { \mathbb { E } \left[ T _ { k } \right] } { \psi } \leq \frac { 1 } { d \left( \sigma _ { k } ^ { 2 } - \delta \| u \right) } , \qquad k \neq k ^ { \star } .\tag{256}
$$

Letting $\delta \downarrow 0$ at fixed $u ,$ and then $u \downarrow \sigma _ { k ^ { \star } } ^ { 2 }$ , replaces the right-hand side by $\frac { 1 } { d \Big ( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \Big ) }$ , by continuity of d; this proves (128). With $T _ { k ^ { \star } } = 0 \mathrm { o n } \mathcal { E } \cap \mathcal { V } _ { \delta }$ , the same bound on ${ \mathcal { E } } ^ { \mathrm { c } } \cup { \mathcal { V } } _ { \delta } ^ { \mathrm { c } }$ gives (129). Finally, $\begin{array} { r } { B \leq \frac { C ( 2 \eta _ { \operatorname* { m a x } } ^ { - } ) ^ { \sim } } { \ell _ { 0 } \log 2 } \sum _ { k = 1 } ^ { K } T _ { k } } \end{array}$ by (100), and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \mathbb { E } \left[ T _ { k } \right] = O ( \psi ) } \end{array}$ by (128)–(129). These two bounds and $\ell _ { 0 } \geq \psi ^ { \frac { 1 } { 3 } }$ give (130).

4) Final channel, proof of (133): On $\mathcal { E } ,$ neither an empty interval nor an incorrect singleton selects the final channel, so the final channel can be incorrect on $\mathcal { E }$ only if exploration reaches its cap with at least two active channels. On such a history, evaluate $\mathcal { A }$ and the counts $\widetilde { L } _ { k }$ at the cap decision. Then, for sufficiently large M,

$$
\operatorname* { m i n } _ { k \in \mathcal { A } } \widetilde { L } _ { k } \ge \frac { 1 } { K } \sum _ { j = 1 } ^ { B } \tau ^ { \left( j \right) } - O \left( \ell _ { 0 } ^ { 2 } \right) \ge \frac { \psi ^ { 2 } } { 2 K } - O \left( \ell _ { 0 } ^ { 2 } \right) \ge \frac { \psi ^ { 2 } } { 3 K } .\tag{257}
$$

To see the first inequality, consider a channel of $\mathcal { A }$ with the smallest count. Since it stays active, the minimum-count rule bounds each explored channel’s count before its last packet by this final count. By (100)–(101), each packet adds $O \left( \ell _ { 0 } ^ { 2 } \right)$ observations, so summing these bounds gives the first inequality. At the cap, exploration uses at least $\lceil \psi ^ { 2 } \rceil$ channel uses, with an initial-use fraction at most $\begin{array} { r } { \frac { C ( 2 \eta _ { \mathrm { m a x } } ) } { \ell _ { 0 } \log 2 } , \ : \mathrm { s o } \sum _ { j = 1 } ^ { B } \tau ^ { ( j ) } \geq \frac { \bar { \psi } ^ { 2 } } { 2 } } \end{array}$

By (248) with $\begin{array} { r } { n = \left\lceil \frac { \psi ^ { 2 } } { 3 K } \right\rceil } \end{array}$ , except on an event of probability $\begin{array} { r } { O \left( \exp \left\{ - \frac { \delta ^ { 2 } \psi ^ { 2 } } { 1 8 K \sigma _ { \mathrm { m a x } } ^ { 4 } } \right\} \right) } \end{array}$ , every channel in A has an estimate within δ of its true variance. With all counts at least $\frac { \psi ^ { 2 } } { 3 K }$ , the fixed positive divergences in (253)–(254) and $b = O ( \psi )$ give sup $\smash {  { \mathcal { T } } _ { k ^ { \star } } < u <  { \mathrm { i n f } }  { \mathcal { T } } _ { k } }$ for every $k \in \mathcal { A } \setminus \{ k ^ { \star } \}$ and sufficiently large M. Only $k ^ { \star }$ remains after elimination, contradicting two active channels. The exceptional probability is $O \left( { \frac { 1 } { \log M } } \right)$ , so (250) gives

$$
\mathbb { P } \left[ J _ { B + 1 } \neq k ^ { \star } \right] \leq \mathbb { P } \left[ \mathcal { E } ^ { \mathrm { c } } \right] + O \left( \exp \left\{ - \frac { \delta ^ { 2 } \psi ^ { 2 } } { 1 8 K \sigma _ { \operatorname* { m a x } } ^ { 4 } } \right\} \right) = O \left( \frac { 1 } { \log M } \right) ,\tag{258}
$$

which proves (133).

5) Exploration powers, proof of (131): Fix $\begin{array} { r } { 0 < \delta ^ { \prime } \leq \frac { \sigma _ { \operatorname* { m i n } } ^ { 2 } } { 2 } } \end{array}$ with $\begin{array} { r } { 2 \delta ^ { \prime } < \operatorname* { m i n } _ { k \neq k ^ { \star } } \left( \sigma _ { k } ^ { 2 } - \sigma _ { k ^ { \star } } ^ { 2 } \right) } \end{array}$ , and let $\gamma _ { \delta ^ { \prime } }$ be $\nu _ { \delta }$ with $\delta ^ { \prime }$ in place of δ. On $\gamma _ { \delta ^ { \prime } }$ , part 2 gives $J _ { 0 } = k ^ { \star }$ , and all estimates at packet starts are within $\delta ^ { \prime }$ of their true variances. For $k \neq k ^ { \star } , P _ { k } ^ { \star } = \Pi _ { [ 0 , 1 ] } \left( 1 + \sigma _ { k ^ { \star } } ^ { 2 } - \sigma _ { k } ^ { 2 } \right)$ . The 1-Lipschitz property of projection and the lower-endpoint change of $\frac { 1 } { \ell _ { 0 } }$ in (53) give $\begin{array} { r } { | P _ { j , 0 } - P _ { k } ^ { \star } | \le 2 \dot { \delta ^ { \prime } } + \frac { 1 } { \ell _ { 0 } } } \end{array}$ on $\gamma _ { \delta ^ { \prime } }$ for every exploration packet on channel k. On the complement, $\begin{array} { r } { | P _ { j , 0 } - P _ { k } ^ { \star } | \leq 1 } \end{array}$ . Splitting the expectation and applying (102) and (248) gives

$$
\mathbb { E } \left[ \sum _ { j = 1 } ^ { B } \left| P _ { j , 0 } - P _ { k } ^ { \star } \right| \left( \tau ^ { ( j ) } + 1 \right) 1 \left\{ J _ { j } = k \right\} \right] \leq \left( 2 \delta ^ { \prime } + \frac { 1 } { \ell _ { 0 } } \right) \mathbb { E } \left[ T _ { k } \right] + O \left( \psi ^ { 2 } \exp \left\{ - \frac { \ell _ { 0 } \delta ^ { \prime 2 } } { 6 \sigma _ { \operatorname* { m a x } } ^ { 4 } } \right\} \right)\tag{259}
$$

$$
\leq \frac { 2 \delta ^ { \prime } \psi } { d \left( \sigma _ { k } ^ { 2 } \| \sigma _ { k ^ { \star } } ^ { 2 } \right) } + o ( \psi ) ,\tag{260}
$$

the last step by (128) and $\ell _ { 0 } \geq \psi ^ { \frac { 1 } { 3 } }$ . Letting $\delta ^ { \prime } \downarrow 0$ gives (131), and the proof of Lemma 12 is complete.

## ACKNOWLEDGMENT

The core ideas of this paper are the author’s, including the problem formulation, the new coding schemes, and all the main arguments of the proofs. In preparing the manuscript, the author used two AI systems, Claude Opus 5.5 (Anthropic, Max effort) and GPT-6 Astra (OpenAI, Ultra effort). They helped draft, revise, and proofread the text; prepared the figures; helped develop and check the proofs of the auxiliary results; and carried out some routine calculations. In the design of the VLSF-Bandit scheme, both systems helped the author find that the ratio-of-mixtures score in (26)–(27) yields a better second-order term than the score he had initially considered, the logarithm of the exponentiated information density averaged over the Jeffrey prior. The author independently verified every mathematical statement and proof in the final manuscript and checked each cited result against its original source. The author alone is responsible for the entire content of the manuscript, including all claims, proofs, and interpretations, and for any remaining errors.

## REFERENCES

[1] C. Shannon, “The zero error capacity of a noisy channel,” IRE Trans. Inf. Theory, vol. 2, no. 3, pp. 8–19, Sep. 1956.

[2] M. Horstein, “Sequential transmission using noiseless feedback,” IEEE Trans. Inf. Theory, vol. 9, no. 3, pp. 136–143, Jul. 1963.

[3] J. Schalkwijk and T. Kailath, “A coding scheme for additive noise channels with feedback–I: No bandwidth constraint,” IEEE Trans. Inf. Theory, vol. 12, no. 2, pp. 172–182, Apr. 1966.

[4] M. V. Burnashev, “Data transmission over a discrete channel with feedback: Random transmission time,” Probl. Inf. Transm., vol. 12, no. 4, pp. 250–265, 1976.

[5] R. Gallager, “A simple derivation of the coding theorem and some applications,” IEEE Trans. Inf. Theory, vol. 11, no. 1, pp. 3–18, Jan. 1965.

[6] H. Yamamoto and K. Itoh, “Asymptotic performance of a modified Schalkwijk-Barron scheme for channels with noiseless feedback (corresp.),” IEEE Trans. Inf. Theory, vol. 25, no. 6, pp. 729–733, Nov. 1979.

[7] J. Y. Chen, R. C. Yavas, and V. Kostina, “Variable-length codes with bursty feedback,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Taipei, Taiwan, Jun. 2023, pp. 1448–1453.

[8] Y. Polyanskiy, H. V. Poor, and S. Verdu, “Feedback in the non-asymptotic regime,” ´ IEEE Trans. Inf. Theory, vol. 57, no. 8, pp. 4903–4925, Aug. 2011.

[9] L. V. Truong and V. Y. F. Tan, “On Gaussian MACs with variable-length feedback and non-vanishing error probabilities,” arXiv:1609.00594v2, Sep. 2016.

[10] S. H. Kim, D. K. Sung, and T. Le-Ngoc, “Variable-length feedback codes under a strict delay constraint,” IEEE Commun. Lett., vol. 19, no. 4, pp. 513–516, Apr. 2015.

[11] A. R. Williamson, T. Chen, and R. D. Wesel, “Variable-length convolutional coding for short blocklengths with decision feedback,” IEEE Trans. Commun., vol. 63, no. 7, pp. 2389–2403, Jul. 2015.

[12] K. Vakilinia, S. V. S. Ranganathan, D. Divsalar, and R. D. Wesel, “Optimizing transmission lengths for limited feedback with nonbinary LDPC examples,” IEEE Trans. Commun., vol. 64, no. 6, pp. 2245–2257, Jun. 2016.

[13] R. C. Yavas, V. Kostina, and M. Effros, “Variable-length feedback codes with several decoding times for the Gaussian channel,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Melbourne, Victoria, Australia, Jul. 2021, pp. 1883–1888.

[14] ——, “Variable-length sparse feedback codes for point-to-point, multiple access, and random access channels,” IEEE Trans. Inf. Theory, vol. 70, no. 4, pp. 2367–2394, Apr. 2024.

[15] H. Yang, R. C. Yavas, V. Kostina, and R. D. Wesel, “Variable-length stop-feedback codes with finite optimal decoding times for BI-AWGN channels,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Espoo, Finland, Jun. 2022, pp. 2327–2332.

[16] L. V. Truong and V. Y. F. Tan, “On Gaussian MACs with variable-length feedback and non-vanishing error probabilities,” IEEE Trans. Inf. Theory, vol. 64, no. 4, pp. 2333–2346, Apr. 2018.

[17] R. C. Yavas and V. Y. F. Tan, “Variable-length feedback codes over known and unknown channels with non-vanishing error probabilities,” IEEE Trans Inf. Theory, vol. 71, no. 5, pp. 3271–3286, May 2025.

[18] M. Naghshvar, M. Wigger, and T. Javidi, “Optimal reliability over a class of binary-input channels with feedback,” in Proc. IEEE Inf. Theory Workshop (ITW), Lausanne, Switzerland, Sep. 2012, pp. 391–395.

[19] M. Naghshvar, T. Javidi, and M. Wigger, “Extrinsic Jensen–Shannon divergence: Applications to variable-length coding,” IEEE Trans. Inf. Theory, vol. 61, no. 4, pp. 2148–2164, Apr. 2015.

[20] H. Yang, M. Pan, A. Antonini, and R. D. Wesel, “Sequential transmission over binary asymmetric channels with feedback,” IEEE Trans. Inf. Theory, vol. 68, no. 11, pp. 7023–7042, Nov. 2022.

[21] A. Antonini, R. Gimelshein, and R. D. Wesel, “Achievable rates for low-complexity posterior matching over the binary symmetric channel,” IEEE Trans. Inf. Theory, vol. 70, no. 8, pp. 5471–5497, Aug. 2024.

[22] R. C. Yavas, “Posterior matching over binary-input memoryless symmetric channels: Non-asymptotic bounds and low-complexity encoding,” arXiv:2604.03038, Apr. 2026.

[23] ——, “A tight second-order converse bound for variable-length feedback codes,” arXiv:2609.11368, Sep. 2026.

[24] J. P. M. Schalkwijk, “A coding scheme for additive noise channels with feedback–II: Band-limited signals,” IEEE Trans. Inf. Theory, vol. 12, no. 2, pp. 183–189, Apr. 1966.

[25] R. G. Gallager and B. Nakiboglu, “Variations on a theme by Schalkwijk and Kailath,” ˘ IEEE Trans. Inf. Theory, vol. 56, no. 1, pp. 6–17, Jan. 2010.

[26] O. Shayevitz and M. Feder, “Optimal feedback communication via posterior matching,” IEEE Trans. Inf. Theory, vol. 57, no. 3, pp. 1186–1222, Mar. 2011.

[27] L. V. Truong, S. L. Fong, and V. Y. F. Tan, “On Gaussian channels with feedback under expected power constraints and with non-vanishing error probabilities,” IEEE Trans. Inf. Theory, vol. 63, no. 3, pp. 1746–1765, Mar. 2017.

[28] V. D. Goppa, “Nonprobabilistic mutual information without memory,” Probl. Control Inf. Theory, vol. 4, pp. 97–102, 1975.

[29] I. Csiszar and J. K ´ orner, ¨ Information Theory: Coding Theorems for Discrete Memoryless Systems, 2nd ed. Cambridge Univ. Press, 2011.

[30] A. Tchamkerten and I. E. Telatar, “Variable length coding over an unknown channel,” IEEE Trans. Inf. Theory, vol. 52, no. 5, pp. 2126–2145, May 2006.

[31] Y. Lomnitz and M. Feder, “Communication over individual channels,” IEEE Trans. Inf. Theory, vol. 57, no. 11, pp. 7333–7358, Nov. 2011.

[32] Y. Lomnitz and M. Feder, “Power adaptive feedback communication over an additive individual noise sequence channel,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Seoul, South Korea, Jun.–Jul. 2009, pp. 1511–1514.

[33] T. M. Cover and J. A. Thomas, Elements of Information Theory, 2nd ed. Wiley, 2006.

[34] T. Lattimore and C. Szepesvari,´ Bandit Algorithms. Cambridge Univ. Press, 2020.

[35] T. L. Lai and H. Robbins, “Asymptotically efficient adaptive allocation rules,” Adv. Appl. Math., vol. 6, no. 1, pp. 4–22, Mar. 1985.

[36] P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time analysis of the multiarmed bandit problem,” Mach. Learn., vol. 47, no. 2, pp. 235–256, 2002.

[37] A. Garivier and E. Kaufmann, “Optimal best arm identification with fixed confidence,” in Proc. 29th Annu. Conf. Learn. Theory (COLT), New York, NY, USA, Jun. 2016, pp. 998–1027.

[38] A. Garivier and O. Cappe, “The KL-UCB algorithm for bounded stochastic bandits and beyond,” in ´ Proc. 24th Annu. Conf. Learn. Theory (COLT), 2011, pp. 359–376.

[39] O. Cappe, A. Garivier, O.-A. Maillard, R. Munos, and G. Stoltz, “Kullback-Leibler upper confidence bounds for optimal sequential allocation,”´ Ann Statist., vol. 41, no. 3, pp. 1516–1541, 2013.

[40] P. Menard and A. Garivier, “A minimax and asymptotically optimal algorithm for stochastic bandits,” in´ Proc. 28th Int. Conf. Algorithmic Learn. Theory (ALT), Oct. 2017, pp. 223–237.

[41] Q. Zhao, Multi-Armed Bandits: Theory and Applications to Online Learning in Networks, ser. Synthesis Lectures on Communication Networks. Morgan & Claypool, Oct. 2019, vol. 12.

[42] M. Egger, R. Bitar, A. Wachter-Zeh, D. Gund¨ uz, and N. Weinberger, “Maximal-capacity discrete memoryless channel identification,”¨ IEEE Trans. Inf. Theory, vol. 71, no. 2, pp. 1248–1265, Feb. 2025.

[43] N. Weinberger and M. Yemini, “Multi-armed bandits with self-information rewards,” IEEE Trans. Inf. Theory, vol. 69, no. 11, pp. 7160–7184, Nov. 2023.

[44] M. Naghshvar and T. Javidi, “Active sequential hypothesis testing,” Ann. Statist., vol. 41, no. 6, pp. 2703–2738, Dec. 2013.

[45] W. Yang, G. Durisi, T. Koch, and Y. Polyanskiy, “Quasi-static multiple-antenna fading channels at finite blocklength,” IEEE Trans. Inf. Theory, vol. 60, no. 7, pp. 4232–4265, Jul. 2014.

[46] E. Kaufmann and W. M. Koolen, “Mixture martingales revisited with applications to sequential tests and confidence intervals,” J. Mach. Learn. Res., vol. 22, no. 246, pp. 1–44, 2021.

[47] B. Acciaio, M. Beiglbock, F. Penkner, W. Schachermayer, and J. Temme, “A trajectorial interpretation of Doob’s martingale inequalities,”¨ Ann. Appl. Probab., vol. 23, no. 4, pp. 1494–1505, Aug. 2013.

[48] D. Williams, Probability with Martingales. Cambridge Univ. Press, 1991.

[49] J. D. Esary, F. Proschan, and D. W. Walkup, “Association of random variables, with applications,” Ann. Math. Statist., vol. 38, no. 5, pp. 1466–1474, Oct. 1967.