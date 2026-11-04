# The Price of List Replicability: Tight Bounds for Bias Estimation

Michael Chen Iowa State University mqychen@iastate.edu

Atieh Bastan Iowa State University atie@iastate.edu

A. Pavan Iowa State University pavan@iastate.edu

N. V. Vinodchandran University of Nebraska-Lincoln vinod@unl.edu

November 2, 2026

## Abstract

We study the sample complexity of list replicable algorithms for estimating the biases of d coins. An algorithm is $( k , \varepsilon , \delta )$ -list replicable if for every bias vector $\mathbf { p } \in [ 0 , 1 ] ^ { d }$ , there is a set of outputs $L _ { \mathbf { p } } \subseteq \mathbb { R } ^ { d }$ of at most k elements such that every $\mathbf { y } \in L _ { \mathbf { p } }$ satisfies $\| \mathbf { y } - \mathbf { p } \| _ { \infty } \leq \varepsilon .$ , and the algorithm output is in $L _ { \mathbf { p } }$ with probability at least 1 − δ. Prior work gave a $( d + 1 , \varepsilon , \delta ) \ – \mathrm { l i s t }$ replicable algorithm with sample complexity $\begin{array} { r } { \widetilde O \left( \frac { d ^ { 3 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ , but no nontrivial sample complexity lower bound was known.

We establish the first non-trivial sample complexity lower bound for this problem and give an improved algorithm that matches it up to polylogarithmic factors. Together, these results determine the optimal sample complexity for list size d + 1 to be $\begin{array} { r } { \widetilde { \Theta } \left( \frac { d ^ { 2 } } { { \varepsilon } ^ { 2 } } \right) } \end{array}$ , showing that the quadratic dependence on dimension is inherent up to polylogarithmic factors. More generally, for every $k \in [ d + 1 , 2 ^ { d } ]$ , we establish matching upper and lower bounds of $\widetilde { \Theta } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right)$ on the total sample complexity.

## 1 Introduction

Typically, learning algorithms observe samples from an unknown distribution and produce an output. In general, such algorithms are not replicable: two diferent runs of the algorithm observe two diferent samples (though drawn from the same distribution), resulting in potentially diferent outputs. Motivated by this, various notions of replicable learning have been introduced in the recent literature. In particular, the work of [ILPS22] formalized a notion of replicability called ρ-replicability, and the works of [DPVWV23] and [CMY23] introduced the notion of list replicability. Since then, the notion of replicable learning has received considerable attention and has been studied in various contexts such as PAC learning, Reinforcement learning, multi-armed bandits, etc. [HIK<sup>+</sup>24, BGH<sup>+</sup>23, BLM20, EKK<sup>+</sup>23, DGSB26, EHKS23, HM25, DPWV22, BHH<sup>+</sup>26]. A basic problem studied in this area is d-CoinBiasEstimationProblem: given sample access to d coins with unknown biases, estimate every bias to within additive error ε. The d-coin bias estimation problem is a basic instance of high-dimensional mean estimation over Bernoulli product distributions and captures central challenges in designing replicable algorithms and understanding their limitations.

In this work, we focus on list replicable [DPVWV23, CMY23] algorithms for d-CoinBiasEstimationProblem. Informally, an algorithm is k-list replicable if, given sample access to an unknown distribution D, it outputs with high probability an element of a set $L _ { \mathcal { D } }$ containing at most k correct answers. This captures replicability by restricting outputs to a small set that depends only on $\mathcal { D } ,$ not on the observed samples; smaller k gives a stronger guarantee. Ideally, we want the list size to be k = 1. However, as observed in [DPVWV23], such a strong notion of replicability is impossible even for the case of single coin bias estimation.

For the d-CoinBiasEstimationProblem, it is known that there is no d-list replicable algorithm, but there is a (d + 1)-list replicable algorithm [DPVWV23]. The (d + 1)-list replicable algorithm obtains a $\scriptstyle { \frac { \epsilon } { 2 d } } - \mathrm { c l o s e }$ estimate of each coin and rounds the resulting vector using a partition scheme known as secluded partitions. Thus, the sample complexity of the (d + 1)-list replicable algorithm is $\begin{array} { r } { \widetilde O \left( \frac { d ^ { 3 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ . Note that if we do not require replicability, then there is a straightforward algorithm whose sample complexity is only $\begin{array} { r } { \widetilde { O } \left( \frac { d } { \varepsilon ^ { 2 } } \right) } \end{array}$ toss each coin $\begin{array} { r } { \widetilde { \Theta } \left( \frac { 1 } { \varepsilon ^ { 2 } } \right) } \end{array}$ times to obtain an ε-close estimate of its bias. This raises the question of whether there is a price to achieve replicability. Do replicable algorithms require inherently higher sample complexity? Motivated by this, we ask the following question:

What is the optimal sample complexity of estimating the biases of d coins with $\ell _ { \infty }$ error at most ε in a (d + 1)-list replicable manner?

We resolve this question up to polylogarithmic factors: we show that the sample complexity of list replicable algorithms for d-CoinBiasEstimationProblem is $\begin{array} { r } { \widetilde { \Theta } \left( \frac { d ^ { 2 } } { { \varepsilon } ^ { 2 } } \right) } \end{array}$ In particular, our upper bound improves the polynomial dependence on d from cubic to quadratic $[ \dot { \mathrm { D P V W V 2 3 } } ]$ , and our lower bound result shows that this improved upper bound is optimal up to polylogarithmic factors. Thus, list replicability comes at a price: a factor-d increase in sample complexity, up to polylogarithmic factors.

We generalize this result to arbitrary $k \in [ d + 1 , 2 ^ { d } ]$ , by showing that the sample complexity of k-list replicable algorithms for the problem is $\widetilde { \Theta } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right)$

## 2 Related Works

List replicability. List replicability was introduced independently in [DPVWV23, CMY23], in the contexts of coin bias estimation and PAC learning. The works most closely related to ours are [DPVWV23, $\mathrm { V W D P ^ { + } 2 4 } ]$ , which study list replicable estimation through geometric partitions and establish bounds on list size and on the sample complexity of the resulting algorithms. These works give a $( d + 1 , \varepsilon , \delta )$ -list replicable algorithm for d-CoinBiasEstimationProblem with sample complexity $\widetilde O \left( \frac { d ^ { 3 } } { \varepsilon ^ { 2 } } \right)$ , but leave open the optimal sample complexity. Our work establishes matching upper and lower bounds up to polylogarithmic factors and characterizes the tradeof between list size and sample complexity. Subsequent work has studied list replicability in other contexts, including for large-margin halfspaces [BHH<sup>+</sup>26], multi-armed bandits $[ \mathrm { C P V ^ { + } 2 5 } ]$ , and reinforcement learning $[ \mathrm { Z C P ^ { + } 2 5 } ]$

ρ-replicability. The notion of ρ-replicability was introduced in [ILPS22]. Here, a learning algorithm observes samples from a distribution and has internal randomness. An algorithm is ρ-replicable if two executions of the algorithm on independent samples from the same distribution, using the same internal random seed, produce identical outputs with probability at least $1 - \rho .$ The probability is over both the samples and the shared seed. This notion has been studied in learning and statistical estimation [ILPS22, BGH<sup>+</sup>23, HIK<sup>+</sup>24], multi-armed bandits [EKK<sup>+</sup>23, DGSB26], and reinforcement learning [EHKS23]. Of particular relevance, [HIK<sup>+</sup>24] establishes nearly matching sample complexity bounds for high-dimensional mean estimation under $\ell _ { 2 }$ and $\ell _ { \infty }$ error. For d-CoinBiasEstimationProblem, these results imply a lower bound of $\begin{array} { r } { \widetilde \Omega \left( \frac { d ^ { 2 } } { { \varepsilon } ^ { 2 } \rho ^ { 2 } } \right) } \end{array}$ total coin tosses for nonadaptive algorithms, with a matching upper bound up to polylogarithmic factors in the relevant parameter regime. While ρ-replicability requires agreement between runs using shared randomness, list replicability requires concentration on a small set of outputs determined solely by the underlying distribution.

Stability, privacy, and randomness. Global stability, introduced in [BLM20] in the study of private learning, requires that, for each distribution, some fixed accurate hypothesis be output with probability bounded below. An equivalence between global stability and list replicability in PAC learning was established in [CMY23]. For a broad class of statistical problems, [BGH<sup>+</sup>23] gives sample-eficient reductions between shared-randomness replicability, approximate diferential privacy, and perfect generalization, while also establishing computational separations between these notions. Certificate replicability, introduced in [DPVWV23], provides another way to quantify the information needed to reproduce an output. The randomness complexity of replicability and diferential privacy, and its relationship to global stability, were further studied in [HM25].

Pseudodeterminism and reproducible computation. The study of replicability in learning has its roots in pseudodeterminism in the context of randomized algorithms. Pseudodeterminism was introduced in [GG11] and further developed in [GGR13]. A randomized algorithm A is pseudodeterministic if for every input x, there is a canonical value $c _ { x }$ such that A on input x outputs $c _ { x }$ with high probability. Multipseudodeterminism [Gol25] relaxes this requirement by allowing a small set of canonical answers: for every $x ,$ there is a canonical list $L _ { x }$ such that the output of $\mathcal { A }$ on input x belongs to $L _ { x }$ with high probability. If the list $L _ { x }$ is of size at most $k ,$ then $\mathcal { A }$ is a k-pseudodeterministic algorithm. The notion of list replicability is analogous to multi-pseudodeterminism in the context of learning: the canonical set depends on the unknown distribution, and the guarantee must hold across independently drawn samples. A related notion is that of algorithms with few influential random bits [GL19]. Let A be a randomized algorithm that uses m random bits. Separate the randomness into two parts rs, where $| r | = \ell$ and $| s | = m - \ell .$ . The algorithm A is ℓ-bit influential if for every input x, and for most choices of ℓ-bit strings r, there is a canonical value $c _ { x , r }$ such that $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } , \boldsymbol { r s } ) = \boldsymbol { c _ { x , r } }$ occurs with high probability when s is randomly chosen. The notion of pseudodeterminism has been investigated in streaming algorithms [GGMW20, BKKS23, GGS23, $\mathrm { C C P V 2 5 ] }$ , communication complexity $\mathrm { [ B H H ^ { + } 2 6 , G H R ^ { + } 2 6 ] }$ , and complexity theory, including connections to probabilistic time, promise problems, and pseudodeterministic constructions [LOS21, DPWV22, $\mathrm { C L O ^ { + } 2 6 ] }$

## 3 Preliminaries

## 3.1 Notation

We use bold lowercase letters to denote vectors $\mathbf { p } , \mathbf { q } , \mathbf { x } , \mathbf { y }$ . We index them using their non-bold counterpart, $\mathrm { e . g . , } \mathrm { \bf p } = \left( p _ { 1 } , \ldots , p _ { d } \right)$ . We say pˆ is ε-close to p to denote the condition $\| { \hat { \mathbf { p } } } - \mathbf { p } \| _ { \infty } \leq \varepsilon$ . The condition $\mathbf { p } \leq \mathbf { q }$ means pointwise $\leq , \ \mathrm { i . e . } , \ \forall i . p _ { i } \ \leq \ q _ { i }$ . And when we say $\mathbf { p } \leq { \frac { \mathbf { 3 } } { \mathbf { 4 } } }$ we mean $\begin{array} { r } { \forall i . p _ { i } \le \frac { 3 } { 4 } } \end{array}$ . For a vector $\mathbf { p } \in \mathbb { R } ^ { d }$ and radius $r > 0 .$ , we use $B _ { \infty } ^ { \circ } ( r , \mathbf { p } )$ to denote an open ball of radius r with respect to $\ell _ { \infty }$ distance, i.e., $\mathcal { B } _ { \infty } ^ { \circ } ( r , \mathbf { p } ) = \{ \mathbf { q } \in \mathbb { R } ^ { d } : \| \mathbf { p } - \mathbf { q } \| _ { \infty } < r \}$ . For $n \in \mathbb N$ we let $[ n ] = \{ 1 , \dots , n \}$ and for $a , b \in \mathbb { R }$ we use $[ a , b ]$ $( a , b ) , [ a , b )$ , and $( a , b ]$ to represent the closed, open intervals, and their corresponding half-open intervals, respectively. We use $ { \mathbf { \tilde { O } } } ( \cdot ) ,  { \widetilde { \Omega } } ( \cdot ) ,  { \widetilde { \Theta } } ( \cdot )$ to subsume polylogarithmic factors, while keeping the dependence on the list size k explicit.

For distributions $P , Q$ on the same measurable space, we use $\scriptstyle { \frac { d Q } { d P } }$ to denote the Radon-Nikodym derivative; e.g., if $P , Q$ are discrete distributions on X then for all $x \in \mathcal { X }$ , it holds that $\begin{array} { r } { \frac { d Q } { d P } ( x ) = \frac { Q ( x ) } { P ( x ) } } \end{array}$ , i.e., it is simply the ratio of their probability mass functions. We use $Q \ll P$ to denote the condition that for all measurable sets A, if $P ( A ) = 0$ then $Q ( A ) = 0$

## 3.2 Sampling Algorithms

We work in the sampling model and consider randomized algorithms that have sample access to d $\in \mathbb { N }$ coins with unknown bias vector $\mathbf { p } = ( p _ { 1 } , \ldots , p _ { d } ) \in [ 0 , 1 ] ^ { d }$ . At each stage, an algorithm A may choose a coin index $I \in [ d ]$ based on its internal randomness, previous coin choices, and outcomes. Conditioned on the choice $I ,$ the algorithm receives an outcome $X \sim \mathrm { B e r n } ( p _ { I } )$ that is independent of the algorithm’s history. For $i \in [ d ]$ we use $N _ { i }$ to denote the sample complexity of the ith coin, i.e., $N _ { i }$ is the number of times coin i was tossed. We let $\begin{array} { r } { N = \sum _ { i = 1 } ^ { d } N _ { i } } \end{array}$ denote the total number of samples the algorithm has used across all d coins. We use Y to denote the output of the algorithm. To refer to this model, we say A has sample access to coins with unknown biases $ { \mathbf { p } } \in [ 0 , 1 ] ^ { d }$

Our main focus will be on the d-CoinBiasEstimationProblem problem defined as follows.

Definition 3.1 ((d, ε, δ)-CoinBiasEstimationProblem). Let $d \in \mathbb { N }$ and $\varepsilon , \delta > 0$ . Given sample access to d coins with unknown biases $\mathbf { p } = ( p _ { 1 } , \ldots , p _ { d } ) \in [ 0 , 1 ] ^ { d }$ , the goal is to output an estimate $\hat { \mathbf { p } } \in \mathbb { R } ^ { d }$ such that

$$
\operatorname* { P r } ( \left\| { \hat { \mathbf { p } } } - \mathbf { p } \right\| _ { \infty } \leq \varepsilon ) \geq 1 - \delta
$$

The randomness is over coin tosses and the algorithm’s internal randomness.

An upper bound for the (d, ε, δ)-CoinBiasEstimationProblem follows directly from Chernof Bound: sampling each coin $\begin{array} { r } { \widetilde { \Theta } \left( \frac { 1 } { \varepsilon ^ { 2 } } \right) } \end{array}$ times and outputting the empirical means gives a total sample complexity of $\begin{array} { r } { \widetilde { O } \left( \frac { d } { \varepsilon ^ { 2 } } \right) } \end{array}$ . We state a corresponding lower bound. Interested readers may see [KCG16, Sca24].

Lemma 3.2. Let $\varepsilon , \delta \in ( 0 , { \frac { 1 } { 8 } } )$ . Every algorithm for the $( d , \varepsilon , \delta )$ -CoinBiasEstimationProblem requires $\begin{array} { r } { N = \Omega \left( \frac { d } { \varepsilon ^ { 2 } } \right) } \end{array}$ samples.

We often drop $( \varepsilon , \delta )$ from the notation when they are clear from context and simply refer to it as d-CoinBiasEstimationProblem.

## 3.3 List Replicable Algorithms

We now define the notion of list replicability in the context of $d -$ CoinBiasEstimationProblem [DPVWV23].

Definition 3.3. Let $k , d \ \in \ \mathbb { N }$ and $\varepsilon , \delta \mathrm { ~  ~ { ~ > ~ } ~ } 0$ An algorithm A is $( k , \varepsilon , \delta )$ -list replicable for the $d -$ CoinBiasEstimationProblem if for each bias vector $\mathbf { p } ~ = ~ ( p _ { 1 } , \ldots , p _ { d } ) ~ \in ~ [ 0 , 1 ] ^ { d }$ there exists a set of outputs $L _ { \mathbf { p } }$ such that

1. $| L _ { \mathbf { p } } | \leq k$

2. $\forall \mathbf { y } \in L _ { \mathbf { p } \cdot } \| \mathbf { y } - \mathbf { p } \| _ { \infty } \leq \varepsilon$

3. $\operatorname* { P r } ( Y \in L _ { \mathbf { p } } ) \geq 1 - \delta$

where Y is the random variable denoting the output of A when it has sample access to coins with bias vector p. For a fixed algorithm, $L _ { \mathbf { p } }$ depends only on p and not the samples or the algorithm’s internal randomness.

A 2<sup>d</sup>-list replicable algorithm is straightforward: estimate each bias to within $^ { \frac { \varepsilon } { 2 } , }$ then round to the nearest point on an ε-spaced grid. This uses $\begin{array} { r } { \widetilde { O } \left( \frac { d } { \varepsilon ^ { 2 } } \right) } \end{array}$  total samples. The list size, however, can be reduced dramatically. We state the relevant results below.

Theorem 3.4 (Theorem 5.2 of [DPVWV23]). Let $d \in \mathbb { N }$ . For every ε, $\delta \in ( 0 , 1 )$ , there exists a $( d \mathrm { + } 1 , \varepsilon , \delta )$ -list replicable algorithm for the d-CoinBiasEstimationProblem using $\begin{array} { r } { N = O \left( \frac { d ^ { 3 } } { \varepsilon ^ { 2 } } \log \frac { d } { \delta } \right) } \end{array}$ samples.

Theorem 3.5 (Corollary 3.3 of $[ \mathrm { V W D P ^ { + } 2 4 } ] )$ . Let $d , k \in \mathbb { N }$ such that $k \in [ d + 1 , 2 ^ { d } ]$ and $\varepsilon , \delta \in ( 0 , 1 )$ There exists a $( k , \varepsilon , \delta ) \ – l i s t$ replicable algorithm for the d-CoinBiasEstimationProblem that uses $N =$ $\begin{array} { r } { O \left( \frac { d ^ { 3 } \log ^ { 2 } d } { \varepsilon ^ { 2 } \log ^ { 2 } k } \log \frac { d } { \delta } \right) } \end{array}$ samples.

As mentioned earlier, we would like the list size to be $k = 1$ to ensure “perfect replicability”, i.e., for each bias vector $\mathbf { p } \in [ 0 , 1 ] ^ { d }$ the algorithm outputs some fixed ε-close output $\mathbf { y _ { p } } \in \mathbb { R } ^ { d }$ with probability at least $1 - \delta .$ . However, such a strong notion of replicability is impossible; any $k < d + 1$ is impossible, as shown in $\left[ \mathrm { D P V W V 2 3 } \right]$

Theorem 3.6 (Theorem 5.5 of $\left[ \mathrm { D P V W V 2 3 } \right] )$ . Let $d , k \in \mathbb { N }$ . For $\begin{array} { r } { \varepsilon \in \bigl ( 0 , \frac { 1 } { 2 } \bigr ) , \delta \in \Bigl ( 0 , \frac { 1 } { d + 2 } \Bigr ) } \end{array}$ and $k < d + 1$ there is no (k, ε, δ)-list replicable algorithm for the d-CoinBiasEstimationProblem.

These results motivate our focus on the sample complexity of $( d + 1 , \varepsilon , \delta )$ -list replicable algorithms and their generalizations to $( k , \varepsilon , \delta )$ -list replicable algorithms.

We focus on list sizes $k \in [ d + 1 , 2 ^ { d } ]$ , since lists of size $k < d + 1$ are impossible, while for $k \geq 2 ^ { d }$ , the sample complexity asymptotically matches that of (d, ε, δ)-CoinBiasEstimationProblem.

## 4 Contributions

We establish tight upper and lower bounds on the sample complexity of $( d + 1 , \varepsilon , \delta )$ -list replicable algorithms for the d-CoinBiasEstimationProblem. We also generalize the result and show matching upper and lower bounds for $( k , \varepsilon , \delta )$ -list replicable algorithms for the d-CoinBiasEstimationProblem for any $k \in [ d { + } 1 , 2 ^ { d } ]$

Our lower bound results can be stated as follows; the proof of which can be found in Section 5.

Theorem 4.1 (Definition 5.7 Restated). Let ε, $\delta \in ( 0 , 1 / 8 )$ . For suficiently large d $\in \mathbb { N } ,$ every $( d + 1 , \varepsilon , \delta ) - l i s t$ replicable algorithm for the d-CoinBiasEstimationProblem requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log d } \right) } \end{array}$ samples.

Theorem 4.2 (Definition 5.11 Restated). Let $\varepsilon , \delta \in ( 0 , 1 / 8 )$ and $d , k \in \mathbb { N }$ . If $k \in [ d + 1 , 2 ^ { d } ]$ then every $( k , \varepsilon , \delta )$ -list replicable algorithm for the d-CoinBiasEstimationProblem requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right) } \end{array}$ samples.

For simplicity of analysis, we first prove the lower bound for algorithms when the number of tosses of each coin is fixed in advance. In Appendix B, we remove that assumption and prove the same lower bound for arbitrary randomized sampling-based algorithms.

Our upper bound results can be stated as follows; the proof of which can be found in Section 6. In the statements below, we do not attempt to optimize polylogarithmic factors.

Theorem 4.3 (Definition 6.8 Restated). Let $d \in \mathbb { N }$ and ε, $\delta \in ( 0 , 1 )$ . There exists a $( d + 1 , \varepsilon , \delta )$ -list replicable algorithm for the d-CoinBiasEstimationProblem that uses $\begin{array} { r } { N = O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { d } { \delta } \log \frac { 1 } { \delta } \right) } \end{array}$ samples.

Theorem 4.4 (Definition 6.13 Restated). Let $d , k \in \mathbb { N }$ such that $k \in [ d + 1 , 2 ^ { d } ]$ and ε, $\delta \in ( 0 , 1 )$ . There exists $a \left( k , \varepsilon , \delta \right)$ -list replicable algorithm for the d-CoinBiasEstimationProblem that uses $\begin{array} { r } { N = O \left( \frac { d ^ { 2 } \log d } { \varepsilon ^ { 2 } \log k } \log ^ { 2 } \frac { d } { \delta } \right) } \end{array}$ samples.  
The following two tables summarize our contributions.
<table><tr><td>Reference</td><td>Lower bound on N Upper bound on N</td></tr><tr><td>[DPVWV23]</td><td> $O \left( \frac { d ^ { 3 } } { \varepsilon ^ { 2 } } \log \frac { d } { \delta } \right)$ </td></tr><tr><td>This work</td><td> $\Omega \left( { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log d } } \right)$   $O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { d } { \delta } \log \frac { 1 } { \delta } \right)$ </td></tr></table>

Table 1: Sample complexity N of $( d + 1 , \varepsilon , \delta )$ -list replicable d-CoinBiasEstimationProblem. Our work is tight up to polylogarithmic factors. Definitions 5.7 and 6.8.

<table><tr><td>Reference</td><td>Lower bound on N Upper bound on N</td></tr><tr><td>[VWDP+24]</td><td> $O \left( \frac { d ^ { 3 } \log ^ { 2 } d } { \varepsilon ^ { 2 } \log ^ { 2 } k } \log \frac { d } { \delta } \right)$ </td></tr><tr><td>This work</td><td> $\Omega \left( { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } } \right)$   $O \left( \frac { d ^ { 2 } \log d } { \varepsilon ^ { 2 } \log k } \log ^ { 2 } \frac { d } { \delta } \right)$ </td></tr></table>

Table 2: Sample complexity N of $( k , \varepsilon , \delta )$ -list replicable d-CoinBiasEstimationProblem where $k \in [ d +$ $1 , 2 ^ { d } ]$ . Our work is tight up to polylogarithmic factors. Definitions 5.11 and 6.13.

## 5 Lower Bound

In Sections 5.1 to 5.3, we cover preliminary results used to prove the lower bound and an overview of the proof of the lower bound. In Section 5.4, we prove that every $( d + 1 , \varepsilon , \delta )$ -list replicable algorithms requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { { \varepsilon } ^ { 2 } \log d } \right) } \end{array}$ . Then in Section 5.5, we generalize it to every $( k , \varepsilon , \delta )$ -list replicable algorithms requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { { \varepsilon } ^ { 2 } \log k } \right) } \end{array}$ samples. The main tool we use is the neighborhood Sperner/KKM lemma $[ \mathrm { V W D P ^ { + } 2 4 } ]$ which we describe next.

## 5.1 The Neighborhood KKM/Sperner Lemma

Definition 5.1 (Sperner/KKM Coloring). For a set of colors C, a coloring is a mapping $\chi \colon [ 0 , 1 ] ^ { d } \to C$ For a color $c \in C$ , let $X _ { c }$ denote the set of points assigned color c by χ, i.e., ${ \boldsymbol X } _ { c } = { \boldsymbol \chi } ^ { - 1 } ( c )$ . A coloring χ is a Sperner/KKM coloring if no two points from opposite faces of the cube get the same color, i.e., for every $c \in C$ the set $X _ { c }$ has the property that for each coordinate $i \in [ d ]$ , the projection $\pi _ { i } ( X _ { c } ) = \{ p _ { i } : ( p _ { 1 } , \dots , p _ { d } ) \in X _ { c } \}$ does not contain both 0 and 1.

An analogous definition applies when we are coloring $[ a , b ] ^ { d }$ where $a < b$ . Still requiring that no two points from opposite faces of the cube get the same color, i.e., for every $c \in C$ the set $X _ { c }$ has the property that for each coordinate $i \in [ d ]$ , the projection $\pi _ { i } ( X _ { c } ) = \{ p _ { i } : ( p _ { 1 } , \dots , p _ { d } ) \in X _ { c } \}$ does not contain both a and b.

In $[ \mathrm { V W D P ^ { + } 2 4 } ]$ , the authors proved the Neighborhood Sperner/KKM Lemma for colorings of $[ 0 , 1 ] ^ { d }$ . We state the corresponding version for $[ a , b ] ^ { d }$ obtained by translating and rescaling the cube.

Theorem 5.2 (Neighborhood Sperner/KKM Lemma $[ \mathrm { V W D P ^ { + } 2 4 } ] )$ . Let $a \ < \ b$ and $s = b - a$ Given a Sperner/KKM coloring of $[ a , b ] ^ { d }$ for any $r \in ( 0 , \frac { s } { 2 } ]$ , there exists a point $\mathbf { p } \in [ a , b ] ^ { d }$ such that $B _ { \infty } ^ { \circ } ( r , \mathbf { p } )$ contains at least $\left( 1 + { \frac { 2 } { 3 } } \cdot { \frac { r } { s } } \right) ^ { d }$ points with distinct colors.

## 5.2 Proof Overview

We now provide a high-level overview of the lower-bound proof. Let A be a $( d + 1 , \varepsilon , \delta )$ -list replicable algorithm for d-CoinBiasEstimationProblem and Y be the random variable denoting the output of A. For simplicity, we assume that A does not use any internal randomness and the sample complexity is $N .$

Given ${ \mathcal { A } } .$ , we color the region $\mathcal { R } = \left\lceil \frac { 1 } { 2 } - 2 \varepsilon , \frac { 1 } { 2 } + 2 \varepsilon \right\rceil ^ { d }$ as follows: for a bias vector $\mathbf { p } \in \mathcal { R }$ , we color it with the most likely output of A when given sample access to p, i.e., color p with arg max $\mathbf { \sigma } _ { \mathbf { y } \in L _ { \mathbf { p } } } \operatorname* { P r } _ { P } ( Y = \mathbf { y } )$ , where $P$ is the distribution that algorithm A observes when given sample access to $\mathbf { p } .$ We first observe that this is a valid Sperner/KKM coloring. By the neighborhood Sperner/KKM lemma, for every radius $r \in ( 0 , 2 \varepsilon ]$ there exists a point $\mathbf { p } \in \mathcal { R }$ such that there are at least $\begin{array} { r } { M : = \left( 1 + \frac { r } { 6 \varepsilon } \right) ^ { d } } \end{array}$ distinct colors in $B _ { \infty } ^ { \circ } ( r , \mathbf { p } )$ . This means that the algorithm must change its most likely output many times in this small region. We prove that for this to happen, the sample complexity of the algorithm must be high.

Let $\mathbf { p } \in \mathcal { R }$ be the point from the Neighborhood Sperner/KKM Lemma, and let $\mathbf { q } _ { 1 } , \cdot \cdot \cdot , \mathbf { q } _ { M } \in B _ { \infty } ^ { \circ } ( r , \mathbf { p } )$ be points with distinct colors $\mathbf { y } _ { 1 } , \ldots , \mathbf { y } _ { M }$ . This means that the most likely output of A when given sample access to $\mathbf { p } _ { i }$ is $\mathbf { y } _ { i }$ . Let $E _ { i }$ be the event that the output Y is $\mathbf { y } _ { i }$ . Let $P$ be the distribution that algorithm $\mathcal { A }$ observes when given sample access to $\mathbf { p } ,$ similarly $Q _ { i }$ is the distribution A observes when given sample access to $\mathbf { q } _ { i }$

Now, we want to look at the probability of $E _ { i }$ when $\mathcal { A }$ has sample access to $\mathbf { p } ,$ let’s denote this with $\mathrm { P r } _ { P } ( E _ { i } )$ . We can relate this to the probability of $E _ { i }$ when $\mathcal { A }$ has sample access to $\mathbf { q } _ { i } .$ , we denote this with $\mathrm { P r } _ { Q _ { i } } ( E _ { i } )$ ). Our first approach was to relate them in an additive manner. Note that $\operatorname* { P r } _ { P } ( E _ { i } ) \geq \operatorname* { P r } _ { Q _ { i } } ( E _ { i } ) -$ $\operatorname { d } _ { \mathrm { T V } } ( P , Q _ { i } )$ , and by Pinsker’s inequality Pr $\begin{array} { r } { { \cal P } ( E _ { i } ) \ge \operatorname* { P r } _ { Q _ { i } } ( E _ { i } ) - \sqrt { \frac { 1 } { 2 } } \operatorname { D } _ { \mathrm { K L } } ( { \cal P } \parallel Q _ { i } ) } \end{array}$ . Since the sample complexity of algorithm is $N ,$ , this implies that $\begin{array} { r } { \operatorname* { P r } _ { P } ( E _ { i } ) \geq \operatorname* { P r } _ { Q _ { i } } ( E _ { i } ) - \sqrt { \frac { 8 } { 3 } } N \| \mathbf { p } - \mathbf { q } _ { i } \| _ { \infty } ^ { 2 } } \end{array}$ . So, we obtain the following inequality

$$
1 \geq \sum _ { i = 1 } ^ { M } \operatorname* { P r } _ { P } ( E _ { i } ) \geq \sum _ { i = 1 } ^ { M } \operatorname* { P r } _ { Q _ { i } } ( E _ { i } ) - \sum _ { i = 1 } ^ { M } { \sqrt { { \frac { 8 } { 3 } } N \| \mathbf { p } - \mathbf { q } _ { i } \| _ { \infty } ^ { 2 } } }
$$

Note that $\begin{array} { r } { \operatorname* { P r } _ { Q _ { i } } ( E _ { i } ) \geq \frac { 1 - \delta } { d + 1 } } \end{array}$ (since most likely output has probability at least $\textstyle { \frac { 1 - \delta } { d + 1 } } )$ , and $\left\| \mathbf { p } - \mathbf { q } _ { i } \right\| \leq r$ From this, we obtain (ignoring the constant $8 / 3 )$

$$
1 \ge \frac { M ( 1 - \delta ) } { d + 1 } - M r \sqrt { N }
$$

When we work out the details and solve the above inequality for $N$ , we do not obtain any non-trivial lower bound on $N$ . The main issue is that the inequality Pr $\begin{array} { r } { \mathbf { \rho } _ { P } ( E _ { i } ) \geq \operatorname* { P r } _ { Q _ { i } } ( E _ { i } ) - \sqrt { \frac { 8 } { 3 } } N \| \mathbf { p } - \mathbf { q } _ { i } \| _ { \infty } ^ { 2 } } \end{array}$ is too weak for our setting. Instead, we use a multiplicative version of the inequality P $\cdot _ { P } ( \dot { E _ { i } } ) \geq \left( \operatorname* { P r } _ { Q _ { i } } ( E ) \right) ^ { 2 } e ^ { - \frac { 1 6 } { 3 } N \| \mathbf { p } - \mathbf { q } _ { i } \| _ { \infty } ^ { 2 } }$ whose details we describe next.

## 5.3 Change of Measure for Coin Tosses

This section covers some rudimentary properties of probability theory. We prove them for the sake of completeness. Interested readers may look at [LSS14, Mir17, Mah26].

Lemma 5.3. Let $P = R _ { 1 } \otimes \ldots \otimes R _ { n }$ and $Q = S _ { 1 } \otimes . . . \otimes S _ { n }$ be product distributions. $I f \forall i \in [ n ] . S _ { i } \ll R _ { i }$ then

$$
\mathbb { E } _ { P } \left[ \left( \frac { d Q } { d P } \right) ^ { 2 } \right] = \prod _ { i = 1 } ^ { n } \mathbb { E } _ { R _ { i } } \left[ \left( \frac { d S _ { i } } { d R _ { i } } \right) ^ { 2 } \right]
$$

Proof. We prove the case $n = 2 .$ , and for simplicity we assume a discrete sample space. By using the definition of expectation and the fact that $P = R _ { 1 } \otimes R _ { 2 }$ and $Q = S _ { 1 } \otimes S _ { 2 }$ are product distributions, we have

$$
\begin{array} { l l r } { \mathbb { E } _ { P } \left[ \left( \displaystyle \frac { d Q } { d P } \right) ^ { 2 } \right] = \displaystyle \sum _ { x _ { 1 } , x _ { 2 } } \left( \frac { Q ( x _ { 1 } , x _ { 2 } ) } { P ( x _ { 1 } , x _ { 2 } ) } \right) ^ { 2 } P ( x _ { 1 } , x _ { 2 } ) } \\ { \displaystyle } & { = \displaystyle \sum _ { x _ { 1 } } \sum _ { x _ { 2 } } \left( \frac { S _ { 1 } ( x _ { 1 } ) S _ { 2 } ( x _ { 2 } ) } { R _ { 1 } ( x _ { 1 } ) R _ { 2 } ( x _ { 2 } ) } \right) ^ { 2 } R _ { 1 } ( x _ { 1 } ) R _ { 2 } ( x _ { 2 } ) } \\ { \displaystyle } & { = \left[ \sum _ { x _ { 1 } } \left( \frac { S _ { 1 } ( x _ { 1 } ) } { R _ { 1 } ( x _ { 1 } ) } \right) ^ { 2 } R _ { 1 } ( x _ { 1 } ) \right] \left[ \sum _ { x _ { 2 } } \left( \frac { S _ { 2 } ( x _ { 2 } ) } { R _ { 2 } ( x _ { 2 } ) } \right) ^ { 2 } R _ { 2 } ( x _ { 2 } ) \right] } \\ { \displaystyle } & { = \mathbb { E } _ { R _ { 1 } } \left[ \left( \displaystyle \frac { d S _ { 1 } } { d R _ { 1 } } \right) ^ { 2 } \right] \mathbb { E } _ { R _ { 2 } } \left[ \left( \displaystyle \frac { d S _ { 2 } } { d R _ { 2 } } \right) ^ { 2 } \right] } \end{array}
$$

The general case follows by induction on n.

□

Lemma 5.4. Let $p , q \in ( 0 , 1 )$ . If $R = \mathrm { B e r n } ( p )$ and $S = \operatorname { B e r n } ( q )$ then

$$
\mathbb { E } _ { R } \left[ \left( { \frac { d S } { d R } } \right) ^ { 2 } \right] = 1 + { \frac { ( p - q ) ^ { 2 } } { p ( 1 - p ) } }
$$

In particular, $\begin{array} { r } { i f \frac { 1 } { 4 } \leq p \leq \frac { 3 } { 4 } } \end{array}$ then

$$
\mathbb { E } _ { R } \left[ \left( { \frac { d S } { d R } } \right) ^ { 2 } \right] \leq e ^ { \frac { 1 6 } { 3 } ( p - q ) ^ { 2 } }
$$

Proof.

$$
\begin{array} { r l } { \mathbb { E } _ { n } \left[ \left( \frac { d S } { d R } \right) ^ { 2 } \right] = p \cdot \left( \frac { q } { p } \right) ^ { 2 } + ( 1 - p ) \cdot \left( \frac { 1 - q } { 1 - p } \right) ^ { 2 } } & { } \\ & { = \frac { q ^ { 2 } } { p } + \frac { ( 1 - q ) ^ { 2 } } { 1 - p } } \\ & { = \frac { q ^ { 2 } ( 1 - p ) + p ( 1 - q ) ^ { 2 } } { p ( 1 - p ) } } \\ & { = \frac { q ^ { 2 } + p - 2 q p } { p ( 1 - p ) } } \\ & { = \frac { p - p ^ { 2 } } { p ( 1 - p ) } + \frac { ( p - q ) ^ { 2 } } { p ( 1 - p ) } } \\ & { = 1 + \frac { ( p - q ) ^ { 2 } } { p ( 1 - p ) } } \end{array}
$$

In particular, if $p \in \left[ { \frac { 1 } { 4 } } , { \frac { 3 } { 4 } } \right]$ then due to convexity of $\frac { 1 } { p ( 1 - p ) }$ we have that $\textstyle { \frac { 1 } { p ( 1 - p ) } } \leq { \frac { 1 6 } { 3 } }$ and therefore

$$
\begin{array} { c c c } { { \mathbb { E } _ { R } \left[ \left( \displaystyle \frac { d S } { d R } \right) ^ { 2 } \right] \leq 1 + \displaystyle \frac { 1 6 } { 3 } ( p - q ) ^ { 2 } } } \\ { { \leq e ^ { \frac { 1 6 } { 3 } ( p - q ) ^ { 2 } } } } \end{array}
$$

$$
( 1 + x \leq e ^ { x } )
$$

This concludes the proof.

Let $\mathbf { p } , \mathbf { q } \in [ 0 , 1 ] ^ { d }$ , and also let $N _ { 1 } , \dots , N _ { d } \in \mathbb { N }$ with $\begin{array} { r } { N = \sum _ { i = 1 } ^ { d } N _ { i } } \end{array}$ . Here $N _ { i }$ denotes the number of times coin i was tossed, and N denotes the total number of tosses across all coins. Define the distributions

$$
P = \bigotimes _ { i = 1 } ^ { d } ( \mathrm { B e r n } ( p _ { i } ) ^ { \otimes N _ { i } } )
$$

$$
Q = \bigotimes _ { i = 1 } ^ { d } ( \mathrm { B e r n } ( q _ { i } ) ^ { \otimes N _ { i } } )
$$

Thus $P , Q$ are distributions on $\{ 0 , 1 \} ^ { N }$ where the first $N _ { 1 }$ bits are from the first coin, the next $N _ { 2 }$ bits are from the second coin, and so on.

Lemma 5.5. $I f { \begin{array} { l } { { \frac { 1 } { 4 } } } \end{array} } \leq \mathbf { p } \leq { \frac { 3 } { 4 } }$ then $\begin{array} { r } { \mathbb { E } _ { P } \left[ \left( \frac { d Q } { d P } \right) ^ { 2 } \right] \leq e ^ { \frac { 1 6 } { 3 } N \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } } } \end{array}$

Proof. By using Definitions 5.3 and 5.4 we have

$$
\begin{array} { c } { \displaystyle \mathbb { E } _ { P } \left[ \left( \displaystyle \frac { d Q } { d P } \right) ^ { 2 } \right] = \displaystyle \prod _ { i = 1 } ^ { d } \left( 1 + \frac { ( p _ { i } - q _ { i } ) ^ { 2 } } { p _ { i } ( 1 - p _ { i } ) } \right) ^ { N _ { \ast } } } \\ { \displaystyle \qquad \leq \prod _ { i = 1 } ^ { d } e ^ { \frac { 1 6 } { 3 } N _ { \ast } ( p _ { i } - q _ { \ast } ) ^ { 2 } } } \\ { \displaystyle = e ^ { \frac { 1 6 } { 3 } \sum _ { i = 1 } ^ { d } N _ { i } ( p _ { i } - q _ { i } ) ^ { 2 } } } \\ { \displaystyle \qquad \leq e ^ { \frac { 1 6 } { 3 } \sum _ { i = 1 } ^ { d } N _ { \ast } \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } } } \\ { \displaystyle = e ^ { \frac { 1 6 } { 3 } N \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } } } \end{array}
$$

Lemma 5.6. $I f { \begin{array} { l } { { \frac { 1 } { 4 } } } \end{array} } \leq \mathbf { p } \leq { \frac { 3 } { 4 } }$ then for any measurable event E

$$
\mathrm { P r } _ { P } ( E ) \geq \left( \mathrm { P r } _ { Q } ( E ) \right) ^ { 2 } e ^ { - \frac { 1 6 } { 3 } N \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } }
$$

In particular, $i f \left\| \mathbf { p } - \mathbf { q } \right\| _ { \infty } \leq r$ then

$$
\operatorname* { P r } _ { P } ( E ) \geq ( \operatorname* { P r } _ { Q } ( E ) ) ^ { 2 } e ^ { - { \frac { 1 6 } { 3 } } N r ^ { 2 } }
$$

Proof. By using properties of expectation and probability, we have

$$
\begin{array} { r l } { ( \operatorname* { P r } _ { Q } ( E ) ) ^ { 2 } = ( \operatorname { \mathbb { E } } _ { Q }  \mathbf { 1 } _ { \{ E \} } ) ^ { 2 } } \\ & { = ( \operatorname { \mathbb { E } } _ { P } [ \mathbf { 1 } _ { \{ E \} } \frac { d Q } { d P } ] ) ^ { 2 } } \\ & { \leq \operatorname { \mathbb { E } } _ { P } [ \mathbf { 1 } _ { \{ E \} } ^ { 2 } ] \operatorname { \mathbb { E } } _ { P } [ ( \frac { d Q } { d P } ) ^ { 2 } ] } \\ & { = \operatorname { \mathbb { E } } _ { P } [ \mathbf { 1 } _ { \{ E \} } ] \operatorname { \mathbb { E } } _ { P } [ ( \frac { d Q } { d P } ) ^ { 2 } ] } \\ & { = \operatorname* { P r } _ { P } ( E ) \operatorname { \mathbb { E } } _ { P } [ ( \frac { d Q } { d P } ) ^ { 2 } ] } \\ & { = \operatorname* { P r } _ { P } ( E ) \operatorname { \mathbb { E } } _ { P } [ ( \frac { d Q } { d P } ) ^ { 2 } ] } \\ & { \leq \operatorname* { P r } _ { P } ( E ) \nu ^ { \frac { \nu \alpha ~ N } { \alpha } N  \mathbf { D }  _ { C } } } \end{array}\tag{Cauchy-Schwarz}
$$

(Definition 5.5)

Rearranging the above gives us the desired bound, and substituting $\left\| \mathbf { p } - \mathbf { q } \right\| _ { \infty }$ with r gives us the second bound. □

## 5.4 Lower Bound for (d + 1)-List Replicable Algorithms

Theorem 5.7. Let $\varepsilon , \delta \in ( 0 , 1 / 8 )$ . For suficiently large $d \in \mathbb { N } _ { i }$ , every $( d + 1 , \varepsilon , \delta )$ -list replicable algorithm for the d-CoinBiasEstimationProblem requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log d } \right) } \end{array}$ samples.

For simplicity’s sake, we assume that the algorithm is deterministic and the sample complexity of each coin $i \in [ d ]$ is fixed to be $N _ { i } \in \mathbb { N }$ in advance, i.e., the algorithm receives as input a binary string of length $\begin{array} { r } { N = \sum _ { i = 1 } ^ { d } N _ { i } } \end{array}$ , where the first $N _ { 1 }$ bits are i.i.d. Bernoulli tosses according to the first coin, and the next $N _ { 2 }$ bits are i.i.d. Bernoulli tosses according to the second coin, and so on. We prove the lower bound against arbitrary algorithms with sample access to d coins in Appendix B.

Let $\mathcal { A } \colon \{ 0 , 1 \} ^ { N } \to \mathcal { V }$ be a deterministic algorithm that is $( d + 1 , \varepsilon , \delta )$ -list replicable algorithm such that each coin $i \in [ d ]$ is tossed exactly $N _ { i }$ with total sample complexity of $\begin{array} { r } { N = \sum _ { i = 1 } ^ { d } N _ { i } } \end{array}$ . Let Y be the random variable denoting the output of this algorithm. Because $\mathcal { A }$ is list replicable, for each bias vector $ { \mathbf { p } } \in [ 0 , 1 ] ^ { d }$ there exists a set $L _ { \mathbf { p } } \subseteq \mathcal { V }$ of size $| L _ { \mathbf { p } } | \leq d + 1$ such that each $\mathbf { y } \in L _ { \mathbf { p } }$ it holds that $\| \mathbf { p } - \mathbf { y } \| _ { \infty } \leq \varepsilon$ and

$$
\operatorname* { P r } _ { P } ( Y \in L _ { \mathbf { p } } ) \geq 1 - \delta
$$

We let $\mathbf { y _ { p } } = \arg \operatorname* { m a x } _ { \mathbf { y } \in L _ { \mathbf { p } } } \operatorname* { P r } _ { P } ( Y = \mathbf { y } )$ denote the most likely output from $L _ { \mathbf { p } } .$ , breaking ties with some fixed rule. Using an averaging argument, the most likely output must appear with probability at least $\frac { 1 - \delta } { | L _ { \mathbf { p } } | }$ otherwise, the sum of probabilities of outputs in $L _ { \mathbf { p } }$ would fail to add up to $1 - \delta .$ . Therefore

$$
\operatorname* { P r } _ { P } ( Y = \mathbf { y _ { p } } ) \geq { \frac { 1 - \delta } { | L _ { \mathbf { p } } | } } \geq { \frac { 7 } { 8 ( d + 1 ) } } \geq { \frac { 1 } { 2 ( d + 1 ) } } = : \alpha
$$

The second inequality was obtained by using $| L _ { \mathbf { p } } | \leq d + 1$ and $\begin{array} { r } { \delta < \frac { 1 } { 8 } } \end{array}$

If an output y is ε-close to p, and the distance between p and q is $> 2 \varepsilon$ then y cannot be ε-close to q. In particular, this must also apply to the most likely output of $L _ { \mathbf { p } }$ formalized below.

## Lemma 5.8. Let $\mathbf { p } , \mathbf { q } \in [ 0 , 1 ] ^ { d } . \ I f \ | | \mathbf { p } - \mathbf { q } | | _ { \infty } > 2 \varepsilon \ t h e n \ \mathbf { y _ { p } } \not \in L _ { \mathbf { q } } .$

Proof. Recall, by the definition of list replicability, we require that $L _ { \mathbf { p } }$ consist only of points that are ε-close to p, i.e., if $\mathbf { y } \in L _ { \mathbf { p } }$ then $\| \mathbf { p } - \mathbf { y } \| _ { \infty } \leq \varepsilon$ . In particular, this must also hold for $\mathbf { y _ { p } }$ , yielding $\| \mathbf { p } - \mathbf { y _ { p } } \| _ { \infty } \leq \varepsilon .$ Using triangle inequality, we have

$$
\| \mathbf { p } - \mathbf { y _ { p } } \| _ { \infty } + \| \mathbf { y _ { p } } - \mathbf { q } \| _ { \infty } \geq \| \mathbf { p } - \mathbf { q } \| _ { \infty }
$$

Substituting $\| \mathbf { p } - \mathbf { q } \| _ { \infty } > 2 \varepsilon$ and $\| \mathbf { p } - \mathbf { y _ { p } } \| _ { \infty } \leq \varepsilon$ and rearranging yields $\| \mathbf { y _ { p } } - \mathbf { q } \| _ { \infty } > \varepsilon$ . Thus $\mathbf { y _ { p } }$ is not ε-close to q. Therefore $\mathbf { y _ { p } } \notin L _ { \mathbf { q } }$ □

We consider the region $\begin{array} { r } { \mathcal { R } = \left[ \frac { 1 } { 2 } - 2 \varepsilon , \frac { 1 } { 2 } + 2 \varepsilon \right] ^ { d } \subseteq \left[ \frac { 1 } { 4 } , \frac { 3 } { 4 } \right] ^ { d } } \end{array}$ and we color the region R with the coloring $\chi \colon { \mathcal { R } }  { \mathcal { V } }$ defined as

$$
\chi ( \mathbf { p } ) = \mathbf { y _ { p } }
$$

## Lemma 5.9. χ is a Sperner/KKM coloring of $\mathcal { R }$

Proof. We now prove the Sperner/KKM coloring property. Consider any two points on opposite faces $\mathbf { p } , \mathbf { q } \in \mathcal { R }$ . Since they are on opposite faces, there exists a coordinate $i \in [ d ]$ such that $| p _ { i } - q _ { i } | = 4 \varepsilon$ . In particular, this implies that $\| \mathbf { p } - \mathbf { q } \| _ { \infty } \geq 4 \varepsilon > 2 \varepsilon$ . Via Definition 5.8 we have that $\chi ( \mathbf { p } ) = \mathbf { y _ { p } } \notin L _ { \mathbf { q } }$ . In particular, $\chi ( \mathbf { p } ) = \mathbf { y _ { p } } \neq \mathbf { y _ { q } } = \chi ( \mathbf { q } )$ □

Since χ is a valid Sperner/KKM coloring, the Neighborhood Sperner/KKM, Definition 5.2, applies with $\begin{array} { r } { a = \frac { 1 } { 2 } - 2 \varepsilon , b = \frac { 1 } { 2 } + 2 \varepsilon } \end{array}$ and $s = b - a = 4 \varepsilon$ . Let $r \in ( 0 , 2 \varepsilon ]$ to be decided later, and $\mathbf { p } \in [ a , b ] ^ { d }$ be the point such that $B _ { \infty } ^ { \circ } ( r , \mathbf { p } )$ that contains at least $\begin{array} { r } { M : = \left( 1 + \frac { r } { 6 \varepsilon } \right) ^ { d } } \end{array}$ colors. Let $C : = \{ \mathbf { y _ { q } } : \mathbf { q } \in B _ { \infty } ^ { \circ } ( r , \mathbf { p } ) \cap \mathcal { R } \}$ be the set of colors used to color $B _ { \infty } ^ { \circ } ( r , \mathbf { p } ) \cap \mathcal { R }$ and for each $\dot { c } \in C$ let $\mathbf { q } _ { c }$ be some point in $B _ { \infty } ^ { \circ } ( r , \mathbf { p } ) \cap \mathcal { R }$ for which $\chi ( \mathbf { q } _ { c } ) = c .$

Lemma 5.10. $\begin{array} { r } { \frac { 1 6 } { 3 } N r ^ { 2 } \geq \frac { d r } { 1 0 \varepsilon } - 2 \ln ( 2 ( d + 1 ) ) } \end{array}$

Proof. For any color $c \in C$ , let $\mathbf { q } _ { c } \in B _ { \infty } ^ { \circ } ( r , \mathbf { p } )$ be as described above, so we have $\| \mathbf { p } - \mathbf { q } _ { c } \| _ { \infty } < r$ . We invoke Definition 5.6 with the event $E = \{ Y = c \}$ yielding

$$
\operatorname* { P r } _ { P } ( Y = c ) \geq \left( \operatorname* { P r } _ { Q _ { c } } ( Y = c ) \right) ^ { 2 } e ^ { - { \frac { 1 6 } { 3 } } N r ^ { 2 } } \geq \alpha ^ { 2 } e ^ { - { \frac { 1 6 } { 3 } } N r ^ { 2 } }
$$

The last inequality follows from our definition of the coloring $\chi ,$ we colored $\mathbf { q } _ { c }$ with the most likely output $\mathbf { y _ { q _ { c } } }$ . Therefore $\operatorname* { P r } _ { Q _ { c } } ( Y = c ) = \operatorname* { P r } _ { Q _ { c } } ( Y = \mathbf { y _ { q } } _ { c } ) \geq \alpha$

Summing over all possible colors, of which there are at least M many, and because $\mathrm { P r } _ { P }$ is a probability measure, we have the following

$$
M \alpha ^ { 2 } e ^ { - \frac { 1 6 } { 3 } N r ^ { 2 } } \leq \sum _ { c \in C } \operatorname* { P r } _ { P } ( Y = c ) \leq 1
$$

Taking ln on both sides and rearranging and simplifying yields

$$
\frac { 1 6 } { 3 } N r ^ { 2 } \geq d \ln \left( 1 + \frac { r } { 6 \varepsilon } \right) - 2 \ln \frac { 1 } { \alpha }
$$

Using the fact that $x \in ( 0 , 1 ) \implies \ln ( 1 + x ) \geq x$ ln $2$ and $r \in ( 0 , 2 \varepsilon ]$ , thus $\textstyle { \frac { r } { 6 \varepsilon } } \in ( 0 , { \frac { 1 } { 3 } } ]$ we have

$$
\ln \left( 1 + { \frac { r } { 6 \varepsilon } } \right) \geq { \frac { r } { 6 \varepsilon } } \ln 2 \geq { \frac { r } { 1 0 \varepsilon } }
$$

Substituting $\begin{array} { r } { \alpha = \frac { 1 } { 2 ( d + 1 ) } } \end{array}$ yields

$$
\frac { 1 6 } { 3 } N r ^ { 2 } \geq \frac { d r } { 1 0 \varepsilon } - 2 \ln ( 2 ( d + 1 ) )
$$

Now pick $\begin{array} { r } { r = { \frac { 4 0 \varepsilon } { d } } \ln ( 2 ( d + 1 ) ) } \end{array}$ so that $\begin{array} { r } { \frac { d r } { 1 0 \varepsilon } = 4 \ln ( 2 ( d + 1 ) ) } \end{array}$ , for suficiently large $d ,$ we have $r \leq 2 \varepsilon$ as $4 0 \ln ( 2 ( d + 1 ) ) \ll d .$ Substituting r and simplifying the expression yields

$$
N \geq { \frac { 3 } { 1 2 8 0 0 } } \cdot { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \ln ( 2 ( d + 1 ) ) } } = \Omega \left( { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log d } } \right)
$$

This concludes the proof for Definition 5.7 for the case of deterministic algorithms whose sample complexities are fixed in advance. Extending it to arbitrary randomized sampling-based algorithms is mostly technical; we need to inspect the distribution on the transcript $( W , I _ { 1 } , X _ { 1 } , \ldots , I _ { N } , X _ { N } )$ , where W is the internal randomness of the algorithm, for $t \in [ N ] , I _ { t }$ is the coin selected at time t and $X _ { t }$ is the outcome of the toss at time t. Bias vectors $\mathbf { p } , \mathbf { q } \in [ 0 , 1 ] ^ { d }$ induce probability distributions $P , Q$ on the transcript as described above. If $\mathbf { p } \in \left[ { \frac { 1 } { 4 } } , { \frac { 3 } { 4 } } \right] ^ { d }$ then the relation in Definition 5.5 also holds in this case $\begin{array} { r } { \mathbb { E } _ { P } \left[ \left( \frac { d Q } { d P } \right) ^ { 2 } \right] \leq e ^ { \frac { 1 6 } { 3 } N \left\| \mathbf { p } - \mathbf { q } \right\| _ { \infty } ^ { 2 } } } \end{array}$ and remaining argument is the same.

## 5.5 Lower Bound for k-List Replicable Algorithms

We now generalize Definition 5.7 to a list of size k as follows

Theorem 5.11. Let $\varepsilon , \delta \in ( 0 , 1 / 8 )$ and $d , k \in \mathbb { N } . ~ I f k \in [ d + 1 , 2 ^ { d } ]$ then every $( k , \varepsilon , \delta )$ )-list replicable algorithm for the d-CoinBiasEstimationProblem requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right) } \end{array}$ samples.

Proof. First assume $k \leq \frac { 1 } { 2 } e ^ { \frac { d } { 2 0 } }$ then the argument remains similar from Section $5 ;$ the main bound that changes is

$$
\operatorname* { P r } _ { P } ( Y = \mathbf { y _ { p } } ) \geq { \frac { 1 - \delta } { | L _ { \mathbf { p } } | } } \geq { \frac { 7 } { 8 k } } \geq { \frac { 1 } { 2 k } } = : \beta
$$

Definition 5.10 now becomes

$$
\frac { 1 6 } { 3 } N r ^ { 2 } \geq \frac { d r } { 1 0 \varepsilon } - 2 \ln ( 2 k )
$$

If $k \leq \textstyle { \frac { 1 } { 2 } } e ^ { \frac { d } { 2 0 } }$ then our choice $\begin{array} { r } { r : = { \frac { 4 0 \varepsilon } { d } } \ln ( 2 k ) \leq 2 \varepsilon \quad } \end{array}$ holds and therefore

$$
N \geq \frac { 3 } { 1 2 8 0 0 } \cdot \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \ln ( 2 k ) } = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right)
$$

For $k \mathrm { ~  ~ { ~ \gamma ~ } ~ } > \frac { 1 } { 2 } e ^ { \frac { d } { 2 0 } }$ every $( k , \varepsilon , \delta ) \ – \mathrm { l i s t }$ replicable algorithm also solves the $( d , \varepsilon , \delta ) \cdot$ CoinBiasEstimationProblem. Using Definition 3.2 gives

$$
N = \Omega \left( { \frac { d } { \varepsilon ^ { 2 } } } \right) = \Omega \left( { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } } \right)
$$

The last bound was by using the fact that $\begin{array} { r } { k > \frac { 1 } { 2 } e ^ { \frac { d } { 2 0 } } } \end{array}$

Note that, even obtaining a list size of $k = { \mathrm { p o l y } } ( d )$ requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log d } \right) } \end{array}$ samples.

## 6 Upper Bound

In this section, we prove Definitions 6.8 and 6.13. We first introduce the main concepts intuitively in Section 6.1 and then formalize the notions in Section 6.2.

## 6.1 Proof Overview

Consider a simplified setting with d = 2 coins and an ε-grid as depicted in Figure 1. Let $\mathbf { p } \in [ 0 , 1 ] ^ { d }$ be a bias vector. A simple idea is to estimate the bias of each coin within $\frac { \varepsilon } { 2 }$ . Then, we round the estimate to the bottom left of the ε-square in which it falls, this has the efect of rounding both coordinates down to the nearest integer multiple of ε. However, this algorithm is not 3-list replicable, because if p lies on a grid point, then the approximation can lie in any of the 4 adjacent ε-squares. To avoid this, we use the standard technique of random shifting and rounding. The technical challenge is to bound the sample complexity.

![](images/fd2b36c44140f3b71aed95e4e6ea3e6006c4621c73e89d681dcf34ea938759ea.jpg)  
Figure 1: Intuition of the Upper Bound Algorithm

Given $\mathbf { q } = ( x , y ) \in [ 0 , 1 ] ^ { 2 }$ , the $( \alpha _ { x } , \alpha _ { y } )$ -confidence interval around q is the rectangle with corners from $( x \pm \alpha _ { x } , y \pm \alpha _ { y } )$ . Let $u \in [ 0 , \varepsilon )$ be a shift parameter that induces a line segment of length ε (in $\ell _ { \infty }$ sense) at a 45<sup>◦</sup> angle. Note that this line intersects at most 3 cells (assuming each grid point belongs to a single cell). We estimate the shifted vector $\mathbf { p } ^ { \prime } = \mathbf { p } + { \boldsymbol { u } } \cdot \mathbf { 1 }$ by adding u to each empirical mean obtained from the original coins. We will focus on the largest confidence interval $( \alpha _ { x } , \alpha _ { y } )$ around $\mathbf { p } ^ { \prime } = ( x , y )$ that is strictly contained inside a single ε-grid cell. If we have an estimate ˆx that is $\alpha _ { x } .$ -close to x and an estimate $\hat { y }$ that is $\alpha _ { y } .$ -close to $y ,$ then all possible estimates lie in a single cell with high probability. Thus, rounding the estimate $( \hat { x } , \hat { y } )$ to the bottom left of the cell yields the same value. The sample complexity required to get samples within this confidence interval with high probability is $\begin{array} { r } { \widetilde O \left( \frac { 1 } { \alpha _ { x } ^ { 2 } } + \frac { 1 } { \alpha _ { y } ^ { 2 } } \right) } \end{array}$ . Since any line segment of length $\varepsilon { \mathrm { ~ a t ~ } } 4 5 ^ { \circ }$ from p intersects at most three cells, this ensures that if we randomly pick $u ,$ the algorithm is 3-list replicable (provided we know $\alpha _ { x }$ and $\alpha _ { y } )$

For the purposes of illustration, we classify the regions along this line into 3 regions colored based on the behavior of their confidence intervals. When $\mathbf { p ^ { \prime } }$ lies in a green region, the required confidence interval is $\alpha _ { x } \approx \varepsilon$ and $\alpha _ { y } \approx \varepsilon$ . In the orange region, the confidence interval needs to satisfy $\alpha _ { x } \ll \varepsilon { \mathrm { ~ o r ~ } } \alpha _ { y } \ll \varepsilon .$ Finally, in the red region, a confidence interval of $\alpha _ { x } \approx 0$ or $\alpha _ { y } \approx 0$ is required. Thus, some shifts (orange region) require higher sample complexity, whereas some shifts (green region) require fewer samples. Shifts that lead to the red region can be considered a failure of the algorithm. Our algorithm limits the work spent on these shifts, keeping the expected sample complexity small when the shift is chosen uniformly at random. A technicality is that our algorithm will not know $\alpha _ { x } , \alpha _ { y }$ a priori, but will iteratively get smaller and smaller estimates until the confidence interval lies in a single cell.

In the case of d coins, our analysis shows that a uniformly at random selected $U \sim \mathrm { U n i f } ( [ 0 , \varepsilon ) )$ yields an expected sample complexity of $\begin{array} { r } { \tilde { O } ( \frac { d } { \varepsilon ^ { 2 } } ) } \end{array}$ per coin; thus the expected total sample complexity is $\begin{array} { r } { \tilde { O } \big ( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \big ) } \end{array}$ . List Replicability follows from the fact that this line segment intersects at most d + 1 grid cells in d dimensions.

## 6.2 Shifts, Rounding and Canonical List

To round down $x \in$ R to the nearest $\varepsilon ,$ it can be done so by the mapping $x \mapsto \lfloor { \frac { x } { \varepsilon } } \rfloor \varepsilon$ . For example, if $x = 2 . 5 \varepsilon$ then it maps to $2 \varepsilon$ . So $\frac { x } { \varepsilon }$ extracts what multiple of $\varepsilon$ is x (in this case 2.5), and then we floor only the multiple part of $\varepsilon$ to the nearest integer. This flooring process reduces the output by at most ε.

For our algorithm, before rounding, we perturb the initial x by a shift $u \in [ 0 , \varepsilon )$ and then round down to the nearest multiple of ε.

![](images/c4b6b35b318232bde98a3f94a33fb52a0bfa906a933392c1ad3a8b80d28eeada.jpg)  
Figure 2: u is an α-good shift for $p$

Definition 6.1. Let $\alpha > 0$ . A shift $u \in [ 0 , \varepsilon )$ is α-good for $p \in [ 0 , 1 ]$ if the interval $[ p - \alpha , p + \alpha )$ when shifted by u is contained between some consecutive integer multiple of ε. Formally, there exists an integer $k \in \mathbb { Z }$ such that

$$
[ ( p + u ) - \alpha , ( p + u ) + \alpha ) \subseteq [ k \varepsilon , ( k + 1 ) \varepsilon )
$$

Definition 6.2. Let $u \in [ 0 , \varepsilon )$ . The u-rounding of $x \in \mathbb { R }$ , denoted by $R _ { u } ( x )$ , is defined as

$$
R _ { u } ( x ) = \left\lfloor { \frac { x + u } { \varepsilon } } \right\rfloor \varepsilon
$$

For a vector $\mathbf { x } \in \mathbb { R } ^ { d }$ , the u-rounding of $\mathbf { x } ,$ denoted by $R _ { u } ( { \bf x } )$ , is obtained by applying index-wise u-rounding to $\mathbf { x } ,$ i.e., if $\mathbf { x } = ( x _ { 1 } , \dots , x _ { d } )$ then $R _ { u } ( \mathbf { x } ) = ( R _ { u } ( x _ { 1 } ) , \ldots , R _ { u } ( x _ { d } ) )$

Lemma 6.3. Let $u \in [ 0 , \varepsilon )$ . For any $x \in \mathbb { R }$

$$
| R _ { u } ( x ) - x | < \varepsilon
$$

For any $\mathbf { x } \in \mathbb { R } ^ { d }$ ，

$$
\| R _ { u } ( \mathbf { x } ) - \mathbf { x } \| _ { \infty } < \varepsilon
$$

Proof. For every $y \in \mathbb R$ , we have $y - 1 < \lfloor y \rfloor \le y$ . Applying this with $\begin{array} { r } { y = \frac { x + u } { \varepsilon } } \end{array}$ and multiplying by ε gives

$$
x + u - \varepsilon < R _ { u } ( x ) \leq x + u
$$

Since $u \in [ 0 , \varepsilon )$ , it holds that

$$
x - \varepsilon < R _ { u } ( x ) < x + \varepsilon
$$

and hence $| R _ { u } ( x ) - x | < \varepsilon$ . Applying this bound to each coordinate gives

$$
\| R _ { u } ( \mathbf { x } ) - \mathbf { x } \| _ { \infty } = \operatorname* { m a x } _ { i \in [ d ] } | R _ { u } ( x _ { i } ) - x _ { i } | < \varepsilon
$$

Consider an interval $[ p - \alpha , p + \alpha )$ of length $2 \alpha \leq \varepsilon$ . This interval need not be contained between some $[ k \varepsilon , ( k + 1 ) \varepsilon )$ , however by shifting it by u that is α-good, $\left[ \left( p + u \right) - \alpha , \left( p + u \right) + \alpha \right)$ is now contained in some $[ k \varepsilon , ( k + 1 ) \varepsilon )$ . Crucially, all points in this shifted interval round down to kε; this property is what we shall be using in our upper bound algorithms. We now formalize this and prove some additional properties that we will require for our algorithm.

Lemma 6.4. Let $u \in [ 0 , \varepsilon ) , \alpha > 0$ and $p , \hat { p } \in [ 0 , 1 ]$ such that $| \hat { p } - p | \leq \alpha$

1. If u is α-good for p then for any x, $y \in [ p - \alpha , p + \alpha )$ it holds that $R _ { u } ( x ) = R _ { u } ( y )$

2. If u is 3α-good for p, then u is 2α-good for pˆ.

3. If u is 2α-good for pˆ, then $R _ { u } ( p ) = R _ { u } ( \hat { p } )$

Proof.

1. It holds that $x + u$ and $y + u$ both belong to $[ k \varepsilon , ( k + 1 ) \varepsilon )$ by definition of u being α-good for p, and therefore both shifted points lie in the same interval and round down to $k \varepsilon = R _ { u } ( x ) = R _ { u } ( y )$

2. Observe that

$$
\left[ ( { \hat { p } } + u ) - 2 \alpha , ( { \hat { p } } + u ) + 2 \alpha \right) \subseteq \left[ ( p + u ) - 3 \alpha , ( p + u ) + 3 \alpha \right)
$$

Thus assuming u is 3α-good for p implies that u is 2α-good for $\hat { p }$ by definition.

3. Since $| p - \hat { p } | \leq \alpha < 2 \alpha$ , both $p , \hat { p } \in [ \hat { p } - 2 \alpha , \hat { p } + 2 \alpha )$ . Because u is 2α-good for $\hat { p } ,$ using item 1 with center ˆp and radius 2α, gives $R _ { u } ( \hat { p } ) = R _ { u } ( p )$ □

Lemma 6.5. Let $\alpha \in \left( 0 , \frac { \varepsilon } { 2 } \right]$ and $p \in [ 0 , 1 ]$ . If $U \sim \operatorname { U n i f } ( [ 0 , \varepsilon ) )$ then

$$
\operatorname* { P r } _ { U } ( U { \mathrm { ~ i s ~ n o t ~ } } \alpha \mathrm { - g o o d ~ f o r ~ } p ) = { \frac { 2 \alpha } { \varepsilon } }
$$

Furthermore, the equation above also holds when p is a random variable and is independent of U.

Proof. For simplicity assume $p \ = \ k \varepsilon$ for some $k \in \mathbb N$ . Observe that a shift u is α-good if and only if $u \in [ \alpha , \varepsilon - \alpha ]$ . Thus we have

$$
\operatorname* { P r } _ { U } ( U { \mathrm { ~ i s ~ n o t ~ } } \alpha { \mathrm { - g o o d ~ f o r ~ } } p ) = \operatorname* { P r } _ { U } ( U \not \in [ \alpha , \varepsilon - \alpha ] ) = { \frac { 2 \alpha } { \varepsilon } }
$$

In general, if p is not an exact multiple of ε, then u is α-good if and only if $( p + u )$ mod $\varepsilon \in [ \alpha , \varepsilon - \alpha ]$ , we also have that $( p + U )$ mod $\varepsilon \sim \operatorname { U n i f } ( [ 0 , \varepsilon ) )$ , so the equality still holds.

For the furthermore part, averaging over all values of the random variable p yields the result.

$$
{ \begin{array} { r l } & { \operatorname* { P r } ( U \operatorname { i s } \ \operatorname { n o t } \ \alpha - \operatorname* { g o o d } \ \operatorname { f o r } \ p ) } \\ & { \quad = \displaystyle \sum _ { x } \operatorname* { P r } _ { U } ( U \ \operatorname { i s } \ \operatorname { n o t } \ \alpha - \operatorname* { g o o d } \ \operatorname { f o r } \ p \ | \ p = x ) \operatorname* { P r } _ { p } ( p = x ) } \\ & { \quad = { \frac { 2 \alpha } { \varepsilon } } \displaystyle \sum _ { x } \operatorname* { P r } _ { p } ( p = x ) } \\ & { \quad = { \frac { 2 \alpha } { \varepsilon } } } \end{array} }
$$

Suppose we have a single coin with bias $p \in [ 0 , 1 ]$ . Getting an estimate $\hat { p }$ such that $\hat { p } \in [ p - \alpha , p + \alpha )$ with high probability can be achieved by simply tossing the coin $\textstyle { \widetilde { \Theta } } ( { \frac { 1 } { \alpha ^ { 2 } } } )$ times and outputting the empirical mean. We sample $U \sim \operatorname { U n i f } ( [ 0 , \varepsilon ) )$ and get iteratively better approximations $\begin{array} { r } { \alpha _ { j } = \frac { \varepsilon } { 2 ^ { j + 1 } } } \end{array}$ , till $U$ is $2 \alpha _ { j } { \mathrm { - g o o d } }$ for ˆp, i.e., the interval $[ \hat { p } - 2 \alpha _ { j } , \hat { p } + 2 \alpha _ { j } )$ when shifted by $U$ fits in some $[ k \varepsilon , ( k + 1 ) \varepsilon ) ]$ , once this condition is met, and assuming our estimates are correct we have $R _ { U } ( \hat { p } ) = R _ { U } ( p )$ , i.e., the u-rounding $\hat { p }$ is exactly the u-rounding of $p ,$ the unknown bias. If this procedure takes too many samples, we can cut it of early and output ⊥. Even though we don’t know the bias $p ,$ we can with high probability get $R _ { U } ( p )$

The rounded value of $p$ crucially depends on $U$ . Suppose $p = 2 . 5 \varepsilon , { \mathrm { i f ~ } } u \in \left[ 0 , { \frac { \varepsilon } { 2 } } \right)$ then $R _ { u } ( p ) = 2 \varepsilon$ , and if $u \in [ \frac { \varepsilon } { 2 } , \varepsilon )$ then $R _ { u } ( p ) = 3 \varepsilon$ . We now extend this idea to higher dimensions.

Definition 6.6. Let $ { \mathbf { p } } \in [ 0 , 1 ] ^ { d }$ . The set ${ \mathcal { L } } _ { \mathbf { p } }$ is defined as

$$
\mathcal { L } _ { \mathbf { p } } = \{ R _ { u } ( \mathbf { p } ) \mid u \in [ 0 , \varepsilon ) \}
$$

Lemma 6.7. For any $ { \mathbf { p } } \in [ 0 , 1 ] ^ { d }$ , the set ${ \mathcal { L } } _ { \mathbf { p } }$ satisfies

1. $\forall \mathbf { y } \in \mathcal { L } _ { \mathbf { p } \cdot } \left\| \mathbf { y } - \mathbf { p } \right\| _ { \infty } < \varepsilon$

2. $| { \mathcal { L } } _ { \bf p } | \leq d + 1$

Proof.

1. Let $\mathbf { y } \in \mathcal { L } _ { \mathbf { p } }$ , then we have $\mathbf { y } = R _ { u } ( \mathbf { p } )$ for some $u \in [ 0 , \varepsilon )$ , by Definition 6.3 we have

$$
\left\| \mathbf { y } - \mathbf { p } \right\| _ { \infty } = \left\| R _ { u } ( \mathbf { p } ) - \mathbf { p } \right\| _ { \infty } < \varepsilon
$$

2. As u increases over $[ 0 , \varepsilon )$ , each coordinate $R _ { u } ( p _ { i } )$ changes at most once. Thus the vector changes at most d times and takes at most $d + 1$ values. □

Example. Consider $d = 4$ and $\begin{array} { r } { \mathbf p = ( \frac { \varepsilon } { 5 } , \frac { 2 \varepsilon } { 5 } , \frac { 3 \varepsilon } { 5 } , \frac { 4 \varepsilon } { 5 } ) } \end{array}$ then

1. If $u \in [ 0 , \frac { \varepsilon } { 5 } )$ then $R _ { u } ( \mathbf { p } ) = ( 0 , 0 , 0 , 0 )$

2. If $u \in [ \frac { \varepsilon } { 5 } , \frac { 2 \varepsilon } { 5 } )$ then $R _ { u } ( \mathbf { p } ) = ( 0 , 0 , 0 , \varepsilon )$

3. If $u \in [ \frac { 2 \varepsilon } { 5 } , \frac { 3 \varepsilon } { 5 } )$ then $R _ { u } ( \mathbf { p } ) = ( 0 , 0 , \varepsilon , \varepsilon )$

4. If $u \in [ \frac { 3 \varepsilon } { 5 } , \frac { 4 \varepsilon } { 5 } )$ then $R _ { u } ( \mathbf { p } ) = ( 0 , \varepsilon , \varepsilon , \varepsilon )$

5. If $u \in [ \frac { 4 \varepsilon } { 5 } , \varepsilon )$ then $R _ { u } ( \mathbf { p } ) = ( \varepsilon , \varepsilon , \varepsilon , \varepsilon )$

Thus, in this example $| { \mathcal { L } } _ { \bf { p } } | = d + 1 \ \mathrm { e x a c t l y }$

Our algorithms will attempt to output some element in the set ${ \mathcal { L } } _ { \mathbf { p } }$ with high probability. Let Y be a random variable denoting the output of a list replicable algorithm which belongs to $\mathbb { R } ^ { d } \cup \{ \bot \}$ , where ⊥ is a failure symbol. The output can be categorized into 3 mutually exclusive cases

1. $Y \in { \mathcal { L } } _ { \mathbf { p } }$ , this is the desired event

2. $Y = \perp$ , not a valid member of $\mathbb { R } ^ { d }$

3. $Y \notin { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \}$ , a valid member $\mathbb { R } ^ { d }$ but not in the list

For the purpose of analysis, we will be upper bounding the quantities $\operatorname* { P r } ( Y = \bot )$ and $\operatorname* { P r } ( Y \not \in { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \} )$ as it is easier to argue over. However, it might be easier to interpret the bounds as follows: suppose

$$
\begin{array} { r } { \operatorname* { P r } ( Y = \bot ) \leq \displaystyle \frac { 1 } { 4 } } \\ { \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathbf { p } } \cup \{ \bot \} ) \leq \eta } \end{array}
$$

This means that, with probability at least $3 / 4$ the algorithm will return a vector in $\mathbb { R } ^ { d }$ , and conditioned on witnessing that $Y \neq \perp$ , the probability that $Y \in { \mathcal { L } } _ { \mathbf { p } }$ is at least $\begin{array} { r } { 1 - \frac { 4 \eta } { 3 } } \end{array}$ . Formally,

$$
\operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathbf { p } } \mid Y \neq \bot ) = \frac { \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathbf { p } } \cup \{ \bot \} ) } { \operatorname* { P r } ( Y \neq \bot ) } \leq \frac { 4 \eta } { 3 }
$$

## 6.3 Upper Bound for (d + 1)-List Replicable Algorithms

We prove the following theorem

Theorem 6.8. Let $d \in \mathbb { N }$ and $\varepsilon , \delta \in ( 0 , 1 )$ . There exists a $( d + 1 , \varepsilon , \delta )$ -list replicable algorithm for the d-CoinBiasEstimationProblem that uses $\begin{array} { r } { N = O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { d } { \delta } \log \frac { 1 } { \delta } \right) } \end{array}$ samples.

We first prove the existence of an algorithm whose expected sample complexity is $\begin{array} { r } { \mathbb { E } [ N ] = \widetilde { \cal O } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ . Then using standard boosting techniques, we get Definition 6.8.

Define the following parameters

ε ∈ (0, 1)   
η ∈ (0, 1/8)   
m = ⌈log (24d)⌉   
α<sub>j</sub> = ε<sub>j+1</sub> for j ∈ [m]   
1 2dm   
n<sub>j</sub> = & <sub>2α</sub>2<sub>j</sub> ln for j ∈ [m]   
η   
1536d<sup>2</sup> 2dm d<sup>2</sup> d   
M = ln = O ln   
ε<sup>2</sup> η ε<sup>2</sup> η

We now describe a $( d + 1 , \varepsilon , O ( 1 ) )$ -list replicable algorithm for the d-CoinBiasEstimationProblem whose expected sample complexity is bounded above $\begin{array} { r } { M = O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \ln \frac { d } { \eta } \right) } \end{array}$ . We use the same ε to perform rounding as described Section 6.2.

```latex
Algorithm 1: $( d + 1 , \varepsilon , O ( 1 ) )$ -List Replicable Algorithm with $\begin{array} { r } { \widetilde O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ expected total sample com
plexity
1 Draw U ∼ Unif([0, ε));
2 $( y _ { 1 } , \dotsc , y _ { d } )  ( \bot , \dotsc , \bot )$
3 m = ⌈log (24d)⌉
4 A ← [d]
5 for $j = 1 , \ldots , m$ do
6 foreach i ∈ A do
7 $\textstyle \alpha _ { j } \gets { \frac { \varepsilon } { 2 ^ { j + 1 } } }$
8 Toss coin i independently $\begin{array} { r } { n _ { j } = \left\lceil \frac { 1 } { 2 \alpha _ { j } ^ { 2 } } \ln \frac { 2 d m } { \eta } \right\rceil } \end{array}$ times and let $\hat { p } _ { i } ^ { ( j ) }$ be the empirical mean
9 if U is $2 \alpha _ { j } -$ good for $\hat { p } _ { i } ^ { ( j ) }$ then
10 $y _ { i }  R _ { U } ( \hat { p } _ { i } ^ { ( j ) } )$
11 $A  A \backslash \{ i \}$
12 if A = ∅ then
13 return $( y _ { 1 } , \ldots , y _ { d } )$
14 return ⊥
```

Theorem 6.9. Algorithm 1 has the following properties

$$
\begin{array} { r } { \operatorname* { P r } ( Y = \bot ) \leq \displaystyle \frac { 1 } { 4 } } \\ { \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathbf { p } } \cup \{ \bot \} ) \leq \eta } \end{array}
$$

The expected sample complexity of the algorithm is bounded above by

$$
\mathbb { E } [ N ] \leq M = O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { d } { \eta } \right)
$$

Suppose Algorithm 1 has sample access to d coins with bias vector $\mathbf { p } = ( p _ { 1 } , \ldots , p _ { d } )$ . Further, let $N _ { i }$ denote the number of tosses of coin i and let $\begin{array} { r } { N = \sum _ { i = 1 } ^ { d } N _ { i } } \end{array}$ denote the total number of tosses. Let $Y$ be the random variable denoting the output of the algorithm.

For each coin $i \in [ d ]$ , we will get progressively smaller estimates $\alpha _ { j }$ as the number of rounds $j$ increases. We will get at most dm estimates across the entire algorithm’s execution, m estimates for each of the d coins. So we picked $\begin{array} { r } { n _ { j } = \left\lceil \frac { 1 } { 2 \alpha _ { j } ^ { 2 } } \ln \frac { 2 d m } { \eta } \right\rceil } \end{array}$ so that for the ith coin in jth round the estimate $\hat { p } _ { i } ^ { ( j ) }$ is $\alpha _ { j }$ close to $p _ { i }$ with probability at least $1 - \frac { \eta } { d m }$ . Union bounding across all coins and rounds: every $| \hat { p } _ { i } ^ { ( j ) } - p _ { i } | \leq \alpha _ { j }$ happens with probability at least $1 - \eta$

For simplicity of analysis, we imagine that an independent block of $n _ { j }$ tosses is assigned to every $i \in [ d ]$ and $j \in [ m ]$ , independently of $U ,$ whether or not the algorithm uses it. Let $\hat { p } _ { i } ^ { ( j ) }$ denote the empirical mean of each block, whether or not this was actually used. Only blocks actually used by the algorithm count toward its sample complexity. Formally define the following events

$$
\begin{array} { r l r } { G _ { i } = \{ U \mathrm { ~ i s ~ } 3 \alpha _ { m } \mathrm { - g o o d ~ f o r ~ } p _ { i } \} } & { \quad \mathrm { f o r ~ } i \in [ d ] } \\ { H _ { i , j } = \{ \hat { p } _ { i } ^ { ( j ) } \in [ p _ { i } - \alpha _ { j } , p _ { i } + \alpha _ { j } ] \} } & { \quad \mathrm { f o r ~ } i \in [ d ] , j \in [ m ] } \\ { G = \cap _ { i } G _ { i } } & \\ { H = \cap _ { i , j } H _ { i , j } } & \end{array}
$$

i.e., G is the event that U is $3 \alpha _ { m } -$ good $p _ { i }$ for all i and H is the event that $\hat { p } _ { i } ^ { ( j ) }$ is $\alpha _ { j } .$ -close to $p _ { i }$ for all $i , j$ Lemma 6.10.

1. $\operatorname* { P r } ( G ^ { c } ) \leq { \frac { 1 } { 8 } }$

2. $\Pr ( H ^ { c } ) \leq \eta$

3. $\begin{array} { r } { \operatorname* { P r } ( G ^ { c } \cup H ^ { c } ) \le \frac { 1 } { 8 } + \eta \le \frac { 1 } { 4 } } \end{array}$

Proof.

1. By union bound and Definition 6.5 on $G _ { i } ^ { c } = \{ U$ is not 3α<sub>m</sub>-good for p<sub>i</sub>}

$$
\operatorname* { P r } ( G ^ { c } ) \leq \sum _ { i = 1 } ^ { d } \operatorname* { P r } ( G _ { i } ^ { c } ) = d \cdot \frac { 2 \cdot ( 3 \alpha _ { m } ) } { \varepsilon } = \frac { 3 d } { 2 ^ { m } } \leq \frac { 1 } { 8 } .
$$

The last inequality is obtained by using $m = \lceil \log _ { 2 } ( 2 4 d ) \rceil \geq \log _ { 2 } ( 2 4 d )$

2. By Chernof Bound, for any i, j

$$
\operatorname* { P r } ( H _ { i , j } ^ { c } ) = \operatorname* { P r } ( | \hat { p } _ { i } ^ { ( j ) } - p _ { i } | > \alpha _ { j } ) \leq 2 e ^ { - 2 n _ { j } \alpha _ { j } ^ { 2 } } \leq \frac { \eta } { d m }
$$

By union bound, we have

$$
\mathrm { P r } ( H ^ { c } ) \leq \sum _ { i , j } \mathrm { P r } ( H _ { i , j } ^ { c } ) \leq d m \cdot \frac { \eta } { d m } = \eta
$$

3. By union bound, we have

$$
\operatorname* { P r } ( G ^ { c } \cup H ^ { c } ) \leq { \frac { 1 } { 8 } } + \eta \leq { \frac { 1 } { 4 } }
$$

The last inequality is because we assumed $\eta \leq \frac { 1 } { 8 }$

For a coin $i \in [ d ]$ , let $Z _ { i }$ be the random variable denoting first round j for which $U$ is $2 \alpha _ { j } .$ -good for $\hat { p } _ { i } ^ { ( j ) }$ , if it never happens we let $Z _ { i } = \infty$ . Observe that $Z _ { i } \in \{ 1 , \ldots , m \} \cup \{ \infty \}$ . We say coin i is successful if $Z _ { i } \neq \infty$ Furthermore, for all rounds $j \leq Z _ { i }$ we say that coin i was active, meaning that we had to draw $n _ { j }$ samples for coin i for that round. Note that the set A consists of indices of those coins that are active. If coin i is successful then $y _ { i } = R _ { U } ( \hat { p } _ { i } ^ { ( Z _ { i } ) } )$ . If some coin i was not successful, then $y _ { i } = \perp$ and consequently $Y = \perp$

1. On H, it holds that $Y \in \{ R _ { U } ( { \bf p } ) , \bot \}$ . In particular,

$$
\operatorname* { P r } ( Y \notin { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \} ) \leq \eta
$$

2. On $G \cap H$ , it holds that $Y = R _ { U } ( \mathbf { p } )$ . In particular,

$$
\operatorname* { P r } ( Y = \bot ) \leq \frac { 1 } { 4 }
$$

Proof.

1. First note that unconditionally, for every coin $i \in [ d ] , y _ { i } \in \{ R _ { U } ( \hat { p } _ { i } ^ { ( 1 ) } ) , \ldots , R _ { U } ( \hat { p } _ { i } ^ { ( m ) } ) , \bot \}$ . The exact value of $y _ { i }$ depends on $Z _ { i } \colon { \mathrm { i f ~ } } Z _ { i } \neq \infty$ then $y _ { i } = R _ { U } ( \hat { p } _ { i } ^ { ( Z _ { i } ) } )$ . Even if a single coin is unsuccessful, we trivially have $Y = \perp$ . Thus we assume all coins are successful and we show that $Y = R _ { U } ( \mathbf { p } )$ , i.e., we show that for all i, it holds that $y _ { i } = R _ { U } ( p _ { i } )$ . Using this, we have that for all i, at round $Z _ { i }$ , it holds that U is 2α<sub>Z</sub> -good for $\hat { p } _ { i } ^ { ( Z _ { i } ) }$ and $y _ { i } = R _ { U } ( \hat { p } _ { i } ^ { ( Z _ { i } ) } )$ by construction of our algorithm.

Condition on H and fix a coin i; we have that $| \hat { p } _ { i } ^ { ( j ) } - p _ { i } | \leq \alpha _ { j }$ for all $j \in [ m ]$ . In particular in at round $Z _ { i } ,$ we have $| \hat { p } _ { i } ^ { ( Z _ { i } ) } - p _ { i } | \leq \alpha _ { Z _ { i } }$ . Using item 3 of Definition 6.4, we have that $y _ { i } = R _ { U } ( \hat { p } _ { i } ^ { ( Z _ { i } ) } ) = R _ { U } ( p _ { i } )$ Since this argument holds for all coins, we have $Y = R _ { U } ( \mathbf { p } )$ . If Y is non-⊥ output and was not in $\mathcal { L } _ { p } .$ then some estimate $\hat { p } _ { i } ^ { ( j ) }$ was not $\alpha _ { j }$ -close to $p _ { i }$ , thus we have

$$
\operatorname* { P r } \left( Y \notin { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \} \right) \leq \operatorname* { P r } ( H ^ { c } ) \leq \eta
$$

2. Conditioning on H, yielded $Y \in \{ R _ { U } ( \mathbf { p } ) , \bot \}$ . We show that on $G \cap H$ , it holds that $Y \neq \perp$ , thus yielding that $Y = R _ { U } ( \mathbf { p } )$ . Fix a coin $i \in [ d ]$ and consider the last round m, in this round, because of conditioning on $G$ we have that $U$ is 3α<sub>m</sub>-good for $p _ { i }$ , and because of conditioning on H we have that $| \hat { p } _ { i } ^ { ( m ) } - p _ { i } | \leq \alpha _ { m }$ . Using item 2 of Definition 6.4 we have that U is also $2 \alpha _ { m }$ -good for $\hat { p } _ { i } ^ { ( m ) }$ , which is exactly the definition of being successful at round $m _ { ; }$ thus if coin i wasn’t successful by round $m - 1$ on $G \cap H$ , on round m it becomes successful, i.e., $Z _ { i } \leq m$ , thus y $\neq \bot$ . Since this argument holds for all coins, we have that $Y \neq \perp$ and it holds that $Y = R _ { U } ( { \bf p } ) \in \mathcal { L } _ { \bf p }$ . Thus we have

$$
\operatorname* { P r } ( Y \in { \mathcal { L } } _ { \mathbf { p } } ) \geq \operatorname* { P r } ( Y = R _ { U } ( \mathbf { p } ) ) \geq \operatorname* { P r } ( G \cap H ) \geq { \frac { 3 } { 4 } }
$$

Finally

$$
\operatorname* { P r } ( Y = \bot ) \leq \operatorname* { P r } ( Y \notin { \mathcal { L } } _ { \mathbf { p } } ) \leq { \frac { 1 } { 4 } }
$$

Lemma 6.12. The expected sample complexity is bounded $b y$

$$
\mathbb { E } [ N ] \le M = \frac { 1 5 3 6 d ^ { 2 } } { \varepsilon ^ { 2 } } \ln \frac { 2 d m } \eta = O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { d } \eta \right)
$$

Proof. Fix a coin $i \in [ d ]$ . If coin i is active in round $j \geq 2$ , then it was active in round $j - 1$ and failed the test in that round, since $\hat { p } _ { i } ^ { ( j - 1 ) }$ is independent of $U$ we have

$$
\begin{array} { r l } & { \operatorname* { P r } ( \mathrm { c o i n } \ i \ \mathrm { w a s ~ a c t i v e ~ a t ~ r o u n d } \ j ) } \\ & { \quad \le \operatorname* { P r } ( U \ \mathrm { i s ~ n o t } \ 2 \alpha _ { j - 1 } \mathrm { - g o o d ~ f o r } \ \hat { p } _ { i } ^ { ( j - 1 ) } ) } \\ & { \quad = \displaystyle \frac { 2 \cdot 2 \alpha _ { j - 1 } } { \varepsilon } } \\ & { \quad = \displaystyle \frac { 4 } { 2 ^ { j } } } \end{array}\tag{Definition 6.5}
$$

$$
\begin{array} { r } { ( \alpha _ { j - 1 } = \frac { \varepsilon } { 2 ^ { j } } ) } \end{array}
$$

The upper bound also trivially holds when $j = 1$ . Applying it to the following, we have

$$
\begin{array} { r l r } { \mathbb { E } [ N _ { i } ] = \displaystyle \sum _ { j = 1 } ^ { \infty } n _ { j } \mathbb { P } _ { r } ( \cos \mathrm { i n } \ i \mathrm { ~ w a r k ~ a c t i v e a t ~ o f ~ t o ~ } ) } \\ & { \le \displaystyle \sum _ { j = 1 } ^ { \infty } \left[ \frac { 1 } { 2 \sigma _ { j } ^ { 2 } } \ln \frac { 2 d m } { \eta } \right] \cdot \frac { 4 } { 2 \sigma _ { j } ^ { 2 } } } \\ & { \le \displaystyle \sum _ { j = 1 } ^ { \infty } \left( \frac { 1 } { \alpha _ { j } ^ { 2 } } \ln \frac { 2 d m } { \eta } \right) \cdot \frac { 4 } { 2 \sigma _ { j } ^ { 2 } } } \\ & { = \displaystyle \frac { 1 6 } { \sigma _ { j } ^ { 2 } } \ln \frac { 2 d m } { \eta } \ \frac { \sqrt { \pi } } { \sigma _ { j - 1 } } } \\ & { \le \displaystyle \frac { 3 2 \cdot 2 ^ { \alpha } } { \sigma ^ { 2 } } \ln \frac { 2 d m } { \eta } } \\ & { \le \displaystyle \frac { 1 5 \delta \delta \alpha } { \eta } \ln \frac { 2 d m } { \eta } } \end{array} \qquad \begin{array} { r l } & { ( x \ge 1 \implies [ \pi ] \le 2 \pi ) } \\ & { ( x \ge 1 ) \le [ \pi ] \le 2 \pi } \\ & { ( \alpha _ { j } = \frac { \pi } { 2 ^ { \alpha + 1 } } ) } \end{array}
$$

The last inequality uses $m = \lceil \log _ { 2 } ( 2 4 d ) \rceil \leq \log _ { 2 } ( 2 4 d ) + 1$ . Finally, using linearity of expectation across all coins, we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ N ] = \sum _ { i = 1 } ^ { d } \mathbb { E } [ N _ { i } ] } \\ { \displaystyle \quad \leq \sum _ { i = 1 } ^ { d } \frac { 1 5 3 6 d } { \varepsilon ^ { 2 } } \ln \frac { 2 d m } { \eta } } \\ { \displaystyle \quad = \frac { 1 5 3 6 d ^ { 2 } } { \varepsilon ^ { 2 } } \ln \frac { 2 d m } { \eta } } \\ { \displaystyle = M = { \cal O } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { d } { \eta } \right) } \end{array}
$$

The asymptotic upper bound is obtained by observing that $m = O ( \log d ) \subseteq O ( d )$ and therefore ln $\left( \frac { 2 d m } { \eta } \right) =$ $O \left( \log { \frac { d } { \eta } } \right)$ □

We now conclude the Definition 6.8. We have an algorithm that is $( d + 1 , \varepsilon , O ( 1 ) )$ -list replicable for d-CoinBiasEstimationProblem with expected sample complexity $\begin{array} { r } { \mathbb { E } [ N ] \leq M = \widetilde { O } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ satisfying $\mathrm { P r } ( Y =$ $\textstyle \perp ) \leq { \frac { 1 } { 4 } }$ and $\operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathbf { p } } \mid Y \neq \perp ) = O ( \eta )$ . We can convert the expected sample complexity algorithm into an algorithm with fixed sample complexity by using standard techniques. Capping the sample complexity to $B : = \lceil 4 M \rceil$ increases the probability we see ⊥ by at most $\textstyle { \frac { 1 } { 4 } }$ by Markov’s Inequality. Thus we repeat this capped algorithm $\begin{array} { r } { T = \left\lceil \log _ { 2 } \frac { 2 } { \delta } \right\rceil } \end{array}$ times and return the first non-⊥ output. We see a non-⊥ output with probability at least $1 - \textstyle { \frac { \delta } { 2 } }$ , and pick $\begin{array} { r } { \eta = \frac { \delta } { 4 } } \end{array}$ so that conditioned on seeing a non-⊥ output, the probability that it belongs to ${ \mathcal { L } } _ { \mathbf { p } }$ is at least $1 - \frac { \delta } { 2 } . \mathrm { ~ A ~ }$ more detailed analysis can be found in Appendix A.

## 6.4 Upper Bound for k-List Replicable Algorithms

Theorem 6.13. Let $d , k \in \mathbb { N }$ such that $k \in [ d + 1 , 2 ^ { d } ]$ and $\varepsilon , \delta \in ( 0 , 1 )$ . There exists a $( k , \varepsilon , \delta )$ -list replicable algorithm for the d-CoinBiasEstimationProblem that uses $\begin{array} { r } { N = O \left( \frac { d ^ { 2 } \log d } { \varepsilon ^ { 2 } \log k } \log ^ { 2 } \frac { d } { \delta } \right) } \end{array}$ samples.

Let $\varepsilon \in ( 0 , 1 ) , \delta \in ( 0 , \frac { 1 } { 2 } )$ and let $\ell \in$ N be decided later. Consider the standard partitioning P of [d] into ℓ consecutive blocks $B _ { 1 } , \ldots , B _ { \ell }$ each of size $\begin{array} { r } { d _ { i } : = | B _ { i } | \le \lceil \frac { d } { \rho } \rceil } \end{array}$ . Run algorithm from Definition 6.8 independently for each block of $d _ { i }$ coins with parameters $\begin{array} { r } { \left( d _ { i } + 1 , \varepsilon , \frac { \delta } { \ell } \right) } \end{array}$ . If any block returns ⊥, we return ⊥. Otherwise, we return the concatenation of the outputs of each block.

Let $Y _ { 1 } , \dots , Y _ { \ell }$ be the outputs of each of the blocks. Also, let Y and N be the random variables denoting the output of this algorithm and the sample complexity.

Fix $ { \mathbf { p } } \in [ 0 , 1 ] ^ { d }$ and for $i \in [ \ell ]$ , let $\mathbf { p } ^ { ( i ) } = ( p _ { j } ) _ { j \in B _ { i } } \in [ 0 , 1 ] ^ { d _ { i } }$ denote the bias vector in $B _ { i }$ . The list achieved by this algorithm is

$$
\mathcal { L } _ { \mathbf { p } , \mathcal { P } } = \mathcal { L } _ { \mathbf { p } ^ { ( 1 ) } } \times . . . \times \mathcal { L } _ { \mathbf { p } ^ { ( \ell ) } }
$$

where $\mathcal { L } _ { \mathbf { p } ^ { ( i ) } }$ is the canonical list as defined in Definition 6.6. We identify each element of this Cartesian product with the concatenation of its block vectors in $\mathbb { R } ^ { d }$

Fix $i \in [ \ell ]$ , by Definition 6.7 each element of $\mathcal { L } _ { \mathbf { p } ^ { ( i ) } }$ is still ε-close to $\mathbf { p } ^ { ( i ) }$ and the size of this set is $\begin{array} { r } { | \mathcal { L } _ { \mathbf { p } ^ { ( i ) } } | \leq d _ { i } + 1 \leq \left\lceil \frac { d } { \ell } \right\rceil + 1 } \end{array}$ . Therefore, for each $\mathbf { y } \in \mathcal { L } _ { \mathbf { p } , \mathcal { P } }$ , it holds that $\| \mathbf { y } - \mathbf { p } \| _ { \infty } < \varepsilon$ and the size of $\mathcal { L } _ { \mathbf { p } , \mathcal { P } }$ is bounded above by

$$
| { \mathcal { L } } _ { \mathbf { p } , \mathcal { P } } | \leq \left( \left\lceil \frac { d } { \ell } \right\rceil + 1 \right) ^ { \ell }\tag{1}
$$

The sample complexity is bounded by

$$
\begin{array} { l } { { \displaystyle N = \sum _ { i = 1 } ^ { \ell } { \cal O } \left( \frac { | B _ { i } | ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { | B _ { i } | \ell } { \delta } \log \frac \ell { \delta } \right) } } \\ { ~ = \displaystyle \sum _ { i = 1 } ^ { \ell } { \cal O } \left( \frac { d ^ { 2 } } { \ell ^ { 2 } \varepsilon ^ { 2 } } \log \frac { d } \delta \log \frac \ell { \delta } \right) } \\ { ~ = { \cal O } \left( \frac { d ^ { 2 } } { \ell \varepsilon ^ { 2 } } \log \frac { d } \delta \log \frac \ell { \delta } \right) } \end{array}\tag{2}
$$

By Definition 6.8, we have $\forall i \in [ \ell ] . \operatorname* { P r } ( Y _ { i } \in \mathcal { L } _ { \mathbf { p } ^ { ( i ) } } ) \geq 1 - \frac { \delta } { \ell }$ . Taking a union bound over all ℓ blocks gives us $\operatorname* { P r } ( \forall i \in [ \ell ] . Y _ { i } \in \mathcal { L } _ { \mathbf { p } ^ { ( i ) } } ) \geq 1 - \delta$ . In particular, when $\forall i \in [ \ell ] . Y _ { i } \in \mathcal { L } _ { \mathbf { p } ^ { ( i ) } }$ then $Y \in { \mathcal { L } } _ { \mathbf { p } , { \mathcal { P } } }$ . Therefore

$$
\operatorname* { P r } ( Y \in { \mathcal { L } } _ { \mathbf { p } , { \mathcal { P } } } ) \geq 1 - \delta
$$

Thus we pick $\ell : = \lfloor \log _ { d + 1 } k \rfloor$ so that the list size in Equation (1) simplifies to

$$
\begin{array} { l } { \displaystyle | \mathcal { L } _ { \mathbf { p } , \mathcal { P } } | \leq ( \lceil \frac { d } { \ell } \rceil + 1 ) ^ { \ell }  } \\ { \displaystyle \qquad \leq ( d + 1 ) ^ { \ell } } \\ { \displaystyle \qquad \leq ( d + 1 ) ^ { \log _ { d + 1 } k } } \\ { \displaystyle \qquad \leq k } \end{array}
$$

$$
\textstyle \left( \left\lceil { \frac { d } { \ell } } \right\rceil \leq d \right)
$$

Using $\ell = \left\lfloor \log _ { d + 1 } k \right\rfloor \geq \textstyle { \frac { 1 } { 2 } } \log _ { d + 1 } k$ , the sample complexity in Equation (2) simplifies to

$$
\begin{array} { r l } & { N = O \left( \frac { d ^ { 2 } } { \ell \varepsilon ^ { 2 } } \log \frac { d } { \delta } \log \frac { \ell } { \delta } \right) } \\ & { \quad = O \left( \frac { d ^ { 2 } \log ( d + 1 ) } { \varepsilon ^ { 2 } \log k } \log \frac { d } { \delta } \log \frac { \log k } { \delta } \right) } \\ & { \quad = O \left( \frac { d ^ { 2 } \log d } { \varepsilon ^ { 2 } \log k } \log ^ { 2 } \frac { d } { \delta } \right) } \\ & { \quad = \widetilde O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right) } \end{array}
$$

In the second last step, we assumed $k \leq 2 ^ { d }$ , thus ${ \frac { \log k } { \log ( d + 1 ) } } \leq d .$

This concludes the proof of Definition 6.13.

## 7 Conclusion

Our results establish that the sample complexity of $( d + 1 , \varepsilon , \delta )$ -list replicable algorithms for the d-CoinBiasEstimationProblem is $\begin{array} { r } { \widetilde { \Theta } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ . More generally, we establish a sample complexity bound of $\widetilde { \Theta } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right)$ for $( k , \varepsilon , \delta )$ -list replicable algorithms for the d-CoinBiasEstimationProblem where $k \in [ d + 1 , 2 ^ { d } ]$ . Our upper bounds improve the polynomial dependence on d from cubic to quadratic compared with [DPVWV23, VWDP<sup>+</sup>24]. More generally, our upper and lower bounds determine the sample complexity throughout $k \in [ d + 1 , \mathsf { 2 } ^ { d } ]$ up to polylogarithmic factors.

Open Problems. Section 6 exhibits algorithms whose sample complexity for each coin is adaptive in the sense that the sample complexity of each $N _ { i }$ is not fixed in advance. So the question of whether there exists a non-adaptive algorithm that is $( d + 1 , \varepsilon , \delta )$ -list replicable for the d-CoinBiasEstimationProblem with sample complexity $\begin{array} { r } { \widetilde O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$ remains open.

Acknowledgements. Vinod is supported in part by NSF awards 2342244, 2413848, and a UNL Grand Challenges Grant. Pavan, Michael, and Atieh are supported by NSF awards 2342245 and 2413849.

AI methodology. We used ChatGPT-6 Astra and Google Gemini 3.6. Specifically, the LLMs helped us identify some of the bounds in Section 5.3 and Appendix B, and ideas from LLMs helped us to arrive at the upper results in Section 6. The authors wrote the manuscript with limited assistance from LLMs. We verified the correctness and originality of all content, including the accuracy of the references.

## References

[BGH<sup>+</sup>23] Mark Bun, Marco Gaboardi, Max Hopkins, Russell Impagliazzo, Rex Lei, Toniann Pitassi, Satchit Sivakumar, and Jessica Sorrell. Stability is stable: Connections between replicability, privacy, and adaptive generalization. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, STOC 2023, page 520–527, New York, NY, USA, 2023. Association for Computing Machinery.

[BHH<sup>+</sup>26] Ari Blondal, Hamed Hatami, Pooya Hatami, Chavdar Lalov, and Sivan Tretiak. Borsuk-ulam and replicable learning of large-margin halfspaces. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, STOC ’26, page 529–540, New York, NY, USA, 2026. Association for Computing Machinery.

[BKKS23] Vladimir Braverman, Robert Krauthgamer, Aditya Krishnan, and Shay Sapir. Lower Bounds for Pseudo-Deterministic Counting in a Stream. In Kousha Etessami, Uriel Feige, and Gabriele Puppis, editors, 50th International Colloquium on Automata, Languages, and Programming (ICALP 2023), volume 261 of Leibniz International Proceedings in Informatics (LIPIcs), pages 30:1–30:14, Dagstuhl, Germany, 2023. Schloss Dagstuhl – Leibniz-Zentrum f¨ur Informatik.

[BLM20] Mark Bun, Roi Livni, and Shay Moran. An equivalence between private classification and online prediction. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science (FOCS), pages 389–402, 2020.

[CCPV25] Michael Chen, Sourav Chakraborty, A. Pavan, and N. V. Vinodchandran. Representation obliviousness and pseudodeterminism in streaming algorithms. Proc. ACM Manag. Data, 3(5), November 2025.

[CLO<sup>+</sup>26] Lijie Chen, Zhenjian Lu, Igor Oliveira, Hanlin Ren, and Rahul Santhanam. Polynomial-time pseudodeterministic construction of primes. J. ACM, 73(2), April 2026.

[CMY23] Zachary Chase, Shay Moran, and Amir Yehudayof. Stability and Replicability in Learning . In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 2430–2439, Los Alamitos, CA, USA, November 2023. IEEE Computer Society.

[CPV<sup>+</sup>25] Michael Chen, A. Pavan, N. V. Vinodchandran, Ruosong Wang, and Lin Yang. Regret-optimal list replicable bandit learning: Matching upper and lower bounds. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 5074–5089, 2025.

[DGSB26] Rohan Deb, Udaya Ghai, Karan Singh, and Arindam Banerjee. Replicable bandits with ucb based exploration, 2026.

[DPVWV23] Peter Dixon, A. Pavan, Jason Vander Woude, and N. V. Vinodchandran. List and certificate complexities in replicable learning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 30784–30806. Curran Associates, Inc., 2023.

[DPWV22] Peter Dixon, A. Pavan, Jason Vander Woude, and N. V. Vinodchandran. Pseudodeterminism: promises and lowerbounds. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2022, page 1552–1565, New York, NY, USA, 2022. Association for Computing Machinery.

[EHKS23] Eric Eaton, Marcel Hussing, Michael Kearns, and Jessica Sorrell. Replicable reinforcement learning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 15172–15185. Curran Associates, Inc., 2023.

[EKK<sup>+</sup>23] Hossein Esfandiari, Alkis Kalavasis, Amin Karbasi, Andreas Krause, Vahab Mirrokni, and Grigoris Velegkas. Replicable bandits. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.

[GG11] Erann Gat and Shafi Goldwasser. Probabilistic search algorithms with unique answers and their cryptographic applications. Electron. Colloquium Comput. Complex., TR11, 2011.

[GGMW20] Shafi Goldwasser, Ofer Grossman, Sidhanth Mohanty, and David P. Woodruf. Pseudo-Deterministic Streaming. In Thomas Vidick, editor, 11th Innovations in Theoretical Computer Science Conference (ITCS 2020), volume 151 of Leibniz International Proceedings in Informatics (LIPIcs), pages 79:1–79:25, Dagstuhl, Germany, 2020. Schloss Dagstuhl – Leibniz-Zentrum f¨ur Informatik.

[GGR13] Oded Goldreich, Shafi Goldwasser, and Dana Ron. On the possibilities and limitations of pseudodeterministic algorithms. In Proceedings of the 4th Conference on Innovations in Theoretical Computer Science, ITCS ’13, page 127–138, New York, NY, USA, 2013. Association for Computing Machinery.

[GGS23] Ofer Grossman, Meghal Gupta, and Mark Sellke. Tight space lower bound for pseudodeterministic approximate counting. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 1496–1504, 2023.

[GHR<sup>+</sup>26] Mika G¨o¨os, Nathaniel Harms, Artur Riazanov, Anastasia Sofronova, Dmitry Sokolov, and Weiqiang Yuan. Pseudodeterministic communication complexity. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, STOC ’26, page 2030–2039, New York, NY, USA, 2026. Association for Computing Machinery.

[GL19] Ofer Grossman and Yang P. Liu. Reproducibility and pseudo-determinism in log-space. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 606–620, 2019.

[Gol25] Oded Goldreich. Multi-pseudodeterministic Algorithms, pages 22–43. Springer Nature Switzerland, Cham, 2025.

[HIK<sup>+</sup>24] Max Hopkins, Russell Impagliazzo, Daniel Kane, Sihan Liu, and Christopher Ye. Replicability in High Dimensional Statistics . In 2024 IEEE 65th Annual Symposium on Foundations of Computer Science (FOCS), pages 1–8, Los Alamitos, CA, USA, October 2024. IEEE Computer Society.

[HM25] Max Hopkins and Shay Moran. The role of randomness in stability. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 23805–23827. PMLR, 13–19 Jul 2025.

[ILPS22] Russell Impagliazzo, Rex Lei, Toniann Pitassi, and Jessica Sorrell. Reproducibility in learning. In Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing, STOC 2022, page 818–831, New York, NY, USA, 2022. Association for Computing Machinery.

[KCG16] Emilie Kaufmann, Olivier Capp´e, and Aur´elien Garivier. On the complexity of best-arm identification in multi-armed bandit models. J. Mach. Learn. Res., 17(1):1–42, January 2016.

[LOS21] Zhenjian Lu, Igor C. Oliveira, and Rahul Santhanam. Pseudodeterministic algorithms and the structure of probabilistic time. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, STOC 2021, page 303–316, New York, NY, USA, 2021. Association for Computing Machinery.

[LSS14] Adeline Langlois, Damien Stehle, and Ron Steinfeld. GGHLite: More eficient multilinear maps from ideal lattices. Cryptology ePrint Archive, Paper 2014/487, 2014.

[Mah26] Aditya Mahajan. Change of measure. https://adityam.github.io/stochastic-control/ probability/change-of-measure.html, 2026.

[Mir17] Ilya Mironov. R´enyi Diferential Privacy . In 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pages 263–275, Los Alamitos, CA, USA, August 2017. IEEE Computer Society.

[Sca24] Jonathan Scarlett. An introduction to statistical lower bounds for estimation and learning. https://www.comp.nus.edu.sg/ scarlett/it\_data\_slides/02-OtherMethods.pdf, 2024.

[VWDP<sup>+</sup>24] Jason Vander Woude, Peter Dixon, A. Pavan, Jamie Radclife, and N. V. Vinodchandran. Replicability in learning: Geometric partitions and kkm-sperner lemma. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 78996–79028. Curran Associates, Inc., 2024.

[ZCP<sup>+</sup>25] Bohan Zhang, Michael Chen, A. Pavan, N. V. Vinodchandran, Lin F. Yang, and Ruosong Wang. List replicable reinforcement learning. CoRR, abs/2512.00553, 2025.

## A Upper Bound with Fixed Sample Complexity

## A.1 Capping the Sample Complexity

Let ε, η be the same parameters as in Algorithm 1. Define the parameters

$$
\begin{array} { l } { { M = { \displaystyle \frac { 1 5 3 6 d ^ { 2 } } { \varepsilon ^ { 2 } } } \ln { \frac { 2 d m } { \eta } } } } \\ { { B = \lceil 4 M \rceil } } \end{array}
$$

where $m = \lceil \log _ { 2 } ( 2 4 d ) \rceil$ and M is the same upper bound on the expected number of samples from Algorithm 1.

```latex
Algorithm 2: $( d + 1 , \varepsilon , O ( 1 ) )$ -List Replicable Algorithm with sample complexity $\begin{array} { r } { N = \widetilde { O } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$
1 Run Algorithm 1 with a cap of $B = \lceil 4 M \rceil$ samples;
2 if Algorithm 1 needs more than B samples then
3 return ⊥
4 else
5 return Output of the Algorithm 1
```

Theorem A.1. Algorithm 2 has the following properties

$$
\begin{array} { r } { \operatorname* { P r } ( Y = \bot ) \leq \displaystyle \frac { 1 } { 2 } } \\ { \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathbf { p } } \cup \{ \bot \} ) \leq \eta } \end{array}
$$

The sample complexity of the algorithm is bounded above by

$$
N \leq B = \lceil 4 M \rceil = O \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \frac { d } { \eta } \right)
$$

For Algorithm 2, let N be the random variable denoting the total number of samples and $Y$ be the random variable denoting the output.

Proof. Imagine running both algorithms with the same shift and samples. Their outputs agree whenever the uncapped algorithm uses at most B samples. By Definition 6.12 and applying Markov’s Inequality, we get

$$
\operatorname* { P r } ( { \mathrm { A l g o r i t h m ~ 1 ~ r e q u i r e s ~ m o r e ~ t h a n ~ } } B { \mathrm { ~ s a m p l e s } } ) \leq { \frac { 1 } { 4 } }
$$

The probability that Algorithm 1 outputs $\perp$ is at most ${ \begin{array} { l } { { \frac { 1 } { 4 } } , } \end{array} }$ , the probability $Y = \perp$ increases by at most $\textstyle { \frac { 1 } { 4 } }$ when the algorithm requires more than B samples. Thus we have

$$
\operatorname* { P r } ( Y = \bot ) \leq \frac { 1 } { 4 } + \frac { 1 } { 4 } \leq \frac { 1 } { 2 }
$$

Let N be the sample complexity of Algorithm 2. We cap the algorithm’s sample complexity bound at $\lceil 4 M \rceil$ so trivially $N \leq \lceil 4 M \rceil$ always holds. This cap can only change outputs from $\mathbb { R } ^ { d }$ to ⊥, but not the other way around. So the following still holds

$$
\operatorname* { P r } ( Y \notin { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \} ) \leq \eta
$$

## A.2 Boosting the Success Probability

Define the parameters

$$
\begin{array} { r l } & { \varepsilon \in \left( 0 , 1 \right) } \\ & { \delta \in \left( 0 , 1 / 2 \right) } \\ & { T = \left\lceil \log _ { 2 } \frac { 2 } { \delta } \right\rceil } \\ & { \eta = \frac { \delta } { 4 } } \end{array}
$$

So that $\eta \in ( 0 , 1 / 8 ]$ . A single execution of Algorithm 2 may return ⊥, so we run the same algorithm independently up to $T$ times till it produces an element of $\mathbb { R } ^ { d }$ . We return the first non-⊥ output, and return ⊥ only if all $T$ attempts do so.

Algorithm 3 outputs ⊥ only if all T instances output ⊥, which happens with probability at most $2 ^ { - T } \leq$ $\delta / 2$ . Each instance returns a valid member of $\mathbb { R } ^ { d }$ with probability at least $1 / 2$ and returns a vector outside ${ \mathcal { L } } _ { \mathbf { p } }$ with probability at most η. Thus, among its non-⊥ outputs, the probability of being outside the list is at most $2 \eta = \delta / 2$ . Therefore, the probability of returning ⊥ or a vector outside the list is at most δ. We now prove the correctness and sample complexity bound of this algorithm formally.

```latex
Algorithm 3: $( d + 1 , \varepsilon , \delta )$ -List Replicable Algorithm with sample complexity $\begin{array} { r } { N = \widetilde { O } \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right) } \end{array}$
1 for $t = 1 , \dots , T$ do
2 Run an i.i.d. instance of Algorithm 2 with ε, $\begin{array} { r } { \eta = \frac { \delta } { 4 } } \end{array}$
3 Let $Y _ { t }$ be its output
4 if $Y _ { t } \neq \perp$ then
5 return $Y _ { t }$
6 return ⊥
```

Theorem A.2. Algorithm 3 is $( d + 1 , \varepsilon , \delta )$ -list replicable with list $\mathcal { L } _ { \mathbf { p } }$ , i.e.,

$$
\operatorname* { P r } ( Y \in { \mathcal { L } } _ { \mathbf { p } } ) \geq 1 - \delta
$$

where $\mathcal { L } _ { \mathbf { p } } = \{ R _ { u } ( \mathbf { p } ) \mid u \in [ 0 , \varepsilon ) \}$ satisfies $| { \mathcal { L } } _ { \bf p } | \leq d + 1$ and each $\mathbf { y } \in \mathcal { L } _ { \mathbf { p } }$ satisfies $\| \mathbf { y } - \mathbf { p } \| _ { \infty } < \varepsilon$

The sample complexity of the algorithm is bounded above by

$$
N \leq B T = O \left( { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } } \log { \frac { d } { \delta } } \log { \frac { 1 } { \delta } } \right)
$$

Proof. The two properties of ${ \mathcal { L } } _ { \mathbf { p } }$ follow directly from Definition 6.7. We prove the remaining properties. For simplicity of analysis, we imagine generating all T independent instances $Y _ { 1 } , \dots , Y _ { T }$ , the algorithm returns the first non-⊥ output.

Let K be the random variable denoting the first index t for which $Y _ { t } \neq \perp$ ; if it doesn’t exist, then $K = \infty$ By using properties of Algorithm 2, note that for any $t \in [ T ]$

$$
{ \begin{array} { r l r } { \operatorname* { P r } ( Y \not \in { \mathcal { L } } _ { \mathbf { p } } \mid K = t ) = \operatorname* { P r } ( Y _ { t } \not \in { \mathcal { L } } _ { \mathbf { p } } \mid Y _ { t } \not = \bot ) } \\ & { \qquad = { \frac { \operatorname* { P r } ( Y _ { t } \not \in { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \} ) } { \operatorname* { P r } ( Y _ { t } \not \subset \bot ) } } } \\ & { \qquad \leq 2 \operatorname* { P r } ( Y _ { t } \not \in { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \} ) } \\ & { \qquad \leq 2 \eta } & { ( \operatorname* { P r } ( Y _ { t } \not \in { \mathcal { L } } _ { \mathbf { p } } \cup \{ \bot \} ) \leq \eta ) } \\ & { \qquad \leq { \frac { \delta } { 2 } } } \end{array} }
$$

And by independence we have

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } ( K = \infty ) = \operatorname* { P r } ( Y _ { 1 } = \bot , \dots , Y _ { T } = \bot ) } \\ { \displaystyle \qquad = \prod _ { t = 1 } ^ { T } \operatorname* { P r } ( Y _ { t } = \bot ) } \\ { \displaystyle \qquad \leq \left( \frac { 1 } { 2 } \right) ^ { T } } \\ { \displaystyle \qquad \leq \frac { \delta } { 2 } } \end{array}
$$

And finally

$$
\begin{array} { r l } & { \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathtt { P } } ) = \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathtt { P } } , K = \infty ) + \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathtt { P } } , K = t ) } \\ & { \quad = \operatorname* { P r } ( K = \infty ) + \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { P r } ( Y \notin \mathcal { L } _ { \mathtt { P } } \mid K = t ) \operatorname* { P r } ( K = t ) } \\ & { \quad \le \operatorname* { P r } ( K = \infty ) + \displaystyle \frac { \delta } { 2 } \sum _ { t = 1 } ^ { T } \operatorname* { P r } ( K = t ) } \\ & { \quad \le \frac { \delta } { 2 } + \frac { \delta } { 2 } \cdot 1 } \\ & { \quad \le \delta } \end{array}
$$

Each attempt uses at most $B = \lceil 4 M \rceil$ tosses, and there are at most $T$ attempts. Substituting $\begin{array} { r } { \eta = \frac { \delta } { 4 } } \end{array}$ in the expression of $B _ { : }$ the total sample count is bounded by

$$
\begin{array} { c } { { N \le B T } } \\ { { = O \left( \displaystyle \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \displaystyle \frac { 4 d } { \delta } \right) \left. \log _ { 2 } \displaystyle \frac { 2 } { \delta } \right. } } \\ { { = O \left( \displaystyle \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \log \displaystyle \frac { d } { \delta } \log \displaystyle \frac { 1 } { \delta } \right) } } \end{array}
$$

## B General Lower Bound

In this section, we prove the lower bound result for arbitrary algorithms.

Theorem B.1 (Definition 5.7 Restated). Let $\varepsilon , \delta \in ( 0 , 1 / 8 )$ . For suficiently large $d \in \mathbb { N }$ N, every $( d + 1 , \varepsilon , \delta )$ list replicable algorithm for the d-CoinBiasEstimationProblem requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log d } \right) } \end{array}$ samples.

Let A be a randomized sampling-based algorithm that draws N samples from d coins with unknown bias and is $( d + 1 , \varepsilon , \delta )$ -list replicable. The algorithm may choose which coin to toss based on its internal randomness and previous outcomes. Let Y be the random variable denoting the algorithm’s output, which belongs to the set Y. Let W be the random variable denoting the algorithm’s internal randomness. Let $H _ { 0 } = W$ and for $t \in [ N ]$ we define the following random variables

$$
{ \begin{array} { r l } & { I _ { t } = \operatorname { C o i n \ s e l e c t e d \ a t \ t i m e } t } \\ & { X _ { t } = \operatorname { O u t c o m e \ o f \ t h e \ t o s s \ a t \ t i m e } t } \\ & { H _ { t } = ( W , I _ { 1 } , X _ { 1 } , \ldots , I _ { t } , X _ { t } ) } \end{array} }
$$

where $H _ { t }$ denotes the history till time t, the choice $I _ { t }$ only depends on $H _ { t - 1 }$

For a bias vector $\mathbf { p } \in [ 0 , 1 ] ^ { d }$ , along with the distribution on W (which is independent of each coin), we let $P$ denote the probability distribution on the full transcript $H _ { N } = ( W , I _ { 1 } , X _ { 1 } , \ldots , I _ { N } , X _ { N } )$ when the coins have bias p. Furthermore, for $t \in [ N ]$ we let $P _ { t }$ be distribution on $H _ { t } = ( W , I _ { 1 } , X _ { 1 } , \dots , I _ { t } , X _ { t } )$ when the coins have bias $\mathbf { p } ,$ noting that $P _ { N } = P$

Similarly, for $\mathbf { q } \in [ 0 , 1 ] ^ { d }$ we let $Q$ denote the distribution on the full transcript induced when the coins have bias q and for $t \in [ N ]$ , we let $Q _ { t }$ be the distribution on $H _ { t } = ( W , I _ { 1 } , X _ { 1 } , \dots , I _ { t } , X _ { t } )$ when the coins have bias q.

Because A is $( d + 1 , \varepsilon , \delta )$ -list replicable, for each $ { \mathbf { p } } \in [ 0 , 1 ] ^ { d }$ there exists a set $L _ { \mathbf { p } } \subseteq \mathcal { V }$ of size $| L _ { \mathbf { p } } | \leq d + 1$ such that each $\mathbf { y } \in L _ { \mathbf { p } }$ it holds that $\| \mathbf { p } - \mathbf { y } \| _ { \infty } \leq \varepsilon$ and

$$
\operatorname* { P r } ( Y \in L _ { \mathbf { p } } ) \geq 1 - \delta
$$

Lemma B.2. Let $\begin{array} { r } { \mathbf { p } , \mathbf { q } \in [ 0 , 1 ] ^ { d } . \ I f \ \frac { 1 } { 4 } \leq \mathbf { p } \leq \frac { 3 } { 4 } \ t h e n \mathbb { E } _ { P } \left[ \left( \frac { d Q } { d P } \right) ^ { 2 } \right] \leq e ^ { \frac { 1 6 } { 3 } N \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } } } \end{array}$

Proof. Define $\Lambda _ { 0 } \equiv 1$ and, for $t \in [ N ]$ , define

$$
\Lambda _ { t } = \frac { d Q _ { t } } { d P _ { t } }
$$

and let $\pi _ { t } ( i \mid H _ { t - 1 } )$ denote the algorithm’s probability of choosing coin $i \in [ d ]$ at time t, given the history $H _ { t - 1 }$ up until that point. Conditioned on $H _ { t - 1 }$ , both $I _ { t }$ and $\Lambda _ { t - 1 } ( H _ { t - 1 } )$ are fixed. Since the algorithm uses the same policy $\pi _ { t }$ , we have

$$
\operatorname* { P r } _ { P } ( I _ { t } = i \mid H _ { t - 1 } ) = \pi _ { t } ( i \mid H _ { t - 1 } ) = \operatorname* { P r } _ { Q } ( I _ { t } = i \mid H _ { t - 1 } )
$$

Because by definition, the algorithm’s tosses can only depend on what it has seen so far. For any $t \in [ N ]$ we have

$$
\begin{array} { r l } & { \Lambda _ { t } ( H _ { t } ) = \Lambda _ { t - 1 } ( H _ { t - 1 } ) \frac { \pi _ { t } ( I _ { t } \mid H _ { t - 1 } ) \cdot q _ { I _ { t } } ^ { X _ { t } } ( 1 - q _ { I _ { t } } ) ^ { 1 - X _ { t } } } { \pi _ { t } ( I _ { t } \mid H _ { t - 1 } ) \cdot p _ { I _ { t } } ^ { X _ { t } } ( 1 - p _ { I _ { t } } ) ^ { 1 - X _ { t } } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \frac { q _ { I _ { t } } ^ { X _ { t } } ( 1 - q _ { I _ { t } } ) ^ { 1 - X _ { t } } } { p _ { I _ { t } } ^ { X _ { t } } ( 1 - p _ { I _ { t } } ) ^ { 1 - X _ { t } } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end \end{array}
$$

Observe the recurrence

$$
\begin{array} { r l } & { \mathbb { E } _ { P } [ \Lambda _ { t } ^ { 2 } ( H _ { t } ) \mid H _ { t - 1 } ] } \\ & { \quad = p _ { I _ { t } } \cdot \left( \Lambda _ { t - 1 } ( H _ { t - 1 } ) \frac { q _ { I _ { t } } } { p _ { I _ { t } } } \right) ^ { 2 } + ( 1 - p _ { I _ { t } } ) \cdot \left( \Lambda _ { t - 1 } ( H _ { t - 1 } ) \frac { 1 - q _ { I _ { t } } } { 1 - p _ { I _ { t } } } \right) ^ { 2 } } \\ & { \quad = \Lambda _ { t - 1 } ^ { 2 } ( H _ { t - 1 } ) \left( \frac { q _ { I _ { t } } ^ { 2 } } { p _ { I _ { t } } } + \frac { ( 1 - q _ { I _ { t } } ) ^ { 2 } } { 1 - p _ { I _ { t } } } \right) } \\ & { \quad = \Lambda _ { t - 1 } ^ { 2 } ( H _ { t - 1 } ) \left( 1 + \frac { ( p _ { I _ { t } } - q _ { I _ { t } } ) ^ { 2 } } { p _ { I _ { t } } ( 1 - p _ { I _ { t } } ) } \right) } \\ & { \quad \le \Lambda _ { t - 1 } ^ { 2 } ( H _ { t - 1 } ) e ^ { \frac { 1 6 } { 3 } \| { \bf p } - { \bf q } \| _ { \infty } ^ { 2 } } } \end{array}
$$

The last three steps can be obtained by using a similar analysis to Definition 5.4. Finally, averaging over all possible histories and using the law of total expectation, we have

$$
\begin{array} { r l } & { \mathbb { E } _ { P } [ \Lambda _ { t } ^ { 2 } ( H _ { t } ) ] = \mathbb { E } _ { P } [ \mathbb { E } _ { P } [ \Lambda _ { t } ^ { 2 } ( H _ { t } ) \mid H _ { t - 1 } ] ] } \\ & { \qquad \le \mathbb { E } _ { P } [ \Lambda _ { t - 1 } ^ { 2 } ( H _ { t - 1 } ) ] e ^ { \frac { 1 6 } { 3 } \| { \bf p - q } \| _ { \infty } ^ { 2 } } } \end{array}
$$

A routine induction on t shows that

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { P } \left[ \left( \frac { d Q } { d P } \right) ^ { 2 } \right] = \mathbb { E } _ { P } [ \Lambda _ { N } ^ { 2 } ( H _ { N } ) ] } \\ { \displaystyle \quad \le \mathbb { E } _ { P } [ \Lambda _ { 0 } ^ { 2 } ( H _ { 0 } ) ] \prod _ { i = 1 } ^ { N } e ^ { \frac { 1 6 } { 3 } \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } } } \\ { \displaystyle \quad = 1 \cdot e ^ { \sum _ { i = 1 } ^ { N } \frac { 1 6 } { 3 } \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } } } \\ { \displaystyle \quad = e ^ { \frac { 1 6 } { 3 } N \| \mathbf { p } - \mathbf { q } \| _ { \infty } ^ { 2 } } } \end{array}
$$

The remaining analysis is identical to Section 5.4. Use A to color $\begin{array} { r } { \mathcal { R } = \left[ \frac { 1 } { 2 } - 2 \varepsilon , \frac { 1 } { 2 } + 2 \varepsilon \right] ^ { d } \subseteq \left[ \frac { 1 } { 4 } , \frac { 3 } { 4 } \right] ^ { d } } \end{array}$ with the most likely output, i.e., define the coloring $\chi \colon { \mathcal { R } }  { \mathcal { V } }$ as

$$
\chi ( \mathbf { p } ) = \underset { \mathbf { y } \in L _ { \mathbf { p } } } { \arg \operatorname* { m a x } } \operatorname* { P r } _ { P } ( Y = \mathbf { y } )
$$

where ties are broken according to a fixed rule. Since $\chi$ is a valid Sperner/KKM coloring, the Neighborhood Sperner/KKM, Definition 5.2, applies. Similar to the analysis from Definition 5.10, we get

$$
\frac { 1 6 } { 3 } N r ^ { 2 } \geq \frac { d r } { 1 0 \varepsilon } - 2 \ln ( 2 ( d + 1 ) )
$$

Picking $\begin{array} { r } { r = \frac { 4 0 \varepsilon } { d } \ln ( 2 ( d + 1 ) ) , } \end{array}$ ), for suficiently large $d ,$ we again get

$$
N \geq { \frac { 3 } { 1 2 8 0 0 } } \cdot { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \ln ( 2 ( d + 1 ) ) } } = \Omega \left( { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log d } } \right)
$$

We can again generalize this result to lists of size k for arbitrary randomized algorithms.

Theorem B.3 (Definition 5.11 Restated). Let $\varepsilon , \delta \in ( 0 , 1 / 8 )$ and $d , k \in \mathbb { N }$ . If $k \in [ d + 1 , 2 ^ { d } ]$ then every $( k , \varepsilon , \delta )$ -list replicable algorithm for the d-CoinBiasEstimationProblem requires $\begin{array} { r } { N = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right) } \end{array}$ samples.

The argument is the same; the main bound that changes is

$$
\operatorname* { P r } _ { P } ( Y = \mathbf { y _ { p } } ) \geq { \frac { 1 - \delta } { | L _ { \mathbf { p } } | } } \geq { \frac { 7 } { 8 k } } \geq { \frac { 1 } { 2 k } } = : \beta
$$

Definition 5.10 now becomes

$$
\frac { 1 6 } { 3 } N r ^ { 2 } \geq \frac { d r } { 1 0 \varepsilon } - 2 \ln ( 2 k )
$$

If $k \leq \textstyle { \frac { 1 } { 2 } } e ^ { \frac { d } { 2 0 } }$ then our choice $\begin{array} { r } { r : = \frac { 4 0 \varepsilon } { d } \ln ( 2 k ) \le 2 \varepsilon } \end{array}$ holds and therefore

$$
N \geq \frac { 3 } { 1 2 8 0 0 } \cdot \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \ln ( 2 k ) } = \Omega \left( \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } \right)
$$

For $\begin{array} { r } { k > \frac { 1 } { 2 } e ^ { \frac { d } { 2 0 } } } \end{array}$ , every $( k , \varepsilon , \delta )$ -list replicable algorithm also solves the $( d , \varepsilon , \delta )$ -CoinBiasEstimationProblem. Using Definition 3.2 gives

$$
N = \Omega \left( { \frac { d } { \varepsilon ^ { 2 } } } \right) = \Omega \left( { \frac { d ^ { 2 } } { \varepsilon ^ { 2 } \log k } } \right)
$$

The last bound was by using the fact that $k > \frac { 1 } { 2 } e ^ { \frac { d } { 2 0 } }$