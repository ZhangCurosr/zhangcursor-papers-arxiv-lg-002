# The Cost of Working Memory in Streaming Mean Estimation

Zeyu Wu Zihao Wang Minghua Deng<sup>∗</sup>

School of Mathematical Sciences, Peking University

## Abstract

A streaming estimator must compute from the evidence that fits in memory. We determine when storing an accurate mean is easier than computing it. For Gaussian mean estimation over the Euclidean unit ball in $\mathbb { R } ^ { d } .$ , we establish the minimax rate under squared error with K persistent states, a known horizon, and a public time index:

min 1, K−<sup>2/d</sup> + <sup>d</sup>n 1 + log(ed)

Unrestricted computation followed by K-level encoding has rate min $\{ 1 , d / n + K ^ { - 2 / d } \}$ . At a fixed suficiently small error, statistically optimal sample size therefore requires order d log d working bits, while order d bits sufice to store the answer. The lower bound tracks repeated state compression through an entropy potential obtained by Gaussian smoothing. A matching construction combines a multiscale modular counter with workspace reused across coordinate groups. In one dimension, it attains the terminal rate $n ^ { - 1 } + K ^ { - 2 }$ . Any fixed confidence requirement that the realized loss not exceed the zero baseline preserves these rates. Increasing confidence creates a further memory requirement: at fixed nontrivial risk in every fixed dimension, failure probability $e ^ { - \gamma n }$ requires and permits order log n working states for suficiently small fixed $\gamma > 0$ . In the scalar problem, a fixed terminal alphabet sufices under the same protection. These bounds separate the memory needed to represent an answer from the memory needed to accumulate evidence and validate a decision.

Keywords: inference with finite memory; Gaussian location model; minimax risk; quantization;   
tradeof between sample size and memory.

## 1 Introduction

Streaming estimation supports decisions from continuously arriving data, as in quantitative trading, where online algorithms update statistics of liquidity and volatility (Loveless et al., 2013). A related need to process long streams with bounded retained information arises in language model inference, where the attention cache must remain manageable as the sequence grows (Xiao et al., 2024). A streaming estimator must extract information before each observation is discarded. Its retained state is the evidence available for the next update, so a memory constraint can change statistical accuracy as well as storage cost. A running average appears to need just one vector, but this description counts real numbers rather than bits. With a finite bit budget, the same state must represent the current estimate and retain enough evidence to improve it. The space needed for an accurate final answer may therefore understate the space needed to compute that answer.

We ask when answer storage sufices for statistically eficient computation. We study independent observations $X _ { t } \sim N _ { d } ( \mu , I _ { d } )$ with $\| \mu \| \leq 1$ , a known sample size n, and at most $K = 2 ^ { b }$ persistent states at each time. The index t is public, allowing scheduled measurements without a stored clock. All information obtained from the data passes through the current state. Comparing this model with unrestricted computation followed by a K-level encoding isolates the statistical cost of working memory.

The main result separates that cost from sampling and answer representation:

$$
\underbrace { K ^ { - 2 / d } } _ { \mathrm { a n s w e r \ r e p r e s e n t a t i o n } } + \underbrace { \frac { d } { n } } _ { \mathrm { s a m p l i n g } } + \underbrace { \frac { d ^ { 2 } \log ( e d ) } { n ( d + \log K ) } } _ { \mathrm { w o r k i n g \ m e m o r y } } ,\tag{1.1}
$$

with the total truncated at a constant. The last term is necessary for randomized procedures with arbitrary joint states and transitions that vary with time. A deterministic construction attains the same rate. Thus the memory budget determines how eficiently the stream can be converted into an estimate, even when the final answer already fits in that budget.

At a fixed suficiently small target risk, the diference is an order log d factor in bits: order d bits represent an adequate answer, whereas achieving that risk with order d samples requires order d log d working bits. Corollary 3.3 gives the precise relationship between sample size and memory. The separation also persists when accuracy improves with dimension: Corollary 3.4 gives sequences for which both risks vanish but their ratio diverges. Working memory is therefore a statistical resource beyond the precision of the output.

The matching bounds explain how this resource is used. In one dimension, a modular counter measures the mean at decreasing frequencies. Its counter shrinks as the estimate label grows, keeping their joint state space within budget and attaining $n ^ { - 1 } + K ^ { - 2 }$ . In higher dimensions, completed coordinate estimates are encoded jointly while the workspace is reused for the next group. The lower bound tracks the information lost at each state update through an entropy potential obtained by Gaussian smoothing. Its cumulative accounting yields the last term of (1.1) for arbitrary joint states, matching the workspace allocation in the construction.

Confidence introduces a second reason to retain working memory. We require the terminal estimate to have loss greater than that of zero with probability at most δ, uniformly over the mean. Fixed δ changes the rates only by constants. As δ decreases, validating a candidate introduces an asymmetric testing problem with its own tradeof between sample size and the number of states. At fixed nontrivial risk, failure probability $e ^ { - \gamma n }$ requires and permits order log n states, or $\log _ { 2 } \log n + O ( 1 )$ bits, for suficiently small fixed $\gamma > 0$ . We establish this law first in one dimension and then extend it to every fixed dimension by reusing the validation workspace across coordinates. Sections 3–5 develop the estimation cost; Section 6 explains the additional cost of confidence.

## 1.1 Related work

Inference with finite memory depends on both the transition model and the sample size. Small machines whose transitions vary with time can accumulate arbitrarily reliable testing evidence (Cover, 1969), and mean estimators with transitions that vary over time achieve prescribed asymptotic accu racy for bounded observations (Wagner, 1972). For Gaussian observations, machines with stationary transitions can approach the risk of an optimal quantizer with finitely many levels as the sample size tends to infinity (Hellman, 1974). Optimality at finite sample sizes was also studied for binary Bayes testing between two known distributions (Cover et al., 1976). Our scalar result gives a deterministic construction that attains min $\{ 1 , n ^ { - 1 } + K ^ { - 2 } \}$ jointly in sample size and state budget, under a public clock and fixed horizon. Deterministic Bernoulli estimation with stationary transitions has asymptotic risk of order $K ^ { - 1 }$ (Berg et al., 2021); these modeling distinctions are surveyed in Berg et al. (2024).

The distinction between the memory needed to represent a solution and the memory needed to learn it also motivates frameworks connecting learning with limited memory to statistical queries (Steinhardt et al., 2016). Lower bounds for protocols with information constraints, including online algorithms with bounded memory, establish separations between sample size and memory for detecting biased coordinates and estimating sparse covariance (Shamir, 2014). We resolve this tradeof for the Gaussian location family through matching bounds in $n , d , K$ , measured against a benchmark based on terminal encoding.

Quantized estimation measures the accuracy lost when an estimate must have a short description. Sharp asymptotic tradeofs are available for Gaussian sequence models over Euclidean balls (Zhu and Laferty, 2014). Our terminal benchmark permits the same access to all observations before encoding; the additional term in (1.1) measures the cost of imposing the alphabet bound at every update.

Estimation under communication constraints imposes a diferent restriction: each observation or device sends a limited message, while a central estimator may retain the transcript. Identities for conditional scores and Fisher information give powerful lower bounds in this setting (Barnes et al., 2020; Barnes and ${ \ddot { \mathrm { O z g i r } } } ,$ , 2021; Acharya et al., 2022). The sign reduction from a Gaussian location observation to a Bernoulli observation is also used in that literature (Acharya et al., 2022). Our lower bound must additionally account for recompression of the old state after every new observation. It does so by telescoping over time an entropy potential obtained by Gaussian smoothing. Gaussian smoothing and its properties in information theory have a broader history (Guo et al., 2005; Polyan skiy and Wu, 2016); the role of smoothing here is to charge the loss caused by repeated compression by conditional expectations.

Streaming estimation in high dimensions is itself well established. Robust mean estimation admits computationally eficient algorithms that use a single pass with space nearly linear in dimension, up to logarithmic factors (Diakonikolas et al., 2022). Our clean Gaussian experiment instead resolves the risk as a function of an arbitrary total state budget, including the logarithmic cost of working memory at statistically optimal sample size. Sample lower bounds under finite memory have also been developed for noiseless Gaussian regression (OpenAI, 2026). That experiment supplies exact linear equations for an unknown direction. The noisy location experiment studied here instead requires averaging independent perturbations, and its cost of working memory is measured against a terminal quantization benchmark. The upper bound uses a scalar procedure with finitely many states and encodes its outputs jointly inside a Euclidean ball. Its modular measurement also has a conceptual relative in modulo quantization with side information (Mayekar et al., 2021); here the side information is an evolving phase label and both the label and counter are charged to the same state budget.

Confidence guarantees for realized loss improvement provide comparisons against an incumbent estimator (Trippe et al., 2024). Taking zero as the incumbent, we determine how baseline protection changes the sample and working memory requirements at fixed target risk.

## 2 Statistical experiment and memory model

Let $B _ { 2 } ^ { d } = \{ \mu \in \mathbb { R } ^ { d } : \| \mu \| \leq 1 \}$ , where $\| \cdot \|$ is Euclidean norm. We observe $X _ { 1 } , \ldots , X _ { n }$ independently from $N _ { d } ( \mu , I _ { d } )$ and evaluate an action $w \in \mathbb { R } ^ { d }$ by

$$
\begin{array} { r } { R _ { \mu } ( w ) = \frac { 1 } { 2 } \| w - \mu \| ^ { 2 } . } \end{array}
$$

The zero action has worst-case loss $1 / 2$

Streaming procedures. A procedure starts in a state $S _ { 0 }$ whose law is independent of the parameter and updates

$$
S _ { t } \sim Q _ { t } ( \cdot \mid S _ { t - 1 } , X _ { t } ) , \qquad S _ { t } \in \{ 1 , \ldots , K \} .\tag{2.1}
$$

The final estimate is decoded from $S _ { n }$ , possibly using fresh independent randomness. The kernels may depend on the public quantities $n , d , K , \delta , t .$ , but not on $\mu .$ . They can involve arbitrary measurable computation within a step. All information retained between observations is part of $S _ { t }$ . This includes any counter, scale, label, or random seed that depends on the data. Fresh independent randomization is allowed without storing a seed. A complete vector $X _ { t }$ arrives at each step, and $S _ { t }$ is the sole record carried to the next observation.

The time index and fixed program are free. Predetermined block and layer boundaries are therefore recognized without storing a clock. Within each block, observations are processed individually and discarded. The resource measured here is persistent memory, with unrestricted computation within each update, including codebook operations.

Baseline protection. A procedure is uniformly δ-safe if

$$
\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { P } _ { \mu } \{ R _ { \mu } ( \widehat { w } ) > R _ { \mu } ( 0 ) \} \le \delta .\tag{2.2}
$$

This guarantee controls the realized loss relative to zero, while the minimax criterion below averages the loss. At $\mu = 0$ , safety requires returning zero with probability at least $1 - \delta$

Let $\mathcal { A } _ { n , K , \delta } ^ { \mathrm { s t r } } ( d )$ denote the procedures satisfying (2.1) and (2.2). Define

$$
\mathcal { R } _ { n , K , \delta } ^ { \mathrm { s t r } } ( d ) = \operatorname* { i n f } _ { \widehat { w } \in A _ { n , K , \delta } ^ { \mathrm { s t r } } ( d ) } \operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) .\tag{2.3}
$$

We write $\mathcal { R } _ { n , K } ^ { \mathrm { f r e e } } ( d )$ for the analogous streaming risk with no safety constraint. All stated lower bounds for fixed confidence also apply to $\mathcal { R } _ { n , K } ^ { \mathrm { f r e e } } ( d )$

Terminal compression. For the terminal benchmark, an encoder may use all observations and unrestricted working memory, but sends one of K labels to the decoder. The final action must satisfy (2.2). Its minimax risk is $\mathcal { R } _ { n , K , \delta } ^ { \mathrm { t e r m } } ( d )$ . Both models impose the same restriction on the final label; only streaming imposes it on every intermediate state.

We take $K = 2 ^ { b }$ , with $b \geq 0$ an integer. Bounds for other integer alphabet sizes follow by rounding with resources changing by at most a constant factor. The notation $a \asymp \delta$ b means that their ratio is bounded above and below by positive constants depending only on $\delta .$ Unsubscripted logarithms are natural, and $\log _ { 2 }$ denotes the logarithm to base two.

## 3 Minimax risk and memory requirements in high dimensions

We first give the benchmark that accounts only for representation of the answer. Its two terms are sampling error and the squared covering radius of a K-point codebook.

Theorem 3.1 (Terminal compression). For every fixed $0 < \delta < 1$ , uniformly in integers $n , d \geq 1$ and $b \geq 0$

$$
\mathcal { R } _ { n , 2 ^ { b } , \delta } ^ { \mathrm { t e r m } } ( d ) \asymp _ { \delta } \operatorname* { m i n } \{ 1 , d / n + 2 ^ { - 2 b / d } \} .
$$

The lower bound holds without the safety requirement and allows randomized encoding and decoding.

The streaming problem charges the same state budget at every update. The resulting risk includes the cost of retaining evidence for future observations.

Theorem 3.2 (Optimal streaming risk). Put

$$
\Psi ( n , d , K ) = \operatorname* { m i n } \left\{ 1 , K ^ { - 2 / d } + \frac { d } { n } \left[ 1 + \frac { \log ( e d ) } { 1 + ( \log K ) / d } \right] \right\} .\tag{3.1}
$$

For every fixed $0 < \delta < 1$ there is a finite $C _ { \delta }$ such that

$$
\frac { 1 } { 2 5 6 } \Psi ( n , d , K ) \leq \mathcal { R } _ { n , K } ^ { \mathrm { f r e e } } ( d ) \leq \mathcal { R } _ { n , K , \delta } ^ { \mathrm { s t r } } ( d ) \leq C _ { \delta } \Psi ( n , d , K )\tag{3.2}
$$

for all integers $n , d \geq 1$ and $K = 2 ^ { b } , b \geq 0$ . The upper bound is achieved by a deterministic procedure whose transitions vary with time.

The factor

$$
1 + \frac { \log ( e d ) } { 1 + ( \log K ) / d }
$$

quantifies the sample penalty for limited working memory. If the available bits per coordinate are of order log d, the factor is bounded. If they remain bounded, it is of order log d. Increasing the number of states can therefore improve sample eficiency even after the answer has enough representational precision.

Corollary 3.3 (Sample eficiency requires working bits). Fix $Q > 0$ and let $0 < r \leq 2 5 6 ^ { - 2 }$ . If a streaming procedure has worst-case risk at most r using $d \leq n \leq Q d / r$ observations, then

$$
\log K \geq \operatorname* { m a x } \left\{ { \frac { d } { 4 } } \log { \frac { 1 } { r } } , { \frac { d \log ( e d ) } { 2 5 6 Q } } - d \right\} .\tag{3.3}
$$

Conversely, for each fixed δ, risk at most $\textit { r }  { \textit { i s } }$ achievable with $O _ { \delta } ( d / r )$ samples and $O _ { \delta } \{ d \operatorname* { m a x } \{ \log ( 1 / r ) , \log ( e d ) \} \}$ persistent bits.

For fixed suficiently small r and suficiently large d, this gives a requirement of order d log d for working memory at statistically optimal sample size. The terminal benchmark needs only order d bits at that same fixed risk. For decreasing r, both problems must also pay the cost of storing the answer, $d \log ( 1 / r )$ ; the corollary identifies when that cost dominates the extra workspace.

Corollary 3.4 (A separation with vanishing risks). Fix $0 < \delta < 1$ . For $d \geq 3$ , set

$$
n _ { d } = \lceil d ( \log d ) ^ { 2 } \rceil , \qquad b _ { d } = \lceil d \log _ { 2 } \log d \rceil , \qquad K _ { d } = 2 ^ { b _ { d } } .
$$

Then, as $d \to \infty$ ,

$$
\mathcal { R } _ { n _ { d } , K _ { d } , \delta } ^ { \mathrm { t e r m } } ( d ) \asymp _ { \delta } \frac { 1 } { ( \log d ) ^ { 2 } } , \qquad \mathcal { R } _ { n _ { d } , K _ { d } , \delta } ^ { \mathrm { s t r } } ( d ) \asymp _ { \delta } \frac { 1 } { \log d \log \log d } .
$$

Their ratio is of order log d/ log log d.

Thus the separation is visible even when both procedures consistently estimate the mean. A final codebook captures the desired precision, while repeated compression slows the accumulation of evidence.

Theorems 3.1–3.2 and both corollaries are proved in Appendix B. The next two sections explain the construction and information bound that produce the matching rate.

## 4 Scalar minimax risk and an optimal construction

Theorem 4.1 (Scalar equivalence). For every fixed $0 < \delta < 1$ ，

$$
\begin{array} { r } { \mathcal { R } _ { n , K , \delta } ^ { \mathrm { s t r } } ( 1 ) \asymp _ { \delta } \mathcal { R } _ { n , K , \delta } ^ { \mathrm { t e r m } } ( 1 ) \asymp _ { \delta } \operatorname* { m i n } \{ 1 , n ^ { - 1 } + K ^ { - 2 } \} . } \end{array}
$$

A deterministic streaming construction attains the upper bound.

In one dimension, the term for working memory in (3.1) is absorbed by $n ^ { - 1 }$ . The constructive content is more informative: an alphabet of order M states can obtain squared error of order $M ^ { - 2 }$ from order $M ^ { 2 }$ observations. Storing an exact count of those observations would require order $M ^ { 2 }$ states. The construction avoids that count by retaining only a modular remainder.

## 4.1 Modular measurements and scheduled layers

For $X _ { t } \sim N ( \mu , 1 )$ , define $Y _ { t } = \mathbf { 1 } \{ X _ { t } \geq 0 \}$ . Then $Y _ { t } \sim \mathrm { B e r n } ( p )$ with $p = \Phi ( \mu ) \in [ a , 1 - a ]$ , where $a = \Phi ( - 1 )$ . The inverse link is uniformly Lipschitz on this interval. It is therefore enough to estimate p to squared accuracy $M ^ { - 2 }$ . Gaussianity enters this upper bound through the link; the mechanism processes Bernoulli data using finitely many states.

Choose a dyadic resolution $M = 2 ^ { J }$ . The algorithm has J layers, determined before data collection from the resource budgets. At layer $j ,$ the frequency is $f _ { j } = M / 2 ^ { j }$ . A measurement block consists of $A f _ { j } ^ { 2 }$ consecutive observations, with a fixed numerical constant A. Throughout the block, the state contains

$$
C \gets ( C + Y _ { t } ) \bmod ( A f _ { j } ) .\tag{4.1}
$$

At its end, $C / ( A f _ { j } )$ estimates $f _ { j } p$ modulo one with a fixed small error probability: its fluctuation scale is of order $A ^ { - 1 / 2 }$ , independently of $f _ { j }$

The old label approximates $2 f _ { j } p$ modulo one. If it is $^ { r , }$ there are two possible lifts to the current frequency, $r / 2$ and $( r + 1 ) / 2$ , on the unit circle. The block measurement votes for the closer lift. Decreasing the frequency resolves a new ambiguity while preserving information from the previous layer. Each measurement block uses fresh samples and resets the counter, and the public clock determines the block and layer boundaries.

## 4.2 Amplification without storing votes

Each layer uses several rounds, each containing several independent block votes. The algorithm retains only two bits (F, H). The bit F is the current candidate. At the start of a round, $H = 1$ ; a vote agreeing with F changes H to zero. At the end of the round, $F$ is flipped exactly when H is still one, that is, when every vote opposed it. The list of votes is never stored. This device is closely related to the tests with finite memory and transitions that vary with time of Cover (1969).

When each block vote is correct with probability at least 0.99, using $\ell _ { j }$ votes per round and $T _ { j }$ rounds reduces the decision error to $\eta _ { j } = 2 ^ { - 3 j } / 1 6$ . The inherited phase determines when this condition holds. Appendix A gives the explicit schedule and the recovery argument for the remaining layers.

## 4.3 The state budget and error accounting

At layer $j ,$ , the old label has $2 ^ { j - 1 }$ values, the counter has $A f _ { j }$ values, and the amplifier has four states. Their joint state space therefore has size

$$
2 ^ { j - 1 } \cdot A f _ { j } \cdot 4 = 2 A M .\tag{4.2}
$$

The label grows at exactly the rate at which the counter shrinks. This exchange of capacity between state components is the allocation mechanism. The public clock supplies the scheduled round and block indices, so they require no additional memory.

The decreasing block lengths ofset the increasing reliability of later decisions. In fact, $\begin{array} { r } { A \sum _ { j = 1 } ^ { J } f _ { j } ^ { 2 } \ell _ { j } T _ { j } \le 3 2 A M ^ { 2 } } \end{array}$ . The error analysis charges an erroneous layer according to the final Pdisplacement it can cause. Later accurate layers recover from earlier errors up to the corresponding spatial scale. The geometrically decreasing $\eta _ { j }$ make the weighted sum of these losses of order $M ^ { - 2 }$ Appendix A gives the recovery argument.

Choosing M as the largest admissible power of two under the state and sample budgets gives $n ^ { - 1 } + K ^ { - 2 }$ . Safety at fixed confidence is then obtained by shrinking the decoded estimate toward zero. If $q$ has uniform mean squared error at most $v ,$ radial soft thresholding at $\sqrt { v / \delta }$ gives a $\delta \mathfrak { - }$ safe action with risk of order $( 1 + \delta ^ { - 1 } ) v$ and requires no new persistent states (Lemma B.1). Thus fixed baseline protection changes constants but not the rates in one or several dimensions.

## 5 Information bounds and workspace allocation

Running d scalar engines simultaneously stores d working counters. Running them one after another saves states but discards most coordinates of each vector observation. The optimal construction balances these costs, and the lower bound shows that no joint encoding can avoid their statistical efect.

## 5.1 Information accumulation under a state constraint

Let $p _ { t , s } ( \mu ) = \mathbb { P } _ { \mu } ( S _ { t } = s )$ and $A _ { t } = \nabla _ { \mu } \log p _ { t , S _ { t } } ( \mu )$ be the score of the current state. With $Z _ { t } = X _ { t } - \mu$ the identity for the conditional score is

$$
A _ { t } = \mathbb { E } _ { \mu } [ A _ { t - 1 } + Z _ { t } \mid S _ { t } ] .\tag{5.1}
$$

Every observation adds a fresh Gaussian score, and the next state keeps only its conditional mean. The score variance $V _ { t } = \mathbb { E } _ { \mu } \Vert A _ { t } \Vert ^ { 2 }$ is the trace of the state Fisher information. It need not increase: an update may erase information. The proof keeps these potentially negative increments throughout the entropy accounting.

We smooth the distribution of $A _ { t }$ by an independent Gaussian. Its entropy above the smoothing noise is bounded by $H ( S _ { t } ) \leq \log K$ . An entropy comparison for conditional expectations bounds the loss from compression by the associated squared error. Choosing the smoothing variance according to remaining time and telescoping gives a harmonic information budget. Freezing the terminal state over a longer horizon and summing by parts converts this budget into

$$
\mathrm { t r } \mathcal { T } _ { n } ( \mu ) \leq \frac { 2 n ( 2 \log K + d ) } { \log ( n + 2 ) } .\tag{5.2}
$$

The numerator in (5.2) measures the information that the state budget can retain; the logarithmic denominator comes from compression across time. The van Trees inequality converts this information bound into risk through integration by parts. For sample sizes at least the dimension, the logarithm of the horizon supplies the dimension dependence in the term for working memory. At smaller sample sizes, the sampling lower bound is already of constant order. Combining these bounds with the codebook lower bound proves the lower half of Theorem 3.2. The argument treats the whole state jointly and imposes no coordinatewise architecture.

## 5.2 Reusing workspace across groups

The upper bound gives stored and active coordinates diferent roles. Completed coordinates are rounded and encoded jointly as a point in a Euclidean ball. A new group uses parallel copies of the scalar engine on a fresh segment of the stream. Its counters are then discarded and its rounded answers join the stored label. The same workspace is reused for the next group. Each coordinate estimate is based on arriving samples; a group boundary never gives access to old observations.

Joint encoding matters. Encoding every coordinate in a separate interval that covers its full range incurs avoidable factors of $\sqrt { d }$ . With grid spacing $\varepsilon / { \sqrt { d } } .$ , a bounded ball in k completed coordinates contains at most $( C \varepsilon ^ { - 1 } \sqrt { d / k } ) ^ { k }$ grid points. A norm check on each partial vector keeps it within a bounded codebook; the procedure returns zero if the check fails. All such failures lie in a single event controlled by the total estimation error, so their risk contribution stays of order $\varepsilon ^ { 2 }$

The group schedule also adapts to available workspace. Write $a = 1 + \log ( 1 / \varepsilon )$ $\begin{array} { r } { h = \frac { 1 } { 2 } } \end{array}$ log d, and let $\beta d$ be the logarithmic state capacity available beyond answer encoding. A fraction approximately $\theta = \operatorname* { m i n } \{ 1 , ( a + \beta ) / ( a + h ) \}$ of the remaining coordinates is activated at each stage. Groups shrink as the stored part grows. Their scalar resolutions balance total squared error and stream duration. Writing $K _ { \mathrm { u s e d } }$ and $N _ { \mathrm { u s e d } }$ for its resource use, the resulting construction satisfies

$$
\begin{array} { c } { \log K _ { \mathrm { u s e d } } \leq d \log ( C / \varepsilon ) + \beta d , } \\ { N _ { \mathrm { u s e d } } \leq C \displaystyle \frac { d } { \varepsilon ^ { 2 } } \left( 1 + \displaystyle \frac { \log d } { 1 + \log ( 1 / \varepsilon ) + \beta } \right) } \end{array}\tag{5.3}
$$

for mean squared error of order $\varepsilon ^ { 2 }$ . The first resource bound splits the state budget between answer precision and active workspace. Increasing the latter allows more coordinates to be processed together, reducing the sample cost in the second bound. Optimizing subject to $K _ { \mathrm { u s e d } } \leq K$ gives the upper

half of (3.2). Fine working precision is shared over time, while finished answers occupy a compact representation.

## 6 Baseline protection with increasing confidence

We first take $d = 1$ and allow δ to decrease. A candidate estimate followed by validation on fresh observations achieves high confidence with a small state space. The validation step introduces a separate tradeof between sample size and memory. Set

$$
L = 1 + \log ( 1 / \delta ) , \qquad U = 1 + \log L , \qquad V = 1 + \log U .\tag{6.1}
$$

Fix $\rho \in ( 0 , 1 / 2 )$ and let

$$
N _ { \rho } ( K , \delta ) = \operatorname* { m i n } \left\{ n \in { \mathbb { N } } : \ \exists \widehat { w } \in \mathcal { A } _ { n , K , \delta } ^ { \mathrm { s t r } } ( 1 ) , \quad \operatorname* { s u p } _ { | \mu | \leq 1 } \mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) \leq \rho \right\} ,
$$

with the minimum of an empty set taken to be infinity.

Theorem 6.1 (Sample size and state requirements at fixed risk). For every fixed $\rho \in ( 0 , 1 / 2 )$ there are positive constants $c _ { \rho } , C _ { \rho } , D _ { \rho } , \delta _ { \rho }$ such that, whenever $0 < \delta \leq \delta _ { \rho }$

$$
c _ { \rho } L \operatorname* { m a x } \{ 1 , U / K \} \leq N _ { \rho } ( K , \delta ) \leq C _ { \rho } L \operatorname* { m a x } \{ 1 , U / K \} .\tag{6.2}
$$

The lower bound holds for every $K = 2 ^ { b } ,$ ; the upper bound holds when $K \ge D _ { \rho } V$

The confidence cost is $L \asymp \log ( 1 / \delta )$ samples. Below order U states, validation with finite memory adds a factor $U / K$ . At order U states, sample complexity returns to the ordinary confidence scale. Thus a slowly growing state space recovers full sample eficiency; matching begins at the explicit threshold $D _ { \rho } V$

Corollary 6.2 (Exponential protection at a prescribed horizon). For each fixed $\rho \in ( 0 , 1 / 2 )$ there exists $\gamma _ { 0 } ( \rho ) > 0$ such that, for every fixed $0 < \gamma \leq \gamma _ { 0 } ( \rho )$ and $\delta _ { n } = e ^ { - \gamma n }$ , the smallest admissible number of states for risk at most ρ satisfies

$$
K _ { \operatorname* { m i n } } ( n , \rho , \delta _ { n } ) = \Theta _ { \rho , \gamma } ( \log n ) .
$$

The minimum number of persistent bits is $\log _ { 2 } \log n + O _ { \rho , \gamma } ( 1 )$

This guarantee concerns the decoded action at the prescribed horizon n. Under the same exponential baseline protection, a fixed terminal alphabet sufices at fixed risk for suficiently small γ (Proposition C.9). The growing cost of working memory therefore comes from validating the decision to act. With bounded working states, the minimax risk instead tends to the value of the zero baseline, $1 / 2$ (Proposition C.6).

Testing and validation. Safety at zero and risk improvement at a separated mean induce a test with small type I error and fixed positive power. Monotone reduction turns it into an ordered test with finitely many states. A bound on weighted certificate area, together with fixed power, yields a short acceptance certificate. Its probability under the null limits the achievable type I error. For a fixed Gaussian mean separation and fixed power, the test decision D satisfies − log $P _ { 0 } ( D = 1 ) \leq$ CKn/ log n for large n. Combining the state restriction with ordinary testing information yields the lower bound. Appendix C also gives the joint risk bound in $n , K , \delta$ (Theorem C.5).

The upper bound first forms a pilot estimate at fixed resolution. It then validates a conservative update using an AND of repeated OR tests on fresh blocks. The OR tests retain power; their AND suppresses unsafe acceptance. The public clock schedules these tests, and their work states are reused across a fixed collection of candidate scales. This asymmetric validation attains (6.2).

The same validation mechanism applies coordinate by coordinate in any fixed dimension. The constants then depend on the dimension and target risk, while the dependence on confidence and total states retains its scalar form. For dimension $d ,$ let $\dot { N } _ { \rho } ^ { ( d ) } ( K , \delta )$ be the minimum sample size for a procedure in $\mathcal { A } _ { n , K , \delta } ^ { \mathrm { s t r } } ( d )$ with worst-case risk at most $\rho ,$ or infinity if none exists.

Corollary 6.3 (Confidence costs in fixed dimension). Fix an integer $d \geq 1$ and $\rho \in ( 0 , 1 / 2 )$ . There are positive constants $c _ { d , \rho } , C _ { d , \rho } , D _ { d , \rho } , \delta _ { d , \rho }$ such that, for $0 < \delta \leq \delta _ { d , \rho }$

$$
c _ { d , \rho } L \operatorname* { m a x } \{ 1 , U / K \} \leq N _ { \rho } ^ { ( d ) } ( K , \delta ) \leq C _ { d , \rho } L \operatorname* { m a x } \{ 1 , U / K \} .\tag{6.3}
$$

The lower bound holds for every $K = 2 ^ { b }$ ; the upper bound holds for $K \ge D _ { d , \rho } V$ . Consequently, there is $\gamma _ { 0 } ( d , \rho ) > 0$ such that, for every fixed $0 < \gamma \leq \gamma _ { 0 } ( d , \rho )$ and $\delta _ { n } = e ^ { - \gamma n }$ , the minimum state and bit budgets for risk at most ρ satisfy, as $n \to \infty$

$$
K _ { \operatorname* { m i n } } ^ { ( d ) } ( n , \rho , \delta _ { n } ) = \Theta _ { d , \rho , \gamma } ( \log n ) , \qquad b _ { \operatorname* { m i n } } ^ { ( d ) } = \log _ { 2 } \log n + O _ { d , \rho , \gamma } ( 1 ) .
$$

Sequential coordinate blocks reuse one validation workspace, retaining completed candidate labels. Each coordinate receives failure probability $\delta / d ;$ restriction to a coordinate axis gives the matching lower bound. The scalar confidence law thus extends to every fixed dimension. Appendix C.7 gives the proof and resource accounting.

## 7 Conclusion

The memory required to compute a mean can exceed the memory required to represent it. At fixed confidence, scalar streaming estimation attains the rate for terminal encoding. In higher dimensions, statistically optimal sample size at a fixed suficiently small risk requires order d log d working bits, compared with order d bits for the answer. Increasing confidence introduces a further cost: at fixed risk in every fixed dimension, failure probability $e ^ { - \gamma n }$ for suficiently small fixed $\gamma > 0$ requires and permits $\log _ { 2 } \log n + O ( 1 )$ working bits.

The constructions attain these bounds by separating stored estimates from active computation. Completed estimates occupy compact labels, and the workspace is reused for modular measurements, coordinate groups, and validation. The matching lower bounds identify when this workspace is statistically necessary. Together, the results quantify how a memory budget controls answer precision, sample eficiency, and confidence.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant 32270689.

## AI use statement

OpenAI’s ChatGPT/Codex was used to assist with mathematical reasoning and proof checking, reference verification, and manuscript organization and language editing. The authors are responsible for the mathematical claims, citations, and final content of this manuscript.

## References

Acharya, J., Canonne, C. L., Sun, Z., and Tyagi, H. (2022). Unified lower bounds for interactive high-dimensional estimation under information constraints. arXiv:2010.06562v6.

Barnes, L. P., Han, Y., and Özgür, A. (2020). Lower bounds for learning distributions under communication constraints via Fisher information. Journal of Machine Learning Research, 21(236), 1–30. jmlr.org/papers/v21/19-737.html.

Barnes, L. P., and Özgür, A. (2021). Fisher information and mutual information constraints. In 2021 IEEE International Symposium on Information Theory, 2179–2184. doi:10.1109/ISIT45174.2021.9518061.

Berg, T., Ordentlich, O., and Shayevitz, O. (2021). Deterministic finite-memory bias estimation. In Proceedings of the 34th Conference on Learning Theory, Proceedings of Machine Learning Research, 134, 566–585. PMLR 134:berg21a.

Berg, T., Ordentlich, O., and Shayevitz, O. (2024). Statistical inference with limited memory: A survey. IEEE Journal on Selected Areas in Information Theory, 5, 623–644. doi:10.1109/JSAIT.2024.3481296.

Cover, T. M. (1969). Hypothesis testing with finite statistics. The Annals of Mathematical Statistics, 40(3), 828–835. doi:10.1214/aoms/1177697590.

Cover, T. M., Freedman, M. A., and Hellman, M. E. (1976). Optimal finite memory learning algorithms for the finite sample problem. Information and Control, 30(1), 49–85. doi:10.1016/S0019-9958(76)90456-3.

Diakonikolas, I., Kane, D. M., Pensia, A., and Pittas, T. (2022). Streaming algorithms for high-dimensional robust statistics. In Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research, 162, 5061–5117. PMLR 162:diakonikolas22a.

Guo, D., Shamai, S., and Verdú, S. (2005). Mutual information and minimum mean-square error in Gaussian channels. IEEE Transactions on Information Theory, 51(4), 1261–1282. doi:10.1109/TIT.2005.844072.

Hellman, M. E. (1974). Finite-memory algorithms for estimating the mean of a Gaussian distribution. IEEE Transactions on Information Theory, 20(3), 382–384. Author’s reprint.

Hoefding, W. (1963). Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301), 13–30. doi:10.1080/01621459.1963.10500830.

Mayekar, P., Suresh, A. T., and Tyagi, H. (2021). Wyner–Ziv estimators: Eficient distributed mean estimation with side-information. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, 130, 3502–3510. PMLR 130:mayekar21a.

OpenAI (2026). Subsphere methods for memory–sample lower bounds in noiseless Gaussian regression. Research manuscript, September 27. Public manuscript.

Polyanskiy, Y., and Wu, Y. (2016). Wasserstein continuity of entropy and outer bounds for interference channels. IEEE Transactions on Information Theory, 62(7), 3992–4002. doi:10.1109/TIT.2016.2562630.

Shamir, O. (2014). Fundamental limits of online and distributed algorithms for statistical learning and estimation. In Advances in Neural Information Processing Systems, 27, 163–171. NeurIPS proceedings.

Steinhardt, J., Valiant, G., and Wager, S. (2016). Memory, communication, and statistical queries. In Proceedings of the 29th Conference on Learning Theory, Proceedings of Machine Learning Research, 49, 1490–1516. PMLR 49:steinhardt16.

Trippe, B. L., Deshpande, S. K., and Broderick, T. (2024). Confidently comparing estimates with the c-value. Journal of the American Statistical Association, 119(546), 983–994. doi:10.1080/01621459.2022.2153688.

Wagner, T. J. (1972). Estimation of the mean with time-varying finite memory. IEEE Transactions on Information Theory, 18(4), 523–525. doi:10.1109/TIT.1972.1054853.

Zhu, Y., and Laferty, J. (2014). Quantized estimation of Gaussian sequence models in Euclidean balls. In Advances in Neural Information Processing Systems, 27, 3662–3670. NeurIPS proceedings.

J. Loveless, S. Stoikov, and R. Waeber, “Online algorithms in high-frequency trading,” Communications of the ACM, vol. 56, no. 10, pp. 50–56, 2013. doi:10.1145/2507771.2507780.

G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, “Eficient streaming language models with attention sinks,” in Proc. 12th International Conference on Learning Representations, 2024. OpenReview.

## Appendix: Proofs

The proofs follow the construction’s dependency structure. Appendix A establishes the scalar engine; Appendix B combines it with the information bound and grouped allocation; Appendix C develops the asymmetric tests needed when baseline protection increases.

## A Scalar estimation with a finite state space

This appendix proves the scalar construction used in Theorem 4.1 and in the upper bound for several dimensions. Its statistical input is a Bernoulli stream; Gaussianity enters only in the final conversion from a sign probability to a mean. Throughout this appendix, put

$$
A = 1 6 3 8 4 , \qquad \epsilon _ { 0 } = \frac { 1 } { 6 4 } , \qquad \kappa = \frac { 1 } { 1 0 0 } , \qquad \rho = \frac { 9 } { 3 2 } , \qquad c _ { \star } = \frac { 9 } { 1 6 } , \qquad D = \frac { 1 3 7 7 } { 4 0 9 6 } .\tag{A.1}
$$

Write dist $\begin{array} { r } { \mathsf { \Pi } _ { : } ( u , v ) = \operatorname* { m i n } _ { m \in \mathbb { Z } } | u - v - m | } \end{array}$ for distance on $\mathbb { T } = \mathbb { R } / \mathbb { Z } .$

Lemma A.1 (Scalar estimation subroutine). Let $a = \Phi ( - 1 )$ and ${ \cal B } = { \cal D } / [ a ^ { 2 } \phi ( 1 ) ^ { 2 } ]$ . For every power of two $M \geq 2$ , there is a deterministic streaming estimator q for observations $X _ { i } \sim N ( \mu , 1 ) , \vert \mu \vert \le 1$ 2 which uses at most $2 A M$ states and $3 2 A M ^ { 2 }$ observations and satisfies

$$
q \in [ - 1 , 1 ] , \qquad \operatorname* { s u p } _ { | \mu | \leq 1 } \mathbb { E } _ { \mu } ( q - \mu ) ^ { 2 } \leq { \frac { B } { M ^ { 2 } } } .\tag{A.2}
$$

The sample schedule is fixed before any observations are read.

We first construct an estimator of a Bernoulli probability on the circle, then prove the lemma by decoding its final state.

## A.1 Modular measurements and recovery from two candidates

For an integer frequency $f \geq 1$ , a measurement uses $L = A f ^ { 2 }$ observations $Y _ { i } \sim$ Bernoulli(p) and the counter

$$
V = A f , \qquad C _ { 0 } = 0 , \qquad C _ { t } = ( C _ { t - 1 } + Y _ { t } ) \bmod V , \qquad z = C _ { L } / V .\tag{A.3}
$$

Only $C _ { t } \in \{ 0 , \ldots , V - 1 \}$ persists between observations. Since $L / V = f$ , the block output satisfies

$$
\mathrm { d i s t } _ { \mathbb { T } } ( z , f p ) \leq \left| \frac { \sum _ { i = 1 } ^ { L } Y _ { i } } { V } - f p \right| = f | \overline { { Y } } - p | .
$$

Hoefding’s inequality (Hoefding, 1963) gives, uniformly over $p \in [ 0 , 1 ]$

$$
\mathbb { P } _ { p } \{ \mathrm { d i s t } _ { \mathbb { T } } ( z , f p ) > \epsilon _ { 0 } \} \le 2 \exp ( - 2 A \epsilon _ { 0 } ^ { 2 } ) = 2 e ^ { - 8 } < \kappa .\tag{A.4}
$$

Thus $A f$ states sufice to measure the phase $f p$ mod 1 to fixed accuracy with fixed confidence, using $A f ^ { 2 }$ samples.

The next elementary observation explains how such measurements recover a probability while the frequencies decrease.

Lemma A.2 (Recovery from a doubled phase). Let $r \in [ 0 , 1 ) , \theta \in \mathbb { T }$ , and $d = \mathrm { d i s t } _ { \mathbb { T } } ( r , 2 \theta )$ . For the two candidates $c _ { 0 } = r / 2$ and $c _ { 1 } = ( r + 1 ) / 2$ , their distances to θ are the unordered pair

$$
\left\{ \mathrm { d i s t } _ { \mathbb { T } } ( c _ { 0 } , \theta ) , \mathrm { d i s t } _ { \mathbb { T } } ( c _ { 1 } , \theta ) \right\} = \left\{ \frac { d } { 2 } , \frac { 1 } { 2 } - \frac { d } { 2 } \right\} .\tag{A.5}
$$

If $d \leq 7 / 1 6$ and dist $_ \mathbb { T } ( z , \theta ) \le \epsilon _ { 0 }$ , the candidate nearest to z is the one nearest to θ, and its error is $d / 2$ $I f d > 7 / 1 6$ , either candidate has error at most $\rho = 9 / 3 2$

Proof. Rotate the circle by $- r / 2 ,$ so the candidates become 0 and $1 / 2$ . If $q = \mathrm { d i s t } _ { \mathbb { T } } ( 0 , \theta - r / 2 )$ , their distances are q and $1 / 2 - q$ , while $d = \operatorname* { m i n } \{ 2 q , 1 - 2 q \}$ . This proves $\left( \mathrm { { A . 5 } } \right)$ . When $d \leq 7 / 1 6$ , the gap between the two distances is $1 / 2 - d \geq 4 \epsilon _ { 0 }$ . Replacing θ by z changes this gap by at most $2 \epsilon _ { 0 }$ , so the nearer candidate remains nearer. When d $> 7 / 1 6$ , the larger distance is $1 / 2 - d / 2 < 1 / 4 + 2 \epsilon _ { 0 } = \rho .$ □

The recovery case is essential: a mistaken earlier decision need not invalidate every later measurement. One accurate layer brings an arbitrary previous error into a region where subsequent accurate layers halve it.

## A.2 Amplification with four states

Lemma A.3 (Amplifier with four states). Suppose independent, identically distributed votes $U _ { i } \in$ {0, 1} satisfy $\mathbb { P } ( U _ { i } = u _ { * } ) \ge 9 9 / 1 0 0$ for some fixed $u _ { * }$ . For $0 < \eta < 1$ , define

$$
\ell = \left\lceil \frac { \log ( 2 / \eta ) } { \log 9 9 } \right\rceil , \qquad T = \left\lceil ( 1 0 0 / 9 9 ) ^ { \ell } \log ( 2 / \eta ) \right\rceil .\tag{A.6}
$$

There is a deterministic program with four states that reads ℓT votes and returns $u _ { * }$ with probability at least $1 - \eta$

Proof. The program stores a current decision $F \in \{ 0 , 1 \}$ and a flag $H \in \{ 0 , 1 \}$ . Each of $T$ rounds contains ℓ votes. At the start of a round set $H = 1$ and hold F fixed throughout that round. Whenever a vote equals $F ,$ set $H = 0$ . At the end of the round, flip F if and only if $H = 1$ . Thus a decision changes only when every vote in a round opposes it. The public clock gives the round boundaries; no votes or round counters are stored. This construction with prescribed runs is related to the tests with transitions that vary over time of Cover (1969); the following argument gives the bound on error probability used here.

Assume $u _ { * } = 1$ and let $u = \mathbb { P } ( U _ { i } = 1 ) \geq 9 9 / 1 0 0$ . At round boundaries, F is a Markov chain with two states with transition probabilities

$$
\alpha = u ^ { \ell } \quad ( 0 \longrightarrow 1 ) , \qquad \beta = ( 1 - u ) ^ { \ell } \quad ( 1 \longrightarrow 0 ) .
$$

Since $\ell \geq 1 , 0 < \alpha + \beta \leq 1$ . For $w _ { t } = \mathbb { P } ( F _ { t } = 0 )$ , the recursion $w _ { t + 1 } = \beta + ( 1 - \alpha - \beta ) w _ { t }$ yields, for any initial decision,

$$
\begin{array} { l } { \displaystyle { w _ { T } \leq \frac { \beta } { \alpha } + \exp ( - T \alpha ) } } \\ { \displaystyle { \quad \leq 9 9 ^ { - \ell } + \exp \{ - T ( 9 9 / 1 0 0 ) ^ { \ell } \} \leq \eta . } } \end{array}\tag{A.7}
$$

Interchanging labels proves the other case. The only persistent variables are $( F , H )$ , giving four states. □

## A.3 The frequency schedule and its memory cost

Fix $M = 2 ^ { J } , J \geq 1$ , and predetermined target errors $\eta _ { 1 } , \dotsc , \eta _ { J } \in ( 0 , 1 )$ . Layer $j$ uses frequency $f _ { j } = M / 2 ^ { j }$ and starts with the label

$$
r _ { j - 1 } = k / 2 ^ { j - 1 } , \qquad 0 \leq k < 2 ^ { j - 1 } , \qquad r _ { 0 } = 0 .
$$

Keep this label fixed during the layer. Each fresh block of $A f _ { j } ^ { 2 }$ samples produces the modular measurement (A.3) and casts one vote for the nearer of the candidates $r _ { j - 1 } / 2$ and $( r _ { j - 1 } + 1 ) / 2 $ break ties in favor of the first candidate. Use the amplifier with target error $\eta _ { j }$ and initial decision $F = 0$ , producing its final decision after $T _ { j }$ rounds of $\ell _ { j }$ votes each. Set

$$
r _ { j } = \frac { r _ { j - 1 } + F } { 2 } .\tag{A.8}
$$

The final output is $r _ { J } \in \{ 0 , 1 / M , \ldots , ( M - 1 ) / M \}$

At every sample time in layer j, all persistent information is

$$
( k , F , H , C ) , \qquad 0 \leq k < 2 ^ { j - 1 } , \quad F , H \in \{ 0 , 1 \} , \quad 0 \leq C < A f _ { j } .\tag{A.9}
$$

Its number of possible values is

$$
2 ^ { j - 1 } \cdot 4 \cdot A f _ { j } = 2 A M .\tag{A.10}
$$

In particular, the growing label and the shrinking counter exactly ofset each other. A layer changes the interpretation of a common state set; it does not add a new state set to those of earlier layers.

For completeness, the implementation updates memory after every sample. An ordinary transition increments $C$ modulo $A f _ { j }$ . At a block boundary it first forms the updated counter, computes the vote temporarily, and updates H. If that boundary also ends a round, it then applies the possible flip of $F$ and resets the flag for the next round. At a layer boundary it replaces the label by

$$
k _ { \mathrm { n e w } } = k + F 2 ^ { j - 1 } ,
$$

which implements (A.8) exactly, and resets the work variables. The public sample index determines every block, round, and layer boundary. These operations compose into one transition, so neither the temporary vote nor its normalized phase needs an extra persistent state. The final M labels also fit in the same state set. After the scheduled computation, the program can freeze its state and ignore any remaining observations.

The total number of observations is exactly

$$
N ( M ; \eta _ { 1 } , \dots , \eta _ { J } ) = A \sum _ { j = 1 } ^ { J } f _ { j } ^ { 2 } \ell _ { j } T _ { j } .\tag{A.11}
$$

For the construction at fixed confidence choose

$$
\eta _ { j } = 2 ^ { - 3 j } / 1 6 .\tag{A.12}
$$

Writing $t _ { j } = \log ( 2 / \eta _ { j } ) = \log 3 2 + 3 j$ log 2 gives $\ell _ { j } \leq j { + } 1 , t _ { j } \leq 4 ( j { + } 1 )$ , and $T _ { j } \leq 4 ( j + 1 ) ( 1 0 0 / 9 9 ) ^ { j + 1 } +$

1. Consequently,

$$
\begin{array} { r l r } {  { \frac { N ( M ) } { A M ^ { 2 } } = \sum _ { j = 1 } ^ { J } 4 ^ { - j } \ell _ { j } T _ { j } } } \\ & { } & { \leq 8 \sum _ { j = 1 } ^ { \infty } \frac { ( j + 1 ) ^ { 2 } } { 3 ^ { j } } + \sum _ { j = 1 } ^ { \infty } \frac { j + 1 } { 4 ^ { j } } = 2 8 + \frac { 7 } { 9 } < 3 2 . } \end{array}\tag{A.13}
$$

Here we used $( 1 0 0 / 9 9 ) / 4 < 1 / 3$ and $1 0 0 / 9 9 < 2 $ . The increasingly accurate decisions are afordable because the blocks become geometrically shorter.

## A.4 Error analysis through the last incorrect layer

Lemma A.4 (Last incorrect layer). For the preceding algorithm with any predetermined $\eta _ { j }$ , let $d _ { 0 } = \mathrm { d i s t _ { \mathbb { T } } } ( 0 , M p )$ and $d _ { j } = \mathrm { d i s t _ { \mathbb { T } } } ( r _ { j } , f _ { j } p )$ . Then

$$
\mathbb { E } _ { p } d _ { J } ^ { 2 } \le \frac { c _ { \star } ^ { 2 } } { M ^ { 2 } } + c _ { \star } ^ { 2 } \sum _ { j = 1 } ^ { J } \frac { \eta _ { j } } { f _ { j } ^ { 2 } } .\tag{A.14}
$$

Proof. Call layer $j$ bad if $d _ { j - 1 } \leq 7 / 1 6$ and its amplifier chooses the candidate farther from the true phase $f _ { j } p$ mod 1. Let $\mathcal { F } _ { j - 1 }$ contain all observations read before this layer. Conditional on $\mathcal { F } _ { j - 1 }$ , the old label is fixed. If $d _ { j - 1 } \leq 7 / 1 6$ , Lemma A.2 and $\left( \mathrm { { A . 4 } } \right)$ imply that each vote chooses the unique nearer candidate with probability at least 99/100. The votes are conditionally independent and identically distributed: their sample blocks are fresh and disjoint, and their rule depends on the fixed old label, not on the current decision F. Lemma A.3 thus gives

$$
\mathbb { P } _ { p } ( B _ { j } \mid \mathcal { F } _ { j - 1 } ) \le \eta _ { j } .\tag{A.15}
$$

If $d _ { j - 1 } > 7 / 1 6$ , the bad event $B _ { j }$ is empty by definition. Hence this conditional bound holds regardless of earlier errors.

On $B _ { j } ^ { c }$ , Lemma $\mathrm { A . 2 }$ gives $d _ { j } \leq \rho ;$ if also $d _ { j - 1 } \leq 7 / 1 6 .$ , it gives $d _ { j } = d _ { j - 1 } / 2$ . Because $\rho < 7 / 1 6$ the first good layer recovers a controlled error and every consecutive good layer thereafter halves it. If there are no bad layers, therefore,

$$
d _ { J } \leq \rho 2 ^ { - ( J - 1 ) } = c _ { \star } / M .
$$

If $j < J$ is the last bad layer, layer $j + 1$ first restores an error of at most $\rho$ and the remaining layers halve it, giving

$$
d _ { J } \leq \rho 2 ^ { - ( J - j - 1 ) } = c _ { \star } / f _ { j } .
$$

For $j = J$ the same bound follows from $d _ { J } \leq 1 / 2 \leq c _ { \star } / f _ { J }$ . Partition the sample space according to the last bad layer, including the event that none occurs. The event that $j$ is the last bad layer is contained in $B _ { j }$ and has probability at most $\eta _ { j }$ by (A.15). Summing its bound on squared error proves (A.14). □

With (A.12) and $f _ { J } = 1$ , this yields

$$
\operatorname* { s u p } _ { p \in [ 0 , 1 ] } \mathbb { E } _ { p } \mathrm { d i s t } _ { \mathbb { T } } ( r _ { J } , p ) ^ { 2 } \leq \frac { c _ { \star } ^ { 2 } } { M ^ { 2 } } \left( 1 + \frac { 1 } { 1 6 } \sum _ { j = 1 } ^ { \infty } 2 ^ { - j } \right) = \frac { D } { M ^ { 2 } } .\tag{A.16}
$$

Together with (A.10) and (A.13), this proves the circular Bernoulli subroutine with $O ( M )$ states, $O ( M ^ { 2 } )$ samples, and squared error $O ( M ^ { - 2 } )$

Proof of Lemma A.1. Form $Y _ { i } = \mathbf { 1 } \{ X _ { i } \geq 0 \}$ as each observation arrives. These are independent Bernoulli(p) observations with $p = \Phi ( \mu ) \in [ a , 1 - a ]$ . Run the circular subroutine and decode its final label $r = r _ { J }$ as

$$
q = \Phi ^ { - 1 } \bigl ( \Pi _ { [ a , 1 - a ] } ( r ) \bigr ) .\tag{A.17}
$$

For every $p \in [ a , 1 - a ]$ and $r \in [ 0 , 1 )$

$$
| r - p | \leq a ^ { - 1 } \mathrm { d i s t _ { \mathbb { T } } } ( r , p ) .\tag{A.18}
$$

Indeed, if $| r - p | \le 1 / 2$ , circular and ordinary distances agree. Otherwise their circular distance is $1 - | r - p | \geq a$ , whereas $| r - p | \leq 1$ . Projection onto $[ a , 1 - a ]$ cannot increase distance to $p ,$ and the derivative of $\Phi ^ { - 1 }$ on that interval is at most $1 / \phi ( 1 )$ . Thus

$$
| q - \mu | \leq \frac { \mathrm { d i s t _ { \mathbb { T } } } ( r , p ) } { a \phi ( 1 ) } .
$$

Equation $\mathrm { ( A . 1 6 ) }$ proves the claimed mean squared error. The sign transformation is performed within a single transition and the transformation using the inverse cdf only at decoding, so neither changes the persistent state count. □

The same construction estimates an arbitrary mean on $[ 0 , 1 ]$ . For $X _ { i } \in [ 0 , 1 ]$ , generate an independent uniform $U _ { i }$ and form $Y _ { i } = \mathbf { 1 } \{ U _ { i } \leq 1 / 4 + X _ { i } / 2 \}$ . Then $\mathbb { E } Y _ { i } = p = 1 / 4 + ( \mathbb { E } X _ { i } ) / 2 \in [ 1 / 4 , 3 / 4 ]$ . Decoding by $\widehat { m } = 2 \Pi _ { [ 1 / 4 , 3 / 4 ] } ( r _ { J } ) - 1 / 2$ and applying (A.18) with $a = 1 / 4$ gives $\mathbb { E } ( \widehat { m } - \mathbb { E } X _ { i } ) ^ { 2 } \leq 6 4 D / M ^ { 2 }$ b bwith the same resources. This reduction uses fresh randomness but no additional persistent memory.

## A.5 The scalar minimax rate at fixed confidence

Proof of Theorem 4.1. For the upper bound, let

$$
H = \operatorname* { m i n } \left\{ \frac { K } { 2 A } , \sqrt { \frac { n } { 3 2 A } } \right\} .
$$

If $H \ \geq \ 2$ , take the largest power of two $M \ \leq \ H$ , so that $H / 2 \ < \ M \ \leq \ H$ . Lemma A.1 fits within both resource budgets. Apply Lemma B.1 in one dimension with the deterministic threshold $\lambda = \sqrt { B / \delta } / M$ . The resulting safe estimator has risk at most

$$
\frac { B ( 1 + \delta ^ { - 1 } ) } { M ^ { 2 } } \le 4 B ( 1 + \delta ^ { - 1 } ) \operatorname* { m a x } \left\{ \frac { 4 A ^ { 2 } } { K ^ { 2 } } , \frac { 3 2 A } { n } \right\} \le C _ { \delta } ( K ^ { - 2 } + n ^ { - 1 } ) .
$$

If $H < 2 .$ , the zero estimator sufices: its risk is at most $1 / 2$ and either $K < 4 A$ or $n < 1 2 8 A$ , so $K ^ { - 2 } + n ^ { - 1 }$ is bounded below by an absolute positive constant. The zero estimator is also available in the first case. Taking the better predetermined choice gives the asserted upper bound with the truncation at one.

The lower bound holds even for terminal compression without the safety requirement. For its statistical term, take $\mu = \pm h$ , where $h = 1 / ( 2 \sqrt { n } )$ . The relative entropy of the two observation laws is $2 n h ^ { 2 } = 1 / 2$ , so their total variation distance is at most $1 / 2$ by Pinsker’s inequality. Any sign test has average error at least $1 / 4$ . Applying the test $1 \{ \widehat { w } > 0 \}$ shows that the worst-case loss $\widehat { \sf 2 } \left( \widehat { w } - \mu \right) ^ { 2 }$ is at least $1 / ( 3 2 n )$

For the memory term, put the uniform prior on $[ - 1 , 1 ]$ and fix all independent algorithmic randomness, including output randomness. The resulting decoder has at most K output values. The union of their intervals of radius $s = 1 / ( 2 K )$ has length at most one. At least half the prior mass is therefore at distance at least s from every output value, even if an encoder were given the true parameter. Its loss averaged over the prior is at least $s ^ { 2 } / 4 = 1 / ( 1 6 K ^ { 2 } )$ . This bound survives averaging over the randomness. Combining the two terms yields

$$
\operatorname* { s u p } _ { \| \mu \| \leq 1 } \mathbb { E } _ { \mu } \frac { ( \widehat { w } - \mu ) ^ { 2 } } { 2 } \geq \operatorname* { m a x } \{ ( 3 2 n ) ^ { - 1 } , ( 1 6 K ^ { 2 } ) ^ { - 1 } \} \geq \frac { 1 } { 6 4 } \operatorname* { m i n } \{ 1 , n ^ { - 1 } + K ^ { - 2 } \} .
$$

Finally, a terminal encoder can simulate every streaming program and transmit its final state. Its minimax risk is at most the streaming risk, so the same lower bound and the streaming upper bound also establish the scalar rate for terminal compression. □

## B Working memory in several dimensions

Throughout this appendix, $X _ { t } ~ \sim ~ N _ { d } ( \mu , I _ { d } )$ independently, $\mu \in B _ { 2 } ^ { d } .$ and $R _ { \mu } ( w ) = \| w - \mu \| _ { 2 } ^ { 2 } / 2$ Logarithms are natural. All state counts refer to the complete persistent state; the deterministic time index is public.

## B.1 Safety and the terminal benchmark

The following observation allows safety at fixed confidence to be imposed after constructing an estimator with a mean squared error guarantee.

Lemma B.1 (Radial safety transformation). Suppose that q is an estimator satisfying $\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { E } _ { \mu } \Vert q -$ $\mu \| _ { 2 } ^ { 2 } \leq r$ . For $0 < \delta < 1$ , set $\lambda = \sqrt { r / \delta }$ and

$$
w = \left\{ \begin{array} { l l } { 0 , } & { \| q \| _ { 2 } \leq \lambda , } \\ { ( 1 - \lambda / \| q \| _ { 2 } ) q , } & { \| q \| _ { 2 } > \lambda . } \end{array} \right.
$$

Then w is uniformly δ-safe and $\operatorname* { s u p } _ { \mu } \mathbb { E } _ { \mu } R _ { \mu } ( w ) \leq ( 1 + \delta ^ { - 1 } ) r$ . If q is decoded from a finite state, this transformation requires no additional states.

Proof. On $\{ \| q - \mu \| _ { 2 } \leq \lambda \}$ , write $t = \| q \| _ { 2 }$ . When $t > \lambda$

$$
\langle q , \mu \rangle \geq t ( t - \lambda ) , \qquad \langle w , \mu \rangle \geq ( t - \lambda ) ^ { 2 } = \| w \| _ { 2 } ^ { 2 } .
$$

Consequently $R _ { \mu } ( w ) - R _ { \mu } ( 0 ) = \| w \| _ { 2 } ^ { 2 } / 2 - \langle w , \mu \rangle \leq 0$ . The same conclusion is immediate when $w = 0$ Markov’s inequality bounds the probability of the complementary event by $\delta .$ Moreover, $\| w - q \| _ { 2 } \leq \lambda$ so

$$
R _ { \mu } ( w ) \leq \| q - \mu \| _ { 2 } ^ { 2 } + \lambda ^ { 2 } .
$$

Taking expectations proves the claim. If $r = 0$ , use $w = q = \mu$ almost surely.

Lower bounds common to both memory models. Any estimator, even without a memory or safety restriction, satisfies

$$
\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) \geq \frac { 1 } { 3 2 } \operatorname* { m i n } \{ 1 , d / n \} .\tag{B.1}
$$

Indeed, let $a = \operatorname* { m i n } \{ d ^ { - 1 / 2 } , ( 2 \sqrt { n } ) ^ { - 1 } \}$ and consider $\mu _ { \theta } ~ = ~ a \theta$ for $\theta \in \{ - 1 , 1 \} ^ { d }$ . Two parameters difering in one sign induce sample laws with relative entropy $2 n a ^ { 2 } \leq 1 / 2$ and hence total variation at most $1 / 2$ . For each coordinate, the average error probability of the test $1 \{ \widehat { w } _ { i } > 0 \}$ under the uniform sign prior is at least $1 / 4$ . A sign error incurs squared error at least $a ^ { 2 }$ b. The Bayes risk is therefore at least $d a ^ { 2 } / 8 \geq \operatorname* { m i n } \{ 1 , d / n \} / 3 2$ , proving (B.1).

Any procedure whose final decoder receives at most K labels also satisfies

$$
\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) \geq \frac { 1 } { 1 6 } K ^ { - 2 / d } .\tag{B.2}
$$

To see this, put the uniform volume prior on $B _ { 2 } ^ { d }$ and fix the algorithm’s random variables whose laws are independent of the parameter, including its decoder randomization. There are then at most $K$ possible decoded points. Their balls of radius $u = ( 2 K ) ^ { - 1 / d }$ cover at most half the volume of $B _ { 2 } ^ { d }$ . At least half the prior parameters are at distance at least u from every decoded point, even if the encoder is given the true parameter. The prior risk is at least $u ^ { 2 } / 4 \geq K ^ { - 2 / d } / 1 6$ for each fixed random seed. Averaging over the seed proves the randomized version. Combining the two bounds gives, for either memory model,

$$
\operatorname* { s u p } _ { \mu } \mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) \geq \frac { 1 } { 6 4 } \operatorname* { m i n } \{ 1 , d / n + K ^ { - 2 / d } \} .\tag{B.3}
$$

Proof of the terminal upper bound in Theorem 3.1. A maximal u-separated subset of $B _ { 2 } ^ { d }$ is a u-net containing at most $( 1 + 2 / u ) ^ { d }$ points: the disjoint balls of radius $u / 2$ about its points lie in the ball of radius $1 + u / 2$ . If $v = K ^ { 1 / d } \ge 3$ , take $u = 3 / v$ , so that the net has at most K points. Project $\overline { { X } }$ onto $B _ { 2 } ^ { d }$ and encode its nearest net point $q .$ Projection does not increase distance to $\mu ,$ and therefore

$$
\begin{array} { r } { \mathbb { E } _ { \mu } \| q - \mu \| _ { 2 } ^ { 2 } \leq 2 d / n + 2 u ^ { 2 } \leq 1 8 ( d / n + K ^ { - 2 / d } ) . } \end{array}
$$

Lemma B.1 gives the safe upper bound, using the same labels. If $v \ < \ 3$ , the quantization term exceeds $1 / 9$ and the zero estimator sufices. Choosing the zero estimator also whenever the displayed rate exceeds a constant proves the bound with the outer minimum with 1. Together with (B.3), this proves Theorem 3.1.

## B.2 A Fisher information bound for joint states

Fix $\mu \in \mathbb { R } ^ { d }$ and write

$$
p _ { t , s } ( \mu ) = \mathbb { P } _ { \mu } ( S _ { t } = s ) , \quad a _ { t } ( s ) = \nabla \log p _ { t , s } ( \mu ) , \quad A _ { t } = a _ { t } ( S _ { t } ) , \quad V _ { t } = \mathbb { E } _ { \mu } \| A _ { t } \| _ { 2 } ^ { 2 } .
$$

Only reachable states are included. Thus $V _ { t }$ is the trace of the Fisher information matrix $\mathcal { T } _ { t } ( \mu )$ of the state $S _ { t }$ . The random vector $A _ { t }$ is a proof device, not additional algorithmic memory.

State probabilities are smooth: after integrating out the randomization, each is an integral of a bounded measurable function against the Gaussian sample density. Diferentiation under the integral is valid on every compact parameter set. Positivity of Gaussian densities also implies that a state is either unreachable for every parameter or has positive probability for every parameter. Put $Z _ { t } =$ $X _ { t } - \mu$ . Diferentiating the transition identity for $p _ { t , s }$ gives the recursion for the conditional score

$$
A _ { t } = \mathbb { E } _ { \mu } [ A _ { t - 1 } + Z _ { t } \mid S _ { t } ] , \qquad A _ { 0 } = 0 .\tag{B.4}
$$

For completeness, if $Q _ { t } { \big ( } s , x ; s ^ { \prime } { \big ) }$ is the transition probability after integrating out fresh randomness,

then

$$
p _ { t , s ^ { \prime } } = \sum _ { s } p _ { t - 1 , s } \int Q _ { t } ( s , x ; s ^ { \prime } ) \phi _ { d } ( x - \mu ) d x .
$$

Its gradient is the same integral with an additional factor $a _ { t - 1 } ( s ) + x - \mu ;$ division by $p _ { t , s ^ { \prime } }$ proves (B.4). Let $g _ { t } = V _ { t } - V _ { t - 1 }$ . Since $Z _ { t }$ is independent of $S _ { t - 1 }$ and has covariance $I _ { d } ,$ orthogonality of conditional expectation yields

$$
\mathbb { E } _ { \mu } \| A _ { t - 1 } + Z _ { t } - A _ { t } \| _ { 2 } ^ { 2 } = d - g _ { t } .\tag{B.5}
$$

In particular, $g _ { t } \leq d ,$ , but $g _ { t }$ need not be nonnegative.

Lemma B.2 (Entropy loss after Gaussian smoothing). Let $A , Y \in \mathbb { R } ^ { d } \ s a t i s f y \ \mathbb { E } [ Y \mid A ] = A$ , where A has finite support and $\mathbb { E } \Vert Y \Vert _ { 2 } ^ { 2 } < \infty$ . If $G \sim N _ { d } ( 0 , I _ { d } )$ is independent of $( A , Y )$ , then, for $\sigma > 0$

$$
\mathfrak { h } ( Y + \sqrt { \sigma } G ) - \mathfrak { h } ( A + \sqrt { \sigma } G ) \leq \frac { \mathbb { E } \| Y - A \| _ { 2 } ^ { 2 } } { 2 \sigma } .\tag{B.6}
$$

Here h denotes diferential entropy; both entropies are finite.

Proof. Let $p$ be the finite Gaussian mixture density of $A { + } \sqrt { \sigma } G$ and $f = - \log p$ . Direct diferentiation gives

$$
\nabla ^ { 2 } f ( x ) = { \frac { I _ { d } } { \sigma } } - { \frac { \operatorname { C o v } ( A \mid A + { \sqrt { \sigma } } G = x ) } { \sigma ^ { 2 } } } \preceq { \frac { I _ { d } } { \sigma } } .
$$

Consequently $f ( x + u ) \leq f ( x ) + \langle \nabla f ( x ) , u \rangle + \| u \| _ { 2 } ^ { 2 } / ( 2 \sigma )$ . Use $x = A + { \sqrt { \sigma } } G$ and $u = Y - A$ . The linear term has zero expectation because $\mathbb { E } [ Y - A \mid A , G ] = 0$ , whence

$$
\begin{array} { r } { \mathbb { E } f ( Y + \sqrt { \sigma } G ) \leq \mathsf { h } ( A + \sqrt { \sigma } G ) + \mathbb { E } \| Y - A \| _ { 2 } ^ { 2 } / ( 2 \sigma ) . } \end{array}
$$

Relative entropy nonnegativity bounds $\mathsf { h } ( Y + { \sqrt { \sigma } } G )$ by the cross entropy on the left. These operations are integrable: one positive mixture component bounds $f ( x )$ above by a constant plus a quadratic function, $p$ is bounded above, and $\| \nabla f ( x ) \| _ { 2 } \leq ( \| x \| _ { 2 } + \operatorname* { m a x } \| A \| _ { 2 } ) / \sigma$ . The Gaussian convolution of Y has a bounded density and finite second moment, which also guarantees its finite entropy. □

Lemma B.3 (Fisher information retained by a finite state). For every $n , d , K \ge 1$ and every randomized algorithm with transitions that vary over time whose complete state has at most K values,

$$
\operatorname { t r } { \mathcal { T } } _ { n } ( \mu ) \leq { \frac { 2 n ( 2 \log K + d ) } { \log ( n + 2 ) } } \leq { \frac { 4 n ( d + \log K ) } { \log ( n + 2 ) } } \qquad ( \mu \in \mathbb { R } ^ { d } ) .\tag{B.7}
$$

Proof. For a vector A taking finitely many values, define

$$
{ \mathcal { T } } _ { \sigma } ( A ) = \mathsf { h } ( A + { \sqrt { \sigma } } G ) - \mathsf { h } ( { \sqrt { \sigma } } G ) .
$$

This is $I ( A ; A + { \sqrt { \sigma } } G )$ , so $0 \leq \mathcal { T } _ { \sigma } ( A _ { t } ) \leq H _ { \mu } ( S _ { t } ) \leq \log K$ . Apply Lemma B.2 to $A = A _ { t }$ and $Y =$ $A _ { t - 1 } + Z _ { t }$ . The hypothesis on the conditional mean holds because $\mathbb { E } [ Y \mid A _ { t } ] = \mathbb { E } [ \mathbb { E } [ Y \mid S _ { t } ] \mid A _ { t } ] = A _ { t }$ Equations (B.4)–(B.5) and the Gaussian entropy formula give

$$
\mathcal { T } _ { \sigma } ( A _ { t } ) \geq \mathcal { T } _ { \sigma + 1 } ( A _ { t - 1 } ) + \frac { d } { 2 } \log ( 1 + 1 / \sigma ) - \frac { d - g _ { t } } { 2 \sigma } .\tag{B.8}
$$

For a fixed prefix length $m ,$ set $\sigma = m - t + 1$ and sum over $t = 1 , \ldots , m$ . The information terms

telescope, $A _ { 0 } = 0$ , and $\begin{array} { r } { \sum _ { j = 1 } ^ { m } \log ( 1 + 1 / j ) = \log ( m + 1 ) } \end{array}$ . Thus

$$
\sum _ { t = 1 } ^ { m } \frac { g _ { t } } { m - t + 1 } \leq 2 H _ { \mu } ( S _ { m } ) + d \{ H _ { m } - \log ( m + 1 ) \} \leq 2 \log K + d ,\tag{B.9}
$$

where $\begin{array} { r } { H _ { m } = \sum _ { i = 1 } ^ { m } 1 / j } \end{array}$ and $H _ { m } - \log ( m + 1 ) \leq 1$

PFor the proof only, extend the algorithm to time 2n by freezing its state after time n and ignoring subsequent observations. Then $g _ { t } = 0$ for $t > n$ . Sum (B.9) over $m = 1 , \ldots , 2 n$ to obtain

$$
2 n ( 2 \log K + d ) \geq \sum _ { t = 1 } ^ { n } g _ { t } H _ { 2 n - t + 1 } .
$$

Although the increments $g _ { t }$ can be negative, summation by parts and $V _ { t } \geq 0$ give

$$
\sum _ { t = 1 } ^ { n } g _ { t } H _ { 2 n - t + 1 } = V _ { n } H _ { n + 1 } + \sum _ { t = 1 } ^ { n - 1 } { \frac { V _ { t } } { 2 n - t + 1 } } \geq V _ { n } \log ( n + 2 ) .
$$

This proves (B.7). No coordinatewise factorization of the state, or restriction on the computation within a transition, was used. □

Lemma B.4 (Risk bound from Fisher information). Define the prior density on $[ - d ^ { - 1 / 2 } , d ^ { - 1 / 2 } ] ^ { d } \subset$ $B _ { 2 } ^ { d } \ b y$

$$
\pi _ { d } ( \mu ) = d ^ { d / 2 } \prod _ { i = 1 } ^ { d } \cos ^ { 2 } ( \pi \sqrt { d } \mu _ { i } / 2 ) .
$$

Every decoder of $S _ { n }$ satisfies

$$
\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) \geq \frac { d ^ { 2 } } { 2 \{ \mathbb { E } _ { \pi _ { d } } \mathrm { t r } \mathcal { I } _ { n } ( \mu ) + \pi ^ { 2 } d ^ { 2 } \} } .\tag{B.10}
$$

Proof. Projection onto the unit ball followed by averaging over independent decoder randomness cannot increase risk. It therefore sufices to consider fixed decoding points $q _ { s } \in B _ { 2 } ^ { d }$ . The prior has total Fisher information $\pi ^ { 2 } d ^ { 2 }$ : each coordinate contributes

$$
\pi ^ { 2 } d \int _ { - 1 } ^ { 1 } \sin ^ { 2 } ( \pi u / 2 ) d u = \pi ^ { 2 } d .
$$

Under the joint law $\pi _ { d } ( \mu ) p _ { n , s } ( \mu )$ put $T = \nabla \log \{ \pi _ { d } ( \mu ) p _ { n , S _ { n } } ( \mu ) \}$ . The cross term between the prior and state scores is zero because $\begin{array} { r } { \sum _ { s } \nabla p _ { n , s } = 0 } \end{array}$ . Hence

$$
\mathbb { E } \| T \| _ { 2 } ^ { 2 } = \pi ^ { 2 } d ^ { 2 } + \mathbb { E } _ { \pi _ { d } } \operatorname { t r } \mathcal { T } _ { n } ( \mu ) .
$$

Integration by parts in each coordinate gives $\mathbb { E } [ ( q _ { S _ { n } , i } - \mu _ { i } ) T _ { i } ] = 1$ ; the boundary term vanishes because $\pi _ { d }$ is zero on the boundary faces. Summing and applying Cauchy–Schwarz yields

$$
d ^ { 2 } \leq \mathbb { E } \Vert q _ { S _ { n } } - \mu \Vert _ { 2 } ^ { 2 } \mathbb { E } \Vert T \Vert _ { 2 } ^ { 2 } .
$$

All quantities are integrable: the Gaussian score identity $\begin{array} { r } { a _ { n } ( S _ { n } ) = \mathbb { E } [ \sum _ { t = 1 } ^ { n } ( X _ { t } - \mu ) \mid S _ { n } ] } \end{array}$ implies tr ${ \mathcal { I } } _ { n } ( \mu ) \leq n d .$ P. The worst-case risk is at least the prior average, proving the lemma. □

Lower bound in Theorem 3.2. Set $B = \log K , a = d / n , z = K ^ { - 2 / d }$ , and

$$
T _ { n } = \frac { d ^ { 2 } \log ( n + 2 ) } { n ( d + B ) } , \qquad T _ { d } = \frac { d ^ { 2 } \log ( e d ) } { n ( d + B ) } .
$$

Lemmas B.3 and B.4 give a risk lower bound

$$
\frac { T _ { n } } { 8 + 2 \pi ^ { 2 } T _ { n } } \geq \frac { 1 } { 6 4 } \operatorname* { m i n } \{ 1 , T _ { n } \} .
$$

Combine this with (B.3) and use max $\{ \operatorname* { m i n } ( 1 , x ) , \operatorname* { m i n } ( 1 , y ) \} \ge \textstyle { \frac { 1 } { 2 } } \operatorname* { m i n } ( 1 , x + y )$ to obtain

$$
\operatorname* { s u p } _ { \mu } \mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) \geq \frac { 1 } { 1 2 8 } \operatorname* { m i n } \{ 1 , a + z + T _ { n } \} .
$$

If $n \geq d ,$ then $\log ( n + 2 ) \geq { \textstyle { \frac { 1 } { 2 } } } \log ( e d )$ , and thus $T _ { d } \ \leq \ 2 T _ { n }$ . If $n < d ,$ the statistical lower bound already gives a positive constant. In both cases the desired lower bound follows with constant $1 / 2 5 6$ The argument did not use safety.

## B.3 A grouped construction with reusable working memory

We use the scalar engine of Lemma A.1 in its Gaussian form: for every power of two $M \geq 2 .$ , it estimates any $\mu _ { i } \in [ - 1 , 1 ]$ using at most 2AM states and $3 2 A M ^ { 2 }$ observations, with output $Q _ { i }$ satisfying

$$
\begin{array} { r } { \mathbb { E } ( Q _ { i } - \mu _ { i } ) ^ { 2 } \leq B _ { 0 } / M ^ { 2 } , \qquad A = 1 6 3 8 4 , } \end{array}\tag{B.11}
$$

where $B _ { 0 }$ is absolute. Identical scalar programs can run in parallel on diferent coordinates of the same vector observations.

Proposition B.5 (Groups, resolution, and persistent states). There are absolute constants $C _ { s } \geq e$ and $C _ { n } \geq 1$ such that, for every $d \geq 1 , 0 < \varepsilon \leq 1$ , and $\beta \geq 0$ , a deterministic streaming estimator $q \in B _ { 2 } ^ { d }$ has

$$
\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { E } _ { \mu } \| q - \mu \| _ { 2 } ^ { 2 } \leq \frac { 3 } { 2 } \varepsilon ^ { 2 } ,\tag{B.12}
$$

$$
n u m b e r ~ o f ~ s t a t e s \leq e ^ { \beta d } ( C _ { s } / \varepsilon ) ^ { d } ,\tag{B.13}
$$

$$
n u m b e r ~ o f ~ v e c t o r ~ o b s e r v a t i o n s \leq C _ { n } \frac { d } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { 1 + \log ( 1 / \varepsilon ) + \frac { 1 } { 2 } \log d } { 1 + \log ( 1 / \varepsilon ) + \beta } \right\} .\tag{B.14}
$$

Its schedule is determined before the observations arrive.

Proof. Put

$$
\begin{array} { r } { a = 1 + \log ( 1 / \varepsilon ) , \quad h = \frac 1 2 \log d , \quad \vartheta = \operatorname* { m i n } \{ 1 , ( a + \beta ) / ( a + h ) \} , \quad H = \operatorname* { m a x } \{ 2 , \sqrt { 8 B _ { 0 } } \} . } \end{array}
$$

Starting with $r _ { 1 } = d$ unprocessed coordinates, form successive groups of sizes

$$
g _ { j } = \operatorname* { m a x } \{ 1 , \lfloor \vartheta r _ { j } \rfloor \} , \qquad r _ { j + 1 } = r _ { j } - g _ { j } ,
$$

until no coordinates remain, and let $S = \textstyle \sum _ { j } { \sqrt { g _ { j } } }$ . For group j choose the smallest power of two

satisfying

$$
M _ { j } \geq H \sqrt { S \sqrt { g _ { j } } } / \varepsilon .\tag{B.15}
$$

Use a fresh consecutive block of vector observations for each group and run its $g _ { j }$ scalar engines in parallel. Observations of inactive coordinates are discarded. After a group finishes, round its outputs to the grid $\Delta \mathbb { Z } ,$ where $\Delta = \varepsilon / { \sqrt { d } } ,$ , and retain only the rounded coordinates. If the completed partial vector has norm greater than 2, enter a permanent abort state and finally output zero. Otherwise, after all groups finish, project the completed rounded vector onto $B _ { ? } ^ { d }$

Bounds on group sizes. Every group satisfies $g _ { j } \ge \vartheta r _ { j } / 2$ . Therefore

$$
\sqrt { g _ { j } } = \frac { \sqrt { r _ { j } } + \sqrt { r _ { j + 1 } } } { \sqrt { g _ { j } } } ( \sqrt { r _ { j } } - \sqrt { r _ { j + 1 } } ) \le 2 \sqrt { 2 / \vartheta } ( \sqrt { r _ { j } } - \sqrt { r _ { j + 1 } } ) .
$$

Summation gives $S ^ { 2 } \leq 8 d / \vartheta$ . Also $\vartheta d \geq 1$ , since $a \geq 1$ and $( d - 1 ) a \geq h$ . Thus $g _ { j } \leq \vartheta d$ even when the maximum defining $g _ { j }$ equals 1, and $S \sqrt { g _ { j } } \le 2 \sqrt { 2 } d < 3 d$ . In particular, the dyadic resolutions satisfy

$$
M _ { j } < 2 H \sqrt { S \sqrt { g _ { j } } } / \varepsilon \leq 2 H \sqrt { 3 d } / \varepsilon .\tag{B.16}
$$

Error. For analysis, define all scalar and rounded outputs $Q _ { i } , Z _ { i }$ on their prespecified sample blocks even if the actual algorithm has aborted. Equation (B.15) gives

$$
\sum _ { i = 1 } ^ { d } \mathbb { E } ( Q _ { i } - \mu _ { i } ) ^ { 2 } \leq B _ { 0 } \sum _ { j } { \frac { g _ { j } } { M _ { j } ^ { 2 } } } \leq { \frac { B _ { 0 } \varepsilon ^ { 2 } } { H ^ { 2 } S } } \sum _ { j } { \sqrt { g _ { j } } } \leq \varepsilon ^ { 2 } / 8 .
$$

Since $| Z _ { i } - Q _ { i } | \le \Delta / 2 , \mathbb { E } \| Z - \mu \| _ { 2 } ^ { 2 } \le 3 \varepsilon ^ { 2 } / 4$ . If a completed partial vector has norm greater than $2 ,$ its distance from the corresponding part of $\mu$ exceeds 1. The abort event is therefore contained in $\{ \| Z - \mu \| _ { 2 } > 1 \}$ and has probability at most $3 \varepsilon ^ { 2 } / 4$ . Projection cannot increase error on the nonabort event, and the squared error of zero is at most 1. This proves (B.12) without requiring independence between errors.

State count. The number $P _ { k }$ of grid points in the radius-2 ball in k completed coordinates obeys

$$
P _ { k } \leq \left( \frac { C _ { g } } { \varepsilon } \sqrt { \frac { d } { k } } \right) ^ { k } , \qquad C _ { g } = \textstyle { \frac { 5 } { 2 } } \sqrt { 2 \pi e } , \qquad P _ { 0 } = 1 .\tag{B.17}
$$

Indeed, disjoint cubes of side $\Delta$ centered at these grid points fit inside the radius- $5 / 2$ ball; its volume is at most $\scriptstyle ( { \frac { 5 } { 2 } } { \sqrt { 2 \pi e / k } } ) ^ { k }$ . The volume bound for the unit ball follows directly from

$$
\mathrm { v o l } ( B _ { 2 } ^ { k } ) \le e ^ { k / 2 } \int _ { \mathbb { R } ^ { k } } e ^ { - k \| x \| _ { 2 } ^ { 2 } / 2 } d x = ( 2 \pi e / k ) ^ { k / 2 } .
$$

Suppose $k = d - r$ coordinates are complete and the current group has size $g = \mathrm { m a x } \{ 1 , \lfloor \vartheta r \rfloor \}$ The joint state consists of the partial grid label and the $g$ scalar working states. Put $C _ { * } = \operatorname* { m a x } \{ C _ { g } , 4 A H \sqrt { 3 } \}$ and $q _ { 0 } = \log ( 1 / \varepsilon ) = a - 1$ . By (B.16) and (B.17), the logarithm of the number of nonabort states is at most

$$
\begin{array} { r l } & { ( k + g ) ( \log C _ { * } + q _ { 0 } ) + \frac { k } { 2 } \log ( d / k ) + g h } \\ & { \quad \leq d ( \log C _ { * } + q _ { 0 } ) + \frac { k } { 2 } \log ( d / k ) + \{ g h - ( r - g ) q _ { 0 } \} . } \end{array}\tag{B.18}
$$

As usual, $k \log ( d / k ) = 0 { \mathrm { ~ a t ~ } } k = 0$ . The final brace is at most $d ( \beta + 1 )$ . To verify this, if $\beta < h$ and $\lfloor \vartheta r \rfloor \geq 1$ , then

$$
g ( q _ { 0 } + h ) - r q _ { 0 } \leq r \{ \vartheta ( q _ { 0 } + h ) - q _ { 0 } \} = r ( \beta + 1 - \vartheta ) \leq d ( \beta + 1 ) .
$$

If $\beta \geq h$ , then $\vartheta = 1$ and the brace is $r h \leq d \beta$ . In the remaining case $g = 1$ , it is at most $h \leq d .$ Since $k \log ( d / k ) \leq d / e$ , equation (B.18) bounds the number of nonabort states by

$$
e ^ { \beta d } \{ e ^ { 1 + 1 / ( 2 e ) } C _ { * } / \varepsilon \} ^ { d } .
$$

One extra abort state is absorbed by taking $C _ { s } \geq 2 e ^ { 1 + 1 / ( 2 e ) } C _ { * }$ . At a group boundary the working states are replaced by the rounded label in the same transition; the old and new encodings are not retained together. All indices for groups and positions within groups follow the public schedule. Diferent stages reuse the same state alphabet, so their state counts are maximized, not added. The terminal grid state also satisfies the same bound.

Samples. All coordinates in a group read the same vector observations. Thus the number of vector observations is at most

$$
\sum _ { j } 3 2 A M _ { j } ^ { 2 } \leq \frac { 1 2 8 A H ^ { 2 } S } { \varepsilon ^ { 2 } } \sum _ { j } \sqrt { g _ { j } } \leq \frac { 1 0 2 4 A H ^ { 2 } d } { \vartheta \varepsilon ^ { 2 } } .
$$

Taking $C _ { n } \geq 1 0 2 4 A H ^ { 2 }$ proves (B.14).

Upper bound in Theorem 3.2. Write $u = ( \log K ) / d$ and

$$
E = K ^ { - 2 / d } + \frac { d } { n } \left( 1 + \frac { \log ( e d ) } { 1 + u } \right) .
$$

Use the constants $C _ { s } \geq e , C _ { n } \geq 1$ of Proposition B.5. If $u \leq 2 \log C _ { s }$ , then $E \geq C _ { s } ^ { - 4 }$ and the zero estimator gives the desired bound after an absolute adjustment of the constant. Otherwise fix $C _ { 0 } \geq \operatorname* { m a x } \{ C _ { s } ^ { 2 } , C _ { n } , 1 \}$ and set $r = C _ { 0 } E$ . The zero estimator also sufices when $r \geq 1$ . When $r < 1$ choose

$$
\varepsilon = \sqrt { r } , \qquad \beta = u - \log ( C _ { s } / \varepsilon ) .
$$

Because $r \ge C _ { s } ^ { 2 } e ^ { - 2 u }$ , we have $\beta \geq 0$ and $e ^ { \beta d } ( C _ { s } / \varepsilon ) ^ { d } = K$ . Moreover, for $a = 1 + \log ( 1 / \varepsilon )$ and $\begin{array} { r } { h = \frac { 1 } { 2 } \log d , } \end{array}$

$$
a + \beta = 1 + u - \log C _ { s } \geq ( 1 + u ) / 2 , \qquad \operatorname* { m a x } \{ 1 , ( a + h ) / ( a + \beta ) \} \leq 1 + \frac { h } { a + \beta } \leq 1 + \frac { \log ( e d ) } { 1 + u } .
$$

The sample bound in Proposition B.5 is consequently at most $C _ { n } d \{ 1 + \log ( e d ) / ( 1 + u ) \} / r \leq n$ . Freeze the state if the computation finishes before time n. Lemma B.1 gives risk at most $\begin{array} { r } { \frac { 3 } { 2 } ( 1 + \delta ^ { - 1 } ) r \leq C _ { \delta } E } \end{array}$ Choosing zero when $E \geq 1$ establishes the asserted outer minimum.

## B.4 Consequences for representation and sample eficiency

Proof of Corollary 3.4. Let ℓ = log d, $n _ { d } = \lceil d \ell ^ { 2 } \rceil$ , and $b _ { d } = \lceil d \log _ { 2 } \ell \rceil$ . Integer rounding gives

$$
d / n _ { d } \asymp \ell ^ { - 2 } , \qquad K _ { d } ^ { - 2 / d } \asymp \ell ^ { - 2 } , \qquad ( \log K _ { d } ) / d = \log \ell + O ( 1 / d ) .
$$

The terminal risk is therefore $\Theta _ { \delta } ( \ell ^ { - 2 } )$ . The additional streaming term equals

$$
\frac { d } { n _ { d } } \frac { \log ( e d ) } { 1 + ( \log K _ { d } ) / d } \asymp \frac { 1 } { \ell \log \ell } ,
$$

which dominates $\ell ^ { - 2 }$ and still converges to zero. The claimed risk ratio follows from Theorem 3.2.

Proof of Corollary 3.3. The lower bound in Theorem 3.2 applies without safety. If the worst-case risk is at most $0 < r \leq 2 5 6 ^ { - 2 }$ , its rate expression is at most $2 5 6 r < 1$ . In particular,

$$
\log K \geq \frac { d } { 2 } \log \frac { 1 } { 2 5 6 r } \geq \frac { d } { 4 } \log \frac { 1 } { r } .
$$

If also $d \leq n \leq Q d / r$ , the term for working memory implies

$$
\log K \geq { \frac { d ^ { 2 } \log ( e d ) } { 2 5 6 r n } } - d \geq { \frac { d \log ( e d ) } { 2 5 6 Q } } - d .
$$

For $\log ( e d ) \geq 5 1 2 Q$ , this yields $b \geq ( d / \log 2 ) \operatorname* { m a x } \{ { \scriptstyle { \frac { 1 } { 4 } } \log ( 1 / r ) , \log ( e d ) } / { ( 5 1 2 Q ) } \}$ . For suficiency enlarge the constant in the upper bound to $C _ { \delta } \geq 1$ and take

$$
\begin{array} { r } { n \geq 8 C _ { \delta } d / r , \qquad \log K \geq d \{ \log ( e d ) + \frac { 1 } { 2 } \log ( 4 C _ { \delta } / r ) \} . } \end{array}
$$

The quantization term is then at most $r / ( 4 C _ { \delta } )$ and the bracket in the sampling term is at most 2. Theorem 3.2 gives risk at most $r ,$ with $n = O _ { \delta } ( d / r )$ and $b = O _ { \delta } ( d \operatorname* { m a x } \{ \log ( 1 / r ) , \log ( e d ) \} )$ . Rounding the sample size and bit budget up to integers changes neither order.

## C Reliability and working memory

Throughout this appendix, $L = 1 + \ln ( 1 / \delta ) , U = 1 + \ln L$ , and $V = 1 + \ln U$ . The testing arguments allow randomized initial states, randomized transitions, and randomized terminal decisions. All constants denoted by C are numerical unless their dependence is indicated.

## C.1 Reduction to monotone tests with finite memory

Lemma C.1 (Monotone reduction). Let $P _ { 0 } = N ( u _ { 0 } , 1 )$ and $P _ { 1 } = N ( u _ { 1 } , 1 )$ , where $u _ { 0 } < u _ { 1 }$ . For every K-state test of n independent observations and every $\lambda > 0$ , there is a deterministic K-state test with no larger weighted error

$$
P _ { 1 } ( D = 0 ) + \lambda P _ { 0 } ( D = 1 ) .
$$

Its efective states at each time can be ordered so that every transition is nondecreasing in both the previous state and the current observation, and the terminal decision is nondecreasing in the final state. Consequently, if the original test has power at least $c > 0$ and false positive probability $\xi \in ( 0 , 1 )$ the monotone test can be chosen with power at least $c / 2$ and false positive probability at most $2 \xi / c$ The same conclusions hold for independent, nonidentically distributed observations whose individual likelihood ratios are nondecreasing.

Proof. First fix an initial state whose weighted error does not exceed the average over the initial randomization. Modify the transitions in chronological order. Suppose that the preceding transitions have already been fixed and their efective states are ordered by the prefix likelihood ratios

$$
\Lambda ( s ) = { \frac { P _ { 1 } ( S _ { t - 1 } = s ) } { P _ { 0 } ( S _ { t - 1 } = s ) } } .
$$

Both probabilities are positive because the two experiments are equivalent. For each possible destination $j ,$ let $a _ { j }$ and $b _ { j }$ be, respectively, the type II and type I error probabilities of the current sufix program started from $j .$ . The sufix may still be randomized. Writing $f _ { i }$ for the current observation density, the contribution of choosing destination $j$ at $( s , x )$ is proportional to

$$
z ( s , x ) a _ { j } + b _ { j } , \qquad z ( s , x ) = \lambda ^ { - 1 } \Lambda ( s ) \frac { f _ { 1 } ( x ) } { f _ { 0 } ( x ) } .
$$

A randomized transition is a convex combination of these quantities. Replace it by a pointwise minimizing destination.

The lower envelope of finitely many afine functions $a _ { j } z + b _ { j }$ admits an ordered partition of $( 0 , \infty )$ into intervals on which a minimizing index is constant. Delete unnecessary functions and resolve each tie consistently. Ordering the destinations from left to right makes the selected index $J ( z )$ nondecreasing. Thus $J ( z ( s , x ) )$ is nondecreasing in both s and x. Under $P _ { 0 }$ , let $Q$ denote the joint law of $( s , x )$ . For a destination interval $I _ { j }$ , its new prefix likelihood ratio is

$$
\frac { P _ { 1 } ( J ( z ) = j ) } { P _ { 0 } ( J ( z ) = j ) } = \lambda \mathbb { E } _ { Q } [ z \mid z \in I _ { j } ] .
$$

The conditional means over distinct ordered intervals of positive probability are strictly increasing. Indeed, independent draws from two such intervals are ordered almost surely, and equality of their means would force both draws to equal the same constant, contradicting the disjoint partition. Hence the new state ordering also agrees with its prefix likelihood ratios. Relabel the sufix entrance states, freeze this transition, and continue. Subsequent modifications cannot change the probabilities of an already fixed prefix. No step increases the number of states or the weighted error.

At the final time choose $D = \mathbf { 1 } \{ \Lambda ( S _ { n } ) \geq \lambda \}$ . This also minimizes the weighted error and is nondecreasing. Gaussian likelihood ratios are strictly increasing in $x ,$ so the induction applies. It uses only independence and the current monotone likelihood ratio, proving the nonidentically distributed version as well. Finally, set $\lambda = c / ( 2 \xi )$ . The original weighted error is at most $1 - c / 2$ , so each of the two nonnegative terms for the new test is at most this value. Its power is therefore at least $c / 2$ , and its false positive probability is at most $( 2 - c ) \xi / c \leq 2 \xi / c .$ □

## C.2 A weighted certificate bound

For a nonconstant monotone function $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ and positive weights $w _ { i } .$ , a one-certificate is a set of coordinates whose being one forces $f = 1 ;$ a zero-certificate is defined analogously. Let $u ( f )$ and $v ( f )$ be the minimum total weights of these certificates.

Lemma C.2 (Certificate area and disjoint certificates). Suppose $f$ is computed by a deterministic monotone read-once program with at most K states at every time, and put $\begin{array} { r } { S = \sum _ { i } w _ { i } ^ { 2 } } \end{array}$ . Then

$$
u ( f ) v ( f ) \leq K S .\tag{C.1}
$$

Moreover, writing $u = u ( f )$ , there are at least $u ^ { 2 } / ( 4 K S )$ pairwise disjoint zero-certificates, each of

weight at most $2 K S / u . ~ H Q$ is a product distribution with $Q ( X _ { i } = 0 ) = e ^ { - w _ { i } }$ , then

$$
Q ( f = 1 ) \leq \exp \left\{ - \frac { u ^ { 2 } } { 4 K S } \exp \left( - \frac { 2 K S } { u } \right) \right\} .\tag{C.2}
$$

If P is a product distribution with $P ( X _ { i } = 1 ) = e ^ { - w _ { i } }$ , then $P ( f = 1 ) \geq e ^ { - u }$

Proof. Associate to each nonconstant residual function g the rectangle $[ 0 , u ( g ) ] \times [ 0 , v ( g ) ]$ ; constant residual functions have empty rectangles. For the two children $g _ { 0 } \le g _ { 1 }$ at coordinate $i ,$ use the conventions $( u ( 0 ) , v ( 0 ) ) = ( \infty , 0 )$ and $( u ( 1 ) , v ( 1 ) ) = ( 0 , \infty )$ . Monotonicity gives

$$
u ( g ) = \operatorname * { m i n } \{ u ( g _ { 0 } ) , w _ { i } + u ( g _ { 1 } ) \} , \qquad v ( g ) = \operatorname * { m i n } \{ w _ { i } + v ( g _ { 0 } ) , v ( g _ { 1 } ) \} .
$$

Apart from the coordinate axes, any point of the parent rectangle not covered by either child rectangle belongs to

$$
( u ( g _ { 1 } ) , u ( g _ { 1 } ) + w _ { i } ] \times ( v ( g _ { 0 } ) , v ( g _ { 0 } ) + w _ { i } ] .
$$

Its area is at most $w _ { i } ^ { 2 }$ . This includes constant children, since $u ( g _ { 1 } )$ and $v ( g _ { 0 } )$ remain finite for a nonconstant parent. Taking the union of rectangles in each layer, its area can increase by at most $K w _ { i } ^ { 2 }$ when the layer is traversed backward. The terminal union is empty. Summing proves (C.1).

For the disjoint certificates, let I be the union of those already selected. While its weight is below $u / 2$ , fix all coordinates in I to one. The restricted function g is nonconstant and has $u ( g ) \geq$ $u - w ( I ) > u / 2$ . It is still computed by a monotone program with at most $K$ states: simply compose the transitions for the fixed coordinates. By (C.1), it has a zero-certificate of weight at most $2 K S / u .$ disjoint from I. It is also a zero-certificate for $f ,$ since lowering the fixed coordinates cannot increase its output. Repeat until $w ( I ) \geq u / 2$ . Each new certificate is nonempty, so the procedure terminates and selects at least $u ^ { 2 } / ( 4 K S )$ certificates.

Under $Q ,$ each selected certificate is all zero with probability at least $\exp ( - 2 K S / u )$ , and these events are independent. If any occurs, $f = 0$ . Their product bound gives (C.2). Under $P ,$ , a minimum one-certificate is all one with probability $e ^ { - u }$ , which proves the final assertion. □

$$
\operatorname { I f } Q ( f = 1 ) \geq c \mathrm { ~ f o r ~ } 0 < c < 1 , \ : H = \ln ( 1 / c ) , \ : K S > H , \ : \mathrm { a n d } \ : t = 2 K S / u \geq 1 , \ : \mathrm { t h e n } 
$$

$$
u \leq \frac { 6 K S } { \ln ( K S / H ) } .\tag{C.3}
$$

Indeed, (C.2) yields $K S e ^ { - t } / t ^ { 2 } \le H$ , so l $\mathrm { n } ( K S / H ) \le t + 2 \ln t \le 3 t$ . When all $w _ { i } ~ \ge ~ \ln 2$ , the condition $t \geq 1$ is automatic, because $\begin{array} { r } { u \leq \sum _ { i } w _ { i } \leq S / \ln 2 } \end{array}$

## C.3 Gaussian testing with limited working memory

Lemma C.3 (Testing bound in terms of the number of states). Let $u _ { 1 } - u _ { 0 } = h > 0$ , and let a randomized K-state test of n independent $N ( u _ { i } , 1 )$ observations have power at least $c \in ( 0 , 1 ]$ under $i = 1$ . Define

$$
A _ { h } = 2 ( \ln 2 ) ^ { 2 } + 2 h ^ { 2 } ( 1 + h ^ { 2 } / 4 ) , \qquad N ( c ) = \left\lceil \operatorname* { m a x } \left\{ e ^ { 2 } , \left( { \frac { \ln ( 4 / c ) } { ( \ln 2 ) ^ { 2 } } } \right) ^ { 2 } \right\} \right\rceil .
$$

For every $n \geq N ( c )$

$$
P _ { 0 } ( D = 1 ) \geq { \frac { c ^ { 2 } } { 1 6 } } \exp \left\{ - { \frac { 9 6 A _ { h } } { c } } { \frac { K n } { \ln n } } \right\} .\tag{C.4}
$$

The bound remains valid for independent Gaussian observations with positive mean diferences bounded by h.

Proof. Write $\xi = P _ { 0 } ( D = 1 )$ ). Equivalence of the experiments rules out $\xi = 0$ , and $\xi = 1$ is immediate. By Lemma C.1, replace the test by a deterministic monotone test with power at least $c / 2$ and false positive probability at most $2 \xi / c .$

Center each observation at $m = ( u _ { 0 } + u _ { 1 } ) / 2$ and write $X _ { i } - m = ( 2 B _ { i } - 1 ) R _ { i }$ , where $R _ { i } = | X _ { i } - m |$ The vector of magnitudes has the same law M under both hypotheses. Conditional on this vector, the signs are independent, with

$$
P _ { 0 } ( B _ { i } = 1 \mid R _ { i } = r _ { i } ) = \frac { 1 } { 1 + e ^ { h r _ { i } } } , \qquad P _ { 1 } ( B _ { i } = 0 \mid R _ { i } = r _ { i } ) = \frac { 1 } { 1 + e ^ { h r _ { i } } } .
$$

For a fixed magnitude vector, the conditional test is a monotone binary read-once program of width at most $K$ . Conditioning only fixes the transition functions for the analysis; it supplies no additional stored information to the algorithm.

Set $\begin{array} { r } { w _ { i } = \ln ( 1 + e ^ { h r _ { i } } ) , S ( \mathbf { r } ) = \sum _ { i } w _ { i } ^ { 2 } } \end{array}$ , and let $q ( \mathbf { r } )$ be the conditional power of the monotone test. Since $\mathbb { E } _ { \mathsf { M } } q \ge c / 2$ and $0 \leq q \leq 1 , \mathsf { M } \{ q \geq c / 4 \} \geq c / 4$ . Also

$$
w _ { i } \geq \ln 2 , \qquad \mathbb { E } _ { \mathsf { M } } w _ { i } ^ { 2 } \leq 2 ( \ln 2 ) ^ { 2 } + 2 h ^ { 2 } \mathbb { E } R _ { i } ^ { 2 } = A _ { h } .
$$

Markov’s inequality therefore gives a set B of magnitude vectors with mass at least $c / 8$ on which

$$
q ( \mathbf { r } ) \geq c / 4 , \qquad n ( \ln 2 ) ^ { 2 } \leq S ( \mathbf { r } ) \leq 8 A _ { h } n / c .
$$

For a nonconstant conditional test on this set, apply (C.3) with $H = \ln ( 4 / c )$ . The definition of $N ( c )$ implies $\begin{array} { r } { \ln ( K S / H ) \ge \frac { 1 } { 2 } } \end{array}$ ln n. A minimum one-certificate then has weight at most

$$
u \leq \frac { 1 2 K S } { \ln n } \leq \frac { 9 6 A _ { h } } { c } \frac { K n } { \ln n } .
$$

Under hypothesis zero, this certificate occurs with probability $e ^ { - u }$ . The test that always returns one satisfies the same lower bound, and a test that always returns zero cannot have the required conditional power. Integrate over $\boldsymbol { B }$ using the common magnitude law, and then use the upper bound $2 \xi / c$ for the monotone test’s false positive probability. This gives $\left( \mathrm { C . 4 } \right)$

For nonidentically distributed observations, use their respective midpoints and diferences $h _ { i } \in$ $( 0 , h ]$ . Monotone reduction still applies, the magnitude law is common under the hypotheses, and the same argument uses $\mathbb { E } w _ { i } ^ { 2 } \le A _ { h }$ □

Lemma C.4 (Efective sample size). For every fixed $c \in ( 0 , 1 ]$ there is $C _ { c } < \infty$ such that, for $0 < h = u _ { 1 } - u _ { 0 } \leq 1$ and every $n , K \geq 1$ , a K-state Gaussian test with power at least c satisfies

$$
- \ln P _ { 0 } ( D = 1 ) \le C _ { c } K \frac { n h ^ { 2 } + 1 } { \ln ( n h ^ { 2 } + 1 + e ) } .\tag{C.5}
$$

Proof. Partition the observations into $N \ = \ \lceil n / m \rceil$ nonempty blocks of size $m _ { j } \ \leq \ m \ = \ \lceil h ^ { - 2 } \rceil$ and let $Z _ { j } = m _ { j } ^ { - 1 / 2 } \sum _ { i \mathrm { i n b l o c k } j } X _ { i }$ . These are independent Gaussians with unit variance with mean diferences $h { \sqrt { m _ { j } } } \leq { \sqrt { 2 } }$ . The original test can be simulated by a randomized K-state program reading the $Z _ { j } \mathrm { ' s }$ . On receiving $Z _ { j }$ , generate independent standard Gaussians $G _ { 1 } , \ldots , G _ { m _ { j } }$ and temporarily reconstruct

$$
X _ { i } ^ { \prime } = Z _ { j } / { \sqrt { m _ { j } } } + G _ { i } - { \overline { { G } } } .
$$

Under either hypothesis, this vector has the required mean and identity covariance; joint Gaussianity gives the original independent block distribution. Execute its original transitions in temporary computation and retain only the state at the end of the block. Thus the simulation preserves both testing errors without increasing persistent memory.

When $N \geq N ( c )$ , Lemma C.3, with $h = { \sqrt { 2 } } .$ 2, g $\mathrm { i v e s } \ - \ln P _ { 0 } ( D = 1 ) \le C _ { c } K N / \ln ( N + e )$ after adjusting the constant. When $N < N ( c )$ , binary data processing for relative entropy gives

$$
\frac { n h ^ { 2 } } { 2 } \geq c \ln \frac { 1 } { P _ { 0 } ( D = 1 ) } - \ln 2 .
$$

Here binary data processing follows by applying Jensen’s inequality to the likelihood ratio separately on the acceptance event and its complement; the binary entropy is at most ln 2. Since $n h ^ { 2 } \leq 2 N$ and N is bounded in this case, the same bound follows with a larger $C _ { c }$ . Finally, $N \leq n h ^ { 2 } + 1$ and $x / \ln ( x + e )$ is increasing for $x \geq 1$ , proving (C.5). □

## C.4 The joint risk lower bound and collapse

Theorem C.5 (A joint confidence lower bound). There is an absolute $c > 0$ such that, for $0 < \delta \leq$ $1 / 1 6$

$$
\mathcal { R } _ { n , K , \delta } ^ { \mathrm { s t r } } ( 1 ) \geq c \operatorname* { m i n } \left\{ 1 , K ^ { - 2 } + \frac { L } { n } + \frac { L \log ( e + L ) } { n K } \right\} .
$$

The bound allows randomized procedures with transitions that vary over time and randomized final decoding.

Proof of Theorem C.5. Fix a uniformly safe program and write $r = \operatorname* { s u p } _ { | \mu | \leq 1 } \mathbb { E } _ { \mu } ( \widehat { w } - \mu ) ^ { 2 } / 2$ . The case $r \ge 1 / 8$ is absorbed into the numerical constant. Otherwise take $h = \sqrt { 8 r } < 1$ . The elementary statistical and quantization lower bounds give $r \geq 1 / ( 3 2 n ) > 0$ and $r \geq 1 / ( 1 6 K ^ { 2 } )$ . For completeness, the first follows by testing $\mu = \pm 1 / ( 2 \sqrt { n } )$ : their total variation is at most $1 / 2$ , and an incorrect sign incurs loss at least $1 / ( 8 n )$ . For the second, replace each randomized terminal output by its conditional mean given its state, which can only reduce squared risk. The resulting codebook has at most K points. Integrating squared error to the nearest codepoint over the uniform distribution on $[ - 1 , 1 ]$ gives risk at least $1 / ( 6 K ^ { 2 } )$ , and hence the stated weaker bound. This reduction is used only for the lower bound, so it need not preserve safety.

Let $E = \{ \widehat { w } \neq 0 \}$ and $s = \ln ( 1 / \delta )$ . Safety at $\mu = 0$ implies $P _ { 0 } ( E ) \leq e ^ { - s }$ . At $\mu = h$ , a zero boutput incurs loss $h ^ { 2 } / 2 = 4 r .$ , so $P _ { h } ( E ) \geq 3 / 4$ . Changing the terminal decoder to 1 yields a test with the same state budget. For the full observations and all independent program randomization, the likelihood ratio of mean h against mean zero is

$$
\Lambda = \exp \left\{ h \sum _ { i = 1 } ^ { n } X _ { i } - \frac { n h ^ { 2 } } 2 \right\} , \qquad \mathbb { E } _ { 0 } \Lambda ^ { 2 } = e ^ { n h ^ { 2 } } .
$$

Cauchy–Schwarz gives

$$
\frac { 3 } { 4 } \leq P _ { h } ( E ) \leq \{ \mathbb { E } _ { 0 } \Lambda ^ { 2 } P _ { 0 } ( E ) \} ^ { 1 / 2 } , \qquad n h ^ { 2 } \geq s - 2 \ln ( 4 / 3 ) .
$$

Since $s \geq$ ln 16, this implies $r \geq L / ( 3 2 n )$ and $x : = n h ^ { 2 } + 1 = 8 n r + 1 \geq s$ . Lemma C.4, with $c = 3 / 4$ now yields

$$
s \leq C K \frac { x } { \ln ( x + e ) } , \qquad x \geq \frac { s \ln ( s + e ) } { C K } = : z .
$$

The target term $T = L \ln ( e + L ) / ( n K )$ is at most $4 C z / n . \mathrm { ~ I f ~ } z \ge 2$ , then $r \geq z / ( 1 6 n ) \geq T / ( 6 4 C )$ . If $z < 2 .$ , use $r \geq 1 / ( 3 2 n ) \geq T / ( 2 5 6 C )$ . Combining this with the quantization and $L / n$ bounds proves

$$
r \geq c _ { * } \operatorname* { m i n } \left\{ 1 , K ^ { - 2 } + \frac { L } { n } + \frac { L \ln ( e + L ) } { n K } \right\}
$$

with a numerical $c _ { * } > 0$ . The argument is uniform in the program, so taking its infimum proves the theorem. □

The following form makes the consequence of insuficient state growth explicit.

Proposition C.6 (Collapse to the baseline). If $K _ { n } \geq 1$ and $\delta _ { n } \in ( 0 , 1 )$ satisfy

$$
{ \frac { \ln ( 1 / \delta _ { n } ) \ln n } { K _ { n } n } } \longrightarrow \infty ,\tag{C.6}
$$

then, uniformly over all $K _ { n } – s t a t e$ uniformly δ<sub>n</sub>-safe programs, $P _ { \mu } ( \widehat { w } \neq 0 )  0$ for every fixed $\mu \in ( 0 , 1 ]$ Consequently, $\mathcal { R } _ { n , K _ { n } , \delta _ { n } } ^ { \mathrm { s t r } } ( 1 ) \to 1 / 2$

Proof. For any fixed power threshold $c > 0 .$ , apply Lemma C.3 to $D = { \bf 1 } \{ \widehat { w } \neq 0 \}$ and the mean pair $( 0 , \mu )$ . Its constants are independent of the selected program and of $K _ { n }$ b. If its power were at least c, safety would imply

$$
\ln ( 1 / \delta _ { n } ) \leq C _ { \mu , c } \frac { K _ { n } n } { \ln n } + \ln ( 1 6 / c ^ { 2 } ) ,
$$

contradicting (C.6) for suficiently large n. Since c is arbitrary, the asserted uniform convergence follows. At $\mu = 1$ , risk is at least $P _ { 1 } ( \widehat { w } = 0 ) / 2$ , whereas the program that always returns zero has worst-case risk exactly $1 / 2$ □

## C.5 A concrete validation construction at fixed target risk

At a fixed target risk, a pilot estimate of fixed resolution supplies a finite collection of candidates.   
The construction retains the pilot while reusing a workspace to validate its sign and magnitude.

Lemma C.7 (A local vote with adjustable accuracy). Fix $0 < a \leq 1 , w \in [ a , \operatorname* { m i n } ( 2 a , 1 ) ] , \sigma \in \{ - 1 , 1 \}$ and $r \geq 1$ . There is a deterministic vote using at most $1 3 1 r / a$ states and $\lceil 8 1 9 2 r / a ^ { 2 } \rceil$ observations such that

$$
\begin{array} { r } { \sigma \mu \leq w / 2 \quad \Longrightarrow \quad P _ { \mu } ( Z = 1 ) \leq e ^ { - r } , } \\ { \sigma \mu \geq 3 w / 4 \quad \Longrightarrow \quad P _ { \mu } ( Z = 0 ) \leq e ^ { - r } . } \end{array}
$$

The parameters $w , \sigma$ are fixed inputs to this subroutine; storing their labels will be charged separately.

Proof. Form increments $Y _ { i } = 2 1 \{ \sigma X _ { i } \geq 5 w / 8 \} - 1$ , and run their random walk from zero with absorbing boundaries $\pm b ,$ where $b = \lceil 6 4 r / a \rceil$ . After the fixed length $m = \lceil 8 1 9 2 r / a ^ { 2 } \rceil$ , accept only if the positive boundary was reached. There are $2 b + 1 \leq 1 3 1 r / a$ states.

Write $d = \mathbb { E } Y _ { i }$ . Under the two alternatives in the lemma, respectively, $d \leq - a / 1 6$ and $d \geq a / 1 6 ,$ since $2 \Phi ( a / 8 ) - 1 \geq a / 1 6$ . For $d > 0$ , let $\theta = \ln ( ( 1 + d ) / ( 1 - d ) ) \geq 2 d$ . The process exp $\textstyle \cdot ( - \theta \sum _ { i < k } Y _ { i } )$ Pis a nonnegative martingale. Stopping at the first visit to −b or at time m bounds the probability of reaching the wrong boundary by $e ^ { - 2 d b } \leq e ^ { - 8 r }$ . Reflecting the walk proves the corresponding bound for $d < 0$ . Under $d \geq a / 1 6$ , outside the event of reaching the negative boundary, nonacceptance implies that the unabsorbed walk ends below b. Nonacceptance is therefore contained in the union of these two events. As md $- b \geq 4 4 7 r / a$ and $m \leq 8 1 9 3 r / a ^ { 2 }$ , Hoefding’s inequality bounds this probability by

$$
\exp \left\{ - \frac { ( m d - b ) ^ { 2 } } { 2 m } \right\} \leq e ^ { - 1 2 r } .
$$

Thus the two errors are at most $e ^ { - 8 r } + e ^ { - 1 2 r } \leq e ^ { - r }$ . Absorbed paths ignore the remaining observations until the prescribed block boundary; disjoint blocks consequently yield independent votes. □

Lemma C.8 (Candidate validation at fixed resolution). Let $A = 1 6 3 8 4$ and let B be the numerical constant for mean squared error in Lemma A.1. Put $C _ { R } = ( 2 5 B + 2 ) / 2$ . For every fixed power of two $M = 2 ^ { J } \geq 2$ , every $0 < \delta < 1$ satisfying $J \le$ ln $L _ { i }$ , and every $t \in [ V , U ]$ , there is a uniformly $\delta \mathrm { - } s a f e$ program with at most 2AMt states, at most $C _ { N } M ^ { 2 } L U / t$ observations, and worst-case risk at most $C _ { R } / M ^ { 2 }$ . Here $C _ { N }$ is numerical.

Proof. Use Lemma A.1 to produce the unshrunk pilot

$$
\begin{array} { r } { q = \Phi ^ { - 1 } \bigl ( \Pi _ { [ \Phi ( - 1 ) , \Phi ( 1 ) ] } ( k / M ) \bigr ) , \qquad k \in \{ 0 , \dots , M - 1 \} . } \end{array}
$$

It uses at most 2AM states and $3 2 A M ^ { 2 }$ observations and satisfies $\mathbb { E } _ { \mu } ( q - \mu ) ^ { 2 } \le B / M ^ { 2 }$ . Candidates with $| q | < 1 / M$ are replaced by zero. For the others, preassign consecutive validation intervals to the amplitude buckets

$$
a _ { j } = 2 ^ { j } / M , \qquad | q | \in [ a _ { j } , 2 a _ { j } ) , \qquad \beta _ { j } = 4 ^ { - j } / 1 6 , \qquad 0 \leq j < J ,
$$

including the endpoint 1 in the last bucket. A path operates only in its candidate’s bucket and ignores the other validation intervals.

For a candidate in bucket $j ,$ put $w = | q | , \sigma = \operatorname { s g n } ( q )$ . Apply Lemma C.7 with accuracy parameter $r = 4 t \colon$ each fresh vote has errors at most $e ^ { - 4 t }$ between $H _ { 0 } : \sigma \mu \leq w / 2$ and $H _ { 1 } : \sigma \mu \geq 3 w / 4$ . Its state and sample costs are bounded by $5 2 4 t / a _ { j }$ and $3 2 7 6 9 t / a _ { j } ^ { 2 }$ . With $s = \ln ( 1 / \delta )$ , set

$$
R = \operatorname* { m a x } \{ 1 , \lceil s / ( 2 t ) \rceil \} , \qquad k _ { j } = \left\lceil \frac { \ln ( R / \beta _ { j } ) } { 4 t } \right\rceil .\tag{C.7}
$$

Take the OR of $k _ { j }$ fresh votes, repeat this on R disjoint groups, and accept the candidate only if all R OR decisions are one. The candidate is otherwise replaced by zero.

Because $t \le U \le L$ , we have $R \leq L / t$ . The condition $j < J \leq$ ln L gives

$$
\ln ( R / \beta _ { j } ) \le \ln L + \ln 1 6 + 2 j \ln 2 \le 3 U , \qquad 1 \le k _ { j } \le 3 U / t \le e ^ { 2 t } ,
$$

where the final inequality follows from $t \geq V = 1 { + } \ln U$ . Under $H _ { 1 }$ , an inner OR fails with probability at most $e ^ { - 4 t k _ { j } } \leq \beta _ { j } / R$ , so the outer AND fails with probability at most $\beta _ { j }$ . Under $H _ { 0 }$ , an inner OR succeeds with probability at most $k _ { j } e ^ { - 4 t } \leq e ^ { - 2 t }$ , and independence gives total acceptance probability at most $e ^ { - 2 t R } < \delta$

These bounds imply safety after conditioning on the pilot: outputting $q \neq 0$ is harmful precisely when $\sigma \mu < w / 2$ , and its conditional acceptance probability is then at most δ. Each candidate is validated only once, so averaging over the pilot requires no union bound over buckets.

For risk, let $e = q - \mu . { \mathrm { ~ I f ~ } } | q | < 1 / M$ , the squared error after rejection is at most $2 e ^ { 2 } + 2 / M ^ { 2 }$ . If $\nu = \sigma \mu < 3 w / 4$ , then $| e | = | w - \nu | \ge w / 4$ and $| \mu | \leq 5 | e |$ , so either accepting or rejecting has squared error at most $2 5 e ^ { 2 }$ . Otherwise rejection has probability at most $\beta _ { j }$ , and the conditional squared error

is at most

$$
e ^ { 2 } + \beta _ { j } \mu ^ { 2 } \leq 3 e ^ { 2 } + 8 a _ { j } ^ { 2 } \beta _ { j } = 3 e ^ { 2 } + \frac { 1 } { 2 M ^ { 2 } } .
$$

Averaging proves worst-case risk at most $( 2 5 B + 2 ) / ( 2 M ^ { 2 } )$

We now charge all persistent information. A bucket contains at most $4 M a _ { j }$ pilot labels. For $a _ { j } < 1 / 2$ , this follows because the two Gaussian probability intervals corresponding to the positive and negative buckets each have length at most $a _ { j }$ , and each contains at most $M a _ { j } + 1$ grid points. For the last bucket, the total number of labels is simply at most $M \leq 4 M a _ { j }$ , including labels clipped to the endpoints. Each active label is paired with the walk state and two Boolean flags for the inner OR and outer AND. The total state count during validation is therefore at most

$$
\left( 4 M a _ { j } \right) 4 \frac { \mathrm { 5 2 4 } t } { a _ { j } } + M + 1 \leq 8 3 8 6 M t < 2 A M t .
$$

The added $M + 1$ states cover inactive labels and zero. Block indices, bucket indices, and positions in the logical schedule are known from the public clock. At a boundary, the same transition releases the workspace and retains the candidate label. Thus pilot and validation stages reuse the same alphabet; their state counts are not added.

Finally, the number of observations for bucket j is at most

$$
R k _ { j } \frac { 3 2 7 6 9 t } { a _ { j } ^ { 2 } } \leq 9 8 3 0 7 M ^ { 2 } \frac { L U } { t } 4 ^ { - j } .
$$

Summing the geometric series and adding $3 2 A M ^ { 2 }$ pilot observations gives the claimed numerical $C _ { N }$ All block lengths and all boundaries in (C.7) are fixed before observing the data. □

## C.6 Matching tradeofs at fixed risk

Proof of Theorem 6.1. For the lower bound, fix $\rho \in ( 0 , 1 / 2 )$ and put $c = 1 - 2 \rho > 0$ . Every uniformly safe program with worst-case risk at most ρ gives the test $D = { \bf 1 } \{ \widehat { w } \neq 0 \}$ satisfying $P _ { 1 } ( D = 1 ) \geq c$ and $P _ { 0 } ( D = 1 ) \leq e ^ { - s }$ , where $s = \ln ( 1 / \delta )$ b. Cauchy–Schwarz with the Gaussian likelihood ratio gives $n \geq s + 2$ ln c. For s suficiently large depending only on $\rho ,$ this implies $n \geq s / 2 \geq N ( c )$ . Lemma C.3 then gives

$$
s \leq C _ { 1 , c } \frac { K n } { \ln n } + \ln ( 1 6 / c ^ { 2 } ) , \qquad n \geq \frac { s \ln n } { 2 C _ { 1 , c } K } .
$$

Increasing the same lower threshold on s ensures $s \geq L / 2$ and ln $n \geq U / 2$ . Hence

$$
n \geq L / 4 , \qquad n \geq L U / ( 8 C _ { 1 , c } K ) ,
$$

which proves the lower bound for every K.

For the upper bound, choose a power of two

$$
M _ { \rho } \geq \operatorname* { m a x } \{ 2 , \sqrt { 2 C _ { R } / \rho } \} , \qquad D _ { \rho } = 2 A M _ { \rho } .
$$

Decrease $\delta _ { \rho }$ if needed so that $\log _ { 2 } M _ { \rho } \leq$ ln L whenever $\delta \le \delta _ { \rho }$ . For $K \ge D _ { \rho } V$ , let

$$
t = \operatorname* { m i n } \{ U , K / D _ { \rho } \} \in [ V , U ] .
$$

Lemma C.8 supplies a safe program with at most $D _ { \rho } t \leq K$ states, risk at most $\rho / 2$ , and observation

count at most

$$
C _ { N } M _ { \rho } ^ { 2 } \frac { L U } { t } = C _ { N } M _ { \rho } ^ { 2 } L \operatorname* { m a x } \{ 1 , D _ { \rho } U / K \} \leq C _ { \rho } L \operatorname* { m a x } \{ 1 , U / K \} .
$$

Absorb integer rounding into $C _ { \rho }$ to obtain the upper bound.

Proof of Corollary 6.2. The preceding test reduction at means zero and one has fixed power $c = 1 - 2 \rho$ For $\delta _ { n } = e ^ { - \gamma n }$ , Lemma C.3 forces

$$
\gamma n \leq C _ { 1 , c } \frac { K n } { \ln n } + \ln ( 1 6 / c ^ { 2 } ) .
$$

Thus $K \geq c _ { \rho } \gamma$ ln n for all suficiently large n.

For suficiency, apply Theorem 6.1 with ${ L _ { n } } \ = \ 1 + \ \gamma n$ and $U _ { n } ~ = ~ 1 + \ln L _ { n }$ . Choose $K _ { n }$ to be the smallest power of two at least max $\{ U _ { n } , D _ { \rho } V _ { n } \}$ . Then $K _ { n } = O _ { \rho , \gamma } ( \ln n )$ , and the theorem’s construction uses at most $C _ { \rho } L _ { n }$ observations. Choose $\gamma _ { 0 } ( \rho ) \leq 1 / ( 2 C _ { \rho } )$ . For every fixed $0 < \gamma \leq \gamma _ { 0 } ( \rho )$ and suficiently large $n ,$ its required sample count is at most n. Ignore any remaining observations. Together with the lower bound, this gives $K _ { \mathrm { m i n } } = \Theta _ { \rho , \gamma } ( \ln n )$ and hence $b _ { \operatorname* { m i n } } = \log _ { 2 } \ln n + O _ { \rho , \gamma } ( 1 )$ □

## C.7 Confidence costs in fixed dimension

Proof of Corollary 6.3. Reduction for the lower bound. Restrict a d-dimensional procedure to means $\mu \ = \ \theta e _ { 1 } , \ \lvert \theta \rvert \ \leq \ 1$ . Given a scalar observation $Y _ { t } ~ \sim ~ N ( \theta , 1 )$ , generate an independent $Z _ { t } ~ \sim$ $N _ { d - 1 } ( 0 , I _ { d - 1 } )$ using fresh randomness and feed $( Y _ { t } , Z _ { t } )$ to the procedure. For $d = 1$ , this is the original scalar observation. The simulated procedure keeps the same K persistent states. If its vector output is W, return $W _ { 1 }$ . Pointwise,

$$
\begin{array} { r } { \frac 1 2 ( W _ { 1 } - \theta ) ^ { 2 } \leq \frac 1 2 \| W - \theta e _ { 1 } \| ^ { 2 } , \qquad R _ { \theta e _ { 1 } } ( 0 ) = \theta ^ { 2 } / 2 . } \end{array}
$$

The scalar unsafe event is therefore contained in the vector unsafe event, and scalar risk is bounded by vector risk. Thus $N _ { \rho } ^ { ( d ) } ( K , \delta ) \ge N _ { \rho } ( K , \delta )$ , and Theorem 6.1 supplies the lower bound for every K. Sequential validation and its state count. Let $A , C _ { R } , C _ { N }$ be the constants in Lemma C.8. Choose a fixed power of two

$$
M = 2 ^ { J } \geq \operatorname * { m a x } \{ 2 , \sqrt { 2 d C _ { R } / \rho } \} , \qquad q = M + 1 , \qquad S = 2 A M q ^ { d - 1 } .
$$

The construction in that lemma outputs either zero or one of the M pilot candidates

$$
q _ { k } = \Phi ^ { - 1 } \bigl ( \Pi _ { [ \Phi ( - 1 ) , \Phi ( 1 ) ] } ( k / M ) \bigr ) , \qquad 0 \le k < M .
$$

Hence at most q output labels sufice, independently of the confidence level. Put

$$
L _ { d } = 1 + \ln ( d / \delta ) , \qquad U _ { d } = 1 + \ln L _ { d } , \qquad V _ { d } = 1 + \ln U _ { d } .
$$

For any $t \in [ V _ { d } , U _ { d } ]$ with $J \leq \ln L _ { d }$ , run the lemma’s construction at confidence $\delta / d$ on coordinate $j ,$ using a predetermined block of

$$
m = \left\lceil C _ { N } M ^ { 2 } L _ { d } U _ { d } / t \right\rceil
$$

fresh vector observations. It reads coordinate $j$ from each observation and ignores the others. Pad each coordinate’s schedule to m observations by retaining its final label.

During block $j ,$ the completed coordinates occupy at most $q ^ { j - 1 }$ labels, and the active scalar procedure has at most 2AM t states. Their product is at most St. At the block boundary, the scalar output is encoded as its candidate label and the next scalar procedure is initialized in the released workspace. The final vector has at most $q ^ { d } \leq S t$ labels, since $q \le 2 A M$ and $t \geq 1$ . The public clock identifies the coordinate block and the local schedule. Thus all stages can be embedded in one alphabet of at most St states; no block index or separate copy of the workspace is retained. The total sample count is

$$
N = d m \leq d ( C _ { N } + 1 ) M ^ { 2 } L _ { d } U _ { d } / t .\tag{C.8}
$$

Here $M ^ { 2 } L _ { d } U _ { d } / t \geq 1$ absorbs the rounding of m.

Each scalar parameter lies in [−1, 1]. The resulting vector therefore satisfies

$$
\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { E } _ { \mu } R _ { \mu } ( W ) \leq \sum _ { j = 1 } ^ { d } C _ { R } / M ^ { 2 } \leq \rho / 2 .
$$

If every coordinate’s loss is at most its loss under the zero baseline, the sum has the same property. A union bound consequently gives

$$
\operatorname* { s u p } _ { \mu \in B _ { 2 } ^ { d } } \mathbb { P } _ { \mu } \{ R _ { \mu } ( W ) > R _ { \mu } ( 0 ) \} \leq \sum _ { j = 1 } ^ { d } \delta / d = \delta .
$$

Matching the prescribed budget. Choose $\delta _ { d , \rho } \leq \delta _ { \rho }$ small enough that $L \geq \operatorname* { m a x } \{ 1 , \ln d , e ^ { J } \}$ . Then $L _ { d } = L + \ln d \le 2 L , U _ { d } \le 2 U , V _ { d } \le 2 V$ , and $J \leq \ln L _ { d }$ . Set $D _ { d , \rho } = 2 S$ . Whenever $K \ge D _ { d , \rho } V$ choose

$$
t = \operatorname* { m i n } \{ U _ { d } , K / S \} \in [ V _ { d } , U _ { d } ] .
$$

The state count is at most $S t \leq K$ , including the final output labels. Equation (C.8) gives

$$
\begin{array} { r } { N \leq d ( C _ { N } + 1 ) M ^ { 2 } L _ { d } \operatorname* { m a x } \{ 1 , S U _ { d } / K \} } \\ { \leq 4 d ( C _ { N } + 1 ) M ^ { 2 } S L \operatorname* { m a x } \{ 1 , U / K \} . } \end{array}
$$

This proves the upper bound in (6.3) with constants depending only on $d , \rho .$ The actual integer states embed directly in the prescribed alphabet $K = 2 ^ { b }$

Exponential confidence. Write $H = d ( C _ { N } + 1 ) M ^ { 2 }$ and choose $\gamma _ { 0 } ( d , \rho ) \ : = \ : 1 / ( 2 H )$ . For fixed $0 < \gamma \leq \gamma _ { 0 } ( d , \rho )$ and $\delta _ { n } = e ^ { - \gamma n }$ , take $t = U _ { d }$ and let $K _ { n }$ be the smallest power of two at least $S U _ { d }$ For all suficiently large n, the lemma’s conditions hold and

$$
N \leq H ( 1 + \ln d + \gamma n ) \leq n , \qquad K _ { n } < 2 S U _ { d } = O _ { d , \rho , \gamma } ( \ln n ) .
$$

After the N scheduled observations, keep the final label and ignore the remaining observations. Conversely, the lower bound already proved gives, for every admissible K and all suficiently large n,

$$
n \geq c _ { \rho } ( 1 + \gamma n ) \frac { 1 + \ln ( 1 + \gamma n ) } { K } , \qquad K \geq c _ { \rho } \gamma [ 1 + \ln ( 1 + \gamma n ) ] .
$$

Together these inequalities yield the asserted state and bit asymptotics.

## C.8 Terminal compression at high confidence

Proposition C.9 (Terminal compression at high confidence). For every $n \geq 1 , K = 2 ^ { b }$ , and $0 < \delta < 1$

$$
\mathcal { R } _ { n , K , \delta } ^ { \mathrm { t e r m } } ( 1 ) \leq 8 \operatorname* { m i n } \{ 1 , L / n + K ^ { - 2 } \} .
$$

Consequently, for each fixed $\rho \in ( 0 , 1 / 2 )$ , a fixed alphabet size $K _ { \rho } \ s u f f i c e s$ for terminal risk at most $\rho$ under $\delta _ { n } = e ^ { - \gamma n }$ , for all suficiently large n and every fixed $0 < \gamma \leq \rho / 3 2$

Proof. Quantize the projection of $\overline { { X } }$ onto $[ - 1 , 1 ]$ to the nearest midpoint of K equal intervals, obtaining q with $| q - \mu | \leq | \overline { { X } } - \mu | + K ^ { - 1 }$ and $\mathbb { E } _ { \mu } ( q - \mu ) ^ { 2 } \leq 2 / n + 2 K ^ { - 2 }$ . Decode its label by soft thresholding at $\lambda = \sqrt { 2 \ln ( 2 / \delta ) / n } + K ^ { - 1 }$ . The Gaussian tail bound makes $| q - \mu | \leq \lambda$ hold with probability at least $1 - \delta$ , so Lemma B.1’s deterministic safety argument applies. Its risk calculation gives

$$
\mathbb { E } _ { \mu } R _ { \mu } ( \widehat { w } ) \leq \frac { 2 + 4 \ln ( 2 / \delta ) } { n } + 4 K ^ { - 2 } \leq 8 ( L / n + K ^ { - 2 } ) .
$$

Choosing the zero action when its bound is smaller proves the minimum. Finally choose a power of two $K _ { \rho } \ge \sqrt { 3 2 / \rho } .$ . For $\gamma \le \rho / 3 2$ , the displayed bound is at most $8 / n + \rho / 2$ , which is at most $\rho$ for suficiently large n. □