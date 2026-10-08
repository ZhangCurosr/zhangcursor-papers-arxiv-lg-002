# Policy Learning with Weak Signals

Benedikt Koch Harvard University benedikt koch@g.harvard.edu

Winston Chou Netflix wchou@netflix.com

Aur´elien Bibaut Netflix abibaut@netflix.com

Nathan Kallus Netflix nkallus@netflix.com

## Abstract

Policy learning in digital experimentation faces three challenges: weak signal-to-noise ratios, rich covariate spaces, and massive data volumes. We formalize this regime by modeling treatment-efect estimates from increasingly fine covariate partitions as Gaussian observations with bounded signal-tonoise ratios. We establish that, in general, the optimal treatment policy is not learnable in this setting. Even learning the optimal policy value sufers from impractically slow rates. However, when treatment efects vary smoothly, we derive minimax-adaptive policies based on linear smoothers that achieve vanishing welfare regret. We demonstrate the practical value of our framework by applying it to large-scale real-world experiments at Netflix, showing that personalized linearsmoothing policies can dominate unpersonalized policies even in this challenging empirical setting.

## 1 Introduction

Randomized experiments are increasingly used not only to decide whether to deploy a treatment to an entire population, but also to learn a personalized treatment policy (Manski, 2004; Hirano and Porter, 2009; Kitagawa and Tetenov, 2018; Athey and Wager, 2021). This is of special interest to digital technology firms which strive to deliver a personalized product experience to each individual user (Gomez-Uribe and Hunt, 2015; Tu et al., 2021).

However, learning efective treatment policies in digital experiments faces three challenges. First, because digital firms run many experiments, large step-change improvements are rare, so that treatment efects tend to be small relative to measurement error even with millions of users (Peysakhovich and Eckles, 2018; Azevedo et al., 2020; Bibaut et al., 2024). Second, the covari ate space is often very rich, which helps detect treatment efect heterogeneity, but increases the the number of treatment decisions to be learned. Lastly, data volume and privacy constraints often only allow storage and analysis of summary- rather than unit-level records (Chou, 2021; Wong et al., 2019).

We formalize this regime by modeling treatment-efect estimates as a Gaussian sequence model over an increasing number of covariate cells, whose signal-tonoise ratios remain bounded. We show that, in such a regime, learning the optimal personalized policy is asymptotically impossible: no decision rule yields vanishing worst-case and (least-favorable prior) Bayes regret, extending the scalar results of Hirano and Porter (2009) and Tetenov (2012). Adapting results from Cai and Low (2011), Lepski et al. (1999), and Han et al. (2020), we show that the optimal policy value is estimable, although its minimax-optimal logarithmic rate is too slow to be useful in practice.

Nevertheless, when the conditional average treatment efect (CATE) function varies smoothly (is β-H¨older) across K covariate cells, simple linear smoothers such as trees, kernels, or spline regression can borrow signal across neighboring cells and induce policies with vanishing welfare-regret. We derive sharp regret bounds for linear smoothers that trade of approximation and estimation error. Under a quasi-uniform design, the right amount of signal sharing between cells balances the two and yields the rate K<sup>−β/(2β+d)</sup> where d is the covariate dimension, matching classical rates for nonparametric regression and classification (Stone, 1982; Yang, 1999; Audibert and Tsybakov, 2007). Our theoretical contribution is to establish that tree, kernel, and spline policies adapt to unknown smoothness and attain this rate under nonuniform welfare weights and a fixed-design, heteroskedastic Gaussian observation model. We obtain a matching minimax lower bound by adapting Assouad’s lemma to weighted welfare and randomized policies.

We demonstrate the practical relevance of our methodological framework by applying it to two large-scale real-world digital experiments at Netflix. We show that, in this challenging empirical setting, treatment efect policies based on linear smoothing estimators can achieve robust welfare gains over unpersonalized policies. Our applications underscore both the importance of personalization and the empirical value of linear smoothing for learning useful treatment policies.

## 2 Framework and Notation

We begin by introducing our notational framework for policy learning with weak signals.

## 2.1 Tasks, Policies, and Welfare

Consider a collection of local decision problems, indexed by $k = 1 , \ldots , K$ , which we call tasks. Each task involves a binary treatment choice, has an associated treatment $e f f e c t \ \tau _ { k } \ \in \ \mathbb { R }$ , and a known, deterministic task feature $x _ { k } \in \mathcal X$ . Each task is also associated with a known task weight $p _ { k } > 0$ with $\textstyle \sum _ { k = 1 } ^ { K } p _ { k } = 1$

In our policy learning setting, a task is an element of a discrete covariate space $\mathcal { X } = \{ x _ { 1 } , . . . , x _ { K } \}$ , with $\tau _ { k } = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x _ { k } ]$ the conditional average treatment efect at value $x _ { k }$ , and $p _ { k } = P ( X = x _ { k } ) $ The task-level decision problem is whether to treat units having that covariate value.<sup>1</sup>

A policy $\pi = ( \pi _ { 1 } , \ldots , \pi _ { K } ) \in [ 0 , 1 ] ^ { K }$ assigns treatment probability $\pi _ { k }$ to task k and yields the welfare given the K tasks

$$
V _ { K } ( \pi ) = \sum _ { k = 1 } ^ { K } p _ { k } \pi _ { k } \pi _ { k } ,
$$

where we drop the policy independent baseline. For shortness, we write $V ( \pi ) = V _ { K } ( \pi )$ . The oracle policy $\pi _ { k } ^ { * } = \mathbf { 1 } \{ \tau _ { k } > 0 \}$ maximizes V and induces the optimal value

$$
V ^ { * } = \operatorname* { s u p } _ { \pi \in [ 0 , 1 ] ^ { K } } V ( \pi ) = \sum _ { k = 1 } ^ { K } p _ { k } \tau _ { k } ^ { + } ,\tag{1}
$$

where $\tau ^ { + } = \operatorname* { m a x } \{ \tau , 0 \}$ . The oracle policy may be nonunique when $\tau _ { k } = 0$ for some tasks, but the optimal value $V ^ { * }$ is unique (Luedtke and Van Der Laan, 2016; Shi et al., 2020; Whitehouse et al., 2025). Recovering $V ^ { * }$ , and a policy that attains ${ \mathrm { i t } } ,$ is our goal.

## 2.2 Observation Model and Decision Rules

The decision maker uses task-level data $( \widehat { \tau } _ { k } , \widehat { \sigma } _ { k } , x _ { k } , n _ { k } ) _ { k = 1 } ^ { K }$ , where $\widehat { \tau } _ { k }$ is an estimate of $\tau _ { k }$ with standard error $\sigma _ { k } / \sqrt { n _ { k } }$ , based on $n _ { k }$ units. For example, $\widehat { \tau } _ { k }$ may be the diference-in-means estimator in a randomized trial. A statistical decision rule is a measurable map $\widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K } \times \mathcal { X } ^ { K } \to [ 0 , 1 ] ^ { K }$ $( { \widehat { \tau } } , { \widehat { \sigma } } , x ) \mapsto { \widetilde { \pi } } ( { \widehat { \tau } } , { \widehat { \sigma } } , x )$ . Each coordinate may depend on the entire triplet, so rules that pool or shrink across tasks $( \mathrm { e . g . }$ , empirical Bayes) are included. Since $x$ is fixed throughout, we suppress its input.

Even if the decision maker observes unit-level data, task-level summaries can suficiently capture treatment information, in large samples under suitable conditions, see Appendix A. Such summaries are also common in industrial settings, where privacy concerns, data warehousing costs, and scale limit the storage and processing of unit-level records.

Because the welfare $V ( \widetilde { \pi } ( \widehat { \tau } , \widehat { \sigma } ) )$ of a decision rule is ran dom, we assess rules by their expected welfare $W ( \widetilde { \pi } ) =$ $\mathbb { E } [ V ( \widetilde { \pi } ( \widehat { \tau } , \widehat { \sigma } ) ) ]$ , with expectation over the randomness of $( \widehat { \tau } , \widehat { \sigma } )$ , and aim to minimize the regret

$$
V ^ { * } - W ( \widetilde { \pi } ) = \sum _ { k = 1 } ^ { K } p _ { k } \tau _ { k } \left( \mathbf { 1 } \{ \tau _ { k } > 0 \} - \mathbb { E } [ \widetilde { \pi } _ { k } ( \widehat { \tau } , \widehat { \sigma } ) ] \right) \geq 0 .
$$

We assume that the task-level estimates follow a Gaussian sequence model centered at the true efects (Peysakhovich and Eckles, 2018; Azevedo et al., 2020; Sudijono et al., 2024; Tripuraneni et al., 2024; Chen et al., 2025, 2026), with justification in Appendix A.

Assumption 1 (Gaussian Sequence Model) For $k = 1 , \ldots , K$

$$
\widehat { \tau } _ { k } \overset { \mathrm { i n d } } { \sim } \mathcal { N } \left( \tau _ { k } , \frac { \sigma _ { k } ^ { 2 } } { n _ { k } } \right) ,
$$

where the standard deviations $\sigma _ { k }$ are bounded, $i . e . , \underline { { \sigma } } \leq$ $\sigma _ { k } \le \overline { { \sigma } }$ with known constants $\underline { { \sigma } } , \overline { { \sigma } } > 0$

For expositional simplicity, we set $n _ { k } \ = \ n .$ but our results extend (up to constants) to settings where the ratios $n _ { k } / n$ are bounded away from zero and infinity by replacing $\sigma _ { k }$ with $\tilde { \sigma } _ { k } = \sigma _ { k } \sqrt { n / n _ { k } }$ below.

## 2.3 The Weak-Signal Regime

The defining feature of our setting is that treatment efects are small relative to sampling noise. Under this setting, even interventions worth deploying move the outcome of interest by an amount comparable to its standard error (Hirano and Porter, 2009; Mikusheva and Sun, 2022; Azevedo et al., 2020; Chou et al., 2025). Assumption 2 encodes this by fixing the information per task.

Assumption 2 (Weak Signal) There is a constant $H > 0$ such that $\tau _ { k } = h _ { k } / \sqrt { n }$ with sup<sub>k</sub> $| h _ { k } | \le H$ for all k.

Under Assumptions 1–2, the observation model can equivalently be written in terms of the rescaled statistics $Z _ { k } : = \sqrt { n } \widehat { \tau } _ { k }$

$$
Z _ { k } \stackrel { \mathrm { i n d } } { \sim } \mathcal N ( h _ { k } , \sigma _ { k } ^ { 2 } ) , \qquad k = 1 , \ldots , K .\tag{2}
$$

Similarly, we equivalently write decision rules as $\widetilde \pi ( Z , \widehat { \sigma } ) \ = \ \widetilde \pi ( \widehat { \tau } , \widehat { \sigma } )$ Thus, each task has bounded signal-to-noise ratio $h _ { k } / \sigma _ { k } ~ = ~ { \cal O } ( 1 )$ To return to the original scale, we target the rescaled oracle value $\begin{array} { r } { \sqrt { n } V ^ { * } = \sum _ { k = 1 } ^ { K } p _ { k } h _ { k } ^ { + } } \end{array}$

We study the regime in which the number of tasks grows, so that the covariate space becomes richer, $K \  \ \infty$ , with approximately balanced task sizes $n _ { k } \asymp n ,$ where n is large but fixed.<sup>2</sup> Thus the total sample size $\begin{array} { r } { N = \sum _ { k = 1 } ^ { K } n _ { k } \asymp K n } \end{array}$ grows as additional observations support increasingly fine partitions of $\mathcal { X } .$ but per-task signal-to-noise ratios remain bounded under Assumption 2. Our empirical analysis in Section 6 is consistent with this regime.

## 3 Policy and Value Learning

We characterize what can be learned in this weaksignal regime. We show that, in the general case without any structure linking tasks, no statistical decision rule can uniformly recover the oracle policy with vanishing worst-case regret. Nevertheless, the scalar optimal value ${ \sqrt { n } } V ^ { * }$ can still be recovered by polynomial approximation of the nonsmooth functional $h _ { k } ^ { + }$ if the standard errors are known. However, because the resulting estimator converges very slowly, learning this value is impractical even if feasible.

Let $\mathcal { P } _ { K }$ denote all joint laws of $( Z , { \widehat { \boldsymbol { \sigma } } } )$ satisfying Assumptions 1 and 2. Write $\pmb { h } \ = \ \left( h _ { 1 } , \ldots , h _ { K } \right)$ and $\pmb { \sigma } ~ = ~ \left( \sigma _ { 1 } , \dots , \sigma _ { K } \right)$ Then, an element $P _ { h , \sigma } ~ \in ~ \mathcal { P } _ { K }$ is a joint law of $( Z , { \widehat { \sigma } } )$ with Gaussian marginal $Z \sim$ $\mathcal { N } ( h , \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , . . . , \sigma _ { K } ^ { 2 } ) )$ . Thus, laws $P _ { h , \sigma } , P _ { h , \sigma } ^ { \prime } \in \mathcal { P } _ { K }$ share the same Z-marginal but may difer in the conditional law of $\widehat { \sigma }$ given Z. Define the optimal value, expected welfare and scaled regret

$$
\begin{array} { c } { { V _ { h } ^ { * } = \displaystyle \frac { 1 } { \sqrt { n } } \sum _ { k = 1 } ^ { K } p _ { k } h _ { k } ^ { + } , } } \\ { { W _ { P _ { h , \sigma } } ( \widetilde { \pi } ) = \displaystyle \frac { 1 } { \sqrt { n } } \sum _ { k = 1 } ^ { K } p _ { k } h _ { k } \mathbb { E } _ { P _ { h , \sigma } } [ \widetilde { \pi } _ { k } ( Z , \widehat { \sigma } ) ] , } } \\ { { R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } ) : = \sqrt { n } \{ V _ { h } ^ { * } - W _ { P _ { h , \sigma } } ( \widetilde { \pi } ) \} , } } \end{array}
$$

We write $\mathbb { E } _ { h , \sigma }$ instead of $\mathbb { E } _ { P _ { h , \sigma } }$ when the expectation only involves Z.

## 3.1 An Optimal Policy is not Learnable

Theorem 1 shows that learning an optimal policy incurs nonvanishing worst-case welfare regret.

Theorem 1 (Unattainable Optimal Policy) Suppose Assumptions 1 and 2 hold. Then,

$$
\begin{array} { l } { { \displaystyle \operatorname* { i n f } _ { \tilde { \pi } } \operatorname* { s u p } _ { P _ { h , \sigma } \in \mathcal { P } _ { \kappa } } R _ { K } ( P _ { h , \sigma } , \tilde { \pi } ) } } \\ { ~ } \\ { { \displaystyle = \operatorname* { m a x } _ { \Pi \in \mathfrak { P } } \operatorname* { i n f } _ { \tilde { \pi } } \int R _ { K } ( P _ { h , \sigma } , \tilde { \pi } ) \Pi ( d P _ { h , \sigma } ) } } \\ { { \displaystyle = \operatorname* { m a x } _ { 0 \le t \le H } t \Phi \left( - \frac { t } { \overline { { \sigma } } } \right) , } } \end{array}
$$

where the infima range over rules $\widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K } $ $[ 0 , 1 ] ^ { K }$ and P is the set of Borel priors on $\mathcal { P } _ { K }$ . The plug-in rule $\widetilde \pi _ { k } ^ { * } ( Z , \widehat { \sigma } ) \ = \ \mathbf { 1 } \{ Z _ { k } > 0 \}$ is both minimax and Bayes under the least-favorable prior.

The proof is given in Appendix F.1. Since n does not grow and the minimax regret value in Theorem 1 does not depend on K, the regret need not vanish as $K  \infty$ Thus, in the weak-signal regime, no decision rule achieves vanishing worst-case regret over any model class that contains the independent Gaussian submodel with bounded noise and uninformative task features. Furthermore, the minimax optimal rule ignores $\widehat { \sigma }$ and is separable, i.e., its decision for task k depends only on $Z _ { k }$

If $K = 1 , \sigma$ is known, and H is large, Theorem 1 matches the exact minimax regret in the scalar Gaussian model (Tetenov, 2012), which is also the local asymptotic minimax bound in (Hirano and Porter, 2009, Theorem 3.5). Our proof is based on a weighted Assouad argument (Lemma 1) rather than a completeclass theorem.

## 3.2 The Optimal Value is Learnable

The value of the optimal policy is learnable if we assume that the standard deviations are known. How-

ever, we show that its estimation error converges very slowly in K. Writing $\begin{array} { r } { h _ { k } ^ { + } = \frac { 1 } { 2 } ( h _ { k } + | h _ { k } | ) } \end{array}$ ,

$$
\sqrt { n } V _ { h } ^ { * } = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } p _ { k } h _ { k } + \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } p _ { k } | h _ { k } | .
$$

The first term is a smooth linear functional in h and can be estimated at parametric rate by $\textstyle { \frac { 1 } { 2 } } \sum _ { k = 1 } ^ { K } p _ { k } Z _ { k }$ The dificulty lies in the second term as $| \cdot |$ is nonsmooth at $h _ { k } = 0$ , so averaging $\lvert Z _ { k } \rvert$ leaves nonvanishing bias. Following Cai and Low (2011); Han et al. (2020), we approximate $| h _ { k } |$ by a polynomial, which we estimate in turn unbiasedly using Hermite polynomials using the known standard errors. The polynomial degree controls the tradeof between the bias from ap proximating $| h _ { k } |$ and the variance of the estimator.

Let $\begin{array} { c c l } { { Q _ { m } ^ { * } ( x ) } } & { { = } } & { { \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } x ^ { 2 j } } } \end{array}$ be the best uniform polynomial approximation of |x| on $[ - H , H ]$ of degree 2m (necessarily even). Denote with $\Theta _ { j } ( x ) =$ $( - 1 ) ^ { j } e ^ { \frac { x ^ { 2 } } { 2 } } \frac { \partial ^ { j } } { \partial x ^ { j } } \big ( e ^ { - \frac { x ^ { 2 } } { 2 } } \big )$ the probabilist’s Hermite polynomial of degree j. A key feature of $\Theta _ { j }$ is that for $Z _ { k } \ \sim \ { \mathcal N } ( h _ { k } , \sigma _ { k } ^ { 2 } )$ , we have $\mathbb { E } [ \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } ( Z _ { k } \bar { / } \sigma _ { k } ) ] = h _ { k } ^ { 2 j }$ This motivates the estimator

$$
\widehat { V } _ { m } ^ { * } : = \frac { 1 } { 2 \sqrt { n } } \sum _ { k = 1 } ^ { K } p _ { k } \left( Z _ { k } + \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right)
$$

The following result is an adaptation of Cai and Low (2011, Theorem 3 and Theorem 4).

Theorem 2 (Estimating the Optimal Value) Suppose Assumptions $1 - 2$ hold and that the true standard errors σ are known. Suppose further that max<sub>k</sub> $\begin{array} { r c l } { p _ { k } } & { = } & { O ( 1 / K ) } \end{array}$ and that $m _ { K } = \textstyle \left\lfloor { \frac { 1 } { 4 } } \log ( K ) / ( \log ( \log ( K ) ) ) \right\rfloor$ . Then,

$$
\begin{array} { r l } & { \underset { \displaystyle h \in [ - H , H ] ^ { K } } { \operatorname* { s u p } } \mathbb E _ { h , \sigma } \left[ \left( \sqrt { n } \{ \widehat { V } _ { m _ { K } } ^ { * } - V _ { h } ^ { * } \} \right) ^ { 2 } \right] } \\ & { \qquad \lesssim _ { H , \overline { { \sigma } } , \underline { { \sigma } } } \left( \frac { \log ( \log ( K ) ) } { \log ( K ) } \right) ^ { 2 } , } \end{array}
$$

This rate is minimax optimal in the submodel $p _ { k } =$ $1 / K , \sigma _ { k } = \sigma$ for all $k = 1 , \ldots , K$ for some $\sigma > 0$ 3

The proof is given in Appendix F.2. As with Theorem 1, the optimality claim covers estimators that additionally use the task features. The results in this section show that, in the weak-signal regime, the sign vector $\{ \mathbf { 1 } \{ \tau _ { k } > 0 \} \} _ { k = 1 } ^ { K }$ cannot be uniformly recovered, but its associated optimal value $\sum _ { k } p _ { k } \tau _ { k } ^ { + }$ can be estimated consistently when the number of tasks grows, even when per-task information is fixed or decreases. However, the slow logarithmic rate limits the utility of the resulting optimal estimator in practice.

## 4 Policy Learning by Linear Smoothing

In Section 3, we showed that, without restrictions link ing tasks, no decision rule can achieve vanishing worstcase welfare regret. Writing $h _ { k } = g ( x _ { k } )$ , this setting is equivalent to allowing g to vary arbitrarily over the task features. We now show that if g is smooth, one can borrow information across tasks, making treatment decisions easier while introducing an approximation bias that can be controlled.

Assumption 3 (H¨older-Smooth Signal) For fixed $\beta ~ \in ~ ( 0 , 1 ]$ and $L , H ~ > ~ 0$ , the task-level signals satisfy $h _ { k } \ = \ g ( x _ { k } )$ for every $k ~ \in ~ [ K ]$ , where $g \in \mathcal { H } _ { d } ( \beta , L , H )$ and

$$
\begin{array} { r l } & { { \mathscr { H } } _ { d } ( \beta , L , H ) : = \Big \{ g : { \mathbb R } ^ { d }  { \mathbb R } : \| g \| _ { \infty } \leq H , } \\ & { \qquad | g ( x ) - g ( y ) | \leq L \| x - y \| _ { 2 } ^ { \beta } , \forall x , y \in { \mathbb R } ^ { d } \Big \} . } \end{array}
$$

Assumption 3 is standard in nonparametric theory (Stone, 1982; Lepski et al., 1999; Audibert and Tsybakov, 2007). We restrict attention to $\beta \in ( 0 , 1 ]$ , which does not require diferentiability of the CATE.

For $g \in \mathcal { H } _ { d } ( \beta , L , H )$ , write $P _ { g , \sigma }$ for a joint law in $\mathcal { P } _ { K }$ with $h _ { k } = g ( x _ { k } )$ . Define the oracle value, expected welfare and scaled regret by, respectively,

$$
\begin{array} { l } { \displaystyle { V _ { g } ^ { * } : = \frac { 1 } { \sqrt { n } } \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) ^ { + } } } \\ { \displaystyle { W _ { P _ { g , \sigma } } ( \widetilde { \pi } ) : = \mathbb { E } _ { P _ { g , \sigma } } [ V ( \widetilde { \pi } ( Z , \widehat { \sigma } ) ) ] } } \\ { \displaystyle { R _ { K } ( P _ { g , \sigma } , \widetilde { \pi } ) : = \sqrt { n } \left\{ V _ { g } ^ { * } - W _ { P _ { g , \sigma } } ( \widetilde { \pi } ) \right\} . } } \end{array}
$$

Appendix E connects our treatment-choice problem to an auxiliary binary classification and shows that $R _ { K }$ is equivalent to the corresponding expected excess risk.

## 4.1 Linear Smoothing

We establish that linear smoothers achieve vanishing regret and are minimax optimal in our regime.

Definition 1 (Linear Smoother) Let $\boldsymbol { w } _ { j } : \mathbb { R } ^ { d } $ R be deterministic weight functions satisfying $\textstyle \sum _ { j = 1 } ^ { K } w _ { j } ( x ) = 1$ . Then

$$
\widehat { g } _ { w } ( x ) : = \sum _ { j = 1 } ^ { K } w _ { j } ( x ) Z _ { j } ,
$$

is a linear smoother for signal g and induces the (threshold) policy $\widetilde { \pi } _ { w } ( x ) = \mathbf { 1 } \{ \widehat { g } _ { w } ( x ) > 0 \}$

Definition 1 permits negative weights and ignores ${ \widehat { \sigma } } .$ We focus on linear smoothers because they are interpretable and are easily fitted to task-level data, making them desirable in practice. Moreover, for estimation, linear estimators have been shown to be minimax or near minimax optimal in certain settings (Donoho et al., 1990; Donoho and Johnstone, 1994, 1998). Un der Assumption 3, we show that rules based on linear smoothers can yield vanishing welfare regret and achieve minimax-optimal rates.<sup>4</sup>

We focus on three classes of linear smoothers: partition pooling (P), kernel smoothing (K) and series regression (S), which are summarized in Table 1. Each is indexed by a resolution parameter $r \ > \ 0$ , which controls the scale of smoothing. Precise definitions of these classes are in Appendix B.

## 4.2 Approximation–Estimation Tradeof

Linear smoothers can reduce variance at the cost of introducing bias. For a linear smoother ${ \widehat { g } } _ { w } ,$ denote its mean $\begin{array} { r } { m _ { w } ( x ) : = \mathbb { E } _ { g , \sigma } [ \widehat { g } _ { w } ( x ) ] = \sum _ { k = 1 } ^ { K } \bar { w _ { k } } ( x ) g ( x _ { k } ) } \end{array}$ and variance $\begin{array} { r } { s _ { w } ^ { 2 } ( x ) : = \mathrm { V a r } _ { g , \sigma } ( \widehat { g } _ { w } ( x ) ) = \sum _ { k = 1 } ^ { K } w _ { k } ( x ) ^ { 2 } \sigma _ { k } ^ { 2 } } \end{array}$

Proposition 1 (Linear-Smoothing Tradeof) Suppose Assumptions 1–3 hold. The policy induced by any linear smoother $\widehat { g } _ { w }$ satisfies

$$
R _ { K } ( P _ { g , \pmb \sigma } , \widetilde { \pi } _ { w } ) \leq \operatorname* { m i n } \{ H , \mathsf { A } _ { g } ( w ) + \gamma ^ { * } \mathsf { E } _ { \pmb \sigma } ( w ) \} ,
$$

where $\begin{array} { r } { \mathsf { A } _ { g } ( w ) = \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) - m _ { w } ( x _ { k } ) | } \end{array}$ is the approximation and $\begin{array} { r } { \mathsf { E } _ { \pmb { \sigma } } ( w ) = \sum _ { k = 1 } ^ { K } p _ { k } s _ { w } ( x _ { k } ) } \end{array}$ is the estimation error, and $\gamma ^ { * } = \operatorname* { s u p } _ { t > 0 } t \Phi ( - t )$ . This bound is sharp over the class of linear smoothers.

The proof is given in Appendix F.3. The two terms in Proposition 1 represent a bias–variance tradeof and formalize the costs and benefits of smoothing. As the weights become more concentrated around a task’s covariate, the smoother estimates the signal more locally and reduces approximation error, but combines fewer noisy task estimates and therefore generally increases estimation error. More difuse weights, on the contrary, yield better estimation accuracy at the cost of larger approximation error. The resolution parameter r governs this tradeof.

Under Assumption 3, the smoothers specified in Table 1 satisfy $\mathsf { \bar { A } } _ { g } ( w _ { r } ^ { T } ) \lesssim L r ^ { \beta }$ for $T \in \{ \mathrm { P } , \mathrm { K } , \mathrm { S } \}$ . To control the corresponding estimation errors, we impose the following condition on the design, which is a discrete analogue of the regularity condition of David and Semmes (1991) and implied by the strong density assumption of Audibert and Tsybakov (2007).

Let $\begin{array} { r } { Q _ { K } = \sum _ { k = 1 } ^ { K } p _ { k } \delta _ { x _ { k } } } \end{array}$ denote the task measure and let $B ( x , r ) = \smash { \big \{ } y ^ { \cdot } \in \mathbb { R } ^ { d } : \| x - y \| _ { 2 } \leq r \big \} $ be the closed Euclidean ball centered at x with radius r.

Assumption 4 (Quasi-Uniform Design) There exist constants $0 < c \leq C < \infty , a _ { 0 } , r _ { 0 } > 0$ , such that

$$
c r ^ { d } \leq Q _ { K } \left( B ( x _ { k } , r ) \right) \leq C r ^ { d }
$$

for every $k \in [ K ]$ and $r \in [ a _ { 0 } K ^ { - 1 / d } , r _ { 0 } ]$

Assumption 4 requires the welfare mass of each ball centered at a task to be proportional to its volume. The cutof at $K ^ { - 1 / d }$ is necessary because $Q _ { K }$ is discrete. Indeed, for arbitrarily small $r ,$ we have $Q _ { K } ( B ( x _ { k } , r ) ) \geq p _ { k } > 0 ;$ , so the mass cannot shrink to zero. The resolution $K ^ { - 1 / d }$ corresponds to the regular spacing of K points in d dimensions.

Theorem 3 (Regret Bounds) Suppose Assumptions 1–4 hold, and let $r \in [ a _ { 0 } \dot { K } ^ { - 1 / d } , r _ { 0 } ]$ . Let $\widetilde { \pi } _ { w _ { r } ^ { T } }$ $T \in \{ \mathrm { P } , \mathrm { K } , \mathrm { S } \}$ , be a linear smoothing policy satisfying the conditions in Appendix B.1. Then there exists $C > 0$ independent of $( K , r , g , \overline { { \sigma } } )$ , such that

$$
R _ { K } ( P _ { g , \pmb \sigma } , \widetilde { \pi } _ { w _ { r } ^ { T } } ) \leq C \left\{ L r ^ { \beta } + \frac { \overline { \sigma } } { \sqrt { K r ^ { d } } } \right\} .
$$

In particular, for $\begin{array} { r } { r _ { K } ^ { * } = \left( \frac { d \overline { { \sigma } } } { 2 \beta L \sqrt { K } } \right) ^ { \frac { 2 } { 2 \beta + d } } ; } \end{array}$

$$
R _ { K } ( P _ { g , \pmb { \sigma } } , \widetilde { \pi } _ { \pmb { w } _ { r _ { K } ^ { * } } } ) \leq C L ^ { d / ( 2 \beta + d ) } \overline { { \sigma } } ^ { 2 \beta / ( 2 \beta + d ) } K ^ { - \beta / ( 2 \beta + d ) } .
$$

The proof is given in Appendix F.4. Gaussianity (Assumption 1) is not necessary for the result; any distribution with appropriate second moment bound sufices. Without a margin condition, the rate $K ^ { - \beta / ( 2 \beta + d ) }$ matches the classical nonparametric regression rate in (Stone, 1982) and the binary classification rate in Yang (1999) and Audibert and Tsybakov (2007). Novel to the existing literature, Theorem 3 derives this rate for three concrete smoothers for welfare regret under a fixed-design, heteroskedastic Gaussian observation model with nonuniform weights.

Theorem 3 explicitly shows that

$$
\mathsf { A } _ { g } ( w _ { r } ) \lesssim L r ^ { \beta } , \qquad \mathsf E _ { \sigma } ( w _ { r } ) \lesssim \frac { \overline { \sigma } } { \sqrt { K r ^ { d } } } .
$$

Thus any sequence satisfying $r _ { K } \to 0$ and $K r _ { K } ^ { d }  \infty$ yields vanishing worst-case regret under the estimators and assumptions of Theorem 3. The oracle resolution balances the two terms. However, the oracle resolution depends on the unknown smoothness parameters $( \beta , L )$ . Nonetheless, in Proposition 3 (Appendix D), we show that cross-validation can be used to select the resolution within a suitable growing sieve of linear smoothers while preserving the rate in Theorem 3. Thus, the selected policy is minimax adaptive, mirroring similar results in nonparametric regression (Tsybakov, 2009).

<table><tr><td>Smoother</td><td> $\mathrm { W e i g h t } \ w _ { r , j } ( x )$ </td><td>Conditions</td></tr><tr><td>Partition pooling</td><td> $\underline { { p _ { j } \mathbf { 1 } \{ x _ { j } \in A _ { r } ( x ) \} } }$   $\textstyle \sum _ { \ell : x _ { \ell } \in A _ { r } ( x ) } p _ { \ell }$ </td><td>Ar a finite partition of  $\mathcal { X } ; A _ { r } ( x ) \in \mathcal { A } _ { r }$  the cell containing x;  $\begin{array} { r } { \operatorname* { m a x } _ { A \in \mathcal { A } _ { r } } \operatorname* { s u p } _ { x , y \in A } \| x - y \| _ { 2 } \leq r ; } \end{array}$ </td></tr><tr><td>Kernel smoothing</td><td> $\mathsf { K } \left( ( x _ { j } - x ) / r \right)$   $\overline { { \sum _ { \ell = 1 } ^ { K } \mathsf { K } \left( ( x _ { \ell } - x ) / r \right) } }$ </td><td> $\mathsf { K } : \mathbb { R } ^ { d } \to [ 0 , \infty )$  bounded, supported on  $B ( 0 , 1 )$  , continuous at 0,  $\mathsf { K } ( 0 ) > 0 .$ </td></tr><tr><td>Series regression</td><td> $a _ { j } b _ { r } ( x ) ^ { \top } G _ { r , a } ^ { - 1 } b _ { r } ( x _ { j } )$ </td><td> $\begin{array} { r } { a _ { k } > 0 , \sum _ { k } a _ { k } = 1 ; b _ { r } : \mathcal { X } \to \mathbb { R } ^ { q _ { r } } } \end{array}$  with  $q _ { r } \le K , q _ { r } \asymp r ^ { - d } ;$   $\begin{array} { r } { \mathrm { c o n s t a n t i s } \stackrel { \_ } { \in } \mathrm { s p a n } ( b _ { r } ) ; G _ { r , a } : = \sum _ { k } a _ { k } b _ { r } ( x _ { k } ) b _ { r } ( x _ { k } ) ^ { \top } } \end{array}$  nonsingular.</td></tr></table>

Table 1: Linear smoothers at resolution r.

## 4.3 When Linear Smoothers are Minimax Optimal

The next result shows that linear smoothers can be minimax optimal under Assumptions 3–4.

Theorem 4 (Minimax Lower Bound) Suppose Assumptions 1–4 hold. Then there exists a constant $c > 0$ such that for K suficiently large and $r \in [ 2 a _ { 0 } K ^ { - 1 / d } , r _ { 0 } / 2 ]$

$$
\begin{array} { r l } {  { \operatorname* { i n f } _ { \widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K } \to [ 0 , 1 ] ^ { K } } \operatorname* { s u p } _ { \scriptstyle { P _ { g , \sigma } \in \mathscr { P } _ { K } } } R _ { K } \big ( { P _ { g , \sigma } , \widetilde { \pi } } \big ) } \quad } & { } \\ & { \geq c \mathscr { H } _ { d } ( \beta , L , H ) } \\ & { \geq c \operatorname* { m i n } \{ H , L r ^ { \beta } , \frac { \overline { { \sigma } } } { \sqrt { K r ^ { d } } } \} . } \end{array}
$$

In particular, for $\begin{array} { r } { r _ { K } ^ { \ast } = \left( \frac { d \overline { { \sigma } } } { 2 \beta L \sqrt { K } } \right) ^ { \frac { 2 } { 2 \beta + d } } } \end{array}$ ，

$$
\begin{array} { r l } & { \underset { \pi : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K } \to [ 0 , 1 ] ^ { K } } { \operatorname* { s u p } } \underset { \substack { { P _ { g , \sigma } \in \mathcal { P } _ { K } } } } { \operatorname* { s u p } } R _ { K } ( P _ { g , \sigma } , \overline { { \pi } } ) } \\ & { \qquad \quad \geq c L ^ { d / ( 2 \beta + d ) } \overline { { \sigma } } ^ { 2 \beta / ( 2 \beta + d ) } K ^ { - \beta / ( 2 \beta + d ) } . } \end{array}
$$

The proof is given in Appendix F.5. The lower bound matches the minimax rates for binary classification (Yang, 1999; Audibert and Tsybakov, 2007) under our observation model. Combined with Theorem 3, this establishes minimax-rate optimality of the partition-, kernel-, and B-spline-smoother policies over H¨oldersmooth signals under quasi-uniform designs.

## 5 Simulation Study

We consider K = 100 equally weighted tasks on a twodimensional grid, with unit-variance Gaussian noise and a curved, Lipschitz signal $g _ { \mathrm { c r v } }$ . The signal is centered, so treat-all and treat-none both have zero welfare. Appendix C.1 provides additional detail on the simulation.

Figure 1 illustrates this regime. The observed estimates display little structure, even though the underlying CATE is highly structured. Consequently, the plug-in policy $1 \{ Z _ { k } > 0 \}$ mis-signs 45% of tasks, against 50% for a coin flip, and attains only 14% of $V _ { g _ { \mathrm { c r v } } } ^ { * }$ Comparing this pattern with Figure 2 shows that this simulation regime is relatively faithful to patterns observed in real-world data.

We consider three families of smoothers: axis-aligned trees with J leaves, tensor-product B-splines of degree two with dimension q, and kernel smoothing with an Epanechnikov kernel of bandwidth h.

The remaining panels of Figure 1 show representative smoothers at their oracle resolution predicted by Theorem 3 $( J = q = 9$ and $h = 3 )$ . Each adapt to the circular frontier and the best of them, the spline with q = 9, attains 46% of $V _ { g _ { \mathrm { c r v } } } ^ { * }$ , more than three times the value of the plug-in policy. Table 2 reports the full comparison. In Figures 5 and 6 in Appendix C, we illustrate the approximation–estimation tradeof of Proposition 1 of the smoothers in terms of their resolution. Another simulation example is discussed in Appendix C.2.

## 6 Empirical Application at Netflix

We demonstrate the empirical relevance of our framework by applying it to two large-scale real-world Netflix experiments. For both experiments, we implement the proposed methodology as follows. We discretize two pre-treatment covariates, predicted engagement level and length of subscription tenure, into ten deciles, yielding 100 distinct covariate cells. Experimental units, which are divided into treatment and control arms, are further randomized to 10 folds. We do not report the outcome metric for confidentiality reasons, but it can be interpreted as a measure of engagement frequency. For each arm-fold-covariate value, we aggregate the data to the count, sum, and sum of squares of the outcome. This allows us to collapse each experimental dataset, which consists of millions of observations, into a smaller dataset of 2,000 rows.

![](images/49823c0e3da2f296127d3a10309bbe8e6756a724d624e26296aad9b8f242268a.jpg)

![](images/c4cc533bfaa5d33469dcd621090188b37940e118497ddb7023e589a377078845.jpg)

![](images/825a3b57c49954e8675df70f75cffd6f66f04640203c81d8cf8ba34413084c3a.jpg)

![](images/e95276ecc2a15ef261d12bdea2224461c5ae73870dfa9f1c3fe58d189bbebfa4.jpg)

Figure 1: The first three panels show the true curved CATE $^ { g , }$ one draw of the rescaled estimates $Z _ { k } = \sqrt { n } \widehat { \tau } _ { k }$ and the plug-in policy $1 \{ Z _ { k } > 0 \}$ that these estimates imply. The remaining panels show representative linear smoothers. The dashed line is the true frontier at which $g = 0$ , while the solid line represents the learner’s decision boundary, $\widehat { g } _ { w } = 0$  
![](images/1e7022115072aae28a9e17aeeac267ec2c4c93435da01652608a1cfa4f38273e.jpg)  
Figure 2: The top row shows the observed CATEs and full-sample fits of a plane, tensor-product B-spline, regression tree, and local-linear Epanechnikov kernel applied to Experiment 1. Each displayed fit uses the candidate selected most often across outer folds in nested CV. The lower-left panel shows the average held-out policy gains over unpersonalized policies of the family with the lowest estimated policy loss. The remaining panels show the calibration of cross-fitted CATE predictions after centering within each fold.

Let k $\in \{ 1 , \ldots , 1 0 0 \}$ index covariate values, $f \in$ $\{ 1 , \ldots , 1 0 \}$ index folds, and $a \in \{ 0 , 1 \}$ index arms. The

![](images/bac0079ba0dab841e3eacfa7beb2e66bf28d48bcbc99271e1c0fce2e7836f170.jpg)  
Figure 3: Panels are laid out as in Figure 2, but for Experiment 2.

CATE is $\tau _ { k } = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X _ { k } = k ]$ . In fold $f ,$ we estimate $\tau _ { k }$ by $\hat { \tau } _ { f , k } = \bar { Y } _ { 1 , f , k } - \bar { Y } _ { 0 , f , k }$ and calculate the empirical cell mass $\begin{array} { r } { p _ { f , k } = ( n _ { 1 , f , k } + n _ { 0 , f , k } ) / \sum _ { i } ( n _ { 1 , f , j } + } \end{array}$ $n _ { 0 , f , j } )$ , which we use as learner weights. We fit four learner families to these 100 cell efects: a linear plane in the two decile indices, tensor-product B-splines, regression trees, and isotropic local-linear Epanechnikov kernel regression.

To jointly select the CATE learner and measure its corresponding policy value, we use nested crossvalidation. For outer fold f, candidate hyperparameter λ in family $m ,$ and inner validation fold $g \neq f ,$ we fit on the other eight folds and predict $\hat { \tau } _ { m , \lambda } ^ { ( - f , - g ) } ( k )$ . The inner-fold loss is the negative policy welfare $\begin{array} { r } { L _ { f , g } ( m , \lambda ) = - \sum _ { k = 1 } ^ { 1 0 0 } p _ { g , k } \mathbf { 1 } \{ \hat { \tau } _ { m , \lambda } ^ { ( - f , - g ) } ( k ) > } \end{array}$ $0 \} \hat { \tau } _ { g , k } ,$ motivated by Proposition 3. We select the simplest model that is within one standard error of the minimum average fold loss. We refit that candidate on all nine non-held-out folds and set the outer-fold policy to $\hat { \pi } _ { m , f } ( k ) = 1 \Big \{ \hat { \tau } _ { m } ^ { ( - f ) } ( k ) > 0 \Big \}$ Lastly, for family m and comparator $c = 0$ (treat none) or $c \ = \ 1$ (treat all), we estimate the welfare gain by $\begin{array} { r } { \hat { \Delta } _ { m } ( c ) = \frac { 1 } { 1 0 } \sum _ { f = 1 } ^ { 1 0 } \sum _ { k = 1 } ^ { 1 0 0 } p _ { f , k } ( \hat { \pi } _ { m , f } ( k ) - } \end{array}$ $c ) \hat { \tau } _ { f , k }$ with variance $\begin{array} { r } { \widehat { \mathrm { S E } } _ { m } ^ { 2 } ( c ) = \frac { 1 } { 1 0 ^ { 2 } } \sum _ { f , k } p _ { f , k } ^ { 2 } ( \hat { \pi } _ { m , f } ( k ) - } \end{array}$ $c ) ^ { 2 } \big ( s _ { 1 , f , k } ^ { 2 } / n _ { 1 , f , k } + s _ { 0 , f , k } ^ { 2 } / n _ { 0 , f , k } \big )$

## 6.1 Experiment 1

This experiment allocated approximately 9.41 million users to treatment and 10.54 million users to control. This experiment modified the weights used in recommender system training to emphasize newer titles. Results are shown in Figure 2.

As the first panel of Figure 2 shows, the grid of observed CATEs over the 100 covariate values is very noisy. However, all four linear smoothing families yield well-calibrated and interpretable CATE functions that indicate that treatment is more efective for highlyengaged and less-tenured units. The selected treefamily policy significantly lifts engagement over both unpersonalized policies (treat none and treat all).

## 6.2 Experiment 2

This experiment allocated approximately 4.73 million users to treatment and 10.65 million users to control. The treatment augmented the features available for newly launching titles. Results are shown in Figure 3.

The raw CATEs, which are shown in the first panel of Figure 3, show somewhat more structure, but still with substantial noise. All model families yield a wellcalibrated and interpretable CATE estimator, which show that the treatment is more efective for moretenured but less-engaged users. The selected kernelfamily policy significantly lifts engagement over both unpersonalized policies.

## 7 Conclusion

We study the problem of policy learning in the weaksignal setting, in which treatment efects are small and on the scale of measurement error. We establish the general impossibility of learning the optimal policy and the slowness of learning its value in this regime. However, when treatment efects are smooth, linear smoothing estimators that exploit this structure can be minimax optimal. We show how to derive the optimal resolution of such learners and apply our framework to two large-scale real-world experiments at Netflix. Our results demonstrate the promise of our framework for policy learning in digital experiments.

## AI Use Statement

We used generative AI tools to assist with data analysis, code and figure generation, manuscript editing (including grammar checks, spelling errors, and improvements to clarity), the derivation of selected critical proof steps, and verifying proofs. Generative AI was not used to generate research ideas or for substantive text generation. We checked the code and results and reviewed the proofs line by line. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## References

Susan Athey and Stefan Wager. Policy learning with observational data. Econometrica, 89(1):133–161, 2021.

Jean-Yves Audibert and Alexandre B. Tsybakov. Fast learning rates for plug-in classifiers. The Annals of Statistics, 35(2):608–633, 2007. ISSN 00905364. URL http://www.jstor.org/stable/25463570.

Eduardo M Azevedo, Alex Deng, Jos´e Luis Montiel Olea, Justin Rao, and E Glen Weyl. A/b testing with fat tails. Journal of Political Economy, 128 (12):4614–4672, 2020.

SS Barsov and Vladimir V Ulyanov. Estimates of the proximity of gaussian measures. In Doklady Mathematics, volume 34, pages 462–466, 1987.

SN Bernstein. Sur la valeur asymptotique de la meilleure approximation de— x—. Acta Math, 37: 1–57, 1913.

Aur´elien Bibaut, Winston Chou, Simon Ejdemyr, and Nathan Kallus. Learning the covariance of treatment efects across many weak experiments. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 153– 162, 2024.

T. Tony Cai and Mark G. Low. Testing composite hypotheses, Hermite polynomials and optimal estimation of a nonsmooth functional. The Annals of Statistics, 39(2):1012 – 1041, 2011. doi: 10.1214/10-AOS849. URL https://doi.org/10. 1214/10-AOS849.

Jiafeng Chen, Lihua Lei, Timothy Sudijono, Liyang Sun, and Tian Xie. Compound selection decisions: An almost SURE approach, 2025. URL https:// arxiv.org/abs/2511.11862.

Jiafeng Chen, Nabarun Deb, and Nikolaos Ignatiadis. Normal approximations in nonparametric empirical bayes. arXiv preprint arXiv:2605.31599, 2026.

Winston Chou. Randomized controlled trials without data retention, 2021. URL https://arxiv.org/ abs/2102.03316.

Winston Chou, Colin Gray, Nathan Kallus, Aur´elien Bibaut, and Simon Ejdemyr. Evaluating decision rules across many weak experiments. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, pages 4365–4374, New York, NY, USA, 2025. Association for Computing Machinery. doi: 10. 1145/3711896.3737217. URL https://doi.org/ 10.1145/3711896.3737217.

G David and S Semmes. Singular integrals and rectifiable sets in R<sup>n</sup>. Au-del\`a des graphes lipschitziens Ast´erisque, 193, 1991.

Luc Devroye, Abbas Mehrabian, and Tommy Reddad. The total variation distance between highdimensional gaussians with the same mean. arXiv preprint arXiv:1810.08693, 2018.

David L Donoho and Iain M Johnstone. Minimax risk over lp-balls for lp-error. Probability theory and related fields, 99(2):277–303, 1994.

David L Donoho and Iain M Johnstone. Minimax estimation via wavelet shrinkage. The annals of Statistics, 26(3):879–921, 1998.

David L Donoho, Richard C Liu, and Brenda MacGibbon. Minimax risk over hyperrectangles, and implications. The Annals of Statistics, pages 1416–1437, 1990.

Carlos A Gomez-Uribe and Neil Hunt. The netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS), 6(4):1–19, 2015.

Yanjun Han, Jiantao Jiao, and Rajarshi Mukherjee. On estimation of L -norms in Gaussian white noise models. Probability Theory and Related Fields, 177 (3):1243–1294, 2020.

Keisuke Hirano and Jack R Porter. Asymptotics for statistical treatment rules. Econometrica, 77(5): 1683–1701, 2009.

Toru Kitagawa and Aleksey Tetenov. Who should be treated? empirical welfare maximization methods for treatment choice. Econometrica, 86(2):591–616, 2018.

Oleg Lepski, Arkady Nemirovski, and Vladimir Spokoiny. On estimation of the l r norm of a regression function. Probability theory and related fields, 113(2):221–253, 1999.

Alex Luedtke and Antoine Chambaz. Performance guarantees for policy learning. In Annales de l’IHP Probabilites et statistiques, volume 56, page 2162, 2020.

Alexander R Luedtke and Mark J Van Der Laan. Statistical inference for the mean outcome under a possibly non-unique optimal treatment strategy. Annals of statistics, 44(2):713, 2016.

Charles F Manski. Statistical treatment rules for heterogeneous populations. Econometrica, 72(4):1221– 1246, 2004.

Anna Mikusheva and Liyang Sun. Inference with many weak instruments. The Review of Economic Studies, 89(5):2663–2686, 2022.

Alexander Peysakhovich and Dean Eckles. Learning causal efects from many randomized experiments using regularized instrumental variables. In Proceedings of the 2018 World Wide Web Conference, pages 699–707, 2018.

Yury Polyanskiy and Yihong Wu. Information theory: From coding to learning. Cambridge university press, 2025.

Chengchun Shi, Wenbin Lu, and Rui Song. Breaking the curse of nonregularity with subagging— inference of the mean outcome under optimal treatment regimes. Journal of Machine Learning Research, 21(176):1–67, 2020.

Charles J Stone. Optimal global rates of convergence for nonparametric regression. The annals of statistics, pages 1040–1053, 1982.

Timothy Sudijono, Simon Ejdemyr, Apoorva Lal, and Martin Tingley. Optimizing returns from experimentation programs. arXiv preprint arXiv:2412.05508, 2024.

Aleksey Tetenov. Statistical treatment choice based on asymmetric minimax regret criteria. Journal of Econometrics, 166(1):157–165, 2012.

Nilesh Tripuraneni, Lee Richardson, Alexander D’Amour, Jacopo Soriano, and Steve Yadlowsky. Choosing a proxy metric from past experiments. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5803–5812, 2024.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, New York, NY, 1 edition, 2009. ISBN 978-0-387-79052-7. doi: 10.1007/b13794.

Ye Tu, Kinjal Basu, Cyrus DiCiccio, Romil Bansal, Preetam Nandy, Padmini Jaikumar, and Shaunak Chatterjee. Personalized treatment selection using causal heterogeneity. In Proceedings of the Web Conference 2021, pages 1574–1585, 2021.

A. W. van der Vaart. Asymptotic statistics, volume 3. Cambridge university press, 2000.

Richard S Varga and A Dzh Karpenter. On a conjecture of s. bernstein in approximation theory. Mathematics of the USSR-Sbornik, 57(2):547–560, 1987.

Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.

Justin Whitehouse, Morgane Austern, and Vasilis Syrgkanis. Inference on optimal policy values and other irregular functionals via smoothing. arXiv preprint arXiv:2507.11780, 2025.

Jefrey Wong, Randall Lewis, and Matthew Wardrop. Eficient computation of linear model treatment effects in an experimentation platform, 2019. URL https://arxiv.org/abs/1910.01305.

Yuhong Yang. Minimax nonparametric classification. i. rates of convergence. IEEE Transactions on Information Theory, 45(7):2271–2284, 1999.

# Supplementary Materials

## A Gaussian Limit Experiments

In Section 2, we formulated decision rules as functions of task-level data, $\widetilde { \pi } ( \widehat { \tau } , \widehat { \sigma } )$ , and assumed a Gaussian sequence model (Assumption 1). This can be justified by local asymptotics. We provide a discussion here and refer the reader to van der Vaart (2000); Hirano and Porter (2009) for more details.

For each task $x _ { k } .$ , denote the individual-level observations by $D _ { k , n } = \{ ( A _ { k i } , Y _ { k i } ) : i = 1 , \ldots , n \}$ , where $A _ { k i } \in \{ 0 , 1 \}$ is the randomized treatment assignment and $Y _ { k i }$ is the observed outcome. For simplicity, we consider independent samples of n units for each task. We assume that n is large. Writing $D _ { n } = ( D _ { 1 , n } , \ldots , D _ { K , n } )$ for the full data, a general decision rule takes the form $\widetilde { \pi } _ { n } ( D _ { n } ) \in [ 0 , 1 ] ^ { K }$

Let $\widehat { \tau } _ { k }$ be an estimator of $\tau _ { k }$ based on $D _ { k , n } .$ , for example the diference in means estimator $\widehat { \tau } _ { k } = \overline { { Y } } _ { 1 , k } - \overline { { Y } } _ { 0 , k }$ . By the central limit theorem, under regularity conditions,

$$
\sqrt { n } ( \widehat { \tau } _ { k } - \tau _ { k } )  \mathcal { N } ( 0 , \sigma _ { k } ^ { 2 } ) ,
$$

in distribution, justifying Assumption 1.

To justify task-level decision rules, by Assumption 2, $\begin{array} { r } { \tau _ { k } = \frac { h _ { k } } { \sqrt { n } } } \end{array}$ , so that

$$
Z _ { k , n } : = \sqrt { n } \widehat { \tau } _ { k }  \mathcal { N } ( h _ { k } , \sigma _ { k } ^ { 2 } ) .
$$

Writing $Z _ { n } = ( Z _ { 1 , n } , \ldots , Z _ { k , n } )$ , then $Z _ { n }  \mathcal { N } ( h , \mathrm { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { K } ) )$ , for $\pmb { h } = ( h _ { 1 } , \ldots , h _ { K } )$

Let $Q _ { n , h }$ be the distribution of $D _ { n }$ in a regular submodel and $\widehat { \tau } _ { k }$ a regular, asymptotically eficient estimator in this submodel. Let $\phi _ { h }$ denote the corresponding Gaussian limiting density of $Z _ { n }$ . The likelihood-ratio expansion (van der Vaart, 2000, Theorem 7.2 and Lemma 8.14) gives

$$
\log \left( \frac { d Q _ { n , h } } { d Q _ { n ,                  { 0 } } } \right) ( D _ { n } ) = \log \left( \frac { \phi _ { h } ( Z _ { n } ) } { \phi _ { \mathbf { 0 } } ( Z _ { n } ) } \right) + o _ { Q _ { n , \mathbf { 0 } } } ( 1 ) ,
$$

i.e., to distinguish h from 0 all the evidence from individual units is summarized (up to first order) by $Z _ { n }$ . In particular, this is all the information needed to determine a cell treatment decision (sign of h)

To connect this to decision rules, suppose that $\widetilde { \pi } _ { n } ( D _ { n } )$ ) converges in distribution under every local alternative. By van der Vaart (2000, Theorem 9.3) its limiting distribution can be reproduced by a randomized policy

$$
\widetilde { \pi } _ { \infty } ( Z , U ) , \qquad Z \sim { \mathcal { N } } ( h , \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { K } ) ) , U \sim \mathrm { U n i f } [ 0 , 1 ] .
$$

Thus, in the limit, every unit-level policy can be replicated by a Gaussian experiment and uniform noise. Finally, since our welfare criterion is based on expectations, for each randomized limiting policy, we can find a corresponding average policy $\pi _ { k } ( z ) = \mathbb { E } _ { U } \left[ \widetilde { \pi } _ { \infty , k } ( z , U ) \right]$ with the same expected welfare. With this justification, we can restrict to limiting decision rules $\overline { { \pi } } ( Z ) \in [ 0 , 1 ] ^ { K }$ . Since noise levels are unknown, we let decision rules additionally depend on estimates of them too.

## B Linear Smoothers

In Section 4 we focus on three classes of linear smoothers, each defined by its weight functions and a resolution parameter $r ,$ which controls the scale of smoothing. A small r corresponds to less smoothing. Smaller r allows for better approximation but typically increases variance. Larger r pools more information across neighboring tasks, reducing variance at the cost of (potentially) greater bias.

Definition 2 (Partition Pooling) Let $r > 0$ and let $\boldsymbol { A } _ { r }$ be a finite partition of X such that every cell contains a task and max $\begin{array} { r } { A \in \mathcal { A } _ { r } \operatorname* { s u p } _ { x , y \in A } \| x - y \| _ { 2 } \leq r } \end{array}$ . Define

$$
w _ { r , j } ^ { \mathrm { P } } ( x ) : = \frac { p _ { j } \mathbf { 1 } \{ x _ { j } \in A _ { r } ( x ) \} } { \sum _ { \ell : x _ { \ell } \in A _ { r } ( x ) } p _ { \ell } } ,
$$

where $A _ { r } ( x )$ denotes the cell containing x.

Definition 2 includes tree-based pooling, where the partition cells are the leaves of a tree and $w _ { r } ^ { \mathrm { P } }$ averages within each leaf. Under the assumptions of Theorem 3, a tree at resolution r has $J _ { r } \asymp r ^ { - d }$ leaves.

Definition 3 (Kernel Smoothing) Let $\mathsf { K } : \mathbb { R } ^ { d } \to [ 0 , \infty )$ be bounded, supported on $B ( 0 , 1 )$ , continuous at the origin, and satisfy $\mathsf { K } ( 0 ) > 0$ . For $r > 0$ , define

$$
w _ { r , j } ^ { \mathrm { K } } ( x ) : = \frac { \mathsf { K } \left( ( x _ { j } - x ) / r \right) } { \sum _ { \ell = 1 } ^ { K } \mathsf { K } \left( ( x _ { \ell } - x ) / r \right) } .
$$

Partition pooling imposes a common action within cells, while kernel smoothing can lead to a diferent decision at every task location. We associate the bandwidth of the kernel with its resolution.

Definition 4 (Series Regression) Fix $r > 0$ . Let ${ \pmb a } = ( a _ { 1 } , \ldots , a _ { K } )$ be fitting weights satisfying $a _ { k } > 0$ and $\textstyle \sum _ { k = 1 } ^ { K } a _ { k } = 1$ , and let $b _ { r } : \mathcal { X } \to \mathbb { R } ^ { q _ { r } }$ be a vector of basis functions with $q _ { r } \leq K$ and $q _ { r } \asymp r ^ { - d }$ . Assume that the constant function belongs to the span $o f b _ { r }$ and that $\begin{array} { r } { G _ { r , \mathbf { a } } : = \sum _ { k = 1 } ^ { K } a _ { k } b _ { r } ( x _ { k } ) b _ { r } ( x _ { k } ) ^ { \top } } \end{array}$ is nonsingular. Define

$$
\begin{array} { r } { w _ { r , \mathbf { a } , j } ^ { \mathrm { S } } ( x ) : = a _ { j } b _ { r } ( x ) ^ { \top } G _ { r , \mathbf { a } } ^ { - 1 } b _ { r } ( x _ { j } ) . } \end{array}
$$

The weights in Definition 4 are induced by the weighted least-squares estimator

$$
\widehat { \theta } _ { r , a } : = \underset { \theta \in \mathbb { R } ^ { q r } } { \arg \operatorname* { m i n } } \sum _ { k = 1 } ^ { K } a _ { k } \big ( Z _ { k } - b _ { r } \big ( x _ { k } \big ) ^ { \top } \theta \big ) ^ { 2 } ,
$$

with resulting signal estimator $\begin{array} { r } { \widehat { g } _ { w _ { r , a } ^ { \mathrm { S } } } ( x ) : = b _ { r } ( x ) ^ { \top } \widehat \theta _ { r , a } = \sum _ { j = 1 } ^ { K } w _ { r , a , j } ^ { \mathrm { S } } ( x ) Z _ { j } } \end{array}$ . By requiring that the constant function is in the span of the basis function, we guarantee $\begin{array} { r } { \sum _ { j = 1 } ^ { K } w _ { r , a , j } ^ { \mathrm { S } } ( x ) = 1 } \end{array}$

Choosing $a _ { k } = p _ { k }$ gives the welfare-weighted series smoother and makes a cell-indicator basis coincide exactly with partition pooling. Choosing $a _ { k } = 1 / K$ gives ordinary least squares, while choosing $a _ { k } \propto \widehat \sigma _ { k } ^ { - 2 }$ gives inversevariance-weighted least squares.

## B.1 Smoothers for Theorem 3

Theorem 3 applies smoothers induced by the following weights:

(a) Partition pooling: Let $\mathcal { A } _ { r }$ be as in Definition 2, and suppose there exists a constant $c _ { \mathrm { P } } ~ > ~ 0$ such that 1 $\mathrm { n i n } _ { A \in { \mathcal { A } } _ { r } } Q _ { K } ( A ) \geq c _ { \mathrm { P } } r ^ { d }$ . Let $w _ { r } ^ { \mathrm { P } }$ denote the weights induced by this partition.

(b) Kernel smoothing: Let K be as in Definition 3 inducing $w _ { r } ^ { \mathrm { K } }$

(c) Spline regression: Suppose $\mathcal { X } \subseteq [ 0 , 1 ] ^ { d }$ and let b<sub>r</sub> be a normalized tensor-product B-spline basis with knot spacing of order r. That is, $b _ { r , \nu } ( x ) \geq 0$ and $\begin{array} { r } { \sum _ { \nu = 1 } ^ { q _ { r } } b _ { r , \nu } ( x ) \equiv 1 } \end{array}$ for $x \in \mathcal { X }$ . Let $a _ { k } = p _ { k }$ and impose the conditions in Definition 4. Let $w _ { r } ^ { \mathrm { S } }$ denote the induced weights.

The assumption on partition pooling is mild. In particular, it is satisfied under Assumption 4 whenever, around each task covariate, one can inscribe a ball of radius $\rho r , \rho \in ( 0 , 1 / 2 )$ ), that is contained in the same partition cell.

## C Additional Simulation Results

This appendix collects and extends the results referenced in Section 5.

## C.1 Simulation Design of Section 5

We place $K ~ = ~ 1 0 0$ tasks on a regular grid at each covariate combination $x _ { k } ~ = ~ ( X _ { 1 } , X _ { 2 } )$ with $X _ { 1 } , X _ { 2 } \in$ $\{ 0 . 0 5 , 0 . 1 5 , \ldots , 0 . 9 5 \} \subseteq [ 0 , 1 ] ^ { 2 }$ and $d = 2 .$ . Task are weighted equally, $p _ { k } = 1 / K$ , and noise is homoskedastic $\sigma _ { k } = 1$ . Adjacent tasks are spaced $K ^ { - 1 / d } = 0 . 1$ apart, which we use as the unit of resolution. Thus Assump tion 4 holds by construction.

For the curved signal, define

$$
g _ { 0 } ( x ) = 0 . 3 - \| x - c \| _ { 2 } ,
$$

where $c = ( 0 . 5 5 , 0 . 5 5 )$ and center it as

$$
g _ { \mathrm { c r v } } ( x ) = g _ { 0 } ( x ) - \frac { 1 } { K } \sum _ { j = 1 } ^ { K } g _ { 0 } ( x _ { j } ) .
$$

The centered signal is H¨older-smooth with $\beta = 1$ and $L = 1$ . Since $\begin{array} { r } { \sum _ { k = 1 } ^ { K } p _ { k } g _ { \mathrm { c r v } } ( x _ { k } ) = 0 } \end{array}$ , the treat-all and treat-none policies both have value zero. The resulting signal satisfies $H \ : = \ : 0 . 3 9$ , so the design lies within Assumption 2.

Recall that we considered three types of smoothers: axis-aligned trees with J leaves, tensor-product B-splines of degree two with dimension q (where $q = 3$ corresponds to ordinary least squares with features $( 1 , X _ { 1 } , X _ { 2 } ) )$ , and kernel smoothing with an Epanechnikov kernel of bandwidth h. To account for grid spacing, we measure resolution in grid units, $\tilde { r } = K ^ { 1 / d } r \mathrm { f o r } r \in [ 0 , 1 ]$ . For trees and splines, we set $\tilde { r } = K ^ { 1 / d } \bar { J } ^ { - 1 / d } \mathrm { ~ a n d ~ } \tilde { r } = K ^ { 1 / d } q ^ { - 1 / d } ,$ respectively, with $J \in \{ 1 , 4 , 9 , 1 6 , 2 5 , 3 6 , 4 9 \}$ and $q \in \{ 1 , 3 , 9 , 1 6 , 2 5 , 3 6 , 4 9 \}$ . For kernel smoothing, we set $\tilde { r } = h$ and vary $h \in [ 1 , 1 0 ]$

The three families agree closely at the grid-unit resolution predicted by Theorem $3 , \tilde { r } _ { K } ^ { * } \approx 3 . 1 6$ . The nearest candidate choices are $J = 9$ leaves for trees, $q = 9$ for splines, and $h = 3$ for kernels.

## C.2 Linear Signal

Alongside the curved signal in Section 5, we consider a linear signal on the same design,

$$
g _ { \mathrm { l i n } } ( x ) = ( x - \bar { x } ) ^ { \top } u
$$

where $u = ( 1 , 1 ) / \sqrt { 2 }$ and $\begin{array} { r } { \bar { \boldsymbol { x } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \boldsymbol { x } _ { k } } \end{array}$ . This is again a H¨older smooth function with $\beta = 1$ and $L = 1$ . It satisfies $H = 0 . 6 4$ and thus sits within Assumption 2.

Figure 4 illustrates the regime. Even through the overall signal is almost twice as strong as in Section 4 the plug-in policy still mis-labels 41% of tasks and attains only 27% of $V _ { g _ { \mathrm { l i n } } } ^ { * }$

The linear smoothers perform quite well in this regime, with the linear regression at $q = 3$ performing best and, unsurprisingly, attaining 88% of $V _ { g _ { \mathrm { l i n } } } ^ { * }$ . The performance of the remaining learners is reported in Table 2.

## C.3 Comparison of Learners

Table 2 compares representative learners from each family at three matched resolutions under both signals. Because both signals are centered, $W / V ^ { * } = 0$ means that a learner does no better than a constant policy, while $W / V ^ { * } = 1$ means that it matches the oracle. In the table and figures, partitions correspond to fixed axis aligned partitions with J cells.

The table highlights the approximation–estimation tradeof of Proposition 1. Within each family, if the resolution is too coarse, approximation error dominates. However, if the resolution is too fine, estimation error dominates. We observe that kernel smoothing adapts well to both signal regimes.

Figure 5 further illustrates this point and also plots the bound from Proposition 1. The bound is close for series regression and less close for trees. Overall, the figure shows that balancing approximation and estimation error predicts the optimal resolution well.

![](images/2532808c65ae1b972c6370acbef75038c94268e3a197515aa362da87a1a6dfc5.jpg)

![](images/84ce541b830f9bf45c890cf6500a6abcf54e2ad421cc31adb97aef35cdc56ba9.jpg)  
Figure 4: The first three panels show the true linear CATE g, one draw of the rescaled estimates $Z _ { k } = \sqrt { n } \widehat { \tau } _ { k }$ and the plug-in policy $1 \{ Z _ { k } > 0 \}$ that these estimates imply. The remaining panels show representative linear smoothers. The dashed line is the true frontier at which $g = 0$ , while the solid line represents the learner’s decision boundary, $\widehat { g } _ { w } = 0$

Figure 6 additionally illustrates the performance of our linear smoothers in terms of their parameter complexity. Especially for the curved signal, striking the right balance in parameter complexity, and hence in resolution, is crucial for good performance.

<table><tr><td colspan="3" rowspan="2">Learner  $\tilde { r }$ </td><td colspan="2">Linear signal</td><td colspan="2">Curved signal</td></tr><tr><td> $R _ { K }$ </td><td> $W / V ^ { * }$ </td><td> $R _ { K }$ </td><td> $W / V ^ { * }$ </td></tr><tr><td>Plug-in</td><td>1</td><td>0.085</td><td>0.27</td><td>0.052</td><td></td><td>0.14</td></tr><tr><td>Pooled</td><td>10</td><td>0.117</td><td>0.00</td><td></td><td>0.060</td><td>0.00</td></tr><tr><td>Tree (J = 4)</td><td>5.0</td><td>0.035</td><td></td><td>0.70</td><td>0.034</td><td>0.43</td></tr><tr><td>Tree (J = 9)</td><td>3.3</td><td>0.040</td><td></td><td>0.65</td><td>0.034</td><td>0.44</td></tr><tr><td>Tree (J = 25)</td><td>2.0</td><td>0.063</td><td></td><td>0.46</td><td>0.043</td><td>0.30</td></tr><tr><td></td><td>5.0</td><td>0.035</td><td></td><td>0.70</td><td>0.056</td><td>0.07</td></tr><tr><td>Partition (J = 4) Partition (J = 9)</td><td>3.3</td><td>0.041</td><td></td><td>0.65</td><td>0.045</td><td>0.25</td></tr><tr><td>Partition (J = 25)</td><td>2.0</td><td>0.060</td><td></td><td>0.49</td><td>0.046</td><td>0.24</td></tr><tr><td></td><td></td><td></td><td></td><td>0.80</td><td></td><td></td></tr><tr><td>Kernel (h = 5)</td><td>5.0 3.3</td><td>0.024 0.033</td><td></td><td>0.72</td><td>0.041 0.038</td><td>0.32 0.37</td></tr><tr><td>Kernel (h = 3) Kernel (h = 2)</td><td>2.0</td><td>0.048</td><td></td><td>0.59</td><td>0.042</td><td>0.31</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Series (q = 3)</td><td>5.8</td><td>0.014</td><td></td><td>0.88 0.72</td><td>0.056</td><td>0.08</td></tr><tr><td>Series (q = 9)</td><td>3.3</td><td>0.033</td><td></td><td></td><td>0.033</td><td>0.46</td></tr><tr><td>Series (q = 25)</td><td>2.0</td><td>0.059</td><td></td><td>0.50</td><td>0.042</td><td>0.30</td></tr></table>

Table 2: Exact welfare regret and proportional welfare for representative learners. The resolution $\tilde { r }$ is measured in units of $K ^ { - 1 / d }$ , so $\tilde { r } = 1$ corresponds to the plug-in rule and $\tilde { r } = 1 0$ corresponds to full pooling, that is, a single treatment decision is made for all cells jointly.

![](images/590b147b8ef5ca3b2fc4e8286c2ea0c8058c098242c549399740177f400dd37f.jpg)  
Figure 5: Approximation error $\mathsf { A } _ { g } ( w _ { r } )$ , estimation error $\gamma ^ { * } \mathsf { E } _ { \sigma } ( w _ { r } )$ , their sum, and exact regret $R _ { K }$ as functions of resolution ˜r. Rows correspond to signal regimes, while columns correspond to the smoother families considered in the main text.

![](images/450984a46104d79843090e3ebc46d77c5d1c91f1f479a952a3f7dce4e843c785.jpg)  
Figure 6: Every candidate is one point in terms of approximation error $\mathsf { A } _ { g } ( w )$ and estimation error $\gamma ^ { * } \mathsf { E } _ { \sigma } ( w )$ Note that the axis are on diferent scales. Better smoothers lie closer to the bottom-left corner.

## D Additional Theoretical Results

Proposition 2 (Welfare Decomposition for Linear Smoothers) Suppose Assumptions 1–2 hold. Let $\widehat { g } _ { w }$ be a linear smoother. Then,

$$
\sqrt { n } W _ { P _ { g , \sigma } } ( \widetilde { \pi } _ { w } ) = \bar { g } _ { p } \bar { \psi } _ { w , p } + \mathrm { C o v } _ { p } ( g , \psi _ { w } ) ,
$$

wher $\begin{array} { r } { e \ \bar { g } _ { p } = \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) , \ \psi _ { w , k } = \mathbb { E } _ { g , \sigma } \left[ \tilde { \pi } _ { w , k } ( Z ) \right] = \Phi \left( \frac { m _ { w } ( x _ { k } ) } { s _ { w } ( x _ { k } ) } \right) , \ \bar { \psi } _ { w , p } = \sum _ { k = 1 } ^ { K } p _ { k } \psi _ { w , k } \ a n d } \end{array}$

$$
\mathrm { C o v } _ { p } ( g , \psi _ { w } ) = \sum _ { k = 1 } ^ { K } p _ { k } ( g ( x _ { k } ) - \bar { g } _ { p } ) ( \psi _ { w , k } - \bar { \psi } _ { w , p } ) = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } \sum _ { l = 1 } ^ { K } p _ { k } p _ { l } ( g ( x _ { k } ) - g ( x _ { l } ) ) ( \psi _ { w , k } - \psi _ { w , l } ) .
$$

The proof is given in Appendix F.6. Proposition 2 decomposes welfare into a baseline term and a covariance term. The baseline term captures what would be achieved by treating every task with probability $\bar { \psi } _ { w , p }$ . The covariance term measures the personalization capacity of the policy. It is larger when tasks with larger treatment efects are more likely to be treated.

Proposition 3 (Cross-Validation) Suppose Assumptions 1–2 hold. Assume further that max<sub>k</sub> $p _ { k } = O ( 1 / K )$ Suppose for each task $x _ { k }$ , the data is split into F folds and satisfies

$$
Z _ { k } ^ { ( f ) } \overset { \mathrm { i n d } } { \sim } \mathcal { N } ( h _ { k } , F \sigma _ { k } ^ { 2 } ) ,
$$

for $k \in [ K ] , f \in [ F ]$ . Denote $\begin{array} { r } { Z _ { k } ^ { ( - f ) } = \frac { 1 } { { \cal F } - 1 } \sum _ { j \neq f } Z _ { k } ^ { ( j ) } } \end{array}$ and $\begin{array} { r } { Z _ { k } \ = \ \frac { 1 } { \cal F } \sum _ { f = 1 } ^ { \cal F } Z _ { k } ^ { ( f ) } } \end{array}$ . Let $\Lambda _ { K }$ be a finite set of resolutions. For each $r \in \Lambda _ { K }$ , define the linear smoothing policies $\begin{array} { r } { \widetilde { \pi } _ { w _ { r } , k } ( z ) = \mathbf { 1 } \Big \{ \sum _ { j = 1 } ^ { K } w _ { r , j } ( x _ { k } ) z _ { j } > 0 \Big \} } \end{array}$ . Select

$$
\widehat { r } \in \mathop { \arg \operatorname* { m a x } } _ { r \in \Lambda _ { K } } \frac { 1 } { F } \sum _ { f = 1 } ^ { F } \sum _ { k = 1 } ^ { K } p _ { k } Z _ { k } ^ { ( f ) } \widetilde { \pi } _ { w _ { r } , k } \bigl ( Z ^ { ( - f ) } \bigr ) .
$$

Then,

$$
R _ { K } ( P _ { g , \sigma } , \widetilde { \pi } _ { w _ { \widehat { r } } } ) \lesssim \operatorname* { m i n } _ { r \in \Lambda _ { K } } R _ { K } ( P _ { g , \sigma _ { F } } , \widetilde { \pi } _ { w _ { r } } ) + \sqrt { \frac { \log ( | \Lambda _ { K } | ) } { K } } ,
$$

where $\pmb { \sigma } _ { F } = \sqrt { F / ( F - 1 ) } \pmb { \sigma }$

The proof is given in Appendix F.7. Proposition 3 justifies our use of cross-validation by maximizing welfare in Section 6. Under Assumption 1, $\begin{array} { r } { Z _ { k } = \frac { 1 } { { \cal F } } \sum _ { f = 1 } ^ { { \cal F } } Z _ { k } ^ { ( f ) } \sim { \cal N } ( h _ { k } , \sigma _ { k } ^ { 2 } ) } \end{array}$ , matches our observation model. The selected smoother achieves the regret of the best candidate resolution on $\Lambda _ { K }$ with an additional selection cost. $\operatorname { I f } \ | \Lambda _ { K } |$ grows at most polynomially in $K$ , this cost is $o ( K ^ { - \beta / ( 2 \beta + d ) } )$ , in which case cross-validation preserves the rate in Theorem 3.

## E Connection to Binary Classification and Regression

Choosing the optimal treatment decision for each task is analogous to binary classification (Yang, 1999; Audibert and Tsybakov, 2007) and, for plug-in rules, bounded by $L ^ { 1 }$ regression error (Stone, 1982). This connection is well established in the policy-learning literature $( \mathrm { e . g . }$ , Kitagawa and Tetenov, 2018; Luedtke and Chambaz, 2020). In particular, our scaled welfare regret is proportional to a weighted excess classification risk. However, we study a nonuniformly weighted, fixed-design, heteroskedastic Gaussian observation model, instead of observing i.i.d. binary labels directly.

Let $\begin{array} { r } { Q _ { K } = \sum _ { k = 1 } ^ { K } p _ { k } \delta _ { x _ { k } } } \end{array}$ denote the task measure. Consider the auxiliary binary classification problem

$$
\begin{array} { c } { { X \sim Q _ { K } } } \\ { { Y \mid X = x \sim \operatorname { B e r } ( \eta _ { g } ( x ) ) , } } \end{array}
$$

where $\begin{array} { r } { \eta _ { g } ( x ) = \frac { 1 } { 2 } \left( 1 + \frac { g ( x ) } { H } \right) \in [ 0 , 1 ] \mathrm { f o r } g \in \mathcal { H } _ { d } ( \beta , L , H ) } \end{array}$ . The Bayes classifier is given by $\pi _ { g } ^ { * } ( x ) = 1 \big \{ \eta _ { g } ( x ) > \frac { 1 } { 2 } \big \} =$ $1 \{ g ( x ) > 0 \}$

We do not observe labels from this auxiliary model. Instead, we observe $( Z , { \widehat { \sigma } } ) \sim P _ { g , \sigma }$ , where $Z _ { k } = g ( x _ { k } ) + \sigma ( x _ { k } ) \epsilon _ { k }$ with $\epsilon _ { k } \overset { \mathrm { i n d } } { \sim } \mathcal { N } ( 0 , 1 )$ , for all $k \in [ K ]$ . Write ${ \widehat { \pi } } : = \widetilde { \pi } ( Z , { \widehat { \sigma } } )$ for the fitted classifier, a map from X to $\{ 0 , 1 \}$ . Its expected excess risk in the auxiliary classification problem is $( \mathrm { e . g . }$ , Audibert and Tsybakov, 2007)

$$
\begin{array} { l } { \displaystyle \mathcal { E } \big ( P _ { g , \sigma } , \widetilde { \pi } \big ) = \mathbb { E } _ { P _ { g , \sigma } } \big [ \mathbb { E } _ { X } \big [ | 2 \eta _ { g } ( X ) - 1 | \mathbf { 1 } \big \{ \widehat { \pi } ( X ) \neq \mathbf { 1 } \big \{ g ( X ) > 0 \big \} \big \} \big ] \big ] } \\ { \displaystyle = \mathbb { E } _ { P _ { g , \sigma } } \left[ \frac { 1 } { H } \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | \mathbf { 1 } \big \{ \widehat { \pi } ( x _ { k } ) \neq \mathbf { 1 } \big \{ g ( x _ { k } ) > 0 \big \} \right\} } \\ { \displaystyle = \frac { 1 } { H } \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | P _ { g , \sigma } \left( \widehat { \pi } ( x _ { k } ) \neq \mathbf { 1 } \big \{ g ( x _ { k } ) > 0 \big \} \right) } \\ { \displaystyle = \frac { 1 } { H } R _ { K } ( P _ { g , \sigma } , \widetilde { \pi } ) , } \end{array}
$$

Thus the two problems have the same loss but the observation model difers. Moreover, in the auxiliary classification model, g determines both the conditional mean and variance. In our Gaussian model, the variances must be controlled independently.

Furthermore, for any estimator $\widehat g$ and its induced plug-in policy $\widehat { \pi } ( X ) = \mathbf { 1 } \{ \widehat { g } ( x ) > 0 \}$ , a decision error implies that $| g ( x _ { k } ) | \leq | { \widehat { g } } ( x _ { k } ) - g ( x _ { k } ) |$ |. Hence,

$$
R _ { K } ( P _ { g , \sigma } , \widetilde { \pi } ) \le \mathbb { E } _ { P _ { g , \sigma } } \left[ \sum _ { k = 1 } ^ { K } p _ { k } | \widehat { g } ( x _ { k } ) - g ( x _ { k } ) | \right] = \mathbb { E } _ { P _ { g , \sigma } } \Vert \widehat { g } - g \Vert _ { L ^ { 1 } ( Q _ { K } ) } .
$$

Thus global $L ^ { 1 }$ regression rates provide an upper bound on the welfare regret.

## F Mathematical Proofs

## F.1 Proof of Theorem 1

We first establish the minimax claim before proving the Bayes identity. Collect $Z ~ = ~ ( Z _ { 1 } , \ldots , Z _ { K } ) , ~ { \widehat { \sigma } } ~ =$ $( \widehat { \sigma } _ { 1 } , \dots , \widehat { \sigma } _ { K } )$ and write $\overline { { \pmb { \sigma } } } : = ( \overline { { \sigma } } , \dots , \overline { { \sigma } } )$ . Fix a probability law $Q _ { 0 }$ on $[ 0 , \infty ) ^ { K }$ . Let $\mathcal { P } _ { K } ^ { 0 } \subseteq \mathcal { P } _ { K }$ consist of the laws under which $\widehat { \sigma } \sim Q _ { 0 }$ and $\widehat { \sigma } \perp \perp Z$ . On $\mathcal { P } _ { K } ^ { 0 }$ , our variance estimates $\widehat { \sigma }$ are uninformative about h. That is, every rule ${ \widetilde { \pi } } ( Z , { \widehat { \sigma } } )$ has the same regret under every law in $\mathcal { P } _ { K } ^ { 0 }$ as $\begin{array} { r } { \overline { { \pi } } ( z ) : = \int \widetilde { \pi } ( z , s ) Q _ { 0 } ( d s ) } \end{array}$ . Thus, on $\mathcal { P } _ { K } ^ { 0 }$ , we can restrict to rules that ignore ${ \widehat { \sigma } } .$

Fix an arbitrary $t \in ( 0 , H ]$ . Since $\{ - t , t \} ^ { K } \subseteq [ - H , H ] ^ { K }$ and ${ \overline { { \pmb { \sigma } } } } \in [ \underline { { \sigma } } , { \overline { { \sigma } } } ] ^ { K }$ , we bound

$$
\begin{array} { r l }  \sum _  \vec { \textbf { q } } \neq \vec { \textbf { S } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec { \textbf { p } } , \vec  \textbf \end{array}
$$

where $P _ { \pmb { h } , \pmb { \sigma } } ^ { Z } = \mathcal { N } ( \pmb { h } , \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , \dots , \sigma _ { K } ^ { 2 } ) )$ is the Z-marginal of $P _ { h , \sigma }$

By Lemma 1 we obtain

$$
\begin{array} { r } { \underset { \boldsymbol { \tilde { \pi } } \cdot \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K } \to [ 0 , 1 ] ^ { K } \underset { P _ { h , \sigma } \in \mathcal { P } _ { K } } { \operatorname* { s u p } } } { \operatorname* { i n f } } R _ { K } ( P _ { h , \sigma } , \boldsymbol { \tilde { \pi } } ) \geq \frac { t } { 2 } \left( 1 - \underset { \rho ( h , h ^ { \prime } ) = 1 } { \operatorname* { m a x } } \mathrm { d } _ { \mathrm { T V } } ( P _ { h , \ \ \overline { { \sigma } } } ^ { Z } , P _ { h ^ { \prime } , \ \overline { { \sigma } } } ^ { Z } ) \right) \underset { k = 1 } { \overset { K } { \sum } } p _ { k } } \\ { \geq \frac { t } { 2 } \left( 1 - \underset { h , h ^ { \prime } \in \{ - t , t \} ^ { K } } { \operatorname* { m a x } } \mathrm { d } _ { \mathrm { T V } } ( P _ { h , \ \ \overline { { \sigma } } } ^ { Z } , P _ { h ^ { \prime } , \ \overline { { \sigma } } } ^ { Z } ) \right) , } \end{array}
$$

where $\begin{array} { r } { \rho ( h , h ^ { \prime } ) = \sum _ { k = 1 } ^ { K } \mathbf { 1 } \{ h _ { k } \neq h _ { k } ^ { \prime } \} } \end{array}$ is the hamming distance and $\operatorname { d } _ { \mathrm { { T V } } } ( \cdot , \cdot )$ the total variation distance between probability measures (Wainwright, 2019, Chapter 15.1.3). Observe that when $\rho ( h , h ^ { \prime } ) = 1 , P _ { h , \overline { { \sigma } } } ^ { Z } , P _ { h ^ { \prime } , \overline { { \sigma } } } ^ { Z }$ are two K-dimensional Gaussian laws that only difer in one component of their mean (by a distance of 2t) but share same covariance structure of ${ \overline { { \sigma } } } ^ { 2 } I$ . By Polyanskiy and Wu (2025, Equation (7.40))

$$
\mathrm { d _ { T V } } \big ( P _ { h , \overline { { { \sigma } } } } ^ { Z } , P _ { h ^ { \prime } , \overline { { { \sigma } } } } ^ { Z } \big ) = 2 \Phi \left( \frac { t } { \overline { { { \sigma } } } } \right) - 1 .
$$

Plugging this into the above,

$$
\begin{array} { l } { \displaystyle \operatorname* { i n f } _ { \widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K } \to [ 0 , 1 ] ^ { K } } \operatorname* { s u p } _ { P _ { h , \sigma } \in \mathcal { P } _ { K } } R _ { K } \big ( P _ { h , \sigma } , \widetilde { \pi } \big ) \geq \displaystyle \frac { t } { 2 } \left[ 1 - \left( 2 \Phi \left( \displaystyle \frac { t } { \overline { { \sigma } } } \right) - 1 \right) \right] } \\ { \displaystyle \qquad = t \left[ 1 - \Phi \left( \displaystyle \frac { t } { \overline { { \sigma } } } \right) \right] } \\ { \displaystyle \qquad = t \Phi \left( - \displaystyle \frac { t } { \overline { { \sigma } } } \right) . } \end{array}
$$

Since this inequality holds for every $t \in ( 0 , H ]$ , we deduce

$$
\operatorname* { i n f } _ { \substack { \widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K }  [ 0 , 1 ] ^ { K } P _ { h , \sigma } \in \mathcal { P } _ { K } } } R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } ) \geq \operatorname* { m a x } _ { 0 \leq t \leq H } t \Phi ( - \frac { t } { \overline { { \sigma } } } ) ,
$$

which concludes the first part of the proof.

Next we show that the plug-in policy $\widetilde { \pi } _ { k } ^ { * } ( Z , \widehat { \sigma } ) = \mathbf { 1 } \{ Z _ { k } > 0 \}$ attains the lower bound. Fix any $P _ { h , \sigma } \in \mathcal { P } _ { K }$ . Since the rule ignores $\widehat { \sigma }$ , we have

$$
\begin{array} { r l } { H _ { \mathrm { c s } } ( \hat { P } _ { n } , \hat { \pi } ^ { * } ) = \mathbf { E } _ { \hat { \pi } , n } \left[ \frac { \mathbf { X } } { n } \sum _ { i = 1 } ^ { n } \| \hat { P } _ { n } \| \hat { \mathbf { I } } _ { n } \| \hat { \mathbf { I } } _ { n } \otimes \hat { \mathbf { P } } _ { n } - \mathbf { I } _ { 1 } ( \mathcal { Z } _ { n } > 0 ) \right] \qquad } & { } \\ &  - \underbrace  \sum _ { i = 1 } ^ { n } \mu _ { i } [ \mathbf { I } _ { n } | \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \langle \hat { \mathbf { I } } _ { n } | \hat { \mathbf { I } } _ { n } \rangle + \mathbf { I } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat { \mathbf { I } } _ { n } \hat \end{array}\tag{3}
$$

Equality is obtained by setting $\sigma = { \overline { { \sigma } } }$ and any h with $| h _ { k } | = t _ { H , \overline { { \sigma } } } ^ { * }$ for all $k ,$ where $\begin{array} { r } { t _ { H , \overline { { \sigma } } } ^ { * } = \arg \operatorname* { m a x } _ { 0 \leq t \leq H } t \Phi \left( - \frac { t } { \overline { { \sigma } } } \right) } \end{array}$ ， since the regret depends on h only through $( | h _ { k } | ) _ { k = 1 } ^ { K }$ . This shows that

$$
\operatorname* { i n f } _ { \substack { \widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K }  [ 0 , 1 ] ^ { K } P _ { h , \sigma } \in \mathcal { P } _ { K } } } R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } ) = \operatorname* { m a x } _ { 0 \leq t \leq H } t \Phi ( - \frac { t } { \overline { { \sigma } } } ) .
$$

Finally, we prove the Bayes claim. Since for any prior $\Pi \in { \mathfrak { P } }$

$$
\operatorname* { i n f } _ { \substack { \widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K }  [ 0 , 1 ] ^ { K } P _ { h , \sigma } \in \mathcal { P } _ { K } } } \operatorname* { s u p } _ { R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } ) \atop \widetilde { \pi } : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K }  [ 0 , 1 ] ^ { K } } \int R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } ) \Pi ( d P _ { h , \sigma } ) ,\tag{4}
$$

it is enough to show that the reverse inequality holds for a least-favorable prior $\Pi ^ { * } \in \mathfrak { P }$ . Write $t ^ { * } = t _ { H , \overline { { \sigma } } } ^ { * }$ and define the prior $\Pi ^ { * }$ by

$$
\Pi ^ { * } \left( \left\{ P _ { h , \overline { { \sigma } } } \right\} \right) = 2 ^ { - K } ,
$$

for $P _ { h , \overline { { \sigma } } } \in \mathcal { P } _ { K } ^ { 0 }$ and $\pmb { h } \in \{ - t ^ { * } , t ^ { * } \} ^ { K }$ . This prior assigns equal probability to all elements in $\mathcal { P } _ { K } ^ { 0 }$ with $\pmb { h } \in \{ - t ^ { * } , t ^ { * } \} ^ { K }$ Then, following the above calculations and Lemma 1,

$$
\begin{array} { r l } { \underset { \leq t \leq T , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } } { \operatorname { f a r } } ( \rho _ { 1 , \infty } , \kappa _ { \infty } , \kappa _ { \infty } ( P _ { \kappa , \infty } , \tilde { \kappa } ) ) \underset { \leq t \leq T , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } } { \operatorname { f a r } } ( \rho _ { 1 , \infty } , \tilde { \kappa } _ { \infty } ( P _ { \kappa , \infty } , \tilde { \kappa } ) ) \mathbb { T } \langle \hat { u } P _ { \kappa , \infty } , \tilde { u } \rangle } & { } \\ { = } & { \underset { \leq t \leq T , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } } { \operatorname { f a r } } ( \rho _ { 1 , \infty } , \tilde { \kappa } _ { \infty } ( P _ { \kappa , \infty } , \tilde { \kappa } ) ) } & { } \\ { = } & { \underset { \leq t \leq T , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } } { \operatorname { f a r } } ( \rho _ { 1 , \infty } , \tilde { \kappa } _ { \infty } ( P _ { \kappa , \infty } , \tilde { \kappa } ) ) } & { } \\ { = } & { \underset { \leq t \leq T , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } , \eta _ { \infty } } { \operatorname { f a r } } ( \rho _ { 1 , \infty } , \tilde { \kappa } _ { \infty } ( P _ { \kappa , \infty } , \tilde { \kappa } ) ) } & { } \\ { = } &  \iota _ { \infty } \underset  \leq t \leq T , \eta _  \infty \end{array}
$$

Thus Π<sup>∗</sup> is least favorable. By Equation (3), the plug-in policy $\widetilde { \pi } ^ { * }$ attains this Bayes risk, which completes the proof.

## F.2 Proof of Theorem 2

The proof follows from an adaptation of Theorem 3 and Theorem 4 in Cai and Low (2011). To give intuition, establish that our estimator achieves the claimed rate and refer to Cai and Low (2011, Theorem 3) for the corresponding minimax lower bound. In the following, we write $\mathbb { E } _ { h , \sigma }$ and $\mathrm { V a r } _ { h , \sigma }$ for expectation and variance under the Gaussian marginal $Z \sim \mathcal { N } ( h , \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , . . . , \sigma _ { K } ^ { 2 } ) )$ .

Decompose

$$
\begin{array} { l } { \sqrt { n } \{ { V } _ { h } ^ { * } - \widehat { V } _ { m } ^ { * } \} = \left[ \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } h _ { k } + \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } | h _ { k } | \right] - \left[ \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } \left( Z _ { k } + \displaystyle \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right) \right] } \\ { = \left[ \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } ( h _ { k } - Z _ { k } ) \right] + \left[ \displaystyle \frac { 1 } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } \left( | h _ { k } | - \displaystyle \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right) \right] } \\ { = R _ { 1 } + R _ { 2 } . } \end{array}
$$

To bound the mean-square error, we study the bias and variance of $R _ { 1 }$ and $R _ { 2 }$ respectively. We start with $R _ { 1 }$ Since $\mathbb { E } _ { h , \sigma } [ Z _ { k } ] = h _ { k } , \mathbb { E } _ { h , \sigma } [ R _ { 1 } ] = 0$ . By independence of $Z _ { 1 } , \dots , Z _ { K }$

$$
\mathrm { V a r } _ { h , \sigma } ( R _ { 1 } ) = \frac { 1 } { 4 } \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } \sigma _ { k } ^ { 2 } \le \frac { \overline { { \sigma } } ^ { 2 } } { 4 } \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } \lesssim \frac { \overline { { \sigma } } ^ { 2 } } { 4 K } ,
$$

as max<sub>k</sub> $p _ { k } = O ( 1 / K )$ . Combining bias and variance yields

$$
\mathbb { E } _ { h , \sigma } [ R _ { 1 } ^ { 2 } ] \lesssim \frac { 1 } { K } .\tag{5}
$$

For the second term $R _ { 2 }$ we recall that for probabilistic Hermite polynomials and $Z _ { k } \sim \mathcal N ( h _ { k } , \sigma _ { k } ^ { 2 } )$

$$
\mathbb { E } _ { h , \sigma } \left[ \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right] = h _ { k } ^ { 2 j } .
$$

We use this to determine the bias,

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { h , \sigma } [ R _ { 2 } ] \Big \lVert =  \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } p _ { k } \mathbb { E } _ { h , \sigma } [ | h _ { k } | - \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } ( \frac { Z _ { k } } { \sigma _ { k } } ) ] } \\ { \displaystyle =  \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } p _ { k } ( | h _ { k } | - \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } h _ { k } ^ { 2 j } )  } \\ { \displaystyle =  \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } p _ { k } ( | h _ { k } | - Q _ { m } ^ { * } ( h _ { k } ) )  } \\ { \displaystyle \leq \frac { 1 } { 2 } \delta _ { m } , } \end{array}
$$

where $\begin{array} { r } { \delta _ { m } = \operatorname* { s u p } _ { x \in [ - H , H ] } | | x | - Q _ { m } ^ { * } ( x ) | } \end{array}$ and $\begin{array} { r } { Q _ { m } ^ { * } = \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } x ^ { 2 j } } \end{array}$ . Since $Q _ { m } ^ { * }$ is the best polynomial approximation of $| x | \mathrm { ~ o n ~ } [ - H , H ]$ we have (Bernstein, 1913; Varga and Karpenter, 1987)

$$
\delta _ { m } \lesssim \frac { H } { m } .
$$

Thus the bias of $R _ { 2 }$ is uniformly bounded by

$$
\operatorname* { s u p } _ { \substack { h \in [ - H , H ] ^ { K } } } | \mathbb { E } _ { h , \sigma } [ R _ { 2 } ] | \lesssim \frac { H } { m } .
$$

To determine the variance of $R _ { 2 }$ , Cai and Low (2011, Lemma 2 and Section 4.2) bounds the coeficients of $Q _ { m } ^ { * }$

$$
| q _ { j , m } ^ { * } | \leq H ^ { 1 - 2 j } 2 ^ { 3 m } .
$$

Hence for $1 \leq j \leq m$ Cai and Low (2011, Lemma 3) implies

$$
\begin{array} { r l } { \mathrm { V a r } _ { h , \sigma } \left( q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right) = \left( q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \right) ^ { 2 } \mathrm { V a r } \left( \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right) } & { } \\ & { \leq ( q _ { j , m } ^ { * } ) ^ { 2 } \sigma _ { k } ^ { 4 j } e ^ { ( h _ { k } / \sigma _ { k } ) ^ { 2 } } ( 2 j ) ^ { 2 j } } \\ & { \leq ( H ^ { 1 - 2 j } \partial _ { 2 } ^ { 3 } \pi ) ^ { 2 } \sigma _ { k } ^ { 4 j } e ^ { ( h _ { k } / \sigma _ { k } ^ { 2 j } ) ^ { 2 j } } ( 2 j ) ^ { 2 j } } \\ & { \leq H ^ { 2 } \mathrm { e } ^ { ( \frac { \sigma _ { k } } { H } ) ^ { 4 j } } e ^ { ( H / \sigma ) ^ { 2 } } ( 2 j ) ^ { 2 j } } \\ & { \leq H ^ { 2 } \mathrm { e } ^ { ( \frac { \sigma _ { k } } { H } ) ^ { 4 j } } e ^ { ( H / \sigma ) ^ { 2 } } 2 ^ { 2 j } m ^ { 2 m } } \\ & { \leq H ^ { 2 } e ^ { ( H / \sigma ) ^ { 2 } } \mathrm { s m ~ a x } \left\{ 1 , \left( \frac { \overline { { \sigma } } } { H } \right) ^ { 4 m } \right\} m ^ { 2 m } } \\ & { \leq H ^ { m } m ^ { 2 m } , } \end{array}
$$

for some large enough $A = A ( H , \overline { { \sigma } } , \underline { { \sigma } } ) ~ > ~ 0$ . For $j = 0$ the above term is constant as $\Theta _ { 0 } ( x ) \equiv 1$ . Using independence across k and the Minkowski’s inequality,

$$
\begin{array} { r l } & { \mathrm { V a r } _ { h , \sigma } ( \bar { R } _ { 2 } ) = \displaystyle \frac { 1 } { 4 } \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } \mathrm { V a r } _ { h , \sigma } \left( \displaystyle \sum _ { j = 0 } ^ { m } q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right) } \\ & { \qquad \le \displaystyle \frac { 1 } { 4 } \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } \left( \displaystyle \sum _ { j = 0 } ^ { m } \mathrm { V a r } _ { h , \sigma } \left( q _ { j , m } ^ { * } \sigma _ { k } ^ { 2 j } \Theta _ { 2 j } \left( \frac { Z _ { k } } { \sigma _ { k } } \right) \right) ^ { 1 / 2 } \right) ^ { \frac { 2 } { 3 } } } \\ & { \qquad \lesssim \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } m ^ { 2 } A ^ { m } m ^ { 2 m } } \\ & { \qquad \lesssim \displaystyle \frac { 1 } { K } A ^ { m } m ^ { 2 m } . } \end{array}
$$

Adding bias and variance yields

$$
\mathbb { E } _ { \boldsymbol { h } , \boldsymbol { \sigma } } [ R _ { 2 } ^ { 2 } ] \lesssim \frac { H ^ { 2 } } { m ^ { 2 } } + \frac { A ^ { m } m ^ { 2 m } } { K } .\tag{6}
$$

Combining Equations (5) and (6), for any $h \in [ - H , H ] ^ { K }$

$$
\begin{array} { r l } { \mathbb { E } _ { h , \sigma } \left[ \left( \sqrt { n } \{ V _ { h } ^ { * } - \widehat { V } _ { m } ^ { * } \} \right) ^ { 2 } \right] = \mathbb { E } _ { h , \sigma } [ ( R _ { 1 } + R _ { 2 } ) ^ { 2 } ] } & { } \\ & { \leq 2 \mathbb { E } _ { h , \sigma } [ R _ { 1 } ^ { 2 } ] + 2 \mathbb { E } _ { h , \sigma } [ R _ { 2 } ^ { 2 } ] } \\ & { \lesssim \displaystyle \frac { 1 } { K } + \frac { H ^ { 2 } } { m ^ { 2 } } + \frac { A ^ { m } m ^ { 2 m } } { K } } \\ & { \lesssim _ { H , \overline { \sigma } , \underline { \sigma } } \left( \frac { \log ( \log ( K ) ) } { \log ( K ) } \right) ^ { 2 } , } \end{array}
$$

where we set $m = m _ { K } = \textstyle \left\lfloor { \frac { 1 } { 4 } } \log ( K ) / ( \log ( \log ( K ) ) ) \right\rfloor$ in the last line to balance the terms. This proves the claimed rate.

## F.3 Proof of Proposition 1

Under Assumptions 1–2, $\widehat { g } _ { w } ( x ) \sim \mathcal { N } ( m _ { w } ( x ) , s _ { w } ( x ) ^ { 2 } )$ . Its induced statistical decision rule is given by $\widetilde { \pi } _ { w , k } ( Z ) =$ $\mathbf { 1 } \{ \widehat { g } _ { w } ( x _ { k } ) > 0 \}$ and incurs a regret of

$$
\begin{array} { l } { \displaystyle R _ { K } ( P _ { g , \sigma } , \widetilde { \pi } _ { w } ) = \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | | \mathbb { E } _ { g , \sigma } [ { \mathbf 1 } \big \{ \widehat { g } _ { w } ( x _ { k } ) > 0 \big \} ] - { \mathbf 1 } \{ g ( x _ { k } ) > 0 \} } \\ { = \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | P _ { g , \sigma } ( { \mathbf 1 } \{ \widehat { g } _ { w } ( x _ { k } ) > 0 \} \neq { \mathbf 1 } \{ g ( x _ { k } ) > 0 \} ) } \\ { = \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | P _ { g , \sigma } ( \widehat { g } _ { w } ( x _ { k } ) g ( x _ { k } ) \leq 0 ) } \\ { = \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | \Phi \left( - \frac { m _ { w } \left( x _ { k } \right) \operatorname { s i g n } ( g ( x _ { k } ) ) } { s _ { w } ( x _ { k } ) } \right) . } \end{array}\tag{7}
$$

We analyze the regret coordinatewise. If $g ( x _ { k } ) m _ { w } ( x _ { k } ) \leq 0$ then

$$
| g ( x _ { k } ) | \Phi \left( - { \frac { m _ { w } ( x _ { k } ) \operatorname { s i g n } ( g ( x _ { k } ) ) } { s _ { w } ( x _ { k } ) } } \right) \leq | g ( x _ { k } ) | \leq | g ( x _ { k } ) - m _ { w } ( x _ { k } ) | .
$$

Now suppose that $g ( x _ { k } ) m _ { w } ( x _ { k } ) > 0$ . Then $m _ { w } ( x _ { k } ) \mathrm { s i g n } ( g ( x _ { k } ) ) = \vert m _ { w } ( x _ { k } ) \vert$ , and hence

$$
\begin{array} { c l } { | g ( x _ { k } ) | \Phi \left( - \displaystyle \frac { | m _ { w } ( x _ { k } ) | } { s _ { w } ( x _ { k } ) } \right) \leq \left( | g ( x _ { k } ) - m _ { w } ( x _ { k } ) | + | m _ { w } ( x _ { k } ) | \right) \Phi \left( - \displaystyle \frac { | m _ { w } ( x _ { k } ) | } { s _ { w } ( x _ { k } ) } \right) } \\ { \leq | g ( x _ { k } ) - m _ { w } ( x _ { k } ) | + s _ { w } ( x _ { k } ) \displaystyle \frac { | m _ { w } ( x _ { k } ) | } { s _ { w } ( x _ { k } ) } \Phi \left( - \displaystyle \frac { | m _ { w } ( x _ { k } ) | } { s _ { w } ( x _ { k } ) } \right) } \\ { \leq | g ( x _ { k } ) - m _ { w } ( x _ { k } ) | + \gamma ^ { * } s _ { w } ( x _ { k } ) , } \end{array}
$$

where $\begin{array} { r } { \gamma ^ { * } : = \operatorname* { s u p } _ { t > 0 } t \Phi ( - t ) } \end{array}$ . Combining both cases gives

$$
\begin{array} { l } { { \displaystyle R _ { K } \big ( P _ { g , \sigma } , { \widetilde \pi } _ { w } \big ) = \sum _ { k = 1 } ^ { K } p _ { k } \big | g ( x _ { k } ) \big | \Phi \left( - \frac { m _ { w } \left( x _ { k } \right) \mathrm { s i g n } \left( g ( x _ { k } ) \right) } { s _ { w } ( x _ { k } ) } \right) } \ ~ } \\ { { \displaystyle \qquad \leq \sum _ { k = 1 } ^ { K } p _ { k } \Big ( \big | g ( x _ { k } ) - m _ { w } ( x _ { k } ) \big | + \gamma ^ { * } s _ { w } ( x _ { k } ) \Big ) } \ ~ } \\ { { \displaystyle \qquad = \mathsf { A } _ { g } ( w ) + \gamma ^ { * } \mathsf { E } _ { \sigma } ( w ) } . } \end{array}
$$

Finally, since $\| g \| _ { \infty } \leq H$ and $\textstyle \sum _ { k = 1 } ^ { K } p _ { k } = 1$ , we have trivially,

$$
R _ { K } ( P _ { g , \pmb \sigma } , \widetilde { \pi } _ { w } ) \leq \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | \leq H .
$$

We now show that the bound is sharp over the class of linear smoothers, both for the approximation term and for the estimation term. To establish sharpness of the coeficient on the estimation error, let $t ^ { * } \in \arg \operatorname* { m a x } _ { t > 0 } t \Phi ( - t )$ and suppose that

$$
g ( x ) = \frac { t ^ { * } \sigma } { \sqrt { K } } ,
$$

$\sigma _ { k } = \sigma$ and $\begin{array} { r } { p _ { k } = \frac { 1 } { K } } \end{array}$ for all $k \in [ K ]$ . Trivially, $g \in \mathcal { H } _ { d } ( \beta , L , H )$ . Consider the uniform linear smoother with $w _ { j } ( x _ { k } ) = 1 / K$ for all $j , k \in [ K ]$ , which sum to one. Then for all $k \in [ K ] , m _ { w } ( x _ { k } ) = g ( x _ { k } )$ and hence $\mathsf { A } _ { g } ( w ) = 0$

Furthermore, $\begin{array} { r } { s _ { w } ( x _ { k } ) = \frac { \sigma } { \sqrt { K } } } \end{array}$ for all $k \in [ K ]$ , which implies $\begin{array} { r } { \mathsf { E } _ { \sigma } ( w ) = \frac { \sigma } { \sqrt { K } } } \end{array}$ . Thus,

$$
\begin{array} { l } { { \displaystyle R _ { K } ( P _ { g , \sigma } , \widetilde { \pi } _ { w } ) = \frac { t ^ { * } \sigma } { \sqrt { K } } \Phi ( - t ^ { * } ) } } \\ { { \displaystyle ~ = \gamma ^ { * } \frac { \sigma } { \sqrt { K } } } } \\ { { \displaystyle ~ = \gamma ^ { * } \mathsf { E } _ { \sigma } ( w ) } } \end{array}
$$

and $\gamma ^ { * }$ is sharp.

Next, we show that the approximation coeficient (of 1) is sharp. Let K be even and define $\begin{array} { r } { \varepsilon _ { K } = \frac { H } { K ^ { 1 / 4 } } } \end{array}$ . Let $\begin{array} { r } { \Delta = \left( \frac { 2 H } { L } \right) ^ { 1 / \beta } } \end{array}$ and define the signal by

$$
g _ { K } ( x ) = { \left\{ \begin{array} { l l } { H , } & { { \mathrm { i f ~ } } x ^ { ( 1 ) } \leq 0 , } \\ { - \varepsilon _ { K } , } & { { \mathrm { i f ~ } } x ^ { ( 1 ) } \geq \Delta , } \end{array} \right. }
$$

with appropriate β-H¨older smoothing in between, where $x ^ { ( 1 ) }$ denotes the first coordinate of x. For example, one may take $g _ { K } ( x ) = - \varepsilon _ { K } + ( H + \varepsilon _ { K } ) f ( x ^ { ( 1 ) } )$ where $\begin{array} { r } { f ( u ) : = \left[ 1 - \left( \frac { u ^ { + } } { \Delta } \right) ^ { \beta } \right] ^ { + } } \end{array}$ . Hence $g _ { K } \in \mathcal { H } _ { d } ( \beta , L , H )$

Let $p _ { k } = 1 / K$ and $\sigma _ { k } = \sigma$ for all $k \in [ K ]$ . Choose task covariates $x _ { 1 } , \ldots , x _ { K / 2 }$ such that $g _ { K } ( x _ { k } ) = H$ , and choose $x _ { K / 2 + 1 } , \dotsc , x _ { K }$ such that $g _ { K } ( x _ { k } ) = - \varepsilon _ { K }$ . Define the smoothing weights

$$
w _ { j } ( x ) = \frac { 2 } { K } \mathbf { 1 } \{ g _ { K } ( x _ { j } ) \leq 0 \} .
$$

These weights are deterministic, nonnegative, and sum to one by construction. They induce the smoother

$$
{ \widehat g } _ { w } ( x ) = \frac { 2 } { K } \sum _ { j : g _ { K } ( x _ { j } ) \leq 0 } Z _ { j } ,
$$

which averages only over the negative-signal tasks.

Its mean is given by

$$
m _ { w } ( x _ { k } ) = \frac { 2 } { K } \sum _ { j : g ( x _ { j } ) \leq 0 } - \epsilon _ { K } = - \epsilon _ { K } ,
$$

and variance by

$$
s _ { w } ( x _ { k } ) ^ { 2 } = \left( { \frac { 2 } { K } } \right) ^ { 2 } \sum _ { j : g ( x _ { j } ) \leq 0 } \sigma ^ { 2 }
$$

Consequently, by Equation (7)

$$
\begin{array} { l } { { \displaystyle R _ { K } ( P _ { \mathcal { G } , \sigma } , { \widetilde { \pi } _ { w } } ) = \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) | \Phi \left( - \frac { m _ { w } ( x _ { k } ) \mathrm { s i g n } ( g ( x _ { k } ) ) } { s _ { w } ( x _ { k } ) } \right) } } \\ { ~ } \\ { { \displaystyle ~ = \frac { 1 } { K } \sum _ { k = 1 } ^ { K / 2 } H \Phi \left( \frac { \epsilon _ { K } } { \sigma \sqrt { 2 / K } } \right) + \frac { 1 } { K } \sum _ { k = K / 2 + 1 } ^ { K } { \epsilon _ { K } \Phi \left( \frac { - \epsilon _ { K } } { \sigma \sqrt { 2 / K } } \right) } } } \\ { { \displaystyle ~ = \frac { 1 } { K } \sum _ { k = 1 } ^ { K / 2 } H \Phi \left( \frac { H K ^ { 1 / 4 } } { \sigma \sqrt { 2 } } \right) + \frac { 1 } { K } \sum _ { k = K / 2 + 1 } ^ { K } { \epsilon _ { K } \Phi \left( \frac { - H K ^ { 1 / 4 } } { \sigma \sqrt { 2 } } \right) } } } \\ { { \displaystyle ~ = \frac { H } { 2 } \Phi \left( \frac { H K ^ { 1 / 4 } } { \sigma \sqrt { 2 } } \right) + \frac { \epsilon _ { K } } { 2 } \Phi \left( \frac { - H K ^ { 1 / 4 } } { \sigma \sqrt { 2 } } \right) . } } \end{array}
$$

Thus, $\begin{array} { r } { R _ { K } ( P _ { g , \pmb { \sigma } } , \widetilde { \pi } _ { w } )  \frac { H } { 2 } } \end{array}$ as $K  \infty$ . Since mistakes are made only on tasks with positive signal, the approxi mation error is

$$
\mathsf { A } _ { g _ { K } } ( w ) = \frac { 1 } { 2 } | H - ( - \epsilon _ { K } ) |  \frac { H } { 2 } .
$$

Since all tasks are equally weighted and have the same standard error, the estimation error is given by

$$
\mathsf E _ { \pmb { \sigma } } ( w ) = \sigma \sqrt { \frac { 2 } { K } }  0 .
$$

Hence, in the limit $R _ { K } ( P _ { g , \pmb { \sigma } } , \widetilde { \pi } _ { w } ) - \mathsf { A } _ { g _ { K } } ( w )  0$ , which proves that the approximation constant is sharp.

## F.4 Proof of Theorem 3

First, by Assumption 4, for every $j \in [ K ]$ , we have

$$
p _ { j } \le Q _ { K } ( B ( x _ { j } , a _ { 0 } K ^ { - 1 / d } ) ) \le \frac { C a _ { 0 } ^ { d } } { K } ,
$$

and consequently,

$$
\operatorname* { m a x } _ { j \in [ K ] } p _ { j } \le \frac { C a _ { 0 } ^ { d } } { K } .\tag{8}
$$

We now prove the theorem for partition pooling, then for kernel smoothing and finally for series smoothing estimators separately.

Partition Pooling. Fix $k \in [ K ]$ and let $A _ { k } = A _ { r } ( x _ { k } )$ be the cell of A<sub>r</sub> that contains $x _ { k }$ . Then $w _ { r , j } ^ { \mathrm { P } } ( x _ { k } ) =$ $\frac { p _ { j } \mathbf { 1 } \{ x _ { j } \in A _ { k } \} } { Q _ { K } ( A _ { k } ) }$ where $\begin{array} { r } { Q _ { K } ( A _ { k } ) = \sum _ { j : x _ { j } \in A _ { k } } p _ { j } } \end{array}$ . By Proposition 1, the approximation error is given by

$$
\begin{array} { r l } { \delta _ { 2 } \langle \mathbf { r } _ { k } ^ { \prime \prime } \rangle = } & { \underset { \mathrm { t } = 1 } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime \prime } ( \xi _ { k } ^ { \prime \prime } ) - m _ { k } \xi _ { k } ^ { \prime \prime } ( \xi _ { k } \xi _ { k } ^ { \prime \prime } ) } \\ & { - \underset { \mathrm { t } = 1 } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime \prime } \left| \phi ( \xi _ { k } ^ { \prime \prime } ) - \underset { \mathrm { t } \neq 1 } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime } \xi _ { k } ^ { \prime \prime } ( \xi _ { k } ^ { \prime \prime } ) \right| } \\ & { \leq \underset { \mathrm { t } = 1 } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime } \left| \underset { \mathrm { s u b } } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime } \left( \underset { \mathrm { d } \xi _ { k } ^ { \prime \prime } } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime } \xi _ { k } ^ { \prime \prime } \right) - \underset { \mathrm { t } \neq 1 } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime } \frac { \xi _ { k } ^ { \prime \prime } } { \mu _ { k } ^ { \prime } \mu _ { k } ^ { \prime } \xi _ { k } ^ { \prime } } \right| } \\ &  \leq \underset { \mathrm { t } = 1 } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime } \left| \underset { \mathrm { s u b } } { \overset { \nabla } { \sum } } \mu _ { k } ^ { \prime } \left( \frac { \xi _ { k } ^ { \prime } } { \xi _ { k } ^ { \prime } \xi _ { k } ^ { \prime } } \right) \right| \left( \overset { \xi _ { k } ^ { \prime \prime } } { \xi _ { k } ^ { \prime } } \right) - \underset { \mathrm { t } \neq 1 } { \overset { \nabla } { \sum } } \mu \end{array}
$$

Likewise, for the estimation error, by Assumption 1,

$$
\begin{array} { r l } { T _ { \mathrm { e f f } } ( x , y ) = \frac { \sqrt { 3 } } { 2 } \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { = \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { = \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } + \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { = \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ & { \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } } { 2 }  \\ &  \quad - \frac { \sqrt { 3 } } { 2 } ( x - y ) ^ { 2 } - \frac  \sqrt  \end{array}
$$

where we used Equation (8) and that min $\smash { \operatorname { \ 4 e \lrcorner } Q _ { K } ( A ) \geq c _ { \mathrm { P } } r ^ { d } }$ by assumption. This proves the partition pooling claim.

Kernel Smoothing. We consider the kernel smoothing claim next. Because the kernel is supported on the unit ball, when $w _ { r , j } ^ { \mathrm { K } } ( x _ { k } ) > 0$ then $\| x _ { j } - x _ { k } \| _ { 2 } \leq r$ . Hence,

$$
\begin{array} { r l } { \left. { \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) - \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) - m _ { k } , g ( x _ { k } ) | } } \\ & { = \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) - \sum _ { k = 1 } ^ { K } w _ { k , j } ^ { K } ( x _ { k } ) g ( x _ { j } ) | } \\ & { = \sum _ { k = 1 } ^ { K } \left| \sum _ { j = 1 } ^ { K } w _ { j , j } ^ { K } ( x _ { j } ) g ( x _ { j } ) - \sum _ { j = 1 } ^ { K } w _ { j , j } ^ { K } ( x _ { j } ) g ( x _ { j } ) \right| } \\ & { \leq \sum _ { k = 1 } ^ { K } \left| \sum _ { j = 1 } ^ { K } w _ { j , j } ^ { K } ( x _ { k } ) g ( x _ { j } ) \right| g ( x _ { k } ) - \sum _ { j = 1 } ^ { K } w _ { j , j } ^ { K } ( x _ { j } ) g ( x _ { j } ) \right| } \\ & { \leq \sum _ { k = 1 } ^ { K } \left| \sum _ { j = 1 } ^ { K } \sum _ { j = 1 } ^ { K } w _ { j , j } ^ { K } ( x _ { k } ) | g ( x _ { k } ) - g ( x _ { j } ) \right| } \\ & { \leq \sum _ { k = 1 } ^ { K } p _ { k } \sum _ { j = 1 } ^ { K } w _ { j , j } ^ { K } ( x _ { k } ) | \mathcal { L } ^ { 0 } } \\ & { \leq L \rho ^ { \rho } , \quad \forall \quad \rho ^ { \rho } \leq 0 . } \end{array}
$$

Next, we upper bound the estimation error. We start by bounding the squared sum of weights,

$$
\begin{array} { r l r } {  { \sum _ { j = 1 } ^ { K } w _ { r , j } ^ { \mathrm { K } } ( x _ { k } ) ^ { 2 } \le \sum _ { j = 1 } ^ { K } \frac { \mathsf { K } ( ( x _ { j } - x _ { k } ) / r ) ^ { 2 } } { ( \sum _ { \ell = 1 } ^ { K } \mathsf { K } ( ( x _ { \ell } - x _ { k } ) / r ) ) ^ { 2 } } } } \\ & { } & { \le \| \mathsf { K } \| _ { \infty } \displaystyle \sum _ { j = 1 } ^ { K } \frac { \mathsf { K } ( ( x _ { j } - x _ { k } ) / r ) } { ( \sum _ { \ell = 1 } ^ { K } \mathsf { K } ( ( x _ { \ell } - x _ { k } ) / r ) ) ^ { 2 } } } \\ & { } & { = \frac { \| \mathsf { K } \| _ { \infty } } { \sum _ { \ell = 1 } ^ { K } \mathsf { K } ( ( x _ { \ell } - x _ { k } ) / r ) } } \end{array}\tag{9}
$$

Convexity also gives $\begin{array} { r } { \sum _ { j = 1 } ^ { K } w _ { r , j } ^ { \mathrm { K } } ( x _ { k } ) ^ { 2 } \le 1 } \end{array}$

Because the kernel is continuous at zero and positive at zero, there are constants $\rho \in ( 0 , 1 ]$ and $\kappa > 0$ such that $\mathsf { K } ( u ) \geq \kappa$ for all $\| u \| _ { 2 } \leq \rho .$ This induces two regimes. If $\rho r \geq a _ { 0 } K ^ { - 1 / d }$ , then

$$
\begin{array} { r l } { \displaystyle \sum _ { \ell = 1 } ^ { K } { \mathsf { K } \left( \frac { x _ { \ell } - x _ { k } } { r } \right) \geq \kappa \sum _ { \ell = 1 } ^ { K } \mathsf { I } \left\{ \| x _ { \ell } - x _ { k } \| _ { 2 } \leq \rho r \right\} } } \\ { \geq \kappa \displaystyle \sum _ { \ell = 1 } ^ { K } \frac { p _ { \ell } } { \operatorname* { m a x } _ { \ell \in [ K ] } p _ { \ell } } \mathsf { I } \left\{ \| x _ { \ell } - x _ { k } \| _ { 2 } \leq \rho r \right\} } \\ { \displaystyle } & { = \kappa \frac { Q _ { K } ( B ( x _ { k } , \rho ) ) } { \operatorname* { m a x } _ { \ell \in [ K ] } p _ { \ell } } } \\ { \geq \kappa \frac { \sigma \mu _ { r } ^ { 2 } d } { C a _ { \ell } ^ { 3 } / K } } \\ { } & { = \frac { \kappa c \rho ^ { d } } { C a _ { \ell } ^ { 3 } } \kappa r ^ { d } , } \end{array}
$$

where we used ma $\mathsf { \tilde { \ell } { \in } } [ K ] p \ell \le C a _ { 0 } ^ { d } / K$ by Equation (8) and $Q _ { K } ( B ( x _ { k } , \rho r ) ) \geq c \rho ^ { d } r ^ { d }$ by Assumption 4. Consequently, if $\rho r \geq a _ { 0 } K ^ { - 1 / d }$ , Equation (9) gives

$$
\sum _ { j = 1 } ^ { K } w _ { r , j } ^ { \mathrm { K } } ( x _ { k } ) ^ { 2 } \leq \frac { C a _ { 0 } ^ { d } \| \mathsf { K } \| _ { \infty } } { \kappa c \rho ^ { d } } \frac { 1 } { K r ^ { d } } .
$$

If instead $\rho r < a _ { 0 } K ^ { - 1 / d }$ , then $\begin{array} { r } { K r ^ { d } < \frac { a _ { 0 } ^ { d } } { \rho ^ { d } } } \end{array}$ . Therefore, using $\begin{array} { r } { \sum _ { j = 1 } ^ { K } w _ { r , j } ^ { \mathrm { K } } ( x _ { k } ) ^ { 2 } \le 1 } \end{array}$

$$
\sum _ { j = 1 } ^ { K } w _ { r , j } ^ { \mathrm { K } } ( x _ { k } ) ^ { 2 } \leq 1 \leq \frac { a _ { 0 } ^ { d } } { \rho ^ { d } } \frac { 1 } { K r ^ { d } } .
$$

Combining the two regimes gives

$$
\sum _ { j = 1 } ^ { K } w _ { r , j } ^ { \mathrm { K } } ( x _ { k } ) ^ { 2 } \leq \frac { a _ { 0 } ^ { d } } { \rho ^ { d } } \operatorname* { m a x } \left\{ 1 , \frac { C \| \mathsf { K } \| _ { \infty } } { \kappa c } \right\} \frac { 1 } { K r ^ { d } } .
$$

It follows that

$$
\begin{array} { r l } { \Gamma _ { \varphi } ( z _ { \nu } ^ { K , \nu } ) } & { \xrightarrow [ ] { K } \mathbb { P } _ { \varepsilon \to \nu _ { \varepsilon } ^ { \prime } } ( \alpha _ { \star } ) } \\ & { = \sum _ { \tilde { \nu } = 1 } ^ { K } \mathcal { P } _ { \varepsilon } \Bigg \{ \underset { \sum _ { i = 1 } ^ { K } \nu _ { i } \leq n _ { i } ^ { K } ( \alpha _ { \star } ) \leq n _ { i } ^ { I } } { \sum _ { i = 1 } ^ { K } \nu _ { i } ( \alpha _ { \star } ) \leq n _ { i } ^ { I } } z _ { \nu } ^ { I } } \\ & { \leq \sigma \sum _ { \tilde { \nu } = \nu _ { \varepsilon } ^ { \prime } } ^ { K } \sqrt { \underset { \sum _ { i = 1 } ^ { K } \nu _ { i } \leq n _ { i } ^ { I } ( \alpha _ { \star } ) \leq 1 } { \sum _ { i = 1 } ^ { K } \nu _ { i } ( \alpha _ { \star } ) } } } \\ & { \leq \frac { \alpha _ { \tilde { \nu } } ^ { \tilde { \nu } / 2 } } { \beta ^ { \tilde { \nu } / 2 } } \operatorname* { m a x } \left\{ 1 , \frac { G \mathrm { K } [ \mathrm { K } ] } { \alpha _ { \star } } \right\} ^ { 1 / 2 } \frac { \sigma } { \sqrt { K \mathcal { H } ^ { I } } \varepsilon ^ { I } \leq 1 } \frac { K } { \sqrt { K \mathcal { H } ^ { I } } \varepsilon ^ { I } \leq 1 } } \\ & { \leq \frac { \alpha _ { \tilde { \nu } } ^ { \tilde { \nu } / 2 } } { \beta ^ { \tilde { \nu } / 2 } } \operatorname* { m a x } \left\{ 1 , \frac { G \mathrm { K } [ \mathrm { K } ] } { \alpha _ { \star } } \right\} ^ { 1 / 2 } \frac { \sigma } { \sqrt { K \mathcal { H } ^ { I } } \varepsilon ^ { I } } \frac { 1 } { \sqrt { K \mathcal { H } ^ { I } } \varepsilon ^ { I } } } \\ &  \lesssim \frac { \alpha _ { \tilde { \nu } } ^ { \tilde { \nu } / 2 } }  \beta  \end{array}
$$

This proves the bounds on the kernel smoothing estimator.

Series Regression. We start with the approximation error. Fix the a potential signal $g \in \mathcal { H } _ { d } ( \beta , L , H )$ . We construct a signal close to gthat lies within the span of the basis functions.

For each $\nu \in [ q _ { r } ]$ choose $\xi _ { \nu } \in \mathrm { s u p p } ( b _ { r , \nu } ) \cap \mathcal { X }$ and define the spline interior signal as

$$
g _ { r } ^ { \circ } ( x ) = \sum _ { \nu = 1 } ^ { q _ { r } } \theta _ { \nu } ^ { \circ } b _ { r , \nu } ( x ) ,\tag{10}
$$

where $\theta _ { \nu } ^ { \circ } = g ( \xi _ { \nu } )$ are the spline coeficients. By assumption, for each $x \in \mathcal { X }$ exists ν such that $b _ { r , \nu } ( x ) > 0$ Furthermore, if $x \in \operatorname { s u p p } ( b _ { r , \nu } )$ then $\| x - \xi _ { \nu } \| _ { 2 } \leq C _ { \mathrm { S } } r$ . Since $\begin{array} { r } { \sum _ { \nu = 1 } ^ { q _ { r } } b _ { r , \nu } ( x ) = 1 } \end{array}$ , for all $x \in \mathcal { X }$

$$
\begin{array} { r l } { \Phi ( z ) = - \Phi ^ { 2 } ( z ) \Big [ z \Big ] = \Bigg [ \theta ( z ) - \displaystyle \sum _ { k = 1 } ^ { \infty } \theta ( k _ { k } \omega ( z ) ) } \\ & { = \Bigg [ \theta ( z ) - \displaystyle \sum _ { k = 1 } ^ { \infty } - \displaystyle \sum _ { k = 1 } ^ { \infty } \theta ( k _ { k } \omega ( z ) ) \kappa _ { k } ( z ) \Bigg ] } \\ & { = \Bigg [ \displaystyle \sum _ { k = 1 } ^ { \infty } \hat { \nu } _ { k } ( x ) ( k _ { k } \omega ( z ) ) - \displaystyle \sum _ { k = 1 } ^ { \infty } \theta ( k _ { k } \omega ( z ) ) k _ { k } ( x ) } \\ & { \leq \displaystyle \sum _ { k = 1 } ^ { \infty } \hat { \nu } _ { k } ( x ) ( k _ { k } \omega ( z ) ) ( \theta ( x ) - \displaystyle \sum _ { k = 1 } ^ { \infty } \theta ( k _ { k } \omega ( z ) )  } \\ & {  \quad - \displaystyle \sum _ { k = 1 } ^ { \infty } \theta ( k _ { k } \omega ( z ) ) ( \theta ( x ) - \displaystyle \sum _ { k = 1 } ^ { \infty } \theta ( k _ { k } \omega ( z ) )   } \\ & {  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \leq \displaystyle \sum _ { k = 1 } ^ { \infty } \hat { \nu } _ { k } ( x ) ( k _ { k } \omega ( z ) ) ( 1 - \delta _ { k } \omega ( k _ { k } \omega ( z ) )  \Bigg ] } \\ & { = \displaystyle \sum _ { k = 1 } ^ { \infty } \hat { \nu } _ { k } ( x ) ( 1 - \delta _ { k } \omega ( k _ { k } \omega ( z ) )  \Bigg ] } \\ & { \leq \displaystyle \mathrm { d } ( k _ { 0 } ^ { \infty } ) ^ { \infty } \displaystyle \sum _ { k = 1 } ^ { \infty } \hat { \nu } _ { k } ( x ) ( \theta ( x ) - \delta _ { k } \omega ( k _ { k } \omega ( z ) )  } \\ &  \quad \quad \quad \quad \quad \quad \end{array}\tag{11}
$$

where we used that $g \in \mathcal { H } _ { d } ( \beta , L , H )$

Next, because $a _ { k } = p _ { k }$ , we have $m _ { w _ { r } ^ { \mathrm { S } } } ( x ) = b _ { r } ( x ) ^ { \top } \theta _ { r , g } ^ { * }$ where

$$
\theta _ { r , g } ^ { * } = \mathbb { E } [ \widehat { \theta } _ { r , p } ] = G _ { r , p } ^ { - 1 } \sum _ { k = 1 } ^ { K } p _ { k } b _ { r } ( x _ { k } ) g ( x _ { k } ) .
$$

Define $\begin{array} { r } { Q _ { r } ( \theta ) = \sum _ { k = 1 } ^ { K } p _ { k } \left( g ( x _ { k } ) - b _ { r } ( x _ { k } ) ^ { \top } \theta \right) ^ { 2 } } \end{array}$ . Observe that $\theta _ { r , g } ^ { * }$ is the unique minimizer of $Q _ { r }$ , hence $Q _ { r } ( \theta _ { r , g } ^ { * } ) \leq$ $Q _ { r } ( \theta ^ { \circ } )$ , where $\theta ^ { \circ }$ is the coeficient vector of $g _ { r } ^ { \circ }$ in Equation (10). Thus

$$
\begin{array} { l } { { \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } \left( g ( x _ { k } ) - m _ { w _ { r } ^ { \mathrm { s } } } ( x _ { k } ) \right) ^ { 2 } = \sum _ { k = 1 } ^ { K } p _ { k } \left( g ( x _ { k } ) - b _ { r } ( x _ { k } ) ^ { \top } \theta _ { r , g } ^ { \ast } \right) ^ { 2 } } } \\ { ~ } \\ { { \displaystyle = Q _ { r } ( \theta _ { r , g } ^ { \ast } ) } } \\ { { \displaystyle ~ } } \\ { { \displaystyle ~ } } \\ { { \displaystyle ~ = \sum _ { k = 1 } ^ { K } p _ { k } \left( g ( x _ { k } ) - b _ { r } ( x _ { k } ) ^ { \top } \theta ^ { \ast } \right) ^ { 2 } } } \\ { ~ } \\ { { \displaystyle ~ } } \\ { { \displaystyle ~ = \sum _ { k = 1 } ^ { K } p _ { k } \left( g ( x _ { k } ) - g _ { r } ^ { \ast } ( x _ { k } ) \right) ^ { 2 } . } } \end{array}
$$

Finally, by Jensen’s inequality, $\textstyle \sum _ { k = 1 } ^ { K } p _ { k } = 1$ and Equation (11),

$$
\begin{array} { r l } & { \mathbb { A } _ { g } ( w _ { r } ^ { \mathbf { S } } ) = \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } | g ( x _ { k } ) - m _ { w _ { r } ^ { \mathbf { S } } } ( x _ { k } ) | } \\ & { \qquad \le \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } ( g ( x _ { k } ) - m _ { w _ { r } ^ { \mathbf { S } } } ( x _ { k } ) ) ^ { 2 } \Bigg | ^ { 1 / 2 } } \\ & { \qquad \le \displaystyle \left[ \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } ( g ( x _ { k } ) - m _ { w _ { r } ^ { \mathbf { S } } } ( x _ { k } ) ) ^ { 2 } \right] ^ { 1 / 2 } } \\ & { \qquad \le \displaystyle \left[ \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } ( g ( x _ { k } ) - g _ { r } ^ { \mathrm { S } } ( x _ { k } ) ) ^ { 2 } \right] ^ { 1 / 2 } } \\ & { \qquad \le \displaystyle \operatorname* { s u p } _ { x \in X } | g ( x ) - g _ { r } ^ { \mathrm { S } } ( x ) | } \\ & { \qquad \le C _ { 5 } L r ^ { \beta } } \\ & { \qquad \le \displaystyle L a ^ { \rho } ) , } \end{array}
$$

which concludes the claim on the approximation error.

For the estimation error, the weights in Definition 4 give, for every $k \in [ K ]$ 2

$$
\begin{array} { l } { { \displaystyle s _ { w _ { r } ^ { \perp } } ( x _ { k } ) ^ { 2 } = \sum _ { j = 1 } ^ { K } \sigma _ { j } ^ { 2 } p _ { j } ^ { 2 } \left( b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { j } ) \right) ^ { 2 } } } \\ { ~ } \\ { { \displaystyle \qquad \leq \overline { { \sigma } } ^ { 2 } \sum _ { j = 1 } ^ { K } p _ { j } ^ { 2 } \left( b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { j } ) \right) ^ { 2 } } } \\ { { \displaystyle \qquad \leq \overline { { \sigma } } ^ { 2 } \operatorname* { m a x } _ { j \in [ K ] } p _ { j } ^ { K } \sum _ { j = 1 } ^ { K } p _ { j } \left( b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { j } ) \right) ^ { 2 } } } \\ { { \displaystyle \qquad = \overline { { \sigma } } ^ { 2 } \operatorname* { m a x } _ { j \in [ K ] } p _ { j } b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { k } ) } . } \end{array}
$$

The final equality follows from the definition of $G _ { r , p }$ and the fact that it is positive definite,

$$
\begin{array} { l } { { \displaystyle \sum _ { j = 1 } ^ { K } p _ { j } \left( b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { j } ) \right) ^ { 2 } = \sum _ { j = 1 } ^ { K } p _ { j } \left( b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { j } ) \right) \left( b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { j } ) \right) } } \\ { { \displaystyle \qquad = \sum _ { j = 1 } ^ { K } p _ { j } \left( b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { j } ) \right) \left( b _ { r } ( x _ { j } ) ^ { \top } G _ { r , p } ^ { - 1 } b _ { r } ( x _ { k } ) \right) } } \\ { { \displaystyle \qquad = b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } \left( \sum _ { j = 1 } ^ { K } p _ { j } b _ { r } ( x _ { j } ) b _ { r } ( x _ { j } ) ^ { \top } \right) G _ { r , p } ^ { - 1 } b _ { r } ( x _ { k } ) } }  \\ { { \displaystyle \qquad = b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } \left( \sum _ { j = 1 } ^ { K } Q _ { r , p } ^ { - 1 } \right) } } \\ { { \displaystyle \qquad = b _ { r } ( x _ { k } ) ^ { \top } G _ { r , p } ^ { - 1 } G _ { r , p } ( x _ { k } ) } . }  \end{array}
$$

Consequently, by Jensen inequality and the definition of $G _ { r , { \pmb p } } .$

$$
\begin{array} { r l } { \mathbb { E } \| \bar { x } ^ { 2 } \| _ { \mathcal { X } } ^ { 2 } = \frac { \bar { x } ^ { 2 } } { 2 } \operatorname* { C o m } ( \log ( 1 0 ) ) } \\ &  = \{ \{ \begin{array} { l l } { \bar { x } ^ { 2 } } &  \mathrm { ~ R e ~ i ~ f ~ a ~ } \{ \begin{array} { l l } { 1 0 \leq r _ { 0 } \leq r _ { 0 } } \\ { \frac { \bar { x } ^ { 2 } } { 2 } } & { \mathrm { ~ R e ~ i ~ f ~ a ~ } \{ \begin{array} { r l } & { 1 0 } \\ { 1 0 \leq r _ { 0 } \leq r _ { 0 } } \\ { 1 0 \leq r _ { 0 } } \end{array} \} \} ^ { 1 / 2 } } \\ & { \leq \{ \{ \begin{array} { l l } { \bar { x } ^ { 2 } } & { \mathrm { ~ R e ~ i ~ f ~ a ~ } \{ \begin{array} { r l } & { 1 0 } \\ { 1 0 \leq r _ { 0 } \leq r _ { 0 } } \\ { 1 0 \leq r _ { 0 } } \end{array} \} \} ^ { 1 / 2 } } \\ & { \mathrm { ~ R e ~ i ~ f ~ a ~ } \{ \begin{array} { r l } & { 1 0 } \\ { 1 0 \leq r _ { 0 } \leq r _ { 0 } } \end{array} \} ^ { 1 / 2 } } \end{array} \} ^ { 1 / 2 } } \\ &  = - \eta _ { \mathrm { s t } } \frac { \bar { x } ^ { 2 } } { \sqrt { 1 + \sqrt { 1 + 1 } } } \{ \begin{array} { l l } &  \{ \begin{array} { l l } { \bar { x } ^ { 2 } } & { \mathrm { ~ R e ~ i ~ f ~ a ~ } \{ \begin{array} { r l } & { 1 0 } \\ { 1 0 \leq r _ { 0 } \leq r _ { 0 } } \\ { 1 0 \leq r _ { 0 } } \end{array} \} \} ^ { 1 / 2 } } \\ &  \qquad \mathrm { ~ R e ~ i ~ f ~ a ~ } \{ \begin{array} { r l } & { 1 0 } \\  1 0 \leq r _ { 0 } \leq r _ { 0 } \leq r _ { 0 } \leq r _ { 0 } \leq r _  0 \end{array} \end{array} \end{array} \end{array} \end{array} \end{array}
$$

where the last inequality follows from Equation (8) and $q _ { r } \asymp r ^ { - d }$

## F.5 Proof of Theorem 4

The proof is based on a weighted version of Assouad’s lemma (Tsybakov, 2009, Theorem 2.12) and adapts techniques from Audibert and Tsybakov (2007). We construct a family of potential signal functions on the hypercube, whose elements are statistically dificult to distinguish, while errors in identifying the right sign of each coordinate incur nonvanishing welfare loss.

We start by constructing a family of signal functions. Fix $r \in [ 2 a _ { 0 } K ^ { - 1 / d } , r _ { 0 } / 2 ]$ . For $x \in \mathbb { R } ^ { d }$ , define $\phi _ { 0 } ( x ) : = $ $( 1 - \| x \| _ { 2 } ^ { 2 } ) _ { + } ^ { \beta }$ and set $\begin{array} { r } { \phi _ { z , r } ( x ) : = \phi _ { 0 } \left( \frac { x - z } { r } \right) } \end{array}$ . By Lemma 2, there exists a set $\mathcal { Z } _ { r } = \{ z _ { 1 } , \dotsc , z _ { M } \} \subseteq \{ x _ { k } : k \in [ K ] \}$ such that $\| z _ { i } - z _ { j } \| _ { 2 } > 2 r$ for $\begin{array} { r } { i \neq j , i , \dot { j } \in [ M ] , M \geq \frac { 1 } { C 2 ^ { d + 1 } r ^ { d } } } \end{array}$ , for every $j \in [ M ]$ ，

$$
\sum _ { k = 1 } ^ { K } \phi _ { z _ { j } , r } ( x _ { k } ) ^ { 2 } \leq 2 ^ { d + 1 } C K r ^ { d } ,
$$

and

$$
\sum _ { j = 1 } ^ { M } Q _ { K } \left( B ( z _ { j } , r / 2 ) \right) \geq \frac { c } { C 2 ^ { 2 d + 1 } } ,
$$

where $0 < c < C < \infty$ are the constants from Assumption 4.

Let $\Omega : = \{ 0 , 1 \} ^ { M }$ and define $\begin{array} { r } { a : = \operatorname* { m i n } \left. H , \frac { L r ^ { \beta } } { 2 ^ { \beta + 1 } } , \frac { \overline { { \sigma } } } { 2 \sqrt { 2 ^ { d + 1 } C K r ^ { d } } } \right. } \end{array}$ . For every $\omega = ( \omega _ { 1 } , \hdots , \omega _ { M } ) \in \Omega$ , set

$$
g _ { \omega } ( x ) : = a \sum _ { j = 1 } ^ { M } ( 2 \omega _ { j } - 1 ) \phi _ { z _ { j } , r } ( x ) ,
$$

which will be our candidate signals, for which we need to show that $g _ { \omega } \in \mathcal { H } _ { d } ( \beta , L , H )$

We verify that $\| g _ { \omega } \| _ { \infty } \leq a \leq H$ . Since $\operatorname* { s u p p } ( \phi _ { 0 } ) = B ( 0 , 1 )$ , we have, for every $j \in [ M ]$ , supp $( \phi _ { z _ { i } , r } ) = B ( z _ { j } , r )$ By Lemma $2 , \| z _ { i } - z _ { j } \| _ { 2 } > 2 r$ for all $i \neq j$ , which implies that the closed balls $\{ B ( z _ { j } , r ) : j \in [ M ] \}$ , and therefore the supports of $\{ \phi _ { z _ { j } , r } : j \in [ M ] \}$ , are pairwise disjoint. Thus at most one summand in $g _ { \omega } ( x )$ is nonzero for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ . Since $0 \leq \bar { \phi } _ { z _ { j } , r } \leq 1$ , it follows that $\| g _ { \omega } \| _ { \infty } \leq a \leq H$

Next, we verify that $g _ { \omega }$ is β-H¨older with constant L. By Lemma 3

$$
\begin{array} { l } { \displaystyle | g _ { \omega } ( x ) - g _ { \omega } ( y ) | \leq a \sum _ { j = 1 } ^ { M } | \phi _ { z _ { j } , r } ( x ) - \phi _ { z _ { j } , r } ( y ) | } \\ { \displaystyle = a \sum _ { j = 1 } ^ { M } | \phi _ { z _ { j } , r } ( x ) - \phi _ { z _ { j } , r } ( y ) | 1 \{ x \in B ( z _ { j } , r ) \mathrm { o r } y \in B ( z _ { j } , r ) \} } \\ { \displaystyle \leq \frac { 2 ^ { \beta _ { a } } } { r ^ { \beta } } | | x - y | | _ { 2 } ^ { \beta } \sum _ { j = 1 } ^ { M } 1 \{ x \in B ( z _ { j } , r ) \mathrm { o r } y \in B ( z _ { j } , r ) \} } \\ { \displaystyle \leq \frac { 2 ^ { \beta + 1 } a } { r ^ { \beta } } | | x - y | | _ { 2 } ^ { \beta } } \\ { \displaystyle \leq L | | x - y | | _ { 2 } ^ { \beta } . } \end{array}
$$

where the penultimate inequality follows because each of x and y belongs to at most one ball. Together with $\| g _ { \omega } \| _ { \infty } \leq H$ , this proves that $g _ { \omega } \in \mathcal { H } _ { d } ( \beta , L , H )$

Next, we bound the total variation distance between the data distributions induced by neighboring hypotheses. We use a similar construction as in the proof of Theorem 1. Fix a probability law $Q _ { 0 }$ on $[ 0 , \infty ) ^ { \bar { K } }$ and let $\mathcal { P } _ { K } ^ { 0 } \subseteq \mathcal { P } _ { K }$ consist of the laws under which $\widehat { \sigma } \sim Q _ { 0 }$ and ${ \widehat { \sigma } } \bot \bot Z .$ . Write $P _ { g , \sigma } ^ { Z }$ for the $Z .$ -marginal of $P _ { g , \sigma }$ . By the same argument as in the proof of Theorem 1, on the submodel $\mathcal { P } _ { K } ^ { 0 }$ , it sufices to prove the lower bound for rules that do not take σb as an input.

For $j \in [ M ]$ , let $\boldsymbol { \omega } ^ { ( j ) }$ denote ω with coordinate j flipped, i.e., $\omega _ { i } ^ { ( j ) } = \omega _ { i }$ for all $i \neq j$ and $\omega _ { j } ^ { ( j ) } = 1 - \omega _ { j }$ . Let $P _ { \omega } : = P _ { g _ { \omega } , \overline { { \pmb { \sigma } } } } ^ { Z }$ , where $P _ { g _ { \omega } , \overline { { \sigma } } } \in \mathcal { P } _ { K } ^ { 0 }$ is the corresponding joint law. Then $P _ { \omega }$ and $P _ { \omega ^ { ( j ) } }$ are K-dimensional Gaussian distributions centered around $\{ g _ { \omega } ( x _ { k } ) \} _ { k = 1 } ^ { K }$ and $\{ g _ { \omega ^ { ( j ) } } ( x _ { k } ) \} _ { k = 1 } ^ { K }$ , respectively, with shared covariance matrix $\overline { { \sigma } } ^ { 2 } I _ { K }$ Since $g _ { \omega } , g _ { \omega ^ { ( j ) } } \in \mathcal { H } _ { d } ( \beta , L , H )$ and $\overline { { \sigma } } ~ \in ~ [ \underline { { \sigma } } , \overline { { \sigma } } ] ^ { K } , ~ P _ { \omega }$ and $P _ { \omega ^ { ( j ) } }$ are potential data-generating distributions of $Z = \left( Z _ { 1 } , \ldots , Z _ { K } \right)$

By Barsov and Ulyanov (1987); Devroye et al. (2018),

$$
d _ { \mathrm { T V } } ( P _ { \omega } , P _ { \omega ^ { ( j ) } } ) = 2 \Phi \left( \frac { \sqrt { \sum _ { k = 1 } ^ { K } ( g _ { \omega } ( x _ { k } ) - g _ { \omega ^ { ( j ) } } ( x _ { k } ) ) ^ { 2 } } } { 2 \overline { { \sigma } } } \right) - 1 .
$$

Since ω and $\boldsymbol { \omega } ^ { ( j ) }$ difer only in coordinate $j ,$

$$
\begin{array} { c c c } { \displaystyle \frac { 1 } { 2 \overline { { \sigma } } } \sqrt { \displaystyle { \sum _ { k = 1 } ^ { K } } ( g _ { \omega } ( x _ { k } ) - g _ { \omega ^ { ( j ) } } ( x _ { k } ) ) ^ { 2 } } = \displaystyle \frac { a } { \overline { { \sigma } } } \sqrt { \displaystyle { \sum _ { k = 1 } ^ { K } \phi _ { z _ { j } , r } ( x _ { k } ) ^ { 2 } } } } \\ { \displaystyle \leq \frac { a \sqrt { 2 ^ { d + 1 } C K r ^ { d } } } { \overline { { \sigma } } } } \\ { \displaystyle \leq \frac { 1 } { 2 } , } \end{array}
$$

where the first inequality follows from Lemma 2 and the second from the definition of a. Therefore,

$$
\operatorname* { m a x } _ { \omega \in \Omega } \operatorname* { m a x } _ { j \in [ M ] } d _ { \mathrm { T V } } ( P _ { \omega } , P _ { \omega ^ { ( j ) } } ) \leq 2 \Phi \left( \frac { 1 } { 2 } \right) - 1 .\tag{12}
$$

Having established that $g _ { \omega } \in \mathcal { H } _ { d } ( \beta , L , H )$ and that neighboring signals are dificult to distinguish, we complete the proof by reducing the policy regret to a weighted Hamming risk over the hypercube.

For each $j \in [ M ]$ , define $I _ { j } : = \{ k \in [ K ] : x _ { k } \in B ( z _ { j } , r / 2 ) \}$ and $\begin{array} { r } { v _ { j } = \sum _ { k \in I _ { j } } p _ { k } } \end{array}$ . Note that $\{ I _ { j } \} _ { j = 1 } ^ { M }$ are pairwise disjoint. For every $k \in I _ { j }$

$$
\phi _ { z _ { j } , r } ( x _ { k } ) = \left( 1 - { \frac { \| x _ { k } - z _ { j } \| _ { 2 } ^ { 2 } } { r ^ { 2 } } } \right) ^ { \beta } \geq \left( 1 - { \frac { 1 } { 4 } } \right) ^ { \beta } = \left( { \frac { 3 } { 4 } } \right) ^ { \beta } ,
$$

and thus

$$
| g _ { \omega } ( x _ { k } ) | \geq a \left( \frac { 3 } { 4 } \right) ^ { \beta } .
$$

Moreover, by Assumption 4,

$$
v _ { j } = \sum _ { k \in I _ { j } } p _ { k } = Q _ { K } \left( B ( z _ { j } , r / 2 ) \right) \geq c \left( \frac { r } { 2 } \right) ^ { d } > 0 .\tag{13}
$$

Fix a decision rule $\widetilde { \pi } : \mathbb { R } ^ { K } \to [ 0 , 1 ] ^ { K }$ . Using that, for $X \in [ 0 , 1 ]$ and $b \in \{ 0 , 1 \}$ , |E[X] − b| = E[|X − b|],

$$
\begin{array} { r l } { \underset { \varrho \to \mathbb { R } _ { + } ( \ell , \hat { \varsigma } , \ell ) } { \operatorname* { s u p } } \mathcal { H } _ { \mathcal { K } } ( \ell , \hat { \varrho } _ { \varphi } , \hat { \varsigma } ) \geq } & { \frac { 1 } { \mathrm { s u c k } } \int _ { \mathbb { R } _ { + } } \chi _ { \mathcal { G } } \mathbb { I } _ { \mathcal { G } } \chi _ { \mathcal { G } } \mathbb { I } _ { \mathcal { G } } ^ { * } } \\ { \underset { \varrho \to \mathbb { R } _ { + } ( \ell , \hat { \varsigma } , \ell ) \leq \mathbb { I } _ { \mathcal { G } } } { \operatorname* { s u p } } \mathbb { I } _ { \mathcal { G } } \chi _ { \mathcal { G } } ( \ell ) \chi _ { \mathcal { G } } \mathbb { I } _ { \mathcal { G } } ( \ell ) - \mathbb { I } _ { \mathcal { G } } \chi _ { \mathcal { G } } ( \ell , \hat { \varsigma } ) > 0 ) \big | } \\ & { \quad - \underset { \varrho \leq \mathbb { I } _ { \mathcal { G } } } { \operatorname* { s u p } } \sum _ { \ell = 1 } ^ { N } \sum _ { \mathcal { G } } \rho _ { \mathcal { G } } ( \boldsymbol { x } _ { \ell } ) \mathbb { I } _ { \mathcal { G } } \chi _ { \mathcal { G } } \mathbb { I } _ { \mathcal { G } } ( \ell ) - \mathbb { I } _ { \mathcal { G } } ( \rho _ { \mathcal { G } } ( \boldsymbol { x } _ { \ell } ) > 0 ) \big | } \\ & { \quad = \underset { \varrho \leq \mathbb { I } _ { \mathcal { G } } } { \operatorname* { s u p } } \sum _ { \ell = 1 } ^ { N } \rho _ { \mathcal { G } } ( \boldsymbol { x } _ { \ell } ) \mathbb { I } _ { \mathcal { G } } ( \ell ) - \mathbb { I } _ { \mathcal { G } } ( \rho _ { \mathcal { G } } ( \boldsymbol { x } _ { \ell } ) > 0 ) \big | } \\ &  \quad \geq \frac { 1 } { \mathrm { s u c k } } \sum _ { \ell = 1 } ^ { N } \sum _ { \mathcal { G } } \rho _  \mathcal  \end{array}
$$

The final equality follows because, for every $k \in I _ { j } , \phi _ { z _ { j } , r } ( x _ { k } ) > 0$ and hence ${ \bf 1 } \{ g _ { \omega } ( x _ { k } ) > 0 \} = \omega _ { j }$ Given ${ \widetilde { \pi } } .$ , define the estimator $\widetilde { \omega } : \mathbb { R } ^ { K } \to [ 0 , 1 ] ^ { M }$ entrywise by

$$
\widetilde { \omega } _ { j } ( Z ) : = \frac { 1 } { v _ { j } } \sum _ { k \in I _ { j } } p _ { k } \widetilde { \pi } _ { k } ( Z ) ,
$$

where we recall $v _ { j } > 0$ by Assumption 4 and Equation (13). Since $\omega _ { j } \in \{ 0 , 1 \}$ and $\widetilde { \pi } _ { k } ( Z ) , \widetilde { \omega } _ { j } ( Z ) \in [ 0 , 1 ]$

$$
\sum _ { k \in I _ { j } } p _ { k } \lvert \widetilde { \pi } _ { k } ( Z ) - \omega _ { j } \rvert = v _ { j } \lvert \widetilde { \omega } _ { j } ( Z ) - \omega _ { j } \rvert .
$$

Applying Lemma 1 together with Equation (12),

$$
\begin{array} { r l } { \underset { \leq t \leq \frac { \rho _ { 1 } } { \rho _ { 1 } } } { \operatorname* { s u p } } } & { B _ { t } [ T _ { 2 } \rho _ { t } , \eta ] \geq 0 \left( \frac { \eta _ { 1 } ^ { 2 } } { 4 } \right) ^ { \frac { 3 } { \rho _ { 1 } } } \underset { \leq t \leq \frac { \rho _ { 2 } } { \rho _ { 2 } } } { \operatorname* { s u p } } \sum _ { \eta \leq 1 } \frac { \eta _ { 1 } } { \rho _ { 1 } } \mathbb { E } _ { \eta } [ \eta _ { 2 } ( \xi ) - \varphi _ { t } ] } \\ & { \qquad = - \eta \left( \frac { \eta _ { 2 } } { 4 } \right) ^ { \frac { 3 } { \rho _ { 1 } } } \underset { \leq t \leq \frac { \rho _ { 2 } } { \rho _ { 2 } } } { \operatorname* { s u p } } \sum _ { \eta \leq 1 } \mathbb { E } _ { \eta } [ \eta _ { 2 } ( \xi ) - \omega _ { t } ] } \\ & { \qquad = \eta _ { 1 } ^ { - 1 } \left( \frac { \eta _ { 1 } } { 4 } \right) ^ { \frac { 3 } { \rho _ { 1 } } } \underset { \leq t \leq \frac { \rho _ { 2 } } { \rho _ { 2 } } } { \operatorname* { s u p } } \sum _ { \eta \leq 1 } \mathbb { E } _ { \eta } [ \eta _ { 2 } ( \xi ) - \omega _ { t } ] } \\ & { \qquad \leq \eta _ { 1 } ^ { - 1 } \left( \frac { \eta _ { 2 } } { 4 } \right) ^ { \frac { 3 } { \rho _ { 1 } } } \underset { \leq t \leq \frac { \rho _ { 2 } } { \rho _ { 2 } } } { \operatorname* { s u p } } \sum _ { \eta \leq 1 } \mathbb { E } _ { \eta } [ \eta _ { 3 } ( \xi ) - \omega _ { t } ] } \\ &  \qquad \leq \eta \left( \frac { \eta _ { 3 } } { 4 } \right) ^ { \frac { 3 } { \rho _ { 1 } } } \lambda ^ { 1 - \frac { 3 } { 2 } } \underset  \leq t \leq \frac { \rho _ { 2 } } \end{array}
$$

where the final inequality follows from the mass property in Lemma 2. Since this holds for every measurable decision rule $\widetilde { \pi } : \mathbb { R } ^ { K } \to [ \bar { 0 } , 1 ] ^ { K }$ , taking the infimum and applying the reduction on $\mathcal { P } _ { K } ^ { 0 } \subseteq \mathcal { P } _ { K }$ gives

$$
\begin{array} { r l } { \underset { \substack { \pi : \mathbb { R } ^ { K } \times [ 0 , \infty ] ^ { K } \to [ 0 , 1 ] ^ { K } } } { \operatorname* { i n f } } \underset { g \in \mathcal { H } _ { d } ( \beta , L , H ) } { \operatorname* { s u p } } } & { R _ { K } ( P _ { g , \sigma } , \tilde { \pi } ) \geq \underset { \substack { \pi : \mathbb { R } ^ { K } \times [ 0 , \infty ] ^ { K } \to [ 0 , 1 ] ^ { K } } } { \operatorname* { i n f } } \underset { g \in \mathcal { H } _ { d } ( \beta , L , H ) } { \operatorname* { s u p } } } & { R _ { K } ( P _ { g , \sigma } , \tilde { \pi } ) } \\ & { = \underset { \substack { \pi : \mathbb { R } ^ { K } \to [ 0 , 1 ] ^ { K } } } { \operatorname* { i n f } } \underset { g \in \mathcal { H } _ { d } ( \beta , L , H ) } { \operatorname* { s u p } } } & { R _ { K } ( P _ { g , \sigma } , \tilde { \pi } ) } \\ & { \geq a \left( \frac { 3 } { 4 } \right) ^ { \beta } \Phi \left( - \frac { 1 } { 2 } \right) \frac { c } { C 2 ^ { 2 d + 1 } } . } \end{array}
$$

By the definition of $^ { a , }$

$$
a \geq \operatorname* { m i n } \left\{ 1 , \frac { 1 } { 2 ^ { \beta + 1 } } , \frac { 1 } { 2 \sqrt { 2 ^ { d + 1 } C } } \right\} \operatorname* { m i n } \left\{ H , L r ^ { \beta } , \frac { \overline { { \sigma } } } { \sqrt { K r ^ { d } } } \right\} .
$$

Consequently,

$$
\operatorname* { i n f } _ { \substack { \pi : \mathbb { R } ^ { K } \times [ 0 , \infty ) ^ { K }  [ 0 , 1 ] ^ { K } \operatorname* { s u p } _ { g \in \mathcal { H } _ { d } ( \beta , L , H ) } } } R _ { K } ( P _ { g , \sigma } , \widetilde { \pi } ) \geq c _ { 1 } \operatorname* { m i n } \{ H , L r ^ { \beta } , \frac { \overline { { \sigma } } } { \sqrt { K r ^ { d } } } \} ,
$$

where

$$
c _ { 1 } : = \left( \frac { 3 } { 4 } \right) ^ { \beta } \Phi \left( - \frac { 1 } { 2 } \right) \frac { c } { C 2 ^ { 2 d + 1 } } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { 2 ^ { \beta + 1 } } , \frac { 1 } { 2 \sqrt { 2 ^ { d + 1 } C } } \right\} .
$$

The constant $c _ { 1 } > 0$ is independent of K and r, proving the claim.

## F.6 Proof of Proposition 2

By Assumption $1 - 2 , \widehat { g } _ { w } ( x _ { k } ) \sim \mathcal { N } ( m _ { w } ( x _ { k } ) , s _ { w } ( x _ { k } ) ^ { 2 } )$ . Therefore

$$
\begin{array} { r l } { \sqrt { n } W _ { P _ { 3 , x } } ( \widetilde { \pi } _ { w } ) = } & { \displaystyle \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) \mathbb { E } _ { g , \sigma } \left[ \overline { { \boldsymbol { w } } } _ { w , \ k } ( Z ) \right] } \\ & { \displaystyle = \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) \mathbb { E } _ { g , \sigma } \left[ \mathbf { 1 } \left\{ \widehat { \mathcal { W } } _ { w } ( x _ { k } ) > 0 \right\} \right] } \\ & { \displaystyle = \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) P _ { 3 , \sigma } \left( \widehat { \mathcal { W } } _ { w } ( x _ { k } ) > 0 \right) } \\ & { \displaystyle = \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) P _ { 3 , \sigma } \left( \widehat { \mathcal { W } } _ { w } ( x _ { k } ) > 0 \right) } \\ & { \displaystyle = \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) \Phi \left( \frac { m _ { \sigma } \left( \widehat { \mathcal { W } } _ { k } \right) } { s _ { k } \left( \widehat { \mathcal { W } } _ { k } \right) } \right) } \\ & { \displaystyle = \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) \psi _ { w , k } , } \\ & { \displaystyle = \sum _ { k = 1 } ^ { K } p _ { k } g ( x _ { k } ) \psi _ { w , k } , } \end{array}
$$

Similarly,

$$
\begin{array} { c l } { { \displaystyle \mathrm { C o v } _ { p } ( g , \psi _ { w } ) = \sum _ { k = 1 } ^ { K } p _ { k } ( g ( x _ { k } ) - \bar { g } _ { p } ) ( \psi _ { w , k } - \bar { \psi } _ { w , p } ) } } \\ { { { } } } \\ { { { } } } \\ { { { } } } \end{array}
$$

Combining the expressions gives

$$
\sqrt { n } W _ { P _ { g , \sigma } } ( \widetilde { \pi } _ { w } ) = \bar { g } _ { p } \bar { \psi } _ { w , p } + \mathrm { C o v } _ { p } ( g , \psi _ { w } ) ,
$$

which proves the decomposition. Next, we show the pairwise representation of $\mathrm { C o v } _ { p } ( g , \psi _ { w } )$

$$
\begin{array} { r l } { \zeta ( \mathbf c ) _ { n ^ { \prime } , \varepsilon } ( g _ { 2 } , g _ { n ^ { \prime } } ) } & { = \displaystyle \sum _ { j = 1 } ^ { K } R e f ( \mathbf x _ { j } ) \psi _ { n , \varepsilon } - \overline { \beta } _ { n } \overline { \psi } _ { n , \varepsilon } } \\ & { = \displaystyle \sum _ { j = 1 } ^ { K } R e f ( \mathbf x _ { j } ) \psi _ { n , \varepsilon } - \left( \sum _ { j = 1 } ^ { K } R e ^ { \beta } ( \mathbf x _ { j } ) \psi _ { n , \varepsilon } \right) \left( \sum _ { i = 1 } ^ { K } p _ { i } \psi _ { n , \varepsilon } \right) } \\ & { = \displaystyle \sum _ { k = 1 } ^ { K } p _ { i } \psi ( \mathbf x _ { k } ) \psi _ { n , \varepsilon } - \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { K } p _ { i } p _ { i } \psi _ { k } ( \mathbf x _ { k } ) \psi _ { n , \varepsilon } } \\ & { = \displaystyle \sum _ { k = 1 } ^ { K } p _ { i } \psi ( \mathbf x _ { k } ) \psi _ { n , \varepsilon } - \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { K } p _ { i } p _ { i } \psi _ { n , \varepsilon } ( \mathbf x _ { k } ) \psi _ { n , \varepsilon } } \\ &  = \displaystyle - \frac 1 2 \sum _ { k = 1 } ^ { K } p _ { i } \psi ( \mathbf x _ { k } ) \psi _ { n , \varepsilon } + \displaystyle \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { K } p _ { i } \psi _ { n , \varepsilon } ( \mathbf x _ { k } ) \psi _ { n , \varepsilon } - \displaystyle \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { K } \sum _ { i = 1 } ^ { K } p _ { i } p _ { i } \psi _ { n , \varepsilon } ( \mathbf x _ { k } ) \psi _ { n , \varepsilon } - \displaystyle \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { K } p _ { i } p _ { i } \psi _ { n , \varepsilon } ( \mathbf x _ { j } ) \psi _ { n , \varepsilon }  \end{array}
$$

## F.7 Proof of Proposition 3

Define the scaled welfare loss $\begin{array} { r } { L _ { h } ( \pi ) : = \sqrt { n } \{ V _ { h } ^ { * } - V _ { h } ( \pi ) \} = \sum _ { k = 1 } ^ { K } p _ { k } h _ { k } ( 1 \{ h _ { k } > 0 \} - \pi _ { k } ) } \end{array}$ . For each $r \in \Lambda _ { K }$ , write

$$
\bar { L } _ { h , r } : = \frac { 1 } { F } \sum _ { f = 1 } ^ { F } L _ { h } ( \widetilde { \pi } _ { w _ { r } } ( Z ^ { ( - f ) } ) ) = \frac { 1 } { F } \sum _ { f = 1 } ^ { F } \sum _ { k = 1 } ^ { K } p _ { k } h _ { k } ( \mathbf { 1 } \{ h _ { k } > 0 \} - \widetilde { \pi } _ { w _ { r } , k } ( Z ^ { ( - f ) } ) ) .
$$

Then $\mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , r } ] = R _ { K } ( P _ { h , \sigma _ { F } } , \widetilde { \pi } _ { w _ { r } } )$

Next, for each fold $f ,$ define

$$
\Delta _ { r , f } : = \sum _ { k = 1 } ^ { K } p _ { k } \mathopen { } \mathclose \bgroup \left( Z _ { k } ^ { ( f ) } - h _ { k } \aftergroup \egroup \right) \widetilde { \pi } _ { w _ { r } , k } \mathopen { } \mathclose \bgroup \left( Z ^ { ( - f ) } \aftergroup \egroup \right) .
$$

Since $Z _ { k } ^ { ( f ) } = h _ { k } + \big ( Z _ { k } ^ { ( f ) } - h _ { k } \big )$ , our selection criterion (depending only on observables) decomposes

$$
\widehat { W } ( r ) : = \frac { 1 } { F } \sum _ { f = 1 } ^ { F } \sum _ { k = 1 } ^ { K } p _ { k } Z _ { k } ^ { ( f ) } \widetilde { \pi } _ { w _ { r } , k } ( Z ^ { ( - f ) } ) = \sqrt { n } V _ { h } ^ { * } - \bar { L } _ { h , r } + \frac { 1 } { F } \sum _ { f = 1 } ^ { F } \Delta _ { r , f } .
$$

Thus, if $\widehat { r }$ maximizes $\widehat { W } ( r )$ it also maximizes $\begin{array} { r } { - \bar { L } _ { h , r } + \frac { 1 } { F } \sum _ { f = 1 } ^ { F } \Delta _ { r , f } } \end{array}$ . Thus for every fixed $r \in \Lambda _ { K }$

$$
\begin{array} { r l } & { \bar { L } _ { h , \widehat { r } } \leq \bar { L } _ { h , r } + \displaystyle \frac { 1 } { F } \sum _ { f = 1 } ^ { F } ( \Delta _ { \widehat { r } , f } - \Delta _ { r , f } ) } \\ & { \qquad \leq \bar { L } _ { h , r } + \displaystyle \frac { 1 } { F } \sum _ { f = 1 } ^ { F } \left( \displaystyle \operatorname* { m a x } _ { u \in \Lambda _ { K } } \Delta _ { u , f } - \Delta _ { r , f } \right) } \end{array}\tag{14}
$$

Now, conditional on the remaining folds $\mathcal { D } _ { - f } = ( Z ^ { ( j ) } ) _ { j \neq f } , \ : \Delta _ { r , f }$ is a centered Gaussian random variable with variance bounded by $\begin{array} { r } { s _ { K } ^ { 2 } = F \overline { { \sigma } } ^ { 2 } \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } \le \frac { C F \overline { { \sigma } } ^ { 2 } } { K } } \end{array}$ , since max<sub>k</sub> $p _ { k } \lesssim K ^ { - 1 }$ by assumption. By the Gaussian maximum inequality Wainwright (2019, Exercise $2 . 1 2 ( \mathrm { a } ) )$ ,

$$
\mathbb { E } _ { P _ { h , \sigma } } [ \operatorname* { m a x } _ { r \in \Lambda _ { K } } \Delta _ { r , f } \ | \ \mathcal { D } _ { - f } ] \leq s _ { K } \sqrt { 2 \log ( | \Lambda _ { K } | ) } .
$$

Thus, taking the expectation in Equation (14) and minimizing over r gives

$$
\mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , \widehat { r } } ] \leq \operatorname* { m i n } _ { r \in \Lambda _ { K } } \mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , r } ] + s _ { K } \sqrt { 2 \log ( | \Lambda _ { K } | ) }
$$

Assume for now that $R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } _ { w _ { \widehat { r } } } ) \leq F \mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , \widehat { r } } ]$ . Then, since $\mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , r } ] = R _ { K } ( P _ { h , \sigma _ { F } } , \widetilde { \pi } _ { w _ { r } } )$ and $s _ { K } ~ \le$ $\sqrt { \frac { C F { \overline { { \sigma } } } ^ { 2 } } { K } }$ by the above,

$$
\begin{array} { r l } & { R _ { K } ( P _ { h , \sigma } , \widetilde { \boldsymbol { \pi } } _ { w _ { \widehat { r } } } ) \leq F \mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , \widehat { r } } ] } \\ & { \leq F \operatorname* { m i n } _ { r \in \Lambda _ { K } } \mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , r } ] + F s _ { K } \sqrt { 2 \log ( | \Lambda _ { K } | ) } } \\ & { \leq F \operatorname* { m i n } _ { r \in \Lambda _ { K } } R _ { K } ( P _ { h , \sigma _ { F } } , \widetilde { \boldsymbol { \pi } } _ { w _ { r } } ) + F \sqrt { \frac { C F \overline { { \sigma } } ^ { 2 } 2 \log ( | \Lambda _ { K } | ) } { K } } } \\ & { \lesssim \operatorname* { m i n } _ { r \in \Lambda _ { K } } R _ { K } ( P _ { h , \sigma _ { F } } , \widetilde { \boldsymbol { \pi } } _ { w _ { r } } ) + \sqrt { \frac { \log ( | \Lambda _ { K } | ) } { K } } , } \end{array}
$$

which proves the claim as F is fixed. It remains to show that

$$
R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } _ { w _ { \widehat { r } } } ) \leq F \mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , \widehat { r } } ] .
$$

Since $\begin{array} { r } { Z _ { k } = \frac { 1 } { F } \sum _ { f = 1 } ^ { F } Z _ { k } ^ { ( f ) } } \end{array}$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { \rho \rho \rho } ^ { \lambda } ( \mathbf { x } , \mathbf { x } , 0 ) } \\ & { = \frac { \lambda } { \lambda } \int _ { 0 } ^ { \infty } \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) } \\ & { \quad - \frac { \lambda } { \lambda } \int _ { 0 } ^ { \infty } \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) } \\ & { = \frac { \lambda } { \lambda } \int _ { 0 } ^ { \infty } \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) } \\ & { \quad - \frac { \lambda } { \lambda } \int _ { 0 } ^ { \infty } \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) } \\ & { = \frac { \lambda } { \lambda } \int _ { 0 } ^ { \infty } \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) } \\ & { \quad - \frac { \lambda } { \lambda } \int _ { 0 } ^ { \infty } \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) } \\ &  \quad - \frac { \lambda } { \lambda } \int _ { 0 } ^ { \infty } \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \lambda ( \mathbf { x } , 0 ) \mathrm { d } \end{array}
$$

This inequality holds for every $r \in \Lambda _ { K }$ , including ${ \widehat { r } } .$ Setting $r = { \widehat { r } }$ and taking expectations gives

$$
R _ { K } ( P _ { h , \sigma } , \widetilde { \pi } _ { w _ { \widehat { r } } } ) = \mathbb { E } _ { P _ { h , \sigma } } [ L _ { h } ( \widetilde { \pi } _ { w _ { \widehat { r } } } ( Z ) ) ] \leq F \mathbb { E } _ { P _ { h , \sigma } } [ \bar { L } _ { h , \widehat { r } } ] ,
$$

which completes the proof.

## F.8 Technical Lemmas

Lemma 1 (Weighted Assouad Lemma for Randomized Decisions) Let $M , K \ge 1$ , let $\Omega : = \{ 0 , 1 \} ^ { M }$ and let $\{ P _ { \omega } : \omega \in \Omega \}$ be Borel probability measures on $\mathbb { R } ^ { K }$ . For $j \in [ M ]$ , let $\boldsymbol { \omega } ^ { ( j ) }$ denote the vector obtained from $\omega$ by flipping its jth coordinate, $i . e . , \omega ^ { ( j ) } : = ( \omega _ { 1 } , \ldots , \omega _ { j - 1 } , 1 - \omega _ { j } , \overleftarrow { \omega _ { j + 1 } } , \ldots , \omega _ { M } )$ . Suppose that

$$
\operatorname* { m a x } _ { \omega \in \Omega } \operatorname* { m a x } _ { j \in [ M ] } \mathrm { d } _ { \mathrm { T V } } ( P _ { \omega } , P _ { \omega ^ { ( j ) } } ) \leq \Delta ,
$$

for some $\Delta \in [ 0 , 1 )$ , where $\mathrm { d _ { T V } } ( P , Q ) : = \operatorname* { s u p } _ { A \in { \mathcal { B } } ( \mathbb { R } ^ { K } ) } | P ( A ) - Q ( A ) |$ is the total variation distance. Then, for any nonnegative weights $v _ { 1 } , \ldots , v _ { M }$

$$
\operatorname* { i n f } _ { \widetilde { \omega } : \mathbb R ^ { K } \to [ 0 , 1 ] ^ { M } } \operatorname* { m a x } _ { \omega \in \Omega } \mathbb { E } _ { \omega } \left[ \sum _ { j = 1 } ^ { M } v _ { j } | \widetilde { \omega } _ { j } ( Z ) - \omega _ { j } | \right] \geq \frac { 1 - \Delta } { 2 } \sum _ { j = 1 } ^ { M } v _ { j } ,
$$

where the infimum ranges over all measurable decision rules $\widetilde { \omega } : \mathbb { R } ^ { K } \to [ 0 , 1 ] ^ { M }$ , and $\mathbb { E } _ { \omega }$ denotes expectation under $P _ { \omega }$ . The same lower bound holds with the maximum replaced by the average over $\Omega$

Proof: The proof follows from similar arguments as in Tsybakov (2009, Theorem 2.12), but extended to nonnegative weights and randomized decisions.

Fix any measurable decision rule $\widetilde { \omega } : \mathbb { R } ^ { K } \to [ 0 , 1 ] ^ { M }$ . Bounding the maximum by the average over Ω gives

$$
\begin{array} { r l r } & { } & { \underset { \omega \in \Omega } { \operatorname* { m a x } } \mathbb { E } _ { \omega } \left[ \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } | \widetilde { \omega } _ { j } ( Z ) - \omega _ { j } | \right] \geq \frac { 1 } { 2 ^ { M } } \displaystyle \sum _ { \omega \in \Omega } \mathbb { E } _ { \omega } \left[ \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } | \widetilde { \omega } _ { j } ( Z ) - \omega _ { j } | \right] } \\ & { } & { \quad \quad \quad \quad = \frac { 1 } { 2 ^ { M } } \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } \displaystyle \sum _ { \omega \in \Omega } \mathbb { E } _ { \omega } \left[ | \widetilde { \omega } _ { j } ( Z ) - \omega _ { j } | \right] . } \end{array}
$$

For each $j \in [ M ]$ , define $\Omega _ { j , 0 } : = \{ \omega \in \Omega : \omega _ { j } = 0 \}$ . Using $\widetilde { \omega } _ { j } ( Z ) \in [ 0 , 1 ]$

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \alpha \in \mathbb { R } } \mathbb { E } _ { \omega } [ \displaystyle \sum _ { j = 1 } ^ { M } \delta | \tilde { \mathcal { E } } _ { j } ( \boldsymbol { \mathcal { Z } } ) - \omega _ { i j } | ] \geq \frac { 1 } { 2 M } \displaystyle \sum _ { j = 1 } ^ { M } \mathbb { E } _ { \boldsymbol { \mu } } \sum _ { \alpha \in \mathbb { R } } \mathbb { E } _ { \boldsymbol { \mu } } [ \tilde { \mathcal { Z } } _ { j } ( \boldsymbol { \mathcal { Z } } ) - \omega _ { i j } | ] } & { } \\ { = \displaystyle \frac { 1 } { 2 M } \displaystyle \sum _ { j = 1 } ^ { M } \mathbb { E } _ { \boldsymbol { \mu } } \sum _ { \omega \in \mathcal { I } _ { \mathcal { A } } ( \boldsymbol { \mathcal { Z } } ) , \boldsymbol { \mu } } \mathbb { E } _ { \omega } [ \mathbb { E } _ { \omega } [ \tilde { \mathcal { A } } _ { j } ( \boldsymbol { \mathcal { Z } } ) ] ] + \mathbb { E } _ { \omega \in \mathcal { I } _ { \Psi } } [ \mathbb { 1 } - \tilde { \mathcal { B } } _ { j } ( \boldsymbol { \mathcal { Z } } ) ] ] \Bigg \} } & { } \\ { = \displaystyle \frac { 1 } { 2 M } \displaystyle \sum _ { j = 1 } ^ { M } \nu _ { j } \displaystyle \sum _ { \omega \in \mathbb { R } _ { j , \mathcal { A } } } \{ \mathbb { E } _ { \omega } [ \tilde { \mathcal { B } } _ { j } ( \boldsymbol { \mathcal { Z } } ) ] + \mathbb { E } _ { \omega \omega } [ \boldsymbol { 1 } - \tilde { \mathcal { B } } _ { j } ( \boldsymbol { \mathcal { Z } } ) ] \} } & { } \\  = \displaystyle \frac { 1 } { 2 M } \displaystyle \sum _ { j = 1 } ^ { M } \nu _ { j } \displaystyle \sum _ { \omega \in \mathbb { R } _ { j , \mathcal { A } } } [ \mathbb { E } _ { \omega } [ \tilde { \mathcal { B } } _ { j } ( \boldsymbol { \mathcal { Z } } ) ] - \mathbb { E } _ { \omega } [ \mathbb { E } _ { j } ( \boldsymbol  \ \end{array}
$$

For any measurable $\delta : \mathbb { R } ^ { K }  [ 0 , 1 ]$ and any probability measures $P , Q$ on R $K$ ,

$$
\begin{array} { r l } { \displaystyle | \mathbb { E } _ { P } [ \delta ( Z ) ] - \mathbb { E } _ { Q } [ \delta ( Z ) ] | = \left| \int _ { 0 } ^ { 1 } \left\{ P ( \delta ( Z ) > t ) - Q ( \delta ( Z ) > t ) \right\} d t \right| } & { } \\ { \displaystyle } & { \leq \int _ { 0 } ^ { 1 } | P ( \delta ( Z ) > t ) - Q ( \delta ( Z ) > t ) | d t } \\ { \displaystyle } & { \leq \int _ { 0 } ^ { 1 } \mathrm { d } _ { \mathrm { T V } } ( P , Q ) d t } \\ { \displaystyle } & { = \mathrm { d } _ { \mathrm { T V } } ( P , Q ) . } \end{array}
$$

Applying this inequality with $\delta = \widetilde { \omega } _ { j } , P = P _ { \omega ^ { ( j ) } }$ , and $Q = P _ { \omega }$ and recalling that $\mathrm { d _ { T V } } ( P _ { \omega } , P _ { \omega ^ { ( j ) } } ) \leq \Delta$ for all $\omega \in \Omega , j \in [ M ]$ 2

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \omega \in \Omega } \mathbb { E } _ { \omega } \left[ \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } | \widetilde { \omega } _ { j } ( Z ) - \omega _ { j } | \right] \geq \frac { 1 } { 2 ^ { M } } \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } \displaystyle \sum _ { \omega \in \Omega _ { j } , \omega } \big [ 1 - \big | \mathbb { E } _ { \omega ( j ) } \big [ \widetilde { \omega } _ { j } ( Z ) \big ] - \mathbb { E } _ { \omega } \big [ \widetilde { \omega } _ { j } ( Z ) \big ] \big | \big ] } & { } \\ { \displaystyle \geq \frac { 1 } { 2 ^ { M } } \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } \displaystyle \sum _ { \omega \in \Omega _ { j } , \omega } \big [ 1 - \mathrm { d r y } \big ( F _ { \omega } , P _ { \omega ( z ) } ) \big ] } & { } \\ { \displaystyle \geq \frac { 1 } { 2 ^ { M } } \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } \displaystyle \sum _ { \omega \in \Omega _ { j } , \omega } \big [ 1 - \Delta \big ] } & { } \\ { \displaystyle } & { = \frac { 1 - \Delta } { 2 } \displaystyle \sum _ { j = 1 } ^ { M } v _ { j } , } \end{array}
$$

where we used $| \Omega _ { j , 0 } | = 2 ^ { M - 1 }$ for every $j \in [ M ]$ . Since this holds for every measurable $\widetilde { \omega } : \mathbb { R } ^ { K } \to [ 0 , 1 ] ^ { M }$ , taking the infimum over ω proves the claim. □

Lemma 2 (Low-Norm Packing) Suppose Assumption 4 holds with constants $0 < c < C < \infty , a _ { 0 } , r _ { 0 } > 0$ and let $r \in [ 2 a _ { 0 } K ^ { - 1 / d } , r _ { 0 } / 2 ]$ Define $\phi _ { 0 } ( x ) : = ( 1 - \| x \| _ { 2 } ^ { 2 } ) _ { + } ^ { \beta }$ and set $\begin{array} { r } { \phi _ { z , r } ( x ) : = \phi _ { 0 } \left( \frac { x - z } { r } \right) } \end{array}$ . Then there exists $\mathcal { Z } _ { r } = \{ z _ { 1 } , \dotsc , z _ { M } \} \subseteq \{ x _ { k } : k \in [ K ] \}$ satisfying the following properties:

(a) Separation: $\| z _ { i } - z _ { j } \| _ { 2 } > 2 r \ f o r \ i \neq j \in [ M ]$

(b) Cardinality: $\begin{array} { r } { M \geq \frac { 1 } { C 2 ^ { d + 1 } r ^ { d } } } \end{array}$

(c) Task Norm: For every $j \in [ M ] , \sum _ { k = 1 } ^ { K } \phi _ { z _ { j } , r } ( x _ { k } ) ^ { 2 } \leq 2 ^ { d + 1 } C K r ^ { d } .$

(d) Mass: $\begin{array} { r } { \sum _ { j = 1 } ^ { M } Q _ { K } \left( B ( z _ { j } , r / 2 ) \right) \geq \frac { c } { C 2 ^ { 2 d + 1 } } , } \end{array}$

Proof: Let $\widetilde { \mathcal { Z } } _ { r } = \{ \widetilde { z } _ { 1 } , \dots , \widetilde { z } _ { \widetilde { M } } \}$ be a maximal 2r-packing of the distinct task locations, i.e., $\| \widetilde { z } _ { i } - \widetilde { z } _ { j } \| _ { 2 } > 2 r$ for $i \neq j$ and

$$
\{ x _ { k } : k \in [ K ] \} \subseteq \bigcup _ { j = 1 } ^ { \widetilde { M } } B ( \widetilde { z } _ { j } , 2 r ) .
$$

Such a set exists because there are finitely many distinct task locations. Hence, by Assumption 4,

$$
1 = Q _ { K } \big ( \{ x _ { k } : k \in [ K ] \} \big ) \leq \sum _ { j = 1 } ^ { \widetilde M } Q _ { K } \big ( B ( \widetilde z _ { j } , 2 r ) \big ) \leq \widetilde M C ( 2 r ) ^ { d } ,
$$

and therefore $\begin{array} { r } { \widetilde { M } \geq \frac { 1 } { C 2 ^ { d } r ^ { d } } } \end{array}$

Next, since supp $\displaystyle ( \phi _ { 0 } ) = B ( 0 , 1 )$ , we have $\mathrm { s u p p } ( \phi _ { z , r } ) = B ( z , r )$ . Thus, because $\mathcal { \widetilde { Z } } _ { r }$ is a 2r-packing, the supports of $\{ \phi _ { \widetilde { z } _ { j } , r } : j \in [ \widetilde { M } ] \}$ are pairwise disjoint. Consequently, since $0 \leq \phi _ { z , r } \leq 1$

$$
\sum _ { j = 1 } ^ { \widetilde M } \sum _ { k = 1 } ^ { K } \phi _ { \widetilde { z } _ { j } , r } ( x _ { k } ) ^ { 2 } \leq K .
$$

Combining this with the lower bound on $\widetilde { M }$ gives

$$
\frac { 1 } { \widetilde { M } } \sum _ { j = 1 } ^ { \widetilde { M } } \sum _ { k = 1 } ^ { K } \phi _ { \widetilde { z } _ { j } , r } ( x _ { k } ) ^ { 2 } \le \frac { K } { \widetilde { M } } \le C 2 ^ { d } K r ^ { d } .
$$

Therefore, at least half $\widetilde { z } _ { j } \in \widetilde { \mathcal { Z } } _ { \ d \tau }$ satisfies

$$
\sum _ { k = 1 } ^ { K } \phi _ { \widetilde { z } _ { j } , r } ( x _ { k } ) ^ { 2 } \leq 2 ^ { d + 1 } C K r ^ { d } ;
$$

otherwise, the preceding average would exceed $C 2 ^ { d } K r ^ { d }$

Only retain these elements and relabel them as $\mathcal { Z } _ { r } = \{ z _ { 1 } , \dots , z _ { M } \}$ . Then separation still holds and

$$
M \geq \frac { \widetilde { M } } { 2 } \geq \frac { 1 } { C 2 ^ { d + 1 } r ^ { d } } ,
$$

by construction. Moreover, for every $j \in [ M ]$

$$
\sum _ { k = 1 } ^ { K } \phi _ { z _ { j } , r } ( x _ { k } ) ^ { 2 } \leq 2 ^ { d + 1 } C K r ^ { d } .
$$

Finally, the balls $\{ B ( z _ { j } , r / 2 ) : j \in [ M ] \}$ are pairwise disjoint. Hence, by Assumption 4,

$$
\sum _ { j = 1 } ^ { M } Q _ { K } \left( B ( z _ { j } , r / 2 ) \right) \geq M c ( r / 2 ) ^ { d } \geq \frac { c } { C 2 ^ { 2 d + 1 } } .
$$

Lemma 3 (H¨older Regularity) Fix $d \geq 1$ and $0 < \beta \leq 1$ . Let $x , z \in \mathbb { R } ^ { d } , r > 0$ . Define $\phi _ { 0 } ( x ) : = ( 1 - \| x \| _ { 2 } ^ { 2 } ) _ { + } ^ { \beta }$ and set $\begin{array} { r } { \phi _ { z , r } ( x ) : = \phi _ { 0 } \left( \frac { x - z } { r } \right) } \end{array}$ . Then, for every $x , y \in \mathbb { R } ^ { d }$

$$
| \phi _ { z , r } ( x ) - \phi _ { z , r } ( y ) | \leq \frac { 2 ^ { \beta } } { r ^ { \beta } } \| x - y \| _ { 2 } ^ { \beta } .
$$

Proof: Define $f ( x ) : = ( 1 - \| x \| _ { 2 } ^ { 2 } ) _ { + }$ such that $\phi _ { z , r } ( x ) = h \left( \frac { x - z } { r } \right) ^ { \beta }$ . Let $x , y \in \mathbb { R } ^ { d }$ . We show that $f$ is 2-Lipschitz. First, assume $\| x \| _ { 2 } , \| y \| _ { 2 } \leq 1$ . Then by the reverse triangle inequality

$$
| f ( x ) - f ( y ) | = { \big | } \| x \| _ { 2 } ^ { 2 } - \| y \| _ { 2 } ^ { 2 } { \big | } \leq 2 \| x - y \| _ { 2 } .
$$

Similarly, if $\| x \| _ { 2 } , \| y \| _ { 2 } > 1$ then $f ( x ) = f ( y ) = 0$ and the condition holds trivially. Assume now that, without loss of generality, $\| x \| _ { 2 } \leq 1 < \| y \| _ { 2 }$ . Then by the reverse triangle inequality again

$$
| f ( x ) - f ( y ) | = f ( x ) = ( 1 + \| x \| _ { 2 } ) ( 1 - \| x \| _ { 2 } ) \leq 2 ( 1 - \| x \| _ { 2 } ) \leq 2 ( \| y \| _ { 2 } - \| x \| _ { 2 } ) \leq 2 \| x - y \| _ { 2 } .
$$

Thus f is globally 2-Lipschitz.

Since $0 < \beta \leq 1 , | u ^ { \beta } - v ^ { \beta } | \leq | u - v | ^ { \beta }$ for all $u , v \geq 0$ . Consequently,

$$
| \phi _ { 0 } ( x ) - \phi _ { 0 } ( y ) | = | f ( x ) ^ { \beta } - f ( y ) ^ { \beta } | \leq 2 ^ { \beta } \| x - y \| _ { 2 } ^ { \beta } ,
$$

which shows that ϕ<sub>0</sub> is β-H¨older with constant $2 ^ { \beta }$ . Finally,

$$
| \phi _ { z , r } ( x ) - \phi _ { z , r } ( y ) | = \left| \phi _ { 0 } \left( \frac { x - z } { r } \right) - \phi _ { 0 } \left( \frac { y - z } { r } \right) \right| \leq 2 ^ { \beta } \left. \frac { x - z } { r } - \frac { y - z } { r } \right. _ { 2 } ^ { \beta } \leq \frac { 2 ^ { \beta } } { r ^ { \beta } } \| x - y \| _ { 2 } ^ { \beta } .
$$