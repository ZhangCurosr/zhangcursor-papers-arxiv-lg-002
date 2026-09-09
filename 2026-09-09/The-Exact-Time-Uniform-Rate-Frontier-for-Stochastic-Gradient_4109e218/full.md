# The Exact Time-Uniform Rate Frontier for Stochastic Gradient Descent on Smooth Convex Objectives

Ruijie Li <sup>∗</sup> Kang Chen <sup>†</sup> Tianyu Wang <sup>‡</sup>

September 9, 2026

## Abstract

We study the time-uniform convergence of the raw iterate of standard stochastic gradient descent (SGD) for unconstrained smooth convex objectives. We prove that, under standard noise assumptions, the time-uniform convergence rate gets arbitrarily close to $\sqrt { \log n / n }$ but never reaches it. More specifically, we prove that for every positive, eventually nondecreasing sequence h satisfying $h ( n ) =$ $o ( \sqrt { n } )$ , a bound of order $h ( n ) / \sqrt { n }$ , holding simultaneously for all n with probability at least $1 - \alpha$ and uniformly over the problem class, is achievable if and only if

$$
\sum _ { j = 1 } ^ { \infty } \frac { 1 } { h ( 2 ^ { j } ) ^ { 2 } } < \infty .
$$

The constructive suficiency result follows from a dyadic horizon-free schedule together with an additive conditional-restart inequality. The necessity counterpart applies to every deterministic nonnegative schedule and holds even for a one-dimensional analytic smooth convex objective with Gaussian noise.

## 1 Introduction

Stochastic gradient descent (SGD), dating back to at least the stochastic approximation algorithms Robbins and Monro [1], finds important applications in modern learning [See e.g. 2, for a review]. In its simplest form, SGD iterates as

$$
x _ { t + 1 } = x _ { t } - \eta _ { t } G ( x _ { t } , \xi _ { t } ) ,\tag{1}
$$

where $G ( x _ { t } , \xi _ { t } )$ is an estimate of $\nabla f ( x _ { t } )$ , and f is the objective function of interest.

Among many types of convergence guarantees, time-uniform guarantees are especially valuable for SGD, because they remain valid under data-dependent stopping. This property is crucial in practice, as stochastic algorithms are often terminated using dynamic, data-dependent rules. From a theoretical perspective, such termination criteria can be formalized as stopping times. Crucially, the stopping time is inherently random, reflecting the stochastic nature of the iterates. A fixed-time guarantee at a predetermined iteration does not directly justify such a stopping rule. To obtain this type of convergence guarantee, we seek to establish bounds of the form:

$$
\begin{array} { r } { \mathbb { P } \left( f ( x _ { n } ) - f ^ { * } \leq b _ { \alpha } ( n ) , \forall n \geq 1 \right) \geq 1 - \alpha , \forall \alpha \in ( 0 , 1 ) , } \end{array}\tag{2}
$$

where $b _ { \alpha } : \mathbb { N }  \mathbb { R } _ { + }$ is a boundary depending on $\alpha .$ . Unlike a collection of pointwise high-probability bounds, (2) is one event over the infinite trajectory and hence remains valid at every almost surely finite data-dependent stopping time. This viewpoint is closely related to confidence sequences in sequential inference [3, 4, 5] in statistical inference.

We study this question for standard SGD on unconstrained smooth convex objectives over real Hilbert spaces under conditionally unbiased norm-sub-Gaussian noise. We ask which profiles h permit (2) with $b _ { \alpha } ( n ) = \mathcal { O } ( h ( n ) / \sqrt { n } )$ under one deterministic horizon- and confidence-independent schedule, uniformly over the problem class and without dimension dependence.

## 1.1 Contributions

Our main result characterizes exactly every achievable overhead profile h.

Theorem 1.1 (Exact profile frontier, informal). Let $h : \mathbb { N } _ { + } \to ( 0 , \infty )$ be an eventually nondecreasing profile satisfying $h ( n ) = o ( { \sqrt { n } } )$ . There exists a deterministic, confidence-independent, horizon-free stepsize schedule for the SGD recursion (1) on smooth convex objectives whose iterates satisfy (2) with $b _ { \alpha } ( n ) =$ $\mathcal { O } ( h ( n ) / \sqrt { n } )$ , if and only if

$$
\sum _ { j = 1 } ^ { \infty } \frac { 1 } { h ( 2 ^ { j } ) ^ { 2 } } < \infty .\tag{3}
$$

In particular, a time-uniform bound of order $\mathcal { O } ( ( \log n ) ^ { ( 1 + \varepsilon ) / 2 } / \sqrt { n } )$ is achievable for every $\varepsilon > 0$ , whereas $\ O \left( { \sqrt { \log n / n } } \right)$ is not. In addition, the time-uniform bound gets arbitrarily close to the order of $\sqrt { \log n / n }$ through progressively slower iterated-logarithmic overheads, but never reaches it.

The formal version of our main result appears in Theorem 2.3. This criterion is exact at the level of arbitrary slowly varying profiles, not merely a prescribed logarithmic power. Corollary 2.4 establishes that no asymptotically smallest profile h exists. Also, the Cauchy condensation test shows that (3) is equivalent to $\textstyle \sum _ { n = 1 } ^ { \infty } ( n h ( n ) ^ { 2 } ) ^ { - 1 } < \infty$ . A machine-checked Lean 4 formalization of the targeted mathematical results can be found in https://github.com/Su-Zi-Zhan/Time-Uniform-SGD-Formalization.

## 1.2 Related works

For non-smooth stochastic optimization, standard polynomial stepsizes incur logarithmic losses in rawiterate convergence [6], and Harvey et al. [7] showed that such losses are unavoidable for the schedules they considered while establishing corresponding high-probability guarantees. Horizon-dependent schedules can recover the optimal rates [8], whereas modified methods such as FTRL-based momentum attain the optimal $\mathcal { O } ( T ^ { - 1 / 2 } )$ expected last-iterate rate without knowing T [9].

In deterministic non-smooth optimization, Zamani and Glineur [10] used a moving-comparator argument to derive exact worst-case last-iterate bounds and showed that no universal stepsize sequence attains the optimal guarantee at every horizon. Their moving-comparator argument terminalizes weighted onestep inequalities. Liu and Zhou [11] extended this mechanism to stochastic composite mirror descent, attaining the optimal fixed-horizon high-probability raw-iterate scale for smooth convex objectives under sub-Gaussian noise. We retain this terminalization mechanism but redesign the remaining-time weights for conditional restarts. Whereas their nonincreasing weights use a non-positive Bregman remainder to absorb directional martingale variation, ours separate the harmonic and confidence costs, replacing $\mathsf { H } _ { r } ( 1 + \log ( 1 / \delta ) )$ by ${ \sf H } _ { r } + \log ( 1 / \delta )$ . This additive separation enables simultaneous control over infinitely many epochs and terminal positions.

Recent results provide important time-uniform benchmarks. For smooth convex objectives, Feng et al. [12] proved an $\mathcal { O } ( ( 1 + \log ( 1 / \beta ) ) \log k / \sqrt { k } )$ anytime bound for a particular stochastic gradient descent with momentum (SGDM) scheme; later [13] sharpened this to $( \log { k } ) ^ { ( 1 + \varepsilon ) / 2 } / \sqrt { k } , \ \varepsilon \in ( \bar { 0 } , 1 / 2 )$ , for a variant of SGDM. Under strong convexity or related contractive structure, Chen et al. [14] and Pham et al. [15] established sharp O((log log $k + \log ( 1 / \beta ) ) / k ) \ – \mathrm { t y p e }$ uniform rates. Moreover, Aolaritei and Jordan [16] derived anytime-valid weighted-suboptimality bounds for projected SGD on general convex domains. These results concern momentum, contractive objectives, or weighted certificates, rather than the raw iterate of standard SGD on the general smooth convex class.

In the non-smooth Lipschitz setting over bounded convex domains, Kornowski and Shamir [17] proved that no fixed infinite stepsize schedule can achieve an anytime raw-iterate rate $o \big ( ( \log T ) ^ { 1 / 8 } / \sqrt { \overline { { T } } } )$ . To the best of our knowledge, no existing work has characterized the exact time-uniform overhead frontier for the raw last iterate of standard SGD over the general smooth convex class when one deterministic schedule must be independent of both the horizon and the confidence level.

## 1.3 Technical overview

For suficiency part, we use dyadic epochs $N _ { j } : = 2 ^ { j }$ with stepsize $\gamma _ { j } \asymp R / ( \sigma \sqrt { N _ { j } } h ( N _ { j } ) )$ where σ and R are constants. The stochastic energy of this epoch is $N _ { j } \gamma _ { j } ^ { 2 } \asymp R ^ { 2 } / ( \sigma ^ { \bar { 2 } } h ( N _ { j } ) ^ { 2 } )$ , so (3) is exactly the condition that the total squared-step energy be finite. The main analytical device is an additive conditional-restart inequality based on remaining-time weights. It controls every terminal position in an epoch while separating the harmonic and confidence terms, after which a summable allocation of failure probabilities yields one event for all n.

To prove the necessity part of Theorem 1.1, we consider a class of one-dimensional analytic quartic-flat convex objectives with Gaussian noise. Such objectives separate the two constraints on the schedule: sufficient cumulative steps are required for deterministic progress, while suficiently small blockwise squaredstep energy is needed to suppress stochastic excursions. Independent Gaussian block events convert these constraints into a deterministic Riccati-type recurrence. If (2) held while the series in (3) diverged, the recurrence would be forced to remain bounded and diverge simultaneously. Thus the two mechanisms identify the same reciprocal-square frontier.

## 2 Preliminaries

We use the standard separability convention for real Hilbert spaces. Every space in scope is finitedimensional or isomorphic to $\ell _ { 2 }$ . Accordingly, let $\mathfrak { H } = \{ \mathbb { R } ^ { d } : d \in \mathbb { N } _ { + } \} \cup \{ \ell _ { 2 } \}$ . The problem class is defined as follows.

Definition 2.1. Fix $L , R , \sigma > 0$ . Let $\mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ denote the class of problem instances $\mathcal { P }$ consisting of the following:

(i) The ambient space $\mathcal { H } \in \mathfrak { H }$ is selected before the run. $W e$ denote the corresponding inner product and norm $b y \ \langle \cdot , \cdot \rangle$ and ∥ · ∥, respectively.

(ii) The objective $f : \mathcal { H } \to \mathbb { R }$ is convex, Fr´echet diferentiable, and L-smooth, with nonempty minimizer set $X ^ { \ast }$

(iii) The initial point $x _ { 0 } \in \mathcal { H }$ satisfies dist $( x _ { 0 } , X ^ { * } ) \leq R$ . Fix $x ^ { * } \in X ^ { * }$ such that $\| x _ { 0 } - x ^ { * } \| \leq R$ , and write $\textstyle f ^ { * } : = \operatorname* { m i n } _ { x \in { \mathcal { H } } } f ( x ) = f ( x ^ { * } )$

(iv) Let $\{ \xi _ { t } \} _ { t \ge 0 }$ be fresh, independent random seeds, and let $\mathcal { F } _ { t } : = \sigma ( \xi _ { 0 } , \ldots , \xi _ { t - 1 } )$ be the pre-query filtration. For every $\mathcal { F } _ { t }$ -measurable query $x _ { t }$ , the oracle returns a stochastic gradient $G ( x _ { t } , \xi _ { t } )$ . We require this response to be $\mathcal { F } _ { t + 1 } { - } m e a s u r a b l e .$ , and define $Z _ { t } : = G ( x _ { t } , \xi _ { t } ) - \nabla f ( x _ { t } )$ . The stochastic gradient oracle is conditionally unbiased and norm-sub-Gaussian in the following almost-sure sense:

$$
\mathbb { E } [ Z _ { t } \mid \mathcal { F } _ { t } ] = 0 , \quad \mathbb { E } \left[ \left. \exp \left( \frac { \| Z _ { t } \| ^ { 2 } } { \sigma ^ { 2 } } \right) \right| \mathcal { F } _ { t } \right] \le \mathrm { e } , \quad \forall t \ge 0 .
$$

We refer to any $\mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ as an admissible problem instance.

For a deterministic infinite schedule $\eta = \{ \eta _ { t } \} _ { t \geq 0 }$ with $\eta _ { t } \geq 0$ , consider the standard SGD (1) and define

$$
\Delta _ { t } : = f ( x _ { t } ) - f ^ { * } .
$$

The schedule is called horizon-free because the full infinite sequence is determined before the run; it is not retuned for a terminal horizon. More precisely, the schedule is $\eta _ { t } = \eta _ { t } ( h , L , R , \sigma )$ . Our goal is to establish a time-uniform bound for standard SGD, which is given in the following definition.

Definition 2.2. Fix $L , R , \sigma > 0$ , a confidence level $\alpha \in ( 0 , 1 )$ , and a deterministic infinite stepsize schedule $\eta = \{ \eta _ { t } \} _ { t \geq 0 }$ with $\eta _ { t } \geq 0$ . Let $\mathcal { M } _ { \eta }$ denote the standard SGD method induced by η on the problem class $\mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ , and let P<sup>η</sup> denote the law of its trajectory on an instance $\mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ . A deterministic function $b _ { \alpha , \eta } : \mathbb { N } _ { + } \to \dot { \mathbb { R } } _ { + }$ is called a time-uniform convergence boundary $f o r \ M _ { \eta }$ over $\mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ at confidence level $1 - \alpha \ i f$ and only if

$$
\operatorname* { i n f } _ { \mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma ) } \mathbb { P } _ { \mathcal { P } } ^ { \eta } \left( \Delta _ { n } \leq b _ { \alpha , \eta } ( n ) , \forall n \geq 1 \right) \geq 1 - \alpha .
$$

This probability concerns one joint event over the entire infinite run, and $b _ { \alpha , \eta }$ must not depend on the realized sample path.

We now define the class of rate profiles considered in this paper:

$$
\mathcal { H } _ { 0 } : = \{ h : \mathbb { N } _ { + } \to ( 0 , \infty ) : h \mathrm { ~ i s ~ e v e n t u a l l y ~ n o n d e c r e a s i n g ~ a n d ~ } h ( n ) = o ( \sqrt { n } ) \} .
$$

Here, “eventually nondecreasing” means that there exists $n _ { 0 } \in  { \mathbb { N } } _ { + }$ such that $h ( n )$ is nondecreasing for all $n \geq n _ { 0 }$ . For any $h \in \mathcal { H } _ { 0 }$ , it acts as a multiplicative overhead on the polynomial rate $\mathcal { O } ( 1 / \sqrt { n } )$ , which is the optimal fixed-horizon rate for SGD on general smooth convex objectives, since we need $h ( n )$ to elevate a fixed-horizon guarantee to a strictly time-uniform guarantee.

With the problem class and operational constraints formally established, we state our main result. The following theorem reveals that the boundary between achievable and unachievable overhead profiles is exactly determined by a reciprocal-square series.

Theorem 2.3 (Exact reciprocal-square profile frontier, formal). For every fixed L, $R , \sigma > 0$ and $h \in \mathcal { H } _ { 0 }$ the following statements are equivalent.

(i) There exists a deterministic infinite schedule $\eta = \eta ( h , L , R , \sigma )$ , independent of the horizon, problem instance and confidence level, such that for every $\alpha \in ( 0 , 1 )$ , there exists $C _ { h , \alpha , L , R , \sigma } < \infty$ satisfying

$$
\operatorname* { i n f } _ { \mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma ) } \mathbb { P } _ { \mathcal { P } } ^ { \eta } \bigg ( \Delta _ { n } \leq C _ { h , \alpha , L , R , \sigma } \operatorname* { m i n } \bigg \{ 1 , \frac { h ( n ) } { \sqrt { n } } \bigg \} , \forall n \geq 1 \bigg ) \geq 1 - \alpha .\tag{ii}
$$

$$
\sum _ { j = 1 } ^ { \infty } \frac { 1 } { h ( 2 ^ { j } ) ^ { 2 } } < \infty .
$$

We call profiles satisfying these equivalent conditions achievable, and denote their collection by ${ \mathcal { H } } _ { \mathrm { a c h } }$

Sections 3 and 4 proves suficiency $( \mathrm { ( i i ) \implies ~ ( i ) ~ }$ , see Theorem 3.4) and necessity $( \mathrm { ( i ) } \implies \mathrm { ( i i ) }$ , see Theorem 4.1). Together, these two directions establish the exact characterization of ${ \mathcal { H } } _ { \mathrm { a c h } }$ . By Theorem 2.3, we have the following corollary, whose proof is deferred to Appendix A.

Corollary 2.4. The set of achievable profiles ${ \mathcal { H } } _ { \mathrm { a c h } }$ has no asymptotically smallest element, $i . e . , \ i f \ h \ \in$ ${ \mathcal { H } } _ { \mathrm { a c h } }$ , then there exists $g \in \mathcal { H } _ { \mathrm { a c h } }$ such that $g ( n ) = o ( h ( n ) )$ ).

$$
\ell _ { k } ( n ) = { \overbrace { \log \cdots \log } } ^ { k { \mathrm { ~ t i m e s } } } n
$$

To illustrate the corollary, consider the following example. Let for all suficiently large n. Each profile below is understood to be extended to $\mathbb { N } _ { + }$ by an arbitrary positive modification on a finite prefix. Then $\sqrt { \ell _ { 1 } ( n ) }$ is not achievable, but $( \ell _ { 1 } ( n ) ) ^ { 1 / 2 + \varepsilon }$ is achievable for any $\varepsilon > 0$ . Similarly, $( \ell _ { 1 } \cdot \cdot \cdot \ell _ { k } ( n ) ) ^ { 1 / 2 }$ is not achievable, but $( \ell _ { 1 } \cdot \cdot \cdot \ell _ { k - 1 } ( n ) \cdot \ell _ { k } ^ { 1 + \varepsilon } ( n ) ) ^ { 1 / 2 }$ is achievable for any $\varepsilon > 0$ . Let $h _ { k } ( n ) =$ $( \ell _ { 1 } \cdot \cdot \cdot \ell _ { k } ^ { 1 + \varepsilon } ( n ) ) ^ { 1 / 2 }$ . Then $h _ { k } \in \mathcal { H } _ { \mathrm { a c h } }$ and $h _ { k + 1 } ( n ) = o ( h _ { k } ( n ) )$ . Hence, there is no asymptotically smallest achievable profile in the family of iterated logarithms, which is consistent with Corollary 2.4. Also, this shows that the time-uniform convergence rate gets arbitrarily close to $\ O ( { \sqrt { \log n / n } } )$ but never reaches it.

## 3 Suficiency of the reciprocal-square profile

In this section, we prove the constructive suficiency, or upper-bound, direction of Theorem 2.3: every profile $h \in \mathcal { H } _ { 0 }$ satisfying $\begin{array} { r } { \sum _ { i > 1 } h ( 2 ^ { j } ) ^ { - 2 } < \infty } \end{array}$ belongs to ${ \mathcal { H } } _ { \mathrm { a c h } }$ . For arbitrary $L , R , \sigma > 0$ , we construct a deterministic infinite schedule $\eta ( h , L , R , \sigma )$ , independent of the horizon, the problem instance, and the confidence level, whose raw SGD iterates satisfy the $\mathcal { O } ( h ( n ) / \sqrt { n } )$ bound simultaneously for all n on a single high-probability event.

## 3.1 The horizon-free dyadic schedule

For brevity and clarity, we employ a dyadic epoch-based schedule that maintains a constant stepsize within each epoch to leverage martingale concentration while systematically decreasing the stepsize across epochs to drive optimization error to zero. Let $N _ { j } = 2 ^ { j }$ and $h _ { j } = h ( N _ { j } )$ . Algorithm 1 completely specifies our schedule.

Algorithm 1 is well defined. Indeed, $\textstyle \sum _ { j \geq J _ { h } } h _ { j } ^ { - 2 } < \infty$ implies $h _ { j } \to \infty$ , and hence $\sqrt { N _ { j } } h _ { j } \to \infty$ . Thus the set defining $j _ { 0 }$ is nonempty and $j _ { 0 } < \infty$ . Moreover, since $h _ { j }$ is nondecreasing for $j \geq J _ { h }$ , the active stepsizes $\gamma _ { j } = R / ( \sigma c _ { h } \sqrt { N _ { j } } h _ { j } )$ are nonincreasing. Crucially, the total squared-step energy of this infinite sequence is bounded above by $R ^ { 2 } / \sigma ^ { 2 }$ , as guaranteed by the convergence of the reciprocal-square series.

Remark 3.1. A constant stepsize is adopted within each epoch primarily for simplicity of the suficiency proof. Alternative stepsize schedules can also be employed.

## 3.2 Core tool: additive conditional restart

The suficiency proof repeatedly applies finite-block last-iterate estimates along the dyadic schedule. Since a block may start from an iterate determined by the preceding trajectory, and since the comparator may be $\mathcal { F } _ { m }$ -measurable, we need an estimate that remains valid conditionally on $\mathcal { F } _ { m }$ . We refer to such an application as an analytical restart; the SGD trajectory itself is not reset.

A direct specialization of the high-probability last-iterate bound of Liu and Zhou [11] yields a stochastic term of the form $\begin{array} { r } { \sigma ^ { 2 } \gamma \mathsf { H } _ { r } ( 1 + \log \frac { 1 } { \delta } ) } \end{array}$ , where $\begin{array} { r } { \mathsf { H } _ { r } = \sum _ { i = 1 } ^ { r } i ^ { - 1 } } \end{array}$ is the r-th harmonic number. Under repeated confidence allocation, this product incurs an additional logarithmic loss. Using the same barycentric lastiterate reduction, but with the remaining weights $w _ { t } = r - t + 1$ , we instead obtain the additive dependence $\sigma ^ { 2 } \eta \big ( \mathsf { H } _ { r } + \log ( 1 / \delta ) \big )$ . The following theorem formalizes this conditional last-iterate estimate.

Algorithm 1: Horizon-Free Dyadic Schedule for an Admissible Profile   
Input: A profile $h \in \mathcal { H } _ { 0 }$ satisfying $\begin{array} { r } { \sum _ { j \geq 1 } h ( 2 ^ { j } ) ^ { - 2 } < \infty } \end{array}$ , and parameters $L , R , \sigma > 0 .$   
Output: One deterministic infinite schedule $\eta$ depending only on $h , L , R , \sigma$   
1 $J _ { h } : = \operatorname* { i n f } \{ j \in \mathbb { N } _ { + } : h ( n + 1 ) \geq h ( n )$ for all $n \in \mathbb { N } _ { + }$ with $n \geq N _ { j } \}$   
2 $c _ { h } : = \textstyle \operatorname* { m a x } \{ 1 , ( \sum _ { j \ge J _ { h } } h _ { j } ^ { - 2 } ) ^ { 1 / 2 } \}$   
3 $j _ { 0 } : = \operatorname* { i n f } \{ j \ge J _ { h } : h _ { j } \ge 1 , R / ( \sigma c _ { h } \sqrt { N _ { j } } h _ { j } ) \le 1 / ( 2 L ) \}$   
4 for $t = 0 , 1 , 2 ,$ . . . do   
5 if $t < N _ { j _ { 0 } }$ then   
6 $\eta _ { t } : = 0 .$   
7 end   
8 if $N _ { j } \leq t < N _ { j + 1 }$ for some $j \geq j _ { 0 }$ then   
9 $\begin{array} { r } { \eta _ { t } = \gamma _ { j } : = \frac { R } { \sigma c _ { h } \sqrt { N _ { j } } h _ { j } } } \end{array}$   
10 end   
11 end   
12 return $\eta = \{ \eta _ { t } \} _ { t \geq 0 } .$

Theorem 3.2. Fix $L , R , \sigma > 0 _ { ; }$ , an instance $\mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ , and a deterministic nonnegative stepsize schedule η. Let $m \in \mathbb { N }$ and $r ~ \in ~ \mathbb { N } _ { + }$ be deterministic, and suppose that $\eta _ { m } ~ = ~ \cdot ~ \cdot ~ = ~ \eta _ { m + r - 1 } ~ = ~ \gamma$ $0 < \gamma \leq 1 / ( 2 L )$ . Then for every almost surely finite $\mathcal { F } _ { m }$ -measurable comparator y and every $\delta \in ( 0 , 1 )$

$$
\mathbb { P } \bigg ( f ( x _ { m + r } ) - f ( y ) \leq \frac { 4 D ( y , x _ { m } ) } { \gamma r } + 6 \sigma ^ { 2 } \gamma \mathsf { H } _ { r } + 9 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } \bigg | \mathcal { F } _ { m } \bigg ) \geq 1 - \delta \quad a l m o s t s u r e l y ,
$$

where $D : \mathcal { H } \times \mathcal { H }  [ 0 , + \infty )$ is given by $D ( a , b ) = \| a - b \| ^ { 2 } / 2 .$

See Appendix B.1 for a complete proof. Theorem 3.2 supplies the conditional finite-block estimate used in each restart. The time-uniform conclusion will follow by combining it with the global radius control (detailed in Appendix B.2) and the dyadic schedule.

## 3.3 Proof of suficiency

Lemma 3.3. $I f \ \{ h _ { j } \} _ { j \in \mathbb { N } }$ is positive, eventually nondecreasing, and $\textstyle \sum _ { j = 1 } ^ { \infty } h _ { j } ^ { - 2 } < \infty$ , then $j / h _ { j } ^ { 2 } \to 0$ as $j \to \infty$

Proof. For every suficiently large $\begin{array} { r } { j , \sum _ { k = \lfloor j / 2 \rfloor } ^ { j } h _ { k } ^ { - 2 } \geq j h _ { j } ^ { - 2 } / 2 } \end{array}$ . The left-hand side is a tail of a convergent series and tends to zero. □

Theorem 3.4 (Suficiency part of Theorem 2.3). Let $h \in \mathcal { H } _ { 0 }$ satisfy $\textstyle \sum _ { i = 1 } ^ { \infty } h ( 2 ^ { j } ) ^ { - 2 } < \infty$ . Then, for every $L , R , \sigma \ > \ 0$ , there exists a deterministic nonnegative infinite stepsize schedule $\eta = \eta ( h , L , R , \sigma )$ independent of the terminal horizon, the problem instance, and the confidence level, such that, for every $\alpha \in ( 0 , 1 )$ , there exists a constant $C _ { h , \alpha , L , R , \sigma } < \infty$ satisfying

$$
\operatorname* { i n f } _ { \mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma ) } \mathbb { P } _ { \mathcal { P } } ^ { \eta } \bigg ( \Delta _ { n } \leq C _ { h , \alpha , L , R , \sigma } \cdot \operatorname* { m i n } \left\{ 1 , \frac { h ( n ) } { \sqrt { n } } \right\} , \forall n \geq 1 \bigg ) \geq 1 - \alpha .
$$

The schedule is the one produced by Algorithm 1; it depends on $h , L , R , \sigma$ alone, and in particular not on $\alpha .$ , the horizon, or the instance. The proof has three stages. First, the finite total energy of the schedule confines the entire infinite trajectory to one fixed ball. Epoch j contributes squared-step energy $N _ { j } \gamma _ { j } ^ { 2 } = \mathcal { O } ( h _ { j } ^ { - 2 } )$ , so reciprocal-square summability makes the total energy finite. This yields a single highprobability radius bound for the entire infinite trajectory and makes every active stepsize admissible for Theorem 3.2 (Lemma B.10 and Lemma B.8).

Second, inside each dyadic epoch, two applications of Theorem 3.2 with diferent comparators control every terminal position. For late terminal positions, enough steps have elapsed that $x ^ { * }$ can be used directly as the comparator in Theorem 3.2. But for early terminal positions, since the bound scales like $1 / r$ with the number of steps r, the penalty is too large to reach the target bound. Instead, we use the starting point $x _ { N _ { j } }$ as the comparator, which eliminates the penalty term. The missing bound on $\Delta _ { N _ { j } }$ is exactly the late-half bound inherited from epoch $j - 1$ . Because adjacent dyadic scales difer only by a constant factor, this handof propagates one constant across all epochs (Lemmas B.12 and B.13).

Finally, uniform control of all $2 ^ { j }$ terminal positions in epoch j requires a summable confidence allocation, for which both the harmonic term and log $( 1 / \delta )$ are $\mathcal { O } ( j )$ . The additive dependence in Theorem 3.2 therefore makes the stochastic cost, relative to the target $h _ { j } / \sqrt { N _ { j } }$ , of order $j / \bar { h _ { j } ^ { 2 } }$ , which vanishes by Lemma 3.3. Here the additive structure is crucial: a multiplicative dependence would instead produce $j ^ { 2 } / h _ { j } ^ { 2 }$ and lose the frontier.

Proof of Theorem $\ 3 . 4 \cdot$ Fix $\alpha \in ( 0 , 1 )$ and an arbitrary $\mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ . By Lemma B.10, the schedule of Algorithm 1 satisfies $\gamma _ { j } \leq 1 / ( 2 L )$ for every $j \geq j _ { 0 }$ and $Q _ { \infty } \leq R ^ { 2 } / \sigma ^ { 2 }$ . Hence Lemma B.8 applies at level $\alpha / 2 ,$ and the global radius event G of (9) satisfies $\mathbb { P } ( \mathcal { G } ) \ge 1 - \alpha / 2$ . With the allocation $\delta _ { j , r } = \stackrel { \textstyle \mathbf { \hat { \alpha } } } { \alpha } / 2 ^ { 2 ( j + 1 ) }$ of (8), the restart events $\boldsymbol { A } _ { j , \boldsymbol { r } }$ satisfy $\mathbb { P } ( \mathcal { A } _ { j , r } ) \geq 1 - \delta _ { j , r } ;$ , and

$$
\sum _ { j = j _ { 0 } } ^ { \infty } \sum _ { r = 1 } ^ { N _ { j } } \delta _ { j , r } = \sum _ { j = j _ { 0 } } ^ { \infty } N _ { j } \frac { \alpha } { 2 ^ { 2 ( j + 1 ) } } \le \frac { \alpha } { 2 } .
$$

A union bound therefore gives $\mathbb { P } ( \mathcal { E } ) \geq 1 - \alpha$ for

$$
\mathcal { E } : = \mathcal { G } \cap \bigcap _ { j \geq j _ { 0 } } \bigcap _ { 1 \leq r \leq N _ { j } } \mathcal { A } _ { j , r } .
$$

On $\mathcal { E } ,$ Lemma B.14 controls every $1 \leq t < 3 N _ { j _ { 0 } } / 2$ with the constant $C _ { \mathrm { p r e f i x } }$ , and Lemma B.13 controls every $t \geq 3 N _ { j _ { 0 } } / 2$ with the constant $C _ { \mathrm { l a t t e r } }$ . Finally, L-smoothness at the minimizer and (9) give $\Delta _ { t } \ \leq$ $\begin{array} { r } { L B _ { \alpha / 2 } \leq \frac { 8 1 } { 4 } L \bar { R } ^ { 2 } \log ( 4 / \alpha ) } \end{array}$ on ${ \mathcal { G } } _ { : }$ , so the boundary may also be capped at one. Setting

$$
C _ { h , \alpha , L , R , \sigma } : = \operatorname* { m a x } \left\{ C _ { \mathrm { p r e f i x } } , C _ { \mathrm { l a t t e r } } , { \frac { 8 1 } { 4 } } L R ^ { 2 } \log { \frac { 4 } { \alpha } } \right\}
$$

yields $\Delta _ { t } \leq C _ { h , \alpha , L , R , \sigma }$ min $\{ 1 , h ( t ) / \sqrt { t } \}$ simultaneously for all $t \geq 1$ on E. Since $\mathcal { P }$ was arbitrary and η does not depend on it, the infimum over $\mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ obeys the same bound. □

## 4 Necessity of reciprocal-square summability

We prove that reciprocal-square summability is necessary for a deterministic schedule to attain a timeuniform $h ( n ) / { \sqrt { n } }$ boundary. It is enough to work with the uncapped envelope $\varepsilon _ { \alpha } ( n ) \leq B \cdot h ( n ) / { \sqrt { n } }$ , where B is a constant. This is because the above envelope is weaker than the capped boundary in Theorem 2.3.

Theorem 4.1 (Necessity part of Theorem 2.3). Fix L $, R , \sigma > 0$ and $\alpha \in ( 0 , 1 )$ . Let $h \in \mathcal { H } _ { 0 }$ and let $\eta = \{ \eta _ { t } \} _ { t \geq 0 }$ be any deterministic nonnegative infinite stepsize schedule, fixed before the run and independent of the terminal horizon and the problem instance. The schedule may depend on h, L, R, σ, and α. Suppose that $\mathcal { M } _ { \eta }$ admits a time-uniform convergence boundary $b _ { \alpha , \eta }$ over $\mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ at confidence level $1 - \alpha$ . If there exist constants $B > 0$ and $n _ { 0 } \in  { \mathbb { N } } _ { + }$ such that

$$
b _ { \alpha , \eta } ( n ) \leq B \cdot \frac { h ( n ) } { \sqrt { n } } , \quad \forall n \geq n _ { 0 } ,
$$

then $\textstyle \sum _ { j = 1 } ^ { \infty } h ( 2 ^ { j } ) ^ { - 2 } < \infty$

Remark 4.2. Allowing η to depend on α only strengthens the necessity statement: the conclusion therefore applies directly to the confidence-independent schedules required in the definition of achievability.

Throughout the proof, the schedule, the profile, and all constants in Theorem 4.1 are fixed. We write $\begin{array} { r } { S _ { n } : = \sum _ { t < n } \bar { \eta _ { t } } , Q _ { n } : = \sum _ { t < n } \eta _ { t } ^ { 2 } } \end{array}$ , and $\varepsilon _ { \alpha } ( n ) : = b _ { \alpha , \eta } ( n )$ . All lemmas and corollaries in the remainder of this section are understood to hold under the conditions of Theorem 4.1.

## 4.1 Fixed-horizon analysis

Since time-uniform boundaries imply fixed-horizon boundaries:

$$
\begin{array} { r } { \mathbb { P } \big ( \Delta _ { n } \leq \varepsilon _ { \alpha } ( n ) \big ) \geq \mathbb { P } \big ( \Delta _ { t } \leq \varepsilon _ { \alpha } ( t ) , \forall t \geq 1 \big ) \geq 1 - \alpha , } \end{array}
$$

we can apply the fixed-horizon analysis to obtain some necessary results, which are summarized in the following two lemmas. The first comes from the amount of deterministic progress required to reduce the initial error.

Lemma 4.3. For any $n \in \mathbb { N } _ { + }$

$$
\varepsilon _ { \alpha } ( n ) \geq { \frac { R ^ { 2 } } { 1 6 } } \operatorname* { m i n } \left\{ L , { \frac { 1 } { S _ { n } } } \right\} ,
$$

where $1 / S _ { n } : = + \infty$ when $S _ { n } = 0$

Lemma 4.3 shows that a suficiently large total stepsize $S _ { n }$ is necessary to reduce the initial error. The second restriction comes from the stochastic energy injected by the oracle noise.

Lemma 4.4. Suppose $S _ { n } \geq 1 / L$ . Let $z _ { \alpha } : = \Phi ^ { - 1 } ( 1 - \alpha / 2 )$ , where Φ is the cumulative distribution function of the standard normal distribution. Then

$$
\varepsilon _ { \alpha } ( n ) \geq { \frac { z _ { \alpha } ^ { 2 } \sigma ^ { 2 } } { 6 4 } } \cdot { \frac { Q _ { n } } { S _ { n } } } .
$$

Lemma 4.4 establishes a trade-of constraint: the error bound is governed by the ratio $Q _ { n } / S _ { n }$ . Consequently, accommodating a large $Q _ { n }$ requires a correspondingly large $S _ { n }$ in order to achieve a suficiently small error bound. The proofs of Lemmas 4.3 and 4.4 are deferred to Appendix C.1.

Now let $N _ { j } = 2 ^ { j } , h _ { j } = h ( 2 ^ { j } )$ , and $\mathcal { Q } _ { j } = Q _ { N _ { j } }$ . We define $p _ { j } : = S _ { N _ { j } } / \sqrt { N _ { j } }$ . Then we have the following corollary, whose proof can also be seen in Appendix C.1:

Corollary 4.5. For suficiently large $j \in \mathbb { N } _ { + }$ , we have

$$
p _ { j } \geq \frac { z _ { \alpha } ^ { 2 } \sigma ^ { 2 } } { 6 4 B } \cdot \frac { \mathcal { Q } _ { j } } { h _ { j } } .
$$

These lemmas provide necessary results, but they cannot place infinitely many epoch constraints on one common event. The next subsection uses one fixed instance throughout the infinite run; this is where the joint time-uniform guarantee is essential.

## 4.2 A fixed flat-active witness and the dyadic epoch ceiling

$$
\phi ( y ) : = \frac { L } { 4 } \left( \sqrt { R ^ { 2 } + y ^ { 2 } } - R \right) ^ { 2 }
$$

It is straightforward to verify that $\phi$ is convex, L-smooth, and non-PL. We can further verify that $\phi$ is quartic-flat at the minimizer by Taylor expansion:

$$
\phi ( y ) = \frac { L } { 1 6 R ^ { 2 } } y ^ { 4 } + \mathcal { O } ( y ^ { 6 } ) , \quad \mathrm { a s ~ } y \to 0 .
$$

Moreover,

$$
\phi ^ { \prime } ( y ) = \frac { L y ^ { 3 } } { 2 \sqrt { R ^ { 2 } + y ^ { 2 } } ( \sqrt { R ^ { 2 } + y ^ { 2 } } + R ) } , \quad | \phi ^ { \prime } ( y ) | \leq \frac { L | y | ^ { 3 } } { 4 R ^ { 2 } } .
$$

Let $\vartheta : = ( 1 - \mathrm { e } ^ { - 2 } ) / 2 , \nu ^ { 2 } : = \vartheta \sigma ^ { 2 }$ . The recursion is initialized at the minimizer and uses independent $\zeta _ { t } \sim \mathcal { N } ( 0 , \nu ^ { 2 } )$ as the oracle noise:

$$
Y _ { t + 1 } = Y _ { t } - \eta _ { t } ( \phi ^ { \prime } ( Y _ { t } ) + \zeta _ { t } ) , \quad Y _ { 0 } = 0 .
$$

The Gaussian oracle satisfies the noise assumption with equality: $\mathbb { E } [ \exp ( \zeta _ { t } ^ { 2 } / \sigma ^ { 2 } ) ] = ( 1 - 2 \nu ^ { 2 } / \sigma ^ { 2 } ) ^ { - 1 / 2 } = \mathrm { e }$ Therefore, it is a valid problem instance defined in Definition 2.1. By the assumption of Theorem 4.1, the all-time success event $\mathcal { C } : = \{ \phi ( Y _ { n } ) \leq \varepsilon _ { \alpha } ( n ) , \forall n \geq 1 \}$ satisfies $\mathbb { P } ( \mathcal { C } ) \geq 1 - \alpha$ , where $\varepsilon _ { \alpha } ( n ) \leq B h ( n ) / \sqrt { n }$ for all suficiently large n.

Validity on this instance first implies that $\eta _ { t } \to 0$ as $t \to \infty$ (see Lemma C.5 in Appendix C.2).

Now we introduce a few quantities that summarize the behavior of the stepsize schedule and the boundary on each dyadic epoch. First, over the interval $[ N _ { j } , N _ { j + 1 } ]$ , we consider the worst-case rescaling of the overhead function and set

$$
\hat { h } _ { j } : = \operatorname* { m a x } _ { \tiny N _ { j } \leq n \leq N _ { j + 1 } } h ( n ) \sqrt { \frac { N _ { j } } { n } } .
$$

This quantity captures the largest efective value of $h ( n ) / { \sqrt { n } }$ within the j-th epoch, expressed at the left endpoint scale $N _ { j }$ . Moreover, there exists a threshold $j _ { h }$ such that, for every $j > i \ge j _ { h }$ , eventually monotonicity gives $h _ { j } \leq \hat { h } _ { j } \leq h _ { j + 1 }$ and $\hat { h } _ { i } \le h _ { i + 1 } \le h _ { j }$ . Now we express the corresponding boundary level

at epoch $j$ as $\bar { \varepsilon } _ { j } : = B \hat { h } _ { j } / \sqrt { N _ { j } }$ . On the event C, all iterates in the j-th epoch satisfy $\phi ( Y _ { n } ) \leq \bar { \varepsilon } _ { j }$ . This motivates defining

$$
\beta _ { j } ^ { 2 } : = \operatorname* { s u p } \{ | y | ^ { 2 } : \phi ( y ) \leq \bar { \varepsilon } _ { j } \} = r ^ { 2 } ( \bar { \varepsilon } _ { j } ) , \quad r ^ { 2 } ( u ) : = 4 R \sqrt { \frac { u } { L } } + \frac { 4 u } { L } .
$$

The following lemma establishes the core time-uniform property of our lower bound.

Lemma 4.6. Let $\gamma _ { 0 } = ( \log 2 ) / 4$ and $C _ { b } = 8 0 0 / ( \vartheta \gamma _ { 0 } )$ . Then for every suficiently large $j ,$ we have

$$
H _ { j } : = \sum _ { t = N _ { j } } ^ { N _ { j + 1 } - 1 } \eta _ { t } \le C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { j } .
$$

The proof is based on a block anti-concentration argument. On the common success event, all iterates in epoch j remain in a shrinking neighborhood of the minimizer, where the quartic-flat objective produces only a small accumulated drift. If the epoch stepsize mass $H _ { j }$ were too large, the epoch could be partitioned into many disjoint blocks, each of which would force an independent Gaussian increment to lie in a prescribed small interval. Gaussian anti-concentration then shows that all these constraints cannot hold with probability at least $1 - \alpha$ . Balancing the number of available blocks against the corresponding Gaussian tail scale yields the factor $1 / j$ . The proof is deferred to Appendices C.2 and C.3.

## 4.3 Reciprocal-square summability

We now combine the fixed-horizon analysis with the dyadic epoch ceiling, yielding the following two lemmas. In Corollary 4.5, we established a lower bound for $p _ { j }$ . The following lemma provides an upper bound for $p _ { j }$ , and together they yield an upper bound for $\mathcal { Q } _ { j }$

Lemma 4.7. There exist constants $C _ { p }$ and $C _ { q }$ such that for suficiently large $j ,$ we have

$$
p _ { j } \le C _ { p } \frac { B } { \sigma ^ { 2 } } \frac { h _ { j } } { j } , \quad \mathcal { Q } _ { j } \le C _ { q } \frac { B ^ { 2 } } { \sigma ^ { 4 } } \frac { h _ { j } ^ { 2 } } { j } .
$$

The following nonlinear lower recurrence for $\mathcal { Q } _ { j }$ complements the upper bounds above. This recurrence is the key mechanism that connects the dyadic energy growth to the reciprocal-square series.

Lemma 4.8. Let m be suficiently large. Then for all $J > m$

$$
\mathcal { Q } _ { J } \geq \left( 2 - \sqrt { 2 } \right) \mathcal { Q } _ { m } + ( \sqrt { 2 } - 1 ) ^ { 2 } \frac { z _ { \alpha } ^ { 4 } \sigma ^ { 4 } } { 2 ^ { 1 2 } B ^ { 2 } } \sum _ { j = m + 1 } ^ { J - 1 } \frac { \mathcal { Q } _ { j } ^ { 2 } } { h _ { j } ^ { 2 } } .
$$

The proofs of Lemmas 4.7 and 4.8 are deferred to Appendix C.4. Now we are ready to prove Theorem 4.1.

Proof of Theorem $4 . 1$ . We use contradiction. Suppose $\textstyle \sum _ { j = 1 } ^ { \infty } h _ { j } ^ { - 2 } = \infty$

We choose m suficiently large that all preceding eventual bounds hold. Let $T _ { m + 1 } : = ( 2 - \sqrt { 2 } ) \mathcal { Q } _ { m }$ . By Lemma 4.3, $S _ { n } \to + \infty$ . Hence, for suficiently large m, $\mathcal { Q } _ { m } > 0$ , and therefore $T _ { m + 1 } > 0$ . By Lemma 4.8, $T _ { m + 1 } \leq \mathcal { Q } _ { m + 1 }$ . We recursively define

$$
T _ { J + 1 } : = T _ { J } + \kappa \frac { T _ { J } ^ { 2 } } { h _ { J } ^ { 2 } } , \quad \forall J > m , \quad 0 < \kappa \leq ( \sqrt { 2 } - 1 ) ^ { 2 } \frac { z _ { \alpha } ^ { 4 } \sigma ^ { 4 } } { 2 ^ { 1 2 } B ^ { 2 } } .
$$

Then by Lemma 4.8 and induction, we have $\mathcal { Q } _ { J } \geq T _ { J }$ for all $J > m$ . By Lemma 4.7, we have $\kappa T _ { J } / h _ { J } ^ { 2 } \leq$ $\kappa C _ { q } B ^ { 2 } \sigma ^ { - 4 } / J \leq 1$ for every suficiently large J. Therefore, we have

$$
\frac { 1 } { T _ { J } } - \frac { 1 } { T _ { J + 1 } } = \frac { T _ { J + 1 } - T _ { J } } { T _ { J } T _ { J + 1 } } = \frac { \kappa T _ { J } ^ { 2 } / h _ { J } ^ { 2 } } { T _ { J } ( T _ { J } + \kappa T _ { J } ^ { 2 } / h _ { J } ^ { 2 } ) } = \frac { \kappa / h _ { J } ^ { 2 } } { 1 + \kappa T _ { J } / h _ { J } ^ { 2 } } \geq \frac { \kappa } { 2 h _ { J } ^ { 2 } } .
$$

Fix an index $J _ { 1 }$ from which this inequality holds. Summing from $J _ { 1 }$ to $\infty$ gives

$$
\frac { 1 } { T _ { J _ { 1 } } } \geq \sum _ { j = J _ { 1 } } ^ { \infty } \frac { 1 } { T _ { j } } - \frac { 1 } { T _ { j + 1 } } \geq \sum _ { j = J _ { 1 } } ^ { \infty } \frac { \kappa } { 2 h _ { j } ^ { 2 } } = \infty ,
$$

which is a contradiction. Therefore, we must have $\textstyle \sum _ { j = 1 } ^ { \infty } h _ { j } ^ { - 2 } <$ ∞.

## 5 Conclusion

We have characterized the exact time-uniform convergence frontier of standard SGD on general smooth convex objectives: achievable overhead profiles are exactly those satisfying reciprocal-square summability $\textstyle \sum _ { i = 1 } ^ { \infty } h ( 2 ^ { j } ) ^ { - 2 } < \infty$ The resulting picture is governed by reciprocal-square summability: rather than admitting a single optimal overhead, the achievable class has no asymptotically smallest element and can approach the plog $n / n$ arbitrarily closely without attaining it. Our necessity result shows that this obstruction already arises for a one-dimensional analytic smooth convex objective with Gaussian noise, even when the schedule may depend on the confidence level. Thus, the frontier is intrinsic to time-uniform rawiterate control in the general noncontractive setting, rather than an artifact of nonsmoothness, projection, high dimension, or confidence-independent tuning. A natural next question is whether the same frontier persists for adaptive or randomized stepsize rules.

## AI use statement

In this work, we used GPT-5.6 Sol to assist in developing and refining proof strategies for the necessity direction, including suggesting intermediate arguments and technical derivations. The mathematical claims were formulated by the authors, and all AI-assisted proof arguments were independently checked and manually verified by the authors. We did not use generative AI tools for generating the proof in the suficiency direction. Specifically, GPT-6-Astra was used to assist in translating the mathematical development into Lean 4 and constructing the supplementary formalization. The resulting formalization was checked by the Lean kernel and independently audited for theorem correspondence, dependencies, and unintended axioms. GPT-5.6 Sol and Gemini 3.1 Pro was used to polish the writing of the introduction and related work sections. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## Reproducibility statement

Complete proofs of all mathematical claims are provided in the appendices. We additionally provide an anonymized Lean 4 formalization of the targeted mathematical results as supplementary material, together with theorem-correspondence and proof-integrity audits, as well as instructions for rebuilding the formalization.

## References

[1] Herbert Robbins and Sutton Monro. A stochastic approximation method. The Annals of Mathematical Statistics, 22(3):400–407, 1951. doi: 10.1214/aoms/1177729586.

[2] L´eon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM review, 60(2):223–311, 2018.

[3] Donald A Darling and Herbert Robbins. Confidence sequences for mean, variance, and median. Proceedings of the National Academy of Sciences, 58(1):66–68, 1967.

[4] Tze Leung Lai. On confidence sequences. The Annals of Statistics, pages 265–280, 1976.

[5] Steven R. Howard, Aaditya Ramdas, Jon McAulife, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080, 2021. doi: 10.1214/20-AOS1991.

[6] Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 71–79. PMLR, 2013. URL https://proceedings.mlr.press/v28/shamir13.html.

[7] Nicholas J. A. Harvey, Christopher Liaw, Yaniv Plan, and Sikander Randhawa. Tight analyses for non-smooth stochastic gradient descent. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 1579–1613. PMLR, 2019. URL https://proceedings.mlr.press/v99/harvey19a.html.

[8] Prateek Jain, Dheeraj Nagaraj, and Praneeth Netrapalli. Making the last iterate of SGD information theoretically optimal. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pages 1752–1755. PMLR, 2019. URL https://proceedings.mlr.press/v99/jain19a.html.

[9] Xiaoyu Li, Mingrui Liu, and Francesco Orabona. On the last iterate convergence of momentum methods. In Proceedings ofthe 33rd International Conference on Algorithmic Learning Theory, volume 167 of Proceedings of Machine Learning Research, pages 699–717. PMLR, 2022. URL https:// proceedings.mlr.press/v167/li22a.html.

[10] Moslem Zamani and Fran¸cois Glineur. Exact convergence rate of the last iterate in subgradient methods. SIAM Journal on Optimization, 35(3):2182–2201, 2025. doi: 10.1137/24M1717762. URL https://doi.org/10.1137/24M1717762.

[11] Zijian Liu and Zhengyuan Zhou. Revisiting the last-iterate convergence of stochastic gradient methods. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=xxaEhwC1I4.

[12] Yasong Feng, Yifan Jiang, Tianyu Wang, and Zhiliang Ying. The anytime convergence of stochastic gradient descent with momentum: From a continuous-time perspective, 2023. URL https://arxiv. org/abs/2310.19598.

[13] Yasong Feng, Yifan Jiang, Tianyu Wang, and Zhiliang Ying. Breaking a logarithmic barrier in the stopping time convergence rate of stochastic first-order methods, 2025. URL https://arxiv.org/ abs/2506.23335.

[14] Kang Chen, Yasong Feng, and Tianyu Wang. Revisiting stochastic gradient descent for strongly convex objectives: Tight uniform-in-time bounds. Systems & Control Letters, 212:106419, 2026. doi: 10.1016/j.sysconle.2026.106419. URL https://doi.org/10.1016/j.sysconle.2026.106419.

[15] Tuan Pham, Alessandro Rinaldo, and Purnamrita Sarkar. Time-uniform concentration bounds for iterative algorithms. arXiv preprint arXiv:2511.18273, 2025. URL https://arxiv.org/abs/2511. 18273.

[16] Liviu Aolaritei and Michael I. Jordan. Stopping rules for stochastic gradient descent via anytime-valid confidence sequences, 2025. URL https://arxiv.org/abs/2512.13123.

[17] Guy Kornowski and Ohad Shamir. Gradient descent’s last iterate is often (slightly) suboptimal, 2026. URL https://arxiv.org/abs/2604.13870.

## A Deferred proofs from Section 2

Proof of Corollary 2.4. Let $h \in \mathcal { H } _ { \mathrm { a c h } }$ . By Theorem 2.3, we have

$$
\sum _ { j = 1 } ^ { \infty } \frac { 1 } { h ( 2 ^ { j } ) ^ { 2 } } < \infty .
$$

Let $h _ { j } = h ( 2 ^ { j } )$ and $a _ { j } = h _ { j } ^ { - 2 }$ . Then $\textstyle \sum _ { j \geq 1 } a _ { j } < \infty$ . In particular, $a _ { j }  0$ and $h _ { j }  \infty$ . Since h is eventually nondecreasing, there exists $J _ { * } ~ \geq ~ 1$ such that h is nondecreasing on $[ 2 ^ { J _ { * } } , \infty )$ . Also, by the convergence of the series, we can choose a strictly increasing sequence of integers $J _ { * } \leq J _ { 1 } < J _ { 2 } < \cdot \cdot \cdot$ such that for every $\begin{array} { r } { k \geq 1 , \sum _ { j = J _ { k } } ^ { \infty } a _ { j } < 2 ^ { - k } k ^ { - 2 } } \end{array}$ . Now we define a positive sequence {c<sub>k</sub>}<sub>k≥1</sub> by

$$
c _ { j } : = \displaystyle \frac { 1 , \quad 1 \leq j < J _ { 1 } } { k , \quad J _ { k } \leq j < J _ { k + 1 } , \quad k \geq 1 }
$$

Then $c _ { j } \to \infty$ . Moreover,

$$
\sum _ { j = 1 } ^ { \infty } c _ { j } a _ { j } = \sum _ { j = 1 } ^ { J _ { 1 } - 1 } a _ { j } + \sum _ { k = 1 } ^ { \infty } k \sum _ { j = J _ { k } } ^ { J _ { k + 1 } - 1 } a _ { j } \le \sum _ { j = 1 } ^ { J _ { 1 } - 1 } a _ { j } + \sum _ { k = 1 } ^ { \infty } k \sum _ { j = J _ { k } } ^ { \infty } a _ { j } < \sum _ { j = 1 } ^ { J _ { 1 } - 1 } a _ { j } + \sum _ { k = 1 } ^ { \infty } \frac { 2 ^ { - k } } { k } < \infty .
$$

Now define $r _ { j } : = 1 / \sqrt { c _ { j } a _ { j } } = h _ { j } / \sqrt { c _ { j } }$ and $g _ { j } : = \mathrm { m a x } _ { 1 \leq i \leq j } r _ { i }$ for $j \geq 1$ . Then $\{ g _ { j } \} _ { j \ge 1 }$ is positive and nondecreasing. Since $g _ { j } \ge r _ { j } , g _ { j } ^ { - 2 } \le r _ { j } ^ { - 2 } = c _ { j } a _ { j }$ . Therefore,

$$
\sum _ { j = 1 } ^ { \infty } \frac { 1 } { g _ { j } ^ { 2 } } \leq \sum _ { j = 1 } ^ { \infty } c _ { j } a _ { j } < \infty .
$$

We next prove $g _ { j } = o ( h _ { j } )$ . For any $\varepsilon > 0$ , there exists an integer $K \in \mathbb { N } _ { + }$ such that $K ^ { - 1 / 2 } < \varepsilon$ . For every $i \geq J _ { K }$ , we have $c _ { i } \geq K$ . Since $J _ { K } \geq J _ { * }$ , for every $J _ { K } \leq i \leq j$ we have $h _ { i } \leq h _ { j }$ . Hence,

$$
r _ { i } = \frac { h _ { i } } { \sqrt { c _ { i } } } \leq \frac { h _ { j } } { \sqrt { K } } < \varepsilon h _ { j } .
$$

Now for the finitely many indices $i < J _ { K }$ , define $M _ { K } : = \operatorname* { m a x } _ { 1 \leq i < J _ { K } } r _ { i }$ with the convention $M _ { K } : = 0$ if $J _ { K } = 1$ . Since $h _ { j } \to \infty$ , there exists $J ^ { \prime } \geq J _ { K }$ such that $M _ { K } < \varepsilon h _ { j }$ for any $j \geq J ^ { \prime }$ . Thus, for every $j \geq J ^ { \prime }$

$$
g _ { j } = \operatorname* { m a x } _ { 1 \leq i \leq j } r _ { i } \leq \varepsilon h _ { j } ,
$$

which implies $g _ { j } / h _ { j } \to 0 \mathrm { ~ a s ~ } j \to \infty$ . Finally, we define $g : \mathbb { N } _ { + } \to ( 0 , \infty )$ by

$$
g ( n ) : = g _ { \mathrm { m a x } \{ 1 , \lfloor \log _ { 2 } n \rfloor \} } .
$$

Now we only need to prove $g ( n ) = o { \bigl ( } h ( n ) { \bigr ) }$ . For suficiently large $n ,$ let $j = \lfloor \log _ { 2 } n \rfloor$ . Then $2 ^ { j } \leq n < 2 ^ { j + 1 }$ and eventual monotonicity of h yields $h ( n ) \geq h ( 2 ^ { j } ) = h _ { j }$ . Therefore,

$$
0 \leq \frac { g ( n ) } { h ( n ) } = \frac { g _ { j } } { h ( n ) } \leq \frac { g _ { j } } { h _ { j } }  0 .
$$

Hence $g ( n ) = o { \bigl ( } h ( n ) { \bigr ) }$ . Since $h ( n ) = o ( { \sqrt { n } } )$ , it follows that $g ( n ) = o ( { \sqrt { n } } )$ and therefore $g \in \mathcal { H } _ { 0 }$ . Moreover,

$$
\sum _ { j = 1 } ^ { \infty } \frac { 1 } { g ( 2 ^ { j } ) ^ { 2 } } = \sum _ { j = 1 } ^ { \infty } \frac { 1 } { g _ { j } ^ { 2 } } < \infty .
$$

By Theorem $2 . 3 , \ g \ \in \ \mathcal { H } _ { \mathrm { a c h } }$ . Since $g ( n ) \ = \ o ( h ( n ) )$ , every $h \in \mathcal { H } _ { \mathrm { a c h } }$ admits another $g \in \mathcal { H } _ { \mathrm { a c h } }$ that is asymptotically strictly smaller. Therefore, ${ \mathcal { H } } _ { \mathrm { a c h } }$ has no asymptotically smallest element. □

## B Proofs and auxiliary results for suficiency

## B.1 Additive conditional restart theorem: proof and supporting lemmas

Let $y \in \mathcal { H }$ be an arbitrary comparator. We define the squared-distance function $D : \mathcal { H } \times \mathcal { H }  \mathbb { R } _ { + } \cup \{ 0 \}$ by $D ( a , b ) = \| a - b \| ^ { 2 } / 2$

## B.1.1 Local smooth geometry

Our analysis is based on two geometric lemmas.

Lemma B.1. For any $x , u , z \in \mathcal { H }$

$$
f ( u ) - f ( z ) \leq \langle \nabla f ( x ) , u - z \rangle + \frac { L } { 2 } \| u - x \| ^ { 2 } .
$$

Proof. By the L-smoothness of $f ,$ we have

$$
f ( u ) \leq f ( x ) + \langle \nabla f ( x ) , u - x \rangle + \frac { L } { 2 } \| u - x \| ^ { 2 } .
$$

By the convexity of $f ,$ we have

$$
f ( x ) - f ( z ) \leq \langle \nabla f ( x ) , x - z \rangle .
$$

Adding these two inequalities yields

$$
f ( u ) - f ( z ) \leq \langle \nabla f ( x ) , u - z \rangle + \frac { L } { 2 } \| u - x \| ^ { 2 } .
$$

Lemma B.2. Let $x ^ { + } = x - \eta ( \nabla f ( x ) + \zeta )$ with $0 < \eta \leq 1 / ( 2 L )$ . Then for any $z \in \mathcal { H }$

$$
f ( x ^ { + } ) - f ( z ) \leq \frac { D ( z , x ) - D ( z , x ^ { + } ) } { \eta } + \langle \zeta , z - x \rangle + \eta \| \zeta \| ^ { 2 } .
$$

Proof. Applying Lemma B.1 with $u = x ^ { + }$ gives

$$
\begin{array} { c l } { f ( x ^ { + } ) - f ( z ) \leq \langle \nabla f ( x ) , x ^ { + } - z \rangle + \displaystyle \frac { L \eta ^ { 2 } } { 2 } \| \nabla f ( x ) + \zeta \| ^ { 2 } } \\ { \displaystyle = \frac { \langle \eta ( \nabla f ( x ) + \zeta ) , x ^ { + } - z \rangle } { \eta } - \langle \zeta , x ^ { + } - z \rangle + \displaystyle \frac { L \eta ^ { 2 } } { 2 } \| \nabla f ( x ) + \zeta \| ^ { 2 } } \end{array}
$$

By the identity

$$
\| a + b \| ^ { 2 } = \| a \| ^ { 2 } + 2 \langle a , b \rangle + \| b \| ^ { 2 } ,
$$

we have

$$
\langle \eta ( \nabla f ( x ) + \zeta ) , x ^ { + } - z \rangle = - \frac { 1 } { 2 } \eta ^ { 2 } \| \nabla f ( x ) + \zeta \| ^ { 2 } - \frac { 1 } { 2 } \| x ^ { + } - z \| ^ { 2 } + \frac { 1 } { 2 } \| x - z \| ^ { 2 } .
$$

Hence, we have

$$
f ( x ^ { + } ) - f ( z ) \leq \frac { D ( x , z ) - D ( x ^ { + } , z ) } { \eta } - \langle \zeta , x ^ { + } - z \rangle + \frac { \eta } { 2 } ( L \eta - 1 ) \| \nabla f ( x ) + \zeta \| ^ { 2 }
$$

Notice that

$$
\eta \langle \zeta , \nabla f ( x ) \rangle - \frac { \eta } { 4 } \| \nabla f ( x ) + \zeta \| ^ { 2 } = - \frac { \eta } { 4 } \| \nabla f ( x ) - \zeta \| ^ { 2 } \leq 0 .
$$

Therefore, we have

$$
f ( x ^ { + } ) - f ( z ) \leq \frac { D ( x , z ) - D ( x ^ { + } , z ) } { \eta } + \langle \zeta , z - x \rangle + \eta \| \zeta \| ^ { 2 } .
$$

Let $\rho \in [ 0 , 1 ]$ and set $z = ( 1 - \rho ) x + \rho q$ . Since $D ( z , x ) = \rho ^ { 2 } D ( q , x ) \leq \rho D ( q , x )$ , Lemma B.2 gives

$$
f ( x ^ { + } ) - f ( z ) \leq \rho \langle \zeta , q - x \rangle + \frac { 1 } { \eta } ( \rho D ( q , x ) - D ( z , x ^ { + } ) ) + \eta \| \zeta \| ^ { 2 } .
$$

## B.1.2 Terminal weights and barycentric identities

In the following, consider the iteration

$$
X ^ { t + 1 } = X ^ { t } - \eta _ { t } ( \nabla f ( X ^ { t } ) + Z ^ { t } ) , \quad 1 \leq t \leq T , 0 < \eta _ { t } \leq \frac { 1 } { 2 L } .
$$

for some $T \geq 1$

Our objective in this section is to bound the global error $f ( X ^ { T + 1 } ) - f ( y )$ evaluated at an arbitrary comparator $y \in \mathcal H$ . Since the one-step inequality in Lemma B.2 bounds the single-step error $f ( X ^ { t + 1 } ) { - } f ( z ^ { t } )$ relative to a reference point $z ^ { t }$ , we need to construct a linear combination of these single-step errors that telescopes exactly to the global error.

Let $a _ { t } > 0$ denote the linear-combination coeficient at time t. We consider the corresponding weighted sum of single-step errors. To relate $f ( z ^ { t } )$ to the iterates $f ( X ^ { t } )$ , we define $z ^ { t }$ via a barycentric recursion. Let $\{ v _ { t } \} _ { t = 1 } ^ { T }$ be a positive, nondecreasing sequence of parameters with $0 < v _ { 0 } = v _ { 1 }$ and $v _ { T } = 1$ , and define

$$
z ^ { t } : = \bigg ( 1 - \frac { v _ { t - 1 } } { v _ { t } } \bigg ) X ^ { t } + \frac { v _ { t - 1 } } { v _ { t } } z ^ { t - 1 } , \quad z ^ { 0 } : = y .
$$

By applying convexity to this recursion, we have

$$
v _ { t } f ( z ^ { t } ) \leq ( v _ { t } - v _ { t - 1 } ) f ( X ^ { t } ) + v _ { t - 1 } f ( z ^ { t - 1 } ) \leq v _ { 0 } f ( y ) + \sum _ { s = 1 } ^ { t } ( v _ { s } - v _ { s - 1 } ) f ( X ^ { s } ) .
$$

Substituting this inequality into the linear combination of single-step errors, we have

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { T } a _ { t } \big ( f \big ( X ^ { t + 1 } \big ) - f ( z ^ { t } ) \big ) \geq \displaystyle \sum _ { t = 1 } ^ { T } a _ { t } f \big ( X ^ { t + 1 } \big ) - \displaystyle \sum _ { t = 1 } ^ { T } \frac { a _ { t } } { v _ { t } } \bigg ( v _ { 0 } f \big ( y \big ) + \displaystyle \sum _ { s = 1 } ^ { t } \big ( v _ { s } - v _ { s - 1 } \big ) f \big ( X ^ { s } \big ) \bigg ) } & { } \\ { \displaystyle } & { = \displaystyle \sum _ { t = 2 } ^ { T } \bigg ( a _ { t - 1 } - \big ( v _ { t } - v _ { t - 1 } \big ) \displaystyle \sum _ { k = t } ^ { T } \frac { a _ { k } } { v _ { k } } \bigg ) f \big ( X ^ { t } \big ) } \\ { \displaystyle } & { \quad - v _ { 0 } f \big ( y \big ) \displaystyle \sum _ { t = 1 } ^ { T } \frac { a _ { t } } { v _ { t } } + a _ { T } f \big ( X ^ { T + 1 } \big ) . } \end{array}
$$

Let $\begin{array} { r } { a _ { t - 1 } = \left( v _ { t } - v _ { t - 1 } \right) \sum _ { k = t } ^ { T } \frac { a _ { k } } { v _ { k } } } \end{array}$ . Then the first term vanishes. Let $\begin{array} { r } { b _ { t } = \frac { a _ { t } } { v _ { t } } } \end{array}$ and $\begin{array} { r } { W _ { t } = \sum _ { k = t } ^ { T } b _ { k } } \end{array}$ . Then we have $a _ { t - 1 } = ( v _ { t } - v _ { t - 1 } ) W _ { t }$ . On the other hand, $a _ { t - 1 } = b _ { t - 1 } v _ { t - 1 }$ . Hence, we have the following identity:

$$
b _ { t - 1 } v _ { t - 1 } = ( v _ { t } - v _ { t - 1 } ) W _ { t } , \quad 2 \leq t \leq T .
$$

Also note that $b _ { t - 1 } = W _ { t - 1 } - W _ { t }$ . Substituting this into the previous identity, we have

$$
v _ { t - 1 } W _ { t - 1 } = v _ { t } W _ { t } \implies v _ { t } W _ { t } = v _ { T } W _ { T } = a _ { T } , \quad 2 \le t \le T .
$$

Since $v _ { 0 } = v _ { 1 }$ , we have $v _ { 0 } W _ { 1 } = v _ { 1 } W _ { 1 } = a _ { T }$ . Substituting this back into the previous inequality, we have

$$
\sum _ { t = 1 } ^ { T } a _ { t } ( f ( X ^ { t + 1 } ) - f ( z ^ { t } ) ) \geq a _ { T } ( f ( X ^ { T + 1 } ) - f ( y ) ) .
$$

Now let $b _ { t } = w _ { t } \eta _ { t }$ , where $w _ { t }$ is a nonincreasing, nonnegative weight sequence. Then $a _ { t } = w _ { t } \eta _ { t } v _ { t }$ . Let $a _ { T } = C$ . Then we have the following lemma.

Lemma B.3. The following inequality holds:

$$
C ( f ( X ^ { T + 1 } ) - f ( y ) ) \leq w _ { 1 } v _ { 0 } D ( y , X ^ { 1 } ) + \sum _ { t = 1 } ^ { T } c _ { t } \| Z ^ { t } \| ^ { 2 } + \sum _ { t = 1 } ^ { T } \beta _ { t } \langle Z ^ { t } , d _ { t } \rangle + \gamma _ { T } ,
$$

where $c _ { t } = w _ { t } \eta _ { t } ^ { 2 } v _ { t } , \beta _ { t } = w _ { t } \eta _ { t } v _ { t - 1 } , d _ { t } = z ^ { t - 1 } - X ^ { t }$ , and

$$
\mathcal { V } _ { T } = \sum _ { t = 2 } ^ { T } ( w _ { t } - w _ { t - 1 } ) v _ { t - 1 } D ( z ^ { t - 1 } , X ^ { t } ) \leq 0 .
$$

Proof. Applying Lemma B.2 to each term in the summation $\begin{array} { r } { \sum _ { t = 1 } ^ { T } a _ { t } ( f ( X ^ { t + 1 } ) - f ( z ^ { t } ) ) } \end{array}$ , we have

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } a _ { t } \big ( f ( X ^ { t + 1 } ) - f ( z ^ { t } ) \big ) } \\ & { \displaystyle \leq \sum _ { t = 1 } ^ { T } w _ { t } \eta _ { t - 1 } D ( z ^ { t - 1 } , X ^ { t } ) - \displaystyle \sum _ { t = 1 } ^ { T } w _ { t } v _ { t } D ( z ^ { t } , X ^ { t + 1 } ) + \displaystyle \sum _ { t = 1 } ^ { T } c _ { 4 } \| Z ^ { t } \| ^ { 2 } + \displaystyle \sum _ { t = 1 } ^ { T } \beta _ { t } ( Z ^ { t } , d _ { t } ) } \\ & { = w _ { 1 } v _ { 0 } D ( y , X ^ { 1 } ) + \displaystyle \sum _ { t = 2 } ^ { T } \big ( w _ { t } - w _ { t - 1 } ) v _ { t - 1 } D ( z ^ { t - 1 } , X ^ { t } ) } \\ & { \displaystyle - w _ { T } v _ { T } D ( z ^ { T } , X ^ { T + 1 } ) + \displaystyle \sum _ { t = 1 } ^ { T } c _ { 4 } \| Z ^ { t } \| ^ { 2 } + \displaystyle \sum _ { t = 1 } ^ { T } \beta _ { t } \langle Z ^ { t } , d _ { t } \rangle } \\ & { \displaystyle \leq w _ { 1 } v _ { 0 } D ( y , X ^ { 1 } ) + \displaystyle \sum _ { t = 1 } ^ { T } c _ { 4 } \| Z ^ { t } \| ^ { 2 } + \displaystyle \sum _ { t = 2 } ^ { T } \beta _ { t } \langle Z ^ { t } , d _ { t } \rangle + V _ { T } . } \end{array}
$$

where the last inequality follows from the fact that $- w _ { T } v _ { T } D ( z ^ { T } , X ^ { T + 1 } ) \leq 0$ and

$$
\mathcal { V } _ { T } = \sum _ { t = 2 } ^ { T } ( w _ { t } - w _ { t - 1 } ) v _ { t - 1 } D ( z ^ { t - 1 } , X ^ { t } ) .
$$

Since $w _ { t }$ is nonincreasing, $\nu _ { T } \leq 0$

## B.1.3 Proof of the additive conditional restart theorem

Let $w _ { t } = T - t + 1$ . Consider the constant stepsize $\eta _ { t } = \gamma$ for $1 \leq t \leq T$ . Then $b _ { t } = w _ { t } \eta _ { t } = \gamma w _ { t }$ and $\begin{array} { r } { W _ { t } = \sum _ { k = t } ^ { T } b _ { k } = \gamma w _ { t } ( w _ { t } + 1 ) / 2 } \end{array}$ . We also have $C = a _ { T } = w _ { T } \gamma = \gamma$ . Then by the identity $v _ { t } W _ { t } = C$ , we have

$$
v _ { t } = { \frac { C } { W _ { t } } } = { \frac { 2 } { w _ { t } ( w _ { t } + 1 ) } } .
$$

Substituting these expressions into Lemma B.3 gives

$$
f ( X ^ { T + 1 } ) - f ( y ) \leq \frac { 2 D ( y , X ^ { 1 } ) } { \gamma ( T + 1 ) } + \sum _ { t = 1 } ^ { T } w _ { t } \gamma v _ { t } \Vert Z ^ { t } \Vert ^ { 2 } + \sum _ { t = 1 } ^ { T } w _ { t } v _ { t - 1 } \langle Z ^ { t } , d _ { t } \rangle - \sum _ { t = 2 } ^ { T } \frac { v _ { t - 1 } } { 2 \gamma } \Vert d _ { t } \Vert ^ { 2 } .
$$

To bound the error, we need the following two lemmas.

Lemma B.4. For any $\delta \in ( 0 , 1 )$ , the following inequality holds:

$$
\mathbb { P } \bigg ( \sum _ { t = 1 } ^ { T } w _ { t } \gamma v _ { t } \| Z ^ { t } \| ^ { 2 } \leq 2 { \sigma } ^ { 2 } \gamma \mathsf { H } _ { T } + { \sigma } ^ { 2 } \gamma \log \frac { 1 } { \delta } \bigg ) \geq 1 - \delta ,
$$

where $\begin{array} { r } { \mathsf { H } _ { T } = \sum _ { i = 1 } ^ { T } \frac { 1 } { i } } \end{array}$

Proof. Let $a _ { * } = \operatorname* { m a x } _ { 1 \leq i \leq T } \left\{ w _ { i } \gamma v _ { i } \right\}$ . Let $\{ \mathcal { F } _ { t } \} _ { t = 1 } ^ { T }$ be a filtration such that $Z ^ { t }$ is measurable with respect to $\mathcal { F } _ { t }$ . Let

$$
N _ { t } = \exp { \bigg ( \sum _ { i = 1 } ^ { t } { \frac { a _ { i } } { a _ { * } } } { \bigg ( } { \frac { \| Z ^ { i } \| ^ { 2 } } { \sigma ^ { 2 } } } - 1 { \bigg ) } { \bigg ) } } , \quad a _ { i } = w _ { i } \gamma v _ { i } .
$$

$\mathrm { B y }$ the sub-Gaussian property of $Z ^ { t }$ , we have

$$
\mathbb { E } \bigg [ \exp \bigg ( \frac { \| Z ^ { t } \| ^ { 2 } } { \sigma ^ { 2 } } - 1 \bigg ) \bigg | \mathcal { F } _ { t - 1 } \bigg ] \leq 1 .
$$

Since $a _ { t } / a _ { * } \leq 1 , x ^ { a _ { t } / a _ { * } }$ <sup>∗</sup> is concave for $x \geq 0$ . By Jensen’s inequality, we have

$$
\mathbb { E } \bigg [ \exp \bigg ( \frac { a _ { t } } { a _ { * } } \bigg ( \frac { \| Z ^ { t } \| ^ { 2 } } { \sigma ^ { 2 } } - 1 \bigg ) \bigg ) \bigg | \mathcal { F } _ { t - 1 } \bigg ] \leq \bigg \{ \mathbb { E } \bigg [ \exp \bigg ( \frac { \| Z ^ { t } \| ^ { 2 } } { \sigma ^ { 2 } } - 1 \bigg ) \bigg | \mathcal { F } _ { t - 1 } \bigg ] \bigg \} ^ { a _ { t } / a _ { * } } \leq 1 .
$$

Since

$$
\frac { N _ { t + 1 } } { N _ { t } } = \exp { \left( \frac { a _ { t + 1 } } { a _ { * } } \left( \frac { \lVert Z ^ { t + 1 } \rVert ^ { 2 } } { \sigma ^ { 2 } } - 1 \right) \right) } ,
$$

we have

$$
\mathbb { E } ( N _ { t + 1 } / N _ { t } \mid \mathcal { F } _ { t } ) \le 1 \implies \mathbb { E } ( N _ { t + 1 } \mid \mathcal { F } _ { t } ) \le N _ { t } .
$$

Therefore, $\{ N _ { t } \}$ is a supermartingale. By Ville’s inequality, we have

$$
\mathbb { P } ( N _ { T } \le 1 / \delta ) \ge 1 - \delta , \quad \forall \delta \in ( 0 , 1 ) .
$$

The event $N _ { T } \leq 1 / \delta$ implies

$$
\sum _ { t = 1 } ^ { T } w _ { t } \gamma v _ { t } \Vert Z ^ { t } \Vert ^ { 2 } \leq 2 \sigma ^ { 2 } \gamma \mathsf { H } _ { T } + \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } ,
$$

where $\begin{array} { r } { \mathsf { H } _ { T } = \sum _ { i = 1 } ^ { T } \frac { 1 } { i } } \end{array}$ . Therefore, we have

$$
\mathbb { P } \bigg ( \sum _ { t = 1 } ^ { T } w _ { t } \gamma v _ { t } \| Z ^ { t } \| ^ { 2 } \leq 2 \sigma ^ { 2 } \gamma \mathsf { H } _ { T } + \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } \bigg ) \geq 1 - \delta ,
$$

Lemma B.5. For any $\delta \in ( 0 , 1 )$ , the following inequality holds:

$$
\mathbb { P } \bigg ( \sum _ { t = 1 } ^ { T } w _ { t } v _ { t - 1 } \langle Z ^ { t } , d _ { t } \rangle - \sum _ { t = 2 } ^ { T } \frac { v _ { t - 1 } } { 2 \gamma } \| d _ { t } \| ^ { 2 } \leq \frac { 2 D ( y , X ^ { 1 } ) } { \gamma ( T + 1 ) ^ { 2 } } + 8 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } \bigg ) \geq 1 - \delta .
$$

Proof. Let $\mathcal { F } _ { t }$ be a filtration generated by $Z ^ { 1 } , \ldots , Z ^ { t }$ . The sub-Gaussian property of $Z ^ { t }$ implies that for any $\lambda > 0$ ，

$$
\begin{array} { r } { \mathbb { E } [ \exp ( \langle Z ^ { t } , \lambda w _ { t } v _ { t - 1 } d _ { t } \rangle ) \mid { \mathcal F } _ { t - 1 } ] \leq \exp ( 2 \sigma ^ { 2 } \lambda ^ { 2 } w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } \| d _ { t } \| ^ { 2 } ) . } \end{array}
$$

Hence, we can construct a supermartingale $M _ { t }$ as follows:

$$
{ \cal M } _ { t } = \exp \left( \lambda \sum _ { i = 1 } ^ { t } w _ { i } v _ { i - 1 } \langle Z ^ { i } , d _ { i } \rangle - 2 \sigma ^ { 2 } \lambda ^ { 2 } \sum _ { i = 1 } ^ { t } w _ { i } ^ { 2 } v _ { i - 1 } ^ { 2 } \| d _ { i } \| ^ { 2 } \right)
$$

It is straightforward to verify that $M _ { t }$ is a supermartingale. By Ville’s inequality, we have

$$
\mathbb { P } ( M _ { T } \le 1 / \delta ) \ge 1 - \delta , \quad \forall \delta \in ( 0 , 1 ) .
$$

The event $M _ { T } \le 1 / \delta$ implies

$$
\sum _ { t = 1 } ^ { T } w _ { t } v _ { t - 1 } \langle Z ^ { t } , d _ { t } \rangle \leq 2 \sigma ^ { 2 } \lambda \sum _ { t = 1 } ^ { T } w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } \| d _ { t } \| ^ { 2 } + \frac { \log ( 1 / \delta ) } { \lambda } .
$$

Note that

$$
\begin{array} { r l r } & { } & { 2 \sigma ^ { 2 } \lambda \displaystyle \sum _ { t = 1 } ^ { T } w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } \| d _ { t } \| ^ { 2 } = 2 \sigma ^ { 2 } \lambda w _ { 1 } ^ { 2 } v _ { 0 } ^ { 2 } \| d _ { 1 } \| ^ { 2 } + 2 \sigma ^ { 2 } \lambda \displaystyle \sum _ { t = 2 } ^ { T } w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } \| d _ { t } \| ^ { 2 } } \\ & { } & { \qquad = 2 \sigma ^ { 2 } \lambda w _ { 1 } ^ { 2 } v _ { 1 } ^ { 2 } \| d _ { 1 } \| ^ { 2 } + 2 \sigma ^ { 2 } \lambda \displaystyle \sum _ { t = 2 } ^ { T } w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } \| d _ { t } \| ^ { 2 } } \\ & { } & { \qquad = \displaystyle \frac { 8 \sigma ^ { 2 } \lambda } { ( T + 1 ) ^ { 2 } } \| d _ { 1 } \| ^ { 2 } + 2 \sigma ^ { 2 } \lambda \displaystyle \sum _ { t = 2 } ^ { T } w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } \| d _ { t } \| ^ { 2 } . } \end{array}
$$

Observe that

$$
\frac { 2 \gamma ( w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } ) } { v _ { t - 1 } } = \frac { 4 \gamma w _ { t } ^ { 2 } } { ( w _ { t } + 1 ) ( w _ { t } + 2 ) } \leq 4 \gamma .
$$

Hence, we have

$$
2 \sigma ^ { 2 } \lambda \sum _ { t = 2 } ^ { T } w _ { t } ^ { 2 } v _ { t - 1 } ^ { 2 } \| d _ { t } \| ^ { 2 } \leq 8 \sigma ^ { 2 } \lambda \gamma \sum _ { t = 2 } ^ { T } \frac { v _ { t - 1 } } { 2 \gamma } \| d _ { t } \| ^ { 2 } .
$$

Taking $\lambda = 1 / ( 8 \sigma ^ { 2 } \gamma )$ , we have

$$
\sum _ { t = 1 } ^ { T } w _ { t } v _ { t - 1 } \langle Z ^ { t } , d _ { t } \rangle - \sum _ { t = 2 } ^ { T } \frac { v _ { t - 1 } } { 2 \gamma } \| d _ { t } \| ^ { 2 } \leq \frac { 2 D ( y , X ^ { 1 } ) } { \gamma ( T + 1 ) ^ { 2 } } + 8 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } .
$$

This holds with probability at least $1 - \delta ,$ or equivalently,

$$
\mathbb { P } \bigg ( \sum _ { t = 1 } ^ { T } w _ { t } v _ { t - 1 } \langle Z ^ { t } , d _ { t } \rangle - \sum _ { t = 2 } ^ { T } \frac { v _ { t - 1 } } { 2 \gamma } \| d _ { t } \| ^ { 2 } \leq \frac { 2 D ( y , X ^ { 1 } ) } { \gamma ( T + 1 ) ^ { 2 } } + 8 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } \bigg ) \geq 1 - \delta .
$$

With the master inequality and the two martingale bounds established, we are ready to prove Theorem 3.2. We first prove the following lemma, which is a direct consequence of Lemmas B.3, B.4, and B.5.

Lemma B.6. The following inequality holds with probability at least $1 - \delta .$

$$
f ( X ^ { T + 1 } ) - f ( y ) \leq \frac { 4 D ( y , X ^ { 1 } ) } { \gamma T } + 6 \sigma ^ { 2 } \gamma \mathsf { H } _ { T } + 9 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } .
$$

Proof. We apply Lemma B.4 with failure probability pδ and Lemma B.5 with failure probability $( 1 - p ) \delta$ for some $p \in ( 0 , 1 )$ . By the union bound, with probability at least $1 - \delta$

$$
\begin{array} { l } { \displaystyle f ( X ^ { T + 1 } ) - f ( y ) \leq \frac { 2 D ( y , X ^ { 1 } ) } { \gamma ( T + 1 ) } \bigg ( 1 + \frac { 1 } { T + 1 } \bigg ) + 2 \sigma ^ { 2 } \gamma \mathsf { H } _ { T } + 9 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } - \sigma ^ { 2 } \gamma ( \log p + 8 \log ( 1 - p ) ) } \\ { \displaystyle \qquad \leq \frac { 4 D ( y , X ^ { 1 } ) } { \gamma T } + 2 \sigma ^ { 2 } \gamma \mathsf { H } _ { T } + 9 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } + \sigma ^ { 2 } \gamma \bigg ( \log 9 + 8 \log \frac { 9 } { 8 } \bigg ) } \\ { \displaystyle \qquad \leq \frac { 4 D ( y , X ^ { 1 } ) } { \gamma T } + 6 \sigma ^ { 2 } \gamma \mathsf { H } _ { T } + 9 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } , } \end{array}
$$

In the second inequality, we take $p = 1 / 9$ to minimize the term $- \log p - 8 \log ( 1 - p )$ . Its minimum has absolute value at most 4, so the last inequality follows. Therefore,

$$
\mathbb { P } \Bigg ( f ( X ^ { T + 1 } ) - f ( y ) \leq \frac { 4 D ( y , X ^ { 1 } ) } { \gamma T } + 6 \sigma ^ { 2 } \gamma \mathsf { H } _ { T } + 9 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } \Bigg ) \geq 1 - \delta .
$$

For clarity, we restate Theorem 3.2.

Theorem B.7. Fix $L , R , \sigma > 0$ , an instance $\mathcal { P } \in \mathfrak { P } _ { \mathrm { c v x } } ( L , R , \sigma )$ , and a deterministic nonnegative stepsize schedule η. Let m ∈ N and $r ~ \in ~ \mathbb { N } _ { + }$ be deterministic, and suppose that $\eta _ { m } = \cdot \cdot \cdot = \eta _ { m + r - 1 } = \gamma _ { \mathrm { ~ } }$ $0 < \gamma \leq 1 / ( 2 L )$ . Then for every almost surely finite $\mathcal { F } _ { m }$ -measurable comparator y and every $\delta \in ( 0 , 1 )$

$$
\mathbb { P } \bigg ( f ( x _ { m + r } ) - f ( y ) \leq \frac { 4 D ( y , x _ { m } ) } { \gamma r } + 6 \sigma ^ { 2 } \gamma \mathsf { H } _ { r } + 9 \sigma ^ { 2 } \gamma \log \frac { 1 } { \delta } \bigg | \mathcal { F } _ { m } \bigg ) \geq 1 - \delta \quad a l m o s t s u r e l y .
$$

Proof. Condition on $\mathcal { F } _ { m } .$ , set $X ^ { t } : = x _ { m + t - 1 } , Z ^ { t } : = Z _ { m + t - 1 }$ , and $T : = r .$ , and use the local filtration $\mathcal { G } _ { t } : = \mathcal { F } _ { m + t }$ for $0 \leq t \leq r$ . Then $X ^ { 1 }$ and y are $\mathcal { G } _ { 0 }$ -measurable, while $Z ^ { t }$ is $\mathcal { G } _ { t }$ -measurable and satisfies the conditional noise assumptions given $\mathcal { G } _ { t - 1 }$ . Lemma B.6 therefore applies conditionally and gives the stated bound. □

## B.2 Global radius and calibration

For a time-uniform analysis based on finite stepsize energy, we must first control the entire trajectory. The next lemma obtains this control from two nonnegative supermartingales: one for the accumulated noise energy and one for the stopped directional noise.

Define $\begin{array} { r } { D _ { t } : = \| x _ { t } - x ^ { * } \| ^ { 2 } / 2 , Q _ { \infty } : = \sum _ { t = 0 } ^ { \infty } \eta _ { t } ^ { 2 } } \end{array}$ , and $\eta _ { * } : = \operatorname* { s u p } _ { t > 0 } \eta _ { t }$ . We assume $D _ { 0 } \leq R ^ { 2 } / 2 , Q _ { \infty } < \infty$ and $0 \leq \eta _ { t } \leq 1 / ( 2 L )$

Lemma B.8. Let $\beta \in ( 0 , 1 )$ and $\ell _ { \beta } : = \log ( 2 / \beta )$ . Define

$$
C _ { \beta } : = \frac { R ^ { 2 } } { 2 } + \frac { \sigma ^ { 2 } Q _ { \infty } } { 2 } + \frac { \sigma ^ { 2 } \eta _ { * } ^ { 2 } \ell _ { \beta } } { 2 } , \quad q _ { \beta } : = \sigma ^ { 2 } Q _ { \infty } \ell _ { \beta } ,
$$

and set

$$
B _ { \beta } : = \left( \sqrt { C _ { \beta } + 4 q _ { \beta } } + 2 \sqrt { q _ { \beta } } \right) ^ { 2 } .
$$

Then

$$
\begin{array} { r } { \mathbb { P } \left( D _ { t } \leq B _ { \beta } , \forall t \geq 0 \right) \geq 1 - \beta . } \end{array}
$$

Proof. We divide the proof into four steps.

Step $1 \colon \ A$ pathwise distance recursion. Set $\bar { x } _ { t + 1 } : = x _ { t } - \eta _ { t } \nabla f ( x _ { t } )$ . Since f is convex and L-smooth and $\nabla f ( x ^ { * } ) = 0 .$ , cocoercivity gives

$$
\langle \nabla f ( x _ { t } ) , x _ { t } - x ^ { * } \rangle \geq { \frac { 1 } { L } } \| \nabla f ( x _ { t } ) \| ^ { 2 } .
$$

Therefore, for every $0 \leq \eta _ { t } \leq 2 / L$

$$
\begin{array} { r l } & { \| \bar { x } _ { t + 1 } - x ^ { * } \| ^ { 2 } = \| x _ { t } - x ^ { * } \| ^ { 2 } - 2 \eta _ { t } \langle \nabla f ( x _ { t } ) , x _ { t } - x ^ { * } \rangle + \eta _ { t } ^ { 2 } \| \nabla f ( x _ { t } ) \| ^ { 2 } } \\ & { \qquad \leq \| x _ { t } - x ^ { * } \| ^ { 2 } - \eta _ { t } \left( \displaystyle \frac { 2 } { L } - \eta _ { t } \right) \| \nabla f ( x _ { t } ) \| ^ { 2 } } \\ & { \qquad \leq \| x _ { t } - x ^ { * } \| ^ { 2 } . } \end{array}
$$

Thus the deterministic gradient step is nonexpansive relative to $x ^ { * }$ . Since $x _ { t + 1 } = \bar { x } _ { t + 1 } - \eta _ { t } Z _ { t }$ , we obtain the pathwise inequality

$$
D _ { t + 1 } \leq D _ { t } + \frac { \eta _ { t } ^ { 2 } } { 2 } \| Z _ { t } \| ^ { 2 } - \eta _ { t } \langle Z _ { t } , \bar { x } _ { t + 1 } - x ^ { * } \rangle .\tag{4}
$$

Step 2: Time-uniform control of the accumulated noise energy. If $\eta _ { * } = 0$ , then every step is zero and the result is immediate. Hence, assume $\eta _ { * } > 0$ and put $c _ { t } : = \eta _ { t } ^ { 2 } / \eta _ { * } ^ { 2 } \in [ 0 , 1 ]$ . The conditional noise assumption and conditional Jensen’s inequality give

$$
\mathbb { E } \left[ \exp \left( c _ { t } \left( \frac { \| Z _ { t } \| ^ { 2 } } { \sigma ^ { 2 } } - 1 \right) \right) \bigg | \mathcal { F } _ { t } \right] \leq \left\{ \mathbb { E } \left[ \exp \left( \frac { \| Z _ { t } \| ^ { 2 } } { \sigma ^ { 2 } } - 1 \right) \bigg | \mathcal { F } _ { t } \right] \right\} ^ { c _ { t } } \leq 1 .
$$

Consequently,

$$
S _ { n } : = \exp \left( \sum _ { t = 0 } ^ { n - 1 } c _ { t } \left( \frac { \| Z _ { t } \| ^ { 2 } } { \sigma ^ { 2 } } - 1 \right) \right) , \quad n \ge 0 ,
$$

is a nonnegative supermartingale with ${ \cal S } _ { 0 } = 1$ . Ville’s inequality shows that, with probability at least $1 - \beta / 2$ , simultaneously for every $n \geq 0$

$$
\sum _ { t = 0 } ^ { n - 1 } \eta _ { t } ^ { 2 } \| Z _ { t } \| ^ { 2 } \leq \sigma ^ { 2 } \sum _ { t = 0 } ^ { n - 1 } \eta _ { t } ^ { 2 } + \sigma ^ { 2 } \eta _ { * } ^ { 2 } \ell _ { \beta } \leq \sigma ^ { 2 } Q _ { \infty } + \sigma ^ { 2 } \eta _ { * } ^ { 2 } \ell _ { \beta } .\tag{5}
$$

Step 3: Time-uniform control of the stopped directional noise. We first record the directional consequence of the conditional noise assumption. For every $a , y \in \mathbb { R }$

$$
\begin{array} { r l } { \displaystyle \mathrm { e } ^ { a y } - 1 - a y \leq \frac { a ^ { 2 } y ^ { 2 } } { 2 } \mathrm { e } ^ { \lvert a y \rvert } } & { } \\ { \leq \frac { a ^ { 2 } } { 2 } \mathrm { e } ^ { a ^ { 2 } / 2 } y ^ { 2 } \mathrm { e } ^ { y ^ { 2 } / 2 } } & { } \\ { \leq \frac { a ^ { 2 } } { 2 } \mathrm { e } ^ { a ^ { 2 } / 2 } \left( \mathrm { e } ^ { y ^ { 2 } } - 1 \right) . } \end{array}
$$

If $u _ { t } \neq 0 ,$ , set $Y _ { t } : = \langle Z _ { t } , u _ { t } \rangle / ( \sigma \| u _ { t } \| )$ . Conditional centering gives $\mathbb { E } [ Y _ { t } \ | \ \mathcal { F } _ { t } ] = 0$ , while the conditional norm-square MGF gives $\mathbb { E } [ \mathrm { e } ^ { Y _ { t } ^ { 2 } } \mid \mathcal { F } _ { t } ] \leq \mathrm { e }$ . Therefore,

$$
\mathbb { E } \left[ \mathrm { e } ^ { a Y _ { t } } \big | \mathcal { F } _ { t } \right] \leq 1 + \frac { \mathrm { e } - 1 } { 2 } a ^ { 2 } \mathrm { e } ^ { a ^ { 2 } / 2 } \leq \mathrm { e } ^ { 2 a ^ { 2 } } .
$$

Here the last inequality follows from $( \mathrm { e } - 1 ) / 2 < 1$ and $s \mathrm { e } ^ { s / 2 } \le \mathrm { e } ^ { 2 s } - 1$ for $s \geq 0$ . The case $u _ { t } = 0$ is immediate. Taking $a = \lambda \sigma \| u _ { t } \|$ yields

$$
\begin{array} { r } { \mathbb { E } \left[ \exp \left( \lambda \langle Z _ { t } , u _ { t } \rangle \right) | \mathcal { F } _ { t } \right] \leq \exp \left( 2 \sigma ^ { 2 } \lambda ^ { 2 } \| u _ { t } \| ^ { 2 } \right) } \end{array}\tag{6}
$$

for every F<sub>t</sub>-measurable $u _ { t } \in \mathcal { H }$ and every $\lambda \in \mathbb { R }$ . Fix $b > 0$ , and define the stopping time $\tau _ { b } : = \operatorname* { i n f } \{ t \geq 0$ $D _ { t } > b \}$ . On $\{ t < \tau _ { b } \}$ , Step 1 gives

$$
\| \bar { x } _ { t + 1 } - x ^ { * } \| ^ { 2 } \leq \| x _ { t } - x ^ { * } \| ^ { 2 } = 2 D _ { t } \leq 2 b .
$$

Define the stopped sum

$$
M _ { n } ^ { ( b ) } : = - \sum _ { t = 0 } ^ { n - 1 } \eta _ { t } \langle Z _ { t } , \bar { x } _ { t + 1 } - x ^ { * } \rangle \mathbf { 1 } _ { \{ t < \tau _ { b } \} } .
$$

For any fixed $\lambda > 0$ , define

$$
\mathcal E _ { n } ^ { ( b ) } ( \lambda ) : = \exp \left( \lambda M _ { n } ^ { ( b ) } - 4 \sigma ^ { 2 } b \lambda ^ { 2 } \sum _ { t = 0 } ^ { n - 1 } \eta _ { t } ^ { 2 } \right) .
$$

The indicator $\mathbf { 1 } _ { \{ t < \tau _ { b } \} }$ and the vector $\bar { x } _ { t + 1 } - x ^ { * }$ are $\mathcal { F } _ { t }$ -measurable. Hence, by (6),

$$
\begin{array} { r l } & { \mathbb { E } \left[ \left. \mathcal { E } _ { n + 1 } ^ { ( b ) } ( \lambda ) \right. \mathcal { F } _ { n } \right] \leq \mathcal { E } _ { n } ^ { ( b ) } ( \lambda ) \exp \left( 2 \sigma ^ { 2 } \lambda ^ { 2 } \eta _ { n } ^ { 2 } \| \bar { x } _ { n + 1 } - x ^ { * } \| ^ { 2 } \mathbf { 1 } _ { \{ n < \tau _ { b } \} } - 4 \sigma ^ { 2 } b \lambda ^ { 2 } \eta _ { n } ^ { 2 } \right) } \\ & { \qquad \leq \mathcal { E } _ { n } ^ { ( b ) } ( \lambda ) . } \end{array}
$$

Thus $\{ \mathcal { E } _ { n } ^ { ( b ) } ( \lambda ) \} _ { n \ge 0 }$ is a nonnegative supermartingale starting from one. Ville’s inequality gives, with probability at least $1 - \beta / 2$ , simultaneously for all $n \geq 0$

$$
M _ { n } ^ { ( b ) } \leq 4 \sigma ^ { 2 } b \lambda Q _ { \infty } + \frac { \ell _ { \beta } } { \lambda } .
$$

Taking $\lambda = \sqrt { \ell _ { \beta } / ( 4 \sigma ^ { 2 } b Q _ { \infty } ) }$ yields

$$
M _ { n } ^ { ( b ) } \leq 4 \sigma \sqrt { b Q _ { \infty } \ell _ { \beta } } , \quad \forall n \geq 0 .\tag{7}
$$

Step 4: Stopping-time contradiction and calibration. Intersect the events in (5) and (7); this intersection has probability at least $1 - \beta$ . Set $b = B _ { \beta }$ , and suppose on this event that $\tau _ { b } < \infty$ . Summing (4) up to τ<sub>b</sub> gives

$$
\begin{array} { c l c r } { { \displaystyle D _ { \tau _ { b } } \leq D _ { 0 } + \frac { 1 } { 2 } \sum _ { t = 0 } ^ { \tau _ { b } - 1 } \eta _ { t } ^ { 2 } \| Z _ { t } \| ^ { 2 } + M _ { \tau _ { b } } ^ { ( b ) } } } \\ { { \leq C _ { \beta } + 4 \sqrt { b q _ { \beta } } . } } \end{array}
$$

The definition of $B _ { \beta }$ is exactly the positive solution of $b = C _ { \beta } + 4 \sqrt { b q _ { \beta } }$ . Therefore, $D _ { \tau _ { b } } \leq b ,$ contradicting the definition of $\tau _ { b }$ . Hence $\tau _ { b } = \infty ,$ and $D _ { t } \leq B _ { \beta }$ for every $t \geq 0$ on the same event. □

Remark B.9. For the dyadic schedule detailed in Algorithm $^ { 1 , }$ we have

$$
Q _ { \infty } = \sum _ { j = j _ { 0 } } ^ { \infty } N _ { j } \gamma _ { j } ^ { 2 } = \sum _ { j = j _ { 0 } } ^ { \infty } N _ { j } \frac { R ^ { 2 } } { \sigma ^ { 2 } c _ { h } ^ { 2 } N _ { j } h _ { j } ^ { 2 } } = \frac { R ^ { 2 } } { \sigma ^ { 2 } } c _ { h } ^ { - 2 } \sum _ { j = j _ { 0 } } ^ { \infty } \frac { 1 } { h _ { j } ^ { 2 } } \le \frac { R ^ { 2 } } { \sigma ^ { 2 } } .
$$

Then Lemma B.8 holds. Suppose that we assign probability $\alpha / 2$ to this event. Then we have

$$
\begin{array} { l } { { C _ { \alpha / 2 } = \displaystyle \frac { R ^ { 2 } } { 2 } + \frac { \sigma ^ { 2 } Q _ { \infty } } { 2 } + \frac { \sigma ^ { 2 } \eta _ { * } ^ { 2 } \ell _ { \alpha / 2 } } { 2 } \nonumber } } \\ { { \le \displaystyle \frac { R ^ { 2 } } { 2 } + \frac { R ^ { 2 } } { 2 } + \frac { \sigma ^ { 2 } \ell _ { \alpha / 2 } } { 2 } \cdot \frac { R ^ { 2 } } { \sigma ^ { 2 } c _ { h } ^ { 2 } N _ { j _ { 0 } } h _ { j _ { 0 } } ^ { 2 } } \le R ^ { 2 } + \frac { R ^ { 2 } \ell _ { \alpha / 2 } } { 2 } . } } \end{array}
$$

The last inequality follows from the fact that $N _ { j _ { 0 } } , h _ { j _ { 0 } } , c _ { h }$ are all at least 1. Together with the fact that $q _ { \alpha / 2 } = \sigma ^ { 2 } Q _ { \infty } \ell _ { \alpha / 2 } \leq R ^ { 2 } \ell _ { \alpha / 2 }$ , we have

$$
B _ { \alpha / 2 } = ( \sqrt { C _ { \alpha / 2 } + 4 q _ { \alpha / 2 } } + 2 \sqrt { q _ { \alpha / 2 } } ) ^ { 2 } \leq R ^ { 2 } \left( \sqrt { 1 + \frac { 9 } { 2 } \ell _ { \alpha / 2 } } + 2 \sqrt { \ell _ { \alpha / 2 } } \right) ^ { 2 } .
$$

Since $\ell _ { \alpha / 2 } = \log ( 4 / \alpha ) \geq 2 \log 2$ and

$$
\sqrt { 1 + \frac { 9 } { 2 } y } \leq \frac { 5 } { 2 } \sqrt { y } , \quad \forall y \geq 2 \log 2 ,
$$

we have

$$
B _ { \alpha / 2 } \leq { \frac { 8 1 } { 4 } } R ^ { 2 } \log { \frac { 4 } { \alpha } } .
$$

## B.3 Proof of the suficiency theorem

Throughout this subsection, h is eventually nondecreasing with $\begin{array} { r } { \sum _ { i > 1 } h ( 2 ^ { j } ) ^ { - 2 } < \infty } \end{array}$ , and $\eta = \eta ( h , L , R , \sigma )$ is the schedule produced by Algorithm 1, with the associated quantities $J _ { h } , c _ { h } , j _ { 0 } , N _ { j } = 2 ^ { j } , h _ { j } = h ( N _ { j } )$ and $\gamma _ { j } = R / ( \sigma c _ { h } \sqrt { N _ { j } } h _ { j } )$ . Recall $\Delta _ { t } = f ( x _ { t } ) - f ^ { * }$ and $D ( y , x ) = \| y - x \| ^ { 2 } / 2$

We first record the two calibration properties of the schedule that make Theorem 3.2 and Lemma B.8 applicable.

Lemma B.10 (Schedule calibration and finite energy). The schedule of Algorithm 1 satisfies:

(i) $\{ \gamma _ { j } \} _ { j \ge j _ { 0 } }$ is nonincreasing, and $0 < \gamma _ { j } \leq 1 / ( 2 L )$ for every $j \geq j _ { 0 }$ ;

(ii) the total squared-step energy is finite, with

$$
Q _ { \infty } : = \sum _ { t \geq 0 } \eta _ { t } ^ { 2 } = \sum _ { j = j _ { 0 } } ^ { \infty } N _ { j } \gamma _ { j } ^ { 2 } \leq \frac { R ^ { 2 } } { \sigma ^ { 2 } } .
$$

Proof. Since $j _ { 0 } \ge J _ { h }$ , the sequence $\{ h _ { j } \} _ { j \ge j _ { 0 } }$ is nondecreasing, and $N _ { j }$ is increasing. Hence $\sqrt { N _ { j } } h _ { j }$ is increasing and $\gamma _ { j } = R / ( \sigma c _ { h } \sqrt { N _ { j } } h _ { j } )$ is nonincreasing. The definition of j gives $\gamma _ { j _ { 0 } } = R / ( \sigma c _ { h } \sqrt { N _ { j _ { 0 } } } h _ { j _ { 0 } } ) \leq$ $1 / ( 2 L )$ , so $\gamma _ { j } \leq \gamma _ { j _ { 0 } } \leq 1 / ( 2 L )$ for every $j \geq j _ { 0 }$ . This proves (i).

For (ii), the steps vanish before epoch $j _ { 0 }$ and are constant on each later epoch, so

$$
Q _ { \infty } = \sum _ { j = j _ { 0 } } ^ { \infty } N _ { j } \gamma _ { j } ^ { 2 } = \sum _ { j = j _ { 0 } } ^ { \infty } N _ { j } \frac { R ^ { 2 } } { \sigma ^ { 2 } c _ { h } ^ { 2 } N _ { j } h _ { j } ^ { 2 } } = \frac { R ^ { 2 } } { \sigma ^ { 2 } c _ { h } ^ { 2 } } \sum _ { j = j _ { 0 } } ^ { \infty } \frac { 1 } { h _ { j } ^ { 2 } } \le \frac { R ^ { 2 } } { \sigma ^ { 2 } } ,
$$

because $j _ { 0 } \geq J _ { h }$ and $\begin{array} { r } { c _ { h } ^ { 2 } \geq \sum _ { j \geq J _ { h } } h _ { j } ^ { - 2 } } \end{array}$ by construction.

The next lemma is where reciprocal-square summability enters the constructive direction: it converts Lemma 3.3 into two constants that do not depend on the epoch index.

Lemma B.11 (Epoch constants). For every $\alpha \in ( 0 , 1 )$ , the quantities

$$
K _ { h _ { 1 } } : = \operatorname* { s u p } _ { j \geq j _ { 0 } } \frac { ( 2 4 \log 2 ) j + 1 8 \log 2 + 6 } { h _ { j } ^ { 2 } } , \qquad K _ { h _ { 2 } } : = \operatorname* { s u p } _ { j \geq j _ { 0 } } \frac { 9 \log ( 1 / \alpha ) } { h _ { j } ^ { 2 } }
$$

are finite. Moreover $K _ { h _ { 2 } } = 9 \log ( 1 / \alpha ) / h _ { j _ { 0 } } ^ { 2 } \leq 9 \log ( 1 / \alpha )$

Proof. $\mathrm { B y }$ the definition of $j _ { 0 }$ we have $h _ { j _ { 0 } } \geq 1$ , and $\{ h _ { j } \} _ { j \ge j _ { 0 } }$ is nondecreasing, so $h _ { j } \geq 1$ for every $j \geq j _ { 0 }$ This gives the stated evaluation of $K _ { h _ { 2 } }$ . For $K _ { h _ { 1 } }$ , Lemma 3.3 gives $j / h _ { j } ^ { 2 }  0$ , while (18 log 2 + 6)/h<sup>2</sup> → 0. Hence the sequence whose supremum defines $K _ { h _ { 1 } }$ converges to zero, and is therefore bounded. □

We now fix the confidence allocation. For $j \geq j _ { 0 }$ and $1 \leq r \leq N _ { j }$ , set

$$
\delta _ { j , r } : = \frac { \alpha } { 2 ^ { 2 ( j + 1 ) } } , \qquad y _ { j , r } : = \left\{ { x _ { N _ { j } } , \begin{array} { l } { 1 \leq r < N _ { j } / 2 , } \\ { x ^ { * } , \quad \ N _ { j } / 2 \leq r \leq N _ { j } , } \end{array} } \right.\tag{8}
$$

and let $A _ { j , r }$ denote the event

$$
\mathcal A _ { j , r } : = \left\{ f ( x _ { N _ { j } + r } ) - f ( y _ { j , r } ) \leq \frac { 4 D ( y _ { j , r } , x _ { N _ { j } } ) } { \gamma _ { j } r } + 6 \sigma ^ { 2 } \gamma _ { j } \mathsf H _ { r } + 9 \sigma ^ { 2 } \gamma _ { j } \log \frac { 1 } { \delta _ { j , r } } \right\} .
$$

Both choices of $y _ { j , r }$ are almost surely finite and $\mathcal { F } _ { N _ { j } }$ -measurable, and $\gamma _ { j } ~ \leq ~ 1 / ( 2 L )$ by Lemma B.10. Theorem 3.2 therefore applies with $m \ : = \ : N _ { j }$ and this $r ,$ giving $\mathbb { P } ( \mathcal { A } _ { j , r } ~ \vert ~ \mathcal { F } _ { N _ { j } } ) ~ \ge ~ 1 - \delta _ { j , r }$ and hence $\mathbb { P } ( \mathcal { A } _ { j , r } ) \geq 1 - \delta _ { j , r }$ . Let

$$
\mathcal { G } : = \left\{ D ( x ^ { * } , x _ { t } ) \leq B _ { \alpha / 2 } , \forall t \geq 0 \right\} , \qquad B _ { \alpha / 2 } \leq \frac { 8 1 } { 4 } R ^ { 2 } \log \frac { 4 } { \alpha } ,\tag{9}
$$

which by Lemma B.10 and Lemma B.8 satisfies $\mathbb { P } ( \mathcal { G } ) \ge 1 - \alpha / 2$

Lemma B.12 (Within-epoch estimate). Let $j \geq j _ { 0 }$ and $1 \leq r \leq N _ { j }$ . On the event $\mathcal { G } \cap \mathcal { A } _ { j , r } .$

(i) if $\dot { } 1 \le r < N _ { j } / 2$ , then

$$
f ( x _ { N _ { j } + r } ) - f ( x _ { N _ { j } } ) \leq \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } \cdot R \sigma \frac { h _ { j } } { \sqrt { N _ { j } } } ;
$$

(ii) if $N _ { j } / 2 \leq r \leq N _ { j }$ , then

$$
\Delta _ { N _ { j } + r } \leq \left( 1 6 2 c _ { h } \log { \frac { 4 } { \alpha } } + { \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } } \right) R \sigma { \frac { h _ { j } } { \sqrt { N _ { j } } } } .
$$

Proof. We first bound the stochastic terms, which are common to both cases. Since $r \leq N _ { i }$ , we have H<sub>r</sub> ≤ log $r + 1 \leq j$ log 2+1, and log $( 1 / \delta _ { j , r } ) = 2 ( j + 1 ) \log 2 + \log ( 1 / \alpha )$ by (8). Using $\sigma ^ { 2 } \gamma _ { j } = R \sigma / ( \bar { c } _ { h } \sqrt { N _ { j } } h _ { j } )$ we obtain

$$
\begin{array} { r l } & { 6 \sigma ^ { 2 } \gamma _ { j } \mathsf { H } _ { r } + 9 \sigma ^ { 2 } \gamma _ { j } \log \frac { 1 } { \delta _ { j , r } } \leq \frac { R \sigma } { c _ { h } \sqrt { N _ { j } } h _ { j } } \left( 6 ( j \log 2 + 1 ) + 9 \left( 2 ( j + 1 ) \log 2 + \log \frac { 1 } { \alpha } \right) \right) } \\ & { \qquad = \frac { 1 } { c _ { h } } \cdot R \sigma \frac { h _ { j } } { \sqrt { N _ { j } } } \left( \frac { ( 2 4 \log 2 ) j + 1 8 \log 2 + 6 } { h _ { j } ^ { 2 } } + \frac { 9 \log ( 1 / \alpha ) } { h _ { j } ^ { 2 } } \right) } \\ & { \qquad \leq \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } \cdot R \sigma \frac { h _ { j } } { \sqrt { N _ { j } } } , } \end{array}
$$

the last step by Lemma B.11.

For (i), the comparator is $y _ { j , r } = x _ { N _ { j } }$ , so the initial distance term vanishes: $D ( x _ { N _ { j } } , x _ { N _ { j } } ) = 0$ . On $A _ { j , i }$ the displayed bound is exactly the stochastic estimate above.

For (ii), the comparator is $y _ { j , r } = x ^ { * }$ , so $f ( x _ { N _ { j } + r } ) - f ( y _ { j , r } ) = \Delta _ { N _ { j } + r }$ . On G we have $D ( x ^ { * } , x _ { N _ { j } } ) \leq B _ { \alpha / 2 }$ and $r \geq N _ { j } / 2$ , so by (9),

$$
\frac { 4 D ( x ^ { * } , x _ { N _ { j } } ) } { \gamma _ { j } r } \leq \frac { 1 6 2 R ^ { 2 } \log ( 4 / \alpha ) } { N _ { j } \gamma _ { j } } = \frac { 1 6 2 R ^ { 2 } \log ( 4 / \alpha ) } { N _ { j } } \cdot \frac { \sigma c _ { h } \sqrt { N _ { j } } h _ { j } } { R } = 1 6 2 c _ { h } \log \frac { 4 } { \alpha } \cdot R \sigma \frac { h _ { j } } { \sqrt { N _ { j } } } .
$$

Adding the stochastic estimate gives (ii).

The latter-half estimate at the end of epoch j − 1 $j - 1$ is exactly the quantity needed to initialize the early half of epoch j. This is what propagates a single constant across all active epochs.

Lemma B.13 (Epoch propagation). Let

$$
C _ { \mathrm { l a t t e r } } : = \left( 3 2 4 c _ { h } \log { \frac { 4 } { \alpha } } + { \frac { ( 2 + \sqrt { 2 } ) ( K _ { h _ { 1 } } + K _ { h _ { 2 } } ) } { c _ { h } } } \right) R \sigma .
$$

On the event $\begin{array} { r } { \mathcal { G } \cap \bigcap _ { j \geq j _ { 0 } } \bigcap _ { 1 \leq r \leq N _ { j } } \mathcal { A } _ { j , r } } \end{array}$ , we have

$$
\Delta _ { t } \le C _ { \mathrm { l a t t e r } } \frac { h ( t ) } { \sqrt { t } } , \qquad \forall t \ge \frac { 3 N _ { j _ { 0 } } } { 2 } .
$$

Proof. Fix $j > j _ { 0 }$ and $1 \leq r < N _ { j } / 2$ . Since $N _ { j } = N _ { j - 1 } + N _ { j - 1 }$ , the index $N _ { j }$ is the terminal position of epoch $j - 1$ reached with $r = N _ { j - 1 } .$ , so Lemma B.12(ii) applied to epoch $j - 1$ gives

$$
\Delta _ { N _ { j } } \leq \left( 1 6 2 c _ { h } \log { \frac { 4 } { \alpha } } + { \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } } \right) R \sigma { \frac { h _ { j - 1 } } { \sqrt { N _ { j - 1 } } } } .
$$

Adding this to the increment bound of Lemma B.12(i) yields

$$
\begin{array} { l } { \Delta _ { N _ { j } + r } = \left( f ( x _ { N _ { j } + r } ) - f ( x _ { N _ { j } } ) \right) + \Delta _ { N _ { j } } } \\ { \leq \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } R \sigma \frac { h _ { j } } { \sqrt { N _ { j } } } + \left( 1 6 2 c _ { h } \log \frac { 4 } { \alpha } + \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } \right) R \sigma \frac { h _ { j - 1 } } { \sqrt { N _ { j - 1 } } } . } \end{array}
$$

Since $h _ { j - 1 } \leq h _ { j }$ and $N _ { j } = 2 N _ { j - 1 }$ , we have $h _ { j - 1 } / \sqrt { N _ { j - 1 } } \leq \sqrt { 2 } h _ { j } / \sqrt { N _ { j } }$ , whence

$$
\Delta _ { N _ { j } + r } \leq \left( 1 6 2 \sqrt { 2 } c _ { h } \log \frac { 4 } { \alpha } + \frac { ( \sqrt { 2 } + 1 ) ( K _ { h _ { 1 } } + K _ { h _ { 2 } } ) } { c _ { h } } \right) R \sigma \frac { h _ { j } } { \sqrt { N _ { j } } } .\tag{10}
$$

Now let $t \ge 3 N _ { j _ { 0 } } / 2$ and write $t = N _ { j } + r$ with $j \geq j _ { 0 }$ and $1 \leq r \leq N _ { j }$ . If $r \geq N _ { j } / 2$ , then $\Delta _ { t }$ obeys Lemma B.12(ii); if $r < N _ { j } / 2$ , then necessarily $j > j _ { 0 }$ and $\Delta _ { t }$ obeys (10), which dominates the former bound. In both cases

$$
\Delta _ { t } \leq \left( 1 6 2 \sqrt { 2 } c _ { h } \log \frac { 4 } { \alpha } + \frac { ( \sqrt { 2 } + 1 ) ( K _ { h _ { 1 } } + K _ { h _ { 2 } } ) } { c _ { h } } \right) R \sigma \frac { h _ { j } } { \sqrt { N _ { j } } } .
$$

Finally, $N _ { j } \leq t \leq 2 N _ { j }$ and $h _ { j } \leq h ( t )$ by monotonicity, so $h _ { j } / \sqrt { N _ { j } } \leq \sqrt { 2 } h ( t ) / \sqrt { t }$ . Multiplying the bracket by $\sqrt { 2 }$ gives $C _ { \mathrm { l a t t e r } } .$ □

It remains to absorb the finitely many iterates preceding $3 N _ { j _ { 0 } } / 2$

Lemma B.14 (Prefix absorption). Let

$$
C _ { \mathrm { p r e f i x } } : = \left( \frac { L R ^ { 2 } } { 2 } + \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } \cdot R \sigma \frac { h _ { j _ { 0 } } } { \sqrt { N _ { j _ { 0 } } } } \right) \Bigg / \operatorname* { m i n } _ { 1 \leq n < 3 N _ { j _ { 0 } } / 2 } \operatorname* { m i n } \Bigg \{ 1 , \frac { h ( n ) } { \sqrt { n } } \Bigg \} .
$$

On the event $\begin{array} { r } { \mathcal { G } \cap \bigcap _ { 1 \leq r < N _ { j _ { 0 } } / 2 } \mathcal { A } _ { j _ { 0 } , r _ { \cdot } } } \end{array}$ , we have

$$
\Delta _ { t } \le C _ { \mathrm { p r e f i x } } \operatorname* { m i n } \left\{ 1 , \frac { h ( t ) } { \sqrt { t } } \right\} , \qquad \forall 1 \le t < \frac { 3 N _ { j _ { 0 } } } { 2 } .
$$

Proof. Algorithm 1 sets $\eta _ { t } = 0$ for every $t < N _ { j _ { 0 } } ,$ , so the iterates remain at the initial point and x ${ N _ { j } } _ { 0 } = x _ { 0 }$ Smoothness at the minimizer gives the deterministic bound

$$
\Delta _ { N _ { j _ { 0 } } } = \Delta _ { t } = \Delta _ { 0 } \leq \frac { L } { 2 } \| x _ { 0 } - x ^ { * } \| ^ { 2 } \leq \frac { L R ^ { 2 } } { 2 } , \qquad \forall 1 \leq t \leq N _ { j _ { 0 } } .\tag{11}
$$

For $1 \leq r < N _ { j _ { 0 } } / 2$ , combining (11) with Lemma B.12(i) applied at $j = j _ { 0 }$ gives

$$
\Delta _ { N _ { j _ { 0 } } + r } = \left( f ( x _ { N _ { j _ { 0 } } + r } ) - f ( x _ { N _ { j _ { 0 } } } ) \right) + \Delta _ { N _ { j _ { 0 } } } \leq \frac { L R ^ { 2 } } { 2 } + \frac { K _ { h _ { 1 } } + K _ { h _ { 2 } } } { c _ { h } } \cdot R \sigma \frac { h _ { j _ { 0 } } } { \sqrt { N _ { j _ { 0 } } } } .
$$

The same constant dominates (11). Since the range $1 \leq t < 3 N _ { j _ { 0 } } / 2$ is finite and min $\{ 1 , h ( n ) / \sqrt { n } \} > 0$ on it, dividing by that minimum gives the claim. □

## C Proofs and auxiliary results for necessity

## C.1 Fixed-time analysis

This subsection proves Lemmas 4.3 and 4.4. For $a > 0$ and $0 < \lambda \leq L$ , define

$$
\varphi _ { \lambda , a } ( y ) : = \frac { \lambda } { 2 } \left( \sqrt { a ^ { 2 } + y ^ { 2 } } - a \right) ^ { 2 } , \quad q _ { \lambda } ( y ) : = \frac { \lambda } { 2 } y ^ { 2 } .
$$

Lemma C.1 (Geometry of the flat-active family). The function $\varphi _ { \lambda , a }$ is real analytic and convex on R. Moreover, it is λ-smooth, and hence L-smooth. Its unique minimizer is $y ^ { * } = 0 ;$ however, it is not strongly convex on any neighborhood of $y ^ { \ast }$ and does not satisfy a PL inequality with any positive constant. Furthermore, for every $y \in \mathbb R$

$$
\left| \varphi _ { \lambda , a } ^ { \prime } ( y ) - \lambda y \right| \leq \lambda a , \quad 0 \leq q _ { \lambda } ( y ) - \varphi _ { \lambda , a } ( y ) \leq \lambda a | y | .
$$

Proof. Let $s : = \sqrt { a ^ { 2 } + y ^ { 2 } }$ . Direct diferentiation gives

$$
\varphi _ { \lambda , a } ^ { \prime } ( y ) = \lambda y \left( 1 - \frac { a } { s } \right) , \quad \varphi _ { \lambda , a } ^ { \prime \prime } ( y ) = \lambda \left( 1 - \frac { a ^ { 3 } } { ( a ^ { 2 } + y ^ { 2 } ) ^ { 3 / 2 } } \right) \in [ 0 , \lambda ] .
$$

Thus, $\varphi _ { \lambda , a }$ is convex and λ-smooth, and hence also L-smooth. It is nonnegative and vanishes only at zero, so zero is its unique minimizer. Since $\varphi _ { \lambda , a } ^ { \prime \prime } ( 0 ) = 0$ , it is not strongly convex on any neighborhood of zero. For $y \ne 0$

$$
\frac { \varphi _ { \lambda , a } ^ { \prime } ( y ) ^ { 2 } } { 2 \varphi _ { \lambda , a } ( y ) } = \lambda \frac { y ^ { 2 } } { a ^ { 2 } + y ^ { 2 } }  0 , \mathrm { a s } y  0 ,
$$

which rules out a PL inequality with a positive constant. The identity

$$
{ \sqrt { a ^ { 2 } + y ^ { 2 } } } - a = { \frac { y ^ { 2 } } { { \sqrt { a ^ { 2 } + y ^ { 2 } } } + a } }
$$

further gives

$$
\varphi _ { \lambda , a } ( y ) = { \frac { \lambda } { 8 a ^ { 2 } } } y ^ { 4 } + { \mathcal O } ( y ^ { 6 } ) , \quad \mathrm { a s ~ } y \to 0 .
$$

Therefore, the objective is quartic-flat at its minimizer.

Finally,

$$
| \varphi _ { \lambda , a } ^ { \prime } ( y ) - \lambda y | = \frac { \lambda a | y | } { \sqrt { a ^ { 2 } + y ^ { 2 } } } \leq \lambda a .
$$

Using $y ^ { 2 } = ( s - a ) ( s + a )$ , we also obtain

$$
q _ { \lambda } ( y ) - \varphi _ { \lambda , a } ( y ) = \lambda a ( s - a ) \in [ 0 , \lambda a | y | ] .
$$

We next show that, over a fixed number of iterations, the recursion for $\varphi _ { \lambda , a }$ converges pathwise to the corresponding quadratic recursion as $a \downarrow 0$

Lemma C.2 (Finite-horizon flat transfer). Fix $n \in \mathbb { N } _ { + }$ and suppose $\lambda S _ { n } \le 1 / 2$ . From the same initial point and with the same noise sequence, define

$$
\begin{array} { r l } & { Y _ { t + 1 } ^ { ( a ) } = Y _ { t } ^ { ( a ) } - \eta _ { t } \left( \varphi _ { \lambda , a } ^ { \prime } ( Y _ { t } ^ { ( a ) } ) + \zeta _ { t } \right) , } \\ & { Y _ { t + 1 } ^ { ( 0 ) } = Y _ { t } ^ { ( 0 ) } - \eta _ { t } \left( \lambda Y _ { t } ^ { ( 0 ) } + \zeta _ { t } \right) . } \end{array}
$$

Then, pathwise,

$$
\operatorname* { m a x } _ { 0 \leq t \leq n } | Y _ { t } ^ { ( a ) } - Y _ { t } ^ { ( 0 ) } | \leq \lambda a S _ { n } \leq \frac { a } { 2 } .
$$

In particular,

$$
\varphi _ { \lambda , a } ( Y _ { n } ^ { ( a ) } )  q _ { \lambda } ( Y _ { n } ^ { ( 0 ) } ) , \quad a s \ a \downarrow 0 .
$$

Proof. Let $D _ { t } : = Y _ { t } ^ { ( a ) } - Y _ { t } ^ { ( 0 ) }$ and $r _ { a } ( y ) : = \varphi _ { \lambda , a } ^ { \prime } ( y ) - \lambda y$ . The coupled recursions imply

$$
\begin{array} { r } { D _ { t + 1 } = ( 1 - \lambda \eta _ { t } ) D _ { t } - \eta _ { t } r _ { a } ( Y _ { t } ^ { ( a ) } ) . } \end{array}
$$

Since $0 \leq \lambda \eta _ { t } \leq \lambda S _ { n } \leq 1 / 2$ , Lemma C.1 gives

$$
\begin{array} { r } { | D _ { t + 1 } | \leq ( 1 - \lambda \eta _ { t } ) | D _ { t } | + \lambda a \eta _ { t } . } \end{array}
$$

Starting from $D _ { 0 } = 0$ , for every $1 \leq k \leq n ,$ , we have

$$
| D _ { k } | \leq \lambda a \sum _ { i = 0 } ^ { k - 1 } \eta _ { i } \prod _ { r = i + 1 } ^ { k - 1 } \left( 1 - \lambda \eta _ { r } \right) \leq \lambda a S _ { n } \leq \frac { a } { 2 } .
$$

This proves the uniform pathwise bound. In particular, $Y _ { n } ^ { ( a ) }  Y _ { n } ^ { ( 0 ) }$ as $a \downarrow 0$ . Lemma C.1 and the continuity of $q _ { \lambda }$ yield

$$
\begin{array} { r l r } & { } & { | \varphi _ { \lambda , a } ( Y _ { n } ^ { ( a ) } ) - q _ { \lambda } ( Y _ { n } ^ { ( 0 ) } ) | \le | \varphi _ { \lambda , a } ( Y _ { n } ^ { ( a ) } ) - q _ { \lambda } ( Y _ { n } ^ { ( a ) } ) | + | q _ { \lambda } ( Y _ { n } ^ { ( a ) } ) - q _ { \lambda } ( Y _ { n } ^ { ( 0 ) } ) | } \\ & { } & { \le \lambda a | Y _ { n } ^ { ( a ) } | + \displaystyle \frac { \lambda } { 2 } | Y _ { n } ^ { ( a ) } - Y _ { n } ^ { ( 0 ) } | ( | Y _ { n } ^ { ( a ) } | + | Y _ { n } ^ { ( 0 ) } | ) \to 0 . } \end{array}
$$

The preceding pathwise convergence also transfers any marginal coverage inequality from the quarticflat family to the limiting quadratic model.

Lemma C.3 (Transfer of marginal coverage). Fix $n \in \mathbb { N } _ { + }$ , an initial point, and a noise distribution, and suppose $\lambda S _ { n } \le 1 / 2$ . If

$$
\begin{array} { r } { \mathbb { P } \left( \varphi _ { \lambda , a } ( Y _ { n } ^ { ( a ) } ) \leq \varepsilon _ { \alpha } ( n ) \right) \geq 1 - \alpha , \quad \forall a > 0 , } \end{array}
$$

then

$$
\begin{array} { r } { \mathbb { P } \left( q _ { \lambda } ( Y _ { n } ^ { ( 0 ) } ) \leq \varepsilon _ { \alpha } ( n ) \right) \geq 1 - \alpha . } \end{array}
$$

Proof. Lemma C.2 gives almost-sure convergence of the terminal objective values. Since $\bigl ( - \infty , \varepsilon _ { \alpha } ( n ) \bigr ]$ is closed, the closed-set part of the Portmanteau theorem gives

$$
\begin{array} { r } { \mathbb { P } \left( q _ { \lambda } ( Y _ { n } ^ { ( 0 ) } ) \leq \varepsilon _ { \alpha } ( n ) \right) \geq \operatorname* { l i m s u p } _ { a \downarrow 0 } \mathbb { P } \left( \varphi _ { \lambda , a } ( Y _ { n } ^ { ( a ) } ) \leq \varepsilon _ { \alpha } ( n ) \right) \geq 1 - \alpha . } \end{array}
$$

We are now ready to prove the two fixed-horizon restrictions stated in the main text.

Proof of Lemma 4.3. If $S _ { n } ~ = ~ 0$ , set $\lambda _ { n } : = L / 2$ . Otherwise, set $\lambda _ { n } : = \operatorname* { m i n } \{ L , 1 / S _ { n } \} / 2$ . In both cases, $0 < \lambda _ { n } \leq L$ and $\lambda _ { n } S _ { n } \leq 1 / 2$ . Start the quartic-flat recursion at $Y _ { 0 } ^ { ( a ) } = R$ and use zero noise. The limiting quadratic recursion satisfies

$$
Y _ { n } ^ { ( 0 ) } = R \prod _ { t < n } ( 1 - \lambda _ { n } \eta _ { t } ) .
$$

Every factor lies in [0, 1]. Therefore, the product inequality $\textstyle \prod _ { i } ( 1 - u _ { i } ) \geq 1 - \sum _ { i } u _ { i }$ gives

$$
| Y _ { n } ^ { ( 0 ) } | \geq R ( 1 - \lambda _ { n } S _ { n } ) \geq \frac { R } { 2 } .
$$

The zero-noise oracle is admissible for every $a > 0$ . Time-uniform validity thus implies the marginal premise of Lemma C.3, and hence

$$
\varepsilon _ { \alpha } ( n ) \geq q _ { \lambda _ { n } } ( Y _ { n } ^ { ( 0 ) } ) \geq \frac { \lambda _ { n } R ^ { 2 } } { 8 } = \frac { R ^ { 2 } } { 1 6 } \operatorname* { m i n } \left\{ L , \frac { 1 } { S _ { n } } \right\} .
$$

Proof of Lemma $4 { \cdot } 4$ . Set $\lambda _ { n } : = 1 / ( 2 S _ { n } )$ , start from $Y _ { 0 } ^ { ( a ) } = 0$ , and take independent noises $\zeta _ { t } \sim \mathcal N ( 0 , \sigma ^ { 2 } / 4 )$ This oracle is admissible because

$$
\mathbb { E } \left[ \exp \left( \frac { \zeta _ { t } ^ { 2 } } { \sigma ^ { 2 } } \right) \right] = \sqrt { 2 } < \mathrm { e } .
$$

Since $S _ { n } \geq 1 / L ,$ we have $\lambda _ { n } \ \leq \ L / 2$ , while $\lambda _ { n } S _ { n } = 1 / 2$ . Lemma C.3 therefore applies. The limiting quadratic recursion can be unrolled as

$$
Y _ { n } ^ { ( 0 ) } = - \sum _ { t < n } \eta _ { t } \zeta _ { t } \prod _ { s = t + 1 } ^ { n - 1 } ( 1 - \lambda _ { n } \eta _ { s } ) .
$$

For every $t < n$ , the product inequality used above gives

$$
\prod _ { s = t + 1 } ^ { n - 1 } \left( 1 - \lambda _ { n } \eta _ { s } \right) \geq 1 - \lambda _ { n } \sum _ { s = t + 1 } ^ { n - 1 } \eta _ { s } \geq \frac { 1 } { 2 } .
$$

Thus, $Y _ { n } ^ { ( 0 ) }$ is centered Gaussian with

$$
\mathrm { V a r } ( Y _ { n } ^ { ( 0 ) } ) = { \frac { \sigma ^ { 2 } } { 4 } } \sum _ { t < n } \eta _ { t } ^ { 2 } \prod _ { s = t + 1 } ^ { n - 1 } ( 1 - \lambda _ { n } \eta _ { s } ) ^ { 2 } \geq { \frac { \sigma ^ { 2 } Q _ { n } } { 1 6 } } .
$$

Marginal coverage of $q _ { \lambda _ { n } } ( Y _ { n } ^ { ( 0 ) } )$ is equivalent to

$$
\mathbb { P } \bigg ( | Y _ { n } ^ { ( 0 ) } | \leq \sqrt { \frac { 2 \varepsilon _ { \alpha } ( n ) } { \lambda _ { n } } } \bigg ) \geq 1 - \alpha .
$$

By the definition $z _ { \alpha } = \Phi ^ { - 1 } ( 1 - \alpha / 2 )$ , the two-sided Gaussian quantile requires

$$
\frac { 2 \varepsilon _ { \alpha } ( n ) } { \lambda _ { n } } \geq z _ { \alpha } ^ { 2 } \operatorname { V a r } ( Y _ { n } ^ { ( 0 ) } ) \geq \frac { z _ { \alpha } ^ { 2 } \sigma ^ { 2 } Q _ { n } } { 1 6 } .
$$

Substituting $\lambda _ { n } = 1 / ( 2 S _ { n } )$ proves the result.

Corollary 4.5 is a direct result from Lemma 4.3 and 4.4.

Proof of Corollary 4.5. Since $h ( n ) = o ( { \sqrt { n } } )$ , the boundary $\varepsilon _ { \alpha } ( n ) \to 0$ as $n \to \infty$ . Therefore, by Lemma 4.3, for suficiently large n, we have $S _ { n } \geq 1 / L$ , and additionally,

$$
{ \frac { R ^ { 2 } } { 1 6 } } { \frac { 1 } { S _ { n } } } \leq B { \frac { h ( n ) } { \sqrt { n } } } \implies S _ { n } \geq { \frac { R ^ { 2 } } { 1 6 B } } { \frac { \sqrt { n } } { h ( n ) } } \to \infty .
$$

Thus, for suficiently large $j ,$ Lemma 4.4 applies, yielding

$$
\frac { z _ { \alpha } ^ { 2 } \sigma ^ { 2 } } { 6 4 } \frac { Q _ { j } } { S _ { N _ { j } } } \leq B \frac { h _ { j } } { \sqrt { N _ { j } } } \implies p _ { j } \geq \frac { z _ { \alpha } ^ { 2 } \sigma ^ { 2 } } { 6 4 B } \frac { Q _ { j } } { h _ { j } } .
$$

## C.2 Preliminaries for the fixed Gaussian instance

As mentioned in Section 4.2, the instance is given by

$$
\phi ( y ) : = \frac { L } { 4 } ( \sqrt { R ^ { 2 } + y ^ { 2 } } - R ) ^ { 2 } = \varphi _ { L / 2 , R } ( y ) .
$$

The oracle noise is Gaussian with variance $\nu ^ { 2 } : = \vartheta \sigma ^ { 2 }$ , where $\vartheta : = ( 1 - \mathrm { e } ^ { - 2 } ) / 2$ . The recursion is initialized at the minimizer and uses independent $\zeta _ { t } \sim \mathcal { N } ( 0 , \nu ^ { 2 } )$ as the oracle noise:

$$
Y _ { t + 1 } = Y _ { t } - \eta _ { t } ( \phi ^ { \prime } ( Y _ { t } ) + \zeta _ { t } ) , \quad Y _ { 0 } = 0 .
$$

We now give some properties of this instance that will be used in the necessity proof.

Lemma C.4. The function $\phi$ is convex, L-smooth, and non-PL. It is quartic-flat at the minimizer, and its derivative satisfies

$$
| \phi ^ { \prime } ( y ) | \leq \frac { L | y | ^ { 3 } } { 4 R ^ { 2 } } , \quad \forall y \in \mathbb { R } .
$$

Proof. The first three properties follow from Lemma C.1. The derivative is given by

$$
\phi ^ { \prime } ( y ) = \frac { L y ^ { 3 } } { 2 \sqrt { R ^ { 2 } + y ^ { 2 } } ( \sqrt { R ^ { 2 } + y ^ { 2 } } + R ) } .
$$

Since $\sqrt { R ^ { 2 } + y ^ { 2 } } ( \sqrt { R ^ { 2 } + y ^ { 2 } } + R ) \geq 2 R ^ { 2 }$ , we have

$$
| \phi ^ { \prime } ( y ) | \leq \frac { L | y | ^ { 3 } } { 4 R ^ { 2 } } .
$$

This instance forces the stepsize to vanish asymptotically, as stated in the following lemma.

Lemma C.5. Under the assumptions of Theorem $4 . 1 ,$ the stepsize sequence $\{ \eta _ { t } \}$ must satisfy $\eta _ { t }  0$ as $t \to \infty$

Proof. For every $u \geq 0 , \phi ( y ) \leq u$ is equivalent to

$$
| y | ^ { 2 } \leq r ^ { 2 } ( u ) : = 4 R \sqrt { \frac { u } { L } } + \frac { 4 u } { L } .
$$

It is easy to see that $r ^ { 2 } ( u ) \to 0 { \mathrm { ~ a s ~ } } u \to 0 ^ { + }$ . Since $h ( n ) / \sqrt { n }  0$ , we have $\varepsilon _ { \alpha } ( n ) \to 0$ as $n \to \infty$ . Hence, $r ^ { 2 } ( { \varepsilon } _ { \alpha } ( n ) ) \to 0$ as $n  \infty$ . Conditional on the past, $Y _ { t + 1 }$ is a translate of $\mathcal { N } ( 0 , \nu ^ { 2 } \eta _ { t } ^ { 2 } )$ . When $\eta _ { t } > 0$ , its conditional density is bounded by $( \sqrt { 2 \pi } \nu \eta _ { t } ) ^ { - 1 }$ , regardless of the conditional mean. Integrating this bound over $[ - r ( \varepsilon _ { \alpha } ( t + 1 ) ) , r ( \varepsilon _ { \alpha } ( t + 1 ) ) ]$ yields

$$
1 - \alpha \le \mathbb { P } \left( | Y _ { t + 1 } | \le r ( \varepsilon _ { \alpha } ( t + 1 ) ) \right) \le \sqrt { \frac { 2 } { \pi } } \frac { r ( \varepsilon _ { \alpha } ( t + 1 ) ) } { \nu \eta _ { t } } .
$$

Thus every positive $\eta _ { t }$ is bounded by a constant multiple of a quantity tending to zero. Zero steps already satisfy the same conclusion, so $\eta _ { t } \to 0$ □

In Section 4.2, we define $N _ { j } = 2 ^ { j } , h _ { j } = h ( N _ { j } )$ , and

$$
\hat { h } _ { j } : = \operatorname* { m a x } _ { \tiny N _ { j } \leq n \leq N _ { j + 1 } } h ( n ) \sqrt { \frac { N _ { j } } { n } } .
$$

We also define $\bar { \varepsilon } _ { j } = B \hat { h } _ { j } / \sqrt { N _ { j } }$ and $\beta _ { j } ^ { 2 } : = r ^ { 2 } ( \bar { \varepsilon } _ { j } )$ . Now we will give a lower bound and an upper bound for $\beta _ { j } ^ { 4 }$

Lemma C.6. For suficiently large $j \in \mathbb { N } _ { + }$ , we have

$$
\frac { 1 6 R ^ { 2 } \bar { \varepsilon } _ { j } } { L } \leq \beta _ { j } ^ { 4 } \leq \frac { 6 4 R ^ { 2 } \bar { \varepsilon } _ { j } } { L } .
$$

Proof. Since $r ^ { 2 } ( \varepsilon _ { j } ) \geq 4 R \sqrt { \varepsilon _ { j } / L }$ , the lower bound is immediate. For the upper bound, we notice that

$$
\bar { \varepsilon } _ { j } = B \frac { \hat { h } _ { j } } { \sqrt { N _ { j } } } \leq B \frac { \hat { h } _ { j + 1 } } { \sqrt { N _ { j } } } = \sqrt { 2 } B \frac { h _ { j + 1 } } { \sqrt { N _ { j + 1 } } }  0 , \quad \mathrm { a s ~ } j  \infty .
$$

Therefore, for suficiently large $j ,$ we have $\bar { \varepsilon } _ { j } \leq L R ^ { 2 }$ and

$$
r ^ { 2 } ( \bar { \varepsilon } _ { j } ) = 4 R \sqrt { \frac { \bar { \varepsilon } _ { j } } { L } } + \frac { 4 \bar { \varepsilon } _ { j } } { L } \leq 8 R \sqrt { \frac { \bar { \varepsilon } _ { j } } { L } } .
$$

Hence, the desired upper bound holds for suficiently large j.

## C.3 Greedy step-mass blocks

Now we define the accumulated stepsize over the j-th dyadic epoch as

$$
H _ { j } : = \sum _ { t = N _ { j } } ^ { N _ { j + 1 } - 1 } \eta _ { t } .
$$

We consider dividing the j-th epoch into several sub-blocks: let $\tau _ { j } > 0$ be a tolerance parameter, and define $e _ { k }$ as the smallest integer such that

$$
H _ { j } ^ { ( k ) } : = \sum _ { { t = e _ { k - 1 } } } ^ { { e _ { k } - 1 } } \eta _ { t } \ge \tau _ { j } , \quad e _ { 0 } : = N _ { j } .
$$

We stop when the remaining step mass is less than $\tau _ { j } .$ . Let K be the number of complete sub-blocks and $m _ { k } : = e _ { k } - e _ { k - 1 }$ be the length of the k-th sub-block. Then we have the following lemma.

Lemma C.7. For suficiently large $j \in \mathbb { N } _ { + }$ , if max $_ { \cdot { N _ { j } } \leq t < { N _ { j + 1 } } } \eta _ { t } \leq \tau _ { j } / 4$ , we have

$$
\tau _ { j } \leq H _ { j } ^ { ( k ) } \leq \frac { 5 } { 4 } \tau _ { j } , \quad \sum _ { k = 1 } ^ { K } m _ { k } \leq N _ { j } .
$$

Note that $\begin{array} { r } { H _ { j } = \sum _ { t = N _ { j } } ^ { N _ { j + 1 } - 1 } \eta _ { t } } \end{array}$ . We also have

$$
K \ge \frac { 4 } { 5 } \cdot \frac { H _ { j } } { \tau _ { j } } - \frac { 4 } { 5 } .
$$

In particular, if $H _ { j } \geq 2 \tau _ { j }$ , then $K \ge ( 2 H _ { j } ) / ( 5 \tau _ { j } )$

Proof. $\begin{array} { r } { \sum _ { k = 1 } ^ { K } m _ { k } \leq N _ { j } } \end{array}$ is trivial since the sub-blocks are disjoint and contained in the j-th epoch. By the definition of $z _ { k } , H _ { j } ^ { ( k ) } \ge \tau _ { j }$ for each k. Since $\eta _ { t }  0 \mathrm { a s } t  \infty$ , we have $\eta _ { t } \leq \tau _ { j } / 4$ for suficiently large j and $t \in [ N _ { j } , N _ { j + 1 } )$ . Therefore, the minimality of $e _ { k }$ implies that the stepsize mass before the last step in the k-th sub-block is less than $\tau _ { j }$ . Meanwhile, the last step contributes at most $\tau _ { j } / 4$ . Hence, $H _ { j } ^ { ( k ) } \leq 5 \tau _ { j } / 4$ Finally, since

$$
\sum _ { k = 1 } ^ { K } H _ { j } ^ { ( k ) } \leq H _ { j } \leq \sum _ { k = 1 } ^ { K } H _ { j } ^ { ( k ) } + \tau _ { j } ,
$$

we have

$$
H _ { j } \le \frac { 5 } { 4 } K \tau _ { j } + \tau _ { j } \implies K \ge \frac { 4 } { 5 } \cdot \frac { H _ { j } } { \tau _ { j } } - \frac { 4 } { 5 } .
$$

If $H _ { j } \geq 2 \tau _ { j }$ , then $K \ge ( 2 H _ { j } ) / ( 5 \tau _ { j } )$

Now we set $\kappa _ { 0 } = 3 2 / 5$ , and $\tau _ { j } : = \kappa _ { 0 } R ^ { 2 } / ( L \beta _ { j } ^ { 2 } )$ . Since $\beta _ { j } \to 0$ , we have $\tau _ { j } \to \infty$ . Together with Lemma C.5, this guarantees max $N _ { j } \le t < N _ { j + 1 } \eta _ { t } \le \tau _ { j } / 4$ for all suficiently large $j .$

Lemma C.8. For a suficiently large $j \in \mathbb N _ { + }$ with $H _ { j } / \tau _ { j } \ \geq \ 2$ , we define $\begin{array} { r } { G _ { k } : = \sum _ { t = e _ { k - 1 } } ^ { e _ { k } - 1 } \eta _ { t } \zeta _ { t } , V _ { k } : = } \end{array}$ $\mathrm { V a r } ( G _ { k } )$ . Then the variables $G _ { 1 } , \dots , G _ { K }$ are independent centered Gaussian, and

$$
\mathcal { C } \subset \bigcap _ { k = 1 } ^ { K } \{ | G _ { k } | \leq 4 \beta _ { j } \} .
$$

Moreover, we have

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { \beta _ { j } ^ { 2 } } { V _ { k } } \le \frac { 2 5 } { \vartheta } B \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { \sigma ^ { 2 } H _ { j } } .
$$

Proof. Independence and zero mean are trivial since the $G _ { k } { } ^ { \ ' } \mathrm { s }$ are linear combinations of disjoint sets of independent centered Gaussian variables. Notice that

$$
Y _ { e _ { k } } - Y _ { e _ { k - 1 } } = - \sum _ { t = e _ { k - 1 } } ^ { e _ { k } - 1 } \eta _ { t } ( \phi ^ { \prime } ( Y _ { t } ) + \zeta _ { t } ) = - \sum _ { t = e _ { k - 1 } } ^ { e _ { k } - 1 } \eta _ { t } \phi ^ { \prime } ( Y _ { t } ) - G _ { k } .
$$

Now we control the deterministic term. By Lemma C.4, on the event ${ \mathcal { C } } ,$ we have

$$
\biggl | \sum _ { t = e _ { k - 1 } } ^ { e _ { k } - 1 } \eta _ { t } \phi ^ { \prime } ( Y _ { t } ) \biggr | \leq \sum _ { t = e _ { k - 1 } } ^ { e _ { k } - 1 } \eta _ { t } | \phi ^ { \prime } ( Y _ { t } ) | \leq \frac { L \beta _ { j } ^ { 3 } } { 4 R ^ { 2 } } H _ { j } ^ { ( k ) } \leq \frac { 5 } { 1 6 } \kappa _ { 0 } \beta _ { j } = 2 \beta _ { j } .
$$

Since $| Y _ { e _ { k - 1 } } | , | Y _ { e _ { k } } | \le \beta _ { j }$ on the event ${ \mathcal { C } } ,$ we have

$$
\vert G _ { k } \vert \leq \vert Y _ { e _ { k } } - Y _ { e _ { k - 1 } } \vert + \Big \vert \sum _ { t = e _ { k - 1 } } ^ { e _ { k } - 1 } \eta _ { t } \phi ^ { \prime } ( Y _ { t } ) \Big \vert \leq 4 \beta _ { j } .
$$

Hence, $\mathcal { C } \subset \bigcap _ { k = 1 } ^ { K } \{ | G _ { k } | \leq 4 \beta _ { j } \}$

Finally, by Cauchy–Schwarz, we have

$$
V _ { k } = \nu ^ { 2 } \sum _ { t = e _ { k - 1 } } ^ { e _ { k } - 1 } \eta _ { t } ^ { 2 } \ge \frac { \nu ^ { 2 } ( H _ { j } ^ { ( k ) } ) ^ { 2 } } { m _ { k } } \ge \frac { \vartheta \sigma ^ { 2 } \tau _ { j } ^ { 2 } } { m _ { k } } .
$$

Using the fact that $\begin{array} { r } { \sum _ { k = 1 } ^ { K } m _ { k } \leq N _ { j } } \end{array}$ and $K \ge ( 2 H _ { j } ) / ( 5 \tau _ { j } )$ , we get

$$
\begin{array} { c } { \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { \beta _ { j } ^ { 2 } } { V _ { k } } \leq \frac { \beta _ { j } ^ { 2 } } { K } \sum _ { k = 1 } ^ { K } \frac { m _ { k } } { \vartheta \sigma ^ { 2 } \tau _ { j } ^ { 2 } } \leq \frac { \beta _ { j } ^ { 2 } N _ { j } } { \vartheta \sigma ^ { 2 } K \tau _ { j } ^ { 2 } } } \\ { \displaystyle \leq \frac { 5 \tau _ { j } \beta _ { j } ^ { 2 } N _ { j } } { 2 \vartheta \sigma ^ { 2 } H _ { j } \tau _ { j } ^ { 2 } } = \frac { 5 \beta _ { j } ^ { 2 } N _ { j } } { 2 \vartheta \sigma ^ { 2 } H _ { j } \tau _ { j } } } \\ { \displaystyle = \frac { 5 L N _ { j } } { 2 \vartheta \sigma ^ { 2 } H _ { j } } \cdot \frac { 1 } { \kappa _ { 0 } R ^ { 2 } } \cdot \beta _ { j } ^ { 4 } } \end{array}
$$

Now use the upper bound in Lemma C.6 to conclude

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { \beta _ { j } ^ { 2 } } { V _ { k } } \le \frac { 5 L N _ { j } } { 2 \vartheta \sigma ^ { 2 } H _ { j } } \cdot \frac { 1 } { \kappa _ { 0 } R ^ { 2 } } \cdot \frac { 6 4 R ^ { 2 } \bar { \varepsilon } _ { j } } { L } = \frac { 2 5 } { \vartheta } B \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { \sigma ^ { 2 } H _ { j } } .
$$

Now we are ready to prove Lemma 4.6.

Proof of Lemma 4.6. We use contradiction. Suppose

$$
H _ { j } > C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { j } .
$$

Then by the definition of $\tau _ { j }$ and Lemma C.6, we have

$$
\begin{array} { r l r } {  { \frac { H _ { j } } { \tau _ { j } } = H _ { j } \cdot \frac { L \beta _ { j } ^ { 2 } } { \kappa _ { 0 } R ^ { 2 } } } } \\ & { } & { \geq C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { j } \cdot \frac { L \beta _ { j } ^ { 2 } } { \kappa _ { 0 } R ^ { 2 } } } \\ & { } & { \geq C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { j } \cdot \frac { L } { \kappa _ { 0 } R ^ { 2 } } \cdot 4 R \sqrt { \frac { \bar { \varepsilon } _ { j } } { L } } } \\ & { } & { = \frac { 4 C _ { b } B ^ { 3 / 2 } } { \kappa _ { 0 } } \frac { L ^ { 1 / 2 } } { R \sigma ^ { 2 } } \frac { \hat { h } _ { j } ^ { 3 / 2 } N _ { j } ^ { 1 / 4 } } { j }  \infty , \quad \mathrm { a s ~ } j  \infty . } \end{array}
$$

Then for suficiently large $j ,$ we have $H _ { j } / \tau _ { j } \geq 2$ . Then Lemma C.8 applies, and we have

$$
\mathcal { C } \subset \bigcap _ { k = 1 } ^ { K } \{ | G _ { k } | \leq 4 \beta _ { j } \} ,
$$

and

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { ( 4 \beta _ { j } ) ^ { 2 } } { V _ { k } } \le \frac { 4 0 0 } { \vartheta } B \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { \sigma ^ { 2 } H _ { j } } .
$$

By Markov’s inequality, this implies that there exists at least $K / 2$ sub-blocks with

$$
\frac { ( 4 \beta _ { j } ) ^ { 2 } } { V _ { k } } \le \frac { 8 0 0 } { \vartheta } B \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { \sigma ^ { 2 } H _ { j } } \le \gamma _ { 0 } j , \quad j \in \mathcal { G } _ { j } , | \mathcal { G } _ { j } | \ge K / 2 .
$$

where the last inequality follows from the contradiction assumption and the definition of $C _ { b }$

Also, Lemma C.7 gives $K \ge ( 2 H _ { j } ) / ( 5 \tau _ { j } )$ . Therefore, we have

$$
K \ge \frac { C _ { b } B ^ { 3 / 2 } } { 4 } \frac { L ^ { 1 / 2 } } { R \sigma ^ { 2 } } \frac { \hat { h } _ { j } ^ { 3 / 2 } N _ { j } ^ { 1 / 4 } } { j } .
$$

Since

$$
{ \mathcal { C } } \subset \bigcap _ { k = 1 } ^ { K } \{ | G _ { k } | \leq 4 \beta _ { j } \} \subset \bigcap _ { k \in { \mathcal { G } } _ { j } } \{ | G _ { k } | \leq 4 \beta _ { j } \} ,
$$

we have

$$
\begin{array} { r l } { \mathbb { P } ( \mathcal { C } ) \leq \underset { k \in \mathcal { G } _ { j } } { \prod \mathcal { F } ( \vert G _ { k } \vert \leq 4 \beta _ { 3 } ) } } \\ & { = \underset { k \in \mathcal { G } _ { j } } { \prod \mathcal { F } \left( \frac { \vert G _ { k } \vert } { \sqrt { V _ { k } } } \leq \frac { 4 \beta _ { j } } { \sqrt { V _ { k } } } \right) } } \\ & { \leq \underset { k \in \mathcal { G } _ { j } } { \prod \mathcal { F } \left( \vert Z \vert \leq \sqrt { \gamma _ { 0 } j } \right) } } \\ & { = \underset { k \in \mathcal { G } _ { j } } { \prod \mathcal { F } ( 1 - 2 \Phi ( - \sqrt { \gamma _ { 0 } j } ) ) } } \\ & { \leq \exp ( - ( - \vert G _ { j } \vert \cdot 2 \Phi ( - \sqrt { \gamma _ { 0 } j } ) ) ) } \\ & { \leq \exp \left( - K \cdot \frac { \sqrt { \gamma _ { 0 } j } } { 1 + N ^ { j } } \frac { 1 } { \sqrt { 1 + N ^ { j } } } \frac { 1 } { \sqrt { 2 } \pi } \mathrm { e } ^ { - \gamma _ { 0 } j / 2 } \right) , } \end{array}
$$

where the last inequality uses the Mills lower bound

$$
\Phi ( - x ) \ge \frac { x } { 1 + x ^ { 2 } } \frac { 1 } { \sqrt { 2 \pi } } \mathrm { e } ^ { - x ^ { 2 } / 2 } , \quad x > 0 .
$$

Now substituting the lower bound of K gives

$$
\mathbb { P } ( \mathcal { C } ) \le \exp \bigg ( - \frac { C _ { b } B ^ { 3 / 2 } } { 4 \sqrt { 2 } \pi } \frac { L ^ { 1 / 2 } } { R \sigma ^ { 2 } } \cdot \frac { \sqrt { \gamma _ { 0 } } \hat { h } _ { j } ^ { 3 / 2 } 2 ^ { j / 8 } } { j ^ { 1 / 2 } ( 1 + \gamma _ { 0 } j ) } \bigg )  0 , \quad \mathrm { a s ~ } j  \infty .
$$

This contradicts the assumption that $\mathbb { P } ( \mathcal { C } ) \geq 1 - \alpha$ . Hence, we must have

$$
H _ { j } \leq C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { \hat { h } _ { j } \sqrt { N _ { j } } } { j } ,
$$

for all suficiently large j.

## C.4 Dyadic sequence estimates

Our main focus here is to prove Lemma 4.7 and 4.8.

Proof of Lemma 4.7. Note that $H _ { j } = S _ { N _ { j + 1 } } - S _ { N _ { j } }$ . By Lemma 4.6, there exists $j _ { 0 } \in \mathbb { N } _ { + }$ such that

$$
\frac { H _ { j } } { \sqrt { N _ { j } } } = \sqrt { 2 } p _ { j + 1 } - p _ { j } \leq C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { \hat { h } _ { j } } { j } , \quad \forall j \geq j _ { 0 } .
$$

The above recursion implies that

$$
p _ { j } \leq 2 ^ { - ( j - j _ { 0 } ) / 2 } p _ { j _ { 0 } } + \sum _ { i = j _ { 0 } } ^ { j - 1 } 2 ^ { - ( j - i ) / 2 } C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { \hat { h } _ { i } } { i } .
$$

Note that for $j > i ,$ , we have $\hat { h } _ { i } \leq h _ { j }$ . Now we split the summation into two parts:

$$
\begin{array} { r l } & { p _ { j } \le 2 ^ { - ( j - j _ { 0 } ) / 2 } p _ { j _ { 0 } } + \displaystyle \sum _ { i = j _ { 0 } } ^ { \lfloor j / 2 \rfloor - 1 } 2 ^ { - ( j - i ) / 2 } C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { h _ { j } } { i } + \displaystyle \sum _ { i = \lfloor j / 2 \rfloor } ^ { j - 1 } 2 ^ { - ( j - i ) / 2 } C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { h _ { j } } { i } } \\ & { \quad \le 2 ^ { - ( j - j _ { 0 } ) / 2 } p _ { j _ { 0 } } + \displaystyle \frac { j } { j _ { 0 } } 2 ^ { - j / 4 } C _ { b } \frac { B } { \sigma ^ { 2 } } h _ { j } + C _ { b } \frac { B } { \sigma ^ { 2 } } \frac { 2 h _ { j } } { j ( 1 - 2 ^ { - 1 / 2 } ) } . } \end{array}
$$

It is straightforward to verify that $j 2 ^ { - j / 4 } = o ( 1 / j )$ . Thus, the first two terms can be absorbed into the last term multiplied by a constant. Therefore, there exists a constant $C _ { p }$ such that

$$
p _ { j } \le C _ { p } \frac { B } { \sigma ^ { 2 } } \frac { h _ { j } } { j } .
$$

By Corollary 4.5, we have

$$
\mathcal { Q } _ { j } \leq \frac { 6 4 B } { z _ { \alpha } ^ { 2 } \sigma ^ { 2 } } p _ { j } h _ { j } \leq C _ { q } \frac { B ^ { 2 } } { \sigma ^ { 4 } } \frac { h _ { j } ^ { 2 } } { j } ,
$$

where $C _ { q }$ is a constant.

Proof of Lemma $4 . 8 .$ Consider the recursion

$$
\mathcal { Q } _ { j + 1 } - \mathcal { Q } _ { j } = \sum _ { t = N _ { j } } ^ { N _ { j + 1 } - 1 } \eta _ { t } ^ { 2 } \ge \frac { H _ { j } ^ { 2 } } { N _ { j } } .
$$

Note that $H _ { j } / \sqrt { N _ { j } } = \sqrt { 2 } p _ { j + 1 } - p _ { j }$ . Then we have

$$
\begin{array} { r l } { \mathcal { Q } _ { J } - \mathcal { Q } _ { n } \frac { \mathcal { J } - 1 } { 2 } ( \sqrt { 2 } p _ { j + 1 } - p _ { j } ) ^ { 2 } } \\ & { \qquad \quad \ : = \frac { \mathcal { J } - 1 } { 2 } ( 2 p _ { j + 1 } ^ { 2 } + p _ { j } ^ { 2 } - 2 \sqrt { 2 } p _ { j } p _ { j - 1 } ) } \\ & { \qquad \quad \ : = \frac { \mathcal { J } - 1 } { 2 \sqrt { 2 } m } ( 2 p _ { j + 1 } ^ { 2 } + p _ { j } ^ { 2 } - 2 \sqrt { 2 } p _ { j } p _ { j } ) } \\ & { \qquad \quad \ : \geq \displaystyle \sum _ { j = m } ^ { J - 1 } ( 2 - \sqrt { 2 } ) p _ { j + 1 } ^ { 2 } + ( 1 - \sqrt { 2 } ) p _ { j } ^ { 2 } ) } \\ & { \qquad \quad = ( 2 - \sqrt { 2 } ) p _ { j } ^ { 2 } + ( 1 - \sqrt { 2 } ) p _ { j } ^ { 2 } + ( \sqrt { 2 } - 1 ) ^ { 2 } \displaystyle \sum _ { i = m + 1 } ^ { J - 1 } p _ { i } ^ { 2 } } \\ & { \qquad \quad \ : \geq ( 1 - \sqrt { 2 } ) p _ { m } ^ { 2 } + ( \sqrt { 2 } - 1 ) ^ { 2 } \displaystyle \sum _ { j = m + 1 } ^ { J - 1 } p _ { j } ^ { 2 } . } \end{array}
$$

Therefore, we have

$$
\mathcal { Q } _ { J } \geq ( 2 - \sqrt { 2 } ) \mathcal { Q } _ { m } + ( \sqrt { 2 } - 1 ) ^ { 2 } \sum _ { j = m + 1 } ^ { J - 1 } p _ { j } ^ { 2 } .
$$

Then by Corollary 4.5, we have

$$
\mathcal { Q } _ { J } \geq \left( 2 - \sqrt { 2 } \right) \mathcal { Q } _ { m } + ( \sqrt { 2 } - 1 ) ^ { 2 } \frac { z _ { \alpha } ^ { 4 } \sigma ^ { 4 } } { 2 ^ { 1 2 } B ^ { 2 } } \sum _ { j = m + 1 } ^ { J - 1 } \frac { \mathcal { Q } _ { j } ^ { 2 } } { h _ { j } ^ { 2 } } .
$$