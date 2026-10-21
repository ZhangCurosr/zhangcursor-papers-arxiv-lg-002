# Neural Closed-Loop Identification ofNonlinear Systems with Stability and Generalization Guarantees <sup>⋆</sup>

Laura Meroi <sup>a</sup>, Mahrokh G. Boroujeni <sup>a</sup>, Leonardo Massai <sup>a</sup>, Giancarlo Ferrari-Trecate <sup>a</sup>

<sup>a</sup>Institute of Mechanical Engineering, EPFL, Switzerland.

## Abstract

Neural networks are expressive models of nonlinear dynamics, but identifying such systems from closed-loop data raises two difficulties: the feedback loop correlates inputs with measurement noise, biasing standard open-loop estimators, and the identified model must be stabilized by the data-collection controller, so that its closed-loop behavior matches the true plant’s. We derive a complete parameterization of all nonlinear systems stabilized by a given controller, generalizing the classical dual-Youla parameterization to the nonlinear setting. Building on this characterization, we propose Internal Controller Identification (ICI), a computationally tractable framework for identifying potentially unstable systems while ensuring closed-loop stability by construction; in the linear case, our indirect method reduces to an open-loop identification problem, for which standard consistency results apply. The framework accommodates contextual information, letting a single certified model identify a family of systems, and supports a PAC-Bayesian training objective that, when data are scarce and prior knowledge is available, bounds the identified operator’s expected prediction loss. Experiments show that ICI identifies openloop unstable plants where uncertified baselines diverge, outperforms direct methods under weak excitation and feedback-correlated noise, generalizes zero-shot to environments unseen during training, and, with limited data, avoids the overfitting of empirical risk minimization while providing a non-vacuous generalization bound.

Key words: Closed-loop identification; Nonlinear system identification; Neural networks; Stability of nonlinear systems; Probabilistic and Bayesian methods for system identification.

## 1 Introduction

Neural networks are flexible, expressive models for learning nonlinear dynamical systems from input-output data [42,38]. However, open-loop data collection is often impractical or unsafe: impossible for unstable systems, which can’t be run without stabilizing feedback, and undesirable for stable ones that operate under feedback for safety, setpoint tracking, or disturbance rejection (e.g., drones or robotic manipulators). In these scenarios, data are gathered while the system operates in closed loop with a stabilizing controller, motivating closed-loop system identification [45,16,32,26]. Compared to the open-loop case, the central difficulty of this setting is the feedback-induced correlation between input signals and measurement noise [45,16]. Classical methods for closedloop identification are typically classified into direct and indirect approaches [43].

Direct methods estimate the system and noise model independently of the feedback; in the linear case they are consistent provided the noise model is accurate, an assumption often violated by the complex disturbances of real-world systems [45,16].

Indirect methods instead exploit knowledge of the controller and target the closed-loop mapping, eliminating the noise contribution at the plant input and reducing the problem to open-loop identification [45,16,31]; in the linear case this yields consistency even under a misspecified noise model, and it is the route we follow.

A third family, joint input-output methods, simultaneously estimates the plant and controller from data jointly modeled as the output of a white-noise-driven system [43]; however, these methods are largely confined to the linear setting, do not extend naturally to general nonlinear plants, and additionally exhibit theoretically unbounded variance in the small-noise regime [1].

For LTI systems, one commonly used framework for indirect identification is the dual-Youla parameterization. This formulation is constructed using coprime factorizations of the known controller and of the nominal plant stabilized by it, augmented by a Youla parameter that is optimized to minimize the discrepancy from the true system. Through this structure, the parameterization characterizes the complete set of all plants stabilized by the given controller [24]; as its name suggests, it is dual to the Youla-Kucera parameter-ˇ ization, which instead characterizes all controllers that stabilize a given plant [48]. A fundamental advantage of this approach is that any identified model is inherently stabilized by the controller, naturally preserving closed-loop stability properties observed in the true system [16]. Furthermore, in the linear case, the Youla parameter reduces to a stable rational transfer function; since the underlying function space is infinite-dimensional, it is truncated to a finite order in practice, yielding a tractable optimization.

Extensions to nonlinear plants under linear control exist (see [4] for a review, and [9,17] for coprime-factorization parameterizations) but face two obstacles: coprime factorizations exist only for subclasses of nonlinear systems and are hard to compute [4], and the parameterization requires optimizing over stable nonlinear operators and their inverses, which complicates the optimization, a difficulty also noted for Youla-based nonlinear controller learning [19]. Existing methods thus fall short of a computationally practical framework.

More recently, [32] proposed an alternative approach for closed-loop identification of nonlinear systems. While effective in practice, this method has two key limitations: (i) the closed-loop system of the learned model and controller may become unstable, and (ii) it assumes a predefined parametric form for the system, limiting flexibility. This leaves closedloop identification of expressive, nonlinear models a critical, yet not sufficiently explored, challenge.

A further challenge is that real-world systems rarely operate in a single environment: friction, payload, or terrain vary, so the data are generated by a family of related plants. Identification across operating conditions is classically addressed by linear parameter-varying (LPV) methods, also in closed loop [47,44], but restricted to LPV model classes without closed-loop stability guarantees; meta- and in-context learning estimate a single meta-model for a class of systems [15,37], again without certificates.

A distinct concern is finite-sample reliability: closed-loop data collection is costly, safety-critical, and biased by the controller, so guarantees from limited data are valuable. Such guarantees can be obtained through PAC-Bayes theory [35,2], a powerful tool for certifying highly parameterized models, applied beyond supervised learning to meta-learning [41] and optimal control [5]. PAC-Bayesian bounds exist for open-loop identification of stochastic LTI systems [12,14] and stable recurrent models [13]; their use in closed-loop identification, under strict closed-loop stability constraints on the learned model, remains unexplored.

Contributions. We propose a computationally tractable framework for closed-loop identification of nonlinear, potentially open-loop unstable systems while guaranteeing stability of the interconnection with the data-collection controller. In particular: (i) we introduce the Internal Controller Identification (ICI) framework, a complete parameterization of all and only the nonlinear systems stabilized by a given controller, which generalizes the dual-Youla parameterization, guarantees closed-loop stability by construction, and reduces to a consistent open-loop problem in the linear case; (ii) we derive finite-sample generalization guarantees, introducing an informative prior over the parameters of the learnable operator and a tractable PAC-Bayesian bound on its expected prediction loss, optimized with Stein Variational Gradient Descent (SVGD); (iii) we develop context-aware identification, where bounded context wrappers let a single model identify an entire family of environments and generalize to members never observed during training, with the closed-loop stability certificate intact; (iv) we conduct a systematic experimental evaluation covering open-loop unstable plants, weak excitation, feedback-correlated noise, and partial observation, against closed-loop baselines.

A preliminary version of this work appeared in the conference paper [7], which introduced the base ICI parameterization and its stability guarantee. The present paper adds the context-aware formulation (Section 3.3), the PAC-Bayesian generalization framework and its SVGD-based training (Section 5), the four-experiment study (Section 6), and complete proofs.

## 1.1 Notation

The set of all sequences $\mathbf { x } = ( x _ { 0 } , x _ { 1 } , x _ { 2 } , \ldots )$ , where $x _ { t } \in \mathbb { R } ^ { d _ { x } }$ $t \in \mathbb { N } .$ , is denoted as $\ell ^ { d _ { x } }$ . Moreover, x belongs to $\ell _ { p } ^ { d _ { x } } \subset \ell ^ { d _ { x } }$ with $\begin{array} { r } { p \geq 1 \mathrm { i f } \left. \mathbf { x } \right. _ { p } = \left( \sum _ { t = 0 } ^ { \infty } | x _ { t } | ^ { p } \right) ^ { \frac { 1 } { p } } < \infty } \end{array}$ and $\mathbf { x } \in \ell _ { \infty } ^ { d _ { x } }$ if sup $\left| x _ { t } \right| < \infty .$ where | · | denotes any vector norm. We use the notation $x _ { i : j }$ to refer to the truncation of x to the finite-dimensional vector $( x _ { i } , x _ { i + 1 } , \ldots , x _ { j } )$ for $j \geq i ,$ and $x _ { i : j } = \varnothing$ otherwise. An operator $\mathbf { Y } : \ell ^ { d _ { \boldsymbol { x } } }  \ell ^ { d _ { \boldsymbol { y } } }$ is a mapping between two sequences and it is causal if $\mathbf { r } ( \mathbf { x } ) = ( \Upsilon _ { 0 } ( \overline { { x _ { 0 } } } ) , \overline { { \mathbf { \Upsilon } } } . . . , \Upsilon _ { t } ( x _ { 0 : t } ) , \dots )$ , and strictly causal if $\mathbf { r } ( \mathbf { x } ) = ( \Upsilon _ { 0 } ( \emptyset ) , \dots , \Upsilon _ { t } ( x _ { 0 : t - 1 } ) , \dots )$ , where $\Upsilon _ { 0 } ( \varnothing )$ is a vector in R<sup>dy</sup>. A causal operator ϒ is:

$\ell _ { p } { \mathrm { - s t a b l e } } \left( \mathbf { Y } \in \mathcal { L } _ { p } \right) \mathrm { i f } \ \mathbf { Y } ( \mathbf { x } ) \in \ell _ { p } ^ { d _ { y } }$ for all $\mathbf { x } \in \ell _ { p } ^ { d _ { x } }$

• incrementally $\ell _ { p } { \mathrm { - s t a b l e } }$ if $\exists \gamma \in [ 0 , \infty )$ such that $\forall \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } \in \ell _ { p } ^ { d _ { x } } , \| \mathbf { \dot { r } } ( \mathbf { x } _ { 1 } ) - \mathbf { \dot { r } } ( \mathbf { x } _ { 2 } ) \| _ { p } \leq \gamma \| \mathbf { x } _ { 1 } - \mathbf { x } _ { 2 } \| _ { p } ,$ where γ is the incremental finite gain $\mathrm { ( i . f . g . ) }$ of ϒ.

We define $\pmb { \Upsilon } _ { i : j } ( x _ { 0 : j } ) = ( \pmb { \Upsilon } _ { i } ( x _ { 0 : i } ) , \pmb { \Upsilon } _ { i + 1 } ( x _ { 0 : i + 1 } ) , \dots , \pmb { \Upsilon } _ { j } ( x _ { 0 : j } ) )$ for strictly causal $\mathbf { \bar { r } } , \mathbf { r } _ { i : j }$ acts on $x _ { 0 : j - 1 }$ . The sets of causal and strictly causal $\mathcal { L } _ { p }$ -stable operators are denoted $\mathcal { L } _ { p } ^ { C }$ and $\mathcal { L } _ { p } ^ { S C }$ , respectively.

## 2 Problem formulation

## 2.1 Closed-loop system dynamics

We consider a discrete-time dynamical system described by the input-output model:

$$
\left\{ \begin{array} { l l } { y _ { 0 } = G _ { 0 } ( \emptyset ) + \nu _ { 0 } , } \\ { y _ { t } = G _ { t } ( u _ { 0 : t - 1 } ) + \nu _ { t } , \qquad t \in \mathbb { N } , } \end{array} \right.\tag{1}
$$

where $G _ { 0 } ( \mathcal { O } ) \in \mathbb { R } ^ { d _ { y } }$ is the nominal initial output. At each time $t \in \mathbb { N } \cup \{ 0 \} , u _ { t } \in \mathbb { R } ^ { d _ { u } }$ denotes the input, $y _ { t } \in \mathbb { R } ^ { d _ { y } }$ is the output, and $\nu _ { t } \in \mathbb { R } ^ { d _ { y } }$ is the measurement noise corrupting the output. The dynamics $G _ { t }$ may be time-varying, nonlinear, and potentially unstable; by (1), y depends only on $u _ { 0 : t - 1 }$ $\mathrm { i . e . }$ , the system is strictly causal. In operator notation, system (1) reads:

$$
\mathbf { y } = \mathbf { G } ( \mathbf { u } ) + \mathbf { v } ,\tag{2}
$$

where G is a strictly causal operator such that $\mathbf { G } ( \mathbf { u } )$ = $\bigl ( G _ { 0 } ( \emptyset ) , G _ { 1 } ( u _ { 0 } ) , \dots \bigr ) , \ : \mathbf { u } = ( u _ { 0 } , u _ { 1 } ^ { \intercal } , \dots )$ and v, y are defined in a similar way. The next example shows the relation between (1) and state-space models.

Example 1 (From state-space to input-output models) Consider a strictly causal, time-invariant system in the state-space form:

$$
{ \left\{ \begin{array} { l l } { x _ { t + 1 } = f ( x _ { t } , u _ { t } ) , \quad x _ { 0 } = { \bar { x } } , } \\ { y _ { t } = h ( x _ { t } ) + \nu _ { t } , } \end{array} \right. }\tag{3}
$$

where $x _ { t } \in \mathbb { R } ^ { d _ { x } }$ is the state at time t and $\bar { \boldsymbol { x } } \in \mathbb { R } ^ { d _ { \boldsymbol { x } } }$ is the initial state. By recursion, the functions $G _ { t }$ in (1) are defined as:

$$
\begin{array} { r } { G _ { 0 } ( \emptyset ) = h \bigl ( \bar { x } \bigr ) , G _ { 1 } ( u _ { 0 } ) = h \bigl ( f ( \bar { x } , u _ { 0 } ) \bigr ) , \dots } \\ { G _ { t } \bigl ( u _ { 0 : t - 1 } \bigr ) = h \bigl ( \underbrace { f \bigl ( \ldots \bigl ( f } _ { t \mathrm { ~ } t i m e s } ( \bar { x } , u _ { 0 } ) , u _ { 1 } \bigr ) , \ldots , u _ { t - 1 } \bigr ) \bigr ) , \dots . } \end{array}
$$

Note that $G _ { t }$ depends on the initial state xfor all t, consistent¯ with input-output models in the literature [36]. When the system (3) is LTI and $\bar { x } = 0 ,$ , G reduces to a linear timeinvariant convolution operator, equivalently described in the Z-domain by the strictly proper transfer matrix $\mathbf { G } ( z )$

Next, we introduce the closed-loop system, illustrated in Figure 1 and described by:

$$
\begin{array} { r l } { \mathrm { C L } [ \mathbf { G } , \mathbf { K } ] : } & { { } \left\{ \begin{array} { l l } { \mathbf { y } = \mathbf { G } ( \mathbf { u } ) + \mathbf { v } , } \\ { \mathbf { u } = \mathbf { K } ( \mathbf { y } ) + \mathbf { r } , } \end{array} \right. } \end{array}\tag{4}
$$

where K is a causal, time-varying dynamical outputfeedback controller and $\mathbf { r } = ( r _ { 0 } , r _ { 1 } , \hdots ) , r _ { t } \in \mathbb { R } ^ { d _ { u } }$ , is an external excitation injected in addition to the feedback. In (4), CL[G,K] denotes the operator $( { \bf r } , { \bf v } ) \mapsto ( { \bf u } , { \bf y } )$ that the two equations define implicitly, i.e., the closed loop of Figure 1 seen as a single system mapping the exogenous signals to the internal ones. The same notation is used for any strictly causal system in place of G.

![](images/12bcbea4262707eeef158c08d81f718c671c3899425d972d18b7e240ee6378fe.jpg)  
Fig. 1. Block diagram of the closed-loop system formed by system G and controller K, with system input u, noisy output y, excitation signal r, and noise v.

As discussed in Section 1, data are collected while the system operates in closed loop. To ensure reliable data collection and prevent unbounded measurements, the closed-loop system must satisfy an appropriate notion of stability.

Definition 1 (Closed-loop $\ell _ { p }$ -stability [18]) $L e t$ G be a strictly causal operator and K a causal operator. The closed loop of G and K in (4) is $\ell _ { p } .$ stable if u $\in \ell _ { p }$ and $\mathbf { y } \in \ell _ { p }$ for every $\mathbf { r } \in \ell _ { p }$ and every $\mathbf { v } \in \ell _ { p } .$ Equivalently, we write $\mathbf { C L } [ \mathbf { \bar { G } } , \mathbf { K } ] \in \dot { \mathcal { L } } _ { p } .$ We denote by $\mathcal { G } _ { \mathbf { K } } : = \left\{ \mathbf { G } \right.$ strictly causal $\mathbf { \sigma } : \dot { \mathbf { C L } } [ \mathbf { G } , \dot { \mathbf { K } } ] \in \dot { \mathcal { L } } _ { p } \}$ the set of systems stabilized by K.

Definition 1 constrains only the interconnection: neither G nor K is required to belong to $\mathcal { L } _ { p }$ individually. When G and K are LTI, Definition 1 recovers the standard notion: $\mathbf { C L } [ \mathbf { G } , \mathbf { K } ] \in \mathcal { L } _ { p }$ for every $p \geq 1$ if and only if the transfer matrix describing CL[G,K] is asymptotically stable.

## 2.2 Identification Objective

Our goal is to learn a dynamical model $\hat { \mathbf { G } }$ that best approximates the true system G using closed-loop data collected with a known stabilizing controller K. While $\hat { \mathbf { G } }$ may be open-loop unstable, capturing potential plant instability, we enforce $\mathbf { C L } [ \hat { \mathbf { G } } , \mathbf { K } ] \in \mathcal { L } _ { p }$ , i.e., we constrain the search to $\mathcal { G } _ { \mathbf { K } }$

The stochasticity of the closed-loop data stems from the unobserved measurement noise v and from the excitation r, actively sampled from a user-designed distribution to make the input informative. Since the system operates under feedback, the control input u is a deterministic function of the measured signals (r,y) through the closed-loop equations (4). Consequently, a data sample from a given experiment is the fully observable, infinite-length trajectory ${ \pmb \tau } = ( { \bf r } , { \bf u } , { \bf y } )$ , which is drawn from the underlying closed-loop data distribution D induced by the distributions of r and v.

Ideally, we seek to minimize an expected identification loss

c over D, as follows:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { { \hat { \mathbf { G } } } } \mathbb { E } _ { \pmb { \tau } \sim \mathcal { D } } c ( \hat { \mathbf { G } } , \pmb { \tau } ) } \\ { \displaystyle \mathrm { s . t . } { \mathrm { C L } } [ \hat { \mathbf { G } } , \mathbf { K } ] \in \mathcal { L } _ { p } , } \end{array}\tag{5}
$$

where $c ( \hat { \mathbf { G } } , \pmb { \tau } )$ quantifies the mismatch between the predicted output yˆ and the measured y. Importantly, (5) does not commit to how yˆ is generated from τ: this choice, deferred to Section 4, gives rise to the direct and indirect strategies.

Directly solving the optimization problem in (5) is impractical due to three fundamental challenges:

(C1) Complex constraints: The set $\mathcal { G } _ { \mathbf { K } }$ is generally nonconvex and difficult to parameterize directly.

(C2) Intractable horizon: Evaluating the objective over infinite-length trajectories is computationally infeasible.

(C3) Unknown distribution: The true data distribution D is unknown, preventing the direct computation of the expectation.

In the following, we address these challenges systematically. We first focus on the stability constraint (C1) via the ICI framework described in Section 3. Subsequently in Section 4 we tackle challenges (C2) and (C3), by adopting an Empirical Risk Minimization (ERM) approach defined over finite-length trajectories.

## 3 Internal Controller Identification (ICI) framework

## 3.1 Parameterization of Stabilized Systems

Problem (5) requires learning over $\mathcal { G } _ { \mathbf { K } } .$ , a constraint that is hard to enforce directly because this set is generally nonconvex. We parameterize all and only such systems through a dual formulation of Internal Model Control (IMC) architectures [11], which we call the Internal Controller Identification (ICI) framework. Specifically, $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ is the negative feedback interconnection in Figure 2 of a copy of the controller K and a trainable strictly causal operator $\hat { \mathbf { S } } \in \mathcal { L } _ { p } ^ { S C }$ , so that, for an input uˆ , the predicted output satisfies

$$
\hat { \mathbf { y } } = \hat { \mathbf { G } } _ { \mathrm { I C I } } ( \hat { \mathbf { u } } ) = \hat { \mathbf { S } } \big ( \hat { \mathbf { u } } - \mathbf { K } ( \hat { \mathbf { y } } ) \big ) .\tag{6}
$$

This parameterization transforms the problem of learning $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ into optimizing $\hat { \bf S }$ over the set $\mathcal { L } _ { p } ^ { S \bar { C } }$ [19,18].

The next theorem relates $\mathcal { G } _ { \mathbf { K } }$ to the models representable in the ICI framework. Note that statements T.1 and T.2 are not symmetric: sufficiency requires $\mathbf { K } \in \mathcal { L } _ { p } ^ { C }$ and incrementally $\ell _ { p } .$ -stable, whereas necessity uses only causality of K. Note that the second hypothesis of T.1 subsumes the first whenever $\mathbf { K } ( \mathbf { 0 } ) \in \ell _ { p }$

![](images/8b1f86954c8eec3f6bba821ff3260d539ec61455cae798875fb37a4bbd08f38a.jpg)

Fig. 2. In the ICI framework, the model $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ consists of the negative feedback interconnection of a copy of the controller K and a trainable operator $\hat { \mathbf { S } } \in \mathcal { L } _ { v } ^ { S C }$ . The model input and output are denoted by uˆ and yˆ, respectively.

Theorem 1 Let K be causal and let $\hat { \bf S } \in \mathcal { L } _ { p } ^ { S C }$ . Let $\hat { \mathbf { G } } _ { I C I }$ be the ICI model defined by (6).

T.1 (Sufficiency.) If $\mathbf { K } \in \mathcal { L } _ { p } ^ { C }$ and K is incrementally $\ell _ { p } -$ stable, then $\hat { \mathbf { G } } _ { I C I }$ is strictly causal and $\mathbf { C L } [ \hat { \mathbf { G } } _ { I C I } , \mathbf { K } ] \in$ $\mathcal { L } _ { p } , i . e . , \hat { \mathbf { G } } _ { I C I } \in \mathcal { G } _ { \mathbf { K } } .$

T.2 (Necessity.) Conversely, for every $\mathbf { G } \in \mathcal { G } _ { \mathbf { K } }$ there exists a unique $\hat { \mathbf { S } } \in \mathcal { L } _ { p } ^ { S C }$ such that the ICI model (6) satisfies $\hat { \mathbf { G } } _ { I C I } = \mathbf { G } .$

Corollary 1 Under the hypotheses ofT.1, the map $\hat { \mathbf { S } } \mapsto \hat { \mathbf { G } } _ { I C I }$ defined by (6) is a bijection from $\mathcal { L } _ { p } ^ { \check { S C } }$ onto $\mathcal { G } _ { \mathbf { K } }$

Proof of Theorem 1 We first state a useful result from [25].

Lemma 1 ([25]) Consider a causal operator ϒ such that $\displaystyle \mathbf { r } ^ { o } : = \mathbf { r } - \mathbf { I }$ is strictly causal, where I is the identity operator. Then, the causal operator $\mathbf { r } ^ { - 1 }$ exists, and the sequence $\mathbf { b } =$ $\mathbf { r } ^ { - 1 } \mathbf { a }$ satisfies

$$
b _ { t } = a _ { t } - \Upsilon _ { t } ^ { o } ( b _ { 0 : t - 1 } ) , \qquad t = 0 , 1 , \ldots .\tag{7}
$$

We start by proving T.1: for every $\mathbf { r } \in \ell _ { p } ^ { d _ { u } }$ and $\mathbf { v } \in \ell _ { p } ^ { d _ { y } }$ we show that both $\hat { \mathbf { y } } \in \ell _ { p } ^ { d _ { y } }$ and $\hat { \mathbf { u } } \in \ell _ { p } ^ { d _ { u } }$ , so that $\mathbf { C L } [ \hat { \mathbf { G } } _ { \mathrm { I C I } } , \mathbf { K } ] \in \mathcal { L } _ { p }$ Denote by ωˆ the input to the operator $\hat { \bf S } .$ In $\mathrm { C L } [ \hat { \mathbf { G } } _ { \mathrm { I C I } } , \mathbf { K } ]$ the external copy of K is driven by the measured output $\hat { \mathbf { y } } + \mathbf { v } ,$ while the copy internal to $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ is driven by yˆ; hence

$$
\pmb { \hat { \omega } } = \hat { \mathbf { u } } - \mathbf { K } \big ( \hat { \mathbf { y } } \big ) = \mathbf { r } + \mathbf { K } \big ( \hat { \mathbf { S } } ( \hat { \pmb { \omega } } ) + \mathbf { v } \big ) - \mathbf { K } \big ( \hat { \mathbf { S } } ( \hat { \pmb { \omega } } ) \big ) .\tag{8}
$$

Since $\hat { \bf S }$ is strictly causal and K is causal, $\hat { \omega } _ { t }$ in (8) depends only on $r _ { 0 : t } , \ \nu _ { 0 : t }$ and $\hat { \omega } _ { 0 : t - 1 } ;$ hence ωˆ exists and is unique, so the loop is well posed, and the same recursion shows that $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ is strictly causal.

Let $P _ { M }$ denote the truncation operator defined as $( P _ { M } \mathbf { x } ) _ { t } = x _ { t }$ for $t \le M$ and $( P _ { M } \mathbf { x } ) _ { t } = 0$ otherwise.

Every truncated signal belongs to $\ell _ { p } ,$ and causality of K gives $P _ { M } \dot { \bf K } ( { \bf x } ) = P _ { M } { \bf K } ( \bar { P _ { M } } { \bf x } )$ . Since $P _ { M } \mathbf { { \dot { x } } } \in \ell _ { p }$ for every x, the i.f.g.

inequality $\| \mathbf { K } ( \mathbf { x } _ { 1 } ) - \mathbf { K } ( \mathbf { x } _ { 2 } ) \| _ { p } \leq \gamma _ { K } \| \mathbf { x } _ { 1 } - \mathbf { x } _ { 2 } \| _ { p }$ holds for the truncated signals [46]. Truncating (8) and using $\| P _ { M } \mathbf { x } \| _ { p } \leq$ $\| \mathbf { x } \| _ { p }$ together with the triangle inequality,

$$
\begin{array} { r l } & { \| P _ { M } \pmb { \hat { \omega } } \| _ { p } \leq \| \mathbf { r } \| _ { p } + \left\| \mathbf { K } \left( P _ { M } \hat { \mathbf { S } } ( \pmb { \hat { \omega } } ) + P _ { M } \mathbf { v } \right) - \mathbf { K } \left( P _ { M } \hat { \mathbf { S } } ( \pmb { \hat { \omega } } ) \right) \right\| _ { p } } \\ & { \qquad \leq \| \mathbf { r } \| _ { p } + \gamma _ { K } \| P _ { M } \mathbf { v } \| _ { p } \leq \| \mathbf { r } \| _ { p } + \gamma _ { K } \| \mathbf { v } \| _ { p } , } \end{array}
$$

uniformly in $M ,$ where $\gamma _ { K }$ is the incremental finite gain of $\mathbf { K }$

Since $M \mapsto \| P _ { M } \hat { \pmb { \omega } } \| _ { p }$ is nondecreasing and bounded above, it converges to $\| \hat { \pmb { \omega } } \| _ { p }$ as $M  \infty$ ; therefore $\hat { \pmb { \omega } } \in \ell _ { p } ^ { d _ { u } }$ and $\| \hat { \pmb { \omega } } \| _ { p } \leq \| \mathbf { r } \| _ { p } + \gamma _ { K } \| \mathbf { v } \| _ { p } .$

Since $\hat { \mathbf { S } } \in \mathcal { L } _ { p } .$ , it follows that $\hat { \mathbf { y } } = \hat { \mathbf { S } } ( \hat { \pmb { \omega } } ) \in \ell _ { p } ^ { d _ { y } }$ , and so is the measured output ${ \hat { \mathbf { y } } } + \mathbf { v } .$ . Consequently, since $\mathbf { K } \in { \mathcal { L } } _ { p } ,$ , we obtain $\hat { \mathbf { u } } = \mathbf { K } ( \hat { \mathbf { y } } + \mathbf { v } ) + \mathbf { r } \in \ell _ { p } ^ { d _ { u } }$ . Therefore $\mathrm { C L } [ \hat { \mathbf { G } } _ { \mathrm { I C I } } , \mathbf { K } ] \in \mathcal { L } _ { p } .$ i.e., $\hat { \mathbf { G } } _ { \mathrm { I C I } } \in \mathcal { G } _ { \mathbf { K } }$

Next, we prove T.2. Given $\mathbf { G } \in \mathcal { G } _ { \mathbf { K } }$ , define $\hat { \mathbf { S } } : = \mathbf { G } ( \mathbf { I } - \mathbf { \Omega }$ $\mathbf { K G } ) ^ { - 1 }$ , where the inverse exists by Lemma 1 since KG is strictly causal. The operator $\hat { \bf S }$ is strictly causal, being the composition of the causal operator $( \mathbf { I } - \mathbf { K } \mathbf { G } ) ^ { - 1 }$ with the strictly causal $\mathbf { G } ;$ moreover $\hat { \mathbf { S } } \in \mathcal { L } _ { p } ,$ , since it coincides with the restriction of CL[G, K] to $\mathbf { v } = \mathbf { 0 } , \mathrm { i . e . }$ , with the map $\mathbf { r } \mapsto \mathbf { y }$ which belongs to $\mathcal { L } _ { p }$ because $\mathbf { G } \in \mathcal { G } _ { \mathbf { K } }$ (see also [10]). Note that this construction gives $\hat { S } _ { 0 } ( \mathcal { O } ) = G _ { 0 } ( \mathcal { O } )$ automatically.

We apply the same input uˆ to both G and $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ , as illustrated in Figure 2, and denote their respective outputs by $\hat { \mathbf { y } } ^ { \mathrm { d e s } }$ and yˆ. Our goal is to prove that $\hat { \mathbf { y } } ^ { \mathrm { d e s } } = \hat { \mathbf { y } }$ , or equivalently, $\dot { \hat { y } } _ { t } ^ { \mathrm { d e s } } = \hat { y } _ { t }$ for all $t \geq 0$

For G we have $\hat { \mathbf { y } } ^ { \mathrm { d e s } } = \mathbf { G } ( \hat { \mathbf { u } } )$ , while from (6),

$$
\begin{array} { r l } & { \hat { \mathbf { y } } = \hat { \mathbf { G } } _ { \mathrm { I C I } } ( \hat { \mathbf { u } } ) = \hat { \mathbf { S } } \big ( \hat { \mathbf { u } } - \mathbf { K } ( \hat { \mathbf { y } } ) \big ) } \\ & { \quad = \mathbf { G } \underbrace { ( \mathbf { I } - \mathbf { K G } ) ^ { - 1 } \big ( \hat { \mathbf { u } } - \mathbf { K } ( \hat { \mathbf { y } } ) \big ) } _ { \mathbf { b } } = \mathbf { G } \mathbf { b } . } \end{array}
$$

These relations are equivalent to the causal relations

$$
\begin{array} { r } { \hat { y } _ { t } ^ { \mathrm { d e s } } = G _ { t } ( \hat { u } _ { 0 : t - 1 } ) , \qquad \hat { y } _ { t } = G _ { t } ( { b } _ { 0 : t - 1 } ) , \qquad \forall t \geq 0 , } \end{array}\tag{9}
$$

where we used the strict causality of G to truncate the operands at time t − 1. Moreover, by the recursion (7) of Lemma 1,

$$
b _ { t } = \hat { u } _ { t } - K _ { t } \big ( \hat { y } _ { 0 : t } \big ) + K _ { t } \big ( G _ { 0 : t } \big ( b _ { 0 : t - 1 } \big ) \big ) , \qquad \forall t \geq 0 .\tag{10}
$$

The second relation in (9) gives $\hat { y } _ { 0 : t } = G _ { 0 : t } ( b _ { 0 : t - 1 } )$ , so the two controller terms in (10) cancel identically and $b _ { t } = \hat { u } _ { t }$ for all $t \geq 0$ . Substituting into (9) yields $\hat { y } _ { t } ^ { \mathrm { d e s } } = \hat { y } _ { t }$ for all $t \geq 0$

Uniqueness is forced by (6). If $\hat { \bf S } \in \mathcal { L } _ { p } ^ { S C }$ satisfies $\hat { \mathbf { G } } _ { \mathrm { I C I } } = \mathbf { G }$ then $\mathbf { G } ( \hat { \mathbf { u } } ) = \hat { \mathbf { S } } \big ( \hat { \mathbf { u } } - \mathbf { K } \big ( \mathbf { G } ( \hat { \mathbf { u } } ) \big ) \big )$ for every uˆ . Setting $\mathbf { b } = ( \mathbf { I } -$ $\mathbf { K G } ) ( \hat { \mathbf { u } } )$ and using the invertibility of I−KG from Lemma 1 gives $\hat { \mathbf { S } } ( \mathbf { b } ) = \mathbf { G } ( \mathbf { I } - \mathbf { K } \mathbf { G } ) ^ { - 1 } ( \mathbf { b } )$ for every b, i.e., $\hat { \mathbf { S } } = \mathbf { G } ( \mathbf { I } -$ $\mathbf { K G } ) ^ { - 1 }$ □

Proof of Corollary 1 By T.1, the map $\hat { \mathbf { S } } \mapsto \hat { \mathbf { G } } _ { \mathrm { I C I } }$ is well defined from $\mathcal { L } _ { p } ^ { S C }$ into $\mathcal { G } _ { \mathbf { K } }$ , and existence in T.2 makes it surjective onto $\mathcal { G } _ { \mathbf { K } }$ . For injectivity, let $\hat { \bf S } _ { 1 } , \hat { \bf S } _ { 2 } \in \mathcal { L } _ { p } ^ { S C }$ induce the same model $\mathbf { G } : = \hat { \mathbf { G } } _ { \mathrm { I C I } } ^ { 1 } = \hat { \mathbf { G } } _ { \mathrm { I C I } } ^ { 2 }$ . Then $\mathbf { G } \in \mathcal { G } _ { \mathbf { K } }$ by T.1, so uniqueness in T.2 gives $\hat { \bf S } _ { 1 } = \hat { \bf S } _ { 2 }$ □

Remark 1 (Open-loop instability of $\hat { \mathbf { G } } _ { \mathbf { I C I } } )$ Note that Theorem 1 certifies $\mathrm { C L } [ \hat { \mathbf { G } } _ { I C I } , \mathbf { K } ] \in \mathcal { L } _ { p } ,$ but $\hat { \mathbf { G } } _ { I C I }$ itself need not be $\ell _ { p } { - } s t a b l e$ : while $\hat { \mathbf { S } } \in \mathcal { L } _ { p }$ by construction, the corresponding $\hat { \mathbf { G } } _ { I C I } ,$ recovered through the feedback equation (6), may be open-loop unstable. This is precisely what allows the ICI framework to identify inherently unstable plants while keeping the identified closed-loop behavior consistent with the true one.

Since $\mathbf { G } \in \mathcal { G } _ { \mathbf { K } } .$ , point T.2 in Theorem 1 associates with G a unique operator $\mathbf { S } \in \mathcal { L } _ { p } ^ { S C }$ , which we call the true ICI operator. By Corollary 1, the constrained problem (5) can be replaced by the following unconstrained optimization over $\hat { \bf { S } } \mathrm { : }$

$$
\operatorname* { m i n } _ { \hat { \mathbf { S } } \in \mathcal { L } _ { p } ^ { S C } } \mathbb { E } _ { \pmb { \tau } \sim \mathcal { D } } c ( \hat { \mathbf { S } } , \pmb { \tau } ) ,\tag{11}
$$

where the associated model $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ is the interconnection of $\hat { \bf S }$ and K defined in (6). The key difference is that the constraint $\mathbf { C L } [ \hat { \mathbf { G } } _ { \mathrm { I C I } } , \mathbf { K } ] \in \mathcal { L } _ { p }$ is now structurally satisfied for every $\hat { \mathbf { S } } \in$ $\mathcal { L } _ { p } ^ { S C }$

Remark 2 (Recursive implementation of $\hat { \bf S } )$ Since $\hat { \bf S }$ is strictly causal, the output yˆ<sub>t</sub> depends on past outputs only up to time $t - 1$ . This observation enables a recursive computation of $\hat { y } _ { t } ,$ , eliminating the need to solve the implicit nonlinear equation (6) at each time step. Additionally, $i f \hat { \mathbf { S } }$ and K are input-output operators associated with dynamical systems (see Example 1), their state-space representations can be used to implement (6) as a Markovian dynamical system, thereby avoiding the need to store the full history of $\hat { u } _ { 0 : t - 1 }$ and $\hat { y } _ { 0 : t - 1 } .$

## 3.2 Unconstrained optimization of the operator S<sup>ˆ</sup>

While the ICI parameterization (11) removes the explicit stability constraint, optimizing over the infinite-dimensional space $\dot { \mathcal { L } } _ { p } ^ { S C }$ remains intractable. To make this feasible, the operator $\hat { \bf S }$ can be modeled using families of $\mathcal { L } _ { p } ^ { S C }$ operators described by a finite-dimensional parameter vector $\boldsymbol { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ such as Recurrent Equilibrium Networks (RENs) [39], certain classes of State-Space Models (SSMs) [23,33,50], or neural network representations of Hamiltonian systems [49], which admit a state-space form enabling recursive output computation (see Remark 2). Crucially, these families yield a free parameterization, guaranteeing that the parameterized operator $\hat { \mathbf { S } } ^ { \theta }$ belongs to $\mathcal { L } _ { p } ^ { S C }$ for any arbitrary $\boldsymbol { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ . Defining $\hat { \mathbf { G } } _ { \mathrm { I C I } } ^ { \theta }$ as the ICI model (6) built from $\hat { \mathbf { S } } ^ { \theta }$ and K, problem $( \romannumeral 1 )$ reduces to a tractable, unconstrained optimization over $\boldsymbol { \theta } \in \mathbb { R } ^ { d _ { \theta } }$

$$
\operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d _ { \theta } } } \mathbb { E } _ { \pmb { \tau } \sim \mathcal { D } } c \big ( \hat { \mathbf { S } } ^ { \theta } , \pmb { \tau } \big ) ,\tag{12}
$$

which can be efficiently solved using gradient-based methods. Since the predictor is uniquely determined by θ, with a small abuse of notation, hereafter we use $c ( \theta , \pmb { \tau } )$ instead of $c ( \hat { \mathbf { S } } ^ { \theta } , \pmb { \tau } )$

## 3.3 Context-enriched ICI

Plants often operate under varying conditions, $\mathrm { e . g . }$ , for robotic systems, changes in payload or terrain-dependent friction. When measurable, we represent information about these conditions by a context sequence $\pmb { \zeta } = ( \zeta _ { 0 } , \zeta _ { 1 } , \ldots )$ , with $\zeta _ { t } \in \mathcal { Z } \subset \mathbb { R } ^ { d _ { \zeta } }$ . Providing context can help a model adapt its predictions across operating regimes without having to infer those conditions solely from the input history.

Crucially, context may persist even when the dynamical input decays: for example, a constant nonzero payload signal is bounded but does not belong to $\ell _ { p }$ for $p < \infty .$ . Simply appending context to the input of an $\ell _ { p }$ -stable operator therefore does not automatically preserve its stability properties. Therefore, to inject context, we instead adapt the operator decomposition of [34] to the ICI predictor:

$$
\hat { \bf S } ( \pmb { \hat { \omega } } ; \pmb { \zeta } ) = \hat { \bf S } _ { \infty } ( \pmb { \hat { \omega } } , \pmb { \zeta } ) \boxtimes \hat { \bf S } _ { p } ( \pmb { \hat { \omega } } ) .\tag{13}
$$

Here, $\hat { \bf S } _ { p } \in \mathcal { L } _ { p } ^ { S C }$ generates stable dynamical features from ωˆ , while the strictly causal mixer $\hat { \mathbf { S } } _ { \infty }$ produces matrices A<sub>t</sub> satisfying sup<sub>t</sub> $\hat { \omega } , \zeta \left\| A _ { t } \right\| \leq \bar { \sigma } < \infty$ . The product ⊠ acts pointwise: $( \mathbf { A } \boxtimes \mathbf { z } ) _ { t } = \dot { A } _ { t } z _ { t }$ . Intuitively, context changes how the features are scaled and combined, while their magnitude remains controlled by the stable module $\hat { \bf S } _ { p }$ . Indeed, the pointwise bound $\left\| A _ { t } z _ { t } \right\| \leq \bar { \sigma } \| z _ { t } \|$ gives [34, Thm. 1]

$$
\| \hat { \mathbf { S } } ( \hat { \pmb { \omega } } ; \pmb { \zeta } ) \| _ { p } \leq \bar { \sigma } \| \hat { \mathbf { S } } _ { p } ( \hat { \pmb { \omega } } ) \| _ { p } , \qquad \forall \hat { \pmb { \omega } } \in \ell _ { p } ,\tag{14}
$$

uniformly over context sequences. Thus, $\hat { \mathbf { S } } ( \cdot ; \pmb { \zeta } ) \in \mathcal { L } _ { p } ^ { S C }$ , and under the controller assumptions of T.1, the resulting ICI model remains stabilized by K for every admissible context.

Under the mild weighted-envelope input assumption and context-uniform envelope-preservation condition of [34, Thm. 2], the needed factorization (13) is also necessary and hence non-conservative.

Finally, context can also be computed from predicted outputs, $\dot { \zeta } _ { t } = \kappa ( \hat { y } _ { t } )$ . The mixer’s strict causality prevents an algebraic loop, and the same bound holds along the resulting trajectory. Consequently, known state-dependent context features can be supplied during simulation, as in Section 6.4, without requiring recorded context trajectories. In the sequel, for simplicity, we omit including context variables in the available data; indeed, they can always be incorporated in a plug-in fashion, as described so far, without altering the main result.

## 4 Tractable Identification via the ICI Parameterization

We now turn (12) into a practical learning problem: we restrict it to finite horizons, introduce the direct (Section 4.2) and indirect (Section 4.3) predictors, and fix the loss to the simulation error criterion (Section 4.4).

## 4.1 Finite-Horizon Empirical Risk Minimization

To address challenges (C2) and (C3), we restrict (12) to finite horizons and approximate the expectation empirically. Let $\mathcal { D } _ { T }$ denote the distribution over closed-loop trajectories truncated to a finite horizon T, where a single trajectory is denoted by $\tau _ { T } = \left( r _ { 0 : T - 1 } , u _ { 0 : T - 1 } , y _ { 0 : T - 1 } \right)$ . We define the true finite-horizon identification risk as:

$$
J ( \theta , \mathcal { D } _ { T } ) : = \mathbb { E } _ { \tau _ { T } \sim \mathcal { D } _ { T } } \left[ c \big ( \theta , \tau _ { T } \big ) \right] ,\tag{15}
$$

where $c ( \cdot , \cdot )$ quantifies the discrepancy between the measured outputs and the predicted sequence $\hat { y } _ { 0 : T - 1 }$ . This is the quantity bounded in Section 5. Since $\mathcal { D } _ { T }$ is unknown, we rely on N independent closed-loop experiments, $\mathbb { S } _ { T } ^ { N } = \{ \tau _ { T } ^ { n } \} _ { n = 0 } ^ { N - 1 }$ with $\tau _ { T } ^ { n } \sim \bar { \mathcal { D } } _ { T }$ i.i.d., and approximate (15) with the empirical risk:

$$
\hat { J } ( \theta , \mathbb { S } _ { T } ^ { N } ) : = \frac { 1 } { N } \sum _ { n = 0 } ^ { N - 1 } [ c ( \theta , \tau _ { T } ^ { n } ) ] .\tag{16}
$$

The value of $\hat { J }$ depends on how $\hat { y } _ { 0 : T - 1 } ^ { n }$ is generated from $\tau _ { T } ^ { n }$ the following subsections introduce the direct and indirect strategies, which differ precisely in this choice.

## 4.2 Direct ICI identification

In the direct ICI strategy the measured plant input is fed directly to the reconstructed model, $\left\{ \hat { u } _ { 0 : T - 1 } ^ { n } \right\} _ { n = 0 } ^ { \bar { N } - 1 }$ = $\{ u _ { 0 : T - 1 } ^ { n } \} _ { n = 0 } ^ { N - 1 }$ , treating the closed-loop data as if collected in an open-loop experiment:

$$
\begin{array} { r l } { \underset { \theta \in \mathbb { R } ^ { d _ { \theta } } } { \operatorname* { m i n } } } & { \hat { J } ( \theta , \mathbb { S } _ { T } ^ { N } ) } \\ { \mathrm { s . t . } } & { \hat { y } _ { t } ^ { n } = \hat { S } _ { t } ^ { \theta } \big ( u _ { 0 : t - 1 } ^ { n } - K _ { 0 : t - 1 } ( \hat { y } _ { 0 : t - 1 } ^ { n } ) \big ) , } \\ & { \forall t = 0 , \ldots , T - 1 , \forall n = 0 , \ldots , N - 1 . } \end{array}\tag{17}
$$

The model $\hat { \mathbf { G } } _ { \mathrm { I C I } } ^ { \theta }$ built from $\hat { \mathbf { S } } ^ { \theta }$ through (6) satisfies $\mathbf { C L } [ \hat { \mathbf { G } } _ { \mathrm { I C I } } ^ { \theta } , \mathbf { K } ] \in \mathcal { L } _ { p } ,$ , as guaranteed by point T.1 in Theorem 1.

While direct methods lead to tractable optimization problems, they have two fundamental drawbacks.

D.1 They can be used only when $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ is stable [16], since with a simulation criterion, as adopted in Section 4.4, the risk J is evaluated on open-loop predictions of $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ which may diverge if the model is unstable.

D.2 The regressor is noise-corrupted. Feedback makes v enter the trainable operator’s input through $\mathbf { u } = \mathbf { K } ( \mathbf { y } ) +$ r (Figure 1), so the regressor and the residual are correlated at first order. Even in the linear case this makes direct methods inconsistent under a misspecified noise model: the true G is not recovered from infinite data [43,45,21].

## 4.3 Indirect ICI identification

![](images/e2ec7db9e8f13ce075cd991f65f1f75757a2136f7deb8201a5fb7b05e4caebd9.jpg)

![](images/6f484a17d9f2ac7c538d73655f21e17c0d8ceac2b32663ddbd525881f1592399.jpg)  
Fig. 3. Top: Closed-loop system connecting the true plant G (represented within the ICI framework by the operator S) and the controller K (left), alongside its linearized approximation (right). Bottom: In the linear case, noise injection can be shifted to enable loop cancellation, yielding the exact simplified form on the right.

Indirect identification estimates the closed-loop response to the exogenous excitation r and uses the known controller to recover the plant [43]. In the ICI framework, this amounts to learning the true operator $\mathbf { S } \in \mathcal { L } _ { p } ^ { S C }$ associated with $\mathbf { G } \in$ $\mathcal { G } _ { \mathbf { K } }$ by point T.2 in Theorem 1, and recovering the plant through (6), evaluated by the recursion in Remark 2; this implements the dual-Youla parameterization idea [4,10] with the finite-dimensional stable operator models of Section 3.2. The closed loop under analysis is then equivalent to the interconnection of S and K in the top-left plot of Figure 3, and the measured output satisfies

$$
\begin{array} { r } { \mathbf { y } = \mathbf { S } ( \pmb { \omega } ) + \mathbf { v } , \quad \pmb { \omega } : = \mathbf { r } + \mathbf { K } ( \mathbf { y } ) - \mathbf { K } ( \mathbf { y } - \mathbf { v } ) . } \end{array}\tag{18}
$$

Identity (18) shows where the noise enters: additively at the output, and through the controller inside the argument of S; only the first entry is exogenous. Writing $\mathbf { y } = \mathbf { S } ( \mathbf { r } ) + \mathbf { e }$ gives the residual in closed form, $\mathbf { e } = \mathbf { S } ( \pmb { \omega } ) - \mathbf { S } ( \mathbf { r } ) + \mathbf { v } ,$ , which vanishes with v: then ${ \pmb { \omega } } = { \bf r }$ and S reduces to the noise-free map $\mathbf { r } \mapsto \mathbf { y }$ of CL[G, K] in T.2. Fitting S from $\displaystyle ( \mathbf { r } , \mathbf { y } )$ thus has the form of an open-loop output-error problem, but with a non-exogenous disturbance whose dependence on r sets the bias. To expose this, assume $\mathbf { G } _ { 0 : T - 1 }$ and $\mathbf { K } _ { 0 : T - 1 }$ twice continuously differentiable near the operating trajectories, so that $\mathbf { S } _ { 0 : T - 1 }$ is as well by (7), and linearize [4] S around r and K around $\mathbf { S } ( \mathbf { r } )$

$$
\begin{array} { r l } & { \mathbf { y } \approx \mathbf { S } ( \mathbf { r } ) + \mathbf { S } _ { L } \big ( \mathbf { K } ( \mathbf { y } ) - \mathbf { K } ( \mathbf { y } - \mathbf { v } ) \big ) + \mathbf { v } } \\ & { \quad \approx \mathbf { S } ( \underbrace { \mathbf { r } } _ { n o i s e - i n d e p e n d e n t \ : s i g n a l \ : n o i s e } ) + \underbrace { \mathbf { S } _ { L } \big ( \mathbf { K } _ { L } ( \mathbf { v } ) \big ) + \mathbf { v } } _ { n o i s e } , } \end{array}\tag{19}
$$

where $\mathbf { S } _ { L } : = D \mathbf { S } [ \mathbf { r } ]$ and $\mathbf { K } _ { L } : = D \mathbf { K } [ \mathbf { S } ( \mathbf { r } ) ]$ are causal linear operators, block lower triangular by causality and strictly so for $\mathbf { S } _ { L } .$ . The second line follows by expanding both terms of $\mathbf { K } ( \mathbf { y } ) - \mathbf { K } ( \mathbf { y } - \mathbf { v } )$ at $\mathbf { S } ( \mathbf { r } )$ : the constant terms cancel and the linear ones combine into ${ \bf K } _ { L } ( { \bf v } )$ . Since $\mathbf { y } - \mathbf { S } ( \mathbf { r } )$ and $\mathbf { y } - \mathbf { v } - $ $\mathbf { S } ( \mathbf { r } )$ are $O ( \left| \mathbf { | v | } \right| )$ by incremental ℓ<sub>p</sub>-stability of K and local Lipschitz continuity of $\mathbf { S } _ { 0 : T - 1 }$ , both approximations in (19) hold up to $O ( \| \mathbf { v } \| ^ { 2 } )$ with coefficients depending on r alone, so that, to first order in the noise, r propagates through S alone (top-right plot of Figure 3). The same identity holds for the ICI model with $\hat { \mathbf { S } } ^ { \theta }$ in place of S, so $\hat { \mathbf { S } } ^ { \theta } ( \mathbf { r } )$ is its firstorder closed-loop response to r, which we adopt as predictor. Accordingly, we adopt the open-loop learning problem

$$
\begin{array} { r l } { \underset { \theta \in \mathbb { R } ^ { d _ { \theta } } } { \operatorname* { m i n } } } & { \hat { J } ( \theta , \mathbb { S } _ { T } ^ { N } ) } \\ { \mathrm { s . t . } } & { \hat { y } _ { t } ^ { n } = \hat { S } _ { t } ^ { \theta } \left( r _ { 0 : t - 1 } ^ { n } \right) , } \\ & { \forall t = 0 , \ldots , T - 1 , \forall n = 0 , \ldots , N - 1 , } \end{array}\tag{20}
$$

with $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ reconstructed as the interconnection of the learned $\hat { \mathbf { S } } ^ { \theta }$ and the known K. Problem (20) is not an exact reformulation: its regressor is r, while the signal driving S in the experiment is ${ \pmb \omega } ,$ which equals r only for $\mathbf { v } = \mathbf { 0 } .$ . By (19) the first-order residual is linear in v with r-dependent coefficients, hence of zero conditional mean given r; the minimizer of (20) is therefore biased at $O ( \| \mathbf { v } \| ^ { 2 } )$ , set by the curvature of S and K, against the first-order correlation of D.2. Superposition makes the separation exact in the linear case (Remark 4).

Remark 3 (Validity of linear approximations) The approximation (19) is exact in the absence of noise and when S and K are affine, with $\mathbf { S } _ { L }$ and $\mathbf { K } _ { L }$ their linear parts; in the linear case the equivalent output disturbance (SK + I)v makes the loops cancel exactly (bottom plots ofFigure 3). In the nonlinear case it is a local small-noise approximation: both v and the induced perturbation $\mathbf { K } ( \mathbf { y } ) - \mathbf { \dot { K } } ( \mathbf { y } - \mathbf { v } )$ must stay where the Taylor remainders are small. Incremental $\ell _ { p } { - } s t a b i l i t y$ of K bounds that perturbation by γ times the noise norm on each truncated horizon, and bounds the induced $\ell _ { p }$ -norm of every truncated Jacobian of K by $\gamma _ { \mathbf { K } } ,$ , so $\mathbf { K } _ { L }$ extends to a bounded operator on $\ell _ { p } .$ Accuracy therefore depends on local curvature and noise amplification; a high signal-to-noise ratio alone does not guarantee it.

Remark 4 (Properties of the indirect predictor) Fitting S from the exogenous pair $\displaystyle ( \mathbf { r } , \mathbf { y } )$ via (20) avoids the openloop simulation issue in D.1: for every $\theta , \hat { \mathbf { S } } ^ { \theta } \in \mathcal { L } _ { p } ^ { S C }$ implies $\hat { \mathbf { y } } \in \ell _ { p }$ whenever $\mathbf { r } \in \ell _ { p } ,$ even if G is unstable. Regarding D.2, the predictor is driven by r, assumed independent $o f { \bf v } ,$ rather than by noise-corrupted measurements; as discussed after (20), this reduces the bias to second order but does not eliminate it. Consistency is thus limited to the linear case, where $\mathbf { y } = \mathbf { S } ( \mathbf { r } ) + ( \mathbf { S } \mathbf { K } + \mathbf { I } ) \mathbf { v }$ exactly and $\mathbf { S } \mathbf { K } + \mathbf { I } = ( \mathbf { I } - \mathbf { G } \mathbf { K } ) ^ { - 1 }$ is the output sensitivity of the true loop: with zero-mean finite-variance noise independent of r, output-error identification is consistent irrespective ofthe noise spectrum, provided the model class contains $\mathbf { s } ,$ the excitation identifies it, and the usual statistical regularity and global optimization conditions hold [16]. Direct methods, instead, can require a correctly specified noise model to avoid feedback-induced bias [45,30].

## 4.4 The Simulation Error Objective

A fundamental advantage of the unconstrained optimization problem (12) is that closed-loop stability is guaranteed entirely by the ICI parameterization, so the identification loss c can be freely designed to incorporate different modeling objectives without jeopardizing the stability of the learned model. For instance, the empirical loss could embed physics-informed penalties to enforce known physical constraints, state-dependent weightings to prioritize accuracy near critical equilibrium points or safety boundaries, or negative log-likelihoods to capture non-Gaussian noise profiles. In this work we adopt the standard Simulation Error Method (SEM): when the intended use of the model is simulation or control design, the multi-step simulation error is the natural fitting criterion and the standard choice for training deep networks on dynamical systems [38], whereas one-step criteria optimize more easily but transfer poorly to free-run simulation. We define the trajectory discrepancy c as the Mean Squared Error (MSE) between predicted and measured outputs, averaged over both the finite horizon $T$ and the $d _ { y }$ output coordinates:

$$
c \big ( \theta , \tau _ { T } \big ) : = \frac { 1 } { T d _ { y } } \sum _ { t = 0 } ^ { T - 1 } \| \hat { y } _ { t } - y _ { t } \| _ { 2 } ^ { 2 } ,\tag{21}
$$

where the sequence of predicted outputs $\hat { \mathbf { y } } _ { 0 : T - 1 }$ is generated according to either the direct or indirect identification strategy defined in the previous subsections. Substituting (21) into (16) yields an unconstrained least-squares problem for each strategy, efficiently minimized by gradient-based methods since $\breve { \mathbf { S } } ^ { \breve { \theta } }$ is differentiable.

## 5 PAC-Bayesian training for improved generalization

System identification must operate on a finite set of noisy closed-loop trajectories. This finite-sample setting introduces a generalization gap: a model that successfully minimizes the empirical risk on the training data may still perform poorly on unseen realizations of excitation and noise. The goal of this section is to quantify this gap using PAC-Bayesian theory, and to translate the resulting bound into a principled, generalization-aware training objective for the ICI operator S<sup>ˆ</sup> .

## 5.1 PAC-Bayes Generalization Bound

To certify the finite-sample reliability of the identified model, we seek to formally upper-bound the true expected loss (15) using its empirical estimate (16). We achieve this via the PAC-Bayesian framework, which casts the learning process as an update from a data-independent prior distribution $\mathcal { P }$ over the model parameters $\theta ,$ , to a data-dependent posterior distribution $\mathcal { Q }$ based on the observed dataset.

Within our system identification framework, both the prior $\mathcal { P }$ and the posterior $\mathcal { Q }$ are defined over the parameters, θ, of the ICI operator $\hat { \mathbf { S } } ^ { \theta }$ . Crucially, this directly integrates the PAC-Bayesian framework with the ICI architecture: because every $\boldsymbol { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ maps to an operator $\hat { S } ^ { \theta } \in \mathcal { L } _ { p } ^ { S C }$ , any model configuration sampled from the posterior is guaranteed by construction to be stabilized by the controller K.

PAC-Bayesian theory requires a bounded loss:

Assumption 1 The finite-horizon identification loss in (21) is upper-bounded by $C \in \mathbb { R } ^ { + } , \ i . e . , \ c ( \theta , \tau _ { T } ) \in [ 0 , C ) ,$ , for all $\pmb { \theta } \in \Theta : = \mathbb { R } ^ { d _ { \theta } }$ and $\tau _ { T }$ in the support of $\mathcal { D } _ { T }$

An unbounded loss is mapped to $[ 0 , C )$ via:

$$
\tilde { c } ( \theta , \tau _ { T } ) : = C \mathrm { t a n h } ( c ( \theta , \tau _ { T } ) / \gamma ) ,\tag{22}
$$

with $\gamma \in \mathbb { R } ^ { + }$ : the map is near-linear on $[ 0 , \gamma ]$ and saturates smoothly to $C ,$ so a suitable $\gamma$ keeps training in the linear regime. Throughout this section J and $\hat { J }$ are understood on the transformed loss ${ \tilde { c } } .$

We now bound the true identification risk J in (15).

Theorem 2 (Adapted from Theorem 2.7 in [2]) Consider the distribution of closed-loop trajectories $\mathcal { D } _ { T } ,$ a dataset $\mathbb { S } _ { T } ^ { N } = \{ \tau _ { T } ^ { n } \} _ { n = 0 } ^ { N - 1 } = \{ ( r _ { 0 : T - 1 } ^ { n } , u _ { 0 : T - 1 } ^ { n } , y _ { 0 : T - 1 } ^ { n } ) \} _ { n = 0 } ^ { N - 1 } \sim \mathcal { D } _ { T }$ , a prior distribution $\mathcal { P }$ independent of $\mathbb { S } _ { T } ^ { N }$ , and the identification loss $c .$ Under Assumption $I , f o { \bar { r } }$ every $\lambda > 0 ,$ confidence level $\delta \in ( 0 , 1 )$ , posterior distribution $\mathcal { Q } ,$ and model parameters $\theta \sim \mathcal { Q }$ , the inequality

$$
J ( \theta , \mathcal { D } _ { T } ) \leq \hat { J } ( \theta , \mathbb { S } _ { T } ^ { N } ) + \frac { 1 } { \lambda } \Bigl ( \ln \frac { d \mathcal { Q } } { d \mathcal { P } } ( \theta ) + \ln \frac { 1 } { \delta } \Bigr ) + \frac { \lambda C ^ { 2 } } { 8 N }\tag{23}
$$

holds with probability at least $1 - \delta$ over simultaneously sampling the N i.i.d. trajectories ofS<sup>N</sup> from $\mathcal { D } _ { T }$ and $\theta \sim \mathcal { Q } .$ The true and empirical risks, J and ${ \bf \widehat J } ,$ are defined in (15) and (16) respectively.

The bound combines the empirical risk, a log-density ratio measuring the deviation of Q from $\mathcal { P }$ at θ, and a θ- independent complexity term. It adapts Theorem 2.7 in [2] to time-series data by treating independent closed-loop trajectories as the i.i.d. samples, so that c compares entire predicted and observed sequences. Standard PAC-Bayesian bounds control the risk averaged over Q, i.e., the risk of a randomized predictor that redraws $\theta \sim \mathcal { Q }$ at every use, which is at odds with deploying one identified model. Theorem 2 is instead a disintegrated bound: it holds for a single parameter drawn once from the posterior, $\theta \sim \mathcal { Q }$ , matching the identification objective of deploying one model.

## 5.2 Optimal Posterior

Unlike Bayesian inference, $\mathcal { Q }$ need not come from Bayes theorem: we select it to minimize the expected value, over $\theta \sim \mathcal { Q }$ , of the upper bound in (23):

$$
\begin{array} { r l } & { \boldsymbol { \mathcal { Q } } ^ { * } : = \underset { \boldsymbol { \mathcal { Q } } } { \arg \operatorname* { m i n } } \underset { \boldsymbol { \theta } \sim \boldsymbol { \mathcal { Q } } } { \mathbb { E } } \left[ \hat { J } ( \boldsymbol { \theta } , \mathbb { S } _ { T } ^ { N } ) + \frac { 1 } { \lambda } \left( \ln \frac { d \boldsymbol { \mathcal { Q } } } { d \mathcal { P } } ( \boldsymbol { \theta } ) + \ln \frac { 1 } { \delta } \right) + \frac { \lambda C ^ { 2 } } { 8 N } \right] } \\ & { \quad = \underset { \boldsymbol { \mathcal { Q } } } { \arg \operatorname* { m i n } } \underset { \boldsymbol { \theta } \sim \boldsymbol { \mathcal { Q } } } { \mathbb { E } } \left[ \hat { J } ( \boldsymbol { \theta } , \mathbb { S } _ { T } ^ { N } ) + \frac { 1 } { \lambda } \ln \frac { d \boldsymbol { \mathcal { Q } } } { d \mathcal { P } } ( \boldsymbol { \theta } ) \right] } \\ & { \quad = \underset { \boldsymbol { \mathcal { Q } } } { \arg \operatorname* { m i n } } \underset { \boldsymbol { \theta } \sim \boldsymbol { \mathcal { Q } } } { \mathbb { E } } \left[ \hat { J } ( \boldsymbol { \theta } , \mathbb { S } _ { T } ^ { N } ) \right] + \frac { 1 } { \lambda } \mathrm { K L } \big ( \boldsymbol { \mathcal { Q } } \| \mathcal { P } \big ) , \qquad ( 2 4 ) } \end{array}
$$

where KL stands for the Kullback–Leibler divergence. The objective trades off, through λ, empirical risk against closeness to the prior (which can encode structural knowledge or regulate complexity). In the two limiting regimes, $\mathcal { Q } ^ { \ast }$ concentrates on the minimizers of J<sup>ˆ</sup> as $\lambda \to \infty$ , and ${ \mathcal { Q } } ^ { * } \to { \mathcal { P } }$ as $\lambda \to 0$

Corollary 2 (Lemma 1.1.3 in [8]) The minimizer of (24) is the Gibbs distribution, characterized by:

$$
\begin{array} { r } { \mathcal { Q } ^ { \ast } ( \pmb { \theta } ) = \mathcal { P } ( \pmb { \theta } ) e ^ { - \lambda \hat { J } ( \pmb { \theta } , \mathbb { S } _ { T } ^ { N } ) } / Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } ) , } \end{array}\tag{25}
$$

$$
\begin{array} { r } { Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } ) : = \underset { \theta \sim \mathcal { P } } { \mathbb { E } } e ^ { - \lambda \hat { J } ( \theta , \mathbb { S } _ { T } ^ { N } ) } , } \end{array}\tag{26}
$$

where $Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } )$ normalizes the distribution $\mathcal { Q } ^ { \ast }$

Substituting the optimal posterior (25) into (23) yields the tightest bound:

$$
J ( \theta , \mathcal { D } _ { T } ) \leq \frac { 1 } { \lambda } \Bigl ( \ln \frac { 1 } { \delta } - \ln Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } ) \Bigr ) + \frac { \lambda C ^ { 2 } } { 8 N } ,\tag{27}
$$

holding with probability at least $1 - \delta$ over simultaneously sampling $\mathbb { S } _ { T } ^ { N }$ and $\theta \sim \dot { \mathcal { Q } } ^ { * }$ . The term $- 1 / \lambda \ln Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } )$ ) lies in $[ \stackrel { \cdot } { 0 } , C ) \stackrel { \cdot } { }$ and decreases when the (data-independent) prior $\mathcal { P }$ assigns more mass to parameters with lower empirical risk (26): the bound tightens when $\mathcal { P }$ is “aligned” with $\mathbb { S } _ { T } ^ { N }$ The bound (27) holds for the optimal posterior $\mathcal { Q } ^ { \ast }$ and does not depend on the drawn θ. Since $\mathcal { Q } ^ { \ast }$ minimizes the expectation over $\theta \sim \mathcal { Q }$ of the right-hand side of (23), it yields the tightest instance of (23) in expectation over the posterior draw, though not necessarily for every realization of θ.

Tightening (27) amounts to tuning $\lambda > 0$ , which Theorem 2 leaves free. Since $- 1 / \lambda$ ln $Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } )$ ) requires an expectation (26) and lies in [0,C) under Assumption 1, we replace it by C and select λ by minimizing the resulting looser bound.

Proposition 1 The optimal $\lambda > 0$ that minimizes the upper bound on the right-hand side of (27) obtained by replacing $- 1 / \lambda$ ln $Z _ { \lambda } ( \mathcal { P } , \bar { \mathbb { S } } _ { T } ^ { N } )$ with C, is given by:

$$
\lambda ^ { * } = \frac { \sqrt { 8 N \mathrm { l n } ( 1 / \delta ) } } { C } .\tag{28}
$$

The same $\lambda ^ { * }$ is obtained by replacing $- 1 / \lambda \ln Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } )$ with the tighter bound $\mathbb { E } _ { \theta \sim \mathcal { P } } \hat { J } ( \theta , \mathbb { S } _ { T } ^ { N } )$ , which follows from Jensen’s inequality and is likewise independent of λ. Crucially, $\lambda ^ { * }$ does not depend on $\mathbb { S } _ { T } ^ { N }$ , as required for Theorem 2 to hold with $\lambda = \lambda ^ { * }$

Given $\mathcal { P }$ and $\mathbb { S } _ { T } ^ { N }$ , (25) and (28) fully specify the optimal posterior. In practice, hyperparameters of the prior (e.g., its width) and λ may also be selected from a finite, dataindependent set of m candidates by minimizing the bound: by a union bound over the candidates, (27) holds for the selected pair with probability at least $1 - \delta$ upon replacing δ with $\delta / m ,$ including in (28).

## 5.3 Sampling from the posterior

The optimal posterior $\mathcal { Q } ^ { * }$ in (25) is known only up to the normalization constant $Z _ { \lambda } ( \mathcal { P } , \mathbb { S } _ { T } ^ { N } )$ , and sampling from unnormalized distributions is generally difficult: Markov chain Monte Carlo methods are applicable in principle, but are typically expensive for large datasets [22]. Following [3], we adopt instead a variational approximation, in which $\mathcal { Q } ^ { \ast }$ is replaced by a tractable distribution that is easy to sample from. Specifically, we employ Stein Variational Gradient Descent (SVGD) [29], which represents the posterior by $k \in \mathbb N$ particles $\big \{ \phi _ { 1 } , \ldots , \phi _ { k } \big \}$ . Each particle $\phi _ { \kappa } , \kappa = 1 , \ldots , k ,$ collects the parameters of one model, $\mathrm { i } . \mathrm { e } . , \phi _ { \kappa } \in \Theta ;$ ; the particles are initialized from the prior and iteratively updated by gradient descent, so that their empirical density in parameter space approximates $\mathcal { Q } ^ { \ast }$ . Since the updates are gradientbased, SVGD integrates naturally with our unconstrained parameterization. After training, $\mathcal { Q } ^ { * }$ is approximated by $\hat { \mathcal { Q } } _ { k } ^ { \mathrm { ~ ~ } } .$ the uniform distribution over the resulting particles, so that sampling from the approximate posterior amounts to selecting one particle uniformly at random. The use of SVGD for approximating Gibbs posteriors arising in PAC-Bayesian methods is well established, e.g. [41,6] and we refer the reader to [29] for further implementation details. One caveat is worth noticing: Theorem 2 and its specialization (27) certify a single draw from Q<sup>∗</sup>, whereas SVGD provides the particle approximation $\hat { \mathcal { Q } } _ { k }$ . The value of (27) depends only on $\mathcal { P }$ and $\mathbb { S } _ { T } ^ { N }$ , not on the particles, and (23) cannot be applied to $\hat { \mathcal { Q } } _ { k }$ directly, since a discrete particle distribution is not absolutely continuous with respect to P. The certificate therefore refers to $\mathcal { Q } ^ { \ast }$ ; the deployed particle is an approximate draw from it, and the discrepancy between $\mathcal { Q } ^ { \ast }$ and $\hat { \mathcal { Q } } _ { k }$ is not quantified by the bound.

## 6 Numerical experiments

We evaluate the ICI framework across four scenarios validating: (i) the identification of open-loop unstable plants with guaranteed closed-loop stability, (ii) accuracy against standard baselines on stable systems, (iii) context-aware identification across environment families, and (iv) the generalization benefits of PAC-Bayesian training under data scarcity. Section 6.1 describes the benchmark systems, the compared strategies, the evaluation metric, and the architectures, sample sizes, and signal levels of each experiment; layer widths, learning rates, whitening, training safeguards, per-seed protocols, and the complete result grids are in the public repository at https: $/ / { \tt g i }$ thub.com/DecodEPFL/Nonlinear\_ system\_identification\_PAC.git.

## 6.1 Experimental Setup

Target Systems. Throughout the experiments, we benchmark our methods on two distinct dynamical systems:

(1) A nonlinear, open-loop unstable system defined by the state-space equations:

$$
\begin{array} { r } { \left\{ { x _ { t + 1 } } = 3 x _ { t } + \operatorname { t a n h } ( x _ { t } ) + u _ { t } , \quad x _ { 0 } = \bar { x } \right. } \\ { y _ { t } = x _ { t } + \nu _ { t } } \end{array}\tag{29}
$$

Being open-loop unstable, the system is operated in closed loop during data collection, stabilized by the nonlinear controller $K _ { t } ( y _ { 0 : t } ) = - 3 . 5 y _ { t } + 0 . 3 \operatorname { t a n h } ( y _ { t } )$ This controller satisfies the hypotheses of Theorem 1 globally: $\mathbf { K ( 0 ) } = \mathbf { 0 } .$ , and $\dot { | } K ^ { \prime } ( y ) | = | - 3 . 5 +$ $0 . 3 \mathrm { s e c h } ^ { 2 } ( y ) | \leq 3 . 5$ for every y, so K is incrementally $\ell _ { p }$ -stable with incremental gain $\hbar \kappa = 3 . 5$ . Together with $\mathbf { K } ( \mathbf { 0 } ) \in \ell _ { p }$ this yields $\mathbf { K } \in \mathcal { L } _ { p } ^ { C }$ . The system input is then $u _ { t } = K _ { t } ( y _ { 0 : t } ) + r _ { t }$

(2) A stable planar robot tracking a target, modeled as a point mass $m \in \mathbb { R } _ { + } \ [ 1 8 ]$ . The state comprises position $\bar { \boldsymbol { x } } _ { 1 , t } \in \mathbb { R } ^ { 2 }$ and velocity $x _ { 2 , t } \in \mathbb { R } ^ { 2 }$ , the output is a noisy position measurement, and the input is a planar force:

$$
\{ \begin{array} { l l } { [ x _ { 1 , t + 1 } ] = [ x _ { 1 , t } ] + T _ { s } [ \qquad x _ { 2 , t }  } \\ {  x _ { 2 , t + 1 } ] + x _ { 2 , t } ] + T _ { s } [ \qquad ( x _ { 1 , t } , x _ { 2 , t } ) + u _ { t } ) \mathrm { { d } } , } \\ { y _ { t } = x _ { 1 , t } + \nu _ { t } , } \end{array} 
$$

where $T _ { s } > 0$ is the sampling time, and the function $C ( x _ { 1 , t } , x _ { 2 , t } ) = b _ { 1 } ( x _ { 1 , t } ) x _ { 2 , t } + \bar { b } _ { 2 } \| \bar { x } _ { 2 , t } \| _ { 2 } x _ { 2 , t }$ , with $0 < b _ { 2 } <$ $b _ { 1 } ( x _ { 1 , t } ) \forall x _ { 1 , t } \in \mathbb { R } ^ { 2 }$ , models nonlinear drag forces, such as air or water resistance. The coefficient $b _ { 1 } ( x _ { 1 , t } )$ can represent either a uniform environment or a positiondependent spatial friction map. The robot is subject to nonlinear drag. To regulate its position toward the target $\boldsymbol { y } ^ { * } = \mathbf { 0 } \in \mathbb { R } ^ { 2 }$ , we apply a static proportional controller with gain matrix $\begin{array} { r } { \bar { K } = \mathrm { d i a g } ( \kappa _ { 1 } , \bar { \kappa _ { 2 } } ) } \end{array}$ , where $\kappa _ { 1 } , \kappa _ { 2 } > 0 .$ The control law is $K _ { t } ( y _ { 0 : t } ) \bar { = } K ( y ^ { \ast } - y _ { t } )$ , and the system input is $u _ { t } = K _ { t } ( y _ { 0 : t } ) + r _ { t }$

Unless otherwise specified, the reference excitation $r _ { t }$ is independently sampled at each time step from a zero-mean Gaussian distribution, $r _ { t } \sim \mathcal { N } ( 0 , \sigma _ { r } ^ { 2 } I )$ , and the output is corrupted by measurement noise $\nu _ { t } \sim \mathcal { N } ( 0 , \sigma _ { \nu } ^ { 2 } I )$ ; both are $\ell _ { p }$ on the finite horizon T of each trajectory, so Theorem 1 applies. While persistent excitation is most commonly defined and analyzed in linear settings, in this nonlinear setting Gaussian excitation qualitatively promotes exploration of an informative range of the state space [30].

Baselines. To highlight the benefits of the ICI stability certificate, we benchmark our approaches (both the indirect strategy (20) and the direct counterpart (17)) against two uncertified baselines that minimize the simulation error (21). Instead of incorporating structural guarantees, these baselines parameterize G<sup>ˆ</sup> as a generic neural network, discarding the prior knowledge that the true plant is stabilized by K. The two baseline strategies differ fundamentally in how the networks are simulated and, consequently, in which data drive the predictor:

• Direct OE (Direct Output Error). This is the standard direct identification baseline, widely used in neural system identification [38,40]. A generic neural model $\hat { \mathbf { G } } _ { \theta }$ is trained to minimize the simulation error on the recorded input u:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d _ { \theta } } } } & { \displaystyle \frac { 1 } { N } \sum _ { n = 0 } ^ { N - 1 } \left[ \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { d _ { y } } \| \hat { y } _ { t } ^ { n } - y _ { t } ^ { n } \| _ { 2 } ^ { 2 } \right] } \\ { \mathrm { s . t . } } & { \hat { y } _ { t } ^ { n } = \hat { G } _ { t } ^ { \theta } \big ( u _ { 0 : t - 1 } ^ { n } \big ) , } \\ & { \forall t = 0 , \ldots , T - 1 , \ \forall n = 0 , \ldots , N - 1 . } \end{array}\tag{30}
$$

• CLOE (Closed-Loop Output Error). This strategy [28,27] simulates the closed-loop system driven by the exogenous reference r. By placing the neural model $\hat { \mathbf { G } } _ { \theta }$ in feedback with the known controller K, thus closing the loop around the model’s own predictions, CLOE avoids feeding the noisy recorded input into the predictor:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d _ { \theta } } } } & { \displaystyle \frac { 1 } { N } \sum _ { n = 0 } ^ { N - 1 } \left[ \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { d _ { y } } | | \hat { y } _ { t } ^ { n } - y _ { t } ^ { n } | | _ { 2 } ^ { 2 } \right] } \\ { \mathrm { s . t . } \hat { y } _ { t } ^ { n } = \hat { G } _ { t } ^ { \theta } \big ( r _ { 0 : t - 1 } ^ { n } + K _ { 0 : t - 1 } \big ( \hat { y } _ { 0 : t - 1 } ^ { n } \big ) \big ) , } & { } \\ { \forall t = 0 , \ldots , T - 1 , \forall n = 0 , \ldots , N - 1 . } \end{array}\tag{31}
$$

Evaluation metrics. All identification strategies are evaluated using the FIT metric on a fixed, noise-free test set, common to every method within each training condition, in both open-loop (OL) and closed-loop (CL) settings; measurement noise affects the training data only, so the test isolates the quality of the identified dynamics.

Goodness of fit is computed per test trajectory n as $\mathrm { F I T } ^ { n } = 1 0 0 \bigl ( 1 - \| y _ { 0 : T _ { t e s t } - 1 } ^ { n } - \hat { y } _ { 0 : T _ { t e s t } - 1 } ^ { n } \| _ { 2 } / \| y _ { 0 : T _ { t e s t } - 1 } ^ { n } - \bar { y } ^ { n } \| _ { 2 } \bigr )$ [%], where $y _ { t } ^ { n }$ is the noise-free true output, $\hat { y } _ { t } ^ { n }$ the mode $z ,$ and ${ \bar { y } } ^ { n }$ the temporal mean of trajectory n, subtracted at every t; FIT ranges in (−∞,100] [38]. For open-loop-unstable plants, the open-loop comparison is shown graphically (on a logarithmic scale); closed-loop FIT/RMSE are computed over the bounded rollouts only, with the divergence rate reported separately.

We report the mean ± one standard deviation across the $N _ { t e s t }$ test trajectories; training seeds are fixed and all protocols are deterministic given the seed, so every number is exactly reproducible from the public repository.

## Models and datasets.

Table 1 collects the model classes, sample sizes, and signal levels of the four experiments.

## 6.2 Feedback-Stabilized Unstable Scalar System

This experiment validates on the unstable plant (29) that (a) every identified ICI model $\hat { \mathbf { G } } _ { \mathrm { I C I } }$ is guaranteed to be stabilized by the data-collection controller, whereas a model returned by an uncertified method may not be, and (b) using the indirect approach training does not diverge and recovers the open-loop unstable dynamics. The two ICI strategies use a contractive SSM as the operator $\hat { \bf S } ,$ while the two baselines use a vanilla RNN with unbounded ReLU activations, a class chosen because saturating networks can’t represent unstable dynamics and would fail the second question by construction. The baselines are given more capacity than the ICI models, so that no conclusion can be attributed to a capacity advantage of the latter.

Results. Table 2 summarizes the structural properties of each method, while Figure 4 shows the open-loop behavior. The analysis yields two main observations:

(a) Only the ICIframework guarantees closed-loop stability. No ICI model destabilized the loop in any of the $S = 4 0$ runs, at any noise level, as Theorem 1 guarantees. Among the uncertified methods, Direct OE returned a model that K fails to stabilize in 25% of the runs, with the loop diverging from a large fraction of the initial conditions. CLOE with the ReLU-RNN instead generates an unstable closed loop in 2.5% of the runs.

(b) Indirect ICI ensures well-posed training and identifies unstable dynamics. As shown in Figure 4, the true plant exhibits unstable open-loop trajectories, which are accurately reproduced by indirect ICI. Both direct strategies fail in every run: as anticipated by Drawback D.1, they simulate the candidate model on the recorded input, and for unstable models the finite-cost region shrinks exponentially around the true plant [40], so unconstrained gradient descent diverges. CLOE avoids open-loop simulation, but its rollout stays bounded only while gradient updates keep the model stabilized by K, which nothing guarantees. For a fair comparison, we therefore equip these methods with heuristic training safeguards (horizon curricula, gradient clipping, learning-rate backoff [40]); with them, the direct approaches converge, but to stable surrogates that miss the open-loop instability. In contrast, indirect ICI restricts the search space to models that K certifiably stabilizes, keeping every training rollout bounded and recovering the unstable dynamics without such heuristics.

![](images/a55693df8091cdec4dbb4ff27519a24f56b088926b7005b80d820698d723ec54.jpg)  
Fig. 4. Unstable scalar plant. Open-loop trajectories (log scale, 8 steps) from $\bar { x } = 0 . 1$ of the models returned by $S = 4 0$ independent training runs (solid = median over runs, shaded = 10–90 percentile band); the indirect strategies reproduce the diverging behavior, while both direct strategies return an inert surrogate.

## 6.3 Point-Mass Robot under Partial Observation

This experiment evaluates the accuracy of the ICI strategies against the generic-network baselines on the stable point-mass robot of Section 6.1 in a uniform environment $( b _ { 1 } ( x _ { 1 , t } ) \equiv b _ { 1 } )$ , under partial observation: only the (noisy) positions are measured, the velocities remain hidden. We evaluate the closed-loop identification performance through two distinct parameter sweeps: varying the reference excitation level $\sigma _ { r }$ on noise-free data, and varying the level of colored measurement noise $\sigma _ { \nu }$ under constant excitation. The two ICI strategies share the same certified deep SSM as the operator S<sup>ˆ</sup> in (17) and (20), and the two baselines share a generic LSTM as the model $\hat { \mathbf { G } } _ { \theta }$ in (30) and (31).

Results. Figure 6 visualizes both sweeps, with their corresponding FIT measures detailed in Table 3. The analysis yields three main observations.

(a) Informativity at low excitation. At low excitation (e.g., $\sigma _ { r } = 1 )$ the input is dominated by the feedback loop (u ≈ $\mathbf { K } ( \mathbf { y } ) )$ , and this lack of informativity makes the direct methods fail in open loop (FIT −28 for direct ICI) despite passing the regulation test. As the excitation grows $( \sigma _ { r } \geq 5 )$ , informativity is restored and the gap between the methods closes.

(b) Bias of direct methods under colored noise. Feedback injects noise into the control input via $\mathbf { K } ( \mathbf { v } )$ , so the regressor loses orthogonality with the residual and direct identification is biased (Drawback D.2) [16]. In Figure 6 (right, $\sigma _ { r } = 1 0 ) .$ , as σ grows from 0.1 to 2 the open-loop FIT of direct ICI collapses $( 8 2 . 9  - 7 . 0 )$ while indirect ICI degrades gracefully (86.7→60.9), its regressor r being noiseindependent by construction. We examine the Pearson correlation between the model’s simulation error and the actual measurement noise. For the u-driven models, this correlation is significant (0.28 for direct ICI, 0.21 for Direct OE), confirming that they erroneously absorb the noise into the learned dynamics.

Table 2  
Model classes and data of the four experiments. N, T are the number and length of the training trajectories, $\sigma _ { r } , \sigma _ { \nu }$ the excitation and measurement-noise levels, and $N _ { t e s t } , T _ { t e s t } ^ { \bar { } }$ the number and length of the test trajectories. Initial conditions are $\bar { x } \sim \mathcal { U } ( - 2 , 2 )$ (training) and $\bar { x } \sim \mathcal { U } ( - 4 , 4 )$ (redeployment test) in Section $_ { 6 . 2 }$ , and $p _ { 0 } \sim \mathcal { U } ( [ - 1 0 , 1 0 ] ^ { 2 } )$ with zero initial velocity elsewhere. The redeployment test of Section 6.2 is repeated at $\sigma _ { \nu } \in \{ 0 , 0 . 3 , 0 . 6 \}$ for each of the $S = 4 0 \mathrm { \ t r a i n i n g }$ runs per strategy.
<table><tr><td>Section</td><td>System</td><td>Observation</td><td>ICI operator  $\hat { \bf S }$ </td><td>Baseline class</td><td>N</td><td>T</td><td> $\sigma _ { r }$ </td><td> $\sigma _ { \nu }$ </td><td> $N _ { t e s t }$ </td><td> $T _ { t e s t }$ </td></tr><tr><td>6.2</td><td>1</td><td>full state</td><td>deep SSM</td><td>ReLU RNN</td><td>40</td><td>100</td><td>0.5</td><td>0.1</td><td>100</td><td>200</td></tr><tr><td>6.3</td><td>2</td><td>positions</td><td>deep SSM</td><td>LSTM</td><td>200</td><td>100</td><td> $\{ 0 . 5 , \ldots , 2 0 \}$ </td><td> $\{ 0 , \ldots , 2 \} , \mathtt { A R } ( 1 )$ </td><td>100</td><td>100</td></tr><tr><td>6.4</td><td>2</td><td>full state</td><td>deep SSM</td><td>deep SSM, no context</td><td>60</td><td>50</td><td>20</td><td>0.1</td><td>100/20</td><td>50</td></tr><tr><td>6.5</td><td>2</td><td>positions</td><td>REN</td><td></td><td>1-64</td><td>50</td><td>20</td><td>0.5</td><td>100</td><td>50</td></tr></table>

Unstable scalar plant (29): identifiability of the unstable plant, certified closed-loop stability with the known controller, and fraction of the $S = 4 0$ independent training runs whose model the datacollection controller fails to stabilize $( \sigma _ { \nu } = 0 . 3 )$
<table><tr><td>Strategy</td><td>Identifiability of unstable G</td><td>Certified CL stability</td><td>Unstable CL runs</td></tr><tr><td>Indirect ICI</td><td>V</td><td>√</td><td>0%</td></tr><tr><td>Direct ICI</td><td>一</td><td>√</td><td>0%</td></tr><tr><td>CLOE</td><td>√</td><td>一</td><td>2.5%</td></tr><tr><td>Direct OE</td><td>一</td><td>一</td><td>25%</td></tr></table>

![](images/635a16146462eb63f601bcda45703e61f59b8ed749b21a7bfd8939a10dab9ef1.jpg)  
Fig. 5. Out-of-distribution excitation waveforms (square, multisine) at the training amplitude; training cell $\sigma _ { r } = 2 0 , \ \sigma _ { \nu } = 1$ $\operatorname { A R } ( 1 ) ;$ channel $p _ { y }$ (the analogous $p _ { x }$ figure and all other training cells are in the repository). Indirect ICI transfers to the unseen shapes more accurately.

(c) Generalization to out-of-distribution excitation. To probe whether the true open-loop dynamics were captured, the identified models are evaluated on unseen input waveforms (square wave, multisine) at the training amplitude (Figure 5).

![](images/caa839b7043b6a005e7a63884d571259d97eb92953c98d80b075e8d3e6d78b4e.jpg)  
Fig. 6. Point-mass robot: open-loop (top) / closed-loop (bottom) FIT, mean±std over 100 test trajectories. Left: noise-free training vs. excitation $\sigma _ { r }$ (observation (a)); right: AR(1) noise sweep at $\sigma _ { r } = 1 0$ (observation (b)).

The indirect ICI model tracks the true response; the direct methods and the generic baselines exhibit drift and unphysical oscillations.

## 6.4 Generalization across environments via context-aware ICI

This experiment evaluates the context-enriched parameterization of Section 3.3 across a family of environments. We consider the stable point-mass robot of Section 6.1 with full-state measurements $y _ { t } = x _ { t } + \nu _ { t }$ and the proportional controller on the position components; unlike Section $6 . 3 ,$ the robot now operates in a family of environments with position-dependent friction: in environment $e ,$ the nominal friction force is scaled by $\begin{array} { r } { s _ { e } ( p ) = 1 + \sum _ { i = 1 } ^ { 3 } w _ { i } \exp \bigl ( - \| p - \| _ { B } \bigr ) } \end{array}$ $c _ { i } \| _ { 2 } ^ { 2 } / ( 2 \sigma _ { i } ^ { 2 } ) )$ , a superposition of random “sticky patches” with intensities $w _ { i } \sim \mathcal { U } ( 2 , 8 )$ , radii $\sigma _ { i } \sim \mathcal { U } ( 2 , 4 )$ , and centers $c _ { i } \sim \mathcal { U } ( [ - 9 , 9 ] ^ { 2 } )$ . Two of these fields are shown in Figure 9: the test trajectories repeatedly traverse the patches, so the field shapes the dynamics along the whole data record.

Two indirect-ICI models share the same certified deep SSM core, with and without access to the local drag scale as context. The contextual model instantiates (13), with the deep SSM acting as the stable feature generator $\hat { \mathbf { S } } _ { p }$ :

Point-mass robot under partial observation (Section 6.3): open-loop and closed-loop FIT [%], mean ± std over the 100 test trajectories. Top: nominal excitation $\scriptstyle \mathbf { \sigma } _ { \sigma _ { r } = 1 0 }$ and AR(1) measurement noise $( \rho \mathrm { = } 0 . 9 )$ of increasing level $\sigma _ { \nu } .$ Bottom: noise-free training $( \sigma _ { \nu } { = } 0 )$ and increasing excitation level σ<sub>r</sub>; at $\sigma _ { r } { = } 0 . 5$ the LSTM baselines fall below −400% open-loop FIT. Colored cells indicate the best result per column (ties are highlighted in both entries). Both sweeps are shown in Figure 6; the complete grids, including $\sigma _ { r } \in \{ 0 . 5 , 2 0 \}$ , are in the repository.
<table><tr><td rowspan="2">Strategy</td><td colspan="2"> $\sigma _ { \nu } { = } 0 . 1$ </td><td colspan="2"> $\sigma _ { \nu } { = } 0 . 5$ </td><td colspan="2"> $\sigma _ { \nu } = 1$ </td><td colspan="2"> $\sigma _ { \nu } = 2$ </td></tr><tr><td>OL</td><td>CL</td><td>OL</td><td>CL</td><td>OL</td><td>CL</td><td>OL</td><td>CL</td></tr><tr><td>Direct OE (LSTM)</td><td> $7 2 . 3 \pm 1 3 . 3$ </td><td> $8 3 . 0 \pm 7 . 2$ </td><td> $6 6 . 3 \pm 1 7 . 2$ </td><td> $8 0 . 7 \pm 7 . 9$ </td><td> $4 5 . 9 \pm 2 2 . 5$ </td><td> $7 2 . 4 \pm 1 1 . 9$ </td><td> $- 3 6 . 2 \pm 5 5 . 0$ </td><td> $4 3 . 5 \pm 3 0 . 7$ </td></tr><tr><td>CLOE (LSTM)</td><td> $7 8 . 9 \pm 1 0 . 6$ </td><td> $8 7 . 0 \pm 5 . 4$ </td><td> $7 0 . 2 \pm 1 6 . 5$ </td><td> $8 2 . 8 \pm 8 . 4$ </td><td> $3 9 . 1 \pm 3 4 . 1$ </td><td> $7 7 . 1 \pm 1 4 . 5$ </td><td> $- 8 6 . 1 \pm 1 2 3 . 0$ </td><td> $3 0 . 3 \pm 5 4 . 8$ </td></tr><tr><td>Direct ICI (SSM)</td><td> $8 2 . 9 \pm 6 . 5$ </td><td> $9 2 . 1 \pm 3 . 1$ </td><td> $7 9 . 0 \pm 6 . 8$ </td><td> $9 1 . 7 \pm 3 . 0$ </td><td> $6 2 . 0 \pm 9 . 7$ </td><td> $9 0 . 2 \pm 3 . 0$ </td><td> $- 7 . 0 \pm 3 3 . 6$ </td><td> $8 7 . 8 \pm 4 . 3$ </td></tr><tr><td>Indirect ICI (SSM)</td><td> $8 6 . 7 \pm 5 . 1$ </td><td> $9 2 . 0 \pm 4 . 4$ </td><td> $8 4 . 3 \pm 6 . 0$ </td><td> $9 1 . 7 \pm 3 . 7$ </td><td> $7 4 . 0 \pm 1 0 . 0$ </td><td> $8 9 . 6 \pm 3 . 9$ </td><td> $6 0 . 9 \pm 1 5 . 7$ </td><td> $8 2 . 7 \pm 6 . 1$ </td></tr><tr><td></td><td colspan="2"> $\sigma _ { r } { = } 1$ </td><td colspan="2"> $\sigma _ { r } { = } 2$ </td><td colspan="2"> $\sigma _ { r } { = } 5$ </td><td colspan="2"> $\sigma _ { r } { = } 1 0$ </td></tr><tr><td>Strategy</td><td>OL</td><td>CL</td><td>OL</td><td>CL</td><td>OL</td><td>CL</td><td>OL</td><td>CL</td></tr><tr><td>Direct OE (LSTM)</td><td> $- 1 7 0 . 1 \pm 1 1 8 . 6$ </td><td> $8 6 . 0 \pm 8 . 9$ </td><td> $- 6 4 . 9 \pm 7 1 . 3$ </td><td> $8 3 . 3 \pm 1 0 . 1$ </td><td> $4 5 . 1 \pm 3 2 . 1$ </td><td> $8 6 . 1 \pm 6 . 0$ </td><td> $7 6 . 0 \pm 1 2 . 3$ </td><td> $8 5 . 1 \pm 4 . 9$ </td></tr><tr><td>CLOE (LSTM)</td><td> $- 1 3 6 . 4 \pm 1 3 5 . 5$ </td><td> $8 9 . 7 \pm 4 . 1$ </td><td> $- 5 9 . 5 \pm 1 0 7 . 3$ </td><td> $8 9 . 7 \pm 5 . 9$ </td><td> $6 9 . 3 \pm 2 0 . 1 $ </td><td> $9 1 . 3 \pm 3 . 3$ </td><td> $8 2 . 3 \pm 1 0 . 1$ </td><td> $9 0 . 1 \pm 4 . 8$ </td></tr><tr><td>Direct ICI (SSM)</td><td> $- 2 7 . 6 \pm 2 8 . 2$ </td><td> $9 5 . 3 \pm 3 . 6 $ </td><td> $6 8 . 6 \pm 1 0 . 6$ </td><td> $9 5 . 4 \pm 3 . 9$ </td><td> $7 8 . 9 \pm 7 . 0$ </td><td> $9 5 . 6 \pm 2 . 5$ </td><td> $8 3 . 2 \pm 6 . 7$ </td><td> $9 2 . 3 \pm 3 . 0$ </td></tr><tr><td>Indirect ICI (SSM)</td><td> $6 4 . 4 \pm 1 5 . 6$ </td><td> $9 6 . 5 \pm 2 . 6 $ </td><td> $7 8 . 7 \pm 9 . 2$ </td><td> $9 6 . 1 \pm 2 . 6$ </td><td> $8 6 . 3 \pm 5 . 6$ </td><td> $9 5 . 5 \pm 2 . 3$ </td><td> $8 7 . 0 \pm 5 . 0$ </td><td> $9 2 . 0 \pm 4 . 4$ </td></tr></table>

• SSM-no-context receives r without terrain information; • SSM-context uses a context-only mixer producing A<sub>t</sub> ∈ $\mathbb { R } ^ { d _ { y } \times d _ { f } }$ , applied to the core features and rescaled so that $\| A _ { t } \| _ { F } \leq 1$ . This also bounds its induced Euclidean norm, as required by Section 3.3.

During training, the context sequence is refreshed once per epoch by evaluating the terrain map at the model’s predicted positions. During deployment, it is recomputed sequentially as $\hat { \zeta } _ { t } = s _ { e } ( \hat { p } _ { t } )$ , where $\hat { p } _ { t }$ is the position component of the predicted output. The mixer uses context with a one-step delay, so $\hat { \zeta } _ { t }$ affects predictions only from step t + 1, preserving strict causality. The terrain map is assumed known, but no measured or recorded test trajectory is used to construct context.

![](images/7898e9ea2d853f5c8868c00282ebd4c0a56ad03e0365b5a76f96cb0f8f2d3878.jpg)  
Fig. 7. Distribution of the open-loop per-trajectory FIT for the two models (box = median and interquartile range, whiskers = Tukey, ⋄ = mean; all seeds and test trajectories pooled). Columns vary the training regime (Case A, a single field / Case B, a family of fields) and rows the test set (seen fields, top / unseen fields, bottom); the y-axis is FIT in percent. The contextual model dominates the uncontextualized one in every panel, and under family training its seen- and unseen-field distributions nearly coincide.

Two training regimes share a budget of 60 trajectories: for Case A the training dataset contains trajectories collected on a single field, while in Case B the training trajectories are collected from 12 fields; both are tested on the training fields and on 6 held-out unseen fields.

Results. The open-loop identification accuracy is illustrated in Figures 7–8, with comprehensive per-environment tables available in the public repository. Bound (14), together with T.1, certifies stability of the identified closed-loop models. All evaluated closed-loop rollouts remained bounded across models, seeds, and fields. From these experiments, two primary conclusions emerge:

rameter count from 2793 to 15× (42249) does not improve unseen-field FIT (36.9–40.7%), consistent with a limitation due to missing environment information. Figure 8 shows the resulting prediction errors inside the strong-friction patches.

(a) Context distinguishes environment-specific responses. The initial state and excitation $( x _ { 0 } , \mathbf { r } )$ do not specify the environment, so a predictor receiving only these inputs must fit a common response across fields. Under family training, the no-context model attains similar open-loop FIT on seen and unseen fields (42.4% and 42.2%). Increasing its pa-(b) Context enables generalization to unseen fields without retraining. Under family training, the contextual model reaches $8 6 . 6 \pm 6 . 7 \%$ unseen-field FIT against 42.2 ± 24.5% without context, an improvement of 44.4 percentage points with a smaller reported standard deviation. Its unseen-field score is close to the seen-field score (86.6% versus 87.0%), showing generalization to the tested unseen environments. This transfer uses the same learned parameters, with each new terrain map available for context evaluation.

![](images/473a958a990a2d265d58ec9a6b781bdfd7cc988c8c5bd07c7c5c91fc0b7a9d5e.jpg)

Fig. 8. Case B, open-loop response on a seen (left) and unseen (right) field: $p _ { x } \ ( \mathrm { t o p } ) , \nu _ { x }$ (bottom); shaded = strong friction $( s _ { e } > 2 . 5 )$ . The no-context prediction departs from the truth inside the patches, while the contextual model tracks it; the analogous $p _ { y } / \nu _ { y }$ figures are in the public repository.  
![](images/4c28fbd2b11814a2b32b140101d7a2f8b6dc4a396f31546999b846fac45fa7ec.jpg)  
Fig. 9. Case B: true open-loop test trajectories (black) and SSM– context rollouts (dashed) over the friction field $s _ { e } ( p )$ (darker = stickier), on a seen (left) and unseen (right) field, under the causal predicted-position context. Dots mark the initial positions.

## 6.5 PAC-Bayesian Identification under Data Scarcity

This experiment considers the regime in which the PAC-Bayesian framework of Section 5 is most useful, namely scarce and noisy data with no validation set, where ERM does not merely add variance but memorizes the measurement noise.

On the point-mass robot example, we demonstrate three key properties: (a) under data scarcity ERM overfits and destroys a good prior, whereas the SVGD approximation of the Gibbs posterior over the same ICI parameterization generalizes and preserves it; (b) Theorem $^ 2$ provides a non-vacuous generalization certificate, spending no held-out data, that tightens with the sample size $N ;$ and (c) this certificate requires only an informative prior, which may be supplied by crossdomain transfer or, in the absence of a source system, obtained from data split.

In this experiment, we apply the transformation in (22) to the identification loss, setting $C = 1$ and γ = 10: every risk bound then takes values in [0,1) and is vacuous as soon as it reaches 1. The identified operator and every posterior particle are contractive RENs, certified closed-loop stable by

Theorem 1. Four estimators share the same parameterization: the un-adapted prior mean, ERM from scratch, ERM fine-tuned from the prior and one particle drawn uniformly from the SVGD approximation of the Gibbs posterior (25) (Section 5.3). The prior parameters $\theta _ { \mathrm { p r e } }$ are obtained by ERM pretraining on (20) using trajectories from a nominal plant with uniform linear drag; at deployment the plant has drifted to a heavier drag, modeling wear, and must be reidentified from only $N \in \{ 1 , \ldots , 6 4 \}$ noisy trajectories, with no validation set. Being trained on a different plant, $\theta _ { \mathrm { p r e } }$ is independent of the target sample, so Theorem 2 applies with no data splitting. The prior distribution is the per-parameter Gaussian $\mathcal { N } \big ( \theta _ { \mathrm { p r e } , i } ^ { - } , ( \alpha \dot { | } \theta _ { \mathrm { p r e } , i } | + 1 0 ^ { - 3 } ) ^ { 2 } \big )$

Results. Figures 10–11 report three complementary quantities. Figure 10 (left) shows the PAC-Bayes certificate (27) of a single draw from the exact Gibbs posterior $\mathcal { Q } ^ { \ast } ;$ Figure 10 (right) shows the empirical risk (16) computed on c rather than on the transformed loss ˜c (22), against noisy outputs; and Figure 11 shows deployment FIT against the noisefree plant. Only the first quantity is certified by Theorem 2; the latter two are complementary empirical diagnostics. The complete numerical table is provided in the repository. The analysis yields three main observations.

(a) At scarce data ERM memorizes the noise; the SVGD estimator does not. For $N \leq 4$ both ERM models drive their training MSE below the noise floor $\sigma _ { \nu } ^ { 2 }$ , fitting the specific realization, while their test error explodes and their deployment FIT drops below the initial pre-trained prior (67.7% vs. 86.6% at N = 4; Figure 11). Fine-tuning from the good prior overfits just the same, isolating the benefit as the KL anchoring of (24) rather than a good initialization: the SVGD estimator, optimizing the same trajectories, keeps its training error at the floor and its FIT at the prior’s level, and is unaffected even at $N = 1$

(b) The SVGD estimator performs well at scarce data; the Gibbs bound becomes non-vacuous. Across N (Figure 11) the posterior holds an almost constant ≈85% FIT, whereas the ERM starts far below the prior, catches up around $N = 8$ and overtakes only as the sample size grows. This plateau is the price of anchoring to the prior, and it is exactly what protects the posterior where unconstrained fitting ruins ERM. The certificate (27) behaves consistently (Figure 10): vacuous at the smallest $N ,$ it becomes informative from $N = 4$ and tightens with more data, at the $\sqrt { N }$ rate of $\lambda ^ { * } ;$ the point where it stops being vacuous thus marks the onset of the useful regime. The bound is fully computed from data (no oracle term) and it holds with probability $\ge 1 - \delta ( \delta = 0 . 1 )$ Every SVGD particle is certified closed-loop stable by Theorem 1, whereas the loss certificate applies to an exact draw from $\mathcal { Q } ^ { * }$ . The approximation error for the deployed particle is not covered (Section 5.3). Neither guarantee uses held-out data.

The prior width α trades deployment fit against certification: with a wide prior (Figure 11, pink) the posterior detaches from $\theta _ { \mathrm { p r e } }$ and reaches a higher FIT, but a wide prior rarely samples low-loss models, so the bound turns vacuous, which is precisely why the selected configuration keeps α small.

![](images/20eb6ce56d4e67a9632e1c18389a204509361aa2915204147e29d8ddfd799011.jpg)

![](images/7014ad59e8352bbc8cb3116be7b69f24896bbd233ec7e7878c081158bebbad81.jpg)  
Fig. 10. Certificate and overfitting diagnostics vs. sample size N. Left: the (27) bound for the transfer prior (blue) and the transfer-free data-split prior (light blue); the shaded band marks the vacuous region $( C = 1 )$ . Right: train (dark) vs. test (light) empirical risk (16) built on the untransformed loss (21) for the ERM (orange) and PAC-Bayes (blue) models, against noisy outputs (log scale). The left panel bounds the true risk in (15) for a draw from $\mathcal { Q } _ { \widehat { \mathbf { \Omega } } } ^ { * }$ ; the right is an empirical diagnostic. The MSE noise floor is $\sigma _ { \nu } ^ { 2 } = 0 . 2 5$ (dotted), and training below it indicates noise memorization.

(c) The informative prior can be built without a source system. When no related plant is available for transfer, an equally good prior is obtained from the target data alone, by splitting it in two (Prop. 2 of [20]): a short ERM run on one half provides the prior center, and the posterior and its certificate are computed on the other half. This route reaches a comparable FIT (≈93%) with a valid certificate (Figure 10), at the cost of needing more data than transfer, since half the sample is spent on the prior. Either way, an informative prior is essential: a non-informative one forfeits the certificate.

Remark 5 (Estimating the log-partition) The certificate (27) requires $Z _ { \lambda } = \mathbb { E } _ { \boldsymbol { \theta } \sim \mathcal { P } } e ^ { - \lambda \hat { J } ( \boldsymbol { \theta } , \mathbb { S } _ { T } ^ { N } ) }$ , an expectation under the prior with no closed form. We estimate it by Monte Carlo, drawing $M = 1 0 2 4$ i.i.d. prior samples $\theta _ { j } \sim \mathcal { P }$ and averaging $e ^ { - \lambda \hat { J } ( \theta _ { j } , \mathbb { S } _ { T } ^ { N } ) }$ . Since the integrand lies in (0,1], a one-sided Hoeffding bound on this average controls the estimation error, so that $- \frac { 1 } { \lambda } \ln Z _ { \lambda }$ is replaced by a highprobability upper bound; the resulting certificate remains valid, now jointly over the sample, the posterior draw, and the Monte-Carlo draws.

## 7 Conclusions and future work

This paper presented a computationally tractable framework for closed-loop identification of nonlinear systems. By leveraging a complete parameterization of all systems stabilized by a given controller, the proposed Internal Controller Identification (ICI) framework guarantees closed-loop stability of the identified model by construction. Moreover, this approach ensures that training remains well posed even for open-loop unstable plants, and reduces to a consistent openloop problem in the linear case.

Our numerical experiments validate these theoretical claims across several distinct operating regimes. For open-loop unstable plants, the certified models successfully captured the unstable dynamics without ever destabilizing the loop, avoiding the divergence seen in uncertified baselines. On a partially observed robotic system, the indirect criterion proved to be the most accurate under conditions of weak excitation and feedback-correlated noise. We also showed that bounded context wrappers allow a single certified model to identify an entire family of environments, generalizing even to members unseen during training. Finally, in data-scarce regimes, the PAC-Bayesian extension prevented the noise memorization that typically degrades ERM, while providing non-vacuous certificates without requiring held-out data.

![](images/cf4173feaff7e2783129fb868fdf649044300dd3fa0f5250a4105179b814d76b.jpg)

![](images/e0e487c25f982ce062deba7d9f71e8c876deabd870e7a965210c06e10d190cf4.jpg)  
Fig. 11. Deployment FIT vs. N (mean over seeds): left, open loop; right, closed loop; against the noise-free plant. Key comparisons: 1) ERM methods (orange/gold) overfit to noise under data scarcity, degrading performance below the baseline prior. 2) The SVGD estimator (blue), anchored to the transfer prior, resists overfitting and preserves the prior’s quality at scarce N. ERM overtakes it only at larger sample sizes. 3) Relying solely on target data, the data-split prior approach (green) adapts well and maintains a valid certificate. 4) Using a wide prior (pink) yields high FIT but results in a vacuous certificate.

Future work will focus on applying these identified models to adaptive control schemes and designing an iterative procedure that seamlessly integrates system identification with controller synthesis.

## References

[1] M. Abdalmoaty and R. S. Smith. Small noise analysis of nonparametric closed-loop identification. In IEEE Conf. Decision and Control (CDC), pages 7988–7993, 2024.

[2] P. Alquier. User-friendly introduction to PAC-Bayes bounds. Found. Trends Mach. Learn., 17(2):174–303, 2024.

[3] P. Alquier, J. Ridgway, and N. Chopin. On the properties of variational approximations of Gibbs posteriors. J. Mach. Learn. Res., 17(236):1–41, 2016.

[4] B. D. Anderson. From Youla–Kucera to identification, adaptive and nonlinear control. Automatica, 34(12):1485–1506, 1998.

[5] M. G. Boroujeni, C. L. Galimberti, A. Krause, and G. Ferrari-Trecate. A PAC-Bayesian framework for optimal control with stability guarantees. In IEEE Conf. Decision and Control (CDC), pages 8237–8244, 2024.

[6] M. G. Boroujeni, A. Krause, and G. Ferrari-Trecate. Personalized federated learning of probabilistic models: A PAC-Bayesian approach. arXiv preprint arXiv:2401.08351, 2024.

[7] M. G. Boroujeni, L. Meroi, L. Massai, C. L. Galimberti, and G. Ferrari-Trecate. Neural identification of feedback-stabilized nonlinear systems. In IEEE Conf. Decision and Control (CDC), pages 4201–4207. IEEE, 2025.

[8] O. Catoni. Pac-bayesian supervised classification: the thermodynamics of statistical learning. arXiv preprint arXiv:0712.0248, 2007.

[9] S. Dasgupta and B. D. Anderson. A parametrization for the closedloop identification of nonlinear time-varying systems. Automatica, 32(10):1349–1360, 1996.

[10] C. A. Desoer and R.-W. Liu. Global parametrization of feedback systems with nonlinear plants. Syst. Control Lett., 1(4):249–251, 1982.

[11] C. Economou, M. Morari, and B. Palsson. Internal model control: Extension to nonlinear system. Industrial and Engineering Chemistry Process Design and Development, 25(2):403–411, 1986.

[12] D. Eringis, J. Leth, Z.-H. Tan, R. Wisniewski, A. F. Esfahani, and M. Petreczky. Pac-bayesian theory for stochastic lti systems. In 2021 60th IEEE Conference on Decision and Control (CDC), pages 6626–6633. IEEE, 2021.

[13] D. Eringis, J. Leth, Z.-H. Tan, R. Wisniewski, and M. Petreczky. Pac-bayes generalisation bounds for dynamical systems including stable rnns. arXiv preprint arXiv:2312.09793, 2023.

[14] D. Eringis, J. Leth, Z.-H. Tan, R. Wisniewski, and M. Petreczky. Pac-bayesian error bound, via renyi divergence, for a class of linear´ time-invariant state-space models. In Forty-first Int. Conf. Machine Learning (ICML), 2024.

[15] M. Forgione, F. Pura, and D. Piga. From system models to class models: An in-context learning paradigm. IEEE Control Syst. Lett., 7:3513–3518, 2023.

[16] U. Forssell and L. Ljung. Closed-loop identification revisited. Automatica, 35(7):1215–1241, 1999.

[17] K. Fujimoto, B. D. Anderson, and F. De Bruyne. A parametrization for closed-loop identification of nonlinear systems based on differentially coprime kernel representations. Automatica, 37(12):1893–1907, 2001.

[18] L. Furieri, C. L. Galimberti, and G. Ferrari-Trecate. Learning to boost the performance of stable nonlinear systems. IEEE Open J. Control Syst., 3:342–357, 2024.

[19] C. L. Galimberti, L. Furieri, and G. Ferrari-Trecate. Parametrizations of all stable closed-loop responses: From theory to neural network control design. Annu. Rev. Control, 60:101012, 2025.

[20] M. Ghoddousi Boroujeni, C. L. Galimberti, A. Krause, and G. Ferrari-Trecate. PAC-Bayesian optimal control with stability and generalization guarantees. arXiv preprint arXiv:2512.02858, 2025.

[21] G. C. Goodwin and J. S. Welsh. Bias issues in closed loop identification with application to adaptive control. Communications in Information and Systems, 2(4):349–370, 2002.

[22] P. J. Green, K. Łatuszynski, M. Pereyra, and C. P. Robert. Bayesian´ computation: a perspective on the current state, and sampling backward and forwards. Stat. Comput, 25(4):835–862, 2015.

[23] A. Gu, K. Goel, and C. Re. Efficiently modeling long sequences´ with structured state spaces. In International Conference on Learning Representations, 2022.

[24] F. Hansen, G. Franklin, and R. Kosut. Closed-loop identification via the fractional representation: Experiment design. In 1989 American Control Conference, pages 1422–1427, 1989.

[25] D. Ho. A system level approach to discrete-time nonlinear systems. In 2020 American Control Conference (ACC), pages 1625–1630, 2020.

[26] D. Jones and M. A. Dahleh. Closed loop system identification with known feedback: A non asymptotic viewpoint. In 2022 American Control Conference (ACC), pages 4094–4100, 2022.

[27] I. Landau, B. D. Anderson, and F. De Bruyne. Recursive identification algorithms for continuous-time nonlinear plants operating in closed loop. Automatica, 37(3):469–475, 2001.

[28] I. D. Landau and A. Karimi. An output error recursive algorithm for unbiased identification in closed loop. Automatica, 33(5):933–938, 1997.

[29] Q. Liu and D. Wang. Stein variational gradient descent: A general purpose Bayesian inference algorithm. In Adv. Neural Inf. Process. Syst. (NeurIPS), volume 29, 2016.

[30] L. Ljung. System Identification: Theory for the User. Prentice Hall PTR, 1999.

[31] L. Ljung and U. Forssell. An alternative motivation for the indirect approach to closed-loop identification. IEEE Trans. Autom. Control, 44(11):2206–2209, 1999.

[32] I. Maruta and T. Sugie. A simple framework for identifying dynamical systems in closed-loop. IEEE Access, 9:31441–31453, 2021.

[33] L. Massai and G. Ferrari-Trecate. Free parametrization of l2-bounded state space models. arXiv e-prints, pages arXiv–2503, 2025.

[34] L. Massai, S. Messina, N. Kirsch, and G. Ferrari-Trecate. Contextenriched performance boosting via operator decomposition, 2026.

[35] D. A. McAllester. Some pac-bayesian theorems. Machine Learning, 37(3):355–363, 1999.

[36] A. Paice and A. van der Schaft. The class of stabilizing nonlinear plant controller pairs. IEEE Trans. Autom. Control, 41(5):634–645, 1996.

[37] D. Piga, F. Pura, and M. Forgione. On the adaptation of in-context learners for system identification. IFAC-PapersOnLine, 58(15):277– 282, 2024.

[38] G. Pillonetto, A. Aravkin, D. Gedon, L. Ljung, A. H. Ribeiro, and T. B. Schon. Deep networks for system identification: A survey.¨ Automatica, 171:111907, 2025.

[39] M. Revay, R. Wang, and I. R. Manchester. Recurrent equilibrium networks: Flexible dynamic models with guaranteed stability and robustness. IEEE Trans. Autom. Control, 69(5):2855–2870, 2023.

[40] A. H. Ribeiro, K. Tiels, J. Umenberger, T. B. Schon, and L. A.¨ Aguirre. On the smoothness of nonlinear system identification. Automatica, 121:109158, 2020.

[41] J. Rothfuss, V. Fortuin, M. Josifoski, and A. Krause. PACOH: Bayesoptimal meta-learning with PAC-guarantees. In Int. Conf. Machine Learning (ICML), pages 9116–9126. PMLR, 2021.

[42] J. Sjoberg, H. Hjalmarsson, and L. Ljung. Neural networks in system¨ identification. IFAC Proceedings Volumes, 27(8):359–382, 1994.

[43] T. Soderstr ¨ om and P. Stoica. ¨ System identification. Prentice-Hall, Inc., 1988.

[44] R. Toth. ´ Modeling and identification of linear parameter-varying systems, volume 403. Springer, 2010.

[45] P. Van den Hof. Closed-loop issues in system identification. Annu. Rev. Control, 22:173–186, 1998.

[46] A. Van der Schaft. L2-gain and passivity techniques in nonlinear control. Springer, 2000.

[47] J.-W. Van Wingerden and M. Verhaegen. Subspace identification of bilinear and lpv systems for open-and closed-loop data. Automatica, 45(2):372–381, 2009.

[48] D. Youla, H. Jabr, and J. Bongiorno. Modern Wiener-Hopf design of optimal controllers–part ii: The multivariable case. IEEE Trans. Autom. Control, 21(3):319–338, 1976.

[49] M. Zakwan and G. Ferrari-Trecate. Neural port-hamiltonian models for nonlinear distributed control: An unconstrained parametrization approach. IEEE Trans. Autom. Control, 2026.

[50] M. Zakwan, V. Gupta, A. Karimi, E. C. Balta, and G. Ferrari-Trecate. Controller design for structured state-space models via contraction theory. arXiv preprint arXiv:2604.07069, 2026.