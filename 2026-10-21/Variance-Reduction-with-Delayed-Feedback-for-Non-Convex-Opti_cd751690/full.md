# Variance Reduction with Delayed Feedback for Non-Convex Optimization

Wei Jiang<sup>1,2</sup> Dingzhi Yu<sup>2</sup> Zechao Li<sup>1</sup>

<sup>1</sup>School of Computer Science and Engineering, Nanjing University of Science and Technology <sup>2</sup>National Key Laboratory for Novel Software Technology, Nanjing University jiangw@lamda.nju.edu.cn, yudz@lamda.nju.edu.cn, zechao.li@njust.edu.cn

October 21, 2026

## Abstract

We study variance reduction for non-convex optimization when gradient evaluations return with a delay. The server can update the parameter while earlier evaluations remain unfinished. We first analyze an ordered recursive momentum estimator for stochastic optimization. Under average smoothness and bounded variance, its oracle complexity is ${ \cal O } ( 1 + \sigma ^ { 2 } \epsilon ^ { - 2 } + L \Delta _ { f } ( 1 + \sigma / \epsilon + \tau ) \epsilon ^ { - \bar { 2 } } )$ where the target is an expected squared gradient norm of at most $\epsilon ^ { 2 }$ and $\tau$ bounds the feedback delay. We next study finite sums with n components and obtain $O ( n + L \Delta _ { f } ( \sqrt { n } + \operatorname * { m i n } \{ \tau , n -$ $1 \} ) \epsilon ^ { - 2 } )$ evaluations. An asynchronous snapshot variant continues updating during full-gradient computations under an explicit bound on their completion time. For the stochastic method, we further replace a maximum-delay bound by a bound on the weighted amount of missing feedback.

## 1 Introduction

This paper studies variance reduction for stochastic non-convex optimization with delayed feedback. Recursive estimators such as SARAH, SPIDER, and STORM employ the same sample at two nearby points to reduce the noise in a gradient diference [Nguyen et al., 2017; Fang et al., 2018; Cutkosky and Orabona, 2019; Wang et al., 2019]. With delayed evaluations, however, the correction associated with one update may be unavailable when the next update is performed. Several corrections can be outstanding, and their arrival order can depend on the sampled data.

A direct approach fixes the query points and waits for a complete batch before updating the parameter. Waiting preserves the usual variance-reduction oracle complexity, since it does not consume gradient evaluations. We investigate methods that continue to update the parameter while corrections are outstanding. Each correction is evaluated at two consecutive server iterates, and the server incorporates it using its generation index when it returns. We count all issued evaluations, including unfinished tasks.

We first consider stochastic optimization under bounded variance and average smoothness. After one initialization batch, the method issues one two-point task per update without further batch-completion waits. To obtain $\mathbb { E } \left\| \nabla f ( x _ { R } ) \right\| ^ { 2 } \leq \epsilon ^ { 2 }$ , its oracle complexity is

$$
O \left( 1 + \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \left( 1 + \frac { \sigma } { \epsilon } + \tau \right) \right) .
$$

Here $\tau$ bounds the correction delay in server updates, $\sigma ^ { 2 }$ bounds the oracle variance, L is the average-smoothness constant, and $\Delta _ { f }$ is the initial objective gap. We next consider a finite sum of n components. Using exact gradient refreshes every n updates and one sampled diference per update, we obtain

$$
O \left( n + \frac { L \Delta _ { f } ( \sqrt { n } + \operatorname* { m i n } \{ \tau , n - 1 \} ) } { \epsilon ^ { 2 } } \right) .
$$

Thus the delay enters additively in both bounds. In the finite-sum result, a refresh also limits how long an unfinished correction can afect the estimator.

We further study two extensions of these results. For stochastic optimization, we control the outstanding corrections through their exponentially decaying weights in Section 6. The resulting condition allows arbitrarily late responses and repeated permanent losses, provided that these losses are suficiently separated relative to the momentum parameter. For finite sums, Section 5 removes the refresh waits after initialization by computing exact gradients at fixed snapshot points while the server continues to update. Under an additional bound D on the completion delay of each exact snapshot, this method has complexity

$$
O \left( n + \frac { L \Delta _ { f } ( \sqrt { n } + \sqrt { D } + \tau ) } { \epsilon ^ { 2 } } \right) .
$$

The service required for these exact snapshots is stated separately from the delay assumption for sampled corrections.

## 2 Problem Setting

We consider the non-convex optimization problem

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { d } } f ( x ) .\tag{1}
$$

The objective is in the form of $f ( x ) = \mathbb { E } _ { \xi } [ f ( x ; \xi ) ]$ (expectation) or $\begin{array} { r } { f ( x ) = n ^ { - 1 } \sum _ { i = 1 } ^ { n } f _ { i } ( x ) } \end{array}$ (finite-sum).

Assumption 1 (Lower boundedness). The objective satisfies $f _ { * } : = \operatorname* { i n f } _ { x } f ( x ) > - \infty$ . The initial point $x _ { 0 }$ satisfies $f ( x _ { 0 } ) - f _ { * } \leq \Delta _ { f }$

Our goal is to return a point $x _ { R }$ such that

$$
\mathbb { E } [ \| \nabla f ( x _ { R } ) \| ^ { 2 } ] \le \epsilon ^ { 2 } .\tag{2}
$$

The output index R is chosen uniformly from the index set $\{ 0 , \ldots , T - 1 \}$ . In particular, equation (2) also implies $\mathbb { E } [ \| \nabla f ( x _ { R } ) \| ] \le \epsilon$ . All guarantees below are in expectation.

## 2.1 Delayed two-point queries

At update t, the server has an estimator $v _ { t }$ and computes

$$
x _ { t + 1 } = x _ { t } - \eta v _ { t } .\tag{3}
$$

It then requests a correction evaluated at the fixed pair $( x _ { t } , x _ { t + 1 } )$ . A fresh sample is drawn after this pair has been chosen. The two gradient evaluations in the task use the same sample, and we

treat such a task as a single response. Let $A _ { t }$ denote the first server round in which the response to task t is available. Our initial assumption is

$$
t + 1 \leq A _ { t } \leq t + 1 + \tau , \qquad t \geq 0 ,\tag{4}
$$

where $\tau$ is a nonnegative integer. Thus $\tau = 0$ means that the correction is available for the next update. Tasks can arrive out of order.

At the beginning of a round, the server incorporates all responses available for that round. It performs (3) even if no new response has arrived, and then issues the next task. We do not assume that each received message triggers exactly one update. Initialization uses a complete preselected batch at a fixed point. The basic finite-sum method also waits for a complete full gradient at each refresh. Section 5 gives a variant that computes these later full gradients while the server continues updating. Each snapshot still evaluates all n components at its fixed point.

## 2.2 The descent inequality

Lemma 1. Suppose that f is L-smooth and (3) is used with $\eta > 0$ . Then

$$
f ( x _ { t + 1 } ) \leq f ( x _ { t } ) - { \frac { \eta } { 2 } } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } - { \frac { \eta } { 2 } } ( 1 - L \eta ) \left\| v _ { t } \right\| ^ { 2 } + { \frac { \eta } { 2 } } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } .\tag{5}
$$

Consequently, under Assumption $\boldsymbol { \mathit { 1 } }$

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta } - ( 1 - L \eta ) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } \right\| ^ { 2 } + \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } .\tag{6}
$$

Proof. Smoothness and (3) give

$$
\begin{array} { l } { f ( x _ { t + 1 } ) \leq f ( x _ { t } ) + \langle \nabla f ( x _ { t } ) , x _ { t + 1 } - x _ { t } \rangle + \displaystyle \frac { L } { 2 } \left. x _ { t + 1 } - x _ { t } \right. ^ { 2 } } \\ { \displaystyle = f ( x _ { t } ) - \eta \langle \nabla f ( x _ { t } ) , v _ { t } \rangle + \displaystyle \frac { L \eta ^ { 2 } } { 2 } \left. v _ { t } \right. ^ { 2 } . } \end{array}
$$

Since

$$
\left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } = \left\| v _ { t } \right\| ^ { 2 } + \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } - 2 \left. \nabla f ( x _ { t } ) , v _ { t } \right. ,
$$

we obtain

$$
\begin{array} { l } { f ( x _ { t + 1 } ) \leq f ( x _ { t } ) - \displaystyle \frac { \eta } { 2 } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } - \displaystyle \frac { \eta } { 2 } \left\| v _ { t } \right\| ^ { 2 } + \displaystyle \frac { \eta } { 2 } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } + \displaystyle \frac { L \eta ^ { 2 } } { 2 } \left\| v _ { t } \right\| ^ { 2 } } \\ { \displaystyle = f ( x _ { t } ) - \displaystyle \frac { \eta } { 2 } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } - \displaystyle \frac { \eta } { 2 } ( 1 - L \eta ) \left\| v _ { t } \right\| ^ { 2 } + \displaystyle \frac { \eta } { 2 } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } , } \end{array}
$$

which proves (5). Rearranging and taking expectations gives

$$
\mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 2 } { \eta } \big ( \mathbb { E } f ( x _ { t } ) - \mathbb { E } f ( x _ { t + 1 } ) \big ) - ( 1 - L \eta ) \mathbb { E } \left\| v _ { t } \right\| ^ { 2 } + \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } .
$$

The objective diferences telescope:

$$
\sum _ { t = 0 } ^ { T - 1 } \bigl ( \mathbb { E } f ( x _ { t } ) - \mathbb { E } f ( x _ { t + 1 } ) \bigr ) = f ( x _ { 0 } ) - \mathbb { E } f ( x _ { T } ) \leq f ( x _ { 0 } ) - f _ { * } \leq \Delta _ { f } .
$$

Summing the preceding inequalities proves (6).

## 3 Variance Reduction for Stochastic Optimization

We first consider $f ( x ) = \mathbb { E } _ { \xi } [ f ( x ; \xi ) ]$ . An exact full gradient is generally unavailable. We use a STORM estimator, whose momentum term gradually reduces the efect of old estimation errors.

Assumption 2 (Stochastic oracle). For all $x , y \in \mathbb { R } ^ { d }$ , the stochastic gradient satisfies

$$
\mathbb { E } _ { \xi } [ \nabla f ( x ; \xi ) ] = \nabla f ( x ) ,\tag{7}
$$

$$
\begin{array} { r } { \mathbb { E } _ { \xi } \left\| \nabla f ( x ; \xi ) - \nabla f ( x ) \right\| ^ { 2 } \leq \sigma ^ { 2 } , } \end{array}\tag{8}
$$

$$
\begin{array} { r } { \mathbb { E } _ { \xi } \left\| \nabla f ( x ; \xi ) - \nabla f ( y ; \xi ) \right\| ^ { 2 } \leq L ^ { 2 } \left\| x - y \right\| ^ { 2 } . } \end{array}\tag{9}
$$

Here $\sigma \geq 0 , L > 0$ , and the same sample $\xi$ is used at both points in (9).

These are the usual bounded-variance and average-smoothness conditions for recursive variance reduction. In particular, we do not assume a uniform bound on $\| \nabla f ( x ; \xi ) \|$ . Jensen’s inequality applied to (9) implies that $f$ is L-smooth.

## 3.1 The proposed method

We first draw B fresh samples and wait for all of their evaluations at $x _ { 0 }$ to compute

$$
v _ { 0 } = \frac { 1 } { B } \sum _ { k = 1 } ^ { B } \nabla f ( x _ { 0 } ; \xi _ { 0 , k } ) .\tag{10}
$$

After the update from $x _ { t }$ to $x _ { t + 1 }$ , task t draws a fresh sample $\xi _ { t }$ and returns

$$
\begin{array} { r l } & { \delta _ { t } = \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - ( 1 - \alpha ) \nabla f ( x _ { t } ; \xi _ { t } ) } \\ & { \quad = \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - \nabla f ( x _ { t } ; \xi _ { t } ) + \alpha \nabla f ( x _ { t } ; \xi _ { t } ) , } \end{array}\tag{11}
$$

where $0 \textless \alpha \textless 1$ . The server uses the available corrections with weights determined by their generation times:

$$
v _ { t } = ( 1 - \alpha ) ^ { t } v _ { 0 } + \sum _ { j = 0 } ^ { t - 1 } { \bf 1 } _ { \{ A _ { j } \leq t \} } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } .\tag{12}
$$

Thus a correction does not receive the weight of a newly generated sample merely because it arrives late. It receives the weight it would have had if it had been available from the next round. Formula (12) can be implemented by

$$
v _ { t } = ( 1 - \alpha ) v _ { t - 1 } + \sum _ { j : A _ { j } = t } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } , \qquad t \geq 1 .\tag{13}
$$

If no response arrives in a round, the estimator is multiplied by $1 - \alpha .$ , the parameter is updated, and another task is issued. Multiple responses can be incorporated in one round.

## 3.2 Convergence guarantee

Theorem 1. Under Assumptions 1 and 2, suppose that (4) holds. For any $\epsilon > 0$ , choose

$$
\eta = \frac { 1 } { 1 6 L ( 1 + \sigma / \epsilon + \tau ) } ,
$$

$$
\alpha = { \frac { 1 } { 1 6 ( 1 + \sigma / \epsilon ) ( 1 + \sigma / \epsilon + \tau ) } } ,\tag{14}
$$

$$
T = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { 6 4 L \Delta _ { f } ( 1 + \sigma / \epsilon + \tau ) } { \epsilon ^ { 2 } } \right\rceil \right\} .\tag{15}
$$

Algorithm 1 Delayed recursive momentum for stochastic optimization   
Require: Initial point $x _ { 0 } ,$ update number T, initialization size $B ,$ parameters η, α   
1: Compute v<sub>0</sub> using the complete preselected batch in (10).   
2: for $t = 0 , \ldots , T - 1$ do   
3: if $t \geq 1$ then   
4: Update $v _ { t }$ by (13), including all responses arriving in round t.   
5: end if   
6: Update $x _ { t + 1 } = x _ { t } - \eta v _ { t } .$   
7: Draw a fresh $\xi _ { t }$ and issue the two-point task (11).   
8: end for   
9: Choose R uniformly from $\{ 0 , \ldots , T - 1 \}$ and return $x _ { R } .$

Set the initialization size to

$$
B = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { \alpha T } \right\} \right\rceil \right\} .\tag{16}
$$

Then Algorithm 1 guarantees

$$
\mathbb { E } \left\| \nabla f ( x _ { R } ) \right\| ^ { 2 } \leq \epsilon ^ { 2 } .\tag{17}
$$

In particular, its oracle complexity is

$$
O \left( 1 + \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \left( 1 + \frac { \sigma } { \epsilon } + \tau \right) \right) .\tag{18}
$$

Remark 1. For $0 < \epsilon \le \sigma ,$ (18) becomes

$$
O \left( { \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } } + { \frac { L \Delta _ { f } \sigma } { \epsilon ^ { 3 } } } + { \frac { L \Delta _ { f } \tau } { \epsilon ^ { 2 } } } \right) .
$$

Hence $\tau = O ( \sigma / \epsilon )$ preserves the usual stochastic variance-reduction order.

Remark 2 (Initialization). The factor min $\{ 1 , 1 / ( \alpha T ) \}$ in (16) uses the decay of the initial estimation error. For $\Delta _ { f } > 0$ , the choices above imply

$$
\alpha T \geq \frac { 4 L \Delta _ { f } } { ( 1 + \sigma / \epsilon ) \epsilon ^ { 2 } } , \qquad B \leq 1 + \operatorname* { m i n } \left\{ \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } , \frac { 4 \sigma ^ { 2 } ( 1 + \sigma / \epsilon ) } { L \Delta _ { f } } \right\} .
$$

Thus a longer run can start with a smaller complete batch. This reduces the number of evaluations required before the first update while preserving the complexity in (18).

## 3.3 Estimation error

We introduce an auxiliary estimator that contains all generated corrections:

$$
u _ { 0 } = v _ { 0 } , \qquad u _ { t + 1 } = ( 1 - \alpha ) u _ { t } + \delta _ { t } .\tag{19}
$$

The sequence $u _ { t }$ is used only for analysis. Its corrections are evaluated at the actual iterates of Algorithm 1 and it does not generate a second sequence of parameters. Define

$$
\mathcal { M } _ { t } = \{ j : 0 \leq j < t , \ A _ { j } > t \} .\tag{20}
$$

For $t \geq 2$ , repeated substitution in (19) gives

$$
\begin{array} { l } { { u _ { t } = ( 1 - \alpha ) u _ { t - 1 } + \delta _ { t - 1 } } } \\ { { \ = ( 1 - \alpha ) ^ { 2 } u _ { t - 2 } + ( 1 - \alpha ) \delta _ { t - 2 } + \delta _ { t - 1 } } } \\ { { \ = \cdot \cdot \cdot = ( 1 - \alpha ) ^ { t } v _ { 0 } + { \displaystyle \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } } . } } \end{array}
$$

The last equality also holds for $t = 0 , 1$ , with an empty sum when $t = 0$ . Comparing with (12),

$$
\begin{array} { l } { { \displaystyle u _ { t } - v _ { t } = \sum _ { j = 0 } ^ { t - 1 } ( 1 - \mathbf { 1 } _ { \{ A _ { j } \leq t \} } ) ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } } } \\ { { \displaystyle \qquad = \sum _ { j \in \mathcal { M } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } . } } \end{array}
$$

Consequently,

$$
v _ { t } - \nabla f ( x _ { t } ) = u _ { t } - \nabla f ( x _ { t } ) - \sum _ { j \in \mathcal { M } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } .\tag{21}
$$

The first term includes all generated samples, whereas the second term records missing responses.   
We bound these two terms separately.

Lemma 2. Under Assumption 2, for any $0 < \alpha < 1$ and $T \geq 1$

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. u _ { t } - \nabla f ( x _ { t } ) \right. ^ { 2 } \leq \frac { \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + \frac { 2 L ^ { 2 } } { \alpha } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. x _ { t + 1 } - x _ { t } \right. ^ { 2 } + 2 \alpha \sigma ^ { 2 } T ,\tag{22}
$$

and each correction satisfies

$$
\begin{array} { r } { \mathbb { E } \left\| \delta _ { t } \right\| ^ { 2 } \leq 2 L ^ { 2 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } + 2 \alpha ^ { 2 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } . } \end{array}\tag{23}
$$

These bounds do not require a maximum-delay assumption.

Proof. Center the new correction at its population value:

$$
\begin{array} { r l } & { \zeta _ { t } = \delta _ { t } - \big ( \nabla f ( x _ { t + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { t } ) \big ) } \\ & { \quad = \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - \nabla f ( x _ { t } ; \xi _ { t } ) - \big ( \nabla f ( x _ { t + 1 } ) - \nabla f ( x _ { t } ) \big ) } \\ & { \qquad + \alpha \big ( \nabla f ( x _ { t } ; \xi _ { t } ) - \nabla f ( x _ { t } ) \big ) . } \end{array}\tag{24}
$$

The points $x _ { t } , x _ { t + 1 }$ are chosen before $\xi _ { t }$ . Unbiasedness therefore gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \xi _ { t } } \zeta _ { t } = \nabla f ( x _ { t + 1 } ) - \nabla f ( x _ { t } ) - \left( \nabla f ( x _ { t + 1 } ) - \nabla f ( x _ { t } ) \right) } \\ & { \qquad + \alpha \big ( \nabla f ( x _ { t } ) - \nabla f ( x _ { t } ) \big ) = 0 , } \end{array}
$$

where the chosen pair is held fixed in this sample average. Also,

$$
\begin{array} { r l } & { \mathbb { E } _ { \xi _ { t } } \big \| \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - \nabla f ( x _ { t } ; \xi _ { t } ) - \big ( \nabla f ( x _ { t + 1 } ) - \nabla f ( x _ { t } ) \big ) \big \| ^ { 2 } } \\ & { \quad = \mathbb { E } _ { \xi _ { t } } \| \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - \nabla f ( x _ { t } ; \xi _ { t } ) \| ^ { 2 } - \| \nabla f ( x _ { t + 1 } ) - \nabla f ( x _ { t } ) \| ^ { 2 } } \\ & { \quad \le \mathbb { E } _ { \xi _ { t } } \| \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - \nabla f ( x _ { t } ; \xi _ { t } ) \| ^ { 2 } . } \end{array}
$$

Using $\left\| a + b \right\| ^ { 2 } \leq 2 \left\| a \right\| ^ { 2 } + 2 \left\| b \right\| ^ { 2 }$ in (24), then averaging the preceding randomness, yields

$$
\begin{array} { r l } & { \mathbb { E } \left\| \zeta _ { t } \right\| ^ { 2 } \leq 2 \mathbb { E } \left\| \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - \nabla f ( x _ { t } ; \xi _ { t } ) \right\| ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } } \\ & { \qquad \leq 2 L ^ { 2 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } . } \end{array}\tag{25}
$$

Subtracting $\nabla f ( x _ { t + 1 } )$ from (19),

$$
\begin{array} { r } { u _ { t + 1 } - \nabla f ( x _ { t + 1 } ) = ( 1 - \alpha ) u _ { t } + \delta _ { t } - \nabla f ( x _ { t + 1 } ) } \\ { = ( 1 - \alpha ) \big ( u _ { t } - \nabla f ( x _ { t } ) \big ) + \zeta _ { t } . } \end{array}
$$

Thus, for $t \geq 2$

$$
\begin{array} { r l } & { u _ { t } - \nabla f ( x _ { t } ) = ( 1 - \alpha ) \big ( u _ { t - 1 } - \nabla f ( x _ { t - 1 } ) \big ) + \zeta _ { t - 1 } } \\ & { \qquad = ( 1 - \alpha ) ^ { 2 } \big ( u _ { t - 2 } - \nabla f ( x _ { t - 2 } ) \big ) + ( 1 - \alpha ) \zeta _ { t - 2 } + \zeta _ { t - 1 } . } \end{array}
$$

Continuing to $u _ { 0 } = v _ { 0 }$ gives, for every $t \geq 0$

$$
u _ { t } - \nabla f ( x _ { t } ) = ( 1 - \alpha ) ^ { t } \big ( v _ { 0 } - \nabla f ( x _ { 0 } ) \big ) + \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { t - j - 1 } \zeta _ { j } .\tag{26}
$$

For each $k \geq 0$ , average $\xi _ { k }$ with the preceding samples and its point pair fixed. The zero mean just proved gives

$$
\mathbb { E } \left. v _ { 0 } - \nabla f ( x _ { 0 } ) , \zeta _ { k } \right. = 0 .
$$

For $0 \leq j < k$ , the same averaging gives E $\langle \zeta _ { j } , \zeta _ { k } \rangle = 0$ . Expanding the square in (26) now gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } \| u _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } = ( 1 - \alpha ) ^ { 2 t } \mathbb { E } \| v _ { 0 } - \nabla f ( x _ { 0 } ) \| ^ { 2 } + \displaystyle \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } \mathbb { E } \| \zeta _ { j } \| ^ { 2 } } \\ { \displaystyle \qquad + 2 \displaystyle \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { 2 t - j - 1 } \underbrace { \mathbb { E } \left. v _ { 0 } - \nabla f ( x _ { 0 } ) , \zeta _ { j } \right. } _ { = 0 } } \\ { \displaystyle \qquad + 2 \sum _ { 0 \leq j < k < t } ( 1 - \alpha ) ^ { 2 t - j - k - 2 } \underbrace { \mathbb { E } \left. \zeta _ { j } , \zeta _ { k } \right. } _ { = 0 } . } \end{array}
$$

For the initialization, distinct samples are independent and centered, so

$$
\begin{array} { r l } & { \mathbb { E } \left\| v _ { 0 } - \nabla f ( x _ { 0 } ) \right\| ^ { 2 } = \displaystyle \frac { 1 } { B ^ { 2 } } \sum _ { k = 1 } ^ { B } \mathbb { E } \left\| \nabla f ( x _ { 0 } ; \xi _ { 0 , k } ) - \nabla f ( x _ { 0 } ) \right\| ^ { 2 } } \\ & { \qquad + \displaystyle \frac { 2 } { B ^ { 2 } } \sum _ { 1 \le k < l \le B } \mathbb { E } \left. \nabla f ( x _ { 0 } ; \xi _ { 0 , k } ) - \nabla f ( x _ { 0 } ) , \nabla f ( x _ { 0 } ; \xi _ { 0 , l } ) - \nabla f ( x _ { 0 } ) \right. } \\ & { \qquad = 0 } \\ & { \qquad \le \displaystyle \frac { B \sigma ^ { 2 } } { B ^ { 2 } } = \displaystyle \frac { \sigma ^ { 2 } } { B } . } \end{array}
$$

Substituting this bound and (25) into the square expansion yields

$$
\begin{array} { r l r } {  { \mathbb { E } \| u _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } \leq ( 1 - \alpha ) ^ { 2 t } \frac { \sigma ^ { 2 } } { B } } } \\ & { } & { \quad + \displaystyle \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } ( 2 L ^ { 2 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } ) . } \end{array}\tag{27}
$$

We now sum this inequality over t. The initial error has total weight

$$
\sum _ { t = 0 } ^ { T - 1 } ( 1 - \alpha ) ^ { 2 t } = \frac { 1 - ( 1 - \alpha ) ^ { 2 T } } { \alpha ( 2 - \alpha ) } \leq \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} .
$$

The first bound follows because each summand is at most one. The second follows by extending the geometric sum to infinity and using $2 - \alpha \geq 1$ . For each subsequent innovation $j ,$ , its total weight is at most

$$
\sum _ { t = j + 1 } ^ { T - 1 } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } \leq \sum _ { k = 0 } ^ { \infty } ( 1 - \alpha ) ^ { 2 k } = \frac { 1 } { \alpha ( 2 - \alpha ) } \leq \frac { 1 } { \alpha } .
$$

Changing the order of summation uses

$$
\{ ( t , j ) : 0 \le t \le T - 1 , \ 0 \le j < t \} = \{ ( t , j ) : 0 \le j \le T - 2 , \ j + 1 \le t \le T - 1 \} .
$$

Therefore,

$$
\begin{array} { r l } & { \underset { t = 0 } { \overset { T - 1 } { \sum } } \underset { j = 0 } { \overset { t - 1 } { \sum } } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } \left( 2 L ^ { 2 } \mathbb { E } \left. x _ { j + 1 } - x _ { j } \right. ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } \right) } \\ & { \quad = \underset { j = 0 } { \overset { T - 2 } { \sum } } \left( 2 L ^ { 2 } \mathbb { E } \left. x _ { j + 1 } - x _ { j } \right. ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } \right) \underset { t = j + 1 } { \overset { T - 1 } { \sum } } \left( 1 - \alpha \right) ^ { 2 ( t - j - 1 ) } } \\ & { \quad \le \frac { 2 L ^ { 2 } } { \alpha } \underset { j = 0 } { \overset { T - 2 } { \sum } } \mathbb { E } \left. x _ { j + 1 } - x _ { j } \right. ^ { 2 } + 2 \alpha \sigma ^ { 2 } ( T - 1 ) } \\ & { \quad \le \frac { 2 L ^ { 2 } } { \alpha } \underset { j = 0 } { \overset { T - 1 } { \sum } } \mathbb { E } \left. x _ { j + 1 } - x _ { j } \right. ^ { 2 } + 2 \alpha \sigma ^ { 2 } T . } \end{array}
$$

All sums with upper endpoint $T - 2$ are empty when $T = 1$ . Adding the initialization term proves (22). Its contribution after division by $T$ is

$$
{ \frac { \sigma ^ { 2 } } { B T } } \operatorname* { m i n } \left\{ T , { \frac { 1 } { \alpha } } \right\} = { \frac { \sigma ^ { 2 } } { B } } \operatorname* { m i n } \left\{ 1 , { \frac { 1 } { \alpha T } } \right\} .
$$

To bound the correction itself, first expand the second moment of a stochastic gradient:

$$
\begin{array} { r l } & { \mathbb { E } \left\| \nabla f ( x _ { t } ; \xi _ { t } ) \right\| ^ { 2 } = \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } + \mathbb { E } \left\| \nabla f ( x _ { t } ; \xi _ { t } ) - \nabla f ( x _ { t } ) \right\| ^ { 2 } } \\ & { \qquad + 2 \underbrace { \mathbb { E } \left. \nabla f ( x _ { t } ) , \nabla f ( x _ { t } ; \xi _ { t } ) - \nabla f ( x _ { t } ) \right. } _ { = 0 } } \\ & { \qquad \le \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } + \sigma ^ { 2 } . } \end{array}
$$

The second expression in (11) then gives

$$
\begin{array} { r l } & { \mathbb { E } \left\| \delta _ { t } \right\| ^ { 2 } \leq 2 \mathbb { E } \left\| \nabla f ( x _ { t + 1 } ; \xi _ { t } ) - \nabla f ( x _ { t } ; \xi _ { t } ) \right\| ^ { 2 } + 2 \alpha ^ { 2 } \mathbb { E } \left\| \nabla f ( x _ { t } ; \xi _ { t } ) \right\| ^ { 2 } } \\ & { \qquad \leq 2 L ^ { 2 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } + 2 \alpha ^ { 2 } \left( \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } + \sigma ^ { 2 } \right) . } \end{array}
$$

This is (23).

Lemma 3. Under Assumption 2, the sampling protocol of Section ${ \mathcal { Q } } ,$ and the delay bound (4), Algorithm 1 with any $0 < \alpha < 1$ satisfies

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + 4 L ^ { 2 } \left( \frac { 1 } { \alpha } + \tau ^ { 2 } \right) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } } \\ { \displaystyle \qquad + 4 \alpha ^ { 2 } \tau ^ { 2 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } + 4 \sigma ^ { 2 } ( \alpha + \alpha ^ { 2 } \tau ^ { 2 } ) T . } \end{array}\tag{28}
$$

Proof. For $j \in \mathcal { M } _ { t }$ , the integer arrival time satisfies

$$
t + 1 \leq A _ { j } \leq j + 1 + \tau , \qquad { \mathrm { h e n c e } } \qquad j \geq t - \tau .
$$

Together with $0 \leq j < t$ , this gives

$$
\mathcal { M } _ { t } \subseteq \{ \operatorname* { m a x } \{ 0 , t - \tau \} , \ldots , t - 1 \} , \qquad | \mathcal { M } _ { t } | \leq \tau .
$$

$\tau = 0$ , all missing sets are empty. Otherwise, Cauchy–Schwarz and $0 \leq ( 1 - \alpha ) ^ { t - j - 1 } \leq 1$ give, for each realized execution,

$$
\begin{array} { r l } { \displaystyle \left\| \displaystyle \sum _ { j \in { \mathcal { M } } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } \right\| ^ { 2 } \leq | { \mathcal { M } } _ { t } | \displaystyle \sum _ { j \in { \mathcal { M } } _ { t } } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } \| \delta _ { j } \| ^ { 2 } } & { } \\ { \leq \tau \displaystyle \sum _ { j = \operatorname* { m a x } \{ 0 , t - \tau \} } ^ { t - 1 } \| \delta _ { j } \| ^ { 2 } . } & { } \end{array}
$$

The index condition in the last sum can also be written as

$$
\operatorname* { m a x } \{ 0 , t - \tau \} \leq j \leq t - 1 \quad \Longleftrightarrow \quad 0 \leq j \leq T - 2 , \quad j + 1 \leq t \leq \operatorname* { m i n } \{ T - 1 , j + \tau \} ,
$$

when $0 \leq t \leq T - 1$ . Thus, after taking expectations,

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 1 } \sum _ { j = \operatorname* { m a x } \{ 0 , t - \tau \} } ^ { t - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } = \sum _ { j = 0 } ^ { T - 2 } \sum _ { t = j + 1 } ^ { m \lceil T - 1 , j + \tau \} } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } } \\ & { } & { \leq \tau \sum _ { j = 0 } ^ { T - 2 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } \leq \tau \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } . } \end{array}
$$

The inner sum contains at most τ terms. Combining these inequalities with (23) yields

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \displaystyle \sum _ { j \in \mathcal { M } _ { t } } \left( 1 - \alpha \right) ^ { t - j - 1 } \delta _ { j } \right\| ^ { 2 } } \\ & { \quad \le { \tau ^ { 2 } } \displaystyle \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } \\ & { \quad \le 2 { L ^ { 2 } } { \tau ^ { 2 } } \displaystyle \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } + 2 { \alpha ^ { 2 } } { \tau ^ { 2 } } \displaystyle \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { j } ) \| ^ { 2 } + 2 { \alpha ^ { 2 } } { \tau ^ { 2 } } \sigma ^ { 2 } T . } \end{array}\tag{29}
$$

Finally, (21) and $\left. a - b \right. ^ { 2 } \leq 2 \left. a \right. ^ { 2 } + 2 \left. b \right. ^ { 2 }$ imply

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } } \\ & { \displaystyle \leq 2 \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| u _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } + 2 \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \sum _ { j \in \mathcal { M } _ { t } } \left( 1 - \alpha \right) ^ { t - j - 1 } \delta _ { j } \right\| ^ { 2 } } \\ & { \displaystyle \leq \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + \left( \frac { 4 L ^ { 2 } } { \alpha } + 4 L ^ { 2 } \tau ^ { 2 } \right) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } } \\ & { \qquad + 4 \alpha ^ { 2 } \tau ^ { 2 } \underset { t = 0 } { \overset { T - 1 } { \sum } } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } + \big ( 4 \alpha \sigma ^ { 2 } + 4 \alpha ^ { 2 } \tau ^ { 2 } \sigma ^ { 2 } \big ) T , } \end{array}
$$

which is (28). Arrival indicators were removed by a bound valid for every execution, before taking expectations. □

## 3.4 Proof of Theorem 1

By (3),

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. x _ { t + 1 } - x _ { t } \right. ^ { 2 } = \eta ^ { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. v _ { t } \right. ^ { 2 } .
$$

Substituting (28) into (6) therefore gives

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta } - ( 1 - L \eta ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } + \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} } \\ { \displaystyle \qquad + 4 L ^ { 2 } \eta ^ { 2 } \left( \frac { 1 } { \alpha } + \tau ^ { 2 } \right) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } } \\ { \displaystyle \qquad + 4 \alpha ^ { 2 } \tau ^ { 2 } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } + 4 \sigma ^ { 2 } ( \alpha + \alpha ^ { 2 } \tau ^ { 2 } ) T . } \end{array}
$$

Move the gradient and estimator sums on the right to the left:

$$
\begin{array} { r l r } {  { \big ( 1 - 4 \alpha ^ { 2 } \tau ^ { 2 } \big ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } } } \\ & { } & { \quad + [ 1 - L \eta - 4 L ^ { 2 } \eta ^ { 2 } ( \frac { 1 } { \alpha } + \tau ^ { 2 } ) ] \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } } \\ & { } & { \quad \leq \frac { 2 \Delta _ { f } } { \eta } + \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \{ T , \frac { 1 } { \alpha } \} + 4 \sigma ^ { 2 } ( \alpha + \alpha ^ { 2 } \tau ^ { 2 } ) T . } \end{array}\tag{30}
$$

We verify the coeficients and the remaining terms separately.

For the gradient coeficient, (14) gives

$$
4 \alpha ^ { 2 } \tau ^ { 2 } = \frac { \tau ^ { 2 } } { 6 4 ( 1 + \sigma / \epsilon ) ^ { 2 } ( 1 + \sigma / \epsilon + \tau ) ^ { 2 } } \le \frac { 1 } { 6 4 } .
$$

For the estimator coeficient, the same parameters give

$$
L \eta = \frac { 1 } { 1 6 ( 1 + \sigma / \epsilon + \tau ) } \leq \frac { 1 } { 1 6 } ,
$$

$$
\frac { 4 L ^ { 2 } \eta ^ { 2 } } { \alpha } = \frac { 1 + \sigma / \epsilon } { 4 ( 1 + \sigma / \epsilon + \tau ) } \leq \frac { 1 } { 4 } ,
$$

$$
4 L ^ { 2 } \eta ^ { 2 } \tau ^ { 2 } = \frac { \tau ^ { 2 } } { 6 4 ( 1 + \sigma / \epsilon + \tau ) ^ { 2 } } \leq \frac { 1 } { 6 4 } .
$$

It follows that

$$
\begin{array} { c } { { 1 - 4 \alpha ^ { 2 } \tau ^ { 2 } \geq \displaystyle \frac { 6 3 } { 6 4 } , } } \\ { { 1 - L \eta - 4 L ^ { 2 } \eta ^ { 2 } \left( \displaystyle \frac { 1 } { \alpha } + \tau ^ { 2 } \right) \geq 1 - \displaystyle \frac { 1 } { 1 6 } - \displaystyle \frac { 1 } { 4 } - \displaystyle \frac { 1 } { 6 4 } = \displaystyle \frac { 4 3 } { 6 4 } > 0 . } } \end{array}
$$

The estimator sum on the left of (30) can therefore be dropped.

The choices of B and T imply

$$
\begin{array} { c } { B \geq \displaystyle \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } \operatorname* { m i n } \left. 1 , \frac { 1 } { \alpha T } \right. , } \\ { \displaystyle \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left. 1 , \frac { 1 } { \alpha T } \right. \leq \displaystyle \frac { \epsilon ^ { 2 } } { 8 } , } \\ { T \geq \displaystyle \frac { 6 4 L \Delta _ { f } ( 1 + \sigma / \epsilon + \tau ) } { \epsilon ^ { 2 } } , } \\ { \displaystyle \frac { 2 \Delta _ { f } } { \eta T } = \displaystyle \frac { 3 2 L \Delta _ { f } ( 1 + \sigma / \epsilon + \tau ) } { T } \leq \displaystyle \frac { \epsilon ^ { 2 } } { 2 } . } \end{array}
$$

When $\sigma = 0$ or $\Delta _ { f } = 0$ , the corresponding numerator is zero, so these bounds still hold. For the remaining terms, use

$$
\frac { \sigma } { 1 + \sigma / \epsilon } \leq \epsilon , \qquad 1 + \frac { \sigma } { \epsilon } \leq 1 + \frac { \sigma } { \epsilon } + \tau , \qquad \frac { \tau } { 1 + \sigma / \epsilon + \tau } \leq 1 .
$$

Then

$$
\begin{array} { c } { { 4 \alpha \sigma ^ { 2 } = \displaystyle \frac { \sigma ^ { 2 } } { 4 ( 1 + \sigma / \epsilon ) ( 1 + \sigma / \epsilon + \tau ) } \le \displaystyle \frac { \sigma ^ { 2 } } { 4 ( 1 + \sigma / \epsilon ) ^ { 2 } } \le \displaystyle \frac { \epsilon ^ { 2 } } { 4 } , } } \\ { { 4 \alpha ^ { 2 } \tau ^ { 2 } \sigma ^ { 2 } = \displaystyle \frac { \sigma ^ { 2 } \tau ^ { 2 } } { 6 4 ( 1 + \sigma / \epsilon ) ^ { 2 } ( 1 + \sigma / \epsilon + \tau ) ^ { 2 } } \le \displaystyle \frac { \epsilon ^ { 2 } } { 6 4 } , } } \end{array}
$$

Dividing (30) by $T ( 1 - 4 \alpha ^ { 2 } \tau ^ { 2 } )$ gives

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { \epsilon ^ { 2 } / 2 + \epsilon ^ { 2 } / 8 + \epsilon ^ { 2 } / 4 + \epsilon ^ { 2 } / 6 4 } { 6 3 / 6 4 } = \frac { 1 9 } { 2 1 } \epsilon ^ { 2 } .
$$

Since the output index is uniform and independent,

$$
\mathbb { E } \left\| \nabla f ( x _ { R } ) \right\| ^ { 2 } = \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { T } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 1 9 } { 2 1 } \epsilon ^ { 2 } \leq \epsilon ^ { 2 } .
$$

For the cost, initialization issues B single-point evaluations. Each of the T updates issues one two-point task, including the last update. The ceiling bounds give

$$
\begin{array} { c } { { B \leq 1 + \displaystyle \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } , } } \\ { { T \leq 1 + \displaystyle \frac { 6 4 L \Delta _ { f } ( 1 + \sigma / \epsilon + \tau ) } { \epsilon ^ { 2 } } , } } \\ { { B + 2 T \leq 3 + \displaystyle \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \displaystyle \frac { 1 2 8 L \Delta _ { f } ( 1 + \sigma / \epsilon + \tau ) } { \epsilon ^ { 2 } } . } } \end{array}
$$

This count includes every issued evaluation, even if its response arrives after the algorithm returns, and proves (18). □

## 4 Variance Reduction for Finite-Sum Optimization

We next consider the finite-sum objective

$$
f ( x ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } f _ { i } ( x ) .\tag{31}
$$

Here an exact full gradient can be computed with n component-gradient evaluations. We use periodic exact refreshes to control the accumulated estimation error, while continuing to incorporate delayed corrections between refreshes.

Assumption 3 (Finite-sum average smoothness). For all $x , y \in \mathbb { R } ^ { d }$ , and $L > 0$

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left\| \nabla f _ { i } ( x ) - \nabla f _ { i } ( y ) \right\| ^ { 2 } \leq L ^ { 2 } \left\| x - y \right\| ^ { 2 } ,
$$

By Jensen’s inequality, Assumption 3 implies that f is L-smooth. Individual L-smoothness of the components is suficient, but is not required. We do not assume bounded component gradients or a uniform bound on their variance.

## 4.1 The proposed method

Let q be the refresh period. At rounds $0 , q , 2 q , . . . ,$ we fix the current parameter, compute its exact full gradient, and discard corrections from earlier epochs. Between refreshes, task t draws b independent uniform indices $i _ { t , 1 } , \ldots , i _ { t , b }$ and returns

$$
\delta _ { t } = \frac { 1 } { b } \sum _ { k = 1 } ^ { b } ( \nabla f _ { i _ { t , k } } ( x _ { t + 1 } ) - \nabla f _ { i _ { t , k } } ( x _ { t } ) ) .\tag{32}
$$

For a given round $t ,$ let $r _ { t } = q \lfloor t / q \rfloor$ be the first round of its epoch. The implemented estimator is

$$
v _ { t } = \nabla f ( x _ { r _ { t } } ) + \sum _ { j = r _ { t } } ^ { t - 1 } \mathbf { 1 } _ { \{ A _ { j } \leq t \} } \delta _ { j } .\tag{33}
$$

If all corrections have returned, their population counterparts telescope from $\boldsymbol { x } _ { r _ { t } }$ to $x _ { t }$ . With delayed feedback, we retain the available corrections and continue updating. Arrival order does not afect the sum.

Algorithm 2 Delayed recursive variance reduction for finite sums   
Require: Initial point $x _ { 0 } ,$ update number ${ \overline { { \mathbf { \nabla } { T } } } } ,$ period $q ,$ batch size $b ,$ learning rate $\eta$   
1: for $t = 0 , \ldots , T - 1$ do   
2: if t is a multiple of $q$ then   
3: Fix $x _ { t } ,$ compute $\nabla f ( x _ { t } )$ , and set $v _ { t } = \nabla f ( x _ { t } )$   
4: Discard all corrections from preceding epochs.   
5: else   
6: $v _ { t } = v _ { t - 1 } + \quad \sum \quad \delta _ { j } .$   
$\scriptstyle j \geq r _ { t } : A _ { j } = t$   
7: end if   
8: Update $x _ { t + 1 } = x _ { t } - \eta v _ { t } .$   
9: Draw b fresh uniform indices and issue task (32).   
10: end for   
11: Choose R uniformly from $\{ 0 , \ldots , T - 1 \}$ and return $x _ { R } .$

## 4.2 Convergence guarantee

Theorem 2. Under Assumptions 1 and 3, suppose that (4) holds. For any positive integers $q , b , T$ Algorithm 2 with

$$
\eta = \frac { 1 } { 2 L ( 1 + \sqrt { q / b } + \operatorname* { m i n } \{ \tau , q - 1 \} ) }\tag{34}
$$

guarantees

$$
\mathbb { E } \| \nabla f ( x _ { R } ) \| ^ { 2 } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta T } .\tag{35}
$$

The number of issued component-gradient evaluations is at most

$$
n \left\lceil { \frac { T } { q } } \right\rceil + 2 b T .\tag{36}
$$

Corollary 1. Set $q = n , b = 1$ , choose η as in (34), and let

$$
T = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { 2 \Delta _ { f } } { \eta \epsilon ^ { 2 } } \right\rceil \right\} .
$$

Then Algorithm 2 satisfies (2) with at most $n + 3 T$ evaluations. Its oracle complexity is

$$
O \left( n + \frac { L \Delta _ { f } ( \sqrt { n } + \operatorname* { m i n } \{ \tau , n - 1 \} ) } { \epsilon ^ { 2 } } \right) .\tag{37}
$$

Remark 3. For $\tau = O ( { \sqrt { n } } )$ , the order in (37) agrees with the usual finite-sum variance-reduction complexity. The delay contribution is at most $n - 1$ , because a correction ceases to afect the estimator at the next exact refresh.

Corollary 2. For an integer $1 \leq b \leq { \sqrt { n } }$ , set $q = \lceil n / b \rceil$ , choose $\eta$ as in (34), and let $T =$ max $\{ 1 , \lceil 2 \Delta _ { f } / ( \eta \epsilon ^ { 2 } ) \rceil \}$ . The number of updates and the number $Q$ of issued evaluations satisfy

$$
T = O \left( 1 + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \left( 1 + \frac { \sqrt { n } } { b } + \tau \right) \right) ,\tag{38}
$$

$$
Q = O \bigg ( n + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \left( \sqrt { n } + b ( 1 + \tau ) \right) \bigg ) .\tag{39}
$$

Thus any choice $b \leq \sqrt { n } / ( 1 + \tau )$ preserves the order $O ( n + L \Delta _ { f } \sqrt { n } / \epsilon ^ { 2 } )$ of the oracle work while reducing the bound on the number of updates. Turning this tradeof into an elapsed-time guarantee requires a service model for the batches and refreshes.

## 4.3 Proof of Theorem 2

We first control the estimation error. For round t, denote the unfinished corrections from its current epoch by

$$
\mathcal M _ { t } = \{ j : r _ { t } \le j < t , \ A _ { j } > t \} .\tag{40}
$$

Under (4), every index in $\mathcal { M } _ { t }$ lies between max $\{ r _ { t } , t - \tau \}$ and $t - 1$ . There are at most $q - 1$ preceding rounds in the current epoch, so $| \mathcal { M } _ { t } | \leq \operatorname* { m i n } \{ \tau , q - 1 \}$

Lemma 4. Under the conditions of Theorem ${ \mathcal { Q } } ,$

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq 2 L ^ { 2 } \left( \frac { q - 1 } { b } + \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } \right) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } .\tag{41}
$$

Proof. For task $j ,$ define the centered batch error by

$$
\zeta _ { j } = \delta _ { j } - \bigl ( \nabla f ( x _ { j + 1 } ) - \nabla f ( x _ { j } ) \bigr ) .
$$

Both query points are fixed before the indices of this task are drawn. For any fixed $x , y .$ uniform sampling gives

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } ( \nabla f _ { i } ( y ) - \nabla f _ { i } ( x ) ) = \nabla f ( y ) - \nabla f ( x ) .
$$

Thus every centered sampled diference has zero mean when averaging its new index. Cross terms between distinct sampled diferences also have zero mean, because the indices in the batch are independent. Expanding the square of the batch average therefore gives

$$
\begin{array} { r l } {  { \mathbb { E } \| \zeta _ { j } \| ^ { 2 } = \frac { 1 } { b ^ { 2 } } \sum _ { k = 1 } ^ { b } \mathbb { E } \| \nabla f _ { i _ { j , k } } ( x _ { j + 1 } ) - \nabla f _ { i _ { j , k } } ( x _ { j } ) \| ^ { 2 } } } \\ & { \quad = \frac { 1 } { b n } \displaystyle \sum _ { i = 1 } ^ { n } \mathbb { E } \| \nabla f _ { i } ( x _ { j + 1 } ) - \nabla f _ { i } ( x _ { j } ) \| ^ { 2 } } \\ & { \quad \quad \quad - \frac { 1 } { b } \mathbb { E } \| \nabla f ( x _ { j + 1 } ) - \nabla f ( x _ { j } ) \| ^ { 2 } } \\ & { \quad \le \displaystyle \frac { 1 } { b n } \sum _ { i = 1 } ^ { n } \mathbb { E } \| \nabla f _ { i } ( x _ { j + 1 } ) - \nabla f _ { i } ( x _ { j } ) \| ^ { 2 } . } \end{array}
$$

The second equality uses uniform sampling and the identity $\mathbb { E } \left. Z - \mathbb { E } Z \right. ^ { 2 } = \mathbb { E } \left. Z \right. ^ { 2 } - \left. \mathbb { E } Z \right. ^ { 2 }$ with the query points held fixed. Assumption 3 now gives

$$
\mathbb { E } \| \zeta _ { j } \| ^ { 2 } \leq \frac { L ^ { 2 } } { b } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } .\tag{42}
$$

We also need a bound on $\delta _ { j }$ , which includes its mean. Convexity of the squared norm and uniform sampling give

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } \left\| \delta _ { j } \right\| ^ { 2 } \leq \frac { 1 } { b } \displaystyle \sum _ { k = 1 } ^ { b } \mathbb { E } \left\| \nabla f _ { i _ { j , k } } ( x _ { j + 1 } ) - \nabla f _ { i _ { j , k } } ( x _ { j } ) \right\| ^ { 2 } } \\ & { \displaystyle = \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \mathbb { E } \left\| \nabla f _ { i } ( x _ { j + 1 } ) - \nabla f _ { i } ( x _ { j } ) \right\| ^ { 2 } } \\ & { \displaystyle \leq L ^ { 2 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } . } \end{array}\tag{43}
$$

The full sequence of population diferences telescopes:

$$
\nabla f ( x _ { t } ) = \nabla f ( x _ { r _ { t } } ) + \sum _ { j = r _ { t } } ^ { t - 1 } \left( \nabla f ( x _ { j + 1 } ) - \nabla f ( x _ { j } ) \right) .
$$

To subtract this identity from (33), first write the sum of returned corrections as

$$
\sum _ { r _ { t } \leq j < t \atop A _ { j } \leq t } \delta _ { j } = \sum _ { j = r _ { t } } ^ { t - 1 } \delta _ { j } - \sum _ { j \in { \mathcal { M } } _ { t } } \delta _ { j } .
$$

It follows that

$$
v _ { t } - \nabla f ( x _ { t } ) = \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } - \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } .\tag{44}
$$

The first sum contains all generated sampling errors in the epoch, rather than just the returned errors. This allows us to control it using fresh sampling even when completion times depend on the samples.

For $i < j$ , the vector $\zeta _ { i }$ and the point pair of task $j$ have already been determined when the new indices of task $j$ are drawn. Averaging these new indices first gives E $\langle \zeta _ { i } , \zeta _ { j } \rangle = 0$ . This argument does not require the two point pairs to be independent. Since $r _ { t }$ is a deterministic epoch boundary, expansion of the squared norm gives

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left\| \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } \right\| ^ { 2 } = \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \left\| \zeta _ { j } \right\| ^ { 2 } + 2 \sum _ { r _ { t } \leq i < j < t } \mathbb { E } \left. \zeta _ { i } , \zeta _ { j } \right. } & { } \\ { \displaystyle = \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \left\| \zeta _ { j } \right\| ^ { 2 } } & { } \\ { \displaystyle } & { \leq \frac { L ^ { 2 } } { b } \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } . } \end{array}\tag{45}
$$

To sum this inequality over $t ,$ fix an index $j .$ It appears only after it is generated and before the next refresh, namely,

$$
j + 1 \le t \le q ( \lfloor j / q \rfloor + 1 ) - 1 .
$$

There are $q ( \lfloor j / q \rfloor + 1 ) - j - 1 \leq q - 1$ such rounds. In particular, the correction issued in the last

round of an epoch appears zero times. Reversing the order of summation makes this count explicit:

$$
\begin{array} { r l } {  { \sum _ { t = 0 } ^ { T - 1 } \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } } } \\ & { = \displaystyle \sum _ { j = 0 } ^ { T - 1 } \sum _ { t = j + 1 } ^ { \operatorname* { m i n } \{ T - 1 , q ( \lfloor j / q \rfloor + 1 ) - 1 \} } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } } \\ & { \le ( q - 1 ) \displaystyle \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } . } \end{array}
$$

Here and below, a sum is zero if its upper endpoint is smaller than its lower endpoint. Applying this bound to (45) yields

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } \right. ^ { 2 } \leq \frac { L ^ { 2 } ( q - 1 ) } { b } \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \left. x _ { j + 1 } - x _ { j } \right. ^ { 2 } .\tag{46}
$$

To bound the second sum in (44), we work with each realized execution before taking expectations. Cauchy–Schwarz gives

$$
\begin{array} { r l r } {  { \| \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } \| ^ { 2 } \leq | { \cal M } _ { t } | \sum _ { j \in \mathcal { M } _ { t } } \| \delta _ { j } \| ^ { 2 } } } \\ & { } & { \leq \operatorname* { m i n } \{ \tau , q - 1 \} \sum _ { j = \operatorname* { m a x } \{ r _ { t } , t - \tau \} } ^ { t - 1 } \| \delta _ { j } \| ^ { 2 } . } \end{array}\tag{47}
$$

The last sum has deterministic endpoints, so taking expectations does not retain a sample-dependent arrival indicator. For a fixed $j ,$ , membership in this interval requires both $j < t \leq j + \tau$ and that t remain in the epoch of $j .$ . Consequently,

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \sum _ { j = \operatorname* { m a x } \{ r _ { t } , t - \tau \} } ^ { t - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } \\ { \displaystyle \quad = \sum _ { j = 0 } ^ { T - 1 } \operatorname* { m i n } \{ T - 1 , j + \tau , q ( \lfloor j / q \rfloor + 1 ) - 1 \} } \\ { \displaystyle \quad \leq \operatorname* { m i n } \{ \tau , q - 1 \} \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } . } \end{array}
$$

The inner sum contains at most τ rounds because $t \le j + \tau$ , and at most $q - 1$ rounds because of the next refresh. Summing (47) and applying this count gives

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } \| ^ { 2 } \leq \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } } \\ & { } & { \leq L ^ { 2 } \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } . } \end{array}\tag{48}
$$

Empty sums are zero, so these inequalities also cover $\tau = 0$ and $q = 1$ . Finally, apply $\left\| a - b \right\| ^ { 2 } \leq$ $2 \| a \| ^ { 2 } + 2 \| b \| ^ { 2 }$ to (44) and sum:

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq 2 \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } \right\| ^ { 2 } + 2 \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } \right\| ^ { 2 } } & { } \\ { \leq 2 L ^ { 2 } \left( \displaystyle \frac { q - 1 } { b } + \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } \right) \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } . } \end{array}
$$

This is (41).

Proof of Theorem 2. The update gives $\| x _ { t + 1 } - x _ { t } \| ^ { 2 } = \eta ^ { 2 } \| v _ { t } \| ^ { 2 }$ . We first use (6) to separate the estimation error from the descent term:

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta } - ( 1 - L \eta ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } + \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } } } \\ & { } & { \leq \frac { 2 \Delta _ { f } } { \eta } - ( 1 - L \eta ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } } \\ & { } & { \quad + 2 L ^ { 2 } \eta ^ { 2 } ( \frac { q - 1 } { b } + \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } , } \end{array}
$$

where the second inequality is Lemma 4. Collecting the two coeficients of $\textstyle \sum _ { t } \mathbb { E } \| v _ { t } \| ^ { 2 }$ yields

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla f ( x _ { t } ) \right. ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta } - \left[ 1 - L \eta - 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q - 1 } { b } + \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } \right) \right] \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. v _ { t } \right. ^ { 2 } .\tag{49}
$$

For the learning rate in (34),

$$
L \eta = \frac { 1 } { 2 ( 1 + \sqrt { q / b } + \operatorname* { m i n } \{ \tau , q - 1 \} ) } \leq \frac { 1 } { 2 } .
$$

Moreover,

$$
\begin{array} { r l } & { 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q - 1 } { b } + \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } \right) \leq \frac { q / b + \operatorname* { m i n } \{ \tau , q - 1 \} ^ { 2 } } { 2 ( 1 + \sqrt { q / b } + \operatorname* { m i n } \{ \tau , q - 1 \} ) ^ { 2 } } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \leq \frac { 1 } { 2 } . } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad \quad  \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

Hence the bracket in (49) is at least $1 - 1 / 2 - 1 / 2 = 0$ . Dropping its nonpositive contribution and dividing by $T$ gives

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. \nabla f ( x _ { t } ) \right. ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta T } .
$$

Since $R$ is sampled uniformly after the run, its output satisfies

$$
\mathbb { E } \| \nabla f ( x _ { R } ) \| ^ { 2 } = \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { T } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } ,
$$

which proves (35).

The full gradients are requested at $0 , q , \ldots , q \lfloor ( T - 1 ) / q \rfloor$ . Their number is $1 + \lfloor ( T - 1 ) / q \rfloor = \lceil T / q \rceil$ and each uses n evaluations. Each of the T rounds also issues one task containing b gradient pairs. Counting both gradients in every pair, including pairs discarded after a refresh or still unfinished at termination, gives 2bT evaluations. Adding the two costs proves (36). □

Proof of Corollary 1. The definition of $T$ ensures $T \geq 2 \Delta _ { f } / ( \eta \epsilon ^ { 2 } )$ , so Theorem 2 gives the required accuracy. It also ensures

$$
T \leq 1 + \frac { 2 \Delta _ { f } } { \eta \epsilon ^ { 2 } } = 1 + \frac { 4 L \Delta _ { f } } { \epsilon ^ { 2 } } \big ( 1 + \sqrt { n } + \operatorname* { m i n } \{ \tau , n - 1 \} \big ) .
$$

This bound includes $\Delta _ { f } = 0$ , in which case $T = 1$ . For the oracle count,

$$
\begin{array} { r l r } {  { n \lceil \frac { T } { n } \rceil + 2 T \le n ( \frac { T } { n } + 1 ) + 2 T = n + 3 T } } \\ & { } & { \le n + 3 + \frac { 1 2 L \Delta _ { f } } { \epsilon ^ { 2 } } ( 1 + \sqrt { n } + \operatorname* { m i n } \{ \tau , n - 1 \} ) . } \end{array}
$$

Using $n + 3 \leq$ 4n and $1 + { \sqrt { n } } \leq 2 { \sqrt { n } }$ gives (37).

Proof of Corollary 2. Since $q = \lceil n / b \rceil$ , we have

$$
{ \frac { n } { b } } \leq q \leq { \frac { n } { b } } + 1 , \qquad { \sqrt { { \frac { q } { b } } } } \leq { \sqrt { { \frac { n } { b ^ { 2 } } } + { \frac { 1 } { b } } } } \leq { \frac { \sqrt { n } } { b } } + { \frac { 1 } { \sqrt { b } } } \leq { \frac { \sqrt { n } } { b } } + 1 .
$$

Also, min $\{ \tau , q - 1 \} \leq \tau$ . Substitution into (34) therefore gives

$$
\begin{array} { r l } & { T \le 1 + \displaystyle \frac { 2 \Delta _ { f } } { \eta \epsilon ^ { 2 } } } \\ & { \quad = 1 + \displaystyle \frac { 4 L \Delta _ { f } } { \epsilon ^ { 2 } } \bigl ( 1 + \sqrt { q / b } + \operatorname* { m i n } \{ \tau , q - 1 \} \bigr ) } \\ & { \quad \le 1 + \displaystyle \frac { 4 L \Delta _ { f } } { \epsilon ^ { 2 } } \left( 2 + \frac { \sqrt { n } } { b } + \tau \right) , } \end{array}
$$

which proves (38). For the evaluation count, $q \geq n / b$ implies $n / q \leq b ,$ , and hence

$$
\begin{array} { l } { \displaystyle Q \leq n \left\lceil \frac T q \right\rceil + 2 b T } \\ { \leq a + \left( \frac n q + 2 b \right) T \leq n + 3 b T } \\ { \leq n + 3 b + \frac { 1 2 L \Delta _ { f } } { \epsilon ^ { 2 } } ( 2 b + \sqrt { n } + b \tau ) . } \end{array}
$$

Finally, $b \leq { \sqrt { n } } \leq n$ gives $n + 3 b \leq 4 n .$ , proving (39). The target accuracy follows from the same choice $T \geq 2 \Delta _ { f } / ( \eta \epsilon ^ { 2 } )$ and Theorem 2. □

Algorithm 3 Delayed recursive variance reduction with overlapping snapshots   
Require: Initial point $x _ { 0 } .$ update number T, period $q ,$ batch size $b ,$ full-gradient deadline $\overline { { D \geq 1 } }$   
learning rate η   
1: Compute $\nabla f ( x _ { 0 } )$ and wait for all n component evaluations.   
2: for $t = 0 , \ldots , T - 1$ do   
3: Receive all feedback available at the beginning of round t.   
4: if $t \geq q$ and t is a multiple of $q$ then   
5: Request $\nabla f ( x _ { t } )$ at the fixed point $x _ { t } ,$ for use at round $t + D .$   
6: end if   
7: Set $r _ { t }$ by (50) and form $v _ { t }$ by (51).   
8: Update $x _ { t + 1 } = x _ { t } - \eta v _ { t } .$   
9: Draw b fresh uniform indices and issue task (32).   
10: end for   
11: Choose R uniformly from $\{ 0 , \ldots , T - 1 \}$ and return $x _ { R } .$

## 5 Finite-Sum Updates without Refresh Waits

Algorithm 2 waits for an exact gradient at every refresh. We now allow these gradients to be computed while the server continues updating. The correction tasks remain unchanged. The additional requirement concerns the completion time of an exact full gradient.

Assumption 4. Fix an integer $D \geq 1$ . At every round $r = q , 2 q , . . . ,$ the server can request the exact gradient $\nabla f ( x _ { r } )$ at the fixed point $x _ { r } .$ All n component evaluations in this request are completed and their average is available before the update at round $r + D$ . The server can continue to issue correction tasks and update the parameter while the request is being processed.

Assumption 4 is separate from (4). The latter bounds the delay of a two-point correction and it does not bound the time needed to complete n component evaluations.

## 5.1 The proposed method

We call an exact gradient at a fixed earlier point a snapshot. The snapshot requested at round r is used starting at the prescribed round $r + D$ , even if it returns sooner. Consequently, the snapshot used at round t has index

$$
r _ { t } = \operatorname* { m a x } \left\{ 0 , \ q \left\lfloor \frac { t - D } { q } \right\rfloor \right\} .\tag{50}
$$

The value $r _ { t } = 0$ refers to the completed initialization gradient. Define

$$
\boldsymbol { v } _ { t } = \nabla f ( \boldsymbol { x } _ { r _ { t } } ) + \sum _ { \substack { r _ { t } \leq j < t } } \delta _ { j } ,\tag{51}
$$

where $\delta _ { j }$ is the same batch of component-gradient diferences as in (32). The point continues to move while a snapshot is being computed. When that snapshot is used, the returned corrections generated since its point was fixed are retained.

## 5.2 Convergence guarantee

Theorem 3. Under Assumptions $\scriptstyle 1 , \ 3 ,$ and $^ { \downarrow , }$ suppose that correction tasks satisfy (4). For positive integers $q , b , T$ and $D \geq 1$ , Algorithm 3 with

$$
\eta = \frac { 1 } { 2 L \left( 1 + \sqrt { ( q + D ) / b } + \tau \right) }\tag{52}
$$

satisfies

$$
\mathbb { E } \| \nabla f ( x _ { R } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta T } .\tag{53}
$$

The number of issued component-gradient evaluations is at most

$$
n \left\lceil { \frac { T } { q } } \right\rceil + 2 b T .\tag{54}
$$

Corollary 3. Set $q = n , b = 1$ , choose η as in (52), and let $T = \operatorname* { m a x } \{ 1 , \lceil 2 \Delta _ { f } / ( \eta \epsilon ^ { 2 } ) \rceil \}$ . Then Algorithm 3 satisfies (2) with oracle complexity

$$
O \left( n + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \left( \sqrt { n } + \sqrt { D } + \tau \right) \right) .\tag{55}
$$

The server does not wait for a refresh after initialization. If the service model gives $D = O ( n + \tau ^ { 2 } )$ , the order in (55) is $O ( n + L \Delta _ { f } ( \sqrt { n } + \tau ) / \epsilon ^ { 2 } )$

## 5.3 Proof of Theorem 3

The new ingredient is the time for which a sampling error remains in the estimator. The exact snapshot cancels all corrections preceding its anchor, even though that anchor is older than the current parameter.

Lemma 5. Under the conditions of Theorem 3,

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq 2 L ^ { 2 } \left( \frac { q + D - 1 } { b } + \tau ^ { 2 } \right) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } .\tag{56}
$$

Proof. For each task $j ,$ , define

$$
\zeta _ { j } = \delta _ { j } - \big ( \nabla f ( x _ { j + 1 } ) - \nabla f ( x _ { j } ) \big ) , \qquad \mathcal { M } _ { t } = \{ j : r _ { t } \leq j < t , \ A _ { j } > t \} .
$$

The samples for task $j$ are drawn after its point pair has been fixed. Therefore, the proof of (42) gives

$$
\mathbb { E } \| \zeta _ { j } \| ^ { 2 } \leq \frac { L ^ { 2 } } { b } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } , \qquad \mathbb { E } \| \delta _ { j } \| ^ { 2 } \leq L ^ { 2 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } .\tag{57}
$$

The first inequality follows by expanding the squared norm of the average of b centered diferences; its $b ( b - 1 )$ cross terms have zero mean. The second is (43), which bounds the uncentered correction. These bounds use only fresh sampling after the point pair is chosen, so they hold for the actual iterates of Algorithm 3 as well.

The population diferences telescope from the snapshot point to the current point:

$$
\nabla f ( x _ { t } ) = \nabla f ( x _ { r _ { t } } ) + \sum _ { j = r _ { t } } ^ { t - 1 } \left( \nabla f ( x _ { j + 1 } ) - \nabla f ( x _ { j } ) \right) .
$$

The returned corrections and missing corrections partition all indices from $r _ { t }$ to $t - 1$ . Therefore,

$$
\begin{array} { r l } { \displaystyle v _ { t } - \nabla f ( \boldsymbol { x } _ { t } ) = \sum _ { j = r _ { t } } ^ { t - 1 } \delta _ { j } - \sum _ { j \in M _ { t } } \delta _ { j } } & { } \\ { \displaystyle - \sum _ { j = r _ { t } } ^ { t - 1 } \big ( \nabla f ( \boldsymbol { x } _ { j + 1 } ) - \nabla f ( \boldsymbol { x } _ { j } ) \big ) } & { } \\ { \displaystyle = \sum _ { j = r _ { t } } ^ { t - 1 } \big [ \delta _ { j } - \big ( \nabla f ( \boldsymbol { x } _ { j + 1 } ) - \nabla f ( \boldsymbol { x } _ { j } ) \big ) \big ] - \sum _ { j \in M _ { t } } \delta _ { j } . } \end{array}
$$

By the definition of $\zeta _ { j }$ , this is the decomposition

$$
v _ { t } - \nabla f ( x _ { t } ) = \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } - \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } .\tag{58}
$$

For the first sum, the endpoints $r _ { t }$ and $t - 1$ are deterministic. If $i < j$ , then $\zeta _ { i }$ and the point pair of task $j$ are determined before the indices of task $j$ are drawn. Averaging those new indices first gives E $\langle \zeta _ { i } , \zeta _ { j } \rangle = 0$ . Expanding the square and using (57), we obtain

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left\| \displaystyle \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } \right\| ^ { 2 } = \displaystyle \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \left\| \zeta _ { j } \right\| ^ { 2 } + 2 \displaystyle \sum _ { r _ { t } \leq i < j < t } \mathbb { E } \left. \zeta _ { i } , \zeta _ { j } \right. } & { } \\ { \displaystyle } & { = \displaystyle \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \left\| \zeta _ { j } \right\| ^ { 2 } } \\ { \displaystyle } & { \leq \frac { L ^ { 2 } } { b } \displaystyle \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } . } \end{array}\tag{59}
$$

We next count the rounds containing a fixed index $j .$ . It enters the sum at round $j + 1$ . The first snapshot with anchor strictly larger than $j$ is requested at $q ( \lfloor j / q \rfloor + 1 )$ and activated $D$ rounds later. The anchor never decreases, so

$$
r _ { t } \le j < t \quad \Longleftrightarrow \quad j + 1 \le t \le q ( \lfloor j / q \rfloor + 1 ) + D - 1 .
$$

This also holds for the initial anchor 0: the first replacement is requested at $q$ and activated at $q + D$ . Thus reversal of the two sums gives

$$
\begin{array} { r l } {  { \sum _ { t = 0 } ^ { T - 1 } \sum _ { j = r _ { t } } ^ { t - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } } } \\ & { \quad = \sum _ { j = 0 } ^ { T - 1 } \overset { \operatorname* { m i n } \{ T - 1 , q ( | j / q | + 1 ) + D - 1 \} } { \underset { t = j + 1 } { \sum } } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } } \\ & { \leq ( q + D - 1 ) \displaystyle \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } . } \end{array}
$$

Indeed, without the finite-horizon cutof, the inner sum has

$$
q ( \lfloor j / q \rfloor + 1 ) + D - j - 1 = q + D - 1 - ( j - q \lfloor j / q \rfloor ) \le q + D - 1
$$

terms. The cutof at $T - 1$ can only reduce this count. Summing (59) now yields

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } \right\| ^ { 2 } \leq \frac { L ^ { 2 } ( q + D - 1 ) } { b } \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } .\tag{60}
$$

For the missing corrections, (4) implies $\mathcal { M } _ { t } \subseteq \{ \operatorname* { m a x } \{ r _ { t } , t - \tau \} , \dots , t - 1 \}$ and $| \mathcal { M } _ { t } | \le \tau$ . On each realized execution, Cauchy–Schwarz gives

$$
\begin{array} { r l } {  {  \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j }  ^ { 2 } \leq \vert \mathcal { M } _ { t } \vert \sum _ { j \in \mathcal { M } _ { t } } \| \delta _ { j } \| ^ { 2 } } } \\ & { \leq \tau \sum _ { j = \operatorname* { m a x } \{ r _ { t } , t - \tau \} } ^ { t - 1 } \| \delta _ { j } \| ^ { 2 } } \\ & { \leq \tau \sum _ { j = \operatorname* { m a x } \{ 0 , t - \tau \} } ^ { t - 1 } \| \delta _ { j } \| ^ { 2 } . } \end{array}
$$

The last step only adds nonnegative terms and removes the anchor from the bound. For a fixed $j ,$ the remaining interval condition is $j + 1 \leq t \leq j + \tau$ . Therefore,

$$
\begin{array} { r l r }   { \sum _ { t = 0 } ^ { T - 1 } \sum _ { j = \operatorname* { m a x } \{ 0 , t - \tau \} } ^ { t - 1 } \mathbb { E }  \delta _ { j }  ^ { 2 } = \displaystyle \sum _ { j = 0 } ^ { T - 1 } \sum _ { t = j + 1 } ^ { m \setminus \lceil T - 1 , j + \tau \} \mathbb { E }  \delta _ { j }  ^ { 2 } } } \\ & { } & { \leq \tau \displaystyle \sum _ { j = 0 } ^ { T - 1 } \mathbb { E }  \delta _ { j }  ^ { 2 } . } \end{array}
$$

Taking expectations in the preceding pointwise bound and using this count gives

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } \| ^ { 2 } \leq \tau ^ { 2 } \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } } \\ & { } & { \leq L ^ { 2 } \tau ^ { 2 } \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } . } \end{array}\tag{61}
$$

For $\tau = 0$ , the missing sets are empty and the same inequality holds. Finally, apply $\left\| a - b \right\| ^ { 2 } \leq$ $2 \| a \| ^ { 2 } + 2 \| b \| ^ { 2 }$ to (58):

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } \leq 2 \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } \| ^ { 2 } + 2 \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } \| ^ { 2 } } } \\ & { } & { \leq 2 L ^ { 2 } ( \frac { q + D - 1 } { b } + \tau ^ { 2 } ) \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } , } \end{array}
$$

where the last line uses (60) and (61). This proves the lemma.

Proof of Theorem 3. The update identity $\left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } = \eta ^ { 2 } \left\| v _ { t } \right\| ^ { 2 }$ and Lemma 5 give

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. v _ { t } - \nabla f ( x _ { t } ) \right. ^ { 2 } \leq 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q + D - 1 } { b } + \tau ^ { 2 } \right) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. v _ { t } \right. ^ { 2 } .
$$

Substituting this bound into (6), we obtain

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta } - ( 1 - L \eta ) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } \right\| ^ { 2 } } \\ { \displaystyle \qquad + 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q + D - 1 } { b } + \tau ^ { 2 } \right) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } \right\| ^ { 2 } . } \end{array}
$$

Combining the two coeficients of $\textstyle \sum _ { t } \mathbb { E } \left\| v _ { t } \right\| ^ { 2 }$ gives

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Vert \nabla f ( x _ { t } ) \Vert ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta } - \left[ 1 - L \eta - 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q + D - 1 } { b } + \tau ^ { 2 } \right) \right] \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \Vert v _ { t } \Vert ^ { 2 } .\tag{62}
$$

The learning rate in (52) satisfies

$$
L \eta = \frac { 1 } { 2 ( 1 + \sqrt { ( q + D ) / b } + \tau ) } \leq \frac { 1 } { 2 } ,
$$

and

$$
\begin{array} { r l } & { 2 L ^ { 2 } \eta ^ { 2 } \left( \displaystyle \frac { q + D - 1 } { b } + \tau ^ { 2 } \right) \le \displaystyle \frac { ( q + D ) / b + \tau ^ { 2 } } { 2 ( 1 + \sqrt { ( q + D ) / b } + \tau ) ^ { 2 } } } \\ & { \qquad \le \displaystyle \frac { ( \sqrt { ( q + D ) / b } + \tau ) ^ { 2 } } { 2 ( 1 + \sqrt { ( q + D ) / b } + \tau ) ^ { 2 } } \le \frac { 1 } { 2 } . } \end{array}
$$

Thus the bracket in (62) is at least $1 - 1 / 2 - 1 / 2 = 0$ . After dropping its nonpositive contribution and dividing by $T .$ , the uniform output rule gives

$$
\mathbb { E } \| \nabla f ( x _ { R } ) \| ^ { 2 } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta T } .
$$

This proves (53).

There is one full gradient at initialization. The remaining snapshots are requested at positive multiples of $q$ smaller than $T .$ . Their total number, including initialization, is $1 + \lfloor ( T - 1 ) / q \rfloor = \lceil T / q \rceil$ Every such request costs n evaluations, whether or not its deadline occurs within the run. There are also T two-point tasks with b indices each, costing 2bT evaluations. This proves (54). □

Proof of Corollary 3. The choice $T \geq 2 \Delta _ { f } / ( \eta \epsilon ^ { 2 } )$ and (53) give the target accuracy. With $q = n$ and $b = 1$ , we also have

$$
T \le 1 + \frac { 2 \Delta _ { f } } { \eta \epsilon ^ { 2 } } = 1 + \frac { 4 L \Delta _ { f } } { \epsilon ^ { 2 } } \big ( 1 + \sqrt { n + D } + \tau \big )
$$

$$
\leq 1 + \frac { 4 L \Delta _ { f } } { \epsilon ^ { 2 } } \big ( 1 + \sqrt { n } + \sqrt { D } + \tau \big ) .
$$

Here ${ \sqrt { n + D } } \leq { \sqrt { n } } + { \sqrt { D } }$ follows by squaring both nonnegative sides. Substitution into (54) yields

$$
\begin{array} { r } { n \left\lceil \displaystyle \frac { T } { n } \right\rceil + 2 T \leq n + 3 T \qquad } \\ { \leq n + 3 + \displaystyle \frac { 1 2 L \Delta _ { f } } { \epsilon ^ { 2 } } \big ( 1 + \sqrt { n } + \sqrt { D } + \tau \big ) . } \end{array}
$$

Since $n \geq 1$ , we have $n + 3 \leq$ 4n and $1 \leq { \sqrt { n } } .$ . These two bounds give (55).

## 6 The Efect of Outstanding Feedback

The bounds based on maximum delay can be conservative when only a few corrections are missing. We first account for the number of outstanding corrections. For the stochastic method, we then use their total weight in the estimator, which also allows a sequence of permanently missing responses.

The maximum number of outstanding tasks and related concurrency quantities have been used in delayed learning and asynchronous SGD [Joulani et al., 2013; Koloskova et al., 2022]. Our purpose here is to establish what these quantities imply for the two recursive variance-reduction estimators. The algorithms are unchanged.

## 6.1 A bound for missing corrections

The following inequality is deterministic. Therefore, it can be applied to every realized execution before any expectation is taken.

Lemma 6. For arbitrary vectors $\delta _ { j }$ , sets $\mathcal { M } _ { t } \subseteq \{ 0 , \ldots , t - 1 \}$ , and nonnegative weights $w _ { t , j }$

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \left\| \sum _ { j \in \mathcal { M } _ { t } } w _ { t , j } \delta _ { j } \right\| ^ { 2 } } \\ { \displaystyle \quad \leq \sum _ { j = 0 } ^ { T - 1 } \left[ \sum _ { t : j \in \mathcal { M } _ { t } } w _ { t , j } \left( \sum _ { i \in \mathcal { M } _ { t } } w _ { t , i } \right) \right] \| \delta _ { j } \| ^ { 2 } . } \end{array}\tag{63}
$$

For any $p \geq 0$ , the following consequences hold:

(i) $I f \left| \mathcal { M } _ { t } \right| \leq p , 0 \leq w _ { t , j } \leq 1$ , and each task belongs to at most τ of the sets $\mathcal { M } _ { t }$ , the right-hand side of (63) is at most $p \tau { \sum _ { j } \| \delta _ { j } \| ^ { 2 } }$

(ii) For the stochastic weights $w _ { t , j } = ( 1 - \alpha ) ^ { t - j - 1 }$ , suppose that $\textstyle \sum _ { j \in { \mathcal { M } } _ { t } } w _ { t , j } \leq p$ at every round. Then the bound is $\begin{array} { r l } {  { ( p / \alpha ) \sum _ { j } \| \delta _ { j } \| ^ { 2 } } } \end{array}$ , without an upper bound on arrival times.

(iii) For the finite-sum method with $w _ { t , j } = 1$ , epoch length q, and $| \mathcal { M } _ { t } | \le p$ , the bound is $p ( q -$ $1 ) \textstyle \sum _ { j } \| \delta _ { j } \| ^ { 2 }$ , without an upper bound on arrival times.

Proof. The triangle inequality and scalar Cauchy–Schwarz give

$$
\begin{array} { r l r } {  {  \sum _ { j \in \mathcal { M } _ { t } } w _ { t , j } \delta _ { j }  ^ { 2 } \le ( \sum _ { j \in \mathcal { M } _ { t } } w _ { t , j }  \delta _ { j }  ) ^ { 2 } } } \\ & { } & { = ( \sum _ { j \in \mathcal { M } _ { t } } \sqrt { w _ { t , j } } \sqrt { w _ { t , j } }  \delta _ { j }  ) ^ { 2 } } \\ & { } & { \le ( \sum _ { i \in \mathcal { M } _ { t } } w _ { t , i } ) ( \sum _ { j \in \mathcal { M } _ { t } } w _ { t , j }  \delta _ { j }  ^ { 2 } ) . } \end{array}
$$

Empty sums are zero. Summing over t and changing the order of the two finite sums yields

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \left\| \displaystyle \sum _ { j \in M _ { t } } w _ { t , j } \delta _ { j } \right\| ^ { 2 } } \\ & { \quad \le \displaystyle \sum _ { t = 0 } ^ { T - 1 } \displaystyle \sum _ { j \in M _ { t } } w _ { t , j } \left( \displaystyle \sum _ { i \in M _ { t } } w _ { t , i } \right) \| \delta _ { j } \| ^ { 2 } } \\ & { \quad = \displaystyle \sum _ { j = 0 } ^ { T - 1 } \left[ \displaystyle \sum _ { 0 \le t \le T _ { t } } w _ { t , j } \left( \displaystyle \sum _ { i \in \mathcal { M } _ { t } } w _ { t , i } \right) \right] \| \delta _ { j } \| ^ { 2 } . } \end{array}
$$

This is (63); the coeficient for $j = T - 1$ is zero.

For (i), the coeficient of each $\| \delta _ { j } \| ^ { 2 }$ satisfies

$$
\begin{array} { r } { \displaystyle \sum _ { t : j \in \mathcal { M } _ { t } } w _ { t , j } \left( \displaystyle \sum _ { i \in \mathcal { M } _ { t } } w _ { t , i } \right) \leq \displaystyle \sum _ { t : j \in \mathcal { M } _ { t } } 1 \cdot | \mathcal { M } _ { t } | } \\ { \leq p \displaystyle \sum _ { t : j \in \mathcal { M } _ { t } } 1 \leq p \tau . } \end{array}
$$

For (ii), use the assumed bound on the total missing weight and then sum a geometric series:

$$
\begin{array} { r l } {  { \sum _ { t : j \in \mathcal { M } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } ( \sum _ { i \in \mathcal { M } _ { t } } ( 1 - \alpha ) ^ { t - i - 1 } ) } } \\ & { \leq p \sum _ { t = j + 1 } ^ { \infty } ( 1 - \alpha ) ^ { t - j - 1 } = p \sum _ { k = 0 } ^ { \infty } ( 1 - \alpha ) ^ { k } = \frac { p } { \alpha } . } \end{array}
$$

For (iii), let r be the first round of the epoch containing task $j .$ . Then $r \leq j \leq r + q - 1$ , and

$$
j \in \mathcal { M } _ { t } \quad \implies \quad j + 1 \leq t \leq r + q - 1 .
$$

There are at most $r + q - 1 - j \leq q - 1$ such rounds. Since every weight is one,

$$
\sum _ { t : j \in \mathcal { M } _ { t } } \left( \sum _ { i \in \mathcal { M } _ { t } } 1 \right) \leq p ( r + q - 1 - j ) \leq p ( q - 1 ) .
$$

Substituting these three coeficient bounds into (63) proves the claims. Every inequality holds for the realized sets and vectors, even when the sets depend on the vectors. □

The bounds $p$ and $\tau$ used below are fixed upper bounds valid throughout the run, up to events of probability zero.

## 6.2 Few outstanding tasks under bounded delay

Corollary 4. Suppose that (4) holds and $| \mathcal { M } _ { t } | \le p$ at every round. For the finite-sum method, $\mathcal { M } _ { t }$ contains only current-epoch corrections. Theorem 1 remains valid after replacing $\tau$ by $\sqrt { p \tau }$ in the choices of η, α, and $T _ { i }$ and then choosing B by (16). In Theorem ${ \mathcal { Q } } ,$ replace the term min $\{ \tau , q - 1 \}$ } in the denominator of the learning rate by

$$
\sqrt { \operatorname* { m i n } \{ p , \tau , q - 1 \} \operatorname* { m i n } \{ \tau , q - 1 \} } .
$$

With $q = n$ and $b = 1$ for the finite-sum method, the two complexities become

$$
O \left( 1 + \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \left( 1 + \frac { \sigma } { \epsilon } + \sqrt { p \tau } \right) \right) ,\tag{64}
$$

$$
{ \cal O } \left( n + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \left( \sqrt { n } + \sqrt { \operatorname* { m i n } \{ p , \tau , n - 1 \} \operatorname* { m i n } \{ \tau , n - 1 \} } \right) \right) ,\tag{65}
$$

respectively.

Proof. For the stochastic method, a task $j$ can be missing only at rounds $j + 1 , \dotsc , j + \tau$ . Lemma $6 ( \mathrm { i } )$ and (23) therefore give

$$
\begin{array} { r l } {  { \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \sum _ { j \in \mathcal { A } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } \| ^ { 2 } } } \\ & { \leq p \tau \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } \\ & { \leq 2 L ^ { 2 } p \tau \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } } \\ & { \qquad + 2 \alpha ^ { 2 } p \tau \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { j } ) \| ^ { 2 } + 2 \alpha ^ { 2 } p \tau \sigma ^ { 2 } T . } \end{array}
$$

Combining this with (21), (22), and (6), exactly as in (30), yields

$$
\begin{array} { r l } & { \displaystyle \left( 1 - 4 \alpha ^ { 2 } p \tau \right) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } } \\ & { \quad \quad + \left[ 1 - L \eta - 4 L ^ { 2 } \eta ^ { 2 } \left( \displaystyle \frac { 1 } { \alpha } + p \tau \right) \right] \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } } \\ & { \quad \quad \le \displaystyle \frac { 2 \Delta _ { f } } { \eta } + \displaystyle \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \displaystyle \frac { 1 } { \alpha } \right\} + 4 \sigma ^ { 2 } ( \alpha + \alpha ^ { 2 } p \tau ) T . } \end{array}
$$

This is (30) with $\tau ^ { 2 }$ replaced by $p \tau$ . In particular, using the parameters with $\sqrt { p \tau }$ in place of $\tau$ gives

$$
\begin{array} { c } { { 4 \alpha ^ { 2 } p \tau = \displaystyle \frac { p \tau } { 6 4 ( 1 + \sigma / \epsilon ) ^ { 2 } ( 1 + \sigma / \epsilon + \sqrt { p \tau } ) ^ { 2 } } \le \displaystyle \frac { 1 } { 6 4 } , } } \\ { { L \eta \le \displaystyle \frac { 1 } { 1 6 } , \quad \quad \displaystyle \frac { 4 L ^ { 2 } \eta ^ { 2 } } { \alpha } = \displaystyle \frac { 1 + \sigma / \epsilon } { 4 ( 1 + \sigma / \epsilon + \sqrt { p \tau } ) } \le \displaystyle \frac { 1 } { 4 } , } } \\ { { 4 L ^ { 2 } \eta ^ { 2 } p \tau = \displaystyle \frac { p \tau } { 6 4 ( 1 + \sigma / \epsilon + \sqrt { p \tau } ) ^ { 2 } } \le \displaystyle \frac { 1 } { 6 4 } . } } \end{array}
$$

Thus the gradient coeficient is at least $6 3 / 6 4$ , and the estimator coeficient is at least $4 3 / 6 4$ . The same substitutions give

$$
\frac { 2 \Delta _ { f } } { \eta T } \leq \frac { \epsilon ^ { 2 } } { 2 } ,
$$

$$
\frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { \alpha T } \right\} \leq \frac { \epsilon ^ { 2 } } { 8 } ,
$$

$$
4 \alpha \sigma ^ { 2 } \leq \frac { \epsilon ^ { 2 } } { 4 } ,
$$

$$
4 \alpha ^ { 2 } p \tau \sigma ^ { 2 } \leq \frac { \epsilon ^ { 2 } } { 6 4 } .
$$

Dropping the estimator term and dividing by T therefore gives

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { \epsilon ^ { 2 } / 2 + \epsilon ^ { 2 } / 8 + \epsilon ^ { 2 } / 4 + \epsilon ^ { 2 } / 6 4 } { 6 3 / 6 4 } = \frac { 1 9 } { 2 1 } \epsilon ^ { 2 } .
$$

Counting $B + 2 T$ evaluations proves (64).

For the finite-sum method, at a round t in an epoch beginning at $r _ { \mathrm { { ; } } }$

$$
\mathcal { M } _ { t } \subseteq \{ \operatorname* { m a x } \{ r , t - \tau \} , \ldots , t - 1 \} , \qquad | \mathcal { M } _ { t } | \leq \operatorname* { m i n } \{ p , \tau , q - 1 \} .
$$

A task $j$ from this epoch can be missing only when

$$
j + 1 \leq t \leq \operatorname* { m i n } \{ j + \tau , r + q - 1 \} .
$$

The number of these rounds is at most min $\{ \tau , q - 1 \}$ . Lemma 6 and (43) now imply

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \displaystyle \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } \| ^ { 2 } } } \\ & { } & { \leq \operatorname* { m i n } \{ p , \tau , q - 1 \} \operatorname* { m i n } \{ \tau , q - 1 \} \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } \\ & { } & { \leq L ^ { 2 } \operatorname* { m i n } \{ p , \tau , q - 1 \} \operatorname* { m i n } \{ \tau , q - 1 \} \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } . } \end{array}
$$

The complete-feedback bound (46) is unchanged. Consequently, the estimator coeficient after descent is

$$
1 - L \eta - 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q - 1 } { b } + \operatorname * { m i n } \{ p , \tau , q - 1 \} \operatorname * { m i n } \{ \tau , q - 1 \} \right) .
$$

The stated learning rate makes this coeficient nonnegative: $L \eta \leq 1 / 2$ , and

$$
\begin{array} { l } { \displaystyle \frac { q - 1 } { b } + \operatorname* { m i n } \{ p , \tau , q - 1 \} \operatorname* { m i n } \{ \tau , q - 1 \} } \\ { \displaystyle \quad \leq \bigg ( 1 + \sqrt { q / b } + \sqrt { \operatorname* { m i n } \{ p , \tau , q - 1 \} \operatorname* { m i n } \{ \tau , q - 1 \} } \bigg ) ^ { 2 } = \frac { 1 } { 4 L ^ { 2 } \eta ^ { 2 } } . } \end{array}
$$

It follows that $\mathbb { E } \left\| \nabla f ( x _ { R } ) \right\| ^ { 2 } \leq 2 \Delta _ { f } / ( \eta T )$ . At $q = n , b = 1$ , the count is at most $n + 3 T$ ; substituting this learning rate and the prescribed T proves (65). □

Under the delay assumption alone, $p = \tau$ recovers the original results. For example, suppose that one correction is delayed for $\tau \geq 1$ rounds and all others return for the next update. Then $p = 1$ and its contribution is controlled by $\sqrt { \tau }$ . For the finite-sum method, the same comparison applies within an epoch long enough to contain the delayed interval. Related dependence on $\sqrt { p \tau }$ is known for asynchronous SGD [Koloskova et al., 2022].

## 6.3 No maximum-delay assumption

We now allow $A _ { j } \in \{ j + 1 , j + 2 , \dots \} \cup \{ \infty \}$ , where $A _ { j } = \infty$ means that the response never returns. The initialization batch and finite-sum full gradients must still be returned in full. Delay and non-return decisions may depend on already generated samples but not on future samples.

One possible condition is a bound on the number of missing responses:

$$
\left| { \mathcal { M } } _ { t } \right| \leq p \qquad { \mathrm { f o r ~ e v e r y ~ r o u n d ~ } } t .\tag{66}
$$

For the stochastic estimator, each old correction has a decreasing weight. We can therefore use the weaker condition

$$
\sum _ { j \in \mathcal { M } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \leq p \qquad \mathrm { f o r ~ e v e r y ~ r o u n d ~ } t ,\tag{67}
$$

where $\mathcal { M } _ { t }$ includes every historical response that has not returned. Condition (66) implies (67), but (67) does not bound the number of permanently missing responses.

Theorem 4 (Stochastic optimization with bounded missing weight). Under Assumptions 1 and 2, fix $p \geq 0$ and $\epsilon > 0$ , and choose

$$
\eta = \frac { 1 } { 1 6 L ( 1 + p ) ( 1 + \sigma / \epsilon ) } ,
$$

$$
\alpha = \frac { 1 } { 1 6 ( 1 + p ) ( 1 + \sigma / \epsilon ) ^ { 2 } } ,\tag{68}
$$

$$
T = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { 1 2 8 L \Delta _ { f } ( 1 + p ) ( 1 + \sigma / \epsilon ) } { \epsilon ^ { 2 } } \right\rceil \right\} .\tag{69}
$$

Choose B by (16) with these values of α and T. If (67) holds for this choice of $\alpha _ { ; }$ then Algorithm 1 guarantees

$$
\mathbb { E } \left\| \nabla f ( x _ { R } ) \right\| ^ { 2 } \leq \epsilon ^ { 2 } .\tag{70}
$$

The method issues at most $B + 2 T$ evaluations, giving the oracle complexity

$$
O \left( 1 + \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { L \Delta _ { f } ( 1 + p ) } { \epsilon ^ { 2 } } \left( 1 + \frac { \sigma } { \epsilon } \right) \right) .\tag{71}
$$

The order of these choices matters: p is prescribed first, the algorithm then fixes $\alpha ,$ and the theorem applies to schedules satisfying (67) for that α. A value estimated from the completed run is not used to set the learning rate.

Proof. Lemma 6(ii) first removes the missing sets from the bound. Taking expectations and applying

(23), we obtain

$$
\begin{array} { r l } & { \displaystyle \sum _ { i = 0 } ^ { T - 1 } \mathbb { E } \bigg \| \sum _ { j \in \mathcal { M } _ { \varepsilon } } \big \| \sum _ { \varepsilon = \mathcal { M } _ { j } } \bigg \| ^ { 2 } } \\ & { \displaystyle \leq \frac { p ^ { T - 1 } } { \alpha } \| \varepsilon \| _ { \mathcal { S } } q ^ { 2 } } \\ & { \quad \leq \frac { p ^ { T - 1 } } { \alpha } \| \beta _ { j } \| ^ { 2 } } \\ & { \quad \leq \frac { p ^ { T - 1 } } { \alpha } \bigg ( 2 L ^ { p } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } + 2 \alpha ^ { 2 } \mathbb { E } \| \nabla f ( x _ { j } ) \| ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } \bigg ) } \\ & { \quad \quad = \frac { 2 L ^ { p } } { \alpha } \frac { p ^ { T - 1 } } { \beta = 0 } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } } \\ & { \quad \quad \quad + 2 \alpha \frac { T - 1 } { \beta = 0 } \mathbb { E } \| \nabla f ( x _ { j } ) \| ^ { 2 } + 2 \alpha p \sigma ^ { 2 } T . } \end{array}
$$

Using (21) and $\left. a - b \right. ^ { 2 } \leq 2 \left. a \right. ^ { 2 } + 2 \left. b \right. ^ { 2 }$ , and then (22), gives

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } \leq 2 \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| u _ { t } - \nabla f ( x _ { t } ) \| ^ { 2 } } & { } \\ & { \qquad + 2 \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \displaystyle \sum _ { j \in M _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \delta _ { j } \right\| ^ { 2 } } \\ & { \leq \displaystyle \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + \frac { 4 L ^ { 2 } ( 1 + p ) } { \alpha } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| x _ { t + 1 } - x _ { t } \| ^ { 2 } } \\ & { \qquad + 4 \alpha p \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } + 4 \alpha ( 1 + p ) \sigma ^ { 2 } T . } \end{array}\tag{72}
$$

Substitute (72) into (6), use $\left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } = \eta ^ { 2 } \left\| v _ { t } \right\| ^ { 2 }$ , and move the gradient and estimator sums to the left. Then

$$
\begin{array} { r l } & { \displaystyle \big ( 1 - 4 \alpha p \big ) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } } \\ & { \quad \quad + \left( 1 - L \eta - \frac { 4 L ^ { 2 } \eta ^ { 2 } ( 1 + p ) } { \alpha } \right) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } } \\ & { \quad \quad \le \frac { 2 \Delta _ { f } } { \eta } + \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + 4 \alpha ( 1 + p ) \sigma ^ { 2 } T . } \end{array}\tag{73}
$$

The chosen parameters satisfy

$$
\begin{array} { c } { { 4 \alpha p = \displaystyle \frac { p } { 4 ( 1 + p ) ( 1 + \sigma / \epsilon ) ^ { 2 } } \le \frac 1 4 , } } \\ { { L \eta = \displaystyle \frac { 1 } { 1 6 ( 1 + p ) ( 1 + \sigma / \epsilon ) } \le \frac 1 { 1 6 } , } } \\ { { \displaystyle \frac { 4 L ^ { 2 } \eta ^ { 2 } ( 1 + p ) } { \alpha } = \displaystyle \frac { 6 4 ( 1 + p ) ^ { 2 } ( 1 + \sigma / \epsilon ) ^ { 2 } } { 2 5 6 ( 1 + p ) ^ { 2 } ( 1 + \sigma / \epsilon ) ^ { 2 } } = \frac 1 4 . } } \end{array}
$$

Thus the two coeficients in (73) are at least

$$
1 - 4 \alpha p \ge \frac { 3 } { 4 } , \qquad 1 - L \eta - \frac { 4 L ^ { 2 } \eta ^ { 2 } ( 1 + p ) } { \alpha } \ge 1 - \frac { 1 } { 1 6 } - \frac { 1 } { 4 } = \frac { 1 1 } { 1 6 } .
$$

The choices of T and B imply

$$
T \geq \frac { 1 2 8 L \Delta _ { f } ( 1 + p ) ( 1 + \sigma / \epsilon ) } { \epsilon ^ { 2 } } = \frac { 8 \Delta _ { f } } { \eta \epsilon ^ { 2 } } ,
$$

$$
B \geq \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { \alpha T } \right\} .
$$

Consequently,

$$
\begin{array} { c } { { \displaystyle \frac { 2 \Delta _ { f } } { \eta T } \le \frac { \epsilon ^ { 2 } } { 4 } , \qquad \displaystyle \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ 1 , \displaystyle \frac { 1 } { \alpha T } \right\} \le \frac { \epsilon ^ { 2 } } { 8 } , } } \\ { { 4 \alpha ( 1 + p ) \sigma ^ { 2 } = \displaystyle \frac { \epsilon ^ { 2 } } { 4 } \left( \displaystyle \frac { \sigma / \epsilon } { 1 + \sigma / \epsilon } \right) ^ { 2 } \le \displaystyle \frac { \epsilon ^ { 2 } } { 4 } . } } \end{array}
$$

The first two bounds also hold when $\Delta _ { f } = 0$ and $\sigma = 0$ , respectively, since the corresponding numerators are zero and $T , B \geq 1$ . Dropping the nonnegative estimator term in (73) and dividing by T gives

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { \epsilon ^ { 2 } / 4 + \epsilon ^ { 2 } / 8 + \epsilon ^ { 2 } / 4 } { 3 / 4 } = \frac { 5 } { 6 } \epsilon ^ { 2 } .
$$

Since R is uniform and independent,

$$
\mathbb { E } \left. \nabla f ( x _ { R } ) \right. ^ { 2 } = \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { T } \mathbb { E } \left. \nabla f ( x _ { t } ) \right. ^ { 2 } \leq \frac { 5 } { 6 } \epsilon ^ { 2 } \leq \epsilon ^ { 2 } .
$$

Finally, max $\{ 1 , \lceil z \rceil \} \leq 1 + z { \mathrm { ~ f o r ~ } } z \geq 0$ , so

$$
\begin{array} { c } { B \leq 1 + \displaystyle \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } , } \\ { T \leq 1 + \displaystyle \frac { 1 2 8 L \Delta _ { f } ( 1 + p ) ( 1 + \sigma / \epsilon ) } { \epsilon ^ { 2 } } , } \\ { B + 2 T \leq 3 + \displaystyle \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { 2 5 6 L \Delta _ { f } ( 1 + p ) ( 1 + \sigma / \epsilon ) } { \epsilon ^ { 2 } } . } \end{array}
$$

This proves (71), including all issued tasks.

Corollary 5 (A bound on the number of outstanding responses). The conclusions of Theorem 4 hold if (66) holds for all historical missing corrections.

Proof. Since $j < t$ and $0 < 1 - \alpha < 1$ ，

$$
\sum _ { j \in \mathcal { M } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \leq \sum _ { j \in \mathcal { M } _ { t } } 1 = | \mathcal { M } _ { t } | \leq p .
$$

Thus (67) holds, and Theorem 4 applies with the same parameters.

Permanently missing responses. As an example beyond the count condition, use $p = 2$ in Theorem 4 and set

$$
m = \left\lceil { \frac { \log 2 } { \alpha } } \right\rceil .
$$

Suppose that permanent losses have generation indices at least m rounds apart, and every other response returns for the next update. At round t, write the missing indices in decreasing order as $j _ { 0 } > j _ { 1 } > \cdots > j _ { | { \mathcal { M } } _ { t } | - 1 }$ . Then

$$
j _ { 0 } \le t - 1 , \qquad j _ { k } \le j _ { 0 } - k m , \qquad t - j _ { k } - 1 \ge k m .
$$

Since $0 < 1 - \alpha < 1$ , increasing an exponent decreases its weight. Hence

$$
\begin{array} { r l } {  { \sum _ { j \in \mathcal { M } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } = \sum _ { k = 0 } ^ { | \mathcal { M } _ { t } | - 1 } ( 1 - \alpha ) ^ { t - j _ { k } - 1 } } } \\ & { \leq \sum _ { k = 0 } ^ { \infty } ( 1 - \alpha ) ^ { k m } = \frac { 1 } { 1 - ( 1 - \alpha ) ^ { m } } \leq 2 , } \end{array}
$$

where the last step follows from

$$
\log ( 1 - \alpha ) \leq - \alpha , \qquad ( 1 - \alpha ) ^ { m } \leq e ^ { - \alpha m } \leq e ^ { - \log 2 } = { \frac { 1 } { 2 } } .
$$

The empty missing set satisfies the bound as well. There is no upper bound on the total number of losses over an indefinitely long execution. Their spacing is accuracy dependent: at $p = 2$

$$
\alpha = \frac { 1 } { 4 8 ( 1 + \sigma / \epsilon ) ^ { 2 } } , \qquad \frac { \log 2 } { \alpha } \leq m < \frac { \log 2 } { \alpha } + 1 .
$$

Thus one loss every m rounds has frequency $1 / m = \Theta ( ( 1 + \sigma / \epsilon ) ^ { - 2 } )$ , which decreases as ϵ decreases when $\sigma > 0$ . Theorem 4 retains the stochastic variance-reduction order for these schedules. Every issued evaluation still counts; a missing response does not by itself mean that its physical computation has been cancelled.

Theorem 5 (Finite sums with bounded outstanding corrections). Under Assumptions 1 and 3, suppose that (66) holds for the current-epoch missing sets of Algorithm 2. No maximum delay is assumed. Choose

$$
\eta = \frac { 1 } { 2 L ( 1 + \sqrt { q / b } + \sqrt { p ( q - 1 ) } ) } .\tag{74}
$$

Then (35) and (36) hold. With $q = n , b = 1$ , and $T = \mathrm { m a x } \{ 1 , \lceil 2 \Delta _ { f } / ( \eta \epsilon ^ { 2 } ) \rceil \}$ , the oracle complexity $f o r \ ( 2 )$ is

$$
O \Bigg ( n + \frac { L \Delta _ { f } \sqrt { n ( 1 + p ) } } { \epsilon ^ { 2 } } \Bigg ) .\tag{75}
$$

Proof. By Lemma 6(iii) and (43),

$$
\begin{array} { r l r } {  { \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \sum _ { j \in \mathcal { M } _ { t } } \delta _ { j } \| ^ { 2 } \leq p ( q - 1 ) \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| \delta _ { j } \| ^ { 2 } } } \\ & { } & { \leq L ^ { 2 } p ( q - 1 ) \sum _ { j = 0 } ^ { T - 1 } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } . } \end{array}
$$

The decomposition (44) and the sampling-error estimate (46) do not use a delay bound. Applying $\left\| a - b \right\| ^ { 2 } \leq 2 \left\| a \right\| ^ { 2 } + 2 \left\| b \right\| ^ { 2 }$ therefore gives

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq 2 \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \displaystyle \sum _ { j = r _ { t } } ^ { t - 1 } \zeta _ { j } \right\| ^ { 2 } } & { } \\ { \displaystyle + 2 \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \displaystyle \sum _ { j \in M _ { t } } \delta _ { j } \right\| ^ { 2 } } & { } \\ { \leq 2 L ^ { 2 } \left( \displaystyle \frac { q - 1 } { b } + p ( q - 1 ) \right) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } . } \end{array}\tag{76}
$$

Substituting into (6) and using $x _ { t + 1 } - x _ { t } = - \eta v _ { t }$ , we obtain

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta } } \\ { \displaystyle \qquad - \left[ 1 - L \eta - 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q - 1 } { b } + p ( q - 1 ) \right) \right] \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| v _ { t } \| ^ { 2 } . } \end{array}
$$

For the learning rate (74),

$$
\begin{array} { r } { L \eta = \cfrac { 1 } { 2 ( 1 + \sqrt { q / b } + \sqrt { p ( q - 1 ) } ) } \leq \frac { 1 } { 2 } , } \\ { 2 L ^ { 2 } \eta ^ { 2 } \left( \frac { q - 1 } { b } + p ( q - 1 ) \right) \leq \cfrac { q / b + p ( q - 1 ) } { 2 ( 1 + \sqrt { q / b } + \sqrt { p ( q - 1 ) } ) ^ { 2 } } \leq \frac { 1 } { 2 } . } \end{array}
$$

The last inequality uses $a ^ { 2 } + c ^ { 2 } \leq ( 1 + a + c ) ^ { 2 }$ for nonnegative $a , c .$ . The bracket is therefore nonnegative. Dropping its nonpositive contribution and dividing by $T$ proves

$$
\mathbb { E } \| \nabla f ( x _ { R } ) \| ^ { 2 } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \| \nabla f ( x _ { t } ) \| ^ { 2 } \leq \frac { 2 \Delta _ { f } } { \eta T } ,
$$

which is (35). Refreshes still occur at rounds $0 , q , 2 q , \ldots < T$ , and at most $T$ correction tasks are issued. Their total cost is at most $n [ T / q ] + 2 b T$ , proving (36).

For $q = n , b = 1$ , each of 1, ${ \sqrt { n } } ,$ and $\sqrt { p ( n - 1 ) }$ is at most $\sqrt { n ( 1 + p ) }$ . It follows that

$$
\begin{array} { c } { { \eta ^ { - 1 } = 2 L ( 1 + \sqrt { n } + \sqrt { p ( n - 1 ) } ) \le 6 L \sqrt { n ( 1 + p ) } , } } \\ { { \displaystyle T \le 1 + \frac { 2 \Delta _ { f } } { \eta \epsilon ^ { 2 } } \le 1 + \frac { 1 2 L \Delta _ { f } \sqrt { n ( 1 + p ) } } { \epsilon ^ { 2 } } , } } \\ { { \displaystyle n [ T / n ] + 2 T \le n + 3 T \le n + 3 + \frac { 3 6 L \Delta _ { f } \sqrt { n ( 1 + p ) } } { \epsilon ^ { 2 } } . } } \end{array}
$$

The prescribed $T$ gives $2 \Delta _ { f } / ( \eta T ) \le \epsilon ^ { 2 }$ , and $n \geq 1$ absorbs the constant three into the stated complexity. □

Interpretation. For a fixed p, Theorems 4 and 5 retain their respective variance-reduction orders without a maximum age for a correction. The stochastic result bounds the remaining weight of missing responses, whereas the finite-sum result bounds their count within the current epoch. Exponential weights gradually remove the influence of an old stochastic correction; an exact refresh removes the influence of an old finite-sum correction. Both arguments allow sample-dependent missingness. Appendix A gives a sharper dependence on the outstanding count when each arrival time is fixed before drawing its task’s sample.

## References

Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex SGD. In Advances in Neural Information Processing Systems, 2019.

Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. SPIDER: Near-optimal non-convex optimization via stochastic path-integrated diferential estimator. In Advances in Neural Information Processing Systems, 2018.

Pooria Joulani, András György, and Csaba Szepesvári. Online learning under delayed feedback. In Proceedings of the 30th International Conference on Machine Learning, Proceedings of Machine Learning Research, pages 1453–1461, 2013.

Anastasiia Koloskova, Sebastian U. Stich, and Martin Jaggi. Sharper convergence guarantees for asynchronous SGD for distributed and federated learning. In Advances in Neural Information Processing Systems, 2022.

Lam M. Nguyen, Jie Liu, Katya Scheinberg, and Martin Takáč. SARAH: A novel method for machine learning problems using stochastic recursive gradient. In Proceedings of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research, pages 2613–2621, 2017.

Zhe Wang, Kaiyi Ji, Yi Zhou, Yingbin Liang, and Vahid Tarokh. SpiderBoost and momentum: Faster variance reduction algorithms. In Advances in Neural Information Processing Systems, pages 2403–2413, 2019.

## A Arrival Times Fixed Before Sampling

The analysis in the main text allows a task’s completion time to depend on its sampled gradient. In this appendix, we impose a stronger timing condition and improve the dependence on the number of outstanding corrections. Algorithm 1 is unchanged.

Assumption 5 (Arrival time chosen before the sample). For each task $j ,$ after its point pair has been chosen, the arrival time $A _ { j }$ is fixed before drawing $\xi _ { j }$ . The arrival time may depend on earlier samples and independent scheduling randomness, and may equal infinity. It is not revised later. The sample $\xi _ { j }$ has the prescribed oracle distribution after the point pair and $A _ { j }$ have been fixed.

This condition includes a schedule fixed in advance. It also permits later arrival times to depend on earlier samples. No maximum delay is required.

Theorem 6. Under Assumptions $\begin{array} { r } { 1 , \ 2 , } \end{array}$ and 5, suppose that (66) holds for all historical missing corrections, with a fixed $p \geq 0$ . Given $\epsilon > 0$ , choose

$$
\alpha = \frac { 1 } { 1 6 \operatorname* { m a x } \{ ( 1 + \sigma / \epsilon ) ^ { 2 } , 1 + p \} } ,
$$

$$
\eta = \frac { \sqrt { \alpha } } { 4 L \sqrt { 1 + p } } ,\tag{77}
$$

$$
T = \mathrm { m a x } \left\{ 1 , \left\lceil \frac { 8 \Delta _ { f } } { \eta \epsilon ^ { 2 } } \right\rceil \right\} ,\tag{78}
$$

and choose B by (16). Then Algorithm 1 satisfies

$$
\mathbb { E } \left\| \nabla f ( x _ { R } ) \right\| ^ { 2 } \leq \epsilon ^ { 2 } .
$$

It issues at most $B + 2 T$ evaluations, with complexity

$$
O \left( 1 + \frac { \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { L \Delta _ { f } } { \epsilon ^ { 2 } } \operatorname* { m a x } \left\{ \left( 1 + \frac { \sigma } { \epsilon } \right) \sqrt { 1 + p } , 1 + p \right\} \right) .\tag{79}
$$

For comparison, Corollary 5 contains the factor $( 1 + p ) ( 1 + \sigma / \epsilon )$ . Under Assumption 5, the stochastic term grows as ${ \sqrt { 1 + p } } ,$ , while the deterministic term grows as $1 + p .$

## A.1 Estimation error

Recall the centered innovation $\zeta _ { j }$ in (24). Separate the received innovations from the missing population corrections by defining

$$
a _ { t } = ( 1 - \alpha ) ^ { t } \bigl ( v _ { 0 } - \nabla f ( x _ { 0 } ) \bigr ) + \sum _ { j = 0 } ^ { t - 1 } \mathbf { 1 } _ { \{ A _ { j } \leq t \} } ( 1 - \alpha ) ^ { t - j - 1 } \zeta _ { j } .\tag{80}
$$

The population corrections telescope:

$$
\begin{array} { l } { \displaystyle \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { t - j - 1 } \big ( \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) \big ) } \\ { \displaystyle \qquad = \displaystyle \sum _ { j = 1 } ^ { t } ( 1 - \alpha ) ^ { t - j } \nabla f ( x _ { j } ) - \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { t - j } \nabla f ( x _ { j } ) } \\ { \displaystyle = \nabla f ( x _ { t } ) - ( 1 - \alpha ) ^ { t } \nabla f ( x _ { 0 } ) . } \end{array}
$$

Subtracting this identity from (12) gives

$$
v _ { t } - \nabla f ( x _ { t } ) = a _ { t } - \sum _ { j \in M _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \big ( \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) \big ) .\tag{81}
$$

Lemma 7. Under the assumptions of Theorem $\delta ,$ for any $0 < \alpha < 1$ -2

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + \frac { 4 L ^ { 2 } ( 1 + p ) } { \alpha } \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } } \\ { \displaystyle \qquad + 4 \alpha p \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } + 4 \alpha \sigma ^ { 2 } T . } \end{array}\tag{82}
$$

Proof. For each $j < t$ , the initialization error, $A _ { j }$ , and the point pair of task $j$ are fixed before drawing $\xi _ { j }$ . For $i < j$ , the same holds for $A _ { i }$ and $\zeta _ { i }$ . Averaging the fresh sample $\xi _ { j }$ first gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ \mathbf { 1 } _ { \{ A _ { i } \leq t \} } \mathbf { 1 } _ { \{ A _ { j } \leq t \} } \left. \zeta _ { i } , \zeta _ { j } \right. \right] = 0 \quad ( i < j ) , } \\ & { \mathbb { E } \left[ \mathbf { 1 } _ { \{ A _ { j } \leq t \} } \left. v _ { 0 } - \nabla f ( x _ { 0 } ) , \zeta _ { j } \right. \right] = 0 . } \end{array}
$$

Thus, expanding (80), we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left\| a _ { t } \right\| ^ { 2 } = ( 1 - \alpha ) ^ { 2 t } \mathbb { E } \left\| v _ { 0 } - \nabla f ( x _ { 0 } ) \right\| ^ { 2 } } \\ & { \qquad + \displaystyle \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } \mathbb { E } \left[ \mathbf { 1 } _ { \{ A _ { j } \leq t \} } \| \zeta _ { j } \| ^ { 2 } \right] } \\ & { \qquad + \displaystyle 2 \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { 2 t - j - 1 } \underbrace { \mathbb { E } \left[ \mathbf { 1 } _ { \{ A _ { j } \leq t \} } \left. v _ { 0 } - \nabla f ( x _ { 0 } ) , \zeta _ { j } \right. \right] } _ { = 0 } } \\ & { \qquad + \displaystyle 2 \sum _ { 0 \leq i < j < t } ( 1 - \alpha ) ^ { 2 t - i - j - 2 } \underbrace { \mathbb { E } \left[ \mathbf { 1 } _ { \{ A _ { i } \leq t \} } \mathbf { 1 } _ { \{ A _ { j } \leq t \} } \left. \zeta _ { i } , \zeta _ { j } \right. \right] } _ { = 0 } . } \end{array}
$$

Using (25) and $\mathbf { 1 } _ { \{ A _ { j } \leq t \} } \leq 1$ gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } \left\| a _ { t } \right\| ^ { 2 } \leq ( 1 - \alpha ) ^ { 2 t } \frac { \sigma ^ { 2 } } { B } } \\ { \displaystyle \quad \quad + \sum _ { j = 0 } ^ { t - 1 } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } \left( 2 L ^ { 2 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } \right) , } \end{array}
$$

where $\mathbb { E } \left\| v _ { 0 } - \nabla f ( x _ { 0 } ) \right\| ^ { 2 } \leq \sigma ^ { 2 } / B$ follows from the initialization batch. The two geometric sums satisfy

$$
\sum _ { t = 0 } ^ { T - 1 } ( 1 - \alpha ) ^ { 2 t } \leq \operatorname* { m i n } \{ T , 1 / \alpha \} , \qquad \sum _ { t = j + 1 } ^ { T - 1 } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } \leq \frac { 1 } { \alpha } .
$$

Changing the order of summation now yields

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| a _ { t } \right\| ^ { 2 } \leq \frac { \sigma ^ { 2 } } { B } \operatorname* { m i n } \{ T , 1 / \alpha \} } \\ { \displaystyle \qquad + \sum _ { j = 0 } ^ { T - 2 } \Big ( 2 L ^ { 2 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } + 2 \alpha ^ { 2 } \sigma ^ { 2 } \Big ) \sum _ { t = j + 1 } ^ { T - 1 } ( 1 - \alpha ) ^ { 2 ( t - j - 1 ) } } \\ { \displaystyle \qquad \leq \frac { \sigma ^ { 2 } } { B } \operatorname* { m i n } \{ T , 1 / \alpha \} + \frac { 2 L ^ { 2 } } { \alpha } \sum _ { j = 0 } ^ { T - 2 } \mathbb { E } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } + 2 \alpha \sigma ^ { 2 } ( T - 1 ) . } \end{array}
$$

Extending the nonnegative sums to $T$ terms gives

$$
\sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. a _ { t } \right. ^ { 2 } \leq \frac { \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + \frac { 2 L ^ { 2 } } { \alpha } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left. x _ { t + 1 } - x _ { t } \right. ^ { 2 } + 2 \alpha \sigma ^ { 2 } T .\tag{83}
$$

Next consider the missing population corrections. Their squared norms satisfy

$$
\begin{array} { r l } { \| \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) \| ^ { 2 } = \| \nabla f ( x _ { j + 1 } ) - \nabla f ( x _ { j } ) + \alpha \nabla f ( x _ { j } ) \| ^ { 2 } } & { } \\ { \le 2 L ^ { 2 } \left\| x _ { j + 1 } - x _ { j } \right\| ^ { 2 } + 2 \alpha ^ { 2 } \left\| \nabla f ( x _ { j } ) \right\| ^ { 2 } . } & { } \end{array}
$$

Since $\begin{array} { r } { \sum _ { j \in { \mathcal { M } } _ { t } } ( 1 - \alpha ) ^ { t - j - 1 } \leq | { \mathcal { M } } _ { t } | \leq p } \end{array}$ , weighted Cauchy–Schwarz gives

$$
\begin{array} { r l } {  { \sum _ { i = 0 } ^ { T - 1 } \mathbb { E } \bigg \| \sum _ { \ell \leq M } ( 1 - \alpha ) ^ { \ell - j - 1 } ( \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) ) \bigg \| ^ { 2 } } } \\ & { \leq \frac { \eta - 1 } { \varepsilon = 0 } ( 1 - \alpha ) ^ { \ell - j - 1 } \mathbb { E } \Big [ \mathbb { 1 } _ { \xi \in \mathcal { A } ( 1 ) } \| \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) \| ^ { 2 } \Big ] } \\ & { \leq \frac { T - 2 } { \varepsilon = 0 } \mathbb { E } \| \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) \| ^ { 2 } \underset { \ell = \phi + 1 } { \sum _ { i = 0 } ^ { T - 1 } } ( 1 - \alpha ) ^ { \ell - j - 1 } } \\ & { \leq \frac { \eta - 1 } { \varepsilon = 0 } \mathbb { E } \| \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) \| ^ { 2 } \underset { \ell = \phi + 1 } { \sum _ { i = 0 } ^ { T - 1 } } ( 1 - \alpha ) ^ { \ell - j - 1 } } \\ & { \leq \frac { \eta } { \alpha } \underset { j \geq 0 } { \sum _ { i = 0 } ^ { T - 1 } } \mathbb { E } \| \nabla f ( x _ { j + 1 } ) - ( 1 - \alpha ) \nabla f ( x _ { j } ) \| ^ { 2 } } \\ & { \leq \frac { 2 L ^ { 2 } \rho } { \alpha } \underset { j \leq 0 } { \sum _ { i = 0 } ^ { T - 1 } } \mathbb { E } \| x _ { j + 1 } - x _ { j } \| ^ { 2 } + 2 \alpha \eta \frac { T - 1 } { \varepsilon = 0 } \mathbb { E } \| \nabla f ( x _ { j } ) \| ^ { 2 } . } \end{array}\tag{84}
$$

The second inequality drops the indicator and exchanges the two sums; the third uses $\scriptstyle \sum _ { t = j + 1 } ^ { \infty } ( 1 -$ $\alpha ) ^ { t - j - 1 } = 1 / \alpha$ . Finally, (81) and $\left. a - b \right. ^ { 2 } \leq 2 \left. a \right. ^ { 2 } + 2 \left. b \right. ^ { 2 }$ imply

$$
\begin{array} { l } { \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } - \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \{ T , 1 / \alpha \} + \left( \frac { 4 L ^ { 2 } } { \alpha } + \frac { 4 L ^ { 2 } p } { \alpha } \right) \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } } \\ { \displaystyle \qquad + 4 \alpha p \displaystyle \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } + 4 \alpha \sigma ^ { 2 } T , } \end{array}
$$

which is (82).

## A.2 Proof of Theorem 6

Substitute (82) and $\left\| x _ { t + 1 } - x _ { t } \right\| ^ { 2 } = \eta ^ { 2 } \left\| v _ { t } \right\| ^ { 2 }$ into (6). Moving the gradient and estimator sums to the left gives

$$
\begin{array} { r l } & { \displaystyle \left( 1 - 4 \alpha p \right) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } } \\ & { \quad \quad + \left( 1 - L \eta - \frac { 4 L ^ { 2 } \eta ^ { 2 } ( 1 + p ) } { \alpha } \right) \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| v _ { t } \right\| ^ { 2 } } \\ & { \quad \quad \le \displaystyle \frac { 2 \Delta _ { f } } { \eta } + \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ T , \frac { 1 } { \alpha } \right\} + 4 \alpha \sigma ^ { 2 } T . } \end{array}\tag{85}
$$

By (77), $\alpha \leq 1 / 1 6$ and

$$
\begin{array} { c c c } { { 4 \alpha p = \displaystyle \frac { p } { 4 \operatorname* { m a x } \{ ( 1 + \sigma / \epsilon ) ^ { 2 } , 1 + p \} } \le \displaystyle \frac { p } { 4 ( 1 + p ) } \le \displaystyle \frac { 1 } { 4 } , } } \\ { { L \eta = \displaystyle \frac { \sqrt { \alpha } } { 4 \sqrt { 1 + p } } \le \displaystyle \frac { 1 } { 1 6 } , } } \\ { { \displaystyle \frac { 4 L ^ { 2 } \eta ^ { 2 } ( 1 + p ) } { \alpha } = \displaystyle \frac { 4 L ^ { 2 } ( 1 + p ) } { \alpha } \cdot \displaystyle \frac { \alpha } { 1 6 L ^ { 2 } ( 1 + p ) } = \displaystyle \frac { 1 } { 4 } . } } \end{array}
$$

Hence the two coeficients on the left of (85) satisfy

$$
1 - 4 \alpha p \ge \frac { 3 } { 4 } , \qquad 1 - L \eta - \frac { 4 L ^ { 2 } \eta ^ { 2 } ( 1 + p ) } { \alpha } \ge 1 - \frac { 1 } { 1 6 } - \frac { 1 } { 4 } = \frac { 1 1 } { 1 6 } > 0 .
$$

The definitions of T and B give

$$
\frac { 1 } { T } \leq \frac { \eta \epsilon ^ { 2 } } { 8 \Delta _ { f } } \quad ( \Delta _ { f } > 0 ) , \qquad \frac { \sigma ^ { 2 } } { B } \operatorname* { m i n } \left\{ 1 , \frac { 1 } { \alpha T } \right\} \leq \frac { \epsilon ^ { 2 } } { 1 6 } .
$$

Therefore,

$$
\begin{array} { c } { \displaystyle \frac { 2 \Delta _ { f } } { \eta T } \leq \frac { \epsilon ^ { 2 } } { 4 } , } \\ { \displaystyle \frac { 2 \sigma ^ { 2 } } { B } \operatorname* { m i n } \bigg \{ 1 , \frac { 1 } { \alpha T } \bigg \} \leq \frac { \epsilon ^ { 2 } } { 8 } , } \\ { \displaystyle 4 \alpha \sigma ^ { 2 } \leq \frac { \sigma ^ { 2 } } { 4 ( 1 + \sigma / \epsilon ) ^ { 2 } } \leq \frac { \epsilon ^ { 2 } } { 4 } . } \end{array}
$$

The first bound also holds when $\begin{array} { r } { \Delta _ { f } = 0 . } \end{array}$ , since its left-hand side is zero. Dropping the nonnegative estimator term in (85) gives

$$
\frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left\| \nabla f ( x _ { t } ) \right\| ^ { 2 } \leq \frac { \epsilon ^ { 2 } / 4 + \epsilon ^ { 2 } / 8 + \epsilon ^ { 2 } / 4 } { 3 / 4 } = \frac { 5 } { 6 } \epsilon ^ { 2 } .
$$

For the uniform output,

$$
\mathbb { E } \left. \nabla f ( x _ { R } ) \right. ^ { 2 } = \sum _ { t = 0 } ^ { T - 1 } \frac { 1 } { T } \mathbb { E } \left. \nabla f ( x _ { t } ) \right. ^ { 2 } \leq \frac { 5 } { 6 } \epsilon ^ { 2 } \leq \epsilon ^ { 2 } .
$$

For the evaluation count,

$$
\begin{array} { r } { \frac { 1 } { \eta } = 1 6 L \sqrt { ( 1 + p ) \operatorname* { m a x } \{ ( 1 + \sigma / \epsilon ) ^ { 2 } , 1 + p \} } } \\ { = 1 6 L \operatorname* { m a x } \left\{ \left( 1 + \frac { \sigma } { \epsilon } \right) \sqrt { 1 + p } , 1 + p \right\} . } \end{array}
$$

Since $B \leq 1 + 1 6 \sigma ^ { 2 } / \epsilon ^ { 2 }$ and $T \leq 1 + 8 \Delta _ { f } / ( \eta \epsilon ^ { 2 } )$ 2

$$
\begin{array} { l } { { \displaystyle B + 2 T \le 3 + \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { 1 6 \Delta _ { f } } { \eta \epsilon ^ { 2 } } } } \\ { { \displaystyle \qquad = 3 + \frac { 1 6 \sigma ^ { 2 } } { \epsilon ^ { 2 } } + \frac { 2 5 6 L \Delta _ { f } } { \epsilon ^ { 2 } } \operatorname* { m a x } \left\{ \left( 1 + \frac { \sigma } { \epsilon } \right) \sqrt { 1 + p } , 1 + p \right\} , } } \end{array}
$$

which proves (79).

Why the timing condition is used. Let the stochastic gradient have additive noise $Z _ { j } \in \{ - \sigma , \sigma \}$ 2 with equal probabilities. Then $\zeta _ { j } = \alpha Z _ { j }$ . If $A _ { j } = j + 1$ for $Z _ { j } = - \sigma$ and $A _ { j } = j + 2$ for $Z _ { j } = \sigma$ , we have

$$
\begin{array} { r } { \mathbb { E } \zeta _ { j } = 0 , \qquad \mathbb { E } \left[ \mathbf { 1 } _ { \{ A _ { j } \leq j + 1 \} } \zeta _ { j } \right] = \frac { 1 } { 2 } ( - \alpha \sigma ) + \frac { 1 } { 2 } \cdot 0 = - \frac { \alpha \sigma } { 2 } . } \end{array}
$$

Thus an arrival indicator depending on its own sample need not preserve zero mean. The main-text analysis instead removes these indicators through a deterministic bound on the missing corrections.