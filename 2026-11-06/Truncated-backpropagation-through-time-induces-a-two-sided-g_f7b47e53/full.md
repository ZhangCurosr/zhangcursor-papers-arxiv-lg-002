# Truncated backpropagation through time induces a two-sided gradient turnpike property

Julian D. Schiller, Matthias A. Muller¨

Abstract— Truncated backpropagation through time reduces the computational burden of training recurrent models by restricting gradient computations to shorter temporal contexts. This induces two distinct boundary effects: missing past information perturbs the forward state evolution, while truncating the backward recursion omits future information. This motivates decoupling the interval over which gradient contributions are averaged for the parameter update from the surrounding temporal context over which they are computed. Under exponential decay of the state-transition Jacobians and mild regularity conditions, this leads to a two-sided gradient turnpike property: the error between truncated- and fullcontext gradient contributions decays exponentially with the distance from both context boundaries. We derive bounds for the averaged update gradient, yielding guarantees on gradient alignment and approximate full-context stationarity; under unbiased interval sampling and a Polyak–Łojasiewicz condition, these further imply a bound on the resulting optimality gap. Numerical experiments on the Wiener–Hammerstein benchmark illustrate the role of past and future context and the trade-off between predictive performance and computational cost.

## I. INTRODUCTION

Recurrent models are widely used across various domains, including dynamical systems, time-series forecasting, and sequence learning [1–3]. Their recurrent state enables them to retain information from the past and thereby model temporal dependencies over long horizons. Training such models, however, requires accounting for the dependence of the loss on the recurrent state trajectory. The standard approach is backpropagation through time (BPTT), which computes gradients by unrolling the recurrent dynamics over the entire training sequence [4]. For long sequences, full BPTT can become computationally and memory intensive, motivating truncated backpropagation through time (TBPTT), where gradients are computed using only shorter segments of the training sequence [5–7].

A central consequence of truncation is that gradient computations are affected by the restricted temporal window on which they are performed. On the one hand, the recurrent state at the beginning of such a window generally differs from the state obtained by propagating the model over the complete preceding sequence. This motivates the use of a burn-in phase, in which additional past samples are processed before the loss is evaluated [8–10]. Related approaches in nonlinear system identification reconstruct the initial state of a training subsequence from preceding input– output data [7,11]. On the other hand, the gradient contribution associated with a given time instant generally depends on losses occurring at later times through the backward recursion. Truncating this recursion at a fixed finite right boundary therefore also discards future loss information. Hence, truncated training naturally gives rise to two distinct boundary effects: one caused by missing past information in the forward state evolution and one caused by missing future information in the backward gradient computation.

Existing theoretical analyses have addressed these two effects largely separately. On the backward side, bounds on the bias introduced by truncating long-range gradient contributions in TBPTT are derived under a geometric decay condition and subsequently used to adapt the truncation horizon [12]. Related work obtains unbiased gradient estimators by randomizing the truncation length and appropriately reweighting the resulting gradient contributions [13]. On the forward side, contractive models imply that the influence of sufficiently remote past information decays with temporal distance [14], while our recent work shows how a finite burn-in phase can approximate full-sequence training under less restrictive finite-memory conditions [10]. To the best of our knowledge, however, existing results do not jointly quantify how missing past information in the forward path and missing future information in the backward path affect gradient computations for recurrent models. Importantly, these effects are coupled, as discrepancies in the forward state evolution also perturb the backward sensitivities.

The resulting two-sided structure is reminiscent of turnpike phenomena in finite-horizon optimal control and estimation, where optimal trajectories remain close to distinguished reference trajectories over most of the horizon, with deviations concentrated near the initial and terminal boundaries [15–18]. Such turnpike properties have been established for states, inputs, outputs, and adjoint variables, and corresponding exponential characterizations typically consist of terms that decay with the distance from the horizon boundaries.

In the same spirit, we show that an analogous twosided phenomenon arises directly at the level of gradient contributions in the training of nonlinear recurrent models with truncated temporal context. To capture the two bound ary effects separately, we distinguish an update interval, over which gradient contributions are averaged to obtain the gradient used in the parameter update law, from a surrounding temporal context over which these contributions are computed. Under exponential decay of the state-transition Jacobians and mild regularity conditions, we show that the error between truncated- and full-context gradient contributions decays exponentially with the distance from both truncation boundaries. Consequently, extending the context into the past or future independently tightens the corresponding contribution to the gradient-error bound. This yields explicit bounds for the averaged gradient used in the parameter update law and, in turn, guarantees on gradient alignment and approximate full-context stationarity. Moreover, under unbiased interval sampling, these guarantees extend to the full training objective; combined with a Polyak–Łojasiewicz condition, this yields a bound on the optimality gap relative to the optimal value of the full-context loss. The theoretical findings are illustrated on a standard nonlinear system identification benchmark, highlighting how past and future context affect the accuracy and computational efficiency of truncated learning.

## II. PROBLEM SETTING

This section introduces the learning problem and mathematical setup for the subsequent analysis.

Notation: The set of non-negative integers is denoted by N. Intervals of time indices are understood as integer intervals. The notation ∥ · ∥ denotes the Euclidean norm for vectors and its induced matrix norm. The cardinality of a finite set A is denoted by $| { \cal A } |$

## A. Full-context learning problem

In this work, we consider the problem of learning the parameters $\theta \in \mathbb { R } ^ { o }$ of a general nonlinear discrete-time statespace model in the form of

$$
\begin{array} { r } { x ^ { + } = f ( x , u ; \theta ) , } \\ { y = h ( x , u ; \theta ) , } \end{array}\tag{1}
$$

(2)

where $x \in \mathbb { R } ^ { n }$ is the state of the system, $x ^ { + } \in \mathbb { R } ^ { n }$ is the successor state, $u \in \mathbb { R } ^ { m }$ is the input, and $y \in \mathbb { R } ^ { p }$ is the model output. The dynamics and output equation of the model are characterized by nonlinear functions $f$ and h, both of which are parameterized by θ. This formulation encompasses standard recurrent neural network (RNN) models, including gated architectures such as LSTMs [19] and GRUs [20], as well as more modern sequence models such as Mamba [21].

For training the model parameters $\theta ,$ we consider a given input sequence $( u _ { t } ) _ { t \in \mathcal { T } }$ with time index set $\mathcal { T } : = [ 0 , T - 1 ]$ $T \in \mathbb { N } , T \geq 1$ , and corresponding measured target outputs $( y _ { t } ^ { \mathrm { d } } ) _ { t \in T }$ . For each $t \in \tau$ , let $l _ { t } ( x ; \theta )$ denote the per-timestep loss induced by the model output $h ( x , u _ { t } ; \theta )$ and the corresponding target $y _ { t } ^ { d }$ . We aim to minimize the full training loss

$$
L ( \theta ) : = \frac { 1 } { T } \sum _ { t \in \mathcal { T } } l _ { t } ( x _ { t } ; \theta )\tag{3}
$$

evaluated along the full-context model simulation

$$
x _ { 0 } = 0 , \quad x _ { t + 1 } = f ( x _ { t } , u _ { t } ; \theta ) , \quad t \in \mathcal { T } .\tag{4}
$$

The initial state is fixed at zero for simplicity, as commonly done in the context of black-box system identification using RNNs; however, alternative initializations may also be

considered. The gradient of $L ( \theta )$ can be decomposed as

$$
\nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ) = \frac { 1 } { T } \sum _ { t \in \mathcal { T } } g _ { t } ( \boldsymbol { \theta } ) ,\tag{5}
$$

where $g _ { t } ( \theta )$ is the contribution of the parameter dependence at time t to the full-context gradient, including its influence on subsequent losses through the dynamics. These contributions can be expressed using the full-context backward sensitivity recursion (BPTT) introduced in Section III-A.

Motivated by TBPTT, we introduce a gradient-update interval $\mathcal { U } = [ a , b ] \subseteq \mathcal { T }$ and define the corresponding fullcontext update gradient as

$$
G _ { \mathcal { U } } ( \boldsymbol { \theta } ) : = \frac { 1 } { | \mathcal { U } | } \sum _ { t \in \mathcal { U } } g _ { t } ( \boldsymbol { \theta } ) ;\tag{6}
$$

see Figure 1 for an illustration. The corresponding stochastic gradient descent (SGD) iterations take the form

$$
\theta _ { i + 1 } = \theta _ { i } - \eta _ { i } G _ { \mathcal { U } _ { i } } ( \theta _ { i } ) , \quad i \in \mathbb { N } ,\tag{7}
$$

where U<sub>i</sub> denotes a randomly sampled gradient-update interval and $\eta _ { i } > 0$ is the learning rate; compare, e.g., [22].

## B. Truncating the context

In the considered TBPTT scheme, both the forward model simulation and backward sensitivity recursion are restricted to a shorter temporal context. Consequently, the full-context update gradient used in (7) is replaced by a truncated-context approximation. Hence, we associate with each interval U a truncated gradient-context interval $\mathcal { C } = [ r , s ] \supseteq \mathcal { U }$ such that

$$
\varnothing \neq { \mathcal { U } } \subseteq { \mathcal { C } } \subseteq { \mathcal { T } } .\tag{8}
$$

In the following, we refer to U and C simply as the update and context intervals. Define

$$
T _ { \mathrm { P } } : = a - r , \qquad T _ { \mathrm { F } } : = s - b .\tag{9}
$$

Then, $T _ { \mathrm { P } }$ and $T _ { \mathrm { F } }$ denote the past- and future-context extensions, respectively; see Figure 1. Now, let ${ \hat { g } } _ { t } ( \theta )$ denote the local parameter-gradient contribution associated with time t, computed using an independently initialized forward simulation and a backward sensitivity recursion restricted to the context $\mathcal { C } ;$ compare Figure 1. Averaging the contributions over U yields the truncated-context update gradient

$$
\widehat { G } _ { \mathcal { U } } ( \boldsymbol { \theta } ) : = \frac { 1 } { | \mathcal { U } | } \sum _ { t \in \mathcal { U } } \hat { g } _ { t } ( \boldsymbol { \theta } ) .\tag{10}
$$

At iteration $i \in \mathbb { N } , \widehat { G } _ { \mathcal { U } _ { i } } ( \theta _ { i } )$ replaces $G _ { { \mathcal { U } } _ { i } } ( \theta _ { i } )$ in $( 7 )$

Remark 1 (Mini-batch SGD): Averaging truncated gradients over independently sampled update intervals in the SGD iterations (7) reduces sampling variance and permits parallel evaluation. The subsequent gradient-error bounds extend directly to mini-batch averages via the triangle inequality. Moreover, the truncated gradients can also be used with adaptive optimizers such as Adam [23]; compare also [22].

The following analysis quantifies the discrepancies be tween the full- and truncated-context gradient contributions and their corresponding update gradients in (6) and (10).

![](images/b1f050b999d2cfd323f3909e8bce1ab40a8d9d3ab7da1b4131bfbf4c8f8fea69.jpg)  
Fig. 1: Illustration of the gradient-update interval $\begin{array} { r } { \begin{array} { r l } { \mathcal { U } } & { { } = } & { \left[ a , b \right] } \end{array} } \end{array}$ and the temporal context $\begin{array} { r } { \boldsymbol { \mathcal { C } } = [ r , \boldsymbol { \bar { s } } ] } \end{array}$ . The full-context update gradient $G _ { u }$ uses forward and backward recursions over the entire training sequence (blue), whereas the truncated-context update gradient $\widehat { G } _ { \mathcal { U } }$ uses recursions restricted to C, with past and future context extensions $T _ { \mathrm { P } }$ and $T _ { \mathrm { F } }$ (orange).

## III. THEORETICAL ANALYSIS

Throughout this section, for brevity we consider a fixed input sequence $u = ( u _ { 0 } , \ldots , u _ { T - 1 } )$ and a particular parameterization $\theta ,$ and suppress algorithm iterations i. Consequently, the following assumptions and results are formulated for this fixed setting; they can be made uniform with respect to inputs and parameters if the required assumptions are imposed uniformly over all admissible input sequences and parameters.

We first introduce the required machinery and assumptions and then analyze the gradient error and its implications for learning. We indicate truncated-context quantities by a hat, while leaving their full-context counterparts undecorated.

## A. Preliminary definitions and main assumptions

For the given input sequence u and parameter $\theta ,$ let $( x _ { t } ) _ { t \in \mathcal { T } }$ denote the full-context state sequence satisfying

$$
x _ { t + 1 } = f ( x _ { t } , u _ { t } ; \theta ) , \quad t \in { \mathcal { T } } , \quad x _ { 0 } = 0 .\tag{11}
$$

For a context interval ${ \mathcal { C } } = [ r , s ] \subseteq T$ , consider its truncatedcontext counterpart $( \hat { x } _ { t } ) _ { t \in \mathcal C }$ such that

$$
\hat { x } _ { t + 1 } = f ( \hat { x } _ { t } , u _ { t } ; \theta ) , \quad t \in \mathcal { C } , \quad \hat { x } _ { r } = 0 .\tag{12}
$$

The zero initialization in (12) is adopted for simplicity. This standard choice facilitates simple, numerically robust, and computationally efficient training and allows context intervals to be sampled and evaluated independently of their temporal order [24,25]. Alternatively, zero initialization may be replaced by any ${ \hat { x } } _ { r } \in { \mathcal { X } } ;$ the bounds derived below then depend on the initial-state mismatch $\| \hat { x } _ { r } - x _ { r } \|$ , with the past-context contribution vanishing if ${ \hat { x } } _ { r } = x _ { r }$

Define the Jacobians evaluated along the two trajectories as

$$
\begin{array} { r l } & { A _ { t } : = D _ { x } f ( x _ { t } , u _ { t } ; \theta ) , \quad B _ { t } : = D _ { \theta } f ( x _ { t } , u _ { t } ; \theta ) , t \in \mathcal { T } , } \\ & { \quad \hat { A } _ { t } : = D _ { x } f ( \hat { x } _ { t } , u _ { t } ; \theta ) , \quad \hat { B } _ { t } : = D _ { \theta } f ( \hat { x } _ { t } , u _ { t } ; \theta ) , t \in \mathcal { C } , } \end{array}
$$

with the corresponding transition Jacobians

$$
\Phi ( t , k ) : = A _ { t - 1 } A _ { t - 2 } \cdot \cdot \cdot A _ { k } , \qquad 0 \leq k < t \leq T ,\tag{13}
$$

$$
\hat { \Phi } ( t , k ) : = \hat { A } _ { t - 1 } \hat { A } _ { t - 2 } \cdot \cdot \cdot \hat { A } _ { k } , \qquad r \leq k < t \leq s ,\tag{14}
$$

and $\Phi ( k , k ) = \hat { \Phi } ( k , k ) = I .$

To derive the gradient contributions $g _ { t }$ and $\hat { g } _ { t }$ in (6) and (10), respectively, consider the Lagrangian associated with the learning problem (3)–(4), omitting normalization:

$$
\mathcal { L } = \sum _ { t \in \mathcal { T } } \left( l _ { t } ( x _ { t } ; \theta ) + p _ { t + 1 } ^ { \top } ( f ( x _ { t } , u _ { t } ; \theta ) - x _ { t + 1 } ) \right) .\tag{15}
$$

The adjoint variables (or backward sensitivities) $p _ { t }$ are determined by imposing $\nabla _ { x _ { t } } { \mathcal { L } } = 0$ for all $t \in [ 1 , T ]$ , yielding the full-context backward recursion

$$
p _ { T } = 0 , \quad p _ { t } = \nabla _ { x } l _ { t } ( x _ { t } ; \theta ) + A _ { t } ^ { \top } p _ { t + 1 } , \quad t \in \mathcal T .\tag{16}
$$

Since the system dynamics (11) and adjoint recursion (16) are satisfied, we have $T \nabla _ { \theta } L ( \theta ) = \nabla _ { \theta } \mathcal { L } .$ , yielding

$$
T \nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ) = \sum _ { t \in \mathcal { T } } \left( \nabla _ { \boldsymbol { \theta } } l _ { t } ( x _ { t } ; \boldsymbol { \theta } ) + B _ { t } ^ { \top } p _ { t + 1 } \right) .
$$

Thus, the local gradient contributions $g _ { t }$ in (6) are given by

$$
g _ { t } : = \nabla _ { \theta } l _ { t } ( x _ { t } ; \theta ) + B _ { t } ^ { \top } p _ { t + 1 } , \quad t \in \mathcal T .\tag{17}
$$

The corresponding truncated-context expressions follow by constructing the analogous Lagrangian on $\mathcal { C } = [ r , s ]$ , yielding

$$
\begin{array} { r } { \hat { p } _ { s + 1 } = 0 , \quad \hat { p } _ { t } = \nabla _ { x } l _ { t } ( \hat { x } _ { t } ; \theta ) + \hat { A } _ { t } ^ { \top } \hat { p } _ { t + 1 } , \quad t \in \mathcal { C } } \end{array}\tag{18}
$$

and

$$
\hat { g } _ { t } : = \nabla _ { \theta } l _ { t } ( \hat { x } _ { t } ; \theta ) + \hat { B } _ { t } ^ { \top } \hat { p } _ { t + 1 } , \quad t \in \mathcal { U } .\tag{19}
$$

Note that the backward recursion (18) incorporates losses throughout the context C, including the $T _ { \mathrm { { F } } }$ future losses $l _ { b + 1 } , . . . , l _ { s }$ , whereas only the contributions $\hat { g } _ { t }$ for $t \in \mathcal { U }$ are averaged to form the gradient $\widehat { G } _ { \mathcal { U } }$ in (10); compare Figure 1.

We now impose the main assumptions on the setup.

Assumption 1 (Compact state domain): There exists $\mathcal { X } \subset$ $\mathbb { R } ^ { n }$ , convex and compact, such that $x _ { t } \in \mathcal X$ for all $t \in \tau$ and $\hat { x } _ { t } \in \mathcal X$ for all $t \in { \mathcal { C } }$

Assumption 2 (Decaying state-transition Jacobians):

There exist $C \geq 1$ and $\rho \in ( 0 , 1 )$ such that

$$
\left\| D _ { x } \phi _ { t , k } ( x ) \right\| \leq C \rho ^ { t - k }\tag{20}
$$

for all $0 \leq k \leq t \leq T$ and all $x \in \mathcal { X }$ , where $\phi _ { t , k } ( x )$ denotes the (t − k)-step flow of system (1) initialized at x and evolving under the parameter θ given the corresponding input sequence $( u _ { k } , \ldots , u _ { t - 1 } )$

Assumption 2 requires exponential decay of the statetransition Jacobians evaluated along a model trajectory. Note that this property allows individual one-step Jacobians to be expansive. Specifically, it implies exponential forgetting of initial-state differences under a common input sequence, which is relevant for recurrent models whose dependence on remote past information decays over time. Related properties are standard in the literature; compare, e.g., [1,7,10].

Assumption 3 (Bounded and Lipschitz derivatives): There exist $\bar { l } , \bar { B } > 0$ such that $\| \nabla _ { x } l _ { t } ( x ; \theta ) \| \le \bar { l }$ for all $t \in \tau$ and $\| D _ { \theta } f ( x , u _ { t } ; \theta ) \| \le \bar { B }$ for all $t \in \mathcal { U }$ , for all $x \in \mathcal { X }$ . Moreover, there exist $L _ { A } , L _ { B } , L _ { x } , L _ { \theta } > 0$ such that

$$
\| D _ { x } f ( x , u _ { t } ; \theta ) - D _ { x } f ( \tilde { x } , u _ { t } ; \theta ) \| \leq L _ { A } \| x - \tilde { x } \| , \quad t \in \mathcal { C }
$$

$$
\begin{array} { r } { \| D _ { \theta } f ( x , u _ { t } ; \theta ) - D _ { \theta } f ( \tilde { x } , u _ { t } ; \theta ) \| \leq L _ { B } \| x - \tilde { x } \| , t \in \mathcal { U } , } \end{array}
$$

$$
\| \nabla _ { x } l _ { t } ( x ; \theta ) - \nabla _ { x } l _ { t } ( \tilde { x } ; \theta ) \| \leq L _ { x } \| x - \tilde { x } \| , \quad t \in \mathcal { C } ,
$$

$$
\| \nabla _ { \theta } l _ { t } ( x ; \theta ) - \nabla _ { \theta } l _ { t } ( \tilde { x } ; \theta ) \| \leq L _ { \theta } \| x - \tilde { x } \| , \quad t \in \mathcal { U }
$$

for all $x , \tilde { x } \in \mathcal { X } .$

Assumption 3 involves standard regularity requirements. They follow, e.g., when the model dynamics and loss functions are sufficiently smooth on X, with bounded input and target sequences.

## B. Gradient turnpike

We first show that our setup introduced in the previous section implies uniform boundedness of the backward sensitivities introduced in (16) and (18).

Lemma 1: Consider an arbitrary context interval ${ \mathcal { C } } \subseteq { \mathcal { T } }$ Let Assumptions 2 and 3 hold. Then, the backward sensitivities $p _ { t }$ and $\hat { p } _ { t }$ are uniformly bounded on their corresponding time intervals. Specifically, there exists $\bar { p } > 0$ such that

$$
\| p _ { t } \| \leq \bar { p } , \ t \in \mathcal T , \qquad \| \hat { p } _ { t } \| \leq \bar { p } , \ t \in \mathcal C .\tag{21}
$$

Moreover, these bounds extend to $t = T$ and $t = s + 1$ respectively.

Proof: We start by unrolling the recursion in (16) over $[ t , T - 1 ]$ for some $t \in \tau$ . Since $p _ { T } = 0$ , this yields

$$
p _ { t } = \sum _ { k = t } ^ { T - 1 } \Phi ( k , t ) ^ { \top } \nabla _ { x } l _ { k } ( x _ { k } ; \theta ) , \quad t \in \mathcal { T } ,
$$

where Φ is from (13). Note that $\Phi ( k , t ) = D _ { x } \phi _ { k , t } ( x _ { t } )$ for all $t , k \in \tau$ with $k \geq t .$ Thus, from the triangle inequality, Assumptions 2 and 3, and the geometric series, we obtain that

$$
\left\| p _ { t } \right\| \leq \sum _ { k = t } ^ { T - 1 } C \bar { l } \rho ^ { k - t } \leq \frac { C \bar { l } } { 1 - \rho } , \quad t \in \mathcal { T } .\tag{22}
$$

Thus, letting $\bar { p } : = C \bar { l } / ( 1 - \rho )$ establishes the first property in (21); the second one directly follows from similar arguments applied to (18).

Finally, the terminal conditions $p _ { T } = 0$ and $\hat { p } _ { s + 1 } = 0$ in (16) and (18) imply that the bounds in (21) also hold for $t = T$ and $t = s + 1$ , respectively, completing the proof. The following result establishes exponential decay of the gradient discrepancy away from both context boundaries.

Theorem 1 (Pointwise gradient turnpike): Suppose that Assumptions 1–3 hold for all admissible pairs of update and context intervals $( \mathcal { U } , \mathcal { C } )$ , with constants independent of the interval choices. Then, there exist constants $K _ { \mathrm { P } } , K _ { \mathrm { F } } > 0$ independent of U and C, such that

$$
\| \hat { g } _ { t } ( \theta ) - g _ { t } ( \theta ) \| \le K _ { \mathrm { P } } \rho ^ { t - r } + K _ { \mathrm { F } } \rho ^ { s - t } , \quad t \in \mathcal { U } ,\tag{23}
$$

with $\rho \in ( 0 , 1 )$ from Assumption 2.

Proof: Consider arbitrary context and update intervals C and U satisfying (8). Define the forward state error $e _ { t } ^ { x } : =$ ${ \hat { x } } _ { t } - x _ { t } , t \in { \mathcal { C } }$ , where we recall that $x _ { t }$ and $\hat { x } _ { t }$ satisfy (11) and (12), respectively. Define $x ( \tau ) : = \hat { x } _ { r } + \tau ( x _ { r } - \hat { x } _ { r } )$ $\tau \in [ 0 , 1 ]$ . Since $\boldsymbol { x } _ { r } , \boldsymbol { \hat { x } _ { r } } \in \mathcal { X }$ by Assumption 1 with X being convex, it follows that $x ( \tau ) \in \mathcal { X }$ for all $\tau \in [ 0 , 1 ]$ . Using the Fundamental Theorem of Calculus, we obtain

$$
\begin{array} { r l r } {  { e _ { t } ^ { x } : = \hat { x } _ { t } - x _ { t } = \phi _ { t , r } ( x ( 0 ) ) - \phi _ { t , r } ( x ( 1 ) ) } } \\ & { } & { = - \int _ { 0 } ^ { 1 } D _ { x } \phi _ { t , r } ( x ( \tau ) ) x _ { r } d \tau . } \end{array}
$$

Taking norms, using the triangle inequality, and applying (20) then leads to

$$
\| e _ { t } ^ { x } \| \leq \int _ { 0 } ^ { 1 } \| D _ { x } \phi _ { t , r } ( x ( \tau ) ) \| \| \hat { x } _ { r } - x _ { r } \| d \tau \leq C \bar { x } \rho ^ { t - r }\tag{24}
$$

for all $t \in { \mathcal { C } } .$ , using (12) and ${ \bar { x } } : = \operatorname* { m a x } _ { x \in { \mathcal { X } } } \| x \| < \infty$

Define the backward sensitivity error $e _ { t } ^ { p } = \hat { p } _ { t } - p _ { t } , t \in$ $[ r , s + 1 ]$ . By using (16) and (18), we obtain, for each $t \in { \mathcal { C } } ,$

$$
\begin{array} { r l } & { e _ { t } ^ { p } = \nabla _ { x } l _ { t } ( \hat { x } _ { t } ; \theta ) + \hat { A } _ { t } ^ { \top } \hat { p } _ { t + 1 } - \nabla _ { x } l _ { t } ( x _ { t } ; \theta ) - A _ { t } ^ { \top } p _ { t + 1 } } \\ & { \quad = A _ { t } ^ { \top } e _ { t + 1 } ^ { p } + \xi _ { t } , \quad \quad \quad ( } \end{array}\tag{25}
$$

with

$$
\xi _ { t } : = \nabla _ { x } l _ { t } \big ( \hat { x } _ { t } ; \theta \big ) - \nabla _ { x } l _ { t } \big ( x _ { t } ; \theta \big ) + \big ( \hat { A } _ { t } - A _ { t } \big ) ^ { \top } \hat { p } _ { t + 1 } .\tag{26}
$$

Taking norms, using the triangle inequality, Lipschitzcontinuity of $\nabla _ { x } l _ { t }$ and $D _ { x } f$ from Assumption 3, uniform boundedness of $\| \hat { p } _ { t } \| \leq \bar { p }$ from Lemma 1, and the definition of the errors $e _ { t } ^ { x }$ and $e _ { t } ^ { p }$ yields, for each $t \in { \mathcal { C } }$

$$
\begin{array} { r l } & { \| \xi _ { t } \| \leq \| \nabla _ { x } l _ { t } ( \hat { x } _ { t } ; \theta ) - \nabla _ { x } l _ { t } ( x _ { t } ; \theta ) \| + \| \hat { A } _ { t } - A _ { t } \| \| \hat { p } _ { t + 1 } \| } \\ & { \qquad \leq L _ { x } \| e _ { t } ^ { x } \| + L _ { A } \bar { p } \| e _ { t } ^ { x } \| . } \end{array}
$$

Hence, using (24) and defining $C _ { \xi } : = C \bar { x } ( L _ { x } + L _ { A } \bar { p } )$ leads to

$$
\begin{array} { r } { \| \xi _ { t } \| \leq C _ { \xi } \rho ^ { t - r } , \quad t \in \mathcal C . } \end{array}\tag{27}
$$

Unrolling (25) to the context boundary s and using (13) and $\hat { p } _ { s + 1 } = 0$ from (18) yields

$$
e _ { t } ^ { p } = - \Phi ( s + 1 , t ) ^ { \top } p _ { s + 1 } + \sum _ { k = t } ^ { s } \Phi ( k , t ) ^ { \top } \xi _ { k } , \quad t \in \mathcal { C } .\tag{28}
$$

Here, by using submultiplicativity of the norm, Assumption 2, and Lemma 1, the first term on the right-hand side satisfies

$$
\begin{array} { r } { \| \Phi ( s + 1 , t ) ^ { \top } p _ { s + 1 } \| \leq C \bar { p } \rho ^ { s + 1 - t } . } \end{array}\tag{29}
$$

The second term in (28) satisfies, by the same arguments together with application of (27) and the geometric series,

$$
\begin{array} { r l } & { \left\| \displaystyle \sum _ { k = t } ^ { s } \Phi ( k , t ) ^ { \top } \xi _ { k } \right\| \le \displaystyle \sum _ { k = t } ^ { s } \| \Phi ( k , t ) \| \| \xi _ { k } \| } \\ & { \qquad \le C C _ { \xi } \rho ^ { t - r } \displaystyle \sum _ { k = t } ^ { s } \rho ^ { 2 ( k - t ) } \le \displaystyle \frac { C C _ { \xi } } { 1 - \rho ^ { 2 } } \rho ^ { t - r } . } \end{array}\tag{30}
$$

Then, from (28) with (29) and (30), we can infer that

$$
\Vert e _ { t } ^ { p } \Vert \leq { \frac { C C _ { \xi } } { 1 - \rho ^ { 2 } } } \rho ^ { t - r } + C { \bar { p } } \rho ^ { s + 1 - t } , \quad t \in \mathcal { C } .\tag{31}
$$

We claim that the bound in (31) also holds for $t = s + 1 \notin { \mathcal { C } }$ as well. This is required below to cover the special case where $b = s ,$ that is, when the right boundaries of U and C coincide. Indeed, since $\hat { p } _ { s + 1 } = 0$ by (18) and $\| p _ { s + 1 } \| \le \bar { p }$ by Lemma 1, it follows that $\| e _ { s + 1 } ^ { p } \| = \| \hat { p } _ { s + 1 } - p _ { s + 1 } \| \le \bar { p } .$ Hence, the right-hand side of (31) evaluated at $t = s + 1$ provides a trivial upper bound for $\| e _ { s + 1 } ^ { p } \|$ , since $C \geq 1$ . Let $\mathbf { \dot { \sigma } } _ { K _ { \mathrm { P } } ^ { p } } : = C C _ { \xi } / ( 1 - \mathbf { \dot { \rho } } ^ { 2 } )$ and $K _ { \mathrm { F } } ^ { p } : = { \check { C } } \bar { p }$ . Hence, we obtain

$$
\begin{array} { r } { \| e _ { t } ^ { p } \| \le K _ { \mathrm { P } } ^ { p } \rho ^ { t - r } + K _ { \mathrm { F } } ^ { p } \rho ^ { s + 1 - t } , \quad t \in [ r , s + 1 ] . } \end{array}\tag{32}
$$

Now, consider the difference of the gradient contributions in (17) and (19) for $t \in { \mathcal { U } } \colon$

$$
\begin{array} { r } { \hat { g } _ { t } - g _ { t } = \nabla _ { \theta } l _ { t } ( \hat { x } _ { t } ; \theta ) + \hat { B } _ { t } ^ { \top } \hat { p } _ { t + 1 } - \nabla _ { \theta } l _ { t } ( x _ { t } ; \theta ) - B _ { t } ^ { \top } p _ { t + 1 } . } \end{array}
$$

Adding $0 ~ = ~ B _ { t } ^ { \top } \hat { p } _ { t + 1 } - B _ { t } ^ { \top } \hat { p } _ { t + 1 }$ to the right-hand side, rearranging, and substituting the backward error $e _ { t } ^ { p }$ leads to

$$
\begin{array} { r l } & { \hat { g } _ { t } - g _ { t } = \nabla _ { \theta } l _ { t } ( \hat { x } _ { t } ; \theta ) - \nabla _ { \theta } l _ { t } ( x _ { t } ; \theta ) } \\ & { \qquad + B _ { t } ^ { \top } e _ { t + 1 } ^ { p } + ( \hat { B } _ { t } - B _ { t } ) ^ { \top } \hat { p } _ { t + 1 } . } \end{array}
$$

Since $x _ { t } , { \hat { x } } _ { t } \in { \mathcal { X } }$ by Assumption 1, after taking norms and leveraging the triangle inequality, we can apply Assumption 3 and Lemma 1 to further obtain that

$$
\begin{array} { r l } & { \| \hat { g } _ { t } - g _ { t } \| \leq \| \nabla _ { \theta } l _ { t } ( \hat { x } _ { t } ; \theta ) - \nabla _ { \theta } l _ { t } ( x _ { t } ; \theta ) \| } \\ & { \qquad + \| B _ { t } \| \| e _ { t + 1 } ^ { p } \| + \| \hat { B } _ { t } - B _ { t } \| \| \hat { p } _ { t + 1 } \| } \\ & { \qquad \leq ( L _ { \theta } + L _ { B } \bar { p } ) \| e _ { t } ^ { x } \| + \bar { B } \| e _ { t + 1 } ^ { p } \| } \end{array}\tag{33}
$$

for all $t \in \mathcal { U }$ . Applying (24) and (32) to (33) then leads to

$$
\begin{array} { r } { \| \hat { g } _ { t } - g _ { t } \| \le \big ( ( L _ { \theta } + L _ { B } \bar { p } ) C \bar { x } + \bar { B } K _ { \mathrm { P } } ^ { p } \rho \big ) \rho ^ { t - r } + \bar { B } K _ { \mathrm { F } } ^ { p } \rho ^ { s - t } . } \end{array}
$$

Thus, letting $K _ { \mathrm { P } } : = ( L _ { \theta } + L _ { B } \bar { p } ) C \bar { x } + \bar { B } K _ { \mathrm { P } } ^ { p } \rho$ and $K _ { \mathrm { F } } =$ $\bar { B } K _ { \mathrm { F } } ^ { p }$ establishes (23). Noting that these constants are uniform over all admissible C and U completes the proof.

Theorem 1 establishes a two-sided gradient turnpike: the error between the truncated and full-context gradient contributions decays exponentially with the distance from either boundary of C. The past-boundary term results from the initialization of the truncated forward dynamics, whereas the future-boundary term reflects the truncation of the backward adjoint recursion. Consequently, the truncated-context gradient contributions closely approximate their full-context counterparts away from both boundaries.

The following two corollaries translate the gradient turnpike property established in Theorem 1 into direct design implications.

Corollary 1 (Gradient-contribution error): Suppose that the conditions of Theorem 1 hold. Then, the gradient contributions in (23) satisfy

$$
\operatorname* { m a x } _ { t \in \mathcal { U } } \| \hat { g } _ { t } ( \theta ) - g _ { t } ( \theta ) \| \leq K _ { \mathrm { P } } \rho ^ { T _ { \mathrm { P } } } + K _ { \mathrm { F } } \rho ^ { T _ { \mathrm { F } } } .
$$

Proof: Recall $( 9 ) ; t \in { \mathcal { U } }$ implies that $\begin{array} { r } { t - r \geq a - r = T _ { \mathrm { P } } } \end{array}$ and $s - t \geq s - b = T _ { \mathrm { F } }$ . Thus, (23) directly yields

$$
\begin{array} { r } { \| \hat { g } _ { t } - g _ { t } \| \leq K _ { \mathrm { P } } \rho ^ { T _ { \mathrm { P } } } + K _ { \mathrm { F } } \rho ^ { T _ { \mathrm { F } } } , \quad t \in \mathcal { U } . } \end{array}
$$

Taking the maximum over $t \in \mathcal { U }$ completes the proof.

Corollary 2 (Update gradient error): Suppose the conditions of Theorem 1 hold. Then, the full- and truncatedcontext update gradients in (6) and (10) satisfy

$$
\lVert \widehat { G } _ { \mathcal { U } } ( \theta ) - G _ { \mathcal { U } } ( \theta ) \rVert \leq \epsilon _ { G } ( T _ { \mathrm { P } } , T _ { \mathrm { F } } , \lvert \mathcal { U } \rvert ) ,\tag{34}
$$

where

$$
\epsilon _ { G } ( T _ { \mathrm { P } } , T _ { \mathrm { F } } , | \mathcal { U } | ) : = \frac { 1 } { 1 - \rho } \cdot \frac { K _ { \mathrm { P } } \rho ^ { T _ { \mathrm { P } } } + K _ { \mathrm { F } } \rho ^ { T _ { \mathrm { F } } } } { | \mathcal { U } | } .\tag{35}
$$

Proof: Under the conditions of Theorem 1, the gradient turnpike property in (23) applies. Thus, the update gradients in (6) and (10) satisfy

$$
\begin{array} { r l r } {  { \| \widehat { G } _ { \mathcal { U } } ( \theta ) - G _ { \mathcal { U } } ( \theta ) \| \le \frac { 1 } { | \mathcal { U } | } \sum _ { t \in \mathcal { U } } \| \widehat { g } _ { t } ( \theta ) - g _ { t } ( \theta ) \| } } \\ & { } & { \le \frac { 1 } { | \mathcal { U } | } \sum _ { t \in \mathcal { U } } ( K _ { \mathrm { P } } \rho ^ { t - r } + K _ { \mathrm { F } } \rho ^ { s - t } ) . } \end{array}
$$

The application of (9) yields

$$
\lVert \widehat { G } _ { \mathcal { U } } ( \boldsymbol { \theta } ) - G _ { \mathcal { U } } ( \boldsymbol { \theta } ) \rVert \leq \frac { K _ { \mathrm { P } } } { | \mathcal { U } | } \rho ^ { T _ { \mathrm { P } } } \sum _ { t \in \mathcal { U } } \rho ^ { t - a } + \frac { K _ { \mathrm { F } } } { | \mathcal { U } | } \rho ^ { T _ { \mathrm { F } } } \sum _ { t \in \mathcal { U } } \rho ^ { b - t } .
$$

Recall that $\mathcal { U } = [ a , b ]$ . Thus, using the geometric series, we obtain that

$$
\| \widehat { G } _ { \mathcal { U } } ( \theta ) - G _ { \mathcal { U } } ( \theta ) \| \le \frac { K _ { \mathrm { P } } } { | \mathcal { U } | } \rho ^ { T _ { \mathrm { P } } } \frac { 1 } { 1 - \rho } + \frac { K _ { \mathrm { F } } } { | \mathcal { U } | } \rho ^ { T _ { \mathrm { F } } } \frac { 1 } { 1 - \rho } ,
$$

which establishes (34) under (35), completing the proof.

Corollaries 1 and 2 directly characterize the dependence of the gradient-error bounds on the past- and future-context extensions $T _ { \mathrm { P } }$ and $T _ { \mathrm { F } }$ . Increasing either context length exponentially reduces the corresponding error contribution. For fixed context extensions, increasing the update-interval length further reduces the error bound for the averaged update gradient: the boundary-layer contributions are diluted through averaging, yielding a decay of order $\mathcal { O } ( 1 / | \mathcal { U } | )$ . Thus, $T _ { \mathrm { P } } , \ : T _ { \mathrm { F } }$ , and |U| provide complementary design parameters for balancing gradient accuracy against computational effort.

## C. Implications for learning

In this section, we investigate the implications of gradientturnpike behavior for learning. Based on the magnitude of the truncated-context gradient ${ \widehat { G } } _ { { \mathcal { U } } } .$ , we distinguish between regimes away from and close to stationarity. The following proposition relates these two regimes to the corresponding full-context gradient. Under an additional assumption on the sampling of the update intervals in (7), the subsequent corollary relates the expected truncated-context update gradient to the gradient of the full loss and, under a Polyak–Łojasiewicz condition, provides a bound on the optimality gap.

Proposition 1 (Gradient alignment and stationarity): For an arbitrary update interval $\mathcal { U } \subseteq \mathcal { T }$ , suppose that

$$
\| \widehat { G } _ { \mathcal { U } } ( \theta ) - G _ { \mathcal { U } } ( \theta ) \| \le \epsilon _ { G }\tag{36}
$$

for some $\epsilon _ { G } > 0 .$ . Then, the following statements hold:

1) If $\| \widehat { G } _ { \mathcal { U } } ( \theta ) \| > \epsilon _ { G } .$ , then

$$
0 \le \angle \bigl ( \widehat { G } _ { \mathcal { U } } ( \theta ) , G _ { \mathcal { U } } ( \theta ) \bigr ) \le \arcsin \left( \frac { \epsilon _ { G } } { \| \widehat { G } _ { \mathcal { U } } ( \theta ) \| } \right) < \frac { \pi } { 2 } .
$$

In particular, $\widehat { G } _ { \mathcal { U } } ( \boldsymbol { \theta } )$ and $G _ { \mathcal { U } } ( \boldsymbol { \theta } )$ are positively aligned. 2) If $\| \widehat { G } _ { \mathcal { U } } ( \theta ) \| \leq \delta$ for some $\delta \geq 0$ , then

$$
\| G _ { \mathcal { U } } ( \theta ) \| \le \delta + \epsilon _ { G } .
$$

In particular, truncated-context stationarity $( \delta ~ = ~ 0 )$ implies full-context ϵ<sub>G</sub>-stationarity.

Proof: In the following, we omit the arguments θ and U of $G _ { M } ( \theta )$ and ${ \widehat { G } } _ { \mathcal { U } } ( \theta )$ for brevity.

First, suppose that $\| \widehat { G } \| > \epsilon _ { G }$ , where $\epsilon _ { G } > 0$ corresponds to the bound in condition (36). Then,

$$
\begin{array} { r } { \widehat { G } ^ { \top } G = \| \widehat { G } \| ^ { 2 } + \widehat { G } ^ { \top } ( G - \widehat { G } ) \geq \| \widehat { G } \| ^ { 2 } - \| \widehat { G } \| \| G - \widehat { G } \| } \\ { \geq \| \widehat { G } \| \big ( \| \widehat { G } \| - \epsilon _ { G } \big ) > 0 . \qquad ( 3 } \end{array}\tag{7}
$$

Since $\widehat { G } ^ { \top } G = \| \widehat { G } \| \| G \| \cos \angle \left( \widehat { G } , G \right)$ , the strict inequality in (37) yields cos $\angle ( { \widehat { G } } , G ) > 0$ , which further implies that

$$
0 \leq \angle ( { \widehat { G } } , G ) < \pi / 2 ,\tag{38}
$$

i.e., $\widehat { G }$ and G are positively aligned.

To tighten the angle bound in (38), consider the unit vector $v : = G / \| G \|$ , noting that $\| G \| \neq 0$ due to (37). Let $P _ { \bot } : =$ $I - v v ^ { \top }$ denote the orthogonal projector onto the subspace perpendicular to v. Since $P _ { \perp } G = 0 , P _ { \perp } \widehat { G } = - P _ { \perp } ( G - \widehat { G } )$ Taking norms, using that an orthogonal projection does not increase the norm, and applying (36) yields

$$
\| P _ { \perp } \widehat { G } \| = \| P _ { \perp } ( G - \widehat { G } ) \| \leq \| G - \widehat { G } \| \leq \epsilon _ { G } .\tag{39}
$$

Moreover, sin $\angle ( \widehat { G } , G ) = \sin \angle ( \widehat { G } , v ) = \| P _ { \bot } \widehat { G } \| / \| \widehat { G } \|$ , so that (39) leads to

$$
\sin \angle ( { \widehat { G } } , G ) \leq \epsilon _ { G } / \| { \widehat { G } } \| .\tag{40}
$$

Due to $\| \widehat { G } \| > \epsilon _ { G } .$ , we have $0 < \epsilon _ { G } / \| \widehat { G } \| < 1$ . Together with (38) and the strict monotonicity of sin on $[ 0 , \pi / 2 ]$ , this implies $\begin{array} { r } { \angle ( \widehat { G } , G ) \leq \arcsin ( \epsilon _ { G } / \| \widehat { G } \| ) } \end{array}$ ), which establishes the first statement.

For the second statement, suppose that $\| { \widehat { G } } \| \leq \delta$ for some $\delta \geq 0$ . The triangle inequality and (36) directly give

$$
\lVert G \rVert \leq \lVert \widehat { G } \rVert + \lVert G - \widehat { G } \rVert \leq \delta + \epsilon _ { G } ,
$$

which completes the proof.

The proposition distinguishes two regimes relative to the truncation-error bound $\epsilon _ { G } . \mathrm { ~ I f ~ } \| \widehat { G } _ { \mathcal { U } } ( \theta ) \| > \epsilon _ { G }$ , the truncatedcontext gradient is positively aligned with the corresponding full-context gradient, with an explicit bound on their angular deviation. If, on the other hand, $\begin{array} { r } { \| \widehat { G } _ { \mathcal { U } } ( \theta ) \| \le \delta , } \end{array}$ , then the full-context gradient satisfies $\| G _ { \mathcal { U } } ( \theta ) \| \le \delta + \epsilon _ { G } .$ , implying approximate full-context stationarity for sufficiently small δ.

Whenever a certified truncation-error bound $\epsilon _ { G }$ is available, for example from Corollary 2, the applicable regime can be determined during optimization from $\Vert \widehat { G } _ { \mathcal { U } } \Vert$ , which is already computed as part of the gradient update.

The preceding results hold for arbitrary update intervals, without assumptions on their sampling in the SGD iterations (7). To translate the guarantees of Proposition 1 to the full training loss, we now additionally assume that the sampling scheme is unbiased, i.e., such that

$$
\mathbb { E } _ { \mathcal { U } } \big [ G _ { \mathcal { U } } ( \boldsymbol { \theta } ) \big ] = \nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ) .\tag{41}
$$

Note that this is a common requirement in stochastic-gradient methods; see, e.g., [22, Sec. 4]. Condition (41) is satisfied

whenever each time index receives the same expected nor malized sampling weight, i.e.,

$$
\mathbb { E } _ { \mathcal { U } } \left[ \frac { \mathbb { 1 } _ { \{ t \in \mathcal { U } \} } } { | \mathcal { U } | } \right] = \frac { 1 } { T } , \qquad t \in \mathcal { T } ,
$$

where $\mathbb { 1 } _ { \{ t \in \mathcal { U } \} }$ denotes the indicator function of the event $t \in \mathcal { U }$ . This holds, for example, when equal-length blocks forming a partition of $\tau$ are sampled uniformly.

Corollary 3 (Expected gradients): Let the sampling condition (41) hold, with each sampled update interval satisfying condition (36) for a uniform $\epsilon _ { G } > 0$ . Then,

$$
\begin{array} { r } { \| \mathbb { E } _ { \mathcal { U } } [ \widehat { G } _ { \mathcal { U } } ( \theta ) ] - \nabla _ { \theta } L ( \theta ) \| \le \epsilon _ { G } . } \end{array}\tag{42}
$$

Consequently, the statements of Proposition 1 hold with $\widehat { G } _ { \mathcal { U } } ( \boldsymbol { \theta } )$ and $G _ { \mathcal { U } } ( \boldsymbol { \theta } )$ replaced by $\mathbb { E } _ { \mathcal { U } } [ \widehat { G } _ { \mathcal { U } } ( \boldsymbol { \theta } ) ]$ and $\nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } )$ respectively.

If, additionally, $V ^ { * } : = \operatorname* { i n f } _ { \theta } L ( \theta ) > - \infty$ and L satisfies the Polyak–Łojasiewicz condition

$$
\frac { 1 } { 2 } \| \nabla _ { \theta } L ( \theta ) \| ^ { 2 } \geq \mu ( L ( \theta ) - V ^ { * } )\tag{43}
$$

for some $\mu > 0$ , then

$$
L ( \theta ) - V ^ { * } \leq \frac { ( \delta + \epsilon _ { G } ) ^ { 2 } } { 2 \mu }\tag{44}
$$

for any $\delta \geq 0$ such that $\lVert \mathbb { E } _ { \mathcal { U } } [ \widehat { G } _ { \mathcal { U } } ( \theta ) ] \rVert \leq \delta .$

Proof: Leveraging condition (41), linearity of expectation, Jensen’s inequality, and (36), we obtain

$$
\begin{array} { r l } & { \| \mathbb { E } _ { \boldsymbol { \mathcal { U } } } [ \widehat { G } _ { \boldsymbol { \mathcal { U } } } ( \boldsymbol { \theta } ) ] - \nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ) \| = \| \mathbb { E } _ { \boldsymbol { \mathcal { U } } } [ \widehat { G } _ { \boldsymbol { \mathcal { U } } } ( \boldsymbol { \theta } ) - G _ { \boldsymbol { \mathcal { U } } } ( \boldsymbol { \theta } ) ] \| } \\ & { \qquad \le \mathbb { E } _ { \boldsymbol { \mathcal { U } } } [ \| \widehat { G } _ { \boldsymbol { \mathcal { U } } } ( \boldsymbol { \theta } ) - G _ { \boldsymbol { \mathcal { U } } } ( \boldsymbol { \theta } ) \| ] \le \epsilon _ { G } , } \end{array}
$$

which establishes (42). Thus, the arguments used in the proof of Proposition 1 directly apply with $\widehat { G } _ { \mathcal { U } }$ and $G _ { u }$ replaced by $\mathbb { E } _ { \mathcal { U } } [ \widehat { G } _ { \mathcal { U } } ]$ and $\nabla _ { \boldsymbol { \theta } } L .$ , respectively.

Now let $\| \mathbb { E } _ { \boldsymbol { \mathcal { U } } } [ \widehat { G } _ { \boldsymbol { \mathcal { U } } } ( \boldsymbol { \theta } ) ] \| \ \leq \ \delta$ for some $\delta ~ \geq ~ 0 , ~ V ^ { * } ~ : =$ inf<sub>θ</sub> $L ( \theta ) > - \infty$ and suppose the PL condition in (43) holds. Using (42) and the triangle inequality yields

$$
\begin{array} { r l } & { L ( \theta ) - V ^ { * } \displaystyle \leq \frac { 1 } { 2 \mu } \| \nabla _ { \theta } L ( \theta ) \| ^ { 2 } } \\ & { \qquad \displaystyle \leq \frac { 1 } { 2 \mu } ( \| \mathbb { E } _ { \boldsymbol { u } } [ \widehat { G } _ { \boldsymbol { u } } ] \| + \| \mathbb { E } _ { \boldsymbol { u } } [ \widehat { G } _ { \boldsymbol { u } } ] - \nabla _ { \theta } L ( \theta ) \| ) ^ { 2 } } \\ & { \qquad \displaystyle \leq \frac { ( \delta + \epsilon _ { G } ) ^ { 2 } } { 2 \mu } , } \end{array}
$$

which establishes (44) and thus completes the proof.

Corollary 3 yields a similar two-regime interpretation for the full training loss. If $\| \mathbb { E } _ { \mathcal { U } } [ \widehat { G } _ { \mathcal { U } } ( \theta ) ] \| \ > \ \epsilon _ { G }$ , then $- \mathbb { E } _ { \mathcal { U } } [ \widehat { G } _ { \mathcal { U } } ( \theta ) ]$ is a strict descent direction for L at θ. If instead $\lVert \mathbb { E } _ { \boldsymbol { \mathcal { U } } } [ \widehat { G } _ { \boldsymbol { \mathcal { U } } } ( \boldsymbol { \theta } ) ] \rVert \ \leq \ \delta ,$ then $\| \nabla _ { \theta } L ( \theta ) \| ~ \le ~ \delta + \epsilon _ { G } ;$ in particular, stationarity of the expected truncated-context gradient implies ϵ<sub>G</sub>-stationarity of the full training loss L.

The Polyak–Łojasiewicz condition in (43) can hold for certain nonconvex objectives and is standard in the analysis of gradient-based optimization methods [26]; however, it is not generally guaranteed for nonlinear training problems. We emphasize that the preceding gradient-error, alignment, and stationarity guarantees are independent of this condition. In the present setting, it allows translating approximate fullcontext stationarity into a bound on the global optimality gap $L ( \theta ) - V ^ { * }$ . Specifically, for $\delta = 0$

$$
L ( \theta ) - V ^ { * } \leq \frac { \epsilon _ { G } ^ { 2 } } { 2 \mu } .
$$

According to Corollary 2, increasing the past- or futurecontext length, or increasing the update-interval length while keeping $T _ { \mathrm { P } }$ and $T _ { \mathrm { F } }$ fixed, reduces $\epsilon _ { G }$ and thereby tightens the preceding guarantees.

## IV. NUMERICAL EXAMPLE

We illustrate the theoretical results on the Wiener– Hammerstein benchmark [27]. The dataset comprises a single input–output trajectory with a predefined split of 80,000 training and 20,000 validation samples. We identify a vanilla RNN model with 16 states described by

$$
x ^ { + } = \operatorname { t a n h } ( A x + B u + b ) , \quad y = C x + D u + d ,
$$

where the weighting matrices $A , B , C , D$ and bias vectors $b ,$ d are learned by minimizing the squared output simulation error using $l _ { t } ( x _ { t } ; \theta ) = \| C x _ { t } + D u _ { t } + d - y _ { t } ^ { d } \| ^ { 2 }$ . Implementation details and code for reproducing the subsequent experiments are provided in the accompanying public repository<sup>1</sup>.

We first illustrate the truncation effects characterized by Theorem 1 and Corollary 2. Since the results hold pointwise for arbitrary parameters θ, we consider five fixed reference parameterizations obtained using BPTT for different random seeds. We then evaluate truncated-context gradients for $T _ { \mathrm { P } } , T _ { \mathrm { F } } \in \{ 0 , 5 , 1 0 , 2 0 , 4 0 \}$ . Figure 2 illustrates the pointwise discrepancy $\lVert \hat { g } _ { t } ( \theta ) - g _ { t } ( \theta ) \rVert$ for different symmetric context lengths $T _ { \mathrm { P } } = T _ { \mathrm { F } } ,$ , using one of the five reference parameterizations and a fixed evaluation interval $\mathcal { U } \ = \ [ a , b ]$ of length 300. For short contexts, pronounced errors occur close to both boundaries, whereas increasing the context yields an extended low-error interior region. This is in line with Theorem 1 and directly reflects the two terms on the righthand side of (23): the errors induced at the left and right context boundaries decay exponentially into the interior.

Figure 3 shows the median update-gradient discrepancy $\lVert \widehat { G } _ { \mathcal { U } } ( \boldsymbol { \theta } ) - G _ { \mathcal { U } } ( \boldsymbol { \theta } ) \rVert$ over the full $( T _ { \mathrm { P } } , T _ { \mathrm { F } } )$ grid. For each pair, the five reference parameterizations are evaluated on the same 20 update intervals, and the median is taken over the resulting 100 gradient discrepancies. Increasing the past context consistently reduces the gradient discrepancy, while the benefit of extending the future context becomes particularly pronounced once the forward truncation error has been sufficiently reduced. This can be attributed to the fact that the forward and backward recursions are coupled: errors in the forward trajectory also perturb the backward sensitivities. Overall, the median error decreases from 1.33 for $( T _ { \mathrm { P } } , T _ { \mathrm { F } } ) = ( 0 , 0 )$ to 0.012 for (40, 40), i.e., by a factor of approximately 110, consistent with Corollary 2.

To complement the preceding results, we numerically assess the contraction mechanism underlying Assumption 2 by estimating $\mathrm { m a x } _ { k \in { \mathcal { K } } , x \in [ - 1 , 1 ] ^ { 1 6 } } \| D _ { x } \phi _ { k + 2 0 , k } ( x ) \| _ { 2 }$ . To this end, we evaluate this norm for all $k \in \mathcal { K } : = [ 0 , 1 0 ^ { 5 } - 2 0 ]$ and sampled initial states, followed by multistart local optimization over the initial states for the largest candidates. Across the five reference parameterizations, the largest value found is 0.9708, providing numerical support for Assumption 2.

![](images/1b38cdebd3f6ad55ab066b093c3d6470627ef27d44d0880ed63445ec76b67ae6.jpg)  
Fig. 2: Pointwise discrepancy between the full and truncated-context gradi ent contributions evaluated over a fixed interval $\mathcal { U } = [ a , b ]$ for symmetric context lengths. Increasing the context length exponentially attenuates the boundary effects and thereby extends the interior region in which the truncated-context gradient contributions closely approximate their fullcontext counterparts.

![](images/9245185008db55a6769a6d1866681a0b88c43ca3cb758a6ef5961f8ce4b0b951.jpg)  
Fig. 3: Median absolute discrepancy between the full and truncated-context update gradients as a function of the past and future context lengths. Values are aggregated over 20 common update intervals for each of the five reference parameterizations. Larger two-sided context substantially improves the full-context-gradient approximation.

Finally, in a separate training experiment, we investigate the practical trade-off between truncation and training performance for a fixed training budget of 5,000 optimizer updates and compare it to a challenging full-context BPTT benchmark. The results are reported in Table I, using the model checkpoint achieving the lowest validation loss among those periodically evaluated during training. Test performance is assessed on a separate test sequence used only for final evaluation. For $\begin{array} { r } { T _ { \mathrm { P } } = T _ { \mathrm { F } } = 4 0 . } \end{array}$ , TBPTT approaches the BPTT reference closely, with median training and test RMSEs only 1.11 and 1.12 times those of the seed-paired BPTT reference, respectively, while reducing the training time by a factor of approximately 5.7. Hence, a sufficiently large two-sided context can retain much of the predictive performance of full BPTT while substantially reducing the computational effort. Here, we observe that the training performance is not monotone in the context length: in particular, $T _ { \mathrm { P } } = T _ { \mathrm { F } } = 2 0$ performs worse than $T _ { \mathrm { P } } = T _ { \mathrm { F } } = 1 0$ . Note that this does not contradict the preceding analysis, which characterizes the approximation of the full-context gradient at a fixed parameterization, but rather reflects the finite-budget, stochastic, and nonconvex nature of the resulting optimization problem.

TABLE I: Training and test performance of full-context BPTT and TBPTT with symmetric context $T _ { \mathrm { e } } : = T _ { \mathrm { P } } = T _ { \mathrm { F } }$ . Reported RMSEs are medians over five seeds; values in parentheses are median seed-paired RMSE ratios relative to BPTT; speedups are median seed-paired training-time ratios.
<table><tr><td>Method</td><td> $T _ { \mathrm { e } }$ </td><td>Train RMSE</td><td>Test RMSE</td><td>Speedup</td></tr><tr><td>BPTT</td><td>full</td><td>0.02364 (1.00×)</td><td>0.02284 (1.00×)</td><td>1.00×</td></tr><tr><td>TBPTT</td><td>0</td><td>0.03950 (1.55×)</td><td>0.03850 (1.58×)</td><td>12.99×</td></tr><tr><td>TBPTT</td><td>10</td><td>0.03307 (1.50×)</td><td>0.03248 (1.51×)</td><td>5.71×</td></tr><tr><td>TBPTT</td><td>20</td><td>0.04591 (2.25×)</td><td>0.04527 (2.25×)</td><td>5.64×</td></tr><tr><td>TBPTT</td><td>40</td><td>0.02691 (1.11×)</td><td>0.02622 (1.12×)</td><td>5.68×</td></tr></table>

BPTT uses $| \mathcal { U } | = 1$ , 000; TBPTT uses $| \boldsymbol { u } | = 5 0$ with mini-batch size 4. All models use 5,000 Adam updates and validation-based model selection. The separate benchmark test data comprise 78,800 samples, with the test RMSE computed after an initialization phase of 50 samples according to the benchmark convention [27].

## V. CONCLUSION

In this paper, we showed that training nonlinear recurrent models with truncated temporal context, as in TBPTT, naturally gives rise to a two-sided gradient turnpike property. Specifically, under exponential decay of state-transition Jacobians and mild regularity conditions, the discrepancy between truncated- and full-context gradient contributions decays exponentially away from both truncation boundaries, reflecting missing past information in the forward path and missing future information in the backward path. The resulting bounds quantify how omitted past and future context, as well as the update-interval length, affect the averaged gradient used for parameter updates, yielding guarantees on gradient alignment, approximate full-context stationarity, and, under an additional Polyak–Łojasiewicz condition, a bound on the corresponding optimality gap. Numerical results on the Wiener–Hammerstein benchmark illustrate the two-sided gradient turnpike property and show that TBPTT with a suitably chosen temporal context can substantially reduce computational effort while retaining performance close to full-context BPTT training.

## ACKNOWLEDGMENT

OpenAI ChatGPT was used during the preparation of this manuscript to assist with language refinement throughout the paper, literature discovery, the discussion and verification of mathematical derivations, and implementation and verification of numerical experiments. All AI-assisted content was critically reviewed, verified, and revised.

## REFERENCES

[1] G. Pillonetto, A. Aravkin, D. Gedon, L. Ljung, A. H. Ribeiro, and T. B. Schon, “Deep networks for system identification: A survey,”¨ Automatica, vol. 171, p. 111907, 2025.

[2] H. Hewamalage, C. Bergmeir, and K. Bandara, “Recurrent neural networks for time series forecasting: Current status and future directions,” Int. J. Forecasting, vol. 37, no. 1, pp. 388–427, 2021.

[3] Z. C. Lipton, J. Berkowitz, and C. Elkan, “A critical review of recurrent neural networks for sequence learning,” arXiv:1506.00019, 2015.

[4] P. J. Werbos, “Backpropagation through time: what it does and how to do it,” Proc. IEEE, vol. 78, no. 10, pp. 1550–1560, 1990.

[5] R. J. Williams and J. Peng, “An efficient gradient-based algorithm for on-line training of recurrent network trajectories,” Neural Comput., vol. 2, no. 4, pp. 490–501, 1990.

[6] I. Sutskever, “Training recurrent neural networks,” Ph.D. dissertation, Univ. Toronto, 2013.

[7] G. I. Beintema, M. Schoukens, and R. Toth, “Deep subspace encoders´ for nonlinear system identification,” Automatica, vol. 156, p. 111210, 2023.

[8] S. Kapturowski, G. Ostrovski, J. Quan, R. Munos, and W. Dabney, “Recurrent experience replay in distributed reinforcement learning,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[9] F. Bonassi, M. Farina, J. Xie, and R. Scattolini, “On recurrent neural networks for learning-based control: Recent results and ideas for future developments,” J. Process Contr., vol. 114, pp. 92–104, 2022.

[10] J. D. Schiller, M. Heinrich, V. G. Lopez, and M. A. Muller, “Tuning ¨ the burn-in phase in training recurrent neural networks improves their performance,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2026.

[11] M. Forgione, M. Mejari, and D. Piga, “Learning neural state-space models: Do we need a state estimator?” arXiv:2206.12928, 2022.

[12] C. Aicher, N. J. Foti, and E. B. Fox, “Adaptively truncating backpropagation through time to control gradient bias,” in Proc. 35th Conf. Uncertain. Artif. Intell, ser. PMLR, vol. 115, 2020, pp. 799–808.

[13] C. Tallec and Y. Ollivier, “Unbiasing truncated backpropagation through time,” arXiv:1705.08209, 2017.

[14] J. Miller and M. Hardt, “Stable recurrent models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[15] T. Faulwasser and L. Grune, “Turnpike properties in optimal con-¨ trol: An overview of discrete-time and continuous-time results,” in Handbook of Numerical Analysis, E. Trelat and E. Zuazua, Eds.´ Amsterdam, The Netherlands: Elsevier, 2022, vol. 23, pp. 367–400.

[16] J. D. Schiller, L. Grune, and M. A. M ¨ uller, “Performance guarantees ¨ for optimization-based state estimation using turnpike properties,” arXiv:2501.18385, 2025, conditionally accepted for publication as full paper in IEEE Trans. Autom. Control.

[17] E. Trelat and E. Zuazua, “The turnpike property in finite-dimensional´ nonlinear optimal control,” J. Differ. Equ., vol. 258, no. 1, pp. 81–114, 2015.

[18] J. D. Schiller and M. A. Muller, “Turnpike properties in nonlinear ¨ system identification,” arXiv:2609.02071, 2026.

[19] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.

[20] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proc. Conf. Empir. Methods Nat. Lang. Process. (EMNLP), 2014, pp. 1724– 1734.

[21] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv:2312.00752, 2024.

[22] L. Bottou, F. E. Curtis, and J. Nocedal, “Optimization methods for large-scale machine learning,” SIAM Review, vol. 60, no. 2, pp. 223– 311, 2018.

[23] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[24] S. Xu, A. Khandelwal, X. Li, X. Jia, L. Liu, J. Willard, R. Ghosh, K. Cutler, M. Steinbach, C. Duffy, J. Nieber, and V. Kumar, “Minibatch learning strategies for modeling long term temporal dependencies: A study in environmental applications,” in Proc. SIAM Int. Conf. Data Mining, 2023, pp. 649–657.

[25] Z. Allen-Zhu, Y. Li, and Z. Song, “On the convergence rate of training recurrent neural networks,” in Adv. Neur. Inf. Process. Syst. (NeurIPS), vol. 32, 2019.

[26] H. Karimi, J. Nutini, and M. Schmidt, “Linear convergence of gradient and proximal-gradient methods under the Polyak-Łojasiewicz condition,” in Proc. Jt. Eur. Conf. Mach. Learn. Knowl. Discov. Databases (ECML PKDD), 2016, pp. 795–811.

[27] M. Schoukens, M. Champneys, G. I. Beintema, and T. J. Rogers, “Benchmarking for nonlinear system identification: Submission platform and baseline results,” Data-Centric Eng., vol. 7, no. e40, 2026.