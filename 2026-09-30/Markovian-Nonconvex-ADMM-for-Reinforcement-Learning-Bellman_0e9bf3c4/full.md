# Markovian Nonconvex ADMM for Reinforcement Learning: Bellman-Resolvent Stability Beyond Smooth Blocks

Zhaojun Peng Hantsukipzj@gmail.com

## Abstract

We identify and study a structural mechanism for Markovian nonconvex ADMM in reinforcement learning. Using finite discounted MDPs as a canonical proving ground, we show that the discounted Bellman resolvent $( I - \gamma P _ { \pi } ) ^ { - 1 }$ can provide the multiplier stability that classical nonconvex ADMM analyses often obtain from a designated smooth block. Starting from this mechanism, we establish convergence under controlled Markov sampling and then under stochastic observations using an empirical Bellman surrogate that jointly represents the random residual and its Jacobian. Markov mixing, initialization drift, observation noise, and decaying bias enter as one operator perturbation, avoiding unbiased product and double sampling requirements. When the perturbations are square summable, the true KKT residual converges almost surely to zero. Under a finite conditional fourth moment condition, a companion iterate satisfies $\begin{array} { r } { \mathbb { E } [ \widetilde { G } _ { K + 1 } ] \le A / T + ( B / T ) \sum _ { k < T } m _ { k } ^ { - 1 } } \end{array}$ , which becomes $O ( T ^ { - 1 } + T / N )$ for total Markov sample budget N, giving $O ( \epsilon ^ { - 1 } )$ iteration complexity and $O ( \epsilon ^ { - 2 } )$ sample complexity for squared KKT accuracy ϵ. Beyond stationarity, discounted occupancy coverage yields $J ^ { \star } - J ( \pi ) = O ( \sqrt { G } )$ for direct tabular policies, so covered exact KKT points are globally optimal, while a statewise quadratic Bellman-improvement condition sharpens the relation to $O ( G )$ . Finally, nonlinear policy, projected Bellman, and explicit occupancy formulations exhibit the same chain of operator invertibility, dual representation, and multiplier stability. This supports discounted operator invertibility as a reusable structural principle for primal-dual reinforcement learning.

## 1 Introduction

The alternating direction method of multipliers is a natural tool for constrained nonconvex optimization because it separates dificult variables while preserving the interaction between primal and dual updates. Its analysis is delicate because the multiplier step increases the augmented Lagrangian. What the convergence proof ultimately needs is a structural estimate that controls $\| \Delta \lambda _ { k + 1 } \|$ by quantities dissipated in the primal steps. Existing nonconvex analyses commonly obtain such multiplier stability from a designated smooth block and a suitable linear operator [16, 34, 33]. This observation shifts the question from smoothness itself to the source of dual stability: when regularity is carried by the constraint geometry, can that geometry provide the mechanism for controlling the multiplier directly? We use the finite discounted MDP as a canonical instantiation in which this mechanism can be exposed and verified end to end, from multiplier stability and Markovian perturbations to complexity bounds at a finite time horizon and policy performance.

Discounted reinforcement learning provides such a constraint. For a policy π, the Bellman equation is $M _ { \pi } V - r _ { \pi } = 0$ with $M _ { \pi } = I - \gamma P _ { \pi }$ . Since $0 < \gamma < 1$ and $P _ { \pi }$ is stochastic, $M _ { \pi }$ is invertible and $\begin{array} { r } { M _ { \pi } ^ { - 1 } = \sum _ { t \geq 0 } \gamma ^ { t } P _ { \pi } ^ { t } } \end{array}$ . The central question is whether this resolvent can replace the classical mechanism based on a smooth block in a nonconvex ADMM proof. The answer is afirmative. The optimality condition for the value update and the dual update give $\lambda ^ { k + 1 } = - \widehat { M } _ { k + 1 } ^ { - \mathsf { T } } ( c + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } )$

Stability of the inverse resolvent then converts changes in the policy and value variables into a bound on the multiplier increment.

The theory is organized in three layers. The first is a structural layer. Bellman resolvent stability closes Lyapunov arguments for deterministic and controlled Markov settings without assuming that $\phi$ is smooth. The second is a stochastic layer that preserves the structure of the Bellman operator. Here the dificulty is not variance alone: the augmented policy gradient couples a random Bellman Jacobian with a random Bellman residual, so separate unbiasedness does not remove their covariance. We therefore generate the residual, its Jacobians, and the dual update from one empirical Bellman operator, turning Markov dependence, initialization drift, observation noise, and decaying bias into a single operator perturbation. The third is an RL geometry layer. Occupancy measures, advantage functions, and the performance diference identity turn the optimization residual into a global policy guarantee. This layer further characterizes two performance regimes: occupancy coverage yields the $O ( { \sqrt { G } } )$ guarantee, while a curvature condition on the Bellman improvement at each state sharpens the conversion to $O ( G )$

Our contributions are as follows. First, we identify the discounted Bellman resolvent as a source of multiplier stability specific to RL and prove the corresponding bounds on dual increments and the Lyapunov function. Second, we show that the stochastic extension must preserve the Bellman operator coupling. Under Markov sampling controlled by resets and uniform exploration, one shared empirical Bellman surrogate jointly generates the residual, its Jacobians, and the dual update, converting dependence between the residual and its Jacobian, observation noise, bias, and Markov dependence into a controlled operator perturbation and yielding convergence of the KKT residual to zero almost surely. Third, assuming a finite conditional fourth moment, we introduce a companion iterate aligned with the policy subproblem and prove $\mathbb { E } [ \widetilde { G } _ { K + 1 } ] = O ( T ^ { - 1 } + T / N )$ , yielding $O ( \varepsilon ^ { - 1 } )$ outer iterations and $O ( \varepsilon ^ { - 2 } )$ Markov samples for accuracy ε measured by the squared KKT residual. Fourth, for direct tabular policies, occupancy coverage gives $J ^ { \star } - J ( \pi ) = O ( { \sqrt { G } } )$ and exact covered KKT points are globally optimal. Statewise quadratic Bellman improvement strengthens this to $O ( G )$ . Matching constructions characterize the roles of occupancy coverage and curvature of statewise Bellman improvement in the two performance regimes. Fifth, we use formulations based on nonlinear policies, projected Bellman equations, and occupancy measures to test which parts of the mechanism are specific to the tabular Bellman model. These extensions show that the reusable ingredient is the existence of a stable invertible discounted operator yielding an explicit dual representation.

## 2 Related Work

Nonconvex ADMM. Classical ADMM is well understood in convex settings [9, 7, 15], while nonconvex analyses require additional structure to control multiplier motion and recover descent [16, 20, 34]. Recent approaches weaken objective regularity through restricted strong convexity, smoothing, increasing penalization, or inexact block updates [6, 32, 33, 4]. The multiafine ADMM theory of [12] controls multiplier increments through a fixed linear subblock whose image contains the image of the remaining constraint map. In our Bellman formulation, the value-block Jacobian $M _ { \pi } = I - \gamma P _ { \pi }$ varies with the policy, so this fixed-operator structure does not directly cover the policy–value splitting. We exploit uniform invertibility and inverse-map stability of $M _ { \pi }$ to control multiplier motion, and preserve the resulting value–dual identity under Markov sampling through a shared empirical Bellman operator.

Table 1: Structural positioning across neighboring theories. Rates are reported in each paper’s native stationarity or performance metric.
<table><tr><td>Ref.</td><td>Method</td><td>Data</td><td>Structural assumption</td><td>Stability / analysis device</td><td>Guarantee</td><td>RL meaning</td></tr><tr><td>[16]</td><td>ADMM</td><td>Det.</td><td>Nonconvex consensus / sharing; large penalty</td><td>Consensus / sharing structure + Stationary-set convergence AL descent</td><td></td><td></td></tr><tr><td>[34]</td><td>MEAL</td><td>Det.</td><td>Weakly convex nonsmooth, lin- Moreau envelope of AL ear constraints</td><td></td><td> $o ( \varepsilon ^ { - 2 } )$  FOSP complexity</td><td></td></tr><tr><td>[6]</td><td>ADMM</td><td>Det.</td><td>Nonsmooth nonconvex RSC</td><td>under Restricted strong convexity</td><td>Convergence without differen- tiability</td><td></td></tr><tr><td>[33]</td><td>Prox- ADMM</td><td>Det.</td><td>Multi-block; one</td><td>continuous Increasing penalty + decreasing ity</td><td> $O ( \varepsilon ^ { - 3 } )$  critical-point complex- -</td><td></td></tr><tr><td>[36]</td><td>Stoch. ADMM</td><td>Finite</td><td>block; BI / SU operator Nonconvex nonsmooth linear</td><td>smoothing SVRG + momentum + poten-</td><td> $\overset { \cdot } { O ( 1 / T ) }$  stationarity; linear un- -</td><td></td></tr><tr><td>[30]</td><td>1st-order opt.</td><td>sum Markov</td><td>constraint</td><td>tial Smooth nonconvex gradient or- Markov-oracle lower-bound con-</td><td>der KL  $\Omega ( \varepsilon ^ { - 4 } )$  countable;  $\Omega ( \varepsilon ^ { - 2 } )$ </td><td>fi- RL-relevant</td></tr><tr><td>[13]</td><td>Actor- critic</td><td>Markov</td><td>acle Neural actor / critic; weak gra-</td><td>struction Critic-error control + gradient</td><td>nite chain  $\widetilde { O } ( \varepsilon ^ { - 3 } )$  global sample complex- Global</td><td>oracle last iterate</td></tr><tr><td>Ours</td><td>Bellman- ADMM</td><td>Ctrl. Markov</td><td>dient domination subproblem; invertible dis-</td><td>domination Proper l.s.c. φ; exact policy Resolvent + shared empir- ical operator + companion</td><td> $\mathbb { E } \widetilde { G } \ = \ O ( T ^ { - 1 } \ + \ T / N ) ; \ N \ = \ O ( \sqrt { G } ) ;$   $O ( \varepsilon ^ { - 2 } )$ </td><td>for squared KKT O(G) with</td></tr></table>

Stochastic ADMM and Markovian optimization. Recent stochastic ADMM methods mainly study finite-sum or expectation objectives using variance reduction, momentum, Bregman geometry, adaptive batches, or hybrid estimators [36, 21, 18, 35]. Complementary Markovian optimization theory develops finite-time stochastic approximation, unbounded-noise analysis, lower bounds, and single-trajectory variance reduction [1, 14, 30, 26, 23]. Our stochasticity enters the equalityconstraint operator itself: the Bellman residual and its Jacobian are generated from the same controlled trajectory. We therefore use one shared empirical Bellman operator across the primal and dual updates, preserving row stochasticity, resolvent invertibility, and the value–dual identity while treating residual–Jacobian dependence as operator perturbation.

Policy optimization and occupancy geometry. Performance-diference and occupancy identities connect local policy information to global return [19, 2]. Recent actor–critic and policyoptimization analyses establish global or nonasymptotic guarantees under Markov sampling, neural or general function approximation, average-reward objectives, constraints, and occupancy-induced geometry [13, 31, 25, 10, 8, 11, 22, 27, 28, 29, 3]. Occupancy has also been used directly as an optimization representation [17, 24, 5]. We first establish convergence of the constrained primal–dual system and then use occupancy geometry to convert its KKT residual into global policy performance and the same discounted occupancy structure also yields a second invertible operator in Section 7.

## 3 Problem Formulation and Algorithm

We develop the mechanism through a finite discounted MDP, where the discounted operator, its inverse, the induced Markov sampling process, and the policy-performance geometry can all be characterized explicitly. This model serves as a concrete proving ground for the operator-stability principle rather than an abstract definition of its scope. Section 7 then tests the same mechanism under nonlinear policies, projected Bellman equations, and an independent occupancy formulation.

In detail, we consider a finite discounted MDP $( S , { \mathcal { A } } , P , r , \rho , \gamma )$ with $0 \textless \gamma \textless 1$ . The direct policy space is $\begin{array} { r } { \Pi = \prod _ { s \in \mathcal { S } } \Delta ( \mathcal { A } ) } \end{array}$ . For $\pi \in \Pi$ , define $\begin{array} { r } { P _ { \pi } ( s , s ^ { \prime } ) = \sum _ { a \in \mathcal { A } } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) , \quad r _ { \pi } ( s ) = } \end{array}$ $\textstyle \sum _ { a \in { \mathcal { A } } } \pi ( a \mid s ) r ( s , a )$ , and $M _ { \pi } = I - \gamma P _ { \pi } , \quad R ( \pi , V ) = M _ { \pi } V - r _ { \pi }$ . We study

$$
\operatorname* { m i n } _ { \pi \in \Pi , V \in \mathbb { R } ^ { | S | } } \phi ( \pi ) + c ^ { \mathsf { T } } V \quad \mathrm { s u b j e c t ~ t o } \quad R ( \pi , V ) = 0 .\tag{1}
$$

The function $\phi$ may be nonsmooth and nonconvex. It is proper, lower semicontinuous, and bounded below on Π, with dom $\phi \cap \Pi \neq \emptyset$ . Define $\psi = \phi + \delta _ { \Pi }$ , where $\delta _ { \Pi }$ is the indicator of Π. We use the

Algorithm 1 Markovian Proximal Bellman-ADMM   
1: Initialize $\overline { { ( \pi ^ { 0 } , V ^ { 0 } , \lambda ^ { 0 } ) } }$ and choose $\beta , \eta _ { \pi } , \eta _ { V } > 0$   
2: for $k = 0 , 1 , \ldots , T - 1$ do   
3: Freeze $\bar { \pi } ^ { k }$ and collect a reset-controlled batch of length $m _ { k }$   
4: Construct $( \widehat { P } _ { k } , \widehat { r } _ { k } )$ from non-reset transitions   
5: $\pi ^ { k + 1 } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } \left\{ \widehat { \mathcal { L } } _ { \beta , k } ( \pi , V ^ { k } , \lambda ^ { k } ) + \frac { 1 } { 2 \eta _ { \pi } } \| \pi - \pi ^ { k } \| ^ { 2 } \right\}$   
6: $V ^ { k + 1 } = \arg \operatorname* { m i n } _ { V } \left\{ \widehat { \mathcal { L } } _ { \beta , k } ( \pi ^ { k + 1 } , V , \lambda ^ { k } ) + \frac { 1 } { 2 \eta _ { V } } \| V - V ^ { k } \| ^ { 2 } \right\}$   
7: $\lambda ^ { k + 1 } = \lambda ^ { k } + \beta \widehat { R } _ { k } \overset { \cdot } { ( } \pi ^ { k + 1 } , V ^ { k + 1 } )$   
8: end for

limiting subdiferential $\partial \psi$ for policy stationarity. Standard policy optimization is recovered with $\phi \equiv 0$ and $c = - \mu ,$ where $\mu$ is a training initial-state distribution.

The augmented Lagrangian is $\begin{array} { r } { \mathcal { L } _ { \beta } ( \pi , V , \lambda ) = \phi ( \pi ) + c ^ { \mathsf { T } } V + \langle \lambda , R ( \pi , V ) \rangle + \frac { \beta } { 2 } \| R ( \pi , V ) \| ^ { 2 } . } \end{array}$

Definition 3.1 (Squared KKT residual). For $z = ( \pi , V , \lambda )$ , let

$$
G ( z ) = \mathrm { d i s t ^ { 2 } } \Big ( 0 , \partial \psi ( \pi ) + J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } \lambda \Big ) + \| c + M _ { \pi } ^ { \mathsf { T } } \lambda \| ^ { 2 } + \| R ( \pi , V ) \| ^ { 2 } ,\tag{2}
$$

where $J _ { \pi } ( \pi , V ) = D _ { \pi } R ( \pi , V ) . ~ F o r ~ \phi \equiv 0 , \partial \psi = N _ { \Pi }$

## Controlled Markov sampling

At iteration k, the algorithm freezes an exploratory behavior policy $\bar { \pi } ^ { k }$ and simulates the reset kernel $K _ { k } ( s , s ^ { \prime } ) = ( 1 - \gamma ) \rho ( s ^ { \prime } ) + \gamma P _ { \bar { \pi } ^ { k } } ( s , s ^ { \prime } )$ . A reset indicator records whether a transition came from the environment or from $\rho .$ Only environment transitions are used to estimate P. This distinction prevents the estimator from converging to $K _ { k }$

Assumption 3.2 (Coverage and stochastic observations). There are constants $\rho _ { \mathrm { m i n } } > 0 , \alpha > 0$ and $R _ { \mathrm { m a x } } < \infty$ such that $\rho ( s ) \geq \rho _ { \mathrm { m i n } } , \quad \bar { \pi } ^ { k } ( a \mid s ) \geq \alpha , \quad | r ( s , a ) | \leq R _ { \mathrm { m a x } } .$ A reward observation has the form $Y _ { k , t } = r ( S _ { k , t } , A _ { k , t } ) + \zeta _ { k , t } ,$ where ${ \mathbb E } [ \zeta _ { k , t } \mid \mathcal { G } _ { k , t } ] = 0 , \quad { \mathbb E } [ \vert \zeta _ { k , t } \vert ^ { 4 } \vert \mathcal { G } _ { k , t } ] \le M _ { 4 }$ . Here $\mathcal { G } _ { k , t }$ contains all past observations and the current state, action, and reset indicator, before the current reward and next state are drawn. Fresh reset coins, reset states, and action randomization are independent of the preceding history. On non-reset steps, the conditional next-state law is $\textstyle P ( \cdot \mid S _ { k , t } , A _ { k , t } )$ . The current reward and next state may be dependent.

A batch of length $m _ { k }$ produces action-conditional empirical rows $\widehat { P } _ { k } ( \cdot \mid s , a )$ and rewards ${ \widehat { r } } _ { k } ( s , a )$ . Every unvisited transition row is replaced by a fixed probability vector, and every unvisited reward row uses a fixed bounded value. Hence all empirical rows remain stochastic. For each candidate policy, define $\begin{array} { r } { \widehat { P } _ { \pi , k } = \sum _ { a } \pi ( a \mid s ) \widehat { P } _ { k } ( \cdot \mid s , a ) , \quad \widehat { r } _ { \pi , k } = \sum _ { a } \pi ( a \mid s ) \widehat { r } _ { k } ( s , a ) , ~ \widehat { M } _ { \pi , k } = } \end{array}$ $I - \gamma \widehat { P } _ { \pi , k } , \quad \widehat { R } _ { k } ( \pi , V ) = \widehat { M } _ { \pi , k } V - \widehat { r } _ { \pi , k }$ . Row stochasticity gives $\begin{array} { r } { \| M _ { \pi } ^ { - 1 } \| _ { \infty } \leq \frac { 1 } { 1 - \gamma } , \quad \| \widehat { M } _ { \pi , k } ^ { - 1 } \| _ { \infty } \leq \frac { 1 } { 1 - \gamma } } \end{array}$ Finite-dimensional norm equivalence therefore yields a common spectral bound $\bar { \kappa } < \infty$

## Markovian proximal Bellman-ADMM

Let $\widehat { \mathcal { L } } _ { \beta , k }$ denote (3) with R replaced by $\widehat { R } _ { k }$ . Algorithm 1 uses the same empirical model in the policy, value, and dual updates.

Assumption 3.3 (Algorithmic margins). The policy subproblem admits a global minimizer and is solved exactly. The value subproblem is solved exactly. The initial policy is deterministic and belongs to dom ψ. There is $L _ { M } < \infty$ such that $\| M _ { \pi ^ { \prime } } - M _ { \pi } \| \leq L _ { M } \| \pi ^ { \prime } - \pi \|$ . The reduced objective $b ( \pi ) = \phi ( \pi ) + c ^ { \mathsf { T } } M _ { \pi } ^ { - 1 } r _ { \pi }$ is bounded below on Π. The parameters satisfy

$$
\beta > \operatorname * { m a x } \left\{ \frac { 6 0 \bar { \kappa } ^ { 2 } } { \eta _ { V } } , 3 0 \eta _ { \pi } \bar { \kappa } ^ { 4 } L _ { M } ^ { 2 } \| c \| ^ { 2 } \right\} , \qquad \tau = \frac { 1 } { 8 \eta _ { V } } .\tag{3}
$$

The exact policy solve and a structured nonsmooth instance are described in Appendix C. The initialization satisfies $\mathbb { E } \| V ^ { 0 } \| ^ { 4 } + \mathbb { E } \| \lambda ^ { 0 } \| ^ { 4 } < \infty$

## 4 Resolvent Stability and Lyapunov Descent

The value-step optimality condition is $c + \widehat { M } _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k } + \beta \widehat { M } _ { k + 1 } ^ { \mathsf { T } } \widehat { R } _ { k } ( \pi ^ { k + 1 } , V ^ { k + 1 } ) + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } = 0$ , where $\widehat { M } _ { k + 1 } = \widehat { M } _ { \pi ^ { k + 1 } , k }$ and $\Delta V _ { k + 1 } = V ^ { k + 1 } - V ^ { k }$ . Combining this relation with the dual update gives $c + \widehat { M } _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k + 1 } + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } = 0$ . Thus $\lambda ^ { k + 1 } = - \widehat { M } _ { k + 1 } ^ { - \mathsf { T } } \left( c + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } \right)$

Define the empirical-model error $\begin{array} { r } { \varepsilon _ { k } = \operatorname* { s u p } _ { \pi \in \Pi } \left( \Vert \widehat { P } _ { \pi , k } - P _ { \pi } \Vert + \Vert \widehat { r } _ { \pi , k } - r _ { \pi } \Vert \right) } \end{array}$

Lemma 4.1 (Empirical Bellman moments). There are deterministic constants $C _ { 2 } , C _ { 4 } < \infty$ such that

$$
\mathbb { E } [ \varepsilon _ { k } ^ { 2 } \mid { \mathcal F } _ { k } ] \leq { \frac { C _ { 2 } } { m _ { k } } } , \qquad \mathbb { E } [ \varepsilon _ { k } ^ { 4 } \mid { \mathcal F } _ { k } ] \leq { \frac { C _ { 4 } } { m _ { k } ^ { 2 } } } .\tag{4}
$$

Lemma 4.2 (A priori fourth-moment stability). Under Assumptions 3.2 and 3.3, there are constants $C _ { V } ^ { ( 4 ) } , C _ { \lambda } ^ { ( 4 ) } < \infty$ such that $\begin{array} { r } { \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \| V ^ { k } \| ^ { 4 } \leq C _ { V } ^ { ( 4 ) } } \end{array}$ 2 $\begin{array} { r } { \operatorname* { s u p } _ { k \geq 0 } \mathbb { E } \| \lambda ^ { k } \| ^ { 4 } \leq C _ { \lambda } ^ { ( 4 ) } } \end{array}$

This argument establishes iterate stability directly from the empirical resolvent and the coupled value–dual recursion, providing the independent boundedness needed by the subsequent Lyapunov analysis.

Lemma 4.3 (Bellman-resolvent multiplier control). For $k \geq 1$ , there are deterministic constants $C _ { \pi } , C _ { V } , C _ { \varepsilon } > 0$ such that

$$
\mathbb { E } \| \Delta \lambda _ { k + 1 } \| ^ { 2 } \leq C _ { \pi } \mathbb { E } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + C _ { V } \mathbb { E } \| \Delta V _ { k + 1 } \| ^ { 2 } + C _ { V } \mathbb { E } \| \Delta V _ { k } \| ^ { 2 } + C _ { \varepsilon } \left( { \frac { 1 } { m _ { k } } } + { \frac { 1 } { m _ { k - 1 } } } \right) .\tag{5}
$$

One valid choice for the deterministic primal coeficients is $\begin{array} { r } { C _ { \pi } = 5 \bar { \kappa } ^ { 4 } L _ { M } ^ { 2 } \| c \| ^ { 2 } , \quad C _ { V } = \frac { 5 \bar { \kappa } ^ { 2 } } { \eta _ { V } ^ { 2 } } } \end{array}$

The lemma is the main structural result. It follows from the value-dual identity, inverse-map stability, and the empirical-model moment bounds.

Structure-preserving stochasticization. Sharing an empirical Bellman operator preserves the multiplier identity under transition estimation. For $\widehat { R } = R + e$ and ${ \widehat { J } } = J + G$ , separate zeromean errors do not eliminate the product term $G ^ { \mathsf { T } } e$ e in the augmented policy gradient. We therefore construct one complete empirical Bellman operator per batch and use it consistently in the residual, Jacobians, and dual update. Lemma 4.4 shows that the resulting stochastic efects enter the true primal–dual system through the single operator error $\varepsilon _ { k }$

Lemma 4.4 (Structure-preserving surrogate closure). Let $e _ { k } ( \pi , V ) = \widehat { R } _ { k } ( \pi , V ) - R ( \pi , V )$ . On every path for which $\textstyle \sum _ { k } \varepsilon _ { k } ^ { 2 } < \infty$ and the a priori iterate bounds hold, there are finite constants such that four true-problem relations hold simultaneously. First, the primal steps satisfy $\mathcal { L } _ { \beta } ( \pi ^ { k } , V ^ { k } , \lambda ^ { k } ) ~ -$ $\begin{array} { r } { \bar { \mathcal { L } } _ { \beta } ( \pi ^ { k + 1 } , \bar { V } ^ { k + 1 } , \lambda ^ { k } ) \ge a _ { \pi } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + a _ { V } \| \Delta V _ { k + 1 } \| ^ { 2 } - C \varepsilon _ { k } ^ { 2 } } \end{array}$ . Second, the empirical value-dual identity becomes $c + M _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k + 1 } + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } = r _ { k + 1 }$ with $\| r _ { k + 1 } \| \le C \varepsilon _ { k }$ . Third, the multiplier update becomes $\Delta \lambda _ { k + 1 } = \beta R _ { k + 1 } + d _ { k + 1 }$ with $\| d _ { k + 1 } \| \le C \varepsilon _ { k }$ . Fourth, the true KKT residual obeys $\mathrm { K K T } _ { k + 1 } \ \leq$ $C ( \| \Delta \pi _ { k + 1 } \| + \| \Delta V _ { k + 1 } \| + \| \Delta \lambda _ { k + 1 } \| + \varepsilon _ { k } )$

Remark 4.5 (Shared empirical operators control product bias). Separate unbiasedness of a random residual $\widehat { R } = R + e$ and a random Jacobian $\widehat { J } = J + \widehat { G }$ does not imply an unbiased augmented gradient. Indeed, $\widetilde { J } ^ { \mathsf { T } } ( \lambda + \beta \widehat { R } ) - J ^ { \mathsf { T } } ( \lambda + \beta R ) = G ^ { \mathsf { T } } ( \lambda + \beta R ) + \beta J ^ { \mathsf { T } } e + \beta G ^ { \mathsf { T } } e ,$ , and generally E $[ G ^ { \mathsf { T } } e ] \neq 0$ . Coupling Rb and Jb through one empirical Bellman surrogate turns $G ^ { \mathsf { T } } e$ into a pathwise $O ( \varepsilon _ { k } ^ { 2 } )$ perturbation. The stochastic analysis therefore proceeds through operator error without an unbiased-product or double-sampling condition.

Define the corrected potential $\begin{array} { r } { \Phi _ { k } = \mathcal { L } _ { \beta } ( \pi ^ { k } , V ^ { k } , \lambda ^ { k } ) + \tau \| \Delta V _ { k } \| ^ { 2 } , \quad \Delta V _ { 0 } = 0 . } \end{array}$

Lemma 4.6 (Expected Lyapunov descent). For $k \geq 1$ , there are constants $c _ { \pi } , c _ { V } , c _ { V } ^ { \prime } , C _ { E } > 0$ such that

$$
\mathbb { E } [ \Phi _ { k } ] - \mathbb { E } [ \Phi _ { k + 1 } ] \ge c _ { \pi } \mathbb { E } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + c _ { V } \mathbb { E } \| \Delta V _ { k + 1 } \| ^ { 2 } + c _ { V } ^ { \prime } \mathbb { E } \| \Delta V _ { k } \| ^ { 2 } - C _ { E } \left( \frac { 1 } { m _ { k } } + \frac { 1 } { m _ { k - 1 } } \right) .\tag{6}
$$

There are $\underline { { \boldsymbol { \Phi } } } \in \mathbb { R }$ and $C _ { L } > 0$ such that $\begin{array} { r } { \mathbb { E } [ \Phi _ { k } ] \geq \underline { { \Phi } } - \frac { C _ { L } } { m _ { k - 1 } } } \end{array}$

The empirical subproblems produce suficient decrease for $\widehat { \mathcal { L } } _ { \beta , k }$ . We transfer this decrease to $\mathcal { L } _ { \beta }$ by retaining the full empirical Bellman operator. Conditional Cauchy–Schwarz, Lemma 4.1, and Lemma 4.2 reduce the statistical contribution to $O ( m _ { k } ^ { - 1 } )$ .

Theorem 4.7 (Controlled-Markov Bellman-ADMM convergence). Assume the finite-MDP, resetaccess, uniform-exploration, resolvent, and parameter conditions above, with no additional observation noise. $I f \textstyle \sum _ { k } m _ { k } ^ { - 1 } < \infty$ , then almost surely $\Delta \pi _ { k } \to 0 , \Delta V _ { k } \to 0 , \Delta \lambda _ { k } \to 0$ , and $R ( \pi ^ { k } , V ^ { k } )  0$ Every accumulation point o $f \left( \pi ^ { k } , V ^ { k } , \lambda ^ { k } \right)$ satisfies the KKT conditions of (1).

Theorem 4.8 (Structure-preserving stochastic KKT convergence). Assume the model, sampling, and algorithmic conditions above, with the fourth-moment noise condition replaced by conditional variance at most $\sigma _ { k } ^ { 2 }$ . Allow the average bias in each visited state-action reward row to have magnitude at most $\begin{array} { r } { b _ { k } . \ I f \sum _ { k = 0 } ^ { \infty } \frac { 1 + \sigma _ { k } ^ { 2 } } { m _ { k } } < \infty , \quad \sum _ { k = 0 } ^ { \infty } b _ { k } ^ { 2 } < \infty } \end{array}$ , then, almost surely, $\Delta \pi _ { k } \to 0 , \quad \Delta V _ { k } \to$ $0 , \quad \Delta \lambda _ { k } \to 0 , \quad R ( \pi ^ { k } , V ^ { k } ) \to 0$ . The true KKT residual converges to zero. Every accumulation point satisfies the KKT conditions of (1).

## 5 KKT Convergence and Complexity

The policy subproblem is solved at $( \pi ^ { k + 1 } , V ^ { k } )$ . We evaluate a companion iterate that preserves this first-order alignment. Define $\widetilde { \lambda } ^ { k + 1 } = \lambda ^ { k } + \beta \widehat { R } _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) , \quad \widetilde { z } ^ { k + 1 } \overset { \widehat { } } { = } ( \pi ^ { k + 1 } , V ^ { k } , \widetilde { \lambda } ^ { k + 1 } )$ , and $\widetilde { G } _ { k + 1 } =$ $G ( \widetilde { z } ^ { k + 1 } )$ .

Lemma 5.1 (Companion residual). For $k \geq 1$ , there are deterministic constants $C _ { G } , C _ { G } ^ { \prime } > 0$ such that

$$
\mathbb { E } [ \widetilde { G } _ { k + 1 } ] \le C _ { G } \mathbb { E } \left[ \| \Delta \pi _ { k + 1 } \| ^ { 2 } + \| \Delta V _ { k + 1 } \| ^ { 2 } + \| \Delta V _ { k } \| ^ { 2 } \right] + C _ { G } ^ { \prime } \left( \frac { 1 } { m _ { k } } + \frac { 1 } { m _ { k - 1 } } \right) .\tag{7}
$$

Theorem 5.2 (Finite-time squared-KKT complexity). Under Assumptions 3.2 and 3.3, let K be uniform on $\{ 0 , \ldots , T - 1 \}$ and independent of the algorithmic randomness. There are constants $A , B > 0$ that do not depend on T or the batch schedule such that $\begin{array} { r } { \mathbb { E } [ \widetilde { G } _ { K + 1 } ] \le \frac { A } { T } + \frac { B } { T } \sum _ { k = 0 } ^ { T - 1 } \frac { 1 } { m _ { k } } } \end{array}$ . If $\begin{array} { r } { N = \sum _ { k = 0 } ^ { T - 1 } m _ { k } } \end{array}$ and $m _ { k } = N / T$ , then $\begin{array} { r } { \mathbb { E } [ \widetilde { G } _ { K + 1 } ] \le \frac { A } { T } + \frac { B T } { N } } \end{array}$ . Consequently, $T ( \varepsilon ) = O ( \varepsilon ^ { - 1 } ) , \quad N ( \varepsilon ) =$ $O ( \varepsilon ^ { - 2 } )$ are suficient for $\mathbb { E } [ \widetilde { G } _ { K + 1 } ] \le \varepsilon$

The theorem measures a squared residual. For the unsquared residual ${ \tilde { R } } = { \sqrt { \widetilde { G } } }$ , Jensen’s inequality gives $T = { \cal { O } } ( \varepsilon ^ { - 2 } )$ and $N = O ( \varepsilon ^ { - 4 } )$

Corollary 5.3 (Iteration and Markov sample complexity). $\begin{array} { r } { I f \left( 1 / T \right) \sum _ { k < T } m _ { k } ^ { - 1 } = O ( T ^ { - 1 } ) } \end{array}$ , then $\mathbb { E } \widetilde { G } _ { K + 1 } \ : = \ : O ( T ^ { - 1 } )$ . For fixed T and total budget N, equal batches minimize $\sum _ { k } m _ { k } ^ { - 1 }$ and give $\mathbb { E } \widetilde { G } _ { K + 1 } \le A / T + B T / N$ . Thus $T = { \cal { O } } ( \varepsilon ^ { - 1 } )$ and $N = O ( \varepsilon ^ { - 2 } )$ sufice for squared residual ε.

## 6 From KKT Residual to Policy Performance

We now set $\phi \equiv 0$ and $c = - \mu$ . Let $V ^ { \pi } = M _ { \pi } ^ { - 1 } r _ { \pi } , \quad J _ { \nu } ( \pi ) = \nu ^ { \mathsf { T } } V ^ { \pi } , \quad F _ { \mu } ( \pi ) = - J _ { \mu } ( \pi )$ . Define the policy-stationarity residual, following the occupancy/gradient-domination viewpoint in [2, 17, 29, 5], $R _ { \mathrm { p o l } } ( \pi ) = \mathrm { d i s t } \left( 0 , \nabla F _ { \mu } ( \pi ) + N _ { \Pi } ( \pi ) \right)$

Residual-transfer chain. The policy-performance result rests on five explicit intermediate statements. Lemma D.1 gives the Bellman adjoint representation $\nabla F _ { \mu } ( \pi ) = J _ { \pi } ( \pi , V ^ { \pi } ) ^ { \mathsf { T } } \lambda _ { \mu } ^ { \pi }$ with $\lambda _ { \mu } ^ { \pi } =$ $M _ { \pi } ^ { - \mathsf { T } } \mu$ . Lemma D.2 gives the tabular Jacobian $\partial R _ { s } ( \pi , V ) / \partial \pi ( a \mid s ) = - Q _ { V } ( s , a )$ and therefore $[ J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } \lambda ] _ { s , a } \ = \ - \lambda _ { s } Q _ { V } ( s , a )$ . Lemma D.3 gives $\| V - V ^ { \pi } \| ~ \leq ~ \bar { \kappa } e _ { R }$ from Bellman feasibility. Lemma D.4 gives $\| \lambda - \lambda _ { \mu } ^ { \pi } \| \leq$ κe¯ from value-stationarity error. Lemma D.5 combines them into $\begin{array} { r } { \| J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } \lambda - \nabla F _ { \mu } ( \pi ) \| \le C _ { V } e _ { V } + C _ { R } e _ { R } + C _ { V R } e _ { V } e _ { R } . } \end{array}$

Lemma 6.1 (KKT residual controls policy stationarity). Let $G = G ( \pi , V , \lambda ) . \ I f \ G \ \leq \ 1$ , then $R _ { \mathrm { p o l } } ( \pi ) \le C _ { \mathrm { s t a t } } \sqrt { G }$ , where $C _ { \mathrm { s t a t } } < \infty$ depends only on the discounted resolvent and bounded model quantities.

Let $d _ { \mu } ^ { \pi } = ( 1 - \gamma ) M _ { \pi } ^ { - \top } \mu$ be the normalized discounted state occupancy. For an optimal policy $\pi ^ { \star }$ under evaluation distribution $\nu ,$ define $\begin{array} { r } { \mathcal { C } _ { 2 } ( \pi ^ { \star } , \pi , \nu , \mu ) = \left\| \frac { d _ { \nu } ^ { \pi ^ { \star } } } { d _ { \mu } ^ { \pi } } \right\| _ { 2 } } \end{array}$ , with $0 / 0 = 0$ and value $+ \infty$ when the numerator is positive and the denominator is zero.

Theorem 6.2 (Occupancy-mismatch performance bound). If supp $d _ { \nu } ^ { \pi ^ { \star } } \subseteq \mathrm { s u p p } d _ { \mu } ^ { \pi }$ , then $J _ { \nu } ( \pi ^ { \star } ) -$ $J _ { \nu } ( \pi ) \le \sqrt { 2 } { \mathcal C } _ { 2 } ( \pi ^ { \star } , \pi , \nu , \mu ) R _ { \mathrm { p o l } } ( \pi ) . \ I f G \le 1$ , then $J _ { \nu } ( \pi ^ { \star } ) - J _ { \nu } ( \pi ) \leq \sqrt { 2 } \mathcal { C } _ { 2 } C _ { \mathrm { s t a t } } \sqrt { G }$ . Every covered exact KKT point is globally optimal under ν.

Corollary 6.3 (Full-support global optimality). If the training distribution satisfies $\mu ( s ) \geq \mu _ { \mathrm { m i n } } >$ 0, then $J _ { \mu } ^ { \star } - J _ { \mu } ( \pi ) \leq \sqrt { 2 } R _ { \mathrm { p o l } } ( \pi ) / ( ( 1 - \gamma ) \mu _ { \mathrm { m i n } } )$ . Hence every exact policy-stationary point is globally optimal.

Corollary 6.4 (KKT global optimality and finite-time policy bound). Every exact KKT point satisfying the occupancy-cover condition in Theorem 6.2 is globally optimal under ν. If along the random output iterate K the mismatch coeficient is bounded by $\bar { C } _ { \mathrm { o c c } }$ and $\begin{array} { r } { \mathbb { E } G _ { K } \le A / T + ( B / T ) \sum _ { k < T } m _ { k } ^ { - 1 } } \end{array}$ then $\begin{array} { r } { \mathbb { E } \big [ J _ { \nu } ^ { \star } - J _ { \nu } ( \pi _ { K } ) \big ] \leq C \sqrt { A / T + ( B / T ) \sum _ { k < T } m _ { k } ^ { - 1 } } } \end{array}$

Proposition 6.5 (Coverage cannot be removed). There exists a three-state discounted MDP and a training distribution concentrated at the initial state for which a policy is first-order stationary but strictly globally suboptimal. In the construction, the current policy never visits a downstream state. Changing only the upstream action or only the downstream action is non-improving, while changing both raises return from 1 to 9. Hence occupancy coverage is a first-order identifiability condition.

Proposition 6.6 (Separation between performance gap and squared policy stationarity). There is a three-stage chain and policies $\pi _ { \varepsilon }$ such that $J ^ { \star } - J ( \pi _ { \varepsilon } ) = \gamma ^ { 2 } \varepsilon ^ { 3 }$ while $R _ { \mathrm { p o l } } ( \pi _ { \varepsilon } ) ^ { 2 } = ( 3 / 2 ) \gamma ^ { 4 } \varepsilon ^ { 4 }$ Therefore no finite constant C can satisfy $J ^ { \star } - J ( \pi ) \le C R _ { \mathrm { p o l } } ( \pi ) ^ { 2 }$ for all direct tabular policies.

For state s, define $\bar { r } _ { s } ( \pi ) = \mathrm { d i s t } \left( 0 , - Q ^ { \pi } ( s , \cdot ) + { \cal N } _ { \Delta ( A ) } ( \pi _ { s } ) \right) , \quad \delta _ { s } ^ { \pi } = T V ^ { \pi } ( s ) - V ^ { \pi } ( s )$

Assumption 6.7 (Quadratic Bellman improvement). There is $\mu _ { B } > 0$ such that $\begin{array} { r } { \delta _ { s } ^ { \pi } \leq \frac { 1 } { 2 \mu _ { B } } \bar { r } _ { s } ( \pi ) ^ { 2 } } \end{array}$ for every relevant state and policy.

Theorem 6.8 (Quadratic performance conversion). Suppose Assumption 6.7 holds and $\mathcal { C } _ { \mathrm { P L } } ~ =$ $\Big | \Big | \frac { d _ { \nu } ^ { \pi ^ { \star } } } { ( d _ { \mu } ^ { \pi } ) ^ { 2 } } \Big | \Big | _ { \infty } < \infty$ . Then $\begin{array} { r } { J _ { \nu } ( \pi ^ { \star } ) - J _ { \nu } ( \pi ) \leq \frac { 1 - \gamma } { 2 \mu _ { B } } \mathcal { C } _ { \mathrm { P L } } R _ { \mathrm { p o l } } ( \pi ) ^ { 2 } . \ I f G \leq 1 } \end{array}$ , the right-hand side is at most $\frac { 1 - \gamma } { 2 \mu _ { B } } \mathcal { C } _ { \mathrm { P L } } C _ { \mathrm { s t a t } } ^ { 2 } G .$

A uniform statewise action gap provides a concrete suficient condition. If every suboptimal action satisfies max $_ b Q ^ { \pi } ( s , b ) - Q ^ { \pi } ( s , a ) \geq \Delta > 0$ on the policy set of interest, then Assumption 6.7 holds with $\mu _ { B } = \Delta / 4$ . Appendix D proves this implication.

Theorem 6.9 (RL-specific stronger conclusion). Suppose a random companion iterate satisfies the finite-time squared-KKT bound in Theorem 5.2. Under a uniform occupancy-mismatch bound, its expected policy gap is $O ( { \sqrt { \mathbb { E } G } } )$ . Hence $G  0$ implies convergence to globally optimal policy performance and every covered exact KKT point is globally optimal. If the statewise quadratic Bellman-improvement condition and the stronger occupancy ratio bound hold, the expected policy gap is O(EG).

## 7 Extensions to Other RL Formulations

The preceding theory raises a structural question: is multiplier stability specific to the tabular Bellman resolvent, or does it follow from a broader property of discounted RL operators? We show that the same mechanism persists across nonlinear policy parameterizations, projected Bellman equations, and explicit occupancy formulations whenever the induced operator is uniformly invertible and stable. In each case, the proof follows the same chain: operator invertibility → explicit dual representation → multiplier control → Lyapunov descent. The stochastic occupancy formulation further shows that this principle survives empirical operator perturbations when the same structure-preserving operator is shared across primal and dual updates.

## Smooth Nonlinear Policies

The operator-stability mechanism does not depend on direct tabular policy coordinates. Let π<sub>θ</sub> be smooth, define $R ( \theta , V ) = M _ { \theta } V - r _ { \theta }$ , and suppose $\lVert M _ { \theta } - M _ { \theta ^ { \prime } } \rVert ~ \leq L _ { M } \lVert \theta - \theta ^ { \prime } \rVert$ and $b _ { A } ( \theta ) =$ $\phi ( \theta ) + c ^ { \mathsf { T } } M _ { \theta } ^ { - 1 } r _ { \theta }$ is lower bounded. E.1 gives $\lVert M _ { \theta } ^ { - 1 } \rVert _ { 2 } ~ \le ~ \sqrt { | \boldsymbol { S } | } / ( 1 - \gamma )$ E.2 gives $\| M _ { \theta } ^ { - 1 } -$ $M _ { \theta ^ { \prime } } ^ { - 1 } \lVert \leq \kappa _ { M } ^ { 2 } \dot { L } _ { M } \lVert \theta - \theta ^ { \prime } \rVert$ . E.3 gives exact parameter-step descent $\begin{array} { r } { \mathcal L _ { \beta } ( \theta ^ { k } , V ^ { k } , \lambda ^ { k } ) - \mathcal L _ { \beta } ( \theta ^ { k + 1 } , V ^ { k } , \check { \lambda } ^ { k } ) \ge } \end{array}$ $( 2 \eta _ { \theta } ) ^ { - 1 } \| \Delta \theta _ { k + 1 } \| ^ { 2 }$ . E.4 gives $\lambda ^ { k + 1 } = - M _ { k + 1 } ^ { - \mathsf { T } } ( c + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } )$ . E.5 gives $\begin{array} { r } { \| \Delta \lambda _ { k + 1 } \| ^ { 2 } \leq A _ { A } \| \Delta \theta _ { k + 1 } \| ^ { 2 } + } \end{array}$ $B _ { A } \| \Delta V _ { k + 1 } \| ^ { 2 } + B _ { A } \| \Delta V _ { k } \| ^ { 2 }$ . E.6 gives $\Phi _ { k } ^ { A } \stackrel { . } { \geq } b _ { A } ^ { \operatorname* { i n f } } + ( \beta / 4 ) \| R _ { k } \| ^ { 2 } + ( \tau - \kappa _ { M } ^ { 2 } / ( \beta \eta _ { V } ^ { 2 } ) ) \| \Delta V _ { k } \| ^ { 2 }$ . E.7 shows that the corresponding positive coeficient conditions imply $\Delta \theta _ { k } , \Delta V _ { k } , \Delta \lambda _ { k } , R _ { k }  0$ and KKT convergence.

For a computable proximal-gradient parameter update, define the smooth augmented term $h _ { k } ( \theta )$ E.8 gives $\bar { \mathcal { L } _ { \beta } ( \boldsymbol { \theta } ^ { k } , \boldsymbol { V } ^ { k } , \bar { \lambda } ^ { k } ) } - \mathcal { L } _ { \beta } ( \bar { \boldsymbol { \theta } ^ { k + 1 } } , \boldsymbol { V } ^ { \bar { k } } , \lambda ^ { k } ) \ge ( 1 / ( \bar { 2 } \eta _ { \theta , k } ) - L _ { \theta , k } / 2 ) \| \Delta \theta _ { k + 1 } \| ^ { 2 }$ . E.9 shows that the margin $1 / ( 2 \eta _ { \theta , k } ) - L _ { \theta , k } / 2 - A _ { A } / \beta \geq \alpha _ { \theta } > 0$ yields Lyapunov descent and mi $\begin{array} { r } { \operatorname { 1 } _ { k \leq T } G _ { k } ^ { A } = O ( T ^ { - 1 } ) } \end{array}$ The step sizes also satisfy $\eta _ { \theta , k } \ge \eta > 0$ , with uniformly bounded $L _ { \theta , k } , D _ { \theta } M _ { \theta }$ , and $D _ { \theta } r _ { \theta }$ . A uniform smoothness bound and the restarted backtracking rule in Appendix E ensure these conditions.

## Projected Bellman Equations

The same mechanism also survives value-function approximation. What is required for primal–dual stability is an invertible projected Bellman operator, rather than an exact state-value representation. Let $V _ { w } = \Xi w$ , fix positive diagonal D, and write $A _ { \theta } = \Xi ^ { \mathsf { T } } D M _ { \theta } \Xi , b _ { \theta } = \Xi ^ { \mathsf { T } } D r _ { \theta }$ . Assume $A _ { \theta }$ is uniformly invertible and Lipschitz, with the transformed regularity and step-size conditions of Theorem F.2. F.1 gives the projected value-dual identity $\nu ^ { k + 1 } = - A _ { k + 1 } ^ { - \mathsf { T } } ( c _ { w } + \eta _ { w } ^ { - 1 } \Delta w _ { k + 1 } )$ Theorem F.2 gives projected-KKT convergence and an $O ( T ^ { - 1 } )$ minimum squared residual. If the projected Bellman map contracts with modulus $q \ < \ 1$ F.3 gives $\lVert \bar { V } ^ { \bar { \theta } } - V ^ { \theta } \rVert ~ \leq ~ \lVert ( I -$ $\Pi _ { D } ) V ^ { \theta } \rVert / ( 1 - q )$ . Proposition F.4 shows that an $O ( \varepsilon )$ value error can coexist with an $O ( \sqrt \varepsilon )$ parameter-derivative error, so value approximation alone cannot control the policy gradient. F.5 instead gives $\begin{array} { r } { \| D _ { \theta } R ( \theta , V ^ { \theta } ) ^ { \dagger } \lambda ^ { \theta } - D _ { \theta } R ( \hat { \theta } , \hat { V } ^ { \theta } ) ^ { \top } \widehat { \lambda } ^ { \theta } \| \leq C _ { V } \| V ^ { \theta } - \bar { V } ^ { \theta } \| + C _ { \lambda } \| \lambda ^ { \theta } - \widehat { \lambda } ^ { \theta } \| } \end{array}$ , while F.6 gives $\begin{array} { r } { \| \lambda ^ { \theta } - \widehat \lambda ^ { \theta } \| \leq C _ { \mathrm { d u a l } } \operatorname* { i n f } _ { z \in \mathrm { r a n g e } ( D \Xi ) } \| \lambda ^ { \theta } - z \| } \end{array}$

## Explicit Occupancy Measures

More importantly, the stability principle is not tied to the Bellman value formulation itself. Discounted occupancy flow provides a second RL operator with the same algebraic role. Let $x = ( q , d )$ $b = ( ( 1 - \gamma ) \rho , 0 )$ , and $K _ { \theta } = \left[ \begin{array} { c c } { { - \gamma P ^ { \mathsf { T } } } } & { { I } } \\ { { I } } & { { - \Pi _ { \theta } } } \end{array} \right]$ . G.1 proves $K _ { \theta }$ invertible and $\| K _ { \theta } ^ { - 1 } \| _ { 1 , \oplus } \le 2 / ( 1 - \gamma )$ Proposition G.2 shows that $K _ { \theta } x \ = \ b$ has the unique nonnegative solution $x ^ { \theta } = ( q ^ { \theta } , d ^ { \theta } )$ . Theorem G.3 proves KKT equivalence between the constrained occupancy problem and the reduced policy objective. G.4 gives the occupancy-dual identity $y ^ { k + 1 } = - K _ { k + 1 } ^ { - \mathsf { T } } ( c _ { x } + \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } )$ . G.5 gives deterministic occupancy-KKT convergence under the regularity, positive step-size lower bound, and strict margins in Theorem G.5. Lemma G.6 gives the error bound $\| x - x ^ { \theta } \| \leq \kappa _ { K } \| K _ { \theta } x - b \|$

## Stochastic occupancy operator

The stochastic occupancy formulation provides a second test of the structure-preserving principle from Section 4. The occupancy gradient contains $\widehat { K } ^ { \mathsf { T } } \widehat { K }$ , and sharing the empirical operator preserves the value–dual identity. We therefore construct one empirical transition matrix per iteration and share the induced $\widehat { K } _ { k }$ across the policy, occupancy, and dual steps. H.1 proves a regenerative estimator with $\mathbb { E } _ { k } \| \widehat { P } _ { k } - P \| _ { F } ^ { 2 } \le C _ { P } / m _ { k }$ and $\mathrm { P r } _ { k } ( \| \hat { P } _ { k } - P \| _ { F } \geq t ) \stackrel { - } \leq 2 | S | ^ { 2 } | A | \bar { e } ^ { - c _ { P } m _ { k } t ^ { 2 } }$ . H.2 shows every empirical occupancy operator is automatically invertible with a deterministic uniform inverse bound, and $\| \widehat { K } _ { k } ^ { - 1 } - \bar { K } _ { \theta } ^ { - \bar { 1 } } \| \le \bar { \kappa } _ { K } \kappa _ { K } \delta _ { k }$ H.3 gives $y ^ { k + 1 } = - \widehat { K } _ { k } ( \theta ^ { k + 1 } ) ^ { - \mathsf { T } } ( c _ { x } + \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } )$ . H.4 bounds $\Delta y _ { k + 1 }$ by $\Delta \theta _ { k + 1 } , \Delta x _ { k + 1 } , \Delta x _ { k } , \delta _ { k }$ , and $\delta _ { k - 1 }$

H.5 gives the perturbed true Lyapunov lower bound $\Phi _ { k } \geq B _ { C } ^ { \mathrm { i n f } } + ( \beta / 4 ) \| F _ { k } \| ^ { 2 } + q _ { x } \| \Delta x _ { k } \| ^ { 2 }$ $C _ { 0 } \delta _ { k - 1 } ^ { 2 }$ . Defining $\Psi _ { k } = \Phi _ { k } - B _ { C } ^ { \mathrm { i n f } } + 1 + C _ { 0 } \delta _ { k - 1 } ^ { 2 }$ , H.6 proves that the direct policy-gradient perturbation vanishes by block-row orthogonality, and hence bounds the policy-gradient perturbation by $| \langle \widehat { g } _ { \theta , k } - g _ { \theta , k } , \Delta \theta _ { k + 1 } \rangle | \leq \epsilon \| \Delta \theta _ { k + 1 } \| ^ { 2 } + C _ { \epsilon } \delta _ { k } ^ { 2 } \Psi _ { k }$ . Under the uniform bounds and parameter conditions in Appendix H, H.7 gives, for $k \geq 1$ , the one-step recursion $\mathbb { E } _ { k } \Psi _ { k + 1 } \leq ( 1 + a / m _ { k } ) \Psi _ { k } -$ $c _ { \theta } \mathbb { E } _ { k } \| \Delta \theta _ { k + 1 } \| ^ { 2 } - c _ { x } \mathbb { E } _ { k } \| \Delta x _ { k + 1 } \| ^ { 2 } - c _ { x } ^ { \prime } \| \Delta x _ { k } \| ^ { 2 } + b / m _ { k }$ . H.8 controls the true squared KKT residual by the increment energy plus $m _ { k } ^ { - 1 } \Psi _ { k } + m _ { k } ^ { - 1 } + \delta _ { k - 1 } ^ { 2 } .$

H.9 Let $\begin{array} { r } { A _ { T } = \sum _ { k < T } m _ { k } ^ { - 1 } } \end{array}$ . Then sup <sub>≤ ≤</sub> $\mathbb { E } \Psi _ { k } \le e ^ { C _ { 1 } A _ { T } } ( C _ { \mathrm { i n i t } } + C _ { 2 } A _ { T } )$ and, for an independent uniform output $K \in \{ 1 , \ldots , T \}$ $\mathbb { E } G _ { K } ^ { \mathrm { t r u e } ^ { - } } \le C _ { 3 } e ^ { C _ { 1 } A _ { T } } \bigl ( 1 + A _ { T } \bigr ) ( T ^ { - 1 } + A _ { T } / T )$ . Corollary H.10 gives $\mathbb { E } G _ { K } ^ { \mathrm { t r u e } } = O ( T ^ { - 1 } + T / N )$ for equal batches when $N \gtrsim T ^ { 2 }$ . H.11 If $\textstyle \sum _ { k } m _ { k } ^ { - 1 } < \infty$ , then $\Psi _ { k }$ converges, $\begin{array} { r } { \sum _ { k } ( \| \Delta \theta _ { k + 1 } \| ^ { 2 } + \| \Delta x _ { k + 1 } \| ^ { 2 } + \| \Delta x _ { k } \| ^ { 2 } ) < \infty , \delta _ { k } \to 0 . } \end{array}$ , and $G _ { k } ^ { \mathrm { t r u e } } \to 0$ almost surely. Corollary H.12 gives the corresponding nonlinear-policy performance conversion under an additional parameterization-specific domination condition.

Unified principle and rate hierarchy. The Bellman resolvent $M _ { \pi } ^ { - 1 }$ , the projected Bellman operator $A _ { \theta } ^ { - 1 }$ , and the occupancy operator $K _ { \theta } ^ { - 1 }$ are three realizations of the same stability mechanism. In each case, uniform invertibility converts primal optimality into an explicit dual representation, whose perturbation stability controls multiplier motion. Under stochastic data, sharing one empirical operator across the coupled updates preserves this representation pathwise and converts Markov dependence, observation noise, and bias into operator perturbations. For the Bellman formulation, this yields the squared-KKT rate $\begin{array} { r } { \mathbb E [ \widetilde G _ { K + 1 } ] = \bar { O ( } T ^ { - 1 } + \bar { T } ^ { - 1 } \sum _ { k < T } m _ { k } ^ { - 1 } ) } \end{array}$ , while the policy-performance conversion gives $J ^ { \star } - J ( \pi ) = O ( { \sqrt { G } } )$ under occupancy coverage and $O ( G )$ under the stronger statewise quadratic improvement condition. The stochastic occupancy formulation exhibits the same operator-stability mechanism with its corresponding sampling rate. Taken together, these results identify discounted-operator invertibility as a reusable structural source of primal–dual stability in reinforcement learning.

## 8 Numerical Mechanism Validation

We validate the structural predictions on eight independently generated finite discounted MDPs, with five trajectory seeds per instance. Figure 1 summarizes shared-operator stability and finitetime scaling. Additional three-way ablations in Appendix I show that, at 320 steps per empirical model, sharing reduces the trailing true squared KKT residual by a factor of 8.27 relative to separate operators, while using one third of the sampling budget. Sharing between the value and dual updates preserves the value–dual identity even with a separate policy model, identifying the consistency underlying multiplier control. Under $m _ { k } = T$ , hence $N = T ^ { 2 }$ , the mean companion squared KKT residual exhibits slope −0.87, consistent with the $O ( T ^ { - 1 } )$ dependence in Theorem 5.2. Further statistical controls and sensitivity results appear in Appendix I.

## 9 Conclusion

We established a Markovian nonconvex ADMM theory in which discounted RL operator structure supplies primal-dual stability. The Bellman resolvent replaces the classical smooth-block mechanism, the structure-preserving empirical Bellman model lifts the argument to controlled Markov and stochastic data, the companion iterate yields finite-time KKT complexity, and occupancy geometry converts stationarity into global policy quality. The nonlinear-policy, projected-Bellman, deterministic occupancy, and stochastic occupancy results show that this mechanism extends beyond the tabular Bellman formulation. Taken together, these results identify uniformly invertible discounted RL operators as a reusable structural source of primal-dual stability.

![](images/5a74ecb9ff251e56691a7c5f05ce0fa46379c5b48d76776586fd265fc4d6d997.jpg)  
(a) KKT residual

![](images/bab3ea83ca12d2862846ed88c2747e8c16cf2af0b1656670a8a920021fe21859.jpg)  
(b) Value–dual identity

![](images/4e8b2eef9d4be666a6a3328ab8972e2b56e715731a17e83e7e9969cf2f690941.jpg)  
(c) Finite-time scaling  
Figure 1: Numerical validation of the operator-stability mechanism. (a) A shared empirical Bellman operator yields substantially smaller true squared KKT residuals than an equal-budget split-operator construction. (b) Sharing the operator preserves the empirical value–dual identity to numerical precision, whereas inconsistent blockwise operators break this identity. (c) With $m _ { k } = T$ and $N = T ^ { 2 }$ , the companion squared KKT residual exhibits slope −0.87, close to the $T ^ { - 1 }$ scaling predicted by Theorem 5.2.

## References

[1] Arman Adibi, Nicolò Dal Fabbro, Luca Schenato, Sanjeev Kulkarni, H. Vincent Poor, George J. Pappas, Hamed Hassani, and Aritra Mitra. Stochastic approximation with delayed updates: Finite-time rates under markovian sampling. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 2746–2754. PMLR, 2024.

[2] Alekh Agarwal, Sham M. Kakade, Jason D. Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22(98):1–76, 2021.

[3] Reza Asad, Reza Babanezhad Harikandeh, Issam H. Laradji, Nicolas Le Roux, and Sharan Vaswani. Fast convergence of softmax policy mirror ascent. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3943–3951. PMLR, 2025.

[4] Jianchao Bai, Miao Zhang, and Hongchao Zhang. An inexact admm for separable nonconvex and nonsmooth optimization. Computational Optimization and Applications, 90:445–479, 2025.

[5] Anas Barakat, Souradip Chakraborty, Peihong Yu, Pratap Tokekar, and Amrit Singh Bedi. On the global optimality of policy gradient methods in general utility reinforcement learning. In Advances in Neural Information Processing Systems, volume 38, 2025.

[6] Rina Foygel Barber and Emil Y. Sidky. Convergence for nonconvex admm, with applications to ct imaging. Journal of Machine Learning Research, 25(38):1–46, 2024.

[7] Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1–122, 2011.

[8] Xuyang Chen and Lin Zhao. On the convergence of continuous single-timescale actor-critic. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 9829–9859. PMLR, 2025.

[9] Daniel Gabay and Bertrand Mercier. A dual algorithm for the solution of nonlinear variational problems via finite element approximation. Computers & Mathematics with Applications, 2(1):17–40, 1976.

[10] Swetha Ganesh, Jiayu Chen, Washim Uddin Mondal, and Vaneet Aggarwal. Order-optimal global convergence for actor-critic with general policy and neural critic parametrization. In Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence, volume 286 of Proceedings of Machine Learning Research, pages 1358–1380. PMLR, 2025.

[11] Swetha Ganesh, Washim Uddin Mondal, and Vaneet Aggarwal. A sharper global convergence analysis for average reward reinforcement learning via an actor-critic approach. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 18206–18227. PMLR, 2025.

[12] Wenbo Gao, Donald Goldfarb, and Frank E. Curtis. ADMM for multiafine constrained optimization. Optimization Methods and Software, 35(2):257–303, 2020.

[13] Mudit Gaur, Amrit Bedi, Di Wang, and Vaneet Aggarwal. Closing the gap: Achieving global convergence (last iterate) of actor-critic under markovian sampling with neural network parametrization. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 15153–15179. PMLR, 2024.

[14] Shaan Ul Haque and Siva Theja Maguluri. Stochastic approximation with unbounded markovian noise: A general-purpose theorem. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3718–3726. PMLR, 2025.

[15] Bingsheng He and Xiaoming Yuan. On the o(1/n) convergence rate of the douglas–rachford alternating direction method. SIAM Journal on Numerical Analysis, 50(2):700–709, 2012.

[16] Mingyi Hong, Zhi-Quan Luo, and Meisam Razaviyayn. Convergence analysis of alternating direction method of multipliers for a family of nonconvex problems. SIAM Journal on Optimization, 26(1):337–364, 2016.

[17] Audrey Huang and Nan Jiang. Occupancy-based policy gradient: Estimation, convergence, and optimality. In Advances in Neural Information Processing Systems, volume 37, 2024.

[18] Jiachen Jin, Kangkang Deng, Boyu Wang, and Hongxia Wang. Stochastic admm with batch size adaptation for nonconvex nonsmooth optimization. Applied Numerical Mathematics, 222:87– 107, 2026.

[19] Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In International Conference on Machine Learning, 2002.

[20] Guoyin Li and Ting Kei Pong. Douglas–rachford splitting for nonconvex optimization with application to nonconvex feasibility problems. Mathematical Programming, 159:371–401, 2016.

[21] Longhui Liu, Congying Han, Tiande Guo, and Shichen Liao. An inertial stochastic bregman generalized alternating direction method of multipliers for nonconvex and nonsmooth optimization. Expert Systems with Applications, 276:126939, 2025.

[22] Alessandro Montenegro, Marco Mussi, Matteo Papini, and Alberto Maria Metelli. Last-iterate global convergence of policy gradients for constrained reinforcement learning. In Advances in Neural Information Processing Systems, volume 37, 2024.

[23] Vrettos Moulos. Optimal best markovian arm identification with fixed confidence. Advances in Neural Information Processing Systems, 32, 2019.

[24] Gergely Neu and Nneka Okolo. Ofline rl via feature-occupancy gradient ascent. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3637–3645. PMLR, 2025.

[25] Bhrij Patel, Wesley A. Suttle, Alec Koppel, Vaneet Aggarwal, Brian M. Sadler, Dinesh Manocha, and Amrit Bedi. Towards global optimality for practical average reward reinforcement learning without mixing time oracles. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 39889– 39907. PMLR, 2024.

[26] Zhaojun Peng. Variance-reduced conditional gradient methods under markovian sampling for nonconvex composite optimization, 2026.

[27] Anirudh Satheesh, Pankaj Kumar Barman, Washim Uddin Mondal, and Vaneet Aggarwal. Global convergence of average reward constrained mdps with neural critic and general policy parameterization. In Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence, volume 337 of Proceedings of Machine Learning Research, pages 5997–6025. PMLR, 2026.

[28] Uri Sherman, Alon Cohen, Tomer Koren, and Yishay Mansour. Rate-optimal policy optimization for linear markov decision processes. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 44815– 44837. PMLR, 2024.

[29] Uri Sherman, Tomer Koren, and Yishay Mansour. Convergence of policy mirror descent beyond compatible function approximation. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 54825– 54863. PMLR, 2025.

[30] Zhenyu Sun and Ermin Wei. Improved lower bounds for first-order stochastic non-convex optimization under markov sampling. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 57754– 57772. PMLR, 2025.

[31] Yudan Wang, Yue Wang, Yi Zhou, and Shaofeng Zou. Non-asymptotic analysis for singleloop (natural) actor-critic with compatible function approximation. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 51771–51824. PMLR, 2024.

[32] Ganzhao Yuan. Smoothing proximal gradient methods for nonsmooth sparsity constrained optimization: Optimality conditions and global convergence. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 57842–57870. PMLR, 2024.

[33] Ganzhao Yuan. Admm for nonconvex optimization under minimal continuity assumption. In International Conference on Learning Representations, 2025.

[34] Jinshan Zeng, Wotao Yin, and Ding-Xuan Zhou. Moreau envelope augmented lagrangian method for nonconvex optimization with linear constraints. Journal of Scientific Computing, 91(2):61, 2022.

[35] Yuxuan Zeng, Jianchao Bai, Shengjia Wang, Zhiguo Wang, and Xiaojing Shen. A hybrid stochastic alternating direction method of multipliers for nonconvex and nonsmooth composite optimization. European Journal of Operational Research, 329(1):63–78, 2026.

[36] Yuxuan Zeng, Zhiguo Wang, Jianchao Bai, and Xiaojing Shen. An accelerated stochastic admm for nonconvex and nonsmooth finite-sum optimization. Automatica, 163:111554, 2024.

## A Proof Map, Conventions, and a Reusable Resolvent Lemma

The appendix follows the same proof organization used in the reference ADMM analysis: the main text states the results and points to an appendix chain, while the appendix proves the lemmas in the order in which they are consumed by the convergence argument. In particular, the logical flow is

$$
{ \mathrm { e m p i r i c a l - m o d e l ~ c o n t r o l } } \Longrightarrow { \mathrm { i t e r a t e ~ s t a b i l i t y } } \Longrightarrow { \mathrm { d u a l ~ c o n t r o l } }
$$

=⇒ Lyapunov descent =⇒ KKT convergence and complexity =⇒ policy performance.

The extensions repeat the same architecture after replacing the Bellman operator by a nonlinearpolicy Bellman operator, a projected Bellman matrix, or an occupancy operator.

We use $\Delta \pi _ { k + 1 } = \pi ^ { k + 1 } - \pi ^ { k } , \Delta V _ { k + 1 } = V ^ { k + 1 } - V ^ { k }$ , and $\Delta \lambda _ { k + 1 } = \lambda ^ { k + 1 } - \lambda ^ { k }$ . We set $\varepsilon _ { - 1 } = 0$ and $m _ { - 1 } ^ { - 1 } = 0$ . The initialization satisfies $\pi ^ { 0 } \in \operatorname { d o m } \psi$ and has the moments in Assumption 3.3. Es timates obtained by subtracting consecutive value-dual identities, and their Lyapunov and residual consequences, are stated for $k \geq 1$ . The first update contributes a finite initialization constant to finite-time sums. The deterministic extensions use the same convention. All vector norms are Euclidean unless stated otherwise, and matrix norms are induced spectral norms. Constants denoted by C may change from line to line but never depend on k, T, or the batch schedule.

Dependency map. The statistical Lemmas B.1–B.3 prove Lemma 4.1 and supply the stochastic error used by Theorems 4.7 and 4.8. Lemma A.1 and Lemma 4.2 are used in Lemma 4.3. Lemmas 4.4 and 4.3 are the two inputs to Lemma 4.6. Lemmas 4.6 and 5.1 imply Theorem 5.2. The policyperformance chain uses Lemmas D.1–D.6 before Theorems 6.2 and 6.8. Every extension result is used either to prove the next result in its subsection or to justify the operator-stability extension stated in Section 7.

Lemma A.1 (Inverse-map stability). Let M and M<sup>′</sup> be invertible with $\lVert M ^ { - 1 } \rVert \leq \kappa$ and $\| M ^ { \prime - 1 } \| \leq \kappa$ Then

$$
\lVert M ^ { \prime - 1 } - M ^ { - 1 } \rVert \leq \kappa ^ { 2 } \lVert M ^ { \prime } - M \rVert .
$$

Consequently, $i f \| M _ { \pi ^ { \prime } } - M _ { \pi } \| \leq L _ { M } \| \pi ^ { \prime } - \pi \|$ , then

$$
\| M _ { \pi ^ { \prime } } ^ { - 1 } - M _ { \pi } ^ { - 1 } \| \le \kappa ^ { 2 } L _ { M } \| \pi ^ { \prime } - \pi \| .
$$

Proof. The resolvent identity gives

$$
M ^ { \prime - 1 } - M ^ { - 1 } = M ^ { \prime - 1 } ( M - M ^ { \prime } ) M ^ { - 1 } .
$$

Taking operator norms yields

$$
\begin{array} { r } { \| M ^ { \prime - 1 } - M ^ { - 1 } \| \leq \| M ^ { \prime - 1 } \| \| M ^ { \prime } - M \| \| M ^ { - 1 } \| \leq \kappa ^ { 2 } \| M ^ { \prime } - M \| . } \end{array}
$$

The policy-dependent bound follows by substituting the Lipschitz estimate for $M _ { \pi }$

Role. Lemma A.1 is the algebraic step that converts a primal policy displacement into a change of the inverse Bellman operator. It is used in Lemma 4.3, in the nonlinear-policy extension, and in the empirical occupancy inverse perturbation bound.

## B Controlled-Markov Empirical-Model Bounds

Fix an outer iteration k and condition on $\mathcal { F } _ { k }$ . During the batch the behavior policy is frozen, so the reset chain is time homogeneous. Let $B _ { t } = 0$ denote a reset and $B _ { t } = 1$ a genuine environment transition. For a state-action pair $( s , a )$ define

$$
Y _ { t } ^ { s , a } = \mathbf { 1 } \{ S _ { t } = s , A _ { t } = a , B _ { t } = 1 \} , \qquad N _ { s , a } ^ { ( k ) } = \sum _ { t = 1 } ^ { m _ { k } } Y _ { t } ^ { s , a } .
$$

The quantity $N _ { s , a } ^ { ( k ) }$ is the denominator of the empirical conditional transition and reward estimators.

Lemma B.1 (Uniform efective-visitation lower tail). Under Assumption 3.2, there are constants $p _ { 0 } , c _ { 0 } , C _ { 0 } > 0$ , independent of k and $m _ { k }$ , such that

$$
\mathbb { P } \Big ( N _ { s , a } ^ { ( k ) } < p _ { 0 } m _ { k } \ | \ \mathcal { F } _ { k } \Big ) \leq C _ { 0 } e ^ { - c _ { 0 } m _ { k } }
$$

for every state-action pair $( s , a )$

Proof. Split the batch into $J = \lfloor m _ { k } / 2 \rfloor$ nonoverlapping two-step blocks and let

$$
Q _ { j } = { \bf 1 } \{ B _ { 2 j - 1 } = 0 , ~ S _ { 2 j } = s , ~ A _ { 2 j } = a , ~ B _ { 2 j } = 1 \} .
$$

If $Q _ { j } = 1$ , then the second step of the block contributes one usable $( s , a )$ transition. Hence $N _ { s , a } ^ { ( k ) } \geq$ $\textstyle \sum _ { j = 1 } ^ { J } Q _ { j }$ . Given the history before block $j ,$ , the reset draws a fresh state from $\rho ,$ the frozen behavior policy chooses action $^ { a , }$ and the next reset coin equals one. Therefore

$$
\begin{array} { r } { \mathbb { E } [ Q _ { j } \mid \mathcal { H } _ { j - 1 } ] = ( 1 - \gamma ) \rho ( s ) \bar { \pi } ^ { k } ( a \mid s ) \gamma \geq q _ { \star } : = \gamma ( 1 - \gamma ) \rho _ { \operatorname* { m i n } } \alpha > 0 . } \end{array}
$$

Set $D _ { j } \ = \ Q _ { j } \ - \ \mathbb { E } [ Q _ { j } \ | \ \mathcal { H } _ { j - 1 } ]$ . Then $( D _ { j } )$ is a martingale-diference sequence and $| D _ { j } | \le 1$ . If $\textstyle \sum _ { j } Q _ { j } < q _ { \star } J / 2$ , then $\begin{array} { r } { \sum _ { i } D _ { j } \le - q _ { \star } J / 2 } \end{array}$ . Azuma–Hoefding gives

$$
\mathbb { P } \left( \sum _ { j = 1 } ^ { J } Q _ { j } < \frac { q _ { \star } J } { 2 } \Bigg | { \mathcal F } _ { k } \right) \le \exp \left( - \frac { q _ { \star } ^ { 2 } J } { 8 } \right) .
$$

Since $J \ge m _ { k } / 3$ for all suficiently large $m _ { k }$ , the desired inequality follows with $p _ { 0 } = q _ { \star } / 6$ . The finitely many smaller batch sizes are absorbed by increasing $C _ { 0 }$ □

Role. Lemma B.1 prevents the random denominators of all state-action conditional estimators from becoming too small. It is used in Lemma B.2 and the proof of Lemma 4.1.

Lemma B.2 (Inverse moments of random visit counts). For every fixed $q \geq 1$ there exists $C _ { q } < \infty$ such that

$$
\mathbb { E } \Big [ ( N _ { s , a } ^ { ( k ) } ) ^ { - q } \mathbf { 1 } \{ N _ { s , a } ^ { ( k ) } > 0 \} \mid \mathcal { F } _ { k } \Big ] \leq \frac { C _ { q } } { m _ { k } ^ { q } } .
$$

Proof. Write $N = N _ { s , a } ^ { ( k ) }$ and split the expectation over $\{ N \ge p _ { 0 } m _ { k } \}$ and $\{ 1 \leq N < p _ { 0 } m _ { k } \}$ . On the first event, $N ^ { - q } \leq ( p _ { 0 } m _ { k } ) ^ { - q }$ . On the second event, $N ^ { - q } \leq 1$ and Lemma B.1 gives probability at most $C _ { 0 } e ^ { - c _ { 0 } m _ { k } }$ . Therefore

$$
{ \mathbb E } [ N ^ { - q } \mathbf { 1 } \{ N > 0 \} ~ \vert ~ \mathcal { F } _ { k } ] \le ( p _ { 0 } m _ { k } ) ^ { - q } + C _ { 0 } e ^ { - c _ { 0 } m _ { k } } .
$$

For fixed $q , e ^ { - c _ { 0 } m } \leq C _ { q } ^ { \prime } m ^ { - q }$ for all integers $m \geq 1$ . Combining the two terms proves the claim.

Role. Lemma B.2 is used to convert martingale moment bounds with a random sample count into deterministic orders $m _ { k } ^ { - 1 }$ and $m _ { k } ^ { - 2 }$

Proof of Lemma 4.1. We prove the transition and reward parts separately.

Part (a): transition rows. For fixed $( s , a )$ define

$$
Z _ { t } ^ { s , a } = Y _ { t } ^ { s , a } \bigl ( e _ { S _ { t + 1 } } - P ( \cdot \mid s , a ) \bigr ) .
$$

Because only non-reset transitions are used, $Y _ { t } ^ { s , a }$ is measurable before $S _ { t + 1 }$ is generated and, on $Y _ { t } ^ { s , a } = 1$ , the next state is drawn from the true environment row. Hence $\mathbb { E } [ Z _ { t } ^ { s , a } \ | \ \mathcal { H } _ { t } ] = 0$ . The increments are bounded. The standard Burkholder–Rosenthal fourth-moment inequality for finitedimensional martingales yields

$$
\mathbb { E } \left[ \left. \sum _ { t = 1 } ^ { m _ { k } } Z _ { t } ^ { s , a } \right. ^ { 4 } \middle | \mathcal { F } _ { k } \right] \leq C _ { Z } m _ { k } ^ { 2 } .
$$

On the good-count event $\mathcal { E } _ { s , a } ^ { ( k ) } = \{ N _ { s , a } ^ { ( k ) } \geq p _ { 0 } m _ { k } \}$

$$
\widehat { P } _ { k } ( \cdot \mid s , a ) - P ( \cdot \mid s , a ) = \frac { 1 } { N _ { s , a } ^ { ( k ) } } \sum _ { t = 1 } ^ { m _ { k } } Z _ { t } ^ { s , a } ,
$$

so

$$
\mathbb { E } \Big [ \| \widehat { P } _ { k } ( \cdot \mid s , a ) - P ( \cdot \mid s , a ) \| ^ { 4 } \mathbf { 1 } _ { \mathscr { E } _ { s , a } ^ { ( k ) } } \Big | \mathscr { F } _ { k } \Big ] \leq \frac { C } { m _ { k } ^ { 2 } } .
$$

On the bad-count event two probability rows are uniformly bounded in norm, while Lemma B.1 gives an exponentially small probability. Thus the bad-event contribution is also $O ( m _ { k } ^ { - 2 } )$ . Consequently

$$
\mathbb { E } \Big [ \| \widehat { P } _ { k } ( \cdot \mid s , a ) - P ( \cdot \mid s , a ) \| ^ { 4 } \mid \mathcal { F } _ { k } \Big ] \leq \frac { C _ { P } } { m _ { k } ^ { 2 } } .
$$

Part (b): reward rows. Write $\begin{array} { r } { m = m _ { k } , I _ { t } = Y _ { t } ^ { s , a } , N = \sum _ { t } I _ { t } } \end{array}$ , and $\begin{array} { r } { S _ { m } = \sum _ { t = 1 } ^ { m } I _ { t } \zeta _ { k , t } . } \end{array}$ The variables $I _ { t }$ are predictable for the pre-observation filtration $\mathcal { G } _ { k , t }$ . Thus $I _ { t } \zeta _ { k , t }$ is a martingale diference. Work throughout conditionally on $\mathcal { F } _ { k }$ . The martingale fourth-moment inequality gives

$$
\mathbb { E } _ { k } | S _ { m } | ^ { 4 } \leq C M _ { 4 } m ^ { 2 } .
$$

Let $p _ { \star }$ be the visitation constant from Lemma B.1 and set $p = { p _ { \star } } / 4 .$ . On $\{ N \ge p m \}$ ,

$$
\begin{array} { r } { \mathbb { E } _ { k } \left[ | S _ { m } / N | ^ { 4 } \mathbf { 1 } \{ N \ge p m \} \right] \le ( p m ) ^ { - 4 } \mathbb { E } _ { k } | S _ { m } | ^ { 4 } \le C M _ { 4 } m ^ { - 2 } . } \end{array}
$$

To control small counts without conditioning on the full trajectory, split the batch into two contiguous halves of lengths $\lfloor m / 2 \rfloor$ and $\lceil m / 2 \rceil$ . Let $N _ { 1 } , N _ { 2 }$ be their usable visit counts and let H contain all observations in the first half. For $m \geq 4$ , each half has length at least $m / 3$ , so Lemma B.1, applied from any starting history, gives

$$
\begin{array} { r } { \mathbb { P } _ { k } \left( N _ { 1 } < p m \right) \le C e ^ { - c m } , \qquad \mathbb { P } \left( N _ { 2 } < p m \mid \mathcal { H } \right) \le C e ^ { - c m } . } \end{array}
$$

On $B = \{ 1 \leq N < p m \}$ , Jensen’s inequality gives

$$
| S _ { m } / N | ^ { 4 } \leq N ^ { - 1 } \sum _ { t } I _ { t } | \zeta _ { k , t } | ^ { 4 } \leq \sum _ { t } I _ { t } | \zeta _ { k , t } | ^ { 4 } .
$$

For a first-half index $t , I _ { t } | \zeta _ { k , t } | ^ { 4 }$ is H-measurable and $B \subset \{ N _ { 2 } < p m \}$ . Hence

$$
\begin{array} { r } { \mathbb { E } _ { k } [ I _ { t } | \zeta _ { k , t } | ^ { 4 } \mathbf { 1 } _ { \mathcal { B } } ] \le C e ^ { - c m } \mathbb { E } _ { k } [ I _ { t } | \zeta _ { k , t } | ^ { 4 } ] \le C M _ { 4 } e ^ { - c m } . } \end{array}
$$

For a second-half index $t ,$ the event $\{ N _ { 1 } ~ < ~ p m \}$ belongs to $\mathcal { G } _ { k , t }$ . Since $B \subset \{ N _ { 1 } \ < \ p m \}$ , the conditional moment assumption gives the same bound. Summing over t yields

$$
\begin{array} { r } { \mathbb { E } _ { k } [ | S _ { m } / N | ^ { 4 } \mathbf { 1 } _ { B } ] \le C M _ { 4 } m e ^ { - c m } \le C M _ { 4 } m ^ { - 2 } . } \end{array}
$$

For $N = 0$ , the bounded fallback reward contributes $C e ^ { - c m }$ . The finitely many $m < 4$ are absorbed in the constant. Therefore $\mathbb { E } _ { k } | \widehat { r } _ { k } ( s , a ) - r ( s , a ) | ^ { 4 } \leq C / m _ { k } ^ { 2 }$ . This argument allows the reward and next state at a given transition to be dependent.

Part (c): policy-uniform model error. Because the state-action space is finite and $P _ { \pi } , r _ { \pi }$ are convex combinations of primitive rows,

$$
\varepsilon _ { k } \leq C _ { \Theta } \left( \operatorname* { m a x } _ { s , a } \| \widehat { P } _ { k } ( \cdot \mid s , a ) - P ( \cdot \mid s , a ) \| + \operatorname* { m a x } _ { s , a } \left. \widehat { r } _ { k } ( s , a ) - r ( s , a ) \right. \right) .
$$

Finite-dimensional norm equivalence and the bounds above give $\mathbb { E } [ \varepsilon _ { k } ^ { 4 } \mid { \mathcal { F } } _ { k } ] \leq C _ { 4 } / m _ { k } ^ { 2 }$ . Conditional Jensen then yields $\mathbb { E } [ \varepsilon _ { k } ^ { 2 } \mid \mathcal { F } _ { k } ] \le ( \mathbb { E } [ \varepsilon _ { k } ^ { 4 } \mid \mathcal { F } _ { k } ] ) ^ { 1 / 2 } \le C _ { 2 } / m _ { k }$ □

Role. Lemma 4.1 supplies the deterministic $O ( m _ { k } ^ { - 1 } )$ error terms in the expected Lyapunov and finite-time complexity arguments. Its fourth-moment part is exactly what permits Cauchy–Schwarz with the random iterates.

Lemma B.3 (General stochastic empirical-model error). Suppose an additional observation perturbation is decomposed as $\xi _ { k , t } = b _ { k , t } + \zeta _ { k , t }$ with $\begin{array} { r } { \mathbb { E } [ \zeta _ { k , t } \ | \ \mathcal { G } _ { k , t } ] = 0 , \ \mathbb { E } [ \| \zeta _ { k , t } \| ^ { 2 } \ | \ \mathcal { G } _ { k , t } ] \leq \sigma _ { k } ^ { 2 } . } \end{array}$ , and the average bias in every visited state-action reward row bounded in magnitude by $b _ { k }$ . Then there are deterministic constants $C _ { 0 } , C _ { \sigma } , C _ { b }$ such that

$$
\mathbb { E } [ \varepsilon _ { k } ^ { 2 } \mid \mathcal { F } _ { k } ] \leq \frac { C _ { 0 } + C _ { \sigma } \sigma _ { k } ^ { 2 } } { m _ { k } } + C _ { b } b _ { k } ^ { 2 } .
$$

Consequently,

$$
\sum _ { k } \frac { 1 + \sigma _ { k } ^ { 2 } } { m _ { k } } < \infty , ~ \sum _ { k } b _ { k } ^ { 2 } < \infty ~ \Longrightarrow ~ \sum _ { k } \varepsilon _ { k } ^ { 2 } < \infty ~ a l m o s t ~ s u r e l y .
$$

Proof. The transition part contributes $C / m _ { k }$ . For each reward row, use the predictable martin gale sum $\begin{array} { r } { S _ { m } = \sum _ { t = 1 } ^ { m } I _ { t } \zeta _ { k , t } } \end{array}$ from Part (b) above. Orthogonality of martingale diferences gives $\mathbb { E } _ { k } | S _ { m } | ^ { 2 } \ \le \ m \sigma _ { k } ^ { 2 }$ . On N ≥ pm the squared average therefore contributes at most $C \sigma _ { k } ^ { 2 } / m$ . On $1 \leq N < p m$ , repeat the two-half argument with exponent two. The first-half terms are controlled by the conditional lower-tail bound for the second half. The second-half terms are controlled by $\mathbb { E } [ | \zeta _ { k , t } | ^ { 2 } \mid \mathcal { G } _ { k , t } ] \le \sigma _ { k } ^ { 2 }$ and the first-half lower-tail event. Their total is at most $C \sigma _ { k } ^ { 2 } m e ^ { - c m } \leq C \sigma _ { k } ^ { 2 } / m$ The bounded zero-count fallback contributes $C / m$ . Adding the squared row-average bias and summing over the finite row set proves the bound.

Taking expectations and summing under the stated schedule gives $\textstyle \sum _ { k } \mathbb { E } \varepsilon _ { k } ^ { 2 } < \infty$ . Since the summands are nonnegative, Tonelli’s theorem implies $\textstyle \sum _ { k } \varepsilon _ { k } ^ { 2 } < \infty$ almost surely. □

Role. Lemma B.3 is the statistical input to Theorem 4.8. The summability condition controls both observation noise and row-average bias.

## C Bellman-Resolvent Stability, Lyapunov Descent, and KKT Complexity

## A priori iterate stability

Proof of Lemma $4 . 2 .$ The empirical value first-order condition and the dual update give the exact identity

$$
\lambda ^ { k + 1 } = - \widehat { M } _ { k + 1 } ^ { - \mathsf { T } } \left( c + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } \right) .\tag{8}
$$

Hence

$$
\| \lambda ^ { k + 1 } \| \leq \bar { \kappa } \| c \| + \frac { \bar { \kappa } } { \eta _ { V } } \left( \| V ^ { k + 1 } \| + \| V ^ { k } \| \right) .\tag{9}
$$

The dual update also gives

$$
\widehat { M } _ { k + 1 } V ^ { k + 1 } = \widehat { r } _ { \pi ^ { k + 1 } , k } + \beta ^ { - 1 } \Delta \lambda _ { k + 1 } ,
$$

so

$$
\| V ^ { k + 1 } \| \leq \bar { \kappa } \| \widehat { r } _ { \pi ^ { k + 1 } , k } \| + \frac { \bar { \kappa } } { \beta } \left( \| \lambda ^ { k + 1 } \| + \| \lambda ^ { k } \| \right) .\tag{10}
$$

Lemma 4.1, bounded true rewards, and the conditional fourth-moment assumption imply $R _ { 4 } : =$ $\mathrm { s u p } _ { k } ( \mathbb { E } \| \widehat { r } _ { \pi ^ { k + 1 } , k } \| ^ { 4 } ) ^ { 1 / 4 } < \infty$ . Set

$$
X _ { k } = ( \mathbb { E } \| V ^ { k } \| ^ { 4 } ) ^ { 1 / 4 } , \quad Y _ { k } = ( \mathbb { E } \| \lambda ^ { k } \| ^ { 4 } ) ^ { 1 / 4 } , \quad a = \frac { \bar { \kappa } } { \eta _ { V } } , \quad b = \frac { \bar { \kappa } } { \beta } .
$$

Minkowski’s inequality applied to (9)–(10) gives

$$
\begin{array} { r } { Y _ { k + 1 } \leq \bar { \kappa } \| c \| + a ( X _ { k + 1 } + X _ { k } ) , \qquad X _ { k + 1 } \leq \bar { \kappa } R _ { 4 } + b ( Y _ { k + 1 } + Y _ { k } ) . } \end{array}
$$

Eliminating the next-iterate variables yields

$$
\left[ \begin{array} { c } { X _ { k + 1 } } \\ { Y _ { k + 1 } } \end{array} \right] \leq \mathsf { A } \left[ \begin{array} { c } { X _ { k } } \\ { Y _ { k } } \end{array} \right] + d , \qquad \mathsf { A } = \frac { 1 } { 1 - a b } \left[ \begin{array} { c c } { a b } & { b } \\ { a } & { a b } \end{array} \right] ,
$$

where d is a fixed nonnegative vector. Let $z = \sqrt { a b } = \bar { \kappa } / \sqrt { \beta \eta _ { V } }$ . The largest eigenvalue of A is

$$
\rho ( \mathsf { A } ) = \frac { a b + \sqrt { a b } } { 1 - a b } = \frac { z } { 1 - z } .
$$

Assumption 3.3 implies $\beta \eta _ { V } > 4 \bar { \kappa } ^ { 2 }$ , hence $z < 1 / 2$ and $\rho ( \mathsf { A } ) < 1$ . Iterating the recursion gives

$$
{ \Big [ } X _ { k } { \Big ] } \leq \mathsf { A } ^ { k } { \Big [ } X _ { 0 } { \Big ] } + \sum _ { j = 0 } ^ { k - 1 } \mathsf { A } ^ { j } d .
$$

The matrix series converges. Thus $\operatorname* { s u p } _ { k } X _ { k } < \infty$ and $\operatorname* { s u p } _ { k } Y _ { k } < \infty$ , which is the claimed fourthmoment stability. □

Role. This lemma is proved before any Lyapunov decrease. It removes the potential circularity that would arise if bounded multipliers were assumed inside the empirical-to-true perturbation estimate.

## Multiplier control

Proof of Lemma $4 . 3 .$ . Let $M _ { k + 1 } = M _ { \pi ^ { k + 1 } }$ . Replacing $\widehat { M } _ { k + 1 }$ in the value-dual identity by the true matrix gives

$$
c + M _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k + 1 } + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } = r _ { k + 1 } , \qquad r _ { k + 1 } = \left( M _ { k + 1 } - { \widehat M } _ { k + 1 } \right) ^ { \mathsf { T } } \lambda ^ { k + 1 } .\tag{11}
$$

Since $\| M _ { k + 1 } - \widehat { M } _ { k + 1 } \| \leq C \varepsilon _ { k }$ , conditional Cauchy–Schwarz, Lemma 4.1, and Lemma 4.2 give

$$
\mathbb { E } \| r _ { k + 1 } \| ^ { 2 } \le C \mathbb { E } [ \varepsilon _ { k } ^ { 2 } \| \lambda ^ { k + 1 } \| ^ { 2 } ]\tag{12}
$$

$$
\leq C ( \mathbb { E } \varepsilon _ { k } ^ { 4 } ) ^ { 1 / 2 } ( \mathbb { E } \| \lambda ^ { k + 1 } \| ^ { 4 } ) ^ { 1 / 2 } \leq \frac { C _ { r } } { m _ { k } } .\tag{13}
$$

Solve (11) for $\lambda ^ { k + 1 }$ and subtract the corresponding identity at iteration k:

$$
\begin{array} { r l } & { \Delta \lambda _ { k + 1 } = \ - \ ( M _ { k + 1 } ^ { - \mathsf { T } } - M _ { k } ^ { - \mathsf { T } } ) c - \eta _ { V } ^ { - 1 } M _ { k + 1 } ^ { - \mathsf { T } } \Delta V _ { k + 1 } + \eta _ { V } ^ { - 1 } M _ { k } ^ { - \mathsf { T } } \Delta V _ { k } } \\ & { \qquad + \ M _ { k + 1 } ^ { - \mathsf { T } } r _ { k + 1 } - M _ { k } ^ { - \mathsf { T } } r _ { k } . } \end{array}\tag{14}
$$

Lemma A.1 gives $\| ( M _ { k + 1 } ^ { - \mathsf { T } } - M _ { k } ^ { - \mathsf { T } } ) c \| \leq \bar { \kappa } ^ { 2 } L _ { M } \| c \| \| \Delta \pi _ { k + 1 } \|$ . Using $\begin{array} { r } { \| \sum _ { i = 1 } ^ { 5 } x _ { i } \| ^ { 2 } \leq 5 \sum _ { i = 1 } ^ { 5 } \| x _ { i } \| ^ { 2 } } \end{array}$ in (14),

$$
\begin{array} { l } { \displaystyle | | \Delta \lambda _ { k + 1 } | | ^ { 2 } \leq 5 \bar { \kappa } ^ { 4 } L _ { M } ^ { 2 } \| c \| ^ { 2 } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + \displaystyle \frac { 5 \bar { \kappa } ^ { 2 } } { \eta _ { V } ^ { 2 } } \left( \| \Delta V _ { k + 1 } \| ^ { 2 } + \| \Delta V _ { k } \| ^ { 2 } \right) } \\ { \displaystyle \qquad + 5 \bar { \kappa } ^ { 2 } ( \| r _ { k + 1 } \| ^ { 2 } + \| r _ { k } \| ^ { 2 } ) . } \end{array}\tag{15}
$$

Taking expectations and using (13) proves (5) with the stated $C _ { \pi }$ and $C _ { V }$ and a suitable $C _ { \varepsilon }$

Role. Lemma 4.3 is the replacement for the classical smooth-block multiplier estimate. It is the term that allows the positive primal decrease to dominate the dual ascent.

## Structure-preserving surrogate closure

Proof of Lemma $4 { \cdot } 4 .$ Fix a sample path for which $\textstyle \sum _ { k } \varepsilon _ { k } ^ { 2 } < \infty$ and the a priori iterate bounds hold. The empirical and true residuals satisfy

$$
\begin{array} { r } { e _ { k } ( \pi , V ) = \widehat { R } _ { k } ( \pi , V ) - R ( \pi , V ) = ( \widehat { M } _ { \pi , k } - M _ { \pi } ) V - ( \widehat { r } _ { \pi , k } - r _ { \pi } ) . } \end{array}
$$

On the bounded iterate set there is $C _ { e } < \infty$ such that

$$
\| e _ { k } ( \pi , V ) \| + \| D _ { \pi , V } e _ { k } ( \pi , V ) \| \le C _ { e } \varepsilon _ { k } .\tag{16}
$$

We verify the four relations in the statement.

Part (a): true primal decrease. Let $\Psi _ { k } = \widehat { \mathcal { L } } _ { \beta , k } - \mathcal { L } _ { \beta }$ . Expanding,

$$
\Psi _ { k } = \langle \lambda , e _ { k } \rangle + \beta \langle R , e _ { k } \rangle + \frac { \beta } { 2 } \| e _ { k } \| ^ { 2 } .
$$

Equation (16) and bounded iterates imply $\| D _ { \pi , V } \Psi _ { k } \| \le C \varepsilon _ { k }$ along the two primal line segments. Optimality of the empirical policy subproblem gives

$$
\widehat { \mathcal { L } } _ { \beta , k } ( \pi ^ { k } , V ^ { k } , \lambda ^ { k } ) - \widehat { \mathcal { L } } _ { \beta , k } ( \pi ^ { k + 1 } , V ^ { k } , \lambda ^ { k } ) \geq \frac { 1 } { 2 \eta _ { \pi } } \| \Delta \pi _ { k + 1 } \| ^ { 2 } .
$$

The perturbation diference is at most $C \varepsilon _ { k } \| \Delta \pi _ { k + 1 } \| .$ , which by Young’s inequality is bounded by $( 4 \eta _ { \pi } ) ^ { - 1 } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + C \varepsilon _ { k } ^ { 2 }$ . Thus the true policy step decreases by at least $( 4 \eta _ { \pi } ) ^ { - 1 } \| \Delta \pi _ { k + 1 } \| ^ { 2 } - C \varepsilon _ { k } ^ { 2 } .$ The same argument for the value step gives $( 4 \eta _ { V } ) ^ { - 1 } \| \Delta V _ { k + 1 } \| ^ { 2 } - C \varepsilon _ { k } ^ { 2 }$ . Adding the two proves the first relation with $a _ { \pi } = 1 / ( 4 \eta _ { \pi } )$ and $a _ { V } = 1 / ( 4 \eta _ { V } )$

Part (b): perturbed value-dual identity. The exact empirical identity is the value-dual identity. Hence

$$
c + M _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k + 1 } + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } = ( M _ { k + 1 } - \widehat M _ { k + 1 } ) ^ { \mathsf { T } } \lambda ^ { k + 1 } = : r _ { k + 1 } .
$$

The pathwise iterate bound and (16) give $\| r _ { k + 1 } \| \le C \varepsilon _ { k }$

Part (c): inexact true-residual multiplier update. Since $\Delta \lambda _ { k + 1 } = \beta \widehat { R } _ { k } ( \pi ^ { k + 1 } , V ^ { k + 1 } )$ 1，

$$
\Delta \lambda _ { k + 1 } = \beta R _ { k + 1 } + d _ { k + 1 } , \qquad d _ { k + 1 } = \beta e _ { k } ( \pi ^ { k + 1 } , V ^ { k + 1 } ) ,
$$

and $\| d _ { k + 1 } \| \le C \varepsilon _ { k }$ by (16).

Part (d): true KKT residual. The empirical policy Fermat condition is

$$
0 \in \partial \psi ( \pi ^ { k + 1 } ) + \widehat { J } _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) ^ { \mathsf { T } } \big [ \lambda ^ { k } + \beta \widehat { R } _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) \big ] + \eta _ { \pi } ^ { - 1 } \Delta \pi _ { k + 1 } .
$$

The bracket difers from $\lambda ^ { k + 1 }$ by $\beta \widehat { M } _ { k + 1 } ( V ^ { k } - V ^ { k + 1 } )$ , and the true and empirical Jacobians difer by $O ( \varepsilon _ { k } )$ . The tabular Bellman Jacobian is afine in $V ,$ , so replacing $V ^ { k }$ by $V ^ { k + 1 }$ costs $O ( \| \Delta V _ { k + 1 } \| )$ Hence the true policy residual is at most $C ( \| \Delta \pi _ { k + 1 } \| + \| \Delta V _ { k + 1 } \| + \varepsilon _ { k } )$ . The value residual follows from Part (b), and feasibility follows from Part (c). Combining the three components gives the stated true KKT bound. □

Role. This lemma is the interface between stochastic RL data and deterministic primal-dual analysis. It packages all sampling efects into one scalar operator error $\varepsilon _ { k }$

## Expected Lyapunov descent and lower bound

Proof of Lemma $4 . 6 .$ We prove the descent and lower bound separately.

Part (a): expected primal decrease. The empirical policy and value subproblems give the same comparison inequalities as above. To justify the moment transfer, define

$$
U _ { k } = 1 + \| V ^ { k } \| + \| \boldsymbol { \lambda } ^ { k } \| , \quad p _ { k } = \operatorname* { m a x } _ { s , a } \| \widehat { P } _ { k } ( \cdot \vert s , a ) - P ( \cdot \vert s , a ) \| , \quad u _ { k } = \operatorname* { m a x } _ { s , a } | \widehat { r } _ { k } ( s , a ) - r ( s , a ) | .
$$

Then $U _ { k }$ is $\mathcal { F } _ { k }$ -measurable, $p _ { k }$ is uniformly bounded, and Lemma 4.1 gives conditional second and fourth moments of orders $m _ { k } ^ { - 1 }$ and $m _ { k } ^ { - 2 }$ for both $p _ { k }$ and $u _ { k }$ . Expanding $\Psi _ { k } = \widehat { \mathcal { L } } _ { \beta , k } - \mathcal { L } _ { \beta }$ on the policy comparison segment, where ${ \boldsymbol { V } } = { \boldsymbol { V } } ^ { k }$ and $\lambda = \lambda ^ { k }$ , gives

$$
\operatorname* { s u p } _ { \pi \in \Pi } \| D _ { \pi } \Psi _ { k } ( \pi , V ^ { k } , \lambda ^ { k } ) \| \leq C \left( p _ { k } U _ { k } ^ { 2 } + u _ { k } U _ { k } + u _ { k } ^ { 2 } \right) .
$$

Indeed, the Jacobian error is at most $C ( p _ { k } \| V ^ { k } \| + u _ { k } )$ and the residual error has the same bound. Boundedness of $p _ { k }$ absorbs its square. Conditioning before the batch therefore yields

$$
\mathbb { E } _ { k } \operatorname* { s u p } _ { \pi \in \Pi } \| D _ { \pi } \Psi _ { k } \| ^ { 2 } \leq \frac { C ( 1 + U _ { k } ^ { 4 } ) } { m _ { k } } .
$$

For the value segment, the explicit value solve gives

$$
V ^ { k + 1 } = ( \beta \widetilde { M } _ { k + 1 } ^ { \mathsf { T } } \widehat { M } _ { k + 1 } + \eta _ { V } ^ { - 1 } I ) ^ { - 1 } \left( \beta \widetilde { M } _ { k + 1 } ^ { \mathsf { T } } \widehat { r } _ { \pi ^ { k + 1 } , k } - \widetilde { M } _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k } - c + \eta _ { V } ^ { - 1 } V ^ { k } \right) ,
$$

so $\| V ^ { k + 1 } \| \le C ( U _ { k } + u _ { k } )$ . On the segment from $V ^ { k }$ to $V ^ { k + 1 }$

$$
\| D _ { V } \Psi _ { k } \| \le C \{ p _ { k } ( U _ { k } + u _ { k } ) + u _ { k } \} , \qquad \mathbb { E } _ { k } \operatorname* { s u p } _ { V \in [ V ^ { k } , V ^ { k + 1 } ] } \| D _ { V } \Psi _ { k } \| ^ { 2 } \le C ( 1 + U _ { k } ^ { 2 } ) / m _ { k } .
$$

Taking expectations and applying Lemma 4.2 controls both comparison-segment errors by $C / m _ { k }$ Young’s inequality therefore yields

$$
\mathbb { E } [ \mathcal { L } _ { \beta } ( \pi ^ { k } , V ^ { k } , \lambda ^ { k } ) - \mathcal { L } _ { \beta } ( \pi ^ { k + 1 } , V ^ { k } , \lambda ^ { k } ) ] \ge \frac { 1 } { 4 \eta _ { \pi } } \mathbb { E } \| \Delta \pi _ { k + 1 } \| ^ { 2 } - \frac { C } { m _ { k } } ,\tag{17}
$$

$$
\mathbb { E } [ \mathcal { L } _ { \beta } ( \pi ^ { k + 1 } , V ^ { k } , \lambda ^ { k } ) - \mathcal { L } _ { \beta } ( \pi ^ { k + 1 } , V ^ { k + 1 } , \lambda ^ { k } ) ] \ge \frac { 1 } { 4 \eta _ { V } } \mathbb { E } \| \Delta V _ { k + 1 } \| ^ { 2 } - \frac { C } { m _ { k } } .\tag{18}
$$

Let $a _ { \pi } = 1 / ( 4 \eta _ { \pi } )$ and $a _ { V } = 1 / ( 4 \eta _ { V } )$

Part (b): dual ascent. By Lemma 4.4, $\Delta \lambda _ { k + 1 } = \beta R _ { k + 1 } + d _ { k + 1 }$ . Therefore

$$
\mathcal { L } _ { \beta } ( \pi ^ { k + 1 } , V ^ { k + 1 } , \lambda ^ { k + 1 } ) - \mathcal { L } _ { \beta } ( \pi ^ { k + 1 } , V ^ { k + 1 } , \lambda ^ { k } )\tag{19}
$$

$$
= \langle \Delta \lambda _ { k + 1 } , R _ { k + 1 } \rangle = \frac { 1 } { \beta } \| \Delta \lambda _ { k + 1 } \| ^ { 2 } - \frac { 1 } { \beta } \langle \Delta \lambda _ { k + 1 } , d _ { k + 1 } \rangle\tag{20}
$$

$$
\leq \frac { 3 } { 2 \beta } \| \Delta \lambda _ { k + 1 } \| ^ { 2 } + \frac { 1 } { 2 \beta } \| d _ { k + 1 } \| ^ { 2 } .\tag{21}
$$

The second moment of $d _ { k + 1 }$ is $O ( m _ { k } ^ { - 1 } )$ . Substituting Lemma 4.3 into (21) and combining with (17)–(18) gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \mathcal { L } _ { \beta } ^ { k } ] - \mathbb { E } [ \mathcal { L } _ { \beta } ^ { k + 1 } ] \ge \left( a _ { \pi } - \frac { 3 C _ { \pi } } { 2 \beta } \right) \mathbb { E } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + ( a _ { V } - D _ { V } ) \mathbb { E } \| \Delta V _ { k + 1 } \| ^ { 2 } - D _ { V } \mathbb { E } \| \Delta V _ { k } \| ^ { 2 } } \\ { \displaystyle \quad \quad - C _ { E } \left( m _ { k } ^ { - 1 } + m _ { k - 1 } ^ { - 1 } \right) , } \end{array}
$$

where $D _ { V } = 3 C _ { V } / ( 2 \beta )$ . Adding $\tau \| \Delta V _ { k } \| ^ { 2 }$ to the potential gives coeficients

$$
c _ { \pi } = a _ { \pi } - \frac { 3 C _ { \pi } } { 2 \beta } , \quad c _ { V } = a _ { V } - D _ { V } - \tau , \quad c _ { V } ^ { \prime } = \tau - D _ { V } .
$$

Assumption 3.3 makes all three positive. This proves (6).

Part (c): lower bound. From the true perturbed value-dual identity, $\lambda ^ { k } = - M _ { k } ^ { - \mathsf { T } } ( c + \eta _ { V } ^ { - 1 } \Delta V _ { k } -$ $r _ { k } )$ , while $V ^ { k } = M _ { k } ^ { - 1 } ( r _ { \pi ^ { k } } + R _ { k } )$ . Substituting both into $\mathcal { L } _ { \beta } ^ { k }$ gives

$$
\mathcal { L } _ { \beta } ^ { k } = b ( \pi ^ { k } ) - \eta _ { V } ^ { - 1 } \langle \Delta V _ { k } , M _ { k } ^ { - 1 } R _ { k } \rangle + \langle r _ { k } , M _ { k } ^ { - 1 } R _ { k } \rangle + \frac { \beta } { 2 } \| R _ { k } \| ^ { 2 } .
$$

Two applications of Young’s inequality yield

$$
\Phi _ { k } \ge b _ { \operatorname* { m i n } } + \frac { \beta } { 4 } \| R _ { k } \| ^ { 2 } + \left( \tau - \frac { 2 \bar { \kappa } ^ { 2 } } { \beta \eta _ { V } ^ { 2 } } \right) \| \Delta V _ { k } \| ^ { 2 } - \frac { 2 \bar { \kappa } ^ { 2 } } { \beta } \| r _ { k } \| ^ { 2 } .
$$

The parameter condition makes the coeficient of $\| \Delta V _ { k } \| ^ { 2 }$ positive. Taking expectations and using (13) gives $\mathbb { E } \Phi _ { k } \ge \underline { { \Phi } } - C _ { L } m _ { k - 1 } ^ { - 1 }$ □

Role. Lemma 4.6 is the summability engine for both the asymptotic and finite-time results.

## Controlled-Markov convergence

Proof of Theorem 4.7. With no additional observation noise, Lemma 4.1 gives $\mathbb { E } \varepsilon _ { k } ^ { 2 } \le C / m _ { k }$ . Hence $\textstyle \sum _ { k } m _ { k } ^ { - 1 } < \infty$ implies $\textstyle \sum _ { k } \mathbb { E } \varepsilon _ { k } ^ { 2 } < \infty$ , and Tonelli gives $\textstyle \sum _ { k } \varepsilon _ { k } ^ { 2 } < \infty$ almost surely.

Fix a path in this probability-one event. The empirical rewards are pathwise bounded because the true rewards are bounded and $\varepsilon _ { k } \to 0$ . The same two-dimensional recursion used in Lemma 4.2, now pathwise, gives bounded $V ^ { k }$ and $\lambda ^ { k }$ . Lemma 4.4 and the pathwise version of the Lyapunov calculation then yield

$$
\begin{array} { r } { \Phi _ { k } - \Phi _ { k + 1 } \geq c _ { \pi } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + c _ { V } \| \Delta V _ { k + 1 } \| ^ { 2 } + c _ { V } ^ { \prime } \| \Delta V _ { k } \| ^ { 2 } - C ( \varepsilon _ { k } ^ { 2 } + \varepsilon _ { k - 1 } ^ { 2 } ) . } \end{array}
$$

The lower-bound calculation in Lemma 4.6 remains valid pathwise up to a summable $O ( \varepsilon _ { k - 1 } ^ { 2 } )$ term. Summing over k shows

$$
\sum _ { k } \| \Delta \pi _ { k + 1 } \| ^ { 2 } < \infty , \qquad \sum _ { k } \| \Delta V _ { k + 1 } \| ^ { 2 } < \infty .
$$

The pathwise version of Lemma 4.3 and $\textstyle \sum _ { k } \varepsilon _ { k } ^ { 2 } < \infty$ give $\begin{array} { r } { \sum _ { k } \| \Delta \lambda _ { k + 1 } \| ^ { 2 } < \infty } \end{array}$ . Thus all increments converge to zero. Lemma 4.4, Part (c), gives $R _ { k + 1 } = \beta ^ { - 1 } ( \Delta \lambda _ { k + 1 } - d _ { k + 1 } ) \to 0$ . Part (d) then gives KKT-residual convergence.

The policy space is compact and the value-dual iterates are bounded, so accumulation points exist. Fix a convergent subsequence $z ^ { k _ { j } }  z ^ { \star }$ . Vanishing increments imply $z ^ { k _ { j } - 1 } \to z ^ { \star }$ . The descent inequality bounds $\Phi _ { k }$ above, so $\phi ( \pi ^ { k } )$ is bounded above on this path. Lower semicontinuity gives $\phi ( \pi ^ { \star } ) < \infty$

Write $H _ { j } ( \pi )$ for the smooth part of $\widehat { \mathcal { L } } _ { \beta , k _ { i } - 1 } ( \pi , V ^ { k _ { j } - 1 } , \lambda ^ { k _ { j } - 1 } )$ . Model-error convergence and bounded iterates imply that $H _ { j }$ converges uniformly on Π to a continuous function. Exact policy minimization, compared with $\pi ^ { \star }$ , gives

$$
\phi ( \pi ^ { k _ { j } } ) + H _ { j } ( \pi ^ { k _ { j } } ) + \frac { \| \pi ^ { k _ { j } } - \pi ^ { k _ { j } - 1 } \| ^ { 2 } } { 2 \eta _ { \pi } } \leq \phi ( \pi ^ { \star } ) + H _ { j } ( \pi ^ { \star } ) + \frac { \| \pi ^ { \star } - \pi ^ { k _ { j } - 1 } \| ^ { 2 } } { 2 \eta _ { \pi } } .
$$

Taking the upper limit and using lower semicontinuity proves $\psi ( \pi ^ { k _ { j } } ) \to \psi ( \pi ^ { \star } )$ . The policy Fermat condition supplies $w _ { j } \in \partial \psi ( \pi ^ { k _ { j } } )$ with $w _ { j }  - J _ { \pi } ( \pi ^ { \star } , V ^ { \star } ) ^ { \mathsf { T } } \lambda ^ { \star }$ . The limiting subdiferential is closed along this function-value-convergent sequence. Hence $0 \in \partial \psi ( \pi ^ { \star } ) + J _ { \pi } ( \pi ^ { \star } , V ^ { \star } ) ^ { \mathsf { T } } { \lambda } ^ { \star }$ . The other two KKT conditions follow by continuity. This proves the accumulation-point claim without a subdifferential sum rule. □

Role. Theorem 4.7 is the first complete Markovian nonconvex ADMM convergence theorem in the paper. It validates that Bellman-resolvent stability survives controlled Markov sampling.

## Companion residual

Proof of Lemma 5.1. The policy-subproblem Fermat condition is

$$
0 \in \partial \psi ( \pi ^ { k + 1 } ) + \widehat { J } _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) ^ { \mathsf { T } } \left[ \lambda ^ { k } + \beta \widehat { R } _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) \right] + \eta _ { \pi } ^ { - 1 } \Delta \pi _ { k + 1 } .\tag{22}
$$

The bracket is $\widetilde { \lambda } ^ { k + 1 }$ . Hence the true policy residual at $\widetilde { z } ^ { k + 1 }$ is bounded by

$$
e _ { \pi , k + 1 } \leq \eta _ { \pi } ^ { - 1 } \| \Delta \pi _ { k + 1 } \| + \| J _ { \pi } ( \pi ^ { k + 1 } , V ^ { k } ) - \widehat { J } _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) \| \| \widetilde { \boldsymbol { \lambda } } ^ { k + 1 } \| .
$$

Use the conditional variables $U _ { k } , p _ { k } , u _ { k }$ from the preceding descent proof. The empirical companion multiplier satisfies $\| \widetilde { \lambda } ^ { k + 1 } \| \le C ( U _ { k } + u _ { k } )$ . Consequently

$$
\begin{array} { r } { \| ( J _ { \pi } - \widehat { J } _ { k } ) ^ { \mathsf { T } } \widetilde { \lambda } ^ { k + 1 } \| \leq C ( p _ { k } U _ { k } ^ { 2 } + u _ { k } U _ { k } + u _ { k } ^ { 2 } ) . } \end{array}
$$

Conditioning on $\mathcal { F } _ { k }$ and then using the fourth-moment stability yields

$$
\mathbb { E } e _ { \pi , k + 1 } ^ { 2 } \leq C \mathbb { E } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + \frac { C } { m _ { k } } .\tag{23}
$$

The companion and final multipliers satisfy

$$
\lambda ^ { k + 1 } - \widetilde { \lambda } ^ { k + 1 } = \beta \widehat { M } _ { k + 1 } \Delta V _ { k + 1 } .
$$

Substitution into the empirical value-dual identity gives

$$
c + \widehat { M } _ { k + 1 } ^ { \mathsf { T } } \widetilde { \lambda } ^ { k + 1 } = - \left( \eta _ { V } ^ { - 1 } I + \beta \widehat { M } _ { k + 1 } ^ { \mathsf { T } } \widehat { M } _ { k + 1 } \right) \Delta V _ { k + 1 } .
$$

Replacing $\widehat { M } _ { k + 1 }$ by $M _ { k + 1 }$ and applying the fourth-moment bounds yields

$$
\mathbb { E } \| c + M _ { k + 1 } ^ { \mathsf { T } } \widetilde { \lambda } ^ { k + 1 } \| ^ { 2 } \leq C \mathbb { E } \| \Delta V _ { k + 1 } \| ^ { 2 } + \frac { C } { m _ { k } } .\tag{24}
$$

Finally,

$$
\widehat { R } _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) = \beta ^ { - 1 } \Delta \lambda _ { k + 1 } - \widehat { M } _ { k + 1 } \Delta V _ { k + 1 } .
$$

Thus

$$
\| R ( \pi ^ { k + 1 } , V ^ { k } ) \| \leq \beta ^ { - 1 } \| \Delta \lambda _ { k + 1 } \| + \| \widehat M _ { k + 1 } \| \| \Delta V _ { k + 1 } \| + \| e _ { k } ( \pi ^ { k + 1 } , V ^ { k } ) \| .
$$

Squaring, taking expectations, and applying Lemma 4.3 controls feasibility by the three increment energies plus $m _ { k } ^ { - 1 } + m _ { k - 1 } ^ { - 1 }$ . Adding this to (23) and (24) proves (7). □

Role. The companion iterate removes endpoint products between random multipliers and the latest value increment. This is the technical step that makes a deterministic-constant fourth-moment complexity bound possible.

## Finite-time complexity

Proof of Theorem 5.2. Sum (6) from $k = 1$ to $T - 1$ and let $c _ { \star } = \operatorname* { m i n } \{ c _ { \pi } , c _ { V } , c _ { V } ^ { \prime } \}$ . The first update has uniformly bounded fourth moments by the explicit value solve and the empirical-model bounds. Its policy comparison with the deterministic $\pi ^ { 0 } \in$ dom ψ also gives $\begin{array} { r } { \operatorname* { s u p } _ { m _ { 0 } } \mathbb { E } \Phi _ { 1 } < \infty . } \end{array}$ , and the companion Fermat condition gives $\mathrm { s u p } _ { m _ { 0 } } \mathbb { E } \widetilde { G } _ { 1 } < \infty$ . Absorb these first-step quantities into the initial constant. The lower bound in Lemma 4.6 absorbs the terminal potential and gives constants $A _ { 0 } , A _ { 1 }$ such that

$$
\sum _ { k = 0 } ^ { T - 1 } \mathbb { E } \left[ \| \Delta \pi _ { k + 1 } \| ^ { 2 } + \| \Delta V _ { k + 1 } \| ^ { 2 } + \| \Delta V _ { k } \| ^ { 2 } \right] \leq A _ { 0 } + A _ { 1 } \sum _ { k = 0 } ^ { T - 1 } m _ { k } ^ { - 1 } .
$$

Apply Lemma 5.1, sum over $k ,$ and divide by $T \colon$

$$
\frac { 1 } { T } \sum _ { k = 0 } ^ { T - 1 } \mathbb { E } \widetilde { G } _ { k + 1 } \leq \frac { A } { T } + \frac { B } { T } \sum _ { k = 0 } ^ { T - 1 } m _ { k } ^ { - 1 } .
$$

Since K is uniform and independent, the left side equals $\mathbb { E } \widetilde { G } _ { K + 1 }$ , proving Theorem 5.2.

For a fixed total budget $\begin{array} { r } { N = \sum _ { k < T } m _ { k } } \end{array}$ , convexity of $x \mapsto 1 / x$ gives

$$
\sum _ { k = 0 } ^ { T - 1 } m _ { k } ^ { - 1 } \geq \frac { T ^ { 2 } } { N } ,
$$

with equality for equal batches $m _ { k } = N / T$ . Substitution gives the equal-batch bound in Theorem 5.2. Taking $T \asymp \sqrt { N }$ yields $\mathbb { E } \widetilde { G } _ { K + 1 } = O ( N ^ { \dot { - } 1 / 2 } )$ . To make the squared residual at most $\varepsilon ,$ it is suficient to choose $T = { \cal { O } } ( \varepsilon ^ { - 1 } )$ and $N = O ( \varepsilon ^ { - 2 } )$ □

Role. Theorem 5.2 is the finite-time optimization guarantee that the policy-performance analysis later converts into a return-gap rate.

Proof of Corollary 5.3. If $\begin{array} { r } { ( 1 / T ) \sum _ { k < T } m _ { k } ^ { - 1 } = O ( T ^ { - 1 } ) } \end{array}$ , Theorem 5.2 gives $\mathbb { E } \widetilde { G } _ { K + 1 } = O ( T ^ { - 1 } )$ . For fixed N the equal-batch conclusion follows from Jensen’s inequality as in the theorem proof. Setting $A / T \le \varepsilon / 2$ and $B T / N \leq \varepsilon / 2$ gives $T = { \cal { O } } ( \varepsilon ^ { - 1 } )$ and $N = O ( \varepsilon ^ { - 2 } )$ □

Role. The corollary translates the theorem into the conventional iteration and sample-complexity language used in the abstract and introduction.

## General stochastic observations

Proof of Theorem $4 . 8 .$ Lemma B.3 and the summability conditions in Theorem 4.8 imply $\textstyle \sum _ { k } \varepsilon _ { k } ^ { 2 } <$ ∞ almost surely. Fix a path in this event. Since $\varepsilon _ { k } \to 0$ and the true reward is bounded, the empirical rewards are pathwise bounded. The pathwise two-dimensional recursion in the proof of Lemma 4.2 therefore gives bounded $V ^ { k }$ and $\lambda ^ { k }$

Lemma 4.4 and the pathwise Lyapunov calculation give

$$
\begin{array} { r } { \Phi _ { k } - \Phi _ { k + 1 } \geq c _ { \pi } \| \Delta \pi _ { k + 1 } \| ^ { 2 } + c _ { V } \| \Delta V _ { k + 1 } \| ^ { 2 } + c _ { V } ^ { \prime } \| \Delta V _ { k } \| ^ { 2 } - C ( \varepsilon _ { k } ^ { 2 } + \varepsilon _ { k - 1 } ^ { 2 } ) . } \end{array}
$$

The lower bound difers from a fixed constant by at most $C \varepsilon _ { k - 1 } ^ { 2 }$ . Summation yields square summability of the policy and value increments. The pathwise five-term dual estimate then gives square summability of $\Delta \lambda _ { k + 1 }$ . Hence all increments vanish. The inexact dual update gives feasibility, and the true KKT bound in Lemma 4.4 gives stationarity. Accumulation-point KKT optimality follows from the exact policy comparison and function-value convergence argument in the proof of Theorem 4.7. □

Role. Theorem 4.8 extends the controlled-Markov result to centered observation noise and decaying bias without imposing pathwise bounded noise.

## Exact policy subproblems

For fixed $( V ^ { k } , \lambda ^ { k } )$ , the empirical Bellman residual is afine in $\pi .$ Its squared norm plus the proximal term is a strongly convex quadratic. When $\phi \equiv 0$ , the policy step is therefore a simplex-constrained quadratic program, separable across states. More generally, consider the nonsmooth, nonconvex regularizer $\begin{array} { r } { \phi ( { \boldsymbol \pi } ) = a \sum _ { s } \| { \boldsymbol \pi } _ { s } \| _ { 0 } } \end{array}$ with $a > 0$ . For each state, enumerate its nonempty action supports, minimize the quadratic on each corresponding simplex face, and select the candidate with smallest original objective. The support of a global minimizer occurs in this finite enumeration, which proves exactness. The enumeration uses at most $2 ^ { | { \cal A } | } - 1$ face problems per state. This example supplies an exact nonsmooth policy oracle for small action spaces. The stated iteration and sample bounds count outer iterations and transitions. They do not count internal policy-subproblem operations.

## D From Bellman KKT Residuals to Global Policy Performance

Throughout this section $\phi \equiv 0$ and $c = - \mu$ . Let $V ^ { \pi } = M _ { \pi } ^ { - 1 } r _ { \pi } , F _ { \mu } ( \pi ) = - \mu ^ { \mathsf { T } } V ^ { \pi }$ , and $\lambda _ { \mu } ^ { \pi } = M _ { \pi } ^ { - \mathsf { T } } \mu .$ For a point $( \pi , V , \lambda )$ write

$$
e _ { \pi } = \mathrm { d i s t } ( 0 , J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } \lambda + N _ { \Pi } ( \pi ) ) , \quad e _ { V } = \| M _ { \pi } ^ { \mathsf { T } } \lambda - \mu \| , \quad e _ { R } = \| M _ { \pi } V - r _ { \pi } \| ,
$$

so $G = e _ { \pi } ^ { 2 } + e _ { V } ^ { 2 } + e _ { R } ^ { 2 } ,$

Lemma D.1 (Bellman adjoint representation of the true policy gradient). The reduced policy objective satisfies

$$
\nabla F _ { \mu } ( \pi ) = J _ { \pi } ( \pi , V ^ { \pi } ) ^ { \mathsf { T } } \lambda _ { \mu } ^ { \pi } .
$$

Proof. Diferentiate the Bellman equation $R ( \pi , V ^ { \pi } ) = 0$ in a feasible policy direction dπ:

$$
J _ { \pi } ( \pi , V ^ { \pi } ) d \pi + M _ { \pi } d V ^ { \pi } = 0 , \qquad d V ^ { \pi } = - M _ { \pi } ^ { - 1 } J _ { \pi } ( \pi , V ^ { \pi } ) d \pi .
$$

Since $F _ { \mu } ( \pi ) = - \mu ^ { \mathsf { T } } V ^ { \pi }$

$$
d F _ { \mu } = \mu ^ { \mathsf { T } } M _ { \pi } ^ { - 1 } J _ { \pi } ( \pi , V ^ { \pi } ) d \pi = ( \lambda _ { \mu } ^ { \pi } ) ^ { \mathsf { T } } J _ { \pi } ( \pi , V ^ { \pi } ) d \pi .
$$

The coeficient of dπ is the claimed gradient.

Role. This identity is the first bridge from the constrained Bellman KKT system to stationarity of the reduced policy objective.

Lemma D.2 (Tabular Bellman policy Jacobian). For direct tabular policies and $Q _ { V } ( s , a ) = r ( s , a ) +$ $\gamma P ( \cdot \mid s , a ) ^ { \mathsf { T } } V$ 2

$$
\frac { \partial R _ { s } ( \pi , V ) } { \partial \pi ( a \mid s ) } = - Q _ { V } ( s , a ) , \qquad [ J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } \lambda ] _ { s , a } = - \lambda _ { s } Q _ { V } ( s , a ) .
$$

In particular $[ \nabla F _ { \mu } ( \pi ) ] _ { s , a } = - \lambda _ { \mu , s } ^ { \pi } Q ^ { \pi } ( s , a )$

Proof. For each state,

$$
R _ { s } ( \pi , V ) = V _ { s } - \sum _ { a } \pi ( a \mid s ) \left[ r ( s , a ) + \gamma P ( \cdot \mid s , a ) ^ { \mathsf { T } } V \right] .
$$

Diferentiating with respect to $\pi ( a \mid s )$ gives the first identity. Multiplication by λ gives the second. The last statement follows from Lemma D.1 with $V = V ^ { \pi }$ □

Role. The explicit Jacobian isolates the only two discrepancies between the ADMM stationarity vector and the true policy gradient: $V - V ^ { \pi }$ and $\lambda - \lambda _ { \mu } ^ { \pi }$

Lemma D.3 (Bellman feasibility controls the value error). For every $( \pi , V )$

$$
\| V - V ^ { \pi } \| \leq \bar { \kappa } e _ { R } .
$$

Proof. Since $M _ { \pi } V - r _ { \pi } = R ( \pi , V )$ and $M _ { \pi } V ^ { \pi } - r _ { \pi } = 0$

$$
M _ { \pi } ( V - V ^ { \pi } ) = R ( \pi , V ) .
$$

Multiplication by $M _ { \pi } ^ { - 1 }$ and the resolvent bound give the claim.

Role. This lemma converts primal feasibility error into the value mismatch needed in Lemma D.5. Lemma D.4 (Value stationarity controls the adjoint error). For every $( \pi , \lambda )$ ，

$$
\| \lambda - \lambda _ { \mu } ^ { \pi } \| \leq \bar { \kappa } e _ { V } .
$$

Proof. By definition $M _ { \pi } ^ { \mathsf { T } } \lambda _ { \mu } ^ { \pi } = \mu$ . Hence

$$
M _ { \pi } ^ { \mathsf { T } } ( \lambda - \lambda _ { \mu } ^ { \pi } ) = M _ { \pi } ^ { \mathsf { T } } \lambda - \mu .
$$

Multiplication by $M _ { \pi } ^ { - \mathsf { T } }$ proves the result.

Role. This is the dual counterpart of Lemma D.3 and controls the occupancy-weight error in the policy gradient.

Lemma D.5 (Adjoint-gradient perturbation). Assume $| r ( s , a ) | \leq R _ { \operatorname* { m a x } }$ and let $Q _ { \mathrm { m a x } } = R _ { \mathrm { m a x } } / ( 1 -$ γ). There are finite constants $C _ { V } , C _ { R } , C _ { V R }$ such that

$$
\begin{array} { r } { \| J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } \lambda - \nabla F _ { \mu } ( \pi ) \| \le C _ { V } e _ { V } + C _ { R } e _ { R } + C _ { V R } e _ { V } e _ { R } . } \end{array}
$$

One may choose constants proportional to $\sqrt { | A | } Q _ { \mathrm { m a x } } \bar { \kappa } , \gamma \sqrt { | A | } \bar { \kappa } ^ { 2 } \| \mu \|$ , and $\gamma \sqrt { | \mathcal { A } | } \bar { \kappa } ^ { 2 }$ , respectively. Proof. Using Lemma D.1, add and subtract the two mixed terms:

$$
\begin{array} { r l } & { J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } \lambda - \nabla F _ { \mu } ( \pi ) = J _ { \pi } ( \pi , V ^ { \pi } ) ^ { \mathsf { T } } ( \lambda - \lambda _ { \mu } ^ { \pi } ) } \\ & { \phantom { J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } } + [ J _ { \pi } ( \pi , V ) - J _ { \pi } ( \pi , V ^ { \pi } ) ] ^ { \mathsf { T } } \lambda _ { \mu } ^ { \pi } } \\ & { \phantom { J _ { \pi } ( \pi , V ) ^ { \mathsf { T } } } + [ J _ { \pi } ( \pi , V ) - J _ { \pi } ( \pi , V ^ { \pi } ) ] ^ { \mathsf { T } } ( \lambda - \lambda _ { \mu } ^ { \pi } ) . } \end{array}
$$

For the first term, Lemma D.2 and $\vert Q ^ { \pi } ( s , a ) \vert \ \leq \ Q _ { \mathrm { m a x } }$ give an operator bound proportional to $\sqrt { | \mathcal { A } | } Q _ { \mathrm { m a x } } \| \lambda - \lambda _ { \mu } ^ { \pi } \|$ . Lemma D.4 yields the e<sub>V</sub> term.

For the second and third terms, the tabular Jacobian satisfies

$$
Q _ { V } ( s , a ) - Q ^ { \pi } ( s , a ) = \gamma P ( \cdot \mid s , a ) ^ { \mathsf { T } } ( V - V ^ { \pi } ) ,
$$

so its change is bounded by $\gamma { \sqrt { | { \mathcal { A } } | } } \| V - V ^ { \pi } \|$ . Also $\| \lambda _ { \mu } ^ { \pi } \| \leq \bar { \kappa } \| \mu \|$ . Lemmas D.3 and D.4 then give the $e _ { R }$ and $e _ { V } e _ { R }$ terms. □

Role. This is the final bridge needed to prove that a small constrained KKT residual implies a small stationarity residual for the true reduced policy objective.

Proof of Lemma 6.1. Distance to a translated closed set is 1-Lipschitz. Therefore

$$
\begin{array} { r l } & { R _ { \mathrm { p o l } } ( \pi ) = \mathrm { d i s t } ( 0 , \nabla F _ { \mu } ( \pi ) + N _ { \Pi } ( \pi ) ) } \\ & { \qquad \le \mathrm { d i s t } ( 0 , J _ { \pi } ( \pi , V ) ^ { \top } \lambda + N _ { \Pi } ( \pi ) ) + \| J _ { \pi } ( \pi , V ) ^ { \top } \lambda - \nabla F _ { \mu } ( \pi ) \| } \\ & { \qquad \le e _ { \pi } + C _ { V } e _ { V } + C _ { R } e _ { R } + C _ { V R } e _ { V } e _ { R } . } \end{array}
$$

If $G \leq 1$ , then $e _ { \pi } , e _ { V } , e _ { R } \le \sqrt { G }$ and $e _ { V } e _ { R } \leq ( e _ { V } ^ { 2 } + e _ { R } ^ { 2 } ) / 2 \leq G / 2 \leq \sqrt { G } / 2$ . Hence

$$
R _ { \mathrm { p o l } } ( \pi ) \leq \left( 1 + C _ { V } + C _ { R } + \frac { 1 } { 2 } C _ { V R } \right) \sqrt { G } = : C _ { \mathrm { s t a t } } \sqrt { G } .
$$

Role. Lemma 6.1 composes the optimization theory with policy geometry. Every global performance theorem below uses it.

Lemma D.6 (Simplex normal-cone geometry). Let $p \in \Delta _ { m } , q \in \mathbb { R } ^ { m }$ , and $\lambda > 0$ . If

$$
r = \mathrm { d i s t } ( 0 , - \lambda q + N _ { \Delta _ { m } } ( p ) ) ,
$$

then

$$
\operatorname* { m a x } _ { i } q _ { i } - p ^ { \mathsf { T } } q \leq \frac { \sqrt { 2 } } { \lambda } r .
$$

Proof. Choose $n ^ { \star } \in N _ { \Delta _ { m } } ( p )$ attaining the distance and set $h = - \lambda q + n ^ { \star }$ , so $\| h \| = r$ . Let $p ^ { \star } \in$ arg $\operatorname* { m a x } _ { u \in \Delta _ { m } } u ^ { \mathsf { T } } q$ . By the normal-cone inequality, $\langle n ^ { \star } , p ^ { \star } - p \rangle \leq 0$ . Therefore

$$
\begin{array} { r l } & { \lambda ( \boldsymbol { p } ^ { \star \mathsf { T } } \boldsymbol { q } - \boldsymbol { p } ^ { \mathsf { T } } \boldsymbol { q } ) = - \langle - \lambda \boldsymbol { q } , \boldsymbol { p } ^ { \star } - \boldsymbol { p } \rangle } \\ & { \qquad = - \langle h - \boldsymbol { n } ^ { \star } , \boldsymbol { p } ^ { \star } - \boldsymbol { p } \rangle } \\ & { \qquad \leq - \langle h , \boldsymbol { p } ^ { \star } - \boldsymbol { p } \rangle \leq \| h \| \| \boldsymbol { p } ^ { \star } - \boldsymbol { p } \| . } \end{array}
$$

The Euclidean diameter of the probability simplex is ${ \sqrt { 2 } } ,$ giving the claim.

Role. This lemma converts statewise policy stationarity into a bound on the maximum advantage and is the key geometric input to Theorem 6.2.

Proof of Theorem 6.2. Let $r _ { s } ( \pi )$ be the statewise component of $R _ { \mathrm { p o l } } ( \pi )$ . By Lemma D.2, the statewise gradient equals $- \lambda _ { \mu , s } ^ { \pi } Q ^ { \pi } ( s , \cdot )$ , where

$$
\lambda _ { \mu , s } ^ { \pi } = \frac { d _ { \mu } ^ { \pi } ( s ) } { 1 - \gamma } .
$$

Applying Lemma D.6 statewise gives

$$
\operatorname* { m a x } _ { a } A ^ { \pi } ( s , a ) \leq \sqrt { 2 } ( 1 - \gamma ) \frac { r _ { s } ( \pi ) } { d _ { \mu } ^ { \pi } ( s ) }
$$

whenever $d _ { \mu } ^ { \pi } ( s ) > 0$ . Under the support condition this is suficient on every state weighted by $d _ { \nu } ^ { \pi ^ { \star } }$ The performance-diference identity yields

$$
\begin{array} { l } { \displaystyle { J _ { \nu } ( \pi ^ { \star } ) - J _ { \nu } ( \pi ) \leq \frac { 1 } { 1 - \gamma } \sum _ { s } d _ { \nu } ^ { \pi ^ { \star } } ( s ) \operatorname* { m a x } _ { a } A ^ { \pi } ( s , a ) } } \\ { \displaystyle { \qquad \leq \sqrt { 2 } \sum _ { s } \frac { d _ { \nu } ^ { \pi ^ { \star } } ( s ) } { d _ { \mu } ^ { \pi } ( s ) } r _ { s } ( \pi ) } } \\ { \displaystyle { \qquad \leq \sqrt { 2 } \left. \frac { d _ { \nu } ^ { \pi ^ { \star } } } { d _ { \mu } ^ { \pi } } \right. _ { 2 } \left( \sum _ { s } r _ { s } ( \pi ) ^ { 2 } \right) ^ { 1 / 2 } . } } \end{array}
$$

The last factor is $R _ { \mathrm { p o l } } ( \pi )$ , proving Theorem 6.2. Equation the KKT bound in Theorem 6.2 follows from Lemma 6.1. If $G = 0$ , the right side vanishes, proving covered KKT global optimality. □

Role. This theorem is the principal RL-specific strengthening of the ADMM stationarity result.

Proof of Corollary 6.3. If $\mu ( s ) \geq \mu _ { \mathrm { m i n } }$ , then the unnormalized discounted occupancy satisfies $\lambda _ { \mu , s } ^ { \pi } =$ $\smash { \sum _ { t > 0 } \gamma ^ { t } \mathbb { P } _ { \mu , \pi } ( S _ { t } = s ) \geq \mu ( s ) \geq \mu _ { \mathrm { m i n } } }$ . Lemma D.6 therefore gives ma $\mathrm { x } _ { a } A ^ { \pi } ( s , a ) \leq \sqrt { 2 } r _ { s } ( \pi ) / \mu _ { \mathrm { m i n } } .$ Substitution into the performance-diference identity under the same evaluation distribution gives

$$
J _ { \mu } ^ { \star } - J _ { \mu } ( \pi ) \leq \frac { \sqrt { 2 } } { ( 1 - \gamma ) \mu _ { \operatorname* { m i n } } } R _ { \mathrm { p o l } } ( \pi ) .
$$

Setting $R _ { \mathrm { p o l } } ( \pi ) = 0$ proves global optimality.

Role. The corollary gives a simple assumption under which the coverage coeficient in Theorem 6.2 is automatically finite.

Proof of Corollary $6 . 4 \cdot$ The exact-KKT statement is the zero-residual case of Theorem 6.2. For the finite-time bound, let $D _ { \mathrm { m a x } } = 2 R _ { \mathrm { m a x } } / ( 1 - \gamma )$ . The return gap is bounded above by $D _ { \mathrm { m a x } }$ . On $G _ { K } \leq 1$ , Theorem 6.2 gives the bound $C _ { 0 } \sqrt { G _ { K } }$ under the uniform occupancy-mismatch assumption. On $G _ { K } > 1$ , the gap is at most $D _ { \operatorname* { m a x } } \sqrt { G _ { K } }$ . Hence the pointwise bound holds for every $G _ { K } \ge 0$ with $C = \operatorname* { m a x } \{ C _ { 0 } , D _ { \operatorname* { m a x } } \}$ . Taking expectations and applying Jensen gives

$$
\begin{array} { r } { \mathbb { E } [ J _ { \nu } ^ { \star } - J _ { \nu } ( \pi _ { K } ) ] \leq C \sqrt { \mathbb { E } G _ { K } } . } \end{array}
$$

Substitution of the companion-residual bound proves the claim, with the policy component of the companion output. □

Role. This corollary directly composes the finite-time KKT bound with the policy-performance conversion.

Proof of Proposition 6.5. Consider states $s _ { 0 } , s _ { 1 } , s _ { T }$ and training distribution $\mu = \delta _ { s _ { 0 } }$ . At $s _ { 0 }$ , action a terminates with reward 1, while action b moves to $s _ { 1 }$ with reward 0. At $s _ { 1 }$ , action c terminates with reward 0, while action d terminates with reward 10. Take $\gamma = 0 . 9$ and the policy choosing a at $s _ { 0 }$ and c at $s _ { 1 }$

Then $V ^ { \pi } ( s _ { 0 } ) = 1$ and $V ^ { \pi } ( s _ { 1 } ) = 0 . \mathrm { ~ A t ~ } s _ { 0 } , Q ^ { \pi } ( s _ { 0 } , a ) = 1$ and $Q ^ { \pi } ( s _ { 0 } , b ) = 0$ , so the current action is strictly greedy. The policy never visits $s _ { 1 }$ , hence $d _ { \mu } ^ { \pi } ( s _ { 1 } ) = 0$ and the policy gradient has zero weight there. Thus the policy is first-order stationary. However the coordinated policy choosing b at $s _ { 0 }$ and d at $s _ { 1 }$ has return $0 . 9 \times 1 0 = 9 > 1$ . The failure is exactly the missing occupancy support.

Role. The proposition shows that the coverage assumption in Theorem 6.2 is necessary for firstorder information to identify all globally relevant states.

Proof of Proposition 6.6. Consider the three-stage chain $s _ { 0 }  s _ { 1 }  s _ { 2 }  s _ { T }$ with actions stop and continue. All rewards are zero except that continuing at $s _ { 2 }$ gives reward −1. Let $\pi _ { \varepsilon }$ continue with probability ε at every state. The optimal value is zero, while a loss occurs only if all three continue actions are chosen, so

$$
J ^ { \star } - J ( \pi _ { \varepsilon } ) = \gamma ^ { 2 } \varepsilon ^ { 3 } .
$$

The unnormalized occupancies are $1 , \gamma \varepsilon , \gamma ^ { 2 } \varepsilon ^ { 2 }$ . The corresponding action-value gaps are $\gamma ^ { 2 } \varepsilon ^ { 2 } , \gamma \varepsilon ,$ 1. For an interior two-action simplex, the statewise normal-cone residual is the weighted action gap divided by ${ \sqrt { 2 } } ,$ so all three state residuals equal $\gamma ^ { 2 } \varepsilon ^ { 2 } / \sqrt { 2 }$ . Therefore

$$
R _ { \mathrm { p o l } } ( \pi _ { \varepsilon } ) ^ { 2 } = 3 \frac { \gamma ^ { 4 } \varepsilon ^ { 4 } } { 2 } .
$$

Consequently

$$
\frac { J ^ { \star } - J ( \pi _ { \varepsilon } ) } { R _ { \mathrm { p o l } } ( \pi _ { \varepsilon } ) ^ { 2 } } = \frac { 2 } { 3 \gamma ^ { 2 } \varepsilon }  \infty .
$$

No uniform constant can satisfy a quadratic PL-type inequality for all direct tabular policies.

Role. This negative result justifies why the generic policy-performance theorem has an $O ( { \sqrt { G } } )$ conversion and why Theorem 6.8 requires additional curvature.

Proof of Theorem 6.8. Let $\bar { r } _ { s } ( \pi )$ be the unweighted statewise stationarity residual. Because the true statewise policy gradient is scaled by $\lambda _ { \mu , s } ^ { \pi } = d _ { \mu } ^ { \pi } ( s ) / ( 1 - \gamma )$ ,

$$
r _ { s } ( \pi ) = \frac { d _ { \mu } ^ { \pi } ( s ) } { 1 - \gamma } \bar { r } _ { s } ( \pi ) , \qquad \bar { r } _ { s } ( \pi ) = \frac { 1 - \gamma } { d _ { \mu } ^ { \pi } ( s ) } r _ { s } ( \pi ) .
$$

Assumption 6.7 therefore gives

$$
\delta _ { s } ^ { \pi } \leq \frac { ( 1 - \gamma ) ^ { 2 } } { 2 \mu _ { B } [ d _ { \mu } ^ { \pi } ( s ) ] ^ { 2 } } r _ { s } ( \pi ) ^ { 2 } .
$$

The Bellman performance bound is

$$
J _ { \nu } ( \pi ^ { \star } ) - J _ { \nu } ( \pi ) \leq \frac { 1 } { 1 - \gamma } \sum _ { s } d _ { \nu } ^ { \pi ^ { \star } } ( s ) \delta _ { s } ^ { \pi } .
$$

Substituting the previous inequality gives

$$
\begin{array} { c l c r } { { J _ { \nu } ( \pi ^ { \star } ) - J _ { \nu } ( \pi ) \leq \displaystyle \frac { 1 - \gamma } { 2 \mu _ { B } } \sum _ { s } \displaystyle \frac { d _ { \nu } ^ { \pi ^ { \star } } ( s ) } { [ d _ { \mu } ^ { \pi } ( s ) ] ^ { 2 } } r _ { s } ( \pi ) ^ { 2 } } } \\ { { \leq \displaystyle \frac { 1 - \gamma } { 2 \mu _ { B } } \mathcal { C } _ { \mathrm { P L } } \sum _ { s } r _ { s } ( \pi ) ^ { 2 } , } } \end{array}
$$

which proves (6.8). Lemma 6.1 then gives the $O ( G )$ form.

Role. The theorem identifies the additional one-step curvature needed for policy performance to inherit the squared-KKT rate without the square-root loss.

Proof of Theorem 6.9. The global truncation argument in Corollary 6.4 gives $\begin{array} { r } { \mathbb { E } [ J ^ { \star } - J ( \pi _ { K } ) ] \le } \end{array}$ $C \sqrt { \mathbb { E } G _ { K } }$ . Under the quadratic improvement and stronger occupancy-ratio assumptions, Theorem 6.8 gives $J ^ { \star } - J ( \pi _ { K } ) \le C _ { 0 } G _ { K }$ on $G _ { K } \leq 1$ . On $G _ { K } > 1$ , bounded returns give $J ^ { \star } - J ( \pi _ { K } ) \le$ $D _ { \operatorname* { m a x } } G _ { K }$ . Thus $\begin{array} { r } { \mathbb { E } [ J ^ { \star } - J ( \pi _ { K } ) ] \le } \end{array}$ max $\{ C _ { 0 } , D _ { \mathrm { m a x } } \} \mathbb { E } G _ { K }$ . The pointwise bounds also imply policyperformance convergence whenever $G _ { K } \to 0$ , and global optimality at covered exact KKT points.

Role. This theorem summarizes the second layer of value supplied by RL structure: it interprets the optimization limit point rather than merely proving that the ADMM iterates become stationary.

A suficient action-gap condition. Suppose that on the policy set of interest every suboptimal action has gap max<sub>b</sub> $Q ^ { \pi } ( s , b ) - Q ^ { \pi } ( s , a ) \ge \Delta > 0$ . Fix a state and write $p = \pi _ { s } , q = Q ^ { \pi } ( s , \cdot )$ and $r = \mathrm { d i s t } ( 0 , - q + N _ { \Delta } ( p ) )$ ). If p is supported on maximizing actions, $\delta _ { s } ^ { \pi } = 0$ . Otherwise choose a supported suboptimal action a and a maximizing action b. For every $n \in N _ { \Delta } ( p )$ , the feasible direction $e _ { b } - e _ { a }$ gives $n _ { b } - n _ { a } \le 0$ . Thus

$$
\begin{array} { r } { \left. - q + n , e _ { b } - e _ { a } \right. \le - \Delta , \qquad r \ge \Delta / \sqrt { 2 } . } \end{array}
$$

Lemma D.6 gives $\delta _ { s } ^ { \pi } \leq \sqrt { 2 } r \leq 2 r ^ { 2 } / \Delta$ . Therefore Assumption 6.7 holds with $\mu _ { B } = \Delta / 4$ . For example, when transitions are action independent and rewards have a positive gap between optimal and suboptimal actions at each nontrivial state, the same $\Delta$ works for every policy.

## E Smooth Nonlinear Policy Parameterization

Let $\Theta \subset \mathbb { R } ^ { p }$ be closed and convex and let $\pi _ { \theta }$ be a smooth policy on a neighborhood of Θ. Set $\psi _ { \Theta } = \phi + \delta _ { \Theta }$ and assume it is proper, lower semicontinuous, and bounded below. Policy stationarity below uses $\partial \psi _ { \Theta }$ . Initialize deterministically with θ<sup>0</sup> ∈ $\theta ^ { 0 } \in$ dom $\psi _ { \Theta }$ and finite $V ^ { 0 } , \lambda ^ { 0 }$ . Define

$$
P _ { \theta } ( s ^ { \prime } \mid s ) = \sum _ { a } \pi _ { \theta } ( a \mid s ) P ( s ^ { \prime } \mid s , a ) , \quad r _ { \theta } ( s ) = \sum _ { a } \pi _ { \theta } ( a \mid s ) r ( s , a ) , \quad M _ { \theta } = I - \gamma P _ { \theta } ,
$$

and $R ( \theta , V ) = M _ { \theta } V - r _ { \theta }$ . Consider

$$
\operatorname* { m i n } _ { \theta , V } ~ \phi ( \theta ) + c ^ { \mathsf { T } } V \quad \mathrm { s . t . } \quad R ( \theta , V ) = 0 .
$$

Assume $\lVert M _ { \theta } - M _ { \theta ^ { \prime } } \rVert \leq L _ { M } \lVert \theta - \theta ^ { \prime } \rVert$ and $b _ { A } ( \theta ) = \phi ( \theta ) + c ^ { \mathsf { T } } M _ { \theta } ^ { - 1 } r _ { \theta }$ is bounded below.

Lemma E.1 (Euclidean Bellman-resolvent bound). For every row-stochastic $P _ { \theta }$

$$
\| ( I - \gamma P _ { \theta } ) ^ { - 1 } \| _ { 2 } \leq \frac { \sqrt { | S | } } { 1 - \gamma } .
$$

Proof. The Neumann series gives $\begin{array} { r } { M _ { \theta } ^ { - 1 } = \sum _ { t > 0 } \gamma ^ { t } P _ { \theta } ^ { t } } \end{array}$ . Since each $P _ { \theta } ^ { t }$ is row stochastic,

$$
\| M _ { \theta } ^ { - 1 } \| _ { \infty } \leq \sum _ { t \geq 0 } \gamma ^ { t } = \frac { 1 } { 1 - \gamma } .
$$

For every vector $x , \| M _ { \theta } ^ { - 1 } x \| _ { 2 } \leq \sqrt { | \mathcal { S } | } \| M _ { \theta } ^ { - 1 } x \| _ { \infty } \leq \sqrt { | \mathcal { S } | } ( 1 - \gamma ) ^ { - 1 } \| x \| _ { \infty } \leq \sqrt { | \mathcal { S } | } ( 1 - \gamma ) ^ { - 1 } \| x \| _ { 2 } .$ □

Role. This lemma supplies the uniform inverse constant $\kappa _ { M }$ used in every nonlinear-policy stability bound.

Lemma E.2 (Nonlinear inverse-map stability). If $\kappa _ { M } = \operatorname* { s u p } _ { \theta } \| M _ { \theta } ^ { - 1 } \| < \infty$ , then

$$
\lVert M _ { \theta } ^ { - 1 } - M _ { \theta ^ { \prime } } ^ { - 1 } \rVert \leq \kappa _ { M } ^ { 2 } L _ { M } \lVert \theta - \theta ^ { \prime } \rVert .
$$

Proof. Apply Lemma A.1 with $M = M _ { \theta ^ { \prime } }$ and $M ^ { \prime } = M _ { \theta }$ and then use the assumed Lipschitz bound for $M _ { \theta }$ □

Role. This is the nonlinear-policy version of the resolvent sensitivity used to control multiplier increments.

The exact proximal updates are

$$
\theta ^ { k + 1 } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } \left\{ \mathcal { L } _ { \beta } ( \theta , V ^ { k } , \lambda ^ { k } ) + \frac { 1 } { 2 \eta _ { \theta } } \| \theta - \theta ^ { k } \| ^ { 2 } \right\} ,
$$

$$
V ^ { k + 1 } = \operatorname * { a r g m i n } _ { V } \left\{ \mathcal { L } _ { \beta } ( \theta ^ { k + 1 } , V , \lambda ^ { k } ) + \frac { 1 } { 2 \eta _ { V } } \| V - V ^ { k } \| ^ { 2 } \right\} , \qquad \lambda ^ { k + 1 } = \lambda ^ { k } + \beta R ( \theta ^ { k + 1 } , V ^ { k + 1 } ) .
$$

Lemma E.3 (Exact policy-step decrease). The exact proximal policy update satisfies

$$
\mathcal { L } _ { \beta } ( \boldsymbol { \theta } ^ { k } , \boldsymbol { V } ^ { k } , \lambda ^ { k } ) - \mathcal { L } _ { \beta } ( \boldsymbol { \theta } ^ { k + 1 } , \boldsymbol { V } ^ { k } , \lambda ^ { k } ) \geq \frac { 1 } { 2 \eta _ { \boldsymbol { \theta } } } \| \Delta \theta _ { k + 1 } \| ^ { 2 } .
$$

Proof. Use $\theta ^ { k }$ as a feasible comparison point in the definition of $\theta ^ { k + 1 }$ . The proximal term vanishes at $\theta ^ { k }$ , while at $\theta ^ { k + 1 }$ it contributes $( 2 \eta _ { \theta } ) ^ { - 1 } \| \Delta \theta _ { k + 1 } \| ^ { 2 }$ □

Role. This is the primal decrease that will dominate the policy-dependent part of the dual ascent. Lemma E.4 (Nonlinear value-dual identity). For every $k _ { i }$

$$
\lambda ^ { k + 1 } = - M _ { k + 1 } ^ { - \mathsf { T } } \left( c + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } \right) , \qquad M _ { k + 1 } = M _ { \theta ^ { k + 1 } } .
$$

Proof. The value-subproblem first-order condition is

$$
0 = c + M _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k } + \beta M _ { k + 1 } ^ { \mathsf { T } } R _ { k + 1 } + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } .
$$

The dual update gives $\lambda ^ { k + 1 } = \lambda ^ { k } + \beta R _ { k + 1 }$ . Substitution gives $M _ { k + 1 } ^ { \mathsf { T } } \lambda ^ { k + 1 } = - ( c + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } )$ , and Lemma E.1 guarantees invertibility. □

Role. This identity is the exact nonlinear analogue of the tabular Bellman multiplier representation.

Lemma E.5 (Nonlinear multiplier increment). There are constants

$$
A _ { A } = 3 \kappa _ { M } ^ { 4 } L _ { M } ^ { 2 } \Vert c \Vert ^ { 2 } , \qquad B _ { A } = \frac { 3 \kappa _ { M } ^ { 2 } } { \eta _ { V } ^ { 2 } } ,
$$

such that

$$
\begin{array} { r } { \| \Delta \lambda _ { k + 1 } \| ^ { 2 } \leq A _ { A } \| \Delta \theta _ { k + 1 } \| ^ { 2 } + B _ { A } \| \Delta V _ { k + 1 } \| ^ { 2 } + B _ { A } \| \Delta V _ { k } \| ^ { 2 } . } \end{array}
$$

Proof. Subtract Lemma E.4 at consecutive iterations:

$$
\Delta \lambda _ { k + 1 } = - \big ( M _ { k + 1 } ^ { - \top } - M _ { k } ^ { - \top } \big ) c - \eta _ { V } ^ { - 1 } M _ { k + 1 } ^ { - \top } \Delta V _ { k + 1 } + \eta _ { V } ^ { - 1 } M _ { k } ^ { - \top } \Delta V _ { k } .
$$

Lemma E.2 bounds the first term by $\kappa _ { M } ^ { 2 } L _ { M } \| c \| \| \Delta \theta _ { k + 1 } \|$ . Bound the other two by $\kappa _ { M } \eta _ { V } ^ { - 1 }$ times their value increments and apply $\| x + y + z \| ^ { 2 } \leq 3 ( \| x \| ^ { 2 } + \| y \| ^ { 2 } + \| z \| ^ { 2 } )$ □

Role. This lemma is the deterministic nonlinear-policy multiplier-control result used in Theorems E.7 and E.9.

Lemma E.6 (Nonlinear Lyapunov lower bound). Let

$$
\begin{array} { r } { \Phi _ { k } ^ { A } = \mathcal { L } _ { \beta } ( \theta ^ { k } , V ^ { k } , \lambda ^ { k } ) + \tau \| \Delta V _ { k } \| ^ { 2 } . } \end{array}
$$

Then

$$
\Phi _ { k } ^ { A } \geq b _ { A } ^ { \operatorname* { i n f } } + \frac { \beta } { 4 } \| R _ { k } \| ^ { 2 } + \left( \tau - \frac { \kappa _ { M } ^ { 2 } } { \beta \eta _ { V } ^ { 2 } } \right) \| \Delta V _ { k } \| ^ { 2 } .
$$

Proof. Since $M _ { k } V ^ { k } = r _ { k } + R _ { k } , V ^ { k } = M _ { k } ^ { - 1 } ( r _ { k } + R _ { k } )$ . Lemma E.4 at the previous update gives $\lambda ^ { k } = - M _ { k } ^ { - \mathsf { T } } ( c + \eta _ { V } ^ { - 1 } \Delta V _ { k } )$ . Substitute both identities into the augmented Lagrangian:

$$
\mathcal { L } _ { \beta } ^ { k } = b _ { A } ( \theta ^ { k } ) - \eta _ { V } ^ { - 1 } \langle \Delta V _ { k } , M _ { k } ^ { - 1 } R _ { k } \rangle + \frac { \beta } { 2 } \| R _ { k } \| ^ { 2 } .
$$

Young’s inequality gives

$$
\eta _ { V } ^ { - 1 } \kappa _ { M } \| \Delta V _ { k } \| \| R _ { k } \| \leq \frac { \beta } { 4 } \| R _ { k } \| ^ { 2 } + \frac { \kappa _ { M } ^ { 2 } } { \beta \eta _ { V } ^ { 2 } } \| \Delta V _ { k } \| ^ { 2 } .
$$

Use $b _ { A } ( \theta ^ { k } ) \geq b _ { A } ^ { \mathrm { i n f } }$ and add $\tau \| \Delta V _ { k } \| ^ { 2 }$

Role. This lemma prevents the deterministic nonlinear-policy Lyapunov from diverging to −∞ and permits summation of the descent inequality.

Theorem E.7 (Exact nonlinear-policy convergence). If

$$
\frac { B _ { A } } { \beta } < \tau < \frac { 1 } { 2 \eta _ { V } } - \frac { B _ { A } } { \beta } , \qquad \frac { 1 } { 2 \eta _ { \theta } } > \frac { A _ { A } } { \beta } ,
$$

then there are $c _ { \theta } , c _ { V } , c _ { V } ^ { \prime } > 0$ such that

$$
\begin{array} { r } { \Phi _ { k } ^ { A } - \Phi _ { k + 1 } ^ { A } \geq c _ { \theta } \| \Delta \theta _ { k + 1 } \| ^ { 2 } + c _ { V } \| \Delta V _ { k + 1 } \| ^ { 2 } + c _ { V } ^ { \prime } \| \Delta V _ { k } \| ^ { 2 } . } \end{array}
$$

Consequently $\Delta \theta _ { k } , \Delta V _ { k } , \Delta \lambda _ { k } , R _ { k } \to 0$ . If $D _ { \theta } M _ { \theta }$ and $D _ { \theta } r _ { \theta }$ are bounded on the visited region, the KKT residual converges to zero.

Proof. Lemma E.3 gives policy decrease. The value comparison point $V ^ { k }$ similarly gives

$$
\mathcal { L } _ { \beta } ( { \boldsymbol { \theta } } ^ { k + 1 } , V ^ { k } , \lambda ^ { k } ) - \mathcal { L } _ { \beta } ( { \boldsymbol { \theta } } ^ { k + 1 } , V ^ { k + 1 } , \lambda ^ { k } ) \geq \frac { 1 } { 2 \eta _ { V } } \| \Delta V _ { k + 1 } \| ^ { 2 } .
$$

Because the deterministic dual update is exact, the multiplier step increases the augmented Lagrangian by

$$
\langle \Delta \lambda _ { k + 1 } , R _ { k + 1 } \rangle = \beta ^ { - 1 } \| \Delta \lambda _ { k + 1 } \| ^ { 2 } .
$$

Insert Lemma E.5. After adding $\tau \| \Delta V _ { k } \| ^ { 2 } - \tau \| \Delta V _ { k + 1 } \| ^ { 2 }$ , the coeficients are

$$
c _ { \theta } = \frac { 1 } { 2 \eta _ { \theta } } - \frac { A _ { A } } { \beta } , ~ c _ { V } = \frac { 1 } { 2 \eta _ { V } } - \frac { B _ { A } } { \beta } - \tau , ~ c _ { V } ^ { \prime } = \tau - \frac { B _ { A } } { \beta } ,
$$

which are positive by assumption. Lemma E.6 bounds $\Phi _ { k } ^ { A }$ from below, so summation gives square summability of the policy and value increments. Lemma E.5 then gives $\Delta \lambda _ { k } \to 0$ , while $\Delta \lambda _ { k + 1 } =$ $\beta R _ { k + 1 }$ gives feasibility.

For stationarity, the policy-subproblem Fermat condition is

$$
0 \in \partial \psi _ { \Theta } ( \theta ^ { k + 1 } ) + D _ { \theta } R ( \theta ^ { k + 1 } , V ^ { k } ) ^ { \mathsf { T } } [ \lambda ^ { k } + \beta R ( \theta ^ { k + 1 } , V ^ { k } ) ] + \eta _ { \theta } ^ { - 1 } \Delta \theta _ { k + 1 } .
$$

The bracket difers from $\lambda ^ { k + 1 }$ by $O ( \| \Delta V _ { k + 1 } \| )$ , and bounded $D _ { \theta } M _ { \theta } , D _ { \theta } r _ { \theta }$ make the change from $V ^ { k }$ to $V ^ { k + 1 }$ linear in $\| \Delta V _ { k + 1 } \|$ . Thus the policy KKT residual tends to zero. Lemma E.4 gives the value residual, and feasibility has already been proved. □

Role. This theorem shows that the Bellman-resolvent mechanism is not tied to tabular direct policies.

For an implementable parameter step define

$$
h _ { k } ( \theta ) = \langle { \lambda } ^ { k } , R ( \theta , V ^ { k } ) \rangle + \frac { \beta } { 2 } \| R ( \theta , V ^ { k } ) \| ^ { 2 } ,
$$

and use

$$
\theta ^ { k + 1 } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } \left\{ \phi ( \theta ) + \langle \nabla h _ { k } ( \theta ^ { k } ) , \theta - \theta ^ { k } \rangle + \frac { 1 } { 2 \eta _ { \theta , k } } \| \theta - \theta ^ { k } \| ^ { 2 } \right\} .
$$

Lemma E.8 (Proximal-gradient policy decrease). $I f \nabla h _ { k }$ is $L _ { \theta , k } – L i p s c h i t z$ on the visited level $s e t ,$ then

$$
\mathcal { L } _ { \beta } ( \boldsymbol { \theta } ^ { k } , \boldsymbol { V } ^ { k } , \lambda ^ { k } ) - \mathcal { L } _ { \beta } ( \boldsymbol { \theta } ^ { k + 1 } , \boldsymbol { V } ^ { k } , \lambda ^ { k } ) \geq \left( \frac { 1 } { 2 \eta _ { \boldsymbol { \theta } , \boldsymbol { k } } } - \frac { L _ { \boldsymbol { \theta } , \boldsymbol { k } } } { 2 } \right) \| \Delta \theta _ { k + 1 } \| ^ { 2 } .
$$

Proof. Optimality of the proximal-gradient subproblem, compared with $\theta ^ { k }$ , gives

$$
\phi ( \theta ^ { k + 1 } ) - \phi ( \theta ^ { k } ) + \langle \nabla h _ { k } ( \theta ^ { k } ) , \Delta \theta _ { k + 1 } \rangle + \frac { 1 } { 2 \eta _ { \theta , k } } \| \Delta \theta _ { k + 1 } \| ^ { 2 } \leq 0 .
$$

The descent lemma gives $h _ { k } ( \theta ^ { k + 1 } ) - h _ { k } ( \theta ^ { k } ) \leq \langle \nabla h _ { k } ( \theta ^ { k } ) , \Delta \theta _ { k + 1 } \rangle + ( L _ { \theta , k } / 2 ) \| \Delta \theta _ { k + 1 } \| ^ { 2 }$ . Adding the two inequalities proves the claim. □

Role. This lemma replaces the exact policy-subproblem decrease in Theorem E.7 and makes the nonlinear-policy algorithm implementable by one first-order step.

Theorem E.9 (Implementable nonlinear-policy convergence and complexity). Suppose the valuestep parameter conditions of Theorem E.7 hold and the policy step is the proximal-gradient update above. Assume $D _ { \theta } M _ { \theta } , D _ { \theta } r _ { \theta }$ are uniformly bounded on the visited region and $\nabla h _ { k }$ is $L _ { \theta , k } – L i p s c h i t z$ on every policy comparison segment, with

$$
0 < \underline { { { \eta } } } \le \eta _ { \theta , k } , \qquad L _ { \theta , k } \le \overline { { { L } } } < \infty .
$$

If

$$
\frac { 1 } { 2 \eta _ { \theta , k } } - \frac { L _ { \theta , k } } { 2 } - \frac { A _ { A } } { \beta } \geq \alpha _ { \theta } > 0
$$

for all k, then the corrected Lyapunov decreases and the KKT residual converges to zero. Moreover there is $C _ { G } < \infty$ such that

$$
G _ { k + 1 } ^ { A } \leq C _ { G } \left( \| \Delta \theta _ { k + 1 } \| ^ { 2 } + \| \Delta V _ { k + 1 } \| ^ { 2 } + \| \Delta V _ { k } \| ^ { 2 } \right) ,
$$

so $\begin{array} { r } { \operatorname* { m i n } _ { 1 \le k \le T } G _ { k } ^ { A } \le C / T } \end{array}$

Proof. Replace Lemma E.3 by Lemma E.8 in the proof of Theorem E.7. The policy coeficient after dual ascent becomes $1 / ( 2 \eta _ { \theta , k } ) - L _ { \theta , k } / 2 - A _ { A } / \beta .$ which is uniformly at least $\alpha _ { \theta }$ . The value coeficients and Lyapunov lower bound are unchanged. Hence the three increment-square sequences are summable. The value and multiplier iterates are bounded independently of this descent argument: the recursion in Lemma 4.2 applies with bounded deterministic rewards, and the value margins imply $\beta \eta _ { V } > 1 2 \kappa _ { M } ^ { 2 } > 4 \kappa _ { M } ^ { 2 }$

The proximal-gradient optimality condition is

$$
0 \in \partial \psi _ { \Theta } ( \theta ^ { k + 1 } ) + \nabla h _ { k } ( \theta ^ { k } ) + \eta _ { \theta , k } ^ { - 1 } \Delta \theta _ { k + 1 } .
$$

Comparison with true policy stationarity at $( \theta ^ { k + 1 } , V ^ { k + 1 } , \lambda ^ { k + 1 } )$ gives

$$
e _ { \theta , k + 1 } \leq ( \underline { { \eta } } ^ { - 1 } + \overline { { L } } ) \| \Delta \theta _ { k + 1 } \| + C \| \Delta V _ { k + 1 } \| .
$$

The second term follows from bounded value and multiplier iterates, bounded policy derivatives, and $\lambda ^ { k + 1 } - [ \lambda ^ { k } + \beta R ( \theta ^ { k + 1 } , V ^ { k } ) ] = \beta M _ { k + 1 } \Delta V _ { k + 1 }$ . The value residual is $\eta _ { V } ^ { - 1 } \lVert \Delta V _ { k + 1 } \rVert$ . Feasibility is controlled by Lemma E.5. Thus all coeficients in the residual bound are uniform in k and the residual converges to zero. Squaring gives the displayed $G _ { k + 1 } ^ { A }$ bound. Summing the Lyapunov decrease from 1 to $T$ gives a constant upper bound on the sum of the increment energies, and therefore

$$
\operatorname* { m i n } _ { 1 \leq k \leq T } G _ { k } ^ { A } \leq \frac { 1 } { T } \sum _ { k = 1 } ^ { T } G _ { k } ^ { A } \leq \frac { C } { T } .
$$

Role. This theorem is the practical nonlinear-policy extension used in the paper’s general operator-stability claim.

Uniform constants and backtracking. A compact convex parameter set and a twice continuously diferentiable policy map on its neighborhood give bounded first and second derivatives. Together with the value-dual bound above, they yield a uniform ${ \overline { { L } } } .$ Choose a fixed trial step $\eta _ { \mathrm { t r i a l } } > 0$ and a shrink factor $q \in ( 0 , 1 )$ . Restart backtracking at $\eta _ { \mathrm { t r i a l } }$ on every iteration and shrink until the descent condition with margin $A _ { A } / \beta + \alpha _ { \theta }$ holds. Every step at most $\eta _ { \star } = ( \overline { { L } } + 2 A _ { A } / \beta + 2 \alpha _ { \theta } ) ^ { - 1 }$ is acceptable. Therefore the accepted steps satisfy $\eta _ { \theta , k } \geq \operatorname* { m i n } \{ \eta _ { \mathrm { t r i a l } } , q \eta _ { \star } \} > 0$

## F Projected Bellman Equations with Function Approximation

Let $V _ { w } = \Xi w$ with a full-column-rank $\Xi \in \mathbb { R } ^ { | S | \times m }$ and $m < | S |$ . Fix a positive diagonal matrix D and define the D-orthogonal projection

$$
\Pi _ { D } = \Xi ( \Xi ^ { \mathsf { T } } D \Xi ) ^ { - 1 } \Xi ^ { \mathsf { T } } D .
$$

The projected Bellman equation $\Xi w = \Pi _ { D } T _ { \theta } ( \Xi w )$ is equivalent to

$$
G ( \theta , w ) = A _ { \theta } w - b _ { \theta } = 0 , \qquad A _ { \theta } = \Xi ^ { \mathsf { T } } D M _ { \theta } \Xi , \qquad b _ { \theta } = \Xi ^ { \mathsf { T } } D r _ { \theta } .
$$

Assume sup $\| A _ { \theta } ^ { - 1 } \| \leq \kappa _ { A }$ and $\| A _ { \theta } - A _ { \theta ^ { \prime } } \| \leq L _ { A } \| \theta - \theta ^ { \prime } \|$ . Let $c _ { w } = \Xi ^ { \mathsf { T } } c$ .

Lemma F.1 (Projected value-dual identity). If the w-subproblem uses a proximal term $( 2 \eta _ { w } ) ^ { - 1 } \Vert w -$ $w ^ { k } \| ^ { 2 }$ and the projected multiplier update is

$$
\nu ^ { k + 1 } = \nu ^ { k } + \beta ( A _ { k + 1 } w ^ { k + 1 } - b _ { k + 1 } ) ,
$$

then

$$
\nu ^ { k + 1 } = - A _ { k + 1 } ^ { - \mathsf { T } } \left( c _ { w } + \eta _ { w } ^ { - 1 } \Delta w _ { k + 1 } \right) .
$$

Proof. The w-subproblem first-order condition is

$$
0 = c _ { w } + A _ { k + 1 } ^ { \mathsf { T } } \nu ^ { k } + \beta A _ { k + 1 } ^ { \mathsf { T } } ( A _ { k + 1 } w ^ { k + 1 } - b _ { k + 1 } ) + \eta _ { w } ^ { - 1 } \Delta w _ { k + 1 } .
$$

The bracket $\nu ^ { k } + \beta ( A _ { k + 1 } w ^ { k + 1 } - b _ { k + 1 } )$ equals $\nu ^ { k + 1 }$ . Thus $A _ { k + 1 } ^ { \top } \nu ^ { k + 1 } = - ( c _ { w } + \eta _ { w } ^ { - 1 } \Delta w _ { k + 1 } )$ , and uniform invertibility gives the formula. □

Role. This identity restores the same multiplier representation that drives the Bellman-resolvent proof, now in the lower-dimensional projected space.

Theorem F.2 (Projected Bellman-ADMM convergence and complexity). Assume $A _ { \theta }$ is uniformly invertible and Lipschitz, $D _ { \theta } A _ { \theta } , D _ { \theta } b _ { \theta }$ are bounded, and the projected reduced objective is bounded below. Use the composite policy regularizer ψ<sub>Θ</sub> and a deterministic feasible-domain initialization as in Appendix E. Require all conditions of Theorem E.9 after the replacements below, including the value-step margins, $\operatorname* { i n f } _ { k } \eta _ { \theta , k } > 0$ , and a uniform Lipschitz constant for the projected smooth augmented gradient. Then a corrected projected Lyapunov function decreases, the projected KKT residual converges to zero, and the minimum squared projected KKT residual satisfies $O ( T ^ { - 1 } )$ .

Proof. We verify that every ingredient of Theorem E.9 transfers under the replacements

$$
M _ { \theta } \mapsto A _ { \theta } , \qquad V \mapsto w , \qquad \lambda \mapsto \nu , \qquad r _ { \theta } \mapsto b _ { \theta } , \qquad c \mapsto c _ { w } .
$$

First, the inverse identity and the assumptions give

$$
\lVert A _ { \theta } ^ { - 1 } - A _ { \theta ^ { \prime } } ^ { - 1 } \rVert \leq \kappa _ { A } ^ { 2 } L _ { A } \lVert \theta - \theta ^ { \prime } \rVert .
$$

Second, Lemma F.1 is exactly the value-dual identity needed to compare consecutive multipliers. Subtraction therefore gives a dual-increment estimate

$$
\begin{array} { r } { \| \Delta \nu _ { k + 1 } \| ^ { 2 } \leq A _ { B } \| \Delta \theta _ { k + 1 } \| ^ { 2 } + B _ { B } \| \Delta w _ { k + 1 } \| ^ { 2 } + B _ { B } \| \Delta w _ { k } \| ^ { 2 } } \end{array}
$$

for finite constants $A _ { B } , B _ { B }$ . Third, the projected w comparison step yields $( 2 \eta _ { w } ) ^ { - 1 } \| \Delta w _ { k + 1 } \| ^ { 2 }$ decrease, and the policy step gives the same proximal-gradient decrease as Lemma E.8. Fourth, substituting $w ^ { k } = A _ { k } ^ { - 1 } ( b _ { k } + G _ { k } )$ and $\nu ^ { k } = - A _ { k } ^ { - \bar { \mathsf { T } } } ( c _ { w } + \eta _ { w } ^ { - 1 } \Delta w _ { k } )$ into the projected augmented Lagrangian gives the same lower-bound calculation as Lemma E.6, with $\kappa _ { M } , \eta _ { V }$ replaced by $\kappa _ { A } , \eta _ { w } .$

Under the strict parameter margins, the resulting corrected Lyapunov therefore has positive coeficients on $\| \Delta \theta _ { k + 1 } \| ^ { 2 } , \ \| \Delta w _ { k + 1 } \| ^ { 2 }$ , and $\| \Delta w _ { k } \| ^ { 2 }$ . Summation gives square summability of the increments, the multiplier identity gives projected feasibility, and the policy/w optimality conditions give convergence of the projected KKT residual. The same residual-to-increment argument as in Theorem E.9 yields $G _ { k + 1 } ^ { B } \ \leq \ C D _ { k + 1 } ^ { B }$ , where the sum of $D _ { k + 1 } ^ { B }$ is bounded. Hence $\mathrm { m i n } _ { k \le T } G _ { k } ^ { B } \ \le$ $C / T$ □

Role. The theorem shows that exact state-value representation is not required for the primal-dual stability mechanism. What is required is an invertible projected Bellman matrix.

Assume now that the projected Bellman map is contractive in a chosen norm: for some $q < 1$

$$
\Vert \Pi _ { D } T _ { \theta } U - \Pi _ { D } T _ { \theta } V \Vert \leq q \Vert U - V \Vert .
$$

Let $\bar { V } ^ { \theta } = \Xi w ^ { \theta }$ be its fixed point.

Lemma F.3 (Projected value approximation). $W i t h \varepsilon _ { \mathrm { v a l } } ( \theta ) = \| ( I - \Pi _ { D } ) V ^ { \theta } \|$

$$
\lVert \bar { V } ^ { \theta } - V ^ { \theta } \rVert \leq \frac { \varepsilon _ { \mathrm { v a l } } ( \theta ) } { 1 - q } .
$$

Proof. Use $\bar { V } ^ { \theta } = \Pi _ { D } T _ { \theta } \bar { V } ^ { \theta }$ and $V ^ { \theta } = T _ { \theta } V ^ { \theta }$

$$
\begin{array} { r l } & { \| \bar { \boldsymbol { V } } ^ { \theta } - \boldsymbol { V } ^ { \theta } \| \leq \| \Pi _ { D } T _ { \theta } \bar { \boldsymbol { V } } ^ { \theta } - \Pi _ { D } T _ { \theta } \boldsymbol { V } ^ { \theta } \| + \| \Pi _ { D } \boldsymbol { V } ^ { \theta } - \boldsymbol { V } ^ { \theta } \| } \\ & { \qquad \leq q \| \bar { \boldsymbol { V } } ^ { \theta } - \boldsymbol { V } ^ { \theta } \| + \varepsilon _ { \mathrm { v a l } } ( \theta ) . } \end{array}
$$

Move the first term to the left.

Role. This lemma quantifies the value-representation error, but the next proposition shows why this alone is insuficient for policy-gradient accuracy.

Proposition F.4 (Value error does not control parameter-derivative error). There exists a smooth policy family for which

$$
\operatorname* { s u p } _ { \theta } \| \bar { V } ^ { \theta } - V ^ { \theta } \| = \varepsilon , \qquad \operatorname* { s u p } _ { \theta } \| D _ { \theta } \bar { V } ^ { \theta } - D _ { \theta } V ^ { \theta } \| = \sqrt { \varepsilon } .
$$

Thus no constant independent $o f \varepsilon$ can bound the derivative error linearly by the value error.

Proof. Consider a one-step terminating MDP with two states. Let the approximation class represent the first state’s value exactly but force the second coordinate to zero. At the second state there are two actions with rewards $+ 1$ and −1. Define

$$
\pi _ { \theta } ( + \mid s _ { 2 } ) = \frac { 1 } { 2 } + \frac { \varepsilon } { 2 } \sin ( \theta / \sqrt { \varepsilon } ) .
$$

Then the true second-state value is $V _ { 2 } ^ { \theta } = { \varepsilon } \sin ( \theta / \sqrt { \varepsilon } )$ , while the projected value is $\bar { V } _ { 2 } ^ { \theta } = 0$ . Hence the uniform value error is $\varepsilon .$ . But

$$
\frac { d } { d \theta } V _ { 2 } ^ { \theta } = \sqrt { \varepsilon } \cos ( \theta / \sqrt { \varepsilon } ) ,
$$

whose supremum is $\sqrt { \varepsilon }$ . Therefore a uniform inequality $\lVert D _ { \theta } \bar { V } - D _ { \theta } V \rVert \leq C \lVert \bar { V } - V \rVert$ would require $\sqrt { \varepsilon } \leq C \varepsilon$ for arbitrarily small $\varepsilon ,$ which is impossible. □

Role. This negative result motivates the adjoint-based error decomposition in Lemmas F.5 and F.6.

Let the true adjoint satisfy $M _ { \theta } ^ { \mathsf { T } } \lambda ^ { \theta } = - c$ . Let $\nu ^ { \theta }$ solve the projected adjoint equation and define the lifted projected dual $\widehat { \lambda } ^ { \theta } = D \Xi \nu ^ { \theta }$

Lemma F.5 (Adjoint-gradient approximation). Suppose $D _ { \theta } M _ { \theta }$ and $D _ { \theta } r _ { \theta }$ are bounded on the relevant set. Then there are constants $C _ { V } , C _ { \lambda } < \infty$ such that

$$
\| D _ { \theta } R ( \theta , V ^ { \theta } ) ^ { \mathsf { T } } \lambda ^ { \theta } - D _ { \theta } R ( \theta , \bar { V } ^ { \theta } ) ^ { \mathsf { T } } \widehat { \lambda } ^ { \theta } \| \leq C _ { V } \| V ^ { \theta } - \bar { V } ^ { \theta } \| + C _ { \lambda } \| \lambda ^ { \theta } - \widehat { \lambda } ^ { \theta } \| .
$$

Proof. Add and subtract $D _ { \theta } R ( \theta , \bar { V } ^ { \theta } ) ^ { \mathsf { T } } \lambda ^ { \theta }$

$$
\begin{array} { r l } & { D _ { \theta } R ( \theta , V ^ { \theta } ) ^ { \mathsf { T } } \lambda ^ { \theta } - D _ { \theta } R ( \theta , \bar { V } ^ { \theta } ) ^ { \mathsf { T } } \widehat { \lambda } ^ { \theta } } \\ & { \ = \Big [ D _ { \theta } R ( \theta , V ^ { \theta } ) - D _ { \theta } R ( \theta , \bar { V } ^ { \theta } ) \Big ] ^ { \mathsf { T } } \lambda ^ { \theta } + D _ { \theta } R ( \theta , \bar { V } ^ { \theta } ) ^ { \mathsf { T } } ( \lambda ^ { \theta } - \widehat { \lambda } ^ { \theta } ) . } \end{array}
$$

Because $D _ { \theta } R ( \theta , V ) [ h ] = D _ { \theta } M _ { \theta } [ h ] V - D _ { \theta } r _ { \theta } [ h ]$ , the first bracket is linear in $V ^ { \theta } - \bar { V } ^ { \theta }$ and is bounded by $C _ { V } \lVert V ^ { \theta } - \bar { V } ^ { \theta } \rVert$ after using bounded $\lambda ^ { \theta }$ . Boundedness of $D _ { \theta } R$ gives the second term with constant $C _ { \lambda }$ □

Role. This lemma replaces the false derivative-from-value estimate by a correct value-plus-adjoint approximation bound.

Lemma F.6 (Lifted-dual quasi-optimality). There is $C _ { \mathrm { d u a l } } < \infty$ such that

$$
\| \lambda ^ { \theta } - \widehat { \lambda } ^ { \theta } \| \leq C _ { \mathrm { d u a l } } \operatorname* { i n f } _ { z \in \mathrm { r a n g e } ( D \Xi ) } \| \lambda ^ { \theta } - z \| .
$$

Proof. Take any $z = D \Xi a$ in the approximation space. The true adjoint equation and the projected adjoint equation imply

$$
A _ { \theta } ^ { \mathsf { T } } \nu ^ { \theta } = \Xi ^ { \mathsf { T } } M _ { \theta } ^ { \mathsf { T } } \lambda ^ { \theta } , \qquad A _ { \theta } ^ { \mathsf { T } } a = \Xi ^ { \mathsf { T } } M _ { \theta } ^ { \mathsf { T } } z .
$$

Subtracting gives

$$
\begin{array} { r } { \nu ^ { \theta } - a = A _ { \theta } ^ { - \mathsf { T } } \Xi ^ { \mathsf { T } } M _ { \theta } ^ { \mathsf { T } } ( \lambda ^ { \theta } - z ) . } \end{array}
$$

Therefore

$$
\begin{array} { r } { \| \widehat { \lambda } ^ { \theta } - z \| \leq \| D \Xi \| \| A _ { \theta } ^ { - \mathsf { T } } \Xi ^ { \mathsf { T } } M _ { \theta } ^ { \mathsf { T } } \| \| \lambda ^ { \theta } - z \| . } \end{array}
$$

By the triangle inequality,

$$
\begin{array} { r } { \| \boldsymbol \lambda ^ { \theta } - \widehat \lambda ^ { \theta } \| \leq \left( 1 + \| D \boldsymbol \Xi \| \operatorname* { s u p } _ { \theta } \| A _ { \theta } ^ { - \mathsf { T } } \boldsymbol \Xi ^ { \mathsf { T } } \boldsymbol M _ { \theta } ^ { \mathsf { T } } \| \right) \| \boldsymbol \lambda ^ { \theta } - \boldsymbol z \| . } \end{array}
$$

Minimize over $z \in$ range(DΞ).

Role. Together with Lemmas F.3 and F.5, this lemma gives a controlled route from projected-Bellman stationarity back to the true policy gradient.

## G Explicit Discounted Occupancy Formulation

Use the policy domain, composite regularizer $\psi _ { \Theta }$ , and initialization conventions of Appendix E. Let

$$
d _ { \theta } ( s ) = ( 1 - \gamma ) \sum _ { t \geq 0 } \gamma ^ { t } \mathbb { P } _ { \rho , \pi _ { \theta } } ( S _ { t } = s ) , \qquad q _ { \theta } ( s , a ) = d _ { \theta } ( s ) \pi _ { \theta } ( a \mid s ) .
$$

Let $P ^ { \mathsf { T } } q$ denote the state inflow and let $\Pi _ { \theta } d$ be the state-action vector with entries $\pi _ { \theta } ( a \mid s ) d ( s )$ The occupancy constraints are

$$
d - \gamma P ^ { \mathsf { T } } q = ( 1 - \gamma ) \rho , \qquad q - \Pi _ { \theta } d = 0 .
$$

Define

$$
x = \left[ \begin{array} { c } { { q } } \\ { { d } } \end{array} \right] , \qquad b = \left[ \begin{array} { c } { { ( 1 - \gamma ) \rho } } \\ { { 0 } } \end{array} \right] , \qquad K _ { \theta } = \left[ \begin{array} { c c } { { - \gamma P ^ { \top } } } & { { I } } \\ { { I } } & { { - \Pi _ { \theta } } } \end{array} \right] .
$$

Lemma G.1 (Occupancy-operator invertibility). For every policy $\pi _ { \theta } , K _ { \theta }$ is invertible. If

$$
K _ { \theta } \left[ { q atop d } \right] = \left[ { u \atop v } \right] ,
$$

then

$$
d = ( I - \gamma P _ { \theta } ^ { \mathsf { T } } ) ^ { - 1 } ( u + \gamma P ^ { \mathsf { T } } v ) , \qquad q = v + \Pi _ { \theta } d .
$$

In the block $\ell _ { 1 }$ norm, $\lVert K _ { \theta } ^ { - 1 } \rVert _ { 1 , \oplus } \leq 2 / ( 1 - \gamma )$ . Hence sup<sub>θ</sub> $\| K _ { \theta } ^ { - 1 } \| _ { 2 } < \infty$

Proof. The second block row gives $q = v + \Pi _ { \theta } d .$ Substitute this into the first row:

$$
- \gamma P ^ { \mathsf { T } } ( v + \Pi _ { \theta } d ) + d = u .
$$

Since $P ^ { \mathsf { T } } \Pi _ { \theta } = P _ { \theta } ^ { \mathsf { T } }$

$$
( I - \gamma P _ { \theta } ^ { \mathsf { T } } ) d = u + \gamma P ^ { \mathsf { T } } v ,
$$

which yields the stated formulas because $I - \gamma P _ { \theta } ^ { \mathsf { T } }$ is invertible.

For the norm bound, $\| ( I - \gamma P _ { \theta } ^ { \mathsf { T } } ) ^ { - 1 } \| _ { 1 } \leq ( 1 - \gamma ) ^ { - 1 } , \| P ^ { \mathsf { T } } \| _ { 1 } = 1$ , and $\| \Pi _ { \theta } \| _ { 1 } = 1$ . Hence

$$
\| d \| _ { 1 } \leq \frac { \| u \| _ { 1 } + \gamma \| v \| _ { 1 } } { 1 - \gamma } , \qquad \| q \| _ { 1 } \leq \| v \| _ { 1 } + \| d \| _ { 1 } .
$$

Adding and simplifying gives a valid bound no larger than $2 ( 1 - \gamma ) ^ { - 1 } ( \| u \| _ { 1 } + \| v \| _ { 1 } )$ after absorbing the fixed coeficients. Finite-dimensional norm equivalence gives the Euclidean bound. □

Role. This lemma shows that discounted occupancy flow provides a second RL-specific invertible operator, independent of the Bellman value representation.

Proposition G.2 (Occupancy equivalence). The equation $K _ { \theta } x = b$ has the unique solution $x ^ { \theta } =$ $( q ^ { \theta } , d ^ { \theta } )$ equal to the discounted state-action and state occupancies. Moreover $q ^ { \theta } \geq 0$ and $d ^ { \theta } \geq 0$

Proof. Apply Lemma G.1 with $u = ( 1 - \gamma ) \rho$ and $v = 0 \mathrm { : }$

$$
d = ( 1 - \gamma ) ( I - \gamma P _ { \theta } ^ { \mathsf { T } } ) ^ { - 1 } { \rho } , \qquad q = \Pi _ { \theta } d .
$$

The first expression is exactly the normalized discounted state occupancy and the second is the corresponding state-action occupancy. Uniqueness follows from invertibility of $K _ { \theta }$ . The Neumann series for $( I - \gamma P _ { \theta } ^ { \mathsf { T } } ) ^ { - 1 }$ is entrywise nonnegative, so $d \geq 0$ , and then $q = \Pi _ { \theta } d \geq 0$ □

Role. The proposition establishes that the square operator formulation is exactly equivalent to the usual discounted occupancy definition, rather than a relaxation.

Let

$$
c _ { x } = \left[ \begin{array} { c } { - r / ( 1 - \gamma ) } \\ { 0 } \end{array} \right]
$$

and consider

$$
\operatorname* { m i n } _ { \theta , x } ~ \phi ( \theta ) + c _ { x } ^ { \mathsf { T } } x ~ \mathrm { s . t . } ~ K _ { \theta } x = b .
$$

Theorem G.3 (KKT equivalence). Define the reduced objective $f ( \theta ) = \phi ( \theta ) + c _ { x } ^ { \mathsf { T } } K _ { \theta } ^ { - 1 } b$ . A triple $( \theta , x , y )$ satisfies the KKT conditions of the occupancy-constrained problem if and only if

$$
\begin{array} { r } { x = K _ { \theta } ^ { - 1 } b , \qquad y = - K _ { \theta } ^ { - \mathsf { T } } c _ { x } , \qquad 0 \in \partial ( f + \delta _ { \Theta } ) ( \theta ) . } \end{array}
$$

Proof. By Proposition G.2, feasibility identifies x with the true discounted occupancy. The feasibility and x-stationarity conditions of the constrained problem are

$$
K _ { \theta } x = b , \qquad c _ { x } + K _ { \theta } ^ { \mathsf { T } } y = 0 .
$$

Lemma G.1 gives $x = K _ { \theta } ^ { - 1 } b$ and $y = - K _ { \theta } ^ { - \mathsf { T } } c _ { x }$ . Diferentiate the identity $K _ { \theta } x ^ { \theta } = b$ in direction h:

$$
D _ { \theta } x ^ { \theta } [ h ] = - K _ { \theta } ^ { - 1 } D _ { \theta } K _ { \theta } [ h ] x ^ { \theta } .
$$

For the smooth reduced part $\psi ( \theta ) = c _ { x } ^ { \mathsf { T } } x ^ { \theta }$

$$
\begin{array} { c } { { D \psi ( \theta ) [ h ] = - c _ { x } ^ { \mathsf { T } } K _ { \theta } ^ { - 1 } D _ { \theta } K _ { \theta } [ h ] x ^ { \theta } } } \\ { { = \langle y ^ { \theta } , D _ { \theta } K _ { \theta } [ h ] x ^ { \theta } \rangle . } } \end{array}
$$

Thus the reduced first-order condition $0 \in \partial \psi _ { \Theta } ( \theta ) + \nabla \psi ( \theta )$ is exactly the $\theta \mathrm { - }$ -stationarity condition of the constrained problem. This proves both directions. □

Role. This theorem verifies that the occupancy KKT residual has the same policy meaning as the reduced parameterized objective.

The deterministic occupancy-ADMM performs a proximal-gradient θ step, then

$$
x ^ { k + 1 } = \arg \operatorname* { m i n } _ { x } \left\{ \mathcal { L } _ { \beta } ( \theta ^ { k + 1 } , x , y ^ { k } ) + \frac { 1 } { 2 \eta _ { x } } \| x - x ^ { k } \| ^ { 2 } \right\} , \qquad y ^ { k + 1 } = y ^ { k } + \beta ( K _ { k + 1 } x ^ { k + 1 } - b ) .
$$

Lemma G.4 (Occupancy-dual identity). The occupancy step and dual update satisfy

$$
y ^ { k + 1 } = - K _ { k + 1 } ^ { - \mathsf { T } } \left( c _ { x } + \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } \right) .
$$

Proof. The x-subproblem first-order condition is

$$
0 = c _ { x } + K _ { k + 1 } ^ { \mathsf { T } } y ^ { k } + \beta K _ { k + 1 } ^ { \mathsf { T } } ( K _ { k + 1 } x ^ { k + 1 } - b ) + \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } .
$$

The bracket $y ^ { k } + \beta ( K _ { k + 1 } x ^ { k + 1 } - b )$ equals $y ^ { k + 1 }$ . Thus $K _ { k + 1 } ^ { \mathsf { T } } y ^ { k + 1 } = - ( c _ { x } + \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } )$ , and Lemma G.1 gives the formula. □

Role. This is the occupancy analogue of the Bellman value-dual identity and is the key input to the next theorem.

Theorem G.5 (Deterministic occupancy-ADMM convergence). Assume $\theta \mapsto K _ { \theta }$ is Lipschitz, $D _ { \theta } K _ { \theta }$ is bounded on the visited set, and the policy proximal-gradient and $( x , y )$ parameters satisfy the strict margins analogous to Theorem E.9. Require the occupancy reduced objective to be bounded below, a positive lower bound on the policy steps, and a uniform Lipschitz bound for the smooth augmented policy gradients, with all constants and value-step margins taken for $K _ { \theta }$ . Then there exists a corrected Lyapunov function

$$
\Phi _ { k } ^ { C } = \mathcal { L } _ { \beta } ( \theta ^ { k } , x ^ { k } , y ^ { k } ) + \tau \| \Delta x _ { k } \| ^ { 2 }
$$

with positive one-step decrease in $\| \Delta \theta _ { k + 1 } \| ^ { 2 } , \ \| \Delta x _ { k + 1 } \| ^ { 2 }$ , and $\| \Delta x _ { k } \| ^ { 2 }$ . Consequently the true occupancy KKT residual converges to zero.

Proof. The proof instantiates the nonlinear-policy argument with the operator $K _ { \theta }$ . Lemma G.1 supplies a uniform inverse bound and the inverse-map identity gives

$$
\lVert K _ { \theta } ^ { - 1 } - K _ { \theta ^ { \prime } } ^ { - 1 } \rVert \leq \kappa _ { K } ^ { 2 } L _ { K } \lVert \theta - \theta ^ { \prime } \rVert .
$$

Lemma G.4 therefore implies a dual-increment bound

$$
\begin{array} { r } { \| \Delta y _ { k + 1 } \| ^ { 2 } \leq A _ { C } \| \Delta \theta _ { k + 1 } \| ^ { 2 } + B _ { C } \| \Delta x _ { k + 1 } \| ^ { 2 } + B _ { C } \| \Delta x _ { k } \| ^ { 2 } . } \end{array}
$$

The proximal-gradient policy step gives the same descent form as Lemma $\mathrm { E } . 8$ , while the exact x step gives $( 2 \eta _ { x } ) ^ { - 1 } \| \Delta x _ { k + 1 } \| ^ { 2 }$ decrease. The exact dual update contributes $\beta ^ { - 1 } \| \Delta y _ { k + 1 } \| ^ { 2 }$ ascent. Under the stated strict margins these terms leave positive coeficients after adding $\tau \| \Delta x _ { k } \| ^ { 2 } - \tau \| \Delta x _ { k + 1 } \| ^ { 2 }$

For the lower bound, write $x ^ { k } = K _ { k } ^ { - 1 } ( b + F _ { k } )$ with $F _ { k } = K _ { k } x ^ { k } - b$ and use Lemma G.4 at the preceding step. The same Young-inequality calculation as Lemma E.6 bounds $\Phi _ { k } ^ { C }$ below by the reduced occupancy objective plus a positive multiple of $\| F _ { k } \| ^ { 2 }$ and $\| \Delta x _ { k } \| ^ { 2 }$ . Summation forces all increments to zero. Since $\Delta y _ { k + 1 } = \beta F _ { k + 1 }$ , feasibility follows. The policy and occupancy first-order residuals are linear in the adjacent increments because $D _ { \theta } K _ { \theta }$ is bounded. Hence the true occupancy KKT residual converges to zero. □

Role. The theorem establishes a second complete deterministic ADMM theory supported by discounted RL operator invertibility.

Lemma G.6 (Occupancy residual controls primal error). Let $x ^ { \theta } = K _ { \theta } ^ { - 1 } b$ . Then

$$
\| x - x ^ { \theta } \| \leq \kappa _ { K } \| K _ { \theta } x - b \| , \qquad \kappa _ { K } = \operatorname* { s u p } _ { \theta } \| K _ { \theta } ^ { - 1 } \| .
$$

Proof. Subtract $K _ { \theta } x ^ { \theta } = b$ from $K _ { \theta } x \mathrm { : }$

$$
x - x ^ { \theta } = K _ { \theta } ^ { - 1 } ( K _ { \theta } x - b ) .
$$

Take norms and use the uniform inverse bound.

Role. This error bound is used in the stochastic occupancy analysis to control the size of the occupancy variable by the true feasibility residual.

## H Stochastic Occupancy-ADMM with a Shared Empirical Operator

The occupancy gradient contains $K ^ { \mathsf { T } } ( K x - b )$ , and its empirical quadratic term generally satisfies $\mathbb { E } [ \widehat { K } ^ { \top } \widehat { K } ] \neq K ^ { \top } K$ . Iteration k constructs one empirical environment matrix $\widehat { P } _ { k }$ and uses its induced occupancy operator consistently in the coupled updates.

Assume the reset distribution and all policies used for sampling satisfy $\rho ( s ) \geq \rho _ { \mathrm { m i n } } > 0$ and $\pi _ { \theta } ( a \mid s ) \geq \pi _ { \operatorname* { m i n } } > 0$ . Assume $m _ { k } \geq 2$ and fresh sampling innovations as in Assumption 3.2. Use ψ<sub>Θ</sub> from Appendix E, a lower-bounded reduced objective, known bounded rewards, and deterministic initial variables with $\theta ^ { 0 } \in \mathrm { d o m } \psi _ { \Theta }$ . Split a length-m<sub>k</sub> trajectory into $L _ { k } = \lfloor m _ { k } / 2 \rfloor$ nonoverlapping two-step blocks. An anchor block is one in which the first step resets and the second step performs an environment transition.

Lemma H.1 (Regenerative transition estimator). Define $p _ { 0 } = \gamma ( 1 - \gamma )$ and, for each block $j ,$

$$
Y _ { j } ( s , a , s ^ { \prime } ) = \frac { \mathbf { 1 } \{ a n c h o r , S = s , A = a , S ^ { \prime } = s ^ { \prime } \} } { p _ { 0 } \rho ( s ) \pi _ { \theta ^ { k } } ( a \mid s ) } .
$$

Conditioned on $\mathcal { F } _ { k }$ , the random tensors $Y _ { j }$ are iid and $\mathbb { E } _ { k } Y _ { j } ( s , a , s ^ { \prime } ) ~ = ~ P ( s ^ { \prime } ~ \mid ~ s , a )$ . Let $\bar { P } _ { k } ~ =$ $L _ { k } ^ { - 1 } \sum _ { j } Y _ { j }$ and project each row of $\bar { P } _ { k }$ onto the probability simplex to obtain $\widehat { P } _ { k }$ . Then there are constants $C _ { P } , c _ { P } > 0$ such that

$$
\mathbb { E } _ { k } \Vert \widehat { P } _ { k } - P \Vert _ { F } ^ { 2 } \leq \frac { C _ { P } } { m _ { k } } , \qquad \mathbb { P } _ { k } ( \Vert \widehat { P } _ { k } - P \Vert _ { F } \geq t ) \leq 2 | { \cal S } | ^ { 2 } | { \cal A } | e ^ { - c _ { P } m _ { k } t ^ { 2 } } .
$$

Proof. Diferent blocks use disjoint reset coins, reset states, action randomization, and environment transition noise. If the first step of a block is not a reset, the corresponding $Y _ { j }$ is zero, so the inherited chain state does not enter the nonzero sample. Hence the $Y _ { j }$ are conditionally iid.

An anchor has probability $p _ { 0 }$ . Given an anchor, $S \sim \rho , A \sim \pi _ { \theta ^ { k } } ( \cdot \mid S )$ , and $S ^ { \prime } \sim P ( \cdot \mid S , A )$ Therefore

$$
\mathbb { E } _ { k } Y _ { j } ( s , a , s ^ { \prime } ) = \frac { p _ { 0 } \rho ( s ) \pi _ { \theta ^ { k } } ( a \\mid s ) P ( s ^ { \prime } \mid s , a ) } { p _ { 0 } \rho ( s ) \pi _ { \theta ^ { k } } ( a \mid s ) } = P ( s ^ { \prime } \mid s , a ) .
$$

Moreover $0 \le Y _ { j } ( s , a , s ^ { \prime } ) \le B = [ p _ { 0 } \rho _ { \operatorname* { m i n } } \pi _ { \operatorname* { m i n } } ] ^ { - 1 }$ . Thus, for every coordinate, Hoefding’s inequality gives

$$
\mathbb { P } _ { k } ( \vert \bar { P } _ { k } ( s ^ { \prime } \mid s , a ) - P ( s ^ { \prime } \mid s , a ) \vert \ge u ) \le 2 e ^ { - 2 L _ { k } u ^ { 2 } / B ^ { 2 } } .
$$

$\mathrm { A }$ union bound over at most $| { \cal S } | ^ { 2 } | { \cal A } |$ coordinates and the implication $\begin{array} { r } { \| A \| _ { F } \geq t \Rightarrow \operatorname* { m a x } _ { i } | A _ { i } | \geq } \end{array}$ $t / \sqrt { | S | ^ { 2 } | A | }$ gives the stated Frobenius tail with a modified constant $c _ { P }$ . Independence and bounded variance also give $\mathbb { E } _ { k } \Vert \bar { P } _ { k } - P \Vert _ { F } ^ { 2 } \leq C / L _ { k } \leq C _ { P } / m _ { k }$

Each true transition row lies in the probability simplex. Euclidean projection onto a closed convex set is nonexpansive, so rowwise $\| \Pi _ { \Delta } ( \bar { p } ) - p \| _ { 2 } \leq \| \bar { p } - p \| _ { 2 }$ . Summing over rows preserves both the mean-square and tail orders for $\widehat { P } _ { k }$ □

Role. This estimator gives an empirical transition matrix that is both statistically controlled and exactly row stochastic on every sample path. The latter property is essential for deterministic invertibility of the empirical occupancy operator.

Let $\widehat { K } _ { k } ( \theta )$ be obtained from $K _ { \theta }$ by replacing $P$ with $\widehat { P } _ { k }$ . Define

$$
E _ { k } ( \theta ) = \widehat { K } _ { k } ( \theta ) - K _ { \theta } , \qquad \delta _ { k } = \operatorname* { s u p } _ { \theta } \| E _ { k } ( \theta ) \| .
$$

Since the environment-transition block does not depend on $\theta , D _ { \theta } E _ { k } ( \theta ) = 0$ . Lemma H.1 implies $\mathbb { E } _ { k } \delta _ { k } ^ { 2 } \le C _ { E } / m _ { k }$

Lemma H.2 (Empirical occupancy-operator invertibility). For every row-stochastic $\widehat { P } _ { k }$ and every policy, $\widehat { K } _ { k } ( \theta )$ is invertible with a deterministic bound su $\begin{array} { r } { \operatorname { \delta _ { \boldsymbol { k } , \boldsymbol { \theta } } } \| \widehat { K } _ { k } ( \boldsymbol { \theta } ) ^ { - 1 } \| \leq \bar { \kappa } _ { K } < \infty } \end{array}$ . Moreover

$$
\lVert \widehat { K } _ { k } ( \theta ) ^ { - 1 } - K _ { \theta } ^ { - 1 } \rVert \leq \kappa _ { K } \bar { \kappa } _ { K } \delta _ { k } .
$$

Proof. The proof of Lemma G.1 uses only row stochasticity of the environment transition matrix. Since every row of $\widehat { P } _ { k }$ is projected onto the simplex, the same Schur-complement argument gives invertibility of $\widehat { K } _ { k } ( \theta )$ and the same block- ${ \bf - } \ell _ { 1 }$ inverse bound, hence a deterministic Euclidean bound $\bar { \kappa } _ { K }$

For the perturbation estimate, use

$$
\widehat K _ { k } ^ { - 1 } - { \cal K } _ { \theta } ^ { - 1 } = { \cal K } _ { \theta } ^ { - 1 } ( K _ { \theta } - \widehat K _ { k } ) \widehat K _ { k } ^ { - 1 } .
$$

Taking norms and using the two uniform inverse bounds gives the result.

Role. This lemma removes any small-error invertibility event. Empirical primal-dual updates are well defined on every sample path.

Let $\widehat F _ { k } ( \theta , x ) = \widehat K _ { k } ( \theta ) x - b$ . The stochastic occupancy-ADMM uses one shared $\widehat { K } _ { k }$ in all updates.

Lemma H.3 (Empirical occupancy-dual identity). If the x update minimizes

$$
c _ { x } ^ { \mathsf { T } } x + \langle y ^ { k } , { \widehat { F } } _ { k } ( \theta ^ { k + 1 } , x ) \rangle + { \frac { \beta } { 2 } } \| { \widehat { F } } _ { k } ( \theta ^ { k + 1 } , x ) \| ^ { 2 } + { \frac { 1 } { 2 \eta _ { x } } } \| x - x ^ { k } \| ^ { 2 }
$$

and $y ^ { k + 1 } = y ^ { k } + \beta \widehat { F } _ { k } ( \theta ^ { k + 1 } , x ^ { k + 1 } )$ , then

$$
y ^ { k + 1 } = - \widehat { K } _ { k } ( \theta ^ { k + 1 } ) ^ { - \mathsf { T } } \left( c _ { x } + \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } \right) .
$$

Proof. The first-order condition of the x subproblem is

$$
\begin{array} { r } { 0 = c _ { x } + \widehat K _ { k } ( \theta ^ { k + 1 } ) ^ { \mathsf T } \left[ y ^ { k } + \beta \widehat F _ { k } ( \theta ^ { k + 1 } , x ^ { k + 1 } ) \right] + \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } . } \end{array}
$$

The bracket equals $y ^ { k + 1 }$ by the shared empirical dual update. Lemma H.2 allows inversion. □

Role. Sharing the same empirical operator preserves an exact empirical multiplier representation, which is the stochastic counterpart of the Bellman value-dual identity.

Uniform bounds and parameter conditions. Let $\bar { \kappa } _ { K }$ bound both true and empirical inverse operators. Assume $D _ { \theta } K _ { \theta }$ is bounded and the smooth augmented policy gradients have a common Lipschitz constant $\overline { { L } }$ on all policy comparison segments. Require

$$
\beta > 3 6 \bar { \kappa } _ { K } ^ { 2 } / \eta _ { x } , \qquad \tau = 1 / ( 8 \eta _ { x } ) , \qquad 0 < \underline { { \eta } } \le \eta _ { \theta , k } ,
$$

$$
\frac { 1 } { 2 \eta _ { \theta , k } } - \frac { \overline { { { L } } } } { 2 } - \frac { 3 A _ { C } } { 2 \beta } \geq a _ { \theta } > 0 , \qquad A _ { C } = 9 \bar { \kappa } _ { K } ^ { 4 } L _ { K } ^ { 2 } \| c _ { x } \| ^ { 2 } .
$$

These conditions can be met with a fixed suficiently small policy step. A compact convex parameter set and a twice continuously diferentiable policy map supply the required derivative bounds. In particular, the empirical dual identity and the dual update imply

$$
\| y ^ { k + 1 } \| \leq \bar { \kappa } _ { K } \| c _ { x } \| + \frac { \bar { \kappa } _ { K } } { \eta _ { x } } ( \| x ^ { k + 1 } \| + \| x ^ { k } \| ) , \qquad \| x ^ { k + 1 } \| \leq \bar { \kappa } _ { K } \| b \| + \frac { \bar { \kappa } _ { K } } { \beta } ( \| y ^ { k + 1 } \| + \| y ^ { k } \| ) .
$$

The two-dimensional recursion from Lemma 4.2, applied pathwise, has spectral radius less than one because $\beta \eta _ { x } > 4 \bar { \kappa } _ { K } ^ { 2 }$ . Thus $x ^ { k } , y ^ { k }$ have deterministic uniform bounds for every sampling path. This also yields a uniform smoothness bound under the compact-set suficient condition above. All subsequent two-iterate estimates apply for $k \geq 1$

Lemma H.4 (Empirical multiplier increment). Assume $\lVert K _ { \theta } - K _ { \theta ^ { \prime } } \rVert \leq L _ { K } \lVert \theta - \theta ^ { \prime } \rVert$ . One may take $A _ { C } = 9 \bar { \kappa } _ { K } ^ { 4 } L _ { K } ^ { 2 } \lVert \dot { c } _ { x } \rVert ^ { 2 } , B _ { C } = 3 \bar { \kappa } _ { K } ^ { 2 } / \eta _ { x } ^ { 2 }$ , and $D _ { C } = 9 \bar { \kappa } _ { K } ^ { 4 } \| c _ { x } \| ^ { 2 }$ , so that

$$
\begin{array} { r } { \| \Delta y _ { k + 1 } \| ^ { 2 } \leq A _ { C } \| \Delta \theta _ { k + 1 } \| ^ { 2 } + B _ { C } \| \Delta x _ { k + 1 } \| ^ { 2 } + B _ { C } \| \Delta x _ { k } \| ^ { 2 } + D _ { C } ( \delta _ { k } ^ { 2 } + \delta _ { k - 1 } ^ { 2 } ) . } \end{array}
$$

Proof. Subtract Lemma H.3 at iterations k and $k - 1$

$$
\begin{array} { r l } & { \Delta y _ { k + 1 } = \mathbf { \Theta } - \left( \widehat { K } _ { k } ( { \boldsymbol { \theta } } ^ { k + 1 } ) ^ { - \mathsf { T } } - \widehat { K } _ { k - 1 } ( { \boldsymbol { \theta } } ^ { k } ) ^ { - \mathsf { T } } \right) c _ { x } } \\ & { \qquad - \eta _ { x } ^ { - 1 } \widehat { K } _ { k } ( { \boldsymbol { \theta } } ^ { k + 1 } ) ^ { - \mathsf { T } } \Delta x _ { k + 1 } + \eta _ { x } ^ { - 1 } \widehat { K } _ { k - 1 } ( { \boldsymbol { \theta } } ^ { k } ) ^ { - \mathsf { T } } \Delta x _ { k } . } \end{array}
$$

The operator diference satisfies

$$
\begin{array} { r l } & { \| \widehat K _ { k } ( \theta ^ { k + 1 } ) - \widehat K _ { k - 1 } ( \theta ^ { k } ) \| \leq \| K _ { \theta ^ { k + 1 } } - K _ { \theta ^ { k } } \| + \| E _ { k } ( \theta ^ { k + 1 } ) \| + \| E _ { k - 1 } ( \theta ^ { k } ) \| } \\ & { \qquad \leq L _ { K } \| \Delta \theta _ { k + 1 } \| + \delta _ { k } + \delta _ { k - 1 } . } \end{array}
$$

Apply the inverse identity with the uniform empirical inverse bound to the first term. Then use $\| a + b + c \| ^ { 2 } \leq 3 ( \| a \| ^ { 2 } + \| b \| ^ { 2 } + \| c \| ^ { 2 } )$ and $( u + v + w ) ^ { 2 } \leq 3 ( u ^ { 2 } + v ^ { 2 } + w ^ { 2 } )$ . All fixed coeficients are absorbed into $A _ { C } , B _ { C } , D _ { C }$ □

Role. This is the stochastic occupancy analogue of multiplier control. It enters the dual-ascent term of Lemma H.7.

Let the true residual be $F _ { k } = K _ { \theta ^ { k } } x ^ { k }$ − b and the true augmented Lagrangian be

$$
\mathcal { L } _ { \beta } ^ { C } ( \boldsymbol { \theta } , \boldsymbol { x } , \boldsymbol { y } ) = \phi ( \boldsymbol { \theta } ) + c _ { x } ^ { \mathsf { T } } \boldsymbol { x } + \langle \boldsymbol { y } , K _ { \theta } \boldsymbol { x } - \boldsymbol { b } \rangle + \frac { \beta } { 2 } \| K _ { \theta } \boldsymbol { x } - \boldsymbol { b } \| ^ { 2 } .
$$

Define

$$
\begin{array} { r } { \Phi _ { k } = \mathcal { L } _ { \beta } ^ { C } ( \boldsymbol { \theta } ^ { k } , \boldsymbol { x } ^ { k } , \boldsymbol { y } ^ { k } ) + \tau \| \Delta x _ { k } \| ^ { 2 } , \qquad B _ { C } ( \boldsymbol { \theta } ) = \phi ( \boldsymbol { \theta } ) + c _ { x } ^ { \mathsf { T } } K _ { \boldsymbol { \theta } } ^ { - 1 } b . } \end{array}
$$

Lemma H.5 (Perturbed true Lyapunov lower bound). For suitable τ there are $q _ { x } > 0$ and $C _ { 0 } < \infty$ such that

$$
\Phi _ { k } \geq B _ { C } ^ { \mathrm { i n f } } + \frac \beta 4 \| F _ { k } \| ^ { 2 } + q _ { x } \| \Delta x _ { k } \| ^ { 2 } - C _ { 0 } \delta _ { k - 1 } ^ { 2 } .
$$

Therefore

$$
\Psi _ { k } = \Phi _ { k } - B _ { C } ^ { \mathrm { i n f } } + 1 + C _ { 0 } \delta _ { k - 1 } ^ { 2 }
$$

is nonnegative and satisfies $\Psi _ { k } \geq 1 + ( \beta / 4 ) \| F _ { k } \| ^ { 2 } + q _ { x } \| \Delta x _ { k } \| ^ { 2 }$

Proof. Write $x ^ { k } = K _ { k } ^ { - 1 } ( b + F _ { k } )$ . Lemma H.3 at the previous iteration gives

$$
y ^ { k } = - \widehat { K } _ { k - 1 } ( \theta ^ { k } ) ^ { - \mathsf { T } } \big ( c _ { x } + \eta _ { x } ^ { - 1 } \Delta x _ { k } \big ) .
$$

Substitute these into the true augmented Lagrangian. The reduced objective contributes $B _ { C } ( \theta ^ { k } ) \geq$ $B _ { C } ^ { \mathrm { i n f } }$ . The terms involving $\Delta x _ { k }$ and $F _ { k }$ have the same form as in Lemma E.6. The additional mismatch is produced by $\widehat { K } _ { k - 1 } ( \theta ^ { k } ) ^ { - 1 } - K _ { \theta ^ { k } } ^ { - 1 }$ , whose norm is at most $\kappa _ { K } \bar { \kappa } _ { K } \delta _ { k - 1 }$ by Lemma H.2. Thus, for fixed constants,

$$
\Phi _ { k } \geq B _ { C } ^ { \operatorname* { i n f } } + \frac { \beta } { 2 } \| F _ { k } \| ^ { 2 } - C _ { 1 } \| \Delta x _ { k } \| \| F _ { k } \| - C _ { 2 } \delta _ { k - 1 } \| F _ { k } \| + \tau \| \Delta x _ { k } \| ^ { 2 } .
$$

Apply Young’s inequality separately to the two cross terms, allocating at most $\beta \| F _ { k } \| ^ { 2 } / 8$ to each. This leaves $\beta \| F _ { k } \| ^ { 2 } / 4$ , a coeficient $q _ { x } = \tau - C _ { 1 } ^ { \prime }$ on $\| \Delta x _ { k } \| ^ { 2 }$ , and − $\cdot C _ { 0 } \delta _ { k - 1 } ^ { 2 }$ . Choosing $\tau$ above $C _ { 1 } ^ { \prime }$ gives $q _ { x } > 0$ . Adding the correcting terms in the definition of $\Psi _ { k }$ proves nonnegativity. □

Role. This lemma supplies a nonnegative stochastic Lyapunov variable despite empirical-operator perturbations and provides the $\sqrt { \Psi _ { k } }$ bound used in the next lemma.

Lemma H.6 (Empirical policy-gradient perturbation). Let

$$
g _ { \theta , k } = D _ { \theta } F ( \theta ^ { k } , x ^ { k } ) ^ { \mathsf { T } } ( y ^ { k } + \beta F _ { k } )
$$

be the true smooth augmented policy gradient and let ${ \widehat { g } } _ { \theta , k }$ be the corresponding empirical gradient. If sup $\| D _ { \theta } K _ { \theta } \| \le G _ { K }$ , then

$$
\left\| \widehat { g } _ { \theta , k } - g _ { \theta , k } \right\| \leq C \delta _ { k } \sqrt { \Psi _ { k } } .
$$

Consequently, for every $\epsilon > 0$

$$
| \langle \widehat { g } _ { \theta , k } - g _ { \theta , k } , \Delta \theta _ { k + 1 } \rangle | \leq \epsilon \| \Delta \theta _ { k + 1 } \| ^ { 2 } + C _ { \epsilon } \delta _ { k } ^ { 2 } \Psi _ { k } .
$$

Proof. The error and policy derivative have the block forms

$$
E _ { k } x = \left[ { \begin{array} { c } { - \gamma ( { \widehat { P } } _ { k } - P ) ^ { \mathsf { T } } q } \\ { 0 } \end{array} } \right] , \qquad D _ { \theta } F ( \theta , x ) [ h ] = \left[ { \begin{array} { c } { 0 } \\ { - D _ { \theta } \Pi _ { \theta } [ h ] d } \end{array} } \right] .
$$

Also $D _ { \theta } E _ { k } = 0$ . Hence their inner product is zero for every direction h, and

$$
\widehat { g } _ { \theta , k } - g _ { \theta , k } = \beta D _ { \theta } F ( \theta ^ { k } , x ^ { k } ) ^ { \mathsf { T } } E _ { k } x ^ { k } = 0 .
$$

Both displayed bounds follow. This cancellation is specific to the block structure of the occupancy constraint. □

Role. This lemma is the key benefit of sharing one empirical operator. It converts stochastic policy-gradient error into a multiplicative Lyapunov perturbation without any unbiased-product assumption.

Lemma H.7 (True one-step stochastic descent). Under the uniform bounds and parameter conditions stated above, for $k \geq 1$ there are constants $c _ { \theta } , c _ { x } , c _ { x } ^ { \prime } , a , b > 0$ such that

$$
\mathbb { E } _ { k } \Psi _ { k + 1 } \leq \left( 1 + \frac { a } { m _ { k } } \right) \Psi _ { k } - c _ { \theta } \mathbb { E } _ { k } \| \Delta \theta _ { k + 1 } \| ^ { 2 } - c _ { x } \mathbb { E } _ { k } \| \Delta x _ { k + 1 } \| ^ { 2 } - c _ { x } ^ { \prime } \| \Delta x _ { k } \| ^ { 2 } + \frac { b } { m _ { k } } .
$$

Proof. The policy gradient is unchanged by the transition perturbation, by Lemma H.6. The true policy step therefore decreases the augmented Lagrangian by at least $( 1 / ( 2 \eta _ { \theta , k } ) - \overline { { L } } / 2 ) \| \Delta \theta _ { k + 1 } \| ^ { 2 }$

On the value comparison segment, deterministic boundedness of $x , y$ and uniform boundedness of $K , { \widehat { K } }$ imply

$$
\begin{array} { r } { \| \nabla _ { x } \widehat { \mathcal { L } } _ { \beta } - \nabla _ { x } \mathcal { L } _ { \beta } \| = \| E _ { k } ^ { \mathsf { T } } ( y + \beta F ) + \beta K ^ { \mathsf { T } } E _ { k } x + \beta E _ { k } ^ { \mathsf { T } } E _ { k } x \| \leq C \delta _ { k } . } \end{array}
$$

Transferring the empirical value-step decrease by Young’s inequality gives $( 4 \eta _ { x } ) ^ { - 1 } \| \Delta x _ { k + 1 } \| ^ { 2 } - C \delta _ { k } ^ { 2 }$ The true dual ascent satisfies

$$
\langle \Delta y _ { k + 1 } , F _ { k + 1 } \rangle \leq \frac { 3 } { 2 \beta } \| \Delta y _ { k + 1 } \| ^ { 2 } + C \delta _ { k } ^ { 2 } ,
$$

since $x ^ { k + 1 }$ is uniformly bounded. Substituting Lemma H.4 and adding the potential correction yields

$$
\begin{array} { r } { \Phi _ { k } - \Phi _ { k + 1 } \geq c _ { \theta } \| \Delta \theta _ { k + 1 } \| ^ { 2 } + c _ { x } \| \Delta x _ { k + 1 } \| ^ { 2 } + c _ { x } ^ { \prime } \| \Delta x _ { k } \| ^ { 2 } - C _ { 1 } \delta _ { k } ^ { 2 } - C _ { 2 } \delta _ { k - 1 } ^ { 2 } , } \end{array}
$$

where

$$
c _ { \theta } \geq a _ { \theta } , \quad c _ { x } = \frac { 1 } { 4 \eta _ { x } } - \frac { 3 B _ { C } } { 2 \beta } - \tau > 0 , \quad c _ { x } ^ { \prime } = \tau - \frac { 3 B _ { C } } { 2 \beta } > 0 .
$$

In Lemma H.5, increase $C _ { 0 }$ to satisfy $C _ { 0 } \geq C _ { 2 }$ . With $c = \operatorname* { m i n } \{ c _ { \theta } , c _ { x } , c _ { x } ^ { \prime } \}$ , the corrected nonnegative potential then obeys the pathwise bound

$$
\Psi _ { k + 1 } \leq \Psi _ { k } - c D _ { k + 1 } + C \delta _ { k } ^ { 2 } .\tag{25}
$$

Conditioning gives $\mathbb { E } _ { k } \Psi _ { k + 1 } \le \Psi _ { k } - c \mathbb { E } _ { k } D _ { k + 1 } + C / m _ { k }$ . Since $\Psi _ { k } \geq 0$ , this also implies the stated inequality with any fixed $a > 0$ after enlarging b. □

Role. This is the stochastic occupancy Lyapunov recursion. Theorem H.9 and Theorem H.11 are direct consequences.

Define the true squared KKT residual

$$
G _ { k } ^ { \mathrm { t r u e } } = e _ { \theta , k } ^ { 2 } + e _ { x , k } ^ { 2 } + \| F _ { k } \| ^ { 2 } ,
$$

and let $D _ { k + 1 } = \| \Delta \theta _ { k + 1 } \| ^ { 2 } + \| \Delta x _ { k + 1 } \| ^ { 2 } + \| \Delta x _ { k } \| ^ { 2 } .$

Lemma H.8 (True KKT residual bound). For $k \geq 1$ , there is $C _ { G } < \infty$ such that

$$
\mathbb { E } _ { k } G _ { k + 1 } ^ { \mathrm { t r u e } } \leq C _ { G } \mathbb { E } _ { k } D _ { k + 1 } + \frac { C _ { G } } { m _ { k } } \Psi _ { k } + \frac { C _ { G } } { m _ { k } } + C _ { G } \delta _ { k - 1 } ^ { 2 } .
$$

Proof. Use the composite policy Fermat condition, the equality of true and empirical policy gradients, the uniform policy smoothness bound, and $\eta _ { \theta , k } ^ { - 1 } \leq \underline { { \eta } } ^ { - 1 }$ . They give

$$
e _ { \theta , k + 1 } \leq C ( \| \Delta \theta _ { k + 1 } \| + \| \Delta x _ { k + 1 } \| + \| \Delta y _ { k + 1 } \| ) .
$$

The empirical dual identity and update give

$$
c _ { x } + K _ { k + 1 } ^ { \mathsf { T } } y ^ { k + 1 } = - \eta _ { x } ^ { - 1 } \Delta x _ { k + 1 } - E _ { k } ^ { \mathsf { T } } y ^ { k + 1 } , \qquad F _ { k + 1 } = \beta ^ { - 1 } \Delta y _ { k + 1 } - E _ { k } x ^ { k + 1 } .
$$

Uniform boundedness of $x ^ { k + 1 } , y ^ { k + 1 }$ and Lemma H.4 imply the pathwise estimate

$$
{ \cal G } _ { k + 1 } ^ { \mathrm { t r u e } } \leq C \{ D _ { k + 1 } + \delta _ { k } ^ { 2 } + \delta _ { k - 1 } ^ { 2 } \} .
$$

Taking $\mathcal { F } _ { k }$ -conditional expectations leaves the previous error $\delta _ { k - 1 } ^ { 2 }$ unchanged and bounds the current error by $C / m _ { k }$ . This is stronger than the displayed estimate because $\Psi _ { k } \geq 0$ □

Role. This lemma translates Lyapunov descent into the true optimization residual and is the final input to the stochastic complexity theorem.

Let $\begin{array} { r } { A _ { T } = \sum _ { k = 0 } ^ { T - 1 } m _ { k } ^ { - 1 } } \end{array}$

Theorem H.9 (Finite-time stochastic occupancy theory). There are constants $C _ { 1 } , C _ { 2 } , C _ { 3 } , C _ { \mathrm { i n i t } }$ independent of the batch schedule such that

$$
\operatorname* { s u p } _ { 1 \leq k \leq T } \mathbb { E } \Psi _ { k } \leq e ^ { C _ { 1 } A _ { T } } ( C _ { \mathrm { i n i t } } + C _ { 2 } A _ { T } ) .
$$

If K is uniform on $\{ 1 , \ldots , T \}$ and independent of the algorithmic randomness, then

$$
\mathbb { E } G _ { K } ^ { \mathrm { t r u e } } \leq C _ { 3 } e ^ { C _ { 1 } A _ { T } } \big ( 1 + A _ { T } \big ) \left( \frac { 1 } { T } + \frac { A _ { T } } { T } \right) .
$$

Proof. The first-step comparison with $\theta ^ { 0 }$ and the deterministic iterate bounds give uniform upper bounds on $\mathbb { E } \Psi _ { 1 }$ and $\mathbb { E } G _ { 1 } ^ { \mathrm { t r u e } }$ . Absorb these into $C _ { \mathrm { i n i t } }$ . Summing the additive bound (25) in expectation and using $\Psi _ { k } \geq 0$ yields

$$
\operatorname* { s u p } _ { 1 \leq k \leq T } \mathbb { E } \Psi _ { k } \leq C _ { \mathrm { i n i t } } + C A _ { T } , \qquad \sum _ { k = 1 } ^ { T - 1 } \mathbb { E } D _ { k + 1 } \leq C ( 1 + A _ { T } ) .
$$

The pathwise residual estimate in Lemma H.8 has two consecutive operator errors. Their expected sum is at most $2 C A _ { T }$ . Including the first residual gives

$$
\frac { 1 } { T } \sum _ { j = 1 } ^ { T } \mathbb { E } G _ { j } ^ { \mathrm { t r u e } } \leq C ( 1 + A _ { T } ) / T .
$$

This proves the stated bounds after increasing constants, since the exponential and additional $( 1 + A _ { T } )$ factors are at least one. Uniform randomization identifies the left side with $\mathbb { E } G _ { K } ^ { \mathrm { t r u e } }$ □

Role. This theorem gives the finite-time guarantee for the stochastic occupancy operator and verifies that the operator-stability mechanism survives a second, explicit RL formulation.

Corollary H.10 (Equal-batch complexity). If $m _ { k } = m = N / T$ , then $A _ { T } = T ^ { 2 } / N . ~ I f N \ge c _ { 0 } T ^ { 2 }$ for a fixed $c _ { 0 } > 0$ , then

$$
\mathbb { E } G _ { K } ^ { \mathrm { t r u e } } = O \left( \frac { 1 } { T } + \frac { T } { N } \right) .
$$

For an unsquared KKT target $\epsilon ,$ it is suficient to take $T = O ( \epsilon ^ { - 2 } )$ and $N = O ( \epsilon ^ { - 4 } )$

Proof. Equal batches give $A _ { T } = T / m = T ^ { 2 } / N$ . If $N \geq c _ { 0 } T ^ { 2 }$ , then $A _ { T } \leq 1 / c _ { 0 }$ , so the exponential and $( 1 + A _ { T } )$ factors in Theorem H.9 are bounded by constants. Moreover $A _ { T } / T = T / N$ , proving the rate. Jensen gives $\mathbb { E } \sqrt { G _ { K } ^ { \mathrm { t r u e } } } \le \sqrt { \mathbb { E } G _ { K } ^ { \mathrm { t r u e } } }$ . Requiring the squared residual to be $O ( \epsilon ^ { 2 } )$ yields $T = O ( \epsilon ^ { - 2 } )$ and $N = O ( \epsilon ^ { \dot { - } 4 } )$ □

Role. The corollary puts the stochastic occupancy theorem on the same squared-versus-unsquared residual scale as the main Bellman complexity result.

Theorem H.11 (Almost-sure stochastic occupancy convergence). $I f \textstyle \sum _ { k } m _ { k } ^ { - 1 } < \infty$ , then $\Psi _ { k }$ converges almost surely,

$$
\sum _ { k } D _ { k + 1 } < \infty \quad a l m o s t \ s u r e l y ,
$$

$\delta _ { k } \to 0$ , and $G _ { k } ^ { \mathrm { t r u e } } \to 0$ almost surely.

Proof. Lemma H.1 and Tonelli’s theorem give $\textstyle \sum _ { k } \delta _ { k } ^ { 2 } < \infty$ almost surely. Fix such a path. From (25),

$$
\Psi _ { k + 1 } + C \sum _ { j \geq k + 1 } \delta _ { j } ^ { 2 } \leq \Psi _ { k } + C \sum _ { j \geq k } \delta _ { j } ^ { 2 } - c D _ { k + 1 } .
$$

The corrected sequence is nonnegative and decreasing. It converges and $\textstyle \sum _ { k } D _ { k + 1 } < \infty$ . Since the error tails vanish, $\Psi _ { k }$ converges as well. The pathwise residual estimate in Lemma H.8 then gives $G _ { k } ^ { \mathrm { t r u e } } \to 0$ □

Role. This theorem is the asymptotic stochastic counterpart of Theorem G.5.

Corollary H.12 (Conditional nonlinear-policy performance). Assume in addition a parameterspace domination inequality

$$
J ^ { \star } - J ( \pi _ { \theta } ) \leq C _ { \mathrm { p a r } } R _ { \theta } ( \theta ) , \qquad R _ { \theta } ( \theta ) ^ { 2 } \leq C _ { G } G ^ { \mathrm { t r u e } } .
$$

Then

$$
\begin{array} { r } { \mathbb { E } [ J ^ { \star } - J ( \pi _ { \theta _ { K } } ) ] \le C \sqrt { \mathbb { E } G _ { K } ^ { \mathrm { t r u e } } } . } \end{array}
$$

If instead $J ^ { \star } - J ( \pi _ { \theta } ) \leq C R _ { \theta } ( \theta ) ^ { 2 }$ , then the performance gap is $O ( \mathbb { E } G _ { K } ^ { \mathrm { t r u e } } )$

Proof. The first two inequalities imply $J ^ { \star } - J ( \pi _ { \theta _ { K } } ) \leq C \sqrt { G _ { K } ^ { \mathrm { t r u e } } }$ . Taking expectations and applying Jensen proves the first claim. Under quadratic parameter-space domination, $J ^ { \star } - J ( \pi _ { \theta _ { K } } ) \le$ $C C _ { G } G _ { K } ^ { \mathrm { t r u e } }$ , and expectation preserves the squared-residual rate. □

Role. This corollary states the domination conditions for performance guarantees with nonlinear policies.

![](images/f1c01b2553bf2b020d0902c4d2edaa0773a2b724800a56b03ac92c641358c7c1.jpg)  
Figure 2: Batch-size sensitivity. Mean companion squared KKT residual at fixed $T = 2 4 0$ as the Markov batch size increases. The dashed line shows an $m ^ { - 1 }$ reference slope.

## I Additional Experimental Results

This section provides additional numerical evidence for the structural predictions studied in Sections 8. All experiments use finite discounted MDPs with direct tabular policies and the same reset-controlled Markov sampling mechanism as in the theoretical analysis. Unless stated otherwise, results are averaged over multiple independently generated MDP instances and trajectory seeds, and error bars denote 95% confidence intervals.

The experiments are designed to isolate four components of the theory: (i) the statistical contribution of the Markov batch size, (ii) the correlated residual–Jacobian product created by shared sampling, (iii) the conversion from KKT residual to policy performance, and (iv) the numerical sensitivity to the ADMM penalty parameter. The corresponding main structural tests for sharedoperator stability and finite-time scaling are reported in Section 8.

## Batch-Size Sensitivity

The finite-time result predicts

$$
\mathbb { E } [ \widetilde { G } _ { K + 1 } ] \le \frac { A } { T } + \frac { B } { T } \sum _ { k < T } \frac { 1 } { m _ { k } } .
$$

For a constant batch size $m _ { k } = m$ , this reduces to

$$
O ( T ^ { - 1 } + m ^ { - 1 } ) .
$$

We therefore fix the outer iteration budget at $T = 2 4 0$ and vary $m \in \{ 6 0 , 1 2 0 , 2 4 0 , 4 8 0 , 9 6 0 \}$

Figure 2 shows that the mean companion squared KKT residual decreases monotonically with the batch size. A log–log fit over the tested range gives slope approximately −0.87, close to the $m ^ { - 1 }$ statistical contribution predicted by the finite-time bound. The deviation from an exact −1 slope is expected because the bound contains the additional $A / T$ term, which produces a finite-iteration floor as m increases.

## Residual–Jacobian Product under Shared Sampling

A central dificulty in the stochastic analysis is that the Bellman residual and its Jacobian are constructed from the same empirical operator. Writing

$$
e = \widehat { R } - R , \qquad G = \widehat { J } - J ,
$$

the stochastic policy-gradient perturbation contains the correlated term

$$
\beta G ^ { \top } e .
$$

Separate unbiasedness of G and $e$ does not imply that this product has zero mean. Remark 4.5 instead controls the product pathwise through the shared empirical operator error:

$$
\| G ^ { \top } e \| = O ( \epsilon _ { k } ^ { 2 } ) .
$$

Since the empirical operator satisfies $\mathbb { E } [ \epsilon _ { k } ^ { 2 } ] = O ( m ^ { - 1 } )$ , the resulting perturbation is predicted to occur at the same $m ^ { - 1 }$ scale.

To isolate this efect, we fix a primal–dual point and repeatedly construct empirical Bellman operators with diferent batch sizes. We compare two estimators. The same-batch estimator uses one empirical operator for both $G$ and $e ,$ matching the algorithm. The independent-batch estimator constructs them from separate samples, thereby suppressing the correlation at the cost of double sampling.

As shown in Figure 3, the same-batch product decays with empirical log–log slope approximately −1.13, consistent with the $O ( m ^ { - 1 } )$ prediction. Independent batches produce a substantially smaller mean product term, as expected from decorrelation. The latter, however, no longer represents the single shared empirical operator used by the algorithm. Together with the exact value–dual identity experiment in the main text, this result illustrates the role of structure-preserving stochasticization: the algorithm retains the coupled empirical operator and controls the induced product perturbation rather than removing it through double sampling.

## From KKT Residual to Policy Performance

Section 6 establishes that, under discounted occupancy coverage, the optimization residual has a direct RL interpretation. In particular, the policy stationarity residual is controlled by the KKT residual, and the policy-performance gap consequently satisfies

$$
J ^ { \star } - J ( \pi ) = O ( { \sqrt { G } } ) .
$$

We examine this relationship along the iterates generated by the Bellman-ADMM dynamics.

For every sampled iterate, we compute the true KKT residual G using the underlying MDP model and evaluate the corresponding policy return exactly. Figure 4 plots $J ^ { \star } - J ( \pi )$ against $\sqrt { G }$ The points exhibit the predicted first-order relation throughout the observed regime. The empirical 95th percentile of

$$
\frac { J ^ { \star } - J ( \pi ) } { \sqrt { G } }
$$

is 2.72, while the ratio remains bounded across the collected iterates. This experiment complements the optimization-based residual results by showing that decreasing KKT error is accompanied by decreasing policy suboptimality.

![](images/94766ba7c4ac1a759c5473f3747295843e7eb75a961a26eff06839c27f266dfb.jpg)  
Figure 3: Residual–Jacobian product scaling. Magnitude of the empirical product term $\beta G ^ { \top } \epsilon$ as the batch size increases. The same-batch construction follows approximately the predicted $m ^ { - 1 }$ scale. Independent batches reduce the correlation but require separate samples.

## Sensitivity to the Penalty Parameter

The convergence analysis imposes a suficient lower bound on the ADMM penalty parameter $\beta$ to guarantee positivity of the coeficients in the Lyapunov descent argument. Such analytical margins are designed to hold uniformly over the admissible problem class and need not coincide with empirically optimal parameter choices.

We therefore vary

$$
\beta \in \{ 2 . 5 , 5 , 1 0 , 2 0 , 4 0 \}
$$

while keeping the remaining algorithmic parameters fixed. As shown in Figure 5, the method remains numerically stable over the entire tested range. The trailing KKT residual changes smoothly with $\beta ,$ with smaller penalties yielding lower residuals on these instances. This behavior is consistent with the role of the theoretical condition as a uniform suficient stability margin rather than a tuning prescription for individual MDPs.

## Operator Consistency: Experimental Design

The following ablations separate the role of empirical-operator consistency from the precision of each empirical model and the total sampling cost. We compare three constructions: all shared, value–dual shared, and all separate. Table 2 specifies the model used by each update. The value– dual-shared construction isolates the consistency needed by the value–dual identity, while retaining an independently estimated model for the policy update.

Instances and parameters. Each MDP has five states and two actions. Every transition row is sampled independently from Dirichlet $\left( 0 . 7 1 _ { 5 } \right)$ . Rewards are independent $\mathcal { N } ( 0 , 0 . 4 5 ^ { 2 } )$ draws plus statewise action trends. For states $s = 0 , \ldots , 4$ , these trends are $0 . 3 0 \mathrm { ~ - ~ } 0 . 1 2 5 s$ for action 0 and $- 0 . 3 5 + 0 . 2 0 s$ for action 1. The generated transition kernel and mean rewards remain fixed during each run. The reset, training, and evaluation distributions are uniform, $\rho = \mu = \nu = \mathbf { 1 } _ { 5 } / 5$ , with $c = - \mu$ and $\phi \equiv 0$ . Unless specified otherwise, $\gamma = 0 . 8 5 , \beta = 1 0$ , and $\eta _ { \pi } = \eta _ { V } = 0 . 1 5$ . We initialize $\pi ^ { 0 }$ uniformly and set $V ^ { 0 } = \lambda ^ { 0 } = 0$ . Each run uses 180 outer iterations. The MDP seeds are 3, 7, 11, 19, 23, 31, 43, 59, and each instance uses trajectory seeds 0, 1, 2, 3, 4.

![](images/cab0ec2d9bcf977cc66b6e88bd762409d848f5ca98c9e9b46957b32f8fb23c92.jpg)  
Figure 4: KKT residual and policy performance. Policy suboptimality versus the true KKT scale $\sqrt { G }$ . The dashed line denotes the empirical 95th-percentile $C \sqrt { G }$ envelope.

Table 2: Empirical-model assignments. Distinct indices denote separately sampled models. Under an equal per-model batch size $m ,$ , the last column gives the total number of reset-controlled sampling steps per iteration.
<table><tr><td>Construction</td><td>Policy</td><td>Value</td><td>Dual</td><td>Total steps</td></tr><tr><td>All shared</td><td>0</td><td>0</td><td>0</td><td>m</td></tr><tr><td>Value-dual shared</td><td>0</td><td>1</td><td>1</td><td>2m</td></tr><tr><td>All separate</td><td>0</td><td>1</td><td>2</td><td>3m</td></tr></table>

Markov sampling and budget accounting. At iteration k, the behavior policy is frozen at $\bar { \pi } ^ { k } = 0 . 7 \pi ^ { k } + 0 . 3 1 _ { 2 } / 2$ . Each step resets to $\rho$ with probability $1 - \gamma$ and otherwise samples an environment transition. Environment rewards are observed with independent $\mathcal { N } ( 0 , 0 . 3 ^ { 2 } )$ noise. Only non-reset transitions contribute to the empirical transition and reward rows. Unvisited rows use the uniform transition distribution and reward zero. Each distinct model has a separate random stream and a persistent chain state, initialized at state 0. Corresponding streams use matched random seeds across constructions. With adaptive behavior, subsequent observations can difer as the policies evolve. A fixed-behavior control below removes this dependence.

We vary $m \in \{ 8 0 , 1 6 0 , 3 2 0 , 6 4 0 \}$ under two budget conventions. For equal total budget, m resetcontrolled steps are divided among one, two, or three models, with remainders assigned to the earliest model indices. For equal per-model batch, every distinct model receives m steps, giving total costs m, $2 m$ , and 3m. Reset steps are included in these costs. Counts of environment transitions and resets are also recorded separately. All three constructions in this study use the same stream protocol. The original two-way experiment uses serial blocks of one trajectory and assigns $\lfloor m / 3 \rfloor$ steps to each separate model. The new controls use exactly m total steps in the equal-budget comparison.

![](images/738aaef1837251df75f4bd442dbb25471a08b3d264d395922d2fe82c596c0bc6.jpg)  
Figure 5: Penalty sensitivity. Trailing true squared KKT residual as a function of the ADMM penalty parameter $\beta .$

Exact subproblem solutions. The two-action policy subproblem separates over states. Write $\begin{array} { r } { x _ { s } ^ { k } = \pi ^ { k } ( 1 \mid s ) , q _ { s a } = \widehat { r } _ { k } ^ { p } ( s , a ) + \gamma \sum _ { s ^ { \prime } } \widehat { P } _ { k } ^ { p } ( s ^ { \prime } \mid s , a ) V ^ { k } ( s ^ { \prime } ) , d _ { s } = q _ { s 1 } - q _ { s 0 } } \end{array}$ , and $b _ { s } = V ^ { k } ( s ) - q _ { s 0 }$ , where the superscript p denotes the policy model. Its exact solution is

$$
x _ { s } ^ { k + 1 } = \mathrm { p r o j } _ { [ 0 , 1 ] } \frac { d _ { s } ( \lambda _ { s } ^ { k } + \beta b _ { s } ) + 2 x _ { s } ^ { k } / \eta _ { \pi } } { \beta d _ { s } ^ { 2 } + 2 / \eta _ { \pi } } .\tag{26}
$$

Let $\widehat { M } _ { k } ^ { v }$ and $\widehat { r } _ { k } ^ { v }$ denote the value model evaluated at $\pi ^ { k + 1 }$ . The value update solves

$$
\left[ \beta ( \widehat { M } _ { k } ^ { v } ) ^ { \top } \widehat { M } _ { k } ^ { v } + \eta _ { V } ^ { - 1 } I \right] V ^ { k + 1 } = - c - ( \widehat { M } _ { k } ^ { v } ) ^ { \top } \lambda ^ { k } + \beta ( \widehat { M } _ { k } ^ { v } ) ^ { \top } \widehat { r } _ { k } ^ { v } + \eta _ { V } ^ { - 1 } V ^ { k } .\tag{27}
$$

The coeficient matrix is positive definite. These exact solves ensure that subproblem approximation does not enter the comparison.

Metrics and statistical aggregation. We evaluate the true squared KKT residual

$$
G ( \pi , V , \lambda ) = \mathrm { d i s t ^ { 2 } } \Big ( 0 , N _ { \Pi } ( \pi ) + J _ { \pi } ( \pi , V ) ^ { \top } \lambda \Big ) + \| c + M _ { \pi } ^ { \top } \lambda \| ^ { 2 } + \| R ( \pi , V ) \| ^ { 2 }\tag{28}
$$

using the underlying MDP. Policy return is evaluated by an exact Bellman solve, and $J ^ { \star }$ is obtained by enumerating the $2 ^ { 5 }$ deterministic policies. We also record $\lVert \lambda ^ { k + 1 } - \lambda ^ { k } \rVert ^ { 2 }$ , the return gap $J ^ { \star } -$ $J ( \pi ^ { k + 1 } )$ , and the identity defect defined below. Unless stated otherwise, each reported outcome averages iterations 151–180. We first average the five trajectory seeds within each MDP, then average the eight MDP means. Figure intervals are 95% percentile bootstrap intervals from 10,000 resamples of these eight means. Paired comparisons use the eight within-MDP diferences and a 95% Student-t interval with seven degrees of freedom. These are individual, unadjusted intervals.

## Identifying the Value–Dual Consistency Mechanism

Let $\widehat { R } _ { k } ^ { v }$ and $\widehat { R } _ { k } ^ { d }$ denote the empirical residuals used by the value and dual updates. In this subsection, both are evaluated at $( \pi ^ { k + 1 } , V ^ { k + 1 } )$ , and $\widehat { M } _ { k } ^ { v }$ is evaluated at $\pi ^ { k + 1 }$ . The exact value solve and the

![](images/b16abea1abfbb4d67bdda44eea2392bad334b24c952482916d05a7681a1a7a20.jpg)

![](images/da9e45e347918609f94cc4e4dfe5727e001dbcb889d2335e4aa98bcbf0080641.jpg)  
Figure 6: Identity preservation and estimator precision. Left: trailing identity defect under equal per-model batches and adaptive behavior. All-shared and value–dual-shared curves overlap at numerical precision. Right: empirical-model squared error under fixed uniform behavior and 320 steps per model. Circles denote policy models and squares denote value models. Model error is measured as the sum of squared Frobenius transition error and squared Euclidean reward error. Error intervals resample MDP means.

multiplier update give

$$
0 = c + ( \widehat { M } _ { k } ^ { v } ) ^ { \top } \lambda ^ { k } + \beta ( \widehat { M } _ { k } ^ { v } ) ^ { \top } \widehat { R } _ { k } ^ { v } + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } ,\tag{29}
$$

$$
\lambda ^ { k + 1 } = \lambda ^ { k } + \beta \widehat { R } _ { k } ^ { d } .\tag{30}
$$

Substitution yields the pathwise relation

$$
\underset { h _ { k + 1 } } { \underbrace { c + ( \widehat M _ { k } ^ { v } ) ^ { \top } \lambda ^ { k + 1 } + \eta _ { V } ^ { - 1 } \Delta V _ { k + 1 } } } = \beta ( \widehat M _ { k } ^ { v } ) ^ { \top } ( \widehat R _ { k } ^ { d } - \widehat R _ { k } ^ { v } ) .\tag{31}
$$

Thus, sharing the value and dual models sets the right-hand side to zero, regardless of the model used in the policy update. This identity explains why both all-shared and value–dual-shared updates retain an explicit multiplier representation. All sharing also reuses the same empirical model for the policy update, reducing the number of sampled models.

We measure the identity defect $D _ { k + 1 } = \| h _ { k + 1 } \| _ { 2 }$ and verify Equation (31) at every update. At equal per-model batch size 320, the mean trailing defects are $1 . 0 5 \times 1 0 ^ { - 1 4 }$ for both sharing constructions and 1.793 for all separate updates. Figure 6 shows the same distinction across batch sizes. Across the full experiment suite, including the large-penalty check below, the maximum norm of the diference between the two sides of Equation (31) is $2 . 0 2 \times 1 0 ^ { - 1 0 }$ . The value–dual-shared construction serves as a mechanism ablation. The convergence results in the main text apply to the specified all-shared algorithm.

## Separating Operator Consistency from Sampling Cost

Figure 7 compares true squared KKT residuals, squared multiplier increments, and return gaps under both budget conventions. Table 3 reports the batch-320 slice. At equal per-model batch size, the mean residual is 0.503 for all shared and 4.159 for all separate, a ratio of 8.27. All sharing uses one third of the total sampling steps in this comparison. Value–dual sharing attains residual 0.639, placing its optimization behavior closer to all sharing while preserving the same multiplier identity. Under equal total budget, the corresponding residuals are 0.503, 1.381, and 13.192. These complementary comparisons demonstrate the contribution of consistency at matched model batch sizes and the sampling benefit of reusing one model.

Table 3: Batch-320 comparison. Entries average the last 30 of 180 iterations and then the eight MDP means. G is the true squared KKT residual, $\Delta _ { \lambda }$ is the squared multiplier increment, and $\Delta _ { J } = J ^ { \star } - J ( \pi )$ is the return gap. Total cost includes reset steps.
<table><tr><td>Budget</td><td>Construction</td><td>Steps</td><td>G</td><td> $\Delta _ { \lambda }$ </td><td> $\Delta _ { J }$ </td></tr><tr><td rowspan="3">Equal total</td><td>All shared</td><td>320</td><td>0.5028</td><td>1.0452</td><td>0.01425</td></tr><tr><td>Value-dual shared</td><td>320</td><td>1.3813</td><td>2.2538</td><td>0.05144</td></tr><tr><td>All separate</td><td>320</td><td>13.1920</td><td>12.8006</td><td>0.17656</td></tr><tr><td rowspan="3">Equal per model</td><td>All shared</td><td>320</td><td>0.5028</td><td>1.0452</td><td>0.01425</td></tr><tr><td>Value-dual shared</td><td>640</td><td>0.6390</td><td>1.1283</td><td>0.02419</td></tr><tr><td>All separate</td><td>960</td><td>4.1585</td><td>4.0575</td><td>0.04168</td></tr></table>

Table 4: Paired contrasts at equal per-model batch size 320. Each contrast subtracts the second construction from the first. Intervals are based on eight paired MDP means.
<table><tr><td>Metric</td><td>Contrast</td><td>Mean difference</td><td>95% interval</td></tr><tr><td rowspan="3">G</td><td>Value-dual shared - all shared</td><td>0.1361</td><td>[0.0463, 0.2260]</td></tr><tr><td>All separate - all shared</td><td>3.6557</td><td>[2.6170, 4.6944]</td></tr><tr><td>All separate – value-dual shared</td><td>3.5196</td><td>[2.5340, 4.5052]</td></tr><tr><td rowspan="3"> $\Delta _ { \lambda }$ </td><td>Value-dual shared – all shared</td><td>0.0831</td><td>[0.0205,0.1457]</td></tr><tr><td>All separate - all shared</td><td>3.0123</td><td>[2.2128, 3.8118]</td></tr><tr><td>All separate - value-dual shared</td><td>2.9293</td><td>[2.1526, 3.7060]</td></tr><tr><td rowspan="3"> $\Delta _ { J }$ </td><td>Value-dual shared - all shared</td><td>0.00994</td><td>[-0.00384, 0.02372]</td></tr><tr><td>All separate - all shared</td><td>0.02742</td><td>[0.01143, 0.04342</td></tr><tr><td>All separate – value-dual shared</td><td>0.01748</td><td>[0.01011, 0.02486]</td></tr></table>

The paired diferences in Table 4 quantify variation across instances. At equal per-model batch size 320, all separate exceeds all shared in squared KKT residual by 3.656, with interval [2.617, 4.694]. Its squared multiplier increment exceeds all shared by 3.012, with interval [2.213, 3.812]. The allshared versus value–dual-shared distinction is supported for the residual and multiplier increment. Their return-gap diference has an interval containing zero. We therefore use this comparison to locate the operator-consistency mechanism and quantify its sampling cost.

## Fixed-Behavior Control

To separate operator consistency from changes in data collection, we repeat the comparisons with $\bar { \pi } ^ { k } ( a \mid s ) = 1 / 2$ at every iteration. At equal per-model batch size 320, the policy model is identical across all three constructions under the matched streams. The value model is also identical between value–dual shared and all separate. All shared reuses its policy model for the value and dual updates. Each distinct model follows the same estimation procedure and sampling distribution. This control removes the feedback from the learned policy to the sampling distribution.

![](images/ead67f9c2d8734d89d200f6a5b99586d982e9df983b2cb7420493ba17b193aa8.jpg)  
Figure 7: Consistency and sampling-budget controls. Top: equal total steps per iteration. Bottom: equal steps per empirical model, with total costs m, 2m, and 3m for all shared, value– dual shared, and all separate. Columns show the true squared KKT residual, squared multiplier increment, and return gap. Curves average the last 30 of 180 iterations. Shading denotes 95% bootstrap intervals over MDP means.

Table 5 retains the same ordering of the three residuals. The paired residual diference between all separate and all shared is 6.151, with 95% interval [4.013, 8.288]. Between all separate and value– dual shared it is 5.854, with interval [3.882, 7.825]. Thus, the residual separation persists when the behavior policy and per-model sampling distributions are fixed.

## Discounted-Resolvent Sensitivity

The inverse-map argument predicts that the sensitivity of the dual representation depends on the discounted resolvent. To examine this dependence with fixed estimation error, we collect one 320- step batch using uniform behavior and sampling discount 0.85 for each MDP and trajectory seed. We then hold both the empirical transition matrix and the uniform evaluation policy fixed while varying $\gamma \in \{ 0 . 5 0 , 0 . 7 0 , 0 . 8 5 , 0 . 9 5 , 0 . 9 8 \}$ . Write $M _ { \gamma } = I - \gamma P _ { \pi }$ and $\widehat { M } _ { \gamma } = I - \gamma \widehat { P } _ { \pi }$ . The resolvent identity gives

$$
\lVert \widehat { M } _ { \gamma } ^ { - 1 } - M _ { \gamma } ^ { - 1 } \rVert _ { 2 } \leq \gamma \lVert \widehat { M } _ { \gamma } ^ { - 1 } \rVert _ { 2 } \lVert \widehat { P } _ { \pi } - P _ { \pi } \rVert _ { 2 } \lVert M _ { \gamma } ^ { - 1 } \rVert _ { 2 } .\tag{32}
$$

Table 5: Uniform-behavior control at 320 steps per model. Total costs are 320, 640, and 960 steps, respectively. The notation and aggregation match Table 3.
<table><tr><td>Construction</td><td> $G$ </td><td> $\Delta _ { \lambda }$ </td><td> $\Delta _ { J }$ </td></tr><tr><td>All shared</td><td>0.8505</td><td>1.7776</td><td>0.02818</td></tr><tr><td>Value-dual shared</td><td>1.1477</td><td>1.8908</td><td>0.04018</td></tr><tr><td>All separate</td><td>7.0014</td><td>6.7681</td><td>0.07335</td></tr></table>

![](images/b7fe967a38be1fb396483579b7485312c4899fb84ce70631c4b8f564cfbd78a4.jpg)

![](images/4fc390b38564aa36fdc3f08bfe5b950333a9e4a8263f6ee5a46de16628daf67a.jpg)

![](images/ca9e3b8154b6efc5a9d9769cfd53fe09145a47e9a68a517ad6fb01c97bd7d3b6.jpg)  
Figure 8: Discount sensitivity with fixed policy and transition error. One empirical transition model per run is reused across all discounts. Panels show the true resolvent norm and the absolute and relative errors in the stationary dual representation. Shading denotes 95% bootstrap intervals over MDP means.

We evaluate the stationary dual representations $\lambda _ { \gamma } = M _ { \gamma } ^ { - \top } \mu$ and $\widehat { \lambda } _ { \gamma } = \widehat { M } _ { \gamma } ^ { - \top } \mu$ . Figure 8 reports the true inverse norm, the absolute dual error, and the relative error $\| \widehat { \lambda } _ { \gamma } - \lambda _ { \gamma } \| _ { 2 } / \| \lambda _ { \gamma } \| _ { 2 }$ . As γ increases from 0.50 to 0.98, the mean inverse norm rises from 2.03 to 52.15, the absolute dual error from 0.0462 to 2.2808, and the relative error from 0.0512 to 0.0978. The relative metric accounts for the change in discounted occupancy mass, $\mathbf { 1 } ^ { \top } \lambda _ { \gamma } = ( 1 - \gamma ) ^ { - 1 }$ . These fixed-perturbation measurements support the sensitivity mechanism used in multiplier control without estimating a worst-case asymptotic rate.

We also run the three algorithms at the same discount grid using uniform behavior and 320 steps per model, holding the remaining parameters fixed. Figure 9 reports these dynamic comparisons. Here, the reset probability is $1 - \gamma$ , so the experiment includes both operator sensitivity and changes in the sampling process. All-shared updates have lower mean KKT residuals than all-separate updates throughout this grid. The fixed-perturbation experiment above isolates the inverse-operator component of this dependence.

## Parameter-Margin Check and Numerical Verification

The preceding finite-budget comparisons use $\beta = 1 0 . \mathrm { ~ A ~ }$ separate parameter-margin check uses the uniform bounds $\bar { \kappa } = \sqrt { 5 } / ( 1 - \gamma )$ and $L _ { M } = \gamma \sqrt { 2 }$ . At $\gamma = 0 . 8 5$ and $\eta _ { \pi } = \eta _ { V } = 0 . 1 5$ , the suficient

![](images/b8a658818dea37188e03bace684dcf76b871d1fa30720b4ab1cf598262a6d9fd.jpg)  
Figure 9: Dynamic discount sweep. Uniform behavior and 320 steps per model, with all other algorithmic parameters fixed. The reset probability changes with the discount. Outcomes average the last 30 of 180 iterations, and shaded intervals resample MDP means.

Table 6: Large-penalty check. Uniform behavior, 320 steps per model, and $\beta = 9 7 7 7 7 . 7 7 8 .$ . All quantities use the same trailing aggregation as the other controls. D denotes the identity defect.
<table><tr><td>Construction</td><td>G</td><td> $\Delta _ { \lambda }$ </td><td> $\Delta _ { J }$ </td><td>D</td></tr><tr><td>All shared</td><td>6.735</td><td>76.779</td><td>2.034</td><td> $2 . 2 2 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>Value-dual shared</td><td>266.470</td><td>1813.805</td><td>2.177</td><td> $3 . 1 3 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>All separate</td><td> $5 . 1 2 \times 1 0 ^ { 8 }$ </td><td> $8 . 3 3 \times 1 0 ^ { 8 }$ </td><td>2.210</td><td> $1 . 8 2 \times 1 0 ^ { 4 }$ </td></tr></table>

lower bound in the main text is

$$
\beta > \operatorname* { m a x } \left\{ \frac { 6 0 \bar { \kappa } ^ { 2 } } { \eta _ { V } } , 3 0 \eta _ { \pi } \bar { \kappa } ^ { 4 } L _ { M } ^ { 2 } \| c \| ^ { 2 } \right\} = 8 8 8 8 8 . 8 8 9 .\tag{33}
$$

We set $\beta = 9 7 7 7 7 . 7 7 8$ , which is 1.1 times this threshold, and run all three constructions with uniform behavior and 320 steps per model. The three coeficients in the all-shared Lyapunov descent bound are 0.57197, 0.07576, and 0.07576, respectively. Table 6 reports the corresponding finite-time outcomes. This check concerns the penalty margin of the all-shared theory. The main comparisons assess finite-budget behavior, and the fixed-batch runs do not invoke the square-summability condition for almost-sure asymptotic convergence.

The suite contains 960 budget-comparison runs, 240 uniform-behavior runs, 600 discount-sweep runs, and 120 parameter-margin runs, totaling 1920 algorithm runs. The fixed-perturbation study adds 200 evaluation points. Some settings coincide across studies and are not treated as additional independent replicates in the statistical analysis. Numerical verification reproduces the 80 original batch-320 runs to a maximum absolute discrepancy of $3 . 5 5 \times 1 0 ^ { - 1 5 }$ across the recorded summary metrics. Independent scalar optimization verifies the policy subproblem objective to $1 . 0 7 \times 1 0 ^ { - 1 4 }$ and the checked value solves have first-order residual at most $6 . 4 4 \times 1 0 ^ { - 1 5 }$ . Configurations, perrun results, selected per-iteration traces, and the analysis scripts are retained with the experiment package.