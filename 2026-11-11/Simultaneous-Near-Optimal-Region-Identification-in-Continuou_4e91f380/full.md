# Simultaneous Near-Optimal Region Identification in Continuous Simulation Optimization: Minimax Complexity and Adaptive Sampling

Yilin Zheng National University of Singapore Singapore

Wenjia Wang National University of Singapore Singapore

Haowei Wang National University of Singapore Singapore

Szu Hui Ng National University of Singapore Singapore

Abstract. Simulation optimization typically returns a single estimated optimal point. When decisions also depend on criteria evaluated independently of simulation, such as resource cost, the decision maker instead needs the set of designs whose expected performance lies within a tolerance of the optimum, often choosing the tolerance only after seeing the tradeofs. We study how to identify these near-optimal regions on a continuous design space, simultaneously for all tolerances up to a prescribed maximum $T ,$ with region estimation error at most η. We show that region identification is not a by-product of optimization. Specifically, when $\alpha > \nu ,$ any procedure, even one given the optimal design in advance, needs at least $\Omega \big ( \bar { \eta } ^ { - ( 2 + d / \nu ) } \operatorname* { m i n } \{ 1 , K \bar { T } ^ { d / \bar { \alpha } } \} \big )$ simulation evaluations, where d is the dimension, ν the smoothness, and $( K , \alpha )$ the covering parameters that describe the geometry of the near-optimal designs. We propose SASORI, which returns an estimated optima point and the regions for all tolerances from the same evaluations, matching this lower bound up to logarithmic factors. SASORI also achieves the optimal tradeof between region estimation error and cumulative regret, and reduces to a minimax-optimal simulation optimization procedure at zero tolerance. Numerical experiments illustrate the efectiveness of our proposed method.

Keywords: simulation optimization; subset selection; near-optimal regions; sample complexity;   
Gaussian process metamodels.

## 1 Introduction

Stochastic simulation is a standard tool for evaluating designs of complex stochastic systems whose performance cannot be computed in closed form, such as service systems, queueing networks, manufacturing systems, and supply chains. In simulation optimization, each simulation evaluation at a design point consists of one replication and returns a noisy observation of expected performance. Because evaluations are costly, the analyst seeks a design with the best expected performance within a fixed evaluation budget. Ranking-and-selection procedures address this problem when the designs form a finite list (Kim and Nelson 2001, Chen et al. 2000, Fan et al. 2016), search and metamodeling methods address large or continuous design spaces (Hong and Nelson 2006, Ankenman et al. 2010, Frazier et al. 2008, Salemi et al. 2019, Fan and Hu 2018), and Bayesian optimization fits a Gaussian process metamodel and chooses each new design by optimizing an acquisition function (Frazier 2018). In all these methods, the output is an estimated optimal point, often accompanied by a statistical statement of how close its performance is to the best achievable.

Such an output sufices when simulated performance is the only criterion. In practice, however, the decision maker often implements not the simulated optimum but the most attractive design among those whose simulated performance is acceptable, according to downstream requirements, such as resource availability and operational restrictions, that may not be fully specified during the optimization stage (Maus et al. 2022, Miller et al. 2024). Related needs arise in molecular design (Verhellen and Van den Abeele 2020), chemical process optimization with multiple objectives (Schweidtmann et al. 2018), and engine control, where Miller et al. (2024) keep several control configurations of comparable performance because restrictions on fuel blends and energy consumption vary across operating conditions.

As an example, consider a contact center whose design variables x are the stafing levels by period and the routing probabilities. Simulation estimates $f ( x )$ , the expected fraction of calls answered within a service target, whereas the resource cost is $c _ { 0 } ( x )$ , including wages, licenses, and trunk capacity. Management will deploy the least-cost design whose service level lies within an acceptable loss $\tau$ of the best achievable, that is, a minimizer of $c _ { 0 }$ over the acceptable region

$$
{ \mathcal { S } } _ { \tau } = \{ x \in { \mathcal { X } } : f ^ { \star } - f ( x ) < \tau \} , \qquad f ^ { \star } = \operatorname* { m a x } _ { x \in { \mathcal { X } } } f ( x ) .
$$

The simulation study must therefore identify a region of the design space rather than a single point. Moreover, $\tau$ is a managerial choice that is best made after the tradeof between service and cost has been seen, so management needs the acceptable regions for all losses it may consider, which an estimated optimal point alone cannot provide.

The simulation literature has long recognized the value of set-valued output when the alternatives are finite. Subset selection returns a bounded-size subset that contains a near-best system with prescribed probability (Sullivan and Wilson 1989), screening procedures eliminate clearly inferior systems before selection (Nelson et al. 2001), multiple comparisons with the best construct simultaneous confidence intervals for all systems to identify the systems that could be the best (Matejcik and Nelson 1995), and feasibility determination classifies systems against a stated threshold, permitting either classification within an indiference zone around it (Andradóttir and Kim 2010, Batur and Kim 2010). These procedures do not extend directly to continuous design variables such as service rates, routing probabilities, or reorder points, for which the acceptable region has unknown shape, extent, and number of components, and infinitely many designs must be classified. Thresholding bandits and kernel level-set estimation classify designs against a prescribed level (Locatelli et al. 2016, Mason et al. 2022, Gotovos et al. 2013, Bogunovic et al. 2016), and continuous-domain methods exploit boundary geometry or local search (Shekhar and Javidi 2019, Bachoc et al. 2021, Ngo et al. 2024, 2026). In our problem, the threshold $f ^ { \star } - \tau$ depends on the unknown optimal value. Methods that target designs within an additive loss of an unknown maximum, on a continuous domain (Nguyen et al. 2021) or among finitely many alternatives (Mason et al. 2020, al Marjani et al. 2023), address a single loss specified in advance $( ^ { 6 6 } \mathrm { a l l } ^ { \dag } )$ in the all-ϵ-good-arm formulation refers to the arms at one fixed ϵ), and only the finite-alternative methods come with sample-complexity guarantees. Related work estimates the set of maximizers (Minsker 2013) or combines region learning with optimization (Ewald et al. 2027, Zhang et al. 2023). Thus neither a continuous domain nor an unknown optimum alone distinguishes our problem. Because $\tau$ is chosen after the study, our output must instead be valid simultaneously for all tolerances in [0, T], where the maximum tolerance $T$ is fixed before sampling. These features raise two questions that have not been answered:

Q1: Can one simulation study deliver both an estimated optimal point and the acceptable regions, and if $s o ,$ at what cost to the eficiency with which the simulation budget is allocated?

Q2: How many simulation evaluations are needed to identify the acceptable regions for all tolerances in a range, to a prescribed accuracy?

To address Q1, we propose Shared-confidence Adaptive Sampling for Optimization and Region Identification (SASORI), a sequential procedure that returns an estimated optimal point together with inner and outer sets $I _ { \tau } \subseteq S _ { \tau } \subseteq O _ { \tau }$ for all tolerances $\tau \in [ 0 , T ]$ from the same evaluations. With high probability, simultaneously for all these tolerances, all points in $I _ { \tau }$ is acceptable, all points outside $O _ { \tau }$ is unacceptable, and all points of the undetermined set $O _ { \tau } \setminus I _ { \tau }$ , which shrinks as evaluations accumulate, have performance within $\eta$ of the threshold $f ^ { \star } - \tau$ . The region estimation error η plays a role similar to the indiference-zone parameter in ranking and selection (Kim and Nelson 2001, Nelson et al. 2001), and the manager may choose $\tau$ after seeing the output, without additional evaluations and with the same probability guarantee. SASORI fits a Gaussian process metamodel (Rasmussen and Williams 2006) to batches of evaluations and maintains a single pair of confidence bounds on the response surface, shared by two tracks: an optimization track refines the estimated optimal point, and a region track identifies the acceptable regions. Each track proceeds in phases with geometrically decreasing target accuracy and samples the maximizer of an acquisition function that is large only at points that are still relevant to the track’s tolerance and whose confidence interval is still too wide. Each completed phase tightens the shared bounds, so evaluations spent on the regions also sharpen the estimated optimal point, and vice versa. SASORI does not require the covering parameters of the near-optimal set as inputs and reduces to a simulation optimization procedure when $T = 0$

Our analysis combines a smoothness assumption that is standard in kernel-based metamodeling (Ankenman et al. 2010, Vakili et al. 2021, Li and Scarlett 2022), with smoothness $\nu$ in dimension $d ,$ and a condition on the geometry of the near-optimal designs in the spirit of Bubeck et al. (2011) and Shekhar and Javidi (2022), where K is a uniform bound on the number of balls, of radius proportional to $u ^ { 1 / \alpha }$ , needed to cover the designs with performance gap at most $u ,$ over a specified range of $u .$ Thus K can reflect the number of separated near-optimal peaks, while α controls how the covering radius decreases with the performance gap. For example, α = 2 if $f$ is locally strongly concave near each peak. Our contributions are as follows.

1. Formulation. We formulate near-optimal region identification on a continuous design space as a set-valued output whose guarantees hold simultaneously for all tolerances up to $T ,$ relative to an unknown optimal value. This brings the set-valued outputs of ranking and selection to continuous design spaces and allows the tolerance to be chosen after the simulation study (Section 2).

2. Evaluation complexity of region identification (Q2). When $\alpha > \nu$ and $\eta$ is small compared with $T .$ , any procedure, even one given the optimal point and its value in advance, needs a number of evaluations of order $\eta ^ { - ( 2 + d / \nu ) }$ min $\{ 1 , K T ^ { d / \alpha } \}$ (Theorem 3.1), and SASORI achieves this order up to logarithmic factors (Corollary 5.1). The exponent of $\eta ^ { - 1 }$ is the one required to approximate the entire response surface uniformly to error $\eta ,$ and the geometry enters only through the factor min{1, $K T ^ { d / \alpha } \}$ , which is small when there are few near-optimal peaks and the near-optimal designs occupy a small part of the design space, and which measures the saving from concentrating evaluations on the near-optimal part of the design space. Solving the optimization problem exactly therefore leaves region identification essentially untouched, and thus region identification is not a by-product of optimization and must be addressed in its own right. The reason is that the two tasks need information in diferent places. An eficient optimization procedure stops simulating a design as soon as it can tell that the design is not optimal, following the logic of screening (Nelson et al. 2001, Kim and Nelson 2001), whereas region identification must determine membership for designs whose performance gap is close to τ , even when the optimum is already known. For a single tolerance fixed before sampling, we give a separate lower bound and a boundary-sensitive specialization of SASORI that can require substantially fewer evaluations under regular boundary geometry (Theorems 3.2 and 5.6).

3. One procedure for both outputs (Q1). We measure the quality of the estimated optimal point by its simple regret and the eficiency of the allocation by the cumulative regret, which is the total performance gap of all simulated points. With $T > 0$ , as long as at least half of the evaluations serve optimization, the estimated optimal point keeps the minimax order of simple regret. Cumulative regret, in contrast, must increase as the target region estimation error decreases. Any procedure with a valid region guarantee incurs cumulative regret at least of order T times the number of evaluations shown in our contributions Point 2, because designs near each tolerance boundary must be simulated even though they are known to be suboptimal (Theorem 5.4). When $\alpha > \nu _ { : }$ , SASORI achieves the resulting optimal tradeof between region estimation error and cumulative regret, up to logarithmic factors (Theorem 5.5 and Corollary 5.2). At one end of this tradeof, it keeps the cumulative regret at its optimal rate while the region estimation error still decreases to zero, although not at the fastest rate; at the other end, it achieves the fastest region identification rate, at the cost of cumulative regret that grows linearly in the number of evaluations.

4. Localized minimax rates for simulation optimization. Of independent interest, with $T = 0$ , the simple and cumulative regret of SASORI achieve the minimax rates, up to logarithmic factors, in two senses (Theorems 5.2, 5.3, and 5.7). Without any condition on the near-optimal points, these are the global Matérn rates N<sup>−ν/(2ν+d)</sup> and $N ^ { ( \nu + d ) / ( 2 \nu + d ) }$ , where N is the number of evaluations (Scarlett et al. 2017). Under the covering condition with $\alpha > \nu .$ the simple regret is of order min $\{ ( K / N ) ^ { 1 / ( 2 + d / \nu - d / \alpha ) } , N ^ { - \nu / ( 2 \nu + d ) } \}$ , uniformly in K, and the cumulative regret is N times this quantity, so the rates improve when K is small and the global rates take over once K is large. Matching rates also hold at $\alpha = \nu \geq 2$ , whereas for $2 \leq \alpha < \nu$ we establish lower bounds (Section 5.6).

Our regret analysis relates to the kernel bandit literature. The global Matérn lower bounds are due to Scarlett et al. (2017), and matching simple and cumulative regret upper bounds, up to logarithmic factors, are obtained by variance maximization and batched elimination, respectively (Vakili et al. 2021, Li and Scarlett 2022). Near-optimal geometry also governs regret in continuum bandits (Bubeck et al. 2011, Wang et al. 2018). The closest RKHS analysis is that of Shekhar and Javidi (2022), who derive instance-dependent cumulative regret bounds for policies that are required to be consistent on the entire RKHS ball. Our covering-class regret analysis results instead concern a minimax problem in which the infimum ranges over all policies, without a consistency requirement on a larger ball, and all alternatives satisfy the same covering condition at all near-optimality levels. The results cover both simple and cumulative regret, keep K explicit, and, for $\alpha > \nu ,$ match the rates through the transition to the global rates as K grows.

The rest of this work is arranged as follows. Section 2 introduces the model, the region outputs, and the smoothness model. Section 3 establishes the lower bounds for region identification, Section 4 presents SASORI, and Section 5 collects its guarantees. Section 6 presents the numerical study on

Branin-2 and Hartmann-3, Section $7$ concludes, and all proofs and additional numerical results are in the appendices.

## 2 Preliminaries

## 2.1 Simulation Model and Region Guarantees

We consider an expensive stochastic simulation with expected response $f : \mathcal { X } \to \mathbb { R }$ , where $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ is a compact design space. Our objective is to maximize $f ,$ and we write $f ^ { \star } = \operatorname* { m a x } _ { x \in \mathcal { X } } f ( x )$ . A design point x may specify, for example, stafing levels, service rates, routing probabilities, or reorder points. In order to find a maximizer of $f _ { \cdot }$ , we select design points sequentially under a total budget of $N$ evaluations. At time $t ,$ we choose $x _ { t }$ using the preceding observations and observe $y _ { t } = f ( x _ { t } ) + \varepsilon _ { t }$ where $\varepsilon _ { t }$ is simulation noise with conditional mean zero. After the final evaluation, the procedure returns the estimated optimal point $\widehat { x } _ { N }$ . We assess the procedure through two criteria: the simple regret $r _ { N } = f ^ { \star } - f ( \widehat { x } _ { N } )$ and the cumulative regret $\begin{array} { r } { R _ { N } = \sum _ { t = 1 } ^ { N } \{ f ^ { \star } - f ( x _ { t } ) \} } \end{array}$ . Simple regret measures the performance lost by implementing the estimated optimal point, whereas cumulative regret measures the performance lost at the design points evaluated during the procedure.

Near-Optimal Regions and Their Confidence Bounds In addition to an estimated optimal point $\widehat { x } _ { N }$ , we seek to identify the near-optimal regions of the design space. For a tolerance $\tau \geq 0$ define the acceptable region

$$
S _ { \tau } = S _ { \tau } ( f ) = \{ x \in \mathcal { X } : f ^ { \star } - f ( x ) < \tau \} ,
$$

which consists of the design points whose expected response is less than $\tau$ below the maximum $f ^ { \star }$ We seek an inner set $I _ { \tau }$ consisting only of acceptable design points and an outer set $O _ { \tau }$ containing all of them. For a target region estimation error $\eta > 0$ , the desired guarantee is

$$
I _ { \tau } \subseteq S _ { \tau } \subseteq O _ { \tau } , \qquad O _ { \tau } \setminus I _ { \tau } \subseteq \{ x \in { \mathcal { X } } : | f ^ { \star } - f ( x ) - \tau | \leq \eta \} .\tag{1}
$$

Here $\tau$ is the acceptable performance loss, whereas $\eta$ bounds the region estimation error measured in response value. Under the desired guarantee, every unclassified design $x \in O _ { \tau } \setminus I _ { \tau }$ satisfies $| f ^ { \star } - f ( x ) - \tau | \leq \eta$ . We require (1) to hold simultaneously for all $0 < \tau \leq T$ on a single event of probability at least $1 - \delta$ , where $T$ is the largest performance loss under consideration and $\delta$ is a prespecified upper bound on the probability that the simultaneous guarantee does not hold. On that event, all design points in $I _ { \tau }$ meet the performance tolerance, and all design points outside $O _ { \tau }$ fail it. The observations leave membership undetermined only in $O _ { \tau } \setminus I _ { \tau }$ . The second inclusion confines these points to a band of width $\eta$ on either side of the value threshold $f ^ { \star } - \tau ;$ it bounds response diferences rather than spatial distances. For example, with $\tau = 0 . 1 0$ and $\eta = 0 . 0 1$ , a point with performance gap below 0.09 must belong to $I _ { \tau }$ , while one with performance gap above 0.11 must lie outside $O _ { \tau }$ . Because a single event covers all tolerances, a user may choose $\tau \in ( 0 , T ]$ after seeing the output while keeping probability at least $1 - \delta$ that the set inclusions and region estimation error bound in (1) hold at the selected tolerance.

At $\tau = 0$ , the strict inequality in the definition requires a negative performance gap, or equivalently $f ( x ) > f ^ { \star }$ , which is impossible. Thus the region is known exactly, $S _ { 0 } = I _ { 0 } = O _ { 0 } = \emptyset$ , and no observations are needed for this set-valued output. A maximizer belongs to $S _ { \tau }$ for any positive $\tau .$ When SASORI uses $T = 0$ , it devotes all of its sampling to improving the estimated optimal point.

## 2.2 From Region Bounds to Decisions

The inner and outer sets support decisions involving a known secondary criterion and quantify how much of the design space remains unclassified. In this subsection, we provide an example illustrating both uses through a service-level and resource-cost decision.

Return to the service-system example of Section 1, where x specifies service rates and routing probabilities, $f ( x )$ is the expected fraction of customers meeting the service target, and $c _ { 0 } ( x )$ is a known resource cost. Before sampling, the manager specifies a largest acceptable service loss T. After observing the simultaneous region guarantees, the manager may choose any $\tau \in ( 0 , T ]$ based on the resulting service–cost tradeof, and can change it after seeing the performance of decision, without simulating the system again. This data-dependent choice keeps the same confidence guarantee, because the region guarantee holds for any such tolerance on a single event.

For the chosen tolerance, the operating-cost decision is bracketed by

$$
\operatorname* { i n f } _ { x \in O _ { \tau } } c _ { 0 } ( x ) \ \leq \ \operatorname* { i n f } _ { x \in S _ { \tau } } c _ { 0 } ( x ) \ \leq \ \operatorname* { i n f } _ { x \in I _ { \tau } } c _ { 0 } ( x ) , \qquad \operatorname* { i n f } \varnothing = + \infty .
$$

A nearly least-cost inner-set configuration is guaranteed to meet the service tolerance, while the outerset value bounds the further cost reduction that could still be available among unclassified design points. Infima are used because the strict regions need not contain a cost-minimizing configuration.

## 2.3 Kernel Ridge Regression and Smoothness

To recover the response surface from noisy observations, we use the Gaussian process (GP) regression predictor, whose posterior mean coincides with the kernel ridge regression estimator for the same kernel and regularization parameter. Throughout, we treat f as a fixed deterministic function in a reproducing kernel Hilbert space (RKHS), rather than as a sample path of a GP prior.

For a positive definite kernel k, its RKHS $\mathcal { H } _ { k } ( \mathcal { X } )$ is a Hilbert space of functions on X such that $k ( x , \cdot ) \in \mathcal { H } _ { k } ( \mathcal { X } )$ and, for any $g \in \mathcal { H } _ { k } ( \mathcal { X } ) , g ( x ) = \langle g , k ( x , \cdot ) \rangle _ { \mathcal { H } _ { k } } , \qquad x \in \mathcal { X }$ , where $\langle \cdot , \cdot \rangle _ { \mathscr { H } _ { k } }$ denotes the inner product of RKHS. In addition, $\| \cdot \| _ { \mathcal { H } _ { k } }$ denotes the RKHS norm. For detailed introduction, we refer to Wendland (2004).

We use a normalized Matérn kernel with fixed smoothness $\nu > 0$ and length scale $\ell > 0$ . For $r = \| x - z \| > 0$ , its form is

$$
k _ { \nu , \ell } ( x , z ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \frac { \sqrt { 2 \nu } r } { \ell } \right) ^ { \nu } K _ { \nu } \left( \frac { \sqrt { 2 \nu } r } { \ell } \right) , \qquad k _ { \nu , \ell } ( x , x ) = 1 ,\tag{2}
$$

where Γ is the gamma function and $K _ { \nu }$ is the modified Bessel function of the second kind. The parameter ν controls smoothness, and ℓ determines the spatial correlation scale (Rasmussen and Williams 2006). We assume $\| f \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B$ for a fixed and known $B > 0$ . The Matérn RKHS is norm-equivalent to the Sobolev space $\hat { H } ^ { s } ( \mathcal { X } )$ with $s = \nu + d / 2$ on the domain X (Wendland 2004). Define the kernel metric

$$
d _ { k } ( x , z ) = \| k ( x , \cdot ) - k ( z , \cdot ) \| _ { \mathcal { H } _ { k } } = \sqrt { k ( x , x ) + k ( z , z ) - 2 k ( x , z ) } .\tag{3}
$$

The reproducing property and the Cauchy–Schwarz inequality give

$$
| f ( x ) | \leq B , | f ( x ) - f ( z ) | \leq B d _ { k } ( x , z ) .\tag{4}
$$

Since the Matérn spectral density $\widehat { k }$ satisfies (Rasmussen and Williams 2006)

$$
\widehat { k } ( \omega ) \asymp ( 2 \nu / \ell ^ { 2 } + \| \omega \| ^ { 2 } ) ^ { - ( \nu + d / 2 ) } ,
$$

we have

$$
d _ { k } ( x , z ) ^ { 2 } = \int _ { \mathbb { R } ^ { d } } \lvert e ^ { i \omega ^ { \top } ( x - z ) } - 1 \rvert ^ { 2 } \widehat { k } ( \omega ) d \omega \leq C _ { \theta } \lVert x - z \rVert ^ { 2 \theta } \int _ { \mathbb { R } ^ { d } } \lVert \omega \rVert ^ { 2 \theta } \widehat { k } ( \omega ) d \omega = L _ { \theta } ^ { 2 } \lVert x - z \rVert ^ { 2 \theta } ,
$$

also because $1 - \cos t \leq C _ { \theta } | t | ^ { 2 \theta }$ and the integral is finite for $\theta < \nu$ . Hence, for any fixed $0 < \theta <$ min $\{ \nu , 1 \}$ , there is a constant $L _ { \theta }$ such that $d _ { k } ( x , z ) \leq L _ { \theta } \| x - z \| ^ { \theta }$ . This Hölder continuity bound is used to choose the confidence-bound parameters of the algorithm in Section 4.

## 3 The Irreducible Cost of Region Learning

Before introducing the algorithm, we first ask how much simulation efort is unavoidable when the desired output is a family of acceptable regions. We show that, even when an optimal design point and the optimal value are known, learning near-optimal region can still be expensive, which shows the essential diference between region learning and regret minimization.

Suppose an optimal design point $x ^ { \star }$ maximizes $f ,$ with value $f ^ { \star } = f ( x ^ { \star } )$ . Knowing $( x ^ { \star } , f ^ { \star } )$ fixes the threshold $f ^ { \star } - \tau$ , but leaves the responses at other design points unknown. For example, two surfaces can share this optimizer and value while assigning a particular design point performance gaps $\tau - 4 \eta$ and $\tau + 4 \eta .$ , respectively. A procedure must distinguish these two cases to construct valid inner and outer sets. We show that this task can remain expensive even when $( x ^ { \star } , f ^ { \star } )$ is known before sampling. Specifically, in the worst case over the fixed smoothness and covering class with $\alpha > \nu ,$ any procedure needs at least a constant multiple of $\eta ^ { - ( 2 + d / \nu ) } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \}$ evaluations at fixed confidence. Thus, knowing the best design point and its value does not remove the simulation efort needed to identify acceptable alternatives.

The simultaneous guarantee concerns all tolerances in $[ 0 , T ]$ together. If instead, one only consider a single positive $\tau$ fixed before sampling, the procedure only needs to determine membership relative to the one threshold $f ^ { \star } - \tau$ . Once a design point is classified to be acceptable or unacceptable, its response need not be estimated more precisely for further decisions. If $f ^ { \star }$ is unknown, however, additional evaluations at a potentially optimal design point may still be needed to locate the threshold accurately. Thus learning one prespecified region and learning all regions over a range of tolerances impose diferent requirements. We define their evaluation budgets separately and give a lower bound for each.

The lower bounds are stated on a function class defined by a geometric covering condition, imposed on the same family of regions that forms the output. Let

$$
S _ { u + } ( f ) : = \bigcap _ { v > u } S _ { v } ( f ) = \{ x \in \mathcal { X } : f ^ { \star } - f ( x ) \leq u \} , \qquad u \geq 0 .
$$

Thus, the subscript u+ adds the points with performance gap exactly u. In particular, $S _ { 0 } ( f )$ is empty, whereas $S _ { 0 + } ( f )$ is the set of maximizers. Let $\textstyle { \mathcal { N } } ( A , r )$ be the smallest number of closed Euclidean radius-r balls covering A. For fixed $B , c _ { \rho } , u _ { 0 } , \alpha > 0$ and integer $K \geq 1$ , define $\mathcal { F } _ { K } ( B , c _ { \rho } , u _ { 0 } , \alpha )$ to consist of functions satisfying

$$
\| f \| _ { \mathcal H _ { k } ( \mathcal X ) } \leq B , \qquad \mathcal N ( S _ { u + } ( f ) , c _ { \rho } u ^ { 1 / \alpha } ) \leq K , \qquad 0 < u \leq u _ { 0 } .
$$

The same $B , c _ { \rho } , u _ { 0 } , \alpha$ apply to all functions in the class and remain fixed as K varies. Here, K bounds the number of balls of radius $c _ { \rho } u ^ { 1 / \alpha }$ needed to cover the u-near-optimal set at any level $u \leq u _ { 0 }$

For $T > 0 , \eta > 0$ , and $0 < \delta < 1 / 2$ , we call a procedure $( \eta , T , \delta )$ -valid on $\mathcal { F } _ { K }$ if, for any $f \in \mathcal { F } _ { K }$ 2 with probability at least $1 - \delta$ , it returns sets $( I _ { \tau } , O _ { \tau } ) _ { 0 \leq \tau \leq T }$ satisfying, simultaneously,

$$
I _ { \tau } \subseteq S _ { \tau } \subseteq O _ { \tau } , \qquad O _ { \tau } \setminus I _ { \tau } \subseteq \{ x : | f ^ { \star } - f ( x ) - \tau | \leq \eta \} , \qquad 0 \leq \tau \leq T .\tag{5}
$$

Let $n _ { \mathrm { a l l } } ( \eta , T , \delta ; \mathcal { F } _ { K } )$ be the smallest deterministic evaluation budget for which such a procedure exists. The requirement in (5) combines correct set inclusions with an accuracy bound on undetermined membership. The following lemma shows how much information this requirement provides when it holds simultaneously over the tolerance range.

Lemma 3.1. Let $T > 0$ and $\eta > 0$ , and let $( I _ { \tau } , O _ { \tau } ) _ { 0 \leq \tau \leq T }$ be random output sets with measurable membership for all fixed design points and tolerances. Suppose that, on a measurable event $\mathcal { E } _ { f }$ , (5) holds for all $\tau \in [ 0 , T ]$ and all $x \in \mathcal { X }$ . Define

$$
\widehat { d } ( x ) = \operatorname* { i n f } \big ( \{ q \in \mathbb { Q } \cap [ 0 , T ] : x \in I _ { q } \} \cup \{ T \} \big ) .
$$

Then ${ \widehat { d } } ( x )$ is a measurable random variable for each fixed x. On $\mathcal { E } _ { f }$

$$
\operatorname* { m i n } \{ f ^ { \star } - f ( x ) , T \} \leq \widehat { d } ( x ) \leq \operatorname* { m i n } \{ f ^ { \star } - f ( x ) + \eta , T \} , x \in \mathcal { X } .\tag{6}
$$

In particular, for an $( \eta , T , \delta )$ )-valid procedure, these inequalities hold simultaneously over $\mathcal { X }$ with probability at least $1 - \delta$

Lemma 3.1 shows that a simultaneous region guarantee determines the performance gap truncated at T, namely the function $x \mapsto$ min $\{ f ^ { \star } - f ( x ) , T \}$ , uniformly to accuracy η. This is more information than either an estimated optimal point or membership in a single acceptable region provides, since the output distinguishes performance levels throughout the part of the design space within T of the optimum. This is particularly useful in practice when the acceptable performance loss is chosen after the simulation results are available, for example when a manager compares stafing costs under several service-level tolerances before selecting a final design.

The following theorem quantifies the evaluations required for simultaneous region accuracy on the fixed covering class. Its last claim shows that the bound persists even when the optimizing design point and the optimal value are the same for all functions in the construction and are known in advance.

Theorem 3.1 (Simultaneous-region minimax lower bound). Fix a compact $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ containing an interior Euclidean ball, a normalized Matérn kernel with $\nu > 0 , \alpha > \nu ,$ and positive $B , c _ { \rho } , u _ { 0 } , \sigma$ Consider the fixed covering class $\mathcal { F } _ { K } ( B , c _ { \rho } , u _ { 0 } , \alpha )$ and independent Gaussian observation noise with variance $\sigma ^ { 2 }$ . There are constants $K _ { \mathrm { r e g } } \in \mathbb { N }$ and $T _ { \mathrm { r e g } } , c > 0$ , depending only on these fixed parameters, such that, for any integer $K \geq K _ { \mathrm { r e g } } , 0 < T \leq T _ { \mathrm { r e g } } , 0 < \eta \leq T / 1 2 8$ , and $0 < \delta \leq 1 / 1 0$

$$
\begin{array} { r } { n _ { \mathrm { a l l } } ( \eta , T , \delta ; \mathcal { F } _ { K } ) \ge c \sigma ^ { 2 } \mathrm { k l } ( \delta , 1 - \delta ) \eta ^ { - 2 - d / \nu } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} , } \end{array}\tag{7}
$$

where $\mathrm { k l } ( p , q ) = p \log ( p / q ) + ( 1 - p ) \log ( ( 1 - p ) / ( 1 - q ) )$ . The bound also holds on a subclass in which a fixed design point $x ^ { \star }$ is the unique global maximizer and $f ( x ^ { \star } ) = f ^ { \star } = 0$ for all functions, even when the procedure is given $( x ^ { \star } , 0 )$ before sampling.

Theorem 3.1 shows that the lower bound separates the dependence on the target error η from the efect of near-optimal geometry, and remains valid even when the optimizer and its value are known. First, the factor $\eta ^ { - ( 2 + d / \nu ) }$ depends only on the accuracy and has the same exponent of $\eta ^ { - 1 }$ as estimating f to within η at under sup-norm, which requires of order $\eta ^ { - ( 2 + d / \nu ) }$ evaluations up to logarithmic factors. The geometric factor min $\{ 1 , K T ^ { d / \alpha } \}$ describes the reduction associated with the size of the requested region. Second, the covering condition does not reduce this exponent, in contrast to the simple regret rates of Section 5.3, because the band where classification remains undetermined is a shell inside the near-optimal set rather than a shrinking neighborhood of the optimizer. Third, the bound is unafected by knowing the optimizer. In the lower-bound construction, the baseline and all perturbed functions attain their unique maximum at the same point $x ^ { \star }$ , with value zero, so the procedure may be given the coordinates of $x ^ { \star }$ and the value zero before collecting any observations. However, these quantities contain no information about which perturbation is present at the other design points. The lower bound therefore measures the cost of learning the regions after the optimization problem has been solved.

A lower bound for one prespecified tolerance. Let $n _ { \mathrm { o n e } } ( \eta , \tau , \delta ; { \mathcal F } _ { K } )$ be the smallest deterministic evaluation budget for a procedure that, for any $f \in \mathcal { F } _ { K }$ , satisfies (1) at a single prespecified $\tau > 0$ , simultaneously over all design points with probability at least $1 - \delta .$ . Since one pair of sets can be extracted from a simultaneous output, $n _ { \mathrm { o n e } } ( \eta , \tau , \delta ; \mathcal { F } _ { K } ) \le n _ { \mathrm { a l l } } ( \eta , \tau , \delta ; \mathcal { F } _ { K } )$

Theorem 3.2 (A fixed-tolerance lower bound). Fix the domain, normalized Matérn kernel, and class parameters of Theorem $3 . 1 ,$ including $\alpha > \nu _ { ; }$ , and use independent Gaussian observation noise with variance $\sigma ^ { 2 }$ . There are constants $K _ { 1 } \in \mathbb { N }$ and $\tau _ { 0 } , c > 0$ , depending only on those fixed parameters, such that, for any integer $K \ge K _ { 1 } , 0 < \tau \le \tau _ { 0 } , 0 < \eta \le \tau / 3 2$ , and $0 < \delta \leq 1 / 1 0$

$$
n _ { \mathrm { o n e } } ( \eta , \tau , \delta ; \mathcal { F } _ { K } ) \ge c \sigma ^ { 2 } \mathrm { k l } ( \delta , 1 - \delta ) \eta ^ { - 2 - ( d - 1 ) / \nu } \bigl ( \operatorname* { m i n } \{ 1 , K \tau ^ { d / \alpha } \} \bigr ) ^ { ( d - 1 ) / d } .\tag{8}
$$

The bound holds on a subclass with a common unique global maximizer $x ^ { \star }$ and value zero, even when $( x ^ { \star } , 0 )$ is provided before sampling.

For fixed admissible $K , \tau , \delta _ { ; }$ , the exponent of $\eta ^ { - 1 }$ in this lower bound is $2 + ( d - 1 ) / \nu ,$ , which is smaller by $1 / \nu$ than the simultaneous exponent $2 + d / \nu$ . This reduction occurs because the perturbations are placed along a single level set of the baseline response surface, which is $( d - 1 )$ dimensional, instead of throughout a d-dimensional set of designs.

Region learning versus regret minimization. Simple regret measures the performance gap of the estimated optimal point, whereas cumulative regret sums these gaps over all evaluated design points (Scarlett et al. 2017). In batched kernel optimization, for example, a design point can be eliminated once its upper confidence bound falls below the largest lower confidence bound (Li and Scarlett 2022). However, such a point may still require refinement to attain the requested region accuracy. Theorems 3.1 and 3.2 together separate region learning from both simple regret and cumulative regret minimization. On the subclass used to establish each lower bound, a procedure given the common optimizer can select it and allocate all evaluations to it, attaining $r _ { N } = R _ { N } = 0$ Yet such observations do not distinguish the response surfaces away from that optimizer, and valid region learning still requires the evaluations prescribed by the theorems. This remains true for a single tolerance fixed in advance, where the information that remains to be determined lies near the corresponding performance boundary. For simultaneous outputs, that information extends throughout a shell as the tolerance varies. Hence neither a design with a small optimality gap nor small cumulative regret guarantees an accurate acceptable region. The cost of obtaining both outputs jointly is quantified in Section 5.4.

## 4 Shared-confidence Adaptive Sampling for Optimization and Region Identification (SASORI)

Section 3 shows that optimizing a response surface does not by itself identify its near-optimal regions. To pursue both objectives under a single evaluation budget, we introduce Shared-confidence Adaptive Sampling for Optimization and Region Identification (SASORI). The policy allocates efort explicitly to each objective: a region quota $m \in \{ 0 , \ldots , N \}$ reserves m evaluations for region refinement, and the remaining $N - m$ evaluations serve optimization. Both tracks share their completed confidence bands, so each can improve the other’s output. The quota therefore controls the tradeof between region accuracy and cumulative regret. For a specified maximum tolerance $T \geq 0 ,$ , the policy returns an estimated optimal point $\widehat { x } _ { N }$ and inner and outer sets for all tolerances $0 \leq \tau \leq T$

SASORI collects evaluations in batches, each assigned to optimization or region learning. $\mathrm { O p - }$ timization batches seek better designs, while region-learning batches determine which designs lie within the requested performance tolerances of the optimum. Each completed nonempty batch tightens confidence bounds used by both tracks. These tighter bounds can improve the estimated optimal point and the estimated regions, regardless of which track collected it. Algorithm 1 schedules the phases and calls the sample-allocation and band-sharing subroutines in Algorithms 2 and $^ { 3 , }$ respectively. Both subroutines use its fixed inputs and calibrated $( \beta _ { N } , \zeta _ { N } )$ . State variables updated by either subroutine are maintained by the main algorithm.

Algorithm 1 SASORI   
Require: $N \geq 2 , \vartheta > 1 , T \geq 0 , m \in \{ 0 , \ldots , N \} \left( m = 0 \mathrm { ~ i f ~ } T = 0 \right) , \mathcal { X } , k , B , \sigma , \lambda , \delta , L _ { \theta } , D _ { \mathcal { X } } , x _ { 0 } \in \mathcal { X } _ { \partial \mathcal { R } } ,$   
and a fixed rule for breaking ties.   
Ensure: The latest saved estimated optimal point and region bounds for all $0 \leq \tau \leq T$   
1: Compute $( \beta _ { N } , \zeta _ { N } )$ by (9), with $\theta = \operatorname* { m i n } \{ \nu , 1 \} / 2 .$   
2: Set $( N _ { 0 } , N _ { 1 } ) = ( N - m , m ) , ( T _ { 0 } ^ { \mathrm { a l g } } , T _ { 1 } ^ { \mathrm { a l g } } ) = ( 0 , T )$ , and $h _ { j } = t _ { j } = 0 , \bar { \varepsilon } _ { j } = + \infty$ for $j = 0 , 1$   
3: Set $\ell = - B , u = B , f = - B , \overline { { f } } = B , \widehat { x } = x _ { 0 } ;$ save the initial output and finish tracks assigned   
no evaluations.   
4: while some track is unfinished do   
5: Select unfinished track $j ,$ starting with 0 and alternating whole phases when both remain;   
set $\varepsilon = 2 B \vartheta ^ { - h _ { j } }$   
6: $\mathbf { I f } ~ \varepsilon < 2 \zeta _ { N }$ , allocate the remaining $N _ { j } - t _ { j }$ evaluations to ${ \widehat x } ;$ set $t _ { j } = N _ { j }$ , finish track $j ,$ and   
continue.   
7: Fix $( \underline { { f } } , W , G )$ by (10), using $\boldsymbol { T } _ { j } ^ { \mathrm { a l g } }$ and the current shared bands.   
8: Call Algorithm 2 for this phase.   
9: If the phase is incomplete, discard its batch, finish track $j ,$ and continue.   
10: Call Algorithm 3 with the completed batch.   
11: If Algorithm 3 returns “inconsistent,” allocate the remaining $N - t _ { 0 } - t _ { 1 }$ evaluations to $x _ { 0 }$   
and return $x _ { 0 }$ and “no region guarantee.”   
12: end while   
13: return the latest saved output.

Sample allocation. The inputs $B , \sigma , k , \lambda ,$ and $L _ { \theta }$ (norm bound, noise level, kernel, regularization parameter, and Hölder constant of the kernel from Section 2.3) are specified in Section 5.1, and are fixed. Each phase uses only its own observations for the GP fit:

$$
\mu _ { n } ( x ) = k _ { n } ( x ) ^ { \top } ( K _ { n } + \lambda I ) ^ { - 1 } y _ { 1 : n } , \qquad s _ { n } ^ { 2 } ( x ) = k ( x , x ) - k _ { n } ( x ) ^ { \top } ( K _ { n } + \lambda I ) ^ { - 1 } k _ { n } ( x ) ,
$$

where $\begin{array} { r } { k _ { n } ( x ) = ( k ( x _ { 1 } , x ) , \ldots , k ( x _ { n } , x ) ) ^ { \top } , K _ { n } = ( k ( x _ { i } , x _ { j } ) ) _ { i , j = 1 } ^ { n } . } \end{array}$ , and $y _ { 1 : n }$ contains the responses from the current phase. The function $\mu _ { n }$ is the kernel ridge regression predictor, and $s _ { n } ^ { 2 }$ is the associated GP posterior variance. Within a phase, the choice of sampling points and the stopping test use the bands from completed phases and $s _ { n } ,$ which depends only on the sampled locations. The current responses $y _ { 1 : n }$ are withheld from these decisions. If the phase completes, those responses are then used to compute $\mu _ { n }$ and update the shared bands.

For $N \geq 2$ and $\delta \in ( 0 , 1 )$ , fix $\theta \in ( 0 , \operatorname* { m i n } \{ \nu , 1 \} )$ and use the deterministic net-size bound $M _ { N } = ( 1 + 2 D \chi N ^ { 4 / \theta } ) ^ { d }$ , which bounds the number of points in a finite $N ^ { - 4 / \theta }$ -net of $\mathcal { X } ,$ , where $\begin{array} { r } { D _ { \mathcal { X } } = \operatorname* { s u p } _ { x , z \in \mathcal { X } } \| x - z \| } \end{array}$ is the diameter of $\mathcal { X }$ . The confidence parameters are

$$
\begin{array} { r l } & { \beta _ { N } = B + \frac { \sigma } { \sqrt { \lambda } } \sqrt { 2 \log ( 4 M _ { N } N ^ { 2 } / \delta ) } , } \\ & { Y _ { N } = B + \sigma \sqrt { 2 \log ( 4 N / \delta ) } , } \\ & { \zeta _ { N } = L _ { \theta } { N ^ { - 4 } } \left( B + \frac { \sqrt { N } Y _ { N } } { 2 \sqrt { \lambda } } + \beta _ { N } \right) . } \end{array}\tag{9}
$$

For example, if $\delta = N ^ { - 2 }$ , then $\beta _ { N } = O ( \sqrt { \log N } )$ and $\zeta _ { N } = \widetilde { O } ( N ^ { - 7 / 2 } )$

To describe one phase of a given track, use $T$ for that track’s tolerance, h for its local phase counter, and $\varepsilon _ { h } = 2 B \vartheta ^ { - h }$ for its accuracy level. The fixed parameter $\vartheta > 1$ determines the reduction in accuracy level between successive phases. For the analysis, write $L _ { \vartheta } = 4 ( \vartheta + 1 )$ ; this equals 12 when $\vartheta = 2$ . Let $\ell _ { h - 1 } , u _ { h - 1 }$ be the shared confidence bounds available at the start of the phase, including the contributions of any intervening completed phases of the other track, with initial values −B and $B ,$ respectively. The following quantities are held fixed throughout the phase:

$$
\underline { { f } } _ { h - 1 } = \operatorname* { m a x } _ { x \in \mathcal { X } } \ell _ { h - 1 } ( x ) , W _ { h - 1 } = u _ { h - 1 } - \ell _ { h - 1 } , G _ { h } ( x ) = [ u _ { h - 1 } ( x ) - \underline { { f } } _ { h - 1 } + T + 4 \varepsilon _ { h } ] _ { + } .\tag{10}
$$

Then the next sampling point is obtained via maximizing the acquisition function

$$
a _ { h , n } ( x ) = \operatorname* { m i n } \{ G _ { h } ( x ) , W _ { h - 1 } ( x ) , 2 \beta _ { N } s _ { n } ( x ) + 2 \zeta _ { N } \} ,\tag{11}
$$

i.e., $x _ { h , n + 1 } \in$ arg max ${ \bf \Phi } _ { x \in \mathcal { X } } a _ { h , n } ( x )$ . The screening function $G _ { h }$ screens out design points that already perform too poorly to matter at tolerance $T ,$ the maintained width $W _ { h - 1 }$ preserves the information in completed batches, and the confidence-band width from the current batch $2 \beta _ { N } s _ { n } + 2 \zeta _ { N }$ directs new evaluations toward locations where the current batch still leaves large uncertainty. The phase ends, possibly without any evaluation, when

$$
\operatorname* { m a x } _ { x \in \mathcal { X } } a _ { h , n } ( x ) \leq 2 \varepsilon _ { h } .\tag{12}
$$

Algorithm 2 implements the sampling loop for one phase. It checks the stopping condition before the remaining track budget, so a phase may complete at the last permitted evaluation or without any new evaluation. Its decisions use the fixed phase quantities and the variance determined by the current batch locations.

## Algorithm 2 Sample allocation

Require: Track j, accuracy level $\varepsilon = \varepsilon _ { h _ { j } }$ , fixed $( G , W )$ , evaluation count $t _ { j }$ , quota $N _ { j }$ , and the   
common inputs of Algorithm 1.   
Ensure: Updated $t _ { j } ,$ , batch $( n , { \mathcal { D } } , s _ { n } )$ , and status “completed” or “incomplete.”   
1: Set $n = 0 , \mathcal { D } = \emptyset ;$ , and $s _ { 0 } = 1 .$   
2: loop   
3: Choose $x \in$ arg $\mathrm { m a x } _ { z \in \mathcal { X } } a _ { h _ { j } , n } ( z )$ using (11); set $M = { a _ { h } } _ { j , n } ( x )$   
4: If $M \leq 2 \varepsilon$ , return the batch with status “completed.”   
5: If $t _ { j } = N _ { j }$ , return the batch with status “incomplete.”   
6: Observe $y = f ( x ) + \varepsilon _ { t } ,$ , append $( x , y )$ to D, and increment n and $t _ { j }$   
7: Update $s _ { n }$ from the batch locations alone, without using the batch responses.   
8: end loop

Remark 4.1 (Selecting the next design point). Exact maximization of ${ a } _ { h _ { j } , n }$ in Algorithm 2 is suficient but not necessary for selecting a point to evaluate. It is enough to find any $x \in \mathcal { X }$ such that $a _ { h , n } ( x ) > 2 \varepsilon _ { h }$ . For $\varepsilon _ { h } \geq 2 \zeta _ { N }$ , this implies $G _ { h } ( x ) > 2 \varepsilon _ { h } , W _ { h - 1 } ( x ) > 2 \varepsilon _ { h }$ , and $s _ { n } ( x ) >$ $( \varepsilon _ { h } - \zeta _ { N } ) / \beta _ { N } \geq \varepsilon _ { h } / ( 2 \beta _ { N } )$ These are the properties used in the proof of Lemma $B . 3$ and the localization argument of Proposition $A . { \mathcal { Q } } .$ The same bounds therefore apply if point selection does not use responses from the current phase and the stopping condition (12) is retained.

Band sharing and output updates. For a completed nonempty phase, let $w _ { n } = \beta _ { N } s _ { n } + \zeta _ { N }$ and form the band $\left[ \mu _ { n } - w _ { n } , \mu _ { n } + w _ { n } \right]$ . Define the shared confidence event

$$
\begin{array} { r l } & { \mathcal { E } _ { \delta } = \big \{ | f ( x ) - \mu _ { n } ( x ) | \leq \beta _ { N } s _ { n } ( x ) + \zeta _ { N } , \mathrm { ~ f o r ~ a l l ~ p h a s e s } , } \\ & { \qquad \mathrm { ~ a l l ~ s a m p l e ~ s i z e s ~ o b t a i n e d ~ w i t h i n ~ a ~ p h a s e ~ b y ~ t i m e ~ } N , \mathrm { ~ a n d ~ a l l ~ } x \in \mathcal { X } \big \} . } \end{array}\tag{13}
$$

On ${ \mathcal { E } } _ { \delta }$ , bands from both tracks can be intersected without excluding the true response surface.

When a nonempty phase completes, its band is intersected with the maintained confidence bounds on all of $x ,$ , while an empty phase leaves them unchanged. With the resulting confidence bounds $\ell _ { h } , u _ { h }$ , set

$$
\underline { { f } } _ { h } = \operatorname* { m a x } _ { x \in \mathcal { X } } \ell _ { h } ( x ) , \qquad \overline { { f } } _ { h } = \operatorname* { m a x } _ { x \in \mathcal { X } } u _ { h } ( x ) , \qquad \widehat x _ { h } \in \mathop { \mathrm { a r g } \operatorname* { m a x } } _ { x \in \mathcal { X } } \ell _ { h } ( x ) ,\tag{14}
$$

and the output regions in this phase are

$$
I _ { h , \tau } = \{ x : \ell _ { h } ( x ) > \overline { { f } } _ { h } - \tau \} , \qquad O _ { h , \tau } = \{ x : u _ { h } ( x ) > \underline { { f } } _ { h } - \tau \} , \qquad 0 < \tau \leq T .\tag{15}
$$

Set $I _ { h , 0 } = O _ { h , 0 } = \emptyset$ . Note that these confidence bounds apply to all tolerances simultaneously, so $\tau \leq T$ can be chosen after observing the output.

Algorithm 3 performs these updates only after a completed phase. A completed empty phase leaves the confidence bounds unchanged but still records the achieved accuracy level and advances the phase counter. If the updated lower and upper bounds are inconsistent, it returns that status before updating extrema or saved outputs; Algorithm 1 then applies the stated rule for the remaining total evaluation budget.

Algorithm 3 Band sharing   
Require: Track $j ,$ accuracy level ε, completed batch $( n , { \mathcal { D } } , s _ { n } )$ , and the shared state and fixed   
inputs of Algorithm 1.   
Ensure: Updated shared state and saved output, or status “inconsistent.”   
1: if $n > 0$ then   
2: Fit $\mu _ { n }$ using $\mathcal { D }$ and set $w _ { n } = \beta _ { N } s _ { n } + \zeta _ { N } .$   
3: On $x ,$ set $\ell = \operatorname* { m a x } \{ \ell , \mu _ { n } - w _ { n } \}$ and u = min $\{ u , \mu _ { n } + w _ { n } \}$   
4: end if   
5: If $\ell ( x ) > u ( x )$ for some $x \in { \mathcal { X } } ,$ return status “inconsistent.”   
6: Set $\bar { \varepsilon } _ { j } = \varepsilon$ and $h _ { j }  h _ { j } + 1$   
7: Update $( \underline { { f } } , \overline { { f } } , \widehat { x } )$ by (14), using the fixed rule for breaking ties.   
8: Save $( \widehat { x } , \underline { { f } } , \overline { { f } } , \ell , u , \bar { \varepsilon } _ { 0 } , \bar { \varepsilon } _ { 1 } )$ and the regions in (15) for the full tolerance range $0 < \tau \leq T ;$ set   
$I _ { 0 } = O _ { 0 } = \emptyset$   
9: return status “consistent.”

On ${ \mathcal { E } } _ { \delta }$ , a completed phase of track $j$ with accuracy level $\bar { \varepsilon } _ { j }$ yields $\overline { { f } } - \underline { { f } } \leq 2 \bar { \varepsilon } _ { j }$ and an estimated optimal point with performance gap at most $2 \bar { \varepsilon } _ { j }$ . A completed region phase with local index h also gives the simultaneous region guarantee (1) with accuracy $\eta = 4 \bar { \varepsilon } _ { 1 } = 4 \varepsilon _ { h } = 8 B \vartheta ^ { - h }$ . These guarantees remain valid after later intersections, since the upper confidence bound can only decrease, the lower confidence bound can only increase, and the set where a given track needs a small width can only shrink. The following proposition makes this precise.

Proposition 4.1. Suppose track j completed a phase with accuracy level ε¯, leaving confidence bounds $( \ell ^ { \mathrm { o l d } } , u ^ { \mathrm { o l d } } )$ with $f ^ { \mathrm { o l d } } = \operatorname* { m a x } _ { \mathcal { X } } \ell ^ { \mathrm { o l d } }$ , and that on ${ \mathcal { E } } _ { \delta }$

$$
u ^ { \mathrm { o l d } } ( x ) - \ell ^ { \mathrm { o l d } } ( x ) \leq 2 \bar { \varepsilon } w h e n e v e r { u } ^ { \mathrm { o l d } } ( x ) > \underline { { f } } ^ { \mathrm { o l d } } - T _ { j } - 2 \bar { \varepsilon } , \underline { { f } } ^ { \star } - \underline { { f } } ^ { \mathrm { o l d } } \leq 2 \bar { \varepsilon } ,\tag{16}
$$

where $T _ { j } = T _ { j } ^ { \mathrm { a l g } }$ . Let $( \ell , u )$ be the confidence bounds at any later phase boundary, obtained $b y$ intersecting confidence bands from completed batches, and $\underline { { f } } = \operatorname* { m a x } _ { \mathcal { X } } \ell$ . Then the same two claims hold for $( \ell , u , f )$ with the same $\bar { \varepsilon } .$

Consequently, both tracks can use the same observations and confidence event ${ \mathcal { E } } _ { \delta }$ , without a separate data set or additional union bound. This is the sharing property used in our theoretical developments in Section 5. An incomplete final phase returns the latest saved output, and any empty phases after quota exhaustion leave the confidence bounds and estimated optimal point determined by the fixed rule for breaking ties unchanged. The band-sharing subroutine in Algorithm 3 checks whether the maintained lower and upper confidence bounds are inconsistent after each intersection, that is, whether $\ell ( x ) > u ( x )$ for some $x \in \mathcal { X }$ . If so, Algorithm 1 uses $x _ { 0 }$ for the remaining evaluations and returns it without a region guarantee. This safeguard makes the policy well defined for any observation sequence. On $\mathcal { E } _ { \delta } .$ , all maintained bands contain $f ( x )$ , so $\ell ( x ) \leq f ( x ) \leq u ( x )$ and the safeguard is never triggered. Its triggering probability is therefore at most $\delta .$ The absence of an inconsistency does not establish that the bounds contain the true response surface.

Remark 4.2. Algorithm 1 concerns all tolerances $\tau \in [ 0 , T ]$ simultaneously. For a single tolerance fixed before sampling, Section 5.5 specifies an alternative screening function and single-tolerance output within the same fixed-horizon phase mechanism.

In summary, SASORI combines a fixed budget split with information sharing through confidence bounds. The acquisition function concentrates evaluations on design points that remain uncertain and relevant to the selected track’s tolerance, while completed nonempty batches can improve both outputs. Thus, SASORI supports both design selection and region decisions within a single sequential simulation study, with the quota setting the allocation of evaluations between the two tracks.

## 5 Theoretical Guarantees for Joint Region Learning and Optimization

This section presents the theoretical guarantees. We first state the assumptions and establish evaluation budgets for optimization and simultaneous region learning in Section 5.1 and Section 5.2, respectively. Section 5.3 then introduces the pure-optimization regret benchmark and its matching lower bounds. Using those results, Section 5.4 quantifies the regret cost of region learning and the accuracy compatible with eficient optimization. Section 5.5 specializes the phase mechanism to a single tolerance fixed before sampling, and Section 5.6 completes the analysis with the critical endpoint $\alpha = \nu ,$ the case $2 \leq \alpha < \nu$ , and a suficient range for the Matérn lower rates without the covering condition.

## 5.1 Assumptions for the Theoretical Results

Our theoretical analysis imposes the following assumptions. Throughout the upper bounds for SASORI and their proofs, the phase ratio $\vartheta > 1$ is fixed. Their multiplicative constants may depend on $\vartheta ,$ in addition to the model and covering parameters specified in each result; the rate orders are unchanged for each fixed ϑ.

Assumption 5.1. The design space $\boldsymbol { \mathcal { X } } \subset \mathbb { R } ^ { d }$ is the closure of a bounded convex Lipschitz domain with nonempty interior and diameter $\begin{array} { r } { D _ { \mathcal { X } } = \operatorname* { s u p } _ { x , z \in \mathcal { X } } \| x - z \| } \end{array}$

Assumption 5.2. For some known $\sigma > 0$ , the noise $\varepsilon _ { t }$ is sub-Gaussian, $i . e . , \mathbb { E } [ \exp ( u \varepsilon _ { t } ) \mid \mathcal { F } _ { t - 1 } , x _ { t } ] \leq$ $\exp ( u ^ { 2 } \sigma ^ { 2 } / 2 )$ for all $u \in \mathbb { R }$ , where $\mathcal { F } _ { t - 1 }$ denotes the history of the first t−1 design points and responses.

Assumption 5.3. The kernel $k = k _ { \nu , \ell }$ is the normalized Matérn kernel (2) with fixed smoothness $\nu > 0$ and fixed length scale $\ell > 0$ , and thus $k ( x , x ) = 1$ . The objective function satisfies $\| f \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B$ for some known $B > 0$ . The regularization $\lambda > 0$ is fixed.

Remark 5.1 (Fixed regularization). For a nonempty batch $( n \geq 1 )$ , the predictor $\mu _ { n }$ is indeed the kernel ridge-regression solution

$$
\mu _ { n } = \underset { g \in \mathcal { H } _ { k } ( \mathcal { X } ) } { \arg \operatorname* { m i n } } \left\{ \sum _ { i = 1 } ^ { n } ( y _ { i } - g ( x _ { i } ) ) ^ { 2 } + \lambda \| g \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } ^ { 2 } \right\} .
$$

In the matched Matérn RKHS setting, $\mu _ { n }$ with a fixed λ can lead to the optimal rate of uniform $( L _ { \infty } )$ estimation under suitable space-filling designs, up to logarithmic factors.

Assumption 5.4. For fixed $\alpha , c _ { \rho } , u _ { 0 } > 0$ and integer $K \geq 1$

$$
\begin{array} { r } { \mathcal { N } \big ( S _ { u + } , c _ { \rho } u ^ { 1 / \alpha } \big ) \leq K , \qquad 0 < u \leq u _ { 0 } . } \end{array}\tag{17}
$$

Assumption 5.1 fixes the design space and makes continuous band extrema well defined. Convexity is assumed for mathematical simplicity, but we note that the upper bounds only needs compactness. Assumption 5.2 is a standard assumption that controls the tails of the simulation error conditional on previous observations. The same upper bound σ can be used for unequal replication variances.

When $\sigma$ is unknown, a conservative upper bound may be derived from the simulation model or calibrated using pilot replications under additional distributional assumptions. Assumption 5.3 expresses response-surface smoothness through a fixed kernel and norm bound. Assumption 5.4 bounds the number of spatial neighborhoods needed to cover all u-near-optimal design points at radius $c _ { \rho } u ^ { 1 / \alpha }$ , uniformly over $0 < u \le u _ { 0 }$ . The exponent α determines how the radius shrinks with the value level $u ,$ while K bounds the number of neighborhoods. A simple suficient condition is the following. For a nonempty set $A \subseteq { \mathcal { X } }$ , define dist $( x , A ) = \operatorname* { i n f } _ { a \in A } \| x - a \|$ . For a fixed set M of at most K maximum points, the error bound

$$
f ^ { \star } - f ( x ) \geq c _ { g } \mathrm { d i s t } ( x , { \mathcal { M } } ) ^ { \alpha } \quad \mathrm { w h e n e v e r } \ f ^ { \star } - f ( x ) \leq u _ { 0 }\tag{18}
$$

implies (17) with $c _ { \rho } = c _ { g } ^ { - 1 / \alpha }$ , where $c _ { g }$ may depend on K. Conversely, (17) can hold without any finite mode decomposition. Another suficient condition is growth around finitely many separated local optima, as formalized in Proposition 5.1.

Proposition 5.1. Let $\mathcal { X } \subset \mathbb { R } ^ { d }$ be compact and $f : \mathcal { X }  \mathbb { R }$ continuous. Fix $\alpha > 0$ and $\tau \geq 0$ Suppose the local optima form a finite nonempty set $\mathcal { M } _ { \tau } = \{ m _ { \ell } \} _ { \ell = 1 } ^ { K } , \Delta _ { \ell } = f ^ { \star } - f ( m _ { \ell } ) \in [ 0 , \tau ]$ Assume that there are $r _ { 0 } > 0$ , constants $\mu _ { 1 } , \ldots , \mu _ { K } > 0$ , and an exterior gap $g _ { \mathrm { 0 } } > \tau$ such that:

1. The closed balls $B _ { \ell } = \overline { { B } } ( m _ { \ell } , r _ { 0 } )$ are mutually disjoint and contained in int(X ). By the growth condition in item $\mathcal { Q } , \ : m _ { \ell }$ is the unique maximizer of f on $B _ { \ell }$

2. Within each ball,

$$
f ( m _ { \ell } ) - f ( x ) \geq \mu _ { \ell } \| x - m _ { \ell } \| _ { 2 } ^ { \alpha } , x \in \mathcal { B } _ { \ell } .\tag{19}
$$

3. For any $x \in \mathcal { X } \setminus \bigcup _ { \ell = 1 } ^ { K } B ( m _ { \ell } , r _ { 0 } )$ , one has $f ^ { \star } - f ( x ) \geq g _ { 0 }$ , where $B ( m _ { \ell } , r _ { 0 } )$ is the open ball. Set $\underline { { \mu } } = \operatorname* { m i n } _ { 1 \leq \ell \leq K } \mu _ { \ell }$ . For any choice $0 < u _ { 0 } < g _ { 0 }$ and all $0 < u \leq u _ { 0 }$

$$
S _ { u + } \subseteq \bigcup _ { \ell : \Delta _ { \ell } \leq u } \overline { { B } } \Bigg ( m _ { \ell } , \Bigg ( \frac { u - \Delta _ { \ell } } { \mu _ { \ell } } \Bigg ) ^ { 1 / \alpha } \Bigg ) .\tag{20}
$$

Consequently, Assumption 5.4 holds with this K, the same exponent $\alpha ,$ and $c _ { \rho } = \mu ^ { - 1 / \alpha } , i . e .$

$$
\begin{array} { r } { \mathcal { N } \big ( S _ { u + } , c _ { \rho } u ^ { 1 / \alpha } \big ) \leq \# \{ \ell : \Delta _ { \ell } \leq u \} \leq K , 0 < u \leq u _ { 0 } . } \end{array}\tag{21}
$$

In particular, $u _ { 0 } = \tau$ is admissible when $\tau > 0$

Proof. If $x \in S _ { u + }$ , then $f ^ { \star } - f ( x ) \leq u < g _ { 0 }$ . The exterior-gap condition therefore places x in some $B ( m _ { \ell } , r _ { 0 } ) . { \mathrm { ~ B y ~ } } ( 1 9 )$

$$
u \geq f ^ { \star } - f ( x ) = \Delta _ { \ell } + f ( m _ { \ell } ) - f ( x ) \geq \Delta _ { \ell } + \mu _ { \ell } \| x - m _ { \ell } \| _ { 2 } ^ { \alpha } .
$$

Thus $\Delta _ { \ell } \leq u$ and x belongs to the corresponding ball in (20), and a ball of radius zero denotes the singleton $\{ m _ { \ell } \}$ . The radius of each ball in (20) is at most $\mu ^ { - 1 / \alpha } u ^ { 1 / \alpha }$ , proving (21). □

## 5.2 Evaluation Complexity for Optimization and Region Learning

We now bound the number of simulation evaluations needed to reach a prescribed accuracy level. The result covers both pure optimization $( T = 0 )$ and simultaneous region learning $( T > 0 )$ . In the positive-tolerance range of Section 3, it yields rates in the accuracy, tolerance, and covering number bound that match the lower bound of Theorem 3.1.

We first present the following lemma that establishes $\mathbb { P } ( \mathcal { E } _ { \delta } ) \geq 1 - \delta$ for the shared confidence event defined in (13).

Lemma 5.1. Let X be a nonempty compact subset of $\mathbb { R } ^ { d }$ , and suppose Assumptions 5.2–5.3 hold. Run Algorithm 1 with $N \geq 2$ and $\delta \in ( 0 , 1 )$ , using the confidence parameters in (9). With probability at least $1 - \delta$ , simultaneously over all phases, all sample sizes within a phase by time $N _ { ; }$ , and all $x \in \mathcal { X } _ { i }$

$$
| f ( x ) - \mu _ { n } ( x ) | \leq \beta _ { N } s _ { n } ( x ) + \zeta _ { N } .\tag{22}
$$

Phase accuracy and the full-budget allocation. Fix a phase h of one track, with tolerance T and accuracy level $\varepsilon _ { h } = 2 B \vartheta ^ { - h }$ . Its acquisition function and maximum are

$$
a _ { n } ( x ) = \operatorname * { m i n } \{ G _ { h } ( x ) , W _ { h - 1 } ( x ) , 2 \beta _ { N } s _ { n } ( x ) + 2 \zeta _ { N } \} , \qquad a _ { n } ^ { \star } = \operatorname * { m a x } _ { x \in \mathcal { X } } a _ { n } ( x ) .\tag{23}
$$

The phase samples a maximizer when $a _ { n } ^ { \star } > 2 \varepsilon _ { h }$ and its track has budget remaining. A sampled point x therefore requires both $G _ { h } ( x ) > 2 \varepsilon _ { h }$ and $W _ { h - 1 } ( x ) > 2 \varepsilon _ { h }$ , that is, the point remains relevant to the tolerance and its maintained interval is still too wide. If $a _ { n } ^ { \star } \leq 2 \varepsilon _ { h }$ , the phase completes and updates the shared bands. If instead the track’s budget is exhausted first, the phase ends and its incomplete batch is discarded.

The following theorem establishes the evaluations needed to obtain an estimated optimal point with optimality gap at most $2 \varepsilon$ and simultaneous region bounds with region estimation error at most 4ε. It applies to the full-budget allocation at both zero and positive tolerances.

Theorem 5.1 (Evaluation complexity for optimization and region learning). Let X be a nonempty compact subset of $\mathbb { R } ^ { d }$ , and suppose Assumptions 5.2–5.3 hold. Run Algorithm 1 with fixed $\vartheta > 1$ $N \ge 2 , \delta \in ( 0 , 1 )$ , and $T \geq 0$ , using $m = 0$ for $T = 0$ and $m = N$ for $T > 0$ . On $\mathcal { E } _ { \delta }$ , which has probability at least $1 - \delta _ { i }$ , the following claims hold simultaneously for all targets $\varepsilon = 2 B \vartheta ^ { - H } \geq 2 \zeta _ { N }$ When phase H completes, its saved output satisfies

$$
\begin{array} { r l } & { \overline { { f } } _ { H } - \underline { { f } } _ { H } \leq 2 \varepsilon , \qquad f ^ { \star } - f ( \widehat { x } _ { H } ) \leq 2 \varepsilon , } \\ & { I _ { H , \tau } \subseteq S _ { \tau } \subseteq O _ { H , \tau } , \qquad 0 \leq \tau \leq T , } \\ & { O _ { H , \tau } \setminus I _ { H , \tau } \subseteq \{ x : | f ^ { \star } - f ( x ) - \tau | \leq 4 \varepsilon \} , \qquad 0 \leq \tau \leq T . } \end{array}\tag{24}
$$

A suficient number of evaluations for phase H to complete is

$$
C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \delta ) \big ( 1 + \varepsilon ^ { - ( 2 + d / \nu ) } \big ) .\tag{25}
$$

Under Assumption $5 . 4$ with $\alpha > \nu _ { ; }$ , uniformly for $0 \le T \le u _ { 0 } / 2$ , the bound improves to

$$
C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \delta ) \left[ 1 + \operatorname* { m i n } \left\{ \varepsilon ^ { - ( 2 + d / \nu ) } , K \varepsilon ^ { - 2 } + K ( T + \varepsilon ) ^ { d / \alpha } \varepsilon ^ { - ( 2 + d / \nu ) } \right\} \right] .\tag{26}
$$

The constants depend only on the fixed model parameters and, for the localized bound, the fixed covering parameters other than K, that is, they are independent of $K , T , N , \delta , \varepsilon$

Here N is the maximum number of evaluations declared before the run, whereas the theorem bounds the number actually needed to reach the requested accuracy. The dependence on N comes from calibrating $\beta _ { N }$ and $\zeta _ { N } ,$ , which provides simultaneous confidence through the declared horizon.

A larger horizon changes these confidence parameters, and hence the suficient evaluation count, even if the target is reached early. To plan a run for a specified ε, one can choose N so that the right-hand side of (25), or of (26), is at most N and $2 \zeta _ { N } \le \varepsilon$ . The corollary below makes such a choice explicitly and eliminates N from the final accuracy-based bound.

At $T = 0$ , the estimated optimal point satisfies $f ^ { \star } - f ( { \widehat x } _ { H } ) \leq 2 \varepsilon$ , while $S _ { 0 } = \varnothing$ . For $\alpha > \nu$ and $\varepsilon \leq 1$ , the localized evaluation bound reduces, apart from the logarithmic factors in (26) and fixed costs of the initial phases, to min $\{ \varepsilon ^ { - ( 2 + d / \nu ) } , \ K \bar { \varepsilon } ^ { - ( 2 + d / \nu - d / \alpha ) } \}$ . The second exponent is smaller than the first by $d / \alpha ,$ which is because as the target error decreases, the covering condition confines the relevant points to smaller neighborhoods of the maximizers, reducing the evaluations needed for optimization. For fixed $T > 0$ , the term $K T ^ { d / \alpha } \varepsilon ^ { - ( 2 + d / \nu ) }$ instead accounts for refining an entire positive-tolerance region, and the same completed output also provides the estimated optimal point.

Applying Theorem 5.1 with a target accuracy level $\eta / ( 8 \vartheta ) < \varepsilon \leq \eta / 8$ , and choosing the horizon N accordingly, gives the following upper bound on the minimax evaluation count, which matches Theorem 3.1.

Corollary 5.1. Under the fixed model and class of Theorem 3.1, uniformly over its stated $K , T , \eta$ range and $0 < \delta \leq 1 / 1 0$ 2

$$
n _ { \mathrm { a l l } } ( \eta , T , \delta ; \mathcal { F } _ { K } ) \le C \eta ^ { - ( 2 + d / \nu ) } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \left[ \log \frac { 2 } { \eta \delta } \right] ^ { 3 + 3 d / ( 2 \nu ) } .\tag{27}
$$

The constants depend only on the fixed model and class parameters and are independent of $K , T , \eta , \delta$

Theorem 3.1 and Corollary 5.1 show that, for any fixed $0 < \delta \leq 1 / 1 0$ , the minimax evaluation count has matching rates

$$
n _ { \mathrm { a l l } } ( \eta , T , \delta ; \mathcal { F } _ { K } ) = \widetilde \Theta \Bigl ( \eta ^ { - ( 2 + d / \nu ) } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \Bigr ) .
$$

Here the tilde suppresses logarithms in the accuracy, with the confidence factors given explicitly in (7) and (27).

If $T = c _ { T } \eta$ for a fixed $c _ { T } \geq 1 2 8$ , the two accuracy terms in the evaluation bound are

$$
\operatorname* { m i n } \{ \eta ^ { - ( 2 + d / \nu ) } , K \eta ^ { - ( 2 + d / \nu - d / \alpha ) } \} .
$$

These are also the terms governing the evaluations needed to achieve an optimality gap of order η in the $T = 0$ specialization, where the first is the global bound, and the second uses the covering condition to reduce its exponent by $d / \alpha$ . For fixed positive $T \leq T _ { \mathrm { r e g } }$ , the inverse-accuracy exponent is $2 + d / \nu _ { ; }$ , and the geometric multiplier is min $\{ 1 , K T ^ { d / \alpha } \}$ . When $\breve { K } < T ^ { - d / \alpha }$ , the bound is smaller than the global rate by the factor $K T ^ { d / \alpha }$ , and when $K \geq T ^ { - d / \alpha }$ , the factor equals one.

From a decision-making perspective, Theorem 3.1 and Corollary 5.1 quantify the evaluation cost of keeping the post-sampling choice of tolerance described in Section 2.2. For $\alpha > \nu$ and $K T ^ { d / \alpha } < 1$ 2 the minimax budget grows as $K T ^ { d / \alpha } \eta ^ { - ( 2 + d / \nu ) }$ , up to logarithmic factors.

Remark 5.2. Theorem 5.1 concerns the full-budget allocation $m \in \{ 0 , N \}$ . For $T > 0$ and $m \leq N / 2 ,$ Section 5.4 quantifies the tradeof between region accuracy and cumulative regret and identifies quotas that preserve the optimization rates. For $T = 0$ and $m = 0$ , Section 5.3 establishes the pure-optimization regret bounds and their matching minimax ranges.

For $T = 0$ , the full-budget allocation is $m = 0$ , and its regret guarantees are those of Theorem 5.2. The following proposition gives the optimization guarantees of the full-budget allocation $m = N$ for $T > 0 ,$ , in which all evaluations are spent by the region track. This case is covered neither by Theorem 5.2, which requires $T = 0$ , nor by Theorem 5.5, which requires $m \leq \lfloor N / 2 \rfloor$

Proposition 5.2. Under the assumptions of Theorem $5 . 1 ,$ run Algorithm 1 with budget $N \geq 2$ $T > 0 , m = N$ , and $\delta = N ^ { - 2 }$ . Then

$$
\begin{array} { r l } & { \mathbb { E } r _ { N } \le C \log ^ { 3 / 2 } ( 2 N ) N ^ { - 1 / ( 2 + d / \nu ) } , } \\ & { \mathbb { E } R _ { N } \le \operatorname* { m i n } \{ 2 B N , ~ N T + C \log ^ { 3 / 2 } ( 2 N ) N ^ { 1 - 1 / ( 2 + d / \nu ) } \} . } \end{array}\tag{28}
$$

The constant depends only on the fixed model parameters and ϑ, and is uniform in T and N.

Thus, devoting all evaluations to region learning keeps the global rate for simple regret. The designs that remain relevant to the region track always include the maximizers of $f ,$ so each completed region phase also gives $f ^ { \star } - f ( \widehat { x } _ { h } ) \leq 2 \varepsilon _ { h }$ by Proposition A.2. In contrast, the cumulative regret bound contains the linear term $N T$ , because learning the full positive-tolerance region may spend evaluations at designs whose performance gap is of order $T .$ . This term cannot be avoided at the fastest region accuracy. With $m = N$ , Corollary 5.1 shows that SASORI achieves, up to logarithmic factors, the fastest region estimation error (min{1, $K T ^ { d / \alpha } \} / N ) ^ { 1 / ( 2 + d / \nu ) }$ . By (35), as shown in Section $5 . 4 .$ , any valid procedure that achieves this accuracy, up to logarithmic factors, has worst-case cumulative regret of order at least T N, again up to logarithmic factors. For fixed $T > 0$ the bound (28) is at most $2 N T$ for all suficiently large N. Hence, at the fastest region accuracy, the full-budget allocation attains the smallest possible order of cumulative regret, up to logarithmic factors. This complements Corollary 5.2, which covers quotas $m _ { \eta } \leq \lfloor N / 2 \rfloor$ . Section 5.4 shows how a region quota $m < N$ reduces the cumulative regret to the optimization rate $N \rho _ { N }$ at the price of a larger region estimation error.

## 5.3 Optimization Regret Benchmarks

We now establish the pure-optimization benchmark used in the joint-learning analysis. Throughout this subsection, SASORI uses $T = 0$ and $m = 0$ , so only the optimization track is active. The strict zero-tolerance region is already known to be empty. For $N \geq 2$ and covering parameters $K \geq 1$ and $\alpha > 0$ , define

$$
\rho _ { N } = \rho _ { N } ( K , \alpha ) : = \operatorname* { m i n } \left\{ ( K / N ) ^ { 1 / \operatorname* { m a x } \{ 2 , 2 + d / \nu - d / \alpha \} } , N ^ { - 1 / ( 2 + d / \nu ) } \right\} .\tag{29}
$$

The quantities $\rho _ { N }$ and $N \rho _ { N }$ describe the simple regret and cumulative regret rates, respectively. We give upper bounds for both, followed by matching lower bounds when $\alpha > \nu$

Theorem 5.2 (Global and localized regret of batched SASORI ). Under Assumptions 5.2–5.3, the policy in Algorithm 1 with $T = 0$ and $\delta = N ^ { - 2 }$ satisfies

$$
\begin{array} { r } { \mathbb { E } r _ { N } = \widetilde { O } ( N ^ { - \nu / ( 2 \nu + d ) } ) , \qquad \mathbb { E } R _ { N } = \widetilde { O } ( N ^ { ( \nu + d ) / ( 2 \nu + d ) } ) . } \end{array}\tag{30}
$$

Under Assumption $5 . 4 ,$ the same policy satisfies

$$
\begin{array} { r l } & { \mathbb { E } r _ { N } \le C \mathrm { l o g } ^ { 3 + 3 d / ( 2 \nu ) } ( 2 N ) \rho _ { N } + C \zeta _ { N } + 2 B / N ^ { 2 } , } \\ & { \mathbb { E } R _ { N } \le C \mathrm { l o g } ^ { 3 + 3 d / ( 2 \nu ) } ( 2 N ) \left[ 1 + N \rho _ { N } \right] + C N \zeta _ { N } + 2 B / N . } \end{array}\tag{31}
$$

Constants in the localized statement may depend on $\vartheta , c _ { \rho } , u _ { 0 } , \alpha$ as well as the fixed model parameters.

The global bounds in (30) hold for arbitrarily large K and without any localization assumption. Their simple regret and cumulative regret rates recover those of Vakili et al. (2021) and Li and Scarlett (2022), respectively, and match the global lower rates of Scarlett et al. (2017). For $\alpha > \nu ,$ the localized bound improves on the global bound until K reaches order $N ^ { d / ( \alpha ( 2 + \dot { d / \nu } ) ) }$ , up to logarithmic factors. Theorem 5.3 below shows that this transition is minimax sharp on the same fixed class.

Matching lower bounds when $\alpha > \nu .$ For a fixed function class F, define the minimax risks

$$
\mathcal { R } _ { N } ^ { s } ( \mathcal { F } ) = \operatorname* { i n f } _ { \pi , \hat { x } _ { N } } \operatorname* { s u p } _ { f \in \mathcal { F } } \mathbb { E } _ { f } r _ { N } , \qquad \mathcal { R } _ { N } ^ { c } ( \mathcal { F } ) = \operatorname* { i n f } _ { \pi } \operatorname* { s u p } _ { f \in \mathcal { F } } \mathbb { E } _ { f } R _ { N } .
$$

Here $\pi$ ranges over sampling policies using N evaluations, and the first infimum is also taken over the final design selection rule. The supremum is over the fixed class, so the resulting lower bounds apply to procedures that are required to work uniformly on that class. These risks give the smallest worst-case expected regret achievable with N evaluations. We now derive lower bounds on the same classes $\mathcal { F } _ { K }$ of Section 3, which are also the classes appearing in Theorem 5.2. These classes are indexed by their covering number bound K, while the norm upper bound and all other parameters are fixed as K and N vary. We prove matching minimax rates for the case $\alpha > \nu$ as shown in the following theorem, while other cases are discussed in Section 5.6.

Theorem 5.3 (Matching lower bounds for $\alpha > \nu )$ . Fix $\alpha > \nu , c _ { \rho } , u _ { 0 } , B , \sigma > 0$ , the kernel, and the domain. Assume independent Gaussian observation errors with variance $\sigma ^ { 2 }$ . There are a finite integer $K _ { \mathrm { o p t } } \geq 1$ and a finite threshold $N _ { \mathrm { o p t } } \geq 2$ , depending only on these fixed parameters, such that, for any integer $K \geq K _ { \mathrm { o p t } }$ and any $N \geq N _ { \mathrm { o p t } }$

$$
\operatorname* { i n f } _ { \pi , \widehat { x } _ { N } } \operatorname* { s u p } _ { f \in \mathcal { F } _ { K } } \mathbb { E } _ { f } r _ { N } \geq c \rho _ { N } ,\tag{32}
$$

$$
\operatorname* { i n f } _ { \pi } \operatorname* { s u p } _ { f \in { \mathcal { F } } _ { K } } \mathbb { E } _ { f } R _ { N } \geq c N \rho _ { N } .\tag{33}
$$

The constants $c > 0 , K _ { \mathrm { o p t } }$ , and $N _ { \mathrm { o p t } }$ are independent of $K , N$

The lower bounds in Theorem 5.3 hold uniformly for $K \ge K _ { \mathrm { o p t } }$ and $N \geq N _ { \mathrm { o p t } }$ . Combining Theorem 5.3 with Theorem 5.2, and writing $a = 2 + d / \nu - d / \alpha > 2$ , gives

$$
\begin{array} { r } { \mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) = \widetilde \Theta ( \rho _ { N } ) , \qquad \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) = \widetilde \Theta ( N \rho _ { N } ) . } \end{array}\tag{34}
$$

Recall that the covering number bound K is the uniform upper bound in Assumption 5.4 on the number of balls of radius $c _ { \rho } u ^ { 1 / \alpha }$ needed to cover $\mathcal { S } _ { u + }$ for $0 < u \leq u _ { 0 }$ . Increasing K weakens this geometric restriction and enlarges the class. The minimax rate switches between the localized rate $( K / N ) ^ { 1 / a }$ and the global rate $\bar { N } ^ { - 1 / ( 2 + d / \nu ) }$ . Equating these two rates yields

$$
K _ { \star } ( N ) = N ^ { 1 - a / ( 2 + d / \nu ) } = N ^ { d / [ \alpha ( 2 + d / \nu ) ] } = N ^ { \nu d / [ \alpha ( 2 \nu + d ) ] } .
$$

Thus localization improves the minimax rates when K is below $K _ { \star } ( N )$ , while for $K \gtrsim K _ { \star } ( N )$ the simple and cumulative regrets have the global Matérn/Sobolev rates $N ^ { - \nu / ( 2 \nu + d ) }$ and $\dot { N } ^ { ( \nu + d ) / ( \dot { 2 } \nu + d ) }$ respectively, up to logarithmic factors (Scarlett et al. 2017, Vakili et al. 2021, Li and Scarlett 2022). Note that the global upper bound remains valid without any covering assumption.

## 5.4 The Regret Cost of Simultaneous Region Learning

Region learning spends evaluations at points that may already be known to be suboptimal. We first quantify the cumulative regret that any valid region guarantee forces, and then choose a region quota that controls this additional cost. Throughout the matching results below, $\alpha > \nu ,$ $K \ge \operatorname* { m a x } \{ K _ { \mathrm { r e g } } , K _ { \mathrm { o p t } } \} , 0 < T \le T _ { \mathrm { r e g } }$ , and $0 < \eta \leq T / 1 2 8$ , and $N \geq N _ { \mathrm { o p t } }$ , with the fixed class parameters of Theorem 3.1.

At a fixed positive T, suitable region quotas preserve the optimization-only rates for both simple and cumulative regret while allowing consistent region estimation. However, keeping cumulative regret at the optimization rate requires a smaller region quota than devoting all N evaluations to region learning, so region estimation error decreases more slowly. Theorem 5.4 gives a lower bound on the regret required by any prescribed region accuracy. Theorem 5.5 then provides an allocation whose cumulative regret matches this lower bound up to the explicit logarithmic factors in Corollary 5.2.

Theorem 5.4. Under the fixed model and parameter range of Theorem 3.1, any $( \eta , T , \delta )$ -valid procedure using a fixed budget of N observations satisfies

$$
\operatorname* { s u p } _ { f \in { \mathcal F } _ { K } } \mathbb { E } _ { f } R _ { N } \ge c \sigma ^ { 2 } T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - ( 2 + d / \nu ) } \operatorname { k l } ( \delta , 1 - \delta ) .\tag{35}
$$

Here $K \ge K _ { \mathrm { r e g } } , 0 < T \le T _ { \mathrm { r e g } } , 0 < \eta \le T / 1 2 8$ , and $0 < \delta \leq 1 / 1 0$ , with constants uniform in $K , T , \eta , \delta , N$ . The result also holds when the procedure knows the common optimum value and global maximizer of the lower-bound family.

Recall that Algorithm 1 assigns quotas $N - m$ and m to the optimization and region tracks, respectively, and alternates whole phases while sharing completed confidence bands. With $T = 0$ and $m = 0$ , all N evaluations use the optimization track. With $T > 0$ and $m = N$ , all evaluations use the region track analyzed in Theorem 5.1. The next theorem considers $m \leq N / 2$ , which reserves at least half the evaluations for optimization.

Theorem 5.5. Let X be a nonempty compact subset $o f \mathbb { R } ^ { d } ,$ , and suppose Assumptions 5.2–5.3 hold. Let $N \geq 2 , T \geq 0$ , and $0 \leq m \leq \lfloor N / 2 \rfloor$ , with $m = 0 \ i f T = 0$ . For $0 < \delta < 1$ , set $\bar { \delta } = \operatorname* { m i n } \{ \delta , N ^ { - 2 } \}$ and calibrate (9) at horizon N and error probability <sup>¯</sup>δ. Under Assumption 5.4, for any $\alpha > 0$ , the simple and cumulative regret satisfy

$$
\begin{array} { r } { \mathbb { E } _ { f } r _ { N } \le C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) \rho _ { N } + C \zeta _ { N } + 2 B \bar { \delta } , } \end{array}\tag{36}
$$

$$
\mathbb { E } _ { f } R _ { N } \le C N \rho _ { N } \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) + C T m + C N \zeta _ { N } + 2 B N \bar { \delta } .\tag{37}
$$

Without Assumption $5 . 4 \AA$ , both inequalities hold with $\rho _ { N }$ replaced by $N ^ { - 1 / ( 2 + d / \nu ) }$

For $T > 0$ , on $\mathcal { E } _ { \bar { \delta } }$ from (13), if the region track completes a phase with accuracy level $\varepsilon =$ $2 B \vartheta ^ { - H } \geq 2 \zeta _ { N }$ , the output satisfies, simultaneously for $0 \leq \tau \leq T$

$$
\begin{array} { r l } & { f ^ { \star } - f ( \widehat { x } _ { N } ) \leq 2 \varepsilon , \qquad I _ { N , \tau } \subseteq S _ { \tau } \subseteq O _ { N , \tau } , } \\ & { O _ { N , \tau } \setminus I _ { N , \tau } \subseteq \{ x : | f ^ { \star } - f ( x ) - \tau | \leq 4 \varepsilon \} . } \end{array}\tag{38}
$$

For any $T > 0$ , on ${ \mathcal E } _ { \bar { \delta } }$ , the region track completes phase H (together with all preceding phases) within its quota of m observations and saves the output in (38) if

$$
m \geq C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) ( 1 + \varepsilon ^ { - ( 2 + d / \nu ) } ) .\tag{39}
$$

Under Assumption $5 . 4$ with $\alpha > \nu ,$ for $0 < T \le u _ { 0 } / 2$ , the following smaller quota sufices for the same target-phase completion:

$$
m \geq C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) \left[ 1 + \operatorname* { m i n } \left\{ \varepsilon ^ { - ( 2 + d / \nu ) } , K \varepsilon ^ { - 2 } + K ( T + \varepsilon ) ^ { d / \alpha } \varepsilon ^ { - ( 2 + d / \nu ) } \right\} \right] .\tag{40}
$$

The shared confidence event $\mathcal { E } _ { \bar { \delta } }$ in (13) has probability at least $1 - \bar { \delta }$ . Constants depend only on the fixed model parameters, ϑ, and, for the covering-dependent bounds (36), (37), and (40), the fixed covering parameters other than K; they are independent of $K , N , T , m , \delta$

Combining the optimization lower bound (33), the region-induced regret lower bound (35), and Theorem 5.5 yields the following characterization.

Corollary 5.2. Fix $\alpha \ > \ \nu$ and $0 ~ < ~ \delta ~ \leq ~ 1 / 1 0$ . Let $K \ge \operatorname* { m a x } \{ K _ { \mathrm { r e g } } , K _ { \mathrm { o p t } } \}$ and $N ~ \geq ~ N _ { \mathrm { o p t } }$ where $K _ { \mathrm { r e g } } , T _ { \mathrm { r e g } }$ are from Theorem 3.1 and $K _ { \mathrm { o p t } } , N _ { \mathrm { o p t } }$ are from Theorem 5.3. Let $0 < T \leq T _ { \mathrm { r e g } }$ $0 < \eta \leq T / 1 2 8$ , and <sup>¯</sup>δ = min $\{ \delta , N ^ { - 2 } \}$ . Suppose $\eta \geq 1 6 \operatorname* { m a x } \{ 2 , \vartheta \} \zeta _ { N }$ and

$$
m _ { \eta } : = \left\lceil C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - ( 2 + d / \nu ) } \right\rceil \leq \lfloor N / 2 \rfloor ,\tag{41}
$$

with a suficiently large fixed C. Define $\Re _ { N } ( \eta , T , \delta ; \mathcal { F } _ { K } )$ as the infimum of worst-case expected cumulative regret among N-observation policies that are $( \eta , T , \delta )$ -valid on $\mathcal { F } _ { K }$ . Then

$$
\Re _ { N } ( \eta , T , \delta ; \mathcal { F } _ { K } ) \ge c \operatorname* { m a x } \Big \{ N \rho _ { N } , T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - ( 2 + d / \nu ) } \mathrm { k l } ( \delta , 1 - \delta ) \Big \} ,\tag{42}
$$

and

$$
\begin{array} { r l } & { \Re _ { N } ( \eta , T , \delta ; \mathcal { F } _ { K } ) \le C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) } \\ & { \qquad \times \operatorname* { m a x } \Bigl \{ N \rho _ { N } , T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - ( 2 + d / \nu ) } \Bigr \} . } \end{array}\tag{43}
$$

The constants may depend on fixed ϑ and are independent of $K , N , T , \eta , \delta$ . SASORI with $m = m _ { \eta }$ attains the upper bound while also satisfying the simple regret bound (36).

For $T > 0$ , the choice

$$
{ \mathfrak { m } } = \left\lfloor \operatorname* { m i n } \left\{ { \frac { N } { 2 } } , { \frac { N \rho _ { N } } { T } } \right\} \right\rfloor\tag{44}
$$

keeps the extra term $T m$ below the optimization contribution in (37). This tuning uses K and α. A choice independent of these parameters is

$$
m = \left\lfloor \operatorname* { m i n } \{ N / 2 , \sqrt { N } / T \} \right\rfloor .\tag{45}
$$

Indeed, $K \geq 1$ and $2 + d / \nu - d / \alpha > 2$ imply $\sqrt { N } \le N \rho _ { N }$ , so this choice also preserves the regret bounds.

For either positive quota $m .$ , a suficient condition on the target region estimation error is

$$
\begin{array} { r l } { \eta \geq 1 6 \operatorname* { m a x } \{ 2 , \vartheta \} \zeta _ { N } , \eta \leq T / 1 2 8 , \eta } & { \geq C \Big [ \frac { \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) } { m } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \Big ] ^ { 1 / ( 2 + d / \nu ) } . } \end{array}\tag{46}
$$

The upper restriction $\eta ~ \leq ~ T / 1 2 8$ is the accuracy range in which the positive-tolerance lower bound applies and the $K \varepsilon ^ { - 2 }$ term is absorbed. The derivations behind the following rates and

the feasibility conditions stated below are given in Section B.9. To obtain explicit rates from (46), fix $K , T > 0$ , and $\delta > 0$ , and let N increase. Since $\alpha > \nu ,$ the localized rate eventually leads to $\rho _ { N } = ( K / N ) ^ { 1 / ( 2 + d / \nu - d / \alpha ) }$ . Thus the quotas in (44) and (45) have respective orders

$$
m \asymp N ^ { 1 - 1 / ( 2 + d / \nu - d / \alpha ) } \ \mathrm { a n d } \ m \asymp N ^ { 1 / 2 } ,
$$

where constants may depend on the fixed $K , T$ . Also, $\bar { \delta } = N ^ { - 2 }$ for all large N, so the logarithmic product in (46) has order $\log ^ { 3 + 3 d / ( 2 \nu ) } ( 2 N )$ . Substituting the two quota orders into (46) therefore gives the following suficient conditions on the target region estimation error:

$$
\eta \geq C \log ^ { 3 / 2 } ( 2 N ) N ^ { - \{ 1 - 1 / ( 2 + d / \nu - d / \alpha ) \} / ( 2 + d / \nu ) } \mathrm { f o r } ( 4 4 ) ,\tag{47}
$$

$$
\eta \geq C \log ^ { 3 / 2 } ( 2 N ) N ^ { - 1 / [ 2 ( 2 + d / \nu ) ] } \mathrm { f o r } ( 4 5 ) .\tag{48}
$$

Both lower thresholds decrease to zero. For large N they exceed 16 max $\{ 2 , \vartheta \} \zeta _ { N }$ and fall below the fixed $T / 1 2 8 ,$ , so the required accuracy interval is nonempty. Meanwhile, $T m \le N \rho _ { N }$ for both quotas, which preserves the regret rates in (36)–(37).

The accuracy cost of preserving cumulative regret can now be made explicit. If all evaluations can be devoted to region learning, its fastest accuracy has rate

$$
\left( \frac { \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} } { N } \right) ^ { 1 / ( 2 + d / \nu ) } .
$$

Fix $K , T > 0$ , and δ. If the target region estimation error is at most a fixed multiple of this fastest region-accuracy rate, substituting it into (35) gives a cumulative regret lower bound $c T N$ Thus achieving that accuracy requires a positive average performance gap sup $_ f \mathbb { E } _ { f } R _ { N } / N \geq c T$ . In contrast, the optimal cumulative regret is sublinear in $N$ . Indeed, for fixed $K$

$$
N \rho _ { N } \asymp N ^ { 1 - 1 / ( 2 + d / \nu - d / \alpha ) } , 1 - \frac { 1 } { 2 + d / \nu - d / \alpha } < 1 .
$$

Hence, the linear lower bound $c T N$ is incompatible with the optimal cumulative regret rate. Consequently, the fastest region accuracy and the optimal cumulative regret cannot both be achieved. More generally, a cumulative regret bound of order $N \rho _ { N }$ implies, at fixed confidence,

$$
\eta \ge c \operatorname * { m a x } \Bigg \{ \left( \frac { \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} } { N } \right) ^ { 1 / ( 2 + d / \nu ) } , \left( \frac { T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} } { N \rho N } \right) ^ { 1 / ( 2 + d / \nu ) } \Bigg \} .\tag{49}
$$

Simple regret, on the other hand, remains compatible with the fastest region accuracy by choosing $m = \lfloor N / 2 \rfloor$

Remark 5.3 (When the tolerance and accuracy ranges are feasible). For a chosen $m \geq 1$ , the suficient interval (46) is nonempty whenever $T \leq T _ { \mathrm { r e g } }$ and

$$
\begin{array} { r l } & { T \geq 2 0 4 8 \operatorname* { m a x } \{ 2 , \vartheta \} \zeta _ { N } , T \geq C \operatorname* { m i n } \biggr \{ \left[ \frac { K \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) } { m } \right] ^ { 1 / ( 2 + d / \nu - d / \alpha ) } , } \\ & { \qquad \left[ \frac { \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \bar { \delta } ) } { m } \right] ^ { 1 / ( 2 + d / \nu ) } \biggr \} . } \end{array}\tag{50}
$$

For (44), at fixed confidence a suficient range when the localized rate $( K / N ) ^ { 1 / ( 2 + d / \nu - d / \alpha ) }$ in $\rho _ { N }$ is smaller is

$$
T \geq C ( K / N ) ^ { 1 / ( 2 + d / \nu - d / \alpha ) } \log ^ { ( 3 + 3 d / ( 2 \nu ) ) / ( 1 + d / \nu - d / \alpha ) } ( 2 N ) ,
$$

when $K T ^ { d / \alpha } \leq 1$ and $( K / N ) ^ { 1 / ( 2 + d / \nu - d / \alpha ) } \leq N ^ { - 1 / ( 2 + d / \nu ) }$ . When the global rate $N ^ { - 1 / ( 2 + d / \nu ) }$ in $\rho _ { N }$ is smaller, a suficient range is

$$
T \geq C N ^ { - 1 / ( 2 + d / \nu ) } \log ^ { ( 3 + 3 d / ( 2 \nu ) ) / ( 1 + d / \nu ) } ( 2 N ) ,
$$

when $K T ^ { d / \alpha } \geq 1$ and $N ^ { - 1 / ( 2 + d / \nu ) } \leq ( K / N ) ^ { 1 / ( 2 + d / \nu - d / \alpha ) }$ . Both also require $T \geq 2 0 4 8 \operatorname* { m a x } \{ 2 , \vartheta \} \zeta _ { N }$ and $T \leq T _ { \mathrm { r e g } }$ . Thus the tolerance can have the same polynomial order as the optimization error, up to logarithmic factors.

## 5.5 Region Identification at a Prespecified Tolerance

When fixing a single tolerance $\tau > 0$ before sampling, the procedure estimates the unknown maximum and determines membership in the single region $S _ { \tau }$ . We use Algorithm 1 with only the region track active $( T = \tau , m = N )$ , for a chosen budget $N \geq 2$ and error probability $\delta \in ( 0 , 1 )$ . Only the screening function for the phase and the returned region change, and all other steps, including the check for inconsistent lower and upper bounds, follow Algorithm 1.

We keep the phase accuracy levels $\varepsilon _ { h } = 2 B \vartheta ^ { - h }$ with fixed $\vartheta > 1$ and the notation for the confidence bounds from Algorithm 1 and (14). Immediately before phase $h ,$ abbreviate the current bands and extrema by $\ell = \ell _ { h - 1 } , u = u _ { h - 1 } , \underline { { f } } = \underline { { f } } _ { h - 1 } ,$ and $\overline { { f } } = \overline { { f } } _ { h - 1 }$ , and replace the region screening function by

$$
\begin{array} { l } { { g _ { h - 1 } ( x ) = \operatorname* { m a x } \{ u ( x ) - \underline { { f } } , \operatorname* { m i n } \{ \overline { { f } } - \tau - \ell ( x ) , u ( x ) - \underline { { f } } + \tau \} \} , } } \\ { { G _ { h } ( x ) = [ g _ { h - 1 } ( x ) + 4 \varepsilon _ { h } ] _ { + } . } } \end{array}\tag{51}
$$

The first term in the maximum in $( 5 1 ) , u ( x ) - f ($ , keeps possible maximizers, and the second term keeps points whose membership at the chosen tolerance is undetermined. After a completed phase, the procedure returns

$$
I _ { h , \tau } = \{ x : \ell _ { h } ( x ) > \overline { { f } } _ { h } - \tau \} , O _ { h , \tau } = \{ x : u _ { h } ( x ) > \underline { { f } } _ { h } - \tau \} .\tag{52}
$$

The following theorem gives accuracy and a suficient budget for this single-tolerance output.

Theorem 5.6. Let X be any nonempty compact subset of $\mathbb { R } ^ { d }$ , and assume that Assumptions 5.2–5.3 hold. With probability at least $1 - \delta$ , simultaneously over all completed phases,

$$
\underline { { f } } _ { h } \leq f ^ { \star } \leq \overline { { f } } _ { h } , \overline { { f } } _ { h } - \underline { { f } } _ { h } \leq 2 \varepsilon _ { h } , f ^ { \star } - f ( \widehat x _ { h } ) \leq 2 \varepsilon _ { h } , \widehat x _ { h } \in \underset { x \in \mathcal { X } } { \arg \operatorname* { m a x } } \ell _ { h } ( x ) ,\tag{53}
$$

and

$$
I _ { h , \tau } \subseteq S _ { \tau } \subseteq O _ { h , \tau } , O _ { h , \tau } \setminus I _ { h , \tau } \subseteq \{ x : | f ^ { \star } - f ( x ) - \tau | \leq 4 \varepsilon _ { h } \} .\tag{54}
$$

For any target phase H with $\varepsilon _ { H } \geq 2 \zeta _ { N }$ , a suficient evaluation budget for completion on the same event is

$$
C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \delta ) \big ( 1 + \varepsilon _ { H } ^ { - ( 2 + d / \nu ) } \big ) ,\tag{55}
$$

provided this expression is at most N. The constant C depends only on the fixed model parameters and $\vartheta ,$ , and is independent of $N , \delta , \tau , H$

Theorem 5.6 provides both an estimated optimal point and an inner/outer approximation to the prescribed region. Choosing an accuracy level $\varepsilon _ { H }$ of order η with 4ε $\ ` { H } \leq \eta$ gives an optimality gap at most $2 \varepsilon _ { H }$ and leaves membership undetermined only where $| f ^ { \star } - f ( x ) - \tau | \leq \eta$ , provided that $\varepsilon _ { H } \geq 2 \zeta _ { N }$ and the suficient evaluation bound in (55) is at most N. The general suficient budget is $\widetilde { \cal O } ( \eta ^ { - 2 - d / \nu } )$ and requires neither Assumption 5.4 nor regularity of the tolerance boundary. Its exponent of $\eta ^ { - 1 }$ is the same as in the global simultaneous-region bound in Theorem 5.1. Thus fixing one tolerance alone does not by itself improve this general upper bound, but can improve the rates under boundary geometry condition, as quantified below.

The first corollary extracts the cumulative regret cost of the testing construction used for Theorem 3.2.

Corollary 5.3. Under the model assumptions and parameter range of Theorem 3.2, consider any procedure using a fixed budget of N observations that, for any $f \in { \mathcal { F } } _ { K }$ , returns a region guarantee satisfying (1) at the prescribed tolerance τ, with region estimation error at most η and probability at least $1 - \delta$ . Then

$$
\operatorname* { s u p } _ { f \in { \mathcal F } _ { K } } \mathbb { E } _ { f } R _ { N } \ge c \sigma ^ { 2 } \tau \mathrm { k l } ( \delta , 1 - \delta ) \eta ^ { - 2 - ( d - 1 ) / \nu } \big ( \operatorname* { m i n } \{ 1 , K \tau ^ { d / \alpha } \} \big ) ^ { ( d - 1 ) / d } .\tag{56}
$$

The constant is independent of $K , N , \tau , \eta , \delta$ . The bound continues to hold when the common optimizer and optimum value of the lower-bound family are provided before sampling.

The next corollary gives simple regret and cumulative regret upper bounds when the procedure refines the region through a prescribed target phase and then uses its saved estimated optimal point for the remaining evaluations. The bound separates the cost of learning the region from the regret accumulated after the target has been reached.

Corollary 5.4. Under the assumptions of Theorem 5.6, fix $\tau > 0 , N \geq 2$ , and $\delta \in ( 0 , 1 )$ . Choose a target phase H with $\varepsilon = \varepsilon _ { H } \ge 2 \zeta _ { N }$ , and suppose the suficient evaluation budget in (55) is at most N. Run the fixed-tolerance specialization through phase H. After its completion, use $\widehat { x } _ { H } \in \arg \operatorname* { m a x } _ { x \in \mathcal { X } } \ell _ { H } ( x )$ for the remaining evaluations and as the final estimated optimal point $\widehat { x } _ { N }$ Then

$$
\begin{array} { r } { \mathbb { E } _ { f } r _ { N } \le 2 \varepsilon + 2 B \delta , } \end{array}\tag{57}
$$

and

$$
\begin{array} { r } { \mathbb { E } _ { f } R _ { N } \leq C \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \delta ) \big [ 1 + ( \tau + \varepsilon ) \varepsilon ^ { - ( 2 + d / \nu ) } \big ] + 2 N \varepsilon + 2 B N \delta . } \end{array}\tag{58}
$$

The constant C depends only on the fixed model parameters and $\vartheta .$

To compare the two corollaries, choose $\varepsilon _ { H }$ of order η with 4ε $\ ` { \boldsymbol { H } } \leq { \boldsymbol { \eta } }$ . For fixed $K , \tau , \delta$ in the parameter range of Theorem 3.2, the necessary cumulative regret has order at least $\tau \eta ^ { - 2 - ( d - 1 ) / \nu }$ whereas the region-learning term in the general upper bound has order $\widetilde { \cal O } ( \tau \eta ^ { - 2 - d / \nu } )$ . Their exponents of $\eta ^ { - 1 }$ difer by $1 / \nu .$ . The lower bound depends on K and τ through the geometric factor $( \operatorname* { m i n } \{ 1 , \dot { K } \tau ^ { d / \alpha } \} ) ^ { ( d - 1 \bar { ) } / d }$ , whereas this factor is absent from the general upper bound. In addition, the upper bound contains $O ( N \eta + B N \delta )$ , which accounts for the evaluations at the saved estimated optimal point and for the event E<sup>c</sup>, whose probability is at most δ. On the other hand, the simple regret upper bound is ${ \cal O } ( \eta + B \delta )$ , so the simple regret bound decreases as the target accuracy and error probability decrease.

The budget in Theorem 5.6 is a general suficient bound. A smaller budget sufices when the remaining maximizer candidates and the uncertain tolerance boundary have small covering numbers.

To quantify this geometry, fix $c _ { r } > 0$ and define

$$
J _ { \tau , f } ( u ) = \mathcal { N } \Big ( S _ { ( 8 u ) + } \cup \{ x : | f ^ { \star } - f ( x ) - \tau | \leq 8 u \} , c _ { r } u ^ { 1 / \nu } \Big ) .\tag{59}
$$

The first set accounts for estimating $f ^ { \star }$ , and the second accounts for classifying the chosen boundary. The following corollary describes the gain available for a single boundary when $J _ { \tau , f }$ grows slowly as u goes to zero.

Corollary 5.5. Fix an error probability $\delta \in ( 0 , 1 )$ and a tolerance $\tau > 0$ . Suppose, in addition to the assumptions of Theorem ${ 5 . 6 } ,$ that $J _ { \tau , f } ( u ) ~ \le ~ C _ { J } u ^ { - \kappa _ { \mathrm { c o v } } }$ for all suficiently small $u > 0 ,$ with fixed $C _ { J } < \infty$ and covering exponent $\kappa _ { \mathrm { c o v } } \geq 0$ . Then the fixed-tolerance specialization of Algorithm 1 identifies the region $S _ { \tau }$ with region estimation error at most η using a budget of $\widetilde { \cal O } ( \eta ^ { - 2 - \kappa _ { \mathrm { c o v } } } )$ evaluations.

Example 5.1. In this example, we show how a regular tolerance boundary reduces the spatial covering exponent from d to $d - 1$ , yielding a smaller suficient budget for one prespecified region. Let $d \geq 2 , \nu > 1 , \chi = \overline { { B } } ( c , R )$ , and $f ( x ) = - \mu \| x - c \| ^ { 2 }$ for fixed $\mu , R > 0$ . Fix $0 < \tau < \mu R ^ { 2 }$ . A smooth compactly supported extension of this quadratic belongs to the Matérn RKHS for any fixed finite $\nu .$ Thus the example satisfies the RKHS assumption whenever the fixed B is at least the norm of its restriction to $\mathcal { X } .$ . The boundary is the sphere of radius $\sqrt { \tau / \mu } ,$ where $\| \nabla f \| = 2 { \sqrt { \mu \tau } } > 0$ . This nonvanishing gradient is the regularity that converts value uncertainty into spatial uncertainty of order u around the boundary. For suficiently small $u ,$ all points x of the set $\{ x : | f ^ { \star } - f ( x ) - \tau | \le 8 u \}$ satisfy $| | x - c | | - \sqrt { \tau / \mu } | \leq C _ { 0 } u$ with $C _ { 0 } = 8 / \sqrt { \mu \tau }$ . Letting $r = c _ { r } u ^ { 1 / \nu }$ , a maximal r-separated subset of this set yields disjoint balls of radius $r / 2$ contained in $\{ x : | \| x - c \| - \sqrt { \tau / \mu } | \leq C _ { 0 } u + r / 2 \}$ , which has volume $O ( u + r )$ . Hence the number of balls of radius r needed to cover $\{ x : | f ^ { \star } - f ( x ) - \tau | \le 8 u \}$ is

$$
O \Big ( r ^ { - ( d - 1 ) } ( 1 + u / r ) \Big ) = O \Big ( u ^ { - ( d - 1 ) / \nu } \Big ) ,
$$

where the equality uses $\nu > 1$ . Meanwhile $\boldsymbol { S } _ { ( 8 u ) + }$ is a ball of radius $O ( \sqrt { u } )$ and has covering number at most

$$
C \left( 1 + u ^ { - \operatorname* { m a x } \{ d / \nu - d / 2 , 0 \} } \right) \leq C ^ { \prime } u ^ { - ( d - 1 ) / \nu } .
$$

The last inequality follows from $1 / \nu \leq d / 2$ . Corollary 5.5 therefore gives the fixed-tolerance suficient budget $\tilde { O } ( \eta ^ { - 2 - ( d - 1 ) / \nu } )$ . This has a strictly smaller exponent than the whole-region bound $\widetilde { \cal O } ( \eta ^ { - 2 - d / \nu } )$ since for the same quadratic and fixed $T = \tau > 0 , S _ { T + }$ contains a ball of fixed positive radius, whose radius-r covering number is of order $r ^ { - d }$ . At fixed confidence and budget $N$ , the bound on region estimation error thus improves from $\widetilde { O } ( N ^ { - \nu / ( 2 \nu + d ) } )$ , the general upper bound of Theorem 5.6, to $\widetilde { \cal O } ( N ^ { - \nu / ( 2 \nu + d - 1 ) } )$ on this example.

## 5.6 Regret Bounds for $2 \leq \alpha \leq \nu$

In this subsection, we provide the optimization-regret results for $2 \leq \alpha \leq \nu .$ . All model and covering parameters other than K and N remain fixed.

Critical endpoint $\alpha = \nu$ . At the critical endpoint $\alpha = \nu ,$ the localized upper exponent in (29) is $a = \operatorname* { m a x } \{ 2 , 2 + d / \nu - d / \alpha \} = 2$ , so that $\rho _ { N } = \operatorname* { m i n } \{ \sqrt { K / N } , N ^ { - 1 / ( 2 + d / \nu ) } \}$ . The following result establishes matching rates at this endpoint.

Theorem 5.7. Fix $\alpha = \nu \geq 2$ , the normalized Matérn kernel, positive $B , c _ { \rho } , u _ { 0 } , \sigma _ { . }$ , and a compact domain containing an interior ball. There are fixed constants $K _ { \mathrm { c r i t } } , N _ { \mathrm { c r i t } } , C _ { \mathrm { c r i t } } , c > 0$ such that, for

all integers $K \geq K _ { \mathrm { c r i t } }$ and $N \geq N _ { \mathrm { c r i t } }$ 2

$$
\begin{array} { r l } & { \mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) \geq c [ \log ( 2 N ) ] ^ { - d / ( 2 \nu ) } \operatorname* { m i n } \{ \sqrt { K / N } , N ^ { - 1 / ( 2 + d / \nu ) } \} , } \\ & { \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) \geq c [ \log ( 2 N ) ] ^ { - d / ( 2 \nu ) } \operatorname* { m i n } \{ \sqrt { K N } , N ^ { 1 - 1 / ( 2 + d / \nu ) } \} . } \end{array}
$$

The logarithmic factor can be omitted if $K \ge C _ { \mathrm { c r i t } } [ \log ( 2 N ) ] ^ { d / \nu }$ . If ν is an even integer, it can be omitted for any $K \geq K _ { \mathrm { c r i t } }$ . All constants are independent of K, N.

Together with Theorem 5.2, this proves

$$
\mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) = \widetilde { \Theta } \Big ( \operatorname* { m i n } \{ \sqrt { K / N } , N ^ { - 1 / ( 2 + d / \nu ) } \} \Big ) , \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) = \widetilde { \Theta } \Big ( \operatorname* { m i n } \{ \sqrt { K N } , N ^ { 1 - 1 / ( 2 + d / \nu ) } \} \Big ) .
$$

The localized and global bounds are equal at $K \asymp N ^ { d / ( 2 \nu + d ) }$ . The lower bounds also apply to joint procedures on this class, whereas the matching SASORI upper bounds use $T = 0$ and $m = 0$

Lower bounds for $2 \leq \alpha \leq \nu$ . Fix $d \geq 1 , 2 \leq \alpha \leq \nu .$ , a normalized Matérn kernel, and positive $B , c _ { \rho } , u _ { 0 } , \sigma$ . Let X be a fixed compact domain with nonempty interior, and let $\mathcal { F } _ { K }$ be the covering class of Section 3. The following proposition also covers $\alpha = \nu _ { \mathrm { { ; } } }$ , while Theorem 5.7 gives the sharper lower bound.

Proposition 5.3. There are constants $K _ { \mathrm { p e r } } , N _ { \mathrm { p e r } } , c > 0$ , independent of $K , N$ , such that for all integers $K \geq K _ { \mathrm { p e r } }$ and $N \geq N _ { \mathrm { p e r } }$ 2

$$
\begin{array} { r l } & { \mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) \geq c \operatorname* { m i n } \{ K ^ { \alpha / ( 2 \nu + d ) } N ^ { - 1 / 2 } , N ^ { - \nu / ( 2 \nu + d ) } \} , } \\ & { \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) \geq c \operatorname* { m i n } \{ K ^ { \alpha / ( 2 \nu + d ) } N ^ { 1 / 2 } , N ^ { ( \nu + d ) / ( 2 \nu + d ) } \} . } \end{array}
$$

When $N \geq C K ^ { 2 \alpha / d }$ , the first term in each minimum is the smaller one. Proposition 5.3 therefore gives simple regret and cumulative regret lower bounds of order $K ^ { \alpha / ( 2 \nu + d ) } N ^ { - 1 / 2 }$ and $K ^ { \alpha / ( 2 \nu + d ) } N ^ { 1 / 2 }$ respectively. The two terms in each minimum coincide at $K \asymp N ^ { d / ( 2 \alpha ) }$ , and above this scale Proposition 5.3 yields the global Matérn lower rates. However, the next proposition shows that the global lower rates already hold at the smaller scale $K \geq C N ^ { \nu d / [ \alpha ( 2 \nu + d ) ] }$

Proposition 5.4. Fix $2 \leq \alpha \leq \nu$ and all model and covering parameters as above. There are fixed $C , c > 0$ and $N _ { \mathrm { s a t } }$ such that, whenever $N \geq N _ { \mathrm { s a t } }$ and

$$
K \geq C N ^ { d / [ \alpha ( 2 + d / \nu ) ] } = C N ^ { \nu d / [ \alpha ( 2 \nu + d ) ] } ,\tag{60}
$$

one has

$$
\mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) \geq c N ^ { - 1 / ( 2 + d / \nu ) } , \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) \geq c N ^ { 1 - 1 / ( 2 + d / \nu ) } .
$$

Combining Proposition 5.4 with the global upper bounds in Theorem 5.2 gives

$$
\begin{array} { r } { \mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) = \widetilde { \Theta } \Big ( N ^ { - \nu / ( 2 \nu + d ) } \Big ) , \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) = \widetilde { \Theta } \Big ( N ^ { ( \nu + d ) / ( 2 \nu + d ) } \Big ) } \end{array}
$$

whenever (60) holds. Thus a suficiently large covering number bound makes the Matérn regret rates without the covering condition unavoidable.

Recall that Theorem 5.2 gives the upper bounds

$$
\begin{array} { r l } & { \mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) = \widetilde O \left( \operatorname* { m i n } \{ \sqrt { K / N } , N ^ { - \nu / ( 2 \nu + d ) } \} \right) , } \\ & { \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) = \widetilde O \left( \operatorname* { m i n } \{ \sqrt { K N } , N ^ { ( \nu + d ) / ( 2 \nu + d ) } \} \right) . } \end{array}
$$

For $2 \leq \alpha < \nu ,$ their localized and global terms coincide at $K \asymp N ^ { d / ( 2 \nu + d ) }$ . In contrast, Proposition 5.4 establishes matching global lower bounds only once $K > C N ^ { \nu d / [ \alpha ( 2 \nu + d ) ] }$ . Because $\nu / \alpha > 1$ this suficient lower-bound threshold has a larger exponent of N.

For $2 \leq \alpha < \nu$ below the upper-bound transition, the localized upper and lower bounds have the same orders $N ^ { - 1 / 2 }$ for simple regret and $N ^ { 1 / 2 }$ for cumulative regret. Their dependence on K difers: the upper prefactor is $K ^ { 1 / 2 }$ , whereas Proposition 5.3 gives $K ^ { \alpha / ( 2 \nu + d ) }$ . Their ratio is $K ^ { 1 / 2 - \alpha / ( 2 \nu + d ) }$ apart from logarithmic factors. For any fixed integer $K \geq K _ { \mathrm { p e r } }$ , both localized bounds apply once $N \geq \operatorname* { m a x } \{ 2 , N _ { \mathrm { p e r } } , C K ^ { ( 2 \nu + d ) / d } \}$ , where $K _ { \mathrm { p e r } } , N _ { \mathrm { p e r } }$ are the thresholds in Proposition 5.3 and C is a suficiently large fixed constant. The ratio is then constant in N, so the rates match in their orders in N. At $\alpha = \nu$ , Theorem 5.7 gives matching rates for $K \geq K _ { \mathrm { c r i t } }$ and $N \geq N _ { \mathrm { c r i t } }$ , with the distinct thresholds defined in Theorem 5.7.

## 6 Numerical Experiments

In this section, we first describe the experimental design and evaluation metrics, then present numerical results for Branin-2 and Hartmann-3.

## 6.1 Experimental Design

We test three variants of Algorithm 1: SASORI-Opt uses $m = 0$ , SASORI-Reg uses $m = N$ , and SASORI-Joint-Free uses $m = \lfloor \operatorname* { m i n } \{ N / 2 , \sqrt { N } / T \} \rfloor$ . Two additional baselines, GP-UCB and Sobol-GP, provide comparisons based on optimization and uniform exploration, respectively. Each method uses $N = 3 5 0 0$ evaluations with 30 macro-replications. We consider Branin-2 and Hartmann-3, two standard benchmarks for simulation optimization. We set $T = 0 . 1$ for Branin-2 and $T = 0 . 3$ for Hartmann-3, with SASORI multiplier $\beta = 2 . 5$ , phase ratio $\vartheta = 3$ , and GP-UCB multiplier $\kappa = 2 . 5$ Function definitions, kernel parameters, and other implementation details appear in Sections D.1–D.2.

Following Remark 4.1, we query a design with acquisition value above $2 \varepsilon _ { h }$ and accept phase completion only after an additional numerical search finds no such design. Section D.2 gives the implementation details.

Following Algorithm 1, we update the retained bands using the GP confidence bands from completed phases. Let $L , U$ denote the resulting lower and upper functions, and let $b _ { \mathrm { n u m } } , c _ { \mathrm { n u m } }$ denote their maxima. We then compute $I _ { \tau } = \{ x : L ( x ) > c _ { \mathrm { n u m } } - \tau \}$ and $O _ { \tau } = \{ x : U ( x ) > b _ { \mathrm { n u m } } - \tau \}$ as in (15). For each benchmark, we use an independent set Q of 32768 scrambled Sobol points for evaluation, shared by all methods and macro-replications. The main text reports the original algorithm outputs, while the E-Companion additionally presents region estimates obtained by fitting a single GP to all observations, including those from incomplete phases.

We report three metrics for region identification. First, the empirical region error is $\eta _ { Q } ( \tau ) =$ $\mathrm { m a x } _ { x \in ( O _ { \tau } \backslash I _ { \tau } ) \cap Q } | f ^ { \star } - f ( x ) - \tau |$ , with value zero when the maximization set is empty. It measures the largest absolute diference between an unclassified evaluation point’s performance and the acceptance threshold $f ^ { \star } - \tau$ . Second, inner recovery is $\operatorname { R e c } _ { Q } ( \tau ) = | I _ { \tau } \cap S _ { \tau } \cap Q | / | S _ { \tau } \cap Q |$ , reported as a percentage. It measures the fraction of truly acceptable evaluation points included in the inner set. Third, the unclassified percentage is $1 0 0 | ( O _ { \tau } \setminus I _ { \tau } ) \cap Q | / | Q |$ , measuring the fraction of the evaluation set whose membership remains unresolved. These three metrics describe how closely the unresolved designs lie to the acceptance threshold, how much of the acceptable region is recovered, and how much of the domain remains unclassified, respectively.

We also assess optimization performance using simple regret $r _ { N }$ and cumulative regret $R _ { N }$ from Section 2. Further reporting details appear in Section D.3.

## 6.2 Branin-2: Region Identification and Optimization

We first show three metrics of region identification at $\tau = T = 0 . 1$

Table 1: Branin-2 algorithm-output region estimates at $\tau = T = 0 . 1$ (mean ± standard error).
<table><tr><td>Method</td><td> $\eta _ { Q } ( \tau )$ </td><td>Recovery (%)</td><td>Unclassified (%)</td></tr><tr><td>SASORI-Opt</td><td> $0 . 0 6 2 1 \pm 0 . 0 0 1 1$ </td><td> $5 9 . 9 7 \pm 0 . 6 4$ </td><td> $2 3 . 4 9 \pm 0 . 1 7$ </td></tr><tr><td>SASORI-Joint-Free</td><td> $0 . 0 6 7 1 \pm 0 . 0 0 1 6$ </td><td> ${ \bf 6 0 . 8 9 \pm 0 . 5 0 }$ </td><td> $2 3 . 4 8 \pm 0 . 1 4$ </td></tr><tr><td>SASORI-Reg</td><td> $\mathbf { 0 . 0 5 1 8 \pm 0 . 0 0 0 7 }$ </td><td> $5 7 . 2 5 \pm 0 . 5 5$ </td><td> ${ \bf 2 0 . 5 8 \pm 0 . 1 7 }$ </td></tr><tr><td>GP-UCB</td><td> $0 . 7 0 2 4 \pm 0 . 0 7 1 6$ </td><td> $2 0 . 1 2 \pm 3 . 9 4$ </td><td> $6 2 . 0 1 \pm 2 . 5 1$ </td></tr><tr><td>Sobol-GP</td><td> $0 . 0 6 6 1 \pm 0 . 0 0 1 0$ </td><td> $5 3 . 2 0 \pm 0 . 7 4$ </td><td> $2 2 . 9 1 \pm 0 . 2 7$ </td></tr></table>

Table 1 shows that all three SASORI variants achieve higher inner recovery than either baseline. Reg attains the smallest empirical region error and unclassified fraction, indicating that fewer designs remain unresolved and that their performance values lie closer to the acceptance threshold. This is consistent with its objective of improving region estimation accuracy. A smaller region error does not necessarily imply higher inner recovery or a smaller unclassified fraction, as these metrics capture diferent aspects of region identification. Joint-Free achieves the highest mean inner recovery while allocating only a limited portion of its evaluation budget to region learning. This illustrates how its two tracks can identify acceptable alternatives while reserving most evaluations for optimization, consistent with its theoretical objective of preserving the optimization regret rates. Opt also recovers a substantial portion of the acceptable region without dedicated region evaluations, since reducing uncertainty around promising designs improves the confidence bounds used to construct the region estimates.

To illustrate the diferent allocations, we first visualize the queried designs from one macroreplication.

![](images/c7da3864c51689ecd4a99edc1fd93caa9bd2f74b9bd381b6ca9be60a5744f8ed.jpg)

![](images/88c10941560f5b95141941797be06c453bfadb9e2a030136626f5bee2a7fc97a.jpg)

![](images/bfb2161559fdd4228bc6c53a4417d0a5bf34378db4248a6da8155c4d0056e29e.jpg)  
Figure 1: Branin-2 evaluations for the three SASORI variants from one replication. Point colors indicate evaluation order. Stars and diamonds denote the true maxima and original recommendations, respectively.

Figure 1 illustrates the diferent sampling behaviors of the three SASORI variants. Opt concentrates many evaluations around the three maxima, but also explores uncertain designs that remain competitive at the current phase accuracy. Joint-Free exhibits an intermediate pattern, combining evaluations near the maxima with additional exploration for region learning. Reg samples more broadly across the acceptable region and its surroundings, consistent with its emphasis on region identification.

We next visualize the corresponding inner and outer region estimates at $\tau = 0 . 1$

Original recommendation  
![](images/3b71ce1168d6e97fda8d722db25eb45c54bdfa30f3a0e69d3bc1be9af3263022.jpg)

![](images/e7f591bee12f56819bd898c9be1733480c2d7abc49195533ea6fc4148f5c251f.jpg)

![](images/4b2cac9b54573a4def7e22d9b9e7dcb8a0b2ea9a4ac1c46d4e81a0811cf7dcc2.jpg)  
Figure 2: Branin-2 algorithm regions at $\tau = 0 . 1$ from the same replication.

Figure 2 shows the resulting diferences in region identification. All three variants recover substantia portions of the acceptable regions around the maxima. In particular, Reg produces a narrower and smoother unclassified region around the true boundary, suggesting that its broader exploration helps resolve uncertainty across the region. This observation agrees with its smaller empirical region error and unclassified fraction in this example. Results at other tolerances appear in Section D.4.1.

We next examine optimization performance through the regret curves.

(a) Recommended point

![](images/3ca8db2392dd9ead98e5bbd322b5512e96bd5a2bda1ff569976baccaa10476af.jpg)

![](images/516139dfdd9deb209c237d8d810fba2e8b9f8f4c767a73ec86af10b6f1b80811.jpg)  
(c) Cumulative regret

![](images/260a649ca8966c72265bde33fac779a987ca010074cd3c73373dcb0e49d327e6.jpg)  
Figure 3: Branin-2 simple and cumulative regret. The middle panel uses the best evaluated point; shading indicates one standard error. Recommendation curves show SASORI.

Figure 3 illustrates how the allocation of evaluations afects optimization performance. Among the SASORI variants, Opt achieves the lowest cumulative regret and slightly lower best query regret, while Joint-Free exhibits intermediate cumulative regret. This is consistent with their respective allocations, as Opt concentrates its evaluations on optimization and Joint-Free reserves a smaller portion for region learning. Interestingly, Reg attains the lowest mean regret at the fina recommendation, despite devoting all its evaluations to region learning. One possible explanation is that its broader exploration may also help refine the confidence bounds near the maximizers, thereby improving the final recommendation. Indeed, both tracks use the same lower confidence bound rule to select the recommended point, so region learning can also contribute to recommendation quality. This is reflected in the diference between the best query and recommendation results, where Opt visits slightly better designs but Reg produces a more accurate final recommendation. As we will see in Section 6.3, this ordering is diferent for Hartmann-3, where Opt achieves the lowest recommendation regret among the SASORI variants. Finally, GP-UCB achieves the lowest best query and cumulative regret overall, but its region estimates are considerably less accurate, as shown in Table 1. These results illustrate the distinction between optimization eficiency, recommendation quality, and region identification accuracy.

## 6.3 Hartmann-3

We next present the region identification and optimization results for Hartmann-3.

Table 2: Hartmann-3 algorithm-output region estimates at $\tau = T = 0 . 3$ (mean ± standard error).
<table><tr><td>Method</td><td> $\eta _ { Q } ( \tau )$ </td><td>Recovery (%)</td><td>Unclassified (%)</td></tr><tr><td>SASORI-Opt</td><td> $0 . 1 6 7 2 \pm 0 . 0 0 2 3$ </td><td> $5 5 . 5 9 \pm 0 . 4 4$ </td><td> $9 . 1 1 \pm 0 . 0 5$ </td></tr><tr><td>SASORI-Joint-Free</td><td> $0 . 1 7 4 7 \pm 0 . 0 0 2 8$ </td><td> ${ \bf 5 5 . 6 1 \pm 0 . 5 4 }$ </td><td> $9 . 1 9 \pm 0 . 0 6$ </td></tr><tr><td>SASORI-Reg</td><td> $\mathbf { 0 . 1 2 0 6 \pm 0 . 0 0 2 0 }$ </td><td> $5 3 . 6 6 \pm 0 . 6 5$ </td><td> ${ \bf 6 . 7 9 \pm 0 . 0 6 }$ </td></tr><tr><td>GP-UCB</td><td> $0 . 6 6 5 6 6 7 \pm 0 . 0 0 0 0 0 4$ </td><td> $1 2 . 5 3 \pm 1 . 3 6$ </td><td> $8 1 . 5 3 \pm 0 . 5 7$ </td></tr><tr><td>Sobol-GP</td><td> $0 . 2 0 3 6 \pm 0 . 0 0 4 4$ </td><td> $3 7 . 7 5 \pm 1 . 1 4$ </td><td> $9 . 6 0 \pm 0 . 1 2$ </td></tr></table>

Table 2 shows that the regional advantages of SASORI extend to three dimensions, with all three variants achieving higher inner recovery and smaller empirical region error than either baseline. As in Branin-2, Reg attains the smallest empirical region error and unclassified fraction, consistent with its emphasis on region estimation accuracy. Opt and Joint-Free achieve similar mean inner recovery, while Joint-Free reserves only 197 evaluations for region learning, leaving most of its budget for optimization. Results at other tolerances appear in Section D.5.1.  
We finally compare the regret curves for Hartmann-3.

![](images/f762c6a4ada736c3ae5ba715bc45489921621287bb9ee0dc86975960898b74b2.jpg)

![](images/312aae6907d35616d2010dd87d70ab9d6af0ba928702056ba755d09f59731ac4.jpg)

![](images/f21f225f6773fadef10c3e519ba0acee61cfa6ac9155cbc2ff4d1613c7face38.jpg)  
Figure 4: Hartmann-3 simple and cumulative regret. The middle panel uses the best evaluated point; shading indicates one standard error. Recommendation curves show SASORI.

Figure 4 shows that Opt achieves the lowest recommendation, best query, and cumulative regret among the SASORI variants. Unlike Branin-2, where Reg produces the most accurate final recommendation, Opt also achieves the lowest recommendation regret on Hartmann-3. This illustrates that the relative recommendation quality can vary across benchmarks, while Opt consistently incurs the lowest cumulative regret among the three variants. Joint-Free maintains recommendation quality close to Opt and incurs substantially less cumulative regret than Reg, while allocating part of its budget to region learning. Reg incurs greater cumulative regret, accompanied by the smallest empirical region error and unclassified fraction reported in Table 2. GP-UCB again achieves the lowest best query and cumulative regret overall, while leaving considerably greater uncertainty in its region estimates. Together with the Branin-2 results, these observations illustrate how diferent evaluation allocations afect optimization performance and region identification accuracy.

## 7 Conclusions and Discussion

This paper studies the identification of near-optimal regions in continuous simulation optimization, where a decision maker needs the set of designs within a tolerance of the optimum and want to choose the tolerance after seeing the results. We show that region identification is not a by-product of optimization. Specifically, when $\alpha > \nu .$ , the minimax number of simulation evaluations needed to identify the acceptable regions for all tolerances in [0, T] with region estimation error at most η is $\widetilde { \Theta } ( \eta ^ { - ( 2 + d / \nu ) } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} )$ , and the lower bound holds even when the optimal design and its value are known in advance. SASORI achieves this rate without knowing the covering parameters, delivers the estimated optimal point and the regions from the same evaluations, and achieves the optimal tradeof between region estimation error and cumulative regret. Minimax simple regret is compatible with the fastest region identification rate, whereas minimax cumulative regret is not. At zero tolerance, SASORI is a simulation optimization procedure with minimax simple and cumulative regret on the covering class. For practice, only the largest tolerance T must be fixed before sampling, and the tolerance used in the decision can be chosen afterwards with the same probability guarantee.

Several directions remain open. The current guarantees assume that the smoothness class, its norm bound, and the noise scale are known. Future work could develop methods for calibrating these quantities from pilot evaluations and quantifying the resulting efect on coverage. A variance-adaptive version that estimates the simulation variance from replications at each design point would address the heteroscedasticity common in queueing and inventory models, and exploiting common random numbers could further reduce the evaluation cost. On the theoretical side, matching bounds for a single prespecified tolerance under general boundary geometry, and the sharp dependence on the covering number bound when $2 \leq \alpha < \nu ,$ remain to be established.

## References

A. al Marjani, T. Kocak, and A. Garivier. On the complexity of all ε-best arms identification. In Machine Learning and Knowledge Discovery in Databases, pages 317–332, 2023.

S. Andradóttir and S. H. Kim. Fully sequential procedures for comparing constrained systems via simulation. Naval Research Logistics, 57(5):403–421, 2010.

B. Ankenman, B. L. Nelson, and J. Staum. Stochastic kriging for simulation metamodeling. Operations Research, 58(2):371–382, 2010.

F. Bachoc, T. Cesari, and S. Gerchinovitz. The sample complexity of level set approximation. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, pages 424–432, 2021.

D. Batur and S. H. Kim. Finding feasible systems in the presence of constraints on multiple performance measures. ACM Transactions on Modeling and Computer Simulation, 20(3): 13:1–13:26, 2010.

I. Bogunovic, J. Scarlett, A. Krause, and V. Cevher. Truncated variance reduction: A unified approach to Bayesian optimization and level-set estimation. In Advances in Neural Information Processing Systems, 2016.

S. Bubeck, R. Munos, G. Stoltz, and C. Szepesvári. X-armed bandits. Journal of Machine Learning Research, 12(46):1655–1695, 2011.

C. H. Chen, J. Lin, E. Yücesan, and S. E. Chick. Simulation budget allocation for further enhancing the eficiency of ordinal optimization. Discrete Event Dynamic Systems, 10(3):251–270, 2000.

F. K. Ewald, M. Binder, M. Feurer, B. Bischl, and G. Casalicchio. CASHomon sets: Eficient Rashomon sets across multiple model classes and their hyperparameters. In Machine Learning and Knowledge Discovery in Databases. Research Track, pages 319–337, 2027.

Q. Fan and J. Hu. Surrogate-based promising area search for Lipschitz continuous simulation optimization. INFORMS Journal on Computing, 30(4):677–693, 2018.

W. Fan, L. J. Hong, and B. L. Nelson. Indiference-zone-free selection of the best. Operations Research, 64(6):1499–1514, 2016.

P. I. Frazier. A tutorial on Bayesian optimization. arXiv preprint arXiv:1807.02811, 2018.

P. I. Frazier, W. B. Powell, and S. Dayanik. A knowledge-gradient policy for sequential information collection. SIAM Journal on Control and Optimization, 47(5):2410–2439, 2008.

A. Gotovos, N. Casati, G. Hitz, and A. Krause. Active learning for level set estimation. In Proceedings of the Twenty-Third International Joint Conference on Artificial Intelligence, pages 1344–1350, 2013.

L. J. Hong and B. L. Nelson. Discrete optimization via simulation using COMPASS. Operations Research, 54(1):115–129, 2006.

S. Iwazaki. Improved regret bounds for Gaussian process upper confidence bound in Bayesian optimization. In Advances in Neural Information Processing Systems, 2025.

E. Kaufmann, O. Cappé, and A. Garivier. On the complexity of best-arm identification in multi-armed bandit models. Journal of Machine Learning Research, 17(1):1–42, 2016.

S. H. Kim and B. L. Nelson. A fully sequential procedure for indiference-zone selection in simulation. ACM Transactions on Modeling and Computer Simulation, 11(3):251–273, 2001.

Z. Li and J. Scarlett. Gaussian process bandit optimization with few batches. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, pages 92–107, 2022.

A. Locatelli, M. Gutzeit, and A. Carpentier. An optimal algorithm for the thresholding bandit problem. In Proceedings of the 33rd International Conference on Machine Learning, pages 1690–1698, 2016.

B. Mason, L. Jain, A. Tripathy, and R. Nowak. Finding all ϵ-good arms in stochastic bandits. In Advances in Neural Information Processing Systems, pages 20707–20718, 2020.

B. Mason, L. Jain, S. Mukherjee, R. Camilleri, K. Jamieson, and R. Nowak. Nearly optimal algorithms for level set estimation. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, pages 7625–7658, 2022.

F. J. Matejcik and B. L. Nelson. Two-stage multiple comparisons with the best for computer simulation. Operations Research, 43(4):633–640, 1995.

N. Maus, K. Wu, D. Eriksson, and J. Gardner. Discovering many diverse solutions with Bayesian optimization. arXiv preprint arXiv:2210.10953, 2022.

J. J. Miller, S. Mak, B. Sun, S. R. Narayanan, S. Yang, Z. Sun, K. S. Kim, and C.-B. M. Kweon. Expected diverse utility (EDU): Diverse Bayesian optimization of expensive computer simulators. arXiv preprint arXiv:2410.01196, 2024.

S. Minsker. Estimation of extreme values and associated level sets of a regression function via selective sampling. In Proceedings of the 26th Annual Conference on Learning Theory, pages 105–121, 2013.

B. L. Nelson, J. Swann, D. Goldsman, and W. Song. Simple procedures for selecting the best simulated system when the number of alternatives is large. Operations Research, 49(6):950–963, 2001.

G. Ngo, D. Nguyen, D. Phan-Trong, and S. Gupta. Active level set estimation for continuous search space with theoretical guarantee. In Proceedings of the 15th Asian Conference on Machine Learning, pages 943–958, 2024.

G. Ngo, D. P. Trong, D. Nguyen, and S. Gupta. High-dimensional level set estimation with trust regions and double acquisition functions. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, pages 1315–1323, 2026.

Q. P. Nguyen, B. K. H. Low, and P. Jaillet. An information-theoretic framework for unifying active learning problems. Proceedings of the AAAI Conference on Artificial Intelligence, 35(10): 9126–9134, 2021.

C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.

P. L. Salemi, E. Song, B. L. Nelson, and J. Staum. Gaussian Markov random fields for discrete optimization via simulation: Framework and algorithms. Operations Research, 67(1):250–266, 2019.

J. Scarlett, I. Bogunovic, and V. Cevher. Lower bounds on regret for noisy Gaussian process bandit optimization. In Proceedings of the 2017 Conference on Learning Theory, pages 1723–1742, 2017.

A. M. Schweidtmann, A. D. Clayton, N. Holmes, E. Bradford, R. A. Bourne, and A. A. Lapkin. Machine learning meets continuous flow chemistry: Automated optimization towards the Pareto front of multiple objectives. Chemical Engineering Journal, 352:277–282, 2018.

S. Shekhar and T. Javidi. Multiscale Gaussian process level set estimation. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, pages 3283–3291, 2019.

S. Shekhar and T. Javidi. Instance dependent regret analysis of kernelized bandits. In Proceedings of the 39th International Conference on Machine Learning, pages 19747–19772, 2022.

D. W. Sullivan and J. R. Wilson. Restricted subset selection procedures for simulation. Operations Research, 37(1):52–71, 1989.

S. Vakili, N. Bouziani, S. Jalali, A. Bernacchia, and D. Shiu. Optimal order simple regret for Gaussian process bandits. In Advances in Neural Information Processing Systems, pages 21202–21215, 2021.

J. Verhellen and J. Van den Abeele. Illuminating elite patches of chemical space. Chemical Science, 11(42):11485–11491, 2020.

Y. Wang, S. Balakrishnan, and A. Singh. Optimization of smooth functions with noisy observations: Local minimax rates. In Advances in Neural Information Processing Systems, 2018.

H. Wendland. Scattered Data Approximation. Cambridge University Press, 2004.

F. Zhang, J. Song, J. C. Bowden, A. Ladd, Y. Yue, T. Desautels, and Y. Chen. Learning regions of interest for Bayesian optimization with adaptive level-set estimation. In Proceedings of the 40th International Conference on Machine Learning, pages 41579–41595, 2023.

## Appendices

The appendices first present additional confidence bounds and algorithmic guarantees (Appendix A), then give the complete proofs (Appendices B and C), and finally provide additional numerical details and results (Appendix D).

## A Confidence Bounds and Algorithmic Guarantees

## A.1 Confidence Bounds for Near-Optimal Regions

The region argument in this subsection does not require a Matérn kernel, a mode count, or Euclidean geometry. Let X be a compact metric space and let $f : \mathcal { X } \to \mathbb { R }$ be continuous. Proposition A.1 below is deterministic. It only requires pointwise bounds $L ( x ) \leq f ( x ) \leq U ( x )$ for all $x \in \mathcal { X }$ and bounds $f \leq f ^ { \star } \leq \overline { { f } }$ on the optimal value. In SASORI, after a completed phase h, these bounds are given by the shared confidence bounds $L = \ell _ { h }$ and $U = u _ { h }$ and by their maxima $\underline { { f } } = \underline { { f } } _ { h }$ and ${ \overline { { f } } } = { \overline { { f } } } _ { h }$ in (14). They hold on the shared confidence event ${ \mathcal { E } } _ { \delta }$ in (13), on which all confidence bands computed by either track contain $f ,$ so that $\ell _ { h } \leq f \leq u _ { h }$ and hence $\begin{array} { r } { \underline { { f } } _ { h } = \operatorname* { m a x } _ { \mathcal { X } } \ell _ { h } \le f ^ { \star } \le \operatorname* { m a x } _ { \mathcal { X } } u _ { h } = \overline { { f } } _ { h } } \end{array}$ . By Lemma 5.1, $\mathbb { P } ( \mathcal { E } _ { \delta } ) \geq 1 - \delta$ . Therefore, on ${ \mathcal { E } } _ { \delta }$ , Proposition A.1 applies simultaneously to all completed phases and to the final output.

Proposition A.1 (Simultaneous region bounds with separate error budgets). Suppose $L ( x ) \leq$ $f ( x ) \leq U ( x )$ on X and $\underline { { f } } \leq f ^ { \star } \leq \overline { { f } } ;$ , where $\Gamma = \overline { { f } } - \underline { { f } } \geq 0$ . Define, for any $\tau \geq 0$

$$
I _ { \tau } = \{ x : L ( x ) > \overline { { { f } } } - \tau \} , O _ { \tau } = \{ x : U ( x ) > f - \tau \} .\tag{A.1}
$$

Then, simultaneously for all $\tau \geq 0$

$$
I _ { \tau } \subseteq S _ { \tau } \subseteq O _ { \tau } .\tag{A.2}
$$

If $U ( x ) - L ( x ) \leq w _ { \tau }$ on $O _ { \tau }$ for a nonnegative bound $w _ { \tau }$ , let $\eta _ { \tau } = \Gamma + w _ { \tau }$ . Then

$$
O _ { \tau } \setminus I _ { \tau } \subseteq \{ x : \tau - \eta _ { \tau } \leq f ^ { \star } - f ( x ) < \tau + \eta _ { \tau } \} .\tag{A.3}
$$

More precisely, at an unclassified design point of width $w ( x ) = U ( x ) - L ( x )$ ，

$$
\tau - ( \overline { { f } } - f ^ { \star } ) - w ( x ) \leq f ^ { \star } - f ( x ) < \tau + ( f ^ { \star } - \underline { { f } } ) + w ( x ) .\tag{A.4}
$$

With ${ \cal S } _ { v } = \emptyset$ for $v \leq 0$ , one also has

$$
S _ { \tau - \eta _ { \tau } } \subseteq I _ { \tau } \subseteq S _ { \tau } \subseteq O _ { \tau } \subseteq S _ { \tau + \eta _ { \tau } } .\tag{A.5}
$$

The optimization uncertainty Γ and local function uncertainty $w _ { \tau }$ need not be equal. If w<sub>T</sub> bounds the width on $O _ { T }$ , the same error $\Gamma + w _ { T }$ works simultaneously for all $0 \leq \tau \leq T$

## A.2 Phase Guarantees of SASORI

This subsection discusses two guarantees for the phases of Algorithm 1 that are used in the analysis of Section 5. The first, Proposition 4.1 in Section 4, shows that the accuracy guarantees established by a track remain valid after any later update of the shared confidence bounds, and the second gives the guarantees of a single phase of either track. Their proofs are given in Sections B.19 and B.20.

The following proposition provides the phase guarantees used by Theorem 5.1 and the subsequent regret analysis. It applies to either track, including when both tracks share confidence bounds.

Proposition A.2. Let X be a nonempty compact subset of $\mathbb { R } ^ { d }$ , and suppose Assumptions 5.2–5.3 hold. Consider a track of Algorithm 1, letting $T \geq 0$ for its tolerance and h for its local phase index. Use the event ${ \mathcal { E } } _ { \delta }$ in (13), whose probability is at least $1 - \delta$ by Lemma 5.1. On ${ \mathcal { E } } _ { \delta }$ , any completed phase with $\varepsilon _ { h } \geq 2 \zeta _ { N }$ satisfies

$$
\begin{array} { r } { u _ { h } ( x ) - \ell _ { h } ( x ) \le 2 \varepsilon _ { h } \ w h e n e v e r \ u _ { h } ( x ) > \underline { { f } } _ { h } - T - 2 \varepsilon _ { h } , \overline { { f } } _ { h } - \underline { { f } } _ { h } \le 2 \varepsilon _ { h } , f ^ { \star } - f ( \widehat { x } _ { h } ) \le 2 \varepsilon _ { h } . } \end{array}
$$

Its region outputs satisfy, simultaneously for every $0 \leq \tau \leq T$

$$
I _ { h , \tau } \subseteq S _ { \tau } \subseteq O _ { h , \tau } , O _ { h , \tau } \setminus I _ { h , \tau } \subseteq \{ x : | f ^ { \star } - f ( x ) - \tau | \leq 4 \varepsilon _ { h } \} .\tag{A.6}
$$

$A t \tau = 0$ , both output sets are empty by convention. Moreover, any point sampled in a phase with $\varepsilon _ { h } \geq 2 \zeta _ { N }$ , including a phase truncated by its track’s budget, satisfies

$$
x _ { h , j } \in S _ { ( T + L _ { \vartheta } \varepsilon _ { h } ) + } .\tag{A.7}
$$

Proposition 4.1 explains why the two tracks of SASORI can share their confidence bands without interfering with each other. After track j completes a phase with accuracy level ε¯, its guarantees consist of a bound 2ε¯ on the width $\boldsymbol { u } ^ { \mathrm { o l d } } - \boldsymbol { \ell } ^ { \mathrm { o l d } }$ over the designs that are still relevant at its tolerance $T _ { j }$ , namely those with $u ^ { \mathrm { o l d } } ( x ) > f ^ { \mathrm { o l d } } - T _ { j } - 2 \bar { \varepsilon }$ , and a bound 2ε¯ on the optimal-value error $f ^ { \star } - f ^ { \mathrm { o l d } }$ Any later band, whether it is provided by the same track or by the other track, is intersected with the current confidence bounds, so the upper confidence bound can only decrease, while the lower confidence bound and its maximum can only increase. Hence the set of relevant designs can only shrink, the widths on this set can only decrease, and $\underline { f }$ can only move closer to $f ^ { \star }$ . In particular, evaluations spent by the region track never invalidate the guarantee of the optimization track, and evaluations spent by the optimization track never invalidate an earlier region guarantee. Because all bands are valid on the single event ${ \mathcal { E } } _ { \delta }$ , this sharing requires neither a separate data set nor an additional union bound.

Proposition A.2 gives the two properties of a single phase on which the complexity and regret analyses rely. The first concerns accuracy. When a phase with accuracy level $\varepsilon _ { h }$ completes, the stopping rule (12) forces the width $u _ { h } - \ell _ { h }$ to be at most $2 \varepsilon _ { h }$ at any design that may still matter at tolerance T. Applied at a maximizer of $u _ { h }$ , this gives $\overline { { f } } _ { h } - \underline { { f } } _ { h } \leq 2 \varepsilon _ { h }$ , so the estimated optimal point has optimality gap at most $2 \varepsilon _ { h }$ . Combined with Proposition $\mathrm { A . 1 }$ , it gives the region guarantee with region estimation error $4 \varepsilon _ { h }$ simultaneously for all $\tau \in [ 0 , T ]$ . The second concerns where the evaluations are spent. By (A.7), any point sampled in phase $h ,$ including one sampled in a phase cut of by the budget, has performance gap at most $T + L _ { \vartheta } \varepsilon _ { h }$ . This has two consequences. First, each evaluation contributes at most $T + L _ { \vartheta } \varepsilon _ { h }$ to the cumulative regret, which yields the regret bounds in Theorem 5.5 and Proposition 5.2. Second, under Assumption 5.4 and when ${ T + L _ { \vartheta } \varepsilon _ { h } } \leq u _ { 0 }$ , the points sampled in phase h lie in a set covered by K balls of radius $c _ { \rho } ( T + L _ { \vartheta } \varepsilon _ { h } ) ^ { 1 / \alpha }$ , which yields the localized evaluation bound (26). For the optimization track $( T = 0 )$ , the sampled points therefore concentrate near the maximizers as $\varepsilon _ { h }$ decreases. For the region track $( T > 0 )$ , they may remain at designs whose performance gap is of order T, which is the source of the regret cost of region learning quantified in Section 5.4.

## B Proofs of the Main Results

This section proves the theorems, propositions, and corollaries stated in the main text and in Section A. The proofs of all lemmas used in this section are provided in Section C.

We use the following notation throughout Sections B and C. With $s = \nu + d / 2$ as in Section $2 . 3 ,$ let $p = 2 + d / \nu$ and $a =$ max $\{ 2 , 2 + d / \nu - d / \alpha \}$ , so that $\rho _ { N } = \operatorname* { m i n } \{ ( K / N ) ^ { 1 / a } , N ^ { - 1 / p } \}$ in (29). For the logarithmic factors, let

$$
\Lambda _ { N , \delta } = \log ^ { 2 + d / \nu } ( 2 N ) \log ^ { 1 + d / ( 2 \nu ) } ( 2 N / \delta ) , L _ { N } = \log ^ { 3 + 3 d / ( 2 \nu ) } ( 2 N ) , \mathcal { V } _ { N } = \beta _ { N } ^ { 2 } \log ^ { 2 } ( 2 N ) ,\tag{B.1}
$$

where $\beta _ { N }$ is calibrated at error probability δ in (9). By the definition of $\beta _ { N }$ and $M _ { N } \ \leq \ ( 1 +$ $2 D _ { \mathcal { X } } ) ^ { d } N ^ { 4 d / \theta }$ , we have $\beta _ { N } ^ { 2 } \le 2 B ^ { 2 } + ( 4 \sigma ^ { 2 } / \lambda ) \log ( 4 M _ { N } N ^ { 2 } / \delta ) \le C \log ( 2 N / \delta )$ . Since also $\beta _ { N } \geq B$ $N \geq 2$ , and $\log ( 2 N ^ { 3 } ) \leq 3 \log ( 2 N )$

$$
B ^ { 2 } \log ^ { 2 } 4 \leq \mathcal { V } _ { N } \leq C \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } \leq C \Lambda _ { N , \delta } , 1 \leq \Lambda _ { N , \delta } , \Lambda _ { N , N ^ { - 2 } } \leq 3 ^ { 1 + d / ( 2 \nu ) } L _ { N } .\tag{B.2}
$$

For $\varphi \in H ^ { s } ( \mathbb R ^ { d } ) , z \in \mathbb R ^ { d }$ , and $t \geq 1$ , the Fourier representation of the $H ^ { s } ( \mathbb R ^ { d } )$ norm gives the scaling inequality

$$
\| \varphi ( t ( \cdot - z ) ) \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } \leq t ^ { s - d / 2 } \| \varphi \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } = t ^ { \nu } \| \varphi \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } .\tag{B.3}
$$

Moreover, the Matérn RKHS on $\mathbb { R } ^ { d }$ coincides with $H ^ { s } ( \mathbb R ^ { d } )$ with an equivalent norm, and the RKHS of the restriction of k to $\mathcal { X } \times \mathcal { X }$ consists of the restrictions to X of the functions in the RKHS on $\mathbb { R } ^ { d }$ where the RKHS norm on X of a function is the smallest RKHS norm on $\mathbb { R } ^ { d }$ among its extensions (Wendland 2004, Chapter 10). Hence, for any $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ and without any regularity condition on its boundary, the RKHS norm on X of a function is at most a fixed constant times the $H ^ { s } ( \mathbb R ^ { d } )$ norm of any extension of it to $\mathbb { R } ^ { d }$ . Unless stated otherwise, C denotes a positive constant depending only on the fixed parameters, whose value may change between occurrences.

## B.1 Proof of Theorem 3.1

We first present a lemma used in this proof, which is also used in the proof of Theorem 5.3, and its proof is given in Section C.2.

Lemma B.1. $I f \alpha > \nu$ and $\varphi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ , then the function $x \mapsto \varphi ( x ) \| x \| ^ { \alpha }$ belongs to $H ^ { \nu + d / 2 } ( \mathbb R ^ { d } )$ Consequently, for any fixed $c \in \mathbb { R } ^ { d }$ , the restriction of the function $x \mapsto \| x - c \| ^ { \alpha }$ to a fixed bounded domain belongs to the Matérn RKHS and has finite RKHS norm.

Proof of Theorem 3.1. Choose $x ^ { \star } \in \mathcal { X }$ and $r _ { * } > 0$ such that $\overline { { B } } ( x ^ { \star } , 2 r _ { * } ) \subset \mathrm { i n t } ( { \mathcal { X } } )$ . By Lemma B.1, $C _ { g } = \left\| \| \cdot - x ^ { \star } \| ^ { \alpha } \right\| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } < \infty$ . Set $\mu = 2 ( 4 / c _ { \rho } ) ^ { \alpha } , \kappa _ { * } = \mu ^ { d / \alpha } r _ { * } ^ { d } , k _ { T } = \operatorname* { m i n } \{ K , \left\lfloor \kappa _ { * } T ^ { - d / \alpha } \right\rfloor \}$ . Choose the fixed $K _ { \mathrm { r e g } }$ large enough that $\mu C _ { g } K _ { \mathrm { r e g } } ^ { - \alpha / d } \le B / 2$ . Choose $T _ { \mathrm { r e g } }$ so that $T _ { \mathrm { r e g } } \leq$ min $\{ 1 , u _ { 0 } / 2 , 2 0 4 8 B \}$ and $\kappa _ { * } T _ { \mathrm { r e g } } ^ { - d / \alpha } \geq 2 K _ { \mathrm { r e g } }$ . This proof does not use the constraint $T _ { \mathrm { r e g } } \leq$ min $\{ 1 , u _ { 0 } / 2 , 2 0 4 8 B \}$ ; it is imposed so that the same $T _ { \mathrm { r e g } }$ can be used in the upper bounds of Corollaries 5.1 and 5.2. Then $k _ { T } \geq K _ { \mathrm { r e g } }$ for all parameters in Theorem 3.1. Define $g ( x ) = - \mu k _ { T } ^ { - \alpha / d } \| x - x ^ { \star } \| ^ { \alpha }$ . The baseline has RKHS norm at most $B / 2$ , unique global maximum $g ( x ^ { \star } ) = 0$ , and a norm bound independent of $K , T$ . Letting $R = ( T / \dot { \mu } ) ^ { 1 / \alpha } k _ { T } ^ { 1 / d }$ , we have $R \leq r _ { * }$ and

$$
R ^ { d } \asymp \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} .\tag{B.4}
$$

The constants in this comparison are fixed.

Let $\psi \in C _ { c } ^ { \infty } ( B ( 0 , 1 ) )$ satisfy $0 \leq \psi \leq 1$ and $\psi ( 0 ) = 1$ . Fourier scaling gives a fixed $C _ { \psi }$ such that $\| \psi ( ( \cdot - z ) / h ) \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \le C _ { \psi } h ^ { - \nu } \mathrm { ~ f o r ~ } 0 < h \le 1$ uniformly for the centers considered below. Set $L = ( 1 6 C _ { \psi } / B ) ^ { 1 / \nu }$ and $h = L \eta ^ { 1 / \nu }$

Let $\mathcal { A } _ { R } = \{ x : 4 ^ { - 1 / \alpha } R \leq \| x - x ^ { \star } \| \leq 2 ^ { - 1 / \alpha } R \}$ . Choose a fixed $\rho _ { 0 } > 0$ small enough that any point within distance $\rho _ { 0 } R$ of $\boldsymbol { \mathcal { A } } _ { R }$ has baseline gap in $[ T / 8 , 3 T / 4 ]$ and lies in $B ( x ^ { \star } , R )$ . Such a

choice exists because the baseline gap is $- g ( x ) = T ( \| x - x ^ { \star } \| / R ) ^ { \alpha }$ , which lies in $[ T / 8 , 3 T / 4 ]$ exactly when $8 ^ { - 1 / \alpha } R \leq \| x - x ^ { \star } \| \leq ( 3 / 4 ) ^ { 1 / \alpha } R$ , and $8 ^ { - 1 / \alpha } < 4 ^ { - 1 / \alpha } < 2 ^ { - 1 / \alpha } < ( 3 / 4 ) ^ { 1 / \alpha } < 1$ . Since $\alpha > \nu ,$ decreasing the fixed $T _ { \mathrm { r e g } }$ further ensures

$$
\frac { h } { R } = L \mu ^ { 1 / \alpha } k _ { T } ^ { - 1 / d } \eta ^ { 1 / \nu } T ^ { - 1 / \alpha } \le L \mu ^ { 1 / \alpha } K _ { \mathrm { r e g } } ^ { - 1 / d } 1 2 8 ^ { - 1 / \nu } T ^ { 1 / \nu - 1 / \alpha } \le \rho _ { 0 } , h \le 1 .\tag{B.5}
$$

Take a maximal 2h-separated set of centers $z _ { i }$ in $\boldsymbol { \mathcal { A } } _ { R }$ . The support balls $A _ { i } = B ( z _ { i } , h )$ are disjoint. By maximality, any point of $\boldsymbol { \mathcal { A } } _ { R }$ lies within distance 2h of some center, so the balls $B ( z _ { i } , 2 h )$ cover $\boldsymbol { \mathcal { A } } _ { R }$ . Since the volume of $\boldsymbol { \mathcal { A } } _ { R }$ is a constant multiple of $R ^ { d }$ , comparing volumes and using (B.4) gives

$$
M \geq c ( R / h ) ^ { d } \geq c \eta ^ { - d / \nu } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} .\tag{B.6}
$$

The fixed $\rho _ { 0 }$ can also be decreased to ensure $M \geq 2$ . At each center, $( - g ( z _ { i } ) ) \in [ T / 4 , T / 2 ]$ , and throughout each support, $( - g ( x ) ) \in [ T / 8 , 3 T / 4 ]$

Define the alternatives $f _ { i } ( x ) = g ( x ) + 8 \eta \psi ( ( x - z _ { i } ) / h )$ . Their RKHS norms are at most $B / 2 + 8 \eta C _ { \psi } h ^ { - \nu } = B$ . Because $8 \eta \leq T / 1 6$ and $- g \ge T / 8$ on the supports, each perturbation stays strictly below zero. Any $f _ { i }$ therefore keeps the same unique global maximizer $x ^ { \star }$ and the same value $f _ { i } ^ { \star } = 0$ . In particular,

$$
\begin{array} { r } { \frac 1 2 ( - g ( x ) ) \le \big ( f _ { i } ^ { \star } - f _ { i } ( x ) \big ) = ( - g ( x ) ) - 8 \eta \psi ( ( x - z _ { i } ) / h ) \le ( - g ( x ) ) \mathrm { ~ f o r ~ } x \in \mathcal { X } . } \end{array}\tag{B.7}
$$

For the baseline and any alternative, (B.7) implies

$$
\begin{array} { r } { S _ { u + } \subseteq \overline { { B } } \left( x ^ { \star } , ( 2 u / \mu ) ^ { 1 / \alpha } k _ { T } ^ { 1 / d } \right) = \overline { { B } } \left( x ^ { \star } , \frac { 1 } { 4 } c _ { \rho } k _ { T } ^ { 1 / d } u ^ { 1 / \alpha } \right) , } \end{array}
$$

for any $u > 0$ . This ball is coverable by at most $k _ { T }$ balls of radius $c _ { \rho } u ^ { 1 / \alpha }$ . For $k _ { T } < 2 ^ { d }$ , one ball sufices; for $k _ { T } \geq 2 ^ { d }$ , a maximal separated packing gives the upper bound $( 1 + k _ { T } ^ { 1 / d } / 2 ) ^ { d } \leq k _ { T }$ . Thus $g , f _ { 1 } , \dots , f _ { M }$ all belong to the same prescribed $\mathcal { F } _ { K }$ , since $k _ { T } \leq K$

Consider any $( \eta , T , \delta )$ )-valid procedure with fixed budget n. For a response surface $f \in \mathcal { F } _ { K }$ , let $\mathcal { E } _ { f }$ be the event that the output $( I _ { \tau } , O _ { \tau } ) _ { 0 \leq \tau \leq T }$ satisfies (5) for all $\tau \in [ 0 , T ]$ , which we call the validity event under f. By the definition of an $( \eta , T , \delta )$ -valid procedure, the probability of $\mathcal { E } _ { f }$ is at least $1 - \delta$ when the response surface is $f .$ . We use this property for $f = g$ and for $f = f _ { i } , 1 \leq i \leq M$ , all of which belong to $\mathcal { F } _ { K }$ by the covering argument above.

For each i, set $\tau _ { i } = ( - g ( z _ { i } ) ) - 4 \eta .$ , which lies in $( 0 , T )$ because $- g ( z _ { i } ) \in [ T / 4 , T / 2 ]$ and $4 \eta \le T / 3 2$ and let $E _ { i } = \{ z _ { i } \in I _ { \tau _ { i } } \}$ . Since $z _ { i }$ and $\tau _ { i }$ are deterministic, $E _ { i }$ is an event by the measurability of membership in the output sets. Because $g ^ { \star } = f _ { i } ^ { \star } = 0$ and $\psi ( 0 ) = 1$ , the performance gaps of $z _ { i }$ are $g ^ { \star } - g ( z _ { i } ) = \tau _ { i } + 4 \eta$ under $g$ and $f _ { i } ^ { \star } - f _ { i } ( z _ { i } ) = - g ( z _ { i } ) - 8 \eta = \tau _ { i } - 4 \eta$ under $f _ { i }$ . We now show that $E _ { i } \cap \mathcal { E } _ { g } = \emptyset$ and $\mathcal { E } _ { f _ { i } } \subseteq E _ { i }$

• On $\mathcal { E } _ { g }$ , since $g ^ { \star } - g ( z _ { i } ) > \tau _ { i }$ , we have $z _ { i } \not \in S _ { \tau _ { i } } ( g )$ , and the inclusion $I _ { \tau _ { i } } \subseteq S _ { \tau _ { i } } ( g )$ gives $z _ { i } \notin I _ { \tau _ { i } }$

• On $\mathcal { E } _ { f _ { i } }$ , since $f _ { i } ^ { \star } - f _ { i } ( z _ { i } ) < \tau _ { i }$ , we have $z _ { i } \in S _ { \tau _ { i } } ( f _ { i } ) \subseteq O _ { \tau _ { i } }$ . Moreover, $| f _ { i } ^ { \star } - f _ { i } ( z _ { i } ) - \tau _ { i } | = 4 \eta > \eta$ so the second condition in (5) gives $z _ { i } \notin O _ { \tau _ { i } } \setminus I _ { \tau _ { i } }$ . Hence $z _ { i } \in I _ { \tau _ { i } }$

Let $P _ { 0 }$ and $P _ { i }$ denote the joint distributions of the sampling history and the output, including the internal randomization of the procedure, when the response surface is $g$ and $f _ { i }$ , respectively. The two inclusions give $P _ { 0 } ( E _ { i } ) \leq P _ { 0 } ( \mathcal { E } _ { q } ^ { c } ) \leq \delta$ and $P _ { i } ( E _ { i } ) \geq P _ { i } ( \mathcal { E } _ { f _ { i } } ) \geq 1 - \delta$ . The first inequality uses the validity of the procedure under the baseline g.

Next, we bound $\mathrm { K L } ( P _ { 0 } , P _ { i } )$ from below. For $p , q \in [ 0 , 1 ] , \operatorname { k l } ( p , q )$ is the relative entropy between the Bernoulli distributions with means $p$ and $q .$ Since $\mathbf { 1 } _ { E _ { i } }$ is a measurable function of the history and the output, the data-processing inequality for relative entropy gives $\operatorname { K L } ( P _ { 0 } , P _ { i } ) \geq \operatorname { k l } ( P _ { 0 } ( E _ { i } ) , P _ { i } ( E _ { i } ) )$ ; see, for example, Kaufmann et al. (2016, Lemma 1). For fixed $q ,$ the function $p \mapsto \operatorname { k l } ( p , q )$ is convex with minimum value zero at $p = q$ , so it is nonincreasing for $p \leq q .$ Similarly, for fixed $p ,$ , the function $q \mapsto \operatorname { k l } ( p , q )$ is nondecreasing for $q \geq p$ . Because $\delta \leq 1 / 1 0$ , we have $P _ { 0 } ( E _ { i } ) \leq \delta < 1 - \delta \leq P _ { i } ( E _ { i } )$ and hence

$$
\operatorname { K L } ( P _ { 0 } , P _ { i } ) \geq \operatorname { k l } \big ( P _ { 0 } ( E _ { i } ) , P _ { i } ( E _ { i } ) \big ) \geq \operatorname { k l } \big ( \delta , P _ { i } ( E _ { i } ) \big ) \geq \operatorname { k l } ( \delta , 1 - \delta ) .\tag{B.8}
$$

We now compute $\mathrm { K L } ( P _ { 0 } , P _ { i } )$ . Let U denote the internal randomization of the procedure, with distribution $P _ { U }$ , let $\mathcal { H } _ { 0 } = U$ , and let $\mathcal { H } _ { t } = ( U , x _ { 1 } , y _ { 1 } , \dots , x _ { t } , y _ { t } )$ be the history after t evaluations. Let $o = ( I _ { \tau } , O _ { \tau } ) _ { 0 \leq \tau \leq T }$ denote the output. The procedure is specified by the following decision rules:

• for each $t = 1 , \ldots , n .$ , a conditional distribution $\pi _ { t } ( \cdot \mid \mathcal { H } _ { t - 1 } )$ on $\mathcal { X }$ , according to which $x _ { t }$ is chosen given $\mathcal { H } _ { t - 1 } ;$

• a conditional distribution $\pi _ { \mathrm { o u t } } ( \cdot \mid \mathcal { H } _ { n } )$ , according to which the output o is chosen given ${ \mathcal { H } } _ { n }$

The procedure does not observe the response surface, which afects the data only through the responses $y _ { t }$ . Hence the same $P _ { U } , \pi _ { 1 } , \ldots , \pi _ { n } ,$ , and $\pi _ { \mathrm { o u t } }$ apply under $g$ and under each $f _ { i }$ , although the distributions of the histories difer. Under a response surface $f ,$ , given $\mathcal { H } _ { t - 1 }$ and $x _ { t }$ , the response is $y _ { t } =$ $f ( x _ { t } ) + \varepsilon _ { t }$ , where $\varepsilon _ { t } \sim N ( 0 , \sigma ^ { 2 } )$ is independent of $( \mathcal { H } _ { t - 1 } , x _ { t } )$ . Let $\phi _ { \sigma } ( z ) = ( 2 \pi \sigma ^ { 2 } ) ^ { - 1 / 2 } \exp \{ - z ^ { 2 } / ( 2 \sigma ^ { 2 } ) \}$ denote the $N ( 0 , \sigma ^ { 2 } )$ density. The joint distribution $P _ { f }$ of $( { \mathcal { H } } _ { n } , o )$ therefore factorizes as

$$
\begin{array} { r l } & { P _ { f } ( d u , d x _ { 1 } , d y _ { 1 } , \ldots , d x _ { n } , d y _ { n } , d o ) } \\ & { \ = P _ { U } ( d u ) \prod _ { t = 1 } ^ { n } \Bigl \{ \pi _ { t } ( d x _ { t } \mid \mathcal { H } _ { t - 1 } ) \phi _ { \sigma } \bigl ( y _ { t } - f ( x _ { t } ) \bigr ) d y _ { t } \Bigr \} \pi _ { \mathrm { o u t } } ( d o \mid \mathcal { H } _ { n } ) . } \end{array}
$$

Hence, with respect to the common measure obtained by replacing each factor $\phi _ { \sigma } ( y _ { t } - f ( x _ { t } ) ) d y _ { t }$ in this factorization by $d y _ { t }$ , the distributions $P _ { 0 }$ and $P _ { i }$ have the positive densities $\textstyle \prod _ { t = 1 } ^ { n } \phi _ { \sigma } ( y _ { t } - g ( x _ { t } ) )$ and $\textstyle \prod _ { t = 1 } ^ { n } \phi _ { \sigma } ( y _ { t } - f _ { i } ( x _ { t } ) )$ , respectively. In particular, $P _ { 0 }$ and $P _ { i }$ are mutually absolutely continuous, and the decision rules cancel in the likelihood ratio as

$$
\log { \frac { d P _ { 0 } } { d P _ { i } } } = \sum _ { t = 1 } ^ { n } \log { \frac { \phi _ { \sigma } ( y _ { t } - g ( x _ { t } ) ) } { \phi _ { \sigma } ( y _ { t } - f _ { i } ( x _ { t } ) ) } } = \sum _ { t = 1 } ^ { n } { \frac { \{ y _ { t } - f _ { i } ( x _ { t } ) \} ^ { 2 } - \{ y _ { t } - g ( x _ { t } ) \} ^ { 2 } } { 2 \sigma ^ { 2 } } } .
$$

Under $P _ { 0 }$ , we have $y _ { t } = g ( x _ { t } ) + \varepsilon _ { t } ,$ so the t-th summand equals

$$
\frac { \{ g ( x _ { t } ) - f _ { i } ( x _ { t } ) + \varepsilon _ { t } \} ^ { 2 } - \varepsilon _ { t } ^ { 2 } } { 2 \sigma ^ { 2 } } = \frac { \{ g ( x _ { t } ) - f _ { i } ( x _ { t } ) \} ^ { 2 } + 2 \varepsilon _ { t } \{ g ( x _ { t } ) - f _ { i } ( x _ { t } ) \} } { 2 \sigma ^ { 2 } } .
$$

Since $\mathbb { E } _ { 0 } [ \varepsilon _ { t } \mid \mathcal { H } _ { t - 1 } , x _ { t } ] = 0$ and $| g - f _ { i } | \le 8 \eta$ , the conditional expectation of the tth summand given $( \mathcal { H } _ { t - 1 } , x _ { t } )$ is $\{ g ( x _ { t } ) - f _ { i } ( x _ { t } ) \} ^ { 2 } / ( 2 \sigma ^ { 2 } )$ . By the tower property and the definition of relative entropy, we have

$$
\operatorname { K L } ( P _ { 0 } , P _ { i } ) = \mathbb { E } _ { 0 } \left[ \log { \frac { d P _ { 0 } } { d P _ { i } } } \right] = { \frac { 1 } { 2 \sigma ^ { 2 } } } \mathbb { E } _ { 0 } \sum _ { t = 1 } ^ { n } \{ g ( x _ { t } ) - f _ { i } ( x _ { t } ) \} ^ { 2 } .\tag{B.9}
$$

Let $\begin{array} { r } { N _ { i } = \sum _ { t = 1 } ^ { n } \mathbf { 1 } \{ x _ { t } \in A _ { i } \} } \end{array}$ be the number of evaluations allocated to $A _ { i }$ . Since $0 \leq f _ { i } - g =$ $8 \eta \psi ( ( \cdot - z _ { i } ) / h ) \le 8 \eta \mathbf { 1 } _ { A _ { i } }$ , we have $\{ g ( x _ { t } ) - f _ { i } ( x _ { t } ) \} ^ { 2 } \leq 6 4 \eta ^ { 2 } \mathbf { 1 } \{ x _ { t } \in A _ { i } \}$ , and therefore

$$
\mathrm { K L } ( P _ { 0 } , P _ { i } ) \leq \frac { 6 4 \eta ^ { 2 } } { 2 \sigma ^ { 2 } } \mathbb { E } _ { 0 } N _ { i } = \frac { 3 2 \eta ^ { 2 } } { \sigma ^ { 2 } } \mathbb { E } _ { 0 } N _ { i } .\tag{B.10}
$$

Combining the two bounds on $\mathrm { K L } ( P _ { 0 } , P _ { i } )$ gives $\mathbb { E } _ { 0 } N _ { i } \ge \sigma ^ { 2 } \mathrm { k l } ( \delta , 1 - \delta ) / ( 3 2 \eta ^ { 2 } )$ for each i. Since the supports A<sub>i</sub> are disjoint, $\begin{array} { r } { \sum _ { i = 1 } ^ { M } N _ { i } \leq n . } \end{array}$ , and hence

$$
n \geq \sum _ { i = 1 } ^ { M } \mathbb { E } _ { 0 } N _ { i } \geq \frac { \sigma ^ { 2 } M } { 3 2 \eta ^ { 2 } } \operatorname { k l } ( \delta , 1 - \delta ) \geq c \sigma ^ { 2 } \operatorname { k l } ( \delta , 1 - \delta ) \eta ^ { - 2 - d / \nu } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} ,
$$

where the last step substitutes (B.6). Since the valid procedure was arbitrary, this proves (7). The same argument applies when the procedure is given the common maximizer and the common optimal value before sampling, because this information is the same under $P _ { 0 }$ and every $P _ { i }$ □

## B.2 Proof of Theorem 5.1

In addition to Lemma 5.1, which gives the simultaneous confidence bounds used throughout Section 5, this proof uses the two lemmas presented below. The proofs of these three lemmas are given in Sections C.3–C.5.

Lemma B.2 controls the information gain, which underlies the bounds (B.12), (B.13), (B.22), and (B.23) on the number of evaluations in a phase in Lemmas B.3 and B.5. For $A \subseteq { \mathcal { X } }$ , define the information gain $\gamma _ { n } ( A ; \lambda ) = \operatorname* { s u p } _ { x _ { 1 : n } \in A ^ { n } } { \frac { 1 } { 2 } }$ log det $\left( I + \lambda ^ { - 1 } K _ { n } \right)$ . Conditioning on additional observations can only reduce posterior variance. Consequently, an arbitrary subsequence of n sampled points in A has total variance at most $C _ { \lambda } \gamma _ { n } ( A ; \lambda )$ , where $C _ { \lambda } = 2 / \log ( 1 + \lambda ^ { - 1 } )$

The following lemma bounds information gain on a union of covering balls, without a separation requirement, for any $\nu > 0$

Lemma B.2. For any $\nu > 0$ and any fixed $\bar { r } > 0$ , there is a constant C, depending only on the kernel, $\lambda , { \bar { r } } ,$ and the fixed domain, such that, for $1 \leq n \leq N$ and $0 < r \leq \bar { r }$

$$
\gamma _ { n } ( A ; \lambda ) \leq C \log ^ { 2 } ( 2 N ) \operatorname* { m i n } \Bigl \{ 1 + ( n / \lambda ) ^ { d / ( 2 \nu + d ) } , K + K ^ { 2 \nu / ( 2 \nu + d ) } r ^ { 2 \nu d / ( 2 \nu + d ) } ( n / \lambda ) ^ { d / ( 2 \nu + d ) } \Bigr \}\tag{B.11}
$$

whenever A is covered by K Euclidean radius-r balls.

Lemma B.3 studies a single phase whose predictor uses only observations from that phase with any nonnegative continuous screening function G fixed at the phase start. The continuous maintained bands ℓ, u are obtained by intersections starting from $[ - B , B ]$ , so $W = u - \ell \leq 2 B$

Define $a _ { n } ( x ) = \operatorname* { m i n } \{ G ( x ) , W ( x ) , 2 \beta _ { N } s _ { n } ( x ) + 2 \zeta _ { N } \}$ and $a _ { n } ^ { \star } = \operatorname* { m a x } _ { x \in { \mathcal { X } } } a _ { n } ( x )$ . The phase samples a maximizer of $a _ { n }$ while $a _ { n } ^ { \star } > 2 \varepsilon$ and budget remains, and completes when $a _ { n } ^ { \star } \leq 2 \varepsilon$

This general statement applies both to the screening function in Algorithm 1 and to the screening function for the chosen boundary in Section 5.5.

Lemma B.3. Any point sampled in a phase with $\varepsilon \geq 2 \zeta _ { N }$ satisfies

$$
G ( x _ { j } ) > 2 \varepsilon , W ( x _ { j } ) > 2 \varepsilon , s _ { j - 1 } ( x _ { j } ) > \frac { \varepsilon - \zeta _ { N } } { \beta _ { N } } \geq \frac { \varepsilon } { 2 \beta _ { N } } .
$$

Conditional on the history available at the start of the phase, define the fixed compact set $D = \{ x \in$ ${ \mathcal { X } } : G ( x ) \geq 2 \varepsilon , W ( x ) \geq 2 \varepsilon \}$ . For its number n of observations, including a truncated phase,

$$
n \varepsilon ^ { 2 } \leq 4 \beta _ { N } ^ { 2 } C _ { \lambda } \gamma _ { n } ( D ; \lambda ) .\tag{B.12}
$$

A phase with no evaluations has cost zero and needs no new confidence event. On ${ \mathcal { E } } _ { \delta }$ , a completed phase has maintained width at most 2ε wherever the function G, held fixed throughout the phase,

satisfies $G ( x ) > 2 \varepsilon$ . If D is covered by $J \geq 1$ balls of radius r in a fixed bounded range, then

$$
n \leq C \left[ 1 + J \mathcal { V } _ { N } \varepsilon ^ { - 2 } + J \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } r ^ { d } \varepsilon ^ { - ( 2 + d / \nu ) } \right] .\tag{B.13}
$$

Proof of Theorem 5.1. Throughout the proof, we work on $\mathcal { E } _ { \delta } .$ , which has probability at least $1 - \delta$ by Lemma 5.1. The choice $m = 0$ for $T = 0$ and $m = N$ for $T > 0$ gives the track under analysis all N evaluations. The other track has quota zero and therefore never makes an evaluation. Each of its phases either completes without any evaluation, which leaves the confidence bounds unchanged and hence, by the fixed rule for breaking ties, also the estimated optimal point and the saved regions, or finishes that track by line 6 or line 9 of Algorithm 1. Hence the zero-quota track does not afect the analysis below. We first show that each completed phase satisfies the accuracy bounds in (24). We then bound the number of evaluations in each phase, sum these bounds over the phases, and show by contradiction that the target phase H completes within the budgets (25) and (26).

By Proposition A.2, any completed phase h with $\varepsilon _ { h } \geq 2 \zeta _ { N }$ satisfies $\overline { { f } } _ { h } - \underline { { f } } _ { h } \le 2 \varepsilon _ { h }$ and $f ^ { \star } - f ( \widehat { x } _ { h } ) \leq$ $2 \varepsilon _ { h }$ , and its region outputs satisfy $I _ { h , \tau } \subseteq S _ { \tau } \subseteq O _ { h , \tau }$ and $O _ { h , \tau } \setminus I _ { h , \tau } \subseteq \{ \overline { { x } } : | f ^ { \star } - f ( x ) - \tau | \leq 4 \varepsilon _ { h } \}$ simultaneously for all $0 \leq \tau \leq T$ . Algorithm 3 saves these quantities when phase h completes. For $h = H$ , where $\varepsilon _ { H } = \varepsilon \geq 2 \zeta _ { N }$ , these are (24). It remains to show that phase H completes within the stated budgets.

We next locate the points sampled in a phase. Fix a phase $h \leq H$ , so that $\varepsilon _ { h } \ge \varepsilon \ge 2 \zeta _ { N }$ , and let $G _ { h }$ and $W _ { h - 1 }$ be the screening function and the maintained width in (10), both of which are fixed at the start of phase h. Define $D _ { h } = \{ x \in \mathcal { X } : G _ { h } ( x ) \geq 2 \varepsilon _ { h } , ~ W _ { h - 1 } ( x ) \geq 2 \varepsilon _ { h } \}$ . By Lemma B.3, any point x sampled in phase h satisfies $G _ { h } ( x ) > 2 \varepsilon _ { h }$ and $W _ { h - 1 } ( x ) > 2 \varepsilon _ { h }$ , and hence lies in $D _ { h }$ . Because $D _ { h }$ depends only on quantities fixed at the start of the phase, this also holds when phase h is cut of by the budget. We now show that

$$
D _ { h } \subseteq S _ { ( T + L _ { \vartheta } \varepsilon _ { h } ) + } ,\tag{B.14}
$$

that is, all points of $D _ { h }$ have performance gap at most $T + L _ { \vartheta } \varepsilon _ { h }$ . For $h = 0$ , this holds because $f ^ { \star } - f ( x ) \leq 2 B = \varepsilon _ { 0 }$ for all $x \in \mathcal { X }$ . For $h \geq 1$ , phase $h - 1$ has completed, because a phase starts only after the previous phase of the same track completes. Let $x \in D _ { h }$ . Since $G _ { h } ( x ) \geq 2 \varepsilon _ { h } > 0$ the definition of $G _ { h }$ in (10) gives $u _ { h - 1 } ( x ) \geq \underline { { f } } _ { h - 1 } - T - 2 \varepsilon _ { h } > \underline { { f } } _ { h - 1 } - T - 2 \varepsilon _ { h - 1 }$ . Proposition A.2, applied to the completed phase $h - 1$ , therefore gives $u _ { h - 1 } ( x ) - \ell _ { h - 1 } ( x ) \le 2 \varepsilon _ { h - 1 } = 2 \vartheta \varepsilon _ { h }$ and $\begin{array} { r } { f ^ { \star } - \underline { { f } } _ { h - 1 } \leq \overline { { f } } _ { h - 1 } - \underline { { f } } _ { h - 1 } \leq 2 \vartheta \varepsilon _ { h } } \end{array}$ , where $f ^ { \star } \leq \overline { { f } } _ { h - 1 }$ because $u _ { h - 1 } \geq f$ on $\mathcal { E } _ { \delta }$ . Since also $\ell _ { h - 1 } \leq f$ on ${ \mathcal { E } } _ { \delta }$

$$
f ^ { \star } - f ( x ) \leq ( f ^ { \star } - \underline { { f } } _ { h - 1 } ) + ( \underline { { f } } _ { h - 1 } - u _ { h - 1 } ( x ) ) + ( u _ { h - 1 } ( x ) - \ell _ { h - 1 } ( x ) ) \leq 2 \vartheta \varepsilon _ { h } + ( T + 2 \varepsilon _ { h } ) + 2 \vartheta \varepsilon _ { h } ,
$$

which is at most $T + L _ { \vartheta } \varepsilon _ { h }$ because $L _ { \vartheta } = 4 ( \vartheta + 1 )$ . This proves (B.14).

We now bound the number $n _ { h }$ of evaluations in phase h, including a phase cut of by the budget. By (B.2), $\mathcal { V } _ { N } \le C \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } \le C \Lambda _ { N , \delta }$ . We also use $1 \le \Lambda _ { N , \delta }$ and $\varepsilon _ { h } ^ { - 2 } \leq 1 + \varepsilon _ { h } ^ { - p }$ , which holds because $p > 2$

The global phase bound uses no geometric condition. The set $D _ { h } \subseteq { \mathcal { X } }$ is covered by one ball of radius $D _ { \mathcal { X } }$ , the diameter of X . Applying (B.13) with $J = 1$ and $r \ : = \ : D _ { \mathcal { X } }$ gives $n _ { h } \ \leq$ $C [ 1 + \mathcal { V } _ { N } \varepsilon _ { h } ^ { - 2 } + \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } D _ { \mathcal { X } } ^ { d } \varepsilon _ { h } ^ { - p } ]$ , and thus

$$
n _ { h } \le C \Lambda _ { N , \delta } ( 1 + \varepsilon _ { h } ^ { - p } ) .\tag{B.15}
$$

The localized phase bound uses Assumption 5.4. Suppose that this assumption holds with $\alpha > \nu$ and that $0 \le T \le u _ { 0 } / 2$ , and first consider a phase with ${ T + L _ { \vartheta } \varepsilon _ { h } } \leq u _ { 0 }$ . By (B.14) and (17)

with $u = T + L _ { \vartheta } \varepsilon _ { h }$ , the set $D _ { h }$ is covered by K balls of radius $r _ { h } = c _ { \rho } ( T + L _ { \vartheta } \varepsilon _ { h } ) ^ { 1 / \alpha } \leq c _ { \rho } u _ { 0 } ^ { 1 / \alpha }$ Applying (B.13) with $J = K$ and $r = r _ { h }$ gives $n _ { h } \le C [ 1 + K \mathcal { V } _ { N } \varepsilon _ { h } ^ { - 2 } + K \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } r _ { h } ^ { d } \varepsilon _ { h } ^ { - p } ]$ . Since $L _ { \vartheta } \geq 1$ , we have $r _ { h } ^ { d } \leq c _ { \rho } ^ { d } L _ { \vartheta } ^ { d / \alpha } ( T + \varepsilon _ { h } ) ^ { d / \alpha }$ , and therefore

$$
n _ { h } \le C \Lambda _ { N , \delta } \bigl [ 1 + K \varepsilon _ { h } ^ { - 2 } + K ( T + \varepsilon _ { h } ) ^ { d / \alpha } \varepsilon _ { h } ^ { - p } \bigr ] .\tag{B.16}
$$

Because $T \le u _ { 0 } / 2$ , the condition ${ T + L _ { \vartheta } \varepsilon _ { h } } \leq u _ { 0 }$ holds whenever $\varepsilon _ { h } \leq u _ { 0 } / ( 2 L _ { \vartheta } )$ . The remaining phases have $\varepsilon _ { h } > u _ { 0 } / ( 2 L _ { \vartheta } )$ . There is at most a fixed number of them, and by (B.15) each satisfies $n _ { h } \le C \Lambda _ { N , \delta } \{ 1 + ( 2 L _ { \vartheta } / u _ { 0 } ) ^ { p } \}$ , which is at most the right-hand side of (B.16) after increasing C. Hence (B.16) holds for every phase $h \leq H$

We now sum the phase bounds. Write $\varepsilon ~ = ~ \varepsilon _ { H }$ . Since $\varepsilon _ { h } = \vartheta ^ { H - h } \varepsilon$ for $h \leq H ,$ , we have $\textstyle \sum _ { h = 0 } ^ { H } \varepsilon _ { h } ^ { - b } = \varepsilon ^ { - b } \sum _ { j = 0 } ^ { H } \hat { \vartheta } ^ { - j b } \le \varepsilon ^ { - b } / ( 1 - \vartheta ^ { - b } )$ for any $b > 0$ , and the number of phases satisfies $H + 1 = 1 + \log _ { \vartheta } ( 2 B / \varepsilon ) \leq C ( 1 + \varepsilon ^ { - 2 } )$ . Summing the global phase bound (B.15) over $h \leq H$ , with $b = p ,$ gives

$$
\sum _ { h = 0 } ^ { H } n _ { h } \leq C _ { 1 } \Lambda _ { N , \delta } \big ( 1 + \varepsilon ^ { - p } \big ) .\tag{B.17}
$$

To sum the localized phase bound (B.16), we use $( T + \varepsilon _ { h } ) ^ { d / \alpha } \leq C _ { d , \alpha } ( T ^ { d / \alpha } + \varepsilon _ { h } ^ { d / \alpha } )$ , so that $( T +$ $\varepsilon _ { h } ) ^ { d / \alpha } \varepsilon _ { h } ^ { - p } \leq C _ { d , \alpha } ( T ^ { d / \alpha } \varepsilon _ { h } ^ { - p } + \varepsilon _ { h } ^ { - ( p - d / \alpha ) } )$ , where $p { - } d / \alpha = 2 { + } d / \nu { - } d / \alpha > 2$ because $\alpha > \nu$ . Applying the geometric sum with $b = 2 , b = p ,$ and $b = p - d / \alpha$ , and then using $T ^ { d / \alpha } + \varepsilon ^ { d / \alpha } \leq 2 ( T + \varepsilon ) ^ { d / \alpha }$ gives

$$
\sum _ { h = 0 } ^ { H } n _ { h } \le C _ { 1 } \Lambda _ { N , \delta } \bigl [ 1 + K \varepsilon ^ { - 2 } + K ( T + \varepsilon ) ^ { d / \alpha } \varepsilon ^ { - p } \bigr ] .\tag{B.18}
$$

Here $C _ { 1 }$ denotes a constant for which both (B.17) and (B.18) hold, and both bounds also apply at $T = 0$ . Only the compactness of X is used in these upper bounds. The right-hand side of (B.17) is the expression in (25) with $C _ { 1 }$ in place of C. Under Assumption 5.4 with $\alpha > \nu$ and $0 \le T \le u _ { 0 } / 2$ taking the smaller of (B.17) and (B.18) gives the expression in (26) with $C _ { 1 }$ in place of C.

Finally, we show that phase H completes within the suficient budgets (25) and (26) of Theorem 5.1, whose constant $C$ we take larger than $C _ { 1 }$ . On ${ \mathcal { E } } _ { \delta }$ , the safeguard for inconsistent bounds in Algorithm 1 is never triggered, and the refinement stop of Algorithm 1, which applies only once the accuracy level falls below $2 \zeta _ { N }$ , does not afect phases with $\varepsilon _ { h } \geq 2 \zeta _ { N }$ . Hence phase H can fail to complete only if the budget is exhausted first. Suppose that N is at least the expression in (25), or at least the expression in (26) under its assumptions, but phase H does not complete. Since phases are run in order, the track then uses all N evaluations in phases $h \leq H$ , and each of these phases, including the one cut of by the budget, satisfies (B.15) and, under the assumptions of (26), (B.16). Thus $\textstyle N = \sum _ { h = 0 } ^ { H } n _ { h }$ , where $n _ { h } = 0$ for phases that are not reached. By (B.17) and (B.18), N is at most the corresponding expression with $C _ { 1 }$ in place of C. Since $C > C _ { 1 }$ , this contradicts the assumption on N. Therefore phase H completes.

Therefore, on $\mathcal { E } _ { \delta } .$ , phase H completes whenever N is at least the suficient budget, and the output saved at its completion satisfies (24). Since ${ \mathcal { E } } _ { \delta }$ does not depend on H, these conclusions hold simultaneously for all targets $\varepsilon = 2 B \vartheta ^ { - H } \geq 2 \zeta _ { N }$ , which proves the theorem. □

## B.3 Proof of Corollary 5.1

Proof of Corollary 5.1. Recall from Section B.1 that $T _ { \mathrm { r e g } } \leq \mathrm { m i n } \{ 1 , u _ { 0 } / 2 , 2 0 4 8 B \}$ . Hence $T \leq 1 { : }$ $T \leq u _ { 0 } / 2$ , so that the localized bound (26) applies; and $\eta \leq T / 1 2 8 \leq 1 6 B$ , so that $\eta / 8 \le 2 B = \varepsilon _ { 0 }$

and there is a phase target $\varepsilon = \varepsilon _ { H }$ with $\eta / ( 8 \vartheta ) < \varepsilon \leq \eta / 8 .$ In the stated range, $T + \varepsilon \asymp T .$ . Since $\varepsilon \leq \eta / 8 \leq T \leq 1$ and $\alpha > \nu _ { ; }$ , we have $\varepsilon ^ { 1 / \nu } \leq \varepsilon ^ { 1 / \alpha } \leq T ^ { 1 / \alpha }$ , and hence $K \varepsilon ^ { - 2 } \leq K T ^ { d / \alpha } \varepsilon ^ { - ( 2 + d / \nu ) }$ Consequently, the suficient evaluation count in Theorem 5.1 is at most

$$
C F ( \eta , T , K ) \Lambda _ { N , \delta } , F ( \eta , T , K ) = \eta ^ { - ( 2 + d / \nu ) } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} .
$$

Since $T \geq 1 2 8 \eta , K \geq K _ { \mathrm { r e g } } \geq 1$ , and $\eta \leq T / 1 2 8 < 1$ , we have

$$
c \eta ^ { - ( 2 + d / \nu - d / \alpha ) } \leq F ( \eta , T , K ) \leq \eta ^ { - ( 2 + d / \nu ) } .\tag{B.19}
$$

Choose the declared horizon

$$
N = \left\lceil A F ( \eta , T , K ) \left[ \log \frac { 2 } { \eta \delta } \right] ^ { 3 + 3 d / ( 2 \nu ) } \right\rceil\tag{B.20}
$$

for a suficiently large fixed constant A, which may depend on ϑ. By (B.19) and (B.20), we have

$$
\log ( 2 N ) \leq \log ( 2 N / \delta ) \leq C ( 1 + \log A ) \log \frac { 2 } { \eta \delta } .
$$

The suficient evaluation count is therefore at most

$$
C ( 1 + \log A ) ^ { 3 + 3 d / ( 2 \nu ) } F ( \eta , T , K ) \left[ \log \frac { 2 } { \eta \delta } \right] ^ { 3 + 3 d / ( 2 \nu ) } .
$$

Choose A so that $C ( 1 + \log A ) ^ { 3 + 3 d / ( 2 \nu ) } \leq A$ . This makes the suficient count at most $N .$ , uniformly in $K , T , \eta , \delta$

The confidence parameters in (9) satisfy $\zeta _ { N } \le C N ^ { - 7 / 2 } \sqrt { \log ( 2 N / \delta ) }$ . Hence

$$
\zeta _ { N } \leq C A ^ { - 7 / 2 } ( 1 + \log A ) ^ { 1 / 2 } \eta ^ { \frac 7 2 ( 2 + d / \nu - d / \alpha ) } \left[ \log \frac 2 { \eta \delta } \right] ^ { \frac 1 2 - \frac 7 2 ( 3 + 3 d / ( 2 \nu ) ) } \leq \frac \eta { 1 6 \operatorname* { m a x } \{ 2 , \vartheta \} } .\tag{B.21}
$$

The last inequality in (B.21) holds after increasing the fixed A, because $2 + d / \nu - d / \alpha > 2 , \eta < 1$ and the exponent of the logarithm is negative. Thus $2 \zeta _ { N } \leq \varepsilon ,$ and phase completion gives the accuracy (24), with region estimation error at most $4 \varepsilon \leq \eta$

The ceiling in N is absorbed into the constant C in (27) because $F ( \eta , T , K )$ is bounded below by a positive constant in the stated range. All constants may depend on the fixed phase ratio $\vartheta ,$ but are independent of $\delta , K , T , \eta$

## B.4 Proof of Theorem 5.2

We first present two lemmas used in this proof, whose proofs are given in Sections C.6 and C.7. Lemma B.4 bounds the band widths after a completed phase and the optimality gaps of the designs sampled in the following phase.

Lemma B.4. On ${ \mathcal { E } } _ { \delta }$ , after phase h completes,

$$
\begin{array} { r } { W _ { h } ( x ) \leq 2 \varepsilon _ { h } \ i f u _ { h } ( x ) > \underline { { f } } _ { h } - T - 2 \varepsilon _ { h } , \overline { { f } } _ { h } - \underline { { f } } _ { h } \leq 2 \varepsilon _ { h } , f ^ { \star } - f ( \widehat { x } _ { h } ) \leq 2 \varepsilon _ { h } . } \end{array}
$$

Any point sampled in phase h has gap at most $T + L _ { \vartheta } \varepsilon _ { h }$ and at most 2B.

For the $T = 0$ setting of Algorithm 1, Lemma B.5 converts the information-gain bound (B.11) into a bound on the number of evaluations in each phase.

Lemma B.5. Under Assumptions 5.2–5.3, consider Algorithm 1 with $T = 0 , N \geq 2$ , and $\delta \in ( 0 , 1 )$ Let $n _ { h }$ count observations collected by the sampling loop in phase h, including repetitions and an unfinished phase truncated at the horizon. $I f \varepsilon _ { h } \geq 2 \zeta _ { N }$ , then deterministically

$$
n _ { h } \le C \Lambda _ { N , \delta } \varepsilon _ { h } ^ { - p } .\tag{B.22}
$$

For the localized bound, work on ${ \mathcal { E } } _ { \delta }$ , defined in (13), with $\mathbb { P } ( \mathcal { E } _ { \delta } ) \geq 1 - \delta$ by Lemma 5.1. On $\mathcal { E } _ { \delta }$ , under Assumption $5 . 4$ and $L _ { \vartheta } \varepsilon _ { h } \leq u _ { 0 }$ 2

$$
\begin{array} { r } { n _ { h } \leq C \Lambda _ { N , \delta } K \varepsilon _ { h } ^ { - a } . } \end{array}\tag{B.23}
$$

The constants are independent of $K , N , \delta .$

Proof of Theorem 5.2. Fix $\vartheta > 1$ . All upper-bound constants in this proof will depend on $\vartheta .$ The region track has quota $m = 0$ and, as noted in the proof of Theorem 5.1, does not afect the analysis. For $\delta = N ^ { - 2 }$ , the factor $\Lambda _ { N , \delta }$ in the phase bounds of Lemma B.5 is at most $C L _ { N }$ by (B.2).

If $\zeta _ { N } > B$ , the deterministic bounds $r _ { N } \leq 2 B$ and $R _ { N } \le 2 B N$ are absorbed by the $\zeta _ { N }$ term. Hence assume $\zeta _ { N } \leq B ,$ , and then any nonempty phase, that is, any phase in which the sampling loop of Algorithm 2 makes at least one evaluation, has $\varepsilon _ { h } \geq 2 \zeta _ { N }$ , because Algorithm 1 allocates all remaining evaluations to the saved estimated optimal point once the accuracy level falls below $2 \zeta _ { N }$ Work on ${ \mathcal { E } } _ { N ^ { - 2 } }$ . By Lemma B.5, all nonempty phases, including truncated ones, satisfy

$$
\begin{array} { r } { n _ { h } \le C L _ { N } \varepsilon _ { h } ^ { - p } \mathrm { ~ f o r ~ a l l ~ } h , n _ { h } \le C L _ { N } K \varepsilon _ { h } ^ { - a } \mathrm { ~ i f ~ } L _ { \vartheta } \varepsilon _ { h } \le u _ { 0 } , } \end{array}\tag{B.24}
$$

and by Lemma B.4 with $T = 0$ , all points sampled in phase h have gap at most $L _ { \vartheta } \varepsilon _ { h }$

The finitely many coarse scales with $L _ { \vartheta } \varepsilon _ { h } > u _ { 0 }$ have $\varepsilon _ { h } \ge u _ { 0 } / L _ { \vartheta }$ fixed, so (B.24) bounds their total weighted cost by $\begin{array} { r } { \sum _ { h : L _ { \vartheta } \varepsilon _ { h } > u _ { 0 } } L _ { \vartheta } \varepsilon _ { h } n _ { h } \leq C L _ { N } } \end{array}$

Split the nonempty phases at any scale $\varepsilon \in [ 2 \zeta _ { N } , 2 B ]$ . We use the following three facts, where $\begin{array} { r } { R _ { N } = \sum _ { t = 1 } ^ { N } \{ f ^ { \star } - f ( x _ { t } ) \} } \end{array}$

• Gap of each sampled point. If $x _ { t }$ is sampled in phase $h \geq 1$ , then $f ^ { \star } - f ( x _ { t } ) \leq L _ { \vartheta } \varepsilon _ { h } .$ , by Lemma B.4 with $T = 0$ . The reason is that $x _ { t }$ is sampled only if its acquisition value exceeds $2 \varepsilon _ { h }$ , which implies $u _ { h - 1 } ( x _ { t } ) > \underline { { f } } _ { h - 1 } - 2 \varepsilon _ { h }$ . Since phase $h - 1$ has completed, Lemma B.4 also gives $u _ { h - 1 } ( x _ { t } ) - f ( x _ { t } ) \leq W _ { h - 1 } ( x _ { t } ) \leq 2 \varepsilon _ { h - 1 } = 2 \vartheta \varepsilon _ { h }$ and $f ^ { \star } - \underline { { f } } _ { h - 1 } \leq 2 \vartheta \varepsilon _ { h }$ . Hence

$$
\begin{array} { r } { f ^ { \star } - f ( x _ { t } ) = \big ( f ^ { \star } - \underline { { f } } _ { h - 1 } \big ) + \big ( \underline { { f } } _ { h - 1 } - u _ { h - 1 } ( x _ { t } ) \big ) + ( u _ { h - 1 } ( x _ { t } ) - f ( x _ { t } ) ) < ( 4 \vartheta + 2 ) \varepsilon _ { h } \leq L _ { \vartheta } \varepsilon _ { h } . } \end{array}
$$

In phase 0, $f ^ { \star } - f ( x _ { t } ) \leq 2 B = \varepsilon _ { 0 } \leq L _ { \vartheta } \varepsilon _ { 0 }$

• Fine phases. The phases with $\varepsilon _ { h } < \varepsilon$ contain at most N evaluations in total. By the previous fact, each of these evaluations has gap at most $L _ { \vartheta } \varepsilon _ { h } < L _ { \vartheta } \varepsilon$ , so together they contribute at most $\cal { L } _ { \vartheta } N \varepsilon$ to $R _ { N }$

• Evaluations at the saved estimated optimal point. Suppose that a phase with accuracy level below $2 \zeta _ { N }$ is reached, and let h<sup>′</sup> be the first such phase. Since $\varepsilon _ { 0 } = 2 B \ge 2 \zeta _ { N }$ , we have $h ^ { \prime } \geq 1$ and phase $h ^ { \prime } - 1$ has completed with $\varepsilon _ { h ^ { \prime } - 1 } = \vartheta \varepsilon _ { h ^ { \prime } } < 2 \vartheta \zeta _ { N }$ . By Lemma $\mathrm { B . 4 }$ , the saved estimated optimal point $\widehat { x } _ { h ^ { \prime } - 1 }$ satisfies $f ^ { \star } - f ( \widehat { x } _ { h ^ { \prime } - 1 } ) \leq 2 \varepsilon _ { h ^ { \prime } - 1 } < 4 \vartheta \zeta _ { N }$ . All remaining evaluations are made at this point, and there are at most N of them, so they contribute at most $4 \vartheta N \zeta _ { N }$ to $R _ { N }$

On ${ \mathcal { E } } _ { N ^ { - 2 } }$ , the safeguard for inconsistent bounds is never triggered. Hence all evaluations are made in a phase with $L _ { \vartheta } \varepsilon _ { h } > u _ { 0 }$ , in a phase with $\varepsilon \leq \varepsilon _ { h } \leq u _ { 0 } / L _ { \vartheta }$ , in a phase with $\varepsilon _ { h } < \varepsilon .$ , or at the saved estimated optimal point after the accuracy level falls below $2 \zeta _ { N }$ . The first group contributes at most $C L _ { N }$ by the preceding paragraph, the second group contributes at most $\begin{array} { r } { L _ { \vartheta } \sum _ { h : \varepsilon \leq \varepsilon _ { h } \leq u _ { 0 } / L _ { \vartheta } } \varepsilon _ { h } n _ { h } } \end{array}$ by the first fact, and the last two groups are bounded by the last two facts. Together they give

$$
R _ { N } \leq C L _ { N } + L _ { \vartheta } \sum _ { h : \varepsilon \leq \varepsilon _ { h } \leq u _ { 0 } / L _ { \vartheta } } \varepsilon _ { h } n _ { h } + L _ { \vartheta } N \varepsilon + 4 \vartheta N \zeta _ { N } .
$$

For any $b > 1$

$$
\sum _ { h : \varepsilon _ { h } \geq \varepsilon } \varepsilon _ { h } ^ { 1 - b } \leq \frac { \varepsilon ^ { 1 - b } } { 1 - \vartheta ^ { 1 - b } } .\tag{B.25}
$$

Inserting (B.24) and applying (B.25) with $b = p$ and $b = a$ gives

$$
\sum _ { \substack { h \colon \varepsilon \le \varepsilon _ { h } \le u _ { 0 } / L \vartheta } } \varepsilon _ { h } n _ { h } \le C L _ { N } \operatorname* { m i n } \Bigl \{ \sum _ { h : \varepsilon _ { h } \ge \varepsilon } \varepsilon _ { h } ^ { 1 - p } , \ K \sum _ { h : \varepsilon _ { h } \ge \varepsilon } \varepsilon _ { h } ^ { 1 - a } \Bigr \} \le C L _ { N } \operatorname* { m i n } \bigl \{ \varepsilon ^ { - 1 - d / \nu } , K \varepsilon ^ { 1 - a } \bigr \} ,
$$

so that

$$
R _ { N } \leq C L _ { N } + C L _ { N } \operatorname* { m i n } \{ K \varepsilon ^ { 1 - a } , \varepsilon ^ { - 1 - d / \nu } \} + C N \varepsilon + C N \zeta _ { N } .\tag{B.26}
$$

The minimum in (B.26) has a localized term $K \varepsilon ^ { 1 - a }$ and a global term $\varepsilon ^ { - 1 - d / \nu } = \varepsilon ^ { 1 }$ <sup>−p</sup>. We balance each of these two terms against the term Nε separately, which leads to the choices $\varepsilon _ { \mathrm { l o c } } = ( L _ { N } K / N ) ^ { 1 / a }$ and $\varepsilon _ { \mathrm { g l o b } } = ( L _ { N } / N ) ^ { 1 / p }$ , each clipped to $[ 2 \zeta _ { N } , 2 B ]$ . When no clipping is active, these choices give

$$
L _ { N } K \varepsilon _ { \mathrm { l o c } } ^ { 1 - a } = N \varepsilon _ { \mathrm { l o c } } = L _ { N } ^ { 1 / a } K ^ { 1 / a } N ^ { 1 - 1 / a } , L _ { N } \varepsilon _ { \mathrm { g l o b } } ^ { 1 - p } = N \varepsilon _ { \mathrm { g l o b } } = L _ { N } ^ { 1 / p } N ^ { 1 - 1 / p } ,
$$

and since $a \geq 2$ and $p > 2 .$ , the terms $L _ { N } ^ { 1 / a }$ and $L _ { N } ^ { 1 / p }$ are at most $L _ { N }$

If the clipping at the lower endpoint $2 \zeta _ { N }$ is active for one of these choices, say $( L _ { N } K / N ) ^ { 1 / a } < 2 \zeta _ { N } ,$ then $L _ { N } K < N ( 2 \zeta _ { N } ) ^ { a }$ , so the second and third terms of (B.26) evaluated at $\varepsilon = 2 \zeta _ { N }$ are $O ( N \zeta _ { N } )$ (similarly $L _ { N } < N ( 2 \zeta _ { N } ) ^ { p }$ when the global term is smaller); if the clipping at the upper endpoint 2B is active, use $R _ { N } \le 2 B N$ . In all cases,

$$
R _ { N } \leq C L _ { N } \big [ 1 + N \operatorname* { m i n } \{ ( K / N ) ^ { 1 / a } , N ^ { - 1 / p } \} \big ] + C N \zeta _ { N } = C L _ { N } [ 1 + N \rho _ { N } ] + C N \zeta _ { N }
$$

on ${ \mathcal { E } } _ { N - 2 }$

Next, we consider simple regret. If the stopping condition $\varepsilon _ { h } < 2 \zeta _ { N }$ has not been met, let h be the last nonempty phase. Phase $h - 1$ has completed (possibly without observations) unless $h = 0$ so the saved estimated optimal point has gap at most $2 \varepsilon _ { h - 1 } = 2 \vartheta \varepsilon _ { h }$ by Lemma B.4. If phase h itself has completed, the bound is smaller, and if no phase completed, $r _ { N } \le 2 B = \varepsilon _ { 0 }$ . Additional empty phases after budget exhaustion leave the bands and the estimated optimal point unchanged.

Since h is the last nonempty phase, all N evaluations are made in phases $0 , 1 , \ldots , h .$ , that is, $\begin{array} { r } { N = \sum _ { j = 0 } ^ { h } n _ { j } } \end{array}$ . By (B.24) and geometric summation, $N \leq C L _ { N } \varepsilon _ { h } ^ { - p }$ in the global analysis and $N \leq C L _ { N } \big [ 1 + K \varepsilon _ { h } ^ { - a } \big ]$ in the localized analysis. In the latter bound, the additive constant 1 accounts for the finitely many phases with $L _ { \vartheta } \varepsilon _ { j } > u _ { 0 }$ , to which the second bound in (B.24) does not apply; each of these phases satisfies $n _ { j } \le \dot { C L } _ { N } \varepsilon _ { j } ^ { - p } \le C L _ { N } ( u _ { 0 } / L _ { \vartheta } ) ^ { - p }$ by the first bound in (B.24). For $N \geq 2 C L _ { N }$ , we then have

$$
\varepsilon _ { h } \le ( C L _ { N } / N ) ^ { 1 / p } , \varepsilon _ { h } \le ( 2 C L _ { N } K / N ) ^ { 1 / a } , r _ { N } \le 2 \vartheta \varepsilon _ { h } \le C L _ { N } \rho _ { N } ,
$$

again bounding the logarithmic terms by $L _ { N }$ . For $N < 2 C L _ { N }$ the bound $r _ { N } \le 2 B \le C L _ { N } \rho _ { N }$ holds because $\rho _ { N } \ge N ^ { - 1 / 2 }$

Once $\varepsilon _ { h } < 2 \zeta _ { N }$ , the estimated optimal point has gap at most $4 \vartheta \zeta _ { N }$ , by the third fact above.

Finally, $\mathcal { E } _ { N ^ { - 2 } } ^ { c }$ has probability at most $N ^ { - 2 }$ and adds at most $2 B N \cdot N ^ { - 2 } = 2 B / N$ to expected cumulative regret and $2 B \cdot N ^ { - 2 }$ to expected simple regret, which gives (31). The global rates (30) follow from the global bound and $\zeta _ { N } \overset { \cdot } { = } \tilde { O } ( N ^ { - 7 / 2 } )$ . This finishes the proof. □

## B.5 Proof of Theorem 5.3

Proof of Theorem 5.3. Choose $c \in \mathcal X$ and $r _ { * } > 0$ with $\overline { { B } } ( c , 2 r _ { * } ) \subset \mathrm { i n t } ( \mathcal { X } )$ . By Lemma B.1, $C _ { g } =$ $\| \| \cdot - c \| ^ { \alpha } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } < \infty$ . Set $\mu _ { 0 } = ( 4 / c _ { \rho } ) ^ { \alpha } , \mu _ { K } = \mu _ { 0 } K ^ { - \alpha / d } , g _ { K } ( x ) = - \mu _ { K } \| x - c \| ^ { \alpha }$ . Take

$$
K _ { \mathrm { o p t } } = \operatorname* { m a x } \left\{ 1 , \left\lceil ( 2 \mu _ { 0 } C _ { g } / B ) ^ { d / \alpha } \right\rceil \right\} .
$$

Then ∥g<sub>K</sub> $\| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B / 2$ for $K \geq K _ { \mathrm { o p t } }$ . Note that this norm decreases with $K .$

We will use the following elementary covering fact. A ball of radius $R _ { u } ~ = ~ ( u / \mu _ { K } ) ^ { 1 / \alpha } ~ =$ $( c _ { \rho } / 4 ) K ^ { 1 / d } u ^ { 1 / \alpha }$ is coverable by at most K balls of radius $r _ { u } = c _ { \rho } u ^ { 1 / \alpha }$ . If $K < 2 ^ { d }$ , then $R _ { u } < r _ { u } / 2$ and one ball sufices. If $K \geq 2 ^ { d }$ , a maximal $r _ { u }$ -separated packing provides a cover with at most $( 1 + 2 R _ { u } / r _ { u } ) ^ { d } = \left( 1 + { \textstyle \frac { 1 } { 2 } } K ^ { 1 / d } \right) ^ { d } \le K$ balls.

Let $\chi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ be a function taking values in $[ 0 , 1 ]$ , with $\chi ( v ) = 1$ for $\| v \| \leq 1 / 2$ and $\chi ( v ) = 0$ for $\lVert \boldsymbol { v } \rVert \geq 1$ . For a small height parameter $\Delta > 0$ , put $R = \operatorname* { m i n } \{ r _ { * } , ( \Delta / \mu _ { K } ) ^ { 1 / \alpha } \} , h = ( 2 C _ { * } \Delta / B ) ^ { 1 / \nu } .$ where $C _ { * }$ is a fixed RKHS norm bound specified below. For suficiently small $\Delta$ , uniformly over $K \geq 1$

$$
h \leq R / 3 2 , h \leq 1 , h \Delta ^ { - 1 / \alpha } \leq c _ { \rho } .\tag{B.27}
$$

Indeed $\alpha > \nu ,$ and $R \ge \operatorname* { m i n } \{ r _ { * } , ( \Delta / \mu _ { 0 } ) ^ { 1 / \alpha } \}$

Take a maximal 2h-separated set of centers $z _ { i }$ in the set $\{ x : R / 2 \leq \| x - c \| \leq 3 R / 4 \}$ . The open balls $A _ { i } = B ( z _ { i } , h )$ are disjoint and lie in $B ( c , R )$ . By maximality, the balls $B ( z _ { i } , 2 h )$ cover this set, so a volume comparison gives

$$
M \ge c _ { d } ( R / h ) ^ { d } \ge c _ { M } \operatorname* { m i n } \{ \Delta ^ { - d / \nu } , K \Delta ^ { - d / \nu + d / \alpha } \} .\tag{B.28}
$$

The constant $c _ { M } > 0$ is independent of $K , \Delta$ . In particular, the volume comparison gives (B.28) with $c _ { d } = ( 3 / 8 ) ^ { d } - ( 1 / 4 ) ^ { d }$ , so $h \leq R / 3 2$ makes the first lower bound in (B.28) at least $1 2 ^ { d } - 8 ^ { d } \geq 4$ so $M \geq 4$

For $v = ( x - z _ { i } ) / h$ , define

$$
f _ { i } ( x ) = [ 1 - \chi ( v ) ] g _ { K } ( x ) + \chi ( v ) [ 4 \Delta - \Delta \| v \| ^ { \alpha } ] .\tag{B.29}
$$

Outside $A _ { i }$ this equals $g _ { K }$ , and for $\| v \| \leq 1 / 2$ it equals $4 \Delta - \Delta \| v \| ^ { \alpha }$

The modification is

$$
f _ { i } - g _ { K } = \Delta \chi ( v ) \left[ 4 - \| v \| ^ { \alpha } + \frac { \mu _ { K } } { \Delta } \| z _ { i } - c + h v \| ^ { \alpha } \right] .
$$

The first two cutof terms have finite $H ^ { \nu + d / 2 }$ norm by Lemma B.1. For the last term write $z _ { i } - c = R a _ { i }$ 2 $\bar { h } = h / R$ , and $\bar { \mu } _ { K } = \mu _ { K } R ^ { \alpha } / \Delta \leq 1$ . The normalized term is

$$
\chi ( v ) \bar { \mu } _ { K } \| a _ { i } + \bar { h } v \| ^ { \alpha } , 1 / 2 \leq \| a _ { i } \| \leq 3 / 4 , 0 < \bar { h } \leq 1 / 8 .
$$

On the support of $\chi .$ , we have $\| v \| \leq 1$ , and hence $3 / 8 \le \| a _ { i } \| - { \bar { h } } \le \| a _ { i } + { \bar { h } } v \| \le \| a _ { i } \| + { \bar { h } } \le 7 / 8$ Since the Euclidean norm is smooth away from the origin, the map $v \mapsto \| a _ { i } + { \bar { h } } v \| ^ { \alpha }$ is smooth on this support. All derivatives of this smooth compactly supported family are uniformly bounded, and hence its $H ^ { \nu + d / 2 }$ norms are uniformly bounded. Fourier scaling and restriction therefore give a fixed $C _ { * }$ such that $\| f _ { i } - g _ { K } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \le C _ { * } \Delta h ^ { - \nu } = B / 2$ . Thus any $f _ { i }$ lies in the same ball of radius $B _ { i }$ independently of $K , N , \Delta$

Since $g _ { K } \leq 0$ , the maximum of $f _ { i }$ is exactly $f _ { i } ^ { \star } = 4 \Delta$ , achieved only at $z _ { i }$ . For $x \in A _ { i }$ , the gap satisfies

$$
\big ( f _ { i } ^ { \star } - f _ { i } ( x ) \big ) = [ 1 - \chi ( v ) ] \big ( 4 \Delta + \mu _ { K } \| x - c \| ^ { \alpha } \big ) + \chi ( v ) \Delta \| v \| ^ { \alpha } \geq \Delta \| v \| ^ { \alpha } .
$$

Outside A<sub>i</sub>, $\big ( f _ { i } ^ { \star } - f _ { i } ( x ) \big ) = 4 \Delta + \mu _ { K } \| x - c \| ^ { \alpha }$ . For $0 < u < \Delta$ it follows that

$$
S _ { u + } ( f _ { i } ) \subseteq \overline { { B } } ( z _ { i } , h ( u / \Delta ) ^ { 1 / \alpha } ) \subseteq \overline { { B } } ( z _ { i } , c _ { \rho } u ^ { 1 / \alpha } ) ,
$$

by (B.27), and thus one ball sufices.

For $u \geq \Delta , \mathrm { { \varepsilon } }$ a point of $S _ { u + } ( f _ { i } )$ outside $A _ { i }$ lies in the ball $B ( c , ( u / \mu _ { K } ) ^ { 1 / \alpha } )$ . Points inside $A _ { i }$ also lie in the same ball, because $A _ { i } \subset B ( c , R )$ and $R \leq ( \Delta / \mu _ { K } ) ^ { 1 / \alpha } \leq ( u / \mu _ { K } ) ^ { 1 / \alpha }$ . The elementary covering fact provides at most K balls of the required radius. Therefore any $f _ { i }$ satisfies the prescribed cover condition for any $u > 0$ , in particular for $0 < u \leq u _ { 0 }$

On $A _ { i } \subset B ( c , R )$ the baseline satisfies $| g _ { K } ( x ) | = \mu _ { K } \| x - c \| ^ { \alpha } \leq \mu _ { K } R ^ { \alpha } \leq \Delta$ , so the modification is supported on $A _ { i }$ and

$$
0 \leq f _ { i } ( x ) - g _ { K } ( x ) = \chi ( v ) \big [ 4 \Delta - \Delta \| v \| ^ { \alpha } - g _ { K } ( x ) \big ] \leq 4 \Delta + \Delta = 5 \Delta \mathrm { ~ f o r ~ } x \in \mathcal { X } .
$$

Let $P _ { 0 }$ denote the joint distribution of the sampling history and final output, including the procedure’s randomization, under $g _ { K }$ , and let $P _ { i }$ denote the corresponding distribution under $f _ { i }$

Let $\begin{array} { r } { N _ { i } = \sum _ { t < N } \mathbf { 1 } \{ x _ { t } \in A _ { i } \} } \end{array}$ be the number of evaluations allocated to $A _ { i }$ . Similar to the proof of Theorem 3.1 in Section B.1, the decision rules of the procedure are the same under $g _ { K }$ and under $f _ { i }$ and therefore cancel in the likelihood ratio. Hence, factoring the conditional Gaussian densities over the N evaluations and using $0 \le f _ { i } - g _ { K } \le 5 \Delta \mathbf { 1 } _ { A _ { i } }$ gives

$$
\operatorname { K L } ( P _ { 0 } , P _ { i } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \mathbb { E } _ { 0 } \sum _ { t = 1 } ^ { N } \{ f _ { i } ( x _ { t } ) - g _ { K } ( x _ { t } ) \} ^ { 2 } \leq \frac { 2 5 \Delta ^ { 2 } } { 2 \sigma ^ { 2 } } \mathbb { E } _ { 0 } N _ { i } .\tag{B.30}
$$

Let $\mathrm { T V } ( P _ { 0 } , P _ { i } ) = \operatorname* { s u p } _ { E } \left| P _ { 0 } ( E ) - P _ { i } ( E ) \right|$ denote the total variation distance, where the supremum is over events $E$ determined by the sampling history and the final output. For any random variable $Z$ that is a function of the sampling history and the final output and takes values in [0, 1], Pinsker’s inequality and (B.30) give

$$
\left| \mathbb { E } _ { i } Z - \mathbb { E } _ { 0 } Z \right| \le \mathrm { T V } ( P _ { 0 } , P _ { i } ) \le \sqrt { \frac { 1 } { 2 } \operatorname { K L } ( P _ { 0 } , P _ { i } ) } \le \frac { 5 \Delta } { 2 \sigma } \sqrt { \mathbb { E } _ { 0 } N _ { i } } .\tag{B.31}
$$

Applying (B.31) with $Z = \mathbf { 1 } \{ \widehat { x } _ { N } \in A _ { i } \}$ , and using the disjointness of the $A _ { i }$ (which gives $\textstyle \sum _ { i } P _ { 0 } ( { \widehat { x } } _ { N } \in$

$A _ { i } ) \leq 1$ and $\begin{array} { r } { \sum _ { i } \mathbb { E } _ { 0 } N _ { i } \le N ) } \end{array}$ and the Cauchy–Schwarz inequality, we obtain

$$
\begin{array} { r l r } & { } & { \displaystyle { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } P _ { i } ( \widehat { x } _ { N } \in A _ { i } ) \leq \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \Bigl [ P _ { 0 } ( \widehat { x } _ { N } \in A _ { i } ) + \sqrt { \frac { 1 } { 2 } \operatorname { K L } ( P _ { 0 } , P _ { i } ) } \Bigr ] } } \\ & { } & { \displaystyle { \leq \frac { 1 } { M } + \frac { 5 \Delta } { 2 \sigma } \sqrt { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { 0 } N _ { i } } \leq \frac { 1 } { M } + \frac { 5 \Delta } { 2 \sigma } \sqrt { \frac { N } { M } } . } } \end{array}\tag{B.32}
$$

The same upper bound holds for the average fraction of evaluations allocated to the correct support. Indeed, since $0 \le N _ { i } \le N$ , the statistic $Z = N _ { i } / N$ takes values in $[ 0 , 1 ]$ , and (B.31) gives $\mathbb { E } _ { i } N _ { i } / N \le \mathbb { E } _ { 0 } N _ { i } / N + ( 5 \Delta / ( 2 \sigma ) ) \sqrt { \mathbb { E } _ { 0 } N _ { i } }$ . Averaging over i and using $\begin{array} { r } { \sum _ { i } \mathbb { E } _ { 0 } N _ { i } \le N } \end{array}$ and the Cauchy–Schwarz inequality as in (B.32), we obtain

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { \mathbb { E } _ { i } N _ { i } } { N } \leq \frac { 1 } { M } + \frac { 5 \Delta } { 2 \sigma } \sqrt { \frac { N } { M } } .\tag{B.33}
$$

Set $\Delta = c _ { * } \operatorname* { m i n } \{ ( K / N ) ^ { 1 / a } , N ^ { - 1 / ( 2 + d / \nu ) } \}$ . By (B.28),

$$
\frac { N \Delta ^ { 2 } } { M } \leq c _ { M } ^ { - 1 } \operatorname* { m a x } \{ N \Delta ^ { 2 + d / \nu } , ( N / K ) \Delta ^ { a } \} \leq c _ { M } ^ { - 1 } \operatorname* { m a x } \{ c _ { * } ^ { 2 + d / \nu } , c _ { * } ^ { a } \} .\tag{B.34}
$$

Choose $c _ { * } > 0$ small enough that the second term on the right-hand sides of (B.32) and (B.33), namely $( 5 \Delta / ( 2 \sigma ) ) \sqrt { N / M } = ( 5 / ( 2 \sigma ) ) \sqrt { N \Delta ^ { 2 } / M }$ , is at most $1 / 4 ;$ this is possible by (B.34). Since $\Delta \le c _ { * } \dot { N } ^ { - 1 / ( 2 + d / \nu ) }$ and the conditions in (B.27) hold for all suficiently small $\Delta$ uniformly over $K \geq 1$ , there is an $N _ { \mathrm { o p t } }$ , independent of $K$ , such that (B.27) holds for all $N \geq N _ { \mathrm { o p t } }$ . For such $N _ { ; }$ we have $M \geq 4 .$ , as shown after (B.28), so $1 / M \le 1 / 2$ . Hence, by (B.32) and (B.33), both the average success probability M<sup>−</sup> $\textstyle { \mathrm { ~ \widehat { ~ } 1 ~ } } \sum _ { i = 1 } ^ { M } P _ { i } ( \widehat { x } _ { N } \in \dot { A } _ { i } )$ and the average fraction $\dot { M } ^ { - 1 } \dot { \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } } N _ { i } / N$ are at most $1 / 2 + 1 / 4 = 3 / 4$ . Outside $A _ { i }$ the gap under $f _ { i }$ is $f _ { i } ^ { \star } - f _ { i } ( x ) = 4 \Delta - g _ { K } ( x ) \geq 4 \Delta$ , hence

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } r _ { N } \geq 4 \Delta \bigg ( 1 - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } P _ { i } ( \widehat { x } _ { N } \in A _ { i } ) \bigg ) \geq \Delta , \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } R _ { N } \geq 4 \Delta \bigg ( N - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } N _ { i } \bigg ) \geq N \Delta .
$$

Taking the supremum separately gives $\begin{array} { r } { \operatorname* { s u p } _ { f \in \mathcal { F } _ { K } } \mathbb { E } _ { f } r _ { N } \ge \Delta = c _ { * } \rho _ { N } } \end{array}$ and $\begin{array} { r } { \operatorname* { s u p } _ { f \in \mathcal { F } _ { K } } \mathbb { E } _ { f } R _ { N } \geq N \Delta = } \end{array}$ $c _ { * } N \rho _ { N }$ . Since the policy and the rule for the estimated optimal point were arbitrary, taking the corresponding infima proves the claimed minimax lower bounds. □

## B.6 Proof of Theorem 5.4

Proof of Theorem $5 . 4 .$ Use the baseline g, alternatives $f _ { i } ,$ , support balls $A _ { i } ,$ , and tests $E _ { i }$ constructed in Section B.1. In particular, all objectives lie in the same fixed $\mathcal { F } _ { K }$ , the common optimum and optimizer are unchanged, and

$$
g ^ { \star } - g ( x ) \geq T / 8 \mathrm { ~ f o r ~ } x \in A _ { i } , M \geq c \eta ^ { - d / \nu } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} .
$$

Let $N _ { i }$ be the number of evaluations allocated to $A _ { i }$ among the fixed total of N observations. As shown in the proof of Theorem 3.1, the event $E _ { i }$ satisfies $P _ { g } ( E _ { i } ) \leq \delta$ and $P _ { f _ { i } } ( E _ { i } ) \geq 1 - \delta .$ , where $P _ { g }$ and $P _ { f _ { i } }$ denote the joint distributions of the sampling history and the output under g and $f _ { i }$ Similar to the proof of Theorem 3.1, combining (B.8) and (B.9), with $n , P _ { 0 }$ , and $P _ { i }$ replaced by $N .$

$P _ { g } ,$ , and $P _ { f _ { i } }$ , and using $| g - f _ { i } | \leq 8 \eta \mathbf { 1 } _ { A _ { i } }$ , we have

$$
\operatorname { k l } ( \delta , 1 - \delta ) \leq \operatorname { K L } ( P _ { g } , P _ { f _ { i } } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \mathbb { E } _ { g } \sum _ { t = 1 } ^ { N } \{ g ( x _ { t } ) - f _ { i } ( x _ { t } ) \} ^ { 2 } \leq \frac { 3 2 \eta ^ { 2 } } { \sigma ^ { 2 } } \mathbb { E } _ { g } N _ { i } .
$$

The supports are disjoint, so

$$
\mathbb { E } _ { g } R _ { N } \ge \frac { T } { 8 } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { g } N _ { i } \ge \frac { \sigma ^ { 2 } T M } { 2 5 6 \eta ^ { 2 } } \mathrm { k l } ( \delta , 1 - \delta ) \ge c \sigma ^ { 2 } T \mathrm { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - ( 2 + d / \nu ) } \mathrm { k l } ( \delta , 1 - \delta ) .
$$

Since $g \in \mathcal { F } _ { K }$ , taking a supremum proves the claim. Knowing the common optimum and maximizer does not change any of these comparisons between data distributions. All range conditions and constants are those verified in the proof of Theorem 3.1. □

## B.7 Proof of Theorem 5.5

Proof of Theorem 5.5. As in Algorithm 1, index the tracks by $j \in \{ 0 , 1 \}$ , with tolerances $T _ { 0 } ^ { \mathrm { a l g } } = 0$ and $T _ { 1 } ^ { \mathrm { a l g } } = T$ and evaluation quotas $N _ { 0 } = N - m \ge N / 2$ and $N _ { 1 } = m$ . Fix $\vartheta > 1$ . Each track has its own phase counter $h _ { j }$ and accuracy levels $\varepsilon _ { h _ { i } } = 2 B \vartheta ^ { - h _ { j } }$ , and both tracks use the same maintained confidence bounds $( \ell , u )$ . For each track $j ,$ let $\bar { \varepsilon } _ { j }$ denote the accuracy level of its last completed phase, as recorded by Algorithm 3, and let $n _ { j , h }$ be the number of observations collected by track j in phase $h ,$ including a phase truncated by the quota $N _ { j }$ . Upper-bound constants in this proof may depend on ϑ. Most steps follow the proofs of Theorems 5.1 and 5.2 for a single track, and we verify that they apply to each track when the bounds are shared.

We first collect the properties of the shared bounds. By Lemma 5.1 with horizon N and error probability <sup>¯</sup>δ, $\mathbb { P } ( \mathcal { E } _ { \bar { \delta } } ) \geq 1 - \bar { \delta }$ . Except where stated otherwise, we work on $\mathcal { E } _ { \bar { \delta } } .$ . Then all bands contain $f$ by (22), and the maintained bounds are obtained from the initial bounds $\ell = - B \le f \le B = u$ by intersections with the bands of completed nonempty batches of either track. Hence, at all phase boundaries,

$$
\ell \leq f \leq u \mathrm { ~ o n ~ } \mathcal { X } ,\tag{B.35}
$$

ℓ can only increase, u can only decrease, and the safeguard for inconsistent bounds is never triggered.

We next show that each track keeps the guarantees of a single track. Since the proof of Proposition $\mathrm { A . 2 }$ accounts for the bands of intervening phases of the other track through Proposition 4.1, Proposition A.2 applies to each track j with $T = T _ { i } ^ { \mathrm { a l g } }$ and its own phase counter. Line 6 of Algorithm 1 starts a phase only if its accuracy level is at least $2 \zeta _ { N } , \mathrm { s o }$ , by (A.7), any point x sampled by track j in phase h satisfies

$$
f ^ { \star } - f ( x ) \leq T _ { j } ^ { \mathrm { a l g } } + L _ { \vartheta } \varepsilon _ { h } .\tag{B.36}
$$

At the completion of the last completed phase of track $j ,$ , Proposition A.2 gives the two bounds in (16) with $\bar { \varepsilon } = \bar { \varepsilon } _ { j }$ and $T _ { j } = T _ { j } ^ { \mathrm { a l g } }$ , where the second bound uses $f ^ { \star } \leq \overline { { f } }$ . By Proposition 4.1, at any later phase boundary, the bounds (ℓ, u) and $\underline { { f } } = \operatorname* { m a x } _ { \mathcal { X } } \ell$ satisfy

$$
\begin{array} { r } { u ( x ) - \ell ( x ) \le 2 \bar { \varepsilon } _ { j } \mathrm { ~ w h e n e v e r ~ } u ( x ) > \underline { { f } } - T _ { j } ^ { \mathrm { a l g } } - 2 \bar { \varepsilon } _ { j } , \underline { { f } } ^ { \star } - \underline { { f } } \le 2 \bar { \varepsilon } _ { j } . } \end{array}\tag{B.37}
$$

In particular, by (B.35), the saved estimated optimal point ${ \widehat { x } } \in \arg \operatorname* { m a x } _ { x \in { \mathcal { X } } } \ell ( x )$ at any such time satisfies

$$
\begin{array} { r } { f ^ { \star } - f ( \widehat { x } ) \leq f ^ { \star } - \underline { { f } } \leq 2 \bar { \varepsilon } _ { j } . } \end{array}\tag{B.38}
$$

Similarly, Lemma B.3 applies to each phase of track j with $G = G _ { h } , \ W = W _ { h - 1 }$ , and $D =$ $D _ { j , h } = \{ x \in \mathcal { X } : G _ { h } ( x ) \geq 2 \varepsilon _ { h } , \ W _ { h - 1 } ( x ) \geq 2 \varepsilon _ { h } \}$ , and (B.98) in the proof of Proposition $\mathrm { A . 2 }$ gives $D _ { j , h } \subseteq S _ { ( T _ { j } ^ { \mathrm { a l g } } + L _ { \vartheta } \varepsilon _ { h } ) + } .$ The derivations of (B.15), and of (B.16) for a phase with ${ T + L _ { \vartheta } \varepsilon _ { h } } \leq u _ { 0 }$ use only these two facts, and the latter does not use $\alpha > \nu .$ . Hence these per-phase bounds hold for all phases of track $j ,$ , with $n _ { h } = n _ { j , h } , T = T _ { j } ^ { \mathrm { a l g } }$ , and $\delta = \bar { \delta }$ , and they depend on the other track only through the bounds at the start of the phase. For track zero, whose tolerance is $T _ { 0 } ^ { \mathrm { a l g } } = 0$ , the derivation in the proof of Lemma B.5, which uses only Lemma B.3 and (B.98) with $T = 0$ , gives the bounds (B.22) and (B.23) with δ replaced by ${ \bar { \delta } } .$

We now prove (38). The bounds change only when a phase completes, and Algorithm 3 then recomputes and saves the output, so the final output is computed from the final bounds $( \ell , u )$ by (14) and (15). Suppose that track one completes phase H, so that $\bar { \varepsilon } _ { 1 } \leq \varepsilon _ { H } = \varepsilon$ . Then (B.38) with $j = 1$ gives the first bound in (38). Moreover, (B.37) with $j = 1$ is the width bound (B.95) in the proof of Proposition $\mathrm { A . 2 } ,$ with the final bounds in place of $( \ell _ { h } , u _ { h } )$ and $\bar { \varepsilon } _ { 1 }$ in place of $\varepsilon _ { h }$ . The argument there for (B.96) and for $\left( \mathrm { A . 6 } \right)$ uses only this width bound, (B.35), and Proposition A.1. Hence it applies to the final bounds and gives the region bounds in (38), with $4 \bar { \varepsilon } _ { 1 } \leq 4 \varepsilon$ in place of 4ε.

We now bound the regret, first assuming $\zeta _ { N } \leq B$ , and decompose $R _ { N } = R ^ { ( 0 ) } + R ^ { ( 1 ) }$ into the cumulative regret of the observations of track zero and of track one. On $\mathcal { E } _ { \bar { \delta } }$ , track zero finishes only when its quota is exhausted or its accuracy level falls below $2 \zeta _ { N }$ . Hence the simple-regret argument in the proof of Theorem 5.2 applies to track zero, with N replaced by $N _ { 0 } \ge N / 2 , L _ { N }$ by $\Lambda _ { N , \bar { \delta } } , ( \mathrm { B } . 2 4 )$ by these bounds for track zero, and Lemma B.4 by (B.38) with $j = 0$ , which remains valid after later intersections by track one. It gives, on $\mathcal { E } _ { \bar { \delta } }$

$$
r _ { N } \le C \Lambda _ { N , \bar { \delta } } \rho _ { N } + C \zeta _ { N } .\tag{B.39}
$$

With the same replacements, the three facts in the proof of Theorem 5.2 hold for the $N _ { 0 } \leq N$ evaluations of track zero, where (B.36) with $j = 0$ gives the first fact. The argument leading to (B.26) and the same choices of $\varepsilon _ { \mathrm { l o c } }$ and $\varepsilon _ { \mathrm { g l o b } }$ then give

$$
R ^ { ( 0 ) } \leq C \Lambda _ { N , \bar { \delta } } \big [ 1 + N \rho _ { N } \big ] + C N \zeta _ { N } \leq C N \Lambda _ { N , \bar { \delta } } \rho _ { N } + C N \zeta _ { N } ,\tag{B.40}
$$

where the last step uses $N \rho _ { N } \ge N ^ { 1 / 2 } \ge 1$

We next show that $R ^ { ( 1 ) }$ is at most the same term plus CTm. Fix a split scale $\tilde { \varepsilon } \in [ 2 \zeta _ { N } , 2 B ]$ , and divide the evaluations of track one into three groups.

• Phases with $\varepsilon _ { h } > \operatorname* { m a x } \{ T , \tilde { \varepsilon } \}$ . Since $T \ < \ \varepsilon _ { h }$ , (B.36) with $j = 1$ bounds the gap of all points sampled in phase h by $( L _ { \vartheta } + 1 ) \varepsilon _ { h }$ . Moreover, $( T + \varepsilon _ { h } ) ^ { d / \alpha } \leq ( 2 \varepsilon _ { h } ) ^ { d / \alpha }$ , so (B.15) and (B.16) for track one give the same bounds as for track zero, namely $n _ { 1 , h } \le C \Lambda _ { N , \bar { \delta } } \varepsilon _ { h } ^ { - p }$ and $n _ { 1 , h } \le C \Lambda _ { N , \bar { \delta } } K \varepsilon _ { h } ^ { - a }$ when $( L _ { \vartheta } + 1 ) \varepsilon _ { h } \leq u _ { 0 }$ . Hence, as for the phases with $\varepsilon _ { h } \geq \varepsilon$ in the proof of Theorem 5.2, with $L _ { \vartheta }$ replaced by $L _ { \vartheta } + 1$ and ε by $\tilde { \varepsilon } ,$ these phases contribute at most $C \Lambda _ { N , \bar { \delta } } [ 1 + \operatorname* { m i n } \{ K \tilde { \varepsilon } ^ { 1 - a } , \tilde { \varepsilon } ^ { 1 - p } \} ]$ to $R ^ { ( 1 ) }$

• Phases with $\varepsilon _ { h } \le \operatorname* { m a x } \{ T , \tilde { \varepsilon } \}$ . By (B.36) with $j = 1$ , all points sampled in such a phase have gap at most $T + L _ { \vartheta } \operatorname* { m a x } \{ T , \tilde { \varepsilon } \} \leq ( L _ { \vartheta } + 1 ) T + L _ { \vartheta } \tilde { \varepsilon }$ . Since track one makes at most $m \leq N$ evaluations, these phases contribute at most $C T m + C N \tilde { \varepsilon }$

• Evaluations at the saved estimated optimal point. As for track zero, (B.38) with $j = 1$ bounds the gap of each of these at most $m \leq N$ evaluations by $4 \vartheta \zeta _ { N }$ , so they contribute at most $4 \vartheta N \zeta _ { N }$

Summing the three contributions gives

$$
R ^ { ( 1 ) } \leq C \Lambda _ { N , \bar { \delta } } \big [ 1 + \operatorname* { m i n } \{ K \tilde { \varepsilon } ^ { 1 - a } , \tilde { \varepsilon } ^ { 1 - p } \} \big ] + C T m + C N \tilde { \varepsilon } + C N \zeta _ { N } .\tag{B.41}
$$

Choosing ε˜ in (B.41) as $\varepsilon _ { \mathrm { l o c } }$ or $\varepsilon _ { \mathrm { g l o b } }$ in the proof of Theorem 5.2, with $L _ { N }$ replaced by $\Lambda _ { N , \bar { \delta } } ^ { \phantom { N , } } ,$ gives

$$
R ^ { ( 1 ) } \leq C \Lambda _ { N , \bar { \delta } } \big [ 1 + N \rho _ { N } \big ] + C T m + C N \zeta _ { N } \leq C N \Lambda _ { N , \bar { \delta } } \rho _ { N } + C T m + C N \zeta _ { N } .\tag{B.42}
$$

If $\zeta _ { N } > B$ , then $r _ { N } \le 2 B < 2 \zeta _ { N } , R ^ { ( 0 ) } \le 2 B N < 2 N \zeta _ { N }$ , and $R ^ { ( 1 ) } \leq 2 B m < 2 N \zeta _ { N }$ , so (B.39), (B.40), and (B.42) hold trivially. Adding (B.40) and (B.42) gives $R _ { N } \leq C N \Lambda _ { N , \bar { \delta } } \rho _ { N } + C T m + C N \zeta _ { N }$ on $\mathcal { E } _ { \bar { \delta } }$ . On the complement of $\mathcal { E } _ { \bar { \delta } } ,$ which has probability at most $\bar { \delta } ,$ we have $r _ { N } \leq 2 B$ and $R _ { N } \le 2 B N$ Taking expectations gives (36) and (37), whose logarithmic factor is $\Lambda _ { N , \bar { \delta } } ^ { \phantom { N , } }$ . Without Assumption 5.4, the same arguments with only the global bounds give these bounds with $\rho _ { N }$ replaced by $N ^ { - 1 / ( 2 + d / \nu ) }$

It remains to show that the quotas in (39) and (40) sufice. Let $T > 0$ , and let H be the target phase of track one, with accuracy level $\varepsilon = \varepsilon _ { H } = 2 B \vartheta ^ { - H } \geq 2 \zeta _ { N }$ . The argument showing that phase H completes in the proof of Theorem 5.1 applies to track one, with the budget N replaced by the quota m and $n _ { h }$ by $n _ { 1 , h }$ . Indeed, all phases $h \leq H$ of track one, including a phase truncated by the quota, satisfy (B.15) and, under Assumption 5.4 with $\alpha > \nu$ and $0 < T \le u _ { 0 } / 2$ , also (B.16), with $\delta = \bar { \delta }$ , and these bounds count only the observations of track one. Moreover, the safeguard is never triggered, line 6 of Algorithm 1 does not stop track one before phase H completes, because $\varepsilon _ { h } \ge \varepsilon _ { H } \ge 2 \zeta _ { N }$ for all $h \leq H$ , and Algorithm 2 tests the stopping condition before the quota. Hence, if phase H of track one did not complete, all m evaluations of track one would be made in phases $h \leq H$ , and summing as in (B.17) and (B.18) would give

$$
m \leq C _ { 1 } \Lambda _ { N , \bar { \delta } } \big ( 1 + \varepsilon ^ { - p } \big ) , m \leq C _ { 1 } \Lambda _ { N , \bar { \delta } } \big [ 1 + K \varepsilon ^ { - 2 } + K ( T + \varepsilon ) ^ { d / \alpha } \varepsilon ^ { - p } \big ] ,
$$

where $C _ { 1 }$ is the constant in (B.17) and (B.18), and the second bound requires the assumptions of (40). Taking the constants C in (39) and (40) larger than $C _ { 1 }$ , this contradicts the assumption on m. Therefore, phase H of track one completes within its quota, and (38) holds by the output guarantees proved above. □

## B.8 Proof of Corollary 5.2

Proof of Corollary 5.2. Any procedure with valid region bounds is also an optimization policy. Thus Theorem 5.3 gives an optimization lower bound, and (35) gives the additional lower bound from region identification. Hence, for any such policy π,

$$
\operatorname* { s u p } _ { f \in { \mathscr F } _ { K } } { \mathbb E } _ { f } ^ { \pi } R _ { N } \geq c \operatorname* { m a x } \Bigl \{ N \rho _ { N } , ~ T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - ( 2 + d / \nu ) } \operatorname { k l } ( \delta , 1 - \delta ) \Bigr \} ,
$$

with $\sigma ^ { 2 }$ absorbed into $c ,$ and taking the infimum over admissible policies gives (42), with $\operatorname { k l } ( \delta , 1 - \delta )$ kept explicitly. This finishes the proof of lower bound.

For the upper bound, as in the proof of Corollary 5.1, $T \leq T _ { \mathrm { r e g } } \leq \mathrm { m i n } \{ 1 , u _ { 0 } / 2 , 2 0 4 8 B \}$ gives $T \le u _ { 0 } / 2$ , so that (40) applies, and $\eta \leq T / 1 2 8 \leq 1 6 B$ , so that there is a phase target $\varepsilon = \varepsilon _ { H }$ with $\eta / ( 8 \vartheta ) < \varepsilon \leq \eta / 8$ . The condition $\eta \geq 1 6 \operatorname* { m a x } \{ 2 , \vartheta \} \zeta _ { N }$ ensures $\varepsilon > 2 \zeta _ { N }$ . In this range, the proof of Corollary 5.1 absorbs $K \varepsilon ^ { - 2 }$ into $K ( T + \varepsilon ) ^ { d / \alpha } \varepsilon ^ { - ( 2 + d / \nu ) }$ . Thus the quota (41) completes that region phase, with region estimation error at most $4 \varepsilon \leq \eta$

This remains valid when phases are interleaved, since the other track only tightens the region

track’s starting bands. Theorem 5.5 with $m = m _ { \eta }$ then gives

$$
\mathbb { E } _ { f } R _ { N } \le C \Lambda _ { N , \bar { \delta } } N \rho _ { N } + C T m _ { \eta } + C N \zeta _ { N } + 2 B N \bar { \delta } .\tag{B.43}
$$

We bound each term on the right-hand side of (B.43) by a constant multiple of

$$
\Lambda _ { N , \bar { \delta } } \operatorname* { m a x } \{ N \rho _ { N } , T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } \} .
$$

By (41), $m _ { \eta } \ \leq \ C \Lambda _ { N , \bar { \delta } }$ min $\{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } + 1$ . Since $T \leq T _ { \mathrm { r e g } } \leq 1$ and $\eta ~ \leq ~ T / 1 2 8$ , we have $\eta ^ { - p } \geq T ^ { - p } \geq 1$ and $K T ^ { d / \alpha } \eta ^ { - p } \geq K T ^ { d / \alpha - p } \geq 1$ , because $K \geq 1$ and $d / \alpha < d / \nu < p$ . Hence min $\{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } \geq 1$ , and since $\Lambda _ { { N } , \bar { \delta } } \geq 1$ , the additional term 1 from rounding is absorbed:

$$
\begin{array} { r } { T m _ { \eta } \le T \big [ C \Lambda _ { N , \bar { \delta } } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } + 1 \big ] \le ( C + 1 ) \Lambda _ { N , \bar { \delta } } T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } . } \end{array}
$$

Moreover, $\zeta _ { N } \leq C N ^ { - 7 / 2 } \log ^ { 1 / 2 } ( 2 N / { \bar { \delta } } )$ by (9), so $C N \zeta _ { N } \le C \Lambda _ { N , \bar { \delta } } N \rho _ { N }$ , and $\bar { \delta } \leq N ^ { - 2 }$ gives $2 B N \bar { \delta } \leq$ $2 B / N \le C \Lambda _ { N , \bar { \delta } } N \rho _ { N }$ . Inserting these bounds into (B.43) and using $x + y \leq 2 \operatorname* { m a x } \{ x , y \}$ for $x , y \geq 0$ proves (43).

The same argument applied to (36) proves the simultaneous simple regret bound. All upper-bound constants may depend on the fixed ϑ and are uniform in δ. □

## B.9 Supporting Calculations for the Accuracy–Regret Tradeof

This subsection provides the direct regret bound behind Corollary 5.2 and derives the rates and feasibility conditions stated in Section 5.4. Throughout, $m _ { \mathrm { o p t } }$ and $m _ { \mathrm { f r e e } }$ denote the quotas in (44) and (45), respectively.

Regret cost of a region run at a prescribed target. The sharper direct bound on cumulative regret underlying the optimal tradeof can also be seen without the two-track balancing split. Suppose $\alpha > \nu , 0 < T \leq T _ { \mathrm { r e g } }$ , and consider the observations allocated to a region track through completion of its first phase with accuracy level comparable to $\eta \le T / 1 2 8$ . For coarse scales $\varepsilon _ { h } \geq T$ , each sampled point has gap at most $( L _ { \vartheta } + 1 ) \varepsilon _ { h }$ by (A.7), and (B.15) and (B.16) give $n _ { h } = \widetilde { \cal O } ( \operatorname* { m i n } \{ \varepsilon _ { h } ^ { - ( 2 + d / \nu ) } , K \varepsilon _ { h } ^ { - a } \} )$ , where $a = 2 + d / \nu - d / \alpha$ because $\alpha > \nu ,$ so the summed weighted cost is

$$
\sum _ { h \colon \varepsilon _ { h } \subset T } ( L _ { \vartheta } + 1 ) \varepsilon _ { h } n _ { h } = \widetilde O \bigl ( \operatorname* { m i n } \{ T ^ { - 1 - d / \nu } , K T ^ { 1 - a } \} \bigr ) = \widetilde O \bigl ( T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} T ^ { - ( 2 + d / \nu ) } \bigr ) ,\tag{B.44}
$$

because $K T ^ { 1 - a } = K T ^ { d / \alpha } \cdot T ^ { - 1 - d / \nu } ;$ ; the fixed initial scales are absorbed into (B.44).

For fine scales $\eta \lesssim \varepsilon _ { h } \le T$ , (B.16) gives $n _ { h } = \widetilde { \cal O } ( \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \varepsilon _ { h } ^ { - ( 2 + d / \nu ) } )$ , including the absorbed $K \varepsilon _ { h } ^ { - 2 }$ term, and each sampled point has gap at most $( L _ { \vartheta } + 1 ) T$ by (A.7), so

$$
\sum _ { h : \eta \leqslant \varepsilon _ { h } \le T } ( L _ { \vartheta } + 1 ) T n _ { h } = \widetilde { O } \big ( T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - ( 2 + d / \nu ) } \big ) ,\tag{B.45}
$$

which also absorbs (B.44) because $\eta \leq T$

Together, (B.44) and (B.45) bound the regret contributed by the observations needed to attain the prescribed regional accuracy. Later observations are still collected within the fixed budget, and their regret is controlled by the proof of Theorem 5.5 in Section B.7.

Quota tuning and tolerance feasibility. For $m _ { \mathrm { o p t } }$ and $m _ { \mathrm { f r e e } }$ , respectively,

$$
T m \leq T \cdot \frac { N \rho _ { N } } { T } = N \rho _ { N } , T m \leq T \cdot \frac { \sqrt { N } } { T } = \sqrt { N } \leq N \rho _ { N } ,
$$

where the last inequality uses $K \geq 1$ and $a \geq 2$ , so that $N \rho _ { N } = \operatorname* { m i n } \{ K ^ { 1 / a } N ^ { 1 - 1 / a } , N ^ { 1 - 1 / p } \} \geq N ^ { 1 / 2 }$ In both cases the term CTm in (37) is dominated by the optimization term, so Theorem 5.5 preserves both regret orders.

In the range $\eta \leq T / 1 2 8$ , where $K \varepsilon ^ { - 2 }$ is absorbed as in Section B.3, the suficient region quota (40) with $\varepsilon \asymp \eta$ reads

$$
m \geq C \Lambda _ { N , \bar { \delta } } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } \Longleftrightarrow \eta \geq \Big [ \frac { C \Lambda _ { N , \bar { \delta } } \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} } { m } \Big ] ^ { 1 / p } ,
$$

which, together with the condition $\eta \geq 1 6 \operatorname* { m a x } \{ 2 , \vartheta \} \zeta _ { N }$ , is (46), and at fixed $K , T , \delta$ the two quota sizes yield (47) and (48).

At fixed confidence, a cumulative regret bound $\mathrm { s u p } _ { f } \mathbb { E } _ { f } R _ { N } \leq C N \rho _ { N }$ combined with (35), and the evaluation lower bound (7) with N evaluations available, give respectively

$$
\begin{array} { r l r } & { } & { c T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } \leq C N \rho _ { N } \implies \eta \geq c \Big [ \frac { T \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} } { N \rho _ { N } } \Big ] ^ { 1 / p } , } \\ & { } & { c \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} \eta ^ { - p } \leq N \implies \eta \geq c \Big [ \frac { \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} } { N } \Big ] ^ { 1 / p } , ~ } \end{array}
$$

which are the two terms in (49).

If $\eta \leq C _ { 1 } [ \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} / N ] ^ { 1 / p }$ , substituting into (35) gives sup $_ f \mathbb { E } _ { f } R _ { N } \ge c C _ { 1 } ^ { - p } T N$ , a regret of order $T N$ . If the accuracy is relaxed by a factor $\log ^ { 3 / 2 } ( 2 N )$ , that is,

$$
\eta \leq C _ { 1 } \log ^ { 3 / 2 } ( 2 N ) \big [ \operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} / N \big ] ^ { 1 / p } ,
$$

then the same substitution gives sup $_ f \mathbb { E } _ { f } R _ { N } \ge c C _ { 1 } ^ { - p } T N / L _ { N }$ , because $\log ^ { 3 p / 2 } ( 2 N ) = L _ { N }$ . For fixed K and $T$ , this lower bound still exceeds the optimal rate $N \rho _ { N }$ for all suficiently large N, because $N \rho _ { N }$ grows polynomially slower than N. The quota $m = \lfloor N / 2 \rfloor$ keeps matching simple regret and region-estimation orders.

Finally, substituting $\eta = T / 1 2 8$ into (46) gives $T \geq 2 0 4 8 \operatorname* { m a x } \{ 2 , \vartheta \} \zeta _ { N }$ . Using

$$
\operatorname* { m i n } \{ 1 , K T ^ { d / \alpha } \} T ^ { - p } = \operatorname* { m i n } \{ T ^ { - p } , K T ^ { - ( p - d / \alpha ) } \} ,
$$

the remaining quota condition becomes

$$
C \Lambda _ { N , \bar { \delta } } \operatorname* { m i n } \{ T ^ { - p } , K T ^ { - ( p - d / \alpha ) } \} \leq m \Longleftrightarrow T \geq C \operatorname* { m i n } \Bigl \{ \Bigl [ \frac { K \Lambda _ { N , \bar { \delta } } } { m } \Bigr ] ^ { 1 / ( p - d / \alpha ) } , \Bigl [ \frac { \Lambda _ { N , \bar { \delta } } } { m } \Bigr ] ^ { 1 / p } \Bigr \} ,
$$

which is (50). Substituting $m _ { \mathrm { o p t } }$ 2 $m \asymp N \rho _ { N } / T$ , into the localized bound, where $\rho _ { N } = ( K / N ) ^ { 1 / a }$ with $a = p - d / \alpha$ , gives

$$
T ^ { a - 1 } \geq \frac { C \Lambda _ { N , \bar { \delta } } K } { N \rho _ { N } } = C \Lambda _ { N , \bar { \delta } } \Big ( \frac { K } { N } \Big ) ^ { 1 - 1 / a } \Longleftrightarrow T \geq C \Big ( \frac { K } { N } \Big ) ^ { 1 / a } \Lambda _ { N , \bar { \delta } } ^ { 1 / ( a - 1 ) } , a - 1 = 1 + \frac { d } { \nu } - \frac { d } { \alpha } ,
$$

and into the global bound, where $\rho _ { N } = N ^ { - 1 / p }$ 2

$$
T ^ { p - 1 } \geq \frac { C \Lambda _ { N , \bar { \delta } } } { N \rho _ { N } } = C \Lambda _ { N , \bar { \delta } } N ^ { - 1 + 1 / p } \Longleftrightarrow T \geq C N ^ { - 1 / p } \Lambda _ { N , \bar { \delta } } ^ { 1 / ( p - 1 ) } , p - 1 = 1 + \frac { d } { \nu } ;
$$

with $\bar { \delta } = N ^ { - 2 } , \Lambda _ { N , \bar { \delta } } \asymp L _ { N }$ , which is the logarithmic inflation stated in the main text.

## B.10 Proof of Theorem 3.2

The proof reuses the construction and fixed constants from the proof of Theorem 3.1 in Section B.1, including the norm bound provided by Lemma B.1.

Proof of Theorem 3.2. We use the fixed constants $x ^ { \star } , r _ { * } , \mu , \kappa _ { * } , C _ { \psi } , L , \rho _ { 0 } , K _ { \mathrm { r e g } } .$ and $T _ { \mathrm { r e g } }$ from the proof of Theorem 3.1. We apply that construction with $T = 4 \tau$ , but place all perturbation centers on a single level set of the baseline, so that the single tolerance $\tau$ distinguishes all alternatives from the baseline.

Set $T = 4 \tau$ and

$$
k _ { T } = \operatorname* { m i n } \{ K , | \kappa _ { * } ( 4 \tau ) ^ { - d / \alpha } | \} , g ( x ) = - \mu k _ { T } ^ { - \alpha / d } \| x - x ^ { \star } \| ^ { \alpha } , R = ( 4 \tau / \mu ) ^ { 1 / \alpha } k _ { T } ^ { 1 / d } , h = L \eta ^ { 1 / \nu } .
$$

These are the quantities $k _ { T } , g , R .$ , and h in the proof of Theorem 3.1 with $T = 4 \tau$ . Take $K _ { 1 } = K _ { \mathrm { r e g } }$ and $\tau _ { 0 } \leq T _ { \mathrm { r e g } } / 4$ , where $\tau _ { 0 }$ may be decreased further below. Then $T \leq T _ { \mathrm { r e g } }$ , and the choices of $K _ { \mathrm { r e g } }$ and $T _ { \mathrm { r e g } }$ in the proof of Theorem 3.1 in Section B.1 give the following bounds. Since $K \geq K _ { \mathrm { r e g } }$ and $\kappa _ { * } ( 4 \tau ) ^ { - d / \alpha } \geq \kappa _ { * } T _ { \mathrm { r e g } } ^ { - d / \alpha } \geq 2 K _ { \mathrm { r e g } }$ , we have $k _ { T } \geq K _ { \mathrm { r e g } }$ . Hence, with $C _ { g }$ as in that proof, $\| g \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } = \mu C _ { g } { k _ { T } } ^ { - \alpha / d } \leq \mu C _ { g } K _ { \mathrm { r e g } } ^ { - \alpha / d } \leq B / 2$ . Moreover, $k _ { T } \le \kappa _ { * } ( 4 \tau ) ^ { - d / \alpha }$ and $\kappa _ { * } = \mu ^ { d / \alpha } r _ { * } ^ { d }$ give $R ^ { d } = ( 4 \tau / \mu ) ^ { d / \alpha } k _ { T } \leq r _ { * } ^ { d }$ , that is, $R \leq r _ { * }$ . Finally, by (B.4) with $T = 4 \tau$

$$
R ^ { d } \asymp \operatorname* { m i n } \{ 1 , K \tau ^ { d / \alpha } \}\tag{B.46}
$$

uniformly over $K$ and $\tau .$ . The baseline has the unique maximizer $x ^ { \star }$ and the maximum value $g ( x ^ { \star } ) = 0$

The baseline gap is $- g ( x ) = 4 \tau ( \| x - x ^ { \star } \| / R ) ^ { \alpha }$ . Hence it equals $\tau +$ 4η exactly on the sphere

$$
\mathcal { C } _ { \tau } = \{ x : \| x - x ^ { \star } \| = r _ { \tau } \} , r _ { \tau } = R \Big ( \frac { \tau + 4 \eta } { 4 \tau } \Big ) ^ { 1 / \alpha } .
$$

Since $0 < \eta \leq \tau / 3 2$ , we have $( \tau + 4 \eta ) / ( 4 \tau ) \in [ 1 / 4 , 9 / 3 2 ] \subset [ 1 / 4 , 1 / 2 ] , \mathrm { ~ s o ~ } 4 ^ { - 1 / \alpha } R \le r _ { \tau } \le 2 ^ { - 1 / \alpha } R$ Thus $\mathcal { C } _ { \tau } \subseteq \mathcal { A } _ { R } .$ , where $\boldsymbol { \mathcal { A } } _ { R }$ is the set defined in the proof of Theorem 3.1 with the present value of $R ,$ and $r _ { \tau } \asymp R$ uniformly. Since $\eta \le \tau / 3 2 = T / 1 2 8$ , (B.5) gives $h \le \rho _ { 0 } R$ and $h \leq 1$ . Because $\alpha > \nu ,$ decreasing $\tau _ { 0 }$ also ensures $h \leq r _ { \tau } / 4$

We next place disjoint supports along $\mathcal { C } _ { \tau }$ . For $d \geq 2$ , take a maximal 2h-separated set of centers $z _ { 1 } , \dots , z _ { M }$ in $\mathcal { C } _ { \tau } .$ . The open balls $A _ { i } = B ( z _ { i } , h )$ are disjoint, and by maximality the closed balls $\overline { { B } } ( z _ { i } , 2 h )$ cover $\mathcal { C } _ { \tau }$ . Let $\omega _ { d - 1 }$ denote the surface area of the unit sphere $\{ x \in \mathbb { R } ^ { d } : \| x \| = 1 \}$ , so that $\mathcal { C } _ { \tau }$ has surface area $\omega _ { d - 1 } r _ { \tau } ^ { d - 1 }$ . Since $2 h \leq r _ { \tau } / 2$ , after a rotation about $x ^ { \star }$ , the set $\overline { { B } } ( z _ { i } , 2 h ) \cap \mathcal { C } _ { \tau }$ is the graph of a function with uniformly bounded slope over a $( d - 1 )$ -dimensional ball of radius $2 h$ and hence its surface area is at most $C _ { d } h ^ { d - 1 }$ . Comparing surface areas gives

$$
\omega _ { d - 1 } r _ { \tau } ^ { d - 1 } \leq \sum _ { i = 1 } ^ { M } \mathrm { a r e a } \bigl ( \overline { { B } } ( z _ { i } , 2 h ) \cap \mathscr { C } _ { \tau } \bigr ) \leq M C _ { d } h ^ { d - 1 } ,
$$

so that $M \geq c _ { d } ( r _ { \tau } / h ) ^ { d - 1 } \geq c ( R / h ) ^ { d - 1 }$ . For $d = 1$ , take the two centers $z _ { 1 } = x ^ { \star } - r _ { \tau }$ and $z _ { 2 } = x ^ { \star } + r _ { \tau }$ They lie in X because $r _ { \tau } \le R \le r _ { * }$ , and the intervals $A _ { i } = ( z _ { i } - h , z _ { i } + h )$ are disjoint because $h \leq r _ { \tau } / 4$ , so $M = 2$ . In both cases, $h = L \eta ^ { 1 / \nu }$ and (B.46) give

$$
M \geq c \eta ^ { - ( d - 1 ) / \nu } \big ( \operatorname* { m i n } \{ 1 , K \tau ^ { d / \alpha } \} \big ) ^ { ( d - 1 ) / d } .\tag{B.47}
$$

Define the alternatives $f _ { i } ( x ) = g ( x ) + 8 \eta \psi ( ( x - z _ { i } ) / h ) , 1 \le i \le M$ , where ψ is the bump function in the proof of Theorem 3.1. As in that proof, $\| f _ { i } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B / 2 + 8 \eta C _ { \psi } h ^ { - \nu } = B$ . All points of $A _ { i }$ lie within distance $h \le \rho _ { 0 } R$ of $\mathcal { C } _ { \tau } \subseteq A _ { R } ,$ so the choice of $\rho _ { 0 } \mathrm { g i v e s } - g ( x ) \in [ T / 8 , 3 T / 4 ]$ on $A _ { i } ,$ whereas $8 \eta \leq T / 1 6$ . Hence, as in (B.7), every $f _ { i }$ has the unique maximizer $x ^ { \star }$ , the maximum value $f _ { i } ^ { \star } = 0$ and

$$
\begin{array} { r } { \frac 1 2 ( - g ( x ) ) \leq f _ { i } ^ { \star } - f _ { i } ( x ) \leq - g ( x ) \mathrm { ~ f o r ~ } x \in \mathcal { X } . } \end{array}
$$

Consequently, for all $u > 0$ , the baseline and all alternatives satisfy $\begin{array} { r } { \mathcal { S } _ { u + } \subseteq \overline { { B } } \big ( x ^ { \star } , \frac { 1 } { 4 } c _ { \rho } k _ { T } { } ^ { 1 / d } u ^ { 1 / \alpha } \big ) } \end{array}$ and, as in the proof of Theorem 3.1, this ball is covered by at most $k _ { T } \leq K$ balls of radius $c _ { \rho } u ^ { 1 / \alpha }$ Hence $g , f _ { 1 } , \dots , f _ { M }$ belong to the same class $\mathcal { F } _ { K }$ , which does not depend on $\eta .$

Consider any procedure with a fixed budget of n evaluations that satisfies (1) at the prescribed tolerance $\tau$ with probability at least $1 - \delta$ for $f \in \mathcal { F } _ { K }$ . For $f \in \{ g , f _ { 1 } , \ldots , f _ { M } \}$ , let $\mathcal { E } _ { f }$ be the event that the output $( I _ { \tau } , O _ { \tau } )$ satisfies (1) at $\tau ,$ so that $\mathcal { E } _ { f }$ has probability at least $1 - \delta$ when the response surface is $f .$ . Let $E _ { i } = \{ z _ { i } \in I _ { \tau } \}$ . Since $g ^ { \star } = f _ { i } ^ { \star } = 0$ and $\psi ( 0 ) = 1$ , the performance gap of $z _ { i }$ is $g ^ { \star } - g ( z _ { i } ) = \tau + 4 \eta$ under g and $f _ { i } ^ { \star } - f _ { i } ( z _ { i } ) = \tau - 4 \eta$ under $f _ { i }$ . Similar to the proof of Theorem 3.1, we have $E _ { i } \cap \mathcal { E } _ { g } = \emptyset$ and $\mathcal { E } _ { f _ { i } } \subseteq E _ { i }$

• on $\mathcal { E } _ { g } ,$ , since $g ^ { \star } - g ( z _ { i } ) > \tau$ , we have $z _ { i } \notin S _ { \tau } ( g )$ , and the inclusion $I _ { \tau } \subseteq S _ { \tau } ( g )$ gives $z _ { i } \notin I _ { \tau }$ ;

• on $\mathcal { E } _ { f _ { i } }$ , since $f _ { i } ^ { \star } - f _ { i } ( z _ { i } ) < \tau$ , we have $z _ { i } \in S _ { \tau } ( f _ { i } ) \subseteq O _ { \tau }$ , and since $| f _ { i } ^ { \star } - f _ { i } ( z _ { i } ) - \tau | = 4 \eta > \eta$ the second condition in (1) gives $z _ { i } \notin O _ { \tau } \ : \backslash \ : I _ { \tau }$ , so that $z _ { i } \in I _ { \tau }$

Let $P _ { 0 }$ and $P _ { i }$ be the joint distributions of the sampling history and the output, including the internal randomization of the procedure, when the response surface is $g$ and $f _ { i } ,$ , respectively. The two inclusions give $P _ { 0 } ( E _ { i } ) \leq \delta$ and $P _ { i } ( E _ { i } ) \geq 1 - \delta ,$ so (B.8) gives $\mathrm { K L } ( P _ { 0 } , P _ { i } ) \ge \mathrm { k l } ( \delta , 1 - \delta )$ . Let $\begin{array} { r } { N _ { i } = \sum _ { t = 1 } ^ { n } \mathbf { 1 } \{ x _ { t } \in A _ { i } \} } \end{array}$ be the number of evaluations allocated to $A _ { i } .$ . Since $0 \leq f _ { i } - g \leq 8 \eta \mathbf { 1 } _ { A _ { i } }$ the derivation of (B.9) and (B.10) applies without change and gives $\begin{array} { r } { \mathrm { K L } ( P _ { 0 } , P _ { i } ) \le ( 3 2 \eta ^ { 2 } / \sigma ^ { 2 } ) \mathbb { E } _ { 0 } N _ { i } } \end{array}$ Combining the two bounds on $\mathrm { K L } ( P _ { 0 } , P _ { i } )$ gives

$$
\mathbb { E } _ { 0 } N _ { i } \ge \frac { \sigma ^ { 2 } } { 3 2 \eta ^ { 2 } } \mathrm { k l } ( \delta , 1 - \delta ) , 1 \le i \le M .\tag{B.48}
$$

Since the supports $A _ { i }$ are disjoint, $\textstyle \sum _ { i = 1 } ^ { M } N _ { i } \leq n$ . Hence (B.48) and (B.47) give

$$
n \geq \sum _ { i = 1 } ^ { M } \mathbb { E } _ { 0 } N _ { i } \geq \frac { \sigma ^ { 2 } M } { 3 2 \eta ^ { 2 } } \mathrm { k l } ( \delta , 1 - \delta ) \geq c \sigma ^ { 2 } \mathrm { k l } ( \delta , 1 - \delta ) \eta ^ { - 2 - ( d - 1 ) / \nu } \big ( \operatorname* { m i n } \{ 1 , K \tau ^ { d / \alpha } \} \big ) ^ { ( d - 1 ) / d } .
$$

Since the procedure was arbitrary, this proves (8). The same argument applies when the procedure is given $( x ^ { \star } , 0 )$ before sampling, because this information is the same under $P _ { 0 }$ and $P _ { i }$ . The constants depend only on the fixed parameters and are independent of $K , \tau , \eta , \delta$ in the stated range. □

## B.11 Proof of Corollary 5.3

Proof of Corollary 5.3. Use the baseline $^ { g , }$ the alternatives $f _ { i } ,$ and the disjoint supports $A _ { i }$ from the proof of Theorem 3.2, where $T = 4 \tau$ . All points of $A _ { i }$ have baseline gap $g ^ { \star } - g ( x ) \geq T / 8 = \tau / 2$

Let $\begin{array} { r } { N _ { i } = \sum _ { t = 1 } ^ { N } \mathbf { 1 } \{ x _ { t } \in A _ { i } \} } \end{array}$ . The procedure in the corollary satisfies (1) at τ with probability at least $1 - \delta$ for $f \in \mathcal { F } _ { K }$ . Hence the argument leading to (B.48), with n replaced by N, gives $\mathbb { E } _ { g } N _ { i } \ge \sigma ^ { 2 } \mathrm { k l } ( \delta , 1 - \delta ) / ( 3 2 \eta ^ { 2 } )$ for each i. Since the supports are disjoint and all evaluations in $A _ { i }$ contribute at least $\tau / 2$ to the cumulative regret under g,

$$
\mathbb { E } _ { g } R _ { N } \ge \frac { \tau } { 2 } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { g } N _ { i } \ge \frac { \sigma ^ { 2 } \tau M } { 6 4 \eta ^ { 2 } } \mathrm { k l } ( \delta , 1 - \delta ) .
$$

Since $g \in { \mathcal { F } } _ { K }$ , substituting (B.47) proves (56). As in the proof of Theorem 3.2, providing the common optimizer and optimal value before sampling does not change these distributions. □

## B.12 Proof of Theorem 5.6

Define the compact sets $Q _ { h } = \{ x \in \mathcal { X } : g _ { h - 1 } ( x ) \geq - 2 \varepsilon _ { h } \}$ . We show in the proof that, on ${ \mathcal { E } } _ { \delta }$ , these sets are nested, contain all global maximizers and all points sampled in phase $h ,$ and satisfy

$$
Q _ { h } \subseteq S _ { ( L _ { \vartheta } \varepsilon _ { h } ) + } \cup \{ x : | f ^ { \star } - f ( x ) - \tau | \leq L _ { \vartheta } \varepsilon _ { h } \} .\tag{B.49}
$$

In addition to the global budget (55) in Theorem 5.6, we prove the refined suficient budget

$$
C \Lambda _ { N , \delta } \sum _ { h = 0 } ^ { H } \xi _ { \tau , f } ( \varepsilon _ { h } ) ,\tag{B.50}
$$

where $\xi _ { \tau , f } ( u ) = \operatorname* { m i n } \{ u ^ { - ( 2 + d / \nu ) } , u ^ { - 2 } J _ { \tau , f } ( u ) \}$ and $J _ { \tau , f }$ is defined in (59). That is, whenever (B.50) is at most N and $\varepsilon _ { H } \geq 2 \zeta _ { N }$ , phase H completes on ${ \mathcal { E } } _ { \delta }$ . Here $C$ may also depend on the fixed phase ratio ϑ and the covering-radius constant $c _ { r }$

Proof of Theorem 5.6. Throughout the proof, we work on ${ \mathcal { E } } _ { \delta }$ , which has probability at least $1 - \delta$ by Lemma 5.1. The lemma applies because, as in Algorithm 1, the screening function in (51) and the maintained width are fixed at the start of each phase, the sampling and stopping decisions within a phase do not use the responses of that phase, and empty phases need no confidence event.

We first show that the bounds and the screening functions are monotone over phases. The bands of completed phases are intersected, so

$$
\ell _ { h } \geq \ell _ { h - 1 } , u _ { h } \leq u _ { h - 1 } , \underline { { f } } _ { h } \geq \underline { { f } } _ { h - 1 } , \overline { { f } } _ { h } \leq \overline { { f } } _ { h - 1 } .\tag{B.51}
$$

Hence $\begin{array} { r } { u _ { h } - \underline { { f } } _ { h } \leq u _ { h - 1 } - \underline { { f } } _ { h - 1 } , \overline { { f } } _ { h } - \tau - \ell _ { h } \leq \overline { { f } } _ { h - 1 } - \tau - \ell _ { h - 1 } , \mathrm { ~ a n d ~ } u _ { h } - \underline { { f } } _ { h } + \tau \leq u _ { h - 1 } - \underline { { f } } _ { h - 1 } + \tau } \end{array}$ Since $g _ { h }$ is obtained from these three functions by the same minimum and maximum as in (51), it follows that $g _ { h } \le g _ { h - 1 }$

We next prove the accuracy claims (53) and (54) for a completed phase h. At completion, the stopping rule (12) gives $a _ { h , n } ( x ) \leq 2 \varepsilon _ { h }$ for all $x \in \mathcal { X }$ , where $a _ { h , n } = \operatorname* { m i n } \{ G _ { h } , W _ { h - 1 } , 2 \beta _ { N } s _ { n } + 2 \zeta _ { N } \}$ Hence, at any x with $G _ { h } ( x ) > 2 \varepsilon _ { h }$

$$
u _ { h } ( x ) - \ell _ { h } ( x ) \le \operatorname* { m i n } \{ u _ { h - 1 } ( x ) - \ell _ { h - 1 } ( x ) , 2 \beta _ { N } s _ { n } ( x ) + 2 \zeta _ { N } \} \le 2 \varepsilon _ { h } .\tag{B.52}
$$

For a nonempty batch, the first inequality holds because the band $\left[ \mu _ { n } - \beta _ { N } s _ { n } - \zeta _ { N } , \mu _ { n } + \beta _ { N } s _ { n } + \zeta _ { N } \right]$ is intersected with $[ \ell _ { h - 1 } , u _ { h - 1 } ]$ . For an empty batch, $2 \beta _ { N } s _ { 0 } + 2 \zeta _ { N } \geq 2 B \geq u _ { h - 1 } - \ell _ { h - 1 }$ because $s _ { 0 } \equiv 1$ and $\beta _ { N } \geq B ,$ so the minimum equals the previous width. If $g _ { h } ( x ) \geq 0$ , then $g _ { h - 1 } ( x ) \geq g _ { h } ( x ) \geq 0$ , so $G _ { h } ( x ) = [ g _ { h - 1 } ( x ) + 4 \varepsilon _ { h } ] _ { + } \geq 4 \varepsilon _ { h } > 2 \varepsilon _ { h }$ , and (B.52) gives $u _ { h } ( x ) - \ell _ { h } ( x ) \le 2 \varepsilon _ { h }$

By (51) with $( \ell , u , \underline { { f } } , \overline { { f } } ) = ( \ell _ { h } , u _ { h } , \underline { { f } } _ { h } , \overline { { f } } _ { h } )$ , we have $\{ g _ { h } \ge 0 \} = V _ { h } \cup C _ { h }$ , where

$$
\begin{array} { r } { V _ { h } = \{ x : u _ { h } ( x ) \geq \underline { { f } } _ { h } \} , C _ { h } = \{ x : \ell _ { h } ( x ) \leq \overline { { f } } _ { h } - \tau , \ u _ { h } ( x ) \geq \underline { { f } } _ { h } - \tau \} . } \end{array}
$$

Any global maximizer $x ^ { \star }$ satisfies $u _ { h } ( x ^ { \star } ) \geq f ^ { \star } \geq \underline { { f } } _ { h }$ and hence belongs to $V _ { h }$ . If $x _ { c } \in \arg \operatorname* { m a x } _ { \chi } u _ { h }$ then $u _ { h } ( x _ { c } ) = \overline { { f } } _ { h } \geq \underline { { f } } _ { h } , \mathrm { { s o } } ~ x _ { c } \in V _ { h }$ , and the width bound at $x _ { c }$ gives

$$
\underline { { f } } _ { h } \leq f ^ { \star } \leq \overline { { f } } _ { h } , \overline { { f } } _ { h } \leq \ell _ { h } ( x _ { c } ) + 2 \varepsilon _ { h } \leq \underline { { f } } _ { h } + 2 \varepsilon _ { h } .\tag{B.53}
$$

Moreover, $\widehat { x } _ { h } \in \arg \operatorname* { m a x } _ { \chi } \ell _ { h }$ satisfies $f ( \widehat { x } _ { h } ) \geq \ell _ { h } ( \widehat { x } _ { h } ) = \underline { { f } } _ { h } \geq f ^ { \star } - 2 \varepsilon _ { h }$ . This proves (53). For (54), Proposition A.1, applied with $\begin{array} { r } { L = \ell _ { h } , U = u _ { h } , \underline { { f } } = \underline { { f } } _ { h } , } \end{array}$ and ${ \overline { { f } } } = { \overline { { f } } } _ { h } ,$ gives $I _ { h , \tau } \subseteq S _ { \tau } \subseteq O _ { h , \tau }$ by (A.2). If $x \in O _ { h , \tau } \setminus I _ { h , \tau }$ , then $u _ { h } ( x ) > \underline { { f } } _ { h } - \tau$ and $\ell _ { h } ( x ) \leq \overline { { f } } _ { h } - \tau .$ , so $x \in C _ { h }$ and $u _ { h } ( x ) - \ell _ { h } ( x ) \leq 2 \varepsilon _ { h }$ Hence

$$
\begin{array} { r } { \underline { { f } } _ { h } - \tau - 2 \varepsilon _ { h } < u _ { h } ( x ) - 2 \varepsilon _ { h } \leq \ell _ { h } ( x ) \leq f ( x ) \leq u _ { h } ( x ) \leq \ell _ { h } ( x ) + 2 \varepsilon _ { h } \leq \overline { { f } } _ { h } - \tau + 2 \varepsilon _ { h } , } \end{array}
$$

and (B.53) gives $f ^ { \star } - \tau - 4 \varepsilon _ { h } < f ( x ) \le f ^ { \star } - \tau + 4 \varepsilon _ { h }$ . This proves the second inclusion in (54). Only the points of $O _ { h , \tau } \backslash I _ { h , \tau }$ , rather than the whole outer set, require this width control.

We now prove the properties of the sets $Q _ { h }$ stated before the proof. $\mathrm { A }$ point x is sampled in phase h only if $a _ { h , n } ( x ) > 2 \varepsilon _ { h }$ , so $G _ { h } ( x ) > 2 \varepsilon _ { h } .$ which gives $g _ { h - 1 } ( x ) > - 2 \varepsilon _ { h }$ and $x \in Q _ { h }$ . Since $g _ { h } \le g _ { h - 1 }$ and $\varepsilon _ { h + 1 } < \varepsilon _ { h }$ , we have ${ \cal Q } _ { h + 1 } = \{ g _ { h } \geq - 2 \varepsilon _ { h + 1 } \} \subseteq \{ g _ { h - 1 } \geq - 2 \varepsilon _ { h } \} = { \cal Q } _ { h }$ . Moreover, if x /∈ $Q _ { h }$ , then $g _ { j } ( x ) \leq g _ { h - 1 } ( x ) < - 2 \varepsilon _ { h } < 0$ for all $j \geq h - 1$ , so x does not belong to any later set $\{ g _ { j } \geq 0 \}$ . Any global maximizer $x ^ { \star }$ satisfies $u _ { h - 1 } ( x ^ { \star } ) - \underline { { { f } } } _ { h - 1 } \geq f ^ { \star } - \underline { { { f } } } _ { h - 1 } \geq 0 .$ , so $g _ { h - 1 } ( x ^ { \star } ) \geq 0$ and $x ^ { \star } \in Q _ { h }$ . For $h = 0$ , the initial bounds $u = B$ and $f = - B { \mathrm { ~ g i v e ~ } } Q _ { 0 } = \mathcal { X }$ , and $f ^ { \star } - f ( x ) \leq 2 B = \varepsilon _ { 0 } \leq L _ { \vartheta } \varepsilon _ { 0 }$ for all $x \in \mathcal { X }$

For $h \geq 1$ , let $x \in Q _ { h }$ . Since $\varepsilon _ { h - 1 } = \vartheta \varepsilon _ { h }$ and $g _ { h - 2 } \geq g _ { h - 1 }$ , the screening function of phase $h - 1$ satisfies

$$
G _ { h - 1 } ( x ) \geq g _ { h - 2 } ( x ) + 4 \varepsilon _ { h - 1 } \geq g _ { h - 1 } ( x ) + 4 \vartheta \varepsilon _ { h } \geq ( 4 \vartheta - 2 ) \varepsilon _ { h } > 2 \varepsilon _ { h - 1 } .
$$

Hence (B.52) for the completed phase $h - 1$ gives $u _ { h - 1 } ( x ) - \ell _ { h - 1 } ( x ) \le 2 \varepsilon _ { h - 1 } = 2 \vartheta \varepsilon _ { h }$ , and (B.53) for phase $h - 1$ gives $f ^ { \star } - \underline { { f } } _ { h - 1 } \leq 2 \vartheta \varepsilon _ { h }$ and $\overline { { f } } _ { h - 1 } - f ^ { \star } \leq 2 \vartheta \varepsilon _ { h }$ . Since $g _ { h - 1 } ( x ) \geq - 2 \varepsilon _ { h }$ , one of the two terms in the maximum in (51) is at least $- 2 \varepsilon _ { h }$

• If $u _ { h - 1 } ( x ) - \underline { { f } } _ { h - 1 } \geq - 2 \varepsilon _ { h }$ , then

$$
\begin{array} { r } { f ^ { \star } - f ( x ) \leq \big ( f ^ { \star } - \underline { { f } } _ { h - 1 } \big ) + \big ( \underline { { f } } _ { h - 1 } - u _ { h - 1 } ( x ) \big ) + \big ( u _ { h - 1 } ( x ) - f ( x ) \big ) \leq ( 4 \vartheta + 2 ) \varepsilon _ { h } . } \end{array}
$$

• Otherwise, min $\{ \overline { { f } } _ { h - 1 } - \tau - \ell _ { h - 1 } ( x ) , u _ { h - 1 } ( x ) - \underline { { f } } _ { h - 1 } + \tau \} \ge - 2 \varepsilon _ { h }$ , which gives

$$
\begin{array} { r } { \underline { { f } } _ { h - 1 } - \tau - 2 \varepsilon _ { h } \le u _ { h - 1 } ( x ) , \ell _ { h - 1 } ( x ) \le \overline { { f } } _ { h - 1 } - \tau + 2 \varepsilon _ { h } . } \end{array}\tag{B.54}
$$

Combining (B.54) with $u _ { h - 1 } ( x ) - \ell _ { h - 1 } ( x ) \leq 2 \vartheta \varepsilon _ { h }$ and with the bounds on $f ^ { \star } - \underline { { f } } _ { h - 1 }$ and ${ \overline { { f } } } _ { h - 1 } - f ^ { \star }$ gives

$$
\begin{array} { r } { f ( \boldsymbol { x } ) \ge u _ { h - 1 } ( \boldsymbol { x } ) - 2 \vartheta \varepsilon _ { h } \ge \underline { { f } } _ { h - 1 } - \tau - ( 2 \vartheta + 2 ) \varepsilon _ { h } \ge f ^ { \star } - \tau - ( 4 \vartheta + 2 ) \varepsilon _ { h } , } \end{array}
$$

$$
f ( x ) \leq \ell _ { h - 1 } ( x ) + 2 \vartheta \varepsilon _ { h } \leq \overline { { f } } _ { h - 1 } - \tau + ( 2 \vartheta + 2 ) \varepsilon _ { h } \leq f ^ { \star } - \tau + ( 4 \vartheta + 2 ) \varepsilon _ { h } ,
$$

that is, $| f ^ { \star } - f ( x ) - \tau | \leq ( 4 \vartheta + 2 ) \varepsilon _ { h }$

Since $4 \vartheta + 2 \leq L _ { \vartheta } = 4 ( \vartheta + 1 )$ , this proves (B.49).

We next bound the number $n _ { h }$ of evaluations in phase $h ,$ including an unfinished phase. Lemma B.3, applied with $G = G _ { h }$ and $W = W _ { h - 1 }$ , shows that all points sampled in phase h lie in $D _ { h } = \{ x \in \mathcal { X } : G _ { h } ( x ) \geq 2 \varepsilon _ { h } , W _ { h - 1 } ( x ) \geq 2 \varepsilon _ { h } \}$ , and $D _ { h } \subseteq Q _ { h }$ because $G _ { h } ( x ) \geq 2 \varepsilon _ { h }$ implies $g _ { h - 1 } ( x ) \geq - 2 \varepsilon _ { h }$ . Hence (B.12) and the monotonicity of the information gain in the set give $n _ { h } \varepsilon _ { h } ^ { 2 } \leq 4 \beta _ { N } ^ { 2 } C _ { \lambda } \gamma _ { n _ { h } } ( Q _ { h } ; \lambda )$ , and (B.13) applies to any cover of $Q _ { h }$ . The case $n _ { h } = 0$ is trivial. For $h \geq 1$ , put $v = \varepsilon _ { h - 1 } = \vartheta \varepsilon _ { h }$ . Then $L _ { \vartheta } \varepsilon _ { h } = 4 ( 1 + \vartheta ^ { - 1 } ) v < 8 v$ , so (B.49) gives

$$
Q _ { h } \subseteq S _ { ( 8 v ) + } \cup \{ x : | f ^ { \star } - f ( x ) - \tau | \leq 8 v \} ,
$$

and, by (59), this set is covered by $J = J _ { \tau , f } ( v )$ balls of radius $r = c _ { r } v ^ { 1 / \nu }$ . Applying (B.13) with this cover, as in the derivation of (B.16), and using $r ^ { d } = c _ { r } ^ { d } v ^ { d / \nu } , v = \vartheta \varepsilon _ { h } , J \geq 1$ , and $v \leq 2 B$ , we obtain

$$
n _ { h } \le C \Lambda _ { N , \delta } \big [ 1 + J \varepsilon _ { h } ^ { - 2 } + J r ^ { d } \varepsilon _ { h } ^ { - ( 2 + d / \nu ) } \big ] = C \Lambda _ { N , \delta } \big [ 1 + ( \vartheta ^ { 2 } + c _ { r } ^ { d } \vartheta ^ { 2 + d / \nu } ) J \upsilon ^ { - 2 } \big ] \le C \Lambda _ { N , \delta } J _ { \tau , f } ( \upsilon ) \upsilon ^ { - 2 } .
$$

In addition, (B.15) gives $n _ { h } \le C \Lambda _ { N , \delta } ( 1 + \varepsilon _ { h } ^ { - ( 2 + d / \nu ) } ) \le C \Lambda _ { N , \delta } v ^ { - ( 2 + d / \nu ) }$ . Hence

$$
n _ { h } \le C \Lambda _ { N , \delta } \operatorname* { m i n } \{ v ^ { - ( 2 + d / \nu ) } , v ^ { - 2 } J _ { \tau , f } ( v ) \} = C \Lambda _ { N , \delta } \xi _ { \tau , f } ( \varepsilon _ { h - 1 } ) , h \ge 1 .\tag{B.55}
$$

The initial phase has $n _ { 0 } = 0$ , because $a _ { 0 , 0 } \leq W _ { - 1 } = 2 B = \varepsilon _ { 0 } \leq 2 \varepsilon _ { 0 }$ , so the stopping rule holds at its start.

Summing (B.55) over $h \leq H$ and adding the nonnegative term $\xi _ { \tau , f } ( \varepsilon _ { H } )$ gives

$$
\sum _ { h = 0 } ^ { H } { n _ { h } } \le C \Lambda _ { N , \delta } \sum _ { h = 1 } ^ { H } \xi _ { \tau , f } ( \varepsilon _ { h - 1 } ) \le C \Lambda _ { N , \delta } \sum _ { h = 0 } ^ { H } \xi _ { \tau , f } ( \varepsilon _ { h } ) .\tag{B.56}
$$

Since $\xi _ { \tau , f } ( u ) \le u ^ { - ( 2 + d / \nu ) }$ and $\varepsilon _ { h } \geq \varepsilon _ { H }$ for $h \leq H$ , (B.25) with $b = 3 + d / \nu$ gives

$$
\sum _ { h = 0 } ^ { H } \xi _ { \tau , f } ( \varepsilon _ { h } ) \le \sum _ { h = 0 } ^ { H } \varepsilon _ { h } ^ { - ( 2 + d / \nu ) } \le \frac { \varepsilon _ { H } ^ { - ( 2 + d / \nu ) } } { 1 - \vartheta ^ { - ( 2 + d / \nu ) } } \le C \varepsilon _ { H } ^ { - ( 2 + d / \nu ) } ,
$$

so (B.50) is at most a constant multiple of (55).

Finally, suppose that (B.50), with a suficiently large constant $C ,$ is at most N, but phase H does not complete on ${ \mathcal { E } } _ { \delta }$ . As in the proof of Theorem 5.1, on ${ \mathcal { E } } _ { \delta }$ the safeguard for inconsistent bounds is never triggered, and the refinement stop in Algorithm 1 does not apply to phases with $\varepsilon _ { h } \ge \varepsilon _ { H } \ge 2 \zeta _ { N }$ . Hence the budget is exhausted during some phase $h ^ { \prime } \leq H$ , and all N evaluations are made in phases $0 , 1 , \ldots , h ^ { \prime }$ . Then (B.56) gives $N \le C _ { 1 } \Lambda _ { N , \delta } \sum _ { h = 0 } ^ { H } \xi _ { \tau , f } ( \varepsilon _ { h } )$ for a constant $C _ { 1 } < C$ which contradicts the assumption. Therefore phase H completes. By the comparison with (55) above, this also proves the suficient budget (55). Therefore, we finish the proof. □

## B.13 Proof of Corollary 5.4

Proof of Corollary 5.4. Work on ${ \mathcal { E } } _ { \delta }$ , which has probability at least $1 - \delta$ by Lemma 5.1. Since (55) is at most N and $\varepsilon _ { H } \geq 2 \zeta _ { N }$ , Theorem 5.6 shows that phase H completes on ${ \mathcal { E } } _ { \delta }$ . By (53) with $h = H$ the final estimated optimal point $\widehat { x } _ { N } = \widehat { x } _ { H }$ satisfies $r _ { N } = f ^ { \star } - f ( \widehat { x } _ { H } ) \leq 2 \varepsilon _ { H } = 2 \varepsilon$

For the cumulative regret, let $n _ { h }$ be the number of evaluations in phase $h ,$ , and let $\begin{array} { r } { m _ { H } = \sum _ { h = 0 } ^ { H } n _ { h } } \end{array}$ All points x sampled in phase h lie in $Q _ { h }$ , so (B.49) gives $f ^ { \star } - f ( x ) \leq \tau + L \vartheta \varepsilon _ { h }$ . The remaining

$N - m _ { H }$ evaluations are made at $\widehat { x } _ { H }$ , whose gap is at most $2 \varepsilon$ . Hence

$$
R _ { N } \leq \sum _ { h = 0 } ^ { H } ( \tau + L _ { \vartheta } \varepsilon _ { h } ) n _ { h } + 2 \varepsilon ( N - m _ { H } ) .\tag{B.57}
$$

By (B.55), we have $n _ { 0 } = 0$ and $n _ { h } \le C \Lambda _ { N , \delta } \xi _ { \tau , f } ( \varepsilon _ { h - 1 } )$ for $h \geq 1$ . Since $\tau + L _ { \vartheta } \varepsilon _ { h } \leq L _ { \vartheta } \big ( \tau + \varepsilon _ { h - 1 } \big )$ substituting these bounds into (B.57), shifting the index by one, and adding the nonnegative term with $h = H { \mathrm { ~ g i v e } } .$ on ${ \mathcal { E } } _ { \delta }$

$$
R _ { N } \leq C \Lambda _ { N , \delta } \sum _ { h = 0 } ^ { H } ( \tau + \varepsilon _ { h } ) \xi _ { \tau , f } ( \varepsilon _ { h } ) + 2 N \varepsilon .\tag{B.58}
$$

Since $\xi _ { \tau , f } ( u ) \le u ^ { - ( 2 + d / \nu ) }$ and $\varepsilon _ { h } \geq \varepsilon$ for $h \leq H ,$ , (B.25) with $b = 3 + d / \nu$ and $b = 2 + d / \nu$ gives

$$
\sum _ { h = 0 } ^ { H } ( \tau + \varepsilon _ { h } ) \xi _ { \tau , f } ( \varepsilon _ { h } ) \le C \big [ \tau \varepsilon ^ { - ( 2 + d / \nu ) } + \varepsilon ^ { - ( 1 + d / \nu ) } \big ] = C ( \tau + \varepsilon ) \varepsilon ^ { - ( 2 + d / \nu ) } .\tag{B.59}
$$

Inserting (B.59) into (B.58) shows that, on $\mathcal { E } _ { \delta } , R _ { N } \leq C \Lambda _ { N , \delta } ( \tau + \varepsilon ) \varepsilon ^ { - ( 2 + d / \nu ) } + 2 N \varepsilon$ , which is at most the right-hand side of (58) without the term 2BNδ.

On $\mathcal { E } _ { \delta } ^ { c }$ , we have $r _ { N } \leq 2 B$ and $R _ { N } \le 2 B N$ . Since $\mathbb { P } ( \mathcal { E } _ { \delta } ^ { c } ) \le \delta$ , taking expectations gives (57) and (58). □

## B.14 Proof of Corollary 5.5

Proof of Corollary 5.5. Choose the target phase H with $\eta / ( 4 \vartheta ) < \varepsilon _ { H } \leq \eta / 4$ , so that the region estimation error bound $4 \varepsilon _ { H }$ in (54) is at most η. Let $u _ { 1 } > 0$ be such that $J _ { \tau , f } ( u ) \le C _ { J } u$ <sup>−κcov</sup> for $0 < u \leq u _ { 1 }$

We bound the refined budget (B.50). The phases with $\varepsilon _ { h } > u _ { 1 }$ are finitely many, and each satisfies $\xi _ { \tau , f } ( \varepsilon _ { h } ) \le \varepsilon _ { h } ^ { - ( 2 + d / \nu ) } \le u _ { 1 } ^ { - ( 2 + d / \nu ) }$ , so together they contribute at most $C \Lambda _ { N , \delta }$ to (B.50). For $\varepsilon _ { h } \leq u _ { 1 }$ , we have $\xi _ { \tau , f } ( \varepsilon _ { h } ) \leq \varepsilon _ { h } ^ { - 2 } \overset { \cdot } { J } _ { \tau , f } ( \varepsilon _ { h } ) \leq C _ { J } \varepsilon _ { h } ^ { - 2 - \kappa _ { \mathrm { c o v } } }$ , and (B.25) with $b = 3 + \kappa _ { \mathrm { c o v } }$ gives

$$
\sum _ { h \le H : \varepsilon _ { h } \le u _ { 1 } } \varepsilon _ { h } ^ { - 2 - \kappa _ { \mathrm { c o v } } } \le \frac { \varepsilon _ { H } ^ { - 2 - \kappa _ { \mathrm { c o v } } } } { 1 - \vartheta ^ { - 2 - \kappa _ { \mathrm { c o v } } } } .
$$

Hence (B.50) is at most $C \Lambda _ { N , \delta } ( 1 + \varepsilon _ { H } ^ { - 2 - \kappa _ { \mathrm { c o v } } } ) \leq C \Lambda _ { N , \delta } \eta ^ { - 2 }$ <sup>−κcov</sup> , where the last step uses $\varepsilon _ { H } > \eta / ( 4 \vartheta )$ Take N to be a suficiently large fixed multiple of $\eta ^ { - 2 - \kappa _ { \mathrm { c o v } } }$ times a suitable logarithmic factor, so that $C \Lambda _ { N , \delta } \eta ^ { - 2 - \kappa _ { \mathrm { c o v } } } \leq N$ and $\varepsilon _ { H } \geq 2 \zeta _ { N } ;$ the latter is possible because $\zeta _ { N } = \widetilde { O } ( N ^ { - 7 / 2 } )$ for fixed $\delta .$ Then phase H completes on $\mathcal { E } _ { \delta } .$ , as shown in the proof of Theorem 5.6, and (54) gives region estimation error at most $4 \varepsilon _ { H } \leq \eta$ with $N = \widetilde { \cal O } ( \eta ^ { - 2 - \kappa _ { \mathrm { c o v } } } )$ □

## B.15 Proof of Theorem 5.7

Proof of Theorem 5.7. The proof modifies the construction in the proof of Theorem 5.3 in Section B.5 with $\alpha = \nu$ . Two changes are needed. First, Lemma B.1 requires $\alpha > \nu ,$ so the baseline $- \mu _ { K } \| x - c \| ^ { \nu }$ is smoothed near c, and the RKHS norm of the smoothed baseline grows logarithmically. Second, the local peaks of the alternatives are quadratic, which is compatible with the covering condition because $\nu \geq 2$ . We first consider $K \ge C _ { \mathrm { c r i t } } [ \log ( 2 N ) ] ^ { d / \nu }$ , where $C _ { \mathrm { c r i t } }$ is specified in (B.69) below.

Let $\phi _ { 0 }$ and $w ( y ) = \{ \phi _ { 0 } ( y ) - \phi _ { 0 } ( 2 y ) \} \| y \| ^ { \nu }$ be the functions in the proof of Lemma B.1 in

Section C.2, with $\alpha = \nu .$ For $0 < h < 1 / 1 6$ , define

$$
\begin{array} { r } { q _ { h } ( x ) = [ 1 - \phi _ { 0 } ( x / ( 2 h ) ) ] \| x \| ^ { \nu } + \phi _ { 0 } ( x / ( 2 h ) ) h ^ { \nu - 2 } \| x \| ^ { 2 } , x \in \mathbb { R } ^ { d } . } \end{array}\tag{B.60}
$$

The function $q _ { h }$ is smooth, and $q _ { h } ( x ) = \| x \| ^ { \nu }$ for $\left\| x \right\| \geq 2 h$ . Moreover, $q _ { h } ( x ) \geq c _ { \nu } \| x \| ^ { \nu }$ for all $x ,$ where $c _ { \nu } = 2 ^ { 2 - \nu } \leq 1$ . Indeed, $q _ { h } ( x )$ is a convex combination of $\| x \| ^ { \nu }$ and $h ^ { \nu - 2 } \| x \| ^ { 2 }$ , and $\mathrm { i f ~ } \| x \| < 2 h$ then $h ^ { \nu - 2 } \| x \| ^ { 2 } = ( h / \| x \| ) ^ { \nu - 2 } \| x \| ^ { \nu } \geq 2 ^ { 2 - \nu } \| x \| ^ { \nu }$ because $\nu \geq 2$

Choose $c \in \mathcal X$ and $r _ { * } > 0$ with $\overline { { B } } ( c , 2 r _ { * } ) \subset \mathrm { i n t } ( \mathcal { X } )$ , and choose $\varphi _ { \mathcal { X } } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ with $\varphi _ { \mathcal { X } } = 1$ on a neighborhood of the compact set $\{ x - c : x \in \mathcal { X } \} \cup \overline { { B } } ( 0 , 1 )$ . We next show that

$$
\| \varphi \chi q _ { h } \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } \leq C [ 1 + \log ( 1 / h ) ] , 0 < h < 1 / 1 6 .\tag{B.61}
$$

Rewrite $\varphi x q _ { h } = \phi _ { 0 } q _ { h } + ( \varphi x - \phi _ { 0 } ) q _ { h }$ . Since $\varphi _ { \mathcal { X } } - \phi _ { 0 }$ vanishes on ${ \overline { { B } } } ( 0 , 1 / 2 )$ and $q _ { h } ( x ) = \| x \| ^ { \nu }$ for $\| x \| \geq 2 h$ , the second term equals $\left( \varphi _ { \mathcal { X } } - \phi _ { 0 } \right) \| \cdot \| ^ { \nu }$ , a fixed function in $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ . For the first term, let $j _ { h }$ be the integer with $r _ { j _ { h } } = 2 ^ { - j _ { h } } \in [ 4 h , 8 h )$ . Telescoping as in (C.1), but only up to level $j _ { h }$ , gives

$$
\phi _ { 0 } q _ { h } = \sum _ { j = 0 } ^ { j _ { h } - 1 } \bigl \{ \phi _ { 0 } ( 2 ^ { j } \cdot ) - \phi _ { 0 } ( 2 ^ { j + 1 } \cdot ) \bigr \} q _ { h } + \phi _ { 0 } ( 2 ^ { j _ { h } } \cdot ) q _ { h } = \sum _ { j = 0 } ^ { j _ { h } - 1 } 2 ^ { - j \nu } w ( 2 ^ { j } \cdot ) + \phi _ { 0 } ( 2 ^ { j _ { h } } \cdot ) q _ { h } .\tag{B.62}
$$

The second equality holds because the jth summand vanishes on $\overline { { B } } ( 0 , 2 ^ { - j - 2 } ) \supseteq \overline { { B } } ( 0 , r _ { j _ { h } } / 2 )$ , while $q _ { h } ( x ) = \| x \| ^ { \nu }$ for $\| x \| \geq 2 h$ and $r _ { j _ { h } } / 2 \geq 2 h$ . By (C.2) with $\alpha = \nu ,$ each term of the last sum has $H ^ { s }$ norm at most $\left\| w \right\| _ { H ^ { s } ( \mathbb R ^ { d } ) }$ . Unlike in the proof of Lemma B.1, these bounds are not summable, which is why the decomposition is truncated at level $j _ { h }$ . For the last term in (B.62), let $b = h / { r _ { j _ { h } } } \in ( 1 / 8 , 1 / 4 ]$ A change of variables in (B.60) gives $\phi _ { 0 } ( 2 ^ { j _ { h } } x ) q _ { h } ( x ) = { r _ { j _ { h } } } ^ { \nu } \Phi _ { b } ( x / r _ { j _ { h } } )$ , where

$$
\Phi _ { b } ( y ) = \phi _ { 0 } ( y ) \big \{ [ 1 - \phi _ { 0 } ( y / ( 2 b ) ) ] \| y \| ^ { \nu } + \phi _ { 0 } ( y / ( 2 b ) ) b ^ { \nu - 2 } \| y \| ^ { 2 } \big \} .
$$

Since $1 - \phi _ { 0 } ( y / ( 2 b ) )$ vanishes for $\| y \| \leq b$ and $b > 1 / 8$ , the functions $\Phi _ { b }$ with $b \in ( 1 / 8 , 1 / 4 ]$ are smooth, are supported in $\overline { { B } } ( 0 , 1 )$ , and have derivatives of all orders bounded uniformly in b. Hence their $H ^ { s }$ norms are uniformly bounded, and (B.3) with $t = 1 / r _ { j _ { h } }$ bounds the $H ^ { s }$ norm of the last term by a constant. Summing these bounds and using $j _ { h } \le \log _ { 2 } ( 1 / ( 4 h ) ) \le 2 \log ( 1 / h )$ proves (B.61).

Since $\varphi _ { \mathcal { X } } = 1$ on $\{ x - c : x \in \mathcal { X } \}$ , the function $( \varphi _ { \scriptstyle { \mathscr { X } } } q _ { h } ) ( \cdot - c )$ extends $x \mapsto q _ { h } ( x - c )$ from $\mathcal { X } .$ Hence $\left( \mathrm { B . 6 1 } \right)$ , the translation invariance of the $H ^ { s } ( \mathbb R ^ { d } )$ norm, and the norm comparison following (B.3) give $\| q _ { h } \big ( \cdot - c \big ) \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq C [ 1 + \log ( 1 / h ) ]$ . Set

$$
\mu _ { 0 } = c _ { \nu } ^ { - 1 } ( 4 / c _ { \rho } ) ^ { \nu } , \mu _ { K } = \mu _ { 0 } K ^ { - \nu / d } , g _ { K , h } ( x ) = - \mu _ { K } q _ { h } ( x - c ) .
$$

Then $g _ { K , h } \leq 0 , g _ { K , h } ( c ) = 0$ , and $\| g _ { K , h } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \le C \mu _ { 0 } K ^ { - \nu / d } [ 1 + \log ( 1 / h ) ]$ . Since $c _ { \nu } \mu _ { 0 } = ( 4 / c _ { \rho } ) ^ { \nu }$ , the ball $\overline { { B } } ( c , ( u / ( c _ { \nu } \mu _ { K } ) ) ^ { 1 / \nu } )$ has radius $( c _ { \rho } / 4 ) K ^ { 1 / d } u ^ { 1 / \nu }$ , so the elementary covering fact in the proof of Theorem 5.3 with $\alpha = \nu$ covers it by at most K balls of radius $c _ { \rho } u ^ { 1 / \nu }$

We now use the construction in the proof of Theorem 5.3 with $\alpha = \nu$ and $g _ { K , h }$ in place of $g _ { K }$ . For a small height parameter $\Delta > 0$ , let $R , h = ( 2 C _ { * } \Delta / B ) ^ { 1 / \nu } , z _ { 1 } , . . . , z _ { M }$ , and $A _ { i } = B ( z _ { i } , h )$ be as there, where the fixed constant $C _ { * }$ is specified in (B.66) below, and write $h = L \Delta ^ { 1 / \nu }$ with $L = ( 2 C _ { * } / B ) ^ { 1 / \nu }$ In place of (B.27), we require

$$
h \leq R / 3 2 , h < 1 / 1 6 .\tag{B.63}
$$

Since $( \Delta / \mu _ { K } ) ^ { 1 / \nu } = ( \Delta / \mu _ { 0 } ) ^ { 1 / \nu } K ^ { 1 / d }$ has the same order in $\Delta$ as $h ,$ the condition $h \le R / 3 2$ now requires K to be large. Indeed, (B.63) holds for any $K \ge K _ { 1 } = \operatorname* { m a x } \{ 1 , ( 3 2 L \mu _ { 0 } ^ { 1 / \nu } ) ^ { d } \}$ and $0 ~ <$

$\Delta \le \Delta _ { 1 }$ , where $\Delta _ { 1 } > 0$ is a fixed constant such that $L \Delta _ { 1 } ^ { 1 / \nu } \leq \operatorname* { m i n } \{ r _ { * } / 3 2 , 1 / 1 7 \}$ . The volume comparison leading to $\left( \mathrm { B . 2 8 } \right)$ gives $M \geq c _ { d } ( R / h ) ^ { d }$ , and $M \geq 4$ because $h \leq R / 3 2$ . Since $R / h =$ min $\{ r _ { * } L ^ { - 1 } \Delta ^ { - 1 / \nu } , L ^ { - 1 } \mu _ { 0 } ^ { - 1 / \nu } K ^ { 1 / d } \}$ ,

$$
M \ge c _ { M } \operatorname* { m i n } \{ K , \Delta ^ { - d / \nu } \} , M \ge 4 ,\tag{B.64}
$$

where $c _ { M } = c _ { d } L ^ { - d } \operatorname* { m i n } \{ r _ { * } ^ { d } , \mu _ { 0 } ^ { - d / \nu } \}$ is independent of K and $\Delta .$ . Moreover, any x in the closure of $A _ { i }$ satisfies $\| x - c \| \geq R / 2 - h \geq 1 5 h > 2 h$ , so $g _ { K , h } ( x ) = - \mu _ { K } \| x - c \| ^ { \nu }$

For $v = ( x - z _ { i } ) / h$ , we replace the local peak $4 \Delta - \Delta \| v \| ^ { \alpha }$ in (B.29) by a quadratic one and define

$$
f _ { i } ( x ) = [ 1 - \chi ( v ) ] g _ { K , h } ( x ) + \chi ( v ) [ 4 \Delta - \Delta \rvert \lvert v \rvert ] ^ { 2 } ] .\tag{B.65}
$$

With $z _ { i } - c = R a _ { i } , \bar { h } = h / R \leq 1 / 3 2$ , and $\bar { \mu } _ { K } = \mu _ { K } R ^ { \nu } / \Delta \leq 1$ , the modification is

$$
f _ { i } ( x ) - g _ { K , h } ( x ) = \Delta \chi ( v ) \left[ 4 - \| v \| ^ { 2 } + \bar { \mu } _ { K } \| a _ { i } + \bar { h } v \| ^ { \nu } \right] , 1 / 2 \leq \| a _ { i } \| \leq 3 / 4 .
$$

This is the modification in the proof of Theorem 5.3 with $\alpha = \nu ,$ except that $\| v \| ^ { \alpha }$ is replaced by the smooth function $\| \boldsymbol { v } \| ^ { 2 }$ . Hence the argument in the proof of Theorem $5 . 3 ,$ which now does not need Lemma B.1, together with (B.3) for $t = 1 / h$ , gives a fixed $C _ { * }$ such that

$$
\| f _ { i } - g _ { K , h } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \le C _ { * } \Delta h ^ { - \nu } = C _ { * } L ^ { - \nu } = B / 2 .\tag{B.66}
$$

Since $g _ { K , h } \leq 0$ , the maximum of $f _ { i }$ is $f _ { i } ^ { \star } = 4 \Delta$ , attained only at $z _ { i } .$ and, as in the proof of Theorem 5.3,

$$
f _ { i } ^ { \star } - f _ { i } ( x ) = [ 1 - \chi ( v ) ] \big ( 4 \Delta - g _ { K , h } ( x ) \big ) + \chi ( v ) \Delta \| v \| ^ { 2 } \ge \left\{ \begin{array} { l l } { \Delta \| v \| ^ { 2 } , } & { x \in A _ { i } , } \\ { 4 \Delta + c _ { \nu } \mu _ { K } \| x - c \| ^ { \nu } , } & { x \notin A _ { i } , } \end{array} \right.\tag{B.67}
$$

where the second case uses $q _ { h } ( x - c ) \geq c _ { \nu } \Vert x - c \Vert ^ { \nu }$ . We now verify the covering condition. For $u \geq \Delta$ the argument in the proof of Theorem 5.3, with $( u / \mu _ { K } ) ^ { 1 / \alpha }$ replaced by $( u / ( c _ { \nu } \mu _ { K } ) ) ^ { 1 / \nu } \geq R ,$ gives $S _ { u + } ( f _ { i } ) \subseteq \overline { { B } } ( c , ( u / ( c _ { \nu } \mu _ { K } ) ) ^ { 1 / \nu } )$ , which is covered by at most K balls of radius $c _ { \rho } u ^ { 1 / \nu }$ . For $0 < u < \Delta$ (B.67) shows that any point of $S _ { u + } ( f _ { i } )$ lies in $A _ { i }$ and satisfies $\Delta \| v \| ^ { 2 } \leq u$ . Hence

$$
\begin{array} { r } { S _ { u + } ( f _ { i } ) \subseteq \overline { { B } } \big ( z _ { i } , h ( u / \Delta ) ^ { 1 / 2 } \big ) = \overline { { B } } \big ( z _ { i } , L u ^ { 1 / \nu } ( u / \Delta ) ^ { 1 / 2 - 1 / \nu } \big ) \subseteq \overline { { B } } ( z _ { i } , L u ^ { 1 / \nu } ) , } \end{array}
$$

where the last inclusion uses $\nu \geq 2$ and $u < \Delta$ . The condition $h \Delta ^ { - 1 / \alpha } \leq c _ { \rho }$ in (B.27) is not available here, because $h \Delta ^ { - 1 / \nu } = L$ is fixed; instead, the maximal packing argument in the elementary covering fact covers $\overline { { B } } \big ( z _ { i } , L u ^ { 1 / \nu } \big )$ by at most $( 1 + 2 L / c _ { \rho } ) ^ { d }$ balls of radius $c _ { \rho } u ^ { 1 / \nu }$ Therefore, if $K _ { \mathrm { c r i t } } \geq \operatorname* { m a x } \{ K _ { 1 } , ( 1 + 2 L / c _ { \rho } ) ^ { d } \}$ , then $\mathcal { N } ( S _ { u + } ( f _ { i } ) , c _ { \rho } u ^ { 1 / \nu } ) \stackrel { \cdot } { \leq } K$ for all $u > 0$ and all $K \geq K _ { \mathrm { c r i t } }$

$$
\Delta = c _ { * } \operatorname* { m i n } \{ \sqrt { K / N } , N ^ { - 1 / ( 2 + d / \nu ) } \} ,\tag{B.68}
$$

where $c _ { * } ~ \in ~ ( 0 , 1 ]$ is a fixed constant specified below. Since $K \geq 1$ , we have $c _ { * } N ^ { - 1 / 2 } \leq \Delta \leq$ $c _ { * } N ^ { - 1 / ( 2 + d / \nu ) }$ . Hence there is a fixed $N _ { 1 } \geq 2$ such that $\Delta \le \Delta _ { 1 }$ for all $N \geq N _ { 1 }$ , uniformly over $K \geq 1$ . Moreover, $h = L \Delta ^ { 1 / \nu } \geq L c _ { * } ^ { 1 / \nu } N ^ { - 1 / ( 2 \nu ) }$ , so $1 + \log ( 1 / h ) \leq C \log ( 2 N )$ . Combining this with the bound on $\| g _ { K , h } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) }$ gives a fixed $C _ { q }$ such that

$$
\begin{array} { r } { \| g _ { K , h } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \le C _ { q } \mu _ { 0 } K ^ { - \nu / d } \log ( 2 N ) \le B / 2 \mathrm { ~ i f ~ } K \ge C _ { \mathrm { c r i t } } [ \log ( 2 N ) ] ^ { d / \nu } , C _ { \mathrm { c r i t } } = ( 2 C _ { q } \mu _ { 0 } / B ) ^ { d / \nu } . } \end{array}\tag{B.69}
$$

By (B.66) and (B.69), $\| f _ { i } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B$ . Together with the covering condition, this shows that $f _ { i } \in \mathcal { F } _ { K }$

for $i = 1 , \ldots , M .$

On $A _ { i } \subset B ( c , R )$ , the baseline satisfies $| g _ { K , h } ( x ) | = \mu _ { K } \| x - c \| ^ { \nu } \leq \mu _ { K } R ^ { \nu } \leq \Delta , \operatorname { s o } \ 0 \leq f _ { i } - g _ { K , h } \leq$ $5 \Delta \mathbf { 1 } _ { A _ { i } }$ by $\left( \mathrm { B . 6 5 } \right)$ . This is the only property of the alternatives used to derive $( \mathrm { B . 3 0 } ) { - } ( \mathrm { B . 3 3 } )$ , which therefore hold with $g _ { K }$ replaced by $g _ { K , h }$ , where $P _ { 0 }$ and $P _ { i }$ are the joint distributions of the sampling history and final output under $g _ { K , h }$ and $f _ { i }$ , and $N _ { i }$ is the number of evaluations allocated to $A _ { i }$ In particular, the average success probability $\begin{array} { r } { M ^ { - 1 } \sum _ { i = 1 } ^ { M } P _ { i } ( \widehat { x } _ { N } \in A _ { i } ) } \end{array}$ and the average fraction $\begin{array} { r } { M ^ { - 1 } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } N _ { i } / N } \end{array}$ are both at most $1 / M + ( 5 / ( 2 \sigma ) ) \sqrt { N \Delta ^ { 2 } / M }$ . By (B.64) and (B.68),

$$
\frac { N \Delta ^ { 2 } } { M } \leq c _ { M } ^ { - 1 } \operatorname* { m a x } \{ ( N / K ) \Delta ^ { 2 } , N \Delta ^ { 2 + d / \nu } \} \leq c _ { M } ^ { - 1 } \operatorname* { m a x } \{ c _ { * } ^ { 2 } , c _ { * } ^ { 2 + d / \nu } \} = c _ { M } ^ { - 1 } c _ { * } ^ { 2 } .\tag{B.70}
$$

Choose $c _ { * } = \operatorname* { m i n } \{ 1 , \sigma c _ { M } ^ { 1 / 2 } / 1 0 \}$ , so that $( 5 / ( 2 \sigma ) ) \sqrt { N \Delta ^ { 2 } / M } \leq 1 / 4$ by (B.70). Since $M \geq 4$ by (B.64), both averages are at most $3 / 4$ . By (B.67), the gap under $f _ { i }$ outside $A _ { i }$ is at least $4 \Delta$ . Hence, as at the end of the proof of Theorem 5.3,

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } r _ { N } \ge 4 \Delta \bigg ( 1 - \frac { 3 } { 4 } \bigg ) = \Delta , \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } R _ { N } \ge 4 \Delta \bigg ( N - \frac { 3 } { 4 } N \bigg ) = N \Delta .
$$

Let $N _ { \mathrm { c r i t } } \geq N _ { 1 }$ . Since the policy and the rule for the estimated optimal point were arbitrary, for any $K \ge \operatorname* { m a x } \{ K _ { \mathrm { c r i t } } , C _ { \mathrm { c r i t } } [ \log ( 2 N ) ] ^ { d / \nu } \}$ and $N \geq N _ { \mathrm { c r i t } }$ t

$$
\mathcal { R } _ { N } ^ { s } ( \mathcal { F } _ { K } ) \ge c _ { * } \operatorname* { m i n } \{ \sqrt { K / N } , N ^ { - 1 / ( 2 + d / \nu ) } \} , \mathcal { R } _ { N } ^ { c } ( \mathcal { F } _ { K } ) \ge c _ { * } \operatorname* { m i n } \{ \sqrt { K N } , N ^ { 1 - 1 / ( 2 + d / \nu ) } \} .\tag{B.71}
$$

This proves the claimed lower bounds without the logarithmic factor when $K \ge C _ { \mathrm { c r i t } } [ \log ( 2 N ) ] ^ { d / \nu }$ ， and hence also with ${ \mathrm { i t } } ,$ since $\log ( 2 N ) \geq 1$

It remains to consider $K _ { \mathrm { c r i t } } \le K < C _ { \mathrm { c r i t } } [ \log ( 2 N ) ] ^ { d / \nu }$ , for which we use a two-point construction that does not depend on K. Take $\psi _ { 1 } , \psi _ { 2 } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ taking values in [0, 1], with disjoint supports $S _ { 1 } , S _ { 2 } \subset \operatorname { i n t } ( \mathcal { X } )$ , such that, for $j = 1 , 2 , \psi _ { j }$ attains its maximum value one only at a point $y _ { j }$ and has a negative definite Hessian at $y _ { j }$ . Choose fixed $a _ { 0 } > 0$ and $\Delta _ { 0 } > 0$ small enough that the functions $\tilde { f } _ { j } = a _ { 0 } ( \psi _ { 1 } + \psi _ { 2 } ) + \Delta \psi _ { j } , j = 1 , 2$ , have RKHS norm at most B for all $0 < \Delta \le \Delta _ { 0 }$ . Let $\Psi = \psi _ { 1 } + \psi _ { 2 }$ The maximum of $\tilde { f } _ { j }$ is $\tilde { f } _ { j } ^ { \star } = a _ { 0 } + \Delta$ , attained only at $y _ { j }$ , and since $S _ { 1 } \cap S _ { 2 } = \emptyset$

$$
\tilde { f } _ { j } ^ { \star } - \tilde { f } _ { j } ( x ) = a _ { 0 } \{ 1 - \Psi ( x ) \} + \Delta \{ 1 - \psi _ { j } ( x ) \} \geq \operatorname* { m a x } \{ a _ { 0 } \{ 1 - \Psi ( x ) \} , \Delta \{ x \not \in S _ { j } \} \} .\tag{B.72}
$$

We verify that $\tilde { f } _ { 1 } , \tilde { f } _ { 2 } \in \mathcal { F } _ { K }$ after enlarging $K _ { \mathrm { c r i t } }$ . By Taylor’s theorem and the negative definite Hessians, there are fixed $r _ { 0 } , \kappa > 0$ such that $1 - \Psi ( x ) \geq \kappa \| x - y _ { j } \| ^ { 2 }$ for $\| x - y _ { j } \| \le r _ { 0 } , j = 1 , 2$ By continuity and compactness, there is a fixed $\kappa ^ { \prime } > 0$ such that $1 - \Psi ( x ) \geq \kappa ^ { \prime }$ for $x \in \mathcal { X }$ outside $B ( y _ { 1 } , r _ { 0 } ) \cup B ( y _ { 2 } , r _ { 0 } )$ . Let $u _ { 1 } = \mathrm { m i n } \{ 1 , a _ { 0 } \kappa ^ { \prime } \}$ and $L _ { 1 } = ( a _ { 0 } \kappa ) ^ { - 1 / 2 }$ . For $0 < u < u _ { 1 }$ , (B.72) gives

$$
S _ { u + } ( \tilde { f } _ { j } ) \subseteq \overline { { \cal B } } ( y _ { 1 } , L _ { 1 } \sqrt { u } ) \cup \overline { { \cal B } } ( y _ { 2 } , L _ { 1 } \sqrt { u } ) ,
$$

and $L _ { 1 } \sqrt { u } \le L _ { 1 } u ^ { 1 / \nu }$ since $\nu \geq 2$ and $u < 1$ . Hence at most $2 ( 1 + 2 L _ { 1 } / c _ { \rho } ) ^ { d }$ balls of radius $c _ { \rho } u ^ { 1 / \nu }$ cover $S _ { u + } ( \tilde { f } _ { j } )$ . For $u \geq u _ { 1 }$ , the domain X is covered by $\mathcal { N } ( \mathcal { X } , c _ { \rho } u _ { 1 } ^ { 1 / \nu } )$ balls of radius $c _ { \rho } u _ { 1 } ^ { 1 / \nu } \leq c _ { \rho } u ^ { 1 / \nu }$ and this number is fixed. Enlarging $K _ { \mathrm { c r i t } }$ to at least max $\{ 2 ( 1 + 2 L _ { 1 } / c _ { \rho } ) ^ { d } , \mathcal { N } ( \mathcal { X } , c _ { \rho } u _ { 1 } ^ { 1 / \nu } ) \}$ , we obtain $\tilde { f } _ { 1 } , \tilde { f } _ { 2 } \in \mathcal { F } _ { K }$ for all $K \geq K _ { \mathrm { c r i t } }$ and $0 < \Delta \le \Delta _ { 0 }$

Set $\Delta = \sigma N ^ { - 1 / 2 }$ , and enlarge $N _ { \mathrm { c r i t } }$ so that $\Delta \le \Delta _ { 0 }$ for $N \geq N _ { \mathrm { c r i t } }$ . Let $P _ { 1 } , P _ { 2 }$ denote the joint distributions of the sampling history and final output under $\tilde { f } _ { 1 } , \tilde { f } _ { 2 }$ , and let $N _ { j } ^ { \prime }$ be the number of evaluations allocated to $S _ { j }$ . Since $| \tilde { f } _ { 1 } - \tilde { f } _ { 2 } | = \Delta | \psi _ { 1 } - \psi _ { 2 } | \leq \Delta$ , the computation leading to (B.30)

gives $\mathrm { K L } ( P _ { 1 } , P _ { 2 } ) \le N \Delta ^ { 2 } / ( 2 \sigma ^ { 2 } ) = 1 / 2$ , and hence $\mathrm { T V } ( P _ { 1 } , P _ { 2 } ) \leq 1 / 2$ by Pinsker’s inequality. Since $S _ { 1 } \cap S _ { 2 } = \emptyset$ , applying the first inequality in (B.31) to $P _ { 1 } , P _ { 2 }$ with $Z = { \bf 1 } \{ \widehat { x } _ { N } \in S _ { 2 } \}$ and with $Z = N _ { 2 } ^ { \prime } / N$ gives

$$
\frac { 1 } { 2 } \sum _ { j = 1 } ^ { 2 } P _ { j } ( \widehat { x } _ { N } \in S _ { j } ) \leq \frac { 1 } { 2 } \big [ 1 + \mathrm { T V } ( P _ { 1 } , P _ { 2 } ) \big ] \leq \frac { 3 } { 4 } , \frac { 1 } { 2 } \sum _ { j = 1 } ^ { 2 } \frac { \mathbb { E } _ { j } N _ { j } ^ { \prime } } { N } \leq \frac { 1 } { 2 } \big [ 1 + \mathrm { T V } ( P _ { 1 } , P _ { 2 } ) \big ] \leq \frac { 3 } { 4 } .
$$

By (B.72), the gap under $\tilde { f } _ { j }$ is at least $\Delta$ outside $S _ { j }$ . Hence, as at the end of the proof of Theorem 5.3,

$$
\frac { 1 } { 2 } \sum _ { j = 1 } ^ { 2 } \mathbb { E } _ { j } r _ { N } \geq \Delta \Big ( 1 - \frac { 3 } { 4 } \Big ) = \frac { \sigma } { 4 \sqrt { N } } , \frac { 1 } { 2 } \sum _ { j = 1 } ^ { 2 } \mathbb { E } _ { j } R _ { N } \geq \Delta \Big ( N - \frac { 3 } { 4 } N \Big ) = \frac { \sigma \sqrt { N } } { 4 } .
$$

If $K < C _ { \mathrm { c r i t } } [ \log ( 2 N ) ] ^ { d / \nu }$ , then $\mathrm { [ l o g ( 2 \it N ) ] ^ { - d / ( 2 \nu ) } } \sqrt { K / N } < \sqrt { C _ { \mathrm { c r i t } } / N }$ and $[ \log ( 2 N ) ] ^ { - d / ( 2 \nu ) } \sqrt { K N } <$ $\sqrt { C _ { \mathrm { c r i t } } N }$ . Since the policy and the rule for the estimated optimal point were arbitrary, the claimed lower bounds therefore hold with the constant min $\{ c _ { * } , \sigma / ( 4 \sqrt { C _ { \mathrm { c r i t } } } ) \}$ } for all $K \geq K _ { \mathrm { c r i t } }$ and $N \geq N _ { \mathrm { c r i t } }$

Finally, suppose that ν is an even integer. Then $\| x \| ^ { \nu } = ( x _ { 1 } ^ { 2 } + \cdot \cdot \cdot + x _ { d } ^ { 2 } ) ^ { \nu / 2 }$ is a polynomial, and we replace $q _ { h }$ by $q ( x ) = \| x \| ^ { \nu }$ throughout. The function $\varphi _ { \mathcal { X } } q$ is a fixed element of $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ , so $\| q ( \cdot - c ) \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq C$ independently of $h ,$ and (B.69) is replaced by $\| \mu _ { K } q ( \cdot - c ) \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq C \mu _ { 0 } K ^ { - \nu / d } \leq$ $B / 2$ , which holds for any $K \geq K _ { \mathrm { c r i t } }$ after enlarging $K _ { \mathrm { c r i t } }$ . The other steps use only $q ( x ) \geq c _ { \nu } \| x \| ^ { \nu }$ and $q ( x ) = \| x \| ^ { \nu }$ for $\| x \| \geq 2 h$ , which hold since $c _ { \nu } \leq 1$ . Hence (B.71) holds for any $K \geq K _ { \mathrm { c r i t } }$ and $N \geq N _ { \mathrm { c r i t } }$ , which proves the claimed lower bounds without the logarithmic factor. □

## B.16 Proof of Proposition 5.3

Proof of Proposition 5.3. The proof uses a periodic baseline whose near-optimal sets are unions of small balls around the points of a lattice, and it places the alternatives at these points. In addition to $C _ { i }$ , we let c denote a positive constant depending only on the fixed parameters, whose value may change between occurrences.

For an integer $m \geq 1 ,$ , put

$$
A _ { m } = A _ { 0 } m ^ { - s } , g _ { m } ( x ) = - A _ { m } \sum _ { r = 1 } ^ { d } \sin ^ { 2 } ( \pi m x _ { r } ) , x \in \mathcal { X } ,\tag{B.73}
$$

where $A _ { 0 } > 0$ is a fixed constant specified below. We first bound the RKHS norm of $g _ { m }$ . Extend $g _ { m }$ to $\mathbb { R } ^ { d }$ by the same formula, and let $\varphi _ { \mathcal { X } } \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ be a fixed function with $\varphi _ { \mathcal { X } } = 1$ on a neighborhood of $\mathcal { X }$ . For any $\omega \in \mathbb { R } ^ { d }$ , the Fourier transform of $x \mapsto \varphi _ { \mathcal { X } } ( x ) e ^ { i \omega ^ { \top } x } ~ \mathrm { i s } ~ \widehat { \varphi _ { \mathcal { X } } } ( \cdot - \omega )$ . Hence, by the inequality $1 + \| v + \omega \| ^ { 2 } \leq 2 ( 1 + \| v \| ^ { 2 } ) ( 1 + \| \omega \| ^ { 2 } )$ 2

$$
\| \varphi \chi e ^ { i \omega ^ { \top } \cdot } \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } \leq 2 ^ { s / 2 } ( 1 + \| \omega \| ^ { 2 } ) ^ { s / 2 } \| \varphi \chi \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } \leq C ( 1 + \| \omega \| ) ^ { s } .\tag{B.74}
$$

Since $\sin ^ { 2 } ( \pi m t ) = \{ 2 - e ^ { 2 \pi i m t } - e ^ { - 2 \pi i m t } \} / 4 .$ , the function $g _ { m }$ is the constant − $\cdot d A _ { m } / 2$ plus 2d harmonics with frequencies ±2πme<sub>r</sub>, $r = 1 , \ldots , d ,$ each with coeficient $A _ { m } / 4$ . By (B.74), $\| \varphi _ { \mathcal { X } } g _ { m } \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } \leq$ $C A _ { m } ( 1 + 2 \pi m ) ^ { s } \ \leq \ C A _ { m } m ^ { s } \ = \ C A _ { 0 }$ . Since $\varphi _ { \mathcal { X } } g _ { m }$ extends $g _ { m }$ from $\mathcal { X } _ { : }$ , the norm comparison following (B.3) gives a fixed $C _ { 0 }$ such that $\lVert g _ { m } \rVert _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq C _ { 0 } A _ { 0 }$ . We set $A _ { 0 } = B / ( 2 C _ { 0 } )$ , so that $\| g _ { m } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B / 2$ for any $m \geq 1$

We next show that the near-optimal sets of $g _ { m }$ satisfy the covering condition with K replaced by a fixed multiple of $m ^ { s d / \alpha }$ . Since X has nonempty interior, there is a fixed $m _ { 0 }$ such that $\mathcal { X }$ contains a point of the lattice $m ^ { - 1 } \mathbb { Z } ^ { d }$ for $m \geq m _ { 0 }$ , and hence $g _ { m } ^ { \star } = 0$ . Let $0 < u < A _ { m }$ and $x \in S _ { u + } ( g _ { m } )$ The inequality sin $\operatorname { \mathrm { \Omega } } ^ { 2 } ( \pi t ) \geq 4 \operatorname { d i s t } ( t , \mathbb { Z } ) ^ { 2 }$ gives

$$
u \geq - g _ { m } ( x ) = A _ { m } \sum _ { r = 1 } ^ { d } \sin ^ { 2 } ( \pi m x _ { r } ) \geq 4 A _ { m } \sum _ { r = 1 } ^ { d } \mathrm { d i s t } ( m x _ { r } , \mathbb { Z } ) ^ { 2 } = 4 A _ { m } m ^ { 2 } \mathrm { d i s t } ( x , m ^ { - 1 } \mathbb { Z } ^ { d } ) ^ { 2 } .
$$

Hence x lies within distance $R _ { u } = \sqrt { u } / ( 2 m \sqrt { A _ { m } } ) < 1 / ( 2 m )$ of a point of $m ^ { - 1 } \mathbb { Z } ^ { d }$ . Since $\mathcal { X }$ is bounded, at most $C _ { X } m ^ { d }$ lattice points lie within distance $1 / ( 2 m )$ of X. By the maximal packing argument in the elementary covering fact in the proof of Theorem 5.3, each ball of radius $R _ { u }$ is covered by at most $( 1 + 2 \bar { R _ { u } } / ( c _ { \rho } u ^ { 1 / \alpha } ) ) ^ { d } \leq C [ 1 + ( \bar { R } _ { u } / ( c _ { \rho } u ^ { 1 / \alpha } ) ) ^ { d } ]$ balls of radius $c _ { \rho } u ^ { 1 / \alpha }$ . Therefore,

$$
\begin{array} { r l } & { \mathcal { N } ( S _ { u + } ( g _ { m } ) , c _ { \rho } u ^ { 1 / \alpha } ) \leq C C _ { \mathcal { X } } m ^ { d } \left[ 1 + \frac { u ^ { d / 2 } } { 2 ^ { d } m ^ { d } A _ { m } ^ { d / 2 } c _ { \rho } ^ { d } u ^ { d / \alpha } } \right] \leq C \left[ m ^ { d } + A _ { m } ^ { - d / 2 } u ^ { d / 2 - d / \alpha } \right] } \\ & { \qquad \leq C \left[ m ^ { d } + A _ { m } ^ { - d / \alpha } \right] = C \left[ m ^ { d } + A _ { 0 } ^ { - d / \alpha } m ^ { s d / \alpha } \right] , } \end{array}\tag{B.75}
$$

where the second line uses $u ^ { d / 2 - d / \alpha } \leq A _ { m } ^ { d / 2 - d / \alpha }$ , which holds since $\alpha \geq 2$ and $u < A _ { m }$ , and then $A _ { m } = A _ { 0 } m ^ { - s }$ . For $u \geq A _ { m }$ , the bounded domain X is covered by at most $C ( 1 + A _ { m } ^ { - d / \alpha } )$ balls of radius $c _ { \rho } A _ { m } ^ { 1 / \alpha } \leq c _ { \rho } u ^ { 1 / \alpha }$ , which is bounded by the last expression in (B.75) after increasing C. Since $s > \nu \geq \alpha$ , we have $m ^ { d } \leq m ^ { s d / \alpha }$ . Hence there is a fixed $C _ { \mathrm { c o v } } \geq 1$ such that, for any $m \geq m _ { 0 }$ J

$$
\mathcal { N } ( S _ { u + } ( g _ { m } ) , c _ { \rho } u ^ { 1 / \alpha } ) \leq C _ { \mathrm { c o v } } m ^ { s d / \alpha } , u > 0 .\tag{B.76}
$$

We now construct the alternatives. Fix a closed ball $\overline { { B } } ( z _ { 0 } , r _ { * } ) \subset \mathrm { i n t } ( \mathcal { X } )$ and a constant $0 < r _ { 1 } <$ $1 / 4$ . After enlarging $m _ { 0 }$ , for any $m \geq m _ { 0 }$ , the ball $\overline { { B } } ( z _ { 0 } , r _ { * } )$ contains M points $z _ { 1 } , \dots , z _ { M }$ of $m ^ { - 1 } \mathbb { Z } ^ { d }$ with

$$
c _ { \mathcal { X } } m ^ { d } \leq M \leq C _ { \mathcal { X } } m ^ { d } ,\tag{B.77}
$$

and the balls ${ \cal B } _ { i } = { \cal B } ( z _ { i } , r _ { 1 } / m )$ are contained in X. These balls are disjoint, because distinct lattice points are at distance at least $1 / m > 2 r _ { 1 } / m$ . Let $\psi \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ be supported in the unit ball, with $0 \leq \psi \leq 1 , \psi ( 0 ) = 1$ , and $\psi ( v ) < 1$ for $v \neq 0$ . For a height parameter $\Delta > 0$ , define

$$
f _ { i } ( x ) = g _ { m } ( x ) + \Delta \psi ( m ( x - z _ { i } ) / r _ { 1 } ) .\tag{B.78}
$$

Since $g _ { m } \leq 0 = g _ { m } ( z _ { i } )$ , the unique maximizer of $f _ { i }$ is $z _ { i } .$ , with $f _ { i } ^ { \star } = \Delta$ , and

$$
\begin{array} { r } { f _ { i } ^ { \star } - f _ { i } ( x ) = - g _ { m } ( x ) + \Delta [ 1 - \psi ( m ( x - z _ { i } ) / r _ { 1 } ) ] \ge \operatorname* { m a x } \big \{ - g _ { m } ( x ) , \Delta \mathbf { 1 } \{ x \notin { \mathcal B _ { i } } \} \big \} . } \end{array}\tag{B.79}
$$

Since $g _ { m } ^ { \star } = 0 , ( \mathrm { B } . 7 9 )$ gives $S _ { u + } ( f _ { i } ) \subseteq S _ { u + } ( g _ { m } )$ for any $u > 0$ . Hence (B.76) holds with $g _ { m }$ replaced by $f _ { i } .$ , regardless of the ratio $\Delta / A _ { m }$

The scaling inequality (B.3) with $t = m / r _ { 1 } \ge 1$ and the norm comparison following it give a fixed $C _ { \psi }$ such that $\| f _ { i } - g _ { m } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq C _ { \psi } \Delta m ^ { \nu }$ . Set

$$
\Delta = \frac { \sigma } { 2 } \sqrt { \frac { M } { N } } .\tag{B.80}
$$

By (B.77), $\| f _ { i } - g _ { m } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \le C _ { \psi } \sigma C _ { \mathcal { X } } ^ { 1 / 2 } m ^ { s } / ( 2 \sqrt { N } )$ , which is at most $B / 2$ whenever

$$
m ^ { 2 s } \leq c _ { B } N , c _ { B } = \big \{ B / ( C _ { \psi } \sigma C _ { \chi } ^ { 1 / 2 } ) \big \} ^ { 2 } .\tag{B.81}
$$

Under $( \mathrm { B } . 8 1 ) , \| f _ { i } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B$ . Together with (B.76) for $f _ { i } ,$ , this shows that $f _ { i } \in \mathcal { F } _ { K }$ whenever $C _ { \mathrm { c o v } } m ^ { s d / \alpha } \leq K$

Let $P _ { 0 }$ and $P _ { i }$ denote the joint distributions of the sampling history and final output under $g _ { m }$ and $f _ { i } ,$ , respectively, and let $N _ { i }$ be the number of evaluations allocated to $B _ { i }$ . By (B.78), $0 \leq f _ { i } - g _ { m } \leq \Delta \mathbf { 1 } _ { B _ { i } }$ . Hence the computation leading to (B.30) gives

$$
\mathrm { K L } ( P _ { 0 } , P _ { i } ) \leq \frac { \Delta ^ { 2 } } { 2 \sigma ^ { 2 } } \mathbb { E } _ { 0 } N _ { i } .
$$

Since the $B _ { i }$ are disjoint, $\textstyle \sum _ { i } P _ { 0 } ( { \widehat { x } } _ { N } \in B _ { i } ) \leq 1$ and $\begin{array} { r } { \sum _ { i } \mathbb { E } _ { 0 } N _ { i } \leq N } \end{array}$ . Therefore, the arguments leading to (B.32) and (B.33), with $5 \Delta$ replaced by $\Delta .$ , give

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } P _ { i } ( \widehat { x } _ { N } \in \mathcal { B } _ { i } ) \leq \frac { 1 } { M } + \frac { \Delta } { 2 \sigma } \sqrt { \frac { N } { M } } = \frac { 1 } { M } + \frac { 1 } { 4 } , \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { \mathbb { E } _ { i } N _ { i } } { N } \leq \frac { 1 } { M } + \frac { 1 } { 4 } ,\tag{B.82}
$$

where the equality uses (B.80). If $M \geq 4$ , both averages in (B.82) are at most $1 / 2$ . By (B.79), any evaluated point or estimated optimal point outside $B _ { i }$ has gap at least $\Delta$ under $f _ { i }$ . Hence

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } r _ { N } \geq \Delta \Big ( 1 - \frac { 1 } { 2 } \Big ) = \frac { \Delta } { 2 } , \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } R _ { N } \geq \Delta \Big ( N - \frac { N } { 2 } \Big ) = \frac { N \Delta } { 2 } .
$$

It remains to choose $m$ . Let $c _ { \mathrm { p e r } } > 0$ be a fixed constant small enough that $C _ { \mathrm { c o v } } c _ { \mathrm { p e r } } ^ { s d / \alpha } \leq 1$ and $c _ { \mathrm { p e r } } ^ { 2 s } \leq c _ { B }$ , and take

$$
m = \left\lfloor c _ { \mathrm { p e r } } \operatorname* { m i n } \left\{ K ^ { \alpha / ( s d ) } , N ^ { 1 / ( 2 s ) } \right\} \right\rfloor .\tag{B.83}
$$

Then $C _ { \mathrm { c o v } } m ^ { s d / \alpha } \leq K$ and $m ^ { 2 s } \leq c _ { B } N$ , so $f _ { i } \in \mathcal { F } _ { K }$ for $i = 1 , \dots , M$ . Choose fixed $K _ { \mathrm { p e r } }$ and $N _ { \mathrm { p e r } }$ such that $K \geq K _ { \mathrm { p e r } }$ and $N \geq N _ { \mathrm { p e r } }$ imply $m \geq m _ { 0 }$ and $c \chi m ^ { d } \geq 4$ , so that $M \geq 4$ by (B.77). Since $\lfloor y \rfloor \ge y / 2$ for $y \geq 1$ , (B.77) and (B.83) give

$$
\sqrt { \frac { M } { N } } \ge \sqrt { \frac { c _ { \mathcal { X } } m ^ { d } } { N } } \ge c \operatorname* { m i n } \left\{ K ^ { \alpha / ( 2 s ) } N ^ { - 1 / 2 } , N ^ { d / ( 4 s ) - 1 / 2 } \right\} .
$$

Since $2 s = 2 \nu + d$ and $d / ( 4 s ) - 1 / 2 = - \nu / ( 2 \nu + d )$ , (B.80) gives

$$
\frac { \Delta } { 2 } \geq c \operatorname* { m i n } \{ K ^ { \alpha / ( 2 \nu + d ) } N ^ { - 1 / 2 } , N ^ { - \nu / ( 2 \nu + d ) } \} , \frac { N \Delta } { 2 } \geq c \operatorname* { m i n } \{ K ^ { \alpha / ( 2 \nu + d ) } N ^ { 1 / 2 } , N ^ { ( \nu + d ) / ( 2 \nu + d ) } \} .
$$

Since the policy and the rule for the estimated optimal point were arbitrary, taking the corresponding infima proves the claimed lower bounds. All class parameters and constants are fixed as $K$ and $N$ vary. □

## B.17 Proof of Proposition 5.4

Proof of Proposition ${ 5 . 4 } .$ We construct a family of compactly supported bumps of height $\Delta \asymp$ $N ^ { - 1 / ( 2 + d / \nu ) }$ and width $h \asymp \Delta ^ { 1 / \nu }$ , show that they all belong to $\mathcal { F } _ { K }$ once K exceeds the threshold in (60), and then apply the testing argument in the proof of Theorem 5.3 with the zero function as the reference.

Let

$$
\psi ( v ) = \left\{ \begin{array} { l l } { \exp \{ - \| v \| ^ { 2 } / ( 1 - \| v \| ^ { 2 } ) \} , } & { \| v \| < 1 , } \\ { 0 , } & { \| v \| \ge 1 . } \end{array} \right.
$$

This function belongs to $C _ { c } ^ { \infty } ( \mathbb { R } ^ { d } )$ and satisfies $\psi ( 0 ) = 1$ and $0 \leq \psi \leq 1$ . For $\| v \| < 1$ , we have $\psi ( v ) \leq \exp ( - \| v \| ^ { 2 } )$ and $1 - e ^ { - t } \geq ( 1 - e ^ { - 1 } ) t$ for $t \in [ 0 , 1 ]$ by the concavity of $t \mapsto 1 - e ^ { - t }$ , while $1 - \psi ( v ) = 1$ for $\lVert \boldsymbol { v } \rVert \geq 1$ . Hence,

$$
\begin{array} { r } { 1 - \psi ( v ) \geq c _ { \psi } \operatorname* { m i n } \{ \| v \| ^ { 2 } , 1 \} , c _ { \psi } = 1 - e ^ { - 1 } . } \end{array}\tag{B.84}
$$

The inequality (B.3) and the norm comparison following it give a fixed constant $C _ { \psi }$ such that, for any center $z \in \mathbb { R } ^ { d } , \Delta > 0$ , and $0 < h \leq 1$ ，

$$
\| \Delta \psi ( ( \cdot - z ) / h ) \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq C _ { \psi } \Delta h ^ { - \nu } .\tag{B.85}
$$

Set

$$
\Delta = c _ { * } N ^ { - 1 / ( 2 + d / \nu ) } , h = ( 2 C _ { \psi } \Delta / B ) ^ { 1 / \nu } ,\tag{B.86}
$$

where $c _ { * } \in ( 0 , 1 ]$ is a fixed constant specified below, so that $C _ { \psi } \Delta h ^ { - \nu } = B / 2$ . Fix a ball $B ( c _ { 0 } , r _ { * } )$ with $\overline { { B } } ( c _ { 0 } , r _ { * } ) \subset \mathcal { X }$ , and take a maximal 2h-separated set of centers $c _ { 1 } , \ldots , c _ { M }$ in $B ( c _ { 0 } , r _ { * } / 2 )$ . For $h < r _ { * } / 2$ , the open balls $A _ { i } = B ( c _ { i } , h )$ are disjoint and lie in $\mathcal { X } .$ . By maximality, the balls $B ( c _ { i } , 2 h )$ cover $B ( c _ { 0 } , r _ { * } / 2 )$ , so a volume comparison gives

$$
M \ge ( r _ { * } / ( 4 h ) ) ^ { d } = ( r _ { * } / 4 ) ^ { d } ( 2 C _ { \psi } / B ) ^ { - d / \nu } \Delta ^ { - d / \nu } .\tag{B.87}
$$

Hence there is a fixed $N _ { \mathrm { s a t } }$ <sub>t</sub> such that $h \leq \operatorname* { m i n } \{ 1 , r _ { * } / 4 \}$ and $M \geq 4$ for all $N \geq N _ { \mathrm { s a t } }$ . Define

$$
f _ { i } ( x ) = \Delta \psi ( ( x - c _ { i } ) / h ) , i = 1 , \dots , M .
$$

By (B.85), $\| f _ { i } \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \leq B / 2$ . The maximum of $f _ { i }$ is $f _ { i } ^ { \star } = \Delta$ , attained only at $c _ { i } .$ , and by (B.84),

$$
\begin{array} { r } { f _ { i } ^ { \star } - f _ { i } ( x ) = \Delta \{ 1 - \psi ( ( x - c _ { i } ) / h ) \} \geq c _ { \psi } \Delta \operatorname* { m i n } \{ \| x - c _ { i } \| ^ { 2 } / h ^ { 2 } , 1 \} , } \end{array}\tag{B.88}
$$

where the left-hand side equals $\Delta$ for x $\notin A _ { i }$

We next verify the covering condition for any $u > 0$ . Let $0 < u < \Delta$ . Since the gap equals $\Delta$ outside $A _ { i } ,$ we have $S _ { u + } ( f _ { i } ) \subseteq A _ { i }$ , and

$$
S _ { u + } ( f _ { i } ) \subseteq \overline { { B } } ( c _ { i } , R _ { u } ) , R _ { u } = h \sqrt { u / ( c _ { \psi } \Delta ) } .
$$

Indeed, if $u < c _ { \psi } \Delta$ , then (B.88) forces $\| x - c _ { i } \| ^ { 2 } / h ^ { 2 } \leq u / ( c _ { \psi } \Delta )$ for any $x \in \mathcal S _ { u + } ( f _ { i } )$ , and if $c _ { \psi } \Delta \le u < \Delta$ , then $A _ { i } \subseteq \overline { { B } } ( c _ { i } , R _ { u } )$ since $R _ { u } \ \ge \ h$ . By the maximal packing argument in the elementary covering fact in the proof of Theorem 5.3, a ball of radius R is covered by at most $( 1 + 2 R / r ) ^ { d } \leq C [ 1 + ( R / r ) ^ { d } ]$ balls of radius r. Since $\alpha \geq 2$ , the exponent $d / 2 - d / \alpha$ is nonnegative, and $u < \Delta$ gives $\stackrel { \cdot } { u } ^ { d / 2 - d / \alpha } \leq \Delta ^ { d / 2 - d / \alpha }$ . Therefore, for $0 < u < \Delta$

$$
\mathcal { N } ( S _ { u + } ( f _ { i } ) , c _ { \rho } u ^ { 1 / \alpha } ) \leq C \left[ 1 + \frac { h ^ { d } u ^ { d / 2 - d / \alpha } } { c _ { \psi } ^ { d / 2 } c _ { \rho } ^ { d } \Delta ^ { d / 2 } } \right] \leq C \big [ 1 + h ^ { d } \Delta ^ { - d / \alpha } \big ] \leq C \big [ 1 + \Delta ^ { d / \nu - d / \alpha } \big ] ,\tag{B.89}
$$

where the last step uses $h ^ { d } = ( 2 C _ { \psi } / B ) ^ { d / \nu } \Delta ^ { d / \nu }$ by (B.86). For $u \geq \Delta$ , the boundedness of $\mathcal { X }$ gives

$$
\begin{array} { r } { \mathcal { N } ( S _ { u + } ( f _ { i } ) , c _ { \rho } u ^ { 1 / \alpha } ) \leq \mathcal { N } ( \mathcal { X } , c _ { \rho } \Delta ^ { 1 / \alpha } ) \leq C \big [ 1 + \Delta ^ { - d / \alpha } \big ] . } \end{array}\tag{B.90}
$$

Since $\Delta \leq 1$ , the right-hand side of (B.89) is at most that of (B.90), which is at most $2 C \Delta ^ { - d / \alpha }$ Hence there is a fixed $C _ { \mathrm { s a t } }$ such that

$$
f _ { i } \in \mathcal { F } _ { K } \mathrm { ~ f o r ~ } i = 1 , \ldots , M , \mathrm { ~ w h e n e v e r ~ } K \ge C _ { \mathrm { s a t } } \Delta ^ { - d / \alpha } .\tag{B.91}
$$

Fix any policy using N evaluations and any rule for the estimated optimal point. Let $P _ { i }$ denote the joint distribution of the sampling history and final output under $f _ { i } ,$ and let $P _ { 0 }$ denote the corresponding distribution under the zero function, which serves only as a reference and need not belong to $\mathcal { F } _ { K }$ . Let $\begin{array} { r } { N _ { i } = \sum _ { t = 1 } ^ { N } \mathbf { 1 } \{ x _ { t } \in A _ { i } \} } \end{array}$ . Since $0 \le f _ { i } \le \Delta \mathbf { 1 } _ { A _ { i } }$ , the computation leading to (B.30) gives

$$
\mathrm { K L } ( P _ { 0 } , P _ { i } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \mathbb { E } _ { 0 } \sum _ { t = 1 } ^ { N } f _ { i } ( x _ { t } ) ^ { 2 } \leq \frac { \Delta ^ { 2 } } { 2 \sigma ^ { 2 } } \mathbb { E } _ { 0 } N _ { i } .
$$

Since the $A _ { i }$ are disjoint, $\textstyle \sum _ { i } P _ { 0 } ( { \widehat { x } } _ { N } \in A _ { i } ) \leq 1$ and $\begin{array} { r } { \sum _ { i } \mathbb { E } _ { 0 } N _ { i } \le N } \end{array}$ . Hence the arguments leading to (B.32) and (B.33), with $5 \Delta$ replaced by $\Delta .$ , give

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } P _ { i } ( \widehat { x } _ { N } \in A _ { i } ) \leq \frac { 1 } { M } + \frac { \Delta } { 2 \sigma } \sqrt { \frac { N } { M } } , \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { \mathbb { E } _ { i } N _ { i } } { N } \leq \frac { 1 } { M } + \frac { \Delta } { 2 \sigma } \sqrt { \frac { N } { M } } .\tag{B.92}
$$

By (B.87) and (B.86),

$$
\Delta \sqrt { \frac { N } { M } } \leq C \sqrt { N \Delta ^ { 2 + d / \nu } } = C c _ { * } ^ { 1 + d / ( 2 \nu ) } .
$$

Choose $c _ { * } \in ( 0 , 1 ]$ small enough that $C c _ { * } ^ { 1 + d / ( 2 \nu ) } / ( 2 \sigma ) \leq 1 / 4$ . Since $M \geq 4$ , both averages in (B.92) are then at most $1 / 2$ . Since the gap under $f _ { i }$ equals $\Delta$ outside $A _ { i }$

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } r _ { N } \geq \Delta \Big ( 1 - \frac { 1 } { 2 } \Big ) = \frac { \Delta } { 2 } , \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } _ { i } R _ { N } \geq \Delta \Big ( N - \frac { N } { 2 } \Big ) = \frac { N \Delta } { 2 } .
$$

Finally, by (B.86), the condition $K \ge C _ { \mathrm { s a t } } \Delta ^ { - d / \alpha }$ in (B.91) is $K \ge C _ { \mathrm { s a t } } c _ { * } ^ { - d / \alpha } N ^ { d / [ \alpha ( 2 + d / \nu ) ] }$ ， which is (60) with $\begin{array} { r } { C = C _ { \mathrm { s a t } } c _ { * } ^ { - d / \alpha } } \end{array}$ . Under this condition, $\Delta / 2 = ( c _ { * } / 2 ) N ^ { - 1 / ( 2 + d / \nu ) }$ and $N \Delta / 2 =$ $( c _ { * } / 2 ) N ^ { 1 - 1 / ( 2 + d / \nu ) }$ . Since the policy and the rule for the estimated optimal point were arbitrary, taking the corresponding infima proves the claimed lower bounds with $c = c _ { * } / 2$ . All constants and $N _ { \mathrm { s a t } }$ depend only on the fixed model and covering parameters. □

## B.18 Proof of Proposition A.1

Proof of Proposition A.1. The argument is deterministic and uses only the bounds $L \leq f \leq U$ on $\mathcal { X }$ and $\underline { { f } } \leq f ^ { \star } \leq \overline { { f } }$ . Recall that $S _ { v } = \{ x : f ^ { \star } - f ( x ) < v \}$

We first prove (A.2). Let $\tau \geq 0$ . If x $\in I _ { \tau }$ , then, by (A.1) and ${ \overline { { f } } } \geq f ^ { \star }$

$$
f ( x ) \geq L ( x ) > { \overline { { f } } } - \tau \geq f ^ { \star } - \tau ,
$$

so $f ^ { \star } - f ( x ) < \tau$ and $x \in S _ { \tau }$ . If $x \in S _ { \tau }$ , then, by $f ^ { \star } \geq \underline { { f } } ,$

$$
U ( x ) \geq f ( x ) > f ^ { \star } - \tau \geq \underline { { f } } - \tau ,
$$

so $x \in O _ { \tau }$ by (A.1).

We next prove (A.4), which implies (A.3). Let $x \in O _ { \tau } \backslash I _ { \tau }$ , so that $L ( x ) \leq { \overline { { f } } } - \tau$ and $U ( x ) > \underline { { f } } - \tau$ by (A.1). Since $w ( x ) = U ( x ) - L ( x )$

$$
f ( x ) \leq U ( x ) = L ( x ) + w ( x ) \leq { \overline { { f } } } - \tau + w ( x ) , f ( x ) \geq L ( x ) = U ( x ) - w ( x ) > f - \tau - w ( x ) .
$$

Subtracting these bounds from $f ^ { \star } \ \mathrm { g i v e s } \ ( \mathrm { A } . 4 )$ . Since $\overline { { f } } - f ^ { \star } \leq \overline { { f } } - \underline { { f } } = \Gamma , f ^ { \star } - \underline { { f } } \leq \Gamma$ , and $w ( x ) \leq w _ { \tau }$ on $O _ { \tau } , \mathrm { ( A . 4 ) }$ gives

$$
\tau - \eta _ { \tau } \leq f ^ { \star } - f ( x ) < \tau + \eta _ { \tau } ,
$$

which is (A.3).

It remains to prove the first and last inclusions in (A.5), since the middle ones are $\left( \mathrm { A . 2 } \right)$ . For the first inclusion, we may assume $\tau - \eta _ { \tau } > 0$ , since otherwise $s _ { \tau - \eta _ { \tau } } = \emptyset$ . Let $x \in S _ { \tau - \eta _ { \tau } }$ . Then $x \in S _ { \tau } \subseteq O _ { \tau }$ by (A.2), so $w ( x ) \leq w _ { \tau } ,$ and

$$
L ( x ) = U ( x ) - w ( x ) \ge f ( x ) - w _ { \tau } > f ^ { \star } - \tau + \eta _ { \tau } - w _ { \tau } = f ^ { \star } - \tau + \Gamma \ge \overline { { f } } - \tau ,
$$

where the last step uses $f ^ { \star } \geq \underline { { f } } .$ . Hence $x \in I _ { \tau }$ . For the last inclusion, let $x \in O _ { \tau }$ . Then $U ( x ) > \underline { { f } } - \tau$ and $w ( x ) \leq w _ { \tau }$ , so

$$
f ( x ) \geq L ( x ) = U ( x ) - w ( x ) > f - \tau - w _ { \tau } .
$$

Hence $f ^ { \star } - f ( x ) < ( f ^ { \star } - f ) + \tau + w _ { \tau } \leq \tau + \eta _ { \tau } , \mathrm { t h a t ~ i s , } x \in S _ { \tau + \eta _ { \tau } }$

Finally, if $0 \leq \tau \leq T$ , then $f - \tau \geq f - T .$ , so $O _ { \tau } \subseteq O _ { T }$ by (A.1). Hence, if w<sub>T</sub> bounds $U - L$ on $O _ { T }$ , then $w _ { \tau } = w _ { T }$ is admissible for all $0 \leq \tau \leq T$ , and (A.3) and (A.5) hold with the same error $\eta _ { \tau } = \Gamma + w _ { T }$ for all these tolerances. We also note that, at $\tau = 0$ , both $I _ { 0 }$ and $ { \boldsymbol { S } } _ { 0 }$ are empty, since $L \leq f \leq f ^ { \star } \leq \overline { { f } }$ . Thus $O _ { 0 }$ may be replaced by ∅ without afecting (A.2)–(A.5). □

## B.19 Proof of Proposition 4.1

Proof of Proposition $4 . 1 .$ . When a nonempty phase of either track completes, its band is intersected with the maintained confidence bounds, and an empty phase leaves them unchanged. Hence the confidence bounds can only shrink, and

$$
u ( x ) \leq u ^ { \mathrm { o l d } } ( x ) , \ell ( x ) \geq \ell ^ { \mathrm { o l d } } ( x ) { \mathrm { ~ f o r ~ } } x \in \mathcal { X } , \underline { { f } } = \operatorname* { m a x } _ { \mathcal { X } } \ell \geq \operatorname* { m a x } _ { \mathcal { X } } \ell ^ { \mathrm { o l d } } = \underline { { f } } ^ { \mathrm { o l d } } .\tag{B.93}
$$

Let x satisfy $u ( x ) > f - T _ { j } - 2 \bar { \varepsilon }$ . By (B.93),

$$
\begin{array} { r } { u ^ { \mathrm { o l d } } ( { x } ) \ge u ( { x } ) > \underline { { f } } - T _ { j } - 2 \bar { \varepsilon } \ge \underline { { f } } ^ { \mathrm { o l d } } - T _ { j } - 2 \bar { \varepsilon } , } \end{array}
$$

so the first bound in (16) applies at x. Using (B.93) again,

$$
u ( x ) - \ell ( x ) \leq u ^ { \mathrm { o l d } } ( x ) - \ell ^ { \mathrm { o l d } } ( x ) \leq 2 \bar { \varepsilon } .
$$

For the optimal-value bound, (B.93) and the second bound in (16) give $f ^ { \star } - f \leq f ^ { \star } - f ^ { \mathrm { o l d } } \leq 2 \bar { \varepsilon }$ .

## B.20 Proof of Proposition A.2

Proof of Proposition A.2. Throughout the proof, we work on ${ \mathcal { E } } _ { \delta }$ , on which $\ell \leq f \leq u$ for the confidence bounds $( \ell , u )$ at any time. For a phase h, let $\ell _ { h - 1 } , u _ { h - 1 } , \ \underline { { f } } _ { h - 1 } , \ W _ { h - 1 }$ , and $G _ { h }$ be the quantities in (10), which are fixed at the start of the phase. If the phase completes, let $\ell _ { h } , u _ { h }$ be the confidence bounds immediately after its completion, with $\underline { { f } } _ { h }$ and ${ \overline { { f } } } _ { h }$ as in (14). Since the bands are only intersected, $u _ { h } \leq u _ { h - 1 } , \ell _ { h } \geq \ell _ { h - 1 }$ , and $\underline { { f } } _ { h } \geq \underline { { f } } _ { h - 1 } .$

We first prove the width bound for a completed phase h with $\varepsilon _ { h } \geq 2 \zeta _ { N }$ . Let x satisfy $u _ { h } ( x ) >$ $\underline { { f } } _ { h } - T - 2 \varepsilon _ { h }$ . Since $u _ { h } \le u _ { h - 1 }$ and $\underline { { f } } _ { h } \geq \underline { { f } } _ { h - 1 }$

$$
\begin{array} { r l } & { u _ { h - 1 } ( x ) \geq u _ { h } ( x ) > \underline { { f } } _ { h } - T - 2 \varepsilon _ { h } \geq \underline { { f } } _ { h - 1 } - T - 2 \varepsilon _ { h } , } \\ & { G _ { h } ( x ) = [ u _ { h - 1 } ( x ) - \underline { { f } } _ { h - 1 } + T + 4 \varepsilon _ { h } ] _ { + } > 2 \varepsilon _ { h } , } \end{array}\tag{B.94}
$$

where the second line follows from the first and (10). By Lemma B.3, the width after the completed phase is at most $2 \varepsilon _ { h }$ wherever $G _ { h } > 2 \varepsilon _ { h }$ . Hence we obtain the width bound

$$
u _ { h } ( x ) - \ell _ { h } ( x ) \leq 2 \varepsilon _ { h } { \mathrm { ~ w h e n e v e r ~ } } u _ { h } ( x ) > \underline { { f } } _ { h } - T - 2 \varepsilon _ { h } .\tag{B.95}
$$

This also covers a phase completed without any evaluation, which leaves the bounds unchanged.

We next prove the bounds on the optimal value and on the region outputs. Let $x _ { c }$ be a maximizer of $u _ { h }$ over $x ,$ so that $\overline { { f } } _ { h } = u _ { h } ( x _ { c } )$ . Since $\ell _ { h } \leq f \leq u _ { h }$ , we have $\underline { { f } } _ { h } \le f ^ { \star } \le \overline { { f } } _ { h }$ . In particular, $u _ { h } ( x _ { c } ) \geq \underline { { { f } } } _ { h } > \underline { { { f } } } _ { h } - T - 2 \varepsilon _ { h }$ , so (B.95) applies at $x _ { c }$ and gives

$$
\overline { { f } } _ { h } = u _ { h } ( x _ { c } ) \leq \ell _ { h } ( x _ { c } ) + 2 \varepsilon _ { h } \leq \underline { { f } } _ { h } + 2 \varepsilon _ { h } .
$$

Hence we obtain the optimal-value bound

$$
\overline { { f } } _ { h } - \underline { { f } } _ { h } \leq 2 \varepsilon _ { h } .\tag{B.96}
$$

Since $\widehat { x } _ { h }$ maximizes $\ell _ { h }$ , we also have $f ( \widehat { x } _ { h } ) \geq \ell _ { h } ( \widehat { x } _ { h } ) = f _ { \mathbf { \delta } } \geq \overline { { f } } _ { h } - 2 \varepsilon _ { h } \geq f ^ { \star } - 2 \varepsilon _ { h }$ . For $0 < \tau \leq T$ the sets $I _ { h , \tau }$ and $O _ { h , \tau }$ in (15) are the sets in (A.1) with $L = \ell _ { h } , U = u _ { h } , \underline { { f } } = \underline { { f } } _ { h }$ , and ${ \overline { { f } } } = { \overline { { f } } } _ { h }$ . Any $x \in O _ { h , \cdot }$ satisfies $u _ { h } ( x ) > \underline { { f } } _ { h } - \tau \ge \underline { { f } } _ { h } - T - 2 \varepsilon _ { h }$ , so (B.95) gives $u _ { h } - \ell _ { h } \le w _ { \tau } = 2 \varepsilon _ { h }$ on $O _ { h , \tau } .$ Therefore, Proposition A.1 applies with $\Gamma = \overline { { f } } _ { h } - \underline { { f } } _ { h } \leq 2 \varepsilon _ { h }$ by (B.96) and $\eta _ { \tau } = \Gamma + w _ { \tau } \leq 4 \varepsilon _ { h }$ and (A.2) and (A.3) give $( \mathrm { A } . 6 ) . \mathrm { ~ A t ~ } \tau = 0$ , both output sets and $ { \boldsymbol { S } } _ { 0 }$ are empty, so (A.6) holds trivially.

Finally, we prove (A.7). Let x be a point sampled in phase h with $\varepsilon _ { h } \geq 2 \zeta _ { N }$ , possibly in a phase truncated by the budget. If $h = 0 .$ , then $f ^ { \star } - f ( x ) \leq 2 \operatorname* { s u p } _ { \chi } | f | \leq 2 B = \varepsilon _ { 0 } \leq T + L _ { \vartheta } \varepsilon _ { 0 }$ , since $| f ( x ) | \leq \| f \| _ { \mathcal { H } _ { k } ( \mathcal { X } ) } \sqrt { k ( x , x ) } \leq B$ . Let $h \geq 1$ . Phase $h - 1$ has completed, because a phase starts only after the previous phase of the same track completes, and $\varepsilon _ { h - 1 } = \vartheta \varepsilon _ { h } \geq 2 \zeta _ { N }$ . Let $( \ell ^ { \mathrm { { e n d } } } , u ^ { \mathrm { { e n d } } } )$ denote the confidence bounds immediately after the completion of phase $h - 1$ , with $\underline { { f } } ^ { \mathrm { e n d } } = \operatorname* { m a x } _ { \mathcal { X } } \ell ^ { \mathrm { e n d } }$ and $\overline { { \boldsymbol f } } ^ { \mathrm { e n d } } = \operatorname* { m a x } _ { \boldsymbol \mathcal { X } } \boldsymbol u ^ { \mathrm { e n d } }$ . By (B.95) and (B.96), applied to phase $h - 1$ , these bounds satisfy (16) with $( \ell ^ { \mathrm { o l d } } , u ^ { \mathrm { o l d } } ) = ( \ell ^ { \mathrm { e n d } } , u ^ { \mathrm { e n d } } ) , \bar { \varepsilon } = \varepsilon _ { h - 1 }$ , and $T _ { j } = T$ , where we use $f ^ { \star } - f ^ { \mathrm { e n d } } \leq \overline { { f } } ^ { \mathrm { e n d } } - f ^ { \mathrm { e n d } }$ . The bounds $( \ell _ { h - 1 } , u _ { h - 1 } )$ fixed at the start of phase h in (10) are obtained from $( \ell ^ { \mathrm { { e n d } } } , u ^ { \mathrm { { e n d } } } )$ by intersecting the bands of the completed phases of the other track that intervene, if any. Hence, by Proposition 4.1, the two bounds in (16) also hold for $( \ell _ { h - 1 } , u _ { h - 1 } , \underline { { f } } _ { h - 1 } )$

$$
\begin{array} { r } { u _ { h - 1 } ( z ) - \ell _ { h - 1 } ( z ) \leq 2 \vartheta \varepsilon _ { h } \mathrm { ~ w h e n e v e r ~ } u _ { h - 1 } ( z ) > \underline { { f } } _ { h - 1 } - T - 2 \vartheta \varepsilon _ { h } , f ^ { \star } - \underline { { f } } _ { h - 1 } \leq 2 \vartheta \varepsilon _ { h } . } \end{array}\tag{B.97}
$$

By Lemma B.3, $G _ { h } ( x ) > 2 \varepsilon _ { h }$ , and hence, by (10), $u _ { h - 1 } ( x ) > \underline { { { f } } } _ { h - 1 } - T - 2 \varepsilon _ { h } > \underline { { { f } } } _ { h - 1 } - T - 2 \vartheta \varepsilon _ { h }$

Thus the first bound in (B.97) applies at x, and since $f \geq \ell _ { h - 1 }$ 2

$$
\begin{array} { r l } & { f ^ { \star } - f ( x ) = \big ( f ^ { \star } - \underline { { f } } _ { h - 1 } \big ) + \big ( \underline { { f } } _ { h - 1 } - u _ { h - 1 } ( x ) \big ) + \big ( u _ { h - 1 } ( x ) - f ( x ) \big ) } \\ & { \qquad < 2 \vartheta \varepsilon _ { h } + ( T + 2 \varepsilon _ { h } ) + 2 \vartheta \varepsilon _ { h } = T + ( 4 \vartheta + 2 ) \varepsilon _ { h } . } \end{array}
$$

Since $1 \vartheta + 2 < L _ { \vartheta } = 4 ( \vartheta + 1 )$ , this proves (A.7). The same argument, with $G _ { h } ( x ) \geq 2 \varepsilon _ { h }$ in place of $G _ { h } ( x ) > 2 \varepsilon _ { h }$ , shows that the closed set $D _ { h }$ in Lemma B.3 satisfies

$$
D _ { h } = \{ x \in \mathcal { X } : G _ { h } ( x ) \geq 2 \varepsilon _ { h } , \ W _ { h - 1 } ( x ) \geq 2 \varepsilon _ { h } \} \subseteq S _ { ( T + L _ { \vartheta } \varepsilon _ { h } ) + } .\tag{B.98}
$$

This is (B.14), now also when phases of the other track intervene.

□

## B.21 Proof of Proposition 5.2

Proof of Proposition 5.2. Fix $\vartheta > 1$ . All upper-bound constants in this proof will depend on $\vartheta .$ Since $m = N$ , all evaluations are made by the region track, whose tolerance is T. The optimization track has quota zero and, as noted in the proof of Theorem 5.1, does not afect the analysis. The proof follows the proof of Theorem 5.2 in Section B.4, with the gap bound $L _ { \vartheta } \varepsilon _ { h }$ of each sampled point replaced by $T + L _ { \vartheta } \varepsilon _ { h }$

With $\delta = N ^ { - 2 }$ , the confidence parameters in (9) give $\zeta _ { N } \le C N ^ { - 7 / 2 } \log ^ { 1 / 2 } ( 2 N )$ . If $\zeta _ { N } > B$ , then $r _ { N } \le 2 B < 2 \zeta _ { N }$ and $R _ { N } \leq 2 B N < 2 N \zeta _ { N }$ , and these bounds are covered by the final step below. Hence assume $\zeta _ { N } \leq B$ . Then any nonempty phase has $\varepsilon _ { h } \geq 2 \zeta _ { N }$ , because Algorithm 1 allocates all remaining evaluations to the saved estimated optimal point once the accuracy level falls below $2 \zeta _ { N }$ Work on $\mathcal { E } _ { N - 2 }$ , on which the safeguard for inconsistent bounds is never triggered. With $\delta = N ^ { - 2 }$ the factor $\Lambda _ { N , \delta }$ in (B.15) is at most $C L _ { N }$ by (B.2), and $1 + \varepsilon _ { h } ^ { - p } \le C \varepsilon _ { h } ^ { - p }$ because $\varepsilon _ { h } \leq 2 B$ . Hence the argument leading to (B.15) in the proof of Theorem 5.1, which applies to any phase with $\varepsilon _ { h } \geq 2 \zeta _ { N }$ gives

$$
n _ { h } \le C L _ { N } \varepsilon _ { h } ^ { - p }\tag{B.99}
$$

for all nonempty phases, including a phase truncated by the budget. Moreover, by Proposition A.2, any completed phase h satisfies $f ^ { \star } - f ( \widehat { x } _ { h } ) \leq 2 \varepsilon _ { h }$ , and, by (A.7), any point sampled in phase h has gap at most $T + L _ { \vartheta } \varepsilon _ { h }$

We first consider cumulative regret. Split the nonempty phases at any scale $\varepsilon \in [ 2 \zeta _ { N } , 2 B ]$ . As in the proof of Theorem 5.2, all evaluations are made in a phase with $\varepsilon _ { h } \geq \varepsilon$ , in a phase with $2 \zeta _ { N } \le \varepsilon _ { h } < \varepsilon$ , or at the saved estimated optimal point after the accuracy level falls below $2 \zeta _ { N }$ By $\mathrm { ( A . 7 ) }$ , an evaluation in phase h contributes at most $T + L _ { \vartheta } \varepsilon _ { h }$ to $R _ { N }$ , which is less than $T + L _ { \vartheta } \varepsilon$ $\mathrm { i f } \ \varepsilon _ { h } < \varepsilon$ . By the third fact in the proof of Theorem 5.2, with Proposition A.2 in place of Lemma B.4, each evaluation at the saved estimated optimal point contributes less than $4 \vartheta \zeta _ { N }$ . Since there are at most N evaluations in total,

$$
R _ { N } \leq N T + L _ { \vartheta } \sum _ { h : \varepsilon _ { h } \geq \varepsilon } \varepsilon _ { h } n _ { h } + L _ { \vartheta } N \varepsilon + 4 \vartheta N \zeta _ { N } .\tag{B.100}
$$

By (B.99) and (B.25) with $b = p ,$

$$
\sum _ { h : \varepsilon _ { h } \geq \varepsilon } \varepsilon _ { h } n _ { h } \leq C L _ { N } \sum _ { h : \varepsilon _ { h } \geq \varepsilon } \varepsilon _ { h } ^ { 1 - p } \leq C L _ { N } \varepsilon ^ { 1 - p } ,
$$

so that

$$
R _ { N } \leq N T + C L _ { N } \varepsilon ^ { - ( 1 + d / \nu ) } + C N \varepsilon + C N \zeta _ { N } .\tag{B.101}
$$

Choose $\varepsilon = ( L _ { N } / N ) ^ { 1 / p }$ , clipped to $[ 2 \zeta _ { N } , 2 B ]$ . Since $( 3 + 3 d / ( 2 \nu ) ) / ( 2 + d / \nu ) = 3 / 2$ , when no clipping is active,

$$
\varepsilon = \log ^ { 3 / 2 } ( 2 N ) N ^ { - 1 / ( 2 + d / \nu ) } , L _ { N } \varepsilon ^ { 1 - p } = N \varepsilon = \log ^ { 3 / 2 } ( 2 N ) N ^ { 1 - 1 / ( 2 + d / \nu ) } .
$$

If the clipping at the lower endpoint $2 \zeta _ { N }$ is active, then $L _ { N } < N ( 2 \zeta _ { N } ) ^ { p }$ , so the second and third terms of $\mathrm { ( B . 1 0 1 ) }$ evaluated at $\varepsilon = 2 \zeta _ { N }$ are $O ( N \zeta _ { N } )$ . If the clipping at the upper endpoint 2B is active, then $N ( L _ { N } / N ) ^ { 1 / p } > 2 B N \ge R _ { N }$ . In all cases, on ${ \mathcal { E } } _ { N ^ { - 2 } }$

$$
R _ { N } \leq N T + C \log ^ { 3 / 2 } ( 2 N ) N ^ { 1 - 1 / ( 2 + d / \nu ) } + C N \zeta _ { N } .\tag{B.102}
$$

We next consider simple regret. Suppose first that no phase with accuracy level below $2 \zeta _ { N }$ is reached, and let h be the last nonempty phase. Phase $h - 1$ has completed unless $h = 0$ , possibly without evaluations, so by Proposition A.2, the saved estimated optimal point has gap at most $2 \varepsilon _ { h - 1 } = 2 \vartheta \varepsilon _ { h }$ . If phase $h$ itself has completed, the bound is smaller, and if no phase has completed, $r _ { N } \le 2 B = \varepsilon _ { 0 }$ . Additional empty phases after the budget is exhausted leave the bands and the estimated optimal point unchanged. Hence $r _ { N } \le 2 \vartheta \varepsilon _ { h }$ in all cases. Since $h$ is the last nonempty phase, all N evaluations are made in phases $0 , 1 , \ldots , h ,$ , so (B.99) and (B.25) with $b = p + 1$ give

$$
N = \sum _ { j = 0 } ^ { h } n _ { j } \leq C L _ { N } \sum _ { j = 0 } ^ { h } \varepsilon _ { j } ^ { - p } \leq C L _ { N } \varepsilon _ { h } ^ { - p } .
$$

Hence $\varepsilon _ { h } \le ( C L _ { N } / N ) ^ { 1 / p }$ , and

$$
r _ { N } \leq 2 \vartheta \varepsilon _ { h } \leq C ( L _ { N } / N ) ^ { 1 / p } = C \log ^ { 3 / 2 } ( 2 N ) N ^ { - 1 / ( 2 + d / \nu ) } .\tag{B.103}
$$

If a phase with accuracy level below $2 \zeta _ { N }$ is reached, then, by the third fact in the proof of Theorem 5.2, with Proposition A.2 in place of Lemma B.4, the saved estimated optimal point has gap less than $4 \vartheta \zeta _ { N }$

Finally, $\mathcal { E } _ { N ^ { - 2 } } ^ { c }$ has probability at most $N ^ { - 2 }$ by Lemma 5.1, and $r _ { N } \leq 2 B$ and $R _ { N } \le 2 B N$ always hold. Hence this event adds at most $2 B N ^ { - 2 }$ to $\mathbb { E } r _ { N }$ and at most $2 B / N$ to $\mathbb { E } R _ { N }$ Since $\zeta _ { N } \le C N ^ { - 7 / 2 } \log ^ { 1 / 2 } ( 2 N )$ , the terms $4 \vartheta \zeta _ { N }$ and $2 B N ^ { - 2 }$ are at most constant multiples of $\log ^ { 3 / 2 } ( 2 N ) N ^ { - 1 / ( 2 + d / \nu ) }$ , and the terms $C N \zeta _ { N }$ and $2 B / N$ are at most constant multiples of $\log ^ { 3 / \breve { 2 } } ( 2 \dot { N } ) N ^ { \dot { 1 } - 1 / ( 2 + d / \nu ) }$ . Combining (B.103) and (B.102) with the deterministic bound $R _ { N } \le 2 B N$ gives (28). □

## C Proofs of Lemmas

This section proves the lemmas stated in the main text and in the subsections of Section B.

## C.1 Proof of Lemma 3.1

Proof of Lemma 3.1. We first prove the measurability of ${ \widehat { d } } ( x )$ and then the two inequalities in (6). Fix $x \in \mathcal { X }$ . For $q \in \mathbb { Q } \cap [ 0 , T ]$ , let $Z _ { q } = q \mathbf { 1 } \{ x \in I _ { q } \} + T \mathbf { 1 } \{ x \not \in I _ { q } \}$ , which is a random variable because membership in $I _ { q }$ is measurable. Then

$$
{ \widehat { d } } ( x ) = \operatorname* { m i n } \Bigl \{ T , \operatorname* { i n f } _ { q \in \mathbb { Q } \cap [ 0 , T ] } Z _ { q } \Bigr \}
$$

is the infimum of countably many random variables and is therefore measurable.

Now work on $\mathcal { E } _ { f }$ , and let $d ( x ) = f ^ { \star } - f ( x )$ . For the lower bound, if $x \in I _ { q }$ for some $q \in \mathbb { Q } \cap [ 0 , T ]$ then $x \in S _ { q }$ by the first inclusion in (5), that is, $q > d ( x )$ . Hence all elements of the set in the definition of ${ \widehat { d } } ( x )$ are at least min $\{ d ( x ) , T \}$ , and so is its infimum.

For the upper bound, we always have $\hat { d } ( x ) \leq T . \mathrm { I f } d ( x ) + \eta \geq T$ , then $\widehat { d } ( x ) \leq T = \operatorname* { m i n } \{ d ( x ) + \eta , T \}$ Suppose now that $d ( x ) + \eta < T$ . For $j \geq 1$ , the interval $( d ( x ) + \eta$ , min $\{ T , d ( x ) + \eta + 1 / j \} )$ has positive length, so it contains a rational number $q _ { j }$ . Since $q _ { j } > d ( x )$ , we have $x \in S _ { q _ { j } } \subseteq O _ { q _ { j } }$ by (5). Since $| f ^ { \star } - f ( x ) - q _ { j } | = q _ { j } - d ( x ) > \eta$ , the second condition in (5) gives $x \notin O _ { q _ { j } } \setminus I _ { q _ { j } }$ . Hence $x \in I _ { q _ { j } }$ and

$$
\widehat { d } ( x ) \leq q _ { j } < d ( x ) + \eta + 1 / j .
$$

Letting $j \to \infty$ gives $\widehat { d } ( x ) \leq d ( x ) + \eta = \operatorname* { m i n } \{ d ( x ) + \eta , T \}$ . Since $x \in \mathcal { X }$ was arbitrary, (6) holds simultaneously for all $x \in \mathcal { X }$ on $\mathcal { E } _ { f }$ . Finally, for an $( \eta , T , \delta )$ -valid procedure, (5) holds for all $\tau \in [ 0 , T ]$ on an event of probability at least $1 - \delta .$ , which can serve as $\mathcal { E } _ { f }$ □

## C.2 Proof of Lemma B.1

Proof of Lemma B.1. Fix a smooth radial function $\phi _ { 0 }$ on $\mathbb { R } ^ { d }$ with $0 \le \phi _ { 0 } \le 1 , \phi _ { 0 } ( x ) = 1$ for $\| x \| \leq 1 / 2$ , and $\phi _ { 0 } ( x ) = 0$ for $\| { \boldsymbol x } \| \geq 1$ . We decompose

$$
\varphi ( x ) \| x \| ^ { \alpha } = \varphi ( x ) \phi _ { 0 } ( x ) \| x \| ^ { \alpha } + \varphi ( x ) \{ 1 - \phi _ { 0 } ( x ) \} \| x \| ^ { \alpha } .
$$

The second term is smooth and compactly supported, because $1 - \phi _ { 0 }$ vanishes when $\| x \| \leq 1 / 2$ and $\| x \| ^ { \alpha }$ is smooth away from the origin. Hence it belongs to $H ^ { s } ( \mathbb { R } ^ { d } )$ . Since multiplication by a smooth compactly supported function preserves $H ^ { s } ( \mathbb { R } ^ { d } )$ , it remains to show that $\phi _ { 0 } ( x ) \| x \| ^ { \alpha }$ belongs to $H ^ { s } ( \mathbb { R } ^ { d } )$ .

Let $w ( x ) = \{ \phi _ { 0 } ( x ) - \phi _ { 0 } ( 2 x ) \} \| x \| ^ { \alpha }$ . Since $\phi _ { 0 } ( x ) - \phi _ { 0 } ( 2 x ) = 0$ when $\| x \| \leq 1 / 4$ or $\| { \boldsymbol x } \| \geq 1$ , the function w is smooth, vanishes unless $1 / 4 \leq \| x \| \leq 1$ , and belongs to $H ^ { s } ( \mathbb R ^ { d } )$ . For $x \neq 0$ , telescoping and $\phi _ { 0 } ( 2 ^ { J } x )  0$ as $J \to \infty$ give

$$
\phi _ { 0 } ( x ) \| x \| ^ { \alpha } = \sum _ { j = 0 } ^ { \infty } \{ \phi _ { 0 } ( 2 ^ { j } x ) - \phi _ { 0 } ( 2 ^ { j + 1 } x ) \} \| x \| ^ { \alpha } = \sum _ { j = 0 } ^ { \infty } 2 ^ { - j \alpha } w ( 2 ^ { j } x ) .\tag{C.1}
$$

For $j \geq 0$ , the scaling inequality (B.3) with $t = 2 ^ { j }$ gives

$$
\| 2 ^ { - j \alpha } w ( 2 ^ { j } \cdot ) \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } \leq 2 ^ { - j \alpha } 2 ^ { j \nu } \| w \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } = 2 ^ { - j ( \alpha - \nu ) } \| w \| _ { H ^ { s } ( \mathbb { R } ^ { d } ) } .\tag{C.2}
$$

Since $\alpha > \nu$ , the norms in (C.2) are summable over $j ,$ so the series in (C.1) converges in $H ^ { s } ( \mathbb R ^ { d } )$ Its limit agrees almost everywhere with the pointwise limit $\phi _ { 0 } ( x ) \| x \| ^ { \alpha }$ , which proves the first claim.

For the second claim, fix $c \in \mathbb { R } ^ { d }$ and a bounded domain $\mathcal { D } _ { \mathrm { : } }$ , and choose $\varphi \in C _ { c } ^ { \infty } (  { \mathbb { R } } ^ { d } )$ equal to one on $\{ x - c : x \in D \}$ . By the first claim and the translation invariance of the $H ^ { s } ( \mathbb R ^ { d } )$ norm, the function $x \mapsto \varphi ( x - c ) \| x - c \| ^ { \alpha }$ belongs to $H ^ { s } ( \mathbb R ^ { d } )$ , and it agrees with $\| x - c \| ^ { \alpha }$ on D. By the norm comparison following (B.3), its restriction to D therefore belongs to the Matérn RKHS on $\mathcal { D }$ and has finite RKHS norm. □

## C.3 Proof of Lemma 5.1

Proof of Lemma 5.1. We first establish the bound at the points of a finite net, using that the sampling locations of a phase do not depend on its responses, and then extend it to all of X through the Hölder bound $d _ { k } ( x , z ) \leq L _ { \theta } \| x - z \| ^ { \theta }$ in Section 2.3.

Set $r _ { N } ^ { \mathrm { m e s h } } = N ^ { - 4 / \theta }$ , and fix a deterministic maximal Euclidean $r _ { N } ^ { \mathrm { m e s h } }$ -separated set $V _ { N } \subset \mathcal { X }$ . By maximality, $V _ { N }$ is an $r _ { N } ^ { \mathrm { m e s h } }$ -net of X. The Euclidean balls of radius $r _ { N } ^ { \mathrm { m e s h } } / 2$ centered at its points have disjoint interiors and lie in a ball of radius $D _ { \mathcal { X } } + r _ { N } ^ { \mathrm { m e s h } } / 2$ . Comparing their volumes gives

$$
\left. V _ { N } \right. \le \left( 1 + \frac { 2 D _ { \mathscr { X } } } { r _ { N } ^ { \mathrm { m e s h } } } \right) ^ { d } = M _ { N } .\tag{C.3}
$$

Consider a nonempty phase of either track, and let $\mathcal { G } _ { h }$ denote the history of both tracks available at its start. Since the two tracks alternate whole phases, all observations made during the phase belong to its batch. Within the phase, the sampling locations, the stopping test, and the quota check depend only on $\mathcal { G } _ { h }$ and, through $s _ { n }$ , on the sampled locations, but not on the current responses. Hence, conditional on $\mathcal { G } _ { h }$ , the sampling locations $x _ { 1 } , \ldots , x _ { n }$ of the phase are fixed at sample size n. Let $\pmb { \varepsilon } = ( \varepsilon _ { 1 } , \ldots , \varepsilon _ { n } ) ^ { \top }$ be the corresponding observation noise, and $a _ { x } = ( K _ { n } + \lambda I ) ^ { - 1 } k _ { n } ( x )$ , so that

$$
f ( x ) - \mu _ { n } ( x ) = \Big \{ f ( x ) - \sum _ { i = 1 } ^ { n } ( a _ { x } ) _ { i } f ( x _ { i } ) \Big \} - a _ { x } ^ { \top } \varepsilon .\tag{C.4}
$$

The kernel ridge identity gives $\begin{array} { r } { s _ { n } ^ { 2 } ( x ) = \| k _ { x } - \sum _ { i } ( a _ { x } ) _ { i } k _ { x _ { i } } \| _ { \mathcal { H } _ { k } } ^ { 2 } + \lambda \| a _ { x } \| _ { 2 } ^ { 2 } } \end{array}$ . Hence, by the reproducing property and $\| f \| _ { \mathcal { H } _ { k } } \le B$

$$
\Big | f ( x ) - \sum _ { i = 1 } ^ { n } ( a _ { x } ) _ { i } f ( x _ { i } ) \Big | \leq \| f \| _ { \mathcal { H } _ { k } } \Big \| k _ { x } - \sum _ { i = 1 } ^ { n } ( a _ { x } ) _ { i } k _ { x _ { i } } \Big \| _ { \mathcal { H } _ { k } } \leq B s _ { n } ( x ) , \| a _ { x } \| _ { 2 } \leq \frac { s _ { n } ( x ) } { \sqrt { \lambda } } .\tag{C.5}
$$

Since $a _ { x }$ is fixed given $\mathcal { G } _ { h }$ , iterating the conditional moment bound in Assumption 5.2 over the evaluations of the phase gives

$$
\begin{array} { r } { \mathbb { E } [ \exp ( u a _ { x } ^ { \top } \varepsilon ) \mid \mathcal { G } _ { h } ] \leq \exp ( u ^ { 2 } \sigma ^ { 2 } \| a _ { x } \| _ { 2 } ^ { 2 } / 2 ) , u \in \mathbb { R } . } \end{array}
$$

Note that this uses no independence assumption on the observation errors.

Fix a nonempty phase, a sample size $n \geq 1$ within it, and a net point $v \in V _ { N }$ , and put

$$
t _ { v } = \frac { \sigma } { \sqrt { \lambda } } \sqrt { 2 \log ( 4 M _ { N } N ^ { 2 } / \delta ) } s _ { n } ( v ) \ \ge \ \sigma \| { a _ { v } } \| _ { 2 } \sqrt { 2 \log ( 4 M _ { N } N ^ { 2 } / \delta ) } ,
$$

where the inequality uses (C.5). The Chernof bound gives

$$
\mathbb { P } \big ( \vert a _ { v } ^ { \top } \pmb { \varepsilon } \vert > t _ { v } \big \vert \mathcal { G } _ { h } \big ) \le 2 \exp \Big ( { - \frac { t _ { v } ^ { 2 } } { 2 \sigma ^ { 2 } \| a _ { v } \| _ { 2 } ^ { 2 } } } \Big ) \le \frac \delta { 2 M _ { N } N ^ { 2 } } .
$$

On the complementary event, (C.4), (C.5), and the definition of $\beta _ { N }$ in (9) give

$$
| f ( v ) - \mu _ { n } ( v ) | \leq B s _ { n } ( v ) + t _ { v } = \beta _ { N } s _ { n } ( v ) .\tag{C.6}
$$

We next take a union bound over a deterministic index set. Since each nonempty phase contains at least one of the N evaluations, there are at most N nonempty phases of the two tracks. Index them by $i = 1 , \ldots , N$ in their order of occurrence, and, for $i \le N , n \le N$ , and $v \in V _ { N }$ , let $F _ { i , n , v }$ be the event that the ith nonempty phase exists, reaches sample size $n ,$ and violates (C.6) at v with sample size n. The start of this phase is a stopping time, and, given the history $\mathcal { G } _ { h }$ at its start, both whether it reaches sample size n and its first n sampling locations are fixed. Hence the conditional bound above, integrated over $\mathcal { G } _ { h }$ , gives $\mathbb { P } ( F _ { i , n , v } ) \le \delta / ( 2 M _ { N } N ^ { 2 } )$ . Therefore (C.6) holds simultaneously for all nonempty phases, all sample sizes within them, and all net points, except on the union of the at most $N \cdot N \cdot M _ { N }$ events $F _ { i , n , v }$ , whose probability is at most $N \cdot N \cdot M _ { N } \cdot \delta / ( 2 M _ { N } N ^ { 2 } ) = \delta / 2$ . Only phases with observations need to be indexed, since a phase completed without any evaluation leaves the bands unchanged and requires no new event. The case $n = 0$ satisfies (22) deterministically, because $\mu _ { 0 } \equiv 0 , s _ { 0 } \equiv 1$ , and $\beta _ { { N } } \geq B \geq | f |$

We next bound the observations and the RKHS norm of $\mu _ { n }$ . Since $| y _ { t } | \leq | f ( x _ { t } ) | + | \varepsilon _ { t } | \leq B + | \varepsilon _ { t } |$ and, by Assumption 5.2,

$$
\mathbb { P } \big ( | \varepsilon _ { t } | > \sigma \sqrt { 2 \log ( 4 N / \delta ) } \big ) \le 2 \exp \{ - \log ( 4 N / \delta ) \} = \frac \delta { 2 N } ,
$$

a union bound over $t \leq N$ gives $| y _ { t } | \le Y _ { N }$ for all $t \leq N$ , except on an event of probability at most $\delta / 2$ . On this event, writing $\begin{array} { r } { \mu _ { n } = \sum _ { i } \alpha _ { i } k _ { x _ { i } } } \end{array}$ with $\alpha = ( K _ { n } + \lambda I ) ^ { - 1 } y _ { 1 : n }$ , the spectral decomposition of $K _ { n }$ gives

$$
\| \mu _ { n } \| _ { \mathcal { H } _ { k } } ^ { 2 } = y _ { 1 : n } ^ { \top } ( K _ { n } + \lambda I ) ^ { - 1 } K _ { n } ( K _ { n } + \lambda I ) ^ { - 1 } y _ { 1 : n } \leq \operatorname* { s u p } _ { q \geq 0 } \frac { q } { ( q + \lambda ) ^ { 2 } } \| y _ { 1 : n } \| _ { 2 } ^ { 2 } = \frac { \| y _ { 1 : n } \| _ { 2 } ^ { 2 } } { 4 \lambda } \leq \frac { N Y _ { N } ^ { 2 } } { 4 \lambda } .\tag{C.7}
$$

We also need the continuity of $s _ { n }$ . Define

$$
\| g \| _ { n , \lambda } ^ { 2 } = \operatorname* { m i n } _ { a \in \mathbb { R } ^ { n } } \Bigl \{ \Bigl \| g - \sum _ { i = 1 } ^ { n } a _ { i } k _ { x _ { i } } \Bigr \| _ { \mathcal { H } _ { k } } ^ { 2 } + \lambda \| a \| _ { 2 } ^ { 2 } \Bigr \} , g \in \mathcal { H } _ { k } .
$$

This is the squared distance from $( g , 0 )$ to the subspace $\{ ( \sum _ { i } a _ { i } k _ { x _ { i } } , - a ) : a \in \mathbb { R } ^ { n } \}$ of $\mathcal { H } _ { k } \oplus \mathbb { R } ^ { n }$ equipped with the norm given by $\| ( g , a ) \| ^ { 2 } = \| g \| _ { \mathcal { H } _ { k } } ^ { 2 } + \lambda \| a \| _ { 2 } ^ { 2 }$ . Hence $\Vert \cdot \Vert _ { n , \lambda }$ is a Hilbert seminorm on $\mathcal { H } _ { k }$ . Moreover, $\| g \| _ { n , \lambda } \leq \| g \| _ { \mathcal { H } _ { k } }$ by taking $a = 0 ,$ , and $s _ { n } ( x ) = \| k _ { x } \| _ { n , \lambda }$ by the kernel ridge identity. Therefore,

$$
| s _ { n } ( x ) - s _ { n } ( v ) | \leq \| k _ { x } - k _ { v } \| _ { n , \lambda } \leq \| k _ { x } - k _ { v } \| _ { \mathcal H _ { k } } = d _ { k } ( x , v ) .\tag{C.8}
$$

Finally, the two exceptional events above have total probability at most δ. Outside them, fix a phase with observations, a sample size $n \geq 1$ , and $x \in \mathcal { X }$ , and choose $v \in V _ { N }$ with $\| x - v \| \leq r _ { N } ^ { \mathrm { m e s h } }$ Since $| g ( x ) - g ( v ) | = | \langle g , k _ { x } - k _ { v } \rangle _ { \mathcal { H } _ { k } } | \leq \| g \| _ { \mathcal { H } _ { k } } d _ { k } ( x , v )$ for any $g \in \mathcal { H } _ { k }$ , as in (4), the bounds (C.6), (C.8), and (C.7) give

$$
\begin{array} { r l r } {  { | f ( \boldsymbol { x } ) - \mu _ { n } ( \boldsymbol { x } ) | \leq | f ( \boldsymbol { v } ) - \mu _ { n } ( \boldsymbol { v } ) | + ( B + \| \mu _ { n } \| _ { \mathcal H _ { k } } ) d _ { k } ( \boldsymbol { x } , \boldsymbol { v } ) } } \\ & { } & { \leq \beta _ { N } s _ { n } ( \boldsymbol { x } ) + ( B + \| \mu _ { n } \| _ { \mathcal H _ { k } } + \beta _ { N } ) d _ { k } ( \boldsymbol { x } , \boldsymbol { v } ) } \\ & { } & { \leq \beta _ { N } s _ { n } ( \boldsymbol { x } ) + ( B + \frac { \sqrt { N } Y _ { N } } { 2 \sqrt { \lambda } } + \beta _ { N } ) L _ { \theta } N ^ { - 4 } = \beta _ { N } s _ { n } ( \boldsymbol { x } ) + \zeta _ { N } , } \end{array}
$$

where the last line uses $d _ { k } ( x , v ) \leq L _ { \theta } ( r _ { N } ^ { \mathrm { m e s h } } ) ^ { \theta } = L _ { \theta } N ^ { - 4 }$ and the definition of $\zeta _ { N }$ in (9). This proves (22). □

## C.4 Proof of Lemma B.2

Proof of Lemma B.2. We first bound the information gain on a single ball, separately for $\nu > 1 / 2$ and $0 < \nu \leq 1 / 2$ , and then combine the balls of a cover. For $\nu > 1 / 2$ , the radius-dependent deterministic bound in Iwazaki (2025, Corollary 8) gives, for $0 < r \leq \bar { r }$

$$
\gamma _ { n } ( B ( z , r ) ; \lambda ) \le C \log ^ { 2 } ( 2 N ) [ 1 + r ^ { 2 \nu d / ( 2 \nu + d ) } ( n / \lambda ) ^ { d / ( 2 \nu + d ) } ] .\tag{C.9}
$$

To verify (C.9), note first that the Matérn kernel is translation invariant, so that $\gamma _ { n } ( B ( z , r ) ; \lambda ) =$ $\gamma _ { n } ( B ( 0 , r ) ; \lambda )$ . We use Theorem 7 and Corollary 8 of Iwazaki (2025), in which the information gain is defined with a noise variance, denoted by $\lambda ^ { 2 }$ there, that corresponds to our λ. In our notation, their bound for the ball $B ( 0 , r )$ is

$$
\begin{array} { r l r } {  { \gamma _ { n } ( B ( 0 , r ) ; \lambda ) \le C _ { n } \bar { \gamma } _ { n } ( r ) + C _ { 0 } , \mathrm { ~ w h e r e ~ } } } \\ & { } & { \bar { \gamma } _ { n } ( r ) = C _ { 1 } \log \Big ( 1 + \frac { 2 n } { \lambda } \Big ) + C _ { 2 } r ^ { \frac { 2 \nu d } { 2 \nu + d } } \Big ( \frac { n } { \lambda \ell ^ { 2 \nu } } \Big ) ^ { \frac { d } { 2 \nu + d } } \log ^ { \frac { 2 \nu } { 2 \nu + d } } \Big ( 1 + \frac { 2 n } { \lambda } \Big ) , } \end{array}\tag{C.10}
$$

$C _ { 0 }$ is an absolute constant, $C _ { 1 }$ and $C _ { 2 }$ depend only on d and $\nu ,$ and

$$
C _ { n } = \operatorname* { m a x } \Bigl \{ 1 , \ \log _ { 2 } \Bigl ( 1 + \frac { \Gamma ( \nu ) } { C _ { \nu } } \log \frac { n ^ { 2 } } { \lambda } \Bigr ) + \frac { 1 } { \nu } \log _ { 2 } \Bigl ( \frac { n ^ { 2 } } { \nu \Gamma ( \nu ) \lambda } \Bigr ) + 1 \Bigr \}
$$

for a constant $C _ { \nu } > 0$ depending only on ν. Let $n _ { 0 } = \lceil \{ \lambda \operatorname* { m a x } \{ 1 , \nu \Gamma ( \nu ) \} \} ^ { 1 / 2 } \rceil$ , so that the logarithms in $C _ { n }$ are nonnegative for $n \geq n _ { 0 }$ . For fixed λ and $n _ { 0 } ~ \le ~ n ~ \le ~ N$ , each of these logarithms is at most $C \log ( 2 N )$ , and hence $C _ { n } \ \leq \ C \log ( 2 N )$ . Moreover, $\log ( 1 + 2 n / \lambda ) \leq C \log ( 2 N )$ and $\log ^ { 2 \nu / ( 2 \nu + d ) } ( 1 + 2 n / \lambda ) \leq \operatorname* { m a x } \{ 1 , \log ( 1 + 2 n / \lambda ) \} \leq C \log ( 2 N )$ . Hence, with the fixed length scale $\ell$ absorbed into the constant,

$$
\bar { \gamma } _ { n } ( r ) \leq C \log ( 2 N ) \big [ 1 + r ^ { 2 \nu d / ( 2 \nu + d ) } ( n / \lambda ) ^ { d / ( 2 \nu + d ) } \big ] .
$$

Inserting these two bounds into (C.10) and using $\log ^ { 2 } ( 2 N ) \geq \log ^ { 2 } 4 > 1$ gives (C.9) for $n _ { 0 } \leq n \leq N$ For $1 \leq n < n _ { 0 }$ , Hadamard’s inequality and $k ( x , x ) = 1$ give det $\begin{array} { r } { ( I + K _ { n } / \lambda ) \le \prod _ { i = 1 } ^ { n } \{ 1 + k ( x _ { i } , x _ { i } ) / \lambda \} = } \end{array}$ $( 1 + \lambda ^ { - 1 } ) ^ { n }$ for any $x _ { 1 } , \ldots , x _ { n } ,$ so that $\gamma _ { n } ( B ( z , r ) ; \lambda ) \le ( n _ { 0 } / 2 ) \log ( 1 + \lambda ^ { - 1 } )$ ). Since $n _ { 0 }$ is fixed, this is at most a constant times the right-hand side of (C.9), uniformly in r.

For $0 < \nu \leq 1 / 2$ , we prove (C.9) by a projection argument. As in Section 2.3, the Matérn spectral density and $| 1 - e ^ { i t } | ^ { 2 } \leq C \operatorname* { m i n } \{ 1 , t ^ { 2 } \}$ give

$$
\begin{array} { r l r } {  { d _ { k } ( x , z ) ^ { 2 } = \int _ { \mathbb R ^ { d } } | e ^ { i \omega ^ { \top } ( x - z ) } - 1 | ^ { 2 } \widehat { k } ( \omega ) d \omega } } \\ & { } & { \leq C \int _ { \mathbb R ^ { d } } \operatorname* { m i n } \{ 1 , \| \omega \| ^ { 2 } \| x - z \| ^ { 2 } \} \big ( 2 \nu / \ell ^ { 2 } + \| \omega \| ^ { 2 } \big ) ^ { - ( \nu + d / 2 ) } d \omega \leq C \| x - z \| ^ { 2 \nu } , } \end{array}\tag{C.11}
$$

where the last step splits the integral at $\| \omega \| = 1 / \| x - z \|$ and uses $0 ~ < ~ \nu ~ < ~ 1$ . Fix $\boldsymbol { B } ( z , \boldsymbol { r } )$ with $0 < r \leq \bar { r }$ and an integer $m \geq 1$ . Cover $\boldsymbol { B } ( \boldsymbol { z } , \boldsymbol { r } )$ by m balls $B ( z _ { q } , r _ { m } ) , q = 1 , . . . , m .$ , with $r _ { m } = C r m ^ { - 1 / d }$ , and assign each design point $x _ { i } \in B ( z , r )$ to a center $z _ { q ( i ) }$ with $\| x _ { i } - z _ { q ( i ) } \| \leq r _ { m }$ Let Π be the orthogonal projection of $\mathcal { H } _ { k }$ onto span $\{ k _ { z _ { 1 } } , \ldots , k _ { z _ { m } } \}$ . The ambient Matérn features may be used because the kernel is defined on $\mathbb { R } ^ { d }$ . Decomposing $k _ { x _ { i } } = \Pi k _ { x _ { i } } + ( I - \Pi ) k _ { x _ { i } }$ splits the Gram matrix as $K _ { n } = K _ { n } ^ { \parallel } + K _ { n } ^ { \perp }$ , where

$$
( K _ { n } ^ { \parallel } ) _ { i j } = \langle \Pi k _ { x _ { i } } , \Pi k _ { x _ { j } } \rangle _ { \mathcal { H } _ { k } } , ( K _ { n } ^ { \perp } ) _ { i j } = \langle ( I - \Pi ) k _ { x _ { i } } , ( I - \Pi ) k _ { x _ { j } } \rangle _ { \mathcal { H } _ { k } }
$$

are both positive semidefinite. Since $k ( x , x ) = 1$ and $\| ( I - \Pi ) k _ { x _ { i } } \| _ { \mathcal { H } _ { k } } \leq \| k _ { x _ { i } } - k _ { z _ { q ( i ) } } \| _ { \mathcal { H } _ { k } } = d _ { k } ( x _ { i } , z _ { q ( i ) } )$ (C.11) gives

$$
\mathrm { r a n k } K _ { n } ^ { \parallel } \le m , \mathrm { t r } K _ { n } ^ { \parallel } \le \mathrm { t r } K _ { n } = n , \mathrm { t r } K _ { n } ^ { \perp } = \sum _ { i = 1 } ^ { n } \| ( I - \Pi ) k _ { x _ { i } } \| _ { \mathcal { H } _ { k } } ^ { 2 } \le C n r ^ { 2 \nu } m ^ { - 2 \nu / d } .\tag{C.12}
$$

For positive semidefinite A and D, we have log det $( I + A + D ) \leq \log \operatorname* { d e t } ( I + A ) + \operatorname { t r } D$ , and log det $( I + A ) \leq$ rank(A) log $\left( 1 + \operatorname { t r } A / \right)$ rank(A) by the concavity of the logarithm. Applying these with $A = K _ { n } ^ { \parallel } / \lambda$ and $D = K _ { n } ^ { \perp } / \lambda$ , and using (C.12), gives

$$
2 \gamma _ { n } ( B ( z , r ) ; \lambda ) \leq m \log \Bigl ( 1 + \frac { n } { \lambda m } \Bigr ) + C \frac { n } { \lambda } r ^ { 2 \nu } m ^ { - 2 \nu / d } .
$$

Choosing $m \asymp \operatorname* { m a x } \{ 1 , [ ( n / \lambda ) r ^ { 2 \nu } ] ^ { d / ( 2 \nu + d ) } \}$ makes both terms at most a constant times

$$
\log ( 2 N ) \big [ 1 + r ^ { 2 \nu d / ( 2 \nu + d ) } ( n / \lambda ) ^ { d / ( 2 \nu + d ) } \big ] ,
$$

which is at most a constant times the right-hand side of (C.9). Hence (C.9) also holds for $0 < \nu \leq 1 / 2$ Now let A be covered by K balls $B ( z _ { 1 } , r ) , \ldots , B ( z _ { K } , r )$ Given design points $x _ { 1 } , \ldots , x _ { n } \in A$ assign each of them to one covering ball that contains it, and let $n _ { i }$ be the number of points assigned to $B ( z _ { i } , r )$ , so that $\textstyle \sum _ { i } n _ { i } = n$ . Relabel the design points so that the points assigned to the same ball are consecutive, which permutes the rows and columns of $K _ { n }$ simultaneously and does not change det $\left( I + K _ { n } / \lambda \right)$ . Let $\boldsymbol { K } ^ { ( i ) } \in \mathbb { R } ^ { n _ { i } \times n _ { i } }$ be the Gram matrix of the points assigned to $B ( z _ { i } , r )$ , so that the matrices $I + K ^ { ( i ) } / \lambda$ with $n _ { i } > 0$ are the diagonal blocks of $I + K _ { n } / \lambda$ . Since $I + K _ { n } / \lambda$ is positive definite, Fischer’s determinant inequality gives

$$
\frac { 1 } { 2 } \log \operatorname* { d e t } ( I + K _ { n } / \lambda ) \le \sum _ { i : n _ { i } > 0 } \frac { 1 } { 2 } \log \operatorname* { d e t } ( I + K ^ { ( i ) } / \lambda ) \le \sum _ { i : n _ { i } > 0 } \gamma _ { n _ { i } } ( B ( z _ { i } , r ) ; \lambda ) ,
$$

where the last step uses the definition of $\gamma _ { n _ { i } }$ and the fact that the points assigned to $B ( z _ { i } , r )$ lie in this ball. Hence, by (C.9) and the concavity of $t \mapsto t ^ { d / ( 2 \nu + d ) }$ , which gives $\begin{array} { r } { \sum _ { i } n _ { i } ^ { d / ( 2 \nu + d ) } \leq } \end{array}$ $K ^ { 2 \nu / ( 2 \nu + d ) } n ^ { d / ( 2 \nu + d ) }$

$$
\begin{array} { r l r } {  { \sum _ { i \colon n _ { i } > 0 } \gamma _ { n _ { i } } ( B ( z _ { i } , r ) ; \lambda ) \le C \log ^ { 2 } ( 2 N ) [ K + r ^ { 2 \nu d / ( 2 \nu + d ) } \lambda ^ { - d / ( 2 \nu + d ) } \sum _ { i = 1 } ^ { K } n _ { i } ^ { d / ( 2 \nu + d ) } ] } } \\ & { } & { \le C \log ^ { 2 } ( 2 N ) [ K + K ^ { 2 \nu / ( 2 \nu + d ) } r ^ { 2 \nu d / ( 2 \nu + d ) } ( n / \lambda ) ^ { d / ( 2 \nu + d ) } ] . } \end{array}
$$

Since the right-hand side does not depend on the design points, taking the supremum over $x _ { 1 } , \ldots , x _ { n } \in$ A gives the same bound for $\gamma _ { n } ( A ; \lambda )$ . This is the second term in the minimum in (B.11). For the first term, X is contained in one ball of the fixed radius $D _ { \mathcal { X } }$ , and (C.9) with $\bar { r }$ replaced by max $\{ \bar { r } , D _ { \mathcal { X } } \}$ gives $\gamma _ { n } ( A ; \lambda ) \leq \gamma _ { n } ( \mathcal { X } ; \lambda ) \leq C \log ^ { 2 } ( 2 N ) [ 1 + ( n / \lambda ) ^ { d / ( 2 \nu + d ) } ]$ , with a constant depending on the fixed domain.

Finally, we verify the bound on the total variance stated before Lemma B.2. Since $0 \leq s _ { j - 1 } ^ { 2 } \leq 1$ and $t \mapsto \log ( 1 + t / \lambda )$ is concave and vanishes at zero, $s _ { j - 1 } ^ { 2 } ( x _ { j } ) \le \log ( 1 + s _ { j - 1 } ^ { 2 } ( x _ { j } ) / \lambda ) / \log ( 1 + \lambda ^ { - 1 } )$ Hence, for design points $x _ { 1 } , \ldots , x _ { n } \in A$ 2

$$
\sum _ { j = 1 } ^ { n } s _ { j - 1 } ^ { 2 } ( x _ { j } ) \leq { \frac { 1 } { \log ( 1 + \lambda ^ { - 1 } ) } } \sum _ { j = 1 } ^ { n } \log \bigl ( 1 + s _ { j - 1 } ^ { 2 } ( x _ { j } ) / \lambda \bigr ) = { \frac { \log \operatorname* { d e t } ( I + K _ { n } / \lambda ) } { \log ( 1 + \lambda ^ { - 1 } ) } } \leq C _ { \lambda } \gamma _ { n } ( A ; \lambda ) .
$$

Since conditioning on additional observations can only reduce each posterior variance, the same bound holds for an arbitrary subsequence of n sampled points in A. □

## C.5 Proof of Lemma B.3

Proof of Lemma B.3. We first show that any sampled point has a large posterior standard deviation and lies in D. Summing the posterior variances then gives (B.12), and solving the resulting inequality under a cover gives (B.13).

A point $x _ { j }$ is sampled only if the stopping test fails, that is, $a _ { j - 1 } ^ { \star } = a _ { j - 1 } ( x _ { j } ) > 2 \varepsilon$ . By the definition of $a _ { j - 1 }$ , all three terms in the minimum exceed 2ε:

$$
G ( x _ { j } ) > 2 \varepsilon , W ( x _ { j } ) > 2 \varepsilon , 2 \beta _ { N } s _ { j - 1 } ( x _ { j } ) + 2 \zeta _ { N } > 2 \varepsilon .
$$

The last inequality and $\zeta _ { N } \leq \varepsilon / 2$ give $s _ { j - 1 } ( x _ { j } ) > ( \varepsilon - \zeta _ { N } ) / \beta _ { N } \geq \varepsilon / ( 2 \beta _ { N } )$ . In particular, all sampled point lie in $D ,$ , and if D is empty, then no point is sampled. Since D is fixed given the history at the start of the phase, the bound on the total variance stated before Lemma B.2 gives, for $n > 0$

$$
{ \frac { n \varepsilon ^ { 2 } } { 4 \beta _ { N } ^ { 2 } } } < \sum _ { j = 1 } ^ { n } s _ { j - 1 } ^ { 2 } ( x _ { j } ) \leq C _ { \lambda } \gamma _ { n } ( D ; \lambda ) .\tag{C.13}
$$

This proves (B.12), which holds trivially for $n = 0$ . The argument also applies to a phase truncated by the budget, since it uses only the points sampled in the phase.

We next consider the maintained width after completion, working on ${ \mathcal { E } } _ { \delta }$ . At completion, $a _ { n } ^ { \star } \leq 2 \varepsilon$ Hence, wherever $G ( x ) > 2 \varepsilon$

$$
\operatorname* { m i n } \{ W ( x ) , 2 \beta _ { N } s _ { n } ( x ) + 2 \zeta _ { N } \} \le 2 \varepsilon .
$$

If $n > 0$ , the new band $\left[ \mu _ { n } - w _ { n } , \mu _ { n } + w _ { n } \right]$ with $w _ { n } = \beta _ { N } s _ { n } + \zeta _ { N }$ is intersected with the maintained bounds $( \ell , u )$ , and both contain $f$ on ${ \mathcal { E } } _ { \delta }$ . Hence the updated bounds $\ell ^ { \prime } = \operatorname* { m a x } \{ \ell , \mu _ { n } - w _ { n } \}$ and $u ^ { \prime } = \operatorname* { m i n } \{ u , \mu _ { n } + w _ { n } \}$ satisfy $\ell ^ { \prime } \leq f \leq u ^ { \prime }$ . Since $u ^ { \prime } \leq u , \ell ^ { \prime } \geq \ell , u ^ { \prime } \leq \mu _ { n } + w _ { n }$ , and $\ell ^ { \prime } \geq \mu _ { n } - w _ { n }$ wherever $G ( x ) > 2 \varepsilon$

$$
u ^ { \prime } ( x ) - \ell ^ { \prime } ( x ) \leq \operatorname* { m i n } \{ u ( x ) - \ell ( x ) , 2 w _ { n } ( x ) \} = \operatorname* { m i n } \{ W ( x ) , 2 \beta _ { N } s _ { n } ( x ) + 2 \zeta _ { N } \} \leq 2 \varepsilon .
$$

$n = 0$ , then $2 \beta _ { N } s _ { 0 } ( x ) + 2 \zeta _ { N } \ge 2 \beta _ { N } \ge 2 B \ge W ( x )$ , so the maintained width $W ( x )$ itself is at most 2ε. In this case, the phase has cost zero and requires no new confidence event.

Finally, suppose that D is covered by $J \geq 1$ balls of radius r in a fixed bounded range, and let $n \geq 1$ . By Lemma B.2 with $A = D$ and $K = J$ ，

$$
\gamma _ { n } ( D ; \lambda ) \leq C \log ^ { 2 } ( 2 N ) \big [ J + J ^ { 2 \nu / ( 2 \nu + d ) } r ^ { 2 \nu d / ( 2 \nu + d ) } n ^ { d / ( 2 \nu + d ) } \big ] .
$$

Inserting this bound into (B.12) gives

$$
n \leq b _ { 0 } + b _ { 1 } n ^ { q } , b _ { 0 } = C J \mathcal { V } _ { N } \varepsilon ^ { - 2 } , b _ { 1 } = C \mathcal { V } _ { N } J ^ { 2 \nu / ( 2 \nu + d ) } r ^ { 2 \nu d / ( 2 \nu + d ) } \varepsilon ^ { - 2 } , q = \frac { d } { 2 \nu + d } .\tag{C.14}
$$

If $b _ { 0 } \geq b _ { 1 } n ^ { q }$ , then (C.14) gives $n \leq 2 b _ { 0 }$ . Otherwise, $n < 2 b _ { 1 } n ^ { q }$ , and hence $n < ( 2 b _ { 1 } ) ^ { 1 / ( 1 - q ) }$ . Since $1 / ( 1 - q ) = 1 + d / ( 2 \nu )$ 2

$$
b _ { 1 } ^ { 1 + d / ( 2 \nu ) } = C \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } J ^ { \frac { 2 \nu } { 2 \nu + d } . \frac { 2 \nu + d } { 2 \nu } } r ^ { \frac { 2 \nu d } { 2 \nu + d } . \frac { 2 \nu + d } { 2 \nu } } \varepsilon ^ { - 2 ( 1 + d / ( 2 \nu ) ) } = C \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } J r ^ { d } \varepsilon ^ { - ( 2 + d / \nu ) } .
$$

Combining the two cases, $n \leq 2 b _ { 0 } + ( 2 b _ { 1 } ) ^ { 1 + d / ( 2 \nu ) }$ , which yields (B.13).

## C.6 Proof of Lemma B.4

Proof of Lemma B.4. Work on ${ \mathcal { E } } _ { \delta }$ . Algorithm 1 starts a phase only if its accuracy level is at least $2 \zeta _ { N }$ and otherwise allocates the remaining evaluations of the track to the saved estimated optimal point. Hence all phase h that completes or contains a sampled point has $\varepsilon _ { h } \geq 2 \zeta _ { N }$ , and Proposition A.2 applies. If phase h completes, the first three bounds in Proposition $\mathrm { A . 2 }$ , with $W _ { h } = u _ { h } - \ell _ { h } ,$ , are the three bounds claimed in the lemma. Any point x sampled in phase h satisfies $f ^ { \star } - f ( x ) \leq T + L _ { \vartheta } \varepsilon _ { h }$ by (A.7), and $f ^ { \star } - f ( x ) \leq 2 B$ because $| f | \le B$ on $\mathcal { X }$ by (4). □

## C.7 Proof of Lemma B.5

Proof of Lemma B.5. Both bounds follow from (B.13) in Lemma B.3, applied to the set $D _ { h } = \{ x \in$ $\mathcal { X } : G _ { h } ( x ) \geq 2 \varepsilon _ { h } , W _ { h - 1 } ( x ) \geq 2 \varepsilon _ { h } \}$ of phase $h ,$ with a single ball covering X for the global bound and with the cover in Assumption 5.4 for the localized bound. Let $\varepsilon _ { h } \geq 2 \zeta _ { N }$ . Since $\beta _ { N } \geq B , \varepsilon _ { h } \leq 2 B$ and $N \geq 2$

$$
\begin{array} { r } { \mathcal { V } _ { N } \varepsilon _ { h } ^ { - 2 } \geq \frac { 1 } { 4 } \log ^ { 2 } ( 2 N ) \geq \frac { 1 } { 4 } \log ^ { 2 } 4 . } \end{array}\tag{C.15}
$$

Hence 1 and $\nu _ { N } \varepsilon _ { h } ^ { - 2 }$ are both at most a fixed constant times $\mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } \varepsilon _ { h } ^ { - p } = ( \mathcal { V } _ { N } \varepsilon _ { h } ^ { - 2 } ) ^ { 1 + d / ( 2 \nu ) }$

For the global bound, $D _ { h } \subseteq { \mathcal { X } }$ is covered by one ball of radius $D _ { \mathcal { X } }$ . Applying (B.13) with $J = 1$ and $r = D _ { \mathcal { X } }$ , and then (C.15), gives

$$
n _ { h } \le C \bigl [ 1 + \mathcal { V } _ { N } \varepsilon _ { h } ^ { - 2 } + \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } D _ { \chi } ^ { d } \varepsilon _ { h } ^ { - p } \bigr ] \le C \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } \varepsilon _ { h } ^ { - p } .\tag{C.16}
$$

This bound holds deterministically, because (B.13) does.

For the localized bound, work on ${ \mathcal { E } } _ { \delta }$ . With $T = 0$ , (B.98) in the proof of Proposition $\mathrm { A . 2 }$ gives $D _ { h } \subseteq S _ { ( L _ { \vartheta } \varepsilon _ { h } ) + }$ . If $L _ { \vartheta } \varepsilon _ { h } \leq u _ { 0 }$ , then (17) with $u = L _ { \vartheta } \varepsilon _ { h }$ shows that $D _ { h }$ is covered by K balls of radius $r _ { h } = c _ { \rho } ( L _ { \vartheta } \varepsilon _ { h } ) ^ { 1 / \alpha } \leq c _ { \rho } u _ { 0 } ^ { 1 / \alpha }$ . Applying (B.13) with $J = K$ and $r = r _ { h } .$ , and using $r _ { h } ^ { d } = c _ { \rho } ^ { d } L _ { \vartheta } ^ { d / \alpha } \varepsilon _ { h } ^ { d / \alpha }$ gives

$$
n _ { h } \le C \bigl [ 1 + K \mathcal { V } _ { N } \varepsilon _ { h } ^ { - 2 } + K \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } \varepsilon _ { h } ^ { - ( 2 + d / \nu - d / \alpha ) } \bigr ] \le C K \mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } \varepsilon _ { h } ^ { - a } ,\tag{C.17}
$$

where the last step uses (B.2), (C.15), $K \geq 1$ , the definition of $^ { a , }$ and $\varepsilon _ { h } \leq 2 B$

Finally, inserting the bound $\mathcal { V } _ { N } ^ { 1 + d / ( 2 \nu ) } \le C \Lambda _ { N , \delta }$ from (B.2) into (C.16) and (C.17) gives (B.22) and (B.23). □

## D Additional Numerical Details and Results

We first provide basic experimental details, including benchmark definitions, experimental settings, query point search and the evaluation metrics. We then present the additional Branin-2 results followed by the Hartmann-3 results. All experiments run 30 independent macro-replications. Tables and curves report means ± sample standard errors over completed runs. $\mathrm { O p t }$ , Joint-Free, and Reg abbreviate the corresponding SASORIs.

## D.1 Benchmark Definitions and Settings

Branin-2. For $z = ( z _ { 1 } , z _ { 2 } ) \in [ - 5 , 1 0 ] \times [ 0 , 1 5 ]$ , define

$$
B _ { \mathrm { B r } } ( z ) = \left( z _ { 2 } - \frac { 5 . 1 z _ { 1 } ^ { 2 } } { 4 \pi ^ { 2 } } + \frac { 5 z _ { 1 } } { \pi } - 6 \right) ^ { 2 } + 1 0 \left( 1 - \frac { 1 } { 8 \pi } \right) \cos z _ { 1 } + 1 0 .\tag{D.1}
$$

We maximize $f ( x ) = - B _ { \mathrm { B r } } ( 1 5 x _ { 1 } - 5 , 1 5 x _ { 2 } ) / 2 0 0$ on $[ 0 , 1 ] ^ { 2 }$ . Its reference maximum is $f ^ { \star } = - 5 / ( 8 0 0 \pi )$ attained at the three transformed Branin minimizers. Multiplying a reported gap or tolerance by 200 converts it to a diference in the original Branin response.

Hartmann-3. On $[ 0 , 1 ] ^ { 3 }$ , define

$$
H _ { 3 } ( x ) = - \sum _ { i = 1 } ^ { 4 } a _ { i } \exp \left\{ - \sum _ { j = 1 } ^ { 3 } A _ { i j } ( x _ { j } - P _ { i j } ) ^ { 2 } \right\} , \qquad f ( x ) = - H _ { 3 } ( x ) / 4 ,\tag{D.2}
$$

where

$$
a = ( 1 , 1 . 2 , 3 , 3 . 2 ) ,
$$

$$
A = \left( { \begin{array} { r r r } { 3 } & { 1 0 } & { 3 0 } \\ { 0 . 1 } & { 1 0 } & { 3 5 } \\ { 3 } & { 1 0 } & { 3 0 } \\ { 0 . 1 } & { 1 0 } & { 3 5 } \end{array} } \right) , \qquad P = 1 0 ^ { - 4 } \left( { \begin{array} { r r r } { 3 6 8 9 } & { 1 1 7 0 } & { 2 6 7 3 } \\ { 4 6 9 9 } & { 4 3 8 7 } & { 7 4 7 0 } \\ { 1 0 9 1 } & { 8 7 3 2 } & { 5 5 4 7 } \\ { 3 8 1 } & { 5 7 4 3 } & { 8 8 2 8 } \end{array} } \right) .
$$

Numerical refinement gives the fixed evaluation reference

$$
\begin{array} { r l } & { x ^ { \star } \approx ( 0 . 1 1 4 5 8 8 8 7 7 , 0 . 5 5 5 6 4 8 8 9 5 , 0 . 8 5 2 5 4 6 9 8 5 ) , } \\ & { f ^ { \star } \approx 0 . 9 6 5 6 9 4 9 4 6 8 . } \end{array}
$$

This reference is numerical, whereas the Branin reference is analytic. Neither reference is supplied to the sampling procedure. Both benchmarks use observations $y _ { t } = f ( x _ { t } ) + \varepsilon _ { t }$ with independent $\varepsilon _ { t } \sim N ( 0 , 0 . 0 2 ^ { 2 } )$

Table D.1: Settings for the two benchmarks.
<table><tr><td>Setting</td><td>Branin-2</td><td>Hartmann-3</td></tr><tr><td>Domain</td><td> $[ 0 , 1 ] ^ { 2 }$ </td><td> $[ 0 , 1 ] ^ { 3 }$ </td></tr><tr><td>Objective</td><td> $- B _ { \mathrm { B r } } / 2 0 0$ </td><td> $- H _ { 3 } / 4$ </td></tr><tr><td>Largest tolerance T</td><td>0.10</td><td>0.30</td></tr><tr><td>Evaluation tolerances</td><td>.04, .06, .08, .10</td><td>.12, .18, .24, .30</td></tr><tr><td>Fixed length scale</td><td> $0 . 3 \sqrt { 2 }$ </td><td> $\phantom { - } 0 . 2 \sqrt { 3 }$ </td></tr><tr><td>Joint-Free region quota</td><td>591</td><td>197</td></tr></table>

Table D.1 gives the settings that difer between benchmarks. Both tasks use $N = 3 5 0 0 .$ , a fixed $\mathrm { M a t e r n { - } 5 / 2 }$ kernel with zero prior mean and unit marginal variance, and regularization 0.0004. Joint-Free’s region quota is $m = \lfloor \operatorname* { m i n } \{ N / 2 , \sqrt { N } / T \} \rfloor$

The runs use the fresh GP fits and band intersections in Algorithm 1. Each track has its own phase index and schedule $\varepsilon _ { h } = 2 B \vartheta ^ { - h }$ . Noisy observations from the current phase do not afect its queries. A completed nonempty phase updates the shared bands. An incomplete phase contributes observations to the budget and regret but does not update the lower and upper bound, therefore not contribute to the region identifications.

We use the practical calibration $\beta = \operatorname* { m a x } \{ \beta _ { \mathrm { f l o o r } } , B \}$ }, with $\beta _ { \mathrm { f l o o r } } = 2 . 5$ . The correction $\zeta _ { N }$ is computed from (9) using this multiplier, $\delta = N ^ { - 2 }$ , and $\theta = 1 / 2$ . The value $B = 2$ is a working bound, not necessarily the valid RKHS norm bound of the benchmarks.

SASORI starts with $x _ { 0 } = ( 0 . 1 , \ldots , 0 . 1 )$ and no separate initialization batch. GP-UCB begins with $\operatorname* { m a x } \{ 8 , 2 d \} = 8$ Sobol observations, counted in $N$ , then maximizes $\mu _ { t } + \kappa s _ { t }$ numerically. Sobol-GP uses a scrambled Sobol sequence throughout.

## D.2 Numerical Search for the Next Query Point

Following Remark 4.1, we seek a point whose acquisition value exceeds $2 \varepsilon _ { h }$ . We first evaluate all 1024 scrambled Sobol points. If any point satisfies this condition, we query the point with the largest acquisition value among these candidates.

Otherwise, we apply L-BFGS-B from up to 10 starting points, with at most 100 iterations per restart. Up to two Powell restarts provide a fallback when gradient evaluation or local optimization fails, or when the searches produce no improvement. During either optimizer, the threshold condition is checked at every acquisition evaluation, including evaluations within line searches. Once a qualifying point is found, optimization terminates without waiting for convergence, and we query the point with the largest acquisition value among all candidates evaluated so far.

If this search finds no qualifying point, we perform an additional search review over the full domain before accepting phase completion. This review uses a fresh scramble of 2048 Sobol points, retains the best previous candidates as anchors, and allows up to 20 L-BFGS-B restarts with the same Powell fallback. Restart locations include points from diferent parts of the domain, with the remaining starts selected by acquisition value. The larger sample and more widely distributed restarts reduce the risk of overlooking a qualifying point. The same threshold rule applies during this review: finding such a point resumes querying. Otherwise, the phase satisfies the numerical stopping condition.

## D.3 Evaluation Metrics

Let $L , U$ denote the lower and upper functions used to compute regions, and let $b _ { \mathrm { n u m } } , c _ { \mathrm { n u m } }$ be their numerical maxima. For $\tau > 0$ , we use

$$
I _ { \tau } = \{ x : L ( x ) > c _ { \mathrm { n u m } } - \tau \} , \qquad O _ { \tau } = \{ x : U ( x ) > b _ { \mathrm { n u m } } - \tau \} .\tag{D.3}
$$

For the algorithm output, SASORI constructs its regions from the retained bounds of completed phases, using its own $\beta$ and $\zeta _ { N }$ . The baseline outputs use $L = \mu _ { N } - 3 s _ { N }$ and $U = \mu _ { N } + 3 s _ { N }$ , without the initial range constraint or $\zeta _ { N }$ correction. These are the region outputs reported in the main text.

Under this construction, SASORI’s observations from incomplete phases do not contribute to its reported regions, whereas GP-UCB and Sobol-GP use all collected observations. To provide a fair comparison of region identification using all available observations, we therefore consider an additional report in which we fit one $G P$ to all observations from each method. This includes repeated observations and those collected during incomplete phases, so all $N$ observations contribute to the $\mathrm { G P }$ fit. Using the same kernel and regularization, we construct the bounds as

$$
\begin{array} { r l } & { L ^ { \mathrm { a l l } } ( x ) = \operatorname* { m a x } \{ - B , \mu _ { N } ( x ) - 3 s _ { N } ( x ) - \zeta _ { N } ^ { \mathrm { r e p } } \} , } \\ & { U ^ { \mathrm { a l l } } ( x ) = \operatorname* { m i n } \{ B , \mu _ { N } ( x ) + 3 s _ { N } ( x ) + \zeta _ { N } ^ { \mathrm { r e p } } \} . } \end{array}\tag{D.4}
$$

Here $\zeta _ { N } ^ { \mathrm { r e p } }$ uses $N = 3 5 0 0$ and multiplier 3. Numerical extrema are recomputed, without intersecting these bounds with earlier fresh $\mathrm { G P }$ bands. For SASORI, this changes both the observations used and the construction of the bounds. Baseline GPs already use all observations. Note that this additional report is no more than a diferent way to report the outputs of SASORIs, and it does not change the implementation procedures of SASORIs.

We briefly recall the metrics of region identification and optimization for reference. Write $\Delta ( x ) = f ^ { \star } - f ( x )$ and $A _ { \tau } = O _ { \tau } \setminus I _ { \tau }$ . On the independent evaluation set $Q _ { i }$ , inner recovery is

$$
\mathrm { R e c } _ { Q } ( \tau ) = \frac { | I _ { \tau } \cap \mathcal { S } _ { \tau } \cap Q | } { | \mathcal { S } _ { \tau } \cap Q | } .\tag{D.5}
$$

It is the fraction of truly acceptable evaluation points included in the inner set. The unclassified fraction and empirical region error are

$$
\mathrm { U n c } _ { Q } ( \tau ) = \frac { | A _ { \tau } \cap Q | } { | Q | } ,\tag{D.6}
$$

$$
\eta _ { Q } ( \tau ) = \operatorname* { m a x } _ { x \in A _ { \tau } \cap Q } | \Delta ( x ) - \tau | ,\tag{D.7}
$$

with $\eta _ { Q } ( \tau ) = 0$ when $A _ { \tau } \cap Q$ is empty. Recovery and Unclassified are reported as percentages. The error $\eta _ { Q } ( \tau )$ is in the objective’s units and describes how far the unresolved designs lie from the acceptance threshold. These metrics describe the reported sets on $Q .$

The main tables evaluate all three metrics at $\tau = T$ . For the additional error summary, we compute

$$
\eta _ { Q , \mathrm { m a x } } = \operatorname* { m a x } _ { \tau \in \{ 0 . 4 T , 0 . 6 T , 0 . 8 T , T \} } \eta _ { Q } ( \tau )
$$

within each run, then report the cross-run mean and standard error of $\eta _ { Q , \mathrm { m a x } } / T$ . The maximum is therefore taken before averaging. Each benchmark uses 32768 fixed scrambled Sobol points as $Q .$ shared by all methods and runs and generated independently of sampling.

Simple regret $r _ { t }$ and cumulative regret $R _ { t }$ have the definitions in Section 2. For SASORI, $\widehat { x } _ { t }$ is its latest recommendation, held fixed between completed phases. We also evaluate the best-query simple regret,

$$
r _ { t } ^ { \mathrm { b e s t } } = \operatorname* { m i n } _ { 1 \leq s \leq t } \{ f ^ { \star } - f ( x _ { s } ) \} .\tag{D.8}
$$

Both $r _ { t } ^ { \mathrm { b e s t } }$ and $R _ { t }$ include observations in incomplete phases.

## D.4 Additional Branin-2 Results

## D.4.1 Other Tolerances

We first compare Branin-2 empirical region error across $\tau \in \{ 0 . 0 4 , 0 . 0 6 , 0 . 0 8 , 0 . 1 0 \}$ for the algorithm output and the GP fitted to all observations.

![](images/b1e6c000269c965b2847cb3d94c9def70f804c068f4297217c978c9834b6b422.jpg)

![](images/48afc61fcac67d837bb2d53a704fe34b02f64f64bdd06863e87c5cd0e22cfbaa.jpg)  
Figure D.1: Branin-2 empirical region error $\eta _ { Q } ( \tau )$ : algorithm output (left) and GP fitted to all observations (right). Error bars show one standard error.

Figure D.1 shows that Reg has the smallest mean empirical region error at all four tolerances for the algorithm output, consistent with its emphasis on region estimation accuracy. With all observations fitted to one ${ \mathrm { G P } } ,$ Joint-Free has the smallest mean error. All three SASORI variants have much smaller errors than GP-UCB under both reporting methods.

We next summarize the maximum error across these tolerances as $\eta _ { Q , \mathrm { m a x } } / T .$

Table D.2: Branin-2 maximum empirical region error across the four tested tolerances (mean ± standard error).
<table><tr><td>Method</td><td>Algorithm output:  $\eta _ { Q , \mathrm { m a x } } / T$ </td><td>All observations:  $\eta _ { Q , \mathrm { m a x } } / T$ </td></tr><tr><td>Opt</td><td> $0 . 6 3 8 \pm 0 . 0 1 0$ </td><td> $0 . 5 6 7 \pm 0 . 0 1 0$ </td></tr><tr><td>Joint-Free</td><td> $0 . 6 8 8 \pm 0 . 0 1 7$ </td><td> $0 . 4 3 5 \pm 0 . 0 0 6$ </td></tr><tr><td>Reg</td><td> $0 . 5 2 8 \pm 0 . 0 0 7$ </td><td> $0 . 5 1 4 \pm 0 . 0 0 8$ </td></tr><tr><td>GP-UCB</td><td> $7 . 1 8 6 \pm 0 . 7 4 1$ </td><td> $7 . 1 8 6 \pm 0 . 7 4 1$ </td></tr><tr><td>Sobol-GP</td><td> $0 . 6 7 8 \pm 0 . 0 0 8$ </td><td> $0 . 6 7 8 \pm 0 . 0 0 8$ </td></tr></table>

Table D.2 shows that Reg has the smallest mean maximum error for the algorithm output, while Joint-Free has the smallest value when all observations are used. These summaries are consistent with the error curves in Figure D.1; the maximum is taken within each run before averaging.

We then compare inner recovery across the same tolerances under both reporting methods.

![](images/fc5f53617472e7bc6592906a15461f8447cf48df8dbd1c0f5cb67f286b684459.jpg)

![](images/8cfc748a430b24d604b72b7196ac56a74ab1814f2b6050e57e5ab06a3c8329ca.jpg)  
Figure D.2: Branin-2 inner recovery across tolerances: algorithm output (left) and GP fitted to all observations (right). Error bars show one standard error.

Figure D.2 shows that all three SASORI variants recover more acceptable designs than either baseline across the tested tolerances. Fitting one GP to all observations increases their mean recovery, with Opt and Joint-Free achieving similar recovery at the largest tolerance.

We also compare the fractions of the domain that remain unclassified.

![](images/dea259d18bac57609b268772c15345c44d0152dab90596d27452d36adf9a6060.jpg)

![](images/b266c03ac4a8c4e2331c61f7bd83ba4d5bcb8a79f5105cfe842448ed6fa8c16a.jpg)  
Figure D.3: Branin-2 unclassified fractions across tolerances: algorithm output (left) and GP fitted to all observations (right). Error bars show one standard error.

Figure D.3 shows that Reg has the smallest mean unclassified fraction at the three larger tolerances for the algorithm output, while Joint-Free has the smallest fraction at these tolerances when all observations are used. Resolving membership includes excluding unacceptable designs as well as enlarging the inner set, so the ranking need not match that of inner recovery.

To illustrate these region estimates, we show the algorithm’s sets at $\tau \in \{ 0 . 0 4 , 0 . 0 6 , 0 . 0 8 \}$ for the same macro-replication as in Figure 2.

![](images/ca08af0cd17aa6d205f490f286aa18779ad3f89eb84f1711e451691a56070d72.jpg)

![](images/16cf4c40b8bbeb889909e592af1e98ad4c20551648d337e058798d501ef3016e.jpg)

![](images/a8bb216db7d64da4fee894de00088fa35101bc3ca2e09ba012e6f2a9cf77eb00.jpg)

![](images/e0eee78c5e64fddf46d2b861239f835ed1ae6da70c2bb82479725ebe8d6533ee.jpg)

![](images/fa4e74ce591a60de63d63b6bb8fcaf392ed3a3bfb7bb6b161bdb316605363c46.jpg)

![](images/e291ab288e985f4bf565568d438fed8e00b82269251c2d4f4f5396807dc97f89.jpg)

![](images/52ead49c0dc8d71c201ed946faa7056f5fde435d7c87082663d4622ed809e08b.jpg)

![](images/91dded7f58e15783fbeee732b6273ef8d914b10e4a8da1a21a7785bd7e6dc171.jpg)

$$
S _ { \tau }
$$

![](images/a3ae3cc6805860047c50d9cf801ce2e93319cc09be4770f2d54b612bb0694555.jpg)  
Figure D.4: Branin-2 algorithm regions at three additional tolerances.

Figure D.4 shows that the inner sets expand around the three maxima as τ increases. Opt recovers the largest portion of the acceptable region in this example, while Reg leaves less of the domain unclassified at $\tau = 0 . 0 6$ and 0.08. As in the main illustration, the green sets identify acceptable alternatives and the yellow sets show unresolved membership.

## D.4.2 Final Regret

We next report the final regret values corresponding to the Branin-2 curves in Figure 3.

Table D.3: Branin-2 final regret (mean ± standard error).
<table><tr><td>Method</td><td> $1 0 ^ { 3 } r _ { N }$ </td><td> $1 0 ^ { 5 } r _ { N } ^ { \mathrm { b e s t } }$ </td><td> $R _ { N }$ </td></tr><tr><td>Opt</td><td> $1 . 9 3 8 \pm 0 . 3 5 6$ </td><td> $3 . 8 5 0 \pm 0 . 5 8 0$ </td><td> $1 5 2 . 7 7 \pm 2 . 1 8$ </td></tr><tr><td>Joint-Free</td><td> $2 . 2 4 3 \pm 0 . 4 3 7$ </td><td> $4 . 8 5 4 \pm 0 . 9 5 5$ </td><td> $2 3 1 . 5 5 \pm 2 . 9 4$ </td></tr><tr><td>Reg</td><td> $1 . 2 4 4 \pm 0 . 2 5 0$ </td><td> $4 . 5 3 6 \pm 0 . 9 6 4$ </td><td> $3 1 8 . 7 2 \pm 1 . 6 9$ </td></tr><tr><td>GP-UCB</td><td> $0 . 3 5 6 \pm 0 . 0 8 1$ </td><td> $0 . 1 4 6 \pm 0 . 0 4 8$ </td><td> $1 7 . 3 7 \pm 0 . 4 6$ </td></tr><tr><td>Sobol-GP</td><td> $2 . 1 2 4 \pm 0 . 4 3 1$ </td><td> $5 . 1 0 7 \pm 0 . 8 3 5$ </td><td> $9 4 3 . 3 4 \pm 0 . 1 1$ </td></tr></table>

Table D.3 shows that Opt has the lowest mean best query and cumulative regret among the SASORI variants, while Reg has the lowest mean regret at the final recommendation. The latter may reflect how regional exploration also improves the confidence bounds used to select the recommended point.

## D.4.3 Running Time

We finally report the recorded sampling times for the five methods on Branin-2.

Table D.4: Branin-2 recorded sampling time in minutes (mean ± standard error).
<table><tr><td>Method</td><td>Minutes</td></tr><tr><td>Opt</td><td> $4 7 . 8 9 \pm 0 . 3 8$ </td></tr><tr><td>Joint-Free</td><td> $4 4 . 7 5 \pm 0 . 4 0$ </td></tr><tr><td>Reg</td><td> $4 8 . 4 2 \pm 0 . 4 0$ </td></tr><tr><td>GP-UCB</td><td> $1 1 9 . 0 1 \pm 4 . 2 9$ </td></tr><tr><td>Sobol-GP</td><td> $0 . 2 4 \pm 0 . 0 1$ </td></tr></table>

Table D.4 shows that the three SASORI variants have similar recorded sampling times and take less time than GP-UCB in these runs. Sobol-GP requires the least computation time because it does not optimize an acquisition function.

## D.5 Additional Hartmann-3 Results

## D.5.1 Other Tolerances

We first compare Hartmann-3 empirical region error across $\tau \in \{ 0 . 1 2 , 0 . 1 8 , 0 . 2 4 , 0 . 3 0 \}$ for the algorithm output and the GP fitted to all observations.

![](images/d2895abbf9cfba1b53bf96b22ac4e9f717a072997aa31900667b52eecc488192.jpg)

![](images/f638456fdb7062bf10f506c211eb5d9ecfe04b0b8527f0f645f6d6ccd7e7b5ac.jpg)  
Figure D.5: Hartmann-3 empirical region error $\eta _ { Q } ( \tau )$ : algorithm output (left) and GP fitted to all observations (right). Error bars show one standard error.

Figure D.5 shows that all three SASORI variants have smaller mean empirical region errors than either baseline across the tested tolerances under both reporting methods. Opt has the smallest mean error at the smallest tolerance, whereas Reg has the smallest error at the three larger tolerances. Reg’s errors vary little across these larger tolerances, consistent with its emphasis on region estimation accuracy.

We next summarize the maximum error across these tolerances as $\eta _ { Q , \mathrm { m a x } } / T .$

Table D.5: Hartmann-3 maximum empirical region error across the four tested tolerances (mean ± standard error).
<table><tr><td>Method</td><td>Algorithm output:  $\eta _ { Q , \mathrm { m a x } } / T$ </td><td>All observations:  $\eta _ { Q , \mathrm { m a x } } / T$ </td></tr><tr><td>Opt</td><td> $0 . 5 7 1 \pm 0 . 0 0 6$ </td><td> $0 . 6 5 5 \pm 0 . 0 7 1$ </td></tr><tr><td>Joint-Free</td><td> $0 . 5 9 3 \pm 0 . 0 0 9$ </td><td> $0 . 5 2 2 \pm 0 . 0 0 8$ </td></tr><tr><td>Reg</td><td> $0 . 4 1 9 \pm 0 . 0 0 6$ </td><td> $0 . 3 6 6 \pm 0 . 0 0 5$ </td></tr><tr><td>GP-UCB</td><td> $2 . 8 1 8 8 1 5 \pm 0 . 0 0 0 0 2 8$ </td><td> $2 . 8 1 8 8 1 5 \pm 0 . 0 0 0 0 2 8$ </td></tr><tr><td>Sobol-GP</td><td> $0 . 7 3 3 \pm 0 . 0 1 7$ </td><td> $0 . 7 3 3 \pm 0 . 0 1 7$ </td></tr></table>

Table D.5 shows that Reg has the smallest mean maximum error under both reporting methods. With all observations, Joint-Free also has a smaller maximum error than $\mathrm { O p t }$ . These values summarize the errors in Figure D.5, with the maximum taken within each run before averaging.

We then compare inner recovery across the same tolerances under both reporting methods.

![](images/eef62c3ffc99a411061015f54cc17f9aafea9a73bcc6d786f751a366685c2a83.jpg)

![](images/a06b015304f836ccb01d037e521b3dff54e506cf8a5a28f4baa783bc0580bf16.jpg)  
Figure D.6: Hartmann-3 inner recovery across tolerances: algorithm output (left) and GP fitted to all observations (right). Error bars show one standard error.

Figure D.6 shows that all three SASORI variants recover more acceptable designs than either baseline across the tested tolerances. Fitting one GP to all observations increases their mean recovery, with Opt and Joint-Free performing similarly.

We also compare the fractions of the domain that remain unclassified.

![](images/d49099865c069830556579780b418b0b8914cbd5fea80007988bce4aa0032db6.jpg)

![](images/2f549c4cd8806d50d51e013e656719778da18e85de2649461c99a2e032f61a5e.jpg)  
Figure D.7: Hartmann-3 unclassified fractions across tolerances: algorithm output (left) and GP fitted to all observations (right). Error bars show one standard error.

Figure D.7 shows that Reg has the smallest mean unclassified fraction at the three larger tolerances for the algorithm output and at the two largest tolerances when all observations are used. All three variants leave less of the domain unclassified than either baseline. As in Branin-2, Reg resolves membership efectively even when its inner recovery is lower than that of Opt or Joint-Free.

## D.5.2 Final Regret

We next report the final regret values corresponding to the Hartmann-3 curves in Figure 4.

Table D.6: Hartmann-3 final regret (mean ± standard error).
<table><tr><td>Method</td><td> $1 0 ^ { 3 } r _ { N }$ </td><td> $1 0 ^ { 5 } r _ { N } ^ { \mathrm { b e s t } }$ </td><td> $R _ { N }$ </td></tr><tr><td>Opt</td><td> $4 . 3 4 5 \pm 0 . 7 5 3$ </td><td> $6 9 . 9 9 0 \pm 7 . 6 9 0$ </td><td> $6 8 7 . 2 7 \pm 5 . 2 4$ </td></tr><tr><td>Joint-Free</td><td> $4 . 5 9 1 \pm 0 . 9 5 1$ </td><td> $7 1 . 2 5 0 \pm 9 . 4 9 0$ </td><td> $8 2 4 . 0 5 \pm 5 . 8 3$ </td></tr><tr><td>Reg</td><td> $5 . 0 3 8 \pm 0 . 9 1 6$ </td><td> $1 8 5 . 3 0 0 \pm 2 2 . 5 0 0$ </td><td> $1 7 1 1 . 0 4 \pm 6 . 8 4$ </td></tr><tr><td>GP-UCB</td><td> $0 . 5 0 0 \pm 0 . 1 0 4$ </td><td> $3 . 0 4 8 \pm 0 . 8 4 4$ </td><td> $9 0 . 7 2 \pm 0 . 6 5$ </td></tr><tr><td>Sobol-GP</td><td> $2 . 6 8 5 \pm 0 . 4 7 7$ </td><td> $3 8 6 . 3 0 0 \pm 4 3 . 8 0 0$ </td><td> $2 5 5 4 . 4 0 \pm 0 . 0 9$ </td></tr></table>

Table D.6 shows that Opt has the lowest mean recommendation, best query, and cumulative regret among the SASORI variants. Joint-Free maintains similar recommendation quality while incurring substantially less cumulative regret than Reg.

## D.5.3 Running Time

We finally report the recorded sampling times for the five methods on Hartmann-3.

Table D.7: Hartmann-3 recorded sampling time in minutes (mean ± standard error).
<table><tr><td>Method</td><td>Minutes</td></tr><tr><td>Opt</td><td> $4 7 . 6 2 \pm 0 . 6 3$ </td></tr><tr><td>Joint-Free</td><td> $4 4 . 3 3 \pm 0 . 6 0$ </td></tr><tr><td>Reg</td><td> $4 5 . 8 2 \pm 0 . 5 7$ </td></tr><tr><td>GP-UCB</td><td> $1 4 1 . 3 3 \pm 4 . 9 1$ </td></tr><tr><td>Sobol-GP</td><td> $0 . 3 1 \pm 0 . 0 1$ </td></tr></table>

Table D.7 shows similar recorded sampling times for the three SASORI variants, all below GP-UCB in these runs. Sobol-GP is fastest because its query locations do not require acquisition optimization.