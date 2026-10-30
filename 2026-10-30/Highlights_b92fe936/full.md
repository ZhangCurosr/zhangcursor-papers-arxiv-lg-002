## Highlights

Carbon-Aware Electric Vehicle Routing Problem with Decision-Focused Learning Jaume Reixach, Wanru Chen, Jiaming Wu, Balázs Kulcsár

• We formulate a carbon-aware EVRP with time-varying grid carbon intensity.

• We embed a time-limited LNS and exact postprocessing in approximate SPO+ training.

• Carbon-aware routing reduces charging-related emissions by 28.76% on average.

• Decision-focused learning reduces realized cost by 2.30% on average.

• Its benefits persist across solver configurations and carbon-cost weights.

# Carbon-Aware Electric Vehicle Routing Problem with Decision-Focused Learning

Jaume Reixach<sup>a,b</sup>, Wanru Chen<sup>a,b</sup>, Jiaming Wu<sup>a,∗</sup> and Balázs Kulcsár<sup>b</sup>

<sup>a</sup>Architecture and Civil Engineering, Chalmers University ofTechnology, Gothenburg, Sweden

<sup>b</sup>Electrical Engineering, Chalmers University ofTechnology, Gothenburg, Sweden

## A R T I C L E I N F O

Keywords: Electric Vehicle Routing Problem Predict-then-Optimize Decision-Focused Learning Carbon-Aware Routing

## A BS T R AC T

Electric vehicles produce no tailpipe emissions, but charging them produces emissions that depend on the time-varying generation mix of the electricity consumed. We study an Electric Vehicle Routing Problem with a carbon-aware objective that accounts for charging-related emissions alongside conventional routing costs. The problem determines routing, charging, and waiting decisions, allowing charging demand to be shifted toward periods with lower grid carbon intensity. Because these decisions depend on forecasts of future carbon intensity, we propose a decision-focused learning approach so that the training of the model employed for predicting the carbon intensity of the grid over time is solver-aware. In particular, we train a linear predictor using approximate SPO+ directions and employ a time-limited Large Neighborhood Search followed by exact postprocessing as the underlying solver. Experiments on realistic Germanybased instances show that explicitly considering charging-related emissions reduces realized emissions by 28.76% on average relative to carbon-blind optimization, although at the expense of higher non-carbon operating cost. Moreover, decision-focused training reduces realized total cost by 2.30% on average relative to conventional prediction-focused learning and performs better on 75.5% of the test instances, despite substantially lower pointwise predictive accuracy. Finally, we show that its average advantage remains positive across varied solver configurations and carbon-cost weights.

## 1. Introduction

Road transport electrification is an important strategy for reducing greenhouse gas emissions. In this context, the EU Electrification Action Plan (European Commission, 2026) proposes increasing electricity’s share of the EU’s final energy consumption to 46% by 2040. Transport is particularly relevant to this transition, as it accounts for almost one third of final energy consumption in the EU, approximately 75% of which is attributable to road transport (Eurostat, 2026). Electrifying road transport can therefore play an important role in achieving the EU’s broader electrification objectives. However, although electric vehicles produce no tailpipe emissions, their operation still generates indirect emissions through the production of electricity used for charging. Reducing these emissions requires attention to both electricity generation and how vehicles charge.

Charging-related emissions depend on both the amount of electricity consumed and its carbon intensity, which varies over time with the electricity generation mix. Consequently, two routing solutions with similar energy consumption can produce substantially diferent emissions. An energy-minimizing solution is therefore not necessarily emission-minimizing, since charging can be shifted toward periods of lower grid carbon intensity. Exploiting this opportunity requires coordinating routing, charging, and waiting decisions while satisfying operational constraints. We therefore study a carbon-aware Electric Vehicle Routing Problem (EVRP) that balances conventional operating costs against charging-related emissions under time-varying grid carbon intensity. Since future carbon intensity is unknown when these decisions are made, this coordination also depends on the forecasts used to guide optimization.

The classical approach to solving such problems is via two sequential steps: first training a predictive model, and then treating the prediction output as an input to an algorithm for solving the optimization problem. This approach, known as prediction-focused learning (PFL) (Mandi et al., 2024), treats prediction and optimization as separate tasks. However, high prediction accuracy does not necessarily translate into better decisions, as prediction errors difer in their impacts on the downstream optimization problem. In contrast, decision-focused learning (DFL) incorporates the downstream optimization problem into model training to improve decision quality (Mandi et al., 2024). In this paper, we develop a DFL-based method for the Electric Vehicle Routing Problem (EVRP), training a linear predictor of grid carbon intensity over the routing horizon to maximize the use of clean electricity for charging. In particular, we train this predictor using approximate SPO+ directions, employing a large neighborhood search (LNS) followed by exact fixed-route optimization of charging quantities and waiting times as the downstream solution method.

The main contributions of this paper are as follows:

• We formulate a carbon-aware EVRP that jointly optimizes routing, charging, and waiting under time-varying grid carbon intensity.

• We develop a scalable solution procedure combining LNS with A<sup>∗</sup> for charging-path construction and exact fixed-route postprocessing. The exact postprocessing step refines the charging quantities and waiting times of the routes produced by the LNS.

• We integrate this solution procedure into an approximate SPO+ training loop, allowing the carbon-intensity predictor to be trained directly with respect to its downstream routing performance.

• We evaluate the approach on realistic instances based on the German road network and electricity data, demonstrating that DFL reduces average realized total cost relative to PFL and other baselines.

The rest of the paper is organized as follows. Section 2 reviews the literature relevant to the study. Section 3 defines the considered EVRP and presents its mixed-integer linear program (MILP) formulation. Section 4 introduces the proposed DFL framework and its downstream solver, including the LNS and the exact fixed-route postprocessing step. Section 5 evaluates the methodology on realistic Germany-based instances and compares it with conventional PFL. Finally, Section 6 summarizes the main conclusions and outlines directions for future research.

## 2. Literature Review

We first review papers where the problem tackled is relevant to the one considered in this work. Then, we present an overview of the most relevant existing methods for DFL. Finally, we review the most relevant applications of DFL, particularly the SPO+ loss, to Vehicle Routing Problems (VRPs) and compare them with the application presented in this paper.

## 2.1. Carbon-Aware Routing and Charging

The Electric Vehicle Routing Problem (EVRP) is a variant of the classical VRP in which the vehicles employed are electric. This variant has become popular in recent years as electric vehicles have been gaining prevalence in real-world routing applications. The main advantages of using electric vehicles in routing applications include lower running costs and lower pollution. Lower costs are due to three main reasons: (i) electricity is usually cheaper than fuel, (ii) electric drivetrains are more eficient, and (iii) electric drivetrains generally require less maintenance than combustion drivetrains. On the other hand, electric vehicles can also produce less pollution as they do not burn fuel during driving, and the electricity used for charging the batteries can be generated from low-polluting sources.

An important detail is that, if the generation of electricity is done using carbon-intensive sources, such as coal, oil or natural gas, the use of electric vehicles can lead to higher lifecycle greenhouse-gas emissions than simply employing eficient hybrid or gas-powered vehicles. This is highlighted in Singh et al. (2024), where the authors show that in some regions of the United States, electric vehicles lead to higher lifecycle greenhouse emissions than some eficient hybrid counterparts due to a high grid carbon intensity. This motivates the main idea of this paper, which is to account for the variability in the carbon intensity of the grid, with the aim of charging during lower carbon intensity periods and ultimately using cleaner electricity for charging.

Most EVRP papers focus on minimizing the amount of vehicles employed, the total energy consumption, the total travelling time, or a combination of these values (Kucukoglu et al., 2021). To our knowledge, there are no EVRP papers focusing on directly minimizing the carbon intensity of the electricity used for charging. In the wider field of routing problems, some similar ideas have been applied to end-to-end routing, for example in Su et al. (2025), the authors consider the time-varying carbon intensity of the grid and present an optimization model to schedule the charging of electric trucks in long-haul end-to-end routes across the United States. Another related paper is Jarvis et al. (2022), which concerns the scheduling of electric bus charging, but instead of explicitly considering the carbon intensity of the grid over the routing horizon, they predict the excess in clean energy originating from wind generation and aim to maximize the charging done during periods with a surplus of clean energy.

One of the EVRP papers conceptually closest to ours is Miyabe et al. (2025), in which the authors consider the coordination of charging with solar energy production. In particular, they consider time-varying local surplus power from solar generation associated with each charging station, aiming to maximize the use of solar power for charging. Our paper instead focuses on treating the carbon intensity of the whole grid as uncertain and on forecasting it before making routing decisions. We consider a single forecasting model with the task of predicting the carbon intensity of the grid over time, while in the compared work they use an independent forecasting model for each station, which gathers features from a local area around it. Moreover, in the present paper we apply DFL to make the training process aware of the downstream routing solver.

## 2.2. Decision-Focused Learning

Many real-world optimization problems follow a similar structure, in which the formulation of the problem depends on a vector of uncertain values that need to be predicted. Most commonly, these uncertain values appear in the objective function. Classical examples of this setting include routing applications in transport where travel times are not known in advance and portfolio allocation problems, in which the future value of assets is also unknown and needs to be predicted instead.

The structure of these types of problems is generally denoted as Predict-then-Optimize and the classical way of solving them consists of treating the prediction and optimization steps as independent. As mentioned in the introduction, this approach is usually denoted as prediction-focused learning (PFL). Under this approach, the prediction model is trained independently of the optimization problem, generally via standard supervised learning aiming to maximize prediction accuracy.

Decision-focused learning (DFL) is an alternative to PFL that consists of making the training of the prediction model aware of the optimization problem. The main idea is to train the prediction model so that the performance of the downstream optimization algorithm is maximized, without explicitly optimizing its pointwise predictive accuracy.

The motivation behind this approach lies in the fact that small prediction errors can lead to big drops in downstream performance. The main challenge in DFL is that training requires information about how the solution of the downstream optimization problem changes with the predicted parameters. This solution mapping is generally available only implicitly through an optimization solver and for combinatorial optimization problems it is often discontinuous and piecewise constant, so its derivatives are either undefined or equal to zero almost everywhere.

Following the taxonomy by Mandi et al. (2024), gradient-based DFL methods can be organized into four families that apply to diferent classes of downstream optimization problems and obtain informative gradients in diferent ways. Broadly, these approaches seek to minimize decision regret, which is the increase in the true objective value caused by using the decision induced by the predicted coeficients rather than an optimal decision under the true coeficients.

First, we have approaches based on analytical diferentiation that diferentiate optimality conditions or iterations of a solution algorithm of continuous optimization problems (Amos and Kolter, 2017; Kotary et al., 2023; Domke, 2012). The second consists of methods that replace the original optimization problem with a smooth approximation, so that the resulting solution mapping can be diferentiated (Wilder et al., 2019; Ferber et al., 2020). The third consists of perturbation-based methods that estimate gradients from the solutions obtained with the initial parameters perturbed (Domke, 2010; Berthet et al., 2020; Niepert et al., 2021). Finally, surrogate-loss methods replace decision regret with a tractable loss designed to promote downstream solution quality (Elmachtoub and Grigas, 2022; Mulamba et al., 2021; Mandi et al., 2022; Shah et al., 2022). These approaches difer in their assumptions about the optimization problem, their need for access to solver internals, and the number and type of optimization calls required during training.

In general, there is no universally preferable DFL method: the appropriate choice depends on the downstream problem, the predictive model, the available access to the optimizer, and the computational cost of solving it.

Among these approaches, the surrogate loss SPO+ (Elmachtoub and Grigas, 2022) is particularly compatible with the problem considered in this work. This is because the uncertain carbon-intensity vector enters the downstream EVRP linearly through the amount of energy charged in each time period, while the feasible region is independent of this prediction. Furthermore, SPO+ requires only solutions under modified objective coeficients and not diferentiation through the internal operations of the solver, which makes it suitable for a problem involving discrete routing decisions. Although alternative perturbation-based DFL methods could also be applied, SPO+ provides a direct, theoretically motivated surrogate for decision regret while requiring comparatively few calls to the computationally expensive downstream solver.

The original SPO+ construction assumes access to an exact optimization oracle. In our setting, repeatedly solving the full EVRP to optimality during training would be computationally impractical. We therefore use an approximate downstream solver consisting of a time-limited LNS followed by exact fixed-route postprocessing. The resulting update directions are approximate SPO+ subgradients and their efectiveness should therefore be assessed empirically as the theoretical guarantees established under exact optimization are not necessarily inherited. Our experimental evaluation shows that training with these approximate subgradients efectively improves downstream solution quality relative to conventional PFL.

## 2.3. Decision-Focused Learning for Vehicle Routing

Applications of DFL to vehicle routing remain relatively limited compared with applications to simpler combinatorial problems such as shortest paths. The work most closely related to our setting is Chu et al. (2023), which considers a last-mile delivery problem formulated as a capacitated VRP with uncertain travel times. The authors train a linear travel-time prediction model using SPO+ and employ a simulated annealing heuristic to obtain the routing solutions required during training. This combination of SPO+, vehicle routing, and a heuristic optimization oracle is particularly close to the approach adopted in this paper. In both cases, the use of a heuristic makes it possible to perform the repeated routing computations required by decision-focused training without solving an exact formulation at every iteration. The main diferences concern the predicted quantity and the downstream problem: Chu et al. (2023) predict arc travel times for a last-mile CVRP and use simulated annealing, whereas we predict time-dependent grid carbon intensities for an EVRP with charging decisions and use a time-limited LNS followed by exact fixed-route postprocessing as the downstream solver.

Another direct application of DFL to vehicle routing is presented by Mandi et al. (2021). Their objective is to learn implicit preferences from historical routes, represented through predicted arc probabilities, and incorporate these into a CVRP. Rather than using SPO+, they treat an exact mathematical-programming solver as a black box and obtain surrogate gradients through additional solves with perturbed objective coeficients. This work demonstrates the applicability of DFL to a CVRP, but difers from ours both in its learning objective and in its reliance on exact optimization. It also illustrates the importance of the amount of available training data, since the authors report that the decision-focused approach does not improve out-of-sample route quality in their limited-data experiments.

DFL has also been applied to other related routing settings. Tian et al. (2023) consider a maritime routing problem in which port-state-control oficers must select and sequence ship inspections based on predictions of ship deficiencies. Since repeatedly solving the NP-hard routing problem during training is computationally expensive, the authors use surrogate losses based on noise-contrastive estimation and a pool of previously generated feasible solutions, thereby reducing the need for reoptimization. Although this is not a vehicle-delivery problem and does not use SPO+, it highlights a central dificulty shared with our setting: incorporating a hard routing problem into the learning loop requires controlling the computational cost of repeated optimization.

Taken together, these studies show that DFL can improve routing decisions, while also emphasizing the need for scalable downstream optimization.

## 3. Problem Description

We consider a carbon-aware electric vehicle routing problem with time windows. A homogeneous fleet serves customers from a central depot while adhering to capacity, battery, time window, and routing horizon constraints. Vehicles leave the depot fully charged and can visit charging stations, where they may wait before charging if beneficial. Upon returning to the depot, vehicles recharge to full capacity. Routing, charging, and waiting decisions are jointly optimized to minimize operational costs and indirect charging emissions. As grid carbon intensity varies over time, the timing of charging can afect the realized emissions even within solutions that consume similar amounts of energy. Figure 1 illustrates how time windows and the time-varying grid carbon intensity can afect charging-station selection. A tight customer time window prevents the vehicle from waiting at the first charging station. The vehicle therefore prefers a later detour over charging immediately at a higher grid carbon intensity.

We denote by 0 and $n + 1$ the depot at the beginning and end of a route, respectively, � the set of customers, and � the set of charging-station copies. Using copies allows a physical charging station to be visited multiple times while retaining a separate arrival time, battery state, and charging decision for each visit. The node set is $V = \{ 0 , n { + } 1 \} \cup C \cup F$

![](images/28849430adf95ae76359bac9b36d4cdc01c3959bd159d007c43aa2e70f6f153d.jpg)  
Figure 1: Example of a carbon-aware routing decision.

We assume a complete directed graph over $V _ { \mathrm { { ; } } }$ excluding self-loops, arcs entering 0, and arcs leaving $n + 1$ . For every admissible arc $( i , j ) , \tau _ { i j }$ and $e _ { i j }$ denote its travel time and energy consumption, respectively. Each customer $i \in C$ has demand $q _ { i }$ , service duration $s _ { i } ,$ and a time window associated $[ T _ { i } ^ { L } , T _ { i } ^ { U } ]$ . Charging stations and depot nodes have no delivery demand, so we set $q _ { i } = 0$ for $i \in F \cup \{ 0 , n + 1 \}$ . All vehicles have delivery capacity $Q _ { \mathrm { m a x } }$ , charging rate $c _ { r } ,$ maximum battery capacity $y _ { \mathrm { m a x } }$ , and minimum permissible state of charge $y _ { \mathrm { m i n } }$

The routing horizon $[ 0 , t _ { \mathrm { m a x } } ]$ is partitioned into time buckets $H ,$ , where bucket $h \in H$ corresponds to the interval $[ \ell _ { h } , u _ { h } )$ and has grid carbon intensity $C I _ { h } .$ , measured in $\mathbf { k g C O _ { 2 } e q / k W h }$ . The objective coeficients $C _ { \mathrm { v e h } } , C _ { \mathrm { t i m e } } ,$ and $C _ { e }$ denote the fixed cost per route, the cost of time, and the electricity price, respectively. Charging-related emissions are converted into monetary cost using the carbon-cost weight $p _ { \mathrm { C O _ { 7 } } }$ , measured in $\boldsymbol { \mathrm { \epsilon } } / \mathrm { k g } \mathrm { C O } _ { 2 } \mathrm { e q }$

We formulate the problem as a mixed-integer linear program (MILP). The binary variable $x _ { i j }$ indicates whether arc $( i , j )$ is traversed. The continuous variables $a _ { i } , b _ { i }$ , and $d _ { i }$ denote arrival, service or charging start, and departure times at node �, respectively. The variables $y _ { i } , L _ { i } , c _ { i }$ , and $w _ { i }$ represent the state of charge upon arrival, cumulative delivered load, energy charged, and waiting time. The binary variable $z _ { i h }$ indicates whether charging at station copy � starts in time bucket ℎ, while $\phi _ { i h }$ denotes the corresponding amount of energy priced using that bucket. For a route whose final arc is $( i , n + 1 )$ , the variables $b _ { i } ^ { D } , w _ { i } ^ { D }$ , and $\bar { c } _ { i } ^ { D }$ describe the start time, waiting time, and amount of terminal recharging at the depot, while $z _ { i h } ^ { D }$ and $\phi _ { i h } ^ { D }$ assign this recharge to a time bucket. The constants $M _ { t }$ and $M _ { e }$ denote suficiently large values; in practice, they can be set to $M _ { t } = t _ { \operatorname* { m a x } } + \operatorname* { m a x } _ { i , j \in V } \tau _ { i , j }$ for the timing constraints and to $\begin{array} { r } { M _ { e } = y _ { \mathrm { m a x } } + \operatorname* { m a x } _ { i , j \in V } e _ { i , j } } \end{array}$ for the battery constraints, or tightened further using arc-specific travel-time and energy bounds. To simplify the notation, we use the convention that arc variables and any summations involving them include only admissible ordered pairs.

The objective minimizes a weighted combination of vehicle, travel-time, waiting, charging-time, electricity, and charging-related emission costs, as shown in Equation (1). The first term counts the number of nonempty routes as the number of arcs leaving the start depot. The second term charges travel time over all selected arcs. The third term penalizes waiting at customers, charging stations, and the terminal depot, as well as charging time at intermediate stations. The final two terms price the energy charged at intermediate charging stations and at the terminal depot, using both the electricity price $C _ { e }$ and the carbon-cost weight $p _ { \mathrm { { C O 2 } } }$ multiplied by the carbon intensity $C I _ { h }$ of the selected time bucket.

$$
\begin{array} { r l } { \operatorname* { m i n } } & { C _ { \mathrm { v e h } } \displaystyle \sum _ { j \in V } x _ { 0 j } + C _ { \mathrm { t i m e } } \displaystyle \sum _ { i , j \in V } \tau _ { i j } x _ { i j } } \\ & { + C _ { \mathrm { t i m e } } \left( \displaystyle \sum _ { i \in C \cup F } w _ { i } + \displaystyle \sum _ { i \in C \cup F } w _ { i } ^ { D } + \displaystyle \sum _ { i \in F } \frac { c _ { i } } { c _ { r } } \right) } \\ & { + \displaystyle \sum _ { i \in F } \displaystyle \sum _ { h \in H } \phi _ { i h } \left( C _ { e } + p _ { \mathrm { C O 2 } } C I _ { h } \right) + \displaystyle \sum _ { i \in C \cup F } \displaystyle \sum _ { h \in H } \phi _ { i h } ^ { D } \left( C _ { e } + p _ { \mathrm { C O 2 } } C I _ { h } \right) . } \end{array}\tag{1}
$$

Note that a charging event is assigned to a single time bucket. Accordingly, the bucket duration is assumed to be at least as long as the maximum charging time. This discretization, aimed at simplifying the model, nevertheless remains approximate, since a charging event starting near the end of a bucket would in reality extend into the subsequent bucket. However, the approximation afects only the portion of the charge extending into that subsequent bucket and is expected to introduce limited error because grid carbon intensity varies continuously over time and therefore generally changes only moderately between contiguous time buckets.

Route connectivity and customer coverage are enforced by the following constraints. Each customer must have exactly one outgoing and one incoming selected arc, ensuring that every customer is visited exactly once. Charging stations are optional: if a copy is entered, it must also be left, and each copy can be used at most once. The final constraint balances the number of routes leaving the start depot with the number of routes arriving at the end depot.

$$
\sum _ { j \in V } x _ { i j } = 1
$$

$$
\forall i \in C\tag{2}
$$

$$
\sum _ { j \in V } x _ { j i } = 1
$$

$$
\forall i \in C\tag{3}
$$

$$
\sum _ { j \in V } x _ { i j } = \sum _ { j \in V } x _ { j i }
$$

$$
\forall i \in F\tag{4}
$$

$$
\sum _ { j \in V } x _ { i j } \le 1
$$

$$
\forall i \in F\tag{5}
$$

$$
\sum _ { j \in V } x _ { 0 j } = \sum _ { i \in V } x _ { i , n + 1 } .\tag{6}
$$

The next constraints track the cumulative demand delivered along each route. The cumulative delivered load is zero at the start depot. At a customer, it must be at least the demand of that customer and cannot exceed the vehicle’s capacity. At a charging station, it is bounded by the same vehicle capacity but its lower bound is 0 as $q _ { i } = 0 \mathrm { i f } i \in F$ The propagation constraint increases the cumulative delivered load by $q _ { j }$ whenever arc $( i , j )$ is selected; again, since $q _ { j } = 0$ for charging stations, the same expression also applies when the next visited node is a charging station.

$$
L _ { 0 } = 0\tag{7}
$$

$$
q _ { i } \le L _ { i } \le Q _ { \mathrm { m a x } }
$$

$$
\forall i \in C \cup F\tag{8}
$$

$$
L _ { j } \ge L _ { i } + q _ { j } - Q _ { \operatorname* { m a x } } ( 1 - x _ { i j } )
$$

$$
\forall i \in V , j \in C \cup F .\tag{9}
$$

$$
L _ { j } \leq L _ { i } + q _ { j } + Q _ { \operatorname* { m a x } } ( 1 - x _ { i j } )
$$

$$
\forall i \in V , j \in C \cup F .\tag{10}
$$

The timing constraints initialize the route at the start depot. When an arc is selected, the arrival time at the next customer or charging station must equal the departure time plus travel time; this is enforced with two big-� inequalities. Service or charging may only start after arrival. Customer service must start inside the corresponding time window. Departure times equal service completion times at customers and charging completion times at charging stations. Waiting is defined as the diference between service or charging start time and arrival time. Finally, any route returning from node � to the end depot must do so within the routing horizon $t _ { \mathrm { m a x } }$

$$
a _ { 0 } = d _ { 0 } = 0\tag{11}
$$

$$
a _ { j } \geq d _ { i } + \tau _ { i j } - M _ { t } ( 1 - x _ { i j } )
$$

$$
\forall i \in V , j \in C \cup F\tag{12}
$$

$$
a _ { j } \leq d _ { i } + \tau _ { i j } + M _ { t } ( 1 - x _ { i j } )
$$

$$
\forall i \in V , j \in C \cup F\tag{13}
$$

$$
b _ { i } \geq a _ { i }
$$

$$
\forall i \in C \cup F\tag{14}
$$

$$
T _ { i } ^ { L } \leq b _ { i } \leq T _ { i } ^ { U }
$$

$$
\forall i \in C\tag{15}
$$

$$
d _ { i } = b _ { i } + s _ { i }
$$

$$
\forall i \in C\tag{16}
$$

$$
d _ { i } = b _ { i } + { \frac { c _ { i } } { c _ { r } } }
$$

$$
\forall i \in F\tag{17}
$$

$$
w _ { i } = b _ { i } - a _ { i }
$$

$$
\forall i \in C \cup F\tag{18}
$$

$$
d _ { i } + \tau _ { i , n + 1 } \leq t _ { \operatorname* { m a x } } + M _ { t } ( 1 - x _ { i , n + 1 } )
$$

$$
\forall i \in C \cup F .\tag{19}
$$

The battery constraints initialize each route with a full battery at the start depot. For selected arcs ending at customers or charging stations, the arrival state of charge is the previous state of charge plus any charge received at the origin, minus the energy required to traverse the arc. The reserve constraint ensures that every selected arc can be traversed while maintaining at least $y _ { \mathrm { m i n } }$ energy. Battery states are bounded by the minimum reserve and the maximum battery capacity. Non-charging nodes cannot receive charge, whereas charging stations may charge up to the remaining battery capacity.

$$
y _ { 0 } = y _ { \mathrm { { m a x } } }\tag{20}
$$

$$
y _ { j } \leq y _ { i } + c _ { i } - e _ { i j } + M _ { e } ( 1 - x _ { i j } )
$$

$$
\forall i \in V , j \in C \cup F\tag{21}
$$

$$
y _ { j } \geq y _ { i } + c _ { i } - e _ { i j } - M _ { e } ( 1 - x _ { i j } )
$$

$$
\forall i \in V , j \in C \cup F\tag{22}
$$

$$
y _ { i } + c _ { i } - e _ { i j } \ge y _ { \operatorname* { m i n } } - M _ { e } ( 1 - x _ { i j } )
$$

$$
\forall i , j \in V\tag{23}
$$

$$
y _ { \operatorname* { m i n } } \le y _ { i } \le y _ { \operatorname* { m a x } }
$$

$$
\forall i \in C \cup F\tag{24}
$$

$$
c _ { i } = 0
$$

$$
\forall i \in C \cup \{ 0 \}\tag{25}
$$

$$
0 \leq c _ { i } \leq y _ { \operatorname* { m a x } } - y _ { i }
$$

$$
\forall i \in F .\tag{26}
$$

Charging at intermediate stations is linked to the time-varying carbon intensity through the following constraints, which assign every visited charging station to exactly one time bucket. The lower and upper bucket bounds force the charging start time $b _ { i }$ to lie in the selected interval $[ \ell _ { h } , u _ { h } )$ . Here, $\varepsilon > 0$ is a small enough constant that ensures the time intervals do not overlap. The variable $\phi _ { i h }$ can be positive only for the selected bucket, and the bucket-level charged energy equals the total amount charged at the charging station.

$$
\sum _ { h \in H } z _ { i h } = \sum _ { j \in V } x _ { i j }
$$

$$
\forall i \in F\tag{27}
$$

$$
\ell _ { h } z _ { i h } \le b _ { i }
$$

$$
\forall i \in F , h \in H\tag{28}
$$

$$
b _ { i } \le u _ { h } - \varepsilon + M _ { t } ( 1 - z _ { i h } )
$$

$$
\forall i \in F , h \in H\tag{29}
$$

$$
0 \leq \phi _ { i h } \leq y _ { \operatorname* { m a x } } z _ { i h }
$$

$$
\forall i \in F , h \in H\tag{30}
$$

$$
c _ { i } = \sum _ { h \in H } \phi _ { i h }
$$

$$
\forall i \in F .\tag{31}
$$

Terminal-depot waiting and recharging are represented separately as follows. When a route returns from node � to the end depot, the model computes the amount of energy required to recharge the vehicle to full after arrival. The first two inequalities define this quantity if arc $( i , n + 1 )$ is selected, and the third constraint forces it to zero otherwise. The next four constraints define the terminal recharge-start time $b _ { i } ^ { D }$ as the depot arrival time $d _ { i } + \tau _ { i , n + 1 }$ plus the optional waiting time $w _ { i } ^ { D }$ . Both variables are forced to zero when the corresponding return arc is not selected. The remaining constraints assign the terminal recharge to the time bucket containing $b _ { i } ^ { D }$ . Thus, a vehicle may wait after returning to the depot and begin recharging in a later time bucket, with this waiting penalized in the objective.

$$
c _ { i } ^ { D } \geq y _ { \operatorname* { m a x } } - y _ { i } - c _ { i } + e _ { i , n + 1 } - M _ { e } ( 1 - x _ { i , n + 1 } )
$$

$$
\forall i \in C \cup F\tag{32}
$$

$$
c _ { i } ^ { D } \leq y _ { \operatorname* { m a x } } - y _ { i } - c _ { i } + e _ { i , n + 1 } + M _ { e } ( 1 - x _ { i , n + 1 } )
$$

$$
\forall i \in C \cup F\tag{33}
$$

$$
0 \leq c _ { i } ^ { D } \leq y _ { \operatorname* { m a x } } x _ { i , n + 1 }
$$

$$
\forall i \in C \cup F\tag{34}
$$

$$
0 \leq b _ { i } ^ { D } \leq t _ { \operatorname* { m a x } } x _ { i , n + 1 }
$$

$$
\forall i \in C \cup F\tag{35}
$$

$$
0 \leq w _ { i } ^ { D } \leq t _ { \operatorname* { m a x } } x _ { i , n + 1 }
$$

$$
\forall i \in C \cup F\tag{36}
$$

$$
b _ { i } ^ { D } \geq d _ { i } + \tau _ { i , n + 1 } + w _ { i } ^ { D } - M _ { t } ( 1 - x _ { i , n + 1 } )
$$

$$
\forall i \in C \cup F\tag{37}
$$

$$
b _ { i } ^ { D } \leq d _ { i } + \tau _ { i , n + 1 } + w _ { i } ^ { D } + M _ { t } ( 1 - x _ { i , n + 1 } )
$$

$$
\forall i \in C \cup F\tag{38}
$$

$$
\sum _ { h \in H } z _ { i h } ^ { D } = x _ { i , n + 1 }
$$

$$
\forall i \in C \cup F\tag{39}
$$

$$
\ell _ { h } z _ { i h } ^ { D } \leq b _ { i } ^ { D }
$$

$$
\forall i \in C \cup F , h \in H\tag{40}
$$

$$
b _ { i } ^ { D } \leq u _ { h } - \varepsilon + M _ { t } ( 1 - z _ { i h } ^ { D } )
$$

$$
\forall i \in C \cup F , h \in H\tag{41}
$$

$$
0 \leq \phi _ { i h } ^ { D } \leq y _ { \operatorname* { m a x } } z _ { i h } ^ { D }
$$

$$
\forall i \in C \cup F , h \in H
$$

$$
c _ { i } ^ { D } = \sum _ { h \in H } \phi _ { i h } ^ { D }\tag{42}
$$

$$
\forall i \in C \cup F .\tag{43}
$$

Finally, the domains of the decision variables are defined below.

$$
x _ { i j } \in \{ 0 , 1 \}
$$

$$
\forall i , j \in V\tag{44}
$$

$$
z _ { i h } \in \{ 0 , 1 \}
$$

$$
\forall i \in F , h \in H\tag{45}
$$

$$
z _ { i h } ^ { D } \in \{ 0 , 1 \}
$$

$$
\forall i \in C \cup F , h \in H\tag{46}
$$

$$
0 \leq a _ { i } , b _ { i } , d _ { i } , w _ { i } \leq t _ { \operatorname* { m a x } }
$$

$$
\forall i \in V\tag{47}
$$

$$
y _ { i } , L _ { i } \geq 0
$$

$$
\forall i \in V\tag{48}
$$

$$
b _ { i } ^ { D } , w _ { i } ^ { D } , c _ { i } , c _ { i } ^ { D } \geq 0
$$

$$
\forall i \in C \cup F\tag{49}
$$

$$
\phi _ { i h } \ge 0
$$

$$
\forall i \in F , h \in H .\tag{50}
$$

$$
\phi _ { i h } ^ { D } \ge 0
$$

$$
\forall i \in C \cup F , h \in H .\tag{51}
$$

## 4. Methodology

This section presents the proposed decision-focused learning framework and its downstream optimization algorithm. Figure 2 summarizes the complete methodology. A linear model predicts grid carbon intensity over the routing horizon, and these predictions parameterize the carbon-related terms of the EVRP objective. A time-limited LNS followed by exact fixed-route postprocessing is used as the downstream solver. As already mentioned, this approximate downstream solver is necessary because repeatedly solving the full MILP to optimality for every training instance and parameter update would be computationally impractical for the realistic problem instances we are interested in.

During training, this solver is applied under transformed carbon-intensity vectors to construct approximate SPO+ subgradients. At the end of each epoch, the current predictor is evaluated on the validation set by computing a validation regret. This value is the average relative diference between the costs of the solutions obtained from the current predictions and those of the corresponding precomputed reference solutions, with both evaluated under the true carbon intensities. Validation regret is used for model selection.

During evaluation, the trained predictor supplies forecasts to the same algorithm, and the resulting solutions are evaluated under the realized carbon intensities.

## 4.1. Approximate SPO+ Training

As mentioned in the introduction, the SPO+ loss falls into the family of DFL methods that diferentiate a surrogate loss rather than the optimization mapping itself. In particular, the SPO+ loss is a surrogate of the so-called true Smart Predict-then-Optimize (SPO) loss (Elmachtoub and Grigas, 2022). Consider an optimization problem of the form

$$
\begin{array} { r } { z ^ { * } ( c ) : = \underset { w } { \operatorname* { m i n } } \quad c ^ { T } w } \\ { \mathrm { s . t . } \quad w \in S , } \end{array}\tag{52}
$$

where $w \in \mathbb { R } ^ { d }$ are the decision variables, $S \subseteq \mathbb { R } ^ { d }$ is the feasible region, and $c \in \mathbb { R } ^ { d }$ is the objective-coeficient vector, which corresponds to the carbon intensity forecast in our setting. We denote by $\begin{array} { r } { W ^ { * } ( c ) = \arg \operatorname* { m i n } _ { w \in S } \{ c ^ { T } w \} } \end{array}$ the set of optimal solutions to (52) for a cost vector �. The SPO loss, which the SPO+ loss is a surrogate of, is the following one:

$$
\ell _ { \mathrm { S P O } } ( \hat { c } , c ) = \operatorname* { m a x } _ { w \in W ^ { * } ( \hat { c } ) } \{ c ^ { T } w \} - z ^ { * } ( c )\tag{53}
$$

The goal is to find a prediction ̂� so that the SPO loss (53) is minimized, meaning that the true cost of the optimal solutions under ̂� is as close as possible to the true cost of the optimal solutions under �. Note that the maximum � over $W ^ { * } ( \hat { c } )$ is taken, as there might be multiple optimal solutions for ̂� that have diferent values under �. The issue

![](images/7efa3fe5a3ebd9f32a1497ba963b442e9a508cba68e507dc786b2c801d0bfdcc.jpg)  
Figure 2: Overview of the proposed methodology: (a) decision-focused training, validation, and model selection; (b) evaluation using the trained predictor.

## (a) DFL training

For each mini-batch B of the training set D

with this loss function is that it is hard to minimize directly, as it may not even be continuous in ̂�. To tackle this, the paper authors derive the SPO+ loss, which is a tractable upper bound for it:

$$
\ell _ { \mathrm { S P O + } } ( \hat { c } , c ) = \operatorname* { m a x } _ { w \in S } \{ ( c - 2 \hat { c } ) ^ { T } w \} + 2 \hat { c } ^ { T } w ^ { * } ( c ) - z ^ { * } ( c ) .\tag{54}
$$

In this expression, $w ^ { * } ( \cdot )$ denotes the solution returned by an optimization oracle. Although the maximum over $w \in S$ is generally non-diferentiable, one can easily compute subgradients of (54), which allows us to employ minibatch subgradient descent, as we do in this paper.

In particular, we now assume that a linear predictor ${ \hat { c } } = X \theta$ is used, and that we have a dataset consisting of the features and true values of � for � problem instances: $\{ ( X _ { i } , \boldsymbol { c } _ { i } ) \} _ { i = 1 } ^ { n }$ . Following the Empirical Risk Minimization (ERM) principle, the goal is to minimize the mean loss over the dataset. That is, to solve

$$
\operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { p \times 1 } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell _ { S P O + } ( X _ { i } \theta , c _ { i } ) ,
$$

where $\theta \in \mathbb { R } ^ { p \times 1 }$ are the parameters of the linear regression, and $X _ { i } \in \mathbb { R } ^ { d \times p }$ and $c _ { i } \in \mathbb { R } ^ { d }$ are the features and the true carbon intensity over the time steps of the �-th problem instance, respectively. Accordingly, � is the number of

time steps and $p$ the number of features associated with each time step. In this setting and following the authors, one can see that a valid subgradient of $\ell _ { S P O + } ( X _ { i } \theta , c _ { i } )$ is

$$
2 X _ { i } ^ { T } \big ( w ^ { * } ( c _ { i } ) - w ^ { * } ( 2 X _ { i } \theta - c _ { i } ) \big ) .\tag{55}
$$

The theoretical appeal of SPO+ is that, for a fixed realization �, it is a convex function of the prediction $\hat { c }$ and an upper bound on the SPO loss. Moreover, under suficient conditions and in the population and unrestricted-prediction setting, Elmachtoub and Grigas (2022) establish Fisher consistency of the SPO+ loss with respect to the SPO loss, meaning minimizing the expected SPO+ loss also minimizes the expected SPO loss.

An important detail is that these properties assume that $w ^ { * }$ solves the problem exactly. Our use of an approximate method produces an approximate subgradient instead and therefore the theoretical guarantees established for exact optimization do not directly carry over. Below, we make the resulting subgradient of our setting explicit.

Let $s \in { \mathcal { F } }$ denote a feasible EVRP solution and let $\phi _ { h } ( s )$ be the total amount of energy charged during time bucket $h ,$ including charging at intermediate stations and terminal depot recharging. After collecting all objective terms that do not depend on carbon intensity in $G ( s )$ , the objective of a fixed solution can be written as

$$
G ( s ) + p _ { \mathrm C 0 2 } \sum _ { h \in H } c _ { h } \phi _ { h } ( s ) ,
$$

where � is the carbon-intensity vector and $p _ { \mathrm { { C O 2 } } }$ is the carbon-cost weight. Notably, the objective is afine in the uncertain parameter $c ,$ and the feasible set $_ { \mathcal { F } }$ does not depend on its predicted value, placing the problem within the standard SPO+ setting.

For a prediction ${ \widehat { c } } = X \theta .$ , the SPO+ subgradient of Equation (55) requires solving the downstream problem twice, once under the true cost vector � and once under the transformed cost vector $2 \hat { c } - c$ . Let $s ^ { \mathrm { r e f } }$ be the precomputed solution under the true carbon intensities and let $s ^ { \mathrm { s p o } }$ be the solution returned under the transformed vector. In the implementation, we define

$$
\alpha = \operatorname* { m a x } \left\{ \sum _ { h \in H } \phi _ { h } ( s ^ { \mathrm { r e f } } ) , 1 \right\}
$$

and use the normalized direction

$$
\widetilde { g } _ { \theta } = \frac { 2 } { d \alpha } X ^ { \top } \left( \phi ( s ^ { \mathrm { r e f } } ) - \phi ( s ^ { \mathrm { s p o } } ) \right) .
$$

Dividing by � prevents instances with longer routes or greater energy consumption from dominating an update solely because of their scale. Division by the number of time steps � keeps the direction on a comparable scale across routing horizons of diferent lengths. The subgradient also contains the constant carbon-cost weight $p _ { \mathrm { { C O 2 } } }$ . Since the same carbon-cost weight is used for every training instance, this common positive factor only rescales all parameter updates and is therefore absorbed into the learning rate. The carbon-cost weight is nevertheless included in every LNS execution and thus still afects the routing and charging solutions used to construct the direction.

Reference solutions under the true carbon intensities can be precomputed, so in practice training requires one LNS and postprocessing execution per instance and gradient step. During the training of our experimental evaluation we precompute these solutions using the same time-limited LNS and postprocessing step, although with a higher time limit than the one used to compute $s ^ { \mathrm { s p o } }$

Because the LNS is a time-limited heuristic rather than an exact optimization algorithm, $s ^ { \mathrm { r e f } }$ and $s ^ { \mathrm { s p o } }$ are approximate solutions. The resulting direction should therefore be interpreted as an approximate $\mathrm { S P O + }$ subgradient associated with the time-limited LNS and the postprocessing. Furthermore, in the implementation we set negative components of the transformed carbon-intensity vector to zero because the embedded $\mathbf { A } ^ { * }$ search assumes nonnegative costs. This can be seen in line 6 of Algorithm 1. This projection provides a solver-compatible approximation but adds another layer of approximation as it departs from the exact SPO+ transformation whenever it is active. As reported in the experimental evaluation, this clipping afected less than 0.01% of all transformed vector components in our setting.

Algorithm 1 presents the training loop, which applies mini-batch subgradient descent using the approximate SPO+ subgradient above. $_ D$ and $\mathcal { V }$ are the training and validation sets, respectively, which contain one tuple per problem instance. Each tuple contains (i) features for every time step, (ii) the true carbon intensity vector over the time steps of the problem instance, and (iii) a precomputed solution under the true carbon intensity vector.

Algorithm 1 Approximate SPO+ training loop   
Input: Training set $D = \{ ( X _ { i } , \mathbf { c } _ { i } , s _ { i } ^ { \mathrm { r e f } } ) \} _ { i = 1 } ^ { n }$ , validation set , initial parameters �, learning rate �, number of epochs   
�, batch size �   
Output: Epoch history $\{ ( \theta _ { e } , R _ { e } ) \} _ { e = 1 } ^ { E }$   
1: for $e = 1 , \ldots , E$ do   
2: Randomly shufle    
3: for each mini-batch $B \subseteq D$ of size at most � do   
4: for each instance $i \in \mathcal { B }$ do   
5: ${ \widehat { \mathbf { c } } } _ { i } \gets X _ { i } \theta$   
6: $\mathbf { c } _ { i } ^ { \mathrm { \tiny { s p o } } } \gets \operatorname* { i m a x } \{ 2 \widehat { \mathbf { c } } _ { i } - \mathbf { c } _ { i } , \mathbf { 0 } \}$ componentwise   
7: $s _ { i } ^ { \flat _ { 0 } }$ ← apply\_postprocessing(L $\mathbf { \cdot } \mathsf { N S } ( \mathbf { c } _ { i } ^ { \mathrm { s p o } } ) )$   
8: $\begin{array} { r } { \alpha _ { i } \gets \operatorname* { m a x } \{ \sum _ { h \in H } \phi _ { h } ( s _ { i } ^ { \mathrm { r e f } } ) , 1 \} } \end{array}$   
9: $g _ { i }  \frac { 2 } { d \alpha _ { i } } X _ { i } ^ { \top } ( \phi ( s _ { i } ^ { \mathrm { r e f } } ) - \phi ( s _ { i } ^ { \mathrm { s p o } } ) )$   
10: end for   
11: $g \gets | B | ^ { - 1 } \sum _ { i \in { \mathcal { B } } } g _ { i }$   
12: $\theta  \theta - \eta g$   
13: end for   
14: Compute the average regret $\widehat { R } _ { \mathcal { V } } ( \boldsymbol { \theta } )$ on    
15: Log $R _ { e } \gets \widehat { R } _ { \mathcal { V } } ( \theta ) , \theta _ { e } \gets \theta$   
16: end for   
17: return $\{ ( \theta _ { e } , R _ { e } ) \} _ { e = 1 } ^ { E }$

For each epoch, the training set is shufled and the parameters � are updated once per mini-batch. For an instance � in the mini-batch the transformed cost vector $c _ { i } ^ { \mathrm { s p o } }$ is computed, with all of its negative components set to 0. Then, the LNS and postprocessing are executed with this transformed cost vector and the obtained solution is used together with the precomputed one $s _ { i } ^ { \mathrm { r e f } }$ to compute the normalized approximate subgradient.

The validation set $\dot { \mathcal { V } }$ is employed at the end of every epoch for computing a validation regret, which can be used to detect overfitting and select the trained weights. More specifically, for a validation instance �, let $s _ { i } ^ { \theta }$ denote the solution obtained using the current prediction $X _ { i } \theta _ { : }$ , and let $s _ { i } ^ { \mathrm { r e f } }$ denote the precomputed reference solution obtained using the true carbon-intensity vector $c _ { i }$ . Writing $\mathrm { c o s t } ( s ; c _ { i } )$ for the total cost of solution � evaluated under $c _ { i } ,$ , the percentage validation regret is:

$$
\widehat { R } _ { \mathcal { V } } ( \theta ) = \frac { 1 0 0 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \frac { \mathrm { c o s t } ( s _ { i } ^ { \theta } ; c _ { i } ) - \mathrm { c o s t } ( s _ { i } ^ { \mathrm { r e f } } ; c _ { i } ) } { \mathrm { c o s t } ( s _ { i } ^ { \mathrm { r e f } } ; c _ { i } ) } .
$$

Ideally, each reference solution $s _ { i } ^ { \mathrm { r e f } }$ would be optimal for the true carbon-intensity vector, so that validation regret would be measured relative to a proven optimum. In our experimental evaluation, due to the computational impracticality of obtaining such optimal solutions, we precomputed them with the LNS and the postprocessing step under the same extended computational limit used for computing the reference solutions $s ^ { \mathrm { r e f } }$ involved in the subgradient computation. Consequently, the reported validation regret should be interpreted as a measure of relative heuristic performance rather than exact optimization regret. In our implementation, we select the weights yielding the lowest five-epoch trailing mean validation regret. That is, we select $\theta _ { e ^ { * } }$ , where $e ^ { * } \ = \ \arg \operatorname* { m i n } _ { e \in \{ 5 , . . . , E \} } \frac { 1 } { 5 } \sum _ { j = e - 4 } ^ { e } R _ { j }$ . The trailing mean is used to reduce the efect of the variability introduced by the stochastic solver.

## 4.2. Large Neighborhood Search

Large neighborhood search (LNS) is a metaheuristic that has been widely applied to vehicle routing and other transportation problems (Pisinger and Røpke, 2018). The main idea of the algorithm is to probabilistically destroy and repair a solution at every iteration with the goal of improving its quality over time. After application of the destroy and repair operators, the solution obtained is given a certain probability of replacing the solution kept by the algorithm, which will be used in the next iteration. Thus, solutions of a higher quality are given a higher replacement probability.

Algorithm 2 Large Neighborhood Search   
1: $S \gets$ generate\_initial\_solution()   
2: $S ^ { \mathrm { b s f } }  S$   
3: $T \gets T _ { \mathrm { i n i t } }$   
4: while time limit not reached do   
5: $S ^ { \mathrm { c a n d } }  S$   
6: apply\_destroy $( S ^ { \mathrm { c a n d } } )$   
7: apply\_repair(�<sup>cand</sup>)   
8: $\Delta _ { \mathrm { c o s t } }  S _ { \mathrm { c o s t } } ^ { \mathrm { c a n d } } - S _ { \mathrm { c o s t } }$   
9: acceptance ← simulated\_annealing\_decision $( \Delta _ { \mathrm { { c o s t } } } , T )$   
10: if acceptance then   
11: $S \gets S$ cand   
12: if $S _ { \mathrm { c o s t } } < S _ { \mathrm { c o s t } } ^ { \mathrm { b s f } }$ then   
13: $S ^ { \mathrm { b s f } }  S$   
14: end if   
15: end if   
16: $T \gets T \cdot c _ { f }$   
17: end while   
18: return $S ^ { \mathrm { b s f } }$

The LNS operators destroy and repair parts of a solution by removing and reinserting a subset of the solution components forming it. In VRPs, this usually corresponds to removing and reinserting customers, which is exactly our setting. Our LNS employs two of the most established repair and destroy operators, a regret-� repair and a Shaw destroy (Pisinger and Røpke, 2018).

Algorithm 2 provides pseudocode for the complete LNS. After each destroy-repair operation, the algorithm decides whether to accept the obtained solution as incumbent using simulated annealing (Kirkpatrick et al., 1983).

With more detail, if $\Delta _ { \mathrm { c o s t } } = S _ { \mathrm { c o s t } } ^ { \mathrm { c a n d } } - S _ { \mathrm { c o s t } }$ is the diference between the cost of the new solution $S ^ { \mathrm { c a n d } }$ and the solution � currently being kept by the LNS, simulated annealing gives the following probability of accepting $S ^ { \mathrm { c a n d } }$ and hence setting $S = S ^ { \mathrm { c a n d } }$ :

$$
\begin{array} { r } { p = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } \Delta _ { \mathrm { c o s t } } \leq 0 , } \\ { e ^ { - \Delta _ { \mathrm { c o s t } } / T } , } & { \mathrm { i f ~ } \Delta _ { \mathrm { c o s t } } > 0 . } \end{array} \right. } \end{array}\tag{56}
$$

Here, $T$ is the so-called temperature, which controls how likely a worse solution is to be accepted. This value is initialized at $T _ { \mathrm { i n i t } }$ and decreased at each iteration by multiplying the so-called cooling factor $c _ { f }$ . This way, a worse solution is less likely to be accepted as the iterations progress, with the goal of escaping local minima early on. Notably, both $T _ { \mathrm { i n i t } } > 0$ and $c _ { f } \in ( 0 , 1 )$ are algorithm parameters.

## 4.2.1. A<sup>∗</sup> charging-path construction

When removing or adding customers within a route, the LNS needs to decide whether to insert charging-station stops within the new connections to ensure battery-feasible paths. To do this, we implement an $\mathbf { A } ^ { * }$ search (Hart et al., 1968) to construct a path between two consecutive route nodes. In the $\mathbf { A } ^ { * }$ , a search state is characterized by the current node, arrival time, and state of charge. From each state, the algorithm considers edges either to the target node or to charging stations reachable from the current node. An expansion is feasible only if the vehicle has suficient energy to traverse the arc while maintaining the minimum battery level. The cost of a travel leg $( i , j )$ includes the driving-time cost $C _ { \mathrm { t i m e } } \tau _ { i j }$ . If the successor node is a charging station, the vehicle is charged to full in the path-construction phase; the added cost is the electricity cost, the carbon cost evaluated at the arrival-time bucket, and the charging-time cost. If the successor is the depot, the terminal recharge is also priced using the carbon intensity at the depot arrival time, but no terminal charging-time cost is added. Thus, for a charging amount $\Delta y$ starting in bucket ℎ, the cost contribution is

$$
\Delta y \left( C _ { e } + p _ { \mathrm { C O 2 } } C I _ { h } \right) + C _ { \mathrm { t i m e } } \frac { \Delta y } { c _ { r } } ,
$$

with the last term omitted for terminal depot recharging. States that would violate the target-node time window upper bound are discarded. The priority of a state is $f = g + \eta$ , where � is the accumulated cost and the heuristic is

$$
\eta ( i ) = C _ { \mathrm { t i m e } } \frac { \mathrm { d i s t } _ { \mathrm { g e o } } ( i , t ) } { v _ { \mathrm { m a x } } } ,
$$

where dist $_ \mathrm { g e o } ( i , t )$ is the straight-line distance from the current node � to the target node � and $v _ { \mathrm { m a x } }$ is an upper bound on the vehicle speed. Thus, the heuristic is admissible as it is a lower bound on the travelling cost.

To limit the number of labels explored, the implementation retains only the state with the lowest accumulated cost at each node. Newly generated states whose cost is not lower than the current node label are discarded. Because arrival time and state of charge are not included in this comparison, this is a cost-based state-pruning heuristic rather than a complete multi-resource dominance rule. It reduces the computational cost of the repeated searches performed within the LNS, at the expense of potentially discarding a more expensive state with greater downstream time or battery flexibility.

## 4.2.2. Construction and Neighborhood Operators

The initial solution is generated through sequential insertion. The following process is iterated until all customers are visited: (i) a route visiting the farthest unvisited customer is built, and (ii) customers are iteratively inserted into it, by inserting, at each step, the customer leading to the lowest cost increase. The $\mathbf { A } ^ { * }$ is used for deciding the paths between nodes, so each route remains feasible with respect to battery, time-window, and capacity constraints.

At every LNS iteration, the Shaw destroy operator removes � customers that are related in terms of distance, arrival time, and demand. First, a random customer is added to the set � of customers to remove. Then, a customer is added to � iteratively, until $| B | = q$ . This is done by first selecting a random customer � from � to serve as the reference customer, and then selecting a customer not in � by giving a higher probability to the ones with a lower relatedness score with respect to �.

The relatedness score is measured by:

$$
R ( r , c ) = \phi _ { 1 } d ( r , c ) + \phi _ { 2 } t ( r , c ) + \phi _ { 3 } \ell ( r , c ) ,
$$

where each component is normalized by dividing it by the maximum value. Hereby, $d ( \boldsymbol { r } , \boldsymbol { c } ) , t ( \boldsymbol { r } , \boldsymbol { c } ) , \ell ( \boldsymbol { r } , \boldsymbol { c } )$ are the normalized diferences in geographical distance, arrival time, and demand, respectively. $\phi _ { 1 } , \phi _ { 2 }$ and $\phi _ { 3 }$ are real parameters that weight each relatedness component. Customers are ranked by increasing relatedness score to the reference customer, and the selected rank i

$$
\rho = { \bigl \lfloor } | P | U ^ { p } { \bigr \rfloor } + 1 ,
$$

where $| P |$ is the size of the current candidate pool, $U \sim \mathcal { V } ( 0 , 1 )$ , and $p \geq 1$ is a real algorithm parameter that controls the degree of determinism. A value of $p = 1$ corresponds to random selection while larger values bias the selection more strongly towards customers related to the reference customer.

After selecting the removed customers, the afected routes are rebuilt by deleting the selected customer nodes and recomputing the consecutive subpaths between the remaining nodes using the $\mathbf { A } ^ { * }$ . Since these subpaths are optimized sequentially, a newly computed subpath may leave insuficient battery or time-window slack at its destination for the subsequent subpath. In this case, the procedure backtracks and repeats the preceding $\mathbf { A } ^ { * }$ search with a higher minimum battery level at arrival, up to a fixed retry limit. If failure occurs on the first subpath or the fixed retry limit is reached, the destroy operation is rejected and the $\mathrm { L N S }$ proceeds to the next iteration with the current solution unchanged. Otherwise, empty routes are discarded and the resulting feasible partial solution, together with the removed customers, is passed to the repair operator.

Removed customers are inserted using regret-� repair. For every unvisited customer �, the algorithm identifies its feasible insertion alternatives and computes

$$
r _ { c } = \sum _ { \ell = 2 } ^ { \operatorname* { m i n } ( k , | T _ { c } | ) } ( \Delta _ { \ell c } - \Delta _ { 1 c } ) ,
$$

where $\scriptstyle { \tau _ { c } }$ is its set of feasible insertions and $\Delta _ { l c }$ the cost of its �-best insertion. The customer with the largest regret is inserted at its cheapest feasible position, and the process is iterated until all customers have been reinserted. Feasible insertions are evaluated using $\mathbf { A } ^ { * }$ , and creating a new singleton route is always considered. This option ensures reinsertion whenever the customer is individually feasible.

## 4.3. Fixed-route Charging and Waiting Optimization

An important observation is that the LNS as presented always assumes charging to full and does not allow deliberate waiting. To control the amount charged and the waiting time without introducing these decisions into every destroy and repair evaluation, we introduce a postprocessing step that we apply to the final solution produced by the LNS. This is important, as waiting, when compatible with the time-window constraints, can move a subsequent charge to a period with lower carbon intensity, while partial charging can shift energy consumption from high-carbon-intensity time buckets to lower-carbon-intensity ones.

Postprocessing fixes each route produced by the LNS, including its customer order, travelled arcs, and chargingstation visits, and optimizes only its schedule and charging amounts. Our implementation solves one MILP independently for each route. It associates a waiting variable with every position of the fixed route: at customers this includes waiting required by the lower time-window bound and may also delay later charging, while at the terminal depot it may postpone the terminal recharge.

Consider a fixed route written as a sequence $\boldsymbol { r } = ( v _ { 0 } , \ldots , v _ { m } )$ , where $v _ { 0 }$ and $v _ { m }$ represent the depot, and the sequence includes every intermediate visit to a charging station selected by the $\mathbf { A } ^ { * }$ . Let $P = \{ 0 , \ldots , m \}$ be the set of route positions, $P ^ { - } = \{ 0 , \ldots , m - 1 \}$ the positions with an outgoing leg, $P _ { C } = \{ p \in P : v _ { p } \in C \}$ the customer positions, and $P _ { F } = \{ p \in P : v _ { p } \in F \}$ the charging-station positions. For each $p \in P ^ { - }$ , we define $\tau _ { p } = \tau _ { v _ { p } v _ { p + 1 } }$ and $e _ { p } = e _ { v _ { p } v _ { p + 1 } } .$ As in the full formulation, each time bucket $h \in H$ has lower and upper bounds $\ell _ { h }$ and $u _ { h }$

The continuous variables $a _ { p } , b _ { p } ,$ and $d _ { p }$ are the arrival, activity start (service for customers, charging for the rest), and departure times at route position �; $w _ { p }$ is the waiting time; $y _ { p }$ is the state of charge upon arrival; and $c _ { p }$ is the amount charged. For $p \in P _ { F }$ , the binary variable $z _ { p h }$ selects the time bucket in which charging starts, and $\phi _ { p h }$ is the amount charged and priced in that bucket. The corresponding variables for the terminal depot recharge are denoted by $z _ { h } ^ { D }$ and $\phi _ { h } ^ { D }$ . The terminal recharge amount itself is represented by $c _ { m } .$

The objective aims at minimizing the same weighted costs as the full formulation. Since the route, the chargingstation visits, and all travelled legs are fixed, the vehicle and driving-time costs are constant and can be omitted from the optimization objective. The remaining terms minimize waiting time, charging time at intermediate stations, and the electricity and carbon costs of intermediate and terminal charging:

$$
\begin{array} { r l } { \operatorname* { m i n } } & { C _ { \mathrm { t i m e } } \left( \displaystyle \sum _ { p \in P } w _ { p } + \displaystyle \sum _ { p \in P _ { F } } \frac { c _ { p } } { c _ { r } } \right) } \\ & { + \displaystyle \sum _ { p \in P _ { F } } \displaystyle \sum _ { h \in H } \phi _ { p h } \left( C _ { e } + p _ { \mathrm { C O 2 } } C I _ { h } \right) + \displaystyle \sum _ { h \in H } \phi _ { h } ^ { D } \left( C _ { e } + p _ { \mathrm { C O 2 } } C I _ { h } \right) . } \end{array}\tag{57}
$$

As with the full MILP, the terminal recharge is priced but its duration is not included, since the route is complete when the vehicle returns to the depot.

The following constraints determine the schedule of travel, service, charging, and waiting along the fixed route. The route starts at time zero. At every position, the activity may start no earlier than arrival, and waiting is exactly the diference between these times. Because the route is fixed, arrival at the next position is given directly by the departure time at the current position plus the known travel time. Customer service starts within its time window and lasts for the prescribed service time. At charging stations, departure occurs after the selected amount has been charged at charging speed $c _ { r } .$ . At the terminal depot, departure is identified with the start of terminal recharging because its duration is outside the route horizon.

$$
a _ { 0 } = b _ { 0 } = d _ { 0 } = w _ { 0 } = 0\tag{58}
$$

$$
w _ { p } = b _ { p } - a _ { p }
$$

$$
\forall p \in P\tag{59}
$$

$$
a _ { p + 1 } = d _ { p } + \tau _ { p }
$$

$$
\forall p \in P ^ { - }\tag{60}
$$

$$
T _ { v _ { p } } ^ { L } \leq b _ { p } \leq T _ { v _ { p } } ^ { U }
$$

$$
\forall p \in P _ { C }\tag{61}
$$

$$
d _ { p } = b _ { p } + s _ { v _ { p } } , \quad c _ { p } = 0
$$

$$
\forall p \in P _ { C }\tag{62}
$$

$$
d _ { p } = b _ { p } + { \frac { c _ { p } } { c _ { r } } }
$$

$$
\forall p \in P _ { F }\tag{63}
$$

$$
0 \leq c _ { p } \leq y _ { \operatorname* { m a x } } - y _ { p }
$$

$$
\forall p \in P _ { F }\tag{64}
$$

$$
d _ { m } = b _ { m } , \quad c _ { m } = y _ { \operatorname* { m a x } } - y _ { m } .\tag{65}
$$

Battery charge is propagated along the fixed route as follows. The vehicle leaves the initial depot with a full battery. Along every leg, the arrival state of charge at the next position equals the arrival state at the current position plus any energy charged there, minus the energy required by the leg. Bounds on every $y _ { p }$ enforce both the minimum reserve and the battery capacity. Together with Constraint (64), these constraints also prevent charging beyond full capacity.

$$
y _ { 0 } = y _ { \mathrm { m a x } } , \quad c _ { 0 } = 0\tag{66}
$$

$$
y _ { p + 1 } = y _ { p } + c _ { p } - e _ { p }
$$

$$
\forall p \in P ^ { - }\tag{67}
$$

$$
y _ { \mathrm { m i n } } \le y _ { p } \le y _ { \mathrm { m a x } }
$$

$$
\forall p \in P .\tag{68}
$$

Intermediate charging is assigned to carbon-intensity time buckets through the following constraints. As in the full MILP, each charging event is assigned entirely to the time bucket in which charging begins. The big-� bounds place its charging start time in the selected bucket, while $\phi _ { p h }$ can be positive only for that bucket. The sum of these bucket-level quantities equals the amount charged at the visit. As in the full MILP, $\varepsilon > 0$ is a small constant that prevents the overlap of time buckets.

$$
\sum _ { h \in H } z _ { p h } = 1
$$

$$
\forall p \in P _ { F }\tag{69}
$$

$$
\ell _ { h } z _ { p h } \le b _ { p }
$$

$$
\forall p \in P _ { F } , h \in H\tag{70}
$$

$$
b _ { p } \leq u _ { h } - \varepsilon + M ( 1 - z _ { p h } )
$$

$$
\forall p \in P _ { F } , h \in H\tag{71}
$$

$$
0 \leq \phi _ { p h } \leq y _ { \operatorname* { m a x } } z _ { p h }
$$

$$
\forall p \in P _ { F } , h \in H\tag{72}
$$

$$
c _ { p } = \sum _ { h \in H } \phi _ { p h }
$$

$$
\forall p \in P _ { F } .\tag{73}
$$

The terminal recharge is assigned to the bucket in which it starts. The same linking constraints used for intermediate stations ensure that $\phi _ { h } ^ { D }$ equals $c _ { m }$ in the selected bucket and is zero in every other bucket. In our implementation, we set $M = t _ { \operatorname* { m a x } }$

$$
\sum _ { h \in H } z _ { h } ^ { D } = 1\tag{74}
$$

$$
\ell _ { h } z _ { h } ^ { D } \leq b _ { m }
$$

$$
\forall h \in H\tag{75}
$$

$$
b _ { m } \leq u _ { h } - \varepsilon + M ( 1 - z _ { h } ^ { D } )
$$

$$
\forall h \in H\tag{76}
$$

$$
0 \leq \phi _ { h } ^ { D } \leq y _ { \operatorname* { m a x } } z _ { h } ^ { D }
$$

$$
\forall h \in H\tag{77}
$$

$$
c _ { m } = \sum _ { h \in H } \phi _ { h } ^ { D } .\tag{78}
$$

Finally, the domains of the remaining variables are specified as follows. All time variables are bounded by the route horizon $t _ { \mathrm { m a x } }$ , and all charging variables are bounded by the battery capacity $y _ { \mathrm { m a x } }$ . The only integer variables are the time-bucket selectors:

$$
z _ { p h } \in \{ 0 , 1 \}
$$

$$
\forall p \in P _ { F } , h \in H\tag{79}
$$

$$
z _ { h } ^ { D } \in \{ 0 , 1 \}
$$

$$
\forall h \in H\tag{80}
$$

$$
0 \leq a _ { p } , b _ { p } , d _ { p } , w _ { p } \leq t _ { \operatorname* { m a x } }
$$

$$
\forall p \in P\tag{81}
$$

The complete solution is postprocessed by solving this model separately for every nonempty route. After solving, the implementation reconstructs the route schedule and recomputes its complete cost, including the fixed vehicle and travel costs omitted from (57).

## 5. Experimental Evaluation

We evaluate the proposed methodology on synthetic EVRP instances constructed using real-world German roadnetwork, charging-infrastructure, weather, and electricity-system data. Germany provides a relevant setting because its grid carbon intensity varies substantially over time as the generation mix changes. This temporal variation creates opportunities to reduce charging-related emissions by optimizing routing, charging, and waiting decisions.

We focus on long-haul routes spanning multiple days. The need for en-route charging, together with carbonintensity variations across multiple days, makes this setting suitable for examining how charging locations, quantities, and timing can be coordinated to balance operating costs and emissions.

## 5.1. Problem Instances

The considered problem instances have a 72-hour routing horizon and 50 to 100 customers, distributed throughout Germany. Each customer is associated with a standardized demand between 5 and 30, a service time between 0.5 and 2 hours, and a time window type, with all three assignments done uniformly at random. We consider three types of time windows: (i) a flexible time window covering the entire routing horizon, [0, 72]; (ii) a day-specific time window covering one of the intervals [0, 24], [24, 48], or [48, 72]; and (iii) a working-hours time window within a selected day. For the latter, the opening and closing times are sampled uniformly from hours 5–9 and 17–21 of that day, respectively.

In order to make the time windows more realistic, we bias customers farther from the depot toward later days when the time windows are day-specific or fixed to the working hours of a day. To do this, we consider the straight-line distance between a customer and the depot, normalized by dividing by the maximum distance over all customers. If this value is smaller than 0.7 we assign probabilities 0.4, 0.3, and 0.3 to select the first, second, and third routing days, respectively. Otherwise, we associate the probabilities 0.1, 0.4, and 0.5, respectively.

The locations of depots, customers, and charging stations are based on OpenStreetMap (OSM) data (Open-StreetMap contributors, 2026). Depot and customers are sampled uniformly from OSM elements with the relevant tags. Candidate depot locations are identified using the tags industrial=depot and landuse=depot, while candidate customer locations are identified using shop=wholesale, industrial=warehouse, and building=warehouse.

Candidate charging stations are identified using amenity=charging\_station, as well as amenity=fuel together with charging\_station or fuel:electricity set to true. Afterwards, two filtering steps are applied. First, we retain only stations located within 300 m of a major road, identified by the tags highway=motorway, highway=motorway\_link, highway=trunk, highway=trunk\_link, and highway=primary. This focuses the charging network on stations relevant to long-haul routing. Second, nearby stations are aggregated using DBSCAN (Ester et al., 1996) with a 5 km neighborhood radius measured by routing distance. This reduces the concentration of stations around urban areas and merges multiple stations representing approximately the same charging location. Using routing distance instead of straight-line distance also avoids merging stations that are geographically close but located on opposite sides of a highway. These steps produce a set of 2,231 charging stations.

A further instance-specific filtering step is applied to reduce computational requirements. For each instance, we first retain only stations reachable from at least one customer or the depot. Germany is then divided into $5 0 \times 5 0$ km grid cells and, for each customer and the depot, only the closest charging station within each grid cell is retained. After this filtering, the generated instances contain an average of 2,041 charging stations.

Figure 3 shows the total number of charging stations without any instance-specific filtering, and the customers and depot of an example problem instance with 75 customers.

The Open Source Routing Machine (OSRM) (Luxen and Vetter, 2011) is used for computing the travelling times and distances between each pair of nodes. Let $d _ { i j }$ and $v _ { i j }$ denote the resulting distance and average speed for an edge, respectively, with $v _ { i j }$ clipped to the interval [1, 80] km/h. Energy consumption is afected by vehicle speed, among other operational and road characteristics (Basso et al., 2019).

![](images/b05d154f9162eaab7de7a8ef02c5921312985b4f46ed531223675c364782e65a.jpg)  
Figure 3: Example problem instance used in the experimental evaluation.

Because the detailed speed profiles, road gradients, and acceleration patterns required by a physical consumption model are not available for every edge, we employ a stylized speed-dependent energy-consumption function:

$$
e _ { i j } = d _ { i j } \left( 0 . 8 + 0 . 0 0 0 4 2 ( v _ { i j } - 7 0 ) ^ { 2 } \right) ,
$$

where $e _ { i j }$ is measured in kWh when $d _ { i j }$ is expressed in km. This model assumes a minimum consumption of 0.8 kWh/km at 70 km/h and a quadratic increase as the average speed moves away from this value.

We consider a fleet of battery-electric heavy-duty vehicles, each with a battery capacity of 850 kWh, a minimum admissible state of charge of $y _ { \mathrm { m i n } } = 5 0 \mathrm { k W h }$ , and a maximum charging power of 1,200 kW. These values are motivated by the specifications of the Tesla Semi, for which a range of 500 miles and an energy consumption of approximately 1.7 kWh/mile imply a battery capacity of approximately 850 kWh, while charging at up to 1.2 MW is supported (Tesla, 2026). For simplicity, we assume that this maximum charging power is available at every charging station.

The fleet is homogeneous and efectively unlimited. Customer demands and vehicle capacity are expressed in standardized units, and the capacity of each vehicle is set to 100. Consequently, the demand values introduced above should be interpreted relative to this standardized capacity rather than as physical units of cargo.

The operational-cost coeficients are set to $C _ { e } = 0 . 4 \ : \mathrm { { \in } / k W h , } C _ { \mathrm { t i m e } } = 8 \ : \mathrm { { \in } / h }$ , and $C _ { \mathrm { v e h } } = 2 0 0 \in$ per route. These are experimental baseline weights used to define the trade-ofs between the components of the objective and should not be interpreted as calibrated estimates of commercial operating costs. Unless stated otherwise, charging-related emissions are assigned a weight of $p _ { \mathrm { C O 2 } } = 1 0 \ : \mathrm { { \in / k g C O _ { 2 } e q } }$ . Because this value is substantially higher than conventional market carbon prices, $p _ { \mathrm { { C O 2 } } }$ should be interpreted as an experimental penalty coeficient controlling the trade-of between operational cost and charging-related emissions, rather than as a literal market price for carbon.

## 5.2. Data

Problem instances are assigned routing start dates between 2023 and 2025. To predict grid carbon intensity over each 72-hour routing horizon, we combine electricity grid observations with weather forecasts at hourly resolution. The electricity grid features comprise the life-cycle carbon intensity and total load of the German grid, while the weather features describe forecast solar irradiance and wind conditions.

The carbon intensity data were obtained from the Electricity Maps Data Portal (Electricity Maps, 2026), while load observations were obtained directly from the ENTSO-E Transparency Platform (European Network of Transmission System Operators for Electricity (ENTSO-E), n.d.). Electricity Maps derives consumption-based carbon intensities from electricity-generation and cross-border-flow data, whose underlying source for Germany is also ENTSO-E. For each routing instance, carbon-intensity and load observations from the 24 hours preceding the routing start are used as lagged predictive features, as described in the following subsection.

Weather forecasts were obtained from the $0 . 2 5 ^ { \circ } \times 0 . 2 5 ^ { \circ }$ Global Forecast System (GFS) grids produced by the National Centers for Environmental Prediction (National Centers for Environmental Prediction, National Weather Service, NOAA, U.S. Department of Commerce, 2015). Specifically, we extracted surface downward short-wave radiation and the horizontal wind components at 100 m from the pgrb2.0p25 product using the Herbie software package (Blaylock, 2026). The GFS forecast horizon covers the three days considered by the routing instances.

To construct Germany-level weather features, the forecast associated with each grid cell is weighted according to the renewable-generation capacity installed within that cell. Solar irradiance is weighted by installed solar capacity, while the two wind components are weighted by installed wind capacity. Installation locations and capacities were obtained from the German Market Master Data Register (MaStR) (Bundesnetzagentur, n.d.) and assigned to the corresponding $0 . 2 5 ^ { \circ } \times 0 . 2 5 ^ { \circ }$ grid cells. Because the MaStR capacity data are available as quarterly snapshots, each forecast date is associated with the temporally closest available snapshot.

The resulting dataset covers 2023–2025 at hourly resolution and contains the grid’s life-cycle carbon intensity and load, the capacity-weighted solar-irradiance forecasts, and the capacity-weighted forecasts of the two horizontal wind components at 100 m. The magnitude of the capacity-weighted wind vector is additionally derived from its two components and also included.

## 5.3. Carbon Intensity Model

As introduced in Section 4.1 when presenting the training loop, we use a linear model to predict the grid carbon intensity over the routing horizon. This choice keeps the predictive component transparent and enables a controlled comparison between PFL and DFL. Moreover, it permits the direct computation of the subgradient given in Equation (55). Investigating whether richer predictive models yield further downstream improvements is left for future work.

We forecast the carbon intensity of the grid at hourly resolution and accordingly use hourly time buckets in the routing objective. Let $H = \{ 0 , \ldots , 7 1 \}$ denote the hourly forecast periods and let $t _ { h } = t _ { s } + h$ , where $t _ { s }$ is the routing start time. The model produces 72 predictions covering the interval $[ t _ { s } , t _ { s } + 7 2 \mathrm { h } )$

For each time step $h \in H$ , we define

$$
\tau _ { h } = t _ { s } - 2 4 \mathrm { h } + ( h \bmod 2 4 ) \mathrm { h } .
$$

Thus, $\tau _ { h }$ is the corresponding hour from the day before the routing start time. The feature vector for period ℎ contains:

1. the observed grid carbon intensity at time $\tau _ { h }$

2. the observed grid load at time $\tau _ { h }$

3. the capacity-weighted solar irradiance forecast at time $t _ { h }$

4. the capacity-weighted forecasts for both wind components and wind-speed magnitude at 100 m at time $t _ { h }$

5. cyclical encodings of the hour of the day, day of the week, and day of the year associated with $t _ { h }$

For a temporal variable � with period �, its cyclical encoding is defined as

$$
\operatorname { c y c } ( q ; P ) = \left( \sin \left( { \frac { 2 \pi q } { P } } \right) , \cos \left( { \frac { 2 \pi q } { P } } \right) \right) .
$$

As is standard in time-series prediction, we use periods 24, 7, and 365 for the hour of the day, day of the week, and day of the year, respectively. This results in six features encoding time.

Before fitting the model, the six non-cyclical features are standardized using means and standard deviations computed from the training period. The cyclical features are left unchanged.

Let $z _ { h } \in \mathbb { R } ^ { 1 2 }$ denote the resulting feature vector. Including an intercept gives $x _ { h } = ( z _ { h } ^ { \top } , 1 ) ^ { \top } \in \mathbb { R } ^ { 1 3 }$ . Stacking the feature vectors for all $d = 7 2$ periods produces a matrix $X \in \mathbb { R } ^ { d \times 1 3 }$ . The predicted carbon-intensity vector is then

$$
\hat { c } = X \theta , \qquad \theta \in  { \mathbb { R } } ^ { 1 3 } , \qquad \hat { c } \in  { \mathbb { R } } ^ { d } .
$$

The total rolling-horizon dataset contains 72 hourly observations for each available forecast origin, with one forecast origin per day at 00:00 UTC. Before removing missing values, the period 2023–2025 contains $( 3 6 5 + 3 6 6 + 3 6 5 ) \cdot 7 2 =$ 78,912 forecast-origin and lead-time combinations. After removing missing observations, 78,765 remain: 52,560 from 2023–2024, which are used for fitting the PFL model, and 26,205 from 2025, which are reserved for predictive evaluation. The DFL training and validation instances are also drawn exclusively from 2023–2024, while downstream evaluation uses problem instances from 2025.

When the PFL or DFL-trained linear model produces a negative carbon-intensity prediction, that prediction is clipped to zero before it is supplied to the LNS. The same clipping is applied when computing the predictiveperformance metrics reported below.

## 5.4. Results

The goal of the experimental evaluation is twofold. First, we compare the proposed DFL pipeline against standard PFL. In the latter, we train the linear regression in a supervised learning manner using Ordinary Least Squares (OLS), with the real carbon intensity values as training labels. Second, we assess whether incorporating the carbon intensity term in the objective function leads to better solutions, as indicated by lower emissions.

## 5.4.1. Training and Model Selection

The DFL weights are initialized using the coeficients of the PFL linear model fitted by OLS on the 2023–2024 hourly observations. The decision-focused training then iteratively updates these parameters using the normalized approximate SPO+ directions described in Section 4.1.

DFL training uses 96 problem instances for training and 24 for validation. For the training set, 48 instances are sampled uniformly at random from each year, while the validation set contains 12 randomly sampled instances per year. Routing start dates are sampled uniformly over the valid days of the corresponding year, subject to the availability of a complete 72-hour forecast horizon. Each instance contains between 50 and 100 customers, with the number of customers sampled uniformly at random. The train-validation split is hence not stratified by routing-start month or number of customers.

The batch size is set to 2, while the learning rate is set to $\eta = 2 5$ . However, this value is not the efective per-instance step size because every instance subgradient is divided by $d \alpha _ { i } ,$ where $d = 7 2$ and $\alpha _ { i }$ is the total energy charged by its reference solution. Across the training instances, the resulting factor $\eta / ( d \alpha _ { i } )$ ranged from $1 . 3 6 \times 1 0 ^ { - 5 } \mathrm { t o } 4 . 7 6 \times 1 0 ^ { - 5 }$ with a median of $2 . 2 5 \times 1 0 ^ { - 5 }$

Regarding the LNS executions during training, we use a time limit of 40 seconds for training executions and 60 seconds for validation ones, which aims to match the 60-second time limit used during evaluation. Moreover, we set the number of customers removed by the Shaw destroy operator, �, to 10, the determinism parameter � to 4, and all relatedness weights $\phi _ { 1 } , \phi _ { 2 } , \phi _ { 3 }$ to one, giving the same weight to each standardized relatedness measure. Finally, we set its �-repair operator’s � parameter to 2, and the initial temperature $T _ { \mathrm { i n i t } }$ and cooling factor $c _ { f }$ of its simulated annealing to 100 and 0.97, respectively.

To produce the reference solutions obtained under the true carbon intensity used in the training loop, we execute the LNS once with a time limit of three hours using the same parameters as above, except for the simulated-annealing parameters, which are adjusted to the long time limit, by setting $T _ { \mathrm { i n i t } } = 1 0 0 0$ and $c _ { f } = 0 . 9 9 9 9 5$

Training was run for 100 epochs and required approximately 64 hours in total, with each epoch taking about 37 minutes on average. Overfitting was evaluated on the validation set after every epoch, and the five-epoch trailing average regret was used to decide which weights to retain. This was done with the goal of smoothing the epoch-byepoch variability produced by the stochasticity of the LNS. Following this, the weights from epoch 72 were the ones retained as they led to the lowest five-epoch trailing average validation regret.

Epoch value 5-epoch trailing mean PFL initialization (23.55%) Selected epoch (72)  
![](images/a183f083de91df9ee17fd74de0a4b02467ce3cbae6ee114f23901401999b4f7f.jpg)

![](images/8fcfb930ff53c2aaf27d504be4ce92bcbd9af035e2e589004a239ad8eee13628.jpg)  
Figure 4: Training dynamics over epochs. The figure on the left presents the evolution of the approximate SPO+ loss, while the figure on the right presents the evolution of the validation regret percentage.

The clipping applied in line 6 of Algorithm 1, aimed at avoiding negative carbon intensity predictions being fed into the LNS, was infrequent during the selected training run. Across all constructed transformed carbon-intensity vectors, merely 0.0047% of their components were negative and therefore clipped to zero. At least one component was clipped in 0.084% of the SPO+ LNS solves. The diference between these percentages indicates that, when clipping occurred, it generally afected only a small part of the 72-period transformed vector. Thus, the projection rarely modified the input supplied to the downstream solver, although it still introduced an additional approximation whenever active.

Figure 4 shows the approximate SPO+ training loss and validation regret over the training epochs. The reported training loss would coincide with the theoretical SPO+ loss if both solutions involved were obtained from exact optimization oracles and the transformed cost vector was not clipped. To avoid additional LNS executions, each epoch value averages losses evaluated at the parameter values current during its successive mini-batches, rather than at a single parameter vector at the end of the epoch. The reported epoch average should therefore be interpreted as an online diagnostic rather than as the empirical loss at a single parameter vector. Validation regret is the average percentage excess cost under the true carbon intensity of the validation solutions relative to the precomputed reference solutions.

The initial validation regret, computed with the warm-started weights from PFL, is represented by a gray horizontal dashed line. As can be observed, most learning happens within the first 20 epochs, as it is when validation regret decreases the most. We also plot the 5-epoch trailing mean of both measures, which in the case of the validation value was used for selecting the trained weights.

All training and evaluation runs were executed on Intel Xeon Gold 6338 ("Ice Lake") processors with 32 cores per CPU (64 cores total per node) running at 2.00 GHz. In particular, the two LNS executions required for processing each batch were done in parallel, while parallelization with 8 concurrent solves was used for the 24 LNS validation executions done at the end of every epoch. For solving the postprocessing model, presented in Section 4.3 and applied at the end of every LNS execution, version 13.0 of the commercial solver Gurobi (Gurobi Optimization, LLC, 2026) was employed, with 8 threads reserved for every solve.

## 5.4.2. Predictive Performance

Before evaluating downstream decision quality, we compare the pointwise predictive performance of the PFL and DFL-trained models. As a baseline, we also consider a naive 24-hour persistence forecast that repeats the carbon intensity observed during the 24 hours preceding the routing start over the three-day routing horizon. The evaluation uses all available forecast-origin and lead-time combinations from 2025, resulting in 26,205 observations.

Table 1 reports the mean absolute error (MAE) and root mean squared error (RMSE) of the three approaches. Both metrics are expressed in grams of carbon-dioxide equivalent per kilowatt-hour $( \mathrm { g C O } _ { 2 } \mathrm { e q } / \mathrm { k W h } )$

The PFL model achieves the best pointwise predictive performance, reducing MAE by 36.5% and RMSE by 39.6% relative to the naive persistence baseline. This indicates that the linear model extracts useful predictive information from the lagged grid variables, weather forecasts, and cyclical time features beyond merely repeating the carbon-intensity profile of the preceding day.

Predictive performance on the 2025 evaluation data. Lower values indicate better pointwise predictive accuracy.
<table><tr><td>Model</td><td>MAE</td><td>RMSE</td></tr><tr><td>PFL</td><td>57.70</td><td>72.46</td></tr><tr><td>DFL</td><td>402.70</td><td>410.71</td></tr><tr><td>Naive 24-hour persistence</td><td>90.88</td><td>120.03</td></tr></table>

Forecast horizon beginning October 1st 2025  
![](images/0ef89e746e84557906553cc7b00413db31b46bfbccb6571f84d5b07d4c40f98d.jpg)  
Figure 5: Observed carbon intensity and predictions produced by the PFL, DFL, and naive 24-hour persistence models over the 72-hour routing horizon beginning on 1 October 2025.

The DFL model produces substantially larger pointwise errors. This is not unexpected, since its parameters are optimized using the SPO+ loss rather than a conventional prediction loss. Consequently, DFL does not directly penalize deviations between predicted and observed carbon intensities. Its outputs are therefore better interpreted as decisionoriented cost coeficients supplied to the routing algorithm than as calibrated carbon-intensity forecasts. Whether sacrificing predictive accuracy is beneficial must consequently be assessed through the quality of the resulting routing decisions, as considered in the following subsection.

Figure 5 illustrates the predictions for the 72-hour routing horizon beginning on October 1, 2025. In this example, the PFL prediction follows the observed carbon intensity more closely than the naive baseline. The DFL prediction exhibits temporal variation but is shifted substantially upward, consistent with its large aggregate MAE and RMSE.

## 5.4.3. Downstream Decision Quality

We evaluate downstream decision quality using 200 problem instances with routing start dates sampled uniformly over 2025. Each LNS execution is assigned a 60-second time limit and uses the baseline parameter configuration employed during training. The only parameter modified is the simulated-annealing cooling factor, which is increased from 0.97 to 0.98 because the evaluation time limit is longer than the 40-second limit used for the LNS executions during training.

We compare the DFL-trained predictor with the PFL-trained predictor, the naive 24-hour persistence forecast introduced above, and a perfect-information heuristic benchmark that supplies the true carbon intensity to the LNS. To account for the stochasticity of the LNS, each method is executed ten times on every instance, resulting in 2,000 executions per method. Regardless of the carbon-intensity forecast supplied during optimization, every resulting solution is subsequently evaluated using the true carbon intensity. Consequently, the comparison measures realized downstream cost under a common set of observed carbon-intensity values.

The 60-second limit applies only to the LNS. The exact fixed-route postprocessing is applied once to the final solution after this limit and requires an additional 0.287 seconds per solution on average. The corresponding median and 95th percentile are 0.247 and 0.58 seconds, respectively.

![](images/4848bb6d2e9b1f7171d809ec1bb9d614427f173a98ff92ab3adc4115bc8cb2f4.jpg)  
Problem instances ordered by improvement (%)  
Figure 6: Sorted downstream improvement of the DFL-trained predictor over the PFL-trained predictor, the naive 24-hour persistence forecast, and the true-carbon-intensity benchmark.

For each method and problem instance, we first average the realized total cost over its ten executions. For baseline �, the instance-level improvement of DFL is

$$
I _ { b } ( i ) = 1 0 0 \frac { \bar { C } _ { b } ( i ) - \bar { C } _ { \mathrm { D F L } } ( i ) } { \bar { C } _ { b } ( i ) } .
$$

Here, $\bar { C } _ { b } ( i )$ and $\bar { C } _ { \mathrm { D F L } } ( i )$ are the mean costs for instance � using baseline � and using the DFL-trained predictor, respectively. Hence, positive values indicate that the DFL-trained predictor achieves a lower mean realized cost than baseline �. Statistical summaries therefore treat the 200 instance-level mean improvements, rather than the 2,000 individual executions per method, as the observations.

Figure 6 presents the sorted instance-level improvement of DFL relative to each benchmark. The instances are ordered independently for each comparison; consequently, the same horizontal position does not generally correspond to the same problem instance across the three curves. Positive values indicate that DFL achieves a lower mean realized cost than the corresponding benchmark. The vertical dashed lines mark the fraction of instances for which this improvement is positive.

Relative to PFL, DFL reduces realized cost by 2.30% on average, with a median improvement of 2.56%, and achieves a lower average cost on 75.5% of the instances. Its advantage is larger relative to the naive persistence forecast, with a mean improvement of 6.16% and a win rate of 87.5%. By contrast, DFL is 0.59% more costly on average than the perfect-information heuristic benchmark and outperforms it on 45% of the instances. This benchmark has access to the true carbon intensity but still relies on the same stochastic, time-limited LNS and is therefore not an optimal lower bound. Together with Table 1, these results show that the lower pointwise accuracy of DFL nevertheless translates into better downstream decisions than the other forecast-based approaches.

Figure 7 examines how the relative performance of DFL and PFL varies with the routing-start month. Figure 7(a) shows the distribution of instance-level improvements within each month. Each point represents the ten-run mean improvement for one problem instance; center lines indicate medians, diamonds indicate means, boxes show interquartile ranges, and whiskers extend from the 5th to the 95th percentile. Figure 7(b) reports the corresponding failure rate, defined as the proportion of instances for which DFL produces a higher mean realized cost than PFL. The error bars indicate 95% percentile bootstrap confidence intervals, computed from 20,000 resamples with replacement within each month.

Performance varies with the routing month. Mean DFL improvement remains positive from January through October but becomes approximately zero in November and negative in December. Across November–December, the mean improvement is −0.69%, compared with 2.93% over January–October, while the corresponding failure rates are 47.5% and 19.6%. Because this seasonal analysis was conducted post hoc and the number of test instances difers across months, it should be interpreted as exploratory evidence rather than as a statistical conclusion.

![](images/5fc92cd4ed308bde76413285ceffcdd26aae71321cd60833878ddc4a3074063b.jpg)

![](images/042dbb5e91350f2a6169c41bf329e2028603d885016b1ad5886290158a600b9f.jpg)  
Figure 7: Downstream solution improvement of the DFL-trained linear regression over the PFL-trained one, with the results organized by the month of the instances’ start date.

## 5.4.4. Small-scale Exact Validation

To assess the solution quality of the LNS against proven-optimal solutions, we generated a deliberately small benchmark consisting of ten instances with each of three, four, and five customers. All locations are in the region of Saxony, and each instance has a 24-hour routing horizon. Depot and customer locations were sampled from the same preprocessed OpenStreetMap locations used for the main instances. For each instance, we retained charging stations within 0.15 km of at least one OSRM route connecting a pair of service locations, comprising the depot and customers, and subsequently removed stations without a feasible incoming connection. This resulted in averages of 5.6, 7.5, and 9.3 charging stations for the three, four, and five customer instances, respectively.

The standardized vehicle load capacity remained $Q _ { \mathrm { m a x } } = 1 0 0$ . To preserve the relevance of the capacity constraint despite the small number of customers, aggregate demand was sampled uniformly between 140 and 180 and distributed evenly across customers. Regarding time windows, each customer was given a $2 / 3$ probability of being assigned a flexible time window covering the whole 24 hours, and $1 / 3$ probability ofbeing assigned a working-hours time window, with the opening and closing times sampled uniformly at random from hours 5–9 and 17–21, respectively. The battery capacity was reduced from 850 to 300 kWh so that intermediate charging remained relevant. Both methods were supplied with the true carbon-intensity trajectory. Gurobi was given a ten-hour time limit and used two copies of each physical charging station, while the LNS was executed ten times per instance with a 60-second limit. Consequently, the optimality statements below refer to the MILP containing two copies of each physical station.

## Table 2

Small-scale comparison of the LNS with the exact MILP.
<table><tr><td>Customers</td><td>Mean stations</td><td>Optimality Proven</td><td>Mean gap (%)</td><td>Median gap (%)</td></tr><tr><td>3</td><td>5.6</td><td>10/10</td><td>1.46</td><td>0.77</td></tr><tr><td>4</td><td>7.5</td><td>5/10</td><td>2.48</td><td>1.43</td></tr><tr><td>5</td><td>9.3</td><td>0/10</td><td>一</td><td>一</td></tr></table>

Table 2 presents the resulting comparison, with each instance-level gap calculated using the mean objective over the ten LNS executions. Gurobi proved optimality for all three-customer instances, against which the LNS obtained a mean gap of 1.46% and a median gap of 0.77%. For the four-customer instances, it proved optimality for five; over this solved subset, the corresponding gaps were 2.48% and 1.43%. Finally, it could not prove optimality for any fivecustomer instance within the ten-hour limit. Although these instances are necessarily small, the results provide a direct validation of LNS solution quality where exact comparison is possible and illustrate the rapid computational growth of the MILP.

![](images/d6e76d44fe0d1cedaa5ce902d57a9ebc40fcc94c2a4abc233f321f7dceffc9c3.jpg)  
Figure 8: Downstream improvement of the DFL-trained predictor over the PFL-trained predictor under diferent LNS parameter configurations.

## 5.4.5. Robustness to the LNS Configuration

The main evaluation uses the same LNS configuration as DFL training, except for the cooling factor, which is adjusted to account for the longer execution time. Furthermore, its 60-second time limit is identical to that used when evaluating validation regret during training. The learned predictor could therefore be specialized to the solver configuration used throughout training and model selection. To assess whether its advantage generalizes beyond this configuration, we reevaluate the fixed DFL and PFL predictors under alternative LNS settings. Notably, the predictors are not retrained in any of these experiments.

Figure 8 presents the instance-level improvement of DFL over PFL under diferent evaluation configurations. We vary one of the LNS main parameters at a time while holding the remaining parameters at their baseline values. Specifically, we consider execution time limits of 20, 60, and 300 seconds; destroy sizes $q \in \{ 5 , 1 0 , 2 0 \}$ ; and regret parameters $k \in \{ 2 , 3 , 4 \}$ . The baseline configuration uses a 60-second time limit, $q = 1 0 .$ , and $k = 2$ . When changing the time limit, the cooling factor is adjusted so that the simulated-annealing temperature follows a comparable cooling schedule over the duration of the search. Particularly, it is set to 0.941192 and 0.995968, for the 20 and 300 second time limits, respectively. Each point represents the ten-run mean improvement for one problem instance; center lines indicate medians, diamonds indicate means, boxes show interquartile ranges, and whiskers extend from the 5th to the 95th percentile.

The DFL advantage remains positive across all tested LNS configurations. Mean improvement over PFL ranges from 2.03% to 2.91% when varying the time limit, destroy size, or regret parameter. Particularly, increasing the time limit seems to reduce variability while the mean and median improvements remain positive. Increasing the number of customers removed seems to decrease the mean and median improvement. Finally, increasing the regret parameter does not result in noticeable changes in performance. These results therefore suggest that the learned predictor is not narrowly tied to the precise solver configuration used during training.

![](images/cd9d14a63f591ac193cd094f0a470d2daf8345571256b1bf6dffebb206d3e7cb.jpg)  
Figure 9: Downstream improvement of the DFL-trained predictor over the PFL-trained predictor for carbon-cost weights of 5, 10, and 20.

## 5.4.6. Sensitivity to Carbon-cost Weight

We also assess sensitivity to the carbon-cost weight $p _ { \mathrm { { C O 2 } } }$ used to value charging-related emissions. The training and evaluation use a carbon-cost weight of $1 0 \in / \mathrm { k g C O } _ { 2 } \mathrm { e q }$ , while the sensitivity analysis additionally considers carbon-cost weights of 5 and 20. As in the previous section, we evaluate these diferent configurations using the same trained predictor as in the evaluation, without retraining. Figure 9 shows a monotonic increase in the mean and median improvement of DFL across the tested carbon-cost weights. Its mean improvement over PFL is 1.52%, 2.30%, and 3.40% for carbon-cost weights of 5, 10, and 20, respectively, while the corresponding fractions of instances on which DFL performs better are 66%, 75.5%, and 74.5%. As before, each point represents the ten-run mean improvement for one problem instance; center lines indicate medians, diamonds indicate means, boxes show interquartile ranges, and whiskers extend from the 5th to the 95th percentile.

## 5.4.7. Efects of Considering the Carbon Cost

Finally, we examine whether explicitly accounting for charging-related emissions changes the solutions produced by the LNS and postprocessing. We compare a carbon-aware configuration, which supplies the LNS with the DFL predictor’s carbon-intensity forecasts and uses the baseline carbon-cost weight, with a carbon-blind configuration obtained by setting this weight to zero during optimization. All other LNS parameters and execution limits are identical to those used in the main evaluation. Similarly, both approaches are executed ten times on each of the 200 evaluation instances, and their results are first averaged within each instance.

Although the carbon-blind solutions ignore carbon intensity during optimization, they are evaluated afterward using the true carbon intensity and the baseline carbon-cost weight $p _ { \mathrm { C O } _ { \gamma } } = 1 0 \ : \mathrm { { \in / k g C O } _ { 2 } \mathrm { { e q } } }$ . This allows us to compare the charging-related emissions, non-carbon operating cost, and realized total cost of both approaches under a common objective.

Figure 10 shows that explicitly considering charging-related emissions produces a substantial reduction in realized emissions. Relative to carbon-blind optimization, the carbon-aware solutions reduce emissions by 28.76% on average across instances. These solutions produce lower emissions on 99.5% of the evaluation instances. In absolute terms, average emissions decrease from 5,525.96 to 4,026.56 kg of ${ \mathrm { C O } } _ { 2 } { \mathrm { e q } }$ per instance.

As expected, this emissions reduction is accompanied by an increase in non-carbon operating cost. Specifically, mean non-carbon operating cost increases by 11.06%, and it increases on every evaluation instance, illustrating the trade-of introduced by the carbon objective. Nevertheless, when both cost components are evaluated using the baseline carbon-cost weight, the carbon-aware solutions reduce realized total cost by 20.24% on average, and a lower realized total cost is achieved on 99% of the instances.

The carbon-aware solutions also require 2.05% less charging energy on average. Because this reduction is considerably smaller than the 28.76% reduction in emissions, the improvement cannot be explained by energy consumption alone. It also reflects changes in routing and charging timing that shift electricity consumption toward periods with lower realized carbon intensity. Overall, these results show that including charging-related emissions materially afects the resulting decisions and achieves substantial emission reductions, although at the cost of higher non-carbon expenditure.

![](images/77b44182738efa67b604376502dc00a14499b64d5a0f096d76003f8a4cb31c6f.jpg)  
Figure 10: Efects of including charging-related emissions in the optimization objective.

The efect on fleet size is comparatively small. Under the baseline objective coeficients, the carbon-aware solutions use an average of 13.787 vehicles per instance, compared with 13.677 for the carbon-blind solutions, corresponding to an increase of 0.110 vehicles, or 0.8%. These values should be interpreted conditionally on the experimental objective coeficients, since the relative weights assigned to vehicles, operating costs, and charging-related emissions determine the trade-of between fleet size and emission reduction. Nevertheless, the small increase in fleet size relative to the 28.76% reduction in charging-related emissions indicates that the emission savings cannot be attributed primarily to the use of additional vehicles.

## 6. Conclusions and Future Work

This paper presented an EVRP formulation that incorporates charging-related emissions into its objective. The proposed formulation jointly optimizes routes, charging, and waiting decisions while accounting for temporal variation in grid carbon intensity. This is motivated by the substantial temporal variation in the electricity generation mix of some regions, since charging during periods of high grid carbon intensity can substantially increase charging-related emissions and, in suficiently carbon-intensive grids, can even erode the lifecycle emissions advantage of electric vehicles over eficient hybrid alternatives.

We presented a decision-focused learning approach for tackling this problem. A linear predictor was trained using approximate SPO+ subgradients to predict the carbon intensity of the grid over the routing horizon, while a time-limited LNS with exact postprocessing was used as the downstream solver.

The experimental results demonstrate the usefulness of this carbon-aware formulation, as the solutions produced reduced emissions by 28.76% on average relative to carbon-blind optimization, while increasing the non-carbon cost by 11.06%. Under the baseline objective coeficients, this trade-of resulted in a 20.24% reduction in realized total cost.

Moreover, the results also illustrate the distinction between predictive accuracy and downstream decision quality. Although the PFL-trained model achieved substantially lower forecasting MAE and RMSE, the DFL-trained predictor produced better solutions on 75.5% of the test instances relative to PFL. Its mean advantage also remained positive across the evaluated LNS configurations and carbon-cost weights, indicating robustness to the solver and carbon cost configurations. The exploratory monthly analysis nevertheless revealed weaker performance toward the end of the year, indicating that the benefits of decision-focused training were heterogeneous across the evaluation period.

The principal limitation is computational cost: the selected epoch (72) was reached after approximately 44 hours, while the complete 100-epoch training run required approximately 64 hours. To alleviate this issue, future work could consider further reducing the time limit assigned to the LNS during training and exploring the trade-of between this time limit and downstream performance. Another option would be to construct a surrogate model to reduce the number of LNS and postprocessing executions.

Future work could also investigate stratified or seasonally balanced training samples and examine whether the observed seasonal deterioration is caused by limited training coverage, changes in grid behavior, or characteristics of the routing instances. Also, richer predictive models could be considered. Finally, further extensions could consider routes spanning several electricity bidding regions, with region-specific carbon-intensity predictions.

## Acknowledgements

The computations were enabled by resources provided by Chalmers e-Commons at Chalmers. The computational experiments were performed on the Vera computing cluster under project C3SE2026-1-13.

## CRediT authorship contribution statement

Jaume Reixach: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data Curation, Writing – original draft, Writing – review & editing. Wanru Chen: Conceptualization, Methodology, Investigation, Writing – review & editing. Jiaming Wu: Conceptualization, Methodology, Supervision, Writing – review & editing. Balázs Kulcsár: Conceptualization, Methodology, Supervision, Writing – review & editing.

## References

Amos, B., Kolter, J.Z., 2017. OptNet: Diferentiable Optimization as a Layer in Neural Networks, in: Proceedings of the 34th International Conference on Machine Learning, PMLR. pp. 136–145. URL: https://proceedings.mlr.press/v70/amos17a.html.

Basso, R., Kulcsár, B., Egardt, B., Lindroth, P., Sanchez-Diaz, I., 2019. Energy consumption estimation integrated into the Electric Vehicle Routing Problem. Transportation Research Part D: Transport and Environment 69, 141–167. doi:10.1016/j.trd.2019.01.006.

Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J.P., Bach, F., 2020. Learning with Diferentiable Pertubed Optimizers, in: Ad vances in Neural Information Processing Systems, pp. 9508–9519. URL: https://proceedings.neurips.cc/paper/2020/hash/ 6bb56208f672af0dd65451f869fedfd9-Abstract.html.

Blaylock, B.K., 2026. Herbie: Retrieve Numerical Weather Prediction Model Data (Version 2026.3.0). doi:10.5281/zenodo.18902673.

Bundesnetzagentur, n.d. Marktstammdatenregister (MaStR): Gesamtdatenexport. Bundesnetzagentur. URL: https://www. marktstammdatenregister.de/MaStR/Datendownload. Quarterly full-data XML snapshots.

Chu, H., Zhang, W., Bai, P., Chen, Y., 2023. Data-driven optimization for last-mile delivery. Complex & Intelligent Systems 9, 2271–2284. doi:10.1007/s40747-021-00293-1.

Domke, J., 2010. Implicit Diferentiation by Perturbation, in: Advances in Neural Information Processing Systems, pp. 523–531. URL: https://proceedings.neurips.cc/paper/2010/hash/6ecbdd6ec859d284dc13885a37ce8d81-Abstract.html.

Domke, J., 2012. Generic Methods for Optimization-Based Modeling, in: Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, PMLR. pp. 318–326. URL: https://proceedings.mlr.press/v22/domke12.html.

Electricity Maps, 2026. Historical carbon intensity data for Germany. Electricity Maps Data Portal. URL: https://portal.electricitymaps. com/datasets/DE. 15-minute life-cycle carbon-intensity data for 2023–2025; export version dated 2026-02-10.

Elmachtoub, A.N., Grigas, P., 2022. Smart “Predict, then Optimize”. Management Science 68, 9–26. doi:10.1287/mnsc.2020.3922.

Ester, M., Kriegel, H.P., Sander, J., Xu, X., 1996. A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise, in: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, AAAI Press. pp. 226–231. URL: https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf.

European Commission, 2026. Communication from the Commission to the European Parliament, the Council, the European Economic and Social Committee and the Committee of the Regions: Electrification Action Plan. COM (2026) 595 final. European Commission, Directorate-Genera for Energy. URL: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52026DC0595.

European Network of Transmission System Operators for Electricity (ENTSO-E), n.d. ENTSO-E Transparency Platform. ENTSO-E. URL: https://transparency.entsoe.eu/. Accessed: 2026-02-10.

Eurostat, 2026. Final energy consumption in transport - detailed statistics. Eurostat Statistics Explained. URL: https://ec.europa.eu/ eurostat/statistics-explained/index.php?title=Final\_energy\_consumption\_in\_transport\_-\_detailed\_statistics. Accessed: 2026-09-21.

Ferber, A., Wilder, B., Dilkina, B., Tambe, M., 2020. MIPaaL: Mixed Integer Program as a Layer, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 1504–1511. doi:10.1609/aaai.v34i02.5509.

Gurobi Optimization, LLC, 2026. Gurobi Optimizer Reference Manual. URL: https://docs.gurobi.com/projects/optimizer/en/ current/. Version 13.0.

Hart, P.E., Nilsson, N.J., Raphael, B., 1968. A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics 4, 100–107. doi:10.1109/TSSC.1968.300136.

Jarvis, P., Climent, L., Arbelaez, A., 2022. Smart and Sustainable Scheduling of Charging Events for Electric Buses, in: EU Cohesion Policy Implementation: Evaluation Challenges and Opportunities. Springer Nature Switzerland. Springer Proceedings in Political Science and International Relations, pp. 121–129. doi:10.1007/978-3-031-18161-0\_8.

Kirkpatrick, S., Gelatt, Jr., C.D., Vecchi, M.P., 1983. Optimization by Simulated Annealing. Science 220, 671–680. doi:10.1126/science.220. 4598.671.

Kotary, J., Dinh, M.H., Fioretto, F., 2023. Backpropagation of Unrolled Solvers with Folded Optimization, in: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pp. 1963–1970. doi:10.24963/ijcai.2023/218.

Kucukoglu, I., Dewil, R., Cattrysse, D., 2021. The electric vehicle routing problem and its variations: A literature review. Computers & Industria Engineering 161, 107650. doi:10.1016/j.cie.2021.107650.

Luxen, D., Vetter, C., 2011. Real-time routing with OpenStreetMap data, in: Proceedings of the 19th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM. pp. 513–516. doi:10.1145/2093973.2094062.

Mandi, J., Bucarey, V., Tchomba, M.M.K., Guns, T., 2022. Decision-Focused Learning: Through the Lens of Learning to Rank, in: Proceedings of the 39th International Conference on Machine Learning, PMLR. pp. 14935–14947. URL: https://proceedings.mlr.press/v162/ mandi22a.html.

Mandi, J., Canoy, R., Bucarey, V., Guns, T., 2021. Data Driven VRP: A Neural Network Model to Learn Hidden Preferences for VRP, in: 27th International Conference on Principles and Practice of Constraint Programming (CP 2021), Schloss Dagstuhl–Leibniz-Zentrum für Informatik. pp. 42:1–42:17. doi:10.4230/LIPIcs.CP.2021.42.

Mandi, J., Kotary, J., Berden, S., Mulamba, M., Bucarey, V., Guns, T., Fioretto, F., 2024. Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities. Journal of Artificial Intelligence Research 80, 1623–1701. doi:10.1613/jair.1.15320.

Miyabe, R., Fujimoto, Y., Hayashi, Y., 2025. Low-carbon routing and charging planning for electric freight trucks utilizing local surplus solar power. Journal of Energy Storage 132, 117626. doi:10.1016/j.est.2025.117626.

Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., Guns, T., 2021. Contrastive Losses and Solution Caching for Predict-and-Optimize, in: Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, International Joint Conferences on Artificial Intelligence Organization. pp. 2833–2840. doi:10.24963/ijcai.2021/390.

National Centers for Environmental Prediction, National Weather Service, NOAA, U.S. Department of Commerce, 2015. NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive. doi:10.5065/D65D8PWK.

Niepert, M., Minervini, P., Franceschi, L., 2021. Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions, in: Advances in Neural Information Processing Systems, pp. 14567–14579. URL: https://proceedings.neurips.cc/paper/2021/hash/ 7a430339c10c642c4b2251756fd1b484-Abstract.html.

OpenStreetMap contributors, 2026. OpenStreetMap. OpenStreetMap. URL: https://www.openstreetmap.org/. Accessed: 2026-09-21.

Pisinger, D., Røpke, S., 2018. Large Neighborhood Search, in: Handbook of Metaheuristics. Springer International Publishing. volume 272 of International Series in Operations Research & Management Science, pp. 99–127. doi:10.1007/978-3-319-91086-4\_4.

Shah, S., Wang, K., Wilder, B., Perrault, A., Tambe, M., 2022. Decision-Focused Learning without Decision-Making: Learning Locally Optimized Decision Losses, in: Advances in Neural Information Processing Systems, Curran Associates, Inc. pp. 1320–1332. doi:10.52202/ 068431-0097.

Singh, M., Yuksel, T., Michalek, J.J., Azevedo, I.M.L., 2024. Ensuring greenhouse gas reductions from electric vehicles compared to hybrid gasoline vehicles requires a cleaner U.S. electricity grid. Scientific Reports 14, 1639. doi:10.1038/s41598-024-51697-1.

Su, J., Lin, Q., Chen, M., 2025. Optimizing carbon footprint in long-haul heavy-duty E-Truck transportation. Nature Communications 16, 9562. doi:10.1038/s41467-025-64792-2.

Tesla, 2026. Tesla Semi. Tesla. URL: https://www.tesla.com/semi. Accessed: 2026-08-03.

Tian, X., Yan, R., Wang, S., Laporte, G., 2023. Prescriptive analytics for a maritime routing problem. Ocean & Coastal Management 242, 106695. doi:10.1016/j.ocecoaman.2023.106695.

Wilder, B., Dilkina, B., Tambe, M., 2019. Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 1658–1665. doi:10.1609/aaai.v33i01.33011658.