# Weighted-AoI-Driven UAV Trajectory and Reporting for Dynamic CKM Tracking

Zeyu Zhao<sup>1</sup>, Yueling Che<sup>2</sup> and Sheng Luo<sup>3</sup>

College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, China 2100271103@email.szu.edu.cn<sup>1</sup>, yuelingche@szu.edu.cn<sup>2</sup>, sluo@szu.edu.cn<sup>3</sup>

Abstract—Existing channel knowledge map (CKM) construction assumes quasi-static environments, failing when channels rapidly decorrelate. This paper investigates continuous online dynamic CKM tracking over an airborne 2D plane via a UAV fleet. We formulate a long-term error minimization problem under UAV mobility and reporting constraints. This problem, however, is intractable with unobservable errors for unmeasured cells. To tackle this, we propose a metric-driven scheduling approach featuring the Weighted-Age of Information (W-AoI). W-AoI evaluates cells by combining exact temporal staleness with spatial uncertainty predicted by a 2D U-Net. We prove that top-K W-AoI selection maximizes the one-slot error reduction bound. Guided by this guarantee, we develop a lightweight greedy trajectory policy incorporating continuous along-path sensing. Simulations demonstrate our method reduces steady-state error by $\mathbf { 3 . 8 \ d B ^ { 2 } }$ and ${ \bf 4 . 8 \ d B ^ { 2 } }$ over AoI-only and FIFO baselines, closely approaching an idealized oracle.

Index Terms—Dynamic channel knowledge maps (CKMs), age of information (AoI), metric-driven scheduling, UAV trajectory and reporting, 2-D U-Net.

## I. INTRODUCTION

As unmanned aerial vehicles (UAVs) increasingly integrate into cellular networks [1], channel knowledge maps (CKMs) have become a vital tool for environment-aware communications [2]. By providing location-specific channel priors such as expected signal-to-interference-plus-noise ratio (SINR), a CKM enables proactive trajectory planning and resource allocation rather than reactive outage handling [3]. Since ground drive tests cannot reach the airspace and ray tracing demands exact 3D physical models, UAVs serve as natural mobile collectors. However, constrained by flight speed, inter-UAV safety distance, and uplink reporting budgets, a UAV fleet can only measure and report a small fraction of the target area per slot, leaving the server with sparse observations.

To recover a complete map from sparse samples, existing studies rely on spatial interpolation techniques ranging from Kriging [4] and deep autoencoders [5] to generative diffusion models [6]. From the data collection perspective, UAV trajectory policies have been designed for active spectrum surveying [7] and partial-CKM navigation [8], alongside theoretical analysis on the spatial sampling density required for target accuracy [9]. While effective against spatial sparsity, most existing approaches rely on a quasi-static assumption: they treat channel statistics as time-invariant during the survey and pool measurements from different slots into a single static map, overlooking short-term temporal variations.

In practical urban networks, dynamic interference, traffic fluctuations, and moving scatterers cause channels to decorrelate before UAVs can sweep the entire grid, necessitating continuous online updates for dynamic CKM maintenance. Although recent works have explored dynamic radio map tracking via statistical hypothesis testing [10] and adaptive channel state information (CSI) map path planning via deep reinforcement learning (DRL) [11], key challenges remain unresolved. First, fixed-sensor updating rules [10] do not account for UAV mobility and collision-avoidance constraints. Second, Age-of-Information (AoI)-based scheduling [12] tracks temporal staleness but treats all spatial cells equally, ignoring that regions with sharp channel variations degrade faster. Third, DRL trajectory planners [11] require heavy offline training without explicit theoretical bounds on estimation error.

In this paper, we investigate the maintenance of a dynamic CKM over an aerial plane using a UAV fleet, and formulate a long-term CKM estimation error minimization problem under practical UAV operational constraints. Since the true error of unmeasured cells is unobservable, we employ a two-phase 2D U-Net at the server to predict both the current CKM and an Absolute Error Map (AEM) from historical sparse measurements and AoI states. Combining the exact AoI with the predicted AEM, we propose the Weighted-AoI (W-AoI) metric and prove that it upper-bounds the true cell error, with Top-K W-AoI selection maximizing the one-slot bound reduction under idealized conditions. Guided by this guarantee, we develop a lightweight greedy trajectory policy where each UAV navigates within a Voronoi subregion toward the highest-W-AoI reachable cell while continuously measuring cells along its flight path. Finally, extensive simulations demonstrate that our method significantly outperforms baseline approaches and closely approaches an idealized oracle.

Notation: |X| is set size, ⊙ element-wise product, $\| \cdot \| _ { 2 }$ the Euclidean norm; bold letters are vectors or matrices.

## II. SYSTEM MODEL

## A. Network and Channel Model

Consider a low-altitude target area modeled as a 2D horizontal plane $\mathcal { D } = [ 0 , L ] ^ { 2 }$ at a fixed altitude $h _ { 0 } . \mathrm { A }$ set of M ground base stations $\mathbf { ( G B S s ) }$ , denoted by $\mathcal { M } = \{ 1 , \ldots , M \}$ with locations $\mathbf q _ { m } = [ x _ { m } , y _ { m } , z _ { m } ] ^ { T }$ , provides downlink coverage over the aerial plane. The region D is discretized into an $I \times J$ grid of square cells, where $I = J = L / \Delta _ { D }$ . The grid resolution $\Delta _ { D }$ is small such that the channel quality is assumed homogeneous within each cell. We denote $\textbf { \em q } \in \mathcal { D }$ as the center coordinate of a generic cell. For brevity, D also denotes the discrete set of $I J$ cell centers in what follows. To construct the CKM, a fleet of $U$ rotary-wing UAVs, denoted by $\mathcal { U } = \{ 1 , \dots , U \}$ , acts as mobile sensors navigating through $\mathcal { D }$ to measure the spatial-temporal channel states and then report to their serving GBSs. The GBSs are connected via fiber backhaul to a central server (e.g., GBS 1), which aggregates all UAV measurements to maintain the global CKM. The system operates in discrete time slots $\mathcal { T } = \{ 1 , \dots , T \}$ with duration $\Delta t$

Let $G _ { m } ( \pmb { q } , t )$ be the channel gain from GBS m to a typical aerial receiver located at cell q in slot t. The gain is composed of large-scale and small-scale fading components:

$$
G _ { m } ( \pmb q , t ) = \bar { G } _ { m } ( \pmb q ) | h _ { m } ( \pmb q , t ) | ^ { 2 } ,\tag{1}
$$

where $\bar { G } _ { m } ( { \pmb q } ) = 1 0 ^ { [ G _ { a } ( \theta _ { m } ( \pmb q ) ) - \mathrm { P L } _ { m } ( \pmb q ) - S _ { m } ( \pmb q ) ] / 1 0 }$ captures the quasi-static large-scale channel gain, including the vertical antenna array gain $G _ { a } ,$ the 3GPP urban-micro path loss $\mathrm { P L } _ { m }$ [1], and the spatial correlated shadowing $S _ { m } ( \pmb q )$ . The term $h _ { m } ( \pmb q , t )$ represents the dynamic small-scale fading, modeled as a first-order Gauss-Markov process:

$$
h _ { m } ( \pmb { q } , t ) = \rho _ { h } h _ { m } ( \pmb { q } , t - 1 ) + \sqrt { 1 - \rho _ { h } ^ { 2 } } e _ { m } ( \pmb { q } , t ) ,\tag{2}
$$

with $e _ { m } \sim \mathcal { C N } ( 0 , 1 )$ and correlation coefficient $\rho _ { h } \in [ 0 , 1 ]$

All GBSs share one frequency band. Let $\alpha _ { m } ( t ) \in \{ 0 , 1 \}$ be one when GBS m transmits in slot t and zero otherwise; $\alpha _ { m } ( t )$ follows a two-state Markov chain with stationary probability $l _ { m } = \mathbb { E } [ \alpha _ { m } ( t ) ]$ and correlation $\rho _ { \alpha } \in [ 0 , 1 ]$

$$
\begin{array} { r l } & { \operatorname* { P r } \{ \alpha _ { m } ( t ) = 1 \mid \alpha _ { m } ( t - 1 ) = 0 \} = \left( 1 - \rho _ { \alpha } \right) l _ { m } , } \\ & { \operatorname* { P r } \{ \alpha _ { m } ( t ) = 1 \mid \alpha _ { m } ( t - 1 ) = 1 \} = \rho _ { \alpha } + \left( 1 - \rho _ { \alpha } \right) l _ { m } , } \end{array}\tag{3}
$$

The activity process reduces to independent Bernoulli activity at $\rho _ { \alpha } \ = \ 0 ;$ otherwise the interference a cell sees stays correlated over $1 / ( 1 - \rho _ { \alpha } )$ slots. The instantaneous downlink SINR experienced at cell $\pmb q$ from GBS m is:

$$
\gamma _ { m } ( { \pmb q } , t ) = \frac { P _ { m } G _ { m } ( { \pmb q } , t ) } { \sum _ { m ^ { \prime } \neq m } \alpha _ { m ^ { \prime } } ( t ) P _ { m ^ { \prime } } G _ { m ^ { \prime } } ( { \pmb q } , t ) + \sigma ^ { 2 } } ,\tag{4}
$$

where $P _ { m }$ is the transmit power and $\sigma ^ { 2 }$ is the noise power.

## B. Dynamic CKM Formulation

To avoid frequent handovers caused by fast fading, a cell q is associated with a serving GBS based on the expected large-scale SINR, given by $\begin{array} { r l } { \bar { \gamma } _ { m } ( \pmb q ) } & { { } = } \end{array}$ $\begin{array} { r } { P _ { m } \bar { G } _ { m } ( \mathbf { \bar { q } } ) / ( \sum _ { m ^ { \prime } \neq m } l _ { m ^ { \prime } } P _ { m ^ { \prime } } \bar { G } _ { m ^ { \prime } } ( \mathbf { q } ) \bar { + } \sigma ^ { 2 } ) } \end{array}$ . The index of the serving GBS is determined as $m ^ { * } ( \pmb { q } ) = \arg \operatorname* { m a x } _ { m \in \mathcal { M } } \bar { \gamma } _ { m } ( \pmb { q } )$ and is kept static.

Existing CKMs predominantly focus on tracking quasistatic large-scale components, attempting to store the reference map $\bar { C } ( \mathbf { q } ) ~ = ~ 1 0 \log _ { 1 0 } \bar { \gamma } _ { m ^ { * } } ( \mathbf { q } )$ (in dB) [3] and [8]. Since the numerator and the denominator of (4) are independent and $1 / x$ is convex, Jensen’s inequality gives $\mathbb { E } [ \gamma _ { m ^ { * } } ] \geq \bar { \gamma } _ { m ^ { * } }$ hence $\bar { C } \leq 1 0 \log _ { 1 0 } \mathbb { E } \left[ \gamma _ { m ^ { * } } \right] \ [ 3 ]$ . The inequality implies $\bar { C } ( \pmb q )$ has a definite downward bias compared to the true mean. More importantly, because the bias varies from cell to cell (depending on local interference and scattering), the bias cannot be eliminated by simply shifting the entire map by a global constant. Thus, the gap $\left| C ( \pmb q , t ) - \bar { C } ( \pmb q ) \right|$ represents an inherent error that no quasi-static construction can remove.

To precisely capture the variations, we define the dynamic CKM as the instantaneous SINR map:

$$
C ( \pmb { q } , t ) = 1 0 \log _ { 1 0 } \gamma _ { m ^ { * } } ( \pmb { q } , t ) \quad [ \mathrm { d } \mathrm { B } ] .\tag{5}
$$

The network’s objective is to estimate and track $C ( \boldsymbol { q } , t )$ slotby-slot, keeping $\bar { C } ( \pmb q )$ only as a reference baseline.

## C. UAV Measurement and Age of Information

Let $\mathbf { \Delta } _ { p _ { u , t } } \in \mathcal { D } _ { \mathbf { \Delta } }$ be the horizontal position of UAV u at the beginning of slot t. Subject to a maximum speed $V _ { \mathrm { m a x } } ,$ the flight distance per slot is $\| p _ { u , t + 1 } - p _ { u , t } \| _ { 2 } \leq V _ { \operatorname* { m a x } } \Delta t$ . During navigation, the UAV u records the channel quality of the cells along the trajectory. The maximum number of cells crossed per slot is bounded by $n _ { \mathrm { m a x } } = \lceil \sqrt { 2 } V _ { \mathrm { m a x } } \Delta t / \Delta _ { D } \rceil + 1$

Measurements are appended to an on-board finite queue $B _ { u } ( t )$ of capacity $Q ,$ whose oldest entries are dropped once $| B _ { u } ( t ) | > Q$ . In each slot, each UAV u reports a subset of measurements

$$
S _ { u , t } = \mathrm { S e l e c t } _ { K } \left( \mathcal { B } _ { u } ( t ) \right) , \qquad | S _ { u , t } | \leq K ,\tag{6}
$$

where Select $\kappa ^ { ( \cdot ) }$ is the selection rule studied in Section IV. The aggregated reports from the fleet is $\textstyle S _ { t } = \bigcup _ { u \in \mathcal { U } } S _ { u , t } .$

Because the fleet reports at most $K U \ll I J$ measurements per slot, most cells remain unmeasured. Moreover, since the dynamic channel decorrelates over time, only sparse measurements are available per slot. We track the staleness of the unmeasured cells using the AoI:

$$
A ( \pmb { q } , t ) = \left\{ \begin{array} { l l } { t - \tau ( \pmb { q } , t ) , } & { \pmb { q } \in S _ { t } , } \\ { \operatorname* { m i n } \{ A ( \pmb { q } , t - 1 ) + 1 , A _ { \operatorname* { m a x } } \} , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{7}
$$

where $\tau ( \boldsymbol { q } , t ) < t$ is the generation slot of the measurement reported at time t for cell ${ \mathbf { } } q ,$ and $A _ { \mathrm { m a x } }$ bounds the AoI. Since relying solely on sparse and outdated measurements inevitably degrades the CKM, to minimize the resultant construction error, the central server needs to jointly optimize the UAV trajectories and reporting decisions.

## III. PROBLEM FORMULATION AND LEARNING-BASED PREDICTION

## A. Dynamic CKM Tracking Problem

To maintain an accurate dynamic CKM despite sparse measurements, the central server maintains a dynamically updated estimate, denoted by $C ^ { \prime } ( \pmb { q } , t )$ . The accuracy of the CKM estimate is evaluated by the spatial error distribution, defined as the AEM: $E ( \pmb { q } , t ) = | C ( \pmb { q } , t ) - C ^ { \prime } ( \pmb { q } , t )$ |. AEM $E ( \boldsymbol { q } , t )$ must be distinguished from the static gap $\left| C ( \pmb q , t ) - \bar { C } ( \pmb q ) \right|$ discussed in Section II. While the static gap represents an unremovable inherent bias of existing static maps, $E ( \boldsymbol { q } , t )$ quantifies how accurately the central server tracks the true dynamic channel $C ( \boldsymbol { q } , t )$

To keep the tracking error as small as possible, the central server needs to jointly optimize the UAV trajectories $Q =$ $\{ p _ { u , t } \}$ and the reporting decisions $S = \{ S _ { u , t } \}$ , to minimize the time-averaged Mean Squared Error (MSE) over the entire area D. The problem is formulated as:

$$
( \mathrm { P } ) \quad \operatorname* { m i n } _ { \pmb { Q } , \mathscr { S } } \quad \frac { 1 } { I J T } \sum _ { t = 1 } ^ { T } \sum _ { \pmb { q } \in \mathscr { D } } E ^ { 2 } ( \pmb { q } , t )
$$

$$
\begin{array} { r l } { \mathrm { s . t . ~ } } & { { } | S _ { u , t } | \leq K , } \end{array}\tag{8}
$$

$$
| B _ { u } ( t ) | \le Q ,\tag{9}
$$

$$
\begin{array} { r } { S _ { u , t } \subseteq B _ { u } ( t ) , } \end{array}\tag{10}
$$

$$
B _ { u } ( t ) = B _ { u } ( t - 1 ) \cup \mathcal { P } _ { u , t } ,\tag{11}
$$

$$
\| p _ { u , t + 1 } - p _ { u , t } \| _ { 2 } \leq V _ { \operatorname* { m a x } } \Delta t , \quad p _ { u , t } \in \mathcal { D } ,\tag{12}
$$

$$
\| p _ { u , t } - p _ { u ^ { \prime } , t } \| _ { 2 } \geq d _ { \operatorname* { m i n } } , \quad u \neq u ^ { \prime } ,\tag{13}
$$

with (8)–(13) required for every $u \in \mathcal { U }$ and $t \in \mathcal T$ . Specifically, (8) limits the maximum number of measurements a UAV can report per slot due to the uplink budget; (9) enforces the finite on-board queue capacity; (10) ensures only measurements currently in the queue can be reported; (11) governs the queue evolution, removing reported data and appending new UAV measurements; (12) bounds the flight distance per slot based on the maximum speed $V _ { \mathrm { m a x } }$ while keeping the UAV within the target area; and (13) mandates a minimum safety distance $d _ { \mathrm { m i n } }$ to prevent inter-UAV collisions.

Problem (P) is a highly coupled, mixed-integer, and stochastic optimization problem. A fundamental paradox in solving (P) directly is that evaluating the true AEM $E ( \boldsymbol { q } , t )$ in the objective requires the true dynamic map $C ( \boldsymbol { q } , t )$ . The dynamic map is inherently unobservable, because reconstructing the map is the very goal of the system.

## B. Spatial–Temporal Prediction via 2-D U-Net

Without access to the true AEM, a common practice is to rely on the observable AoI, prioritizing cells that have not been measured for the longest time. However, AoI alone is an inadequate proxy for actual channel deviation. AoI merely measures the time elapsed since the last update; a cell with a large AoI in a stable environment may still hold an accurate value, whereas a cell with a small AoI under bursty interference might already be highly erroneous.

Therefore, rather than relying solely on AoI, predicting the actual spatial error distribution is imperative. We introduce two 2-D U-Nets to predict the estimated map $\hat { C } ( t )$ and the predicted AEM $\hat { \mathbf { { \mathcal { E } } } } ( t )$ respectively, at the beginning of slot t. The two predictors use 2-D U-Nets similar to the design in [5], with three down-sampling stages and $N _ { \mathrm { n e } } = 1 . 2 \ : \mathrm { M }$ parameters each, but with independent weights.

The predictors utilize the previous map estimate and the label of the previous slot, $C ^ { \prime } ( t - 1 )$ and $\mathbf { } Y ( t - 1 )$ , together with a mask of fresh measurements. Let $I _ { t } = \mathbf { 1 } \{ A ( \cdot , t ) \leq a _ { \mathrm { t h } } \}$ be a binary indicator for cells whose AoI is below a freshness threshold $a _ { \mathrm { t h } }$ . Since a cell with $A > a _ { \mathrm { t h } }$ was measured long ago and may mislead the network [9], the label of slot t keeps the measured value only at fresh cells and the previous estimate elsewhere,

$$
Y ( { \pmb q } , t ) = \left\{ { \begin{array} { l l } { C ( { \pmb q } , \tau ( { \pmb q } , t ) ) , } & { A ( { \pmb q } , t ) \leq a _ { \mathrm { t h } } , } \\ { C ^ { \prime } ( { \pmb q } , t - 1 ) , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{14}
$$

The label is fixed once the reports of slot t are in, so the same image serves two purposes: as the training target of slot t and, at the start of slot $t + 1$ , as an input carrying the last reliable measurements. The input tensor to the networks is therefore $\pmb { X } ^ { l } ( t ) = \lceil \pmb { M } ^ { l } ( t ) , \ \pmb { I } _ { t - 1 } , \ \pmb { C } ^ { \prime } ( t - 1 ) , \ \pmb { Y } ( t - 1 ) \rceil$ , where $\mathbf { } Y ( t - 1 )$ supplies the measurements that were fresh at $t - 1$ . Unlike $C ^ { \prime } ( t - 1 ) , Y ( t - 1 )$ distinguishes the measured values from predicted values. $M ^ { l } ( t )$ is a Bernoulli mask generated with probability $p = 0 . 8$ (Table I), used to randomly hide a fraction of fresh cells during training to force spatial interpolation (set to an all-ones mask at inference). The two predictors, $f ^ { C }$ and $f ^ { E }$ , are defined as:

$$
\hat { C } ^ { l } ( t ) = f ^ { C } \big ( \boldsymbol { X } ^ { l } ( t ) \big ) , \quad \hat { E } ^ { l } ( t ) = f ^ { E } \big ( \boldsymbol { X } ^ { l } ( t ) , \hat { C } ^ { l } ( t ) \big ) .\tag{15}
$$

The error predictor $f ^ { E }$ takes the predicted map $\hat { C } ^ { l } ( t )$ as an additional input to evaluate the residual uncertainty.

## C. Online Update and Training

Once the subset of measurements $S _ { t }$ is reported to the central server by the end of slot t, the global map is updated online. The reported cells adopt the actual measured values, while the unmeasured cells retain the predicted values:

$$
\begin{array} { r } { C ^ { \prime } ( \pmb { q } , t ) = \left. \begin{array} { l l } { C ( \pmb { q } , \tau ( \pmb { q } , t ) ) , } & { \pmb { q } \in S _ { t } , } \\ { \hat { C } ( \pmb { q } , t ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{16}
$$

The neural networks are trained online to adapt to the timevarying channel statistics. Since stale cells $( A > a _ { \mathrm { t h } } )$ contain severely drifted values that mislead the learning process, the network is trained exclusively on fresh measurements, creating a self-supervised target $Y ( \pmb { q } , t )$ in (14). The loss function $L ^ { C }$ for the map predictor $f ^ { C }$ comprises three components: map-estimation loss on visible fresh cells, interpolation loss on hidden fresh cells (masked by $M ^ { l } )$ , and a temporal consistency loss on stale cells:

$$
\begin{array} { r l } & { \displaystyle L ^ { C } = \frac { 1 } { N _ { \mathrm { m a } } } \sum _ { l = 1 } ^ { N _ { \mathrm { m a } } } { \left[ \frac { \| ( \hat { C } ^ { l } - { \cal Y } ) \odot { \cal M } ^ { l } \odot { \cal I } _ { t } \| _ { 2 } ^ { 2 } } { I J } \right. } } \\ & { \displaystyle + \left. \lambda _ { 1 } \frac { \| ( \hat { C } ^ { l } - { \cal Y } ) \odot ( 1 - { \cal M } ^ { l } ) \odot { \cal I } _ { t } \| _ { 2 } ^ { 2 } } { I J } + \lambda _ { 2 } \frac { \| ( \hat { C } ^ { l } - { \cal Y } ) \odot ( 1 - I _ { t } ) \| _ { 2 } ^ { 2 } } { I J } \right] } ,  \end{array}\tag{17}
$$

where ⊙ denotes the element-wise product, $N _ { \mathrm { m a } }$ is the number of masks per batch, and $\lambda _ { 1 } , \lambda _ { 2 }$ are weighting coefficients. The error predictor $f ^ { E }$ is trained similarly against the residual $| Y ( t ) - \hat { C } ^ { l } ( t ) |$ with gradients detached from $\hat { C } ^ { l }$

With the predicted AEM $\dot { E } ( \pmb q , t )$ available, the unobservable true error $E ( \boldsymbol { q } , t )$ in the objective of Problem (P) is substituted by the predicted AEM. Nevertheless, solving (P) remains highly intractable. The variables are combinatorial sets, the stochastic channel transitions lack a closed-form model, and the decisions are strictly coupled across time slots via the queue evolution in (11) and the UAV mobility in (12). To handle the complexity, we design a greedy yet theoretically guaranteed metric-driven online algorithm in Section IV.

## IV. METRIC-DRIVEN UAV TRAJECTORY AND REPORTING

In this section, we first theoretically derive the weighted-AoI (W-AoI) as a unified cell-level priority metric, and subsequently design a mobility-constrained greedy trajectory policy to actively suppress the W-AoI.

## A. Weighted-AoI and Error Upper Bound

Intuitively, the longer a cell remains unmeasured, the more its true channel drifts; and this drift occurs faster in regions with severe spatial variations. Hence, the W-AoI is defined as the product of the AoI and the predicted AEM as follows:

$$
W ( \pmb q , t ) = A ( \pmb q , t ) \hat { E } ( \pmb q , t ) .\tag{18}
$$

To theoretically justify W-AoI, consider any unmeasured cell ${ \textbf { \textit { q } } } \notin S _ { t }$ with $A ( \mathbf { q } , t ) < A _ { \operatorname* { m a x } }$ . Let $t _ { 0 } ~ < ~ t$ be its most recent reporting slot. Its AoI was reset to $t _ { 0 } - \tau ( \pmb q , t _ { 0 } )$ upon reception at $t _ { 0 }$ . Since it remains unreported for the subsequent $t - t _ { 0 }$ slots, its AoI increments by 1 per slot, unfolding to $A ( \pmb q , t ) = ( t _ { 0 } - \tau ( \pmb q , t _ { 0 } ) ) + ( t - t _ { 0 } ) = t - \tau ( \pmb q , t _ { 0 } )$ . Defining $u ( \pmb q , t ) \triangleq \tau ( \pmb q , t _ { 0 } )$ as the generation slot of the currently stored measurement, we obtain $A ( \pmb q , t ) = t - u ( \pmb q , t ) \geq 1$

Lemma 1. Assume that, for every cell $\pmb q \in \mathcal { D }$ and every slot t, the learning-based predictor satisfies two per-cell calibration conditions: (i) the drift that the true map accumulates since the stored measurement is bounded by the predicted AEM, i.e., $\begin{array} { r } { | C ( \pmb { q } , t ) - C ( \pmb { q } , u ( \pmb { q } , t ) ) | \leq \kappa _ { 1 } \hat { E } ( \pmb { q } , t ) A ( \pmb { q } , t ) , } \end{array}$ ; and (ii) the prediction baseline error is bounded by the predicted AEM, $| C ( \pmb { q } , u ( \pmb { q } , t ) ) - C ^ { \prime } ( \pmb { q } , t ) | \leq \kappa _ { 2 } \hat { E } ( \pmb { q } , t )$ , for constants $\kappa _ { 1 } > 0 , \kappa _ { 2 } \geq 0$ . Then, the true AEM of every cell is upperbounded by the W-AoI:

$$
E ( \pmb { q } , t ) \le \kappa W ( \pmb { q } , t ) , \quad \kappa = \kappa _ { 1 } + \kappa _ { 2 } .\tag{19}
$$

Proof: Inserting the true sampled value $C ( \pmb q , \boldsymbol u ( \pmb q , t ) )$ and applying the triangle inequality yields $E ( { \bf q } , t ) = | C ( { \bf q } , t ) -$ $\begin{array} { r l r } { C ^ { \prime } ( { \bf q } , t ) \vert } & { { } \le } & { \vert C ( { \bf q } , t ) - C ( { \bf q } , u ( { \bf q } , t ) ) \vert + \vert C ( { \bf q } , u ( { \bf q } , t ) ) - \vert } \end{array}$ $C ^ { \prime } ( \pmb { q } , t ) |$ |, which conditions (i) and (ii) bound by $\big ( \kappa _ { 1 } A ( \pmb q , t ) +$ $\kappa _ { 2 } ) \hat { E } ( \pmb q , t )$ . For every cell with $A ( \pmb q , t ) ~ \geq ~ 1$ , we have $\begin{array} { r l r } { \kappa _ { 2 } } & { { } \le } & { \kappa _ { 2 } A ( \pmb { q } , t ) } \end{array}$ , which establishes $\begin{array} { l c l } { E ( \pmb q , t ) } & { \leq } & { ( \kappa _ { 1 } \ + } \end{array}$ $\kappa _ { 2 } ) A ( \pmb { q } , t ) \hat { E } ( \pmb { q } , t ) = \kappa W ( \pmb { q } , t )$

Remark 1: Conditions (i) and (ii) in Lemma 1 constrain the predictor, not the physical channel. They require the U-Net to be properly calibrated such that its announced spatial error $\hat { E }$ structurally encapsulates both the true temporal drift rate $\kappa _ { 1 }$ and the inherent prediction noise $\kappa _ { 2 }$

## B. Optimal Cell Set for AEM Reduction

By squaring and summing the bound from Lemma 1 over all cells, the total squared AEM at slot t is upper-bounded by $\begin{array} { r } { \sum _ { \pmb { q } \in \mathcal { D } } E ^ { 2 } ( \pmb { q } , t ) \le \bar { \kappa ^ { 2 } } \sum _ { \pmb { q } \in \mathcal { D } } W ^ { 2 } ( \pmb { q } , t ) } \end{array}$

Suppose UAV u selects a subset of cells $\mathcal { Q } _ { u , t }$ to measure. Due to the limited uplink budget and on-board queueing, a new measurement may take $\delta \ \geq \ 0$ slots to be delivered to the central server. Upon successful delivery, the AoI of a refreshed cell resets to this delay δ. Assuming that the predicted AEM $\hat { E } ( \boldsymbol { q } , t )$ remains stable during this short delay, the squared AEM bound of this specific cell drops from $\kappa ^ { 2 } A ^ { 2 } ( \pmb q , t ) \hat { E } ^ { 2 } ( \pmb q , t )$ to $\kappa ^ { 2 } \delta ^ { 2 } \hat { E } ^ { 2 } ( \boldsymbol { q } , t )$ . Therefore, the total decrease in the squared AEM upper bound, which can serve as the reward function for selecting $\mathcal { Q } _ { u , t } ,$ , is defined as $\begin{array} { r } { R ( \mathcal { Q } _ { u , t } ) = \kappa ^ { 2 } \sum _ { \pmb { q } \in \mathcal { Q } _ { u , t } } \left( A ^ { 2 } ( \pmb { q } , t ) - \delta ^ { \hat { 2 } } \right) \hat { E } ^ { 2 } ( \pmb { q } , t ) } \end{array}$

Next, we consider an idealized scenario where the UAV is unrestricted by mobility and collision constraints and measurements are delivered instantly $( \mathrm { i . e . , ~ } \delta ~ = ~ 0 )$ , and obtain Proposition 1.

Proposition 1. For a fixed UAV u with a reporting capacity of K cells per slot, assume the delivery delay is zero $( \delta = 0 )$ To maximize the AEM reduction reward $R ( \mathcal { Q } _ { u , t } )$ , the optimal cell set $\mathcal { Q } _ { u , t } ^ { * }$ consists of the K cells in D possessing the highest $W { \cdot } A o I { \cdot }$

$$
\begin{array} { r } { \mathcal { Q } _ { u , t } ^ { * } = \mathrm { T o p K } _ { \pmb q \in \mathcal { D } } W ( \pmb q , t ) , \quad | \mathcal { Q } _ { u , t } ^ { * } | \leq K . } \end{array}\tag{20}
$$

Proof: When $\begin{array} { r l r l r l } { \delta } & { { } } & { = } & { } & { { } 0 , } & { { } R ( \mathcal { Q } _ { u , t } ) } \end{array}$ reduces to $\textstyle \kappa ^ { 2 } \sum _ { { \pmb q } \in \mathcal { Q } _ { * } , \textit { \pmb + } } W ^ { 2 } ( { \pmb q } , t )$ Under the cardinality constraint $\left| \mathcal { Q } _ { u , t } \right| \leq \mathrm { ~ \bar { ~ } { ~ K ~ } ~ }$ , maximizing $R ( \mathcal { Q } _ { u , t } )$ is achieved by selecting the K elements with the largest ${ \dot { W } } ^ { 2 } ( \pmb q , t )$ . Since the mapping $s \mapsto s ^ { 2 }$ is strictly increasing for $s \geq 0$ , ranking cells by $W ^ { 2 } ( \pmb q , t )$ or $W ( \pmb { q } , t )$ induces the same Top-K selection. ■

![](images/9fdf4a1f6fa2c46ed4f5a849c60f0eaa605f6d3c26f509fda03dc8aec359ebed.jpg)  
(a)

![](images/5fbea4b1ee071b3b92c7580dd59bce4e39950851f80deba93e11bbc49d0ba2b3.jpg)  
(b)  
Fig. 1. Mobility-constrained trajectory. (a) Fixed Voronoi subregions, one per UAV. (b) A UAV can reach the cells within $V _ { \mathrm { m a x } } \Delta t$ of $_ { p _ { u , t } }$ . The numbers are weighted-AoIs and the UAV flies to the largest valid one (41.5). Cells crossed on the way are measured (shaded) but do not enter the decision.

## C. Reachable Set and Greedy Trajectory Policy

To implement the optimal Top-K rule under practical mobility and collision constraints, we design a greedy trajectory policy.

To distribute the measurement workload across the fleet and prevent all UAVs from flocking to a single hotspot, the target area D is partitioned into U disjoint subregions $\mathcal { R } _ { 1 } , \ldots , \mathcal { R } _ { U }$ via a Voronoi tessellation based on the initial UAV positions $\mathbf { \mathit { p } } _ { u , 0 } { } ^ { 1 }$ . Each UAV u is restricted to measure cells within its designated fixed subregion $\mathcal { R } _ { u } ,$ as illustrated in Fig. 1(a).

Bounded by the maximum flight speed $V _ { \mathrm { m a x } }$ and the slot duration $\Delta t .$ the reachable set of cells for UAV u from its current position ${ \mathbf { } } p _ { u , t }$ in the next slot is limited to:

$$
\mathcal { Q } _ { u , \mathrm { f e a s } } ( \pmb { p } _ { u , t } ) = \{ \pmb { q } \in \mathcal { R } _ { u } : \Vert \pmb { p } _ { u , t } - \pmb { q } \Vert _ { 2 } \leq V _ { \operatorname* { m a x } } \Delta t \} .\tag{21}
$$

At each slot, UAV u greedily targets the reachable cell exhibiting the highest W-AoI priority:

$$
\pmb { q } _ { u , t + 1 } ^ { * } = \arg \operatorname* { m a x } _ { \pmb { q } \in \mathcal { Q } _ { u , \mathrm { f e a s } } ( \pmb { p } _ { u , t } ) } W ( \pmb { q } , t ) .\tag{22}
$$

To guarantee collision-free navigation, if flying towards $\pmb q _ { u , t + 1 } ^ { * }$ brings UAV u within the minimum safety distance $d _ { \mathrm { m i n } }$ of another UAV, this candidate cell is skipped in favor of the next best cell in $\mathcal { Q } _ { u , \mathrm { f e a s } }$

The selection process is visually depicted in Fig. 1(b). It is worth noting that while the decision logic only explicitly evaluates the candidate cells, the UAV continuously senses the environment during its flight. As shown by the shaded cells in Fig. 1(b), all cells physically crossed along the straightline trajectory to the target cell are measured. These alongpath measurements form $\mathcal { P } _ { u , t }$ and join the on-board queue $B _ { u } ( t )$ , from which (6) reports the $K$ cells with the largest W-AoI, the queue-constrained counterpart of (20), thereby prioritizing informative measurements and facilitating AEM error reduction across the map.

The complete procedure is summarized in Algorithm 1. The per-slot computational cost is dominated by the U-Net forward passes, each requiring $\mathcal { O } ( N _ { \mathrm { n e } } )$ operations, where $N _ { \mathrm { n e } }$ is the network size. Waypoint selection scans only the local reachable set, costing $\mathcal { O } ( | \mathcal { Q } _ { u , \mathrm { f e a s } } | )$ per UAV, while grid traversal costs $\mathcal { O } ( n _ { t } )$ for $n _ { t }$ crossed cells. Both costs scale with the single-slot UAV coverage area rather than the global map size.

Algorithm 1 Joint Mobility and Measurement Scheduling   
Require: $U , K , Q , T , a _ { \mathrm { t h } } ,$ predictors $f ^ { C } , f ^ { E } , C ^ { \prime } ( 0 ) ,$ A<sub>0</sub>   
1: A←A<sub>0</sub>, $\{ { \cal C } ^ { \prime } \bar { ( 0 ) } , { \cal Y } ( \bar { 0 ) } \}  \bar { { \cal C } } , \ B _ { u } \bar {  } \emptyset$   
2: for $t = 1 , \cdot \cdot \cdot , \stackrel { \cdot } { T }$ do   
3: $X  [ M , \ I _ { t - 1 } , \ C ^ { \prime } ( t - 1 ) , \ Y ( t - 1 ) ]$ ▷ M=1   
4: $\hat { C } \gets \bar { f } ^ { C } ( \boldsymbol { X } ) ; \hat { E } \gets f ^ { E } ( \boldsymbol { X } , \hat { C } )$ ▷ no slot-t measurement   
5: $\begin{array} { r } { W ( \pmb q )  A ( \pmb q ) \hat { E } ( \pmb q ) ; \ q _ { u , t + 1 } ^ { * }  \arg \operatorname* { m a x } _ { Q _ { u , \mathrm { f e a s } } } W } \end{array}$   
6: Fly to $\pmb q _ { u , t + 1 } ^ { * }$ of (22), measure the crossed cells $\mathcal { P } _ { u , t }$   
7: $B _ { u } ( t ) \gets \ddot { B } _ { u } ( t - 1 ) \cup \mathcal { P } _ { u , t } ,$ , drop oldest if $| B _ { u } ( t ) | > Q$   
8: $\boldsymbol { S } _ { u , t } \gets \mathrm { S e l }$ lect $\kappa ( B _ { u } ( t ) )$ of (6)   
9: update $C ^ { \prime } ( t )$ by (16); update $A ( \pmb q , t )$ by (7)   
10: form $\mathbf { } I _ { t }$ and the label $\dot { \mathbf { Y } } ( t )$ of (14)   
11: if a full batch of training examples is collected then   
12: update $f ^ { C } , f ^ { E }$ by one gradient step on (17)   
13: end if   
14: end for

## V. PERFORMANCE EVALUATION

## A. Scenario and Parameters

We simulate a $3 0 0 \times 3 0 0$ m area on a 2-D grid with $\Delta _ { D } = 5 \textrm { m } ( I = J = 6 0 )$ , UAV altitude $h _ { 0 } = 1 0 0$ m, and $M \ = \ 6 \ \mathrm { G B S s }$ . Each episode lasts 1000 slots $( \Delta t = 1 \mathrm { ~ s ~ } )$ The predictors are trained for 100 episodes $( 1 0 ^ { 5 }$ slots) and evaluated over 10 unseen seeds; results are averaged over the last 100 slots of each evaluation episode to exclude the initial transient. The area contains eight non-overlapping buildings with side lengths uniformly drawn from [50, 70] m and truncated-Rayleigh heights (mean 30 m), covering about 30% of the area. Building blockage determines each GBS–cell link’s LoS/NLoS branch, while UAVs fly above all rooftops. Each GBS uses an eight-element vertical array with $- 1 0 ^ { \circ }$ downtilt [3]. The shared band is 180 kHz at 2 GHz, with load factors $\{ 0 . 0 5 , 0 . 9 5 , 0 . 1 0 , 0 . 9 0 , 0 . 0 5 , 0 . 9 5 \}$ . The maximum flight distance is 30 m per slot, corresponding to $n _ { \mathrm { m a x } } = 1 0$ crossed cells (seven for an axis-aligned flight), and $d _ { \operatorname* { m i n } { } } =$ 10 m. Unless otherwise stated, $U ~ = ~ 4 , ~ K ~ = ~ 4 .$ and $Q = 3 2 ,$ , so $K U \mathrm { ~ = ~ } 1 6$ cells can be refreshed per slot and $a _ { \mathrm { t h } } ~ = ~ \lceil I J / ( K U ) \rceil ~ = ~ 2 2 5$ . The remaining parameters are listed in Table I.

TABLE I  
SIMULATION PARAMETERS
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td> $V _ { \mathrm { m a x } }$ </td><td>30 m/s</td><td> $A _ { 0 }$ </td><td>10</td></tr><tr><td> ${ \mathrm { Q u e u e ~ } } Q$ </td><td>32</td><td> $A _ { \mathrm { m a x } }$ </td><td>2000</td></tr><tr><td>Tx power</td><td>24 dBm</td><td>Noise PSD</td><td>-169 dBm/Hz</td></tr><tr><td>Noise fig.</td><td>9 dB</td><td> $\rho _ { h } / \rho _ { \alpha }$ </td><td>0.99</td></tr><tr><td>Params (each)</td><td>1.2 M</td><td> $\lambda _ { 1 } / \lambda _ { 2 }$ </td><td>4/0.4</td></tr><tr><td>Optimizer</td><td>Adam</td><td> $N _ { \mathrm { m a } } / p$ </td><td>8/0.8</td></tr><tr><td>U-net base ch.</td><td>24</td><td>Batch</td><td>32</td></tr><tr><td>lr (train/eval)</td><td> $1 0 ^ { - 3 } / 1 0 ^ { - 4 }$ </td><td>Episodes (train/eval)</td><td>100/10</td></tr></table>

## B. Baselines and Metrics

All policies share the same channel realizations, initial maps and UAV positions, Voronoi subregions, measurement budgets, and random seeds. Random selects reachable waypoints and reports randomly; FIFO follows a serpentine route and reports the oldest queued samples. AoI-only $( W = A )$ and AEMonly $( W = { \hat { E } } )$ use one factor to rank both waypoints and reports, while W-AoI uses their product. Oracle replaces $\hat { E }$ with the true AEM E and serves only as a reference, since the trajectory policy remains greedy. Kriging [4] and Diffusion [6] use the same waypoint and report selection rules but replace the proposed temporal CKM/AEM predictor. Kriging performs online interpolation using measurements with $A \ \leq \ a _ { \mathrm { t h } }$ , whereas Diffusion adopts the same 2-D U-Net backbone with iterative denoising and online training. All predictors are trained and evaluated on the same hardware platform under identical software configurations.

## C. Results

Table II reports the steady-state MSE, its reduction relative to the static map C<sup>¯</sup> (Gain), and the amortized per-map computational time (Time), including online training overhead. The pure inference times are 3.06 ms for 2D-U-Net and 56.9 ms for Diffusion. The amortized times in Table II are 8 ms for the proposed predictor, compared with 39.4 ms for Kriging and 145.6 ms for Diffusion, demonstrating its clear computational advantage. W-AoI achieves $\mathrm { 5 . 5 1 \ d B ^ { 2 } }$ , reducing the error by 3.77 and 4.26 $\mathrm { { d B ^ { 2 } } }$ over AoI-only (9.28) and AEM-only (9.77), and by 4.82 and 5.01 $\mathrm { { d B ^ { 2 } } }$ over FIFO (10.33) and Random (10.52), respectively. Neither freshness nor predicted error alone captures both sources of estimation error. Oracle obtains $5 . 0 6 ~ \mathrm { d B ^ { \bar { 2 } } }$ , only 0.45 $\mathrm { { d B ^ { 2 } } }$ below W-AoI. A two-sided paired t-test across the ten episodes gives $p < 0 . 0 5$ against every baseline and $p \approx 0 . 5 6$ against Oracle, indicating significant gains over the baselines but no significant gap from Oracle.

Replacing the temporal predictor with Kriging or Diffusion increases the MSE to 11.49 and 13.21 $\mathrm { { d B ^ { 2 } } }$ , respectively, both worse than Random. Kriging treats measurements collected up to $a _ { \mathrm { t h } }$ slots apart as samples of one stationary map, thereby smoothing temporal drift; the temporal predictor instead uses the previous reconstruction to track this drift. Diffusion generates plausible maps through iterative denoising, whereas the cell-wise MSE favors direct regression and is affected by generation variance.

TABLE II  
STEADY-STATE ESTIMATION ERROR AT $K = 4 , U = 4 ; { \mathrm { M E A N } } \pm { \mathrm { S T D ~ O V E } }$ R THE CELLS AND THE 10 EVALUATION EPISODES.
<table><tr><td>Policy</td><td>Constructor</td><td> $\mathbf { M S E _ { \theta } ( d B ^ { 2 } ) }$ </td><td> $\mathrm { { G a i n } ~ ( d B ^ { 2 } ) }$ </td><td>Time (ms)</td></tr><tr><td>Random</td><td>2-D U-Net</td><td> $1 0 . 5 2 \pm 6 . 0 1$ </td><td> $4 . 4 7 \pm 6 . 4 3$ </td><td>8</td></tr><tr><td>FIFO</td><td>2-D U-Net</td><td> $1 0 . 3 3 \pm 5 . 0 7$ </td><td> $4 . 6 6 \pm 5 . 3 9$ </td><td>8</td></tr><tr><td>AoI-only</td><td>2-D U-Net</td><td> $9 . 2 8 \pm 5 . 4 9$ </td><td> $5 . 7 2 \pm 4 . 9 7$ </td><td>8</td></tr><tr><td>AEM-only</td><td>2-D U-Net</td><td> $9 . 7 7 \pm 5 . 5 9$ </td><td> $5 . 2 2 \pm 4 . 8 6$ </td><td>8</td></tr><tr><td>W-AoI (Proposed)</td><td>2-D U-Net</td><td> ${ \bf 5 . 5 1 \pm 4 . 7 0 }$ </td><td> ${ \bf 9 . 4 9 \pm 5 . 2 2 }$ </td><td>8</td></tr><tr><td>Oracle</td><td>2-D U-Net</td><td> ${ \bf 5 . 0 6 \pm 4 . 8 9 }$ </td><td> ${ \bf 9 . 9 4 \pm 5 . 4 2 }$ </td><td>8</td></tr><tr><td>W-AoI</td><td>Kriging [4]</td><td> $1 1 . 4 9 \pm 4 . 7 7$ </td><td> $3 . 5 1 \pm 4 . 8 6$ </td><td>39.4</td></tr><tr><td>W-AoI</td><td>Diffusion [6]</td><td> $1 3 . 2 1 \pm 3 . 3 3$ </td><td> $1 . 7 9 \pm 2 . 6 6$ </td><td>145.6</td></tr><tr><td>No refresh</td><td>Static </td><td> $1 5 . 0 0 \pm 1 . 8 7$ </td><td>0</td><td>0</td></tr></table>

![](images/4b04625ad6295d21ba31a7a9ee2629eb8a945c9eeeccb0d4dec493a7db461b0c.jpg)

![](images/a2f29ec33579fb5bdf55ebd60d6c259962f76c57b48ad3961153d1bbe1f736c4.jpg)  
Fig. 2. Estimation error against the per-UAV reporting budget K at $U = 4$ and the number of UAVs U at $K \bar { = } 4$ . Each point averages the last 100 slots of the 10 evaluation episodes; the dotted line is the MSE of the static reference map C<sup>¯</sup>

Effect of reporting budget K. Fig. 2 (a) evaluates $K \in$ {1, 2, 4, 6, 8, 10} at $U = 4$ . At K = 1, only four measurements are reported per slot although a flight can cross ten cells; the queue therefore grows and the proposed rule improves over AoI-only by just 1.41 dB<sup>2</sup> (8.91 versus 10.32). Increasing K from 2 to 4 reduces the MSE by $1 . 6 1 ~ \mathrm { d B ^ { 2 } } ;$ ; beyond $K = 4 ,$ it changes by less than $\mathrm { 0 . 3 ~ d B ^ { 2 } ~ ( 5 . 5 1 , 5 . 3 9 , 5 . 2 5 }$ , and 5.23 for $K = 4 , 6 , 8 , 1 0 )$ . Once the reporting budget can accommodate the cells crossed during flight, the queue is cleared within a slot and mobility, rather than reporting capacity, determines freshness. For $K \geq 2 ,$ , the gain over AoI-only remains 2.1– $3 . 8 \ \mathrm { d B ^ { 2 } }$ , while the Oracle gap narrows from $\mathrm { 0 . 6 2 \ d B ^ { 2 } }$ at $K =$ 2 to about $0 . 3 { \mathrm { ~ d B } } ^ { 2 }$ for $K \geq 6 .$ , showing that prediction error becomes the main remaining limitation.

Effect of fleet size U. Fig. 2 (b) evaluates $U \in \{ 1 , 2 , 4 , 9 , 1 6 \}$ at $K = 4$ . The fleet refreshes KU cells per slot, giving a mean revisit interval of $I J / ( K U )$ , from 900 slots at $U = 1 ~ \mathrm { t o }$ 56 at $U = 1 6 . \mathrm { A t } U = 1$ , all policies have MSE above $8 . 5 \ \mathrm { d B ^ { 2 } }$ , and W-AoI improves over AoI-only by $0 . 9 4 { \mathrm { ~ d B } } ^ { 2 }$ . Its gain peaks at $U = 4 ~ ( 3 . 7 7 ~ \mathrm { d B ^ { 2 } } )$ ), where the revisit interval is about twice the fading correlation time, $1 / ( 1 - \rho _ { h } ) \approx 1 0 0$ slots: channel changes are large enough for predicted errors to distinguish cells, while their ages remain relevant to scheduling. With fewer UAVs, most cells become similarly stale; with more ${ \mathrm { U A V s } } ,$ fewer cells remain stale. For $U > 9$ , the revisit interval falls below the correlation time and all policies improve. The gap between Random and W-AoI narrows from $5 . 0 \ \mathrm { d B ^ { 2 } }$ at $U = 4$ to 0.82 and 1.21 $\mathrm { { d B ^ { 2 } } }$ at $U = 9$ and 16, respectively. At $U \ = \ 1 6 .$ , W-AoI achieves 3.89 $\mathrm { { d B ^ { 2 } } }$ , outperforming the $U = 4$ Oracle (5.06 dB<sup>2</sup>).

## VI. CONCLUSION

This paper studied the long-term CKM estimation error minimization problem, to properly maintain the dynamic CKM. We proposed the W-AoI driven UAV scheduling approach to tackle the unobservable dynamic errors. We proved that by fusing exact temporal staleness with U-Net-predicted spatial uncertainty, the W-AoI metric provides a theoretically guaranteed upper bound on the reconstruction error. We proposed the lightweight greedy trajectory policy incorporating continuous along-path sensing, which maximizes data yield to accelerate map updates. Extensive simulations validated the superiority of our proposed method.

## REFERENCES

[1] 3GPP, “Enhanced LTE support for aerial vehicles (release 15),” 3rd Generation Partnership Project (3GPP), Technical Report TR 36.777, 2018.

[2] Y. Zeng, J. Chen, J. Xu, D. Wu, X. Xu, S. Jin, X. Gao, D. Gesbert, S. Cui, and R. Zhang, “A tutorial on environment-aware communications via channel knowledge map for 6G,” IEEE communications surveys & tutorials, vol. 26, no. 3, pp. 1478–1519, 2024.

[3] S. Zhang and R. Zhang, “Radio map-based 3D path planning for cellularconnected UAV,” IEEE Transactions on Wireless Communications, vol. 20, no. 3, pp. 1975–1989, 2020.

[4] K. Sato and T. Fujii, “Kriging-based interference power constraint: Integrated design of the radio environment map and transmission power,” IEEE Transactions on Cognitive Communications and Networking, vol. 3, no. 1, pp. 13–25, 2017.

[5] Y. Teganya and D. Romero, “Deep completion autoencoders for radio map estimation,” IEEE Transactions on Wireless Communications, vol. 21, no. 3, pp. 1710–1724, 2021.

[6] H. Rong, Y. Zeng, X. Xu, R. Li, and J. Xu, “3DCKMDiff: 3D CKM construction based on generative diffusion for environment-aware lowaltitude communication and sensing,” in 2025 IEEE/CIC International Conference on Communications in China (ICCC Workshops). IEEE, 2025, pp. 1–6.

[7] R. Shrestha, D. Romero, and S. P. Chepuri, “Spectrum surveying: Active radio map estimation with autonomous UAVs,” IEEE transactions on wireless communications, vol. 22, no. 1, pp. 627–641, 2022.

[8] Y. Song, H. Lu, C. Zhang, B. Zheng, and Y. Zeng, “Trajectory optimization for cellular-connected UAV in complex environment with partial CKM,” IEEE Transactions on Communications, 2026.

[9] X. Xu and Y. Zeng, “How much data is needed for channel knowledge map construction?” IEEE Transactions on Wireless Communications, vol. 23, no. 10, pp. 13 011–13 021, 2024.

[10] K. Katagiri, K. Sato, K. Inage, and T. Fujii, “Dynamic radio map using statistical hypothesis testing,” IEEE Transactions on Cognitive Communications and Networking, vol. 7, no. 3, pp. 752–766, 2020.

[11] X. Zhu, T. Qiu, W. Qu, X. Zhou, Y. Wang, and D. O. Wu, “Path planning for adaptive csi map construction with a3c in dynamic environments,” IEEE Transactions on Mobile Computing, vol. 22, no. 5, pp. 2925–2937, 2021.

[12] R. D. Yates, Y. Sun, D. R. Brown, S. K. Kaul, E. Modiano, and S. Ulukus, “Age of information: An introduction and survey,” IEEE Journal on Selected Areas in Communications, vol. 39, no. 5, pp. 1183– 1210, 2021.