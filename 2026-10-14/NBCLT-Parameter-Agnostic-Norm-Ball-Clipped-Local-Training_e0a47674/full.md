# NBCLT: Parameter-Agnostic Norm-Ball Clipped Local Training

Linxuan Pan

Shucheng Yang

Junchi Yang

panlinxuan@link.cuhk.edu.cn yangsc@mail.sim.ac.cn yangjunchi@cuhk.edu.cn

## Abstract

Distributed learning with multiple local steps reduces communication but aggravates drift, and its practical performance often depends on carefully tuned hyperparameters under heterogeneous data and communication constraints. At the same time, recent Muon-type optimizers suggest that constraining updates within suitable norm balls provides a simple way to encode geometric priors into optimization. In this work, we propose NBCLT (Norm-Ball Clipped Local Training), a unified local-training framework for both federated learning (FL) and decentralized learning (DL). NBCLT combines a predict-and-correct local update rule with Euclidean projections onto flexible norm balls, thereby performing norm-ball clipping under flexible update geometries while retaining a simple implementation. In FL, its problem-parameter-free regime achieves $\mathcal { O } ( \epsilon ^ { - 2 } )$ communication complexity and $\mathcal { O } ( \epsilon ^ { - 4 } )$ total stochastic-gradient complexity without assuming bounded heterogeneity, when the per-round local-computation budget is scaled appropriately with the horizon. In DL, NBCLT is compatible with both static and time-varying mixing and preserves these accuracy exponents under the analogous computation scaling, up to fixed network constants and an additive initialization cost.

## 1 Introduction

Machine learning has achieved significant progress in domains such as computer vision, natural language processing, and scientific discovery [3, 11, 26, 27, 32]. To address requirements arising from data privacy, system topology, and communication constraints, several distributed learning paradigms have been developed. Federated Learning (FL) studies training over decentralized and typically non-IID data under a central coordinator [14, 21], while Decentralized Learning (DL) removes the coordinator and relies on peer-to-peer communication over a network [17]. Throughout this paper, we use distributed learning to encompass both settings.

Formally, a distributed system comprises M nodes (e.g., clients, workers, or organizations). Node $i \in [ M ]$ holds a local objective $f _ { i } ( x ) = \mathbb { E } _ { \xi _ { i } \sim \mathcal { D } _ { i } } [ \ell ( x ; \xi _ { i } ) ]$ , and the global model is trained by solving

$$
\operatorname* { m i n } _ { x \in \mathcal { X } } \left[ f ( x ) : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { i } ( x ) \right] .\tag{1}
$$

In distributed optimization, two challenges are fundamental. First, communication is often the dominant systems bottleneck, which makes multiple local updates between synchronization rounds highly desirable in both FL [21, 29] and DL [8]. Second, local training amplifies drift caused by data heterogeneity or imperfect consensus. To reduce this drift, FL algorithms use control variates or momentum corrections, as in SCAFFOLD [12] and momentum-based FL methods [4], while DL methods rely on mechanisms such as gradient tracking or Exact Difusion [1, 20]. Yet their guarantees typically require well-tuned hyperparameters that depend on inaccessible quantities such as smoothness, noise, or heterogeneity levels.

Table 1: Comparison in FL (nonconvex). Communication complexity counts server-client rounds; sample complexity counts total stochastic gradients across participating clients. The NBCLT entry uses $N K = \Theta ( T )$ with $N K \leq T$ and fixed problem/system constants, not fixed N, K as accuracy increases.
<table><tr><td>Method</td><td>Unbounded het. Comm. compl. Sample compl. Prob.-param.-free Local steps</td><td></td><td></td><td></td><td></td></tr><tr><td>NBCLT-FL (ours)</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 2 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>√</td><td>√</td></tr><tr><td>PAdaMFed [36]</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 3 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>√</td><td>√</td></tr><tr><td>AdaMasFL [35]</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 8 / 3 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>√</td><td>√</td></tr><tr><td>FedMuon [31]</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 3 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>X</td><td>√</td></tr><tr><td>FedMuon [19]</td><td>X</td><td> $\mathcal { O } ( \epsilon ^ { - 3 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>√</td><td>√</td></tr><tr><td>Diloco [7]</td><td></td><td>− (empirical)</td><td> $\mathrm { ( e m p i r i c a l ) }$ </td><td>X</td><td>√</td></tr></table>

Table 2: Comparison in DL (nonconvex). “Mixing” indicates whether the theory covers static doubly-stochastic matrices or time-varying mixing matrices. The NBCLT entry uses $M K = \Theta ( T )$ with $M K \leq T$ and fixed problem/network constants; exact initialization has an additional networkdependent communication cost.
<table><tr><td>Method</td><td>Mixing</td><td></td><td></td><td></td><td>Unbounded het. Comm. compl. Sample compl. Prob.-param.-free Local steps</td><td></td></tr><tr><td>NBCLT-DL (ours)</td><td>Both</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 2 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>√</td><td>√</td></tr><tr><td>D-NASA [16]</td><td>static</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>√</td><td>X</td></tr><tr><td>LED [1]</td><td>static</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 2 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>X</td><td>√</td></tr><tr><td>GT-FT [23]</td><td>dynamic</td><td>√</td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td> $\mathcal { O } ( \epsilon ^ { - 4 } )$ </td><td>X</td><td>X</td></tr></table>

Recent progress on problem-parameter-free optimization has reduced the burden of manual tuning, but a gap remains in both settings. In FL, methods such as PAdaMFed [36] and AdaMasFL [35] avoid prior knowledge of problem constants, yet their communication guarantees still fall short of the $\mathcal { O } ( \epsilon ^ { - 2 } )$ benchmark achieved by well-tuned local training methods. In DL, the problemparameter-free method D-NASA [16] does not support multiple local updates, which likewise leads to suboptimal communication eficiency. This raises a first question:

## Can a problem-parameter-free local-training algorithm match the $\mathcal { O } ( 1 / \epsilon ^ { 2 } )$ communication complexity of well-tuned methods in both FL and DL, without sacrificing the benefits of multiple local updates?

In addition to tuning the update scale, local training also depends on the geometry used to constrain local updates. We view a local update as lying in a norm budget $\mathcal { D } = \{ \Delta : \| \Delta \| ^ { g } \leq \eta \}$ , where the radius η sets the overall scale of motion and the norm $\| \cdot \| ^ { g }$ determines the shape of admissible updates. This separates how far the algorithm moves from the geometry under which admissible updates are measured; Section 2 makes this distinction concrete through a quadratic example.

This viewpoint is particularly relevant in distributed learning, where many clients may have highly heterogeneous local loss landscapes. Recent Muon-type optimizers make this geometric perspective explicit by constraining updates in carefully chosen norm balls [2, 10, 18, 24, 28]. Several recent works have begun integrating Muon into federated learning, but the current analyses remain limited to $\mathcal { O } ( \epsilon ^ { - 3 } )$ communication complexity [19, 31, 38]. This leads to a second question:

Can we design a distributed local-training algorithm that exploits flexible update geometry while retaining the communication and sample-complexity guarantees expected from eficient local methods?

Motivated by these two questions, we propose NBCLT (Norm-Ball Clipped Local Training). Our main contributions are as follows:

• NBCLT provides a unified framework for both federated learning (FL) and decentralized learning (DL). It introduces two key local-update components: (i) Euclidean projection of tentative update variables onto flexible norm balls, i.e., norm-ball clipping under the chosen update geometry, and (ii) a novel predict-and-correct mechanism, conceptually adjacent to extragradient and optimistic ideas [22, 25], that stabilizes clipped local training. For spectralnorm matrix blocks, we instantiate this oracle by PolyGram-RCClip, a Gram-polynomial clipping projector.

• In $\mathrm { F L } ,$ we prove a rate of order $T ^ { - 1 / 2 } + ( N K T ) ^ { - 1 / 4 }$ plus lower-order terms without assuming bounded heterogeneity. With $N K = \Theta ( T )$ and $N K \leq T$ , this yields $\mathcal { O } ( \epsilon ^ { - 2 } )$ communication and $\mathcal { O } ( \epsilon ^ { - 4 } )$ total stochastic-gradient complexity for fixed problem/system constants.

• In DL, to the best of our knowledge, NBCLT is the first problem-parameter-free algorithm that supports multiple local updates while remaining compatible with both static and time-varying gossip matrices. Under the analogous scaling $M K = \Theta ( T )$ with $M K \leq T$ , its accuracy exponents match those of well-tuned local-update baselines, up to fixed network constants and exact-initialization overhead.

## 2 Preliminaries: Update Geometry under a Norm Budget

This section isolates the update-geometry viewpoint used throughout the paper. A norm budget $\mathcal { D } = \{ \Delta : \| \Delta \| ^ { g } \leq \eta \}$ contains two distinct design choices: the radius $\eta ,$ which sets the overall scale of motion, and the norm $\| \cdot \| ^ { g }$ , which shapes the admissible update directions. We first illustrate this distinction on a quadratic model and then connect it to Muon-type norm-ball updates, which motivate the projection-based clipping used by NBCLT.

## 2.1 A Quadratic Example

Consider the local quadratic model

$$
f ( x + \Delta ) - f ( x ) \approx \langle \nabla f ( x ) , \Delta \rangle + \frac { 1 } { 2 } \langle \Delta , \mathbf { H } ( x ) \Delta \rangle .
$$

Writing $\Delta = \eta u$ with $\| u \| ^ { g } \leq 1$ separates the scale η from the shape of the unit-budget set. To see why the shape matters, consider $q ( z ) = { \textstyle { \frac { 1 } { 2 } } } ( \lambda _ { 1 } z _ { 1 } ^ { 2 } + \lambda _ { 2 } z _ { 2 } ^ { 2 } )$ with $\lambda _ { 1 } \gg \lambda _ { 2 }$ and Hessian $\mathbf { H } = \mathrm { d i a g } ( \lambda _ { 1 } , \lambda _ { 2 } )$ Under the Euclidean budget $\begin{array} { r } { \| u \| _ { 2 } \leq 1 , \operatorname* { s u p } _ { \| u \| _ { 2 } \leq 1 } u ^ { \top } \mathbf { H } u = \lambda _ { 1 } } \end{array}$ , so stability of the local quadratic model is dictated by the stifest direction and forces the radius to scale with $1 / \lambda _ { 1 }$ in the worst case. In contrast, under the weighted norm $\| u \| ^ { g } : = \sqrt { \lambda _ { 1 } u _ { 1 } ^ { 2 } + \lambda _ { 2 } u _ { 2 } ^ { 2 } }$ , we have

$$
\operatorname* { s u p } _ { \| u \| ^ { g } \leq 1 } u ^ { \top } \mathbf { H } u = 1 ,
$$

$$
\operatorname* { s u p } _ { \| u \| ^ { g } \leq 1 } - \langle g , u \rangle = \| g \| ^ { g , * } = \sqrt { \frac { g _ { 1 } ^ { 2 } } { \lambda _ { 1 } } + \frac { g _ { 2 } ^ { 2 } } { \lambda _ { 2 } } } .
$$

Thus the norm absorbs the anisotropy: high-curvature directions become expensive, flatter directions become relatively afordable, and a single radius η need not be dictated solely by the worst direction.

## 2.2 Norm-Ball Update Geometry and Muon-type Motivation

Muon-type optimizers instantiate the same norm-budget viewpoint by selecting directions through a Linear Minimization Oracle (LMO), $\Delta ^ { t } \in$ arg min $\Delta \in \mathcal { D } ^ { \langle m ^ { t } , \Delta \rangle }$ , over a norm ball $\mathcal { D } = \{ \Delta$ $\| \Delta \| ^ { g } \leq \eta \} \ [ 2 , 1 0 , 2 4 ]$ . For example, the Euclidean ball yields normalized momentum directions, while a spectral-norm matrix ball yields the orthogonalized direction used in Muon [10]. The full uSCG template is recalled in Appendix D [24]; here it serves only as motivation for norm balls as update-geometry priors, rather than as the algorithmic primitive of NBCLT.

For deep models, this viewpoint is naturally applied block-wise. If $\mathbf { X } = ( \mathbf { X } ^ { ( 1 ) } , \ldots , \mathbf { X } ^ { ( L ) } )$ and each block carries a norm $\| \cdot \| _ { ( l ) }$ , define

$$
\begin{array} { r l } & { \| \mathbf { X } \| ^ { g } : = \underset { l = 1 , \ldots , L } { \operatorname* { m a x } } \| \mathbf { X } ^ { ( l ) } \| _ { ( l ) } , } \\ & { \mathcal { D } = \{ \Delta \in \mathcal { V } : \| \Delta \| ^ { g } \leq \eta \} . } \end{array}\tag{2}
$$

This covers layer-wise spectral constraints as in Muon [10], heterogeneous layer geometries as in Scion [24], and finer matrix-block partitions such as MuonSBW [2].

NBCLT difers from LMO-based Muon-type methods. It applies Euclidean projection onto the same norm balls, so a tentative local update is modified only when it leaves the prescribed budget set. Thus the norm determines the admissible update geometry, while the algorithm implements this geometry as clipping rather than normalization.

## 3 NBCLT Framework

We now instantiate the norm-budget viewpoint in the distributed objective (1). The system contains M nodes. In FL, a server broadcasts the current model and aggregates updates from participating clients; in DL, nodes replace the server step by peer-to-peer mixing. In both cases, NBCLT performs K local predict-and-correct steps between communication phases, while constraining each local update variable to the norm budget D.

## 3.1 Local Update Design

The core NBCLT update is shared by the FL and DL variants. At communication round t, local step k, and node i, let $g _ { i } ^ { t , k }$ denote the corrected gradient surrogate. NBCLT maintains an auxiliary update variable $\hat { \Delta } _ { i } ^ { t , k }$ and applies

$$
\begin{array} { r l } & { \hat { \Delta } _ { i } ^ { t , k - 1 } = \Pi _ { \mathcal { D } } \Big ( \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { t , k } \Big ) , } \\ & { { \Delta } _ { i } ^ { t , k - 1 } = \Pi _ { \mathcal { D } } \Big ( \hat { \Delta } _ { i } ^ { t , k - 1 } - g _ { i } ^ { t , k } \Big ) , } \\ & { \quad \quad x _ { i } ^ { t , k } = x _ { i } ^ { t , k - 1 } + { \Delta } _ { i } ^ { t , k - 1 } . } \end{array}
$$

The two components of this update are described below.

Local predict-and-correct. The auxiliary variable $\hat { \Delta } _ { i } ^ { t , k - 1 }$ is obtained by taking a projected step from the previous auxiliary update and can be viewed as a prediction of the next update. The actual update $\displaystyle \dot { \Delta _ { i } ^ { t , k - 1 } }$ then corrects this prediction by applying the same gradient surrogate once more before updating the model. This two-step mechanism is the main device that stabilizes clipped local training with multiple local steps. Equivalently, $\hat { \Delta } _ { i } ^ { t , k - 1 }$ acts as a momentum-like update variable that accumulates past gradient information, while $\Delta _ { i } ^ { t , k - 1 }$ combines this history with the current corrected gradient surrogate.

Projection oracle. NBCLT applies Euclidean projection $\begin{array} { r } { \Pi _ { \mathcal { D } } ( z ) : = \arg \operatorname* { m i n } _ { u \in \mathcal { D } } \frac { 1 } { 2 } \| u - z \| _ { 2 } ^ { 2 } } \end{array}$ to tentative update variables, where $\mathcal { D } = \{ \Delta : \| \Delta \| ^ { g } \leq \eta \}$ . Operationally, this is clipping the update under the chosen norm. For the Euclidean ball, the projection has the familiar form

$$
\Pi _ { \mathcal { D } } ( z ) = \left\{ { z , \atop \eta z / \| z \| _ { 2 } , } \right. \| z \| _ { 2 } \leq \eta ,
$$

Matrix geometries fit the same formulation through vectorization. If $Z \in \mathbb { R } ^ { m \times n }$ and $v = \operatorname { v e c } ( Z )$ define the induced vector norm $\| v \| _ { \mathrm { v e c ( o p ) } } : = \| \operatorname { u n v e c } ( v ) \| _ { \mathrm { o p } }$ , where $\| \cdot \| _ { \mathrm { o p } }$ denotes the matrix spectral norm. Since vectorization preserves Frobenius/Euclidean distances, Euclidean projection of v onto the induced vector norm ball is equivalent to Frobenius projection of $Z$ onto $\mathcal { D } _ { \mathrm { o p } } = \{ Y \in \mathbb { R } ^ { m \times n }$ $\| Y \| _ { \mathrm { o p } } \leq \eta \}$ . If $Z = U \mathrm { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { r } ) V ^ { \top }$ , then

$$
\Pi _ { \mathcal { D } _ { \mathrm { o p } } } ( Z ) = U \mathrm { d i a g } \big ( \operatorname* { m i n } \{ \sigma _ { 1 } , \eta \} , \dots , \operatorname* { m i n } \{ \sigma _ { r } , \eta \} \big ) V ^ { \top } .
$$

This difers from LMO-based Muon-type updates [10, 24], which select an update by solving a linear minimization problem over the norm ball. NBCLT instead projects a tentative update and changes it only when it leaves the prescribed budget. Plain stochastic gradient descent with gradient clipping can fail to converge without a careful coupling of the clipping threshold and the stepsize [13]; the predict-and-correct mechanism above yields the convergence guarantees proved in Section 4.

## 3.2 NBCLT-FL

NBCLT-FL is designed for federated learning and is summarized in Algorithm 1. The method supports client sampling: at each communication round t, the server samples a subset of clients $S _ { t }$ and broadcasts the current global model $x ^ { t }$ to the selected clients. Each participating client then performs K local update steps using the predict-and-correct mechanism described above. After completing the local updates, clients transmit their accumulated model updates $\Delta _ { i } ^ { t }$ to the server, which aggregates them to form the next global model $x ^ { t + 1 }$

To mitigate client drift, NBCLT-FL incorporates control variates in the spirit of SCAFFOLD [12]. Specifically, each client maintains local control variates $\boldsymbol { \hat { c } } _ { i } ^ { t }$ and $c _ { i } ^ { t }$ , while the server maintains a global control variate $m ^ { t }$ . We introduce two key modifications to the standard SCAFFOLD design: a shrinkage parameter $\beta _ { 1 }$ in the calibrated gradient surrogate and an exponential moving average with parameter $\beta _ { 2 }$ for the local control variate,

$$
\begin{array} { r l } & { ~ g ^ { k } \gets ( 1 - \beta _ { 1 } ) \big ( \nabla f ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \big ) + m ^ { t } , } \\ & { ~ c _ { i } ^ { t + 1 } \gets \beta _ { 2 } c _ { i } ^ { t } + ( 1 - \beta _ { 2 } ) \hat { c } _ { i } ^ { t } . } \end{array}
$$

```latex
Algorithm 1 NBCLT-FL
1: Input: Rounds T, clients M, local steps $K$ , update-budget radius $\eta ,$ subset size $\overline { { N } } = | S _ { t } |$
domain $\mathcal { D } ,$ hyperparameters $\beta _ { 1 } , \beta _ { 2 }$
2: Initialize: $\begin{array} { r } { \tilde { X ^ { 1 } } \in \mathsf { \bar R } ^ { d } , \{ c _ { i } ^ { 1 } = \nabla f _ { i } ( X ^ { 1 } , \xi _ { i } ^ { - 1 } ) \} _ { i = 1 } ^ { M } , m ^ { 1 } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } c _ { i } ^ { 1 } } \end{array}$
3: for $t = 1 , \dots , T$ do
4: Server selects subset $S _ { t } \subset [ M ]$ uniformly at random.
5: Server broadcasts $x ^ { t }$ and $m ^ { t }$ to clients in $S _ { t } .$
6: Client Execution (in parallel for $\boldsymbol { i } \in S _ { t } )$
7: Initialize $\hat { \Delta } _ { i } ^ { t , - 1 } \gets 0 , \ : \dot { x } _ { i } ^ { t , 0 ^ { * } } \gets x ^ { t } .$
8: for $k = 1 , \ldots , K$ do
9: $g ^ { k } \gets ( \bar { 1 } - \beta _ { 1 } ) ( \nabla f ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } ) + m ^ { t }$
10: $\hat { \Delta } _ { i } ^ { t , k - 1 } \gets \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 2 } - g ^ { k } )$
11: $\Delta _ { i } ^ { t , k - 1 } \gets \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 1 } - g ^ { k } )$
12: $x _ { i } ^ { t , k } \gets x _ { i } ^ { t , k - 1 } + \bar { \Delta } _ { i } ^ { t , k - 1 }$
13: end for
14: $\begin{array} { r } { \hat { c } _ { i } ^ { t } \gets \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) } \end{array}$
15: $c _ { i } ^ { t + 1 } \longleftarrow \beta _ { 2 } c _ { i } ^ { t } + ( 1 - \beta _ { 2 } ) \hat { c } _ { i } ^ { t }$
16: Send $\Delta _ { i } ^ { t } = \dot { x } _ { i } ^ { t , K } - x _ { i } ^ { t , 0 }$ and $c _ { i } ^ { t + 1 }$ to server.
17: For each client i /∈ $\begin{array} { r l } { S _ { t } \colon } & { { } \ \dot { c } _ { i } ^ { t + 1 } \gets c _ { i } ^ { t } . } \end{array}$
18: Server Aggregation:
19: Average local models: $\begin{array} { r } { x ^ { t + 1 } \gets x ^ { t } + \frac { 1 } { N } \sum _ { i \in S _ { t } } \Delta _ { i } ^ { t } . } \end{array}$
20: Update global control: $\begin{array} { r } { m ^ { t + 1 } \gets \frac { 1 } { M } \sum _ { i = 1 } ^ { M } c _ { i } ^ { t + 1 } } \end{array}$
21: end for
```

## 3.3 NBCLT-DL

NBCLT-DL is designed for decentralized learning over peer-to-peer networks, where nodes communicate directly with their neighbors, and is summarized in Algorithm 2. Unlike classical gradienttracking methods [34], NBCLT-DL follows a design philosophy similar to NBCLT-FL: each node maintains local control variates $\boldsymbol { \hat { c } } _ { i } ^ { t }$ and $c _ { i } ^ { t } .$ , along with a local estimate of the global control variate, denoted by $m _ { i } ^ { t } .$

At each communication round, every node first performs K local update steps. This is followed by a communication phase. In the absence of a central coordinator, local models and control variates cannot be directly averaged. Instead, nodes exchange information with their neighbors according to a mixing matrix $\mathbf { W } ^ { t } = [ w _ { i j } ^ { t } ]$ , which encodes the network topology. The local models are updated via neighbor averaging, $\begin{array} { r } { x _ { i } ^ { t + \bar { 1 } }  \sum _ { j = 1 } ^ { M } w _ { i j } ^ { t } x _ { j } ^ { t , K } } \end{array}$ . For the update of the local estimate $m _ { i } ^ { t } ,$ each node first incorporates its newly updated local control variate and then performs neighbor mixing, $\begin{array} { r } { m _ { i } ^ { t + 1 } \gets \sum _ { j = 1 } ^ { M ^ { - } } w _ { i j } ^ { t } ( m _ { j } ^ { t } + c _ { j } ^ { t + 1 } - c _ { j } ^ { \bar { t } } ) } \end{array}$ . This ensures that the local estimates $m _ { i } ^ { t }$ progressively approximate the global gradient, thereby reducing discrepancies among local models over time.

## 4 Convergence Guarantees

Before presenting the convergence for NBCLT, we first state the underlying assumptions. We assume that each local objective function $f _ { i }$ is Lipschitz smooth, but not necessarily convex. For each $f _ { i } ,$ we have access to an unbiased stochastic gradient oracle $\nabla f _ { i } ( \cdot , \xi )$

Algorithm 2 NBCLT-DL   
1: Input: Rounds ${ \overline { { \mathbf { \Omega } _ { T , } } } }$ clients M, local steps K, mixing matrix W, domain ${ \mathcal { D } } ,$ hyperparameters   
$\beta _ { 1 } , \beta _ { 2 } .$   
2: Initialize: $x _ { i } ^ { 1 } = x _ { \mathrm { i n i t } }$ and $c _ { i } ^ { 1 } = \nabla f _ { i } ( x _ { \mathrm { i n i t } } , \xi _ { i } ^ { - 1 } )$ , using one stochastic-gradient query per node.   
3: Exactly average the $c _ { i } ^ { 1 } \mathrm { { ^ { s } } }$ using the initialization in Appendix B, and set $\begin{array} { r } { m _ { i } ^ { 1 } = { M ^ { - 1 } } { \Sigma } _ { j = 1 } ^ { M } c _ { j } ^ { 1 } } \end{array}$ at   
every node.   
4: for $t = 1 , \dots , T$ do   
5: for all node $i \in [ M ]$ in parallel do   
6: Initialize $\hat { \Delta } _ { i } ^ { t , - 1 } \gets 0 .$   
7: for $k = 1 , \ldots , K$ do   
8: $g _ { i } ^ { k } \gets ( 1 - \beta _ { 1 } ) ( \nabla f ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } ) + m _ { i } ^ { t }$   
9: $\begin{array} { r } { \hat { \Delta } _ { i } ^ { t , k - 1 } \gets \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { k } ) } \end{array}$   
10: $\Delta _ { i } ^ { t , k - 1 } \gets \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 1 } - g _ { i } ^ { k } )$   
11: $x _ { i } ^ { t , k } \gets x _ { i } ^ { t , k - 1 } + \bar { \Delta } _ { i } ^ { t , k - 1 }$   
12: end for   
13: $\begin{array} { r } { \hat { c } _ { i } ^ { t } \gets \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) } \end{array}$   
14: $c _ { i } ^ { t + 1 } \longleftarrow \beta _ { 2 } c _ { i } ^ { t } + ( 1 - \beta _ { 2 } ) \hat { c } _ { i } ^ { t }$   
15: Communication:   
16: Mix models: $\begin{array} { r } { x _ { i } ^ { t + 1 }  \sum _ { j = 1 } ^ { M } w _ { i j } ^ { t } x _ { j } ^ { t , K } } \end{array}$   
17: Update tracker: $m _ { i } ^ { t + 1 } \gets \sum _ { j = 1 } ^ { M ^ { ^ { \prime } } } \tilde { w } _ { i j } ^ { t } ( m _ { j } ^ { t } + c _ { j } ^ { t + 1 } - c _ { j } ^ { t } )$   
18: end for   
19: end for

Assumption 1 (Smoothness). Each $f _ { i }$ is L-smooth: $\| \nabla f _ { i } ( x ) - \nabla f _ { i } ( y ) \| _ { 2 } \leq L \| x - y \| _ { 2 }$ for all $x , y \in \mathbb { R } ^ { d }$

Assumption 2 (Stochastic gradients). For each client $i ,$ the stochastic gradient $\nabla f _ { i } ( x , \xi )$ is conditionally unbiased and has bounded conditional variance:

$$
\mathbb { E } [ \nabla f _ { i } ( x , \xi ) \mid x ] = \nabla f _ { i } ( x ) ,
$$

$$
\mathbb { E } [ \| \nabla f _ { i } ( x , \xi ) - \nabla f _ { i } ( x ) \| _ { 2 } ^ { 2 } \mid x ] \le \sigma ^ { 2 } .
$$

Moreover, conditioned on the past, sampling noises are independent across gradient queries.

Assumption 3 (Lower boundedness). The global objective $\begin{array} { r } { F ( \boldsymbol { x } ) : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { i } ( \boldsymbol { x } ) } \end{array}$ is lower bounded: $\forall x \in \mathbb { R } ^ { d } , F ( x ) \geq F ^ { \star }$

Assumption 4 (Norm equivalence and dual norm). The projection domain is a norm ball $\mathcal { D } = \{ \Delta$ $\| \Delta \| ^ { g } \leq \eta \}$ with respect to some norm $\| \cdot \| ^ { g }$ . There exist constants $\rho , \bar { \rho } \geq 1$ such that $\| v \| ^ { g } \leq \rho \| v \| _ { 2 }$ and $\| v \| _ { 2 } \leq { \bar { \rho } } \| v \| ^ { g }$ for all $v \in \mathbb { R } ^ { d }$ . The dual norm is $\| v \| ^ { g , * } : = \operatorname* { s u p } _ { \| u \| ^ { g } \leq 1 } \langle v , u \rangle$ . Consequently, $\| \cdot \| ^ { g , * }$ and $\| \cdot \| _ { 2 }$ are comparable up to constants: $\rho ^ { - 1 } \| v \| _ { 2 } \leq \| v \| ^ { g , * } \leq \bar { \rho } \| v \| _ { 2 }$ for all v.

In practice, we adopt the composite norm described in Section 2. This assumption on the constraint set D can be relaxed to a closed and convex set that contains a norm ball.

## 4.1 Convergence of NBCLT-FL

The FL analysis admits two complementary regimes. The main text focuses on the problemparameter-free instantiation, where all algorithmic parameters are chosen from known system quantities such as $( N , K , T )$ . This is a fixed-horizon guarantee: $T$ is specified in advance, while unknown problem constants are not needed. When problem-dependent quantities such as $\Delta _ { F } , L ,$ and σ are available, the same algorithm also admits a well-tuned guarantee with a fixed smoothing parameter and no coupling condition among N, K, and $T ;$ we defer this complementary result to Appendix A.1. We now state the main problem-parameter-free guarantee.

Theorem 5 (Problem-parameter-free convergence of NBCLT-FL). Suppose Assumptions $1 \mathrm { ~ - ~ } \not \llap { / } 4$ hold. Run Algorithm 1 for T communication rounds, with K local updates per round and subset size N (out of M total clients). Choose the hyperparameters as $\eta : = N ^ { 1 / 4 } / ( K ^ { 3 / 4 } T ^ { 3 / 4 } ) , 1 - \beta _ { 1 } : = \eta$ and $1 - \beta _ { 2 } : = \sqrt { N K / ( T + K ) }$ . Define $\Delta _ { F } : = F ( x ^ { 1 } ) - F ^ { \star }$ and $p : = N / M$ . Then for all $T \geq K$ such that $N K \leq T$ , we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ \| \nabla F ( x ^ { t } ) \| ^ { g , * } \Big ] \leq \ \frac { C _ { 1 } } { \sqrt { T } } + \frac { C _ { 2 } } { ( N K T ) ^ { 1 / 4 } } + \ C _ { 0 } R _ { \mathrm { l o } } ( T , K ) ,\tag{3}
$$

where

$$
C _ { 1 } : = \frac { 5 L \bar { \rho } ^ { 2 } } { 2 } , \qquad C _ { 2 } : = \Delta _ { F } + \frac { 6 4 L \bar { \rho } ^ { 2 } } { p } + 3 2 \sigma \bar { \rho } .
$$

The lower-order remainder term is

$$
\begin{array} { l } { { \displaystyle R _ { \mathrm { l o } } ( T , K ) = \frac { N ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } + \frac { N ^ { 1 / 2 } } { K ^ { 1 / 2 } T ^ { 3 / 2 } } \ ~ } } \\ { { \displaystyle ~ + \frac { N ^ { 3 / 4 } } { K ^ { 9 / 4 } T ^ { 9 / 4 } } + \frac { 1 } { N ^ { 1 / 4 } K ^ { 5 / 4 } T ^ { 3 / 4 } } \ ~ } } \\ { { \displaystyle ~ + \frac { 1 } { \sqrt { N K T } } , } } \end{array}
$$

and the constant $C _ { 0 }$ depends only on fixed problem parameters $( e . g . , L , \sigma , \bar { \rho } , p )$ , but not on $( T , K )$

Theorem 5 is problem-parameter-free: the update-budget radius and smoothing parameters depend only on known system quantities $( N , K , T )$ , not on $L , \sigma$ , or $\Delta _ { F }$ . Its leading terms are $\mathcal { O } ( T ^ { - 1 / 2 } )$ and $\mathcal { O } ( ( N K T ) ^ { - 1 / 4 } )$ , while $R _ { \mathrm { l o } }$ is lower order. Thus, taking $T = \Theta ( \epsilon ^ { - 2 } )$ and choosing $N K = \Theta ( T )$ within $N K \leq T$ yields $\mathcal { O } ( \epsilon ^ { - 2 } )$ communication rounds and $\mathcal { O } ( \epsilon ^ { - 4 } )$ total stochasticgradient queries, including the additive M initialization queries for a fixed client population. This communication guarantee trades additional local computation for fewer synchronization rounds; $N K \leq T$ alone does not imply it. With N, K fixed, the bound instead gives $\mathcal { O } ( \epsilon ^ { - 4 } )$ rounds. Constants, including their dependence on $p ,$ are held fixed in these accuracy exponents. The participation size N improves the second leading term through both the denominator $( N K T ) ^ { 1 / 4 }$ and the participation ratio $p = N / M$ appearing in $C _ { 2 } ;$ in the full-participation case $( N = M )$ , the leading stochastic term scales as $\dot { \Theta } ( ( M K T ) ^ { - 1 / 4 } )$ up to problem-dependent constants.

## 4.2 Efect of Inexact Projection

The preceding FL guarantee assumes exact Euclidean projection onto D. The following theorem shows that NBCLT-FL retains its convergence rates with feasible inexact projections, provided the accumulated projection residual is suficiently small.

Theorem 6 (Convergence of NBCLT-FL with inexact projections). Suppose Assumptions $1 \mathrm { - } \mathit { 4 }$ hold. Run NBCLT-FL with the same hyperparameter choices as in either Theorem 5 or the well-tuned FL guarantee in Theorem A.1, but replace the exact Euclidean projection calls in Algorithm 1 by

feasible inexact projection calls. For a projection subproblem with input z and feasible output $y \in \mathcal { D }$ define the single-call projection residual

$$
\mathrm { r e s } _ { \mathcal { D } } ( z , y ) : = \biggl [ \underset { u \in \mathcal { D } } { \operatorname* { s u p } } \langle z - y , u - y \rangle \biggr ] _ { + } ,\tag{4}
$$

$$
= [ \eta \| z - y \| ^ { g , * } - \langle z - y , y \rangle ] _ { + } .\tag{5}
$$

Here $\mathcal { D } = \{ u : \| u \| ^ { g } \leq \eta \}$ . For the prediction and correction projection calls at round $t ,$ client $i ,$ and local step k, let

$$
\begin{array} { r l } & { \hat { z } _ { i } ^ { t , k } : = \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { t , k } , } \\ & { z _ { i } ^ { t , k } : = \hat { \Delta } _ { i } ^ { t , k - 1 } - g _ { i } ^ { t , k } , } \end{array}
$$

and let $\hat { \Delta } _ { i } ^ { t , k - 1 } , \Delta _ { i } ^ { t , k - 1 } \in \mathcal { D }$ be the corresponding feasible outputs. Write $\hat { r } _ { i } ^ { t , k } : = \mathrm { r e s } _ { \mathcal { D } } ( \hat { z } _ { i } ^ { t , k } , \hat { \Delta } _ { i } ^ { t , k - 1 } )$ and $r _ { i } ^ { t , k } : = \mathrm { r e s } _ { \mathcal { D } } ( z _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } )$ . Define the averaged accumulated projection residual

$$
\overline { { \mathfrak { R } } } _ { T } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \left( \widehat { r } _ { i } ^ { t , k } + r _ { i } ^ { t , k } \right) \right] .\tag{6}
$$

Then the corresponding exact-projection FL convergence bounds remain valid, up to enlarged numerical constants with the same dependence on problem parameters, with only an additional perturbation term ${ \mathcal O } \big ( \overline { { \mathfrak { R } } } _ { T } / ( \eta K ) \big )$ on the right-hand side. Consequently, if $\overline { { \mathfrak { R } } } _ { T } = { \mathcal { O } } ( T ^ { - 1 } )$ , then the inexactprojection implementation preserves the communication and sample-complexity exponents of both the problem-parameter-free and well-tuned FL regimes. When this residual budget is distributed uniformly over the K local steps, the natural single-call target is of order $1 / ( K T )$

Theorem 6 turns exact projection into a residual-budget condition rather than a mandatory oracle. For spectral-norm matrix blocks, this motivates an SVD-free clipping oracle for $B _ { \tau } = \{ Y : \| Y \| _ { 2 } \leq \tau \}$ Let $Z \in \mathbb { R } ^ { m \times n }$ be a matrix update block and assume m $\geq n$ for notation. Choose any safe scale $s \geq \| Z \| _ { 2 }$ and block radius $\tau ;$ this scale is easy to obtain, since $s = \| Z \| _ { F }$ already sufices, while our certified implementation uses a deterministic tighter upper bound described in Appendix C.4. Set $G : = Z ^ { \top } Z / s ^ { 2 }$ and $\alpha _ { \mathrm { c l i p } } : = \tau / s$ . In the nontrivial case $\tau < \| Z \| _ { 2 } \leq s$ , PolyGram-RCClip outputs

$$
Y _ { d , s } : = Z p _ { d , \alpha _ { \mathrm { c l i p } } } ( G ) ,
$$

where the degree-d transfer $p _ { d , \alpha _ { \mathrm { c l i p } } }$ is selected by the clipping ratio; the case $m < n$ uses the left Gram matrix and is given in Algorithm C.1. Appendix C.2 proves that for every $\alpha _ { \mathrm { c l i p } } \in ( 0 , 1 )$ there exists a feasible degree-d transfer such that, for all suficiently large d,

$$
\mathrm { r e s } _ { { \mathscr { B } } _ { \tau } } ( Z , Y _ { d , s } ) \leq \frac { C _ { J } } { \alpha _ { \mathrm { c l i p } } ^ { 2 } d } \bigl ( \tau \| Z \| _ { * } + 2 \| Z \| _ { F } ^ { 2 } \bigr ) ,
$$

where $C _ { J }$ is a universal Jackson-type approximation constant. This bound exposes an accuracy– compute tradeof: for fixed $\alpha _ { \mathrm { c l i p } }$ , increasing d lowers the certified residual upper bound as $1 / d ,$ while Algorithm C.1 uses exactly d sequential multiplications by the smaller Gram matrix after it is formed. For fixed d and $\alpha _ { \mathrm { c l i p } }$ , Appendix C.3 formulates coeficient search as a semi-infinite LP; after normalization, it depends only on $\alpha _ { \mathrm { c l i p } } .$ not on $Z , \tau ,$ s separately. Hence the runtime oracle uses an ofline one-dimensional table: for $d = 2 0$ and 1000 grid points, it stores only 21,000 coeficients, below 0.2 MB in float64. Each call computes s, forms $\alpha _ { \mathrm { c l i p } }$ , looks up coeficients, and evaluates the Gram polynomial. Newton–Schulz-5 Muon post-processing instead uses three matrix products per iteration and five iterations in the reference implementation, and it orthogonalizes rather than clips [10].

The degree–residual bound is an existence guarantee for a suficiently accurate, feasible polynomial on each call. Our fixed-degree table is a practical approximation: its safe scale does not by itself establish the accumulated residual budget required by Theorem 6. Appendix C.4 separates these requirements from the experimental implementation.

## 4.3 Convergence of NBCLT-DL

We analyze NBCLT-DL under two network models: a static symmetric doubly stochastic mixing matrix, controlled by its spectral gap, and time-varying doubly stochastic mixing matrices satisfying finite-time consensus $\mathrm { ( F T C ) }$ . For both topologies, we use the same problem-parameter-free choice $\eta = M ^ { 1 / 4 } / ( K ^ { 3 / 4 } T ^ { 3 / 4 } ) , 1 - \beta _ { 1 } = \eta ,$ , and $1 - \beta _ { 2 } = \sqrt { M K / ( T + K ) }$ . The formal statements are given in Theorems B.1 and B.2 in Appendix B.1; the proofs appear later in Appendix B. For $T \geq M K$ both results imply

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \leq { \cal O } ( T ^ { - 1 / 2 } ) + { \cal O } \big ( ( M K T ) ^ { - 1 / 4 } \big ) } \\ & { \quad \quad \quad \quad + \mathrm { l o w e r - o r d e r ~ t e r m s } , } \end{array}\tag{7}
$$

where the hidden constants may depend on the network parameter and fixed problem constants, but not on $( T , K )$ . Consequently, choosing $T = \Theta ( \epsilon ^ { - 2 } )$ and $M K = \Theta ( T )$ within $M K \leq T$ gives $\mathcal { O } ( \epsilon ^ { - 2 } )$ communication rounds and $\mathcal { O } ( \epsilon ^ { - 4 } )$ total stochastic-gradient queries for fixed problem and network constants. For fixed $M , K$ , the bound instead gives $\mathcal { O } ( \epsilon ^ { - 4 } )$ rounds. Exact tracker initialization adds 2h rounds on a precomputed static spanning tree of depth $h ,$ or $\tau$ rounds under $\mathrm { F T C } ;$ the total oracle count is $M ( K T + 1 )$ , as detailed in Appendix B.

## 4.3.1 Static doubly stochastic mixing

Assume the mixing matrix $\mathbf { W } \in \mathbb { R } ^ { M \times M }$ is fixed, symmetric, and doubly stochastic: $\mathbf { W 1 } = \mathbf { 1 }$ and $\mathbf { 1 } ^ { \top } \mathbf { W } = \mathbf { 1 } ^ { \top }$ . Let $\mathbf { J } : = { \cal M } ^ { - 1 } \mathbf { 1 1 } ^ { \top }$ and define $\lambda : = \| \mathbf { W } - \mathbf { J } \| _ { 2 } \in [ 0 , 1 )$ , equivalently the second-largest eigenvalue magnitude of W. The spectral gap $1 - \lambda$ measures the contraction of disagreement under one mixing step. Theorem B.1 gives the complete static-gossip guarantee, including the explicit constants and lower-order terms; the constants depend on the static-network parameter through $1 / ( 1 - \lambda )$ , but the polynomial dependence on $( T , K , M )$ is the one summarized in (7).

## 4.3.2 Time-varying mixing with finite-time consensus

Static sparse topologies can have deteriorating spectral gaps as the network grows; for example, a ring has contraction factor $1 - \mathcal { O } ( M ^ { - 2 } )$ , equivalently spectral gap $\mathcal { O } ( M ^ { - 2 } )$ [30], while faster static constructions such as the exponential graph improve the rate at the cost of increasing node degree [37]. Time-varying communication schemes avoid part of this tradeof by scheduling sparse mixing patterns over time while keeping the per-round communication degree small [23, 37]. We assume the FTC property: there exists an integer $\tau \geq 1$ such that, for every $t ,$

$$
\mathbf { W } ^ { t + \tau - 1 } \mathbf { W } ^ { t + \tau - 2 } \cdot \cdot \cdot \mathbf { W } ^ { t } = \mathbf { J } , \qquad \mathbf { J } : = M ^ { - 1 } \mathbf { 1 } \mathbf { 1 } ^ { \top } .\tag{8}
$$

We refer to $\tau$ as the FTC horizon. Theorem B.2 gives the complete FTC guarantee, including the initialization condition used to control the initial tracker error. Its constants depend on τ, while the communication and sample-complexity exponents remain those in (7).

![](images/6b574dfae02308894bf33a157f0ad9af3108edcdaeaafde174849ddb6b72582b.jpg)  
Figure 1: Main CIFAR-10 FL comparison. Best baseline is the best tuned run among the five baselines, selected by average test accuracy over the last 100 rounds; the full nine-curve comparison appears in Appendix E.

## 5 Experiments

Setup. We evaluate NBCLT-FL on CIFAR-10 [15] under Dirichlet label-skew non-IID partitions. The 50k training examples are split across $M = 1 0 0$ clients with approximately balanced client sizes, and the heterogeneity level is controlled by $\alpha _ { \mathrm { d i r } } \in \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ , where smaller values indicate stronger label skew. Each round samples $N = 1 0$ clients, each selected client performs $K = 5$ local steps with mini-batch size 50, and training runs for $T = 1 0 0 0$ communication rounds. We use a CIFAR-style ResNet-18 with GELU activations and GroupNorm. We compare against FedAvg [21], SCAFFOLD-M [4], FedAMS [33], FedMuon [31], and PAdaMFed [36], each tuned over its own hyperparameter grid. For FedMuon, we use exact orthogonalization for the LMO-based layers rather than a Newton–Schulz approximation.

We report four NBCLT variants: exact-SVD tuned/PF (SVD-T/SVD-PF) and PolyGram-RCClip tuned/PF (Poly-T/Poly-PF). For PF, the theorem-suggested choice gives $\eta \approx 0 . 0 0 2 9 9$ $\beta _ { 1 } \approx 0 . 9 9 7$ , and $\beta _ { 2 } \approx 0 . 7 7 7$ for $N = 1 0 , K = 5 , T = 1 0 0 0$ . Additional implementation details, hyperparameter grids, and the full nine-curve comparison are deferred to Appendix E.

Figure 1 shows that tuned NBCLT-FL improves over the strongest tuned baseline across all three heterogeneity levels. For $\alpha _ { \mathrm { d i r } } = 0 . 3$ and 0.5, the degree-20 PolyGram-RCClip curves remain close to their exact-SVD counterparts in both the tuned and PF regimes.

The most heterogeneous split, $\alpha _ { \mathrm { d i r } } = 0 . 1$ , is more sensitive to the projector approximation, where the gap between Poly-T and SVD-T is larger. Figure 2 is an empirical degree ablation: increasing the polynomial degree reduces the observed gap to the exact-clipping reference in this setting. It does not measure the projection residual or verify the $1 / d$ residual bound.

![](images/f77efa35a0acca5e1257d3c006f2695cdd4481e8a6762e7a88e951dc2be5a121.jpg)  
Figure 2: Projector-degree ablation on $\alpha _ { \mathrm { d i r } } = 0 . 1 ; d = \infty$ denotes exact spectral clipping.

## AI Use Statement

Generative AI tools were used to assist with manuscript editing, literature searches, and discussion and revision of mathematical arguments. The authors remain responsible for all claims, proofs, experiments, and citations.

## References

[1] Sulaiman A Alghunaim. Local exact-difusion for decentralized optimization and learning. IEEE Transactions on Automatic Control, 69(11):7371–7386, 2024.

[2] Valentyn Boreiko, Zhiqi Bu, and Sheng Zha. Towards understanding of orthogonalization in muon. In High-dimensional Learning Dynamics 2025.

[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[4] Ziheng Cheng, Xinmeng Huang, Pengfei Wu, and Kun Yuan. Momentum benefits non-iid federated learning simply and provably. arXiv preprint arXiv:2306.16504, 2023.

[5] Ronald A. DeVore and George G. Lorentz. Constructive Approximation, volume 303 of Grundlehren der mathematischen Wissenschaften. Springer, 1993.

[6] Zeev Ditzian and Vilmos Totik. Moduli of Smoothness, volume 9 of Springer Series in Computational Mathematics. Springer, 1987.

[7] Arthur Douillard, Qixuan Feng, Andrei Alex Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, MarcAurelio Ranzato, Arthur Szlam, and Jiajun Shen. Diloco: Distributed lowcommunication training of language models. In 2nd Workshop on Advancing Neural Network Training: Computational Eficiency, Scalability, and Resource Optimization (WANT@ICML 2024), 2024. URL https://openreview.net/forum?id=pICSfWkJIk.

[8] Abolfazl Hashemi, Anish Acharya, Rudrajit Das, Haris Vikalo, Sujay Sanghavi, and Inderjit Dhillon. On the benefits of multiple gossip steps in communication-constrained decentralized optimization. arXiv preprint arXiv:2011.10643, 2020.

[9] Dunham Jackson. On approximation by trigonometric sums and polynomials. Transactions of the American Mathematical Society, 13(4):491–515, 1912.

[10] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github.io/posts/muon/.

[11] John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. nature, 596(7873):583–589, 2021.

[12] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scafold: Stochastic controlled averaging for federated learning. In International conference on machine learning, pages 5132–5143. PMLR, 2020.

[13] Anastasia Koloskova, Hadrien Hendrikx, and Sebastian U Stich. Revisiting gradient clipping: Stochastic bias and tight convergence guarantees. In International Conference on Machine Learning, pages 17343–17363. PMLR, 2023.

[14] Jakub Konečn\`y, H Brendan McMahan, Daniel Ramage, and Peter Richtárik. Federated optimization: Distributed machine learning for on-device intelligence. arXiv preprint arXiv:1610.02527, 2016.

[15] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[16] Jiaxiang Li, Xuxing Chen, Shiqian Ma, and Mingyi Hong. Problem-parameter-free decentralized nonconvex stochastic optimization. arXiv preprint arXiv:2402.08821, 2024.

[17] Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. Advances in neural information processing systems, 30, 2017.

[18] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025.

[19] Junkang Liu, Fanhua Shang, Junchao Zhou, Hongying Liu, Yuanyuan Liu, and Jin Liu. Fedmuon: Accelerating federated learning with matrix orthogonalization. arXiv preprint arXiv:2510.27403, 2025.

[20] Yue Liu, Tao Lin, Anastasia Koloskova, and Sebastian U Stich. Decentralized gradient tracking with local steps. Optimization Methods and Software, 40(5):1153–1180, 2025.

[21] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-eficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273–1282. PMLR, 2017.

[22] Arkadi Nemirovski. Prox-method with rate of convergence o (1/t) for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 15(1):229–251, 2004.

[23] Edward Duc Hien Nguyen, Xin Jiang, Bicheng Ying, and César A Uribe. On graphs with finite-time consensus and their use in gradient tracking. SIAM Journal on Optimization, 35(2): 872–898, 2025.

[24] Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529, 2025.

[25] Sasha Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. Advances in Neural Information Processing Systems, 26, 2013.

[26] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.

[27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[28] Ishaan Shah, Anthony M Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, et al. Practical eficiency of muon for pretraining. arXiv preprint arXiv:2505.02222, 2025.

[29] Sebastian U Stich. Local sgd converges fast and communicates little. arXiv preprint arXiv:1805.09767, 2018.

[30] Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, and Makoto Yamada. Beyond exponential graph: Communication-eficient topologies for decentralized learning via finite-time convergence. Advances in Neural Information Processing Systems, 36:76692–76717, 2023.

[31] Yuki Takezawa, Anastasia Koloskova, Xiaowen Jiang, and Sebastian U Stich. Fedmuon: Federated learning with bias-corrected lmo-based optimization. arXiv preprint arXiv:2509.26337, 2025.

[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[33] Yujia Wang, Lu Lin, and Jinghui Chen. Communication-eficient adaptive federated learning. In Proceedings of the 39th International Conference on Machine Learning, pages 22802–22838. PMLR, 2022.

[34] Jinming Xu, Ye Tian, Ying Sun, and Gesualdo Scutari. Accelerated primal-dual algorithms for distributed smooth convex optimization over networks. In International Conference on Artificial Intelligence and Statistics, pages 2381–2391. PMLR, 2020.

[35] Wenjing Yan, Xiangyu Zhong, Xiaolu Wang, and Ying-Jun Angela Zhang. Momentum-driven adaptivity: Towards tuning-free asynchronous federated learning. In Forty-second International Conference on Machine Learning.

[36] Wenjing Yan, Kai Zhang, Xiaolu Wang, and Xuanyu Cao. Problem-parameter-free federated learning. In The Thirteenth International Conference on Learning Representations, 2025.

[37] Bicheng Ying, Kun Yuan, Yiming Chen, Hanbin Hu, Pan Pan, and Wotao Yin. Exponential graph is provably eficient for decentralized deep training. Advances in Neural Information Processing Systems, 34:13975–13987, 2021.

[38] Xinwen Zhang and Hongchang Gao. On provable benefits of muon in federated learning. arXiv preprint arXiv:2510.03866, 2025.

## A Federated-learning analysis

## A.1 Well-tuned FL guarantee

Theorem A.1 (Well-tuned convergence of NBCLT-FL). Suppose Assumptions $1 \mathrm { - } \mathit { 4 }$ hold. Run Algorithm 1 for T communication rounds, with K local updates per round and subset size N (out of M total clients). Define $\Delta _ { F } : = F ( x ^ { 1 } ) - F ^ { \star }$ and $p : = N / M$ . Fix any constant $\gamma \in ( 0 , 1 ]$ and choose

$$
\begin{array} { r } { 1 - \beta _ { 2 } : = \gamma , \qquad 1 - \beta _ { 1 } : = \eta . } \end{array}
$$

Let

$$
A _ { \mathrm { w t } } ( \gamma ) : = \left( \frac { 1 1 } { 2 } + \frac { 3 2 \sqrt { 2 } } { p \gamma } \right) L \bar { \rho } ^ { 2 } K + \left( \frac { 5 } { 2 } + L ^ { 2 } \right) \bar { \rho } ^ { 2 } + 2 \sigma ^ { 2 } + 2 \bar { \rho } \sigma + \frac { 8 \sqrt { 2 } \bar { \rho } \sigma \sqrt { \gamma } } { \sqrt { K p } } .
$$

Choose the update-budget radius as

$$
\eta : = \mathrm { m i n } \left\{ 1 , \sqrt { \frac { \Delta _ { F } } { A _ { \mathrm { w t } } ( \gamma ) K T } } \right\} .
$$

Then for any $T \geq 1$ , we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ \| \nabla F ( x ^ { t } ) \| ^ { g , * } \Big ] \ \leq \ 2 \sqrt { \frac { A _ { \mathrm { w t } } ( \gamma ) \Delta _ { F } } { K T } } + \frac { \Delta _ { F } } { K T } + \frac { 1 6 \bar { \rho } \sigma } { p \gamma T } + \frac { 8 \bar { \rho } \sigma } { \sqrt { N \gamma T } } + \frac { 8 \bar { \rho } \sigma \sqrt { \gamma } } { \sqrt { K M } } .\tag{A.1}
$$

This result is not problem-parameter-free, since its update-budget radius depends on $\Delta _ { F }$ , L, $\sigma ,$ and $p .$ . Its role is complementary: when these quantities are available, NBCLT-FL admits a well-tuned regime with no auxiliary condition on the relative sizes of N, K, and T. If $\gamma$ is restricted to a fixed interval $[ \underline { { \gamma } } , 1 ] \subset ( 0 , 1 ]$ , then all γ-dependence is absorbed into constants, yielding the usual $\mathcal { O } ( \epsilon ^ { - 2 } )$ communication and $\mathcal { O } ( \epsilon ^ { - 4 } )$ sample-complexity exponents under the standard tuned choice of $( T , K )$

## A.2 Notation and preliminaries

Let $\begin{array} { r } { F ( x ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { i } ( x ) } \end{array}$ and $p : = N / M$ . In round t, the server samples $S _ { t } \subset [ M ]$ uniformly at random with $| S _ { t } | = N$

Local iterates. For each active client $i \in S _ { t }$ , let $x _ { i } ^ { t , 0 } = x ^ { t }$ . Define local increments

$$
\Delta _ { i } ^ { t , k - 1 } : = x _ { i } ^ { t , k } - x _ { i } ^ { t , k - 1 } , \qquad \Delta _ { i } ^ { t } : = \sum _ { k = 1 } ^ { K } \Delta _ { i } ^ { t , k - 1 } .
$$

The server update is

$$
x ^ { t + 1 } - x ^ { t } = \frac { 1 } { N } \sum _ { i \in S _ { t } } \Delta _ { i } ^ { t } .\tag{A.2}
$$

Projection operator. Throughout this appendix, $\Pi _ { \mathcal { D } } ( \cdot )$ denotes the Euclidean projection onto $\mathcal { D } \colon$

$$
\Pi _ { \mathcal { D } } ( z ) : = \arg \operatorname* { m i n } _ { u \in \mathcal { D } } \| u - z \| _ { 2 } .
$$

Recall $\mathcal { D } = \{ \Delta : \| \Delta \| ^ { g } \leq \eta \}$

Norm equivalence. By Assumption 4, in finite-dimensional spaces, for any $\Delta \in \mathcal { D }$

$$
\| \Delta \| _ { 2 } \le \bar { \rho } \| \Delta \| ^ { g } \le \bar { \rho } \eta .\tag{A.3}
$$

In particular, $\| \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } \le \bar { \rho } \eta$ and

$$
\| x ^ { t + 1 } - x ^ { t } \| _ { 2 } \leq \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \| \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } \leq K \bar { \rho } \eta .\tag{A.4}
$$

Stochastic gradients. We write the stochastic gradient queried at client i as $\nabla f _ { i } ( X , \xi )$ , where ξ denotes the (mini-)batch sampling randomness. We assume conditional unbiasedness and bounded conditional variance: for all x and all $i \in [ M ]$

$$
\begin{array} { r } { \mathbb { E } \big [ \nabla f _ { i } ( x , \xi ) \mid x \big ] = \nabla f _ { i } ( x ) , \qquad \mathbb { E } \Big [ \big \| \nabla f _ { i } ( x , \xi ) - \nabla f _ { i } ( x ) \big \| _ { 2 } ^ { 2 } \mid x \Big ] \leq \sigma ^ { 2 } . } \end{array}\tag{A.5}
$$

Filtration. Let $\{ \mathcal { F } _ { t , k } \}$ be the natural filtration generated by all randomness revealed up to (right before) drawing the mini-batch $\xi _ { i } ^ { t , k - 1 }$ used to query $\nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } )$ in round t. Conditioned on $\mathcal { F } _ { t , k }$ , the fresh sample $\xi _ { i } ^ { t , k - 1 }$ is independent of the past.

## A.3 Auxiliary lemmas

Lemma A.2 (L-smoothness descent). If F is L-smooth, then for all t,

$$
\begin{array} { r l } & { F ( { x } ^ { t + 1 } ) \le F ( { x } ^ { t } ) + \langle \nabla F ( { x } ^ { t } ) , { x } ^ { t + 1 } - { x } ^ { t } \rangle + \displaystyle \frac { L } { 2 } \| { x } ^ { t + 1 } - { x } ^ { t } \| _ { 2 } ^ { 2 } } \\ & { \qquad \le F ( { x } ^ { t } ) + \langle \nabla F ( { x } ^ { t } ) , { x } ^ { t + 1 } - { x } ^ { t } \rangle + \displaystyle \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } . } \end{array}\tag{A.6}
$$

Lemma A.3 (Support function of D). Let $\| \cdot \| ^ { g , * }$ be the dual norm induced $b y \parallel \cdot \parallel ^ { g } .$ , and define

$$
u _ { t } \in \arg \operatorname* { m i n } _ { u \in \mathcal { D } } \langle \nabla F ( x ^ { t } ) , u \rangle .
$$

Then

$$
\langle \nabla F ( x ^ { t } ) , u _ { t } \rangle = - \eta \| \nabla F ( x ^ { t } ) \| ^ { g , * } .\tag{A.7}
$$

Lemma A.4 (Projection three-point inequality). Let $\Pi _ { \mathcal { D } } ( \cdot )$ be the Euclidean projection onto a nonempty closed convex set D. Let $y = \Pi _ { \mathcal { D } } ( x - g )$ . Then for any $u \in \mathcal { D }$ 2

$$
\langle g , y - u \rangle \leq \frac { 1 } { 2 } \Big ( \| u - x \| _ { 2 } ^ { 2 } - \| u - y \| _ { 2 } ^ { 2 } - \| y - x \| _ { 2 } ^ { 2 } \Big ) .\tag{A.8}
$$

Proof. Optimality of Euclidean projection implies $\langle ( x - g ) - y , u - y \rangle \leq 0$ for all $u \in \mathcal { D }$ . Rearranging gives $\langle g , y - u \rangle \leq \langle x - y , u - y \rangle$ . Finally use $2 \langle x - y , u - y \rangle = \| u - x \| _ { 2 } ^ { 2 } - \| u - y \| _ { 2 } ^ { 2 } - \| y - x \| _ { 2 } ^ { 2 }$ .

Lemma A.5 (One-gradient predict-and-correct inequality). Fix (t, i) and set $\hat { \Delta } _ { i } ^ { t , - 1 } = 0$ . For $k = 1 , \ldots , K$ , Algorithm 1 generates

$$
\begin{array} { r } { \hat { \Delta } _ { i } ^ { t , k - 1 } = \Pi _ { \mathcal { D } } \big ( \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { t , k } \big ) , \qquad \Delta _ { i } ^ { t , k - 1 } = \Pi _ { \mathcal { D } } \big ( \hat { \Delta } _ { i } ^ { t , k - 1 } - g _ { i } ^ { t , k } \big ) . } \end{array}
$$

Then for any $u \in \mathcal { D }$

$$
\sum _ { k = 1 } ^ { K } \langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u \rangle \leq \frac { \| u \| _ { 2 } ^ { 2 } } { 2 } + \frac { 1 } { 2 } \sum _ { k = 2 } ^ { K } \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } + 2 K ( \bar { \rho } \eta ) ^ { 2 } .\tag{A.9}
$$

Proof. Apply Lemma A.4 to $\Delta _ { i } ^ { t , k - 1 } = \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 1 } - g _ { i } ^ { t , k } )$ with $x = \hat { \Delta } _ { i } ^ { t , k - 1 }$ :

$$
\langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u \rangle \leq \frac { 1 } { 2 } \Big ( \| u - \hat { \Delta } _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } - \| u - \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \Big ) .
$$

For $k = 1 , \hat { \Delta } _ { i } ^ { t , 0 } = \Pi _ { \mathcal { D } } ( - g _ { i } ^ { t , 1 } ) \in \mathcal { D }$ , hence $\| u - \hat { \Delta } _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } \leq ( 2 \bar { \rho } \eta ) ^ { 2 }$

For $k \geq 2$ , by non-expansiveness of Euclidean projection,

$$
\begin{array} { r } { \| \hat { \Delta } _ { i } ^ { t , k - 1 } - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } = \Big \| \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { t , k } ) - \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { t , k - 1 } ) \Big \| _ { \mathrm { , } } \leq \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } . } \end{array}
$$

Thus

$$
\begin{array} { r l } & { \| u - \hat { \Delta } _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \leq 2 \| u - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } + 2 \| \hat { \Delta } _ { i } ^ { t , k - 1 } - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 \| u - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } + 2 \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{A.10}
$$

Summing the inequality (A.10) over $k = 1 , \ldots , K$ , and treating the $k = 1$ case separately, we have:

$$
\begin{array} { r l } { \displaystyle \sum _ { k = 1 } ^ { K } \langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u \rangle \leq \underbrace { \frac { 1 } { 2 } \| u - \hat { \Delta } _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| u - \Delta _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } } _ { k = 1 } + \displaystyle \sum _ { k = 2 } ^ { K } \frac { 1 } { 2 } \Big ( \underbrace { \| u - \hat { \Delta } _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } } _ { \mathrm { s u b s t i t u t e ~ ( A . 1 0 ) } } - \| u - \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \Big ) } & { } \\ { \leq \frac { 1 } { 2 } \| u - \hat { \Delta } _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| u - \Delta _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } } & { } \\ { \quad \quad } & { + \displaystyle \sum _ { k = 2 } ^ { K } \Big ( \| u - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } + \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } - \frac { 1 } { 2 } \| u - \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \Big ) . \qquad ( \mathrm { A } . } \end{array}\tag{11}
$$

Rearranging the distance terms in (A.11) reveals a telescoping sum:

$$
\begin{array} { l } { { \displaystyle - \frac 1 2 \| u - \Delta _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } + \sum _ { k = 2 } ^ { K } \| u - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } - \sum _ { k = 2 } ^ { K } \frac 1 2 \| u - \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle = - \frac 1 2 \| u - \Delta _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } + \Big ( \| u - \Delta _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } + \sum _ { j = 1 } ^ { K - 2 } \| u - \Delta _ { i } ^ { t , j } \| _ { 2 } ^ { 2 } \Big ) - \Big ( \sum _ { j = 1 } ^ { K - 2 } \frac 1 2 \| u - \Delta _ { i } ^ { t , j } \| _ { 2 } ^ { 2 } + \frac 1 2 \| u - \Delta _ { i } ^ { t , K - 1 } \| _ { 2 } ^ { 2 } \Big ) } } \\ { { \displaystyle = \frac 1 2 \| u - \Delta _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } + \sum _ { j = 1 } ^ { K - 2 } \frac 1 { 2 } \| u - \Delta _ { i } ^ { t , j } \| _ { 2 } ^ { 2 } - \frac 1 2 \| u - \Delta _ { i } ^ { t , K - 1 } \| _ { 2 } ^ { 2 } } } \\ { { \displaystyle \quad \sum _ { j = 0 } ^ { K - 2 } \frac 1 { 2 } ( 2 \bar { \rho } \eta ) ^ { 2 } \le 2 K ( \bar { \rho } \eta ) ^ { 2 } , } } \end{array}
$$

where we used the diameter bound $\lVert u - \Delta _ { i } ^ { t , j } \rVert _ { 2 } ~ \leq ~ 2 \bar { \rho } \eta$ and dropped the final negative term. Substituting this back into (A.11), and using the bound $\| u - \hat { \Delta } _ { i } ^ { t , 0 } \| _ { 2 } ^ { 2 } \leq \| u \| _ { 2 } ^ { 2 } + ( 2 \bar { \rho } \eta ) ^ { 2 }$ , we obtain (A.9). □

Lemma A.6 (Single-call approximate projection certificate). Let $\mathcal { D } = \{ u : \| u \| ^ { g } \leq \eta \}$ and let $z = x - g$ . Suppose an inexact projection call returns $y \in \mathcal { D }$ with residual

$$
\mathrm { r e s } _ { \mathcal { D } } ( z , y ) : = \biggl [ \underset { u \in \mathcal { D } } { \operatorname* { s u p } } \langle z - y , u - y \rangle \biggr ] _ { + } \leq \varepsilon .\tag{A.12}
$$

Then, for every $u \in \mathcal { D }$

$$
\langle g , y - u \rangle \leq \frac { 1 } { 2 } \Big ( \| u - x \| _ { 2 } ^ { 2 } - \| u - y \| _ { 2 } ^ { 2 } - \| y - x \| _ { 2 } ^ { 2 } \Big ) + \varepsilon .\tag{A.13}
$$

Moreover, $i f p = \Pi _ { \mathcal { D } } ( z )$ is the exact Euclidean projection $o f z$ onto ${ \mathcal { D } } ,$ then

$$
\| y - p \| _ { 2 } ^ { 2 } \leq \varepsilon .\tag{A.14}
$$

Consequently, if $y _ { a } , y _ { b }$ are inexact projections of $z _ { a } , z _ { b }$ with residuals at most $\varepsilon _ { a } , \varepsilon _ { b } ,$ respectively, then

$$
\begin{array} { r } { \| y _ { a } - y _ { b } \| _ { 2 } \leq \| z _ { a } - z _ { b } \| _ { 2 } + \sqrt { \varepsilon _ { a } } + \sqrt { \varepsilon _ { b } } . } \end{array}\tag{A.15}
$$

For the norm ball $\mathcal { D } = \{ u : \| u \| ^ { g } \leq \eta \}$ , the residual in (A.12) can be evaluated as

$$
\mathrm { r e s } _ { \mathcal { D } } ( z , y ) = [ \eta \| z - y \| ^ { g , * } - \langle z - y , y \rangle ] _ { + } .\tag{A.16}
$$

Proof. The residual condition gives $\langle z - y , u - y \rangle \leq \varepsilon$ for all $u \in \mathcal { D }$ . Since $z = x - g$ , this implies

$$
\langle ( x - y ) - g , u - y \rangle \leq \varepsilon ,
$$

and hence

$$
\left. g , y - u \right. \leq \left. x - y , u - y \right. + \varepsilon .
$$

Using the identity $2 \langle x - y , u - y \rangle = \| u - x \| _ { 2 } ^ { 2 } - \| u - y \| _ { 2 } ^ { 2 } - \| y - x \| _ { 2 } ^ { 2 }$ gives (A.13).

Let $p = \Pi _ { \mathcal { D } } ( z )$ . The exact projection optimality condition gives $\langle z - p , y - p \rangle \leq 0$ because $y \in \mathcal { D }$ Taking $u = p$ in the approximate condition gives $\langle z - y , p - y \rangle \leq \varepsilon$ , or equivalently $\langle z - y , y - p \rangle \geq - \varepsilon$ Therefore,

$$
\begin{array} { r } { \| y - p \| _ { 2 } ^ { 2 } = \left. y - p , y - p \right. = \left. z - p , y - p \right. - \left. z - y , y - p \right. \le \varepsilon , } \end{array}
$$

which proves (A.14). The approximate non-expansiveness bound (A.15) follows by the triangle inequality and the non-expansiveness of the exact projection:

$$
\begin{array} { r } { \| y _ { a } - y _ { b } \| _ { 2 } \leq \| y _ { a } - \Pi _ { \mathcal { D } } ( z _ { a } ) \| _ { 2 } + \| \Pi _ { \mathcal { D } } ( z _ { a } ) - \Pi _ { \mathcal { D } } ( z _ { b } ) \| _ { 2 } + \| \Pi _ { \mathcal { D } } ( z _ { b } ) - y _ { b } \| _ { 2 } \leq \sqrt { \varepsilon _ { a } } + \| z _ { a } - z _ { b } \| _ { 2 } + \sqrt { \varepsilon _ { b } } . } \end{array}
$$

Finally,

$$
\operatorname* { s u p } _ { u \in \mathcal { D } } \langle z - y , u - y \rangle = \operatorname* { s u p } _ { \| u \| ^ { g } \leq \eta } \langle z - y , u \rangle - \langle z - y , y \rangle = \eta \| z - y \| ^ { g , * } - \langle z - y , y \rangle ,
$$

which proves (A.16) after applying the positive part.

Lemma A.7 (Inexact predict-and-correct from single-call residuals). Fix (t, i) and set $\hat { \Delta } _ { i } ^ { t , - 1 } =$ 0. Suppose the prediction and correction calls in the local trajectory return feasible points $\hat { \Delta } _ { i } ^ { t , k - 1 } , \hat { \Delta } _ { i } ^ { t , k - 1 } \in \mathcal { D }$ for $k = 1 , \ldots , K$ . Let

$$
\hat { z } _ { i } ^ { t , k } : = \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { t , k } , \qquad z _ { i } ^ { t , k } : = \hat { \Delta } _ { i } ^ { t , k - 1 } - g _ { i } ^ { t , k } ,
$$

and define

$$
\hat { \varepsilon } _ { i } ^ { t , k } : = \left[ \operatorname* { s u p } _ { v \in \mathcal { D } } \left. \hat { z } _ { i } ^ { t , k } - \hat { \Delta } _ { i } ^ { t , k - 1 } , v - \hat { \Delta } _ { i } ^ { t , k - 1 } \right. \right] _ { + } , \qquad \varepsilon _ { i } ^ { t , k } : = \left[ \operatorname* { s u p } _ { v \in \mathcal { D } } \left. z _ { i } ^ { t , k } - \Delta _ { i } ^ { t , k - 1 } , v - \Delta _ { i } ^ { t , k - 1 } \right. \right] _ { + } .
$$

Then for every $u \in \mathcal { D }$

$$
\sum _ { k = 1 } ^ { K } \left. g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u \right. \leq \frac { \| u \| _ { 2 } ^ { 2 } } { 2 } + 2 \sum _ { k = 2 } ^ { K } \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } + 2 K ( \bar { \rho } \eta ) ^ { 2 } + 5 \sum _ { k = 1 } ^ { K } \left( \hat { \varepsilon } _ { i } ^ { t , k } + \varepsilon _ { i } ^ { t , k } \right) .\tag{A.17}
$$

Proof. For the correction call at step k, Lemma $\mathrm { A . 6 }$ gives, with $x = \hat { \Delta } _ { i } ^ { t , k - 1 }$ and $y = \Delta _ { i } ^ { t , k - 1 }$

$$
\left. g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u \right. \leq \frac { 1 } { 2 } \left( \| u - \hat { \Delta } _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } - \| u - \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \right) + \varepsilon _ { i } ^ { t , k } ,\tag{A.18}
$$

where we dropped the nonpositive term $- \| \Delta _ { i } ^ { t , k - 1 } - \hat { \Delta } _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } / 2$ . For $k = 1$ , feasibility gives $\lVert u - \hat { \Delta } _ { i } ^ { t , 0 } \rVert _ { 2 } \leq 2 \bar { \rho } \eta$ . For $k \geq 2$ , apply the approximate non-expansiveness bound in Lemma $\mathrm { A . 6 }$ to the prediction call at step k and the correction call at step $k - 1$ . Their inputs difer by $g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 }$ ， and hence

$$
\begin{array} { r } { \| \hat { \boldsymbol { \Delta } } _ { i } ^ { t , k - 1 } - { \boldsymbol { \Delta } } _ { i } ^ { t , k - 2 } \| _ { 2 } \leq \| { \boldsymbol { g } } _ { i } ^ { t , k } - { \boldsymbol { g } } _ { i } ^ { t , k - 1 } \| _ { 2 } + \sqrt { \hat { \varepsilon } _ { i } ^ { t , k } } + \sqrt { \varepsilon _ { i } ^ { t , k - 1 } } . } \end{array}\tag{A.19}
$$

Using $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ and $( { \sqrt { a } } + { \sqrt { b } } ) ^ { 2 } \leq 2 ( a + b )$ , we get

$$
\begin{array} { r l } & { \| u - \hat { \Delta } _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \leq 2 \| u - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } + 2 \| \hat { \Delta } _ { i } ^ { t , k - 1 } - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 \| u - \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } + 4 \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } + 8 \left( \hat { \varepsilon } _ { i } ^ { t , k } + \varepsilon _ { i } ^ { t , k - 1 } \right) . } \end{array}\tag{A.20}
$$

Substituting (A.20) into (A.18) for $k \geq 2$ , and treating $k = 1$ separately, yields the same distance telescoping as in Lemma A.5, except that the gradient-variation term is multiplied by a universal constant and the residual terms are added. The distance terms are bounded by $2 K ( \bar { \rho } \eta ) ^ { 2 }$ using the diameter of D. The residual contribution is bounded by

$$
\sum _ { k = 1 } ^ { K } \varepsilon _ { i } ^ { t , k } + 4 \sum _ { k = 2 } ^ { K } \left( \hat { \varepsilon } _ { i } ^ { t , k } + \varepsilon _ { i } ^ { t , k - 1 } \right) \leq 5 \sum _ { k = 1 } ^ { K } \left( \hat { \varepsilon } _ { i } ^ { t , k } + \varepsilon _ { i } ^ { t , k } \right) ,
$$

which proves (A.17).

Lemma A.8 (Bounding gradient variation). In Algorithm 1, for any $( t , i , k \geq 2 )$

$$
\begin{array} { r } { \mathbb { E } \Big [ \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \Big ] \leq 2 ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \mathbb { E } \| x _ { i } ^ { t , k - 1 } - x _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } + 4 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } . } \end{array}\tag{A.21}
$$

Proof. Fix $( t , i , k \geq 2 )$ . Recall the client-side gradient signal in Algorithm 1:

$$
\begin{array} { r } { { g _ { i } ^ { t , k } = ( 1 - \beta _ { 1 } ) \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \big ) + m ^ { t } } . } \end{array}
$$

Since $c _ { i } ^ { t }$ and $m ^ { t }$ are fixed within round $t ,$ we have

$$
\begin{array} { r } { \begin{array} { r } { g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } = ( 1 - \beta _ { 1 } ) \Big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 2 } , \xi _ { i } ^ { t , k - 2 } ) \Big ) . } \end{array} } \end{array}
$$

Define the stochastic gradient noise

$$
\begin{array} { r } { \zeta _ { i } ^ { t , k - 1 } : = \nabla f _ { i } ( { x } _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( { x } _ { i } ^ { t , k - 1 } ) , \qquad \zeta _ { i } ^ { t , k - 2 } : = \nabla f _ { i } ( { x } _ { i } ^ { t , k - 2 } , \xi _ { i } ^ { t , k - 2 } ) - \nabla f _ { i } ( { x } _ { i } ^ { t , k - 2 } ) . } \end{array}
$$

Then

$$
\nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 2 } , \xi _ { i } ^ { t , k - 2 } ) = \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 2 } ) \big ) + \big ( \zeta _ { i } ^ { t , k - 1 } - \zeta _ { i } ^ { t , k - 2 } \big ) .
$$

Using $\| a + b \| ^ { 2 } \leq 2 \| a \| ^ { 2 } + 2 \| b \| ^ { 2 }$ , we obtain

$$
\begin{array} { r l } & { \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } = ( 1 - \beta _ { 1 } ) ^ { 2 } \Big \| \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 2 } ) \big ) + \big ( \zeta _ { i } ^ { t , k - 1 } - \zeta _ { i } ^ { t , k - 2 } \big ) \Big \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 \big ( 1 - \beta _ { 1 } \big ) ^ { 2 } \| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 2 } ) \| _ { 2 } ^ { 2 } + 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \| \zeta _ { i } ^ { t , k - 1 } - \zeta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } . } \end{array}
$$

For the first term, by L-smoothness of $f _ { i } ,$

$$
\| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 2 } ) \| _ { 2 } \leq L \| x _ { i } ^ { t , k - 1 } - x _ { i } ^ { t , k - 2 } \| _ { 2 } .
$$

For the noise term, conditioned on $\mathcal { F } _ { t , k }$ (right before drawing $\xi _ { i } ^ { t , k - 1 } ) , \zeta _ { i } ^ { t , k - 1 }$ is independent of the past and satisfies $\mathbb { E } [ \| \zeta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , k } ] \le \sigma ^ { 2 }$ by (A.5), while $\zeta _ { i } ^ { t , k - 2 }$ is $\mathcal { F } _ { t , k }$ -measurable. Hence

$$
\begin{array} { r l } & { \mathbb { E } [ \| \zeta _ { i } ^ { t , k - 1 } - \zeta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , k } ] \le 2 \mathbb { E } [ \| \zeta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , k } ] + 2 \| \zeta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } } \\ & { \qquad \le 2 \sigma ^ { 2 } + 2 \| \zeta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Taking expectation again and using $\mathbb { E } \| \zeta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } \leq \sigma ^ { 2 }$ yields

$$
\begin{array} { r } { \mathbb { E } \big [ \| \zeta _ { i } ^ { t , k - 1 } - \zeta _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } \big ] \leq 4 \sigma ^ { 2 } . } \end{array}
$$

Combining the above bounds gives (A.21).

## A.4 Descent decomposition

Define the ideal direction $\begin{array} { r } { u _ { t } \in \arg \operatorname* { m i n } _ { u \in { \mathcal { D } } } \langle \nabla F ( x ^ { t } ) , u \rangle } \end{array}$ . Using Lemma A.2 and the server update (A.2), we have

$$
\begin{array} { r l } & { \displaystyle F ( \boldsymbol { x } ^ { t + 1 } ) \le F ( \boldsymbol { x } ^ { t } ) + \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \langle \nabla F ( \boldsymbol { x } ^ { t } ) , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } \rangle + \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } \\ & { \displaystyle \quad = F ( \boldsymbol { x } ^ { t } ) + K \langle \nabla F ( \boldsymbol { x } ^ { t } ) , u _ { t } \rangle + \underbrace { \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \langle \nabla F ( \boldsymbol { x } ^ { t } ) , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } - u _ { t } \rangle } _ { = : E _ { t } } + \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } . } \end{array}\tag{A.22}
$$

By Lemma A.3, $K \langle \nabla F ( x ^ { t } ) , u _ { t } \rangle = - \eta K \| \nabla F ( x ^ { t } ) \| ^ { g , * }$ . Rearranging (A.22) yields

$$
\eta K \| \nabla F ( x ^ { t } ) \| ^ { g , * } \leq F ( x ^ { t } ) - F ( x ^ { t + 1 } ) + \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } + E _ { t } .\tag{A.23}
$$

## A.5 Bounding $E _ { t }$ via predict-and-correct

For convenience, set $x ^ { 0 } : = x ^ { 1 }$ , and define the tracking error

$$
B _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \big [ \| c _ { i } ^ { t } - \nabla f _ { i } ( x ^ { t - 1 } ) \| _ { 2 } ^ { 2 } \big ] .
$$

Recall the client-side gradient signal in Algorithm 1:

$$
\begin{array} { r } { { g _ { i } ^ { t , k } = ( 1 - \beta _ { 1 } ) \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \big ) + m ^ { t } } . } \end{array}
$$

Using $\begin{array} { r } { m ^ { t } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } c _ { j } ^ { t } } \end{array}$ and the identity

$$
\nabla F ( { x } ^ { t } ) = m ^ { t } + \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \big ( \nabla f _ { j } ( { x } ^ { t } ) - { c } _ { j } ^ { t } \big ) ,
$$

we decompose, for each $( i , k )$ ，

$$
\begin{array} { r l } { \langle \nabla F ( \boldsymbol { x } ^ { t } ) , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } - \boldsymbol { u } _ { t } \rangle = \langle m ^ { t } , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } - \boldsymbol { u } _ { t } \rangle + \Big \langle \displaystyle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( \boldsymbol { x } ^ { t } ) - c _ { j } ^ { t } ) , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } - \boldsymbol { u } _ { t } \Big \rangle } & { } \\ { = \langle g _ { i } ^ { t , k } , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } - \boldsymbol { u } _ { t } \rangle - ( 1 - \beta _ { 1 } ) \langle \nabla f _ { i } ( \boldsymbol { x } _ { i } ^ { t , k - 1 } , \boldsymbol { \xi } _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } - \boldsymbol { u } _ { t } \rangle } & { } \\ { + \Big \langle \displaystyle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( \boldsymbol { x } ^ { t } ) - c _ { j } ^ { t } ) , \boldsymbol { \Delta } _ { i } ^ { t , k - 1 } - \boldsymbol { u } _ { t } \Big \rangle . } & { } \end{array}
$$

Summing over $i \in S _ { t } , k \in [ K ]$ and dividing by N, we obtain

$$
E _ { t } = E _ { t } ^ { ( a ) } + E _ { t } ^ { ( b ) } + E _ { t } ^ { ( c ) } ,\tag{A.24}
$$

where

$$
E _ { t } ^ { ( a ) } : = \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle ,\tag{A.25}
$$

$$
E _ { t } ^ { ( b ) } : = - ( 1 - \beta _ { 1 } ) \cdot \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \langle \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle ,\tag{A.26}
$$

$$
E _ { t } ^ { ( c ) } : = \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \Big \langle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( \boldsymbol x ^ { t } ) - \boldsymbol c _ { j } ^ { t } ) , \Delta _ { i } ^ { t , k - 1 } - \boldsymbol u _ { t } \Big \rangle .\tag{A.27}
$$

Step 1: bound $E _ { t } ^ { ( a ) }$ (predict-and-correct). Applying Lemma A.5 with $u = u _ { t }$ and using $\| u _ { t } \| _ { 2 } \le \bar { \rho } \eta$ , we get

$$
E _ { t } ^ { ( a ) } \leq \frac { ( \bar { \rho } \eta ) ^ { 2 } } { 2 } + \frac { 1 } { 2 N } \sum _ { i \in S _ { t } } \sum _ { k = 2 } ^ { K } \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } + 2 K ( \bar { \rho } \eta ) ^ { 2 } .\tag{A.28}
$$

Taking expectation and using Lemma A.8 together with $\| x _ { i } ^ { t , k - 1 } - x _ { i } ^ { t , k - 2 } \| _ { 2 } = \| \Delta _ { i } ^ { t , k - 2 } \| _ { 2 } \leq \bar { \rho } \eta$ yields

$$
\mathbb { E } [ E _ { t } ^ { ( a ) } ] \le \Big ( \frac { 1 } { 2 } + 2 K \Big ) ( \bar { \rho } \eta ) ^ { 2 } + \frac { K - 1 } { 2 } \Big ( 2 ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } ( \bar { \rho } \eta ) ^ { 2 } + 4 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } \Big ) .\tag{A.29}
$$

Dividing by ηK gives

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( a ) } ] } { \eta K } \le \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } .\tag{A.30}
$$

Step 2: bound $E _ { t } ^ { ( b ) }$ (tracking + noise). Define the stochastic gradient noise for each query $( t , i , k )$

$$
\begin{array} { r } { \zeta _ { i } ^ { t , k - 1 } : = \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) , \qquad \mathbb { E } [ \zeta _ { i } ^ { t , k - 1 } \mid \mathcal { F } _ { t , k } ] = 0 , \ \mathbb { E } [ \| \zeta _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , k } ] \leq \sigma ^ { 2 } . } \end{array}
$$

Then

$$
\nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } = \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \big ) + \zeta _ { i } ^ { t , k - 1 } .
$$

Since $\Delta _ { i } ^ { t , k - 1 } \in \mathcal { D }$ and $u _ { t } \in \mathcal { D }$ , by (A.3),

$$
\| \Delta _ { i } ^ { t , k - 1 } - u _ { t } \| _ { 2 } \leq \| \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } + \| u _ { t } \| _ { 2 } \leq 2 \bar { \rho } \eta .\tag{A.31}
$$

Using Cauchy–Schwarz, (A.26), and (A.31), we obtain

$$
\mathbb { E } [ E _ { t } ^ { ( b ) } ] \leq ( 1 - \beta _ { 1 } ) \cdot \frac { 2 \bar { \rho } \eta } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \mathbb { E } [ \| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \| _ { 2 } ] + ( 1 - \beta _ { 1 } ) \cdot \frac { 2 \bar { \rho } \eta } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \mathbb { E } \| \zeta _ { i } ^ { t , k - 1 } \| _ { 2 } .\tag{A.32}
$$

Noise term in (A.32). The projected multiplier $\Delta _ { i } ^ { t , k - 1 } - u _ { t }$ depends on the current query noise, so martingale cancellation cannot be applied to these weighted inner products. Instead, apply Cauchy– Schwarz separately to each query. Conditional Jensen and Assumption 2 give $\mathbb { E } \| \bar { \zeta } _ { i } ^ { \bar { t } , \bar { k } - 1 } \| _ { 2 } \leq \sigma$ Together with the pathwise diameter bound, this yields

$$
- \frac { 1 - \beta _ { 1 } } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \mathbb { E } \langle \zeta _ { i } ^ { t , k - 1 } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle \le 2 \bar { \rho } \eta ( 1 - \beta _ { 1 } ) K \sigma .
$$

Thus (A.32) has the normalized noise contribution

$$
\frac { 2 \bar { \rho } \eta ( 1 - \beta _ { 1 } ) K \sigma } { \eta K } = 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma .\tag{A.33}
$$

This argument uses only feasibility and the conditional second-moment bound; it does not require the projected multiplier to be independent of the current noise.

Tracking + drift term in (A.32). For the first term in (A.32), use triangle inequality and smoothness:

$$
\| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \leq \| \nabla f _ { i } ( x ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } + L \| x _ { i } ^ { t , k - 1 } - x ^ { t - 1 } \| _ { 2 } .
$$

Moreover, by (A.4),

$$
\lVert x _ { i } ^ { t , k - 1 } - x ^ { t - 1 } \rVert _ { 2 } \leq \lVert x ^ { t } - x ^ { t - 1 } \rVert _ { 2 } + \lVert x _ { i } ^ { t , k - 1 } - x ^ { t } \rVert _ { 2 } \leq K \bar { \rho } \eta + ( k - 1 ) \bar { \rho } \eta = ( K + k - 1 ) \bar { \rho } \eta .
$$

Hence

$$
\sum _ { k = 1 } ^ { K } \| x _ { i } ^ { t , k - 1 } - x ^ { t - 1 } \| _ { 2 } \leq \bar { \rho } \eta \sum _ { k = 1 } ^ { K } ( K + k - 1 ) = \bar { \rho } \eta \cdot \frac { K ( 3 K - 1 ) } { 2 } .
$$

Therefore,

$$
\frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \mathbb { E } \big [ \| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \big ] \leq K \cdot \mathbb { E } \Big [ \frac { 1 } { N } \sum _ { i \in S _ { t } } \| \nabla f _ { i } ( x ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \Big ] + \frac { L \bar { \rho } \eta } { 2 } K ( 3 K - 1 ) .\tag{A.34}
$$

Finally, by Jensen, Cauchy–Schwarz, and the definition of $B _ { t }$

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \Big [ \frac { 1 } { N } \sum _ { i \in S _ { t } } \| \nabla f _ { i } ( x ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \Big ] \leq \sqrt { \mathbb { E } \Big [ \Big ( \frac { 1 } { N } \sum _ { i \in S _ { t } } \| \nabla f _ { i } ( x ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \Big ) ^ { 2 } \Big ] } } & { } \\ { \leq \sqrt { \mathbb { E } \Big [ \frac { 1 } { N } \sum _ { i \in S _ { t } } \| \nabla f _ { i } ( x ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } ^ { 2 } \Big ] } = \sqrt { B _ { t } } . } \end{array}\tag{A.35}
$$

Combining (A.32)–(A.35) and dividing by ηK, we conclude

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( b ) } ] } { \eta K } \le 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { B _ { t } } + \Big ( 3 - \frac { 1 } { K } \Big ) \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) ( K \eta ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma .\tag{A.36}
$$

Step 3: bound $E _ { t } ^ { ( c ) }$ (global control mismatch). Decompose

$$
\frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( x ^ { t } ) - c _ { j } ^ { t } ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( x ^ { t } ) - \nabla f _ { j } ( x ^ { t - 1 } ) ) + \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( x ^ { t - 1 } ) - c _ { j } ^ { t } ) .
$$

Accordingly write $E _ { t } ^ { ( c ) } = E _ { t , 1 } ^ { ( c ) } + E _ { t , 2 } ^ { ( c ) }$ , where

$$
E _ { t , 1 } ^ { ( c ) } : = \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \Big \langle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( x ^ { t } ) - \nabla f _ { j } ( x ^ { t - 1 } ) ) , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \Big \rangle ,
$$

$$
E _ { t , 2 } ^ { ( c ) } : = \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \Big \langle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( x ^ { t - 1 } ) - c _ { j } ^ { t } ) , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \Big \rangle .
$$

For $E _ { t , 1 } ^ { ( c ) }$ , by L-smoothness, $\left( \mathrm { { A . 4 } } \right)$ , and (A.31),

$$
\Big \| \frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( x ^ { t } ) - \nabla f _ { j } ( x ^ { t - 1 } ) ) \Big \| _ { 2 } \leq L \| x ^ { t } - x ^ { t - 1 } \| _ { 2 } \leq L K \bar { \rho } \eta .
$$

Therefore,

$$
| { E } _ { t , 1 } ^ { ( c ) } | \leq \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } L K \bar { \rho } \eta \cdot \| \Delta _ { i } ^ { t , k - 1 } - u _ { t } \| _ { 2 } \leq 2 L \bar { \rho } ^ { 2 } K ^ { 2 } \eta ^ { 2 } ,
$$

and hence

$$
\frac { \mathbb { E } [ E _ { t , 1 } ^ { ( c ) } ] } { \eta K } \le 2 L \bar { \rho } ^ { 2 } ( K \eta ) .\tag{A.37}
$$

To sharpen the stochastic part of $E _ { t , 2 } ^ { ( c ) }$ , we split the control error into a bias part and a noise part. Let $\gamma : = 1 - \beta _ { 2 }$ and define, on an enlarged probability space,

$$
\begin{array} { r } { h _ { i } ^ { t } : = \nabla f _ { i } ( x ^ { t - 1 } ) , \qquad e _ { i } ^ { t } : = c _ { i } ^ { t } - h _ { i } ^ { t } , \qquad d _ { i } ^ { t } : = \nabla f _ { i } ( x ^ { t } ) - \nabla f _ { i } ( x ^ { t - 1 } ) . } \end{array}
$$

For every client $i \in [ M ]$ , let

$$
b _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x ^ { t - 1 } ) \big ) , \qquad n _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \zeta _ { i } ^ { t , k - 1 } ,
$$

where inactive clients are equipped with virtual local trajectories and virtual stochastic gradients. Now set

$$
r _ { i } ^ { 1 } : = 0 , \qquad s _ { i } ^ { 1 } : = c _ { i } ^ { 1 } - \nabla f _ { i } ( x ^ { 0 } ) ,
$$

and for $t \geq 1$ 2

$$
r _ { i } ^ { t + 1 } : = ( 1 - \gamma I _ { i } ^ { t } ) r _ { i } ^ { t } + \gamma I _ { i } ^ { t } b _ { i } ^ { t } - d _ { i } ^ { t } ,\tag{A.38}
$$

$$
s _ { i } ^ { t + 1 } : = ( 1 - \gamma I _ { i } ^ { t } ) s _ { i } ^ { t } + \gamma I _ { i } ^ { t } n _ { i } ^ { t } ,\tag{A.39}
$$

where $I _ { i } ^ { t } : = \mathbf { 1 } \{ i \in S _ { t } \}$ . By construction,

$$
e _ { i } ^ { t } = r _ { i } ^ { t } + s _ { i } ^ { t } \qquad { \mathrm { f o r ~ a l l ~ } } i , t .
$$

Define the averaged bias and averaged noise as

$$
r ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } r _ { i } ^ { t } , \qquad v ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } s _ { i } ^ { t } ,
$$

and their second moments as

$$
R _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| r _ { i } ^ { t } \| _ { 2 } ^ { 2 } , \qquad V _ { t } : = \mathbb { E } \| v ^ { t } \| _ { 2 } ^ { 2 } .
$$

Then

$$
\frac { 1 } { M } \sum _ { j = 1 } ^ { M } ( \nabla f _ { j } ( x ^ { t - 1 } ) - c _ { j } ^ { t } ) = - \frac { 1 } { M } \sum _ { j = 1 } ^ { M } e _ { j } ^ { t } = - ( r ^ { t } + v ^ { t } ) .
$$

Hence

$$
E _ { t , 2 } ^ { ( c ) } = - \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \langle r ^ { t } + v ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle .
$$

Using Cauchy–Schwarz and (A.31),

$$
| E _ { t , 2 } ^ { ( c ) } | \leq \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \big ( \| r ^ { t } \| _ { 2 } + \| v ^ { t } \| _ { 2 } \big ) \cdot \| \Delta _ { i } ^ { t , k - 1 } - u _ { t } \| _ { 2 } \leq 2 \bar { \rho } \eta K \big ( \| r ^ { t } \| _ { 2 } + \| v ^ { t } \| _ { 2 } \big ) .
$$

Dividing by $\eta K$ and taking expectation gives

$$
\frac { \mathbb { E } [ E _ { t , 2 } ^ { ( c ) } ] } { \eta K } \le 2 \bar { \rho } \mathbb { E } \| r ^ { t } \| _ { 2 } + 2 \bar { \rho } \mathbb { E } \| v ^ { t } \| _ { 2 } \le 2 \bar { \rho } \sqrt { R _ { t } } + 2 \bar { \rho } \sqrt { V _ { t } } ,\tag{A.40}
$$

where we used $\begin{array} { r } { \| r ^ { t } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { M } \sum _ { i } \| r _ { i } ^ { t } \| _ { 2 } ^ { 2 } } \end{array}$ and Jensen’s inequality. Combining (A.37) and (A.40), we conclude

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( c ) } ] } { \eta K } \le 2 L \bar { \rho } ^ { 2 } ( K \eta ) + 2 \bar { \rho } \sqrt { R _ { t } } + 2 \bar { \rho } \sqrt { V _ { t } } .\tag{A.41}
$$

## A.6 Control-variates recursion and averaging bounds

We initialize the client control variates using a single stochastic gradient query per client: for each $i \in [ M ]$ , draw an independent mini-batch $\xi _ { i } ^ { 0 }$ and set

$$
c _ { i } ^ { 1 } : = \nabla f _ { i } ( x ^ { 1 } , \xi _ { i } ^ { 0 } ) , \qquad m ^ { 1 } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } c _ { i } ^ { 1 } .
$$

This initialization requires only one mini-batch per client and will be suficient for our rates.

Lemma A.9 (One-batch initialization bound). Under (A.5), the above initialization satisfies

$$
B _ { 1 } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \big [ \| c _ { i } ^ { 1 } - \nabla f _ { i } ( x ^ { 0 } ) \| _ { 2 } ^ { 2 } \big ] \leq \sigma ^ { 2 } .
$$

Proof. Since $x ^ { 0 } = x ^ { 1 }$ by convention,

$$
c _ { i } ^ { 1 } - \nabla f _ { i } ( x ^ { 0 } ) = \nabla f _ { i } ( x ^ { 1 } , \xi _ { i } ^ { 0 } ) - \nabla f _ { i } ( x ^ { 1 } ) .
$$

Applying the bounded conditional variance in (A.5) and taking expectation yields E $\| c _ { i } ^ { 1 } - \nabla f _ { i } ( x ^ { 0 } ) \| _ { 2 } ^ { 2 } \leq$ $\sigma ^ { 2 }$ for each i. Averaging over $i = 1 , \dots , M$ gives $B _ { 1 } \leq \sigma ^ { 2 }$ □

Lemma A.10 (Clientwise control recursion and average $\sqrt { B _ { t } }$ bound). Assume each $f _ { i }$ is L-smooth and (A.5) holds. Let $\textstyle p : = { \frac { N } { M } }$ and

$$
B _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \big \| c _ { i } ^ { t } - \nabla f _ { i } ( x ^ { t - 1 } ) \big \| _ { 2 } ^ { 2 } .
$$

Then for all $t \geq 1$

$$
B _ { t + 1 } \leq \Big ( 1 - { \frac { p ( 1 - \beta _ { 2 } ) } { 4 } } \Big ) B _ { t } + { \frac { 3 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { p ( 1 - \beta _ { 2 } ) } } + { \frac { 8 ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K } } .\tag{A.42}
$$

Consequently, for any $T \geq 1$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \le \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } + \frac { 4 \sqrt { 2 } \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K p } } + \frac { 8 \sigma } { p ( 1 - \beta _ { 2 } ) T } .\tag{A.43}
$$

Proof. Fix $t \geq 1$ and let $\gamma : = 1 - \beta _ { 2 }$ . Define

$$
\begin{array} { r l r l r l } & { h _ { i } ^ { t } : = \nabla f _ { i } ( x ^ { t - 1 } ) , } & & { e _ { i } ^ { t } : = c _ { i } ^ { t } - h _ { i } ^ { t } , } & & { d _ { i } ^ { t } : = \nabla f _ { i } ( x ^ { t } ) - \nabla f _ { i } ( x ^ { t - 1 } ) , } \end{array}
$$

so that $\begin{array} { r } { B _ { t } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } } \end{array}$ . Let $I _ { i } ^ { t } : = \mathbf { 1 } \{ i \in S _ { t } \}$ , so $\mathbb { E } [ I _ { i } ^ { t } ] = p$ and $( I _ { i } ^ { t } ) ^ { 2 } = I _ { i } ^ { t }$

Step 1: error dynamics. The control update is

$$
c _ { i } ^ { t + 1 } = c _ { i } ^ { t } + I _ { i } ^ { t } \gamma ( \hat { c } _ { i } ^ { t } - c _ { i } ^ { t } ) , \qquad \hat { c } _ { i } ^ { t } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) .
$$

Subtracting $\nabla f _ { i } ( x ^ { t } ) = h _ { i } ^ { t } + d _ { i } ^ { t }$ , we obtain

$$
e _ { i } ^ { t + 1 } = ( 1 - \gamma I _ { i } ^ { t } ) e _ { i } ^ { t } + \gamma I _ { i } ^ { t } \big ( \hat { c } _ { i } ^ { t } - h _ { i } ^ { t } \big ) - d _ { i } ^ { t } .\tag{A.44}
$$

Step 2: decompose $\hat { c } _ { i } ^ { t } - h _ { i } ^ { t }$ . On the event $\{ I _ { i } ^ { t } = 1 \}$ , let

$$
\zeta _ { i } ^ { t , k - 1 } : = \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) , \qquad n _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \zeta _ { i } ^ { t , k - 1 } ,
$$

and

$$
b _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x ^ { t - 1 } ) \right) .
$$

Then

$$
\hat { c } _ { i } ^ { t } - h _ { i } ^ { t } = b _ { i } ^ { t } + n _ { i } ^ { t } .
$$

By the martingale-diference property of $\{ \zeta _ { i } ^ { t , k - 1 } \} _ { k = 1 } ^ { K }$ ，

$$
\mathbb { E } [ n _ { i } ^ { t } \mid \mathcal { F } _ { t , 1 } , I _ { i } ^ { t } = 1 ] = 0 , \qquad \mathbb { E } [ \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \mid I _ { i } ^ { t } = 1 ] \le \frac { \sigma ^ { 2 } } { K } .
$$

Moreover, by L-smoothness and

$$
\| x _ { i } ^ { t , k - 1 } - x ^ { t - 1 } \| _ { 2 } \leq \| x ^ { t } - x ^ { t - 1 } \| _ { 2 } + \| x _ { i } ^ { t , k - 1 } - x ^ { t } \| _ { 2 } \leq ( K + k - 1 ) \bar { \rho } \eta \leq 2 K \bar { \rho } \eta ,
$$

we have

$$
\| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq \frac { 1 } { K } \sum _ { k = 1 } ^ { K } L ^ { 2 } \| x _ { i } ^ { t , k - 1 } - x ^ { t - 1 } \| _ { 2 } ^ { 2 } \leq 4 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } .\tag{A.45}
$$

Step 3: active case $I _ { i } ^ { t } = 1$ . Condition on $\mathcal { F } _ { t , 1 }$ and the event $\{ I _ { i } ^ { t } = 1 \}$ . Then, by (A.44),

$$
e _ { i } ^ { t + 1 } = \beta _ { 2 } e _ { i } ^ { t } + \gamma b _ { i } ^ { t } + \gamma n _ { i } ^ { t } - d _ { i } ^ { t } .
$$

Since $e _ { i } ^ { t }$ and $d _ { i } ^ { t }$ are $\mathcal { F } _ { t , 1 }$ -measurable and $\mathbb { E } [ n _ { i } ^ { t } \mid \mathcal { F } _ { t , 1 } ] = 0$

$$
\begin{array} { r } { \mathbb { E } \Big [ \| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , 1 } , I _ { i } ^ { t } = 1 \Big ] \leq \Big ( 1 - \displaystyle \frac { \gamma } { 2 } \Big ) \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + 3 \gamma \mathbb { E } \Big [ \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , 1 } , I _ { i } ^ { t } = 1 \Big ] } \\ { + 2 \gamma ^ { 2 } \mathbb { E } \Big [ \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , 1 } , I _ { i } ^ { t } = 1 \Big ] + \displaystyle \frac { 3 } { \gamma } \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } . } \end{array}\tag{A.46}
$$

Step 4: inactive case $I _ { i } ^ { t } = 0$ , then average over $I _ { i } ^ { t }$ . If $I _ { i } ^ { t } = 0$ , then $c _ { i } ^ { t + 1 } = c _ { i } ^ { t }$ , so by (A.44),

$$
e _ { i } ^ { t + 1 } = e _ { i } ^ { t } - d _ { i } ^ { t } .
$$

Using Young’s inequality with parameter $p \gamma / 4$ , we get

$$
\| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } = \| e _ { i } ^ { t } - d _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq \Big ( 1 + \frac { p \gamma } { 4 } \Big ) \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \Big ( 1 + \frac { 4 } { p \gamma } \Big ) \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } .\tag{A.47}
$$

Now take expectation over $I _ { i } ^ { t }$ and use $\mathbb { P } ( I _ { i } ^ { t } = 1 ) = p$ . Applying (A.46), (A.47), and $\mathbb { E } \Vert n _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } \leq$ $\sigma ^ { 2 } / K$ , we obtain

$$
\mathbb { E } \Big [ \| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \Big ] \leq \Big ( 1 - \frac { p \gamma } { 4 } \Big ) \mathbb { E } \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + 3 \mathbb { E } [ \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \ | \ I _ { i } ^ { t } = 1 ] + \frac { 8 \gamma ^ { 2 } \sigma ^ { 2 } } { K } + \frac { 8 } { p \gamma } \mathbb { E } \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } ,\tag{A.48}
$$

where we used $p \gamma \leq 1$

Step 5: bound $b _ { i } ^ { t }$ and $d _ { i } ^ { t } ,$ , and average over i. By (A.45),

$$
\mathbb { E } [ \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } | I _ { i } ^ { t } = 1 ] \le 4 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } .
$$

Also by L-smoothness and (A.4),

$$
\| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } = \| \nabla f _ { i } ( x ^ { t } ) - \nabla f _ { i } ( x ^ { t - 1 } ) \| _ { 2 } ^ { 2 } \leq L ^ { 2 } \| x ^ { t } - x ^ { t - 1 } \| _ { 2 } ^ { 2 } \leq L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } .
$$

Substituting these bounds into (A.48) and using $p \gamma \leq 1$ yields

$$
\mathbb { E } \| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { p \gamma } { 4 } \Big ) \mathbb { E } \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { 3 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { p \gamma } + \frac { 8 \gamma ^ { 2 } \sigma ^ { 2 } } { K } .
$$

Averaging over $i = 1 , \ldots , M$ gives (A.42).

Step 6: average $\sqrt { B _ { t } }$ . Let $\textstyle \alpha : = { \frac { p \gamma } { 4 } }$ and

$$
C : = \frac { 3 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { p \gamma } + \frac { 8 \gamma ^ { 2 } \sigma ^ { 2 } } { K } .
$$

Then $B _ { t + 1 } \leq ( 1 - \alpha ) B _ { t } + C$ implies $\begin{array} { r } { B _ { t } \leq ( 1 - \alpha ) ^ { t - 1 } B _ { 1 } + \frac { C } { \alpha } } \end{array}$ . Thus $\begin{array} { r } { \sqrt { B _ { t } } \le ( 1 - \alpha ) ^ { \frac { t - 1 } { 2 } } \sqrt { B _ { 1 } } + \sqrt { \frac { C } { \alpha } } } \end{array}$ and

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \leq \sqrt { \frac { C } { \alpha } } + \frac { 2 \sqrt { B _ { 1 } } } { \alpha T } .
$$

Using $B _ { 1 } \leq \sigma ^ { 2 }$ (one-batch init) gives $\sqrt { B _ { 1 } } \le \sigma$ . Moreover,

$$
\sqrt { \frac { C } { \alpha } } \leq \sqrt { \frac { 3 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { p \gamma \cdot \alpha } } + \sqrt { \frac { 8 \gamma ^ { 2 } \sigma ^ { 2 } / K } { \alpha } } = \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p \gamma } + \frac { 4 \sqrt { 2 } \sigma \sqrt { \gamma } } { \sqrt { K p } } .
$$

Finally, $\begin{array} { r } { \frac { 2 \sqrt { B _ { 1 } } } { \alpha T } \le \frac { 8 \sigma } { p \gamma T } } \end{array}$ . Recalling $\gamma = 1 - \beta _ { 2 }$ proves (A.43).

Lemma A.11 (Bias recursion and average $\sqrt { R _ { t } }$ bound). Assume each $f _ { i }$ is L-smooth. Let $R _ { t } : = $ $\begin{array} { r } { \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| r _ { i } ^ { t } \| _ { 2 } ^ { 2 } } \end{array}$ , where $\{ r _ { i } ^ { t } \}$ is defined in (A.38). Then for all $t \geq 1$

$$
R _ { t + 1 } \leq \Big ( 1 - \frac { p ( 1 - \beta _ { 2 } ) } { 4 } \Big ) R _ { t } + \frac { 3 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { p ( 1 - \beta _ { 2 } ) } .\tag{A.49}
$$

Consequently,

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } { \sqrt { R _ { t } } } \leq \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } .\tag{A.50}
$$

Proof. Fix $t \geq 1$ and let $\gamma : = 1 - \beta _ { 2 }$ . From (A.38), on the event $\{ I _ { i } ^ { t } = 1 \}$ we have

$$
\boldsymbol { r } _ { i } ^ { t + 1 } = \beta _ { 2 } \boldsymbol { r } _ { i } ^ { t } + \gamma \boldsymbol { b } _ { i } ^ { t } - \boldsymbol { d } _ { i } ^ { t } ,
$$

while on the event $\{ I _ { i } ^ { t } = 0 \}$ we have

$$
\begin{array} { r } { r _ { i } ^ { t + 1 } = r _ { i } ^ { t } - d _ { i } ^ { t } . } \end{array}
$$

The proof now follows exactly the same steps as in Lemma A.10, except that the stochastic term $n _ { i } ^ { t }$ is absent. Using (A.45) and the bound $\| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 }$ , one obtains

$$
\mathbb { E } \| r _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { p \gamma } { 4 } \Big ) \mathbb { E } \| r _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { 3 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { p \gamma } .
$$

Averaging over i proves (A.49). Since $r _ { i } ^ { 1 } = 0$ for all $i ,$ we have $R _ { 1 } ~ = ~ 0$ . Applying the same geometric-series argument as in Step 6 of Lemma A.10 yields (A.50). □

Lemma A.12 (Server-average noise recursion and average $\sqrt { V _ { t } }$ bound). Assume (A.5) holds. Let

$$
Q _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| s _ { i } ^ { t } \| _ { 2 } ^ { 2 } , \qquad v ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } s _ { i } ^ { t } , \qquad V _ { t } : = \mathbb { E } \| v ^ { t } \| _ { 2 } ^ { 2 } ,
$$

where $\{ s _ { i } ^ { t } \}$ is defined in (A.39). Then, for all $t \geq 1$

$$
Q _ { t + 1 } \leq \big ( 1 - p ( 1 - \beta _ { 2 } ) \big ) Q _ { t } + \frac { p ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K } ,\tag{A.51}
$$

$$
V _ { t + 1 } \leq \Big ( 1 - \frac { p ( 1 - \beta _ { 2 } ) } { 2 } \Big ) V _ { t } + \frac { p ( 1 - \beta _ { 2 } ) ^ { 2 } } { M } Q _ { t } + \frac { p ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K M } .\tag{A.52}
$$

Moreover,

$$
V _ { 1 } = \mathbb { E } \Big \| \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \big ( c _ { i } ^ { 1 } - \nabla f _ { i } ( x ^ { 0 } ) \big ) \Big \| _ { 2 } ^ { 2 } \leq \frac { \sigma ^ { 2 } } { M } .\tag{A.53}
$$

Consequently, for any $T \geq 1$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { V _ { t } } \le \frac { 4 \sigma } { \sqrt { N ( 1 - \beta _ { 2 } ) T } } + \frac { 4 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K M } } .\tag{A.54}
$$

Proof. Fix $t \geq 1$ and let $\gamma : = 1 - \beta _ { 2 }$

Step 1: recursion for $Q _ { t }$ . From (A.39), on the event $\{ I _ { i } ^ { t } = 1 \}$ we have

$$
s _ { i } ^ { t + 1 } = \beta _ { 2 } s _ { i } ^ { t } + \gamma n _ { i } ^ { t } ,
$$

while on the event $\{ I _ { i } ^ { t } = 0 \}$ we have $s _ { i } ^ { t + 1 } = s _ { i } ^ { t }$ . Conditioned on $\mathcal { F } _ { t , 1 }$ and $I _ { i } ^ { t } = 1$ , the cross term vanishes because $\mathbb { E } [ n _ { i } ^ { t } \mid \mathcal { F } _ { t , 1 } , I _ { i } ^ { t } = 1 ] = 0 .$ . Hence

$$
\mathbb { E } [ \| s _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t , 1 } , I _ { i } ^ { t } = 1 ] = \beta _ { 2 } ^ { 2 } \| s _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \gamma ^ { 2 } \mathbb { E } [ \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \mid I _ { i } ^ { t } = 1 ] \le ( 1 - \gamma ) \| s _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { \gamma ^ { 2 } \sigma ^ { 2 } } { K } ,
$$

where we used $\beta _ { 2 } ^ { 2 } \le \beta _ { 2 } = 1 - \gamma$ . Taking expectation over $I _ { i } ^ { t }$ yields

$$
\mathbb { E } \Vert s _ { i } ^ { t + 1 } \Vert _ { 2 } ^ { 2 } \leq ( 1 - p \gamma ) \mathbb { E } \Vert s _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + \frac { p \gamma ^ { 2 } \sigma ^ { 2 } } { K } .
$$

Averaging over $i = 1 , \ldots , M$ proves (A.51).

Step 2: recursion for $V _ { t }$ . Averaging (A.39) over i gives

$$
v ^ { t + 1 } = ( 1 - p \gamma ) v ^ { t } + \underbrace { \frac { \gamma } { M } \sum _ { i = 1 } ^ { M } ( I _ { i } ^ { t } - p ) s _ { i } ^ { t } } _ { = : \delta _ { t } } + \underbrace { \frac { \gamma } { M } \sum _ { i = 1 } ^ { M } I _ { i } ^ { t } n _ { i } ^ { t } } _ { = : \xi _ { t } } .
$$

Conditioned on the past, $\mathbb { E } [ \delta _ { t } \mid \mathcal { F } _ { t , 1 } ] = 0$ and $\mathbb { E } [ \xi _ { t } \ | \ \mathcal { F } _ { t , 1 } , \{ I _ { i } ^ { t } \} _ { i = 1 } ^ { M } ] = 0$ . Moreover, the cross terms vanish after taking conditional expectation. Therefore,

$$
\mathbb { E } \big [ \| v ^ { t + 1 } \| _ { 2 } ^ { 2 } \bigm | \mathcal { F } _ { t , 1 } \big ] = ( 1 - p \gamma ) ^ { 2 } \| v ^ { t } \| _ { 2 } ^ { 2 } + \mathbb { E } \big [ \| \delta _ { t } \| _ { 2 } ^ { 2 } \bigm | \mathcal { F } _ { t , 1 } \big ] + \mathbb { E } \big [ \| \xi _ { t } \| _ { 2 } ^ { 2 } \bigm | \mathcal { F } _ { t , 1 } \big ] .\tag{A.55}
$$

For the sampling fluctuation term, using the fact that for uniform sampling without replacement the cross-covariances satisfy $\mathbb { E } [ ( I _ { i } ^ { t } - p ) ( I _ { j } ^ { t } - p ) ] \leq 0$ for $i \neq j$ , we have

$$
\mathbb { E } \left[ \Vert \delta _ { t } \Vert _ { 2 } ^ { 2 } \ \vert \ \mathcal { F } _ { t , 1 } \right] \leq \frac { \gamma ^ { 2 } } { M ^ { 2 } } \sum _ { i = 1 } ^ { M } \mathbb { E } \big [ ( I _ { i } ^ { t } - p ) ^ { 2 } \big ] \ \Vert s _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } \leq \frac { p \gamma ^ { 2 } } { M } \cdot \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \Vert s _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } .\tag{A.56}
$$

For the averaged stochastic noise term, conditioned on the sampled set and the past, $\{ n _ { i } ^ { t } \} _ { i \in S _ { t } }$ are zero-mean and independent across clients. Hence

$$
\mathbb { E } \big [ \| \xi _ { t } \| _ { 2 } ^ { 2 } \bigm | \mathcal { F } _ { t , 1 } \big ] \le \frac { \gamma ^ { 2 } } { M ^ { 2 } } \sum _ { i = 1 } ^ { M } \mathbb { E } \big [ I _ { i } ^ { t } \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \bigm | \mathcal { F } _ { t , 1 } \big ] \le \frac { p \gamma ^ { 2 } \sigma ^ { 2 } } { K M } .\tag{A.57}
$$

Substituting (A.56) and (A.57) into (A.55), then taking expectation, yields

$$
V _ { t + 1 } \leq ( 1 - p \gamma ) ^ { 2 } V _ { t } + \frac { p \gamma ^ { 2 } } { M } Q _ { t } + \frac { p \gamma ^ { 2 } \sigma ^ { 2 } } { K M } .
$$

Using $( 1 - a ) ^ { 2 } \leq 1 - a / 2$ for $a \in [ 0 , 1 ]$ gives (A.52).

Step 3: average $\sqrt { V _ { t } }$ . From (A.51), summing the recursion for $Q _ { t }$ over t gives

$$
{ \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } Q _ { t } \leq { \frac { Q _ { 1 } } { p \gamma T } } + { \frac { \gamma \sigma ^ { 2 } } { K } } .
$$

Since $Q _ { 1 } \leq \sigma ^ { 2 }$ by the one-batch initialization, we obtain

$$
{ \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } Q _ { t } \leq { \frac { \sigma ^ { 2 } } { p \gamma T } } + { \frac { \gamma \sigma ^ { 2 } } { K } } .\tag{A.58}
$$

Now sum (A.52) over $t = 1 , \dots , T$ . Since $\begin{array} { r } { V _ { t + 1 } - V _ { t } \le - \frac { p \gamma } { 2 } V _ { t } + \frac { p \gamma ^ { 2 } } { M } Q _ { t } + \frac { p \gamma ^ { 2 } \sigma ^ { 2 } } { K M } } \end{array}$ , we get

$$
\frac { p \gamma } { 2 } \sum _ { t = 1 } ^ { T } V _ { t } \leq V _ { 1 } + \frac { p \gamma ^ { 2 } } { M } \sum _ { t = 1 } ^ { T } Q _ { t } + \frac { T p \gamma ^ { 2 } \sigma ^ { 2 } } { K M } .
$$

Dividing by T and using (A.58) yields

$$
\begin{array} { r l } & { \cfrac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } V _ { t } \leq \frac { 2 V _ { 1 } } { p \gamma T } + \frac { 2 \gamma } { M } \Big ( \frac { \sigma ^ { 2 } } { p \gamma T } + \frac { \gamma \sigma ^ { 2 } } { K } \Big ) + \frac { 2 \gamma \sigma ^ { 2 } } { K M } } \\ & { \qquad \leq \frac { 2 \sigma ^ { 2 } / M } { p \gamma T } + \frac { 2 \sigma ^ { 2 } } { M p T } + \frac { 2 \gamma ^ { 2 } \sigma ^ { 2 } } { K M } + \frac { 2 \gamma \sigma ^ { 2 } } { K M } } \\ & { \qquad \leq \frac { 4 \sigma ^ { 2 } } { N \gamma T } + \frac { 4 \gamma \sigma ^ { 2 } } { K M } , } \end{array}
$$

where we used $M p = N , \gamma \leq 1$ , and (A.53). Finally, Jensen’s inequality gives

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { V _ { t } } \le \sqrt { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } V _ { t } } \le \frac { 4 \sigma } { \sqrt { N \gamma T } } + \frac { 4 \sigma \sqrt { \gamma } } { \sqrt { K M } } ,
$$

which proves (A.54).

## A.7 A generic FL bound shared by both instantiations

Let

$$
\mathcal { G } _ { T } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \big [ \| \nabla F ( x ^ { t } ) \| ^ { g , * } \big ] .
$$

Assume in addition that $1 - \beta _ { 1 } = \eta$ . Summing (A.23) over $t = 1 , \dots , T$ , taking expectation, and telescoping yields

$$
\mathcal { G } _ { T } \le \frac { \Delta _ { F } } { T \eta K } + \frac { L \bar { \rho } ^ { 2 } } { 2 } ( K \eta ) + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { \mathbb { E } [ E _ { t } ] } { \eta K } , \qquad \Delta _ { F } : = F ( x ^ { 1 } ) - F ^ { \star } .
$$

Using (A.24) together with (A.30), (A.36), and (A.41), and then invoking Lemmas A.10, A.11, and A.12, we obtain

$$
\begin{array} { r l } & { \mathcal { G } _ { T } \leq \displaystyle \frac { \Delta _ { F } } { T \eta K } + \frac { 5 L \bar { \rho } ^ { 2 } } { 2 } ( K \eta ) + \left( 2 + \frac { 1 } { 2 K } \right) \bar { \rho } ^ { 2 } \eta + L ^ { 2 } \bar { \rho } ^ { 2 } \eta ^ { 3 } + 2 \sigma ^ { 2 } \eta + \left( 3 - \frac { 1 } { K } \right) L \bar { \rho } ^ { 2 } K \eta ^ { 2 } + 2 \bar { \rho } \sigma \eta } \\ & { \quad \quad \quad \quad \quad + 2 \bar { \rho } \eta \left[ \displaystyle \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } + \frac { 4 \sqrt { 2 } \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K p } } + \frac { 8 \sigma } { p ( 1 - \beta _ { 2 } ) T } \right] } \\ & { \quad \quad \quad \quad \quad + 2 \bar { \rho } \left[ \displaystyle \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } \right] + 2 \bar { \rho } \left[ \displaystyle \frac { 4 \sigma } { \sqrt { N ( 1 - \beta _ { 2 } ) T } } + \frac { 4 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K M } } \right] . } \end{array}\tag{A.59}
$$

This generic bound will be specialized below to a well-tuned regime with fixed $\gamma = 1 - \beta _ { 2 } \in ( 0 , 1 ]$ and to the problem-parameter-free regime.

## A.8 Proof of Theorem A.1

Proof. Set $1 - \beta _ { 2 } = \gamma$ in (A.59). Since $\eta \leq 1$ , we have

$$
L ^ { 2 } \bar { \rho } ^ { 2 } \eta ^ { 3 } \leq L ^ { 2 } \bar { \rho } ^ { 2 } \eta , \qquad \Big ( 3 - \frac { 1 } { K } \Big ) L \bar { \rho } ^ { 2 } K \eta ^ { 2 } \leq 3 L \bar { \rho } ^ { 2 } K \eta , \qquad \frac { 1 6 \sqrt { 2 } L \bar { \rho } ^ { 2 } K \eta ^ { 2 } } { p \gamma } \leq \frac { 1 6 \sqrt { 2 } L \bar { \rho } ^ { 2 } K \eta } { p \gamma } ,
$$

and

$$
\frac { 1 6 \bar { \rho } \sigma \eta } { p \gamma T } \leq \frac { 1 6 \bar { \rho } \sigma } { p \gamma T } , \qquad 2 + \frac { 1 } { 2 K } \leq \frac { 5 } { 2 } .
$$

Therefore,

$$
\mathcal { G } _ { T } \leq \frac { \Delta _ { F } } { T \eta K } + A _ { \mathrm { w t } } ( \gamma ) \eta + \frac { 1 6 \bar { \rho } \sigma } { p \gamma T } + \frac { 8 \bar { \rho } \sigma } { \sqrt { N \gamma T } } + \frac { 8 \bar { \rho } \sigma \sqrt { \gamma } } { \sqrt { K M } } .\tag{A.60}
$$

Let

$$
a : = \frac { \Delta _ { F } } { K T } , \qquad b : = A _ { \mathrm { w t } } ( \gamma ) .
$$

By the choice η = min $\{ 1 , \sqrt { a / b } \}$ , if $a \leq b ,$ then

$$
\frac { a } { \eta } + b \eta = 2 \sqrt { a b } .
$$

If $a > b$ , then $\eta = 1$ and

$$
{ \frac { a } { \eta } } + b \eta = a + b \leq a + 2 { \sqrt { a b } } .
$$

Hence, in all cases,

$$
\frac { \Delta _ { F } } { T \eta K } + { \cal A } _ { \mathrm { w t } } ( \gamma ) \eta \leq \frac { \Delta _ { F } } { K T } + 2 \sqrt { \frac { { \cal A } _ { \mathrm { w t } } ( \gamma ) \Delta _ { F } } { K T } } .
$$

Substituting this into (A.60) proves (A.1).

## A.9 Proof of Theorem 5

Proof. Recall (A.23):

$$
\eta K \| \nabla F ( x ^ { t } ) \| ^ { g , * } \leq F ( x ^ { t } ) - F ( x ^ { t + 1 } ) + \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } + E _ { t } .
$$

Summing over $t = 1 , \dots , T$ , taking expectation, and telescoping gives

$$
\begin{array} { r } { \displaystyle \mathcal { G } _ { T } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \big [ \| \nabla F ( x ^ { t } ) \| ^ { g , * } \big ] \leq \frac { \Delta _ { F } } { T \eta K } + \frac { L } { 2 } \cdot \frac { ( K \bar { \rho } \eta ) ^ { 2 } } { \eta K } + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { \mathbb { E } [ E _ { t } ] } { \eta K } } \\ { \displaystyle = \frac { \Delta _ { F } } { T \eta K } + \frac { L \bar { \rho } ^ { 2 } } { 2 } ( K \eta ) + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { \mathbb { E } [ E _ { t } ] } { \eta K } , } \end{array}\tag{A.61}
$$

where $\Delta _ { F } \mathrel { \mathop : } = F ( x ^ { 1 } ) - F ^ { \star }$

Step 1: substitute the $E _ { t }$ decomposition bound. Using (A.24) together with (A.30), (A.36), and (A.41), we have for each t,

$$
\begin{array} { l } { \displaystyle \frac { \mathbb { E } [ E _ { t } ] } { \eta K } \le \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } } \\ { \displaystyle \qquad + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { B _ { t } } + \Big ( 3 - \frac { 1 } { K } \Big ) \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) ( K \eta ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma } \\ { \displaystyle \qquad + 2 L \bar { \rho } ^ { 2 } ( K \eta ) + 2 \bar { \rho } \sqrt { R _ { t } } + 2 \bar { \rho } \sqrt { V _ { t } } . } \end{array}\tag{A.62}
$$

Step 2: choose hyperparameters and simplify deterministic terms. Assume $T \geq K$ and $N K \leq T$ . Set

$$
\eta : = \frac { N ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } , \qquad 1 - \beta _ { 1 } : = \eta , \qquad 1 - \beta _ { 2 } : = \sqrt { \frac { N K } { T + K } } .\tag{A.63}
$$

Note that $N K \leq T$ implies

$$
K \eta = \frac { ( N K ) ^ { 1 / 4 } } { T ^ { 3 / 4 } } \leq \frac { 1 } { \sqrt { T } } , \qquad \eta \leq \frac { 1 } { \sqrt { T } } \leq 1 .\tag{A.64}
$$

Plugging $1 - \beta _ { 1 } = \eta$ into (A.62) gives

$$
\begin{array} { l } { \displaystyle \frac { \mathbb { E } [ E _ { t } ] } { \eta K } \le \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + L ^ { 2 } \bar { \rho } ^ { 2 } \eta ^ { 3 } + 2 \sigma ^ { 2 } \eta + \Big ( 3 - \frac { 1 } { K } \Big ) L \bar { \rho } ^ { 2 } K \eta ^ { 2 } + 2 \bar { \rho } \sigma \eta } \\ { \displaystyle \qquad + 2 \bar { \rho } \eta \sqrt { B _ { t } } + 2 L \bar { \rho } ^ { 2 } ( K \eta ) + 2 \bar { \rho } \sqrt { R _ { t } } + 2 \bar { \rho } \sqrt { V _ { t } } . } \end{array}\tag{A.65}
$$

Averaging over $t = 1 , \dots , T$ and substituting into (A.61) gives

$$
\begin{array} { c l } { \displaystyle { \mathcal { G } _ { T } \leq \frac { \Delta _ { F } } { T \eta K } + \frac { L \bar { \rho } ^ { 2 } } { 2 } ( K \eta ) + \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + L ^ { 2 } \bar { \rho } ^ { 2 } \eta ^ { 3 } + 2 \sigma ^ { 2 } \eta + \Big ( 3 - \frac { 1 } { K } \Big ) L \bar { \rho } ^ { 2 } K \eta ^ { 2 } + 2 \bar { \rho } \sigma \eta } } \\ { \displaystyle { + 2 \bar { \rho } \eta \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } + 2 L \bar { \rho } ^ { 2 } ( K \eta ) + 2 \bar { \rho } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { R _ { t } } + 2 \bar { \rho } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { V _ { t } } . } } \end{array}\tag{A.66}
$$

Step 3: plug in Lemmas A.10, A.11, and A.12. By Lemmas A.10, A.11, and A.12,

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \le \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } + \frac { 4 \sqrt { 2 } \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K p } } + \frac { 8 \sigma } { p ( 1 - \beta _ { 2 } ) T } ,\tag{A.67}
$$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { R _ { t } } \leq \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } ,\tag{A.68}
$$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { V _ { t } } \le \frac { 4 \sigma } { \sqrt { N ( 1 - \beta _ { 2 } ) T } } + \frac { 4 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K M } } .\tag{A.69}
$$

Substituting these into (A.66) gives

$$
\begin{array} { l } { \displaystyle { \mathcal { G } _ { T } \leq \frac { \Delta _ { F } } { T \eta K } + \frac { 5 L \bar { \rho } ^ { 2 } } { 2 } ( K \eta ) + \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + L ^ { 2 } \bar { \rho } ^ { 2 } \eta ^ { 3 } + 2 \sigma ^ { 2 } \eta + \Big ( 3 - \frac { 1 } { K } \Big ) L \bar { \rho } ^ { 2 } K \eta ^ { 2 } + 2 \bar { \rho } \sigma \eta } } \\ { \displaystyle { \qquad + 2 \bar { \rho } \eta \left[ \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } + \frac { 4 \sqrt { 2 } \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K p } } + \frac { 8 \sigma } { p ( 1 - \beta _ { 2 } ) T } \right] } } \\ { \displaystyle { \qquad + 2 \bar { \rho } \left[ \frac { 8 \sqrt { 2 } L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } \right] + 2 \bar { \rho } \left[ \frac { 4 \sigma } { \sqrt { N ( 1 - \beta _ { 2 } ) T } } + \frac { 4 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K M } } \right] . } } \end{array}\tag{A.70}
$$

Step 4: evaluate the dominant terms under (A.63). Using $T \geq K$ so that $T + K \leq 2 T$ , we have

$$
\frac { 1 } { 1 - \beta _ { 2 } } = \sqrt { \frac { T + K } { N K } } \leq \sqrt { \frac { 2 T } { N K } } , \qquad \sqrt { 1 - \beta _ { 2 } } = \left( \frac { N K } { T + K } \right) ^ { 1 / 4 } \leq \left( \frac { N K } { T } \right) ^ { 1 / 4 } .
$$

Also, by (A.64),

$$
K \eta \le { T ^ { - 1 / 2 } } .
$$

For the drift contribution coming from $\sqrt { R _ { t } }$

$$
\frac { L K \eta } { p ( 1 - \beta _ { 2 } ) } = \frac { L } { p } \cdot \frac { ( N K ) ^ { 1 / 4 } } { T ^ { 3 / 4 } } \cdot \sqrt { \frac { T + K } { N K } } \le \frac { \sqrt { 2 } L } { p ( N K T ) ^ { 1 / 4 } } .\tag{A.71}
$$

For the averaged stochastic contribution coming from $\sqrt { V _ { t } }$

$$
\frac { \sigma } { \sqrt { N ( 1 - \beta _ { 2 } ) T } } = \sigma \sqrt { \frac { 1 } { N T } } \left( \frac { T + K } { N K } \right) ^ { 1 / 4 } \leq \frac { 2 ^ { 1 / 4 } \sigma } { ( N K T ) ^ { 1 / 4 } } ,\tag{A.72}
$$

$$
\frac { \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K M } } \le \sigma \cdot \frac { ( N K / T ) ^ { 1 / 4 } } { \sqrt { K M } } = \frac { \sigma \sqrt { p } } { ( N K T ) ^ { 1 / 4 } } \le \frac { \sigma } { ( N K T ) ^ { 1 / 4 } } .\tag{A.73}
$$

Hence the $\sqrt { V _ { t } }$ contribution produces a stochastic leading term of order $( N K T ) ^ { - 1 / 4 }$ with constant independent of M.

Step 5: lower-order terms. The two dominant contributions are therefore

$$
\frac { \Delta _ { F } } { T \eta K } = \frac { \Delta _ { F } } { ( N K T ) ^ { 1 / 4 } } , \qquad 2 \bar { \rho } \cdot \frac { 8 \sqrt 2 L K \bar { \rho } \eta } { p ( 1 - \beta _ { 2 } ) } \lesssim \frac { L \bar { \rho } ^ { 2 } / p } { ( N K T ) ^ { 1 / 4 } } ,
$$

and, by (A.72)–(A.73),

$$
2 \bar { \rho } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { V _ { t } } \lesssim \frac { \sigma \bar { \rho } } { ( N K T ) ^ { 1 / 4 } } .
$$

Moreover,

$$
\frac { 5 L \bar { \rho } ^ { 2 } } { 2 } ( K \eta ) \le \frac { 5 L \bar { \rho } ^ { 2 } } { 2 \sqrt { T } } ,
$$

which yields the $T ^ { - 1 / 2 }$ term. All remaining terms are lower-order in $( T , K )$ under the choice (A.63). Indeed,

$$
\eta = \frac { N ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } , \qquad K \eta ^ { 2 } = \frac { N ^ { 1 / 2 } } { K ^ { 1 / 2 } T ^ { 3 / 2 } } , \qquad \eta ^ { 3 } = \frac { N ^ { 3 / 4 } } { K ^ { 9 / 4 } T ^ { 9 / 4 } } ,
$$

In particular, the query-wise noise term is $2 \bar { \rho } \sigma \eta ,$ which is a constant multiple of the first term of $R _ { \mathrm { l o } } ( T , K )$ . It therefore enlarges only $C _ { 0 }$ , leaving $C _ { 1 }$ and $C _ { 2 }$ unchanged. and the terms coming from 2 $\textstyle { \mathcal { I } } \sum _ { t } { \sqrt { B _ { t } } }$ are also lower-order:

$$
\eta \cdot \frac { L K \eta } { p ( 1 - \beta _ { 2 } ) } \lesssim \frac { 1 } { K T } , \qquad \eta \cdot \frac { \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K p } } \lesssim \eta , \qquad \eta \cdot \frac { \sigma } { p ( 1 - \beta _ { 2 } ) T } \lesssim \frac { 1 } { N ^ { 1 / 4 } K ^ { 5 / 4 } T ^ { 5 / 4 } } ,
$$

all of which are absorbed by $R _ { \mathrm { l o } } ( T , K )$ after enlarging the constant $C _ { 0 }$ . Therefore, one may take

$$
C _ { 1 } : = \frac { 5 L \bar { \rho } ^ { 2 } } { 2 } ,\tag{A.74}
$$

$$
C _ { 2 } : = \Delta _ { F } + 6 4 \cdot \frac { L \bar { \rho } ^ { 2 } } { p } + 3 2 \cdot \sigma \bar { \rho } ,\tag{A.75}
$$

and obtain

$$
\mathcal { G } _ { T } \leq \frac { C _ { 1 } } { \sqrt { T } } + \frac { C _ { 2 } } { ( N K T ) ^ { 1 / 4 } } + C _ { 0 } R _ { \mathrm { l o } } ( T , K ) ,\tag{A.76}
$$

where the lower-order remainder can be taken explicitly as

$$
R _ { \mathrm { l o } } ( T , K ) : = \frac { N ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } + \frac { N ^ { 1 / 2 } } { K ^ { 1 / 2 } T ^ { 3 / 2 } } + \frac { N ^ { 3 / 4 } } { K ^ { 9 / 4 } T ^ { 9 / 4 } } + \frac { 1 } { N ^ { 1 / 4 } K ^ { 5 / 4 } T ^ { 3 / 4 } } + \frac { 1 } { \sqrt { N K T } } .\tag{A.77}
$$

Here $C _ { 0 }$ depends only on $( L , \bar { \rho } , \sigma , p )$ (and absolute numerical constants), and is independent of $( T , K )$ . This completes the proof. □

## A.10 Proof of Theorem 6

Proof. We show that inexact projection only adds the perturbation term stated in Theorem $6 ;$ the complexity statement then follows by specializing this perturbation to the hyperparameter choices in Theorem 5 and the well-tuned guarantee in Theorem A.1.

The proof is identical to the exact-projection proof except for the predict-and-correct estimate. Because the inexact solver always returns feasible update variables, the domain bound (A.3) and the round-movement bound $\left( \mathrm { { A . 4 } } \right)$ remain valid. Consequently, the smoothness descent inequality (A.23), the bounds for $E _ { t } ^ { ( b ) }$ and $E _ { t } ^ { ( c ) }$ , and all control-variate recursions are unchanged. In particular, the corrected query-wise noise bound (A.33) uses only feasibility and hence also applies to inexact projections; it invokes no independence between a projected update and its current noise.

The only afected term is $E _ { t } ^ { ( a ) }$ . Let

$$
\begin{array} { r } { \hat { \varepsilon } _ { i } ^ { t , k } : = \mathrm { r e s } _ { \mathcal { D } } \big ( \hat { z } _ { i } ^ { t , k } , \hat { \Delta } _ { i } ^ { t , k - 1 } \big ) , \qquad { \varepsilon } _ { i } ^ { t , k } : = \mathrm { r e s } _ { \mathcal { D } } ( z _ { i } ^ { t , k } , { \Delta } _ { i } ^ { t , k - 1 } ) . } \end{array}
$$

Applying Lemma A.7 with $u = u _ { t }$ and averaging over active clients gives

$$
\begin{array} { r l } & { E _ { t } ^ { ( a ) } \leq \frac { ( \bar { \rho } \eta ) ^ { 2 } } { 2 } + \displaystyle \frac { 2 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 2 } ^ { K } \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } + 2 K ( \bar { \rho } \eta ) ^ { 2 } } \\ & { \quad \quad \quad + \displaystyle \frac { 5 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \left( \hat { \varepsilon } _ { i } ^ { t , k } + \varepsilon _ { i } ^ { t , k } \right) . } \end{array}\tag{A.78}
$$

Taking expectation and using Lemma $\mathrm { A } . 8$ as in the exact-projection proof yields the same collection of non-residual terms as in the proofs of Theorem 5 and the well-tuned guarantee in Theorem $\mathrm { { A . 1 } }$ up to universal numerical constants. The extra contribution, after summing over $t = 1 , \dots , T$ and normalizing by $T \eta K$ , is

$$
\frac { 5 } { T \eta K } \sum _ { t = 1 } ^ { T } \mathbb { E } \Bigg [ \frac { 1 } { N } \sum _ { i \in S _ { t } } \sum _ { k = 1 } ^ { K } \left( \hat { \varepsilon } _ { i } ^ { t , k } + \varepsilon _ { i } ^ { t , k } \right) \Bigg ] = \frac { 5 \overline { { \mathfrak { R } } } _ { T } } { \eta K } .\tag{A.79}
$$

This proves that the exact-projection bounds acquire only the perturbation term ${ \mathcal { O } } ( { \overline { { \mathfrak { R } } } } _ { T } / ( \eta K ) )$ , with all other constants enlarged only by universal factors.

It remains to check that this perturbation does not alter the main complexity exponents under the residual budget $\overline { { \mathfrak { R } } } _ { T } = { \mathcal { O } } ( T ^ { - 1 } )$ . For the well-tuned regime in Theorem A.1, the update-budget radius is

$$
\eta = \mathrm { m i n } \left\{ 1 , \sqrt { \frac { \Delta _ { F } } { A _ { \mathrm { w t } } ( \gamma ) K T } } \right\} .
$$

If $\eta \ : = \ : 1$ , then $\mathrm { ( A . 7 9 ) }$ is $\mathcal { O } ( ( K T ) ^ { - 1 } )$ under $\overline { { \mathfrak { R } } } _ { T } ~ = ~ \mathcal { O } ( T ^ { - 1 } )$ . Otherwise, the perturbation is $\mathcal { O } \big ( \sqrt { A _ { \mathrm { w t } } ( \gamma ) / ( \Delta _ { F } K T ) } \big )$ . For fixed positive $\Delta _ { F }$ , it has the same $( T , K )$ dependence as the term $\sqrt { A _ { \mathrm { w t } } ( \gamma ) \Delta _ { F } / ( K T ) }$ in the well-tuned bound; here $A _ { \mathrm { w t } } ( \gamma )$ retains its stated dependence on K. Thus the well-tuned communication and sample-complexity exponents are unchanged.

For the problem-parameter-free regime of Theorem 5,

$$
\eta = \frac { N ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } , \qquad T \eta K = ( N K T ) ^ { 1 / 4 } .
$$

Hence $\overline { { \mathfrak { R } } } _ { T } = { \mathcal { O } } ( T ^ { - 1 } )$ implies

$$
\frac { \overline { { \mathfrak { R } } } _ { T } } { \eta K } = \mathcal { O } \Bigl ( \frac { 1 } { T \eta K } \Bigr ) = \mathcal { O } \Bigl ( ( N K T ) ^ { - 1 / 4 } \Bigr ) ,
$$

which has the same order as the second leading term in Theorem 5. Therefore the problem-parameterfree communication and sample-complexity exponents are unchanged as well. □

## B Decentralized-learning analysis

Exact initialization and its cost. All nodes start from the common model $x _ { \mathrm { i n i t } }$ and form $c _ { i } ^ { 1 } = \nabla f _ { i } ( x _ { \mathrm { i n i t } } , \xi _ { i } ^ { - 1 } )$ using one stochastic-gradient query each. To initialize $\begin{array} { r } { m _ { i } ^ { 1 } = \bar { c } ^ { 1 } : = M ^ { - 1 } \sum _ { j } c _ { j } ^ { 1 } } \end{array}$ exactly, we use a one-time averaging primitive before the $T$ training rounds. On the static connected support graph of $W$ , a reduce-and-broadcast on a precomputed breadth-first spanning tree of depth h takes 2h communication rounds and transmits $2 ( M - 1 )$ parameter-sized vectors. This is separate from repeated W-mixing, which need not reach exact consensus in finitely many rounds. Under FTC, one τ-round mixing block gives the exact average; the subsequent valid mixing sequence is indexed from $t = 1$ . Both procedures ensure $D _ { 1 } = 0$ , and Assumption 2 gives $B _ { 1 } \leq \sigma ^ { 2 }$ with one query per node. The overall counts are $T + 2 h$ or $T + \tau$ communication rounds and $K T + 1$ gradient queries per node, hence $M K T + M$ in total. These additive initialization costs do not change the accuracy exponents for a fixed network.

## B.1 Formal DL guarantees

Theorem B.1 (NBCLT-DL under static gossip matrix). Suppose that Assumptions $1 \mathrm { - } \mathit { 4 }$ hold, and that the mixing matrix $\mathbf { W } \in \mathbb { R } ^ { M \times M }$ is fixed, symmetric, and doubly stochastic. Define $\begin{array} { r } { \mathbf { J } = \frac { 1 } { M } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ and $\lambda : = \| \mathbf { W } - \mathbf { J } \| _ { 2 } \in [ 0 , 1 )$ . Use the one-query consensual initialization and exact tree averaging described in Appendix B. Run Algorithm 2 for T training rounds with K local updates per round, and choose

$$
\eta = \frac { M ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } , \qquad 1 - \beta _ { 1 } = \eta , \qquad 1 - \beta _ { 2 } = \sqrt { \frac { M K } { T + K } } ,\tag{B.1}
$$

Define $\begin{array} { r } { \bar { x } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } x _ { i } ^ { t } } \end{array}$ and $\Delta _ { F } : = F ( \bar { x } ^ { 1 } ) - F ^ { \star }$ . Then for all $T \geq M K$ , we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \Big ] \ \leq \ \frac { C _ { 1 } } { \sqrt { T } } + \frac { C _ { 2 } } { ( M K T ) ^ { 1 / 4 } } + C _ { 0 } R _ { \mathrm { l o } } ( T , K ) ,\tag{B.2}
$$

where

$$
C _ { 1 } = L \bar { \rho } ^ { 2 } \Bigg ( \frac { 5 } { 2 } + \frac { 2 } { 1 - \lambda } + \frac { 3 2 \sqrt { 3 } + 4 \sqrt { 1 5 } } { ( 1 - \lambda ) ^ { 2 } } \Bigg ) + \frac { 2 ( \sqrt { 6 } + \sqrt { 3 } ) \bar { \rho } \sigma \sqrt { M } } { 1 - \lambda } ,
$$

$$
C _ { 2 } = \Delta _ { F } + \Bigl ( \frac { 5 } { 2 } + L ^ { 2 } \Bigr ) \bar { \rho } ^ { 2 } + 2 \sigma ^ { 2 } + \frac { 3 2 \sqrt { 2 } L \bar { \rho } ^ { 2 } } { 1 - \lambda } + 2 \sqrt { 2 } \bar { \rho } \sigma \sqrt { M } .
$$

The lower-order remainder terms are

$$
R _ { \mathrm { l o } } ( T , K ) = \frac { 1 } { K \sqrt { T } } + \frac { 1 } { \sqrt { K T } } + \frac { 1 } { K ^ { 5 / 4 } T ^ { 3 / 4 } } + \frac { 1 } { K T } + \frac { 1 } { T } ,\tag{B.3}
$$

and the constant $C _ { 0 }$ may depend on $( L , \sigma , \bar { \rho } , M , \lambda , \Delta _ { F } )$ but is independent of $( T , K )$ .

Theorem B.2 (Convergence of NBCLT-DL under FTC). Suppose Assumptions $1 \mathrm { - } \mathit { 4 }$ hold. Let $\{ W ^ { t } \} _ { t \geq 1 }$ be doubly stochastic and satisfy the finite-time consensus $( F T C )$ property (8) with horizon $\tau \geq 1$ . Assume consensual initialization $x _ { i } ^ { 1 } = \bar { x } ^ { 1 }$ for all $i ,$ and initialize trackers by $c _ { i } ^ { 1 } = \nabla f _ { i } ( x _ { i } ^ { 1 } , \xi _ { i } ^ { - 1 } )$ using one query per node, followed by one τ -round FTC block to obtain $\begin{array} { r } { m _ { i } ^ { 1 } = M ^ { - 1 } \sum _ { j = 1 } ^ { M } c _ { j } ^ { 1 } } \end{array}$ exactly, as in Appendix B. Run Algorithm 2 for T rounds with K local steps and choose

$$
\eta = \frac { M ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } , \qquad 1 - \beta _ { 1 } = \eta , \qquad 1 - \beta _ { 2 } = \sqrt { \frac { M K } { T + K } } .\tag{B.4}
$$

Let $\Delta _ { F } : = F ( \bar { x } ^ { 1 } ) - F ^ { \star }$ , and let $\begin{array} { r } { B _ { 1 } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| c _ { i } ^ { 1 } - \nabla f _ { i } ( x _ { i } ^ { 1 } ) \| _ { 2 } ^ { 2 } } \end{array}$ . Under the stated initialization, $B _ { 1 } \leq \sigma ^ { 2 }$ . For any $T \geq M K$ , we have

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \leq \frac { C _ { 1 } } { \sqrt { T } } + \frac { C _ { 2 } } { ( M K T ) ^ { 1 / 4 } } + C _ { 0 } \mathcal { R } _ { \mathrm { l o } } ,\tag{B.5}
$$

where

$$
C _ { 1 } = L \bar { \rho } ^ { 2 } \Big ( \frac { 5 } { 2 } + 2 \tau + ( 3 2 \sqrt { 3 } + 4 \sqrt { 1 5 } ) \tau \sqrt { 1 + \tau ^ { 2 } } \Big ) + 2 ( \sqrt { 6 } + \sqrt { 3 } ) \tau \bar { \rho } \sigma \sqrt { M } ,
$$

$$
C _ { 2 } = \Delta _ { F } + \Bigl ( \frac { 5 } { 2 } + L ^ { 2 } \Bigr ) \bar { \rho } ^ { 2 } + 2 \sigma ^ { 2 } + 3 2 \sqrt { 2 } L \bar { \rho } ^ { 2 } \sqrt { 1 + \tau ^ { 2 } } + 2 \sqrt { 2 } \bar { \rho } \sigma \sqrt { M } .\tag{B.6}
$$

The lower-order remainder terms are

$$
\mathcal { R } _ { \mathrm { l o } } : = \frac { M ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } + \frac { \sqrt { M } } { \sqrt { K } T ^ { 3 / 2 } } + \frac { \sqrt { M } } { K T } + \frac { 1 } { M ^ { 1 / 4 } K ^ { 5 / 4 } T ^ { 3 / 4 } } + \frac { 1 } { \sqrt { M K T } } ,\tag{B.7}
$$

and the constant $C _ { 0 }$ is independent of $( T , K , M )$ .

## B.2 Notation and preliminaries

Let $\begin{array} { r } { F ( x ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } f _ { i } ( x ) } \end{array}$ . Algorithm 2 maintains $( x _ { i } ^ { t } , c _ { i } ^ { t } , m _ { i } ^ { t } ) \in ( \mathbb R ^ { d } ) ^ { 3 }$ for each node $i \in [ M ]$ . Define the averages

$$
{ \bar { x } } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } x _ { i } ^ { t } , \qquad { \bar { c } } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } c _ { i } ^ { t } , \qquad { \bar { m } } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } m _ { i } ^ { t } .
$$

Local iterates. For convenience, set $x _ { i } ^ { 0 } : = x _ { i } ^ { 1 }$ . In round t, define $x _ { i } ^ { t , 0 } : = x _ { i } ^ { t }$ and for $k = 1 , \ldots , K$ 2

$$
\Delta _ { i } ^ { t , k - 1 } : = x _ { i } ^ { t , k } - x _ { i } ^ { t , k - 1 } , \qquad \Delta _ { i } ^ { t } : = \sum _ { k = 1 } ^ { K } \Delta _ { i } ^ { t , k - 1 } = x _ { i } ^ { t , k } - x _ { i } ^ { t , 0 } .
$$

Mixing and spectral gap. Assume W $\in \mathbb { R } ^ { M \times M }$ is fixed, symmetric and doubly stochastic. Let $\begin{array} { r } { \mathbf { J } : = \frac { 1 } { M } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ and $\lambda : = \| \mathbf { W } - \mathbf { J } \| _ { 2 } \in [ 0 , 1 )$ . The spectral gap is $1 - \lambda$ . Mixing is

$$
x _ { i } ^ { t + 1 } = \sum _ { j = 1 } ^ { M } w _ { i j } x _ { j } ^ { t , K } , \qquad m _ { i } ^ { t + 1 } = \sum _ { j = 1 } ^ { M } w _ { i j } \big ( m _ { j } ^ { t } + c _ { j } ^ { t + 1 } - c _ { j } ^ { t } \big ) .
$$

By double stochasticity, mixing preserves the average:

$$
\bar { x } ^ { t + 1 } = \frac 1 M \sum _ { i = 1 } ^ { M } x _ { i } ^ { t + 1 } = \frac 1 M \sum _ { j = 1 } ^ { M } x _ { j } ^ { t , K } = \bar { x } ^ { t } + \frac 1 M \sum _ { j = 1 } ^ { M } \Delta _ { j } ^ { t } .\tag{B.8}
$$

Similarly,

$$
\bar { m } ^ { t + 1 } = \bar { m } ^ { t } + \bar { c } ^ { t + 1 } - \bar { c } ^ { t } .\tag{B.9}
$$

Projection operator and domain bound. Let $\Pi _ { \mathcal { D } } ( \cdot )$ be Euclidean projection onto $\mathcal { D } = \{ \Delta$ $\| \Delta \| ^ { g } \leq \eta \}$ . By Assumption 4, for any $\Delta \in \mathcal { D }$

$$
\| \Delta \| _ { 2 } \le \bar { \rho } \| \Delta \| ^ { g } \le \bar { \rho } \eta .\tag{B.10}
$$

Hence $\lVert \Delta _ { i } ^ { t , k - 1 } \rVert _ { 2 } \leq \bar { \rho } \eta$ and

$$
\| \bar { x } ^ { t + 1 } - \bar { x } ^ { t } \| _ { 2 } \leq \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| \Delta _ { i } ^ { t } \| _ { 2 } \leq \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { K } \| \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } \leq K \bar { \rho } \eta .\tag{B.11}
$$

Stochastic gradients. Assume conditional unbiasedness and bounded conditional variance as in Assumption 2, and conditional independence across gradient queries.

Error measures. We quantify the accuracy of the local control variates, the disagreement of the gradient trackers, and the one-step model transition by

$$
\begin{array} { r l } & { { \cal B } _ { t } : = \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| c _ { i } ^ { t } - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) \| _ { 2 } ^ { 2 } , } \\ & { { \cal D } _ { t } : = \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| m _ { i } ^ { t } - \bar { m } ^ { t } \| _ { 2 } ^ { 2 } , } \\ & { { \cal R } _ { t } : = \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - x _ { i } ^ { t - 1 } \| _ { 2 } ^ { 2 } . } \end{array}\tag{B.12}
$$

where $\begin{array} { r } { \bar { m } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } m _ { i } ^ { t } } \end{array}$ . Note that, by the convention $x _ { i } ^ { 0 } = x _ { i } ^ { 1 }$ , this definition agrees with the usual initialization quantity $B _ { 1 }$

## B.3 Auxiliary lemmas

Lemma B.3 (L-smoothness descent on $\bar { x } ^ { t } )$ . If F is L-smooth, then for all $t ,$

$$
\begin{array} { r l } & { F ( \bar { x } ^ { t + 1 } ) \le F ( \bar { x } ^ { t } ) + \langle \nabla F ( \bar { x } ^ { t } ) , \bar { x } ^ { t + 1 } - \bar { x } ^ { t } \rangle + \displaystyle \frac { L } { 2 } \| \bar { x } ^ { t + 1 } - \bar { x } ^ { t } \| _ { 2 } ^ { 2 } } \\ & { \qquad \le F ( \bar { x } ^ { t } ) + \langle \nabla F ( \bar { x } ^ { t } ) , \bar { x } ^ { t + 1 } - \bar { x } ^ { t } \rangle + \displaystyle \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } . } \end{array}\tag{B.13}
$$

Lemma B.4 (Support function of D). Let $\| \cdot \| ^ { g , * }$ be the dual norm and define

$$
u _ { t } \in \arg \operatorname* { m i n } _ { u \in \mathcal { D } } \langle \nabla F ( \bar { x } ^ { t } ) , u \rangle .
$$

Then

$$
\begin{array} { r } { \langle \nabla F ( \bar { x } ^ { t } ) , u _ { t } \rangle = - \eta \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } . } \end{array}\tag{B.14}
$$

Lemma B.5 (Projection three-point inequality). Let $\Pi _ { \mathcal { D } } ( \cdot )$ be Euclidean projection onto a nonempty closed convex set D. Let $y = \Pi _ { \mathcal { D } } ( x - g )$ . Then for any $u \in \mathcal { D }$ 2

$$
\langle g , y - u \rangle \leq \frac { 1 } { 2 } \Big ( \| u - x \| _ { 2 } ^ { 2 } - \| u - y \| _ { 2 } ^ { 2 } - \| y - x \| _ { 2 } ^ { 2 } \Big ) .
$$

Lemma B.6 (One-gradient predict-and-correct inequality). Fix (t, i) and set $\hat { \Delta } _ { i } ^ { t , - 1 } = 0$ . For $k = 1 , \ldots , K$ , Algorithm 2 generates

$$
\begin{array} { r } { \hat { \Delta } _ { i } ^ { t , k - 1 } = \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 2 } - g _ { i } ^ { t , k } ) , \qquad { \Delta } _ { i } ^ { t , k - 1 } = \Pi _ { \mathcal { D } } ( \hat { \Delta } _ { i } ^ { t , k - 1 } - g _ { i } ^ { t , k } ) . } \end{array}
$$

Then for any $u \in \mathcal { D }$

$$
\sum _ { k = 1 } ^ { K } \langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u \rangle \leq \frac { \| u \| _ { 2 } ^ { 2 } } { 2 } + \frac { 1 } { 2 } \sum _ { k = 2 } ^ { K } \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } + 2 K ( \bar { \rho } \eta ) ^ { 2 } .\tag{B.15}
$$

Lemma B.7 (Bounding gradient variation). In Algorithm 2, for any $( t , i , k \geq 2 )$

$$
\begin{array} { r } { \mathbb { E } \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } \leq 2 ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \mathbb { E } \| x _ { i } ^ { t , k - 1 } - x _ { i } ^ { t , k - 2 } \| _ { 2 } ^ { 2 } + 4 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } . } \end{array}\tag{B.16}
$$

Proofs omitted. The proofs of Lemmas B.5, B.6, and B.7 are identical to their federated-learning counterparts (Lemmas A.4, A.5, and A.8, respectively), and are therefore omitted.

Lemma B.8 (Static consensus bound for $C _ { t } )$ . Let

$$
C _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - \bar { x } ^ { t } \| _ { 2 } ^ { 2 } , \qquad \bar { x } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } x _ { i } ^ { t } .
$$

Assume the mixing matrix W is fixed, symmetric, doubly stochastic, and satisfies $\lambda : = \| \mathbf { W } - \mathbf { J } \| _ { 2 } < 1$ where $\begin{array} { r } { \mathbf { J } : = \frac { 1 } { M } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ . Assume also that each round satisfies the per-node movement bound

$$
\| \boldsymbol { x } _ { i } ^ { t , k } - \boldsymbol { x } _ { i } ^ { t } \| _ { 2 } \leq K \bar { \rho } \eta , \qquad \forall i \in [ M ] .
$$

Then for all $t \geq 1$

$$
\sqrt { C _ { t + 1 } } \leq \lambda \sqrt { C _ { t } } + \lambda K \bar { \rho } \eta ,
$$

and consequently,

$$
\sqrt { C _ { t } } \le \lambda ^ { t - 1 } \sqrt { C _ { 1 } } + \frac { \lambda } { 1 - \lambda } K \bar { \rho } \eta \le \sqrt { C _ { 1 } } + \frac { K \bar { \rho } \eta } { 1 - \lambda } .
$$

In particular, if the initialization is consensual $( e . g . , x _ { i } ^ { 1 } = x ^ { 1 }$ for all i so that $C _ { 1 } = 0 )$ , then

$$
\sqrt { C _ { t } } \le \frac { \lambda } { 1 - \lambda } K \bar { \rho } \eta \le \frac { K \bar { \rho } \eta } { 1 - \lambda } .
$$

Proof. Stack the node variables row-wise into the matrix $\mathbf { X } ^ { t } \in \mathbb { R } ^ { M \times d }$ whose i-th row is $( x _ { i } ^ { t } ) ^ { \top }$ . Define the averaging projector $\begin{array} { r } { \mathbf { J } : = \frac { 1 } { M } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ and the centered (disagreement) component

$$
\begin{array} { r } { \tilde { \mathbf { X } } ^ { t } : = ( \mathbf { I } - \mathbf { J } ) \mathbf { X } ^ { t } . } \end{array}
$$

Then

$$
C _ { t } = \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { X } } ^ { t } \| _ { F } ^ { 2 } .
$$

Step 1: relate $\tilde { \mathbf { X } } ^ { t + 1 }$ to $\tilde { \mathbf { X } } ^ { t , K }$ . In the DL algorithm, after K local updates, mixing gives

$$
\mathbf { X } ^ { t + 1 } = \mathbf { W } \mathbf { X } ^ { t , K } .
$$

Using double stochasticity, we have $\mathbf { J } \mathbf { W } = \mathbf { W } \mathbf { J } = \mathbf { J }$ , hence $( { \bf I } - { \bf J } ) { \bf W } = { \bf W } - { \bf J }$ . Therefore,

$$
\begin{array} { r } { \tilde { \mathbf { X } } ^ { t + 1 } = ( \mathbf { I } - \mathbf { J } ) \mathbf { X } ^ { t + 1 } = ( \mathbf { I } - \mathbf { J } ) \mathbf { W } \mathbf { X } ^ { t , K } = ( \mathbf { W } - \mathbf { J } ) \mathbf { X } ^ { t , K } = ( \mathbf { W } - \mathbf { J } ) \tilde { \mathbf { X } } ^ { t , K } , } \end{array}
$$

where the last equality uses $( \mathbf { W } - \mathbf { J } ) \mathbf { J } = \mathbf { 0 }$

Step 2: contract under mixing. By definition of $\lambda = \lVert \mathbf { W } - \mathbf { J } \rVert _ { 2 }$ and the fact that $\| \mathbf { A Z } \| _ { F } \leq$ $\| \mathbf { A } \| _ { 2 } \| \mathbf { Z } \| _ { F }$ , we obtain

$$
\| \tilde { \mathbf { X } } ^ { t + 1 } \| _ { F } = \| ( \mathbf { W } - \mathbf { J } ) \tilde { \mathbf { X } } ^ { t , K } \| _ { F } \leq \| \mathbf { W } - \mathbf { J } \| _ { 2 } \cdot \| \tilde { \mathbf { X } } ^ { t , K } \| _ { F } \leq \lambda \| \tilde { \mathbf { X } } ^ { t , K } \| _ { F } .
$$

Taking expectation and using Jensen’s inequality,

$$
\sqrt { \mathbb { E } \| \tilde { \mathbf { X } } ^ { t + 1 } \| _ { F } ^ { 2 } } \le \lambda \sqrt { \mathbb { E } \| \tilde { \mathbf { X } } ^ { t , K } \| _ { F } ^ { 2 } } .
$$

Dividing by $\sqrt { M }$ yields

$$
\sqrt { C _ { t + 1 } } \leq \lambda \sqrt { \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { X } } ^ { t , K } \| _ { F } ^ { 2 } } .\tag{B.17}
$$

Step 3: bound the injected disagreement from local steps. Write the local-update increment matrix $\Delta ^ { t } : = \mathbf { X } ^ { t , K } - \mathbf { X } ^ { t }$ (row i equals $( x _ { i } ^ { t , k } - x _ { i } ^ { t } ) ^ { \top } )$ . Then

$$
\tilde { \mathbf { X } } ^ { t , K } = ( \mathbf { I } - \mathbf { J } ) \mathbf { X } ^ { t , K } = ( \mathbf { I } - \mathbf { J } ) ( \mathbf { X } ^ { t } + \mathbf { A } ^ { t } ) = \tilde { \mathbf { X } } ^ { t } + ( \mathbf { I } - \mathbf { J } ) \Delta ^ { t } .
$$

Hence by triangle inequality and $\| \mathbf { I } - \mathbf { J } \| _ { 2 } \leq 1$

$$
\| \tilde { \mathbf { X } } ^ { t , K } \| _ { F } \leq \| \tilde { \mathbf { X } } ^ { t } \| _ { F } + \| ( \mathbf { I } - \mathbf { J } ) \mathbf { \Delta } ^ { t } \| _ { F } \leq \| \tilde { \mathbf { X } } ^ { t } \| _ { F } + \| \mathbf { \Delta } ^ { t } \| _ { F } .
$$

Moreover, using the assumed per-node movement bound,

$$
\| \boldsymbol { \Delta } ^ { t } \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { M } \| \boldsymbol { x } _ { i } ^ { t , k } - \boldsymbol { x } _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq \sum _ { i = 1 } ^ { M } ( K \bar { \rho } \eta ) ^ { 2 } = M ( K \bar { \rho } \eta ) ^ { 2 } ,
$$

so $\| \Delta ^ { t } \| _ { F } \le \sqrt { M } K \bar { \rho } \eta$ . Taking expectation and again using Jensen gives

$$
\sqrt { \mathbb { E } \| \tilde { \mathbf { X } } ^ { t , K } \| _ { F } ^ { 2 } } \leq \sqrt { \mathbb { E } \| \tilde { \mathbf { X } } ^ { t } \| _ { F } ^ { 2 } } + \sqrt { \mathbb { E } \| \Delta ^ { t } \| _ { F } ^ { 2 } } \leq \sqrt { \mathbb { E } \| \tilde { \mathbf { X } } ^ { t } \| _ { F } ^ { 2 } } + \sqrt { M } K \bar { \rho } \eta .
$$

Divide by $\sqrt { M }$ to obtain

$$
\sqrt { \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { X } } ^ { t , K } \| _ { F } ^ { 2 } } \leq \sqrt { C _ { t } } + K \bar { \rho } \eta .\tag{B.18}
$$

Step 4: combine and unroll. Plugging (B.18) into (B.17) yields

$$
\sqrt { C _ { t + 1 } } \leq \lambda \bigl ( \sqrt { C _ { t } } + K \bar { \rho } \eta \bigr ) = \lambda \sqrt { C _ { t } } + \lambda K \bar { \rho } \eta .
$$

Unrolling this recursion gives

$$
\sqrt { C _ { t } } \le \lambda ^ { t - 1 } \sqrt { C _ { 1 } } + \lambda K \bar { \rho } \eta \sum _ { s = 0 } ^ { t - 2 } \lambda ^ { s } = \lambda ^ { t - 1 } \sqrt { C _ { 1 } } + \frac { \lambda ( 1 - \lambda ^ { t - 1 } ) } { 1 - \lambda } K \bar { \rho } \eta \le \lambda ^ { t - 1 } \sqrt { C _ { 1 } } + \frac { \lambda } { 1 - \lambda } K \bar { \rho } \eta .
$$

Finally, since $\lambda ^ { t - 1 } \leq 1$ and $\lambda \leq 1$ , we also have

$$
\sqrt { C _ { t } } \le \sqrt { C _ { 1 } } + \frac { K \bar { \rho } \eta } { 1 - \lambda } .
$$

This completes the proof.

Lemma B.9 (One-step model transition under static gossip). Under the assumptions of Lemma B.8, for all $t \geq 2$

$$
R _ { t } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - x _ { i } ^ { t - 1 } \| _ { 2 } ^ { 2 } \leq \frac { 9 ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } .\tag{B.19}
$$

Proof. For each $i ,$ write

$$
x _ { i } ^ { t } - x _ { i } ^ { t - 1 } = ( x _ { i } ^ { t } - \bar { x } ^ { t } ) + ( \bar { x } ^ { t } - \bar { x } ^ { t - 1 } ) + ( \bar { x } ^ { t - 1 } - x _ { i } ^ { t - 1 } ) .
$$

Using $\| a + b + c \| _ { 2 } ^ { 2 } \leq 3 ( \| a \| _ { 2 } ^ { 2 } + \| b \| _ { 2 } ^ { 2 } + \| c \| _ { 2 } ^ { 2 } )$ , averaging over $i ,$ and taking expectation yield

$$
R _ { t } \leq 3 C _ { t } + 3 C _ { t - 1 } + 3 \mathbb { E } \Vert \bar { x } ^ { t } - \bar { x } ^ { t - 1 } \Vert _ { 2 } ^ { 2 } .
$$

By (B.11),

$$
\begin{array} { r } { \mathbb { E } \| \bar { x } ^ { t } - \bar { x } ^ { t - 1 } \| _ { 2 } ^ { 2 } \leq ( K \bar { \rho } \eta ) ^ { 2 } . } \end{array}
$$

Moreover, Lemma B.8 with consensual initialization gives

$$
C _ { t } \le \frac { ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } , \qquad C _ { t - 1 } \le \frac { ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } .
$$

Since $( 1 - \lambda ) ^ { 2 } \leq 1$ , we conclude

$$
R _ { t } \le 6 \frac { ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } + 3 ( K \bar { \rho } \eta ) ^ { 2 } \le \frac { 9 ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } .
$$

This proves (B.19).

## B.4 Descent decomposition

Define $\begin{array} { r } { u _ { t } \in \arg \operatorname* { m i n } _ { u \in \mathcal { D } } \langle \nabla F ( \bar { x } ^ { t } ) , u \rangle } \end{array}$ . Using Lemma B.3 and (B.8),

$$
\begin{array} { r l } & { \displaystyle F ( \bar { x } ^ { t + 1 } ) \leq F ( \bar { x } ^ { t } ) + \left. \nabla F ( \bar { x } ^ { t } ) , \frac { 1 } { M } \underset { i = 1 } { \overset { M } { \sum } } \underset { k = 1 } { \overset { K } { \sum } } \Delta _ { i } ^ { t , k - 1 } \right. + \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } \\ & { \quad \quad \quad = F ( \bar { x } ^ { t } ) + K \langle \nabla F ( \bar { x } ^ { t } ) , u _ { t } \rangle + \underbrace { \frac { 1 } { M } \underset { i = 1 } { \overset { M } { \sum } } \underset { k = 1 } { \overset { K } { \sum } } \langle \nabla F ( \bar { x } ^ { t } ) , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle } _ { = : E _ { t } } + \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } . } \end{array}\tag{B.20}
$$

By Lemma B.4, $K \langle \nabla F ( \bar { x } ^ { t } ) , u _ { t } \rangle = - \eta K \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * }$ . Rearranging gives

$$
\eta K \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \leq F ( \bar { x } ^ { t } ) - F ( \bar { x } ^ { t + 1 } ) + \frac { L } { 2 } ( K \bar { \rho } \eta ) ^ { 2 } + E _ { t } .\tag{B.21}
$$

## B.5 Bounding $E _ { t }$ via predict-and-correct

Recall the local signal

$$
\begin{array} { r } { { g _ { i } ^ { t , k } = ( 1 - \beta _ { 1 } ) \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \big ) + m _ { i } ^ { t } } . } \end{array}
$$

For each (i, k),

$$
\begin{array} { r l } & { \langle \nabla F ( \bar { x } ^ { t } ) , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle = \langle m _ { i } ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle + \langle \nabla F ( \bar { x } ^ { t } ) - m _ { i } ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle } \\ & { \qquad = \langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle - ( 1 - \beta _ { 1 } ) \langle \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle } \\ & { \qquad + \langle \nabla F ( \bar { x } ^ { t } ) - m _ { i } ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle . } \end{array}
$$

Summing over $i \in [ M ] , k \in [ K ]$ and dividing by M yields

$$
E _ { t } = E _ { t } ^ { ( a ) } + E _ { t } ^ { ( b ) } + E _ { t } ^ { ( c ) } ,
$$

where

$$
E _ { t } ^ { ( a ) } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { K } \langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle ,\tag{B.22}
$$

$$
E _ { t } ^ { ( b ) } : = - ( 1 - \beta _ { 1 } ) \cdot \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { K } \langle \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle ,\tag{B.23}
$$

$$
E _ { t } ^ { ( c ) } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { K } \langle \nabla F ( \bar { x } ^ { t } ) - m _ { i } ^ { t } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle .\tag{B.24}
$$

Step 1: bound $E _ { t } ^ { ( a ) }$ (predict-and-correct). We apply the one-gradient predict-and-correct inequality in Lemma B.6 to each local sequence $\{ \Delta _ { i } ^ { t , k - 1 } \} _ { k = 1 } ^ { K }$ with the reference point chosen as $u = u _ { t }$ This yields, for each node i, an upper bound on the cumulative inner products ${ \textstyle \sum _ { k = 1 } ^ { K } } \langle g _ { i } ^ { t , k } , \Delta _ { i } ^ { t , k - 1 } - u _ { t } \rangle$ in terms of the radius of the domain D and the temporal variation of the local gradient signals $\{ g _ { i } ^ { t , k } \} _ { k = 1 } ^ { K }$

Averaging over all nodes and using $\| u _ { t } \| _ { 2 } \le \bar { \rho } \eta$ together with the diameter bound $\| \Delta _ { i } ^ { t , k - 1 } \| _ { 2 } \le \bar { \rho } \eta$ we obtain the deterministic radius contribution of order $\bar { \rho } ^ { 2 } \eta$ . Moreover, the gradient-variation term $\begin{array} { r } { \sum _ { k = 2 } ^ { K } \| g _ { i } ^ { t , k } - g _ { i } ^ { t , k - 1 } \| _ { 2 } ^ { 2 } } \end{array}$ is controlled by Lemma B.7, and the local iterate diference $\| x _ { i } ^ { t , k - 1 } - x _ { i } ^ { t , k - 2 } \| _ { 2 }$ appearing there is bounded by $\lVert \Delta _ { i } ^ { t , k - 2 } \rVert _ { 2 } \leq \bar { \rho } \eta$ due to the projection constraint.

Collecting these bounds and summing over $i = 1 , \dots , M$ , we obtain

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( a ) } ] } { \eta K } \le \Big ( 2 + \frac { 1 } { 2 K } \Big ) \hat \rho ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \hat \rho ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } .\tag{B.25}
$$

Step 2: bound $E _ { t } ^ { ( b ) }$ (tracking + noise). Define the gradient noise $\zeta _ { i } ^ { t , k - 1 } : = \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } )$ $\nabla f _ { i } ( x _ { i } ^ { t , k - 1 } )$ . Using $\| \Delta _ { i } ^ { t , k - 1 } - u _ { t } \| _ { 2 } \leq 2 \bar { \rho } \eta$ and $\mathbb { E } \| \zeta _ { i } ^ { t , k - 1 } \| _ { 2 } \leq \sigma$ , apply Cauchy–Schwarz to each query exactly as in (A.33), now with full participation. No martingale cancellation of the noise-dependent projected multipliers is used. We obtain

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( b ) } ] } { \eta K } \leq \frac { 2 \bar { \rho } ( 1 - \beta _ { 1 } ) } { M K } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { K } \mathbb { E } \| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \| _ { 2 } + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma .\tag{B.26}
$$

Insert the stale target $\nabla f _ { i } ( x _ { i } ^ { t - 1 } )$ :

$$
\| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \leq \| \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } + L \| x _ { i } ^ { t , k - 1 } - x _ { i } ^ { t - 1 } \| _ { 2 } .
$$

Moreover,

$$
\| { \boldsymbol x } _ { i } ^ { t , k - 1 } - { \boldsymbol x } _ { i } ^ { t - 1 } \| _ { 2 } \le \| { \boldsymbol x } _ { i } ^ { t } - { \boldsymbol x } _ { i } ^ { t - 1 } \| _ { 2 } + ( k - 1 ) \bar { \rho } \eta .
$$

Therefore,

$$
\begin{array} { r l } & { \frac { 1 } { M K } \displaystyle \sum _ { i = 1 } ^ { M } \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } \| \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \leq \frac { 1 } { M } \displaystyle \sum _ { i = 1 } ^ { M } \mathbb { E } \| \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } + L \cdot \frac { 1 } { M } \displaystyle \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - x _ { i } ^ { t - 1 } \| _ { 2 } } \\ & { \qquad + L \bar { \rho } \eta \cdot \frac { K - 1 } { 2 } . } \end{array}
$$

Using Jensen and the definitions of $B _ { t }$ and $R _ { t }$ ,

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) - c _ { i } ^ { t } \| _ { 2 } \leq \sqrt { B _ { t } } , \qquad \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - x _ { i } ^ { t - 1 } \| _ { 2 } \leq \sqrt { R _ { t } } .
$$

Plugging these estimates into (B.26) yields

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( b ) } ] } { \eta K } \le 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { B _ { t } } + 2 L \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { R _ { t } } + \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) \eta ( K - 1 ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma .\tag{B.27}
$$

Step 3: bound $E _ { t } ^ { ( c ) }$ (tracker mismatch). By (B.10), $\| \Delta _ { i } ^ { t , k - 1 } - u _ { t } \| _ { 2 } \leq 2 \bar { \rho } \eta$ . Hence

$$
| E _ { t } ^ { ( c ) } | \leq \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { k = 1 } ^ { K } \| \nabla F ( \bar { x } ^ { t } ) - m _ { i } ^ { t } \| _ { 2 } \cdot 2 \bar { \rho } \eta = 2 \bar { \rho } \eta K \cdot \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| \nabla F ( \bar { x } ^ { t } ) - m _ { i } ^ { t } \| _ { 2 } .
$$

Dividing by $\eta K$ and taking expectation gives

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( c ) } ] } { \eta K } \leq 2 \bar { \rho } \mathbb { E } \Big [ \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| \nabla F ( \bar { x } ^ { t } ) - m _ { i } ^ { t } \| _ { 2 } \Big ] .\tag{B.28}
$$

Insert and subtract $\bar { m } ^ { t }$ :

$$
\| \nabla F ( \bar { x } ^ { t } ) - m _ { i } ^ { t } \| _ { 2 } \leq \| \nabla F ( \bar { x } ^ { t } ) - \bar { m } ^ { t } \| _ { 2 } + \| \bar { m } ^ { t } - m _ { i } ^ { t } \| _ { 2 } .
$$

For the disagreement term, Jensen yields

$$
\mathbb { E } \big [ \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \| \bar { m } ^ { t } - m _ { i } ^ { t } \| _ { 2 } \big ] \leq \sqrt { D _ { t } } .
$$

For the bias term, using $\begin{array} { r } { \nabla F ( \bar { x } ^ { t } ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \nabla f _ { j } ( \bar { x } ^ { t } ) } \end{array}$ and Jensen,

$$
\| \nabla F ( \bar { x } ^ { t } ) - \bar { m } ^ { t } \| _ { 2 } = \Big \| \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \big ( \nabla f _ { j } ( \bar { x } ^ { t } ) - m _ { j } ^ { t } \big ) \Big \| _ { 2 } \leq \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \| \nabla f _ { j } ( \bar { x } ^ { t } ) - m _ { j } ^ { t } \| _ { 2 } .
$$

Now insert and subtract $c _ { j } ^ { t }$ and then the stale target $\nabla f _ { j } ( x _ { j } ^ { t - 1 } )$ :

$$
\begin{array} { r l r } {  { \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \| \nabla f _ { j } ( \bar { x } ^ { t } ) - m _ { j } ^ { t } \| _ { 2 } \le \displaystyle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \| \nabla f _ { j } ( \bar { x } ^ { t } ) - c _ { j } ^ { t } \| _ { 2 } + \displaystyle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \| c _ { j } ^ { t } - m _ { j } ^ { t } \| _ { 2 } } } \\ & { } & { \le \displaystyle \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \| \nabla f _ { j } ( x _ { j } ^ { t - 1 } ) - c _ { j } ^ { t } \| _ { 2 } + \displaystyle \frac { L } { M } \sum _ { j = 1 } ^ { M } \| \bar { x } ^ { t } - x _ { j } ^ { t - 1 } \| _ { 2 } + \sqrt { D _ { t } } . } \end{array}
$$

Using Jensen, the definition of $B _ { t }$ , and

$$
\| \bar { \boldsymbol { x } } ^ { t } - \boldsymbol { x } _ { j } ^ { t - 1 } \| _ { 2 } \leq \| \bar { \boldsymbol { x } } ^ { t } - \bar { \boldsymbol { x } } ^ { t - 1 } \| _ { 2 } + \| \boldsymbol { x } _ { j } ^ { t - 1 } - \bar { \boldsymbol { x } } ^ { t - 1 } \| _ { 2 } ,
$$

we obtain

$$
\| \nabla F ( \bar { x } ^ { t } ) - \bar { m } ^ { t } \| _ { 2 } \leq \sqrt { B _ { t } } + L \| \bar { x } ^ { t } - \bar { x } ^ { t - 1 } \| _ { 2 } + L \sqrt { C _ { t - 1 } } + \sqrt { D _ { t } } .
$$

Finally, by (B.11), $\lVert \bar { x } ^ { t } - \bar { x } ^ { t - 1 } \rVert _ { 2 } \leq K \bar { \rho } \eta$ . Therefore,

$$
\frac { \mathbb { E } [ E _ { t } ^ { ( c ) } ] } { \eta K } \le 2 \bar { \rho } ( \sqrt { D _ { t } } + \sqrt { B _ { t } } ) + 2 L \bar { \rho } ^ { 2 } K \eta + 2 L \bar { \rho } \sqrt { C _ { t - 1 } } .\tag{B.29}
$$

## B.6 Control-variates recursion

Lemma B.10 (Control recursion in the decentralized setting). Assume each $f _ { i }$ is L-smooth and Assumption 2 holds. Define

$$
B _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \big \| c _ { i } ^ { t } - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) \big \| _ { 2 } ^ { 2 } .
$$

Assume the network mixing matrix W is fixed, symmetric and doubly stochastic, and $\| \mathbf { W } - \mathbf { J } \| _ { 2 } =$ $\lambda < 1$ . Then there exist absolute constants $c _ { B } , c _ { B } ^ { \prime } > 0$ such that for all $t \geq 1$ ，

$$
B _ { t + 1 } \leq \Big ( 1 - { \frac { 1 - \beta _ { 2 } } { 8 } } \Big ) B _ { t } + c _ { B } { \frac { L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \beta _ { 2 } ) ( 1 - \lambda ) ^ { 2 } } } + { \frac { 8 ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K } } .\tag{B.30}
$$

Consequently, for any $T \geq 1$ 9

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \le c _ { B } ^ { \prime } \frac { L K \bar { \rho } \eta } { ( 1 - \beta _ { 2 } ) ( 1 - \lambda ) } + \frac { 8 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K } } + \frac { 1 6 \sqrt { B _ { 1 } } } { ( 1 - \beta _ { 2 } ) T } .\tag{B.31}
$$

If moreover the one-batch initialization is used (so $B _ { 1 } \leq \sigma ^ { 2 } )$ , then

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \le c _ { B } ^ { \prime } \frac { L K \bar { \rho } \eta } { ( 1 - \beta _ { 2 } ) ( 1 - \lambda ) } + \frac { 8 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K } } + \frac { 1 6 \sigma } { ( 1 - \beta _ { 2 } ) T } .\tag{B.32}
$$

Proof. Let $h _ { i } ^ { t } : = \nabla f _ { i } ( x _ { i } ^ { t - 1 } )$ and $e _ { i } ^ { t } : = c _ { i } ^ { t } - h _ { i } ^ { t }$ . Then $\begin{array} { r } { B _ { t } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } } \end{array}$ . Recall the control update

$$
\boldsymbol { c } _ { i } ^ { t + 1 } = \boldsymbol { c } _ { i } ^ { t } + ( 1 - \beta _ { 2 } ) ( \hat { c } _ { i } ^ { t } - c _ { i } ^ { t } ) , \qquad \hat { \boldsymbol { c } } _ { i } ^ { t } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \nabla f _ { i } ( \boldsymbol { x } _ { i } ^ { t , k - 1 } , \xi _ { i } ^ { t , k - 1 } ) .
$$

Subtract $h _ { i } ^ { t + 1 } = \nabla f _ { i } ( x _ { i } ^ { t } )$ to obtain

$$
\begin{array} { r } { e _ { i } ^ { t + 1 } = \beta _ { 2 } e _ { i } ^ { t } + ( 1 - \beta _ { 2 } ) ( \hat { c } _ { i } ^ { t } - h _ { i } ^ { t } ) - d _ { i } ^ { t } , \qquad d _ { i } ^ { t } : = h _ { i } ^ { t + 1 } - h _ { i } ^ { t } = \nabla f _ { i } ( x _ { i } ^ { t } ) - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) . } \end{array}\tag{B.33}
$$

Step 1: decompose $\hat { c } _ { i } ^ { t } - h _ { i } ^ { t }$ Write

$$
\hat { c } _ { i } ^ { t } - h _ { i } ^ { t } = b _ { i } ^ { t } + n _ { i } ^ { t } ,
$$

where

$$
b _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) \big ) , \qquad n _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \zeta _ { i } ^ { t , k - 1 } .
$$

By Assumption 2 and independence across queries,

$$
\mathbb { E } [ n _ { i } ^ { t } \mid \mathcal { F } _ { t } ] = 0 , \qquad \mathbb { E } \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq \frac { \sigma ^ { 2 } } { K } .\tag{B.34}
$$

Moreover, by L-smoothness,

$$
\lVert b _ { i } ^ { t } \rVert _ { 2 } ^ { 2 } \leq \frac { 1 } { K } \sum _ { k = 1 } ^ { K } L ^ { 2 } \lVert x _ { i } ^ { t , k - 1 } - x _ { i } ^ { t - 1 } \rVert _ { 2 } ^ { 2 } .
$$

Using

$$
\| { \boldsymbol x } _ { i } ^ { t , k - 1 } - { \boldsymbol x } _ { i } ^ { t - 1 } \| _ { 2 } \le \| { \boldsymbol x } _ { i } ^ { t } - { \boldsymbol x } _ { i } ^ { t - 1 } \| _ { 2 } + ( k - 1 ) \bar { \rho } \eta ,
$$

and $\begin{array} { r } { \| u + v \| _ { 2 } ^ { 2 } \leq 2 \| u \| _ { 2 } ^ { 2 } + 2 \| v \| _ { 2 } ^ { 2 } } \end{array}$ , we obtain

$$
\| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq 2 L ^ { 2 } \| x _ { i } ^ { t } - x _ { i } ^ { t - 1 } \| _ { 2 } ^ { 2 } + 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } .
$$

Averaging over i and using Lemma B.9 gives

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq 2 L ^ { 2 } R _ { t } + 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } \leq \frac { 2 0 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } .\tag{B.35}
$$

Likewise, by L-smoothness and Lemma B.9,

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq L ^ { 2 } R _ { t } \leq \frac { 9 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } .\tag{B.36}
$$

Step 2: one-step recursion for $\mathbb { E } \lVert e _ { i } ^ { t + 1 } \rVert ^ { 2 }$ . Condition on the history up to the beginning of round t. Since $d _ { i } ^ { t }$ is measurable with respect to this history and $\mathbb { E } [ n _ { i } ^ { t } \ | \ F _ { t } ] = 0$ , the cross term involving $n _ { i } ^ { t }$ vanishes. Applying Young’s inequality twice to (B.33), and using $\beta _ { 2 } ^ { 2 } \leq 1$ , yields the safe bound

$$
\mathbb { E } \Vert e _ { i } ^ { t + 1 } \Vert _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { 1 - \beta _ { 2 } } { 8 } \Big ) \mathbb { E } \Vert e _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + c _ { 1 } ( 1 - \beta _ { 2 } ) \mathbb { E } \Vert b _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + \frac { c _ { 2 } } { 1 - \beta _ { 2 } } \mathbb { E } \Vert d _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + \frac { 8 ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K } ,\tag{B.37}
$$

for some absolute constants $c _ { 1 } , c _ { 2 } > 0$ . Averaging (B.37) over $i = 1 , \dots , M$ and using (B.35)–(B.36) gives (B.30) after enlarging the numerical constant to $c _ { B }$

Step 3: average $\sqrt { B _ { t } }$ . Let $\begin{array} { r } { \alpha : = \frac { 1 - \beta _ { 2 } } { 8 } } \end{array}$ and

$$
C : = c _ { B } \frac { L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \beta _ { 2 } ) ( 1 - \lambda ) ^ { 2 } } + \frac { 8 ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K } .
$$

Then (B.30) is $B _ { t + 1 } \leq ( 1 - \alpha ) B _ { t } + C$ . Unrolling gives $\begin{array} { r } { B _ { t } \le ( 1 - \alpha ) ^ { t - 1 } B _ { 1 } + \frac { C } { \alpha } } \end{array}$ . Using $\sqrt { x + y } \leq$ ${ \sqrt { x } } + { \sqrt { y } }$

$$
\sqrt { B _ { t } } \leq ( 1 - \alpha ) ^ { \frac { t - 1 } { 2 } } \sqrt { B _ { 1 } } + \sqrt { \frac { C } { \alpha } } .
$$

Averaging over $t = 1 , \dots , T$ and using $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( \sqrt { 1 - \alpha } ) ^ { t - 1 } \leq \frac { 2 } { \alpha } } \end{array}$ yields

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \leq \sqrt { \frac { C } { \alpha } } + \frac { 2 \sqrt { B _ { 1 } } } { \alpha T } .
$$

Since $\alpha = ( 1 - \beta _ { 2 } ) / 8$ , the tail term is bounded by $\frac { 1 6 \sqrt { B _ { 1 } } } { ( 1 - \beta _ { 2 } ) T }$ . Moreover,

$$
\sqrt { \frac { C } { \alpha } } \le c _ { B } ^ { \prime } \frac { L K \bar { \rho } \eta } { ( 1 - \beta _ { 2 } ) ( 1 - \lambda ) } + \frac { 8 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K } }
$$

for a suitable absolute constant $c _ { B } ^ { \prime } > 0$ . This proves (B.31). The one-batch initialization version follows from $B _ { 1 } \leq \sigma ^ { 2 }$ □

## B.7 Network tracker recursion

Lemma B.11 (Average $\sqrt { D _ { t } }$ bound with explicit spectral gap). Assume W is fixed, symmetric and doubly stochastic with $\lambda : = \| \mathbf { W } - \mathbf { J } \| _ { 2 } \in [ 0 , 1 )$ . Recall

$$
D _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| m _ { i } ^ { t } - \bar { m } ^ { t } \| _ { 2 } ^ { 2 } = \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { m } } ^ { t } \| _ { F } ^ { 2 } , \qquad \tilde { \mathbf { m } } ^ { t } : = ( \mathbf { I } - \mathbf { J } ) \mathbf { m } ^ { t } .
$$

Then there exists an absolute constant $c _ { D } > 0$ such that for any $T \geq 1$ ，

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } \le \frac { c _ { D } } { 1 - \lambda } \bigl ( 1 - \beta _ { 2 } \bigr ) \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } + \frac { c _ { D } } { 1 - \lambda } \bigl ( 1 - \beta _ { 2 } \bigr ) \Bigl ( \frac { L K \bar { \rho } \eta } { 1 - \lambda } + \frac { \sigma } { \sqrt { K } } \Bigr ) + \frac { 2 \sqrt { D _ { 1 } } } { \bigl ( 1 - \lambda \bigr ) T } .\tag{B.38}
$$

Moreover, under the initialization $\begin{array} { r } { m _ { i } ^ { 1 } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } c _ { j } ^ { 1 } } \end{array}$ for all i, we have $D _ { 1 } = 0$ and hence the last term vanishes.

Proof. Stack trackers and control variates row-wise as matrices $\mathbf { m } ^ { t } , \mathbf { c } ^ { t } \in \mathbb { R } ^ { M \times d }$ whose i-th rows are $( m _ { i } ^ { t } ) ^ { \top }$ and $( c _ { i } ^ { t } ) ^ { \top }$ . Define the centered components

$$
\tilde { \mathbf { m } } ^ { t } : = ( \mathbf { I } - \mathbf { J } ) \mathbf { m } ^ { t } , \qquad \tilde { \mathbf { c } } ^ { t } : = ( \mathbf { I } - \mathbf { J } ) \mathbf { c } ^ { t } ,
$$

so that $\begin{array} { r } { D _ { t } = \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { m } } ^ { t } \| _ { F } ^ { 2 } } \end{array}$

Step 1: a contraction recursion for $\tilde { \mathbf { m } } ^ { t }$ . The tracker update is

$$
\mathbf { m } ^ { t + 1 } = \mathbf { W } ( \mathbf { m } ^ { t } + \mathbf { c } ^ { t + 1 } - \mathbf { c } ^ { t } ) .
$$

Using $\mathbf { J } \mathbf { W } = \mathbf { W } \mathbf { J } = \mathbf { J }$ , we have $( { \bf I } - { \bf J } ) { \bf W } = { \bf W } - { \bf J }$ and hence

$$
\tilde { \mathbf { m } } ^ { t + 1 } = ( \mathbf { W } - \mathbf { J } ) \big ( \tilde { \mathbf { m } } ^ { t } + \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t } \big ) .
$$

Therefore

$$
\| \tilde { \mathbf { m } } ^ { t + 1 } \| _ { F } \leq \lambda \left\| \tilde { \mathbf { m } } ^ { t } + \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t } \right\| _ { F } .
$$

Square both sides and apply the weighted inequality $\begin{array} { r } { \| A + B \| _ { F } ^ { 2 } \leq ( 1 + \theta ) \| A \| _ { F } ^ { 2 } + ( 1 + \theta ^ { - 1 } ) \| B \| _ { F } ^ { 2 } } \end{array}$ with $\begin{array} { r } { \theta : = \frac { 1 - \lambda ^ { 2 } } { 2 \lambda ^ { 2 } } } \end{array}$ (for $\lambda > 0 ;$ ; the case $\lambda = 0$ is trivial), giving

$$
\lVert \tilde { { \boldsymbol { \mathbf { m } } } } ^ { t + 1 } \rVert _ { F } ^ { 2 } \leq \frac { 1 + \lambda ^ { 2 } } { 2 } \lVert \tilde { { \boldsymbol { \mathbf { m } } } } ^ { t } \rVert _ { F } ^ { 2 } + \frac { 2 \lambda ^ { 2 } } { 1 - \lambda ^ { 2 } } \lVert \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t } \rVert _ { F } ^ { 2 } .\tag{B.39}
$$

Taking expectation and dividing by M yields

$$
D _ { t + 1 } \leq \frac { 1 + \lambda ^ { 2 } } { 2 } D _ { t } + \frac { 2 \lambda ^ { 2 } } { 1 - \lambda ^ { 2 } } \cdot \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t } \| _ { F } ^ { 2 } .\tag{B.40}
$$

Step 2: bound the driving term by $B _ { t }$ and local drift/noise. Since $\| \mathbf { I } - \mathbf { J } \| _ { 2 } \leq 1$

$$
\lVert \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t } \rVert _ { F } \leq \lVert \mathbf { c } ^ { t + 1 } - \mathbf { c } ^ { t } \rVert _ { F } .
$$

Moreover, with full participation the control update is ${ c _ { i } ^ { t + 1 } - c _ { i } ^ { t } = ( 1 - \beta _ { 2 } ) ( \hat { c } _ { i } ^ { t } - c _ { i } ^ { t } ) }$ , hence

$$
\| { \mathbf { c } ^ { t + 1 } - { \mathbf { c } ^ { t } } } \| _ { F } ^ { 2 } = ( 1 - \beta _ { 2 } ) ^ { 2 } \sum _ { i = 1 } ^ { M } \| \hat { c } _ { i } ^ { t } - c _ { i } ^ { t } \| _ { 2 } ^ { 2 } .
$$

Therefore

$$
\frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t } \| _ { F } ^ { 2 } \leq ( 1 - \beta _ { 2 } ) ^ { 2 } \cdot \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| \hat { c } _ { i } ^ { t } - c _ { i } ^ { t } \| _ { 2 } ^ { 2 } .\tag{B.41}
$$

Decompose

$$
\hat { c } _ { i } ^ { t } - c _ { i } ^ { t } = \underbrace { \left( \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) - c _ { i } ^ { t } \right) } _ { = : a _ { i } ^ { t } } + \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) \right) } _ { = : b _ { i } ^ { t } } + \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \zeta _ { i } ^ { t , k - 1 } } _ { = : n _ { i } ^ { t } } .
$$

Using $\| u + v + w \| _ { 2 } ^ { 2 } \leq 3 ( \| u \| _ { 2 } ^ { 2 } + \| v \| _ { 2 } ^ { 2 } + \| w \| _ { 2 } ^ { 2 } )$ , Jensen, Assumption $^ { 2 , }$ and the same estimate as in (B.35), we obtain

$$
\mathbb { E } \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq \frac { \sigma ^ { 2 } } { K } , \qquad \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq \frac { 2 0 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } .
$$

Finally, by definition, 1 $\begin{array} { r } { \sum _ { i } \mathbb { E } \| a _ { i } ^ { t } \| _ { 2 } ^ { 2 } = B _ { t } } \end{array}$ . Hence

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| \hat { c } _ { i } ^ { t } - c _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq 3 B _ { t } + c _ { 3 } \frac { L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } + \frac { 3 \sigma ^ { 2 } } { K }\tag{B.42}
$$

for some absolute constant $c _ { 3 } > 0$ . Combining (B.40)–(B.41)–(B.42) yields

$$
D _ { t + 1 } \leq \frac { 1 + \lambda ^ { 2 } } { 2 } D _ { t } + \frac { 2 \lambda ^ { 2 } } { 1 - \lambda ^ { 2 } } ( 1 - \beta _ { 2 } ) ^ { 2 } \Big ( B _ { t } + c _ { 4 } \frac { L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } + \frac { \sigma ^ { 2 } } { K } \Big )\tag{B.43}
$$

for some absolute constant $c _ { 4 } > 0$

Step 3: average $\sqrt { D _ { t } }$ . Let $\begin{array} { r } { q : = \frac { 1 + \lambda ^ { 2 } } { 2 } \in ( 0 , 1 ) } \end{array}$ and

$$
V _ { t } : = \frac { 2 \lambda ^ { 2 } } { 1 - \lambda ^ { 2 } } ( 1 - \beta _ { 2 } ) ^ { 2 } \Big ( B _ { t } + c _ { 4 } \frac { L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } + \frac { \sigma ^ { 2 } } { K } \Big ) .
$$

Then (B.43) is $D _ { t + 1 } \leq q D _ { t } + V _ { t }$ . Unrolling gives $\begin{array} { r } { D _ { t } \leq q ^ { t - 1 } D _ { 1 } + \sum _ { s = 1 } ^ { t - 1 } q ^ { t - 1 - s } V _ { s } } \end{array}$ . Using $\sqrt { x + y } \leq$ ${ \sqrt { x } } + { \sqrt { y } }$ and $\textstyle \sum _ { r = 0 } ^ { \infty } ( { \sqrt { q } } ) ^ { r } = { \frac { 1 } { 1 - { \sqrt { q } } } }$ , we obtain

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } \leq \frac { \sqrt { D _ { 1 } } } { T } \sum _ { t = 1 } ^ { T } ( \sqrt { q } ) ^ { t - 1 } + \frac { 1 } { T } \sum _ { s = 1 } ^ { T - 1 } \sqrt { V _ { s } } ^ { T - 1 - s } ( \sqrt { q } ) ^ { r } .
$$

Since $\textstyle 1 - { \sqrt { q } } \geq { \frac { 1 - q } { 2 } } = { \frac { 1 - \lambda ^ { 2 } } { 4 } } \geq { \frac { 1 - \lambda } { 4 } }$ , we have $\begin{array} { r } { \frac { 1 } { 1 - \sqrt { q } } \leq \frac { 4 } { 1 - \lambda } } \end{array}$ . Also,

$$
\sqrt { V _ { s } } \leq \frac { c _ { D } } { 1 - \lambda } ( 1 - \beta _ { 2 } ) \Big ( \sqrt { B _ { s } } + \frac { L K \bar { \rho } \eta } { 1 - \lambda } + \frac { \sigma } { \sqrt { K } } \Big )
$$

for a suitable absolute constant $c _ { D } > 0$ . Combining the last two displays yields (B.38).

Step 4: $D _ { 1 } = 0$ under the stated initialization. I $\begin{array} { r } { \mathrm { ~ f ~ } m _ { i } ^ { 1 } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } c _ { j } ^ { 1 } } \end{array}$ for all $i ,$ then all $m _ { i } ^ { 1 }$ are equal to the same vector, hence $m _ { i } ^ { 1 } - { \bar { m } } ^ { 1 } = 0$ for each i and thus $D _ { 1 } = 0$ □

## B.8 Proof of Theorem B.1

Proof. Let

$$
\Delta _ { F } : = F ( \bar { x } ^ { 1 } ) - F ^ { \star } , \qquad \alpha : = 1 - \beta _ { 2 } .
$$

Summing the descent core inequality (B.21) over $t = 1 , \dots , T$ , taking expectation, and telescoping yields

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \leq \frac { \Delta _ { F } } { T \eta K } + \frac { L } { 2 } K \bar { \rho } ^ { 2 } \eta + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { \mathbb { E } [ E _ { t } ] } { \eta K } .\tag{B.44}
$$

Recall $E _ { t } = E _ { t } ^ { ( a ) } + E _ { t } ^ { ( b ) } + E _ { t } ^ { ( c ) }$ . Using (B.25), (B.27), and (B.29), we have for each t

$$
\begin{array} { r l r } {  { \frac { \mathbb { E } [ E _ { t } ] } { \eta K } \le \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } } } \\ & { } & { + \ 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { B _ { t } } + 2 \bar { \rho } \sqrt { B _ { t } } } \\ & { } & { + \ 2 L \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { R _ { t } } + \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) \eta ( K - 1 ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma } \\ & { } & { + \ 2 \bar { \rho } \sqrt { D _ { t } } + 2 L \bar { \rho } ^ { 2 } K \eta + 2 L \bar { \rho } \sqrt { C _ { t - 1 } } . } \end{array}\tag{B.45}
$$

By Lemma B.8 and Lemma B.9,

$$
\sqrt { C _ { t - 1 } } \leq \frac { K \bar { \rho } \eta } { 1 - \lambda } , \qquad \sqrt { R _ { t } } \leq \frac { 3 K \bar { \rho } \eta } { 1 - \lambda } .
$$

Substituting these bounds into (B.45), averaging over $t = 1 , \dots , T$ , and inserting into (B.44) gives

$$
\begin{array} { c c } { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \leq \frac { \Delta _ { F } } { T \eta K } + \frac { L } { 2 } K \bar { \rho } ^ { 2 } \eta + \Big ( 2 + \displaystyle \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } } & \\ { \displaystyle + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } + 2 \bar { \rho } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } } & \\ { \displaystyle + \frac { 6 I \bar { \rho } ^ { 2 } ( 1 - \beta _ { 1 } ) } { 1 - \lambda } K \eta + \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) \eta ( K - 1 ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma } & \\ { \displaystyle + 2 \bar { \rho } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } + 2 L \bar { \rho } ^ { 2 } K \eta + \frac { 2 L \bar { \rho } ^ { 2 } } { 1 - \lambda } K \eta . } & { \eqno { ( \mathrm { 2 } } } \end{array}\tag{B.46}
$$

We next make the $B _ { t }$ and $D _ { t }$ contributions fully explicit.

Step 1: an explicit average bound for $\textstyle { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \sqrt { B _ { t } } }$ . Starting from (B.33), write

$$
e _ { i } ^ { t + 1 } = ( 1 - \alpha ) e _ { i } ^ { t } + \alpha b _ { i } ^ { t } + \alpha n _ { i } ^ { t } - d _ { i } ^ { t } , \qquad \alpha : = 1 - \beta _ { 2 } ,
$$

where, conditionally on the history up to round $t , d _ { i } ^ { t }$ is measurable, $\mathbb { E } [ n _ { i } ^ { t } \mid \mathcal { F } _ { t } ] = 0$ , and $\mathbb { E } \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq$ $\sigma ^ { 2 } / K$ by (B.34). Hence the cross term involving $n _ { i } ^ { t }$ vanishes and

$$
\mathbb { E } \big [ \| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } \big ] = \big \| ( 1 - \alpha ) e _ { i } ^ { t } + \alpha b _ { i } ^ { t } - d _ { i } ^ { t } \big \| _ { 2 } ^ { 2 } + \alpha ^ { 2 } \mathbb { E } \big [ \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } \big ] .
$$

Applying the weighted Young inequality $\begin{array} { r } { \| u + v \| _ { 2 } ^ { 2 } \leq ( 1 + \gamma ) \| u \| _ { 2 } ^ { 2 } + ( 1 + \gamma ^ { - 1 } ) \| v \| _ { 2 } ^ { 2 } } \end{array}$ with $u = ( 1 - \alpha ) e _ { i } ^ { t }$ $v = \alpha b _ { i } ^ { t } - d _ { i } ^ { t }$ , and $\gamma = \alpha / ( 2 ( 1 - \alpha ) )$ (the case $\alpha = 1$ is trivial), we get

$$
\big \| ( 1 - \alpha ) e _ { i } ^ { t } + \alpha b _ { i } ^ { t } - d _ { i } ^ { t } \big \| _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { \alpha } { 2 } \Big ) \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { 2 } { \alpha } \| \alpha b _ { i } ^ { t } - d _ { i } ^ { t } \| _ { 2 } ^ { 2 } .
$$

Using $\begin{array} { r } { \| u - v \| _ { 2 } ^ { 2 } \leq 2 \| u \| _ { 2 } ^ { 2 } + 2 \| v \| _ { 2 } ^ { 2 } } \end{array}$ , we therefore obtain

$$
\mathbb { E } \| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { \alpha } { 2 } \Big ) \mathbb { E } \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + 4 \alpha \mathbb { E } \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { 4 } { \alpha } \mathbb { E } \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { K } .\tag{B.47}
$$

Averaging over i and using (B.35) and (B.36) yields

$$
B _ { t + 1 } \leq \Bigl ( 1 - \frac { \alpha } { 2 } \Bigr ) B _ { t } + \frac { 1 2 8 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { \alpha ( 1 - \lambda ) ^ { 2 } } + \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { K } .\tag{B.48}
$$

Let

$$
\delta : = \frac { \alpha } { 2 } , \qquad U _ { B } : = \frac { 1 2 8 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { \alpha ( 1 - \lambda ) ^ { 2 } } + \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { K } .
$$

Then (B.48) implies $B _ { t } \le ( 1 - \delta ) ^ { t - 1 } B _ { 1 } + U _ { B } / \delta$ . Using ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ and $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( \sqrt { 1 - \delta } ) ^ { t - 1 } \leq 2 / \delta } \end{array}$ and recalling $B _ { 1 } \leq \sigma ^ { 2 }$ under the one-batch initialization, we obtain

$$
\begin{array} { r l r } {  { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \leq \sqrt { \frac { U _ { B } } { \delta } } + \frac { 2 \sqrt { B _ { 1 } } } { \delta T } } } \\ & { } & { \leq \frac { 1 6 L K \bar { \rho } \eta } { \alpha ( 1 - \lambda ) } + \frac { \sqrt { 2 } \sigma \sqrt { \alpha } } { \sqrt { K } } + \frac { 4 \sigma } { \alpha T } . } \end{array}\tag{B.49}
$$

Step 2: an explicit average bound for $\textstyle { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \sqrt { D _ { t } } }$ . Let $\Delta \tilde { \mathbf { c } } ^ { t } : = \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t }$ . From the tracker recursion in the proof of Lemma B.11,

$$
\begin{array} { r } { \tilde { \mathbf { m } } ^ { t + 1 } = ( \mathbf { W } - \mathbf { J } ) ( \tilde { \mathbf { m } } ^ { t } + \Delta \tilde { \mathbf { c } } ^ { t } ) . } \end{array}
$$

Hence, using $\| \mathbf { W } - \mathbf { J } \| _ { 2 } = \lambda$ and Minkowski’s inequality in $L _ { 2 }$ ,

$$
\sqrt { D _ { t + 1 } } \leq \lambda \sqrt { D _ { t } } + \frac { \lambda } { \sqrt { M } } \Big ( \mathbb { E } \| \Delta \tilde { \mathbf { c } } ^ { t } \| _ { F } ^ { 2 } \Big ) ^ { 1 / 2 } .
$$

Repeating the decomposition used in the proof of Lemma B.11, and using (B.41) together with (B.35), we have

$$
\frac { 1 } { M } \mathbb { E } \| \Delta \tilde { \mathbf { c } } ^ { t } \| _ { F } ^ { 2 } \leq \alpha ^ { 2 } \Big ( 3 B _ { t } + \frac { 6 0 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } + \frac { 3 \sigma ^ { 2 } } { K } \Big ) .
$$

Substituting this estimate into the previous display and using $\lambda \leq 1$ gives

$$
\sqrt { D _ { t + 1 } } \leq \lambda \sqrt { D _ { t } } + \alpha \sqrt { 3 } \sqrt { B _ { t } } + \frac { 2 \sqrt { 1 5 } \alpha L K \bar { \rho } \eta } { 1 - \lambda } + \frac { \sqrt { 3 } \alpha \sigma } { \sqrt { K } } .\tag{B.50}
$$

Under the stated initialization, $D _ { 1 } = 0$ . Unrolling (B.50), averaging over $t = 1 , \dots , T$ , and using $\begin{array} { r } { \sum _ { r = 0 } ^ { \infty } \lambda ^ { r } = ( 1 - \lambda ) ^ { - 1 } } \end{array}$ , we obtain

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } \leq \frac { \sqrt { 3 } \alpha } { 1 - \lambda } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } + \frac { 2 \sqrt { 1 5 } \alpha L K \bar { \rho } \eta } { ( 1 - \lambda ) ^ { 2 } } + \frac { \sqrt { 3 } \alpha \sigma } { ( 1 - \lambda ) \sqrt { K } } .\tag{B.51}
$$

Substituting (B.49) into (B.51) and using $\alpha \leq 1$ yields

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } \le \frac { ( 1 6 \sqrt { 3 } + 2 \sqrt { 1 5 } ) L K \bar { \rho } \eta } { ( 1 - \lambda ) ^ { 2 } } + \frac { ( \sqrt { 6 } + \sqrt { 3 } ) \sigma \alpha } { ( 1 - \lambda ) \sqrt { K } } + \frac { 4 \sqrt { 3 } \sigma } { ( 1 - \lambda ) T } .\tag{B.52}
$$

Substituting (B.49) and (B.52) into (B.46), we obtain

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { \ell , * } \leq \frac { \Delta _ { F } } { T \eta K } + \frac { L } { T } K \bar { \rho } ^ { 2 } \eta + \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 \big ( 1 - \beta _ { 1 } \big ) ^ { 2 } \sigma ^ { 2 } } { \eta } } \\ & { \qquad + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \Big [ \frac { 1 6 L K \bar { \rho } \eta } { \alpha ( 1 - \lambda ) } + \frac { \sqrt { 2 } \sigma \sqrt { \alpha } } { \sqrt { K } } + \frac { 4 \sigma } { \alpha T } \Big ] } \\ & { \qquad + 2 \bar { \rho } \Big [ \frac { 1 6 L K \bar { \rho } \eta } { \alpha ( 1 - \lambda ) } + \frac { \sqrt { 2 } \sigma \sqrt { \alpha } } { \sqrt { K } } + \frac { 4 \sigma } { \alpha T } \Big ] + \frac { 6 L \bar { \rho } ^ { 2 } ( 1 - \beta _ { 1 } ) } { 1 - \lambda } K \eta } \\ & { \qquad + \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) \eta ( K - 1 ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma + \frac { 2 \bar { \rho } ( 1 6 \sqrt { 3 } + 2 \sqrt { 1 5 } ) L K \bar { \rho } \eta } { \big ( 1 - \lambda ) ^ { 2 } } } \\ & { \qquad + \frac { 2 \bar { \rho } \big ( \sqrt { 6 } + \sqrt { 3 } \big ) \sigma \alpha } { \big ( 1 - \lambda ) \sqrt { K } } + \frac { 8 \sqrt { 3 } \bar { \rho } \sigma } { ( 1 - \lambda ) T } + 2 L \bar { \rho } ^ { 2 } K \eta + \frac { 2 L \bar { \rho } ^ { 2 } } { 1 - \lambda } K \eta . } \end{array}\tag{.53}
$$

We now substitute the hyperparameter choice

$$
\eta = \frac { M ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } , \qquad 1 - \beta _ { 1 } = \eta , \qquad \alpha = 1 - \beta _ { 2 } = \sqrt { \frac { M K } { T + K } } , \qquad \mathrm { a n d ~ a s s u m e ~ } T \geq M K .
$$

Since $T \geq M K$ and $M \geq 1$ , we also have $K \leq T$ and hence $T + K \leq 2 T$ . Moreover,

$$
\begin{array} { r l } & { \quad K \eta = \frac { M ^ { 1 / 2 } K ^ { 1 / 4 } } { T ^ { 3 / 4 } } \le \frac { 1 } { \sqrt { T } } , \qquad \eta = \frac { 1 } { \{ M K T \} ^ { 1 / 4 } } \Bigg \} \frac { M } { K T } \le \frac { 1 } { \{ M K T \} ^ { 1 / 4 } } , } \\ & { \frac { K \eta } { \alpha } = \frac { \sqrt { T + K } } { M ^ { 1 / 4 } K ^ { 1 / 4 } T ^ { 3 / 4 } } \le \frac { \sqrt { 2 } } { \{ M K T \} ^ { 1 / 4 } } , \qquad \sqrt { \overline { { \Lambda } } } = \frac { M ^ { 1 / 4 } } { K ^ { 1 / 4 } ( T + K ) ^ { 1 / 4 } } } \\ & { \frac { \alpha } { \sqrt { K } } = \sqrt { \frac { M } { T + K } } \le \sqrt { \frac { M } { T } } , \qquad \frac { 1 } { \alpha T } = \frac { \sqrt { T + K } } { \sqrt { M K T } } \le \frac { \sqrt { 2 } } { \sqrt { M K T } } , } \\ & { \frac { K \eta ^ { 2 } } { K ^ { 2 } } = \frac { \sqrt { M } } { \sqrt { K ^ { 1 / 2 } } } , \qquad \frac { K \eta ^ { 2 } } { \alpha } = \frac { \sqrt { T + K } } { K T ^ { 3 / 2 } } \le \frac { \sqrt { 2 } } { K T ^ { 2 } } , } \\ & { \frac { \eta \sqrt { \alpha } } { \sqrt { K } } = \frac { \sqrt { M } } { K ^ { 1 / 4 } ( T + K ) ^ { 1 / 4 } } \le \frac { \sqrt { M } } { K T } , \qquad \frac { \eta } { \alpha T } = \frac { \sqrt { T + K } } { M ^ { 1 / 4 } K ^ { 3 / 4 } T ^ { 3 / 4 } } \le \frac { \sqrt { 2 } } { M ^ { 1 / 4 } K ^ { 5 / 4 } T ^ { 3 / 4 } } , } \\ & { \eta = \frac { M ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } \le \frac { 1 } { K \sqrt { T } } . } \end{array}\tag{B.54}
$$

Also,

$$
\frac { \Delta _ { F } } { T \eta K } = \frac { \Delta _ { F } } { ( M K T ) ^ { 1 / 4 } } .\tag{B.55}
$$

Using (B.54) and (B.55) in (B.53), and bounding $2 + \textstyle { \frac { 1 } { 2 K } } \leq \frac { 5 } { 2 }$ , we arrive at the explicit estimate

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { P } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { \alpha , s } \leq } & { \left[ \frac { L \bar { p } ^ { 2 } } { 2 } + 2 L \bar { p } ^ { 2 } + \frac { 2 L \bar { p } ^ { 2 } } { 1 - \lambda } + \frac { ( 3 2 \sqrt { 3 } + 4 \sqrt { 1 5 } ) L \bar { p } ^ { 2 } } { ( 1 - \lambda ) ^ { 2 } } + \frac { 2 ( \sqrt { 6 } + \sqrt { 3 } ) \bar { p } \sigma \sqrt { M } } { 1 - \lambda } \right] \frac { 1 } { \sqrt { T } } } \\ & { \qquad + \left[ \Delta _ { \mathcal { S } } + \frac { 5 } { 2 } \bar { p } ^ { 2 } + L ^ { 2 } \bar { p } ^ { 4 } + 2 \sigma ^ { 2 } + \frac { 3 2 \sqrt { 2 } L \bar { p } ^ { 2 } } { 1 - \lambda } + 2 \sqrt { 2 } \bar { p } \sigma \sqrt { M } \right] \frac { 1 } { ( M K T ) ^ { 3 / 4 } } } \\ & { \quad + \frac { 6 L \bar { p } ^ { 2 } \sqrt { M } } { ( 1 - \lambda ) \sqrt { K T } \bar { M } ^ { 2 / 2 } } + \frac { L \bar { p } ^ { 2 } \sqrt { M } } { \sqrt { K T } \bar { M } ^ { 2 / 2 } } } \\ & { \quad + \frac { 3 2 \sqrt { 2 } L \bar { p } ^ { 2 } } { ( 1 - \lambda ) K T } + \frac { 2 \sqrt { 2 } \bar { p } \sigma \sqrt { M } } { K T } } \\ & { \quad + \frac { 8 4 \sqrt { 2 } \bar { p } \sigma } { ( 1 - \lambda ) K T } + \frac { 2 \sqrt { 2 } \bar { p } \sigma \sqrt { M } ^ { 1 / 4 } } { K T } } \\ &  \quad + \frac { 8 4 \sqrt { 2 } \bar { p } \sigma } { ( 1 - \lambda ) \bar { p } \sigma \sqrt { M } ^ { 2 / 4 } } + \frac { 2 \bar { p } \sigma \bar { M } ^ { 1 / 4 } }  ( 1 - \lambda ) \bar { p } \sigma \sqrt  M  \end{array}\tag{6}
$$

The first line is of order $T ^ { - 1 / 2 }$ and the second is of order $( M K T ) ^ { - 1 / 4 }$ . In particular, the corrected query-wise noise contribution is $2 \bar { \rho } \sigma \eta \leq 2 \bar { \rho } \sigma / ( K \sqrt { T } )$ under $M K \leq T$ , and is absorbed by the first term of $R _ { \mathrm { l o } } ( T , K )$ . The remaining terms are also dominated by that remainder. Thus the stated $C _ { 1 }$ and $C _ { 2 }$ are unchanged, and choosing $C _ { 0 }$ large enough completes the proof. □

Network assumption (finite-time consensus). Let $\{ \mathbf { W } ^ { t } \} _ { t \geq 1 }$ be a sequence of doubly stochastic mixing matrices. Assume it satisfies the finite-time consensus $\mathrm { ( F T C ) }$ property: there exists an integer $\tau \geq 1$ such that for any t,

$$
\mathbf { W } ^ { t + \tau - 1 } \mathbf { W } ^ { t + \tau - 2 } \cdot \cdot \cdot \mathbf { W } ^ { t } = \mathbf { J } , \qquad \mathbf { J } : = { \frac { 1 } { M } } \mathbf { 1 } \mathbf { 1 } ^ { \top } .\tag{B.57}
$$

We refer to τ as the FTC horizon.

Lemma B.12 (FTC consensus bound for $C _ { t } )$ . Let

$$
C _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - \bar { x } ^ { t } \| _ { 2 } ^ { 2 } , \qquad \bar { x } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } x _ { i } ^ { t } .
$$

Assume consensual initialization $x _ { i } ^ { 1 } = x ^ { 1 }$ for all $i ,$ and assume each round satisfies

$$
\| x _ { i } ^ { t , k } - x _ { i } ^ { t } \| _ { 2 } \leq K \bar { \rho } \eta , \qquad \forall i \in [ M ] , t \geq 1 .
$$

Under the FTC property (8), for all $t \geq 1$

$$
\sqrt { C _ { t } } \leq \tau K \bar { \rho } \eta , \qquad a n d h e n c e \qquad C _ { t } \leq \tau ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } .\tag{B.58}
$$

Proof. Stack the node models row-wise into $\mathbf { X } ^ { t } \in \mathbb { R } ^ { M \times d }$ whose i-th row is $( x _ { i } ^ { t } ) ^ { \top }$ . Let $\tilde { \mathbf { X } } ^ { t } : = ( \mathbf { I } - \mathbf { J } ) \mathbf { X } ^ { t }$ 2 so

$$
C _ { t } = \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { X } } ^ { t } \| _ { F } ^ { 2 } .
$$

Step 1: expand $\mathbf { X } ^ { t + \tau }$ over τ rounds. The DL mixing step is $\mathbf { X } ^ { s + 1 } = \mathbf { W } ^ { s } \mathbf { X } ^ { s , K }$ . Introduce the local-step increment $\Delta ^ { s } : = \mathbf { X } ^ { s , K } - \mathbf { X } ^ { s }$ . Then $\mathbf { X } ^ { s , K } = \mathbf { X } ^ { s } + \Delta ^ { s }$ and

$$
\mathbf { X } ^ { s + 1 } = \mathbf { W } ^ { s } ( \mathbf { X } ^ { s } + \pmb { \Delta } ^ { s } ) .
$$

Unrolling τ steps from time t gives

$$
\mathbf { X } ^ { t + \tau } = \mathbf { W } ^ { t + \tau - 1 } \cdot \cdot \cdot \mathbf { W } ^ { t } \mathbf { X } ^ { t } + \sum _ { r = 0 } ^ { \tau - 1 } \mathbf { W } ^ { t + \tau - 1 } \cdot \cdot \cdot \mathbf { W } ^ { t + r } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { t + r } .
$$

Apply (I − J) to both sides. Using $( \mathbf { I } - \mathbf { J } ) \mathbf { W } ^ { s } = ( \mathbf { I } - \mathbf { J } ) \mathbf { W } ^ { s } ( \mathbf { I } - \mathbf { J } )$ (which holds since $\mathbf { W } ^ { s }$ is doubly stochastic and $\mathbf { J } \mathbf { W } ^ { s } = \mathbf { W } ^ { s } \mathbf { J } = \mathbf { J } )$ , we obtain

$$
\tilde { \mathbf { X } } ^ { t + \tau } = ( \mathbf { I } - \mathbf { J } ) \mathbf { W } ^ { t + \tau - 1 } \cdot \cdot \mathbf { \cdot } \mathbf { W } ^ { t } \tilde { \mathbf { X } } ^ { t } + \sum _ { r = 0 } ^ { \tau - 1 } ( \mathbf { I } - \mathbf { J } ) \mathbf { W } ^ { t + \tau - 1 } \cdot \mathbf { \cdot } \mathbf { \cdot } \mathbf { \cdot } \mathbf { W } ^ { t + r } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { t + r } .
$$

By FTC, $\mathbf { W } ^ { t + \tau - 1 } \cdot \cdot \cdot \mathbf { W } ^ { t } = \mathbf { J }$ , hence $( \mathbf { I } - \mathbf { J } ) \mathbf { W } ^ { t + \tau - 1 } \cdot \cdot \cdot \mathbf { W } ^ { t } = \mathbf { 0 }$ and the first term vanishes:

$$
\tilde { \mathbf { X } } ^ { t + \tau } = \sum _ { r = 0 } ^ { \tau - 1 } ( \mathbf { I } - \mathbf { J } ) \mathbf { W } ^ { t + \tau - 1 } \cdot \cdot \cdot \mathbf { W } ^ { t + r } \mathbf { \Delta } \mathbf { \Delta } \mathbf { \Delta } ^ { t + r } .
$$

Step 2: bound the RHS by triangle inequality. Since $\| ( \mathbf { I } - \mathbf { J } ) \| _ { 2 } \leq 1$ and $\| \mathbf { W } ^ { s } \| _ { 2 } \leq 1$ for doubly stochastic ${ \bf W } ^ { s }$

$$
\| \tilde { \mathbf { X } } ^ { t + \tau } \| _ { F } \leq \sum _ { r = 0 } ^ { \tau - 1 } \| \Delta ^ { t + r } \| _ { F } .
$$

Using the per-node movement bound,

$$
\| \boldsymbol { \Delta } ^ { t + r } \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { M } \| x _ { i } ^ { t + r , K } - x _ { i } ^ { t + r } \| _ { 2 } ^ { 2 } \leq M ( K \bar { \rho } \eta ) ^ { 2 } , \quad \Rightarrow \quad \| \boldsymbol { \Delta } ^ { t + r } \| _ { F } \leq \sqrt { M } K \bar { \rho } \eta .
$$

Therefore,

$$
\| \tilde { \mathbf { X } } ^ { t + \tau } \| _ { F } \leq \tau \sqrt { M } K \bar { \rho } \eta .
$$

Divide by $\sqrt { M }$ and take expectation:

$$
\sqrt { C _ { t + \tau } } = \frac { 1 } { \sqrt { M } } \sqrt { \mathbb { E } \| \tilde { \mathbf { X } } ^ { t + \tau } \| _ { F } ^ { 2 } } \leq \frac { 1 } { \sqrt { M } } \cdot \mathbb { E } \| \tilde { \mathbf { X } } ^ { t + \tau } \| _ { F } \leq \tau K \bar { \rho } \eta ,
$$

where we used Jensen in the second inequality. This yields (B.58) for indices of the form $t + \tau$ . For intermediate times, the same bound holds because each round injects at most $K \bar { \rho } \eta$ movement per node, so the worst-case disagreement over any τ-window is still bounded by $\tau K \bar { \rho } \eta$ □

Lemma B.13 (One-step model transition under FTC). Under the assumptions of Lemma $B . 1 2 ,$ for all $t \geq 2$

$$
R _ { t } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - x _ { i } ^ { t - 1 } \| _ { 2 } ^ { 2 } \leq 9 ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } .\tag{B.59}
$$

Proof. For each $i ,$ write

$$
x _ { i } ^ { t } - x _ { i } ^ { t - 1 } = ( x _ { i } ^ { t } - \bar { x } ^ { t } ) + ( \bar { x } ^ { t } - \bar { x } ^ { t - 1 } ) + ( \bar { x } ^ { t - 1 } - x _ { i } ^ { t - 1 } ) .
$$

Using $\| a + b + c \| _ { 2 } ^ { 2 } \leq 3 ( \| a \| _ { 2 } ^ { 2 } + \| b \| _ { 2 } ^ { 2 } + \| c \| _ { 2 } ^ { 2 } )$ , averaging over $i ,$ and taking expectation yield

$$
R _ { t } \leq 3 C _ { t } + 3 C _ { t - 1 } + 3 \mathbb { E } \Vert \bar { x } ^ { t } - \bar { x } ^ { t - 1 } \Vert _ { 2 } ^ { 2 } .
$$

By (B.11),

$$
\begin{array} { r } { \mathbb { E } \| \bar { x } ^ { t } - \bar { x } ^ { t - 1 } \| _ { 2 } ^ { 2 } \leq ( K \bar { \rho } \eta ) ^ { 2 } . } \end{array}
$$

Moreover, Lemma B.12 gives

$$
C _ { t } \le \tau ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } , \qquad C _ { t - 1 } \le \tau ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } .
$$

Hence

$$
R _ { t } \leq ( 6 \tau ^ { 2 } + 3 ) ( K \bar { \rho } \eta ) ^ { 2 } \leq 9 ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } .
$$

This proves (B.59).

## B.9 Control-variates recursion under FTC

Lemma B.14 $( B _ { t }$ recursion driven by consensus in DL). Define

$$
B _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| c _ { i } ^ { t } - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) \| _ { 2 } ^ { 2 } , \qquad C _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| x _ { i } ^ { t } - \bar { x } ^ { t } \| _ { 2 } ^ { 2 } , \quad \bar { x } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } x _ { i } ^ { t } .
$$

Assume each $f _ { i }$ is L-smooth and the stochastic gradients satisfy $\mathbb { E } \| \nabla f _ { i } ( x , \xi ) - \nabla f _ { i } ( x ) \| _ { 2 } ^ { 2 } \leq \sigma ^ { 2 }$ Assume the local drift bound $\lVert x _ { i } ^ { t , k } - x _ { i } ^ { t } \rVert _ { 2 } \leq K \bar { \rho } \eta$ holds for all $i , t .$ . Then there exists an absolute constant $c _ { B , \tau } > 0$ such that for all $t \geq 1$

$$
B _ { t + 1 } \leq \Big ( 1 - \frac { 1 - \beta _ { 2 } } { 8 } \Big ) B _ { t } + c _ { B , \tau } \frac { L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } } { 1 - \beta _ { 2 } } + \frac { 8 ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K } .\tag{B.60}
$$

Proof. Let $h _ { i } ^ { t } : = \nabla f _ { i } ( x _ { i } ^ { t - 1 } )$ and $e _ { i } ^ { t } : = c _ { i } ^ { t } - h _ { i } ^ { t }$ . As in the static case,

$$
\begin{array} { r } { e _ { i } ^ { t + 1 } = \beta _ { 2 } e _ { i } ^ { t } + ( 1 - \beta _ { 2 } ) ( \hat { c } _ { i } ^ { t } - h _ { i } ^ { t } ) - d _ { i } ^ { t } , \qquad d _ { i } ^ { t } : = \nabla f _ { i } ( x _ { i } ^ { t } ) - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) . } \end{array}
$$

Decompose $\hat { c } _ { i } ^ { t } - h _ { i } ^ { t } = b _ { i } ^ { t } + n _ { i } ^ { t }$ with

$$
b _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \big ( \nabla f _ { i } ( x _ { i } ^ { t , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { t - 1 } ) \big ) , \qquad n _ { i } ^ { t } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \zeta _ { i } ^ { t , k - 1 } .
$$

Exactly as in the repaired static proof, conditioning on the history at the beginning of round t removes the cross term involving $n _ { i } ^ { t }$ , and Young’s inequality yields a recursion of the form

$$
\mathbb { E } \Vert e _ { i } ^ { t + 1 } \Vert _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { 1 - \beta _ { 2 } } { 8 } \Big ) \mathbb { E } \Vert e _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + c _ { 1 } ( 1 - \beta _ { 2 } ) \mathbb { E } \Vert b _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + \frac { c _ { 2 } } { 1 - \beta _ { 2 } } \mathbb { E } \Vert d _ { i } ^ { t } \Vert _ { 2 } ^ { 2 } + \frac { 8 ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K }
$$

for absolute constants $c _ { 1 } , c _ { 2 } > 0$ . By L-smoothness and Lemma B.13,

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq 2 L ^ { 2 } R _ { t } + 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } \leq c _ { 3 } L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } ,
$$

and

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq L ^ { 2 } R _ { t } \leq 9 L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } .
$$

Averaging over i and enlarging the numerical constant prove (B.60).

Lemma B.15 (Average bound on $\textstyle { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \sqrt { B _ { t } } }$ in DL under FTC). Assume (B.60) holds. Then there exists an absolute constant $c _ { B , \tau } ^ { \prime } > 0$ such that for any $T \geq 1$ 2

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \le c _ { B , \tau } ^ { \prime } \frac { L K \bar { \rho } \eta } { 1 - \beta _ { 2 } } \sqrt { 1 + \tau ^ { 2 } } + \frac { 8 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K } } + \frac { 1 6 \sqrt { B _ { 1 } } } { ( 1 - \beta _ { 2 } ) T } .\tag{B.61}
$$

Proof. Let $\begin{array} { r } { \alpha : = \frac { 1 - \beta _ { 2 } } { 8 } } \end{array}$ and

$$
C : = c _ { B , \tau } \frac { L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } } { 1 - \beta _ { 2 } } + \frac { 8 ( 1 - \beta _ { 2 } ) ^ { 2 } \sigma ^ { 2 } } { K } .
$$

Then (B.60) is $B _ { t + 1 } \leq ( 1 - \alpha ) B _ { t } { + } C$ . Unrolling gives $B _ { t } \leq ( 1 - \alpha ) ^ { t - 1 } B _ { 1 } + \frac { C } { \alpha }$ . Using ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ and averaging over t yield

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \leq \sqrt { \frac { C } { \alpha } } + \frac { 1 6 \sqrt { B _ { 1 } } } { ( 1 - \beta _ { 2 } ) T } .
$$

Since $\alpha = ( 1 - \beta _ { 2 } ) / 8$ , the first term is bounded by

$$
\sqrt { \frac { C } { \alpha } } \leq c _ { B , \tau } ^ { \prime } \frac { L K \bar { \rho } \eta } { 1 - \beta _ { 2 } } \sqrt { 1 + \tau ^ { 2 } } + \frac { 8 \sigma \sqrt { 1 - \beta _ { 2 } } } { \sqrt { K } }
$$

for a suitable absolute constant $c _ { B , \tau } ^ { \prime } > 0$ . This proves (B.61).

## B.9.1 FTC version of the tracker-disagreement recursion

Lemma B.16 (Average $\sqrt { D _ { t } }$ bound under FTC ). Assume the time-varying mixing matrices $\{ \mathbf { W } ^ { t } \} _ { t \geq 1 }$ are doubly stochastic (with nonnegative entries as in standard gossip), and satisfy the finite-time consensus (FTC) condition (8) with horizon τ. Recall

$$
D _ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| m _ { i } ^ { t } - \bar { m } ^ { t } \| _ { 2 } ^ { 2 } , \qquad \bar { m } ^ { t } : = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } m _ { i } ^ { t } .
$$

Then there exists an absolute constant $c _ { D , \tau } > 0$ such that for any $T \geq 1$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } \le c _ { D , \tau } \tau \left( 1 - \beta _ { 2 } \right) \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } + c _ { D , \tau } \tau \left( 1 - \beta _ { 2 } \right) \left( L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } + \frac { \sigma } { \sqrt { K } } \right) + \frac { 2 \tau \sqrt { D _ { 1 } } } { T } .\tag{B.62}
$$

Moreover, under the initialization m $\begin{array} { r } { \ i _ { i } ^ { 1 } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } } \end{array}$ c<sup>1</sup><sub>j</sub> for all $i ,$ we have $D _ { 1 } = 0$

Proof. Stack trackers row-wise into $\mathbf { m } ^ { t } \in \mathbb { R } ^ { M \times d }$ and define

$$
{ \mathbf J } : = \frac { 1 } { M } { \mathbf 1 } { \mathbf 1 } ^ { \top } , \qquad \tilde { \mathbf m } ^ { t } : = ( { \mathbf I } - { \mathbf J } ) { \mathbf m } ^ { t } , \qquad \tilde { \mathbf c } ^ { t } : = ( { \mathbf I } - { \mathbf J } ) { \mathbf c } ^ { t } .
$$

Then $\begin{array} { r } { D _ { t } = \frac { 1 } { M } \mathbb { E } \| \tilde { \mathbf { m } } ^ { t } \| _ { F } ^ { 2 } } \end{array}$ . The tracker update is

$$
\mathbf { m } ^ { t + 1 } = \mathbf { W } ^ { t } ( \mathbf { m } ^ { t } + \mathbf { c } ^ { t + 1 } - \mathbf { c } ^ { t } ) .
$$

Applying (I − J) and using $\mathbf { J } \mathbf { W } ^ { t } = \mathbf { J }$ gives

$$
\tilde { \mathbf { m } } ^ { t + 1 } = ( \mathbf { I } - \mathbf { J } ) \mathbf { W } ^ { t } \big ( \tilde { \mathbf { m } } ^ { t } + \tilde { \mathbf { c } } ^ { t + 1 } - \tilde { \mathbf { c } } ^ { t } \big ) .\tag{B.63}
$$

Exactly as in the current proof, FTC implies the τ-window bound

$$
\sqrt { D _ { t + \tau } } \leq \sum _ { r = 0 } ^ { \tau - 1 } \sqrt { \frac { 1 } { M } \mathbb { E } \| \mathbf { c } ^ { t + r + 1 } - \mathbf { c } ^ { t + r } \| _ { F } ^ { 2 } } .\tag{B.64}
$$

It remains to bound each control diference. Since $c _ { i } ^ { s + 1 } - c _ { i } ^ { s } = ( 1 - \beta _ { 2 } ) ( \hat { c } _ { i } ^ { s } - c _ { i } ^ { s } )$ ，

$$
\frac { 1 } { M } \mathbb { E } \| \mathbf { c } ^ { s + 1 } - \mathbf { c } ^ { s } \| _ { F } ^ { 2 } = ( 1 - \beta _ { 2 } ) ^ { 2 } \cdot \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| \hat { c } _ { i } ^ { s } - c _ { i } ^ { s } \| _ { 2 } ^ { 2 } .
$$

Now decompose

$$
\hat { c } _ { i } ^ { s } - c _ { i } ^ { s } = \underbrace { \left( \nabla f _ { i } ( x _ { i } ^ { s - 1 } ) - c _ { i } ^ { s } \right) } _ { = : a _ { i } ^ { s } } + \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( \nabla f _ { i } ( x _ { i } ^ { s , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { s - 1 } ) ) } _ { = : b _ { i } ^ { s } } + \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \zeta _ { i } ^ { s , k - 1 } } _ { = : n _ { i } ^ { s } } .
$$

Using $\| u + v + w \| _ { 2 } ^ { 2 } \leq 3 ( \| u \| _ { 2 } ^ { 2 } + \| v \| _ { 2 } ^ { 2 } + \| w \| _ { 2 } ^ { 2 } )$ , Jensen, Assumption $^ { 2 , }$ and Lemma B.13, we obtain

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| \hat { c } _ { i } ^ { s } - c _ { i } ^ { s } \| _ { 2 } ^ { 2 } \leq 3 B _ { s } + c _ { 5 } L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } + \frac { 3 \sigma ^ { 2 } } { K }
$$

for an absolute constant $c _ { 5 } > 0$ . Therefore,

$$
\sqrt { \frac { 1 } { M } \mathbb { E } \| \mathbf { c } ^ { s + 1 } - \mathbf { c } ^ { s } \| _ { F } ^ { 2 } } \leq c _ { D , \tau } ( 1 - \beta _ { 2 } ) \Big ( \sqrt { B _ { s } } + L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } + \frac { \sigma } { \sqrt { K } } \Big )\tag{B.65}
$$

for a suitable absolute constant $c _ { D , \tau } > 0$ . Plugging (B.65) into (B.64), summing over $t ,$ and using that each index appears in at most τ windows give (B.62). The last statement follows exactly as in the current proof. □

## B.10 Proof of Theorem B.2

Proof. Summing the descent core inequality (B.21) over $t = 1 , \dots , T$ , taking expectation, and telescoping yields

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \leq \frac { \Delta _ { F } } { T \eta K } + \frac { L } { 2 } K \bar { \rho } ^ { 2 } \eta + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { \mathbb { E } [ E _ { t } ] } { \eta K } .\tag{B.66}
$$

Recall $E _ { t } = E _ { t } ^ { ( a ) } + E _ { t } ^ { ( b ) } + E _ { t } ^ { ( c ) }$ . Using (B.25), (B.27), and (B.29), we have for each t

$$
\begin{array} { r l r } {  { \frac { \mathbb { E } [ E _ { t } ] } { \eta K } \le \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } } } \\ & { } & { + \ 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { B _ { t } } + 2 \bar { \rho } \sqrt { B _ { t } } } \\ & { } & { + \ 2 L \bar { \rho } ( 1 - \beta _ { 1 } ) \sqrt { R _ { t } } + \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) \eta ( K - 1 ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma } \\ & { } & { + \ 2 \bar { \rho } \sqrt { D _ { t } } + 2 L \bar { \rho } ^ { 2 } K \eta + 2 L \bar { \rho } \sqrt { C _ { t - 1 } } . } \end{array}\tag{B.67}
$$

By Lemma B.12 and Lemma B.13,

$$
\sqrt { C _ { t - 1 } } \le \tau K \bar { \rho } \eta , \qquad \sqrt { R _ { t } } \le 3 \sqrt { 1 + \tau ^ { 2 } } K \bar { \rho } \eta .
$$

Substituting these bounds into (B.67), averaging over $t = 1 , \dots , T$ , and inserting into (B.66) gives

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( \bar { x } ^ { t } ) \| ^ { g , * } \leq \frac { \Delta _ { F } } { T \eta K } + \frac { L } { 2 } K \bar { \rho } ^ { 2 } \eta + \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } } } \\ & { } & { \quad \quad \quad \quad + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \cdot \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } + 2 \bar { \rho } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } } \\ & { } & { \quad \quad \quad \quad + 6 L \bar { \rho } ^ { 2 } ( 1 - \beta _ { 1 } ) \sqrt { 1 + \tau ^ { 2 } } K \eta + \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) \eta ( K - 1 ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma } \\ & { } & { \quad \quad \quad \quad + 2 \bar { \rho } \cdot \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } + 2 L \bar { \rho } ^ { 2 } K \eta + 2 \tau L \bar { \rho } ^ { 2 } K \eta . } \end{array}\tag{.68}
$$

We next make the $B _ { t }$ and $D _ { t }$ contributions fully explicit.

Step 1: an explicit average bound for $\textstyle { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \sqrt { B _ { t } } }$ . Starting from (B.33), write

$$
e _ { i } ^ { t + 1 } = ( 1 - \alpha ) e _ { i } ^ { t } + \alpha b _ { i } ^ { t } + \alpha n _ { i } ^ { t } - d _ { i } ^ { t } , \qquad \alpha : = 1 - \beta _ { 2 } ,
$$

where, conditionally on the history up to round $t , d _ { i } ^ { t }$ is measurable, $\mathbb { E } [ n _ { i } ^ { t } \mid \mathcal { F } _ { t } ] = 0$ , and $\mathbb { E } \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq$ $\sigma ^ { 2 } / K$ by (B.34). Hence the cross term involving $n _ { i } ^ { t }$ vanishes and

$$
\mathbb { E } \big [ \| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } \big ] = \big \| ( 1 - \alpha ) e _ { i } ^ { t } + \alpha b _ { i } ^ { t } - d _ { i } ^ { t } \big \| _ { 2 } ^ { 2 } + \alpha ^ { 2 } \mathbb { E } \big [ \| n _ { i } ^ { t } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } \big ] .
$$

Applying the weighted Young inequality $\begin{array} { r } { \| u + v \| _ { 2 } ^ { 2 } \leq ( 1 + \gamma ) \| u \| _ { 2 } ^ { 2 } + ( 1 + \gamma ^ { - 1 } ) \| v \| _ { 2 } ^ { 2 } } \end{array}$ with $u = ( 1 - \alpha ) e _ { i } ^ { t }$ $v = \alpha b _ { i } ^ { t } - d _ { i } ^ { t }$ , and $\gamma = \alpha / ( 2 ( 1 - \alpha ) )$ (the case $\alpha = 1$ is trivial), we get

$$
\big \| ( 1 - \alpha ) e _ { i } ^ { t } + \alpha b _ { i } ^ { t } - d _ { i } ^ { t } \big \| _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { \alpha } { 2 } \Big ) \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { 2 } { \alpha } \| \alpha b _ { i } ^ { t } - d _ { i } ^ { t } \| _ { 2 } ^ { 2 } .
$$

Using $\begin{array} { r } { \| u - v \| _ { 2 } ^ { 2 } \leq 2 \| u \| _ { 2 } ^ { 2 } + 2 \| v \| _ { 2 } ^ { 2 } } \end{array}$ , we therefore obtain

$$
\mathbb { E } \| e _ { i } ^ { t + 1 } \| _ { 2 } ^ { 2 } \leq \Big ( 1 - \frac { \alpha } { 2 } \Big ) \mathbb { E } \| e _ { i } ^ { t } \| _ { 2 } ^ { 2 } + 4 \alpha \mathbb { E } \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { 4 } { \alpha } \mathbb { E } \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } + \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { K } .\tag{B.69}
$$

By L-smoothness and Lemma B.13,

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| b _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq 2 L ^ { 2 } R _ { t } + 2 L ^ { 2 } ( K \bar { \rho } \eta ) ^ { 2 } \leq 2 0 L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } ,\tag{B.70}
$$

and

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| d _ { i } ^ { t } \| _ { 2 } ^ { 2 } \leq L ^ { 2 } R _ { t } \leq 9 L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } .\tag{B.71}
$$

Averaging (B.69) over i and using (B.70) and (B.71) yield

$$
B _ { t + 1 } \leq \Big ( 1 - \frac { \alpha } { 2 } \Big ) B _ { t } + \frac { 1 2 8 L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } } { \alpha } + \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { K } ,\tag{B.72}
$$

where we used $\alpha \leq 1$ to absorb the term proportional to α into the $\alpha ^ { - 1 }$ term. Let

$$
\delta : = \frac { \alpha } { 2 } , \qquad U _ { B } : = \frac { 1 2 8 L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } } { \alpha } + \frac { \alpha ^ { 2 } \sigma ^ { 2 } } { K } .
$$

Then (B.72) implies $B _ { t } \le ( 1 - \delta ) ^ { t - 1 } B _ { 1 } + U _ { B } / \delta$ . Using ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ and $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( \sqrt { 1 - \delta } ) ^ { t - 1 } \leq 2 / \delta } \end{array}$ and recalling that the initialization in the theorem statement implies $B _ { 1 } \leq \sigma ^ { 2 }$ , we obtain

$$
\begin{array} { r l r } {  { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } \leq \sqrt { \frac { U _ { B } } { \delta } } + \frac { 2 \sqrt { B _ { 1 } } } { \delta T } } } \\ & { } & { \leq \frac { 1 6 L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } } { \alpha } + \frac { \sqrt { 2 } \sigma \sqrt { \alpha } } { \sqrt { K } } + \frac { 4 \sigma } { \alpha T } . } \end{array}\tag{B.73}
$$

Step 2: an explicit average bound for $\textstyle { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } { \sqrt { D _ { t } } }$ . Exactly as in the proof of Lemma B.16, the same τ -window unrolling (with shorter initial windows when $t \leq \tau )$ and the initialization $D _ { 1 } = 0$ imply that for every $t \geq 1$

$$
\sqrt { D _ { t } } \leq \sum _ { r = \operatorname* { m a x } \{ 1 , t - \tau \} } ^ { t - 1 } \sqrt { \frac { 1 } { M } \mathbb { E } \| \mathbf { c } ^ { r + 1 } - \mathbf { c } ^ { r } \| _ { F } ^ { 2 } } ,
$$

where the sum is interpreted as 0 when $t = 1$ . Since $c _ { i } ^ { s + 1 } - c _ { i } ^ { s } = \alpha ( \hat { c } _ { i } ^ { s } - c _ { i } ^ { s } )$ 2

$$
\frac { 1 } { M } \mathbb { E } \| \mathbf { c } ^ { s + 1 } - \mathbf { c } ^ { s } \| _ { F } ^ { 2 } = \alpha ^ { 2 } \cdot \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| \hat { c } _ { i } ^ { s } - c _ { i } ^ { s } \| _ { 2 } ^ { 2 } .
$$

Now decompose

$$
\hat { c } _ { i } ^ { s } - c _ { i } ^ { s } = \underbrace { \left( \nabla f _ { i } ( x _ { i } ^ { s - 1 } ) - c _ { i } ^ { s } \right) } _ { = : a _ { i } ^ { s } } + \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( \nabla f _ { i } ( x _ { i } ^ { s , k - 1 } ) - \nabla f _ { i } ( x _ { i } ^ { s - 1 } ) ) } _ { = : b _ { i } ^ { s } } + \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \zeta _ { i } ^ { s , k - 1 } } _ { = : n _ { i } ^ { s } } .
$$

Using $\| u + v + w \| _ { 2 } ^ { 2 } \leq 3 ( \| u \| _ { 2 } ^ { 2 } + \| v \| _ { 2 } ^ { 2 } + \| w \| _ { 2 } ^ { 2 } )$ , Jensen, Assumption 2, and (B.70), we obtain

$$
\frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { E } \| \hat { c } _ { i } ^ { s } - c _ { i } ^ { s } \| _ { 2 } ^ { 2 } \leq 3 B _ { s } + 6 0 L ^ { 2 } ( 1 + \tau ^ { 2 } ) ( K \bar { \rho } \eta ) ^ { 2 } + \frac { 3 \sigma ^ { 2 } } { K } .
$$

Therefore,

$$
\sqrt { \frac { 1 } { M } \mathbb { E } \| \mathbf { c } ^ { s + 1 } - \mathbf { c } ^ { s } \| _ { F } ^ { 2 } } \leq \alpha \Big ( \sqrt { 3 } \sqrt { B _ { s } } + 2 \sqrt { 1 5 } L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } + \frac { \sqrt { 3 } \sigma } { \sqrt { K } } \Big ) .\tag{B.74}
$$

Summing the FTC window bound over $t = 1 , \dots , T$ and using that each index appears in at most τ windows yields

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } \le \sqrt { 3 } \tau \alpha \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { B _ { t } } + 2 \sqrt { 1 5 } \tau \alpha L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } + \frac { \sqrt { 3 } \tau \alpha \sigma } { \sqrt { K } } .\tag{B.75}
$$

Substituting (B.73) into (B.75) and using $\alpha \leq 1$ yields

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sqrt { D _ { t } } \leq ( 1 6 \sqrt { 3 } + 2 \sqrt { 1 5 } ) \tau L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } + \frac { ( \sqrt { 6 } + \sqrt { 3 } ) \tau \alpha \sigma } { \sqrt { K } } + \frac { 4 \sqrt { 3 } \tau \sigma } { T } .\tag{B.76}
$$

Substituting (B.73) and (B.76) into (B.68), we obtain

$$
\begin{array} { l } { \displaystyle \frac { 1 } { T } \sum _ { \ell = 1 } ^ { T } \mathbb { E } \| \nabla F ( \overline { { x } } ^ { \ell } ) \| ^ { \ell * } \leq \frac { \Delta _ { F } } { T \eta \mathcal { H } } + \frac { L } { 2 } K \bar { \rho } ^ { 2 } \eta + \Big ( 2 + \frac { 1 } { 2 K } \Big ) \bar { \rho } ^ { 2 } \eta + ( 1 - \beta _ { 1 } ) ^ { 2 } L ^ { 2 } \bar { \rho } ^ { 2 } \eta + \frac { 2 ( 1 - \beta _ { 1 } ) ^ { 2 } \sigma ^ { 2 } } { \eta } } \\ { \displaystyle \qquad + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \Big [ \frac { 1 6 L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } } { \alpha } + \frac { \sqrt { 2 } \sigma \sqrt { \alpha } } { \sqrt { K } } + \frac { 4 \sigma } { \alpha T } \Big ] } \\ { \displaystyle \qquad + 2 \bar { \rho } \Big [ \frac { 1 6 L K \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } } { \alpha } + \frac { \sqrt { 2 } \sigma \sqrt { \alpha } } { \sqrt { K } } + \frac { 4 \sigma } { \alpha T } \Big ] + 6 L \bar { \rho } ^ { 2 } ( 1 - \beta _ { 1 } ) \sqrt { 1 + \tau ^ { 2 } } K \eta } \\ { \displaystyle \qquad + \bar { \rho } ^ { 2 } L ( 1 - \beta _ { 1 } ) \eta ( K - 1 ) + 2 \bar { \rho } ( 1 - \beta _ { 1 } ) \sigma } \\ { \displaystyle \qquad + 2 \bar { \rho } ( 1 6 \sqrt { 3 } + 2 \sqrt { 1 5 } ) \tau L \bar { \rho } \eta \sqrt { 1 + \tau ^ { 2 } } } \\ { \displaystyle \qquad + \frac { 2 \bar { \rho } ( \sqrt { 6 } + \sqrt { 3 } ) \tau \alpha } { \sqrt { K } } + \frac { 8 \sqrt { 3 } \bar { \rho } \tau \sigma } { \sqrt { \pi } \sigma } } \\ { \displaystyle \qquad + 2 L \bar { \rho } ^ { 2 } K \eta + 2 \tau L \bar { \rho } ^ { 2 } K \eta . } \end{array}
$$

We now substitute the hyperparameter choice

$$
\eta = \frac { M ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } , \qquad 1 - \beta _ { 1 } = \eta , \qquad \alpha = 1 - \beta _ { 2 } = \sqrt { \frac { M K } { T + K } } , \qquad \mathrm { a n d ~ a s s u m e ~ } T \geq M K .
$$

Since $T \geq M K$ and $M \geq 1$ , we also have $K \leq T$ and hence $T + K \leq 2 T$ . Moreover,

$$
\begin{array} { r l } & { \quad K \eta = \frac { M I ^ { 1 / 4 } K ^ { 1 / 4 } } { T ^ { 9 / 4 } } \leq \frac { 1 } { \sqrt { T } } , \qquad \eta = \frac { 1 } { ( M K T ) ^ { 1 / 4 } } \sqrt { \frac { M } { K T } } \leq \frac { 1 } { ( M K T ) ^ { 1 / 4 } } , } \\ & { \frac { K \eta } { \alpha } = \frac { \sqrt { T } + K } { 3 M ^ { 1 / 4 } K ^ { 1 / 1 2 } \sqrt { 4 \eta } } \leq \frac { \sqrt { 2 } } { ( M K T ) ^ { 1 / 4 } } , \qquad \sqrt { \alpha } = \frac { M ^ { 1 / 4 } } { K ^ { 1 / 4 } ( T + K ) ^ { 1 / 4 } } \leq \frac { \sqrt { M } } { ( M K T ) ^ { 1 / 4 } } , } \\ & { \frac { \alpha } { \sqrt { K } } = \sqrt { \frac { M } { T + K } } \leq \sqrt { \frac { M } { T } } , \qquad \frac { 1 } { \alpha T } = \frac { \sqrt { T } + K } { \sqrt { M K T } } \leq \frac { \sqrt { 2 } } { \sqrt { M K T } } , } \\ & { \quad K \eta ^ { 2 } = \frac { \sqrt { M } } { \sqrt { K T ^ { 3 / 2 } } } , \qquad \frac { K \eta ^ { 2 } } { \alpha } = \frac { \sqrt { T } + K } { K T ^ { 3 / 2 } } \leq \frac { \sqrt { 2 } } { K T ^ { 2 } } , } \\ & { \frac { \eta \sqrt { \alpha } } { \sqrt { K } } = \frac { \sqrt { M } } { K T ^ { 3 / 4 } ( T + K ) ^ { 1 / 4 } } \leq \frac { \sqrt { M } } { K T } , \qquad \frac { \eta } { \alpha T } = \frac { \sqrt { T } + K } { M ^ { 1 / 4 } K ^ { 5 / 4 } T ^ { 7 / 4 } } \leq \frac { \sqrt { 2 } } { M ^ { 1 / 4 } K ^ { 5 / 4 } T ^ { 5 / 4 } } , } \\ & { \quad \eta = \frac { M ^ { 1 / 4 } } { K ^ { 3 / 4 } T ^ { 3 / 4 } } \leq \frac { 1 } { K \sqrt { T } } . } \end \end{array}\tag{B.78}
$$

Also,

$$
\frac { \Delta _ { F } } { T \eta K } = \frac { \Delta _ { F } } { ( M K T ) ^ { 1 / 4 } } .\tag{B.79}
$$

Using (B.78) and (B.79) in (B.77), and bounding $\begin{array} { r } { 2 + \frac { 1 } { 2 K } \le \frac { 5 } { 2 } } \end{array}$ , we arrive at the explicit estimate

$$
\begin{array} { r l } { \frac { 1 } { T } \underset { \leq - 1 } { \overset { \mathcal { T } } { \geq } } \mathbb { E } \| \nabla F ( \bar { z } ^ { * } ) \| ^ { 2 s ^ { * } } \leq } & { \left[ \begin{array} { l } { \frac { L \bar { \mathcal { F } } ^ { 2 } } { 2 } + 2 L \bar { \rho } ^ { 2 } + 2 \eta L \bar { \rho } ^ { 2 } } \\ { + \left( 3 \bar { 2 } \sqrt { 3 } - 4 \sqrt { 1 5 } \right) T \bar { \rho } ^ { 2 } \sqrt { 1 + \tau ^ { 2 } } + 2 ( \sqrt { 6 } + \sqrt { 3 } ) \tau \bar { \rho } \sqrt { M } } \\ { + \left[ 3 \bar { 2 } \sqrt { 3 } - 4 \bar { 2 } \sqrt { 3 } \right] \tau \bar { \rho } ^ { 2 } \sqrt { 1 + \tau ^ { 2 } } + 2 \sigma ^ { 2 } } \end{array} \right] \frac { 1 } { T \bar { \mathcal { W } } } } \\ & { \quad + \left[ \begin{array} { l } { \Delta _ { \phi } + \frac { 5 } { 2 } \varepsilon ^ { 2 } + L ^ { 2 } \rho ^ { 2 } + 2 \sigma ^ { 2 } } \\ { + 3 2 \sqrt { 3 } \tau \bar { L } \rho ^ { 2 } \sqrt { 1 + \tau ^ { 2 } } + 2 \sqrt { 2 } \bar { \rho } \sqrt { M } } \\ { + \frac { 8 \bar { L } \rho ^ { 2 } \sqrt { 1 1 + \tau ^ { 2 } } \sqrt { 3 } \bar { M } } { \sqrt { 1 7 } \mathrm { ~ \rho ~ \varepsilon ~ } } } \end{array} \right] \frac { 1 } { ( M K T ) ^ { 1 / 4 } } } \\ & { \quad + \frac { 6 \hat { L } \bar { \rho } ^ { 2 } \sqrt { 1 1 + \tau ^ { 2 } } \sqrt { 3 } \bar { M } } { \sqrt { K } T \mathrm { ~ \rho ~ \varepsilon ~ } } + \frac { L \bar { \rho } ^ { 2 } \sqrt { 3 } \bar { L } \bar { \rho } } { \sqrt { K } T \mathrm { ~ \hat { \rho } ~ } } } \\ &  \quad + \frac  3 2 \sqrt { 2 } \bar { \rho } ^ { 2 } \ \end{array}\tag{B.80}
$$

The first line is of order $T ^ { - 1 / 2 }$ and the second is of order $( M K T ) ^ { - 1 / 4 }$ . The corrected query-wise noise contribution is $2 \bar { \rho } \sigma \eta .$ , which is explicitly retained through the first term of $\mathcal { R } _ { \mathrm { l o } }$ . Its coeficient is independent of $( T , K , M )$ , and $\eta \le 1 / ( K \sqrt { T } )$ under $M K \leq T$ . The remaining terms are dominated by the other remainder terms. Hence the stated $C _ { 1 }$ and $C _ { 2 }$ are unchanged, and a suficiently large $C _ { 0 }$ , with the stated parameter dependence, completes the proof. □

## C PolyGram-RCClip: Degree–Residual Certificates for Spectral Clipping

This appendix develops the SVD-free spectral clipping oracle used for matrix blocks with spectralnorm geometry. The key point is that a Gram-polynomial map can be made feasible for the spectral-norm ball and can be assigned a residual certificate compatible with Theorem 6.

## C.1 Setup and notation

For a matrix $\boldsymbol { Z } \in \mathbb { R } ^ { m \times n }$ and a clipping radius $\tau > 0$ , define the spectral-norm ball $B _ { \tau } : = \{ Y \in$ $\mathbb { R } ^ { m \times n } : \| Y \| _ { 2 } \leq \tau \} . \mathrm { ~ I f ~ } \| Z \| _ { 2 } \leq \tau$ , the oracle returns $Y = Z$ and the residual is zero. We therefore consider the nontrivial case $0 < \tau < \| Z \| _ { 2 } \leq s$ , where s is any safe upper scale. Set $\alpha : = \tau / s \in ( 0 , 1 )$ and define the exact clipping transfer in the Gram variable by

$$
\phi _ { \alpha } ( \lambda ) : = \operatorname * { m i n } \left\{ 1 , \frac { \alpha } { \sqrt { \lambda } } \right\} , \qquad \phi _ { \alpha } ( 0 ) : = 1 , \qquad \lambda \in [ 0 , 1 ] .
$$

For a degree-d polynomial $p _ { d } .$ , PolyGram-RCClip uses the map

$$
Y _ { d , s } = Z p _ { d } ( Z ^ { \top } Z / s ^ { 2 } ) \quad ( m \geq n ) , \qquad Y _ { d , s } = p _ { d } ( Z Z ^ { \top } / s ^ { 2 } ) Z \quad ( m < n ) .
$$

If $Z = U \mathrm { d i a g } ( \sigma _ { i } ) V ^ { \top }$ and $\lambda _ { i } = \sigma _ { i } ^ { 2 } / s ^ { 2 }$ , both forms preserve the singular vectors and transform singular values as

$$
\widetilde { \sigma } _ { i } = \sigma _ { i } p _ { d } ( \lambda _ { i } ) .
$$

For any feasible output $Y \in B _ { \tau }$ , the single-call variational residual is

$$
\mathrm { r e s } _ { { B _ { \tau } } } ( Z , Y ) : = \left[ \operatorname* { s u p } _ { U \in { B _ { \tau } } } \langle Z - Y , U - Y \rangle _ { { F } } \right] _ { + } .
$$

Using duality between the spectral and nuclear norms,

$$
\mathrm { r e s } _ { { \mathscr { B } _ { \tau } } } ( Z , Y ) = [ \tau \| Z - Y \| _ { * } - \langle Z - Y , Y \rangle _ { F } ] _ { + } .
$$

## C.2 Existence and degree–scale residual bound

Let $\mathcal { P } _ { d , \alpha } ^ { \mathrm { f e a s } }$ be the class of degree-d polynomials satisfying

$$
\begin{array} { r } { p _ { d } ( \lambda ) \geq 0 , \qquad \sqrt { \lambda } p _ { d } ( \lambda ) \leq \alpha , \qquad \forall \lambda \in [ 0 , 1 ] , } \end{array}
$$

and define the optimal feasible fitting radius

$$
\delta _ { d } ^ { \star } ( \alpha ) : = \operatorname* { i n f } _ { p \in \mathcal { P } _ { d , \alpha } ^ { \mathrm { f e a s } } } \| p - \phi _ { \alpha } \| _ { \infty , [ 0 , 1 ] } .
$$

Proposition C.1 (Residual controlled by feasible fitting radius). Let $p _ { d } \in \mathcal { P } _ { d , \alpha } ^ { \mathrm { f e a s } }$ and define $Y _ { d , s }$ by the Gram-polynomial map above. $I f \| p _ { d } - \phi _ { \alpha } \| _ { \infty , [ 0 , 1 ] } \leq \delta$ with $0 \leq \delta \leq 1$ , then $\dot { Y } _ { d , s } \in \mathcal { B } _ { \tau }$ and

$$
\mathrm { r e s } _ { {  { \mathcal { B } } } _ { \tau } } ( Z , Y _ { d , s } ) \leq \delta \left( \tau \| Z \| _ { * } + 2 \| Z \| _ { F } ^ { 2 } \right) .\tag{C.1}
$$

Consequently, with $q = \operatorname* { m i n } \{ m , n \}$ ，

$$
\mathrm { r e s } _ { { \mathscr { B } _ { \tau } } } ( Z , Y _ { d , s } ) \leq \delta \left( \tau \sqrt { q } \| Z \| _ { F } + 2 \| Z \| _ { F } ^ { 2 } \right) .\tag{C.2}
$$

Proof. Let $q _ { i } = p _ { d } ( \lambda _ { i } )$ . Feasibility gives $\sigma _ { i } q _ { i } = s \sqrt { \lambda _ { i } } p _ { d } ( \lambda _ { i } ) \leq s \alpha = \tau$ , so $Y _ { d , s } \in B _ { \tau }$ . The supportfunction form of the residual gives

$$
\| Z - Y _ { d , s } \| _ { * } = \sum _ { i } \sigma _ { i } | 1 - q _ { i } | , \qquad \langle Z - Y _ { d , s } , Y _ { d , s } \rangle _ { F } = \sum _ { i } \sigma _ { i } ^ { 2 } q _ { i } ( 1 - q _ { i } ) .
$$

If $\sigma _ { i } \leq \tau$ , then $\phi _ { \alpha } ( \lambda _ { i } ) = 1 , \mathrm { s o } \ | 1 - q _ { i } | \leq \delta$ . Since $q _ { i } \geq 0$ and $\delta \leq 1$ , also $q _ { i } \leq 2$ , and the contribution of this index is at most $\delta \tau \sigma _ { i } + 2 \delta \sigma _ { i } ^ { 2 }$ . If $\sigma _ { i } > \tau ,$ then $\phi _ { \alpha } ( \lambda _ { i } ) = \tau / \sigma _ { i }$ . Feasibility implies $q _ { i } \leq \tau / \sigma _ { i }$ , so $q _ { i } = \tau / \sigma _ { i } - e _ { i }$ for some $e _ { i } \in [ 0 , \delta ]$ . Hence $\tau - \sigma _ { i } q _ { i } = \sigma _ { i } e _ { i }$ and $1 - q _ { i } \ge 0$ , so this index contributes at most $\delta \sigma _ { i } ^ { 2 }$ . Summing over i proves (C.1), and (C.2) follows from $\| Z \| _ { * } \leq \sqrt { q } \| Z \| _ { F }$ □

Lemma C.2 (Jackson direct theorem). For a continuous function f on $[ 0 , 1 ]$ , define the best degree-d uniform approximation error and the ordinary modulus of continuity by

$$
E _ { d } ( f ; [ 0 , 1 ] ) : = \operatorname* { i n f } _ { \deg q \leq d } \| f - q \| _ { \infty , [ 0 , 1 ] } , \qquad \omega ( f , t ; [ 0 , 1 ] ) : = \operatorname* { s u p } _ { x , y \in [ 0 , 1 ] } | f ( x ) - f ( y ) | .
$$

There exists a universal constant $C _ { J } ^ { \mathrm { J a c } } > 0$ such that, for every $d \geq 1$

$$
E _ { d } ( f ; [ 0 , 1 ] ) \le C _ { J } ^ { \mathrm { J a c } } \omega ( f , 1 / d ; [ 0 , 1 ] ) .
$$

Consequently, if f is $L _ { f } – L i p s c h i t z$ on $[ 0 , 1 ]$ , then $E _ { d } ( f ; [ 0 , 1 ] ) \leq C _ { J } ^ { \mathrm { J a c } } L _ { f } / d$ . This is the classical Jackson direct theorem for algebraic polynomial approximation $[ 5 , \ 6 , \ g ]$

Theorem C.3 (Degree–scale residual certificate). There exists a universal constant $C _ { J } > 0$ such that the following holds. In the nontrivial case $0 < \tau < \| Z \| _ { 2 } \leq s ,$ let $\alpha = \tau / s$ and

$$
d _ { 0 } ( \alpha ) : = \left\lceil \operatorname* { m a x } \left\{ 1 , \frac { C _ { J } } { \alpha ^ { 3 } } \right\} \right\rceil .
$$

For every $d \geq d _ { 0 } ( \alpha )$ , there exists a degree-d polynomial $p _ { d }$ satisfying $0 \leq p _ { d } ( \lambda ) \leq \phi _ { \alpha } ( \lambda )$ for all $\lambda \in [ 0 , 1 ]$ and

$$
\| p _ { d } - \phi _ { \alpha } \| _ { \infty , [ 0 , 1 ] } \leq \frac { C _ { J } } { \alpha ^ { 2 } d } = \frac { C _ { J } s ^ { 2 } } { \tau ^ { 2 } d } .
$$

Consequently, there exists a feasible degree-d Gram-polynomial clipping map satisfying

$$
\mathrm { r e s } _ { {  { \mathcal { B } } } _ { \tau } } ( Z , Y _ { d , s } ) \leq \frac { C _ { J } s ^ { 2 } } { \tau ^ { 2 } d } \left( \tau \| Z \| _ { * } + 2 \| Z \| _ { F } ^ { 2 } \right) .\tag{C.3}
$$

Proof. The clipping transfer $\phi _ { \alpha }$ is constant on $[ 0 , \alpha ^ { 2 } ]$ and has derivative $- \alpha / ( 2 \lambda ^ { 3 / 2 } )$ on $( \alpha ^ { 2 } , 1 ]$ Hence $\phi _ { \alpha }$ is Lipschitz on [0, 1] with $\mathrm { L i p } ( \phi _ { \alpha } ) \leq 1 \bar { / } ( 2 \alpha ^ { 2 } )$ ). Applying Lemma C.2 with $f = \phi _ { \alpha } ,$ and absorbing universal numerical factors into $C _ { J }$ , gives a degree-d polynomial $q _ { d }$ such that

$$
\| q _ { d } - \phi _ { \alpha } \| _ { \infty , [ 0 , 1 ] } \leq \frac { C _ { J } } { 2 \alpha ^ { 2 } d } = : \varepsilon _ { d } .
$$

Define the shifted polynomial $p _ { d } : = q _ { d } - \varepsilon _ { d }$ . Then $p _ { d } \leq \phi _ { \alpha }$ pointwise and

$$
\| p _ { d } - \phi _ { \alpha } \| _ { \infty , [ 0 , 1 ] } \leq 2 \varepsilon _ { d } = \frac { C _ { J } } { \alpha ^ { 2 } d } .
$$

Since $\phi _ { \alpha } ( \lambda ) \ \geq \ \alpha$ for all $\lambda ~ \in ~ [ 0 , 1 ]$ , the condition $d \ge d _ { 0 } ( \alpha )$ implies $2 \varepsilon _ { d } \ \leq \ \alpha$ and therefore $p _ { d } ( \lambda ) \ge \phi _ { \alpha } ( \lambda ) - 2 \varepsilon _ { d } \ge 0$ . Thus $0 \leq p _ { d } \leq \phi _ { \alpha }$ , which also implies $\sqrt { \lambda } p _ { d } ( \lambda ) \leq \sqrt { \lambda } \phi _ { \alpha } ( \lambda ) \leq \alpha$ Proposition C.1 then yields (C.3). □

Cheap scales. The bound is valid for every $s \geq \| Z \| _ { 2 }$ . Thus a tighter scale improves the certificate through the factor $s ^ { 2 } / \tau ^ { 2 }$ . The deterministic choice $s = \| Z \| _ { F }$ is always valid and gives

$$
\mathrm { r e s } _ { \mathcal { B } _ { \tau } } ( Z , Y _ { d , \| Z \| _ { F } } ) \leq \frac { C _ { J } \| Z \| _ { F } ^ { 2 } } { \tau ^ { 2 } d } \left( \tau \| Z \| _ { * } + 2 \| Z \| _ { F } ^ { 2 } \right) .
$$

Using $\| Z \| _ { * } \le \sqrt { q } \| Z \| _ { F }$ and $\tau < \| Z \| _ { 2 } \leq \| Z \| _ { F }$ yields the Frobenius-only certificate

$$
\mathrm { r e s } _ { { \mathscr { B } } _ { \tau } } ( Z , Y _ { d , \| Z \| _ { F } } ) \leq \frac { C _ { J } ( \sqrt { q } + 2 ) } { d } \frac { \| Z \| _ { F } ^ { 4 } } { { \tau } ^ { 2 } } .
$$

## C.3 Coeficient design as a semi-infinite LP

After scaling by $s ,$ all singular values enter through $\lambda = \sigma ^ { 2 } / s ^ { 2 } \in [ 0 , 1 ]$ , and the exact clipping transfer is determined only by the scalar $\alpha = \tau / s$ . Thus, for a fixed degree $d ,$ coeficient design is a one-dimensional family indexed by $\alpha ,$ rather than by $Z , \tau , s$ separately. Write $p _ { d , \alpha } ( \lambda ) =$ $\begin{array} { r } { \sum _ { j = 0 } ^ { d } c _ { j } ( \alpha ) T _ { j } ( 2 \lambda - 1 ) } \end{array}$ in a Chebyshev basis. The continuous coeficient-design problem is

$$
\begin{array} { r l r l } { \underset { c _ { 0 } , \ldots , c _ { d } , \delta } { \operatorname* { m i n } } } & { \delta } & & { } \\ { \mathrm { s . t . } } & { p _ { d , \alpha } ( \lambda ) \geq 0 , } & & { \sqrt { \lambda } p _ { d , \alpha } ( \lambda ) \leq \alpha , } & & { \forall \lambda \in [ 0 , 1 ] , } \\ & { p _ { d , \alpha } ( \lambda ) - \phi _ { \alpha } ( \lambda ) \leq \delta , } & & { \phi _ { \alpha } ( \lambda ) - p _ { d , \alpha } ( \lambda ) \leq \delta , } & & { \forall \lambda \in [ 0 , 1 ] , } \\ & { \delta \geq 0 . } \end{array}\tag{C.4}
$$

The value of this semi-infinite LP is $\delta _ { d } ^ { \star } ( \alpha )$ , and Theorem C.3 gives a degree–scale upper bound on this value.

## C.4 Table-based PolyGram-RCClip oracle

Because the LP in Appendix C.3 is indexed only by α, the runtime oracle does not solve an LP. We precompute a table $\mathcal { T } = \{ ( \alpha _ { j } , c _ { 0 } ^ { ( j ) } , \ldots , c _ { d } ^ { ( j ) } ) \} _ { j = 1 } ^ { J }$ . In the experiments, $d = 2 0$ and $J = 1 0 0 0$ , so the table contains only $J ( d + 1 ) = 2 1$ ,000 scalar coeficients.

Algorithm C.1 gives the practical fixed-degree runtime oracle used in the experiments. The safe scale is chosen deterministically. Let $G _ { 0 } = Z ^ { \top } Z$ if $m \geq n$ and $G _ { 0 } = Z Z ^ { \top }$ otherwise. We use $s _ { \mathrm { s a f e } } : = \operatorname* { m i n } \{ \| Z \| _ { F } , \sqrt { \| Z \| _ { 1 } \| Z \| _ { \infty } } , \sqrt { \| G _ { 0 } \| _ { \infty } } \}$ and $s : = 1 . 0 0 0 1 s _ { \mathrm { s a f e } } .$ , where $\| Z \| _ { 1 }$ is the maximum absolute column sum and $\| Z \| _ { \infty }$ is the maximum absolute row sum. Each term in the minimum upper bounds $\| Z \| _ { 2 }$ , hence $s \geq \| Z \| _ { 2 }$ . The identity-return check uses the strict bound $s _ { \mathrm { s a f e } } \leq \tau .$ , in which case $Z \in B _ { \tau }$ .

The scale bound alone does not certify the entire oracle. The residual theorem additionally requires continuous-interval feasibility and fitting-error bounds for the selected polynomial. Floor lookup and low-α rescaling preserve feasibility in exact arithmetic when the table entries are feasible on the full interval, but their approximation errors must be included in the residual budget. A discretized LP and floating-point evaluation require additional verification for a rigorous certificate. We do not claim that the experimental fixed choice $d = 2 0$ satisfies the accumulated budget of Theorem 6 for arbitrary horizons.

Algorithm C.1 Table-based PolyGram-RCClip for one matrix block   
1: Input: matrix $Z \in \mathbb { R } ^ { m \times n }$ , radius $\tau ,$ safe scale $s \geq \| Z \| _ { 2 } ,$ coeficient table $\mathcal { T } = \{ ( \alpha _ { j } , c ^ { ( j ) } ) \}$ , degree d.   
2: if $s \leq \tau$ then   
3: return $Z$   
4: end if   
5: Set $\alpha _ { \mathrm { c l i p } }  \tau / s$ and let $\alpha _ { \mathrm { { m i n } } }$ be the smallest grid value in $\tau .$   
6: if $\alpha _ { \mathrm { c l i p } } < \alpha _ { \mathrm { m i n } }$ then   
7: Load coeficients for $\alpha _ { \mathrm { m i n } }$ and set $r  \alpha _ { \mathrm { c l i p } } / \alpha _ { \mathrm { m i n } }$   
8: else   
9: Choose the largest grid value $\tilde { \alpha } \leq \alpha _ { \mathrm { c l i p } }$ , load its coeficients, and set $r \gets 1$   
10: end if   
11: Write the selected coeficients as $c _ { 0 } , \ldots , c _ { d }$ in the basis $T _ { j } ( 2 \lambda - 1 )$   
12: if $m \geq n$ then   
13: Form $\widetilde { G } \gets 2 Z ^ { \top } Z / s ^ { 2 } - I _ { n }$ and set $V _ { 0 }  Z , Y  c _ { 0 } V _ { 0 }$   
14: if $d \geq 1$ then   
15: $V _ { 1 }  V _ { 0 } \widetilde { G } , Y  Y + c _ { 1 } V _ { 1 } .$   
16: for $j = 1 , \dots , d - 1$ do   
17: $V _ { j + 1 } \gets 2 V _ { j } \widetilde { G } - V _ { j - 1 } ; \quad Y \gets Y + c _ { j + 1 } V _ { j + 1 } .$   
18: end for   
19: end if   
20: else   
21: Form $\tilde { H } \gets 2 Z Z ^ { \top } / s ^ { 2 } - I _ { m }$ and set $V _ { 0 }  Z , Y  c _ { 0 } V _ { 0 }$   
22: if $d \geq 1$ then   
23: $V _ { 1 }  \widetilde { H } V _ { 0 } , Y  Y + c _ { 1 } V _ { 1 } .$   
24: for $j = 1 , \dotsc , d - 1$ do   
25: $\begin{array} { r } { V _ { j + 1 }  2 \widetilde { H } V _ { j } - V _ { j - 1 } ; \quad Y  Y + c _ { j + 1 } V _ { j + 1 } . } \end{array}$   
26: end for   
27: end if   
28: end if   
29: return $r Y .$

After the smaller Gram matrix is formed, Algorithm C.1 uses exactly d sequential Gram-action multiplications to evaluate a degree-d transfer. Thus the table degree is both the approximation degree in Theorem C.3 and the main compute knob of the oracle.

## D Additional preliminaries: uSCG template

For completeness, we recall the Unconstrained Stochastic Conditional Gradient (uSCG) template [24], which provides the LMO-based abstraction used in Section 2.2 to motivate norm-ball update geometries.

```latex
Algorithm D.1 Unconstrained Stochastic Conditional Gradient (uSCG) [24]
Input: Initial iterate $x ^ { 1 } \in \mathbb { R } ^ { d } ,$ , momentum factor $\beta \in [ 0 , 1 ]$ , constraint set D.
Initialize momentum bufer m $\mathbf { \partial } ^ { - 1 } = \mathbf { 0 } .$
for $t = 1$ to $T$ do
Compute stochastic gradient $\boldsymbol { g } ^ { t } = \nabla f ( \boldsymbol { x } ^ { t } , \xi _ { t } )$
Update momentum:
$m ^ { t } = ( 1 - \beta ) m ^ { t - 1 } + \beta g ^ { t } .$
Compute update direction via Linear Minimization Oracle (LMO):
$\Delta ^ { t } \in \underset { \Delta \in \mathcal { D } } { \arg \operatorname* { m i n } } \langle m ^ { t } , \Delta \rangle .$
Update parameters: $\boldsymbol { x } ^ { t + 1 } = \boldsymbol { x } ^ { t } + \boldsymbol { \Delta } ^ { t }$
end for
```

## E Experimental details and additional results

## E.1 Model architecture

We use a CIFAR-style ResNet-18 for $3 2 \times 3 2$ inputs. The stem is a bias-free $3 \times 3$ convolution from 3 to 64 channels, followed by GroupNorm with 8 groups and a GELU activation; there is no ImageNet-style large stem or max-pooling layer. Each residual block contains two bias-free $3 \times 3$ convolutions, GroupNorm after each convolution, GELU after the first normalization, and GELU after the residual addition. Downsampling blocks use a bias-free $1 \times 1$ convolution followed by GroupNorm. The four stages have widths 64, 128, 256, 512, with two basic blocks per stage and strides 1, 2, 2, 2. The network ends with adaptive average pooling and a bias-free linear classifier $\mathbb { R } ^ { 5 1 2 }  \mathbb { R } ^ { 1 0 }$ . All GroupNorm layers keep the trainable scale parameter but remove the bias parameter. In the main NBCLT experiments, we use Scion initialization.

## E.2 Baselines and tuning

For each Dirichlet parameter, we tune each baseline over its own hyperparameter grid and select the best run by average test accuracy over the last 100 communication rounds. All baselines and NBCLT variants use the shared weight-decay grid wd $\in \{ 0 , 1 0 ^ { - 3 } \}$ . We use lr for client-side local learning rates, and η or $\eta _ { g }$ for server-side learning rates following the notation of the corresponding methods. Momentum parameters are denoted by $\beta ,$ or by $( \beta _ { 1 } , \beta _ { 2 } )$ when the method uses two such parameters. For FedMuon [31], Muon and SGD layers have separate stepsizes, and α denotes the momentum parameter; our implementation uses exact orthogonalization for LMO-based layers rather than a Newton–Schulz approximation.

Table 3: Hyperparameter grids for CIFAR-10 FL experiments. All methods additionally sweep wd $\in \{ 0 , 1 0 ^ { - 3 } \}$  
Method Grid   
FedAvg $\mathrm { { l r } } \in \lbrace 1 0 ^ { - 3 } , 3 \times 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 3 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \rbrace ;$ client momentum $\in \{ 0 . 9 , 0 . 9 5 \}$   
SCAFFOLD-M local l $\cdot \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 3 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ ; server $\eta _ { g } \in \{ 1 , 0 . 1 \} ; \beta \in \{ 0 . 9 , 0 . 9 5 \}$   
FedAMS local $\mathrm { l r } \in \lbrace 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 3 { \times } 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \rbrace$ ; server $\eta \in \{ 1 , 0 . 1 \} ; \beta _ { 1 } = 0 . 9 ; \beta _ { 2 } \in \{ 0 . 9 9 , 0 . 9 9 9 \} ;$   
$\epsilon = 1 0 ^ { - 6 } .$   
FedMuon Muon stepsize $\in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 4 } \}$ ; SGD stepsize $\in \{ 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \} ; \alpha \in \{ 0 . 0 5 , 0 . 1 \}$   
PAdaMFed local l $\cdot \in \{ 1 0 ^ { - 2 } , 3 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ ; server $\eta _ { g } \in \{ 1 , 0 . 1 \} ; \beta \in \{ 0 . 9 , 0 . 9 5 \}$   
NBCLT tuned $\eta \in \{ 0 . 0 0 3 , 0 . 0 1 \} ; \beta _ { 1 } \in \{ 0 . 9 , 0 . 9 5 \} ; \beta _ { 2 } \in \{ 0 . 7 5 , 0 . 9 , 0 . 9 5 \} .$

For NBCLT-PF, we do not tune $\eta , \beta _ { 1 } , \beta _ { 2 } ;$ we use the theorem-suggested values reported in Section 5 and only apply the shared weight-decay $\mathrm { g r i d } .$

## E.3 PolyGram-RCClip implementation

The exact-SVD implementation projects each spectral matrix block by clipping singular values to the block radius. PolyGram-RCClip replaces this training-time SVD by the table-based Grampolynomial oracle in Appendix C. For a matrix block Z and radius $\tau ,$ , the reported PolyGram-RCClip curves use the certified safe-min scale from Appendix C.4: with $G _ { 0 } = Z ^ { \top } Z { \mathrm { ~ i f ~ } } m \geq n$ and $G _ { 0 } = Z Z ^ { \top }$ otherwise,

$$
s = 1 . 0 0 0 1 \cdot \operatorname* { m i n } \left\{ \| Z \| _ { F } , \sqrt { \| Z \| _ { 1 } \| Z \| _ { \infty } } , \sqrt { \| G _ { 0 } \| _ { \infty } } \right\} .
$$

The clipping ratio $\alpha _ { \mathrm { c l i p } } = \tau / s$ is distinct from the Dirichlet parameter $\alpha _ { \mathrm { d i r } }$ . We use a precomputed Chebyshev coeficient table over $\alpha _ { \mathrm { c l i p } } \in [ 1 0 ^ { - 3 } , 0 . 9 9 9 ]$ with degree $d = 2 0$ and 1000 grid points; training-time calls use floor lookup and evaluate either $Z p _ { d } ( Z ^ { \top } Z / s ^ { 2 } )$ or $p _ { d } ( Z Z ^ { \top } / s ^ { 2 } ) Z$ , choosing the smaller Gram matrix. For the hardest split $\alpha _ { \mathrm { d i r } } = 0 . 1$ , we also ran additional experiments with diferent polynomial degrees d while keeping the same certified scale rule.

## E.4 Norm geometry and block scales

NBCLT uses block-wise norm-ball projection with layer-dependent radii $\tau _ { \ell } = \eta r _ { \ell }$ . In the CIFAR-10 ResNet-18 experiments, $3 \times 3$ convolutions and $1 \times 1$ downsampling convolutions use spectral-norm clipping, while GroupNorm weights and the final classifier head use $\ell _ { 2 }$ clipping. For a convolutional kernel $\Delta \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k \times k }$ , we unfold it to $Z \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times ( C _ { \mathrm { i n } } k ^ { 2 } ) }$ and use

$$
r _ { \mathrm { c o n v , s p e c } } = \sqrt { \frac { C _ { \mathrm { o u t } } } { C _ { \mathrm { i n } } k ^ { 2 } } } , \qquad \tau _ { \mathrm { c o n v } } = \eta \sqrt { \frac { C _ { \mathrm { o u t } } } { C _ { \mathrm { i n } } k ^ { 2 } } } .
$$

For a linear head $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ with $\ell _ { 2 }$ geometry, we use $r _ { \mathrm { l i n e a r , } \ell _ { 2 } } = 1 / \sqrt { d _ { \mathrm { o u t } } d _ { \mathrm { i n } } } ;$ ; for a onedimensional normalization weight of dimension $d ,$ we use $r _ { \mathrm { n o r m } , \ell _ { 2 } } = 1 / \sqrt { d }$

## E.5 Additional curves and projector ablations

Figure 3 expands the main comparison by showing the five tuned baseline curves together with the four NBCLT-FL variants. The degree-d projector ablation for the hardest split $\alpha _ { \mathrm { d i r } } = 0 . 1$ is shown in the main text as Figure 2; it uses the same certified safe-min scale rule described in Appendix C.4.

![](images/333c8d371902af09bc108c6b120b8346ce72c7b263478ad2212c0019f18b0f84.jpg)  
Figure 3: Full CIFAR-10 federated-learning comparison under Dirichlet label-skew partitions. Each panel shows all five tuned baselines–FedAvg, SCAFFOLD-M, FedAMS, FedMuon, and PAdaMFed– together with the four NBCLT-FL variants. The main text reports the compressed comparison in which these five baselines are represented by the strongest tuned baseline curve.

## F Limitations, Compute, and Societal Considerations

Scale and comparison to Muon. The NBCLT update rule and the PolyGram-RCClip projector are not specific to federated aggregation and could in principle be instantiated as a centralized optimizer for large-scale neural-network training. This makes the method conceptually close in scope to Muon-type optimizers, while difering in the mathematical primitive: NBCLT clips tentative updates by projection, whereas Muon-type methods select LMO/orthogonalized directions. Our FL experiments include FedMuon as a federated Muon-type baseline and show favorable performance in that setting, but they do not replace a direct large-scale centralized comparison with Muon on language-model pretraining or other large neural-network benchmarks. Such comparisons are an important future direction.

Empirical scope. The experiments evaluate CIFAR-10 federated learning with Dirichlet labelskew partitions and a CIFAR-style ResNet-18. The best run for each method is selected by average test accuracy over the final 100 communication rounds. Appendix E provides the training setup, hyperparameter grids, and implementation details.

Choosing geometry at scale. NBCLT allows diferent update geometries for diferent parameter blocks, but the best norm choice for very large models is not yet understood. In particular, it remains open whether clipping-based flexible geometry can outperform LMO-based Muon geometry on large-scale language-model training, and which layerwise or blockwise norm choices are most efective in that regime. The current theory treats the chosen norm as fixed and analyzes the resulting local-training dynamics; automatically selecting or adapting the geometry is beyond the scope of this work.

Projector certification. The PolyGram-RCClip certificate requires a safe scale $s \geq \| Z \| _ { 2 } .$ a feasible polynomial with a certified fitting error, and an accumulated residual budget. The experiments use the deterministic safe-min scale in Appendix C.4, but the fixed-degree table implementation is a practical approximation; we do not certify its accumulated residual for arbitrary horizons. Heuristic scales without an upper-bound guarantee, table discretization, and floating-point efects also require additional verification.

Compute and assets. The CIFAR-10 experiments were run on a workstation with four NVIDIA RTX 4090 GPUs. We use the public CIFAR-10 dataset and standard open-source deep-learning libraries; no new dataset or human-subject data are introduced. NBCLT is a general optimization method for distributed training. It may reduce communication and make distributed training more eficient, but it does not by itself provide privacy, security, fairness, or robustness guarantees. In particular, federated learning should not be interpreted as formal privacy protection unless combined with mechanisms such as diferential privacy or secure aggregation.