# Resilient Cell-Free MIMO ISAC Under Uncertainties: An Impairment-Aware Graph Neural Network Framework

Oways Alsoloh, Robert-Jeron Reifert, Aymen Khaleel, Stefan Roth and Aydin Sezgin Institute of Digital Communication Systems, Ruhr University Bochum, Bochum, Germany Email: {alsoloh.oways, robert-.reifert, aymen.khaleel, stefan.roth, aydin.sezgin}@rub.de

Abstract—Beamforming designs for integrated sensing and communication (ISAC) in cell-free multiple-input multipleoutput (MIMO) systems face the challenge to stay reliable under uncertainties from both channel state information (CSI) as well as clutter. To this end, this paper proposes an impairment-aware heterogeneous graph neural network (GNN) framework for beamforming that obtains the CSI error level and the clutter covariance as edge features, which enables a single trained model to adapt to both impairments. In particular, the GNN is trained on a low quantile of the minimum user rate with a clutter-aware signal-to-clutter-plus-noise ratio, balanced by a normalized Tchebycheff scalarization. Simulation results show that the proposed GNN framework outperforms the considered baselines in sensing-only and communication-only settings. Depending on the operating point, a single GNN model attains approximately 97% of each baseline’s rate while serving both functionalities at once. The model further generalizes to unseen CSI error levels and clutter counts and renders the ISAC system resilient.

Index Terms—Cell-free, MIMO, integrated sensing and communication, GNN, imperfect CSI, clutter awareness, resilience.

## I. INTRODUCTION

Integrated sensing and communication (ISAC) describes the simultaneous utilization of the same radio hardware for communication and sensing, and is considered an important technology for future 6G systems [1], [2]. Cell-free multipleinput multiple-output (MIMO) is an attractive architecture for ISAC, since its distributed access points (APs) provide macro diversity to the user equipments (UEs) and observe the target from several positions [3]. Still, this diversity alone does not guarantee reliable communication, since the beamformers are computed based on imperfect channel state information (CSI). In particular, beamformer designs that treat the channel estimates as exact leave residual interference that grows with the estimation error. Further, a beamformer designed for good average rates can lead to outages, in which weak users fall far below their target rate in a non-negligible fraction of channel realizations. This outcome motivates optimizing a low quantile of the minimum user rate instead [4]. For sensing, similar problems arise. Here, clutter reflectors create undesired echoes that compete with the sensing target (ST)

echo. Since clutter arrives from directions that the receiving AP cannot fully separate from the target direction, part of the clutter power cannot be suppressed, which reduces the signal-to-clutter-plus-noise ratio (SCNR). This makes sensing performance dependent not only on target illumination, but also on how the beamformers interact with the spatial clutter structure [5]. Thus, a robust ISAC design must account for both CSI uncertainty in communication and clutter in sensing. Moreover, sensing and communication affect each other, since they share the same transmit resources [6]. As no beamformer maximizes both functionalities at once, balancing these priorities needs to be addressed. A common way to select a suitable operating point is to maximize a weighted sum of the robust communication and sensing rates. However, the two rates differ in scale, which makes the effect of a given weight hard to control. The normalized Tchebycheff formulation adopted in this work instead measures each rate relative to its own reference value [7]. Yet, the resulting problem is non-convex and would have to be solved anew for every channel and clutter realization, which motivates a learning-based mapping from the available system information to the beamformers [8].

Graph neural networks (GNNs) are well suited for this mapping, since a cell-free ISAC system can be represented directly as a graph. The APs, UEs and ST form the nodes, while the AP-UE and AP-ST links form the edges. This representation offers two advantages. First, a GNN shares its weights across all nodes of the same type, which makes the beamformer independent of the ordering of APs and UEs and keeps the number of trainable parameters independent of the number of APs and UEs. Second, the impairments considered here belong to the individual links, namely the CSI error to the AP-UE links and the clutter to the AP-ST links. Therefore, they can be passed to the GNN directly as edge features [9].

Existing cell-free ISAC designs address CSI uncertainty, clutter, and the communication-sensing trade-off only partially. A cell-free ISAC beamforming problem is solved in [3] by iterative convex methods under perfect CSI and a clutter-free target. In [9], [10], cell-free ISAC beamforming is learned by heterogeneous GNNs that exploit the AP-UE-target topology to maximize an average-rate objective with a sensing signalto-noise ratio (SNR) term. Outage-oriented robustness to CSI error is addressed in [4] through rate-quantile maximization with uncertainty sampling, but limited to a single-transmitter multiple-input single-output downlink without sensing. Clutter and imperfect channel estimation are modeled jointly in [5], where power allocation is optimized over fixed beam directions under an ergodic-rate criterion. None of the prior designs learns full ISAC beamformers under clutter and imperfect CSI with an outage-oriented criterion, nor examines resilience to uncertainties, which is the regime targeted in this paper.

The main contribution of this work is a heterogeneous GNN that receives a per-AP clutter covariance and the CSI-error level as edge features and maps them, together with the channel estimates and target steering vectors, directly to the beamformers. The GNN is trained on an outage-oriented quantile of the minimum user rate and a clutter-aware SCNR, combined by a normalized Tchebycheff scalarization whose weight traces the communication-sensing trade-off. Beyond robustness to the impairments seen during training, the impairment-aware inputs make a single trained model resilient at CSI-error levels and clutter counts outside the training range. Here, resilience refers to maintaining the system functionality under unforeseen impairments, of which robustness is one aspect [11].

Notation: Bold uppercase and lowercase letters denote matrices and column vectors, respectively. $( \cdot ) ^ { T }$ and $( \cdot ) ^ { H }$ denote the transpose and Hermitian transpose, ∥ · ∥ the Euclidean norm, | · | the absolute value, and ℜ(·) and $\Im ( \cdot )$ the real and imaginary parts. ${ \mathbf { I } } _ { N }$ is the $N \times N$ identity matrix, $\delta _ { u u ^ { \prime } }$ the Kronecker delta, and vec(·) stacks the columns of a matrix into a vector. E[·] and $\operatorname* { P r } ( \cdot )$ denote expectation and probability, CN (0, Σ) the zero-mean circularly symmetric complex Gaussian distribution with covariance $\Sigma ,$ and Unif[a, b) the uniform distribution on [a, b). Finally, $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ denotes the positive part.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

We consider a downlink cell-free ISAC system with M distributed APs and K single-antenna communication users, each served by a single data stream. The AP set is denoted as M. Each AP is equipped with a transmit and a receive uniform linear array (ULA), each consisting of N antennas. All APs are coordinated by a central processing unit and simultaneously serve the users while illuminating the target embedded in a cluttered scene, using the same time and frequency resources. The full setup is illustrated in Fig. 1.

## A. Signal Model

The signal $\mathbf { x } _ { m } \in \mathbb { C } ^ { N \times 1 }$ transmitted from the antenna array of the m-th AP combines the data symbol $x _ { k }$ intended for user $k \in \mathcal { K }$ and dedicated sensing signal $x _ { s }$ for $s \in S .$ , where K and S are the sets of communication and sensing streams, respectively, with $\kappa \cap { \mathcal { S } } = \emptyset$ . For compactness, the overall set of both streams is denoted as $\mathcal { U } = \mathcal { K } \cup \mathcal { S }$ . The transmitted signal from the m-th AP is expressed as [3]

$$
\mathbf { x } _ { m } = \sum _ { k \in \mathcal { K } } \mathbf { w } _ { m k } x _ { k } + \sum _ { s \in \mathcal { S } } \mathbf { w } _ { m s } x _ { s } = \sum _ { u \in \mathcal { U } } \mathbf { w } _ { m u } x _ { u } ,\tag{1}
$$

with $\mathbf { w } _ { m u } \in \mathbb { C } ^ { N \times 1 }$ the beamforming vector of the u-th stream at AP m. The streams are mutually uncorrelated with unit power, $\mathbb { E } [ x _ { u } x _ { u ^ { \prime } } ^ { * } ] = \delta _ { u u ^ { \prime } }$ , and each AP satisfies a total transmit

![](images/af0d1bd996be8ccac21b5e067142aeee3b658abded9c3a608d9e8fa8352b4109.jpg)  
Fig. 1. Cell-free MIMO ISAC scenario, with distributed APs serving the UEs and sensing a target in the presence of clutter.

power constraint given by the sum of the squared norms of its beamformers,

$$
\mathbb { E } \left[ \| \mathbf { x } _ { m } \| ^ { 2 } \right] = \sum _ { u \in \mathcal { U } } \| \mathbf { w } _ { m u } \| ^ { 2 } \leq P _ { m } .\tag{2}
$$

Since all APs transmit the same symbol $x _ { u }$ , the per-AP beamformers of stream u can be stacked into a single vector $\begin{array} { r l } { { \bf w } _ { u } = } & { { } \left[ { \bf w } _ { 1 u } ^ { T } , \ldots , { \bf w } _ { M u } ^ { T } \right] ^ { T } \in \mathbb { C } ^ { M N \times 1 } } \end{array}$ , which is denoted by $\mathbf { w } _ { k }$ for a communication stream k and by ${ \bf w } _ { s }$ for a sensing stream s.

## B. Communication Model

The communication channel between AP m and user k is denoted as $\mathbf { h } _ { m k } \in \mathbb { C } ^ { N \times 1 }$ . By stacking the channel vectors associated with all APs, we obtain the aggregated channel vector $\mathbf { h } _ { k } \in \mathbb { C } ^ { M N \times 1 }$ , which characterizes the communication channel between user k and the entire set of APs. Hence, the true channel is written as the sum of the channel estimate $\tilde { \mathbf { h } } _ { k }$ available at the APs and an independent error vector $\mathbf { e } _ { k }$ as [4]

$$
\begin{array} { r } { \mathbf { h } _ { k } = \tilde { \mathbf { h } } _ { k } + \mathbf { e } _ { k } , \qquad \mathbf { e } _ { k } \sim \mathcal { C N } ( \mathbf { 0 } , \sigma _ { e } ^ { 2 } \mathbf { I } _ { M N } ) , } \end{array}\tag{3}
$$

where $\begin{array} { r l r } { \tilde { \mathbf { h } } _ { k } } & { { } \sim } & { \mathcal { C } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { M N } ) } \end{array}$ is modeled as uncorrelated Rayleigh fading channel. Under the block-fading assumption, the received signal at the k-th UE is

$$
y _ { k } = \sum _ { m \in \mathcal { M } } \mathbf { h } _ { m k } ^ { H } \mathbf { x } _ { m } + n _ { k } ,\tag{4}
$$

with $n _ { k } ~ \sim ~ \mathcal { C N } ( 0 , \sigma _ { k } ^ { 2 } )$ being the additive white Gaussian noise with power $\sigma _ { k } ^ { 2 } .$ The received signal separates into the desired stream, the multi-user interference, the sensing interference, and the noise. Based on this decomposition, the signal-to-interference-plus-noise ratio (SINR) of the communication signal received at the k-th UE follows as [3]

$$
\Gamma _ { k } ^ { ( \mathrm { c } ) } = \frac { \vert \mathbf { h } _ { k } ^ { H } \mathbf { w } _ { k } \vert ^ { 2 } } { \sum _ { k ^ { \prime } \in \mathcal { K } \backslash \{ k \} } \vert \mathbf { h } _ { k } ^ { H } \mathbf { w } _ { k ^ { \prime } } \vert ^ { 2 } + \sum _ { s \in \mathcal { S } } \vert \mathbf { h } _ { k } ^ { H } \mathbf { w } _ { s } \vert ^ { 2 } + \sigma _ { k } ^ { 2 } } .\tag{5}
$$

Subsequently, the corresponding user rate is

$$
R _ { k } ^ { \mathrm { ( c ) } } = \log _ { 2 } ( 1 + \Gamma _ { k } ^ { \mathrm { ( c ) } } ) .\tag{6}
$$

## C. Sensing Model

A distributed monostatic sensing model is adopted, in which each AP processes only its own echo. This avoids the tight inter-AP synchronization required by multistatic processing, limits the fronthaul load, and makes the clutter seen by each AP a local quantity that can be attached to its sensing edge in Section III. Each AP operates in full-duplex mode with colocated transmit and receive arrays, and the transmit-to-receive self-interference is assumed to be cancelled prior to sensing processing, which is feasible through combined analog and digital cancellation [12]. The signal transmitted by AP m returns after a single bounce from a point-like target and $C$ undesired clutter reflectors [5]. The target gain $\alpha _ { m } \sim \mathcal { C N } ( 0 , \zeta _ { m } ^ { 2 } )$ captures the path loss and the radar cross section (RCS), whereas $\alpha _ { m , c } \sim \mathcal { C N } ( 0 , \sigma _ { \alpha } ^ { 2 } )$ denotes the reflection coefficient of the c-th clutter source. Let $\mathbf { a } ( \phi )$ denote the array response vector with unit-modulus entries $\| \mathbf { a } ( \phi ) \| ^ { 2 } = N$ $\phi _ { m }$ denotes the angle of the target, and $\phi _ { m , c }$ the angle of the c-th clutter reflector seen by AP m. The sensing channel is given by

$$
\mathbf { G } _ { m } = \alpha _ { m } \mathbf { a } ( \phi _ { m } ) \mathbf { a } ^ { H } ( \phi _ { m } ) + \sum _ { c = 1 } ^ { C } \alpha _ { m , c } \mathbf { a } ( \phi _ { m , c } ) \mathbf { a } ^ { H } ( \phi _ { m , c } ) .\tag{7}
$$

Under the Swerling-I model, the reflection gains are constant over the coherent processing interval and vary independently across intervals. With this model, the echo received at AP m is

$$
{ \bf y } _ { m } = { \bf G } _ { m } { \bf x } _ { m } + { \bf n } _ { m } ,\tag{8}
$$

where $\mathbf { n } _ { m } \sim \mathcal { C N } ( \mathbf { 0 } , \varsigma _ { m } ^ { 2 } \mathbf { I } _ { N } )$ is the radar receiver noise with the power $\varsigma _ { m } ^ { 2 }$ . Each AP m applies the normalized target-matched combiner $\mathbf { a } ^ { H } ( \phi _ { m } ) / N$ to its own echo. Since the target gains are independent across $\mathrm { A P s } ,$ , the squared magnitudes of the combiner outputs are aggregated across the APs [3]. Clutter directions are generally not orthogonal to the target direction, and part of the clutter therefore leaks through the matched combiner. The normalized target-clutter correlation at AP m is expressed as $\begin{array} { r } { \kappa _ { m , c } \ = \ \frac { { \bf a } ^ { H } ( \breve { \phi } _ { m } ) { \bf a } ( \phi _ { m , c } ) } { N } } \end{array}$ , where the absolute value of $\kappa _ { m , c }$ is bounded by zero and one. For a given clutter realization, we define the effective clutter covariance matrix at $\mathbf { A P }$ m as

$$
\mathbf { Q } _ { \mathrm { c l } , m } = \sum _ { c = 1 } ^ { C } | \alpha _ { m , c } | ^ { 2 } | \kappa _ { m , c } | ^ { 2 } \mathbf { a } ( \phi _ { m , c } ) \mathbf { a } ^ { H } ( \phi _ { m , c } ) .\tag{9}
$$

Subsequently, the clutter power leakage through the targetmatched combiner is given by

$$
I _ { \mathrm { c l } } = \sum _ { m \in \mathcal { M } } \sum _ { u \in \mathcal { U } } \mathbf { w } _ { m u } ^ { H } \mathbf { Q } _ { \mathrm { c l } , m } \mathbf { w } _ { m u } .\tag{10}
$$

The echo power is averaged over the Swerling-I target gain, while the clutter coefficients and angles $\{ \phi _ { m , c } , \alpha _ { m , c } \}$ are fixed. Consequently, we define the aggregate SCNR as the ratio of the total target echo power to the total clutter and noise power to assess sensing performance using observations from all APs

$$
\Gamma ^ { ( \mathrm { s } ) } = \frac { \sum _ { m \in \mathcal { M } } \zeta _ { m } ^ { 2 } \sum _ { u \in \mathcal { U } } \left| \mathbf { a } ^ { H } ( \phi _ { m } ) \mathbf { w } _ { m u } \right| ^ { 2 } } { I _ { \mathrm { c l } } + \bar { \varsigma } ^ { 2 } } ,\tag{11}
$$

with the normalized noise power $\textstyle { \bar { \varsigma } } ^ { 2 } = \sum _ { m \in { \mathcal { M } } } \varsigma _ { m } ^ { 2 } / N$ . The sensing performance is then expressed as

$$
R ^ { ( \mathrm { s } ) } = \log _ { 2 } \Bigl ( 1 + \Gamma ^ { ( \mathrm { s } ) } \Bigr ) .\tag{12}
$$

For a Gaussian target gain observed at a single AP, maximizing this rate is equivalent to minimizing the mean squared error (MSE) of the gain estimate [13].

## D. Problem Formulation

The minimum user rate is denoted by $R _ { \mathrm { m i n } } ^ { \mathrm { ( c ) } } = \operatorname* { m i n } _ { k \in \mathcal { K } } R _ { k } ^ { \mathrm { ( c ) } }$ and $r ^ { \mathrm { ( c ) } }$ denotes the reliable communication rate level associated with the outage probability $\rho .$ . The sensing objective is the sensing rate $R ^ { \mathrm { ( s ) } }$ . The minimum over the users ensures fairness, which a single target does not require. The two objectives generally conflict, since the communication and sensing streams share the same transmit power budget and the sensing streams contribute interference to the communication users as mentioned in (5). Therefore, a multi-objective optimization problem (MOOP) is formulated as

$$
\begin{array} { r l } { \underset { \{ \mathbf { w } _ { m u } \} } { \operatorname* { m a x } } } & { { } \left( r ^ { ( \mathrm { c } ) } , ~ R ^ { ( \mathrm { s } ) } \right) } \end{array}\tag{13}
$$

$$
\mathrm { s . t . } \quad \operatorname* { P r } \Bigl ( R _ { \mathrm { m i n } } ^ { ( \mathrm { c } ) } \le r ^ { ( \mathrm { c } ) } \Bigr ) \le \rho ,\tag{13a}
$$

$$
\sum _ { u \in \mathcal { U } } \| \mathbf { w } _ { m u } \| ^ { 2 } \leq P _ { m } , \qquad \forall m \in \mathcal { M } .\tag{13b}
$$

Constraint (13a) means that the minimum user rate falls below $r ^ { \mathrm { ( c ) } }$ with a probability no larger than $\rho .$ Constraint (13b) is the per-AP transmit power budget. The MOOP is non-convex due to the coupled SINR and SCNR expressions, while the outage constraint has no closed-form expression under imperfect CSI. The non-convexity and the missing closed form of the outage constraint prevent a direct solution with standard convex solvers.

## III. PROPOSED ROBUST LEARNING METHOD

To address the MOOP, we propose a learning framework illustrated in Fig. 2. First, we introduce a CSI error- and clutter-aware GNN that maps the available information to the beamformers. Subsequently, the training objective is presented.

## A. CSI Error- and Clutter-Aware GNN

The beamformer is parameterized by a heterogeneous GNN whose graph follows the AP-UE-ST structure of [9], with the AP-UE edges carrying communication channel features and the AP-ST edges carrying sensing and clutter features. The GNN represents each node and edge by an embedding, which is a numerical vector that summarizes its associated information. Let $\mathbf { a p } _ { m } ^ { ( \ell ) }$ $\mathbf { u e } _ { k } ^ { ( \ell ) }$ , and $\mathbf { s t } ^ { ( \ell ) }$ denote the node embeddings at layer $\ell ,$ and $\mathbf { c } \mathbf { e } _ { m k } ^ { ( \ell ) } , \mathbf { s e } _ { m } ^ { ( \ell ) }$ the communication and sensing edge embeddings. The input maps $\mathrm { E m b _ { c } }$ and $\mathrm { E m b _ { s } }$ are linear embedding layers, whereas the edge-update, nodeupdate and readout maps are multi-layer perceptrons (MLPs) with LeakyReLU activations. Each communication edge is initialized from the channel estimate together with the error level that characterizes its accuracy,

$$
\mathbf { c e } _ { m k } ^ { ( 0 ) } = \operatorname { E m b } _ { \mathrm { c } } \big ( \big [ \Re \big ( \tilde { \mathbf { h } } _ { m k } \big ) ^ { T } , \Im \big ( \tilde { \mathbf { h } } _ { m k } \big ) ^ { T } , \sigma _ { e } \big ] ^ { T } \big ) ,\tag{14}
$$

![](images/343d414c5a78f9ddd5a62a68dad4cd1e1cb978d022ca166aa8bef75035bc2819.jpg)  
Fig. 2. Proposed impairment-aware GNN framework, which maps the clutteraware and CSI error-aware edge features to the beamformers $\{ \bar { \bf w } _ { m u } \}$ , trained with the Tchebycheff loss of (18) and (19).

which allows a single trained model to be operated across CSI-error levels. With the target and clutter blocks

$$
\begin{array} { r l } & { \pmb { \eta } _ { m } ^ { \mathrm { t g } } = \big [ \Re \big ( \mathbf { a } ( \phi _ { m } ) \big ) ^ { T } , \Im \big ( \mathbf { a } ( \phi _ { m } ) \big ) ^ { T } \big ] ^ { T } , } \\ & { \pmb { \eta } _ { m } ^ { \mathrm { c l } } = \big [ \Re \big ( \mathrm { v e c } ( \mathbf { Q } _ { \mathrm { c l } , m } ) \big ) ^ { T } , \Im \big ( \mathrm { v e c } ( \mathbf { Q } _ { \mathrm { c l } , m } ) \big ) ^ { T } \big ] ^ { T } , } \end{array}
$$

the sensing edge associated with AP m is initialized as

$$
\mathbf { s e } _ { m } ^ { ( 0 ) } = \mathrm { E m b } _ { \mathrm { s } } \big ( \big [ \pmb { \eta } _ { m } ^ { \mathrm { t g } ^ { T } } , \pmb { \eta } _ { m } ^ { \mathrm { c l } ^ { T } } \big ] ^ { T } \big ) .\tag{15}
$$

Thus, the sensing branch receives the clutter covariance that determines the leakage in (10). At message-passing layer $\ell ,$ the edge embeddings are updated as

$$
\mathbf { c e } _ { m k } ^ { ( \ell ) } = f _ { \mathrm { c } } ^ { ( \ell - 1 ) } \left( \mathbf { c e } _ { m k } ^ { ( \ell - 1 ) } , \mathbf { c e } _ { m k } ^ { ( 0 ) } , \mathbf { a p } _ { m } ^ { ( \ell - 1 ) } , \mathbf { u e } _ { k } ^ { ( \ell - 1 ) } \right) ,\tag{16}
$$

$$
\mathbf { s e } _ { m } ^ { ( \ell ) } = f _ { \mathrm { s } } ^ { ( \ell - 1 ) } \left( \mathbf { s e } _ { m } ^ { ( \ell - 1 ) } , \mathbf { s e } _ { m } ^ { ( 0 ) } , \mathbf { a p } _ { m } ^ { ( \ell - 1 ) } , \mathbf { s t } ^ { ( \ell - 1 ) } \right) ,\tag{17}
$$

where the initial edge feature is reinjected at every layer, keeping the impairment state available throughout the network. Fig. 2 depicts one layer, which is repeated L times. The updated edge embeddings are averaged at each node, and each node type updates its embedding with its own MLP. After L layers, readout MLPs map the final edge embeddings to unconstrained beamformers, which a projection step scales to satisfy (2). This yields the beamformers $\left\{ \mathbf { w } _ { m u } \right\}$ in Fig. 2, comprising $\mathbf { w } _ { m k }$ and $\mathbf { w } _ { m s }$ for the communication and sensing beamformers, respectively.

## B. Normalized Tchebycheff Loss

The MOOP is scalarized by a Tchebycheff function [14], which is defined with respect to a reference point that expresses the desired operating level of each objective. The GNN is trained in an unsupervised manner. Both communication and sensing metrics are estimated empirically over each minibatch of size B. To account for imperfect CSI, the reliable communication rate is estimated through data augmentation, following [4], [15]. For each batch sample $b ,$ we generate J channel realizations for every user k by adding independent error vectors to the channel estimate $\ddot { \mathbf { h } } _ { k , b }$

$$
\mathbf { h } _ { k , b } ^ { ( j ) } = \tilde { \mathbf { h } } _ { k , b } + \mathbf { e } _ { k , b } ^ { ( j ) } , \qquad j = 1 , \ldots , J .
$$

The minimum user rate $R _ { \mathrm { m i n } , b , j } ^ { \mathrm { ( c ) } }$ is computed for each realization $j ,$ , and the lower-tail value $r _ { b } ^ { \mathrm { ( c ) } }$ is taken as the empirical

TABLE I MAIN SIMULATION AND TRAINING PARAMETERS.
<table><tr><td>System parameters</td><td></td><td>GNN and optimization parameters</td><td></td></tr><tr><td>APs</td><td> $M = 3$ </td><td>Layers</td><td> $L = 4$ </td></tr><tr><td>Antennas per ULA</td><td> $N = 4$ </td><td>Hidden size</td><td>256</td></tr><tr><td>Users</td><td> $K = 1 2$ </td><td>Mini-batch size</td><td> $B = 1 0 0$ </td></tr><tr><td>Sensing streams</td><td> $S = | \boldsymbol { S } | = 1$ </td><td>Learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Total Tx power</td><td> $P _ { \mathrm { { t o t } } } \stackrel { \cdot } { = } 6$ </td><td>Training samples</td><td>20000</td></tr><tr><td>Per-AP power</td><td> $P _ { \underline { { m } } } = 2$ </td><td>Val. samples</td><td>2000</td></tr><tr><td>Noise power</td><td> $\sigma _ { k } ^ { 2 } = 0 . 1$ </td><td>Test samples</td><td>1000</td></tr><tr><td>Sensing noise</td><td> $\bar { \varsigma } ^ { 2 } = 0 . 1$ </td><td>Trade-off weight</td><td> $\mu \in \{ 0 , 0 . 2 5 , \ldots , 1 \}$ </td></tr><tr><td>Per-AP Tx SNR</td><td> $1 3 . 0 \ \mathrm { d B }$ </td><td>Comm. reference</td><td> $z _ { \mathrm { c } } ^ { \star } = \dot { 0 } . 6$  65</td></tr><tr><td>Target gain</td><td> $\zeta _ { m } ^ { 2 } = 1$ </td><td>Comm. lower</td><td> $z _ { c } ^ { \mathrm { { m i n } } } = 0$ </td></tr><tr><td>Nominal CSI error</td><td> $\sigma _ { e } = 0 . 4$ </td><td>Sensing reference</td><td> $z _ { \mathrm { s } } ^ { \mathrm { ~ \tiny ~ * ~ } } = 6 . 0$ </td></tr><tr><td>Nominal clutter</td><td> $C = 2$ </td><td>Sensing lower</td><td> $z _ { \mathrm { s } } ^ { \mathrm { \tilde { m i n } } } = 0 . 9$ </td></tr><tr><td>Rate quantile</td><td> $\rho = 0 . 0 5$ </td><td>Training clutter</td><td> $\breve { C } \in \{ 0 , 1 , 2 , 3 , 5 \}$ </td></tr><tr><td>Samples/estimate</td><td> $\stackrel { \cdot } { J } = 1 2 8$ </td><td>Training CSI range</td><td> $\sigma _ { e } \sim \mathrm { \tilde { U } n i f { [ 0 . 2 , 0 . 6 ] } }$ </td></tr></table>

ρ-quantile of the resulting samples. Averaging over the minibatch gives

$$
\hat { r } ^ { ( \mathrm { c } ) } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } r _ { b } ^ { ( \mathrm { c } ) } .\tag{18}
$$

The beamformers are designed from the available estimates alone, whereas the rates entering (18) are evaluated on the sampled channel realizations $\mathbf { h } _ { k , b } ^ { ( j ) } ,$ which emulate the unknown true channels. The GNN therefore learns beamformers that remain reliable under the estimation error. The sensing value is the mini-batch average

$$
\hat { r } ^ { ( \mathrm { s } ) } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } R _ { b } ^ { ( \mathrm { s } ) } ,\tag{19}
$$

with $R _ { b } ^ { \mathrm { ( s ) } }$ evaluated for the corresponding clutter realization. For the reference point, let $z _ { \mathrm { c } } ^ { \star }$ and $z _ { \mathrm { s } } ^ { \star }$ denote the communication and sensing reference values, and let $z _ { \mathrm { c } } ^ { \mathrm { m i n } }$ and $z _ { \mathrm { s } } ^ { \mathrm { m i n } }$ denote the lower normalization values that render two deficits dimensionless. The reference point is not required to be attainable. Both reference values are fixed hyperparameters and the values used are reported in Section IV. For a trade-off parameter $\mu \in [ 0 , 1 ]$ , the normalized deficits are

$$
d _ { \mathrm { c } } = \mu \left[ \frac { z _ { \mathrm { c } } ^ { \star } - \hat { r } ^ { ( \mathrm { c } ) } } { z _ { \mathrm { c } } ^ { \star } - z _ { \mathrm { c } } ^ { \mathrm { m i n } } } \right] _ { + } , d _ { \mathrm { s } } = ( 1 - \mu ) \left[ \frac { z _ { \mathrm { s } } ^ { \star } - \hat { r } ^ { ( \mathrm { s } ) } } { z _ { \mathrm { s } } ^ { \star } - z _ { \mathrm { s } } ^ { \mathrm { m i n } } } \right] _ { + } .\tag{20}
$$

A metric that already exceeds its reference thus contributes no deficit and the loss is governed by the remaining metric. The GNN is trained to minimize the loss function

$$
\mathcal { L } = \operatorname* { m a x } \{ d _ { \mathrm { c } } , d _ { \mathrm { s } } \} .\tag{21}
$$

Finally, the GNN parameters are updated with the gradient $\nabla { \mathcal { L } } .$ , which is back-propagated through the network, as illustrated in Fig. 2.

## IV. SIMULATION RESULTS

This section evaluates the proposed GNN framework under imperfect CSI and clutter-limited sensing. Unless stated otherwise, the nominal setup uses $C = 2$ clutter reflectors per AP and $\sigma _ { e } = 0 . 4 ,$ , with the full parameter set listed in Table I. The robust communication rate denotes the empirical 0.05- quantile of the minimum user rate. All compared methods are evaluated on the same paired channel, CSI error, and clutter realizations. The target and clutter angles are drawn independently and uniformly per AP, $\phi _ { m } , \phi _ { m , c } \sim \mathrm { U n i f } [ 0 , \pi / 2 )$ Channels and clutter scenes are redrawn at every training step, and no realization is seen twice. Each epoch therefore covers 20000 independent clutter scenes, with the reflector count randomized per sample over $C \in \{ 0 , 1 , 2 , 3 , 5 \}$ , while the test set uses 1000 further independent scenes that are identical across all compared methods. Training uses early stopping on a fixed validation set of 2000 samples.

![](images/ae51b4964679c8615f3e94b05ed3cb823fdbb82095a17f8670c6a86936fcd65a.jpg)  
Fig. 3. Communication–sensing operating plane.

Baselines. Two analytical baselines are considered, each dedicated to one functionality. The first baseline for communication utilizes a regularized zero-forcing (RZF) structure in [16] computed from the stacked vector of channel estimates as

$$
\mathbf { w } _ { k } ^ { \mathrm { ( R Z F ) } } = \left( \operatorname* { m i n } _ { m \in \mathcal { M } } \sqrt { \frac { P _ { m } } { \sum _ { j \in \mathcal { K } } \| \bar { \mathbf { w } } _ { m j } \| ^ { 2 } } } \right) \bar { \mathbf { w } } _ { k } ,\tag{22}
$$

$$
\bar { \mathbf { w } } _ { k } = \left( \sum _ { j \in \mathcal { K } } \tilde { \mathbf { h } } _ { j } \tilde { \mathbf { h } } _ { j } ^ { H } + \lambda \mathbf { I } _ { M N } \right) ^ { - 1 } \tilde { \mathbf { h } } _ { k } ,\tag{23}
$$

with $\lambda = K \left( \sigma _ { k } ^ { 2 } + \sigma _ { e } ^ { 2 } P _ { \mathrm { t o t } } \right)$ . The second baseline, dedicated to sensing, is a clutter-aware minimum variance distortionless response (MVDR) beamformer as follows [17]

$$
\mathbf { w } _ { m } ^ { ( \mathrm { M V D R } ) } = \sqrt { P _ { m } } \frac { ( \mathbf { Q } _ { \mathrm { c l } , m } + \epsilon \mathbf { I } _ { N } ) ^ { - 1 } \mathbf { a } ( \phi _ { m } ) } { \left\| ( \mathbf { Q } _ { \mathrm { c l } , m } + \epsilon \mathbf { I } _ { N } ) ^ { - 1 } \mathbf { a } ( \phi _ { m } ) \right\| } ,\tag{24}
$$

where $\epsilon = 0 . 1$ . For the resilience studies in Fig. 4, we additionally include a non-robust GNN trained with perfect CSI $\sigma _ { e } = 0$ and without clutter $C = 0$ utilizing the same architecture and a scalarization weight of $\mu = 0 . 7 5$ . This model is evaluated under the same test conditions as the proposed GNN.

## A. Communication-Sensing Trade-off

The benefit of jointly optimizing robust communication and sensing performance is illustrated in Fig. 3. When only optimizing for sensing by setting $\mu = 0$ , a mean sensing rate of 6.036 bit/s/Hz is reached, exceeding the clutter-aware MVDR baseline by 43.9%. The MVDR beamformer maximizes the ratio between target gain and clutter leakage at each AP separately, whereas the joint SCNR in (11) combines the contributions of all APs in a single ratio. The GNN optimizes this joint ratio directly and therefore attains higher sensing rates. When only optimizing communication by setting $\mu = 1$ a robust communication rate of 0.659 bit/s/Hz is reached, which exceeds RZF by 27.0% at a comparable sensing rate of 1.414 bit/s/Hz. RZF is computed from the channel estimates without accounting for the CSI error, whereas the GNN is trained on the outage quantile. In both cases, the remaining sensing rate stems from the incidental target illumination of the communication beams. The intermediate models reveal the trade-off between communication reliability and sensing performance. $\mathrm { A t } \mu = 0 . 7 5$ , the proposed method retains 97.4% of the RZF robust communication rate while achieving 96.7% of the MVDR sensing rate. This point is therefore selected for the resilience studies below. The dash-dotted line corresponds to time sharing between RZF and MVDR, where each baseline is active for a fraction of the time. All learned operating points lie above this line, which shows that the GNN reaches tradeoffs that switching between the two baselines cannot achieve.

## B. Resilience to CSI Uncertainty and Clutter

This subsection examines whether the operating point selected in Section IV-A remains reliable when the impairments deviate from the nominal setup, including CSI-error levels and clutter counts not seen during training. The proposed GNN with $\mu = 0 . 7 5$ is evaluated at $C = 2$ over different CSI-error levels in Fig. 4(a). At perfect CSI, RZF achieves the highest robust communication rate, followed by the non-robust GNN. However, the proposed GNN outperforms the non-robust GNN in both communication and sensing at every tested positive variance between 0.1 and 0.4. At $\sigma _ { e } ^ { 2 } = 0 . 2$ , the proposed GNN achieves approximately 28% higher robust rate than the non-robust model, while providing 3.725 versus 1.403 bit/s/Hz sensing rate. The robust model also surpasses RZF at all tested variances from 0.2 onward. For sensing, the proposed GNN exceeds the clutter-aware MVDR baseline at $\sigma _ { e } ^ { 2 } = 0$ and 0.1, and retains approximately 88.8% of the MVDR sensing rate at $\sigma _ { e } ^ { 2 } = 0 . 2$ , while simultaneously serving communication users. Training covers standard deviations $\sigma _ { e } \in [ 0 . 2 , 0 . 6 ]$ , corresponding to variances $\sigma _ { e } ^ { 2 } \in [ 0 . 0 4 , 0 . 3 6 ]$ . Thus, the evaluation includes error levels outside the training range. At the unseen variance $\sigma _ { e } ^ { 2 } = 0 . 4$ which is 2.5 times the nominal variance, the proposed GNN achieves 0.447 bit/s/Hz robust communication rate, exceeding RZF and the non-robust GNN by 50.4% and 58.2%, respectively. Since the impairment levels are inputs to the GNN, it adapts its beamformers to this unforeseen condition, which corresponds to the adaptation capability of resilience. Nevertheless, its sensing rate decreases as the CSI error grows, because the communication deficit increasingly dominates the Tchebycheff loss and shifts resources toward communication.

Fig. 4(b) studies the effect of the number of clutter reflectors at $\sigma _ { e } = 0 . 4$ . For $C = 1 , \ldots , 5 ,$ , the proposed GNN provides approximately 23% to 26% higher robust communication rates than the non-robust GNN. Additionally, the proposed model outperforms the non-robust one in sensing throughout the sweep. At $C \ = \ 4 ,$ , a clutter count never presented during training, the proposed model achieves 2.079 bit/s/Hz sensing rate, approximately 86.1% of the MVDR rate, while still providing 0.506 bit/s/Hz robust communication rate. The cluttercovariance input therefore allows the GNN to adapt its beamformers to clutter densities of varying size, including an unseen one, while the Tchebycheff objective prevents the solution from collapsing into a pure sensing design. This result demonstrates resilience to an unseen intermediate clutter count, while maintaining both communication and sensing functionality.

![](images/3acfd877d0d71c2b53f9fc7104a0f95bd7a54b9997e0293e582cd3f7b37d97ac.jpg)

![](images/2d49be47b245fbe989e0513dd83092e1022d8fb4f19d48573cc2e1d0b76c90a6.jpg)  
Fig. 4. Robust communication and sensing performance (a) under CSI uncertainty and (b) versus the number of clutter reflectors.

## V. CONCLUSION

This work proposes an impairment-aware heterogeneous GNN for cell-free ISAC beamforming under imperfect CSI and sensing clutter. A single jointly learned operating point approaches the respective rates of the two baselines, whereas training under ideal conditions provides weaker performance under the tested impairments. The evaluation at an unseen clutter count and unseen CSI error variances provides evidence of resilience. However, the sensing rate of the proposed GNN decreases as the CSI uncertainty increases, although its robust communication rate remains above that of RZF and the nonrobust GNN at large CSI error levels. This motivates future designs with an explicit sensing rate constraint alongside the communication reliability objective.

## REFERENCES

[1] F. Liu, Y. Cui, C. Masouros, J. Xu, T. X. Han, Y. C. Eldar, and S. Buzzi, “Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond,” IEEE J. Sel. Areas Commun., vol. 40, no. 6, pp. 1728–1767, 2022.

[2] X. Luo, Q. Lin, R. Zhang, H.-H. Chen, X. Wang, and M. Huang, “ISAC—A survey on its layered architecture, technologies, standardizations, prototypes, and testbeds,” IEEE Commun. Surveys Tuts., vol. 28, pp. 485–526, 2026.

[3] U. Demirhan and A. Alkhateeb, “Cell-free ISAC MIMO systems: Joint sensing and communication beamforming,” IEEE Trans. Commun., vol. 73, no. 6, pp. 4454–4468, 2025.

[4] K. Liang, G. Zheng, Z. Li, K.-K. Wong, and C.-B. Chae, “A data and model-driven deep learning approach to robust downlink beamforming optimization,” IEEE J. Sel. Areas Commun., vol. 42, no. 11, pp. 3278– 3292, 2024.

[5] J. K. Dassanayake and G. Amarasuriya, “Near-field cell-free massive MIMO-aided ISAC,” in Proc. IEEE GLOBECOM, 2025, pp. 2874–2879.

[6] A. Umra, A. M. I. Ahmed, and A. Sezgin, “RL-aided cognitive ISAC: Robust detection and sensing-communication trade-offs,” EURASIP J. Wireless Commun. Netw., 2026.

[7] C. Meng, Z. Wei, D. Ma, W. Ni, L. Su, and Z. Feng, “Multiobjectiveoptimization-based transmit beamforming for multitarget and multiuser MIMO-ISAC systems,” IEEE Internet Things J., vol. 11, no. 18, pp. 29 260–29 274, 2024.

[8] R.-J. Reifert, H. Dahrouj, and A. Sezgin, “Robust energy efficiency for joint communication and computation in ML-aided cell-free networks,” in Proc. Asilomar Conf. Signals, Syst., Comput., 2025.

[9] U. Demirhan and A. Alkhateeb, “Learning beamforming in cell-free massive MIMO ISAC systems,” in Proc. IEEE SPAWC, 2024, pp. 326– 330.

[10] Z. Wang and V. W. S. Wong, “Heterogeneous graph neural network for cooperative ISAC beamforming in cell-free MIMO systems,” in Proc. ACM MobiCom Workshop ISACom, 2024.

[11] R.-J. Reifert, S. Roth, A. A. Ahmad, and A. Sezgin, “Comeback kid: Resilience for mixed-critical wireless network resource management,” IEEE Trans. Veh. Technol., vol. 72, no. 12, pp. 16 177–16 194, 2023.

[12] C. Bayesteh Barneto, T. Riihonen, M. Turunen, L. Anttila, M. Fleischer, K. Stadius, J. Ryynanen, and M. Valkama, “Full-duplex OFDM radar¨ with LTE and 5G NR waveforms: Challenges, solutions, and measurements,” IEEE Trans. Microw. Theory and Techn., vol. 67, no. 10, pp. 4042–4054, 2019.

[13] Y. Yang and R. S. Blum, “MIMO radar waveform design based on mutual information and minimum mean-square error estimation,” IEEE Trans. Aerosp. Electron. Syst., vol. 43, no. 1, pp. 330–343, 2007.

[14] K. Miettinen, Nonlinear multiobjective optimization. Kluwer Academic Publishers Boston, 1999, vol. 12.

[15] R.-J. Reifert, H. Dahrouj, A. Alameer Ahmad, H. Gacanin, and A. Sezgin, “Robust communication and computation using deep learning via joint uncertainty injection,” in Proc. ISWCS, 2024.

[16] C. B. Peel, B. M. Hochwald, and A. L. Swindlehurst, “A vector-perturbation technique for near-capacity multiantenna multiuser communication–Part I: channel inversion and regularization,” IEEE Trans. Commun., vol. 53, no. 1, pp. 195–202, 2005.

[17] J. Capon, “High-resolution frequency-wavenumber spectrum analysis,” Proc. IEEE, vol. 57, no. 8, pp. 1408–1418, 1969.