# Learning by Examples: Power Grid Steady State Analysis with a Tabular Foundation Model

Olayiwola Arowolo<sup>∗</sup>, Oluwadara Adedeji<sup>‡</sup>, Peter Palensky<sup>∗</sup>, Jochen L. Cremer<sup>∗†</sup>

<sup>∗</sup>Delft University of Technology, Delft, the Netherlands

<sup>†</sup>Austrian Institute of Technology, Vienna, Austria

<sup>‡</sup>University College Dublin, Belfield Dublin, Ireland

o.a.arowolo, j.l.cremer, p.palensky{@tudelft.nl}, oluwadara.adedeji{@ucdconnect.ie}

Abstract—Foundation models have shown strong generalization across scientific domains, but their use in power systems remains largely task-specific. We investigate whether a pretrained tabular foundation model (TFM), TabPFN, can serve as a general-purpose neural solver for power grid steady-state analysis. We evaluate the TFM on three tasks: power flow (PF) prediction, N-k contingency screening (NCS), and voltage stability margin (VSM) prediction. We performed experiments on 118- and 2000-bus systems, comparing the TFM against taskspecific neural baselines. Without retraining or fine-tuning, the TFM matches or outperforms the best task-specific model in several settings across the three tasks. For 118-bus PF prediction, a variant of the TFM achieves errors of 1 × 10<sup>−3</sup> pu for voltage magnitudes and 0.30<sup>◦</sup> for voltage angles, outperforming all baselines. On unseen N-1 contingency and close-to-infeasible PF cases, the TFM is competitive with the DCPF approximation. For millions of unseen N-2 contingencies, the TFM achieves AUCs of 0.92 and 0.95 for branch and voltage violation screening, respectively. The TFM also attains accurate VSM prediction performance for the 2000-bus system with limited labelled data. The TFM attains strong results across the tested tasks despite pretraining without any power-system-specific dataset and requiring no fine-tuning step. These results offer a glimpse into the potential of pretrained TFMs as a neural solver for a range of power grid analysis tasks.

Index Terms—Tabular Foundation Model, Power Flow, Nk Contingency Screening, Voltage Stability Margin Prediction, Neural Solver

## I. INTRODUCTION

Driven by massive integration of renewable energy and rapid electrification of mobility and industry, the global energy transition introduces significant variability and uncertainty into grid operations [1]. Therefore, grid operators must rapidly assess grid security and reliability while maintaining adequate safety margins in the constantly changing operational landscape [2]. Power flow (PF) is the core task in many steady-state security analyses, including contingency screening, voltage stability analysis, and topology optimization [3]. Classical AC PF solvers provide accurate solutions but become computationally expensive at large scales. The computational cost becomes especially burdensome for applications which require repeated PF solver calls. [4]. For example, exhaustive N-3 branch contingency screening on the modest IEEE-118 bus system requires over 1 million PF solver calls per operating point (OP). Operators typically rely on DC power flow approximation (DCPF) for such applications, but DCPF has large errors for high loading conditions [5]. These errors lead to conservative operation of the power system and increases the risks of blackouts. Therefore, conventional computational tools for planning and operating the grid are increasingly inadequate for real-time monitoring and analysis across multiple scenarios.

Machine learning (ML) models have been proposed as capable neural surrogates for many power system analyses. Despite the promising speed advantages offered by ML, its adoption in practical power system operation remains limited by several factors. These include human barriers such as lack of trust in black-box models [6] and technical barriers such as scarcity of large, high-quality datasets [7]. Limited generalization is also an important technical barrier that current ML approaches face in power systems. We define generalization as the ability of an ML model to give accurate predictions for OPs outside its training data [8], where the OP includes the loading, injection, and topology of the grid. Since it is practically impossible to cover all possible topology and parameter combinations in the training data of large grids, many ML models have to be retrained when an unseen topology/parameter configuration is encountered during operation [2]. Transfer learning in power systems [9] can suffer from negative transfer and catastrophic forgetting [10]. Separate ML models may also be trained and interfaced for individual components in the grids [11]. However, these approaches are not scalable.

Graph Neural Networks (GNNs) have been explored as a way to improve the generalization of ML models to topological changes in power grids because of their inherent generalization properties [12]. GNNs have demonstrated promising generalization to N-1 topology changes on tasks such as PF [13] and ACOPF [14]. However, GNNs face several architectural limitations such as limited expressivity, oversmoothing, and oversquashing [15]. These limitations can cause poor performance of GNN models in applications with little or no locality dependence compared to topology-agnostic models [16]. While different modifications have been proposed to overcome these limitations in isolation, developing a model that combines topology flexibility and sufficient accuracy across a range of power system tasks remains challenging. Existing GNN approaches also remain largely task- and gridspecific. Different models are typically trained for tasks which have strong dependence on each other, for example, PF and N-k contingency screening, which highlights the limited generalization capability of existing GNN approaches.

A foundation model is a large-scale transformer-based model which is pretrained on large, diverse datasets through self-supervision and learns transferable knowledge that generalizes to new, unseen tasks [4]. FMs have recently emerged as a path towards better generalization of ML models in power systems and have already demonstrated success in domains such as language, vision, and audio. Application-specific FMs have been proposed in power systems. [17] proposed time series FMs for electricity price data, [18], [19] investigated integrating existing LLMs into power system workflows, while [20] proposed a foundation model for power system dynamic simulations. While these approaches extend the generalization of ML to specific applications through custom-made FMs or existing large language models (LLMs), they still fall short of a general-purpose FM model which can be applied to a wide range of power system tasks. [4] discussed the idea of a multitask grid foundation model (GridFM), while [21] proposed a GNN architecture towards a GridFM. While promising, these approaches apply to a limited set of tasks covered in the pretraining.

In this work, we investigate tabular foundation model (TFM) based on prior-fitted networks [22] as a promising generalpurpose GridFM. TFMs are pretrained transformer models designed for making predictions on tabular datasets. These models have demonstrated impressive data efficiency and accuracy across a wide range of tasks [23]. Unlike classic ML, TFM works by in-context learning (ICL). That is, the labelled training data serves as examples/contexts for the model during inference. This is similar to how large language models infer responses from context provided in the problem. For a power grid, this means the same model can be used to predict for a new task, distributions of OP or grid topologies, given a relatively small number of labelled samples, all without model retraining. We previously demonstrated the use of TFM for power grid dynamic security assessment in [24].

In this paper, we propose an approach using TFM as a general-purpose neural solver for power grid steady-state analyses. The proposed approach formulates PF and voltage stability margin (VSM) prediction as tabular regression tasks and N-k contingency screening as a tabular classification task. Our main contributions are as follows:

• We propose a TFM approach for power system steadystate analyses. We demonstrate the proposed approach on PF, VSM prediction and N-k contingency screening.

• We present simple methods to extract topological features for the proposed TFM approach. The presented methods improve accuracy and scalability to larger grids.

• We demonstrate the accuracy and robustness of our proposed approach through comprehensive case studies on the 118- and 2000-bus grids.

The remainder of the paper is structured as follows: Section 2 formulates the relevant problems, Section 3 presents our proposed methodology, Section 4 presents the case studies, Section 5 discusses the results, and Section 6 concludes the paper.

## II. PROBLEM FORMULATION

This section provides an overview of the three steady-state analysis tasks: AC PF and its DC approximation, $N - k$ contingency screening, and VSM/loadability margin prediction.

## A. AC power flow

AC PF determines the flow of electrical power in a steadystate power grid, and it is mathematically formulated as a set of non-linear equations. Let G be a power grid graph with $\mathcal { N }$ buses and E branches (transmission lines and transformers). The branch status vector $s \in \{ 0 , 1 \} ^ { \varepsilon }$ represents the set of active branches, and $Y ( s ) = G ( s ) + j B ( s )$ is the admittance matrix which represents the grid topology. For each bus $i \in \mathcal { N } .$ the following four variables are defined: voltage magnitude $V _ { i } ,$ voltage angle $\theta _ { i } ,$ active power injection $P _ { i } ,$ , and reactive power injection $Q _ { i } ,$ where $P _ { i } = P ^ { \mathrm { g e n } } { - } P ^ { \mathrm { l o a d } }$ and $Q _ { i } = Q ^ { \mathrm { g e n } } { - } Q ^ { \mathrm { l o a d } }$ The AC PF equations are given by:

$$
P _ { i } = V _ { i } \sum _ { k \in \mathcal { N } } V _ { k } \left( G _ { i k } \cos \theta _ { i k } + B _ { i k } \sin \theta _ { i k } \right) ,\tag{1}
$$

$$
Q _ { i } = V _ { i } \sum _ { k \in \mathcal { N } } V _ { k } \left( G _ { i k } \sin \theta _ { i k } - B _ { i k } \cos \theta _ { i k } \right) ,\tag{2}
$$

where $\theta _ { i k } = \theta _ { i } - \theta _ { k }$ . In $\mathrm { P F , }$ each bus has two known variables and two unknown variables, depending on the bus type, as shown in Table I. The above leads to a set of 2N equations with 2N unknowns. Classic AC PF solvers use the iterative Newton–Raphson (NR) approach to calculate the unknown voltage variables at PQ and PV buses, while the other variables are analytically derived [3].

## B. DC Power Flow Approximation

AC PF in Eqs. 1-2 has no analytical solution and must be evaluated through NR iterations. For transmission systems, AC PF is often simplified with the following assumptions: i) conductances $G _ { i k }$ are zero; ii) phase angle differences $\theta _ { i k }$ are small; iii) voltages are constant and set to 1.0 pu. Eqs. 1-2 reduce to:

$$
P _ { i } = \sum _ { k \in \mathcal { N } } B _ { i k } \theta _ { i k } .\tag{3}
$$

DCPF given in Eq. 3 is linear and can be easily solved to obtain the unknown voltage angles at each bus in the grid.

## C. AC Power Flow Prediction

Solving the AC PF equations can be formulated as a regression learning task. The task objective is to predict the unknown

TABLE I  
KNOWN AND UNKNOWN VARIABLES PER BUS TYPE.
<table><tr><td rowspan="2">Bus type</td><td colspan="2">Variable</td></tr><tr><td>Known</td><td>Unknown</td></tr><tr><td>PQ bus</td><td>Pi, Qi</td><td>Vi, θi</td></tr><tr><td>PV bus</td><td> $\dot { P _ { i } } , \dot { V _ { i } }$ </td><td> $Q _ { i } , \theta _ { i }$ </td></tr><tr><td>Slack bus</td><td> $V _ { i } , \theta _ { i }$ </td><td> $P _ { i } , Q _ { i }$ </td></tr></table>

2N variables. The implicit function in $\operatorname { E q . }$ . 4 describes the true AC PF solution:

$$
F ( x , u , s ) = 0 ,\tag{4}
$$

where $x \in \mathcal { X } \subseteq \mathbb { R } ^ { 2 \mathcal { N } }$ are the known variables, $u \in \mathcal { U } \subseteq \mathbb { R } ^ { 2 N }$ are the unknown variables, and $s \in \{ 0 , 1 \} ^ { \varepsilon }$ are the active branches. A machine learning model aims to learn an approximate solution

$$
{ \tilde { F } } ( x , s ) \to u ,\tag{5}
$$

where AC PF problem instances are provided in the training data.

## D. N − k Contingency Screening

Let c be a set of contingencies or simultaneous outages of interest, and let $S _ { i }$ be the vector of nodal apparent power injections, $S _ { i } = \sqrt { ( P _ { i } ) ^ { 2 } + ( Q _ { i } ) ^ { 2 } }$ . The system operation seeks to know if $S _ { i }$ is feasible for all possible contingencies $c , \forall 1 \leq$ $c \leq k . \operatorname { A s }$ previously mentioned, c can be very large for $k > 1$ even for a moderately sized grid. Therefore, operators typically rely on $N - 1$ contingencies and a few heuristically defined higher-order cases [25] to assess steady-state feasibility. From Eqs. 1-2, the active and reactive power flowing on a branch is given by:

$$
P _ { i j } = | V _ { i } | | V _ { j } | \left( G _ { i j } \cos \theta _ { i j } + B _ { i j } \sin \theta _ { i j } \right) - G _ { i j } | V _ { i } | ^ { 2 } ,\tag{6}
$$

$$
\begin{array} { r } { Q _ { i j } = | V _ { i } | | V _ { j } | \left( G _ { i j } \sin \theta _ { i j } - B _ { i j } \cos \theta _ { i j } \right) + B _ { i j } | V _ { i } | ^ { 2 } , } \end{array}\tag{7}
$$

$\forall ( i , j ) \in \mathcal { E }$ . Feasibility is defined by two security criteria. The first is the branch thermal limit defined by

$$
S _ { i j } < d \cdot R _ { i j } \quad \forall ( i , j ) \in \mathcal { E } ,\tag{8}
$$

where $R _ { i j }$ is the maximum thermal rating of the branch, and d is some allowance factor. The second is the voltage limit:

$$
V _ { i } ^ { \operatorname* { m i n } } \leq V _ { i } \leq V _ { i } ^ { \operatorname* { m a x } } \quad \forall i \in \mathcal { N } .\tag{9}
$$

The prediction task is to determine the feasibility of a set $\{ x , s ^ { c } \}$ , where $s ^ { c }$ is the post-contingency topology. The prediction task can be formulated in two ways. The first is to learn a classifier $G \ : \ ( x , s ^ { c } ) \ \to \ { \hat { y } }$ , which directly maps the known variables into the contingency label $y \in \{ 0 , 1 \}$ The second approach uses the learnt AC PF solution map ${ \tilde { F } } : ( x , s ^ { c } ) \to u$ . With all PF variables obtained from the predictions, contingency screening is done by applying the feasibility thresholds in Eqs. 8-9 to the predicted variables.

## E. Voltage Stability Margin Prediction

Voltage stability margin (VSM) represents how close the grid is to voltage collapse and quantifies the maximum additional load the grid can admit before instability occurs. Numerical solvers such as NR are prone to divergence at or close to the point of instability [3], which is also known as the ’nose’ of the PV curve. The nose point is conventionally derived numerically through continuation power flow (CPF). CPF is done by scaling all specified injections along a direction while holding voltage setpoints fixed:

$$
\begin{array} { r } { P _ { i } = \tau P _ { 0 } , } \end{array}\tag{10}
$$

$$
Q _ { i } = \tau Q _ { 0 } .\tag{11}
$$

Here we assume proportional scaling of power injections from the base case $P _ { 0 } , Q _ { 0 }$ . For each value of τ, NR is used to trace out the PV curve. At the maximum loading $\tau _ { \mathrm { m a x } }$ , det $J = 0$ for the PF Jacobian (singular Jacobian). The loadability factor σ at every point along $1 \leq \tau \leq \tau _ { \operatorname* { m a x } }$ is given by

$$
\sigma = \frac { \tau _ { \mathrm { m a x } } } { \tau } ,\tag{12}
$$

and decreases monotonically from the base injection to the nose point where $\sigma = 1 . \ \sigma - 1$ is the loadability margin and is the fraction by which every injection in the grid can be scaled before voltage instability is reached. Determining VSM is computationally expensive because CPF involves repeated AC PF solutions. DCPF also cannot accurately determine VSM because the assumptions of DCPF are violated at high loading (stressed grid condition). Hence, we formulate the VSM prediction task as a regression task:

$$
H ( x , s ) \to \sigma\tag{13}
$$

## III. METHODOLOGY

## A. Tabular Foundation Model

A tabular foundation model is a single transformer model pretrained on millions of synthetic datasets sampled from a tabular data-generating prior [22]. The pretraining step is done once during the model development. Afterwards, the pretrained transformer becomes a generic learning algorithm that can be used to predict arbitrary unseen datasets without fine-tuning or retraining [26]. During inference, the TFM carries out in-context learning, where the model receives an unseen dataset with both labelled training samples and unlabelled test samples; the model then simply processes the new input-output pairs in a single forward pass without any weight update. Given a table of labelled examples,

$$
\begin{array} { r } { \mathcal { C } = \{ ( z _ { j } , y _ { j } ) \} _ { j = 1 } ^ { N } , } \end{array}\tag{14}
$$

which serve as the model context, and unlabelled rows $z _ { * }$ which are the model query, the TFM predicts labels for all query rows during the forward pass, as shown in Fig. 1.

ICL by TFM is fundamentally different from standard supervised learning. In supervised learning, the model weight is fitted to each dataset via gradient update:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \sum _ { j = 1 } ^ { N } \mathcal { L } \big ( f _ { \theta } ( z _ { j } ) , y _ { j } \big ) , \qquad \hat { y } _ { * } = f _ { \theta ^ { * } } ( z _ { * } ) ,\tag{15}
$$

This means a new task, grid topology or range of OPs require retraining by solving Eq. 15 again. By contrast, TFM is pretrained on datasets drawn from a prior over data-generating process:

$$
\phi ^ { * } = \arg \operatorname* { m i n } _ { \phi } \mathbb { E } _ { \phi \sim \psi \sim p ( \psi ) } \big [ - \log q _ { \phi } ( y _ { * } \mid z _ { * } , C ) \big ] ,\tag{16}
$$

After pre-training, the TFM performs ICL and predicts labels for any new tabular dataset directly at inference [23]. Adapting the TFM to a new task, grid topology or range of OPs in power systems only requires changing C, no gradient update is required. As described in [27], the TFM approximates the posterior predictive distribution at inference:

![](images/497eeb393d638f0601b754dd68457aada950be8f7d2a082c9c05e289c7487364.jpg)  
Fig. 1. The whole table of labelled context rows and unlabelled query rows is passed to the pretrained TFM in one step. The TFM returns labels for all query rows in a single forward pass.

$$
p ( y _ { * } \mid z _ { * } , C ) = \int p ( y _ { * } \mid z _ { * } , \psi ) p ( \psi \mid C ) d \psi ,\tag{17}
$$

where $\psi$ is the latent parameter governing the data generation process of the prior. The prediction step is therefore an approximate Bayesian inference conditioned on the available training data. It is important to stress that no real data is used during pretraining, but the synthetic data comes from a prior of randomly sampled structural causal models (SCM) [26]. We have used TabPFNv3 as the TFM in this paper [28]. The TabPFN model uses both row and column attention; therefore, predictions are independent of row or column order. The model also supports classification and regression tasks and returns a predictive distribution for each label rather than a point prediction.

## B. Predicting power flow residuals

For TabPFN, each PF example is one row in the tabular data, with the 2N known variables as feature columns, while unknown variables are predicted per column. A table of size $n _ { \mathrm { s a m p l e s } } \times 2 \mathcal { N }$ is used to predict each unknown variable in turn. For example, V<sub>i</sub> at a given load bus is predicted for all samples in one inference step. Rather than predicting the voltage state directly, we can predict its residual or the deviation from a linear approximation of the state. DCPF is a fast linear approximation of the AC PF equations: under the DC assumptions, the voltage magnitude at every bus is 1 pu (the flat profile) and $\theta = B ^ { \prime - 1 } p ,$ where $B ^ { \prime }$ is the DC susceptance matrix Laplacian with the slack row removed as defined in Eq.3. The difference between the DC approximation and the true AC solution captures the nonlinear effects of the PF problem. We define the residual of the DC solution as:

$$
\Delta V = V _ { \mathrm { A C } } - 1 , \qquad \Delta \theta = \theta _ { \mathrm { A C } } - B ^ { \prime - 1 } p ,\tag{18}
$$

and reformulate the voltage-state prediction tasks as residual learning. The unknown injections are reconstructed from the voltage states.

## C. Scaling TFM to larger grids with power decoupling

The inference cost of TFMs scales with the size of the context table. For example, the TabPFN model we use in this paper is pretrained on tables with at most 2000 feature columns. Larger tables may be processed during inference if they fit into the GPU memory, but this may come at the cost of degraded accuracy. In PF, the number of features in the context table grows as 2N . For the 2000-bus system, we have 4000 feature columns. The original TabPFN model can be used to process all 4000 features. Alternatively, the decoupled power-flow assumption can be applied for feature selection. In transmission networks, voltage angles are strongly coupled with the active-power injections, while voltage magnitudes are strongly coupled to the reactive-power flow. This separates the power-flow equations into two independent half-systems,

$$
\Delta \theta = B ^ { \prime - 1 } \Delta P , \qquad \Delta V = B ^ { \prime \prime - 1 } \Delta Q ,\tag{19}
$$

where $B ^ { \prime \prime }$ is the reduced susceptance matrix of the PQ buses. Under the assumption that the P–V and Q–θ cross-couplings are weak, TabPFN can approximate θ using only the known $P$ as feature columns, and approximates V using only $Q$ as feature columns.

## D. Encoding grid topology in tabular data

A TFM does not automatically encode a notion of the grid topology. Unlike GNN models, where the grid topology is provided directly to the model as inductive bias, the grid topology can only be specified to a TFM through the data provided as context. The topology may be encoded using onehot vectors specifying connected and disconnected branches, but this leads to a large increase in the number of features for large grids and also generalizes poorly [29]. Here, we propose encoding the grid topology in the TFM using the well-known line outage distribution factors (LODF).

For a grid with $\mathcal { N }$ buses and E branches, let $A \in \mathbb { R } ^ { \mathcal { E } \times ( N - 1 ) }$ be the reduced branch incidence matrix with rows $a _ { e } ^ { \top }$ , and let $B _ { D } = \mathrm { d i a g } ( 1 / x _ { e } )$ be the diagonal branch-susceptance matrix, so that $B ^ { \prime } = A ^ { \top } B _ { D } A$ is the Laplacian of the DC susceptance matrix. The power transfer distribution factor (PTDF) matrix is:

$$
\mathrm { P T D F } = B _ { D } A B ^ { \prime - 1 } ,\tag{20}
$$

whose entry $\mathrm { P T D F } ( e , i )$ gives the share of a unit active-power injection at bus i that flows on line e. The $\mathrm { L O D F } ( e , h )$ is the fraction of the flow of line h that is rerouted onto line e when line h is disconnected, and is related to the PTDF through the line-to-line sensitivities Ψ = PTDF $A ^ { \top }$

$$
\mathrm { L O D F } ( e , h ) = \frac { \Psi _ { e h } } { 1 - \Psi _ { h h } } .\tag{21}
$$

We propose a topology feature that combines the LODF with the pre-outage active power flows,

$$
\Delta \theta ( s , p ) = \big ( B ^ { \prime } ( s ) ^ { - 1 } - B ^ { \prime - 1 } \big ) p ,\tag{22}
$$

one column per non-slack bus. For an $N - k$ contingency, the N−k LODF matrix is obtained by a Woodbury matrix update [30]: with U the incidence columns of the outaged branches and $B _ { D } ^ { s }$ the diagonal of $B _ { D }$ on the outaged set s,

$$
\begin{array} { r } { B ^ { \prime } ( s ) = B ^ { \prime } - U B _ { D } ^ { s } U ^ { \top } , } \end{array}\tag{23}
$$

$$
\begin{array} { c } { { B ^ { \prime } ( s ) ^ { - 1 } = B ^ { \prime - 1 } + B ^ { \prime - 1 } U \big ( ( B _ { D } ^ { s } ) ^ { - 1 } - U ^ { \top } B ^ { \prime - 1 } U \big ) ^ { - 1 } } } \\ { { \times U ^ { \top } B ^ { \prime - 1 } . } } \end{array}\tag{24}
$$

The inverse is fast to compute for low-rank changes to $B ^ { \prime }$ , such as $N - 1$ up to $N - 3$ contingencies. The proposed topology feature $\Delta \theta$ adds power-grid inductive bias to the TFM.

## IV. CASE STUDIES

## A. Dataset

We use the PF∆ [3] PF benchmark dataset in our experiments. The dataset consists of solved AC PF instances across various grids. The dataset also includes separate subsets to evaluate generalization to N-1 and N-2 component outages as well as generalization to close-to-infeasible (C2I) instances. The C2I instances are obtained by using MATPOWER CPF. We specifically use the C2I full topology data subset for the VSM prediction task. The CPF starts from a base OP, and all loads and injections are proportionally scaled up to 2.5× the base values. The 118-bus grid has 3600 training samples and 200 testing samples, while the 2000-bus grid has 600 training samples and 92 testing samples. PF∆ provides the base OP and nose OP for each training/testing sample. The samples have a ’family’ index to connect each base OP to its nose OP. Furthermore, we generate 8 OPs for each family according to the following setting: $\tau = 1 + f ( \tau _ { m a x } - 1 ) \forall f \in$ $\{ 0 . 3 , 0 . 5 , 0 . 7 , 0 . 8 , 0 . 9 , 0 . 9 5 , 0 . 9 7 , 0 . 9 9 \}$ . We use the NR solver at each τ by warm-starting from the previous $\tau$ in the OP family. We call these samples augmented OPs. We label every augmented OP with its proportional loadability margin $\sigma = \tau _ { m a x } / \tau$

In addition to the original PF∆ dataset, we generate additional data for the contingency screening task using the GridFM datakit tool. Because of the use of a custom, relaxed ACOPF formulation in the PF∆ dataset, all the PF instances violate the steady-state operating limits [31]. On the 118-bus and 2000-bus grids we consider in this work, all samples were outside the steady-state voltage or branch power operating limits. This makes the dataset alone unsuitable for predicting steady-state violations. The GridFM datakit dataset, however, has a more realistic balance between samples within and outside normal operating limits. We generated 1,558,730 N-1 scenarios by disconnecting each branch across 8,760 load scenarios. About 64% of the N-1 scenarios are secure. We similarly generated 3,133,892 N-2 scenarios by disconnecting each combination of two branches across 200 load scenarios. About 41% of the N-2 scenarios are secure. Scenarios which lead to grid islanding are discarded.

## B. Experimental setting

The case studies for the PF and the VSM prediction tasks are on the 118-bus and 2000-bus grids. Only the 118-bus grid was used for the contingency screening task because the problem becomes intractably large for the 2000-bus system. We use a wide range of baseline models in the experiments. We have kept the same baseline models across the 3 tasks for consistency in our analysis. The baseline models are:

1) Graph Neural Solver (GNS) [32]: This is a selfsupervised GNN approach which predicts nodal voltage angles and magnitudes by minimizing the power imbalance in the network. As such, the model does not require ground truth PF solutions. Generator setpoints are analytically derived from the predicted voltages. GNS uses the following hyperparameters: $k \ = \ 1 0 .$ hidden dimension $= 6 4 , \gamma = 0 . 0 1$

2) Typed-GNN (TGN) [33]: This is also a self-supervised GNN approach where nodal voltages and angles are determined by minimizing power imbalances while deriving generator injections from predicted voltages. TGN defines branches as node types such that different node types can have different update functions and output dimensions. We use TGN with 15 layers, 3 message passing steps and a hidden dimension of 16 per node.

3) PowerFlowNet [34]: This is a supervised learning GNN approach where all known PF variables are predicted by minimizing the L2 loss between the predictions and the ground truth. We use PowerFlowNet with $k = 3$ and 4 TAGConv layers.

4) Physics-Informed Edge Varying GNN (PI-EVGNN) [35]: An edge varying GNN differs from standard graph convolution by allowing different trainable weights per node and edge for each convolution layer. It uses a semi-supervised GNN approach which combines the L2 loss between the model predictions and ground truth with a physics-informed power balance loss. We use a weighting factor $( \lambda = 0 . 1 )$ for the physics loss for training stability, with the following parameters: $k = 3 .$ number of layers = 2, hidden dimension = 128.

5) LeapNet [36]: This is a supervised learning approach which uses standard MLPs. LeapNet encodes the grid topology through a learnable vector ⋌. On the reference, full-topology dataset, $\ { \cal A } = 0 \ \quad$ . We use LeapNet with 2 layers and a hidden dimension of 128.

The TabPFN model uses the training split of each experiment as its context and the test split as its query. The same context is used for all queries in experiments where inference is done in batches. Model hyperparameters are selected to be consistent with the literature. For the VSM prediction and N-k contingency screening tasks, the nodal outputs of the graph models are aggregated through an MLP readout layer with a hidden dimension of 64 and an output layer of 1. All baseline models were trained with a learning rate of $1 0 ^ { - 4 }$ , a batch size of 256, the AdamW optimizer and the OneCycleLR learning rate scheduler over 200 epochs. All experiments were repeated with 3 random seeds (3, 7, 11), with the average results reported. The seed changes the sampled training/test data. All training and inference experiments were performed on a computer with an AMD EPYC-7302 CPU with 64 cores and an NVIDIA H100 GPU with 128 GB memory.

## C. Power flow analysis

The full topology dataset has 56000 samples for the 118- bus and 28000 samples for the 2000-bus grids. We split the data into 80% for training, 10% for validation and 10% for testing. Fig. 2 compares TabPFN to the baseline approaches for the in-distribution generalization task on the full topology datasets. On the 118-bus grid, TabPFN has the lowest error at predicting voltage magnitudes and reactive power injection of all models, with V RMSE of 0.0011 and Q RMSE of 0.047. TabPFN also has the second-lowest error to LeapNet in predicting slack bus active power injection P and voltage angles θ. TabPFN attains lower errors than all 4 GNN-based baselines across all 4 predicted variables. It is also notable that relative to DCPF, TabPFN’s RMSE is 36× lower for V, 30× lower for P, and 4.4× lower for θ. On the 2000-bus grid, all the models record higher errors. TabPFN maintains the lowest V error, with errors lower than the GNN baselines across all the predicted variables. We also see that the physics-informed GNN models have similar errors to the DCPF method. Further, we assess two TabPFN variants: the residual variant (TabPFN-

![](images/4ab451f4ed0574e3dcce19af6426c9fa730fae25eaf3824fa0e7bafc0f8cd988.jpg)  
Fig. 2. In-distribution accuracy on the full-topology test sets: RMSE of voltage magnitude V , voltage angle θ, active power P and reactive power Q (DCPF does not output Q). Bars show the mean and error bars the standard deviation over three seeds in log scale.

![](images/d32f010997c00ad370cfe1d51fe6b0490538bd0e861a674fbba1be90f5a57ce7.jpg)  
Fig. 3. Generalization to unseen line outages on the 118-bus (top) and 2000-bus (bottom) systems. Zero-shot: outage topologies absent from training (and from TabPFN’s context); mixed: N-1 samples included. Mean and standard deviation over three seeds in log scale; DCPF does not output Q.

R) where the model predicts the DCPF residual of V and θ according to Eq. 16, and the fast-decoupled variants where (V, Q) and (θ, P) are predicted separately according to Eq. 17. TabPFN-R has the lowest error for 3 of the 4 predicted variables on the 118-bus grid and the lowest V and θ errors on the 2000-bus grid. This suggests that residual prediction is an effective way to further improve the accuracy of TabPFN on the task. Also, TabPFN-FD has slightly lower errors for θ and P prediction errors on the 2000-bus system and similar errors to the original model for V and Q. This highlights how TabPFN may be scaled to larger systems, via similar decomposition, without loss of accuracy.

We perform 3 topology generalization experiments as shown in Fig. 3. The first is ‘N-1 zero-shot’ where the models are trained with only the full topology training set and evaluated on the N-1 test set. The second is ‘N-1 mixed’ where the models are trained with a mix of 50% of the full topology training set and the whole N-1 data training set. We evaluate the models on the N-1 test set. The third experiment uses the same training data as in the second experiment, but models are evaluated zero-shot on the N-2 data test set (N-2 zero-shot). On both test grids, the pretrained TabPFN has comparable errors to the trained GNN baselines. The performance of TabPFN is especially strong on the 2000-bus grid, where the model achieves the lowest or second-lowest errors for all 4 predicted variables across the 3 experiments. This is in contrast to the GNN models, which have substantially larger errors on the 2000-bus grid, which points to a structural limitation of message-passing architectures on larger grids. In the zeroshot experiments, TabPFN has comparable θ and P errors to DCPF. However, the use of N-1 samples in its context in the mixed experiment leads to lower errors compared to DCPF. The GNN models also have worse errors in the mixed experiment compared to TabPFN and the MLP-based LeapNet. Overall, zero-shot generalization remains a weakness for the ML models, but TabPFN can adjust its prediction to new topologies via ICL alone, without retraining.

![](images/8f337c11356db1656493dba398a80b4ae4a8722ea8ed44a1bc37e9060f8116c9.jpg)  
Fig. 4. Generalization to close-to-infeasible (C2I) operating points on the 118-bus (top) and 2000-bus (bottom) systems. Zero-shot: C2I samples absent from training (and from TabPFN’s context); mixed: C2I samples included. Mean and standard deviation over three seeds (log scale); DCPF does not output Q.

Next, we investigate generalization to close-to-infeasible (C2I) PF cases with two experiments as shown in Fig. 4. In the ‘C2I zero-shot’ experiment, the ML models are trained with the feasible full topology data train set and evaluated on the C2I data test set. In the ‘C2I mixed’ experiment, the ML models are trained with a mix of 50% of the feasible full topology data train set, the C2I data train set, and the ’approaching infeasibility’ data train set from [3]. The trained models are evaluated on the C2I data test set. On the C2I zeroshot experiment, TabPFN has lower V and θ errors for both test grids, compared to DCPF, which is known to be unsuitable for C2I PF cases. However, TabPFN has a larger P error. Once again, TabPFN has the best performance on the 2000-bus grid, with the lowest V, θ, and Q errors across both zero-shot and mixed experiments. Notably, TabPFN errors are significantly reduced in the mixed experiment compared to the zero-shot experiment, while the GNN models trained with physics loss maintain consistently high error levels in both experiments. The superior accuracy of TabPFN on the large test grid and its adaptivity to samples from different PF scenarios via ICL highlight its potential for PF analysis.

While RMSE and MSE typically measure the prediction errors of ML models in PF analysis, predictions must also be physically consistent and satisfy the power balance (PB). PB residual measures the deviation from the expected condition of zero power imbalance. Fig. 5 shows the mean and maximum PB residual as in [3] for all experiments. We observe that the type of loss function employed during training impacts PB residuals. PowerFlowNet and LeapNet, which are trained on supervised loss only, have comparable residuals to TabPFN on the 118-bus grid and higher residuals on the 2000-bus grid. GNS and TGN, which are trained in a self-supervised fashion specifically to minimize PB residuals, have consistently lower PB errors than TabPFN. While training with PB loss improves the physical consistency of model predictions, the predictions may be physically consistent but still have high errors compared to the ground truth solutions. For example, GNS and TGN have larger RMSE errors than TabPFN in most experiments despite having lower PB residual. Since a low

![](images/ee6cfe364c4531bf582e9100ca532e2df393978025ff8093123158a077481be4.jpg)  
Fig. 5. Mean (top) and maximum (bottom) power-balance residual of the learned models across all test conditions: We report the mean and standard deviation over three seeds (log scale).

PB residual implies the predicted voltages and injections are physically consistent, not that they match the true operating point, PB residuals and predicted variable errors should be reported and assessed together.

## D. N-k contingency screening

In this study, we compare the baseline models trained as classifiers with the TabPFN model. We randomly sample 10000 full topology samples from PF∆ (with violations) and 10000 from the GridFM data, split 80% and 20% for training (context) and validation sets. In addition, 10000 randomly selected N-1 samples are added to the training data. No N-2 samples are seen during training. Each baseline model is trained as a multi-label classifier. For the GNNbased models, the aggregated graph embeddings are passed into an MLP prediction head with 2 output neurons, one for each violation type (bus voltage and branch violations). BCEWithLogitsLoss is used as a loss function to output the probability of branch/voltage violations independently. At inference, each output neuron is independently thresholded with sigmoid > 0.5 indicating a violation. TabPFN classifier only predicts one label, so we use different classifiers for voltage and branch violation screening. We show the performance of the models for the combined violation set and for voltage and branch violations separately in Fig. 6. We make the following observations: First, the topology-augmented TabPFN model (TabPFN+) has strong screening performance. On the held-out N-1 dataset, it had a combined screening average recall of 0.95 and 0.86 on the unseen N-2 dataset. The TabPFN+ model also had better screening accuracy than the base TabPFN model, which lacked topology features. This highlights the importance of informative engineered features for ICL in this application since the model is more constrained in learning only from the provided context samples. TabPFN+ is also better at screening branch violations than voltage violations. We hypothesize that this may be due to the use of linear change in bus voltage angles as topology features, as shown in Eq. 20, since the topology feature augments the model with knowledge of how power flows redistribute after a branch outage.

Second, the ML classifiers were much better than DCPF at screening branch violations. The base TabPFN model achieved an average recall of 0.85 for branch violations compared to DCPF with 0.60 for held-out N-1 contingencies and 0.79 for N-2 contingencies compared to DCPF with 0.62. All the trained classifiers perform better than DCPF at contingency screening except for PI-EVGNN. DCPF is also unable to screen voltage violations since voltages are assumed to be 1 p.u. at every bus. On the other hand, the neural surrogates excel at screening voltage violations with AUC of 0.99 and 0.95 for LeapNet and TabPFN, respectively, on held-out N-1 contingencies and 0.97 and 0.95 on millions of unseen N-2 contingencies.

Third, the base TabPFN model without topology-specific features shows strong generalization to unseen N-2 topologies. Using the threshold-free AUC metric, TFM performs on par with the best trained classifier models (LeapNet and GNS) on N-2 topologies, which are never present in its context. The TabPFN model has an AUC of 0.92 for branch violation screening on N-2 contingencies compared to 0.91 and 0.95 for GNS and LeapNet, respectively. It also has an AUC of 0.95 for voltage violation screening compared to 0.97 and 0.97 for LeapNet and GNS. This shows that TabPFN can separate N-2 scenarios which lead to violations and those scenarios which do not, only through ICL. However, the lower average recall (0.80) suggests that a classification threshold of 0.5 is not optimal.

TABLE II  
![](images/0ea102bfe13473c166277c63c3b7ba711d5d17953e7e8ebc5a4ae8fa64743d85.jpg)  
Fig. 6. N-k contingency classification on the 118-bus grid for N-1 contingencies (top) and unseen N-2 contingencies (bottom): average recall, macro F and AUC for the combined, branch and voltage violation prediction. We report mean and standard deviation over three seeds. DCPF cannot be used to assess voltage violations.

Finally, direct classification for contingency screening is much better than screening with a PF neural surrogate. With a PF surrogate, voltage setpoints are predicted for each contingency sample. The predicted voltage magnitudes can be compared to the voltage limits, while the branch flows can be derived from the predicted voltage angles according to Eqs. 6-7 and compared to branch thermal limits as well. This predictthen-threshold approach is used in [35] and is how the DCPF screening is performed. We compare the DCPF baseline to other model architectures when they are used as PF surrogates in Fig. 7. All the PF neural surrogates had much worse screening accuracy compared to their direct classifiers. Also, only TGN outperforms DCPF at combined branch and voltage violation screening on both datasets. Since contingency screening relies on accurately determining extreme errors in branch angles or voltage magnitudes, the poor screening accuracy of the predict-then-threshold approach of PF neural surrogates can be attributed to the smoothing of extreme errors induced by the MSE loss used for training.

## E. Voltage stability margin prediction

In this study, we predict the loadability factor σ in two experiments. The VSM data is split by the family index so no family of OPs appear in both training and testing subset. Each experiment uses known PF variables as input features and predicts σ directly without PF solutions. The first experiment assesses the models’ in-distribution accuracy and sample efficiency. The in-distribution experiment uses one OP per family, drawn randomly from the base, augmented and nose OPs. The second experiment assesses out-of-distribution generalization of the models. The models are trained with only the base OPs. Base OPs and nose OPs have different distributions. For example, base OPs on the 118-bus grid have $\sigma \in [ 1 . 1 1 , 2 . 3 6 ]$ at a total loading of 59-65p.u., while the nose OPs have $\sigma = 1 . 0$ and a total loading of 70-152 p.u. Both experiments have similar training data sizes. (3600 and 600 independent OPs for 118- and 2000-bus systems respectively.) We evaluate the models on all test samples in both experiments.

PERCENTAGE OF VSM PREDICTIONS WITHIN SPECIFIED TOLERANCE FORIN-DISTRIBUTION GENERALIZATION EXPERIMENT: $| \hat { \sigma } - \sigma | \leq 0 . 0 1$ FORTRUE MARGINS σ − 1 ≤ 0.05 AND $| \hat { \sigma } - \sigma | \leq 0 . 0 5$ FOR $\sigma - 1 > 0 . 0 5 . \mathrm { W E }$ REPORT MEAN ± STANDARD DEVIATION OVER THREE SEEDS; n IS THENUMBER OF TEST SAMPLES. BEST VALUE PER COLUMN IN BOLD.
<table><tr><td rowspan="2">True margin  $\sigma - 1$   $| \hat { \sigma } - \sigma |$ </td><td colspan="2">118-bus</td><td colspan="2">2000-bus</td></tr><tr><td> $\leq 0 . 0 5$   $\leq 0 . 0 1$   $( n = 9 7 8 )$ </td><td> $> 0 . 0 5$   $\leq 0 . 0 5$   $( n = 1 0 2 2 )$ </td><td> $\leq 0 . 0 5$   $\leq 0 . 0 1$   $( n = 7 9 7 )$ </td><td> $> 0 . 0 5$   $\leq 0 . 0 5$   $( n = 1 2 3 )$ </td></tr><tr><td>GNS</td><td> ${ \bar { \bf 5 7 . 7 } } \pm { \bf 1 . 2 }$ </td><td> ${ \bf 9 0 . 4 \pm 1 . 3 }$ </td><td> $6 9 . 6 \pm 0 . 8$ </td><td> $9 4 . 6 \pm 2 . 0$ </td></tr><tr><td>TabPFN</td><td> $3 3 . 6 \pm 0 . 4$ </td><td> $5 7 . 1 \pm 0 . 4$ </td><td> ${ \bf 8 3 . 8 \pm 1 . 0 }$ </td><td> $9 6 . 5 \pm 1 . 2$ </td></tr><tr><td>PI-EVGNN</td><td> $3 0 . 2 \pm 0 . 6$ </td><td> $4 9 . 0 \pm 2 . 3$ </td><td> $6 0 . 1 \pm 4 . 5$ </td><td> $7 6 . 4 \pm 1 1 . 0$ </td></tr><tr><td>LeapNet</td><td> $3 4 . 3 \pm 0 . 3$ </td><td> $4 4 . 0 \pm 1 . 0$ </td><td> $7 7 . 2 \pm 3 . 2$ </td><td> ${ \bf 9 6 . 7 \pm 0 . 8 }$ </td></tr><tr><td>PowerFlowNet</td><td> $2 9 . 0 \pm 5 . 9$ </td><td> $4 5 . 8 \pm 3 . 9$ </td><td> $3 6 . 3 \pm 1 2 . 3$ </td><td> $4 9 . 3 \pm 1 . 2$ </td></tr><tr><td>Typed-GNN</td><td> $1 3 . 8 \pm 3 . 1$ </td><td> $3 0 . 1 \pm 4 . 0$ </td><td> $3 2 . 7 \pm 6 . 0$ </td><td> $5 4 . 7 \pm 1 5 . 3$ </td></tr></table>

Tables II and III compare TabPFN to the baseline models in the in-distribution and out-of-distribution experiments, respectively. We report the share of predictions with absolute error $| \hat { \sigma } - \sigma | \leq 0 . 0 1$ for true margins $\sigma - 1 \leq 0 . 0 5$ and $| \hat { \sigma } - \sigma | \le 0 . 0 5$ for larger margins. The selected low-margin range of $\sigma - 1$ $\leq ~ 0 . 0 5$ is motivated by the WSCC-recommended voltage stability margin in [37]. For the in-distribution experiment shown in Table II, GNS is the most accurate model for the 118-bus grid in the low margin range, with 57.7% of its predictions within the ±0.01 tolerance of the true margin. TabPFN is second to GNS in the high margin range and on par with LeapNet in the low margin range. On the 2000-bus grid, TabPFN is the most accurate in the low margin range, with similar prediction accuracy to LeapNet in the high margin range. For the out-of-distribution experiments shown in Table

Direct classification Thresholded regression TabPFN+ (Direct classification)  
![](images/6c0dee2600e1ef10ff8a2338c942b5c888f37fa0948aae50448d3135e7f4c01f.jpg)  
Fig. 7. Average recall of contingency screening obtained by thresholding the PF regression outputs (hatched) compared with the directly trained classifiers (solid). The dashed line shows the performance of the direct TabPFN+ classifier. We report mean and standard deviation over three seeds.

TABLE III  
PERCENTAGE OF VSM PREDICTIONS WITHIN SPECIFIED TOLERANCE FOR OUT-OF-DISTRIBUTION GENERALIZATION EXPERIMENT: |σˆ − $\sigma | \le 0 . 0 1$ FOR TRUE MARGINS $\sigma - 1 \leq 0 . 0 5$ AND $| \hat { \sigma } - \sigma | \leq 0 . 0 5$ FOR $\sigma - 1 > 0 . 0 5$ WE REPORT MEAN ± STANDARD DEVIATION OVER THREE SEEDS; n IS THE NUMBER OF TEST SAMPLES. BEST VALUE PER COLUMN IN BOLD.
<table><tr><td>True margin  $\sigma - 1$   $| \hat { \sigma } - \sigma |$ </td><td>118-bus  $\leq 0 . 0 5$   $> 0 . 0 5$ </td><td> $\leq 0 . 0 5$ </td><td>2000-bus</td></tr><tr><td></td><td> $( n = 9 7 8 )$ </td><td> $( n = 1 0 2 2 )$ </td><td> $( n = 7 9 7 )$   $( n = 1 2 3 )$ </td></tr><tr><td>GNS</td><td> ${ \bf 4 . 7 \pm 0 . 7 }$ </td><td> ${ \bf 4 0 . 8 \pm 7 . 6 }$ </td><td> $5 . 0 \pm 0 . 1$   $6 9 . 1 \pm 4 . 5$ </td></tr><tr><td>TabPFN</td><td> $0 . 0 \pm 0 . 0$ </td><td> $1 2 . 8 \pm 0 . 2$ </td><td> $3 . 2 \pm 0 . 4$   $7 5 . 9 \pm 0 . 5$ </td></tr><tr><td>PI-EVGNN</td><td> $0 . 0 \pm 0 . 0$   $0 . 3 \pm 0 . 3$ </td><td> $2 0 . 7 \pm 3 . 1$ </td><td> $5 . 2 \pm 0 . 6$   $7 5 . 6 \pm 4 . 5$ </td></tr><tr><td>LeapNet PowerFlowNet</td><td> $0 . 1 \pm 0 . 2$ </td><td> $1 0 . 9 \pm 0 . 6$   $1 6 . 3 \pm 1 . 2$ </td><td> ${ \bf 6 . 3 \pm 1 . 1 }$   $7 2 . 1 \pm 2 . 0$   $2 . 9 \pm 3 . 3$ </td></tr><tr><td></td><td></td><td></td><td> $7 9 . 4 \pm 4 . 5$ </td></tr><tr><td>Typed-GNN</td><td> $0 . 0 \pm 0 . 0$ </td><td> $5 . 4 \pm 0 . 8$ </td><td> $2 . 2 \pm 1 . 9$   $\mathbf { 7 9 . 9 \pm 5 . 0 }$ </td></tr></table>

III, we see that all the models have poor accuracy in the low margin range, including TabPFN. GNS is the best model on 118-bus grid but only has 4.7% of predictions within the ±0.01 tolerance threshold in the low margin range. This once again shows the limitations of the models at extrapolating to previously unseen loading levels and highlights the importance of low margin/near-infeasible OPs in the training data to accurately predict VSM at or near the nose point.

Furthermore, Fig. 8 shows the error distribution of the top 3 models on both test grids for the in-distribution experiment. For all models, the errors are smallest close to the nose point and larger with increasing margin. This is desirable from a power system security standpoint, but also reflects the distribution of the training data. Note that the models overestimate the margin near the nose, which may be of security concern. In addition to the point prediction, TabPFN also provides a predictive distribution for each prediction. The 90% quantile interval of its predictive distribution is shown in Fig. 8. We see that the lower bound of the 90% quantile interval is a conservative estimate for the margin at the nose point, with 88% and 96% of samples within the interval on the 118-bus and 2000-bus grids, respectively. This highlights the potential of TabPFN for probabilistic VSM prediction.

## V. DISCUSSIONS

The proposed TFM approach uses the pretrained TabPFN model without task-specific fine-tuning, and we find that its performance is competitive with the baseline models, which are always retrained for every task and experiment. We only need to change the model’s context for a new experiment in the case of TFM. Across the three evaluated tasks, no single model was consistently ranked as the best performing. For example, on the 118-bus grid, TabPFN had the lowest prediction error for V and Q for the in-distribution power-flow experiment. However, LeapNet had lower P and θ errors, while GNS had the most accurate VSM prediction. On the 2000-bus grid, TabPFN consistently ranked near the top, with lower powerflow prediction errors and VSM prediction accuracy. Notably, GNN-based models have larger errors on the 2000-bus system, which highlights the potential of TabPFN to scale to large systems better than GNN-based models.

![](images/978dec9d25d746339ee5709e5a9100c7111951a433ca3a3881ef99e52f41b6fe.jpg)  
Fig. 8. Prediction error of the loadability factor, σˆ − σ, against the true margin σ − 1 in the in-distribution experiment on the 118-bus (top) and 2000-bu (bottom) grids. The solid line is the median error, and the shaded bands contain 50% and 90% of the test points. The dashed lines show the predicted 90% interval of TabPFN.

All the ML models remain brittle in extrapolation scenarios, including the TFM. In zero-shot experiments for PF and VSM prediction, the performance of the TabPFN models was significantly degraded. However, TabPFN holds an advantage when it comes to generalization because it does not require model retraining for generalization, unlike the baselines. Only a new context is required. The model accuracy increased by about 80% for the VSM task on the 2000-bus system when the context data is changed from base OPs alone to in-distribution OPs. The baselines are unable to match the improved performance even with retraining with the in-distribution OPs. The proposed TabPFN variants further demonstrate that the inductive bias for model prediction can be added via handcrafted features and/or targets, instead of via the model architecture, as in GNNs. Predicting DCPF residuals (TabPFN-R) leads to improved accuracy, while decoupling P θ and VQ enables scaling to the 2000-bus grid without sacrificing accuracy. The proposed linear topology feature also improves the average recall in contingency classification. We note that there is further scope for improved feature engineering.

TABLE IV  
AVERAGE COMPUTATIONAL COST IN SECONDS: TOTAL TRAINING TIME (tr.) AND TOTAL INFERENCE TIME ON THE TEST SET (inf.); n IS THE NUMBER OF TEST SAMPLES. TABPFN REQUIRES NO TRAINING; ITS CONTEXT IS PROCESSED AT INFERENCE AND INCLUDED IN inf.
<table><tr><td></td><td colspan="2">PF 118-bus  $( n = 5 , 6 0 0 )$ </td><td colspan="2"> $\operatorname { P F } 2 0 0 0 – \mathrm { b u s }$   $( n = 2 , 8 0 0 )$ </td><td colspan="2">N-2 screening  $( n = 3 , 1 3 3 , 8 \bar { 9 2 } )$ </td></tr><tr><td>Model</td><td>tr</td><td>inf</td><td>tr</td><td>inf</td><td>tr</td><td>inf</td></tr><tr><td>DCPF</td><td></td><td>10</td><td></td><td>10</td><td>1</td><td>637</td></tr><tr><td>LeapNet</td><td>164</td><td>11</td><td>140</td><td>11</td><td>101</td><td>38</td></tr><tr><td>PowerFlowNet</td><td>829</td><td>8</td><td>4,369</td><td>10</td><td>512</td><td>176</td></tr><tr><td>Typed-GNN</td><td>7,291</td><td>10</td><td>9,944</td><td>12</td><td>4,392</td><td>621</td></tr><tr><td>GNS</td><td>1,901</td><td>12</td><td>4,065</td><td>13</td><td>914</td><td>177</td></tr><tr><td>PI-EVGNN</td><td>614</td><td>7</td><td>2,197</td><td>10</td><td>437</td><td>163</td></tr><tr><td>TabPFN</td><td></td><td>4,731</td><td></td><td>72,577</td><td>一</td><td>2,752</td></tr></table>

Finally, Table IV compares the computational cost of the models. TabPFN requires no training, but its inference is slower than that of the baseline models. For contingency screening, TabPFN assesses 3.1 million contingencies in about 2752s, compared to 637s for DCPF, while for PF, its inference is considerably slower because each target variable is predicted sequentially. In theory, each target can be predicted in parallel since they are predicted independently. However, this may rapidly increase memory consumption. The computational benefit of the TabPFN model therefore lies in not requiring training and not in its inference speed.

## VI. CONCLUSION

This paper investigated the use of a pretrained TFM (TabPFN) as a unified model for power system steady-state analysis tasks. The proposed model does not require re-training or fine-tuning, but performs ICL on the provided context samples during inference. Through case studies on the 118- and 2000-bus grids, the paper evaluated the accuracy and generalization performance of the proposed approach on PF, VSM prediction and N-k contingency screening. The proposed TFM method is competitive with task-specific baselines across the three tasks and is especially accurate for the 2000-bus PF and VSM tasks. This shows the potential for pretrained TFM in power system workflows. Future work will investigate the scaling up of TFM to even larger test systems, methods to select the best context for ICL, extraction of task-relevant features and computationally efficient inference methods for tasks with multiple outputs such as PF. Overall, our results suggest that pretrained TFMs can serve as a practical and simple alternative to task-specific neural surrogates for powersystem steady-state analysis.

## REFERENCES

[1] B. Kroposki, B. Johnson, Y. Zhang, V. Gevorgian, P. Denholm, B.-M. Hodge, and B. Hannegan, “Achieving a 100% Renewable Grid: Operating Electric Power Systems with Extremely High Levels of Variable Renewable Energy,” IEEE Power and Energy Magazine, vol. 15, no. 2, pp. 61–73, Mar. 2017.

[2] F. Bellizio, J. L. Cremer, and G. Strbac, “Machine-learned security assessment for changing system topologies,” International Journal of Electrical Power & Energy Systems, vol. 134, p. 107380, Jan. 2022.

[3] A. K. Rivera, A. Bhagavathula, A. Carbonero, and P. L. Donti, “PFDelta: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations,” Advances in Neural Information Processing Systems (NeurIPS 2025) Datasets and Benchmarks Track, 2025.

[4] H. F. Hamann et al., “Foundation models for the electric power grid,” Joule, vol. 8, no. 12, pp. 3245–3258, Dec. 2024.

[5] J. W. Simpson-Porco, “Lossy DC Power Flow,” IEEE Transactions on Power Systems, vol. 33, no. 3, pp. 2477–2485, May 2018.

[6] S. Chatzivasileiadis, A. Venzke, J. Stiasny, and G. Misyris, “Machine Learning in Power Systems: Is It Time to Trust It?” IEEE Power and Energy Magazine, vol. 20, no. 3, pp. 32–41, May 2022.

[7] M. Gillioz, G. Dubuis, and P. Jacquod, “A large synthetic dataset for machine learning applications in power transmission grids,” Scientific Data, vol. 12, no. 1, p. 168, Jan. 2025.

[8] O. Arowolo, J. Stiasny, and J. Cremer, “Exploring extrapolation of machine learning models for power system time domain simulation,” Sustainable Energy, Grids and Networks, vol. 43, p. 101908, Sep. 2025.

[9] C. Ren and Y. Xu, “Transfer Learning-Based Power System Online Dynamic Security Assessment: Using One Model to Assess Many Unlearned Faults,” IEEE Transactions on Power Systems, vol. 35, no. 1, pp. 821–824, Jan. 2020.

[10] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, Mar. 2017.

[11] J. Stiasny, B. Zhang, and S. Chatzivasileiadis, “PINNSim: A simulator for power system dynamics based on Physics-Informed Neural Networks,” Electric Power Systems Research, vol. 235, p. 110796, Oct. 2024.

[12] F. Gama, J. Bruna, and A. Ribeiro, “Stability Properties of Graph Neural Networks,” IEEE Transactions on Signal Processing, vol. 68, pp. 5680– 5695, 2020, arXiv:1905.04497 [cs].

[13] J. B. Hansen, S. N. Anfinsen, and F. M. Bianchi, “Power Flow Balancing with Decentralized Graph Neural Networks,” IEEE Transactions on Power Systems, vol. 38, no. 3, pp. 2423–2433, May 2023.

[14] O. Arowolo and J. L. Cremer, “Towards generalization of graph neural networks for AC optimal power flow,” Energy and AI, vol. 25, p. 100842, Sep. 2026.

[15] U. Alon and E. Yahav, “On the Bottleneck of Graph Neural Networks and its Practical Implications,” in International Conference on Learning Representations (ICLR). arXiv, Mar. 2021.

[16] T. Falconer and L. Mones, “Leveraging Power Grid Topology in Machine Learning Assisted Optimal Power Flow,” IEEE Transactions on Power Systems, vol. 38, no. 3, pp. 2234–2246, May 2023.

[17] R. Yu, C. Gu, J. Stiasny, Q. Wen, W. S. Dilov, L. Qi, and J. L. Cremer, “PriceFM: Foundation Model for Probabilistic Electricity Price Forecasting,” Sep. 2025, arXiv:2508.04875 [cs].

[18] J. L. Cremer, “Customising Electricity Contracts at Scale With Large Language Models,” IEEE Transactions on Power Systems, vol. 41, no. 4, pp. 2953–2966, Jul. 2026.

[19] C. Huang, S. Li, R. Liu, H. Wang, and Y. Chen, “Large Foundation Models for Power Systems,” in 2024 IEEE Power & Energy Society General Meeting (PESGM). Seattle, WA, USA: IEEE, Jul. 2024, pp. 1–5.

[20] H. Li, L. Mai, C. Xiao, E. Blasch, and Y. Weng, “Predicting Power-System Dynamic Trajectories with Foundation Models,” 2026, arXiv:2604.14991 [cs.AI].

[21] A. Puech et al., “GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis,” Aug. 2026, arXiv:2608.09921 [cs.AI].

[22] S. Muller, N. Hollmann, S. P. Arango, J. Grabocka, and F. Hutter,¨ “Transformers can do bayesian inference,” in International Conference on Learning Representations, 2022.

[23] N. Hollmann, S. Muller, L. Purucker, A. Krishnakumar, M. K¨ orfer, S. B.¨ Hoo, R. T. Schirrmeister, and F. Hutter, “Accurate predictions on small data with a tabular foundation model,” Nature, vol. 637, no. 8045, pp. 319–326, Jan. 2025.

[24] O. Arowolo, M. Yang, and J. Cremer, “Revisiting data-driven dynamic security assessment with a tabular foundation model,” 2026, arXiv:2607.16031 [cs.LG].

[25] N. Christianson, W. Cui, S. Low, W. Yang, W. Yang, and B. Zhang, “Fast and Reliable n − k Contingency Screening with Input-Convex Neural Networks,” Proceedings of Machine Learning Research, vol. 283, pp. 1–13, 2025.

[26] N. Hollmann, S. Muller, K. Eggensperger, and F. Hutter, “Tabpfn: A¨ transformer that solves small tabular classification problems in a second,” in International Conference on Learning Representations, 2023.

[27] T. Nagler, “Statistical Foundations of Prior-Data Fitted Networks,” Proceedings of the 40th International Conference on Machine Learning, vol. 202, 2023.

[28] L. Grinsztajn et al., “TabPFN-3: Technical Report,” May 2026, arXiv:2605.13986 [cs.LG].

[29] H. Qi, C. Li, Y. Liu, R. Sun, and J. Yan, “Continuous Feature Representation of Fault Location Based on Electrical Distance for Generalizable Data-Driven Dynamic Security Assessment,” IEEE Transactions on Power Systems, vol. 40, no. 4, pp. 2930–2942, Jul. 2025.

[30] J. L. Cremer, “Polynomial Line Outage Distribution Factors for Estimating Expected Congestion and Security,” IEEE Transactions on Power Systems, vol. 40, no. 3, pp. 2532–2544, May 2025.

[31] A. Puech et al., “gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation,” Dec. 2025, arXiv:2512.14658 [cs].

[32] B. Donon, R. Clement, B. Donnot, A. Marot, I. Guyon, and M. Schoe-´ nauer, “Neural networks for power flow: Graph neural solver,” Electric Power Systems Research, vol. 189, p. 106547, 2020.

[33] T. B. Lopez-Garcia and J. A. Dom´ınguez-Navarro, “Power flow analysis via typed graph neural networks,” Engineering Applications of Artificial Intelligence, vol. 117, p. 105567, Jan. 2023.

[34] N. Lin, S. Orfanoudakis, N. O. Cardenas, J. S. Giraldo, and P. P. Vergara, “PowerFlowNet: Power flow approximation using message passing Graph Neural Networks,” International Journal of Electrical Power & Energy Systems, vol. 160, p. 110112, Sep. 2024.

[35] A. M. Nakiganda and S. Chatzivasileiadis, “Graph Neural Networks for Fast Contingency Analysis of Power Systems,” Mar. 2025, arXiv:2310.04213 [eess.SY].

[36] B. Donon, B. Donnot, I. Guyon, Z. Liu, A. Marot, P. Panciatici, and M. Schoenauer, “LEAP nets for system identification and application to power systems,” Neurocomputing, vol. 416, pp. 316–327, Nov. 2020.

[37] A. Abed, “WSCC voltage stability criteria, undervoltage load shedding strategy, and reactive power reserve monitoring methodology,” in 199 IEEE Power Engineering Society Summer Meeting. Conference Proceedings (Cat. No.99CH36364), vol. 1. Edmonton, Alta., Canada: IEEE, 1999, pp. 191–197.