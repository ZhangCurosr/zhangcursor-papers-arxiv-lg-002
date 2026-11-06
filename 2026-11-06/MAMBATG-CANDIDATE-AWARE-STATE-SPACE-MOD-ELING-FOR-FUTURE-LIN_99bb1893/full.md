# MAMBATG: CANDIDATE-AWARE STATE SPACE MOD-ELING FOR FUTURE LINK PREDICTION ON TEMPORAL GRAPHS

Nguyen Hong Quan<sup>1</sup>, Sitao Luan<sup>2,3</sup> <sup>∗</sup>, Yilun Zheng<sup>1</sup>, Lihui Chen<sup>1</sup> <sup>∗</sup>

<sup>1</sup>Nanyang Technological University, Singapore

<sup>2</sup>University of Montreal

<sup>3</sup>Mila – Quebec Artificial Intelligence Institute

## ABSTRACT

Future link prediction on temporal graphs involves both repeated interactions between previously connected node pairs and first-time interactions between pairs with no prior interaction. While repeated-edge prediction can exploit direct pairwise evidence, unseen-edge prediction must generalize beyond historical recurrence. Existing temporal graph models often exploit signals tied to recurring interactions or historical overlap, which are highly informative for repeated edges but become limited or unavailable for unseen edges, contributing to substantial performance degradation on unseen-edge prediction. Based on this observation, we identify two complementary signals for modeling sequential dynamics beyond historical recurrence: (1) historical-sequence evolution, which captures dependencies among the source node’s chronologically ordered interactions, and (2) candidate–history compatibility, which captures how well a candidate fits the source node’s evolving history. Based on the decomposition, we propose MambaTG, a candidate-aware Mamba-based framework that jointly models these two signals. Mamba provides a natural mechanism for this design: its selective recurrence captures the evolution of ordered interaction histories, while its inputdependent state transitions allow the history state to adapt to different candidate inputs. Extensive experiments on 13 temporal graph datasets against 13 representative baselines show that MambaTG achieves strong and consistent performance across both repeated and unseen edges while remaining computationally efficient.

## 1 INTRODUCTION

Temporal graphs represent entities as nodes and their interactions as time-stamped edges (Kazemi et al., 2020). They provide a natural framework for modeling evolving relationships in applications such as social networks (Alvarez-Rodriguez et al., 2021) and recommender systems (Yu et al., 2024). A fundamental task in temporal graph learning isfuture linkprediction, which predicts interactions at a future query time from historical interactions (Divakaran & Mohan, 2020). Future interactions can be divided into two categories, (1) repeated edges, connecting previously interacting node pairs, and (2) unseen edges, connecting pairs with no prior interaction (Yi et al., 2025b). Both arise naturally in real-world systems: users on a social platform may repeatedly interact with existing connections while also forming new connections with previously unrelated users. Despite their coexistence, the two regimes present distinct predictive challenges. Repeated-edge prediction can directly exploit prior pairwise interactions, whereas unseen-edge prediction must infer future relationships from broader patterns in historical behaviors. A practical temporal graph model must therefore capture recurring relationships while also generalizing to new ones without prior pairwise interactions.

However, some existing temporal link prediction methods still rely heavily on signals associated with repeated interactions. The competitiveness of EdgeBank, a non-parametric baseline that predicts links by memorizing previously observed edges, shows that historical recurrence alone can provide a remarkably strong signal on conventional benchmarks (Poursafaei et al., 2022). This bias toward recurrence is reflected in representative temporal graph neural network (T-GNN) designs. Recent methods such as DyGFormer(Yu et al., 2023), FreeDyG(Tian et al., 2024), DyG-Mamba(Li et al., 2025), and DyGMamba(Ding et al., 2025) explicitly encode source–destination historical overlap through neighbor co-occurrence or interaction-frequency features. Such signals are highly informative for repeated edges but may be sparse or unavailable for unseen edges, particularly in bipartite graphs. Consistent with this limitation, recent benchmarks report substantially lower performance on unseen edges than on repeated edges (Yi et al., 2025b). These observations suggest that effective future link prediction requires moving beyond recurrence-based evidence and learning historical patterns that can generalize to previously unseen interactions.

![](images/b4c926c38ab28de24f90b7bbf2e0a39cc3c032e6f1ce99a9232b5e8d4ec8e49a.jpg)  
Figure 1: Illustration of two complementary signals for predicting Learner A’s next course choice from previously taken courses in Python, Machine Learning, Computer Vision, and Robotics.

This raises a more fundamental question: What predictive information should a temporal graph model learnfrom historical interactions to generalize beyond direct pairwise recurrence? We argue that the answer lies in two complementary signals: historical-sequence evolution and candidate– history compatibility, as illustrated through Learner A’s historical course interactions in Figure 1. (1) Historical-sequence evolution (Figure 1(a)) captures how historical interactions evolve in temporal order. The same interactions can imply different future behavior when ordered differently, as later interactions may reflect intentions developed through earlier ones. Capturing this signal therefore requires modeling dependencies among historical interactions as an ordered sequence, such that later interactions are interpreted in the context of the preceding history. (2) Candidate– history compatibility (Figure 1(b)) captures how well a candidate fits the source node’s evolving history. Explicitly modeling this relationship allows predictions to adapt to each candidate. However, many T-GNNs construct source and candidate representations largely independently and combine them only at the prediction stage (Kumar et al., 2019; Cong et al., 2023), limiting direct modeling of the relationship between the candidate and the source history. CRAFT (Yi et al., 2025a) directly matches each candidate against the source node’s individual historical interactions, but does not explicitly model dependencies arising from the history’s sequential evolution before candidate matching. Overall, the two signals capture distinct but complementary aspects of future interaction prediction: historical-sequence evolution characterizes dependencies within the ordered interaction history, whereas candidate–history compatibility reflects how that evolving history informs the pre diction of a candidate interaction.

Jointly modeling these complementary signals poses a non-trivial design challenge: historicalsequence evolution is shared across candidates, whereas candidate–history compatibility is inherently candidate-specific. To address this challenge, we propose MambaTG, an SSM-based framework that uses Mamba (Gu & Dao, 2024) as its sequential encoder. Its Candidate-Aware Mamba Encoder (CAME) first encodes the source’s ordered historical interactions into a shared history state and then adapts this state to each candidate through Mamba’s input-dependent state transitions. This design evaluates each candidate in the context of the evolved history while avoiding repeated history encoding. In addition, interaction histories may contain predictive patterns at different temporal scales, from slowly evolving preferences to rapid behavioral shifts. When modeled directly, dominant patterns at one scale may obscure weaker but predictive dynamics at another. We therefore introduce Dual-Branch Representation Refinement (DRR) to adaptively refine historical and candidate representations, providing CAME with more informative inputs for sequential encoding.

The main contributions of this work are summarized as follows:

• A unified view of sequential dynamics. We characterize sequential dynamics for future link prediction through two complementary signals: historical-sequence evolution and candidate–history compatibility. Through empirical analysis, we show that both signals contribute to strong performance across repeated- and unseen-edge prediction.

• An efficient candidate-aware Mamba-based framework. We propose MambaTG, which jointly captures these two signals through a Candidate-Aware Mamba Encoder and improves input representations through Dual-Branch Representation Refinement. By reusing the encoded historical state across candidates, MambaTG avoids repeated history encoding and enables efficient candidate evaluation.

• Extensive empirical validation. We evaluate MambaTG on 13 future link prediction datasets. MambaTG achieves strong performance across repeated- and unseen-edge prediction and attains the best average rank among all baselines. Efficiency and ablation studies further demonstrate its scalability and validate its principal design choices.

## 2 RELATED WORK

Temporal Graph Learning Methods for Future Link Prediction. Temporal graph models use diverse architectures to encode interaction histories, including memory-based methods such as JODIE (Kumar et al., 2019), DyRep (Trivedi et al., 2019), and TGN (Rossi et al., 2020), attentionbased methods such as TGAT (Xu et al., 2020), TCL (Wang et al., 2021a), DyGFormer (Yu et al., 2023), and SimpleDyG (Wu et al., 2024), and alternative encoders such as CAWN (Wang et al., 2021b), GraphMixer (Cong et al., 2023), and FreeDyG (Tian et al., 2024). From the perspective of sequential dynamics, these methods differ in how they capture historical-sequence evolution and candidate–history compatibility. Many methods encode the source history and candidate largely independently and combine their representations only at the prediction stage (Kumar et al., 2019; Trivedi et al., 2019; Rossi et al., 2020; Xu et al., 2020; Cong et al., 2023; Wu et al., 2024), limiting direct candidate conditioning of the source history. Other approaches incorporate information from both endpoints through temporal motifs, cross-subgraph interactions, neighbor co-occurrence, or pairwise statistics (Wang et al., 2021b;a; Yu et al., 2023; Tian et al., 2024), but primarily characterize relationships between the source and destination histories rather than how a candidate relates to the source’s evolving history. CRAFT (Yi et al., 2025a) directly matches each candidate against the source node’s individual historical interactions, but does not explicitly model dependencies arising from the history’s sequential evolution before candidate matching. In contrast, MambaTG jointly models historical-sequence evolution and candidate–history compatibility by first encoding the or dered source history to model its evolution and then adapting the resulting state to each candidate.

Mamba-Based Models for Continuous-Time Dynamic Graphs. Mamba (Gu & Dao, 2024) is a selective state-space model that efficiently models long sequences through input-dependent state transitions with linear sequence-length complexity. Recent studies have adapted Mamba to continuous-time dynamic graphs. DyGMamba (Ding et al., 2025) uses node- and time-level Mamba to encode long histories and identify temporally relevant interactions, while DyG-Mamba (Li et al., 2025) incorporates irregular inter-event times into state transitions for elapsed-time-aware forgetting and selective retention. These methods primarily focus on efficient long-history encoding and irregular temporal dynamics. In contrast, MambaTG uses candidate-aware continuation from the evolving source history to jointly model historical-sequence evolution and candidate–history compatibility, targeting effective prediction across both repeated and unseen edges.

## 3 PRELIMINARIES

Temporal Graph and Future Link Prediction. A temporal graph is represented as $\mathcal { G } = ( \nu , \mathcal { E } )$ where V is the node set and $\mathcal { E } = ( ( u _ { i } , v _ { i } , t _ { i } ) ) _ { i = 1 } ^ { M }$ is a chronologically ordered stream of M interactions. Here, $u _ { i } \in \mathcal V$ and $v _ { i } \in \mathcal V$ denote the source and destination nodes of the i-th interaction, respectively, and $t _ { i }$ denotes its timestamp, with $t _ { 1 } \leq t _ { 2 } \leq \cdot \cdot \cdot \leq t _ { M }$ . At a query time $t _ { q } ,$ , the observed history is $\mathcal { E } _ { < t _ { q } } = \{ ( u _ { i } , v _ { i } , t _ { i } ) \in \mathcal { E } \mid t _ { i } < t _ { q } \}$ . Given a source node u and a candidate set $\mathcal { C } _ { u } ^ { t _ { q } }$ future link prediction assigns each candidate destination $c _ { i } \in \mathcal { C } _ { u } ^ { t _ { q } }$ a score conditioned on $\mathcal { E } _ { < t _ { q } }$ and ranks the true destination above negative candidates. A candidate edge $( u , c _ { i } )$ is repeated if the pair appears in $\mathcal { E } _ { < t _ { q } }$ , and unseen otherwise (Yi et al., 2025b).

Discrete Fourier Transform. Let $\mathbf { X } = ( \mathbf { x } _ { n } ) _ { n = 0 } ^ { N - 1 } \in \mathbb { R } ^ { N \times d }$ be a sequence of d-dimensional representations. The discrete Fourier transform (DFT) and its inverse (IDFT), applied along the temporal dimension, are defined as

$$
\widehat { \mathbf { x } } _ { m } = \sum _ { n = 0 } ^ { N - 1 } \mathbf { x } _ { n } \exp \left( - \mathrm { i } \frac { 2 \pi m n } { N } \right) , \qquad \mathbf { x } _ { n } = \frac { 1 } { N } \sum _ { m = 0 } ^ { N - 1 } \widehat { \mathbf { x } } _ { m } \exp \left( \mathrm { i } \frac { 2 \pi m n } { N } \right) ,\tag{1}
$$

where $\widehat { \mathbf { x } } _ { m } \in \mathbb { C } ^ { d }$ is the m-th frequency component and $\mathrm { i } ^ { 2 } = - 1$ . For real-valued inputs, the real FFT (rFFT) and its inverse (irFFT) are commonly used (Sorensen et al., 1987).

State Space Models. State space models (SSMs) represent sequences through latent states that evolve over time. Given $x ( t ) \in \mathbb { R } .$ , an SSM produces y(t) ∈ R from a latent state $h ( t ) \in \mathbb { R } ^ { N }$ :

$$
h ^ { \prime } ( t ) = A h ( t ) + B x ( t ) , \qquad y ( t ) = C h ( t ) .\tag{2}
$$

Here, $A \in \mathbb { R } ^ { N \times N }$ is the state-transition matrix, $B \in \mathbb { R } ^ { N \times 1 }$ projects the input into the latent state, and $C \in \mathbb { R } ^ { 1 \times N }$ maps the latent state to the output. To process a discrete sequence, the continuous system is discretized using a step size $\Delta$

$$
h _ { t } = \bar { A } h _ { t - 1 } + \bar { B } x _ { t } , \qquad y _ { t } = C h _ { t } .\tag{3}
$$

Under zero-order-hold discretization, the discrete parameters are given by $\bar { A } ~ = ~ \exp ( \Delta A )$ and $\bar { B } = ( \Delta A ) ^ { - 1 } \big ( \exp ( \Delta A ) - I \big ) \Delta B$ . Unlike conventional SSMs with input-independent parameters, Mamba uses input-dependent $B , C ,$ , and $\Delta$ with a hardware-aware selective scan for efficient recurrence computation (Gu & Dao, 2024).

## 4 METHOD

The overall MambaTG framework is illustrated in Figure 2. Given a source node u, a query timestamp $t _ { q } .$ and its chronologically ordered interaction history $\mathcal { H } _ { u } ^ { t _ { q } } = \left( \left( v _ { 1 } , t _ { 1 } \right) , \ldots , \left( v _ { \ell _ { u } } , t _ { \ell _ { u } } \right) \right)$ , where $\ell _ { u } \ \leq \ \dot { L } _ { \mathrm { m a x } }$ and $t _ { 1 } \leq \cdot \cdot \cdot \leq t _ { \ell _ { u } } < t _ { q } ,$ MambaTG assigns a link-prediction score to each candidate $c _ { i } \in \mathcal { C } _ { u } ^ { t _ { q } }$ . It first combines node embeddings with query-relative time encodings to construct historical and candidate representations. Dual-Branch Representation Refinement then enhances these representations through separate historical and candidate refinement branches, after which Candidate-Aware Mamba Encoder recurrently encodes the refined history and incorporates each candidate through an additional state update. Finally, a link classifier maps each history-conditioned candidate representation to a prediction score.

![](images/4699605831544acb9b62fe7881f713273802d4a251852b94a88127f9392ddc72.jpg)  
Figure 2: Framework of the proposed MambaTG model.

## 4.1 NODE AND TIME ENCODING.

Node embedding. Following Yi et al. (2025a), we associate node $v \in \mathcal V$ with a trainable identifier embedding $\mathrm { E m b } ( v ) \in \mathbb { R } ^ { d _ { e } }$ . Given a query time $t _ { q } ,$ let $\mathcal { C } _ { u } ^ { t _ { q } }$ denote the set of $q = | \mathcal { C } _ { u } ^ { t _ { q } } |$ candidate destinations. We arrange the embeddings of the $\ell _ { u }$ historical neighbors and q candidate destinations row-wise as ${ \bf E } _ { H } = [ \bar { \mathrm { E m b } } ( v _ { i } ) ] _ { i = 1 } ^ { \ell _ { u } } \in \bar { \mathbb { R } ^ { \ell _ { u } \times d _ { e } } }$ and ${ \bf E } _ { D } = [ \mathrm { E m b } ( c _ { j } ) ] _ { j = 1 } ^ { q } \in \mathbb { R } ^ { q \times d _ { \epsilon } }$ , respectively.

Time encoding. To incorporate temporal information, we encode elapsed times relative to the query time $t _ { q } .$ For each historical interaction occurring at time $t _ { i }$ , its elapsed time is $\Delta t _ { i } ^ { H } = t _ { q } - t _ { i }$ For candidate $c _ { j } .$ , we instead use its activity recency $\Delta t _ { c _ { i } } ^ { D } ~ = ~ t _ { q } - t _ { c _ { i } } ^ { \mathrm { l a s t } }$ , where $t _ { c _ { i } } ^ { \mathrm { l a s t } } < t _ { q }$ is its most recent interaction with any node in the global graph. For cold-start candidates with no prior interactions, we set $\Delta t _ { c _ { j } } ^ { D } = 0$ . Following Xu et al. (2020), each elapsed time is mapped to a $d _ { T ^ { - } }$ dimensional representation using $\phi : \mathbb { R } _ { \geq 0 }  \mathbb { R } ^ { d _ { T } }$ . The resulting historical and candidate time encodings are $\mathbf { T } _ { H } = [ \phi ( \Delta t _ { i } ^ { H } ) ] _ { i = 1 } ^ { \ell _ { u } } \in \mathbb { R } ^ { \ell _ { u } \times d _ { T } }$ and $\mathbf { T } _ { D } = [ \phi ( \Delta t _ { c _ { i } } ^ { D } ) ] _ { j = 1 } ^ { q } \in \mathbb { R } ^ { q \times d _ { T } }$ , respectively.

Feature fusion. To jointly capture node identity and temporal context, we combine the node and time representations: $\mathbf { \bar { Z } } _ { H } \overset { \cdot } { = } \operatorname { L i n e a r } ( [ \mathbf { E } _ { H } | | \mathbf { T } _ { H } ] ) \overset { \cdot } { \in } \mathbb { R } ^ { \ell _ { u } \times d }$ and ${ \bf Z } _ { D } = \mathrm { L i n e a r } ( [ { \bf E } _ { D } \Vert { \bf T } _ { D } ] ) \in \mathbb { R } ^ { q \times d }$ where ∥ denotes feature-wise concatenation and Linear(·) is a shared linear projection.

## 4.2 DUAL-BRANCH REPRESENTATION REFINEMENT (DRR).

An interaction history may simultaneously contain slowly evolving trends, recurring behavior, and abrupt short-term changes. Without prior refinement, stronger patterns at one scale may obscure subtler yet predictive dynamics at other scales. We therefore introduce Dual-Branch Representation Refinement, comprising Historical Spectral Refinement (HSR) and Candidate Feature Refinement (CFR). HSR exposes and adaptively refines frequency components associated with these patterns, while CFR recalibrates candidate features for compatibility with the refined history.

Historical Spectral Refinement. We transform $\mathbf { Z } _ { H }$ into the frequency domain along its temporal dimension, yielding $\widehat { \mathbf { Z } } _ { H } = \mathrm { r F F T } _ { L _ { \mathrm { m a x } } } ( \mathbf { Z } _ { H } ) \in \mathbb { C } ^ { ( \lfloor L _ { \mathrm { m a x } } / 2 \rfloor + 1 ) \times d }$ , where $L _ { \mathrm { m a x } } \ge \ell _ { u }$ is the transform length and zero-padding is applied when $\ell _ { u } < L _ { \mathrm { m a x } }$ . Inspired by spectral filtering in FreeDyG (Tian et al., 2024), HSR modulates each frequency component using learnable complex-valued weights:

$$
\widehat { \mathbf { Z } } _ { H } ^ { \mathrm { H S R } } = \widehat { \mathbf { Z } } _ { H } \odot \mathbf { W } _ { H } , \qquad \mathbf { W } _ { H } \in \mathbb { C } ^ { ( \lfloor L _ { \operatorname* { m a x } } / 2 \rfloor + 1 ) \times d } .\tag{4}
$$

Here, ⊙ denotes element-wise multiplication. The complex-valued weights $\mathbf { W } _ { H }$ enable the model to selectively adjust the magnitude and phase of temporal patterns across different frequencies. $\mathsf { A p - }$ plying the inverse transform with output length $L _ { \mathrm { m a x } }$ yields $\mathbf { Z } _ { H } ^ { \mathrm { H S R } } = \left[ \mathrm { i r F F T } _ { L _ { \mathrm { m a x } } } \left( \widehat { \mathbf { Z } } _ { H } ^ { \mathrm { H S R } } \right) \right] _ { 1 : \ell _ { u } } \in$ $\mathbb { R } ^ { \ell _ { u } \times d }$ , where the first $\ell _ { u }$ positions corresponding to the original history are retained.

Candidate Feature Refinement. To promote alignment between candidate representations and the spectrally refined historical representations, CFR learnably rescales candidate features through a lightweight feature-wise transformation. Specifically, for each candidate representation $\begin{array} { r l } { \mathbf { z } _ { c _ { i } } } & { { } = } \end{array}$ $\mathbf { Z } _ { D } [ i ] \in \mathbb { R } ^ { d }$ , we compute $\mathbf { z } _ { c _ { i } } ^ { \mathrm { C F R } } = \mathbf { z } _ { c _ { i } } \odot \mathbf { w } _ { D }$ where $\mathbf { w } _ { D } \in \mathbb { R } ^ { d }$ is a learnable vector shared across candidates.

## 4.3 CANDIDATE-AWARE MAMBA ENCODER (CAME).

Modeling sequential dynamics requires capturing both historical-sequence evolution and candidate– history compatibility. Mamba (Gu & Dao, 2024) is well suited to this objective because its selective recurrence summarizes ordered sequences into recurrent states while its input-dependent state transitions allow the same historical state to respond differently to different candidates. CAME stacks R layers, each consisting of a Mamba block followed by a position-wise feed-forward network (FFN). Detailed formulations of these standard components are provided in Appendix C.1.

Historical-sequence encoding. CAME first encodes the source node’s interaction history in chronological order to capture its sequential evolution. The refined historical representations $\mathbf { Z } _ { H } ^ { \mathrm { H S R } } \in \mathbb { R } ^ { \ell _ { u } \times d }$ are padded to $L _ { \mathrm { m a x } }$ for batching. We initialize ${ \bf X } _ { H } ^ { ( 0 ) } = \mathrm { P a d } ( { \bf Z } _ { H } ^ { \mathrm { H S R } } , L _ { \mathrm { m a x } } )$ . At each layer r, CAME processes the complete historical sequence once and retains its terminal recurrent cache:

$$
\left( \mathbf { M } _ { H } ^ { \left( r \right) } , K _ { H } ^ { \left( r \right) } \right) = \mathrm { M a m b a B l o c k } ^ { \left( r \right) } \left( \mathbf { X } _ { H } ^ { \left( r - 1 \right) } \right) , \qquad \widetilde { \mathbf { X } } _ { H } ^ { \left( r \right) } = \mathrm { L N } _ { 1 } ^ { \left( r \right) } \left( \mathbf { X } _ { H } ^ { \left( r - 1 \right) } + \mathbf { M } _ { H } ^ { \left( r \right) } \right) ,
$$

$$
{ \bf X } _ { H } ^ { ( r ) } = \mathrm { L N } _ { 2 } ^ { ( r ) } \left( \widetilde { { \bf X } } _ { H } ^ { ( r ) } + \mathrm { F F N } ^ { ( r ) } \left( \widetilde { { \bf X } } _ { H } ^ { ( r ) } \right) \right) , \qquad r = 1 , \ldots , R .\tag{5}
$$

Here, $\mathbf { M } _ { H } ^ { ( r ) }$ denotes the Mamba output representations for the historical sequence, while ${ \kappa _ { H } ^ { \left( r \right) } }$ contains the terminal SSM and causal-convolution states required to continue the r-th Mamba layer with one additional token. Through recurrent state propagation, historical representations incorporate information from earlier interactions, enabling CAME to capture historical-sequence evolution.

Candidate continuation. For candidate $c _ { i } .$ we initialize $\mathbf { x } _ { c _ { i } } ^ { ( 0 ) } = \mathbf { z } _ { c _ { i } } ^ { \mathrm { C F R } }$ . At each layer, the candidate is processed as one additional input following the historical sequence by continuing from the corresponding terminal history cache:

$$
\begin{array} { r l } & { \mathbf { m } _ { c _ { i } } ^ { ( r ) } = \mathrm { M a m b a S t e p } ^ { ( r ) } \left( \mathbf { x } _ { c _ { i } } ^ { ( r - 1 ) } ; \boldsymbol { K } _ { H } ^ { ( r ) } \right) , \qquad \widetilde { \mathbf { x } } _ { c _ { i } } ^ { ( r ) } = \mathrm { L N } _ { 1 } ^ { ( r ) } \left( \mathbf { x } _ { c _ { i } } ^ { ( r - 1 ) } + \mathbf { m } _ { c _ { i } } ^ { ( r ) } \right) , } \\ & { \qquad \mathbf { x } _ { c _ { i } } ^ { ( r ) } = \mathrm { L N } _ { 2 } ^ { ( r ) } \left( \widetilde { \mathbf { x } } _ { c _ { i } } ^ { ( r ) } + \mathrm { F F N } ^ { ( r ) } \left( \widetilde { \mathbf { x } } _ { c _ { i } } ^ { ( r ) } \right) \right) , \qquad r = 1 , \dots , R . } \end{array}\tag{6}
$$

Here, MambaStep<sup>(r)</sup> shares its parameters with MambaBlock<sup>(r)</sup> and continues the recurrence for one additional input using ${ \kappa } _ { H } ^ { \left( r \right) }$ . Since the selective state-space parameters depend on the candidate input $\mathbf { x } _ { c _ { i } } ^ { ( r - 1 ) }$ , each candidate induces a different state update from the shared historical cache. The resulting representation is therefore conditioned jointly on the candidate and the evolved historical state, capturing candidate–history compatibility. By reusing the terminal history cache across candidates, MambaTG encodes the history only once, while each candidate requires only one additional recurrent update per layer, avoiding redundant history encoding. After R layers, the historyconditioned candidate representation is $\mathbf { h } _ { u , c _ { i } } ^ { \bar { t } _ { q } } = \mathbf { x } _ { c _ { i } } ^ { ( R ) } \in \mathbb { R } ^ { d }$

## 4.4 LINK CLASSIFIER.

Following Yi et al. (2025a), we project the candidate’s activity recency, $\Delta t _ { c _ { i } } ^ { D } = t _ { q } - t _ { c _ { i } } ^ { \mathrm { l a s t } }$ , and concatenate it with the candidate representation $\mathbf { h } _ { u , c _ { i } } ^ { t _ { q } }$ . Prediction score is computed as:

$$
\hat { y } _ { u , c _ { i } } ^ { t _ { q } } = \mathrm { M L P } \big ( \big [ \mathbf { h } _ { u , c _ { i } } ^ { t _ { q } } \big | \big | \mathrm { L i n e a r } _ { \Delta } \big ( \Delta t _ { c _ { i } } ^ { D } \big ) \big ] \big ) ,\tag{7}
$$

where ∥ denotes feature-wise concatenation. For repeated-edge-dominant datasets, we use MambaTG-R, which additionally projects and concatenates the historical repetition frequency $n _ { u , c _ { i } } ^ { t _ { q } }$ , defined as the number of interactions between u and $c _ { i }$ before $t _ { q } .$ Both variants are optimized using Bayesian Personalized Ranking (BPR) (Rendle et al., 2009), which encourages observed destinations to receive higher scores than sampled negatives.

## 4.5 TIME COMPLEXITY ANALYSIS

Table 1: Candidate-scoring time complexity. Entries omit O(·) and fixed architectural constants.
<table><tr><td>Method</td><td></td><td>Historical-Neighbor Extraction Edge Likelihood Computation</td></tr><tr><td>TGN</td><td> $B ( 1 + q ) \log { \bar { d } }$ </td><td> $B ( 1 + q ) \ell d ^ { 2 }$ </td></tr><tr><td>TGAT</td><td> $B ( 1 + q ) ( 1 + \ell ) \log \bar { d }$ </td><td> $B ( \overset { \cdot } { 1 } + \overset { \cdot } { q } ) \ell ^ { 2 } d ^ { 2 }$ </td></tr><tr><td>GraphMixer</td><td> $B ( 1 + q ) \log { \bar { d } }$ </td><td> $B ( 1 + q ) \ell d ^ { 2 }$ </td></tr><tr><td>DyGFormer</td><td> $B ( 1 + q ) \log { \bar { d } }$ </td><td> $B ( 1 + \dot { q } ) ( \ell \dot { d } ^ { 2 } + \ell ^ { 2 } d )$ </td></tr><tr><td>DyG-Mamba</td><td> $B ( 1 + q ) \log { \bar { d } }$ </td><td> $B q \ell d ^ { 2 }$ </td></tr><tr><td>DyGMamba</td><td> $B ( 1 + q ) \log { \bar { d } }$ </td><td> $B \bar { q } \ell d ^ { 2 }$ </td></tr><tr><td>CRAFT</td><td> $B \log \bar { d }$ </td><td> $B ( \ell + q ) \bar { d } ^ { 2 } + B q \ell d$ </td></tr><tr><td>MambaTG</td><td> $B \log \bar { d }$ </td><td> $B ( \ell + q ) d ^ { 2 } + B \ell d \log \ell$ </td></tr></table>

We analyze the time complexity of future link prediction for a batch of source nodes and their candidate destinations. Let B denote the number of source queries, q the number of candidates per source, ℓ the number of historical interactions considered per node, d the feature dimension, and <sup>¯</sup>d the average node degree. Table 1 summarizes the resulting complexities; detailed derivations and assumptions are provided in Appendix I.

MambaTG retrieves only the source history, requiring $O ( B \log \bar { d } )$ time for temporal-index lookup. For edge likelihood computation, DRR refines each history in $O ( B \ell d \log \ell )$ time, while CAME encodes it once and reuses its layer-wise terminal caches across candidates, costing $O ( B ( \ell + q ) d ^ { 2 } )$ The total complexity is $O \big ( B ( \ell + q ) d ^ { 2 } + B \ell d \log \ell \big )$ . Unlike methods that encode both endpoints, MambaTG uses only the source history, avoiding candidate-dependent neighborhood retrieval. The separation of history encoding from candidate evaluation further avoids repeated processing across candidates and yields more favorable asymptotic scaling than CRAFT when $q > \log \ell ,$ as in our evaluation, with q = 101 and ℓ up to 1000. Together, they enable scalable prediction over long histories and large candidate sets in applications such as recommendation and social-link prediction.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Datasets. We evaluate MambaTG on 13 future link prediction datasets, including seven datasets from TGB-Seq (Yi et al., 2025b) and six commonly used temporal graph datasets. Following Yi et al. (2025a), we categorize them as repeated-edge-dominant or unseen-edge-dominant according to the relative prevalence of repeated and unseen edges. Dataset details are provided in Appendix B.2.

Baselines. We compare MambaTG with twelve representative temporal graph baselines: Edge-Bank (Poursafaei et al., 2022), CRAFT (Yi et al., 2025a), JODIE (Kumar et al., 2019), DyRep (Trivedi et al., 2019), TGAT (Xu et al., 2020), TGN (Rossi et al., 2020), CAWN (Wang et al., 2021b), GraphMixer (Cong et al., 2023), DyGFormer (Yu et al., 2023), FreeDyG (Tian et al., 2024), DyG-Mamba (Li et al., 2025), and DyGMamba (Ding et al., 2025). Following Yi et al. (2025a), we additionally include SGNN-HN (Pan et al., 2020), a session-based recommendation method, as a sequential baseline. Together, these methods span non-parametric, memory-based, attentionbased, walk-based, MLP-based, frequency-based, Mamba-based, and sequential recommendation architectures. Baseline details are provided in Appendix B.3.

Table 2: Future link prediction performance on (a) unseen-edge-dominant and (b) repeated-edgedominant datasets. Results are reported as MRR (%), averaged over three random seeds. The best, second-best, and third-best results are highlighted. Avg.Rk. denotes the average rank over datasets with reported results (lower is better); parentheses indicate coverage in (a).  
(a) Unseen-edge-dominant datasets
<table><tr><td>Model</td><td>ML-20M</td><td>Taobao</td><td>Yelp</td><td>GoogleLocal</td><td>Flickr</td><td>YouTube</td><td>WikiLink</td><td>Avg.Rk.</td></tr><tr><td>JODIE</td><td>21.16±0.73</td><td>48.36±2.18</td><td>69.88±0.31</td><td>41.86±1.49</td><td>46.21±0.83</td><td>41.67±2.86</td><td>57.94±1.33</td><td>6.43 (7/7)</td></tr><tr><td>DyRep</td><td>19.00±1.69</td><td>40.03±2.40</td><td>57.69±1.05</td><td>37.73±1.34</td><td>38.04±4.19</td><td>35.12±4.13</td><td>42.63±1.33</td><td>8.29 (7/7)</td></tr><tr><td>TGAT</td><td>10.47±0.20</td><td></td><td></td><td>19.78±0.24</td><td>23.53±3.35</td><td>43.56±2.53</td><td></td><td>10.75(4/7)</td></tr><tr><td>TGN</td><td>23.99±0.20</td><td>60.28±0.54</td><td>69.79±0.24</td><td>54.13±1.97</td><td>46.03±6.78</td><td>55.16±5.89</td><td>62.94±2.16</td><td>5.29 (7/7)</td></tr><tr><td>CAWN</td><td>12.31±0.02</td><td></td><td>25.71±0.09</td><td>18.26±0.02</td><td>48.69±6.08</td><td>47.55±1.08</td><td></td><td>9.00 (5/7)</td></tr><tr><td>GraphMixer</td><td>21.97±0.17</td><td>31.54±0.02</td><td>33.96±0.19</td><td>21.31±0.14</td><td>45.01±0.08</td><td>58.87±0.12</td><td>48.57±0.02</td><td>6.86 (7/7)</td></tr><tr><td>DyĠFormer</td><td></td><td></td><td>21.68±0.20</td><td>18.39±0.02</td><td>49.58±2.87</td><td>46.08±3.44</td><td></td><td>9.00 4/7</td></tr><tr><td>DyGMamba</td><td>5.86±0.21</td><td></td><td></td><td>17.64±0.06</td><td>52.07±0.43</td><td>54.09±1.68</td><td></td><td>9.25 (4/7)</td></tr><tr><td>DyG-Mamba</td><td></td><td></td><td></td><td>17.34±0.36</td><td>55.86±0.14</td><td>56.71±0.23</td><td></td><td>8.00 (3/7)</td></tr><tr><td>FreeDyG</td><td>20.67±0.70</td><td></td><td></td><td>20.49±0.13</td><td>56.22±0.33</td><td>60.35±0.21</td><td></td><td>5.00 (4/7)</td></tr><tr><td>SGNN-HN</td><td>33.12±0.01</td><td>68.58±0.21</td><td>69.34±0.44</td><td>62.88±0.51</td><td>60.15±0.20</td><td>59.64±0.22</td><td>69.37±0.39</td><td>3.14 (7/7)</td></tr><tr><td>CRAFT</td><td>35.91±0.65</td><td>70.68±0.42</td><td>72.69±0.61</td><td>62.35±0.46</td><td>62.34±0.84</td><td>58.92±0.16</td><td>75.48±0.84</td><td>2.29 (7/7)</td></tr><tr><td>MambaTG</td><td>37.98±0.11</td><td>73.06±0.21</td><td>75.85±0.17</td><td>64.62±0.45</td><td>63.94±0.72</td><td>59.81±0.74</td><td>75.38±0.13</td><td>1.29 (7/7)</td></tr></table>

(b) Repeated-edge-dominant datasets
<table><tr><td>Model</td><td>Wikipedia</td><td>Reddit</td><td>MOOC</td><td>LastFM</td><td>UCI</td><td>Flights</td><td>Avg.Rk.</td></tr><tr><td>EdgeBank</td><td>77.94±0.00</td><td>77.88±0.00</td><td>9.76±0.00</td><td>19.07±0.00</td><td>39.43±0.00</td><td>54.46±0.00</td><td>11.50</td></tr><tr><td>JODIE</td><td>76.48±1.72</td><td> $7 7 . 1 6 { \pm } 1 . 2 7$ </td><td>19.91±3.06</td><td>18.49±2.87</td><td>51.43±1.74</td><td>20.37±4.18</td><td>11.83</td></tr><tr><td>DyRep</td><td>67.42±4.21</td><td> $7 2 . 3 3 { \pm } 2 . 1 4 $ </td><td>17.71±1.27</td><td>17.76±6.56</td><td>14.00±2.71</td><td>15.62±0.47</td><td>13.67</td></tr><tr><td>TGAT</td><td>72.72±1.50</td><td> $7 8 . 0 3 { \pm } 0 . 2 5 $ </td><td>31.55±4.14</td><td>24.10±1.26</td><td>31.50±0.48</td><td>53.90±1.07</td><td>11.17</td></tr><tr><td>TGN</td><td>82.22±0.39</td><td> $7 9 . 2 5 { \pm } 0 . 2 4 $ </td><td>39.03±4.53</td><td>25.59±4.43</td><td>45.29±6.43</td><td>64.24±2.11</td><td>9.25</td></tr><tr><td>CAWN</td><td>86.07±0.08</td><td> $8 7 . 6 6 { \pm } 0 . 0 4$ </td><td>27.02±0.57</td><td>35.10±0.34</td><td>66.96±0.48</td><td>80.06±2.10</td><td>8.00</td></tr><tr><td>GraphMixer</td><td>73.57±1.48</td><td>70.87±0.38</td><td>29.02±0.28</td><td>26.76±1.42</td><td>59.58±1.00</td><td>42.26±0.49</td><td>10.83</td></tr><tr><td>DyGFormer</td><td>88.69±0.04</td><td>88.70±0.05</td><td>42.20±1.31</td><td>46.50±0.51</td><td>76.61±0.26</td><td>84.13±3.46</td><td>4.83</td></tr><tr><td>DyGMamba</td><td>89.10±0.17</td><td>89.00±0.03</td><td>47.17±1.43</td><td>46.39±0.49</td><td>76.98±0.46</td><td>85.02±0.28</td><td>3.50</td></tr><tr><td>DyG-Mamba</td><td>88.50±0.21</td><td>88.77±0.14</td><td>47.33±0.23</td><td>53.42±0.19</td><td>78.25±0.60</td><td>81.47±0.04</td><td>4.00</td></tr><tr><td>FreeDyG</td><td>90.18±0.12</td><td>89.16±0.05</td><td>44.51±0.37</td><td>49.19±0.55</td><td>74.55±1.12</td><td>81.10±0.06</td><td>4.33</td></tr><tr><td>SGNN-HN</td><td>82.22±0.35</td><td>87.42±0.04</td><td>57.25±0.17</td><td>40.72±0.26</td><td>57.23±0.39</td><td>80.80±0.14</td><td>7.08</td></tr><tr><td>CRAFT-R</td><td>88.25±0.26</td><td>89.33±0.11</td><td>62.32±0.39</td><td>55.21±0.23</td><td>75.11±0.09</td><td>83.16±0.16</td><td>3.50</td></tr><tr><td>MambaTG-R</td><td>88.99±0.02</td><td>89.71±0.10</td><td>64.77±0.22</td><td>56.98±0.24</td><td>79.31±0.26</td><td>84.51±0.03</td><td>1.50</td></tr></table>

Evaluation Settings and Metrics. Following TGB-Seq (Yi et al., 2025b), we use Mean Reciprocal Rank (MRR) (Craswell, 2009) as the evaluation metric and adopt official negative samples when available. Otherwise, we randomly sample 100 negative destinations per positive interaction. All datasets are split chronologically into 70%/15%/15% for training, validation, and testing, respectively. We report the average performance over three random seeds. Further evaluation and implementation details are provided in Appendix B.4.

## 5.2 PREDICTIVE EFFECTIVENESS AND COMPUTATIONAL EFFICIENCY

Overall Predictive Performance. Table 2 reports MRR across all datasets. We use MambaTG for unseen-edge-dominant datasets and MambaTG-R for repeated-edge-dominant datasets, with the latter additionally incorporating source–candidate repetition frequency. EdgeBank, FreeDyG, DyG-Mamba, and DyGMamba results are from our experiments, while the remaining baseline results are taken from prior studies (Yi et al., 2025a;b). Missing entries indicate experiments exceeding the specified runtime or GPU memory budget. Further details are provided in Appendix B.4.

MambaTG and MambaTG-R achieve the best performance on 9 of the 13 datasets, with strong performance across the remaining datasets, and attain the best average rank in both regimes: 1.29 on unseen-edge-dominant datasets and 1.50 on repeated-edge-dominant datasets. We attribute these gains primarily to two design choices: (i) CAME jointly captures historical-sequence evolution and candidate–history compatibility, supporting prediction across repeated and unseen edges, as validated in Section 5.3 and Appendix G; and (ii) DRR refines historical patterns associated with different temporal scales while promoting compatibility between the refined history and candidate representations, as examined in Section 5.4 and Appendix F. Moreover, layer-wise caching encodes each source history once and reuses its terminal states across candidates, enabling MambaTG to scale to large datasets on which several competing methods exceed the computational budget, as analyzed in Figure 4 and Appendix J.1.

Long-Sequence Effectiveness. To assess MambaTG’s ability to exploit long interaction histories, we evaluate it on LastFM, which benefits from longer historical sequences. We vary the maximum source-history length ℓ and compare MambaTG-R with other representative baselines.<sup>1</sup> As shown in Figure 3, MambaTG-R maintains strong performance as ℓ increases, while CRAFT-R and DyGFormer degrade with longer histories. These results demonstrate MambaTG’s robustness to increasing history lengths and its ability to leverage extended temporal context effectively.

![](images/b863dbc86d2c31fad2d2bd563089fa9feffdbf9272f986f5730b28ef65258036.jpg)  
Figure 3: Long-sequence effectiveness on LastFM.

Performance–Efficiency Trade-off. Figure 4 illustrates the trade-off between predictive performance and computational efficiency on MOOC. MambaTG-R consistently achieves the highest test MRR while remaining competitive in parameter size and per-epoch training time. In particular, MambaTG-R achieves up to 8.9× faster per-epoch training and a 6.3× smaller parameter footprint than competing methods. Overall, MambaTG-R offers the most favorable tradeoff between predictive performance and computational efficiency across the evaluated datasets. Results on UCI and LastFM in Appendix J.2.

![](images/772220aa30a3e75216ce280a3be0c6a634f22d7090eb48c53ba6cedaa45138d9.jpg)  
Figure 4: Performance versus efficiency on MOOC. Bubble size indicates model size.

## 5.3 VALIDATING HISTORICAL-SEQUENCE EVOLUTION AND CANDIDATE–HISTORY COMPATIBILITY

Our hypothesis is that MambaTG’s effectiveness arises from jointly modeling historical-sequence evolution and candidate–history compatibility. To validate this hypothesis, we examine whether performance decreases when (i) the chronological order of the source node’s interaction history is disrupted and (ii) candidate-conditioned state updating is replaced with separate history–candidatefusion.

For (i), following Kim et al. (2025), we adopt ShuffledTrain and ShuffledInference, which permute the source node’s interaction history only during training or evaluation, respectively. We additionally introduce ShuffledBoth, which shuffles histories during both training and evaluation. Table 3 shows lower macro-average MRR for all three variants in both dataset groups. In particular, the degradation under ShuffledBoth, which preserves neighbor identities while disrupting chronological order, shows that chronological order provides predictive information beyond neighbor identity. Degradation under ShuffledInference indicates that MambaTG’s predictions depend on the chronological ordering of historical interactions. Together, these results show that MambaTG leverages historical-sequence evolution rather than treating the history as an unordered set.

Table 3: Macro-average MRR change (pp) relative to full MambaTG under signal ablations.
<table><tr><td>Variant</td><td>Repeated</td><td>Unseen</td></tr><tr><td>ShuffledTrain</td><td>-8.23</td><td>-1.56</td></tr><tr><td>ShuffledInference</td><td>-27.21</td><td>-3.53</td></tr><tr><td>ShuffledBoth</td><td>-4.22</td><td>-1.78</td></tr><tr><td>LateFusion</td><td>-2.88</td><td>-3.98</td></tr></table>

For (ii), we introduce LateFusion, which retains the original history encoder but replaces the candidate-conditioned Mamba continuation with MLP-based fusion of the separately encoded history and candidate representations, a common prediction design in T-GNNs. The resulting degradation indicates that late-stage fusion does not fully recover the benefit of candidate continuation from the evolved historical state. The larger degradation on GoogleLocal and YouTube further suggests that candidate–history compatibility is particularly important for unseen-edge prediction. Overall, these findings support our hypothesis that jointly modeling historical-sequence evolution and candidate–history compatibility yields MambaTG’s strongest performance. Detailed settings and per-dataset results are provided in Appendix D.

## 5.4 ABLATION STUDY

We ablate five components of MambaTG: (1) Historical Spectral Refinement (HSR) and (2) Candidate Feature Refinement (CFR) within DRR, (3) Query-relative time encoding, (4) Candidate activity recency, and (5) Historical repetition frequency. Table 4 reports the macro-average MRR (%) reduction across two repeated-edge-dominant (UCI, MOOC) and two unseen-edge-dominant (GoogleLocal, Flickr) datasets. Variant definitions and perdataset results are provided in Appendix E.

Table 4: Macro-average MRR change (pp) under component ablations. Repetition frequency applies only to repeated-edge-dominant datasets.
<table><tr><td>Removed component</td><td>Repeated</td><td>Unseen</td></tr><tr><td>HSR</td><td>-0.21</td><td>-1.74</td></tr><tr><td>CFR</td><td>-0.57</td><td>-1.28</td></tr><tr><td>Query-relative time encoding</td><td>-2.87</td><td>-1.77</td></tr><tr><td>Candidate activity recency</td><td>-2.35</td><td>-1.03</td></tr><tr><td>Historical repetition freq.</td><td>-5.67</td><td>N/A</td></tr></table>

Overall, all components contribute to MambaTG, but their importance differs across prediction regimes. Query-relative time encoding, candidate activity recency, and historical repetition frequency contribute more strongly to repeated-edge prediction, where the timing and frequency of previous source–candidate interactions provide direct evidence of recurrence. In contrast, HSR and CFR show more significant contributions to unseen-edge prediction, where prediction relies more heavily on patterns extracted from the source history and informative candidate representations.

## 6 CONCLUSION

In this paper, we identified historical-sequence evolution and candidate–history compatibility as complementary sources of predictive information beyond direct pairwise recurrence. We proposed MambaTG, a candidate-aware Mamba-based framework that jointly models these signals to capture how ordered interaction histories evolve and relate to each candidate. Across 13 temporal graph datasets, MambaTG achieves strong performance in repeated-edge-dominant and unseenedge-dominant settings while efficiently handling long histories and large candidate sets.

## AI USE STATEMENT

We used generative AI tools to assist with code implementation, provide writing feedback, identify relevant literature, and provide feedback on experimental methodology. We reviewed the AI-assisted work and take responsibility for the final content.

## REFERENCES

Unai Alvarez-Rodriguez, Federico Battiston, Guilherme Ferraz De Arruda, Yamir Moreno, Matjazˇ Perc, and Vito Latora. Evolutionary dynamics of higher-order interactions in social networks. Nature Human Behaviour, 5(5):586–595, January 2021. ISSN 2397-3374. doi: 10.1038/s41562 -020-01024-1. URL https://www.nature.com/articles/s41562-020-01024-1.

Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hanghang Tong, and Mehrdad Mahdavi. Do We Really Need Complicated Model Architectures For Temporal Networks? 2023. URL https://openreview.net/forum?id=ayPPc0SyLv1.

Nick Craswell. Mean Reciprocal Rank. In Ling Liu and M. Tamer Ozsu (eds.),<sup>¨</sup> Encyclopedia of Database Systems, pp. 1703–1703. Springer US, Boston, MA, 2009. ISBN 978-0-387-35544-3 978-0-387-39940-9. doi: 10.1007/978-0-387-39940-9 488. URL http://link.springer.com/10.1 007/978-0-387-39940-9 488.

Zifeng Ding, Yifeng Li, Yuan He, Antonio Norelli, Jingcheng Wu, Volker Tresp, Michael M. Bronstein, and Yunpu Ma. DyGMamba: Efficiently Modeling Long-Term Temporal Dependency on Continuous-Time Dynamic Graphs with State Space Models. Transactions on Machine Learning Research, February 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=sq5AJvVuha.

Aswathy Divakaran and Anuraj Mohan. Temporal Link Prediction: A Survey. New Generation Computing, 38(1):213–258, March 2020. ISSN 0288-3635, 1882-7055. doi: 10.1007/s00354-0 19-00065-z. URL http://link.springer.com/10.1007/s00354-019-00065-z.

Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. 2024. URL https://openreview.net/forum?id=tEYskw1VY2.

Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, and Reihaneh Rabbany. Temporal Graph Benchmark for Machine Learning on Temporal Graphs. In Advances in Neural Information Processing Systems 36, pp. 2056–2073, New Orleans, Louisiana, USA, 2023. Neural Information Processing Systems Foundation, Inc. (NeurIPS). ISBN 978-1-7138-9911-2. doi: 10.52202/075 280-0099. URL http://www.proceedings.com/075280-0099.html.

Wang-Cheng Kang and Julian McAuley. Self-Attentive Sequential Recommendation. 2018 IEEE International Conference on Data Mining (ICDM), pp. 197–206, November 2018. doi: 10.1109/ ICDM.2018.00035. URL https://ieeexplore.ieee.org/document/8594844/. Conference Name: 2018 IEEE International Conference on Data Mining (ICDM) ISBN: 9781538691595.

Seyed Mehran Kazemi, Rishab Goel, Kshitij Jain, Ivan Kobyzev, Akshay Sethi, Peter Forsyth, and Pascal Poupart. Representation Learning for Dynamic Graphs: A Survey. Journal of Machine Learning Research, 21(70):1–73, 2020. ISSN 1533-7928. URL http://jmlr.org/papers/v21/19-4 47.html.

Sein Kim, Hongseok Kang, Kibum Kim, Jiwan Kim, Donghyun Kim, Minchul Yang, Kwangjin Oh, Julian McAuley, and Chanyoung Park. Lost in Sequence: Do Large Language Models Understand Sequential Recommendation? In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, pp. 1160–1171, New York, NY, USA, August 2025. Association for Computing Machinery. ISBN 979-8-4007-1454-2. doi: 10.1145/3711896.3737035. URL https://dl.acm.org/doi/10.1145/3711896.3737035.

Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1269–1278, Anchorage AK USA, July 2019. ACM. ISBN 978-1-4503-6201-6. doi: 10.1145/3292500.3330895. URL https://dl.acm.org/doi/10.1145/3292500.3330895.

Dongyuan Li, Shiyin Tan, Ying Zhang, Ming Jin, Shirui Pan, Manabu Okumura, and Renhe Jiang. DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs. In Advances in Neural Information Processing Systems 38, pp. 143195–143224, San Diego, California, USA and Mexico City, Mexico, 2025. Neural Information Processing Systems Foundation, Inc. (NeurIPS). ISBN 979-8-3313-3827-5. doi: 10.52202/085713-4299. URL http://www.proceedings.com/085713-4 299.html.

Chengkai Liu, Jianghao Lin, Jianling Wang, Hanzhou Liu, and James Caverlee. Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models, June 2024. URL http://arxiv.org/abs/2403.03900. arXiv:2403.03900 [cs.IR].

Zhiqiang Pan, Fei Cai, Wanyu Chen, Honghui Chen, and Maarten De Rijke. Star Graph Neural Networks for Session-based Recommendation. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pp. 1195–1204, Virtual Event Ireland, October 2020. ACM. ISBN 978-1-4503-6859-9. doi: 10.1145/3340531.3412014. URL https: //dl.acm.org/doi/10.1145/3340531.3412014.

Farimah Poursafaei, Andy Huang, Kellin Pelrine, and Reihaneh Rabbany. Towards Better Evaluation for Dynamic Link Prediction. June 2022. URL https://openreview.net/forum?id=1GVpwr2Tfdg.

Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, UAI ’09, pp. 452–461, Arlington, Virginia, USA, June 2009. AUAI Press. ISBN 978-0-9749039-5-8. URL https://dl.acm.org/doi/10.5555/1795114.1795167.

Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein. Temporal Graph Networks for Deep Learning on Dynamic Graphs, October 2020. URL http://arxiv.org/abs/2006.10637. arXiv:2006.10637 [cs.LG].

H. Sorensen, D. Jones, M. Heideman, and C. Burrus. Real-valued fast Fourier transform algorithms. IEEE Transactions on Acoustics, Speech, and Signal Processing, 35(6):849–863, June 1987. ISSN 0096-3518. doi: 10.1109/TASSP.1987.1165220. URL http://ieeexplore.ieee.org/do cument/1165220/.

Yuxing Tian, Yiyan Qi, and Fan Guo. FreeDyG: Frequency Enhanced Continuous-Time Dynamic Graph Model for Link Prediction. 2024. URL https://openreview.net/forum?id=82Mc5ilInM.

Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. DyRep: Learning Representations over Dynamic Graphs. 2019. URL https://openreview.net/forum?id=HyePrhR5 KX.

Lu Wang, Xiaofu Chang, Shuang Li, Yunfei Chu, Hui Li, Wei Zhang, Xiaofeng He, Le Song, Jingren Zhou, and Hongxia Yang. TCL: Transformer-based Dynamic Graph Modelling via Contrastive Learning. ArXiv, May 2021a. URL https://www.semanticscholar.org/paper/TCL%3A-Transfo rmer-based-Dynamic-Graph-Modelling-via-Wang-Chang/d37a8e9cea2f1e1b45b4912d243eee1 b9dd5f756.

Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, and Pan Li. Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. 2021b. URL https://openreview .net/forum?id=KYPz4YsCPj.

Yuxia Wu, Yuan Fang, and Lizi Liao. On the Feasibility of Simple Transformer for Dynamic Graph Modeling. In Proceedings of the ACM Web Conference 2024, pp. 870–880, Singapore Singapore, May 2024. ACM. ISBN 979-8-4007-0171-9. doi: 10.1145/3589334.3645622. URL https: //dl.acm.org/doi/10.1145/3589334.3645622.

Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. Inductive representation learning on temporal graphs. 2020. URL https://openreview.net/forum?id=rJeW1yHYwH.

Lu Yi, Runlin Lei, Fengran Mo, Yanping Zheng, Zhewei Wei, and Yuhang Ye. Future Link Prediction Without Memory or Aggregation. In Advances in Neural Information Processing Systems 38, pp. 2672–2700, San Diego, California, USA and Mexico City, Mexico, 2025a. Neural Information Processing Systems Foundation, Inc. (NeurIPS). ISBN 979-8-3313-3827-5. doi: 10.52202/085713-0084. URL http://www.proceedings.com/085713-0084.html.

Lu Yi, Jie Peng, Yanping Zheng, Fengran Mo, Zhewei Wei, Yuhang Ye, Yue Zixuan, and Zengfeng Huang. TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics. International Conference on Learning Representations, 2025:88022–88042, May 2025b. URL https://proceedings.iclr.cc/paper files/paper/2025/hash/db5ca61dbc08cf5143c05ad2d1b0b2ca-A bstract-Conference.html.

Le Yu. An Empirical Evaluation of Temporal Graph Benchmark, July 2023. URL https://arxiv.org/ abs/2307.12510v5.

Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. Towards Better Dynamic Graph Learning: New Architecture and Unified Library. In Advances in Neural Information Processing Systems 36, pp. 67686–67700, New Orleans, Louisiana, USA, 2023. Neural Information Processing Systems Foundation, Inc. (NeurIPS). ISBN 978-1-7138-9911-2. doi: 10.52202/075280-2960. URL http://www.proceedings.com/075280-2960.html.

Le Yu, Zihang Liu, Leilei Sun, Bowen Du, Chuanren Liu, and Weifeng Lv. Continuous-Time User Preference Modelling for Temporal Sets Prediction. IEEE Transactions on Knowledge and Data Engineering, 36(4):1475–1488, April 2024. ISSN 1041-4347, 1558-2191, 2326-3865. doi: 10.1109/TKDE.2023.3309982. URL https://ieeexplore.ieee.org/document/10234655/.

## A ABBREVIATIONS

Table 5: Key abbreviations and shorthand.
<table><tr><td>Term</td><td>Meaning</td></tr><tr><td>CAME</td><td>Candidate-Aware Mamba Encoder</td></tr><tr><td>DRR</td><td>Dual-Branch Representation Refinement</td></tr><tr><td>HSR</td><td>Historical Spectral Refinement</td></tr><tr><td>CFR</td><td>Candidate Feature Refinement</td></tr><tr><td>MRR</td><td>Mean Reciprocal Rank</td></tr><tr><td>-R</td><td>Suffix indicating historical repetition frequency</td></tr><tr><td>w/o</td><td>Without the specified component</td></tr><tr><td>pp</td><td>Percentage points</td></tr><tr><td>OOM</td><td>Exceeds the memory budget</td></tr><tr><td>OOT</td><td>Exceeds the time budget</td></tr></table>

## B EXPERIMENTAL DETAILS

## B.1 HARDWARE AND SOFTWARE ENVIRONMENT

All experiments were conducted on NVIDIA RTX PRO 6000 Blackwell GPUs with 96GB of memory. Our implementation uses Python 3.11, PyTorch 2.7.1, and CUDA 12.8, together with mamba-ssm 2.3.1 and causal-conv1d 1.6.2.

## B.2 DATASET DETAILS

Our main evaluation includes 13 future link prediction datasets. Seven are from TGB-Seq (Yi et al., 2025b): ML-20M, Taobao, Yelp, GoogleLocal, Flickr, YouTube, and WikiLink. The remaining six are commonly used temporal graph datasets: Wikipedia, Reddit, MOOC, LastFM, UCI, and Flights. We additionally evaluate MambaTG on three datasets from TGB (Huang et al., 2023): tgbl-comment, tgbl-coin, and tgbl-flight. These additional results are reported in Appendix J.3.

Following Yi et al. (2025a), we classify datasets as unseen-edge-dominant if most target interactions are previously unseen edges, and as repeated-edge-dominant if most are repeated edges. The datasets are obtained from the TGB-Seq collection, the TGB repository, and the public Zenodo archive. Table 6 summarizes their statistics; for bipartite datasets, the two node counts correspond to the source and destination node types.

## B.3 BASELINE DETAILS

We compare MambaTG with established temporal graph models, recent Mamba-based architectures, a sequential recommendation method, and a candidate-aware future link predictor. Their main designs are summarized below:

EdgeBank. EdgeBank (Poursafaei et al., 2022) is a non-parametric baseline that stores previously observed edges and scores candidate links according to whether they have appeared in the interaction history. It captures edge recurrence without learning node representations or model parameters. Given its explicit focus on edge recurrence, we report EdgeBank only on repeated-edge-dominant datasets.

JODIE. JODIE (Kumar et al., 2019) maintains dynamic source and destination embeddings using two coupled recurrent neural networks. After each interaction, it jointly updates the embeddings of both endpoints and projects the source embedding forward to the query time for future interaction prediction.

DyRep. DyRep (Trivedi et al., 2019) models communication and association events as two interacting temporal processes. It updates persistent node representations using temporally attentive neighborhood information and parameterizes event occurrence through a temporal point process.

TGAT. TGAT (Xu et al., 2020) computes time-dependent node representations by applying selfattention over sampled temporal neighborhoods. It uses a functional time encoding to represent the elapsed time of historical interactions and supports inductive inference without maintaining persistent node memories.

TGN. TGN (Rossi et al., 2020) maintains a persistent memory for each node. Incoming interactions generate messages that are aggregated and used to update the endpoint memories, after which a graph-based embedding module combines the memories with temporal-neighborhood information for prediction.

CAWN. CAWN (Wang et al., 2021b) samples causal temporal walks from the two endpoints of a query edge. It anonymizes node identities through walk-based occurrence patterns and encodes the resulting walks to capture temporal motifs and structural relationships relevant to the queried pair.

GraphMixer. GraphMixer (Cong et al., 2023) uses an MLP-Mixer-based link encoder to summarize recent temporal interactions, a mean-pooling node encoder to aggregate node features, and an MLP classifier for link prediction. It provides a memory-free alternative to recurrent and attentionbased temporal graph models.

DyGFormer. DyGFormer (Yu et al., 2023) extracts the first-hop interaction histories of both endpoints and uses neighbor co-occurrence encoding to represent correlations between them. The histories are divided into patches and processed by a Transformer to capture long-range temporal dependencies efficiently.

FreeDyG. FreeDyG (Tian et al., 2024) is a frequency-enhanced continuous-time dynamic graph model for link prediction. It introduces node interaction frequency encoding to explicitly capture the proportion of common neighbors and the interaction frequency of each node pair. By modeling interaction patterns in the frequency domain, FreeDyG aims to capture periodic behavior and temporal shifts in node interactions.

DyGMamba. DyGMamba (Ding et al., 2025) employs two state-space components. A node-level SSM encodes historical node interactions, while a time-level SSM captures temporal patterns and dynamically selects relevant information from the extended interaction history.

DyG-Mamba. DyG-Mamba (Li et al., 2025) formulates dynamic graph learning using a continuous state-space architecture inspired by Mamba. It uses irregular time intervals as control signals for the state transitions, allowing the model to adjust how historical information is retained or forgotten.

SGNN-HN. SGNN-HN (Pan et al., 2020) was originally proposed for session-based recommendation. It represents an ordered interaction sequence as a star graph, allowing the star node to exchange information with all items in the sequence. A highway network combines item representations before and after graph propagation, and an attention mechanism aggregates them into a sequence representation for candidate prediction.

SASRec. SASRec (Kang & McAuley, 2018) is a self-attention-based sequential recommendation model that encodes a user’s ordered item-interaction history to predict the next item, capturing dependencies among past interactions.

Mamba4Rec. Mamba4Rec (Liu et al., 2024) is a Mamba-based sequential recommendation model that encodes a user’s ordered item-interaction history for next-item prediction through selective state-space updates.

CRAFT. CRAFT (Yi et al., 2025a) represents nodes using trainable identifier embeddings and applies cross-attention from each candidate destination to the source node’s recent interactions. It therefore performs direct candidate-to-history matching without maintaining persistent node memories or aggregating the candidate’s temporal neighborhood. Following its original design, the repetition-enhanced CRAFT-R variant additionally supplies historical repetition frequency to the predictor on repeated-edge-dominant datasets.

Table 6: Statistics of the 16 datasets used in our experiments. Regime indicates whether a dataset is unseen-edge-dominant (Unseen) or repeated-edge-dominant (Repeated).
<table><tr><td>Dataset</td><td>Nodes</td><td>Edges</td><td>Timestamps</td><td>Repeat Ratio (%)</td><td>Density (%)</td><td>Bipartite</td><td>Domain</td><td>Regime</td></tr><tr><td>ML-20M</td><td>100,785/9,646</td><td>14,494,325</td><td>9,993,250</td><td>0</td><td> $1 . 4 9 \times 1 0 ^ { 0 }$ </td><td>Yes</td><td>Movie rating</td><td>Unseen</td></tr><tr><td>Taobao</td><td>760,617/863,016</td><td>18,853,792</td><td>139,171</td><td>16.58</td><td> $2 . 8 7 \times 1 0 ^ { - 3 }$ </td><td>Yes</td><td>E-commerce interaction</td><td>Unseen</td></tr><tr><td>Yelp</td><td>1,338,688/405,081</td><td>19,760,293</td><td>14,646,734</td><td>25.18</td><td> $3 . 6 4 \times 1 0 ^ { - 3 }$ </td><td>Yes</td><td>Business review</td><td>Unseen</td></tr><tr><td>GoogleLocal</td><td>206,244/267,336</td><td>1,913,967</td><td>1,771,060</td><td>0</td><td> $3 . 4 7 \times 1 0 ^ { - 3 }$ </td><td>Yes</td><td>Business review</td><td>Unseen</td></tr><tr><td>Flickr</td><td>233,836</td><td>7,223,559</td><td>134</td><td>0</td><td> $1 . 3 2 \times 1 0 ^ { - 2 }$ </td><td>No</td><td>Who-to-follow</td><td>Unseen</td></tr><tr><td>YouTube</td><td>402,422</td><td>3,288,028</td><td>203</td><td>0</td><td> $2 . 0 3 \times 1 0 ^ { - 3 }$ </td><td>No</td><td>Who-to-follow</td><td>Unseen</td></tr><tr><td>WikiLink</td><td>1,361,972</td><td>34,163,774</td><td>2,198</td><td>0</td><td> $1 . 8 4 \times 1 0 ^ { - 3 }$ </td><td>No</td><td>Web link</td><td>Unseen</td></tr><tr><td>Wikipedia</td><td>8,227/1,000</td><td>157,474</td><td>152,757</td><td>88.41</td><td> $1 . 9 1 \times 1 0 ^ { 0 }$ </td><td>Yes</td><td>Interaction</td><td>Repeated</td></tr><tr><td>Reddit</td><td>10,000/984</td><td>672,447</td><td>669,065</td><td>88.32</td><td> $6 . 8 3 \times 1 0 ^ { 0 }$ </td><td>Yes</td><td>Social</td><td>Repeated</td></tr><tr><td>MOOC</td><td>7,047/97</td><td>411,749</td><td>345,600</td><td>56.66</td><td> $6 . 0 2 \times 1 0 ^ { 1 }$ </td><td>Yes</td><td>Interaction</td><td>Repeated</td></tr><tr><td>LastFM</td><td>980/1,000</td><td>1,293,103</td><td>1,283,614</td><td>88.01</td><td> $1 . 3 2 \times 1 0 ^ { 2 }$ </td><td>Yes</td><td>Interaction</td><td>Repeated</td></tr><tr><td>UCI</td><td>1,899</td><td>59,835</td><td>58,911</td><td>66.06</td><td> $1 . 6 6 \times 1 0 ^ { 0 }$ </td><td>No</td><td>Social</td><td>Repeated</td></tr><tr><td>Flights</td><td>13,169</td><td>1,927,145</td><td>122</td><td>79.50</td><td>1.11 × 100</td><td>No</td><td>Transport</td><td>Repeated</td></tr><tr><td>tgbl-comment</td><td>994,790</td><td>44,314,507</td><td>30,998,030</td><td>19.81</td><td> $4 . 4 8 \times 1 0 ^ { - 3 }$ </td><td>No</td><td>Social</td><td>Unseen</td></tr><tr><td>tgbl-coin</td><td>638,486</td><td>22,809,486</td><td>1,295,720</td><td>82.93</td><td> $5 . 6 0 \times 1 0 ^ { - 3 }$ </td><td>No</td><td>Transaction</td><td>Repeated</td></tr><tr><td>tgbl-flight</td><td>18,143</td><td>67,169,570</td><td>1,385</td><td>96.48</td><td> $2 . 0 4 \times 1 0 ^ { 1 }$ </td><td>No</td><td>Transport</td><td>Repeated</td></tr></table>

## B.4 EVALUATION AND IMPLEMENTATION DETAILS

Following TGB-Seq (Yi et al., 2025b) and DyGLib (Yu et al., 2023), we use the Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ and a batch size of 200, which is increased to 400 for large datasets. Models are trained for up to 100 epochs with early stopping using a patience of 20 epochs. We impose a computational budget of 1.5 hours per epoch and 32 GB of GPU memory; exceeding these limits is denoted as out of time (OOT) and out of memory (OOM), respectively. We use official negative samples whenever available. TGB (Huang et al., 2023) provides either 20 or 100 negative destinations per positive edge, depending on the dataset, while TGB-Seq provides 100 randomly sampled negative destinations. For datasets without official negative samples, we follow TGB-Seq and randomly sample 100 negative destinations per positive edge. For small datasets, we additionally follow CRAFT (Yi et al., 2025a) and perform collision checks to exclude positive edges from the sampled negatives. To ensure fair comparison, we use published baseline results when they follow the same evaluation protocol and closely matched training settings; otherwise, we reproduce the methods using our unified pipeline.

## B.5 HYPERPARAMETER SETTINGS

Baseline configurations. For each baseline we reproduce, we use the best configuration reported for the corresponding baseline–dataset pair in the original paper or benchmark study, including TGB-Seq (Yi et al., 2025b), CRAFT (Yi et al., 2025a), or DyGLib (Yu et al., 2023). When no dataset-specific configuration is reported, we use the default configuration recommended by the baseline’s official implementation.

MambaTG configurations. We select MambaTG hyperparameters according to validation MRR. Specifically, we tune the maximum history length $L _ { \mathrm { m a x } } .$ , number of CAME layers R, hidden-layer dropout rate $p _ { \mathrm { h } }$ , and node-embedding dropout rate $p _ { \mathrm { e } }$ using the search space in Table 7. Across all datasets, we fix the Mamba state dimension to $d _ { \mathrm { s t a t e } } = 3 2$ , the expansion factor to 2, and the causal-convolution width to 4. Table 8 reports the selected configurations, where $d _ { e }$ denotes the node-embedding dimension and B the training batch size. Following Yi et al. $( 2 0 2 5 \mathrm { a } )$ , we adopt the same dataset-specific settings for the node-embedding dimension $d _ { e }$ and training batch size B to ensure a fair comparison.

Table 7: Hyperparameter search space for MambaTG.
<table><tr><td>Hyperparameter</td><td>Search space</td></tr><tr><td>Maximum history length,  $L _ { \mathrm { m a x } }$ </td><td>{30, 60, 90, 120, 250}</td></tr><tr><td>Number of CAME layers, R</td><td>{1, 2}</td></tr><tr><td>Hidden-layer dropout,  $p _ { \mathrm { h } }$ </td><td>{0.1,0.2}</td></tr><tr><td>Node-embedding dropout,  $p _ { \mathrm { e } }$ </td><td>{0.1,0.2}</td></tr></table>

Table 8: Selected MambaTG hyperparameters for each dataset.
<table><tr><td>Dataset</td><td> $\pmb { L _ { \mathrm { m a x } } }$ </td><td> $\pmb { R }$ </td><td> ${ \pmb p } _ { \mathbf { h } }$ </td><td> $\scriptstyle { p _ { \mathrm { e } } }$ </td><td> $d _ { e }$ </td><td> $\textbf {  { B } }$ </td></tr><tr><td>UCI Wikipedia</td><td>30 90</td><td>1 2</td><td>0.1 0.2</td><td>0.1 0.2</td><td>64 64</td><td>200 200</td></tr><tr><td>Reddit MOOC Flights LastFM</td><td>120 30 250 250</td><td>1 2 1 2</td><td>0.1 0.1 0.1 0.1</td><td>0.1 0.1 0.1 0.1</td><td>64 64 64 64</td><td>200 200 200 200</td></tr><tr><td>GoogleLocal Flickr YouTube Taobao</td><td>250 90 120 60</td><td>2 1 2 2</td><td>0.1 0.1 0.1 0.1</td><td>0.2 0.1 0.1 0.1</td><td>128 128 128 128</td><td>200 400 400 400</td></tr><tr><td>Yelp ML-20M</td><td>60 60</td><td>2 1</td><td>0.1 0.1</td><td>0.1 0.1</td><td>128 128</td><td>400 400</td></tr><tr><td>WikiLink</td><td>250</td><td>1</td><td>0.1</td><td>0.1</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>128</td><td>400</td></tr><tr><td>tgbl-comment</td><td>30</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>1</td><td>0.1</td><td>0.1</td><td>128</td><td>200</td></tr><tr><td>tgbl-coin</td><td>250</td><td>1</td><td>0.1</td><td>0.1</td><td></td><td></td></tr><tr><td>tgbl-flight</td><td>90</td><td>1</td><td>0.1</td><td>0.1</td><td>128 128</td><td>200 200</td></tr></table>

## C METHOD DETAILS

## C.1 MAMBA BLOCK AND FEED-FORWARD NETWORK

Mamba block. Each Mamba block takes an input sequence $\mathbf { X } \in \mathbb { R } ^ { m \times d }$ . Following Gu & Dao (2024), it applies an input projection and a causal one-dimensional convolution, processes the resulting sequence with a selective SSM, and gates the SSM output:

$$
\begin{array} { r l } & { \mathbf { U } = \mathrm { S i L U } ( \mathrm { C o n v 1 D } ( \mathrm { L i n e a r _ { i n } } ( \mathbf { X } ) ) ) , \qquad \mathbf { Y } = \mathrm { S S M } _ { \overline { { \mathbf { A } } } , \overline { { \mathbf { B } } } , \mathbf { C } } ( \mathbf { U } ) , } \\ & { \mathrm { M a m b a B l o c k } ( \mathbf { X } ) = \mathrm { L i n e a r _ { o u t } } ( \mathbf { Y } \odot \mathrm { S i L U } ( \mathrm { L i n e a r _ { g a t e } } ( \mathbf { X } ) ) ) . } \end{array}\tag{8}
$$

The selective SSM processes U recurrently. At position $t ,$ it maps $\mathbf { u } _ { t }$ to $\mathbf { y } _ { t }$ through the hidden state $\mathbf { h } _ { t } \mathbf { : }$

$$
\mathbf { h } _ { t } = \overline { { \mathbf { A } } } _ { t } \mathbf { h } _ { t - 1 } + \overline { { \mathbf { B } } } _ { t } \mathbf { u } _ { t } , \qquad \mathbf { y } _ { t } = \mathbf { C } _ { t } \mathbf { h } _ { t } .\tag{9}
$$

Here, $\overline { { \mathbf { A } } } _ { t }$ controls how the preceding state is retained, $\overline { { \mathbf { B } } } _ { t }$ determines how the current input updates the state, and $\mathbf { C } _ { t }$ maps the hidden state to the output. Mamba derives $\mathbf { B } _ { t } , \mathbf { C } _ { t }$ , and the discretization step $\Delta _ { \mathrm { { \ell } } }$ <sub>t</sub> from the current input:

$$
{ \bf B } _ { t } = { \bf u } _ { t } { \bf W } _ { B } , \qquad { \bf C } _ { t } = { \bf u } _ { t } { \bf W } _ { C } , \qquad { \bf A } _ { t } = \mathrm { s o f t p l u s } \big ( \mathrm { B r o a d c a s t } _ { d _ { \mathrm { i n n e r } } } \left( { \bf u } _ { t } { \bf W } _ { \Delta } \right) + { \bf b } _ { \Delta } \big ) .\tag{10}
$$

Under zero-order-hold discretization, the shared continuous state-transition parameter A and inputdependent parameter $\mathbf { B } _ { t }$ are converted into their discrete forms:

$$
\begin{array} { r } { \overline { { \mathbf { A } } } _ { t } = \exp ( \Delta _ { t } \mathbf { A } ) , \qquad \overline { { \mathbf { B } } } _ { t } = ( \Delta _ { t } \mathbf { A } ) ^ { - 1 } \left[ \exp ( \Delta _ { t } \mathbf { A } ) - \mathbf { I } \right] \Delta _ { t } \mathbf { B } _ { t } . } \end{array}\tag{11}
$$

Feed-forward network. Each Mamba block is followed by a position-wise FFN:

$$
\mathrm { F F N } ( \mathbf { X } ) = \mathrm { G E L U } ( \mathbf { X } \mathbf { W } _ { 1 } + \mathbf { b } _ { 1 } ) \mathbf { W } _ { 2 } + \mathbf { b } _ { 2 } ,\tag{12}
$$

where $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { d \times 4 d }$ and $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { 4 d \times d }$

## C.2 CAME IMPLEMENTATION.

At a given query time, CAME encodes each source history once and retains its layer-wise terminal recurrent caches $K _ { H } ^ { ( r ) }$ for candidate continuation during both training and inference. Each candidate then continues independently from the same terminal history caches, so candidates share the encoded history without affecting one another.

For training, however, the terminal SSM state returned by the standard mamba-ssm selective-scan interface does not support gradient propagation. To allow the BPR objective to propagate gradients through candidate continuation to the history encoder, we extend each history scan from length $L _ { \mathrm { m a x } } \ \mathrm { t o } \ L _ { \mathrm { m a x } } + d _ { \mathrm { s t a t e } } .$ , and append $d _ { \mathrm { s t a t e } }$ auxiliary positions to expose the terminal SSM state through the scan outputs. For each auxiliary position $t = L _ { \mathrm { m a x } } + j$ , we set $\mathbf { u } _ { t } = \mathbf { 0 } , \mathbf { B } _ { t } = \mathbf { 0 }$ and $\bar { \Delta } _ { t } = \Delta _ { \mathrm { o f f } } \mathbf { 1 }$ , where $\Delta _ { \mathrm { o f f } }$ is chosen such that $\tilde { \Delta } _ { t } + \mathbf { b } _ { \Delta }$ is componentwise sufficiently negative. Thus, $\Delta _ { t } =$ softplus $\left( \widetilde { \Delta } _ { t } + \mathbf { b } _ { \Delta } \right) \approx \mathbf { 0 }$ , giving $\overline { { \mathbf { A } } } _ { t } = \exp ( \Delta _ { t } \mathbf { A } )$ ≈ I and $\overline { { \mathbf { B } } } _ { t } \mathbf { u } _ { t } = \mathbf { 0 }$ , and hence $\mathbf { h } _ { L _ { \operatorname* { m a x } } + j } \approx \mathbf { h } _ { L _ { \operatorname* { m a x } } }$ . Although the softplus parameterization keeps $\Delta _ { t } > 0$ , the sufficiently negative $\Delta _ { \mathrm { o f f } }$ makes the resulting state decay numerically negligible. We then read out the terminal state from the auxiliary outputs by setting $\mathbf { C } _ { L _ { \operatorname* { m a x } } + j } = \mathbf { e } _ { j } ^ { \intercal }$ at auxiliary position $t = L _ { \mathrm { m a x } } + j$ , where $\mathbf { e } _ { j } \in \mathbb { R } ^ { d _ { \mathrm { s t a t e } } }$ is the j-th standard basis vector. Since $\mathbf { u } _ { L _ { \mathrm { m a x } } + j } = \mathbf { 0 }$ and $\mathbf { h } _ { L _ { \operatorname* { m a x } } + j } \approx \mathbf { h } _ { L _ { \operatorname* { m a x } } } \colon$

$$
\mathbf { y } _ { L _ { \operatorname* { m a x } } + j } = \mathbf { C } _ { L _ { \operatorname* { m a x } } + j } \mathbf { h } _ { L _ { \operatorname* { m a x } } + j } \approx \mathbf { e } _ { j } ^ { \top } \mathbf { h } _ { L _ { \operatorname* { m a x } } } .\tag{13}
$$

The $d _ { \mathrm { s t a t e } }$ auxiliary outputs therefore jointly recover the terminal SSM state through the standard selective-scan interface, without modifying its CUDA kernel. The SiLU gate is applied only to the original history outputs $\mathbf { y } _ { 1 : L _ { \operatorname* { m a x } } }$ , while the auxiliary outputs are used directly for state readout.

For batched histories with different true lengths, each history is right-padded to $L _ { \mathrm { m a x } } .$ . At padded positions, we set $\mathbf { u } _ { t } = \mathbf { 0 } , \mathbf { B } _ { t } = \mathbf { 0 }$ , and $\tilde { \Delta } _ { t } = \Delta _ { \mathrm { o f f } } \mathbf { 1 }$ , following the approximate-identity construction above. Consequently, the scan can operate on the common length $L _ { \mathrm { m a x } }$ without introducing state decay after the last valid interaction. The terminal causal-convolution state is gathered from the last valid positions of each history rather than from the padded positions.

## D ADDITIONAL DETAILS AND RESULTS FOR SIGNAL ABLATIONS

We provide the experimental settings for ShuffledTrain, ShuffledInference, ShuffledBoth, and Late-Fusion, which examine historical-sequence evolution and candidate–history compatibility. Unless otherwise stated, all experiments follow the same data splits, historical sampling strategy, negative samples, model configuration, optimization settings, and evaluation protocol as MambaTG.

## D.1 SHUFFLEDTRAIN

ShuffledTrain examines how disrupting chronological order during training affects predictive performance. We randomly permute the source node’s historical interactions for every training query. Each permutation is generated once before training and remains fixed across epochs, preventing repeated shuffling from acting as stochastic data augmentation. All attributes associated with an interaction, including its timestamp, are permuted together, while the candidate remains the final input token. The model is trained from scratch on these shuffled histories and evaluated using the original chronologically ordered test histories. This variant probes the role of learning from ordered interactions in capturing historical-sequence evolution.

## D.2 SHUFFLEDINFERENCE

ShuffledInference examines whether the trained model relies on chronological ordering when making predictions. We train MambaTG normally and randomly permute the source histories only during test-time inference, without retraining or modifying its parameters. As in ShuffledTrain, all attributes associated with each interaction are permuted together, and the candidate remains the final token. A performance reduction relative to standard chronological inference indicates that the model’s predictions are sensitive to historical order.

## D.3 SHUFFLEDBOTH

ShuffledBoth examines the benefit of chronological ordering under matched training and evaluation ordering conditions. Since ShuffledTrain and ShuffledInference use different ordering conditions during training and evaluation, their performance changes may also reflect sensitivity to this mismatch. To address this, ShuffledBoth uses shuffled histories during both training and evaluation.

The model is trained from scratch following the ShuffledTrain procedure and evaluated on shuffled test histories. This comparison tests whether the chronological order of a source node’s interaction history provides predictive information for future link prediction beyond historical-neighbor identities.

## D.4 LATEFUSION

LateFusion examines whether candidate–history compatibility is better captured by incorporating the candidate into the state transition rather than combining independently encoded representations at the prediction stage. In MambaTG, the complete history encoder processes the historical sequence $H _ { u } ( t )$ and retains the layer-wise history caches. The candidate then continues through all R CAME layers using these caches, following Equation (6):

$$
\begin{array} { r l } & { ( { \bf h } _ { H } , { \cal K } _ { H } ) = \mathrm { H i s t o r y E n c o d e r } ( H _ { u } ( t ) ) , } \\ & { \quad \quad \quad { \bf h } _ { c } = { \mathrm { C a n d i d a t e C o n t i n u a t i o n } } \big ( { \bf z } _ { H , c } ^ { \mathrm { C F R } } ; { \cal K } _ { H } \big ) , } \\ & { \quad \quad \quad \hat { y } _ { u , c } = \mathrm { M L P } ( { \bf h } _ { c } ) . } \end{array}\tag{14}
$$

Here, h<sub>H</sub> denotes the final history representation, $\kappa _ { H }$ contains the recurrent and convolutional states retained at each layer, and ${ \bf z } _ { H , c } ^ { \mathrm { C F R } }$ denotes the refined candidate representation.

LateFusion removes this candidate continuation while keeping the complete chronological history encoder unchanged. Instead, it independently refines the candidate with a residual MLP and combines it with the history representation only at the prediction stage:

$$
\begin{array} { r l } & { { \bf { h } } _ { H } = \mathrm { { H i s t o r y E n c o d e r } } \left( { H _ { u } ( t ) } \right) , } \\ & { { \bf { \widetilde { z } } } _ { H , c } ^ { \mathrm { C F R } } = \mathrm { { R e s M L P } } \left( { \bf { z } } _ { H , c } ^ { \mathrm { C F R } } \right) , } \\ & { \quad { \hat { y } } _ { u , c } = \mathrm { { M L P } } \left( { \left[ { { \bf { h } } _ { H } } \right] \left. { \bf { \widetilde { z } } } _ { H , c } ^ { \mathrm { C F R } } \right.}  \right) . } \end{array}\tag{15}
$$

For LateFusion, only the history representation is used at the prediction stage, and the history caches are not used for candidate processing. The residual MLP is sized to approximately match the parameter count of one CAME layer. MambaTG reuses its history-encoder parameters for candidate continuation, while LateFusion adds a separate residual MLP and uses a wider prediction head. As a result, LateFusion has more total trainable parameters. Thus, the comparison evaluates candidateconditioned continuation against late fusion with a larger parameter budget, ensuring that any ad vantage of MambaTG cannot be attributed simply to a higher total parameter count. LateFusion is trained from scratch using the same training and evaluation settings as MambaTG. The explicit features used by the MambaTG link classifier are retained but omitted from both equations for clarity.

## D.5 PER-DATASET RESULTS

Figure 5 reports the per-dataset results for all four signal ablations.

## E ABLATION STUDY DETAILS

We conduct an ablation study on four datasets to evaluate the contributions of MambaTG’s key com ponents. Specifically, we consider five variants: (1) w/o HSR, which removes the Historical Spectral Refinement (HSR) applied to the source node’s historical interaction sequence; (2) w/o CFR, which removes Candidate Feature Refinement (CFR), preventing the candidate representations from be ing adaptively aligned with the refined historical representation space; (3) w/o query-rel, which removes the query-relative time encoding from the input representations; (4) w/o recency, which removes the candidate activity recency feature concatenated at the prediction layer; and (5) w/o repeat, which removes the candidate repetition-frequency feature from the prediction layer. The repetition-frequency feature is evaluated only on repeated-edge datasets. Figure 6 presents the complete per-dataset results.

![](images/5c26513e9a1ffbcf796d7d9885d2d38c944f4ddaec085279562c3b342141a89d.jpg)  
Figure 5: Per-dataset signal-ablation results on the repeated-edge-dominant datasets (MOOC and LastFM) and the unseen-edge-dominant datasets (GoogleLocal and YouTube). Results are reported as MRR (%), averaged over three random seeds.

![](images/dd1c9d142f20ac34589c4f77d00ab94d94bed7310eaaa3b80017d730eb95382a.jpg)  
Figure 6: Per-dataset component-ablation results on the repeated-edge-dominant datasets (UCI and MOOC) and the unseen-edge-dominant datasets (GoogleLocal and Flickr). MambaTG-R denotes MambaTG augmented with historical repetition frequency. Each ablation variant removes one component from the corresponding complete model. Results are reported as MRR (%), averaged over three random seeds.

## F EFFECTIVENESS AND TRANSFERABILITY OF DUAL-BRANCH REPRESENTATION REFINEMENT (DRR)

DRR improves the inputs to candidate-aware temporal modeling through two complementary branches: HSR refines historical patterns at different temporal scales, while CFR recalibrates candidate features to promote compatibility with the refined history. To examine whether its benefits extend beyond MambaTG, we incorporate DRR into CRAFT and compare the resulting CRAFT-DRR with the original CRAFT. Specifically, we construct CRAFT-DRR by applying HSR to the history embeddings after positional encoding and CFR to the candidate embeddings, with the resulting representations passed to CRAFT’s cross-attention module. We also compare MambaTG with MambaTG without DRR to isolate the contribution of DRR within MambaTG. We conduct these comparisons on UCI and GoogleLocal, representing repeated- and unseen-edge-dominant settings, respectively. Within each framework, the variants with and without DRR use the same model configuration and training settings, differing only in the inclusion of DRR. On UCI, all variants additionally use historical repetition frequency and are denoted by the suffix “-R.” The CRAFT results in this analysis are obtained from our own experiments and may therefore differ from the originally reported results presented in Table 2.

Results. Figure 7 shows that DRR improves both CRAFT and MambaTG on both datasets, indi cating that its benefits are not specific to either architecture. Comparing the two frameworks without DRR, MambaTG consistently outperforms CRAFT, showing that its candidate-aware sequential modeling provides an advantage independently of DRR. This advantage is retained when both frameworks use DRR, with MambaTG still outperforming CRAFT-DRR on both datasets. Thus, the gains from MambaTG cannot be attributed solely to representation refinement. Instead, the results support complementary contributions from DRR and MambaTG’s joint modeling of historicalsequence evolution and candidate–history compatibility. The consistent improvement of CRAFT with DRR further suggests that the refined representations are independently useful for candidate– history compatibility, particularly when predicting unseen edges where the source and candidate have no previous direct interaction.

![](images/852c34a90266cb3618cfc0c71167901f04c3ffc2d9f708576cef0e4fe48628de.jpg)

![](images/1a6f1bf94ba5e21fb9fc1b35e11b0c795e9109f570b4f6a596730fa2c603eca2.jpg)  
Figure 7: Impact of DRR on CRAFT and MambaTG on UCI (repeated-edge-dominant) and Google-Local (unseen-edge-dominant) datasets. Results are reported as MRR (%), averaged over three random seeds.

## G EFFECTIVENESS OF CANDIDATE-AWARE MAMBA ENCODER (CAME)

CAME jointly models historical-sequence evolution and candidate–history compatibility by treating each candidate as a continuation of the source node’s interaction sequence. To assess the role of Mamba-based sequence modeling within CAME, we compare MambaTG with GRUTG. GRUTG replaces each Mamba block with a Gated Recurrent Unit (GRU) of the same hidden dimension while retaining all other model components, training settings, and the candidate-continuation procedure. We evaluate both models on UCI and GoogleLocal, representing repeated-edge-dominant and unseen-edge-dominant settings, respectively. On UCI, both models additionally use historical repetition frequency and are denoted as GRUTG-R and MambaTG-R.

Results. As shown in Figure 8, MambaTG consistently outperforms GRUTG on both datasets, indicating that Mamba provides a more effective sequential backbone for CAME. This advantage may stem from Mamba’s structured SSM state and selective recurrence, which capture different aspects of the interaction history over different timescales and selectively retain informative events. The larger gain on the unseen-edge-dominant GoogleLocal dataset further suggests that Mamba is particularly useful when direct source–candidate evidence is limited and prediction relies more heavily on patterns in the source history. Overall, these results support the use of Mamba as the sequential encoder for jointly modeling historical-sequence evolution and candidate–history compatibility in CAME.

## H COMPARISON WITH RECOMMENDATION MODELS

MambaTG targets future link prediction on temporal graphs, a general task arising in diverse applications such as recommendation, social networks, communication systems, and transportation networks. It supports both bipartite and non-bipartite graphs and covers both repeated and unseen future edges. Recommendation can be viewed as a specific application of this task, where temporal interactions typically occur between users and items in a bipartite graph.

To further examine the effectiveness of MambaTG, we compare it with two representative sequential recommendation models: the Transformer-based SASRec (Kang & McAuley, 2018) and the

![](images/f80f9bf512fe53b8805e33415405ce3781c7b137c64624e8002f759c48ce0424.jpg)

![](images/eb3312c38cd771480a3f08b5dd96d7431fe5cb46a001295da31172612b3a3209.jpg)  
Figure 8: Effect of CAME on the repeated-edge-dominant UCI dataset and the unseen-edgedominant GoogleLocal dataset. MambaTG is compared with GRUTG. Results are reported as MRR (%), averaged over three random seeds.

Mamba-based Mamba4Rec (Liu et al., 2024). The comparison covers MOOC and GoogleLocal, representing repeated-edge-dominant and unseen-edge-dominant settings, respectively. As shown in Figure 9, MambaTG or MambaTG-R consistently outperforms both recommendation models across both datasets. These results suggest that models designed specifically for sequential user–item recommendation may not fully capture the broader characteristics of future link prediction, particularly the distinction between repeated and unseen edges.

![](images/f91afb77029fdf4dfa3ef890e7bee69814278c753fca06e39e5ebb989975ed03.jpg)

![](images/1563761dd0efd653ae8cc97ac35d8cbe176c584b809194bc8fb17eb86ce7db29.jpg)  
Figure 9: Comparison with sequential recommendation models on MOOC (repeated-edgedominant) and GoogleLocal (unseen-edge-dominant) datasets. Results are reported as MRR (%), averaged over three random seeds.

## I TIME COMPLEXITY ANALYSIS

We analyze a batch of B source nodes, each evaluated against q candidate destinations. Let ℓ denote the number of historical interactions per node, d the feature dimension, and <sup>¯</sup>d the average node degree. We report only dominant terms and treat architectural hyperparameters, including the number of layers and attention heads, as well as the Mamba expansion factors and state dimensions, as constants. For TGAT, however, we explicitly account for its two-layer neighborhood expansion, which introduces two-hop histories and changes the dependence on ℓ.

Historical-neighbor extraction. We use neighbor extraction to refer to the temporal-index lookup operations required to locate the relevant histories in node adjacency lists. Following the common implementation in DyGLib (Yu et al., 2023), retrieving the most recent historical interactions before the query time requires O(log <sup>¯</sup>d) per node. TGN (Rossi et al., 2020), GraphMixer (Cong et al., 2023), DyGFormer, DyG-Mamba (Li et al., 2025), and DyGMamba (Ding et al., 2025) retrieve histories for both the source and all candidate destinations, resulting in $O ( B ( 1 + q ) \log \bar { d } )$ . TGAT (Xu et al., 2020) additionally performs temporal-index lookups for up to ℓ first-hop neighbors in its twolayer architecture. Thus, it performs $\bar { O ( } B ( 1 + q ) ( 1 + \ell ) \bar { ) }$ lookup operations, each costing $O ( \log \bar { d } )$ resulting in a neighbor-extraction cost of $\dot { O } ( B ( 1 + q ) ( \bar { 1 } + \ell ) \log \bar { d } )$ . In contrast, CRAFT (Yi et al., 2025a) and MambaTG retrieve only the source history, reducing the cost to $O ( B \log \bar { d } )$

Edge-likelihood computation. TGN and GraphMixer process the ℓ historical interactions of both the source and every candidate destination, resulting in ${ \bf \bar { \Lambda } } O ( B ( 1 + q ) \ell d ^ { 2 } )$ TGAT aggregates twohop histories with two graph-attention layers, where ℓ is the sampling size at each hop, requiring $O \dot { ( } B ( 1 + q ) \ell ^ { 2 } d ^ { 2 } )$ . DyGFormer applies feature projection and self-attention to each endpoint, with costs of $\bar { O ( \ell d ^ { 2 } ) }$ and $\operatorname { \dot { O } } ( \ell ^ { 2 } d )$ , respectively. Its total complexity is therefore $O ( B ( 1 \stackrel { . } { + } q ) ( \ell d ^ { 2 } +$ $\ell ^ { 2 } d ) )$ . DyG-Mamba and DyGMamba process the full history for every source–candidate pair using state-space models, resulting in $O ( B \dot { q } \ell d ^ { 2 } )$ . CRAFT projects the source history and candidates in $O ( B ( \bar { \ell } + q ) d ^ { 2 } )$ , followed by candidate-to-history attention with complexity $O ( B q \ell d )$ . Its total cost is thus $O ( \bar { B } ( \ell + q ) d ^ { 2 } + B q \bar { \ell } d )$ . MambaTG incurs $O ( B \ell d \log \ell )$ time for DRR, whose dominant cost arises from frequency-domain refinement of each source history. It then encodes each history once in $O ( B \ell d ^ { 2 } )$ and evaluates all candidates using the cached states in $O ( B q d ^ { 2 } )$ . Therefore, its total edge-likelihood computation is $O ( B \ell d \log \ell + \bar { B } ( \ell + q ) d ^ { 2 } )$

## J ADDITIONAL EXPERIMENTAL RESULTS

## J.1 LONG SEQUENCE EFFICIENCY

![](images/f816f76d0a8b52d8c46d5cd31b285f339f81a50c17ee7522c8b8219b5e14ccff.jpg)  
Figure 10: Computational efficiency on LastFM Dataset as the maximum source-history length ℓ increases. Training time per epoch, inference time, and peak GPU memory usage.

We evaluate the scalability of MambaTG as the source-node history length increases on LastFM, measuring per-epoch training time, inference time, and peak GPU memory usage. Training time measures the wall-clock time of one training epoch, excluding validation and data loading before the epoch. Inference time measures the wall-clock time of the 100-negative test pass, including the operations required for each method to produce predictions. Peak GPU memory is measured as the maximum allocated GPU memory over the full run, including training and evaluation.

Results are shown in Figure 10. (1) Training efficiency. MambaTG scales efficiently with increasing history length and remains among the faster methods. At ℓ = 750, it achieves up to 12.91× speedup over DyG-Mamba while remaining competitive with CRAFT and JODIE. (2) Inference efficiency. Although inference time scales near-linearly with ℓ in theory, it remains nearly constant empirically because candidate evaluation dominates wall-clock time. MambaTG caches historical states and reuses them across candidates, avoiding repeated history encoding. $\mathrm { A t } \ell = 1 0 0 0$ , it achieves up to 104.54× speedup over the compared baselines. (3) Memory efficiency. MambaTG exhibits approximately linear memory growth with history length and remains substantially more memory-efficient than several baselines. Although JODIE becomes more memory-efficient for longer histories due to its fixed-size node states, MambaTG maintains a competitive memory footprint across the evaluated sequence lengths. Overall, these results demonstrate that MambaTG can effectively scale to long source histories while maintaining low computational overhead, making it well suited to large-scale recommendation scenarios where a user’s extensive interaction history must be evaluated against many candidate items.

![](images/830a13b226d0f77a2a43c1f82783fb994c04012db25b976c900cead01cf4694b.jpg)  
(a)

![](images/f51b997903a895b49749d89e655f347914d010cba6042c2a075cdd8e177535b1.jpg)  
(b)  
Figure 11: Performance–efficiency trade-off on (a) LastFM and (b) UCI. Test MRR is plotted against per-epoch training time, with bubble size representing parameter footprint.

## J.2 PERFORMANCE–EFFICIENCY TRADE-OFF

We additionally evaluate the performance–efficiency trade-off on LastFM and UCI. Figure 11 plots test MRR against per-epoch training time, with bubble size representing parameter footprint.

## J.3 ADDITIONAL RESULTS ON OTHER DATASETS

We further evaluate MambaTG on three datasets from the Temporal Graph Benchmark (TGB) (Huang et al., 2023): tgbl-comment, tgbl-coin, and tgbl-flight. We compare against DyGFormer (Yu et al., 2023), SGNN-HN (Pan et al., 2020), and CRAFT (Yi et al., 2025a). The DyGFormer results are taken from Yu (2023), whereas the SGNN-HN and CRAFT results are obtained from our experiments. The two prior Mamba-based baselines, DyGMamba and DyG-Mamba, are not reported because they exceed the available computational budget on all three datasets. As shown in Table 9, MambaTG achieves superior performance across all three datasets. These results further demonstrate the effectiveness and generalization of MambaTG across diverse temporal graph datasets.

Table 9: Future link prediction results on three additional TGB datasets. Results are reported as MRR (%), averaged over three random seeds. The best and second-best results in each column are highlighted. Missing entries indicate experiments that exceeded computational budget.
<table><tr><td colspan="2">Unseen-edge-dominant</td><td colspan="3">Repeated-edge-dominant</td></tr><tr><td>Model</td><td>tgbl-comment</td><td>Model</td><td>tgbl-coin</td><td>tgbl-flight</td></tr><tr><td>DyGFormer</td><td> $6 7 . 0 3 \pm 0 . 1 4$ </td><td>DyGFormer</td><td> ${ \underline { { 7 5 . 1 7 } } } \pm 0 . 3 8$ </td><td></td></tr><tr><td>SGNN-HN</td><td> $4 0 . 3 8 \pm 0 . 1 8$ </td><td>SGNN-HN</td><td> $6 6 . 2 5 \pm 0 . 4 1$ </td><td> $8 1 . 8 6 \pm 0 . 0 7$ </td></tr><tr><td>CRAFT</td><td> $7 5 . 6 9 \pm 0 . 7 5$ </td><td>CRAFT-R</td><td> $7 4 . 4 3 \pm 0 . 3 0$ </td><td> $8 6 . 4 2 \pm 0 . 0 5$ </td></tr><tr><td>MambaTG</td><td> $7 5 . 9 3 \pm 0 . 2 1$ </td><td>MambaTG-R</td><td> $7 5 . 7 4 \pm 0 . 2 8$ </td><td> $8 6 . 5 3 \pm 0 . 0 8$ </td></tr></table>

## K DISCUSSION

## K.1 LIMITATIONS AND FUTURE WORK

Although this work focuses on future link prediction, we have not examined whether the two signal modeled by MambaTG generalize to other temporal-graph prediction tasks. Extending MambaTG

to tasks such as temporal edge classification and temporal anomaly detection, while investigating the roles of historical-sequence evolution and candidate–history compatibility in these settings, is a promising direction for future work.