# Link Both Views: Cross-Image Routing Attention for Dual-Endpoint V2I Beam-Pair Power Prediction

Shiliang Lu, Graduate Student Member, IEEE, Lu Bai, Senior Member, IEEE, Ziwei Huang, Member, IEEE, and Xiang Cheng, Fellow, IEEE

Abstract—For sensing-aided communications, single-endpoint sensing offers only a partial view of propagation, motivating linkcentric sensing across both endpoints. We propose cross-image routing attention (CIRA) for sparse association and mutual enhancement of propagation-relevant features from dual-endpoint RGB images. Building on CIRA, CiFormer predicts millimeterwave (mmWave) beam-pair power matrices for vehicle-toinfrastructure (V2I) communications. Experiments show that CiFormer reduces mean absolute error and improves top-1 beam-pair selection accuracy by 3.15 percentage points over the strongest baseline. Visualizations and ablation studies support dual-endpoint sensing and selective cross-view interaction in capturing propagation-relevant dependencies, suggesting broader applications in wireless tasks beyond beam prediction.

Index Terms—Dual-endpoint sensing, link-centric sensing, cross-image routing attention, beam-pair power prediction, V2I communications.

## I. INTRODUCTION

RTIFICIAL INTELLIGENCE (AI) and machine learning (ML) are revolutionizing wireless network design, enabling more efficient, adaptive, and autonomous operation [1]. In the Internet of Vehicles (IoV), environment sensing further supports this evolution by providing observations of the propagation environment, offering new opportunities for informed and proactive decision-making [2]. However, leveraging high-dimensional environment observations for IoV communication system optimization is far from straightforward. This difficulty arises from the complex and dynamic relationship between environment observations and wireless propagation. Consequently, developing effective mechanisms to identify, represent, and exploit propagation-relevant sensing information remains a key challenge for environment-aware intelligent communications.

Most existing sensing-aided approaches rely on singleendpoint sensing, with sensors deployed at either the vehicle or the infrastructure endpoint [3], [4]. However, a single sensing endpoint captures only a partial view of the propagation environment due to its limited sensing coverage [5]. This limitation reveals a structural mismatch: a wireless link is inherently defined by two communication endpoints, i.e., the transmitter (Tx) and receiver (Rx), whereas single-endpoint sensing observes the propagation environment from only one side. Accordingly, this mismatch highlights the need to better align sensing with the two-endpoint structure of the wireless link, motivating a transition from endpoint-local sensing to link-centric sensing.

Nevertheless, effective characterization of the wireless link requires more than simply acquiring dual-endpoint observations. The two endpoint views are captured from distinct positions and orientations, resulting in substantial differences in perspective and semantic content. Given these differences, independently processing the two views or directly concatenating their features is insufficient to capture the link-relevant cross-view relationships [6]. Therefore, dual-endpoint sensing calls for an effective mechanism to identify and associate propagation-relevant information across the two views, jointly highlighting regions in both views associated with propagation paths that couple Tx-side departure and Rx-side arrival, rather than treating them as two independent observations. More broadly, capturing propagation-relevant relationships across the Tx and Rx views could support a range of tasks, including channel modeling, beam management, and resource allocation.

A representative link-centric task arises in millimeter-wave (mmWave) vehicle-to-infrastructure (V2I) communications. High mobility, dynamic blockage, and scattering can induce rapid channel fluctuations and consequently change the received powers of different beam pairs [7], [8]. A beam-pair power matrix (BPPM) characterizes the received powers across all candidate beam pairs, providing a more comprehensive description of the beam-domain link state than the strongest beam-pair index alone [8]. Accordingly, BPPM prediction depends jointly on the states of both endpoints and the propagation-relevant information of their surroundings. However, existing sensing-aided beam prediction approaches [3]– [5] predominantly exploit observations from only one side of the communication link, leaving link-relevant information across the two endpoint views insufficiently exploited.

To address these challenges, we develop CiFormer, a novel link-centric framework that captures propagation-relevant information through interactions between Tx- and Rx-side images. Then, this resulting joint representation is used for BPPM prediction. The main contributions and novelties are summarized as follows:

• A novel cross-image routing attention (CIRA) mechanism is proposed. Specifically designed for sensing-aided wireless communications, CIRA recasts visual attention from individual views to dual-endpoint views through a propagation-inspired coupling mechanism, making it the first attention mechanism to incorporate link-level dependencies into visual perception. Motivated by endpoint covisibility, it links both views through sparse cross-view associations and potentially supports broader wireless environment-aware communication tasks.

• A link-centric BPPM prediction framework, CiFormer, is proposed. Through CIRA, Tx- and Rx-side visual features are enabled to interact and form mutually enhanced representations. The resulting joint propagation representation alleviates single-endpoint partial observability, improving power prediction across candidate beam pairs and enabling more reliable beam-pair selection.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

Consider a V2I link from a vehicle Tx with $N _ { \mathrm { T x } }$ antennas to a roadside base station (BS) Rx with $N _ { \mathrm { R x } }$ antennas. The corresponding channel is denoted by $\mathbf { H } \in \mathbb { C } ^ { N _ { \mathrm { R x } } \times N _ { \mathrm { T x } } }$ . At each link snapshot, an RGB image ${ \bf \cal I } ^ { v }$ is acquired at each viewpoint $v \in \mathcal { V } = \{ \mathrm { T x } , \mathrm { R x } \}$ , where v¯ denotes the opposite side of v. The paired observations (I<sup>v</sup>, I<sup>v¯</sup>) provide complementary views of the propagation environment.

To characterize the link in the beam domain, fixed discrete Fourier transform (DFT)-based codebooks with unitnorm codewords are employed at both endpoints [9]. Let $\mathbf { F } = [ \mathbf { f } _ { 0 } , \dots , \mathbf { f } _ { P - 1 } ] \in \mathbb { C } ^ { \tilde { N _ { \mathrm { T x } } } \times P }$ and $\mathbf { W } = [ \mathbf { w } _ { 0 } , \ldots , \mathbf { w } _ { Q - 1 } ] \in$ $\mathbb { C } ^ { N _ { \mathrm { R x } } \times Q }$ denote the Tx precoding and Rx combining codebooks, respectively, where P and $Q$ denote the numbers of Tx and Rx beams. The codebook beams cover different azimuth directions, with elevation steering fixed at broadside. Following [8], each beam pair has a power gain $G _ { q , p } = \left| \mathbf { w } _ { q } ^ { H } \mathbf { H } \mathbf { f } _ { p } \right| ^ { 2 }$ and the complete BPPM is computed in matrix form as

$$
\mathbf { G } = \left| \mathbf { W } ^ { H } \mathbf { H } \mathbf { F } \right| ^ { \odot 2 } .\tag{1}
$$

Here, $( \cdot ) ^ { H }$ denotes the conjugate transpose, $| \cdot | ^ { \circledcirc 2 }$ denotes the entrywise squared magnitude, and $\dot { \bf G } \in \dot { \mathbb { R } } _ { > 0 } ^ { Q \dot { \times } P }$ contains the dimensionless channel power gains for all $\bar { Q } P$ candidate beam pairs. Its row index $q \in \{ 0 , \ldots , Q - 1 \}$ and column index $p \in \{ 0 , \ldots , P \mathrm { ~ - ~ } 1 \}$ correspond to the Rx and Tx beams, respectively. For a fixed transmit power, these gains are proportional to the received signal powers.

Given the endpoint-indexed image pair, the prediction target, i.e., the entire BPPM, is expressed in dB as $Y _ { q , p } =$ $1 0 \log _ { 1 0 } ( \operatorname* { m a x } \{ G _ { q , p } , \epsilon \} )$ , where $\epsilon \ > \ 0$ ensures numerical stability, and the target matrix is $\mathbf { Y } = [ Y _ { q , p } ] \in \mathbb { R } ^ { Q \times P }$ . The prediction problem is formulated as learning the mapping

$$
\widehat { \mathbf { Y } } = f _ { \theta } \big ( \{ \mathbf { I } ^ { v } \} _ { v \in \mathcal { V } } \big ) \in \mathbb { R } ^ { Q \times P }\tag{2}
$$

where θ denotes the learnable parameters and the input images retain their endpoint identities. Training uses paired endpoint images and their corresponding BPPM labels. H is used only to construct the labels and is not an input to the predictor. The predicted beam pair is selected as $( \hat { q } , \hat { p } ) = \mathrm { a r g } \operatorname* { m a x } _ { q , p } \hat { Y } _ { q , p }$ [8].

Overall, this formulation connects dual-endpoint visual observations with the power gains of all candidate beam pairs. The following section presents CiFormer for learning this mapping through propagation-relevant cross-view interactions.

## III. PROPOSED METHOD

In this letter, we propose CiFormer, which couples propagation-relevant cross-view feature association with structured beam-pair power prediction. CIRA provides the underlying interaction mechanism, while the BPPM prediction head estimates beam-pair powers from the jointly enhanced endpoint features.

![](images/0bdf777ab5e98c773bc38786ca948911a1f04c2b85689d724642d2dbd3cb79f6.jpg)  
(a) Physical Motivation of Tx–Rx Co-visible Reflection

![](images/af9b4b89679f6a0675bbbc53986904e03a1e64a64e4b6085e5c4a30ca8c56c82.jpg)  
Fig. 1. Illustration of the proposed cross-image routing attention. (a) Physical Top-k RouterTop-k Routermotivation of Tx–Rx co-visible reflection in wireless channels, where objects <sup>…</sup>…simultaneously visible from both views are more likely to contain scatterers Linear Projectionthat give rise to NLoS paths. (b) Cross-image routing attention, which Attention Logitsmmenhances the modeling of potential reflection-related windows guided by the Tx–Rx co-visible propagation mechanism.

## A. Cross-Image Routing Attention

The RGB images captured at both link ends provide visual semantics of the propagation environment. However, the Tx and Rx cameras observe the same physical scene from substantially different viewpoints, resulting in disparities in object appearance, scale, and occlusion. This challenge is further motivated by wireless channel propagation, where a singlebounce non-line-of-sight (NLoS) path requires a scattering region to be visible from both Tx and Rx, as illustrated in Fig. 1(a). Therefore, co-visible regions are more informative for characterizing such propagation, whereas dense attention computes interactions for all regions without accounting for their propagation relevance. Motivated by this observation, we propose CIRA, which establishes sparse cross-view interactions through a learnable top-k routing mechanism. As shown in Fig. 1(b), CIRA extends BiFormer’s bi-level routing [10] to cross-image attention, where queries from one view are routed to the most relevant windows in the other view rather than within the same feature map. By focusing cross-view interaction on selected windows, CIRA reduces redundant computation while enhancing propagation-relevant feature association. Algorithm 1 summarizes the overall procedure.

Window partitioning and QKV projection. Given RGB features from two views $\mathbf { X } ^ { v } \in \mathbb { R } ^ { H \times W \times C }$ , we first partition each into $N = n _ { h } \times n _ { w }$ windows of size $s _ { h } \times s _ { w } ,$ , yielding $\mathbf { X } _ { p } ^ { v } \in \mathbb { R } ^ { s _ { h } s _ { w } \times N \times C }$ . Each window is then mapped to query, key, and value via learnable linear projection as

$$
\mathbf { Q } _ { \mathrm { t o k } } ^ { v } = \mathbf { X } _ { p } ^ { v } \mathbf { W } ^ { q } , \quad \mathbf { K } _ { \mathrm { t o k } } ^ { v } = \mathbf { X } _ { p } ^ { v } \mathbf { W } ^ { k } , \quad \mathbf { V } _ { \mathrm { t o k } } ^ { v } = \mathbf { X } _ { p } ^ { v } \mathbf { W } ^ { v } .\tag{3}
$$

Note that $\mathbf { W } ^ { q } , \mathbf { W } ^ { k } , \mathbf { W } ^ { v }$ are shared across both views to ensure a consistent embedding space. To reduce the cost of subsequent attention, the key and value features are spatially downsampled via adaptive average pooling to $k _ { v } \times k _ { v }$ tokens, with $k _ { v } = 4$ following [10], yielding $\mathbf { K } _ { \mathrm { t o k } } ^ { v } , \mathbf { \bar { V } } _ { \mathrm { t o k } } ^ { v } \in \mathbb { R } ^ { \bar { k } _ { v } ^ { 2 } \times N \times C }$

![](images/35834deabc0a50cc1a5f1855034057c45306a0b582831aaff75cdc438117e72e.jpg)  
Fig. 2. Overall architecture of CiFormer, with detailed structures of the CiFormer block on the left and the BPPM prediction head on the right.

Algorithm 1 Cross-Image Routing Attention   
Input: Tx, Rx features $\overline { { \mathbf { X } ^ { \mathrm { T x } } , \mathbf { X } ^ { \mathrm { R x } } } } \in \mathbb { R } ^ { H \times W \times C }$   
Output: Tx, Rx features $\mathbf { X } _ { \mathrm { R G B } } ^ { \mathrm { T x } } , \mathbf { X } _ { \mathrm { R G B } } ^ { \mathrm { R x } }$   
1: Partition $\mathbf { X } ^ { \mathrm { { T x } } } , \mathbf { X } ^ { \mathrm { { R x } } }$ into N windows   
2: Shared QKV projection; downsample $\mathbf { K } , \mathbf { V }$ to $k _ { v } \times k _ { v }$   
3: Obtain ${ \bf Q } _ { \mathrm { w i n } } , { \bf \bar { K } } _ { \mathrm { w i n } }$ by average pooling   
4: for $v \in \mathcal { V } = \{ \mathrm { T x } . \mathrm { ~ }$ , Rx} do   
5: $\mathbf { A } = ( \mathbf { Q } _ { \mathrm { w i n } } ^ { v } \mathbf { \tilde { W } } ^ { r } ) ( \dot { \mathbf { K } } _ { \mathrm { w i n } } ^ { \bar { v } } \mathbf { W } ^ { r } ) ^ { \top } / \sqrt { C }$   
6: $\mathcal { T } = \mathrm { \hat { T o p K I n d e x } } ( \mathbf { A } , k )$   
7: r = Softmax(TopK(A, k))   
8: $\mathbf { r }  \mathbf { g } ( \mathbf { Q } _ { \mathrm { w i n } } ^ { v } ) \odot \mathbf { r }$   
9: for $i = 1$ to N do   
10: $( \mathbf { K } _ { i } ^ { \bar { v } } , \mathbf { V } _ { i } ^ { \bar { v } } ) = \operatorname { G a t h e r } ( \mathbf { K } ^ { \bar { v } } , \mathbf { V } ^ { \bar { v } } , \mathcal { T } _ { i } , \mathbf { r } _ { i } )$   
11: $\dot { \mathbf { B } } _ { i } \overset {  } { = } \mathrm { M L P } ( \phi _ { q } ( \mathbf { Q } _ { \mathrm { w i n } , i } ^ { v } ) - \phi _ { k } ( \mathbf { K } _ { \mathrm { w i n } , \mathcal { T } _ { i } } ^ { \bar { v } } ) )$   
12: $\mathbf { O } _ { i } ^ { v } = \operatorname { S o f t m a x } ( \mathbf { Q } _ { i } ^ { v } ( \mathbf { K } _ { i } ^ { \bar { v } } ) ^ { \top } / \sqrt { C } + \mathbf { B } _ { i } ^ { \bar { \mathbf { \alpha } } } ) \mathbf { V } _ { i } ^ { \bar { v } }$   
13: end for   
14: $\mathbf { X } _ { \mathrm { R G B } } ^ { v } = \mathbf { W } ^ { o } ( \operatorname { U n p a t c h i f y } ( \{ \mathbf { O } _ { i } ^ { v } \} ) + \operatorname { L C E } ( \mathbf { V } ^ { v } ) )$   
15: end for   
16: Return $\mathbf { X } _ { \mathrm { R G B } } ^ { \mathrm { T x } } , \mathbf { X } _ { \mathrm { R G B } } ^ { \mathrm { R x } }$

Window-to-window cross-image routing. To capture the co-visible prior, we perform routing at the window level. Specifically, each window is summarized by average pooling over the spatial dimension to obtain window-level descriptors ${ \bf Q } _ { \mathrm { w i n } } ^ { v } , { \bf K } _ { \mathrm { w i n } } ^ { v } \in \mathbb { R } ^ { N \times C }$ . A learnable projection $\mathbf { W } ^ { r } \in \mathbb { R } ^ { \mathbf { \hat { C } } \times C }$ is adopted to enhance their discriminability, after which the cross-view affinity is computed as

$$
\mathbf { A } ^ { v } = \frac { ( \mathbf { Q } _ { \mathrm { w i n } } ^ { v } \mathbf { W } ^ { r } ) ( \mathbf { K } _ { \mathrm { w i n } } ^ { \bar { v } } \mathbf { W } ^ { r } ) ^ { \top } } { \sqrt { C } } \in \mathbb { R } ^ { N \times N }\tag{4}
$$

where $\mathbf { A } ^ { v }$ , i.e., affinity score, quantifies the semantic similarity between two windows. For each window in view $v ,$ the top-k most relevant windows from view v¯ are retained, yielding a routing index set ${ \mathcal { T } } ^ { v } = { \mathrm { t o p k I n d e x } } ( \mathbf { A } ^ { v } )$ and routing weights $r _ { i j } ^ { v }$ via softmax, where i and j denote the query and routed key window indices, respectively. Then, a soft query gate $g ( \cdot )$ modulates routing weights to suppress unreliable queries.

Semantic-aware bias. Without a shared geometric reference, conventional relative position biases offer limited benefit. Instead, we introduce a semantic-aware bias (SAB) that derives a pairwise additive term from the feature discrepancies between a query window and its routed key windows, formulated as

$$
\mathbf { B } _ { i } ^ { v } = \mathrm { M L P } ( \phi _ { q } ( \mathbf { Q } _ { \mathrm { w i n } , i } ^ { v } ) - \phi _ { k } ( \mathbf { K } _ { \mathrm { w i n } , \mathcal { T } _ { i } ^ { v } } ^ { \bar { v } } ) )\tag{5}
$$

where $\phi _ { q } , \phi _ { k }$ are learnable linear projections, MLP denotes a multilayer perceptron, and $\mathbf { B } _ { i } ^ { v }$ provides the additive bias for

the top-k routed windows.

Token-to-token cross-image attention. With the routing index set $\mathcal { T } ^ { v }$ and weights $r _ { i j } ^ { v }$ from window-level routing, the downsampled KV tokens of the target window are gathered for efficient computation [10]. For the i-th Tx window, $\mathbf { K } _ { i } ^ { \bar { v } } , \mathbf { V } _ { i } ^ { \bar { v } }$ can be given as

$$
\mathbf { K } _ { i } ^ { \bar { v } } = \mathrm { g a t h e r } ( \mathbf { K } _ { \mathrm { t o k } } ^ { \bar { v } } , \mathcal { T } _ { i } ^ { v } , r _ { i } ^ { v } ) , \quad \mathbf { V } _ { i } ^ { \bar { v } } = \mathrm { g a t h e r } ( \mathbf { V } _ { \mathrm { t o k } } ^ { \bar { v } } , \mathcal { T } _ { i } ^ { v } , r _ { i } ^ { v } ) .\tag{6}
$$

Then, the gathered tokens are re-weighted by the routing scores $r _ { i } ^ { v }$ before being stacked as $\mathbf { K } _ { i } ^ { \tilde { v } } , \mathbf { V } _ { i } ^ { \tilde { v } } \in \mathbf { \bar { \mathbb { R } } } ^ { k k _ { v } ^ { 2 } \times C }$ . The token-level cross-attention, augmented by the semantic-aware bias $\mathbf { B } _ { i } ^ { v }$ , is written as

$$
\mathbf { O } _ { i } ^ { v } = \mathrm { S o f t m a x } \Big ( \frac { \mathbf { Q } _ { \mathrm { t o k } , i } ^ { v } ( \mathbf { K } _ { i } ^ { \bar { v } } ) ^ { \top } } { \sqrt { C } } + \mathbf { B } _ { i } ^ { v } \Big ) \mathbf { V } _ { i } ^ { \bar { v } }\tag{7}
$$

where $\mathbf { Q } _ { \mathrm { t o k } , i } ^ { v } \in \mathbb { R } ^ { s _ { h } s _ { w } \times C }$ contains the query tokens in window i of view v. The reverse direction is computed symmetrically. Window outputs are merged into the spatial grid and augmented by local context enhancement $\operatorname { L C E } ( \cdot )$ following [10], ${ \textrm { a 3 } } \times 3$ depthwise convolution over $\mathbf { V } _ { \mathrm { t o k } } ^ { v }$ . Finally, a projection $\mathbf { W } ^ { o } \in \mathbb { R } ^ { C \times C }$ produces the fused output as

$$
\mathbf { X } _ { \mathrm { R G B } } ^ { v } = \mathbf { W } ^ { o } ( \mathrm { U n p a t c h i f y } ( \{ \mathbf { O } _ { i } ^ { v } \} _ { i = 1 } ^ { N } ) + \mathrm { L C E } ( \mathbf { V } _ { \mathrm { t o k } } ^ { v } ) ) .\tag{8}
$$

## B. CiFormer Architecture and BPPM Prediction

Building on CIRA, CiFormer realizes (2) through a fourstage backbone and an azimuth-aware head that explicitly models beam-pair interactions.

Backbone architecture. As illustrated in Fig. 2, the backbone combines convolutional downsampling with stacked CIRA blocks for cross-view interaction. Each CIRA-based block comprises a 3×3 depthwise convolution for implicit positional encoding, CIRA, and a two-layer MLP. Final projection convolutions produce the endpoint features $\mathbf { X } _ { \mathrm { o u t } } ^ { v } \in \mathbf { \bar { R } } ^ { \mathbf { \bar { H } } _ { f } \times W _ { f } \times C _ { f } }$ which retain spatial information for subsequent prediction.

BPPM prediction head. As illustrated in Fig. 2, the head first applies convolutional projection followed by azimuthoriented pooling. Since the codebooks scan azimuth with fixed elevation, pooling aggregates features over height while preserving horizontal order. Specifically, the pooled feature at column w is computed as

$$
\mathbf { s } _ { w } ^ { v } = \frac { 1 - \beta ^ { v } } { H _ { f } } \sum _ { h = 1 } ^ { H _ { f } } \mathbf { z } _ { h , w } ^ { v } + \beta ^ { v } \sum _ { h = 1 } ^ { H _ { f } } \alpha _ { h , w } ^ { v } \mathbf { z } _ { h , w } ^ { v }\tag{9}
$$

![](images/31e016964d6fbb633e9a1e763a4e13371e4bad5441be2543a0fcba24799c2172.jpg)  
Fig. 3. Bird’s-eye view of the V2I simulation scenario, with endpoint locations, vehicle trajectories, BS orientations, and seven vehicle (Tx)–BS (Rx) links.

where $\mathbf { z } _ { h , w } ^ { v }$ is the projected feature, $\alpha _ { h , w } ^ { v }$ is a learned attention weight normalized over height, and $\beta ^ { v } \in ( 0 , 1 )$ controls the mixture. The pooled features are then embedded into a vector $\mathbf { e } _ { m } ^ { v }$ for each codebook beam m at view v.

To construct the BPPM, the head computes dot products between the two sets of beam vectors. Taking $v = \mathrm { T x }$ and following the BPPM row–column convention, the score for Rx beam q and Tx beam p is

$$
Y _ { q , p } ^ { \mathrm { d o t } } = \frac { ( \mathbf { e } _ { q } ^ { \bar { \mathrm { v } } } ) ^ { \top } \mathbf { e } _ { p } ^ { \mathrm { v } } } { \sqrt { D } }\tag{10}
$$

where D denotes the beam embedding dimension. Collecting all scores forms the BPPM $\mathbf { Y } ^ { \mathrm { d o t } } \in \mathbb { R } ^ { Q \times P }$ . The resulting BPPM is refined by a residual branch comprising two $3 \times 3$ convolutions with an intermediate GELU. CiFormer is trained end to end on standardized absolute-dB BPPM targets using a weighted Huber loss as

$$
\mathcal { L } = \frac { 1 } { Q P } \sum _ { q = 0 } ^ { Q - 1 } \sum _ { p = 0 } ^ { P - 1 } w _ { q , p } \rho _ { \delta } \Bigl ( \widehat { Y } _ { q , p } - Y _ { q , p } \Bigr )\tag{11}
$$

where $\rho _ { \delta }$ is the Huber-type loss and $w _ { q , p }$ is a mean-normalized weight derived from the ground-truth , thereby prioritizing the optimal beam pair during training.

Overall, CIRA provides a general mechanism for propagation-relevant cross-view interaction through reciprocal feature selection and enhancement. Building on this, CiFormer integrates CIRA with a dedicated BPPM prediction head to jointly exploit dual-endpoint observations and realize the BPPM prediction mapping in (2), as formulated in Section II.

## IV. EXPERIMENTAL RESULTS

## A. Simulation Setup

Following the aligned sensing–communication datageneration methodology in [11], we construct the Dualendpoint Sensing Dataset for Beam-pair Power Prediction (DuSen-BP<sup>3</sup>). As illustrated in Fig. 3, DuSen-BP<sup>3</sup> is generated in a V2I environment comprising six vehicles and five BSs, which form 7 communication links. The carrier frequency is 28 GHz, and each endpoint uses a 4 × 4 uniform planar array (UPA) and a seven-beam codebook. Each sample pairs dual-endpoint RGB images with the corresponding $7 \times 7$ BPPM computed via (1). Finally, DuSen-BP<sup>3</sup> comprises 6,965 valid samples, which are split into training, validation, and test sets in a 7:1:2 ratio.

TABLE I  
COMPARISON OF BPPM PREDICTION PERFORMANCE
<table><tr><td>Model</td><td>Params. (M)</td><td>RMSE (dB)</td><td>MAE (dB)</td><td>Acc@1 (%)</td><td>Acc@3 (%)</td><td>Mean BF loss (dB)</td></tr><tr><td>ViT</td><td>10.995</td><td>6.318</td><td>4.143</td><td>68.25</td><td>91.81</td><td>0.943</td></tr><tr><td>MobileNetV4</td><td>14.950</td><td>6.553</td><td>4.156</td><td>68.39</td><td>91.16</td><td>0.924</td></tr><tr><td>MambaOut</td><td>11.831</td><td>6.014</td><td>3.864</td><td>69.26</td><td>93.03</td><td>0.658</td></tr><tr><td>CiFormer</td><td>10.986</td><td>5.810</td><td>3.638</td><td>72.41</td><td>94.68</td><td>0.641</td></tr></table>

RGB inputs are resized to 256 × 448. The four-stage backbone has depths (2, 2, 8, 2), where the first three stages use CIRA with routing top-k values (4, 8, 12), and the fourth adopts self-attention to refine the features. CiFormer is trained with AdamW for 150 epochs using a batch size of 64, a learning rate of $1 0 ^ { - 4 }$ , and weight decay of $1 0 ^ { - 3 }$ . The learning rate uses a five-epoch linear warm-up followed by cosine decay. For comparison, three representative vision backbones, i.e., Vision Transformer (ViT), MobileNetV4, and MambaOut, are adopted as baselines. BPPM prediction is evaluated using root mean squared error (RMSE) and mean absolute error (MAE) in dB. For the optimal beam-pair index $b _ { s } ^ { \star } ~ = ~ \arg \operatorname* { m a x } _ { { \hat { \mathbf { \alpha } } } ^ { q } , p } Y _ { s , q , p }$ and the predicted top-k beam-pair index set $\mathcal { T } _ { k } ( \hat { \mathbf { Y } } _ { s } )$ of sample s, beam-pair selection is evaluated using top-k accuracy (Acc@k, %) and beamforming (BF) loss (dB), defined as

$$
\begin{array} { c } { \displaystyle \mathrm { A c c @ } k = \frac { 1 0 0 } { S } \sum _ { s = 1 } ^ { S } \mathbf { 1 } \{ b _ { s } ^ { \star } \in \mathcal { T } _ { k } ( \widehat { \mathbf { Y } } _ { s } ) \} , } \\ { \displaystyle L _ { \mathrm { B F } , s } = \operatorname* { m a x } _ { q , p } Y _ { s , q , p } - Y _ { s , \hat { q } _ { s } , \hat { p } _ { s } } } \end{array}\tag{12}
$$

where $S$ is the number of evaluation samples and $\mathbf { 1 } \{ \cdot \}$ is an indicator. Here we report Acc@1, Acc@3, and mean BF loss.

## B. Simulation Results

Table I compares CiFormer with the three baseline backbones on DuSen-BP<sup>3</sup>. It can be seen that CiFormer achieves lower RMSE, MAE, and mean BF loss and higher Acc@1 and Acc@3 than all three baselines. Relative to MambaOut, the strongest baseline, RMSE and MAE decrease by 3.39% and 5.85%, respectively. CiFormer’s Acc@1 and Acc@3 reach 72.41% and 94.68%, exceeding MambaOut by 3.15 and 1.65 percentage points, respectively. CiFormer achieves these gains with a parameter count comparable to ViT and lower than the other baselines. ViT’s global self-attention, MobileNetV4’s spatial and channel mixing, and MambaOut’s gated convolutions extract intra-view context, leaving crossendpoint association to later feature fusion. CIRA instead uses opposite-view information to guide feature selection and enhancement, better aligning visual representations with the coupled Tx–Rx structure of beam-pair powers.

This interpretation is further supported by the Grad-CAM [12] visualizations in Fig. 4. CiFormer highlights covisible buildings and roadside tree clusters in both endpoint views, with more consistent activations than the baselines despite differences in image position and scale. ViT appears to rely on a few localized visual cues, as indicated by its isolated hotspots. MobileNetV4 also activates sky and road regions, suggesting distraction by backgrounds. MambaOut exhibits a broad, diffuse response in the Rx view. These paired activations suggest that CIRA encourages both endpoint representations to focus on shared scene content, providing qualitative support for its use of opposite-view information to guide feature selection and enhancement.

![](images/9f87bd18792c60b86dd724edd3ab0f38ad08f264f035fc9a94e0eeb34937cde0.jpg)  
Fig. 4. Grad-CAM visualizations of CiFormer, ViT, MobileNetV4, and MambaOut. CiFormer’s activation maps emphasize the building and roadside trees in both views, whereas those of the baselines exhibit weaker cross-view correspondence.

TABLE II  
ABLATION STUDY OF DUAL-ENDPOINT INPUTS AND CIRA COMPONENTS
<table><tr><td>Variant</td><td>RMSE (dB)</td><td>MAE (dB)</td><td>Acc@1 (%)</td><td>Acc@3 (%)</td><td>Mean BF loss (dB)</td></tr><tr><td>w/o Tx</td><td>10.097</td><td>7.047</td><td>40.95</td><td>65.30</td><td>6.590</td></tr><tr><td>w/o Rx</td><td>12.344</td><td>6.867</td><td>54.96</td><td>73.56</td><td>4.414</td></tr><tr><td>w/o Interaction</td><td>6.001</td><td>3.740</td><td>70.98</td><td>93.89</td><td>0.723</td></tr><tr><td>w/o top-k routing</td><td>5.922</td><td>3.672</td><td>71.19</td><td>93.97</td><td>0.703</td></tr><tr><td>w/o Gating</td><td>5.827</td><td>3.678</td><td>71.19</td><td>93.53</td><td>0.756</td></tr><tr><td>w/o SAB</td><td>5.893</td><td>3.767</td><td>71.05</td><td>93.53</td><td>0.708</td></tr><tr><td>Full CiFormer</td><td>5.810</td><td>3.638</td><td>72.41</td><td>94.68</td><td>0.641</td></tr></table>

To further validate the effectiveness of CiFormer, an ablation study is also conducted. As shown in Table II, using only one endpoint image substantially degrades CiFormer’s prediction and beam-selection performance. This degradation in both single-endpoint settings highlights the necessity of dual-endpoint sensing. In addition, RMSE increases more when the Rx view is removed, indicating greater sensitivity to the loss of receiver-side information. Disabling cross-view interaction degrades prediction and beam-selection performance, highlighting the importance of associating features across endpoints. Removing top-k routing turns CIRA into dense cross-view attention over all windows and reduces performance, suggesting that irrelevant background regions can interfere with propagation-relevant feature association. The degradation without gating or SAB supports their respective roles in suppressing unreliable queries and modeling semantic correspondence across viewpoints without a shared geometric reference. These results demonstrate the effectiveness of CiFormer’s link-centric design, which combines dualendpoint observations with selective cross-view association to improve BPPM prediction and beam-pair selection, suggesting its applicability to broader sensing-aided communication tasks.

## V. CONCLUSION

In this letter, we proposed CIRA for selective association and enhancement of propagation-relevant features across dualendpoint views. CiFormer applies CIRA to link-centric V2I BPPM prediction from paired RGB images. Experiments on DuSen-BP<sup>3</sup> showed improved prediction and beam selection with fewer parameters. Grad-CAM and ablation results further supported dual-endpoint sensing and selective interaction. These results demonstrate the effectiveness of CIRA’s design, suggesting its applicability to broader wireless communication tasks that rely on environment sensing.

## REFERENCES

[1] X. Cheng et al., “Intelligent multi-modal sensing-communication integration: Synesthesia of Machines,” IEEE Commun. Surveys Tuts., vol. 26, no. 1, pp. 258–301, 1st Quart. 2024.

[2] L. Bai, Z. Huang, M. Sun, X. Cheng, and L. Cui, “Multi-modal intelligent channel modeling: A new modeling paradigm via Synesthesia of Machines,” IEEE Commun. Surveys Tuts., vol. 28, pp. 2612–2649, 2026.

[3] G. Charan, M. Alrabeiah, T. Osman, and A. Alkhateeb, “Camera based mmWave beam prediction: Towards multi-candidate real-world scenarios,” IEEE Trans. Veh. Technol., vol. 74, no. 4, pp. 5897–5913, Apr. 2025.

[4] W. Xu, F. Gao, X. Tao, J. Zhang, and A. Alkhateeb, “Computer vision aided mmWave beam alignment in V2X communications,” IEEE Trans. Wireless Commun., vol. 22, no. 4, pp. 2699–2714, Apr. 2023.

[5] B. Lin, F. Gao, Y. Zhang, C. Pan, and G. Liu, “Multi-camera views based beam searching and BS selection with reduced training overhead,” IEEE Trans. Commun., vol. 72, no. 5, pp. 2793–2805, May 2024.

[6] J. Zeng, C. Pan, H. Weng, R. Liu, H. Ren, and J. Wang, “A BEV-Fusion based framework for sequential multi-modal beam prediction in mmWave systems,” arXiv:2604.05668, 2026.

[7] J. Tan et al., “Beam alignment in mmWave V2X communications: A survey,” IEEE Commun. Surveys Tuts., vol. 26, no. 3, pp. 1676–1709, 3rd Quart. 2024.

[8] Y. Wang, M. Narasimha, and R. W. Heath, “MmWave beam prediction with situational awareness: A machine learning approach,” in Proc. IEEE 19th Int. Workshop Signal Process. Adv. Wireless Commun. (SPAWC), 2018, pp. 1–5.

[9] D. Yang, L.-L. Yang, and L. Hanzo, “DFT-based beamforming weightvector codebook design for spatially correlated channels in the unitary precoding aided multiuser downlink,” in Proc. IEEE Int. Conf. Commun. (ICC), 2010, pp. 1–5.

[10] L. Zhu, X. Wang, Z. Ke, W. Zhang, and R. W. H. Lau, “BiFormer: Vision transformer with bi-level routing attention,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Vancouver, Canada, 2023, pp. 10 323–10 333.

[11] X. Cheng et al., “SynthSoM: A synthetic intelligent multi-modal sensing-communication dataset for Synesthesia of Machines (SoM),” Sci. Data, vol. 12, May 2025, Art. no. 819.

[12] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626.