# RAG-CSI: Global–Local Retrieval via Joint State and Dynamics Matching for CSI Prediction

Xinyue Jiang<sup>†</sup>, Tong Li<sup>†</sup>, Shiceng Zhang<sup>§</sup>, Ke Chen<sup>‡</sup>, Zhu Xiao<sup>†</sup>

<sup>†</sup>College of Computer Science and Electronic Engineering, Hunan University, Changsha, China.

<sup>§</sup>Xidian University, Xian, China.

<sup>‡</sup>Peng Cheng Laboratory, Shenzhen 518066, China.

Abstract—Channel state information (CSI) prediction forecasts future channel states from historical observations, supporting transceiver design and resource allocation with reduced measurement overhead. However, existing deep learning predictors mainly encode channel knowledge in model parameters, which can limit their ability to represent diverse patterns of channel dynamics across scenes. In this paper, we propose RAG-CSI, a retrieval-enhanced Transformer framework that augments prediction with relevant historical channel knowledge. Specifically, we construct a global–local CSI knowledge store containing representative observations across scenes and historical observations from the current serving scene. We further develop joint state– dynamics retrieval (JSDR), which jointly characterizes channel states, phase dynamics, and amplitude dynamics to retrieve relevant CSI sequences. A Transformer predictor then uses crossattention to fuse the retrieved knowledge with the current CSI history and predict future CSI with limited additional model parameters. Experiments on a multi-scene CSI dataset show overall improvements exceeding 10% across all five evaluation metrics over the best baseline for each metric, including a 15.42% reduction in normalized mean squared error.

Index Terms—CSI prediction, Channel Knowledge, Retrieval-Augmented Generation (RAG), Transformer.

## I. INTRODUCTION

Channel state information (CSI) describes the propagation characteristics of wireless links and provides the basis for transceiver design and resource allocation [1]–[4]. CSI acquisition typically relies on pilot-based channel measurements and can incur substantial signaling overhead, motivating CSI prediction to reduce the need for additional measurements [5], [6]. CSI prediction typically forecasts future channel states from historical observations by exploiting temporal correlations in CSI. This task is challenging because multipath propagation and user mobility induce complex channel variations across time, antennas, and subcarriers.

Deep learning has been increasingly explored for CSI prediction owing to its ability to learn nonlinear dependencies from channel observations [7]. Representative architectures include recurrent neural networks with long short-term memory (LSTM) or gated recurrent unit (GRU) modules [8], and Transformers that capture temporal dependencies through attention mechanisms [9]–[11]. More recently, wireless foundation models such as WiFo [12] have extended prediction across different channel configurations through training on heterogeneous CSI datasets. Despite these advances, existing deep learning methods primarily rely on network parameters to encode channel patterns. As the scale and diversity of CSI data increase, limited model capacity can constrain the representation of channel dynamics patterns across various scenes.

![](images/4eee94e3c011091259a49ec19d3af54721634d8c404ee80c426ed527a3633e43.jpg)  
Fig. 1: Comparison of conventional and retrieval-augmented CSI prediction frameworks.

In this paper, we aim to develop a CSI prediction framework that accommodates diverse channel dynamics with a modest number of model parameters. Achieving such an objective involves three challenges. First, CSI prediction requires learning complex channel dynamics from high-dimensional historical observations, but limited observation windows and model capacity restrict the use of diverse channel dynamics patterns. Second, differences in propagation conditions across scenes can lead to distinct CSI statistics and temporal variation patterns. Consequently, learned global channel knowledge may not adequately support CSI prediction in a specific scene. Third, CSI exhibits complex variations across time, antennas, and subcarriers. Sequences with similar overall values may exhibit different directions of phase dynamics and different amplitude trends. Constructing a relevance measure to capture the channel dynamics patterns from limited historical observations is challenging.

To address these challenges, we propose RAG-CSI, a retrieval-enhanced framework for CSI prediction across diverse scenes. As illustrated in Fig. 1, RAG-CSI retrieves relevant channel knowledge from external memory to augment the context available to the predictor. Specifically, we construct a global–local CSI knowledge store to provide complementary references. The global store, maintained on the user side, contains representative historical CSI accumulated or preloaded across scenes, while the local store, maintained at the serving base station, contains previously observed CSI from the current scene. We then develop a joint state–dynamics retrieval method to identify references relevant to the prediction task.

Using historical complex-valued CSI, we construct descriptors of channel states, phase dynamics, and amplitude dynamics. Normalized complex conjugate products characterize phase dynamics, while logarithmic amplitude differences capture amplitude dynamics. These descriptors are independently normalized and combined with weights into a unified retrieval representation, enabling the retrieval of references with similar channel states and historical dynamics patterns to provide informative context for CSI prediction. Finally, a Transformer predictor integrates the retrieved channel knowledge with the input CSI history through cross-attention, enabling prediction to benefit from both learned patterns and explicitly retrieved references.

Our main contributions are summarized as follows:

• We investigate retrieval-enhanced CSI prediction by explicitly exploiting historical channel observations beyond the current input window. The proposed framework supplements historical observations with relevant channel knowledge retrieved using CSI, expanding the available prediction context and reducing reliance on knowledge encoded solely in model parameters.

• We propose RAG-CSI, a retrieval-enhanced Transformer framework for CSI prediction. RAG-CSI combines global and local knowledge stores to provide complementary channel references, and develops joint state–dynamics retrieval (JSDR) to identify relevant references by jointly considering channel states, phase dynamics, and amplitude dynamics. The retrieved knowledge is integrated through cross-attention to improve prediction accuracy with limited additional model parameters.

• We conduct experiments on a multi-scene CSI dataset to evaluate RAG-CSI. The results demonstrate that RAG-CSI outperforms state-of-the-art baselines by more than 10% across all five metrics: mean squared error (MSE), normalized mean squared error (NMSE), mean absolute error (MAE), normalized mean absolute error (NMAE), and cosine similarity $\rho .$

## II. SYSTEM MODEL

## A. Channel Model

We consider a MIMO-OFDM system in which a base station (BS) equipped with $N _ { t }$ antennas serves multiple singleantenna users over $N _ { c }$ subcarriers. Although the BS serves multiple users within a propagation scene, channel prediction is performed on a per-user basis. For simplicity, we describe the channel between the BS and one user. The transmitted signal reaches the user through multiple propagation paths, and the resulting channel is represented as the superposition of P paths. The channel vector on the n-th subcarrier is written as

$$
{ \bf h } _ { n } = \sum _ { p = 1 } ^ { P } \alpha _ { p } e ^ { - j 2 \pi f _ { n } \tau _ { p } } { \bf a } ( f _ { n } , { \bf s } _ { p } ) ,\tag{1}
$$

where $\alpha _ { p } \in \mathbb { C } , \tau _ { p }$ , and $\mathbf { s } _ { p }$ denote the complex path gain, propagation delay, and departure direction of the p-th path,

respectively, and $f _ { n }$ is the frequency of the n-th subcarrier. The array response vector $\mathbf { a } ( f _ { n } , \mathbf { s } _ { p } ) \in \mathbb { C } ^ { N _ { t } }$ is given by

$$
\mathbf { a } ( f _ { n } , \mathbf { s } _ { p } ) = \left[ e ^ { - j \frac { 2 \pi f _ { n } } { c } \mathbf { r } _ { 1 } ^ { T } \mathbf { s } _ { p } } , e ^ { - j \frac { 2 \pi f _ { n } } { c } \mathbf { r } _ { 2 } ^ { T } \mathbf { s } _ { p } } , \dots , e ^ { - j \frac { 2 \pi f _ { n } } { c } \mathbf { r } _ { N _ { t } } ^ { T } \mathbf { s } _ { p } } \right] ^ { T } ,\tag{2}
$$

where $c$ is the speed of light and $\mathbf { r } _ { m }$ denotes the threedimensional position vector of the m-th antenna relative to a reference antenna. For a uniform linear array (ULA), this response reduces to a phase progression determined by the antenna spacing and angle of departure. Stacking the channel vectors across all subcarriers yields the complete CSI matrix

$$
\mathbf { H } = [ \mathbf { h } _ { 1 } , \mathbf { h } _ { 2 } , \dots , \mathbf { h } _ { N _ { c } } ] \in \mathbb { C } ^ { N _ { t } \times N _ { c } } ,\tag{3}
$$

where the n-th column $\mathbf { h } _ { n } ~ \in ~ \mathbb { C } ^ { N _ { t } }$ represents the channel vector on the n-th subcarrier.

## B. Problem Definition

CSI prediction estimates future channels from historical observations. We use DeepMIMO [13] to generate CSI at sampled user locations under specified antenna and OFDM configurations, and then organize the CSI snapshots along simulated user trajectories in temporal order. Let H<sub>t</sub> $\in \mathbb { C } ^ { \mathbf { \mathcal { N } } _ { t } \times N _ { c } }$ denote the CSI at time t. The historical and future CSI sequences are

$$
\begin{array} { r } { \mathcal { H } _ { t } ^ { \mathrm { h i s t } } = \left( \mathbf { H } _ { t - L + 1 } , \ldots , \mathbf { H } _ { t } \right) , } \\ { \mathcal { H } _ { t } ^ { \mathrm { f u t } } = \left( \mathbf { H } _ { t + 1 } , \ldots , \mathbf { H } _ { t + Q } \right) , } \end{array}\tag{4}
$$

where $L$ and $Q$ are the history and prediction lengths, respectively, with $\mathcal { H } _ { t } ^ { \mathrm { { i s t } } } \in \mathbb { C } ^ { L \times N _ { t } \times \mathbf { \bar { N } _ { c } } }$ and $\mathcal { H } _ { t } ^ { \mathrm { f u t } } \in \mathbb { C } ^ { Q \times \mathbf { \breve { N } } _ { t } \times N _ { c } }$

A conventional predictor estimates future CSI solely from the observed history:

$$
\begin{array} { r } { \widehat { \mathcal { H } } _ { t } ^ { \mathrm { f u t } } = F _ { \theta } \big ( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } \big ) . } \end{array}\tag{5}
$$

In RAG-CSI, relevant channel knowledge is additionally retrieved from a CSI memory M:

$$
\begin{array} { r } { \mathcal { K } _ { t } = \mathcal { R } \big ( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } ; \mathcal { M } \big ) , } \\ { \widehat { \mathcal { H } } _ { t } ^ { \mathrm { f u t } } = F _ { \theta } \big ( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } , \mathcal { K } _ { t } \big ) , } \end{array}\tag{6}
$$

where $\mathcal { R } ( \cdot )$ denotes retrieval and $\textstyle { \mathcal { K } } _ { t }$ denotes the retrieved channel knowledge. The predictor is trained by minimizing

$$
\operatorname* { m i n } _ { \theta } \ \mathcal { L } \big ( F _ { \theta } \big ( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } , \mathcal { K } _ { t } \big ) , \mathcal { H } _ { t } ^ { \mathrm { f u t } } \big ) ,\tag{7}
$$

where $\mathcal L ( \cdot , \cdot )$ measures the prediction error.

## III. METHOD

## A. Overview

RAG-CSI is a retrieval-augmented framework that enhances Transformer-based CSI prediction with relevant historical channel knowledge. As shown in Fig. 2, it consists of a Global–Local CSI knowledge store, Joint State–Dynamics Retrieval (JSDR), and a Transformer predictor with crossattention fusion. The Global store contains representative CSI across scenes, while the Local store maintains historical CSI from the current serving scene. Given the observed CSI history, JSDR jointly considers channel states, phase dynamics, and amplitude dynamics to retrieve relevant sequences from both stores. Cross-attention then fuses the retrieved knowledge with the historical representation, and the Transformer encoder–decoder models channel dynamics to predict future CSI through a projection layer.

![](images/566bba9b8b1770ea5165cdeb7caf5e7c61736e8434854f095c460bb9957f0734.jpg)  
Fig. 2: The overview of the proposed RAG-CSI. The framework consists of JDSR, a Transformer predictor, and CSI reconstruction. JSDR combines CSI states with phase and amplitude dynamics to retrieve relevant sequences from globa and local CSI stores, offering historical references for Transformer-based prediction.

## B. Retrieval-Augmented Generation (RAG) for CSI Prediction

Retrieval-augmented generation (RAG) enhances a parametric model with relevant information retrieved from external memory. For CSI prediction, it introduces reusable historical channel patterns beyond the current observation window. As shown in Fig. 2, RAG-CSI uses $\mathcal { H } _ { t } ^ { \mathrm { h i s t } }$ as the query to retrieve channel references from the knowledge store M, which are then incorporated into the Transformer to predict CSI.

1) Global–Local CSI Knowledge Store: To capture both cross-scene and local channel knowledge, we construct $\mathcal { M } =$ $\left( \mathcal { M } _ { \mathrm { G } } , \mathcal { M } _ { \mathrm { L } } \right)$ . The Global store $\mathcal { M } _ { \mathrm { G } }$ is maintained on the user side and contains historical CSI knowledge accumulated or preloaded across multiple scenes, while the Local store $\mathcal { M } _ { \mathrm { L } }$ is maintained at the serving BS and contains knowledge collected within its serving area. Both stores use the same storage format, enabling the current CSI history $\mathcal { H } _ { t } ^ { \mathrm { h i s t } }$ to retrieve complementary references from both sources.

2) Global–Local CSI Top-K Retrieval: Given $\mathcal { H } _ { t } ^ { \mathrm { h i s t } }$ , a total of K reference windows are retrieved from the two stores. A balancing factor $\lambda \in [ 0 , 1 ]$ determines the nominal allocation:

$$
K _ { \mathrm { G } } = \mathrm { r o u n d } ( \lambda K ) , \qquad K _ { \mathrm { L } } = K - K _ { \mathrm { G } } .\tag{8}
$$

If the Local store contains fewer than $K _ { \mathrm { L } }$ entries, all available Local entries are used and the remaining budget is assigned to the Global store. Retrieval from each store is independently performed using the proposed JSDR similarity:

$$
\boldsymbol { \mathcal { K } } _ { t } ^ { a } = \mathrm { T o p K } _ { K _ { a } } \left( \boldsymbol { \mathcal { M } } _ { a } ; \boldsymbol { S } _ { \mathrm { J S D R } } \left( \boldsymbol { \mathcal { H } } _ { t } ^ { \mathrm { h i s t } } , \cdot \right) \right) , \quad a \in \{ \mathrm { G } , \mathrm { L } \} ,\tag{9}
$$

where $\mathrm { T o p K } _ { K _ { a } }$ selects the $K _ { a }$ entries with the highest similarity scores. The retrieved knowledge is represented as

$$
{ \mathcal K } _ { t } = \left( { \mathcal K } _ { t } ^ { \mathrm { G } } , { \mathcal K } _ { t } ^ { \mathrm { L } } \right) = { \mathcal R } \left( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } ; \mathcal { M } \right) .\tag{10}
$$

## C. Joint State–Dynamics Retrieval (JSDR)

Existing retrieval-augmented forecasting methods commonly use cosine similarity or Euclidean distance to retrieve knowledge [14], [15]. However, MIMO-OFDM CSI varies complexly across time, antennas, and subcarriers, similarity measures applied directly to CSI observations may not explicitly distinguish channel states from phase and amplitude dynamics. To address this, we propose JSDR, which characterizes channel states from historical CSI, phase dynamics via normalized complex conjugate products, and amplitude dynamics via logarithmic magnitude differences. These descriptors are normalized and combined into a unified retrieval key for more informative CSI knowledge retrieval.

1) CSI State Representation: We represent historical CSI using temporally weighted real and imaginary components. Each observation is vectorized as

$$
\mathbf { x } _ { t , \ell } = \left[ \mathrm { v e c } ( \mathrm { R e } ( \mathbf { H } _ { t - L + \ell } ) ) \right] , \quad \ell = 1 , \ldots , L ,\tag{11}
$$

where $\mathrm { v e c } ( \cdot )$ follows a fixed antenna–subcarrier order. With temporal weights $r _ { \ell } = \gamma ^ { L - \ell }$ and $0 < \gamma \leq 1$ , we concatenate the weighted observations chronologically:

$$
\mathbf { k } _ { \mathrm { s } } \left( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } \right) = \left[ \sqrt { r _ { 1 } } \mathbf { x } _ { t , 1 } ^ { T } , \dots , \sqrt { r _ { L } } \mathbf { x } _ { t , L } ^ { T } \right] ^ { T } .\tag{12}
$$

The descriptor is subsequently $\ell _ { 2 }$ -normalized. Smaller γ emphasizes recent observations, whereas $\gamma = 1$ assigns uniform weights. The same representation is applied to query and stored CSI sequences.

2) Phase-Dynamics Representation: To identify CSI knowledge with similar phase dynamics while avoiding phasewrapping discontinuities, we use the normalized complex conjugate product at temporal lag ℓ:

$$
\mathbf { G } _ { t } ^ { ( \ell ) } = \frac { \mathbf { H } _ { t } \mathbf { H } _ { t - \ell } ^ { * } } { | \mathbf { H } _ { t } | | \mathbf { H } _ { t - \ell } | } ,\tag{13}
$$

where $( \cdot ) ^ { * }$ denotes complex conjugation and all operations are element-wise. For nonzero coefficients, this product retains relative phase changes while removing magnitudes. The phasedynamics descriptor is obtained through $f _ { \phi } ( \cdot )$

$$
\mathbf { k } _ { \phi } \big ( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } \big ) = f _ { \phi } \Big ( \Big \{ \mathbf { G } _ { u } ^ { ( \ell ) } \Big \} \Big ) ,\tag{14}
$$

where the collection includes valid observation pairs at the selected lags within $\mathcal { H } _ { t } ^ { \mathrm { h i s t } }$ , capturing phase dynamics over different intervals. Consistent with the CSI state representation, $f _ { \phi } ( \cdot )$ vectorizes the real and imaginary components, applies temporal weighting to emphasize recent changes, and concatenates the features in a fixed temporal, lag, and antenna– subcarrier order, followed by $\ell _ { 2 }$ normalization.

3) Amplitude-Dynamics Representation: We characterize relative amplitude dynamics using logarithmic magnitude differences between consecutive observations:

$$
\mathbf { D } _ { t } = \log \left| \mathbf { H } _ { t } \right| - \log \left| \mathbf { H } _ { t - 1 } \right| ,\tag{15}
$$

where all operations are element-wise. This representation captures relative amplitude changes while reducing sensitivity to absolute channel magnitudes.

The amplitude-dynamics descriptor is

$$
\mathbf { k } _ { a } \big ( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } \big ) = f _ { a } \big ( \{ \mathbf { D } _ { u } \} \big ) ,\tag{16}
$$

where $f _ { a } ( \cdot )$ applies temporal weighting, vectorizes and concatenates the amplitude changes in temporal and antenna– subcarrier order, and performs $\ell _ { 2 }$ normalization.

4) Joint State–Dynamics Retrieval Key: We combine the CSI state, phase-dynamics, and amplitude-dynamics descriptors into a unified retrieval key:

$$
\mathbf { k } \left( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } \right) = \mathrm { L 2 N o r m } \left( \left[ \sqrt { \omega _ { s } } \mathbf { k } _ { s } ; \sqrt { \omega _ { \phi } } \mathbf { k } _ { \phi } ; \sqrt { \omega _ { a } } \mathbf { k } _ { a } \right] \right) ,\tag{17}
$$

where $\omega _ { s } , \omega _ { \phi } , \omega _ { a } \geq 0$ sum to one and control the contributions of the three descriptors. For a stored CSI sequence $\mathcal { H } _ { i } ^ { \mathrm { r e f } }$ , the retrieval similarity is computed as

$$
\begin{array} { r } { S _ { \mathrm { J S D R } } \left( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } , \mathcal { H } _ { i } ^ { \mathrm { r e f } } \right) = \mathbf { k } \left( \mathcal { H } _ { t } ^ { \mathrm { h i s t } } \right) ^ { T } \mathbf { k } \left( \mathcal { H } _ { i } ^ { \mathrm { r e f } } \right) . } \end{array}\tag{18}
$$

This similarity guides Global–Local Top-K retrieval based on joint channel-state and dynamics information.

## D. Retrieval-Augmented Prediction

We use a lightweight Transformer backbone for CSI prediction. Cross-attention fuses features of the current CSI history with representations of the retrieved sequences, producing a retrieval-enhanced historical representation. The backbone models temporal dependencies from this enriched context and generates features for future channel states. A linear projection followed by reshaping produces the real and imaginary components of the predicted CSI sequence $\widehat { \mathcal { H } } _ { t } ^ { \mathrm { f u t } } \in \mathbb { C } ^ { \breve { Q } \times N _ { t } \times N _ { c } }$

TABLE I: Dataset properties and experimental settings.
<table><tr><td>Properties</td><td>Value</td></tr><tr><td>Number of Scenes</td><td>10</td></tr><tr><td>Number of antennas</td><td>8</td></tr><tr><td>Antenna array form</td><td>ULA</td></tr><tr><td>Number of subcarriers</td><td>8</td></tr><tr><td>Number of paths</td><td>25</td></tr><tr><td>Carrier frequency</td><td>3.5 GHz</td></tr><tr><td>Bandwidth</td><td>20 MHz</td></tr><tr><td>Historical CSI sequence length</td><td>12</td></tr><tr><td>Prediction horizon</td><td>1</td></tr></table>

## IV. EXPERIMENTS

## A. Experimental settings

We evaluate RAG-CSI on a multi-scene CSI dataset generated using the open-source DeepMIMO framework [13]. DeepMIMO provides ray-tracing-based channel data from realistic wireless environments, supporting the evaluation of CSI prediction across different propagation scenes. Using DeepMIMO v4 scenario configurations, we generate MIMO-OFDM channel samples from 10 scenes, including ASU Campus, New York, Miami, and San Diego. Within each scene, we simulate user trajectories and assemble the corresponding channel observations into CSI sequences. The historical sequence length is set to 12, and the prediction horizon is one step. For retrieval, the default number of retrieved CSI sequences is set to K = 10, and the Global–Local balancing factor is set to $\lambda = 0 . 5$ . The generated sequences are divided into training and test sets at a ratio of 8:2. The detailed dataset properties and channel settings are summarized in Table I.

## B. Baselines and Evaluation Metrics

We compare RAG-CSI with six representative baselines:

LSTM [8]: A recurrent predictor that uses LSTM layers to learn temporal dependencies from historical CSI for future channel prediction.

LSTM-GRU [16]: A hybrid architecture comprising cascaded LSTM and GRU blocks for CSI prediction.

Transformer [9]: A transformer-based prediction framework for parallel time-domain channel prediction.

RCDNet [17]: A recurrent channel-deduction network that uses channel mixing to model spatial-frequency correlations and an LSTM to capture temporal correlations.

WiFo [12]: A wireless foundation model pretrained through masked CSI reconstruction across space, time, and frequency, and applied to channel prediction.

PromptFM [18]: A prompt-guided foundation model designed for CSI modeling across multiple frequency bands and evaluated here on CSI prediction.

We evaluate prediction performance using mean squared error (MSE), normalized mean squared error (NMSE), mean absolute error (MAE), normalized mean absolute error (NMAE), and cosine correlation $\rho$ [17]. Lower values of the four error metrics and a higher $\rho$ indicate better prediction performance.

TABLE II: Overall CSI prediction performance comparison.
<table><tr><td>Method</td><td>MSE↓</td><td>NMSE↓</td><td>MAE↓</td><td>NMAE↓</td><td> $\rho \uparrow$ </td></tr><tr><td>LSTM [8]</td><td>0.8301</td><td>0.7395</td><td>0.6883</td><td>0.7863</td><td>0.7115</td></tr><tr><td>LSTM-GRU [16]</td><td>1.1752</td><td>1.0031</td><td>0.8915</td><td>1.0013</td><td>0.3614</td></tr><tr><td>Transformer [9]</td><td>0.8150</td><td>0.7296</td><td>0.6765</td><td>0.7756</td><td>0.6776</td></tr><tr><td>RCDNet [17]</td><td>0.8128</td><td>0.7396</td><td>0.6773</td><td>0.7793</td><td>0.6982</td></tr><tr><td>WiFO [12]</td><td>1.0039</td><td>0.8727</td><td>0.7941</td><td>0.8996</td><td>0.5047</td></tr><tr><td>PromptFM [18]</td><td>0.9850</td><td>0.8602</td><td>0.7785</td><td>0.8812</td><td>0.4131</td></tr><tr><td>RAG-CSI</td><td>0.6961</td><td>0.6172</td><td>0.6051</td><td>0.6970</td><td>0.7949</td></tr><tr><td>Improv.</td><td>14.36%</td><td>15.42%</td><td>10.56%</td><td>10.13%</td><td>11.72%</td></tr></table>

## C. Overall Performance

Table II summarizes the overall CSI prediction performance. RAG-CSI achieves the best results across all five metrics. Compared with the best baseline for each metric, RAG-CSI reduces MSE, NMSE, MAE, and NMAE by 14.36%, 15.42%, 10.56%, and 10.13%, respectively, while improving cosine correlation $\rho$ by 11.72%. The reductions in both absolute and normalized errors indicate more accurate CSI prediction, while the higher $\rho$ reflects better alignment between the predicted and ground-truth channel vectors. These results support the effectiveness of augmenting historical observations with relevant channel knowledge.

## D. Performance Across Scenes

Table III compares CSI prediction across four different scenes. RAG-CSI consistently achieves the lowest NMSE and NMAE and the highest $\rho$ across all four scenes. Compared with the best baseline for each scene and metric, it reduces NMSE by 4.17%–12.74% and NMAE by 4.93%–7.76%, while improving ρ by 4.06%–12.06%, with the largest gains observed in Miami. These results demonstrate the effectiveness of RAG-CSI across diverse propagation environments.

## E. Ablation Study

We ablate retrieval augmentation and JSDR using two variants: w/o RAG, which removes CSI sequence retrieval and uses only the Transformer-based backbone for prediction, and w/o JSDR, which replaces JSDR with cosine correlation for CSI retrieval. As shown in Table IV, removing RAG increases NMSE from 0.6172 to 0.7344 and reduces $\rho$ from 0.7949 to 0.6351, confirming the benefit of retrieved CSI knowledge. Replacing JSDR also degrades performance, with NMSE increasing to 0.6783 and $\rho$ decreasing to 0.6844, demonstrating the importance of incorporating channel states together with phase and amplitude dynamics.

## F. Hyperparameter Sensitivity

We evaluate the sensitivity of RAG-CSI to the retrieval size K and the Global–Local balancing factor λ. Fig. 3(a) shows the results for $K \in \{ 2 , 4 , 6 , 8 , 1 0 , 2 0 \}$ with $\lambda = 0 . 5$ NMSE decreases markedly as K increases to $^ { 8 , }$ while further increasing K brings only marginal gains. This indicates that a moderate number of retrieved sequences captures most of the benefit. Fig. 3(b) shows the results for varying λ with $K = 1 0$ The best performance is achieved at $\lambda = 0 . 5 .$ , while shifting the allocation toward either Global or Local knowledge increases the error. These results support jointly using Global and Local knowledge, balancing access to diverse channel patterns with relevance to the current scene.

![](images/9bab3ca449df006eaa66b3314e3c110fcf3d85041937c1331a3791fac4e3294c.jpg)  
(a)

![](images/035170666de009fc9d7d8d1b5f1069f0e94e488960251ac77f1ea35b0262a8f8.jpg)  
(b)  
Fig. 3: Hyperparameter sensitivity: (a) retrieval size K with $\lambda = 0 . 5 ;$ (b) Global–Local balancing factor λ with $K = 1 0$

## G. Complexity Analysis

Table V compares parameter counts and average inference times per CSI sequence. RAG-CSI has the fewest parameters (0.451 million), 50.39% fewer than the Transformer baseline. Its average inference time is 0.3242 ms per sequence, 65.72% lower than that of the Transformer baseline and the third shortest overall, behind only LSTM and LSTM-GRU. Together with the accuracy gains above, these results demonstrate that RAG-CSI combines a compact prediction network with competitive inference efficiency.

## V. RELATED WORK

CSI prediction exploits temporal correlations to mitigate channel aging and reduce pilot overhead. Early methods used linear prediction [19] or combined autoregression with principal component analysis [20], but predefined models may struggle with complex channel dynamics. Deep learning enables nonlinear CSI modeling [21]. LSTM and GRU were explored for fading-channel prediction [8], while Transformers support parallel multi-step prediction [9]. More recently, WiFo [12] used masked CSI reconstruction for prediction across channel configurations. However, these methods primarily encode channel knowledge in fixed model parameters, increasing demands on model capacity.

Retrieval-Augmented Generation (RAG) complements parametric models with knowledge retrieved from external memory, enabling access to information beyond model parameters. Initially developed for knowledge-intensive NLP tasks [22], RAG has recently been extended to spatiotemporal forecasting [14], where retrieved historical patterns provide additional prediction context. However, retrieval augmentation remains largely unexplored for CSI prediction, motivating the use of historical CSI knowledge to capture diverse channel dynamics without relying solely on model parameters.

## VI. CONCLUSION

This paper proposed RAG-CSI, a retrieval-enhanced Transformer framework for CSI prediction across diverse propagation scenes. RAG-CSI combines a Global–Local CSI knowledge store, JSDR-based retrieval, and cross-attention fusion to exploit relevant historical channel knowledge beyond the current observation window. JSDR jointly captures channel states, phase dynamics, and amplitude dynamics to retrieve informative CSI sequences. Experiments on a multi-scene CSI dataset showed improvements exceeding 10% over the best baselines across all five evaluation metrics, demonstrating the effectiveness of retrieval augmentation for CSI prediction.

TABLE III: CSI prediction performance across different scenes.
<table><tr><td>Scenes</td><td colspan="3">ASU Campus</td><td colspan="3">New York</td><td colspan="3">Miami</td><td colspan="3">San Diego</td></tr><tr><td>Methods</td><td>NMSE↓</td><td>NMAE↓</td><td> $\rho \uparrow$ </td><td>NMSE↓</td><td>NMAE↓</td><td> $\rho \uparrow$ </td><td>NMSE↓</td><td>NMAE↓</td><td> $\rho \uparrow$ </td><td>NMSE↓</td><td>NMAE↓</td><td> $\rho \uparrow$ </td></tr><tr><td>LSTM [8]</td><td>0.6528</td><td>0.7112</td><td>0.7904</td><td>0.6939</td><td>0.7921</td><td>0.7299</td><td>1.0133</td><td>1.0054</td><td>0.4542</td><td>0.6038</td><td>0.7186</td><td>0.7240</td></tr><tr><td>LSTM-GRU [16]</td><td>1.0043</td><td>1.0020</td><td>0.2954</td><td>0.9995</td><td>0.9999</td><td>0.3053</td><td>1.0008</td><td>0.9989</td><td>0.6889</td><td>1.0006</td><td>1.0005</td><td>0.3005</td></tr><tr><td>Transformer [9]</td><td>0.6745</td><td>0.7339</td><td>0.7202</td><td>0.7778</td><td>0.8283</td><td>0.6573</td><td>0.7586</td><td>0.7800</td><td>0.6684</td><td>0.6466</td><td>0.7492</td><td>0.6493</td></tr><tr><td>RCDNet [17]</td><td>0.7012</td><td>0.7462</td><td>0.7260</td><td>0.7018</td><td>0.7826</td><td>0.6887</td><td>0.8110</td><td>0.8394</td><td>0.6750</td><td>0.6558</td><td>0.7490</td><td>0.6939</td></tr><tr><td>WiFO [12]</td><td>0.8435</td><td>0.8713</td><td>0.5763</td><td>0.8486</td><td>0.9016</td><td>0.4397</td><td>1.0035</td><td>1.0016</td><td>0.2779</td><td>0.7508</td><td>0.8319</td><td>0.6177</td></tr><tr><td>PromptFM [18]</td><td>0.8277</td><td>0.8494</td><td>0.4547</td><td>0.8163</td><td>0.8852</td><td>0.4716</td><td>1.0031</td><td>1.0005</td><td>0.0857</td><td>0.6718</td><td>0.7749</td><td>0.6620</td></tr><tr><td>RAG-CSI</td><td>0.5819</td><td>0.6717</td><td>0.8225</td><td>0.6650</td><td>0.7440</td><td>0.8100</td><td>0.6620</td><td>0.7195</td><td>0.7719</td><td>0.5570</td><td>0.6825</td><td>0.7677</td></tr><tr><td>Improv.</td><td>10.86%</td><td>5.55%</td><td>4.06%</td><td>4.17%</td><td>4.93%</td><td>10.97%</td><td>12.74%</td><td>7.76%</td><td>12.06%</td><td>7.75%</td><td>5.02%</td><td>6.03%</td></tr></table>

TABLE IV: Ablation study of RAG-CSI.
<table><tr><td>Method</td><td>MSE↓</td><td> $\mathbf { N M S E } \downarrow$ </td><td> $\mathbf { M A E } \downarrow$ </td><td>NMAE↓</td><td> $\rho \uparrow$ </td></tr><tr><td>w/o JSDR</td><td>0.7910</td><td>0.6783</td><td>0.6648</td><td>0.7563</td><td>0.6844</td></tr><tr><td>w/o RAG</td><td>0.8388</td><td>0.7344</td><td>0.6970</td><td>0.7943</td><td>0.6351</td></tr><tr><td>RAG-CSI</td><td>0.6961</td><td>0.6172</td><td>0.6051</td><td>0.6970</td><td>0.7949</td></tr></table>

TABLE V: Model size and inference time comparison.
<table><tr><td>Method</td><td>Params (M)</td><td>Inference Time (ms)</td></tr><tr><td>LSTM [8]</td><td>6.435</td><td>0.1422</td></tr><tr><td>LSTM-GRU [16]</td><td>1.744</td><td>0.1718</td></tr><tr><td>Transformer [9]</td><td>0.909</td><td>0.9457</td></tr><tr><td>RCDNet [17]</td><td>4.341</td><td>0.3791</td></tr><tr><td>WiFO [12]</td><td>21.326</td><td>0.4114</td></tr><tr><td>PromptFM [18]</td><td>21.598</td><td>0.5042</td></tr><tr><td>RAG-CSI</td><td>0.451</td><td>0.3242</td></tr></table>

## REFERENCES

[1] J. Guo, Y. Cui, C.-K. Wen, and S. Jin, “Prompt-enabled large ai models for CSI feedback,” IEEE Journal on Selected Areas in Communications, 2025.

[2] J. Guo, C.-K. Wen, S. Jin, and G. Y. Li, “Overview of deep learningbased CSI feedback in massive MIMO systems,” IEEE Transactions on Communications, vol. 70, no. 12, pp. 8017–8045, 2022.

[3] C. Zhou, X. Jiang, T. Li, Z. Xiao, and J. Zhang, “3dckm: A large-scale multiband 3d channel knowledge map dataset for low-altitude corridors,” in 2026 IEEE International Conference on Communications Workshops (ICC Workshops). IEEE, 2026, pp. 1–6.

[4] X. Jiang, T. Li, Z. Xiao, K. Chen, A. C. L. Tai, Z. Tang, and K. Li, “Pihg: Physics-inspired heterogeneous graph neural networks for multi-band radio map prediction,” IEEE Transactions on Wireless Communications, 2026.

[5] M. Sattari, J. Aliakbari, A. G. i Amat, and T. Svensson, “CSI prediction using diffusion models,” IEEE Transactions on Wireless Communications, vol. 25, pp. 21 418–21 432, 2026.

[6] T. Zhou, X. Liu, Z. Xiang, H. Zhang, B. Ai, L. Liu, and X. Jing, “Transformer network based channel prediction for csi feedback enhancement in ai-native air interface,” IEEE Transactions on Wireless Communications, vol. 23, no. 9, pp. 11 154–11 167, 2024.

[7] Y. Gao, Z. Lu, X. Wu, W. Yu, S. Liu, J. Du, Y. Jin, S. Zhang, X. Chu, and S. Xu, “AI-driven channel state information (CSI) extrapolation for 6G: Current situations, challenges and future research,” IEEE communications surveys & tutorials, 2026.

[8] W. Jiang and H. D. Schotten, “Deep learning for fading channel prediction,” IEEE Open Journal of the Communications Society, vol. 1, pp. 320–332, 2020.

[9] H. Jiang, M. Cui, D. W. K. Ng, and L. Dai, “Accurate channel prediction based on transformer: Making mobility negligible,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2717–2732, 2022.

[10] D. Luan and J. S. Thompson, “Channelformer: Attention based neural solution for wireless channel estimation and effective online training,” IEEE Transactions on Wireless Communications, vol. 22, no. 10, pp. 6562–6577, 2023.

[11] Y. Jin, Y. Wu, Y. Gao, S. Zhang, S. Xu, and C.-X. Wang, “LinFormer: A linear-based lightweight transformer architecture for time-aware mimo channel prediction,” IEEE Transactions on Wireless Communications, vol. 24, no. 9, pp. 7177–7190, 2025.

[12] B. Liu, S. Gao, X. Liu, X. Cheng, and L. Yang, “Wifo: Wireless foundation model for channel prediction,” Science China Information Sciences, vol. 68, no. 6, p. 162302, 2025.

[13] A. Alkhateeb, “DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications,” in Proceedings of the Information Theory and Applications Workshop (ITA), 2019, pp. 1–8. [Online]. Available: https://arxiv.org/abs/1902.06435

[14] W. Ruan, X. Dang, Z. Zhou, S. Lyu, and Y. Liang, “A retrieval augmented spatio-temporal framework for traffic prediction,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 46, 2026, pp. 39 163–39 172.

[15] S. Ma, Z. Zhou, Q. Huang, B. Wang, and Y. Wang, “QuiZSF: A retrieval-augmented framework for zero-shot time series forecasting,” in Proceedings of the ACM Web Conference 2026, 2026, pp. 1947–1958.

[16] I. Helmy, P. Tarafder, and W. Choi, “LSTM-GRU model-based channel prediction for one-bit massive mimo system,” IEEE Transactions on Vehicular Technology, vol. 72, no. 8, pp. 11 053–11 057, 2023.

[17] Z. Chen, Z. Zhang, Z. Yang, C. Huang, and M. Debbah, “Channel deduction: A new learning framework to acquire channel from outdated samples and coarse estimate,” IEEE Journal on Selected Areas in Communications, vol. 43, no. 3, pp. 944–958, 2025.

[18] Z. Huang, T. Li, X. Lv, H. Rui, and Y. Li, “A universal prompt-guided foundation model for multi-band channel state information modeling,” in ICC 2026-IEEE International Conference on Communications. IEEE, 2026, pp. 1–6.

[19] K. E. Baddour and N. C. Beaulieu, “Improved pilot-assisted prediction of unknown time-selective rayleigh channels,” in Proceedings of the IEEE International Conference on Communications (ICC), vol. 11, 2006, pp. 5192–5199.

[20] R. Nagashima, T. Ohtsuki, W. Jiang, Y. Takatori, and T. Nakagawa, “Channel prediction for massive MIMO with channel compression based on principal component analysis,” in Proceedings of the IEEE 27th Annual International Symposium on Personal, Indoor, and Mobile Radio Communications (PIMRC), 2016, pp. 1–6.

[21] X. Jiang, T. Li, Z. Xiao, K. Chen, S. Ma, Z. Wang, and K. Li, “Unirm: A universal large model for multiband 3d radio map construction,” IEEE Journal on Selected Areas in Communications, 2025.

[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel, S. Riedel, and D. Kiela,¨ “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474.