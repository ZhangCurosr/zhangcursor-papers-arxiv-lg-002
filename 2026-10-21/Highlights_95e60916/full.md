## Highlights

Foundation-Style Pretraining for Scalable and Transferable Distributed PV Forecasting Dayin Chen,Xiaodan Shi,Xin Xu,Zhongyue Su,Shibo Zhu,Ruizhe Deng,Dongxiao Zhang,Yuntian Chen,Jinyue Yan

• A shared pretrained model enables zero-shot forecasting at unseen PV sites.

• NWP-assisted PV forecasting is formulated as a sequence-completion task.

• Three pretraining objectives are evaluated; mixed reconstruction performs best.

• Multi-site pretraining outperforms station-specific and generalist baselines.

• The pretrained model adapts to region-specific operational objectives.

# Foundation-Style Pretraining for Scalable and Transferable Distributed PV Forecasting

Dayin Chen<sup>a,b,c</sup>, Xiaodan Shi<sup>d,∗</sup>, Xin Xu<sup>a,b,c</sup>, Zhongyue Su<sup>e</sup>, Shibo Zhu<sup>a,b,c</sup>, Ruizhe Deng<sup>a,b,c</sup>, Dongxiao Zhang<sup>e,∗∗</sup>, Yuntian Chen<sup>e</sup> and Jinyue Yan<sup>a,b,∗∗∗</sup>

<sup>a</sup>Department ofBuilding Environment and Energy Engineering, The Hong Kong Polytechnic University, Hong Kong SAR, China

<sup>b</sup>International Centre of Urban Energy Nexus, The Hong Kong Polytechnic University, Hung Hom, Kowloon, Hong Kong SAR, China

<sup>c</sup>Ningbo Institute ofDigital Twin, Eastern Institute ofTechnology, Ningbo, 315200, China

<sup>d</sup>Department ofComputer and Systems Sciences, Stockholm University, Stockholm, Sweden

<sup>e</sup>Zhejiang Key Laboratory ofIndustrial Intelligence and Digital Twin, Ningbo Institute ofDigital Twin, Eastern Institute of Technology, Ningbo, 315200, China

## A R T I C L E I N F O

Keywords:   
Photovoltaic power forecasting   
Distributed PV systems   
Foundation-style pretraining   
Multi-horizon forecasting   
Zero-shot transfer

## A BS T RA C T

The rapid growth of distributed photovoltaic (PV) generation is increasing the demand for accurate forecasting across large numbers of geographically dispersed stations. Existing approaches are commonly developed for individual sites or specific forecasting horizons, requiring repeated training and maintenance as new stations are integrated and limiting direct reuse at previously unseen sites. To address this scalability gap, this paper presents a domain-specific foundation-style pretraining framework that learns a shared forecasting model from heterogeneous multi-site and multi-horizon PV data. NWP-assisted forecasting is formulated as a sequence-completion problem that unifies historical PV power, meteorological observations, future numerical weather prediction (NWP) forecasts, and time encodings. Three pretraining objectives are designed and systematically evaluated, including forecast-aligned reconstruction, generic reconstruction, and mixed reconstruction. The mixed objective combines the operational structure of PV forecasting with random sequence reconstruction and provides the most balanced performance across forecasting accuracy, degraded NWP conditions, and downstream adaptation. Experiments on 479 distributed PV stations use 340 stations for model development and 139 spatially held-out stations for zero-shot evaluation. Across horizons from 6 hours to 7 days, the multi-site pretrained MR model outperforms representative station-specific and shared baselines, as well as generalist time-series models, including a recent large-scale pretrained foundation model. The pretrained model can also be directly deployed at new PV stations and subsequently adapted to region-specific operational objectives when local data become available, providing a scalable forecasting paradigm for large distributed PV fleets.

## 1. Introduction

Solar photovoltaic (PV) deployment is expanding rapidly while becoming increasingly geographically distributed. Global cumulative PV capacity approached 3 TW by the end of 2025, with approximately 698 GW newly installed during the year [1]. In China, distributed PV alone added 153 GW in 2025 and reached a cumulative installed capacity of 530 GW [2]. Unlike utility-scale PV plants, which are typically developed and monitored as spatially concentrated generation units, distributed PV consists of numerous userside systems deployed across residential, commercial, and industrial settings [3–5]. These geographically dispersed systems are heterogeneous in installed capacity, operating conditions, and local meteorological environments, resulting in substantial diversity in their generation characteristics [6]. As distributed PV continues to penetrate modern power systems, accurate forecasting of such heterogeneous resources becomes increasingly important for power-system dispatch, day-ahead trading, and balancing [7, 8].

The rapid growth in the number of distributed PV sites also creates a scalability challenge for conventional forecasting pipelines. In many practical applications, forecasting models are still developed and maintained separately for individual stations and prediction horizons [6]. Such stationspecific and horizon-specific modeling requires repeated development and maintenance. This efort increases as new PV systems are connected and forecasting requirements change. Transfer learning and multi-site forecasting have therefore been investigated to enable knowledge sharing across PV systems and reduce dependence on site-specific model development [9–11]. Despite these advances, several limitations remain. Transfer-learning approaches often require adaptation to individual target stations [9, 12, 13], while newly commissioned PV systems may provide only limited local historical data [8]. Multi-site forecasting methods, in contrast, are commonly developed for a predefined group of stations [10, 11, 14], limiting their direct applicability to previously unseen sites. Moreover, both paradigms are often designed for specific forecasting horizons, meaning that separate models or additional customization may still be required when operational forecasting requirements change. Taken together, these limitations constrain model reuse across heterogeneous and unseen stations and forecasting horizons, motivating a shared pretrained forecasting framework.

![](images/b5f2c9c51e524e18ea0972dfbf4a79a215946203f744124a852a04bf69151ed3.jpg)  
Figure 1: Overview of the proposed foundation-style pretraining framework for scalable distributed PV forecasting. A shared model is pretrained across multiple PV sites and forecasting horizons using a sequence-completion formulation and reconstruction-based objectives, enabling zero-shot transfer to unseen stations, robustness to degraded NWP availability, and downstream adaptation to region-specific operational objectives.

Recent advances in pretrained time-series models and foundation-model paradigms have demonstrated the potential of learning reusable temporal representations within a shared backbone [15–17]. Instead of repeatedly training independent models for individual forecasting tasks, such a paradigm enables common temporal patterns to be learned from heterogeneous time series and subsequently reused across datasets, forecasting horizons, and downstream applications. This idea is particularly attractive for large-scale distributed PV systems, where a shared pretrained model could potentially serve many geographically dispersed stations and multiple operational forecasting horizons. In this work, foundation-style pretraining refers to a domain-specific pretrain–generalize–adapt paradigm, where a shared forecasting backbone learns reusable representations across multiple PV sites and forecasting horizons.

However, PV forecasting has a distinct operational information structure. At the forecast origin, historical PV generation and meteorological records are available, while future PV generation remains unknown and NWP forecasts over the prediction horizon are already accessible [18]. In this study, these variables are organized according to their forecast-time availability, and NWP-assisted PV forecasting is formulated as a sequence-completion problem. Historical observations and known-future NWP retain their original temporal positions within a unified multivariate sequence, while future PV generation is treated as the missing target. This formulation preserves the temporal alignment among variables and provides a common basis for shared pretraining across multiple PV stations and forecasting horizons.

Building on this formulation, we develop a foundationstyle pretraining framework for scalable and transferable distributed PV forecasting. A shared forecasting backbone is trained on variable-horizon samples, allowing a single set of model parameters to support multiple operational forecasting horizons. During pretraining, a mixed reconstruction objective combines forecasting-oriented supervision with auxiliary masked reconstruction to improve cross-site and multi-horizon transferability. The pretrained model can then be directly deployed at unseen PV stations without stationspecific retraining, while the pretrained model can be further adapted when specific operational objectives are considered. The overall framework is illustrated in Fig. 1.

The framework is evaluated on a real-world dataset containing 479 distributed PV stations in Ningbo, China, integrating measured PV generation, weather reanalysis data, and rolling NWP forecasts. A strict spatiotemporal evaluation protocol is adopted, in which the shared model is developed using 340 stations and evaluated on 139 previously unseen stations during a temporally separated test period. Experiments cover four forecasting horizons, including 6 hours, 1 day, 3 days, and 7 days. Beyond standard forecasting accuracy, the evaluation examines zero-shot crossstation transfer, robustness under degraded NWP availability, and adaptation to region-specific operational criteria. The results demonstrate competitive forecasting accuracy while enabling a single shared model to be reused across unseen stations and multiple forecasting horizons.

The main contributions of this work are summarized as follows:

• We develop a foundation-style pretraining framework for scalable distributed PV forecasting, where a single shared model is pretrained across multiple PV sites and forecasting horizons and directly applied to previously unseen stations without station-specific retraining.

• We formulate NWP-assisted PV forecasting as a sequence-completion task and design three pretraining objectives. Mixed reconstruction achieves the best balance between forecasting accuracy and robustness to degraded NWP availability.

• We conduct a large-scale evaluation on 479 distributed PV stations under a strict spatiotemporal protocol, including 139 spatially held-out stations. The experiments cover multi-horizon zero-shot forecasting, cross-station generalization, NWP robustness, and downstream adaptation to region-specific operational objectives.

## 2. Related Work

## 2.1. PV Forecasting with Weather Information

Weather information has long played a central role in PV power forecasting. Early studies commonly formulated the problem as an NWP-driven regression task, where forecasted irradiance, temperature, humidity, and cloud cover were used as major predictive inputs. Luo et al. developed a physics-constrained LSTM for day-ahead PV forecasting using forecasted meteorological variables [9, 19]. Böök and Lindfors proposed a site-specific post-processing framework that combined weather forecasts, a physical PV model, and rolling historical-output correction to reduce systematic bias [20]. Mayer et al. converted European Centre for Medium-Range Weather Forecasts (ECMWF) and Applications of Research to Operations at Mesoscale (AROME) forecasts into PV output through a physical model chain for short-term and day-ahead prediction [21]. Brester et al. replaced historical NWP archives with weather observations during model development and subsequently validated the trained model using operational forecasts, thereby reducing dependence on archived forecast products [22]. These studies established the importance of future meteorological information for operational PV forecasting.

With the development of machine learning and deep learning, another line of research has focused on extracting temporal patterns from historical PV generation and multivariate observations. Qu et al. proposed an adversarial domain-adaptation framework for cross-regional distributed PV forecasting under limited target-site data availability [23]. Wang et al. developed RCLformer by integrating Crossformer, LSTM, and RevIN for long-horizon forecasting under non-stationary conditions [24]. Liu and Gai introduced a patch-based MLP for long-sequence PV forecasting using historical multivariate observations [25], while Zhu et al. combined LSTM and XGBoost to capture temporal dependencies and nonlinear relationships in long-term plant measurements [26]. Such history-driven approaches efectively exploit temporal persistence and local operating patterns. However, their predictive information becomes increasingly limited as the forecasting horizon extends.

More recent studies have therefore combined historical observations with future NWP information. Representative approaches include temporal alignment between historical power sequences and future weather forecasts [27], direct feature fusion of NWP variables with historical inputs [28], and dedicated encoder–decoder structures for meteorological covariates [29]. Overall, existing studies provide efective mechanisms for exploiting future meteorological information. In contrast to treating NWP primarily as an additional exogenous input, the present work organizes historical observations, known-future NWP, and unknown future PV within their original temporal positions and formulates forecasting as a sequence-completion task. This formulation is subsequently used as the basis for shared pretraining rather than as an isolated NWP-fusion mechanism.

## 2.2. Multi-site and Transferable PV Forecasting

The increasing number of distributed PV systems has motivated forecasting methods that share information across sites rather than training independent models for every station. Two major directions have emerged: transfer learning for adapting knowledge to target sites and multi-site forecasting for jointly modeling groups of spatially distributed PV systems.

Transfer-learning approaches aim to reduce the dependence on target-site training data by reusing representations or model parameters learned from related PV systems. Qu et al. investigated adversarial domain adaptation for crossregional distributed PV forecasting under limited local data [23]. Tang et al. transferred a pretrained forecasting model to newly built PV sites through parameter fine-tuning [12], while Sarmas et al. systematically examined several weighttransfer and fine-tuning strategies for solar power forecasting under data scarcity [13]. These studies demonstrate the value of knowledge transfer for newly commissioned or datalimited PV systems. Nevertheless, deployment generally still requires target-specific adaptation or access to a certain amount of target-site data.

Multi-site forecasting methods instead exploit information from multiple PV systems during model training. Wen et al. developed RAD-FEDformer, which jointly trains data from multiple PV sites and incorporates geographical correlations through a relevance-amplification module [10]. Simeunovic et al. proposed spatio-temporal graph neural network models, including GCLSTM and GCTrafo, to capture spatial dependencies among hundreds of PV systems [11]. Yang et al. introduced DEST-GNN to model dynamic inter-site relationships through adaptive graph learning [14]. Other studies have explored ensemble-based and multiobjective learning frameworks for improved cross-site generalization [30], while difusion-based data augmentation has been investigated to increase robustness under heterogeneous site conditions [31]. Many of these approaches are constructed around a predefined collection of stations or rely on explicit spatial modeling and additional taskspecific modules. Their direct reuse can therefore become less straightforward when new stations are introduced or the forecasting configuration changes.

These limitations motivate a complementary direction based on parameter sharing, where a common forecasting model can be learned from heterogeneous PV systems and reused across stations and operational forecasting horizons.

## 2.3. Pretrained and Generalist Time-Series Forecasting

Recent advances in large-scale time-series modeling have shifted forecasting from task-specific models toward pretrained models with reusable temporal representations. Chronos [15] has become a representative general-purpose foundation-model baseline, while more recent models such as TimeMoE [16] and the TimesFM family [32] have further advanced large-scale pretrained forecasting. The latest TimesFM-3 [17] further advances this line of largescale foundation forecasting. Similar developments have also appeared in weather and climate forecasting [33]. These models provide an important reference for distributed PV forecasting, but their pretraining is designed to capture broadly transferable patterns across diverse domains rather than characteristics specific to distributed PV fleets. A key challenge therefore remains to learn a shared PV forecasting model from multiple stations and forecasting horizons that can be directly reused at unseen sites. This work addresses this challenge through a domain-specific foundation-style pretraining paradigm tailored to distributed PV forecasting.

## 3. Methodology

## 3.1. Shared Pretraining Setting

We consider a fleet of distributed PV stations with heterogeneous generation characteristics and multiple operational forecasting horizons. Let $S _ { \mathrm { t r } }$ denote the set of stations available during pretraining and  denote the set of forecasting horizons. For each station $i \in S _ { \mathrm { t r } }$ and horizon $h \in \mathcal H$ , forecasting samples are constructed from the corresponding PV and meteorological data. Instead of training separate station-specific or horizon-specific models, a single forecasting model $f _ { \theta }$ with shared parameters � is optimized jointly over samples from diferent stations and horizons.

Let $\mathcal { D } _ { i , h }$ denote the set of training samples associated with station � and forecasting horizon ℎ. The shared model is learned through

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \sum _ { i \in S _ { \mathrm { t r } } } \sum _ { h \in \mathcal { H } } \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim \mathcal { D } _ { i , h } } \left[ \mathcal { L } _ { \mathrm { p r e } } \left( f _ { \theta } ( \mathbf { x } ) , \mathbf { y } \right) \right]\tag{1}
$$

where $\mathcal { L } _ { \mathrm { p r e } }$ denotes the reconstruction-based pretraining objective introduced in Sec. 3.5. The same parameter set is shared across all training stations and forecasting horizons. No station identifiers, geographical embeddings, or horizonspecific embeddings are provided to the model. The shared backbone therefore remains independent of any predefined station set or spatial topology and learns station and horizon variations directly from the input sequences. This stationagnostic design enables the same pretrained model to be applied to unseen PV stations without introducing stationspecific parameters or modifying the model architecture.

Within this setting, foundation-style pretraining refers to the joint learning of a reusable base model from heterogeneous multi-site and multi-horizon data. The pretrained model is then directly applied to a disjoint set of PV stations that are never observed during pretraining. For an unseen station $j \not \in S _ { \mathrm { t r } } ,$ , the learned parameters $\theta ^ { * }$ are used without station-specific retraining, forming the zero-shot deployment setting considered in this study.

Variables included in the multivariate forecasting sequence, grouped into four categories. ERA5 weather reanalysis is used as a proxy for meteorological observations (OBS).
<table><tr><td>Category</td><td>Variables</td></tr><tr><td>PV power Meteorological</td><td>PV power output (elec) Relative humidity (rh); 10-m wind</td></tr><tr><td>observations (OBS)</td><td>components  $( u _ { 1 0 } , ~ v _ { 1 0 } ) ;$  2-m air tem- perature  $\left( t _ { 2 m } \right) ;$  surface solar radia- tion downwards (ssrd); surface ther- mal radiation downwards (strd); top- of-atmosphere solar radiation (tsr); to- tal cloud cover (tcc)</td></tr><tr><td>Numerical weather prediction (NWP)</td><td>Forecasted global horizontal irradiance (ghi_f); surface pressure (sp_f); rel- ative humidity (rh f); 2-m air tem- perature  $( \mathsf { t } 2 \mathsf { m \_ f } )$  ;10-m wind direction (wd10_f); 10-m wind speed (ws10_f); total cloud cover (tcc_f)</td></tr><tr><td>Time encoding (TE)</td><td>Hour of day (sine, cosine); minute of hour (sine, cosine); day of year (sine, cosine)</td></tr></table>

Beyond zero-shot deployment, the pretrained model can also serve as an initialization for downstream adaptation when target-station data and specific operational objectives are available. The overall framework therefore follows a pretrain–generalize–adapt workflow, combining shared pretraining, zero-shot generalization to unseen stations, and optional downstream fine-tuning.

A key characteristic of PV forecasting is that future NWP is available at the forecast origin, whereas future PV generation remains unknown. The following subsection formulates this partially observed future context as a sequencecompletion problem.

## 3.2. Sequence-Completion Formulation

The forecasting framework considers four groups of variables: PV power, meteorological observations (OBS), numerical weather prediction (NWP) forecasts, and deterministic time encodings (TE). In this study, the OBS variables are represented by ERA5 weather reanalysis. The variables included in each group are summarized in Table 1.

For a given forecast origin, all variables are organized according to their original temporal positions within a unified multivariate sequence. Let

$$
\mathbf { X } = \left[ \mathbf { X } ^ { \mathrm { { h i s t } } } ; \mathbf { X } ^ { \mathrm { { f u t } } } \right] \in \mathbb { R } ^ { S \times C } ,\tag{2}
$$

where $S = S _ { h } + S _ { f }$ is the total sequence length, $S _ { h }$ and $S _ { f }$ denote the historical and future segments, respectively, and � is the total number of variables.

All four variable groups are available in the historical segment. Their availability difers in the future segment: future PV generation and meteorological observations are unavailable at the forecast origin, whereas future NWP forecasts and time encodings remain available. The resulting forecast-time information structure can therefore be summarized as

<table><tr><td></td><td>PV</td><td>OBS</td><td>NWP</td><td>TE</td></tr><tr><td>History</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Future</td><td>X</td><td>X</td><td>√</td><td>√</td></tr></table>

(3)

Based on this forecast-time availability pattern, an operational availability mask is defined as

$$
\mathbf { A } \in \{ 0 , 1 \} ^ { S \times C } ,\tag{4}
$$

where $A _ { t , c } = 1$ indicates that variable � at time � is available at the forecast origin, and $A _ { t , c } = 0$ otherwise. The partially observed sequence is then constructed as

$$
\widetilde { \bf X } = { \mathrm { M a s k } } \left( { \bf X } , { \bf A } \right) ,\tag{5}
$$

where Mask(⋅) constructs the partially observed sequence according to the availability mask �.

Under this representation, NWP-assisted PV forecasting is formulated as a sequence-completion task. Given the partially observed sequence $\widetilde { \mathbf { X } } ,$ the shared forecasting model reconstructs the multivariate sequence as

$$
\widehat { \mathbf { X } } = f _ { \theta } \left( \widetilde { \mathbf { X } } \right) ,\tag{6}
$$

where $f _ { \theta }$ denotes the forecasting model with parameters $\theta ,$ and $\hat { \mathbf { X } }$ is the reconstructed sequence. The final PV forecast is obtained from the future PV component,

$$
\begin{array} { r } { \widehat { \bf Y } = \widehat { \bf X } _ { \mathrm { P V } } ^ { \mathrm { f u t } } . } \end{array}\tag{7}
$$

By organizing historical information and future NWP forecasts within a unified temporal sequence, the sequencecompletion formulation provides a common representation across PV stations and forecasting horizons. The construction of multi-horizon forecasting samples and forecast-time NWP alignment is described next.

## 3.3. Multi-Horizon Sample Construction and Forecast-Time NWP Alignment

To support multiple forecasting horizons within a single shared model, training samples are constructed with variable prediction lengths. For each sample, the forecasting horizon � is selected from

$$
F \in \{ 2 4 , 4 0 , 1 2 0 , 2 8 0 \} ,\tag{8}
$$

corresponding to forecasting tasks of 6 hours, 1 day, 3 days, and 7 days, respectively. Since only the daytime operating period from 07:00 to 17:00 is retained, one operational day contains 40 samples at the 15-minute resolution.

![](images/b8c686aadf7aa3e3bba59e715b7235231ff14a41436dd2e9718676ddb84e3349.jpg)  
Figure 2: Illustration of rolling NWP alignment for the same target day. Solar irradiance forecasts for day � are shown using NWP products issued from days � − 7 to � − 1.

A common sequence structure is adopted to enable multi-horizon forecasting within a fixed-length model architecture. The historical and future segments are both fixed to

$$
{ \cal S } _ { h } = { \cal S } _ { f } = 2 8 0 ,\tag{9}
$$

giving a total sequence length of � = 560. For a forecasting horizon $F \ < \ S _ { f }$ , only the first � positions of the future segment constitute the valid forecasting interval, while the remaining positions are padded and excluded from supervision. This design allows samples with diferent forecasting horizons to share the same input dimensions and be processed by a single forecasting backbone.

NWP forecasts are aligned with the forecast origin of each sample. Let � denote the forecast origin. For a sample with forecasting horizon �, the NWP inputs are selected from the forecast product available at time � and covering the corresponding future interval

$$
[ t + 1 , t + F ] .\tag{10}
$$

As NWP products are updated over time, forecasts for the same target period may difer according to their issue time. The rolling NWP construction therefore associates each forecasting sample with the meteorological forecast that would have been available when the PV forecast was issued. An example of this rolling alignment is illustrated in Fig. 2.

## 3.4. Shared Forecasting Backbone

PatchTSMixer [34] is used as the base backbone of the proposed forecasting framework to process the multivariate sequences constructed above.

Given the partially observed sequence $\widetilde { \mathbf { X } } \in \mathbb { R } ^ { S \times C }$ , the temporal dimension is divided into non-overlapping patches

![](images/c1bd7f9fb97fd04fdf53b2d63e8986908f16267437afe87ada2d652bae171e28.jpg)  
Figure 3: Overview of the multi-horizon sample construction, reconstruction-based pretraining objectives, and shared forecasting backbone. FA applies forecasting-aligned reconstruction, GR performs generic reconstruction with random masking, and MR combines the two masking strategies. The resulting masked sequences are processed by a shared PatchTSMixer-based backbone and reconstruction head, from which the future PV segment is extracted as the forecasting output. White blocks denote masked entries. Pretraining masks are used only during pretraining.

of length $p .$ The number of patches is

$$
N = { \frac { S } { p } } ,\tag{11}
$$

and the resulting patch representation can be written as

$$
\mathbf { X } _ { p } \in \mathbb { R } ^ { N \times p \times C } .\tag{12}
$$

Each patch is projected into a latent representation before being processed by the mixer backbone.

The backbone consists of � stacked mixer blocks. Each block performs feature mixing along the temporal-patch, within-patch, and variable dimensions. These operations capture temporal dependencies across patches, local variations within individual patches, and interactions among PV, OBS, NWP, and time-encoding variables. Residual connections and normalization are applied throughout the mixer blocks to facilitate representation learning over the long input sequence.

The output of the final mixer block is mapped by a reconstruction head to the original sequence space,

$$
\begin{array} { r } { \widehat { \mathbf { X } } = g _ { \phi } \Big ( h _ { \theta } ( \widetilde { \mathbf { X } } ) \Big ) , } \end{array}\tag{13}
$$

where $h _ { \theta } ( \cdot )$ denotes the shared PatchTSMixer backbone and $g _ { \phi } ( \cdot )$ denotes the reconstruction head. The reconstructed future PV component is used as the forecasting output according to the sequence-completion formulation in Sec. 3.2.

## 3.5. Reconstruction-Based Pretraining Objectives

Based on the sequence-completion formulation, diferent pretraining objectives can be constructed by varying the masking pattern and the positions used for reconstruction supervision. These masking schemes are applied during pretraining to expose the model to diferent partial-observation patterns and encourage transferable representation learning. Three objectives are considered in this work: forecastaligned reconstruction (FA), generic reconstruction (GR), and mixed reconstruction (MR), as illustrated in Fig. 3.

Let $\mathcal { M } _ { q }$ denote the set of masked positions under objective $q ~ \in ~ \{ \mathrm { F A } , \mathrm { G R } , \mathrm { M R } \}$ , and let $\Omega _ { q }$ denote the corresponding supervision set. Given the masked input, the reconstruction loss is defined as

$$
\mathcal { L } _ { q } = \frac { 1 } { | \Omega _ { q } | } \sum _ { ( t , c ) \in \Omega _ { q } } \left( \widehat { X } _ { t , c } - X _ { t , c } \right) ^ { 2 } .\tag{14}
$$

Forecast-Aligned Reconstruction (FA) FA follows the forecast-time information structure defined in Sec. 3.2. Future PV and OBS values are masked, while historical variables, future NWP forecasts, and future time encodings remain available. The reconstruction loss is computed only over the future PV segment,

$$
\Omega _ { \mathrm { F A } } = \Omega _ { \mathrm { P V } } ^ { \mathrm { f u t } } ,\tag{15}
$$

thereby aligning the supervision directly with the PV forecasting task.

![](images/95524ffcfffb84d8922b83018a24f599f955951538960344041520f4b0f38896.jpg)  
Figure 4: Spatial distribution and dataset split of the 479 distributed PV stations in Ningbo. Colors distinguish the 340 development stations from the 139 spatially held-out stations used for zero-shot evaluation, while marker size represents the installed PV capacity of each station.

Generic Reconstruction (GR) GR adopts a generic maskedreconstruction objective by randomly masking temporal patches of the non-temporal variables across the sequence, while time-encoding variables remain available. The model reconstructs the randomly masked entries, and the corresponding masked positions are used for supervision. Since the forecast-time availability pattern is not explicitly imposed during GR pretraining, some future PV and OBS positions may remain visible when they are not selected for random masking. By varying the observed and masked regions across training samples, GR encourages the model to infer missing information from diverse temporal and cross-variable contexts rather than learning only the fixed information pattern of the forecasting task.

Mixed Reconstruction (MR) MR integrates the forecastingoriented objective of FA with the generic reconstruction objective of GR. Starting from the FA masking pattern, an additional random mask is applied to all non-temporal variables across the sequence with probability �. The resulting masking pattern is therefore given by

$$
\mathcal { M } _ { \mathrm { M R } } = \mathcal { M } _ { \mathrm { F A } } \cup \mathcal { M } _ { \mathrm { r a n d } } .\tag{16}
$$

The reconstruction loss is computed over all valid masked positions, including the future PV and OBS targets defined by FA and the additional positions introduced by random masking. Time-encoding variables and padded positions are excluded. This allows MR to learn from both forecasting-oriented supervision and broader temporal and cross-variable reconstruction. The random masking probability is set to $a = 0 . 3$

The above masking schemes are used only during pretraining. During zero-shot forecasting and downstream deployment, no additional random masking is applied, and the input sequence follows the operational availability pattern defined in Sec. 3.2: future PV and OBS values are unavailable, whereas future NWP forecasts and time encodings remain available.

## 4. Experiments

## 4.1. Dataset and Evaluation Protocol

The experiments are conducted on a real-world distributed PV dataset collected from 479 stations in Ningbo, China, covering the period from January 2022 to June 2023. PV power measurements are accompanied by meteorological information from two sources. ERA5 weather reanalysis with a spatial resolution of $0 . 2 5 ^ { \circ }$ is used to represent meteorological observations, while NWP forecasts are obtained from 12 predefined reference locations across Ningbo. Each PV station is associated with the nearest NWP reference location according to geographical proximity. The spatial distribution and dataset split of the PV stations are shown in Fig. 4, where marker size represents the installed capacity of each station.

All data sources are resampled to a unified 15-minute resolution. Only daytime periods from 07:00 to 17:00 are retained, resulting in 40 samples per operational day. Missing values are filled using linear interpolation. PV power is normalized by the installed capacity of each station and expressed in per-unit (p.u.) form.

A strict spatial and temporal separation is adopted throughout model development and evaluation. For the spatial split, four NWP reference locations are randomly selected, and all PV stations associated with these locations are assigned to the held-out set. This results in 340 development stations and 139 held-out stations, with no overlap between the two groups. Splitting the dataset at the NWP-reference level prevents neighboring stations that share the same NWP inputs from being distributed across the development and held-out sets. The held-out stations therefore form several spatially concentrated groups, providing a more stringent setting for evaluating cross-station generalization than a station-wise random split.

For the 340 development stations, the data are divided chronologically into training, validation, and test periods. The training period spans January 1, 2022 to January 13, 2023, followed by validation from January 14 to March 8, 2023 and testing from March 9 to June 24, 2023. For the shared pretrained models, both pretraining and model selection are performed exclusively using the development stations.

The 139 held-out stations are evaluated over the same final test period from March 9 to June 24, 2023. These stations are completely excluded from shared pretraining and shared-model selection. For zero-shot evaluation, the pretrained model is directly applied to these stations without station-specific retraining or parameter adaptation. This protocol therefore evaluates whether a model pretrained on the development stations can generalize directly to spatially unseen PV stations during a temporally future test period.

## 4.2. Experimental Settings and Baselines

The PatchTSMixer-based backbone uses a patch length of $p = 4 ,$ a hidden dimension of $H = 6 4 , K = 6$ stacked mixer blocks, and an MLP expansion factor of $\gamma = 3$ . The proposed models are trained using the Adam optimizer with a batch size of 128, an initial learning rate of $2 \times 1 0 ^ { - 4 }$ , and a dropout rate of 0.1. Training is performed for up to 20 epochs with early stopping based on validation performance. Unless otherwise stated, the random masking probability used in GR and MR is set to $a = 0 . 3$

Three groups of baseline methods are considered to evaluate the proposed framework from complementary perspectives.

Station-specific baselines represent conventional forecasting pipelines in which separate models are developed for individual PV stations and forecasting horizons. The compared methods include persistence-based predictors (Pers-L and Pers-D), Ridge regression, LightGBM [35], XGBoost [36], Transformer [37], and Informer [38]. For the learnable baselines, a separate model is trained using historical data from each target station for each forecasting horizon. These methods therefore provide locally customized reference models for comparison with the zero-shot shared forecasting framework.

Shared forecasting baselines are introduced to evaluate cross-station parameter sharing under the same development and evaluation protocol as the proposed framework. A shared Transformer is trained jointly on the 340 development stations and subsequently applied to the held-out stations without station-specific retraining.

Generalist time-series baselines include Chronos [15], its multivariate variant Chronos-MV, and TimesFM-3 [17]. Chronos and Chronos-MV are retrained on the 340 development stations under the forecasting setting considered in this study and are subsequently evaluated on the held-out stations without target-station adaptation.

TimesFM-3, released by Google Research in August 2026 [17], is the latest large-scale time-series foundation model in the TimesFM family and delivers state-of-theart benchmark performance. It contains approximately 330 million parameters and is pretrained on a broad mixture of real-world and synthetic time series comprising more than one trillion time points. TimesFM-3 natively supports multivariate forecasting together with past-only and pastand-future covariates, making it particularly suitable for NWP-assisted PV forecasting. Given its large-scale crossdomain pretraining, the oficial pretrained checkpoint is directly adopted in this study without retraining or fine-tuning on the development stations. Historical PV power is used as the target series, historical meteorological observations as past-only covariates, and NWP together with time encodings as past-and-future covariates.

To reduce the influence of model capacity where practical, the shared Transformer, Chronos, and the proposed model are configured with comparable parameter counts of approximately one million parameters. Chronos-MV is retained as a moderately larger multivariate reference, whereas TimesFM-3 is evaluated at its original pretrained scale. A common model-independent quality-control procedure is applied before evaluation, and all methods are compared on the same valid test samples.

## 4.3. Evaluation Metrics

Forecasting performance is evaluated using standard statistical errors and region-specific operational metrics. The two types of metrics are computed in diferent representations. MAE and RMSE are evaluated using capacitynormalized PV power in per-unit (p.u.) form, whereas the regional KPIs are calculated after restoring the predictions and observations to the physical power scale of each station.

## 4.3.1. Statistical Metrics

Let $\{ y _ { t } \} _ { t = 1 } ^ { N }$ and $\{ \hat { y } _ { t } \} _ { t = 1 } ^ { N }$ denote the ground-truth and predicted PV power in p.u., respectively. The mean absolute error (MAE) is defined as

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \left| y _ { t } - \hat { y } _ { t } \right| ,\tag{17}
$$

and the root mean squared error (RMSE) is

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \left( y _ { t } - \hat { y } _ { t } \right) ^ { 2 } } .\tag{18}
$$

Both metrics are dimensionless under the p.u. representation, with lower values indicating better forecasting accuracy.

## 4.3.2. Region-Specific Operational Metrics

For the regional KPI evaluation, the predicted and observed PV values are first converted from p.u. values back to the physical power scale of each station. Let $\{ P _ { t } \} _ { t = 1 } ^ { N }$ and $\{ \hat { P } _ { t } \} _ { t = 1 } ^ { N }$ denote the corresponding observed and predicted power, and let cap denote the installed capacity of the station in the same physical unit.

Nanfang KPI For the Nanfang region, low-output periods are excluded using a threshold � cap. The evaluated time index set is

$$
\mathcal { T } = \left\{ t \Big | P _ { t } \geq \tau \mathrm { c a p o r } \hat { P } _ { t } \geq \tau \mathrm { c a p } \right\} .\tag{19}
$$

The corresponding KPI is

$$
\mathrm { K P I } _ { \mathrm { N F } } = \operatorname* { m a x } \left( 0 , 1 - \sqrt { \frac { 1 } { \left| \mathcal { T } \right| } \sum _ { t \in \mathcal { T } } \frac { \left( P _ { t } - \hat { P _ { t } } \right) ^ { 2 } } { \operatorname* { m a x } \left( P _ { t } ^ { 2 } , ( \delta \operatorname { c a p } ) ^ { 2 } \right) } } \right) ,
$$

Table 2  
Comparison with shared and generalist time-series models on the common quality-controlled test set of the spatially held-out PV stations under diferent forecasting horizons. Lower values indicate better performance. Parameter counts are approximately matched for Shared Transformer, Chronos, and the proposed models, while TimesFM-3 is evaluated at its original pretrained scale. All models are evaluated without target-station adaptation.
<table><tr><td rowspan="2">Method</td><td rowspan="2">#Params</td><td colspan="2">6 h</td><td colspan="2">1 day</td><td colspan="2">3 days</td><td colspan="2">7 days</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td>Shared forecasting baseline</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Shared Transformer</td><td>1.07M</td><td>0.124</td><td>0.161</td><td>0.122</td><td>0.159</td><td>0.125</td><td>0.163</td><td>0.131</td><td>0.170</td></tr><tr><td>Generalist time-series baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Chronos</td><td>1.08M</td><td>0.229</td><td>0.307</td><td>0.237</td><td>0.317</td><td>0.260</td><td>0.340</td><td>0.262</td><td>0.344</td></tr><tr><td>Chronos-MV</td><td>1.52M</td><td>0.130</td><td>0.196</td><td>0.161</td><td>0.237</td><td>0.208</td><td>0.292</td><td>0.226</td><td>0.309</td></tr><tr><td>TimesFM-3</td><td>330M</td><td>0.096</td><td>0.138</td><td>0.102</td><td>0.146</td><td>0.117</td><td>0.162</td><td>0.134</td><td>0.184</td></tr><tr><td>Proposed framework</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FA (Ours)</td><td>1.03M</td><td>0.100</td><td>0.137</td><td>0.106</td><td>0.145</td><td>0.116</td><td>0.156</td><td>0.133</td><td>0.178</td></tr><tr><td>MR (Ours)</td><td>1.03M</td><td>0.095</td><td>0.129</td><td>0.100</td><td>0.135</td><td>0.111</td><td>0.147</td><td>0.127</td><td>0.168</td></tr><tr><td>GR (Ours)</td><td>1.03M</td><td>0.107</td><td>0.140</td><td>0.116</td><td>0.150</td><td>0.130</td><td>0.165</td><td>0.143</td><td>0.180</td></tr></table>

(20)

where � = 0.1 and $\delta = 0 . 2$

Shandong KPI For the Shandong region, forecasting error is evaluated using capacity-normalized RMSE:

$$
\mathrm { K P I } _ { \mathrm { S D } } = 1 - \frac { \sqrt { \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \left( P _ { t } - \hat { P _ { t } } \right) ^ { 2 } } } { \mathrm { c a p } } .\tag{21}
$$

Huabei KPI For the Huabei region, larger point-wise forecasting errors receive higher weights. The normalized error weight is

$$
w _ { t } = \frac { \left| P _ { t } - \hat { P } _ { t } \right| } { \sum _ { k = 1 } ^ { N } \left| P _ { k } - \hat { P } _ { k } \right| } ,\tag{22}
$$

and the corresponding KPI is

$$
\mathrm { K P I } _ { \mathrm { H B } } = 1 - \frac { \sqrt { \sum _ { t = 1 } ^ { N } w _ { t } \left( P _ { t } - \hat { P _ { t } } \right) ^ { 2 } } } { \mathrm { c a p } } .\tag{23}
$$

Higher KPI values indicate better performance under the corresponding regional assessment rule.

## 4.4. Zero-Shot Multi-Horizon Forecasting

We first evaluate the forecasting performance on the 139 held-out PV stations. The shared models are trained using only the 340 development stations and are directly applied to the held-out stations without station-specific retraining. Results are reported for four forecasting horizons: 6 hours, 1 day, 3 days, and 7 days.

## 4.4.1. Comparison with Shared and Generalist Models

Table 2 compares the proposed models with the shared Transformer and generalist time-series baselines. MR achieves the lowest MAE and RMSE at all four forecasting horizons.

Compared with the shared Transformer, its gains are more evident at short and medium horizons, while the diference becomes small at 7 days.

The three pretraining objectives show a consistent ranking under the standard forecasting setting, with MR performing best, followed by FA and GR. This indicates that combining forecast-aligned reconstruction with additional random reconstruction provides a stronger pretraining signal than using either objective alone.

The generalist baselines represent two diferent settings. Chronos and Chronos-MV are classic foundation-model baselines retrained on the development dataset, whereas TimesFM-3 uses large-scale pretrained parameters learned from a broad collection of real-world and synthetic time series. TimesFM-3 gives the strongest performance among the generalist models, but MR achieves lower MAE and RMSE across all horizons. The performance gap also widens as the forecasting horizon increases, particularly for the 3-day and 7-day forecasts. These results show that large-scale generalpurpose pretraining does not automatically translate into dominant performance for distributed PV forecasting, while domain-specific pretraining remains particularly efective at longer forecasting horizons.

## 4.4.2. Comparison with Station-Specific Models

Table 3 compares the zero-shot models with conventional station-specific methods. For the learnable stationspecific baselines, separate models are trained for each target station and forecasting horizon using historical data from that station. The proposed models, in contrast, are applied directly to the held-out stations without parameter updating.

MR remains competitive with, and in most cases outperforms, the locally trained baselines. At the 6-hour horizon, MR achieves an RMSE of 0.129, compared with 0.136 for the station-specific Transformer and 0.137 for Informer. At the 1-day and 3-day horizons, MR obtains RMSE values of 0.135 and 0.147, respectively, both lower than the corresponding results of the compared station-specific models. At the 7-day horizon, MR reaches 0.168, compared with 0.169 for Informer and 0.172 for the station-specific Transformer.

Table 3  
Comparison with conventional station-specific baselines on 139 unseen PV stations under diferent forecasting horizons. Lower values indicate better performance. The proposed framework is evaluated in a zero-shot transfer setting without station-specific retraining.
<table><tr><td></td><td colspan="2">6 h</td><td colspan="2">1 day</td><td colspan="2">3 days</td><td colspan="2">7 days</td></tr><tr><td>Method</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td colspan="9">Station-specific baselines</td></tr><tr><td>Pers-L</td><td>0.188</td><td>0.257</td><td>0.195</td><td>0.264</td><td>0.231</td><td>0.302</td><td>0.242</td><td>0.311</td></tr><tr><td>Pers-D</td><td>0.239</td><td>0.309</td><td>0.243</td><td>0.311</td><td>0.255</td><td>0.324</td><td>0.251</td><td>0.320</td></tr><tr><td>Ridge</td><td>0.105</td><td>0.141</td><td>0.107</td><td>0.143</td><td>0.116</td><td>0.154</td><td>0.131</td><td>0.172</td></tr><tr><td>LGBM</td><td>0.109</td><td>0.147</td><td>0.112</td><td>0.151</td><td>0.122</td><td>0.163</td><td>0.138</td><td>0.182</td></tr><tr><td>XGB</td><td>0.108</td><td>0.146</td><td>0.111</td><td>0.149</td><td>0.121</td><td>0.161</td><td>0.137</td><td>0.179</td></tr><tr><td>Transformer</td><td>0.102</td><td>0.136</td><td>0.104</td><td>0.139</td><td>0.116</td><td>0.153</td><td>0.131</td><td>0.172</td></tr><tr><td>Informer</td><td>0.103</td><td>0.137</td><td>0.104</td><td>0.139</td><td>0.114</td><td>0.151</td><td>0.129</td><td>0.169</td></tr><tr><td colspan="9">Proposed framework</td></tr><tr><td>FA (Ours)</td><td>0.100</td><td>0.137</td><td>0.106</td><td>0.145</td><td>0.116</td><td>0.156</td><td>0.133</td><td>0.178</td></tr><tr><td>MR (Ours)</td><td>0.095</td><td>0.129</td><td>0.100</td><td>0.135</td><td>0.111</td><td>0.147</td><td>0.127</td><td>0.168</td></tr><tr><td>GR (Ours)</td><td>0.107</td><td>0.140</td><td>0.116</td><td>0.150</td><td>0.130</td><td>0.165</td><td>0.143</td><td>0.180</td></tr></table>

![](images/2db29f762122a967885ffb17f35e87c0275651f0a765aca6dbe56fe04fa46fd2.jpg)  
(a) 6 h

![](images/9b7acc8a0c1f0ac3ad3d4f4790b1f513c301ea32a52e208785fd95e92399f818.jpg)  
(b) 1 day

![](images/0bf9803ce2cda42a783e18e44ebfa6e164dd4922c2071e98db8c771ea2d31e5a.jpg)  
(c) 3 days

![](images/3c77b66cfb49befa9ded055139302037c7d026b89029ed8516b56d13110c7c55.jpg)  
(d) 7 days  
Figure 5: Station-level RMSE distributions of Ridge, Transformer, Informer, and MR across unseen PV stations under four forecasting horizons. Lower values indicate better predictive accuracy.

![](images/8349ed2b0e8229ffaf09a8c03d02f25ce39aad8417b97825e2eb057d6ce646a2.jpg)  
(a) � = 6 hours

![](images/13f8e510d56d15bdc0161d6b7144c29601fec58a5bb8d05c1c898d994894c53c.jpg)

![](images/932c189987fd723f8df7effe7d8f1f88b15ecfe7b60b93a0d4aeca2c1ae642fe.jpg)  
(c) � = 3 days

(b) � = 1 day  
![](images/95a46d5b1ff3f69b13830c0b0c6ec7ddf532fdcf5b5c76027332a62457463d6f.jpg)  
(d) � = 7 days  
Figure 6: Representative zero-shot forecasting trajectories on unseen PV stations under diferent prediction horizons. The four subplots compare the proposed variants with conventional station-specific baselines. While most methods capture the general generation trend at short horizons, larger discrepancies appear at longer horizons, particularly around rapid ramps and multi-peak patterns.

The comparison is particularly relevant for newly introduced PV stations. The station-specific models benefit from target-station historical data and require a separate training procedure for each station and horizon. MR reaches comparable or better accuracy without this retraining step. A single pretrained model can therefore be reused across heldout stations and forecasting horizons while maintaining forecasting accuracy close to, or better than, locally customized models.

## 4.5. Station-Level Generalization Analysis

Beyond the averaged results in Sec. 4.4, we further examine how forecasting performance varies across individual held-out stations. Fig. 5 shows the station-level RMSE distributions of MR and representative station-specific baselines under the four forecasting horizons.

MR maintains a low median RMSE across all horizons and shows a relatively compact error distribution. The advantage is more evident for the 6-hour, 1-day, and 3-day forecasts, while the distributions become closer at the 7-day horizon. This trend is consistent with the averaged results in Table 3. More importantly, the station-level distributions show that the overall performance of MR is not determined by a small number of favorable stations, but is observed across a broad range of the 139 held-out PV systems.

Representative zero-shot forecasts are further shown in Fig. 6. The four panels present representative held-out cases under diferent forecasting horizons and compare the prediction behavior of the proposed variants with conventional station-specific baselines. At short horizons, most methods reproduce the overall PV generation profile with relatively small deviations. Larger diferences appear at the 3-day and 7-day horizons, particularly around peak generation periods and rapid changes in the power trajectory.

Among the three pretraining objectives, GR tends to underestimate several generation peaks, whereas FA shows larger deviations in some long-horizon periods. MR generally follows the observed trajectory more closely and provides a more balanced prediction across the four horizons. These examples are consistent with the quantitative results and illustrate how the mixed reconstruction objective behaves when transferred directly to an unseen PV station.

## 4.6. Robustness to Degraded NWP Availability

The previous experiments consider the standard forecasting setting in which the complete future NWP sequence is available at the forecast origin. In practical operation, however, NWP inputs may be incomplete because of delayed updates, communication failures, or missing forecast records. To examine the efect of reduced NWP availability, we evaluate FA, GR, and MR under three settings: full forecast, partial NWP availability with 30% patch missingness, and no future forecast. Table 4 reports the absolute forecasting errors, while Fig. 7 shows the corresponding relative RMSE degradation with respect to the full-forecast setting.

Forecasting performance under degraded future NWP availability on 139 held-out PV stations. Results are averaged over all test stations. Lower values indicate better performance.
<table><tr><td colspan="2"></td><td colspan="2">6 h</td><td colspan="2">1 day</td><td colspan="2">3 days</td><td colspan="2">7 days</td></tr><tr><td>Scenario</td><td>Method</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td></tr><tr><td rowspan="3">Full forecast</td><td>FA</td><td>0.100</td><td>0.137</td><td>0.106</td><td>0.145</td><td>0.116</td><td>0.156</td><td>0.133</td><td>0.178</td></tr><tr><td>MR</td><td>0.095</td><td>0.129</td><td>0.100</td><td>0.135</td><td>0.111</td><td>0.147</td><td>0.127</td><td>0.168</td></tr><tr><td>GR</td><td>0.107</td><td>0.140</td><td>0.116</td><td>0.150</td><td>0.130</td><td>0.165</td><td>0.143</td><td>0.180</td></tr><tr><td rowspan="3">Partial NWP (30% masked)</td><td>FA</td><td>0.111</td><td>0.148</td><td>0.118</td><td>0.157</td><td>0.127</td><td>0.167</td><td>0.142</td><td>0.187</td></tr><tr><td>MR</td><td>0.097</td><td>0.131</td><td>0.102</td><td>0.137</td><td>0.113</td><td>0.149</td><td>0.129</td><td>0.170</td></tr><tr><td>GR</td><td>0.113</td><td>0.146</td><td>0.122</td><td>0.157</td><td>0.137</td><td>0.172</td><td>0.148</td><td>0.185</td></tr><tr><td rowspan="3">No forecast</td><td>FA</td><td>0.145</td><td>0.187</td><td>0.156</td><td>0.200</td><td>0.170</td><td>0.216</td><td>0.180</td><td>0.228</td></tr><tr><td>MR</td><td>0.123</td><td>0.169</td><td>0.140</td><td>0.186</td><td>0.164</td><td>0.210</td><td>0.174</td><td>0.219</td></tr><tr><td>GR</td><td>0.140</td><td>0.181</td><td>0.152</td><td>0.194</td><td>0.170</td><td>0.214</td><td>0.175</td><td>0.218</td></tr></table>

![](images/c9e3c5f76c400d4cbf6321d537806105f46ed5e6b2ed7c52f2ebcc7c09cfd613.jpg)  
(a) Partial forecast availability

![](images/29ae9ed2a3b5e8b7152734ae30d49e7b457e1c95035866850d2f06d5c7155fc3.jpg)  
(b) No future forecast  
Figure 7: Relative RMSE degradation under degraded future NWP availability. Degradation is computed relative to the full-forecast setting. Lower values indicate stronger robustness to reduced or unavailable future NWP inputs.

Reducing future NWP availability increases the forecasting error of all three models, confirming that future meteorological forecasts provide important information for PV prediction. The extent of this degradation, however, difers considerably among the three pretraining objectives.

Under partial NWP availability, MR is only slightly afected across all four forecasting horizons. Its RMSE increases from 0.129 to 0.131 at 6 hours, from 0.135 to 0.137 at 1 day, from 0.147 to 0.149 at 3 days, and from 0.168 to 0.170 at 7 days. Accordingly, Fig. 7 shows a relative RMSE degradation of only about 1–2% for MR, compared with larger changes for FA and GR. This behavior is consistent with the mixed reconstruction objective: in addition to forecasting-aligned supervision, MR is exposed to randomly masked variables during pretraining and therefore learns to reconstruct missing information from diferent parts of the available temporal and meteorological context.

When future NWP is completely unavailable, all three models experience a substantial loss of accuracy. GR shows the smallest relative RMSE degradation across the four forecasting horizons. At the 3-day horizon, for example, its RMSE increases from 0.165 to 0.214, corresponding to a degradation of approximately 30%, compared with about 38% for FA and 43% for MR. This stronger relative tolerance is consistent with the generic reconstruction objective of GR, which exposes the model to diferent partial-observation patterns during pretraining and reduces its dependence on a fixed set of available variables. However, GR starts from a higher error under the full-NWP setting, and its smaller relative degradation therefore does not translate into the best absolute forecasting accuracy. MR still achieves the lowest

MAE and RMSE across most horizons after future NWP is removed.

These results reveal diferent characteristics of the three pretraining objectives. FA is closely aligned with the standard forecasting setting and therefore benefits strongly from the availability of future NWP. GR is less sensitive to the complete removal of NWP, but its generic reconstruction objective results in lower forecasting accuracy when full meteorological information is available. MR combines forecasting-oriented and random reconstruction supervision, providing the best absolute accuracy under normal and partially degraded NWP conditions while retaining reasonable performance when future NWP is completely unavailable.

## 4.7. Downstream Adaptation to Region-Specific Operational Objectives

The previous experiments evaluate direct zero-shot transfer without using target-station data for parameter updating. We next consider a downstream adaptation setting in which historical data from a target station are available and the pretrained model is further optimized for region-specific operational objectives.

For each target station, fine-tuning starts from the same pretrained checkpoint and is performed independently using the historical data of that station. All model parameters remain trainable during adaptation. The training, validation, and test periods follow the same chronological split defined in Sec. 4.1. Fine-tuning uses a learning rate of $1 \times 1 0 ^ { - 5 }$ for up to 30 epochs, with early stopping based on validation performance. Since the regional KPIs contain non-smooth operations that are not directly suitable for gradient-based optimization, KPI-aligned diferentiable surrogate losses are used during fine-tuning, including p.u. MSE for Shandong, soft-gated normalized MSE for Nanfang, and softmax-weighted MSE for Huabei. The final KPI values are computed using the original definitions in Sec. 4.3.

Table 5 summarizes the adaptation results for the 1- day and 3-day forecasting horizons. A large majority of stations obtain positive KPI changes after fine-tuning, and the proportion of improved stations exceeds 90% in almost all settings. This shows that the shared pretrained model can provide a useful initialization when target-station data become available and the forecasting objective changes from generic prediction accuracy to a specific operational assessment rule.

The final KPI results also reveal diferences among the three pretraining objectives. MR achieves the highest final KPI in Huabei and Shandong for both forecasting horizons, reaching 79.60% and 86.81%, respectively, for the 1-day setting and 78.35% and 85.54% for the 3-day setting. GR gives slightly higher final scores in Nanfang, reaching 57.02% at 1 day and 53.61% at 3 days. These results show that no single pretraining objective dominates every regional criterion, although MR provides consistently strong initialization across the three operational settings.

GR exhibits larger average KPI improvements in several cases. This should be interpreted together with its lower zero-shot forecasting accuracy, which leaves more room for improvement during station-specific fine-tuning. The final KPI score therefore provides a more direct measure of the adapted model than the magnitude of the improvement alone.

Overall, the adaptation results complement the zero-shot experiments in Sec. 4.4. The pretrained model can either be deployed directly at an unseen station or further adapted when local historical data and region-specific operational objectives are available.

## 5. Conclusion

This paper presents a foundation-style pretraining framework for scalable distributed PV forecasting. A shared model is pretrained across multiple PV stations and forecasting horizons, while NWP-assisted forecasting is formulated as a sequence-completion task under the operational availability of historical observations and future weather forecasts. Among the three reconstruction objectives, MR achieves the best overall performance. Experiments on 479 distributed PV stations, including 340 development stations and 139 spatially held-out stations, show that MR provides strong zero-shot forecasting performance from 6 hours to 7 days and outperforms representative station-specific, shared, and generalist time-series baselines. Comparison with a recent large-scale foundation model further indicates that broad general-purpose pretraining does not necessarily outperform domain-specific pretraining for distributed PV forecasting, particularly at longer horizons. The proposed framework also remains robust under degraded NWP availability and can be further adapted to region-specific operational objectives when local data become available. Future work will investigate cross-regional transfer, probabilistic forecasting, and the integration of pretrained PV forecasts with downstream power-system decision-making.

## Acknowledgment

This work is supported by the RISUD Projects (No. P0042845) and the RISE Project (No. P0051003) of The Hong Kong Polytechnic University, International Centre of Urban Energy Nexus of The Hong Kong Polytechnic University (No. P0047700), and KKS Synergy project, Energy flexibility through synergies of big data, novel technologies & systems, and innovative markets (202000739). This work is financially supported by The Major Science and Technology Projects of Ningbo (No. 2022Z236), Natural Science Foundation of Ningbo of China (No. 2023J027), as well as by the High Performance Computing Centers at Eastern Institute of Technology, Ningbo, and Ningbo Institute of Digital Twin.

Downstream adaptation to region-specific operational objectives under the 1-day and 3-day forecasting settings. Results are averaged over 139 held-out PV stations. Bold values indicate the highest final Mean KPI for each forecasting horizon and regiona criterion.
<table><tr><td>Horizon</td><td>Method</td><td>Region</td><td>Improved stations (%)</td><td>Mean ∆KPI (pp)</td><td>Mean KPI (%)</td></tr><tr><td rowspan="10">1 day</td><td>FA</td><td>Huabei</td><td>97.84</td><td>0.98</td><td>77.66</td></tr><tr><td></td><td>Nanfang</td><td>99.28</td><td>7.44</td><td>52.52</td></tr><tr><td>MR</td><td>Shandong</td><td>90.65</td><td>0.20</td><td>85.74</td></tr><tr><td></td><td>Huabei</td><td>98.56</td><td>1.03</td><td>79.60</td></tr><tr><td></td><td>Nanfang</td><td>100.00</td><td>8.67</td><td>56.86</td></tr><tr><td>GR</td><td>Shandong</td><td>99.28</td><td>0.29</td><td>86.81</td></tr><tr><td></td><td>Huabei</td><td>99.28</td><td>1.71</td><td>79.54</td></tr><tr><td></td><td>Nanfang</td><td>100.00</td><td>17.31</td><td>57.02</td></tr><tr><td>FA</td><td>Shandong</td><td>98.56</td><td>1.70</td><td>86.75</td></tr><tr><td></td><td>Huabei</td><td>99.28</td><td>1.29</td><td>76.54</td></tr><tr><td rowspan="8">3 days</td><td></td><td>Nanfang</td><td>100.00</td><td>8.74</td><td>49.72</td></tr><tr><td>MR</td><td>Shandong</td><td>93.53</td><td>0.25</td><td>84.65</td></tr><tr><td></td><td>Huabei</td><td>99.28</td><td>1.03</td><td>78.35</td></tr><tr><td></td><td>Nanfang</td><td>100.00</td><td>10.40</td><td>52.84</td></tr><tr><td>GR</td><td>Shandong</td><td>94.96</td><td>0.25</td><td>85.54</td></tr><tr><td></td><td>Huabei</td><td>98.56</td><td>1.74</td><td>78.06</td></tr><tr><td></td><td>Nanfang</td><td>100.00</td><td>18.01</td><td>53.61</td></tr><tr><td></td><td>Shandong</td><td>99.28</td><td>1.92</td><td>85.45</td></tr></table>

Notes. “Improved stations (%)” denotes the percentage of stations with a positive KPI gain after adaptation. “Mean ΔKPI (pp)” denotes the average KPI change from the pretrained zero-shot model to the adapted model, measured in percentage points. “Mean KPI (%)” reports the final averaged KPI after adaptation under each regional assessment rule.

## References

[1] M. de l’Epine, G. Dominguez, I. Kaizuka, A. Jäger-Waldau, A Snapshot of Global PV Markets, Technical Report T1-49:2026, IEA Photovoltaic Power Systems Programme, Task 1, 2026. URL: https://iea-pvps.org/snapshot-reports/snapshot-2026/. doi:10.69766/WXLW1389.

[2] National Energy Administration of China, Renewable energy grid-connected operation in 2025, National Energy Administration of China, 2026. URL: https://www.nea.gov.cn/20260212/ 742b8c6a078347b0b39de676c05c5d58/c.html.

[3] National Energy Administration, Notice on issuing the administrative measures for the development and construction of distributed photovoltaic power generation, Online, 2025. URL: https://app.www. gov.cn/govdata/gov/202502/18/524700/article.html, accessed: 2026- 08-08.

[4] P. Rahdan, E. Zeyen, C. Gallego-Castillo, M. Victoria, Distributed photovoltaics provides key benefits for a highly renewable european energy system, Applied Energy 360 (2024) 122721.

[5] S. Thapar, Centralized vs decentralized solar: A comparison study (india), Renewable Energy 194 (2022) 687–704.

[6] R. Ahmed, V. Sreeram, Y. Mishra, M. Arif, A review and evaluation of the state-of-the-art in pv solar power forecasting: Techniques and optimization, Renewable and sustainable energy reviews 124 (2020) 109792.

[7] O. Lindberg, D. Lingfors, J. Arnqvist, D. van der Meer, J. Munkhammar, Day-ahead probabilistic forecasting at a co-located wind and solar power park in sweden: Trading and forecast verification, Advances in Applied Energy 9 (2023) 100120.

[8] T. Lu, C. Wang, Y. Cao, H. Chen, Photovoltaic power prediction under insuficient historical data based on dendrite network and coupled information analysis, Energy reports 9 (2023) 1490–1500.

[9] X. Luo, D. Zhang, X. Zhu, Combining transfer learning and constrained long short-term memory for power generation forecasting of newly-constructed photovoltaic plants, Renewable Energy 185 (2022) 1062–1077.

[10] Y. Wen, S. Pan, X. Li, Z. Li, W. Wen, Improving multi-site photovoltaic forecasting with relevance amplification: Deepfedformerbased approach, Energy 299 (2024) 131479.

[11] J. Simeunović, B. Schubnel, P.-J. Alet, R. E. Carrillo, Spatio-temporal graph neural networks for multi-site pv power forecasting, IEEE Transactions on Sustainable Energy 13 (2022) 1210–1220.

[12] Y. Tang, K. Yang, S. Zhang, Z. Zhang, Photovoltaic power forecasting: A hybrid deep learning model incorporating transfer learning strategy, Renewable and Sustainable Energy Reviews 162 (2022) 112473.

[13] E. Sarmas, N. Dimitropoulos, V. Marinakis, Z. Mylona, H. Doukas, Transfer learning strategies for solar power forecasting under data scarcity, Scientific Reports 12 (2022) 14643.

[14] Y. Yang, Y. Liu, Y. Zhang, S. Shu, J. Zheng, Dest-gnn: A doubleexplored spatio-temporal graph neural network for multi-site intrahour pv power forecasting, Applied Energy 378 (2025) 124744.

[15] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor, et al., Chronos: Learning the language of time series, arXiv preprint arXiv:2403.07815 (2024).

[16] S. Xiaoming, W. Shiyu, N. Yuqi, L. Dianqi, Y. Zhou, W. Qingsong, M. Jin, Time-moe: Billion-scale time series foundation models with mixture of experts, in: ICLR 2025: The Thirteenth International Conference on Learning Representations, International Conference on Learning Representations, 2025.

[17] A. Jain, R. Sen, Timesfm-3: A zero-shot foundation model for multivariate forecasting, Google Research Blog, 2026. URL: https://research.google/blog/ timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/, accessed: 2026-09-04.

[18] D. Chen, X. Shi, M. Jiang, S. Zhu, H. Zhang, D. Zhang, Y. Chen, J. Yan, Selecting efective nwp integration approaches for pv power forecasting with deep learning, Solar Energy 301 (2025) 113939.

[19] X. Luo, D. Zhang, X. Zhu, Deep learning based forecasting of photovoltaic power generation by incorporating domain knowledge, Energy 225 (2021) 120240.

[20] H. Böök, A. V. Lindfors, Site-specific adjustment of a nwp-based photovoltaic production forecast, Solar Energy 211 (2020) 779–788.

[21] M. J. Mayer, D. Yang, B. Szintai, Comparing global and regional downscaled nwp models for irradiance and photovoltaic power forecasting: Ecmwf versus arome, Applied Energy 352 (2023) 121958.

[22] C. Brester, V. Kallio-Myers, A. V. Lindfors, M. Kolehmainen, H. Niska, Evaluating neural network models in site-specific solar pv forecasting using numerical weather prediction data and weather observations, Renewable Energy 207 (2023) 266–274.

[23] J. Qu, Q. Sun, Z. Qian, H. Zareipour, Z. Y. Dong, A transferable framework of pv power forecasting for cross-regional distributed pv

systems using domain adversarial temporal network, IEEE Transactions on Industrial Informatics (2025).

[24] J. Wang, S. Gao, X. Zhao, X. Huang, J. Lu, C. Wu, A cross-domain information fusion method for non-stationary distributed photovoltaic forecasting, Applied Energy 402 (2025) 126867.

[25] W. Liu, M. Gai, Pv-mlp: A lightweight patch-based multi-layer perceptron network with time–frequency domain fusion for accurate long-sequence photovoltaic power forecasting, Renewable Energy 251 (2025) 123277.

[26] C. Zhu, Y. Tu, Q. Wei, Y. Zang, P. Zhou, L. Yang, J. Wang, Y. Yu, R. Lv, J. Du, et al., Short-term power prediction of photovoltaic power station based on lstm-xgboost model, Solar Energy 300 (2025) 113819.

[27] Y. Chen, D. Zhang, Theory-guided deep-learning for electrical load forecasting (tgdlf) via ensemble long short-term memory, Advances in Applied Energy 1 (2021) 100004.

[28] K. Tao, J. Zhao, Y. Tao, Q. Qi, Y. Tian, Operational day-ahead photovoltaic power forecasting based on transformer variant, Applied Energy 373 (2024) 123825.

[29] Z. Hu, Y. Gao, S. Ji, M. Mae, T. Imaizumi, Improved multistep ahead photovoltaic power prediction model based on lstm and self-attention with weather forecast data, Applied Energy 359 (2024) 122709.

[30] W. Zheng, H. Xiao, W. Pei, Cross-site point and interval forecasting of distributed photovoltaic power under limited-data conditions, SSRN (2026). Preprint.

[31] A. Demir, A. Najafi, Difusion-augmented lstm for improved generalization in cross-site solar power forecasting, International Transactions on Electrical Energy Systems 2026 (2026) 6970304.

[32] A. Das, W. Kong, R. Sen, Y. Zhou, A decoder-only foundation model for time-series forecasting, in: Forty-first International Conference on Machine Learning, 2024.

[33] T. Nguyen, J. Brandstetter, A. Kapoor, J. K. Gupta, A. Grover, Climax: A foundation model for weather and climate, in: Proceedings of the 40th International Conference on Machine Learning, ICML’23, JMLR.org, 2023.

[34] V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, J. Kalagnanam, Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting, in: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, Association for Computing Machinery, New York, NY, USA, 2023, p. 459–469. URL: https://doi.org/10.1145/3580305.3599533. doi:10. 1145/3580305.3599533.

[35] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, Lightgbm: A highly eficient gradient boosting decision tree, Advances in neural information processing systems 30 (2017).

[36] T. Chen, C. Guestrin, Xgboost: A scalable tree boosting system, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, Association for Computing Machinery, New York, NY, USA, 2016, p. 785–794. URL: https://doi.org/10.1145/2939672.2939785. doi:10. 1145/2939672.2939785.

[37] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, Advances in neural information processing systems 30 (2017).

[38] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, W. Zhang, Informer: Beyond eficient transformer for long sequence time-series forecasting, in: Proceedings of the AAAI conference on artificial intelligence, volume 35, 2021, pp. 11106–11115.