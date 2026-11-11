# TimeSeek: Multi-Task and Multi-Domain Pretraining for Generalized Time Series Analytics

Peng Chen, Yihang Wang, Zhongwen Rao, Yang Shu, Yuying Qiu, Yuxuan Chen, Lujia Pan, Bin Yang, Senior Member, IEEE, Chenjuan Guo<sup>B</sup>

Abstract—Time series analytics has been widely applied across various domains. While existing time series foundation models (TSFMs) have demonstrated promising results, they still face two major challenges: 1) Most TSFMs mainly rely on stacking Transformer layers to increase parameter size, lacking the ability to adaptively model complex time series patterns from multidomain data. 2) Current TSFMs are primarily pre-trained for specific tasks, e.g., forecasting or anomaly detection, and a few models support multiple tasks, but they fail to fully leverage task specificity and commonality, resulting in suboptimal performance. To address these challenges, we propose TimeSeek, a novel generalized time series analytics model that adaptively models time series patterns from multi-domain data and unifies multiple tasks in pre-training. Conceptually, TimeSeek is guided by selective information sharing under domain and task heterogeneity. From the perspective of multi-domain data, we propose a Multi-Domain Feature Extractor that adaptively captures temporal dependencies and channel correlations based on domain-specific characteristics, including progressive multi-scale modeling and adaptive slot channel modeling. From the perspective of the diversity of tasks, we propose multi-task collaborative training, which leverages task specificity and task commonality through task customization and cross-task training, effectively unifying multiple tasks in pre-training and better adapting to diverse downstream scenarios. After pre-training on large-scale time series data, TimeSeek achieves state-of-the-art performance in forecasting, anomaly detection, and classification tasks across both few-shot learning and zero-shot situations, together with favorable inference efficiency.

Index Terms—Time Series Analytics, Time Series Foundation Models

## I. INTRODUCTION

W <sup>ITH</sup> <sup>the</sup> <sup>rapid</sup> <sup>advancement</sup> <sup>of</sup> <sup>the</sup> <sup>Internet</sup> <sup>of</sup> <sup>Things,</sup> a vast amount of multi-domain time series data has been generated in real-world scenarios, leading to increasing demand for time series analytics, including time series forecasting, anomaly detection, and classification tasks. Time series analytics primarily focuses on modeling complex time series patterns to uncover underlying regularities, thereby providing robust support for decision-making [1]. It has been widely applied across various domains such as finance [2], meteorology [3], transportation [4], and energy [5].

![](images/6aa05de8b888e5a00bc1a14964af81e248690fe1ce0c11aad98a151dc8b7d2c0.jpg)  
Fig. 1. TimeSeek leverages multi-domain data and unifies multiple tasks in pre-training.

In recent years, Foundation Models have emerged as a groundbreaking paradigm in deep learning, demonstrating unprecedented success in fields like natural language processing [6], [7] and computer vision [8], [9]. These models leverage massive datasets and large-scale parameter architectures to learn unified representations, exhibiting exceptional generalization capabilities and adaptability across different tasks. Inspired by these achievements, researchers have begun exploring the potential of foundation models for the time series field [10]–[13].

Existing Time Series Foundation Models (TSFMs) [10], [11], [13] typically adopt a two-stage paradigm: pre-training on multi-source time series data followed by fine-tuning or direct inference for downstream scenarios. As shown in Figure 1, time series data exhibit distinct temporal dependencies and channel correlations for different domains. In the temporal aspect, time series data often display hierarchical temporal patterns—spanning daily, weekly, and seasonal cycles—with the significance of each scale varying considerably across domains. In the channel aspect, channel correlations show significant inter-domain variability. In some domains, such as IoT sensor networks, strong channel dependencies are inherently present, while in others, such as medical vital sign monitoring, correlations may be comparatively weaker. Therefore, effectively modeling multi-scale features and channel correlations during multi-domain pre-training is necessary.

Furthermore, downstream scenarios involve a variety of tasks, including forecasting, anomaly detection, and classification. The prevalent task-specific pre-training approaches in TSFMs [12]–[14] not only incur substantial computational costs due to every task requiring a specific foundation model but also limit the model’s capacity to a single task. In contrast, establishing a unified pre-training paradigm that harmonizes heterogeneous tasks would enable more effective adaptation to multi-task scenarios. Considering both the heterogeneity of domain data and the diversity of tasks, we identify two key challenges that need to be addressed to advance time series foundation models.

For the heterogeneity ofdomain data, time series data across different domains exhibit distinct multi-scale and channel correlation patterns. Adaptively modeling these patterns can significantly enhance performance, yet existing methods fall short here. For multi-scale modeling, current TSFMs [10], [13]–[15] are typically designed for a single scale, which limits their ability to capture multi-scale features required across different domains. While models like MOIRAI [14] adapt patch sizes for different frequency domains, they still lack dynamic scale integration. For correlation modeling, most TSFMs [10]– [13], [16] adopt a channel-independent approach, overlooking the intrinsic correlations between channels. Moreover, the few TSFMs [14] that employ a uniform channel modeling strategy across all domains have the risk of introducing spurious correlations among inherently uncorrelated channels, leading to suboptimal generalization. Therefore, a flexible modeling approach that can dynamically capture two patterns for each domain is essential.

For the diversity of tasks, time series foundation models are expected to serve as versatile tools for diverse downstream applications, necessitating the capability for multi-task learning. While time series tasks share some commonalities, they vary significantly in their objectives and requirements. This inherent heterogeneity can complicate the balancing of learning objectives during pre-training and may even lead to a negative impact between tasks. Existing TSFMs [12]– [14] are primarily pre-trained for specific tasks, which not only incur substantial computational costs due to every task requiring a specific foundation model but also restrict the model’s ability to generalize across tasks. While a few TSFMs support multiple tasks, they fail to balance task specificity and commonality. For instance, Moment [11] relies on reconstruction pretraining and requires swapping specific task heads for fine-tuning, while UniTS [15] simply utilizes task tokens without addressing task heterogeneity. Therefore, designing a pre-training framework that unifies task commonality and specificity to improve overall task performance is necessary.

To address these challenges, we propose Multi-Task and Multi-Domain Pre-training for Generalized Time Series Analytics, namely TimeSeek, to adaptively model time series patterns of multi-domain data and unify multiple time series tasks in pre-training. Conceptually, TimeSeek is motivated by selective information sharing under domain and task heterogeneity. Instead of assuming that information should be uniformly exchanged across all temporal scales, channels, and tasks, it learns when and how such sharing is beneficial while retaining task- and domain-specific representations.

For the first aspect: To better model time series patterns for multi-domain data, we propose a Multi-Domain Feature Extractor that captures complex temporal dependencies and channel correlations. In the temporal aspect, we propose progressive multi-scale modeling. It gradually fuses features from coarse-grained to fine-grained features and incorporates a gating mechanism to dynamically adjust the weights of each scale. Through this approach, the model adaptively filters redundant features and highlights key scale information, enhancing its generalization and adaptability. In the channel aspect, we propose adaptive slot channel modeling to capture channel correlations in multi-domain data. It consists of two steps: 1) the channel router adaptively determines whether meaningful channel correlations exist in the input; if so, 2) channel slot attention adaptively groups the strongly correlated channels into the same slots, combined with a masking strategy to dynamically filter less informative slots and retain important ones. Together, these designs selectively transfer information across temporal scales and channel dependencies, enabling TimeSeek to adapt to heterogeneous multi-domain data.

For the second aspect: To unify the multi-task time series pre-training framework and leverage task specificity and commonality, we propose Multi-Task Collaborative Training. It incorporates task-customized pre-training modules alongside cross-task collaborative training. In task customization, we design specialized modules for three core time series tasks: a non-autoregressive predictor for forecasting, a patch-based reconstruction module for anomaly detection, and a global feature extractor for classification. These innovative designs equip TimeSeek with capabilities tailored to specific tasks, such as generative and discriminative abilities. In the cross-task training, these customized modules are structurally unified over the shared multi-domain feature extractor and collaboratively optimized via a dual-loss mechanism. Crucially, tasks mutually reinforce each other; For example, the global prior extracted from the classification task enhances the semantic consistency of anomaly detection, while the gradients propagated from the reconstruction loss simultaneously improve the classifier’s feature extraction capabilities. Specifically, our contributions can be summarized as follows:

• We propose TimeSeek, a novel generalized time series analytics model that adaptively models time series patterns from multi-domain data and unifies multiple tasks in pre-training, including forecasting, anomaly detection, and classification tasks.

• We propose a multi-domain feature extractor, including progressive multi-scale modeling and adaptive slot channel modeling to effectively capture temporal dependencies and channel correlations from multi-domain data.

• We propose multi-task collaborative training to effectively unify the multi-task pre-training and balance task specificity and commonality.

• After pre-training on large-scale time series data, Time-Seek achieves state-of-the-art performance on three time series tasks across both few-shot learning and zero-shot inference situations.

## II. RELATED WORK

Time series analytics has garnered extensive research attention [17], extending to several time series tasks such as time series forecasting [18]–[20], time series anomaly detection [21]– [24], and time series classification [25]. Existing models can be categorized into two types: time-series-specific models, which are trained and tested on each specific dataset, and time series foundation models, which leverage pre-training on multi-domain datasets and evaluate on downstream datasets.

## A. Time-Series-Specific Models

Although the time-series-specific models can achieve good results on their respective tasks, they all need to be retrained and require tailored parameter tuning for each specific dataset, which causes high costs and limits their performance in datascarce scenarios.

For the task of time series forecasting, with the development of deep learning, many works based on MLP [26]– [28], RNN [29], [30], CNN [17], [31], and GNN [32]–[35] emerged. DLinear [26] decomposes a raw data input into a trend component and a seasonal component and then applies two one-layer linear layers to each component. TimesNet [17] models multi-period temporal variations through adaptively aggregated two-dimensional representations. In recent years, the Transformer has shown great advantages in capturing longterm dependencies [36]–[40]. PatchTST [38] segments time series into subseries-level patches that serve as input tokens to the Transformer and applies the channel-independence strategy for training on multivariate time series. CrossFormer [37] captures hierarchical cross-time and cross-dimension dependencies, and iTransformer [40] learns cross-variable correlations through attention over variate tokens. These methods provide multi-scale or multivariate representations in taskspecific training settings. TimeSeek addresses the additional challenge of selectively controlling information sharing across temporal scales, channels, and task-specific pathways within multi-domain and multi-task pre-training.

For the task of time series classification [25], deep learningbased methods [41]–[46] require less domain knowledge from time series data to handle classification tasks than traditional methods [47]–[49], but these methods require a large amount of labeled data to train model parameters. To address this, some works have explored prototype-based networks [50] or representation learning [51]–[54], enhancing classification performance.

For the task of time series anomaly detection, deep learningbased methods [55]–[59] mainly include reconstruction-based methods and prediction-based methods. Reconstruction-based methods [60]–[65] compress the original input data and then attempt to reconstruct the series, treating high reconstruction errors as anomalies. Prediction-based methods [56] utilize past observations to predict current values and use the difference between the predicted and true values as an anomaly criterion.

## B. Time Series Foundation Models

Recently, the foundation model for time series has gained significant attention [10]–[14], [16], [66]–[69]. By leveraging pre-training on large-scale and diverse time series datasets, time series foundation models demonstrate remarkable adaptability to new tasks with minimal fine-tuning, while even achieving zero-shot transfer capabilities across domains. Timer [10] adopts a decoder-only architecture employing autoregressive modeling for generative pre-training. Chronos [16] converts time series data into discrete tokens by scaling and quantization with a fixed vocabulary, then trains a Transformer-based language architecture on these tokenized time series. TiRex [70] targets zero-shot forecasting with an xLSTM-based architecture, FlowState [71] focuses on sampling-rate-invariant forecasting, and Chronos-2 [72] extends zero-shot forecasting to univariate, multivariate, and covariate-informed settings. Beyond the autoregressive paradigm, other generative paradigms have been explored to enhance time series foundation models. TEMPO [73] adapts pre-trained language models by decomposing time series into trend, seasonal, and residual components, utilizing promptbased learning to facilitate distribution adaptation across non-stationary domains. TimeDiT [74] introduces a Diffusion Transformer architecture equipped with unified masking schemes, coupling transformer-based temporal dependency learning with diffusion processes to capture uncertainties and generate high-quality probabilistic forecasts. However, the above approaches mainly focus on the problem of time series forecasting, and it is difficult to take into account other tasks.

Research on time series foundation models that unify the three tasks of time series forecasting, classification, and anomaly detection remains at a nascent stage, but has shown great potential. Moment [11] pre-trains via masked reconstruction with a lightweight decoder, but requires task-specific heads in downstream forecasting scenarios, limiting its zeroshot ability. UNITS [15] introduces task tokens to unify generative tasks and discriminative tasks into one format. However, it does not fully consider the balance between task heterogeneity and commonality in pre-training.

Summary: Table I compares representative TSFMs in terms of multi-task pre-training, multi-scale modeling, channel modeling, and zero-shot inference. TimeSeek’s distinction lies in selectively sharing information across temporal scales, channel dependencies, and task-specific pathways under domain and task heterogeneity.

TABLE I  
COMPARE TIMESEEK WITH EXISTING TSFMS.
<table><tr><td>Models</td><td>Multi-Task Pre-training</td><td>Multi-Scale Modeling</td><td>Channel Modeling</td><td>Zero-Shot</td></tr><tr><td>Timer [10]</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>MOIRAI [14]</td><td>x</td><td>X</td><td>√</td><td>√</td></tr><tr><td>TimesFM [13]</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Chronos [16]</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Moment [11]</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>UniTS [15]</td><td>√</td><td>x</td><td>√</td><td>√</td></tr><tr><td>Time-MOE [12]</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>TEMPO [73]</td><td>x</td><td>X</td><td>x</td><td>√</td></tr><tr><td>TimeDiT [74]</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>TimeSeek (our model)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## III. PRELIMINARY

## A. Time Series

Time series exhibit the fluctuation of observed data across time. Given a series of length L, denoted by X = $\langle \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { L } \rangle$ , every $\mathbf { x } _ { t } \in \mathbb { R } ^ { C }$ represents the values from C different channels at the timestampe t. When there is more than one channel $( C > 1 )$ , the series is termed multivariate; conversely, if there is only a single channel $( C = 1 )$ , it is referred to as univariate.

## B. Time Series Analytics

Time series analytics studies historical time series data to understand the underlying patterns, supporting various tasks and decision-making. In this paper, we study time series forecasting, anomaly detection, and classification tasks.

Time Series Forecasting refers to the task of predicting future values based on past observations. Given a sequence of inputs $\langle \mathbf { x } _ { t - T + 1 } , \mathbf { x } _ { t - T + 2 } , \ldots , \mathbf { x } _ { t } \rangle$ , the goal is to forecast the future sequence $\langle \mathbf { x } _ { t + 1 } , \mathbf { x } _ { t + 2 } , \ldots , \mathbf { x } _ { t + F } \rangle$ , where $T$ and $F$ are the length of the input sequence and the real future sequence. This forecasting problem can be formalized as a mapping function ${ \mathcal F } ,$ , as shown below:

$$
\mathcal { F } _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t - T + 1 } , \mathbf { x } _ { t - T + 2 } , \ldots , \mathbf { x } _ { t } ) = \big ( \hat { \mathbf { x } } _ { t + 1 } , \hat { \mathbf { x } } _ { { \mathbf { t } } + 2 } , \ldots , \hat { \mathbf { x } } _ { { \mathbf { t } } + { \mathbf { F } } } \big ) ,\tag{1}
$$

where θ refers to the learnable parameters of the model ${ \mathcal F } ,$ and $\langle \hat { \mathbf { x } } _ { t + 1 } , \hat { \mathbf { x } } _ { t + 2 } , \ldots , \hat { \mathbf { x } } _ { t + F } \rangle$ represents the predicted future values.

Time Series Anomaly Detection focuses on identifying points or intervals in a time series that significantly deviate from normal patterns. Given a sequence of inputs $\langle \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { T } \rangle$ , the goal is to detect whether the sequence contains any anomalies, represented as $\langle y _ { 1 } , y _ { 2 } , \dots , y _ { T } \rangle$ , where $y _ { t } \in \{ 0 , 1 \}$ indicates whether the observation $\mathbf { x } _ { t }$ is anomalous. The anomaly detection problem can be formalized as a mapping function ${ \mathcal { A } } ,$ as shown below:

$$
\begin{array} { r } { \mathcal { A } _ { \theta } ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { T } ) = ( y _ { 1 } , y _ { 2 } , \ldots , y _ { T } ) , } \end{array}\tag{2}
$$

where θ refers to the learnable parameters of the model A. Time Series Classification aims to assign time series to predefined categories. Given a sequence of inputs $\langle \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { T } \rangle$ , the objective is to classify it into the correct category $y ~ \in ~ \{ 1 , 2 , \dotsc , M \}$ , where M denotes the total number of classes. This classification task can be formulated as a mapping function C, as shown below:

$$
\begin{array} { r } { \mathcal { C } _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { T } ) = \boldsymbol { y } , } \end{array}\tag{3}
$$

where θ refers to the learnable parameters of the model $\mathcal { C } .$

## C. Time Series Foundation Model

Time series foundation model is pre-trained using a multisource dataset $\mathbf { D } _ { \mathrm { p r e - t r a i n } } = \{ ( X _ { t } ^ { j } , Y _ { t } ^ { j } ) \} _ { j = 1 } ^ { N }$ , where N represents the number of pre-training datasets. For downstream tasks, the model is fine-tuned using the training set of the downstream dataset, $\mathbf { D } _ { \mathrm { t r a i n } } = \{ \mathbf { X } _ { t } ^ { \mathrm { t r a i n } } , \mathbf { Y } _ { t } ^ { \mathrm { t r a i n } } \}$ , and evaluated on the test set $\mathbf { D } _ { \mathrm { t e s t } } = \{ \mathbf { X } _ { t } ^ { \mathrm { t e s t } } , \mathbf { Y } _ { t } ^ { \mathrm { t e s t } } \}$ to infer $\hat { \mathbf { Y } } _ { \mathrm { t e s t } }$ . The datasets $ { \mathbf { D } } _ { \mathrm { p r e - t r a i n } } ,  { \mathbf { D } } _ { \mathrm { t r a i n } }$ , and $\mathbf { D } _ { \mathrm { t e s t } }$ are mutually exclusive. Alternatively, the model can directly perform inference on $\mathbf { D } _ { \mathrm { t e s t } }$ without fine-tuning on $\mathbf { D } _ { \mathrm { t r a i n } }$ , allowing it to predict $\hat { \mathbf { Y } } _ { \mathrm { t e s t } }$ in a zero-shot manner. When applied to different tasks, the model can be viewed as $\mathcal { F }$ for forecasting, A for anomaly detection, and $\mathcal { C }$ for classification.

## IV. METHODOLOGY

## A. Framework of TimeSeek

![](images/16335007339e840ba15b4c7525c4bac9080c04f1673f34e05d6a3746148ef78b.jpg)  
Fig. 2. The framework of TimeSeek.

To build a versatile foundation model capable of handling diverse downstream applications, TimeSeek tackles two primary challenges: the heterogeneity of multi-domain data and the diversity of time series tasks. As illustrated in Figure 2, the overall architecture introduces two core components:

Multi-Domain Feature Extractor (MFE): To address multi-domain data heterogeneity, the MFE serves as the shared, generalized backbone of TimeSeek. Unlike existing methods that rely on fixed-scale windows or channelindependent assumptions, the MFE adaptively captures complex temporal dependencies and channel correlations based on domain-specific characteristics. It achieves this through a progressive multi-scale modeling strategy that hierarchically filters redundant temporal information, combined with an adaptive slot channel modeling mechanism that dynamically clusters correlated channels. This design strictly prevents the introduction of spurious correlations among independent variables. Details about the MFE are provided in Section IV-B.

Multi-Task Collaborative Training (MCT): To address task diversity, we propose the MCT paradigm, which leverages task specificity and commonality. Instead of pre-training entirely separate models for different tasks, TimeSeek incorporates lightweight, task-customized modules: a global encoder for classification, a non-autoregressive decoder for forecasting, and a patch-masking reconstruction module for anomaly detection. These modules are built on top of the shared MFE representations and are collaboratively optimized via a dual-loss mechanism (prediction loss and reconstruction loss). This symbiotic optimization enables distinct pre-training objectives to mutually reinforce the learned representations. Details about MCT are provided in Section IV-C.

![](images/c213fbab14f6bc36ecf99c1e0813f863a3b67aed3884fef928490b74b5d896f6.jpg)  
Fig. 3. The framework of the Multi-Domain Feature Extractor (MFE).

## B. Multi-Domain Feature Extractor

To address the domain heterogeneity of time series data in multi-domain pre-training and generation, we propose a multidomain feature extractor (MFE) that adaptively models complex time series patterns based on the characteristics of each domain. It consists of two components: progressive multi-scale modeling and adaptive slot channel modeling. Specifically, the input time series X undergoes downsampling through convolution to obtain time series at different scales: $\mathbf { X } _ { \mathrm { c } } , \mathbf { X } _ { \mathrm { m } } ,$ and $\mathbf { X } _ { \mathrm { f } }$ . These are progressively fed into their corresponding scale extractors, from coarse to fine scales, for feature extraction and dynamic fusion. Each scale extractor incorporates temporal modeling and adaptive channel modeling to capture complex temporal dependencies and channel correlations, ultimately generating the representations.

1) Progressive Multi-scale Modeling: Time series data exhibits significant multi-scale characteristics, including local fluctuations, short-term dynamics, and long-term trends. Multiscale modeling in pre-training can enhance the understanding of patterns and improve generalization capabilities. However, directly fusing information at different scales may lead to feature conflicts or redundancy problems, and also ignore the differences in preferences for different scales in different domains, thereby weakening the model’s generalization ability. To address this, we propose a progressive multi-scale modeling method. Through progressive fusion, the model progressively extracts and integrates features from coarse to fine scales, building a comprehensive representation that retains both global dependencies and fine-grained patterns. To prevent information conflict during this integration, we further introduce an adaptive gating mechanism to dynamically adjust the fusion weights, filtering redundant features and highlighting key scale information based on domain-specific characteristics.

Multi-scale Construction. Given a raw time series input $\mathbf { X } \in \mathbb { R } ^ { C \times T }$ , we first derive multi-scale representations using convolution networks (Conv) for downsampling, yielding coarse-grained, medium-grained, and fine-grained sequences $\mathbf { X } _ { c } \in \bar { \mathbb { R } } ^ { C \times T / 4 } , \mathbf { X } _ { m } \in \mathbb { R } ^ { \bar { C } \times T / 2 }$ . The original input X is treated as the fine-grained sequence, represented as $\mathbf { \bar { X } } _ { f } ~ \in ~ \mathbb { R } ^ { C \times T }$ where C and T represent the number of channels and the input length of the original input, respectively.

Algorithm 1 Progressive Multi-Scale Modeling   
Input: Raw time series X   
Output: Final representation H   
1: Stage 1: Multi-Scale Construction   
2: Obtain different scales $\mathbf { X } _ { c } , \mathbf { X } _ { m } , \mathbf { X } _ { f }$ via Conv (Eq. 4)   
3: Generate embeddings $\mathbf { E } _ { c } , \mathbf { E } _ { m } , \mathbf { E } _ { f }$   
4: Initialize $\mathbf { E } _ { c } ^ { f u s e } = \bar { \mathbf { E } } _ { c }$ (Start with coarse-grained scale)   
5:   
6: for each scale $s \in \{ c , m , f \}$ do   
7: Stage 2: Feature Extraction per Scale   
8: $\mathbf { H } _ { s } = \mathbf { A }$ daptiveSlotChannelModeling $\left( \mathbf { E } _ { s } ^ { f u s e } \right)$ (Algo  
rithm 2)   
9:   
10: Stage 3: Progressive Fusion (Coarse to Fine)   
11: if scale s $\neq f$ then   
12: $\tilde { \mathbf { F } } _ { s } =$ Interpolate(H<sub>s</sub>) (Align with next finer scale)   
13: Compute gating weights $\mathbf { G } = \sigma ( \mathbf { W } _ { g } [ \tilde { \mathbf { F } } _ { s } ; \mathbf { E } _ { n e x t } ] +$   
b<sub>g</sub>) (Eq. 5)   
14: $\mathbf { E } _ { n e x t } ^ { \breve { f } u s e } = \left( \mathbf { G } \odot \tilde { \mathbf { F } } _ { s } \right) + \mathbf { E } _ { n e x t }$   
15: end if   
16: end for   
17:   
18: return $\mathbf { H } = \mathbf { H } _ { f }$ (Output fine-grained representation)

$$
\begin{array} { r } { \mathbf { X } _ { \mathrm { c } } = \mathrm { C o n v } ( \mathbf { X } , \mathrm { k e r n e l } = 4 , \mathrm { s t r i d e } = 4 ) , } \\ { \mathbf { X } _ { \mathrm { m } } = \mathrm { C o n v } ( \mathbf { X } , \mathrm { k e r n e l } = 2 , \mathrm { s t r i d e } = 2 ) } \end{array}\tag{4}
$$

To preserve local semantic information, a patching strategy partitions each scaled sequence into non-overlapping patches of size P. Each patch is then embedded along the patch size dimension to generate corresponding embeddings $\mathbf { E } _ { c } \in$ $\mathbb { R } ^ { C \times N / 4 \times D } , \mathbf { E } _ { m } \in \breve { \mathbb { R } } ^ { C \times N / 2 \times D } , \mathbf { E } _ { f } ^ { \bullet } \in \mathbb { R } ^ { \breve { C } \times N \times D }$

Progressive Fusion with Adaptive Gating. As illustrated in Figure 2, the MFE progressively fuses multi-scale features from coarse to fine granularity. This hierarchical integration, including the coarse-to-medium fusion and the medium-to-fine fusion, allows complementary interactions across scales while alleviating feature conflicts caused by direct fusion.

![](images/af0fac177c453e1da4e8c97d40410edbc86bbd897e3b1b31e25c0987cc844d71.jpg)  
Fig. 4. Multi-Tasks pre-training framework of TimeSeek, where the Fine-Scale Extractor is put on top to ease description.

Specifically, for the coarse-to-medium fusion, the coarsegrained embedding is first encoded by the coarse-scale extractor, which includes temporal modeling and adaptive channel modeling, to capture temporal dependencies and channel correlations, yielding the coarse-grained features $\mathbf { F } _ { c } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { C \times N / 4 \times D }$ . To align with the medium-grained scale, $\mathbf { F } _ { c }$ undergoes interpolation to double its patch count, yielding $\tilde { \mathbf { F } } _ { c } \in \mathbf { \overline { { R } } } ^ { C \times N / 2 \times D }$ . Then an adaptive gating, which is composed of linear projections and sigmoid activation, dynamically reweights $\tilde { \bar { \mathbf { F } } } _ { c }$ to suppress redundant features.

$$
\mathbf { G } = \sigma ( \mathbf { W } _ { g } [ \tilde { \mathbf { F } } _ { c } ; \mathbf { E } _ { m } ] + \mathbf { b } _ { g } ) , \quad \hat { \mathbf { F } } _ { c } = \mathbf { G } \odot \tilde { \mathbf { F } } _ { c } ,\tag{5}
$$

where $\mathbf { W } _ { g } \in \mathbb { R } ^ { 2 D \times D }$ and $\mathbf { b } _ { g } \in \mathbb { R } ^ { D }$ represent the parameters of linear projections in gating mechanism, $\mathbf { G } \in \dot { \mathbb { R } } ^ { C \times N / 2 \times D }$ represents the weights of $\tilde { \mathbf { F } } _ { c }$ and $\sigma$ denotes the sigmoid activation. Then the gated coarse features $\hat { \mathbf { F } } _ { c }$ are fused with the medium-grained embeddings $\mathbf { E } _ { m }$ via addition, producing enhanced medium-grained embeddings $\mathbf { E } _ { m } ^ { \mathrm { f u s e } } \in \mathbb { R } ^ { C \times N / 2 \times D }$

For the medium-to-fine fusion, we adopt the same operation as used in the coarse-to-medium fusion, taking the embeddings $\mathbf { E } _ { m } ^ { \mathrm { f u s e } }$ as input to generate the fine-grained embeddings $\mathbf { E } _ { f } ^ { \mathrm { f u s e } } \in$ $\mathbb { R } ^ { \stackrel { \cdot \cdot } { C } \times N \times D }$ . Finally, ${ \bf E } _ { f } ^ { \mathrm { f u s e } }$ is input into the fine-scale feature extractor to obtain the final representation H.

2) Adaptive Slot Channel Modeling: To further enhance the generalization of time series representations across different domains, we consider another key characteristic of time-series data: channel correlations. However, these channel correlations show significant differences across different domains. To address this, we propose an adaptive slot channel modeling method, which primarily includes channel correlation adaptive detection based on a channel router and a channel adaptive clustering process based on the slot attention mechanism.

Channel Router. To prevent performance degradation caused by unnecessary channel modeling in time series that lack significant channel correlations, we introduce a channel router that dynamically determines whether meaningful channel correlations exist in the input. Specifically, taking the fused medium-grained embedding $\bf E _ { \mathrm { m } } ^ { \mathrm { f u s e } }$ as an example, the router first generates simplified embedding $\mathbf { E } _ { \mathrm { r } } ~ \in ~ \mathbb { R } ^ { D }$ through dimension-wise average operations: patch-wise and channel-wise averaging. The embedding $\mathbf { E } _ { \mathrm { r } }$ is then projected through linear layers and a softmax operation to compute the routing probability w, where w represents the probabilities of activating channel modeling. A predefined threshold τ controls the subsequent operations: when $w > \tau .$ , the model activates adaptive channel modeling; otherwise, it bypasses this process, and the router directly adds it to the representations obtained by temporal modeling, avoiding redundant computations and potential noise amplification.

Algorithm 2 Adaptive Slot Channel Modeling   
Input: Current scale embedding E, Threshold τ, Slots S   
Output: Channel-refined representation H   
1: Stage 1: Channel Routing   
2: Compute dimension-wise average of E to get simplified   
embedding $\mathbf { E } _ { r }$   
3: Calculate routing probability w via softmax   
4:   
5: Stage 2: Adaptive Clustering & Masking   
6: if $w > \tau$ then   
7: Project E and S to obtain $\mathbf { Q } _ { c } , \mathbf { K } _ { c } , \mathbf { V } _ { c }$ (Eq. 6)   
8: Extract slot information ${ \bf { S } } ^ { \prime } = $ Softmax $( \mathbf { Q } _ { c } \bar { \mathbf { K } } _ { c } ^ { T } / \sqrt { D } ) \mathbf { V } _ { c }$   
9: Update global slots $\mathbf { S } _ { n e w } = \mathbf { S } + \mathbf { M L P } ( \mathbf { S } + \bar { \mathbf { S } } ^ { \prime } )$ (Eq. 7)   
10: Sample slot mask $\mathbf { M } _ { s }$ via Gumbel-Softmax sampling   
(Eq. 8)   
11: Compute masked slots $\mathbf { S } _ { m a s k } = \mathbf { M } _ { s } \odot \mathbf { S } _ { n e w }$   
12: H = Gumbel-Softmax $( { \bf E } \cdot { \bf S } _ { m a s k } ^ { T } / \sqrt { D } ) { \bf S } _ { m a s k }$ (Eq. 9)   
13: else   
14: $\mathbf { H } _ { c } =$ TemporalModeling(E) (Bypass channel mod  
eling)   
15: end if   
16:   
17: return $\mathbf { H } _ { c }$

Channel Slot Attention. Since time series data often exhibit varying channel correlations across different domains, applying a fixed channel modeling method across all domains may introduce spurious correlations between uncorrelated channels, leading to suboptimal generalization. To enhance cross-domain adaptability, we propose an innovative Channel Slot Attention mechanism. By dynamically clustering channels, strongly correlated channels are grouped within the same cluster while preventing unnecessary modeling of independent channels. This adaptive clustering mechanism not only preserves essential channel interactions but also mitigates domain-specific biases. For channel correlation modeling, we initialize K slots $\mathbf { S } \in \mathbb { R } ^ { K \times N \times D } \left( K < C . \right.$ , where $C$ denotes the number of channels and N represents the number of patches.). These slots not only serve as dynamic memory units to store and update channel interaction patterns but also provide a shared slot space for channels in different domains, thereby reducing dependence on specific domains and better adapting to channel correlations in different domains. At each patch, the Slot Attention mechanism automatically aggregates cross-channel information through competitive attention weights, allowing for iterative refinement of slot representations. Specifically, we apply linear transformations to both the slots S and the input embedding $\mathbf { E } _ { \mathrm { m } } ^ { \mathrm { f u s e } }$ to obtain the Query $\mathbf { Q } _ { \mathrm { c } } \in \mathbb { R } ^ { K \times N \times D }$ Key $\mathbf { K } _ { \mathrm { c } }$ , and Value $\mathbf { \ddot { V } _ { c } } \in \mathbb { R } ^ { C \times N \times D }$ . Then, the Channel Slot Attention mechanism uses the slots as queries to selectively focus on the input embedding, producing the channel slot information $\mathbf { S } ^ { \prime } \in \dot { \mathbb { R } } ^ { K \times N \times D }$

$$
\begin{array} { r l r } & { \mathbf { Q } _ { \mathrm { c } } = \mathbf { W } _ { Q } \mathbf { S } , \quad \mathbf { K } _ { \mathrm { c } } = \mathbf { W } _ { K } \mathbf { E } _ { \mathrm { m } } ^ { \mathrm { f u s e } } , \quad \mathbf { V } _ { \mathrm { c } } = \mathbf { W } _ { V } \mathbf { E } _ { \mathrm { m } } ^ { \mathrm { f u s e } } , } & \\ & { \mathbf { S } ^ { \prime } = \mathrm { S o f t m a x } ( \mathbf { Q } _ { \mathrm { c } } \mathbf { K } _ { \mathrm { c } } ^ { T } / \sqrt { D } ) \mathbf { V } _ { \mathrm { c } } , } & \end{array}\tag{6}
$$

where $\mathbf { W } _ { Q } , \ \mathbf { W } _ { K }$ and $\mathbf { W } _ { V } ~ \in ~ \mathbb { R } ^ { D \times D }$ represent learnable projections matrices. Next, we combine the obtained channel slot information $\mathbf { S } ^ { \prime }$ with the previous slots S using a Multi-Layer Perceptron (MLP). This combination is then added to the original slots S to update the slots, resulting in the updated slots $\breve { \mathbf { S } _ { \mathrm { n e w } } } \in \mathbb { R } ^ { K \times N \times D }$

$$
\mathbf { S } _ { \mathrm { n e w } } = \mathbf { S } + \mathrm { M L P } ( \mathbf { S } + \mathbf { S } ^ { \prime } )\tag{7}
$$

To avoid all slots focusing on similar patterns or redundant information, we introduce a masking strategy to dynamically filter less informative slots and retain important ones, while adaptively selecting the number of slots. We employ a lightweight MLP based on the $\mathbf { S } _ { \mathrm { n e w } }$ to generate a probability matrix $\bar { \Pi } \in \mathbb { R } ^ { K \times 1 }$ for slots, indicating whether each slot should be retained or masked. Here, $\Pi _ { i , 0 }$ represents the probability that the i-th slot is retained. We then perform Gumbel-Softmax sampling on the probability matrix Π to obtain the slot mask $\mathbf { M } _ { s }$ . Based on the Slot mask $\mathbf { M } _ { s }$ , we apply a masking operation to $\mathbf { S } _ { \mathrm { n e w } } .$ , resulting in $\mathbf { S } _ { \mathrm { m a s k } }$ , where $\mathbf { S } _ { \mathrm { m a s k } }$ denotes the key channel cluster information.

$$
\mathbf { M } _ { \mathrm { s } } = \mathrm { G u m b e l S o f t m a x } ( \mathbf { I } ) , \quad \mathbf { S } _ { \mathrm { m a s k } } = \mathbf { M } _ { \mathrm { s } } \odot \mathbf { S } _ { \mathrm { n e w } }\tag{8}
$$

Subsequently, we compute attention weights between the masked slots $\mathbf { S } _ { \mathrm { m a s k } }$ and input embeddings $\mathbf { E } _ { \mathrm { m } } ^ { \mathrm { f u s e } }$ , considering that directly using weighted summation may cause the channel representation to mix the information of multiple slots, making it blurred. We use Gumbel-Softmax sampling and $\mathbf { S } _ { \mathrm { m a s k } }$ to produce the final channel representation $\bar { \mathbf { H } } _ { c } \in \mathbb { R } ^ { C \times N \times D }$

$$
\mathbf { H } _ { \mathrm { c } } = \mathrm { G u m b e l S o f t m a x } ( \mathbf { E } _ { \mathrm { m } } ^ { \mathrm { f u s e } } \cdot \mathbf { S } _ { \mathrm { m a s k } } ^ { \mathrm { T } } / \sqrt { \mathrm { D } } ) \mathbf { S } _ { \mathrm { m a s k } }\tag{9}
$$

## C. Multi-Task Collaborative Training

Considering that time series tasks are diverse in downstream scenarios, designing a foundation model that can support multi-task learning becomes essential. However, time series tasks exhibit both heterogeneity and commonality: task heterogeneity may lead to negative interference between tasks, while task commonality provides potential synergies for collaborative learning. To address these, we propose multi-task collaborative training, including task customization and crosstask training. In task customization, we design task-specific modules based on the unique characteristics of each task, enhancing the capabilities in prediction generation, anomaly detection, and categorical discrimination. For cross-task training, we further propose unified components for shared feature extraction and loss computation, strengthening the ability to learn task commonality.

1) Task Customization: We design task-specific customization modules for classification, forecasting, and anomaly detection to better capture the unique characteristics of each task while mitigating negative interference. This design improves the capacity to handle diverse tasks in a unified framework.

As illustrated in Figure 4, multi-scale modeling refers to Section IV-B1, which is stacked together. Take the Fine-scale Extractor as an example, we perform non-overlapping patch segmentation to obtain $\mathbf { X } _ { \mathrm { p a t c h } } \mathbf { \bar { \Pi } } \in \mathbb { R } ^ { C \times N \times P }$ , followed by a linear projection to produce the embeddings $\mathbf { E } _ { \mathrm { p a t c h } } \in \mathbb { R } ^ { C \times N \times D }$

Customization for Classification. Motivated by the inherent reliance of classification on global context to distinguish categories, we propose a novel module designed to extract these global representations and enhance the model’s discriminative capability.

The input time series X is first transformed into the frequency domain using the Fast Fourier Transform. The dominant frequency $f$ is identified by selecting the component with the highest amplitude, from which we compute the corresponding period $P _ { d }$

$$
A = \operatorname { A v g } ( \operatorname { A m p } ( \operatorname { F F T } ( \mathbf { X } ) ) ) , \quad f = \operatorname { a r g T o p 1 } ( A ) , \quad P _ { d } = \lceil \frac { T } { f } \rceil
$$

Based on the period $P _ { d } ,$ the time series is transformed from a 1D temporal space into a 2D temporal space, resulting in $\mathbf { X } _ { \mathrm { d } } \in$ $\mathbb { R } ^ { C \times f \times P _ { d } }$ . To better highlight the temporal differences for classification, we apply a linear resizing operation along the period and frequency dimensions, yielding $\mathbf { \bar { X } } _ { \mathrm { r } } \in \mathbb { R } ^ { C \times D _ { r } \times D _ { r } }$ where $D _ { r }$ represents the feature dimension after resize. The resized time series $\mathbf { X } _ { \mathrm { r } }$ is passed through the global encoder to extract discriminative features, producing the classification embedding $\mathbf { E } _ { \mathrm { c } }$ . Finally, this embedding is flattened and projected to generate the class token $\mathbf { E } _ { \mathrm { c l s } } ~ \in ~ \mathbb { R } ^ { C \times D }$ , which is fed into the feature extractor to obtain the final classification representation $\mathbf { H } _ { \mathrm { c l s } } \in \mathbb { R } ^ { C \times D }$

Customization for Forecasting. To enhance the generative ability for forecasting, we propose a Non-Autoregressive Forecasting module, including a Non-Autoregressive Transformer Decoder (NAT Decoder) and Prediction Layer, that generates all predictions at once, avoiding the error accumulation inherent in autoregressive methods.

After patching, the time series is progressively fed into the corresponding Scale Feature Extractor to model complex patterns at different scales, yielding representations $\mathbf { H } _ { \mathrm { p a t c h } } \in$ $\mathbb { R } ^ { C \times N \times D }$ . Considering that the final patch of the representation retains the overall information of the input sequence and is closest to the future prediction sequence, we replicate the last patch of the representations $\mathbf { H } _ { \mathrm { p a t c h } } \ M$ times as the input and inject Rotary Position Embedding (RoPE) to the NAT Decoder, denoted as $\mathbf { \bar { H } } _ { \mathrm { d } } \in \mathbb { R } ^ { C \times M \times D }$ , where M corresponds to the prediction patch count during pre-training. Subsequently, $\mathbf { H } _ { \mathrm { d } }$ is passed through causal attention, followed by cross-attention with ${ \bf X } _ { \mathrm { p a t c h } }$ , and is refined via a feedforward layer to yield prediction representation $\mathbf { H } _ { \mathrm { f } } \in \mathbb { R } ^ { C \times M \times D }$ . Finally, the prediction layer is applied, followed by flattening, to produce the predicted sequence $Y _ { f } \in \mathbb { R } ^ { C \times F }$

Customization for Anomaly Detection. To enhance the model’s capability to capture localized anomalies, we introduce a patch-masking reconstruction module.

Given the patch embedding $\mathbf { E } _ { \mathrm { p a t c h } }$ , we apply random masking along the patch dimension to produce a masked embedding $\mathbf { E } _ { \mathrm { m a s k } } \in \mathbb { R } ^ { C \times N \times D }$ . Considering that anomaly detection needs global information, we concatenate the masked embedding $\mathbf { E } _ { \mathrm { m a s k } }$ with the class token obtained from the global encoder along the patch dimension to form the new masked embedding $\mathbf { E ^ { \prime } } _ { \mathrm { m a s k } } \in \mathbf { \bar { \mathbb { R } } } ^ { C \times ( N + 1 ) \times D }$ . The embedding is then progressively input into its corresponding scale extractor to model time series patterns, yielding the representation $\mathbf { H } _ { \mathrm { m a s k } }$ , which is finally decoded by the reconstruction layer to get the reconstructed output $\dot { \mathbf { Y } _ { \mathrm { r } } } \in \mathbb { R } ^ { C \times N \times P }$

2) Cross-task Training: Our innovative architectural design effectively addresses the requirements of multi-task pretraining by mitigating negative interference across diverse objectives while enhancing predictive generation, anomaly detection, and categorical discrimination capabilities. Considering both the unique requirements and inherent commonalities among different tasks, we further propose unified components for shared feature extraction and loss computation to facilitate collaborative cross-task learning. This approach enables Time-Seek to leverage complementary information while preserving task-specific advantages.

At the structural level, all tasks share a common multidomain feature extractor that comprehensively models time series patterns, learning generalized representations through mutual collaboration across tasks. For classification and anomaly detection, we concatenate the class token from the global encoder with masked input sequences. This combination injects global semantic information into the reconstruction process, providing valuable prior knowledge for anomaly identification. Simultaneously, the reconstruction pre-training not only reciprocally updates the classification module but also deepens the model’s structured understanding of time series, enabling the downstream classification task to leverage richer representations for more accurate discrimination.

At the loss level, TimeSeek employs a dual-loss mechanism comprising prediction loss $\mathcal { L } _ { \mathrm { p r e d } }$ and reconstruction loss $\mathcal { L } _ { \mathrm { r e c o n } }$ to optimize model parameters. The prediction loss strengthens the model’s generation capability through non-autoregressive forecasting objectives, particularly beneficial for few-shot and zero-shot prediction scenarios. The reconstruction loss focuses on anomaly detection by enforcing precise recovery of masked tokens, while its gradient signals directly enhance the classification module’s feature extraction. This symbiotic optimization ensures the learned representations maintain high adaptability for anomaly detection and classification tasks. In contrast, directly applying the prediction loss to optimize classification could lead to over-smoothing of critical discriminative features, ultimately resulting in conflicting representations between the tasks.

$$
\mathcal { L } _ { \mathrm { p r e - t r a i n } } = \mathcal { L } _ { \mathrm { p r e d } } + \mathcal { L } _ { \mathrm { r e c o n } }\tag{11}
$$

Through these designed interactions, TimeSeek makes different pre-training objectives mutually reinforce each other, ultimately producing robust and generalized representations for diverse tasks.

In the downstream scenarios, we fine-tune the pre-trained model or perform direct inference for each task. For classification, the time series is encoded by the Global Encoder to extract a class token, which is then processed by the MFE to obtain $\mathbf { H } _ { \mathrm { c l s } }$ . These representations are fed into a downstream classifier to produce the final classification results. For forecasting, the time series is processed by the MFE to obtain representations $\mathbf { H } _ { \mathrm { p a t c h } }$ , which are fed into the NAT decoder and prediction layer to generate the output $\mathbf { Y } _ { \mathrm { f } } .$ . The model is optimized using prediction loss. For anomaly detection, the time series undergoes patching and masking before being processed by the MFE to get $\mathbf { H } _ { \mathrm { m a s k } }$ . These representations are fed to the reconstruction layer to produce ${ \bf Y } _ { r }$ , with the model updated by reconstruction loss.

## V. EXPERIMENT

## A. Experimental Setup

1) Datasets: We perform pre-training and evaluation of TimeSeek on diverse datasets and ensure no overlap between the pre-training datasets and the evaluation datasets. More specific information about datasets is in the Supplemental Material.

Pre-train Datasets. To achieve general time series analysis, we compile an extensive array of publicly accessible datasets for pre-training, encompassing various domains such as energy, nature, transportation, web, health, and economics. In total, the pre-training data comprises approximately 1 billion time series data points.

Evaluation Datasets. To validate TimeSeek under multitask, we use the widely used evaluation datasets for forecasting, anomaly detection, and classification tasks, which also span multiple domains to validate the generalizability and versatility of TimeSeek. In our evaluation, we adopt the same train-validation-test split as the baselines for each task to ensure a fair comparison.

• Forecasting. We conduct experiments for forecasting on eight widely recognized forecasting benchmarks, including ETT (ETTm1, ETTm2, ETTh1, ETTh2), Weather, Traffic, Solar, Exchange, and GIFT-Eval Benchmark.

• Anomaly Detection. We evaluate five widely used anomaly detection benchmarks: SMD, MSL, SMAP, SWaT, and PSM.

• Classification. We follow existing works [17] to use 10 datasets for UEA, which feature from 2 to 26 categories for the experiments of classification, including Ethanol-Concentration, FaceDetection, Handwriting, Heartbeat, JapaneseVowels, PEMS-SF, SelfRegulationSCP1, Self-RegulationSCP2, SpokenArabicDigits, and UWaveGestureLibrary.

2) Evaluation Setting: We partition the downstream evaluation dataset into training, validation, and test sets. The evaluation settings consist of the following three scenarios: Zero-shot Setting: The model is directly applied to the test set without fine-tuning, assessing its generalization capability of the foundation models. Few-shot Setting: A small subset of the training data, along with the entire validation set, is used for fine-tuning, evaluating the model’s performance in few-shot scenarios. Full-shot Setting: The model is trained using the entire training and validation datasets, measuring its performance under ample data conditions. In our experiments, TimeSeek is evaluated under the zero-shot and fewshot settings, while the comparison time-series-specific model is trained and tested in the full-shot setting.

3) Baselines: We select both time series foundation models and time-series-specific models of each task as baselines.

Time Series Foundation Baselines. Since existing time series foundation models primarily focus on time series forecasting tasks, we select six state-of-the-art foundation models for comparison in a zero-shot forecasting setting, including Timer [10], MOIRAI [14], Chronos [16], TimesFM [13], UniTS [15], Time-MOE [12],TEMPO [73].

Forecasting Baselines. We select six SOTA time series forecasting models as baselines, including iTransformer [40], FITS [27], TimeMixer [75], PatchTST [38], Crossformer [27], and DLinear [26].

Anomaly Detection Baselines. We compare TimeSeek with 12 anomaly detection baselines for comprehensive evaluations, including the linear transformation-based model OCSVM; the outlier-based methods IForest; the neural network-based models: AutoEncoder (AE) [60], DAGMM [55], LSTM [56], Beat-GAN [76], OmniAnomaly (Omni) [57], Anomaly Transformer (A.T.) [77], DCdetector [78], D3R [64], and ModernTCN [58].

Classification Baselines. We compare TimeSeek with nine baselines, including traditional classification methods: DTW [79], XGBoost [80], and Rocket [81], as well as deep learning-based methods: OS-CNN [82], TapNet [50], PatchTST [38], TimesNet [17], DLinear [26], and Crossformer [37], Moment [11].

4) Metrics: Following prior studies, we use Mean Squared Error (MSE) and Mean Absolute Error (MAE) as evaluation metrics for the forecasting task. The lower the metric values, the better the prediction accuracy. We use Affiliation-based F1- score (F1), a widely used metric for anomaly detection that balances affiliated precision (P) and recall (R) [83]. The higher the metric values, the better the anomaly detection accuracy. Average Accuracy (Avg. ACC) and the count of datasets achieving the highest accuracy (Num. Top-1) as evaluation metrics for classification. The higher the metric values, the better the classification accuracy.

5) Experiment Settings and Implementations: During pretraining, TimeSeek employs the Adam optimizer with an initial learning rate of 0.0005 and undergoes 30 epochs of training, incorporating an early stopping strategy with a patience of 10 epochs. The batch size is set to 2048, and the input time series length is 576. For the forecasting task, all methods predict future values with lengths F = {96, 192, 336, 720} to ensure a fair comparison. Additionally, all models do not employ the drop-last strategy [18]. For the anomaly detection task, we use a sliding window approach with a window size of 96. After obtaining the anomaly scores, we apply the widely used SPOT method, following existing works [64], to determine the threshold for classifying each point as an outlier. All experiments are implemented with PyTorch and the pretraining experiment is executed on four NVIDIA Tesla-A800- 80GB GPUs.

## B. Main Results

1) Forecasting Results: For a fair comparison, we perform zero-shot predictions for each foundational model on downstream datasets that are not part of its pre-training datasets. As shown in Table II, TimeSeek shows leading performance on all datasets, surpassing the previous best method, Time-MOE, by a clear margin and achieving an average MSE reduction of 16.9% and MAE reduction of 13.3%, demonstrating the generalization of the model under different downstream datasets. Compared to forecasting foundation models such as Timer, TimesFM, and Chronos, TimeSeek achieves superior forecasting accuracy. This highlights the effectiveness of its multitask pre-training, which enhances the model’s generalization ability in forecasting tasks. Additionally, compared to UniTS, another multi-task learning model, TimeSeek achieves 32.7% and 23.5% reduction on the MSE and MAE metrics, further underscoring the importance of considering both task commonality and task specificity in multi-task learning. Moreover, TimeSeek also outperforms the channel modeling approach MOIRAI, demonstrating that its adaptive channel modeling for multi-domain data effectively improves generalization.

Table III presents the performance of TimeSeek in the 10% few-shot setting compared to time-series-specific forecasting baselines in the full-shot setting. Despite using significantly less train data from downstream datasets, TimeSeek achieves superior prediction accuracy over these strong baselines. This result highlights TimeSeek’s effectiveness in data-limited scenarios and demonstrates its strong generalization and transfer ability. Compared to single-dataset multi-scale methods like TimeMixer and Crossformer, TimeSeek effectively adapts to multi-domain data through progressive multi-scale modeling, enabling more accurate predictions.

2) Anomaly Detection Results: To ensure a fair comparison in the anomaly detection task, we apply a consistent thresholding protocol across all methods. Unlike baseline models that are trained on the full dataset for each specific scenario, TimeSeek performs zero-shot anomaly detection without any fine-tuning on downstream data, directly conducting inference. As shown in Table V, TimeSeek achieves state-of-the-art results on most datasets without requiring fine-tuning on downstream data. This demonstrates that TimeSeek has a robust general detection capability through extensive multi-domain pre-training, effectively distinguishing between various normal and anomalous patterns.

3) Classification Results: Table VI presents the classification results of TimeSeek in the 10% few-shot setting compared to time-series-specific classification models in the full-shot setting. Remarkably, despite utilizing significantly less training data from downstream datasets, TimeSeek achieves top performance on 6 out of 10 datasets, with an overall accuracy of 0.794. This represents an average accuracy improvement of 7.3% over the second-best model, TimesNet. These results highlight TimeSeek’s exceptional adaptability and generalization capability across diverse classification scenarios, even with limited data.

TABLE II  
THE ZERO-SHOT FORECASTING RESULTS OF TIME SERIES FOUNDATION MODELS. THE AVERAGE RESULTS OF ALL PREDICTED LENGTHS {96, 192, 336, 720} ARE LISTED HERE. A DASH (’-’) DENOTES DATASETS INCLUDED IN THE MODEL’S PRETRAINING AND THEREFORE EXCLUDED FROM TESTING. ETTM REPRESENTS THE AVERAGE RESULTS ACROSS ETTM1 AND ETTM2, WHILE ETTH REPRESENTS THE AVERAGE ACROSS ETTH1 AND ETTH2. BOLD: THE BEST, AND UNDERLINE: THE SECOND BEST.
<table><tr><td>Method</td><td colspan="2">TimeSeek</td><td colspan="2">Timer</td><td colspan="2">TimesFM</td><td colspan="2">Chronos</td><td colspan="2">Time-MoE</td><td colspan="2">UniTS</td><td colspan="2">MOIRAI</td><td colspan="2">TEMPO</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm(Avg)</td><td>0.311</td><td>0.350</td><td>0.636</td><td>0.515</td><td>0.391</td><td>0.389</td><td>0.422</td><td>0.392</td><td>0.322</td><td>0.368</td><td>0.548</td><td>0.457</td><td>0.333</td><td>0.354</td><td>0.391</td><td>0.393</td></tr><tr><td>ETTh(Avg)</td><td>0.390</td><td>0.406</td><td>0.466</td><td>0.441</td><td>0.439</td><td>0.422</td><td>0.462</td><td>0.424</td><td>0.391</td><td>0.413</td><td>0.394</td><td>0.416</td><td>0.432</td><td>0.423</td><td>0.394</td><td>0.413</td></tr><tr><td>Weather</td><td>0.217</td><td>0.250</td><td>0.292</td><td>0.313</td><td></td><td></td><td>0.288</td><td>0.309</td><td>0.256</td><td>0.289</td><td>0.275</td><td>0.298</td><td>0.260</td><td>0.275</td><td>0.282</td><td>0.316</td></tr><tr><td>Traffic</td><td>0.555</td><td>0.367</td><td>0.613</td><td>0.407</td><td></td><td></td><td>0.615</td><td>0.421</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.503</td><td>0.358</td></tr><tr><td>Solar</td><td>0.192</td><td>0.257</td><td>0.771</td><td>0.604</td><td>0.500</td><td>0.397</td><td>0.393</td><td>0.319</td><td>0.411</td><td>0.428</td><td>0.845</td><td>0.669</td><td>0.714</td><td>0.704</td><td>0.416</td><td>0.374</td></tr><tr><td>Exchange</td><td>0.341</td><td>0.390</td><td>0.392</td><td>0.425</td><td>0.386</td><td>0.417</td><td>0.370</td><td>0.412</td><td>0.390</td><td>0.417</td><td>0.432</td><td>0.454</td><td>0.354</td><td>0.400</td><td>0.398</td><td>0.426</td></tr></table>

TABLE III

THE FORECASTING RESULTS OF TIMESEEK AND TIME SERIES-SPECIFIC MODELS ARE PRESENTED, WITH AVERAGE PERFORMANCE REPORTED ACROSS ALL PREDICTION LENGTHS {96, 192, 336, 720}. TIMESEEK UTILIZES ONLY 10% OF THE TRAINING DATA FOR FEW-SHOT LEARNING, WHEREAS TIME SERIES-SPECIFIC MODELS ARE TRAINED USING THE FULL TRAINING DATA. BOLD: THE BEST AND UNDERLINE: THE SECOND BEST. FULL RESULTS ARE IN THE SUPPLEMENTAL MATERIAL.
<table><tr><td>Method</td><td colspan="2">TimeSeek (10%)</td><td colspan="2">iTransformer</td><td colspan="2">FITS</td><td colspan="2">TimeMixer</td><td colspan="2">PatchTST</td><td colspan="2">Crossformer</td><td colspan="2">DLinear</td></tr><tr><td>Metric</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTm(avg)</td><td>0.285</td><td>0.337</td><td>0.303</td><td>0.348</td><td>0.306</td><td>0.345</td><td>0.307</td><td>0.349</td><td>0.303</td><td>0.348</td><td>0.481</td><td>0.479</td><td>0.307</td><td>0.350</td></tr><tr><td>ETTh(avg)</td><td>0.365</td><td>0.394</td><td>0.400</td><td>0.421</td><td>0.372</td><td>0.407</td><td>0.387</td><td>0.418</td><td>0.385</td><td>0.416</td><td>0.666</td><td>0.57</td><td>0.456</td><td>0.455</td></tr><tr><td>Weather</td><td>0.210</td><td>0.246</td><td>0.232</td><td>0.270</td><td>0.244</td><td>0.281</td><td>0.225</td><td>0.263</td><td>0.224</td><td>0.261</td><td>0.233</td><td>0.292</td><td>0.239</td><td>0.290</td></tr><tr><td>Traffic</td><td>0.413</td><td>0.273</td><td>0.397</td><td>0.281</td><td>0.429</td><td>0.302</td><td>0.410</td><td>0.279</td><td>0.397</td><td>0.275</td><td>0.521</td><td>0.282</td><td>0.434</td><td>0.295</td></tr><tr><td>Solar</td><td>0.188</td><td>0.253</td><td>0.202</td><td>0.260</td><td>0.232</td><td>0.268</td><td>0.203</td><td>0.261</td><td>0.200</td><td>0.284</td><td>0.204</td><td>0.232</td><td>0.247</td><td>0.309</td></tr><tr><td>Exchange</td><td>0.332</td><td>0.384</td><td>0.321</td><td>0.384</td><td>0.349</td><td>0.396</td><td>0.385</td><td>0.418</td><td>0.322</td><td>0.385</td><td>0.389</td><td>0.423</td><td>0.349</td><td>0.411</td></tr></table>

TABLE IV

PERFORMANCE OF TIMESEEK ON THE GIFT-EVAL LEADERBOARD. A LOWER MASE OR CRPS INDICATES A BETTER PERFORMANCE. BASELINE RESULTS ARE FROM THE GIFT-EVAL LEADERBOARD.
<table><tr><td>Type</td><td colspan="4">Statistical Methods</td><td colspan="5">Task-Specific Models (Superwised)</td><td colspan="5">Time Series Foundation Models (Zero-Shot)</td></tr><tr><td>Model</td><td>Naïve</td><td>Auto ETS</td><td>Auto ARIMA</td><td>Auto Theta</td><td>DeepAR (2020)</td><td>TiDE (2023)</td><td>N-BEATS (2019)</td><td>PTST. (2022)</td><td>DLinear (2023)</td><td>TimesFM (2023)</td><td>TabPFN (2025)</td><td>Chronos (2024)</td><td>MOIRAI (2024)</td><td>TimeSeek (Ours)</td></tr><tr><td>MASE</td><td>1.270</td><td>1.212</td><td>1.074</td><td>1.090</td><td>1.343</td><td>1.091</td><td>0.938</td><td>0.849</td><td>1.061</td><td>1.077</td><td>0.771</td><td>0.822</td><td>0.875</td><td>0.732</td></tr><tr><td>CRPS</td><td>1.591</td><td>7.489</td><td>0.912</td><td>1.244</td><td>0.853</td><td>0.772</td><td>0.816</td><td>0.587</td><td>0.846</td><td>0.680</td><td>0.544</td><td>0.577</td><td>0.599</td><td>0.528</td></tr></table>

Summary: As illustrated in Tables II to VI, TimeSeek achieves state-of-the-art results across time series forecasting, anomaly detection, and classification tasks. Notably, Time-Seek requires only a small amount of downstream data for fine-tuning or even performs inference directly without any downstream training, while significantly outperforming the second-best model trained on the full downstream data across multiple benchmarks. These indicate that TimeSeek effectively learns generalized knowledge through multi-task and multidomain pre-training, substantially enhancing its generalization and adaptability across diverse scenarios.

## C. Ablation Study

1) Multi-scale modeling: To evaluate the effectiveness of progressive multi-scale modeling, we conduct ablation and substitution experiments using 10% few-shot learning. Scale Feature Summation and Scale Feature Concatenation represent two different approaches for integrating multi-scale features by addition and concatenation operations respectively to examine the necessity of adaptive fusion. Single Scale refers to using only a single fine scale for modeling, serving as a baseline to assess the necessity of multi-scale modeling. As shown in Figure 5, compared with single-scale modeling, TimeSeek achieves better prediction performance, demonstrating the effectiveness of its multi-scale modeling approach. Additionally, compared with scale feature summation and feature concatenation, TimeSeek significantly outperforms these alternative multi-scale fusion methods. Notably, Scale Feature Concatenation yields the worst prediction results, indicating that directly fusing all scale features may introduce scale feature redundancy or noise, which negatively impacts the model. This highlights the necessity of progressive and adaptive multi-scale fusion in multi-domain pre-training.

2) Channel modeling: To validate the effectiveness of adaptive slot channel modeling, we conduct ablation and substitution experiments using 10% few-shot learning. W/O Router refers to removing the channel router, forcing all data to undergo channel slot attention modeling. Attention replaces the proposed channel modeling with standard attention for channel processing. Channel-independent removes channel modeling entirely, ignoring inter-channel correlations. As shown in Figure 6, removing the channel router leads to a drop in prediction accuracy, indicating that not all time series samples exhibit strong channel correlations. This highlights the necessity of assessing whether correlation modeling is needed for each sample. Compared to Channel-independent, Time-

TABLE V  
THE ANOMALY DETECTION RESULTS ARE PRESENTED. TIMESEEK PERFORMS ZERO-SHOT INFERENCE WITHOUT UTILIZING ANY TRAINING DATA, WHEREAS BASELINES ARE TRAINED ON THE FULL TRAINING DATA. BOLD: THE BEST AND UNDERLINE: THE SECOND BEST. MORE METRIC RESULTS ARE IN THE SUPPLEMENTAL MATERIAL.
<table><tr><td colspan="3">Dataset</td><td rowspan="2"></td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td></tr><tr><td>Metric</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>OCSVM</td><td>66.98</td><td>82.03</td><td>73.75</td><td>50.26</td><td>99.86</td><td>66.87</td><td>41.05</td><td>69.37</td><td>51.58</td><td>56.80</td><td>98.72</td><td>72.11</td><td>57.51</td><td>58.11</td><td>57.81</td></tr><tr><td>IForest</td><td>71.94</td><td>94.27</td><td>81.61</td><td>53.87</td><td>94.58</td><td>68.65</td><td>41.12</td><td>68.91</td><td>51.51</td><td>53.03</td><td>99.95</td><td>69.30</td><td>69.66</td><td>88.79</td><td>78.07</td></tr><tr><td>AE</td><td>69.22</td><td>98.48</td><td>81.30</td><td>55.75</td><td>96.66</td><td>70.72</td><td>39.42</td><td>70.31</td><td>50.52</td><td>54.92</td><td>98.20</td><td>70.45</td><td>60.67</td><td>98.24</td><td>75.01</td></tr><tr><td>DAGMM</td><td>63.57</td><td>70.83</td><td>67.00</td><td>54.07</td><td>92.11</td><td>68.14</td><td>50.75</td><td>96.38</td><td>66.49</td><td>59.42</td><td>92.36</td><td>72.32</td><td>68.22</td><td>70.50</td><td>69.34</td></tr><tr><td>LSTM</td><td>60.12</td><td>84.77</td><td>70.35</td><td>58.82</td><td>14.68</td><td>23.49</td><td>55.25</td><td>27.70</td><td>36.90</td><td>49.99</td><td>82.11</td><td>62.15</td><td>57.06</td><td>95.92</td><td>71.55</td></tr><tr><td>BeatGAN</td><td>74.11</td><td>81.64</td><td>77.69</td><td>55.74</td><td>98.94</td><td>71.30</td><td>54.04</td><td>98.30</td><td>69.74</td><td>61.89</td><td>83.46</td><td>71.08</td><td>58.81</td><td>99.08</td><td>73.81</td></tr><tr><td>Omni</td><td>79.09</td><td>75.77</td><td>77.40</td><td>51.23</td><td>99.40</td><td>67.61</td><td>52.74</td><td>98.51</td><td>68.70</td><td>62.76</td><td>82.82</td><td>71.41</td><td>69.20</td><td>80.79</td><td>74.55</td></tr><tr><td>CAE-Ensemble</td><td>73.05</td><td>83.61</td><td>77.97</td><td>54.99</td><td>93.93</td><td>69.37</td><td>62.32</td><td>64.72</td><td>63.50</td><td>62.10</td><td>82.90</td><td>71.01</td><td>73.17</td><td>73.66</td><td>73.42</td></tr><tr><td>A.T.</td><td>54.08</td><td>97.07</td><td>66.42</td><td>51.04</td><td>95.36</td><td>66.49</td><td>56.91</td><td>96.69</td><td>71.65</td><td>53.63</td><td>98.27</td><td>69.39</td><td>54.26</td><td>82.18</td><td>65.37</td></tr><tr><td>DCdetector</td><td>50.93</td><td>95.57</td><td>66.45</td><td>55.94</td><td>95.53</td><td>70.56</td><td>53.12</td><td>98.37</td><td>68.99</td><td>53.25</td><td>98.12</td><td>69.03</td><td>54.72</td><td>86.36</td><td>66.99</td></tr><tr><td>D3R</td><td>64.87</td><td>97.93</td><td>78.02</td><td>66.85</td><td>90.83</td><td>77.02</td><td>61.76</td><td>92.55</td><td>74.09</td><td>60.14</td><td>97.57</td><td>74.39</td><td>73.32</td><td>88.71</td><td>80.29</td></tr><tr><td>ModernTCN</td><td>74.07</td><td>94.79</td><td>83.16</td><td>65.94</td><td>93.00</td><td>77.17</td><td>69.50</td><td>65.45</td><td>67.41</td><td>59.14</td><td>89.22</td><td>71.13</td><td>73.47</td><td>86.83</td><td>79.59</td></tr><tr><td>TimeSeek (zero shot)</td><td>77.09</td><td>94.00</td><td>84.71</td><td>68.22</td><td>94.48</td><td>79.24</td><td>64.98</td><td>88.06</td><td>74.78</td><td>58.92</td><td>94.37</td><td>72.55</td><td>74.58</td><td>86.55</td><td>80.66</td></tr></table>

TABLE VI

THE CLASSIFICATION RESULTS ARE PRESENTED. TIMESEEK UTILIZES ONLY 10% OF THE TRAINING DATA FOR FEW-SHOT LEARNING, WHEREASBASELINES ARE TRAINED USING THE FULL TRAINING DATA. BOLD: THE BEST AND UNDERLINE: THE SECOND BEST.
<table><tr><td>Dataset</td><td>TimeSeek (10%)</td><td>Moment</td><td>TimesNet</td><td>PatchTST</td><td>DLinear</td><td>Crossformer</td><td>OS-CNN</td><td>TapNet</td><td>Rocket</td><td>XGBoost</td><td>DTW</td></tr><tr><td>EthanolConcentration</td><td>0.718</td><td>0.300</td><td>0.357</td><td>0.328</td><td>0.362</td><td>0.380</td><td>0.240</td><td>0.323</td><td>0.447</td><td>0.437</td><td>0.323</td></tr><tr><td>FaceDetection</td><td>0.651</td><td>0.689</td><td>0.686</td><td>0.683</td><td>0.680</td><td>0.687</td><td>0.575</td><td>0.556</td><td>0.694</td><td>0.633</td><td>0.529</td></tr><tr><td>Handwriting</td><td>0.361</td><td>0.351</td><td>0.321</td><td>0.296</td><td>0.270</td><td>0.288</td><td>0.668</td><td>0.357</td><td>0.567</td><td>0.158</td><td>0.286</td></tr><tr><td>Heartbeat</td><td>0.814</td><td>0.737</td><td>0.780</td><td>0.749</td><td>0.751</td><td>0.776</td><td>0.489</td><td>0.751</td><td>0.718</td><td>0.732</td><td>0.717</td></tr><tr><td>Japanese Vowels</td><td>0.986</td><td>0.957</td><td>0.984</td><td>0.975</td><td>0.962</td><td>0.991</td><td>0.991</td><td>0.965</td><td>0.965</td><td>0.865</td><td>0.949</td></tr><tr><td>PEMS-SF</td><td>1.000</td><td>0.855</td><td>0.896</td><td>0.893</td><td>0.751</td><td>0.859</td><td>0.760</td><td>0.751</td><td>0.856</td><td>0.983</td><td>0.711</td></tr><tr><td>SelfRegulationSCP1</td><td>0.918</td><td>0.887</td><td>0.918</td><td>0.907</td><td>0.873</td><td>0.921</td><td>0.835</td><td>0.652</td><td>0.866</td><td>0.846</td><td>0.777</td></tr><tr><td>SelfRegulationSCP2</td><td>0.594</td><td>0.550</td><td>0.572</td><td>0.578</td><td>0.505</td><td>0.583</td><td>0.532</td><td>0.550</td><td>0.514</td><td>0.489</td><td>0.539</td></tr><tr><td>SpokenArabicDigits</td><td>0.936</td><td>0.994</td><td>0.990</td><td>0.983</td><td>0.814</td><td>0.979</td><td>0.997</td><td>0.983</td><td>0.630</td><td>0.696</td><td>0.963</td></tr><tr><td>UWaveGestureLibrary</td><td>0.959</td><td>0.900</td><td>0.853</td><td>0.858</td><td>0.821</td><td>0.853</td><td>0.927</td><td>0.894</td><td>0.944</td><td>0.759</td><td>0.903</td></tr><tr><td>Avg. ACC</td><td>0.794</td><td>0.722</td><td>0.736</td><td>0.725</td><td>0.679</td><td>0.732</td><td>0.701</td><td>0.678</td><td>0.720</td><td>0.660</td><td>0.670</td></tr><tr><td>Num. Top-1</td><td>6</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>2</td><td>0</td><td>1</td><td>0</td><td>0</td></tr></table>

![](images/9bddc7a976ec5b235f6549483fcacef439554718ae053226ac9f40f93f740ee3.jpg)

![](images/a739ef841fb0c2d75e4e0d12dbfc952afc75d46bdeb12c390a11f57fa1e832bd.jpg)  
Fig. 5. Ablation about multi-scale modeling on Weather and ETTm1 datasets. Sum and Concat represent scale feature summation and scale feature concatenation, respectively. Single Scale denotes modeling with only the fine scale. The results are averaged across four prediction lengths.

Seek significantly outperforms it, demonstrating that ignoring inter-channel dependencies negatively impacts performance. Additionally, compared to using standard attention for modeling channels, TimeSeek achieves better results. Notably, on the ETTm1 dataset, the standard attention-based method performs the worst, suggesting that a uniform channel modeling strategy may introduce spurious correlations between uncorrelated channels, ultimately leading to degraded performance. Therefore, an adaptive channel modeling approach is necessary to dynamically cluster channels, ensuring that strongly correlated channels are grouped together.

![](images/4627e48fbe73a0d63fab7ba9454506eb3abfed4421e7e2de263172d92f8ed8a3.jpg)

![](images/272233e7bebfa7642229fcbfbbf23b0cc91d67105a4fc1f25cfa388c2b35d2a8.jpg)  
Fig. 6. Ablation about channel modeling for the Weather and ETTm1 datasets. W/O Router refers to removing the channel router. Attention represents using standard attention for channel modeling. Channel-independent denotes ignoring channel modeling entirely. The results are averaged across four prediction lengths.

3) Classification module: To validate the effectiveness of the proposed classification module, we conduct replacement experiments on five benchmark datasets: EthanolConcentration, Heartbeat, SelfRegulationSCP1, SelfRegulation-SCP2, and UwaveGestureLibrary. Our comparative analysis focused on contrasting TimeSeek’s classification results with the vanilla class token approach in ViT [8] — a widely adopted classification method that directly initializes trainable class tokens. The experimental results demonstrate that TimeSeek achieves significant improvements in classification accuracy, outperforming the baseline method by an average margin of 17%. This substantial performance gain indicates that Time-Seek effectively captures comprehensive global information from time series through its innovative architecture design, thereby enabling more informed classification decisions.

![](images/c841caa0fc93cdec3abf98a76733e11d4ba3eeb9275872feb1942e2c9144edb6.jpg)  
Fig. 7. Ablation study on the classification module across five datasets. Vanilla class token denotes directly initializing a trainable class token for classification.

## D. Model Analysis

1) Fine-tune with downstream data: To verify the impact of downstream fine-tuning data volume on model performance, we evaluate the model’s scalability by gradually increasing the amount of downstream Traffic and Solar datasets. As shown in Figure 8, the forecast performance steadily improves as the downstream data increases, reaching best when fine-tuned with 100% of the data. This demonstrates the strong scalability of our proposed model.

![](images/06abf120b3693227a86bcb864f64ae64605f7d84052bd54f449175cd0ea1f062.jpg)

![](images/d09411035c29834279151b13853918bceb81e9a06a700fdfe78b4a373a6eacfa.jpg)  
Fig. 8. Results of fine-tuning TimeSeek under different data percentages with prediction length 96.

2) Parameter sensitivity analysis.: We conduct parameter sensitivity experiments on Traffic and Solar datasets to investigate the effect of the number of slots K. Under the setting with prediction length 96, Figure 9 reveals that insufficient slots (K=5) lead to suboptimal performance, indicating that a sufficient number of slots are needed to store the complex inter-channel correlations for multi-domain data. And K depends on pre-training data, it may be small if data are more homogeneous. As the number of slots increases $( K \le 1 0 )$ the model’s prediction accuracy stabilizes, showing that it effectively masks redundant information while preserving key content, thus avoiding interference from irrelevant information.

![](images/f2c3753c682dc963e7765cdcb2293eaf84e5f2a0108af78303373249e13749ce.jpg)

![](images/fe46810983719adc92ca1d25e188dd202a2030a7a4d32e5ba21331cdfa21601d.jpg)  
Fig. 9. The effect of the number of slots K.

3) Efficiency analysis.: Model efficiency is a key factor in determining whether a foundation model can be effectively deployed in downstream scenarios. To evaluate the efficiency of TimeSeek and other foundation models, we conduct zeroshot forecasting on the Weather dataset with a prediction length of 96. The comparison is made from three aspects: zeroshot inference time, parameter count, and prediction accuracy. As shown in Figure 10, TimeSeek demonstrates significant advantages in all three aspects, highlighting its superior computational efficiency and strong applicability to downstream scenarios. Notably, compared to large-parameter models such as Time-MOE, TimeSeek achieves better inference speed, lower parameters, and higher prediction accuracy. At the same time, the pretraining cost of TimeSeek is much lower than TimeMOE. This indicates that simply increasing model size is not the only viable approach to building foundation models. Instead, effectively leveraging multi-domain time series data and diverse tasks provides a more efficient way to construct foundation models.

![](images/d52e202033dcbc38ee8e5c4c277f2a3a55d266c2c5c50d7a10d85f397799ff22.jpg)  
Fig. 10. A comparison of inference time and the parameter counts of time series foundation models, along with their zero-shot performance on the Weather dataset.

## VI. CONCLUSION

In this paper, we propose TimeSeek, a novel generalized time series analytics model that adaptively models time series patterns from multi-domain data and unifies multiple tasks in pre-training. From the perspective of multi-domain data, we propose a Multi-Domain Feature Extractor that adaptively captures temporal dependencies and channel correlations based on domain-specific characteristics, including progressive multiscale modeling and adaptive slot channel modeling. From the perspective of the diversity of tasks, we propose Multi-Task Collaborative Training, which leverages task specificity and task commonality through task customization and cross-task training, effectively unifying multiple tasks in pre-training and better adapting to diverse downstream scenarios. After pretraining on large-scale time series data, TimeSeek achieves state-of-the-art performance in forecasting, anomaly detection, and classification tasks across both few-shot learning and zeroshot inference situations, demonstrating strong task versatility and generalization ability. At the same time, compared with existing time series foundation models, TimeSeek shows better inference efficiency and analysis accuracy. We provide our code at https://github.com/decisionintelligence/TimeSeek.

## VII. ACKNOWLEDGMENT

This work was supported by the National Natural Science Foundation of China (62406112, 62372179).

## REFERENCES

[1] I. Kaastra and M. S. Boyd, “Designing a neural network for forecasting financial and economic time series,” Neurocomputing, vol. 10, no. 3, pp. 215–236, 1996.

[2] T. Li, Z. Liu, Y. Shen, X. Wang, H. Chen, and S. Huang, “MASTER: market-guided stock transformer for stock price forecasting,” in AAAI, 2024, pp. 162–170.

[3] Y. Liang, Y. Xia, S. Ke, Y. Wang, Q. Wen, J. Zhang, Y. Zheng, and R. Zimmermann, “Airformer: Predicting nationwide air quality in china with transformers,” in AAAI, 2023, pp. 14 329–14 337.

[4] N. L. Nihan and K. O. Holmesland, “Use of the box and jenkins time series technique in traffic forecasting,” Transportation, 1980.

[5] O. F. Demirel, S. Zaim, A. C¸ alis¸kan, and P. <sup>¨</sup> Ozuyar, “Forecasting natural<sup>¨</sup> gas consumption in istanbul using neural networks and multivariate time series methods,” Turkish Journal of Electrical Engineering and Computer Sciences, 2012.

[6] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar\` et al., “Llama: open and efficient foundation language models. arxiv,” arXiv preprint arXiv:2302.13971, 2023.

[7] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language models are few-shot learners,” in NeurIPS, 2020.

[8] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.

[9] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in ICCV, 2021, pp. 9992–10 002.

[10] Y. Liu, H. Zhang, C. Li, X. Huang, J. Wang, and M. Long, “Timer: Generative pre-trained transformers are large time series models,” in ICML, 2024.

[11] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “MOMENT: A family of open time-series foundation models,” in ICML, 2024.

[12] X. Shi, S. Wang, Y. Nie, D. Li, Z. Ye, Q. Wen, and M. Jin, “Time-moe: Billion-scale time series foundation models with mixture of experts,” CoRR, vol. abs/2409.16040, 2024.

[13] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in ICML, 2024.

[14] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” in ICML, 2024.

[15] S. Gao, T. Koker, O. Queen, T. Hartvigsen, T. Tsiligkaridis, and M. Zitnik, “Units: A unified multi-task time series model,” in NeurIPS, 2024.

[16] A. F. Ansari, L. Stella, A. C. Turkmen, X. Zhang, P. Mercado,¨ H. Shen, O. Shchur, S. S. Rangapuram, S. Pineda-Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, M. W. Mahoney, K. Torkkola, A. G. Wilson, M. Bohlke-Schneider, and Y. Wang, “Chronos: Learning the language of time series,” CoRR, vol. abs/2403.07815, 2024.

[17] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” in ICLR, 2023.

[18] X. Qiu, J. Hu, L. Zhou, X. Wu, J. Du, B. Zhang, C. Guo, A. Zhou, C. S. Jensen, Z. Sheng, and B. Yang, “TFB: towards comprehensive and fair benchmarking of time series forecasting methods,” Proc. VLDB Endow., vol. 17, no. 9, pp. 2363–2377, 2024.

[19] C. Faloutsos, J. Gasthaus, T. Januschowski, and Y. Wang, “Forecasting big time series: Old and new,” Proc. VLDB Endow., vol. 11, no. 12, pp. 2102–2105, 2018.

[20] P. Chen, Z. Zhuang, H. Chen, J. Huang, A. Yang, M. Wu, Y. Liu, X. Dai, and Z. Dong, “Beyond numerical time series: A unified benchmark for multimodal forecasting with heterogeneous context,” arXiv preprint arXiv:2609.15087, 2026.

[21] Z. Z. Darban, G. I. Webb, S. Pan, C. Aggarwal, and M. Salehi, “Deep learning for time series anomaly detection: A survey,” ACM Comput. Surv., vol. 57, no. 1, pp. 15:1–15:42, 2025.

[22] Q. Liu, P. Boniol, T. Palpanas, and J. Paparrizos, “Time-series anomaly detection: Overview and new trends,” VLDB, vol. 17, no. 12, pp. 4229– 4232, 2024.

[23] J. Paparrizos, Y. Kang, P. Boniol, R. S. Tsay, T. Palpanas, and M. J. Franklin, “Tsb-uad: an end-to-end benchmark suite for univariate timeseries anomaly detection,” VLDB, vol. 15, no. 8, pp. 1697–1711, 2022.

[24] J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore, and M. J. Franklin, “Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection,” Proceedings of the VLDB Endowment, vol. 15, no. 11, pp. 2774–2787, 2022.

[25] N. M. Foumani, L. Miller, C. W. Tan, G. I. Webb, G. Forestier, and M. Salehi, “Deep learning for time series classification and extrinsic regression: A current survey,” ACM Comput. Surv., vol. 56, no. 9, pp. 217:1–217:45, 2024.

[26] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in AAAI, 2023, pp. 11 121–11 128.

[27] Z. Xu, A. Zeng, and Q. Xu, “FITS: modeling time series with 10k parameters,” in ICLR, 2024.

[28] S. Zhong, S. Song, W. Zhuo, G. Li, Y. Liu, and S. G. Chan, “A multiscale decomposition mlp-mixer for time series analysis,” Proc. VLDB Endow., vol. 17, no. 7, pp. 1723–1736, 2024.

[29] V. Flunkert, D. Salinas, and J. Gasthaus, “Deepar: Probabilistic forecasting with autoregressive recurrent networks,” CoRR, vol. abs/1704.04110, 2017.

[30] R.-G. Cirstea, B. Yang, and C. Guo, “Graph attention recurrent neural networks for correlated time series forecasting.” in SIGKDD, 2019.

[31] R. Cirstea, T. Kieu, C. Guo, B. Yang, and S. J. Pan, “Enhancenet: Plugin neural networks for enhancing correlated time series forecasting,” in ICDE, 2021, pp. 1739–1750.

[32] K. Zhao, C. Guo, Y. Cheng, P. Han, M. Zhang, and B. Yang, “Multiple time series forecasting with dynamic graph modeling,” VLDB., vol. 17, no. 4, pp. 753–765, 2023.

[33] X. Wu, D. Zhang, C. Guo, C. He, B. Yang, and C. S. Jensen, “Autocts: Automated correlated time series forecasting,” VLDB, vol. 15, no. 4, pp. 971–983, 2021.

[34] Y. Cui, K. Zheng, D. Cui, J. Xie, L. Deng, F. Huang, and X. Zhou, “METRO: A generic graph neural network framework for multivariate time series forecasting,” VLDB, vol. 15, no. 2, pp. 224–236, 2021.

[35] M. Ma, J. Hu, C. S. Jensen, F. Teng, P. Han, Z. Xu, and T. Li, “Learning time-aware graph structures for spatially correlated time series forecasting,” in ICDE, 2024, pp. 4435–4448.

[36] R. Cirstea, C. Guo, B. Yang, T. Kieu, X. Dong, and S. Pan, “Triformer: Triangular, variable-specific attentions for long sequence multivariate time series forecasting,” in IJCAI, 2022, pp. 1994–2001.

[37] Y. Zhang and J. Yan, “Crossformer: Transformer utilizing crossdimension dependency for multivariate time series forecasting,” in ICLR, 2023.

[38] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in ICLR, 2023.

[39] P. Chen, Y. Zhang, Y. Cheng, Y. Shu, Y. Wang, Q. Wen, B. Yang, and C. Guo, “Pathformer: Multi-scale transformers with adaptive pathways for time series forecasting,” in ICLR, 2024.

[40] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” in ICLR, 2024.

[41] Y. Zheng, Q. Liu, E. Chen, Y. Ge, and J. L. Zhao, “Time series classification using multi-channels deep convolutional neural networks,” in WAIM, vol. 8485, 2014, pp. 298–310.

[42] F. Karim, S. Majumdar, H. Darabi, and S. Harford, “Multivariate lstmfcns for time series classification,” Neural Networks, vol. 116, pp. 237– 245, 2019.

[43] Y. Yao, H. Jie, L. Chen, T. Li, Y. Gao, and S. Wen, “Tsec: An efficient and effective framework for time series classification,” in ICDE, 2024, pp. 1394–1406.

[44] H. I. Fawaz, B. Lucas, G. Forestier, C. Pelletier, D. F. Schmidt, J. Weber, G. I. Webb, L. Idoumghar, P. Muller, and F. Petitjean, “Inceptiontime: Finding alexnet for time series classification,” Data Min. Knowl. Discov., vol. 34, no. 6, pp. 1936–1962, 2020.

[45] A. Ismail-Fawaz, M. Devanne, S. Berretti, J. Weber, and G. Forestier, “LITE: light inception with boosting techniques for time series classification,” in DSAA, 2023, pp. 1–10.

[46] N. M. Foumani, C. W. Tan, G. I. Webb, and M. Salehi, “Improving position encoding of transformers for multivariate time series classification,” Data Min. Knowl. Discov., vol. 38, no. 1, pp. 22–48, 2024.

[47] M. Wistuba, J. Grabocka, and L. Schmidt-Thieme, “Ultra-fast shapelets for time series classification,” CoRR, vol. abs/1503.05018, 2015.

[48] I. Karlsson, P. Papapetrou, and H. Bostrom, “Generalized random¨ shapelet forests,” Data Min. Knowl. Discov., vol. 30, no. 5, pp. 1053– 1085, 2016.

[49] G. Li, B. Choi, J. Xu, S. S. Bhowmick, K. Chun, and G. L. Wong, “Efficient shapelet discovery for time series classification,” TKDE, vol. 34, no. 3, pp. 1149–1163, 2022.

[50] X. Zhang, Y. Gao, J. Lin, and C. Lu, “Tapnet: Multivariate time series classification with attentional prototypical network,” in AAAI, 2020, pp. 6845–6852.

[51] Z. Yue, Y. Wang, J. Duan, T. Yang, C. Huang, Y. Tong, and B. Xu, “Ts2vec: Towards universal representation of time series,” in AAAI, 2022, pp. 8980–8987.

[52] E. Eldele, M. Ragab, Z. Chen, M. Wu, C. K. Kwoh, X. Li, and C. Guan, “Time-series representation learning via temporal and contextual contrasting,” in IJCAI, 2021, pp. 2352–2359.

[53] G. Woo, C. Liu, D. Sahoo, A. Kumar, and S. C. H. Hoi, “Cost: Contrastive learning of disentangled seasonal-trend representations for time series forecasting,” in ICLR, 2022.

[54] J. Dong, H. Wu, H. Zhang, L. Zhang, J. Wang, and M. Long, “Simmtm: A simple pre-training framework for masked time-series modeling,” in NeurIPS, 2023.

[55] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen, “Deep autoencoding gaussian mixture model for unsupervised anomaly detection,” in ICLR, 2018.

[56] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstr¨ om, “Detecting spacecraft anomalies using lstms and non-¨ parametric dynamic thresholding,” in SIGKDD, 2018, pp. 387–395.

[57] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust anomaly detection for multivariate time series through stochastic recurrent neural network,” in SIGKDD, 2019, pp. 2828–2837.

[58] D. Luo and X. Wang, “Moderntcn: A modern pure convolution structure for general time series analysis,” in ICLR, 2024.

[59] X. Chen, L. Deng, F. Huang, C. Zhang, Z. Zhang, Y. Zhao, and K. Zheng, “Daemon: Unsupervised anomaly detection and interpretation for multivariate time series,” in ICDE, 2021, pp. 2225–2230.

[60] D. Campos, T. Kieu, C. Guo, F. Huang, K. Zheng, B. Yang, and C. S. Jensen, “Unsupervised time series outlier detection with diversity-driven convolutional ensembles,” Proc. VLDB Endow., vol. 15, no. 3, pp. 611– 623, 2021.

[61] Z. Li, Y. Zhao, J. Han, Y. Su, R. Jiao, X. Wen, and D. Pei, “Multivariate time series anomaly detection and interpretation using hierarchical intermetric and temporal embedding,” in SIGKDD, 2021, pp. 3220–3230.

[62] T. Kieu, B. Yang, C. Guo, C. S. Jensen, Y. Zhao, F. Huang, and K. Zheng, “Robust and explainable autoencoders for unsupervised time series outlier detection,” in ICDE, 2022, pp. 3038–3050.

[63] Y. Fang, J. Xie, Y. Zhao, L. Chen, Y. Gao, and K. Zheng, “Temporalfrequency masked autoencoders for time series anomaly detection,” in ICDE, 2024, pp. 1228–1241.

[64] C. Wang, Z. Zhuang, Q. Qi, J. Wang, X. Wang, H. Sun, and J. Liao, “Drift doesn’t matter: Dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection,” in NeurIPS, 2023.

[65] T. Kieu, B. Yang, C. Guo, R.-G. Cirstea, Y. Zhao, Y. Song, and C. S. Jensen, “Anomaly detection in time series with robust variational quasirecurrent autoencoders,” in ICDE, 2022, pp. 1342–1354.

[66] V. Ekambaram, A. Jati, P. Dayama, S. Mukherjee, N. Nguyen, W. M. Gifford, C. Reddy, and J. Kalagnanam, “Tiny time mixers (ttms): Fast pre-trained models for enhanced zero/few-shot forecasting of multivariate time series,” in NeurIPS, 2024.

[67] Y. Wang, Y. Qiu, P. Chen, K. Zhao, Y. Shu, Z. Rao, L. Pan, B. Yang, and C. Guo, “ROSE: register assisted general time series forecasting with decomposed frequency learning,” CoRR, vol. abs/2405.17478, 2024.

[68] H. Kamarthi and B. A. Prakash, “Large pre-trained time series models for cross-domain time series analysis tasks,” in NeurIPS, 2024.

[69] P. Chen, S. Wang, S. Hu, X. Wu, Y. Shu, Z. Rao, M. Wang, Y. Li, B. Yang, and C. Guo, “Empowering time series analysis with largescale multimodal pretraining,” arXiv preprint arXiv:2602.05646, 2026.

[70] A. Auer, P. Podest, D. Klotz, S. Bock, G. Klambauer, and S. Hochreiter,¨ “Tirex: Zero-shot forecasting across long and short horizons with enhanced in-context learning,” NeurIPS, 2026.

[71] L. Graf, T. Ortner, S. WoLsniak, A. Pantazi et al., “Flowstate: Sampling rate invariant time series forecasting,” arXiv preprint arXiv:2508.05287, 2025.

[72] A. F. Ansari, O. Shchur, J. Kuken, A. Auer, B. Han, P. Mercado, S. S.¨ Rangapuram, H. Shen, L. Stella, X. Zhang et al., “Chronos-2: From univariate to universal forecasting,” arXiv preprint arXiv:2510.15821, 2025.

[73] D. Cao, F. Jia, S. O. Arik, T. Pfister, Y. Zheng, W. Ye, and Y. Liu,<sup>¨</sup> “TEMPO: prompt-based generative pre-trained transformer for time series forecasting,” in ICLR, 2024.

[74] D. Cao, W. Ye, Y. Zhang, and Y. Liu, “Timedit: General-purpose diffusion transformers for time series foundation model,” arXiv preprint arXiv:2409.02322, 2024.

[75] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, and J. Zhou, “Timemixer: Decomposable multiscale mixing for time series forecasting,” in ICLR, 2024.

[76] B. Zhou, S. Liu, B. Hooi, X. Cheng, and J. Ye, “Beatgan: Anomalous rhythm detection using adversarially generated time series,” in IJCAI, 2019, pp. 4433–4439.

[77] J. Xu, H. Wu, J. Wang, and M. Long, “Anomaly transformer: Time series anomaly detection with association discrepancy,” in ICLR, 2022.

[78] Y. Yang, C. Zhang, T. Zhou, Q. Wen, and L. Sun, “Dcdetector: Dual attention contrastive representation learning for time series anomaly detection,” in SIGKDD, 2023, pp. 3033–3045.

[79] T. Rakthanmanon, B. J. L. Campana, A. Mueen, G. E. A. P. A. Batista, M. B. Westover, Q. Zhu, J. Zakaria, and E. J. Keogh, “Searching and mining trillions of time series subsequences under dynamic time warping,” in SIGKDD, 2012, pp. 262–270.

[80] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in SIGKDD, 2016, pp. 785–794.

[81] A. Dempster, F. Petitjean, and G. I. Webb, “ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels,” Data Min. Knowl. Discov., vol. 34, no. 5, pp. 1454–1495, 2020.

[82] W. Tang, G. Long, L. Liu, T. Zhou, M. Blumenstein, and J. Jiang, “Omni-scale cnns: a simple and effective kernel size configuration for time series classification,” in ICLR, 2022.

[83] A. Huet, J. M. Navarro, and D. Rossi, “Local evaluation of time series anomaly detection algorithms,” in SIGKDD, 2022, pp. 635–645.