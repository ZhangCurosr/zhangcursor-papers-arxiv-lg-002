# ScaleBrain: A Scale-Consistent Framework for EEG Foundation Models

Minyu Zhou, Yajing Wu, Yongqiang Tang, Wang Jia and Wensheng Zhang

Abstract—Foundation models pre-trained on large unlabeled electroencephalography (EEG) corpora have emerged as a transformative paradigm for Brain-Computer Interface (BCI) applications. Recent EEG foundation models have shown that multi-scale representations derived from fixed-length patches can improve EEG representation learning. Existing frameworks commonly construct temporal scales through feature-level operations. However, these operations leave the correspondence between learned features and scale-specific characteristics of the original EEG signals implicit. Moreover, when the selfsupervised objective is not aligned with the multi-scale input representations, information at some temporal scales may receive insufficient supervision during pre-training. In light of this, we propose ScaleBrain, a scale-consistent pre-training framework that explicitly coordinates multi-scale representations of EEG from both the model input and optimization objective sides. Specifically, ScaleBrain introduces multi-scale EEG tokenization to represent the same EEG segment through analysis windows with different temporal spans and shared temporal anchors, and pairs these representations with scale-specific reconstruction to align with the multi-scale input structure and provide supervision across temporal resolutions during self-supervised learning. We further develop an axial spatio-temporal attention mechanism to contextualize the fused multi-scale representation along spatial and temporal dimensions. Extensive evaluations on six public BCI benchmarks against 12 representative baselines show that ScaleBrain achieves the highest mean Balanced Accuracy on five datasets and ranks third on the remaining one, with only 5.0M parameters, demonstrating strong performance and favorable parameter efficiency.

Index Terms—Electroencephalography, Foundation models, Multi-scale modeling, Self-supervised learning.

## I. INTRODUCTION

Understanding and interpreting brain activity has long been a central goal in cognitive neuroscience and artificial intel-

This work was supported in part by the National Key Research and Development Program of China (No. 2021ZD0201600), in part by the National Natural Science Foundation of China under Grant 62476274 and Grant U22B2048, and in part by the Special Project of Joint Funding for Municipal, University (Institute) and Enterprise under the Guangzhou Basic Research Program (2024A03J0395), Science and Technology Projects in Guangzhou (2024D03J0010). (Corresponding authors: Yajing Wu and Yongqiang Tang.) Science. Nankai University

Minyu Zhou is with the College of Computer Science, Nankai University, Tianjin 300350, China (e-mail: minyu.zhou@mail.nankai.edu.cn).

Yajing Wu and Yongqiang Tang are with the State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences, Beijing 100190, China (e-mail: yajing.wu@ia.ac.cn; yongqiang.tang@ia.ac.cn).

![](images/b9dcdc2b9dc2961e483a6ff3e0e2fb59325a5a5ec1e03fc7c01480a6d7279948.jpg)  
Fig. 1. Comparison of pre-training paradigms for EEG foundation models. Unlike (a) the existing multi-scale framework which focuses on the input side with different temporal anchors, our proposed (b) ScaleBrain explicitly maintains scale consistency by coordinating coarse-to-fine representations across input construction and self-supervised optimization. Long/mid/shortwin: long/middle/short window.

ligence [1], [2]. Scalp-based electroencephalography (EEG), which records electrical potentials at the scalp surface, is widely used as a non-invasive brain recording modality due to its high temporal resolution and practical feasibility [3]. By decoding multi-channel EEG signals, Brain-Computer Interface (BCI) technologies have enabled a wide range of applications, including motor imagery, emotion recognition, sleep staging, and abnormality detection [4]–[7].

Over the past few decades, EEG decoding methods have evolved through three major paradigms: traditional statistical methods [8], [9], task-specific deep learning models [10]– [12], and more recently, EEG foundation models that learn transferable representations across diverse tasks [13]–[15]. This paradigm shift toward EEG foundation models follows the success of large-scale pre-training in natural language processing and computer vision, where Self-Supervised Learning (SSL) enables transferable representation learning from unlabeled data. Similarly, EEG foundation models leverage SSL objectives such as masked signal modeling [15], [16] and neural code prediction [14] to learn from large-scale unlabeled

EEG corpora, capturing transferable patterns in EEG signals without requiring exhaustive manual annotations. Once pretrained, these representations can be adapted to downstream BCI tasks through fine-tuning, especially when labeled data is limited. However, the effectiveness of such transfer depends on whether the pre-trained representations adequately capture the diverse temporal characteristics of EEG signals.

Neural activity can exhibit both slowly varying rhythmic dynamics and rapidly changing transient patterns [17], [18]. These dynamics may be represented at different temporal and spectral resolutions, motivating the use of multiple modeling scales in EEG analysis. As illustrated in Fig. 1(a), recent EEG foundation models such as CSBrain [15] and CodeBrain [19], which build multi-scale representations based on patches and convolutions at the feature level, have demonstrated the potential of incorporating multi-scale mechanisms into largescale EEG representation learning. Nevertheless, existing approaches commonly introduce temporal scales through featurelevel operations, potentially overlooking scale-dependent temporal and spectral characteristics of the original EEG signals. The correspondence between the resulting representations and these signal characteristics remains implicit. Moreover, when the self-supervised objective is not aligned with the multiscale representations, information at some temporal scales may receive insufficient supervision during pre-training.

On the one hand, constructing multiple temporal scales introduces a correspondence issue. Applying an EEG tokenizer on the EEG signals with multiple window lengths produces tokens with different temporal spans and spectral resolutions. Although these tokens originate from the same EEG segment, their temporal semantics are not directly comparable unless their temporal anchors are explicitly aligned. Without such alignment, directly combining heterogeneous tokens may obscure the correspondence between representations at different temporal scales and make subsequent multi-scale fusion difficult. On the other hand, multi-scale design at the input side is insufficient if the self-supervised objective does not preserve the same scale structure. Existing EEG foundation models typically rely on a single-scale self-supervised objective, which provides supervision at only one temporal resolution. Such an objective may bias representation learning toward the characteristics of a particular scale and leaves information in other temporal scales insufficiently supervised, leading to potentially incomplete representations. Consequently, an effective multi-scale foundation model should therefore maintain consistency between scale diversity in the input representation and scale-specific supervision in the self-supervised objective. Meanwhile, representations at different scales should be jointly fused and contextualized as they are inherently coupled.

In this study, we propose ScaleBrain, a scale-consistent framework for EEG foundation models that explicitly coordinates multiple temporal scales during self-supervised pretraining, as illustrated in Fig. 1(b). Specifically, ScaleBrain uses multi-scale tokenization to construct multiple representations of the same raw EEG segment using analysis windows with different temporal spans but shared temporal anchors, thereby establishing correspondence among tokens at different scales. At each scale, a spectral branch captures time-frequency information while a temporal branch captures waveform-level dynamics. The resulting representations are equipped with scale-aware spatio-temporal position encodings and integrated through gated scale fusion into a unified representation. The fused representation is then contextualized by an axial spatio-temporal attention mechanism along the spatial and temporal dimensions. Finally, ScaleBrain introduces scale-specific reconstruction targets so that the scale structure introduced at the input side is explicitly preserved in the selfsupervised objective. In this way, ScaleBrain explicitly models scale diversity from both the input and objective sides while maintaining a common temporal reference across scales.

The contributions are summarized as follows:

• We formulate EEG foundation-model pre-training from the perspective of scale consistency, and propose Scale-Brain, which consistently models diverse temporal scales from both the input representation and self-supervised objective sides.

• We introduce a multi-scale EEG tokenization strategy that analyzes the same raw EEG segment using multiple temporal windows with shared temporal anchors. The resulting scale-specific representations preserve correspondence across temporal resolutions while capturing complementary spectral and temporal information.

• We introduce scale-specific reconstruction to provide objective-side supervision at multiple temporal scales aligned to input representations, and employ gated scale fusion and axial spatio-temporal attention to integrate and contextualize the multi-scale representations.

• We evaluate ScaleBrain on six public benchmark datasets spanning diverse downstream BCI tasks. The results show that ScaleBrain obtains the highest mean Balanced Accuracy on five datasets and the third-highest on the remaining one, while using only 5.0M parameters, validating the effectiveness of the proposed framework.

## II. RELATED WORK

## A. Task-Specific EEG Decoding

Historically, EEG decoding relied heavily on hand-crafted features, such as power spectral density and common spatial patterns, combined with shallow classifiers like support vector machines [8], [9]. With advances in deep learning, Convolutional Neural Networks (CNNs) such as EEGNet [10] and DeepConvNet [20] have become standard backbones for capturing spatio-temporal patterns. More recently, Transformerbased architectures, such as EEGConformer [21], have been introduced to model long-range temporal dependencies. While these task-specific models have achieved competitive performance on particular datasets, they are typically trained from scratch, require substantial labeled data, and often suffer from poor generalization due to high inter-subject variability and cross-session domain shifts.

## B. Self-Supervised EEG Foundation Models

To mitigate the scarcity of labeled EEG data, recent research has shifted toward SSL to build EEG foundation models.

Pre-trained on massive unlabeled EEG corpora, models like BIOT [13], LaBraM [14], CBraMod [16], BraSTORM [22], and REVE [23] utilize masked autoencoding or contrastive objectives to learn generalizable neural representations. Despite their impressive generalization across diverse downstream tasks, these models typically partition EEG signals into a single series of fixed-length patches. Recent multi-scale foundation models have alleviated this limitation by introducing representations at multiple temporal scales. However, how these scales should be temporally coordinated and consistently supervised during pre-training remains less explored. Our proposed ScaleBrain departs from these methodologies by explicitly modeling multi-scale dynamics from both the input and objective sides to form a consistent multi-scale framework.

## C. Multi-Scale EEG Modeling

Multi-scale modeling has been widely explored in EEG analysis. Traditional signal-processing methods often use band-pass filtering to extract frequency-specific features [24], while task-specific deep models employ temporal convolutions with multiple receptive fields [25], multi-resolution spatiotemporal graph learning [26], or multi-granularity patching [27] to capture patterns at different temporal resolutions. Within EEG foundation modeling, CSBrain [15] and Code-Brain [19] explicitly incorporate multi-scale representations, demonstrating the potential of modeling resolution diversity during pre-training. Nevertheless, the use of multiple scales does not by itself ensure consistent correspondence and supervision across the pre-training pipeline. In contrast, ScaleBrain treats scale consistency as a property that should be maintained across both the input and objective levels: different analysis windows are temporally aligned through shared anchors, and each scale is associated with a dedicated reconstruction target. This formulation allows scale information and temporal correspondence to be preserved from tokenization through selfsupervised optimization.

## III. METHODOLOGY

## A. Framework Overview and Notations

Fig. 2 illustrates the overall framework of ScaleBrain, which reformulates EEG foundation-model pre-training as a scaleconsistent masked reconstruction problem, addressing both the need for temporal scale correspondence at the input side and the need for scale-consistent supervision at the objective side. Given an EEG segment, the framework extracts coarse-, medium-, and fine-scale representations using temporally aligned spectral and temporal branches. These scale-specific representations are then masked by a shared span mask, enriched with scale-aware spatio-temporal position encodings, fused into a unified representation, encoded by an axial spatiotemporal Transformer, and finally used to reconstruct masked spectral targets at each scale.

Formally, let $\textbf { X } \in \ \mathbb { R } ^ { C \times T }$ denote a preprocessed EEG segment, where $\mathcal { C } _ { 0 }$ is a reference electrode set with known coordinates following the international 10-10 system, ${ \mathcal { C } } \subseteq { \mathcal { C } } _ { 0 }$ is the observed electrode set, $C = | \mathcal { C } |$ is the number of channels, and $T$ is the number of time points. ScaleBrain defines EEG scales by the temporal span of the analysis window, where $s _ { n }$ denotes the number of sampled timepoints at the n-th scale and corresponds to $s _ { n } / f _ { s }$ seconds under sampling rate $f _ { s } .$ The pre-training is performed over a set of temporal scales $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { N } \}$ . For each scale $s _ { n } ,$ the model produces a scale-specific representation $\mathbf { E } ^ { s _ { n } } \in \mathbb { R } ^ { C \times M \times D }$ , where M is the number of temporally aligned windows and D is the hidden dimension. The representations from all scales are aligned and fused into $\mathbf { E } \in \mathbb { R } ^ { \bar { C } \times M \times D }$ , which is then processed by the axial spatio-temporal encoder.

## B. Multi-Scale EEG Tokenization

To capture the complementary temporal information represented at different scales, we propose a multi-scale EEG tokenization that does not rely on a single series of fixedlength patches but generates temporally aligned tokens across multiple window scales, as presented in Fig. 2(a). Longer windows provide coarser temporal views but finer frequency resolution, making them suitable for characterizing slow rhythmic structures, whereas shorter windows provide finer temporal localization at the cost of coarser frequency resolution, helping preserve transient and rapidly varying patterns. We use a shared hop size h and define the window anchor locations as $\tau _ { i } = i h$ , where $0 \leq i \leq M - 1$ denotes the window index, ensuring that tokens from different scales share a common temporal reference. The windows are centred at $\tau _ { i }$ with zero padding at segment boundaries, so M is identical across scales. This explicit temporal correspondence provides a common basis for cross-scale fusion and representation learning.

Spectral Branch. For each scale $s _ { n }$ , the spectral branch applies Short-Time Fourier Transform (STFT) with window length $s _ { n }$ and a shared hop size h. Given an input EEG segment X, it transforms the c-th channel $\mathbf { X } _ { c }$ into a scalespecific spectrogram $\mathbf { Z } _ { c } ^ { s _ { n } } \in \mathbb { C } ^ { M \times F ^ { s _ { n } } }$

$$
\begin{array} { r } { \mathbf { Z } _ { c } ^ { s _ { n } } ( \tau _ { i } ) = \mathcal { F } \big [ \mathbf { X } _ { c } ( t ) w ^ { s _ { n } } ( t - \tau _ { i } ) \big ] , \quad 1 \leq c \leq C , } \end{array}\tag{1}
$$

where t is time, $w ^ { s _ { n } } ( \cdot )$ is the windowing function for scale $s _ { n } , \mathcal { F [ \cdot ] }$ denotes the Fourier transform, and $F ^ { s _ { n } }$ is the number of frequency bins at this scale. The channel-wise spectrograms are stacked to form $\mathbf { Z } ^ { s _ { n } } \in \mathbb { C } ^ { C \times M \times F ^ { s _ { n } } }$ . We use the log magnitude $\mathbf { R } ^ { s _ { n } } \in \mathbb { R } ^ { C \times M \times F ^ { s _ { n } } }$ computed from the EEG segment as the scale-specific reconstruction target:

$$
\mathbf { R } ^ { s _ { n } } = \log \bigl ( 1 + | \mathbf { Z } ^ { s _ { n } } | \bigr ) .\tag{2}
$$

The spectral feature $\mathbf { H } _ { \mathrm { s p e c } } ^ { s _ { n } } \in \mathbb { R } ^ { C \times M \times D }$ is obtained by projecting this target into the hidden space with a scale-specific projector transforming the feature dimension:

$$
\mathbf { H } _ { \mathrm { s p e c } } ^ { s _ { n } } = \operatorname { P r o j } ^ { s _ { n } } ( \mathbf { R } ^ { s _ { n } } ) .\tag{3}
$$

Temporal Branch. To complement the spectral representation with waveform-level temporal details, we introduce a temporal branch in parallel with the spectral branch. Specifically, we use a three-layer convolutional network, which uses kernel sizes matched to $s _ { n }$ and strides/paddings matched to $h ,$ so that its receptive field and temporal hop are aligned with the spectral tokens at the same scale. The temporal branch produces the temporal feature $\mathbf { H } _ { \mathrm { t e m p } } ^ { s _ { n } } \in \mathbb { R } ^ { C \times M \times \hat { D } }$

![](images/e0a54e49ddd8422d343ab9287ca1499e2b1f29494fa43782047e23aec90d0d9e.jpg)  
Fig. 2. The overall framework of ScaleBrain, which is composed of (a) multi-scale EEG tokenization, (b) axial spatio-temporal encoder, and (c) scale-specific reconstruction. PE: Positional Encoding. FFN: Feed-Forward Network.

Scale-Aware Spatio-Temporal Position Encodings. We propose scale-aware spatio-temporal Position Encodings (PEs) to guide the Transformer in distinguishing scale-specific representations while preserving their common spatial and temporal correspondence. Unlike conventional models that share a universal positional encoding, our scale-aware PEs project spatial coordinates and temporal bases into scale-specific representation spaces, allowing the model to distinguish different temporal granularities across scales. For scale $s _ { n } .$ , the scaleaware spatial positional encoding $\mathbf { E } _ { \mathrm { s p a t } } ^ { s _ { n } } \in \mathbb { R } ^ { C \times D }$ is obtained by a two-layer multi-layer perceptron from the electrode coordinates, while the scale-aware temporal positional encoding $\mathbf { E } _ { \mathrm { t e m p } } ^ { s _ { n } } \in \mathbb { R } ^ { M \times D }$ is built upon a fixed sinusoidal basis, with a projector injecting scale-specific temporal information. Notably, for bipolar channels, the coordinate is defined as the midpoint of the two constituent electrodes in the 10-10 layout, which keeps the spatial PE well-defined while preserving the geometry of the montage.

To enable masked self-supervised pre-training, we apply a scale-aware span masking strategy to the tokens. We randomly generate a shared span mask $\mathcal { M } \subset \{ ( c , m ) \mid 1 \leq c \leq C , 1 \leq$ $m \le M \}$ with a mask ratio r and a span length l. Here, l means the number of consecutive tokens grouped as one masking unit along the temporal dimension, and the units are selected so that approximately r of the tokens are masked. For each scale $s _ { n } .$ , the tokens at the masked positions are replaced with a learnable scale-specific mask token $\mathbf { e } ^ { s _ { n } } \in \mathbb { R } ^ { D }$

$$
\tilde { \mathbf { H } } ^ { s _ { n } } = \mathrm { S p a n M a s k } \bigl ( \mathbf { H } _ { \mathrm { s p e c } } ^ { s _ { n } } + \mathbf { H } _ { \mathrm { t e m p } } ^ { s _ { n } } , \mathbf { e } ^ { s _ { n } } \bigr ) .\tag{4}
$$

The scale-specific token embeddings are then formed by

adding the spatial and temporal PEs with dimension broadcasting, followed by RMSNorm [28]:

$$
\mathbf { E } ^ { s _ { n } } = \mathrm { R M S N o r m } \Bigl ( \tilde { \mathbf { H } } ^ { s _ { n } } + \mathbf { E } _ { \mathrm { s p a t } } ^ { s _ { n } } + \mathbf { E } _ { \mathrm { t e m p } } ^ { s _ { n } } \Bigr ) .\tag{5}
$$

Gated Scale Fusion. We leverage a gated projection to fuse multiple scale embeddings into a unified representation. The gated projection is inspired by SwiGLU [29]:

$$
\mathrm { G a t e d P r o j } ( \mathbf { x } ) = \mathrm { S w i s h } ( \mathbf { x } \mathbf { W } ^ { g } ) \odot \mathbf { x } \mathbf { W } ^ { o } ,\tag{6}
$$

where $\mathbf { W } ^ { o }$ and ${ \bf W } ^ { g }$ are learnable projection matrices, and ⊙ denotes the Hadamard product. Here, $\mathbf { x W } ^ { o }$ produces the projected scale-fusion output, while Swish(xW<sup>g</sup>) acts as a data-dependent gate that modulates this output. Unlike the original SwiGLU used mainly within feed-forward layers, our gated projection is used to adaptively integrate complementary representations from different temporal scales into a unified representation. The fused embedding E is produced by projecting the concatenation of the scale-specific embeddings:

$$
\mathbf { E } = \operatorname { G a t e d P r o j } \left( [ \mathbf { E } ^ { s _ { 1 } } ; \cdot \cdot \cdot ; \mathbf { E } ^ { s _ { N } } ] \right) ,\tag{7}
$$

where [; ] means concatenation along the feature dimension. The fused representation E is then fed into the axial spatiotemporal encoder for contextualized token modeling.

## C. Axial Spatio-Temporal Encoder

The Axial Spatio-Temporal (Axial ST) encoder takes the fused representation E as input and contextualizes it to model spatial and temporal dependencies in EEG. As shown in

Fig. 2(b), the encoder consists of L stacked Axial ST Transformer blocks. Each Axial ST block contains Axial ST attention, a Feed-Forward Network (FFN), residual connections, and RMSNorm [28]. The FFN uses SwiGLU [29] to enhance nonlinear modeling capacity.

Axial ST Attention. The Axial ST attention consists of two parallel attentions: spatial attention and temporal attention, each using the full hidden dimension D with K attention heads. The spatial attention S-Attn(·) performs standard multi-head self-attention MHA(·) along the spatial dimension at each temporal window:

$$
\begin{array} { r l } & { \mathbf { F } _ { m } = \mathrm { M H A } \left( \mathbf { E } _ { m } \mathbf { W } _ { \mathrm { s p a t } } ^ { Q } , \mathbf { E } _ { m } \mathbf { W } _ { \mathrm { s p a t } } ^ { K } , \mathbf { E } _ { m } \mathbf { W } _ { \mathrm { s p a t } } ^ { V } \right) , } \\ & { \mathrm { S } { \mathrm { - } } \mathrm { A t t n } ( \mathbf { E } ) = [ \mathbf { F } _ { 1 } , \cdots , \mathbf { F } _ { M } ] , } \end{array}\tag{8}
$$

where $\mathbf { E } _ { m } = \mathbf { E } _ { : , m , : } \in \mathbb { R } ^ { C \times D } , 1 \leq m \leq M$ denotes the spatial-wise slice at the m-th temporal window, $\mathbf { W } _ { \mathrm { s p a t } } ^ { Q } , \mathbf { W } _ { \mathrm { s p a t } } ^ { K } ,$ $\mathbf { W } _ { \mathrm { s p a t } } ^ { V }$ are projection matrices of the query, key, and value in the spatial attention, and [, ] denotes stacking along the temporal dimension. Analogously, the temporal branch applies multi-head self-attention across temporal windows for each channel, using $\mathbf { E } _ { c } = \mathbf { E } _ { c , : , : } \in \mathbb { R } ^ { M \times D } , 1 \leq c \leq C$ . The Axial ST attention output is obtained by summing the two branch outputs. Thus, the axial design further integrates the fused information across spatial and temporal dimensions while retaining the interactions among multiple temporal scales.

## D. Scale-Specific Reconstruction

We pre-train ScaleBrain with scale-specific reconstruction targets illustrated in Fig. 2(c), building upon the standard masked autoencoding framework. Consistent with the multi-scale EEG tokenization, we use the scale-specific logmagnitude spectra $\mathbf { R } ^ { s _ { n } }$ extracted by the spectral branches as reconstruction targets. Reconstructing these targets in parallel encourages the model to preserve information associated with each temporal scale throughout self-supervised learning.

For each scale $s _ { n } .$ , we define the reconstruction loss $\mathcal L ^ { s _ { n } }$ using Mean Squared Error (MSE):

$$
\mathcal { L } ^ { s _ { n } } = \frac { 1 } { F ^ { s _ { n } } | \mathcal { M } | } \sum _ { ( c , m ) \in \mathcal { M } } \left\| \hat { \mathbf { R } } _ { c , m , : } ^ { s _ { n } } - \mathbf { R } _ { c , m , : } ^ { s _ { n } } \right\| _ { 2 } ^ { 2 } ,\tag{9}
$$

where $\hat { \mathbf { R } } ^ { s _ { n } }$ is the reconstructed log magnitudes for scale $s _ { n } ,$ and M is the span mask shared across scales. The overall pretraining objective $\mathcal { L }$ is a weighted sum of the scale-specific losses:

$$
\mathcal { L } = \sum _ { n = 1 } ^ { N } \lambda _ { n } \mathcal { L } ^ { s _ { n } } ,\tag{10}
$$

where $\lambda _ { n }$ is the weight for scale $s _ { n }$ . The purpose of the scalespecific objective is to balance supervision across temporal resolutions rather than to explicitly separate frequency bands. Under equal weighting, the optimization may favor the coarse reconstruction scale which has simpler signal structure. We therefore assign progressively larger weights to finer temporal scales, reducing optimization bias toward the coarse-scale reconstruction target and encouraging the encoder to retain localized information.

## IV. EXPERIMENTS

## A. Pre-Training and Downstream Datasets

Pre-Training Dataset Preprocessing. We pre-train Scale-Brain on the Temple University Hospital EEG corpus (TUEG) [30], which is commonly used in EEG foundation model studies [15], [16]. Following prior preprocessing protocols [14], [16], all signals are filtered by a 0.3-75 Hz bandpass filter and a 60 Hz notch filter, resampled to 200 Hz, and segmented into non-overlapping 30-second EEG samples. This preprocessing yields a total of 1,175,693 samples, corresponding to approximately 9,797 hours of EEG data. For consistency with prior foundation-model benchmarks [15], [16], [19], our main comparison follows the commonly used full-TUEG pre training protocol, in which recordings belonging to TUAB and TUEV datasets are retained in the pre-training corpus.

Downstream BCI Tasks and Datasets. We conduct evaluations across 6 public EEG datasets spanning various representative BCI tasks, as summarized in Table I. These tasks include Motor Imagery Classification (PhysioNet-MI [31]), Emotion Recognition (SEED-V [32]), Sleep Staging (HMC [33]), Mental Stress Detection (MentalArithmetic [31], [34]), Abnormality Detection (TUAB [30]), and Event Type Classification (TUEV [30]). We detail the preprocessing and data split of each dataset in the following.

• PhysioNet-MI [31] is a widely used Motor Imagery (MI) dataset collected from 109 subjects, with four MI tasks: left fist, right fist, both fists, and both feet. The recordings were collected at 160 Hz with 64 electrodes. Following prior work [15], the signals are segmented into 4-second samples and resampled to 200 Hz. We follow the subjectindependent split in prior work [15], where subjects 1-70 are used for training, 71-89 for validation, and 90-109 for testing.

• SEED-V [32] is an emotion recognition dataset comprising 16 subjects, covering five categories of emotion (happy, sad, neutral, disgust, fear). The recordings were collected at 1000 Hz with 62 electrodes. Following prior work [15], we segment the signals into 1-second samples and resample to 200 Hz, and divide the 15 trials of each subject into three equal parts for training, validation, and testing.

• HMC [33] is a large-scale sleep dataset containing 151 subjects, annotated with five standard sleep stages (Wake, N1, N2, N3, REM). The recordings were collected at 256 Hz with 4 electrodes. Following prior work [15], the signals are segmented into 30-second samples and resampled to 200 Hz. The data is split in a subjectindependent manner, where subjects 1-100 are used for training, 101-125 for validation, and 126-151 for testing.

• MentalArithmetic [31], [34] is a small-scale dataset intended to identify mental stress, where recordings collected before the mental arithmetic task are labeled as “without mental stress”, and recordings collected during the task are labeled as “with mental stress”. The recordings were collected at 500 Hz with 20 electrodes. Following prior work [15], the signals are segmented into 5-second samples and resampled to 200 Hz. We adopt a subject-independent split as in prior work [15], where subjects 1-28 are used for training, 29-32 for validation, and 33-36 for testing.

TABLE I  
OVERVIEW OF DOWNSTREAM BCI TASKS AND DATASETS.
<table><tr><td>Task</td><td>Dataset</td><td>Rate (Hz)</td><td>Duration (s)</td><td>#Channels</td><td>#Samples</td><td>#Subjects</td><td>Label</td></tr><tr><td>Motor Imagery</td><td>PhysioNet-MI</td><td>160</td><td>4</td><td>64</td><td>9837</td><td>109</td><td>4-class</td></tr><tr><td>Emotion Recognition</td><td>SEED-V</td><td>1000</td><td>1</td><td>62</td><td>117744</td><td>16</td><td>5-class</td></tr><tr><td>Sleep Staging</td><td>HMC</td><td>256</td><td>30</td><td>4</td><td>137243</td><td>151</td><td>5-class</td></tr><tr><td>Mental Stress Detection</td><td>MentalArithmetic</td><td>500</td><td>5</td><td>20</td><td>1707</td><td>36</td><td>2-class</td></tr><tr><td>Abnormality Detection</td><td>TUAB</td><td>250</td><td>10</td><td>16</td><td>409455</td><td>2383</td><td>2-class</td></tr><tr><td>Event Type Classification</td><td>TUEV</td><td>250</td><td>5</td><td>16</td><td>112491</td><td>370</td><td>6-class</td></tr></table>

• TUAB [30] is a subset of the TUEG corpus that focuses on the abnormality detection (or “abnormal detection” in some literature) task, with annotations of “normal” or “abnormal”. The recordings were collected at 250 Hz with 23 electrodes. The signals are segmented into 10- second samples, resampled to 200 Hz, and converted to the commonly adopted 16-channel bipolar montage as in prior works [15], [16]. The training and testing subjects are specified by the original dataset. Following prior work [15], we split the training subjects into training and validation sets at a 8:2 ratio.

• TUEV [30] is a subset of the TUEG corpus that focuses on event type classification, containing six types of events (SPSW, GPED, PLED, EYEM, ARTF, BCKG). The recordings were collected at 250 Hz with 23 electrodes. The signals are segmented into 5-second samples, resampled to 200 Hz, and converted to the commonly adopted 16-channel bipolar montage as in prior works [15], [16]. The training and testing subjects are specified by the original dataset. Following prior work [15], we split the training subjects into training and validation sets at a 8:2 ratio.

## B. Implementation Details

Pre-Training Settings. To instantiate the coarse-to-fine temporal scales defined in our tokenization module, we use N = 3 scales, with $s _ { 1 } = 2 0 0 , s _ { 2 } = 1 0 0$ , and $s _ { 3 } = 4 0$ . The shared hop size is set to h = 20. We use a masking ratio of r = 50% and a span length of l = 10 tokens. The objective weights are fixed as $\lambda _ { 1 } = 0 . 5 , \lambda _ { 2 } = 1 . 0$ , and $\lambda _ { 3 } = 2 . 0$ , matching their design purpose. The encoder contains L = 8 Axial ST blocks with hidden dimension $D = 1 9 2$ . We pre-train ScaleBrain for 5 epochs with a batch size of 128. Optimization is performed using AdamW with a learning rate of $2 \times 1 0 ^ { - 4 }$ , weight decay of $1 \times 1 0 ^ { - 2 }$ , and a cosine annealing learning-rate schedule with warmup. The warmup is a linear incremental process from zero to the target learning rate in the 10% of total steps, then the cosine annealing takes over. The detailed pre-training hyperparameters are listed in Table II, which are fixed before evaluation and are not tuned for any individual downstream datasets. Pre-training is conducted on a Linux server equipped with four NVIDIA RTX 3090 GPUs, each with 24 GB VRAM, using Python 3.12 and PyTorch 2.7.1 with CUDA 12.8.

TABLE II  
PRE-TRAINING HYPERPARAMETERS FOR SCALEBRAIN.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Input Channels Time Points per Sample Mask Ratio Mask Token</td><td>19 6000 0.5 Learnable vector</td></tr><tr><td>Number of Layers Hidden Dimension Attention Heads Feed-forward Dimension Dropout Weight Initialization</td><td>8 192 6 512 0.1 Kaiming Normal</td></tr><tr><td>Optimizer Learning Rate Adamβ</td><td>AdamW 2e-4 (0.9, 0.999)</td></tr><tr><td>Adam € Weight Decay Gradient Clipping Training Epochs Batch Size Learning Rate Scheduler CosineAnnealingLR</td><td>1e-8 1e-2 1.0 5 128</td></tr></table>

Fine-Tuning Settings. To fine-tune ScaleBrain on a downstream dataset, we load the pre-trained weights and replace the reconstruction heads with a classification head composed of temporal or spatial pooling and a two-layer perceptron. Specifically, we apply spatial pooling to HMC and TUEV, and temporal pooling to other datasets. Because the model does not use channel-specific parameters tied to a fixed electrode set and instead encodes channel geometry through electrode coordinates, the pre-trained encoder can be applied to downstream datasets with different numbers of channels. We then fine-tune the whole model on the target dataset with default hyperparameters listed in Table III. We use binary cross-entropy loss for binary classification and cross-entropy for multi-class classification.

## C. Comparison Baselines

As listed in Table IV, we compare our model with two major types of baselines: (i) task-specific EEG decoding models and (ii) foundation models for EEG decoding. The baselines are introduced briefly in the following.

Task-specific EEG decoding models include:

TABLE III  
DEFAULT HYPERPARAMETER SETTINGS FOR DOWNSTREAM FINE-TUNING.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Training Epochs</td><td>10</td></tr><tr><td>Batch Size</td><td>128</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-4</td></tr><tr><td>Adamβ Adam €</td><td>(0.9, 0.999)</td></tr><tr><td>Weight Decay</td><td>1e-8</td></tr><tr><td>Dropout</td><td>1e-2</td></tr><tr><td>Learning Rate Scheduler</td><td>0.1 CosineAnnealingLR</td></tr><tr><td>Warmup Epochs</td><td>1</td></tr><tr><td>Minimum Learning Rate</td><td></td></tr><tr><td>Label Smoothing (for multi-class)</td><td>1e-6 0.1</td></tr></table>

• EEGNet [10] is a compact, lightweight Convolutional Neural Network (CNN) specifically engineered for EEG, utilizing depthwise and separable convolutions to extract robust temporal and spatial features.

• EEGConformer [21] integrates CNNs with self-attention mechanisms to seamlessly capture local spatio-temporal patterns alongside global, long-range temporal dependencies in EEG signals.

• SPaRCNet [35] is a deep convolutional neural network with dense residual connections, designed for expert-level classification of seizures and periodic/rhythmic patterns in clinical EEG monitoring.

• ContraWR [36] is a self-supervised representation learning framework that converts EEG signals into spectrograms and employs contrastive learning on top of a 2D ResNet backbone to learn robust feature embeddings.

• FFCL [37] is a hybrid deep learning model that effectively captures comprehensive EEG dynamics by fusing spatial representations from a CNN branch with temporal features extracted by a parallel LSTM encoder.

• ST-Transformer [12] is a Transformer-based model where the attention mechanisms are applied to both spatial and temporal dimensions to capture the structural dependencies in EEG signals.

Foundation models for EEG decoding include:

• BIOT [13] is a flexible, transformer-based framework that utilizes spectral energy vectors extracted from raw signals to facilitate scalable supervised or self-supervised pretraining across diverse setups.

• LaBraM-Base [14] is a pioneering EEG foundation model that tokenizes continuous raw EEG patches using a vector-quantized neural spectrum tokenizer and performs self-supervised pre-training via masked neural code prediction.

• CBraMod [16] is a criss-cross EEG foundation model that decomposes spatial and temporal dependencies into parallel attention mechanisms to address the spatiotemporal structure of EEG signals.

• CSBrain [15] is a cross-scale spatio-temporal EEG foundation model that employs cross-scale tokenization and structured sparse attention to capture multi-resolution

neural dynamics.

• REVE-Base [23] is an adaptable EEG foundation model that utilizes a 4D positional encoding scheme and masked autoencoding to generalize robustly across arbitrary electrode layouts and recording protocols.

• CodeBrain [19] is an EEG foundation model combining a decoupled tokenizer and a multi-scale attention mechanism to model both inter- and intra-patch dependencies.

## D. Evaluation Metrics

Consistent with prior works [15], [16], we report Balanced Accuracy, AUC-PR, and AUROC for binary classification tasks; Balanced Accuracy, Cohen’s Kappa, and Weighted F1- score for multiclass classification tasks. Balanced Accuracy is the average recall across classes, providing a class-balanced measure of classification performance. Cohen’s Kappa measures agreement between predictions and ground-truth labels beyond chance, while Weighted F1-score averages per-class F1-scores weighted by class support. AUC-PR and AUROC measure the areas under the precision-recall and receiver operating characteristic curves, respectively, with AUC-PR being particularly informative for imbalanced datasets and AUROC assessing the model’s ability to distinguish positive from negative samples. We use Balanced Accuracy as the monitor score to select the best model for evaluation.

## E. Performance Comparison

We compare ScaleBrain with 12 representative baselines, including 6 task-specific models and 6 EEG foundation models, across 6 BCI classification datasets. The results are presented in Table IV. For ScaleBrain, we report the mean and standard deviation over five random seeds, while baseline results are taken from the corresponding works when available.

Overall, EEG foundation models provide stronger baselines than task-specific models on most datasets, suggesting the benefit of large-scale EEG pre-training for transferable representation learning. Among foundation models, ScaleBrain achieves the best Balanced Accuracy on five of the six datasets and ranks third on SEED-V behind CSBrain and CodeBrain. The largest gain appears on TUEV, where ScaleBrain reaches 0.7210 and improves over CSBrain by 3.07 percentage points. It also obtains strong results on MentalArithmetic and HMC, with balanced accuracy of 0.7771 and 0.7540, respectively. The improvements over existing multi-scale foundation models further suggest that coordinating multi-scale representations across the pre-training pipeline can provide benefits beyond simply increasing scale diversity. As TUAB and TUEV are subsets of the TUEG corpus, we additionally exclude recordings associated with TUAB and TUEV from TUEG pretraining to examine the effect of potential pre-training exposure. The resulting “ScaleBrain\*” achieves comparable or slightly better performance than the full-TUEG setting, indicating that the observed performance is not directly attributable to exposure of downstream recordings.

ScaleBrain also shows favorable parameter efficiency. With 5.0M parameters, it outperforms REVE-Base across all six datasets, despite REVE-Base using 69M parameters. These results indicate that scale-consistent multi-scale pre-training can improve EEG representation learning without relying on a substantially larger model size.

TABLE IV  
COMPARISON RESULTS OF DIFFERENT METHODS ACROSS SIX EEG DOWNSTREAM TASKS. THE BEST, SECOND-BEST, AND third-best RESULTS ARE MARKED. \* MEANS RECORDINGS FROM TUAB AND TUEV ARE EXCLUDED IN PRE-TRAINING. ACC.: ACCURACY.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Params</td><td colspan="3">PhysioNet-MI (4-class)</td><td colspan="2">SEED-V (5-class)</td><td rowspan="2"></td><td rowspan="2">Weighted F1</td></tr><tr><td>Balanced Acc.</td><td>Cohen&#x27;s Kappa</td><td>Weighted F1</td><td>Balanced Acc.</td><td>Cohen&#x27;s Kappa</td></tr><tr><td>EEGNet</td><td>0.003M</td><td>0.5814 ± 0.0125</td><td>0.4468 ± 0.0199</td><td>0.5796 ± 0.0115</td><td>0.2961 ± 0.0102</td><td></td><td>0.1006 ± 0.0143</td><td>0.2749 ± 0.0098</td><td></td></tr><tr><td>EEGConformer</td><td>0.55M</td><td>0.6049 ± 0.0104</td><td>0.4736 ± 0.0171</td><td>0.6062 ± 0.0095</td><td></td><td>0.3537 ± 0.0112</td><td>0.1772 ± 0.0174</td><td>0.3487 ± 0.0136</td><td></td></tr><tr><td>SPaRCNet</td><td>0.76M</td><td>0.5932 ± 0.0152</td><td>0.4564 ± 0.0234</td><td>0.5937 ± 0.0147</td><td></td><td>0.2949 ± 0.0078</td><td>0.1121 ± 0.0139</td><td>0.2979 ± 0.0083</td><td></td></tr><tr><td>ContraWR</td><td>1.6M</td><td>0.5892 ± 0.0133</td><td>0.4527 ± 0.0248</td><td>0.5918 ± 0.0116</td><td></td><td>0.3546 ± 0.0105</td><td>0.1905 ± 0.0188</td><td>0.3544 ± 0.0121</td><td></td></tr><tr><td>FFCL</td><td>2.4M</td><td>0.5726 ± 0.0092</td><td>0.4323 ± 0.0182</td><td>0.5701</td><td>± 0.0079</td><td>0.3641 ± 0.0092</td><td>0.2078 ± 0.0201</td><td>0.3645 ± 0.0132</td><td></td></tr><tr><td>ST-Transformer</td><td>3.5M</td><td>0.6035 ± 0.0081</td><td>0.4712 2 ± 0.0199</td><td>0.6053</td><td>3 ± 0.0075</td><td>0.3052 ± 0.0072</td><td>0.1083 ± 0.0121</td><td>0.2833 ± 0.0105</td><td></td></tr><tr><td>BIOT</td><td>3.2M</td><td>0.6153 ± 0.0154</td><td>0.4875 ± 0.0272</td><td>0.6158</td><td> ± 0.0197</td><td>0.3837 ± 0.0187</td><td></td><td></td><td></td></tr><tr><td>LaBraM-Base</td><td>5.8M</td><td>0.6173 ± 0.0122</td><td>0.4912 2 ± 0.0192</td><td>0.6177 ± 0.0141</td><td></td><td>0.3976 ± 0.0138</td><td>0.2261 ± 0.0262</td><td>0.3974 ± 0.0111</td><td>0.3856 ± 0.0203</td></tr><tr><td>CBraMod</td><td>4.0M</td><td>0.6174 ± 0.0036</td><td>0.4898 ± 0.0048</td><td>0.6179 ± 0.0035</td><td></td><td>0.4091 ± 0.0097</td><td>0.2386 ± 0.0209</td><td>0.4101 ± 0.0108</td><td></td></tr><tr><td>CSBrain</td><td>4.9M</td><td>0.6304 ± 0.0090</td><td>0.5071 ± 0.0120</td><td>0.6308 ± 0.0095</td><td></td><td>0.4197 ± 0.0033</td><td>0.2569 ± 0.0143 0.2785 ± 0.0034 0.4280 ± 0.0023</td><td></td><td></td></tr><tr><td>REVE-Base</td><td>69M</td><td>0.6480 ± 0.0140</td><td>0.5306 ± 0.0187</td><td>0.6484 ± 0.0170</td><td></td><td>0.4048 ± 0.0023</td><td>0.2622 ± 0.0021</td><td>0.4145 ± 0.0017</td><td></td></tr><tr><td>CodeBrain</td><td>15M</td><td></td><td></td><td></td><td></td><td>0.4137 ± 0.0023</td><td>0.2735 ± 0.0032</td><td>0.4235 ± 0.0022</td><td></td></tr><tr><td>ScaleBrain</td><td>5.0M</td><td></td><td>0.6524 ± 0.0111 0.5365 ± 0.0148 0.6514 ± 0.0105</td><td></td><td></td><td>0.4116 ± 0.0056</td><td>0.2666 ± 0.0084</td><td></td><td>0.4181 ± 0.0076</td></tr></table>

<table><tr><td rowspan="2">Method</td><td rowspan="2">Params</td><td colspan="3">HMC (5-class)</td><td rowspan="2">MentalArithmetic (2-class)</td><td colspan="4"></td></tr><tr><td>Balanced Acc.</td><td>Cohen&#x27;s Kappa</td><td>Weighted F1</td><td>Balanced Acc.</td><td></td><td>AUC-PR</td><td>AUROC</td></tr><tr><td>EEGNet</td><td>0.003M</td><td>0.6534 ± 0.0122</td><td>0.5886 ± 0.0201</td><td></td><td>0.6536 ± 0.0168</td><td>0.6770 ± 0.0116</td><td>0.5763</td><td>± 0.0102</td><td>0.7321 ± 0.0108</td></tr><tr><td>EEGConformer</td><td>0.55M</td><td>0.7149 ± 0.0086</td><td>0.6432 ± 0.0055</td><td></td><td>0.7080 ± 0.0039</td><td>0.6805 ± 0.0123</td><td>0.5829 ± 0.0134</td><td></td><td>0.7424 ± 0.0128</td></tr><tr><td>SPaRCNet</td><td>0.76M</td><td>0.4756 ± 0.1109</td><td>0.3147</td><td>± 0.1315</td><td>0.4108 ± 0.1310</td><td>0.6879 ± 0.0107</td><td>0.5825 ± 0.0193</td><td></td><td>0.7418 ± 0.0132</td></tr><tr><td>ContraWR</td><td>1.6M</td><td>0.4242 ± 0.0541</td><td>0.2340</td><td>± 0.0554</td><td>0.2987 ± 0.0288</td><td>0.6631 ± 0.0097</td><td>0.5787 7 ± 0.0164</td><td></td><td>0.7332 ± 0.0082</td></tr><tr><td>FFCL</td><td>2.4M</td><td>0.4427 ± 0.0702</td><td>0.2542 ± 0.0654</td><td></td><td>0.2902 ± 0.0485</td><td>0.6798 ± 0.0142</td><td>0.5786 ± 0.0266</td><td></td><td>0.7330 ± 0.0198</td></tr><tr><td>ST-Transformer</td><td>3.5M</td><td>0.2559 ± 0.0141</td><td>0.0503 ±</td><td>0.0183 0.1428</td><td> ± 0.0122</td><td>0.6631 ± 0.0173</td><td>0.5672 ± 0.0259</td><td></td><td>0.7132 ± 0.0174</td></tr><tr><td>BIOT</td><td>3.2M</td><td>0.6862 ± 0.0041</td><td>0.6295</td><td>± 0.0113 0.7091</td><td>± 0.0147</td><td>0.6875 ± 0.0186</td><td>0.6004 ± 0.0195</td><td></td><td>0.7536 ± 0.0144</td></tr><tr><td>LaBraM-Base</td><td>5.8M</td><td>0.7277 ± 0.0101</td><td>0.6813</td><td>± 0.0053</td><td>0.7454 ± 0.0027</td><td>0.6909 ± 0.0125</td><td>0.5999 ± 0.0155</td><td></td><td>0.7721 ± 0.0093</td></tr><tr><td>CBraMod</td><td>4.0M</td><td>0.7269 ± 0.0041</td><td>0.6685 ±</td><td>0.0104</td><td>0.7395 ± 0.0089</td><td>0.7256 ± 0.0132</td><td>0.6267 7 ± 0.0099</td><td>0.7905 ± 0.0073</td><td></td></tr><tr><td>CSBrain</td><td>4.9M</td><td>0.7345 ± 0.0047</td><td>0.6818 ± 0.0046</td><td></td><td>0.7506 ± 0.0042</td><td>0.7558 ± 0.0106</td><td>0.6696 ± 0.0221</td><td>0.8478 ± 0.0297</td><td></td></tr><tr><td>REVE-Base</td><td>69M</td><td>0.7401 ± 0.0075</td><td>0.6982 ± 0.0078</td><td></td><td>0.7638 ± 0.0074</td><td>0.7660 ± 0.0355</td><td>0.7470 ± 0.0807</td><td></td><td>0.8450 ± 0.0514</td></tr><tr><td>CodeBrain</td><td>15M</td><td></td><td></td><td></td><td></td><td>0.7514 ± 0.0203</td><td>0.7177 ± 0.0421</td><td></td><td>0.8707 ± 0.0209</td></tr><tr><td>ScaleBrain</td><td>5.0M</td><td>0.7540 ± 0.0098</td><td>0.6961 ± 0.0145</td><td></td><td>0.7646 ± 0.0122</td><td>0.7771 ± 0.0429</td><td>0.8062 ± 0.0809</td><td></td><td>0.8872 ± 0.0437</td></tr></table>

<table><tr><td rowspan="2">Method</td><td rowspan="2">Params</td><td colspan="3">TUAB (2-class)</td><td colspan="2">TUEV (6-class)</td><td colspan="2"></td></tr><tr><td>Balanced Acc.</td><td>AUC-PR</td><td>AUROC</td><td>Balanced Acc.</td><td></td><td>Cohen&#x27;s Kappa</td><td>Weighted F1</td></tr><tr><td>EEGNet</td><td>0.003M</td><td>0.7642 ± 0.0036</td><td>0.8299 ± 0.0043</td><td>0.8412 ± 0.0031</td><td>0.3876 ± 0.0143</td><td>0.3577 ± 0.0155</td><td></td><td>0.6539 ± 0.0120</td></tr><tr><td>EEGConformer</td><td>0.55M</td><td>0.7758 ± 0.0049</td><td>0.8427 ± 0.0054</td><td>0.8445 ± 0.0038</td><td>0.4074 ± 0.0164</td><td>0.3967 ± 0.0195</td><td></td><td>0.6983 ± 0.0152</td></tr><tr><td>SPaRCNet</td><td>0.76M</td><td>0.7896 ± 0.0018</td><td>0.8414 ± 0.0018</td><td>0.8676 ± 0.0012</td><td>0.4161 ± 0.0262</td><td>0.4233</td><td>± 0.0181</td><td>0.7024 ± 0.0104</td></tr><tr><td>ContraWR</td><td>1.6M</td><td>0.7746 ± 0.0041</td><td>0.8421 ± 0.0104</td><td>0.8456 ± 0.0074</td><td>0.4384 ± 0.0349</td><td>0.3912 土</td><td>0.0237</td><td>0.6893 ± 0.0136</td></tr><tr><td>FFCL</td><td>2.4M</td><td>0.7848 ± 0.0038</td><td>0.8448 ± 0.0065</td><td>0.8569 ± 0.0051</td><td>0.3979 ± 0.0104</td><td>0.3732</td><td>± 0.0188</td><td>0.6783 ± 0.0120</td></tr><tr><td>ST-Transformer</td><td>3.5M</td><td>0.7966 ± 0.0023</td><td>0.8521 ± 0.0026</td><td>0.8707 ± 0.0019</td><td>0.3984 ±</td><td>0.0228 0.3765</td><td>± 0.0306 0.6823</td><td>± 0.0190</td></tr><tr><td>BIOT</td><td>3.2M</td><td>0.7959 ± 0.0057</td><td>0.8792 ± 0.0023</td><td>0.8815 ± 0.0043</td><td>0.5281 ± 0.0225</td><td>0.5273 ± 0.0249</td><td></td><td>0.7492 ± 0.0082</td></tr><tr><td>LaBraM-Base</td><td>5.8M</td><td>0.8140 ± 0.0019</td><td>0.8965 ± 0.0016</td><td>0.9022 ± 0.0009</td><td>0.6409 ± 0.0065</td><td>0.6637</td><td>± 0.0093</td><td>0.8312 ± 0.0052</td></tr><tr><td>CBraMod</td><td>4.0M</td><td>0.7891 ± 0.0030</td><td>0.8636 ± 0.0063</td><td>0.8606 ± 0.0057</td><td>0.6671 ± 0.0107</td><td>0.6772 ± 0.0096</td><td></td><td>0.8342 ± 0.0064</td></tr><tr><td>CSBrain</td><td>4.9M</td><td>0.8172 ± 0.0043</td><td>0.9005 ± 0.0066</td><td>0.8957 ± 0.0046</td><td>0.6903 ± 0.0059</td><td>0.6833 ± 0.0047</td><td></td><td>0.8333 ± 0.0057</td></tr><tr><td>REVE-Base</td><td>69M</td><td>0.8315 ± 0.0014</td><td>0.9281 ± 0.0009</td><td>0.9245 ± 0.0013</td><td>0.6759 ± 0.0229</td><td>0.6783 ± 0.0199</td><td></td><td>0.8451 ± 0.0129</td></tr><tr><td>CodeBrain</td><td>15M</td><td>0.8294 ± 0.0013</td><td>0.9100 ± 0.0006</td><td>0.9030 ± 0.0009</td><td>0.6428 ± 0.0062</td><td>0.6912 ± 0.0101</td><td></td><td>0.8362 ± 0.0048</td></tr><tr><td>ScaleBrain</td><td>5.0M</td><td>0.8324 ± 0.0007</td><td>0.9030 ± 0.0018</td><td>0.8899 ± 0.0022</td><td>0.7210 ± 0.0259</td><td>0.7187 ± 0.0434</td><td></td><td>0.8514 ± 0.0229</td></tr><tr><td>ScaleBrain*</td><td>5.0M</td><td>0.8327 ± 0.0033</td><td>0.9054 ± 0.0027</td><td>0.8942 ± 0.0015</td><td>0.7236 ± 0.0283</td><td>0.7222 ± 0.0153</td><td></td><td>0.8539 ± 0.0078</td></tr></table>

## F. Ablation Study

To evaluate the main components of ScaleBrain and examine the role of scale consistency during pre-training, we conduct ablation experiments in Table V.

Effectiveness of Pre-Training. To assess self-supervised pretraining, we compare the full model with a variant trained from scratch (“w/o Pre-training”). This variant keeps the same architecture but is trained directly on each downstream dataset with a classification head. Removing pre-training consistently degrades performance across all 6 datasets, with the largest drop on MentalArithmetic, confirming the importance of large-scale self-supervised pre-training for EEG representation learning.

Effectiveness of Scale-Aware PEs. To assess scale-aware PEs, we compare the full model with a variant using shared PEs across scales (“w/o Scale-Aware PEs”), where the same spatial and temporal PEs are applied after the gated scale fusion. Scale-aware PEs improve performance on 5 of the 6 datasets. The only exception is TUEV, where shared PEs achieve 0.7312 compared with 0.7210 for the full model, suggesting that explicit scale-aware positional alignment brings generally useful but task-dependent benefits. Nevertheless, the scale-aware formulation provides a generally effective way to preserve scale identity while maintaining cross-scale temporal correspondence.

![](images/648311308b52fba9c17c0932690fc7c4189b865181e187d30fa7e7dfb105e193.jpg)

![](images/a87c47f4dc2251a148c235b79a806781b6ed6e1a46c939332edb0e963f813d73.jpg)

![](images/fe4929d03acb3f9d15372c61ed016bbd34feacd8bff0e41e594a3d93b7c015f9.jpg)

![](images/33c3bf9d6f98f76b165eb87f1ed922fcecda0aa1fe77299c204d389e89c12d1d.jpg)

![](images/872aa4f0e9b7846c529077aa35e8ae1de81fd551a6ec9d01f458d2d33693de43.jpg)

![](images/d11bfe36765a6511a338f6f6598cc9c4d2bc9fc8662076fa57817e3da1c594de.jpg)

![](images/211981138a93dcd2ab9f04ded4892ff8e515dd95ccfa43b26dd1ddf4e5892230.jpg)

![](images/b1896809ec9f5351f20021a28cb1eed502a6a614e5fcf62e1cd2e2c98c6a7917.jpg)

![](images/d61a435d413c070fc85652ac036688dd4fc8075f7c1709b96b324eb5f7277e92.jpg)

![](images/8e81303221e7d86c701f0d5e5a65a140e070e72237ce28f6e7fa850cb78333cd.jpg)

![](images/334aa5ba4169d68aac4c9efc05e4ec2243711aa52bcc8f9762e801fad6f8bd2d.jpg)

![](images/a41f8d8503c9a5671fede244bef85ad20c91060eaaf4f1b1978da6e1d732fd29.jpg)

![](images/e0bd6d5c5bfa8e99ffc3defc4c85898d63790952dbba02df66f6c3e8d694104d.jpg)

Fig. 3. Analysis of the necessity of scales. Acc.: Accuracy.  
Fig. 4. Analysis of the effect of additional scales. Acc.: Accuracy.  
![](images/9718c9bb87bc3203d520773aa68c6b95d5a8b2a876c857cd711bc026dd090141.jpg)

![](images/6555e32c8f9638c490c97fe505697517d15031f836e4a609312eef7d7dd36306.jpg)

![](images/3c8775e841ab80b554ccbbf1788bec954fab95d865e24d53fd671ba528cae397.jpg)

![](images/c1230f7ec71a819bc2c39070f45dd4161a5b32b1a775f7037f8bd30af9d6b9f4.jpg)

![](images/65242e212c780268bc2e71af836cf362345e2873feaedaa2f702a83c72562c60.jpg)

![](images/aaf0d9b87e606d83961999ada9bcdaf8bad101379eb8a19821d03706952b0b43.jpg)

![](images/75221854c5b6e6ab097e0820f9ccbf94a3a29556354c928facadb993ab5490cf.jpg)

![](images/e7c694f4cd8a394cc7a59c7661d6f6d9faec9a5608875a2c4317ee72139bcae5.jpg)

![](images/9e50881c4d16782c4eda29ea627e3fccdb9acb83a2c415818fe96aacd85b35f9.jpg)

![](images/2ab06d4f196a65ca0b0b496552fa8f52bce5373b3d8389d9bc49f3aa0290404e.jpg)

![](images/bfca1ec75a77b4a1306b0e65d6f41f69435619447bf7c8ab3f94661b9accac6b.jpg)

![](images/64dac5c8f948ec088bbbe3e7439428db8811a596cc395820b763fb3f62652b68.jpg)  
Fig. 5. Analysis of the objective weights. Acc.: Accuracy.

reconstruction objective; “w/o MS Tokenization” uses singlescale tokenization with scale-specific reconstruction; “w/o Scale-Specific Recon” uses multi-scale tokenization with a single-scale reconstruction objective; and the full model uses both multi-scale tokenization and scale-specific reconstruction. This design allows us to examine the effects of input-side and objective-side multi-scale modeling as well as their interaction.

Effect of Input- and Objective-Side Multi-Scale Modeling. The four scale-related configurations in Table V form a 2 × 2 factorial design that separates input-side multi-scale representation from objective-side scale-specific supervision, allowing us to examine their interaction. Specifically, “w/o Multi-Scale” uses a single-scale input representation and a single-scale

Across the six datasets, the two factors exhibit complementary but task-dependent effects. Notably, the effect of scalespecific reconstruction depends strongly on the presence of multi-scale tokenization. For example, on MentalArithmetic, scale-specific reconstruction decreases balanced accuracy from 0.7292 to 0.6764 when the input remains single-scale, but increases it from 0.6882 to 0.7771 when multi-scale tokenization is used. This interaction suggests that objective-side scalespecific supervision is most effective when coupled with multiscale input representations. Overall, the factorial comparison supports the central design of ScaleBrain: multi-scale input representation and scale-specific supervision should be jointly coordinated rather than introduced independently.

![](images/c9f54e0890b02b8066a7eb812f7739d1fd2e71c29dacf39b30f4ab43b7b9a2e4.jpg)

![](images/a0d6ffab99edd98f86cbb2dc5a764bda479168293642d39b3f9fe7c2769948fc.jpg)

![](images/44dd34209af8900304bfcd231c75dcbbc4c66bc02d3e55d21490d0b3a601ea4b.jpg)

![](images/487845de6e2ef784a19b9d3a417d0b686e29555afc032b0a13c8e06c2cb877a9.jpg)

![](images/1a257ea805ec9d77ab839553075c9dae99877b4466131b0f073ec1cb92762524.jpg)

![](images/7bfb8f221a525c149fcac248ac6981abcbf55db81b22a3fdbd1383d52f151030.jpg)  
Fig. 6. Analysis of the attention mechanism. Acc.: Accuracy.

![](images/bae7c59b5b5d93c71b3fc74a5779a59e8ee17a7253ec4cf8fb36c6022aa11d45.jpg)

![](images/0fc9d9fe1409f0e4102e5fc18b1a5ae5d776e72c7f6d32932f5cbb4baf402bc1.jpg)

![](images/5f51d50242e6e34898c2932cd7003437f94942784ccbd64d25cb0e5f2e23b19d.jpg)

![](images/5477225df67a55fdc37baaa8b79d411ba6200cd25ffff7ba9acb6122fb51f0cc.jpg)

![](images/b2838c154de4e89b1e83ee83033e524889d7024be4d636027d1daeb4b6b6ce22.jpg)

![](images/d8801aa4c337535e114d00861ca19c9606ed61807072c90a09522ade47c0be6e.jpg)

TABLE V  
ABLATION RESULTS REPORTED IN BALANCED ACCURACY (MEAN ± STD). NOTE THAT THE FOUR SCALE-RELATED SETUPS FORM A 2 × 2 FACTORIAL DECOMPOSITION OVER INPUT-SIDE AND OBJECTIVE-SIDE SCALE MODELING. THE BEST RESULTS ARE MARKED.
<table><tr><td>Setup</td><td>PhysioNet-MI</td><td>SEED-V</td></tr><tr><td>w/o Pre-training</td><td> $0 . 6 1 8 0 \pm 0 . 0 0 4 7$ </td><td> $0 . 3 9 5 7 \pm 0 . 0 0 4 8$ </td></tr><tr><td>w/o Scale-Aware PEs</td><td> $0 . 6 4 5 8 \pm 0 . 0 0 8 1$ </td><td> $0 . 4 1 1 1 \pm 0 . 0 0 8 8$ </td></tr><tr><td>w/o Multi-Scale</td><td> $0 . 6 3 5 8 \pm 0 . 0 0 7 9$ </td><td> $0 . 4 1 0 0 \pm 0 . 0 0 5 0$ </td></tr><tr><td>w/o MS Tokenization</td><td> $0 . 6 5 1 8 \pm 0 . 0 0 6 7$ </td><td> $0 . 4 0 5 4 \pm 0 . 0 0 9 5$ </td></tr><tr><td>w/o Scale-Specific Recon</td><td> $0 . 6 4 0 2 \pm 0 . 0 0 5 4$ </td><td> $0 . 3 9 6 9 \pm 0 . 0 0 5 2$ </td></tr><tr><td>Full</td><td> $\mathbf { 0 . 6 5 2 4 \ : \pm { \ : 0 . 0 1 1 1 } }$ </td><td> $\mathbf { 0 . 4 1 1 6 \ : \pm { \ : 0 . 0 0 5 6 } }$ </td></tr><tr><td>Setup</td><td>HMC</td><td>MentalArithmetic</td></tr><tr><td>w/o Pre-training</td><td> $0 . 7 2 7 7 \pm 0 . 0 0 4 7$ </td><td> $0 . 5 7 9 2 \pm 0 . 0 1 9 3$ </td></tr><tr><td>w/o Scale-Aware PEs</td><td> $0 . 7 5 0 4 \pm 0 . 0 0 9 1$ </td><td> $0 . 7 2 2 9 \pm 0 . 0 3 4 4$ </td></tr><tr><td>w/o Multi-Scale</td><td> $0 . 7 4 7 1 \pm 0 . 0 0 6 4$ </td><td> $0 . 7 2 9 2 \pm 0 . 0 8 3 4$ </td></tr><tr><td>w/o MS Tokenization</td><td> $0 . 7 4 4 5 \pm 0 . 0 0 7 1$ </td><td> $0 . 6 7 6 4 \pm 0 . 0 3 0 7$ </td></tr><tr><td>w/o Scale-Specific Recon</td><td> $0 . 7 5 3 2 \pm 0 . 0 0 5 7$ </td><td> $0 . 6 8 8 2 \pm 0 . 0 4 9 8$ </td></tr><tr><td>Full</td><td> $\mathbf { 0 . 7 5 4 0 \ : \pm { \ : 0 . 0 0 9 8 } }$ </td><td> $\mathbf { 0 . 7 7 7 1 \pm 0 . 0 4 2 9 }$ </td></tr><tr><td>Setup</td><td>TUAB</td><td>TUEV</td></tr><tr><td>w/o Pre-training</td><td></td><td></td></tr><tr><td>w/o Scale-Aware PEs</td><td> $0 . 7 9 0 2 \pm 0 . 0 0 6 1$ </td><td> $0 . 6 7 3 6 \pm 0 . 0 3 5 4$ </td></tr><tr><td></td><td> $0 . 8 3 2 0 \pm 0 . 0 0 3 8$ </td><td> $\mathbf { 0 . 7 3 1 2 \ : \pm { \ : 0 . 0 1 4 5 } }$ </td></tr><tr><td>w/o Multi-Scale</td><td> $\mathbf { 0 . 8 3 3 7 \ : \pm { \ : 0 . 0 0 3 1 } }$ </td><td> $0 . 6 9 8 6 \pm 0 . 0 2 1 2$ </td></tr><tr><td>w/o MS Tokenization</td><td> $0 . 8 3 0 1 \pm 0 . 0 0 4 2$ </td><td> $0 . 6 9 3 2 \pm 0 . 0 1 6 9$ </td></tr><tr><td>w/o Scale-Specific Recon</td><td> $0 . 8 3 2 4 \pm 0 . 0 0 2 9$ </td><td> $0 . 6 9 4 5 \pm 0 . 0 2 2 1$ </td></tr><tr><td>Full</td><td> $0 . 8 3 2 4 \pm 0 . 0 0 0 7$ </td><td> $0 . 7 2 1 0 \pm 0 . 0 2 5 9$ </td></tr></table>

## G. Further Analyses

To further analyze ScaleBrain, we study the scale selection, the attention mechanism, and the objective weights.

Analysis of the Necessity of Scales. To validate the necessity of coarse, medium, and fine scales $( s _ { 1 } = 2 0 0 , s _ { 2 } =$ $1 0 0 , s _ { 3 } = 4 0 )$ , we compare the full ScaleBrain model against three single-scale configurations and three double-scale combinations on four representative datasets. As illustrated in Fig. 3, the multi-scale combinations generally outperform their single-scale counterparts. Furthermore, the three-scale configuration (“Ours”) consistently yields the best or highly competitive results across Balanced Accuracy, Cohen’s Kappa, and Weighted F1. This suggests that coarse-to-fine temporal scales provide complementary information for EEG representation learning.

Analysis of the Effect of Additional Scales. To investigate whether incorporating more temporal scales beyond the default $ { \boldsymbol { N } } \_ = \ 3$ can further enhance representation capacity, we conduct experiments with larger scale numbers, specifically N = 4 and $N = 5 .$ . For $N = 4 ,$ we add a new scale $s _ { 4 } = 2 0$ and for $N = 5 ,$ we further add $s _ { 5 } = 1 0 $ . We evaluate these variants across four representative datasets. As illustrated in Fig. 4, the results consistently show that the default $N = 3$ configuration yields the most favorable performance among the evaluated configurations. On PhysioNet-MI, SEED-V, and MentalArithmetic, increasing the scale number to $N = 4$ or $N = 5$ leads to varying degrees of performance degradation and sometimes introduces larger variances. On the TUEV dataset, while the $N = 5$ variant achieves comparable results to the baseline, it does not provide a meaningful improvement. This phenomenon suggests that simply stacking more temporal scales does not necessarily improve representation learning. Instead, an excessive number of scales may introduce redundant representations and increase optimization difficulty. Therefore, the coarse, medium, and fine three-scale partition (N = 3) provides a favorable trade-off between temporal coverage, representation diversity, and optimization complexity.

Analysis of the Objective Weights. To validate the effectiveness of the scale-balanced design of the objective weights, i.e., $\lambda _ { 1 } = 0 . 5 , \lambda _ { 2 } = 1 . 0 , \lambda _ { 3 } = 2 . 0$ (“Balanced $\mathrm { W e i g h t s ^ { \prime \prime } ) }$ , we conduct pre-training with “Equal Weights” $( \lambda _ { 1 } = \lambda _ { 2 } = \lambda _ { 3 } = 1 . 0 )$ and evaluate the model on four representative datasets. As illustrated in Fig. 5, the proposed scale-balanced weighting scheme consistently outperforms the equal-weight baseline across all four datasets and all evaluated metrics. Notably, the performance improvements are particularly pronounced on the MentalArithmetic and TUEV datasets. One possible explanation is that the reconstruction targets at different scales exhibit different statistical distributions and energy characteristics. Under equal weighting, these differences may cause the optimization to favor some scales over others. We therefore assign progressively larger weights to finer temporal resolutions to reduce optimization bias toward coarse-scale structure and balance the contribution of different temporal scales. These results provide empirical support for our balanced objective weighting design.

Analysis of the Attention Mechanism. We compare Axial ST attention with the criss-cross attention used in CBraMod [16]. For a fair comparison, the criss-cross variant uses 12 layers to maintain a similar parameter budget. Fig. 6 reports the results on four datasets, where Axial ST achieves competitive or superior performance across the majority of evaluated datasets and metrics, although criss-cross attention performs better on some individual settings. These results suggest that decoupling spatial and temporal attention provides a competitive alternative for contextualizing the fused multi-scale representation, while neither attention design dominates uniformly across all downstream tasks.

## V. CONCLUSION

In this paper, we propose ScaleBrain, a scale-consistent pre-training framework for EEG foundation models that addresses the limitation of current multi-scale pre-training. By incorporating multi-scale input tokenization and scale-specific reconstruction targets, ScaleBrain explicitly models EEG representations at different temporal resolutions from both the input and optimization objective sides. Scale-aware spatiotemporal position encodings preserve spatial and temporal correspondence across scales, while gated scale fusion and axial spatio-temporal attention integrate and contextualize the resulting multi-scale representations. Extensive evaluations across six BCI benchmarks demonstrate the effectiveness of the proposed framework. ScaleBrain achieves the best mean Balanced Accuracy on five datasets and ranks third on the remaining one, while using only 5.0M parameters. These results demonstrate that effective multi-scale pre-training requires coordinated representation and supervision across temporal scales. Moving forward, we plan to extend this scale-consistent approach to other neuroimaging modalities such as magnetoencephalography (MEG), and investigate more efficient attention mechanisms to improve the practicality of large-scale EEG foundation models.

## REFERENCES

[1] K. N. Kay, T. Naselaris, R. J. Prenger, and J. L. Gallant, “Identifying natural images from human brain activity,” Nature, vol. 452, no. 7185, pp. 352–355, 2008.

[2] S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, J. Schmidt, and M. Shah, “Decoding brain representations by multimodal learning of neural activity and visual features,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 11, pp. 3833–3849, 2020.

[3] A. Biasiucci, B. Franceschiello, and M. M. Murray, “Electroencephalography,” Current Biology, vol. 29, no. 3, pp. R80–R85, 2019.

[4] A. Al-Saegh, S. A. Dawwd, and J. M. Abdul-Jabbar, “Deep learning for motor imagery EEG-based classification: A review,” Biomedical Signal Processing and Control, vol. 63, p. 102172, 2021.

[5] Y. Chen, S. Zhao, S. Li, and G. Pan, “EMOD: A unified EEG emotion representation framework leveraging v-a guided contrastive learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40. AAAI Press, 2026, pp. 17 427–17 435.

[6] J. Wang, S. Zhao, H. Jiang, S. Li, T. Li, and G. Pan, “Generalizable sleep staging via multi-level domain alignment,” in Thirty-Eighth AAAI Conference on Artificial Intelligence. AAAI Press, 2024, pp. 265–273.

[7] S. Roy, I. Kiral-Kornek, and S. Harrer, “ChronoNet: A deep recurrent neural network for abnormal EEG identification,” in Artificial Intelligence in Medicine: Conference on Artificial Intelligence in Medicine. Springer, 2019, pp. 47–56.

[8] B. Blankertz, R. Tomioka, S. Lemm, M. Kawanabe, and K.-R. Muller, “Optimizing spatial filters for robust EEG single-trial analysis,” IEEE Signal Processing Magazine, vol. 25, no. 1, pp. 41–56, 2007.

[9] I. Guler and E. D. Ubeyli, “Multiclass support vector machines for EEGsignals classification,” IEEE Transactions on Information Technology in Biomedicine, vol. 11, no. 2, pp. 117–126, 2007.

[10] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGnet: a compact convolutional neural network for EEG-based brain–computer interfaces,” Journal of Neural Engineering, vol. 15, no. 5, p. 056013, 2018.

[11] A. Demir, T. Koike-Akino, Y. Wang, M. Haruna, and D. Erdogmus, “EEG-GNN: Graph neural networks for classification of electroencephalogram (EEG) signals,” in Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2021, pp. 1061–1067.

[12] Y. Song, X. Jia, L. Yang, and L. Xie, “Transformer-based spatial-temporal feature learning for EEG decoding,” arXiv preprint arXiv:2106.11170, 2021.

[13] C. Yang, M. Westover, and J. Sun, “BIOT: Biosignal transformer for cross-data learning in the wild,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 78 240–78 260.

[14] W.-B. Jiang, L. Zhao, and B.-L. Lu, “Large brain model for learning generic representations with tremendous EEG data in bci,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 16 405–16 426.

[15] Y. Zhou, J. Wu, Z. Ren, Z. Yao, W. Lu, K. Peng, Q. Zheng, C. Song, W. Ouyang, and C. Gou, “CSBrain: A cross-scale spatiotemporal brain foundation model for EEG decoding,” in Advances in Neural Information Processing Systems, vol. 38, 2025, pp. 87 150–87 195.

[16] J. Wang, S. Zhao, Z. Luo, Y. Zhou, H. Jiang, S. Li, T. Li, and G. Pan, “CBraMod: A criss-cross brain foundation model for EEG decoding,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 75 310–75 346.

[17] G. Buzsaki and A. Draguhn, “Neuronal oscillations in cortical networks,” Science, vol. 304, no. 5679, pp. 1926–1929, 2004.

[18] R. T. Canolty and R. T. Knight, “The functional role of cross-frequency coupling,” Trends in Cognitive Sciences, vol. 14, no. 11, pp. 506–515, 2010.

[19] J. Ma, F. Wu, Q. Lin, Y. Xing, C. Liu, Z. Jia, and M. Feng, “CodeBrain: Bridging decoupled tokenizer and multi-scale architecture for EEG foundation model,” in The Fourteenth International Conference on Learning Representations, 2026.

[20] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, “Deep learning with convolutional neural networks for EEG decoding and visualization,” Human Brain Mapping, vol. 38, no. 11, pp. 5391– 5420, 2017.

[21] Y. Song, Q. Zheng, B. Liu, and X. Gao, “EEG conformer: Convolutional transformer for EEG decoding and visualization,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 710–719, 2022.

[22] Y. Wang, D.-H. Lee, and X. Bruce, “BraSTORM: A dual-branch selfsupervised framework for EEG representation learning via input-level spatio-temporal decomposition,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40. AAAI Press, 2026, pp. 17 805–17 813.

[23] Y. El Ouahidi, J. Lys, P. Tholke, N. Farrugia, B. Pasdeloup, V. Gripon,¨ K. Jerbi, and G. Lioi, “REVE: A foundation model for EEG-adapting to any setup with large-scale pretraining on 25,000 subjects,” in Advances in Neural Information Processing Systems, vol. 38, 2026, pp. 22 541– 22 577.

[24] K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, “Filter bank common spatial pattern (FBCSP) in brain-computer interface,” in 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence). IEEE, 2008, pp. 2390–2397.

[25] Y. Ding, N. Robinson, S. Zhang, Q. Zeng, and C. Guan, “TSception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition,” IEEE Transactions on Affective Computing, vol. 14, no. 3, pp. 2238–2250, 2022.

[26] W. Fan, J. Fei, D. Guo, K. Yi, X. Song, H. Xiang, H. Ye, and M. Li, “Towards multi-resolution spatiotemporal graph learning for medical time series classification,” in Proceedings of the ACM on Web Conference 2025, 2025, pp. 5054–5064.

[27] Y. Wang, N. Huang, T. Li, Y. Yan, and X. Zhang, “Medformer: A multi-granularity patching transformer for medical time-series classification,” Advances in Neural Information Processing Systems, vol. 37, pp. 36 314–36 341, 2024.

[28] B. Zhang and R. Sennrich, “Root mean square layer normalization,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[29] N. Shazeer, “GLU variants improve transformer,” arXiv preprint arXiv:2002.05202, 2020.

[30] I. Obeid and J. Picone, “The Temple University Hospital EEG data corpus,” Frontiers in Neuroscience, vol. 10, p. 196, 2016.

[31] A. L. Goldberger, L. A. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals,” Circulation, vol. 101, no. 23, pp. e215–e220, 2000.

[32] W. Liu, J.-L. Qiu, W.-L. Zheng, and B.-L. Lu, “Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition,” IEEE Transactions on Cognitive and Developmental Systems, vol. 14, no. 2, pp. 715–729, 2021.

[33] D. Alvarez-Estevez and R. M. Rijsman, “Inter-database validation of a deep learning approach for automatic sleep scoring,” PloS one, vol. 16, no. 8, p. e0256111, 2021.

[34] I. Zyma, S. Tukaev, I. Seleznov, K. Kiyono, A. Popov, M. Chernykh, and O. Shpenkov, “Electroencephalograms during mental arithmetic task performance,” Data, vol. 4, no. 1, p. 14, 2019.

[35] J. Jing, W. Ge, S. Hong, M. B. Fernandes, Z. Lin, C. Yang, S. An, A. F. Struck, A. Herlopian, I. Karakis et al., “Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation,” Neurology, vol. 100, no. 17, pp. e1750–e1762, 2023.

[36] C. Yang, C. Xiao, M. B. Westover, and J. Sun, “Self-supervised electroencephalogram representation learning for automatic sleep staging: model development and evaluation study,” JMIR AI, vol. 2, no. 1, p. e46769, 2023.

[37] H. Li, M. Ding, R. Zhang, and C. Xiu, “Motor imagery EEG classification algorithm based on cnn-lstm feature fusion network,” Biomedical Signal Processing and Control, vol. 72, p. 103342, 2022.