# OneEEG: A Unified Signal-Visual Multimodal Language Model with Task-Aware Mixture of Projectors for Multi-Task EEG Understanding

Jiacheng Zhang, Xueyi Zhang, Xin Lu, Siqi Cai, Haizhou Li

Abstract—Electroencephalography (EEG) is widely used in clinical assessment, brain-computer interfaces, and cognitivestate analysis, yet EEG modeling remains fragmented across task paradigms. Recent multimodal large language models (MLLMs) provide a promising interface for flexible multimodal reasoning, but their ability to understand EEG inputs remains limited. In this work, we propose OneEEG, a unified EEG-MLLM framework for multi-task EEG understanding. By training on our curated OneEEG-Instruct, a multi-task EEG instruction-tuning dataset, OneEEG jointly models numerical EEG representations and visual EEG representations, and introduces a task-aware Mixture of Projectors (MoP) to align heterogeneous EEG features across different task families with the LLM embedding space. We further establish OneEEG-Bench, a benchmark for evaluating current MLLMs and our method on multi-task EEG understanding. Experiments show that existing MLLMs have limited ability to interpret EEG inputs, while OneEEG achieves stronger performance by integrating numerical and visual EEG representations with task-aware EEG-to-LLM projection. Extensive ablations and analyses further reveal the effects of visual EEG renderings, MoP variants, router behavior, and feature dispersion across EEG tasks.

Index Terms—Electroencephalography analysis, multimodal large language models, biomedical signal processing, mixture of experts.

## I. INTRODUCTION

LECTROENCEPHALOGRAPHY (EEG) has long activity, owing to its non-invasive acquisition, relatively low cost, and high temporal resolution [1], [2]. These properties have made EEG central to a wide range of research and application scenarios, including neurological assessment [3], seizure analysis [1], sleep monitoring [4], brain-computer interfaces [5], cognitive-state estimation [6], and affective computing [7]. Despite this broad relevance, EEG modeling remains largely fragmented across task paradigms. EEG applications are often associated with distinct experimental protocols, channel configurations, subject populations, annotation schemes, and physiological phenomena of interest. More importantly, the discriminative patterns in EEG are highly task-dependent, spanning temporal dynamics, spectral characteristics, and spatial distributions. As a result, most EEG models are still developed under taskspecific assumptions [8]–[10], with dedicated preprocessing procedures and model architectures tailored to individual datasets or label spaces. This task-specific modeling paradigm limits the development of EEG systems that can support heterogeneous tasks under a unified representation and interaction framework.

![](images/814d5a394d10f7754e3c8eca854d3f2eeee66fc76583228f6991669956d991af.jpg)

![](images/37160fdaa9b43f3e64e63da253d5ab46e5f76dc18ad8fa31691671548813c73d.jpg)  
Fig. 1. Motivation of MoP. Colored points denote EEG features from different task families or datasets. A shared EEG-to-LLM projector forces heterogeneous EEG features through a single projector, which may cause cross-task conflicts as indicated by the red dashed paths. MoP mitigates this bottleneck by aligning task-dependent EEG features through task-aware projection.

Recent advances in large language models (LLMs) have motivated efforts to connect EEG signals with LLMs or multimodal LLMs (MLLMs) to enable multi-task EEG understanding [11]–[13]. Existing efforts generally follow two directions. One line of work adopts a standard MLLM paradigm [11], [13], where the EEG encoder extracts features from numerical signals, and a shared projector maps them into the LLM embedding space. Another line of work studies image-form EEG understanding, where EEG signals are rendered as waveform plots, spectra, or topographic maps and interpreted by visionlanguage models [12], [14]. However, these paradigms suffer from two important limitations. (1) Single projector. A single shared projector may be insufficient for aligning heterogeneous EEG representations with the semantic space of an LLM. Unlike natural images or text, EEG is a low-signal-to-noise physiological signal with weak intrinsic semantic information, and EEG features can be highly dispersed across different tasks. Projecting such heterogeneous features through a single mapping module may introduce cross-task interference and weaken task-specific semantic alignment. (2) Single input form. Existing approaches often support only one form of EEG input, either a frame of multiple 1-dimensional EEG signals, i.e., numerical EEG representations, or a frame of 2- dimensional visual image of EEG signals, i.e., visual EEG representations. Numerical EEG representations preserve the original signal form and retain fine-grained temporal information, while visual EEG representations capture the cooccurrence of EEG events across channels within a frame. Each has its pros and cons. In many real-world settings, such as sleep monitoring, clinical diagnosis, and brain function assessment, visual EEG representations are useful for experts to inspect and interpret EEG recordings. On the other hand, although numerical signals provide more direct access to finegrained EEG information, an encoder pre-trained on one set of numerical signals may not be compatible with others. In contrast, visual EEG representations provide a more flexible interface for datasets with non-standard channel systems. These considerations motivate us to embrace both numerical and visual EEG representations.

Therefore, we propose OneEEG, a unified EEG-MLLM framework for multi-task EEG understanding that jointly models numerical EEG representations and visual EEG representations. OneEEG adopts a dual-stream architecture, in which the numerical signal stream extracts neural features from numerical EEG representations, while the visual stream processes rendered EEG waveform images. The features from both streams are projected into the semantic space of the LLM backbone, supporting EEG-oriented question answering over both signal-derived features and visual EEG patterns. To address the heterogeneous alignment problem across EEG tasks, OneEEG further introduces a task-aware Mixture of Projectors (MoP) module. Rather than relying on a single shared EEG-to-LLM projection, MoP employs a task-aware routing mechanism over a bank of projectors to align EEG representations from different task families with the LLM embedding space through task-aware projection. By combining dual-stream EEG input modeling with task-aware MoP, OneEEG provides a unified baseline for heterogeneous EEG understanding.

To support the training of OneEEG, we curate OneEEG-Instruct, a multi-task EEG instruction-tuning dataset that transforms heterogeneous EEG tasks into natural-language supervision. OneEEG-Instruct supports dual-stream EEG inputs, covering both numerical EEG representations and waveformbased visual EEG representations, and spans diverse EEG task paradigms involving clinical, event-related, cognitive, and motor-intent understanding. Beyond aggregating heterogeneous tasks, OneEEG-Instruct adopts a hierarchical instruction design to integrate both task-specific EEG comprehension and general signal-level perception. The task-QA tier is closely tied to downstream EEG tasks and captures dataset-specific prediction objectives, whereas the basic-QA tier focuses on task-agnostic EEG attributes that reflect low-level signal characteristics. In parallel, we establish OneEEG-Bench, an EEG-MLLM benchmark for systematically assessing EEG understanding in multimodal language models. OneEEG-Bench evaluates general closed-source MLLMs, open-source MLLMs, medical MLLMs, EEG-specialized MLLMs, and our proposed OneEEG under a meticulously designed evaluation protocol. Together, OneEEG-Instruct and OneEEG-Bench provide a consistent basis for training EEG-specific MLLMs and evaluating the capability of existing multimodal language models in multi-task EEG understanding.

In summary, our main contributions are as follows:

• We propose OneEEG, a unified EEG-MLLM framework that supports both numerical EEG representations and visual EEG representations for multi-task EEG understanding. We further introduce a task-aware Mixture of Projectors (MoP) to mitigate the single-projector bottleneck and improve task-aware EEG-to-LLM alignment across heterogeneous EEG tasks.

• We construct OneEEG-Instruct and OneEEG-Bench for training and evaluation, covering diverse EEG task families, channel settings, and closed-set QA formats, with hierarchical annotations spanning task-oriented EEG reasoning and basic signal attribute perception.

• We conduct extensive experiments on OneEEG-Bench to systematically evaluate current MLLMs and analyze the effectiveness of OneEEG across diverse EEG tasks, demonstrating the superior performance of OneEEG over existing baselines.

## II. RELATED WORKS

In this section, we review the literature relevant to OneEEG, including EEG representation learning and foundation models, multimodal large language models for EEG understanding, and the development of the mixture of experts mechanism.

## A. EEG Representation Learning and Foundation Models

EEG modeling has traditionally been developed in taskspecific settings, where models are designed for individual applications such as seizure detection [1], sleep staging [4], motor imagery decoding [8], or emotion recognition [7], [15], [16]. Such task-dependent pipelines often rely on dedicated preprocessing strategies, feature designs, and supervised objectives, making it difficult to build transferable EEG models across heterogeneous paradigms. To improve the generality of EEG representations, recent studies have explored EEG foundation models trained on large-scale or multi-dataset EEG corpora. Representative works, including LaBraM [17], EEGPT [18], Brant [19], CBraMod [20], BrainOmni [21],

BIOT [22], Neuro-GPT [23], and NeuroLM [24], adopt selfsupervised or large-scale pretraining strategies to learn transferable representations from EEG signals.

These models share the goal of moving beyond isolated task-specific training toward general-purpose EEG encoders. Although EEG foundation models are primarily designed as representation encoders and are not directly equipped with natural-language interaction or instruction-following capabilities, they have substantially advanced transferable EEG representation learning and provide a strong foundation for building more general EEG physiological signal understanding systems. Our work builds on this progress by integrating EEG signal representations with multimodal language models for unified, instruction-driven EEG understanding.

## B. Multimodal Large Language Models for EEG Understanding

Recent advances in multimodal large language models (MLLMs) have enabled strong visual-language reasoning across natural images, documents, charts, and medical images. General-purpose MLLMs, such as GPT-5.5 [25], Gemini-3 Pro [26], Qwen VL series [27], [28], and LLaVA series [29], have demonstrated impressive open-ended visual understanding capabilities, while medical MLLMs, such as Lingshu [30] and MedGemma [31], have extended multimodal reasoning to biomedical and clinical scenarios. However, visual EEG representations differ substantially from both natural images and conventional medical images, since waveforms, spectra, and topographic maps are visual renderings of physiological time-series signals. Their interpretation depends on temporal rhythms, frequency-band patterns, and task-specific neural phenomena. Therefore, despite the rapid progress of MLLMs, their capability for EEG understanding remains insufficiently studied, motivating systematic evaluation across heterogeneous EEG tasks.

Recent studies have begun to explore EEG-oriented LLMs and MLLMs. CerebraGloss [12] constructs an instructiontuned vision-language model for fine-grained clinical EEG interpretation based on waveform images and structured instruction data. WaveMind [11] aligns EEG signals with textual and visual modalities to build a conversational EEG model. EEG-VLM [14] investigates EEG image-based visionlanguage modeling for sleep stage prediction through hierarchical visual feature alignment and language-guided reasoning. These works demonstrate the potential of languagemodel interfaces for EEG analysis. However, most existing efforts still emphasize either signal-to-language modeling or visual EEG interpretation alone, while unified support for both numerical EEG representations and visual EEG representations remains underexplored. In addition, existing models, including general MLLMs, medical MLLMs, and emerging EEG-oriented LLM/MLLM methods, have not been comprehensively evaluated under a unified benchmark across diverse EEG task paradigms. These gaps motivate unified signalvisual EEG modeling and systematic benchmarking for EEG understanding.

## C. Mixture-of-Experts for EEG Analysis

Mixture-of-Experts (MoE) architectures have been widely studied as a mechanism for conditional computation and expert specialization [32]–[35]. By routing inputs to different experts, MoE-based models can allocate specialized capacity to heterogeneous data distributions while avoiding the need for a single shared module to fit all patterns. This idea is particularly relevant to EEG modeling, where heterogeneity arises from multiple sources, including tasks, subjects, channels, frequency patterns, and recording protocols. Recent EEG studies have explored MoE designs from different perspectives. EEGMamba [36] introduces task-aware MoE to enable coverage of multiple EEG tasks in Mamba. BrainMoE [37] employs channel-wise expert routing to model channel-dependent neural information, while EEGMoE [38] explores domain-decoupled experts for self-supervised EEG representation learning.

Despite these advances, existing EEG-MoE methods are primarily concerned with EEG feature extraction and representation learning. In the context of EEG-MLLMs, the key challenge further involves modality alignment: heterogeneous EEG features with weak intrinsic semantic information must be transformed into the LLM feature space. This makes the EEG-to-LLM projection stage a critical bottleneck. In contrast to encoder-level or classifier-level MoE designs, our proposed MoP introduces task-aware specialization at the projection stage, guiding different EEG task families to specialized EEGto-LLM projections. This provides a lightweight mechanism for mitigating cross-task alignment conflicts in instructionfollowing EEG understanding.

## III. DATA CONSTRUCTION

We now discuss how we construct OneEEG-Instruct and OneEEG-Bench. We first demonstrate the source datasets and task taxonomy, then describe the process of EEG preprocessing and visual rendering, and finally explain how labelconstrained question-answer samples are generated. Fig. 2 provides an overview of the dataset construction pipeline. (The EEG datasets used in this study are either publicly available or accessed through approved data-use requests. No new humansubject data were collected by the authors, and informed consent and ethical approvals were handled by the original dataset providers.)

## A. Dataset Sources and Task Taxonomy

OneEEG-Instruct and OneEEG-Bench are built from multiple public EEG datasets covering diverse task scenarios and channel configurations. As summarized in Table I, we organize these datasets into six task families: motor, state, clinical, event, sleep, and vigilance. Motor and state datasets focus on movement-related and cognitive/affective EEG patterns, clinical datasets involve neurological diagnosis, event datasets contain transient EEG events such as artifacts, epileptiform discharges, and seizures, while sleep and vigilance datasets aim at sleep staging and drowsiness estimation.

We further divide the datasets according to their compatibility with a standard 19-channel 10–20 setting. Specifically, we define a common 19-channel subset from the international 10– 20 system [39] (Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, P7, P3, Pz, P4, P8, O1, O2). This subset covers major scalp regions, including frontal, central, parietal, occipital, temporal, and midline areas, and provides a compact channel system that can be consistently shared across multiple EEG datasets. Datasets that can be reliably mapped to this subset are categorized into the standard setting. Datasets that cannot be reliably mapped to the standard 19-channel 10–20 setting are categorized into the non-standard setting. The non-standard setting includes recordings with bipolar derivations, recordings concentrated on specific functional regions, and cases where the available channels do not match the selected 19-channel subset.

![](images/e293f5b1dbe2e87ce28baebffda3a572bc93502bd3e2ea4c441f7c1942fd5fc0.jpg)  
Fig. 2. Construction pipeline of OneEEG-Instruct and OneEEG-Bench. Raw EEG datasets are first converted into EEG signal frames with metadata, followed by preprocessing, subject-level splitting, and EEG image rendering. Question-answer annotations are then generated from the processed signals, images, and metadata, forming label-constrained MCQ and yes/no QA samples.

## B. EEG Preprocessing and Visual Rendering

For each source dataset, we parse the raw EEG recordings together with the official annotations and convert them into a unified frame-level format. The EEG signals are processed with basic signal preprocessing, including band-pass filtering, notch filtering, resampling, and dataset-specific channel handling. Datasets in the standard setting are mapped to the standard 19-channel 10–20 subset defined above, while datasets in the non-standard setting keep their native channel configurations. Most datasets are segmented into 4-second frames, while SleepEDF follows the standard 30-second sleep epoch. Each frame is associated with the corresponding task label according to the original dataset annotation and saved as a numerical EEG representation frame for signal-side modeling when applicable.

For each EEG frame, we also render the multi-channel frame into a time-channel 2D waveform image. In this way, we form a signal-image pair between the EEG signal frame and its visual image, providing two views for one EEG frame as a numerical time series or as a visual EEG representation.

TABLE I  
DATASET COMPOSITION OF ONEEEG-INSTRUCT AND ONEEEG-BENCH.
<table><tr><td>Dataset</td><td>Family</td><td>Setting</td><td>Train/Eval</td><td>Format</td></tr><tr><td>BCIC-IV-2a [40]</td><td>Motor</td><td>Std.</td><td>4.6k / 320</td><td>MCQ+Y/N</td></tr><tr><td>EEG Motor [41]</td><td>Motor</td><td>Std.</td><td>17.7k / 320</td><td>MCQ+Y/N</td></tr><tr><td>EEGMAT [42]</td><td>State</td><td>Std.</td><td>3.8k / 240</td><td>MCQ+Y/N</td></tr><tr><td>SEED-IV [43]</td><td>State</td><td>Std.</td><td>60.1k / 360</td><td>MCQ+Y/N</td></tr><tr><td>TUAB 3 [44], [45]</td><td>Clinical</td><td>Std.</td><td>50.0k / 300</td><td>Y/N</td></tr><tr><td>TUEP [44], [45]</td><td>Clinical</td><td>Std.</td><td>50.9k / 300</td><td>Y/N</td></tr><tr><td>TUAR [44]-[46]</td><td>Event</td><td>Std.</td><td>65.4k /  240</td><td>MCQ+Y/N</td></tr><tr><td>TUEV [44], [45]</td><td>Event</td><td>Std.</td><td>3.9k / 360</td><td>MCQ+Y/N</td></tr><tr><td>TUSZ [44], [45], [47]</td><td>Event</td><td>Std.</td><td>49.0k / 320</td><td>MCQ+Y/N</td></tr><tr><td>SleepEDF [48]</td><td>Sleep</td><td>Non-std.</td><td>14.0k / 270</td><td>MCQ+Y/N</td></tr><tr><td>DRÉAMS [49]</td><td>Sleep</td><td>Non-std.</td><td>9.0k / 304</td><td>Y/N</td></tr><tr><td>SEED-VIG [50]</td><td>Vigilance</td><td>Non-std.</td><td>74.3k / 185</td><td>Y/N</td></tr><tr><td>Basic attributes</td><td>Basic</td><td>Mixed</td><td>76.4k / 609</td><td>MCQ</td></tr></table>

We further split the data into OneEEG-Instruct and OneEEG-Bench at the subject level when subject identifiers are available, enabling evaluation of cross-subject generalization rather than memorization of subject-specific patterns. We also generate spectrogram images as an auxiliary visual representation for analysis, but the main training and evaluation protocol is based on waveform images.

## C. Question-Answer Annotation

We design both OneEEG-Instruct and OneEEG-Bench under a closed-set QA protocol. Each question is associated with a finite candidate label set, and the answer is required to be selected from this set. In practice, we mainly use two QA formats: multiple-choice questions and binary yes/no questions, with details summarized in Table I.

To cover both dataset-specific reasoning and general signal perception, we organize QA samples into two levels, with examples shown in Fig. 3. Task-oriented QA follows the original prediction objective of each source dataset, evaluating whether the model can perform EEG reasoning under dataset-specific label spaces. In addition, we construct basic EEG attribute QA from subsets of the source datasets. These samples focus on task-agnostic signal properties, such as amplitude level, dominant frequency band, and slow-versusfast activity tendency, enabling the evaluation of low-level EEG characteristics beyond dataset-specific task labels.

For each processed EEG frame, we construct QA annotations from its structured metadata. For task-oriented samples, the ground-truth answer is derived from the original dataset label or event annotation and mapped to the corresponding candidate label space. For basic EEG attribute samples, the answer label is computed from signal-level properties of the sampled EEG frame and converted into predefined closed-set attribute categories.

We then use Qwen3 [51] to diversify the wording of questions under the predefined candidate label space. Specifically, we design multiple templates to facilitate the LLM to generate questions with various expressions, thereby improving the ability of OneEEG to generate accurate responses even when encountering customized queries. During this process, the candidate labels and ground-truth answers remain unchanged, which preserves label correctness while increasing instruction diversity. This label-driven annotation strategy provides natural-language QA samples with deterministic closed-set answers for both training and evaluation.

![](images/a0fa2679171f8c6a9dcb9072cd66c25ddc11ae03466a32be87927fd55b521ba2.jpg)  
Fig. 3. Example of the closed-set QA annotation design in OneEEG. Taskoriented QA focuses on dataset-specific EEG tasks, while basic attribute QA focuses on general signal-level properties.

## IV. METHOD

OneEEG is a unified EEG-oriented MLLM trained on numerical EEG signals, waveform images, and text instructions. We first present the overall architecture and the dual-stream EEG input design, then the task-aware mixture of projectors to align the EEG features with the language model space. Finally, we detail the two-stage training strategy used for projector alignment and multimodal instruction tuning.

## A. Overview of OneEEG

Fig. 4 illustrates the overall architecture of OneEEG. Given an EEG frame and a text instruction, OneEEG constructs a multimodal input consisting of a numerical EEG representation, a waveform-based visual EEG representation, and the instruction text. For our method, we utilize Qwen2.5-VL-7B [27] as our LLM backbone with the native visual encoder, and EEGPT [18] as the numerical EEG signal encoder. The signal stream encodes numerical EEG representations with a pretrained EEG encoder, and projects the resulting features into the LLM hidden space through MoP. The visual stream processes the waveform image through the native visual pathway of the MLLM, and the text stream follows the standard tokenization and embedding procedure. The resulting EEG, visual, and text tokens are concatenated and fed into the language model for answer generation.

## B. Dual-stream EEG Input Modeling

We first describe how OneEEG handles channel-compatible and non-compatible EEG datasets. We then introduce how numerical and visual EEG representations are encoded and organized for the LLM backbone.

1) Channel-Compatible Input Modes: Following the dataset taxonomy in Sec. III-A, OneEEG adopts two input modes for both task-oriented QA and basic EEG attribute QA according to the channel setting. For standard-setting EEG frames, OneEEG uses both numerical and visual EEG representations. For non-standard-setting EEG frames, OneEEG omits the numerical EEG signal representation and uses only the visual EEG representation.

2) Numerical Stream: OneEEG uses the numerical signal when it is compatible with the selected 19-channel setting. Given an EEG frame with C channels and $T$ sampled time points, we denote the numerical signal as

$$
\mathbf { X } \in \mathbb { R } ^ { C \times T } .
$$

Subsequently, the EEG numerical signal encoder partitions X along the temporal dimension into non-overlapping patches. With a patch length of P, the input is divided into $N _ { e } = T / P$ temporal patches:

$$
\mathbf { X } = \{ \mathbf { X } _ { i } \} _ { i = 1 } ^ { N _ { e } } , \quad \mathbf { X } _ { i } \in \mathbb { R } ^ { C \times P } .
$$

Each patch contains the multi-channel EEG signals within a short time interval. The encoder encodes these temporal patches with channel-aware embeddings and Transformer layers, producing a sequence of EEG token features:

$$
\mathbf { Z } _ { \mathrm { e e g } } = f _ { \mathrm { e e g } } ( \mathbf { X } ) \in \mathbb { R } ^ { N _ { e } \times d _ { e } } ,
$$

where $N _ { e }$ denotes the number of EEG tokens and $d _ { e }$ is the EEG encoder feature dimension.

The EEG token features are then mapped to the hidden dimension of the language model by MoP:

$$
\begin{array} { r } { { \bf { { H } } } _ { \mathrm { { e e g } } } = \mathrm { M o P } ( { \bf { Z } } _ { \mathrm { e e g } } , t ) , \quad { \bf { { H } } } _ { \mathrm { { e e g } } } \in \mathbb { R } ^ { N _ { e } \times d _ { l } } , } \end{array}
$$

where $d _ { l }$ is the language-model hidden dimension and t denotes the task-family label used by MoP.

3) Visual Stream: For the visual stream, OneEEG takes the visual EEG representation, i.e., the waveform image I, as input and processes it through the native visual module of the underlying Qwen2.5-VL backbone:

$$
\mathbf { H } _ { \mathrm { v i s } } = P _ { v } ( f _ { v } ( \mathbf { I } ) ) \in \mathbb { R } ^ { N _ { v } \times d _ { l } } ,
$$

where $f _ { v }$ denotes the pretrained visual encoder, $P _ { v }$ denotes the original visual projection module, $N _ { v }$ is the number of visual tokens, and $d _ { l }$ is the hidden dimension of the language model.

4) Prefix-Augmented Visual Conditioning: For channelcompatible EEG frames, OneEEG further uses prefixaugmented visual conditioning to place waveform-derived visual tokens near the EEG tokens. In prefix-augmented visual conditioning, the waveform image is used in two ways. First, its visual features are appended after the EEG tokens as an auxiliary visual prefix:

$$
\mathbf { H } _ { \mathrm { p r e f i x } } = [ \mathbf { H } _ { \mathrm { e e g } } ; \mathbf { H } _ { \mathrm { v i s } } ^ { p } ] ,
$$

![](images/cb7478c20c2e6b42f69bf0da2ef32540d3b9bd283ab189959f21f21269386056.jpg)  
Fig. 4. Overview of OneEEG. OneEEG integrates numerical EEG representations, visual EEG representations, and text instructions into a unified MLLM input. OneEEG uses both numerical and visual streams for standard-setting datasets and the visual stream only for non-standard-setting datasets. The left panel details MoP: basic EEG attribute questions are assigned to a fixed basic projector, while task-oriented questions are handled by a router branch that selects one projector from the projector bank.

where $\mathbf { H } _ { \mathrm { v i s } } ^ { p }$ denotes the waveform-image features used as prefix tokens.

Second, the original image placeholder is retained in the text prompt, allowing the same waveform image to be processed through the native image pathway of the MLLM. The final input sequence is written as

$$
\mathbf { H } _ { \mathrm { i n p u t } } = [ \mathbf { H } _ { \mathrm { p r e f i x } } ; \mathbf { H } _ { \mathrm { p r o m p t } } ( \mathbf { I } , q ) ] ,
$$

where ${ \bf H } _ { \mathrm { p r o m p t } } ( { \bf I } , q )$ denotes the standard MLLM prompt representation formed by the waveform image I and the text instruction q.

The two visual paths serve different roles. The native image pathway preserves the pretrained image-text interface of the MLLM, whereas the auxiliary prefix provides local visual context adjacent to the newly introduced EEG signal tokens. This design helps bridge EEG signal features with waveform patterns in the pretrained visual space.

For non-standard channel settings where signal-side EEG tokens are omitted, OneEEG relies on the native waveformimage pathway without applying prefix-augmented visual conditioning. In this case, the input sequence reduces to the native MLLM prompt representation:

$$
\mathbf { H } _ { \mathrm { i n p u t } } = \mathbf { H } _ { \mathrm { p r o m p t } } ( \mathbf { I } , q ) ,
$$

where ${ \bf H } _ { \mathrm { p r o m p t } } ( { \bf I } , q )$ contains the native waveform-image tokens and the text-instruction tokens.

## C. Task-aware Mixture of Projectors

Given EEG features from the frozen EEG numerical signal encoder, OneEEG uses a task-aware Mixture of Projectors (MoP) module to map them into the LLM hidden space. As shown in Fig. 4, instead of using a single shared EEG-to-LLM projector, MoP maintains a lightweight projector bank and selects the projection pathway according to the EEG token features and the task embedding. Specifically, basic

EEG attribute questions are assigned to a fixed individual projector, while task-oriented questions are routed to candidate projectors through a hard-selected router.

1) Projector Bank: MoP maintains a lightweight EEG-to-LLM projector bank:

$$
\mathcal { P } = \{ P _ { 1 } , P _ { 2 } , \ldots , P _ { K } , P _ { \mathrm { b a s i c } } \} ,
$$

where $P _ { 1 } , \ldots , P _ { K }$ are candidate projectors for task-oriented questions and $P _ { \mathrm { b a s i c } }$ is the fixed projector for basic EEG attribute questions. Each projector is implemented as an MLP with the same architecture and maps EEG encoder features to the LLM hidden dimension:

$$
P _ { k } : \mathbb { R } ^ { d _ { e } }  \mathbb { R } ^ { d _ { l } } ,
$$

where $d _ { e }$ and $d _ { l }$ denote the EEG encoder feature dimension and the LLM hidden dimension, respectively.

For task-agnostic basic EEG attribute questions, OneEEG bypasses the router and directly applies the basic projector:

$$
\mathbf { H } _ { \mathrm { e e g } } = P _ { \mathrm { b a s i c } } ( \mathbf { Z } _ { \mathrm { e e g } } ) , \quad q \in \mathcal { Q } _ { \mathrm { b a s i c } } .
$$

For task-oriented questions, the projection path is selected from $\{ P _ { 1 } , \ldots , P _ { K } \}$ by the task-oriented router.

2) Task-oriented Router: For task-oriented EEG questions, OneEEG uses a router to select one projection path from the projector bank. Unlike basic EEG attribute questions that follow the basic projector, task-oriented samples are routed among candidate projector slots according to both EEG token features and the task embedding. The task embedding is derived from the task family of each sample and provides task context for projector selection.

Given EEG encoder features $\begin{array} { r c l } { { \mathbf { Z } _ { \mathrm { e e g } } } } & { { = } } & { { \{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N _ { e } } } } \end{array}$ and the task-family label t, OneEEG obtains the corresponding task embedding $\mathbf { e } _ { t } .$ The router concatenates this task embedding with each EEG token and computes token-level routing scores:

$$
\mathbf { a } _ { i } = W _ { g } [ \mathbf { z } _ { i } ; \mathbf { e } _ { t } ] , \quad \mathbf { a } _ { i } \in \mathbb { R } ^ { K } .
$$

![](images/bf36f6a663a2bfeb69e0f73997471b714be7a44c8e746b6c71a72ab7844211ae.jpg)

Stage 2: Multimodal Instruction Tuning  
![](images/ec11b80ab36ee8d3d4ea0b89caf19cc5c5fa419572b42c8c29e7c20b199cb68e.jpg)  
Fig. 5. Two-stage training strategy of OneEEG. In Stage 1, projector alignment trains MoP while keeping the signal encoder, image encoder, visual projector, and the LLM backbone frozen. In Stage 2, multimodal instruction tuning further trains MoP, the visual projector, and LoRA parameters of the LLM, while the signal and image encoders remain frozen. Snowflakes denote frozen modules, and flames denote trainable modules.

where K is the number of candidate projector slots. The token-level scores are averaged across EEG tokens to obtain sequence-level routing scores:

$$
\mathbf { s } = \frac { 1 } { N _ { e } } \sum _ { i = 1 } ^ { N _ { e } } \mathbf { a } _ { i } .
$$

The selected projector is determined by hard top-1 routing:

$$
k ^ { * } = \arg \operatorname* { m a x } _ { k \in \{ 1 , \dots , K \} } s _ { k } , \quad P _ { k ^ { * } } \in \{ P _ { 1 } , \dots , P _ { K } \} .
$$

The EEG tokens are then mapped into the LLM hidden space by the selected projector:

$$
\mathbf { H } _ { \mathrm { e e g } } = P _ { k ^ { * } } ( \mathbf { Z } _ { \mathrm { e e g } } ) , \quad q \in \mathcal { Q } _ { \mathrm { t a s k } } .
$$

We adopt a sparse routing strategy that activates only one projector slot for each task-oriented sample, keeping the EEGto-LLM projection efficient without dense mixtures over all projectors.

## D. Training Objectives and Stages

OneEEG is optimized with a standard autoregressive language modeling objective and trained in two stages. The details are as follows.

1) Training Objective: OneEEG is trained with the standard autoregressive language modeling objective. Given the multimodal input embeddings $\mathbf { H } _ { \mathrm { i n p u t } }$ and the target answer sequence $\mathbf { Y } = \{ y _ { t } \} _ { t = 1 } ^ { M }$ , the training loss is defined as

$$
\mathcal { L } _ { \mathrm { L L M } } = - \sum _ { t = 1 } ^ { M } \log p _ { \theta } \left( y _ { t } \mid y _ { < t } , \mathbf { H } _ { \mathrm { i n p u t } } \right) .
$$

2) Two-stage Training: As shown in Fig. 5, OneEEG is trained with a two-stage strategy, including projector alignment and multimodal instruction tuning.

In Stage 1, projector alignment, the EEG encoder, visual module, and LLM backbone are frozen, and only the EEG-to-LLM projection modules are optimized. For MoP, this stage trains the projector bank under hard top-1 assignment, establishing an initial alignment between EEG encoder features and the LLM hidden space.

In Stage 2, multimodal instruction tuning, OneEEG is initialized from the Stage 1 checkpoint. The EEG encoder and visual encoder remain frozen, while the MoP, the visual projector, and the LoRA [52] parameters are updated. This stage further adapts the alignment modules and instructionfollowing components while preserving the pretrained encoders and LLM backbone.

## V. EXPERIMENTS

In this section, we evaluate current MLLMs and OneEEG on our OneEEG-Bench and analyze the effectiveness of our method. Furthermore, we conduct detailed ablation experiments and in-depth analysis.

Benchmark and input setting. We evaluate all models on OneEEG-Bench, which contains basic EEG attribute QA and task-oriented EEG QA. Following the dataset taxonomy in Sec. III-A, we report results under standard-setting and nonstandard-setting groups. In the main evaluation, OneEEG follows the channel-compatible input setting described in Sec. IV-B1. All baseline MLLMs are evaluated with the rendered EEG waveform image and textual question only, since these baselines do not support numerical EEG representations. Compared methods. We compare OneEEG with representative closed-source, open-source, medical, and EEG-specialized MLLMs, as listed in Table II.

Evaluation metrics. Each model output is normalized and matched against the candidate labels defined for the corresponding question. A prediction is counted as correct if it matches the ground-truth label and is marked as invalid if it cannot be mapped to any candidate label. For each task family, we report the overall accuracy over all samples belonging to that family, while Overall applies the same accuracy measure to all evaluation samples in OneEEG-Bench. Macro is the unweighted average of the seven family-level overall accuracies, including Basic, Motor, State, Clinical, Event, Sleep, and Vigilance. Invalid is the proportion of outputs that cannot be mapped to any candidate label.

## A. Main Results

Table II reports the main results on OneEEG-Bench. Our proposed OneEEG achieves the best performance in both

TABLE II  
MAIN EVALUATION RESULTS ON ONEEEG-BENCH. INVALID DENOTES THE INVALID-OUTPUT RATE, WHILE ALL REMAINING COLUMNS REPORT ACCURACY.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Invalid↓ Basic</td><td rowspan="2"></td><td colspan="4">Standard channel setting Motor</td><td colspan="2">Non-standard channel setting</td><td rowspan="2">Overall Macro</td><td rowspan="2"></td></tr><tr><td>State</td><td></td><td>Clinical</td><td>Event</td><td>Sleep</td><td>Vigilance</td></tr><tr><td colspan="10">Closed-source MLLMs</td></tr><tr><td>Gemini-3 Flash [53]</td><td>5.94</td><td>26.77</td><td>41.09</td><td>52.50</td><td>51.50</td><td>52.72</td><td>53.66</td><td>52.97</td><td>47.02</td><td>47.32</td></tr><tr><td>Gemini-3 Pro [26]</td><td>2.62</td><td>38.75</td><td>31.72</td><td>50.67</td><td>51.83</td><td>48.04</td><td>71.25</td><td>69.73</td><td>49.27</td><td>51.71</td></tr><tr><td>GPT-5.5 [25]</td><td>0.02</td><td>46.31</td><td>36.56</td><td>49.67</td><td>52.83</td><td>51.30</td><td>58.71</td><td>75.68</td><td>50.39</td><td>53.01</td></tr><tr><td colspan="10">Open-source MLLMs</td></tr><tr><td>Qwen2.5-VL-7B [27]</td><td>0.00</td><td>29.39</td><td>32.19</td><td>52.00</td><td>50.33</td><td>25.43</td><td>65.33</td><td>84.32</td><td>42.73</td><td>48.43</td></tr><tr><td>Qwen3-VL-8B [28]</td><td>0.00</td><td>32.18</td><td>33.59</td><td>52.00</td><td>48.50</td><td>29.78</td><td>44.43</td><td>45.41</td><td>39.41</td><td>40.84</td></tr><tr><td>InternVL3.5-8B [54]</td><td>0.00</td><td>27.26</td><td>37.03</td><td>48.50</td><td>49.50</td><td>30.54</td><td>62.72</td><td>82.16</td><td>43.22</td><td>48.24</td></tr><tr><td>InternVL3.5-38B [54]</td><td>0.00</td><td>29.23</td><td>32.34</td><td>50.50</td><td>49.50</td><td>30.65</td><td>64.98</td><td>85.41</td><td>43.56</td><td>48.94</td></tr><tr><td>LLaVA-OneVision-2-8B [29]</td><td>0.07</td><td>26.60</td><td>41.56</td><td>50.67</td><td>49.50</td><td>31.85</td><td>53.31</td><td>51.35</td><td>41.74</td><td>43.55</td></tr><tr><td colspan="10">Medical MLLMs</td></tr><tr><td>MedGemma-1.5-4B [31]</td><td>4.43</td><td>28.41</td><td>31.41</td><td>45.00</td><td>50.33</td><td>25.33</td><td>63.76</td><td>80.54</td><td>41.04</td><td>46.40</td></tr><tr><td>Lingshu-7B [30] Lingshu-32B [30]</td><td>0.07 0.00</td><td>29.72 28.24</td><td>42.03</td><td>52.00</td><td>51.17</td><td>38.48</td><td>54.36 64.11</td><td>81.08 84.86</td><td>45.66 46.44</td><td>49.83 50.96</td></tr><tr><td></td><td></td><td></td><td>34.69</td><td>52.00 EEG-specialized Models</td><td>52.50</td><td>40.33</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td></tr><tr><td>CerebraGloss [12]</td><td>4.70</td><td>49.26</td><td>24.38</td><td>50.83</td><td>49.00</td><td>36.09</td><td>61.50</td><td>85.95</td><td>46.00</td><td>51.00</td></tr><tr><td>OneEEG</td><td>0.05</td><td>72.74</td><td>46.09</td><td>56.83</td><td>66.67</td><td>67.72</td><td>76.83</td><td>90.81</td><td>65.67</td><td>68.24</td></tr></table>

Overall and Macro scores, reaching 65.67% and 68.24%, respectively. Against the strongest baseline, GPT-5.5, OneEEG improves Overall accuracy by 15.28 and Macro accuracy by 15.23. Compared with the Qwen2.5-VL backbone used by OneEEG, OneEEG brings substantial gains on Basic, Clinical, Event, Sleep, and Vigilance tasks, suggesting that EEG-specific training and signal-aware modeling are especially effective for clinically relevant EEG patterns and lowlevel physiological attributes, while gains on the Motor and State tasks are relatively smaller. In contrast to physiological attributes and clinical/event-related tasks, motor imagery and state-related EEG tasks require distinguishing more finegrained EEG patterns, which still remain challenging. These results show that OneEEG provides a stronger unified solution for multi-task EEG question answering, while also indicating room for improving signal representations on fine-grained motor and state tasks.

The baseline results further reveal the difficulty of OneEEG-Bench. Closed-source models generally perform better than open-source models. For instance, GPT-5.5 reaches 50.39% Overall accuracy, while the best open-source baseline, InternVL3.5-38B, obtains 43.56%. In addition, larger models tend to perform better: InternVL3.5-38B outperforms InternVL3.5-8B, and Lingshu-32B is better than Lingshu-7B, which indicates the benefit of model capacity in improving performance. As a specialized EEG-trained baseline, CerebraGloss performs competitively on several tasks, but its training data mainly cover sleep and clinical datasets (HMC, DREAMS, and TUH-series datasets), which restricts its performance under the broader task and dataset coverage of OneEEG-Bench.

![](images/c0b4317a5ef72af495bce2c8a74c887a2c4d425907b63943868009208ef30330.jpg)  
Fig. 6. Per-dataset evaluation results on OneEEG-Bench. Each axis reports the accuracy computed over all samples from the corresponding dataset, with larger values indicating better performance. For readability, only representative baselines are shown in the radar plot.

Fig. 6 further provides a dataset-level view of the results. OneEEG shows more consistent performance across EEG datasets, indicating that its advantage is not driven by a single dataset or task family. At the same time, the gains are not uniform across all datasets, suggesting that the frozen EEG encoder still provides imperfect and task-dependent signal features.

TABLE III  
ABLATION STUDY ON THE CORE DESIGN CHOICES OF ONEEEG UNDER THE PHASE 1 SETTING. ALL METRICS FOLLOW THE SAME DEFINITIONS AS IN THE MAIN EVALUATION, RESTRICTED TO THE PHASE 1 EVALUATION SET.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">Projector</td><td rowspan="2">Route Strategy</td><td rowspan="2">Visual Prefix</td><td colspan="5">Task Family</td><td rowspan="2">Overall Macro</td><td rowspan="2"></td></tr><tr><td>Basic</td><td>Motor</td><td>State</td><td>Clinical</td><td>Event</td></tr><tr><td>Shared</td><td>Shared</td><td>一</td><td>一</td><td>67.96</td><td>44.69</td><td>47.83</td><td>63.00</td><td>59.13</td><td>56.42</td><td>56.52</td></tr><tr><td>Shared + Prefix</td><td>Shared</td><td></td><td>√</td><td>67.78</td><td>41.88</td><td>49.33</td><td>65.17</td><td>61.09</td><td>57.06</td><td>57.05</td></tr><tr><td>MoP-Free</td><td>MoP</td><td>Free</td><td></td><td>67.78</td><td>39.84</td><td>43.33</td><td>56.83</td><td>58.48</td><td>53.33</td><td>53.25</td></tr><tr><td>MoP-Free + Prefix</td><td>MoP</td><td>Free</td><td>V</td><td>69.63</td><td>43.91</td><td>50.83</td><td>65.50</td><td>58.59</td><td>57.39</td><td>57.69</td></tr><tr><td>MoP</td><td>MoP</td><td>Fixed</td><td>一</td><td>71.11</td><td>39.22</td><td>43.00</td><td>64.17</td><td>55.00</td><td>54.06</td><td>54.50</td></tr><tr><td>OneEEG</td><td>MoP</td><td>Fixed</td><td>√</td><td>70.19</td><td>44.53</td><td>53.33</td><td>69.17</td><td>60.65</td><td>59.30</td><td>59.57</td></tr></table>

TABLE IV

ABLATION ON EEG VISUAL REPRESENTATIONS. WE COMPARE WAVEFORM IMAGES, SPECTROGRAM IMAGES, AND THEIR COMBINATION USING GPT-5.5 AND QWEN2.5-VL-7B. ALL METRICS FOLLOW THE SAME DEFINITIONS AS IN THE MAIN EVALUATION.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Input</td><td rowspan="2">Basic</td><td colspan="4">Standard channel setting</td><td colspan="2">Non-standard channel setting</td><td rowspan="2">Overall Macro</td><td rowspan="2"></td></tr><tr><td>Motor</td><td>State</td><td>Clinical</td><td>Event</td><td>Sleep</td><td>Vigilance</td></tr><tr><td rowspan="3">GPT-5.5</td><td>Waveform</td><td>46.31</td><td>36.56</td><td>49.67</td><td>52.83</td><td>51.30</td><td>58.71</td><td>75.68</td><td>50.39</td><td>53.01</td></tr><tr><td>Spectrogram</td><td>28.41</td><td>36.72</td><td>49.33</td><td>50.00</td><td>32.83</td><td>60.10</td><td>55.68</td><td>42.49</td><td>44.72</td></tr><tr><td>Waveform+Spectrogram</td><td>45.65</td><td>33.44</td><td>50.67</td><td>52.00</td><td>50.65</td><td>55.57</td><td>75.14</td><td>49.22</td><td>51.87</td></tr><tr><td rowspan="3">Qwen2.5-VL-7B</td><td>Waveform</td><td>29.39</td><td>32.19</td><td>52.00</td><td>50.33</td><td>25.43</td><td>65.33</td><td>84.32</td><td>42.73</td><td>48.43</td></tr><tr><td>Spectrogram</td><td>26.60</td><td>31.09</td><td>41.17</td><td>49.33</td><td>20.98</td><td>63.76</td><td>76.22</td><td>38.86</td><td>44.16</td></tr><tr><td>Waveform+Spectrogram</td><td>27.59</td><td>29.69</td><td>51.33</td><td>49.83</td><td>24.35</td><td>67.42</td><td>83.24</td><td>41.91</td><td>47.64</td></tr></table>

(a) Frozen EEG encoder features  
![](images/24e300c637e084d052d67261060c79b3995447566673f2f1b633635c5c246f85.jpg)

(b) Shared-projected EEG features  
![](images/5152d5b989ed6040d91cdb2a833fef97fb0d46d8c407ddb5e894060553ddef21.jpg)

(c) MoP-projected EEG features  
![](images/23d2c9b5f063129754b70d4f904f4df2c1047969f82825894afaf54bd97eb4c5.jpg)  
Fig. 7. t-SNE visualization of EEG representations before projection (a) and after projection (b), (c). Each point denotes an evaluation sample, and colors indicate task families. MoP produces more structured family-level grouping after projection.

## B. Ablation Study

Table III studies the core design choices of OneEEG. We perform the ablation in Phase 1 to isolate the EEG-to-LLM projection design, since this stage freezes the LLM backbone and visual modules and only optimizes the EEG projection components, which allows a cleaner comparison of the projector structure, routing strategy, and prefix-augmented visual conditioning. We compare the full OneEEG with several variants, including a shared-projector baseline, a free-routing MoP variant where basic questions are routed together with task questions, and variants with or without visual prefix conditioning. In Table III, “Free” denotes routing basic questions together with task questions, while “Fixed” assigns basic questions to the basic projector. Macro denotes the unweighted average over Basic, Motor, State, Clinical, and Event families.

The results show that MoP is the key component of OneEEG. With prefix-augmented visual conditioning, the full OneEEG achieves the best Overall and Macro scores, reaching 59.30% and 59.57%, respectively. It outperforms both Shared w/ Prefix and MoP-Free w/ Prefix in these two aggregate metrics, indicating that neither a single shared projector nor unconstrained routing can fully replace the task-aware MoP design. Prefix-augmented visual conditioning further improves MoP: adding an additional visual prefix to MoP improves overall accuracy from 54.06% to 59.30% and macro accuracy from 54.50% to 59.57%. However, the prefix alone is insufficient, as Shared w/ Prefix is still inferior to OneEEG in Overall and Macro. These results suggest that MoP serves as the primary mechanism for task-aware EEG-to-LLM projection, while the visual prefix provides useful context.

![](images/07bac8efce8af3ecf5afb1850673e3c3f832d915ffd85dbe27a87ac74dd6f42e.jpg)  
Fig. 8. Comparison of MoP router variants under the Phase 1 setting. Hard top-1 routing and gated top-1 routing with noise achieve comparable Overall and Macro performance.

## C. In-Depth Analysis

Beyond the main results and ablation study, we further conduct additional analyses to better understand the behavior of OneEEG, including the effect of visual EEG representations, feature-space visualization, routing-strategy comparison, and projector-usage analysis.

1) Effect of Visual Representations: We first analyze how different EEG visual representations affect the performance of general MLLMs. As shown in Table IV, we compare waveform images, spectrogram images, and their combination using GPT-5.5 and Qwen2.5-VL-7B without additional training. Waveform images provide the most stable performance across the two models. For GPT-5.5, waveform input achieves the best Overall and Macro scores, reaching 50.39% and 53.01%, respectively. For Qwen2.5-VL-7B, waveform input also obtains the best Overall and Macro scores, with 42.73% and 48.43%, respectively. In contrast, spectrogram-only input leads to clear performance degradation, especially on Vigilance, where GPT-5.5 drops from 75.68% to 55.68%, and Qwen2.5-VL-7B drops from 84.32% to 76.22%. Combining waveform and spectrogram does not consistently improve over waveform-only input. These results suggest that waveform images are the most reliable visual EEG representation for current MLLMs, and we therefore use waveform images as the default visual input in our training process.

2) Feature Visualization: We further visualize the EEG features before and after the EEG-to-LLM projection using t-SNE. As shown in Fig. 7, the frozen EEG encoder features before projection are highly mixed across task families, indicating that the pretrained EEG encoder alone does not directly organize features according to the downstream QA taxonomy. The shared-projector baseline slightly reshapes the feature space, but different task families remain largely entangled. In contrast, MoP-projected features exhibit clearer familylevel structures, with Basic, Clinical, Motor, State, and Event samples forming more distinguishable regions. This visualization suggests that MoP helps organize EEG features into a more task-aware projected space for the LLM. The remaining overlap between some families also indicates that fine-grained EEG representation alignment remains challenging.

![](images/ff32302b478bb5efa9c959ff326f61e5704c8d222085004d90d8a3353af8e1d4.jpg)  
Fig. 9. MoP router projector usage under hard top-1 routing. Rows denote projector slots and columns denote datasets, with each cell showing the selection ratio.

3) Effect of Routing Strategy: We further compare two routing strategies of MoP under the Phase 1 setting. Besides the hard top-1 router used in the main configuration, we also evaluate a gated top-1 router with routing noise [55], which follows a commonly used soft gating design for encouraging routing exploration during training. As shown in Fig. 8, the two variants achieve very similar aggregate performance. Hard top-1 routing obtains 59.30% Overall and 59.57% Macro, while gated top-1 routing with noise obtains 59.24% Overall and 59.15% Macro, only 0.06 and 0.42 percentage points lower, respectively. This result suggests that the gain of MoP mainly comes from the multi-projector formulation rather than a specific router realization. By assigning different EEG task families to separate projection paths, MoP reduces the interference that may arise when heterogeneous EEG features are forced to share a single EEG-to-LLM projector. The comparable performance of the two router variants further indicates that sparse task-aware projection is a robust design choice for EEG-to-LLM alignment.

4) Router Projector Usage: We further analyze the projector usage induced by the hard routing configuration of MoP. Since each sample activates one projection path, the usage map in Fig. 9 provides a direct view of how samples are mapped to projector slots. The assignments show clear task-family preferences rather than uniform usage. Specifically, Clinical and Event tasks show relatively stable and distinct projector usage, while Motor and State datasets are more likely to share projector slots. This suggests that Clinical and Event tasks are more clearly separated in the current routing space, whereas Motor and State remain harder to distinguish. The observation is also consistent with the t-SNE visualization, where Motor and State representations remain partially overlapped. Overall, the usage pattern supports the interpretability of hard routing and indicates that more discriminative signal representations may further improve motor- and state-related EEG understanding.

## VI. CONCLUSION

In this work, we proposed OneEEG, a unified EEG-MLLM framework that jointly models numerical EEG representations, visual EEG representations, and text instructions for multitask EEG question answering. OneEEG introduces a MoP module to align heterogeneous EEG signal features with the LLM hidden space. To support training and evaluation, we constructed OneEEG-Instruct and OneEEG-Bench, covering diverse EEG tasks, channel settings, and closed-set QA formats. Extensive experiments show that OneEEG consistently outperforms general, medical, and EEG-specialized MLLM baselines across heterogeneous EEG tasks. Further analyses demonstrate the effectiveness of our proposed OneEEG.

## REFERENCES

[1] U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, and H. Adeli, “Deep convolutional neural network for the automated detection and diagnosis of seizure using eeg signals,” Computers in biology and medicine, vol. 100, pp. 270–278, 2018.

[2] H.-T. Lee, M. Shim, X. Liu, H.-R. Cheon, S.-G. Kim, C.-H. Han, and H.- J. Hwang, “A review of hybrid eeg-based multimodal human–computer interfaces using deep learning: applications, advances, and challenges,” Biomedical Engineering Letters, vol. 15, no. 4, pp. 587–618, 2025.

[3] R. Fratangelo, F. Lolli, M. Scarpino, and A. Grippo, “Point-of-care electroencephalography in acute neurological care: A narrative review,” Neurology International, vol. 17, no. 4, p. 48, 2025.

[4] S. K. Satapathy, T. Patel, M. Modi, H. Davra, B. Brahma, S. K. Tripathy, and A. K. Bhoi, “Deep learning-based sleep stage classification: A systematic review,” Journal of The Institution of Engineers (India): Series B, vol. 106, no. 6, pp. 2087–2109, 2025.

[5] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces,” Journal of neural engineering, vol. 15, no. 5, p. 056013, 2018.

[6] J. Yedukondalu, K. Sunkara, V. Radhika, S. Kondaveeti, M. Anumothu, and Y. Murali Krishna, “Cognitive load detection through eeg lead wise feature optimization and ensemble classification,” Scientific Reports, vol. 15, no. 1, p. 842, 2025.

[7] Y. Wang, L. Zhang, and Y. Zhang, “Band-level adaptive fusion network for cross-subject eeg emotion recognition,” IEEE Transactions on Instrumentation and Measurement, 2025.

[8] K. Liu, T. Yang, Z. Yu, W. Yi, H. Yu, G. Wang, and W. Wu, “Msvtnet: Multi-scale vision transformer neural network for eeg-based motor imagery decoding,” IEEE Journal ofBiomedical and Health Informatics, vol. 28, no. 12, pp. 7126–7137, 2024.

[9] C. Liu, Y. Deng, Y. Wu, R. Yang, Z. Wang, L. Zhang, S. Chen, T. Zhang, Y. Liu, Y. Ding et al., “Graph neural networks in eeg-based emotion recognition: a survey,” arXiv preprint arXiv:2402.01138, 2024.

[10] M. A. Ozdemir, M. Degirmenci, E. Izci, and A. Akan, “Eeg-based emotion recognition with deep convolutional neural networks,” Biomedical Engineering/Biomedizinische Technik, vol. 66, no. 1, pp. 43–57, 2021.

[11] Z. Zeng, Z. Cai, Y. Cai, X. Wang, J. Chen, R. Wang, Y. Liu, S. Cai, B. Wang, Z. Zhang et al., “Wavemind: Towards a conversational eeg foundation model aligned to textual and visual modalities,” arXiv preprint arXiv:2510.00032, 2025.

[12] W. Gu, T. Luo, Q. Zhang, M. Ye, X. Shen, W. Chen, Y. Li, Y. Zhang, J. Hong, B.-l. Lu, and W.-L. Zheng, “Cerebragloss: Instruction-tuning a large vision-language model for fine-grained clinical eeg interpretation,” in The Fourteenth International Conference on Learning Representations, 2026.

[13] F. Ma, H. Lin, Y. Xie, H. Ren, X. Shen, W. Ding, and Q. Tian, “Eˆ 2- llm: Bridging neural signals and interpretable affective analysis,” arXiv preprint arXiv:2601.07877, 2026.

[14] X. Qiu, G. Ma, H. Wang, C. Zhan, X. Tan, and S. Li, “Eeg-vlm: A hierarchical vision-language model with multi-level feature alignment and visually enhanced language-guided reasoning for eeg image-based sleep stage prediction,” IEEE Journal of Biomedical and Health Informatics, 2026.

[15] M. Jin, C. Du, H. He, T. Cai, and J. Li, “PGCN: Pyramidal graph convolutional network for EEG emotion recognition,” IEEE Transactions on Multimedia, vol. 26, pp. 9070–9082, 2024.

[16] Y. Pi, M. Li, X. Zhang, W. Wu, X. Qi, J. Yue, G. Bian, and W. Wu, “Enhancing generalization for EEG-based emotion recognition models with the personality cross-attention mechanism,” IEEE Transactions on Multimedia, pp. 1–14, 2026.

[17] W.-B. Jiang, L. Zhao, and B.-L. Lu, “Large brain model for learning generic representations with tremendous eeg data in bci,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 16 405–16 426.

[18] G. Wang, W. Liu, Y. He, C. Xu, L. Ma, and H. Li, “Eegpt: Pretrained transformer for universal and reliable representation of eeg signals,” Advances in Neural Information Processing Systems, vol. 37, pp. 39 249– 39 280, 2024.

[19] D. Zhang, Z. Yuan, Y. Yang, J. Chen, J. Wang, and Y. Li, “Brant: Foundation model for intracranial neural signal,” Advances in Neural Information Processing Systems, vol. 36, pp. 26 304–26 321, 2023.

[20] J. Wang, S. Zhao, Z. Luo, Y. Zhou, H. Jiang, S. Li, T. Li, and G. Pan, “Cbramod: A criss-cross brain foundation model for eeg decoding,” in International conference on learning representations, vol. 2025, 2025, pp. 75 310–75 346.

[21] Q. Xiao, Z. Cui, C. Zhang, S. Chen, W. Wu, A. Thwaites, A. Woolgar, B. Zhou, and C. Zhang, “Brainomni: A brain foundation model for unified eeg and meg signals,” Advances in Neural Information Processing Systems, vol. 38, pp. 41 179–41 212, 2026.

[22] C. Yang, M. Westover, and J. Sun, “Biot: Biosignal transformer for cross-data learning in the wild,” Advances in Neural Information Processing Systems, vol. 36, pp. 78 240–78 260, 2023.

[23] W. Cui, W. Jeong, P. Tholke, T. Medani, K. Jerbi, A. A. Joshi, and R. M.¨ Leahy, “Neuro-gpt: Towards a foundation model for eeg,” in 2024 IEEE International Symposium on Biomedical Imaging (ISBI). IEEE, 2024, pp. 1–5.

[24] W.-B. Jiang, Y. Wang, B.-L. Lu, and D. Li, “Neurolm: A universal multitask foundation model for bridging the gap between language and eeg signals,” arXiv preprint arXiv:2409.00101, 2024.

[25] OpenAI, “GPT-5.5 Thinking,” https://chatgpt.com/, 2026, large language model, accessed May 26, 2026.

[26] Google DeepMind, “Gemini 3 Pro,” https://deepmind.google/models/ gemini/pro/, 2026, large language model, accessed May 26, 2026.

[27] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2502.13923

[28] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[29] X. An, Y. Xie, F. Tang, Y. Yan, H. Tan, D. Zhu, C. Chen, X. Zhao, B. Qin, K. Yang, Y. Shen, Y. Zhang, K. Zhang, W. Zhang, Z. Cheng, N. Zhang, C. Wu, C. Ge, Z. Ran, D. Song, C. Li, S. Feng, M. Hu, Z. Chen, J. Niu, B. Li, Z. Feng, Z. Liu, Z. Ge, and J. Deng, “Llava-onevision-2: Towards next-generation perceptual intelligence,” 2026. [Online]. Available: https://arxiv.org/abs/2605.25979

[30] W. Xu, H. P. Chan, L. Li, M. Aljunied, R. Yuan, J. Wang, C. Xiao, G. Chen, C. Liu, Z. Li et al., “Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning,” arXiv preprint arXiv:2506.07044, 2025.

[31] A. Sellergren, C. Gao, F. Mahvar, T. Kohlberger, F. Jamil, M. Traverse, A. Tono, B. Sadjad, L. Yang, C. Lau et al., “Medgemma 1.5 technical report,” arXiv preprint arXiv:2604.05081, 2026.

[32] Y. Wang, S. Agarwal, S. Mukherjee, X. Liu, J. Gao, A. Hassan, and J. Gao, “Adamix: Mixture-of-adaptations for parameter-efficient model tuning,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 5744–5760.

[33] S. Chen, Z. Jie, and L. Ma, “Llava-mole: Sparse mixture of lora experts for mitigating data conflicts in instruction finetuning mllms,” arXiv preprint arXiv:2401.16160, 2024.

[34] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus, “St-moe: Designing stable and transferable sparse expert models,” arXiv preprint arXiv:2202.08906, 2022.

[35] L. Zhang, S. Huang, and W. Liu, “Enhancing mixture-of-experts by leveraging attention for fine-grained recognition,” IEEE Transactions on Multimedia, vol. 24, pp. 4409–4421, 2022.

[36] Y. Gui, M. Chen, Y. Su, G. Luo, and Y. Yang, “Eegmamba: Bidirectional state space model with mixture of experts for eeg multi-task classification,” arXiv preprint arXiv:2407.20254, 2024.

[37] F. Ma, M. Jiang, L. Qian, Z. Gu, and H. Xu, “Brainmoe: Towards universal eeg foundation models with channel-wise mixture-of-experts.”

[38] X. Gao, D. Wang, and Y. Zhao, “Eegmoe: A domain-decoupled mixtureof-experts model for self-supervised eeg representation learning,” IEEE Transactions on Neural Networks and Learning Systems, 2026.

[39] G. H. Klem, “The ten-twenty electrode system of the international federation. the international federation of clinical neurophysiology,” Electroencephalogr. Clin. Neurophysiol. Suppl., vol. 52, pp. 3–6, 1999.

[40] C. Brunner, R. Leeb, G. R. Muller-Putz, A. Schl¨ ogl, and G. Pfurtscheller,¨ “BCI Competition 2008 – Graz Data Set A,” https://www.bbci.de/ competition/iv/desc 2a.pdf, 2008, BCI Competition IV, Dataset 2a.

[41] G. Schalk, “EEG Motor Movement/Imagery Dataset,” PhysioNet, Sep. 2009, version 1.0.0. [Online]. Available: https://doi.org/10.13026/ C28G6P

[42] I. Zyma, S. Tukaev, I. Seleznov, K. Kiyono, A. Popov, M. Chernykh, and O. Shpenkov, “Electroencephalograms during mental arithmetic task performance,” Data, vol. 4, no. 1, p. 14, 2019.

[43] W. Zheng, W. Liu, Y. Lu, B. Lu, and A. Cichocki, “Emotionmeter: A multimodal framework for recognizing human emotions,” IEEE Transactions on Cybernetics, pp. 1–13, 2018.

[44] Neural Engineering Data Consortium, “Temple University Hospital EEG Corpus,” https://isip.piconepress.com/projects/nedc/html/tuh eeg/, 2026, accessed May 26, 2026.

[45] I. Obeid and J. Picone, “The Temple University Hospital EEG Data Corpus,” Frontiers in Neuroscience, vol. 10, p. 196, 2016.

[46] A. Hamid, K. Gagliano, S. Rahman, S. Tobochnik, I. Obeid, and J. Picone, “The Temple University Artifact Corpus: An Annotated Corpus of EEG Artifacts,” in Proceedings of the IEEE Signal Processing in Medicine and Biology Symposium, 2020. [Online]. Available: https: //isip.piconepress.com/conferences/ieee spmb/2020/papers/p01 09.pdf

[47] V. Shah, E. Von Weltin, S. Lopez, J. R. McHugh, L. Veloso, M. Golmohammadi, I. Obeid, and J. Picone, “The temple university hospital seizure detection corpus,” Frontiers in neuroinformatics, vol. 12, p. 83, 2018.

[48] B. Kemp, A. H. Zwinderman, B. Tuk, H. A. Kamphuisen, and J. J. Oberye, “Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the eeg,” IEEE Transactions on Biomedical Engineering, vol. 47, no. 9, pp. 1185–1194, 2000.

[49] S. Devuyst, “The dreams databases and assessment algorithm,” Jan. 2005. [Online]. Available: https://doi.org/10.5281/zenodo.2650142

[50] W.-L. Zheng and B.-L. Lu, “A multimodal approach to estimating vigilance using eeg and forehead eog,” Journal of Neural Engineering, vol. 14, no. 2, p. 026017, 2017. [Online]. Available: http://stacks.iop. org/1741-2552/14/i=2/a=026017

[51] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[52] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9

[53] Google DeepMind, “Gemini 3 Flash,” https://deepmind.google/models/ model-cards/, 2025, model card, accessed May 26, 2026.

[54] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.

[55] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538, 2017.