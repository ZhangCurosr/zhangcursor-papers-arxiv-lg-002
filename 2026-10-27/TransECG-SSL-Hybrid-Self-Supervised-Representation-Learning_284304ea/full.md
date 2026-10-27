# TransECG-SSL: Hybrid Self-Supervised Representation Learning for ECG Arrhythmia Classification

Ahtisham Ayyub, Student Member, IEEE, Fasih Haider, Senior Member, IEEE, Muhammad Rehan Usman, Member, IEEE, Saturnino Luz, Member, IEEE, Christos Politis, Senior Member, IEEE, and Muhammad Arslan Usman, Member, IEEE

Abstract— Cross-dataset generalisation remains a major challenge for automated electrocardiogram (ECG) arrhythmia classification because of diferences in patient populations, acquisition protocols, annotation practices, and class distributions. Learning representations that remain informative across independent datasets could reduce reliance on datasetspecific labelled data. We propose TransECG-SSL, a selfsupervised framework for transferable heartbeat representation learning from unlabelled ECG data. A Transformer encoder is pretrained using a hybrid objective combining masked signal reconstruction and contrastive learning, then frozen and evaluated by linear probing under patient-disjoint evaluation. Experiments across three public ECG databases and seven source– target settings show that Hybrid SSL consistently outperforms randomly initialised Transformers and achieves higher balanced accuracy than architecture-matched source-supervised pretraining. Reconstruction-only SSL remains competitive with Hybrid SSL, whereas contrastive-only pretraining performs substantially worse, highlighting the importance of masked reconstruction for transferability. Cross-dataset transfer nevertheless reduces balanced accuracy relative to target-specific self-supervised pretraining, while minority arrhythmia classes remain challenging under severe class imbalance. These findings demonstrate the potential of label-eficient self-supervised learning for cross-dataset ECG representation transfer while highlighting persistent limitations from domain shift and class imbalance.

Index Terms— Arrhythmia classification, Cross-dataset generalisation, Electrocardiogram (ECG), Self-supervised learning, Transfer learning, Transformer, Representation learning, Domain shift

## I. INTRODUCTION

ume of ECG data generated in clinical, ambulatory and wearable-monitoring settings has motivated substantial research on automated arrhythmia classification. Although Deep-learning (DL) methods can learn discriminative representations directly from ECG signals; reported performance remains dificult to compare because studies vary substantially in datasets, preprocessing, architectures, evaluation protocols, and metrics [1], [2]. Reviews also report a high dependency on MIT-BIH, a greater use of intra-patient rather than inter-patient evaluation, and less favourable class-sensitive metrics than headline accuracy [3]. These concerns reinforce the need for patientindependent evaluation and careful interpretation of high reported performance [4].

Heartbeat-level evaluation is significantly afected by patient overlap because individual patients contribute many correlated beats; patient-disjoint partitioning therefore provides a stricter test of generalisation. Severe class imbalance adds a second dificulty, since normal beats dominate many arrhythmia datasets and minority classes may be poorly recognised even when overall accuracy is high [5].

Another challenge is the dependence of supervised ECG models on time-consuming and expensive expert-labelled data. Self-supervised learning (SSL) ofers a way to exploit unlabelled ECG signals for representation learning. ECG SSL has been investigated through contrastive approaches [6], broader self-supervised representation learning [7], masked reconstruction [8], out-of-distribution evaluation [9], and larger hybrid or foundation-model frameworks [10]. These strands motivate a controlled comparison of reconstructive and contrastive objectives for transferable heartbeat representations.

Transfer learning can reduce dependence on targetdomain labels [11], [12], but transfer across independent ECG databases is complicated by diferences in patient populations, acquisition conditions, annotation practices, and class distributions; reported performance can also depend materially on partitioning choices [13]. Recent approaches address such shifts through domain adaptation [14], domain generalisation [15], and domaininvariant or multi-dataset learning [16], [17]. However, these approaches difer in the information available during training, for instance transfer learning typically uses labelled target samples for fine-tuning, domain adaptation exploits unlabelled target data to align source and target representations, whereas domain generalisation and domain-invariant learning aim to learn representations that remain robust to unseen target distributions without target-specific adaptation. These observations motivate a narrower representation-transfer question: whether an ECG encoder pretrained on one arrhythmia database can remain fixed and provide useful heartbeat representations on an independent target database when only a lightweight target classifier is fitted. Such an evaluation difers from target-domain adaptation because the source-pretrained encoder is not updated using target data, and it difers from conventional transfer learning in which the pretrained encoder is fine-tuned on the target task. It also requires architecture-matched comparisons between selfsupervised objectives so that any apparent advantage is not attributable simply to diferences in encoder capacity or downstream evaluation. To investigate this question, this study proposes TransECG-SSL, a hybrid selfsupervised framework for ECG heartbeat representation learning.

The main contributions of this study are: (i) TransECG-SSL, a Transformer-based framework combining masked reconstruction and contrastive learning to learn transferable heartbeat representations from unlabelled ECG data; (ii) systematic evaluation of frozen representation transfer across seven within- and cross-dataset settings spanning MIT-BIH, INCART, and SVDB under patient-disjoint evaluation; (iii) architecture-matched comparison with reconstruction-only, contrastive-only, source-supervised, and random-encoder baselines under a common linearprobing protocol; and (iv) analysis of cross-dataset generalisation and class-specific performance, demonstrating transferable self-supervised representations while identifying domain shift and minority-class recognition as persistent limitations.

## II. RELATED WORK

## A. Patient-Independent Arrhythmia Classification and Class Imbalance

Patient-independent evaluation has been adopted in several heartbeat classification studies to prevent beats from same individual contributing to both training and testing partitions. Existing inter-patient approaches include single-lead classification, dual-attention networks, residual convolutional architectures, weighted classifiers, symbolic representations, and morphological methods [18]–[23]. Reviews nevertheless indicate that inter-patient evaluation remains less common and generally more demanding than intra-patient evaluation [3], [4].

Patient variability is closely coupled with severe class imbalance. In commonly used heartbeat datasets, normal beats substantially outnumber several abnormal classes, creating conditions in which optimisation can favour majority-class recognition. HCTNet, for example, explicitly addresses the combination of inter-patient variability and imbalanced arrhythmia data [5]. Other studies have approached imbalance through adversarial data synthesis, augmentation, adaptive training, class-aware weighting, or specialised rare-beat learning [24]–[26]. Yuan et al. further considered imbalance jointly with domain adaptation, using target-oriented augmentation and class-aware mechanisms to address changes in both class distribution and source–target domain characteristics [27]. These studies demonstrate that minority-class performance cannot be inferred reliably from overall accuracy alone and motivate class-sensitive metrics and classwise analysis.

Most of this literature, however, focuses on improving the final supervised classifier within a particular dataset or evaluation protocol. Attention mechanisms, class weighting, synthetic data generation, and adaptive sampling may improve minority-class decision boundaries, but they do not directly determine whether the underlying representation itself remains transferable across independent databases. The present study therefore treats patientdisjoint partitioning and class imbalance primarily as constraints on representation-transfer evaluation rather than introducing a specialised imbalance-correction algorithm.

## B. Self-Supervised ECG Representation Learning

ECG SSL spans contrastive, reconstructive, and hybrid approaches. Contrastive methods include CLECG [6] and broader ECG-specific self-supervised pretraining [7], with recent extensions exploring multimodal objectives, physiology-aware augmentations, temporal–spatial learning, and multi-lead classification [28]–[31]. RhythmX additionally combines contrastive representation learning with heartbeat classification and post-hoc interpretability [32].

Masked reconstruction provides a complementary learning signal. MaeFE demonstrated ECG-specific maskedautoencoder pretraining through transfer, linear probing, and fine-tuning [8], while subsequent work has combined masked representations with clustering or discriminative objectives [33] and Transformer-based downstream evaluation [34]. These studies indicate that reconstruction alone can provide a strong ECG representation-learning signal.

Hybrid approaches, on the other hand, combine reconstruction with contrastive alignment, physiological priors, multimodal learning, or large-scale pretraining [10], [29], [33]; ECG-FM, for example, combines masked reconstruction and contrastive learning [10]. Because contrastive performance also depends on view construction and augmentation [29], [35], controlled objective-level comparisons are important. Hybrid SSL, reconstruction-only SSL, and contrastive-only SSL use the same unlabelled source training partitions and the same Transformer encoder architecture, allowing diferences in frozen downstream representations to be attributed primarily to the pretraining objective. This distinction is important because prior ECG SSL studies difer substantially in dataset, encoder architecture, augmentation policy, downstream classifier, and fine-tuning strategy, making numerical comparisons across publications insuficient for determining whether reconstruction, contrastive learning, or their combination is responsible for improved transfer.

## C. ECG Transfer Learning and Cross-Dataset Generalisation

ECG transfer-learning studies have shown that pretrained representations can be reused across cardiac tasks and can reduce target-label requirements [11]–[13], [36], although conventional transfer learning generally permits target fine-tuning.

Methods addressing ECG domain shift include unsupervised domain adaptation [14], [27], domain-adaptive normalisation [37], domain generalisation [15], domaininvariant representation learning [16], unified multidataset training [17], and distributionally adaptive frameworks [38]. These approaches may use target-domain information, multiple training domains, or explicit adaptation mechanisms. Frozen representation transfer asks a narrower question: whether a source-pretrained encoder can remain unchanged while supporting a lightweight classifier on an independent target dataset.

Existing literature therefore provides substantial evidence for patient-independent ECG evaluation, selfsupervised representation learning, transfer learning, and explicit methods for addressing domain shift. However, these strands have generally been investigated using heterogeneous datasets, label spaces, architectures, optimisation procedures, and assumptions about access to targetdomain data. Cross-publication performance comparisons consequently cannot isolate whether diferences arise from the representation-learning objective, model architecture, patient partitioning, target adaptation, or downstream classifier.

Accordingly, a controlled evaluation is needed in which encoder architecture and downstream probing are held fixed while alternative representation-learning objectives are compared under patient-disjoint transfer. This study addresses that gap through TransECG-SSL and architecture-matched baselines.

## III. METHODOLOGY

## A. Overview of TransECG-SSL

Figure 1 summarises the proposed TransECG-SSL framework. The method is designed to evaluate whether self-supervised ECG representations learned from one arrhythmia database can transfer to independent target databases under patient-disjoint evaluation. The framework follows three main stages. First, a Transformerbased ECG encoder is pretrained using only the training records of a source dataset. The self-supervised objective combines masked ECG reconstruction with contrastive representation learning, allowing the encoder to learn morphology-preserving and augmentation-invariant heartbeat representations without using class labels during pretraining. Second, the pretrained encoder is frozen and used to extract clean heartbeat embeddings separately from the target training and test partitions. Third, a linear probe is trained on embeddings from the target training partition and evaluated on embeddings from the patientdisjoint target test partition.

![](images/b1b87352335d47b9abf0d50a74962e91bfcae58cf5c8886b0869d6f94442d0c5.jpg)  
Fig. 1. Proposed Methodology

This design separates representation learning from supervised classification and avoids direct optimisation of the encoder on the target test set. Within-dataset experiments provide target-matched reference performance, whereas cross-dataset experiments evaluate the transferability of frozen representations across independent ECG databases.

## B. ECG Datasets and Preprocessing

Three publicly available ECG arrhythmia databases were used to evaluate TransECG-SSL: the MIT-BIH Arrhythmia Database [39], the St Petersburg INCART 12-lead Arrhythmia Database [40], and the MIT-BIH Supraventricular Arrhythmia Database (SVDB) [41]. We adopted the five ANSI/AAMI heartbeat superclasses: normal beats (N), supraventricular ectopic beats (S), ventricular ectopic beats (V), fusion beats (F), and unknown or paced beats (Q) [42]. This grouping is also consistent with the widely used heartbeat-classification protocol of de Chazal et al. [43]. Moreover, SVDB was evaluated using the shared N/S/V classes only, because F and Q beats were severely under-represented after annotation mapping and beat segmentation. Unknown or non-beat annotation symbols were excluded rather than being forced into the Q class. Table III summarises the class setup.

All datasets were processed using the same beat-centred preprocessing pipeline to reduce dataset-specific preprocessing bias before self-supervised learning and transfer evaluation. Table I summarises the preprocessing and beat-segmentation settings used throughout all experiments.

## C. Patient-Disjoint Class-Aware Splitting

All experiments used fixed patient-disjoint train/test partitions. Recordings from the same patient were assigned jointly to one partition, preventing patient overlap between training and testing. For SVDB, each recording corresponds to a distinct patient; therefore, patient- and record-level partitioning are equivalent.

TABLE I  
ECG PREPROCESSING AND BEAT-SEGMENTATION PARAMETERS USED IN TRANSECG-SSL.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Input source</td><td>WFDB ECG records and reference beat annotations.</td></tr><tr><td>ECG lead</td><td>First available ECG lead was used con- sistently across datasets.</td></tr><tr><td>Bandpass filtering</td><td>Fourth-order Butterworth bandpass fil- ter from 0.5-40 Hz using zero-phase second-order-section filtering.</td></tr><tr><td>Resampling</td><td>Continuous ECG signals were resampled to 256 Hz before beat extraction. R- peak locations were mapped to the re- sampled signal index space.</td></tr><tr><td>Beat alignment</td><td>Beats were centred on the annotated R- peak.</td></tr><tr><td>Window length</td><td>256 samples per beat.</td></tr><tr><td>Window position</td><td>128 samples before and 128 samples after the R-peak.</td></tr><tr><td>Boundary handling</td><td>Beats were discarded if the complete 256-sample window could not be ex- tracted. No padding was used.</td></tr><tr><td>Normalisation Output representa- tion</td><td>Per-beat z-score normalisation. Fixed-length single-channel ECG beat segments used for SSL pretraining, em- bedding extraction, and linear probing.</td></tr></table>

TABLE II

PATIENT-DISJOINT CLASS-AWARE TRAIN/TEST SPLITS USED IN THE FINAL EXPERIMENTS.
<table><tr><td>Dataset</td><td>Split</td><td>Pts.</td><td>Recs.</td><td>N</td><td>S</td><td>V</td><td>F</td><td>Q</td><td>Total</td></tr><tr><td>MIT</td><td>Train</td><td>31</td><td>31</td><td>56,914</td><td>1,861</td><td>4,802</td><td>435</td><td>6,038</td><td>70,050</td></tr><tr><td></td><td>Test</td><td>16</td><td>17</td><td>33,668</td><td>920</td><td>2,433</td><td>367</td><td>2,132</td><td>39,520</td></tr><tr><td>INC</td><td>Train</td><td>21</td><td>51</td><td>104,015</td><td>1,333</td><td>14,351</td><td>147</td><td>4</td><td>119,850</td></tr><tr><td></td><td>Test</td><td>11</td><td>24</td><td>49,581</td><td>625</td><td>5,655</td><td>72</td><td>2</td><td>55,935</td></tr><tr><td>SVDB</td><td>Train</td><td>52</td><td>52</td><td>108,676</td><td>8,167</td><td>6,656</td><td></td><td></td><td>123,499</td></tr><tr><td></td><td>Test</td><td>26</td><td>26</td><td>53,575</td><td>4,028</td><td>3,282</td><td></td><td></td><td>60,885</td></tr></table>

Note: MIT: MIT-BIH; INC: INCART; Pts.: patients; Recs.: ECG records; Total: total beats. SVDB evaluation was restricted to the N/S/V classes; therefore, F and Q are not reported. Owing to class imbalance, an N-only majority classifier would achieve accuracies of .852, .886, and .880 on the MIT-BIH, INCART, and SVDB test sets, respectively, while its balanced accuracy would remain at chance level (.200 for the five-class tasks and .333 for SVDB).

To improve rare-class coverage, patient-level AAMI class counts were used to construct class-aware splits with an approximately 2:1 train/test patient ratio. Split generation used only patient identifiers and class counts and was completed before model training or evaluation. The resulting partitions, which were fixed across all five seeded runs, are summarised in Table II. INCART retained the extremely sparse Q class to preserve the five-class AAMI setting; its efect on the aggregate metrics was therefore examined separately in the sensitivity analysis reported in Table IX.

## D. Self-Supervised Encoder and Hybrid Objective

TransECG-SSL uses a one-dimensional Transformer encoder to learn fixed-length ECG heartbeat representations. Each input beat $x _ { i } ~ \in ~ \mathbb { R } ^ { T }$ , where $T \ = \ 2 5 6 ,$ is first projected into a token sequence using a convolutional stem. A learnable class token is then prepended to the sequence and sinusoidal positional encoding is added. The sequence is passed through a Transformer encoder, based on the self-attention architecture [44]. The class-token output is projected to a compact embedding vector:

$$
z _ { i } = f _ { \theta } ( x _ { i } ) , \qquad z _ { i } \in \mathbb { R } ^ { p }\tag{1}
$$

where $f _ { \theta }$ is the ECG encoder and $p = 1 2 8$ is the dimension of the embedding.

The encoder was pretrained using a hybrid selfsupervised objective that combines masked ECG reconstruction and contrastive representation learning. For masked reconstruction, a binary temporal mask $m _ { i } \in \mathsf { \Gamma }$ $\{ 0 , 1 \} ^ { T }$ was sampled using contiguous time blocks. The masked input was defined as:

$$
\tilde { x } _ { i } = \left( 1 - m _ { i } \right) \odot x _ { i } + m _ { i } \odot v\tag{2}
$$

where v is the mask replacement value and ⊙ denotes element-wise multiplication. The masked beat was passed through the encoder and decoder:

$$
z _ { i } ^ { m } = f _ { \theta } ( \tilde { x } _ { i } ) , \qquad \hat { x } _ { i } = g _ { \phi } ( z _ { i } ^ { m } )\tag{3}
$$

where $g _ { \phi }$ denotes the reconstruction decoder. The reconstruction loss was computed only over masked positions:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { \sum _ { i } \sum _ { t } m _ { i , t } } \sum _ { i } \sum _ { t = 1 } ^ { T } m _ { i , t } \left( \hat { x } _ { i , t } - x _ { i , t } \right) ^ { 2 }\tag{4}
$$

For the contrastive branch, a weakly augmented view $x _ { i } ^ { + }$ of the same beat was generated using additive Gaussian noise and amplitude scaling. The augmented view was passed through the same encoder:

$$
z _ { i } ^ { + } = f _ { \theta } ( x _ { i } ^ { + } )\tag{5}
$$

The masked-view and augmented-view embeddings were then mapped to a contrastive projection space:

$$
h _ { i } ^ { m } = q _ { \psi } ( z _ { i } ^ { m } ) , \qquad h _ { i } ^ { + } = q _ { \psi } ( z _ { i } ^ { + } )\tag{6}
$$

where $q _ { \psi }$ is the projection head. The contrastive loss used cosine similarity:

$$
\sin ( a , b ) = { \frac { a ^ { \top } b } { \| a \| _ { 2 } \| b \| _ { 2 } } }\tag{7}
$$

For a batch of B beats, the contrastive objective was computed using the normalized temperature-scaled crossentropy loss (NT-Xent) from the SimCLR formulation [35]:

$$
\mathcal { L } _ { \mathrm { c o n } } = - \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { 2 B } \log \frac { \exp { \left( \sin ( h _ { i } , h _ { p ( i ) } ) / \tau \right) } } { \sum _ { j = 1 } ^ { 2 B } \mathrm { \it ~ j \neq i } \exp { \left( \sin ( h _ { i } , h _ { j } ) / \tau \right) } }\tag{8}
$$

where $p ( i )$ indexes the positive counterpart of sample i, and $\tau$ is the temperature parameter.

The final hybrid SSL objective was:

$$
{ \mathcal { L } } _ { \mathrm { S S L } } = \lambda _ { \mathrm { r e c } } { \mathcal { L } } _ { \mathrm { r e c } } + \lambda _ { \mathrm { c o n } } { \mathcal { L } } _ { \mathrm { c o n } }\tag{9}
$$

In the final configuration, $\lambda _ { \mathrm { r e c } } = 1 . 0 , \lambda _ { \mathrm { c o n } } = 0 . 1 0$ , and $\tau = 0 . 2 0$ were used. Pretraining was performed on sourcedataset training records only.

## E. Architecture-Matched Baselines and Supervised References

To separate the efect of the training objective from encoder architecture, additional baselines were implemented using the same Transformer encoder as TransECG-SSL. The shared encoder contains a convolutional stem, class token, four Transformer layers with four attention heads, model dimension 256, feed-forward dimension 1024, dropout 0.1, and a 128-dimensional output embedding. This encoder contains 3,358,592 trainable parameters in all architecture-matched conditions.

The reconstruction-only SSL condition uses the same masking configuration, reconstruction decoder, encoder architecture, source records, and optimisation schedule as Hybrid SSL, but removes the contrastive contribution by setting $\lambda _ { \mathrm { c o n } } ~ = ~ 0$ . It therefore isolates the contribution of adding contrastive representation learning to masked reconstruction.

The contrastive-only SSL condition uses the same Transformer encoder and 64-dimensional contrastive projection head, but removes masking and reconstruction entirely. Two independently generated weakly augmented views of each clean heartbeat are used as the positive pair. Both views use the same Gaussian-noise and amplitudescaling augmentation strengths as Hybrid SSL, and the encoder is optimised solely with the NT-Xent loss using temperature $\tau ~ = ~ 0 . 2 0$ . As in Hybrid SSL, contrastive negatives are formed within each distributed-process batch rather than gathered across processes.

The source-supervised Transformer uses the same encoder but replaces self-supervised pretraining with a temporary linear classification head trained using labelled source training records and ordinary multiclass crossentropy. After source training, the classification head is discarded, the encoder is frozen, and target train/test embeddings are evaluated using exactly the same target linear-probe protocol as the SSL conditions. This provides an architecture-matched comparison between supervised and self-supervised source pretraining.

The random frozen Transformer uses the identical encoder architecture with random initialisation and no encoder pretraining. It is frozen before target embedding extraction and serves as an architecture-matched lowerreference condition. Because no source-domain training occurs, this condition is target-specific rather than sourcespecific.

Finally, a target-supervised end-to-end Transformer was trained from random initialisation using labelled target training records. The same Transformer encoder and a linear classification head were optimised jointly and evaluated directly on the corresponding disjoint target test records. This condition provides a target-supervised classification reference rather than a source-to-target representation-transfer ablation.

Hybrid SSL, reconstruction-only SSL, contrastive-only SSL, the source-supervised Transformer, and the targetsupervised Transformer were trained for 50 epochs using AdamW, a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 4 } ,$ , and global batch size 256. All frozen-representation conditions were evaluated using the same unweighted target linear probe trained for 30 epochs with Adam, learning rate $\bar { 1 } 0 ^ { - 3 }$ , and batch size 512. In these comparisons, no inversefrequency class weighting or weighted sampling was used.

## F. Frozen Representation Transfer with Linear Probing

After self-supervised pretraining, the reconstruction decoder and contrastive projection head were discarded. The pretrained encoder was frozen and used only as a feature extractor. For each target dataset, clean embeddings were extracted separately from the target training and test records:

$$
z _ { i } = f _ { \theta ^ { * } } ( x _ { i } )\tag{10}
$$

where $f _ { \theta ^ { * } }$ denotes the pretrained frozen encoder and $z _ { i } \in$ $\mathbb { R } ^ { 1 2 8 }$ is the resulting heartbeat embedding. No masking or reconstruction operation was applied during embedding extraction.

A linear probe was then trained on the target training embeddings and evaluated on the disjoint target test embeddings:

$$
\hat { y } _ { i } = \mathrm { s o f t m a x } ( W z _ { i } + b )\tag{11}
$$

where W and b are the trainable parameters of the linear classifier. The encoder parameters $\theta ^ { * }$ remained fixed during probe training. The probe was optimised using ordinary cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { p r o b e } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { C } \mathbf { 1 } ( y _ { i } = c ) \log ( \hat { y } _ { i , c } )\tag{12}
$$

In the main analysis, no class-weighted loss or weighted sampler was used because the objective was to evaluate the transferability of the frozen self-supervised representation rather than to optimise minority-class decision thresholds.

Table III summarises the final frozen-transfer evaluation matrix. Three within-target experiments were used as target-matched references, while four cross-dataset experiments assessed representation transfer across independent ECG databases. Evaluated classes for each experiment setting have been placed in ’Evaluated classes’ column.

## G. Evaluation Metrics and Transfer Analysis

Performance was evaluated on the held-out target test records for each source–target setting. Overall performance was assessed using accuracy, balanced accuracy (BA), macro-F1, macro-AUROC, macro-AUPRC,

TABLE III  
FROZEN REPRESENTATION TRANSFER AND LINEAR-PROBE EVALUATION MATRIX USED IN THE FINAL TRANSECG-SSL EXPERIMENTS.
<table><tr><td>Experiment</td><td></td><td>SSL pretraining source</td><td>Frozen embedding extraction</td><td>Evaluated classes</td><td>Purpose</td></tr><tr><td>MIT-BIH BIH</td><td>→ MIT- tition</td><td>MIT-BIH training par-</td><td>MIT-BIH patient-disjoint training and test partitions</td><td>N/S/V/F/Q</td><td>Target-matched reference for BIH.</td></tr><tr><td>INCART CART</td><td>→ IN-</td><td>INCART training par- tition</td><td>INCART patient-disjoint training and test partitions</td><td>N/S/V/F/Q</td><td>Target-matched reference INCART.</td></tr><tr><td>SVDB → SVDB</td><td></td><td>SVDB training parti- tion</td><td>SVDB patient-disjoint training and test par- titions</td><td>N/S/V</td><td>Target-matched reference for SVDB.</td></tr><tr><td>MIT-BIH CART</td><td>→ IN-</td><td>MIT-BIH training par- tition</td><td>INCART patient-disjoint training and test partitions</td><td>N/S/V/F/Q</td><td>Cross-dataset transfer from MIT-BIH to INCART.</td></tr><tr><td>INCART → MIT- BIH</td><td></td><td>INCART training par- tition</td><td>MIT-BIH patient-disjoint training and test partitions</td><td>N/S/V/F/Q</td><td>Cross-dataset transfer from INCART to MIT-BIH.</td></tr><tr><td>MIT-BIH SVDB</td><td>→</td><td>MIT-BIH training par- tition</td><td>SVDB patient-disjoint training and test par- titions</td><td>N/S/V</td><td>Cross-dataset transfer from MIT-BIH to SVDB.</td></tr><tr><td>INCART → SVDB</td><td></td><td>INCART training par- tition</td><td>SVDB patient-disjoint training and test par- titions</td><td>N/S/V</td><td>Cross-dataset transfer from INCART to SVDB.</td></tr></table>

Matthews correlation coeficient (MCC) [45], and Cohen’s kappa [46]. Given the substantial class imbalance in the evaluated datasets, BA and macro-F1 were used as the principal class-balanced metrics. BA represents the macroaverage of class-wise recall, whereas macro-F1 gives equal weight to the F1 score of each class. Macro-AUROC and macro-AUPRC were used to assess one-vs-rest ranking performance across evaluable classes.

Class-specific performance was further examined using sensitivity, positive predictive value (PPV), F1 score, AU-ROC, and AUPRC. Row-normalised confusion matrices were used to identify systematic misclassification patterns, particularly for minority arrhythmia classes. All final experiments were repeated using five random seeds, and results are reported as mean ± sample standard deviation (SD).

To quantify cross-dataset transfer relative to each method’s own within-target performance, targetnormalised transfer gaps were calculated for Hybrid SSL, reconstruction-only SSL, and the source-supervised Transformer. For method m, source dataset $s \mathbf { \Omega } \ne \ t ,$ target dataset t, metric M, and matched seed k, the target-normalised transfer gap was defined as:

$$
G _ { m , s  t } ^ { ( k ) } = M _ { m , s  t } ^ { ( k ) } - M _ { m , t  t } ^ { ( k ) } .\tag{13}
$$

Thus, $G _ { m , s  t } ^ { ( k ) } < 0$ indicates that a method performed worse after cross-dataset transfer than when the same method was trained within the target dataset, whereas a positive value indicates higher cross-dataset performance. MIT-BIH → INCART was referenced to INCART → INCART, INCART → MIT-BIH to MIT-BIH → MIT-BIH, and both SVDB-target transfer routes to SVDB → SVDB. The reference was always taken from the same representation-learning method as the corresponding cross-dataset result.

Transfer gaps were calculated separately for balanced accuracy and macro-F1, the two principal class-balanced metrics, for each of the five matched seeds and then averaged across seeds. The resulting mean gaps were reported alongside the corresponding mean cross-dataset performance to distinguish absolute transfer performance from degradation relative to each method’s own withintarget reference.

## IV. RESULTS

## A. Overall Transfer Performance of Hybrid SSL

Table IV summarises the overall performance of the proposed Hybrid SSL encoder across the seven source–target evaluation routes, comprising three within-target references and four cross-dataset transfers. In all experiments, the encoder was pretrained using unlabelled data from the source training partition only and then frozen before target-dataset embedding extraction. A linear probe was trained on embeddings from the target training partition and evaluated on embeddings from the patient-disjoint target test partition. Therefore, the reported results assess the transferability of the source-pretrained representation rather than end-to-end supervised optimisation on the target test set.

Hybrid SSL showed measurable performance across all seven source–target routes under the patient-disjoint evaluation. The highest overall accuracy was obtained for INCART → SVDB (0.8968 ± 0.0083), followed by MIT-BIH → SVDB (0.8755±0.0147) and MIT-BIH → INCART (0.8737 ± 0.0222). However, accuracy remained substantially higher than balanced accuracy in the INCARTtarget settings. MIT-BIH → INCART achieved balanced accuracy of only $0 . 3 6 6 9 \pm 0 . 0 0 7 3$ , while INCART → IN-CART achieved $0 . 3 8 3 4 \pm 0 . 0 1 4 2$ , indicating a strong influence of the majority classes on overall accuracy. The efect of majority classes has been discussed further in IV-D.

The highest within-target balanced accuracy was obtained for SVDB → SVDB at $0 . 6 4 6 4 \pm 0 . 0 2 1 7$ , with macro-F1 of 0.5647 ± 0.0232. Among the cross-dataset routes, INCART → SVDB achieved the highest balanced accuracy (0.5889 ± 0.0079) and macro-F1 (0.6077 ± 0.0156). INCART → MIT-BIH achieved the highest cross-dataset macro-AUROC at $0 . 8 8 6 1 \pm 0 . 0 1 0 2$ , whereas INCART → SVDB achieved the highest cross-dataset macro-AUPRC at 0.6337±0.0116. Absolute SVDB metric levels should not be compared directly with the five-class MIT-BIH and INCART routes as if task dificulty were identical, because SVDB was evaluated only on the shared $\mathrm { N } / \mathrm { S } / \mathrm { V }$ classes.

TABLE IV  
OVERALL TRANSFER PERFORMANCE OF THE PROPOSED HYBRID SSL ENCODER.
<table><tr><td>Experiment</td><td>Acc.</td><td>BA</td><td></td><td>A-BA gap</td><td>F1</td><td>AUROC</td><td>AUPRC</td><td>MCC</td><td></td><td>κ</td></tr><tr><td> $\mathrm { M I T }  \mathrm { M I T }$ </td><td> $. 8 4 8 \pm . 0 0 8 . 5 8 2 \pm . 0 1 2$ </td><td></td><td></td><td>.266</td><td> $. 4 9 7 \pm . 0 1 3$ </td><td> $. 9 0 2 \pm . 0 1 1$ </td><td> $. 5 9 2 \pm . 0 1 4$ </td><td> $. 5 6 2 \pm . 0 1 4$ </td><td></td><td> $. 5 4 3 \pm . 0 1 5$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { I N C }$ </td><td> $. 8 4 5 \pm . 0 3 3 . 3 8 3 \pm . 0 1 4$ </td><td></td><td></td><td>.461</td><td></td><td>.324 ± .017 .833 ± .024</td><td> $. 3 6 7 \pm . 0 2 2$ </td><td> $5 5 3 \pm . 0 5 0$ </td><td></td><td> $. 5 0 3 \pm . 0 6 2$ </td></tr><tr><td> $\mathrm { S V D B } \to \mathrm { S V D B }$ </td><td> $. 8 3 0 \pm . 0 1 4 . 6 4 6 \pm . 0 2 2$ </td><td></td><td></td><td>.184</td><td> $. 5 6 5 \pm . 0 2 3$ </td><td> $. 8 4 6 \pm . 0 1 7$ </td><td> $. 6 3 7 \pm . 0 1 7$ </td><td> $. 4 0 5 \pm . 0 2 9$ </td><td></td><td> $. 3 9 1 \pm . 0 3 1$ </td></tr><tr><td> $\mathrm { M I T }  \mathrm { I N C }$ </td><td> $. 8 7 4 \pm . 0 2 2 . 3 6 7 \pm . 0 0 7$ </td><td></td><td></td><td>.507</td><td> $. 3 2 5 \pm . 0 1 2 . 8 1 5 \pm . 0 2 9 . 3 4 9 \pm . 0 2 0 . 5 5 9 \pm . 0 5 1$ </td><td></td><td></td><td></td><td></td><td> $. 5 3 2 \pm . 0 5 8$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { M I T }$ </td><td> $. 8 6 7 \pm . 0 0 6 . 5 5 1 \pm . 0 1 2$ </td><td></td><td></td><td>.316</td><td> $. 4 8 5 \pm . 0 1 3 . 8 8 6 \pm . 0 1 0 . 5 8 4 \pm . 0 1 3 . 5 7 6 \pm . 0 0 6$ </td><td></td><td></td><td></td><td></td><td> $. 5 6 6 \pm . 0 0 9$ </td></tr><tr><td> $\mathrm { M I T }  \mathrm { S V D B }$ </td><td> $. 8 7 6 \pm . 0 1 5 . 5 5 6 \pm . 0 3 7$ </td><td></td><td></td><td>.320</td><td> $. 5 6 2 \pm . 0 3 5 . 8 6 6 \pm . 0 2 6 . 5 6 1 \pm . 0 5 4 . 3 9 2 \pm . 0 5 2$ </td><td></td><td></td><td></td><td></td><td> $. 3 8 8 \pm . 0 5 0$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { S V D B }$ </td><td> $. 8 9 7 \pm . 0 0 8 . 5 8 9 \pm . 0 0 8$ </td><td></td><td></td><td>.308</td><td> $. 6 0 8 \pm . 0 1 6 . 8 5 9 \pm . 0 1 9 . 6 3 4 \pm . 0 1 2 . 4 7 5 \pm . 0 2 5$ </td><td></td><td></td><td></td><td></td><td> $. 4 6 7 \pm . 0 2 3$ </td></tr></table>

Note: MIT: MIT-BIH; INC: INCART; Acc.: accuracy; BA: balanced accuracy; A–BA gap: diference between mean accuracy and mean balanced accuracy; F1: macro-F1; AUROC: macro-AUROC; AUPRC: macro-AUPRC; MCC: Matthews correlation coeficient; κ: Cohen’s kappa. Acc., BA, F1, AUROC, AUPRC, MCC, and κ are reported as mean SD across five runs.

The two MIT-BIH–INCART cross-dataset routes produced diferent absolute performance profiles, but these values should not be interpreted as a direct comparison of transfer direction because the target dataset difers between the two experiments. Their transfer behaviour is therefore assessed relative to the corresponding withintarget reference in Section IV-C.

Overall, the Hybrid SSL encoder retained transferable representation structure across all seven evaluated routes, but class-balanced performance remained strongly dependent on the source–target combination and target class distribution. Ranking-based metrics remained comparatively stronger than balanced accuracy in several settings, while the low balanced accuracy and macro-F1 observed for the INCART-target evaluations indicate persistent dificulty in recognising minority classes under patientdisjoint testing.

## B. Architecture-Matched Baseline and Ablation Comparison

Table V compares Hybrid SSL with architecturematched reconstruction-only SSL, contrastive-only SSL, source-supervised Transformer pretraining, and a random frozen Transformer under the same frozen target linearprobing protocol. All conditions use the same Transformer encoder architecture, while their trainable parameter counts are summarised in Table VII. Diferences between the architecture-matched conditions can therefore be interpreted primarily in terms of the representation-learning objective rather than model-family diferences.

Hybrid SSL produced higher mean balanced accuracy and macro-F1 than the random frozen Transformer across all seven source–target routes, confirming that the pretrained encoder learned useful representations beyond architectural inductive bias alone. Hybrid SSL produced higher mean balanced accuracy than the architecture-matched source-supervised Transformer in all seven routes. For macro-F1, Hybrid SSL was higher in four routes, whereas the source-supervised Transformer was slightly higher in three. The comparison should therefore be interpreted as showing a consistent advantage for Hybrid SSL in class-balanced recall, but a mixed routedependent pattern for macro-F1, rather than a general superiority of self-supervised over supervised learning.

The reconstruction-only comparison showed that the benefit of adding the contrastive term was metric- and route-dependent. Hybrid SSL achieved higher balanced accuracy in four of the seven routes, whereas reconstructiononly SSL was higher in three. For macro-F1, Hybrid SSL was higher in five routes and reconstruction-only SSL in two. These findings indicate that the hybrid objective did not uniformly improve performance over masked reconstruction alone, although it showed a modest overall advantage across the evaluated routes.

In contrast, the contrastive-only Transformer performed substantially worse than both Hybrid SSL and reconstruction-only SSL across all seven routes. Balanced accuracy ranged from 0.2015 to 0.3520 for the contrastiveonly condition, compared with 0.3669–0.6464 for Hybrid SSL. This indicates that, under the present augmentation and optimisation configuration, contrastive learning alone was insuficient to learn comparably useful heartbeat representations. The stronger performance of Hybrid SSL relative to contrastive-only SSL therefore appears to depend critically on the reconstruction component, while the incremental benefit of adding contrastive learning to reconstruction remains source–target dependent.

The architecture-matched source-supervised Transformer provides the most direct supervised sourcepretraining comparison. Hybrid SSL achieved higher mean balanced accuracy than the source-supervised Transformer in all seven routes, including 0.5818 versus 0.5597 for MIT-BIH → MIT-BIH, 0.3834 versus 0.3691 for INCART → INCART, and 0.6464 versus 0.5668 for SVDB → SVDB. Macro-F1 showed a less uniform pattern: Hybrid SSL was higher in four of the seven routes, while the source-supervised Transformer was slightly higher for INCART → INCART, SVDB → SVDB, and MIT-BIH → INCART. These results do not establish that self-supervised learning is universally preferable to supervised learning; they show that, with the matched Transformer architecture, patient-disjoint data partitions, training budget, and downstream probe used here, Hybrid SSL produced consistently stronger balanced accuracy while macro-F1 remained route-dependent.

TABLE V  
ARCHITECTURE-MATCHED TRANSFORMER COMPARISON UNDER FROZEN TARGET LINEAR PROBING.
<table><tr><td>Train → Test</td><td> ${ \mathrm { H y b } } .$ </td><td>Rec.</td><td> $\operatorname { C o n } .$ </td><td>Src.</td><td>Rand.</td></tr><tr><td colspan="6">Balanced accuracy</td></tr><tr><td> $\mathrm { M I T }  \mathrm { M I T }$ </td><td>.582±.012</td><td> $. 5 8 6 \pm . 0 1 8$ </td><td> $. 2 4 2 \pm . 0 2 9$ </td><td> $. 5 6 0 \pm . 0 2 9$ </td><td> $. 4 8 2 \pm . 0 3 1$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { I N C }$ </td><td> $. 3 8 3 \pm . 0 1 4$ </td><td> $. 3 8 7 \pm . 0 0 8$ </td><td> $. 2 2 1 \pm . 0 2 0$ </td><td> $. 3 6 9 \pm . 0 0 7$ </td><td> $. 2 8 1 \pm . 0 1 4$ </td></tr><tr><td> $\mathrm { S V D B } \to \mathrm { S V D B }$ </td><td> $. 6 4 6 \pm . 0 2 2$ </td><td> $. 6 1 4 \pm . 0 2 4$ </td><td> $. 3 3 6 \pm . 0 0 5$ </td><td> $. 5 6 7 \pm . 0 6 0$ </td><td> $. 3 8 8 \pm . 0 1 8$ </td></tr><tr><td> $\mathrm { M I T }  \mathrm { I N C }$ </td><td> $. 3 6 7 \pm . 0 0 7$ </td><td> $. 3 7 1 \pm . 0 1 5$ </td><td> $. 2 0 2 \pm . 0 0 3$ </td><td> $. 3 3 2 \pm . 0 2 5$ </td><td> $. 2 8 1 \pm . 0 1 4$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { M I T }$ </td><td> $. 5 5 1 \pm . 0 1 2$ </td><td> $. 5 1 8 \pm . 0 2 3$ </td><td> $. 3 5 2 \pm . 0 7 7$ </td><td> $. 4 2 8 \pm . 0 7 1$ </td><td> $. 4 8 2 \pm . 0 3 1$ </td></tr><tr><td> $\mathrm { M I T }  \mathrm { S V D B }$ </td><td> $. 5 5 6 \pm . 0 3 7$ </td><td> $. 5 1 8 \pm . 0 2 5$ </td><td> $. 3 3 3 \pm . 0 0 2$ </td><td> $. 4 8 6 \pm . 0 4 5$ </td><td> $. 3 8 8 \pm . 0 1 8$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { S V D B }$ </td><td> $. 5 8 9 \pm . 0 0 8$ </td><td>.587±.008</td><td> $. 3 4 6 \pm . 0 1 6$ </td><td> $. 4 2 9 \pm . 0 2 4$ </td><td> $. 3 8 8 \pm . 0 1 8$ </td></tr><tr><td colspan="6">Macro-F1</td></tr><tr><td> $\mathrm { M I T }  \mathrm { M I T }$ </td><td>.497±.013</td><td> $. 4 8 4 \pm . 0 2 4$ </td><td> $. 2 4 9 \pm . 0 3 6$ </td><td> $. 4 5 8 \pm . 0 1 4$ </td><td></td></tr><tr><td> $\mathrm { I N C }  \mathrm { I N C }$ </td><td></td><td>.324±.017.335±.012</td><td> $. 2 2 5 \pm . 0 3 2$ </td><td> $. 3 2 5 \pm . 0 1 2$ </td><td>.482±.035 .304±.015</td></tr><tr><td> $\mathrm { S V D B } \to \mathrm { S V D B }$ </td><td>.565±.023</td><td> $. 5 5 4 \pm . 0 2 6$ </td><td> $. 3 1 8 \pm . 0 0 9$ </td><td> $. 5 7 4 \pm . 0 5 7$ </td><td>.410±.026</td></tr><tr><td> $\mathrm { M I T }  \mathrm { I N C }$ </td><td>.325±.012</td><td> $. 3 3 6 \pm . 0 2 3$ </td><td> $. 1 9 2 \pm . 0 0 6$ </td><td> $. 3 2 6 \pm . 0 2 6$ </td><td>.304±.015</td></tr><tr><td> $\mathrm { I N C }  \mathrm { M I T }$ </td><td>.485±.013</td><td> $. 4 5 4 \pm . 0 2 8$ </td><td> $. 3 3 1 \pm . 0 6 2$ </td><td> $. 4 1 5 \pm . 0 6 1$ </td><td>.482±.035</td></tr><tr><td> $\mathrm { M I T }  \mathrm { S V D B }$ </td><td>.562±.035</td><td> $. 5 2 7 \pm . 0 2 2$ </td><td> $. 3 1 4 \pm . 0 0 1$ </td><td> $. 4 9 1 \pm . 0 4 4$ </td><td>.410±.026</td></tr><tr><td> $\mathrm { I N C }  \mathrm { S V D B }$ </td><td>.608±.016</td><td> $. 5 9 9 \pm . 0 0 9$ </td><td> $. 3 3 3 \pm . 0 2 1$ </td><td> $. 4 4 7 \pm . 0 3 3$ </td><td>.410±.026</td></tr></table>

Note: MIT: MIT-BIH; INC: INCART; Hyb.: Hybrid SSL; Rec.: reconstruction-only SSL; Con.: contrastive-only SSL; Src.: sourcesupervised Transformer; Rand.: randomly initialised frozen Transformer. Bold values indicate the highest mean performance for each source–target setting.

Table VI provides an additional target-supervised endto-end reference. The end-to-end Transformer achieved balanced accuracies of 0.5502±0.0299, 0.3607±0.0110, and 0.5508 ± 0.0310 on MIT-BIH, INCART, and SVDB, respectively. These values were lower than the corresponding within-target Hybrid SSL frozen-representation results of 0.5818±0.0117, $0 . 3 8 3 4 { \pm } 0 . 0 1 4 2$ , and $0 . 6 4 6 4 { \pm } 0 . 0 2 1 7$ . Mean macro-F1 showed the same ordering on all three datasets. However, the target-supervised condition addresses a diferent question from the frozen-transfer experiments because its encoder and classifier are jointly optimised using target labels. It should therefore be interpreted as a supervised classification reference rather than as a directly matched source-pretraining ablation. The results nevertheless show that, under the same patient-disjoint partitions and unweighted class-imbalanced training setting, end-to-end target supervision did not automatically yield stronger class-balanced performance than the representations obtained through self-supervised pretraining followed by linear probing.

## C. Target-Normalised Transfer Gaps

To compare cross-dataset generalisation across the principal representation-learning strategies, Table VIII reports both the absolute cross-dataset performance and the corresponding target-normalised transfer gap for Hybrid SSL, reconstruction-only SSL, and the source-supervised Transformer. For each method, the gap compares crossdataset performance with that method’s own within-target reference on the same target dataset. Balanced accuracy and macro-F1 are reported because they are the principal class-balanced metrics used throughout this study.

TABLE VI  
TARGET-SUPERVISED PERFORMANCE OF THE END-TO-END TRANSFORMER MODEL. VALUES ARE REPORTED AS MEAN STANDARD DEVIATION.
<table><tr><td>Metric</td><td>MIT-BIH</td><td>INCART</td><td>SVDB</td></tr><tr><td>Accuracy</td><td> $0 . 8 1 6 \pm 0 . 0 7 6$ </td><td> $0 . 8 4 0 \pm 0 . 0 2 8$ </td><td> $0 . 8 5 0 \pm 0 . 0 2 9$ </td></tr><tr><td>Balanced Accuracy</td><td> $0 . 5 5 0 \pm 0 . 0 3 0$ </td><td> $0 . 3 6 1 \pm 0 . 0 1 1$ </td><td> $0 . 5 5 1 \pm 0 . 0 3 1$ </td></tr><tr><td>Macro-F1</td><td> $0 . 4 5 9 \pm 0 . 0 3 0$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 6$ </td><td> $0 . 5 3 8 \pm 0 . 0 4 4$ </td></tr><tr><td>Macro-AUROC</td><td> $0 . 8 6 1 \pm 0 . 0 3 4$ </td><td> $0 . 7 7 4 \pm 0 . 0 4 3$ </td><td> $0 . 8 3 4 \pm 0 . 0 0 8$ </td></tr><tr><td>Macro-AUPRC</td><td> $0 . 5 0 4 \pm 0 . 0 4 0$ </td><td> $0 . 3 1 4 \pm 0 . 0 2 6$ </td><td> $0 . 5 4 8 \pm 0 . 0 4 5$ </td></tr><tr><td>MCC</td><td> $0 . 5 1 9 \pm 0 . 0 9 2$ </td><td> $0 . 5 1 9 \pm 0 . 0 4 5$ </td><td> $0 . 3 3 9 \pm 0 . 0 6 1$ </td></tr><tr><td>Cohen&#x27;s κ</td><td> $0 . 4 9 2 \pm 0 . 1 1 4$ </td><td> $0 . 4 7 4 \pm 0 . 0 5 2$ </td><td> $0 . 3 3 5 \pm 0 . 0 5 8$ </td></tr></table>

TABLE VII

TRAINABLE PARAMETER COUNTS ACROSS ARCHITECTURE-MATCHED CONDITIONS. ALL CONDITIONS USE THE SAME TRANSFORMER ENCODER WITH 3,358,592 PARAMETERS.
<table><tr><td>Condition Trainable parameters</td></tr><tr><td>Hybrid SSL 3,482,176 Reconstruction-only SSL 3,457,408</td></tr><tr><td>Contrastive-only SSL 3,383,360</td></tr><tr><td>Source-supervised (5-class) 3,359,237</td></tr><tr><td>Source-supervised (3-class) 3,358,979</td></tr><tr><td>Target-supervised (5-class) 3,359,237</td></tr><tr><td>Target-supervised (3-class) 3,358,979</td></tr><tr><td>Random frozen Transformer</td></tr></table>

Note: Parameter counts include all trainable components used during pretraining or supervised training. The random Transformer is not pretrained; its encoder remains frozen during downstream evaluation.

The comparative results show that Hybrid SSL achieved the highest mean cross-dataset balanced accuracy and macro-F1 in three of the four transfer routes. For INCART → MIT-BIH, Hybrid SSL achieved a balanced accuracy of 0.551 and macro-F1 of 0.485, compared with 0.518 and 0.454 for reconstruction-only SSL and 0.428 and 0.415 for the source-supervised Transformer. Hybrid SSL was also strongest for both transfers to SVDB, achieving balanced accuracy of 0.556 and 0.589 for MIT-BIH → SVDB and INCART → SVDB, respectively. Reconstruction-only SSL produced the highest values for MIT-BIH → INCART, although its advantage over Hybrid SSL was small for both balanced accuracy (0.371 versus 0.367) and macro-F1 (0.336 versus 0.325).

The target-normalised gaps provide complementary information about how much each method changed relative to its own within-target reference. These gaps did not uniformly favour Hybrid SSL. For example, the source-supervised Transformer had the smallest balancedaccuracy reduction for MIT-BIH→ SVDB, despite having substantially lower absolute cross-dataset balanced accuracy than Hybrid SSL (0.486 versus 0.556). Similarly, reconstruction-only SSL had the smallest balancedaccuracy reduction for INCART → SVDB, while Hybrid SSL achieved the higher absolute balanced accuracy and macro-F1. Thus, resistance to degradation relative to within-target performance was method and routedependent, whereas Hybrid SSL provided the strongest absolute cross-dataset performance in three of the four evaluated transfer settings.

TABLE VIII  
CROSS-DATASET PERFORMANCE AND TARGET-NORMALISED TRANSFER GAPS. PANEL A COMPARES BALANCED ACCURACY (BA) AND PANEL B COMPARES MACRO-F1 ACROSS THE THREE PRINCIPAL REPRESENTATION-LEARNING STRATEGIES.  
(a) Balanced accuracy and target-normalised gaps
<table><tr><td>Transfer</td><td></td><td>Hyb. BA (∆BA) Rec. BA (∆BA) Src. BA (∆BA)</td><td></td></tr><tr><td> $\mathrm { M I T }  \mathrm { I N C }$ </td><td> $. 3 6 7 \left( - . 0 1 6 \right)$ </td><td> $. 3 7 1 \left( - . 0 1 6 \right)$ </td><td> $. 3 3 2 ( - . 0 3 8 )$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { M I T }$ </td><td>.551 (−.031)</td><td> $. 5 1 8 \left( - . 0 6 8 \right)$ </td><td>.428 (−.132)</td></tr><tr><td> $\mathrm { M I T }  \mathrm { S V D B }$ </td><td> $. 5 5 6 \left( - . 0 9 1 \right)$ </td><td> $. 5 1 8 \left( - . 0 9 6 \right)$ </td><td> $. 4 8 6 \left( - . 0 8 0 \right)$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { S V D B }$ </td><td> $. 5 8 9 \left( - . 0 5 7 \right)$ </td><td> $. 5 8 7 \left( - . 0 2 7 \right)$ </td><td> $. 4 2 9 \left( - . 1 3 8 \right)$ </td></tr></table>

(b) Macro-F1 and target-normalised gaps
<table><tr><td>Transfer</td><td>Hyb. F1 (∆F1) Rec. F1 (∆F1) Src. F1 (∆F1)</td><td></td><td></td></tr><tr><td> $\mathrm { M I T }  \mathrm { I N C }$ </td><td> $. 3 2 5 \left( + . 0 0 1 \right)$ </td><td> $. 3 3 6 \left( + . 0 0 1 \right)$ </td><td> $. 3 2 6 \left( + . 0 0 1 \right)$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { M I T }$ </td><td> $. 4 8 5 \left( - . 0 1 2 \right)$ </td><td> $. 4 5 4 \left( - . 0 3 0 \right)$ </td><td> $. 4 1 5 \left( - . 0 4 3 \right)$ </td></tr><tr><td> $\mathrm { M I T }  \mathrm { S V D B }$ </td><td> $. 5 6 2 \left( - . 0 0 3 \right)$ </td><td> $. 5 2 7 \left( - . 0 2 6 \right)$ </td><td> $. 4 9 1 \ : ( - . 0 8 4 )$ </td></tr><tr><td> $\mathrm { I N C }  \mathrm { S V D B }$ </td><td> $. 6 0 8 \left( + . 0 4 3 \right)$ </td><td> $. 5 9 8 \left( + . 0 4 5 \right)$ </td><td> $. 4 4 7 \left( - . 1 2 8 \right)$ </td></tr></table>

Note: MIT: MIT-BIH; INC: INCART; Hyb.: Hybrid SSL; Rec.: reconstruction-only SSL; Src.: source-supervised Transformer. Values report mean cross-dataset performance, with the corresponding target-normalised gap in parentheses. For each method, ∆ is defined as cross-dataset performance minus that method’s corresponding withintarget reference. Negative values indicate degradation relative to within-target training, whereas positive values indicate improvement. Bold values indicate the highest mean cross-dataset performance for each transfer route and metric.

## D. Class-Imbalance Efects

Class imbalance considerably afected how the transfer results should be interpreted, particularly for INCARTtarget evaluation. The patient-disjoint target partitions were strongly imbalanced (Table II); an N-only classifier would achieve accuracies of 0.852, 0.886, and 0.880 on the MIT-BIH, INCART, and SVDB test sets, respectively, despite balanced accuracy remaining at the corresponding chance level. Notably, the INCART majority-class reference accuracy of 0.886 exceeds the mean raw accuracy of both INCART → INCART (0.845) and MIT-BIH → IN-CART (0.874). Raw accuracy alone therefore does not establish whether a learned representation retains useful class-discriminative information.

Hybrid SSL nevertheless achieved balanced accuracy substantially above chance in all evaluated settings, showing that its representations did not simply collapse to majority-class prediction. Table IV also quantifies the divergence between accuracy and balanced accuracy. The two largest gaps occurred in the INCART-target settings, indicating that useful representation structure was retained but did not translate into uniformly strong recall across the five AAMI classes.

The row-normalised confusion matrices in Fig. 2 further show that the performance limitation was class-specific rather than a uniform degradation across heartbeat categories.

Because INCART contains only two Q-class test beats, a sensitivity analysis was performed to assess how strongly this extremely sparse class afected the aggregate classbalanced metrics. Without retraining the models or changing their predictions, balanced accuracy and macro-F1 were recalculated for each seeded run after excluding Q from metric aggregation. Table IX shows substantial improvements in both metrics; however, balanced accuracy remained below 0.50 for both INCART → INCART and MIT-BIH → INCART. The poor class-balanced performance therefore cannot be attributed solely to the twoexample Q class. The original five-class results remain the principal analysis for consistency with the predefined AAMI mapping, while Q-dependent estimates should be interpreted cautiously.

TABLE IX  
EFFECT OF EXCLUDING THE SPARSELY REPRESENTED Q CLASS FROM INCART-TARGET EVALUATION. VALUES ARE MEAN  SD ACROSS FIVE RUNS.
<table><tr><td>Experiment</td><td>Classes</td><td>BA</td><td>Macro-F1</td></tr><tr><td>INCART → INCART</td><td>5-class</td><td> $0 . 3 8 3 4 \pm 0 . 0 1 4 2$ </td><td> $0 . 3 2 3 8 \pm 0 . 0 1 6 9$ </td></tr><tr><td></td><td>Excl. Q</td><td> $\mathbf { 0 . 4 7 9 2 \pm 0 . 0 1 7 7 }$ </td><td> $\mathbf { 0 . 4 0 4 8 \pm 0 . 0 2 1 1 }$ </td></tr><tr><td>MIT-BIH → INCART</td><td>5-class</td><td> $0 . 3 6 6 9 \pm 0 . 0 0 7 3$ </td><td> $0 . 3 2 4 8 \pm 0 . 0 1 2 4$ </td></tr><tr><td></td><td>Excl. Q</td><td> $\mathbf { 0 . 4 5 8 6 \ : \pm 0 . 0 0 9 1 }$ </td><td> $\mathbf { 0 . 4 0 6 0 \pm 0 . 0 1 5 5 }$ </td></tr></table>

TABLE X

CLASSWISE HYBRID SSL PERFORMANCE FOR THE TWO INCART-TARGET EXPERIMENTS.
<table><tr><td>Exp. C</td><td></td><td>Sup.</td><td>Rec.</td><td>PPV</td><td>F1</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>ONI</td><td>N</td><td></td><td>49,581 .841 ± .036</td><td></td><td></td><td></td><td>.986 ± .001 .907 ± .022 .939 ± .009.990 ± .001</td></tr><tr><td></td><td>S</td><td>625</td><td>.104 ± .049</td><td></td><td></td><td></td><td>.041 ± .040 .055 ± .047 .602 ± .028 .031 ± .024</td></tr><tr><td>←</td><td>V</td><td>5,655</td><td>.973 ± .005</td><td></td><td></td><td></td><td>.497 ± .044 .657 ± .037 .974 ± .012.807 ± .103</td></tr><tr><td>ONI</td><td>F</td><td>72</td><td> $. 0 0 0 \pm . 0 0 0$ </td><td></td><td></td><td>.000 ± .000 .000 ± .000 .731 ± .031</td><td> $. 0 0 4 \pm . 0 0 2$ </td></tr><tr><td></td><td>Q</td><td>2</td><td> $. 0 0 0 \pm . 0 0 0$ </td><td>.000 ± .000 .000 ± .000</td><td></td><td> $. 9 1 7 \pm . 0 8 5$ </td><td> $. 0 0 5 \pm . 0 0 4$ </td></tr><tr><td></td><td>N</td><td>49,581</td><td> $. 8 8 5 \pm . 0 2 4$ </td><td>.975 ± .003</td><td> $. 9 2 8 \pm . 0 1 4$ </td><td> $. 9 2 4 \pm . 0 1 9$ </td><td> $. 9 8 7 \pm . 0 0 3$ </td></tr><tr><td>ONI</td><td>S</td><td>625</td><td> $. 0 7 2 \pm . 0 4 4$ </td><td>.073 ± .038</td><td> $. 0 7 0 \pm . 0 3 6$ </td><td> $. 6 7 0 \pm . 0 4 4$ </td><td> $. 0 3 9 \pm . 0 2 4$ </td></tr><tr><td>↑</td><td>V</td><td>5,655</td><td> $. 8 7 5 \pm . 0 2 4$ </td><td>.483 ± .055</td><td> $. 6 2 1 \pm . 0 4 9$ </td><td> $. 9 5 4 \pm . 0 1 5$ </td><td> $. 7 1 2 \pm . 0 9 0$ </td></tr><tr><td>MII</td><td>F</td><td>72</td><td> $. 0 0 3 \pm . 0 0 6$ </td><td>.067 ± .149 .005 ± .012</td><td></td><td> $. 7 0 3 \pm . 0 4 1$ </td><td> $. 0 0 6 \pm . 0 0 3$ </td></tr><tr><td></td><td>Q</td><td></td><td></td><td>2 .000 ± .000 .000 ± .000 .000 ± .000</td><td></td><td> $. 8 2 3 \pm . 0 7 6$ </td><td> $. 0 0 1 \pm . 0 0 1$ </td></tr></table>

Note: MIT: MIT-BIH; INC: INCART; Exp.: experiment; C: class; Sup.: support; Rec.: recall; PPV: positive predictive value; AUROC: area under the receiver operating characteristic curve; AUPRC: area under the precision–recall curve. Values are mean SD across five runs.

The classwise results in Table X clarify the remaining limitation. Q achieved zero recall, PPV, and F1 in both INCART-target settings, and its ranking metrics are inherently unstable given only two positive test examples. More importantly, poor recognition was not confined to Q: F- and S-class recall also remained very low, whereas N and V were recognised substantially more reliably. This pattern occurred for both MIT-BIH → INCART and the INCART → INCART within-target reference. The minority-class weakness is therefore not attributable solely to cross-dataset transfer; it also reflects the severe class distribution and patient-disjoint characteristics of the INCART target.

Overall, these results distinguish useful representation transfer from balanced arrhythmia recognition. Hybrid SSL retained class-discriminative information beyond a trivial majority-class solution, but neither cross-dataset transfer nor target-specific self-supervised pretraining produced reliable recognition of all minority classes under the INCART patient-disjoint evaluation. The Q-exclusion analysis further demonstrates that extreme Q sparsity explains only part of the observed class-balanced performance limitation.

Hybrid SSL: aggregate row-normalised confusion matrices (n = 5 seeds)  
![](images/a357f3d32ac5d18da1d34451b89959de7e75ae25cc1590afccbc85ae8c80ad46.jpg)

![](images/324c8e08f0d9b5dc2e5e411030567d12b44fecc264683b13010f61034d4f7932.jpg)

![](images/1f435369504d07a9a77323e1bb97d8cc165f1fc20d2e74a26d0db8c7e9f650aa.jpg)

![](images/b1403eb21c55140efd1074fda59bd637be50e3bf09abd762de68ea55666d7549.jpg)

![](images/6f5b95fe7a40113cfd346a93b562af0e8731504249d993ea1e93894a87fe5721.jpg)

![](images/c26d633ea7f18e88a7a0704188cecd24601f2c9c0a54e2932213e6590588431d.jpg)

![](images/59586428ebc20777409cc568574cd8b0dd3c4c544282d80c40196b8b8bff6be0.jpg)  
Fig. 2. Aggregated row-normalised confusion matrices for Hybrid SSL across the seven source–target routes.

The discrepancy between accuracy and balanced accuracy was less pronounced for the MIT-BIH and SVDB targets than for INCART. However, SVDB was evaluated only on the N/S/V classes and excluded the severely under-represented F and Q categories; its class-balanced results are therefore not directly comparable with the fiveclass MIT-BIH and INCART settings.

## V. DISCUSSION

Table XI summarises the principal experimental findings. This study evaluated whether self-supervised ECG representations learned from one arrhythmia database can transfer to independent target databases under patientdisjoint evaluation. The proposed Hybrid SSL encoder consistently outperformed the random frozen Transformer across all seven source–target routes, showing that sourcedomain self-supervised pretraining learned transferable ECG structure beyond architectural inductive bias alone. The architecture-matched ablations further clarify where this benefit originated. Reconstruction-only SSL remained highly competitive with Hybrid SSL, with Hybrid SSL achieving higher balanced accuracy in four of seven routes and higher macro-F1 in five of seven routes. In contrast, contrastive-only SSL performed substantially worse than both reconstruction-only and Hybrid SSL across all routes. These findings indicate that masked reconstruction was a critical contributor to useful heartbeat representations under the tested configuration, whereas the additional contrastive objective provided a modest and source–targetdependent benefit rather than a uniform improvement.

The architecture-matched source-supervised Transformer provides a controlled comparison of source supervision because it used the same encoder architecture, source partitions, training budget, and frozen target-probing protocol. Hybrid SSL achieved higher mean balanced accuracy in all seven routes, while macro-F1 was higher for Hybrid SSL in four routes and for the source-supervised Transformer in three. This should not be interpreted as evidence that self-supervised learning is generally superior to supervised ECG learning; the comparison is specific to the present architecture, datasets, optimisation settings, class distributions, and frozen linear-probe evaluation. The target-supervised end-to-end Transformer also addresses a diferent experimental question because both its encoder and classifier were optimised directly using target labels. Its inclusion therefore provides an additional supervised classification reference rather than a matched sourcepretraining ablation. Together, these comparisons show that the pretraining objective materially afected frozen representation quality while highlighting the importance of architecture- and task-matched baselines when attributing performance diferences to self-supervision.

The results also demonstrate the importance of classsensitive evaluation. Raw accuracy substantially overstated performance in the most imbalanced settings, particularly when INCART was the target dataset. For example, MIT-BIH → INCART achieved an accuracy of 0.8737 but a balanced accuracy of only 0.3669, while INCART → INCART achieved 0.8445 and 0.3834, respectively. Classwise analysis showed that this divergence was driven by weak minority-class recognition: Q recall was zero in both INCART-target settings, F recall was zero or near zero, and S recall remained low, whereas N and V were comparatively better recognised. The

TABLE XI  
SUMMARY OF THE PRINCIPAL EXPERIMENTAL FINDINGS.
<table><tr><td>Comparison</td><td>Outcome</td><td>Evidence</td></tr><tr><td>Hybrid SSL vs. Random Hybrid SSL Transformer</td><td></td><td>Higher balanced accuracy and macro-F1 in all seven source-target evaluation routes.</td></tr><tr><td>Hybrid SSL vs. Source- supervised Transformer</td><td>Hybrid SSL for BA</td><td>Higher balanced accuracy in all seven routes. Macro-F1 was higher for Hybrid SSL in 4/7 routes and for the source-supervised Transformer in 3/7 routes.</td></tr><tr><td>Hybrid SSL Reconstruction-only SSL</td><td>Mixed</td><td>Hybrid SSL achieved higher balanced accuracy in 4/7 routes and higher macro-F1 in 5/7 routes, confirming that the incremental effect of the contrastive objective was route- and</td></tr><tr><td>Hybrid SSL vs. Contrastive-</td><td>Hybrid SSL</td><td>metric-dependent. Contrastive-only SSL produced substantially lower balanced accuracy and macro-F1 across all seven evaluation routes.</td></tr><tr><td>only SSL Cross-dataset generalisation</td><td>Hybrid SSL in 3/4 routes</td><td>Hybrid SSL achieved the highest cross-dataset balanced accuracy and macro-F1 in three of four transfer routes; reconstruction-only SSL was highest for MIT-BIH → INCART. Target-</td></tr><tr><td>Minority-class recognition</td><td>Unresolved</td><td>normalised degradation remained method and route-dependent. Near-zero fusion-beat (F) recall and consistently low supraventricular ectopic-beat (S) recall</td></tr></table>

Q-excluded sensitivity analysis further showed that the extremely small INCART Q test set materially depressed the five-class balanced accuracy and macro-F1 values. These findings reinforce that high overall accuracy cannot be interpreted as uniformly reliable performance across AAMI classes and justify the use of balanced accuracy, macro-F1, classwise recall, ranking-based metrics, and confusion-matrix analysis alongside raw accuracy.

The comparative cross-dataset analysis further clarifies the generalisation behaviour of the learned representations. Hybrid SSL achieved the highest mean crossdataset balanced accuracy and macro-F1 in three of the four transfer routes, while reconstruction-only SSL was strongest for MIT-BIH → INCART. In particular, Hybrid SSL outperformed both reconstruction-only SSL and the source-supervised Transformer on both metrics for INCART → MIT-BIH, MIT- BIH → SVDB, and INCART → SVDB. This advantage was not equivalent to consistently having the smallest target-normalised degradation, however. The relative gaps were methodand route-dependent; for example, the source-supervised Transformer showed a smaller balanced-accuracy reduction than Hybrid SSL for MIT-BIH → SVDB despite substantially lower absolute cross-dataset balanced accuracy. Thus, the results support stronger absolute cross-dataset performance for Hybrid SSL in most evaluated transfer routes, while also showing that resistance to degradation relative to within-target training is not uniformly superior. Useful representation transfer therefore does not imply that domain shift has been eliminated.

Several limitations should be considered. First, although Hybrid SSL source pretraining was unlabelled, labelled target training records were still required to fit the linear probe; the framework should therefore be interpreted as label-eficient source pretraining rather than fully labelfree arrhythmia classification. Second, the experiments used public ECG databases and a single-lead setting, and further validation is required on larger, prospective, multicentre, and multi-lead datasets before clinical generalisability can be assessed. The results should therefore be interpreted as methodological evidence concerning ECG representation transfer rather than evidence of diagnostic utility, patient-level efectiveness, or deployment readiness. Third, the use of a linear probe was intentional because it isolates the quality of the frozen representation, but it does not establish whether non-linear target classifiers, calibration methods, cost-sensitive learning, or class-balanced decision rules could improve minority-class operating points. Finally, the extremely small support of some minority classes limits the stability of classspecific estimates. In particular, INCART contained only four Q training beats and two Q test beats, while the F test set contained 72 beats. The corresponding Qspecific recall, precision, F1, AUROC, and AUPRC values should therefore be interpreted descriptively rather than as population-level estimates. Patient-disjoint splitting prevents the same patient from contributing to both training and test partitions, but it does not remove broader sources of domain shift arising from diferences in annotation policy, acquisition conditions, and class prevalence across datasets.

Future work should investigate semi-supervised and fewshot target adaptation, class-balanced calibration, multilead representation learning, and domain-aware pretraining. It would also be valuable to determine whether the ranking structure learned by Hybrid SSL can be converted into more reliable minority-class decisions through calibrated thresholds, cost-sensitive target probing, or other target-eficient adaptation strategies without compromising patient-disjoint evaluation.

## VI. CONCLUSION

This study presented TransECG-SSL, a hybrid selfsupervised framework for learning transferable ECG heartbeat representations under patient-disjoint evaluation. Across the four cross-dataset transfer routes, Hybrid SSL achieved the highest mean balanced accuracy and macro-F1 in three settings, while reconstruction-only SSL was strongest for MIT-BIH → INCART. Hybrid SSL also consistently outperformed the random frozen Transformer and showed stronger cross-dataset performance than the architecture-matched source-supervised Transformer in the evaluated transfer settings. Reconstruction-based learning contributed strongly to representation quality, whereas the incremental benefit of the contrastive objective remained source–target dependent.

Despite this transferability, Hybrid SSL did not consistently show the smallest degradation relative to its own within-target reference, and its balanced accuracy remained below target-specific Hybrid SSL in all four cross-dataset routes. This distinction indicates that strong absolute cross-dataset performance does not imply complete invariance to domain shift. Minority-class recognition also remained challenging, particularly in the fiveclass settings. Overall, TransECG-SSL provides a labeleficient approach to cross-dataset ECG representation learning and demonstrates strong comparative transfer performance, while motivating further work on domainrobust pretraining and improved minority-class representation.

## CODE AVAILABILITY

The source code supporting this study is maintained at https://github.com/ahtisham-86/TransECG-SSL and will be made publicly available after publication.

## REFERENCES

[1] S. Hong, Y. Zhou, J. Shang, C. Xiao, and J. Sun, “Opportunities and challenges of deep learning methods for electrocardiogram data: A systematic review,” Computers in biology and medicine, vol. 122, p. 103801, 2020, DOI: 10.1016/j.compbiomed.2020.103801.

[2] A. Ayyub, C. Politis, and M. A. Usman, “A comprehensive review of ai-based detection of arrhythmia using electrocardiogram (ecg),” Computers in Biology and Medicine, vol. 196, p. 110594, 2025, DOI: 10.1016/j.compbiomed.2025.110594.

[3] Q. Xiao, K. Lee, S. A. Mokhtar, I. Ismail, A. L. b. M. Pauzi, Q. Zhang, and P. Y. Lim, “Deep learning-based ecg arrhythmia classification: A systematic review,” Applied Sciences, vol. 13, no. 8, p. 4964, 2023, DOI: 10.3390/app13084964.

[4] G. A. Silva, P. H. Silva, G. J. Moreira, V. L. d. S. Freitas, J. C. Gertrudes, and E. J. d. S. Luz, “A systematic review of ecg arrhythmia classification: embedded feasibility, adherence to standards, and fair evaluation,” Research on Biomedical Engineering, vol. 42, no. 3, p. 42, 2026, DOI: 10.1007/s42600- 026-00485-z.

[5] C. Han, P. Wang, R. Huang, and L. Cui, “Hctnet: An experience-guided deep learning network for inter-patient arrhythmia classification on imbalanced dataset,” Biomedical Signal Processing and Control, vol. 78, p. 103910, 2022, DOI: 10.1016/j.bspc.2022.103910.

[6] H. Chen, G. Wang, G. Zhang, P. Zhang, and H. Yang, “Clecg: A novel contrastive learning framework for electrocardiogram arrhythmia classification,” IEEE Signal Processing Letters, vol. 28, pp. 1993–1997, 2021, DOI: 10.1109/LSP.2021.3114119.

[7] T. Mehari and N. Strodthof, “Self-supervised representation learning from 12-lead ecg data,” Computers in biology and medicine, vol. 141, p. 105114, 2022, DOI: 10.1016/j.compbiomed.2021.105114.

[8] H. Zhang, W. Liu, J. Shi, S. Chang, H. Wang, J. He, and Q. Huang, “Maefe: Masked autoencoders family of electrocardiogram for self-supervised pretraining and transfer learning,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–15, 2022, DOI: 10.1109/TIM.2022.3228267.

[9] S. Soltanieh, J. Hashemi, and A. Etemad, “In-distribution and out-of-distribution self-supervised ecg representation learning for arrhythmia detection,” IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 2, pp. 789–800, 2023, DOI: 10.1109/JBHI.2023.3331626.

[10] K. McKeen, S. Masood, A. Toma, B. Rubin, and B. Wang, “Ecgfm: An open electrocardiogram foundation model,” Jamia Open, vol. 8, no. 5, p. ooaf122, 2025, DOI: 10.1093/jamiaopen/ooaf122.

[11] M. Kachuee, S. Fazeli, and M. Sarrafzadeh, “Ecg heartbeat classification: A deep transferable representation,” in 2018 IEEE international conference on healthcare informatics (ICHI). IEEE, 2018, pp. 443–444, DOI: 10.1109/ICHI.2018.00092.

[12] K. Weimann and T. O. Conrad, “Transfer learning for ecg classification,” Scientific reports, vol. 11, no. 1, p. 5251, 2021, DOI: 10.1038/s41598-021-84374-8.

[13] M. Cao, T. Zhao, Y. Li, W. Zhang, P. Benharash, and R. Ramezani, “Ecg heartbeat classification using deep transfer learning with convolutional neural network and stft technique,” in Journal of Physics: Conference Series, vol. 2547, no. 1. IOP Publishing, 2023, p. 012031, DOI: 10.1088/1742- 6596/2547/1/012031.

[14] M. N. Imtiaz and N. Khan, “Cross-database and crosschannel electrocardiogram arrhythmia heartbeat classification based on unsupervised domain adaptation,” Expert Systems with Applications, vol. 244, p. 122960, 2024, DOI: 10.1016/j.eswa.2023.122960.

[15] A. Ballas and C. Diou, “A domain generalization approach for out-of-distribution 12-lead ecg classification with convolutional neural networks,” in 2022 IEEE Eighth International Conference on Big Data Computing Service and Applications (BigDataService). IEEE, 2022, pp. 9–13, DOI: 10.1109/Big-DataService55688.2022.00009.

[16] R. Li, Y. Aierken, Y. Xu, J. Liu, and Y. Tang, “Research on cross-dataset cardiac signal domain generalization and feature interpretability,” Scientific Reports, vol. 16, no. 1, p. 3138, 2025, DOI: 10.1038/s41598-025-33057-9.

[17] M. Kim and M. Shin, “Enhancing ecg classification generalization through unified multi-dataset training,” Sensors, vol. 26, no. 6, p. 1830, 2026, DOI: 10.3390/s26061830.

[18] F. M. Dias, H. L. Monteiro, T. W. Cabral, R. Naji, M. Kuehni, and E. J. d. S. Luz, “Arrhythmia classification from singlelead ecg signals using the inter-patient paradigm,” Computer Methods and Programs in Biomedicine, vol. 202, p. 105948, 2021, DOI: 10.1016/j.cmpb.2021.105948.

[19] H. Lyu, X. Li, J. Zhang, C. Zhou, X. Tang, F. Xu, Y. Yang, Q. Huang, W. Xiang, and D. Li, “Automated inter-patient arrhythmia classification with dual attention neural network,” Computer Methods and Programs in Biomedicine, vol. 236, p. 107560, 2023, DOI: 10.1016/j.cmpb.2023.107560.

[20] Y. Li, R. Qian, and K. Li, “Inter-patient arrhythmia classification with improved deep residual convolutional neural network,” Computer Methods and Programs in Biomedicine, vol. 214, p. 106582, 2022, DOI: 10.1016/j.cmpb.2021.106582.

[21] Y. Xu, S. Zhang, and W. Xiao, “Inter-patient ecg classification with intra-class coherence based weighted kernel extreme learning machine,” Expert Systems with Applications, vol. 227, p. 120095, 2023, DOI: 10.1016/j.eswa.2023.120095.

[22] J. Niu, Y. Tang, Z. Sun, and W. Zhang, “Inter-patient ecg classification with symbolic representations and multi-perspective convolutional neural networks,” IEEE journal of biomedical and health informatics, vol. 24, no. 5, pp. 1321–1332, 2019, DOI: 10.1109/JBHI.2019.2942938.

[23] H. Zakaria, E. S. H. Nurdiniyah, A. M. Kurniawati, D. Naufal, and N. Sutisna, “Morphological arrhythmia classification based on inter-patient and two leads ecg using machine learning,” IEEE Access, vol. 12, pp. 147 372–147 386, 2024, DOI: 10.1109/ACCESS.2024.3469640.

[24] T. H. Rafi and Y. W. Ko, “Heartnet: Self multihead attention mechanism via convolutional network with adversarial data synthesis for ecg-based arrhythmia classification,” IEEE Access, vol. 10, pp. 100 501–100 512, 2022, DOI: 10.1109/AC-CESS.2022.3206431.

[25] J. Tifany and Q. Zhang, “Robust multilabel ecg classification under class imbalance using augmentation and adaptive iterative training,” Biomedical Signal Processing and Control, vol. 118, p. 109767, 2026, DOI: 10.1016/j.bspc.2026.109767.

[26] S. Dehghani and S. Pierre, “Rare-beat detection in heterogeneous ecg: Federated learning under severe class imbalance and class-missing clients,” IEEE Access, 2026, DOI: 10.1109/AC-CESS.2026.3715579.

[27] L. Yuan and M. Y. Siyal, “Target-oriented augmentation privacy-protection domain adaptation for imbalanced ecg beat

classification,” Biomedical Signal Processing and Control, vol. 86, p. 105308, 2023, DOI: 10.1016/j.bspc.2023.105308.

[28] J. Hu, C. Li, J. Cao, and B. Kou, “A novel multimodal self-supervised framework for ecg arrhythmia classification,” Computers in Biology and Medicine, vol. 198, p. 111137, 2025, DOI: 10.1016/j.compbiomed.2025.111137.

[29] N. Maghsoodi, S. Nassar, P. F. Wilson, M. N. N. To, S. Mannina, S. Addas, S. Sibley, D. Pichora, D. Maslove, P. Abolmaesumi et al., “Domain knowledge is power: Leveraging physiological priors for self-supervised representation learning in electrocardiography,” IEEE Transactions on Biomedical Engineering, 2026, DOI: 10.1109/TBME.2026.3656904.

[30] W. Chen, H. Wang, L. Zhang, and M. Zhang, “Temporal and spatial self supervised learning methods for electrocardiograms,” Scientific Reports, vol. 15, no. 1, p. 6029, 2025, DOI: 10.1038/s41598-025-90084-2.

[31] K. Ma, T. Zhang, H. Zhang, and W. Huang, “Selfsupervised contrastive learning achieves 12-lead ecg classification,” Biomedical Signal Processing and Control, vol. 112, p. 108420, 2026, DOI: 10.1016/j.bspc.2025.108420.

[32] Abdullah, Z. Fatima, H. A. Safder, M. Manzoor, C. G. Sánchez-Mejorada, M. J. Torres Ruiz, and R. Quintero Téllez, “Rhythmx™: An interpretable self-supervised contrastive learning framework for heartbeat classification,” Technologies, vol. 14, no. 3, p. 148, 2026, DOI: 10.3390/technologies14030148.

[33] Y. Zhang, X. Li, L. Zhang, J. Wang, S. Jiang, Y. Ma, and D. Li, “A sequential mae-clustering self-supervised learning method for arrhythmia detection,” Expert Systems with Applications, vol. 269, p. 126379, 2025, DOI: 10.1016/j.eswa.2025.126379.

[34] M. Chatterjee, A. D. Chan, and M. Komeili, “Toward robust automated cardiovascular arrhythmia detection using self-supervised learning and 1-dimensional vision transformers,” Scientific Reports, vol. 16, no. 1, p. 11793, 2026, DOI: 10.1038/s41598-026-41549-5.

[35] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PmLR, 2020, pp. 1597–1607, DOI: 10.48550/arXiv.2002.05709.

[36] S. Mavaddati, “Ecg arrhythmias classification based on deep learning methods and transfer learning technique,” Biomedical Signal Processing and Control, vol. 101, p. 107236, 2025, DOI: 10.1016/j.bspc.2024.107236.

[37] M. Niloy, M. Islam, M. Ullah, J. Alom, M. Ahmed, M. Mridha, and M. Hossen, “Lead-aware multi-resolution transformer with domain adaptation for beat-level ecg arrhythmia classification, vol. 6 (2025), ieee open journal of the computer society, issn: 2644-1268,” 2025, DOI: 10.1109/OJCS.2025.3637851.

[38] X. Wu, Z. Rezki, B. Balusu, J. Zhou, and J. Bai, “One model, dual tasks: a novel distributionally adaptive learning framework for ecg classification and generation addressing intra-and inter-patient variability,” Expert Systems with Applications, p. 130096, 2025, DOI: 10.1016/j.eswa.2025.130096.

[39] G. B. Moody and R. G. Mark, “The impact of the mitbih arrhythmia database,” IEEE engineering in medicine and biology magazine, vol. 20, no. 3, pp. 45–50, 2001, DOI: 10.1109/51.932724.

[40] E. Yakushenko, “St petersburg incart 12-lead arrhythmia database,” 2008, DOI: . [Online]. Available: https://physionet. org/content/incartdb/1.0.0/

[41] S. D. Greenwald, R. S. Patil, and R. G. Mark, Improved detection and classification of arrhythmias in noise-corrupted electrocardiograms using contextual information. IEEE, 1990, DOI: 10.1109/CIC.1990.144257.

[42] Association for the Advancement of Medical Instrumentation, “ANSI/AAMI EC57:2012/(R)2020: Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms,” Arlington, VA, USA, 2012, DOI: 10.2345/9781570204784.ch1.

[43] P. De Chazal, M. O’Dwyer, and R. B. Reilly, “Automatic classification of heartbeats using ecg morphology and heartbeat interval features,” IEEE transactions on biomedical engineering, vol. 51, no. 7, pp. 1196–1206, 2004, DOI: 10.1109/TBME.2004.827359.

[44] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017, DOI: 10.48550/arXiv.1706.03762.

[45] D. Chicco and G. Jurman, “The advantages of the matthews correlation coeficient (mcc) over f1 score and accuracy in binary classification evaluation,” BMC genomics, vol. 21, no. 1, p. 6, 2020, DOI: 10.1186/s12864-019-6413-7.

[46] T. O. Kvålseth, “Note on cohen’s kappa,” Psychological reports, vol. 65, no. 1, pp. 223–226, 1989, DOI: 10.2466/pr0.1989.65.1.223.