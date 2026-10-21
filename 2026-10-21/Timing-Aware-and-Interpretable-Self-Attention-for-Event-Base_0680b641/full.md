# Timing-Aware and Interpretable Self-Attention for Event-Based Wearable Task Transition Detection

Anda Ouyang, Student Member, IEEE, Siyuan Chen, Member, IEEE and Julien Epps, Senior Member, IEEE

Abstract— Task transition detection aims to identify when a person shifts from one task to another, providing a foundation for continuous health monitoring and timely context-aware interventions. While statistical and deep feature representations have demonstrated strong classification performance, they often rely on aggregated or latent features that are difficult to interpret in behavioural terms. Event-based features provide a compact and interpretable alternative by encoding salient behavioural and physiological patterns as discrete tokens with explicit semantic meaning, making them well suited for behavioural analysis in health informatics. Self-attention offers a natural mechanism for modelling interactions among such tokens, however conventional self-attention classifiers primarily rely on token order and do not explicitly incorporate real-valued timing or duration information. As a result, temporally informative cues present in event representations may not be fully exploited for task transition detection. This study proposes an event-based time-aware self-attention framework that represents behavioural signals as sequences of discrete events augmented with onset timestamps and durations. Event timing is encoded using a parameter-efficient kernel-based representation, while a novel type-aware duration embedding modulates information propagation according to event persistence. Experiments on a seminaturalistic wearable dataset show that the proposed model outperforms existing event-based approaches, achieves performance beyond or comparable with statistical and deep-learning models with a more compact input representation, and provides improved interpretability through integrated-gradient-based analysis. These results indicate that explicit modelling of event timing and duration enhances both classification performance and interpretability, supporting reliable deployment in biomedical and health informatics applications.

Index Terms— Event-based representation, attention attribution, integrated gradients, self-attention, task transition detection, time-aware modelling, wearable sensors

## I. INTRODUCTION

UMAN task transition refers to the shift of attention and behaviour from one task to another and is a fundamental   
component of daily human activity. Accurate detection of task   
transitions is a key enabler for many biomedical and health   
informatics applications, particularly in continuous monitoring

and context-aware assistive intervention systems. In elderly care, for example, reminders of the change of presbyopic eyeglasses in the transition between sedentary and physical tasks can help reduce the risk of accidental falls [1]. Furthermore, many existing task classification systems operate at the task level [2], [3], requiring tasks to be manually segmented during experiments—an approach that demands substantial human effort and is impractical for real-life systems. Automatic task segmentation based on transition point identification is not only helpful in reducing annotation effort in research studies, but essential for real-world task classification systems that must function without predefined task boundaries [4]. By identifying transitions between functional tasks, such systems enable quantitative analysis of patient activity patterns, therapy engagement, and recovery progression without manual labeling [5], [6].

Fixed sensors and wearable sensors are two main approaches to collect physiological and behavioural signals to detect task transition points. While fixed sensors like cameras and infrared motion sensors installed in the environment [7] provide better infrastructure integration in smart environments, e.g. smart house and car drive systems, wearable sensor devices, such as mobile phones [8], [9] and smart eyeglasses [1], [10] do not restrict the spatial activity range of users and are therefore more suitable for continuous daily-life monitoring. Among the physiological and behavioral signals collected from wearable sensors, body movement characteristics, for example linear acceleration and angular velocities collected from accelerometers and gyroscopes, and eye characteristics, for example blinking, saccade, and pupil diameter change collected from near-field cameras embedded into smart eyeglasses, were shown to be effective in task transition detection [1], [10], as they reflect both physical activity execution and cognitive workload.

Although conventional task transition detection systems primarily rely on statistical features [1], [8], [9] or deep features [11], [12], event-based features, which tokenize continuousvalued signals into discrete events, offer a representation that aligns more closely with how humans perceive and annotate task-related behaviours. Previous studies have shown that event-based features can achieve performance comparable to statistical features while providing enhanced interpretability [10]. Moreover, the discrete nature of event-based features enables the application of various Natural Language Processing (NLP) techniques. While the effectiveness of combining Bagof-Words with a conventional Support Vector Machine (SVM) has been validated [10], the potential of more advanced NLP models, such as neural network-based architectures, remains underexplored in the context of event-based task transition detection. In parallel, there is a growing trend across domains such as weather forecasting, electrical grid analysis, and traffic modeling [13], [14] to tokenize continuous time series into discrete deep-learned representations—not primarily for interpretability, but to enable compatibility with transformerbased architectures and to facilitate large-scale pre-training and cross-field transfer learning. These approaches have demonstrated consistent performance improvements in imputation, forecasting, anomaly detection and classification.

Among modern NLP techniques, self-attention-based models such as Transformers [15] and their variants have become dominant in recent research due to their ability to model token-level interactions and their compatibility with parallel computation on GPUs. However, event timing information, which has been demonstrated to be important for task transition detection [10], is typically absent from conventional NLP token representations. While several recommender systems in e-commerce have proposed mechanisms to incorporate event timestamps into self-attention-based models [16]–[19], such time-aware mechanisms have not yet been explored in the context of event-based task transition detection. Moreover, these models only encode timestamps and do not account for the duration of each event, which may also carry essential temporal cues.

Interpretability is critical for deploying wearable-based machine-learning systems in healthcare and human-centred bioinformatics, where model transparency supports clinical trust, system validation, and principled feature design. For task transition detection in healthcare and health informatics applications, understanding event-level contributions provides insight into how behavioural and physiological changes are used by the model, supporting transparent analysis of task transitions that may reflect changes in cognitive or physical state. Network analysis techniques such as Integrated Gradients (IG) [20], [21] enable the analysis of attention mechanisms to reveal token-level contributions, providing a principled basis for interpreting model decisions at the level of individual events. However, such attribution analyses have received limited attention in event-based task transition detection, particularly in conjunction with self-attention models designed to preserve temporal structure.

In this paper, we propose an Event-based Time-Aware Self-Attention model (EvtTiSA) for task transition detection. In addition to adopting the Mercer Kernel approach [16] to encode event timestamps, we introduce a novel encoding scheme for representing event durations within the self-attention framework. Furthermore, we apply the IG technique to analyse the contribution of individual input events to the model’s predictions, enhancing the transparency (interpretability) and reliability of event-based task transition detection in real-world scenarios.

## II. BACKGROUND AND RELATED WORK

## A. Task Transition Detection

Wearable-sensor-based task transition detection generally comprises data acquisition, feature extraction, and the deployment of machine learning algorithms. For data acquisition, accelerometers and gyroscopes can capture body movement signals such as linear and angular acceleration and velocity across different directions. These signals have been shown to effectively classify non-sedentary physical tasks and detect transitions between them [1], [4], [8], [10]. However, a large proportion of daily human activities are sedentary in nature, involving limited physical movement while requiring sustained cognitive engagement and frequent task switching. Such sedentary cognitive activities are common in modern lifestyles and are particularly relevant in health-related monitoring scenarios, where prolonged sedentary behaviour and cognitive workload have been associated with fatigue, reduced well-being, and long-term health risks [22], [23].

For sedentary tasks, electroencephalography (EEG) can provide a direct neurophysiological measure of task activity [24], however the application and maintenance of electrodes are relatively intrusive and less practical in daily settings. In contrast, eye-related features, collected non-invasively via near-field infrared cameras mounted on smart glasses [1], [10], can provide indirect indicators of cognitive effort and attention. These systems typically process infrared video from eye cameras to extract indicative features such as pupil diameter, pupil position, blink rate, and their temporal dynamics.

Most existing task transition detection systems adopt either statistical features or end-to-end deep features. Conventional machine learning models, such as Support Vector Machine [1], [8], k-Nearest-Neighbor [1], [9] and Decision Tree [1] rely heavily on statistical features, including but not limited to mean, standard deviation, skewness, and energies in different frequency bands. With the prevalence of neural networks, recent research is shifting the focus towards end-to-end deep features due to their demonstrated performance advantages [11], [12]. For example, Wang et al. [11] employed a CNN model to process body movement signals into deep features map, which were then fed into an LSTM to classify activity types and their transitions. Khan et al. [12] arranged their CNN and LSTM modules in parallel and constructed an ensemble by stacking the outputs of the last neural layer. However, deep feature representations are often suffer from limited interpretability and typically require a relatively large amounts of training data.

Event-based features, by contrast, offer improved interpretability compared to both statistical features and deep feature representations [10], [25]. Event-based features are obtained by tokenizing continuous-valued signals, such as angular velocities collected from gyroscopes or pupil diameters extracted from eye videos, into discrete labels. This representation enables compact longitudinal recording of behaviour, since hours of wearable data can be stored and analysed as sparse event sequences in contrast to fully processing every frame of signal data, which is particularly advantageous for resource-constrained wearable systems. Some eye-related features, including blinking, saccade and fixation, naturally occur as events [26]. For body movement, Chen and Epps [27] proposed three atomic head movement events, arguing that head motion is particularly informative for detecting sedentary task processing, where other body parts such as legs, hips, and chest tend to be less active. They derived three head events by determining whether the head is moving and whether the motion is accelerating or decelerating. Ouyang et al. [10] adopted the aforementioned eye and head features into task transition detection, and further proposed features to capture significant pupil size variations within predefined temporal windows. The effectiveness of events and their timing-related features - such as duration and/or inter-event intervals - has been demonstrated to achieve comparable performance to statistical features in various human task processing problems, including task load estimation [26], [27] and task transition detection [10]. Furthermore, human-understandable events facilitate visualisations and more intuitive analysis comparing behavioural patterns during task transition and non-transition periods. For example, saccades tend to become more frequent but shorter in duration near transition points, whereas head movement often remains relatively static for extended periods during nontransition segments [10]. However, the only existing work on event-based task transition detection relies on a conventional SVM model [10]. The application of more advanced NLP architectures and the analysis of individual event contributions within such models remain unexplored.

## B. Self-Attention Classifiers

Since the introduction of the high-performance transformer model [15], self-attention-based classifiers have become a dominant model for token sequence processing, including NLP. Although the key advantage of self-attention-based models lies in their ability to efficiently and effectively capture token interactions in large-scale unsupervised pre-training— enabling easy adaptation through transfer learning—they have also demonstrated effectiveness on small datasets without pretraining in various fields [28], [29]. In classification tasks, only the encoder component of the Transformer is typically employed, as it sufficiently models contextual relationships among input tokens [16], [21], [30]. In this subsection, we review the core architectural components of self-attention models and the standard positional encoding formulations, which will serve as the basis for analysing their timing representations in subsequent sections.

The core component of a transformer is the self-attention block. Given $\bar { X ^ { ' } } = E ^ { t o k e n } \in \mathbb { R } ^ { N \times d _ { x } }$ representing the input sequence where each row $E _ { i , : } ^ { t o k e n }$ contains an embedded token, and $W ^ { Q } , W ^ { K } , W ^ { V } \in \mathbb { R } ^ { d _ { x } \times \ddot { d } _ { k } }$ as parameter matrices, the selfattention can be formulated as:

$$
Q = X W ^ { Q } , K = X W ^ { K } , V = X W ^ { V }\tag{1}
$$

$$
A = \operatorname { s o f t m a x } \left( S \right) = \operatorname { s o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } } \right)\tag{2}
$$

$$
H = A V\tag{3}
$$

where the ith row jth column element of the A matrix, denoted $A _ { i , j } .$ , is the interaction from token i to token $j$ and $A _ { j , i }$ is vice versa: $A _ { i , j }$ is the inner product of linearly transformed token i and token j after a per-row softmax normalization, further multiplied with the token $j$ in the value matrix in equation (3).

The self-attention mechanism described above is inherently order-invariant, meaning that it cannot capture the positional relationships between tokens in a sequence. To address this limitation, positional encodings can be incorporated to inject information about token order into the model. Two widely adopted approaches are absolute positional encoding and relative positional encoding. Absolute positional encodings [15], [30] represent the absolute index of each token as a vector, which is then added to or concatenated with the token embeddings in X in equation (1). For the commonly used additive formulation, the input to the self-attention layer becomes

$$
X = E ^ { t o k e n } + E ^ { p o s }\tag{4}
$$

Relative positional encodings [31] instead embed the position difference between tokens into $\bar { M } \in \mathbb { R } ^ { N \times N \times d _ { x } }$ such that $M _ { i , j , : } = { \mathrm { e n c o d e } } ( | i - j | )$ , and reformat equations (2) and (3) into:

$$
{ \cal A } = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } + Q \bullet M } { \sqrt { d _ { k } } } \right)\tag{5}
$$

$$
H = A V + A \bullet M\tag{6}
$$

where • denotes a last-dimension dot product. Specifically, $( Q \bullet M ) _ { i , j } = Q _ { i , : } M _ { i , j , : } ^ { \top }$ , and $( A \bullet M ) _ { i , j } = A _ { i , : } M _ { i , j , : } ^ { \top }$

With the absolute or relative positional encoding, $A _ { i , j }$ becomes a learned function of both token content and positional information, rather than a pure similarity measure in embedding space. This positional dependency fundamentally alters the structure of the attention matrix, enabling the model to encode order-sensitive and distance-aware interaction patterns that cannot be represented by content-only self-attention.

## C. Engaging Timing Information

Standard attention-based NLP models primarily encode the sequential order of tokens and do not explicitly account for real-valued timing information, which could be useful for modelling task transition. In natural language, tokens are treated as discrete symbolic units without associated physical timestamps; consequently, modelling their positional order is typically sufficient for capturing syntactic and semantic dependencies. Nevertheless, timing information has been explicitly incorporated in temporal recommender systems, such as those for movies, music, and e-commerce [16]–[19]. In these application contexts, user interactions—such as ratings, clicks, or purchases—occur at irregular and heterogeneous time intervals, and modelling such temporal dynamics has been shown to improve recommendation performance.

Existing methods for incorporating timing information are derived from the absolute and/or relative order positional encoding. Xu et. al. [16] proposed a parameter-efficient Mercer kernel to embed arbitrary absolute timestamps into a Fourier cosine map, and Tran et. al. [18] further fused this formulation with an LSTM to capture long-term dependencies of events. Li et. al. [17] proposed a relative timing encoding scheme (TiSASRec) by replacing the order position differences with quantized inter-event time intervals in Eq. (5), such that $M _ { i , j , : } = { \mathrm { L i n e a r E m b e d } } ( { \mathrm { q u a n t i z e } } ( | t _ { i } - t _ { j } | ) )$ . Building upon this idea, Cho et. al. (MEANTIME) [19] replaced the linear embedding with non-linear sinusoidal, logarithmic, and exponential kernels to model non-linear temporal interactions. Incorporating timing information in these models has been shown to improve recommendation performance in an ecommerce context, as measured by hit rate, to different degrees.

However, to the best of the authors’ knowledge, none of these timing-aware self-attention methods have been applied to human task processing problems, including task transition detection. Moreover, because events in recommender systems are typically represented as instantaneous timestamped interactions, none of the aforementioned self-attention methods account for the event duration. In contrast, event duration has been shown to provide discriminative information in human task processing [10]. However, how to incorporate event duration into self-attention classifiers remains an open problem.

## D. Analysis of Attention via Attribution Methods

Attribution analysis aims to quantify how input features or intermediate representations contribute to a model’s output. Early approaches such as gradient-based saliency maps [32], Layer-wise Relevance Propagation (LRP) [33], and Grad-CAM [34] primarily provide relative importance patterns and lack a well-defined absolute scale. Consequently, their attribution values are most meaningful when compared across features or layers, rather than as direct measures of contribution to the model output.

Integrated Gradients (IG) [20] address this limitation by introducing an attribution framework grounded in a completeness property. For a system $y \ = \ F ( { \pmb x } )$ with input or intermediate variable x and a reference point $\mathbf { x } ^ { \prime }$ , the integrated gradient is defined as

$$
\operatorname { I G } ( { \pmb x } ) : = ( { \pmb x } - { \pmb x } ^ { \prime } ) \odot \int _ { 0 } ^ { 1 } \frac { \partial F \big ( { \pmb x } ^ { \prime } + \alpha ( { \pmb x } - { \pmb x } ^ { \prime } ) \big ) } { \partial { \pmb x } } d \alpha\tag{7}
$$

where ⊙ denotes element-wise (Hadamard) multiplication. The integrated gradient satisfies the completeness property [20], such that the sum of IG across all dimension equals the output difference:

$$
\sum _ { i } \operatorname { I G } ( \pmb { x } _ { i } ) = F ( \pmb { x } ) - F ( \pmb { x } ^ { \prime } )\tag{8}
$$

This property enables each attribution component to be interpreted as a quantifiable contribution to the output change relative to the chosen reference point $\mathbf { x } ^ { \prime }$

For Transformer-based models, interpretability is often analysed through attention-weight visualisations [35], [36]. However, several studies have questioned whether raw attention scores reliably reflect feature importance [37], [38]. One contributing factor is that attention outputs are subsequently transformed by learned projections and feedforward layers, weakening the direct correspondence between attention magnitude and final prediction.

To address this limitation, Hao et al. [21] extended Integrated Gradients to the self-attention mechanism by treating the attention matrix as an intermediate activation and computing gradient-based attributions with respect to it. Unlike raw attention-weight visualisation, this formulation attributes attention elements based on their propagated effect on the final model output, which can be written as:

$$
{ \mathrm { A t t r } } ( A ^ { i _ { \mathrm { h e a d } } } ) : = A ^ { i _ { \mathrm { h e a d } } } \odot \int _ { 0 } ^ { 1 } { \frac { \partial F { \bigl ( } \alpha A ^ { i _ { \mathrm { h e a d } } } { \bigr ) } } { \partial A ^ { i _ { \mathrm { h e a d } } } } } d \alpha\tag{9}
$$

Their results show that IG-based attention attributions better reflect the functional importance of attention heads than raw attention weights, particularly in the context of multi-head pruning. Despite this advance, existing attention attribution methods implicitly assume that the chosen reference point corresponds to a consistent neutral model state [20]. Specifically, the reference output $F ( { \pmb x } ^ { \prime } )$ should represent the absence of contribution, such that positive and negative IG values can be interpreted as contributions toward and against a given prediction, respectively. Moreover, this reference output must be consistent across attribution instances, so that attribution values reflect output changes from a common baseline and are therefore directly comparable across different inputs or model components.

In standard Transformer architectures, however, this assumption is difficult to satisfy, as the model output generally depends on multiple parallel attention heads and residual pathways that remain active under common attribution settings. The architectural origins of this non-neutrality are discussed in detail in Section III.D. Consequently, the reference output may already reflect a non-neutral prediction state, meaning that the resulting IG attributions quantify changes relative to an already active baseline rather than the absence of contribution. This baseline dependency can introduce offsets in the attribution scores and undermine their interpretability. As a result, the completeness interpretation of IG-based attention attributions may no longer hold in practice, leaving open the question of how to obtain faithful and interpretable attention attributions under realistic architectural settings. Nevertheless, this issue can be addressed through careful architectural design that enforces a well-defined neutral reference state.

## III. METHODS

In this paper, we propose a self-attention-based classifier to combine event tokens, timing, and duration. As shown in Fig. 1, the grey box area is the structure of a conventional self-attention classifier, from which basis we 1) replace the order position embedding with timing positional embedding in the structure below the grey box, to capture the timing interactions of events; 2) propose an event duration embedding system in the structure above the grey box, to engage event duration information into the classfier. More details and reasoning for the structure design are explained in the following subsections. The inputs to the system are $[ e _ { 1 } , e _ { 2 } , . . . , e _ { N } ]$ a list of event onset token indices for a sliding window, $[ t _ { 1 } , t _ { 2 } , . . . , t _ { N } ]$ , where each $t _ { i }$ represents the timing position or onset timestamps of $e _ { i } ,$ and $[ d _ { i } , d _ { 2 } , . . . , d _ { N } ]$ , where each $d _ { i }$ represents the timing duration of event $e _ { i }$ . The self-attention output is passed through a two-layer feedforward network with ReLU activation, followed by a sigmoid activation to produce the final score $y \in [ 0 , 1 ]$ . A threshold is applied to this score for binary classification.

![](images/a02212c37a880993103110b9d03864739deb1507c11d0938cd29252542c7fd90.jpg)  
Fig. 1. Overview of the proposed EvtTiSA system that combines events, event timestamps, and event duration information in attention. The structure inside the light grey box corresponds to a conventional self-attention classifier (with the skip connection from X to H removed to enable integrated-gradient-based attribution analysis). EvtTiSA (1) replaces order positional embeddings with timing positional embeddings to capture temporal interactions between events, (2) incorporates event duration information D into the value matrix $V ,$ and (3) enables post-hoc IG-based Attribution Analysis for attention mechanism interpretability.

## A. Signal Tokenization: Event-based Features

Event-based features aim to represent continuous-valued physiological and behavioural signals as discrete, semantically meaningful events. Such representations not only improve interpretability but also enable the application of a wide range of NLP models. In prior work, event-based features were commonly extracted by converting continuous signals into frame-based binary indicator sequences [10], [27]. In this representation, each event type is encoded as a binary channel over a fixed temporal grid with the same temporal resolution as the original signals. While this representation is convenient for event detection and visualisation, modern NLP models typically operate on token sequences rather than multidimensional time-aligned arrays.

Representing behavioural data as a sequence of event tokens enables direct attribution of model outputs to individual events. In contrast, frame-based binary indicator sequences represent a single event across multiple consecutive time frames, causing its influence to be distributed over many input positions and thereby diluting its attribution and reducing interpretability. Another advantage is that event-token sequences reduce input dimensionality because the number of inputs no longer scales with event duration, thereby lowering computational cost and improving statistical efficiency. Although some existing approaches further convert binary indicators into event-onset sequences, they often discard event duration or relative timing information, which are essential for event-driven sequence modelling. To address these limitations, we propose a temporal event representation in which each event is explicitly characterised by its type and timing information.

The proposed temporal event sequence is constructed from event indicator sequences. Following [10], we extract ten event indicator sequences corresponding to head movement, eye movement, and pupil dynamics:

• head central, head increase, head decrease: Head movement events are extracted using the algorithm proposed by Chen and Epps [27]. Let $\omega _ { h }$ denote the angular velocity magnitude of head movement. A threshold of $5 ^ { \circ } / s$ is applied to distinguish voluntary head movement from involuntary micro-oscillations. Head movement events are defined as

$$
e ^ { h } = \left\{ \begin{array} { l l } { h e a d _ { - } c e n t r a l } & { \mathrm { i f ~ } \omega _ { h } \leqslant 5 ^ { \circ } / s } \\ { h e a d _ { - } i n c r e a s e } & { \mathrm { i f ~ } \omega _ { h } > 5 ^ { \circ } / s , t ^ { p c } < t < t ^ { p k } } \\ { h e a d _ { - } d e c r e a s e } & { \mathrm { i f ~ } \omega _ { h } > 5 ^ { \circ } / s , t ^ { p k } \leqslant t < t ^ { n c } } \end{array} \right.\tag{10}
$$

where $t ^ { p c }$ and $t ^ { n c }$ denote the times of the previous and next head central events, and $t ^ { p k }$ denotes the time of peak angular velocity.

• is blink, not blink: Blinking is detected by the eyelandmark algorithm [39].

• is saccade, not saccade: Saccade and fixation are extracted from the dispersion-based algorithm [40].

• pupil size central, pupil size increase, pupil size decrease: Pupil size changes are detected by identifying diameter variations exceeding 7% within 2 s, following the principle of threshold-based change detection in [10], using a relative change criterion.

For visualisation purposes, each event type is denoted by an abbreviation listed in Table I. After event extraction, we obtain 10 frame-based binary-valued event indicator sequences, one for each event type, indicating whether the corresponding event is active at each time sample.

Fig. 2 illustrates the conversion from continuous sensor signals to event-based representations. In (a), the dispersionbased algorithm identifies a saccade within the time interval bounded by the red and purple dashed lines. The corresponding event indicator sequence in (c) shows the $s +$ (is saccade) indicator activated during this interval, with its complement s− (not saccade) inactive. Similarly, (b) shows a head movement segment in which head acceleration (h+) and deceleration (h−) phases are identified, whose corresponding indicator sequences are also shown in (c). These examples visually illustrate how continuous-valued signals are converted into binary-valued event indicator sequences prior to temporal event construction.

Rather than using the binary indicator sequences directly, we group each contiguous active segment into a temporal event instance characterised by its event type, onset time, and duration, as shown in Fig. 2(d). Specifically, each event instance is represented as a tuple $( e _ { i } , t _ { i } , d _ { i } )$ , where $e _ { i }$ denotes the event type, $t _ { i }$ the onset timestamp, and $d _ { i }$ the event duration. The resulting representation is an ordered sequence of temporal event tuples $\{ ( e _ { i } , t _ { i } , d _ { i } ) \} _ { i = 1 } ^ { N }$ , sorted by ascending $t _ { i } ,$ which serves as the input to the proposed EvtTiSA model. This representation aligns with the tokenbased processing paradigm of self-attention models, facilitates event-level attribution analysis, and explicitly preserves timing information for subsequent encoding. For example, the temporal event sequence in Fig. 2(d) is {(s+, 43.70s, 0.672s), $( h + , 4 3 . 7 2 s , 0 . 1 2 s )$ , (h−, 43.83s, 0.57s), $( s - , 4 4 . 3 7 s , d _ { s - } )$ $( h - , 4 4 . 4 0 s , d _ { h - } ) \}$

TABLE I  
EVENT TOKEN ABBREVIATIONS
<table><tr><td>Event Type</td><td>Abbrev.</td><td>Event Type</td><td>Abbrev.</td></tr><tr><td>head_central</td><td>h0</td><td>pupil_size_central</td><td> $p 0$ </td></tr><tr><td>head_increase</td><td>h+</td><td>pupil_size_increase</td><td> $^ { p + }$ </td></tr><tr><td>head_decrease</td><td> $h -$ </td><td>pupil_size_decrease</td><td> $p -$ </td></tr><tr><td>is_blink</td><td> $^ { b + }$ </td><td>is_saccade</td><td>s+</td></tr><tr><td>not_blink</td><td> $b -$ </td><td>not_saccade</td><td>S-</td></tr></table>

## B. Timestamp Embedding: Mercer Kernel

To embed the non-uniform timing positions or timestamps of events, we adopt the Mercer timing kernel proposed in e-commerce by Xu et al. [16] for its parameter efficiency, approximate translation invariance, positive semidefiniteness, and demonstrated effectiveness in modelling temporal dependencies. An arbitrary timestamp t is mapped to a vector to match the dimensionality of the event token embeddings:

$$
\mathcal { K } ( t ) = [ \sqrt { c _ { 2 i } } \cos ( \omega _ { i } t ) , \sqrt { c _ { 2 i + 1 } } \sin ( \omega _ { i } t ) ] _ { i = 1 } ^ { d _ { x } }\tag{11}
$$

![](images/4215606ea8e8a8aabc77bbdf929aa93c7357aa9d347d61953f322fc556709c97.jpg)  
(a)

![](images/c1a31334d8363ea422ffcedba22210b110671e623fca12d338af7c9c7914fdd1.jpg)

(b)  
![](images/d3deec89c4bdcb603d81316b599ef62751f8008f6ed209df17d23c73046b336c.jpg)  
(c)

![](images/66dea0007cba4e4c89645b282922344f8e0503909e9e6eb711c1a4ce7234758f.jpg)  
Fig. 2. Converting continuous-valued sensor-derived feature streams into temporal events. (a) Pupil center position time series (pixels). (b) Head angular velocity time series (deg/s). (c) Uniform-time binaryvalued event indicator sequences $( \mathbf { 0 } / \overset { . } { \mathbf { 1 } }$ ; false/true for each event type at each time sample), extracted from the continuous signals in (a) and (b); contiguous runs of ones correspond to event instances. (d) Temporal event sequence obtained by grouping the indicator sequences in (c), serving as the input to the proposed EvtTiSA. Each event instance is represented as a tuple $( e _ { i } , t _ { i } , d _ { i } )$ , where $e _ { i }$ is the event type, $\mathbf { { \boldsymbol { t } } } _ { i }$ is the onset timestamp, and $\mathbf { \mathbf { { \alpha } } } _ { d _ { i } }$ is the duration; the model input is the ordered sequence $\{ ( e _ { i } , \dot { t } _ { i } , d _ { i } ) \dot  \} _ { i = } ^ { N }$ 1 in ascending temporal order of $\mathbf { \Delta } _ { t _ { i } }$

The Fourier coefficients $c _ { i }$ are learnable parameters that adaptively weight the importance of different temporal frequencies. The frequencies $\omega _ { i }$ can be fixed as hyperparameters or can be jointly optimized with the network parameters; in either case they need to be initialized across a wide range (e.g. logarithmic scale [16]) to capture interactions across multiple temporal scales. This practice can be understood in light of the highly non-convex nature of frequency learning, in which multiple local optima can arise. As an example, for periodic signals, frequencies at integer multiples of a base frequency (harmonics) can act as local minima, causing optimization to stall near them.

Given a sequence of event timestamps $\pmb { t } = [ t _ { 1 } , t _ { 2 } , \ldots , t _ { N } ]$ the Mercer kernel produces a timestamp embedding matrix

$$
E ^ { t } = \boldsymbol { \mathcal { K } } ( t )\tag{12}
$$

The timestamp embedding $E ^ { t }$ is added to the embedded event onset tokens $E ^ { e v t }$ to form the query and key matrices, so

that the resulting attention scores will capture the interactions of both event type and event timing:

$$
\boldsymbol { Q } = ( E ^ { e v t } + E ^ { t } ) \boldsymbol { W } ^ { Q } , \boldsymbol { K } = ( E ^ { e v t } + E ^ { t } ) \boldsymbol { W } ^ { K }\tag{13}
$$

$$
S _ { i , j } = \frac { Q _ { i , : } K _ { j , : } ^ { \top } } { \sqrt { d _ { k } } } = \frac { ( E _ { i , : } ^ { e v t } + E _ { i , : } ^ { t } ) W ^ { Q } ( ( E _ { j , : } ^ { e v t } + E _ { j , : } ^ { t } ) W ^ { K } ) ^ { \top } } { \sqrt { d _ { k } } }\tag{14}
$$

## C. Duration Embedding

The effectiveness of event durations has been demonstrated in [10]. However, to the best of the authors’ knowledge, explicit duration embeddings have not been explored in selfattention-based sequence models. Existing timing-aware selfattention architectures have primarily been developed in the context of recommender systems [16], [17], [19], where event sequences (e.g., shopping or website click logs) are typically characterised by timestamps but lack well-defined event durations.

Rather than simply adding or concatenating the event durations to some layers of the system, we propose a type-aware duration embedding scheme, in which duration information is modulated according to event type. As shown by the section above the grey block in Fig. (1), each event onset token is first embedded into a duration-type-specific representation $E _ { i , : } ^ { d }$ , which is then scaled by a nonlinear function of the event duration $d _ { i }$ . We employ a hyperbolic tangent function to introduce nonlinearity and prevent long-duration events from overwhelming the short-duration ones:

$$
D _ { i , : } = \operatorname { t a n h } ( \alpha d _ { i } ) E _ { i , : } ^ { d }\tag{15}
$$

where α is a learnable scaling parameter. The resulting duration embedding matrix D is added to the value matrix V:

$$
V = E ^ { e v t } W ^ { V } + D\tag{16}
$$

so that event duration influences the amount of information propagated through attention, while the attention weights themselves remain determined by event semantics and timing.

## D. Attention Attribution with a Neutral Reference

Although IG-based attention attribution provides a principled alternative to raw attention analysis, its faithfulness depends on the choice of reference point. For the completeness property of Integrated Gradients to admit a meaningful interpretation, the reference output should correspond to a neutral and consistent model state, so that attribution values represent contributions relative to the absence of influence and identical attribution magnitudes correspond to identical output shifts. In Eq. (9), Hao et al. [21] implicitly adopt $A ^ { \prime } = \mathbf { 0 }$ as the reference attention matrix by scaling one attention head at a time while keeping all remaining heads unchanged. However, in standard Transformer architectures this choice does not correspond to a consistent neutral model state, thereby undermining the completeness interpretation of the resulting attributions.

The non-neutrality of the reference point $A ^ { \prime } = \mathbf { 0 }$ arises from two architectural factors. First, in multi-head attention, masking or scaling a single head while leaving the remaining heads active produces a reference output that still contains non-zero attention contributions. Consequently, the reference output $F ( A ^ { \prime } = \mathbf { 0 } )$ varies across heads, and the completeness property no longer holds consistently. Second, many selfattention-based models incorporate residual or skip connections that bypass the attention module, allowing the input token representations X to directly influence downstream feedforward layers [15]–[17], [31]. Even when ${ \cal A } \ = \ { \bf 0 } ,$ these bypass pathways can contribute substantially to the final output, further violating the assumption that $F ( \mathbf { 0 } )$ represents a neutral baseline.

To ensure a well-defined neutral reference for attention attribution, we impose two architectural constraints. First, we adopt a single-head attention mechanism, eliminating interhead interactions and ensuring that all attention contributions are simultaneously removed when $A = \mathbf { 0 }$ . Second, we remove the skip connection from the input X to the post-attention representation H (Fig. 1), such that the model output depends solely on the attention-mediated pathway. Under these conditions, setting $A \ = \ \mathbf { 0 }$ yields a fixed neutral reference output $F ( \mathbf { 0 } )$ , and the integrated gradients computed in Eq. (9) faithfully quantify the contribution of each attention element to the final prediction.

## IV. EXPERIMENTAL SETUP

## A. Dataset and Pre-processing

Experiments were conducted on a semi-naturalistic dataset in which the everyday activities of 11 participants were recorded in a laboratory environment for durations ranging from 38 to 89 minutes $( \mathrm { M } = 6 6 . 1 , \mathrm { S D } = 1 7 . 8 )$ [10]. Informed consent was signed by each participant prior to the experiment and all the processes were approved by the UNSW Human Research Ethics Committee (approval date: 9 September 2014; HC No. 08/2014/23). To enable data collection without constraining participants’ behaviour, an unobtrusive eyeglassesbased device, the Pupil Labs Pupil Core [41], was employed. The device integrates near-field infrared cameras for eye video capture, along with a triaxial accelerometer and gyroscope for recording head movements. Compared with tightly controlled experimental task designs [11], [12], this dataset captures a broader range of spontaneous behaviours that more closely reflect natural activity patterns. The most frequent activities during the recordings were sedentary and included, but were not limited to, laptop use, mobile phone use, reading, and writing. In total, the dataset comprises 12 hours and 7 minutes of recordings, with 725 manually annotated task transitions as ground truth.

To prepare the multimodal data for analysis, time-series features were extracted from the eye videos and inertial signals were preprocessed for temporal alignment. An eye-landmark detection algorithm [39] was used to derive pupil diameter, pupil centre positions, and blink events from the raw eye videos. Linear and angular accelerations and velocities from the accelerometer and gyroscope were resampled to 60 Hz to match the sampling rate of the eye-derived features. The event detection algorithms described in Section III-A were then applied to convert the continuous signals into sequences of discrete events.

## B. Task Formulation and Evaluation Protocol

Task transition detection was formulated as a binary classification problem using a sliding-window approach, where each window was labelled according to whether a task transition occurred within it. Following [10], a window length of 2 seconds and a step size of 1 second were adopted. To alleviate class imbalance, transition windows were oversampled by reducing the step size to 4 frames (66.67 ms), resulting in transition samples accounting for 33.7% of the total 63,113 sliding windows. Performance was evaluated using 11-fold leave-oneparticipant-out cross-validation.

To evaluate the effectiveness of timestamp embedding, we compared the Mercer timing kernel [16] in Eq. (11) with the sinusoidal positional encoding [15] for representing temporal order and timing information. The embedding dimension was swept from 20 to 120 in increments of 20 to identify a suitable dimensionality. To reduce the influence of random initialisation, each configuration was trained 15 times with different random seeds, and the average performance was reported.

As for the analysis of event attribution, integrated gradients were computed for each trained model and averaged across the 165 models obtained from 11-fold leave-one-participant-out cross-validation with 15 random seeds per fold. The consistency and reproducibility of attribution values across models were quantified using the one-way random-effects intraclass correlation coefficient (ICC) [42]. Specifically, ICC(1,1) was used to assess the reliability of attribution values from a single trained model instance, while ICC(1,k) was used to evaluate the reliability of the averaged attribution across all 165 model instances. This formulation is appropriate as each trained model can be regarded as a random realisation of the same training procedure. In addition, attribution-based feature pruning experiments were conducted by retaining either the most contributing or the least contributing features, in order to evaluate whether the attribution scores reflect meaningful feature importance.

The Area Under the Receiver Operating Characteristic Curve (AUC) was adopted as the primary evaluation metric, as it is independent of decision threshold selection. Accuracy and F1 score were additionally reported to provide complementary perspectives on classification performance. Reported results correspond to the mean performance across the 11 leave-oneparticipant-out folds, with standard deviations reflecting interparticipant variability.

Paired Wilcoxon signed-rank tests were conducted on the fold-wise AUCs to assess whether performance differences between EvtTiSA and reference models were statistically significant.

## C. Model Configuration and Implementation

Regarding hyperparameter settings, the angular frequencies $w _ { i }$ for both the Mercer kernel and the sinusoidal encoding were fixed on a logarithmic scale from $1 \times 1 0 ^ { - 4 } ~ \mathrm { t o } ~ 1$ , following [15], enabling the models to capture interactions across both short and long temporal scales. All Fourier coefficients $c _ { i }$ in the Mercer timestamp embedding formulation in Eq. (11) were initialised to 1. The duration scaling parameter α in Eq. (15) was initialised to 0.1 so that the duration embedding does not saturate for durations shorter than approximately 15 seconds at initialisation. All remaining parameters were randomly initialised, with their exact values determined by the random seed.

![](images/159711e33d61b1a7fd32ed61f51c40a2ec4c85096042a33e97b2089ccb4817b2.jpg)  
Fig. 3. AUC performance comparison of Mercer and sinusoidal encoding for timing and order information, and no position embedding baseline, with the variation of embedding dimensions. The solid lines are the average performances across 15 different random seeds, and the light colour regions are ± standard deviation. All models, except for the ”event indicator seq” (uniform sampled event indicator sequence), use only event onsets occurring at irregular (non-uniform) timestamps. In contrast, the uniform timestep model uses the full event sequence sampled at a fixed rate of 60 Hz.

All models were trained on NVIDIA V100 GPUs on the Gadi supercomputing cluster at the National Computational Infrastructure (NCI), Australia. The Adam optimiser was employed due to its robustness and stable convergence. As 99% of the sliding windows contained fewer than 30 events, event sequences were truncated to a maximum length of 30 and zero-padded where necessary to form fixed-length batches. To exclude padded tokens from attention computation, negative infinity was applied to the corresponding entries in the score matrix S, and zero values were applied in the attention matrix $A ,$ i.e., $S _ { i , j } ~ = ~ - 1 \times 1 0 ^ { 9 }$ and $A _ { i , j } ~ = ~ 0$ whenever either $e _ { i }$ or $e _ { j }$ corresponded to a padded token. Early stopping was applied with a tolerance of 0.5% loss change over 10 consecutive epochs, with training typically converging within 40–80 epochs.

## V. RESULTS AND DISCUSSION A. Comparison of Order Embeddings and Timing Embeddings

We conducted an ablation study of the proposed EvtTiSA model to examine how order-based and timing-based position and its embedding dimensions impact performance, by comparing sinusoidal embeddings and Mercer kernel embeddings across different embedding dimensions. As shown in Fig. (3), the performance of all embedding schemes becomes stable for embedding dimensions greater than or equal to 40, indicating that these dimensions are sufficient to capture the information contained in event types together with their temporal or ordinal positions.

![](images/f853f02e1c65dc106b6407381162d04bbc976c372bf6e4d7531353f19afb0fc0.jpg)

![](images/2e016d8fecd3770bef598bd1858c466de2f61a0dd78e5a7272be2e2f4cf482a3.jpg)  
(d)

(a)  
![](images/0d552ec66812580163002c46c1659a4ed5af5b5d6b683f5c42aebf6e8b05dd69.jpg)  
(b)

![](images/61d19e6ef3ba67430d13efa987811d246f2f8d66fff8961ad3d9fedc95628b58.jpg)  
(e)

![](images/7e629e4ff1340c6e25bb6e7496388f0111c76acfa7b65c494412f98d09b4433c.jpg)  
(c)

![](images/71f610d34651c5f622136ee3a16668475debb5546ae8cf2eb17568f7d329b11a.jpg)  
(f)  
Fig. 4. Attribution insights, where a) is the integrated-gradient-based attribution heatmap of a non-transition sliding window, b) is the duration of each event onset of the same sliding window, and c) is the timestamps of each event onset of the same sliding window. d) e) f) are those of a transition sliding window

From the ■, ▼, ♦ and ▲ curves, timing embeddings consistently outperformed order embeddings under both encoding mechanisms, while the Mercer kernel consistently outperformed the sinusoidal embedding for both timing- and orderbased representations across embedding dimensions. Although the resulting performance gains are modest, both effects are systematically observed, suggesting that the absolute timing of event onsets captures marginally richer task transition–related information than relative ordering alone, and that the Mercer kernel provides a more flexible temporal representation.

Inspection of the learned Mercer kernel coefficients reveals a smooth, scale-dependent structure that consistently emerges across cross-validation folds and random initialisations, with slightly greater emphasis on mid-scale temporal variations (approximately 0.3–2 s) and modest attenuation of higherfrequency variations (approximately 0.02–0.2 s). This suggests that discriminative information in the dataset is not concentrated in instantaneous micro-fluctuations, but rather in short-range temporal grouping of events spanning sub-second to few-second intervals. The improvement over sinusoidal encoding therefore likely arises from the ability to adaptively suppress very high-frequency variations while modestly emphasising intermediate temporal dependencies.

To further examine whether temporal event tuples (Fig. 2(d)) can sufficiently capture the information contained in uniformly sampled event indicator sequences (Fig. 2(c)), we compared in Fig. 3 the performance of the uniform time-spaced event indicator sequence model (•) with models that rely solely on event onsets and their timestamps or orders. We encoded the uniform timestamps of the event indicator sequence using the Mercer kernel to ensure consistency with the proposed timing representations and to enable a fair comparison. When compared with the event-onset-only model using the same Mercer timestamp encoding (■), the AUC decreased by only 0.005.

By further incorporating event duration information through the proposed complete EvtTiSA (⋆) formulation in equations (15) and (16), the average AUC increased to 0.909 for embedding dimensions 20–80, slightly outperforming the event indicator sequence model. Importantly, this temporal event representation reduces the input size from a $1 0 \times 1 2 0$ event indicator sequence to a compact $3 \times 3 0$ tuple sequence per sliding window. Given the limited computational resources and memory capacity of wearable devices, this lightweight representation improves the practicality of real-time deployment in personal healthcare applications.

Taken together, these results indicate that stacking timingbased encoding, Mercer kernel representations, and explicit duration modelling leads to a cumulative performance gain. In particular, the proposed complete EvtTiSA model (⋆) achieves an average AUC of 0.909, outperforming a standard Transformer encoder with sinusoidal order-based positional encoding (▲) (0.883) as well as a variant without positional information (+) (0.877). This comparison highlights the benefit of jointly modelling event timing and duration using adaptive temporal embeddings, beyond what can be achieved by conventional order-based positional encodings alone.

## B. Analysis of Event Contributions: Attention Attribution in Self-attention

To analyse attention attributions, we averaged the integrated gradients across 165 trained models (15 random seeds of each of the 11 cross-validation folds). All analyses were conducted on the Mercer timestamps + duration models (EvtTiSA) with an input embedding dimension of 60, for which they achieved the best overall performance. For consistency analysis, each IG matrix was vectorised by concatenating its rows, and the 165 model instances were treated as independent random realisations of the same training procedure. ICC(1,1) and ICC(1,k) were computed for each sliding window to quantify singlemodel reliability and reliability of the averaged attribution, respectively. Averaging across the 63,113 sliding windows yielded a mean ICC(1,1) of 0.744 and a mean ICC(1,k) of 0.998, indicating substantial agreement among individual models and near-perfect reliability of the averaged attribution. We further observed that the baseline attention matrix $A ^ { \prime } = \mathbf { 0 }$ yields $F ( A ^ { \prime } ) = F ( \mathbf { 0 } ) = 0 . 4 2 4$ , which lies near the center of the output range [0, 1], suggesting that $A ^ { \prime } = \mathbf { 0 }$ provides a reasonable neutral reference in Eq. (7).

Fig. 4 presents representative examples of attention attribution analysis for transition windows (a)–(c) and non-transition windows (d)–(f). Panels (a) and (d) show the attribution heatmaps, (b) and (e) show event durations, and (c) and (f) show event timestamps. In these heatmaps (a) and (d), each element $\mathrm { A t t r } ( A _ { i , j } )$ quantifies the contribution of the attention weight from event token $e _ { i } \ t { \bf 0 } \ e _ { j }$ to the change in model output, $\Delta y .$ , indicated by color. Summing a row, $\textstyle \sum _ { j } \operatorname { A t t r } ( A _ { i , j } )$ aggregates the influence of $e _ { i }$ on all other events, whereas summing a column, $\textstyle \sum _ { i } \operatorname { A t t r } ( A _ { i , j } )$ , aggregates the influence of all events on the representation of $e _ { j }$

Across the dataset, column-wise attributions were more consistent than row-wise attributions. Column-wise analysis further revealed that long-duration head-central-movement event h0 often dominates the model decision when present. This observation is consistent with the manual annotation protocol, which relies on visible behavioural cues: annotated task transitions frequently co-occur with head movements (e.g., shifting from paperwork to laptop use typically involves raising the head to look at the screen). Such movements reduce the duration of the head-central state, thereby decreasing the prevalence of long-duration h0 segments around transitions. Fig. 4(a)–(c) illustrate a typical case, where the h0 column contributes strongly and negatively (−0.423) reducing the output to 0.001. When h0 duration is shorter, as in Fig. 4(d)– (f), other events such as $h + ~ :$ head increase and $b + :$ is blink exhibit non-trivial contributions.

![](images/25598754470ca8cea561cfc1c0dd97f0900353add31bf4fefea2d6252eb55639.jpg)  
Fig. 5. Effect of attribution-guided feature pruning on AUC performance. Models were trained using either the most contributive event types (highattribution features) or the least contributive event types (low-attribution features), as determined by Table II. High-attribution features achieve strong performance with few event types, whereas low-attribution features require substantially more event types to approach comparable performance.

The attribution heatmaps also enable feature pruning based on aggregated column-wise contributions. We adopt columnwise aggregation not only due to its higher empirical consistency, but also because each column corresponds to a specific event token whose influence is further modulated by the value matrix V that encodes the token content. Specifically, we summed column-wise attributions and aggregated them by event type to obtain the attribution contribution of each event type to $\Delta y .$ . These contributions were then averaged across sliding windows in which the corresponding event type occurred. The resulting rankings are reported in Table II.

TABLE II  
AVERAGE ATTRIBUTION PER SLIDING WINDOW  
$( y = 0 . 4 2 4 + \Delta y \in [ 0 , 1 ] )$ , AGGREGATED BY EVENT TYPE.
<table><tr><td>Event Type</td><td>Attribution</td><td>Event Type</td><td>Attribution</td></tr><tr><td>h0</td><td>-0.348</td><td>p0</td><td>-0.051</td></tr><tr><td>h+</td><td>0.286</td><td>p+</td><td>0.050</td></tr><tr><td>h-</td><td>0.157</td><td>p−</td><td>0.024</td></tr><tr><td>S-</td><td>-0.112</td><td> $b -$ </td><td>0.003</td></tr><tr><td> $^ { b + }$ </td><td>0.102</td><td>s+</td><td>0.001</td></tr></table>

According to Table II, we conducted feature pruning using two complementary strategies: retaining event types with the highest absolute attributions, and retaining event types with the lowest absolute attributions. We then evaluated the resulting models to assess how attribution-based feature selection affects classification performance. As shown by the blue curve in Fig. 5, which retains the highest-attribution features, a singlefeature model using only h0 achieved an AUC of 0.891. Retaining the top three contributing event types, all of which are head-motion events, increased AUC to 0.906, indicating that head events alone provide strong predictive power for task transitions.

TABLE III  
PERFORMANCE COMPARISON OF DIFFERENT MODELS ON THE TASK TRANSITION DETECTION DATASET. VALUES IN PARENTHESES INDICATE THE STANDARD DEVIATION OVER 11 LEAVE-ONE-PARTICIPANT-OUT CROSS VALIDATION FOLDS. AMONG THE CITED MODELS, ONLY THE SVM MODELS WERE EVALUATED ON THIS DATASET IN PRIOR WORK [10]; ALL OTHER MODELS WERE FRESHLY IMPLEMENTED AND TRAINED UNDER THE SAME EXPERIMENTAL PROTOCOL FOR FAIR COMPARISON. WILCOXON p-VALUES WERE OBTAINED FROM PAIRED SIGNED-RANK TESTS COMPARING EACH MODEL’S AUC AGAINST EVTTISA ACROSS THE CORRESPONDING CROSS-VALIDATION FOLDS.
<table><tr><td>Feature Type</td><td>Model</td><td>AUC</td><td>Accuracy</td><td>F1-Score</td><td>Wilcoxon test p-value vs. EvtTiSA</td></tr><tr><td>Statistical</td><td>SVM [10]</td><td>0.911(0.035)</td><td>82.8%(5.3%)</td><td>72.1%(7.1%)</td><td> $\overline { { 5 . 2 0 \times 1 0 ^ { - 1 } } }$ </td></tr><tr><td>Deep</td><td>TST [43]</td><td>0.918(0.032)</td><td>83.1%(4.8%)</td><td>74.6%(5.9%)</td><td> $8 . 3 0 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Deep</td><td>LSTM</td><td>0.915(0.032)</td><td>82.8%(5.1%)</td><td>73.9%(4.9%)</td><td> $6 . 7 4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Event + Timing</td><td>SVM [10]</td><td>0.888(0.052)</td><td>82.8% (5.5%)</td><td>72.1%(4.7%)</td><td> $\overline { { 1 . 8 6 \times 1 0 ^ { - 2 } } }$ </td></tr><tr><td>Event + Order</td><td>Transformer Encoder [15]</td><td>0.884(0.039)</td><td>79.0%(5.6%)</td><td>68.1%(3.4%)</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Event + Timing</td><td>TiSASRec [17]</td><td>0.894(0.036)</td><td>80.5%(5.0%)</td><td>70.6%(3.4%)</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> ${ \mathrm { E v e n t } } + { \mathrm { T i m i n g } }$ </td><td>MEANTIME [19]</td><td>0.899(0.034)</td><td>81.0%(5.0%)</td><td>71.8%(3.2%)</td><td> $9 . 7 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Event + Timing</td><td>EvtTiSA*</td><td>0.909 (0.031)</td><td>82.4% (4.6%)</td><td>73.9% (3.5%)</td><td></td></tr></table>

In contrast, a model retaining low-attribution features (orange curve), all of which are eye-related event types, achieved a substantially lower AUC of 0.776 despite including seven event types. Nevertheless, the most contributive eye events (s− and b+) can partially substitute for head-motion events: a model using the feature set $\{ h 0 , s - , b + \}$ achieved a comparable AUC of 0.900.

Beyond improving model transparency, the proposed attention attribution analysis provides a structured way to interpret behavioural information underlying task transitions. By identifying which event types dominate model decisions and validating their importance through attribution-guided feature pruning, this analysis supports a more principled understanding of how observable behaviours contribute to changes in human activity. Such insights are valuable for healthcare and health informatics applications, where they can inform sensor selection, reduce feature redundancy, and guide the design of efficient and interpretable monitoring systems for cognitive workload and functional-state assessment.

## C. Comparison ofModels

We compared the performance of the proposed EvtTiSA with various models using statistical, deep, and event-based features. As many of these models were originally evaluated on different datasets or in different research domains, we reproduced their implementations and trained them on our dataset following the descriptions in their respective papers. To the best of the authors’ knowledge, self-attention–based event models have not previously been explored for human task processing; accordingly, this work implements and evaluates multiple Transformer-based event classifiers, including a standard encoder with order-based positional encoding, timeaware variants with timestamp encoding, and the proposed EvtTiSA model.

For each cross-validation fold, performance was first averaged over 15 random seeds to reduce random initialisation variability; the reported mean and standard deviation were then computed across the 11 folds, and the resulting AUC-

ROC, Accuracy, and F1-score are reported in Table III. SVM results were re-computed using the same 11-fold leave-oneparticipant-out cross-validation protocol, whereas the original publication [10] reported concatenated-score evaluation. Given the substantial overlap in fold-level standard deviations, statistical significance was evaluated using paired Wilcoxon signedrank tests on the corresponding fold-wise AUCs; the resulting p-values are reported in Table III.

As shown in Table III, the proposed event-and-timingbased EvtTiSA achieves the strongest overall performance among event-based neural models, outperforming Transformer encoders with order-based positional encoding as well as timeaware self-attention models originally developed for eCommerce applications, including MEANTIME [19] and TiSAS-Rec [17]. These improvements may be attributed to both the parameter-efficient Mercer kernel and the explicit modelling of event duration. Paired Wilcoxon signed-rank tests on the foldwise AUCs confirm that these improvements over the eventbased baselines are statistically significant $( p < 0 . 0 5 )$

When compared with the event-based SVM model [10], EvtTiSA achieves comparable accuracy while providing consistent improvements in AUC-ROC and F1-score. On the one hand, this highlights the importance of incorporating event duration information in human task processing, as the SVM model also accounts for duration through handcrafted features. On the other hand, it suggests that self-attention–based models may require larger training datasets or effective transfer learning to achieve consistently dominant performance.

Nevertheless, we observe a clear improvement in AUC over the event-based SVM, which is statistically significant according to the paired Wilcoxon signed-rank test $( p \ : = \ : 0 . 0 1 8 6 )$ This indicates that EvtTiSA exhibits improved discriminative ability, assigning higher confidence scores to transition instances than to non-transition instances more consistently across decision thresholds. Such an improvement is especially relevant in the context of wearable task transition detection, where operating points may vary across applications and users, and different scenarios may prioritise precision or recall differently. A higher AUC therefore enables more flexible threshold selection at deployment time, improving robustness under varying class imbalance and practical usage conditions.

From an interpretability perspective, the SVM model relies on bag-of-N-gram features, which aggregate event patterns and make it difficult to attribute model decisions to individual events. In contrast, while neural-network-based models allow gradient-based attribution, they often lack a well-defined static neutral reference state, which complicates the faithful application of integrated gradients. By operating on explicit event tokens with well-defined temporal semantics, EvtTiSA provides a more interpretable representation that better supports event-level attribution analysis.

For non-event-based approaches, we considered three representative reference systems: (1) a statistical-feature-based SVM [10], (2) the Time Series Transformer (TST) [43], a state-of-the-art attention-based model for time-series processing, and (3) an end-to-end LSTM. As shown in Table III, statistical-feature-based and deep time-series models achieve performance comparable to that of EvtTiSA. In particular, the difference in AUC between EvtTiSA and the statistical SVM is not statistically significant according to the paired Wilcoxon signed-rank test $( p = 0 . 5 2 )$ , indicating comparable discriminative performance. Similarly, EvtTiSA and TST exhibit very similar results across evaluation metrics; although TST achieves a slightly higher mean AUC, the difference is not statistically significant (p > 0.05).

Despite using a substantially more compact event-based representation, EvtTiSA achieves performance on par with deep models operating directly on raw time series. This indicates that event-based representations augmented with timing information preserve most of the discriminative information contained in the original signals. Moreover, while deepfeature-based models such as TST and LSTM rely on latent representations that are difficult to interpret, EvtTiSA benefits from more transparent feature semantics and enables eventlevel attribution analysis, as discussed in Section V-B, without sacrificing performance.

It is worth noting that we also implemented several highperforming models previously proposed for human task transition detection, including a series-connected CNN–LSTM [11] and an ensemble CNN–LSTM [12]. In our experiments, however, these models exhibited strong overfitting on the present dataset, yielding AUC, accuracy, and F1 scores of approximately 80%, 73%, and 40% respectively, and are therefore not included in Table III. This behaviour is likely attributable to the large number of parameters introduced by the multilayer CNN components relative to the dataset size. Notably, the performance gains of these deep architectures over the SVM trained with event-based statistical features were similarly modest, further supporting the observation that deep learning models for human task processing may require larger datasets or effective transfer learning strategies to substantially outperform conventional approaches.

Despite these encouraging results, several limitations should be noted. First, the dataset comprises recordings from 11 participants in a semi-naturalistic laboratory setting, which may limit the generalisability of the findings to larger and more diverse populations. Second, although class imbalance was mitigated through transition-window oversampling, realworld deployment may involve different imbalance ratios and activity distributions. Evaluating the framework on larger-scale and fully in-the-wild datasets will therefore be an important direction for future research.

## VI. CONCLUSION

This work presented EvtTiSA, an event-based, time-aware self-attention framework for wearable task transition detection. By representing behavioural data as compact temporal event tuples and explicitly modelling both event onset timing and event duration, the proposed approach addresses key limitations of conventional self-attention models that rely solely on token order. The use of a Mercer kernel enables flexible and parameter-efficient timestamp encoding, while the proposed type-aware duration embedding allows persistent events to influence attention in a controlled and interpretable manner.

Across extensive cross-validated experiments, EvtTiSA achieved improved or comparable performance to representative statistical, deep, and event-based models, with strong discrimination as measured by the area under the receiver operating characteristic curve. Importantly, these results were obtained using a substantially more compact representation than uniformly sampled event indicators or raw time-series inputs, highlighting the efficiency advantages of temporal event modelling for wearable systems with limited computational resources.

Beyond performance, this work emphasised interpretability as a priority design objective. By enforcing a well-defined neutral reference state for attention, integrated-gradient-based attribution analysis was shown to yield stable and meaningful event-level explanations. The resulting attribution patterns aligned with known behavioural cues of task transitions and enabled effective attribution-guided feature pruning, supporting principled sensor and feature selection.

Overall, the findings demonstrate that incorporating explicit temporal structure—particularly event duration—into self-attention models enhances both predictive capability and transparency in human task modelling. While the current study focuses on task transition detection, the proposed framework is broadly applicable to other event-driven wearable physiological and behavioural analysis tasks. Future work will explore larger-scale datasets, cross-domain transfer learning, and pre-training strategies to further improve generalisability and unlock the full potential of time-aware event-based attention models.

## ACKNOWLEDGMENT

This work was supported in part by US Army Cooperative Agreement W911NF1920330. Opinions expressed are the authors’, and may not reflect those of the US Army.

## REFERENCES

[1] M. Raychoudhury, H. Yu, and J. Kiper, “Activisee: Activity transition detection for human users through wearable sensor-augmented glasses,” in Adjunct Proceedings ofthe 2022 ACM International Joint Conference on Pervasive and Ubiquitous Computing and the 2022 ACM International Symposium on Wearable Computers, 2022, pp. 371–378.

[2] F. Attal, S. Mohammed, M. Dedabrishvili, F. Chamroukhi, L. Oukhellou, and Y. Amirat, “Physical human activity recognition using wearable sensors,” Sensors, vol. 15, no. 12, pp. 31 314–31 338, 2015.

[3] W. Jiang and Z. Yin, “Human activity recognition using wearable sensors by deep convolutional neural networks,” in Proceedings ofthe 23rd ACM international conference on Multimedia, 2015, pp. 1307–1310.

[4] N. Alhammad and H. Al-Dossari, “Dynamic segmentation for physical activity recognition using a single wearable sensor,” Applied Sciences, vol. 11, no. 6, p. 2633, 2021.

[5] C.-Y. Chang, C.-Y. Hsieh, H.-Y. Huang, Y.-T. Wu, L.-C. Chen, C.-T. Chan, and K.-C. Liu, “Automatic functional shoulder task identification and sub-task segmentation using wearable inertial measurement units for frozen shoulder assessment,” Sensors, vol. 21, no. 1, p. 106, 2020.

[6] H. Nguyen, K. Lebel, P. Boissy, S. Bogard, E. Goubault, and C. Duval, “Auto detection and segmentation of daily living activities during a timed up and go task in people with parkinson’s disease using multiple inertial sensors,” Journal of neuroengineering and rehabilitation, vol. 14, no. 1, p. 26, 2017.

[7] K. D. Feuz, D. J. Cook, C. Rosasco, K. Robertson, and M. Schmitter-Edgecombe, “Automated detection of activity transitions for prompting,” IEEE transactions on human-machine systems, vol. 45, no. 5, pp. 575– 585, 2014.

[8] J.-L. Reyes-Ortiz, L. Oneto, A. Sama, X. Parra, and D. Anguita,\` “Transition-aware human activity recognition using smartphones,” Neurocomputing, vol. 171, pp. 754–767, 2016.

[9] J. Saha, C. Chowdhury, D. Ghosh, and S. Bandyopadhyay, “A detailed human activity transition recognition framework for grossly labeled data from smartphone accelerometer,” Multimedia Tools and Applications, vol. 80, pp. 9895–9916, 2021.

[10] A. Ouyang, S. Chen, and J. Epps, “Evaluation of event-based and timing features from head movement and eye activity for wearable task transition detection,” in 2024 International Conference on Activity and Behavior Computing (ABC). IEEE, 2024, pp. 1–10.

[11] H. Wang, J. Zhao, J. Li, L. Tian, P. Tu, T. Cao, Y. An, K. Wang, and S. Li, “Wearable sensor-based human activity recognition using hybrid deep learning techniques,” Security and communication Networks, vol. 2020, no. 1, p. 2132138, 2020.

[12] S. I. Khan, H. Dawood, M. Khan, G. F. Issa, A. Hussain, M. M. Alnfiai, and K. M. Adnan, “Transition-aware human activity recognition using an ensemble deep learning framework,” Computers in Human Behavior, vol. 162, p. 108435, 2025.

[13] S. J. Talukder, Y. Yue, and G. Gkioxari, “TOTEM: TOkenized time series EMbeddings for general time series analysis,” Transactions on Machine Learning Research, 2024.

[14] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” in The Twelfth International Conference on Learning Representations, 2024.

[15] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[16] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan, “Selfattention with functional time representation learning,” Advances in neural information processing systems, vol. 32, 2019.

[17] J. Li, Y. Wang, and J. McAuley, “Time interval aware self-attention for sequential recommendation,” in Proceedings of the 13th international conference on web search and data mining, 2020, pp. 322–330.

[18] V. A. Tran, G. Salha-Galvan, B. Sguerra, and R. Hennequin, “Attention mixtures for time-aware sequential recommendation,” in Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023, pp. 1821–1826.

[19] S. M. Cho, E. Park, and S. Yoo, “Meantime: Mixture of attention mechanisms with multi-temporal embeddings for sequential recommendation,” in Proceedings of the 14th ACM Conference on recommender systems, 2020, pp. 515–520.

[20] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in International conference on machine learning. PMLR, 2017, pp. 3319–3328.

[21] Y. Hao, L. Dong, F. Wei, and K. Xu, “Self-attention attribution: Interpreting information interactions inside transformer,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 35, no. 14, 2021, pp. 12 963–12 971.

[22] L. D. Ellingson, J. D. Meyer, R. P. Shook, P. M. Dixon, G. A. Hand, M. D. Wirth, A. E. Paluch, S. Burgess, J. R. Hebert, and S. N. Blair, “Changes in sedentary time are associated with changes in mental wellbeing over 1 year in young adults,” Preventive medicine reports, vol. 11, pp. 274–281, 2018.

[23] T. J. Saunders, T. McIsaac, K. Douillette, N. Gaulton, S. Hunter, R. E. Rhodes, S. A. Prince, V. Carson, J.-P. Chaput, S. Chastin et al., “Sedentary behaviour and health in adults: an overview of systematic reviews,” Applied Physiology, Nutrition, and Metabolism, vol. 45, no. 10, pp. S197–S217, 2020.

[24] S. Cutini, G. M. Duma, and G. Mento, “How time shapes cognitive control: A high-density eeg study of task-switching,” Biological psychology, vol. 160, p. 108030, 2021.

[25] S. Chen and J. Epps, “Multimodal coordination measures to understand users and tasks,” ACM Transactions on Computer-Human Interaction (TOCHI), vol. 27, no. 6, pp. 1–26, 2020.

[26] ——, “Task load estimation from multimodal head-worn sensors using event sequence features,” IEEE Transactions on Affective computing, vol. 12, no. 3, pp. 622–635, 2019.

[27] ——, “Atomic head movement analysis for wearable four-dimensional task load recognition,” IEEE Journal of Biomedical and Health Informatics, vol. 23, no. 6, pp. 2464–2474, 2019.

[28] Y. Ding, Y. Li, H. Sun, R. Liu, C. Tong, C. Liu, X. Zhou, and C. Guan, “Eeg-deformer: A dense convolutional transformer for brain-computer interfaces,” IEEE Journal of Biomedical and Health Informatics, 2024.

[29] Q. Guo, X. Qiu, P. Liu, X. Xue, and Z. Zhang, “Multi-scale selfattention for text classification,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 05, 2020, pp. 7847–7854.

[30] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[31] P. Shaw, J. Uszkoreit, and A. Vaswani, “Self-attention with relative position representations,” in North American Chapter of the Association for Computational Linguistics, 2018.

[32] K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep inside convolutional networks: Visualising image classification models and saliency maps,” in Workshop at International Conference on Learning Representations, 2014.

[33] A. Binder, G. Montavon, S. Lapuschkin, K.-R. Muller, and W. Samek,¨ “Layer-wise relevance propagation for neural networks with local renormalization layers,” in International conference on artificial neural networks. Springer, 2016, pp. 63–71.

[34] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 618–626.

[35] X. Ji, Y. Tu, W. He, J. Wang, H.-W. Shen, and P.-Y. Yen, “Usevis: Visual analytics of attention-based neural embedding in information retrieval,” Visual Informatics, vol. 5, no. 2, pp. 1–12, 2021.

[36] J. Vig, “A multiscale visualization of attention in the transformer model,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, M. R. Costa-jussa\` and E. Alfonseca, Eds. Florence, Italy: Association for Computational Linguistics, Jul. 2019, pp. 37–42.

[37] S. Jain and B. C. Wallace, “Attention is not explanation,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 3543–3556.

[38] S. Wiegreffe and Y. Pinter, “Attention is not not explanation,” in Conference on Empirical Methods in Natural Language Processing, 2019.

[39] S. Chen and J. Epps, “A high-quality landmarked infrared eye video dataset (ireye4task): Eye behaviors, insights and benchmarks for wearable mental state analysis,” IEEE Transactions on Affective Computing, vol. 14, no. 4, pp. 3078–3093, 2023.

[40] D. D. Salvucci and J. H. Goldberg, “Identifying fixations and saccades in eye-tracking protocols,” in Proceedings of the 2000 Symposium on Eye Tracking Research & Applications. Association for Computing Machinery, 2000, p. 71–78.

[41] M. Kassner, W. Patera, and A. Bulling, “Pupil: an open source platform for pervasive eye tracking and mobile gaze-based interaction,” in Proceedings of the 2014 ACM international joint conference on pervasive and ubiquitous computing: Adjunct publication, 2014, pp. 1151–1160.

[42] T. K. Koo and M. Y. Li, “A guideline of selecting and reporting intraclass correlation coefficients for reliability research,” Journal of chiropractic medicine, vol. 15, no. 2, pp. 155–163, 2016.

[43] G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, and C. Eickhoff, “A transformer-based framework for multivariate time series representation learning,” in Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, 2021, pp. 2114–2124.