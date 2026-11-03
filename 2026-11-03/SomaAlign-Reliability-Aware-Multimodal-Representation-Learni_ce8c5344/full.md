# SomaAlign: Reliability-Aware Multimodal Representation Learning for Soft Robotic Grasping

Haoyuan Wang, Shuai Peng, Yongkai Liao, Yuxiao Sun, Wei Luo, Xinxing Chen, Member, IEEE, and Jian Huang, Senior Member, IEEE

Abstract—Robust multimodal perception is essential for reliable soft robotic grasping, where compliant embodiment deformation, unstable contact, mechanical vibration, and heterogeneous sensor characteristics introduce modality-dependent noise and semantic ambiguity. Progress in this area is hindered by two challenges: the scarcity of soft robotic datasets with coordinated multisensory observations and the limited ability of existing alignment methods to account for varying reliability across contact-rich modalities. To address these challenges, we introduce Soma, a multimodal dataset for soft robotic grasping containing 100 objects, 1,000 continuous grasping trajectories, and 74,914 multimodal samples grouped by software acquisition cycle, spanning remote RGB, hand RGB, depth, tactile, IMU, and structured text observations. We further collect Soma-Success, an independently collected grasp-outcome dataset comprising 1,000 trials, of which 998 are retained after quality control, to assess whether frozen encoders learned from Soma can be reused for post-hoc grasp outcome classification and early grasp success prediction on a dataset unseen during representation learning. Built on Soma, we propose SomaAlign, a hierarchical multimodal representation learning framework with three complementary stages tailored to heterogeneous soft robotic signals. Masked reconstruction first derives a data-driven proxy for relative modality reliability; reliability-aware contrastive alignment then uses pair-specific adaptive temperatures to adjust alignment sharpness according to reconstruction-derived modality-pair reliability; finally, conflict-aware semantic constraints improve intrasample cross-modal consistency while preserving global feature geometry. SomaAlign achieves an average Recall@10 of 90.66% across 30 directed cross-modal retrieval tasks, while the learned representations further support physical attribute recognition, multimodal question answering, cross-modal generation, and cross-dataset transfer to grasp outcome prediction. On Soma-Success, frozen SomaAlign encoders with lightweight probes achieve 94.59% post-hoc classification accuracy and 84.07% early prediction accuracy. Together, Soma, Soma-Success, and SomaAlign provide a data and representation-learning foundation for robust and transferable multimodal perception in soft robotic manipulation.

Index Terms—Soft robotics, multimodal alignment, contrastive learning, tactile perception, embodied intelligence.

## I. INTRODUCTION

OFT robotic grippers and hands exploit compliant struc-S tures to enable gentle interaction and manipulation of fragile or irregular objects, while their deformation-dependent contact dynamics differ fundamentally from those of rigid manipulators [1]–[3]. Reliable grasping requires precise mul timodal perception of contact states, material properties, and time-varying interaction dynamics; otherwise, large elastic deformation may lead to unstable holding and unreliable slip estimation [4], [5]. Multimodal sensing combines comple mentary signals from vision, distributed tactile arrays, and proprioceptive inertial measurements to mitigate unimodal limitations such as visual occlusion and missing contact cues, thereby improving perceptual consistency [6], [7]. Such integration is particularly important for soft robots, whose intrinsic compliance enables gentle physical interaction but also introduces deformation-dependent perceptual variations that are less pronounced in rigid platforms [8].

Despite these advances, soft robots present two core perceptual challenges that are more pronounced than in rigid manipulators. First, structural deformation and material compliance introduce heterogeneous, modality-dependent sensor noise: tactile pressure arrays and IMUs are more susceptible to vibration during grasping, whereas RGB-D observations are generally less affected by mechanical disturbances [6]. Second, deformable contact introduces semantic ambiguity, as identical categorical grasp labels may correspond to substantially different tactile, inertial, and depth observations under different deformation magnitudes and contact configurations. Together, these factors challenge cross-modal alignment pipelines developed primarily for rigid robotic systems and may degrade conventional multimodal fusion. Robust soft-grasping perception therefore requires both coordinated multisensor data collection and representation learning explicitly adapted to modalitydependent noise and ambiguous contact semantics.

Large-scale real-world multimodal datasets provide an essential foundation for data-driven robotic manipulation, supporting imitation learning, policy generalization, and the pretraining of embodied foundation models [9]–[14]. Existing tactile and visuo-tactile datasets have advanced contact-aware perception by supporting tactile localization, material classification, grasp outcome estimation, and cross-modal signal prediction [15]–[17]. These studies demonstrate that tactile sensing captures physical interaction cues that are difficult to infer from vision alone, including contact geometry, surface properties, and object compliance. However, many existing visuo-tactile datasets are collected with rigid end-effectors or specific optical tactile sensors, where embodiment deformation and contact nonlinearity are relatively constrained. Soft grippers instead exhibit highly nonlinear kinematics, contactstate-dependent sensory responses, and heterogeneous noise distributions across visual, tactile, and inertial modalities. Consequently, datasets developed for rigid robots or specific tactile sensing systems do not fully capture the coupled multimodal dynamics intrinsic to soft grasping.

Existing multimodal tactile and grasping datasets individually support visuo-tactile learning, grasp outcome prediction, object-centric sensing, or object-in-hand pose estimation. To the best of our knowledge, no existing multimodal dataset simultaneously combines software-coordinated vision, depth, tactile, and IMU acquisition using soft grippers with multiple fingers, structured text descriptions, explicit stationary and sliding grasping states, and an independently collected graspoutcome dataset for frozen-encoder evaluation. This combination therefore defines a specific data gap in soft robotic perception. The scarcity of such datasets stems mainly from two inherent complexities of soft systems: their high-dimensional kinematics complicate motion capture and semantic state labeling [18], while high-fidelity simulation of soft-body contact suffers substantial Sim2Real gaps, limiting the utility of synthetic data for perceptual learning [19]. This shortage further constrains downstream perception: limited training data may reduce generalization across diverse object geometries and material properties, impairing contact-state estimation and semantic attribute recognition in unstructured environments. For soft hands in particular, grasp success depends not only on static object appearance but also on transient contact formation, spatiotemporal tactile pressure distributions, and dynamic motion stability during physical interaction—information not fully represented in existing rigid-robot datasets. Without highquality temporally associated multimodal recordings, adapting large-scale perception models to soft manipulation remains dif ficult, and standardized resources for grasp outcome prediction remain limited. This yields our first core research challenge: constructing a multimodal dataset with software-coordinated sensory acquisition and structured semantic annotations that reflects the physical characteristics of soft robotic grasping, together with an independently collected dataset for evaluating the reuse of learned representations in grasp outcome prediction.

Parallel to dataset development, multimodal representation learning has advanced rapidly over the past decade. Contrastive vision–language architectures such as CLIP demonstrate that visual concepts can be embedded into transferable semantic spaces via language supervision [20]; unified frameworks such as ImageBind further associate heterogeneous modalities, including images, text, audio, depth, thermal signals, and IMU, within a shared embedding manifold [21]. Systematic surveys have summarized multimodal alignment and fusion paradigms across diverse modalities and generalpurpose perception tasks [22]. Despite these advances, such models are primarily optimized for generic passive perception rather than dynamic, contact-rich robotic interaction. Within the manipulation literature, recent architectures including UniTouch, AnyTouch, CLTP, and Sparsh-X integrate tactile signals with vision and language to model physical contact events [23]–[26]. However, these methods are built for rigid manipulators paired with optical vision-based tactile sensors, whose sensing mechanism relies on tracking localized elastomer deformation under fixed mechanical constraints. Soft robotic platforms instead employ heterogeneous sensor suites combining distributed pressure tactile arrays, inertial measurement units, and RGB-D cameras, each with distinct signal distributions and noise profiles [27], [28].

This sensor-domain mismatch creates two critical difficulties when transferring existing alignment models to soft grippers. First, pre-trained encoders may not generalize well to the signal characteristics of flexible tactile arrays and finger-mounted IMUs. Second, conventional global alignment objectives do not explicitly account for soft-robot-specific semantic ambiguity and heterogeneous noise. Fixed or globally shared contrastive temperatures may over-constrain noisy tactile/IMU modality pairs while underutilizing reliable visual observations, leading to suboptimal embeddings and weakened preservation of physically meaningful contact information. Existing alignment frameworks also generally lack mechanisms to incorporate estimated modality reliability into pairspecific alignment strength. This forms our second core research challenge: designing a dedicated multimodal alignment framework adapted to the heterogeneous sensor characteristics and noisy, semantically ambiguous observations of soft robotic manipulation.

Collectively, these analyses reveal two coupled bottlenecks in robust multimodal perception for soft grasping: the lack of domain-specific, temporally associated multimodal data tailored to soft contact dynamics, and the absence of explicit reliability-aware mechanisms in generic cross-modal alignment pipelines for heterogeneous noisy sensory streams. Addressing either limitation alone is insufficient. We therefore present an integrated solution that combines a softwarecoordinated soft-grasping multimodal dataset with a hierarchical reliability-aware multimodal alignment framework. The dataset provides cycle-aligned sensory observations of contactdependent deformation dynamics, while the alignment framework learns unified cross-modal embeddings that explicitly account for modality-dependent reliability and cross-modal semantic consistency. Together, they provide the data and algorithmic foundations for robust and transferable multimodal perception in soft robotic systems.

Our main contributions are as follows:

1) We introduce Soma (SOft-robotic Multimodal Acquisition), which, to the best of our knowledge, is the first six-modality dataset dedicated to unified representation learning for soft robotic grasping. Soma combines cyclealigned remote RGB, hand RGB, depth, distributed tactile pressure, and IMU observations with structured text annotations, covering stationary and sliding grasp states. We further introduce Soma-Success, an independently collected grasp-outcome dataset excluded from representation learning, enabling frozen-encoder evaluation of post-hoc outcome classification and early success prediction.

![](images/6f72e7da0f1f638e1567641952e2edc2248bb5d798cfa1cf474560dca434c29a.jpg)  
Fig. 1. Overview of Soma, Soma-Success, and the SomaAlign framework. The soft robotic platform acquires six coordinated modalities: Remote RGB, Hand RGB, Depth, Tactile, IMU, and Text. Soma comprises 100 objects, 1,000 trajectories, and 74,914 aligned multimodal samples, while Soma-Success contains 1,000 collected grasp trials, of which 998 valid trials are retained for downstream evaluation. SomaAlign learns a unified semantic representation through Reliability Assessment, Adaptive Feature Alignment, and Conflict-aware Refinement. The learned representation supports cross-modal retrieval, physical attribute recognition, multimodal question answering, cross-modal generation, and frozen-encoder grasp success prediction.

2) We propose SomaAlign, which, to the best of our knowledge, is the first multimodal alignment framework to jointly incorporate modality reliability and crossmodal conflict into unified representation learning for soft robotic grasping. Reconstruction-derived reliability regulates modality-pair-specific contrastive temperatures. Subsequent refinement uses batch-level conflict modulation to adjust intra-sample contraction and frozen Stage-II reference centers to limit excessive geometric drift. The framework connects these mechanisms through three progressive training stages and retains only the modality encoders at inference.

3) We establish a systematic evaluation protocol for multimodal representations in soft robotic grasping, covering cross-modal correspondence, physical-attribute recognition, and frozen-encoder reuse on independently collected grasp-outcome data. Baseline comparisons, sensor-corruption tests, and component ablations assess the proposed alignment mechanisms, while post-hoc classification and early prediction examine their downstream utility with complete and partial observations.

The implementation code of SomaAlign is publicly available at https://github.com/Hijackoo/SomaAlign to facilitate reproducibility. The Supplementary Information provides additional experimental details and analyses, while accompanying videos illustrate the proposed approach and representative experimental results. An overview of Soma, Soma-Success, and the SomaAlign framework is presented in Fig. 1. The remainder of this paper is organized as follows: Section II introduces the Soma and Soma-Success datasets; Section III presents the SomaAlign methodology; Section IV reports the experimental results and discussion; and Section V concludes the paper.

## II. SOMA AND SOMA-SUCCESS DATASETS

Soma is a software-coordinated multimodal dataset for soft robotic grasping, comprising 74,914 cycle-aligned samples collected from 100 objects and 1,000 grasping trajectories using two- and three-finger pneumatic soft grippers. Each sample contains remote RGB, hand-mounted RGB, depth, tactile pressure, IMU, and structured text observations with stationary or sliding grasp-state annotations. We additionally collect Soma-Success as an independent grasp-outcome evaluation set; it is excluded from SomaAlign representation learning and used only to assess the downstream reusability of frozen SomaAlign encoders.

## A. Comparison with Existing Multimodal Tactile and Grasping Datasets

We compare Soma with representative multimodal tactile and grasping datasets, including SoftGrasp [29], The Feeling of Success [13], Touch and Go [14], VisGel [16], TVL [9], the ObjectFolder series [10]–[12], and VinT-6D [30]. Table I summarizes their characteristics along six dimensions relevant to soft robotic perception.

Table I highlights the distinct characteristics of Soma for soft robotic perception. Soma combines software-coordinated multimodal observations from two pneumatic soft-gripper configurations with explicit stationary and sliding grasp states and structured semantic descriptions. Soma-Success further provides an independently collected grasp-outcome dataset for evaluating the downstream reusability of Soma-trained encoders. Together, these characteristics make the proposed datasets well suited for studying multimodal perception under the deformation, contact variability, and heterogeneous sensing conditions of soft robotic grasping.

TABLE I  
COMPARISON OF REPRESENTATIVE MULTIMODAL TACTILE AND GRASPING DATASETS ALONG SIX DIMENSIONS RELEVANT TO SOFT ROBOTIC PERCEPTION.
<table><tr><td>Dataset</td><td>Acquisition Coordination</td><td>Modalities</td><td>Embodiment</td><td>Sliding States</td><td>Text Annotation</td><td>Independent Evaluation Set</td></tr><tr><td>SoftGrasp [29]</td><td>Not reported</td><td>RGB, joint angles, and joint torques</td><td>Articulated dexterous hand</td><td>Not explicitly included</td><td>None</td><td>No</td></tr><tr><td>The Feeling of Success [13]</td><td>Not reported</td><td>RGB and dual GelSight tactile images</td><td>Parallel-jaw gripper</td><td>Not explicitly included</td><td>None</td><td>No</td></tr><tr><td>Touch and Go [14]</td><td>Timestamped visual and tactile streams</td><td>Egocentric RGB and GelSight tactile video</td><td>Human-held tactile sensor</td><td>No grasp-state annotation</td><td>Material labels only</td><td>No</td></tr><tr><td>VisGel [16]</td><td>Paired visual-tactile</td><td>RGB and GelSight tactile images</td><td>Robot-mounted tactile sensor</td><td>No explicit grasp-state</td><td>None</td><td>No</td></tr><tr><td>TVL [9]</td><td>sequences Paired vision-touch</td><td>RGB, tactile images, and</td><td>Human-held tactile sensor</td><td>annotation No grasp-state annotation</td><td>Natural-language descriptions</td><td>No</td></tr><tr><td>ObjectFolder series [10]-[12]</td><td>samples Object-level aligned data</td><td>language Vision, touch, and audio</td><td>Object-centric measurement</td><td>No explicit grasp-state</td><td>Object metadata only</td><td>No separate subset</td></tr><tr><td>VinT-6D [30]</td><td>Calibrated and aligned samples</td><td>RGB, depth, touch, proprioception, and 6D pose</td><td>and simulation Multi-finger robotic hands</td><td>annotation No explicit sliding-state annotation</td><td>None</td><td>No separate subset</td></tr><tr><td>Soma / Soma- Success</td><td>Sequential software polling</td><td>Remote RGB, hand RGB, depth, tactile pressure, IMU, and text</td><td>Two- and three-finger pneumatic soft grippers</td><td>Stationary and sliding states</td><td>Structured attribute and state descriptions</td><td>Independent Soma-Success evaluation set</td></tr></table>

## B. Data Collection

Acquisition System: To establish cycle-level correspondence among heterogeneous sensory observations, we develop a software-coordinated acquisition system centered on a pneumatic soft robotic hand. All modalities are queried sequentially within a unified host-side polling loop, and observations returned in each cycle are grouped into one multimodal sample, providing deterministic cycle-level association across modalities.

Sensing Configuration: The perception unit comprises four sensor groups. An Intel RealSense D435i depth camera (640 × 480) captures remote RGB images and depth maps of the global scene. A hand-mounted industrial camera (JIERUI-WEITONG GF100, 2.5 mm distortion-free lens, 320 × 240, 100<sup>◦</sup> FOV) captures local object and fingertip views. One Xsens DOT sensor is mounted on each soft finger to record three-axis Euler angles, acceleration, and angular velocity, together with a timestamp. Only the nine sensor channels are used as encoder inputs. HiT Palm flexible pressure arrays are mounted on the finger contact surfaces, with each array producing an 8 × 14 tactile pressure map. The complete acquisition system and representative modality outputs are shown in Fig. 2.

Execution System: The execution unit employs pneumatic soft robotic hands actuated by regulated air pressure. An oilfree air compressor (Denair DW35, China) supplies air to an iPCU2-HMN driver (ROUCHU, China), which regulates the output pressure to control the FM-A6V5/LS1[P] soft actuators (ROUCHU, China) [6]. Two gripper configurations are used: a two-finger opposing configuration (2F) and a three-finger enveloping configuration (3F).

Object and Trajectory Collection: We select 100 everyday objects spanning diverse materials, shapes, sizes, hardness levels, colors, and functional purposes, as shown in Fig. 3. The collection includes plastic, glass, metal, ceramic, wood, fabric, foam, and sponge objects with both regular and irregular geometries.

To capture the transition from stationary grasping to sliding and unstable release, we design a controllable sliding device. The lower end of each object is connected by a fishing line to a motorized rod through a fixed pulley. During acquisition, the rod pulls the grasped object downward until it approaches release or fully drops from the soft fingers. Each object is tested in 10 independent trajectories, with five using the 2F gripper and five using the 3F gripper, yielding 1,000 continuous grasping trajectories in total.

![](images/a15854e91058e826d8bf78cf3dd7f0a9a733d820af441c80d3fd853b1dc0a7d6.jpg)  
Fig. 2. Overview of the Soma acquisition system. A unified software polling loop coordinates remote RGB-D sensing, hand-mounted RGB cameras, finger-mounted IMUs, and flexible tactile pressure arrays. Structured text descriptions are generated from experimental metadata using predefined templates Representative sensor outputs and an example text annotation are shown on the right.

![](images/d23b242cd0c2b73541a082db53da3ae963450fbeec6436386341d0f287ade8c5.jpg)  
Fig. 3. Object collection in Soma. The 100 everyday objects cover diverse materials, shapes, sizes, hardness levels, colors, and functional purposes.

## C. Annotation and Preprocessing

Temporal Segmentation: Each continuous trajectory is segmented using a non-overlapping temporal window with both the window size and step size set to three acquisition cycles. This three-cycle window provides compact local temporal context, retaining short-term variations across consecutive multimodal observations while reducing redundancy from densely sampled neighboring cycles. After segmentation, Soma contains 74,914 cycle-aligned multimodal samples, each comprising remote RGB, hand-mounted RGB, depth, tactile pressure, IMU, and structured text fields.

Semantic Annotation: Structured text descriptions are deterministically generated from experimental metadata to ensure semantic accuracy and consistency. For each sample, material, size, shape, purpose, hardness, color, object identity, and grasping state are mapped to standardized keywords, with grasping state represented by two labels: stationary and sliding. The final description follows the template: “A [material] [size] [shape] [purpose] with [hardness] hardness and [color] color. Object ID: [id], current state: [stationary/sliding].” This protocol yields sample-level descriptions for individual threecycle windows and experiment-level summaries for complete trajectories.

Dataset Split: Soma is divided into training, validation, and test sets at a ratio of 60%, 20%, and 20%, respectively. The split is performed by experiment ID rather than by individual sample, ensuring that all temporal windows from the same object-pulling trajectory remain in the same subset and preventing temporally correlated windows from appearing across training and evaluation sets.

## D. Dataset Statistics and Quality Control

To quantify the annotation diversity of Soma, we analyze the grasping state and six semantic attributes using Shannon entropy and normalized evenness. For an annotation dimension with K categories, let $p _ { k }$ denote the proportion of samples in the k-th category. We compute the Shannon entropy H, its theoretical maximum $H _ { \mathrm { m a x } } .$ and the normalized evenness J as

$$
H = - \sum _ { k = 1 } ^ { K } p _ { k } \ln p _ { k } , H _ { \operatorname* { m a x } }  &  = \ln K , J = { \frac { H } { H _ { \operatorname* { m a x } } } } .\tag{1}
$$

Here, $0 \leq J \leq 1$ , with larger values indicating a more even category distribution. The six semantic attributes contain 75 attribute values in total. As shown in Table II, shape, hardness, and color exhibit relatively high evenness, whereas material and purpose are more concentrated in common everyday categories; size remains moderately balanced. The grasping-state distribution is moderately skewed toward sliding $( J = 0 . 8 1 6 )$ , while both stationary and sliding states remain substantially represented. This imbalance reflects the collection process, in which the stationary phase is generally shorter than the subsequent sliding phase. We therefore retain the original binary state distribution rather than artificially resampling the data or introducing unannotated intermediate states.

The 2F and 3F subsets contain 37,492 (50.05%) and 37,422 (49.95%) samples, respectively, and both cover all 100 objects. Before dataset splitting, all segmented samples are checked for modality completeness, file readability, expected dimensions, and numerical validity; only samples containing all six valid modality fields are retained. Detailed gripper-wise distribution and object-level evenness statistics are provided in the Supplementary Information.

TABLE II  
DIVERSITY AND DISTRIBUTION EVENNESS OF THE GRASPING-STATE AND SEMANTIC ANNOTATIONS IN SOMA. K DENOTES THE NUMBER OF CATEGORIES, H IS THE SHANNON ENTROPY, $H _ { \mathrm { m a x } } =$ ln $K ,$ AND $J = H / H _ { \mathrm { m a x } }$ IS THE NORMALIZED EVENNESS.
<table><tr><td>Annotation Dimension</td><td>K</td><td>H</td><td> $H _ { \mathbf { m a x } }$ </td><td> $\textbf {  { J } }$ </td></tr><tr><td>Material</td><td>10</td><td>1.549</td><td>2.303</td><td>0.673</td></tr><tr><td>Hardness</td><td>5</td><td>1.419</td><td>1.609</td><td>0.882</td></tr><tr><td>Size</td><td>6</td><td>1.330</td><td>1.792</td><td>0.742</td></tr><tr><td>Shape</td><td>14</td><td>2.184</td><td>2.639</td><td>0.828</td></tr><tr><td>Color</td><td>28</td><td>2.929</td><td>3.332</td><td>0.879</td></tr><tr><td>State</td><td>2</td><td>0.566</td><td>0.693</td><td>0.816</td></tr><tr><td>Purpose</td><td>12</td><td>1.667</td><td>2.485</td><td>0.671</td></tr></table>

## E. Soma-Success: Independent Grasp Outcome Evaluation Set

Collection Protocol: Soma-Success is independently collected to evaluate whether Soma-trained encoders can be reused for grasp outcome prediction on data unseen during representation learning. It is excluded from SomaAlign training and used only for frozen-encoder evaluation with lightweight downstream probes. Using a UR-mounted soft robotic gripper, we collect 1,000 trials across 100 everyday objects, with each object tested five times under the 2F configuration and five times under the 3F configuration, yielding 500 trials per configuration.

Key-Snapshot Protocol: Each trial contains three actiontriggered snapshots rather than uniformly sampled frames. $T _ { a }$ is the last complete acquisition cycle after the manipulator reaches the pre-grasp pose and before the gripper-closing command. $T _ { b }$ is the first complete cycle after the gripper reaches its commanded closing state and establishes object contact, but before lifting begins. $T _ { c }$ is the first complete cycle after the manipulator reaches the predefined terminal lifting pose. Because the snapshots are action-triggered, their temporal intervals may vary across trials. If slip or release occurs during lifting, $T _ { c }$ remains defined by the terminal lifting pose rather than the slip onset, preventing outcomedependent snapshot selection. This protocol supports post-hoc classification using $( T _ { a } , T _ { b } , T _ { c } )$ and early prediction using only $( T _ { a } , T _ { b } )$ . The complete procedure, snapshot definitions, and representative successful and failed trials are shown in Fig. 4.

Trial Representation and Outcome Labels: At each key snapshot, we record frontal RGB, top-view RGB, depth, tactile pressure, IMU, and structured text observations. Each piezoresistive tactile sensor contains an $8 \times 1 4$ pressure grid, yielding 224 values for 2F trials and 336 values for 3F trials. One IMU is mounted on each finger to record accelerometer, gyroscope, and orientation signals. The three snapshots are grouped into one trial-level sample. A trial is labeled as successful if the object remains held when the manipulator reaches the predefined terminal lifting pose at $T _ { c } ,$ and as failed otherwise. Before quality control, the success/failure counts are 311/189 for 2F and 291/209 for 3F.

Data Quality Control: All trials are checked for modality completeness, file readability, expected dimensions, numerical validity, and visual-frame quality. A trial is excluded if any required modality at $T _ { a } , \ T _ { b }$ , or $T _ { c }$ is missing, unreadable, dimensionally invalid, contains NaN or Inf values, or includes severely corrupted or blurred visual observations. Finite extreme sensor responses are retained because they may reflect genuine contact, impact, or slip events. Of the 1,000 collected trials, two are excluded due to NaN tactile readings caused by temporary sensor disconnection; no other trials are removed. This leaves 998 valid trials, including 600 successful and 398 failed grasps. All quality-control decisions are made before data splitting and without using grasp outcome labels or model predictions.

## III. METHODOLOGY

## A. Framework Overview and Design Rationale

In this section, we introduce SomaAlign, a multimodal representation learning framework for heterogeneous perception in soft robotic grasping. SomaAlign maps vision, depth, IMU, tactile pressure, and text observations into a unified semantic space. Rather than treating multimodal alignment as a uniform feature-matching problem, it explicitly addresses two characteristics of soft robotic perception: modality-dependent reliability and cross-modal semantic inconsistency among observations of the same physical interaction.

Conceptually, SomaAlign is organized around two core methodological designs. Reliability-Aware Multimodal Alignment estimates relative modality reliability and propagates it to modality-pair-specific contrastive alignment, allowing alignment strength to vary across heterogeneous modality pairs. Batch-Adaptive Conflict-Aware Geometry-Preserving Refinement estimates sample-level cross-modal inconsistency, aggregates it at the batch level to regulate refinement strength, and constrains excessive distortion of the global representation geometry. A progressive optimization strategy connects these two designs while retaining only the modality-specific feature encoders for inference.

The overall workflow is shown in Fig. 5. The two methodological designs are implemented through three progressive optimization stages: Stage I estimates modality reliability and constructs pairwise reliability scores; Stage II performs reliability-conditioned contrastive alignment; and Stage III refines the aligned representation using conflict-aware and geometry-preserving constraints. Thus, the two methodological designs describe the core contributions of SomaAlign, whereas the three stages specify their optimization sequence. After training, all auxiliary components are removed, and raw observations are mapped directly into the unified semantic space by the learned modality-specific encoders.

## B. Reliability-Aware Multimodal Alignment

A fundamental challenge in soft robotic multimodal perception is that different sensing modalities may exhibit substantially different levels of reliability. Recent quality-aware multimodal learning methods have shown that explicitly modeling modality uncertainty or credibility can improve robustness when individual sensory streams are degraded or unreliable [31], [32]. SomaAlign therefore estimates a reconstructionderived modality reliability proxy before cross-modal alignment and uses it to regulate pair-specific contrastive learning. Tactile pressure and IMU measurements are susceptible to contact deformation, mechanical vibration, and environmental disturbance, whereas visual and textual observations exhibit different and comparatively more structured noise characteristics. Conventional multimodal contrastive learning generally applies the same alignment objective and temperature setting to all modality pairs, which may over-constrain unreliable pairs while underutilizing highly consistent ones.

![](images/eac23efa070216bed6729d11e348404bca34a57511d8283d97a7a3b9f6d63478.jpg)  
Fig. 4. Soma-Success collection protocol. The action-triggered snapshots $T _ { a } , T _ { b } ,$ and $T _ { c }$ correspond to the pre-grasp, closed-gripper, and terminal-lifting stages, respectively. The same snapshot-selection rule is applied to successful and failed trials. Representative examples are shown for the two-finger (2F) an three-finger (3F) configurations.

The reliability-aware alignment design comprises three connected components. In the optimization sequence shown in Fig. 5, reconstruction-based modality reliability estimation and pairwise reliability modeling constitute Stage I, while reliability-conditioned contrastive alignment constitutes Stage II.

1) Reconstruction-Based Modality Reliability Estimation: Because explicit sensor-noise annotations are generally unavailable for heterogeneous multimodal observations, SomaAlign uses masked reconstruction, inspired by masked autoencoding [33], as an auxiliary self-supervised probe of modality-specific reconstruction difficulty. The resulting reliability proxy reflects relative reconstruction stability rather than directly measuring the physical noise level of a sensor: observations with lower predictability or stronger corruption are generally more difficult to recover from partially observed inputs.

For each modality m in a training batch, a random mask is applied to the input $X _ { m } \in \mathbb { R } ^ { B \times D _ { m } }$ , where B is the batch size and $D _ { m }$ is the data dimension. The masked input is

$$
\tilde { X } _ { m } = X _ { m } \odot M _ { m } ,\tag{2}
$$

where $\odot$ denotes the Hadamard product and $ { \boldsymbol { M } } _ { m } \in  { \mathcal { M } }$ $\{ 0 , 1 \} ^ { B \times D _ { m } }$ is a binary mask matrix. Each element of $M _ { m }$ is set to 0 with probability $p _ { m }$ and retained with probability $1 - p _ { m }$ . The masking probability $p _ { m }$ is selected according to modality characteristics, with higher values for spatially or temporally redundant modalities such as tactile maps and IMU sequences and lower values for information-dense modalities such as text.

A modality-specific reconstruction network $G _ { m }$ reconstructs the original input from $\tilde { X } _ { m }$

$$
\hat { X } _ { m } = G _ { m } ( \tilde { X } _ { m } ) ,\tag{3}
$$

where $\hat { X } _ { m }$ is the reconstructed observation of modality m. The architecture of $G _ { m }$ is adapted to each modality: a lightweight convolutional autoencoder is used for visual data, a 2D CNN for tactile pressure maps, a 1D CNN-LSTM for IMU sequences, and a Transformer encoder for text.

Since $M _ { m } = 0$ denotes a masked position, we define the complementary mask as $\bar { M } _ { m } = \mathbf { 1 } - M _ { m }$ , where 1 denotes an all-ones matrix with the same dimensions as $M _ { m }$ . The reconstruction loss is computed only over masked positions:

$$
\mathcal { L } _ { \mathrm { r e c o n } } ^ { ( m ) } = \frac { \left. \left( \hat { X } _ { m } - X _ { m } \right) \odot \bar { M } _ { m } \right. _ { F } ^ { 2 } } { \left. \bar { M } _ { m } \right. _ { 1 } } ,\tag{4}
$$

![](images/c51fdef7a8e455666774473e96d1e01976c037bab7614b206662f91479f98330.jpg)  
Fig. 5. Overview of the SomaAlign framework. SomaAlign follows three progressive optimization stages: (1) Reconstruction-Based Modality Reliability Estimation estimates modality reliability from normalized masked-reconstruction difficulty and constructs pairwise reliability scores; (2) Reliability-Conditioned Contrastive Alignment uses the pairwise reliability scores to adapt cross-modal contrastive temperatures and learn a shared representation; and (3) Batch-Adaptive Conflict-Aware Geometry-Preserving Refinement calibrates pairwise similarities, estimates fuzzy sample-level conflict, and adaptively modulates intra-sample refinement while imposing conflict-consistency and reference-center anchoring constraints. The reference geometry constructed from Stage-II representations, including the sample-specific centers and global representation scale, is frozen and reused during Stage-III optimization. After training, only the modality-specific encoders are retained for inference.

where $\Vert \cdot \Vert _ { F }$ denotes the Frobenius norm and $\Vert \bar { M } _ { m } \Vert$ <sub>1</sub> denotes the number of masked elements. To make reconstruction difficulty comparable across heterogeneous modalities, a historical maximum reconstruction loss is maintained for each modality. At optimization step t,

$$
\begin{array} { r } { \mathcal { L } _ { \operatorname* { m a x } , t } ^ { ( m ) } = \operatorname* { m a x } \left( \mathcal { L } _ { \operatorname* { m a x } , t - 1 } ^ { ( m ) } , \mathcal { L } _ { \mathrm { r e c o n } , t } ^ { ( m ) } \right) , } \end{array}\tag{5}
$$

with $\mathcal { L } _ { \mathrm { m a x } , 1 } ^ { ( m ) } = \mathcal { L } _ { \mathrm { r e c o n } , 1 } ^ { ( m ) }$ . Omitting the optimization-step index t hereafter, the normalized reconstruction difficulty is

$$
\eta _ { m } = \frac { \mathcal { L } _ { \mathrm { r e c o n } } ^ { ( m ) } } { \mathcal { L } _ { \mathrm { m a x } } ^ { ( m ) } } , \qquad \eta _ { m } \in [ 0 , 1 ] .\tag{6}
$$

The corresponding modality reliability is

$$
r _ { m } = 1 - \eta _ { m } ,\tag{7}
$$

where a larger $r _ { m }$ indicates relatively more stable reconstruction behavior.

2) Pairwise Reliability Modeling: For each modality pair $( m , n )$ , the individual modality reliabilities $r _ { m }$ and $r _ { n }$ are combined into a composite reliability score:

$$
R _ { m , n } = w _ { m , n } \frac { r _ { m } + r _ { n } } { 2 } ,\tag{8}
$$

where $w _ { m , n } \in ( 0 , 1 ]$ is a fixed prior representing the coarse physical or semantic association strength between modalities m and n. We set $w _ { m , n } ~ = ~ 0 . 8$ for vision–text pairs and $w _ { m , n } ~ = ~ 0 . 5$ for all remaining modality pairs. Importantly, $w _ { m , n }$ does not encode sample-specific reliability: the dynamic component of $R _ { m , n }$ is determined by the reconstructionderived reliabilities $r _ { m }$ and $r _ { n } ,$ , while $w _ { m , n }$ only modulates the global association strength of the modality pair. The resulting $R _ { m , n }$ connects modality-level reliability estimation to pairwise feature alignment.

3) Reliability-Conditioned Contrastive Alignment: Stage II in Fig. 5 uses the pairwise reliability scores obtained above to regulate cross-modal contrastive alignment. The temperature parameter in contrastive learning affects the hardness-aware weighting of negative samples and the geometry of the learned representation space [34], while adaptive temperature strategies can condition this parameter on the current alignment or confidence state [35]. We therefore use $R _ { m , n }$ to generate pairspecific temperatures, allowing contrastive sharpness to vary across heterogeneous modality pairs.

We adopt the UniTouch architecture [23] with the official pretrained weights as the unified backbone for modalityspecific feature extraction. For sample i and modality m, the encoder $E _ { m }$ produces

$$
\mathbf { f } _ { i , m } = E _ { m } ( X _ { i , m } ) \in \mathbb { R } ^ { d } ,\tag{9}
$$

where $X _ { i , m }$ denotes the observation of modality m for sample i, and d is the shared embedding dimension. The feature is $\ell _ { 2 ^ { - } }$ normalized as

$$
\hat { \bf f } _ { i , m } = \frac { { \bf f } _ { i , m } } { \Vert { \bf f } _ { i , m } \Vert _ { 2 } + \epsilon } , \qquad \epsilon = 1 0 ^ { - 8 } ,\tag{10}
$$

where $\| \cdot \| _ { 2 }$ denotes the Euclidean norm and ϵ is a small numerical constant used to avoid division by zero.

For each modality pair $( m , n )$ , the temperature is determined by $R _ { m , n }$

$$
\tau _ { m , n } = \tau _ { \mathrm { b a s e } } \exp \left( - \kappa R _ { m , n } \right) ,\tag{11}
$$

where $\tau _ { \mathrm { b a s e } }$ is the baseline temperature and κ is a learnable scaling coefficient. This monotonic mapping assigns smaller temperatures to more reliable modality pairs, producing sharper similarity distributions, while less reliable pairs receive larger temperatures and smoother distributions. To ensure $\kappa > 0 .$ , it is reparameterized as

$$
\kappa = \exp ( \hat { \kappa } ) ,\tag{12}
$$

where $\hat { \kappa }$ is optimized by backpropagation. A quadratic regularization term $\mathcal { L } _ { \mathrm { r e g } }$ is applied to κˆ to prevent excessive temperature scaling.

Feature alignment is optimized using the InfoNCE objective [36]. Averaging over all unique modality pairs gives

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \frac { 2 } { M ( M - 1 ) } \sum _ { m = 1 } ^ { M - 1 } \sum _ { n = m + 1 } ^ { M } \mathcal { L } _ { m , n } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } ,\tag{13}
$$

where M is the number of modalities, $\mathcal { L } _ { \mathrm { r e g } }$ denotes the quadratic regularization term applied to κˆ, and $\lambda _ { \mathrm { r e g } }$ is its weighting coefficient.

For modality pair $( m , n )$

$$
\mathcal { L } _ { m , n } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \sin ( \hat { \mathbf { f } } _ { i , m } , \hat { \mathbf { f } } _ { i , n } ) / \tau _ { m , n } \right) } { \sum _ { j = 1 } ^ { B } \exp \left( \sin ( \hat { \mathbf { f } } _ { i , m } , \hat { \mathbf { f } } _ { j , n } ) / \tau _ { m , n } \right) } ,\tag{14}
$$

where i and j index samples within the same mini-batch. The pair $( \hat { \mathbf { f } } _ { i , m } , \hat { \mathbf { f } } _ { i , n } )$ is the positive cross-modal pair, whereas $( \hat { \mathbf { f } } _ { i , m } , \hat { \mathbf { f } } _ { j , n } )$ with $j \neq i$ is a negative pair. Because the features are ℓ<sub>2</sub>-normalized, their similarity is

$$
\sin ( \hat { \mathbf { f } } _ { i , m } , \hat { \mathbf { f } } _ { j , n } ) = \hat { \mathbf { f } } _ { i , m } ^ { \top } \hat { \mathbf { f } } _ { j , n } .\tag{15}
$$

Thus, the reconstruction-derived pairwise reliability directly controls contrastive sharpness through $R _ { m , n } \to \tau _ { m , n } .$

C. Batch-Adaptive Conflict-Aware Geometry-Preserving $R e \mathrm { - }$ finement

Although reliability-conditioned contrastive learning establishes a globally aligned multimodal representation space, cross-modal consistency may still vary across samples because of transient contact conditions, motion disturbances, partial visual observations, and semantic ambiguity. SomaAlign therefore introduces batch-adaptive conflict-aware geometrypreserving refinement, corresponding to Stage III in Fig. 5. It combines sample-level conflict estimation, batch-level adaptive modulation, and frozen Stage-II reference geometry to improve local cross-modal consistency without excessively distorting the global representation structure.

1) Reference Geometry Construction: After Stage II, let $\mathbf { f } _ { i , m } \in \mathbb { R } ^ { d }$ denote the aligned feature vector of sample i from modality m. A frozen reference center is constructed for each training sample:

$$
\mathbf { c } _ { i } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbf { f } _ { i , m } ,\tag{16}
$$

where $\mathbf { c } _ { i } ~ \in ~ \mathbb { R } ^ { d }$ is the Stage-II reference center of sample i. These centers characterize the sample-level geometry established by Stage-II alignment and remain fixed throughout Stage III.

To normalize center displacement relative to the overall representation scale, we compute the mean pairwise distance between all reference centers:

$$
D _ { \mathrm { g l o b a l } } = \frac { 2 } { N ( N - 1 ) } \sum _ { i < j } \left\| \mathbf { c } _ { i } - \mathbf { c } _ { j } \right\| _ { 2 } ,\tag{17}
$$

where N is the number of training samples used to construct the reference geometry. Both $\{ { \mathbf { c } } _ { i } \} _ { i = 1 } ^ { N }$ and the scalar $D _ { \mathrm { g l o b a l } }$ are computed once from the Stage-II representations and kept fixed during Stage III.

2) Distribution-Calibrated Fuzzy Conflict Estimation: Because heterogeneous modality pairs may exhibit substantially different similarity distributions, directly comparing their raw similarities can introduce pair-dependent scale bias. We therefore calibrate each modality pair against its own Stage-II similarity distribution before estimating cross-modal conflict.

Let π index a modality pair $( m , n )$ , with $\pi \in \{ 1 , \ldots , P \}$ and $P = M ( M - 1 ) / 2$ denoting the total number of unique modality pairs. For sample i, the cosine similarity associated with pair π is

$$
s _ { i , \pi } = \frac { \mathbf { f } _ { i , m } ^ { \top } \mathbf { f } _ { i , n } } { \Vert \mathbf { f } _ { i , m } \Vert _ { 2 } \Vert \mathbf { f } _ { i , n } \Vert _ { 2 } } ,\tag{18}
$$

where $\mathbf { f } _ { i , m }$ and $\mathbf { f } _ { i , n }$ denote the current feature vectors of the two modalities associated with pair π. They are initialized from the Stage-II aligned representations and progressively updated during Stage-III optimization.

For each modality pair π, an empirical cumulative distribution function (ECDF) $F _ { \pi } ( \cdot )$ is estimated exclusively from the fixed Stage-II representations of the training split. The raw similarity is converted into a calibrated pairwise conflict indicator:

![](images/55cd7367638c5e215d2e523e18214ab9b10eb5d480310727c60ebf71d1218241.jpg)  
Fig. 6. Conceptual illustration of pair-specific distribution calibration and fuzzy membership assignment. Different modality pairs may exhibit different raw similarity distributions; therefore, the same similarity value can represent different degrees of relative inconsistency. Pair-specific ECDFs convert raw similarities into comparable conflict indicators $\bar { \xi _ { i , \pi } } .$ . The calibrated indicator is then softly assigned to overlapping ordered Gaussian fuzzy sets, allowing transitional cross-modal conflicts to partially activate multiple adjacent conflict levels.

$$
\xi _ { i , \pi } = 1 - F _ { \pi } ( s _ { i , \pi } ) ,\tag{19}
$$

where a larger $\xi _ { i , \pi }$ indicates stronger inconsistency relative to the typical similarity distribution of modality pair π. All pair-specific ECDFs remain fixed throughout Stage III.

The calibrated indicator is processed by $K _ { \mathrm { f } }$ ordered Gaussian membership functions:

$$
\mu _ { i , k , \pi } = \exp \left[ - \frac { ( \xi _ { i , \pi } - a _ { k } ) ^ { 2 } } { 2 \sigma _ { k } ^ { 2 } } \right] ,\tag{20}
$$

where $k \in \{ 1 , \ldots , K _ { \mathrm { f } } \}$ indexes the fuzzy rules, and $a _ { k }$ and $\sigma _ { k }$ denote the center and width of the k-th Gaussian membership function, respectively.

As illustrated in Fig. 6, the memberships are normalized across the rule dimension to obtain fuzzy-rule weights:

$$
\omega _ { i , k , \pi } = \frac { \mu _ { i , k , \pi } } { \sum _ { \ell = 1 } ^ { K _ { \mathrm { f } } } \mu _ { i , \ell , \pi } } ,\tag{21}
$$

where $\ell$ is a summation index over the fuzzy rules and $\begin{array} { r } { \sum _ { k = 1 } ^ { K _ { \mathrm { f } } } \omega _ { i , k , \pi } = 1 } \end{array}$ . The normalized rule weights are combined with ordered consequents $v _ { k }$ to obtain a continuous pair-level conflict score:

$$
q _ { i , \pi } = \sum _ { k = 1 } ^ { K _ { \mathrm { f } } } \omega _ { i , k , \pi } v _ { k } ,\tag{22}
$$

where $v _ { k }$ is the consequent of the k-th fuzzy rule and increases from low to high conflict. Consequently, stronger activation of high-conflict rules produces a larger $q _ { i , \pi }$

To capture both overall inconsistency and a small number of strongly conflicting modality pairs, the pair-level scores are aggregated as

$$
q _ { i } = ( 1 - \rho ) \mathrm { M e a n } _ { \pi } ( q _ { i , \pi } ) + \rho \mathrm { T o p K M e a n } _ { \pi } ( q _ { i , \pi } ; K _ { \mathrm { t } } ) ,\tag{23}
$$

where $\rho ~ \in ~ [ 0 , 1 ]$ controls the emphasis on salient pairwise conflicts. The global mean is

$$
\mathrm { M e a n } _ { \pi } ( q _ { i , \pi } ) = \frac { 1 } { P } \sum _ { \pi = 1 } ^ { P } q _ { i , \pi } ,\tag{24}
$$

and TopKMean $_ { \cdot \pi } ( q _ { i , \pi } ; K _ { \mathrm { t } } )$ denotes the average of the $K _ { \mathrm { t } }$ largest values among $\{ q _ { i , \pi } \} _ { \pi = 1 } ^ { P }$ . Thus, $\rho = 0$ uses only the global mean, whereas $\rho = 1$ uses only the $K _ { \mathrm { t } }$ most salient modality-pair conflicts.

The aggregated score is finally mapped to a bounded sample-level conflict estimate:

$$
\gamma _ { i } = \mathrm { s i g m o i d } \left[ \alpha ( q _ { i } - b ) \right] ,\tag{25}
$$

where α and b are the scale and offset of the sigmoid mapping, respectively. Rather than setting them manually, we calibrate them using lower and upper quantiles $Q _ { \mathrm { l } }$ and $Q _ { \mathrm { u } }$ of the training-set distribution of $q _ { i } \mathbf { \cdot }$

$$
\alpha = \frac { \mathrm { l o g i t } ( \gamma _ { \mathrm { { u } } } ) - \mathrm { l o g i t } ( \gamma _ { \mathrm { l } } ) } { Q _ { \mathrm { { u } } } - Q _ { \mathrm { 1 } } } , \qquad b = Q _ { \mathrm { l } } - \frac { \mathrm { l o g i t } ( \gamma _ { \mathrm { l } } ) } { \alpha } ,\tag{26}
$$

where $\gamma _ { 1 }$ and $\gamma _ { \mathrm { u } }$ are the prescribed target conflict levels corresponding to $Q _ { \mathrm { l } }$ and $Q _ { \mathrm { { u } } } ,$ respectively, and logi $\begin{array} { r } { \mathrm { ~ t } ( z ) = \log \frac { z } { 1 - z } } \end{array}$ is the inverse of the sigmoid function.

3) Batch-Adaptive Conflict Modulation: The sample-level conflict estimates $\{ \gamma _ { i } \}$ are aggregated to obtain a batch-level conflict factor:

$$
\bar { \gamma } _ { B } = \frac 1 B \sum _ { i = 1 } ^ { B } \mathrm { s g } ( \gamma _ { i } ) ,\tag{27}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. The scalar $\bar { \gamma } _ { B }$ summarizes the overall cross-modal conflict level of the current batch and modulates the strength of representation refinement.

Let $\mathbf { f } _ { i , m } ^ { \mathrm { I I I } } \in \mathbb { R } ^ { d }$ denote the current Stage-III feature vector of sample i from modality $m .$ . The conflict-modulated intrasample refinement objective is

$$
\mathcal { L } _ { \mathrm { { i n t r a } } } = ( 1 - \bar { \gamma } _ { B } ) \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left. \mathbf { f } _ { i , m } ^ { \mathrm { { I I I } } } - \mathbf { c } _ { i } \right. _ { 2 } ,\tag{28}
$$

where $\mathbf { c } _ { i }$ is the frozen Stage-II reference center defined above. A lower batch-level conflict strengthens the contraction toward the Stage-II reference centers, whereas a higher conflict weakens it. Because all samples in the batch share the same modulation factor $1 - \bar { \gamma } _ { B }$ , their relative weighting within $\mathcal { L } _ { \mathrm { i n t r a } }$ is preserved.

4) Conflict-Consistency Learning: To provide an explicit learning signal for the fuzzy conflict scorer, we construct a calibrated target from the ECDF-normalized pairwise conflict indicators:

$$
t _ { i } = ( 1 - \rho ) \mathrm { M e a n } _ { \pi } ( \xi _ { i , \pi } ) + \rho \mathrm { T o p K M e a n } _ { \pi } ( \xi _ { i , \pi } ; K _ { \mathrm { t } } ) ,\tag{29}
$$

where $t _ { i }$ is the conflict target for sample i, using the same aggregation operators, $\rho ,$ and $K _ { \mathrm { t } }$ defined above. The fuzzy

![](images/c9ef8b2f6e7aa4b5ad49f6556e29b39dc31b94bd63dcb2bca41d6222258a0ea7.jpg)  
Fig. 7. Motivation of the center-protection loss. Local cross-modal refinement may cause sample centers to deviate from the geometry established in Stage II. The center-protection constraint penalizes excessive center drift while allowing local multimodal representations to become more compact.

conflict scorer is optimized by matching the predicted samplelevel conflict $\gamma _ { i }$ to this stop-gradient target:

$$
\mathcal { L } _ { \mathrm { c o n f } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \mathrm { S m o o t h L 1 } \left( \gamma _ { i } , \mathrm { s g } ( t _ { i } ) \right) ,\tag{30}
$$

where SmoothL $1 ( \cdot , \cdot )$ denotes the standard smooth- ${ \cal - L } _ { 1 }$ (Huber-type) regression loss. Thus, ${ \mathcal { L } } _ { \mathrm { c o n f } }$ trains the fuzzy scorer to reproduce the calibrated conflict structure without propagating gradients through the target construction.

5) Reference-Center Anchoring for Geometry Preservation: Although intra-sample refinement promotes local cross-modal compactness, excessive displacement of sample centers may distort the global geometry established in Stage II. We therefore define the current Stage-III center of sample i as

$$
\hat { \mathbf { c } } _ { i } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbf { f } _ { i , m } ^ { \mathrm { I I I } } ,\tag{31}
$$

where $\bar { \mathbf { c } } _ { i } \in \mathbb { R } ^ { d }$ denotes the current Stage-III center relative to the frozen Stage-II reference center $\mathbf { c } _ { i }$

As illustrated in Fig. 7, the center-protection loss is activated only when the displacement from the frozen reference exceeds a scale-adaptive tolerance:

$$
\mathcal { L } _ { \mathrm { c e n t e r } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m a x } \left( 0 , \left. \bar { \mathbf { c } } _ { i } - \mathbf { c } _ { i } \right. _ { 2 } - \lambda _ { \mathrm { t o l } } D _ { \mathrm { g l o b a l } } \right) ,\tag{32}
$$

where $\lambda _ { \mathrm { t o l } }$ controls the allowable center displacement relative to the global representation scale $D _ { \mathrm { g l o b a l } }$ . It determines the activation margin of the center-protection penalty rather than its loss weight.

The geometry-preserving effect can be further characterized by the triangle inequality. Let

$$
d _ { i j } = \lVert { \bf c } _ { i } - { \bf c } _ { j } \rVert _ { 2 }\tag{33}
$$

denote the Stage-II distance between the frozen reference centers of samples i and j. If the refined centers satisfy

$$
\begin{array} { r } { \| \bar { \mathbf { c } } _ { i } - \mathbf { c } _ { i } \| _ { 2 } \leq \lambda _ { \mathrm { t o l } } D _ { \mathrm { g l o b a l } } , } \\ { \| \bar { \mathbf { c } } _ { j } - \mathbf { c } _ { j } \| _ { 2 } \leq \lambda _ { \mathrm { t o l } } D _ { \mathrm { g l o b a l } } , } \end{array}\tag{34}
$$

then

Algorithm 1 SomaAlign Progressive Multimodal Alignment   
Input: Dataset $\overline { { \mathcal { D } ; } }$ number of modalities M; stage epoch   
counts $N _ { \mathrm { e p } } ^ { ( 1 ) } , \dot { N } _ { \mathrm { e p } } ^ { ( 2 ) }$ , and $N _ { \mathrm { e p } } ^ { ( 3 ) }$   
Output: Fine-tuned modality encoders $\lbrace E _ { m } \rbrace _ { m = 1 } ^ { M }$   
1: Stage I: Reliability Estimation   
2: for epoch = 1 to $\dot { N } _ { \mathrm { e p } } ^ { ( 1 ) }$ do   
3: Optimize each reconstruction network $G _ { m }$ using $\mathcal { L } _ { \mathrm { r e c o r } } ^ { ( m ) }$   
4: Update modality-specific historical reconstruction ref  
erences   
5: end for   
6: Estimate $\{ r _ { m } \}$ and construct pairwise reliabilities $\{ R _ { m , n } \}$   
7: Stage II: Reliability-Conditioned Alignment   
8: for epoch = 1 to $ { N _ { \mathrm { e p } } } ^ { ( 2 ) }$ do   
9: Compute pair-specific temperatures $\{ \tau _ { m , n } \}$   
10: Optimize modality encoders using $\mathcal { L } _ { \mathrm { s t a g e 2 } }$   
11: end for   
12: Construct and freeze {c<sub>i</sub>}, $D _ { \mathrm { g l o b a l } }$ , and pair-specific   
ECDFs   
13: Calibrate and freeze sigmoid parameters α and b   
14: Stage III: Batch-Adaptive Conflict Refinement   
15: for epoch = 1 to $N _ { \mathrm { e p } } ^ { ( 3 ) }$ do   
16: Estimate sample conflicts $\{ \gamma _ { i } \}$ by calibrated fuzzy   
inference   
17: Compute batch conflict factor $\bar { \gamma } _ { B }$   
18: Compute $\mathcal { L } _ { \mathrm { i n t r a } } , \mathcal { L } _ { \mathrm { c o n f } } .$ , and L<sub>center</sub>   
19: Update modality encoders and fuzzy conflict scorer   
using $\mathcal { L } _ { \mathrm { s t a g e 3 } }$   
20: end for   
21: return $\{ E _ { m } \} _ { m = 1 } ^ { M }$

$$
\left\| \bar { \mathbf { c } } _ { i } - \bar { \mathbf { c } } _ { j } \right\| _ { 2 } \geq d _ { i j } - 2 \lambda _ { \mathrm { t o l } } D _ { \mathrm { g l o b a l } } .\tag{35}
$$

Thus, bounding each Stage-III center around its frozen Stage-II reference limits excessive distortion of inter-sample geometry while still permitting local multimodal refinement.

6) Overall Conflict-Aware Refinement Objective: The complete objective used for Stage-III refinement is

$$
\mathcal { L } _ { \mathrm { s t a g e 3 } } = \mathcal { L } _ { \mathrm { i n t r a } } + \beta \mathcal { L } _ { \mathrm { c e n t e r } } + \lambda _ { \mathrm { c o n f } } \mathcal { L } _ { \mathrm { c o n f } } ,\tag{36}
$$

where $\beta$ is the weight of the geometry-preservation loss and $\lambda _ { \mathrm { c o n f } }$ is the weight of the conflict-consistency loss. Unlike $\beta ,$ which controls the magnitude of the center-protection penalty, $\lambda _ { \mathrm { t o l } }$ defines the center-drift margin at which $\mathcal { L } _ { \mathrm { c e n t e r } }$ becomes active.

## D. Progressive Optimization and Encoder-Only Inference

The two methodological designs above are connected through the progressive optimization strategy summarized in Algorithm 1.

Quantities used as statistical or geometric references in Stage III—including the Stage-II reference centers, $D _ { \mathrm { g l o b a l } } .$ pair-specific ECDFs, and sigmoid calibration parameters— are constructed from the training split after Stage II and remain fixed throughout Stage-III optimization. The modality encoders and fuzzy conflict scorer are optimized using these fixed references.

All auxiliary mechanisms are training-only. After optimization, the reconstruction networks, reliability estimation and temperature modulation, ECDF calibration, fuzzy conflict scorer, batch-level conflict modulation, and referencecenter constraints are discarded. Only the fine-tuned modalityspecific encoders $\{ E _ { m } \} _ { m = 1 } ^ { M }$ are retained for inference.

## IV. EXPERIMENTS

## A. Experimental Setup

SomaAlign is implemented in PyTorch with modalityspecific encoders initialized from UniTouch pretrained weights. Experiments are conducted on NVIDIA RTX 4090 GPUs, each modified to provide 48 GB of VRAM, using AdamW and a cosine annealing learning-rate schedule. Stage I is trained for 50 epochs with a batch size of 64. Stages II and III are each trained for 200 epochs with an initial learning rate of $1 \times 1 0 ^ { - 5 }$ , a per-GPU batch size of $^ { 8 , }$ and four gradientaccumulation steps.

For Stage I, the masking probabilities $p _ { m }$ are fixed at 0.7 for tactile and IMU, 0.6 for both RGB streams and depth, and 0.3 for text. For Stage II, we set $\tau _ { \mathrm { b a s e } } = 0 . 3$ and $\lambda _ { \mathrm { r e g } } = 0 . 1$ . The temperature scaling coefficient κ is initialized to 1 by setting $\hat { \kappa } = 0 ;$ , and κˆ is optimized jointly with the encoders.

For Stage III, we use $K _ { \mathrm { f } } ~ = ~ 5$ ordered Gaussian fuzzy rules, with centers $a _ { k }$ initialized to $\{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ widths $\sigma _ { k }$ initialized to 0.15, and consequents $v _ { k }$ initialized to {0, 0.25, 0.5, 0.75, 1}. Conflict aggregation in Eq. (23) uses $\rho = 0 . 5$ and $K _ { \mathrm { t } } ~ = ~ 3$ . The 5th and 95th percentiles of the training-set distribution of $q _ { i }$ are mapped to target conflict levels $\gamma _ { 1 } = 0 . 2$ and $\gamma _ { \mathrm { u } } = 0 . 8$ , respectively. Calibration using Eq. (26) yields $\alpha = 5 . 1 0$ and $b = \ : 0 . 5 8 2$ . The center-drift tolerance and loss weights are $\lambda _ { \mathrm { t o l } } ~ = ~ 0 . 1 , ~ \beta ~ = ~ 0 . 2 5$ , and $\lambda _ { \mathrm { c o n f } } ~ = ~ 0 . 1$ . The reference centers, global distance scale, pair-specific ECDFs, and sigmoid calibration parameters are constructed using only the Stage-II training representations and remain fixed throughout Stage III.

Experiments on Soma follow the 60%/20%/20% training/validation/test split defined in Section II-C, with each grasping trajectory assigned entirely to one subset. The training subset is used for representation learning and reference construction, the validation subset for model and hyperparameter selection, and the test subset for final evaluation. The independent Soma-Success evaluation protocol is specified in its corresponding subsection. Additional parameter-effect and diagnostic analyses are provided in the Supplementary Information.

## B. Cross-Modal Retrieval Performance

Cross-modal retrieval evaluates correspondence in the shared representation space by ranking gallery features according to their cosine similarity to a query. For query $q ,$ let $\mathcal { P } _ { q }$ denote its valid positive set. Recall@ $K$ is defined as

$$
\mathrm { R e c a l l @ } K = \frac { 1 } { Q } \sum _ { q = 1 } ^ { Q } \mathbb { I } \left( \mathrm { T o p K } ( q ) \cap \mathcal { P } _ { q } \neq \emptyset \right) ,\tag{37}
$$

TABLE III  
COMPARISON WITH CONTROLLED MULTIMODAL ALIGNMENT BASELINES ON SOMA. ALL RECALL VALUES ARE PERCENTAGES.
<table><tr><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>Pairwise InfoNCE [36]</td><td>42.00</td><td>60.10</td><td>65.77</td></tr><tr><td>Hub Alignment [21]</td><td>44.55</td><td>64.30</td><td>70.55</td></tr><tr><td>SomaAlign</td><td>63.04</td><td>85.55</td><td>90.66</td></tr></table>

where $Q$ is the number of queries, TopK(q) contains the top-K retrieved gallery entries, and $\mathbb { I } ( \cdot )$ is the indicator function.

For sensory-to-sensory retrieval, $\mathcal { P } _ { q }$ contains only the paired observation from the same temporal window. For sensory-totext retrieval, the gallery contains one entry per unique structured description, and the query’s corresponding description is the positive match. For text-to-sensory retrieval, each unique description is queried once, and all sensory observations sharing exactly that description are valid positives. Thus, sensoryonly retrieval uses observation-level correspondence, whereas text-related retrieval uses description-level correspondence.

We evaluate all 30 directed retrieval tasks among the six modalities on a fixed 500-sample subset of the test split. Recall@K is computed separately for each direction and then macro-averaged, giving equal weight to all directions despite differences in query and gallery sizes. SomaAlign achieves mean Recall@1, Recall@5, and Recall@10 values of 63.04%, 85.55%, and 90.66%, respectively.

As shown in Fig. 8, Remote→Hand and Remote→Text achieve Recall@10 values of 99.60% and 100.00%, respectively. Contact-related modalities are more challenging at the top rank: Tactile→Text and IMU→Text achieve Recall@1 values of 59.40% and 50.40%, increasing to 87.60% and 85.00% at Recall@10. These results indicate that the aligned representations recover cross-modal correspondence, while top-ranked matching remains less reliable for tactile and IMU observations than for visual observations.

Comparison with Controlled Alignment Baselines. We compare SomaAlign with two controlled alignment baselines: Pairwise InfoNCE [36], which aligns all modality pairs, and ImageBind-style Hub Alignment [21], which aligns each of the other five modalities with Remote RGB as the visual hub. Both controlled baselines use the same modality encoders and Uni-Touch pretrained initialization as SomaAlign. Their encodertraining configurations, including the optimizer, learning rate and schedule, batch configuration, and training budget, are matched to those of SomaAlign. All three methods follow the same data split and evaluation protocol. Both baselines use a fixed temperature of $\tau = 0 . 1 5$ without reconstruction-based reliability estimation, reliability-conditioned temperature modulation, or Stage-III refinement.

As shown in Table III, SomaAlign outperforms both baselines across all three retrieval metrics, improving R@1 by 21.04 and 18.49 percentage points over Pairwise InfoNCE and Hub Alignment, respectively.

To assess training variability, we evaluate three independently trained Pairwise InfoNCE checkpoints on the same fixed test subset under the retrieval protocol defined above. The mean ± sample standard deviation is $4 3 . 7 2 \pm 1 . 6 2 \% , \ 6 3 . 3 8 \pm 3 . 0 9 \%$ , and 69.53±3.54% for R@1,

![](images/65c70a130cf83bec842823935e61d0cf2fc7acda7aaef793124026ba16c8b505.jpg)

![](images/9846d09a64cc7027308f41a7f7a70b5067f721bfef99c24d436d69636c23e523.jpg)

![](images/3eb864ed9795a3e3ee25e5ec0bf31129f5ef5e6b19222a02296933402b620cbd.jpg)  
Fig. 8. Recall@1, Recall@5, and Recall@10 across 30 directed cross-modal retrieval tasks. Sensory-only retrieval uses paired temporal-window correspondence; text-related retrieval uses identical structured descriptions. Remote and Hand denote remote-view and hand-mounted RGB, respectively. Same-modality retrieval is excluded.

TABLE IV  
COMPARISON WITH PUBLISHED MULTIMODAL METHODS UNDER THE COMMON V–T–L RETRIEVAL PROTOCOL ON SOMA. RECALL VALUES ARE PERCENTAGES MACRO-AVERAGED OVER SIX DIRECTED TASKS.
<table><tr><td>Method</td><td> ${ \overline { { \mathrm { A v g . } } } }$  R@1</td><td> ${ \overline { { \mathrm { A v g . } } } }$  R@5</td><td> ${ \overline { { \mathrm { A v g . } } } }$  R@10</td></tr><tr><td>TVL-Adapted [9]</td><td>55.64</td><td>75.09</td><td>81.02</td></tr><tr><td>TLV-Link-Adapted [37]</td><td>40.51</td><td>60.65</td><td>69.89</td></tr><tr><td>UniTouch-Adapted [23]</td><td>48.16</td><td>64.47</td><td>76.86</td></tr><tr><td>SomaAlign</td><td>69.71</td><td>88.07</td><td>92.05</td></tr></table>

R@5, and R@10, respectively. Separately, evaluating one fixed SomaAlign checkpoint on five independently sampled 500- instance test subsets yields 60.78±1.64%, 84.03±1.13%, and 89.11±1.11%. The primary results in Table III and Fig. 8 are reported on the same fixed 500-instance test subset.

Comparison with Published Multimodal Methods. We compare SomaAlign with TVL [9], TLV-Link [37], and Uni-Touch [23], adapted to a common vision–tactile–language (V–T–L) retrieval setting on Soma using Hand RGB, tactile pressure, and structured English text. All methods follow the same experiment-level data split and are evaluated across the six directed V–T–L retrieval tasks using the correspondence rules defined above. SomaAlign retains six-modality training and is evaluated here on the shared V–T–L interface.

As shown in Table IV, SomaAlign achieves the highest average Recall across all three metrics, exceeding TVL-Adapted, the strongest baseline by average retrieval performance, by 14.07 and 11.03 percentage points in R@1 and R@10, respectively.

Robustness to Sensor Corruption. We evaluate retrieval robustness by adding Gaussian noise to tactile or IMU observations:

$$
\begin{array} { r } { \tilde { \mathbf { x } } = \mathbf { x } + \nu \sigma ( \mathbf { x } ) \mathbf { z } , \qquad \mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) , } \end{array}\tag{38}
$$

where $\sigma ( \mathbf { x } )$ is the standard deviation of the clean observation and $\nu$ controls the noise magnitude relative to this scale. We test $\nu \in \{ 0 , 0 . 0 1 , 0 . 0 3 , 0 . 0 5 , 0 . 1 , 0 . 3 , 0 . 7 \}$ , perturbing one modality at a time while keeping the others unchanged. SomaAlign, Pairwise InfoNCE, and Hub Alignment are evaluated under identical corruption settings, with results averaged over five noise seeds using fixed trained checkpoints and test samples.

TABLE V  
MEAN PROBE ACCURACY ACROSS THE FIVE CLASSIFICATION TASKS (%).
<table><tr><td>Representation</td><td>Linear</td><td>MLP</td></tr><tr><td>Pairwise InfoNCE</td><td>92.06</td><td>94.40</td></tr><tr><td>Uniform Prior</td><td>92.89</td><td>94.90</td></tr><tr><td>SomaAlign</td><td>93.84</td><td>95.45</td></tr></table>

Across the tested corruption levels, SomaAlign maintains higher R@1 and R@10 than both controlled baselines. $\mathbf { A } \mathbf { t } \ \nu =$ 0.7, SomaAlign retains 75.1%/73.4% of its clean R@1/R@10 under tactile corruption, compared with 70.3%/70.0% for Pairwise InfoNCE and 71.0%/70.7% for Hub Alignment. Under IMU corruption, the corresponding retention rates are 78.1%/72.9%, 73.8%/68.5%, and 73.2%/69.1%, respectively. Detailed results across all corruption levels are provided in the Supplementary Information.

## C. Downstream Classification Performance

We evaluate frozen 1024-dimensional non-text features extracted by SomaAlign using logistic regression and an MLP with a 512–256 architecture and ReLU activations. Structured annotations provide the target labels, while text features are excluded from the probe inputs. Independent classifiers are trained for hardness, grasping state, material, size, and shape, with data split by experiment ID to keep grasping trajectories disjoint across subsets.

Across the five tasks, the linear probe achieves a mean accuracy of 93.84% and a mean F1-score of 92.62%, while the MLP reaches 95.45% accuracy and 94.03% F1-score. Detailed per-task results are provided in the Supplementary Information.

We further compare Pairwise InfoNCE and a Uniform Prior variant under the same frozen-feature probing protocol. Uniform Prior sets $w _ { m , n } ~ = ~ 1$ for all modality pairs while retaining the reconstruction-derived modality reliabilities and the remaining training pipeline.

As shown in Table V, SomaAlign achieves the highest mean accuracy with both probe architectures, exceeding Pairwise InfoNCE and Uniform Prior by 1.78 and 0.95 percentage points with the linear probe and by 1.05 and 0.55 percentage points with the MLP, respectively.

TABLE VI  
FROZEN-ENCODER GRASP-OUTCOME CLASSIFICATION ACCURACY ON SOMA-SUCCESS (%), REPORTED AS MEAN ± STANDARD DEVIATION OVER FIVE FOLDS.
<table><tr><td rowspan="2">Representation</td><td colspan="2">Post-hoc</td><td colspan="2">Early</td></tr><tr><td>Linear</td><td>MLP</td><td>Linear</td><td>MLP</td></tr><tr><td rowspan="4">Pairwise InfoNCE Uniform Prior Hub Alignment SomaAlign</td><td></td><td>89.78±1.95 90.28±1.43 82.87±2.61 84.07±3.24</td><td></td><td></td></tr><tr><td></td><td>92.49±1.6492.38±1.1682.87±1.7585.07±1.84</td><td></td><td></td></tr><tr><td></td><td>89.18±1.06 90.98±0.70 81.67±2.1986.48±2.52</td><td></td><td></td></tr><tr><td></td><td>93.18±2.1894.59±1.9483.07±2.6084.07±1.55</td><td></td><td></td></tr></table>

## D. Frozen-Encoder Evaluation on Soma-Success

We evaluate the reuse of Soma-trained representations for grasp-outcome prediction on the independently collected Soma-Success dataset, which is not used during representation learning. All modality encoders remain frozen, and only linear or MLP probes are trained on Soma-Success.

For each trial, $\mathbf { x } _ { t , m }$ denotes the observation of modality m at stage $t \in \{ \mathrm { p r e } .$ , mid, post}, corresponding to $T _ { a } , \ T _ { b }$ and $T _ { c } ,$ respectively. The same stage-selection rule is applied independently of success/failure labels. Features from the available modalities are aggregated as

$$
\mathbf { z } _ { t } = \operatorname { A g g } \left( \left\{ E _ { m } ( \mathbf { x } _ { t , m } ) \mathbf { \Phi } \middle | m \in \mathcal { M } _ { t } \right\} \right) ,\tag{39}
$$

where $\mathcal { M } _ { t }$ is the set of available modalities at stage t and Agg(·) denotes the feature aggregation operation used by the probe.

Post-hoc classification uses all three observation stages:

$$
\begin{array} { r } { { \bf z } _ { \mathrm { a l l } } = [ { \bf z } _ { \mathrm { p r e } } ; { \bf z } _ { \mathrm { m i d } } ; { \bf z } _ { \mathrm { p o s t } } ] , } \end{array}\tag{40}
$$

whereas early prediction uses only pre-grasp and mid-grasp observations:

$$
\begin{array} { r } { \mathbf { z } _ { \mathrm { e a r l y } } = [ \mathbf { z } _ { \mathrm { p r e } } ; \mathbf { z } _ { \mathrm { m i d } } ] , } \end{array}\tag{41}
$$

where [·; ·] denotes concatenation. A probe with parameters $\phi$ predicts the binary outcome:

$$
\hat { y } = h _ { \phi } ( \mathbf { z } _ { s } ) , \qquad s \in \{ \mathrm { a l l } , \mathrm { e a r l y } \} ,\tag{42}
$$

with labels $y = 0$ for failure and $y = 1$ for success.

Linear and two-layer MLP probes are evaluated using fivefold cross-validation grouped by object identity. All trials associated with the same object, including both gripper configurations, are assigned to the same fold. In each iteration, the probes are trained on four folds and evaluated on the remaining fold, while all modality encoders remain frozen. Each trial therefore receives one out-of-fold prediction from a probe trained without any trials of that object. After quality control, the dataset contains 998 valid trials, comprising 600 successful and 398 failed grasps.

As shown in Table VI, SomaAlign achieves the highest mean post-hoc accuracy with both probes: 93.18% for the linear probe and 94.59% for the MLP. These results support the reuse of its frozen representations for grasp-outcome classification using all three observation stages.

Early-prediction performance remains competitive when only pre-grasp and mid-grasp observations are available. SomaAlign achieves the highest mean linear accuracy of 83.07%, while its MLP probe reaches 84.07%, compared with 86.48% for the best-performing Hub Alignment. These results indicate that the frozen SomaAlign representations retain useful predictive information even before the post-grasp observation becomes available. Confusion matrices obtained by pooling the out-of-fold predictions across all five folds are provided in the Supplementary Information.

## E. Cross-Modal Generation Analysis

We conduct an auxiliary leave-one-out generation experiment to assess whether aligned features from the remaining modalities support reconstruction of a missing stream. For each target modality, a conditional diffusion model uses features from the other five modalities as conditioning inputs. For visual, depth, and tactile modalities, generation is performed in a VAE-compressed latent space, whereas IMU sequences are modeled directly with a 1D U-Net due to their lowdimensional temporal structure. Qualitative generation examples are provided in the Supplementary Information.

We further evaluate attribute-level semantic preservation using two linear-probe strategies. Single-Modality Probe trains and tests the classifier using only features from the generated target modality. Multi-Modality Probe trains the classifier on averaged features from all six real modalities and replaces only the target modality with its generated counterpart at test time.

Accuracy retention is defined as classification accuracy under the generated-data condition divided by that under the corresponding real-data condition. As shown in Fig. 9, singlemodality retention varies across targets: Depth achieves the highest retention, the two RGB streams show intermediate retention, and IMU achieves the lowest at 44.9%. Tactile also exhibits high retention, although its relatively low realdata baseline accuracy limits the absolute classification performance implied by this ratio.

Average multi-modality retention remains at 99.8%–100.0% across the evaluated target modalities. Thus, replacing one modality with its generated counterpart leaves fused classification performance close to the all-real baseline, despite larger losses in standalone semantic retention for some modalities.

## F. Multimodal Question Answering

We further evaluate whether SomaAlign representations support language-based reasoning over soft robotic multimodal observations. A LLaMA-based language model [38] is fine-tuned using the aligned features from Remote RGB, Hand RGB, Depth, Tactile, and IMU observations. QAaware augmentation constructs 44 training variants per sample, including one original structured description and 43 question/instruction–answer variants covering object descriptions, identity, individual attributes, grasping state, and multiattribute queries.

For quantitative evaluation, the model generates a structured description for each of the 15,085 Soma test samples. Exact Match is computed by case-insensitive full-string comparison with the reference description; normalized Edit Distance and BLEU-4 are additionally reported.

![](images/c54029ac3bbf849252eb5508fe28ec2ddd0e25c3a879dd1e6e2761ab4c756cdd.jpg)  
(a)

![](images/6b848f0eda24bb3a764e3c3fdb4d8176aeb71cb18263a9ec9199ec9f2946438b.jpg)  
(b)

![](images/661975c51c64a2d8bad1d81f06477d80ab5328e5fe6b18ebfb9e740bac2c325b.jpg)  
Fig. 9. Attribute classification accuracy retention under cross-modal generation, measured relative to the corresponding real-data baseline. (a) Single-modalit probes. (b) Multi-modality probes with one generated modality replacing its real counterpart. (c) Average retention for the two strategies.

TABLE VII  
OVERALL MULTIMODAL QUESTION-ANSWERING METRICS ON THE SOMA TEST SPLIT.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Test samples</td><td>15,085</td></tr><tr><td>Exact Match</td><td>92.0%</td></tr><tr><td>Mean normalized Edit Distance</td><td>0.0059</td></tr><tr><td>Mean BLEU-4</td><td>0.9878</td></tr></table>

As shown in Table VII, 13,880 of the 15,085 generated descriptions exactly match their references. The mean normalized Edit Distance of 0.0059 and BLEU-4 score of 0.9878 further indicate high agreement with the reference descriptions. A complementary field-level evaluation yields a pooled attribute accuracy of 98.41%. Detailed field-level results, gripper-specific statistics, and qualitative question-answering examples are provided in the Supplementary Information.

## G. Ablation Studies

We evaluate five core variants: (1) w/o Stage I removes reconstruction-based reliability estimation and uses uniform pairwise reliability; (2) w/o Adaptive Temp replaces reliability-conditioned temperature modulation with $\tau = 0 . 1 5 ;$ (3) w/o Stage III removes the entire conflict-aware refinement stage; (4) w/o Center loss removes $\mathcal { L } _ { \mathrm { c e n t e r } } ;$ and (5) w/o Intra loss removes $\mathcal { L } _ { \mathrm { i n t r a } }$ . Other hyperparameters are kept unchanged except where required by the ablation.

Table VIII shows that removing Stage I or replacing adaptive temperature modulation reduces R@1 by 15.10 and 15.77 percentage points, respectively. Both variants also reduce R@5 and R@10, supporting the contribution of reconstructionderived reliability and its use in temperature modulation.

Removing Stage III produces the largest R@1 decrease among the core variants, at 40.32 percentage points. Removing its center-protection or intra-sample refinement loss decreases R@1 by 28.56 and 36.06 percentage points, respectively. These results support the contribution of both geometric refinement losses. Notably, w/o Stage III and w/o Intra loss achieve higher Fisher scores than SomaAlign, and the latter also achieves a higher CHI. Their lower retrieval performance shows that these separability metrics alone do not capture cross-modal correspondence.

Effect of the Modality-Pair Prior. The Uniform Prior variant defined above achieves R@1, R@5, and R@10 values of 45.47%, 67.41%, and 74.98%, respectively, with $w _ { m , n } = 1$ for all modality pairs. Its R@1 is 17.57 percentage points below SomaAlign, supporting the use of the specified modality-pair priors in the reliability-conditioned alignment scheme.

Effect of Conflict Modulation Strategy. We compare four strategies while keeping the remaining Stage-III components unchanged. No-Conflict uses an intra-sample refinement coefficient of 1. Fixed-Weight uses a constant matched to the average coefficient of Mean-Gamma. Sample-Gamma applies the sample-specific coefficient $1 - \operatorname { s g } ( \gamma _ { i } )$ , whereas Mean-Gamma uses the batch-level coefficient $1 - \bar { \gamma } _ { B }$ adopted in SomaAlign.

Mean-Gamma achieves the highest Recall across all three metrics (Table IX), improving R@1 by 8.60, 4.43, and 4.04 percentage points over No-Conflict, Fixed-Weight, and Sample-Gamma, respectively. Fixed-Weight and Sample-Gamma perform similarly. These results favor batch-level adaptation of refinement strength, which applies a shared coefficient while preserving relative sample weighting within the intra-sample loss.

## H. Inference Complexity and Runtime

SomaAlign representation inference retains only the modality-specific encoders required for the six sensing modalities, resulting in 1,332.2M active inference parameters. Encoding one six-modality sample requires 1.666 TFLOPs, and FP16 inference takes 43.67 ms per sample on the evaluated hardware. Detailed runtime, batch-processing, and parameterstorage statistics are provided in the Supplementary Information.

## I. Discussion

The experimental findings suggest that multimodal representation learning for soft robotic perception requires both accommodating modality-dependent reliability and preserving meaningful cross-modal correspondence during feature alignment. Soft robotic sensing is affected by deformation-dependent contact responses, transient mechanical disturbances, and heterogeneous modality-specific noise, making it difficult to assume that all modality pairs should be aligned with identical strength. The results therefore support a progressive strategy in which modality reliability is first estimated, then incorporated into cross-modal optimization, and finally complemented by geometry-preserving semantic refinement.

TABLE VIII  
ABLATION OF THE CORE COMPONENTS IN SOMAALIGN.
<table><tr><td>Model</td><td>R@1 (%)</td><td>R@5 (%)</td><td>R@10 (%)</td><td>k-NN (%)</td><td>Fisher</td><td>CHI</td></tr><tr><td>SomaAlign</td><td>63.04</td><td>85.55</td><td>90.66</td><td>87.2</td><td>1967.1</td><td>7.318</td></tr><tr><td>w/o Stage I</td><td>47.94</td><td>73.92</td><td>85.79</td><td>85.6</td><td>1863.9</td><td>7.241</td></tr><tr><td>w/o Adaptive Temp (τ = 0.15)</td><td>47.27</td><td>73.63</td><td>84.67</td><td>86.6</td><td>1937.9</td><td>7.168</td></tr><tr><td>w/o Stage III</td><td>22.72</td><td>50.85</td><td>68.45</td><td>86.4</td><td>2220.7</td><td>7.315</td></tr><tr><td>w/o Center loss</td><td>34.48</td><td>65.26</td><td>73.64</td><td>86.2</td><td>1928.0</td><td>6.282</td></tr><tr><td>w/o Intra loss</td><td>26.98</td><td>57.51</td><td>65.17</td><td>86.2</td><td>2261.7</td><td>7.661</td></tr></table>

TABLE IX  
ABLATION OF STAGE-III CONFLICT MODULATION STRATEGIES. RECALL VALUES ARE PERCENTAGES.
<table><tr><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>No-Conflict</td><td>54.44</td><td>80.72</td><td>86.38</td></tr><tr><td>Fixed-Weight</td><td>58.61</td><td>83.57</td><td>89.39</td></tr><tr><td>Sample-Gamma</td><td>59.00</td><td>83.48</td><td>89.05</td></tr><tr><td>Mean-Gamma</td><td>63.04</td><td>85.55</td><td>90.66</td></tr></table>

An important implication of the reliability-aware design is that modality reliability need not be treated only as a featurelevel weighting factor. In SomaAlign, reconstruction-derived reliability modulates the temperature of pairwise contrastive learning and therefore changes the sharpness of the alignment objective itself. More reliable modality pairs can receive stronger alignment constraints, whereas less reliable pairs are optimized more conservatively. This is particularly relevant for heterogeneous visual, tactile, inertial, and semantic observations whose stability and noise characteristics can differ substantially. Reliability thus serves not only as an estimate of modality stability, but also as a mechanism for regulating how strongly different modalities are brought into a shared representation space.

The experiments also highlight that global feature separability and cross-modal correspondence are distinct representation properties. A feature space may exhibit favorable class-level organization while still failing to correctly associate heterogeneous observations originating from the same physical interaction. This distinction is important for multimodal robotic perception, where semantic organization and observation-level correspondence are both necessary. The Stage-III design can be interpreted from this perspective: intrasample refinement promotes local cross-modal compactness, whereas center protection limits excessive displacement from the Stage-II reference geometry. These objectives therefore address complementary aspects of representation refinement, balancing local consistency against preservation of the broader

feature structure.

The conflict-modulation results provide a further methodological observation. Sample-level conflict estimates are useful indicators of the current multimodal consistency state, but they need not directly define relative sample importance. Directly applying sample-specific conflict weights may couple optimization too strongly to instantaneous local inconsistency. By contrast, batch-level aggregation allows the overall refinement strength to adapt to the current cross-modal state while preserving the relative contributions of individual samples within the batch. This suggests that conflict information is more appropriately used as an optimization-state signal than as a direct sample-weighting criterion.

The downstream evaluations further indicate that multimodal representation quality is inherently multidimensional. Cross-modal retrieval measures correspondence across modalities, physical-attribute probes assess semantic accessibility, Soma-Success evaluates the reusability of frozen representations on independently collected data, and generation and question answering provide additional views of the information retained in the shared representation. Differences across these tasks therefore reflect distinct representation properties rather than a single notion of performance. In particular, the Soma-Success results show that the learned features remain useful under both complete and partial observation settings, although downstream performance can vary with the available observations and probe architecture. This also suggests that representation quality should be assessed through complementary tasks rather than inferred from any single metric.

Several aspects of the current framework define the scope of these conclusions. The Stage-I reliability score is derived from masked reconstruction and should therefore be interpreted as a reconstruction-based proxy for relative modality stability rather than a direct measurement of physical sensor noise. The uniform-masking diagnostic suggests that the observed cross-modal differences are not determined solely by predefined masking probabilities, while explicit sensor calibration or uncertainty modeling could provide more physically grounded estimates. The structured text annotations provide consistent semantic supervision but limit linguistic diversity, motivating future extensions toward more open-ended descrip tions and question formulations. In addition, Soma-Success demonstrates frozen-encoder reuse with lightweight supervised probes rather than zero-shot generalization, and broader evaluation across platforms, sensor configurations, and endeffectors would further characterize transferability. Extending SomaAlign toward action-conditioned prediction, robot world models [39], [40], and closed-loop soft robotic control is a natural next step.

## V. CONCLUSION

We presented Soma, Soma-Success, and SomaAlign for multimodal perception in soft robotic grasping. Soma provides software-coordinated, cycle-aligned Remote RGB, Hand RGB, depth, tactile, and IMU observations together with structured text descriptions. SomaAlign combines reliability-aware multimodal alignment with batch-adaptive conflict-aware geometrypreserving refinement to learn a shared representation of these heterogeneous observations.

Experiments demonstrate the utility of the learned representations for cross-modal retrieval and physical attribute recognition, with ablations supporting the contributions of reliability-conditioned alignment and conflict-aware refinement. Additional generation and question-answering experiments illustrate downstream uses of the aligned features. Frozen-encoder evaluation on the independently collected Soma-Success dataset yields the highest mean post-hoc graspoutcome accuracy among the evaluated representations, while early-prediction performance varies with probe architecture.

Future work will extend the data coverage to more diverse manipulation scenarios, incorporate physically grounded reliability estimates, and explore the use of SomaAlign representations in action-conditioned prediction and closed-loop soft robotic control.

## REFERENCES

[1] H. -S. Fang et al., “AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains,” in IEEE Transactions on Robotics, vol. 39, no. 5, pp. 3929-3945, Oct. 2023.

[2] T. Yue, C. Lu, K. Tang, Q. Qi, Z. Lu, L. Y. Lee, H. Bloomfield-Gadelha, and J. Rossiter, “Embodying soft robots with octopus-inspired hierarchical suction intelligence,” in Science Robotics, vol. 10, no. 102, Art. no. eadr4264, May 2025.

[3] R. Zhang, U. Yoo, Y. Li, A. Argawal, and W. Yuan, “PneuGelSight: Soft robotic vision-based proprioception and tactile sensing,” in The International Journal of Robotics Research, OnlineFirst, Oct. 2025, doi: 10.1177/02783649251378153.

[4] H. Wang et al., “Grasping State Analysis of Soft Manipulator Based on Flexible Tactile Sensor and High-Dimensional Fuzzy System,” in IEEE/ASME Transactions on Mechatronics, vol. 30, no. 3, pp. 2165-2176, June 2025.

[5] H. Wang et al., “Grasping State Analysis of Soft Manipulator Based on Flexible Tactile Sensor Array,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Kyoto, Japan, Oct. 2022, pp. 8515-8520.

[6] H. Wang et al., “Grasping State Analysis of Multi-DOF Soft Manipulators Based on Multimodal Sensing and Deep Spiking Fuzzy Network,” in IEEE Transactions on Fuzzy Systems, vol. 33, no. 8, pp. 2397-2410, Aug. 2025.

[7] J. Urain et al., “A Survey on Deep Generative Models for Robot Learning From Multimodal Demonstrations,” in IEEE Transactions on Robotics, vol. 42, pp. 60-79, 2026.

[8] G. Gu, N. Zhang, H. Xu, et al., “A soft neuroprosthetic hand providing simultaneous myoelectric control and tactile feedback,” in Nature Biomedical Engineering, vol. 7, pp. 589-598, 2023.

[9] L. Fu, G. Datta, H. Huang, W. C.-H. Panitch, J. Drake, J. Ortiz, M. Mukadam, M. Lambeta, R. Calandra, and K. Goldberg, “A touch, vision, and language dataset for multimodal alignment,” in Proceedings of the 41st International Conference on Machine Learning (ICML), Vienna, Austria, Jul. 2024, pp. 14080-14101.

[10] R. Gao, Y.-Y. Chang, S. Mall, L. Fei-Fei, and J. Wu, “ObjectFolder: A dataset of objects with implicit visual, auditory, and tactile representations,” in Proceedings of the 5th Conference on Robot Learning (CoRL), London, UK, Nov. 2021, pp. 466-476.

[11] R. Gao, Z. Si, Y.-Y. Chang, S. Clarke, J. Bohg, L. Fei-Fei, W. Yuan, and J. Wu, “ObjectFolder 2.0: A multisensory object dataset for Sim2Real transfer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, Jun. 2022, pp. 10588-10598.

[12] R. Gao, Y. Dou, H. Li, T. Agarwal, J. Bohg, Y. Li, L. Fei-Fei, and J. Wu, “The ObjectFolder Benchmark: Multisensory learning with neural and real objects,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, Jun. 2023, pp. 17276-17286.

[13] R. Calandra, A. Owens, M. Upadhyaya, W. Yuan, J. Lin, E. H. Adelson, and S. Levine, “The feeling of success: Does touch sensing help predict grasp outcomes?” in Proceedings of the 1st Conference on Robot Learning (CoRL), Mountain View, CA, USA, Nov. 2017, pp. 314-323.

[14] F. Yang, C. Ma, J. Zhang, J. Zhu, W. Yuan, and A. Owens, “Touch and Go: Learning from human-collected vision and touch,” in Advances in Neural Information Processing Systems 35 (NeurIPS 2022), New Orleans, LA, USA, Nov.–Dec. 2022, pp. 8081-8103.

[15] S. Suresh, Z. Si, S. Anderson, M. Kaess, and M. Mukadam, “Midas-Touch: Monte-Carlo inference over distributions across sliding touch,” in Proc. Conf. Robot Learning (CoRL), Auckland, New Zealand, Dec. 2022, vol. 205, pp. 319–331.

[16] Y. Li, J.-Y. Zhu, R. Tedrake, and A. Torralba, “Connecting touch and vision via cross-modal prediction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Long Beach, CA, USA, Jun. 2019, pp. 10609- 10618.

[17] W. Yuan, Y. Mo, S. Wang and E. H. Adelson, “Active Clothing Material Perception Using Tactile Sensing and Deep Learning,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), Brisbane, QLD, Australia, May 2018, pp. 4842-4849.

[18] X. Zhang, S. Diao, T. Yang, Y. Qin, and N. Sun, “Modeling and control for PAM-actuated robots: An overview,” in Smart Materials and Structures, vol. 34, no. 10, Art. no. 103001, Oct. 2025.

[19] Z. Chen et al., “Data-Driven Methods Applied to Soft Robot Modeling and Control: A Review,” in IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 2241-2256, 2025.

[20] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 139, pp. 8748–8763, Jul. 2021.

[21] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “ImageBind: One embedding space to bind them all,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Vancouver, BC, Canada, Jun. 2023, pp. 15180–15190.

[22] S. Li and H. Tang, “Multimodal alignment and fusion: A survey,” in International Journal of Computer Vision, vol. 134, no. 3, Art. no. 103, Feb. 2026.

[23] F. Yang, C. Feng, Z. Chen, H. Park, D. Wang, Y. Dou, Z. Zeng, X. Chen, R. Gangopadhyay, A. Owens, and A. Wong, “Binding touch to everything: Learning unified multimodal tactile representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, Jun. 2024, pp. 26340-26353.

[24] R. Feng, J. Hu, W. Xia, T. Gao, A. Shen, Y. Sun, B. Fang, and D. Hu, “AnyTouch: Learning unified static-dynamic representation across multiple visuo-tactile sensors,” in Proc. Int. Conf. Learn. Represent. (ICLR), Singapore, Apr. 2025. Available: arXiv:2502.12191.

[25] W. Ma, X. Cao, Y. Zhang, C. Zhang, S. Yang, P. Hao, B. Fang, Y. Cai, S. Cui, and S. Wang, “CLTP: Contrastive language-tactile pre-training for 3D contact geometry understanding,” in Biomimetic Intelligence and Robotics, Art. no. 100324, Mar. 2026.

[26] C. Higuera, A. Sharma, T. Fan, C. K. Bodduluri, B. Boots, M. Kaess, M. Lambeta, T. Wu, Z. Liu, F. R. Hogan, and M. Mukadam, “Tactile beyond pixels: Multisensory touch representations for robot manipulation,” in Proc. Conf. Robot Learning (CoRL), vol. 305, pp. 105–123, Seoul, Korea, Sept. 2025.

[27] S. Wang, L. Weng, L. Hao, S. Zuo, Z. Yang, M. Wang, and X. Du, “Design and application of multimodal visual-tactile sensor for object information perception,” in Sensors and Actuators A: Physical, vol. 400, Art. no. 117529, Apr. 2026.

[28] Y. Dong et al., “Look-to-Touch: A Vision-Enhanced Proximity and Tactile Sensor for Distance and Geometry Perception in Robotic Manipulation,” in IEEE/ASME Transactions on Mechatronics, early access, doi: 10.1109/TMECH.2025.3645239.

[29] Y. Li, C. Guo, J. Ren, B. Chen, C. Cheng, H. Zhang, and H. Lu, “SoftGrasp: Adaptive grasping for dexterous hand based on multimodal

imitation learning,” Biomimetic Intelligence and Robotics, vol. 5, no. 2, Art. no. 100217, 2025.

[30] Z. Wan et al., “VinT-6D: A large-scale object-in-hand dataset from vision, touch and proprioception,” in Proceedings ofthe 41st International Conference on Machine Learning (ICML), vol. 235, Vienna, Austria, Jul. 2024, pp. 49921–49940.

[31] Q. Zhang, H. Wu, C. Zhang, Q. Hu, H. Fu, J. T. Zhou, and X. Peng, “Provable dynamic fusion for low-quality multimodal data,” in Proc. 40th Int. Conf. Mach. Learn. (ICML), Honolulu, HI, USA, Jul. 2023, vol. 202, pp. 41753–41769.

[32] S. Sidheekh, P. Tenali, S. Mathur, E. Blasch, K. Kersting, and S. Natarajan, “Credibility-aware multimodal fusion using probabilistic circuits,” in Proc. 28th Int. Conf. Artif. Intell. Statist. (AISTATS), Phuket, Thailand, May. 2025, vol. 258, pp. 2305–2313.

[33] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked´ autoencoders are scalable vision learners,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, Jun. 2022, pp. 15979–15988.

[34] F. Wang and H. Liu, “Understanding the behaviour of contrastive loss,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp. 2495-2504.

[35] Z. Huang, H. Chen, Z. Wen, C. Zhang, H. Li, B. Wang, and C. Chen, “Model-aware contrastive learning: Towards escaping the dilemmas,” in Proc. 40th Int. Conf. Mach. Learn. (ICML), Honolulu, HI, USA, Jul. 2023, vol. 202, pp. 13774–13790.

[36] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[37] N. Cheng, J. Xu, C. Guan, J. Gao, W. Wang, Y. Li, F. Meng, J. Zhou, B. Fang, and W. Han, “Touch100k: A large-scale touch-language-vision dataset for touch-centric multimodal representation,” Information Fusion, vol. 124, Art. no. 103305, 2025.

[38] H. Touvron et al., “LLaMA: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

[39] C.-L. Cheang, S. Chen, Z. Cui, Y. Hu, L. Huang, T. Kong, H. Li, Y. Li, Y. Liu, X. Ma, H. Niu, W. Ou, W. Peng, Z. Ren, H. Shi, J. Tian, H. Wu, X. Xiao, Y. Xiao, J. Xu, and Y. Yang, “GR-3 technical report,” arXiv preprint arXiv:2507.15493, 2025.

[40] Y. Liao, P. Zhou, S. Huang, D. Yang, S. Chen, Y. Jiang, Y. Hu, S. Liu, J. Luo, L. Chen, S. Yan, M. Yao, and G. Ren, “Genie Envisioner: A unified world foundation platform for robotic manipulation,” in Proc. Int. Conf. Learn. Represent. (ICLR), Rio de Janeiro, Brazil, Apr. 2026. Available: arXiv:2508.05635.