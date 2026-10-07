# Uncertainty Quantification Is Indispensable for Reliable Connectome-Based Graph Learning: A Narrative Review and Case Study

Mansooreh Pakravan<sup>1,∗</sup>

<sup>1</sup>Department of Electrical and Computer Engineering, Tarbiat Modares University, Al Ahmad Street, Tehran 111-14115, Iran

<sup>∗</sup>Corresponding author: Mansooreh Pakravan, mpakravan@modares.ac.ir ORCID: 0000-0003-3148-4071

## Abstract

While graph neural networks (GNNs) have shown substantial promise in connectome-based diagnostic classification, deterministic models inevitably suppress pipeline-induced noise and model ambiguities, yielding overconfident predictions. Although uncertainty quantification (UQ) is widely adopted in voxel-level segmentation, its role in connectomic graph learning remains largely unaddressed. This paper presents a comprehensive narrative review of UQ frameworks tailored to connectome graph learning alongside an empirical case study demonstrating the perils of uncalibrated predictions. We delineate sources of aleatoric and epistemic uncertainty across neuroimaging pipelines and review prominent UQ paradigms, from Bayesian approximations and ensemble methods to evidential learning and conformal prediction. In our case study, a temporal Graph Attention Network (GAT) trained on dynamic functional connectivity (dFC) matrices from the SUDMEX CONN dataset achieves 80.0% diagnostic accuracy (F1 = 0.794) for Cocaine Use Disorder. However, a post-hoc uncertainty audit via Monte Carlo dropout reveals severe overconfidence (ECE = 0.127), with misclassified subjects assigned prediction confidences up to 95%. This empirical divergence between discrimination and calibration underscores the confidence paradox in deep connectomics. Our findings establish that rigorous UQ, calibration, and selective prediction mechanisms are indispensable for deploying trustworthy graph-based biomarkers in clinical neuroscience.

## 1 Introduction

The quest to decode the human connectome has fundamentally shifted from traditional pairwise statistical correlations toward a sophisticated paradigm of geometric deep learning. Because macroscale brain organization is inherently governed by complex structural wiring and time-varying functional interactions, Graph Neural Networks (GNNs) have emerged as the standard mathematical language for computational neuroimaging [1, 2]. By representing discrete anatomical parcels or regions of interest (ROIs) as nodes and their synchronized blood-oxygen-level-dependent (BOLD) dynamics or tractography-derived streamlines as edges, graph-based architectures have demonstrated outstanding discriminatory power. These models have achieved remarkable success across diverse translational objectives, including the discovery of network biomarkers for psychiatric and neurodevelopmental disorders [1], mapping fine-grained sex-specific dimorphisms across distinct parcellation schemes [3, 4], and elucidating the topological reorganization and functional hub degradation associated with chronic pain syndromes [5]. Recent methodological extensions into explainable graph architectures and causal attention mechanisms have further advanced our mechanistic understanding, enabling researchers to isolate aberrant white-matter pathways in developmental prosopagnosia [6] and track the dynamic trajectory of narrative comprehension hubs during naturalistic continuous listening [7, 8].

However, as geometric deep learning transitions from exploratory proof-of-concept benchmarks toward safety-critical clinical deployment, a silent crisis of diagnostic trustworthiness has surfaced. The vast majority of neuroimaging machine learning studies remain exclusively evaluated on aggregate performance metrics, such as classification accuracy, sensitivity, specificity, and the area under the receiver operating characteristic curve (AUC) [9]. Although these population-averaged metrics confirm broad statistical capability, they provide no mathematical guarantee regarding the reliability or confidence of individual, subject-level predictions [10]. In high-stakes neurological and psychiatric assessments, a diagnostic framework reporting an impressive 80% overall accuracy remains clinically hazardous if its remaining 20% error margin is accompanied by unyielding, unwarranted statistical certainty [11]. Modern deep architectures are notorious for producing uncalibrated, overconfident softmax probabilities that diverge dramatically from empirical likelihoods of diagnostic correctness [10, 12].

This vulnerability is exceptionally pronounced in graph-structured neuroimaging pipelines. Recent theoretical and empirical analyses reveal that standard post-hoc calibration strategies, such as temperature scaling and vector scaling, frequently break down when applied to graph architectures [13]. The iterative message-passing schemes fundamental to GNNs propagate and aggregate localized node representations across topological neighborhoods, which inadvertently homogenizes feature distributions, dampens predictive entropy, and masks localized ambiguities [13]. In the context of whole-brain clinical network classification, such structural miscalibration exacerbates false-positive diagnostics and blinds the model to genuine neuroanatomical atypicalities, posing a direct threat to safe patient stratification [14].

Predictive uncertainty in neuroimaging is not a monolithic artifact but a multi-faceted phenomenon arising from distinct mathematical and physical origins. It is broadly partitioned into aleatoric uncertainty, reflecting irreducible stochasticity such as physiological noise, head motion artifacts, slice-timing discrepancies, and the profound biological heterogeneity underlying complex brain disorders [15, 16], and epistemic uncertainty, which stems from deficiencies in model knowledge, severe sample-to-feature ratio imbalances $( N \ll P )$ , and class imbalance [17, 18]. Crucially, epistemic uncertainty becomes dominant under inter-site domain shifts caused by scanner hardware diferences, field strengths, gradient non-linearities, and disparate acquisition protocols [19]. In multi-center functional connectomics, when a predictive model encounters out-of-distribution (OOD) geometries induced by scanner variability, unquantified epistemic uncertainty manifests as silent, highly confident diagnostic failures [19, 20].

To counteract these vulnerabilities, specialized uncertainty-aware frameworks have gradually been explored. Initial attempts incorporated fuzzy set theory into graph modeling to capture the intrinsic vagueness and boundary ambiguity of resting-state functional connectivity [21]. In particular, Type-II fuzzy GNNs demonstrated that modeling edge-weight imprecision enables the extraction of robust biological signatures that remain obscured within deterministic counterparts [4]. Similarly, fuzzy causal attention mechanisms have bolstered the stability of psychiatric classifications against noise-induced graph perturbations [21]. Parallel eforts within the field of probabilistic machine learning have leveraged Bayesian Neural Networks (BNNs), Monte Carlo (MC) Dropout, Deep Ensembles, Laplace approximations, and Evidential Deep Learning to estimate predictive posterior distributions and capture model hesitation [16, 22].

Despite these advances, a striking methodological disparity persists across the neuroimaging domain. While uncertainty quantification (UQ) and spatial calibration protocols have achieved widespread validation in localized voxel-level tasks, such as lesion and brain tumor segmentation [23, 24, 25], systematic UQ remains critically underrepresented in connectomics and whole-graph classification [22, 18]. Functional connectivity matrices are notoriously susceptible to low signal-to-noise ratios, arbitrary thresholding criteria, and atlas-induced structural biases [5]. When connectomic graphs are processed deterministically, upstream measurement errors propagate through iterative graph convolutional layers unchecked, compromising the integrity of clinical decision boundaries [22, 14].

To address this methodological divide and provide rigorous finite-sample error control, the machine learning community has increasingly turned toward Conformal Prediction [26, 27]. Conformal prediction ofers a distribution-free, model-agnostic framework that transforms brittle point predictions into statistically valid prediction sets guaranteed to cover the ground truth at a user-specified confidence level (1 − α) [28]. While the classical exchangeability assumption of conformal inference is routinely violated by the spatial and topological dependencies of brain networks, recent theoretical breakthroughs have established valid node-level and whole-graph-level conformal frameworks capable of handling non-exchangeable topological data [27, 29]. By shifting the diagnostic question from “which single class is most likely?” to “which set of candidate diagnoses contains the true clinical state with provable coverage guarantees?”, conformalized medical imaging establishes a principled safeguard for human-in-the-loop decision-making [28, 29].

This narrative review synthesizes the current state of uncertainty-aware machine learning in neuroscience and neuroimaging. We first introduce a problem-oriented taxonomy of uncertainty, distinguishing aleatoric and epistemic sources across neuroimaging pipelines. We then review the principal estimation paradigms used in the literature, including Bayesian approximations, Monte Carlo dropout, deep ensembles, evidential learning, Laplace-based inference, and conformal prediction. Next, we examine how these ideas have been reflected in current neuroscience applications, with particular attention to segmentation, diagnostic classification, and connectomic analysis. To ground the discussion, the review also presents an empirical case study in dynamic functional connectomics, illustrating how discrimination, calibration, and confidence may diverge in graph-based prediction. Together, these sections highlight the practical need for uncertainty-aware and bettercalibrated models in neuroimaging, and they motivate future work on trustworthy inference in clinical neuroscience.

## 2 Related Works: Uncertainty in Brain Modelling

Uncertainty is central to neuroimaging machine learning, capturing both intrinsic data noise and the limitations of model knowledge. Conventional deep models output point estimates (e.g., softmax vectors), which fail to distinguish between typical cases and those that are ambiguous or out-ofdistribution (OOD) [12, 30]. A more robust approach shifts from point estimates to predictive distributions $p ( \boldsymbol { y } ^ { * } \mid \boldsymbol { x } ^ { * } , \mathcal { D } )$ , which quantify both belief and confidence [31, 32].

## 2.1 Sources of Uncertainty: Aleatoric and Epistemic

Uncertainty is commonly decomposed into aleatoric and epistemic components [16]. Aleatoric uncertainty (data-intrinsic) arises from head motion, noise, or biological ambiguity (e.g., overlap between ADHD and ASD). It is irreducible and answers: “Is this observation inherently noisy?” Epistemic uncertainty (model-related) reflects limited knowledge, high when data deviates from training patterns (e.g., scanner shifts, rare subtypes). It answers: “Does the model have suficient experience with this case?”

Conceptualization of Uncertainty Types  
![](images/29d4549b599b9554ca46bcf49c2c173fdaf57a75e00209f6a99911ae789c0d25.jpg)  
Figure 1: Conceptual simulation of aleatoric and epistemic uncertainty. Persistent noise around the latent process represents aleatoric uncertainty. The widening predictive interval beyond the training boundary $( x = 7 )$ illustrates epistemic uncertainty due to extrapolation.

These concepts are illustrated in Fig. 1. Aleatoric uncertainty persists as scatter around the latent process, while epistemic uncertainty widens predictive intervals beyond the training boundary $( x = 7 )$

A Bayesian formulation provides a formal framework for this decomposition by averaging predictions over plausible parameters θ:

$$
p ( y ^ { * } \mid x ^ { * } , \mathcal { D } ) = \int p ( y ^ { * } \mid x ^ { * } , \theta ) p ( \theta \mid \mathcal { D } ) d \theta .\tag{1}
$$

Fig. 2 visualizes how agreement among plausible models minimizes epistemic uncertainty, whereas disagreement increases it, serving as a proxy for OOD detection.

## 2.2 Methodological Landscape for UQ

Various methods address these uncertainties, ranging from parameter-focused approaches to distributional modeling:

Epistemic and Ensemble Methods: Bayesian Neural Networks (BNNs) [31] and Laplace approximations [33] represent parameter uncertainty through distributions. Monte Carlo (MC) dropout [32] ofers a computationally eficient proxy by performing stochastic forward passes to estimate model disagreement (Fig. 3). Deep Ensembles [34] aggregate independently trained models, providing strong empirical calibration, while SWAG [35] approximates the posterior via gradient trajectory.

Evidential and Aleatoric Models: Evidential Deep Learning (EDL) [36] models the evidence supporting each class (often via a Dirichlet distribution), allowing for the distinction between low support and conflicting evidence. Conversely, heteroscedastic models explicitly predict inputdependent variance, directly capturing aleatoric uncertainty.

![](images/67f4d0a969beec671550ec38dd2af618de604236eeaa3b9e0940b5e53e559a9a.jpg)  
Figure 2: Bayesian view of predictive uncertainty: predictions are averaged over a posterior of plausible parameter configurations p(θ|D). Model disagreement signifies epistemic uncertainty.

Set-based and Vague Representations: Conformal Prediction (CP) [26] provides distributionfree prediction sets with rigorous coverage guarantees, moving beyond explicit probability modeling. Similarly, fuzzy systems [37] represent graded membership in overlapping categories, suitable when class boundaries are linguistically vague rather than probabilistically ambiguous.

## 2.3 Calibration and Reliability Assessment

In high-stakes clinical neuroimaging applications, raw predictive accuracy alone is an insuficient criterion for deployment; model calibration is critical to ensure that predicted confidence bounds faithfully reflect empirical likelihood [10]. Formally, a classifier is perfectly calibrated when the predicted confidence matches its true posterior accuracy:

$$
\mathbb { P } \left( \hat { Y } = Y \Big | \hat { P } = p \right) = p , \quad \forall p \in [ 0 , 1 ] .\tag{2}
$$

Empirical deviations from this ideal alignment fall into distinct calibration regimes, as illustrated in Fig. 4:

• Overconfidence (Fig. 4a): The model assigns high subjective confidence that drastically exceeds empirical accuracy $( { \hat { P } } > \operatorname { a c c } )$ , a pathology pervasive in deep graph architectures.

• Ideal Calibration (Fig. 4b): The empirical accuracy curve tracks the diagonal identity line perfectly across all confidence bins.

• Underconfidence (Fig. 4c): The model underestimates its predictive reliability $( { \hat { P } } < \operatorname { a c c } )$ ， leading to overly cautious clinical deferrals.

![](images/921dcefc9b1eb5366c17b6ebc1574006d99b754df6cd10da28d1bfc9f95fc7db.jpg)  
Figure 3: MC dropout estimation: stochastic masks generate an ensemble of sub-networks, where variance quantifies model uncertainty.

To systematically evaluate and rectify these calibration discrepancies, standard diagnostic and post-hoc mitigation pipelines are employed (Fig. 5):

1. Reliability Diagrams and ECE (Fig. 5a): Grouping predictions into M equally spaced confidence intervals $B _ { m }$ allows computing the Expected Calibration Error (ECE):

$$
\mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { N } \left| \operatorname { a c c } ( B _ { m } ) - \operatorname { c o n f } ( B _ { m } ) \right| ,\tag{3}
$$

which quantifies the weighted absolute deviation from the ideal diagonal.

2. Strict Proper Scoring Rules (Fig. 5b): The Brier score assesses both sharpness and probabilistic correctness by measuring the mean squared error between the predictive probability vector $\hat { \mathbf { p } } _ { i }$ and the one-hot ground-truth label ${ \bf y } _ { i } \mathrm { \cdot } $

$$
\mathrm { B S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } \left( \hat { p } _ { i , k } - y _ { i , k } \right) ^ { 2 } .\tag{4}
$$

3. Post-hoc Logit Recalibration (Fig. 5c): When a network exhibits structural overconfidence, Temperature Scaling [10] softens the unnormalized output logits $\mathbf { z } _ { i }$ via a single learned parameter $T > 0 ;$

$$
\hat { p } _ { i , k } = \frac { \exp ( z _ { i , k } / T ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { i , j } / T ) } .\tag{5}
$$

By setting $T > 1$ , the entropy of the predictive distribution is elevated without altering the argmax classification boundary or requiring costly architectural retraining.

![](images/8ca7e15d7ad47cdba4245219eecd6371c765a0fde31c45ae4e07c40d3fafcaf8.jpg)  
Figure 4: Calibration regimes in predictive modeling: (a) overconfidence, where assigned probabilities outpace true accuracy; (b) ideal calibration, adhering to the diagonal identity; and (c) underconfidence, where the network is more accurate than its output probabilities indicate.

## 3 Empirical Case Study: A Cautionary Tale

## 3.1 Study Objective and Analytical Rationale

To illustrate why predictive performance alone is insuficient for trustworthy neuroimaging machine learning, we conducted an empirical case study using subject-specific dynamic functional brain graphs. The central purpose of this analysis was not to claim that a high-performing classifier is necessarily unreliable, but to examine whether its confidence estimates were consistent with its actual predictive correctness. In particular, we investigated the following apparent paradox: a graph neural network (GNN) may achieve an accuracy close to 80% while assigning high predictive uncertainty to a substantial proportion of individual subjects.

This situation is plausible in neuroimaging because the brain is not organized according to sharply separated diagnostic categories. Subjects within the same clinical group may exhibit diferent patterns of functional dysconnectivity, whereas subjects from diferent groups may share partially overlapping network profiles. Consequently, a classifier may learn discriminative populationlevel patterns while remaining uncertain about individual cases. This distinction is especially important in substance-use and psychiatric neuroimaging, where biological heterogeneity, measurement noise, motion-related artifacts, and site-specific efects may coexist [15, 12].

## 3.2 Dataset and Participant Characteristics

We used the publicly available SUDMEX CONN dataset, a multimodal neuroimaging resource developed to investigate substance-use disorders [38]. The cohort included 138 participants, comprising 74 individuals diagnosed with cocaine use disorder (CUD) and 64 age- and sex-matched healthy controls (HC). Although the database also contains structural MRI and difusion-weighted imaging, the present analysis used only resting-state functional magnetic resonance imaging (rsfMRI) data for dynamic functional-connectivity analysis and graph-based classification.

![](images/c93b8b7cf1b314b520d664a5ae82133501edceafb4899175853e3439d4797c17.jpg)

![](images/2467c7c86146e1a56717591e5d93bffd8607dfc33d856ed6ff9b9faa90a589f3.jpg)  
b) Brier Score (Probabilistic Error)

![](images/720b5399410249a74e3304dd524053ce68c677f87209ccbf438f44eb2936959a.jpg)  
c) Post-hoc Temperature Scaling

![](images/3841323cd26d69a6a889fba455d44df3193ef079c09181322aafda097f31f649.jpg)  
Figure 5: Comprehensive calibration assessment and mitigation framework: (a) Reliability diagram depicting the gap between binned confidence and empirical accuracy (ECE); (b) Brier score formulation for penalty-based probabilistic alignment; and (c) Temperature scaling mechanism for post-hoc logit entropy adjustment.

The demographic and clinical characteristics of the analytical cohort are summarized in Table 1. The CUD and HC groups were comparable in age and sex distribution. The mean age was $3 0 . 6 0 \pm$ 8.26 years in the CUD group and $3 0 . 9 9 \pm 7 . 2 5$ years in the HC group $( p = 0 . 4 2 )$ The CUD group included 65 males and 9 females, whereas the HC group included 51 males and 11 females. As expected, the CUD group had a significantly higher mean AMAI score than the HC group $( 1 3 3 . 4 5 \pm 5 0 . 6 3$ versus $1 0 7 . 9 6 \pm 5 0 . 6 4 , \ p \ < \ 0 . 0 0 1 )$ . The dataset also included information on substance-use history, psychiatric status, and psychosocial characteristics, including the Addiction Severity Index (ASI). These variables were used for cohort characterization and were not included as graph input features unless explicitly stated otherwise.

Table 1: Demographic and clinical characteristics of the SUDMEX CONN cohort.
<table><tr><td>Characteristic</td><td>CUD  $( n = 7 4 )$ </td><td> $\mathbf { H } \mathbf { C } \ ( n = 6 4 )$ </td></tr><tr><td>Age, years; mean ± SD</td><td> $3 0 . 6 0 \pm 8 . 2 6$ </td><td> $3 0 . 9 9 \pm 7 . 2 5$ </td></tr><tr><td>Sex, male/female</td><td>65/9</td><td>51/11</td></tr><tr><td>AMAI score; mean ± SD</td><td> $1 3 3 . 4 5 \pm 5 0 . 6 3$ </td><td> $1 0 7 . 9 6 \pm 5 0 . 6 4$ </td></tr></table>

## 3.3 rs-fMRI Preprocessing and Anatomical Parcellation

The rs-fMRI data were preprocessed using the Analysis of Functional NeuroImages (AFNI) software suite [39]. The preprocessing pipeline included slice-timing correction, rigid-body motion correction, coregistration of functional images to each participant’s anatomical T1-weighted image, spatial normalization to a common MNI template, and spatial smoothing using a Gaussian kernel with a full width at half maximum (FWHM) of 5 mm.

The data were then temporally band-pass filtered between 0.01 and 0.15 Hz. Nuisance regression was performed to reduce the influence of non-neuronal signals using white-matter, cerebrospinalfluid, and motion-related regressors, together with their temporal derivatives. Finally, a brain mask was applied to exclude non-brain voxels.

To reduce the dimensionality of the voxel-wise data, we used the Harvard–Oxford cortical structural atlas, specifically the cort-maxprob-thr25-2mm version [40, 39]. This parcellation contains 48 cortical regions of interest (ROIs). For each participant, the mean BOLD signal was extracted from every ROI, resulting in the regional time-series matrix

$$
\mathbf { X } ^ { \left( s \right) } \in \mathbb { R } ^ { T \times N } ,\tag{6}
$$

where s indexes the participant, $T = 3 0 0$ denotes the number of temporal samples, and $N = 4 8$ denotes the number of cortical nodes. Each column of $\mathbf { X } ^ { ( s ) }$ represents time series of one cortical region.

## 3.4 Construction of Dynamic Functional Brain Graphs

To represent temporal changes in functional connectivity, the regional time series of each participant were divided into K non-overlapping temporal windows of length $L = 1 0 0$ time points. Given $T = 3 0 0$ time points, this procedure yielded $\begin{array} { r } { K = \lfloor \frac { T } { L } \rfloor = 3 } \end{array}$ graph snapshots per participant.

For participant s and temporal window k, we calculated the partial correlation matrix $\mathbf { P } ^ { ( s , k ) } =$ $\left[ p _ { i j } ^ { \left( s , k \right) } \right] _ { i , j = 1 } ^ { N } ,$

where $p _ { i j } ^ { ( s , k ) }$ estimates the direct linear association between regions i and $j$ after statistically controlling for the remaining regions. To obtain non-negative edge weights, we used the magnitude of the partial correlations: $a _ { i j } ^ { ( s , k ) } = \left| p _ { i j } ^ { ( s , k ) } \right| , \qquad i \neq j$

and set $a _ { i i } ^ { ( s , k ) } = 0 .$

This transformation preserves the strength of the association but discards its sign; consequently, positive and negative functional relationships are treated as equally strong connections, which is a limitation of the adopted representation.

Because partial-correlation matrices are typically dense, we applied subject- and window-specific percentile-based sparsification. Let $\boldsymbol { \tau } ^ { ( s , k ) }$ denote the 75th percentile of the non-zero of-diagonal connection weights in the corresponding snapshot. The resulting weighted sparse graph was defined as $G ^ { ( s , k ) } = ( V , E ^ { ( s , k ) } )$ , where $( i , j ) \in E ^ { ( s , k ) } \quad \Longleftrightarrow \quad a _ { i j } ^ { ( s , k ) } \geq \tau ^ { ( s , k ) }$

The edge weight $a _ { i j } ^ { ( s , k ) }$ was retained for all surviving edges. Thus, each participant was represented by a sequence of three dynamic graph snapshots: ${ \mathcal { G } } ^ { ( s ) } = \{ G ^ { ( s , 1 ) } , G ^ { ( s , 2 ) } , G ^ { ( s , 3 ) } \}$

## 3.5 Node-Level Graph Features

In addition to the connectivity matrices, we extracted two node-level graph-theoretic features from every snapshot: unweighted node degree and mean neighbor degree. For an undirected graph $G = ( V , E )$ , the degree of node v is defined as $d ( v ) = | \{ u : ( u , v ) \in E \} |$

For the weighted graphs, the corresponding weighted degree, or node strength, was defined as $\begin{array} { r } { s ( v ) = \sum _ { u \in \mathcal { N } ( v ) } a _ { u v } } \end{array}$ ，

where $\mathcal { N } ( \dot { v } )$ denotes the set of neighbors of node v. Although node strength was available from the weighted adjacency matrix, the degree-based representation was used as the primary node feature. The mean neighbor degree was calculated as

$$
d _ { \mathrm { n n } } ( v ) = \frac { 1 } { | \mathcal { N } ( v ) | } \sum _ { u \in \mathcal { N } ( v ) } d ( u ) .\tag{7}
$$

Accordingly, the node-feature matrix for each graph snapshot was represented as $\mathbf { H } ^ { ( s , k ) } \in$ $\mathbb { R } ^ { N \times F } , \qquad F = 2$

where the two columns correspond to $d ( v )$ and $d _ { \mathrm { n n } } ( v )$ , respectively. These features provide a compact description of local connectivity and nodal influence while maintaining a manageable input dimensionality.

## 3.6 Graph Neural Network Classifier

The dynamic graph sequence was analyzed using a graph neural network (GNN) classifier. The first graph-convolutional operation was implemented using a graph-attention layer. For node $i ,$ the updated representation was computed as

$$
\mathbf { h } _ { i } ^ { \prime } = \sigma \left( \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } \mathbf { W } \mathbf { h } _ { j } \right) ,\tag{8}
$$

where $\mathbf { h } _ { i } \in \mathbb { R } ^ { d }$ is the input feature vector, $\mathbf { h } _ { i } ^ { \prime } \in \mathbb { R } ^ { d ^ { \prime } }$ is the updated feature vector, $\mathbf { W } \in \mathbb { R } ^ { d ^ { \prime } \times d }$ is a trainable linear transformation, and $\sigma ( \cdot )$ is a nonlinear activation function.

The attention coeficient between nodes i and j was defined as

$$
\alpha _ { i j } = \frac { \exp { ( \mathrm { L e a k y R e L U ( \mathbf { a } ^ { \mathsf { T } } [ \mathbf { W } \mathbf { h } _ { i } \mid  { W } \mathbf { h } _ { j } ] ) } ) } } { \displaystyle \sum _ { \ell \in \mathcal { N } ( i ) } \exp { ( \mathrm { L e a k y R e L U ( \mathbf { a } ^ { \mathsf { T } } [ \mathbf { W } \mathbf { h } _ { i } \mid  { W } \mathbf { h } _ { \ell } ] ) } ) } }\tag{9}
$$

where ∥ denotes vector concatenation and $\mathbf { a } \in \mathbb { R } ^ { 2 d ^ { \prime } }$ is a learnable attention vector. This mechanism enables the model to assign diferent importance to neighboring brain regions rather than treating all functional connections identically.

The node-level representations were subsequently aggregated into a graph-level representation for each temporal snapshot. Let READOUT(·) denote the graph-pooling operation. The snapshotlevel embedding was defined as $\mathbf { z } ^ { ( s , k ) } = \mathrm { R E A D O U T } \left( \left\{ \mathbf { h } _ { i } ^ { ( s , k ) } \right\} _ { i = 1 } ^ { N } \right)$

The embeddings from the three temporal windows were then combined to form a participantlevel representation: z<sup>(s)</sup> = AGGREGATE $( \mathbf { z } ^ { ( s , 1 ) } , \mathbf { z } ^ { ( s , 2 ) } , \mathbf { z } ^ { ( s , 3 ) } )$

Finally, a fully connected classification head produced the class probabilities for CUD and HC $\mathbf { p } ^ { ( s ) } = \mathrm { s o f t m a x } \left( \mathbf { W } _ { c } \mathbf { z } ^ { ( s ) } + \mathbf { b } _ { c } \right)$ , where $\mathbf { p } ^ { ( s ) } = \left[ p _ { \mathrm { C U D } } ^ { ( s ) } , p _ { \mathrm { H C } } ^ { ( s ) } \right]$

## 3.7 Model Training, Cross-Validation, and Hyperparameters

The temporal graph neural network was trained and evaluated using a 5-fold stratified crossvalidation (CV) scheme to ensure that each validation split preserved the empirical class ratio between CUD and HC participants. To guarantee reproducibility, random seeds were explicitly controlled across iterations $( \mathrm { s e e d } = 4 2 + \mathrm { f o l d } )$

The model architecture consisted of a multi-head Graph Attention Network backbone with $h = 8$ attention heads, an input feature dimension of $F = 2$ , a hidden dimensionality of $d _ { h } = 6 4$ and an output embedding dimension of $d _ { o } = 1 2 8$ . A dropout rate of $p = 0 . 7 0$ was applied across layers for regularization and to facilitate post-hoc Monte-Carlo Dropout uncertainty quantification. Optimization was performed using the Adam optimizer with an initial learning rate of $\eta = 1 0 ^ { - 3 }$ and an $L _ { 2 }$ weight decay coeficient of $\lambda = 1 0 ^ { - 4 }$ , minimizing the standard categorical cross-entropy loss function.

Training was executed for a maximum of 35 epochs per fold with a batch size of 32. To mitigate overfitting, a learning rate scheduler (ReduceLROnPlateau) dynamically halved the learning rate $( \mathrm { f a c t o r } = 0 . 5 )$ when validation loss plateaued for 3 consecutive epochs. Furthermore, early stopping with a patience of 10 epochs was employed, and the model weights yielding the minimum validation cross-entropy loss were checkpointed for final inference and downstream uncertainty auditing. Table 2 provides a consolidated summary of the model and training hyperparameters.

## 3.8 Classification Results

Table 3 reports the per-fold and aggregate classification performance of the temporal GAT model under 5-fold stratified cross-validation on the SUDMEX CONN dataset. Across all folds, the model achieved a mean accuracy of $0 . 8 0 0 \pm 0 . 0 6 8$ and a mean $F _ { 1 }$ score of $0 . 7 9 4 \pm 0 . 0 7 2$ , indicating consistent discrimination between CUD and HC participants. Fold 3 yielded the strongest individual performance $( \mathrm { a c c u r a c y } = 0 . 9 1 7 , F _ { 1 } = 0 . 9 1 6$ , best epoch = 26), while the remaining folds converged to accuracy values between 0.750 and 0.792. The relatively low inter-fold variance in accuracy $( \sigma ^ { 2 } = 4 . 6 9 \times 1 0 ^ { - 3 } )$ and ${ F _ { 1 } } \ ( { \sigma } ^ { 2 } = 5 . 1 7 \times { 1 0 ^ { - 3 } } )$ ) suggests that the learned representations generalise robustly across data splits despite the limited cohort size.

Table 2: Specification of network architecture and optimization hyperparameters.
<table><tr><td>Evaluation scheme</td><td>5-Fold Stratified Cross-Validation</td></tr><tr><td>GNN layer type</td><td>Graph Attention Network (GATConv)</td></tr><tr><td>Attention heads (h)</td><td>8</td></tr><tr><td>Input node features (F)</td><td>2 (d(v) and  $d _ { \mathrm { n n } } ( v ) )$ </td></tr><tr><td>Hidden / output dimensions</td><td>64 / 128</td></tr><tr><td>Dropout rate (p)</td><td>0.70</td></tr><tr><td>Optimizer</td><td>Adam  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ </td></tr><tr><td>Initial learning rate (η)</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Weight decay (λ)</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Maximum training epochs</td><td>35</td></tr><tr><td></td><td>ReduceLROnPlateau (factor = 0.5, patience = 3)</td></tr><tr><td>Learning rate scheduler Early stopping patience</td><td>10 epochs</td></tr></table>

Table 3: Per-fold and aggregate classification performance of the temporal GAT model under 5-fold stratified cross-validation. Metrics are computed on the held-out validation split of each fold; bold marks the best-performing fold. Mean and standard deviation are computed over the five folds.
<table><tr><td>Fold</td><td>Best Epoch</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td> $F _ { 1 }$  Score</td></tr><tr><td>Fold 1</td><td>35</td><td>0.7917</td><td>0.8111</td><td>0.7917</td><td>0.7822</td></tr><tr><td>Fold 2</td><td>24</td><td>0.7917</td><td>0.8529</td><td>0.7917</td><td>0.7822</td></tr><tr><td>Fold 3</td><td>26</td><td>0.9167</td><td>0.9286</td><td>0.9167</td><td>0.9161</td></tr><tr><td>Fold 4</td><td>30</td><td>0.7500</td><td>0.7600</td><td>0.7500</td><td>0.7450</td></tr><tr><td>Fold 5</td><td>28</td><td>0.7500</td><td>0.7600</td><td>0.7500</td><td>0.7450</td></tr><tr><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td><td></td><td> $0 . 8 0 0 \pm 0 . 0 6 8$ </td><td> $0 . 8 3 5 \pm 0 . 0 6 5$ </td><td> $0 . 8 0 0 \pm 0 . 0 6 8$ </td><td> $0 . 7 9 4 \pm 0 . 0 7 2$ </td></tr></table>

![](images/b572259f806d442cde20fce33953fccdcf150b550d80756dc99f5d0e38192944.jpg)  
Figure 6: Five-fold cross-validation learning curves of the temporal GAT model trained on the SUDMEX CONN connectome dataset. Left: cross-entropy loss trajectories for training (solid) and validation (dashed) sets across all five folds, together with fold-averaged curves (black: mean train; dark red: mean validation ± one standard deviation shaded). All folds exhibit rapid loss decrease within the first five epochs, with training loss approaching zero by epoch 35, while validation loss stabilises between 0.45 and 0.65, consistent with the applied dropout regularization $( p \ =$ 0.70) and early stopping. Right: corresponding accuracy curves. Mean validation accuracy (dark red) converges to approximately 0.80, with Fold 3 (teal dashed) reaching the highest validation accuracy (0.917) at epoch 26. The gap between mean training accuracy (black, ≈ 1.0) and mean validation accuracy reflects the high dropout rate used to support Monte-Carlo Dropout uncertainty quantification at inference time.

## 3.9 Empirical Uncertainty Audit: High Discrimination Does Not Guarantee Trustworthy Decisions

To rigorously quantify parameter uncertainty, Monte Carlo (MC) dropout was conducted across $M = 2 0 0$ stochastic forward passes with dropout rate $p = 0 . 7 0$ active during inference, while maintaining deterministic layer normalization states. From the resulting predictive distributions $p ( \boldsymbol { y } \mid \mathcal { G } ^ { ( s ) } , \mathbf { w } ^ { ( m ) } )$ , the mean posterior class probabilities were obtained as ¯p(y $\begin{array} { r } { \mathrm { ~ \bar { ~ } ~ } \stackrel { } { | } \mathcal { G } ^ { ( s ) } ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } p ( y } \end{array}$ | $\mathcal { G } ^ { ( s ) } , \mathbf { w } ^ { ( m ) } )$ . Total predictive uncertainty was quantified via normalized predictive entropy $\mathcal { H } [ \bar { p } ( y$ ${ \mathcal G } ^ { ( s ) } ) ]$ , whereas epistemic uncertainty was isolated through mutual information:

$$
\mathcal { Z } ( y , \mathbf { w } \mid \mathcal { G } ^ { ( s ) } , \mathcal { D } ) = \mathcal { H } [ \bar { p } ( y \mid \mathcal { G } ^ { ( s ) } ) ] - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathcal { H } [ p ( y \mid \mathcal { G } ^ { ( s ) } , \mathbf { w } ^ { ( m ) } ) ] ,\tag{10}
$$

reflecting parameter-induced model uncertainty decoupled from data-inherent ambiguity (aleatoric uncertainty, approximated via expected entropy).

Figure 7 provides an empirical uncertainty and calibration audit across all pooled cross-validation predictions. Rather than focusing solely on label accuracy, this analysis evaluates whether the subjective posterior confidence emitted by the connectomic GNN is commensurate with its empirical reliability.

Reliability analysis and empirical overconfidence (Panel a). Figure 7a displays the reliability diagram evaluating the agreement between mean predicted posterior confidence and empirical accuracy across discretised bins. While lower-confidence bins exhibit conservative, slightly underconfident predictions (e.g., bins near 0.70 confidence achieving perfect empirical accuracy), the dominant high-confidence bins $( > 0 . 8 5 )$ demonstrate pronounced overconfidence. Across all bins, the aggregate Expected Calibration Error is $\mathrm { E C E } = 0 . 1 2 7$ . Crucially, for predictions assigned posterior confidence exceeding 0.90, empirical correctness degrades to approximately 87.5%, revealing a non-trivial calibration gap where automated diagnostic certainty systematically outpaces true predictive precision.

Posterior confidence shift across outcomes (Panel b). Figure 7b contrasts the kernel density estimates of maximum posterior confidence, $\begin{array} { r } { p _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { c } \bar { p } ( y = c \mid \mathcal { G } ^ { ( s ) } ) } \end{array}$ , between correct predictions and misclassifications. While correctly classified connectomes span a broad confidence continuum (0.75–1.00), misclassifications do not cluster near the decision boundary $\left( p _ { \mathrm { m a x } } \approx 0 . 5 0 \right)$ Instead, error density peaks sharply between 0.92 and 0.95, with virtually all misclassifications exceeding the high-confidence operational threshold $( p _ { \mathrm { m a x } } = 0 . 8 0 )$ . This shift demonstrates that model failures are not driven by indecision, but represent confident misidentifications.

Aleatoric and epistemic uncertainty decomposition (Panel c). Figure 7c partitions the normalized uncertainty scores into total predictive entropy, expected entropy, and mutual information. Total predictive entropy exhibits a modest elevation for misclassified subjects relative to correct predictions. However, decomposing this signal reveals that this diference is driven primarily by expected entropy (data-level ambiguity). In contrast, mutual information (epistemic uncertainty) remains universally low (≤ 0.12) across both correct and incorrect cohorts, with substantial distribution overlap. Under MC dropout, the network expresses minimal subjective model uncertainty even when committing severe classification errors, indicating that parameter dispersion alone cannot act as a failsafe warning signal.

a)  
![](images/6014d7edee8e51c9026b889618a3e87a646d3e3715bf72adf1c889101b4c181c.jpg)

b)  
![](images/9565726290a98bde0ea616aac3084c944b08b70adf6798aed2ec430bd2caaa22.jpg)

c)  
![](images/98419e4b26c78a011641a3b1e80afd53d8c17b0e5a1a9227037f6a846fa16fef.jpg)  
d)

![](images/de21021f643c76fb306d82edb13dbd62fca36870b758493373fd0ba0ae535c3b.jpg)  
Figure 7: Uncertainty quantification and calibration audit of the temporal GNN on SUDMEX CONN connectomes. (a) Reliability diagram showing systematic overconfidence in dominant high-confidence bins, resulting in an expected calibration error of $\mathrm { E C E } = 0 . 1 2 7 .$ . (b) Posterior confidence density distributions across correct predictions (teal) versus misclassifications (red), highlighting that errors are sharply concentrated in the high-confidence regime $( p _ { \mathrm { m a x } } \ >$ 0.80). (c) Decomposition of normalized uncertainty into total predictive entropy, expected entropy (aleatoric), and mutual information (epistemic) for correct versus incorrect classifications. (d) Subject-level confidence paradox map (predictive confidence vs. predictive entropy), identifying high-certainty diagnostic failures (Subjects #10 and #11) residing within the critical risk region.

The confidence paradox and critical failure modes (Panel d). Figure 7d maps individual subjects across the confidence–entropy landscape. While most correct cases follow a normative trajectory where high confidence corresponds to low predictive entropy, critical exceptions emerge. Two specific participants (Subject #10 with $p = 0 . 9 2$ and Subject #11 with $p \ : = \ : 0 . 9 5 )$ reside squarely within the designated “Critical Risk Region” (high predictive confidence paired with elevated predictive entropy). These high-certainty failure modes illustrate the diagnostic hazard of naive thresholding: standard automated deployment protocols (e.g., “accept if $p _ { \mathrm { m a x } } > 0 . 8 0 ^ { \mathfrak { N } } )$ would uncritically accept these false classifications without triggering human neuroradiological review.

Together, these findings demonstrate that high discriminative capability (80.0% aggregate validation accuracy; Table 3) coexists with clinically significant overconfidence. Incorporating post-hoc calibration, epistemic decomposition, and selective referral mechanisms is therefore indispensable for translating graph-based connectomics into trustworthy clinical environments.

## 4 Discussion

This empirical case study demonstrates a critical methodological reality in connectome-based graph neural networks: high discriminative performance does not inherently translate to trustworthy probabilistic decisions at the individual-subject level. While our temporal GNN achieved robust cross-validation accuracy discrimination across cohorts (Table 3), comprehensive uncertainty quantification revealed systematic miscalibration and counter-intuitive error modes. In clinical decision support, relying solely on point metrics such as accuracy or $F _ { 1 }$ -score obscures whether the model’s assigned predictive confidence is commensurate with its empirical reliability.

The primary finding from our calibration audit is the presence of marked overconfidence within the dominant operational regime (Figure 7a). Across the cohort, the Expected Calibration Error $\mathrm { ( E C E = 0 . 1 2 7 ) }$ reflects a global mismatch of 12.7 percentage points between subjective model confidence and empirical correctness. Notably, while the model exhibited conservative behavior in intermediate confidence intervals (0.65–0.80), significant overconfidence emerged in the highestconfidence bin $\left( \bar { p } _ { \mathrm { m a x } } \approx 0 . 9 5 \right)$ , where empirical accuracy dropped to 87.5%. This generates a severe local calibration gap:

$$
\Delta _ { \mathrm { c a l i b r a t i o n } } \approx 0 . 9 5 0 - 0 . 8 7 5 = 0 . 0 7 5 ,\tag{11}
$$

indicating that roughly one in eight predictions emitted with near-absolute certainty was incorrect. Such high-confidence calibration failures pose substantial clinical hazards, as decision-makers are most inclined to accept automated outputs without secondary radiological inspection when probability estimates approach unity [10, 41].

This phenomenon is corroborated by the posterior confidence shift depicted in Figure 7b. Rather than misclassifications clustering as indeterminate cases near the decision boundary $\left( p _ { \mathrm { m a x } } \approx 0 . 5 0 \right)$ their density peaked sharply within the 0.92–0.95 interval, with nearly all erroneous assignments exceeding the operational confidence threshold of $p _ { \operatorname* { m a x } } = 0 . 8 0$ . In connectomics, where functional connectivity profiles exhibit substantial inter-individual heterogeneity and clinical boundaries are non-discrete, deep GNNs can exploit idiosyncratic topological features that yield overfitted, overconfident representations rather than expressing prudent hesitation.

Our decomposition into predictive, aleatoric, and epistemic uncertainty components (Figure 7c) ofers important mechanistic insight into this failure mode. Although total predictive entropy was marginally elevated for incorrect cases, this elevation was predominantly driven by expected entropy (aleatoric data ambiguity). Epistemic uncertainty, operationalized as mutual information under Monte Carlo (MC) dropout, remained uniformly suppressed $\left( \leq 0 . 1 2 \right)$ across both correct and erroneous predictions. Consequently, parameter sampling under standard MC dropout failed to provide an efective case-level warning signal for catastrophic classification errors. While MC dropout provides a computationally tractable approximation to deep Bayesian inference, these findings underscore its architectural limitations in active subspace exploration and support the notion that dropout distributions can underestimate epistemic variance in overparameterized graph encoders [32, 34].

The individual-level confidence–entropy map (Figure 7d) contextualizes the clinical gravity of this breakdown, illustrating the “confidence paradox.” While the normative trajectory couples high predictive confidence with minimal predictive entropy, Subjects #10 $( p = 0 . 9 2 )$ and #11 $( p =$ 0.95) occupied the critical risk quadrant—characterized by elevated entropy alongside near-certain classification outputs. In an automated triage pipeline, conventional confidence thresholding (e.g., “deploy if $p _ { \mathrm { m a x } } > 0 . 8 0 ^ { \mathfrak { N } } )$ ) would uncritically validate these false beliefs rather than routing them to human experts for adjudication. This highlights the insuficiency of naive probability thresholding and motivates the formal adoption of selective classification and risk–coverage budgeting [42, 43], wherein abstention mechanisms are governed jointly by post-hoc calibrated confidence and epistemic bounds.

Beyond algorithmic parameters, a fundamental and pervasive source of epistemic uncertainty in graph-based neuroimaging stems from anatomical and functional parcellation. Topological metrics—including node degree distributions, path lengths, and modular structure—are inherently conditional on the chosen atlas (e.g., Schaefer-200 vs. Lausanne-60). Coarse parcellations risk conflating functionally distinct territories into single nodes (introducing partial-volume artifacts), whereas ultra-fine parcellations inflate edge-estimation variance and topological noise. In existing deep pipelines, treating the parcellation atlas as a static ground truth introduces latent epistemic uncertainty that remains invisible to downstream GNN layers. Future trustworthy architectures must parameterize parcellation selection, leveraging multiscale topological ensembles or probabilistic graph structures to guard against atlas-induced overconfidence.

In conclusion, our audit demonstrates that establishing trustworthy connectomic AI requires looking beyond conventional classification benchmarks. Robust clinical deployment mandates an integrated evaluation paradigm combining post-hoc calibration methods (such as temperature scaling or conformal prediction), principled epistemic decomposition, and selective referral pathways to identify silent, high-certainty failure modes.

## 5 Conclusion

This narrative review highlights that uncertainty quantification (UQ) and post-hoc calibration must be regarded as foundational components of trustworthy connectome-based machine learning, rather than optional post-hoc diagnostics. Across canonical Bayesian formulations, scalable Laplace approximations, deep ensembles, evidential neural networks, and distribution-free conformal prediction, principled uncertainty estimates expose systematic vulnerabilities that remain entirely latent under conventional discrimination metrics such as accuracy, sensitivity, or ROC-AUC.

Our empirical connectomic audit demonstrates this divergence concretely: while the temporal GNN achieved solid aggregate discrimination under cross-validation (80.0% mean accuracy), rigorous post-hoc evaluation revealed notable miscalibration $\left( \mathrm { E C E } = 0 . 1 2 7 \right)$ and severe overconfidence in the dominant prediction regime. Most critically, classification errors did not manifest as ambiguous or boundary-adjacent cases; instead, the model frequently assigned extreme posterior certainty $( p _ { \mathrm { m a x } } > 0 . 9 0 )$ to diagnostic failures (e.g., Subjects #10 and #11), exemplifying the clinical hazard of the confidence paradox. This decoupling underscores a vital imperative for translational neuroimaging: point discriminative performance cannot be equated with individual-level decision reliability. For connectomic deep learning to safely inform clinical decision support, the joint reporting of post-hoc calibration, epistemic decomposition, and selective referral mechanisms must become standard scientific practice alongside standard discriminative benchmarks.

## 6 Future Directions

Bridging the persistent chasm between theoretical uncertainty quantification (UQ) and trustworthy, bedside clinical translation in connectomics demands rigorous methodological breakthroughs across several foundational axes. While empirical calibration metrics, such as the Expected Calibration Error (ECE) and standard Monte Carlo Dropout approximations, have begun permeating neuroimaging workflows, they remain fundamentally insuficient for high-stakes psychiatric and neurological decision support. In this section, we delineate six prioritized frontiers that address open mathematical, topological, and translational bottlenecks.

## 6.1 End-to-End Propagation of Upstream Acquisition and Parcellation Uncertainties

Conventional connectomic graph learning pipelines operate under an unstated and brittle assumption: that the reconstructed graph topology $\mathcal { G } = ( \gamma , \mathcal { E } , \mathbf { A } , \mathbf { X } )$ represents deterministic ground truth. In reality, node geometries and adjacency matrices are downstream projections of upstream pipelines saturated with stochasticity. Non-linear spatial coregistration ambiguities, partial volume averaging along complex cortical folds, tractographic artifacts such as false-positive streamline bottlenecking in difusion MRI, and arbitrary edge thresholding heuristically inject unquantified variance directly into subsequent message-passing operations.

To resolve this limitation, future architectures must transition from deterministic graph modeling toward continuous-space probabilistic representations and Stochastic Graph Neural Networks (SGNNs). Under this paradigm, edge existence probabilities and streamline densities are explicitly formalized as random adjacency tensors $\mathbf { A } \sim \mathbb { P } ( \mathbf { A }$ | dMRI, fMRI). Leveraging diferentiable Riemannian manifolds and non-Euclidean soft-assignment operators provides a mathematically grounded mechanism to propagate upstream voxel-level displacement covariances directly through topological convolutions $\begin{array} { r } { ( \mathbf { h } _ { v } ^ { ( l + 1 ) } = \sigma ( \sum _ { u } p ( e _ { u v } ) \mathbf { W } \mathbf { h } _ { u } ^ { ( l ) } ) ) } \end{array}$ , thereby preventing artificial overconfidence induced by rigid anatomical parcellations.

## 6.2 Conformal Prediction Beyond Exchangeability: Multi-Site Covariate and Domain Shifts

Conformal Prediction (CP) ofers a powerful distribution-free framework by guaranteeing finitesample predictive error bounds under the marginal coverage criterion 1 − α. Nevertheless, standard split-conformal guarantees strictly depend on the exchangeability of calibration and test distributions. This core assumption is systematically violated across multi-center neuroimaging consortia such as ADNI, ABIDE, and the UK Biobank, where inter-site variance is driven by disparate scanner field strengths (e.g., 1.5T, 3T, and 7T), vendor-specific pulse sequences, and demographic drift.

Rendering conformalized connectomics clinically robust necessitates adopting advanced nonexchangeable conformal formulations. Weighted conformal prediction can directly incorporate density-ratio weights $w ( \mathbf { x } ) = d \mathbb { P } _ { \mathrm { t a r g e t } } ( \mathbf { x } ) / d \mathbb { P } _ { \mathrm { s o u r c e } } ( \mathbf { x } )$ into non-conformity quantiles to neutralize continuous scanner drift. Simultaneously, group-conditional or Mondrian conformal calibration must be enforced to prevent marginal validity from masking severe under-coverage in vulnerable minority sub-populations or atypical diagnostic phenotypes. Complementing these methods with optimal transport and online adaptive conformal inference will enable dynamic recalibration as acquisition protocols and clinical demographics evolve in hospital deployment.

## 6.3 Decoupling Biological Transitions from Pipeline Noise in Spatiotemporal Connectomes

Dynamic functional connectivity (dFC) and structural–functional (SC–FC) coupling inherently conflate two distinct forms of variance: aleatoric uncertainty arising from the endogenous nonstationarity of brain state transitions across multistable attractor basins, and epistemic uncertainty stemming from short scan durations, hemodynamic lag, and the high-dimensional small-sample regime $( N \ll P )$ . Disentangling these mechanisms requires spatiotemporal graph state-space models coupled with evidential deep learning.

Within the context of structural–functional coupling, divergence along the principal unimodalto-transmodal functional gradient $\mathcal { F } _ { \pmb { \theta } } ( \mathbf { S } _ { i } )$ must be systematically decomposed. Elevated epistemic uncertainty should be leveraged algorithmically to flag ill-conditioned geometric projections, such as deep subcortical tract dropouts or regional BOLD signal dropouts. Conversely, localized high aleatoric variance must be preserved and interpreted as genuine metabolic and functional decoupling reflective of higher-order cognitive flexibility and evolutionary cortical expansion.

## 6.4 Uncertainty-Aware Explainability (XAI) and Topological Biomarker Stability

Post-hoc graph explainers—including GNNExplainer, SubgraphX, and graph attention mechanisms—are widely used to isolate critical functional subnetworks and hub regions for biomarker discovery. However, standard graph explainability remains deeply vulnerable to topological instability: imperceptible perturbations in connectivity matrices often trigger drastic shifts in saliency subgraphs without altering the downstream classification outcome.

A vital methodological priority lies in establishing Bayesian and conformalized graph explainability frameworks. Rather than generating single, brittle saliency attributions $\mathbf { M } \in [ 0 , 1 ] ^ { | \mathcal { V } | \times | \mathcal { V } | }$ future models should compute full posterior distributions over node and edge attributions:

$$
p ( \mathbf { M } \mid { \mathcal { G } } , \mathbf { y } ) = \int p ( \mathbf { M } \mid { \mathcal { G } } , \mathbf { y } , \theta ) p ( \theta \mid { \mathcal { D } } ) d \theta\tag{12}
$$

Assigning rigorous confidence intervals to topological attributions is essential if deep graph representations are to safely inform targeted stereotactic neurosurgery, deep brain stimulation (DBS) trajectory planning, or mechanistic clinical trial stratification.

## 6.5 Cost-Sensitive Clinical Triage and Selective Classification Frameworks

In clinical neurology and psychiatry, forced-choice classifiers that compel an artificial binary assignment on ambiguous border-zone cases violate foundational principles of patient safety. Predictive systems must instead institutionalize selective classification mechanisms governed by formal deferral policies $\rho ( \mathbf { x } ) \in \{ 0 , 1 \}$

By integrating distribution-free conformal prediction sets, an automated routing pipeline can be constructed wherein singleton prediction sets ${ \mathcal C } ( { \bf x } ) = \{ \hat { y } \}$ trigger rapid autonomous triaging, whereas high epistemic ambiguity leading to multi-label prediction sets $( | { \mathcal { C } } ( \mathbf { x } ) | > 1 )$ or empty null sets (∅) immediately diverts the scan to expert neuroradiological evaluation. The emergence of an empty conformal set serves as an intrinsic alarm for severe out-of-distribution pathology, gross acquisition artifacts, or unexpected intracranial lesions that require human clinical oversight.

## 6.6 Scalable, Single-Pass Inference for High-Resolution Graph Topologies

Iterative sampling paradigms such as Markov Chain Monte Carlo (MCMC) and large Deep Ensembles impose prohibitive computational and memory bottlenecks when scaled to fine-grained brain parcellations, such as the Glasser-360 or Schaefer-1000 atlases, or to dense voxel-level dynamic connectomes exceeding 10<sup>5</sup> nodes.

To facilitate seamless integration into institutional Picture Archiving and Communication Sys tems (PACS), methodological innovation must prioritize single-pass and deterministic uncertainty approximations. Linearized and Kronecker-Factored Approximate Curvature (KFAC) Laplace approximations ofer scalable post-hoc parameter covariances over pre-trained GNNs without repeated training overhead. Concurrently, Evidential Graph Neural Networks place conjugate priors (such as Dirichlet distributions for discrete targets or Normal-Inverse-Gamma distributions for continuous age regressions) directly over graph readouts, yielding analytic epistemic and aleatoric bounds in a single forward evaluation. Finally, the field requires unified open-source software libraries that seamlessly interface graph calibration with standard neuroimaging toolkits (NiBabel, Nilearn, and PyTorch Geometric) to guarantee clinical reproducibility.

## Declarations

## Conflict of interest

The author declares that there are no competing financial or non-financial interests.

## Funding

The author received no financial support for the research, authorship, or publication of this article.

## Author contributions

M. Pakravan conceived the study, designed the methodology, developed the software, performed the analysis, interpreted the results, and wrote the original manuscript draft. The author reviewed and approved the final version of the manuscript.

## Data availability

The dataset used in this study is the SUDMEX CONN dataset available on OpenNeuro under accession number ds003346, version 1.1.2: https://openneuro.org/datasets/ds003346/versions/1.1.2

## References

[1] M. Alavi, A. Valiollahi, and M. Pakravan. Bibliometric analysis of research trends on graph neural networks. In Proceedings of the 20th CSI International Symposium on Artificial Intelligence and Signal Processing (AISP), pages 1–8, 2024.

[2] A. Bessadok, M. A. Mahjoub, and I. Rekik. Graph neural networks in network neuroscience. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45:5833–5848, 2023.

[3] M. Pakravan. A multi-level fuzzy explainable prototype network for sex classification across brain atlases. Iranian Journal of Fuzzy Systems, 23:165–191, 2026.

[4] M. Pakravan. Uncertainty-aware type-II fuzzy graph modeling of resting-state fMRI uncovers robust sex diferences. Journal of Neuroscience Methods, 431:110745, 2026.

[5] M. Pakravan. Chronic knee osteoarthritis network on resting state fMRI dynamic functional connectivity reveals cerebellar hub breakdown in chronic knee osteoarthritis pain. Iranian Journal of Science and Technology, Transactions of Electrical Engineering, 50:289–300, 2026.

[6] M. Gholami, M. Faramarzi, and M. Pakravan. ERGAN: An explainable residual graph attention network identifies left superior corona radiata as a key region in fMRI causal graphs of developmental prosopagnosia. Biomedical Signal Processing and Control, 126:110910, 2026.

[7] M. A. Saket and M. Pakravan. Cross-atlas identification of narrative hubs via multi-embedding graph models in fMRI data. Neuroinformatics, 24:29, 2026.

[8] M. Gholami, M. Faramarzi, N. Alipour, and M. Pakravan. Node embedding extraction for causal brain graphs in fMRI data. In Proceedings of the 29th International Computer Conference, Computer Society of Iran (CSICC), pages 1–6, 2025.

[9] R. Kumar, K. Sporn, and T. Kumar. Graph neural networks in neuroimaging: current status and biostatistical considerations for clinical deployment. Annals of Biomedical Engineering, 54:2408–2437, 2026.

[10] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.

[11] Shahriar Faghani, Mana Moassefi, Pouria Rouzrokh, Bardia Khosravi, Francis I Bafour, Michael D Ringler, and Bradley J Erickson. Quantifying uncertainty in deep learning of radiologic images. Radiology, 308(2):e222217, 2023.

[12] Benjamin Lambert, Florence Forbes, Senan Doyle, Harmonie Dehaene, and Michel Dojat. Trustworthy clinical ai solutions: a unified review of uncertainty quantification in deep learning models for medical image analysis. Artificial Intelligence in Medicine, 150:102830, 2024.

[13] Hans Hao-Hsun Hsu, Yuesong Shen, Christian Tomani, and Daniel Cremers. What makes graph neural networks miscalibrated? Advances in Neural Information Processing Systems, 35:13775–13786, 2022.

[14] Xiaotian Xie, Biao Luo, Yingjie Li, Chunhua Yang, and Weihua Gui. Exploring heterophily in calibration of graph neural networks. Neurocomputing, 604:128294, 2024.

[15] E. H¨ullermeier and W. Waegeman. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning, 110:457–506, 2021.

[16] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? volume 30, 2017.

[17] Ling Huang, Su Ruan, Yucheng Xing, and Mengling Feng. A review of uncertainty quantification in medical image analysis: Probabilistic and non-probabilistic methods. Medical Image Analysis, 97:103223, 2024.

[18] Hamed Mohammadi and Waldemar Karwowski. Graph neural networks in brain connectivity studies: Methods, challenges, and future directions. Brain Sciences, 15(1):17, 2024.

[19] Silvia Seoni, Vicnesh Jahmunah, Massimo Salvi, Prabal Datta Barua, Filippo Molinari, and U Rajendra Acharya. Application of uncertainty quantification to artificial intelligence in healthcare: A review of last decade (2013–2023). Computers in Biology and Medicine, 165:107441, 2023.

[20] Fernando Vega Lara, Lisa D Koopmans, Christian Roest, Baris Turkbey, Derya Yakar, and Thomas C Kwee. Uncertainty quantification for artificial intelligence in medical imaging: what every radiologist needs to know. Abdominal Radiology, pages 1–20, 2026.

[21] M. Pakravan. Fuzzycal: A fuzzy-logic enhanced causal attention GNN for robust cocaine use disorder classification. Iranian Journal of Fuzzy Systems, 22:167–182, 2025.

[22] Fangxin Wang, Yuqing Liu, Kay Liu, Yibo Wang, Sourav Medya, and Philip S Yu. Uncertainty in graph neural networks: A survey. arXiv preprint arXiv:2403.07185, 2024.

[23] Thomas Buddenkotte, Lorena Escudero Sanchez, Mireia Crispin-Ortuzar, Ramona Woitek, Cathal McCague, James D Brenton, Ozan Oktem, Evis Sala, and Leonardo Rundo. Cali-<sup>¨</sup> brating ensembles for scalable uncertainty quantification in deep learning-based medical image segmentation. Computers in Biology and Medicine, 163:107096, 2023.

[24] Moritz Fuchs, Camila Gonzalez, and Anirban Mukhopadhyay. Practical uncertainty quantification for brain tumor segmentation. In Medical Imaging with Deep Learning, 2021.

[25] Nataliia Molchanova, Pedro M Gordaliza, Alessandro Cagol, Mario Ocampo-Pineda, Po-Jui Lu, Matthias Weigel, Xinjie Chen, Erin S Beck, Haris Tsagkas, Daniel Reich, et al. Explaining uncertainty in multiple sclerosis lesion segmentation beyond prediction errors. ArXiv, pages arXiv–2504, 2025.

[26] A. N. Angelopoulos and S. Bates. Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning, 16:445–532, 2023.

[27] Kexin Huang, Ying Jin, Emmanuel Candes, and Jure Leskovec. Uncertainty quantification over graph with conformalized graph neural networks. volume 36, pages 26699–26721, 2023.

[28] Vineeth N Balasubramanian, Shen-Shyang Ho, and Vladimir Vovk. Conformal prediction for reliable machine learning: theory, adaptations and applications. 2014.

[29] Xixun Lin, Yanan Cao, Nan Sun, Lixin Zou, Chuan Zhou, Peng Zhang, Shuai Zhang, Ge Zhang, and Jia Wu. Conformal graph-level out-of-distribution detection with adaptive data augmentation. page 4755–4765. Association for Computing Machinery, 2025.

[30] Alexander Kurz, Katja Hauser, Hendrik Alexander Mehrtens, Eva Krieghof-Henning, Achim Hekler, Jakob Nikolas Kather, Stefan Fr¨ohling, Christof Von Kalle, and Titus Josef Brinker. Uncertainty estimation in medical image classification: systematic review. JMIR Medical Informatics, 10(8):e36427, 2022.

[31] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In International conference on machine learning, pages 1613–1622. PMLR, 2015.

[32] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016.

[33] Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace redux-efortless bayesian deep learning. volume 34, pages 20089– 20103, 2021.

[34] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. volume 30, 2017.

[35] W. J. Maddox, T. Garipov, P. Izmailov, D. Vetrov, and A. G. Wilson. A simple baseline for Bayesian uncertainty in deep learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

[36] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, pages 3183–3193, 2018.

[37] Oscar Castillo, Patricia Melin, Janusz Kacprzyk, and Witold Pedrycz. Type-2 fuzzy logic: theory and applications. 2007.

[38] Diego Angeles-Valdez, Jalil Rasgado-Toledo, Victor Issa-Garcia, Thania Balducci, Viviana Villica˜na, Alely Valencia, Jorge Julio Gonzalez-Olvera, Ernesto Reyes-Zamorano, and Eduardo A Garza-Villarreal. The mexican magnetic resonance imaging dataset of patients with cocaine use disorder: Sudmex conn. Scientific data, 9(1):133, 2022.

[39] R. W. Cox. AFNI: Software for analysis and visualization of functional magnetic resonance neuroimages. Computers and Biomedical Research, 29:162–173, 1996.

[40] Nikos Makris, Jill M Goldstein, David Kennedy, Steven M Hodge, Verne S Caviness, Stephen V Faraone, Ming T Tsuang, and Larry J Seidman. Decreased volume of left and total anterior insular lobule in schizophrenia. Schizophrenia research, 83(2-3):155–171, 2006.

[41] A. Mehrtash et al. Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. IEEE Transactions on Medical Imaging, 39:3868–3878, 2020.

[42] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. volume 30, 2017.

[43] Yukun Ding, Jinglan Liu, Jinjun Xiong, and Yiyu Shi. Revisiting the evaluation of uncertainty estimation and its application to explore model complexity-uncertainty trade-of. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 22–31. IEEE, 2020.