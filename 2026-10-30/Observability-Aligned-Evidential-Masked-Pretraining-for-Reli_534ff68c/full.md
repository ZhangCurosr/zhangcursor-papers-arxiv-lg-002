# Observability-Aligned Evidential Masked Pretraining for Reliable VRU Motion Prediction

Shang Gao<sup>∗1</sup>, Franz Götz-Hahn<sup>†1</sup>, Konrad Doll<sup>‡2</sup>, and Bernhard Sick<sup>§1</sup>

<sup>1</sup>Intelligent Embedded Systems, University of Kassel, Germany,

{sgao,bsick,franz.goetz-hahn}@uni-kassel.de

<sup>2</sup>Aschafenburg University of Applied Sciences, Germany, konrad.doll@th-ab.de

## Abstract

Vulnerable road user (VRU) prediction must operate under partial observability and distribution shift, so reliable uncertainty matters alongside accuracy. Masked pretraining is attractive in this setting because it naturally exposes the model to missing content during self-supervision, but existing time-series masked autoencoding methods are typically deterministic and do not constrain how confidence should change with observability. We introduce eCoMAE, an evidential masked pretraining method for multivariate time series that explicitly aligns predictive evidence with what is observed. eCoMAE combines block-wise time-step masking with a Normal–Inverse–Gamma decoder and three observability-aware mechanisms: uncertainty-weighted mean consistency on positions visible in both a masked view and a reference pass, evidence suppression on masked positions, and representation consistency across stochastic masks. On IMPTC benchmark for VRU intention prediction under leave-one-environment-out weather shift, eCoMAE improves the strongest baseline from 0.7971 to 0.8068 in OOD accuracy and from 0.0718 to 0.0648 in OOD AURC. Additional evaluation on pedestrian trajectory forecasting across IMPTC, Waymo, nuScenes, and inD further yields the best fine-tuning MSE on all four datasets. These results show that observability-aligned evidential pretraining improves reliability under shift and remains efective beyond the primary benchmark.

Keywords: self-supervised learning; masked autoencoding; uncertainty quantification; evidential learning; vulnerable road user intention prediction

## 1 Introduction

Vulnerable road user (VRU) motion prediction is a core capability for intelligent transportation systems and safety-critical automation. It includes both discrete intention recognition and continuous trajectory forecasting, and both settings operate under partial observability, changing sensing conditions, and distribution shift. In such environments, reliable decision making requires not only strong predictive performance but also calibrated uncertainty and confidence rankings that remain meaningful under shift.

Masked pretraining is a natural fit for VRU prediction because it exposes the model to missing content during self-supervision. However, existing time-series masked autoencoding methods mainly optimize deterministic reconstruction and leave uncertainty to downstream add-ons such as ensembles [1], Bayesian approximations [2]. As a result, pretraining does not directly constrain how confidence should respond to partial observability.

![](images/a534362a7cc0d677bb83abd875eb23eb7b03c7b2a911b74ed2a8fce8a495a39b.jpg)  
Figure 1: eCoMAE workflow. For each input sequence, K stochastic masked views and one unmasked reference view are processed by a shared encoder–decoder with an NIG head. Training combines masked fitting, visible-token mean consistency against the reference, masked-token evidence suppression, and cross-view representation consistency.

We instead ask a more specific question: can masked pretraining shape both representation quality and uncertainty formation so that confidence is high on shared visible content and restrained on masked content? eCoMAE answers this by combining an evidential decoder with observability-aligned regularisation. The key novelty is not merely adding evidential outputs to masked autoencoding, but coupling evidence formation to what is actually observed under each mask.

This work proposes eCoMAE, an evidential consistency masked autoencoder for timeseries pretraining that learns representations and confidence signals within the self-supervised objective. As shown in Fig. 1, eCoMAE adopts block-wise time-step token masking that masks all channels at selected time indices, reducing leakage across features and preserving temporal structure. A Transformer encoder–decoder architecture is equipped with a Normal-Inverse-Gamma (NIG) evidential head [3], producing distributional predictions rather than point reconstructions. Crucially, eCoMAE aligns confidence with observability through evidential consistency. Predictions on positions that are visible in both a masked view and a reference pass are encouraged to match the reference, while masked positions are discouraged from showing high evidence through an explicit evidence cap. A representation consistency term further stabilises embeddings across diferent mask realisations. The method supports a configurable compute–performance trade-of: the number of stochastic mask views and the subset used for evidential alignment are configurable, and an optional visible-token encoder reduces pretraining cost at high mask rates. Padding-aware masking and losses also support variable-length sequences with missingness.

eCoMAE is evaluated primarily on four-class VRU intention classification using the public IMPTC [4] benchmark, under both in-distribution (ID) testing and out-of-distribution (OOD) evaluation induced by a leave-one-environment-out weather split. The evaluation extends beyond predictive performance to include calibration, selective-prediction reliability, extensive sensitivity studies, and targeted ablations of the proposed design. Broader applicability is further assessed by evaluating the method for pedestrian trajectory forecasting on a unified benchmark built from IMPTC, Waymo [5], nuScenes [6], and inD [7], following Hetzel et al. [8].

The main contributions are as follows:

• eCoMAE, an observability-aligned evidential masked pretraining method for multivariate time series, integrates masked modelling with predictive uncertainty learning.

• Observability-aligned confidence learning is achieved through consistency on overlapvisible positions, evidence suppression on masked positions, and representation consistency across stochastic masks.

• A comprehensive experimental evaluation is conducted, centered on IMPTC intention prediction under environment shift, with reliability-focused analysis, sensitivity studies, and targeted ablations, and further supported by additional evaluation on pedestrian trajectory forecasting benchmarks.

## 2 Related Work

Self-supervised time series representation learning. Existing time series self-supervised methods are dominated by contrastive and reconstruction-based objectives [9, 10, 11]. These methods learn transferable features, but their pretraining objectives are typically deterministic and do not shape predictive uncertainty. Reliability under shift is therefore usually addressed only after pretraining.

Uncertainty estimation and evidential learning. Uncertainty in deep learning is often estimated downstream using deep ensembles, Bayesian approximations, or post-hoc calibration procedures [1, 2, 12]. Evidential deep learning provides an alternative by predicting distribution parameters directly, enabling uncertainty to be learned end-to-end from the training objective [13, 3]. Related to uncertainty-aware reconstruction, prior work has explored an economical evidential autoencoder (EAE) design that produces distributional outputs from an autoencoding objective with reduced overhead [14]. However, this line of work is primarily developed for uncertaintyaware autoencoding in a specific downstream setting, rather than for learning broadly transferable time-series representations with predictive uncertainty as part of self-supervised pretraining. Moreover, directly optimizing distributional losses does not by itself guarantee that confidence remains aligned with partial observability under masking.

Consistency across masked views. Consistency across views or perturbations is a central principle in self-supervised learning, including representation alignment and self-distillation approaches [15, 16]. In masked pretraining, this issue becomes particularly relevant because diferent stochastic masks can produce inconsistent predictions for the same underlying content. This problem has been studied in masked image modelling, where EMAE introduces selfconsistency regularisation across overlapping masked regions [17]. More broadly, related work has explored uncertainty-aware consistency regularisation, where uncertainty is used to modulate the strength of alignment according to prediction reliability [18]. Motivated by these ideas, eCoMAE studies consistency across masked views for multivariate time series and incorporates evidential uncertainty into this consistency regularisation.

## 3 Approach

This section defines eCoMAE, a masked pretraining objective for multivariate time series that jointly models reconstruction and predictive evidence under partial observability.

## 3.1 Overview

Given a multivariate time-series window $\mathbf { X } \in \mathbb { R } ^ { L \times F }$ , eCoMAE pretrains a Transformer encoder $f _ { \theta }$ through masked modelling at the time-step token level. A masked set of time indices $\mathcal { M } \subset \{ 1 , \ldots , L \}$ is sampled on valid (non-padding) steps to form $\tilde { \mathbf { X } } = g ( \mathbf { X } ; \mathcal { M } )$ . From a masked forward pass, the model predicts both a reconstruction and a NIG predictive distribution. Beyond fitting masked targets, eCoMAE aligns evidential outputs with observability induced by masking: (i) predictions on visible steps should be invariant to masking elsewhere; (ii) masked steps should not receive unwarranted evidence. In addition, eCoMAE stabilises learned representations across diferent mask realisations and implements an eficient visible-token encoder to reduce the computational cost at high mask rates. eCoMAE supports K stochastic mask views per sample, where K is configurable to trade computation for robustness. For each view k, a block-wise time-step mask $\mathcal { M } ^ { ( k ) }$ yields NIG outputs. Masked fitting losses are averaged across views. Representation consistency is computed over all masked-view pairs, whereas evidential consistency is computed only on a selectable subset of masked views against the reference pass.

## 3.2 Mask Invariant Predictions on Visible Tokens

Let $\pi \in \{ 0 , 1 \} ^ { L }$ be a padding mask where $\pi _ { t } = 1$ denotes padding at time step t, and let $v _ { t } = \mathbb { I } [ \pi _ { t } = 0 ]$ indicate validity. A masked index set M is sampled only from valid steps; a block length b is used to encourage blocky masks. The masking operator replaces entire time-step tokens,

$$
\begin{array} { r } { \tilde { \mathbf { x } } _ { t } = \left\{ \begin{array} { l l } { \mathbf { m } , } & { t \in \mathcal { M } , } \\ { \mathbf { x } _ { t } , } & { t \not \in \mathcal { M } , } \end{array} \right. \quad \tilde { \mathbf { X } } = g ( \mathbf { X } ; \mathcal { M } ) , } \end{array}\tag{1}
$$

where m $\in \mathbb { R } ^ { F }$ is a learnable mask embedding. This time-step token masking avoids feature leakage by masking all F channels at selected steps.

The masked forward pass uses a visible-valid set

$$
\mathcal { V } = \{ t : \ v _ { t } = 1 , \ t \not \in \mathcal { M } \} , \qquad L _ { \mathrm { v i s } } = | \mathcal { V } | ,\tag{2}
$$

and encodes only the packed visible sequence,

$$
\begin{array} { r } { \mathbf { Z } _ { \mathrm { v i s } } ^ { ( m ) } = f _ { \theta } ( \tilde { \mathbf { X } } _ { \mathcal { V } } ) \in \mathbb { R } ^ { L _ { \mathrm { v i s } } \times D } . } \end{array}\tag{3}
$$

A full-length decoder reconstructs all positions by placing learned mask tokens at every step and injecting the encoded visible tokens at their original indices. The decoder outputs

$$
( \hat { \mathbf { X } } ^ { ( m ) } , \pmb { \mu } ^ { ( m ) } , \pmb { \nu } ^ { ( m ) } , \pmb { \alpha } ^ { ( m ) } , \pmb { \beta } ^ { ( m ) } ) = d _ { \phi } ( \mathbf { Z } _ { \mathrm { v i s } } ^ { ( m ) } ) ,\tag{4}
$$

where each output lies in R $L \times F$ . Positivity is enforced via standard transforms $\left( \mathrm { e . g . } \right.$ , softplus) and $\alpha > 1$ is ensured to obtain a proper NIG predictive distribution.

Define masked-valid and visible-valid index sets

$$
\Omega _ { m } = \{ ( t , f ) \mid t \in \mathcal { M } , \ v _ { t } = 1 , \ f \in \{ 1 , \ldots , F \} \} ,\tag{5}
$$

$$
\Omega _ { u } = \{ ( t , f ) \mid t \not \in \mathcal { M } , \ v _ { t } = 1 , \ f \in \{ 1 , \ldots , F \} \} ,\tag{6}
$$

and the averaging operator

$$
\langle a \rangle _ { \Omega } = \frac { 1 } { | \Omega | } \sum _ { ( t , f ) \in \Omega } a _ { t , f } .\tag{7}
$$

Masked modelling fits only $\Omega _ { m }$ to prevent leakage:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { e v i } } = \Big \langle _ { + \lambda _ { \mathrm { r e g } } \ell _ { \mathrm { r e g } } ( x _ { t , f } , \mu _ { t , f } ^ { ( m ) } , \nu _ { t , f } ^ { ( m ) } , \nu _ { t , f } ^ { ( m ) } , \alpha _ { t , f } ^ { ( m ) } ) } ^ { \ell _ { \mathrm { N I G } } ( m ) } \big \rangle _ { \Omega _ { m } } . } \end{array}\tag{8}
$$

To align predictions with observability, an unmasked reference pass is computed on valid steps,

$$
\begin{array} { r } { ( \hat { \mathbf { X } } ^ { ( r ) } , \pmb { \mu } ^ { ( r ) } , \pmb { \nu } ^ { ( r ) } , \pmb { \alpha } ^ { ( r ) } , \pmb { \beta } ^ { ( r ) } ) = d _ { \phi } \big ( f _ { \theta } ( \mathbf { X } _ { \{ t : v _ { t } = 1 \} } ) \big ) . } \end{array}\tag{9}
$$

EvMu enforces that visible-valid predictions remain invariant to masking elsewhere:

$$
\mathcal { L } _ { \mu } = \left. \omega _ { t , f } \ \phi ( \mu _ { t , f } ^ { ( m ) } - \mu _ { t , f } ^ { ( r ) } ) \right. _ { \Omega _ { u } } , \qquad \omega _ { t , f } = \frac { 1 } { \sigma _ { t , f } ^ { 2 ( r ) } + \tau } ,\tag{10}
$$

where $\phi ( \cdot )$ is robust Huber-style penalty, $\tau > 0$ is a stabiliser, and

$$
\sigma _ { t , f } ^ { 2 ( r ) } = \frac { \beta _ { t , f } ^ { ( r ) } ( 1 + \nu _ { t , f } ^ { ( r ) } ) } { \nu _ { t , f } ^ { ( r ) } ( \alpha _ { t , f } ^ { ( r ) } - 1 ) } , \quad \quad \alpha _ { t , f } ^ { ( r ) } > 1 ,\tag{11}
$$

is the predictive variance implied by the reference NIG parameters. This quantity corresponds to the total predictive uncertainty and can be decomposed into an aleatoric term $\beta _ { t , f } ^ { ( r ) } / ( \alpha _ { t , f } ^ { ( r ) } - 1 )$ and an epistemic term $\beta _ { t , f } ^ { ( r ) } / ( \nu _ { t , f } ^ { ( r ) } ( \alpha _ { t , f } ^ { ( r ) } - 1 ) )$ . The weighting therefore down-weights entries that are intrinsically uncertain under the reference pass, focusing the invariance constraint on stable, well-observed predictions.

## 3.3 Evidence Suppression on Masked Tokens

Masked supervision alone does not prevent the model from producing overly confident evidence at unobserved locations. EvAlpha explicitly discourages excessive evidence on masked-valid entries by penalising the evidential strength

$$
\Phi _ { t , f } ^ { ( m ) } = 2 \nu _ { t , f } ^ { ( m ) } + \alpha _ { t , f } ^ { ( m ) }\tag{12}
$$

above a cap $\bar { \Phi } ;$ :

$$
\mathcal { L } _ { \alpha } = \left. [ \Phi _ { t , f } ^ { ( m ) } - \bar { \Phi } ] _ { + } \right. _ { \Omega _ { m } } ,\tag{13}
$$

where $[ x ] _ { + } = \operatorname* { m a x } ( 0 , x )$ . In our implementation, $\bar { \Phi }$ is a fixed hyperparameter that bounds evidential strength on masked positions throughout training.

In practice, the consistency terms need not use all masked views: the EvMu and EvAlpha penalties are computed on a configurable subset of mask views and averaged across the selected views, enabling cost-aware training when K is large.

## 3.4 Mask-Consistent Representations and Visible-Token Eficiency

To stabilise representations across missingness patterns, K mask passes are computed. Representation consistency is applied to all view pairs $( i , j )$ by pooling only overlap-visible positions and averaging the cosine penalty over pairs.

$$
\mathcal { L } _ { \mathrm { r e p } } = \frac { 2 } { K ( K - 1 ) } \sum _ { 1 \le i < j \le K } \Big ( 1 - \cos \big ( \mathbf { z } ^ { m _ { i } } , \mathbf { z } ^ { m _ { j } } \big ) \Big ) , \qquad \mathbf { z } ^ { m _ { i } } = h _ { w } \big ( \mathrm { p o o l } \big ( \mathbf { Z } _ { v i s } ^ { m _ { i } } \big ) \big ) ,\tag{14}
$$

We additionally include a standard masked reconstruction term on the predictive mean,

$$
\mathcal { L } _ { \mathrm { r e c } } = \left. \left( \hat { x } _ { t , f } ^ { ( m ) } - x _ { t , f } \right) ^ { 2 } \right. _ { \Omega _ { m } } ,
$$

averaged across masked views.

The overall pretraining objective combines masked fitting with observability- and maskconsistency regularisation:

$$
{ \mathcal { L } } _ { \mathrm { s s l } } = \underbrace { \lambda _ { \mathrm { r e c } } { \mathcal { L } } _ { \mathrm { r e c } } + \lambda _ { \mathrm { e v i } } { \mathcal { L } } _ { \mathrm { e v i } } } _ { { \mathcal { L } } _ { \mathrm { f i t } } } + \underbrace { \lambda _ { \mu } { \mathcal { L } } _ { \mu } + \lambda _ { \alpha } { \mathcal { L } } _ { \alpha } + \lambda _ { \mathrm { r e p } } { \mathcal { L } } _ { \mathrm { r e p } } } _ { { \mathcal { L } } _ { \mathrm { c o n s - r e g } } } .\tag{15}
$$

Finally, the visible-token encoder provides an eficiency benefit. Self-attention cost in the encoder reduces from $\mathcal { O } ( L ^ { 2 } )$ to approximately $\mathcal { O } ( L _ { \mathrm { v i s } } ^ { 2 } )$ , while the decoder still reconstructs the full length-L sequence and losses are computed on $\Omega _ { m }$ and $\Omega _ { u }$ . At high mask rates, where $L _ { \mathrm { v i s } } \ll L$ , this yields substantial compute savings without changing the supervision signal or the observability constraints.

![](images/8678ca474c38bcc7a260807f62396cb7d99660bc4f0a00d27e1c69aca5295bca.jpg)  
Figure 2: Mask-rate sensitivity of finetuning accuracy

## 4 Experiments

The experiments are designed to answer three questions: (i) whether eCoMAE improves predictive performance and reliability on the primary IMPTC intention benchmark under environment shift, (ii) which design choices are most responsible for the gains, and (iii) whether the method remains efective beyond the primary setting on diferent downstream tasks and datasets.

## 4.1 Primary Benchmark: IMPTC Intention Prediction

The primary evaluation is conducted on a trajectory-window benchmark derived from the public IMPTC VRU dataset. Each sample is represented as a fixed-length multivariate timeseries window $X \in \mathbb { R } ^ { L \times F }$ . The downstream task is four-class intention prediction, with labels corresponding to straight, left, right, and stop. To mitigate class imbalance, supervised objectives use class-weighted cross-entropy.

Evaluation is performed under both ID and OOD settings. The dataset provides weatherrelated metadata, which is used to partition samples into discrete environment groups. OOD generalisation is assessed with a leave-one-environment-out protocol, where one environment group is held out for testing and the remaining groups are used for training. ID evaluation uses the standard test split, which follows the same environment mixture as the training data. This setup makes it possible to assess not only predictive performance in the nominal setting, but also robustness and reliability under environment shift.

## 4.2 Training Protocol and Evaluation Metrics

Unless stated otherwise, evaluation follows a three-stage protocol consisting of self-supervised pretraining, linear probing, and end-to-end finetuning. The main analysis focuses on finetuned models, while probing results are reported to assess representation quality before downstream adaptation.

Performance is evaluated with both predictive and uncertainty-aware metrics. Classification quality is measured by accuracy. Calibration is quantified by expected calibration error (ECE) [12]. Under distribution shift, selective-prediction reliability is assessed by the area under the risk– coverage curve (AURC), where lower values indicate better abstention behaviour [19]. Alignment between uncertainty and errors is further measured by uncertainty–error correlation (UE-corr), computed as the Pearson correlation between predictive entropy and the corresponding 0–1 error indicator, where higher values indicate better error awareness [20, 21].

The analysis on IMPTC is organized in two stages. First, a set of model studies examines the role of mask rate, model capacity, evidential sensitivity, and individual objective components. Second, the resulting configuration is compared against supervised, reconstruction-based pretraining, and contrastive self-supervised baselines under both ID and OOD evaluation.

<table><tr><td rowspan="2">Capacity</td><td colspan="2">ID finetune</td><td colspan="2">OOD finetune</td><td>Cost</td></tr><tr><td> $\operatorname { A c c } _ { f } \uparrow$ </td><td> $\operatorname { E C E } _ { f } \downarrow$ </td><td> $\operatorname { A c c } _ { f } \uparrow$ </td><td> $\operatorname { E C E } _ { f } \downarrow$ </td><td>min↓</td></tr><tr><td colspan="6">Capacity scaling at MR=0.80</td></tr><tr><td>Small</td><td>0.8040</td><td>0.0384</td><td>0.7824</td><td>0.0642</td><td>40.5</td></tr><tr><td>Medium</td><td>0.8247</td><td>0.0184</td><td>0.7941</td><td>0.0781</td><td>41.4</td></tr><tr><td>Large</td><td>0.8384</td><td>0.0129</td><td>0.7971</td><td>0.0698</td><td>48.7</td></tr></table>

Table 1: Capacity sweep results and backbone definitions. Small, Medium, and Large set $( d _ { m o d e l } , d _ { f f } , E , D , H )$ to (32, 64, 2, 1, 4), (64, 128, 2, 1, 8), and (128, 256, 3, 2, 8), respectively. Here $d _ { m o d e l }$ is the Transformer hidden size, $d _ { f f }$ is the width of the feed-forward network inside each Transformer block, E is the number of encoder layers, D is the number of decoder layers, and H is the number of attention heads.

## 4.3 Model Analysis on IMPTC

This subsection analyzes the main design choices of eCoMAE on the primary IMPTC benchmark. The study first examines mask-rate selection, then evaluates model capacity, and finally investigates evidential sensitivity and component ablations to isolate the contributions of the proposed objectives.

Mask-rate selection. Figure 2 summarises the mask-rate study. Finetune accuracy improves steadily as the mask rate increases from low to moderate values and reaches its best point around 0.80, after which the gain saturates. This supports the intended trade-of of masked pretraining: stronger masking increases the dificulty of reconstruction and encourages more informative representations, but overly aggressive masking ofers little further benefit. Based on the joint consideration of accuracy, calibration, and computational cost, the default mask rate is set to 0.80.

Capacity analysis. Table 1 reports the capacity sweep at the selected mask rate. Increasing capacity improves finetune accuracy in both ID and OOD settings, with the large model achieving the best ID accuracy, the best ID calibration, and the best OOD accuracy. The small model is cheapest and attains the best OOD ECE, but at a clear accuracy cost. The medium model ofers a similar runtime to the small model while substantially narrowing the performance gap. Since the large model provides the strongest overall predictive performance with only a moderate increase in cost, it is used as the default setting for the subsequent ablation studies unless noted otherwise.

Evidential sensitivity and component ablations. This study analyses how the evidential objective and its components afect robustness and uncertainty under environment shift. The 18 variants are grouped into six categories: (i) a core comparison between a plain MAE objective and its evidential counterpart, (ii) a sweep over mask-wise representation consistency controlled by $\lambda _ { \mathrm { { r e p } } } .$ , (iii) ablations of the mean and evidence consistency terms with NIG and REP retained, (iv) sensitivity to the evidential weight $\lambda _ { \mathrm { e v i } } .$ , (v) sensitivity to the evidential regularizer $\lambda _ { \mathrm { r e g } }$ and the evidence cap $\bar { \Phi } ,$ and (vi) sanity checks that enable representation consistency without the evidential objective. The settings follow the training objective in Sec. 3. Here, EVI denotes the NIG negative log-likelihood together with evidential regularisation, REP denotes mask-wise representation consistency, $\lambda _ { \mathrm { e v i } }$ weights the evidential term, $\lambda _ { \mathrm { r e g } }$ penalises excessive evidence when prediction error is large, $\bar { \Phi }$ caps the evidence-related parameter, and $\lambda _ { \mu } , \lambda _ { \alpha } ,$ and $\lambda _ { \mathrm { { r e p } } }$ control visible-token mean consistency, masked-token evidence suppression, and representation consistency across mask realisations, respectively. In the component ablations, NIG and REP are kept enabled, so “only” refers exclusively to which of the $\mu$ and α consistency terms is active.

<table><tr><td rowspan="2">Setting</td><td colspan="4">ID</td><td colspan="4">OOD</td></tr><tr><td> $\operatorname { A c c } _ { p }$ </td><td> $\mathrm { E C E } _ { p }$ </td><td> $\operatorname { A c c } _ { f }$ </td><td> $\mathrm { E C E } _ { f }$ </td><td> $\operatorname { A c c } _ { p }$ </td><td> $\mathrm { E C E } _ { p }$ </td><td> $\operatorname { A c c } _ { f }$ </td><td> $\mathrm { E C E } _ { f }$ </td></tr><tr><td colspan="9">Evidential objective (REP off)</td></tr><tr><td>MAE (no EVI/REP)</td><td>0.7987</td><td>0.0436</td><td>0.8379</td><td>0.0226</td><td>0.7794</td><td>0.0721</td><td>0.8000</td><td>0.0799</td></tr><tr><td>EVI (no REP)</td><td>0.7993</td><td>0.0434</td><td>0.8387</td><td>0.0124</td><td>0.7765</td><td>0.0595</td><td>0.8000</td><td>0.0658</td></tr><tr><td colspan="9">Representation consistency (EVI on)</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 0 5$ </td><td>0.7998</td><td>0.0436</td><td>0.8394</td><td>0.0148</td><td>0.7765</td><td>0.0557</td><td>0.7971</td><td>0.0660</td></tr><tr><td> $\mathrm { e C o M A E } \ ( \lambda _ { \mathrm { r e p } } = 0 . 1 0 )$ </td><td>0.7996</td><td>0.0436</td><td>0.8385</td><td>0.0111</td><td>0.7765</td><td>0.0563</td><td>0.8029</td><td>0.0681</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 2 0$ </td><td>0.7995</td><td>0.0442</td><td>0.8387</td><td>0.0137</td><td>0.7765</td><td>0.0570</td><td>0.8029</td><td>0.0698</td></tr><tr><td colspan="9">Evidential consistency (EVI + REP on)</td></tr><tr><td>No µ/α consistency</td><td>0.7999</td><td>0.0375</td><td>0.8380</td><td>0.0130</td><td>0.7765</td><td>0.0566</td><td>0.8029</td><td>0.0738</td></tr><tr><td>µ consistency only</td><td>0.7998</td><td>0.0425</td><td>0.8386</td><td>0.0127</td><td>0.7765</td><td>0.0555</td><td>0.8029</td><td>0.0728</td></tr><tr><td>α consistency only</td><td>0.7994</td><td>0.0425</td><td>0.8387</td><td>0.0123</td><td>0.7765</td><td>0.0555</td><td>0.8000</td><td>0.0631</td></tr><tr><td>2× µ/α weights</td><td>0.7994</td><td>0.0429</td><td>0.8397</td><td>0.0137</td><td>0.7765</td><td>0.0582</td><td>0.8000</td><td>0.0658</td></tr><tr><td colspan="9">Evidential loss weight  $\left( \lambda _ { \mathrm { e v i } } \right)$ </td></tr><tr><td> $\lambda _ { \mathrm { e v i } } = 0 . 1$ </td><td>0.8004</td><td>0.0430</td><td>0.8383</td><td>0.0128</td><td>0.7765</td><td>0.0576</td><td>0.7971</td><td>0.0620</td></tr><tr><td> $\lambda _ { \mathrm { e v i } } = 5$ </td><td>0.7974</td><td>0.0332</td><td>0.8397</td><td>0.0118</td><td>0.7706</td><td>0.0433</td><td>0.7971</td><td>0.0624</td></tr><tr><td colspan="9">Regularisation and evidence cap</td></tr><tr><td> $\lambda _ { \mathrm { { r e g } } } = 0$ </td><td>0.7994</td><td>0.0453</td><td>0.8395</td><td>0.0121</td><td>0.7765</td><td>0.0542</td><td>0.8029</td><td>0.0594</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 1 0 ^ { - 3 }$ </td><td>0.7993</td><td>0.0453</td><td>0.8385</td><td>0.0142</td><td>0.7765</td><td>0.0617</td><td>0.8000</td><td>0.0584</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 0 . 1$ </td><td>0.7990</td><td>0.0386</td><td>0.8386</td><td>0.0129</td><td>0.7765</td><td>0.0575</td><td>0.7971</td><td>0.0749</td></tr><tr><td> $\bar { \Phi } = 2 . 5$  Φ = 6</td><td>0.7994</td><td>0.0425</td><td>0.8385</td><td>0.0119</td><td>0.7765</td><td>0.0582</td><td>0.8000</td><td>0.0704</td></tr><tr><td></td><td>0.7996</td><td>0.0423</td><td>0.8395</td><td>0.0122</td><td>0.7765</td><td>0.0586</td><td>0.8029</td><td>0.0695</td></tr><tr><td colspan="9">REP-only controls (EVI off)</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 1 0$ </td><td>0.7995</td><td>0.0466</td><td>0.8395</td><td>0.0139</td><td>0.7765</td><td>0.0953</td><td>0.8000</td><td>0.0679</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 2 0$ </td><td>0.7995</td><td>0.0396</td><td>0.8376</td><td>0.0137</td><td>0.7765</td><td>0.0519</td><td>0.8000</td><td>0.0663</td></tr></table>

Table 2: Main results of 18 variants under ID and OOD. Subscripts $p$ and $f$ denote probing and finetuning. Higher accuracy and lower ECE are better. Best finetuning values are in bold. In the component ablations, NIG and REP remain enabled; only the $\mu / \alpha$ consistency terms vary.

Table 2 reports ID and OOD accuracy and calibration for both probing and finetuning. The full hyperparameter dictionary and additional OOD uncertainty diagnostics are provided in Appendix A (Tables 5 and 6). In the core comparison, adding EVI leaves OOD finetuning accuracy unchanged at 0.8000 but improves OOD calibration, reducing OOD finetune ECE from 0.0799 to 0.0658; the corresponding OOD probing ECE decreases from 0.0721 to 0.0595 (Table 2). This indicates that the main efect of the evidential objective is improved reliability under shift rather than improved separability.

Adding REP on top of EVI mainly changes the trade-of between OOD accuracy and uncertainty quality. Moderate REP $( \lambda _ { \mathrm { r e p } } = 0 . 1 0 \ \mathrm { o r } \ 0 . 2 0 )$ improves OOD finetuning accuracy to 0.8029, whereas $\lambda _ { \mathrm { r e p } } = 0 . 0 5$ yields an OOD probe ECE of 0.0557 (Table 2). This suggests that representation consistency is helpful, but its strength should remain moderate. The component ablations indicate complementary roles. The $\lambda _ { \mu } \mathrm { - o n l y }$ variant best preserves OOD finetune accuracy among the component-only settings, whereas the $\lambda _ { \alpha } \mathrm { - o n l y }$ variant yields the best OOD finetune ECE. Strengthening all consistency terms mainly benefits ID performance rather than OOD generalisation.

The sensitivity results reveal non-monotonic behaviour. Changing $\lambda _ { \mathrm { e v i } }$ away from the default improves some probing uncertainty metrics, but reduces OOD finetuning accuracy. For $\lambda _ { \mathrm { { r e g } } } ,$ mild settings work best: $\lambda _ { \mathrm { r e g } } = 0$ gives the strongest OOD finetuning accuracy, while $\lambda _ { \mathrm { r e g } } = 1 0 ^ { - 3 }$ gives the best OOD finetuning ECE. By contrast, changing the evidence cap produces only modest diferences in this range. The sanity checks show that REP without EVI is not suficient to improve OOD accuracy, although it still improves calibration relative to the plain MAE baseline.

NIG Predictive Variance  
Input and Reconstruction  
Evidence Quantity  
![](images/0786e339d876de4c6339542a4f2d9e16397b6bff50c70ef27b228353a613c45a.jpg)

Absolute Reconstruction Error  
![](images/e5405286747ec0bc0528e692b97df5f379e43f91c8ae292d03ca0d4c41cf38ec.jpg)

![](images/93f5dc6acadd847b3df4596daa8518fa804526284918fa0af4a8af54b1f2387d.jpg)

![](images/6cf335d947d610991170d79cc86a7dad2f971900b31ad021aa6c4b141dfd5c94.jpg)

![](images/5e871591685f3afea488b1d4dd1d423dd04dbd90095068c70bb4d7cf4492d4fb.jpg)

![](images/c1d03f3011a97a08ed5b0fff428dce8385b03d55a50e47a47c9cd9f27b84ae28.jpg)  
Figure 3: Representative pretraining diagnostic for one sample-feature pair. Shaded regions indicate masked positions. Compared with Evidential ${ \mathrm { M A E } } ,$ adding representation consistency (eCoMAE) yields lower reconstruction error on masked regions and a clearer increase of predictive variance with reconstruction error, supporting better observability-aligned uncertainty during self-supervised pretraining.

Figure 3 provides a representative diagnostic from the self-supervised pretraining stage and is consistent with the downstream trends. Relative to Evidential MAE, adding Representation Consistency yields lower reconstruction error on masked regions together with a clearer increase of predictive variance as reconstruction error grows, indicating better observability-aligned uncertainty before downstream finetuning.

A related question is how much of the multi-view design is needed in practice. Figure 4 studies the interaction between the total number of stochastic mask passes $n _ { \mathrm { m a s k } }$ and the number of passes used for evidential consistency $k _ { \mathrm { e v i } }$ . Increasing $n _ { \mathrm { m a s k } }$ generally improves finetune accuracy, especially under OOD evaluation, indicating that additional masked views provide a stronger pretraining signal. At the same time, strong results are already obtained with $k _ { \mathrm { e v i } } < n _ { \mathrm { m a s k } }$ , showing that evidential alignment does not need to be applied exhaustively to every view. This supports a controllable compute–performance trade-of in eCoMAE: more stochastic views are useful, but selective evidential consistency is often suficient to retain a favorable accuracy–calibration trade-of.

![](images/7f23f1f60e57c8ad4ba19d4b6fb241b3e8bf78df0ba517252d77fe50b5b839d0.jpg)

![](images/a64d313303c2b50f1b40c9c7afe6f220e06a74a543d5b746ee73ff32a99222e2.jpg)

Figure 4: Multi-view scaling design of eCoMAE under diferent numbers of mask views $( n _ { \mathrm { m a s k } } )$ and evidential-consistency views $\left( k _ { \mathrm { e v i } } \right)$ . Cell text reports finetune accuracy, while color indicates ECE, for both ID and OOD evaluation. Hatched cells denote invalid settings with $k _ { \mathrm { e v i } } > n _ { \mathrm { m a s k } } .$
<table><tr><td>Method</td><td colspan="4">ID</td><td colspan="4">OOD</td></tr><tr><td></td><td>Accf ↑</td><td>ECEf ↓</td><td> $\mathrm { A U R C } _ { f }$ </td><td>↓UE-corrf ↑ Accf ↑</td><td></td><td>ECEf ↓</td><td> $\operatorname { A U R C } _ { f } .$ </td><td>↓ UE-corr f ↑</td></tr><tr><td>eCoMAE-S</td><td>0.8387</td><td>0.0123</td><td>0.0388</td><td>0.5380</td><td>0.8014</td><td>0.0701</td><td>0.0667</td><td>0.4464</td></tr><tr><td>eCoMAE-M</td><td>0.8389</td><td>0.0129</td><td>0.0390</td><td>0.5388</td><td>0.8068</td><td>0.0659</td><td>0.0672</td><td>0.4186</td></tr><tr><td>eCoMAE-L</td><td>0.8397</td><td>0.0144</td><td>0.0382</td><td>0.5391</td><td>0.8066</td><td>0.0678</td><td>0.0648</td><td>0.4323</td></tr><tr><td>Transformer (sup.)</td><td>0.7838</td><td>0.0516</td><td>0.0991</td><td>0.4145</td><td>0.7676</td><td>0.0825</td><td>0.1497</td><td>0.2969</td></tr><tr><td>Ti-MAE [11]</td><td>0.8379</td><td>0.0226</td><td>0.0409</td><td>0.5231</td><td>0.7971</td><td>0.0857</td><td>0.0718</td><td>0.4255</td></tr><tr><td>Gaussian Ti-MAE</td><td>0.8367</td><td>0.0205</td><td>0.0431</td><td>0.5212</td><td>0.7912</td><td>0.0735</td><td>0.0748</td><td>0.4595</td></tr><tr><td>DAE [22]</td><td>0.7848</td><td>0.0145</td><td>0.0892</td><td>0.4418</td><td>0.7706</td><td>0.0724</td><td>0.1428</td><td>0.2877</td></tr><tr><td>EAE [14]</td><td>0.7841</td><td>0.0232</td><td>0.0934</td><td>0.4322</td><td>0.7647</td><td>0.0665</td><td>0.1527</td><td>0.3029</td></tr><tr><td>SimCLR-TS [23]</td><td>0.7823</td><td>0.0255</td><td>0.0738</td><td>0.5087</td><td>0.7765</td><td>0.0815</td><td>0.1592</td><td>0.2181</td></tr><tr><td>TS2Vec [9]</td><td>0.7850</td><td>0.0374</td><td>0.0933</td><td>0.4345</td><td>0.7710</td><td>0.0869</td><td>0.1655</td><td>0.1874</td></tr></table>

Table 3: Main comparison. Higher Acc and UE-corr are better. Lower ECE and AURC are better. Best values are shown in bold. All self-supervised methods are evaluated by end-to-end fine-tuning. DAE denotes a denoising autoencoder baseline. Results are averaged over five random seeds.

## 4.4 Comparison with Baselines on IMPTC

Table 3 compares eCoMAE with supervised training, reconstruction-based pretraining, and contrastive self-supervised baselines on the primary IMPTC intention prediction benchmark. Across the eCoMAE family, ID performance is consistently strong, with the large variant achieving the best finetune ID accuracy of 0.8397 and the best ID AURC of 0.0382. Under OOD evaluation, the medium variant attains the highest finetune accuracy of 0.8068, while the large variant achieves the best AURC of 0.0648. In comparison, the strongest non-eCoMAE baseline, Ti-MAE, reaches 0.7971 OOD accuracy and 0.0718 OOD AURC. This shows that eCoMAE improves not only predictive performance under shift but also selective-prediction reliability.

Compared with deterministic masked pretraining and Gaussian or evidential reconstruction variants, eCoMAE more consistently improves OOD reliability. Gaussian Ti-MAE is constructed as a direct Gaussian counterpart to the deterministic Ti-MAE baseline by replacing the point-wise MSE reconstruction loss with a Gaussian negative log-likelihood (NLL) head that predicts a mean and variance. Similarly, EAE replaces the deterministic reconstruction objective with a Normal– Inverse–Gamma negative log-likelihood for masked reconstruction, providing distributional outputs without any observability-aware mechanisms. Together, these comparisons support the central claim that the gains arise from observability-aligned evidence shaping rather than from stronger reconstruction alone.

<table><tr><td rowspan="2">Method</td><td colspan="4">Finetune MSE↓</td></tr><tr><td>IMPTC</td><td></td><td>Waymo nuScenes</td><td>inD</td></tr><tr><td>eCoMAE-L</td><td>0.2691</td><td>0.2912</td><td>0.2514</td><td>0.3483</td></tr><tr><td>Transformer (sup.)</td><td>0.3209</td><td>0.4809</td><td>0.3279</td><td>0.4113</td></tr><tr><td>Ti-MAE</td><td>0.2997</td><td>0.3317</td><td>0.2678</td><td>0.3753</td></tr><tr><td>Gaussian Ti-MAE</td><td>0.2977</td><td>0.3328</td><td>0.2679</td><td>0.3752</td></tr><tr><td>DAE</td><td>0.2857</td><td>0.3012</td><td>0.3139</td><td>0.3756</td></tr><tr><td>EAE</td><td>0.3571</td><td>0.2984</td><td>0.3209</td><td>0.4628</td></tr><tr><td>SimCLR-TS</td><td>0.3099</td><td>0.3015</td><td>0.2773</td><td>0.3896</td></tr><tr><td>TS2Vec</td><td>0.3108</td><td>0.2970</td><td>0.2756</td><td>0.4528</td></tr></table>

Table 4: Additional evaluation on pedestrian trajectory forecasting on four datasets. Results report finetune MSE under a shared forecasting protocol, lower is better. Best values are shown in bold. Results are averaged over five random seeds.

## 4.5 Evaluation on Pedestrian Trajectory Forecasting

To assess applicability beyond intention prediction, all methods are further evaluated on pedestrian trajectory forecasting using a unified benchmark constructed from IMPTC, Waymo, nuScenes, and inD, following Hetzel et al. [8]. Under this protocol, each method is first selfsupervisedly pretrained on the target dataset and then fine-tuned for forecasting. Table 4 reports the resulting fine-tuning mean squared error on all four datasets. The large eCoMAE variant is shown because it provided the strongest overall pretraining quality in the model-selection stage.

eCoMAE achieves the lowest fine-tuning MSE on all four benchmarks, with values of 0.2691 on IMPTC, 0.2912 on Waymo, 0.2514 on nuScenes, and 0.3483 on inD. By contrast, the strongest competing baselines show mixed behaviour across datasets and do not match this consistency. These results are best interpreted as evidence that the method remains efective beyond the primary intention-prediction benchmark, rather than as a complete forecasting study.

## 5 Discussion

The results support a more specific interpretation of eCoMAE than simply “evidential masked autoencoding works better.” The main empirical pattern is that the gains concentrate on reliability under shift rather than on large jumps in separability. In the primary IMPTC evaluation, the strongest improvements appear in calibration- and selective-risk-related metrics, while accuracy remains competitive and only moderately improved. This is consistent with the intended role of the method: the pretraining objective does not merely reconstruct masked content, but also shapes how confidence should depend on observability. In this sense, the benefit of eCoMAE is best understood as better uncertainty formation during representation learning.

The ablations also clarify that the three components play diferent roles rather than contributing symmetrically. Visible-token mean consistency encourages agreement where information is actually shared across views, masked-token evidence suppression discourages overconfident predictions on unobserved content, and representation consistency stabilises the latent space across stochastic masks. The model studies suggest that these mechanisms are complementary but must be balanced carefully. Increasing mask rate or consistency strength can strengthen the self-supervised signal up to a point, but overly strong constraints reduce downstream adaptability. Likewise, the evidential sensitivity analysis suggests that mild regularisation is more efective than either no evidential pressure or aggressive evidence control. The method therefore works not by maximising all regularisers simultaneously, but by introducing enough structure to align confidence with observability while preserving flexibility for fine-tuning.

The additional evaluation results further indicate that the learned representations are not limited to a single downstream formulation. Although the deepest analysis in this paper remains centred on intention prediction under environment shift, fine-tuning on pedestrian trajectory forecasting across IMPTC, Waymo, nuScenes, and inD shows that the method remains useful beyond the original classification task. These results are best interpreted as supporting evidence for broader applicability, rather than as a complete forecasting study. The present work still has important boundaries: the strongest evidence comes from one primary benchmark, the shift analysis is concentrated on a specific leave-one-environment-out weather protocol, and the objective weights are fixed rather than adapted during training. Future work should therefore test broader types of shift, richer multimodal settings, and adaptive balancing strategies for evidential and consistency terms.

## 6 Conclusion

This paper introduced eCoMAE, an observability-aligned evidential masked pretraining method for multivariate time series. Rather than treating masked pretraining purely as a reconstruction problem, eCoMAE explicitly links masked modelling to uncertainty formation, so that the pretraining objective constrains not only what should be reconstructed, but also how confidence should behave under partial observability. To achieve this, eCoMAE combines block-wise time-step masking with a Normal–Inverse–Gamma evidential decoder and three complementary mechanisms: uncertainty-weighted consistency on overlap-visible positions, explicit evidence suppression on masked positions, and representation consistency across stochastic mask realisations.

The empirical results support this design in two ways. On IMPTC VRU intention prediction under environment shift, eCoMAE improves reliability-related performance, including calibration and selective-prediction behaviour, while maintaining competitive predictive accuracy. The ablations further indicate that these gains do not arise from stronger reconstruction or increased model capacity alone, but from coupling the evidential objective with observability-aligned consistency. Beyond the primary benchmark, additional evaluation on pedestrian trajectory forecasting across IMPTC, Waymo, nuScenes, and inD provides supporting evidence that the method remains useful beyond a single downstream task.

Overall, the results indicate that masked pretraining can be made more reliable under distribution shift when evidence formation is explicitly tied to observability. The present study is nevertheless bounded by the concentration of the deepest analysis on one primary benchmark and by the use of fixed objective weights. Future work will study broader forms of shift, richer multimodal temporal settings, and more adaptive balancing strategies for evidential and consistency terms.

## Acknowledgments

This research has been partly funded by the Hessian State Ministry for Higher Education, Research, Science and the Arts (HMWK) as part of the LOEWE program (Kapitel 15 02, Förderbuchungskreis 2995, Förderprodukt 11 – Landes-Ofensive zur Entwicklung Wissenschaftlichökonomischer Exzellenz).

Use of Generative AI. Generative AI tools were used only for language polishing and readability improvement. All technical content, experimental design, results, and claims were produced and verified by the authors.

## References

[1] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

[2] Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In Proceedings of the 33rd International Conference on Machine Learning (ICML), pages 1050–1059. PMLR, 2016.

[3] Alexander Amini, Wilko Schwarting, Ava P. Amini, and Daniela Rus. Deep evidential regression. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

[4] Manuel Hetzel, Hannes Reichert, Günther Reitberger, Erich Fuchs, Konrad Doll, and Bernhard Sick. The imptc dataset: An infrastructural multi-person trajectory and context dataset. In 2023 IEEE Intelligent Vehicles Symposium (IV), pages 1–7, 2023.

[5] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Sheng Zhao, Shuyang Cheng, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset, 2020.

[6] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[7] Julian Bock, Robert Krajewski, Tobias Moers, Stefen Runde, Lennart Vater, and Lutz Eckstein. The inD dataset: A drone dataset of naturalistic road user trajectories at German intersections. In 2020 IEEE Intelligent Vehicles Symposium (IV), pages 1929–1934, 2020.

[8] Manuel Hetzel, Hannes Reichert, Konrad Doll, and Bernhard Sick. Reliable probabilistic human trajectory prediction for autonomous applications. In Workshop on Uncertainty Quantification for Computer Vision (UNCV), ECCV, pages 135–152. Springer, 2024.

[9] Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, and Bixiong Xu. Ts2vec: Towards universal representation of time series. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 8980–8987, 2022.

[10] Pankaj Malhotra, Vishnu T. V., Lovekesh Vig, Puneet Agarwal, and Gautam Shrof. TimeNet: Pre-trained deep recurrent neural network for time series classification. In European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), pages 607–612, 2017.

[11] Zhe Li, Zhongwen Rao, Lujia Pan, Pengyun Wang, and Zenglin Xu. Ti-MAE: Self-supervised masked time series autoencoders, 2023.

[12] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (ICML), 2017.

[13] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[14] Shang Gao, Zhixin Huang, Ghassan Al-Falouji, Bernhard Sick, and Sven Tomforde. Towards cognitive situational awareness in maritime trafic using federated evidential learning. In 2025 IEEE Conference on Cognitive and Computational Aspects of Situation Management (CogSIMA), pages 9–16, 2025.

[15] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning, 2020.

[16] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language, 2022.

[17] Zhaowen Li, Yousong Zhu, Zhiyang Chen, Wei Li, Chaoyang Zhao, Rui Zhao, Ming Tang, and Jinqiao Wang. Eficient masked autoencoders with self-consistency. arXiv preprint arXiv:2302.14431, 2023.

[18] Qianyu Zhou, Zhengyang Feng, Qiqi Gu, Guangliang Cheng, Xuequan Lu, Jianping Shi, and Lizhuang Ma. Uncertainty-aware consistency regularization for cross-domain semantic segmentation. Computer Vision and Image Understanding, 221:103448, August 2022.

[19] Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems 30 (NeurIPS), 2017.

[20] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems, 2019.

[21] Alex Kendall and Yarin Gal. What uncertainties do we need in Bayesian deep learning for computer vision? In Advances in Neural Information Processing Systems, 2017.

[22] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th International Conference on Machine Learning (ICML), pages 1096–1103, 2008.

[23] Johannes Pöppelbaum, Gavneet Singh Chadha, and Andreas Schwung. Contrastive learning based self-supervised time-series analysis. Applied Soft Computing, 117:108397, 2022.

## A Detailed Ablations and Reproducibility Settings

We provide the exact experiment settings and additional OOD uncertainty metrics underlying the full eight-metric ablation table in the main text. All 18 variants follow the same experiment order as Table 2; EVI denotes the evidential NIG objective and REP denotes cross-view representation consistency.

## A.1 Evidential-objective configuration dictionary

Table 5 specifies the weights and cap values used by each ablation. Unless explicitly changed, the default configuration uses $\lambda _ { \mathrm { e v i } } = 1 , \lambda _ { \mathrm { r e g } } = 1 0 ^ { - 2 } , \bar { \Phi } = 2 , \lambda _ { \mu } = \lambda _ { \alpha } = 0 . 0 5$ , and $\lambda _ { \mathrm { r e p } } = 0 . 1 0$ In the component ablations, the NIG objective and REP remain active while the designated consistency terms are selectively deactivated.
<table><tr><td>Ablation setting</td><td> $\lambda _ { \mathrm { e v i } }$ </td><td> $\lambda _ { \mathrm { r e g } }$ </td><td>Φ</td><td> $\lambda _ { \mu }$ </td><td> $\lambda _ { \alpha }$ </td><td> $\lambda _ { \mathrm { r e p } }$ </td></tr><tr><td colspan="7">Evidential objective (REP off)</td></tr><tr><td>MAE (no EVI/REP)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>EVI (no REP)</td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.00</td></tr><tr><td colspan="7">Representation consistency (EVI on)</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 0 5$ </td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td> $\mathrm { e C o M A E } \left( \lambda _ { \mathrm { r e p } } = 0 . 1 0 \right)$ </td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 2 0$ </td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.20</td></tr><tr><td colspan="7">Evidential consistency (EVI + REP on)</td></tr><tr><td>No  $\mu / \alpha$  consistency</td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.00</td><td>0.00</td><td>0.10</td></tr><tr><td>µ consistency only</td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.05</td><td>0.00</td><td>0.10</td></tr><tr><td>α consistency only</td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.00</td><td>0.05</td><td>0.10</td></tr><tr><td>2× µ/α weights</td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.10</td><td>0.10</td><td>0.10</td></tr><tr><td colspan="7">Evidential loss weight</td></tr><tr><td> $\lambda _ { \mathrm { e v i } } = 0 . 1$ </td><td>0.10</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td> $\lambda _ { \mathrm { e v i } } = 5$ </td><td>5.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td colspan="7">Regularisation and evidence cap</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 0$ </td><td>1.0</td><td>0.0</td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 1 0 ^ { - 3 }$ </td><td>1.0</td><td> $1 0 ^ { - 3 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 0 . 1$ </td><td>1.0</td><td> $1 0 ^ { - 1 }$ </td><td>2.0</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td> $\bar { \Phi } = 2 . 5$ </td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>2.5</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td> $\bar { \Phi } = 6$ </td><td>1.0</td><td> $1 0 ^ { - 2 }$ </td><td>6</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td colspan="7">REP-only controls (EVI off)</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 1 0$ </td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.00</td><td>0.00</td><td>0.10</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 2 0$ </td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.00</td><td>0.00</td><td>0.20</td></tr></table>

Table 5: Hyperparameters for the 18 experimental variants (same order as Table 2). Evidence-cap values (Φ = 2 <sup>¯</sup> .5 and 6) follow the main-text ablation study.

## A.2 OOD selective-prediction and uncertainty diagnostics

Table 6 complements calibration results with the area under the risk–coverage curve (AURC) and uncertainty–error correlation (UE-corr), for both probing and finetuning. Lower AURC and higher UE-corr indicate stronger selective-prediction reliability and error awareness, respectively.
<table><tr><td rowspan="2">Ablation setting</td><td colspan="2">Probe</td><td colspan="2">Finetune</td></tr><tr><td> $\operatorname { A U R C } _ { p } \downarrow$ </td><td> $\operatorname { U E - c o r r } _ { p } \uparrow$ </td><td> $\operatorname { A U R C } _ { f } \downarrow$ </td><td> $\operatorname { U E - c o r r } _ { f } \ \uparrow$ </td></tr><tr><td colspan="5">Evidential objective (REP off)</td></tr><tr><td> $\mathrm { M A E \ ( n o \ E V I / R E P ) }$ </td><td>0.1807</td><td>0.2082</td><td>0.0681</td><td>0.4397</td></tr><tr><td>EVI (no REP)</td><td>0.1660</td><td>0.2615</td><td>0.0666</td><td>0.4538</td></tr><tr><td colspan="5">Representation consistency (EVI on)</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 0 5$ </td><td>0.1656</td><td>0.2616</td><td>0.0660</td><td>0.4740</td></tr><tr><td> $\mathrm { e C o M A E } \left( \lambda _ { \mathrm { r e p } } = 0 . 1 0 \right)$ </td><td>0.1658</td><td>0.2633</td><td>0.0656</td><td>0.4500</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 2 0$ </td><td>0.1651</td><td>0.2596</td><td>0.0680</td><td>0.4354</td></tr><tr><td colspan="5">Evidential consistency  $( \mathbf { E V I } + \mathbf { R E P 0 n } )$ </td></tr><tr><td>No  $\mu / \alpha$  consistency</td><td>0.1673</td><td>0.2589</td><td>0.0646</td><td>0.4450</td></tr><tr><td>consistency only  $\mu$ </td><td>0.1631</td><td>0.2585</td><td>0.0660</td><td>0.4380</td></tr><tr><td>α consistency only</td><td>0.1644</td><td>0.2628</td><td>0.0660</td><td>0.4638</td></tr><tr><td>weights  $2 \times \mu / \alpha$ </td><td>0.1631</td><td>0.2655</td><td>0.0658</td><td>0.4575</td></tr><tr><td colspan="5">Evidential loss weight</td></tr><tr><td> $\lambda _ { \mathrm { e v i } } = 0 . 1$ </td><td>0.1589</td><td>0.2743</td><td>0.0676</td><td>0.4627</td></tr><tr><td> $\lambda _ { \mathrm { e v i } } = 5$ </td><td>0.1659</td><td>0.2856</td><td>0.0674</td><td>0.4722</td></tr><tr><td colspan="5">Regularisation and evidence cap</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 0$ </td><td>0.1647</td><td>0.2625</td><td>0.0658</td><td>0.4462</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 1 0 ^ { - 3 }$ </td><td>0.1649 0.1669</td><td>0.2625 0.2598</td><td>0.0659 0.0660</td><td>0.4615</td></tr><tr><td> $\lambda _ { \mathrm { r e g } } = 0 . 1$ </td><td>0.1639</td><td></td><td></td><td>0.4712</td></tr><tr><td> $\bar { \Phi } = 2 . 5$ </td><td>0.1640</td><td>0.2660</td><td>0.0682</td><td>0.4462</td></tr><tr><td> $\bar { \Phi } = 6$   $\mathbf { R E P - o n l y }$ </td><td></td><td>0.2578</td><td>0.0661</td><td>0.4404</td></tr><tr><td colspan="5">controls (EVI off)</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 1 0$ </td><td>0.1762 0.1742</td><td>0.2141 0.2514</td><td>0.0659 0.0667</td><td>0.4664</td></tr><tr><td> $\lambda _ { \mathrm { r e p } } = 0 . 2 0$ </td><td></td><td></td><td></td><td>0.4480</td></tr></table>

Table 6: OOD uncertainty quality for the same 18 ablation variants. Lower AURC and higher UE-corr are better; best entries in each column are bold. Subscripts p and f indicate linear probing and finetuning, respectively.