# MiCLA: Mixture-to-Constituent Latent Alignment for Polyphonic Audio Representation Learning

Kohei Yamamoto<sup>1,2</sup>, Kosuke Okusa<sup>2</sup>

Abstract—Real-world audio typically contains concurrent sound events, yet standard self-supervised methods treat each clip as an isolated training unit without explicitly modeling mixture– constituent relationships. We propose Mixture-to-Constituent Latent Alignment (MiCLA), a two-stage self-supervised pretraining framework that uses synthetic log-mel mixing to align mixture representations with their constituent clips. Stage 1 structures single-clip representations via masked self-distillation and contrastive regularization. Stage 2 introduces bidirectional alignment: for a synthetic mixture, MiCLA suppresses the component aligned with one constituent and aligns the residual with the other, using projective residualization on patch tokens and sample-adaptive soft residualization on the global class token. In downstream evaluations, MiCLA achieves 43.2% mean average precision (mAP) on AS-20K and 50.3% mAP on AS-2M under fine-tuning, and achieves the highest mAP among evaluated models on both AudioSet settings under frozen Convex Gated Probing (CGP). Furthermore, a Joint Semantic Retrieval diagnostic shows that MiCLA improves simultaneous retrieval of both constituent classes over target-averaging baselines. These results indicate that directional mixture-to-constituent alignment provides an effective pre-training signal for polyphonic audio representation learning.

Index Terms—Audio representation learning, audio tagging, latent alignment, masked audio modeling, multi-label classification, polyphonic audio, self-distillation, self-supervised learning.

## I. INTRODUCTION

R EAL-WORLD audio recordings rarely contain only a single isolated sound event. Even within a short recording, multiple acoustic events, such as speech, machine sounds, music, and background noise, routinely overlap. Representations learned from such polyphonic audio should ideally capture global acoustic context while preserving constituentrelated semantic information in a form accessible to downstream models. This property is particularly critical for multilabel classification, where multiple co-occurring sound events must be predicted simultaneously.

Recent advances in self-supervised audio representation learning have demonstrated strong transfer performance by leveraging large-scale unlabeled audio data. Masked audio modeling [1]–[5], teacher–student self-distillation [6]– [9], and contrastive learning [10]–[12] learn representations from masked or augmented views. However, most of these approaches treat each audio clip as a single training unit.

Consequently, these methods do not explicitly constrain how a mixture representation should relate to the representations of the clips used to construct it. SSLAM [13], a closely related method, supervises masked prediction from a mixed input by averaging the corresponding teacher patch targets from its two pre-mixture inputs. Such target averaging does not explicitly encourage the model to retain information about both constituent clips in a downstream-accessible manner.

To address this limitation, we propose Mixture-to-Constituent Latent Alignment (MiCLA), a self-supervised pre-training framework for polyphonic audio representation learning. Rather than averaging the two targets, MiCLA uses the known correspondence between a synthetic mixture and the two pre-mixture clips used to construct it (hereafter referred to as constituent clips) to define two directional residualalignment tasks. Given two constituent clips, MiCLA convexly interpolates their log-mel spectrograms to construct a synthetic mixture input. In one direction, MiCLA suppresses the component of the student mixture representation aligned with the student representation of one constituent and aligns the resulting residual with the EMA-teacher representation of the other. It then reverses the roles of the two constituents. This bidirectional formulation is designed to retain information associated with both constituent clips rather than collapsing their supervision into a single averaged target. Importantly, constituent clips may themselves contain multiple sound events, and MiCLA requires neither class annotations nor isolated source tracks.

MiCLA adopts a two-stage pre-training strategy [13] based on a Vision Transformer (ViT) [14] architecture. In Stage 1, the ViT model is trained exclusively on constituent clips using teacher–student masked self-distillation [9] and instancelevel contrastive regularization (ICR) [15] to establish the constituent-side latent geometry. Stage 2 then continues pretraining on constituent clips and synthetic mixtures, retaining the Stage 1 objectives to preserve the established constituent-side geometry while adding bidirectional mixtureto-constituent alignment. Furthermore, MiCLA accounts for the differing representational roles of class (CLS) and patch tokens during bidirectional alignment. Because patch tokens encode localized time–frequency content, MiCLA applies hard projective residualization to eliminate components projected onto the counterpart representation. In contrast, because the CLS token aggregates global clip-level context that may be shared across constituents, MiCLA employs sample-adaptive soft residualization to adjust the suppression ratio for each sample, avoiding the forced removal of shared acoustic and semantic context. This dual-granularity design is intended to retain constituent-specific information while avoiding unnecessary removal of potentially shared global context from synthetic-mixture representations.

We evaluate MiCLA through supervised fine-tuning, frozenencoder probing, and prototype retrieval. In multi-label finetuning, MiCLA achieves 43.2% mean average precision (mAP) on AudioSet-20K (AS-20K) and 50.3% mAP on AudioSet-2M (AS-2M) [16]. Under frozen Convex Gated Probing (CGP) [17], MiCLA attains 37.8% mAP on AS-20K and 46.3% mAP on AS-2M, achieving the highest performance among the evaluated models under the common CGP protocol on both benchmarks. Additionally, in a Joint Semantic Retrieval diagnostic on synthetic ESC-50 [18] mixtures, MiCLA improves simultaneous retrieval of both constituent classes compared with target-averaging baselines. These results suggest that MiCLA provides an effective approach for polyphonic audio representation learning, while maintaining competitive general-purpose audio representation capabilities across diverse sound domains.

The primary contributions of this paper are summarized as follows:

• We propose MiCLA, a self-supervised pre-training framework that leverages known mixture–constituent correspondence for bidirectional residual alignment.

• We instantiate this idea at patch and clip levels using hard projective residualization for patch tokens and sampleadaptive soft residualization for the global class token.

• We evaluate MiCLA across 11 downstream datasets and a Joint Semantic Retrieval diagnostic, demonstrating competitive transfer on polyphonic audio classification and improved joint constituent retrieval compared with targetaveraging baselines.

## II. RELATED WORK

This section reviews three areas related to MiCLA: masked audio pre-training, contrastive representation learning, and learning from audio mixtures.

## A. Masked Self-Distillation and Masked Audio Pre-training

Masked audio pre-training divides a spectrogram into time–frequency patches and learns representations from partially observed inputs. Reconstruction-based methods, including SSAST [1], MAE-AST [2], Audio-MAE [3], and MaskSpec [4], reconstruct masked spectrogram regions using vision transformer encoders such as the Audio Spectrogram Transformer (AST) [19]. In contrast, BEATs [5] predicts discrete acoustic tokens.

Masked self-distillation instead uses a teacher network, often updated by an exponential moving average, to provide contextualized latent targets for a masked student. This paradigm includes the modality-agnostic data2vec family [6], [7] and audio methods that combine masking, augmentation, and frame- or clip-level feature matching [8], [9], [20]–[24]. In this paper, we use the term “masked self-distillation” for this teacher–student latent-prediction setting. While existing masked self-distillation methods learn primarily from masked views of the same clip, MiCLA retains this same-clip training and introduces a cross-input signal relating a synthetic mixture to its two constituent clips. For frozen evaluation, we adopt Convex Gated Probing (CGP), introduced in BAT [17], to compare encoders under a common probing recipe; BAT’s front-end and attention modifications address a design axis distinct from MiCLA.

## B. Contrastive Structuring of Audio Representations

Contrastive learning directly shapes similarities in the representation space. Audio methods have combined contrastive objectives with momentum teachers, multi-scale features, clustering, or structured time–frequency views [10]– [12]. AaSP [15], which also builds on teacher–student masked modeling, uses aliasing-aware patch representations and multimask contrastive regularization to stabilize patch tokens across masked views.

MiCLA uses instance-level contrastive regularization (ICR) to structure visible patch representations before and during mixture alignment. Its positives are different masked views of the same constituent clip, and its negatives are views of other constituent clips in the mini-batch; synthetic mixtures are excluded from the negative set. Because unlabeled constituent clips may share sound events, acoustic scenes, or semantics, these negatives distinguish clip instances rather than semantic classes. ICR thereby maintains the constituent-side geometry on which Stage 2 imposes mixture-to-constituent alignment.

## C. Learning from Mixtures and Polyphonic Audio

Real-world recordings often contain multiple simultaneous sound events, and robustness to polyphonic inputs is important for audio classification [16]. Mixup [25] on spectrograms or waveforms is commonly used during supervised fine-tuning to improve classifier robustness [9], [12], [13], [21], [23]. Such label-space regularization does not by itself constrain how a mixture representation relates to its constituent clips. MiCLA instead retains that correspondence during self-supervised pretraining and uses mixing as a latent alignment signal rather than only as augmentation.

A closely related approach is SSLAM [13], which also exploits the correspondence between mixed and pre-mixture spectrograms during self-supervised pre-training. For masked patch prediction, SSLAM uses the average of the corresponding teacher targets from the two constituent inputs. MiCLA replaces this averaged-target objective with directional residual alignment and also introduces constituent-specific alignment for CLS tokens. It uses hard projective residualization for patch tokens and sample-adaptive soft residualization for CLS tokens.

Source separation methods, including permutation-invariant training [26] and MixIT [27], typically seek source-specific waveform estimates or time–frequency masks from physical mixtures. MixIT, for example, trains a separator using mixtures of mixtures and constrains the estimated sources to explain the observed mixtures. MiCLA reconstructs neither waveforms nor masks; its targets are representations of the full clips used in a controlled log-mel interpolation. Accordingly, it addresses constituent-clip-aware representation learning, not signal separation.

MixMAE [28] and MixedAE [29] provide related examples in masked image modeling: they create mosaic-style mixed images by combining patches from different source images and use source correspondence for reconstruction or source recognition. MiCLA applies the broader principle of source correspondence to audio but differs in the mixing mechanism. Rather than replacing patches in a mosaic pattern, MiCLA performs time–frequency-bin-level convex interpolation over entire log-mel spectrograms and aligns teacher latent representations at both patch and clip levels.

## III. METHODOLOGY

This section presents Mixture-to-Constituent Latent Alignment (MiCLA). First, Subsection III-A introduces the selfsupervised objectives used to establish foundational representations for isolated constituent clips. Next, Subsection III-B formulates the challenges of polyphonic mixing and provides geometric motivation for directional residual alignment. Finally, Subsection III-C builds upon these geometric principles to describe the complete two-stage, dual-granularity MiCLA framework.

## A. Preliminaries

We first describe the self-supervised objectives used to learn representations of individual constituent clips.

Masked Self-Distillation. An audio clip is converted into a log-mel spectrogram $\mathbf { x } \in \mathbb { R } ^ { F \times T }$ . Here, $F$ and $T$ denote the numbers of mel-frequency bins and time frames, respectively. The spectrogram is divided into N time–frequency patches, excluding the CLS token. For each masked view, the patch indices are partitioned into a visible set V and a masked set M. These sets satisfy $\nu \cap \mathcal { M } = \emptyset$ and $\mathcal { V } \cup \mathcal { M } = \left\{ 1 , \ldots , N \right\}$ For each clip, multiple masked views are generated by independently sampling a random mask pattern according to the inverse block masking scheme described in Section IV-C [7], [9]. Let $f _ { \theta }$ denote the student audio encoder [14], [19], and let $f _ { \xi }$ denote its exponential moving average (EMA) teacher [6]. The student processes a masked view to output visible patch tokens $\mathbf { u } ^ { ( j ) } \in \mathbb { R } ^ { D } \ ( j \ \in \ \mathcal { V } )$ along with a final-layer CLS representation $\textbf { z } \in \mathbb { R } ^ { D }$ . Concurrently, the teacher processes the unmasked input to provide a patch target $\mathbf { p } ^ { ( j ) } \in \mathbb { R } ^ { D }$ at each patch position $j$ and a layer-normalized clip target $\textbf { t } \in \mathbb { R } ^ { D }$ . To predict the masked regions, the visible student tokens are restored to their original positions, learnable mask tokens are inserted at positions in M, and a predictor $g _ { \mathrm { p } }$ estimates the corresponding teacher patch targets. The masked self-distillation loss is defined as follows:

$$
\ell _ { \mathrm { b a s e } } ( \mathbf { x } ) = \frac { 1 } { | \mathcal { M } | } \sum _ { j \in \mathcal { M } } \left\| \hat { \mathbf { p } } ^ { ( j ) } - \mathrm { s g } \left( \mathbf { p } ^ { ( j ) } \right) \right\| _ { 2 } ^ { 2 } + \left\| \mathbf { z } - \mathrm { s g } \left( \mathbf { t } \right) \right\| _ { 2 } ^ { 2 } .\tag{1}
$$

Here, $\hat { \mathbf { p } } ^ { ( j ) }$ is the predictor output at position $j .$ The operator $\operatorname { s g } ( \cdot )$ denotes stop-gradient. The teacher targets are specified in Section IV-C.

Instance-Level Contrastive Regularization. Following AaSP [15], we apply instance-level contrastive regularization (ICR) to the visible patch representations. The visible patch tokens $\{ \mathbf { u } ^ { ( j ) } \} _ { j \in \mathcal { V } }$ of each masked view are averaged and projected by an MLP $m _ { \phi }$ to obtain an embedding s. Different masked views of the same clip form positive pairs, whereas views of other constituent clips in the mini-batch serve as negatives. For an anchor representation s, let $\mathbf { s } ^ { + }$ denote a positive representation and $\mathcal { N } ^ { - }$ its set of negative representations. With temperature parameter $\tau$ and cosine similarity sim $( \cdot , \cdot )$ , we define $q ^ { + } = \mathrm { e x p } ( \mathrm { s i m } ( \mathbf { s } , \mathbf { s } ^ { + } ) / \tau )$ and $\begin{array} { r } { q ^ { - } \ = \ \sum _ { { \bf s } ^ { - } \in { \mathcal N } ^ { - } } \exp ( \sin ( { \bf s } , { \bf s } ^ { - } ) / \tau ) } \end{array}$ . The contrastive loss is then defined as follows:

$$
\ell _ { \mathrm { c t r } } = - \log \frac { q ^ { + } } { q ^ { + } + q ^ { - } } .\tag{2}
$$

The loss is averaged over the anchor–positive pairs in the mini-batch. Together, $\ell _ { \mathrm { b a s e } }$ and $\ell _ { \mathrm { c t r } }$ establish the constituentside representation space used by MiCLA; however, extending these representations to polyphonic audio with overlapping sound events introduces critical geometric challenges, as formulated next.

## B. Problem Formulation and Motivation

For multi-label classification of polyphonic audio, a mixture representation should preserve the overall acoustic context while retaining information associated with both constituents in a form accessible by downstream models. We seek to learn this property without class labels by using the known correspondence between a synthetic mixture and its two constituent clips.

A naive approach is to align a single mixture representation $\mathbf { z } _ { \mathrm { m i x } }$ with both constituent targets $\mathbf { t } _ { a }$ and $\mathbf { t } _ { b } .$ . With equal weighting, the objective $\| { \bf z } _ { \mathrm { m i x } } - { \bf t } _ { a } \| _ { 2 } ^ { 2 } + \| { \bf z } _ { \mathrm { m i x } } - { \bf t } _ { b } \| _ { 2 } ^ { 2 }$ is minimized at $\mathbf { z } _ { \mathrm { m i x } } = ( \mathbf { t } _ { a } + \mathbf { t } _ { b } ) / 2$ . This optimum represents linear additivity up to scale. However, it exists for any relative angle between the constituent targets and therefore provides no direct pressure to separate their directions.

Projective residualization adds such a separation pressure. To isolate its geometric effect, consider an idealized setting in which each student constituent representation equals its teacher target, the predictor is the identity, and the targets are unit-normalized as $\| \mathbf { t } _ { a } \| _ { 2 } ~ = ~ \| \mathbf { t } _ { b } \| _ { 2 } ~ = ~ 1$ . Let θ denote the angle between $\mathbf { t } _ { a }$ and $\mathbf { t } _ { b } .$ Removing the projection of $\mathbf { z } _ { \mathrm { m i x } }$ onto $\mathbf { t } _ { b }$ yields $\mathbf { z } _ { \mathrm { m i x } } - ( \mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { t } _ { b } ) \mathbf { t } _ { b } .$ , which is orthogonal to $\mathbf { t } _ { b } .$ The residual error relative to $\mathbf { t } _ { a }$ must therefore retain the component of $\mathbf { t } _ { a }$ parallel to $\mathbf { t } _ { b } .$ The minimum squared error is

$$
\operatorname* { m i n } _ { \mathbf { z } _ { \mathrm { m i x } } } \left\| \mathbf { z } _ { \mathrm { m i x } } - ( \mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { t } _ { b } ) \mathbf { t } _ { b } - \mathbf { t } _ { a } \right\| _ { 2 } ^ { 2 } = ( \mathbf { t } _ { a } ^ { \top } \mathbf { t } _ { b } ) ^ { 2 } = \cos ^ { 2 } \theta .\tag{3}
$$

The reverse direction yields the same minimum error, with both directional minima achieved simultaneously at ${ \bf z } _ { \mathrm { m i x } } =$ $\mathbf { t } _ { a } + \mathbf { t } _ { b }$ , giving a minimum bidirectional loss of $2 \cos ^ { 2 } \theta .$ Under these idealized conditions, zero loss is achieved when target directions are orthogonal, while directional overlap incurs a strictly positive loss. This analysis motivates residual alignment as a geometric pressure toward reduced directional overlap, rather than a guarantee of orthogonality in practical training.

MiCLA realizes this geometric pressure through bidirectional alignment. In each direction, it suppresses the mixture component aligned with one constituent and aligns the residual with the teacher target of the other. The counterpart representation remains on the differentiable student path, allowing the residual loss to shape the constituent geometry. Reversing the target and counterpart roles applies this pressure to both constituent representations.

MiCLA applies the alignment at two granularities. Patch tokens encode local time–frequency content and undergo full projection removal. In contrast, because the CLS token summarizes global context that may be shared across clips, MiCLA employs bounded, sample-adaptive suppression ranging from no removal to full orthogonal residualization. In practice, optimizing these alignment losses provides an effective inductive bias that promotes an additive and separated latent geometry during training, while soft residualization at the CLS level adaptively preserves shared context across constituent clips.

## C. The MiCLA Pre-Training Framework

Guided by the geometric motivation established above, we operationalize directional residual alignment into a concrete pre-training framework. Below, we outline the two-stage training strategy, detail the dual-granularity alignment operations, and define the complete pre-training objectives.

1) Overview: Two-Stage Pre-Training Strategy: MiCLA adopts a two-stage pre-training strategy following prior work [13]. As illustrated in Fig. 1, Stage 1 optimizes masked self-distillation and ICR using only constituent clips to establish a structured constituent-side representation space.

Stage 2 starts from the Stage 1 model and introduces mixture-to-constituent alignment, as shown in Fig. 2. Importantly, the constituent-side objectives are retained during Stage 2 to preserve the representation structure established in Stage 1 while introducing mixture-based alignment.

For each aligned view in Stage 2, the two constituent clips and their synthetic mixture share the same visible set V and masked set M. Shared random masking establishes token-wise correspondence at the same time–frequency positions.

2) Dual-Granularity Mixture-to-Constituent Alignment: Building upon the two-stage schedule and shared masking scheme, we now formalize the core directional alignment operations introduced in Stage 2. Given the standardized logmel spectrograms $\mathbf { x } _ { a }$ and $\mathbf { x } _ { b }$ of two constituent clips, MiCLA constructs a synthetic mixture by convexly interpolating them as follows:

$$
\mathbf { x } _ { \mathrm { m i x } } = \mu \mathbf { x } _ { a } + ( 1 - \mu ) \mathbf { x } _ { b } .\tag{4}
$$

The mixing coefficient is sampled as $\mu \sim \mathrm { B e t a } ( \eta , \eta )$ . The interpolation is applied after standardizing the constituent log-mel spectrograms. It provides a controlled feature-space intervention with exact constituent correspondence. It is not intended to model physical waveform mixing.

![](images/72966e2834be12b9e924a5d1ec3f472fec2e5707633ae490589e106e03af17d1.jpg)  
Fig. 1. Constituent-side pre-training in Stage 1. Masked self-distillation optimizes $\ell _ { \mathrm { b a s e } } .$ . Instance-level contrastive regularization $\ell _ { \mathrm { c t r } }$ structures representations across masked views. Both objectives are retained in Stage 2. Boxes labeled $\mathbf { \ddot { C } } ^ { 5 }$ denote CLS tokens.

Directional Alignment Setup. For the ordered pair $( \mathbf { x } _ { a } , \mathbf { x } _ { b } ) , \mathbf { x } _ { a }$ is the target and $\mathbf { x } _ { b }$ is the counterpart to be suppressed. The reverse direction $\left( \mathbf { x } _ { b } , \mathbf { x } _ { a } \right)$ exchanges these roles. For latent vectors ${ \mathbf { u } } , { \mathbf { v } } \in \mathbb { R } ^ { D }$ , we define the residualization operator as follows:

$$
\rho _ { \alpha } ( \mathbf { u } , \mathbf { v } ) = \mathbf { u } - \alpha \frac { \mathbf { u } ^ { \top } \mathbf { v } } { \| \mathbf { v } \| _ { 2 } ^ { 2 } } \mathbf { v } .\tag{5}
$$

Here, u is the representation to be residualized, v defines the counterpart direction, and $\alpha \in [ 0 , 1 ]$ controls the suppression strength. Specifically, setting $\alpha = 0$ leaves u unchanged, while setting $\alpha = 1$ fully removes the orthogonal projection of u onto v, yielding a hard residual orthogonal to v when ${ \textbf { v } } \neq$ 0. This residualization serves as a latent-space inductive bias without assuming that a learned mixture representation is the exact linear sum of its constituent representations.

Patch-Level Hard Residualization. Because patch tokens capture localized time–frequency content, MiCLA fully removes the component aligned with the counterpart at each visible position. Let $\mathbf { u } _ { \mathrm { m i x } } ^ { ( j ) }$ and $\mathbf { u } _ { b } ^ { ( j ) }$ denote the student patch tokens of the mixture and counterpart at a shared visible position $j \in \mathcal V$ . For direction $\left( \mathbf { x } _ { a } , \mathbf { x } _ { b } \right)$ , the hard residual is defined as $\mathbf { r } _ { a | b } ^ { ( j ) } = \rho _ { 1 } ( \mathbf { u } _ { \operatorname* { m i x } } ^ { ( j ) } , \mathbf { u } _ { b } ^ { ( j ) } )$ for $j \in \mathcal V$ . These residual tokens are restored to their original positions, and learnable mask tokens are inserted at positions in $\mathcal { M }$ . The resulting sequence is processed by the same predictor $g _ { \mathrm { p } }$ used for masked self-distillation, and the patch-level loss aligns each resulting output $\mathbf { h } _ { a \mid b } ^ { ( j ) }$ with the target constituent’s teacher representation $\mathbf { p } _ { a } ^ { ( j ) }$ at masked position $j \in \mathcal { M }$ . The patchlevel alignment loss is defined as follows:

$$
\ell _ { \mathrm { p a t c h } } ( \mathbf { x } _ { a } , \mathbf { x } _ { b } ) = \frac { 1 } { \vert \mathcal { M } \vert } \sum _ { j \in \mathcal { M } } \left. \mathbf { h } _ { a \vert b } ^ { ( j ) } - \mathrm { s g } \left( \mathbf { p } _ { a } ^ { ( j ) } \right) \right. _ { 2 } ^ { 2 } .\tag{6}
$$

Only the teacher target is detached. Gradients propagate through both the mixture and counterpart student paths.

Clip-Level Soft Residualization. The CLS token contains global context that may be shared by the two constituent clips, such as common acoustic environments or overlapping sound events. Full orthogonal projection removal could discard this useful shared information. MiCLA therefore selects the suppression strength adaptively for each sample. Let ${ \bf z } _ { \mathrm { m i x } } , { \bf z } _ { a }$ and $\mathbf { z } _ { b }$ denote the student CLS representations of the mixture, target constituent, and counterpart constituent, respectively. For direction $\left( \mathbf { x } _ { a } , \mathbf { x } _ { b } \right)$ , MiCLA selects the suppression strength from the student-side geometry by minimizing the squared distance between the residualized mixture representation and the target constituent’s student representation ${ \bf z } _ { a }$

![](images/a011f6c4155433af0711ea437563c75f6910f0804225bec2b6f8edaa12429ec9.jpg)  
Fig. 2. Mixture-to-constituent latent alignment introduced in Stage 2. In each target–counterpart direction, MiCLA suppresses the mixture component aligned with the counterpart. The resulting residual is aligned with the target constituent’s EMA-teacher representation. Patch tokens use hard projective residualization. CLS tokens use sample-adaptive soft residualization. Boxes labeled $\mathrm { ^ { * } C ^ { * } } { } ^ { * }$ denote CLS tokens.

$$
\boldsymbol { \alpha } _ { a | b } ^ { \star } = \mathrm { s g } \left( \arg \operatorname* { m i n } _ { \boldsymbol { \alpha } \in [ 0 , 1 ] } \left\| \rho _ { \boldsymbol { \alpha } } \left( \mathbf { z } _ { \mathrm { m i x } } , \mathbf { z } _ { b } \right) - \mathbf { z } _ { a } \right\| _ { 2 } ^ { 2 } \right) .\tag{7}
$$

$\mathbf { A } \mathbf { s }$ derived in Appendix A, this student-side gate-selection problem has the following closed-form solution:

$$
\boldsymbol { \alpha } _ { a | b } ^ { \star } = \mathrm { s g } \left( \mathrm { c l i p } _ { [ 0 , 1 ] } \left( 1 - \frac { \mathbf { z } _ { a } ^ { \top } \mathbf { z } _ { b } } { \mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { z } _ { b } } \right) \right) .\tag{8}
$$

Clipping restricts the operation to the range between no suppression $( \alpha = 0 )$ and full projection removal $( \alpha = 1 )$ When $\mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { z } _ { b }$ is positive and non-negligible, a near-zero target–counterpart overlap ${ \mathbf z } _ { a } ^ { \top } { \mathbf z } _ { b }$ yields $\alpha _ { a | b } ^ { \star } \approx 1$ , whereas a larger positive overlap ratio reduces $\alpha _ { a | b } ^ { \star } ,$ thereby retaining more counterpart-aligned latent content that may represent shared context. Computing the gate entirely from student representations keeps all inner products in a common student latent coordinate system. Applying the stop-gradient operator $\operatorname { s g } ( \cdot )$ detaches $\alpha _ { a | b } ^ { \star }$ from backpropagation; thus, ${ \bf z } _ { a }$ affects the forward gate value but receives no gradient through this gate-selection path, preventing target-side information from creating a direct optimization shortcut. Within the subsequent residualization $\rho _ { \alpha _ { a \mid b } ^ { \star } } \bigl ( \mathbf { z } _ { \operatorname* { m i x } } , \mathbf { z } _ { b } \bigr )$ , however, $\mathbf { z } _ { \mathrm { m i x } }$ and $\mathbf { z } _ { b }$ remain differentiable. Finally, let $\mathbf { t } _ { a }$ denote the EMA-teacher clip target of the target constituent $\mathbf { x } _ { a }$ . The residualized CLS representation is then aligned with the detached teacher target $\mathbf { t } _ { a }$ via the clip-level alignment loss:

$$
\begin{array} { r } { \ell _ { \mathrm { c l i p } } ( \mathbf x _ { a } , \mathbf x _ { b } ) = \left\| \rho _ { \alpha _ { a \left| b \right. } ^ { \star } } ( \mathbf z _ { \mathrm { m i x } } , \mathbf z _ { b } ) - \mathrm { s g } \left( \mathbf t _ { a } \right) \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{9}
$$

Unlike patch-level alignment, clip-level alignment operates directly on the representations without using the predictor.

3) Pre-Training Objectives: Having formulated the constituent-side and mixture-alignment losses, we now consolidate them into the complete optimization objectives for Stage 1 and Stage 2.

Stage 1: Constituent-Side Objective. Stage 1 uses only constituent clips. The expectation is taken over the Stage 1 training distribution, including sampled clips and masked views. The Stage 1 objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s t a g e 1 } } = \mathbb { E } \left[ \lambda _ { \mathrm { b a s e } } \ell _ { \mathrm { b a s e } } ( \mathbf { x } ) + \lambda _ { \mathrm { c t r } } \ell _ { \mathrm { c t r } } \right] . } \end{array}\tag{10}
$$

Here, $\lambda _ { \mathrm { b a s e } }$ and $\lambda _ { \mathrm { c t r } }$ weight masked self-distillation and ICR, respectively.

Stage 2: Joint Mixture–Constituent Objective. Stage 2 retains the constituent-side objectives and adds bidirectional mixture-to-constituent alignment. For a constituent pair $\left( \mathbf { x } _ { a } , \mathbf { x } _ { b } \right)$ , the averaged base loss is $\bar { \ell } _ { \mathrm { b a s e } } =$ $\begin{array} { r } { \frac { 1 } { 2 } \left[ \ell _ { \mathrm { b a s e } } ( \mathbf { x } _ { a } ) + \ell _ { \mathrm { b a s e } } ( \mathbf { x } _ { b } ) \right] } \end{array}$ ]. For the fixed mixture, the directional loss is $\ell _ { \mathrm { m i x } } ( \mathbf x _ { a } , \mathbf x _ { b } ) = \ell _ { \mathrm { p a t c h } } ( \mathbf x _ { a } , \mathbf x _ { b } ) + \ell _ { \mathrm { c l i p } } ( \mathbf x _ { a } , \mathbf x _ { b } )$ where the first argument denotes the target and the second denotes the counterpart. The reverse term uses the same mixture and mixing coefficient while exchanging only these roles. The complete bidirectional loss is $\bar { \ell } _ { \mathrm { m i x } } ~ =$ $\begin{array} { r } { \frac { 1 } { 2 } \left[ \ell _ { \mathrm { m i x } } ( \mathbf x _ { a } , \mathbf x _ { b } ) + \ell _ { \mathrm { m i x } } ( \mathbf x _ { b } , \mathbf x _ { a } ) \right] } \end{array}$ . The patch- and clip-level losses and the two directions are weighted equally. Although the mixing ratio $\mu$ is used to interpolate the input spectrograms, it does not weight the alignment terms, encouraging both constituents to remain accessible even in imbalanced mixtures. The expectation is taken over the Stage 2 training distribution, including constituent pairings, masked views, and mixing coefficients. The Stage 2 objective is

$$
{ \mathcal { L } } _ { \mathrm { s t a g e 2 } } = \mathbb { E } \left[ \lambda _ { \mathrm { b a s e } } { \bar { \ell } } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { c t r } } { \ell } _ { \mathrm { c t r } } + \lambda _ { \mathrm { m i x } } { \bar { \ell } } _ { \mathrm { m i x } } \right] .\tag{11}
$$

Here, $\lambda _ { \operatorname* { m i x } }$ weights the proposed alignment objective. In Stage $2 , \ell _ { \mathrm { c t r } }$ is computed only from constituent views, whereas synthetic mixture views contribute only to $\bar { \ell } _ { \mathrm { m i x } }$

After pre-training, only the student encoder $f _ { \theta }$ is retained for downstream tasks. The EMA teacher $f _ { \xi }$ is discarded. The predictor $g _ { \mathrm { p } }$ and projection head $m _ { \phi }$ are also discarded. Mixture construction and residualization are not used during downstream inference. MiCLA therefore introduces no mixture-specific inference cost.

## IV. EXPERIMENTS

We evaluate MiCLA along four complementary dimensions: (1) supervised fine-tuning to assess downstream adaptability on polyphonic multi-label classification; (2) frozen-encoder probing via Convex Gated Probing (CGP) to measure multilabel transfer using layer- and token-wise features; (3) finetuning and frozen CGP across single-label datasets to characterize transfer beyond the target multi-label setting; and (4) Joint Semantic Retrieval to evaluate whether both known constituent classes remain jointly accessible from the frozen CLS representation of a synthetic mixture.

The classification benchmarks span two broad regimes: multi-label sound-event classification (AudioSet and FSD-50K), where clips may contain multiple overlapping acoustic events, and single-label classification across nine diverse speech, music, and environmental sound benchmarks. In finetuning, the encoder and classification head are updated jointly to evaluate adaptability. For frozen evaluation, the encoder remains fixed. We measure representational accessibility primarily with CGP, which accesses features across layers and tokens, and use a final-layer linear probe as a supplementary diagnostic (Appendix B).

## A. Multi-Label Datasets

AudioSet [16] is a YouTube-derived multi-label dataset with 527 sound-event classes. AS-2M denotes the available unbalanced training split of approximately 1.96M clips, and AS-20K denotes the available balanced subset of approximately 21k clips; both use the same available evaluation set of approximately 19k clips and report mean average precision (mAP).

FSD-50K [30] is a Freesound-based multi-label soundevent dataset with 51,197 clips and 200 classes from the AudioSet ontology. We use its official 36,796/4,170/10,231 train/validation/evaluation split for frozen probing and report mAP, thereby evaluating transfer beyond AudioSet’s recording source and label distribution.

## B. Single-Label Datasets

For single-label classification, we use the downstream datasets included in the EVAR<sup>1</sup> platform. Although EVAR is often used for linear probing, we use its dataset suite mainly for frozen CGP evaluation [17]; fine-tuning performance on ESC-50 and SPC-2 is also reported in Table III.

ESC-50 [18] is a single-label dataset for environmental sound classification. It contains 2,000 five-second clips across 50 classes, with 40 clips per class; we follow the official fivefold cross-validation protocol and report average accuracy.

UrbanSound8K [31], abbreviated as US8K, is a dataset for classifying urban sound events. It contains 8,732 clips of up to four seconds across ten classes; we follow the official ten-fold split and report average accuracy.

Speech Commands V2 [32], abbreviated as SPC-2, is a keyword spotting dataset for classifying short spoken commands. It contains 105,829 approximately one-second utterances across 35 classes; the official split provides 84,843 training, 9,981 validation, and 11,005 test clips, and we report accuracy.

VoxCeleb-1 [33] is a large-scale speech dataset for speaker identification. It contains 153,516 approximately eight-second utterances from 1,251 speakers; following prior work [21], we use 138,361 training, 6,904 validation, and 8,251 test clips.

VoxForge [34] is a speech dataset used for language identification. It contains 176,428 approximately six-second utterances across six languages; following prior work [21], we use 121,281 training, 26,684 validation, and 28,463 test clips.

CREMA-D [35] is a dataset for speech emotion recognition. It contains 7,438 approximately 2.5-second utterances across six emotions; following prior work [21], we use 5,155 training, 732 validation, and 1,551 test clips.

GTZAN [36] is a dataset for music genre classification. Following prior work [21], we use 930 thirty-second music clips across ten genres, split into 443 training, 197 validation, and 290 test clips.

NSynth [37] is a large-scale dataset used for instrument timbre classification. It contains 305,979 four-second clips across 11 instrument families; the official split provides 289,205 training, 12,678 validation, and 4,096 test clips.

Surge [38] is a pitch classification dataset generated by the Surge synthesizer. It contains 183,392 four-second clips across 88 MIDI-note classes; following prior work [21], we use 148,896 training, 17,160 validation, and 17,336 test clips.

## C. Implementation Details

Unless otherwise stated, all audio data are resampled to 16 kHz. Log-mel spectrograms are computed using a 400-sample Hann window, a hop length of 160 samples, 128 mel bins, and an FFT size of 1024. Before encoder input, each log-mel spectrogram is standardized using a fixed mean of −4.268 and a standard deviation of 4.596. All experiments are conducted on NVIDIA H100 80-GB GPUs. Due to the substantial computational cost of repeated pre-training, unless explicitly noted otherwise, downstream evaluations are conducted across three random seeds using a single shared pre-trained checkpoint, reporting the mean and standard deviation.

1) Pre-Training Setup: Table I summarizes the hyperparameters used for both stages of pre-training. Pre-training is performed on AS-2M. AudioSet clips are approximately ten seconds long; following prior work [8], [21], we randomly crop six-second segments for training efficiency. We set the log-mel spectrogram length to 608 time frames by truncation or zero padding.

Encoder and Positional Embeddings. The audio encoder is a post-norm 12-layer Vision Transformer [14] with hidden dimension $D = 7 6 8 .$ . Input spectrograms are partitioned into non-overlapping $1 6 ~ \times ~ 1 6$ time–frequency patches [9]. For positional representations, absolute positional embeddings are applied along the frequency axis to preserve absolute spectral positions, while Rotary Positional Embeddings (RoPE) [43] are applied along the time axis to model relative temporal ordering. In downstream tasks where input spectrogram length differs from pre-training, shorter inputs are processed directly, and longer inputs are extrapolated using YaRN [44].

TABLE I  
HYPERPARAMETERS FOR PRE-TRAINING AND FINE-TUNING.
<table><tr><td rowspan="2">Hyperparameters</td><td rowspan="2">Pre-Training (Stage 1) AS-2M</td><td rowspan="2">Pre-Training (Stage 2) AS-2M</td><td colspan="4">Fine-Tuning</td></tr><tr><td>AS-2M</td><td>AS-20K</td><td>ESC-50</td><td>SPC-2</td></tr><tr><td colspan="7">Optimizer</td></tr><tr><td>LR schedule</td><td></td><td></td><td>Cosine Annealing [40] with linear warm-up</td><td></td><td></td><td></td></tr><tr><td>Peak LR</td><td> $5 . 0 \times 1 0 ^ { - 4 }$ </td><td> $5 . 0 \times 1 0 ^ { - 5 }$ </td><td colspan="4"> $\dot { 5 } . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Minimum LR</td><td>0.0</td><td>0.0</td><td> $1 . 0 \times 1 0 ^ { - 6 }$ </td><td> $1 . 0 \times 1 0 ^ { - 6 }$ </td><td> $1 . 0 \times 1 0 ^ { - 7 }$ </td><td> $1 . 0 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Layer decay [41]</td><td>N/A</td><td>N/A</td><td>0.65</td><td>0.80</td><td>0.75</td><td>0.75</td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Training steps</td><td>400K</td><td>200K</td><td>300K</td><td>40K</td><td>8K</td><td>160K</td></tr><tr><td>Warm-up steps</td><td>50K</td><td>25K</td><td>30K</td><td>4K</td><td>800</td><td>16K</td></tr><tr><td>Weighted sampling</td><td>N/A</td><td>N/A</td><td>200K</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Batch size per GPU</td><td>64</td><td>64</td><td>32</td><td>32</td><td>128</td><td>128</td></tr><tr><td>Masked views</td><td>8</td><td> $^ { 8 , 4 ^ { * } }$ </td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td># of GPUs</td><td>4</td><td>4</td><td>4</td><td>4</td><td>1</td><td>1</td></tr><tr><td>Mixup [25]</td><td>N/A</td><td>N/A</td><td>0.8</td><td>0.8</td><td>0.0</td><td>0.8</td></tr><tr><td>Drop path [42]</td><td>0.0</td><td>0.0</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr></table>

<sup>\*</sup> In Stage 2, “8,4” denotes 8 constituent views per clip; 4 mixture views share masks with 4 of them. Mixture views enter only $\bar { \ell } _ { \mathrm { m i x } } ;$ all constituent views enter $\bar { \ell } _ { \mathrm { b a s e } }$ and $\ell _ { \mathrm { c t r } } ,$ and matched ones also enter $\bar { \ell } _ { \mathrm { m i x } } .$

Masking and Self-Distillation Architecture. Following EAT [9], we use $5 \times 5$ inverse block masking with a masking ratio of 80%. Multi-mask training [7] applies multiple independently sampled mask patterns to the patch sequence of the same input spectrogram. The EMA teacher parameters are updated with a momentum linearly scheduled from 0.9998 to 0.99999, warmed up across the entire duration in Stage 1 and for the first 100K steps in Stage 2 [13]. Patch-level teacher targets $\mathbf { p } ^ { ( j ) }$ are formed by averaging representations across all EMA teacher layers at patch position j. The cliplevel teacher target t is obtained by averaging all final-layer patch tokens from the EMA teacher and then applying layer normalization. The patch predictor $g _ { \mathrm { p } }$ is structured as a sixlayer convolutional neural network (CNN) following EAT.

Instance-Level Contrastive Regularization. The projection MLP $m _ { \phi }$ consists of Linear, GELU, and Linear layers, with an intermediate hidden dimension of $D / 2 = 3 8 4$ . The contrastive temperature is set to $\tau = 0 . 2$ . The contrastive regularization weight is set to $\lambda _ { \mathrm { c t r } } = 0 . 0 1$ following AaSP [15].

Mixture Construction and Residualization. In Stage 2, synthetic mixtures are formed via round-robin pairing within each GPU mini-batch as ${ \bf x } _ { \mathrm { m i x } , i } = \mu _ { i } { \bf x } _ { i } + ( 1 - \mu _ { i } ) { \bf x } _ { i + 1 }$ with wrap-around indexing. Each $\mu _ { i }$ is sampled from Beta $\iota ( \eta , \eta )$ with default parameter $\eta = 4$ . For each pair, eight masked views are generated per constituent clip, four of which share mask patterns with four mixture views. Mixture views enter only $\bar { \ell } _ { \mathrm { m i x } }$ , whereas constituent views enter $\bar { \ell } _ { \mathrm { b a s e } }$ and $\ell _ { \mathrm { c t r } }$ (with matched views also entering $\bar { \ell } _ { \mathrm { m i x } } )$ . In all experiments, the squared-error losses $\ell _ { \mathrm { b a s e } }$ and $\ell _ { \mathrm { m i x } }$ are normalized by dividing by $\sqrt { D }$ before weighting. We set loss weights $\lambda _ { \mathrm { b a s e } } = 1 . 0$ and $\lambda _ { \operatorname* { m i x } } = 1 . 0 .$ In projective residualization and clip gating, divisions by $\| \mathbf { v } \| _ { 2 } ^ { 2 }$ and $\mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { z } _ { b }$ are stabilized with small signpreserving offset $( 1 0 ^ { - 6 } )$ to prevent numerical instability.

2) Fine-Tuning Protocol: Table I summarizes the hyperparameters used for fine-tuning. For fine-tuning evaluation, a classification head is attached to the CLS token of the pre-trained encoder. The encoder and classification head are updated jointly using supervised data from each downstream task. For augmentation, we employ Mixup [25] together with SpecAugment [45]. In SpecAugment, the maximum timemask width and frequency-mask width are each set to 20% of the corresponding spectrogram dimension. The ESC-50 experiments are conducted without any augmentation. For all fine-tuning experiments other than ESC-50, we keep an exponential moving average of the model parameters, using a decay rate of 0.995, to improve inference stability. The training objective is chosen according to the dataset type. Binary crossentropy (BCE) with logits is used for multi-label classification. Cross-entropy is used for single-label classification. When Mixup is enabled for a single-label task, BCE is used instead of cross-entropy to handle soft targets. Fine-tuning reflects not only the quality of the pre-trained representation but also its adaptability to each downstream task.

3) Frozen Probing Protocol: For our primary frozenrepresentation evaluation, we use Convex Gated Probing (CGP), proposed in BAT [17]. CGP is based on the observation that a standard final-layer linear probe, commonly used in image representation learning, may underestimate self-supervised learning (SSL) representations for audio. In CGP, the encoder is frozen, and patch tokens and CLS tokens are extracted from each layer. Layer-wise information is aggregated using learnable convex weights parameterized by a softmax function, and a classifier is trained using similarities to learnable prototypes. This allows useful acoustic information contained not only in the final layer but also in intermediate layers to contribute to the evaluation. We interpret CGP results as measuring the accessibility of information distributed across layers and tokens, rather than the linear separability of finallayer summary features. The latter is separately examined by the supplementary linear probe in Appendix B. Table II summarizes the hyperparameters used for CGP. Following

TABLE II  
HYPERPARAMETERS FOR CONVEX GATED PROBING (CGP). THE LOSS FUNCTION IS ASYMMETRIC BINARY CROSS-ENTROPY (A-BCE) [46] FOR MULTI-LABEL DATASETS AND CROSS-ENTROPY (CE) FOR SINGLE-LABEL DATASETS.
<table><tr><td rowspan="2">Hyperparameters</td><td colspan="10">AS-2M AS-20K FSD-50K ESC-50</td><td rowspan="2"></td><td rowspan="2">NSynth Surge</td></tr><tr><td></td><td></td><td></td><td></td><td>US8K</td><td></td><td>SPC-2 VoxCeleb-1</td><td>VoxForge</td><td>CREMA-D</td><td>GTZAN</td></tr><tr><td>Optimizer</td><td colspan="10">AdamW [39] with</td><td></td><td></td><td></td></tr><tr><td>LR schedule</td><td></td><td colspan="10"></td></tr><tr><td>Peak LR</td><td></td><td></td><td></td><td></td><td>Cosine Annealing [40] with linear warm-up</td><td> $\bar { 1 . 0 } \times 1 0 ^ { - 3 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Minimum LR</td><td colspan="10"></td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td> $1 . 0 \times 1 0 ^ { - 6 }$  0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>Warm-up steps</td><td>1K</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Training steps</td><td>200K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td><td>25K</td></tr><tr><td>Weighted sampling</td><td>200K</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Batch size per GPU</td><td>32</td><td>32</td><td>32</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td><td>16</td><td>128</td><td>128</td></tr><tr><td># of GPUs</td><td>4</td><td>4</td><td>4</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Loss function</td><td>A-BCE</td><td>A-BCE</td><td>A-BCE</td><td>CE</td><td>CE</td><td>CE</td><td>CE</td><td>CE</td><td>CE</td><td>CE</td><td>CE</td><td>CE</td></tr></table>

BAT [17], the number of prototypes is set to 10,000. For all CGP evaluations, no data augmentation is used. We evaluate all baselines with the same CGP probing head and training recipe, strictly adhering to the authors’ official data preprocessing pipelines in addition to using their official weights: EAT [9]<sup>2</sup>, SSLAM [13]<sup>3</sup>, AudioMosaic $[ 1 2 ] ^ { 4 }$ , and BAT [17]<sup>5</sup>. For FSD-50K and GTZAN, we average chunk-level predictions, using each checkpoint’s native length: 608 frames for MiCLA and 1024 frames for the other models.

As a supplementary diagnostic, we also conduct simple linear probing on the multi-label settings and report the results in Appendix B. For this diagnostic, the encoder is frozen and only a linear classifier is trained. The classifier input is constructed by concatenating the final-layer CLS token with the average of all final-layer patch tokens. A sigmoid output is used for multi-label classification. Unlike CGP, this probe does not use intermediate-layer aggregation or prototype-based readout. We interpret it as a diagnostic of linear accessibility from final-layer summary features rather than as our primary frozen-probing protocol.

## D. Fine-Tuning Results

Table III presents the fine-tuning results. MiCLA demonstrates competitive transfer performance across multi-label benchmarks: among the models listed in Table III, MiCLA achieves 43.2% mAP on AS-20K, improving upon prior baselines on this balanced benchmark. On the full-scale AS-2M dataset, MiCLA achieves 50.3% mAP, performing competitively with prior top-performing models (e.g., 50.2% for SSLAM and AudioMosaic).

On single-label classification, MiCLA achieves 97.6% accuracy on ESC-50, performing on par with or slightly above top baselines, while attaining competitive performance on SPC-2 $( 9 8 . 1 \pm 0 . 1 \% )$ .

## E. CGP Results on Multi-Label Datasets

Table IV compares frozen encoders under the common CGP protocol. MiCLA achieves the highest performance among the compared models on both AudioSet benchmarks, obtaining $4 6 . 3 \pm 0 . 0$ mAP on AS-2M and $3 7 . 8 \pm 0 . 0$ mAP on AS-20K, exceeding the next-highest values by 0.6 and 0.5 points, respectively. On the cross-dataset FSD-50K benchmark, MiCLA obtains $6 2 . 1 \pm 0 . 1$ mAP, improving upon several prior selfsupervised models including EAT (59.5%), SSLAM (57.8%), and AudioMosaic (57.0%), while performing close to BAT (62.5%).

TABLE III  
FINE-TUNING RESULTS. MICLA RESULTS ARE REPORTED AS THE MEAN ± STANDARD DEVIATION OVER THREE RANDOM SEEDS, WHILE THE RESULTS OF THE COMPARED METHODS ARE CITED AS REPORTED IN THE CORRESPONDING PAPERS.
<table><tr><td>Model</td><td>#Params.</td><td>AS-2M mAP (%)</td><td>AS-20K mAP (%)</td><td>ESC-50 Acc (%)</td><td>SPC-2 Acc (%)</td></tr><tr><td></td><td>86M</td><td>47.1</td><td>32.3</td><td>89.6</td><td>97.7</td></tr><tr><td>MaskSpec [4] data2vec [6]</td><td>94M</td><td></td><td>34.5</td><td></td><td></td></tr><tr><td>Audio-MAE [3]</td><td>86M</td><td>47.3</td><td>37.1</td><td>94.1</td><td>98.3</td></tr><tr><td>M2D/0.7 [21]</td><td>86M</td><td>47.9</td><td>38.6</td><td>96.0</td><td>98.4</td></tr><tr><td>ASiT [22]</td><td>86M</td><td>48.0</td><td>38.3</td><td>95.6</td><td>98.9</td></tr><tr><td>BEATsiter3 [5]</td><td>90M</td><td>48.0</td><td>38.3</td><td>95.6</td><td>98.3</td></tr><tr><td>ATST-Frame [8]</td><td>86M</td><td>48.0</td><td>39.0</td><td></td><td>98.1</td></tr><tr><td>EAT [9]</td><td>88M</td><td>48.6</td><td>40.2</td><td>95.9</td><td>98.3</td></tr><tr><td>ASDA [23]</td><td>93M</td><td>49.0</td><td>41.5</td><td>96.1</td><td>98.3</td></tr><tr><td>BAT [17]</td><td>93M</td><td>48.9</td><td>41.6</td><td>96.0</td><td>99.1</td></tr><tr><td>SSLAM [13]</td><td>88M</td><td>50.2</td><td>40.9</td><td>96.2</td><td>98.1</td></tr><tr><td>AaSP [15]</td><td>87M</td><td>49.8</td><td>41.9</td><td>97.5</td><td>97.9</td></tr><tr><td>AudioMosaic [12]</td><td>86M</td><td>50.2</td><td>42.5</td><td>97.5</td><td>98.4</td></tr><tr><td>MiCLA (ours)</td><td>86M</td><td>50.3±0.0</td><td>43.2±0.1</td><td>97.6±0.1 98.1±0.1</td><td></td></tr></table>

TABLE IV  
CGP RESULTS ON MULTI-LABEL EVALUATION SETTINGS. VALUES ARE REPORTED AS THE MEAN ± STANDARD DEVIATION OVER THREE RANDOM SEEDS.
<table><tr><td>Model</td><td>AS-2M mAP (%)</td><td>AS-20K mAP (%)</td><td>FSD-50K mAP (%)</td></tr><tr><td>EAT [9]</td><td> $4 5 . 0 { \pm } 0 . 0 $ </td><td> $3 6 . 1 { \pm } 0 . 0 $ </td><td> $5 9 . 5 { \pm } 0 . 1 $ </td></tr><tr><td>SSLAM [13]</td><td> $4 3 . 8 { \pm } 0 . 1$ </td><td> $3 4 . 5 { \pm } 0 . 0$ </td><td> $5 7 . 8 { \pm } 0 . 1 $ </td></tr><tr><td>AudioMosaic [12]</td><td> $4 4 . 1 { \pm } 0 . 0 $ </td><td> $3 5 . 0 { \pm } 0 . 0 $ </td><td> $5 7 . 0 { \pm } 0 . 3 $ </td></tr><tr><td>BAT [17]</td><td> $4 5 . 7 { \pm } 0 . 0 $ </td><td> $3 7 . 3 { \pm } 0 . 1 $ </td><td> $6 2 . 5 { \pm } 0 . 1 $ </td></tr><tr><td>MiCLA (ours)</td><td> ${ \bf 4 6 . 3 2 0 . 0 }$ </td><td> $3 7 . 8 { \pm } 0 . 0$ </td><td>62.1±0.1</td></tr></table>

TABLE V  
CGP RESULTS ON SINGLE-LABEL CLASSIFICATION TASKS. VALUES ARE REPORTED AS THE MEAN ± STANDARD DEVIATION OVER THREE RANDOM SEEDS.
<table><tr><td>Model</td><td>ESC-50 Acc (%)</td><td>US8K Acc (%)</td><td>SPC-2  $\mathrm { A c c } \ ( \% )$ </td><td>VoxCeleb-1 Acc (%)</td><td>VoxForge  $\mathrm { A c c } \ ( \% )$ </td><td>CREMA-D Acc (%)</td><td>GTZAN Acc (%)</td><td>NSynth Acc (%)</td><td>Surge Acc (%)</td></tr><tr><td>EAT [9]</td><td> $9 3 . 2 { \pm } 0 . 1 $ </td><td> $8 9 . 9 2 0 . 1 $ </td><td> $9 5 . 7 \pm 0 . 0$ </td><td> $6 6 . 3 { \pm } 0 . 1 $ </td><td> $9 8 . 4 \pm 0 . 2 $ </td><td> $7 4 . 6 { \pm } 0 . 1$ </td><td> $8 3 . 8 { \pm } 0 . 2 $ </td><td> $8 3 . 5 { \pm } 0 . 2 $ </td><td> $4 9 . 4 { \pm } 0 . 2 $ </td></tr><tr><td>SSLAM [13]</td><td> $9 2 . 6 { \pm } 0 . 1$ </td><td> $8 8 . 7 { \pm } 0 . 3 $ </td><td> $9 5 . 9 2 0 . 1 $ </td><td>64.0±0.2</td><td> $9 8 . 3 { \pm } 0 . 2 \ $ </td><td> $7 4 . 9 { \pm } 0 . 2 $ </td><td> $8 4 . 6 { \pm } 0 . 2 $ </td><td> $8 2 . 3 { \pm } 0 . 3 $ </td><td>48.7±0.0</td></tr><tr><td>AudioMosaic [12]</td><td> $9 1 . 0 { \pm } 0 . 1$ </td><td> $8 9 . 9 2 0 . 1 $ </td><td> $9 5 . 9 2 0 . 1 $ </td><td> $7 6 . 2 { \pm } 0 . 2 $ </td><td> $9 7 . 7 { \pm } 0 . 1 $ </td><td> $7 4 . 1 { \pm } 0 . 3 $ </td><td> $8 3 . 4 { \pm } 0 . 3 $ </td><td> $8 2 . 7 { \pm } 0 . 1 $ </td><td>47.9±0.1</td></tr><tr><td>BAT [17]</td><td> ${ \bf 9 4 . 2 \pm 0 . 1 }$ </td><td> $9 1 . 2 { \pm } 0 . 1$ </td><td> $\mathbf { 9 7 . 3 { \pm 0 . 0 } }$ </td><td> $\mathbf { 8 1 . 7 \pm 0 . 2 }$ </td><td> ${ \bf 9 9 . 0 { \pm } 0 . 1 }$ </td><td> $7 6 . 0 { \pm } 0 . 1 $ </td><td> $\mathbf { 8 7 . 9 2 0 . 0 }$ </td><td> $8 4 . 2 { \pm } 0 . 6 $ </td><td>50.7±0.1</td></tr><tr><td>MiCLA (ours)</td><td> $9 2 . 5 { \pm } 0 . 1 $ </td><td> $9 2 . 5 { \pm } 0 . 1 $ </td><td>96.6±0.1</td><td> $7 0 . 0 { \pm } 0 . 1 $ </td><td> $9 8 . 1 { \pm } 0 . 0 \ $ </td><td> $7 3 . 5 { \pm } 0 . 2 $ </td><td> $8 5 . 6 { \pm } 0 . 7 \ $ </td><td>86.4±0.1</td><td>49.6±0.1</td></tr></table>

![](images/7024ed6a3dd8c8b4c1ae0787f6b91952b456d61bd4f4190f7275dc0c7321039d.jpg)  
MiCLA MiCLA (no mixtures) MiCLA (avg. constituent targets) EAT SSLAM AudioMosaic BAT  
Fig. 3. Joint Semantic Retrieval on ESC-50. The panels show JTA@5 and JTA@10 for waveform and log-mel mixtures across mixing ratios µ. A query is correct only when both constituent classes appear among the Top-K retrieved classes.

## F. CGP Results on Single-Label Datasets

Table V shows the CGP results on single-label classification tasks. MiCLA demonstrates strong transfer across diverse sound domains, ranking first on US8K $( 9 2 . 5 \pm 0 . 1 \% )$ and NSynth $( 8 6 . 4 \pm 0 . 1 \% )$ , while placing second on SPC-2, GTZAN, and Surge. While BAT incorporates specialized architectural and front-end modifications, MiCLA achieves competitive performance across several environmental and musical domains using a standard Vision Transformer backbone, highlighting the efficacy of its mixture-to-constituent pre-training objective.

## G. Joint Semantic Retrieval

Fine-tuning mAP does not directly show whether a mixture representation retains information about both constituents. We conduct Joint Semantic Retrieval on ESC-50 [18] to evaluate whether a frozen CLS representation of a synthetic mixture can retrieve both known constituent classes.

We compare MiCLA with two controlled ablations from Table VII using the same pre-trained checkpoints: (1) “Mi-CLA (no mixtures),” which continues constituent-only training without $\ell _ { \mathrm { m i x } } ;$ and (2) “MiCLA (avg. constituent targets),” a matched baseline that preserves the Stage-2 schedule, synthetic mixtures, and ICR, but replaces directional residual alignment with arithmetic averaging of constituent EMA targets at both patch and clip levels. We also include external self-supervised baselines: EAT [9], SSLAM [13], AudioMosaic [12], and BAT [17]. The ablations isolate the mixture-to-constituent signal, whereas external models provide broader context across different objectives and architectures.

For all models, the encoder is frozen. For each fold and model, we average the $L _ { 2 } .$ -normalized CLS embeddings of all unmixed training clips within each class and then $L _ { 2 ^ { - } }$ normalize the resulting class prototype.

Evaluation uses clips from the held-out partition of each fold. We consider all ${ \binom { 5 0 } { 2 } } \ = \ 1 { , } 2 2 5$ unordered class pairs. For each class pair, we construct 10 mixture queries by pairing held-out clips from the two classes using a randomized ordering. Clip instances may be reused across different class pairs and, when necessary, within a class pair. This procedure yields $1 0 { \binom { 5 0 } { 2 } } = 1 2 { , } 2 5 0$ queries for each fold, model, mixing ratio $\mu ,$ and mixing domain.

For each query, the two clips are mixed with the specified mixing ratio. Waveform mixtures are formed before spectrogram extraction. Log-mel mixtures are formed in prenormalization log-mel space before the remaining front-end processing. We report $\mu$ from 0.05 to 0.50 because larger values are equivalent up to swapping the two constituent clips.

For each mixture, cosine similarities are computed between the mixture CLS token and the 50 class prototypes. We evaluate Joint Top-K Accuracy (JTA@K). A retrieval is counted as correct only when both ground-truth constituent classes are included in the Top-K retrieved classes. For reference, random retrieval in this 50-class setting would achieve $\binom { 5 } { 2 } / \binom { 5 0 } { 2 } \ \approx \ 0 . 8 2 \%$ for JTA@5 and $\binom { 1 0 } { 2 } / \binom { 5 \breve { 0 } } { 2 } \approx 3 . 6 7 \%$ for JTA@10.

Figure 3 shows JTA@5 and JTA@10 across mixing ratios $\mu ;$ each plotted point is averaged over the five folds. MiCLA achieves the highest JTA across mixing ratios, mixture domains, and Top-K settings. All scalar comparisons below use the mean over all reported $\mu$ values after fold averaging.

TABLE VI  
STAGE-1 ABLATION ON CONSTITUENT-CLIP REPRESENTATION STRUCTURING. ALL ROWS USE MASKED SELF-DISTILLATION AND ARE EVALUATED BY AS-20K FINE-TUNING MAP. VALUES ARE THE MEAN ± STANDARD DEVIATION OVER THREE RANDOM SEEDS.
<table><tr><td>Ablation setting</td><td>ICR</td><td>RoPE</td><td>mAP</td><td>Δ</td></tr><tr><td>No ICR</td><td>x</td><td>√</td><td>41.3±0.1</td><td>-1.3</td></tr><tr><td>Absolute PosEmb.</td><td>√</td><td>x</td><td>41.4±0.2</td><td>-1.2</td></tr><tr><td>Full Stage 1</td><td>√</td><td>√</td><td>42.6±0.2</td><td>0.0</td></tr></table>

“ICR” denotes instance-level contrastive regularization over visible-token aggregates. “RoPE” denotes temporal rotary positional embedding. “Absolute PosEmb.” replaces temporal RoPE with absolute positional embedding. $^ { 6 6 } \Delta ^ { , 9 }$ is relative to Full Stage 1.

For waveform mixtures, the averaged-target variant achieves µ-averaged JTA@5/JTA@10 of 11.5%/26.1%, modestly exceeding the no-mixture baseline (10.2%/23.1%). Full Mi-CLA reaches 17.2%/35.3%, improving over averaged targets by 5.7 and 9.2 percentage points, and also exceeding SSLAM (16.7%/33.2%). For log-mel mixtures (the pretraining domain), the gains are larger: averaged targets achieve 14.7%/32.6% (vs. 6.9%/17.8% for no-mixture), while Full MiCLA reaches 19.7%/40.0%, exceeding averaged targets by 5.0 and 7.4 points and performing higher than the evaluated baselines.

Across all four aggregate comparisons, performance consistently improves from the no-mixture baseline to averaged targets and Full MiCLA. This progression indicates that while synthetic-mixture supervision alone aids joint retrieval, directional residual alignment provides clear additional gains. Notably, whereas Full MiCLA and averaged targets differ by only 0.3 mAP in AS-20K fine-tuning, their gap in Joint Semantic Retrieval reaches 5.0–9.2 percentage points. This contrast suggests that standard fine-tuning mAP may not fully expose the constituent-accessibility properties induced by pretraining objectives, which Joint Semantic Retrieval evaluates more directly.

## H. Ablation Study

The ablations test how constituent-side structuring, mixture alignment, alignment granularity, soft clip-level residualization, the mixture construction domain, and the mixing-ratio distribution affect AS-20K fine-tuning mAP. The component ablations report the mean and standard deviation over three random seeds, whereas the mixing-ratio sensitivity analysis in Table IX reports single representative runs.

1) Stage-1 Constituent-Clip Structuring: Table VI analyzes the Stage-1 components. Both ICR and temporal RoPE improve the three-seed mean over their respective variants. With masked self-distillation in every row, removing ICR lowers mAP from $4 2 . 6 \pm 0 . 2$ to $4 1 . 3 \pm 0 . 1$ , while replacing temporal RoPE with absolute positional embedding yields $4 1 . 4 \pm 0 . 2$

2) Stage-2 Mixture-to-Constituent Alignment: Table VII organizes the Stage-2 ablations by design question. Additional optimization alone produces only a minor gain: Stage-1 training achieves 42.6±0.2 mAP, while continuing training without mixtures yields $4 2 . 7 \pm 0 . 1$

To isolate directional residual alignment from mixture supervision itself, we evaluate a matched baseline (“Averaged constituent targets”). This variant shares the Stage-2 schedule, synthetic mixtures, and ICR, but replaces directional residualization with arithmetic averaging of constituent EMA targets at both patch and clip levels. Within this comparison, averaged targets achieve $4 2 . 9 \pm 0 . 1$ mAP (+0.2 over no mixtures), while Full MiCLA reaches $4 3 . 2 \pm 0 . 1$ (+0.3 over averaged targets). In addition to this consistent fine-tuning gain, Section IV-G shows that directional residual alignment provides a clearer separation when directly evaluating joint constituent accessibility.

TABLE VII  
STAGE-2 ABLATION ON SYNTHETIC-MIXTURE SUPERVISION AND MIXTURE-TO-CONSTITUENT ALIGNMENT. ALL MODELS ARE EVALUATED BY AS-20K FINE-TUNING MAP. VALUES ARE THE MEAN ± STANDARD DEVIATION OVER THREE RANDOM SEEDS.
<table><tr><td>Ablation setting</td><td>S2</td><td>Mix</td><td>P</td><td>C</td><td>Soft</td><td>ICR</td><td>mAP</td><td>∆</td></tr><tr><td colspan="9">Comparison of Stage-2 training and mixture supervision</td></tr><tr><td>Stage-1 training</td><td>x</td><td>x</td><td>x</td><td>X</td><td></td><td>√</td><td>42.6±0.2</td><td>-0.6</td></tr><tr><td>Stage-2 no mixtures</td><td>√</td><td>x</td><td>x</td><td>x</td><td>一</td><td>√</td><td> $4 2 . 7 { \pm } 0 . 1$ </td><td>-0.5</td></tr><tr><td>Avg. constituent targets</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td> $4 2 . 9 { \pm } 0 . 1$ </td><td>-0.3</td></tr><tr><td colspan="9">Comparison of alignment granularities</td></tr><tr><td>Patch align. only</td><td>√</td><td>√</td><td>√</td><td>x</td><td>一</td><td>√</td><td>42.9±0.0</td><td>-0.3</td></tr><tr><td>Clip align. only</td><td>√</td><td>√</td><td>x</td><td>√</td><td>√</td><td>√</td><td>43.1±0.0</td><td>-0.1</td></tr><tr><td colspan="9">Comparison of clip-level residualization variants</td></tr><tr><td>Hard clip removal</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td> $4 3 . 0 { \pm } 0 . 1$ </td><td>-0.2</td></tr><tr><td colspan="9">Comparison with and without Stage-2 ICR</td></tr><tr><td>No ICR in Stage 2</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td> $4 2 . 0 { \pm } 0 . 1$ </td><td>-1.2</td></tr><tr><td>Full MiCLA</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td> $4 3 . 2 \pm \mathrm { 0 . 1 }$ </td><td>0.0</td></tr></table>

“S2” denotes Stage 2 training. “Mix” denotes training with synthetic mixtures. “P” and “C” denote patch- and clip-level objectives. “Avg. constituent targets” uses the arithmetic mean of the two constituent EMA targets at both levels, withou residualization; the other mixture variants use directional residual alignment. “Soft” denotes sample-adaptive clip residualization; ✗ denotes hard removal, and “–” indicates not applicable. “ICR” denotes instance-level contrastive regularization. $" \Delta ^ { , * }$ is the difference in mean mAP relative to Full MiCLA.

For alignment granularities, patch-only, clip-only, and joint alignment achieve 42.9 ± 0.0, 43.1 ± 0.0, and $4 3 . 2 \pm 0 . 1$ mAP, respectively, indicating that localized and global residual alignments provide complementary benefits to reach the highest performance in this comparison. Similarly, sample-adaptive soft residualization achieves 43.2 ± 0.1 mAP compared with $4 3 . 0 \pm 0 . 1$ for hard removal, consistent with the intuition that adaptively retaining shared global context prevents oversuppression of common acoustic information. Finally, removing Stage-2 ICR drops mAP to $4 2 . 0 \pm 0 . 1$ , indicating that preserving constituent-side representation structure plays an important role during mixture alignment.

3) Mixture Construction Domain: Table VIII compares waveform, linear-mel, and log-mel domains for constructing synthetic mixtures while keeping the Stage-2 objective fixed. Waveform mixing interpolates the two raw waveforms before front-end extraction, linear-mel mixing interpolates mel spectrograms before logarithmic compression, and log-mel mixing interpolates log-mel spectrograms as in Eq. (4). Logmel mixing yields the highest AS-20K fine-tuning mAP. This trend is consistent with the interpretation that log-mel interpolation provides a cleaner correspondence signal for alignment: it avoids the phase-dependent interference introduced by waveform mixing and, unlike linear-mel mixing, performs interpolation after dynamic-range compression, which may reduce the dominance of high-amplitude regions.

TABLE VIII  
ABLATION ON THE MIXTURE CONSTRUCTION DOMAIN. VALUES ARE THREE-SEED AS-20K FINE-TUNING MAP REPORTED AS MEAN ± STANDARD DEVIATION.
<table><tr><td>Setting</td><td>Waveform Linear-mel Log-mel</td><td></td><td></td></tr><tr><td>mAP (%)</td><td> $4 2 . 9 { \pm } 0 . 0 $ </td><td> $4 2 . 8 { \pm } 0 . 1$ </td><td> $4 3 . 2 { \pm } 0 . 1 $ </td></tr></table>

TABLE IX

SENSITIVITY TO THE MIXING-RATIO DISTRIBUTION. VALUES AREAS-20K FINE-TUNING MAP FROM SINGLE REPRESENTATIVE RUNS.“FIXED” DENOTES DETERMINISTIC MIXING WITH $\mu = 0 . 5 ,$ WHEREAS ηDENOTES THE BETA DISTRIBUTION PARAMETER IN $\mu \sim$ Beta(η, η).
<table><tr><td>Setting</td><td>Fixed</td><td> $\eta = 1$ </td><td> $\eta = 2$ </td><td> $\eta = 4$ </td><td> $\eta = 6$ </td><td> $\eta = 8$ </td></tr><tr><td>mAP (%)</td><td>42.9</td><td>43.0</td><td>43.1</td><td>43.3</td><td>43.2</td><td>43.1</td></tr></table>

4) Mixing-Ratio Sensitivity: Performance varies little across the tested mixing-ratio distributions. Table IX shows AS-20K fine-tuning mAP for $\eta ~ \in ~ \{ 1 , 2 , 4 , 6 , 8 \}$ , together with a deterministic mixing baseline using fixed $\mu \ : = \ : 0 . 5$ The highest single-run score occurs at $\eta \ : = \ : 4 .$ , while fixed $\mu ~ = ~ 0 . 5$ also reaches 42.9% mAP. These results indicate that pre-training remains effective across various mixing-ratio distributions, supporting Beta(4, 4) as a reliable default.

## V. CONCLUSION

We presented MiCLA, a two-stage self-supervised pretraining method that uses synthetic mixture–constituent correspondence for polyphonic audio representation learning. Stage 1 structures single-clip representations through masked self-distillation and contrastive regularization, while Stage 2 adds bidirectional alignment by residualizing a mixture against one constituent and aligning the residual with the other. MiCLA applies hard projective residualization to local patch tokens and sample-adaptive soft residualization to the global class token.

Downstream evaluations show that MiCLA is particularly competitive in polyphonic multi-label settings, achieving 43.2% mAP on AS-20K and 50.3% on AS-2M under fine-tuning, as well as competitive scores on both AudioSet benchmarks under frozen Convex Gated Probing. In Joint Semantic Retrieval, MiCLA improves mixing-ratio-averaged (µ-averaged) JTA by 5.0–9.2 percentage points over matched target-averaging supervision. This retrieval gap, accompanied by consistent fine-tuning performance, indicates that direct retrieval diagnostics are particularly sensitive to the constituentretention properties targeted by residual alignment. Ablation studies support the design of the integrated MiCLA formulation, where dual-granularity alignment and constituentstructure preservation jointly contribute to downstream transfer.

MiCLA focuses on representation learning from pairwise log-mel mixtures rather than explicit source separation. Future work includes evaluating on unconstrained real-world mixtures, unknown acoustic events, and mixtures with more than two sources. Overall, our findings indicate that leveraging mixture–constituent correspondence is a promising direction for learning rich representations from polyphonic audio.

## APPENDIX A

## DERIVATION OF SAMPLE-ADAPTIVE CLIP-LEVEL SOFT RESIDUALIZATION

We derive the closed-form solution for the sample-adaptive suppression coefficient $\alpha _ { a | b } ^ { \star }$ in Eqs. (7) and (8). Let ${ \bf z } _ { \mathrm { m i x } } , { \bf z } _ { a } ,$ and $\mathbf { z } _ { b } \in \mathbb { R } ^ { D }$ denote the student CLS representations of the mixture, target constituent $\mathbf { x } _ { a } ,$ , and counterpart constituent $\mathbf { x } _ { b } .$ respectively.

Consider a linear subtraction of the counterpart representation from the mixture, ${ \bf z } _ { \mathrm { m i x } } - k { \bf z } _ { b }$ , parameterized by a scalar coefficient $k \in \mathbb { R }$ . To find the unconstrained coefficient $k _ { a \mid b } ^ { \star }$ that minimizes the squared Euclidean distance to the target representation ${ \bf z } _ { a }$ , we set up the objective:

$$
J ( k ) = \left\| \mathbf { z } _ { \operatorname* { m i x } } - k \mathbf { z } _ { b } - \mathbf { z } _ { a } \right\| _ { 2 } ^ { 2 } .\tag{12}
$$

Differentiating $J ( k )$ with respect to k yields $\mathrm { d } J ( k ) / \mathrm { d } k =$ $- 2 \mathbf { z } _ { b } ^ { \top } ( \mathbf { z } _ { \operatorname* { m i x } } - k \mathbf { z } _ { b } - \mathbf { z } _ { a } )$ . Setting the derivative to zero, the unconstrained minimizer satisfies $( { \bf z } _ { \mathrm { m i x } } - k { \bf z } _ { b } - { \bf z } _ { a } ) \perp { \bf z } _ { b } $ yielding:

$$
k _ { a | b } ^ { \star } = \frac { ( \mathbf { z } _ { \mathrm { m i x } } - \mathbf { z } _ { a } ) ^ { \top } \mathbf { z } _ { b } } { \| \mathbf { z } _ { b } \| _ { 2 } ^ { 2 } } .\tag{13}
$$

Recall from Eq. (5) that the full orthogonal projection of $\mathbf { z } _ { \mathrm { m i x } }$ onto $\mathbf { z } _ { b }$ corresponds to subtracting $k _ { a | b } \mathbf { z } _ { b }$ with

$$
k _ { a | b } = \frac { \mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { z } _ { b } } { \| \mathbf { z } _ { b } \| _ { 2 } ^ { 2 } } .\tag{14}
$$

When the residualization operator is expressed as $\rho _ { \alpha } ( \mathbf { z } _ { \operatorname* { m i x } } , \mathbf { z } _ { b } )$ $= \mathbf { z } _ { \mathrm { m i x } } - \alpha k _ { a | b } \mathbf { z } _ { b }$ , the unconstrained optimal suppression ratio becomes $\alpha = k _ { a | b } ^ { \star } / k _ { a | b }$ . Because both coefficients share the denominator $\| \mathbf { z } _ { b } \| _ { 2 } ^ { 2 } .$ , their ratio simplifies directly to:

$$
\frac { k _ { a | b } ^ { \star } } { k _ { a | b } } = \frac { ( \mathbf { z } _ { \operatorname* { m i x } } - \mathbf { z } _ { a } ) ^ { \top } \mathbf { z } _ { b } } { \mathbf { z } _ { \operatorname* { m i x } } ^ { \top } \mathbf { z } _ { b } } = 1 - \frac { \mathbf { z } _ { a } ^ { \top } \mathbf { z } _ { b } } { \mathbf { z } _ { \operatorname* { m i x } } ^ { \top } \mathbf { z } _ { b } } .\tag{15}
$$

Constraining α to the interval [0, 1] via clipping and applying stop-gradient sg(·) yields the constrained solution in Eq. (8):

$$
\boldsymbol { \alpha } _ { a | b } ^ { \star } = \mathrm { s g } \left( \mathrm { c l i p } _ { [ 0 , 1 ] } \left( 1 - \frac { \mathbf { z } _ { a } ^ { \top } \mathbf { z } _ { b } } { \mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { z } _ { b } } \right) \right) .\tag{16}
$$

As described in Section IV-C, the denominator $\mathbf { z } _ { \mathrm { m i x } } ^ { \top } \mathbf { z } _ { b }$ is stabilized with a small sign-preserving offset $( 1 0 ^ { - 6 } )$ to prevent numerical instability.

## APPENDIX B

## SUPPLEMENTARY LINEAR PROBING ON MULTI-LABEL DATASETS

We additionally conduct simple linear probing on the multilabel settings as a supplementary diagnostic of linear accessibility. This diagnostic is not used as the primary frozenprobing protocol in this paper. For all models, the encoder is frozen and only a linear classifier is trained. The classifier input is constructed by concatenating the final-layer CLS token with the average of all final-layer patch tokens. Unlike

TABLE X  
SUPPLEMENTARY LINEAR PROBING RESULTS ON MULTI-LABEL SETTINGS.
<table><tr><td>Model</td><td>AS-2M mAP (%)</td><td>AS-20K mAP (%)</td><td>FSD-50K mAP (%)</td></tr><tr><td>EAT [9]</td><td>35.2</td><td>29.1</td><td>50.3</td></tr><tr><td>SSLAM [13]</td><td>33.2</td><td>27.4</td><td>47.2</td></tr><tr><td>AudioMosaic [12]</td><td>36.2</td><td>30.9</td><td>47.6</td></tr><tr><td>BAT [17]</td><td>37.4</td><td>32.5</td><td>55.8</td></tr><tr><td>MiCLA (ours)</td><td>34.7</td><td>29.5</td><td>52.1</td></tr></table>

CGP, this probe does not use intermediate-layer aggregation or prototype-based readout.

Table X reports the supplementary linear probing results. Under this simplified linear readout, MiCLA outperforms SSLAM across all three benchmarks (+1.5 to +4.9 mAP) and performs comparably to or higher than EAT on AS-20K and FSD-50K, while models with specialized front-ends or attention mechanisms, such as BAT, achieve higher performance under this protocol. Comparing these results with the CGP evaluations in Table IV suggests that a single final-layer linear classifier captures only part of the learned representations, and that multi-layer probing architectures can better access constituent-relevant information distributed across intermediate representations.

## REFERENCES

[1] Y. Gong, C.-I. Lai, Y.-A. Chung, and J. Glass, “SSAST: Self-Supervised Audio Spectrogram Transformer,” in AAAI Conference on Artificial Intelligence (AAAI), vol. 36, no. 10, 2022, pp. 10 699–10 709.

[2] A. Baade, P. Peng, and D. Harwath, “MAE-AST: Masked Autoencoding Audio Spectrogram Transformer,” in Annual Conference of the International Speech Communication Association (INTERSPEECH), 2022, pp. 2438–2442.

[3] P.-Y. Huang, H. Xu, J. Li, A. Baevski, M. Auli, W. Galuba, F. Metze, and C. Feichtenhofer, “Masked autoencoders that listen,” in Advances in Neural Information Processing Systems (NeurIPS). Red Hook, NY, USA: Curran Associates Inc., 2022.

[4] D. Chong, H. Wang, P. Zhou, and Q. Zeng, “Masked spectrogram prediction for self-supervised audio pre-training,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[5] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, W. Che, X. Yu, and F. Wei, “BEATs: audio pre-training with acoustic tokenizers,” in International Conference on Machine Learning (ICML), 2023.

[6] A. Baevski, W.-N. Hsu, Q. Xu, A. Babu, J. Gu, and M. Auli, “data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language,” in International Conference on Machine Learning (ICML), 2022, pp. 1298–1312.

[7] A. Baevski, A. Babu, W.-N. Hsu, and M. Auli, “Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language,” in International Conference on Machine Learning (ICML), 2023, pp. 1416–1429.

[8] X. Li, N. Shao, and X. Li, “Self-supervised audio teacher-student transformer for both clip-level and frame-level tasks,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 1336– 1351, 2024.

[9] W. Chen, Y. Liang, Z. Ma, Z. Zheng, and X. Chen, “Eat: Self-supervised pre-training with efficient audio transformer,” in International Joint Conference on Artificial Intelligence (IJCAI), K. Larson, Ed. IJCAI, 8 2024, pp. 3807–3815, main Track.

[10] S. Ghosh, A. Seth, S. Umesh, and D. Manocha, “MAST: Multiscale Audio Spectrogram Transformers,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2023, pp. 1–5.

[11] A. Seth, S. Ghosh, S. Umesh, and D. Manocha, “SLICER: Learning Universal Audio Representations Using Low-Resource Self-Supervised Pre-Training,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2023, pp. 1–5.

[12] H. Huang, Q. Wang, X. Ma, C. Xie, C. Leckie, and S. Erfani, “Audiomosaic: Contrastive masked audio representation learning,” in International Conference on Machine Learning (ICML), 2026.

[13] T. Alex, S. Atito, A. Mustafa, M. Awais, and P. J. B. Jackson, “SSLAM: Enhancing self-supervised models with audio mixtures for polyphonic soundscapes,” in International Conference on Learning Representations (ICLR), 2025.

[14] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations (ICLR), 2021.

[15] K. Yamamoto and K. Okusa, “Aasp: Aliasing-aware self-supervised pretraining for audio spectrogram transformers,” IEEE Transactions on Audio, Speech and Language Processing, vol. 34, pp. 2815–2830, 2026.

[16] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “Audio set: An ontology and humanlabeled dataset for audio events,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). IEEE, 2017, pp. 776–780.

[17] H. Ghaffari, L. Rauch, C. Scholz, and P. Devos, “Bat: Better audio transformer guided by convex gated probing,” in International Conference on Machine Learning (ICML), 2026.

[18] K. J. Piczak, “Esc: Dataset for environmental sound classification,” in ACM International Conference on Multimedia (ACM MM), ser. MM ’15. New York, NY, USA: Association for Computing Machinery, 2015, pp. 1015–1018.

[19] Y. Gong, Y.-A. Chung, and J. Glass, “AST: Audio Spectrogram Transformer,” in Annual Conference of the International Speech Communication Association (INTERSPEECH), 2021, pp. 571–575.

[20] A. Seth, R. Selvakumar, S. Sakshi, S. Kumar, S. Ghosh, and D. Manocha, “EH-MAM: Easy-to-Hard Masked Acoustic Modeling for Self-Supervised Speech Representation Learning,” in Empirical Methods in Natural Language Processing (EMNLP), 2024, pp. 6386–6400.

[21] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “Masked modeling duo: Towards a universal audio pre-training framework,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 2391–2406, 2024.

[22] S. A. A. Ahmed, M. Awais, W. Wang, M. D. Plumbley, and J. Kittler, “Asit: Local-global audio spectrogram vision transformer for event classification,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 3684–3693, 2024.

[23] J. Wang, T. Wang, M. Ge, L. Wang, and J. Dang, “ASDA: Audio Spectrogram Differential Attention Mechanism for Self-Supervised Representation Learning,” in Annual Conference of the International Speech Communication Association (INTERSPEECH), 2025, pp. 5803–5807.

[24] A. Quelennec, P. Chouteau, G. Peeters, and S. Essid, “Masked latent prediction and classification for self-supervised audio representation learning,” in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2025, pp. 1–5.

[25] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, “mixup: Beyond empirical risk minimization,” in International Conference on Learning Representations (ICLR), 2018.

[26] D. Yu, M. Kolbæk, Z.-H. Tan, and J. Jensen, “Permutation invariant training of deep models for speaker-independent multi-talker speech separation,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 241–245.

[27] S. Wisdom, E. Tzinis, H. Erdogan, R. Weiss, K. Wilson, and J. Hershey, “Unsupervised sound separation using mixture invariant training,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33. Curran Associates, Inc., 2020, pp. 3846–3857.

[28] J. Liu, X. Huang, J. Zheng, Y. Liu, and H. Li, “Mixmae: Mixed and masked autoencoder for efficient pretraining of hierarchical vision transformers,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6252–6261.

[29] K. Chen, Z. Liu, L. Hong, H. Xu, Z. Li, and D.-Y. Yeung, “Mixed autoencoder for self-supervised visual representation learning,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 22 742–22 751.

[30] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra, “Fsd50k: An open dataset of human-labeled sound events,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 829–852, Dec. 2021.

[31] J. Salamon, C. Jacoby, and J. P. Bello, “A dataset and taxonomy for urban sound research,” in ACM International Conference on Multimedia (ACM MM), 2014, pp. 1041–1044.

[32] P. Warden, “Speech commands: A dataset for limited-vocabulary speech recognition,” arXiv:1804.03209, 2018.

[33] A. Nagrani, J. S. Chung, and A. Zisserman, “Voxceleb: A large-scale speaker identification dataset,” in Annual Conference ofthe International Speech Communication Association (INTERSPEECH), 2017, pp. 2616– 2620.

[34] K. MacLean, Voxforge, 2018, available at http://www.voxforge.org.

[35] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “Crema-d: Crowd-sourced emotional multimodal actors dataset,” IEEE Transactions on Affective Computing, vol. 5, no. 4, pp. 377–390, 2014.

[36] G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293–302, 2002.

[37] J. Engel, C. Resnick, A. Roberts, S. Dieleman, M. Norouzi, D. Eck, and K. Simonyan, “Neural audio synthesis of musical notes with wavenet autoencoders,” in International Conference on Machine Learning (ICML). PMLR, 2017, pp. 1068–1077.

[38] J. Turian, J. Shier, G. Tzanetakis, K. McNally, and M. Henry, “One billion audio sounds from gpu-enabled modular synthesis,” in International Conference on Digital Audio Effects (DAFx). IEEE, 2021, pp. 222–229.

[39] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations (ICLR), 2019.

[40] ——, “Sgdr: Stochastic gradient descent with warm restarts,” in International Conference on Learning Representations (ICLR), 2017.

[41] H. Bao, L. Dong, S. Piao, and F. Wei, “Beit: Bert pre-training of image transformers,” arXiv:2106.08254, 2021.

[42] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, “Deep networks with stochastic depth,” in European Conference on Computer Vision (ECCV), 2016.

[43] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “Roformer: Enhanced transformer with rotary position embedding,” Neurocomput., vol. 568, no. C, Feb. 2024.

[44] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, “YaRN: Efficient context window extension of large language models,” in International Conference on Learning Representations (ICLR), 2024.

[45] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, “Specaugment: A simple data augmentation method for automatic speech recognition,” in Annual Conference ofthe International Speech Communication Association (INTERSPEECH), 2019.

[46] T. Ridnik, E. Ben-Baruch, N. Zamir, A. Noy, I. Friedman, M. Protter, and L. Zelnik-Manor, “Asymmetric loss for multi-label classification,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 82–91.