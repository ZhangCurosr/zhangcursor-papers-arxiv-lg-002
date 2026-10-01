# Predicting Multi-View Rashomon Representation: Can We Learn Where Models Disagree?

Mingyue Ma<sup>1</sup> Zongbo Han<sup>1∗</sup> Changqing Zhang<sup>2</sup> Guangyu Wang<sup>1</sup>

<sup>1</sup> Beijing University of Posts and Telecommunications <sup>2</sup> Tianjin University

## Abstract

Foundation models are increasingly adopted across a wide range of applications, often serving as core blocks within AI systems. Yet different foundation models may encode the same input from multiple different views, leading to substantial representation disagreement, which we term Rashomon Representation. Such disagreement often signals inputs that a given model encodes in a way inconsistent with other models, offering a valuable yet underexplored signal for input reliability estimation. While prior work has largely focused on measuring disagreement across multiple models with a representation set, we instead focus on predicting disagreement from a single representation. We hypothesize that this disagreement follows some consistent, input-dependent patterns rather than occurring at random. To test this, we quantify disagreement by comparing each sample’s nearest neighbors across different models’ representation spaces, then train a lightweight predictor that estimates disagreement from a single model’s representation. At inference time, given a new input, the predictor uses that input’s representation to tell whether it aligns with or diverges from those of other models. Extensive experiments across diverse foundation models and datasets show that representational disagreement is indeed input-dependent, predictable, and generalizable, enabling efficient reliability estimation of foundation models.

## 1 Introduction

“The same event, told by different witnesses, becomes different events entirely.”

The rapid development of foundation models has enabled powerful and generalizable capabilities across a wide range of domains and applications [Awais et al., 2025]. Models such as CLIP [Radford et al., 2021], DINOv2 [Oquab et al., 2024] have demonstrated remarkable transferability, establishing a “pretrain once, adapt everywhere” paradigm. Despite their shared versatility, however, these models can learn fundamentally different representations. Differences in training objectives, architectures, and data distributions can lead them to encode the same input in substantially different ways [Ciernik et al., 2025, Klabunde et al., 2025]. For example, given an image, one model may focus more on high-level semantic attributes, while another may emphasize structural patterns [Barsellotti et al., 2025, Zhang and Tan, 2025]. As shown in Fig. 1, different foundation models can produce distinct representations of the same input despite capturing similarly — Rashomon, Akira Kurosawa, 1950

![](images/4b240676ce8fbeac6b8163fdf52dd2c0177048a4700517cb57257afa5cfb99ec.jpg)  
Figure 1: Illustration of Rashomon Representation: different foundation models can produce distinct representations of the same input, reflecting complementary views of its underlying information.

meaningful information. We refer to this phenomenon as the Rashomon Representation. Such discrepancies reflect representation-level uncertainty that can affect downstream reliability [Park et al., 2024]. Understanding when they arise is therefore important for building robust and reliable AI systems.

Prior studies have largely focused on measuring representation differences through direct comparisons between representation spaces, at either the dataset or sample level. At the dataset level, methods such as RSA, SVCCA, and CKA measure global similarity between representation spaces [Kriegeskorte et al., 2008, Raghu et al., 2017, Kornblith et al., 2019]. Related studies further examine whether representations learned by different models converge toward similar structures [Li et al., 2015, Huh et al., 2024]. Such analyses provide an aggregate view of representation similarity, but do not directly capture how cross-model disagreement varies across individual inputs [Kolling et al., 2025]. Recent sample-level approaches address this limitation by quantifying representational agreement for individual inputs. For example, PNKA compares an input’s similarity profiles across two representation spaces, while neighborhood consistency measures the overlap of its nearest-neighbor sets across multiple pretrained models [Kolling et al., 2025, Park et al., 2024]. In essence, prior methods ask a measurement question: given representations, how much do models disagree on a particular input? We instead ask a prediction question: can such Rashomon Representation, i.e., cross-model representational disagreement, be predicted from a representation?

We introduce a supervised framework for sample-level prediction of rashomon representation across foundation models. Specifically, we introduce a neighborhood-consistency measure to quantify inter-model representational disagreement. The resulting agreement scores then serve as supervision for training a lightweight predictor that predicts rashomon representation directly from the represen tation produced by a single foundation model. Then, we systematically evaluate the predictability of rashomon representation across diverse experimental settings and find that it is consistently predictable from individual model representations. The observed predictability reveals three properties of rashomon representation. First, the degree of disagreement varies substantially across different inputs. Second, this variation is not random, but follows systematic and learnable patterns. Third, these patterns can be identified from the representation of a single foundation model, indicating that cross-model disagreement is partially encoded in individual representations. Together, these findings provide a practical mechanism for anticipating representation-level uncertainty at inference time. Our contributions are as follows:

• We identify and formalize Rashomon Representation, the phenomenon that different foundation models may encode the same input in substantially different ways, and formulate a new task of Rashomon Representation prediction to anticipate such cross-model disagreement from the representation of a single model.

• We develop a practical framework for this task. Specifically, we quantify sample-level disagreement using cross-model nearest-neighbor consistency over a reference dataset and train a lightweight predictor to estimate this disagreement from representations.

• Through extensive experiments, we show that representational disagreement is predictable, and generalizable, and further demonstrate its practical utility for efficiently identifying samples that are potentially vulnerable to unreliable representations across foundation models.

## 2 Methods

We investigate whether Rashomon Representation, referring to sample-level representational disagreement across foundation models, can be predicted from the representation of a single foundation model. To this end, we formulate its prediction as a supervised learning problem. We first quantify sample-level cross-model representational disagreement by comparing local neighborhoods across the representation spaces of a pool of foundation models, yielding a disagreement score for each sample. These scores are then used as supervision to train a lightweight predictor that predicts agreement scores from representations produced by a designated anchor model. Once trained, the predictor can estimate the expected disagreement of a new sample using only its anchor-model representation at inference time.

![](images/20112f75fcd6d5abeec2b2f4837471e34acb1c070f2ee07a2c217a2e3ca71163.jpg)  
Figure 2: Training framework for predicting inter-model representational disagreement. We measure neighborhood agreement between an anchor model and comparison models over a shared reference set to obtain the supervision signal $y ( x )$ . A lightweight predictor $g _ { \phi }$ learns to approximate $y ( x )$ using only the anchor-model representation by minimizing $\bar { \mathcal { L } } ( \phi )$ . All foundation models remain frozen, and only the predictor parameters are optimized.

## 2.1 Quantifying Representational Disagreement

Setup and motivation. Given an input sample x and a pool of M pretrained foundation models $\mathcal { H } = \overline { { \{ h _ { 0 } , h _ { 1 } , . . . , h _ { M - 1 } \} } }$ , our goal is to predict, using only the representation $h _ { 0 } ( x )$ produced by an anchor model $h _ { 0 }$ , how consistently x is represented across the model pool. We denote the remaining models by $\mathcal { H } _ { - 0 } = \mathcal { H } \backslash \{ h _ { 0 } \}$ and use them to characterize cross-model representational agreement. Directly comparing $h _ { 0 } ( x )$ with $\{ h _ { m } ( x ) \} _ { m = 1 } ^ { M - 1 }$ is generally not meaningful, since independently trained models encode samples in model-specific representation spaces that are not directly comparable. Rather than aligning these spaces explicitly, we compare each sample’s local neighborhoods across models over the same reference samples, providing a parameter-free comparison that requires neither additional training nor assumptions about cross-model representation alignment. Specifically, for each model, we retrieve the local neighborhood of x from the same reference set and measure cross-model agreement by the overlap between these neighborhoods. This alignment-free consistency score is then used as supervision for our predictor, with lower scores indicating stronger representational disagreement.

Model-specific neighborhood construction. Let $\mathcal { D } = \{ x _ { 1 } , \ldots , x _ { N } \}$ denote a reference dataset shared across all models. For each model $h _ { m } ,$ , we characterize how the sample x is locally positioned in the representation space by identifying its k nearest neighbors in $\mathcal { D } .$ Our construction only requires a model-specific similarity measure to rank the reference samples and does not depend on a particular choice of metric. In our implementation, we use cosine similarity, a scale-invariant measure commonly used for comparing learned representations. Equivalently, after $\ell _ { 2 }$ normalization, the similarity between x and a reference sample $x _ { j } \in \mathcal { D }$ is given by

$$
s _ { m } ( x , x _ { j } ) = \frac { h _ { m } ( x ) ^ { \top } h _ { m } ( x _ { j } ) } { \| h _ { m } ( x ) \| _ { 2 } \| h _ { m } ( x _ { j } ) \| _ { 2 } } .
$$

We then define $\mathcal { N } _ { k } ^ { ( m ) } ( x )$ as the index set of the k nearest neighbors with the largest similarity scores:

$$
\begin{array} { r } { \mathcal { N } _ { k } ^ { ( m ) } ( x ) = \mathrm { T o p K } _ { j \in \{ 1 , . . . , N \} } s _ { m } ( x , x _ { j } ) . } \end{array}
$$

Although the resulting neighborhoods are constructed independently in model-specific representation spaces, they are all expressed over the same set of reference indices. Thus, the neighborhoods can be compared directly across models without explicitly aligning their representation spaces.

Cross-model neighborhood agreement. Based on model-specific neighborhoods, we quantify how consistently the anchor model $h _ { 0 }$ represents x relative to the remaining models in $\mathcal { H } _ { - 0 }$ . We define

$$
\mathrm { N A } _ { k } ( x ; h _ { 0 } , \mathcal { H } ) = \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M - 1 } \frac { | \mathcal { N } _ { k } ^ { ( 0 ) } ( x ) \cap \mathcal { N } _ { k } ^ { ( m ) } ( x ) | } { k } .\tag{1}
$$

where $| \mathcal { N } _ { k } ^ { ( 0 ) } ( x ) \cap \mathcal { N } _ { k } ^ { ( m ) } ( x ) |$ denotes the number of k-nearest neighbors shared by the anchor model $h _ { 0 }$ and comparison model $h _ { m } . \mathrm { N A } _ { k }$ measures the neighborhood agreement of x at neighborhood size $k ,$ quantifying how consistently the local neighborhood identified by the anchor model is preserved across the comparison models. For each comparison model $h _ { m }$ , the corresponding term measures the fraction of reference neighbors selected by the anchor model that are also selected by $h _ { m }$ . Averaging these pairwise overlaps over $\mathcal { H } _ { - 0 }$ yields a sample-level measure of representational consistency relative to $h _ { 0 }$ . The score ranges from 0 to 1, with higher values indicating greater agreement in local neighborhood structure across models and lower values indicating stronger representational disagreement.

Neighborhood agreement as supervision. Since $\mathrm { N A } _ { k }$ is defined solely through neighborhood overlap, its computation requires no additional model training and does not require other assumptions. It is therefore model-agnostic and can be applied to a broad range of representation models, as long as their representations can be used to construct local neighborhoods. These properties make $\mathrm { N A } _ { k }$ a simple and broadly applicable measure, and we use $\mathrm { N A } _ { k }$ as the sample-level supervision signal for predicting cross-model representational consistency from $h _ { 0 } ( x )$ alone.

## 2.2 From Measurement to Prediction

Predicting cross-model agreement. The neighborhood agreement score introduced above quantifies, for each sample, the degree of Rashomon Representation, i.e., how consistently the same input is represented across different foundation models. However, explicitly computing this score requires access to multiple models and a reference dataset at inference time, limiting its practical applicability. This raises a more fundamental question: can the Rashomon Representation be inferred from its representation in a single model? To test this, we use the measured consistency scores as prediction targets and train a lightweight predictor whose only input is the representation produced by an anchor model. Successful prediction would indicate that rashomon representation is already reflected, at least partially, in the representation space of an individual model, rather than being a property that becomes observable only after comparing multiple models.

Prediction formulation. For an input $x ,$ let $\tilde { z } _ { 0 } ( x ) \ = \ h _ { 0 } ( x ) / \lVert h _ { 0 } ( x ) \rVert _ { 2 } \ \in \ \mathbb { R } ^ { d _ { 0 } }$ denote the $\ell _ { 2 } \cdot$ normalized representation produced by the anchor model, where $d _ { 0 }$ is its output dimensionality. We introduce a predictor $g _ { \phi } : \mathbb { R } ^ { d _ { 0 } }  ( 0 , 1 )$ that produces $\hat { y } ( x ) = g _ { \phi } ( \tilde { z } _ { 0 } ( x ) )$ to approximate the target $y ( x ) = \mathrm { N A } _ { k } ( x ; h _ { 0 } , \mathcal { H } )$ . Higher predicted values indicate stronger expected agreement between the anchor and comparison models, while lower values indicate greater representational disagreement. Crucially, the predictor takes only $\tilde { z } _ { 0 } ( x )$ as input, without access to the comparisonmodel representations or the cross-model neighborhood comparisons used to define the target score. It therefore learns to predict cross-model consistency solely from the representation. If a lightweight predictor can do so reliably, this indicates that cross-model agreement is systematically associated with information already present in the representation.

Predictor construction. We instantiate $g _ { \phi }$ as a lightweight residual feed-forward network that maps the normalized anchor representation to a scalar consistency estimate. The input is first projected into a hidden space and then processed by residual blocks composed of LayerNorm, GELU activations, dropout, and identity skip connections before the final prediction layer. Following standard residual feed-forward designs [Vaswani et al., 2017, Tolstikhin et al., 2021], we keep the predictor deliberately small and capacity-controlled so that its performance reflects how readily cross-model consistency can be predicted from a single representation, rather than the fitting power of a high-capacity auxiliary model. We additionally evaluate alternative predictor architectures in the experiments to examine whether the predictability depends on a particular architectural choice.

Predictor training. We use the predefined, disjoint training, validation, and test splits, denoted by $\mathcal { D } _ { \mathrm { t r } } , \mathcal { D } _ { \mathrm { v a l } } ,$ and $\mathcal { D } _ { \mathrm { t e } } .$ , respectively, and set $\mathcal { D } _ { \mathrm { r e f } } = \mathcal { D } _ { \mathrm { t r } }$ . For each sample $x ,$ we precompute the target $y ( x ) = \mathrm { N A } _ { k } ( x ; h _ { 0 } , \mathcal { H } )$ using Eq. 1, excluding the query itself for samples in ${ \mathcal { D } } _ { \mathrm { t r } }$ . The predictor is

trained on ${ \mathcal { D } } _ { \mathrm { t r } }$ by minimizing

$$
\mathcal { L } ( \phi ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { t r } } | } \sum _ { x _ { i } \in \mathcal { D } _ { \mathrm { t r } } } \ell ( g _ { \phi } ( \tilde { z } _ { 0 } ( x _ { i } ) ) , y ( x _ { i } ) ) ,
$$

where ℓ denotes the regression loss. The checkpoint with the lowest validation loss on $\mathcal { D } _ { \mathrm { v a l } }$ is selected and finally evaluated on $\mathcal { D } _ { \mathrm { t e } } .$ . All foundation models remain fixed, and only the predictor parameters ϕ are optimized.

## 3 Related Work

Representational Similarity. A central question in representation analysis is whether different models organize the same inputs in similar ways. Existing methods typically compare the similarity structures induced by different models on a shared dataset. Representational similarity analysis (RSA) measures the correspondence between these structures [Kriegeskorte et al., 2008], while SVCCA and CKA provide more systematic tools for comparing representations across layers, architectures, and training procedures [Raghu et al., 2017, Kornblith et al., 2019]. Related studies further investigate the convergence of independently trained representations and the effects of architectural and pretraining choices [Li et al., 2015, Nguyen et al., 2021, Raghu et al., 2021, Neyshabur et al., 2020, Xie et al., 2023]. These approaches primarily produce retrospective summaries at the model, layer, or dataset level. More recent work extends representation comparison to individual inputs. For example, PNKA compares an input’s similarity profiles across representation spaces, while neighborhood-based methods measure agreement through the overlap of nearest-neighbor sets [Park et al., 2024, Kolling et al., 2025]. Such methods capture input-dependent variation more directly, but remain measurement oriented: they require representations from multiple models to quantify the disagreement of a given input. In contrast, our work reformulates this problem as prediction. We ask whether cross-model representational disagreement, which we term Rashomon Representation, can be inferred from the representation of a single model.

Vision Foundation Models are commonly developed with different pretraining objectives and architectures. Image-language contrastive learning, as used in CLIP, aligns visual representations with natural language [Radford et al., 2021]. Self-supervised methods adopt contrastive, momentumbased, bootstrapping, or self-distillation objectives, including SimCLR, MoCo, BYOL, DINO, and DINOv2 [Chen et al., 2020, He et al., 2020, Grill et al., 2020, Caron et al., 2021, Oquab et al., 2024]. Masked-image modeling methods, such as BEiT and MAE, learn representations by recovering masked visual content [Bao et al., 2022, He et al., 2022]. In terms of architecture, Vision Transformers provide a scalable alternative to convolutional visual encoders [Dosovitskiy et al., 2021]. In our experiments, we consider a broader range of vision foundation models. Taking vision foundation models as a representative domain, we conduct extensive experiments to investigate the predictability and generality of cross-model representational disagreement.

## 4 Experiments

Our experiments are designed to test the central hypothesis underlying Rashomon Representation: inter-model representational disagreement is not random, but follows input-dependent patterns that can be predicted from a single model representation. We organize our investigation around two aspects. First, we establish predictability from a single representation and examine whether it persists across model configurations, distribution shifts, and predictor architectures, as well as when the anchor model is excluded from the disagreement target. Second, we assess the practical value of predicted disagreement by examining its association with downstream classification risk and the reductions in computation and reference data required to estimate disagreement.

Experimental setup. Unless otherwise specified, we use disjoint ImageNet-1K [Deng et al., 2009, Russakovsky et al., 2015] splits for predictor training, validation, and testing, with the full training split as the neighborhood reference set. Our default configuration uses CLIP ViT-B/32 [Radford et al., 2021] as the anchor model, with DINOv2 ViT-B/14 [Oquab et al., 2024] and a supervised ViT-B/16 (IN-21K) [Dosovitskiy et al., 2021] as comparison models. Below, we refer to these models as CLIP, DINOv2, and ViT-21K, respectively. We use a lightweight Residual FFN predictor built around residual feed-forward blocks and set the neighborhood size to k = 50 when computing the neighborhood agreement target $\mathrm { N A } _ { k }$ . In this default setting, the predictor estimates neighborhood agreement using only the anchor representation, with agreement serving as an operational proxy for inter-model representational disagreement. To address different experimental questions, we vary model combinations, predictor architectures, data distributions, reference-set sizes, and other relevant settings. We report the key settings for each experiment in the main text and provide additional implementation details in the appendix.

## 4.1 Disagreement is predictable

Disagreement is predictable from a single representation. We first ask whether inter-model representational disagreement can be predicted from the representation of a single model. To test this, we take each of CLIP, DINOv2, and ViT-21K in turn as the anchor model and use the other two as comparison models. For each anchor, we train a separate predictor using only the anchor’s representation as input to estimate the measured $\mathrm { N A } _ { k } .$ As shown in $\mathrm { F i g . } 3 ,$ the predicted and measured $\mathrm { \bar { N A } } _ { k }$ values are positively correlated for all three anchor models, with Pearson correlation coefficients ranging from 0.708 to 0.746. These results support the hypothesis that inter-model representational disagreement leaves a detectable sample-level signature within each model’s representation space.

![](images/a26e72ab6eb99b043dd58725900f3a511bf40d62322031240260b8591f124c45.jpg)

![](images/390682f5cf8e4a4435950d75737fec41ca2ec477d37bc7b362d7230165c05199.jpg)

![](images/0ef849a430bbe90068e77b5bdab670911d8c090d87cc681f17af9008a269dba1.jpg)  
Figure 3: Measured and predicted $\mathrm { N A } _ { k }$ across three encoders. Hexagons indicate sample density, while markers and error bars show binned means and 90% intervals. The strong agreement between measured and predicted values demonstrates that cross-model representational disagreement can be reliably inferred from single-model representations.

Predictability across model configurations. We next examine whether disagreement predictability depends on the anchor model or the set of comparison models. Building on the preceding experiment, we further introduce Swin-B pretrained on ImageNet-22K [Liu et al., 2021], SigLIP 2 with a ViT-B/16 image encoder at $2 2 4 \times 2 2 4$ resolution [Tschannen et al., 2025], and ConvNeXt-B pretrained on ImageNet-22K [Liu et al., 2022]. Across 11 configurations, we train a separate predictor using only the designated anchor representation. The dataset setup remains unchanged from the preceding experiment, while we vary the composition and size of the comparison set, as well as the choice of anchor model.

(i) Anchor model choice. To assess how the choice of anchor model affects disagreement predictability, we evaluate six different models. We use Swin-B, SigLIP 2, and ConvNeXt-B as the comparison model set when evaluating CLIP, DI-NOv2, and ViT-21K, and use CLIP, DINOv2, and ViT-21K as the comparison model set when evaluating the other three models. As shown in Table 1, predicted and measured $\mathrm { N A } _ { k }$ remain positively correlated across all six configurations, with Pear son r ranging from 0.701 to 0.770. These results show that inter-model representational disagree-

Table 1: Disagreement prediction across anchor-model choices.
<table><tr><td>Comparison models</td><td>Anchor</td><td>Pearson r ↑</td></tr><tr><td rowspan="3">Swin-B SigLIP 2 ConvNeXt-B</td><td>CLIP</td><td>0.734</td></tr><tr><td>DINOv2</td><td>0.751</td></tr><tr><td>ViT-21K</td><td>0.701</td></tr><tr><td rowspan="3">CLIP DINOv2 ViT-21K</td><td>Swin-B</td><td>0.770</td></tr><tr><td>SigLIP 2</td><td>0.750</td></tr><tr><td>ConvNeXt-B</td><td>0.767</td></tr></table>

ment can be learned regardless of which model supplies the representation to the predictor and serves as the anchor model in the agreement computation.

(ii) Composition of the comparison model set. We first fix DINOv2 as the anchor model and the number of models in the comparison model set at three, while varying which model types constitute the set. As shown in Table $^ { 2 \mathrm { a , } }$ , the three comparison model sets yield Pearson correlations of 0.751, 0.754, and 0.762, respectively. The consistently high positive correlations indicate that inter-model representational disagreement remains predictable across different compositions of the comparison model set and is not limited to one particular model combination.

(iii) Comparison model set size. We next examine whether inter-model representational disagreement remains predictable when the number of models in the comparison model set changes. To test this, we fix CLIP as the anchor model and use nested comparison model sets containing two to five models. As shown in Table 2b, performance remains comparable across all four settings, with Pearson r ranging from 0.727 to 0.750. These results show that inter-model representational disagreement can be learned across the tested comparison model set sizes and maintains strong predictive performance when more models are included in the comparison model set.

Table 2: Disagreement prediction across comparison-model compositions and comparison model set sizes. (a) Composition (DINOv2 anchor). (b) Set size (CLIP anchor).
<table><tr><td>Comparison models</td><td>Pearson r ↑</td><td>Comparison models</td><td>Pearson r ↑</td></tr><tr><td>Swin-B, SigLIP 2, ConvNeXt-B</td><td>0.751</td><td>Swin-B, SigLIP 2</td><td>0.727</td></tr><tr><td rowspan="2">Swin-B, SigLIP 2, ViT-21K</td><td>0.754</td><td>Swin-B, SigLIP 2, ConvNeXt-B</td><td>0.734</td></tr><tr><td></td><td>Swin-B, SigLIP 2, ConvNeXt-B, DINOv2</td><td>0.744</td></tr><tr><td>CLIP, SigLIP 2, ConvNeXt-B</td><td>0.762</td><td>Swin-B, SigLIP 2, ConvNeXt-B, DINOv2, ViT-21K</td><td>0.750</td></tr></table>

Taken together, these experiments show that the predictability of inter-model representational disagreement persists across comparison panel compositions, anchor model choices, and panel sizes. These variations modify complementary components of the protocol, including the models used to construct the agreement target, the predictor input, and the breadth of the comparison panel. Nevertheless, the positive association between predicted and measured $\mathrm { N A } _ { k }$ remains consistent, indicating that the observed predictability is not confined to any single model combination.

Disagreement remains predictable under distribution shift. We next ask whether disagreement remains predictable when the query distribution changes. To test this, we evaluate the ImageNet-1K-trained predictor on nine external datasets, four natural ImageNet shifts, and five Gaussian-noise levels without retraining or test-time adaptation. For each query, predicted $\mathrm { N A } _ { k }$ is produced by applying this fixed predictor to the query’s CLIP representation, while measured $\mathrm { N A } _ { k }$ is computed by comparing the neighborhoods retrieved by CLIP, DINOv2, and ViT-21K from the same fixed ImageNet-1K training reference set. As shown in Tables 3 and 4, predicted and measured $\mathrm { N A } _ { k }$ remain positively correlated across all settings, with Pearson r ranging from 0.405 to 0.661 on the external datasets, from 0.710 to 0.720 on ImageNet-V2, and reaching 0.512 on ImageNet-R. These results show that disagreement predictability transfers beyond the training distribution, although its strength varies across domains. This setting reflects practical use: the data available during training are finite, while future queries may come from a much broader range of distributions.

Table 3: Cross-dataset disagreement prediction. Pearson correlations between predicted and measured $\mathrm { N A } _ { k }$ on nine external datasets using the fixed ImageNet-1K-trained predictor. Dataset sources: Aircraft [Maji et al., 2013], Caltech101 [Fei-Fei et al., 2004], Cars [Krause et al., 2013], DTD [Cimpoi et al., 2014], EuroSAT [Helber et al., 2019], Food101 [Bossard et al., 2014], Pets [Parkhi et al., 2012], SUN397 [Xiao et al., 2010], and UCF101 [Soomro et al., 2012].
<table><tr><td></td><td>Aircraft</td><td>Caltech101</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>Food101</td><td>Pets</td><td>SUN397</td><td>UCF101</td></tr><tr><td>Pearson r ↑</td><td>0.661</td><td>0.656</td><td>0.425</td><td>0.587</td><td>0.405</td><td>0.585</td><td>0.499</td><td>0.597</td><td>0.637</td></tr></table>

Table 4: Disagreement prediction under ImageNet distribution shifts. Pearson correlations between predicted and measured $\mathrm { N A } _ { k }$ on ImageNet-V2 [Recht et al., 2019], ImageNet-R [Hendrycks et al., 2021], and Gaussian-corrupted ImageNet-C [Hendrycks and Dietterich, 2019]. The labels S1, S2, S3, S4, and S5 denote corruption severity levels 1, 2, 3, 4, and 5, respectively.
<table><tr><td rowspan="2"></td><td colspan="3">ImageNet-V2</td><td rowspan="2">ImageNet-R</td><td colspan="4">ImageNet-C: Gaussian</td><td rowspan="2">S5</td></tr><tr><td>Matched</td><td>Threshold 0.7</td><td>Top Images</td><td>S1</td><td>S2</td><td>S3</td><td>S4</td></tr><tr><td>Pearson r ↑</td><td>0.710</td><td>0.720</td><td>0.717</td><td>0.512</td><td>0.726</td><td>0.732</td><td>0.724</td><td>0.650</td><td>0.394</td></tr></table>

Under Gaussian noise, Pearson r decreases from 0.726 at severity 1 to 0.394 at severity 5. In the three examples shown in Figure 4, absolute errors increase from 0.013 to 0.031 and 0.037 as the neighborhoods retrieved by CLIP, DINOv2, and ViT-21K become increasingly divergent. These observations suggest that severe corruption pushes query representations beyond the regime learned from ImageNet-1K, causing the learned mapping from anchor representations to cross-model agreement to become unreliable and the predictions to drift.

![](images/adc25bd65c631f34b23eecd33b030162c2083779131fe690be9fddfa0e8c032f.jpg)  
Figure 4: Prediction drift under Gaussian-noise corruption. From left to right, corruption severity increases; each example shows a query and its three nearest neighbors retrieved by CLIP, DINOv2, and ViT-21K.

Predictability beyond the anchor. In the preceding experiments, the anchor model provides the predictor input, and its neighborhood also contributes to the supervision target. We further ask whether a single model’s representation can predict representational disagreement within a specified set of other models. To test this, we train a predictor using only CLIP representations, with the target defined as the average pairwise top-50 neighborhood agreement among DINOv2, ViT-21K, and Swin-B, excluding CLIP. As Figure 5 shows, the predicted and measured agreement remain correlated, with Pearson $r = 0 . 5 7 8 .$ . For this model set, the result shows that a CLIP representation provides information for predicting disagreement among models that do not include CLIP.

![](images/6e77e3eb75a9487dc37dd3d07b7ca72f9ceba14b875a91d6aa8bba1e5af7fede.jpg)

![](images/35f5c609b59710af808425a859e437a529206230dc42dea84c5588574b92b8e1.jpg)  
Figure 5: Leave-anchor-out diagnosis on ImageNet-1K. Both panels use only CLIP as input; the target includes CLIP on the left and excludes it on the right.

Predictability holds across predictor architectures. We assess architectural sensitivity by comparing a 2-layer MLP, Residual FFN, and Transformer Encoder across three anchor models, holding the predictor input and measured $\mathrm { N A } _ { k }$ target fixed across architectures for each anchor. As shown in Table 5, all three architectures yield positive correlations for every anchor; the 2-layer MLP achieves Pearson r values ranging from 0.490 to 0.544, while the Residual FFN reaches from 0.708 to 0.746. These results show that different architectures recover the disagreement signal, indicating that its predictability is not an artifact of a particular design, although architecture affects prediction accuracy.

Table 5: Predictor architecture comparison measured by Pearson correlation. All predictors use the same single-representation input and measured consistency supervision.
<table><tr><td>Model</td><td>2-layer MLP</td><td>Transformer Encoder</td><td>Residual FFN</td></tr><tr><td>CLIP</td><td>0.544</td><td>0.712</td><td>0.737</td></tr><tr><td>DINOv2</td><td>0.523</td><td>0.724</td><td>0.746</td></tr><tr><td>ViT-21K</td><td>0.490</td><td>0.696</td><td>0.708</td></tr></table>

## 4.2 Predicted disagreement is practical

Prediction substantially reduces measurement cost. Computing measured $\mathrm { N A } _ { k }$ requires neighborhood search across multiple representation spaces for every query. We compare its PyTorch and FAISS implementations with predicted $\mathrm { N A } _ { k } .$ , obtained through a single forward pass over the CLIP representation, on 100,000 ImageNet-1K test samples. As shown in Table 6, predicted 1K test samples. As shown in Table 6

Table 6: Post-embedding cost on 100K ImageNet-1K samples. Lower is better.
<table><tr><td>Method</td><td>Time  $\downarrow$  (ms/sample)</td><td>Peak memory ↓ (MiB)</td><td>Storage ↓ (MiB)</td></tr><tr><td>Measured (PyTorch)</td><td>49.54</td><td>20,643</td><td>10,009</td></tr><tr><td>Measured (FAISS)</td><td>20.54</td><td>29,729</td><td>10,009</td></tr><tr><td>Predicted</td><td>5.34</td><td>9</td><td>5</td></tr></table>

$\mathrm { N A } _ { k }$ takes only 5.34 ms per sample, making it

3.85× faster than FAISS and 9.28× faster than PyTorch, while reducing GPU memory and asset storage by over three orders of magnitude. These results show that prediction provides an efficient alternative to repeatedly measuring inter-model representational disagreement.

Predicted disagreement tracks downstream classification risk. Inter-model representational disagreement reflects differences in how foundation models organize the same input, which may lead to different behavior on downstream tasks. We therefore ask whether predicted disagreement retains the association between measured disagreement and downstream classification risk. Across nine benchmarks, we compute Kendall $\tau _ { b }$ correlations of measured and predicted disagreement, each defined as $1 - \mathrm { N A } _ { k } ,$ with the per-sample multiclass Brier risk of fixed, dataset-specific CLIP linear probes. As shown in Table $^ { 7 , }$ , predicted disagreement is positively associated with risk on every dataset, and the average $\tau _ { b }$ increases from 0.18 for measured disagreement to 0.21 for predicted disagreement. One possible explanation is that training filters some noise in the measured $\mathrm { N A } _ { k }$ targets, allowing predicted disagreement to align more closely with downstream risk. These results indicate that predicted disagreement retains an association with downstream classification risk.

Table 7: Predicted representational disagreement tracks downstream classification risk. We report sample-wise Kendall $\tau _ { b }$ correlations between measured or predicted disagreement and multiclass Brier risk across nine benchmarks. Red values indicate increases over directly measured disagreement; green values indicate decreases.
<table><tr><td></td><td>Aircraft</td><td>Caltech101</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>Food101</td><td>Pets</td><td>SUN397</td><td>UCF-101</td><td>Average</td></tr><tr><td>Measured</td><td>0.24</td><td>0.20</td><td>0.09</td><td>0.16</td><td>0.09</td><td>0.32</td><td>0.16</td><td>0.20</td><td>0.18</td><td>0.18</td></tr><tr><td>Predicted</td><td>0.26</td><td>0.23</td><td>0.10</td><td>0.17</td><td>0.08</td><td>0.40</td><td>0.17</td><td>0.23</td><td>0.23</td><td>0.21</td></tr><tr><td></td><td>+0.02</td><td>+0.03</td><td>+0.01</td><td>+0.01</td><td>-0.01</td><td>+0.08</td><td>+0.01</td><td>+0.03</td><td>+0.05</td><td>+0.03</td></tr></table>

Prediction reduces data requirements. Obtaining measured $\mathrm { N A } _ { k }$ typically requires a large reference set. In contrast, estimating inter-model disagreement through prediction can leverage learned generalization to reduce dependence on reference-set size. We construct nested reference subsets ranging from 5K to 500K samples and use the measured $\mathrm { N A } _ { k }$ from each subset to train a predictor on the same fixed 50,000 training samples. We then compute the Pearson correlation between predicted $\mathrm { N A } _ { k } ^ { - }$ and measured $\mathrm { N A } _ { k }$ obtained offline using the full reference set. To quantify the retained ability to approximate full-reference measured $\mathrm { N A } _ { k }$ , we divide this correlation by that achieved by a predictor trained with full-reference supervision. As shown in Figure $^ { 6 , }$ even 100K references retain 82.3% of the full-reference predictive correlation, while 500K references, fewer than half of the full reference set, retain 90.6% and approach full-reference performance with substantially less data.

![](images/04cb7aa8494e00a4b86aa001dbad0541f19ae6eb0b771ffd3ddb88c9913456b7.jpg)  
Figure 6: Reference set scaling for disagreement prediction. Pearson correlation retained relative to using the full ImageNet-1K training set as the reference set; all predictors use the same 50,000 training samples and are evaluated against the same full-reference $\mathrm { N A } _ { k }$ target.

## 5 Conclusion

We presented a framework for predicting Rashomon Representation, the sample-dependent disagreement among foundation models, from a single model’s representation. Cross-model neighborhood agreement operationalizes this disagreement and provides supervision for a lightweight predictor. Across diverse vision foundation models and evaluation settings, predicted agreement consistently tracks measured agreement, indicating that inter-model representational disagreement is structured, input-dependent, and partially encoded in individual representation spaces. This enables efficient inference-time estimation without computing representations and neighborhoods for every comparison model. Future work will investigate the mechanisms underlying this predictability and develop self-supervised approaches that learn representational disagreement without explicit cross-model measurements.

## References

Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Foundation models defining a new era in vision: A survey and outlook. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(4):2245–2264, 2025.

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEiT: BERT pre-training of image transformers. In International Conference on Learning Representations, 2022.

Luca Barsellotti, Lorenzo Bianchi, Nicola Messina, Fabio Carrara, Marcella Cornia, Lorenzo Baraldi, Fabrizio Falchi, and Rita Cucchiara. Talking to DINO: Bridging self-supervised vision backbones with language for open-vocabulary segmentation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 22025–22035, 2025.

Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In Computer Vision – ECCV 2014, volume 8694 of Lecture Notes in Computer Science, pages 446–461. Springer International Publishing, 2014.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 9650–9660, 2021.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 1597– 1607. PMLR, 2020.

Laure Ciernik, Lorenz Linhardt, Marco Morik, Jonas Dippel, Simon Kornblith, and Lukas Muttenthaler. Objective drives the consistency of representational similarity across datasets. In Proceedings of the 42nd International Conference on Machine Learning, pages 10920–10948, 2025.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2014.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2004, Washington, DC, USA, June 27 - July 2, 2004, page 178. IEEE Computer Society, 2004.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Ávila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent - a new approach to self-supervised learning. In Advances in Neural Information Processing Systems, volume 33, pages 21271–21284, 2020.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, June 2022.

Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, July 2019.

Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 8340–8349, October 2021.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. Position: The platonic representation hypothesis. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 20617–20642. PMLR, 2024.

Max Klabunde, Tassilo Wald, Tobias Schumacher, Klaus Maier-Hein, Markus Strohmaier, and Florian Lemmerich. Resi: A comprehensive benchmark for representational similarity measures. In The Thirteenth International Conference on Learning Representations, 2025.

Camila Kolling, Till Speicher, Vedant Nanda, Mariya Toneva, and Krishna P. Gummadi. Investigating the effects of fairness interventions using pointwise representational similarity. Transactions on Machine Learning Research, 2025. ISSN 2835-8856.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3519–3529. PMLR, 2019.

Jonathan Krause, Michael Stark, Jia Deng, and Fei-Fei Li. 3d object representations for fine-grained categorization. In 2013 IEEE International Conference on Computer Vision Workshops, pages 554–561. IEEE, December 2013. 4th IEEE Workshop on 3D Representation and Recognition.

Nikolaus Kriegeskorte, Marieke Mur, and Peter A. Bandettini. Representational similarity analysis - connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience, 2:4, 2008.

Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John Hopcroft. Convergent learning: Do different neural networks learn the same representations? In Proceedings ofthe 1st International Workshop on Feature Extraction: Modern Questions and Challenges at NIPS 2015, volume 44 of Proceedings ofMachine Learning Research, pages 196–212. PMLR, 2015.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986, 2022.

Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft, June 2013.

Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang. What is being transferred in transfer learning? In Advances in Neural Information Processing Systems, volume 33, pages 512–523, 2020.

Thao Nguyen, Maithra Raghu, and Simon Kornblith. Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth. In International Conference on Learning Representations, 2021.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael

Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

Young-Jin Park, Hao Wang, Shervin Ardeshir, and Navid Azizan. Quantifying representation reliability in self-supervised learning models. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of Proceedings ofMachine Learning Research, pages 2835–2860. PMLR, 2024.

Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 3498–3505. IEEE, June 2012.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR, 2021.

Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? In Advances in Neural Information Processing Systems, volume 34, pages 12116–12128, 2021.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do ImageNet classifiers generalize to ImageNet? In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pages 5389–5400. PMLR, 2019.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Fei-Fei Li. ImageNet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211–252, 2015.

Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. Technical Report CRCV-TR-12-01, Center for Research in Com puter Vision, University of Central Florida, November 2012. Also available as arXiv:1212.0402.

Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. MLP-Mixer: An all-MLP architecture for vision. In Advances in Neural Information Processing Systems, volume 34, pages 24261–24272, 2021.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008, 2017.

Jianxiong Xiao, James Hays, Krista A. Ehinger, Aude Oliva, and Antonio Torralba. SUN database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 3485–3492. IEEE, June 2010.

Zhenda Xie, Zigang Geng, Jingcheng Hu, Zheng Zhang, Han Hu, and Yue Cao. Revealing the dark secrets of masked image modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14475–14485, 2023.

Xin Zhang and Robby T. Tan. Mamba as a bridge: Where vision foundation models meet vision language models for domain-generalized semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14527–14537, 2025.

## A Common Experimental Setup

## A.1 Data Splits

We use predefined ImageNet-1K splits for predictor training, model selection, and final evaluation. The training, validation, and test splits, denoted by $\mathcal { D } _ { \mathrm { t r } } , \mathcal { D } _ { \mathrm { v a l } }$ , and $\mathcal { D } _ { \mathrm { t e } } .$ , contain 1,281,167, 50,000, and 100,000 samples, respectively.

Except in the reference-set scaling experiment, the full training split serves as the shared neighborhood reference set:

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { r e f } } = \mathcal { D } _ { \mathrm { t r } } . } \end{array}\tag{2}
$$

Neighborhoods for training, validation, and test queries are constructed with respect to this reference set. Representations and neighborhood sets are aligned by image identity across models so that cross-model comparisons refer to the same reference samples.

## A.2 Foundation Models and Representations

We use six frozen vision encoders. Each model follows its corresponding image preprocessing pipeline, and the extracted representations are used for neighborhood construction and predictor training. Table 8 summarizes the model configurations.

Table 8: Foundation models and extracted representations.
<table><tr><td>Model</td><td>Model source</td><td>Representation</td><td>Dimension</td></tr><tr><td>CLIP ViT-B/32</td><td>openai/clip-vit-base-patch32</td><td>Projected image features</td><td>512</td></tr><tr><td>DINOv2 ViT-B/14</td><td>facebook/dinov2-base</td><td>Pooled output; CLS if unavailable</td><td>768</td></tr><tr><td>ViT-B/16</td><td>google/vit-base-patch16-224-in21k</td><td>Pooled output; CLS if unavailable</td><td>768</td></tr><tr><td>Swin-B</td><td>microsoft/swin-base-patch4-window7-224-in22k</td><td>Pooled output</td><td>1024</td></tr><tr><td>SigLIP 2 ViT-B/16</td><td>google/siglip2-base-patch16-224</td><td>Projected image features</td><td>768</td></tr><tr><td>ConvNeXt-B</td><td>timm/convnext_base.fb_in22k</td><td>Pooled pre-logit features</td><td>1024</td></tr></table>

The normalized representation of sample x under model $h _ { m }$ is

$$
\tilde { z } _ { m } ( x ) = \frac { h _ { m } ( x ) } { \| h _ { m } ( x ) \| _ { 2 } } .\tag{3}
$$

All foundation models remain frozen throughout the experiments. Cross-model neighborhood comparisons operate on reference image identities and do not require equal representation dimensionality.

## A.3 Neighborhood Construction and Supervision

Following the main text, let

$$
\mathcal { H } = \left\{ h _ { 0 } , h _ { 1 } , \dotsc , h _ { M - 1 } \right\}\tag{4}
$$

denote the foundation-model pool, where $h _ { 0 }$ is the anchor and the remaining models form the comparison set. The similarity between query x and reference sample $x _ { j }$ under model $h _ { m }$ is

$$
\begin{array} { r } { s _ { m } ( x , x _ { j } ) = \tilde { z } _ { m } ( x ) ^ { \top } \tilde { z } _ { m } ( x _ { j } ) . } \end{array}\tag{5}
$$

We perform exact nearest-neighbor search using FAISS FlatIP to obtain $\mathcal { N } _ { k } ^ { ( m ) } ( x )$ , with $k = 5 0$ by default. Pairwise neighborhood agreement is

$$
a _ { k } ^ { ( u , v ) } ( x ) = \frac { \left| \mathcal { N } _ { k } ^ { ( u ) } ( x ) \cap \mathcal { N } _ { k } ^ { ( v ) } ( x ) \right| } { k } .\tag{6}
$$

The default supervision target is

$$
y ( x ) = \mathrm { N A } _ { k } ( x ; h _ { 0 } , \mathcal { H } ) = \frac { 1 } { M - 1 } \sum _ { m = 1 } ^ { M - 1 } a _ { k } ^ { ( 0 , m ) } ( x ) .\tag{7}
$$

This target lies in $[ 0 , 1 ] .$ . Higher values indicate greater agreement between the local neighborhood structures induced by the anchor and comparison models, corresponding to weaker inter-model representational disagreement.

## A.4 Predictor Training

The predictor uses only the anchor representation:

$$
\hat { y } ( x ) = g _ { \phi } ( \tilde { z } _ { 0 } ( x ) ) .\tag{8}
$$

Comparison-model representations and neighborhoods are used to construct offline supervision and do not enter the predictor’s forward pass. Except in the architecture comparison, we use the Residual FFN described in Appendix I.

The predictor minimizes

$$
\mathcal { L } ( \phi ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { t r } } | } \sum _ { x _ { i } \in \mathcal { D } _ { \mathrm { t r } } } \ell ( g _ { \phi } ( \tilde { z } _ { 0 } ( x _ { i } ) ) , y ( x _ { i } ) ) .\tag{9}
$$

We use a mean squared error (MSE) objective with a fixed scaling factor of $1 / 2$ . The per-sample loss is

$$
\ell ( \hat { y } , y ) = \frac { 1 } { 2 } ( \hat { y } - y ) ^ { 2 } .\tag{10}
$$

Unless otherwise specified, we use AdamW with a learning rate and weight decay of $1 0 ^ { - 4 }$ and train for 10 epochs. Checkpoints are selected by validation MAE. The selected parameters, denoted by $\phi ^ { \star }$ are evaluated on the test split. Only predictor parameters are optimized.

## B Common Evaluation and Visualization Procedures

## B.1 Pearson Correlation

For two nonconstant sample sequences $a = ( a _ { i } ) _ { i = 1 } ^ { n }$ and $b = ( b _ { i } ) _ { i = 1 } ^ { n }$ , Pearson correlation is

$$
\mathrm { C o r r } _ { \mathrm { P } } ( a , b ) = \frac { \sum _ { i = 1 } ^ { n } ( a _ { i } - \bar { a } ) ( b _ { i } - \bar { b } ) } { \sqrt { \sum _ { i = 1 } ^ { n } ( a _ { i } - \bar { a } ) ^ { 2 } } \sqrt { \sum _ { i = 1 } ^ { n } ( b _ { i } - \bar { b } ) ^ { 2 } } } ,\tag{11}
$$

where

$$
{ \bar { a } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } a _ { i } , \qquad { \bar { b } } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } b _ { i } .\tag{12}
$$

In the disagreement prediction experiments, let

$$
y _ { i } = y ( x _ { i } ) , \qquad \hat { y } _ { i } = g _ { \phi ^ { \star } } \big ( \tilde { z } _ { 0 } ( x _ { i } ) \big ) .\tag{13}
$$

The reported correlation is

$$
r _ { \mathrm { p r e d } } = \mathrm { C o r r } _ { \mathrm { P } } ( \hat { y } , y ) ,\tag{14}
$$

which measures the linear association between predicted and measured neighborhood agreement.

## B.2 Binned Means and Distribution Intervals

Sample-level figures display the two-dimensional density of $( y _ { i } , \hat { y } _ { i } )$ and group samples into bins of measured agreement. For a nonempty bin $I _ { b } ,$ , define

$$
{ \mathcal { T } } _ { b } = \{ i : y _ { i } \in I _ { b } \} .\tag{15}
$$

The mean measured and predicted agreement within the bin are

$$
\bar { y } _ { b } = \frac { 1 } { \left| \mathcal { T } _ { b } \right| } \sum _ { i \in \mathcal { Z } _ { b } } y _ { i } , \qquad \bar { \hat { y } } _ { b } = \frac { 1 } { \left| \mathcal { T } _ { b } \right| } \sum _ { i \in \mathcal { T } _ { b } } \hat { y } _ { i } .\tag{16}
$$

The binned means summarize prediction trends across measured agreement levels. Vertical intervals extend from the 5th to the 95th percentile of predictions within each bin, representing the central 90% of the prediction distribution.

## C Qualitative Visualization of Inter-Model Disagreement

Figure 7 illustrates Rashomon Representation through high- and low-agreement queries. For the envelope and hornbill examples, CLIP, DINOv2, and ViT-21K retrieve similar neighbors, and their linear probes predict the correct class. For the flying-bird and black-swan examples, the neighborhoods diverge and the probes assign different classes. The Residual FFN, using only the CLIP representation, preserves this high–low ordering in its predicted $\mathrm { N A } _ { 5 0 }$

Across ImageNet-1K and ImageNet-V2 variants, we use the ImageNet-1K training set as the shared reference set. Measured $\mathrm { N A } _ { 5 0 } ^ { - }$ averages CLIP–DINOv2 and CLIP–ViT-21K Top-50 overlap; the figure shows only Top-5 neighbors. In the 10,000-query case-study cohort, all three probes are correct for 24.1% of queries with $\mathrm { \check N A } _ { 5 0 } < 0 . 1 0 $ , versus 68.3% with $\mathrm { N A } _ { 5 0 } \geq 0 . 4 0$ , indicating an association between neighborhood organization and classification outcomes.

![](images/27cc4ee3f40c7daaaa333dd84429629ed7d9d20ea9d6a1e9e5a64dce0e5845b2.jpg)  
Figure 7: High (top) and low (bottom) cross-model neighborhood agreement. Rows correspond to CLIP, DINOv2, and ViT-21K.

## D Predicting Disagreement from a Single Representation

This experiment tests whether inter-model disagreement can be predicted from a single model representation. We separately use CLIP, DINOv2, and ViT as anchors, with the other two models defining the supervision target.

Table 9: Configurations for prediction from a single representation.
<table><tr><td>Anchor model</td><td>Comparison models</td><td>Input dimension</td></tr><tr><td>CLIP</td><td>DINOv2, ViT-21K</td><td>512</td></tr><tr><td>DINOv2</td><td>CLIP, ViT-21K</td><td>768</td></tr><tr><td>ViT-21K</td><td>CLIP, DINOv2</td><td>768</td></tr></table>

A corresponding Residual FFN is trained for each configuration using only the designated anchor representation. All three configurations share the data splits, reference set, neighborhood size, and training protocol. We evaluate the association between predicted and measured agreement on the ImageNet-1K test split.

## E Effect of Comparison-Model Composition

This experiment fixes DINOv2 as the anchor and uses three comparison models while varying their composition.

Table 10: Comparison-model compositions with DINOv2 fixed as the anchor.
<table><tr><td>Configuration</td><td>Comparison models</td></tr><tr><td>1</td><td>Swin, SigLIP 2, ConvNeXt</td></tr><tr><td>2</td><td>Swin, SigLIP 2, ViT-21K</td></tr><tr><td>3</td><td>CLIP, SigLIP 2, ConvNeXt</td></tr></table>

All configurations use the 768-dimensional DINOv2 representation as predictor input. We construct a separate supervision target for each comparison set and train a corresponding Residual FFN. The data splits, reference set, $\bar { k } = 5 0$ , and training protocol remain fixed.

## F Effect of Anchor-Model Choice

This experiment examines disagreement predictability across different anchor representations. Each configuration uses three comparison models.

Table 11: Configurations for evaluating anchor-model choice.
<table><tr><td>Anchor model</td><td>Comparison models</td></tr><tr><td>CLIP</td><td>Swin, SigLIP 2, ConvNeXt</td></tr><tr><td>DINOv2</td><td>Swin, SigLIP 2, ConvNeXt</td></tr><tr><td>ViT-21K Swin</td><td>Swin, SigLIP 2, ConvNeXt DINOv2, ViT-21K, CLIP</td></tr><tr><td>SigLIP 2</td><td>DINOv2, ViT-21K, CLIP</td></tr><tr><td>ConvNeXt</td><td>DINOv2, ViT-21K, CLIP</td></tr></table>

Each predictor receives only its corresponding anchor representation. Input dimensionality varies with the anchor, while the hidden dimension remains 512. A Residual FFN is trained for each configuration using the same data splits, reference set, neighborhood size, and training protocol.

## G Effect of Comparison-Set Size

This experiment fixes CLIP as the anchor and uses nested comparison sets containing two to five models.

Table 12: Nested comparison sets with CLIP fixed as the anchor.
<table><tr><td>Number of models</td><td>Comparison set</td></tr><tr><td>2</td><td>Swin, SigLIP 2</td></tr><tr><td>3</td><td>Swin, SigLIP 2, ConvNeXt</td></tr><tr><td>4</td><td>Swin, SigLIP 2, ConvNeXt, DINOv2</td></tr><tr><td>5</td><td>Swin, SigLIP 2, ConvNeXt, DINOv2, ViT-21K</td></tr></table>

For each set, the target averages pairwise neighborhood agreement between CLIP and each comparison model. All settings use the same CLIP input representations, reference set, and $k = 5 0$ with a separate Residual FFN trained for each target. This evaluates prediction performance as the comparison set is progressively expanded.

## H Generalization under Distribution Shift

This experiment reuses the CLIP-anchor Residual FFN trained on ImageNet-1K without retraining or test-time adaptation. Measured agreement is computed using CLIP, DINOv2, and ViT with respect to the full ImageNet-1K training reference set.

The evaluation datasets match the tables in the main text:

• External datasets: Aircraft, Caltech101, Cars, DTD, EuroSAT, Food101, Oxford-IIIT Pets, SUN397, and UCF101.

• Natural distribution shifts: ImageNet-R and the ImageNet-V2 Matched Frequency, Threshold 0.7, and Top Images variants.

• Image corruption: ImageNet-C Gaussian Noise at severity levels 1 through 5.

The nine external datasets use their corresponding validation splits. Pearson correlation between predicted and measured agreement is computed separately for each dataset and corruption severity. The predictor and reference set remain fixed across evaluation settings.

## I Predictor Architecture Comparison

## I.1 Experimental Setup

We evaluate the 2-layer MLP, Residual FFN, and Transformer Encoder under the three anchor/comparison configurations in Appendix D. For a given anchor, all architectures use the same input representations, supervision targets, data splits, and training protocol. Each architecture produces a scalar agreement prediction through a sigmoid output.

Below, Linea $\cdot { \xrightarrow { } } b$ denotes a learnable affine map from a to b dimensions, LN denotes LayerNorm, and $\mathrm { D r o p } _ { 0 . 1 }$ denotes dropout with probability 0.1. Maps and normalization layers at different locations have independent parameters. Dropout is enabled only during training.

## I.2 2-layer MLP

The 2-layer MLP maps the anchor representation to a 512-dimensional hidden vector and then to a scalar:

$$
u = \mathrm { D r o p } _ { 0 . 1 } ( \mathrm { G E L U } ( \mathrm { L i n e a r } _ { d _ { 0 }  5 1 2 } ( \tilde { z } _ { 0 } ( x ) ) ) ) ,\tag{17}
$$

$$
\hat { y } ( x ) = \sigma ( \mathrm { L i n e a r } _ { 5 1 2  1 } ( u ) ) ,\tag{18}
$$

where σ denotes the sigmoid function.

## I.3 Residual FFN

The input projection is

$$
u ^ { ( 0 ) } = \mathrm { D r o p } _ { 0 . 1 } ( \mathrm { G E L U } ( \mathrm { L N } ( \mathrm { L i n e a r } _ { d _ { 0 }  5 1 2 } ( \tilde { z } _ { 0 } ( x ) ) ) ) ) .\tag{19}
$$

Two residual feed-forward blocks are then applied:

$$
\begin{array} { r } { u ^ { ( \ell ) } = u ^ { ( \ell - 1 ) } + F _ { \ell } \left( \mathrm { L N } _ { \ell } ( u ^ { ( \ell - 1 ) } ) \right) , \qquad \ell = 1 , 2 . } \end{array}\tag{20}
$$

Each residual branch is

$$
F _ { \ell } ( v ) = \mathrm { D r o p } _ { 0 . 1 } \Big ( \mathrm { L i n e a r } _ { 5 1 2  5 1 2 } \big ( \mathrm { D r o p } _ { 0 . 1 } \big ( \mathrm { G E L U } ( \mathrm { L i n e a r } _ { 5 1 2  5 1 2 } ( v ) ) \big ) \big ) \Big ) .\tag{21}
$$

The prediction is

$$
\begin{array} { r } { \hat { y } ( x ) = \sigma ( \mathrm { L i n e a r } _ { 5 1 2  1 } ( \mathrm { L N } ( u ^ { ( 2 ) } ) ) ) . } \end{array}\tag{22}
$$

## I.4 Transformer Encoder

Input tokens. The anchor representation is first projected to a 512-dimensional token:

$$
t ( x ) = \mathrm { L N } ( \mathrm { D r o p } _ { 0 . 1 } ( \mathrm { G E L U } ( \mathrm { L i n e a r } _ { d _ { 0 }  5 1 2 } ( \tilde { z } _ { 0 } ( x ) ) ) ) ) .\tag{23}
$$

Let $c _ { \mathrm { C L S } } \in \mathbb { R } ^ { 5 1 2 }$ be a learnable CLS token and $P \in \mathbb { R } ^ { 2 \times 5 1 2 }$ be learnable positional embeddings. The initial sequence is

$$
\begin{array} { r } { T ^ { ( 0 ) } = \left[ \stackrel { c _ { \mathrm { C L S } } ^ { \top } } { t ( x ) ^ { \top } } \right] + P . } \end{array}\tag{24}
$$

The encoder therefore receives one CLS token and one representation token.

Multi-head self-attention. Each layer uses eight attention heads, each with dimension $5 1 2 / 8 = 6 4 $ For an input sequence S, head j computes

$$
\begin{array} { r } { Q _ { j } = \mathrm { L i n e a r } _ { 5 1 2  6 4 } ^ { Q _ { j } } ( S ) , } \\ { K _ { j } = \mathrm { L i n e a r } _ { 5 1 2  6 4 } ^ { K _ { j } } ( S ) , } \\ { V _ { j } = \mathrm { L i n e a r } _ { 5 1 2  6 4 } ^ { V _ { j } } ( S ) . } \end{array}\tag{25}
$$

Its output is

$$
A _ { j } ( S ) = \mathrm { D r o p } _ { 0 . 1 } \left[ \mathrm { s o f t m a x } \left( \frac { Q _ { j } K _ { j } ^ { \top } } { \sqrt { 6 4 } } \right) \right] V _ { j } ,\tag{26}
$$

where softmax is applied over the key-token dimension. Head outputs are concatenated and projected:

$$
\mathrm { M H S A } ( S ) = \operatorname { L i n e a r } _ { 5 1 2 \to 5 1 2 } \left( \mathrm { C o n c a t } \left( A _ { 1 } ( S ) , \ldots , A _ { 8 } ( S ) \right) \right) .\tag{27}
$$

Encoder layers. Both layers use a pre-norm configuration. For ℓ = 1, 2,

$$
U ^ { ( \ell ) } = T ^ { ( \ell - 1 ) } + \mathrm { D r o p } _ { 0 . 1 } \left( \mathrm { M H S A } _ { \ell } \left( \mathrm { L N } _ { \ell } ^ { \mathrm { a t t n } } ( T ^ { ( \ell - 1 ) } ) \right) \right) ,\tag{28}
$$

$$
\begin{array} { r } { T ^ { ( \ell ) } = U ^ { ( \ell ) } + \mathrm { F F N } _ { \ell } \left( \mathrm { L N } _ { \ell } ^ { \mathrm { f n } } ( U ^ { ( \ell ) } ) \right) . } \end{array}\tag{29}
$$

The feed-forward branch is

$$
\begin{array} { r } { \operatorname { F F N } _ { \ell } ( v ) = \operatorname { D r o p } _ { 0 . 1 } \Big ( \operatorname { L i n e a r } _ { 2 0 4 8 \to 5 1 2 } \big ( \operatorname { D r o p } _ { 0 . 1 } \big ( \operatorname { G E L U } ( \operatorname { L i n e a r } _ { 5 1 2 \to 2 0 4 8 } ( v ) ) \big ) \big ) \Big ) . } \end{array}\tag{30}
$$

Output head. The CLS output of the second encoder layer, $T _ { \mathrm { C L S } } ^ { ( 2 ) }$ , is mapped to

$$
\begin{array} { r } { \hat { y } ( x ) = \sigma ( \mathrm { L i n e a r } _ { 5 1 2  1 } ( \mathrm { L N } ( T _ { \mathrm { C L S } } ^ { ( 2 ) } ) ) ) . } \end{array}\tag{31}
$$

## J Leave-Anchor-Out Diagnostic

This experiment fixes CLIP representations as predictor inputs and compares supervision targets that include or exclude CLIP neighborhoods. Let $h _ { 0 } , h _ { 1 } , h _ { 2 } , h _ { 3 }$ denote CLIP, DINOv2, ViT, and Swin, respectively. The matched control target is

$$
y _ { \mathrm { a n c h o r } } ( x ) = \frac { 1 } { 3 } \left[ a _ { k } ^ { ( 0 , 1 ) } ( x ) + a _ { k } ^ { ( 0 , 2 ) } ( x ) + a _ { k } ^ { ( 0 , 3 ) } ( x ) \right] ,\tag{32}
$$

and the leave-anchor-out target is

$$
y _ { \mathrm { L A O } } ( x ) = \frac { 1 } { 3 } \left[ a _ { k } ^ { ( 1 , 2 ) } ( x ) + a _ { k } ^ { ( 1 , 3 ) } ( x ) + a _ { k } ^ { ( 2 , 3 ) } ( x ) \right] .\tag{33}
$$

Both settings use 512-dimensional CLIP inputs, a Residual FFN, the full training reference set, and $k = 5 0$ , with the same training protocol. They differ only in the supervision target, and a corresponding predictor is trained for each.

Both targets average three pairwise agreement terms, but $y _ { \mathrm { L A O } }$ contains no CLIP neighborhoods. We compute Pearson correlation between predictions and the corresponding measured target to assess predictability when the anchor does not participate in target construction.

## K Computational Cost Comparison

This experiment compares FAISS GPU FlatIP, PyTorch matrix multiplication followed by Top-k retrieval, and a forward pass through the Residual FFN. All methods process the same 100,000 ImageNet-1K test queries individually. Direct measurement uses the full training reference set and computes neighborhood agreement across CLIP, DINOv2, and ViT with $k = 5 0$

Timing starts from precomputed representations. FAISS indexes are constructed and loaded before the timed loop, and the PyTorch implementation keeps reference representations on the GPU. After warm-up, CUDA synchronization is performed before and after processing, and total elapsed time is recorded.

For n queries processed in T seconds, average per-sample time is

$$
t _ { \mathrm { s a m p l e } } = { \frac { 1 0 0 0 T } { n } } \quad { \mathrm { m s / s a m p l e } } .\tag{34}
$$

The speedup over direct measurement is

$$
\mathrm { S p e e d u p } = { \frac { T _ { \mathrm { m e a s u r e d } } } { T _ { \mathrm { p r e d i c t e d } } } } .\tag{35}
$$

Space costs are reported as peak GPU memory and disk storage for required assets, both in MiB. Direct measurement includes reference representations or indexes, whereas prediction storage is measured using the predictor checkpoint.

## L Association with Downstream Classification Risk

## L.1 Linear Classification Probes

For Aircraft, Caltech101, Cars, DTD, EuroSAT, Food101, Oxford-IIIT Pets, SUN397, and UCF101, we train separate linear probes on frozen CLIP representations using the corresponding training splits. For a dataset with C classes, the probe outputs

$$
p ( x ) = \mathrm { s o f t m a x } \left( W _ { \mathrm { c l s } } \tilde { z } _ { 0 } ( x ) + b _ { \mathrm { c l s } } \right) ,\tag{36}
$$

where $W _ { \mathrm { c l s } } \in \mathbb { R } ^ { C \times d _ { 0 } }$ . Ground-truth classes are denoted by $c _ { i }$ , distinguishing them from agreement targets y<sub>i</sub>. The probe training objective is

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \log p _ { c _ { i } } ( x _ { i } ) .\tag{37}
$$

The disagreement predictor is the CLIP-anchor Residual FFN trained on ImageNet-1K and remains fixed during downstream evaluation.

## L.2 Multiclass Brier Score

The sample-wise multiclass Brier score is

$$
B _ { i } = \sum _ { c = 1 } ^ { C } \left( p _ { c } ( x _ { i } ) - \mathbf { 1 } [ c = c _ { i } ] \right) ^ { 2 } ,\tag{38}
$$

where $\mathbf { 1 } [ \cdot ]$ is the indicator function. Equivalently,

$$
B _ { i } = \sum _ { c = 1 } ^ { C } p _ { c } ( x _ { i } ) ^ { 2 } - 2 p _ { c _ { i } } ( x _ { i } ) + 1 .\tag{39}
$$

We sum over classes without dividing by the number of classes, giving $B _ { i } \in [ 0 , 2 ]$ . Lower scores indicate that class probabilities are closer to the one-hot ground-truth label.

## L.3 Kendall Rank Correlation between Disagreement and Brier Risk

For downstream evaluation only, agreement scores are expressed in the direction of disagreement:

$$
d _ { i } ^ { \mathrm { m e a s u r e d } } = 1 - y _ { i } , \qquad d _ { i } ^ { \mathrm { p r e d i c t e d } } = 1 - \hat { y } _ { i } .\tag{40}
$$

Predictor training continues to use the original agreement target.

We quantify the association between disagreement and per-sample Brier risk using Kendall’s $\tau _ { b } ,$ which accounts for tied values. For paired sequences d and $B .$ , let $C$ and $D$ denote the numbers of concordant and discordant sample pairs. Let $T _ { d }$ and $T _ { B }$ denote the numbers of pairs tied only in d and only in $B ,$ respectively. Then

$$
\tau _ { b } ( d , B ) = \frac { C - D } { \sqrt { ( C + D + T _ { d } ) ( C + D + T _ { B } ) } } .\tag{41}
$$

For each dataset, the two correlations are

$$
\tau _ { \mathrm { m e a s u r e d } , B } = \tau _ { b } ( d ^ { \mathrm { m e a s u r e d } } , B ) , \qquad \tau _ { \mathrm { p r e d i c t e d } , B } = \tau _ { b } ( d ^ { \mathrm { p r e d i c t e d } } , B ) .\tag{42}
$$

A positive value indicates that samples with higher disagreement scores tend to have higher Brier risk.

Correlations are computed independently within each of the nine datasets. For $m \in$ {measured, predicted}, the equal-weight average is

$$
\bar { \tau } _ { m , B } = \frac { 1 } { 9 } \sum _ { j = 1 } ^ { 9 } \tau _ { m , B } ^ { ( j ) } .\tag{43}
$$

The unrounded difference on dataset $j$ is

$$
\begin{array} { r } { \Delta \tau _ { j } = \tau _ { \mathrm { p r e d i c t e d } , B } ^ { ( j ) } - \tau _ { \mathrm { m e a s u r e d } , B } ^ { ( j ) } . } \end{array}\tag{44}
$$

Table 7 displays correlations to two decimal places. Its colored differences are calculated from those displayed values; the Average column is calculated from unrounded dataset-level correlations before being displayed to two decimal places.

## M Effect of Reference-Set Size

This experiment fixes CLIP as the anchor, DINOv2 and ViT as comparison models, a Residual FFN predictor, and $k = 5 0$ . Predictor training queries are fixed to 50,000 samples, denoted by $\mathcal { Q } _ { \mathrm { t r } }$

Let $\mathcal { R } _ { L } \subseteq \mathcal { D } _ { \mathrm { t } 1 }$ <sub>r</sub> be a reference subset containing L samples, where

$$
L \in \{ 5 , 0 0 0 , 1 0 , 0 0 0 , 2 0 , 0 0 0 , 5 0 , 0 0 0 , 1 0 0 , 0 0 0 , 2 0 0 , 0 0 0 , 5 0 0 , 0 0 0 \} .\tag{45}
$$

Reference subsets are nested and satisfy

$$
{ \mathcal { Q } } _ { \mathrm { t r } } \cap { \mathcal { R } } _ { L } = { \mathcal { O } } .\tag{46}
$$

The full-reference control uses all 1,281,167 training samples.

Let $y ^ { ( L ) } ( x )$ denote supervision constructed using $\mathcal { R } _ { L }$ , and let $g _ { \phi _ { L } }$ denote the corresponding predictor. All predictors use the same training queries and are selected on validation data and evaluated on test data against the full-reference target $\bar { y } ^ { ( \mathrm { f u l l } ) } ( x )$ . Defining

$$
\hat { y } _ { i } ^ { ( L ) } = g _ { \phi _ { L } } ( \tilde { z } _ { 0 } ( x _ { i } ) ) ,\tag{47}
$$

we compute

$$
r _ { L } = \mathrm { C o r r } _ { \mathrm { P } } \left( \hat { y } ^ { ( L ) } , y ^ { ( \mathrm { f u l l } ) } \right) .\tag{48}
$$

The full-reference control also uses 50,000 training queries, and its correlation is denoted by $r _ { \mathrm { f u l l } }$ Pearson retention is

$$
\mathrm { R e t e n t i o n } ( L ) = \frac { r _ { L } } { r _ { \mathrm { f u l l } } } \times 1 0 0 \%\tag{49}
$$

When multiple reference subsets are evaluated at the same size, their correlations are averaged before computing the reported retention.