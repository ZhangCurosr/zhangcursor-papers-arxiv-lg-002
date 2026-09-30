# MoTIF-X: A Multimodal Tokenized Framework for Interpretable and Extensible Molecular

Representation Learning

Linqing Mo<sup>1</sup>, Jiayu Zhou<sup>1,2\*</sup>, Bin Chen<sup>1,3,4,5\*</sup>

<sup>1</sup>Department of Computer Science and Engineering, Michigan State University, East Lansing, 48824, MI, USA.

<sup>2</sup>School of Information, University of Michigan, Ann Arbor, 48109, MI, USA.

<sup>3</sup>Department of Pediatrics and Human Development, Michigan State University, Grand Rapids, MI, 49503, USA.

<sup>4</sup>Department of Pharmacology and Toxicology, Michigan State University, Grand Rapids, MI, 49503, USA.

<sup>5</sup>Center for AI-enabled Drug Discovery, College of Human Medicine, Michigan State University, Grand Rapids, MI, 49503, USA.

\*Corresponding author(s). E-mail(s): jiayuz@umich.edu; chenbi12@msu.edu;

## Abstract

Molecular representation learning is central to computer-aided drug discovery, where molecules are described through complementary views such as two-dimensional (2D) molecular graphs, motif-level substructures, simplified molecular-input line-entry system (SMILES) strings, and three-dimensional (3D) conformations. Although each view captures distinct structural information, integrating them efectively remains challenging. Many existing multimodal approaches learn modality-specific representations independently and align them only at a later stage, restricting fine-grained cross-modal interaction and substructure-level interpretability.

Here, we introduce MoTIF-X, a motif-centered approach that uses molecular motifs as structural anchors for integrating molecular graphs, SMILES strings,

and 3D conformational information. Its two-stage pretraining first learns graphgrounded motif representations through hierarchical contrastive learning across atomic, motif, and molecular scales. It then jointly contextualizes motif, SMILES, and discretized torsion-angle tokens through multimodal masked token modeling. Pretrained on a large collection of drug-like molecules with multiple 3D conformers, MoTIF-X achieved the lowest mean absolute error on all nine OpenADMET ExpansionRx endpoints and the best overall performance among the evaluated methods. Significance analyses supported these improvements in the vast majority of endpoint-baseline comparisons after multiple-testing correction. Systematic ablation experiments demonstrated the complementary contributions of motif-level token contextualization, multimodal integration, and the twostage pretraining strategy. When extended to drug-target interaction prediction, MoTIF-X generalized to an external drug-cold-start dataset without additional fine-tuning and achieved the best average classification performance across standard and generalization-oriented benchmarks. Furthermore, MoTIF-X supported chemically grounded interpretation at the substructure level: motif attribution scores were positively associated with experimentally measured activity variation, and the model preferentially assigned high attribution to motifs with larger activity shifts.

Scientific Contribution: MoTIF-X advances multimodal molecular representation learning by moving beyond the late-stage alignment of modality-specific embeddings and instead using graph-grounded chemical motifs as cross-modal integration anchors and interpretable molecular units. By combining hierarchical pretraining across chemical scales with the joint contextualization of motif, SMILES, and 3D conformational tokens, MoTIF-X provides a domaingrounded token space for molecular property prediction, drug-target interaction modeling, and substructure-level attribution. Controlled ablations further support the contributions of motif-grounded token organization and cross-modal contextualization beyond the inclusion of additional downstream modalities alone.

Keywords: Molecular representation learning, Multimodal integration, Token-based modeling, Chemical interpretability, drug-target interaction prediction, ADMET modeling

## 1 Introduction

Molecular representations provide a computational basis for computer-aided drug discovery by encoding chemical structures for predictive modeling, molecular analysis and design [41]. Recent advances in artificial intelligence have broadened the use of molecular representations beyond task-specific prediction to applications in therapeutic discovery and molecular design [44]. The quality of these representations determines a model’s ability to capture chemical properties, generalize across chemical space, and transfer knowledge across tasks. Despite substantial progress, learning representations that are expressive, broadly transferable, and interpretable remains challenging, reflecting the complexity of molecular systems and the need for more unified strategies for organizing molecular information.

This challenge arises in part because molecular structure is described through multiple complementary computational views. Two-dimensional (2D) molecular graphs encode topological connectivity, whereas fragment- or motif-level representations (used interchangeably in this work to denote functional substructures) capture intermediate chemical semantics between atoms and whole molecules. Three-dimensional (3D) conformations describe spatial geometry [15], and simplified molecular-input line-entry system (SMILES) strings provide a symbolic, sequence-based encoding of molecular structure [26]. Other representations, including molecular fingerprints [46], experimental measurements and textual annotations, further enrich molecular characterization.

Computational models have been developed to exploit these molecular descriptions individually, including graph-based models for topological connectivity, geometryaware models for 3D structure and sequence-based models for SMILES encodings [13, 34, 35]. More recent multimodal approaches have begun to integrate complementary molecular views through graph-geometry alignment, SMILES-graph modeling and the incorporation of motif-level or domain-informed abstractions into graph-based molecular representations [21, 39, 42, 48].

Despite this progress, several limitations remain. Many existing multimodal approaches learn modality-specific representations independently and align them only at a later stage. This paradigm embeds diferent molecular descriptions into separate representation spaces, limiting fine-grained integration of heterogeneous molecular information and joint modeling of chemically coupled features. Moreover, molecular representation frameworks are commonly developed and evaluated for molecular-only tasks and are not explicitly designed to integrate chemical and biological information. Extending a shared molecular representation to cross-domain tasks such as drugtarget interaction (DTI) prediction while retaining substructure-level interpretability therefore remains challenging.

To address these limitations, we present MoTIF-X, a motif-centered token integration framework for constructing interpretable and extensible molecular representations. Rather than treating molecular views as independently encoded representations that are aligned only after separate learning, MoTIF-X introduces graph-grounded motif tokens as structural anchors that connect atomic-level chemistry with molecularlevel context. The framework first uses hierarchical contrastive learning to ground motif embeddings across atomic, motif and molecular scales. These graph-grounded embeddings are then used to generate motif tokens, which guide the integration of SMILES-derived and 3D conformational tokens within a shared Transformer sequence. This motif-centered design enables integrated molecular representations while preserving substructure-level interpretability.

MoTIF-X achieves the best overall performance among the evaluated methods on OpenADMET ExpansionRx, with additional molecular property evaluations on MoleculeNet. It also transfers efectively to DTI prediction under both standard and generalization-oriented settings. Controlled ablations demonstrate the complementary contributions of motif-level token contextualization, multimodal integration, and the two-stage pretraining strategy. The motif-based formulation further supports interpretable analysis, with motif-level attribution capturing experimentally associated activity variation at the substructure level. Together, these findings demonstrate the utility of MoTIF-X as a unified, transferable, and interpretable framework for computational molecular modeling.

## 2 Results

## 2.1 Overview of MoTIF-X

MoTIF-X is organized around graph-grounded motif tokens that provide structural anchors for integrating SMILES-derived and geometric information. The framework consists of two pretraining stages, followed by downstream applications to molecular property prediction and DTI modeling (Fig. 1).

The first pretraining stage of MoTIF-X focuses on grounding motif representations across chemical scales. Each molecule is decomposed into a molecular graph and a corresponding fragment graph, where fragments represent chemically meaningful substructures mined from compound libraries (Fig. 1a). Hierarchical contrastive learning is then applied at both local and global levels. At the local level, atom-level representations pooled within each fragment are aligned with the corresponding fragment representation. At the global level, molecule-level representations derived from molecular and fragment graphs are aligned to promote consistency between local composition and overall molecular context. Together, these objectives enable motif representations to capture intermediate chemical semantics that connect atomic detail with functional substructure context.

The second pretraining stage integrates heterogeneous molecular information into a unified token-based representation for joint modeling (Fig. 1b). Motif tokens combine motif-pooled atom-level representations from the pretrained molecular graph encoder with corresponding fragment-level representations, yielding tokens grounded in both atomic detail and motif semantics. In addition to motif tokens, SMILES strings and 3D conformational information are represented as sequence-based tokens using a SMILES tokenizer and torsion angle discretization, respectively. All tokens are then jointly processed by a Transformer using a masked token modeling objective [10]. This stage contextualizes graph-grounded motif tokens together with SMILES and 3D torsion tokens, enabling the Transformer to learn cross-modal dependencies among topological, symbolic and conformational molecular descriptions.

a Pretraining Stage I: Hierarchical Contrastive Learning  
b Pretraining Stage II: Multimodal MLM  
![](images/3084d76841455c16c1fdc40222a31b42b5d47359d807cd28670f02835a620730.jpg)  
Fig. 1 Overview of the MoTIF-X framework. a Pretraining Stage I: hierarchical contrastive learning grounds motif representations by aligning atom-, motif-, and molecule-level representations using molecular and fragment graphs, enabling motif representations to bridge atomic detail and molecular context. b Pretraining Stage II: multimodal tokenization converts motif, SMILES, and 3D torsion information into tokens, which are jointly modeled using a Transformer with a masked token modeling objective to integrate topological, symbolic, and conformational molecular information. c Downstream molecular property prediction by fine-tuning the pretrained Transformer using motif and SMILES tokens. d Downstream drug-target interaction prediction by jointly modeling molecular motif tokens and pretrained protein representations, with attention over motif tokens indicating the substructures emphasized by the model.

The learned representations are applied to downstream molecular property prediction and DTI modeling (Fig. 1c,d). For molecular property prediction, motif and SMILES tokens are used during fine-tuning, avoiding explicit 3D conformations that are often computationally expensive to obtain while retaining geometric information captured during pretraining. For DTI prediction, molecular motif tokens are integrated with pretrained protein language representations [12] and jointly modeled by the Transformer to capture cross-domain interactions. Beyond predictive performance, the motif-based formulation supports substructure-level interpretation, as attention over motif tokens highlights the substructures emphasized by the model during prediction.

## 2.2 Molecular Property Prediction Benchmark

Molecular property prediction serves as a standard testbed for evaluating the quality and generality of molecular representations. Within this setting, we evaluate MoTIF-X on OpenADMET ExpansionRx [29], covering nine ADME regression endpoints, and compare against representative approaches spanning fingerprint-based methods [32], graph neural networks (GNNs) [13, 45], pretrained molecular models [21, 24, 48], SMILES-based Transformers [33], and 3D-aware molecular models [49]. We also evaluate MoTIF-X on MoleculeNet classification and regression benchmarks [43] to provide supplementary comparisons across a broader range of molecular properties (Supplementary Section A.1). For each benchmark, all methods use identical training, validation, and test partitions, following the evaluation protocol described in Section 5.4. The relationship between each baseline and MoTIF-X, along with implementation details, is summarized in Supplementary Section C.1 of Additional file 1.

![](images/3893b9a76e357095948a40776f9ff6da0ae0a06de3593d4ddbc2708536dd0f81.jpg)

![](images/bd31fe718a5c299f2db4637feeab4379ccd1e11aa83e2146a13a2ba82f0442f5.jpg)

![](images/9871d2b2898192e15cb093ce3c12be6b6eb312074e53719bc85f80a572d12242.jpg)

![](images/1bdc7478981505d55775bcd9382f0beca2e2c4e2079093b0a115a07fe6e442ce.jpg)  
Fig. 2 MoTIF-X achieves strong molecular property prediction performance and benefits from two-stage multimodal pretraining on OpenADMET ExpansionRx. a-b, Comparison with nine baseline models. MoTIF-X achieves the lowest mean MAE on all nine endpoints and the lowest MA-RAE among the compared methods. a, Mean absolute error (MAE) across nine ADME endpoints. b, Overall performance measured by macro-averaged relative absolute error (MA-RAE). c-d, Comparison of modality combinations and stage-wise pretraining ablations. c, MA-RAE across pretraining configurations. d, Endpoint-specific MAE across the same configurations.

Across the nine endpoints, MoTIF-X achieves the lowest mean absolute error (MAE) among all evaluated methods (Fig. 2a; Supplementary Table 1). It also achieves the best overall performance, with a macro-averaged relative absolute error (MA-RAE) of 0.626, compared with 0.687 for CheMeleon [5], the strongest baseline by this metric (Fig. 2b).

The magnitude of improvement over the strongest baseline varies across tasks. Larger relative reductions are observed on endpoints such as LogD and MLM CLint, with smaller numerical diferences on KSOL, MPPB, and MBPB. Significance analyses further support these improvements, showing significantly lower MAE for MoTIF-X in the vast majority of endpoint-baseline comparisons after Holm correction (Supplementary Table 2). Together, these results demonstrate a broad predictive advantage across the benchmark, with gains that vary in magnitude and statistical significance across endpoints and comparators.

## 2.3 Benefits of Two-Stage Multimodal Pretraining for Molecular Property Prediction

Given this strong performance, we next examine which aspects of the MoTIF-X pretraining design contribute to these gains. We compare pretraining configurations that disentangle the efects of individual molecular views, their combinations, and the two pretraining stages on OpenADMET ExpansionRx (Fig. 2c,d; Supplementary Section A.2). Overall performance is assessed using MA-RAE, with endpoint-specific MAEs providing a more detailed comparison. For configurations involving Stage II pretraining, downstream inputs match those used during pretraining, except that 3D conformational tokens are excluded during fine-tuning and inference because reliable conformations are often unavailable in downstream datasets.

In the following comparisons, S, M, and 3D denote SMILES, motif, and torsionangle modalities, respectively, and their combinations indicate the modalities used during pretraining. $S { + } M { + } 3 D$ represents the complete MoTIF-X configuration. To assess the benefit of retaining multiple motif tokens, M (MLP) pools all motif representations into a single molecular representation processed by a multilayer perceptron (MLP). In contrast, M retains individual motif tokens for Transformer contextualization.

We first assess whether motif-centered tokenization provides a stronger basis for downstream molecular modeling than SMILES-based pretraining. Pretraining with

SMILES tokens alone (S) yields limited performance, and adding 3D conformational information in this SMILES-only setting (S+3D) leads to only modest overall improvement. By contrast, both motif-based settings achieve better overall performance, supporting the value of motif-level representations for downstream learning relative to SMILES-based alternatives. Beyond motif extraction, retaining motifs as individual tokens provides a further benefit: M outperforms M (MLP) overall and on most endpoints, supporting the use of token-level contextualization.

Building on the motif-centered setting, we next ask whether incorporating complementary molecular views yields further gains. Relative to M, all multimodal variants, including M+3D, S+M, and S+M+3D, achieve stronger overall performance. To assess whether these gains can be explained by the inclusion of additional downstream inputs alone, we introduce Stage I only as a control. This configuration uses the same motif and SMILES inputs as the complete MoTIF-X model and initializes the graph encoders through Stage I, but omits Stage II pretraining; the token embeddings, Transformer, and prediction head are therefore randomly initialized before downstream fine-tuning. Stage I only performs at a similar level to M, suggesting that the addition of SMILES inputs alone does not account for the observed gains. In contrast, the multimodal variants pretrained during Stage II achieve better overall performance than both configurations. Together, these comparisons support the contribution of joint multimodal pretraining beyond simply increasing the number of downstream input modalities.

We next assess whether Stage II alone can recover these benefits without the graph-grounded initialization provided by Stage I. In the Stage II only configuration, multimodal Stage II pretraining is performed without Stage I: motif tokens are generated by randomly initialized, frozen graph encoders, while the token embeddings and Transformer are trained. Stage II only achieves better overall performance than S+3D but remains below M, with the same ordering observed on most endpoints. This pattern suggests that motif decomposition contributes useful structural information even without Stage I initialization, but that graph-grounded initialization helps the model make more efective use of motif-token representations.

Overall, the strongest performance arises from the full MoTIF-X design, which combines motif-centered token modeling with multimodal information integration: Stage I establishes graph-grounded motif representations, while Stage II further contextualizes them with SMILES and 3D information to improve downstream property prediction.

## 2.4 Multimodal Pretraining Reshapes Attention and Representation Geometry

To characterize how multimodal pretraining alters molecular representations, we analyze token-level attention during pretraining and examine downstream attention patterns and learned [CLS] embeddings on MoleculeNet benchmarks (Fig. 3; Supplementary Section D). During pretraining, we quantify the incoming attention received by each token type from diferent source token types. Without 3D information (MoTIF-X (S+M)), attention is largely concentrated on SMILES tokens, indicating greater attention allocation to SMILES-derived sequence information. Incorporating 3D tokens (MoTIF-X (S+M+3D)) redistributes attention toward motif and 3D sources, reducing the dominance of SMILES self-attention and promoting more balanced interactions among motif, SMILES and 3D tokens (Fig. 3b).

These efects persist during downstream prediction. The attention received by the [CLS] token, which provides an attention-based view of token contributions to the final prediction, shifts toward motif tokens and away from SMILES tokens across tasks when the model is pretrained with 3D information (Fig. 3a). This pattern is consistent with geometric information acquired during pretraining influencing the downstream utilization of motif-level representations.

c  
![](images/a8ce2141c6874ded013afa903d1a585a6ef4c9476c9eca90c219ad78005ae15a.jpg)

![](images/bed1030afd2f1a802d8abbdd7a1cb6d4ee9fb6e1f5bf495b84c103c372edf27b.jpg)

![](images/a209307bb9c2c8bb37362e01a9e43990b52f978641374954f9d26816ad5bc611.jpg)

![](images/8bb204fdde55fae9f06bfbb8a4f701e79ad71d5a05bfe11522bee72f641a7bd8.jpg)

![](images/b99bb789f8d5c7b04d671f7f9c5a8d883e1019b3657cfd0e61d00d79a7779115.jpg)

![](images/bde285bf89c2c22d073110845a3f0a56fe2261b148d6a5b39cd3ac7377852e91.jpg)

![](images/e6f88c11927ff80ef0611b1239c7b65dfda7590aefc7cdae5497adc5eb138fc5.jpg)

![](images/a75a98a18b90edbb2a25150726ea6b8b128c94c921758095df090b8e703c04b4.jpg)

![](images/1e45de3d9fd6763867a8b3a20af16fd38aadabebe1c4cc5d048db9e728d7e301.jpg)

![](images/0b179f20e56d5302c9b612c7bef2d2298fb520724fd612920e737f020869e4b0.jpg)  
Fig. 3 Multimodal pretraining redistributes attention across token types and shapes representation geometry in MoTIF-X. a-b, Attention redistribution across token types. Pretraining with 3D tokens shifts downstream attention toward motif tokens and promotes interactions among motif, SMILES, and 3D tokens during pretraining. a, Incoming attention received by the classification token ([CLS]) from motif and SMILES tokens during downstream prediction across MoleculeNet benchmarks. b, Final-layer incoming attention composition for motif, SMILES, and 3D target tokens during pretraining. Top: models pretrained without 3D information. Bottom: models pretrained with 3D information. c-d, Representation geometry induced by multimodal pretraining. Multimodal pretraining yields more structured and separable representations. c, Cluster-level positive rate on BBBP, showing stronger alignment between embedding clusters and downstream labels after pretraining with 3D tokens. d, Uniform manifold approximation and projection (UMAP) of learned [CLS] representations on BBBP. From left to right: motif+SMILES+3D, motif+SMILES, and extended-connectivity fingerprint (ECFP) representations.

We next ask whether this shift in token utilization is reflected in embedding geometry. Using BBBP as an example, cluster-level analysis of CLS embeddings shows that incorporating 3D information yields more compact and better separated clusters than pretraining without 3D. This more organized structure is also more closely aligned with the downstream task: clusters from the 3D-pretrained model show a larger diference in positive-class prevalence between clusters than those from the model without 3D pretraining (Fig. 3c). These cluster-level trends are consistent across most MoleculeNet classification benchmarks (Supplementary Tables 18, 19). The main exception is Toxicology in the 21st Century (Tox21), where weaker cluster-level label separation coincides with relatively lower predictive performance (Supplementary Table 3). UMAP visualization provides a qualitative view of the same pattern, with 3Dpretrained representations forming a more structured manifold and ECFP features showing a more difuse distribution (Fig. 3d).

Together, these analyses indicate that multimodal pretraining reshapes both attention allocation and representation geometry, shifting MoTIF-X toward motif-centered, geometry-enriched representations that better align with downstream prediction signals.

## 2.5 MoTIF-X Extends to Drug-Target Interaction Prediction

Molecular property prediction benchmarks provide a useful yet incomplete evaluation of molecular representations, as they focus primarily on intrinsic properties of individual molecules. To assess interaction-level modeling, we extend MoTIF-X to DTI prediction, where the model must reason over interactions between small-molecule compounds and protein targets. Unlike task-specific DTI architectures that couple separately designed drug and protein encoders through late-stage fusion [8, 17, 19, 27, 30], MoTIF-X represents molecules as chemically meaningful motif-token sequences shared across tasks. In the DTI setting, these molecular motif tokens are jointly modeled with ProtBERT-derived protein representations [12] in a unified Transformer, enabling interaction prediction while preserving the generality and interpretability of the learned molecular representation.

![](images/334e83553ecbeb6efe91e98615b232041dd8aa23e9c5ec51a72d1ba2b09eecc8.jpg)

![](images/c3d2fde812cbfd2092a676ee4903e41ea7eb9b57c6f7d996ed099b33eb8b03c4.jpg)

![](images/e4fe82b21eb8797b242695abfc402223510ce7435df2af2fd037d64e45608e28.jpg)  
Fig. 4 MoTIF-X transfers to drug-target interaction prediction. a, In-domain half-maximal inhibitory concentration $\mathrm { ( I C _ { 5 0 } ) }$ regression on the BindingDB test set, shown as predicted versus measured $\mathrm { p I C } _ { 5 0 }$ values, where $\mathrm { p I C } _ { 5 0 }$ denotes the negative base-10 logarithm of the molar $\mathrm { I C } _ { 5 0 }$ value. Pearson correlation coeficient (PCC) indicates agreement with measured binding afinities. $\mathbf { b } ,$ External evaluation under $^ \mathrm { a }$ drug cold-start setting on a ChEMBL $\mathrm { I C } _ { 5 0 }$ dataset without additional fine-tuning, evaluated by PCC, indicating generalization to previously unseen compounds across datasets. $\mathbf { c } ,$ DTI classification performance across multiple benchmarks, reported as area under the precision-recall curve (AUPR; %) with comparison to baseline methods. $M o l$ pretrain only uses pretrained molecular embeddings, while the Transformer is initialized randomly before downstream fine-tuning and no DTI-specific pretraining is applied. DTI pretrain only initializes the molecular encoders randomly before DTI-specific pretraining, rather than using encoders initialized from molecular pretraining. Single-motif token and Multi-motif tokens both use pretrained molecular embeddings and DTI-specific pretraining, with Multi-motif tokens denoting the standard MoTIF-X formulation that represents each molecule using multiple motif tokens. MoTIF-X achieves the highest average AUPR and performs best on generalization-oriented settings.

We first evaluate MoTIF-X on large-scale DTI activity regression using a BindingDB-based $\mathrm { I C } _ { 5 0 }$ dataset (Sections 5.3, 5.4). On the BindingDB test set, MoTIF-X achieves a Pearson correlation coeficient (PCC) of 0.8858, showing strong agreement with measured binding afinities (Fig. 4a). To assess drug cold-start transfer in an external dataset, we directly evaluate the BindingDB-trained model on a ChEMBL $\mathrm { I C } _ { 5 0 }$ dataset after excluding any ChEMBL compounds appearing in the BindingDB training set. MoTIF-X achieves a PCC of 0.6147 without additional fine-tuning (Fig. 4b), demonstrating measurable cross-dataset transfer to compounds unseen during training. Given these in-domain and external drug cold-start results, this BindingDB-pretrained checkpoint is used as the common initialization for subsequent DTI fine-tuning and model comparisons (Section 5.1.3).

We further evaluate MoTIF-X on widely used DTI classification benchmarks, including the Stanford Biomedical Network Dataset Collection (BIOSNAP) [50], BindingDB [22], and DAVIS [9], together with two BIOSNAP-derived generalization settings, Unseen Drugs and Unseen Targets (Section 5.3; Fig. 4c; Supplementary Table 9). Across these datasets and settings, MoTIF-X is compared with representative DTI baselines spanning convolutional, transformer-based, protein-informed, contrastive, and attention-based architectures [14, 17, 19, 23, 37]. The complete MoTIF-X model achieves the highest average AUPR, with the strongest gains on DAVIS, Unseen Drugs, and Unseen Targets. Statistical comparisons with the baseline methods are provided in Supplementary Table 10. These settings place greater emphasis on generalization, either by evaluating unseen drugs or targets, or, in DAVIS, by requiring discrimination among kinase interactions with similar binding pockets and limited training data [9]. This motivates examining how molecular pretraining, DTI-specific pretraining, and motif-level granularity contribute to these gains.

Controlled DTI variants further clarify the sources of improvement (Fig. 4c). Both Mol pretrain only and DTI pretrain only fall short of the complete MoTIF-X model, showing that the full gains do not arise from molecular pretraining or DTI-specific pretraining alone, but from their combination. Between these two reduced variants, DTI pretrain only generally remains closer to the complete model across benchmarks, indicating the importance of task-aligned interaction pretraining. By contrast, the clearest relative benefit of molecular pretraining appears in the more generalizationoriented Unseen Drugs setting, consistent with its role in drug-side generalization. Finally, Multi-motif tokens outperforms Single-motif token, indicating that preserving motif-level granularity improves drug-protein interaction modeling. Together, these results indicate that MoTIF-X naturally extends to cross-domain DTI prediction, with motif-level tokens supporting both generalization and interaction-aware modeling.

## 2.6 Motif-level Attribution Reflects Experimental Activity Variation

The strong predictive performance of MoTIF-X across the evaluated DTI benchmarks demonstrates its utility for DTI modeling, while its motif-token architecture also supports attribution at the level of chemically meaningful substructures. As individual molecular tokens correspond to motifs, final-layer attention from the learnable [CLS] token to motif tokens provides an attention-based substructure-level attribution sig nal. We use this signal to examine whether model attributions are consistent with experimentally observed activity variation. Specifically, we perform a motif-centered analysis of experimentally measured $\mathrm { p I C } _ { 5 0 }$ values from ChEMBL for signal transducer and activator of transcription 3 (STAT3)-targeting compounds (Supplementary Section E.1) using a STAT3-specific model trained as described in Supplementary Section E.3. For each motif, we calculate a motif-associated activity shift $( \Delta p I C _ { 5 0 } )$ , defined as the diference in mean $\mathrm { p I C } _ { 5 0 }$ between molecules with and without the motif, and compare its magnitude with mean motif attribution. Across all motifs occurring in at least 10 molecules, motif attribution is moderately correlated with the magnitude of the observed activity shift (Spearman $\rho = 0 . 4 4$ for $| \Delta p I C _ { 5 0 } | ,$ ), indicating that motifs associated with larger activity diferences tend to receive higher attribution scores.

We next examine whether motifs associated with larger experimental activity differences are preferentially prioritized within individual molecules. Motifs are first ranked globally by $| \Delta p I C _ { 5 0 } |$ , and diferent fractions of the top-ranked motifs are selected. For each ratio, we compute a Top-k hit rate, defined as the proportion of selected motif-molecule pairs for which the motif is ranked among the top-k motifs within the corresponding molecule according to model attribution (Fig. 5a). Stronger enrichment is observed at lower selection ratios, indicating that motifs with larger experimentally observed activity shifts are more likely to receive high attribution scores. At a 0.05 selection ratio, the Top-1 hit rate is enriched by approximately 2.55× relative to random selection, and the Top-2 hit rate shows a consistent enrichment trend. Consistent with this enrichment analysis, the top 20 motifs ranked by $| \Delta p I C _ { 5 0 } |$ also receive high attribution scores and are frequently prioritized within individual molecules (Fig. 5b; Supplementary Section E.4). These motifs include both activityenhancing and activity-reducing substructures, indicating that MoTIF-X highlights motifs associated with both increased and decreased binding activity.

Together, these results show that motif-level attribution in MoTIF-X is consistent with experimentally observed activity variation. To further assess the consistency of attention-based attribution, we compare it with Input×Gradient [36], a commonly used gradient-based attribution method. The two approaches show strong agreement across motif instances in the STAT3 dataset (Pearson $r ~ = ~ 0 . 8 8 1$ ; Supplementary Section E.2), demonstrating methodological agreement between the two attribution approaches. These findings motivate further examination of whether attribution signals can yield chemically informative insight into drug-target interactions in a disease-relevant setting.

## 2.7 Motif-level Attribution Highlights Chemically Informative Substructures in Drug-Target Interaction Prediction

We next examine whether motif-level attribution can highlight chemically informative substructures in a disease-relevant DTI setting. This setting is particularly relevant to practical drug discovery, where predictive accuracy alone is often insuficient for rational lead optimization and attribution can help identify chemical motifs associated with predicted interactions [2]. We therefore use a case study to assess whether MoTIF-X identifies chemically plausible motif-level attribution patterns and whether these patterns recur across related compounds and a larger chemical space.

![](images/1e5558f4e8ef16c3a011e6e876664a05e99113e7697f04968df3ee0309cbef26.jpg)

![](images/f87dd9bb361b5fd632a2d892fdb9715e11f8e4cb05fed10a6c5d01cf466b75e9.jpg)

![](images/671bc2e5d2fd2ef38e83b19eb4013cd20dd9a3a2f2d8d285fa02ee17fecc626a.jpg)

d  
![](images/bad3f4f36fb7925f3fbe07ee9398e7af643d0f7e6154fc71adbf33085a04585e.jpg)

![](images/847e95b1e0955eee9c49eccbecc14ab2a174a8f3257fd3d834d93d0a608739c9.jpg)

![](images/e1fd60fecc365147f38d73ddd9e5fecde113334735b4a55e5abbf839c06dee55.jpg)  
Fig. 5 Motif-level attribution in MoTIF-X aligns with activity variation and highlights chemically informative substructures. a-b, Motif-level attribution aligns with experimental activity variation, with activity-relevant motifs receiving higher attribution and being preferentially ranked by the model. a, Top-k hit rate versus motif selection ratio. Higher values at lower ratios indicate that the most activity-relevant motifs are preferentially ranked. b, Examples of top-ranked motifs plotted by activity change $\left( \Delta p I C _ { 5 0 } \right)$ and motif attribution. MoTIF-X assigns high attribution to motifs associated with both increased and decreased activity and frequently prioritizes them within molecules. c-e, Motif-level attribution identifies localized and consistent substructure-level signals across structurally diverse hepatocellular carcinoma (HCC)-related compounds, with the Salicylamide motif highlighted in orange-red and consistently receiving high attribution. c, Niclosamide against STAT3. d-e, Additional HCC-related compounds showing a similar attribution pattern despite structural variation. f, Structural plausibility of motif-level attribution examined by molecular docking. The top-ranked predicted Niclosamide-STAT3 pose yields a favorable docking score, with predicted protein-ligand contacts involving the Salicylamide motif.

The case study centers on Niclosamide, a clinically approved anthelmintic previously implicated in transcriptomic and systems-level studies of hepatocellular carcinoma (HCC) [6, 7, 20]. To place the analysis in a biologically informed context, we evaluate it against a curated HCC-associated pathway protein set rather than the full proteome (Supplementary Section F.3). Within this protein space, MoTIF-X produces a selective target profile for Niclosamide, with STAT3 and inhibitor of nuclear factor kappa-B kinase subunit beta (IKBKB) emerging as the top-ranked predicted targets and showing higher predicted interaction strengths than other pathway proteins. For the Niclosamide-STAT3 pair, MoTIF-X predicts a strong interaction $\mathrm { ( p I C _ { 5 0 } }$ $= 6 . 5 6 ; \mathrm { I C } _ { 5 0 } \approx 0 . 2 8 \ \mu \mathrm { M } ;$ Fig. 5c). The Salicylamide motif receives the highest motiflevel attribution, identifying it as the substructure most strongly emphasized by the model for this prediction. Additional HCC-related compounds [1, 28] show a similar model-attribution pattern: despite broader structural variation, MoTIF-X again assigns high attribution to the Salicylamide motif in compounds predicted to interact with STAT3 (Fig. 5d,e).

To test whether this pattern extends to a broader chemical space, we computationally screen a 430k-molecule ChEMBL diversity subset [25] against STAT3. The predicted interaction scores show a broad but selective distribution, with high-scoring compounds concentrated in a small fraction of the library (Supplementary Section F.1). Motif enrichment analysis of the top-ranked molecules identifies 798 unique motifs across the dataset, among which the Salicylamide motif ranks second by enrichment score (Supplementary Section F.2). This indicates that the motif highlighted in the Niclosamide case is not isolated but forms part of a broader structural pattern enriched among top-ranked compounds in the screening, consistent with MoTIF-X capturing recurrent substructure-level signals associated with predicted STAT3 interactions.

We finally examine whether the highlighted motif is structurally plausible in a predicted binding pose. Molecular docking with AutoDock Vina [11] yields a topranked Niclosamide-STAT3 pose with a docking score of −6.702 kcal/mol. In this pose, the predicted protein-ligand contacts involve atoms within the Salicylamide motif (Fig. 5f; Supplementary Sections F.4, F.5). This result provides a complementary structure-based consistency check for the motif-level attribution produced by MoTIF-X.

Taken together, these results show that MoTIF-X identifies motif-level attribution patterns that are consistent across individual compounds, structurally related candidates, and large-scale chemical libraries. By highlighting substructures that are repeatedly assigned high attribution in predicted drug-target interactions, the model provides a structure-grounded view of the chemical motifs it emphasizes during prediction. This motif-centered view may therefore help prioritize substructures for subsequent experimental evaluation during lead optimization and analog design.

## 3 Discussion

This study proposes MoTIF-X, a motif-centered shared-token framework for multimodal molecular representation learning. MoTIF-X uses graph-grounded motif tokens as structural anchors for jointly contextualizing motif, SMILES, and 3D torsion information within a shared Transformer, and extends this formulation to DTI prediction by integrating molecular motif tokens with protein representations. The framework was evaluated on OpenADMET ExpansionRx and DTI benchmarks, with supplementary molecular property comparisons on MoleculeNet. Additional analyses included controlled ablations, attention and representation geometry, motif-level attribution, and an HCC-focused computational case study. Together, these results show that structurally grounded tokenization can improve predictive performance, support transfer across task settings, and enable chemically interpretable analysis.

Across these evaluations, performance gains arise not simply from adding more modalities, but from organizing molecular information at an appropriate structural granularity. Motif-level tokens provide an intermediate scale between atomic detail and whole-molecule embeddings, preserving compositional structure while enabling explicit substructure interactions. The ablations further show that both pretraining stages contribute to this process: hierarchical contrastive pretraining grounds motifs in atomic and molecular context, whereas multimodal masked token modeling contextualizes these motifs with SMILES and 3D torsion tokens. This combination supports robust molecular property prediction and extends naturally to DTI modeling, where protein binding often depends on specific combinations of functional groups rather than a single global molecular summary.

The attention and representation analyses further clarify how this organization changes multimodal learning. Rather than combining independently formed modalityspecific representations only at a final stage, MoTIF-X allows heterogeneous molecular views to interact within a shared token sequence. The observed redistribution of attention across token types and the accompanying changes in [CLS] embedding geometry indicate that multimodal pretraining alters both attention allocation and representation organization. Together with the ablation results, these observations support the view that the benefits of MoTIF-X arise from learned contextualization among motif, SMILES, and 3D information rather than from simply adding or aligning separate molecular views.

Despite these strengths, several limitations merit consideration. First, although geometric information is incorporated during pretraining, explicit 3D representations are not required at inference time. While this improves practical applicability, tasks that depend strongly on spatial configuration, such as energetic and thermodynamic property prediction [31], remain to be systematically evaluated in settings that incorporate explicit 3D information during downstream inference. Second, although motif-level attribution and attention analyses provide interpretable signals, these explanations remain model-derived and do not constitute direct mechanistic evidence of molecular binding. The STAT3 attribution analysis, virtual screening, and molecular docking therefore provide computational consistency evidence rather than experimental confirmation of binding afinity or mechanism. Prospective biochemical and structure-activity experiments will be required to validate the highlighted targets and substructures.

Looking forward, the token-based formulation of MoTIF-X ofers a flexible foundation for extension. Additional modalities, such as textual annotations, physicochemical descriptors, or other structured signals, could be incorporated within the same token space without fundamental redesign. The motif-centered representation may also support generative and molecular modification tasks, including motif-level editing, controllable generation, and structure-guided optimization.

## 4 Conclusions

In this work, we introduced MoTIF-X, a motif-centered framework that uses graphgrounded chemical motifs to integrate molecular graphs, SMILES strings, and 3D conformational information within a shared Transformer representation. MoTIF-X achieved consistently strong performance across molecular property prediction and drug-target interaction tasks, including external and generalization-oriented evaluations. Controlled ablations indicated that the performance gains arose from motifcentered organization and cross-modal contextualization rather than from the addition of modalities alone. Motif-level attribution was also associated with experimentally observed activity variation, linking model predictions to chemically meaningful substructures. Together, these findings demonstrate how heterogeneous descriptions of a molecular system can be organized within a shared, domain-grounded token space while preserving transferability and chemical interpretability.

## 5 Methods

## 5.1 Model architecture

MoTIF-X consists of two sequential pretraining stages followed by task-specific finetuning. The molecular and fragment encoders are implemented using Graph Isomorphism Networks (GINs) [45], with five and two message-passing layers, respectively, and a hidden dimension of 300. Molecular graphs use categorical physicochemical and stereochemical atom and bond features, whereas fragment graphs use motif-vocabulary identifiers as node features and untyped bidirectional adjacency edges. A six-layer Transformer with six attention heads and the same hidden dimension processes the heterogeneous token sequence. Stage I aligns atomic, motif, and molecular representations through contrastive learning, whereas Stage II performs multimodal masked token modeling.

## 5.1.1 Stage I: Hierarchical Contrastive Pretraining

The local objective grounds motif representations in their constituent atoms, whereas the global objective aligns fragment-based and atom-level molecular representations.

## Molecule Fragmentation

To construct the motif graph used in Stage I pretraining, we apply the Principal Subgraph Mining algorithm [18, 24] to derive a motif vocabulary from the Stage I pretraining molecules. The resulting vocabulary is fixed for all subsequent pretraining and downstream datasets.

Using this vocabulary, each molecule is partitioned into disjoint motifs that collectively cover the molecular graph. A fragment graph is then constructed in which each node represents a motif, and two nodes are connected when their corresponding motifs are adjacent in the original molecular graph. The resulting fragment graph is used as input to the fragment-level encoder.

## Local-level Contrastive Learning

A local-level contrastive objective is employed to align atom-level and fragment-level representations, encouraging each motif embedding to preserve the chemical information of its constituent atoms. The atom-level representation is computed by $\mathrm { G N N _ { M } }$ on the molecular graph, while the fragment-level representation is computed by GNN<sub>F</sub> on the fragment graph.

Given a molecular graph $G _ { M } = ( V _ { M } , E _ { M } )$ and its fragment graph $G _ { F } = ( V _ { F } , E _ { F } )$ node-level atom embeddings and fragment-node embeddings are computed as

$$
{ \bf h } _ { \mathrm { a t o m } } = \mathrm { G N N } _ { \mathrm { M } } ( V _ { M } , E _ { M } ) ,\tag{1}
$$

$$
{ \bf h } _ { \mathrm { f r a g } } = \mathrm { G N N } _ { \mathrm { F } } ( V _ { F } , E _ { F } ) .\tag{2}
$$

Atom embeddings are aggregated within each fragment using mean pooling based on the atom-to-fragment mapping:

$$
\mathbf { h } _ { \mathrm { f , a t o m } } = \mathrm { M E A N P O O L } ( \mathbf { h } _ { \mathrm { a t o m } } , \mathrm { a t o m - t o - f r a g m e n t ~ m a p p i n g } ) .\tag{3}
$$

Both $\mathbf { h } _ { \mathrm { f , a t o m } }$ and $\mathbf { h } _ { \mathrm { f r a g } }$ are projected into a shared latent space via a local projection head $f _ { \mathrm { l o c a l } } ( \cdot )$ (a two-layer MLP) followed by $\ell _ { 2 }$ normalization:

$$
\mathbf { z } _ { \mathrm { f , a t o m } } = \frac { f _ { \mathrm { l o c a l } } ( \mathbf { h } _ { \mathrm { f , a t o m } } ) } { \| f _ { \mathrm { l o c a l } } ( \mathbf { h } _ { \mathrm { f , a t o m } } ) \| _ { 2 } } ,\tag{4}
$$

$$
{ \bf z } _ { \mathrm { f , l o c a l } } = \frac { f _ { \mathrm { l o c a l } } ( \bf h _ { \mathrm { f r a g } } ) } { \Vert f _ { \mathrm { l o c a l } } ( \bf h _ { \mathrm { f r a g } } ) \Vert _ { 2 } } .\tag{5}
$$

Negative samples are drawn from other fragments within the same mini-batch. The local contrastive loss is defined as

$$
\mathcal { L } _ { \mathrm { l o c a l } } = - \frac { 1 } { B _ { f } } \sum _ { i = 1 } ^ { B _ { f } } \log \frac { \exp \left( \langle \mathbf { z } _ { \mathrm { f , a t o m } } ^ { ( i ) } , \mathbf { z } _ { \mathrm { f , l o c a l } } ^ { ( i ) } \rangle / \tau \right) } { \sum _ { j = 1 } ^ { B _ { f } } \exp \left( \langle \mathbf { z } _ { \mathrm { f , a t o m } } ^ { ( i ) } , \mathbf { z } _ { \mathrm { f , l o c a l } } ^ { ( j ) } \rangle / \tau \right) } ,\tag{6}
$$

where $B _ { f }$ is the number of fragments in the mini-batch.

## Global-level Contrastive Learning

A global-level contrastive objective is defined to align molecule-level representations derived from atom-level and fragment-level embeddings, encouraging the fragmentbased representation to capture how local motifs are organized within the complete molecular graph.

Molecule-level representations are obtained via mean pooling:

$$
{ \bf h } _ { \mathrm { m o l , a t o m } } = \mathrm { M E A N P O O L ( { \bf h } _ { a t o m } , a t o m - t o - m o l e c u l e ~ m a p p i n g ) } ,\tag{7}
$$

$$
\mathbf { h } _ { \mathrm { m o l , f r a g } } = \mathrm { M E A N P O O L } ( \mathbf { h } _ { \mathrm { f r a g } } , \mathrm { f r a g m e n t - t o - m o l e c u l e ~ m a p p i n g } ) .\tag{8}
$$

These representations are projected into a shared latent space using a global projection head $f _ { \mathrm { g l o b a l } } ( \cdot )$ (a two-layer MLP) followed by $\ell _ { 2 }$ normalization:

$$
\mathbf { z } _ { \mathrm { m o l , a t o m } } = \frac { \mathit { f } _ { \mathrm { g l o b a l } } ( \mathbf { h } _ { \mathrm { m o l , a t o m } } ) } { \| f _ { \mathrm { g l o b a l } } ( \mathbf { h } _ { \mathrm { m o l , a t o m } } ) \| _ { 2 } } ,\tag{9}
$$

$$
{ \bf z } _ { \mathrm { m o l , f r a g } } = \frac { f _ { \mathrm { g l o b a l } } ( \bf h _ { \mathrm { m o l , f r a g } } ) } { \| f _ { \mathrm { g l o b a l } } ( \bf h _ { \mathrm { m o l , f r a g } } ) \| _ { 2 } } .\tag{10}
$$

Negative samples are drawn from other molecules within the same mini-batch. The global contrastive loss is defined as

$$
\mathcal { L } _ { \mathrm { g l o b a l } } = - \frac { 1 } { B _ { m } } \sum _ { i = 1 } ^ { B _ { m } } \log \frac { \exp \left( \langle \mathbf { z } _ { \mathrm { m o l , a t o m } } ^ { ( i ) } , \mathbf { z } _ { \mathrm { m o l , f r a g } } ^ { ( i ) } \rangle / \tau \right) } { \sum _ { j = 1 } ^ { B _ { m } } \exp \left( \langle \mathbf { z } _ { \mathrm { m o l , a t o m } } ^ { ( i ) } , \mathbf { z } _ { \mathrm { m o l , f r a g } } ^ { ( j ) } \rangle / \tau \right) } ,\tag{11}
$$

where $B _ { m }$ is the number of molecules in the mini-batch.

The overall Stage I objective is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S t a g e ~ I } } = \mathcal { L } _ { \mathrm { l o c a l } } + \lambda \mathcal { L } _ { \mathrm { g l o b a l } } , } \end{array}\tag{12}
$$

where $\lambda \in \mathbb { R } _ { + }$ controls the relative contribution of the global objective. In practice, both the temperature τ and the global-loss weight λ are set to 0.1 based on preliminary hyperparameter tuning.

Together, these objectives provide a graph-grounded initialization for motifcentered representations before multimodal masked token modeling in Stage II.

## 5.1.2 Stage II: Multimodal Masked Token Modeling

Stage II constructs motif, SMILES, and torsion angle tokens and jointly trains them within a unified Transformer using a masked token modeling objective.

## Motif Tokens

Using the pretrained and frozen graph encoders from Stage I, the token representation of motif k is constructed by combining its pooled atom-level representation with the corresponding fragment-level representation:

$$
\mathbf { h } _ { \mathrm { m o t i f } } ^ { ( k ) } = \mathbf { h } _ { \mathrm { f , a t o m } } ^ { ( k ) } + \mathbf { h } _ { \mathrm { f r a g } } ^ { ( k ) } .\tag{13}
$$

The resulting motif tokens are used as graph-grounded inputs to the Transformer in Stage II.

## SMILES Tokens

SMILES tokens are generated from each canonical SMILES string using a casesensitive greedy longest-match tokenizer. At each position, the longest matching entry in the shared sequence vocabulary is selected, whereas unmatched characters are retained as single-character tokens. The resulting tokens are embedded using a learnable token embedding matrix.

## Torsion Angle Tokens

Torsion angle tokens are constructed from the molecular conformations described in Supplementary Section B.1. For each conformer, all valid dihedral angles associated with rotatable bonds are enumerated as ordered atom quadruples $( i , a , b , l )$ . To ensure a consistent torsion sequence across conformers of the same molecule, the torsions are ordered according to a depth-first search traversal of the molecular graph.

The continuous dihedral angles are computed in radians and discretized over the range $[ - \pi , \pi ]$ using a uniform bin size of 0.01 radians. Each discretized value is mapped to an integer identifier using entries reserved for torsion-angle bins in the sequence-token vocabulary. Although the same vocabulary also includes SMILESsymbol entries, these entries are disjoint from the torsion-angle-bin entries. Thus, the shared vocabulary provides a common indexing and embedding mechanism while keeping SMILES and torsion tokens as separate token types.

The resulting torsion token sequence provides a topology-consistent encoding of molecular conformation. These tokens are embedded using the shared token embedding matrix and combined with fixed sinusoidal positional encodings before being processed jointly with motif and SMILES tokens in the Transformer.

The multimodal input concatenates motif, SMILES, and torsion tokens in this order and is right-padded or right-truncated to a maximum length of 200 before the [CLS] token is prepended.

## Masked Token Modeling Objective

Let $\mathbf { x } ~ = ~ ( x _ { 1 } , \ldots , x _ { N } )$ denote the concatenated motif, SMILES, and torsion token sequence. During training, 15% of non-padding positions are selected as prediction targets, with at least one position selected from each available modality. Selected SMILES and torsion tokens follow the standard 80/10/10 replacement strategy. At selected motif positions, the graph-derived motif embeddings remain as inputs and are used to predict the corresponding motif identities.

A token prediction head is applied to the representations at selected positions to predict the original token identities. For motif tokens, prediction is optimized using a standard cross-entropy loss over the motif vocabulary generated during molecule fragmentation. For SMILES tokens, prediction is optimized using cross-entropy over the SMILES-symbol entries in the sequence-token vocabulary. For torsion tokens, prediction over the torsion-angle-bin entries in the same sequence-token vocabulary is optimized using the Gaussian cross-entropy (GCE) loss described in [40].

The Stage II objective is the sum of the motif cross-entropy, SMILES cross-entropy, and torsion GCE losses over the selected positions.

## 5.1.3 Downstream Fine-tuning

For downstream tasks, the molecular graph encoders and Transformer are fine-tuned end-to-end with task-specific prediction heads. In all settings, a learnable [CLS] token is prepended to the input sequence, and its final hidden representation is used for prediction.

## Molecular Property Prediction

For molecular property prediction, the input consists of motif tokens derived from the 2D molecular graph and SMILES tokens. No torsion tokens are used during finetuning. The graph encoders and Transformer are initialized from the Stage II masked token modeling pretraining and fine-tuned jointly. The final hidden representation of the [CLS] token is passed to a task-specific linear head to predict molecular properties.

## Drug-Target Interaction Prediction

For DTI prediction, each protein sequence is encoded using the pretrained ProtBERT model (Rostlab/prot bert) [12]. Protein sequences longer than 1,022 residues are truncated from the C-terminal end so that the complete input, including two special tokens, does not exceed the maximum length of 1,024 tokens. Final-layer residue embeddings are extracted after excluding special and padding tokens and mean-pooled to obtain a 1,024-dimensional protein representation. The ProtBERT parameters remain frozen during embedding extraction.

The pooled protein representation is projected into the shared 300-dimensional space through a learnable linear layer and introduced as a single protein token alongside the molecular motif tokens. Protein and motif tokens are jointly processed by the Transformer, and the resulting [CLS] representation is used for task-specific prediction.

## 5.2 Training and Implementation Details

Stage I is trained for 100 epochs using AdamW with a learning rate of $1 \times 1 0 ^ { - 3 }$ , a weight decay of $1 \times 1 0 ^ { - 2 }$ , and a batch size of 256. During Stage II, the graph encoders are frozen, and the remaining components are trained for 100 epochs using AdamW with a learning rate of $5 \times 1 0 ^ { - 5 }$ , a weight decay of $1 \times 1 0 ^ { - 2 }$ , and a batch size of 512. A linear learning-rate schedule with 10% warmup is used in Stage II.

For DTI pretraining, the model is trained on BindingDB afinity data for 100 epochs using AdamW without weight decay and a batch size of 512. Mean squared error is used as the training loss. The molecular encoders and protein projection layer use a learning rate of $1 \times 1 0 ^ { - 3 }$ , whereas the Transformer uses $1 \times 1 0 ^ { - 4 }$

All downstream MoTIF-X models are trained for 100 epochs with a batch size of 512 using AdamW without weight decay. Classification and regression models are optimized using binary cross-entropy with logits and mean squared error, respectively. Transformer dropout is set to 0 during Stage II pretraining and regression fine-tuning, 0.2 for molecular classification, and 0.1 for DTI classification. No early stopping is applied to downstream MoTIF-X training.

For OpenADMET regression, checkpoints are selected by the lowest validation MAE. For the supplementary MoleculeNet experiments, checkpoints are selected by the highest validation ROC-AUC for classification and the lowest validation mean squared error for regression. DTI checkpoints are selected by the highest validation AUPR for classification and Pearson correlation for regression.

All MoTIF-X models are trained on a single NVIDIA H100 graphics processing unit. For molecular property fine-tuning, the graph encoders use a learning rate of $5 \times 1 0 ^ { - 4 }$ and the Transformer uses $1 \times 1 0 ^ { - 4 }$ . For DTI classification, the molecular encoders and protein projection layer use $1 \times 1 0 ^ { - 3 }$ , and the Transformer uses $1 \times 1 0 ^ { - 4 }$

## 5.3 Datasets

Additional details and statistics for the multimodal pretraining and DTI regression datasets are provided in Supplementary Section B.

## Multimodal Pretraining Data

Multimodal pretraining is conducted using molecules with 3D conformations from the GEOM-Drugs dataset [3], which contains approximately 304,000 drug-like molecules with multiple conformers. Following prior work [38], the five lowest-energy conformers

are selected for each molecule during pretraining. Each selected conformer is treated as a separate training instance, with the corresponding motif and canonical SMILES tokens shared across conformers.

## Molecular Property Prediction

OpenADMET ExpansionRx [29] is used as the primary benchmark for molecular property prediction. The benchmark comprises nine ADME regression endpoints: LogD, KSOL, HLM CLint, MLM CLint, Caco-2 Papp, Caco-2 Eflux, MPPB, MBPB, and MGMB. Each endpoint is modeled separately using molecules with available labels for that endpoint.

Additional evaluations are conducted on the MoleculeNet benchmarks [43], including BBBP, Tox21, Toxicity Forecaster (ToxCast), Side Efect Resource (SIDER), ClinTox, Maximum Unbiased Validation (MUV), HIV, the beta-secretase 1 inhibitor activity (BACE) dataset, ESOL, and Lipophilicity. These experiments provide supplementary comparisons across classification and regression tasks related to toxicity, biological activity, and physicochemical properties.

## DTI Regression

For DTI regression, we construct an $\mathrm { I C } _ { 5 0 }$ dataset from BindingDB [22]. Dataset construction and preprocessing follow the Therapeutics Data Commons benchmark protocol [16], retaining high-confidence binding measurements for single-protein targets.

Before constructing the pretraining splits, all compound-protein pairs appearing in the downstream DTI classification benchmarks were removed from the BindingDB afinity dataset. Pair identity was determined using the combination of canonical SMILES and target protein sequence. This filtering ensured that the supervised DTI pretraining data were pair-disjoint from all downstream DTI classification benchmarks.

For external evaluation, we use ChEMBL version 35 [47] to construct a drug coldstart dataset of compound-protein pairs with experimentally measured $\mathrm { I C } _ { 5 0 }$ values, which are converted to $\mathrm { p I C } _ { 5 0 }$ . Compounds present in the BindingDB training set are excluded, ensuring that all compounds in the external dataset are unseen during training.

## DTI Classification

For DTI classification, we follow the benchmark datasets and data preparation protocol introduced in MolTrans [17]. Experiments are conducted on the BIOSNAP, BindingDB, and DAVIS datasets, together with the Unseen Drugs and Unseen Targets generalization settings. Dataset construction, train/validation/test splits, and negative sampling procedures strictly follow the protocol described in [17].

## 5.4 Evaluation Protocol

For OpenADMET ExpansionRx, the oficial training and test partitions are used without transferring molecules between them. For each endpoint, 10% of the labeled oficial training partition is randomly sampled as the validation set using a fixed splitting seed of 42, and the remaining samples are used for training. The resulting train, validation, and test assignments are fixed across all compared methods and downstream random seeds. The oficial test partition is not used for model selection.

Label scales and performance metrics follow the oficial OpenADMET ExpansionRx evaluation protocol [29]. LogD is modeled and evaluated on its original scale. For the remaining endpoints, labels are transformed as $\log _ { 1 0 } ( \operatorname* { m a x } ( y , 0 ) + 1 )$ before training. Predictions for these endpoints are clipped to a minimum of zero on the transformed scale during validation and testing. Training predictions are not clipped. Endpoint performance is evaluated using mean absolute error (MAE). Relative absolute error (RAE) is calculated as the MAE divided by the mean absolute deviation of the test labels from their mean. Overall performance is summarized by macro-averaged relative absolute error (MA-RAE), the equally weighted average of RAE across the nine endpoints.

For the supplementary MoleculeNet evaluations, a deterministic, chirality-aware Bemis-Murcko scafold split [4] is used with an 80:10:10 train/validation/test ratio. MoTIF-X and all baseline models use identical training, validation, and test partitions, which remain fixed across all random seeds. Classification and regression performance are evaluated using ROC-AUC and root mean squared error (RMSE), respectively.

For BindingDB afinity regression, compound-protein pairs are randomly partitioned at the pair level into training, validation, and test sets using a 70:10:20 ratio and a fixed random seed of 42. This split does not enforce disjoint compounds or proteins across subsets. The ChEMBL drug-cold-start dataset is used exclusively for external evaluation and contains no compounds present in the BindingDB training set. DTI classification datasets use their predefined benchmark splits. DTI regression is evaluated using PCC. DTI classification is evaluated using AUPR, computed as average precision, following the MolTrans protocol [17].

Model selection is based exclusively on validation performance. The selected checkpoint is evaluated once on the corresponding test set after training. Molecular and DTI pretraining are each performed once, and the resulting checkpoints are reused across five downstream runs with diferent random seeds. Endpoint results are reported as the mean and standard deviation of individual-model scores across these five runs.

## Statistical Comparisons

For statistical comparisons, all methods are trained using the same set of five random seeds. Predictions from these five downstream runs are averaged for each test sample to form an ensemble prediction for each method. Test samples are aligned by molecular identity for property prediction and by drug-protein pair identity for DTI classification, with matching reference labels. Statistical comparisons assess diferences between fiveseed ensembles. Performance summaries report the mean and standard deviation of individual-model scores across the same five seeds.

For OpenADMET, two-sided paired sign-flip permutation tests are performed on per-molecule absolute-error diferences between MoTIF-X and each comparator using 10,000 permutations. Holm correction is applied separately to the 81 baseline comparisons across nine endpoints and nine baselines, and the 72 ablation comparisons across nine endpoints and eight configurations.

For DTI classification, diferences in ensemble AUPR between MoTIF-X and each baseline are assessed using two-sided paired stratified bootstrap tests with 10,000 replicates. Positive and negative test samples are resampled separately, and identical resampling indices are applied to both methods. Holm correction is applied jointly to the 25 comparisons across five benchmark settings and five baselines.

Statistical significance is defined as a Holm-adjusted p-value below 0.05. Detailed comparison procedures and results are provided in Supplementary Section A.

## References

[1] Adams JL, Dufy KJ, Graybill TL, et al (2017) 5-sulfamoyl-2-hydroxybenzamide derivatives. International patent WO 2017/153952 A1, assignee: GlaxoSmithKline Intellectual Property Development Ltd

[2] Arrowsmith CH, Audia JE, Austin C, et al (2015) The promise and peril of chemical probes. Nature chemical biology 11(8):536–541

[3] Axelrod S, Gomez-Bombarelli R (2022) Geom, energy-annotated molecular conformations for property prediction and molecular generation. Scientific Data 9(1):185

[4] Bemis GW, Murcko MA (1996) The properties of known drugs. 1. molecular frameworks. Journal of medicinal chemistry 39(15):2887–2893

[5] Burns J, Zalte A, Green W (2025) Descriptor-based foundation models for molecular property prediction. arXiv preprint arXiv:250615792

[6] Chen B, Ma L, Paik H, et al (2017) Reversal of cancer gene expression correlates with drug eficacy and reveals therapeutic targets. Nature communications 8(1):16022

[7] Chen B, Wei W, Ma L, et al (2017) Computational discovery of niclosamide ethanolamine, a repurposed drug candidate that reduces growth of hepatocellular carcinoma cells in vitro and in mice by inhibiting cell division cycle 37 signaling. Gastroenterology 152(8):2022–2036

[8] Chen L, Tan X, Wang D, et al (2020) Transformercpi: improving compound– protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments. Bioinformatics 36(16):4406–4414

[9] Davis MI, Hunt JP, Herrgard S, et al (2011) Comprehensive analysis of kinase inhibitor selectivity. Nature biotechnology 29(11):1046–1051

[10] Devlin J, Chang MW, Lee K, et al (2019) Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp 4171–4186

[11] Eberhardt J, Santos-Martins D, Tillack AF, et al (2021) Autodock vina 1.2. 0: new docking methods, expanded force field, and python bindings. Journal of chemical information and modeling 61(8):3891–3898

[12] Elnaggar A, Heinzinger M, Dallago C, et al (2021) Prottrans: toward understanding the language of life through self-supervised learning. IEEE transactions on pattern analysis and machine intelligence 44(10):7112–7127

[13] Gilmer J, Schoenholz SS, Riley PF, et al (2017) Neural message passing for quantum chemistry. In: International conference on machine learning, Pmlr, pp 1263–1272

[14] Goldman S, Das R, Yang KK, et al (2022) Machine learning modeling of family wide enzyme-substrate specificity screens. PLoS computational biology 18(2):e1009853

[15] Guo Z, Guo K, Nan B, et al (2023) Graph-based molecular representation learning. In: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI ’23

[16] Huang K, Fu T, Gao W, et al (2021) Therapeutics data commons: Machine learning datasets and tasks for drug discovery and development. Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks

[17] Huang K, Xiao C, Glass LM, et al (2021) Moltrans: molecular interaction transformer for drug–target interaction prediction. Bioinformatics 37(6):830–836

[18] Kong X, Huang W, Tan Z, et al (2022) Molecule generation by principal subgraph mining and assembling. Advances in Neural Information Processing Systems 35:2550–2563

[19] Lee I, Keum J, Nam H (2019) Deepconv-dti: Prediction of drug-target interactions via deep learning with convolution on protein sequences. PLoS computational biology 15(6):e1007129

[20] Li Y, Li PK, Roberts MJ, et al (2014) Multi-targeted therapy of cancer by niclosamide: A new application for an old drug. Cancer letters 349(1):8–14

[21] Liu S, Wang H, Liu W, et al (2021) Pre-training molecular graph representation with 3d geometry. arXiv preprint arXiv:211007728

[22] Liu T, Lin Y, Wen X, et al (2007) Bindingdb: a web-accessible database of experimentally determined protein–ligand binding afinities. Nucleic acids research 35(suppl 1):D198–D201

[23] Lu Z, Song G, Zhu H, et al (2025) Dtiam: a unified framework for predicting drug-target interactions, binding afinities and drug mechanisms. Nature Communications 16(1):2548

[24] Luong KD, Singh AK (2023) Fragment-based pretraining and finetuning on molecular graphs. Advances in Neural Information Processing Systems 36:17584– 17601

[25] Mayr A, Klambauer G, Unterthiner T, et al (2018) Large-scale comparison of machine learning methods for drug target prediction on chembl. Chemical science 9(24):5441–5451

[26] Mswahili ME, Jeong YS (2024) Transformer-based models for chemical smiles representation: A comprehensive literature review. Heliyon 10(20)

[27] Nguyen T, Le H, Quinn TP, et al (2021) Graphdta: predicting drug–target binding afinity with graph neural networks. Bioinformatics 37(8):1140–1147

[28] Omran MM, Kamal MM, Ammar YA, et al (2024) Pharmacological investigation of new niclosamide-based isatin hybrids as antiproliferative, antioxidant, and apoptosis inducers. Scientific Reports 14(1):19818

[29] Open ADMET Consortium (2026) openadmet-expansionrx-challenge-data. Hugging Face, https://doi.org/10.57967/hf/9687, URL https://doi.org/10.57967/hf/ 9687, dataset

[30] Ozt¨urk H, <sup>¨</sup> Ozg¨ur A, Ozkirimli E (2018) Deepdta: deep drug–target binding <sup>¨</sup> afinity prediction. Bioinformatics 34(17):i821–i829

[31] Ramakrishnan R, Dral PO, Rupp M, et al (2014) Quantum chemistry structures and properties of 134 kilo molecules. Scientific data 1(1):1–7

[32] Rogers D, Hahn M (2010) Extended-connectivity fingerprints. Journal of chemical information and modeling 50(5):742–754

[33] Ross J, Belgodere B, Chenthamarakshan V, et al (2022) Large-scale chemical language representations capture molecular structure and properties. Nature Machine Intelligence 4(12):1256–1264

[34] Satorras VG, Hoogeboom E, Welling M (2021) E (n) equivariant graph neural networks. In: International conference on machine learning, PMLR, pp 9323–9332

[35] Schwaller P, Laino T, Gaudin T, et al (2019) Molecular transformer: a model for uncertainty-calibrated chemical reaction prediction. ACS central science 5(9):1572–1583

[36] Shrikumar A, Greenside P, Kundaje A (2017) Learning important features through propagating activation diferences. In: International conference on machine learning, PMlR, pp 3145–3153

[37] Singh R, Sledzieski S, Bryson B, et al (2023) Contrastive learning in protein language space predicts interactions between drugs and protein targets. Proceedings of the National Academy of Sciences 120(24):e2220778120

[38] St¨ark H, Beaini D, Corso G, et al (2022) 3d infomax improves gnns for molecular property prediction. In: International Conference on Machine Learning, PMLR, pp 20479–20502

[39] Sun M, Xing J, Wang H, et al (2021) Mocl: data-driven molecular fingerprint via knowledge-aware contrastive learning from molecular graph. In: Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pp 3585–3594

[40] Wang J, Qin R, Wang M, et al (2025) Token-mol 1.0: tokenized drug design with large language models. Nature Communications 16(1):4416

[41] Wigh DS, Goodman JM, Lapkin AA (2022) A review of molecular representation in the age of machine learning. Wiley Interdisciplinary Reviews: Computational Molecular Science 12(5):e1603

[42] Wu T, Tang Y, Sun Q, et al (2023) Molecular joint representation learning via multi-modal information of smiles and graphs. IEEE/ACM transactions on computational biology and bioinformatics 20(5):3044–3055

[43] Wu Z, Ramsundar B, Feinberg EN, et al (2018) Moleculenet: a benchmark for molecular machine learning. Chemical science 9(2):513–530

[44] Xing J, Tan M, Leshchiner D, et al (2026) Deep-learning-based de novo discovery and design of therapeutics that reverse disease-associated transcriptional phenotypes. Cell 189(9):2556–2572

[45] Xu K, Hu W, Leskovec J, et al (2018) How powerful are graph neural networks? arXiv preprint arXiv:181000826

[46] Yang J, Cai Y, Zhao K, et al (2022) Concepts and applications of chemical fingerprint for hit and lead screening. Drug discovery today 27(11):103356

[47] Zdrazil B, Felix E, Hunter F, et al (2024) The chembl database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic acids research 52(D1):D1180–D1192

[48] Zhang Z, Liu Q, Wang H, et al (2021) Motif-based graph self-supervised learning for molecular property prediction. Advances in Neural Information Processing Systems 34:15870–15882

[49] Zhou G, Gao Z, Ding Q, et al (2023) Uni-mol: A universal 3d molecular representation learning framework. In: The eleventh international conference on learning representations

[50] Zitnik M, Sosiˇc R, Maheshwari S, et al (2018) BioSNAP Datasets: Stanford biomedical network dataset collection. http://snap.stanford.edu/biodata

## Abbreviations

<table><tr><td>Abbreviation</td><td>Definition</td></tr><tr><td>2D</td><td>two-dimensional</td></tr><tr><td>3D</td><td>three-dimensional</td></tr><tr><td>ADMET</td><td>absorption, distribution, metabolism, excretion, and toxicity</td></tr><tr><td>AUPR</td><td>area under the precision-recall curve</td></tr><tr><td>BACE</td><td>beta-secretase 1 inhibitor activity dataset</td></tr><tr><td>BBBP</td><td>blood-brain barrier penetration</td></tr><tr><td>BIOSNAP</td><td>Stanford Biomedical Network Dataset Collection</td></tr><tr><td>AbbreviationDefinition</td><td></td></tr><tr><td>[CLS]</td><td>classification token</td></tr><tr><td>ClinTox</td><td>clinical toxicity dataset</td></tr><tr><td>DTI</td><td>drug-target interaction</td></tr><tr><td>ECFP</td><td>extended-connectivity fingerprint</td></tr><tr><td>ESOL</td><td>estimated solubility dataset</td></tr><tr><td>GCE</td><td>Gaussian cross-entropy</td></tr><tr><td>GIN</td><td>Graph Isomorphism Network</td></tr><tr><td>GNN</td><td>graph neural network</td></tr><tr><td>HCC</td><td>hepatocellular carcinoma</td></tr><tr><td>HIV</td><td>human immunodeficiency virus</td></tr><tr><td> $\mathrm { I C } _ { 5 0 }$ </td><td>half-maximal inhibitory concentration</td></tr><tr><td>IKBKB</td><td>inhibitor of nuclear factor kappa-B kinase subunit beta</td></tr><tr><td>MAE</td><td>mean absolute error</td></tr><tr><td>MA-RAE</td><td>macro-averaged relative absolute error</td></tr><tr><td>MLP</td><td>multilayer perceptron</td></tr><tr><td>MUV</td><td>Maximum Unbiased Validation</td></tr><tr><td>PCC</td><td>Pearson correlation coefficient</td></tr><tr><td> $\mathrm { p I C } _ { 5 0 }$ </td><td>negative base-10 logarithm of the molar  $\mathrm { I C } _ { 5 0 }$  value</td></tr><tr><td>RAE</td><td>relative absolute error</td></tr><tr><td>RMSE</td><td>root mean squared error</td></tr><tr><td>ROC-AUC</td><td>area under the receiver operating characteristic curve</td></tr><tr><td>SIDER</td><td>Side Effect Resource</td></tr><tr><td>SMILES</td><td>simplified molecular-input line-entry system</td></tr><tr><td>STAT3</td><td>signal transducer and activator of transcription 3</td></tr><tr><td>Tox21</td><td>Toxicology in the 21st Century</td></tr><tr><td>Abbreviation</td><td>Definition</td></tr><tr><td>ToxCast</td><td>Toxicity Forecaster</td></tr><tr><td>UMAP</td><td>uniform manifold approximation and projection</td></tr></table>

## Declarations

## Availability of data and materials

The source code for MoTIF-X, together with the molecular property and DTI classification benchmark data used in this study, is available at https://github.com/ Bin-Chen-Lab/Motif-X. The OpenADMET ExpansionRx dataset [29] is available at https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data. The GEOM-Drugs molecular pretraining data are available from Harvard Dataverse at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi: 10.7910/DVN/JNGTDF. The BindingDB $\mathrm { I C } _ { 5 0 }$ data were obtained through Therapeutics Data Commons at https://tdcommons.ai/multi pred tasks/dti/. ChEMBL version 35 is available from the ChEMBL release archive at https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl 35/. The STAT3 structure used for molecular docking is available from the Protein Data Bank under accession code 6QHD.

## Competing interests

The authors declare that they have no competing interests.

## Funding

This work was supported by the National Institutes of Health (R01GM145700 and R61HL177451), the National Institute on Aging of the National Institutes of Health

(R01AG072449), the National Science Foundation (IIS-2212174), and the Michigan State University Strategic Partnership Grant.

## Authors’ contributions

L.M., J.Z. and B.C. conceived the study. L.M. developed the MoTIF-X framework, implemented the models, constructed the datasets, performed the experiments, analyzed the results, prepared the figures, and wrote the initial manuscript. J.Z. and B.C. jointly supervised the study, provided conceptual guidance, contributed to the interpretation of the results, reviewed the implementation, and revised the manuscript. All authors read and approved the final manuscript.

Acknowledgements

Not applicable.

## Additional files

File name: Additional file 1.pdf

File format: PDF (.pdf)

Title: Supplementary Information for “MoTIF-X: A Multimodal Tokenized Framework for Interpretable and Extensible Molecular Representation Learning” Description: Supplementary benchmark results, dataset statistics, baseline descriptions, attention and representation analyses, motif-level attribution analyses, libraryscale STAT3 screening, and molecular docking methods and results.