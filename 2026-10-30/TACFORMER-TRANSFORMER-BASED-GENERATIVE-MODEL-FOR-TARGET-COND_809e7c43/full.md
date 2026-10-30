# TACFORMER: TRANSFORMER-BASED GENERATIVE MODEL FOR TARGET-CONDITIONED De Novo PROTAC DESIGN

Yossra Gharbi<sup>1</sup>, Rocío Mercado<sup>1,2,∗</sup>

<sup>1</sup>AI Laboratory for Molecular Engineering (AIME), Department of Computer Science and Engineering, Chalmers University of Technology & University of Gothenburg, Gothenburg, Sweden <sup>2</sup>Science for Life Laboratory (SciLifeLab), Gothenburg, Sweden

<sup>∗</sup>Corresponding author

Abstract Targeted protein degradation (TPD) represents a fundamental shift in drug discovery, moving beyond traditional target inhibition toward the event-driven removal of disease-causing proteins. A prominent strategy for TPD is the use of proteolysis-targeting chimeras (PROTACs), heterobifunctional molecules that selectively route a protein of interest (POI) to the ubiquitin–proteasome system for degradation. While PROTAC design still relies heavily on trial-and-error synthesis and experimental testing, deep generative models offer a promising means to explore PROTAC chemical space in silico and generate novel molecules. However, many existing generative methods formulate PROTAC design primarily as a linker-generation problem, with POI-binding (warhead) and E3-ligase-binding components fixed in advance. This restricts exploration of the full PROTAC chemical space and often leaves the POI implicit through the predefined warhead, rather than using protein-level information as an explicit conditioning signal during generation. To address these limitations, we propose TACFormer: a transformer-based generative model for target-aware de novo PROTAC design. The model conditions a GPT-style molecular decoder on pretrained ESM-2 representations of POI sequences, allowing POIlevel information to steer full-molecule generation. Our results show that TACFormer generates highly novel and chemically diverse PROTACs while showing consistent POI-dependent shifts in generated chemical space. In particular, molecules generated under specific target conditions preferentially populate chemical neighborhoods associated with POIs in the held-out dataset, while extrapolating to novel chemotypes. By natively integrating biological context into the generative process, TACFormer moves PROTAC design toward a more biology-informed exploration of degrader chemical space.

Correspondence: rocio.mercado@chalmers.se Code: https://github.com/yossragh/TACFormer Date: October 30, 2026

## 1 INTRODUCTION

Targeted protein degradation (TPD) has emerged as a promising paradigm in drug discovery, shifting from sustained occupancy of a functional site to the event-driven removal of the target protein itself [1]. Because TPD does not require a druggable active site, it can extend to targets that have long been considered “undruggable” [2, 3]. Proteolysis-targeting chimeras (PROTACs) are among the most developed TPD modalities, with several having advanced into clinical trials [4]. A PROTAC is a heterobifunctional molecule composed of a warhead that binds a protein of interest (POI), an E3 ligand that recruits an E3 ubiquitin ligase, and a linker that tethers the two components [5]. By simultaneously binding the POI and E3 ligase, a PROTAC brings the two proteins into proximity, inducing polyubiquitination of the POI and its subsequent degradation by the proteasome [5].

Despite this promise, PROTAC discovery still relies heavily on iterative synthesis and experimental screening, which must be repeated for each new POI [6]. Deep generative models offer an alternative by learning from known chemistry and proposing molecules beyond existing libraries [7]. Although three-dimensional molecular generation is gaining attention, language-based models remain among the most sample-efficient approaches, and string-based molecular representations continue to be widely used [8–10].

These models typically represent molecules using the simplified molecular-input line-entry system (SMILES). SMILES is non-injective, meaning that the same molecule can be represented by multiple valid strings because atoms can be traversed in different orders [11]. This property offers two advantages for molecular generation. First, randomized SMILES can improve the quality and coverage of generated chemical space [12]. Second, different representations of the same molecule provide a natural form of data augmentation, helping models learn from scarce datasets [8, 13]. This is particularly valuable for PROTAC generation, where only a few thousand experimentally characterized degraders are available.

Existing generative methods for PROTAC design largely focus on linker generation [14], requiring the warhead and E3 ligand to be known beforehand. This confines the search to the linker and restricts de novo design to targets for which suitable binders are already known. As a result, these methods cannot jointly optimize the complete PROTAC, narrowing the explorable chemical space, notably by excluding the discovery of novel warheads.

The few methods that generate complete degraders introduce target information through intermediate proxies, such as target degradation predictors [15], affinity-based fragment selection [16], docking-based rewards [17], or downstream optimization objectives [18], rather than conditioning the molecular generator on the target protein itself. This is an important limitation because the chemical structure of an effective PROTAC fundamentally depends on the biological properties of the POI it is intended to degrade [19]. To our knowledge, no existing method uses a learned representation of the target protein to condition de novo generation of complete PROTAC molecules.

In this work, we aim to address the gaps in existing work by developing a method that (i) generates complete PROTAC molecules rather than restricting generation to the linker; (ii) integrates the biological representation of the POI as a conditioning signal; and (iii) operates effectively in low-data regimes inherent to PROTAC discovery. We summarize our contributions as follows:

• We introduce TACFormer, which is, to our knowledge, the first generative model that directly conditions de novo full-molecule PROTAC generation on a learned representation of the target protein. We use pretrained ESM-2 [20] to encode the POI amino-acid sequence and aggregate the resulting residue embeddings into a fixed-length protein representation. This representation conditions a GPT-style decoder-only transformer that generates PROTAC molecules as SMILES.

• We address the low-data regime through two-stage training. We curate 79,406 PROTAC-like molecules by filtering ZINC20 and ChEMBL 36 for physicochemical and structural properties characteristic of degraders, and use this corpus to pretrain a PROTAC-oriented molecular prior. We then fine-tune the pretrained model on reported PROTACs. We also contribute a normalization pipeline that harmonizes POI and E3-ligase annotations across the three PROTAC source databases.

• We establish that POI identity carries a measurable chemical signature in PROTAC data. Molecules targeting the same POI can span multiple chemotypes, but their local chemical neighborhoods remain strongly enriched for that target. This reveals that POI-associated chemistry is globally heterogeneous yet locally strongly organized.

• We propose an evaluation protocol for target-conditioned molecular generation that combines a likelihood-based measure $( \Delta _ { \mathrm { N L L } } )$ with nearest-neighbor purity in chemical space. We use these complementary metrics to assess both where conditioning succeeds and where it fails, including zero-shot evaluation on POIs unseen during training.

![](images/13cf829fe0e0af36ea2559c4233171e62e10a78aea0f8157870431ae9e23e2f6.jpg)  
Figure 1 Overview of the TACFormer architecture. The protein of interest (POI) amino-acid sequence is encoded using pretrained ESM-2. The resulting residue representations are mean-pooled and projected into a 512-dimensional POI embedding. A GPT-style decoder generates PROTAC SMILES autoregressively. Protein conditioning is introduced in each decoder block by adding the projected POI embedding to the hidden representations after self-attention, scaled by a learnable layer-specific gate. The model generates a complete PROTAC SMILES conditioned on the specified POI.

## 2 RELATED WORK

Transformer-based molecular generation. Transformer architectures are widely used for molecular design with string-based representations [9]. Encoder–decoder models can handle sequence-to-sequence tasks such as reaction prediction [21, 22], retrosynthesis [23–25], and molecular optimization [25, 26]. Encoder-only models learn bidirectional representations of molecular sequences and are mainly used for predictive tasks, including molecular property prediction, activity modeling, and compound–protein interaction prediction [27]. Decoder-only models use causal self-attention to generate molecular sequences autoregressively, making them naturally suited to molecular generation conditioned on physicochemical properties [28, 29], scaffolds [28], or target-related information [30]. Conditioning on protein sequence information is less common: Chen & Bajorath[31] use protein-language-model embeddings to condition a SMILES decoder and generate compounds with desired potency against a target protein, but consider single-ligand rather than bifunctional design. Across these architectures, pretraining on a broad molecular corpus followed by fine-tuning on smaller domain-specific datasets supports transfer learning to data-limited settings [9, 32].

Generative PROTAC linker design. Most generative work on PROTACs formulates the problem as linker design [14]. Early methods apply general-purpose fragment-linking models trained on small-molecule data to PROTAC linker design, without training on actual PROTAC data [33, 34]. Subsequent methods are tailored to PROTAC linker generation through PROTAC-specific training, fine-tuning, or optimization objectives that account for their chemical and structural requirements. These include sequence-based autoregressive generation optimized with reinforcement learning [35], graph-based linker generation [36], and approaches incorporating three-dimensional objectives [37] or diffusion-based modeling [38, 39]. All these methods require the warhead and E3 ligand as input, restricting generation to the linker and leaving the combinatorial space of the whole molecule unexplored.

Full-molecule PROTAC generation and target information. Few methods address full-molecule PROTAC generation. Nori et al.[15] generate PROTAC-like molecular graphs from an empty graph and introduce target specificity through reinforcement learning, using a surrogate model that predicts degradation activity for a given target as the reward. TRL-PROTAC [18] generates warhead and E3 ligands separately, connects them using a linker model, and optimizes the assembled PROTACs using pharmacokinetic and degradation scores. Shao et al.[16] use a protein–compound affinity model to select target-relevant fragments, which guide PROTAC generation under structural and physicochemical constraints. SynPROTAC [17] assembles synthesizable PROTACs from a supplied warhead or E3 ligand using reaction-based molecular assembly and uses docking-guided reinforcement learning to optimize binding-related properties. Although these methods are not target-agnostic, they do not directly condition PROTAC generation on representations of the target protein.

## 3 PRELIMINARIES

## 3.1 AUTOREGRESSIVE FORMULATION

PROTACs are represented as SMILES strings, allowing molecular generation to be formulated as autoregressive language modeling. For a molecule tokenized as

$$
\mathbf { s } = ( s _ { 1 } , \ldots , s _ { T } ) ,\tag{1}
$$

the conditional distribution is factorized as

$$
P _ { \theta } ( \mathbf { s } \mid \mathbf { z } _ { p } ) = \prod _ { t = 1 } ^ { T } P _ { \theta } ( s _ { t } \mid s _ { < t } , \mathbf { z } _ { p } ) .\tag{2}
$$

where θ denotes the model parameters, p the conditioning POI, and $\mathbf { z } _ { p }$ its fixed protein embedding, defined below. The conditional model therefore learns a family of POI-dependent molecular distributions rather than a single unconditional distribution over PROTACs. Tokenization and SMILES randomization are described in Appendix A.1.

## 3.2 PROTEIN REPRESENTATION USING ESM-2

We represent each POI using ESM-2 [20], a pretrained transformer protein language model that learns contextual residue representations from amino-acid sequences. These representations capture sequence patterns related to protein evolution and structure. For each POI, we retrieve its amino-acid sequence from UniProtKB [40] and encode it using the pretrained esm2\_t33\_650M\_UR50D variant (33 layers, 650M parameters, hidden size 1280). For a protein of length L, the resulting residue representations are

$$
\mathbf { H } _ { p } = [ \mathbf { h } _ { 1 } , \dots , \mathbf { h } _ { L } ] ^ { \top } \in \mathbb { R } ^ { L \times { 1 2 8 0 } }\tag{3}
$$

We apply mean pooling over the residue representations to obtain a fixed-length POI embedding,

$$
{ \bf z } _ { p } = L ^ { - 1 } \sum _ { i = 1 } ^ { L } { \bf h } _ { i } \in \mathbb { R } ^ { 1 2 8 0 } .\tag{4}
$$

These sequence-derived embeddings allow molecular generation to be conditioned on protein sequence information rather than discrete target identifiers or static protein descriptors.

Proteins longer than the 1022-residue processing window are encoded in overlapping windows. The residue representations are merged to reconstruct the full-length sequence before mean pooling, ensuring that each residue contributes once to the final embedding (Appendix A.3). ESM-2 remains frozen throughout training.

## 3.3 CONDITIONAL DECODER

The molecular generator is a decoder-only transformer with eight pre-LayerNorm blocks, a hidden dimension of $d _ { \mathrm { m o d e l } } = 5 1 2$ , and eight attention heads (Appendix A.4). The same backbone is used for molecular pretraining, unconditional fine-tuning, and target-conditioned fine-tuning.

The pooled protein representation is projected into the decoder hidden space as

$$
\begin{array} { r } { \mathbf { c } _ { p } = \mathbf { W } _ { c } \amalg _ { p } ( \mathbf { z } _ { p } ) + \mathbf { b } _ { c } , \qquad \mathbf { W } _ { c } \in \mathbb { R } ^ { 5 1 2 \times 1 2 8 0 } . } \end{array}\tag{5}
$$

We introduce protein conditioning through gated residual addition in each transformer block. The projected protein embedding $\mathbf { c } _ { p }$ is broadcast across token positions and added after the causal self-attention residual, scaled by a trainable layer-specific gate $\gamma _ { \ell }$ . For block ℓ, the attention residual is

$$
\mathbf { A } ^ { \left( \ell \right) } = \mathbf { X } ^ { \left( \ell \right) } + \mathrm { M H A } \left( \mathrm { L N } _ { 1 } \left( \mathbf { X } ^ { \left( \ell \right) } \right) \right) ,\tag{6}
$$

and the conditioned block update is

$$
\widetilde { \mathbf { A } } ^ { ( \ell ) } = \mathbf { A } ^ { ( \ell ) } + \operatorname { t a n h } ( \gamma _ { \ell } ) \mathbf { c } _ { p } , \qquad \mathbf { X } ^ { ( \ell + 1 ) } = \widetilde { \mathbf { A } } ^ { ( \ell ) } + \operatorname { M L P } \left( \operatorname { L N } _ { 2 } \left( \widetilde { \mathbf { A } } ^ { ( \ell ) } \right) \right) .\tag{7}
$$

The protein condition $\mathbf { c } _ { p }$ is shared across all transformer blocks, while the layer-specific gates $\gamma _ { \ell }$ control its contribution at each depth. The conditional model is initialized from the pretrained molecular prior and augmented with the protein adapter and one gate per block. Each gate is initialized to $\gamma _ { \ell } = 0 ,$ , so tanh $\mathbf { \partial } ( \gamma _ { \ell } ) \mathbf { c } _ { p } = \mathbf { 0 }$ and the model initially behaves identically to the pretrained molecular decoder. Protein conditioning is then learned during fine-tuning using teacher-forced causal language modeling on molecule– POI pairs.

## 4 EXPERIMENTS

## 4.1 DATA

PROTAC-like pretraining data. To learn a PROTAC-oriented molecular prior, compounds from ZINC20 [41] and ChEMBL 36 [42] are filtered for physicochemical and structural properties characteristic of PROTAC-like molecules. After standardization, filtering, and deduplication, the pretraining corpus contains 79,406 unique molecules. The filtering criteria are detailed in Appendix B.1.

PROTAC data. Two datasets are built by merging PROTAC-DB [43], TPDdb [44], and PROTAC-Pedia [45].

For conditional generation, each observation is a molecule–POI pair. We harmonize target annotations across the three databases by mapping POI names to UniProt accessions and using the corresponding primary gene names as canonical labels. After deduplication, the dataset contains 30,053 molecule–POI pairs across 408 POIs.

We reserve four POIs—WDR5, DHFR, FGFR3, and MAP3K13—for zero-shot evaluation. These targets span different levels of similarity to the training POIs in ESM-2 embedding space. We verify that molecules targeting the zero-shot POIs do not occur under any training POI. Removing their 217 molecule–POI pairs before splitting leaves 29,836 pairs across 404 POIs.

Because the data are unevenly distributed across targets, we select the 20 most represented POIs for target-wise held-out evaluation, hereafter referred to as the seen POIs. These targets account for 21,246 pairs (71.2% of the remaining dataset), providing sufficient reference molecules for separate evaluation of each POI.

For the seen POIs, we split pairs into training (80%), validation (10%), and test (10%) subsets. The remaining 384 POIs contribute 8,590 pairs exclusively to training, exposing the model to a broader range of protein conditions. To prevent molecular leakage, all pairs sharing the same standardized SMILES are assigned to one partition. The final conditional dataset contains 25,584 training, 2,126 validation, and 2,126 test pairs.

For unconditional generation, we discard POI annotations from the conditional partitions and deduplicate standardized SMILES within each partition. This removes repeated molecules with multiple POI labels while preserving their split assignments. The resulting dataset contains 22,880 (84.37%) training, 2,122 (7.82%) validation, and 2,117 (7.81%) test molecules. Both models therefore use matched molecular partitions.

Dataset composition, POI harmonization, zero-shot target selection, and splitting procedures are provided in Appendices B.3, B.2, B.4, and B.5.

## 4.2 EXPERIMENTAL SETUP

Both downstream models are initialized from the same molecular prior. We first pretrain the decoder-only transformer on PROTAC-like molecules, then fine-tune separate unconditional and conditional models on PROTAC data.

The unconditional model is fine-tuned without protein information, learning a distribution $P _ { \phi } ( \mathbf { s } )$ over complete PROTAC molecules. The conditional model is fine-tuned on molecule–POI pairs using the conditioning mechanism described in Section 3, learning $P _ { \theta } ( \mathbf { s } \mid \mathbf { z } _ { p } )$

Both models share the same molecular backbone. For each sampling seed, the unconditional model generates 200,000 molecules. The conditional model generates 10,000 molecules per POI, yielding the same total generation budget.

We evaluate the conditional generation in two settings. In the seen-POI setting, the model generates molecules for the seen POIs and is evaluated against their held-out test data. In the zero-shot setting, the same model generates molecules for WDR5, DHFR, FGFR3, and MAP3K13, which are excluded from training, without additional fine-tuning. Each zero-shot target receives 10,000 generated samples per seed.

Training hyperparameters and sampling settings are provided in Appendices A.8 and B.7, respectively.

## 4.3 EVALUATION

Let G denote the set of generated SMILES, $\nu \subseteq { \mathcal { G } }$ the subset of samples successfully parsed and sanitized by RDKit, and U the set of unique valid molecules obtained by canonicalizing and deduplicating V. For conditional generation, we define the corresponding collections $\mathcal { G } _ { p } , \mathcal { V } _ { p } .$ , and $\mathcal { U } _ { p }$ for each POI p. Metrics are computed per POI and summarized by an unweighted macro average across targets. Implementation details are provided in Appendix B.7.

Evaluation rationale. We first examine whether PROTAC chemistry in the reference data contains a POIspecific signal that can support conditional generation. We compare Morgan–Tanimoto similarity between molecules targeting the same POI and those targeting different POIs. Because many PROTACs share E3- binding ligands, whole-molecule similarity may partly reflect shared E3 identity rather than the POI. We therefore repeat this comparison within CRBN and VHL groups, keeping the E3 ligand class fixed. We also assess whether nearby molecules in chemical space tend to share the same POI label. Observed nearest neighbor purity is compared with a null distribution obtained by shuffling POI labels within each E3 group. This preserves POI frequencies while breaking the relationship between POI identity and molecular structure.

![](images/544a11f753dd765a20a17fd98720c30d5431c73d0e0d35ef2e2d0704f64dbf02.jpg)

![](images/ed497f92478a2e1013da0654b28d87d4708c2a866ad98d788b760255e8620609.jpg)

![](images/7b3a4fe16af9a2a95b3e645346265ad9adea76824c252c98ab15be965d30a595.jpg)

![](images/6807f35928ee181cdf9e274ab2d25a2b7bed6123c951f0d698a4abbe7a2389b6.jpg)  
Figure 2 POI-specific PROTAC chemistry is globally heterogeneous but locally organized. All analyses use 2048-bit Morgan fingerprints with radius 2. (a,b) The same two-dimensional embedding of PROTAC chemical space, colored by (a) E3-ligase identity and (b) POI identity. Both panels contain 3,000 molecules, with 300 sampled from each of the 10 most represented POIs. E3 identity broadly structures chemical space, while individual POIs show local enrichment across distinct regions. (c) Morgan–Tanimoto similarity between same-POI and different-POI molecular pairs across the seen POIs. We sample 10,000 pairs of each type. Same-POI similarity is higher overall (0.385 versus 0.190), within CRBN (0.421 versus 0.223), and within VHL (0.535 versus 0.329), indicating that the POI-specific similarity signal persists after controlling for E3 identity. (d) Local POI purity within CRBN and VHL chemical space. Purity measures the fraction of the k nearest neighbors sharing the query’s POI, with k = 1–30. Solid lines show observed purity, while dashed lines represent the null obtained from 300 within-E3 label permutations. At k = 30, observed purity remains high for CRBN (0.977) and VHL (0.933), well above the shuffled-label null. Strong local enrichment despite moderate global similarity indicates that individual POIs can span multiple locally coherent chemotypes.

The analysis shows that molecules targeting the same POI are more similar on average, even within fixed E3 groups. Their local chemical neighborhoods also show strong POI enrichment. At the same time, individual POIs can span multiple chemotypes rather than forming a single homogeneous cluster. PROTAC chemical space is therefore globally heterogeneous but locally organized by POI identity. This motivates evaluating conditional generation using both likelihood-based sensitivity to the protein condition and local enrichment in held-out target-specific chemical neighborhoods, rather than similarity to a single chemical centroid.

The full analysis is detailed in Appendix B.6, with results shown in Figure 2.

Generation quality. We evaluate generation quality using validity, uniqueness, and novelty, defined as

$$
\mathrm { V } = { \frac { | \mathcal { V } | } { | \mathcal { G } | } } , \qquad \mathrm { U } = { \frac { | \mathcal { U } | } { | \mathcal { V } | } } , \qquad \mathrm { N o v } = { \frac { | \{ g \in \mathcal { U } : g \notin \mathcal { R } _ { \mathrm { t r a i n } } \} | } { | \mathcal { U } | } } .\tag{8}
$$

Here, $\mathcal { R } _ { \mathrm { t r a i n } }$ denotes the set of unique molecules in the training data. Validity measures the fraction of valid generated molecules, uniqueness the fraction of valid molecules that are distinct, and novelty the fraction of unique valid molecules that are not in the training set.

For conditional generation, these metrics are computed for each POI p using $( \mathcal { G } _ { p } , \mathcal { V } _ { p } , \mathcal { U } _ { p } )$ . Novelty is measured against all molecules in the training set, regardless of POI identity. A generated molecule is therefore novel only if it does not appear anywhere in the training data, providing a stricter and more conservative measure of novelty.

We measure structural diversity among unique valid generated molecules using internal diversity defined as

$$
\mathrm { I n t D i v } = 1 - \binom { | \mathcal { U } | } { 2 } ^ { - 1 } \sum _ { i < j } T ( f _ { i } , f _ { j } ) ,\tag{9}
$$

Here, $f _ { i }$ is the Morgan fingerprint of molecule i and $T ( f _ { i } , f _ { j } )$ is the Tanimoto similarity between fingerprints.   
Higher internal diversity indicates greater structural variation among generated molecules.

We further characterize generated molecules using molecular weight, topological polar surface area (TPSA), LogP, heavy-atom count, rotatable-bond count, aromatic and non-aromatic ring counts, and hydrogen-bond acceptor and donor counts.

Fingerprint and molecular-property calculations are detailed in Appendix B.7.

POI-warhead analysis. Whole-molecule diversity does not necessarily reflect diversity at the POI-binding end of a PROTAC. We therefore use PROTAC-Splitter [46] to decompose unique valid generated PROTACs into warhead, linker, and E3 ligand components. We then evaluate the predicted warheads per POI.

The extracted warheads are converted to canonical parent structures. Let $\mathcal { W } _ { p }$ denote the set of successfully processed parent warheads from PROTACs generated for $\mathrm { P O I } \ p ,$ , and $\mathcal { W } _ { p } ^ { \ast }$ be its set of unique canonical warheads. Let ${ \mathcal R } _ { p } ^ { W }$ denote the set of unique canonical parent warheads extracted from training PROTACs targeting the same POI $p .$

Warhead uniqueness and novelty are defined as

$$
\operatorname { W U n i q } ( p ) = \frac { | \mathcal { W } _ { p } ^ { * } | } { | \mathcal { W } _ { p } | } , \qquad \operatorname { W N o v } ( p ) = \frac { | \mathcal { W } _ { p } ^ { * } \setminus \mathcal { R } _ { p } ^ { W } | } { | \mathcal { W } _ { p } ^ { * } | } ,\tag{10}
$$

with internal diversity over unique generated warheads,

$$
\mathrm { W D i v } ( p ) = 1 - \left( \stackrel { | \mathcal { W } _ { p } ^ { * } | } { 2 } \right) ^ { - 1 } \sum _ { i < j } T ( w _ { i } , w _ { j } ) .\tag{11}
$$

where $T ( w _ { i } , w _ { j } )$ denotes the Tanimoto similarity between warhead fingerprints.

PROTAC-Splitter processing, parent-warhead construction, and quality control are detailed in Appendix B.10.

Conditional fidelity. Generation quality does not establish whether the decoder responds to the protein condition. We therefore assess conditional fidelity using two complementary measures.

First, we examine whether held-out molecules receive higher likelihood under their correct POI condition than under competing conditions. For a held-out molecule $x$ and POI $p ,$ the length-normalized negative log-likelihood is defined as

$$
\operatorname { N L L } ( x \mid p ) = - { \frac { 1 } { T _ { x } } } \sum _ { t = 1 } ^ { T _ { x } } \log P _ { \theta } \left( s _ { t } \mid s _ { < t } , \mathbf { z } _ { p } \right) .\tag{12}
$$

Let $\tau$ denote the set of POIs in the current evaluation. For a held-out molecule $x ,$ let ${ \mathcal { P } } ( x )$ denote the subset of POIs that are known targets of x. Conditional preference is measured by

$$
\Delta _ { \mathrm { N L L } } ( x , p ) = \frac { 1 } { | \mathcal { T } \setminus \mathcal { P } ( x ) | } \sum _ { q \in \mathcal { T } \setminus \mathcal { P } ( x ) } \mathrm { N L L } ( x \mid q ) - \mathrm { N L L } ( x \mid p ) .\tag{13}
$$

A positive value indicates higher likelihood under the correct POI condition than under competing conditions, on average. If a molecule has multiple valid POIs, all of them are excluded from the competing set to avoid treating correct conditions as incorrect.

Second, we measure POI enrichment among the nearest held-out reference neighbors of generated molecules. Queries are restricted to valid, unique, training-novel molecules. For molecule $g$ generated under $\mathrm { P O I } p ,$ let $\textstyle { \mathcal { N } } _ { k } ( g )$ denote its k nearest molecules in the held-out test reference pool by Morgan–Tanimoto similarity. POI purity is

$$
\rho _ { k } ( g , p ) = \frac { 1 } { k } \sum _ { r \in \mathcal { N } _ { k } ( g ) } \mathbf { 1 } \left[ p \in \mathcal { P } ( r ) \right] ,\tag{14}
$$

We average this score over queries generated for POI $p$ and report $k \in \{ 5 , 1 0 , 2 0 \}$

To provide context for generated purity, we also measure the local POI organization of the held-out reference molecules using leave-one-out nearest-neighbor analysis. Let $\mathcal { R } _ { p }$ denote the unique held-out reference molecules targeting $\mathrm { P O I } p ,$ , and $\mathcal { N } _ { k } ^ { - x } ( x )$ denote the k nearest reference molecules to $x ,$ excluding x itself. Reference purity is defined as

$$
\mathrm { R e f P u r i t y } _ { k } ( p ) = \frac { 1 } { | \mathcal { R } _ { p } | } \sum _ { x \in \mathcal { R } _ { p } } \frac { 1 } { k } \sum _ { r \in \mathcal { N } _ { k } ^ { - x } ( x ) } \mathbf { 1 } \left[ p \in \mathcal { P } ( r ) \right] .\tag{15}
$$

This provides a reference point for interpreting generated neighborhood purity.

Reference-pool construction and multi-label handling are described in Appendix B.9.

Baseline. We use the unconditional model as a target-agnostic baseline to quantify the contribution of protein conditioning to POI-associated chemical enrichment. For each POI $p ,$ we compare the expected k-NN purity under conditional and unconditional generation as

$$
\begin{array} { r } { \Delta _ { k } ( p ) = \underbrace { \mathbb { E } _ { g \sim P _ { \theta } ( \cdot | \mathbf { z } _ { p } ) } \big [ \rho _ { k } ( g , p ) \big ] } _ { \mathrm { C o n d i t i o n a l } } - \underbrace { \mathbb { E } _ { g \sim P _ { \phi } } \big [ \rho _ { k } ( g , p ) \big ] } _ { \mathrm { U n c o n d i t i o n a l } } . } \end{array}\tag{16}
$$

Both expectations are estimated using valid, unique, training-novel generated molecules against the same held-out reference. A positive $\Delta _ { k } ( p )$ indicates greater local enrichment for POI $p$ under conditional generation. We report $\Delta _ { k } ( p )$ for $k \in \{ 5 , 1 0 , 2 0 \}$

## 5 RESULTS

## 5.1 MOLECULAR PRETRAINING AND UNCONDITIONAL GENERATION

Pretraining on PROTAC-like molecules produces a molecular prior capable of generating valid, novel, and diverse structures. From 10,000 generated samples, the pretrained model achieves 98.18% validity, 91.99% uniqueness, 66.67% novelty, and an internal diversity of 0.836.

Fine-tuning this molecular prior on PROTAC data without protein conditioning produces the unconditional model. Across three seeds, with 200,000 generated molecules per seed, the unconditional model achieves $( 9 9 . 0 5 \pm 0 . 0 1 ) $ % validity, $( 4 7 . 1 3 \pm 0 . 1 4 ) \%$ uniqueness, $( 8 9 . 5 4 \pm 0 . 0 8 ) \%$ novelty, and an internal diversity of $0 . 7 9 2 \pm 0 . 0 0 1$

## 5.2 TARGET-CONDITIONED GENERATION FOR SEEN POIS

Table 1 summarizes generation quality and conditional fidelity for the seen POIs. We generate 10,000 molecules per target and report metrics averaged across three sampling seeds.

Table 1 Target-wise generation quality and conditional fidelity across the seen POIs. Generation metrics and nearest-neighbor purity are reported as mean (standard deviation) across three generation seeds. Best per-target values in each column are shown in bold
<table><tr><td>POI</td><td>Validity (%)</td><td>Uniqueness (%)</td><td>Novelty (%)</td><td>Diversity</td><td> $\Delta _ { \mathrm { N L L } }$ </td><td>5-NN (%)</td><td>10-NN (%)</td><td>20-NN (%)</td></tr><tr><td>AR</td><td>99.65 (0.07)</td><td>48.64 (0.53)</td><td>79.38 (0.09)</td><td>0.689 (0.004)</td><td>0.278</td><td>97.89 (0.15)</td><td>97.32 (0.16)</td><td>96.33 (0.15)</td></tr><tr><td>ESR1</td><td>99.53 (0.11)</td><td>41.12 (0.38)</td><td>80.54 (0.13)</td><td>0.656 (0.001)</td><td>0.366</td><td>97.94 (0.20)</td><td>97.12 (0.25)</td><td>94.78 (0.23)</td></tr><tr><td>CDK4</td><td>99.58 (0.05)</td><td>30.60 (0.14)</td><td>81.86 (0.20)</td><td>0.566 (0.006)</td><td>0.618</td><td>95.17 (0.10)</td><td>93.62 (0.23)</td><td>92.66 (0.36)</td></tr><tr><td>KRAS</td><td>99.46 (0.06)</td><td>49.80 (0.26)</td><td>85.59 (0.18)</td><td>0.641 (0.004)</td><td>0.498</td><td>97.91 (0.08)</td><td>97.18 (0.06)</td><td>95.56 (0.09)</td></tr><tr><td>EZH2</td><td>98.40 (0.01)</td><td>31.50 (0.55)</td><td>94.57 (0.14)</td><td>0.629 (0.003)</td><td>0.358</td><td>82.38 (0.25)</td><td>74.63 (0.13)</td><td>56.15 (0.30)</td></tr><tr><td>BTK</td><td>97.85 (0.15)</td><td>43.06 (0.19)</td><td>87.63 (0.31)</td><td>0.716 (0.003)</td><td>0.428</td><td>88.58 (0.21)</td><td>85.50 (0.20)</td><td>81.07 (0.21)</td></tr><tr><td>BRD4</td><td>97.16 (0.07)</td><td>41.63 (0.30)</td><td>91.02 (0.26)</td><td>0.720 (0.006)</td><td>0.287</td><td>66.19 (0.77)</td><td>59.97 (0.82)</td><td>50.83 (0.73)</td></tr><tr><td>BRAF</td><td>99.70 (0.01)</td><td>28.77 (0.24)</td><td>88.36 (0.14)</td><td>0.612 (0.001)</td><td>0.355</td><td>92.59 (0.27)</td><td>90.22 (0.33)</td><td>80.50 (0.32)</td></tr><tr><td>SMARCA4</td><td>98.62 (0.02)</td><td>41.18 (0.38)</td><td>90.29 (0.27)</td><td>0.671 (0.001)</td><td>0.347</td><td>27.37 (1.20)</td><td>26.29 (1.17)</td><td>23.59 (0.98)</td></tr><tr><td>MAP4K1</td><td>98.29 (0.11)</td><td>33.90 (0.19)</td><td>94.63 (0.11)</td><td>0.726 (0.003)</td><td>0.418</td><td>61.59 (0.74)</td><td>40.43 (0.49)</td><td>22.65 (0.15)</td></tr><tr><td>IRAK4</td><td>97.68 (0.11)</td><td>57.13 (0.68)</td><td>94.56 (0.07)</td><td>0.701 (0.005)</td><td>0.507</td><td>93.97 (0.11)</td><td>92.49 (0.18)</td><td>85.74 (0.09)</td></tr><tr><td>EGFR</td><td>97.91 (0.10)</td><td>52.05 (0.44)</td><td>89.51 (0.30)</td><td>0.723 (0.005)</td><td>0.381</td><td>90.86 (0.16)</td><td>85.26 (0.23)</td><td>76.40 (0.34)</td></tr><tr><td>BCL2L1</td><td>98.94 (0.05)</td><td>37.25 (0.26)</td><td>88.20 (0.07)</td><td>0.682 (0.003)</td><td>0.422</td><td>91.71 (0.38)</td><td>88.58 (0.54)</td><td>82.54 (0.67)</td></tr><tr><td>SMARCA2</td><td>98.58 (0.06)</td><td>42.12 (0.45)</td><td>89.86 (0.36)</td><td>0.672 (0.003)</td><td>0.327</td><td>79.30 (0.32)</td><td>73.34 (0.22)</td><td>67.61 (0.11)</td></tr><tr><td>METTL14</td><td>99.84 (0.02)</td><td>13.74 (0.29)</td><td>81.58 (0.20)</td><td>0.386 (0.001)</td><td>0.564</td><td>99.57 (0.14)</td><td>99.55 (0.15)</td><td>99.41 (0.13)</td></tr><tr><td>PTPN1</td><td>97.55 (0.09)</td><td>24.61 (0.24)</td><td>92.79 (0.13)</td><td>0.589 (0.007)</td><td>0.538</td><td>81.92 (0.63)</td><td>78.96 (0.65)</td><td>77.22 (0.66)</td></tr><tr><td>BCL2</td><td>97.29 (0.16)</td><td>28.87 (0.41)</td><td>93.85 (0.17)</td><td>0.599 (0.010)</td><td>0.394</td><td>75.60 (0.17)</td><td>71.92 (0.25)</td><td>64.41 (0.48)</td></tr><tr><td>SOS1</td><td>99.25 (0.11)</td><td>33.99 (0.64)</td><td>95.34 (0.07)</td><td>0.645 (0.001)</td><td>0.356</td><td>86.51 (0.49)</td><td>74.80 (0.27)</td><td>51.12 (0.14)</td></tr><tr><td>BCL6</td><td>99.30 (0.13)</td><td>28.50 (0.64)</td><td>91.04 (0.14)</td><td>0.467 (0.007)</td><td>0.516</td><td>98.60 (0.09)</td><td>98.52 (0.13)</td><td>98.04 (0.14)</td></tr><tr><td>MAPT</td><td>99.61 (0.03)</td><td>30.16 (0.16)</td><td>93.13 (0.03)</td><td>0.562 (0.003)</td><td>0.494</td><td>89.00 (0.54)</td><td>86.40 (0.54)</td><td>83.60 (0.51)</td></tr><tr><td>Macro</td><td>98.71 (0.02)</td><td>36.93 (0.04)</td><td>89.19 (0.06)</td><td>0.633 (0.001)</td><td>0.423</td><td>84.73 (0.11)</td><td>80.60 (0.11)</td><td>74.01 (0.14)</td></tr></table>

Validity remains high across all POIs, with a macro average of 98.71% and a minimum of 97.16%. Mean novelty reaches 89.19%. Uniqueness and internal diversity vary more substantially, with macro averages of 36.93% and 0.633, respectively.

Both conditional-fidelity measures show sensitivity to POI identity. $\Delta _ { \mathrm { N L L } }$ is positive for all seen POIs, with a macro average of +0.423. Held-out molecules therefore receive lower negative log-likelihood, on average, under their correct POI condition than under competing conditions.

The generated molecules also show strong POI-specific neighborhood enrichment. Macro purity reaches 84.73% at k = 5, 80.60% at k = 10, and 74.01% at k = 20. For comparison, the corresponding leaveone-out reference purity values are 92.27%, 86.72%, and 78.33% (Appendix Table 8). Thus, generated molecules reproduce much of the local POI enrichment observed in the held-out reference. The strength of this enrichment nevertheless varies across targets. At k = 5, generated purity ranges from 27.37% for SMARCA4 to 99.57% for METTL14.

Warhead analysis further shows that the model generates novel warheads, with a macro average of 86.62% (Appendix Table 6). PROTAC-Splitter yields usable three-component decompositions for 99.99% of unique valid generated PROTACs, averaged across POIs. Of these decompositions, 99.97% yield valid parent warheads (Appendix Table 7).

Figure 3 illustrates generated PROTACs for BRD4, MAP4K1, and BTK, which have the highest warhead internal diversity. The examples show distinct warhead scaffolds generated under the same protein condition.

![](images/4805ce0b19846c20e43fdefab3daab821b9a25429dbf63c2926ac0ccf4d417dd.jpg)

Figure 3 Generated PROTACs span distinct warhead scaffolds under the same POI condition. Rows show BRD4, MAP4K1, and BTK, each with one reference PROTAC and three generated examples. These targets have the highest warhead internal diversity (0.869, 0.835, and 0.817, respectively). Gold highlighting identifies the warhead predicted by PROTAC-Splitter in each generated molecule. Bemis–Murcko scaffolds are extracted from the parent warheads to group molecules by their core structures. For each POI, the generated examples represent three distinct warhead-scaffold groups.

## 5.3 COMPARISON WITH THE UNCONDITIONAL BASELINE

To quantify the contribution of protein conditioning, we compare TACFormer with the matched unconditional baseline. Both models are evaluated against the same held-out reference pool, and $\Delta _ { k } ( p )$ measures their difference in k-NN purity for POI p (Section 4.3).

Table 2 shows positive improvements for every seen POI at k = 5, k = 10, and k = 20. The macro improvements are 79.71, 75.58, and 68.99 percentage points, respectively.

At k = 5, the largest improvements occur for METTL14 and BCL6, reaching 97.05 and 96.85 percentage points, respectively. SMARCA4 shows the smallest improvement at 26.45 percentage points. These positive differences demonstrate that protein conditioning increases enrichment for the requested POI in held-out chemical neighborhoods compared with unconditional generation.

The target-wise reference, conditional, and unconditional purity values are reported in Appendix Table 8.

Table 2 Improvement in POI-neighborhood purity over unconditional generation. $\Delta _ { k } ( p )$ measures the difference between conditional and unconditional k-NN purity for each POI. Values are reported in percentage points as mean (standard deviation) across three generation seeds.
<table><tr><td>POI</td><td>∆5 (pp)</td><td>∆10 (pp)</td><td>∆20 (pp)</td></tr><tr><td>AR</td><td>75.14 (0.02)</td><td>74.14 (0.10)</td><td>72.32 (0.10)</td></tr><tr><td>ESR1</td><td>88.03 (0.18)</td><td>86.86 (0.24)</td><td>84.10 (0.21)</td></tr><tr><td>CDK4</td><td>89.38 (0.06)</td><td>87.94 (0.15)</td><td>87.01 (0.27)</td></tr><tr><td>KRAS</td><td>87.95 (0.06)</td><td>87.13 (0.05)</td><td>85.41 (0.10)</td></tr><tr><td>EZH2</td><td>80.82 (0.27)</td><td>73.01 (0.12)</td><td>54.64 (0.28)</td></tr><tr><td>BTK</td><td>82.32 (0.23)</td><td>79.23 (0.20)</td><td>74.75 (0.20)</td></tr><tr><td>BRD4</td><td>60.87 (0.74)</td><td>54.94 (0.80)</td><td>46.24 (0.73)</td></tr><tr><td>BRAF</td><td>89.65 (0.33)</td><td>87.16 (0.39)</td><td>77.41 (0.34)</td></tr><tr><td>SMARCA4</td><td>26.45 (1.19)</td><td>25.38 (1.16)</td><td>22.70 (0.97)</td></tr><tr><td>MAP4K1</td><td>60.05 (0.74)</td><td>39.16 (0.50)</td><td>21.58 (0.15)</td></tr><tr><td>IRAK4</td><td>90.26 (0.11)</td><td>88.69 (0.17)</td><td>82.00 (0.05)</td></tr><tr><td>EGFR</td><td>76.64 (0.22)</td><td>71.84 (0.27)</td><td>63.90 (0.37)</td></tr><tr><td>BCL2L1</td><td>89.20 (0.38)</td><td>86.02 (0.56)</td><td>79.96 (0.67)</td></tr><tr><td>SMARCA2</td><td>76.44 (0.31)</td><td>70.59 (0.21)</td><td>64.92 (0.06)</td></tr><tr><td>METTL14</td><td>97.05 (0.12)</td><td>96.66 (0.14)</td><td>96.13 (0.12)</td></tr><tr><td>PTPN1</td><td>80.92 (0.61)</td><td>77.96 (0.63)</td><td>76.20 (0.64)</td></tr><tr><td>BCL2</td><td>74.18 (0.19)</td><td>70.46 (0.26)</td><td>63.04 (0.48)</td></tr><tr><td>SOS1</td><td>85.23 (0.48)</td><td>73.44 (0.26)</td><td>49.83 (0.13)</td></tr><tr><td>BCL6</td><td>96.85 (0.10)</td><td>96.66 (0.13)</td><td>96.01 (0.15)</td></tr><tr><td>MAPT</td><td>86.76 (0.56)</td><td>84.34 (0.55)</td><td>81.62 (0.51)</td></tr><tr><td>Macro</td><td>79.71 (0.11)</td><td>75.58 (0.11)</td><td>68.99 (0.14)</td></tr></table>

## 5.4 ZERO-SHOT GENERATION

Table 3 reports generation quality and conditional fidelity for the four zero-shot POIs. Generation quality remains high overall, with macro validity of 91.76%, novelty of 98.19%, and internal diversity of 0.712. Validity is notably lower for WDR5 (75.44%) than for the other targets, which reach at least 95.33%.

Likelihood-based conditional fidelity differs across the four proteins. WDR5 and DHFR both show negative $\Delta _ { \mathrm { N L I } }$ values of −0.089, indicating that their correct POI conditions are not preferred over competing conditions on average. In contrast, FGFR3 and MAP3K13 show positive values of +0.080 and +0.096, respectively. These results suggest a relationship between protein-representation similarity and likelihood preference among the selected zero-shot targets.

Nearest-neighbor fidelity reveals a different pattern.Macro purity reaches 30.52% at k = 5, 31.55% at k = 10, and 32.29% at k = 20. At k = 5, WDR5 achieves the highest purity (56.23%), followed by DHFR (31.05%), FGFR3 (20.26%), and MAP3K13 (14.57%). By comparison, the held-out reference molecules exhibit strong local POI enrichment, with a macro 5-NN purity of 99.25% (Appendix Table 9).

Table 3 Generation quality and conditional fidelity for the four zero-shot POIs. WDR5, DHFR, FGFR3, and MAP3K13 are excluded entirely from conditional training and provided to the model only at inference. Generation metrics and nearest-neighbor purity are reported as mean (standard deviation) across three generation seeds.
<table><tr><td>POI</td><td>Validity (%)</td><td>Uniq. (%)</td><td>Novelty (%)</td><td>Diversity</td><td> $\overline { { \Delta _ { \mathrm { N L L } } } }$ </td><td>5-NN (%)</td><td>10-NN (%)</td><td>20-NN (%)</td></tr><tr><td>WDR5</td><td>75.44 (0.53)</td><td>99.21 (0.08)</td><td>99.96 (0.01)</td><td>0.737 (0.002)</td><td>-0.089</td><td>56.23 (0.10)</td><td>55.35 (0.35)</td><td>50.58 (0.44)</td></tr><tr><td>DHFR</td><td>95.33 (0.03)</td><td>88.59 (0.20)</td><td>99.57 (0.01)</td><td>0.669 (0.003)</td><td>-0.089</td><td>31.05 (0.32)</td><td>37.06 (0.28)</td><td>45.18 (0.14)</td></tr><tr><td>FGFR3</td><td>96.96 (0.04)</td><td>45.95 (0.57)</td><td>97.31 (0.08)</td><td>0.716 (0.003)</td><td>0.080</td><td>20.26 (0.15)</td><td>20.51 (0.07)</td><td>20.70 (0.05)</td></tr><tr><td>MAP3K13</td><td>99.29 (0.12)</td><td>36.23 (0.22)</td><td>95.90 (0.25)</td><td>0.727 (0.004)</td><td>0.096</td><td>14.57 (0.46)</td><td>13.27 (0.31)</td><td>12.71 (0.12)</td></tr><tr><td>Macro</td><td>91.76 (0.11)</td><td>67.50 (0.15)</td><td>98.19 (0.04)</td><td>0.712 (0.001)</td><td>-0.0005</td><td>30.52 (0.06)</td><td>31.55 (0.11)</td><td>32.29 (0.14)</td></tr></table>

Thus, the positive likelihood preference observed for FGFR3 and MAP3K13 does not translate into strong enrichment in their held-out chemical neighborhoods. These results suggest that similarity in ESM-2

embedding space may support likelihood-based generalization without ensuring comparable improvement in generated chemical distributions. Zero-shot generalization remains limited relative to the seen-POI results.

## 5.5 INVALID SMILES FAILURE MODES

The model generates 2,582 invalid SMILES across the seen POI conditions. The most frequent recorded failures are kekulization errors (43.45%), unclosed ring labels (42.41%), and incorrect parentheses (10.34%). Valence or atomic-property violations account for 2.13%, and duplicate ring-closure bonds account for 1.16%.

Failure patterns also differ across POIs. Some conditions produce more aromaticity-related errors, while others show more ring-closure or branching-syntax failures. These findings indicate that most invalid SMILES arise from difficulties in maintaining valid ring connectivity and aromatic structures during autoregressive generation.

## 6 DISCUSSION

The results demonstrate that protein conditioning enables TACFormer to capture POI-specific patterns in PROTAC chemical space beyond those learned without target information. For the seen POIs, likelihood preference and generated neighborhood purity provide complementary evidence that the protein condition influences both how the model assigns likelihood to molecules and the chemical distributions it generates.

The unconditional baseline provides an important comparison. It generates highly valid, novel, and diverse PROTACs but achieves lower neighborhood purity. This shows that learning the overall PROTAC distribution is not sufficient to reproduce the chemical patterns that distinguish individual targets. Because both models are fine-tuned on matched molecular data, the improvement achieved by TACFormer supports the contribution of protein conditioning rather than molecular fine-tuning alone.

Conditional fidelity depends on the structure of the reference chemical space. Generated k-NN purity varies considerably across POIs, but its value must be interpreted in the context of the corresponding reference chemistry. Held-out reference molecules also differ in neighborhood purity. For some targets, local neighborhoods remain strongly enriched for the same POI, whereas for others, they become increasingly mixed as k increases. Consequently, generated purity alone does not fully characterize how well the model captures the chemical organization of the reference data.

MAP4K1 illustrates this distinction. Its generated 5-NN purity reaches 61.59%, below targets such as AR or METTL14, but its held-out reference purity is only 70.00%. At k = 20, generated and reference purity are nearly identical (22.65% and 23.33%, respectively). The lower purity therefore largely reflects the organization of the available MAP4K1 reference chemistry rather than a failure to reproduce it. SMARCA4 shows a different pattern. Its reference 5-NN purity reaches 77.39%, whereas generated purity is only 27.37%. Here, the large reference–generation gap indicates that TACFormer reproduces the local POI enrichment of the reference molecules less consistently. Nevertheless, its generated purity still exceeds the unconditional baseline by 26.45 percentage points, showing that protein conditioning still improves enrichment relative to target-agnostic generation.

Comparing generated and reference purity therefore provides important context for evaluating conditional fidelity. For several targets, including AR, ESR1, EGFR, BCL2L1, METTL14, BCL6, and MAPT, generated purity closely approaches the corresponding reference values. For KRAS and SOS1, it even exceeds reference purity at larger neighborhood sizes. However, this does not imply greater biological target specificity or more complete coverage of relevant chemistry. Instead, the model may preferentially generate molecules from highly enriched regions while sampling other chemotypes less frequently. This would produce high local purity without necessarily covering the broader reference chemical space. Neighborhood purity therefore measures local POI enrichment, not the completeness of chemotype coverage.

This distinction is important when interpreting the diversity of generated molecules. High neighborhood purity can coexist with limited chemical diversity if generation concentrates within a few POI-enriched regions. METTL14 illustrates this possibility. It achieves 99.57% generated 5-NN purity but only 13.74% uniqueness and 0.386 internal diversity. BCL6 similarly combines high neighborhood purity with low diversity. These findings are consistent with generation concentrated within a narrower range of chemical structures, although they do not establish which reference chemotypes are covered. Conversely, a more diverse generated distribution may span several chemical regions with different levels of local POI enrichment.

The relationship between internal diversity and conditional fidelity also depends on the evaluation measure. Across the seen POIs, internal diversity is negatively correlated with $\Delta _ { \mathrm { N L L } } ~ ( R ^ { 2 } = 0 . 3 7 5 , p = 0 . 0 0 4 1 )$ whereas its relationship with 5-NN purity is weaker and not statistically significant $( R ^ { 2 } = 0 . 1 1 8 , p = 0 . 1 3 9 )$ Importantly, $\Delta _ { \mathrm { N L I } }$ measures likelihood preference for held-out molecules, while internal diversity measures variation among generated molecules. Their correlation therefore does not establish that reducing molecular diversity improves conditional fidelity. Instead, these findings show that likelihood preference, local POI enrichment, and structural diversity capture different properties of generation and should be interpreted together.

Target-wise variation in conditional fidelity may also reflect differences in the amount and composition of training data. The number of training examples varies across the seen POIs (Appendix B.5), and individual targets can span multiple chemotypes. TACFormer must therefore learn from unequal amounts of data and chemical distributions with different levels of heterogeneity. Training-set size alone does not appear to explain the observed variation, as some targets with few examples achieve strong conditional fidelity while others while others with more examples remain difficult. These observations suggest that both data availability and chemical heterogeneity may jointly contribute to target-wise performance.

Protein conditioning supports diverse warhead generation. The warhead analysis shows that structural novelty extends to the warhead component. Macro POI-warhead novelty reaches 86.62%, and several targets show substantial warhead diversity. This indicates that TACFormer can also explore new structures at the POI-binding end of the molecule rather than relying exclusively on warheads observed for the same target during training.

However, warhead novelty alone does not establish whether generated PROTACs reflect the chemistry of the intended target. SMARCA4, for example, achieves 95.87% warhead novelty but only 27.37% generated 5-NN purity. This contrast shows that generating novel warheads does not necessarily produce molecules enriched in the chemical neighborhoods of the requested target. Warhead novelty should therefore be evaluated alongside conditional fidelity to assess both structural exploration and how effectively protein conditioning guides molecular generation.

Protein conditioning generalizes partially to unseen POIs. The zero-shot results reveal a distinction between likelihood-based preference and the ability to generate chemistry characteristic of an unseen target. FGFR3 and MAP3K13, whose ESM-2 representations are similar to those of training proteins, show positive likelihood preference. However, this preference does not translate into strong enrichment in their held-out chemical neighborhoods. Conversely, WDR5 shows negative likelihood preference but achieves the highest generated neighborhood purity among the four zero-shot POIs. These findings indicate that likelihood-based transfer and generated neighborhood enrichment do not necessarily follow the same pattern.

One possible explanation is that similarity in ESM-2 embedding space supports partial transfer of learned chemical preferences but does not sufficiently constrain generation toward the chemotypes of a new POI. This suggests that likelihood-based transfer may be easier to achieve than reproducing the corresponding chemical distributions. The high reference neighborhood purity of the zero-shot POIs indicates strong local chemical organization in the available held-out data, but TACFormer captures this structure only partially.

## 6.1 LIMITATIONS AND FUTURE DIRECTIONS.

The current evaluation demonstrates that protein conditioning guides generation toward chemistry characteristic of the requested POI, but it does not establish biological activity. Neither $\Delta _ { \mathrm { N L I } }$ nor neighborhood purity provides evidence of target binding, degradation potency, or selectivity. A generated molecule outside the observed reference neighborhoods may still be active, and similarity to known PROTACs does not guarantee degradation activity. Future work could explore degradation-activity prediction using PROTACs with measured outcomes, such as $\mathrm { D C } _ { 5 0 }$ and $D _ { \mathrm { m a x } }$ . Such a predictor could estimate degradation potency and efficacy for generated molecules, complementing the current work by allowing generated PROTACs to be ranked according to their predicted degradation potency and efficacy.

Another limitation concerns how TACFormer represents the protein condition. Each POI is encoded using a single mean-pooled ESM-2 embedding. Although this provides a compact representation of the protein sequence, pooling compresses residue-level information into one vector and may obscure sequence regions important for ligand recognition. This may be particularly limiting to zero-shot generation, where no molecular examples for the unseen targets are available during training to compensate for information lost during pooling.

Future work could investigate residue-level ESM-2 conditioning using attention mechanisms that allow the molecular decoder to access information from individual protein positions during SMILES generation. This would require modifications to the current architecture and evaluation of its effect on conditional generation. Protein structural information could also be explored as a complementary approach for evaluating and prioritizing generated PROTACs, where suitable data are available. These directions could help assess whether more detailed protein information improves generation and generalization to unseen POIs.

## 7 CONCLUSION

TACFormer demonstrates that protein sequence conditioning can steer PROTAC generation toward chemical patterns characteristic of the requested target. Across the seen POIs, positive likelihood preference, strong enrichment in held-out chemical neighborhoods, and consistent improvements over the matched unconditional baseline provide complementary evidence that protein information guides molecular generation beyond the general PROTAC prior. The results also highlight that conditional fidelity, chemical diversity, chemotype coverage, and warhead novelty represent distinct aspects of generation that should be evaluated together rather than in isolation.

The zero-shot experiments reveal an important limitation. Positive likelihood preference for some unseen POIs does not consistently translate into strong enrichment in their reference chemical neighborhoods. Similarity between protein representations may support likelihood-based transfer but does not guarantee that generated molecules reproduce the chemistry observed for unseen targets.

TACFormer provides a foundation for sequence-conditioned PROTAC generation, while improving zero-shot generalization and integrating degradation-activity prediction remain important directions for future research.

## AUTHOR CONTRIBUTIONS

YG: Conceptualization, data curation, methodology, software, analysis, investigation, validation, visualization, writing – original draft, and writing – review & editing. RM: Conceptualization, methodology, supervision, funding acquisition, resources, writing – original draft, and writing – review & editing.

## ACKNOWLEDGMENTS

YG and RM acknowledge funding from the Wallenberg AI, Autonomous Systems and Software Program (WASP), supported by the Knut and Alice Wallenberg Foundation. Computational resources were provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), funded by the Swedish Research Council.

## SOFTWARE AND DATA AVAILABILITY

Code is available at https://github.com/yossragh/TACFormer.

## REFERENCES

[1] Miklós Békés, David R Langley, and Craig M Crews. PROTAC targeted protein degraders: the past is prologue. Nature Reviews Drug Discovery, 21(3):181–200, 2022.

[2] Andrew L Hopkins and Colin R Groom. The druggable genome. Nature Reviews Drug Discovery, 1(9):727–730, 2002.

[3] John H Bushweller. Targeting transcription factors in cancer—from undruggable to reality. Nature Reviews Cancer, 19(11):611–624, 2019.

[4] Matthew N O’Brien Laramy, Suman Luthra, Matthew F Brown, and Derek W Bartlett. Delivering on the promise of protein degraders. Nature Reviews Drug Discovery, 22(5):410–427, 2023.

[5] Lin Zhao, Jia Zhao, Kunhong Zhong, Aiping Tong, and Da Jia. Targeted protein degradation: mechanisms, strategies and application. Signal Transduction and Targeted Therapy, 7(1):113, 2022.

[6] Andy Pike, Esther CY Lee, Iacovos N Michaelides, Markus Schade, Ankit Sharma, James S Scott, and Abhishek Srivastava. Lessons learned in linking PROTACs from discovery to the clinic. Nature Reviews Chemistry, 10(2): 117–132, 2026.

[7] Chao Pang, Jianbo Qiao, Xiangxiang Zeng, Quan Zou, and Leyi Wei. Deep generative models in de novo drug molecule generation. Journal ofChemical Information and Modeling, 64(7):2174–2194, 2024.

[8] Michael A Skinnider, R Greg Stacey, David S Wishart, and Leonard J Foster. Chemical language models enable navigation in sparsely populated chemical space. Nature Machine Intelligence, 3(9):759–770, 2021.

[9] Francesca Grisoni. Chemical language models for de novo drug design: Challenges and opportunities. Current Opinion in Structural Biology, 79:102527, 2023.

[10] Jeff Guo, Junwu Chen, Anthony Gx-Chen, and Philippe Schwaller. Sample-efficient generative molecular design using memory manipulation. Nature Machine Intelligence, 8(3):449–460, 2026.

[11] David Weininger, Arthur Weininger, and Joseph L Weininger. SMILES. 2. algorithm for generation of unique SMILES notation. Journal ofChemical Information and Computer Sciences, 29(2):97–101, 1989.

[12] Josep Arús-Pous, Simon Viet Johansson, Oleksii Prykhodko, Esben Jannik Bjerrum, Christian Tyrchan, Jean-Louis Reymond, Hongming Chen, and Ola Engkvist. Randomized SMILES strings improve the quality of molecular generative models. Journal ofCheminformatics, 11(1):71, 2019.

[13] Michael Moret, Lukas Friedrich, Francesca Grisoni, Daniel Merk, and Gisbert Schneider. Generative molecular design in low data regimes. Nature Machine Intelligence, 2(3):171–180, 2020.

[14] Yossra Gharbi and Rocío Mercado. A comprehensive review of emerging approaches in machine learning for de novo PROTAC design. Digital Discovery, 3(11):2158–2176, 2024.

[15] Divya Nori, Connor W. Coley, and Rocío Mercado. De Novo PROTAC design using graph-based deep generative models. arXiv preprint arXiv:2211.02660, 2022.

[16] Jinsong Shao, Qineng Gong, Zeyu Yin, Yu Chen, Yajie Hao, Lei Zhang, Linlin Jiang, Min Yao, Jinlong Li, Fubo Wang, and Li Wang. Language model driven: A PROTAC generation pipeline with dual constraints of structure and property. arXiv preprint arXiv:2412.09661, 2024.

[17] Mingyuan Xu, Chaoming Huang, Li Pang, Qirui Deng, Hao Zhang, Anjie Qiao, Zhiwen Luo, Zhen Wang, Chang-Yu Hsieh, Zhang Zhang, Tie-Gen Chen, Hongming Chen, and Jinping Lei. Design of synthesizable PROTACs through synthesis constrained generative model and reinforcement learning. JACS Au, 6(7):4239–4252, 2026. doi: 10.1021/jacsau.6c00732.

[18] Yuhao Dai and Fei Zhu. TRL-PROTAC: A pre-trained generator of PROTACs targeting specific proteins optimized by reinforcement learning. Computer Science and Information Systems, 21(4):1293–1320, 2024. doi: 10.2298/ CSIS240327039D.

[19] Jonathan M Tsai, Radosław P Nowak, Benjamin L Ebert, and Eric S Fischer. Targeted protein degradation: from mechanisms to clinic. Nature Reviews Molecular Cell Biology, 25(9):740–757, 2024.

[20] Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023

[21] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A Hunter, Costas Bekas, and Alpha A Lee. Molecular transformer: a model for uncertainty-calibrated chemical reaction prediction. ACS Central Science, 5(9):1572, 2019.

[22] Igor V Tetko, Pavel Karpov, Ruud Van Deursen, and Guillaume Godin. State-of-the-art augmented NLP transformer models for direct and single-step retrosynthesis. Nature Communications, 11(1):5575, 2020. doi: 10.1038/ s41467-020-19266-y.

[23] Philippe Schwaller, Riccardo Petraglia, Valerio Zullo, Vishnu H Nair, Rico Andreas Haeuselmann, Riccardo Pisoni, Costas Bekas, Anna Iuliano, and Teodoro Laino. Predicting retrosynthetic pathways using transformer-based models and a hyper-graph exploration strategy. Chemical Science, 11(12):3316–3325, 2020.

[24] Kangjie Lin, Youjun Xu, Jianfeng Pei, and Luhua Lai. Automatic retrosynthetic route planning using template-free models. Chemical Science, 11(12):3355–3364, 2020. doi: 10.1039/C9SC03666K.

[25] Ross Irwin, Spyridon Dimitriadis, Jiazhen He, and Esben Jannik Bjerrum. Chemformer: a pre-trained transformer for computational chemistry. Machine Learning: Science and Technology, 3(1):015022, 2022.

[26] Jiazhen He, Eva Nittinger, Christian Tyrchan, Werngard Czechtizky, Atanas Patronov, Esben Jannik Bjerrum, and Ola Engkvist. Transformer-based molecular optimization beyond matched molecular pairs. Journal of Cheminformatics, 14(1):18, 2022. doi: 10.1186/s13321-022-00599-3.

[27] Linhui Yu, Yansen Su, Yuansheng Liu, and Xiangxiang Zeng. Review of unsupervised pretraining strategies for molecules representation. Briefings in Functional Genomics, 20(5):323–332, 2021.

[28] Viraj Bagal, Rishal Aggarwal, PK Vinod, and U Deva Priyakumar. MolGPT: molecular generation using a transformer-decoder model. Journal ofChemical Information and Modeling, 62(9):2064–2076, 2021.

[29] Hao Lu, Zhiqiang Wei, Xuze Wang, Kun Zhang, and Hao Liu. Graphgpt: A graph enhanced generative pretrained transformer for conditioned molecular generation. International Journal of Molecular Sciences, 24(23):16761, 2023.

[30] Ye Wang, Honggang Zhao, Simone Sciabola, and Wenlu Wang. cMolGPT: a conditional generative pre-trained transformer for target-specific de novo molecular generation. Molecules, 28(11):4430, 2023.

[31] Hengwei Chen and Jürgen Bajorath. Generative design of compounds with desired potency from target protein sequences using a multimodal biochemical language model. Journal ofCheminformatics, 16(1):55, 2024.

[32] Medard Edmund Mswahili and Young-Seob Jeong. Transformer-based models for chemical SMILES representation: A comprehensive literature review. Heliyon, 10(20), 2024.

[33] Fergus Imrie, Anthony R Bradley, Mihaela van der Schaar, and Charlotte M Deane. Deep generative models for 3D linker design. Journal ofChemical Information and Modeling, 60(4):1983, 2020.

[34] Jeff Guo, Franziska Knuth, Christian Margreitter, Jon Paul Janet, Kostas Papadopoulos, Ola Engkvist, and Atanas Patronov. Link-INVENT: generative linker design with reinforcement learning. Digital Discovery, 2(2):392–408, 2023.

[35] Shuangjia Zheng, Youhai Tan, Zhenyu Wang, Chengtao Li, Zhiqing Zhang, Xu Sang, Hongming Chen, and Yuedong Yang. Accelerated rational PROTAC design via deep learning and molecular simulations. Nature Machine Intelligence, 4(9):739–748, 2022.

[36] Chien-Ting Kao, Chieh-Te Lin, Cheng-Li Chou, and Chu-Chung Lin. Fragment linker prediction using the deep encoder-decoder network for PROTACs drug design. Journal ofChemical Information and Modeling, 63(10): 2918, 2023.

[37] Rebecca M Neeser, Mehmet Akdel, Daniel Kovtun, and Luca Naef. Reinforcement learning-driven linker design via fast attention-based point cloud alignment. arXiv preprint arXiv:2306.08166, 2023.

[38] Fenglei Li, Qiaoyu Hu, Yongqi Zhou, Hao Yang, and Fang Bai. DiffPROTACs is a deep learning-based generator for proteolysis targeting chimeras. Briefings in Bioinformatics, 25(5):bbae358, 2024.

[39] Zixing Song, Ziqiao Meng, and José Miguel Hernández-Lobato. Domain-adapted diffusion model for PROTAC linker design through the lens of density ratio in chemical space. In ICML, 2025.

[40] Shadab Ahmad, Leonardo Jose da Costa Gonzales, Emily H Bowler-Barnett, Daniel L Rice, Minjoon Kim, Supun Wijerathne, Aurélien Luciani, Swaathi Kandasaamy, Jie Luo, Xavier Watkins, et al. The UniProt website API: facilitating programmatic access to protein knowledge. Nucleic Acids Research, 53(W1):W547–W553, 2025.

[41] John J Irwin, Khanh G Tang, Jennifer Young, Chinzorig Dandarchuluun, Benjamin R Wong, Munkhzul Khurelbaatar, Yurii S Moroz, John Mayfield, and Roger A Sayle. ZINC20–a free ultra large-scale chemical database for ligand discovery. Journal ofChemical Information and Modeling, 60(12):6065, 2020.

[42] Barbara Zdrazil, Eloy Felix, Fiona Hunter, Emma J Manners, James Blackshaw, Sybilla Corbett, Marleen De Veij, Harris Ioannidis, David Mendez Lopez, Juan F Mosquera, et al. The ChEMBL database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Research, 52(D1):D1180–D1192, 2024.

[43] Jingxuan Ge, Shimeng Li, Gaoqi Weng, Huating Wang, Meijing Fang, Huiyong Sun, Yafeng Deng, Chang-Yu Hsieh, Dan Li, and Tingjun Hou. PROTAC-DB 3.0: an updated database of PROTACs with extended pharmacokinetic parameters. Nucleic Acids Research, 53(D1):D1510–D1515, 2025.

[44] Xinran Qin, Yinpeng Zhang, Yajunzi Wang, Yintao Zhang, Jiachen Jing, Yuyuan Zhang, Gaoxiang Xu, Haoping Teng, Tianjun Wang, Lei Fu, et al. TPDdb: the comprehensive database of targeted protein degrader. Nucleic Acids Research, 54(D1):D1683–D1691, 2026.

[45] Nir London and Jaime Prilusky. PROTAC-Pedia: A crowd-sourced database of PROTACs. https://protacpedia. weizmann.ac.il/, 2020. Version 1.62. Accessed: 2026-09-24.

[46] Stefano Ribes, Ranxuan Zhang, Télio Cropsal, Anders Källberg, Christian Tyrchan, Eva Nittinger, and Rocío Mercado. PROTAC-Splitter: a machine learning framework for automated identification of PROTAC substructures. Journal of Cheminformatics, 18(1):30. ISSN 1758-2946. doi: 10.1186/s13321-025-01135-9. URL https: //doi.org/10.1186/s13321-025-01135-9.

## A MODEL AND IMPLEMENTATION DETAILS

## A.1 SMILES TOKENIZATION AND AUGMENTATION

Molecules are represented as SMILES strings and tokenized using the same molecular tokenizer during pretraining, unconditional fine-tuning, and conditional fine-tuning. Each sequence begins with a beginning-ofsequence (BOS) token and ends with an end-of-sequence (EOS) token. For a tokenized sequence

$$
[ \mathrm { B O S } , s _ { 1 } , \ldots , s _ { T } , \mathrm { E O S } ] ,\tag{17}
$$

causal language modeling uses the shifted input–target pair

$$
\mathbf { x } = [ \mathrm { B O S } , s _ { 1 } , \ldots , s _ { T } ] , \qquad \mathbf { y } = [ s _ { 1 } , \ldots , s _ { T } , \mathrm { E O S } ] .\tag{18}
$$

Sequences are limited to 320 tokens. Within each minibatch, input sequences are right-padded. The corresponding padding positions in the targets are assigned the ignore index −100 and excluded from the cross-entropy loss.

SMILES randomization is performed on the fly during training. Randomized SMILES provide alternative traversals of the same molecular graph, preserving molecular identity while exposing the decoder to different sequence representations. Randomization is applied only to the training data; validation examples retain fixed SMILES representations. For conditional training, the randomized SMILES remains paired with the original POI embedding.

## A.2 PROTEIN SEQUENCE REPRESENTATION

For each POI, we retrieve the amino-acid sequence from UniProtKB using its UniProt accession. Protein representations are computed using the pretrained esm2\_t33\_650M\_UR50D model [20], which consists of 33 transformer layers, 650M parameters, and a hidden dimension of 1280.

For a $\mathrm { P O I } p$ with amino-acid sequence

$$
\mathbf { a } _ { p } = ( a _ { 1 } , \ldots , a _ { L } ) ,\tag{19}
$$

ESM-2 produces contextual residue representations. We exclude the initial and terminal special-token representations, retaining one vector per amino-acid residue,

$$
\mathbf { H } _ { p } = \left[ \mathbf { h } _ { 1 } , \ldots , \mathbf { h } _ { L } \right] ^ { \top } \in \mathbb { R } ^ { L \times 1 2 8 0 } .\tag{20}
$$

We obtain a fixed-length POI embedding by mean pooling over residues,

$$
{ \bf z } _ { p } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } { \bf h } _ { i } \in \mathbb { R } ^ { 1 2 8 0 } .\tag{21}
$$

Mean pooling removes the variable sequence-length dimension while preserving the 1280-dimensional feature space. The resulting vector $\mathbf { z } _ { p }$ is supplied to the conditional molecular model.

ESM-2 is used as a frozen feature extractor. Protein embeddings are precomputed in FP32, stored by UniProt accession, and reused during molecular-model training and generation. ESM-2 parameters are not updated during conditional fine-tuning.

## A.3 LONG-PROTEIN ENCODING

The ESM-2 model is applied to windows containing at most 1022 amino-acid residues. Proteins satisfying

$$
L  \leq 1 0 2 2\tag{22}
$$

are encoded in a single pass. Longer sequences are divided into overlapping windows of maximum length

$$
W = 1 0 2 2\tag{23}
$$

with an overlap of

$$
O = 2 5 6\tag{24}
$$

residues. The resulting stride is

$$
S = W - O = 7 6 6 .\tag{25}
$$

Overlapping windows provide additional sequence context for residues near window boundaries. Each window is encoded independently by ESM-2, and the resulting residue representations are stitched together before mean pooling.

For adjacent windows, the 256-residue overlap is divided into two equal parts. We retain the representations from the earlier window for the first 128 residues and from the later window for the remaining 128 residues. This favors representations computed farther from window boundaries, where more surrounding sequence context is available. The resulting representation contains exactly one vector per residue.

The reconstructed representation must satisfy

$$
\mathrm { r o w s } ( { \bf H } _ { p } ) = { \cal L } ,\tag{26}
$$

so that it contains exactly one row for each residue in the original sequence. Mean pooling is then applied to the reconstructed representation:

$$
\mathbf { z } _ { p } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \mathbf { H } _ { p , i } .\tag{27}
$$

We do not average window-level embeddings, because doing so would weight windows rather than individual residues and could overrepresent shorter terminal windows. Instead, each residue contributes exactly once to the final protein embedding.

## A.4 DECODER ARCHITECTURE

The molecular model is a decoder-only transformer with eight pre-LayerNorm blocks, a hidden dimension $d _ { \mathrm { m o d e l } } = 5 1 2$ , eight attention heads, a feed-forward expansion ratio of four, and dropout probability 0.05. Within each block, $\mathrm { L N } _ { 1 }$ and $\mathrm { L N _ { 2 } }$ denote the layer-normalization operations preceding the attention and MLP sublayers, respectively.

For token $s _ { t }$ at sequence position $t ,$ the input representation is the sum of learned token and positional embeddings,

$$
\mathbf { x } _ { t } ^ { ( 0 ) } = \mathbf { E } _ { \mathrm { t o k } } ( s _ { t } ) + \mathbf { E } _ { \mathrm { p o s } } ( t ) .\tag{28}
$$

The sequence representations pass through transformer blocks with causal self-attention. In block $\ell ,$ the input is first normalized,

$$
\mathbf { H } ^ { \left( \ell \right) } = \mathrm { L N } _ { 1 } \left( \mathbf { X } ^ { \left( \ell \right) } \right) .\tag{29}
$$

For one attention head, queries, keys, and values are computed as

$$
{ \bf Q } = { \bf H } ^ { ( \ell ) } { \bf W } _ { Q } , \qquad { \bf K } = { \bf H } ^ { ( \ell ) } { \bf W } _ { K } , \qquad { \bf V } = { \bf H } ^ { ( \ell ) } { \bf W } _ { V } .\tag{30}
$$

Causal scaled dot-product attention is defined as

$$
\mathrm { A t t n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { h } } } + \mathbf { M } \right) \mathbf { V } ,\tag{31}
$$

where $d _ { h } = 6 4$ is the head dimension and M is the causal mask that prevents attention to future tokens.

The eight attention-head outputs are concatenated and projected back to the model dimension. We denote this operation by MHA. The attention residual of block ℓ is

$$
\mathbf { A } ^ { \left( \ell \right) } = \mathbf { X } ^ { \left( \ell \right) } + \mathrm { M H A } \left( \mathrm { L N } _ { 1 } \left( \mathbf { X } ^ { \left( \ell \right) } \right) \right) .\tag{32}
$$

The position-wise multilayer perceptron (MLP) expands the hidden dimension by a factor of four and uses a GELU activation,

$$
\begin{array} { r } { \mathrm { M L P } ( \mathbf { x } ) = \mathbf { W } _ { 2 } \mathrm { G E L U } \left( \mathbf { W } _ { 1 } \mathbf { x } + \mathbf { b } _ { 1 } \right) + \mathbf { b } _ { 2 } . } \end{array}\tag{33}
$$

For the unconditional decoder, the block output is

$$
\begin{array} { r } { { \bf X } ^ { ( \ell + 1 ) } = { \bf A } ^ { ( \ell ) } + \mathrm { M L P } \left( \mathrm { L N } _ { 2 } \left( { \bf A } ^ { ( \ell ) } \right) \right) . } \end{array}\tag{34}
$$

After the final transformer block, layer normalization and a linear language-model head produce logits over the SMILES vocabulary. The input token embeddings and output projection share their weights.

## A.5 PROTEIN-TO-MOLECULE CONDITIONING

The conditional model retains the molecular decoder and adds a protein adapter that maps the fixed POI embedding to the decoder hidden space. The pooled ESM-2 embedding

$$
\mathbf { z } _ { p } \in \mathbb { R } ^ { 1 2 8 0 }\tag{35}
$$

is first normalized and linearly projected:

$$
\begin{array} { r } { \mathbf { c } _ { p } = \mathbf { W } _ { c } \amalg _ { p } ( \mathbf { z } _ { p } ) + \mathbf { b } _ { c } , \qquad \mathbf { W } _ { c } \in \mathbb { R } ^ { 5 1 2 \times 1 2 8 0 } , } \end{array}\tag{36}
$$

yielding

$$
\mathbf { c } _ { p } \in \mathbb { R } ^ { 5 1 2 } .\tag{37}
$$

Here, $\mathrm { L N } _ { p }$ is the protein-adapter normalization module, distinct from $\mathrm { L N _ { 1 } }$ and $\mathrm { L N _ { 2 } }$ in the decoder blocks.

The projected protein condition $\mathbf { c } _ { p }$ is computed once per forward pass and shared across all decoder blocks. Each block ℓ has a trainable scalar gate $\gamma _ { \ell } .$ . The gated condition is added after the self-attention residual

$$
\widetilde { \mathbf { A } } ^ { ( \ell ) } = \mathbf { A } ^ { ( \ell ) } + \mathrm { t a n h } ( \gamma _ { \ell } ) \mathbf { c } _ { p } .\tag{38}
$$

The protein condition is broadcast across all SMILES token positions. The conditioned block output is

$$
\begin{array} { r } { { \bf X } ^ { ( \ell + 1 ) } = \widetilde { \bf A } ^ { ( \ell ) } + \mathrm { M L P } \left( \mathrm { L N } _ { 2 } \left( \widetilde { \bf A } ^ { ( \ell ) } \right) \right) . } \end{array}\tag{39}
$$

The same projected protein embedding is used in every block, while the layer-specific gates control its contribution at each depth.

## A.6 INITIALIZATION OF THE CONDITIONAL MODEL

The conditional model is initialized from the pretrained molecular prior. Its token embeddings, positional embeddings, attention layers, feed-forward networks, normalization layers, and language-model head are transferred from the prior checkpoint.

The conditional extension adds the protein layer-normalization module $\mathrm { L N } _ { p } ,$ the 1280 → 512 linear projection, and one scalar gate in each of the eight decoder blocks. The protein LayerNorm is initialized with unit scale and zero bias, the projection is newly initialized, and all gates are initialized to zero,

$$
\gamma _ { \ell } ^ { ( 0 ) } = 0 .\tag{40}
$$

Therefore,

$$
\mathrm { t a n h } ( \gamma _ { \ell } ^ { ( 0 ) } ) \mathbf { c } _ { p } = \mathbf { 0 } .\tag{41}
$$

At initialization, the protein-conditioning pathway contributes zero to every block, so the conditional model retains the pretrained decoder’s hidden-state computation. During conditional fine-tuning, the protein adapter, layer-specific gates, and decoder parameters are optimized to incorporate POI information.

The resulting fine-tuning procedure adapts the pretrained unconditional molecular distribution

$$
P _ { \theta _ { 0 } } ( \mathbf { s } )\tag{42}
$$

to a POI-conditioned distribution

$$
P _ { \theta } ( \mathbf { s } \mid \mathbf { z } _ { p } ) ,\tag{43}
$$

where $\theta _ { 0 }$ denotes the pretrained molecular parameters.

## A.7 CAUSAL LANGUAGE-MODEL OBJECTIVE

During conditional training, each SMILES sequence is paired with the precomputed embedding of its associated POI. For a minibatch of size B, the protein embeddings are stacked into

$$
\begin{array} { r } { { \bf Z } = [ { \bf z } _ { p _ { 1 } } , \ldots , { \bf z } _ { p _ { B } } ] ^ { \top } \in \mathbb { R } ^ { B \times { 1 2 8 0 } } . } \end{array}\tag{44}
$$

The nth protein embedding is aligned with the nth SMILES sequence in the minibatch.

Training uses teacher-forced causal language modeling. The loss is the mean token-level negative loglikelihood over non-padding target positions,

$$
\mathcal { L } _ { \mathrm { L M } } = - \frac { 1 } { \sum _ { n = 1 } ^ { B } T _ { n } } \sum _ { n = 1 } ^ { B } \sum _ { t = 1 } ^ { T _ { n } } \log P _ { \theta } \left( y _ { t } ^ { ( n ) } \mid x _ { \le t } ^ { ( n ) } , \mathbf { z } _ { p _ { n } } \right) ,\tag{45}
$$

where $T _ { n }$ is the number of non-padding target tokens in sequence n.

At each position, the decoder receives the ground-truth token prefix and the fixed POI embedding to predict the next token. Padding positions are excluded from the loss. ESM-2 is not optimized; training updates the protein adapter, layer-specific gates, and molecular decoder.

## A.8 OPTIMIZATION DETAILS

Conditional fine-tuning uses a per-device training batch size of 32, a validation batch size of 32, and no gradient accumulation. The learning rate is $1 \times 1 0 ^ { - 4 }$ , with weight decay 0.01 and gradient clipping at a maximum norm of 1.0. We use a cosine learning-rate schedule with a warm-up ratio of 0.05 and bfloat16 mixed precision. The same optimization configuration is used for molecular pretraining and unconditional fine-tuning.

## A.9 AUTOREGRESSIVE GENERATION

At inference, generation begins with a single BOS token. For the conditional generation, the same POI embedding $\mathbf { z } _ { p }$ is supplied at every autoregressive decoding step.

Given decoder logits $\mathbf { o } _ { t }$ at step t, the temperature-scaled sampling distribution is

$$
P _ { \theta } \left( s _ { t } \mid s _ { < t } , \mathbf { z } _ { p } \right) = \operatorname { s o f t m a x } \left( \frac { \mathbf { o } _ { t } } { \tau } \right) ,\tag{46}
$$

where τ is the sampling temperature. PAD, BOS, and UNK tokens are excluded from sampling, while EOS remains available to terminate generation. Decoding stops when EOS is sampled, or the model context limit is reached. The sampling settings are reported in Appendix B.7.

## B ADDITIONAL EXPERIMENTAL DETAILS

## B.1 PROTAC-LIKE PRETRAINING CORPUS

The PROTAC-like pretraining corpus described in Section 4 is built using physicochemical and structural filters intended to enrich for large molecules with PROTAC-like properties. We retain compounds with molecular weight of 650–1200 Da, topological polar surface area (TPSA) of 120–320 $\mathring \mathrm { A } ^ { 2 }$ , 12–45 rotatable bonds, 8–24 hydrogen-bond acceptors, and 0–8 hydrogen-bond donors.

To favor bifunctional structures, molecules must also contain at least two distinct ring systems. We identify the two largest ring systems and require a minimum topological distance of 8–60 bonds between them.

Molecules satisfying all criteria are converted to canonical isomeric SMILES and deduplicated within each source, yielding 68,550 molecules from ZINC20 and 19,056 from ChEMBL 36. After merging and deduplicating across sources, the final corpus contains 79,406 unique PROTAC-like molecules.

## B.2 POI AND E3-LIGASE ANNOTATION HARMONIZATION

POI and E3-ligase annotations are harmonized across PROTAC-DB, TPDdb, and PROTAC-Pedia before merging the datasets. When a UniProt accession is available, we retrieve the corresponding primary gene name from UniProtKB through its REST API [40] and use it as the canonical POI label. This ensures consistent target identification across databases despite differences in the original annotations.

When an accession is unavailable, we apply manually curated normalization rules only if the annotation can be mapped unambiguously to a single protein. For example, cIAP1 is mapped to BIRC2, XIAP to BIRC4, Cereblon to CRBN, and AhR to AHR. We then use the normalized gene names to retrieve the corresponding protein annotations from UniProtKB, ensuring that these targets follow the same naming convention. Ambiguous annotations remain unresolved to avoid incorrect protein assignments.

Finally, we cross-check the mappings across the three databases to identify inconsistencies and unresolved annotations before constructing the final datasets.

## B.3 DATASET COMPOSITION AND COMPONENT-LEVEL CHEMISTRY

Figure 4 shows the composition of the PROTAC dataset, revealing substantial imbalance across POIs. The 20 most represented POIs, referred to as the seen POIs in Section 4.1, account for 21,246 molecule–POI pairs (70.7% of the complete dataset; Figure 4a,b), with counts ranging from 4,933 for AR to 230 for SMARCA4.

These targets provide sufficient data for separate training, validation, and test subsets. In contrast, many less represented POIs have too few examples for reliable held-out evaluation. We therefore include them only in training to expose the model to a broader range of protein conditions.

E3-ligase usage is also highly imbalanced. CRBN and VHL account for 20,724 and 7,690 molecules, respectively, together representing 94.5% of the complete dataset (Figure 4c). Although their relative contributions vary across the seen POIs, many targets share the same E3 ligases (Figure 4d).

To examine how the chemistry of the two binding components varies across targets, we apply PROTAC-Splitter to molecules targeting the seen POIs. We extract Bemis–Murcko scaffolds from the predicted warheads and E3 ligands, then deduplicate them within each POI.

For two POIs $p$ and $q ,$ let $ { \boldsymbol { S } } _ { p }$ and $ { \boldsymbol { S } } _ { q }$ denote their respective sets of unique component scaffolds. For each scaffold in $p ,$ we identify its most similar scaffold in q and average these maximum similarities,

$$
D ( p  q ) = \frac { 1 } { | S _ { p } | } \sum _ { s \in S _ { p } } \operatorname* { m a x } _ { t \in S _ { q } } T ( s , t ) ,\tag{47}
$$

where $T ( s , t )$ is the Tanimoto similarity between scaffold fingerprints. Because this measure depends on the comparison direction, we define a symmetric similarity,

$$
S _ { \mathrm { s y m } } ( p , q ) = \frac { 1 } { 2 } \left[ D ( p \to q ) + D ( q \to p ) \right] .\tag{48}
$$

We calculate this measure for warhead and E3-ligand scaffolds.

Figure 4e shows that E3-ligand scaffolds are substantially more similar across POIs than warhead scaffolds, with mean similarities of 0.600 and 0.247, respectively. This indicates that E3-ligand chemistry is more widely shared across targets, whereas warhead chemistry differs more strongly. Consequently, whole-PROTAC similarity may partly reflect shared E3-ligand structures rather than differences related to POI identity. To account for this potential confounding effect, we evaluate chemical similarity both globally and within fixed CRBN and VHL groups (Figure 2).

## B.4 ZERO-SHOT POI SELECTION

We select four unseen POIs to evaluate zero-shot generalization across different levels of protein similarity. Candidates must fall outside the seen-POI panel and have sufficient reference molecules for held-out evaluation. We also require that their standardized SMILES do not occur under any POI in conditional training. Thus, neither the selected POIs nor their molecules appear in the conditional training. Together, the four zero-shot POIs provide 217 held-out molecule–POI pairs. We measure protein similarity using cosine similarity between mean-pooled ESM-2 embeddings.

We first identify WDR5 and DHFR as the candidate pair most distant from the proteins available for training. For a candidate pair $( a , b )$ , let $\mathcal { P } _ { \mathrm { t r a i n } } ^ { ( a , b ) }$ denote the POIs remaining after both candidates are excluded. For each candidate $p \in \{ a , b \}$ , the distance to its nearest training POI is

$$
d _ { \mathrm { t r a i n } } ( p ) = \operatorname* { m i n } _ { q \in \mathcal { P } _ { \mathrm { t r a i n } } ^ { ( a , b ) } } \left[ 1 - \frac { \mathbf { z } _ { p } ^ { \top } \mathbf { z } _ { q } } { \| \mathbf { z } _ { p } \| _ { 2 } \| \mathbf { z } _ { q } \| _ { 2 } } \right] ,\tag{49}
$$

where $\mathbf { z } _ { p }$ and $\mathbf { z } _ { q }$ are mean-pooled ESM-2 embeddings. Larger distances correspond to lower similarity to the nearest training POI.

To ensure that both candidates are distant from training, we score each pair using the smaller of the two distances,

$$
D _ { \mathrm { p a i r } } ( a , b ) = \operatorname* { m i n } \left\{ d _ { \mathrm { t r a i n } } ( a ) , d _ { \mathrm { t r a i n } } ( b ) \right\} .\tag{50}
$$

![](images/186fbb53317a4a85f4f5a40322c2a136a1a93f7965177670e5e70f0614f2cedf.jpg)  
Figure 4 Dataset composition and component-level scaffold similarity. (a) Number of molecules of seen POIs. (b) Cumulative fraction of the dataset covered by POIs ranked by molecule count. (c) E3-ligase distribution across the complete dataset. (d) E3-ligase composition across the seen POIs. (e) Pairwise scaffold similarity across the seen POIs, comparing predicted POI-warhead scaffolds (lower triangle) and E3-ligand scaffolds (upper triangle). Similarity is measured using symmetric nearest-scaffold Tanimoto similarity.

We select the pair maximizing this score. WDR5 and DHFR have nearest-training distances of 0.1234 and 0.0864, respectively.

We then select two additional zero-shot targets to cover higher levels of protein similarity. For each remaining candidate p, we calculate its maximum cosine similarity to the seen POIs,

$$
S _ { 2 0 } ( p ) = \operatorname* { m a x } _ { q \in \mathcal { P } _ { 2 0 } } \frac { \mathbf { z } _ { p } ^ { \top } \mathbf { z } _ { q } } { \| \mathbf { z } _ { p } \| _ { 2 } \| \mathbf { z } _ { q } \| _ { 2 } } ,\tag{51}
$$

where $\mathcal { P } _ { 2 0 }$ denotes the set of seen POIs. FGFR3 lies at the median of the similarity distribution and is selected as the medium-similarity target. MAP3K13 has the highest similarity and is selected as the high-similarity target.

Table 4 reports each selected POI and its nearest match in the seen-POI panel. WDR5 and DHFR represent lower-similarity conditions, while FGFR3 and MAP3K13 provide medium- and high-similarity conditions. This selection allows us to examine whether zero-shot performance varies with similarity in ESM-2 embedding space.

Table 4 Protein-representation similarity of the four zero-shot POIs. For each zero-shot POI, the closest seen POI is identified by cosine similarity between mean-pooled ESM-2 embeddings.
<table><tr><td>Role</td><td>POI</td><td>Records</td><td>Closest top-20 POI</td><td>Cosine similarity</td></tr><tr><td>Most dissimilar</td><td>WDR5</td><td>66</td><td>PTPN1</td><td>0.760</td></tr><tr><td>Dissimilar</td><td>DHFR</td><td>43</td><td>EGFR</td><td>0.894</td></tr><tr><td>Medium</td><td>FGFR3</td><td>58</td><td>MAP4K1</td><td>0.969</td></tr><tr><td>Very similar</td><td>MAP3K13</td><td>50</td><td>AR</td><td>0.987</td></tr></table>

## B.5 CONDITIONAL DATA SPLITTING

After excluding the four zero-shot POIs, 29,836 molecule–POI pairs across 404 POIs remain for conditional model development. The seen POIs are AR, KRAS, ESR1, EGFR, CDK4, BTK, IRAK4, SMARCA2, BCL2L1, BRAF, BRD4, BCL6, BCL2, MAPT, METTL14, PTPN1, SOS1, EZH2, MAP4K1, and SMARCA4. Together, they account for 21,246 pairs (71.2% of the remaining dataset).

To prevent molecular leakage between partitions, we group all records sharing the same standardized SMILES before splitting. Each group is assigned entirely to training, validation, or test, ensuring that the same molecular structure never appears in different partitions.

We split the records of seen POIs into 80% training, 10% validation, and 10% test subsets. The resulting top-20 split contains 16,994 training, 2,126 validation, and 2,126 test pairs. The 384 training-only POIs contribute another 8,590 training pairs, yielding 25,584 training, 2,126 validation, and 2,126 test pairs overall.

The exact target-wise counts are reported in Table 5.

## B.6 POI-ASSOCIATED CHEMICAL-SPACE CHARACTERIZATION

We examine whether PROTAC chemistry in the reference dataset exhibits patterns related to POI identity. All analyses use 2048-bit Morgan fingerprints with radius 2, and chemical similarity is measured using the Tanimoto coefficient,

$$
T ( \mathbf { f } _ { i } , \mathbf { f } _ { j } ) = \frac { \mathbf { f } _ { i } ^ { \top } \mathbf { f } _ { j } } { \lvert \lvert \mathbf { f } _ { i } \rvert \rvert _ { 1 } + \lvert \lvert \mathbf { f } _ { j } \rvert \rvert _ { 1 } - \mathbf { f } _ { i } ^ { \top } \mathbf { f } _ { j } } .\tag{52}
$$

Chemical-space visualization. We sample 300 molecules from each of the 10 most represented POIs, yielding a balanced set of 3,000 molecules. Equal sampling prevents targets with more examples from dominating the visualization. We first reduce the fingerprint matrix to 50 principal components using PCA, then apply t-SNE to obtain a two-dimensional embedding.

Table 5 Target-wise counts for the seen POIs. Data for each POI are split into 80% training, 10% validation, and 10% test subsets while keeping molecular structures disjoint across partitions.
<table><tr><td>POI</td><td>Total</td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>AR</td><td>4933</td><td>3947</td><td>493</td><td>493</td></tr><tr><td>KRAS</td><td>2776</td><td>2220</td><td>278</td><td>278</td></tr><tr><td>ESR1</td><td>2212</td><td>1770</td><td>221</td><td>221</td></tr><tr><td>EGFR</td><td>1617</td><td>1293</td><td>162</td><td>162</td></tr><tr><td>CDK4</td><td>1297</td><td>1037</td><td>130</td><td>130</td></tr><tr><td>BTK</td><td>1293</td><td>1035</td><td>129</td><td>129</td></tr><tr><td>IRAK4</td><td>972</td><td>778</td><td>97</td><td>97</td></tr><tr><td>SMARCA2</td><td>955</td><td>763</td><td>96</td><td>96</td></tr><tr><td>BCL2L1</td><td>856</td><td>684</td><td>86</td><td>86</td></tr><tr><td>BRAF</td><td>749</td><td>599</td><td>75</td><td>75</td></tr><tr><td>BRD4</td><td>599</td><td>479</td><td>60</td><td>60</td></tr><tr><td>BCL6</td><td>577</td><td>461</td><td>58</td><td>58</td></tr><tr><td>BCL2</td><td>451</td><td>361</td><td>45</td><td>45</td></tr><tr><td>MAPT</td><td>350</td><td>280</td><td>35</td><td>35</td></tr><tr><td>METTL14</td><td>348</td><td>278</td><td>35</td><td>35</td></tr><tr><td>PTPN1</td><td>305</td><td>245</td><td>30</td><td>30</td></tr><tr><td>SOS1</td><td>252</td><td>202</td><td>25</td><td>25</td></tr><tr><td>EZH2</td><td>237</td><td>189</td><td>24</td><td>24</td></tr><tr><td>MAP4K1</td><td>237</td><td>189</td><td>24</td><td>24</td></tr><tr><td>SMARCA4</td><td>230</td><td>184</td><td>23</td><td>23</td></tr><tr><td>Total</td><td>21246</td><td>16994</td><td>2126</td><td>2126</td></tr></table>

Figure 2a,b shows the same coordinates colored by E3-ligase and POI identity, respectively. Broad regions reflect E3 identity, while smaller regions are enriched for individual POIs. Several POIs occupy separated regions, that their molecules span different chemotypes.

Global POI-associated similarity. We assess whether molecules targeting the same POI are more chemically similar than those targeting different POIs. Because the seen POIs have unequal numbers of records, sampling molecular pairs directly from the pooled dataset would favor targets with more examples, such as AR. We therefore balance pair sampling across POIs.

For same-POI comparisons, we uniformly sample a seen POI and select two distinct molecules targeting it. For different-POI comparisons, we uniformly sample two distinct POIs and select one molecule from each. We repeat this procedure to obtain 10,000 pairs of each type. A molecule may be reused across comparisons, but the two molecules in each same-POI pair are distinct.

Let $\mathcal { P } _ { \mathrm { s a m e } }$ and $\mathcal { P } _ { \mathrm { d i f f } }$ denote the sampled pair sets, each containing M = 10,000 pairs. Their mean Tanimoto similarities are

$$
\overline { { T } } _ { \mathrm { s a m e } } = \frac { 1 } { M } \sum _ { ( i , j ) \in \mathcal { P } _ { \mathrm { s a m e } } } T ( \mathbf { f } _ { i } , \mathbf { f } _ { j } ) , \qquad \overline { { T } } _ { \mathrm { d i f f } } = \frac { 1 } { M } \sum _ { ( i , j ) \in \mathcal { P } _ { \mathrm { d i f f } } } T ( \mathbf { f } _ { i } , \mathbf { f } _ { j } ) ,\tag{53}
$$

and their difference is

$$
\Delta _ { T } = \overline { { T } } _ { \mathrm { s a m e } } - \overline { { T } } _ { \mathrm { d i f f } } .\tag{54}
$$

Mean similarity is 0.385 for same-POI pairs and 0.190 for different-POI pairs, giving $\Delta _ { T } = 0 . 1 9 5$ . This difference shows that molecules targeting the same target are more similar on average. However, the moderate same-POI similarity also suggests that a single target can span multiple chemotypes rather than forming a single compact chemical cluster.

Controlling for E3-ligase identity. Because E3-binding ligands contribute to whole-PROTAC similarity, part of the observed signal may reflect shared E3 chemistry. To account for this, we repeat the pairwise analysis within CRBN and VHL groups. Within each group, we uniformly sample POIs to construct 10,000 same-POI and 10,000 different-POI pairs.

Same-POI similarity remains higher in both E3 groups:

$$
\begin{array} { r l r } { \mathrm { C R B N : } } & { { } \overline { { T } } \mathrm { _ { s a m e } } = 0 . 4 2 1 , } & { \overline { { T } } \mathrm { _ { d i f f } } = 0 . 2 2 3 , } \\ { \mathrm { V H L : } } & { { } \overline { { T } } \mathrm { _ { s a m e } } = 0 . 5 3 5 , } & { \overline { { T } } \mathrm { _ { d i f f } } = 0 . 3 2 9 . } \end{array}\tag{55}
$$

The similarity gaps are 0.198 for CRBN and 0.206 for VHL, comparable to the global gap of 0.195. Thus, POIspecific chemical similarity persists when E3 identity is fixed and cannot be explained solely by differences between CRBN- and VHL-based chemistry.

Local POI organization. Global pairwise similarity does not imply that all molecules targeting the same POI form a single cluster. A target may instead be represented by several chemotypes, each of which is locally enriched for that POI. We therefore assess this local organization using nearest-neighbor POI purity.

The analysis uses molecules for the seen POIs within each CRBN or VHL group. For query molecule i with POI label $y _ { i }$ , let $\mathcal { N } _ { k } ^ { ( e ) } ( i )$ denote its k nearest molecules by Tanimoto similarity within E3 group e, excluding the query itself. We consider neighborhood sizes $k = 1 , \ldots , 3 0$

For each query, POI purity is defined as the fraction of neighbors sharing its POI label,

$$
\rho _ { k } ^ { ( e ) } ( i ) = \frac { 1 } { k } \sum _ { j \in \mathcal { N } _ { k } ^ { ( e ) } ( i ) } \mathbf { 1 } \left[ y _ { j } = y _ { i } \right] .\tag{56}
$$

For an E3 group containing $N _ { e }$ query molecules, mean purity is

$$
\overline { { \rho } } _ { k } ^ { ( e ) } = \frac { 1 } { N _ { e } } \sum _ { i = 1 } ^ { N _ { e } } \rho _ { k } ^ { ( e ) } ( i ) .\tag{57}
$$

Within-E3 shuffled-label null. Unequal POI frequencies can increase same-POI neighborhood matches by chance. To account for this, we construct a shuffled-label null separately within each E3 group. Molecular fingerprints and nearest-neighbor relationships remain fixed, while POI labels are randomly permuted. This preserves target frequencies but breaks the relationship between POI identity and chemical structure.

Let $y _ { i } ^ { ( b ) }$ denote the label assigned to molecule i in permutation b. The corresponding query-level null purity is

$$
\rho _ { k , \mathrm { n u l l } } ^ { ( e , b ) } ( i ) = \frac { 1 } { k } \sum _ { j \in \mathcal { N } _ { k } ^ { ( e ) } ( i ) } \mathbf { 1 } \left[ y _ { j } ^ { ( b ) } = y _ { i } ^ { ( b ) } \right] ,\tag{58}
$$

and its mean across the E3 group is

$$
\overline { { \rho } } _ { k , \mathrm { n u l l } } ^ { ( e , b ) } = \frac { 1 } { N _ { e } } \sum _ { i = 1 } ^ { N _ { e } } \rho _ { k , \mathrm { n u l l } } ^ { ( e , b ) } ( i ) .\tag{59}
$$

We repeat the permutation procedure B = 300 times per E3 group. The mean null purity is

$$
\overline { { \rho } } _ { k , \mathrm { n u l l } } ^ { ( e ) } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \overline { { \rho } } _ { k , \mathrm { n u l l } } ^ { ( e , b ) } ,\tag{60}
$$

with the 2.5th–97.5th percentile range across permutations shown in Figure 2d.

Observed purity remains substantially higher than the shuffled-label null across neighborhood sizes. At $k = 3 0$ , purity reaches 0.977 for CRBN and 0.933 for VHL, compared with null expectations of 0.124 and 0.137, respectively. This indicates that local enrichment for POI identity cannot be explained by unequal target frequencies.

Interpretation. Two complementary patterns emerge. Globally, molecules targeting the same POI are more similar than those targeting different POIs, even within fixed E3 groups. However, the moderate mean same-POI similarity indicates that individual targets can span multiple chemotypes.

Locally, molecules with similar chemical structures are strongly enriched for the same POI, well above the shuffled-label null. PROTAC chemical space is therefore globally heterogeneous but locally organized with respect to POI identity.

These findings motivate complementary fidelity measures: likelihood preference under the requested protein condition and POI enrichment among held-out reference neighbors. Neither measure assumes one chemical centroid per POI.

## B.7 GENERATION AND METRIC IMPLEMENTATION

The same decoding settings are used for the final unconditional and conditional generation experiments. Generation begins with the BOS token and uses temperature τ = 1.0 and nucleus sampling with $p = 0 . 9$ without top-k filtering. We repeat generation using seeds 71, 101, and 123. PAD, BOS, and UNK tokens are excluded from sampling, while EOS can terminate generation. The maximum sequence length is 320 tokens, allowing up to 319 new tokens after BOS.

Generated SMILES are parsed and sanitized using RDKit. Molecules that pass both steps are considered valid and converted to canonical isomeric SMILES for deduplication.

For conditional evaluation, deduplication is performed independently for each POI. Repeated molecules count only once within a given target but can be counted separately if generated under different POI conditions. Novelty is evaluated against the entire conditional training set, regardless of target identity. A molecule is therefore considered non-novel if its canonical structure appears anywhere in the training data.

Internal diversity is calculated over unique valid molecules using 2048-bit Morgan fingerprints with radius 2. When more than 1,000 molecules are available, a random subset of 1,000 is selected for pairwise Tanimoto similarity calculations.

Physicochemical descriptors are also computed over unique valid molecules, including molecular weight, TPSA, LogP, heavy-atom count, rotatable-bond count, aromatic and non-aromatic ring counts, and hydrogenbond acceptor and donor counts.

## B.8 PHYSICOCHEMICAL COMPARISON OF PROTAC-LIKE, PROTAC, AND GENERATED MOLECULES

Figure 5 compares physicochemical distributions across the PROTAC-like pretraining corpus, reference PROTACs, and conditionally generated molecules. Each property uses shared histogram bins and normalized densities across datasets. Dashed vertical lines indicate the mean of each distribution.

The PROTAC-like corpus broadly overlaps the reference PROTAC distributions, with differences consistent with its property-based construction from general molecular databases. Generated molecules also overlap the reference distributions across these descriptors. This comparison complements the generation-quality metrics in Section 4.3.

![](images/1cc0b21e480191f91438e58aeabb3a830477c0dfe67584c584f7fcf982eaf133.jpg)

![](images/5f80fa7554fe1482fe0763e8303e14a56a35aa0a1244c0e8c4a5d96356d7d98e.jpg)

![](images/a8a8118b940f604d596d78d09f711ac44d7b07077787774d19d6cb6dd99c9fa8.jpg)

![](images/38fe66fda6e782612dc9543c0cf353223cbd50e0c72fe31e0f89d9bfd6a9d59f.jpg)

![](images/544f362ac987349382cd70a9a1fb7e423b1509066808aff933cfd6d671113054.jpg)

![](images/a842f5e0b3bc7fde84a5756f32f79e31f8f62c2a71d592c564912c030a811fdf.jpg)

![](images/e432d4b684233e5701b70b8fa133915ce3d99421b31984cb85099bb0043f8e3f.jpg)

![](images/2d738edf5807f5de414f6754b18641c8bc49b5a559619443874688b30910eeed.jpg)  
Figure 5 Physicochemical distributions of PROTAC-like, reference PROTAC, and generated molecules. Normalized density distributions for PROTAC-like structures, reference $\mathrm { P R O T A C s } ,$ and molecules generated by the conditional model. Dashed vertical lines indicate distribution means. (a) Molecular weight (MW). (b) LogP. (c) Topological polar surface area (TPSA). (d) Number of rotatable bonds. (e) Number of aromatic rings. (f) Number of non-aromatic rings. (g) Number of hydrogen-bond acceptors (HBA). (h) Number of hydrogen-bond donors (HBD).

## B.9 CONDITIONAL-FIDELITY EVALUATION

Conditional fidelity is evaluated using the held-out test set. The reference pool consists of unique test molecules, each with its corresponding POI labels. For a reference molecule x, we define the set of POIs it targets as

$$
{ \mathcal { P } } ( x ) = \{ p \in { \mathcal { T } } : x { \mathrm { ~ i s ~ a n n o t a t e d ~ a s ~ t a r g e t i n g ~ } } p { \mathrm { ~ i n ~ t h e ~ h e l d - o u t ~ t e s t ~ s e t } } \} ,\tag{61}
$$

where $\tau$ is the set of POIs included in the evaluation.

For example, suppose the evaluation includes $\mathcal { T } = \{ \mathrm { A R , B T K , E G F R } \}$ . If a held-out molecule x targets both AR and BTK, then $\mathcal { P } ( x ) = \{ \mathrm { A R } , \mathrm { B T K } \}$ . The molecule appears once in the reference pool, but both AR and BTK are treated as correct conditions.

Likelihood-based conditional fidelity. Each reference molecule is evaluated under every POI condition in T by computing its negative log-likelihood (NLL).

For each POI $p \in \mathcal { P } ( x )$ , the pair $( x , p )$ defines one correct-condition evaluation. The competing POIs are defined as

$$
{ \mathcal { C } } ( x ) = { \mathcal { T } } \backslash { \mathcal { P } } ( x ) .\tag{62}
$$

This excludes all known targets of x from the competing set, ensuring that a correct POI is never treated as an incorrect condition.

For each correct molecule–POI pair $( x , p )$ , conditional preference is calculated as

$$
\Delta _ { \mathrm { N L L } } ( x , p ) = \frac { 1 } { \vert \mathcal { C } ( x ) \vert } \sum _ { q \in \mathcal { C } ( x ) } \mathrm { N L L } ( x \mid q ) - \mathrm { N L L } ( x \mid p ) .\tag{63}
$$

A positive value indicates lower NLL under the correct condition than the mean across competing conditions. For a multi-target molecule, we calculate the score separately for each correct POI. If $\mathcal C ( x )$ is empty, the molecule is excluded because there is no competing condition.

In the previous example, $\mathcal { P } ( x ) ~ = ~ \{ \mathrm { A R } , \mathrm { B T K } \}$ , so ${ \mathcal C } ( x ) = \{ \mathrm { E G F R } \}$ . The molecule contributes two evaluations, $( x , \mathrm { A R } )$ and $( x , \mathrm { B T K } )$ , with EGFR as the only competing condition. Neither AR nor BTK is treated as incorrect when evaluating the other.

Nearest-neighbor query construction. Nearest-neighbor fidelity uses valid, unique, training-novel generated molecules. A generated molecule may be absent from training yet identical to a held-out test molecule. We retain such queries because they represent recovery of reference chemistry not seen during fine-tuning.

Generated queries and reference molecules are encoded using 2048-bit Morgan fingerprints with radius 2. For each molecule $g$ generated under POI $p ,$ we calculate its Tanimoto similarity to every molecule in the reference pool. Its k nearest reference molecules form the nearest-neighbor set $\textstyle { \mathcal { N } } _ { k } ( g )$ , where $k \in \{ 5 , 1 0 , 2 0 \}$

Multi-target handling in nearest-neighbor purity. For a molecule g generated under condition $p ,$ a reference neighbor r contributes to purity whenever $p$ is among the POIs targeted by $r \mathrm { : }$

$$
p \in { \mathcal { P } } ( r ) .\tag{64}
$$

This allows a reference molecule targeting multiple POIs to count as a correct neighbor for any of those targets. POI purity is defined as

$$
\rho _ { k } ( g , p ) = \frac { 1 } { k } \sum _ { r \in \mathcal { N } _ { k } ( g ) } \mathbf { 1 } \left[ p \in \mathcal { P } ( r ) \right] ,\tag{65}
$$

as defined in Section 4.3. The indicator equals 1 if reference molecule r targets POI $p$ and 0 otherwise. Purity is calculated for each generated query and then averaged across queries for the same POI.

For example, the reference molecule x targeting AR and BTK contributes to purity for queries generated under either AR or BTK. It does not contribute to purity for EGFR queries because EGFR $\not \in { \mathcal { P } } ( x )$

## B.10 POI-WARHEAD PROCESSING

POI-warhead evaluation is performed using unique valid PROTACs generated under each POI condition. We apply PROTAC-Splitter to decompose these molecules into their structural components for subsequent warhead analysis.

Warhead extraction. PROTAC-Splitter returns three dot-separated SMILES components in the following order:

$$
{ \mathrm { E 3 ~ l i g a n d ~ . ~ l i n k e r ~ . ~ P O I ~ w a r h e a d . } }\tag{66}
$$

We retain predictions with exactly three components and identify the third as the POI-binding warhead. Predictions with missing or additional components are excluded because the warhead cannot be reliably identified.

Parent-warhead construction. The extracted warhead may contain an attachment marker, [\*:1], at its linker connection. This marker identifies the split point but is not part of the standalone warhead used for evaluation. We replace it with hydrogen to obtain a parent warhead for chemical comparison.

We parse each predicted fragment with RDKit. If an attachment marker is present, its dummy atom must have exactly one neighbor connected by a single bond. We replace the dummy atom with hydrogen and sanitize the molecule. Explicit hydrogens are then removed where chemically appropriate, followed by a second sanitization. We exclude fragments if conversion or sanitization fails or if any dummy atom remains.

We convert successfully processed parent warheads to canonical isomeric SMILES and apply the same procedure to training and generated PROTACs.

Target-specific training reference. For each POI $p ,$ we extract warheads from conditional training PRO-TACs targeting $p .$ We apply the same parent-warhead construction procedure and remove duplicate structures. The resulting set of unique canonical parent warheads forms the target-specific training reference ${ \mathcal R } _ { p } ^ { W }$

A generated warhead is considered non-novel for POI p only if its canonical structure appears in ${ \mathcal R } _ { p } ^ { W }$ . Its occurrence under another POI does not affect novelty for the current target.

Warhead uniqueness, novelty, and internal diversity are defined in Section 4.3. Internal diversity is calculated over unique successfully processed generated warheads using 2048-bit Morgan fingerprints with radius 2.

Processing coverage. Warhead evaluation requires two successful processing steps. First, PROTAC-Splitter must produce a usable three-component decomposition. Second, the extracted POI warhead must be converted into a valid parent structure. We measure the coverage of each step.

For POI $p ,$ let $N _ { p } ^ { G }$ denote the number of unique valid generated PROTACs, $N _ { p } ^ { S }$ the number with a usable three-component decomposition, and $N _ { p } ^ { W }$ the number successfully converted to parent warheads. We define the coverage metrics as

$$
\operatorname { S p l i t } \mathrm { C o v } ( p ) = \frac { N _ { p } ^ { S } } { N _ { p } ^ { G } } , \qquad \operatorname { W a r h e a d C o v } ( p ) = \frac { N _ { p } ^ { W } } { N _ { p } ^ { S } } .\tag{67}
$$

SplitCov is the proportion of generated PROTACs successfully decomposed, and WarheadCov is the proportion of those decompositions yielding valid parent warheads.

Target-wise coverage results are reported in Appendix Table 7.

## C ADDITIONAL RESULTS

## C.1 POI-WARHEAD GENERATION ANALYSIS

Table 6 reports warhead-level generation metrics for the seen POIs, complementing the results in Section 5.2. Warheads are extracted using PROTAC-Splitter and processed as described in Appendix B.10. Metrics are evaluated across three generation seeds.

Warhead diversity varies across POIs. BRD4 and MAP4K1 have the highest internal diversity, while METTL14 and BCL2 have the lowest.

Molecular uniqueness also differs from warhead uniqueness. For example, BCL6 achieves 28.50% molecular uniqueness but only 8.61% warhead uniqueness, indicating that distinct PROTACs can share the same POI-binding warhead.

Table 6 Warhead-level generation metrics across the seen POIs. Warheads are identified using PROTAC-Splitter and converted to parent structures. Uniqueness is measured over successfully processed parent warheads, and novelty is evaluated against parent warheads extracted from training PROTACs for the same POI. Values are reported as mean (sample standard deviation) across three seeds. Best per-target values in each column are shown in bold.
<table><tr><td>POI</td><td>Unique PROTACs</td><td>Unique warheads</td><td>Warhead uniqueness (%)</td><td>Warhead novelty (%)</td><td>Warhead diversity</td></tr><tr><td>AR</td><td>4,847 (50)</td><td>1,005 (30)</td><td>20.75 (0.40)</td><td>61.20 (0.39)</td><td>0.811 (0.001)</td></tr><tr><td>ESR1</td><td>4,093 (37)</td><td>893 (33)</td><td>21.85 (0.64)</td><td>69.09 (0.50)</td><td>0.776 (0.005)</td></tr><tr><td>CDK4</td><td>3,047 (13)</td><td>619 (15)</td><td>20.31 (0.52)</td><td>72.99 (1.88)</td><td>0.780 (0.004)</td></tr><tr><td>KRAS</td><td>4,954 (25)</td><td>898 (29)</td><td>18.12 (0.62)</td><td>80.01 (1.10)</td><td>0.749 (0.005)</td></tr><tr><td>EZH2</td><td>3,100 (54)</td><td>788 (41)</td><td>25.42 (1.13)</td><td>96.14 (0.23)</td><td>0.758 (0.005)</td></tr><tr><td>BTK</td><td>4,214 (24)</td><td>1,350 (50)</td><td>32.04 (1.05)</td><td>84.86 (0.53)</td><td>0.817 (0.001)</td></tr><tr><td>BRD4</td><td>4,045 (32)</td><td>1,125 (25)</td><td>27.86 (0.46)</td><td>94.13 (0.19)</td><td>0.869 (0.001)</td></tr><tr><td>BRAF</td><td>2,869 (24)</td><td>450 (22)</td><td>15.70 (0.71)</td><td>87.49 (0.81)</td><td>0.756 (0.003)</td></tr><tr><td>SMARCA4</td><td>4,061 (38)</td><td>1,291 (45)</td><td>31.80 (0.88)</td><td>95.87 (0.13)</td><td>0.766 (0.006)</td></tr><tr><td>MAP4K1</td><td>3,332 (21)</td><td>1,163 (17)</td><td>34.93 (0.33)</td><td>96.41 (0.18)</td><td>0.835 (0.003)</td></tr><tr><td>IRAK4</td><td>5,580 (63)</td><td>1,588 (26)</td><td>28.47 (0.27)</td><td>90.54 (0.55)</td><td>0.816 (0.002)</td></tr><tr><td>EGFR</td><td>5,096 (38)</td><td>1,384 (11)</td><td>27.17 (0.29)</td><td>82.85 (0.48)</td><td>0.803 (0.002)</td></tr><tr><td>BCL2L1</td><td>3,685 (25)</td><td>784 (36)</td><td>21.28 (0.94)</td><td>85.09 (1.09)</td><td>0.762 (0.006)</td></tr><tr><td>SMARCA2</td><td>4,152 (44)</td><td>1,352 (34)</td><td>32.56 (0.51)</td><td>87.66 (0.62)</td><td>0.776 (0.001)</td></tr><tr><td>METTL14</td><td>1,372 (28)</td><td>156 (5)</td><td>11.39 (0.22)</td><td>86.13 (0.68)</td><td>0.581 (0.003)</td></tr><tr><td>PTPN1</td><td>2,401 (23)</td><td>491 (25)</td><td>20.48 (0.98)</td><td>93.39 (0.62)</td><td>0.760 (0.007)</td></tr><tr><td>BCL2</td><td>2,809 (42)</td><td>1,417 (15)</td><td>50.47 (0.55)</td><td>95.32 (0.07)</td><td>0.633 (0.001)</td></tr><tr><td>SOS1</td><td>3,374 (60)</td><td>996 (48)</td><td>29.51 (0.91)</td><td>95.64 (0.21)</td><td>0.786 (0.006)</td></tr><tr><td>BCL6</td><td>2,830 (64)</td><td>244 (4)</td><td>8.61 (0.20)</td><td>85.91 (0.94)</td><td>0.791 (0.006)</td></tr><tr><td>MAPT</td><td>3,004 (17)</td><td>594 (16)</td><td>19.78 (0.56)</td><td>91.75 (0.38)</td><td>0.729 (0.005)</td></tr><tr><td>Macro</td><td>3,643 (3)</td><td>929 (8)</td><td>24.93 (0.20)</td><td>86.62 (0.15)</td><td>0.768 (0.001)</td></tr></table>

Warhead novelty does not establish target relevance. SMARCA4, for example, achieves 95.87% warhead novelty but only 27.37% generated 5-NN purity.

Warhead coverage. PROTAC-Splitter produces usable three-component decompositions for nearly all unique valid generated PROTACs. Across the seen POIs, macro splitter coverage reaches 99.99%, and 99.97% of the extracted warheads are successfully converted to valid parent structures. Table 7 reports the target-wise results.

## C.2 POI-NEIGHBORHOOD PURITY: TARGET-WISE REFERENCE, CONDITIONAL, AND UNCONDI-TIONAL MODELS

We compare the k-NN purity of held-out reference molecules with that of conditional and unconditional generation. Reference purity is calculated using leave-one-out nearest neighbors within the test set, while both generated molecular sets are evaluated against the same held-out reference pool. This comparison provides context for interpreting generated neighborhood enrichment relative to the local POI organization of the reference chemistry.

Seen POIs. Table 8 reports the target-wise reference, conditional, and unconditional k-NN purity values across the seen POIs. Macro 5-NN purity reaches 92.27% for the reference molecules, 84.73% for TACFormer, and 5.02% for unconditional generation. Conditional purity exceeds unconditional purity for every target at $k \in \{ 5 , 1 0 , 2 0 \}$ , confirming consistent improvements in POI-specific neighborhood enrichment.

The reference–generation gap is largest for SMARCA4 at k = 5, whereas METTL14 and BCL6 show closely matched purity. These differences indicate that the degree of local POI enrichment varies across the conditional output distributions.

Table 7 PROTAC-Splitter and warhead processing coverage for seen POIs. Splitter coverage measures the proportion of unique valid generated PROTACs with a usable three-component decomposition. Warhead processing coverage measures the proportion of extracted POI warheads successfully converted to valid parent structures. Values are reported as mean (sample standard deviation) across three generation seeds. The macro row reports the unweighted mean across POIs.

<table><tr><td>POI</td><td>Splitter coverage (%)</td><td>Warhead processing (%)</td></tr><tr><td>AR</td><td>100.00 (0.00)</td><td>99.91 (0.03)</td></tr><tr><td>ESR1</td><td>99.99 (0.01)</td><td>99.88 (0.07)</td></tr><tr><td>CDK4</td><td>100.00 (0.00)</td><td>100.00 (0.00)</td></tr><tr><td>KRAS</td><td>100.00 (0.00)</td><td>100.00 (0.00)</td></tr><tr><td>EZH2</td><td>100.00 (0.00)</td><td>99.96 (0.02)</td></tr><tr><td>BTK</td><td>100.00 (0.00)</td><td>100.00 (0.00)</td></tr><tr><td>BRD4</td><td>99.99 (0.01)</td><td>99.88 (0.04)</td></tr><tr><td>BRAF</td><td>99.99 (0.02)</td><td>100.00 (0.00)</td></tr><tr><td>SMARCA4</td><td>99.98 (0.03)</td><td>100.00 (0.00)</td></tr><tr><td>MAP4K1</td><td>99.92 (0.02)</td><td>99.97 (0.00)</td></tr><tr><td>IRAK4</td><td>100.00 (0.00)</td><td>99.97 (0.01)</td></tr><tr><td>EGFR</td><td>100.00 (0.00)</td><td>99.97 (0.01)</td></tr><tr><td>BCL2L1</td><td>99.99 (0.02)</td><td>99.99 (0.02)</td></tr><tr><td>SMARCA2</td><td>99.99 (0.01)</td><td>99.99 (0.01)</td></tr><tr><td>METTL14</td><td>100.00 (0.00)</td><td>100.00 (0.00)</td></tr><tr><td>PTPN1</td><td>99.97 (0.05)</td><td>99.87 (0.04)</td></tr><tr><td>BCL2</td><td>99.99 (0.02)</td><td>99.98 (0.02)</td></tr><tr><td>SOS1</td><td>100.00 (0.00)</td><td>99.98 (0.02)</td></tr><tr><td>BCL6</td><td>100.00 (0.00)</td><td>100.00 (0.00)</td></tr><tr><td>MAPT</td><td>99.97 (0.00)</td><td>99.99 (0.02)</td></tr><tr><td>Macro</td><td>99.99 (0.00)</td><td>99.97 (0.00)</td></tr></table>

Table 8 Reference, conditional, and unconditional k-NN purity for seen POIs. Reference purity is fixed across seeds. TACFormer and unconditional results are mean (sample standard deviation) across three seeds. All values are percentages; macro values are unweighted means across seen POIs. Bold values indicate the highest target-wise results within each evaluation type and neighborhood size.
<table><tr><td>POI</td><td>Evaluation</td><td>5-NN</td><td>10-NN</td><td>20-NN</td></tr><tr><td rowspan="2">AR</td><td>Reference TACFormer</td><td>98.50 97.89 (0.15)</td><td>98.05 97.32 (0.16)</td><td>96.68 96.33 (0.15)</td></tr><tr><td>Unconditional Reference</td><td>22.75 (0.14) 97.74</td><td>23.18 (0.13) 97.42</td><td>24.01 (0.15) 94.89</td></tr><tr><td>ESR1 CDK4</td><td>TACFormer Unconditional Reference</td><td>97.94 (0.20) 9.91 (0.02) 98.31</td><td>97.12 (0.25) 10.27 (0.02) 96.69</td><td>94.78 (0.23) 10.68 (0.03) 95.19</td></tr><tr><td>KRAS</td><td>TACFormer Unconditional Reference</td><td>95.17 (0.10) 5.80 (0.08) 98.35</td><td>93.62 (0.23) 5.68 (0.09) 96.26</td><td>92.66 (0.36) 5.65 (0.09) 92.79</td></tr><tr><td>EZH2</td><td>TACFormer Unconditional Reference TACFormer</td><td>97.91 (0.08) 9.96 (0.02) 95.83 82.38 (0.25)</td><td>97.18 (0.06) 10.04 (0.04) 90.83 74.63 (0.13)</td><td>95.56 (0.09) 10.15 (0.04) 70.42 56.15 (0.30)</td></tr><tr><td>BTK</td><td>Unconditional Reference TACFormer Unconditional</td><td>1.56 (0.03) 92.71 88.58 (0.21)</td><td>1.62 (0.03) 90.08 85.50 (0.20)</td><td>1.51 (0.02) 85.54 81.07 (0.21) 6.32 (0.01)</td></tr><tr><td>BRD4</td><td>Reference TACFormer Unconditional</td><td>6.26 (0.02) 81.00 66.19 (0.77) 5.32 (0.04)</td><td>6.27 (0.02) 72.33 59.97 (0.82) 5.03 (0.03)</td><td>57.08 50.83 (0.73) 4.59 (0.01)</td></tr><tr><td>BRAF</td><td>Reference TACFormer Unconditional</td><td>98.13 92.59 (0.27) 2.94 (0.08)</td><td>95.60 90.22 (0.33) 3.06 (0.07)</td><td>83.40 80.50 (0.32) 3.09 (0.05)</td></tr><tr><td>SMARCA4</td><td>Reference TACFormer Unconditional</td><td>77.39 27.37 (1.20) 0.93 (0.02)</td><td>73.48 26.29 (1.17) 0.91 (0.02)</td><td>60.22 23.59 (0.98) 0.89 (0.02)</td></tr><tr><td>MAP4K1</td><td>Reference TACFormer Unconditional</td><td>70.00 61.59 (0.74) 1.54 (0.02)</td><td>43.33 40.43 (0.49) 1.27 (0.01)</td><td>23.33 22.65 (0.15) 1.07 (0.01)</td></tr><tr><td>IRAK4</td><td>Reference TACFormer Unconditional</td><td>97.32 93.97 (0.11) 3.71 (0.04)</td><td>96.39 92.49 (0.18) 3.80 (0.04)</td><td>85.72 85.74 (0.09) 3.74 (0.03)</td></tr><tr><td>EGFR</td><td>Reference TACFormer Unconditional</td><td>90.99 90.86 (0.16) 14.23 (0.06)</td><td>84.94 85.26 (0.23) 13.41 (0.06)</td><td>76.33 76.40 (0.34) 12.50 (0.04)</td></tr><tr><td>BCL2L1 SMARCA2</td><td>Reference TACFormer Unconditional Reference</td><td>94.88 91.71 (0.38) 2.51 (0.03)</td><td>90.00 88.58 (0.54) 2.56 (0.03)</td><td>83.66 82.54 (0.67) 2.58 (0.02)</td></tr><tr><td>METTL14</td><td>TACFormer Unconditional</td><td>95.63 79.30 (0.32) 2.86 (0.03)</td><td>87.08 73.34 (0.22) 2.75 (0.04)</td><td>81.61 67.61 (0.11) 2.69 (0.04)</td></tr><tr><td></td><td>Reference TACFormer Unconditional</td><td>100.00 99.57 (0.14) 2.52 (0.03)</td><td>100.00 99.55 (0.15) 2.89 (0.03)</td><td>100.00 99.41 (0.13) 3.28 (0.02)</td></tr><tr><td>PTPN1</td><td>Reference TACFormer Unconditional</td><td>90.67 81.92 (0.63) 1.01 (0.02)</td><td>86.33 78.96 (0.65) 1.00 (0.03)</td><td>84.33 77.22 (0.66) 1.02 (0.03)</td></tr><tr><td>BCL2</td><td>Reference TACFormer Unconditional</td><td>86.67 75.60 (0.17) 1.42 (0.03)</td><td>79.56 71.92 (0.25) 1.46 (0.03)</td><td>70.89 64.41 (0.48) 1.37 (0.03)</td></tr><tr><td>SOS1</td><td>Reference TACFormer Unconditional Reference</td><td>90.40 86.51 (0.49) 1.29 (0.01)</td><td>69.20 74.80 (0.27) 1.36 (0.01)</td><td>42.40 51.12 (0.14) 1.29 (0.01)</td></tr><tr><td>BCL6</td><td>TACFormer Unconditional Reference</td><td>100.00 98.60 (0.09) 1.75 (0.02) 90.86</td><td>100.00 98.52 (0.13) 1.86 (0.02)</td><td>98.19 98.04 (0.14) 2.03 (0.02) 84.00</td></tr><tr><td>MAPT</td><td>TACFormer Unconditional</td><td>89.00 (0.54) 2.25 (0.02)</td><td>86.86 86.40 (0.54) 2.06 (0.02)</td><td>83.60 (0.51) 1.98 (0.01)</td></tr><tr><td>Macro</td><td>Reference TACFormer Unconditional</td><td>92.27 84.73 (0.11) 5.02 (0.00)</td><td>86.72 80.60 (0.11) 5.02 (0.00)</td><td>78.33 74.01 (0.14) 5.02 (0.00)</td></tr></table>

Zero-shot POIs. Table 9 compares held-out reference and generated k-NN purity for the four zero-shot POIs. At k = 5, macro reference purity reaches 99.25%, compared with 30.52% for generated molecules. The gap persists at k = 10 and k = 20. Generated 5-NN purity ranges from 56.23% for WDR5 to 14.57% for MAP3K13. Thus, the reference data show strong local POI organization, but the conditional model achieves much weaker enrichment under unseen protein conditions.

Table 9 Reference versus conditional k-NN purity for four zero-shot POIs. Reference values are fixed across seeds. Generated values are reported as mean (sample standard deviation) across three generation seeds. All values are percentages. Macro values are unweighted means across the four POIs.
<table><tr><td>POI</td><td>Ref. 5-NN</td><td>Gen. 5-NN</td><td>Ref. 10-NN</td><td>Gen. 10-NN</td><td>Ref. 20-NN</td><td>Gen. 20-NN</td></tr><tr><td>WDR5</td><td>100.00</td><td>56.23 (0.10)</td><td>97.88</td><td>55.35 (0.35)</td><td>94.09</td><td>50.58 (0.44)</td></tr><tr><td>DHFR</td><td>98.60</td><td>31.05 (0.32)</td><td>94.42</td><td>37.06 (0.28)</td><td>90.70</td><td>45.18 (0.14)</td></tr><tr><td>FGFR3</td><td>100.00</td><td>20.26 (0.15)</td><td>100.00</td><td>20.51 (0.07)</td><td>100.00</td><td>20.70 (0.05)</td></tr><tr><td>MAP3K13</td><td>98.40</td><td>14.57 (0.46)</td><td>96.80</td><td>13.27 (0.31)</td><td>95.30</td><td>12.71 (0.12)</td></tr><tr><td>Macro</td><td>99.25</td><td>30.52 (0.06)</td><td>97.27</td><td>31.55 (0.11)</td><td>95.02</td><td>32.29 (0.14)</td></tr></table>

## C.3 CHEMICAL SPACE ORGANIZATION OF GENERATED PROTACS

Figure 6 visualizes molecules generated under the seen POI conditions. Each molecule is represented by a 2048-bit Morgan fingerprint with radius 2, and all generated molecules are projected jointly into a two-dimensional t-SNE space.

The embedding reveals POI-dependent regional structure, with molecules generated under the same condition can occupy multiple regions rather than a single compact cluster. This resembles the heterogeneous, multi modal pattern observed in the reference PROTAC data, where individual POIs also span distinct regions of chemical space. The visualization complements the quantitative neighborhood-purity results in Section 5.2.

![](images/54ea9399842edad17ad244a4f54d43a53f985c5a42c6100d851f046ed0955260.jpg)  
Figure 6 Generated PROTACs retain the heterogeneous chemical-space organization observed in reference data. Two-dimensional t-SNE projection of novel molecules generated under the seen POI conditions, colored by conditioning POI. Molecules are represented using 2048-bit Morgan fingerprints with radius 2 and projected using a single common t-SNE fit. Example structures are novel generated PROTACs, with arrows indicating their conditioning POIs.

## D GENERATION DYNAMICS DURING FINE-TUNING

## D.1 GENERATION QUALITY AND CONDITIONAL FIDELITY DURING FINE-TUNING

We monitor molecular generation quality throughout fine-tuning using validity, uniqueness, novelty, and internal diversity. These metrics capture different aspects of generation and do not necessarily improve together.

For the unconditional model, validity generally increases during fine-tuning, while novelty decreases. Uniqueness remains high in the monitoring samples, and internal diversity changes little across epochs.

The conditional model shows a similar trade-off. Validity increases from 90.42% at epoch 2 to 98.84% at epoch 60. In contrast, uniqueness decreases from 97.64% to 71.30%, while novelty decreases from 97.60% to 72.33%. Internal diversity remains relatively stable, ranging from 0.775 to 0.798.

Conditional fidelity also improves throughout training. $\Delta _ { \mathrm { N L I } }$ increases from 0.0155 to 0.4155, while generated 5-NN purity rises from 9.48% to 87.12%, reaching its highest value of 89.72% at epoch 58. These trends indicate that fine-tuning strengthens the model’s preference for the correct POI condition and increases enrichment in target-associated chemical neighborhoods.

Figures 7 and 8 show the generation and conditional-fidelity trajectories during fine-tuning.

## D.2 CHECKPOINT SELECTION

Conditional model. Checkpoint selection for the conditional model must account for both generation quality and sensitivity to the protein condition. We therefore use a composite score that combines molecular generation quality with conditional fidelity. Checkpoint selection uses the validation set to compute $\Delta _ { \mathrm { N L I } }$ and generated 5-NN purity.

Generation quality is defined as

$$
Q _ { \mathrm { g e n } } ( e ) = ( V _ { e } U _ { e } N _ { e } D _ { e } ) ^ { 1 / 4 } ,\tag{68}
$$

where $V _ { e } , U _ { e } , N _ { e }$ , and $D _ { e }$ denote validity, uniqueness, novelty, and internal diversity at epoch e, respectively. Conditional fidelity combines generated 5-NN purity with normalized likelihood preference:

$$
F _ { \mathrm { c o n d } } ( e ) = \Big ( P _ { 5 } ( e ) \widetilde { \Delta } _ { \mathrm { N L L } } ( e ) \Big ) ^ { 1 / 2 } ,\tag{69}
$$

where

$$
\widetilde { \Delta } _ { \mathrm { N L L } } ( e ) = \frac { \operatorname* { m a x } ( 0 , \Delta _ { \mathrm { N L L } } ( e ) ) } { \operatorname* { m a x } _ { e ^ { \prime } } \operatorname* { m a x } ( 0 , \Delta _ { \mathrm { N L L } } ( e ^ { \prime } ) ) } .\tag{70}
$$

The final checkpoint score is

$$
S _ { \mathrm { c k p t } } ( e ) = 1 0 0 \left( Q _ { \mathrm { g e n } } ( e ) F _ { \mathrm { c o n d } } ( e ) \right) ^ { 1 / 2 } .\tag{71}
$$

All generation-quality metrics and $P _ { 5 }$ are expressed as fractions between 0 and 1. The normalization scales the positive likelihood preference relative to its maximum across monitored epochs. The score therefore rewards checkpoints that maintain molecular quality while producing stronger POI-associated chemical enrichment.

Table 10 reports the five highest-scoring checkpoints. Epoch 58 achieves the highest score (86.82), with 98.78% validity, 71.04% uniqueness, 71.77% novelty, and 0.796 internal diversity. It also achieves the highest monitored likelihood preference (+0.4155) and a generated 5-NN purity of 89.72%.

Epoch 46 achieves higher generation quality but lower conditional fidelity. Epoch 60 achieves slightly higher validity but lower 5-NN purity. Epoch 58 therefore provides the best balance between generation quality and POI-specific chemical steering under the defined score. We select checkpoint epoch 58 for the final conditional generation experiments.

![](images/d89cf50a49b8a704f3d482f2a7fed8a32893bfa1511e74c9d6fde1903bdb33d6.jpg)  
Figure 7 Generation quality during molecular pretraining, unconditional fine-tuning, and conditional fine-tuning. Panels (a,b) show the molecular prior, (c,d) the unconditional model, and (e,f) the conditional model. $\mathbf { ( a , c , e ) }$ Validity, uniqueness among valid molecules, and novelty among unique valid molecules across training epochs. (b,d,f) Internal diversity among unique valid molecules. For the conditional model, solid lines represent the mean across the seen POIs, and shaded regions indicate ±1 standard deviation across targets.

Unconditional model. The unconditional model does not use protein conditions, so conditional-fidelity metrics are not applicable. We instead prioritize molecular validity while maintaining uniqueness, novelty, and internal diversity.

Epochs 51 and 58 achieve the highest observed validity (99.80%). Compared with epoch 51, epoch 58 achieves higher uniqueness (96.79% versus 95.99%) and internal diversity (0.790 versus 0.785), while maintaining comparable novelty (66.46% versus 67.43%). We therefore select epoch 58 for unconditional generation. Table 11 reports the corresponding metrics.

![](images/ba42f32e8dd96f275529765029f439929a0af58a3c7f879115aeeb8e93921982.jpg)

![](images/554efe4733359f05d339a47f8af44658b87fb1ac32cddb5b023dddc659358064.jpg)  
Figure 8 Emergence of POI-specific conditioning during fine-tuning. (a) $\Delta _ { \mathrm { N L L } }$ and (b) 5-NN purity across training epochs. Solid lines show the mean across the seen POIs, and shaded regions indicate ±1 standard deviation. Likelihood evaluation uses a fixed set of validation molecules, while nearest-neighbor purity uses a fixed validation reference pool. At each monitored epoch, 500 molecules are generated per POI, with up to 25 unique valid molecules per target selected for 5-NN evaluation.

Table 10 Conditional checkpoint selection during fine-tuning. The five highest-scoring checkpoints are reported. Generation quality combines validity, uniqueness, novelty, and internal diversity. Conditional fidelity combines generated 5-NN purity and normalized $\Delta _ { \mathrm { N L L } }$ . The final score balances both components. Bold values indicate the highest values among the reported checkpoints.
<table><tr><td>Epoch</td><td>Validity (%)</td><td>Uniq. (%)</td><td>Novelty (%)</td><td>Diversity</td><td> $\overline { { \Delta _ { \mathrm { N L L } } } }$ </td><td>5-NN (%)</td><td> $\overline { { Q _ { \mathrm { g e n } } } }$ </td><td> $\overline { { F _ { \mathrm { c o n d } } } }$ </td><td>Score (%)</td></tr><tr><td>46</td><td>98.40</td><td>72.90</td><td>74.13</td><td>0.795</td><td>0.4058</td><td>87.56</td><td>0.8063</td><td>0.9247</td><td>86.34</td></tr><tr><td>60</td><td>98.84</td><td>71.30</td><td>72.33</td><td>0.796</td><td>0.4154</td><td>87.12</td><td>0.7981</td><td>0.9332</td><td>86.30</td></tr><tr><td>52</td><td>98.63</td><td>72.03</td><td>72.85</td><td>0.795</td><td>0.4106</td><td>87.44</td><td>0.8009</td><td>0.9295</td><td>86.28</td></tr><tr><td>56</td><td>98.66</td><td>71.34</td><td>71.38</td><td>0.796</td><td>0.4144</td><td>86.76</td><td>0.7952</td><td>0.9302</td><td>86.00</td></tr><tr><td>58</td><td>98.78</td><td>71.04</td><td>71.77</td><td>0.796</td><td>0.4155</td><td>89.72</td><td>0.7957</td><td>0.9472</td><td>86.82</td></tr></table>

Table 11 Unconditional checkpoint selection during fine-tuning. The two checkpoints with the highest observed validity are compared. Metrics are computed from 500 generated molecules per epoch. Bold values indicate the highest values in each column.
<table><tr><td>Epoch</td><td>Validity (%)</td><td>Uniq. (%)</td><td>Novelty (%)</td><td>Diversity</td></tr><tr><td>51</td><td>99.80</td><td>95.99</td><td>67.43</td><td>0.785</td></tr><tr><td>58</td><td>99.80</td><td>96.79</td><td>66.46</td><td>0.790</td></tr></table>