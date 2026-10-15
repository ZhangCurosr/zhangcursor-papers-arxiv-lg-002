# Synthetic reaction data from transformation rules for graph-to-sequence pretraining in biodegradation pathway prediction

Irsyaad Rijwan<sup>1\*</sup>, Liam Brydon<sup>1,2</sup>, Aidan Jared<sup>1</sup>, Jörg Simon Wicker<sup>1,2</sup>

<sup>1</sup>School of Computer Science, University of Auckland, Auckland, New Zealand. <sup>2</sup>enviPath Limited, Auckland, New Zealand.

\*Corresponding author(s). E-mail(s): irij533@aucklanduni.ac.nz;

## Abstract

The scarcity of observed biodegradation reactions constrains the training of product and pathway predictors. We present enviGraph, a framework that generates and curates synthetic reaction data, uses these data to pretrain a graph-to-sequence predictor, and fine-tunes the predictor on observed environmental reactions for product and recursive pathway prediction. Applying inherited environmental transformation rules to structures from PubChem, NORMAN SusDat and EPA DSSTox, followed by curation, produces 776,851 canonical parent–product pairs for pretraining.

We compare this dataset with three United States Patent and Trademark Ofice (USPTO) reaction datasets: USPTO-50K, USPTO-MIT and USPTO-Stereo. We evaluate enviGraph and the sequencebased enviFormer using ten folds split by pathway on the EAWAG Biocatalysis/Biodegradation Database (BBD), regulatory pesticide biodegradation data (Soil) and activated-sludge biodegradation data (Sludge). For both predictors, synthetic pretraining gives the highest BBD mean precision–recall area under the curve (PR-AUC) among the evaluated pretraining datasets for single-generation and multi-generation prediction. enviGraph achieves BBD single-generation PR-AUC of 0.202±0.050 with synthetic pretraining, compared with 0.169 ± 0.065 for the retrospectively selected strongest USPTO condition, USPTO-Stereo (mean ± sample standard deviation across folds). Its exact BBD top-1 and top-5 accuracies with synthetic pretraining are 0.355 and 0.594. The strongest USPTO conditions give higher PR-AUC means than synthetic pretraining in most Soil and Sludge comparisons. enviGraph has higher Soil multi-generation PR-AUC means than enviFormer under each of the four main pretraining datasets, and with synthetic pretraining it also leads enviFormer on BBD single-generation PR-AUC and exact top-1 and top-5 accuracy.

Scientific Contribution. We introduce enviGraph, a framework that generates and curates rule-derived synthetic reaction data for graph-to-sequence pretraining, followed by fine-tuning for biodegradation product and pathway prediction. The resulting curated corpus contains 776,851 canonical depth-1 parent–product pairs. Evaluation against USPTO pretraining with two predictors across BBD, Soil and Sludge, in both single-generation and multi-generation tasks, shows that the usefulness of the synthetic data varies across environmental datasets.

Keywords: biodegradation; synthetic reaction data; graph-to-sequence learning; pretraining; transformation products; pathway prediction

## 1 Introduction

Organic contaminants can form transformation products with persistence, mobility and biological efects that difer from those of their parent compounds. The disappearance of a parent compound therefore provides an incomplete account of its environmental fate. Identifying transformation products experimentally is dificult when reference standards are unavailable or analytical methods do not capture short-lived intermediates. Compu tational prediction can help prioritise candidate structures and pathways for investigation, provided that predicted structures are distinguished from experimentally confirmed products [1–3].

Environmental reaction prediction faces a shortage of observed training examples. The enviPath resource provides compounds, reactions, pathways and transformation rules for microbial biodegradation, including the records inherited from the EAWAG Biocatalysis/Biodegradation Database (BBD), as well as Soil and Sludge collections [4, 5]. These resources describe diferent environmental settings and experimental records. They are substantially smaller than the reaction datasets extracted from United States Patent and Trademark Ofice (USPTO) records [6, 7] that are commonly used to train neural networks for reaction prediction.

Transformation rules encode reusable local changes that can be applied to many structures. Rule-based systems identify applicable transfor mations and construct candidate products; their coverage depends on the available rules and their specificity. Recursive application can produce a large number of candidate pathways. Earlier work combined transformation rules with learned prioritisation to control this expansion [8, 9]. More recently, enviRule automated the extraction and generalisation of reaction patterns and evaluated them through downstream product prediction [10]. A rule-generated product remains a hypothesis about a reaction’s environmental occurrence. Using these products as training targets requires explicit choices about input structures, representation, acceptance, duplicate handling and evaluation-pair exclusion.

Neural networks can learn to generate reaction products from molecular inputs. Transformer models formulate product prediction as translation between molecular strings [11]. Pretrained initialisation improves generalisation over random initialisation in deep networks [12, 13]. In reaction prediction, a USPTO-pretrained Molecular Transformer fine-tuned on about 20,000 carbohydrate reactions reached 70.3% top-1 accuracy, compared with 30.4% for a model trained only on those reactions [14]. Pretrained Chemformer models likewise outperformed randomly initialised counterparts on reaction and retrosynthesis prediction [15].

In the environmental setting, enviFormer pretrains on USPTO reactions before adaptation to biodegradation data and evaluates both immediate products and recursively generated pathways [16]. Its authors concluded that the model’s ability to predict products depends on transfer learning. This transfer strategy addresses the limited size of environmental datasets, but the USPTO examples describe intended chemical syntheses rather than transformation under environmental conditions. Whether rule-derived environmental examples provide a more useful pretraining dataset is an empirical question, and the answer may difer across target collections.

Recent approaches difer in their pretrain ing data, prediction inputs and output tasks. ReactionT5 combines compound pretraining with reaction pretraining on Open Reaction Database records, then evaluates product prediction, retrosynthesis and yield prediction, including limiteddata settings [17]. Enzyformer combines molecular and reaction pretraining for enzymatic retrosyn thesis [18]. Template-driven generation of training reactions has precedents in synthesis prediction. RSGPT, a transformer for retrosynthesis, was pretrained on more than 10 billion reactions generated by applying templates extracted from reported reactions to molecular fragments [19]. Grandjean et al. [20] applied retrosynthesis templates to USPTO molecules, validated 27.5 million fictive reactions and found that a retrosynthesis model trained on a template-equilibrated subset outperformed one trained on USPTO reactions only. These evaluations concern synthetic chemistry and enzymatic retrosynthesis. To our knowledge, no earlier study builds a pretraining corpus from environmental transformation rules, and our study tests such a corpus for environmental products and pathways. ECLIPSE predicts enzyme-class associations from compounds and evaluates their use as additional inputs to product prediction [21]. In enviGraph, transformation rules supply training pairs, while prediction requires only the parent structure.

Molecular representation is a second design choice. A molecular graph directly exposes atom connectivity to an encoder, whereas a SMILES (Simplified Molecular Input Line Entry System) input specifies a traversal of that graph. Graph2SMILES combines local directed message passing and global graph-aware attention with sequence generation, providing a graph-tosequence architecture for reaction prediction and retrosynthesis [22]. UGTformer also combines graph pretraining with metabolic prediction, but its outputs are substrate and site-of-metabolism labels rather than product structures [23]. We use an encoder derived from Graph2SMILES and an autoregressive product-SMILES decoder for environmental prediction. Comparing this predictor with enviFormer sets graph-to-sequence against sequence-to-sequence prediction on the same environmental data.

We introduce enviGraph through three connected contributions:

1. We connect the generation and curation of rulederived environmental reaction examples to graph-to-sequence pretraining, fine-tuning on observed reactions and biodegradation product and pathway prediction.

2. We construct and document a curated corpus of 776,851 canonical depth-1 parent–product pairs, including its seed sources, transformation-rule provenance and curation procedure.

3. We compare synthetic and USPTO pretraining for enviGraph and enviFormer across BBD, Soil and Sludge in single-generation and multigeneration tasks. We also compare the predictors under the same pretraining datasets. Synthetic pretraining gives the highest BBD mean precision–recall area under the curve (PR-AUC) for both predictors; the strongest USPTO conditions lead most Soil and Sludge comparisons.

Rule execution and neural optimisation remain separate stages within the framework. Multigeneration evaluation follows the procedure used for enviFormer and established work on missing intermediates and multi-step recovery [16, 24].

## 2 The enviGraph framework

Figure 1 connects the four stages of enviGraph: synthetic reaction generation and curation (a), graph-to-sequence pretraining (b), fine-tuning on observed environmental reactions (c), and prediction (d). Stage (a) exports parent–product pairs for pretraining. We write each training reaction as a reaction SMILES string, <sub>reactant>>product</sub>. In USPTO reactions, the reactant side can list several reactants and reagents. In the generated pairs and in the BBD, Soil and Sludge pathways, it is a single parent compound. During both training stages, the reactant supplies the molecular graph and the product supplies shifted decoder inputs and nexttoken targets. Fine-tuning adapts the pretrained weights to each environmental training partition.

At inference, the selected predictor receives only the parent structure and returns scored product candidates. Newly predicted products can become inputs for recursive pathway expansion. Rule identity and environmental scenario are retained in construction records where available, but are not conditioning inputs to the predictor.

## 2.1 Transformation rules and seed structures

We obtained transformation rules from the EAWAG-BBD package in enviPath [4]. The retained rule set contained 498 top-level rule objects and 578 extracted transformation entries; exact-string deduplication produced 289 executable SMIRKS entries. These counts are not interchangeable because composite and simple rules can expose the same transformation. We compiled the cached strings using RDKit’s <sub>ReactionFromSmarts</sub> [25]. Compilation establishes executability under RDKit’s reaction-SMARTS (SMILES arbitrary target specification) semantics, which difer from the original Daylight SMIRKS specification [26].

Seed structures came from PubChem, NOR-MAN SusDat and the United States Environmental Protection Agency (EPA) Distributed Structure-Searchable Toxicity (DSSTox) database [27–29]. Cleaning selected the largest carboncontaining fragment, applied an organic-element filter and canonicalised SMILES, without a separate charge-neutralisation or tautomer-canonicalisation step. Merging used stored or recomputed Standard

![](images/79f5b836272683d7fe08c000bd6a17a47c3368d1d13b1095fd76c267760e8b16.jpg)  
Fig. 1 The integrated enviGraph framework.

(a) Seed structures and inherited transformation rules produce depth-1 synthetic reactions. Prescreening measures valid-product rule coverage as a diagnostic. Curation standardises and filters pairs, excludes evaluation-pair matches and caps frequent rules. Additional file 1 describes the smaller PS seed mixture and its corpora. (b) A parent–product pair supplies a molecular graph and a tokenised product. A directed graph attention network (D-GAT), followed by global attention using shortest-path distances in bonds, produces atom representations for

decoder cross-attention. Shifted product tokens supply decoder inputs and next-token targets, with   
beginning-of-sequence (BOS) and end-of-sequence (EOS) markers. Cross-entropy trains the graph encoder and   
decoder jointly. The molecular sketch is schematic. (c) Pretrained weights initialise models that are fine-tuned on   
the training portion of each pathway-aware fold; internal validation selects the checkpoint. (d) Held-out parents   
are decoded into candidate products. Recursive expansion applies the selected predictor to newly encountered   
products and is evaluated using multi-generation PR-AUC. Solid arrows show computational flow; dashed arrows   
show parameter transfer. Seed inventory details, curation stages, search settings and metric definitions are given in the text. The full-size vector figure accompanies the manuscript for zoomed viewing.

InChIKeys (hashed International Chemical Identifiers), with canonical SMILES as a fallback [30]. Source identifiers and aggregate provenance were retained.

The principal seed mixture, PSD, combines 14,047 cleaned source records from PubChem keyword searches for pharmaceutical, industrialchemical and pollutant terms, 96,070 from NOR-MAN SusDat, 1,056,537 from DSSTox and 2,029 from a PubChem keyword search for pesticides. Its 1,168,683 cleaned source records yield 1,090,572 unique seeds. Additional file 1 describes PS, a smaller mixture without DSSTox and the pesticide search.

## 2.2 Generating and curating synthetic reactions

Generation applies each compiled transformation rule to each cleaned seed molecule unless a cached negative reactant filter matches, pro ducing candidate parent–product records. Before generation, reaction-mode prescreening measures valid-product rule coverage as a diagnostic; it does not restrict which rules generation applies.

Each generated product is sanitised with RDKit and reduced to its largest carboncontaining fragment. Generation rejects sanitisation failures, excluded elements, negative productfilter matches, identity products and products with fewer than two heavy atoms. Duplicate canonical products are removed within each parent–rule application. When a rule returns several product molecules, each surviving molecule is recorded separately. A parent–product pair therefore need not describe a complete stoichiometric reaction.

The evaluated synthetic corpora use depth 1: rules act on seed molecules without recursively expanding their products. Generation records preserve the parent, product, rule, SMIRKS, generation depth and available origin metadata. Exported model input files contain canonical <sub>parent>>product</sub> pairs shared by the graph and sequence predictors. The generated products are weak training labels, without experimental confirmation of their environmental occurrence, rate or yield.

Curation begins by clearing isotope labels, removing redundant explicit hydrogens, and writing non-isomeric canonical SMILES. This policy also removes stereochemical distinctions. Dropping isotope-labelled rows instead would have removed 55 of the 134 represented SMIRKS entries while discarding only about 0.14% of rows. Failed and identity pairs are discarded, and stable pair deduplication retains one rule provenance per example. We then remove pairs losing more than 45 heavy atoms and pairs containing query-bond or dummy-atom tokens. The heavy-atom loss threshold follows the observed 99th percentile of loss magnitude in the PSD construction audit. Gains remain unbounded because a corresponding gain threshold would exclude conjugation families.

We exclude synthetic pairs matching evaluation reactions from BBD, Soil or Sludge. The overlap key selects the largest-heavy-atom reactant and product after fragment selection and canonicalises their isotope-normalised, non-isomeric pair. This exclusion removes direct matches under that reduced representation; it does not separate substrates or scafolds and does not remove the transformations that the BBD reactions share with the EAWAG-BBD rules.

Rule-frequency capping follows evaluation-pair exclusion, so removing matched pairs cannot leave a capped rule below its cap. Among the 1,935,577 normalised unique pairs, the five, ten and twenty most frequent rules accounted for 56.14%, 74.25% and 91.95% of pairs. To limit this concentration, PSD-Curated retains at most 36,526 pairs per SMIRKS, the rounded-up 90th percentile of per-SMIRKS counts among these pairs. Stable sorting by rule, canonical structures and identifiers makes retention deterministic. Capping reduces frequentrule exposure without oversampling rare rules. The thresholds were selected from corpus diagnostics, and the experiments compare the complete corpora that curation produces.

## 2.3 Graph-to-sequence product prediction

The enviGraph predictor combines a Graph2SMILES-derived molecular encoder [22] with an autoregressive product-SMILES decoder using Transformer-XL-style relative self-attention [31]. We use the name enviGraph for both the framework and its trained predictor. We adapted these inherited mechanisms to the environmen tal corpus, training and pathway interfaces. The encoder first aggregates local atom and bond information, then uses global attention to relate atoms across the molecule. The decoder reads these atom representations while generating the product SMILES one token at a time.

Each molecular bond becomes two directed edges. Atom features describe element, degree, formal charge, hybridisation, valence, hydrogen count, chirality and aromaticity; bond features describe type, stereochemistry, conjugation and ring membership. The directed graph attention network (D-GAT) uses four shared attention and gated-update iterations to aggregate incoming bond messages while excluding the immediate reverse edge. A subsequent atom-level aggregation produces 256-dimensional states.

Six global attention layers then combine these atom states using bucketed shortest-path distances measured in bonds. Each layer has eight attention heads and feed-forward width 2,048. Distances zero through seven have separate buckets; eight through fourteen share one, and fifteen or more share another. Disconnected pairs have their own bucket.

Examples that fail graph parsing or SMILES tokenisation are omitted.

The six-layer decoder has dimension 256, eight heads and feed-forward width 2,048. Masked self-attention reads earlier product tokens, and cross-attention reads the contextual atom representations. Absolute sinusoidal input positions accompany relative self-attention with signed ofset $\mathrm { c l i p } ( j - i , - 4 , 4 )$ . Clipping groups distant ofsets; it does not restrict attention to four tokens. The decoder has no recurrence between sequence segments.

For reactant $x ,$ graph $G ( x )$ and product token sequence $y ,$ both pretraining and fine-tuning minimise

$$
\mathcal { L } ( \theta ) = - \sum _ { ( x , y ) \in \mathcal { D } } \sum _ { t = 1 } ^ { | y | }\tag{1}
$$

Teacher forcing supplies the preceding reference product tokens, and minibatch loss averages over non-padding target tokens, including the end-ofsequence token. At inference, the decoder instead uses its preceding predicted tokens. The reference configuration uses zero label smoothing.

## 3 Experimental design

## 3.1 Datasets and baseline

We evaluate two types of comparison: pretraining conditions within each predictor, and the graph and sequence predictors under the same pretraining dataset. The main pretraining datasets are PSD-Curated, USPTO-50K [32], USPTO-MIT [33] and USPTO-Stereo [34]. We also compare PSD-Curated with two corpora from the PS seed mixture (Section 4.5). The profiled PSD-Curated file contains 776,851 unique valid canonical reactions.

We compare enviGraph with enviFormer, the environmental sequence-to-sequence predictor introduced by Brydon et al. [16]. We pretrain both predictors using the same pretraining datasets. Within-predictor comparisons assess complete pretraining conditions; comparisons using the same pretraining dataset set graph-to-sequence against sequence-to-sequence prediction, each predictor in its reference configuration. The reference enviFormer configuration uses dimension 256, eight heads, four encoder and four decoder layers, and feed-forward width 2,048. Approximate reference parameter counts are 18.2 million for enviGraph and 11.7 million for enviFormer.

BBD, Soil and Sludge are evaluated as separate datasets. BBD contains curated biodegradation pathways; Soil originates from regulatory pesticide studies; Sludge records activated-sludge transformations [4, 35, 36]. Table 1 describes retained local package objects before fold construction. BBD is the one dataset from the same EAWAG-BBD pack age as the transformation rules used in synthetic generation.

The object counts of the local packages dif fer from the deduplicated reaction populations obtained after pathway preprocessing and from the subsets scored in multi-generation evaluation. We therefore use the package counts to describe the data resources and use the saved pathway-fold records to identify evaluation partitions.

## 3.2 Splits and training

Pretraining normally groups reactions by reactant into $8 0 / 1 0 / 1 0$ partitions. For downstream evaluation, we construct ten pathway-aware folds by shufling whole pathways with split seed 1. Prepro cessing bypasses pseudo-nodes, canonicalises node structures, extracts edge-derived pairs and omits carbon-dioxide-only products. Reaction strings are deduplicated separately within training and test partitions. For each fold, a training reaction is excluded if its reactant equals a test reactant or its entire product string equals a test product. This preserves pathway allocation while allowing related structures and some cross-side overlaps. Within the remaining training set, a reactant-grouped 90/10 split with seed 1 supplies internal validation; the training-framework seed is 2.

Each downstream fold starts from pretrained weights and updates all trainable parameters. Both predictors use Adam [37] and Transformerstyle warm-up [38]. We select checkpoints using whole-sequence validation accuracy under teacherforced argmax decoding. For fresh fine-tuning, enviGraph creates a new optimiser but ofsets its learning-rate schedule by the global step reached during pretraining. In the implementation evaluated here, enviFormer restarts its learning-rate schedule during fresh fine-tuning, whereas the published enviFormer study continued the pretraining schedule [16].

Batch size and learning-rate scale difered between pretraining runs but were the same for all fine-tuning runs, and dropout difered between predictors.

We conducted the experiments using NVIDIA RTX PRO 6000 and L4 GPUs.

## 3.3 Single-generation prediction and evaluation

We assess single-generation predictions in two ways. Precision–recall evaluation counts recovery when a candidate contains all components of a reference product. Exact top-k accuracy requires equality between the predicted and reference component sets. Both use candidates returned by beam decoding, scored by raw sequence log-probability,

$$
s _ { \theta } ( y \mid x ) = \sum _ { t } \log p _ { \theta } ( y _ { t } \mid y _ { < t } , x ) .\tag{2}
$$

Partial sequences are retained during beam search using length normalisation [39], with $\alpha ~ = ~ 0$ for enviGraph and −0.5 for enviFormer. Completed candidates are ranked and thresholded using their raw scores. Reference single-generation beam widths are five and eight, respectively. These scores depend on string length and are not calibrated occurrence probabilities.

Table 1 Object counts in the local environmental packages before fold construction.
<table><tr><td>Dataset</td><td>Compounds</td><td>Reactions</td><td>Rules</td><td>Pathways</td></tr><tr><td>BBD</td><td>1,399</td><td>1,480</td><td>290</td><td>219</td></tr><tr><td>Soil</td><td>2,609</td><td>2,447</td><td>80</td><td>317</td></tr><tr><td>Sludge</td><td>1,070</td><td>521</td><td>0</td><td>184</td></tr></table>

For precision–recall evaluation, we run inference for each reference reaction row, then pool candidates by canonical parent. The evaluator deduplicates product-component sets and retains the first stored score for duplicate candidates. It counts recovered reference rows and unique parent– candidate pairs separately. Let X be the distinct test parents, $R _ { j }$ the reference component set in row j for parent x, and $m _ { x }$ its number of reference rows. For unique predictions $\widehat { \mathcal { P } } _ { \tau } ( x )$ satisfying s<sub>θ</sub> $> \tau$ , we calculate

$$
\begin{array} { r l r } {  { H ( \tau ) = \sum _ { x \in \mathcal { X } } \sum _ { j = 1 } ^ { m _ { x } } } } \\ & { } & { \quad \mathbf { 1 } [ \exists \widehat { R } \in \widehat { \mathcal { P } } _ { \tau } ( x ) : R _ { j } \subseteq \widehat { R } ] , } \end{array}\tag{3}
$$

$$
\begin{array} { l } { \displaystyle { C ( \tau ) = \sum _ { x } | \widehat { \mathcal { P } } _ { \tau } ( x ) | , } } \\ { \displaystyle { P ( \tau ) = \frac { H ( \tau ) } { C ( \tau ) } , \qquad R ( \tau ) = \frac { H ( \tau ) } { \sum _ { x } m _ { x } } , } } \end{array}\tag{4}
$$

with $P = 0$ for an empty candidate set. Recall counts recovered reference rows, whereas the precision denominator counts unique parent–candidate pairs. One candidate can recover several reference rows, so this precision is a reference-recovery ratio and is not intrinsically bounded by one.

Exact top-k accuracy instead counts each distinct parent once: success requires equality between a reference component set and one of its first k non-empty score-ranked saved predictions. We report top-1 and top-5, including parents whose complete returned lists are empty or shorter than the requested rank in the denominator. Productcomponent order does not afect either metric, but precision–recall containment and top-k equality define diferent recovery criteria. The published enviFormer evaluation used containment for topk matching [16]; our exact, parent-level measure therefore should not be compared directly with its reported top-k values.

## 3.4 Multi-generation prediction and evaluation

We use the multi-generation biodegradation pathway evaluation employed by Brydon et al. [16] for enviFormer, which builds on the depth weighting and intermediate-product treatment introduced by Tam et al. [24]. Starting from an initial compound, the predictor generates products and uses newly encountered products as inputs for subsequent generations. This recursive expansion constructs a predicted pathway graph. Evaluation prunes the graph at a series of score thresholds and compares its retained compounds with the reference pathway using the depth-weighted recovery procedure below.

We evaluate reference pathways containing exactly one initial compound, defined as a depth-0 node. A predicted node first reached from parent u receives cumulative score $S ( v ) = S ( u ) + s _ { \theta } ( y _ { v } \mid u )$ with root score zero; components of the same string share its sequence score. Rediscovery adds an edge without replacing the first stored score.

Reference search settings request five beams per molecule and retain at most 50 product strings per layer by local sequence score. The nominal depth argument is seven, but a stored reference pathway depth can override it. If D is the argument after that override, generation runs through $D - 1$ Expansion also stops at an empty frontier or after a completed layer exceeds 1,000 nodes.

At each threshold, pruning removes cycles by deleting edges, then removes nodes below threshold and non-root nodes disconnected from the root. Following Tam et al. [24], compound recovery is weighted by $2 ^ { - d } .$ , where d denotes depth. Intermediates are internal nodes on a shortest predicted path between recovered reference neighbours; identified intermediates absent from the reference receive zero weight. Other predicted nodes use depths adjusted by subtracting identified intermediates along a selected shortest root path.

True-positive and false-negative weights use reference depths; false-positive weights use adjusted predicted depths. Roots are excluded. This measures weighted compound recovery without requiring every edge to match.

We calculate precision and recall for each pathway at each threshold, then average each metric across the evaluated pathways within a fold. Varying the threshold produces one precision– recall curve per fold, whose area gives its multigeneration PR-AUC. Pathway averaging therefore occurs within each fold before the fold-level areas are summarised.

For both single-generation and multigeneration evaluation, the threshold grid spans −20.0 to −2.1 by 0.1 and $- 2 . 0 0 \ \mathrm { ~ t o ~ } - 0 . 0 1$ by 0.01. PR-AUC uses trapezoidal integration after stable ascending-recall ordering, without added endpoints or a precision envelope. For both tasks, reported values are the mean and sample standard deviation of ten fold-specific PR-AUC values. Shared training data make folds dependent; these dispersions describe partition variation, not independent optimisation repetitions.

Fine-tuned performance is the main analysis. Comparisons between pretraining datasets use the highest eligible mean among USPTO-50K, USPTO-MIT and USPTO-Stereo, selected retrospectively for each predictor, evaluation dataset and prediction task. Summaries of the highestscoring predictor also select pretraining datasets retrospectively; comparisons using the same pretraining dataset fix it for both predictors. Because every model is pretrained before fine-tuning, the comparisons concern the choice of pretraining dataset rather than whether to pretrain.

## 4 Results

## 4.1 Construction and curation of the synthetic corpus

The generation component of enviGraph applied environmental transformation rules to xenobiotic structure inventories to produce pretraining pairs for the graph and sequence predictors (Fig. 1). The PSD seed mixture contained 1,090,572 unique seed structures after merging 1,168,683 cleaned source records. Generation produced 2,003,522 depth-1 rows. The complete curation sequence retained 776,851 depth-1 parent–product pairs, or 38.8% of the generated rows, spanning 120 represented SMIRKS entries (Table 2). Representation processing and pair deduplication removed 67,945 rows, the heavy-atom loss filter 12,521, the query-bond and dummy-atom filter 96,438 and evaluation-pair exclusion 426. The rule cap removed a further 1,049,341 rows, and the share assigned to the five most frequent rules decreased from 56.14% to 23.51%.

The corpus profile identified 337,512 distinct reactants, 550,977 distinct products and 81,283 non-empty product scafolds in PSD-Curated. Its export contained no unparsable rows and no query-bond or dummy-atom tokens, and no additional matches were found under the recorded exact-pair overlap check. The PSD-Uncurated file contained 1,934,327 unique valid reactions after profiling; it was characterised but was not used in a predictive-performance comparison. Curation reduced molecular and scafold counts. The top-ten scafold share among ring-bearing unique products increased from 22.12% to 29.78% for PSD, showing that reduced rule concentration did not imply increased scafold diversity. The two profiles describe diferent-sized outputs rather than size-controlled diversity gains.

## 4.2 Comparison of synthetic and USPTO pretraining

Table 3 compares pretraining conditions within each predictor for single-generation and multigeneration PR-AUC on BBD, Soil and Sludge. Results are means and sample standard deviations across ten pathway-aware folds, with one execution per fold. Among the four main pretraining datasets, PSD-Curated gave both predictors their highest BBD means in both tasks.

On BBD, enviGraph pretrained on PSD-Curated achieved single-generation PR-AUC of $0 . 2 0 2 \pm \ : 0 . 0 5 0 .$ , compared with $0 . 1 6 9 ~ \pm ~ 0 . 0 6 5$ for $\mathrm { U S P T O - }$ Stereo, its strongest USPTO condition for this task. Its multi-generation PR-AUC increased from $0 . 0 7 8 \pm 0 . 0 2 8$ with USPTO-Stereo to $0 . 1 1 0 \pm 0 . 0 2 9$ with PSD-Curated. The corresponding synthetic-minus-USPTO diferences were 0.033 and 0.032 (Fig. 2). enviFormer also had higher BBD means with synthetic pretraining: $0 . 1 7 5 \pm 0 . 0 5 4$ for single-generation prediction and $0 . 1 1 2 \pm 0 . 0 3 7$ for multi-generation prediction, exceeding the USPTO-MIT and USPTO-Stereo conditions by 0.006 and 0.020, respectively. The BBD diferences were positive for both predictors, with larger diferences for enviGraph.

Table 2 Construction and curation of the PSD synthetic corpus.
<table><tr><td>PSD construction population</td><td>Count</td></tr><tr><td>Cleaned source records</td><td>1,168,683</td></tr><tr><td>Unique seed structures</td><td>1,090,572</td></tr><tr><td>Represented SMIRKS entries, before / after curation</td><td>134 / 120</td></tr><tr><td>Top-five rule share, before / after curation</td><td>56.14% / 23.51%</td></tr></table>

<table><tr><td>Curation stage</td><td>Rows</td><td></td><td>Removed Retained (%)</td></tr><tr><td>Generated depth-1 rows</td><td>2,003,522</td><td></td><td>100.0</td></tr><tr><td>Representation processing and pair deduplication</td><td>1,935,577</td><td>67,945</td><td>96.6</td></tr><tr><td>Heavy-atom loss filter (loss above 45)</td><td>1,923,056</td><td>12,521</td><td>96.0</td></tr><tr><td>Query-bond and dummy-atom filter</td><td>1,826,618</td><td>96,438</td><td>91.2</td></tr><tr><td>Evaluation-pair exclusion</td><td>1,826,192</td><td>426</td><td>91.1</td></tr><tr><td>Rule cap (36,526 pairs per SMIRKS)</td><td>776,851</td><td>1,049,341</td><td>38.8</td></tr></table>

Each stage removes the listed rows from the output of the previous stage; retention is relative to the generated depth-1 rows. From the second stage onward, each row is a normalised unique parent–product pair. Rule shares use separate rankings before and after the complete curation sequence. File profiles of the curated and uncurated corpora are given in Additional file 1: Table S1.

PSD-Curated did not consistently outperform USPTO pretraining on Soil and Sludge. For enviGraph on Soil, PSD-Curated was slightly above USPTO-Stereo in single-generation prediction, 0.154 versus 0.152, but below it in multi-generation prediction, 0.119 versus 0.148. enviFormer achieved higher Soil PR-AUC with USPTO-Stereo in both tasks. On Sludge, the highest USPTO means exceeded PSD-Curated for both predictors and tasks: USPTO-MIT gave the highest single-generation means and USPTO-Stereo the highest multi-generation means. The complete values and fold dispersions are reported in Table 3.

The USPTO condition in each pretraining comparison was selected retrospectively from USPTO-50K, USPTO-MIT and USPTO-Stereo. Paired tests between pretraining datasets gave adjusted q-values of 0.0496 and 0.0139 for the two positive enviGraph BBD diferences; the corresponding enviFormer values were 0.6853 and 0.3690. Benjamini–Hochberg adjustment was applied to a broader set of tests between pretraining datasets, of which these comparisons are a subset. Because pretraining-dataset selection was post hoc and the cross-validation training sets overlap, these tests are exploratory.

Multi-generation PR-AUC was lower than single-generation PR-AUC in all 24 combinations of predictor, main pretraining dataset and evaluation dataset. The closest means were obtained by enviGraph/USPTO-Stereo on Soil, at 0.148 for multi-generation and 0.152 for single-generation prediction.

## 4.3 Comparison of the graph and sequence predictors

We compared the predictors within each pretraining dataset, evaluation dataset and prediction task. Across all four main pretraining datasets, envi-Graph had the higher Soil multi-generation mean, whereas enviFormer had the higher Sludge singlegeneration mean. BBD single-generation rankings changed with the pretraining dataset: enviGraph led under PSD-Curated and USPTO-Stereo, while enviFormer led under USPTO-50K and USPTO-MIT. With PSD-Curated, enviGraph exceeded enviFormer by 0.027 on BBD single-generation PR-AUC, 0.023 on Soil single-generation PR-AUC and 0.031 on Soil multi-generation PR-AUC, using unrounded means. enviFormer was higher on both Sludge tasks and by 0.002 on BBD multi-generation PR-AUC. Across the four main pretraining datasets, enviGraph led six of the twelve multi-generation comparisons, against three of the twelve single-generation comparisons.

Table 3 Fine-tuned single-generation and multi-generation PR-AUC.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">BBD</td><td colspan="2">Soil</td><td colspan="2">Sludge</td></tr><tr><td>Single PR-AUC ↑</td><td>Multi PR-AUC ↑</td><td>Single PR-AUC ↑</td><td>Multi PR-AUC ↑</td><td>Single PR-AUC ↑</td><td>Multi PR-AUC ↑</td></tr><tr><td>enviGraph</td><td>PSD-Curated</td><td>0.202 ± 0.050</td><td>0.110 ± 0.029</td><td>0.154 ± 0.037</td><td>0.119 ± 0.035</td><td>0.017 ± 0.017</td><td>0.010 ± 0.004</td></tr><tr><td>enviFormer</td><td>PSD-Curated</td><td>0.175 ± 0.054</td><td>0.112 ± 0.037</td><td>0.130 ± 0.028</td><td>0.088 ± 0.027</td><td>0.044 ± 0.045</td><td>0.018 ± 0.021</td></tr><tr><td>enviGraph</td><td>USPTO-50K</td><td>0.107 ± 0.027</td><td>0.052 ± 0.017</td><td>0.103 ± 0.024</td><td>0.098 ± 0.046</td><td>0.043 ± 0.033</td><td>0.019 ± 0.014</td></tr><tr><td>enviFormer</td><td>USPTO-50K</td><td>0.111 ± 0.034</td><td>0.051 ± 0.024</td><td>0.106 ± 0.019</td><td>0.071 ± 0.022</td><td>0.060 ± 0.060</td><td>0.014 ± 0.010</td></tr><tr><td>enviGraph</td><td>USPTO-MIT</td><td>0.148 ± 0.048</td><td>0.068 ± 0.025</td><td>0.149 ± 0.038</td><td>0.126 ± 0.057</td><td>0.112 ± 0.049</td><td>0.037 ± 0.015</td></tr><tr><td>enviFormer</td><td>USPTO-MIT</td><td>0.169 ± 0.043</td><td>0.089 ± 0.035</td><td>0.175 ± 0.032</td><td>0.116 ± 0.029</td><td>0.154 ± 0.061</td><td>0.038 ± 0.017</td></tr><tr><td>enviGraph</td><td>USPTO-Stereo</td><td>0.169 ± 0.065</td><td>0.078 ± 0.028</td><td>0.152 ± 0.032</td><td>0.148 ± 0.053</td><td>0.061 ± 0.037</td><td>0.040 ± 0.027</td></tr><tr><td>enviFormer</td><td>USPTO-Stereo</td><td>0.163 ± 0.052</td><td>0.092 ± 0.047</td><td>0.182 ± 0.039</td><td>0.130 ± 0.038</td><td>0.137 ± 0.070</td><td>0.044 ± 0.022</td></tr></table>

Means and sample standard deviations over ten pathway-aware folds. Single and Multi denote single-generation PR-AUC and multi-generation PR-AUC. Bold identifies the largest unrounded mean across all eight rows in each column; emphasis is descriptive. Results for the PS corpora are in Additional file 1: Table S2.

![](images/fd4644a0a7990b8d29b1ab1234426de43b1bd2d1ad28b9962d2f0a70ee196e9f.jpg)  
Diference in mean PR-AUC, PSD-Curated minus the strongest USPTO condition  
Fig. 2 PR-AUC diferences between synthetic and USPTO pretraining.

Positive diferences favour PSD-Curated. Each USPTO value is the highest mean among USPTO-50K,   
USPTO-MIT and USPTO-Stereo for that predictor, evaluation dataset and prediction task, selected retrospectively.   
Points are mean diferences over the ten pathway-aware folds; error bars show 1 sample standard deviation of the ten paired fold diferences. The folds share training data, so the error bars are not confidence intervals.

## 4.4 Exact top-k accuracy

Exact top-k accuracy provides a stricter productmatching complement to PR-AUC (Table 4). On BBD, enviGraph with PSD-Curated achieved the highest top-1 accuracy, $0 . 3 5 5 \pm 0 . 0 5 2$ , and top-5 accuracy, 0.594 ± 0.077. Relative to enviFormer with the same pretraining dataset, the diferences were 0.060 and 0.064. The BBD advantage of enviGraph with PSD-Curated was therefore also observed when recovery required an exact match to a reference component set.

For Soil, enviFormer/USPTO-Stereo led top-1 and top-5 accuracy at 0.332 ± 0.051 and 0.610 ± 0.051. The highest Sludge accuracies came from different pretraining datasets: enviFormer/USPTO MIT reached top-1 accuracy of 0.345 ± 0.099, while enviFormer/USPTO-Stereo reached top-5 accuracy of 0.577 ± 0.128. All reported top-k conditions were evaluated over ten folds. These results measure exact-reference recovery within ranked candidate lists; they do not assess the chemical validity of additional, unrecorded products.

## 4.5 Comparison with a smaller seed mixture

We also pretrained both predictors on PS-Curated and PS-Uncurated, two corpora generated from the smaller PS seed mixture of 103,079 unique seeds (Additional file 1: Tables S2 and S3). Across singlegeneration PR-AUC, multi-generation PR-AUC and exact top-1 and top-5 accuracy, PSD-Curated gave the highest mean of the three synthetic corpora in 20 of the 24 combinations of predictor, evaluation dataset and metric. It led all twelve enviFormer combinations and the eight enviGraph combinations on BBD and Soil. The four excep tions were enviGraph on Sludge, where both PS corpora gave higher means for every metric. Seed mixture, corpus size, rule cap and some fine-tuning settings changed together between PS and PSD, so the comparison supports PSD-Curated, built from the larger seed mixture, as the principal synthetic corpus.

## 5 Discussion

## 5.1 Rule-derived training data and corpus curation

Earlier biodegradation systems applied rules directly to substrates and ranked the resulting products [8, 40, 41]. In enviGraph, rule execution supplies synthetic reaction examples for pretraining; fine-tuning then adapts the predictor to observed environmental reactions. The contribution is the construction and evaluation of these training examples within a framework for biodegradation product and pathway prediction. The transformation rules remain those inherited from the EAWAG-BBD package in enviPath [4].

Curation filtered and rebalanced the generated pairs before pretraining. Evaluation-pair exclusion removed direct matches to BBD, Soil and Sludge reactions under the reduced overlap key. The structural filters removed extreme fragmentation and query-bond or dummy-atom tokens, and rule capping limited exposure to a few frequent transformations (Table 2). Reduced rule concentration did not imply greater scafold diversity, since the top-ten scafold share increased during curation. In the PS mixture, the curated corpus gave lower PR-AUC means than its uncurated counterpart in eleven of the twelve combinations of predictor, evaluation dataset and prediction task, under training conditions that were not fully matched (Additional file 1: Table S2). A natural next experiment separates the efect of curation from that of corpus size.

The generated pairs expand the substrate contexts available for pretraining, while remaining limited to transformations in the rule library. They do not confirm environmental occurrence, specify reaction conditions or retain every reaction participant. The predictor receives only the parent structure. Its subsequent recovery of recorded products and pathways provides the test of whether these generated examples are useful for the evaluation datasets.

## 5.2 Synthetic pretraining across BBD, Soil and Sludge

PSD-Curated gave the highest BBD PR-AUC means for both predictors in both prediction tasks (Table 3). The shared direction across the graph and sequence predictors supports the usefulness of PSD-Curated for BBD, although the diferences from USPTO were smaller for enviFormer. The shared origin of the generation rules and the BBD reactions ofers a plausible explanation: both come from the EAWAG-BBD package, which supplied the transformations from which the cached rules were obtained. Brydon et al. [16] likewise found expert rules to be a strong BBD baseline. Evaluation-pair exclusion removes direct matches under the overlap key, but related scafolds, motifs and transformation types can remain. The BBD result therefore supports transfer to reactions from the database that supplied the rules; it does not show the same advantage on Soil or Sludge, which come from other enviPath packages.

Table 4 Fine-tuned exact top-1 and top-5 accuracy.
<table><tr><td></td><td></td><td colspan="2">BBD</td><td colspan="2">Soil</td><td colspan="2">Sludge</td></tr><tr><td>Predictor</td><td>Pretraining</td><td>Top-1 ↑</td><td>Top-5 ↑</td><td>Top-1 ↑</td><td>Top-5 ↑</td><td>Top-1 ↑</td><td>Top-5 ↑</td></tr><tr><td>enviGraph</td><td>PSD-Curated</td><td>0.355 ± 0.052</td><td>0.594 ± 0.077</td><td>0.316 ± 0.046</td><td>0.563 ± 0.070</td><td>0.063 ± 0.035</td><td>0.263 ± 0.083</td></tr><tr><td>enviFormer</td><td>PSD-Curated</td><td>0.295 ± 0.057</td><td>0.530 ± 0.056</td><td>0.268 ± 0.050</td><td>0.525 ± 0.063</td><td>0.108 ± 0.087</td><td>0.344 ± 0.109</td></tr><tr><td>enviGraph</td><td>USPTO-50K</td><td>0.246 ± 0.032</td><td>0.461 ± 0.053</td><td>0.248 ± 0.052</td><td>0.509 ± 0.045</td><td>0.125 ± 0.055</td><td>0.379 ± 0.139</td></tr><tr><td>enviFormer</td><td>USPTO-50K</td><td>0.229 ± 0.047</td><td>0.396 ± 0.069</td><td>0.243 ± 0.042</td><td>0.475 ± 0.058</td><td>0.161 ± 0.099</td><td>0.351 ± 0.150</td></tr><tr><td>enviGraph</td><td>USPTO-MIT</td><td>0.282 ± 0.063</td><td>0.482 ± 0.059</td><td>0.308 ± 0.052</td><td>0.580 ± 0.070</td><td>0.252 ± 0.052</td><td>0.492 ± 0.114</td></tr><tr><td>enviFormer</td><td>USPTO-MIT</td><td>0.307 ± 0.058</td><td>0.530 ± 0.055</td><td>0.323 ± 0.049</td><td>0.600 ± 0.041</td><td>0.345 ± 0.099</td><td>0.530 ± 0.109</td></tr><tr><td>enviGraph</td><td>USPTO-Stereo</td><td>0.298 ± 0.072</td><td>0.535 ± 0.069</td><td>0.300 ± 0.044</td><td>0.602 ± 0.061</td><td>0.119 ± 0.071</td><td>0.503 ± 0.170</td></tr><tr><td>enviFormer</td><td>USPTO-Stereo</td><td>0.286 ± 0.050</td><td>0.520 ± 0.069</td><td>0.332 ± 0.051</td><td>0.610 ± 0.051</td><td>0.288 ± 0.103</td><td>0.577 ± 0.128</td></tr></table>

Means and sample standard deviations over ten pathway-aware folds. Accuracy is the fraction of parents for which one of the first one or five non-empty score-ranked predictions exactly matches a reference component set. Bold identifies the largest unrounded mean across all eight rows in each column; emphasis is descriptive. Results for the PS corpora are in Additional file 1: Table S3.

The BBD pretraining advantage did not extend consistently to Soil or Sludge (Fig. 2), so the pretraining condition with the highest mean depended on the evaluation dataset and prediction task. Chemical coverage, dataset size and training conditions difer between the pretraining datasets, and varying them one at a time would show which of them accounts for the Soil and Sludge results. Lower performance than USPTO pretraining is also not evidence that synthetic pretraining harms performance relative to random initialisation. The strongest USPTO conditions were selected retrospectively from the same cross-validation results; these comparisons remain exploratory.

## 5.3 Predictor comparison and multi-generation prediction

The Soil multi-generation results distinguish predictor performance from pretraining-dataset performance. enviGraph had higher means than envi-Former under all four main pretraining datasets, but its highest Soil multi-generation mean came from USPTO-Stereo: 0.148, compared with 0.119 for PSD-Curated. On BBD with PSD-Curated, enviGraph led enviFormer on single-generation PR-AUC, 0.202 against 0.175, and on exact top-1 and top-5 accuracy.

Multi-generation prediction depends on successive product predictions. Missing an intermediate can prevent recovery of its descendants, while retaining an incorrect intermediate can create additional branches. Single-generation and multigeneration PR-AUC use diferent matching and aggregation rules, so the lower multi-generation values (Table 3) are not a direct measure of error accumulation. Recall at each depth and the num ber of predicted branches per layer would show where multi-generation recovery is lost and help explain enviGraph’s Soil multi-generation ranking. Reference pathways can also omit products and intermediates, so multi-generation PR-AUC measures recovery of recorded compounds under this evaluation procedure. Unmatched predictions are candidate products rather than confirmed errors, and every predicted product remains a hypothesis about environmental formation.

## 6 Conclusions

enviGraph integrates the construction of synthetic reaction data with graph-to-sequence pretraining and fine-tuning for biodegradation product and pathway prediction. Its principal curated corpus contains 776,851 depth-1 parent–product pairs generated from inherited environmental transformation rules and xenobiotic structures. Curation excluded pairs matching evaluation reactions and reduced the share of the five most frequent rules from 56.14% to 23.51%. This corpus gave higher means than both corpora from a smaller seed mix ture in 20 of the 24 combinations of predictor, evaluation dataset and metric.

Among the evaluated conditions, PSD-Curated gave both predictors their highest BBD PR-AUC means in single-generation and multi-generation prediction. enviGraph pretrained on PSD-Curated also achieved the highest exact BBD top-1 and top-5 accuracy. The strongest USPTO conditions gave higher PR-AUC means than PSD-Curated in most Soil and Sludge comparisons. enviGraph had higher Soil multi-generation means than envi-Former under all four main pretraining datasets, with its best mean obtained using USPTO-Stereo. The strongest evidence for the synthetic corpus is therefore on BBD, where it also gave enviGraph its lead over enviFormer in single-generation PR-AUC and exact top-k accuracy. The graph-to-sequence predictor’s standing improved on the recursive task: it led six of the twelve multi-generation comparisons, against three of the twelve single-generation comparisons.

## Abbreviations

<table><tr><td>BBD</td><td>EAWAG Biocatalysis/Biodegradation</td></tr><tr><td>BOS D-GAT DSSTox</td><td>Database Beginning of sequence Directed graph attention network Distributed Structure-Searchable</td></tr><tr><td>EOS EPA</td><td>Toxicity End of sequence United States Environmental</td></tr><tr><td>GPU InChIKey</td><td>Protection Agency Graphics processing unit Hashed International Chemical</td></tr><tr><td>PR-AUC PS</td><td>Identifier Precision-recall area under the curve PubChem and NORMAN SusDat</td></tr><tr><td>PSD</td><td>seed mixture PubChem, NORMAN SusDat and</td></tr><tr><td>SMARTS</td><td>DSSTox seed mixture SMILES arbitrary target specification</td></tr><tr><td>SMILES</td><td>Simplified Molecular Input Line Entry System</td></tr><tr><td>SMIRKS USPTO</td><td>Daylight reaction transform language United States Patent and Trademark Office</td></tr></table>

## Declarations

## Availability of data and materials

The study uses the public resources described and cited in Sections 2 and 3. The seed structures used to generate the synthetic corpora, the trained checkpoints and the evaluation outputs will be made publicly available. The Python source code of enviGraph will be released under the GNU Afero General Public License v3.0 (AGPL-3.0).

## Competing interests

JSW is one of the founders of enviPath Limited, a scientific software development company that develops and maintains enviPath. LB is a Machine Learning Engineer at enviPath Limited. The authors declare no other competing interests.

## Funding

Not applicable.

## Authors’ contributions

IR designed and implemented enviGraph, conducted the experiments and analysis, and drafted the manuscript. LB provided code and guidance for running the enviFormer experiments and evaluation, and reviewed and proofread the manuscript. AJ reviewed and proofread the manuscript. JSW supervised the research, provided expertise, guidance and support throughout, and reviewed and proofread the manuscript. All authors read and approved the final manuscript.

## Acknowledgements

Not applicable.

## References

[1] Escher BI, Fenner K (2011) Recent advances in environmental risk assessment of transformation products. Environ Sci Technol 45(9):3835–3847. https://doi.org/10.1021/ es1030799

[2] Zahn D, Arp HPH, Fenner K, et al (2024) Should transformation products change the way we manage chemicals? Environ Sci Technol 58(18):7710–7718. https://doi.org/10. 1021/acs.est.4c00125

[3] Schymanski EL, Jeon J, Gulde R, et al (2014) Identifying small molecules via high resolution mass spectrometry: Communicating confidence. Environ Sci Technol 48(4):2097–2098. https://doi.org/10.1021/es5002105

[4] Wicker J, Lorsbach T, Gütlein M, et al (2016) enviPath – The environmental contaminant biotransformation pathway resource. Nucleic Acids Res 44(D1):D502–D508. https://doi. org/10.1093/nar/gkv1229

[5] Hafner J, Lorsbach T, Schmidt S, et al (2024) Advancements in biotransformation pathway prediction: Enhancements, datasets, and novel functionalities in enviPath. J Cheminform 16(1):93. https://doi.org/10.1186/ s13321-024-00881-6

[6] Lowe DM (2012) Extraction of chemical structures and reactions from the literature. PhD

thesis, University of Cambridge, https://doi. org/10.17863/CAM.16293

[7] Lowe DM (2017) Chemical reactions from US patents (1976-Sep2016). Dataset, figshare, https : / / doi . org / 10 . 6084 / m9 . figshare . 5104873.v1

[8] Fenner K, Gao J, Kramer S, et al (2008) Datadriven extraction of relative reasoning rules to limit combinatorial explosion in biodegradation pathway prediction. Bioinformatics 24(18):2079–2085. https://doi.org/10.1093/ bioinformatics/btn378

[9] Wicker J, Fenner K, Ellis LBM, et al (2010) Predicting biodegradation products and pathways: A hybrid knowledge- and machine learning-based approach. Bioinformatics 26(6):814–821. https://doi.org/10. 1093/bioinformatics/btq024

[10] Zhang K, Fenner K (2023) enviRule: An endto-end system for automatic extraction of reaction patterns from environmental contaminant biotransformation pathways. Bioinformatics 39(7):btad407. https://doi.org/10. 1093/bioinformatics/btad407

[11] Schwaller P, Laino T, Gaudin T, et al (2019) Molecular Transformer: A model for uncertainty-calibrated chemical reaction prediction. ACS Cent Sci 5(9):1572–1583. https: //doi.org/10.1021/acscentsci.9b00576

[12] Erhan D, Bengio Y, Courville A, et al (2010) Why does unsupervised pre-training help deep learning? J Mach Learn Res 11(19):625–660. https://www.jmlr.org/papers/v11/erhan10a. html

[13] Yosinski J, Clune J, Bengio Y, et al (2014) How transferable are features in deep neural networks? In: Advances in Neural Information Processing Systems 27

[14] Pesciullesi G, Schwaller P, Laino T, et al (2020) Transfer learning enables the Molecular Transformer to predict regio- and stereoselective reactions on carbohydrates. Nat Commun 11(1):4874. https://doi.org/10.1038/ s41467-020-18671-7

[15] Irwin R, Dimitriadis S, He J, et al (2022) Chemformer: A pre-trained transformer for computational chemistry. Mach Learn Sci Technol 3(1):015022. https://doi.org/10.1088/ 2632-2153/ac3fb

[16] Brydon L, Zhang K, Dobbie G, et al (2025) Predictive modeling of biodegradation pathways using transformer architectures. J Cheminform 17(1):21. https://doi.org/10.1186/ s13321-025-00969-7

[17] Sagawa T, Kojima R (2025) ReactionT5: A pre-trained transformer model for accurate chemical reaction prediction with limited data. J Cheminform 17(1):126. https://doi.org/10. 1186/s13321-025-01075-4

[18] Liu T, Xu J, Zhan X, et al (2026) Enzyformer: A two-stage pretrained model for enzymatic retrosynthesis. J Cheminform 18(1):35. https: //doi.org/10.1186/s13321-026-01164-y

[19] Deng Y, Zhao X, Sun H, et al (2025) RSGPT: A generative transformer model for retrosynthesis planning pre-trained on ten billion datapoints. Nat Commun 16(1):7012. https: //doi.org/10.1038/s41467-025-62308-6

[20] Grandjean Y, Kreutter D, Reymond JL (2026) Data augmentation in a triple transformer loop retrosynthesis model. Digital Discovery 5(2):653–661. https://doi.org/10.1039/ d5dd00465a

[21] Brydon-Brown L, Dobbie G, Taškova K, et al (2026) Predicting enzyme-compound associations for enzyme-catalysed reactions. J Cheminform 18(1):77. https://doi.org/10. 1186/s13321-026-01190-w

[22] Tu Z, Coley CW (2022) Permutation invariant graph-to-sequence model for template-free retrosynthesis and reaction prediction. J Chem Inf Model 62(15):3503–3513. https://doi.org/ 10.1021/acs.jcim.2c00321

[23] Zhan M, Li X, Xiong L, et al (2026) UGTformer: A pretrained graph transformer model for predicting UDP-glucuronosyltransferasemediated drug metabolism. J Cheminform 18(1):33. https : / / doi . org / 10 . 1186 /

s13321-026-01163-z

[24] Tam JYC, Lorsbach T, Schmidt S, et al (2021) Holistic evaluation of biodegradation pathway prediction: Assessing multi-step reactions and intermediate products. J Cheminform 13(1):63. https://doi.org/10.1186/ s13321-021-00543-x

[25] Landrum GA, et al (2026) RDKit: Opensource cheminformatics software. https : / / www.rdkit.org, accessed 11 August 2026

[26] Daylight Chemical Information Systems (2011) SMIRKS—a reaction transform language. https://www.daylight.com/dayhtml/ doc/theory/theory.smirks.html, accessed 11 August 2026

[27] Kim S, Chen J, Cheng T, et al (2023) PubChem 2023 update. Nucleic Acids Res 51(D1):D1373–D1380. https://doi.org/10. 1093/nar/gkac956

[28] Mohammed Taha H, Aalizadeh R, Alygizakis N, et al (2022) The NORMAN Suspect List Exchange (NORMAN-SLE): Facilitating European and worldwide collaboration on suspect screening in high resolution mass spectrometry. Environ Sci Eur 34(1):104. https: //doi.org/10.1186/s12302-022-00680-6

[29] Williams AJ, Grulke CM, Edwards J, et al (2017) The CompTox Chemistry Dashboard: A community data resource for environmental chemistry. J Cheminform 9(1):61. https://doi. org/10.1186/s13321-017-0247-6

[30] Heller SR, McNaught A, Pletnev I, et al (2015) InChI, the IUPAC International Chemical Identifier. J Cheminform 7(1):23. https: //doi.org/10.1186/s13321-015-0068-4

[31] Dai Z, Yang Z, Yang Y, et al (2019) Transformer-XL: Attentive language models beyond a fixed-length context. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp 2978–2988, https://doi.org/10.18653/v1/P19-1285

[32] Schneider N, Stiefl N, Landrum GA (2016) What’s what: The (nearly) definitive guide to

reaction role assignment. J Chem Inf Model 56(12):2336–2346. https://doi.org/10.1021/ acs.jcim.6b00564

[33] Jin W, Coley CW, Barzilay R, et al (2017) Predicting organic reaction outcomes with Weisfeiler-Lehman network. In: Advances in Neural Information Processing Systems 30

[34] Schwaller P, Gaudin T, Lányi D, et al (2018) “Found in Translation”: Predicting outcomes of complex organic chemistry reactions using neural sequence-to-sequence models. Chem Sci 9(28):6091–6098. https://doi.org/10.1039/ C8SC02339E

[35] Latino DARS, Wicker J, Gütlein M, et al (2017) Eawag-Soil in enviPath: A new resource for exploring regulatory pesticide soil biodegradation pathways and half-life data. Environ Sci Process Impacts 19(3):449–464. https://doi.org/10.1039/C6EM00697C

[36] Trostel L, Coll C, Fenner K, et al (2023) Combining predictive and analytical methods to elucidate pharmaceutical biotransformation in activated sludge. Environ Sci Process Impacts 25(8):1322–1336. https://doi.org/10. 1039/D3EM00161J

[37] Kingma DP, Ba J (2015) Adam: A method for stochastic optimization. In: 3rd International Conference on Learning Representations, https://doi.org/10.48550/arXiv.1412. 6980

[38] Vaswani A, Shazeer N, Parmar N, et al (2017) Attention is all you need. In: Advances in Neural Information Processing Systems 30

[39] Wu Y, Schuster M, Chen Z, et al (2016) Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144, https://doi. org/10.48550/arXiv.1609.08144

[40] Hou BK, Wackett LP, Ellis LBM (2003) Microbial pathway prediction: A functional group approach. J Chem Inf Comput Sci 43(3):1051–1057. https://doi.org/10.1021/ ci034018f

[41] Ellis LBM, Gao J, Fenner K, et al (2008) The University of Minnesota pathway prediction system: Predicting metabolic logic. Nucleic Acids Res 36(suppl\_2):W427–W432. https: //doi.org/10.1093/nar/gkn315

## Additional files

Additional file 1

File format: .pdf

Title: PS seed-mixture corpora and fine-tuned results

Description: Construction and curation of the corpora from the smaller PS seed mixture, with file profiles of the PS and PSD corpora (Section S1, Table S1); PR-AUC and exact top-1 and top-5 accuracy for PS-Curated, PS-Uncurated and PSD-Curated after fine-tuning (Section S2, Tables S2 and S3).