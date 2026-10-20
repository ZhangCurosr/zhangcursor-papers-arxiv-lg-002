# PRISM: Prior Relational Information for Self-supervised Modeling to Enhance Solubility OOD Generalization

Xinyi Chen<sup>1</sup> chenxinyi@sjtu.edu.cn

Jiahuan Pang<sup>1</sup> pangjiahuan@sjtu.edu.cn

Titus Chima<sup>1</sup> titusschima@sjtu.edu.cn

Paul Weng<sup>3,4,\*</sup> paul.weng@dukekunshan.edu.cn

Wendong Wang<sup>1,2,\*</sup> wendong.wang@sjtu.edu.cn

## Abstract

Solubility depends on intermolecular interactions between solutes and solvents, and is central to drug discovery, chemical and materials process engineering. Its prediction requires out-of-distribution (OOD) generalization to unseen solutes, solvents, and solvent mixtures, where supervised methods suffer from data scarcity, and existing self-supervised methods do not explicitly learn chemically meaningful relations. Learning chemically meaningful relational information without labels can alleviate data scarcity and improve OOD generalization, but remains challenging. We introduce Prior Relational Information for Self-supervised Modeling (PRISM) and apply it to solubility OOD generalization. PRISM is a label-free pretraining framework that converts expert-defined chemical priors into relational teachers. For solubility prediction, it trains four encoders to preserve the population-level relational information from priors capturing hydrophobic, electrostatic, polarizability, and substructural effects, using Kullback-Leibler (KL) divergence to match distance-based neighborhood distributions during pretraining. Across scaffold, solute, and solvent OOD settings, PRISM matches or surpasses supervised and self-supervised baselines. Under mixed-solvent OOD, freezing the four PRISM encoders and training a lightweight fusion-and-composition head outperforms strong baselines and substantially reduces run-to-run variation. Ablations and control studies confirm that gains stem from chemically meaningful relational structures baked into the encoders by PRISM. We expect this strategy of turning relational priors into self-supervision signals to extend beyond solubility prediction to domains where chemically or biologically meaningful relational priors are available.

## 1 Introduction

Solubility measures how much of a solute dissolves in a specified pure or mixed solvent under defined temperature and pressure, and it is a critical property across drug discovery, formulation, and pharmaceutical process development [1]. In practical applications, models often encounter unseen scaffolds, unseen solutes and solvents, and unseen binary solvent pairs in mixed-solvent systems, requiring solubility prediction beyond the training distribution. These shifts directly affect the solute–solvent and solvent–solvent interactions governing solubility [2, 3], making them natural out-of-distribution (OOD) scenarios for evaluating chemically meaningful molecular representations.

Recent solubility prediction methods model solvent context through recommendation, interactiongraph, hypergraph, or multicomponent architectures [4–7]. Broader molecular relational learning methods explicitly model drug–drug or solute–solvent interactions [8, 9, 2], but usually require supervised pairs or task-specific data. Molecular self-supervised objectives based on augmentationdefined pairs, masking, or reconstruction [10–14] do not specify scientifically grounded populationlevel neighborhoods: which molecules should be close or far under a chemically meaningful relation. Classical manifold and molecular metric-learning methods can preserve predefined relations, but leave open how to transfer scientifically grounded population-level neighborhood geometry into molecular encoders without labels [15, 16]. These studies leave a research gap: label-free pretraining remains weakly aligned with the relational structure needed for solubility OOD generalization.

For solubility, a chemically grounded route can close this gap: by the empirical “like dissolves like” rule, solute–solvent compatibility is reflected by descriptor- or fingerprint-based physicochemical similarity [17–19]. We propose PRISM (Prior Relational Information for Self-supervised Modeling), which turns such similarity into a label-free relational teacher. PRISM uses these prior-defined relations to shape the geometry of learned molecular embeddings.

PRISM has two design choices: the prior source, which specifies the expert-defined relation, and the alignment design, which transfers that relation into embedding space. We use van der Waals surface area (VSA) descriptors [19] and Morgan fingerprints [18] as complementary physicochemical and substructural priors, and adopt distance-based distribution matching with prior-specific encoders fused only downstream (late fusion) as the alignment design among several candidates.

Our main results are OOD generalization of solubility prediction. PRISM outperforms supervised and SSL-pretrained baselines in cross-dataset generalization from BigSolDB to Leeds and SolProp for predicting the solubility of diverse solutes in pure solvents. With frozen prior-specific encoders, trainable fusion, and a composition-weighted head, it matches or improves end-to-end mixed-solvent baselines under solvent-pair-disjoint splits with better stability. Label energy (a form of graph Dirichlet energy) diagnostics and ablation controls further show that these gains come from prior–task compatibility, relational pretraining, and late fusion of prior-specific embeddings, rather than from descriptor inputs, reconstruction, architectural capacity, or coordinate-level imitation.

## Our contributions are the following:

1. We propose PRISM, a label-free molecular pretraining framework that bakes chemically meaningful prior knowledge into molecular representations through prior-induced relational modeling, providing a practical route to label-free relational representation learning.

2. We build a PRISM-based solubility prediction model using VSA descriptors and Morgan fingerprints as complementary physicochemical and substructural priors. The model includes two stages: the pretraining of four prior-specific encoders and the weighted-sum late fusion of prior-specific embeddings during downstream tasks.

3. We show that PRISM improves solubility-centered OOD generalization from scaffold splits to pure- and mixed-solvent settings, matching or surpassing strong baselines and improving stability under mixed-solvent OOD. Ablations and label-energy diagnostics support relational pretraining, prior complementarity, and prior–task compatibility.

## 2 Methodology

PRISM treats each chemical prior as a relational teacher, defining molecular closeness under a specific physicochemical criterion. We adopt D-MPNN as a strong 2D graph backbone [20]; the framework itself remains encoder-agnostic. We first discuss which prior relations to choose in Section 2.1, and then describe how they are aligned in the embedding space and fused in Section 2.2 and 2.3.

## 2.1 Solubility-grounded prior spaces

The chemistry of solubility guides which prior relations PRISM uses. Solubility depends on solute– solvent interactions and molecular structure. Following the empirical “like dissolves like” principle, solubility is favored when solute and solvent share similar interaction components, including hydrophobic, electrostatic, and dispersion-related effects, or similar substructures capturing interactions not accounted for by those effects. PRISM therefore uses four complementary priors: SlogP\_VSA for hydrophobic/lipophilic effects, SMR\_VSA for polarizability/dispersion-relevant effects, PEOE\_VSA for electrostatic effects, and Morgan fingerprints for substructure and functional-group patterns. Detailed descriptions of these prior spaces and other prior spaces are provided in Appendix D.

## 2.2 Prior-induced relational alignment

Let $\mathcal { D } _ { \mathrm { p r e } } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ be an unlabeled set of molecular graphs and let $\boldsymbol { S } = \{ s _ { k } \} _ { k = 1 } ^ { K }$ be a set of prior spaces. For each prior space $s _ { k } \in S ,$ , a fixed prior function maps molecule $x _ { i } \mathrm { t o } \pi _ { i } ^ { s _ { k } }$ , and a prior-space comparator $d ^ { s _ { k } }$ measures pairwise relations. A prior-specific encoder $G _ { \theta _ { k } } ^ { s _ { k } }$ with parameters $\theta _ { k }$ maps the same molecule to an embedding $r _ { i } ^ { s _ { k } }$ . PRISM aligns the relation induced by $\{ r _ { i } ^ { s _ { k } } \} _ { i = 1 } ^ { N }$ with the relation induced by $\{ \pi _ { i } ^ { s _ { k } } \} _ { i = 1 } ^ { N }$ in a minibatch . The alignment procedure is illustrated in Figure 1a.

## a Prior-induced relational alignment

![](images/9886443963051531c416b992bee402f0f072fb5b42dd0b7be8461e4c5bc45d3a.jpg)

![](images/681cac20cf3ae736633337b003661b169bc7087c898a141832013aecb1d63c8a.jpg)  
Figure 1: PRISM overview. (a) Prior-guided alignment pretrains an encoder per prior space. (b) Weighted fusion produces a fused molecular embedding.

Concretely, two relational constructors $\mathcal { A } _ { p r i o r } ^ { s _ { k } }$ and $\boldsymbol { A } _ { e m b } ^ { s _ { k } }$ map pairwise comparisons in $\boldsymbol { B }$ to two relational objects $\mathcal { R } _ { \mathrm { p r i o r } } ^ { s _ { k } }$ and $\mathcal { R } _ { \mathrm { e m b } } ^ { s _ { k } }$

$$
\mathcal { R } _ { \mathrm { p r i o r } } ^ { s _ { k } } = \mathcal { A } _ { p r i o r } ^ { s _ { k } } \big ( \{ d ^ { s _ { k } } ( \pi _ { i } ^ { s _ { k } } , \pi _ { j } ^ { s _ { k } } ) \} _ { i , j \in \mathcal { B } } \big ) , \qquad \mathcal { R } _ { \mathrm { e m b } } ^ { s _ { k } } = \mathcal { A } _ { e m b } ^ { s _ { k } } \big ( \{ \delta ^ { s _ { k } } ( r _ { i } ^ { s _ { k } } , r _ { j } ^ { s _ { k } } ) \} _ { i , j \in \mathcal { B } } \big ) .\tag{1}
$$

The objective of PRISM is to align the two relational objects, $\mathcal { R } _ { \mathrm { p r i o r } } ^ { s _ { k } }$ and $\mathcal { R } _ { \mathrm { e m b } } ^ { s _ { k } }$ , using a suitable discrepancy metric as the loss function:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P R I S M } } ^ { s _ { k } } = \mathcal { M } ^ { s _ { k } } \Big ( \mathcal { R } _ { \mathrm { p r i o r } } ^ { s _ { k } } , \mathcal { R } _ { \mathrm { e m b } } ^ { s _ { k } } \Big ) . } \end{array}\tag{2}
$$

Table 1: Relational objects and alignment objectives in PRISM with related instances.
<table><tr><td>Relational object</td><td>Objective</td><td>Discrepancy metric</td><td>Related instances</td></tr><tr><td>Pairwise distance matrix</td><td>Pointwise metric matching</td><td>Distance regression</td><td>Talibart et al. [21]</td></tr><tr><td>Neighbor ordering</td><td>Rank preservation</td><td>Rank / ListMLE loss</td><td>PH-TDL [22]</td></tr><tr><td>Top-n prior-space kNN graph</td><td>Neighborhood-graph alignment</td><td>InfoNCE over graph positives</td><td>Moleco [23]</td></tr><tr><td>Pair-wise distance-based distribution</td><td>Distribution matching</td><td>KL divergence</td><td>SNE-style [24] / GraphMSL [25]</td></tr></table>

Table 1 classifies alignment objectives by the relational objects they preserve. As a concrete example, pairwise distances can be converted into anchor-wise stochastic neighborhood distributions. For each prior space $s _ { k } .$ , anchor molecule $x _ { i }$ , and candidate set $B _ { i } = B \setminus \{ x _ { i } \} , R _ { i } ^ { s _ { k } }$ denotes the prior-space probability distribution of $x _ { i } ,$ , while $P _ { i } ^ { s _ { k } }$ denotes the corresponding embedding-space probability distribution. The relational objects are sets of anchor-wise distributions

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { p r i o r } } ^ { s _ { k } } = \{ R _ { i } ^ { s _ { k } } \} _ { x _ { i } \in \mathcal { B } } , \qquad \mathcal { R } _ { \mathrm { e m b } } ^ { s _ { k } } = \{ P _ { i } ^ { s _ { k } } \} _ { x _ { i } \in \mathcal { B } } . } \end{array}
$$

The probabilities in these two distributions are defined by

$$
R _ { i } ^ { s _ { k } } ( x _ { j } ) = \frac { \exp \bigl ( - \beta _ { k } d ^ { s _ { k } } \bigl ( \pi _ { i } ^ { s _ { k } } , \pi _ { j } ^ { s _ { k } } \bigr ) \bigr ) } { \sum _ { x _ { l } \in \mathcal { B } _ { i } } \exp \bigl ( - \beta _ { k } d ^ { s _ { k } } \bigl ( \pi _ { i } ^ { s _ { k } } , \pi _ { l } ^ { s _ { k } } \bigr ) \bigr ) } , \qquad P _ { i } ^ { s _ { k } } ( x _ { j } ) = \frac { \exp \bigl ( - \beta _ { \tau } \delta \bigl ( r _ { i } ^ { s _ { k } } , r _ { j } ^ { s _ { k } } \bigr ) \bigr ) } { \sum _ { x _ { l } \in \mathcal { B } _ { i } } \exp \bigl ( - \beta _ { \tau } \delta \bigl ( r _ { i } ^ { s _ { k } } , r _ { l } ^ { s _ { k } } \bigr ) \bigr ) } .\tag{3}
$$

Here $d ^ { s _ { k } }$ is the prior-space distance, $\delta ( r _ { i } ^ { s _ { k } } , r _ { j } ^ { s _ { k } } ) = 1 - \cos ( r _ { i } ^ { s _ { k } } , r _ { j } ^ { s _ { k } } )$ , and $\beta _ { k } , \beta _ { \tau }$ control locality. The loss function aggregates anchor-wise KL divergence over the minibatch :

$$
\mathcal { L } _ { \mathrm { K L } } ^ { s _ { k } } = \frac { 1 } { | \mathcal { B } | } \sum _ { x _ { i } \in \mathcal { B } } \mathrm { K L } ( R _ { i } ^ { s _ { k } } \parallel P _ { i } ^ { s _ { k } } ) .\tag{4}
$$

Because Eq. 3 normalizes each anchor separately, these neighborhood distributions are conditional and generally asymmetric, retaining anchor-specific neighborhood geometry. Section 3.2 evaluates the four alignment objectives and explains why distribution matching is used in PRISM. Appendix F.2 provides additional theoretical analysis on these alignment objectives and compares the conditional formulation with a symmetric SNE-style alternative.

## 2.3 Weighted-sum fusion

The second stage of PRISM is weighted-sum fusion (Figure 1b). PRISM uses K prior-specific encoders, each producing $h _ { i } ^ { ( s _ { k } ) } = q ^ { s _ { k } } ( r _ { i } ^ { s _ { k } } ) \in \mathbb { R } ^ { D }$ , with branch embeddings stacked as $H _ { i } \ =$ $[ h _ { i } ^ { ( s _ { 1 } ) } ; \ldots ; h _ { i } ^ { ( s _ { K } ) } ] \in \mathbb { R } ^ { K \times D }$ . PRISM fuses these branch embeddings by weighted summation:

$$
z _ { i } = \sum _ { s _ { k } \in S } w _ { i } ^ { s _ { k } } h _ { i } ^ { s _ { k } } .\tag{5}
$$

We consider two fusion variants. The first uses task-specific weights, $w _ { i } ^ { s _ { k } } = w ^ { s _ { k } }$ , providing a tasklevel summary of prior importance. The second uses molecule-specific weights $w _ { i } = \operatorname { s o f t m a x } ( g ( H _ { i } ) )$ from a small gating network $^ { g , }$ allowing PRISM to emphasize different prior-induced relations across molecules as solubility-governing interactions vary. Implementation details and fusion analyses are provided in Appendix E.5 and Appendix I.1.

![](images/23137c6579dbc0f1dcc58f8223ff5fcfc6f76afc8eb85dd014cd0c1b47be3765.jpg)  
Figure 2: Task-specific prediction heads. (a) Single-molecule prediction. (b) Pure-solvent prediction combines solute, solvent, and temperature embeddings. (c) Mixed-solvent prediction freezes encoders, trains fusion, and compositionally pools solvent embeddings.

## 2.4 Task-specific prediction heads

Figure 2 summarizes the downstream task settings. For single-molecule tasks, the model predicts directly from the fused molecular embedding, ${ \hat { y } } _ { i } = f _ { \mathrm { s i n g l e } } ( z _ { i } )$ . For pure-solvent tasks, the head

combines solute embedding ${ \cal z } _ { \mathrm { s o l } }$ , solvent embedding ${ \mathit { z } } _ { \mathrm { s o l v } }$ , and temperature embedding $t = h _ { T } ( T )$ where $h _ { T }$ is an MLP, as $\hat { y } = f _ { \mathrm { p u r e } } ( [ z _ { \mathrm { s o l } } ; z _ { \mathrm { s o l v } } ; t ] )$ .

For mixed-solvent prediction, we freeze the original PRISM-pretrained encoders, initialize the fusion module from the pure-solvent task, and train it together with a lightweight mixed-solvent head $f _ { \mathrm { m i x } }$ Given two solvents $x _ { 1 } , x _ { 2 }$ and their mole fractions $\alpha _ { 1 } , \alpha _ { 2 } ,$ where $\alpha _ { 1 } + \alpha _ { 2 } = 1$ , we form

$$
z _ { \mathrm { m i x } } = \alpha _ { 1 } z _ { 1 } + \alpha _ { 2 } z _ { 2 } , \qquad \hat { y } = f _ { \mathrm { m i x } } \bigl ( [ z _ { \mathrm { s o l } } ; z _ { \mathrm { m i x } } ; t ] \bigr ) .\tag{6}
$$

Because a real solvent mixture may have nonlinear solvent–solvent interactions, the linear pooling restricts head capacity. This deliberate low-capacity design probes whether PRISM-pretrained solvent embeddings contain relational information about solvent–solvent interactions.

## 2.5 Label energy for prior–task compatibility

PRISM relies on prior spaces that induce task-relevant molecular neighborhoods. We therefore introduce label energy as a practical diagnostic of prior–task compatibility: if a prior is relevant to a downstream task, molecules with similar labels should be nearby in its prior space. Using n nearest neighbors for each molecule (we use $n = 1 0$ in all experiments), we define

$$
\mathcal { E } _ { n } ^ { s _ { k } } ( y ) = \frac { 1 } { \vert Q \vert } \sum _ { i \in Q } \frac { 1 } { n } \sum _ { j \in \mathcal { N } _ { n } ^ { s _ { k } } ( x _ { i } ) } ( y _ { i } - y _ { j } ) ^ { 2 } .\tag{7}
$$

Here $Q$ denotes the labeled molecules available for diagnosis, and $\mathcal { N } _ { n } ^ { s _ { k } } ( x _ { i } )$ is the set of n nearest neighbors of $x _ { i }$ in prior space $s _ { k }$ . Lower energy indicates molecules with smoother labels over the prior-induced neighborhood and therefore stronger prior–task compatibility.

Because the raw energy depends on label scale and class balance, we normalize it against a Gaussian random prior:

$$
\rho ^ { s _ { k } } = \frac { \mathcal { E } _ { n } ^ { s _ { k } } ( y ) } { \mathcal { E } _ { n } ^ { ( \mathrm { r a n d o m } ) } ( y ) } .\tag{8}
$$

Thus, $\rho ^ { s _ { k } } < 1$ indicates that prior $s _ { k }$ induces smoother label neighborhoods than random geometry. Appendix F.3 further shows that label energy bounds the error of a local predictor in the prior space, providing a theoretical justification for this diagnostic. When prior choice is less obvious, candidate priors should be scientifically relevant, inexpensive, and complementary, with label energy on training/validation labels used to filter task-incompatible choices (Appendix H.9).

## 3 Experiments

## 3.1 Experimental setup

In this work, we evaluate PRISM under four chemically significant OOD scenarios:

1. Structural OOD: MoleculeNet scaffold splits test unseen molecular scaffolds.

2. Solute OOD: pure-solvent transfer tests unseen solutes on Leeds and SolProp.

3. Solvent OOD: pure-solvent transfer evaluates the subset of SolProp whose solvents are absent from the processed BigSolDB reference set.

4. Solvent-pair OOD: mixed-solvent transfer tests unseen solvent pairs from reference [26].

We report RMSE for regression tasks. For pure and mixed solubility tasks, we also report $\begin{array} { r } { \mathcal { \gamma } _ { 0 } | \Delta \log ( S ) | \leq 1 = N ^ { - 1 } \sum _ { i } \mathbf { 1 } ( | \widehat { \log } ( \widehat { S _ { i } } ) - \log ( S _ { i } ) | \leq 1 ) \times 1 0 0 , } \end{array}$ , the fraction of predictions within one log-solubility unit, a practically meaningful error range [27] that complements RMSE. Split construction and leakage checks are detailed in Appendix C.6. Because the evaluated tasks require single-molecule, solute–solvent, and multi-molecule inputs, we use capability- and protocol-matched baselines for each setting; Appendix C.9 summarizes their selection and training protocols.

Table 2: Alignment objective selection on ESOL. C1–C3 denote fixed early, intermediate, and late checkpoints from one uninterrupted pretraining run for each objective.
<table><tr><td rowspan="2">Objective</td><td colspan="3">Alignment metric↓</td><td colspan="3">ESOL RMSE↓</td></tr><tr><td>C1</td><td>C2</td><td>C3</td><td>C1</td><td>C2</td><td>C3</td></tr><tr><td> $\ell _ { 2 }$  distance</td><td>0.968</td><td>0.933</td><td>0.851</td><td>0.956</td><td>0.865</td><td>0.987</td></tr><tr><td>Rank</td><td>0.180</td><td>0.109</td><td>0.025</td><td>1.023</td><td>1.027</td><td>1.022</td></tr><tr><td>Neighborhood-graph alignment</td><td>2.53</td><td>2.02</td><td>1.34</td><td>2.011</td><td>1.241</td><td>1.083</td></tr><tr><td>Distribution matching</td><td>0.813</td><td>0.736</td><td>0.723</td><td>0.844</td><td>0.776</td><td>0.772</td></tr></table>

## 3.2 Selecting the default alignment objective for PRISM

Table 1 defines four types of relational alignment objectives, and Table 2 compares one representative from each type on ESOL under the same SlogP\_VSA prior, D-MPNN backbone, pretraining corpus, and downstream protocol. C1–C3 trace how each objective evolves during pretraining; final checkpoints are selected using held-out unlabeled alignment loss.

The objectives exhibit different behaviors under this matched comparison. $\ell _ { 2 }$ is overly rigid: alignment improves across C1–C3, but downstream RMSE first decreases and then increases, suggesting that exact distance matching can suppress graph-derived information not represented by the prior. Rank is too coarse: it preserves relative ordering while discarding graded similarity strength [28], and its downstream RMSE changes little across checkpoints. Neighborhood-graph alignment and distribution matching both preserve neighborhood structure, while distribution matching additionally retains graded similarity rather than reducing relations to hard positive/negative assignments, and achieves the best late-checkpoint RMSE. We therefore select distribution matching as PRISM’s default alignment objective and keep this objective fixed in all subsequent experiments.

## 3.3 MoleculeNet structural OOD

Table 3: MoleculeNet structural OOD on solubility-related tasks under scaffold splits. RMSE ( ). Results averaged over three seeds with independent splits.
<table><tr><td>Model</td><td>ESOL↓</td><td> ${ \mathrm { F r e e S o l v } } \downarrow$ </td><td>Lipo ↓</td></tr><tr><td>XGBoost</td><td> $0 . 9 2 5 \pm 0 . 0 0$ </td><td> $3 . 3 9 8 \pm 0 . 0 0$ </td><td> $0 . 7 4 1 \pm 0 . 0 0$ </td></tr><tr><td>D-MPNN[20]</td><td> $1 . 0 5 0 \pm 0 . 0 1$ </td><td> ${ \bf 1 . 6 7 3 \pm 0 . 0 8 }$ </td><td> $0 . 6 8 3 \pm 0 . 0 2$ </td></tr><tr><td>MolCLR[10]</td><td> $1 . 1 1 3 \pm 0 . 0 2$ </td><td> $2 . 3 0 1 \pm 0 . 2 5$ </td><td> $0 . 7 8 9 \pm 0 . 0 1$ </td></tr><tr><td>MoAMa[29]</td><td> $1 . 3 9 4 \pm 0 . 1 2$ </td><td> $2 . 7 0 0 \pm 0 . 1 1$ </td><td> $0 . 7 7 2 \pm 0 . 0 3$ </td></tr><tr><td>PRISM</td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 2 }$ </td><td> $1 . 7 7 0 \pm 0 . 0 9$ </td><td> ${ \bf 0 . 5 3 8 \pm 0 . 0 2 }$ </td></tr></table>

In addition to $\mathrm { E S O L } ,$ we evaluate PRISM on FreeSolv and Lipo to assess structural OOD for additional solubility-related properties. PRISM achieves the best results on ESOL and Lipo. FreeSolv is a boundary case: its hydration free energy depends on 3D geometry and electrostatic structure that 2D encoders cannot fully capture. Replacing the 2D backbone with a 3D-aware encoder substantially improves performance on FreeSolv (Appendix H.10), supporting the value of 3D-aware representations for this task. The complete results on the remaining MoleculeNet benchmarks (BBBP, BACE, HIV) are in the Appendix G.1, suggesting that PRISM extends to broader properties.

## 3.4 Pure-solvent external transfer

Practical solubility prediction goes beyond single-molecule aqueous solubility and requires modeling solute–solvent relations across diverse solvents. Therefore, we test generalizations of solvent OOD and solute OOD using cross-dataset transfer: PRISM is pretrained on ZINC, fine-tuned on BigSolDB, and evaluated on Leeds and SolProp. The result is shown in Table 4.

PRISM achieves the best results on both external datasets, improving RMSE and $\% | \Delta$ log $S | \le 1$ over supervised and SSL-pretrained baselines. Its advantage holds over XGBoost with direct access to the same prior features, suggesting that the gain does not arise from the priors alone. It further outperforms Fastprop, a strong solubility model with thermodynamic constraints, without using the

Table 4: Pure-solvent external transfer from BigSolDB to Leeds and SolProp. Published baseline values are from prior reported protocols; XGBoost and PRISM are mean std over three seeds.
<table><tr><td rowspan="3">Model</td><td colspan="2">RMSE↓</td><td colspan="2"> $\% | \Delta$  log  $S | \leq 1 \uparrow$ </td></tr><tr><td>Leeds</td><td>SolProp</td><td>Leeds</td><td>SolProp</td></tr><tr><td>Vermeire [27]</td><td>2.16</td><td>1.43</td><td>41.2</td><td>66.9</td></tr><tr><td>MolFormer [27]</td><td>1.07</td><td>0.98</td><td>64.1</td><td>70.7</td></tr><tr><td>ChemBERTa [27]</td><td>1.26</td><td>1.01</td><td>57.0</td><td>72.4</td></tr><tr><td>Chemprop [27]</td><td>0.99</td><td>0.83</td><td>70.9</td><td>76.1</td></tr><tr><td>Fastprop[27]</td><td>0.95</td><td>0.83</td><td>73.6</td><td>78.1</td></tr><tr><td>XGBoost</td><td> $1 . 0 0 \pm 0 . 0 0 3$ </td><td> $0 . 7 8 \pm 0 . 0 0 8$ </td><td> $6 9 . 4 \pm 0 . 2$ </td><td> $7 8 . 3 \pm 1 . 2$ </td></tr><tr><td>PRISM</td><td> $\mathbf { 0 . 9 1 \pm 0 . 0 0 2 }$ </td><td> ${ \bf 0 . 7 7 \pm 0 . 0 3 0 }$ </td><td> ${ \bf 7 5 . 7 \pm 0 . 9 }$ </td><td> ${ \bf 8 2 . 3 \pm 0 . 8 }$ </td></tr></table>

same loss. These results are particularly notable given the pretraining scale: MolFormer uses 1.1B molecules and ChemBERTa 77M, whereas PRISM is pretrained on only 250K molecules. Together with the temperature-trend analysis in Appendix G.3, these results suggest that prior-induced relational pretraining extracts compact, transfer-relevant structure for solubility prediction.

## 3.5 Mixed-solvent OOD generalization and representation reuse

Table 5: Mixed-solvent solubility prediction under random and solvent-pair-disjoint splits. All results are mean std over five independently resampled splits under the corresponding protocol.
<table><tr><td></td><td colspan="3">Random split</td><td colspan="3">Solvent-pair-disjoint OOD</td></tr><tr><td>Model</td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td> $\% | \Delta \log S | \le 1 \uparrow$ </td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td> $\% | \Delta \log S | \le 1 \uparrow$ </td></tr><tr><td colspan="7">Direct-prior control</td></tr><tr><td>XGBoost</td><td> $0 . 0 5 5 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 1 0 0 \pm 0 . 0 }$ </td><td> $0 . 4 5 9 \pm 0 . 0 2 2$ </td><td> $0 . 7 9 6 \pm 0 . 0 8 6$ </td><td> $9 6 . 0 \pm 0 . 8$ </td></tr><tr><td colspan="7">End-to-end D-MPNN aggregation controls</td></tr><tr><td>D-MPNN + Concat</td><td> $0 . 1 0 3 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 0$ </td><td> $9 9 . 9 \pm 0 . 0$ </td><td> $0 . 5 8 7 \pm 0 . 0 9 3$ </td><td> $0 . 7 0 3 \pm 0 . 0 7 2$ </td><td> $9 1 . 6 \pm 3 . 8$ </td></tr><tr><td> $_ { \mathrm { D - M P N N } + \mathrm { W e i g h t e d ~ S u m } }$ </td><td> $0 . 1 0 1 \pm 0 . 0 0 5$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 1$ </td><td> ${ \bf 1 0 0 \pm 0 . 0 }$ </td><td> $0 . 4 9 8 \pm 0 . 1 6 1$ </td><td> $0 . 7 6 6 \pm 0 . 1 3 4$ </td><td> $9 3 . 8 \pm 4 . 5$ </td></tr><tr><td> $\mathrm { D } { \cdot } \mathrm { M P N N } + \mathrm { H y p e r G r a p h }$ </td><td> $0 . 0 8 9 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 6 \pm 0 . 0 0 3$ </td><td> ${ \bf 1 0 0 \pm 0 . 0 }$ </td><td> $0 . 4 5 7 \pm 0 . 2 1 1$ </td><td> $0 . 7 8 5 \pm 0 . 1 4 2$ </td><td> $9 5 . 2 \pm 6 . 0$ </td></tr><tr><td> $\mathbf { D - M P N N + C r o s s ~ A t t e n t i o n }$ </td><td> $0 . 1 1 9 \pm 0 . 0 0 5$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 1$ </td><td> ${ \bf 1 0 0 \pm 0 . 0 }$ </td><td> $0 . 4 8 3 \pm 0 . 1 8 6$ </td><td> $0 . 7 6 1 \pm 0 . 1 1 6$ </td><td> $9 4 . 2 \pm 6 . 8$ </td></tr><tr><td colspan="7">Specialized interaction models</td></tr><tr><td>Triple-CIGIN[30]</td><td> $\mathbf { 0 . 0 5 2 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 1 0 0 \pm 0 . 0 }$ </td><td> $0 . 6 7 4 \pm 0 . 1 2 3$ </td><td> $0 . 6 3 9 \pm 0 . 1 0 4$ </td><td> $8 8 . 1 \pm 5 . 0 $ </td></tr><tr><td>Triple-MMGNN[31]</td><td> $0 . 0 6 5 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 1 0 0 \pm 0 . 0 }$ </td><td> $0 . 6 3 9 \pm 0 . 1 9 0$ </td><td> $0 . 6 6 1 \pm 0 . 1 7 8$ </td><td> $8 9 . 6 \pm 5 . 4$ </td></tr><tr><td colspan="7">Frozen SSL + composition-weighted pooling</td></tr><tr><td>ChemBERTa[32]</td><td> $0 . 1 0 \overset { \cdot } { 9 } \pm 0 . \overset { \cdot } { 0 } 1 5$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 2$ </td><td> $9 9 . 9 \pm 0 . 1 $ </td><td> $0 . 5 0 3 \pm 0 . 1 0 6$ </td><td> $0 . 7 8 7 \pm 0 . 0 5 3$ </td><td> $9 3 . 0 \pm 4 . 9$ </td></tr><tr><td>MolCLR-GCN[10]</td><td> $0 . 1 3 5 \pm 0 . 0 2 0$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 4$ </td><td> $9 9 . 8 \pm 0 . 2 $ </td><td> $0 . 5 9 2 \pm 0 . 1 4 6$ </td><td> $0 . 7 0 9 \pm 0 . 1 4 0$ </td><td> $8 8 . 5 \pm 8 . 7 $ </td></tr><tr><td>GROVER[33]</td><td> $0 . 1 1 9 \pm 0 . 0 0 9$ </td><td>0.993 ± 0.001</td><td> $9 9 . 9 \pm 0 . 1 $ </td><td> $0 . 7 3 7 \pm 0 . 3 2 7$ </td><td> $0 . 5 2 3 \pm 0 . 3 0 5$ </td><td> $9 0 . 0 \pm 9 . 5$ </td></tr><tr><td>MolCLR-GIN[10]</td><td> $0 . 1 1 5 \pm 0 . 0 0 8$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 1$ </td><td> $9 9 . 9 \pm 0 . 0$ </td><td> $0 . 7 9 7 \pm 0 . 4 4 8$ </td><td> $0 . 4 7 8 \pm 0 . 3 7 0$ </td><td> $8 5 . 6 \pm 1 3 . 3$ </td></tr><tr><td>PRISM</td><td> $0 . 0 7 7 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 0 . 4 3 8 \pm 0 . 0 3 0 }$ </td><td> $\mathbf { 0 . 8 4 6 \pm 0 . 0 3 1 }$ </td><td> ${ \bf 9 6 . 6 \pm 0 . 9 }$ </td></tr></table>

Beyond pure-solvent transfer, practical solubility prediction often involves mixed solvents, where composition-dependent solvent–solvent interactions also matter. We evaluate random and solventpair-disjoint splits using frozen PRISM encoders, fusion initialized from the pure-solvent task, and the composition-weighted pooling head in Eq. (6) (Figure 2c), testing whether mixture-relevant information can be recovered through lightweight downstream adaptation.

We compare against direct-prior XGBoost, same-backbone D-MPNN controls, specialized interaction models, and frozen SSL baselines. Under the random split, most methods are near saturation, with differences largely within measurement uncertainty. Under solvent-pair-disjoint OOD, PRISM achieves the best mean RMSE, $R ^ { 2 } .$ , and $\begin{array} { r } { \% | \Delta \log \bar { S } | \ \leq \ 1 . } \end{array}$ , with substantially lower split-to-split variation than most baselines. The comparison with XGBoost shows that direct access to the priors alone is insufficient, while the D-MPNN and interaction-model controls show that model capacity or explicit pairwise interaction modeling does not recover the same transferability. Together with the frozen-SSL baselines, these results suggest that separating complementary prior-defined relations during pretraining yields representations that are more reusable under unseen solvent combinations. Additional ablations support solvent aggregation and fusion methods(Appendix H.7).

We further evaluate PRISM on the independent Cenci et al. dataset [34], covering pure, binary, and ternary solvent systems. PRISM improves overall held-out prediction over UNIFAC–PLS, while combining PRISM with UNIFAC gives the strongest overall and ternary results (Appendix G.4).

## 4 Ablation Study

This section examines the two representation-learning choices introduced in Section 1: how prior relations should be encoded and which priors are useful for a downstream task. We first use controlled variants to isolate the roles of relational pretraining and prior-specific late fusion, distinguishing them from direct prior input, reconstruction, increased model capacity, and alternative multi-prior fusion strategies. We then use label energy to diagnose prior–task alignment and characterize when a scientific prior is likely to provide useful supervision. Together, these analyses examine both the alignment design and the prior source underlying PRISM.

## 4.1 Solubility gains require PRISM pretraining

We first test whether PRISM’s solubility gains come from prior-induced relational pretraining rather than model capacity or direct descriptor use. Table 6 compares PRISM with five controls: D-MPNN w. prior directly concatenates prior descriptors; D-MPNN (reconstruction) uses the same encoder architecture but replaces PRISM’s relational alignment objective with pointwise descriptor reconstruction during pretraining; D-MPNN (all spaces) uses one encoder trained with all prioralignment losses; 4 D-MPNN (scratch) keeps the multi-branch architecture without pretraining; and PRISM-Distilled compresses PRISM’s four pretrained encoders into a single encoder by aligning it with the multi-prior representation of PRISM under the same KL objective (Appendix H.2).

Table 6: Effect of PRISM pretraining on solubility transfer. Controls use comparable downstream heads. Mixed OOD denotes the solvent-pair-disjoint split.
<table><tr><td></td><td colspan="2">Leeds</td><td colspan="2">SolProp</td><td colspan="2">Mixed OOD</td></tr><tr><td>Model</td><td>RMSE↓</td><td>%|∆ log S| ≤ 1 ↑</td><td>RMSE↓</td><td>%|∆ log S| ≤ 1 ↑</td><td>RMSE↓</td><td>%|∆ log S| ≤ 1 ↑</td></tr><tr><td>D-MPNN w. prior</td><td>0.990±0.013</td><td>72.3±1.2</td><td>0.820±0.035</td><td>80.2±0.9</td><td>0.466±0.078</td><td>95.3±1.7</td></tr><tr><td>D-MPNN (reconstruction)</td><td>1.076±0.033</td><td>62.1±2.1</td><td>0.956±0.082</td><td>69.0±4.7</td><td>0.618±0.133</td><td>87.3±2.1</td></tr><tr><td>D-MPNN (all spaces)</td><td>0.930±0.004</td><td>73.9±1.6</td><td>0.860±0.017</td><td>77.2±0.8</td><td>0.580±0.107</td><td>92.5±2.1</td></tr><tr><td>4× D-MPNN (scratch)</td><td>1.001±0.020</td><td>71.6±0.7</td><td>0.901±0.038</td><td>74.7±3.4</td><td>0.472±0.123</td><td>92.1±3.6</td></tr><tr><td>PRISM-Distilled</td><td>0.950±0.014</td><td>73.9±0.6</td><td>0.970±0.024</td><td>77.2±1.1</td><td>0.457±0.110</td><td>95.6±3.8</td></tr><tr><td>PRISM</td><td>0.910±0.002</td><td>75.7±0.9</td><td>0.770±0.030</td><td>82.3±0.8</td><td>0.440±0.010</td><td>96.6±0.9</td></tr></table>

These controls support two conclusions about why PRISM works.

First, the selected physicochemical priors are informative, but are most effective as relational supervision. D-MPNN w. prior is competitive, showing that the priors carry task-relevant signal; yet descriptor reconstruction is among the weakest variants, and 4 D-MPNN (scratch) underperforms despite matching PRISM’s branch capacity. Thus, PRISM’s gain is not explained by descriptor access, pointwise prior prediction, or architecture alone, but by transferring prior-induced relational information into the embedding space. This distinguishes the information contained in the prior coordinates from the relational structure they induce over molecules.

Second, multi-prior structure matters. D-MPNN (all spaces) and PRISM-Distilled are competitive in some settings, showing that prior-induced information can be partially recovered by fused or compressed designs, but neither matches full PRISM. D-MPNN (all spaces) corresponds to GraphMSL-style early fusion [25]: supervising one shared embedding distribution with all priors is equivalent to first fusing the prior-induced target distributions and then matching this fused target by KL (Appendix F.2). Its weaker performance shows that learning prior-specific geometries before fusion preserves more task-relevant information than fusing prior targets before learning. PRISM-Distilled probes the opposite direction: relational distillation compresses the multi-branch teacher into a single encoder, preserving useful signal on relation-centered solubility tasks but losing ground relative to the full model in Appendix H.2. Together, these results support PRISM’s late-fusion design: per-prior geometries are partly compromised by early fusion and only partly recovered by post-hoc compression, so PRISM keeps them separate during pretraining and combines them only downstream. The comparison therefore isolates not only whether multiple priors are useful, but also when their distinct relational geometries should be combined.

## 4.2 Label energy diagnoses prior–task alignment

We next test whether the selected priors are meaningful for solubility-related tasks. Table 7 shows that the default VSA and fingerprint priors induce smoother-than-random label neighborhoods on

Table 7: Prior–task alignment and Gaussian-prior control on solubility-related tasks. Lower $\rho _ { s }$ indicates smoother prior-induced label neighborhoods. All downstream columns report RMSE ( ).
<table><tr><td>Dataset</td><td> $\rho _ { \mathrm { V S A } }$ </td><td> $\rho _ { \mathrm { F P } }$ </td><td>Gaussian prior trained PRISM 4×D-MPNN scratch</td><td></td><td>D-MPNN</td><td>PRISM</td></tr><tr><td>ESOL</td><td>0.107</td><td>0.150</td><td> $1 . 2 4 \pm 0 . 0 2 5$ </td><td> $0 . 8 4 8 \pm 0 . 0 2 5$ </td><td> $1 . 0 5 0 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 2 }$ </td></tr><tr><td>FreeSolv</td><td>0.072</td><td>0.090</td><td> $2 . 7 9 \pm 0 . 0 5 8$ </td><td> $2 . 2 2 2 \pm 0 . 0 5 4$ </td><td> $\mathbf { 1 . 6 7 3 \pm 0 . 0 8 2 }$ </td><td> $1 . 7 7 0 \pm 0 . 0 9 3$ </td></tr><tr><td>Lipo</td><td>0.305</td><td>0.315</td><td> $1 . 0 7 \pm 0 . 0 1 4$ </td><td> $0 . 6 4 2 \pm 0 . 0 1 4$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 6$ </td><td> $\mathbf { 0 . 5 3 8 \pm 0 . 0 1 8 }$ </td></tr></table>

ESOL, FreeSolv, and Lipo $( \rho _ { s } < 1 )$ , and PRISM improves over scratch across all three. This supports the initial prior choices: priors related to intermolecular interaction provide chemically meaningful relational supervision to encoders for solubility prediction.

The Gaussian-prior control separates prior quality from pretraining. Replacing chemical priors with an uninformative random prior degrades performance, showing that the gains are not merely due to an extra pretraining stage. Nevertheless, performance does not collapse, indicating that the model can still retain or recover graph-derived information beyond the imposed prior. This further supports the flexibility of distribution matching, which shapes prior-consistent relations without forcing the embedding to discard complementary structural information. Thus, relational pretraining is no sufficient by itself: its benefit depends on the quality of the relational structure being transferred.

FreeSolv remains a boundary case for PRISM with a 2D backbone: although its labels are smooth under the selected priors, PRISM is slightly weaker than supervised D-MPNN, suggesting that the 2D representation misses task-relevant factors such as molecular conformation. With a 3D-aware SchNet backbone, PRISM improves substantially on FreeSolv (Appendix H.10), showing that a task-aligned prior is most useful when the encoder can also represent the relevant physical factors.

Label energy can also guide prior selection when the relevant task factors are less obvious (Appendix H.9). First, the dynamic-selection experiments show that label energy can identify taskcompatible priors: when several priors are similarly aligned, the selected set retains their performance, while poorly aligned priors can be filtered out. Second, these results show that more priors are not necessarily better. On CYP2D6, for example, using all candidate priors yields a ROC-AUC of $4 2 . 0 8 \pm 9 . 3 5$ , whereas selecting the label-energy-compatible Pharm2D prior increases it to $7 3 . 3 1 \pm 1 . 0 2$ . Thus, complementary priors are useful when their relational geometries are taskaligned, but adding incompatible views can substantially degrade representation learning.

## 5 Prior-Grounded Interpretability

Finally, we examine whether PRISM’s prior spaces provide meaningful interpretation on solubility predictions. Figure 3 gives two examples: a temperature-dependent ranking study and molecule-wise prior-space contribution study.

## 5.1 Temperature-dependent ranking beyond structural similarity

Figure 3a compares experimental and predicted log S–T trends for two solutes in a fixed solvent, testing whether the solute ordering is preserved across temperature. Both PRISM and the baseline model Fastprop successfully predict the temperature trend. This demonstrates that PRISM can capture the observed temperature-dependent trends without explicit temperature-gradient supervision. Although pyrimidine-4-carboxamide and isatoic anhydride are almost equidistant from the solvent in the fingerprint, PEOE, and SMR prior spaces, their distances in the SlogP prior space differ significantly, making SlogP the only clearly distinguishing space among the four priors in this comparison. While the baseline reverses the experimental solute ordering, suggesting difficulty in resolving these subtle chemical differences, PRISM effectively captures those differences through prior-induced relational pretraining, which allows PRISM to preserve the correct solute ranking.

## 5.2 Molecule-wise prior-space attribution in water

Figure 3b reports molecule-wise space attribution on ESOL in water, measured as adaptive fusionweight deviation from a uniform baseline. Because each branch is aligned with a distinct prior space, these deviations indicate which chemical views PRISM emphasizes for each solute.

![](images/700c8f7edc312984817a37a37dcedce25c794f7c9cc4570c37fe19cd845fd93e.jpg)

![](images/56f2f5afca0db7f43fc424dc06edd2e8af1558b2fc975c2ba0fb26f1ae035dcb.jpg)

![](images/fcc75cadd66ff85bd35449dc8d6316b31e45592e9201a40b65a97e6fcca72a4b.jpg)  
Figure 3: Interpretability diagnostics in pure-solvent solubility. (a) Prior-space profiles and temperature-dependent predictions for two representative solutes in a fixed solvent. PRISM preserves the correct solute ranking, whereas Fastprop reverses it. (b) Molecule-wise space attribution on ESOL in water, shown as fusion-weight deviation from a uniform baseline.

For cyclopentane-1,2-dicarboximide and glucose, the positive SlogP-related contribution suggests that hydrophobicity modulates how polar functionality translates into aqueous solubility. Glucose has multiple hydrogen-bond donors and acceptors and is highly water-compatible, whereas cyclopentane-1,2-dicarboximide reflects a stronger trade-off between polar hydration and hydrophobic structure.

For perylene, a nonpolar π-rich aromatic solute, attribution shifts toward PEOE-related information. We interpret this as a partial-charge-based electrostatic signal. This may reflect interactions between water-derived protons, or hydrated protons, and the extended aromatic π system, including cation– π-like effects [35]. Because the partial charges of conjugated aromatic π systems are not explicitly calculated in PEOE, this interaction can be attributed to the collective effect of sp<sup>2</sup>-connected atoms.

Overall, these diagnostics suggest that PRISM’s adaptive weights show prior-space contributions, rather than causal decompositions of individual predictions. Perturbation-based faithfulness tests in Appendix I.1 further support this interpretation.

## 6 Conclusion

We introduced PRISM, a label-free framework that uses expert-defined chemical priors as relational teachers, preserving prior-induced neighborhood geometry through soft distribution matching across prior-specific branches fused only downstream. Empirically, PRISM improves solubility MoleculeNet results, pure-solvent external transfer, and mixed-solvent OOD stability with frozen encoders and a lightweight head. Ablations show that these gains require both relational pretraining and prior-specific late fusion: descriptor input, reconstruction, scratch multi-branch, and shared early-fusion variants all fall short, while PRISM-Distilled compresses only part of the branch-specific signal. A label-energy diagnostic further turns prior selection into a measurable property, separating useful priors from uninformative ones before pretraining. Low-cost chemical priors thus provide structured relational supervision for molecular OOD generalization, and we expect the principle of PRISM to extend to other domains where chemically or biologically meaningful priors are available.

## Code Availability

The code for PRISM is available at https://github.com/cxy0206/PRISM.

## Acknowledgements and Disclosures

Funding. This work was supported by the Shanghai Jiao Tong University - Global College Research Start-up Fund and Young Top-notch Talent Support Program.

Competing interests. The authors declare no competing interests.

## References

[1] Luigi Da Vià, Matthias Depoortere, Robert D Willacy, Alastair J Roberts, Pandian Sokkar, Mathieu Fossépré, Andrew Ruba, and Magdalena A Zwierzyna. Enabling data-driven solubility modeling at gsk: Enhancing purge predictions for mutagenic impurities. Organic Process Research & Development, 28(12):4215–4224, 2024.

[2] Yeyun Chen. Relational invariant learning for robust solvation free energy prediction. In International Conference on Machine Learning, pages 72210–72227. PMLR, 2025.

[3] Jianyuan Deng, Zhibo Yang, Hehe Wang, Iwao Ojima, Dimitris Samaras, and Fusheng Wang. A systematic study of key elements underlying molecular property prediction. Nature Commu nications, 14(1):6395, 2023.

[4] Jan Wollschläger and Floriane Montanari. Solvmate – a hybrid physical/ML approach to solvent recommendation leveraging a rank-based problem framework. Digital Discovery, 3:1749–1760, 2024. doi: 10.1039/D4DD00138A.

[5] Vansh Ramani and Tarak Karmakar. Graph neural networks for predicting solubility in diverse solvents using MolMerger incorporating solute–solvent interactions. Journal of Chemical Theory and Computation, 20(15):6549–6558, 2024. doi: 10.1021/acs.jctc.4c00382.

[6] Wenjie Du, Shuai Zhang, Zhaohui Cai, Xuqiang Li, Zhiyuan Liu, Junfeng Fang, Jianmin Wang, Xiang Wang, and Yang Wang. Molecular merged hypergraph neural network for explainable solvation Gibbs free energy prediction. Research, 8:0740, 2025. doi: 10.34133/research.0740.

[7] Hojin Jung, Christopher D. Stubbs, Sabari Kumar, Raúl Pérez-Soto, Su-min Song, Yeonjoon Kim, and Seonah Kim. Enhancing predictive models for solubility in multicomponent solvent systems using semi-supervised graph neural networks. Digital Discovery, 4:1492–1504, 2025. doi: 10.1039/D5DD00015G.

[8] Namkyeong Lee, Dongmin Hyun, Gyoung S Na, Sungwon Kim, Junseok Lee, and Chanyoung Park. Conditional graph information bottleneck for molecular relational learning. In International Conference on Machine Learning, pages 18852–18871. PMLR, 2023.

[9] Namkyeong Lee, Kanghoon Yoon, Gyoung S. Na, Sein Kim, and Chanyoung Park. Shiftrobust molecular relational learning with causal substructure. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1200–1212. Association for Computing Machinery, 2023. doi: 10.1145/3580305.3599437.

[10] Y. Wang, J. Wang, Z. Cao, and A. Barati Farimani. Molecular contrastive learning of representations via graph neural networks. Nature Machine Intelligence, 4:279–287, 2022. doi: 10.1038/s42256-022-00447-x. URL https://www.nature.com/articles/ s42256-022-00447-x.

[11] Tianyang Hu, Zhili Liu, Fengwei Zhou, Wenjia Wang, and Weiran Huang. Your contrastive learning is secretly doing stochastic neighbor embedding. arXiv preprint arXiv:2205.14814, 2022.

[12] Gabriel A Pinheiro, Juarez LF Da Silva, and Marcos G Quiles. Smiclr: contrastive learning on multiple molecular representations for semisupervised and unsupervised representation learning. Journal ofChemical Information and Modeling, 62(17):3948–3960, 2022.

[13] Ross Irwin, Spyridon Dimitriadis, Jiazhen He, and Esben Jannik Bjerrum. Chemformer: a pretrained transformer for computational chemistry. Machine Learning: Science and Technology, 3(1):015022, 2022.

[14] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.

[15] Michele Ceriotti, Gareth A. Tribello, and Michele Parrinello. Simplifying the representation of complex free-energy landscapes using sketch-map. Proceedings of the National Academy of Sciences, 108(32):13023–13028, 2011. doi: 10.1073/pnas.1108486108.

[16] Damien E. Coupry and Peter Pogány. Application of deep metric learning to molecular graph similarity. Journal ofCheminformatics, 14:11, 2022. doi: 10.1186/s13321-022-00595-7.

[17] Roberto Todeschini and Viviana Consonni. Molecular descriptorsfor chemoinformatics: volume I: alphabetical listing/volume II: appendices, references. John Wiley & Sons, 2009.

[18] David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of chemical information and modeling, 50(5):742–754, 2010.

[19] Paul Labute. A widely applicable set of descriptors. Journal of Molecular Graphics and Modelling, 18(4-5):464–477, 2000.

[20] Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao, Angel Guzman-Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, Andrew Palmer, Volker Settels, Tommi Jaakkola, Klavs Jensen, and Regina Barzilay. Analyzing learned molecular representations for property prediction. Journal of Chemical Information and Modeling, 59(8):3370–3388, 2019. doi: 10.1021/acs.jcim.9b00237.

[21] Hugo Talibart and Dimitri Gilis. Learning a chemistry-aware latent space for molecular encoding and generation with a large-scale transformer variational autoencoder. Journal of Cheminformatics, 18(1):123, 2026. doi: 10.1186/s13321-026-01243-0.

[22] Yuankai Luo, Lei Shi, and Veronika Thost. Improving self-supervised molecular representation learning using persistent homology. Advances in Neural Information Processing Systems, 36: 34043–34073, 2023.

[23] Jun-Hyung Park, Hyuntae Park, Yeachan Kim, Woosang Lim, and SangKeun Lee. Moleco: Molecular contrastive learning with chemical language models for molecular property prediction. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 408–420, 2024.

[24] Geoffrey E Hinton and Sam Roweis. Stochastic neighbor embedding. Advances in neural information processing systems, 15, 2002.

[25] Hao Xu, Zhengyang Zhou, and Pengyu Hong. Graph multi-similarity learning for molecular property prediction. arXiv preprint arXiv:2401.17615, 2024.

[26] Zeqing Bao, Gary Tom, Austin Cheng, Jeffrey Watchorn, Alán Aspuru-Guzik, and Christine Allen. Towards the prediction of drug solubility in binary solvent mixtures at various temperatures using machine learning. Journal of Cheminformatics, 16(1):117, 2024.

[27] Lucas Attia, Jackson W Burns, Patrick S Doyle, and William H Green. Data-driven organic solubility prediction at the limit of aleatoric uncertainty. Nature Communications, 16(1):7497, 2025.

[28] Laurens Van Der Maaten and Kilian Weinberger. Stochastic triplet embedding. In 2012 IEEE International Workshop on Machine Learningfor Signal Processing, pages 1–6. IEEE, 2012.

[29] Eric Inae, Gang Liu, and Meng Jiang. Motif-aware attribute masking for molecular graph pre-training. arXiv preprint arXiv:2309.04589, 2023.

[30] Yashaswi Pathak, Siddhartha Laghuvarapu, Sarvesh Mehta, and U. Deva Priyakumar. Chemically interpretable graph interaction network for prediction of pharmacokinetic properties of drug-like molecules. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 873–880, 2020. doi: 10.1609/aaai.v34i01.5433.

[31] Wenjie Du, Shuai Zhang, Di Wu, Jun Xia, Ziyuan Zhao, Junfeng Fang, and Yang Wang. MMGNN: A molecular merged graph neural network for explainable solvation free energy prediction. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 5808–5816, 2024. doi: 10.24963/ijcai.2024/642.

[32] Walid Ahmad, Elana Simon, Seyone Chithrananda, Gabriel Grand, and Bharath Ramsundar. Chemberta-2: Towards chemical foundation models, 2022.

[33] Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. Self-supervised graph transformer on large-scale molecular data. Advances in neural information processing systems, 33:12559–12571, 2020.

[34] Francesca Cenci, Samir Diab, Paola Ferrini, Catajina Harabajiu, Massimiliano Barolo, Fabrizio Bezzo, and Pierantonio Facco. Predicting drug solubility in organic solvents mixtures: A machine-learning approach supported by high-throughput experimentation. International Journal ofPharmaceutics, 660:124233, 2024. doi: 10.1016/j.ijpharm.2024.124233.

[35] Dennis A Dougherty. The cation- π interaction in chemistry and biology. Chemical Reviews, 125(5):2793–2808, 2025.

[36] Boshra Ariguib, Mathias Niepert, and Andrei Manolache. Learning the neighborhood: Contrastfree self-supervised molecular graph pretraining. In New Perspectives in Graph Machine Learning, 2025.

[37] Kisung Moon, Hyeon-Jin Im, and Sunyoung Kwon. 3d graph contrastive learning for molecular property prediction. Bioinformatics, 39(6):btad371, 2023.

[38] Jiying Zhang, Zijing Liu, Yu Wang, Bin Feng, and Yu Li. Subgdiff: A subgraph diffusion model to improve molecular representation learning. Advances in Neural Information Processing Systems, 37:29620–29656, 2024.

[39] Zewei Ji, Runhan Shi, Jiarui Lu, Fang Li, and Yang Yang. Relmole: Molecular representation learning based on two-level graph similarities. Journal ofChemical Information and Modeling, 62(22):5361–5372, 11 2022. ISSN 1549-9596. doi: 10.1021/acs.jcim.2c00798. URL https: //doi.org/10.1021/acs.jcim.2c00798.

[40] Xiaohong Ji, Zhen Wang, Zhifeng Gao, Hang Zheng, Linfeng Zhang, Guolin Ke, and Weinan E. Exploring molecular pretraining model at scale. volume 37, pages 46956–46978, 2024.

[41] Oscar Méndez-Lucio, Christos A Nicolaou, and Berton Earnshaw. Mole: a foundation model for molecular graphs using disentangled attention. Nature Communications, 15(1):9431, 2024.

[42] Yi Zhong, Gaozheng Li, Ji Yang, Houbing Zheng, Yongqiang Yu, Jiheng Zhang, Heng Luo, Biao Wang, and Zuquan Weng. Learning motif-based graphs for drug–drug interaction prediction via local–global self-attention. Nature Machine Intelligence, 6(9):1094–1105, 2024.

[43] Wenbin Ye and Quan Qian. Forecasting the molecular interactions: A hypergraph-based neural network for molecular relational learning. Knowledge-Based Systems, 300:112177, 2024.

[44] Ella Miray Rajaonson, Mahyar Rajabi Kochi, Luis Martin Mejía Mendoza, Seyed Mohamad Moosavi, and Benjamin Sanchez-Lengeling. Chemixhub: Datasets and benchmarks for chemical mixture property prediction. arXiv preprint arXiv:2506.12231, 2025.

[45] Charles M Hansen. Hansen solubility parameters: a user’s handbook. CRC press, 2007.

[46] Dávid Bajusz, Anita Rácz, and Károly Héberger. Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations? Journal ofCheminformatics, 7(1):20, 2015. doi: 10.1186/s13321-015-0069-3.

[47] Connor W Coley. Defining and exploring chemical spaces. Trends in Chemistry, 3(2):133–145, 2021.

[48] Yu Liu, Cole Mathis, Michał Dariusz Bajczyk, Stuart M Marshall, Liam Wilbraham, and Leroy Cronin. Exploring and mapping chemical space with molecular assembly trees. Science Advances, 7(39):eabj2465, 2021.

[49] Alexios Koutsoukas, Shardul Paricharak, Warren RJD Galloway, David R Spring, Adriaan P IJzerman, Robert C Glen, David Marcus, and Andreas Bender. How diverse are diversity assessment methods? a comparative analysis and benchmarking of molecular descriptor space. Journal ofchemical information and modeling, 54(1):230–242, 2014.

[50] Andreas Bender, Jeremy L Jenkins, Josef Scheiber, Sai Chetan K Sukuru, Meir Glick, and John W Davies. How similar are similarity searching methods? a principal component analysis of molecular descriptor space. Journal ofchemical information and modeling, 49(1):108–119, 2009.

[51] Florence H Vermeire, Yunsie Chung, and William H Green. Predicting solubility limits of organic solutes for a wide range of solvents and temperatures. Journal ofthe American Chemical Society, 144(24):10785–10797, 2022.

[52] Samuel Boobier, David RJ Hose, A John Blacker, and Bao N Nguyen. Machine learning with physicochemical relationships: solubility prediction in organic solvents and water. Nature communications, 11(1):5753, 2020.

[53] Jeremy R. Ash, Cas Wognum, Raquel Rodríguez-Pérez, Matteo Aldeghi, Alan C. Cheng, Djork-Arné Clevert, Ola Engkvist, Cheng Fang, Daniel J. Price, Jacqueline M. Hughes-Oliver, and W. Patrick Walters. Practically significant method comparison protocols for machine learning in small molecule drug discovery. Journal of Chemical Information and Modeling, 65(18): 9398–9411, 2025. doi: 10.1021/acs.jcim.5c01609.

[54] Henrike Veith, Noel Southall, Ruili Huang, Tim James, Darren Fayne, Natalia Artemenko, Min Shen, James Inglese, Christopher P Austin, David G Lloyd, et al. Comprehensive characterization of cytochrome p450 isozyme selectivity across chemical libraries. Nature biotechnology, 27(11):1050–1055, 2009.

[55] Yin Fang, Qiang Zhang, Ningyu Zhang, Zhuo Chen, Xiang Zhuang, Xin Shao, Xiaohui Fan, and Huajun Chen. Knowledge graph-enhanced molecular contrastive learning with functional prompt. Nature Machine Intelligence, 5:542–553, May 2023. ISSN 2522-5839. URL https://www.nature.com/articles/s42256-023-00654-0.

## A Notation and Organization of the Appendix

Table 8 summarizes the notation used throughout the paper and appendix.

Table 8: Summary of notation. Symbols match the main text.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $x$ </td><td>Molecular graph input</td></tr><tr><td> $y$ </td><td>Downstream label</td></tr><tr><td>y</td><td>Model prediction</td></tr><tr><td> $\mathbf { \bar { \mathcal { D } } _ { p r e } } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ </td><td>Unlabeled molecular graph dataset for pretraining</td></tr><tr><td> $\boldsymbol { B }$ </td><td>Mini-batch of molecules</td></tr><tr><td> $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { K } \}$ </td><td>Set of expert-defined prior spaces</td></tr><tr><td> $\pi _ { i } ^ { s _ { k } }$ </td><td>Prior vector of molecule i in prior space  $s _ { k }$ </td></tr><tr><td> $d ^ { s _ { k } } ( \cdot , \cdot )$ </td><td>Prior-space distance in prior space  $s _ { k }$ </td></tr><tr><td> $\delta ^ { s _ { k } } ( \cdot , \cdot )$ </td><td>Embedding-space distance</td></tr><tr><td> $\smash { \mathcal { A } _ { \mathrm { n r i c r } } ^ { s _ { k } } }$ </td><td>Relational constructor in prior space  $s _ { k }$ </td></tr><tr><td> $\mathcal { A } _ { \mathrm { e m h } } ^ { \mathrm { } s _ { k } }$ </td><td></td></tr><tr><td></td><td>Relational constructor in embedding space for branch  $s _ { k }$ </td></tr><tr><td> $\mathcal { M } ^ { s _ { k } }$ </td><td>Discrepancy between prior- and embedding-induced relational objects</td></tr><tr><td> $\mathcal { R } _ { \mathrm { p r i o r } } ^ { s _ { k } }$ </td><td>Relational object induced by prior space  $s _ { k }$ </td></tr><tr><td> $\mathcal { R } ^ { \dot { s } _ { k } }$  emb</td><td>Relational object induced by learned embeddings for branch  $s _ { k }$ </td></tr><tr><td> $G _ { \theta _ { L } } ^ { s _ { k } }$ </td><td>Encoder branch for prior space  $s _ { k }$ </td></tr><tr><td></td><td></td></tr><tr><td> $r _ { i } ^ { s _ { k } ^ { s _ { k } ^ { s } } }$ </td><td>Branch embedding produced by  $G _ { \theta _ { k } } ^ { s _ { k } }$ </td></tr><tr><td> $R _ { i } ^ { s _ { k } }$ </td><td>Prior-induced soft neighborhood distribution for anchor i</td></tr><tr><td> $P _ { i } ^ { s _ { k } }$ </td><td>Learned soft neighborhood distribution for anchor i</td></tr><tr><td> $\beta _ { k }$ </td><td>Locality scale for prior space  $s _ { k }$ </td></tr><tr><td> $\beta _ { \tau }$ </td><td>Embedding-space inverse temperature</td></tr><tr><td> $q _ { s _ { k } }$ </td><td>Per-branch projection head for prior space  $s _ { k }$ </td></tr><tr><td> $h _ { i } ^ { s _ { k } }$ </td><td>Aligned branch embedding  $q _ { s _ { k } } ( r _ { i } ^ { s _ { k } } ) \in \mathbb { R } ^ { D }$ </td></tr><tr><td> $H _ { i }$ </td><td>Stack of all branch embeddings  $\doteq \mathbb { R } ^ { K \times D }$ </td></tr><tr><td></td><td>Gating network for fusion</td></tr><tr><td> $\mathbf { \Pi } _ { w _ { i } ^ { s _ { k } } } ^ { g }$ </td><td>Fusion weight for branch  $s _ { k }$  on input i</td></tr><tr><td> $w _ { i }$ </td><td>Fusion weight sequence on input i</td></tr><tr><td> $z _ { i }$ </td><td>Fused molecular embedding of molecule i</td></tr><tr><td> $\mathcal { N } _ { n } ^ { s _ { k } } ( x _ { i } )$ </td><td>n nearest neighbors of  $x _ { i }$  in prior space  $s _ { k }$ </td></tr><tr><td> $\mathcal { E } _ { n } ^ { s _ { k } } ( y )$ </td><td>Label energy on prior-space nNN graph</td></tr><tr><td> $\rho ^ { s _ { k } }$ </td><td>Normalized label-energy ratio for prior space  $s _ { k }$ </td></tr><tr><td> $Q$ </td><td>The set of labeled molecules used for diagnosis</td></tr><tr><td></td><td>Single-molecule prediction head</td></tr><tr><td> $f _ { \mathrm { s i n g l e } }$   $f _ { \mathrm { p u r e } }$ </td><td>Pure-solvent solubility prediction head</td></tr><tr><td> $f _ { \mathrm { m i x } }$ </td><td>Mixed-solvent solubility prediction head</td></tr><tr><td> $x _ { \mathrm { s o l } }$ </td><td>Solute molecule</td></tr><tr><td> $x _ { \mathrm { s o l v } }$ </td><td>Pure solvent molecule</td></tr><tr><td></td><td>Two solvents in a binary solvent mixture</td></tr><tr><td> $x _ { 1 } , x _ { 2 }$ </td><td>Fused solute embedding</td></tr><tr><td> ${ \cal z } _ { \mathrm { s o l } }$ </td><td>Fused pure-solvent embedding</td></tr><tr><td> ${ \mathit { z } } _ { \mathrm { s o l v } }$ </td><td></td></tr><tr><td> $z _ { 1 } , z _ { 2 }$ </td><td>Fused embeddings of the two mixture solvents</td></tr><tr><td> $\alpha _ { 1 } , \alpha _ { 2 }$ </td><td>Mole fractions of the two mixture solvents</td></tr><tr><td> $z _ { \mathrm { m i x } }$ </td><td>Composition-weighted solvent-mixture embedding</td></tr><tr><td> $T$ </td><td>Temperature</td></tr><tr><td> $h _ { T }$ </td><td>Temperature embedding network</td></tr><tr><td> $t = h _ { T } ( T )$ </td><td>Temperature embedding</td></tr></table>

## B Related Work

We discuss three lines of work most relevant to PRISM.

Molecular self-supervised learning. Molecular SSL commonly defines pretraining signals through contrastive learning, generative pretraining, multi-view learning, or latent representation learning [36]. Contrastive and multi-view methods construct reference neighborhoods from augmentations, alternative molecular views, or paired representations [10–12, 37], while generative and denoising methods use masking, reconstruction, or corruption kernels [14, 13, 38]. Recent molecular SSL further moves beyond single negative/positive-pair construction by using multi-similarity signals, fingerprint-based similarity, or persistent homology to provide richer neighborhood, ranking, or topological supervision [25, 23, 22]. ReLMole [39] further uses atom-level and functional-grouplevel graph similarities to guide contrastive molecular pretraining. Large-scale molecular foundation models scale these pretext mechanisms and often incorporate 3D conformational information [40, 41]. PRISM belongs to this relational view of SSL but changes the source and organization of the relational target: reference neighborhoods are induced by expert-defined chemical priors, and heterogeneous prior geometries are preserved in separate branches before downstream fusion. Thus, PRISM turns chemically meaningful prior spaces into label-free relational teachers, rather than relying only on augmentation, reconstruction, or fused similarity targets.

Molecular relational learning and solubility modeling. Molecular relational learning explicitly models relations among molecules, including drug–drug interactions, solute–solvent interactions, molecular complexes, and higher-order molecular sets [8, 9, 42, 43]. Solubility-specific models further encode solute–solvent or mixture interactions through solvent recommendation, merged molecular graphs, hypergraphs, or multicomponent architectures [1, 4–7]. Recent binary- and multicomponent-solvent benchmarks also highlight the need to evaluate prediction under unseen solvent environments [26, 44]. These methods are strong supervised relation predictors; PRISM is complementary, pretraining reusable molecular encoders from unlabeled molecules through priorinduced relational supervision.

Descriptor-based chemical spaces. Molecular descriptors and fingerprints provide interpretable chemical coordinate systems for physicochemical surfaces, substructures, pharmacophores, and 3D conformation [45, 19, 46]. They are commonly used as inputs, auxiliary features, reconstruction targets, or visualization tools [47, 48]. PRISM uses them differently: descriptors and fingerprints are not appended to downstream predictors or reconstructed pointwise; they define population-level stochastic neighborhoods for pretraining. Because different priors induce different neighborhoods over the same molecules [49, 50], PRISM keeps prior-specific branches and fuses them only downstream, enabling chemically structured relational supervision with interpretable prior-level contributions.

## C Datasets, Splits, and Evaluation Protocols

This section details the datasets, split protocols, leakage controls, and evaluation metrics used throughout the paper. Unless otherwise noted, molecules are identified by canonical SMILES strings.

## C.1 Dataset Overview

We evaluate PRISM on three families of downstream tasks. First, we use standard MoleculeNet benchmarks to assess molecular representation quality under scaffold-disjoint molecular property prediction. Second, we evaluate pure-solvent solubility prediction, where each example contains a solute, a solvent, temperature, and a solubility label. Third, we evaluate mixed-solvent solubility prediction, where each example contains a solute, two solvents, mixture composition, temperature, and a solubility label.

Table 9 summarizes the datasets used in this work. The unlabeled ZINC250k subset is used only for PRISM pretraining. All labeled downstream datasets are used only for supervised fine-tuning and evaluation.

Table 9: Summary of datasets used in this work.
<table><tr><td>Dataset</td><td>Task</td><td>#Samples</td><td>Split</td><td>Metric</td><td></td></tr><tr><td>ZINC250k</td><td>pretraining</td><td>~250k</td><td></td><td></td><td></td></tr><tr><td>BBBP</td><td>classification</td><td>2,039</td><td>scaffold 8:1:1</td><td>ROC-AUC ↑</td><td></td></tr><tr><td>BACE</td><td>classification</td><td>1,513</td><td>scaffold 8:1:1</td><td>ROC-AUC↑</td><td></td></tr><tr><td>HIV</td><td>classification</td><td>41,127</td><td>scaffold 8:1:1</td><td>ROC-AUC↑</td><td></td></tr><tr><td>ESOL</td><td>regression</td><td>1,128</td><td>scaffold 8:1:1</td><td>RMSE↓</td><td></td></tr><tr><td>FreeSolv</td><td>regression</td><td>642</td><td>scaffold 8:1:1</td><td>RMSE↓</td><td></td></tr><tr><td>Lipo</td><td>regression</td><td>4,200</td><td>scaffold 8:1:1</td><td>RMSE↓</td><td></td></tr><tr><td>CYP2D6</td><td>classification</td><td>13,130</td><td>scaffold 8:1:1</td><td>ROC-AUC ↑</td><td></td></tr><tr><td>KOR</td><td>classification</td><td>7,963</td><td>scaffold 8:1:1</td><td>ROC-AUC ↑</td><td></td></tr><tr><td>BigSolDB[27]</td><td>pure-solvent solubility</td><td>41,724</td><td>solute-group 95/5</td><td></td><td>RMSE / %|∆ log S| ≤ 1</td></tr><tr><td>SolProp[51]</td><td>pure-solvent solubility</td><td>5,230</td><td>external test</td><td>RMSE / %|∆ log S|</td><td>|≤ 1</td></tr><tr><td>Leeds[52]</td><td>pure-solvent solubility</td><td>1,294</td><td>external test</td><td></td><td>RMSE / %|∆ log S| ≤ 1</td></tr><tr><td>Mixed-solvent [26]</td><td>binary-mixture solubility</td><td>26,730</td><td>random 8:1:1 / pair-disjoint</td><td></td><td>RMSE / %|∆ log S| ≤ 1</td></tr></table>

## C.2 MoleculeNet Property Benchmarks

We use MoleculeNet benchmarks to evaluate structural generalization under scaffold-disjoint splits. The main text focuses on three regression tasks that are most directly related to solubility and solventrelevant physicochemical behavior: ESOL, FreeSolv, and Lipophilicity. Additional classification benchmarks, including BBBP, BACE, and HIV, are reported in the appendix to provide a broader view of representation quality beyond solubility-centered regression.

Each sample is represented by a canonical SMILES string, and the target corresponds to an experimentally measured or curated molecular property.

Regression tasks emphasized in the main text. ESOL predicts aqueous solubility from molecular structure. FreeSolv predicts hydration free energy of small molecules in water. Lipophilicity predicts experimental lipophilicity, reflecting hydrophobicity and partitioning behavior. These tasks are used as the primary MoleculeNet structural-OOD evaluation because they are closest to the solubility-centered scope of this work.

Additional classification tasks. BBBP is a binary classification task indicating whether a compound can penetrate the blood–brain barrier. BACE is a binary classification task for inhibition activity against the β-secretase 1 target. HIV is a binary classification task indicating whether a compound shows activity in HIV replication inhibition assays. We include these tasks in the extended appendix results to assess whether PRISM also provides useful structural-OOD representations outside solubility-related regression.

Split protocol. For all MoleculeNet benchmarks, we use scaffold splits with an 8:1:1 train/validation/test ratio. This protocol evaluates generalization across scaffold-disjoint chemical series rather than random interpolation among closely related molecules. We report mean standard deviation across three random seeds. For classification tasks, we report ROC-AUC; for regression tasks, we report RMSE.

Note on MoleculeNet evaluation. We acknowledge known limitations of MoleculeNet evaluation, including scaffold-split inconsistency, benchmark leakage, and seed sensitivity [53]. Accordingly, MoleculeNet is used as supplementary structural-OOD evidence. The main empirical support for PRISM comes from solubility-centered external transfer and mixed-solvent solvent-pair-disjoint evaluation.

## C.3 Additional Bioactivity Benchmarks for Prior Generality

In addition to the six MoleculeNet benchmarks used in the main comparison, we include two additional bioactivity datasets, CYP2D6 [54] and KOR, in the prior-generality analyses. These datasets are not used as primary benchmark tasks in the main performance table. They are used to test whether PRISM remains useful when the prior space is changed, corrupted, or selected using the label-energy diagnostic.

CYP2D6. CYP2D6 is a small-molecule bioactivity classification task related to cytochrome P450 2D6, with 13,130 samples after canonicalization and deduplication. It provides a pharmacologically relevant setting where functional-group and pharmacophore information can be more directly relevant than generic physicochemical descriptors.

KOR. KOR is a small-molecule bioactivity classification task associated with the kappa opioid receptor, with 7,963 samples. We include it as an additional receptor-level activity task to evaluate whether pharmacophore-informed priors provide useful relational supervision beyond standard MoleculeNet tasks.

Usage in this work. CYP2D6 and KOR are used only in the supplementary prior-generality experiments, including Gaussian random prior degradation and dynamic prior selection with Pharm2D. For these datasets, we follow the same supervised fine-tuning protocol as the MoleculeNet classification tasks and report ROC-AUC, averaged over three seeds.

## C.4 Pure-Solvent Solubility Datasets

Pure-solvent solubility prediction evaluates whether a model can predict solute solubility across different solvents and temperatures. Each record is $\langle x _ { \mathrm { s o l } } , x _ { \mathrm { s o l v } } , T , \log S \rangle$ , where T is temperature in units of K and log S is the experimental solubility label.

BigSolDB. We use BigSolDB as the training and validation source. BigSolDB contains experimental solubility measurements across diverse organic solvents and temperatures. To evaluate solute extrapolation, we group all records by canonical solute SMILES and assign each solute group entirely to either training or validation. We use 95% of the filtered BigSolDB data for training and 5% for validation/model selection.

SolProp and Leeds. We use SolProp and Leeds as external test sets. They are not used for training, validation, or hyperparameter selection. Before training on BigSolDB, we remove all BigSolDB records whose canonical solute SMILES appear in either SolProp or Leeds. This overlap-removal step ensures that external evaluation measures extrapolate to solutes not seen during training.

## C.5 Mixed-Solvent Solubility Dataset

For mixed-solvent solubility prediction, we use the binary-solvent drug solubility dataset introduced by Bao et al. [26], which is also included as the drug-solubility task in CheMixHub [44]. This dataset was curated from the literature to support the prediction of small-molecule solubility in binary solvent mixtures under varying temperatures.

$$
\langle x _ { \mathrm { s o l } } , x _ { 1 } , x _ { 2 } , \alpha _ { 1 } , \alpha _ { 2 } , T , \log S \rangle ,\tag{9}
$$

where $x _ { \mathrm { s o l } }$ is the solute, $x _ { 1 }$ and $x _ { 2 }$ are the two solvents, $\alpha _ { 1 }$ and $\alpha _ { 2 }$ are their mole fractions $\left( \alpha _ { 1 } + \alpha _ { 2 } = \right.$ 1), T is temperature, and log S is solubility. The original dataset standardizes solubility values to log S, which denotes the base-10 logarithm of S with S measured in g/100g, and converts reported solvent ratios into consistent mixture-composition representations. Following the CheMixHub curation convention, we use mole fractions as composition variables in all mixed-solvent experiments.

After cleaning, deduplication, and retaining samples with valid molecular identifiers and mole-fraction fields, our working mixed-solvent dataset contains 26,730 experimental samples. It covers 125 unique solutes, 44 unique solvents, 110 unique binary solvent pairs, and 377 unique solute–binary-solvent systems. The temperature range is 252.55–368.15 K, the mole-fraction range is 0.0–1.0, and the log S range is 8.85 to 0.02.

Because the dataset includes endpoint compositions (α = 0 or 1), we additionally check exact endpoint duplicates defined by solute, effective pure solvent, temperature, and label. No exact endpoint record is shared across train/validation/test splits under the solvent-pair-disjoint protocol.

Why this dataset. Mixed-solvent solubility is a natural testbed for PRISM because the target depends not only on the solute and each solvent individually, but also on solvent–solvent and solute–environment interactions. The dataset therefore evaluates whether prior-induced molecular representations remain useful in a multi-molecular setting where composition and temperature modulate the effective solvent environment.

Dataset statistics. Table 10 summarizes the processed mixed-solvent dataset.

Table 10: Statistics of the processed mixed-solvent solubility dataset.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Total samples</td><td>26,730</td></tr><tr><td>Unique solutes</td><td>125</td></tr><tr><td>Unique solvents</td><td>44</td></tr><tr><td>Unique binary solvent pairs</td><td>110</td></tr><tr><td>Unique solute-binary-solvent systems</td><td>377</td></tr><tr><td>Temperature range (K)</td><td>252.55–368.15</td></tr><tr><td>Mole fraction range</td><td>0.0-1.0</td></tr><tr><td>Solubility range, log S</td><td>-8.85--0.02</td></tr><tr><td>Target unit</td><td>log(S), S in g/100g</td></tr></table>

## C.6 Splits and Leakage Control

MoleculeNet scaffold splits. For MoleculeNet benchmarks, we use scaffold-disjoint 8:1:1 train/validation/test splits. This split evaluates generalization across molecular scaffolds and is used for all MoleculeNet classification and regression experiments.

Pure-solvent solute-level split. For BigSolDB, all records associated with the same canonical solute SMILES are assigned to the same split. This prevents the same solute from appearing in both training and validation under different solvent or temperature conditions. For external evaluation, we additionally remove from BigSolDB all solutes that appear in SolProp or Leeds.

Mixed-solvent random split. For the mixed-solvent dataset, we first consider a sample-level random split with an 8:1:1 train/validation/test ratio. This split follows the same random-split principle as the default random benchmark setting in CheMixHub, but uses the 8:1:1 ratio adopted in our experiments. All samples are randomly shuffled with a fixed seed and then assigned to training, validation, and test sets. This split is not solvent-pair-disjoint, so the same binary solvent pair may appear across training and test sets. We report this split as an in-distribution benchmark rather than as a strict OOD evaluation.

Mixed-solvent solvent-pair canonicalization. A binary solvent mixture should be invariant to the ordering of the two solvents. To prevent representing the same physical mixture as either (A, B) or $( B , A )$ , we canonicalize each solvent pair by sorting the two canonical solvent SMILES:

$$
\operatorname { p a i r } ( s _ { 1 } , s _ { 2 } ) = ( \operatorname* { m i n } ( s _ { 1 } , s _ { 2 } ) , \operatorname* { m a x } ( s _ { 1 } , s _ { 2 } ) ) .\tag{10}
$$

All pair-level split keys are computed using this canonicalized representation.

Mixed-solvent solvent-pair-disjoint split. We further evaluate a stricter solvent-pair-disjoint split, designed to test generalization to unseen binary solvent environments. We collect the set of unique canonical solvent-pair keys , randomly shuffle them with a fixed seed, and split them into train/validation/test subsets. A sample is assigned to a split only through its canonical solvent-pair key:

$$
\begin{array} { r l } { ( x _ { \mathrm { s o l } } , s _ { 1 } , s _ { 2 } , \alpha _ { 1 } , \alpha _ { 2 } , T , y ) \in \mathcal { D } _ { \mathrm { t r a i n } } } & { { } \Longleftrightarrow \quad \mathrm { p a i r } ( s _ { 1 } , s _ { 2 } ) \in \mathcal { P } _ { \mathrm { t r a i n } } , } \end{array}\tag{11}
$$

and analogously for validation and test. This protocol guarantees that no binary solvent pair appears in more than one split. Because different solvent pairs have different numbers of measurements, the final sample-level proportions may deviate from the nominal pair-level proportions. Across repeated runs, the solvent-pair split is resampled with different random seeds.

Interpretation of the two mixed-solvent splits. The random split measures whether the model can learn the overall solubility landscape under the same data distribution used for training. The solvent-pair-disjoint split measures environment-OOD generalization, where the model must predict solubility for binary solvent pairs not observed during training. We treat the random split as the standard in-distribution benchmark and the solvent-pair-disjoint split as the main OOD stress test.

Relation to CheMixHub splitting protocols. CheMixHub defines four benchmark splitting strategies for chemical-mixture property prediction: random cross-validation, mixture-size splits, leavemolecules-out splits, and temperature splits. Its default random benchmark uses 5-fold random cross-validation with a 70/10/20 ratio. Our mixed-solvent random split follows the same sample-level random-split principle but uses the 8:1:1 ratio adopted throughout our experiments. Our solvent-pairdisjoint split differs from leave-molecules-out: leave-molecules-out removes individual chemical components from training, whereas our split removes entire binary solvent environments. We do not use mixture-size splits because the drug-solubility dataset is restricted to solute–binary-solvent systems and does not provide meaningful variation in mixture size.

## C.7 OOD Split Visualization and Diagnostics

We provide additional diagnostics to clarify the distribution shifts induced by the evaluation splits.

Structural OOD under scaffold splits. Figure 4 visualizes the FreeSolv scaffold split in Morganfingerprint PCA space and compares the corresponding hydration-free-energy distributions. The scaffold split induces visible structural shift: validation and test molecules occupy several fingerprintspace regions that are sparse in the training set. At the same time, FreeSolv also exhibits nonnegligible label-distribution differences across splits, which is expected given its small sample size. This supports our treatment of FreeSolv as a boundary case: its evaluation reflects scaffold-level structural generalization, but performance is also affected by limited data, label-range imbalance, and hydration-specific physics that may require 3D information.

Solute and solvent OOD in pure-solvent transfer. For pure-solvent solubility, the external-transfer setting from BigSolDB to SolProp and Leeds follows the dataset analysis in Attia et al. [27]. Their study shows that these external test sets contain solute and solvent shifts relative to the BigSolDB training source, making them suitable for evaluating extrapolation beyond seen solute–solvent combinations. In our protocol, all BigSolDB records whose solute appears in SolProp or Leeds are removed before training, so every external-test solute is unseen during supervised fine-tuning. SolProp contains both seen- and unseen-solvent cases.

Unseen solvent-pair OOD in mixed-solvent prediction. Figure 5 provides diagnostics for the solvent-pair-disjoint mixed-solvent split. By construction, each binary solvent pair is canonicalized by sorting the two canonical solvent SMILES and is assigned exclusively to train, validation, or test. We explicitly verify that the canonical pair-key intersections are empty: $\mathcal { P } _ { \mathrm { t r a i n } } \cap \mathcal { P } _ { \mathrm { v a l } } = \emptyset$ $\mathcal { P } _ { \mathrm { t r a i n } } \cap \mathcal { P } _ { \mathrm { t e s t } } = \bar { \varnothing }$ , and $\mathcal { P } _ { \mathrm { v a l } } \cap \mathcal { P } _ { \mathrm { t e s t } } = \emptyset$ . Thus, every validation or test example contains a binary solvent environment absent from training.

![](images/06bc0be18d2d32f88a5150796f6a41ec9a814470429c62e1318895bceb323826.jpg)  
Figure 5: Diagnostics for the solvent-pair-disjoint mixed-solvent split.

The marginal distributions in Figure 5 show that log-solubility, temperature, and solvent composition remain broadly overlapping across train, validation, and test sets. Therefore, the solvent-pair-disjoint split is not primarily a temperature-extrapolation or composition-extrapolation benchmark. Rather, it evaluates whether a model can generalize to unseen binary solvent environments while remaining within broadly comparable label, temperature, and composition support.

## C.8 Evaluation Metrics and Repeated-Run Protocol

For classification tasks, we report ROC-AUC. For regression tasks, we report RMSE. For solubility prediction, we additionally report

$$
\mathcal { \gamma } _ { 0 } | \Delta \log S | \leq 1 = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \mathbf { 1 } \{ | \hat { y } _ { i } - y _ { i } | \leq 1 \} \times 1 0 0 \% .\tag{12}
$$

Unless otherwise specified, we report mean standard deviation across three random seeds. For split-dependent evaluations, the split is regenerated under the corresponding constraint for each seed. For the mixed-solvent random split, each seed corresponds to a new sample-level random 8:1:1 split; for the solvent-pair-disjoint split, each seed corresponds to a new split over canonical solvent-pair keys. Thus, the reported standard deviations capture run-to-run variation including both initialization and split resampling.

Limitations of three-seed evaluation. We acknowledge that three seeds is the minimum statistical protocol and may underestimate true variance, particularly on small datasets such as FreeSolv. This is a known limitation of standard MoleculeNet protocols [53]. We mitigate this by reporting full standard deviations and by emphasizing the larger and more stable solubility benchmarks for the main empirical claims.

## C.9 Baseline Selection and Comparison Protocols

Baseline sets differ across experiments because the evaluated tasks require different input structures and support different native model capabilities. We select baselines according to five criteria: input compatibility or transparent adaptability, protocol compatibility, established competitiveness, learningparadigm coverage, and the ability to test a specific alternative explanation for PRISM’s gains. We distinguish controlled comparisons from broader capability-matched and domain-level comparisons. Table 11 summarizes the resulting comparison groups and protocols.

Source note. The GROVER result in the single-molecule comparison is taken from Fang et al. [55], rather than directly from the original GROVER paper.

Table 11: Baseline selection and comparison protocols. “Reported” denotes values taken from prior studies under compatible published protocols, while “Reproduced” denotes baselines evaluated in our experiments.
<table><tr><td>Setting</td><td>Comparison type</td><td>Baseline group</td><td>Selection rationale</td><td>Protocol</td><td>Source</td></tr><tr><td>Single-molecule prediction</td><td>Supervised control</td><td>Supervised GNNs</td><td>Tests whether relational pretraining improves over direct supervised learning from molecular</td><td>One molecular graph as input; the Reported full model is trained on the task-specific split.</td><td></td></tr><tr><td>Single-molecule prediction</td><td>Generic molecular SSL</td><td>2D and geometry-aware pretrained models</td><td>graphs. Compares scientific relational supervision with established molecular pretraining</td><td>Respective pretrained encoders with task-specific downstream training.</td><td>Reported</td></tr><tr><td>Single-molecule prediction</td><td>Direct-prior control</td><td>Traditional ML with fingerprints and VSA priors</td><td>paradigms. Tests whether gains can be explained by directly supplying the informative prior</td><td>Concatenated raw prior features; no relational pretraining.</td><td>Reproduced</td></tr><tr><td>External pure-solvent transfer</td><td>Protocol-matched supervised</td><td>Fastprop, Chemprop, Vermeire</td><td>Compares PRISM with established supervised solute-solvent predictors</td><td>Trained on cleaned BigSolDB and tested directly on Leeds and SolProp.</td><td>Reported</td></tr><tr><td>External pure-solvent transfer</td><td>Large-scale pretrained</td><td>MolFormer, ChemBERTa</td><td>Tests whether generic large-scale pretraining alone explains transfer performance.</td><td>Pretrained molecular encoders are fine-tuned on BigSolDB and evaluated unchanged on Leeds and SolProp.</td><td>Reported</td></tr><tr><td>External pure-solvent transfer</td><td>Direct-prior control</td><td>XGBoost</td><td>Tests whether relational pretraining adds transfer value beyond direct access to the same priors.</td><td>Raw prior features for solute and solvent together with temperature; trained on BigSolDB and tested on Leeds and SolProp.</td><td>Reproduced</td></tr><tr><td>Mixed-solvent pair-disjoint OOD</td><td>Controlled architecture</td><td>Same-backbone D-MPNN aggregation controls</td><td>Isolates relational pretraining from backbone capacity and downstream aggregation.</td><td>Same D-MPNN backbone and five pair-disjoint splits; models are trained end-to-end without molecular pretraining. Full models are trained and</td><td>Reproduced</td></tr><tr><td>Mixed-solvent pair-disjoint OOD</td><td>Capability- matched interaction</td><td>Triple-CIGIN, Triple-MMGNN</td><td>Tests whether specialized end-to-end interaction models can learn transferable</td><td>evaluated on the same five pair-disjoint splits.</td><td>Reproduced</td></tr><tr><td>Mixed-solvent pair-disjoint OOD</td><td>Lightweight frozen ChemBERTa, SSL</td><td>MolCLR, GROVER</td><td>solvent-pair relations. Tests whether PRISM&#x27;s advantage is specific to relational representations rather than frozen molecular pretraining in</td><td>Encoders are frozen; matched downstream pooling and prediction modules are trained on the same five splits.</td><td>Reproduced</td></tr><tr><td>Mixed-solvent pair-disjoint OOD</td><td>Direct-prior control</td><td>XGBoost and prior-input MLP</td><td>general. Tests whether flexible supervised models can match PRISM through direct access to the prior</td><td>Raw prior features for the solute and two solvents together with composition and temperature; evaluated on the same five pair-disjoint splits.</td><td>Reproduced</td></tr><tr><td>mixture evaluation comparison</td><td>Pure/binary/ternary Domain-level</td><td>Cenci-style PLS and UNIFAC-based models</td><td>Tests whether PRISM adds value beyond descriptor-based and thermodynamic approaches and whether their information is</td><td>Dataset-specific compatible protocols; PRISM is used only in representation-based or combined-model variants.</td><td>Reproduced</td></tr></table>

## D Chemical Prior Spaces

This section describes the chemical prior spaces used in PRISM. The default PRISM configuration uses three VSA descriptor families and one Morgan fingerprint prior. We additionally evaluate a Pharm2D pharmacophore prior and a 3D shape prior in supplementary generality experiments; these auxiliary priors are not part of the default four-branch model unless explicitly stated.

## D.1 Overview of Prior Spaces

PRISM uses expert-defined chemical spaces as relational teachers. Each prior space defines a chemically meaningful notion of molecular similarity, and the encoder is trained to preserve the corresponding neighborhood structure in the learned embedding space. The goal is not to copy descriptor values directly, but to preserve the relative organization induced by a chosen chemical view.

The prior spaces used in this work fall into five categories: (i) VSA descriptor priors, including SMR\_VSA, SlogP\_VSA, and PEOE\_VSA; (ii) Morgan fingerprint priors, encoding local substructure similarity; (iii) Pharm2D priors, encoding pharmacophore-level arrangements of chemical features; (iv) 3D shape priors, encoding global molecular shape and spatial mass distribution; (v) Gaussian random priors, used only as a negative control.

The first two categories form the default PRISM prior set. Pharm2D and 3D priors are used to evaluate whether PRISM is prior-agnostic and whether task-dependent prior selection can identify useful non-default priors. The Gaussian random prior is used to test degradation under uninformative priors.

## D.2 VSA Descriptor Families

We use three van der Waals surface area (VSA) descriptor families: SMR\_VSA, SlogP\_VSA, and PEOE\_VSA. For a molecule with atoms $a = 1 , \ldots , N$ , each atom is assigned an approximate surfacearea contribution $A _ { a }$ and an atom-level property value $q _ { a } .$ . Atoms are then assigned to predefined bins according to $q _ { a } .$ , and the descriptor coordinate for bin k is

$$
v _ { k } = \sum _ { a : q _ { a } \in [ b _ { k } , b _ { k + 1 } ] } A _ { a } .\tag{13}
$$

Each coordinate measures how much molecular surface area is contributed by atoms whose property value falls into a particular range.

SlogP\_VSA: hydrophobic surface partitioning. SlogP\_VSA bins atoms according to their Crippen logP fragment contributions and sums the surface area within each bin. It provides a surface-weighted description of hydrophobic and lipophilic contributions.

SMR\_VSA: polarizability-related surface partitioning. SMR\_VSA bins atoms according to their Crippen molar refractivity contributions. Molar refractivity is closely related to molecular size and electronic polarizability and is associated with dispersion-related interactions.

PEOE\_VSA: electrostatic surface partitioning. PEOE\_VSA bins atoms according to partial charges estimated by the partial equalization of orbital electronegativity procedure. It summarizes how molecular surface area is distributed across negatively charged, neutral, and positively charged regions. We therefore interpret PEOE\_VSA as an electrostatic surface prior rather than as a dispersion or polarizability prior.

This distinction is important for interpretation: SMR\_VSA is associated with polarizability/dispersionrelevant surface information, SlogP\_VSA with hydrophobicity/lipophilicity-related surface information, and PEOE\_VSA with partial-charge/electrostatic surface information.

## D.3 Morgan FP as Structural Priors

To complement the continuous physicochemical VSA priors, we use Morgan FP as a structural prior. Morgan FP encode local molecular substructures by iteratively expanding atom-centered neighborhoods and hashing the resulting identifiers into a fixed-length bit vector. Unless otherwise stated, we use radius $r = 2$ and 2048 bits.

Unlike VSA descriptors, which summarize surface-weighted physicochemical properties, Morgan FP emphasize graph topology and discrete substructure motifs. They capture information about ring systems, substituent patterns, local functional groups, and other structural fragments. Two molecules can be structurally similar under fingerprints while differing in hydrophobic, polarizability, or electrostatic surface distributions, making the fingerprint prior a useful complement to the VSA priors.

## D.4 Pharm2D Pharmacophore Priors

We additionally evaluate a Pharm2D prior in the generality experiments. Pharm2D fingerprints encode pharmacophore-level patterns, such as arrangements of hydrogen-bond donors, hydrogenbond acceptors, aromatic features, hydrophobic features, and ionizable groups. Compared with Morgan FP, which emphasize local graph substructures, Pharm2D captures a more interactionoriented view of molecular similarity.

This prior is especially relevant for bioactivity tasks where ligand–target recognition depends on pharmacophore compatibility rather than only on local substructure similarity. We therefore use Pharm2D in the dynamic prior-selection experiments on HIV, CYP2D6, and KOR (Appendix H.9). The purpose is not to claim that Pharm2D is universally better than the default priors, but to test whether PRISM can benefit from a task-relevant alternative when the label-energy diagnostic indicates stronger prior–label alignment.

## D.5 3D Shape Priors

We also evaluate a compact 3D shape prior in the prior-agnostic and encoder-agnostic experiments. The 3D prior uses standard RDKit shape descriptors covering complementary aspects of molecular geometry: global shape: normalized principal moments NPR1 and NPR2; anisotropy and compactness: Asphericity, Eccentricity, and SpherocityIndex; spatial extent: RadiusOfGyration; principal-axis mass distribution: PMI1, PMI2, and PMI3.

These descriptors provide a low-dimensional summary of molecular shape, elongation, compactness, and mass distribution. They are used to test whether PRISM can align learned representations to a geometry-derived prior space. For the 3D prior, conformers are generated from SMILES using RDKit; molecules for which conformer generation fails are excluded from the corresponding 3D-prior experiment but retained in all 2D-prior experiments. The compatibility of PRISM with 3D-aware molecular encoders such as SchNet is discussed in Appendix H.10.

## D.6 Distance Metrics and Standardization

Each prior space is paired with a distance metric matching its representation type.

Continuous descriptor spaces. For VSA descriptors and 3D shape descriptors, we apply coordinate-wise standardization within each mini-batch before computing Euclidean distances. For a mini-batch B and a descriptor vector $\pi ^ { ( s ) } ( x )$ , we compute

$$
\widetilde { \pi } ^ { ( s ) } ( x ) = \frac { \pi ^ { ( s ) } ( x ) - \mu _ { B } ^ { ( s ) } } { \sigma _ { B } ^ { ( s ) } + \epsilon } ,\tag{14}
$$

where $\mu _ { B } ^ { ( s ) }$ and $\sigma _ { B } ^ { ( s ) }$ are the coordinate-wise mean and standard deviation over the current mini-batch, and ϵ is a small numerical constant. The prior distance is then

$$
d _ { s } ( x _ { i } , x _ { j } ) = \left. \widetilde { \pi } ^ { ( s ) } ( x _ { i } ) - \widetilde { \pi } ^ { ( s ) } ( x _ { j } ) \right. _ { 2 } .\tag{15}
$$

This batch-wise standardization prevents descriptor coordinates with large numerical ranges from dominating the neighborhood construction, while preserving relative variation within each descriptor coordinate in the current batch.

Binary or sparse fingerprint spaces. For Morgan FP and Pharm2D fingerprints, we use the Tanimoto similarity:

$$
d _ { s } ( x _ { i } , x _ { j } ) = 1 - \mathrm { T a n i m o t o } \left( \pi ^ { ( s ) } ( x _ { i } ) , \pi ^ { ( s ) } ( x _ { j } ) \right) .\tag{16}
$$

Learned embedding space. For the embedding-space comparator δ, we use cosine similarity after $\ell _ { 2 }$ normalization, as in the main text.

These choices follow standard practice for representation-consistent comparisons [46] and separate the metric used to define prior-space neighborhoods from the metric used to compare learned embeddings.

## D.7 Default and Auxiliary Prior Configurations

The default PRISM configuration uses four prior branches:

$$
\{ \mathrm { S M R \_ V S A } , \mathrm { \ S l o g P \_ V S A } , \mathrm { \ P E O E \_ V S A } , \mathrm { \ M o r g a n \ F P } \} .\tag{17}
$$

This configuration is used for the main MoleculeNet, pure-solvent, and mixed-solvent experiments unless otherwise specified.

The auxiliary prior configurations are used only in supplementary experiments: Pharm2D prior: dynamic prior-selection experiments on HIV, CYP2D6, and KOR (Appendix H.9); 3D shape prior: compatibility experiments with 3D backbones and 3D priors (Appendix H.10); Gaussian random prior: negative control for uninformative prior geometry (Appendix H.8).

This separation avoids conflating two claims. The default four-prior PRISM tests whether chemically interpretable 2D priors improve molecular representation learning. The auxiliary prior experiments test whether PRISM is flexible with respect to the choice of prior space.

## D.8 Prior Quality Diagnostics

Not every prior space is expected to be useful for every downstream task. We use label energy as a diagnostic for prior–label alignment: a prior has low label energy when nearby molecules in that prior space tend to have similar labels. The formal definition of raw label energy, the normalized labelenergy ratio, and the prediction-theoretic interpretation are given in Appendix F.3 (Proposition 1). We use label energy in two ways: (i) to compare chemically meaningful priors against a Gaussian random prior, and (ii) to select among candidate priors for bioactivity tasks (Appendix H.9).

## E PRISM Implementation Details

This section provides implementation details for PRISM pretraining and PRISM downstream adaptation. We describe molecular featurization, mini-batch stochastic-neighborhood construction, locality scaling, the pretraining algorithm, fusion modes, task-specific prediction heads, hyperparameter selection, and computational cost. Unless otherwise stated, the default PRISM configuration uses four prior-specialized branches corresponding to SMR\_VSA, SlogP\_VSA, PEOE\_VSA, and Morgan FP.

## E.1 D-MPNN Backbone and Molecular Featurization

All molecular graphs are constructed from canonical SMILES using RDKit. Given an input SMILES string, we parse it into an RDKit Mol object and discard molecules that fail sanitization. We keep RDKit’s default perception of aromaticity, valence, and bond order. Unless otherwise specified, we do not perform additional tautomer enumeration or salt stripping. The same canonical SMILES strings are used for deduplication, split construction, and leakage control.

We use a directed message passing neural network (D-MPNN) [20] as the default molecular encoder, with hidden dimension 512. For multi-prior PRISM, each prior space has its own encoder branch $G _ { \theta _ { s } }$ . This design allows each branch to specialize in preserving the relational geometry induced by one prior space.

We use the standard 2D atom and bond features adopted by Chemprop [20]: atomic number, total degree, formal charge, chiral tag, total number of attached hydrogens, hybridization, aromaticity, and scaled atomic mass for atoms; bond type, conjugation, ring membership, and stereo for bonds. The atom feature dimension is 133, and the bond feature dimension is 14. Full feature definitions are listed in Table 12.

Table 12: Atom and bond features used in our D-MPNN featurization. All features are 2D RDKitderived.
<table><tr><td>Level</td><td>Feature group</td><td>RDKit field</td><td>Dim.</td></tr><tr><td>Atom</td><td>Atomic number</td><td>GetAtomicNum() shifted by —1</td><td>101</td></tr><tr><td>Atom</td><td>Total degree</td><td>GetTotalDegree()</td><td>7</td></tr><tr><td>Atom</td><td>Formal charge</td><td>GetFormalCharge()</td><td>6</td></tr><tr><td>Atom</td><td>Chiral tag</td><td>GetChiralTag()</td><td>5</td></tr><tr><td>Atom</td><td>Total #H</td><td>GetTotalNumHs()</td><td>6</td></tr><tr><td>Atom</td><td>Hybridization</td><td>GetHybridization()</td><td>6</td></tr><tr><td>Atom</td><td>Aromaticity</td><td>GetIsAromatic()</td><td>1</td></tr><tr><td>Atom</td><td>Scaled atomic mass</td><td>GetMass()×0.01</td><td>1</td></tr><tr><td>Bond</td><td>Padding indicator</td><td>bond is None</td><td>1</td></tr><tr><td>Bond</td><td>Bond type</td><td>single/double/triple/aromatic</td><td>4</td></tr><tr><td>Bond</td><td>Conjugation</td><td>GetIsConjugated()</td><td>1</td></tr><tr><td>Bond</td><td>Ring membership</td><td>IsInRing()</td><td>1</td></tr><tr><td>Bond</td><td>Stereo</td><td>GetStereo() one-hot + UNK</td><td>7</td></tr></table>

## E.2 Mini-Batch Stochastic Neighborhood Construction

Because PRISM uses a mini-batch approximation, batch size can affect neighborhood quality. We tested batch-size sensitivity in a representative SlogP\_VSA setup. Increasing batch size from 256 to 512 reduces the held-out KL loss after pretraining from 0.0102 to 0.0086, while downstream performance remains essentially unchanged. We use B = 512 as the default because it provides a slightly better neighborhood approximation while remaining feasible on a single RTX 3090, with peak memory around 75%. We do not claim that 512 is globally optimal; it is an effective, practical choice.

## E.3 Locality Scaling β<sub>s</sub> and Embedding Temperature τ

Different prior spaces have different distance scales. For example, standardized VSA descriptors are continuous vectors, whereas Morgan FP are sparse binary vectors compared by Tanimoto distance.

To make neighborhood sharpness comparable across prior spaces, we scale the locality parameter $\beta _ { s }$ using the median off-diagonal distance within the mini-batch:

$$
\widetilde { d } _ { s } = \mathrm { m e d i a n } \left\{ d _ { i j } ^ { ( s ) } : i \neq j , i , j \in \mathcal { B } \right\} , \qquad \beta _ { s } = \frac { 1 } { \gamma _ { s } \widetilde { d } _ { s } + \epsilon } ,\tag{18}
$$

where $\epsilon = 1 0 ^ { - 8 }$ is used for numerical stability. The default setting is $\gamma _ { s } = 0 . 2 5$ . This makes $\gamma _ { s }$ a dimensionless locality control: smaller $\gamma _ { s }$ produces a sharper reference neighborhood; larger $\gamma _ { s }$ produces a softer one. The embedding temperature is $\begin{array} { r } { \tau = \frac { 1 } { \beta _ { \tau } } } \end{array}$ . Per-task hyperparameter values are reported in Section E.9.

## E.4 Pretraining Algorithm

Algorithm 1 summarizes PRISM pretraining for one prior space. Each prior branch is pretrained independently using its own prior-induced stochastic neighborhoods. This avoids forcing a single encoder to simultaneously match heterogeneous prior geometries; the architectural justification for separate branches is given in Appendix H.3.

Algorithm 1 PRISM pretraining for one prior space s.   
1: Sample mini-batch with $| B | = B$ molecules.   
2: Compute prior-space distances $d _ { i j } ^ { ( s ) }$ for $i , j \in B .$   
3: Compute $\widetilde { d } _ { s }$ median $\{ d _ { i j } ^ { ( s ) } : i \neq j \}$ and set $\beta _ { s } \gets 1 / ( \gamma _ { s } \widetilde { d } _ { s } + \epsilon )$   
4: Compute reference neighborhoods $R _ { i } ^ { ( s ) }$ for each anchor $i \in \boldsymbol { B }$ using Eq. (3).   
5: Encode molecules with $G _ { \theta _ { s } }$ , apply projection head and $\ell _ { 2 }$ normalization.   
6: Compute learner neighborhoods $P _ { i } ^ { ( s ) }$ for each anchor using Eq. (3).   
7: Compute $\mathcal { L } _ { \mathrm { a l i g n } } ^ { ( s ) }$ using $\operatorname { E q . }$ . (4).   
8: Update $G _ { \theta _ { s } }$ and the projection head by AdamW.

Table 13 reports the default PRISM pretraining hyperparameters. Each prior branch is pretrained independently on the same ZINC250k corpus. Unless otherwise specified, the same optimizer and schedule are used for all four default prior spaces.

Table 13: Reference hyperparameters for PRISM pretraining.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Pretraining corpus</td><td>ZINC250k</td></tr><tr><td>Prior branches</td><td>SMR_VSA, SlogP_VSA, PEOE_VSA, Morgan FP D-MPNN</td></tr><tr><td>Backbone</td><td></td></tr><tr><td>Encoder hidden dimension</td><td>512</td></tr><tr><td>Projection head</td><td>2-layer MLP, 512–64, ReLU</td></tr><tr><td>Batch size Optimizer</td><td>512</td></tr><tr><td>Learning rate</td><td>AdamW  $1 0 ^ { - 4 }$ </td></tr><tr><td></td><td></td></tr><tr><td>Weight decay Epochs</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Embedding temperature τ</td><td>100 task-selected; default grid in Tables 17–19</td></tr><tr><td>Locality scale  $\gamma _ { s }$ </td><td>task-selected; default grid in Tables 17–19</td></tr><tr><td>Local scaling constant €</td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>Checkpoint selection</td><td>Lowest validation KL loss</td></tr></table>

## E.5 Projection Heads and Fusion Modes

During downstream adaptation, PRISM combines representations from multiple prior-specialized branches. For molecule $x _ { i } ,$ each branch produces $r _ { i } ^ { ( s ) } = G _ { \theta _ { s } } ( x _ { i } )$ . Before fusion, each branch embedding is passed through a two-layer projection module:

$$
h _ { i } ^ { ( s ) } = W _ { 2 } \sigma ( W _ { 1 } r _ { i } ^ { ( s ) } + b _ { 1 } ) + b _ { 2 } \in \mathbb { R } ^ { D } ,\tag{19}
$$

where σ is ReLU and both the input and output dimensions are $D = 5 1 2$ . The branch embeddings are stacked as

$$
H _ { i } = [ h _ { i } ^ { ( s _ { 1 } ) } ; \ldots ; h _ { i } ^ { ( s _ { K } ) } ] \in \mathbb { R } ^ { K \times D } .\tag{20}
$$

The aligned branch embeddings are then combined by weighted fusion to produce $z _ { i }$

We use the term weighted fusion throughout the paper. This module is not a sparse mixture-of-experts routing: all branches remain active, and the learned weights represent soft preferences over prior spaces.

Task-specific weighted fusion. For tasks with global prior-space weights, PRISM learns a sampleindependent logit vector $\boldsymbol { \ell } \in \mathbb { R } ^ { K }$ by logistic regression. The branch weights are

$$
w ^ { ( s _ { k } ) } = \frac { \exp ( \ell _ { k } ) } { \sum _ { m = 1 } ^ { K } \exp ( \ell _ { m } ) } ,\tag{21}
$$

and the fused representation is

$$
z _ { i } = \sum _ { k = 1 } ^ { K } w ^ { ( s _ { k } ) } h _ { i } ^ { ( s _ { k } ) } .\tag{22}
$$

This task-specific variant is used for classification and solute–solvent prediction, where a stable task-level combination of prior spaces is empirically more robust.

Input-adaptive weighted fusion. For single-molecule regression tasks, PRISM also supports molecule-specific fusion weights. For each branch embedding $h _ { i } ^ { ( s _ { k } ) }$ , a small gating MLP produces a branch logit:

$$
a _ { i } ^ { ( s _ { k } ) } = v _ { 2 } ^ { \top } \sigma \left( W _ { 1 } h _ { i } ^ { ( s _ { k } ) } + b _ { 1 } \right) + b _ { 2 } ,\tag{23}
$$

where the gating MLP has dimensions $D \ \to \ \operatorname* { m a x } ( 1 2 8 , D / 2 ) \ \to \ 1$ . We also add a learnable branch-specific bias $b ^ { ( s _ { k } ) }$ . The input-adaptive fusion weights are

$$
w _ { i } ^ { ( s _ { k } ) } = \frac { \exp \left( ( a _ { i } ^ { ( s _ { k } ) } + b ^ { ( s _ { k } ) } ) / \tau _ { f } \right) } { \sum _ { m = 1 } ^ { K } \exp \left( ( a _ { i } ^ { ( s _ { m } ) } + b ^ { ( s _ { m } ) } ) / \tau _ { f } \right) } ,\tag{24}
$$

where $\tau _ { f } = 1$ is a learnable softmax temperature clamped to be positive in implementation. The fused representation is then

$$
z _ { i } = \sum _ { k = 1 } ^ { K } w _ { i } ^ { ( s _ { k } ) } h _ { i } ^ { ( s _ { k } ) } .\tag{25}
$$

This input-adaptive variant is useful when the dominant physicochemical factor varies across molecules, as in ESOL, FreeSolv, and Lipophilicity. A comparison of task-specific and input-adaptive fusion is provided in Appendix H.5.

Regularization and branch-level representations. After weighted summation, we apply dropout and, when enabled, layer normalization to the fused representation. The branch embedding stack $H _ { i } \in \mathbb { R } ^ { K \times D }$ is retained for diagnostic analyses and ablation variants. Each row of $H _ { i }$ corresponds to one prior-aligned branch, preserving heterogeneous chemical perspectives before they are collapsed by weighted fusion. The main mixed-solvent head uses only the fused embeddings $z _ { i }$ , keeping the evaluation focused on whether frozen PRISM-pretrained encoders can be compositionally reused through lightweight downstream fusion.

Interpretation of fusion weights. The default prior spaces are chemically related and partially redundant. Therefore, fusion weights should be interpreted as soft preference reweighting rather than hard causal assignments. A larger weight indicates that the corresponding branch is more useful for a task or instance, but it does not imply that other branches are irrelevant. We provide faithfulness tests for the learned fusion weights in Appendix I.1.

## E.6 Single-Molecule and Pure-Solvent Heads

Single-molecule prediction. For single-molecule tasks, the model first computes the fused molecular representation $z _ { i }$ using either task-specific or input-adaptive weighted fusion. The prediction is ${ \hat { y } } _ { i } = f _ { \mathrm { s i n g l e } } ( z _ { i } )$ , where $\bar { f } _ { \mathrm { s i n g l e } }$ is an MLP head. For classification, the head outputs logits and is trained with binary cross-entropy. For regression, the head outputs a scalar and is trained with mean squared error.

Pure-solvent solubility prediction. For pure-solvent solubility, each sample is $( x _ { \mathrm { s o l } } , x _ { \mathrm { s o l v } } , T , y )$ The solute and solvent are encoded by the same PRISM encoder:

$$
z _ { \mathrm { s o l } } = E _ { \Theta } ( x _ { \mathrm { s o l } } ) , \qquad z _ { \mathrm { s o l v } } = E _ { \Theta } ( x _ { \mathrm { s o l v } } ) , \qquad t = h _ { T } ( T ) \in \mathbb { R } ^ { d _ { T } } ,\tag{26}
$$

where $h _ { T }$ maps standardized scalar temperature to an embedding. Temperature is standardized using training-set statistics and mapped by a two-layer M $\begin{array} { r } { { \cal P } ( [ 1  \frac { d _ { T } } { 2 }  d _ { T } ] ) } \end{array}$

$$
h _ { T } ( T ) = W _ { 2 } \sigma ( W _ { 1 } \widetilde { T } + b _ { 1 } ) + b _ { 2 } .\tag{27}
$$

The prediction is $\hat { y } = f _ { \mathrm { p u r e } } ( [ z _ { \mathrm { s o l } } ; z _ { \mathrm { s o l v } } ; t ] )$ as in the main text.

In our implementation, $d _ { T } = 3 2$ and $f _ { \mathrm { p u r e } }$ is an MLP with hidden sizes (1024, 128, 64) and dropout 0.2. We train with mean squared error using Adam. Early stopping uses patience 50, and the learning-rate scheduler uses patience 30 with decay factor 0.5.

## E.7 Mixed-Solvent Head: Implementation Details

For mixed-solvent prediction, we reload and freeze the original PRISM-pretrained encoders rather than the pure-solvent-fine-tuned branches. Only the weighted-fusion module is initialized from the pure-solvent checkpoint and then trained together with the lightweight mixed-solvent head. This initialization is used as a warm start for prior-space weighting, while the frozen encoders ensure that no molecular message-passing parameters are updated using mixed-solvent supervision.

Given a solute embedding ${ \cal z } _ { \mathrm { s o l } }$ , two solvent embeddings $z _ { 1 } , z _ { 2 }$ , mole fractions $\alpha _ { 1 } , \alpha _ { 2 }$ with $\alpha _ { 1 } + \alpha _ { 2 } =$ 1, and temperature embedding $t = h _ { T } ( T )$ ), the main mixed-solvent head constructs a compositionweighted solvent representation:

$$
z _ { \mathrm { m i x } } = \alpha _ { 1 } z _ { 1 } + \alpha _ { 2 } z _ { 2 } , \qquad \hat { y } = f _ { \mathrm { m i x } } \left( \left[ z _ { \mathrm { s o l } } ; z _ { \mathrm { m i x } } ; t \right] \right) .\tag{28}
$$

The weighted-fusion module, temperature embedding $h _ { T } ,$ , and prediction head $f _ { \mathrm { m i x } }$ are trained in the default mixed-solvent protocol. The PRISM-pretrained encoders remain frozen.

Architecture details. Temperature is standardized using training-set statistics and mapped by a two-layer MLP:

$$
h _ { T } ( T ) = W _ { 2 } \sigma ( W _ { 1 } \widetilde { T } + b _ { 1 } ) + b _ { 2 } .\tag{29}
$$

The temperature embedding dimension is $d _ { T } = 3 2$ . The prediction head $f _ { \mathrm { m i x } }$ is an MLP with two hidden layers of size 512 and dropout 0.1.

Training. We train with mean squared error using Adam, learning rate $1 0 ^ { - 4 }$ , batch size 64, and up to 100 epochs. The best checkpoint is selected by validation MSE. We report RMSE and $\% | \Delta \log S | \stackrel { . } { \leq } 1$ on the test set.

Interpretation. This linear composition operation is not intended to fully model all non-ideal solvent–solvent interactions. Instead, it provides a deliberately lightweight test of representation transfer: if PRISM has transferred chemically meaningful relational structure into the frozen encoderbranch representations, then lightweight fusion and composition-weighted pooling should be sufficient to support generalization to unseen solvent-pair environments.

## E.8 Three-Component Interaction Baselines for Mixed Solvents

CIGIN [30] and MMGNN [31] were originally developed for interactions between two molecular components. To construct capability-matched interaction baselines for binary mixed-solvent prediction, we extend their respective interaction principles to a three-component system consisting

of a solute $G _ { s }$ and two solvents $G _ { 1 }$ and $G _ { 2 }$ . Both models are trained end-to-end on mixed-solvent labels without molecular pretraining. The solvent composition α and standardized temperature $\widetilde { T }$ are separately mapped to environment embeddings,

$$
e _ { \mathrm { e n v } } = \left[ f _ { \alpha } ( \alpha ) ; f _ { T } ( \widetilde { T } ) \right] ,\tag{30}
$$

which are supplied to the final prediction head.

Triple-CIGIN. CIGIN models intermolecular dependence by constructing interactions between atom-level representations of two molecules. For the three-component setting, we generalize this idea to a joint interaction tensor over the solute and both solvents. Separate MPNN encoders first produce atom representations $H _ { s } = \{ h _ { i } ^ { s } \} , H _ { 1 } = \{ h _ { i } ^ { 1 } \}$ , and $H _ { 2 } = \{ h _ { k } ^ { 2 } \}$ , with the same solvent encoder shared between the two solvent components. After learned linear projections,

$$
q _ { i } ^ { s } = W _ { s } h _ { i } ^ { s } , \qquad q _ { j } ^ { 1 } = W _ { 1 } h _ { j } ^ { 1 } , \qquad q _ { k } ^ { 2 } = W _ { 2 } h _ { k } ^ { 2 } ,\tag{31}
$$

we define the three-way interaction tensor

$$
A _ { i j k } = \frac { 1 } { \sqrt { d } } \sum _ { \ell = 1 } ^ { d } q _ { i \ell } ^ { s } q _ { j \ell } ^ { 1 } q _ { k \ell } ^ { 2 } ,\tag{32}
$$

where $i , j ,$ and k index atoms in the solute, solvent 1, and solvent 2, respectively.

For each component, A is normalized over atoms belonging to the other two components. For the solute, for example,

$$
P _ { i j k } ^ { s } = \frac { \exp ( A _ { i j k } ) } { \sum _ { j ^ { \prime } , k ^ { \prime } } \exp ( A _ { i j ^ { \prime } k ^ { \prime } } ) } , \qquad m _ { i } ^ { s } = \sum _ { j , k } P _ { i j k } ^ { s } \left( q _ { j } ^ { 1 } \odot q _ { k } ^ { 2 } \right) .\tag{33}
$$

Analogous operations yield $m _ { j } ^ { 1 }$ and $m _ { k } ^ { 2 }$ for the two solvents. The resulting interaction-enriched atom representations are

$$
\widetilde { h } _ { i } ^ { s } = [ h _ { i } ^ { s } ; m _ { i } ^ { s } ] , \qquad \widetilde { h } _ { j } ^ { 1 } = [ h _ { j } ^ { 1 } ; m _ { j } ^ { 1 } ] , \qquad \widetilde { h } _ { k } ^ { 2 } = [ h _ { k } ^ { 2 } ; m _ { k } ^ { 2 } ] .\tag{34}
$$

Each component is independently pooled with Set2Set to obtain $r _ { s } , r _ { 1 }$ , and $r _ { 2 } .$ , and the final prediction is

$$
\begin{array} { r } { \widehat { \boldsymbol { y } } = f _ { \mathrm { C I G I N } } \left( \left[ \boldsymbol { r } _ { s } ; \boldsymbol { r } _ { 1 } ; \boldsymbol { r } _ { 2 } ; \boldsymbol { e } _ { \mathrm { e n v } } \right] \right) . } \end{array}\tag{35}
$$

This construction extends the interaction-map principle of CIGIN to a genuine three-component interaction: the representation of each molecule is conditioned jointly on the other two components rather than on independent pairwise predictions.

Triple-MMGNN. MMGNN represents intermolecular interactions through a merged graph containing both covalent molecular structure and candidate intermolecular relations. We retain this merged-graph and interaction-selection principle while using the same MPNN family as the other end-to-end mixed-solvent controls. For a solute and two solvents, we construct

$$
G _ { \mathrm { m e r g e } } ^ { ( 3 ) } = G _ { s } \cup G _ { 1 } \cup G _ { 2 } \cup E _ { s 1 } ^ { \mathrm { i n t e r } } \cup E _ { s 2 } ^ { \mathrm { i n t e r } } \cup E _ { 1 2 } ^ { \mathrm { i n t e r } } ,\tag{36}
$$

where the three intermolecular edge sets connect atoms belonging to different molecular components. A learned component embedding is appended to each atom feature to identify membership in the solute, solvent 1, or solvent 2.

An initial merged-graph MPNN produces node states $h _ { v } ^ { ( 0 ) }$ . The model then learns a soft node mask

$$
m _ { v } = \sigma \Big ( f _ { \mathrm { n o d e } } \big ( h _ { v } ^ { ( 0 ) } \big ) \Big )\tag{37}
$$

and a soft edge mask

$$
\begin{array} { r } { m _ { u v } = \sigma \Big ( f _ { \mathrm { e d g e } } \big ( [ h _ { u } ^ { ( 0 ) } ; h _ { v } ^ { ( 0 ) } ; e _ { u v } ] \big ) \Big ) . } \end{array}\tag{38}
$$

Covalent edges are retained with unit weight, whereas the learned edge masks are applied to candidate intermolecular edges. A second merged-graph MPNN refines the node representations using the masked graph. The refined nodes are then pooled separately by molecular component,

$$
r _ { s } = \mathrm { P o o l } ( G _ { s } ) , \qquad r _ { 1 } = \mathrm { P o o l } ( G _ { 1 } ) , \qquad r _ { 2 } = \mathrm { P o o l } ( G _ { 2 } ) ,\tag{39}
$$

and the prediction is

$$
\widehat { \boldsymbol { y } } = f _ { \mathrm { M M G N N } } \left( \left[ \boldsymbol { r } _ { s } ; \boldsymbol { r } _ { 1 } ; \boldsymbol { r } _ { 2 } ; \boldsymbol { e } _ { \mathrm { e n v } } \right] \right) .\tag{40}
$$

Following the interaction-selection motivation of MMGNN, the learned node and intermolecular-edge masks are regularized toward a sparse Bernoulli prior during training. Our implementation should therefore be viewed as an MMGNN-style three-component control: it preserves the merged-graph and learned interaction-selection mechanisms while matching the MPNN-based backbone family used by the other end-to-end controls.

Role of the interaction baselines. Triple-CIGIN and Triple-MMGNN provide complementary high-capacity controls for explicit downstream interaction modeling. Triple-CIGIN introduces dense higher-order interaction conditioning while keeping the three molecular graphs separate, whereas Triple-MMGNN allows information exchange through a shared three-component graph with learned interaction selection. Both are trained end-to-end using mixed-solvent supervision. Their comparison with PRISM therefore tests whether explicit interaction capacity alone can provide the same generalization to unseen solvent combinations as relationally pretrained molecular representations combined with lightweight downstream composition.

## E.9 Hyperparameter Selection

Hyperparameters are selected using validation performance. For PRISM pretraining, we sweep the embedding temperature τ and locality scale $\gamma _ { s }$ over a small grid. For downstream fine-tuning, we tune dropout, learning rate, batch size, fusion mode, and weight decay.

Table 14 reports reference hyperparameters for single-molecule fine-tuning. Table 15 reports reference hyperparameters for pure-solvent solubility fine-tuning. Table 16 reports reference hyperparameters for mixed-solvent solubility fine-tuning.

Table 14: Reference hyperparameters for single-molecule fine-tuning.
<table><tr><td></td><td>ESOL</td><td>FreeSolv</td><td>Lipo</td><td>BBBP</td><td>BACE</td><td>HIV</td></tr><tr><td>Pretrain τ Pretrain</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.1</td><td>0.2</td><td>0.1</td></tr><tr><td> $\gamma _ { s }$  Fusion mode</td><td>0.75</td><td>0.75</td><td>0.25 Adaptive</td><td>0.25 Static</td><td>0.75</td><td>0.25</td></tr><tr><td>Dropout</td><td>Adaptive 0.3</td><td>Adaptive 0.3</td><td>0.3</td><td>0.1</td><td>Static 0.1</td><td>Static 0.1</td></tr><tr><td>Batch size</td><td>64</td><td>64</td><td>64</td><td>64</td><td>16</td><td>16</td></tr><tr><td>Weight decay</td><td>10⁻5</td><td>0</td><td>0</td><td>0</td><td>10⁻⁵</td><td>10-5</td></tr><tr><td>Epochs</td><td>200</td><td>200</td><td>200</td><td>200</td><td>100</td><td>200</td></tr></table>

Table 15: Reference hyperparameters for pure-solvent solubility fine-tuning.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Pretrain τ</td><td>0.1</td></tr><tr><td>Pretrain  $\gamma _ { s }$ </td><td>0.25</td></tr><tr><td>Fusion mode Embedding dimension D Temperature embedding  $d _ { T }$ </td><td>Static 512 32</td></tr><tr><td>Batch size Validation ratio Epochs</td><td>64 0.05 200</td></tr><tr><td>Weight decay MLP head hidden sizes MLP head dropout Encoder LR / head LR</td><td>10⁻5 (1024, 128, 64) 0.2  $1 0 ^ { - 4 } / 1 0 ^ { - 3 }$ </td></tr></table>

Table 16: Reference hyperparameters for mixed-solvent solubility fine-tuning.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Embedding dimension D</td><td>512</td></tr><tr><td>Temperature embedding dT</td><td>32</td></tr><tr><td>Fusion mode</td><td>Static weighted fusion</td></tr><tr><td>encoders</td><td>Original PRISM-pretrained branches, frozen</td></tr><tr><td>Fusion module</td><td>Initialized from pure-solvent checkpoint and trainable</td></tr><tr><td>Mixed head</td><td>Linear composition-weighted MLP</td></tr><tr><td>Trainable modules MLP head hidden sizes</td><td>Fusion module, temperature embedding, and prediction head</td></tr><tr><td>MLP head dropout</td><td>(512, 512) 0.1</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Optimizer</td><td></td></tr><tr><td></td><td>Adam</td></tr><tr><td>Learning rate</td><td>10⁻4</td></tr><tr><td>Epochs</td><td>100</td></tr><tr><td>Checkpoint selection</td><td>Validation MSE</td></tr></table>

## E.9.1 Sensitivity to pre-training hyperparameters (τ and $\gamma _ { s } )$

We study sensitivity to embedding temperature τ and the locality scale constant $\gamma _ { s }$ . We sweep $\tau \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ and $\gamma _ { s } \in \{ 0 . 2 5 , 0 . 5 0 , 0 . 7 5 \}$ and report downstream test performance.

Discussion (overall). Across tasks, performance varies smoothly over the grid without brittle behavior, suggesting that the method is reasonably robust. Best settings are mildly task-dependent. All pre-training and downstream hyperparameters are selected via validation set performance, and the chosen configurations for each task are fully reported.

MoleculeNet classification. Table 17 indicates that BBBP prefers a sharper locality, while BACE benefits from smoother scaling. This is consistent with task-dependent neighborhood granularity: some tasks reward emphasizing only very close prior neighbors, while others benefit from broader neighborhoods.

MoleculeNet regression. Table 18 shows mild-to-moderate variation across the grid with no universally optimal choice. In practice we adopt a simple default that performs consistently and avoid per-task over-tuning.

Pairwise solubility (pure-solvent). Table 19 indicates that performance remains stable across reasonable settings. In particular, $\tau { = } 0 . 1 , \gamma _ { s } { = } 0 . 2 5$ achieves the best Leeds performance, while SolProp is relatively insensitive. This supports using a fixed pre-training setting for the main solubility comparisons.

Table 17: Hyperparameter sensitivity on classification tasks. Test ROC-AUC ( ) under different pre-training settings of embedding temperature τ and locality scale $\gamma _ { s } .$ . HIV is omitted due to its much larger scale.
<table><tr><td rowspan=1 colspan=1>T     $\gamma _ { s }$ </td><td rowspan=1 colspan=1>BBBP   BACE</td></tr><tr><td rowspan=1 colspan=1>0.1  0.25</td><td rowspan=1 colspan=1>0.9459  0.8389</td></tr><tr><td rowspan=1 colspan=1>0.1  0.50</td><td rowspan=1 colspan=1>0.8997  0.8548</td></tr><tr><td rowspan=1 colspan=1>0.1  0.75</td><td rowspan=1 colspan=1>0.9006  0.8227</td></tr><tr><td rowspan=1 colspan=1>0.2  0.25</td><td rowspan=1 colspan=1>0.9017  0.8510</td></tr><tr><td rowspan=1 colspan=1>0.2  0.50</td><td rowspan=1 colspan=1>0.9071   0.8480</td></tr><tr><td rowspan=1 colspan=1>0.2  0.75</td><td rowspan=1 colspan=1>0.9085  0.8946</td></tr><tr><td rowspan=1 colspan=1>0.3  0.25</td><td rowspan=1 colspan=1>0.9036  0.8298</td></tr><tr><td rowspan=1 colspan=1>0.3  0.50</td><td rowspan=1 colspan=1>0.9044  0.8451</td></tr><tr><td rowspan=1 colspan=1>0.3  0.75</td><td rowspan=1 colspan=1>0.8994  0.8474</td></tr></table>

Table 18: Hyperparameter sensitivity on regression tasks. Test RMSE ( ) under different pretraining settings of τ and $\gamma _ { s }$
<table><tr><td>T</td><td> $\gamma _ { s }$ </td><td>ESOL</td><td>FreeSolv</td><td>Lipo</td></tr><tr><td>0.1</td><td>0.25</td><td>0.874</td><td>1.835</td><td>0.538</td></tr><tr><td>0.1</td><td>0.50</td><td>0.878</td><td>1.831</td><td>0.673</td></tr><tr><td>0.1</td><td>0.75</td><td>0.772</td><td>1.812</td><td>0.678</td></tr><tr><td>0.2</td><td>0.25</td><td>0.863</td><td>1.820</td><td>0.681</td></tr><tr><td>0.2</td><td>0.50</td><td>0.925</td><td>2.067</td><td>0.656</td></tr><tr><td>0.2</td><td>0.75</td><td>0.848</td><td>1.828</td><td>0.679</td></tr><tr><td>0.3</td><td>0.25</td><td>0.850</td><td>1.789</td><td>0.688</td></tr><tr><td>0.3</td><td>0.50</td><td>0.864</td><td>1.834</td><td>0.689</td></tr><tr><td>0.3</td><td>0.75</td><td>0.860</td><td>1.770</td><td>0.685</td></tr></table>

Table 19: Hyperparameter sensitivity on pure-solvent pairwise solubility. We report best validation RMSE, and test RMSE $/ \% l o g S \pm 1 ( \uparrow )$ on Leeds and SolProp. $\% l o g S \pm 1$ is shown in %. Only configurations available in our logs are listed.
<table><tr><td>T</td><td> $\gamma _ { s }$ </td><td>best val RMSE↓</td><td>Leeds RMSE↓</td><td>Leeds %logS ± 1 ↑</td><td>SolProp RMSE ↓</td><td>SolProp %logS ± 1 ↑</td></tr><tr><td>0.1</td><td>0.50</td><td>0.1414</td><td>0.9340</td><td>73.65</td><td>0.7940</td><td>80.71</td></tr><tr><td>0.1</td><td>0.25</td><td>0.1398</td><td>0.9099</td><td>75.66</td><td>0.7719</td><td>82.29</td></tr><tr><td>0.2</td><td>0.25</td><td>0.1393</td><td>0.9352</td><td>74.81</td><td>0.7853</td><td>81.72</td></tr><tr><td>0.2</td><td>0.75</td><td>0.1429</td><td>0.9179</td><td>74.65</td><td>0.7739</td><td>82.87</td></tr><tr><td>0.3</td><td>0.25</td><td>0.1438</td><td>0.9331</td><td>74.03</td><td>0.7563</td><td>82.73</td></tr></table>

## E.10 Compute Resources and Runtime

For K prior spaces and batch size B, the minibatch relational construction scales as $O ( K B ^ { 2 } )$ in pairwise comparisons. Table 20 summarizes the compute resources used in our experiments. All reported times are wall-clock times per run. Pretraining is the most expensive stage because it requires computing dense mini-batch pairwise neighborhood distributions. Downstream fine-tuning is substantially cheaper.

Table 20: Computational cost and hardware. Reported times are wall-clock per run.
<table><tr><td>Stage</td><td>Hardware</td><td>Time per run</td><td>Notes</td></tr><tr><td>PRISM pretraining per branch</td><td>RTX 3090</td><td>7–8 hours</td><td>Batch size  $5 1 2 ; \sim 7 5 \%$ </td></tr><tr><td>Pure-solvent fine-tuning</td><td>RTX 3090</td><td>~6 hours</td><td>peak GPU memory End-to-end</td></tr><tr><td>MoleculeNet fine-tuning</td><td>RTX 4070</td><td>≤1 hour</td><td>Per task, per seed</td></tr><tr><td>Mixed-solvent training</td><td>RTX 4070 mobile</td><td>3 hours</td><td>Frozen encoders; train- able fusion and head</td></tr></table>

The complete pipeline is feasible on a single consumer-grade GPU. The multi-branch design increases pretraining cost because each prior space has its own encoder, but the branches can be pretrained independently and in parallel. At downstream time, the additional cost is moderate relative to the molecular encoder forward passes and remains practical for the datasets studied here.

At fixed batch size, PRISM scales linearly with the pretraining-corpus size, while increasing the number of priors increases total work approximately linearly because prior-specific branches are trained independently and can be parallelized. In our profiling, D-MPNN encoding and backpropagation dominate runtime and memory, whereas the minibatch relational matrices add relatively little overhead. Restricting relational matching to sampled candidates reduces matching cost but yields only a 2–4% runtime reduction at the current scale, since all molecules still require graph encoding. Larger-scale implementations may therefore benefit more from reducing or reusing molecular encoding than from approximating the relational matrix alone.

## F Theoretical Analysis

This section provides theoretical support for the main objective-level claims used in the paper.

## F.1 Alignment Objectives in the PRISM

This appendix gives the formal definitions of the four alignment objectives compared. Throughout, denotes a mini-batch, $d _ { i j } ^ { ( s ) } = d _ { s } ( \pi _ { i } ^ { ( s ) } , \pi _ { j } ^ { ( s ) } )$ is the prior-space distance between molecules i and $j$ under prior space $s ,$ and $r _ { i } ^ { ( s ) } = G _ { \theta _ { s } } ( x _ { i } )$ is the corresponding branch embedding. We write sim $. ( \cdot , \cdot )$ for cosine similarity after $\ell _ { 2 }$ normalization, and define the raw embedding-space distance as

$$
\begin{array} { r } { \delta _ { i j } ^ { ( s ) } = 1 - \sin ( r _ { i } ^ { ( s ) } , r _ { j } ^ { ( s ) } ) . } \end{array}\tag{41}
$$

## F.2 Loss families

## F.2.1 Pointwise metric matching $( \ell _ { 2 } )$

Pointwise metric matching directly regresses pairwise embedding distances onto pairwise prior-space distances. Because prior-space distances and embedding-space distances can have different numerical scales, both distances are rescaled within each mini-batch before applying the regression loss. Let

$$
m _ { d } ^ { ( s ) } = \mathrm { m e d i a n } \left\{ d _ { i j } ^ { ( s ) } : i \neq j , i , j \in { \cal B } \right\} , \qquad m _ { \delta } ^ { ( s ) } = \mathrm { m e d i a n } \left\{ \delta _ { i j } ^ { ( s ) } : i \neq j , i , j \in { \cal B } \right\} .\tag{42}
$$

The scaled distances are

$$
\tilde { d } _ { i j } ^ { ( s ) } = \frac { d _ { i j } ^ { ( s ) } } { m _ { d } ^ { ( s ) } + \epsilon } , \qquad \tilde { \delta } _ { i j } ^ { ( s ) } = \frac { \delta _ { i j } ^ { ( s ) } } { m _ { \delta } ^ { ( s ) } + \epsilon } = \frac { 1 - \mathrm { s i m } ( r _ { i } ^ { ( s ) } , r _ { j } ^ { ( s ) } ) } { m _ { \delta } ^ { ( s ) } + \epsilon } ,\tag{43}
$$

where ϵ is a small numerical constant. The loss is

$$
\mathcal { L } _ { \ell _ { 2 } } ^ { ( s ) } = \frac { 1 } { | \mathcal { B } | ( | \mathcal { B } | - 1 ) } \underset { i \in \mathcal { B } } { \sum _ { j \in \mathcal { B } } } \underset { j \neq i } { \sum _ { \ell } } \Big ( \tilde { \delta } _ { i j } ^ { ( s ) } - \tilde { d } _ { i j } ^ { ( s ) } \Big ) ^ { 2 } .\tag{44}
$$

This objective is the most rigid in the alignment family: it requires the embedding to reproduce descriptor distances pointwise, including their magnitudes. When the prior space does not exhaust the information in the molecular graph, this rigidity over-constrains the encoder.

## F.2.2 Rank preservation

Rank-based alignment preserves only the ordinal structure of prior-space neighborhoods, discarding distance magnitude. For a fixed anchor $i ,$ each unordered neighbor pair $\{ j , k \} \subset B \setminus \{ i \}$ contributes one comparison. To avoid imposing an artificial order when the prior space gives equal distances, we define the set of strict unordered comparisons as

$$
\mathcal { T } _ { i } ^ { ( s ) } = \left\{ ( j , k ) : j , k \in \mathcal { B } \setminus \{ i \} , \ j < k , \ d _ { i j } ^ { ( s ) } \neq d _ { i k } ^ { ( s ) } \right\} .\tag{45}
$$

Anchors with $\vert \mathcal { T } _ { i } ^ { ( s ) } \vert = 0$ are skipped; this case is rare in continuous descriptor spaces and can occur only when no strict prior ordering is available for that anchor.

The non-differentiable rank disagreement rate is

$$
\mathcal { L } _ { \mathrm { r a n k } , 0 } ^ { ( s ) } = \frac { 1 } { | \mathcal { Z } ^ { ( s ) } | } \sum _ { i \in \mathcal { Z } ^ { ( s ) } } \frac { 1 } { | \mathcal { T } _ { i } ^ { ( s ) } | } \sum _ { ( j , k ) \in \mathcal { T } _ { i } ^ { ( s ) } } \mathbf { 1 } \Big [ \mathrm { s g n } ( d _ { i j } ^ { ( s ) } - d _ { i k } ^ { ( s ) } ) \neq \mathrm { s g n } ( \delta _ { i j } ^ { ( s ) } - \delta _ { i k } ^ { ( s ) } ) \Big ] ,\tag{46}
$$

where

$$
\mathcal { T } ^ { ( s ) } = \{ i \in \mathcal { B } : | \mathcal { T } _ { i } ^ { ( s ) } | > 0 \} .
$$

When there are no tied prior-space distances, this is equivalent to averaging over $\binom { | B | - 1 } { 2 }$ unordered neighbor pairs per anchor. The loss equals zero when the embedding ranking agrees with the priorspace ranking for every unordered comparison. Thus, unlike an ordered-pair double sum, Eq. (46) cannot produce negative loss values under perfect agreement.

For optimization, we use a smooth logistic surrogate. For each $( j , k ) \in \mathcal { T } _ { i } ^ { ( s ) }$ , define the prior-order label

$$
\begin{array} { r } { y _ { i j k } ^ { ( s ) } = \mathrm { s g n } \Big ( d _ { i k } ^ { ( s ) } - d _ { i j } ^ { ( s ) } \Big ) \in \{ - 1 , + 1 \} . } \end{array}\tag{47}
$$

Here $y _ { i j k } ^ { ( s ) } = + 1$ means that $j$ is closer than k to anchor i in prior space. The corresponding embedding-space score difference is

$$
s _ { i j k } ^ { ( s ) } = \delta _ { i k } ^ { ( s ) } - \delta _ { i j } ^ { ( s ) } .\tag{48}
$$

Thus, $y _ { i j k } ^ { ( s ) } s _ { i j k } ^ { ( s ) } > 0$ means that the embedding-space ordering agrees with the prior-space ordering. The differentiable rank loss is

$$
\mathcal { L } _ { \mathrm { r a n k } } ^ { ( s ) } = \frac { 1 } { \left| \mathcal { Z } ^ { ( s ) } \right| } \sum _ { i \in \mathcal { Z } ^ { ( s ) } } \frac { 1 } { \left| \mathcal { T } _ { i } ^ { ( s ) } \right| } \sum _ { ( j , k ) \in \mathcal { T } _ { i } ^ { ( s ) } } \log \left( 1 + \exp \left[ - y _ { i j k } ^ { ( s ) } s _ { i j k } ^ { ( s ) } / \tau _ { r } \right] \right) ,\tag{49}
$$

where $\tau _ { r } > 0$ is a rank-temperature parameter.

## F.2.3 Neighborhood-graph alignment (kNN + InfoNCE)

Neighborhood-graph alignment first uses prior-space kNN to determine positive and negative samples, then applies a contrastive objective. For each anchor $i \in { \cal B }$ , we define the positive set

$$
\mathcal { P } _ { i } ^ { ( s ) } = \{ j \in \mathcal { B } \setminus \{ i \} : j \in \mathcal { N } _ { k _ { \mathrm { N N } } } ^ { ( s ) } ( i ) \} ,\tag{50}
$$

i.e., the top- $\boldsymbol { \cdot } k _ { \mathrm { N N } }$ prior-space neighbors of i within the batch, and use all remaining in-batch molecules as negatives. We then apply the contrastive form used by MolCLR [10]:

$$
\mathcal { L } _ { \mathrm { N N G } } ^ { ( s ) } = \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \frac { - 1 } { | \mathcal { P } _ { i } ^ { ( s ) } | } \sum _ { j \in \mathcal { P } _ { i } ^ { ( s ) } } \log \frac { \exp \Bigl ( \sin ( r _ { i } ^ { ( s ) } , r _ { j } ^ { ( s ) } ) / \tau \Bigr ) } { \sum _ { \ell \in \mathcal { B } \backslash \{ i \} } \exp \Bigl ( \sin ( r _ { i } ^ { ( s ) } , r _ { \ell } ^ { ( s ) } ) / \tau \Bigr ) } ,\tag{51}
$$

where ℓ is a dummy summation index in the denominator, and τ is the contrastive temperature. The objective treats the kNN graph as a binary structure—each pair is either “positive” or “negative”—and discards the gradation of similarity within the neighbor set. When $\big | \mathcal { P } _ { i } ^ { ( s ) } \big | = 1$ , Eq. (51) reduces to standard InfoNCE with a single positive.

## F.2.4 Distribution matching (default)

The soft-neighborhood KL instantiation, used as the default in PRISM, defines anchor-wise stochastic neighborhoods on both sides and aligns them via KL divergence:

$$
R _ { i } ^ { ( s ) } ( j ) = \frac { \exp ( - \beta _ { s } d _ { i j } ^ { ( s ) } ) } { \sum _ { \ell \in \mathcal { B } \backslash \{ i \} } \exp ( - \beta _ { s } d _ { i \ell } ^ { ( s ) } ) } , \qquad P _ { i } ^ { ( s ) } ( j ) = \frac { \exp ( \sin ( r _ { i } ^ { ( s ) } , r _ { j } ^ { ( s ) } ) / \tau ) } { \sum _ { \ell \in \mathcal { B } \backslash \{ i \} } \exp ( \sin ( r _ { i } ^ { ( s ) } , r _ { \ell } ^ { ( s ) } ) / \tau ) } .\tag{52}
$$

$$
\mathcal { L } _ { \mathrm { K L } } ^ { ( s ) } = \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \sum _ { j \in \mathcal { B } \setminus \{ i \} } R _ { i } ^ { ( s ) } ( j ) \log \frac { R _ { i } ^ { ( s ) } ( j ) } { P _ { i } ^ { ( s ) } ( j ) } .\tag{53}
$$

Unlike the hard NNG instantiation, the soft form does not require a binary positive/negative partition: every in-batch molecule contributes to the anchor’s neighborhood with a graded weight determined by its prior-space distance. This preserves both neighbor identity (which molecules are close) and similarity magnitude (how close they are).

Conditional asymmetry and symmetric SNE alternative. The distributions in Eq. (52) are conditional on the anchor molecule and are therefore generally asymmetric: $R _ { i } ^ { ( s ) } ( j )$ need not equal $R _ { i } ^ { ( s ) } ( i )$ , even though the underlying distance $d _ { i j } ^ { ( s ) } = d _ { j i } ^ { ( s ) }$ is symmetric. This asymmetry arises from anchor-specific normalization and retains the local neighborhood context of each molecule.

To test whether this choice is important, we additionally consider a symmetric SNE-style formulation. Following symmetric SNE, we define

$$
\bar { R } _ { i j } ^ { ( s ) } = \frac { R _ { i } ^ { ( s ) } ( j ) + R _ { j } ^ { ( s ) } ( i ) } { 2 | \mathcal { B } | } , \qquad \bar { P } _ { i j } ^ { ( s ) } = \frac { P _ { i } ^ { ( s ) } ( j ) + P _ { j } ^ { ( s ) } ( i ) } { 2 | \mathcal { B } | } ,
$$

and minimize

$$
\mathcal { L } _ { \mathrm { s y m } } ^ { ( s ) } = \sum _ { i \neq j } \bar { R } _ { i j } ^ { ( s ) } \log \frac { \bar { R } _ { i j } ^ { ( s ) } } { \bar { P } _ { i j } ^ { ( s ) } } .
$$

Both formulations transfer useful relational information, but the conditional objective performs consistently better in our matched comparison. We interpret this result as evidence that retaining anchor-specific neighborhood context is useful for molecular representation learning, whereas symmetrization averages the two conditional neighborhood views.

Table 21: Comparison of conditional and symmetric SNE-style alignment under matched settings. Results are mean std over three seeds. RMSE ( ).
<table><tr><td>Objective</td><td>Lipo</td><td>FreeSolv</td><td>ESOL</td></tr><tr><td>Symmetric SNE</td><td> $0 . 5 9 2 \pm 0 . 0 0 9$ </td><td> $2 . 1 4 6 \pm 0 . 0 7 9$ </td><td> $0 . 8 0 3 \pm 0 . 0 3 2$ </td></tr><tr><td>PRISM conditional KL</td><td> $0 . 5 3 8 \pm 0 . 0 1 8$ </td><td> $1 . 7 7 0 \pm 0 . 0 9 3$ </td><td> $0 . 7 7 2 \pm 0 . 0 0 2$ </td></tr></table>

Connection to InfoNCE. We next relate the PRISM stochastic-neighborhood alignment objective to InfoNCE. For an anchor i in prior space s, PRISM aligns the prior-induced soft neighborhood $R _ { i } ^ { ( s ) }$ to the embedding-induced soft neighborhood $P _ { i } ^ { ( s ) }$ via KL divergence. The KL divergence decomposes as

$$
\mathrm { K L } ( R _ { i } ^ { ( s ) } \parallel P _ { i } ^ { ( s ) } ) = H ( R _ { i } ^ { ( s ) } , P _ { i } ^ { ( s ) } ) - H ( R _ { i } ^ { ( s ) } ) ,\tag{54}
$$

where $\begin{array} { r } { H ( p , q ) = - \sum _ { j } p _ { j } } \end{array}$ log $q _ { j }$ and $\begin{array} { r } { H ( p ) = - \sum _ { j } p _ { j } } \end{array}$ log $p _ { j }$ . Since $H ( R _ { i } ^ { ( s ) } )$ does not depend on encoder parameters, minimizing the KL divergence is equivalent to minimizing the soft cross-entropy from $R _ { i } ^ { ( s ) }$ to $P _ { i } ^ { ( s ) }$

In the hard-neighborhood special case where the prior distribution assigns all mass to one positive $j _ { i } ^ { + }$ $H ( R _ { i } ^ { ( s ) } ) = 0$ and Eq. (54) reduces to  log $P _ { i } ^ { ( s ) } ( j _ { i } ^ { + } )$ . With a softmax over scaled cosine similarities,

$$
- \log P _ { i } ^ { ( s ) } ( j _ { i } ^ { + } ) = - \log \frac { \exp ( \sin ( r _ { i } ^ { ( s ) } , r _ { j _ { i } ^ { + } } ^ { ( s ) } ) / \tau ) } { \sum _ { \ell \neq i } \exp ( \sin ( r _ { i } ^ { ( s ) } , r _ { \ell } ^ { ( s ) } ) / \tau ) } ,\tag{55}
$$

which is the InfoNCE loss with $j _ { i } ^ { + }$ as the positive and the remaining in-batch molecules as negatives. Thus, InfoNCE corresponds to a hard-target specialization of PRISM’s soft-neighborhood objective, whereas PRISM uses the soft distribution form natively to reflect graded chemical similarity. The difference is that PRISM does not require selecting a single positive example. Instead, the prior space provides a soft neighborhood distribution over all in-batch molecules, allowing multiple chemically related molecules to contribute with graded weights and avoiding a binary positive/negative partition when molecular similarity is continuous.

## F.2.5 TDL as a Rank-Projected Version of PRISM-KL

We next relate rank-style TDL/ListMLE objectives to PRISM-KL. Fix an anchor molecule i and let $\mathcal { C } _ { i } = B \setminus \{ i \}$ with $| { \mathcal { C } } _ { i } | = M$ . For prior space s, PRISM defines a prior-induced soft neighborhood distribution

$$
R _ { i } ^ { ( s ) } ( j ) = \frac { \exp ( - \beta _ { s } d _ { i j } ^ { ( s ) } ) } { \sum _ { \ell \in \mathcal { C } _ { i } } \exp ( - \beta _ { s } d _ { i \ell } ^ { ( s ) } ) } , \qquad j \in \mathcal { C } _ { i } ,
$$

and an embedding-induced distribution

$$
P _ { i } ^ { ( s ) } ( j ) = \frac { \exp ( u _ { i j } ^ { ( s ) } ) } { \sum _ { \ell \in \mathcal { C } _ { i } } \exp ( u _ { i \ell } ^ { ( s ) } ) } , \qquad u _ { i j } ^ { ( s ) } = \sin ( r _ { i } ^ { ( s ) } , r _ { j } ^ { ( s ) } ) / \tau .
$$

PRISM minimizes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P R I S M } } ^ { ( s ) } ( i ) = \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } ^ { ( s ) } \Big ) . } \end{array}
$$

Table 22: Controlled comparison of TDL-style and conditional KL alignment. Only the alignment objective is changed. Results are mean standard deviation over three seeds. Lower RMSE is better.
<table><tr><td>Objective</td><td>Lipo RMSE ↓</td><td>FreeSolv RMSE ↓</td><td>ESOL RMSE ↓</td></tr><tr><td>TDL-style alignment</td><td> $0 . 6 2 2 \pm 0 . 0 1 1$ </td><td> $2 . 2 2 1 \pm 0 . 0 2 8$ </td><td> $0 . 8 1 9 \pm 0 . 0 0 3$ </td></tr><tr><td>PRISM (conditional KL)</td><td> $\mathbf { 0 . 5 3 8 \pm 0 . 0 1 8 }$ </td><td> $\mathbf { 1 . 7 7 0 \pm 0 . 0 9 3 }$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 2 }$ </td></tr></table>

If the alignment is exact, $P _ { i } ^ { ( s ) } = R _ { i } ^ { ( s ) }$ , then for any $j , k \in \mathcal { C } _ { i } .$

$$
\frac { P _ { i } ^ { ( s ) } ( j ) } { P _ { i } ^ { ( s ) } ( k ) } = \frac { R _ { i } ^ { ( s ) } ( j ) } { R _ { i } ^ { ( s ) } ( k ) } = \exp \left[ - \beta _ { s } \left( d _ { i j } ^ { ( s ) } - d _ { i k } ^ { ( s ) } \right) \right] .\tag{56}
$$

Since the softmax normalizer cancels in the probability ratio, we have

$$
u _ { i j } ^ { ( s ) } - u _ { i k } ^ { ( s ) } = - \beta _ { s } \left( d _ { i j } ^ { ( s ) } - d _ { i k } ^ { ( s ) } \right) .\tag{57}
$$

Thus, PRISM-KL not only preserves the prior-induced ordering

$$
d _ { i j } ^ { ( s ) } < d _ { i k } ^ { ( s ) } \quad \Longrightarrow \quad u _ { i j } ^ { ( s ) } > u _ { i k } ^ { ( s ) } ,\tag{58}
$$

but also preserves the magnitude of the distance gap through the log-probability ratio.

TDL keeps only the rank projection of this condition. Let $\pi _ { 1 } , \ldots , \pi _ { M }$ be the candidates sorted by increasing prior distance,

$$
d _ { i \pi _ { 1 } } ^ { ( s ) } < d _ { i \pi _ { 2 } } ^ { ( s ) } < \dots < d _ { i \pi _ { M } } ^ { ( s ) } ,
$$

assuming a strict order for simplicity. For each position t, define the suffix set

$$
S _ { t } = \{ \pi _ { t } , \pi _ { t + 1 } , \ldots , \pi _ { M } \} = \{ m \in { \mathcal { C } } _ { i } : d _ { i m } ^ { ( s ) } \geq d _ { i \pi _ { t } } ^ { ( s ) } \} .
$$

On this suffix set, define the truncated embedding distribution

$$
P _ { i } ^ { ( t ) } ( m ) = \frac { \exp ( u _ { i m } ^ { ( s ) } ) } { \sum _ { \ell \in S _ { t } } \exp ( u _ { i \ell } ^ { ( s ) } ) } , \qquad m \in S _ { t } ,
$$

and a hard target $Q _ { i } ^ { ( t ) } ( m ) = \mathbf { 1 } [ m = \pi _ { t } ]$ . Applying hard-target KL on every suffix gives

$$
\sum _ { t = 1 } ^ { M } \mathrm { K L } \Big ( Q _ { i } ^ { ( t ) } \Big \| P _ { i } ^ { ( t ) } \Big ) = \sum _ { t = 1 } ^ { M } - \log \frac { \exp ( u _ { i \pi _ { t } } ^ { ( s ) } ) } { \sum _ { \ell = t } ^ { M } \exp ( u _ { i \pi _ { \ell } } ^ { ( s ) } ) } .\tag{59}
$$

Re-indexing $m = \pi _ { t }$ , this becomes

$$
\mathcal { L } _ { \mathrm { T D L } } ( i ) = \sum _ { m \in \mathcal { C } _ { i } } - \log \frac { \exp ( u _ { i m } ^ { ( s ) } ) } { \sum _ { \ell \in \mathcal { C } _ { i } } \mathbf { 1 } [ d _ { i \ell } ^ { ( s ) } \geq d _ { i m } ^ { ( s ) } ] \exp ( u _ { i \ell } ^ { ( s ) } ) } ,\tag{60}
$$

which is the Plackett–Luce/ListMLE form of TDL. Therefore, TDL can be viewed as first projecting the PRISM target distribution $R _ { i } ^ { ( s ) }$ to its rank ordering and then applying sequential hard-target KL over suffix sets. It matches rank $( R _ { i } ^ { ( s ) } )$ , whereas PRISM matches the full soft neighborhood distribution $R _ { i } ^ { ( s ) }$ itself.

Controlled empirical comparison. To complement the analysis above, we compare the TDL-style objective with PRISM’s conditional KL objective while holding the prior configuration, D-MPNN backbone, ZINC250k pretraining corpus, downstream model, and evaluation protocol fixed. This i an objective-level comparison rather than a full reproduction of the original method. Table 22 reports the results.

Discussion. Conditional KL achieves lower downstream RMSE on all three tasks. Together with the analysis above, these results support preserving graded prior-induced similarities rather than their ordering alone under this matched comparison, providing additional empirical justification for PRISM’s default alignment objective.

## F.2.6 Shared-Encoder Multi-Prior KL as Early-Fused Target Matching

We show that a shared-encoder multi-prior objective, where all prior spaces are optimized against the same embedding-induced distribution $P _ { i }$ , is mathematically equivalent to GraphMSL-style early fusion of target similarity distributions.

Fix an anchor molecule $i ,$ and let $\mathcal { C } _ { i } = B \setminus \{ i \}$ be the in-batch candidate set. Suppose that $K$ prior spaces define K target neighborhood distributions

$$
R _ { i } ^ { ( 1 ) } , R _ { i } ^ { ( 2 ) } , \ldots , R _ { i } ^ { ( K ) } , \qquad \sum _ { j \in { \mathcal C } _ { i } } R _ { i } ^ { ( s ) } ( j ) = 1 \quad \mathrm { f o r ~ a l l } \ s .
$$

In the shared-encoder control, all prior losses are optimized against the same embedding-induced distribution

$$
P _ { i } ( j ) = \frac { \exp ( u _ { i j } ) } { \sum _ { \ell \in \mathcal { C } _ { i } } \exp ( u _ { i \ell } ) } ,
$$

where $u _ { i j }$ is the shared embedding logit. The unweighted shared- $. P _ { i }$ multi-prior objective is

$$
\mathcal { L } _ { \mathrm { s h a r e d } } ( i ) = \sum _ { s = 1 } ^ { K } \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } \Big ) .\tag{61}
$$

Define the normalized early-fused target distribution as the average of the prior-induced distributions:

$$
\bar { R } _ { i } ( j ) = \frac { 1 } { K } \sum _ { s = 1 } ^ { K } R _ { i } ^ { ( s ) } ( j ) .\tag{62}
$$

Since each $R _ { i } ^ { ( s ) }$ is a probability distribution, $\bar { R } _ { i }$ is also a probability distribution.

Expanding the shared objective gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s h a r e d } } ( i ) = \displaystyle \sum _ { s = 1 } ^ { K } \sum _ { j \in \mathcal { C } _ { i } } R _ { i } ^ { ( s ) } ( j ) \log \frac { R _ { i } ^ { ( s ) } ( j ) } { P _ { i } ( j ) } } \\ & { \qquad = \displaystyle \sum _ { s = 1 } ^ { K } \sum _ { j } R _ { i } ^ { ( s ) } ( j ) \log R _ { i } ^ { ( s ) } ( j ) - \displaystyle \sum _ { s = 1 } ^ { K } \sum _ { j } R _ { i } ^ { ( s ) } ( j ) \log P _ { i } ( j ) . } \end{array}\tag{63}
$$

Using

$$
\sum _ { s = 1 } ^ { K } R _ { i } ^ { ( s ) } ( j ) = K \bar { R } _ { i } ( j ) ,
$$

the second term becomes

$$
- K \sum _ { j } \bar { R } _ { i } ( j ) \log P _ { i } ( j ) .
$$

Therefore,

$$
\mathcal { L } _ { \mathrm { s h a r e d } } ( i ) = \sum _ { s = 1 } ^ { K } \sum _ { j } R _ { i } ^ { ( s ) } ( j ) \log R _ { i } ^ { ( s ) } ( j ) - K \sum _ { j } \bar { R } _ { i } ( j ) \log P _ { i } ( j ) .\tag{64}
$$

Now consider the GraphMSL-style early-fusion objective. In this form, the multiple target distributions are first fused into a single target distribution

$$
T _ { i } ( j ) = \bar { R } _ { i } ( j ) ,
$$

and the model learns one embedding-induced distribution

$$
Q _ { i } ( j ) = P _ { i } ( j ) .
$$

The early-fusion cross-entropy objective is

$$
\mathcal { L } _ { \mathrm { e a r l y } } ( i ) = - \sum _ { j } T _ { i } ( j ) \log Q _ { i } ( j ) = - \sum _ { j } \bar { R } _ { i } ( j ) \log P _ { i } ( j ) .\tag{65}
$$

Comparing this with Eq. (64), we obtain

$$
\mathcal { L } _ { \mathrm { s h a r e d } } ( i ) = K \mathcal { L } _ { \mathrm { e a r l y } } ( i ) + \underbrace { \sum _ { s = 1 } ^ { K } \sum _ { j } { R _ { i } ^ { ( s ) } ( j ) \log R _ { i } ^ { ( s ) } ( j ) } } _ { \mathrm { c o n s t a n t w } . \mathrm { t } . \mathrm { ~ } P _ { i } } .\tag{66}
$$

Thus, the shared- ${ \mathbf { \nabla } } \cdot P _ { i }$ objective and the GraphMSL-style early-fusion objective differ only by a scalar factor $K$ and a term independent of the encoder parameters.

Equivalently, the same relation can be written in KL form:

$$
\left| \sum _ { s = 1 } ^ { K } \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } \Big ) = K \mathrm { K L } \big ( \bar { R } _ { i } \Big \| P _ { i } \big ) + \sum _ { s = 1 } ^ { K } \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| \bar { R } _ { i } \Big ) \right| .\tag{67}
$$

The second term is independent of $P _ { i }$ , and hence independent of the encoder parameters. The factor K only rescales the fused-target KL objective and does not change its minimizer. Hence,

$$
\arg \operatorname* { m i n } _ { \theta } \sum _ { s = 1 } ^ { K } \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } ^ { \theta } \Big ) = \arg \operatorname* { m i n } _ { \theta } \mathrm { K L } \big ( \bar { R } _ { i } \big \| P _ { i } ^ { \theta } \big ) .\tag{68}
$$

We can also verify the same equivalence by comparing gradients. For one prior space s,

$$
\mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } \Big ) = \sum _ { \ell } R _ { i } ^ { ( s ) } ( \ell ) \log R _ { i } ^ { ( s ) } ( \ell ) - \sum _ { \ell } R _ { i } ^ { ( s ) } ( \ell ) \log P _ { i } ( \ell ) .
$$

Using the softmax identity

$$
\frac { \partial \log P _ { i } ( \ell ) } { \partial u _ { i j } } = { \bf 1 } [ \ell = j ] - P _ { i } ( j ) ,
$$

we obtain

$$
\frac { \partial } { \partial u _ { i j } } \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } \Big ) = P _ { i } ( j ) - R _ { i } ^ { ( s ) } ( j ) .\tag{69}
$$

Summing over all prior spaces gives

$$
\frac { \partial \mathcal { L } _ { \mathrm { s h a r e d } } ( i ) } { \partial u _ { i j } } = \sum _ { s = 1 } ^ { K } \left( P _ { i } ( j ) - R _ { i } ^ { ( s ) } ( j ) \right) = K \left( P _ { i } ( j ) - \hat { R } _ { i } ( j ) \right) .\tag{70}
$$

For the GraphMSL-style early-fusion objective,

$$
\mathcal { L } _ { \mathrm { e a r l y } } ( i ) = - \sum _ { \ell } \bar { R } _ { i } ( \ell ) \log P _ { i } ( \ell ) ,
$$

the gradient is

$$
\frac { \partial \mathcal { L } _ { \mathrm { e a r l y } } ( i ) } { \partial u _ { i j } } = P _ { i } ( j ) - \bar { R } _ { i } ( j ) .\tag{71}
$$

Therefore,

$$
\boxed { \frac { \partial \mathcal { L } _ { \mathrm { s h a r e d } } ( i ) } { \partial u _ { i j } } = K \frac { \partial \mathcal { L } _ { \mathrm { e a r l y } } ( i ) } { \partial u _ { i j } } } .\tag{72}
$$

Thus, the unweighted shared-P<sub>i</sub> multi-prior objective and the GraphMSL-style early-fusion objective induce the same optimization direction, differing only by the scalar factor $\dot { K }$ . This scalar changes the loss scale but not the target relational geometry being learned.

By contrast, multi-branch PRISM does not optimize all prior spaces against the same $P _ { i } .$ . Instead, each prior space s has its own embedding-induced distribution $P _ { i } ^ { ( s ) }$ , and PRISM optimizes

$$
\mathcal { L } _ { \mathrm { P R I S M } } ( i ) = \sum _ { s = 1 } ^ { K } \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } ^ { ( s ) } \Big ) .
$$

For each branch, the gradient is

$$
\frac { \partial } { \partial u _ { i j } ^ { ( s ) } } \mathrm { K L } \Big ( R _ { i } ^ { ( s ) } \Big \| P _ { i } ^ { ( s ) } \Big ) = P _ { i } ^ { ( s ) } ( j ) - R _ { i } ^ { ( s ) } ( j ) .
$$

These are separate prior-specific gradient signals. They cannot be collapsed into one averaged signal $P _ { i } ( j ) - \bar { R } _ { i } ( j )$ , because the learner distributions $P _ { i } ^ { ( { \bar { s } } ) }$ are different across prior branches. Hence, GraphMSL-style early fusion or a shared-P<sub>i</sub> multi-prior objective preserves only the fused relational geometry ${ \bar { R } } _ { i }$ , whereas multi-branch PRISM preserves the prior-specific geometries $\{ R _ { i } ^ { ( s ) } \} _ { s = } ^ { K }$ before downstream fusion.

## F.3 Label Energy as a Prediction-Theoretic Upper Bound

Proposition 1 (Label energy upper-bounds prior-space kNN risk). Fix a prior space s and its neighbor sets $\mathcal { N } _ { k } ^ { ( s ) } ( i )$

Regression. For real-valued labels $y _ { i } \in \mathbb { R } ,$ , define the prior-space kNN regressor

$$
\widehat { y } _ { i } = \frac { 1 } { k } \sum _ { j \in \mathcal { N } _ { k } ^ { ( s ) } ( i ) } y _ { j } .\tag{73}
$$

Then

$$
\frac { 1 } { | Q | } \sum _ { i \in Q } ( y _ { i } - \widehat { y } _ { i } ) ^ { 2 } \leq { \mathcal E } _ { k } ^ { ( s ) } ( y ) , \qquad \mathrm { R M S E } ( \widehat { y } ) \leq \sqrt { { \mathcal E } _ { k } ^ { ( s ) } ( y ) } .\tag{74}
$$

Binary classification. For binary labels $y _ { i } \in \{ 0 , 1 \}$ and $k = 1$ , define the prior-space nearestneighbor classifier $\widehat { y } _ { i } = y _ { j ^ { \star } ( i ) }$ with $j ^ { \star } ( i ) \in \mathcal { N } _ { 1 } ^ { ( s ) } ( i )$ . Then

$$
\frac { 1 } { | Q | } \sum _ { i \in Q } \mathbf { 1 } \{ \widehat { y } _ { i } \neq y _ { i } \} = \mathcal { E } _ { 1 } ^ { ( s ) } ( y ) .\tag{75}
$$

For kNN majority vote with odd $k ,$ the misclassification rate satisfies

$$
\frac { 1 } { | Q | } \sum _ { i \in Q } \mathbf { 1 } \{ \widehat { y } _ { i } \neq y _ { i } \} \leq 2 \mathcal { E } _ { k } ^ { ( s ) } ( y ) .\tag{76}
$$

Proof. Regression. By definition,

$$
y _ { i } - \widehat { y } _ { i } = \frac { 1 } { k } \sum _ { j \in \mathcal { N } _ { k } ^ { ( s ) } ( i ) } ( y _ { i } - y _ { j } ) .\tag{77}
$$

By Jensen’s inequality applied to the squared function,

$$
( y _ { i } - \widehat { y } _ { i } ) ^ { 2 } \leq \frac { 1 } { k } \sum _ { j \in \mathcal { N } _ { k } ^ { ( s ) } ( i ) } ( y _ { i } - y _ { j } ) ^ { 2 } .\tag{78}
$$

Averaging over $i \in Q$ gives the regression bound. Taking the square root yields the RMSE bound. Binary classification, $k = 1$ . For y<sub>i</sub>, $y _ { j \star ( i ) } \in \{ 0 , 1 \}$ ,

$$
\mathbf { 1 } \{ y _ { i } \neq y _ { j ^ { \star } ( i ) } \} = ( y _ { i } - y _ { j ^ { \star } ( i ) } ) ^ { 2 } .\tag{79}
$$

Averaging over $i \in Q$ yields Eq. (75).

Binary classification, kNN majority vote. If the majority vote disagrees with $y _ { i }$ , then at least half of the k neighbors must have labels different from y<sub>i</sub>:

$$
\mathbf { 1 } \{ \widehat { y } _ { i } \neq y _ { i } \} \leq 2 \cdot \frac { 1 } { k } \sum _ { j \in \mathcal { N } _ { k } ^ { ( s ) } ( i ) } \mathbf { 1 } \{ y _ { j } \neq y _ { i } \} .\tag{80}
$$

Using ${ \bf 1 } \{ y _ { j } \neq y _ { i } \} = ( y _ { i } - y _ { j } ) ^ { 2 }$ for binary labels and averaging over $i \in Q$ yields Eq. (76). □

Interpretation. Proposition 1 shows that low label energy implies the existence of a low-error local predictor operating directly in the prior space. Label energy is therefore a prediction-theoretic diagnostic of prior–task compatibility. For regression, it directly controls the RMSE of a prior-space kNN regressor. For binary classification, it controls nearest-neighbor disagreement in the prior space. Although our classification metric is ROC-AUC rather than 0–1 error, low local disagreement is a useful indicator that the prior geometry separates labels in a way that can support downstream classification.

## F.4 No-Worse Approximation on a Finite Set

We finally give a finite-set justification for preserving prior information in learned representations. The purpose is not to prove that PRISM always improves generalization, but to clarify why retaining recoverable prior information should not reduce empirical predictive power when the downstream predictor class is sufficiently expressive.

Let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ be a finite task dataset. For prior space $s ,$ let $\pi _ { i } ^ { ( s ) }$ denote the prior feature and $r _ { i } ^ { ( s ) }$ the learned representation produced by the corresponding branch.

Proposition 2 (No-worse approximation on a finite set). Assume there exists a recovery map f<sub>rec</sub> (s) such that

$$
\pi _ { i } ^ { ( s ) } = f _ { \mathrm { r e c } } ^ { ( s ) } ( r _ { i } ^ { ( s ) } ) \quad f o r a l l i = 1 , \dots , n .\tag{81}
$$

Let $\mathcal { F }$ be a class of predictors on prior features and  a class of predictors on learned representations. Define the empirical squared errors

$$
\widehat { e } _ { \pi } = \operatorname* { i n f } _ { f \in \mathcal { F } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - f ( \pi _ { i } ^ { ( s ) } ) ) ^ { 2 } , \qquad \widehat { e } _ { r } = \operatorname* { i n f } _ { h \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - h ( r _ { i } ^ { ( s ) } ) ) ^ { 2 } .\tag{82}
$$

$$
I f \left\{ f \circ f _ { \mathrm { r e c } } ^ { ( s ) } : f \in \mathscr { F } \right\} \subseteq \mathscr { H } , t h e n \widehat { e } _ { r } \leq \widehat { e } _ { \pi } .
$$

Proof. For any $f \in { \mathcal { F } }$ , let $h _ { f } ( r ) = f ( f _ { \mathrm { r e c } } ^ { ( s ) } ( r ) )$ . By the predictor-class assumption, $h _ { f } \in \mathcal { H }$ . By the recovery assumption, $h _ { f } ( r _ { i } ^ { ( s ) } ) = f ( \pi _ { i } ^ { ( s ) } )$ for all i. Therefore the empirical error of $h _ { f }$ on $\mathcal { D }$ equals the empirical error of f on . Taking infima on both sides yields $\widehat { e } _ { r } \leq \widehat { e } _ { \pi }$ □

Interpretation. Proposition 2 states that if prior features are recoverable from learned representations on a finite dataset, then any predictor that could have been built from prior features can also be represented as a predictor on the learned representations, provided the downstream predictor class is sufficiently expressive. Therefore, preserving prior information need not make the representation empirically worse than using prior features directly. This is a finite-set approximation argument, not a generalization theorem. It depends on the recoverability assumption and on the expressiveness of the downstream predictor class.

## G Additional Results

This section provides additional results that complement the main tables in Section 4. We include the extended MoleculeNet comparison, pure-solvent OOD slice analysis, and temperature-dependent behavior analysis. We avoid repeating main-text tables unless additional decomposition or diagnostic information is provided. Mixed-solvent architectural ablations are reported separately in Appendix H.6, because they test alternative prediction heads rather than additional main-result settings.

## G.1 Extended MoleculeNet Comparison

The main text reports a compact subset of representative baselines. Table 23 provides an extended MoleculeNet comparison, including traditional feature-based models, supervised GNNs, 2D pretrained molecular models, geometry-aware pretrained models, and PRISM controls. We include HIV where matched results are available in our runs; external baselines without matched HIV results under our protocol are marked as not reported rather than imputed.

We emphasize three details about this comparison. First, the default PRISM model uses only 2D molecular graphs and chemically interpretable 2D prior spaces. Geometry-aware baselines are included for completeness but are grouped separately. Second, the reported GROVER\_large values are taken from the standardized KANO benchmark rather than directly from the original GROVER paper. This avoids mixing incompatible evaluation protocols and addresses known sensitivity of MoleculeNet scaffold splits. Third, HIV is included as a classification task, but not every external baseline has a matched HIV value under the same evaluation protocol; we therefore report available HIV values explicitly and avoid filling missing entries with incompatible numbers.

Table 23: Extended MoleculeNet comparison on classification tasks (ROC-AUC %, ) and regression tasks (RMSE, ). Bold indicates the best result among reported values in each column. PRISM uses only 2D molecular graphs; geometry-aware baselines are reported for completeness and grouped separately. Dashes indicate that a matched value under our protocol is not reported.
<table><tr><td></td><td colspan="3">Classification (ROC-AUC %, ↑)</td><td colspan="3">Regression (RMSE, ↓)</td></tr><tr><td>Dataset # Molecules</td><td>BBBP 2,039</td><td>BACE 1,513</td><td>HIV 41,127</td><td>ESOL 1,128</td><td>FreeSolv 642</td><td>Lipo 4,200</td></tr><tr><td colspan="7"></td></tr><tr><td>Traditional ML (fingerprints + VSA) MLP</td><td> $8 5 . 1 \pm 1 . 6$ </td><td> $8 5 . 2 \pm 0 . 8$ </td><td> $7 0 . 0 \pm 1 . 7$ </td><td> $1 . 4 6 1 \pm 0 . 0 6 0$ </td><td> $4 . 9 4 2 \pm 0 . 0 8 4$ </td><td> $0 . 8 7 8 \pm 0 . 0 1 6$ </td></tr><tr><td>XGBoost</td><td> $8 4 . 7 \pm 0 . 0$ </td><td> $8 8 . 5 \pm 0 . 0$ </td><td> $7 4 . 7 \pm 0 . 8$ </td><td> $0 . 9 2 5 \pm 0 . 0 0 0$ </td><td> $3 . 3 9 8 \pm 0 . 0 0 0$ </td><td> $0 . 7 4 1 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="7">Supervised GNNs (without pretraining)</td></tr><tr><td>GIN</td><td> $6 5 . 8 \pm 4 . { \bar { 5 } }$ </td><td> $7 0 . 1 \pm 5 . 4$ </td><td> $7 5 . 3 \pm 1 . 9$ </td><td> $1 . 4 5 2 \pm 0 . 0 2 0$ </td><td> $2 . 7 6 5 \pm 0 . 1 8 0$ </td><td> $0 . 8 5 0 \pm 0 . 0 7 1$ </td></tr><tr><td>D-MPNN</td><td> $9 1 . 9 \pm 3 . 0$ </td><td> $8 5 . 2 \pm 0 . 6$ </td><td> $7 7 . 1 \pm 0 . 5$ </td><td> $1 . 0 5 0 \pm 0 . 0 0 8$ </td><td> $1 . 6 7 3 \pm 0 . 0 8 2$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 6$ </td></tr><tr><td>Attentive FP</td><td> $6 4 . 3 \pm { 1 . 8 }$ </td><td> $7 8 . 4 \pm 0 . 0$ </td><td> $7 5 . 8 \pm 1 . 5$ </td><td> $0 . 8 7 7 \pm 0 . 0 2 9$ </td><td> $2 . 0 7 3 \pm 0 . 1 8 3$ </td><td> $0 . 7 2 1 \pm 0 . 0 0 1$ </td></tr><tr><td>D-MPNN w. Prior</td><td> $9 0 . 6 \pm 0 . 1$ </td><td> $8 7 . 7 \pm 0 . 2$ </td><td> $7 3 . 0 \pm 0 . 2$ </td><td> $0 . 7 9 8 \pm 0 . 0 0 3$ </td><td> $2 . 6 6 5 \pm 0 . 0 8 4$ </td><td> $0 . 6 4 5 \pm 0 . 0 0 4$ </td></tr><tr><td colspan="7">Pretrained models (2D-only)</td></tr><tr><td>PretrainGNN</td><td> $6 8 . 7 \pm 1 . 3$ </td><td> $8 4 . 5 \pm 0 . 7$ </td><td> $7 9 . 5 \pm 0 . 7$ </td><td> $1 . 1 0 0 \pm 0 . 0 0 6$ </td><td> $2 . 7 6 4 \pm 0 . 0 0 2$ </td><td> $0 . 7 3 9 \pm 0 . 0 0 3$ </td></tr><tr><td>GROVERlarge MolCLR</td><td> $8 6 . 8 \pm 2 . 2$   $7 3 . 3 \pm 1 . 0$ </td><td> $8 2 . 4 \pm 3 . 6$   $8 2 . 8 \pm 0 . 7$ </td><td> $6 8 . 2 \pm 1 . 1$   $7 7 . 4 \pm 0 . 6$ </td><td> $1 . 4 2 3 \pm 0 . 2 8 8$   $1 . 1 1 3 \pm 0 . 0 2 3$ </td><td> $2 . 9 4 7 \pm 0 . 6 1 5$   $2 . 3 0 1 \pm 0 . 2 4 7$ </td><td> $0 . 8 2 3 \pm 0 . 0 1 0$ </td></tr><tr><td>MoAMa</td><td> $7 2 . 8 \pm 0 . 9$ </td><td> $8 0 . 1 \pm 0 . 5$ </td><td> $7 9 . 3 \pm 0 . 6$ </td><td> $1 . 3 9 4 \pm 0 . 1 2 0$ </td><td> $2 . 7 0 0 \pm 0 . 1 1 0$ </td><td> $0 . 7 8 9 \pm 0 . 0 0 9$   $0 . 7 7 2 \pm 0 . 0 3 0$ </td></tr><tr><td> $\mathrm { M o l e c u l e S D E _ { G I N } }$ </td><td> $6 7 . 4 \pm 1 . 2$ </td><td> $7 8 . 1 \pm 4 . 1$ </td><td> $7 6 . 5 \pm 0 . 8$ </td><td> $2 . 0 8 0 \pm 0 . 1 9 0$ </td><td> $3 . 6 4 0 \pm 0 . 1 7 0$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 6 1 0 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="7">Pretrained models (3D / geometry-aware)</td></tr><tr><td>3D-Infomax</td><td> $6 9 . 1 \pm 1 . 1$ </td><td> $7 9 . 4 \pm 1 . 9$ </td><td> $7 5 . 9 \pm 0 . 5 9$ </td><td> $0 . 8 9 4 \pm 0 . 0 2 8$ </td><td> $2 . 3 3 7 \pm 0 . 2 2 7$ </td><td> $0 . 7 3 9 \pm 0 . 0 0 9$ </td></tr><tr><td>GEM</td><td> $8 8 . 8 \pm 0 . 4$ </td><td> $8 7 . 9 \pm 1 . 1$ </td><td> ${ \bf 8 1 . 3 \pm 0 . 3 }$ </td><td> $0 . 8 1 3 \pm 0 . 0 2 8$ </td><td> $1 . 7 4 8 \pm 0 . 1 1 4$ </td><td> $0 . 6 7 4 \pm 0 . 0 2 2$ </td></tr><tr><td>Uni-Mol</td><td> $7 2 . 9 \pm 0 . 6$ </td><td> $8 5 . 7 \pm 0 . 2$ </td><td> $7 9 . 6 \pm 0 . 5$ </td><td> $0 . 7 8 8 \pm 0 . 0 2 9$ </td><td> ${ \bf 1 . 6 2 0 \pm 0 . 0 3 5 }$ </td><td> $0 . 6 0 3 \pm 0 . 0 1 0$ </td></tr><tr><td>PRISMrandom</td><td> $9 3 . 0 \pm 0 . 6$ </td><td> $8 7 . 8 \pm { 1 . 7 }$ </td><td> $7 5 . 1 \pm 0 . 5$ </td><td> $0 . 8 4 8 \pm 0 . 0 2 5$ </td><td> $2 . 2 2 2 \pm 0 . 0 5 4$ </td><td></td></tr><tr><td>D-MPNN (reconstruction)</td><td> $9 4 . 1 \pm 0 . 6 $ </td><td> $7 1 . 1 \pm 6 . 7$ </td><td> $7 3 . 4 \pm 0 . 9$ </td><td> $0 . 8 2 9 \pm 0 . 0 7 6$ </td><td> $2 . 3 0 8 \pm 0 . 0 7 5$ </td><td> $0 . 6 4 2 \pm 0 . 0 1 4$ </td></tr><tr><td>PRISM</td><td> ${ \bf 9 4 . 6 \pm 0 . 3 }$ </td><td> ${ \bf 8 9 . 4 \pm 0 . 4 }$ </td><td> $7 7 . 7 \pm 0 . 6$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 2 }$ </td><td></td><td> $0 . 5 7 7 \pm 0 . 0 1 9$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $1 . 7 7 0 \pm 0 . 0 9 3$ </td><td> ${ \bf 0 . 5 3 8 \pm 0 . 0 1 8 }$ </td></tr></table>

Discussion. PRISM achieves the best reported performance on BBBP, BACE, ESOL, and Lipophilicity, and remains competitive on HIV and FreeSolv. The comparison with PRISM<sub>random</sub> isolates the effect of PRISM initialization under the same four-branch architecture and downstream heads. The weaker relative performance on HIV and FreeSolv reflects two different limitations. HIV has weaker alignment with the default VSA/fingerprint priors, although Appendix H.9 shows that selecting a Pharm2D prior using label energy improves HIV performance. FreeSolv shows that low local label energy alone does not guarantee an advantage over a strong supervised baseline when task size, baseline strength, and missing task-specific factors such as 3D hydration geometry matter.

Table 24: OOD slice analysis on unseen-solute tests. Slices are defined relative to the complete processed BigSolDB dataset using canonical isomeric SMILES. Temperature slices use the solvent-specific temperature range and apply only to seen solvents. For Leeds, the seen-solvent and temperature-in-range slices coincide with the overall set. Metrics are RMSE ( ) and pct-in-1 ( ), the percentage of predictions within one log S unit of the measured value. Differences are PRISM minus Fastprop; differences in pct-in-1 are reported in percentage points (pp).
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Slice</td><td rowspan="2">n</td><td colspan="2">Fastprop</td><td colspan="2">PRISM</td><td colspan="2">∆ (PRISM-Fastprop)</td></tr><tr><td>RMSE↓</td><td>pct-in-1 (%) ↑</td><td>RMSE↓</td><td>pct-in-1 (%) ↑</td><td>RMSE</td><td>pct-in-1 (pp)</td></tr><tr><td colspan="9">Leeds</td></tr><tr><td></td><td>All solutes are unseen; all solvents are seen; all temperatures are in-range.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Leeds</td><td>Overall</td><td>1,294</td><td>0.954</td><td>73.57</td><td>0.909</td><td>75.43</td><td>-0.045</td><td>+1.85</td></tr><tr><td colspan="9">SolProp</td></tr><tr><td>SolProp</td><td>Overall</td><td>5,230</td><td>0.833</td><td>78.07</td><td>0.773</td><td>82.28</td><td>-0.060</td><td>+4.21</td></tr><tr><td>SolProp</td><td>Solvent: Seen</td><td>4,988</td><td>0.835</td><td>77.83</td><td>0.772</td><td>82.40</td><td>-0.062</td><td>+4.57</td></tr><tr><td>SolProp</td><td>Solvent: Unseen</td><td>242</td><td>0.795</td><td>83.06</td><td>0.781</td><td>79.75</td><td>-0.014</td><td>-3.31</td></tr><tr><td>SolProp</td><td>Temp. (solvent-conditioned): In-range</td><td>4,791</td><td>0.840</td><td>77.46</td><td>0.767</td><td>82.84</td><td>-0.073</td><td>+5.39</td></tr><tr><td>SolProp</td><td>Temp. (solvent-conditioned): Extrapolation</td><td>197</td><td>0.693</td><td>86.80</td><td>0.894</td><td>71.57</td><td>+0.201</td><td>-15.23</td></tr></table>

## G.2 Pure-Solvent OOD Slice Analysis

Aggregate external-test metrics do not distinguish between different sources of distribution shift. We therefore decompose the Leeds and SolProp results by solvent identity and solvent-conditioned temperature range. All test solutes are absent from the processed BigSolDB dataset. We use this complete dataset, including its training and validation partitions, as the reference for defining the slices. Solute and solvent identities are matched using canonical isomeric SMILES.

A test solvent is classified as seen if its canonical SMILES appears in the BigSolDB reference set, and unseen otherwise. All three Leeds solvents—acetone, benzene, and ethanol—are seen under this definition. SolProp contains both seen and unseen solvents.

For each seen solvent, we define the reference temperature range by the minimum and maximum temperatures observed for that solvent in BigSolDB. A test measurement is classified as temperature in-range if its temperature lies within this inclusive interval, and as temperature extrapolation otherwise. Temperature slices are defined only for seen solvents. All Leeds measurements are temperature in-range, so its overall, seen-solvent, and temperature-in-range slices contain the same records.

Table 24 reports the resulting comparisons. All 1,294 Leeds records and 5,230 SolProp records are retained. Repeated measurements at the same solute–solvent–temperature conditions remain separate records. The reported metrics summarize the fixed prediction sets used for this analysis; seed-level uncertainty is not reported.

Discussion. PRISM achieves lower overall RMSE and higher pct-in-1 on both external datasets. On SolProp, it improves both metrics for seen solvents and for temperatures within the solvent-specific reference range. The results are mixed for unseen solvents: PRISM reduces RMSE from 0.795 to 0.781, but pct-in-1 decreases from 83.06% to 79.75%. Temperature extrapolation remains a limitation. On the 197 measurements outside the corresponding solvent-specific temperature ranges, PRISM has higher RMSE than Fastprop (0.894 versus 0.693) and lower pct-in-1 (71.57% versus 86.80%). These results support improved aggregate external-test performance while showing that the gains do not extend uniformly across solvent and temperature shifts.

## G.3 Temperature-Dependent Behavior and Normalized EMD Analysis

Solubility prediction is not only a pointwise regression problem. For temperature-dependent measurements, a useful model should also reproduce the empirical distribution of temperature sensitivity. We therefore compare the distribution of finite-difference temperature gradients

$$
\frac { d \log S } { d T }\tag{83}
$$

computed from experimental measurements and model predictions. This analysis supports the main-text claim that PRISM better matches the empirical temperature-gradient distribution, with normalized EMD 0.01 compared with 0.02 for Fastprop.

Gradient construction. For each solute–solvent pair with at least two measurements, we compute numerical derivatives using numpy.gradient, supplying the recorded temperatures as the coordinate array. Let $\mathbf { T } = ( T _ { 1 } , \dots , T _ { n } ) , \mathbf { y } = ( \log S _ { 1 } , \dots , \log S _ { n } )$ , and ${ \widehat { \mathbf { y } } } = ( \widehat { \log S } _ { 1 } , \ldots , \widehat { \log S } _ { n } )$ denote the temperatures, experimental values, and model predictions at the same measurement conditions. The experimental gradients are

$$
\mathbf { g } ^ { \mathrm { e x p } } = \mathrm { g r a d i e n t } ( \mathbf { y } , \mathbf { T } ) .\tag{84}
$$

For each model, we apply the same operator to its predictions:

$$
\mathbf { g } ^ { \mathrm { m o d e l } } = \mathrm { g r a d i e n t } ( \widehat { \mathbf { y } } , \mathbf { T } ) .\tag{85}
$$

Here, gradient uses centered finite differences at interior points and first-order one-sided differences at the boundaries, with the supplied temperature coordinates accounting for nonuniform spacing. We retain entries only when both the experimental and predicted gradients are finite.

Distribution construction and normalized EMD. We aggregate gradients across all eligible solute–solvent temperature series in the evaluation set. For the CDFs and metric computation, we construct histograms using $M \ = \ 2 0 0$ equally spaced bins over the fixed gradient interval $[ - 0 . 0 5 , 0 . 0 5 ]$ . Gradients outside this interval are excluded from the respective histograms, and each histogram is normalized by its retained count to obtain a probability mass function. Panel (e) uses 120 bins over the same interval to visualize the counts. Let $p \in \mathbf { \mathbb { R } } ^ { M }$ be the empirical gradient distribution and $q \in \mathbb { R } ^ { M }$ be the model-gradient distribution.

For one-dimensional histograms with ordered bins, Earth Mover’s Distance can be computed from cumulative distribution functions. Let

$$
F _ { p } ( b ) = \sum _ { m = 1 } ^ { b } p _ { m } , \qquad F _ { q } ( b ) = \sum _ { m = 1 } ^ { b } q _ { m } .\tag{86}
$$

We report normalized EMD, defined as the mean absolute difference between the two CDF arrays:

$$
\mathrm { E M D } _ { \mathrm { n o r m } } ( p , q ) = \frac { 1 } { M } \sum _ { b = 1 } ^ { M } | F _ { p } ( b ) - F _ { q } ( b ) | .\tag{87}
$$

Because both CDF arrays are nondecreasing, this expression is exactly the value returned by applying wasserstein\_distance to the two arrays. For equal-width bins with interval length $L = 0 .$ 1 and bin width $\Delta = L / M$ , it is also equivalent to the histogram EMD $\begin{array} { r } { \sum _ { b } | F _ { p } ( b ) - F _ { q } ( b ) | \Delta } \end{array}$ divided by L. The reported metric is therefore dimensionless. Lower normalized EMD indicates closer agreement between the experimental and predicted gradient distributions within the analyzed interval.

Comparison with Fastprop. Fastprop explicitly incorporates a temperature-gradient loss on d log $S / d T$ , whereas PRISM does not impose such a thermodynamic derivative constraint. Despite this, PRISM obtains a lower normalized EMD between the predicted and experimental gradient distributions. This suggests that prior-induced molecular representations, combined with solute– solvent relation modeling, help capture temperature-dependent solubility trends without directly optimizing a gradient-matching objective.

Caveat. This distribution-level result should not be interpreted as solving all forms of temperature extrapolation. As shown in Appendix G.2, the small solvent-conditioned temperature-extrapolation subset in SolProp remains difficult. The normalized EMD analysis supports a weaker and more specific claim: across the evaluated external-test temperature intervals, PRISM reproduces the aggregate distribution of temperature sensitivity more faithfully than the baseline within the analyzed gradient range.

![](images/d7be1c25feabbb75dc8ff56d792804c9ec2d268dcfd237b1fd637924a3fa3205.jpg)  
Figure 6: Pure-solvent external-test behavior and temperature-gradient analysis. Panels (a,b) show PRISM predictions on Leeds and SolProp, together with error distributions for seen/unseen solvent slices. Panels (c,d) show the corresponding Fastprop predictions. Panels (e,f) compare the empirical and model-induced finite-difference temperature-gradient distributions. PRISM achieves lower normalized EMD to the empirical d log $S / d T$ distribution than Fastprop (0.01 vs. 0.02), despite not using an explicit temperature-gradient loss.

## G.4 Independent Pure/Binary/Ternary Mixture Evaluation

To evaluate whether PRISM representations remain useful beyond the binary-solvent benchmark in Section 3, we additionally use the independent dataset of Cenci et al. [34]. The dataset contains solubility measurements for one active pharmaceutical ingredient (API) in 14 organic solvents and includes pure solvents, binary mixtures, and ternary mixtures at multiple temperatures. Following the original calibration–validation protocol, 176 pure- and binary-solvent measurements are used for calibration, while the 288-sample validation set contains pure, binary, and ternary solvent systems under new compositions and temperatures.

Because this dataset contains only one solute and a relatively small calibration set, we use it as a representation-reuse test rather than a new end-to-end molecular-learning benchmark. We freeze the PRISM-pretrained encoders, construct composition-weighted solvent representations, and fit a low-capacity quadratic Ridge regressor on the calibration data. We compare this model with the UNIFAC–PLS approach used by Cenci et al., and additionally evaluate a hybrid model combining UNIFAC and PRISM information.

Table 25: Independent pure-, binary-, and ternary-solvent evaluation on the Cenci et al. dataset. The original calibration–validation split is retained. Lower log-RMSE and higher $R ^ { 2 }$ indicate better performance.
<table><tr><td>Model</td><td>Overall log-RMSE↓</td><td>Overall  $R ^ { 2 } \uparrow$ </td><td>Ternary log-RMSE ↓</td></tr><tr><td>UNIFAC-PLS</td><td>0.3061</td><td>0.9042</td><td>0.0921</td></tr><tr><td>PRISM-Ridge</td><td>0.2738</td><td>0.9217</td><td>0.1954</td></tr><tr><td>UNIFAC + PRISM</td><td>0.2502</td><td>0.9409</td><td>0.0904</td></tr></table>

PRISM–Ridge improves the overall validation performance relative to UNIFAC–PLS, reducing log-RMSE from 0.3061 to 0.2738 and increasing $\dot { R } ^ { 2 }$ from 0.9042 to 0.9217. However, PRISM alone is weaker on the small ternary subset, where UNIFAC–PLS achieves a lower log-RMSE. Combining UNIFAC and PRISM yields the best overall performance and also slightly improves the ternary result. These results suggest that PRISM representations can be reused beyond binary mixtures and provide information complementary to a specialized physicochemical mixture model, while not replacing such domain-specific modeling in every regime.

Table 26: Unlabeled target-domain adaptation for pure-solvent external transfer. “Target” denotes and $\mathbf { \ddot { j } o i n t } ^ { \prime \prime }$ denotes $\mathcal { U } = \mathcal { Z } \cup \mathcal { T } .$ . Epoch counts, where shown, refer to the relationalpretraining stage. Results are mean standard deviation over three seeds. The metric pct-in-1 is the percentage of predictions within one log-solubility unit of the experimental value.
<table><tr><td>Protocol</td><td></td><td></td><td></td><td>Leeds RMSE ↓ SolProp RMSE ↓ Leeds pct-in-1 (%) ↑ SolProp pct-in-1 (%) ↑</td></tr><tr><td>ZINC-only pretraining (100 epochs)</td><td> $0 . 9 1 0 \pm 0 . 0 0 2$ </td><td> $0 . 7 7 0 \pm 0 . 0 3 0$ </td><td> $7 5 . 7 \pm 0 . 9$ </td><td> $8 2 . 3 \pm 0 . 8$ </td></tr><tr><td>Target-only adaptation</td><td> $0 . 9 2 0 \pm 0 . 0 0 2$ </td><td> $0 . 7 4 0 \pm 0 . 0 4 1$ </td><td> $7 4 . 1 \pm 0 . 7$ </td><td> $8 2 . 5 \pm 0 . 6$ </td></tr><tr><td>Target anchors, joint neighbors (100 epochs)</td><td> $0 . 9 1 0 \pm 0 . 0 0 2$ </td><td> $0 . 7 7 5 \pm 0 . 0 4 0$ </td><td> $7 5 . 6 \pm 0 . 6$ </td><td> $8 2 . 4 \pm 0 . 5$ </td></tr><tr><td>Target anchors, joint neighbors (300 epochs)</td><td> $0 . 9 4 1 \pm 0 . 0 0 2$ </td><td> $0 . 7 9 6 \pm 0 . 0 2 2$ </td><td> $7 3 . 1 \pm 0 . 6$ </td><td> $7 9 . 1 \pm 0 . 4$ </td></tr><tr><td>Joint anchors and neighbors (100 epochs)</td><td> $0 . 9 0 8 \pm 0 . 0 0 2$ </td><td> $0 . 7 6 6 \pm 0 . 0 3 0$ </td><td> $7 5 . 7 \pm 0 . 9$ </td><td> $8 2 . 5 \pm 0 . 6$ </td></tr></table>

## G.5 Unlabeled Target-Domain Adaptation

We investigate whether additional relational pretraining on unlabeled target-domain molecules improves pure-solvent external transfer. Let  denote the ZINC250k pretraining corpus and the set of unique solute and solvent molecules collected from BigSolDB, Leeds, and SolProp. We canonicalize molecular SMILES and remove duplicates when constructing the joint pool $\mathcal { U } = \mathcal { Z } \cup \mathcal { T }$ Only molecular structures and their computed prior features are used during relational pretraining; Leeds and SolProp solubility labels are not used for pretraining or supervised fine-tuning. These experiments therefore evaluate transductive adaptation separately from the default inductive protocol used in the main results.

Adaptation protocols. The default protocol uses ZINC250k molecules as both anchors and candidate neighbors. We first evaluate target-only adaptation, where both are drawn from . We then retain target-domain molecules as anchors while expanding the candidate-neighbor pool to , evaluating this setting at 100 and 300 epochs. Finally, we evaluate joint source–target pretraining, where both anchors and candidate neighbors are drawn from , for 100 epochs. In each case, relational pretraining is followed by the same supervised BigSolDB fine-tuning and external evaluation on Leeds and SolProp.

Discussion. The effect of additional adaptation depends on the protocol and evaluation dataset. Target-only adaptation improves SolProp RMSE but worsens Leeds performance. Expanding the candidate-neighbor pool restores Leeds RMSE to the default level at 100 epochs, without improving SolProp RMSE. Extending this target-anchor protocol to 300 epochs worsens performance on both datasets, indicating that longer adaptation is not necessarily beneficial in this setting.

Joint source–target pretraining yields small numerical improvements over the default protocol: RMSE decreases from 0.910 to 0.908 on Leeds and from 0.770 to 0.766 on SolProp. Leeds pct-in-1 remains unchanged, while SolProp pct-in-1 increases from 82.3% to 82.5%. Thus, although joint pretraining provides modest gains in this comparison, additional access to unlabeled target-domain molecules does not produce consistent improvements across the evaluated adaptation protocols. We retain the default inductive protocol for the main experiments, as it achieves comparable performance without using external-test molecules during pretraining.

## H Additional Ablation Studies

This section provides additional ablation studies that complement the main-text results without repeating tables already reported in Section 4. The main paper reports the mixed-solvent random and solvent-pair-disjoint results, the Gaussian-random-prior control on the primary MoleculeNet tasks, the solubility pretraining ablation, and the K $\mathcal { N } _ { 2 } /$ rank alignment-objective comparison. Here we include additional controls that are not fully shown in the main text: (i) the effect of PRISM initialization on MoleculeNet under the same PRISM architecture; (ii) the single-shared-encoder baseline on MoleculeNet; (iii) leave-one-prior-space-out ablations; (iv) task-specific versus instancewise weighted fusion; (v) mixed-solvent head ablations under the frozen-encoder protocol; (vi) extended prior-quality and dynamic prior-selection analyses on additional bioactivity tasks; (vii) compatibility with 3D encoders and 3D prior spaces; and (viii) the effect of pretraining-data scale.

Protocol note on pretraining hyperparameters. The sensitivity analyses in Appendix E.9.1 are used as a small validation-driven selection grid for PRISM pretraining hyperparameters. Thus, the main reported PRISM results use the task-specific settings listed in Table 14, rather than a single universal pretraining checkpoint for all tasks. The statement that $\tau = 0 . 1 , \gamma _ { s } = 0 . 2 5$ is a simple robust default should therefore be interpreted as a practical baseline setting that performs reasonably across several tasks, not as the exact setting used for every reported main result. All PRISM controls and ablations for a given task use the corresponding task-level setting unless explicitly stated otherwise. This avoids attributing gains to downstream heads alone, but it also means that comparisons against external baselines should be read as comparisons under each method’s reported model-selection protocol rather than as a strict equal-pretraining-grid comparison.

Protocol note on mixed-solvent baselines. The mixed-solvent PRISM experiments use original PRISM-pretrained encoders and initialize the fusion module from a pure-solvent checkpoint. The four end-to-end MPNN baselines in the main mixed-solvent table are trained on the mixed-solvent dataset under the reported mixed-solvent protocols and do not use the same ZINC-plus-BigSolDB pretraining path. Therefore, the mixed-solvent comparison is best interpreted as a representation-reuse probe: it tests whether PRISM-pretrained molecular representations can be reused under frozen encoders and lightweight composition-weighted pooling. It is not a strict equal-data-budget comparison between pretraining strategies. This distinction is important because the fusion initialization provides PRISM with additional task-relevant solute–solvent experience from BigSolDB.

## H.1 PRISM Initialization Control on MoleculeNet

We first isolate the effect of PRISM pretraining on MoleculeNet. Both models in Table 27 use the same four-branch PRISM architecture and the same downstream heads. The only difference is initialization: $\mathrm { P R I S M } _ { \mathrm { r a n d o m } }$ is trained from random initialization, whereas PRISM uses PRISMpretrained branches.

Table 27: Effect of PRISM pretraining on MoleculeNet. Same four-branch PRISM architecture with identical heads trained from random initialization $\left( \mathrm { P R I S M } _ { \mathrm { r a n d o m } } \right)$ versus PRISM-pretrained initialization (PRISM).
<table><tr><td></td><td colspan="3">Classification (ROC-AUC %, ↑)</td><td colspan="3">Regression (RMSE, ↓)</td></tr><tr><td>Init</td><td>BBBP</td><td>BACE</td><td>HIV</td><td>ESOL</td><td>FreeSolv</td><td>Lipo</td></tr><tr><td>PRISMrandom</td><td> $9 3 . 0 \pm 0 . 6$ </td><td> $8 7 . 8 \pm 1 . 7$ </td><td> $7 5 . 1 \pm 0 . 5$ </td><td> $0 . 8 4 8 \pm 0 . 0 2 5$ </td><td> $2 . 2 2 2 \pm 0 . 0 5 4$ </td><td> $0 . 6 4 2 \pm 0 . 0 1 4$ </td></tr><tr><td>PRISM</td><td> ${ \bf 9 4 . 6 \pm 0 . 3 }$ </td><td> ${ \bf 8 9 . 4 \pm 0 . 4 }$ </td><td> ${ \bf 7 7 . 7 \pm 0 . 6 }$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 1 . 7 7 0 \pm 0 . 0 9 3 }$ </td><td> $\mathbf { 0 . 5 3 8 \pm 0 . 0 1 8 }$ </td></tr><tr><td>∆ (PRISM − random)</td><td>+1.6</td><td>+1.6</td><td>+2.6</td><td>-0.076</td><td>-0.452</td><td>-0.104</td></tr></table>

Discussion. PRISM pretraining improves the same PRISM architecture on all six MoleculeNet tasks. This rules out the explanation that the gains come only from increasing model capacity through a multi-branch architecture.

## H.2 Compression diagnostic via PRISM-Distilled

Construction. PRISM-Distilled is a single-encoder variant trained from PRISM under the same alignment family (Section 2.2). The teacher is the multi-prior PRISM model with four pretrained branch encoders. For each molecule $x _ { i } ,$ we extract its concatenated multi-prior representation

$$
H _ { i } = [ h _ { i } ^ { ( s _ { 1 } ) } ; h _ { i } ^ { ( s _ { 2 } ) } ; h _ { i } ^ { ( s _ { 3 } ) } ; h _ { i } ^ { ( s _ { 4 } ) } ] \in \mathbb { R } ^ { K \times D }
$$

and treat $H _ { i }$ as a new “prior representation” that defines a teacher-induced relational structure over the molecular population. We then train a fresh single MPNN student encoder $G _ { \theta }$ from random initialization to align with this teacher relation under the same KL alignment objective in Eq. (4), with locality scale $\beta$ rescaled to the median pairwise distance of $\{ H _ { i } \}$ in the mini-batch. The student therefore inherits the relational geometry of the teacher (which molecules are close to which) rather than absolute coordinates, scales, or branch-specific information.

Pretraining and downstream protocol. Distillation is run on the same 250K ZINC subset used for PRISM pretraining, with batch size 512, embedding temperature $\tau = 0 . 1$ , and AdamW optimizer. For all downstream evaluations, we replace the PRISM encoder $E _ { \Theta }$ with the distilled student $G _ { \theta }$ and reuse the same task-specific heads and training protocols described in the main text (Section 2.3 and Appendix E). We test three downstream regimes: pure-solvent external transfer (Table 28), mixedsolvent prediction under both frozen and fine-tuned encoder settings (Table 29), and single-molecule downstream tasks with encoder fine-tuning (Table 30). All results are the mean std over three seeds.

Results. On pure-solvent external transfer (Table 28), distillation raises Leeds RMSE from 0.91 to 0.95 and SolProp RMSE from 0.77 to 0.97, while $\% | \Delta$ log $S | \le 1$ drops by under 6 points on both datasets. On mixed-solvent prediction (Table 29), the frozen distilled encoder reaches 0.123 0.013 on the random split and $0 . 4 \mathrm { \bar { 5 } 7 \pm 0 . 1 }$ 10 on the pair-disjoint split, compared with $0 . 0 9 5 \pm 0 . 0 0 8$ and $0 . 4 4 0 \pm 0 . 0 1 0$ for PRISM; the mean-error gap is small but the run-to-run standard deviation on the pair-disjoint split widens by roughly an order of magnitude. Fine-tuning the distilled encoder does not close this gap. On single-molecule downstream tasks (Table 30), distillation degradation is substantially larger: FreeSolv RMSE rises from 1.770 to 3.987, BACE ROC-AUC drops from 89.4 to 75.08, and BBBP from 94.6 to 85.80. The contrast across the three settings supports the reading in Section 4: solubility tasks survive relational compression because they depend on neighborhood structure that the student inherits, while single-molecule endpoints rely more heavily on coordinatelevel or branch-specific signals that purely relational distillation does not preserve.

Table 28: Pure-solvent external transfer with distilled vs. normal encoder. RMSE is lower better, $\% | \Delta$ log $S | \le 1$ is higher better.
<table><tr><td rowspan="3">Dataset</td><td colspan="2">Distilled encoder</td><td colspan="2">Normal encoder</td></tr><tr><td>RMSE↓</td><td> $\% | \Delta \log S | \le 1 \uparrow$ </td><td>RMSE↓</td><td> $\% | \Delta \log S | \le 1 \uparrow$ </td></tr><tr><td>Leeds</td><td> $0 . 9 5 \pm 0 . 0 1 3 6$ </td><td> $7 3 . 9 3 \pm 0 . 6 2$ </td><td> $0 . 9 1 \pm 0 . 0 0 2$ </td><td> $7 5 . 7 \pm 0 . 9$ </td></tr><tr><td>SolProp</td><td> $0 . 9 7 \pm 0 . 0 2 3 5$ </td><td> $7 7 . 2 4 \pm 1 . 1 1$ </td><td> $0 . 7 7 \pm 0 . 0 3 0$ </td><td> $8 2 . 3 \pm 0 . 8$ </td></tr></table>

Table 29: Mixed-solvent prediction with distilled vs. normal encoder, under random and pair-disjoint splits. RMSE is lower better, $\% | \Delta$ log $S | \le 1$ is higher better.
<table><tr><td>Encoder setting</td><td>Split</td><td>RMSE↓</td><td> $\% | \Delta \log S | \le 1 \uparrow$ </td></tr><tr><td>Frozen distilled encoder</td><td>Random</td><td> $0 . 1 2 3 \pm 0 . 0 1 3$ </td><td> $9 9 . 9 5 \pm 0 . 0 2$ </td></tr><tr><td>Frozen normal encoder</td><td>Random</td><td> $0 . 0 9 5 \pm 0 . 0 0 8$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td></tr><tr><td>Fine-tuned distilled encoder</td><td>Random</td><td> $0 . 1 3 2 1 \pm 0 . 0 1 5$ </td><td> $9 9 . 9 6 \pm 0 . 0 3$ </td></tr><tr><td>Frozen distilled encoder</td><td>Pair-disjoint</td><td> $0 . 4 5 7 \pm 0 . 1 1 0$ </td><td> $9 5 . 5 9 \pm 3 . 7 5$ </td></tr><tr><td>Frozen normal encoder</td><td>Pair-disjoint</td><td> $0 . 4 4 0 \pm 0 . 0 1 0$ </td><td> $9 6 . 6 \pm 0 . 9$ </td></tr><tr><td>Fine-tuned distilled encoder</td><td>Pair-disjoint</td><td> $0 . 4 6 2 \pm 0 . 0 2 6$ </td><td> $9 6 . 4 1 \pm 3 . 5 2$ </td></tr></table>

Table 30: Single-molecule downstream results with distilled vs. normal encoder, both fine-tuned. RMSE is lower better for regression, ROC-AUC is higher better for classification.
<table><tr><td>Dataset</td><td>Metric</td><td>Distilled encoder</td><td>Normal encoder</td></tr><tr><td>ESOL</td><td>RMSE↓</td><td> $0 . 9 0 7 2 \pm 0 . 0 1 2$ </td><td> $0 . 7 7 2 \pm 0 . 0 0 2$ </td></tr><tr><td>FreeSolv</td><td>RMSE↓</td><td> $3 . 9 8 7 \pm 0 . 1 0 6$ </td><td> $1 . 7 7 0 \pm 0 . 0 9$ </td></tr><tr><td>Lipo</td><td>RMSE↓</td><td> $0 . 8 8 8 \pm 0 . 0 3 5$ </td><td> $0 . 5 3 8 \pm 0 . 0 2$ </td></tr><tr><td>BACE</td><td>ROC-AUC ↑</td><td> $7 5 . 0 8 \pm 5 . 2$ </td><td> $8 9 . 4 \pm 0 . 4$ </td></tr><tr><td>BBBP</td><td>ROC-AUC ↑</td><td> $8 5 . 8 0 \pm 3 . 2$ </td><td> $9 4 . 6 \pm 0 . 3$ </td></tr></table>

## H.3 Single-Shared-Encoder Baseline on MoleculeNet

PRISM pretrains one encoder branch per prior space. A natural alternative is to use a single encoder and optimize it with the sum of all prior-alignment losses:

$$
\mathcal { L } _ { \mathrm { s h a r e d } } = \sum _ { s \in \mathcal { S } } \mathcal { L } _ { \mathrm { a l i g n } } ^ { ( s ) } .\tag{88}
$$

This baseline tests whether separate prior-specialized branches are necessary, or whether one embed ding space can absorb all prior geometries simultaneously.

Table 31 compares the single-shared-encoder baseline with the multi-branch PRISM design.

Table 31: Single shared encoder versus multi-branch PRISM on MoleculeNet. The sharedencoder baseline uses one D-MPNN trained with the sum of all prior-alignment losses. Classification metrics are ROC-AUC, and regression metrics are RMSE.
<table><tr><td>Model</td><td>BBBP↑</td><td>BACE↑</td><td>ESOL↓</td><td>FreeSolv ↓</td><td>Lipo ↓</td></tr><tr><td>Single shared encoder</td><td> $0 . 8 5 1 \pm 0 . 0 3 2$ </td><td> $0 . 8 6 0 \pm 0 . 0 2 0$ </td><td> $0 . 9 2 0 9 \pm 0 . 0 2 7$ </td><td> $2 . 0 4 0 \pm 0 . 1 9 7$ </td><td> $0 . 7 4 0 \pm 0 . 0 1 1$ </td></tr><tr><td>PRISM</td><td> $\mathbf { 0 . 9 4 6 \pm 0 . 0 0 3 }$ </td><td> $\mathbf { 0 . 8 9 4 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 1 . 7 7 0 \pm 0 . 0 9 3 }$ </td><td> $\mathbf { 0 . 5 3 8 \pm 0 . 0 1 8 }$ </td></tr></table>

Discussion. The shared-encoder baseline is consistently weaker than the multi-branch design. This suggests that heterogeneous prior geometries are difficult to absorb into a single embedding space through a simple summed alignment loss. Separate branches allow each encoder to specialize in one chemically meaningful relational view before downstream fusion.

## H.4 Leave-One-Prior-Space-Out Ablations

We evaluate whether each prior branch contributes to downstream performance. For each ablation, we remove one prior branch while keeping the same training and evaluation protocol. Tables 32 and 33 report leave-one-prior-space-out results on MoleculeNet classification and regression tasks. Table 34 reports the corresponding ablations on pure-solvent and mixed-solvent solubility tasks.

Table 32: Leave-one-prior-space-out ablations on MoleculeNet classification. AUC is reported in the standard [0, 1] scale.
<table><tr><td>Spaces used</td><td>BBBP</td><td>BACE</td><td>HIV</td></tr><tr><td>all</td><td> $0 . 9 4 5 6 \pm 0 . 0 0 3 1$ </td><td> $0 . 8 9 3 6 \pm 0 . 0 0 4 7$ </td><td> $0 . 7 7 6 8 \pm 0 . 0 0 9 2$ </td></tr><tr><td>w/o FP</td><td> $0 . 9 3 1 6 \pm 0 . 0 0 3 1$ </td><td> $0 . 8 8 4 6 \pm 0 . 0 0 1 9$ </td><td> $0 . 7 1 4 0 \pm 0 . 0 1 0 7$ </td></tr><tr><td>w/o PEOE</td><td> $0 . 9 3 8 0 \pm 0 . 0 0 5 2$ </td><td> $0 . 8 8 2 1 \pm 0 . 0 0 1 3$ </td><td> $0 . 7 2 4 9 \pm 0 . 0 0 7 9$ </td></tr><tr><td>w/o SlogP</td><td> $0 . 9 2 5 6 \pm 0 . 0 0 2 3$ </td><td> $0 . 8 8 3 9 \pm 0 . 0 0 1 9$ </td><td> $0 . 7 1 9 1 \pm 0 . 0 0 4 5$ </td></tr><tr><td>w/o SMR</td><td> $0 . 9 2 8 3 \pm 0 . 0 0 8 5$ </td><td> $0 . 8 7 6 8 \pm 0 . 0 0 9 6$ </td><td> $0 . 7 2 7 4 \pm 0 . 0 0 9 1$ </td></tr></table>

Discussion. Removing any prior branch generally weakens performance. On MoleculeNet classification, the effect is especially clear on HIV, where removing any branch substantially lowers ROC-AUC. For regression tasks, single-prior variants are generally weaker than the full multi-prior model on FreeSolv and Lipophilicity. ESOL is a small exception: the FP-only variant obtains $0 . 7 7 0 6 \pm 0 . 0 1 1 2 ,$ slightly below the all-space result $0 . 7 7 2 2 \pm \mathrm { { \bar { 0 } } . 0 2 7 5 }$ . Because this difference is small and within the reported variability, we interpret it as evidence that ESOL can be well supported by a strong structural prior, rather than as evidence against multi-prior complementarity overall. The solubility ablations show a clearer multi-prior pattern: removing individual prior spaces weakens external-transfer performance on pure-solvent solubility and degrades the frozen-encoder-branch mixed-solvent OOD setting under the same linear-mixing protocol used in the main text. This supports the use of multiple chemically motivated prior spaces for relation-centric solubility prediction.

Table 33: Leave-one-prior-space-out and single-prior ablations on MoleculeNet regression. Metrics are RMSE ( ).
<table><tr><td>Spaces used</td><td>ESOL</td><td>FreeSolv</td><td>Lipo</td></tr><tr><td>all</td><td> $0 . 7 7 2 2 \pm 0 . 0 2 7 5$ </td><td> $1 . 7 7 0 2 \pm 0 . 0 4 4 4$ </td><td> $0 . 5 3 8 0 \pm 0 . 0 1 5 6$ </td></tr><tr><td>w/o FP</td><td> $0 . 8 1 2 9 \pm 0 . 0 1 7 5$ </td><td> $2 . 5 0 1 1 \pm 0 . 2 0 0 5$ </td><td> $0 . 5 7 3 9 \pm 0 . 0 0 9 9$ </td></tr><tr><td>w/o PEOE</td><td> $0 . 7 8 2 6 \pm 0 . 0 2 7 4$ </td><td> $2 . 2 3 3 5 \pm 0 . 1 5 3 3$ </td><td> $0 . 5 5 6 3 \pm 0 . 0 0 8 2$ </td></tr><tr><td>w/o SlogP</td><td> $0 . 8 0 8 0 \pm 0 . 0 0 4 6$ </td><td> $2 . 4 0 7 4 \pm 0 . 1 0 3 5$ </td><td> $0 . 5 5 2 5 \pm 0 . 0 0 0 7$ </td></tr><tr><td>w/o SMR</td><td> $0 . 7 7 8 0 \pm 0 . 0 0 8 5$ </td><td> $2 . 4 3 1 2 \pm 0 . 1 6 3 0$ </td><td> $0 . 5 7 3 3 \pm 0 . 0 1 0 6$ </td></tr><tr><td>SMR only</td><td> $0 . 8 2 4 7 \pm 0 . 0 2 9 5$ </td><td> $2 . 4 8 1 2 \pm 0 . 1 6 7 3$ </td><td> $0 . 5 7 3 6 \pm 0 . 0 0 4 1$ </td></tr><tr><td>SlogP only</td><td> $0 . 7 7 9 1 \pm 0 . 0 0 7 9$ </td><td> $2 . 6 4 7 2 \pm 0 . 4 1 9 7$ </td><td> $0 . 5 6 7 1 \pm 0 . 0 0 7 1$ </td></tr><tr><td>PEÕE only</td><td> $0 . 8 3 1 1 \pm 0 . 0 2 5 6$ </td><td> $2 . 4 1 3 4 \pm 0 . 0 2 9 2$ </td><td> $0 . 5 7 6 4 \pm 0 . 0 1 5 9$ </td></tr><tr><td>FP only</td><td> $0 . 7 7 0 6 \pm 0 . 0 1 1 2$ </td><td> $2 . 5 0 5 8 \pm 0 . 2 6 1 4$ </td><td> $0 . 5 9 8 4 \pm 0 . 0 0 2 6$ </td></tr></table>

Table 34: Leave-one-prior-space-out ablations on solubility tasks. Metrics are RMSE ( ) and % ∆ log $S | \leq 1 ( \% , \uparrow )$ . Pure-solvent results are evaluated on Leeds and SolProp. Mixed-solvent results are evaluated under the solvent-pair-disjoint OOD split using the same frozen-encoder linearmixing protocol as the main text.
<table><tr><td rowspan="2">Spaces used</td><td colspan="3">RMSE (↓)</td><td colspan="3">%|∆ log S| ≤ 1 (%, ↑)</td></tr><tr><td>Leeds</td><td>SolProp</td><td>Mixed OOD</td><td>Leeds</td><td>SolProp</td><td>Mixed OOD</td></tr><tr><td>all</td><td> $0 . 9 1 0 \pm 0 . 0 0 2$ </td><td> $0 . 7 7 0 \pm 0 . 0 3 0$ </td><td> $0 . 4 4 0 \pm 0 . 0 0 8$ </td><td> $7 5 . 7 \pm 0 . 8$ </td><td> $8 2 . 3 \pm 0 . 8$ </td><td> $9 6 . 6 \pm 0 . 9$ </td></tr><tr><td>w/o SMR</td><td> $0 . 9 6 6 0 \pm 0 . 0 0 2$ </td><td> $0 . 8 0 4 0 \pm 0 . 0 0 8$ </td><td> $0 . 5 0 1 \pm 0 . 0 0 9$ </td><td> $7 3 . 2 \pm 0 . 3$ </td><td> $8 0 . 8 \pm 0 . 5$ </td><td> $8 9 . 4 \pm 1 . 5$ </td></tr><tr><td>w/o SlogP</td><td> $0 . 9 5 8 6 \pm 0 . 0 0 2$ </td><td> $0 . 8 1 2 0 \pm 0 . 0 3 1$ </td><td> $0 . 4 7 2 \pm 0 . 0 1 0$ </td><td> $7 3 . 2 \pm 0 . 6$ </td><td> $7 8 . 2 \pm 2 . 1$ </td><td> $9 0 . 6 \pm 1 . 0$ </td></tr><tr><td>w/o PEOE</td><td> $0 . 9 5 8 0 \pm 0 . 0 0 2$ </td><td> $0 . 8 0 6 5 \pm 0 . 0 2 7$ </td><td> $0 . 4 8 8 \pm 0 . 0 0 7$ </td><td> $7 3 . 2 \pm 0 . 8$ </td><td> $8 0 . 0 \pm 0 . 1$ </td><td> $9 1 . 7 \pm 0 . 7$ </td></tr><tr><td>w/o FP</td><td> $0 . 9 7 2 0 \pm 0 . 0 0 4$ </td><td> $0 . 8 2 0 3 \pm 0 . 1 0 2$ </td><td> $0 . 6 3 7 \pm 0 . 0 2 4$ </td><td> $7 2 . 5 \pm 1 . 3$ </td><td> $7 9 . 3 \pm 0 . 9$ </td><td> $8 7 . 1 \pm 2 . 7$ </td></tr></table>

Note on ESOL reporting across tables. The ESOL numbers in Tables 23, 33, and 35 are produced under different diagnostic protocols. Table 23 reports the main selected PRISM result. Table 33 reports leave-one-prior and single-prior ablations, where the “all” row repeats the corresponding ablation protocol rather than the exact main-table run. Table 35 isolates the fusion mechanism under a fixed fusion-mode comparison. These values are therefore not intended to be numerically identical, although they support the same qualitative conclusion that PRISM is strong on ESOL.

## H.5 Fusion Strategy: Task-specific versus Instance-Wise Weights

PRISM supports both task-specific and instance-wise weighted fusion. The task-specific variant uses a single learnable weight vector for all molecules in a task, whereas the instance-wise variant computes sample-dependent weights from the branch representations. Table 35 compares these two fusion strategies for classification, single-molecule regression, and pure-solvent solubility tasks.

We do not include mixed-solvent prediction in this fusion-mode comparison because the main mixedsolvent protocol freezes the pretrained PRISM encoders but keeps the pure-solvent-initialized fusion module trainable. In that setting, the goal is not to compare static versus molecule-specific fusion modes, but to test whether frozen prior-specialized encoders can be compositionally reused through lightweight downstream fusion and composition-weighted pooling.

Discussion. Molecule-specific fusion helps single-molecule regression, where the dominant physicochemical factor can vary substantially across molecules. In contrast, task-specific fusion is more stable for classification and solute–solvent prediction. This suggests that relation-level or conditional prediction tasks can make sample-wise gating more prone to overfitting, especially when the target depends on interactions among multiple molecules rather than a single molecular property.

Table 35: Task-specific versus instance-wise weighted fusion. Classification reports ROC-AUC (%, ); single-molecule regression reports RMSE ( ); pure-solvent solubility reports $\ \% | \Delta \log S | \leq 1$ (%, ) and RMSE ( ). The pure-solvent RMSE values follow the same external-transfer protocol as the main pure-solvent table.
<table><tr><td>Setting</td><td>Task-specific weights</td><td>Instance-wise weights</td></tr><tr><td>BBBP (AUC)</td><td> $9 4 . 6 \pm 0 . 3$ </td><td> $7 2 . 2 8 \pm 1 . 5$ </td></tr><tr><td>BACE (AUC)</td><td> $8 7 . 9 \pm 0 . 4$ </td><td> $8 0 . 9 1 \pm 0 . 4$ </td></tr><tr><td>HIV (AUC)</td><td> $7 7 . 7 \pm 0 . 6$ </td><td> $7 3 . 2 \pm 0 . 8$ </td></tr><tr><td>ESOL (RMSE)</td><td> $0 . 9 2 9 \pm 0 . 0 0 3$ </td><td> $0 . 7 8 2 \pm 0 . 0 0 2$ </td></tr><tr><td>FreeSolv (RMSE)</td><td> $2 . 0 6 9 \pm 0 . 0 0 7$ </td><td> $1 . 7 7 0 \pm 0 . 0 9 3$ </td></tr><tr><td>Lipo (RMSE)</td><td> $0 . 6 6 4 \pm 0 . 0 3 4$ </td><td> $0 . 5 3 8 \pm 0 . 0 1 8$ </td></tr><tr><td>Leeds  $( \% | \Delta \log S | \leq 1 )$ </td><td> $7 5 . 7 \pm 0 . 8$ </td><td> $6 8 . 6 2 \pm 0 . 7 0$ </td></tr><tr><td>SolProp  $( \dot { \% } | \Delta \log \dot { S } | \le 1 )$ </td><td> $8 2 . 3 \pm 0 . 8$ </td><td> $7 2 . 8 4 \pm 1 . 2$ </td></tr><tr><td>Leeds (RMSE)</td><td> $0 . 9 1 \pm 0 . 0 0 2$ </td><td> $1 . 0 7 0 \pm 0 . 0 0 3$ </td></tr><tr><td>SolProp (RMSE)</td><td> $0 . 7 7 \pm 0 . 0 3 0$ </td><td> $0 . 9 0 3 \pm 0 . 0 2 2$ </td></tr></table>

## H.6 Mixed-Solvent Head Ablation

The main mixed-solvent experiments use frozen PRISM encoders with a trainable pure-solventinitialized fusion module and a simple composition-weighted head. All variants use the same frozen PRISM encoders and train the fusion module together with the mixed-solvent prediction head. The simple composition-weighted head performs best and has the lowest variance, suggesting that the main mixed-solvent gain comes from reusable PRISM-pretrained encoders and a symmetry-respecting mixture representation rather than from a more expressive mixture-specific head.

The three heads differ in how composition, temperature, and solvent representations are combined. The weighted-sum head first forms the main-text solvent-mixture representation

$$
z _ { \mathrm { m i x } } = \alpha _ { 1 } z _ { 1 } + \alpha _ { 2 } z _ { 2 } , \qquad \alpha _ { 1 } + \alpha _ { 2 } = 1 ,\tag{89}
$$

and predicts from $[ z _ { \mathrm { s o l } } ; z _ { \mathrm { m i x } } ; t ]$ . This representation is permutation-invariant to swapping the two solvent identities together with their mole fractions. The direct-concat head instead avoids forming an explicit mixture embedding: it maps the scalar composition and temperature into embeddings,

$$
e _ { \alpha } = h _ { \alpha } ( \alpha _ { 1 } ) , \qquad t = h _ { T } ( T ) ,\tag{90}
$$

and predicts directly from the concatenated representation

$$
[ z _ { \mathrm { s o l } } ; z _ { 1 } ; z _ { 2 } ; e _ { \alpha } ; t ] .\tag{91}
$$

Unlike the weighted-sum head, this direct concatenation is not permutation-invariant by construction, so the MLP must learn the solvent-order symmetry from data. The attention interaction head explicitly decomposes the mixture representation into an ideal additive term and a non-ideal interaction residual. It first forms the same ideal composition-weighted solvent embedding as the main-text head,

$$
z _ { \mathrm { i d e a l } } = \alpha _ { 1 } z _ { 1 } + \alpha _ { 2 } z _ { 2 } , \qquad \alpha _ { 1 } + \alpha _ { 2 } = 1 .\tag{92}
$$

It then uses a cross-attention module over the two solvent representations to estimate a non-ideal solvent–solvent interaction term:

$$
z _ { \mathrm { n o n i d e a l } } = \alpha _ { 1 } \alpha _ { 2 } \mathrm { C r o s s A t t n } ( z _ { 1 } , z _ { 2 } ) .\tag{93}
$$

The multiplicative factor $\alpha _ { 1 } \alpha _ { 2 }$ makes the residual vanish at pure-solvent endpoints and emphasizes it near mixed compositions, where non-ideal solvent–solvent effects are most relevant. The final solvent-mixture representation is

$$
z _ { \mathrm { m i x } } = z _ { \mathrm { i d e a l } } + z _ { \mathrm { n o n i d e a l } } ,\tag{94}
$$

and the prediction head uses $[ z _ { \mathrm { s o l } } ; z _ { \mathrm { m i x } } ; t ]$ as input. Compared with the linear weighted-sum head, this variant gives the model additional capacity to learn composition-dependent deviations from ideal additive mixing.

Table 36 reports the mixed-solvent head ablation results.

Table 36: Mixed-solvent head ablation under the solvent-pair-disjoint OOD split. All variants use the same frozen PRISM encoders and train the fusion module together with the mixed-solvent prediction head. The simple composition-weighted head performs best and has the lowest variance.
<table><tr><td>Head design</td><td>RMSE↓</td><td> $\% | \Delta \log S | \le 1 \uparrow$ </td></tr><tr><td>Direct concat head</td><td> $0 . 5 5 0 \pm 0 . 0 3 1$ </td><td> $9 1 . 3 \pm 2 . 8$ </td></tr><tr><td>Attention interaction head</td><td> $0 . 4 4 5 \pm 0 . 0 2 0$ </td><td> $9 4 . 3 \pm 0 . 8$ </td></tr><tr><td>Linear weighted-sum head</td><td> $\mathbf { 0 . 4 4 0 \pm 0 . 0 0 8 }$ </td><td> ${ \bf 9 6 . 6 \pm 0 . 9 }$ </td></tr></table>

Discussion. The direct-concat head underperforms the composition-weighted sum. This result should be interpreted carefully: the gap may reflect both the value of forming a compositionweighted mixture embedding and the fact that the weighted-sum head enforces the natural permutation symmetry of binary solvent mixtures. Thus, the experiment does not by itself prove that separated solvent embeddings are intrinsically worse. Rather, it shows that, under the frozen-encoder protocol, a symmetry-respecting linear composition head is a stronger and more stable way to reuse PRISM solvent representations. The attention interaction head is more expressive and achieves a similar RMSE to the weighted-sum head, but it has lower $\% | \Delta \log S | \leq 1$ and larger variance. The simple weighted-sum head is therefore preferred because it is both more stable and more aligned with the representation-transfer hypothesis tested in the main text. Rather than relying on a task-specific interaction module to relearn solvent-pair effects, this head asks whether frozen PRISM-pretrained encoders, combined with lightweight trainable fusion, can be compositionally reused under an interpretable linear operation.

## H.7 Solvent Aggregation, Prior Fusion, and Warm-Start Initialization

We further examine how solvent aggregation, prior-branch fusion, and warm-start initialization affect mixed-solvent OOD performance. The preceding head comparison evaluates alternative prediction heads; here, we additionally vary the fusion mechanism and its initialization to distinguish their contributions.

Experimental setup. All configurations retain the frozen PRISM-pretrained molecular encoders and train the downstream fusion or aggregation components together with the mixed-solvent prediction head. We consider two ways to aggregate solvent representations: composition-weighted summation and concatenation. We separately compare learned prior-branch fusion (Fusion Head) with concatenation of the prior-branch representations (Branch Concatenation). Thus, solvent concatenation combines representations of different solvent molecules, whereas branch concatenation combines representations from different prior-specialized encoders.

For configurations with a fusion head, we evaluate both pure-solvent warm-start initialization and random initialization. Warm start initializes the fusion module from the pure-solvent task before mixed-solvent training. Removing warm start changes this initialization while retaining the frozen PRISM-pretrained encoders. It therefore tests the contribution of supervised pure-solvent fusion initialization, rather than removing PRISM pretraining. Results are reported as mean and standard deviation across five solvent-pair-disjoint splits.

Effects of aggregation and fusion. Table 37 supports the use of composition-weighted solvent aggregation and learned prior fusion. With warm start disabled, replacing learned prior fusion with branch concatenation under weighted-sum solvent aggregation increases mean RMSE from 0.442 to 0.525 and its standard deviation from 0.080 to 0.255. Under the same initialization setting, retaining learned prior fusion but replacing weighted-sum solvent aggregation with concatenation increases mean RMSE from 0.442 to 0.557. These comparisons hold warm-start status fixed and therefore separate the architectural effects from pure-solvent initialization.

The solvent-aggregation comparison should nevertheless be interpreted with the symmetry distinction in mind. Composition-weighted summation is invariant to jointly swapping solvent identities and their mole fractions, whereas direct concatenation does not enforce this invariance by construction. The observed advantage therefore supports the weighted-sum design as a whole, rather than isolating composition weighting from permutation symmetry.

Table 37: Ablation of solvent aggregation, prior-branch fusion, and warm-start initialization under solvent-pair-disjoint OOD. All configurations use frozen PRISM-pretrained encoders. Warm start refers to initialization of the fusion module from the pure-solvent task. Results are mean standard deviation across five splits. The first row is the default PRISM configuration. The metric pct-in-1 is the percentage of predictions with absolute error at most one log-solubility unit.
<table><tr><td>Solvent aggregation</td><td>Prior aggregation</td><td>Warm start</td><td>RMSE↓</td><td> $\mathrm { p c t - i n - 1 } \left( \% \right) \uparrow$ </td></tr><tr><td>Weighted sum</td><td>Fusion head</td><td>Yes</td><td> $\mathbf { 0 . 4 3 8 \pm 0 . 0 3 0 }$ </td><td> ${ \bf 9 6 . 6 \pm 0 . 9 }$ </td></tr><tr><td>Weighted sum</td><td>Fusion head</td><td>No</td><td> $0 . 4 4 2 \pm 0 . 0 8 0$ </td><td> $9 6 . 1 \pm 1 . 4$ </td></tr><tr><td>Weighted sum</td><td>Branch concatenation</td><td>No</td><td> $0 . 5 2 5 \pm 0 . 2 5 5$ </td><td> $9 3 . 7 \pm 6 . 4$ </td></tr><tr><td>Concatenation</td><td>Fusion head</td><td>Yes</td><td> $0 . 5 5 0 \pm 0 . 0 5 1$ </td><td> $9 1 . 3 \pm 3 . 1$ </td></tr><tr><td>Concatenation</td><td>Fusion head</td><td>No</td><td> $0 . 5 5 7 \pm 0 . 1 8 3$ </td><td> $9 1 . 2 \pm 7 . 1$ </td></tr><tr><td>Concatenation</td><td>Branch concatenation</td><td>No</td><td> $0 . 5 7 0 \pm 0 . 3 1 1$ </td><td> $9 1 . 7 \pm 8 . 3$ </td></tr></table>

Effect of warm-start initialization. Warm start changes mean RMSE only slightly but reduces the observed variation across splits under both solvent aggregation schemes. With weighted-sum aggregation, RMSE changes from $0 . 4 4 2 { \pm } 0 . 0 8 0$ without warm start to 0. $. 4 3 8 \pm 0 . 0 3 0$ with warm start. With concatenation, it changes from $0 . 5 5 7 \pm 0 . 1 8 3$ to $0 . 5 5 0 \pm 0 . 0 5 1$ . The standard deviation of pctin-1 also decreases in both cases. Thus, the main benefit observed for warm start in this experiment is improved consistency across splits, with a smaller change in average predictive accuracy.

The default configuration achieves the lowest mean RMSE and highest mean pct-in-1 among the evaluated configurations. Its similar mean RMSE without warm start suggests that pure-solvent fusion initialization alone does not explain its OOD performance. These results support compositionweighted aggregation and learned prior fusion for reusing frozen PRISM representations, with warm start providing an additional reduction in observed split-to-split variability.

## H.8 Extended Gaussian Random Prior Control on Classification Tasks

The main text reports the Gaussian-random-prior control on the three solubility-related MoleculeNet regression tasks. Here we provide the corresponding classification-side control, covering MoleculeNet classification benchmarks and two additional bioactivity datasets used in the prior-generality analysis. Each molecule is assigned a 10-dimensional i.i.d. Gaussian vector sampled from $\mathcal { N } ( 0 , 1 )$ , independent of molecular structure and labels. This random vector defines a structureless prior space for PRISM pretraining.

Table 38 reports the results. Across all classification tasks, Gaussian-prior pretraining degrades performance relative to the non-pretrained D-MPNN baseline, confirming that PRISM depends on prior quality. However, performance does not abruptly collapse, suggesting that the D-MPNN backbone and downstream supervised training can still recover part of the molecular signal even under an uninformative relational target.

Table 38: Gaussian random prior control on classification and bioactivity tasks. Classification performance is reported as ROC-AUC (%, ). The random prior degrades performance relative to supervised D-MPNN but does not cause abrupt collapse.
<table><tr><td>Dataset</td><td>Energy (Gaussian)</td><td>PRISM + Gaussian prior</td><td>D-MPNN w/o pretraining</td></tr><tr><td>BBBP</td><td>0.356</td><td> $8 3 . 0 7 \pm 3 . 0 4$ </td><td> $9 1 . 9 \pm 3 . 0$ </td></tr><tr><td>BACE</td><td>0.498</td><td> $7 1 . 1 7 \pm 9 . 7 0$ </td><td> $8 5 . 2 \pm 0 . 6$ </td></tr><tr><td>HIV</td><td>0.067</td><td> $7 2 . 2 9 \pm 1 . 9 8$ </td><td> $7 7 . 1 \pm 0 . 5$ </td></tr><tr><td>CYP2D6</td><td>0.025</td><td> $6 6 . 3 4 \pm 0 . 7 5$ </td><td> $7 2 . 7 \pm 0 . 3 7$ </td></tr><tr><td>KOR</td><td>0.406</td><td> $7 4 . 4 2 \pm 0 . 2 2$ </td><td> $8 2 . 0 \pm 0 . 1 8$ </td></tr></table>

Discussion. The Gaussian prior is intentionally uninformative. The additional CYP2D6 and KOR results show the same trend as the main-text Gaussian-prior control: performance degrades compared with the supervised D-MPNN baseline, but the model does not collapse. This supports the interpretation that PRISM depends on prior quality, and that poor priors can be identified by label-energy diagnostics.

## H.9 Dynamic Prior Selection via Label Energy Ratio

A ratio $\rho _ { s } < 1$ indicates that prior s induces smoother label neighborhoods than a random prior and is selected for PRISM pretraining. We test this task-level prior-selection rule using Pharm2D, fingerprint, and VSA priors on MoleculeNet classification benchmarks and two additional bioactivity datasets, CYP2D6 and KOR.

Table 39 summarizes the selected priors and downstream results.

Table 39: Dynamic prior selection using label-energy ratio. A ratio below one indicates that the candidate prior is more label-smooth than a Gaussian random prior. Classification performance is reported as ROC-AUC (%, ).
<table><tr><td>Dataset</td><td> $\rho _ { \mathrm { P h a r m 2 D } }$ </td><td>ρFP</td><td>ρVSA</td><td>Selected priors</td><td>PRISM with selected priors PRISM with all priors</td><td></td><td>D-MPNN w/o pretraining</td></tr><tr><td>BBBP</td><td>0.166</td><td>0.096</td><td>0.109</td><td> $\mathrm { P h a r m 2 D + F P + V S A }$ </td><td> $9 5 . 8 \pm 0 . 6$ </td><td> $9 5 . 8 \pm 0 . 6$ </td><td> $9 1 . 9 \pm 3 . 0$ </td></tr><tr><td>BACE</td><td>0.264</td><td>0.109</td><td>0.118</td><td> $\mathrm { P h a r m 2 D + F P + V S A }$ </td><td> $8 9 . 3 \pm 0 . 9$ </td><td> $8 9 . 3 \pm 0 . 9$ </td><td> $8 5 . 2 \pm 0 . 6$ </td></tr><tr><td>HIV</td><td>0.73</td><td>2.87</td><td>2.63</td><td>Pharm2D</td><td> $7 8 . 6 6 \pm 2 . 0 0$ </td><td> $7 6 . 3 5 \pm 1 . 9 0$ </td><td> $7 7 . 1 \pm 0 . 5 0$ </td></tr><tr><td>CYP2D6</td><td>0.96</td><td>1.05</td><td>1.06</td><td>Pharm2D</td><td> $7 3 . 3 1 \pm 1 . 0 2$ </td><td> $4 2 . 0 8 \pm 9 . 3 5$ </td><td> $7 2 . 7 \pm 0 . 3 7$ </td></tr><tr><td>KOR</td><td>0.48</td><td>0.47</td><td>0.56</td><td> $\mathrm { P h a r m 2 D + F P + V S A }$ </td><td> $8 8 . 9 1 \pm 0 . 2 9$ </td><td> $8 8 . 9 1 \pm 0 . 2 9$ </td><td> $8 2 . 0 \pm 0 . 1 8$ </td></tr></table>

Discussion. This experiment reframes prior choice as a measurable task-dependent selection problem. For BBBP, BACE, and KOR, all three candidate priors satisfy $\rho _ { s } < 1 .$ , so Pharm2D, FP, and VSA are jointly selected. For HIV and CYP2D6, only Pharm2D satisfies the selection criterion, while FP and VSA have $\rho _ { s } > 1$ and are excluded. Selected-prior PRISM consistently outperforms the non-pretrained D-MPNN baseline across the five classification tasks.

Comparing selected-prior PRISM with PRISM trained over all candidate priors further shows why prior selection matters. When all candidate priors are label-compatible, as in BBBP, BACE, and KOR, selected-prior and all-prior PRISM coincide. When label-incompatible priors are forcibly included, as in CYP2D6 and $\mathrm { H I } \bar { \mathrm { V } } ,$ downstream performance can degrade. This pattern shows that prior quality is not only quantifiable but also actionable: the label-energy ratio identifies useful prior spaces before pretraining and filters out priors that may harm representation learning.

## H.10 Compatibility with 3D Backbones and 3D Priors

The default PRISM implementation uses a 2D D-MPNN backbone and 2D prior spaces. To test whether PRISM is tied to 2D graph encoders, we replace the backbone with SchNet, a standard 3D geometry-aware molecular encoder. RDKit-generated conformers are used as input. We consider three settings: (i) randomly initialized SchNet; (ii) PRISM-pretrained SchNet using Morgan FP as the prior space; and (iii) PRISM-pretrained SchNet using a compact 3D descriptor prior. For both PRISM settings, pretraining uses 50k randomly sampled ZINC molecules.

Table 40 reports this 3D-backbone compatibility experiment.

Table 40: PRISM with a 3D SchNet backbone. Classification metrics are ROC-AUC, and regression metrics are RMSE. The 3D prior uses RDKit shape descriptors covering global shape, anisotropy, compactness, spatial extent, and principal-axis mass distribution.
<table><tr><td>Model</td><td>BBBP↑</td><td>BACE ↑</td><td>ESOL↓</td><td>FreeSolv ↓</td><td>Lipo ↓</td></tr><tr><td>Random  $\operatorname { s c h N e t }$ </td><td> $0 . 8 3 0 \pm 0 . 0 3 3$ </td><td> $0 . 8 0 7 \pm 0 . 0 2 2$ </td><td> $0 . 7 1 6 \pm 0 . 0 4 5$ </td><td> $1 . 3 1 8 \pm 0 . 3 6 1$ </td><td> $0 . 7 7 5 \pm 0 . 0 3 2$ </td></tr><tr><td> $\mathrm { P R I S M } + \mathrm { S c h N e t } \ \mathrm { ( F P ~ p r i o r ) }$ </td><td> $0 . 8 3 5 \pm 0 . 0 4 8$ </td><td> $\mathbf { 0 . 8 3 0 \pm 0 . 0 4 3 }$ </td><td> $\mathbf { 0 . 6 4 3 \pm 0 . 0 3 2 }$ </td><td> $\mathbf { 1 . 0 6 2 \pm 0 . 1 5 0 }$ </td><td> $\mathbf { 0 . 6 8 6 \pm 0 . 0 3 8 }$ </td></tr><tr><td> $\mathrm { P R I S M } + \mathrm { S c h N e t } \ ( 3 \mathrm { D } \ \mathrm { \hat { d e s c r i p t o r } \ p r i o r } )$ </td><td> $\mathbf { 0 . 8 4 4 \pm 0 . 0 2 2 }$ </td><td> $0 . 8 0 7 \pm 0 . 0 7 0$ </td><td> $0 . 7 0 4 \pm 0 . 0 7 9$ </td><td> $1 . 7 5 6 \pm 0 . 2 6 2$ </td><td> $0 . 8 0 1 \pm 0 . 0 1 0$ </td></tr></table>

Discussion. These preliminary results suggest that PRISM is not tied to a 2D graph backbone. With SchNet as a 3D geometry-aware encoder, PRISM pretraining with the FP prior improves BACE, ESOL, FreeSolv, and Lipo over random SchNet, while the compact 3D descriptor prior improves BBBP. The 3D descriptor prior is not uniformly stronger than the FP prior, likely because it summarizes only coarse global shape and does not encode detailed local chemistry. The goal of this ablation is compatibility rather than state-of-the-art 3D performance. The default results in this paper remain centered on 2D graph-based property prediction, but the PRISM objective itself is encoder-agnostic: it only requires a prior-induced relational structure and an embedding-induced relational structure to align.

## H.11 Effect of Pretraining Data Size

Finally, we evaluate whether PRISM benefits from more unlabeled pretraining data. Table 41 compares pretraining on 25k, 50k, and 250k molecules before fine-tuning on regression tasks.

Table 41: Effect of pretraining data size on regression tasks. Metrics are RMSE ( , mean  std).
<table><tr><td>Pretrain size</td><td>ESOL</td><td>FreeSolv</td><td>Lipo</td></tr><tr><td>25k</td><td> $1 . 0 1 1 \pm 0 . 0 3 1$ </td><td> $2 . 0 2 9 \pm 0 . 1 1 3$ </td><td> $0 . 6 9 8 \pm 0 . 0 1 1$ </td></tr><tr><td>50k</td><td> $0 . 9 3 3 \pm 0 . 0 0 9$ </td><td> $2 . 1 3 8 \pm 0 . 0 2 5$ </td><td> $0 . 6 6 4 \pm 0 . 0 0 3$ </td></tr><tr><td>250k</td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 1 . 7 7 0 \pm 0 . 0 9 3 }$ </td><td> $\mathbf { 0 . 5 3 8 \pm 0 . 0 1 8 }$ </td></tr></table>

Discussion. Performance improves substantially when scaling from 25k or 50k molecules to 250k molecules. This suggests that PRISM benefits from a larger unlabeled corpus because the encoder observes a broader range of prior-induced neighborhoods during pretraining. The result is consistent with the role of PRISM as a relational pretraining objective rather than a task-specific regularizer.

## I Interpretability and Diagnostic Analyses

This section provides additional analyses for interpreting PRISM. We focus on three complementary questions. First, we test whether fusion weights are directionally faithful to perturbation-based importance. Second, we visualize whether PRISM pretraining preserves prior-induced molecular geometry. Third, we test whether and where prior information remains recoverable from frozen PRISM-pretrained embeddings. The final probe is interpreted as a diagnostic rather than as a full proof of recoverability for every prior space.

## I.1 Faithfulness Tests for Fusion Weights

PRISM fusion weights are intended to represent soft preferences over prior spaces. Because the prior spaces are chemically related and partially redundant, these weights should not be interpreted as hard causal assignments. Nevertheless, a meaningful fusion mechanism should be directionally consistent with perturbation-based importance tests. We therefore evaluate faithfulness separately for task-specific and molecule-specific fusion.

Task-specific fusion. For tasks with task-specific weights, leave-one-prior-space-out evaluation provides a direct faithfulness test. If a prior branch receives a larger learned weight, removing that branch should generally cause a larger performance drop. Across Leeds/SolProp and BBBP/BACE/HIV, the rank agreement between learned weights and leave-one-space-out importance is consistently positive, with Spearman $\rho = 0 . 8 0$ across the evaluated settings. This suggests that task-specific fusion weights capture meaningful task-level prior-space importance.

Molecule-specific fusion. For molecule-specific fusion, leave-one-space-out is less direct because the weights vary across molecules. We therefore perform a suppression test on ESOL, FreeSolv, and Lipophilicity. For each test molecule, we compare the degradation caused by suppressing the top-weighted branch with the degradation caused by suppressing the bottom-weighted branch. Suppressing the top-weighted branch more often causes larger degradation, with probabilities of 0.64, 0.76, and 0.64 on ESOL, FreeSolv, and Lipophilicity, respectively.

Table 42 summarizes both faithfulness tests.

Table 42: Faithfulness tests for PRISM fusion weights. Task-specific weights are evaluated by rank agreement with leave-one-space-out importance. Instance-wise weights are evaluated by top-weight versus bottom-weight suppression.
<table><tr><td>Fusion setting</td><td>Faithfulness test</td><td>Result</td></tr><tr><td>Task-specific weights</td><td>Weight-LOSO importance rank agreement</td><td>Spearman  $\rho = 0 . 8 0$ </td></tr><tr><td>Instance-wise, ESOL</td><td>Top-vs-bottom suppression probability</td><td>0.64</td></tr><tr><td>Instance-wise, FreeSolv</td><td>Top-vs-bottom suppression probability</td><td>0.76</td></tr><tr><td>Instance-wise, Lipo</td><td>Top-vs-bottom suppression probability</td><td>0.64</td></tr></table>

Interpretation. These results provide directional evidence that the learned weights are meaningful rather than arbitrary. For task-specific fusion, higher-weighted spaces tend to be more important under removal. For molecule-specific fusion, the branch receiving the largest weight for a molecule is more likely to be the branch whose suppression causes the larger error increase. However, the probabilities are not close to one, which is expected because the four default prior spaces are partially interdependent. Thus, fusion weights should be interpreted as soft preference reweighting over complementary chemical views, not as exclusive hard routing or causal attribution.

## I.2 Training Diagnostics: Geometry Preservation

We next visualize whether PRISM pretraining preserves the geometry induced by prior spaces. Figure 7a compares two-dimensional t-SNE visualizations of the original VSA prior spaces and the corresponding PRISM embedding spaces. We show three VSA priors: SMR\_VSA, SlogP\_VSA, and PEOE\_VSA.

The purpose of this diagnostic is not to use t-SNE as a quantitative metric. Instead, it provides a qualitative sanity check: molecules that form local neighborhoods in the prior-feature space should remain locally organized in the learned embedding space. The highlighted anchors show representative local regions used for the anchor-wise distribution analysis in Appendix I.3.

a  
![](images/8c6c61c7942c8fdef49cee0d8fdb6ba12cc79e1811bfc8350683b52096654b94.jpg)

b  
![](images/014cb3c23fb2823f8d10e279218088818ab2cbd2b6429d666be6d45af2ce5776.jpg)  
Figure 7: Training diagnostics for PRISM prior-space alignment. (a) Two-dimensional t-SNE visualizations comparing the prior-feature geometry and the PRISM embedding geometry for SMR\_VSA, SlogP\_VSA, and PEOE\_VSA. Representative anchors are highlighted to illustrate local-neighborhood preservation. (b) Anchor-wise neighborhood distribution matching. For each anchor, we compare the distance-to-anchor distribution induced by the prior space and by the learned embedding. The small KL divergences indicate that PRISM pretraining preserves prior-induced stochastic neighborhoods. The KL values in panel (b) are absolute anchor-wise KL divergences, whereas Table 2 reports normalized KL ratios for checkpoint-level comparison.

Interpretation. The learned embedding spaces qualitatively preserve the local organization of the corresponding prior spaces. This supports the basic mechanism of PRISM: the encoder is not merely trained with an auxiliary scalar loss, but is shaped to reproduce prior-induced relational structure.

## I.3 Anchor-wise Neighborhood Distribution Matching

Figure 7b provides a more direct diagnostic of the PRISM objective. For each highlighted anchor, we compare the normalized distance-to-anchor distribution in the prior space with the corresponding distribution in the learned embedding space. The three examples correspond to SMR\_VSA, SlogP\_VSA, and PEOE\_VSA.

The resulting absolute anchor-wise KL divergences are small:

$$
\mathrm { K L } _ { \mathrm { S M R } } = 0 . 0 4 5 , \qquad \mathrm { K L } _ { \mathrm { S l o g P } } = 0 . 0 5 5 , \qquad \mathrm { K L } _ { \mathrm { P E O E } } = 0 . 0 4 9 .\tag{95}
$$

These are absolute KL values for representative anchor-wise neighborhood distributions, not the normalized KL ratios used in the checkpoint-level objective comparison in Table 2. They indicate that the learned embedding neighborhoods closely track the prior-induced neighborhoods for representative anchors.

Interpretation. This diagnostic directly visualizes the training target. PRISM aligns distributions over neighbors, not individual descriptor values. The close overlap between prior-space and embedding-space distance distributions therefore supports that the pretraining objective enforces distribution-level neighborhood matching. This also helps explain why PRISM can transfer descriptorinduced relational information without feeding descriptors as downstream model inputs.

Table 43: Prior-recoverability probe on ZINC250k (8:1:1 scaffold split). We freeze the pre-trained encoder and train a two-layer MLP decoder to reconstruct prior features from embeddings. For VSA priors, we report MSE; for fingerprints, we report 1 Tanimoto (lower is better). Metrics are computed on the test split.
<table><tr><td>Prior space</td><td>Probe error ↓</td></tr><tr><td>SMR_VSA (MSE)</td><td>0.476</td></tr><tr><td>SlogP_VSA (MSE)</td><td>0.767</td></tr><tr><td>PEÓE_VSA (MSE)</td><td>1.450</td></tr><tr><td>FP (1 − Tanimoto)</td><td>0.377</td></tr></table>

## I.4 Prior Recoverability from Frozen Embeddings

Beyond neighborhood matching, we test prior recoverability by freezing the pre-trained encoder and training a lightweight two-layer MLP decoder to map embeddings back to prior features (VSA coordinates or fingerprint bits). We conduct this probe on ZINC250k under an 8:1:1 scaffold split. Table 43 reports the test errors.

Discussion. Table 43 provides quantitative evidence of prior recoverability in PRISM embeddings. The learned embeddings support reconstruction of the injected priors using a lightweight decoder. This probe complements the neighborhood-matching diagnostics by evaluating prior-relevant information accessible from individual molecular embeddings while keeping the encoder fixed.

## J Limitations and Broader Impacts

Limitations. PRISM depends on the quality of the chosen prior spaces: when a prior induces labelirrelevant neighborhoods, relational supervision can degrade downstream performance (Gaussianprior control, Appendix H.8). This dependence is well-aligned with our use-inspired scope: chemistry provides mature prior families, and label energy turns prior selection from an empirical question into a diagnosable one. Compatibility with 3D priors and 3D-aware backbones is verified in Appendix H.10; we keep the main results centered on 2D inputs because 3D pretraining brings orthogonal design choices (notably conformer generation) outside our scope. On extreme temperature extrapolation, Fastprop [27] remains stronger because it explicitly adds a d log S/dT gradient term as an auxiliary loss; we deliberately omit such a thermodynamic constraint to keep the empirical results attributable to PRISM’s prior-induced relational supervision alone, since thermodynamic regularization is itself a separate form of domain-knowledge injection that would confound the comparison. The two are orthogonal and can be combined in future work, but disentangling them in the present study is what allows us to characterize what relational supervision contributes on its own. The mixed-solvent evidence is based on one curated binary-solvent benchmark; extending the evaluation to additional mixture datasets and cross-dataset mixture transfer remains an important next step.

Broader impacts. The intended positive impact is to improve molecular property prediction under distribution shifts, which may support more efficient solvent selection, molecular screening, and experimental prioritization. Potential negative impacts are indirect: improved molecular prediction tools could be misused in harmful chemical design workflows if combined with inappropriate objectives or unsafe candidate-generation pipelines. We release only representation-learning and prediction code, not generative design tools, and encourage users to follow standard safety, licensing, and domain-review practices when applying the method.

Assets and licenses. We use publicly available molecular datasets and open-source cheminformatics and machine-learning libraries. Table 44 summarizes the main external assets, their roles in this work, and the license or access terms reported by their original sources. The PRISM implementation, training and evaluation scripts, and usage instructions are publicly available at https://github.com/cxy0206/PRISM. The repository includes a code license and documentation for data preparation.

Table 44: External assets used in this work.
<table><tr><td>Asset</td><td>Role</td><td>License / access terms</td></tr><tr><td>RDKit</td><td>Molecular parsing, descriptors, fingerprints, and featurization</td><td>BSD-style open-source license</td></tr><tr><td>PyTorch</td><td>Neural-network implementation and training</td><td>BSD-style open-source license</td></tr><tr><td>MoleculeNet 7 DeepChem</td><td>Single-molecule property benchmarks and scaffold-split evaluation</td><td>Open-source benchmark resources; DeepChem/MoleculeNet code is released under the MIT license</td></tr><tr><td>BigSolDB</td><td>Training source for pure-solvent solubility transfer</td><td>Creative Commons Attribution 4.0 International (CC BY 4.0)</td></tr><tr><td>SolProp</td><td>External pure-solvent solubility test set</td><td>Creative Commons Attribution 4.0 International (CC BY 4.0)</td></tr><tr><td>Leeds</td><td>External pure-solvent solubility test set</td><td>Open Zenodo record; license marked as &quot;Other (Open)&quot;</td></tr><tr><td>solvent dataset / CheMixHub drug-</td><td>Bao et al. binary- Mixed-solvent solubility benchmark</td><td>CC BY 4.0 for the CheMixHub drug-solubility task; original source cited in the paper</td></tr><tr><td>solubility task ZINC250k subset</td><td>Unlabeled molecular pretraining corpus</td><td>Public molecular dataset for research use; source and preprocessing details will be provided in the released</td></tr></table>

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

## Answer: [Yes].

Justification: The abstract and introduction state the paper’s main contributions: PRISM as a label-free molecular pretraining framework that turns expert-defined chemical prior spaces into relational teachers; prior-specific encoder branches aligned through soft neighborhood distribution matching; solubility-centered OOD evaluation across scaffold, solute, solvent, and solvent-pair shifts; and label energy as a diagnostic for prior–task compatibility. The claims are supported by the main experiments in Section 3, the ablations in Section 4, and the additional analyses in Appendix G, Appendix H, and Appendix I. Scope and limitations are further qualified in Appendix J.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes].

Justification: The paper discusses limitations related to prior–task mismatch, uninformativeprior degradation, the low-cost prior-bank scope, the 2D-centered default implementation, missing 3D factors for tasks such as FreeSolv, MoleculeNet split sensitivity, three-seed evaluation, the single mixed-solvent benchmark, and extreme temperature extrapolation. These limitations are discussed in Section 6, Appendix C.8, Appendix G.2, Appendix H.8, Appendix H.10, and Appendix J.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes].

Justification: The theoretical analysis is provided in Appendix F. The appendix formally defines the alignment objectives, relates PRISM-KL to InfoNCE and rank/ListMLE-style objectives, proves the label-energy kNN risk bound, and gives a finite-set no-worse approximation result under an explicit recoverability assumption. The assumptions, statements, and proofs are stated in the appendix and cross-referenced from the main text.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes].

Justification: The paper specifies datasets, preprocessing assumptions, split protocols, leakage controls, unlabeled pretraining protocol, evaluation metrics, model architectures, prior spaces, pretraining objective, downstream heads, hyperparameters, checkpoint selection, repeated-run protocol, and compute resources in Appendix C and Appendix E. Implementation code and instructions for pretraining, fine-tuning, and evaluation are publicly available at https://github.com/cxy0206/PRISM.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes].

Justification: The PRISM implementation, training and evaluation scripts, and usage instructions are publicly available at https://github.com/cxy0206/PRISM. Data access, preprocessing, split construction, leakage controls, and the unlabeled pretraining protocol are described in Appendix C, with dataset citations and license or access information summarized in Appendix J.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes].

Justification: Section 3 summarizes the experimental settings, and Appendix C and Appendix E provide detailed dataset descriptions, split construction, leakage checks, metrics, molecular featurization, model heads, optimizer settings, hyperparameters, checkpoint selection, and compute resources. Additional ablation-specific protocols are described in Appendix H.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes].

Justification: Unless otherwise specified, the main experimental results report mean standard deviation over three seeds. Appendix C.8 states that split-dependent evaluations regenerate the corresponding split for each seed, so the reported standard deviations capture both initialization and split resampling when applicable. Table 5 and the ablation in Appendix H.7 use five seeds.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes].

Justification: Appendix E.10 reports the hardware, wall-clock time, batch size, and memory usage for PRISM pretraining and downstream training. The paper notes that prior branches can be pretrained independently, gives the O(KB<sup>2</sup>) minibatch relational-construction cost, and states that the reported pipeline is feasible on consumer-grade GPUs.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes].

Justification: The work uses public molecular datasets and standard molecular property prediction benchmarks, does not involve human subjects, and does not release high-risk generative models. Appendix J discusses potential misuse and responsible-use considerations. The supplementary repository will be anonymized at submission.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes].

Justification: Appendix J discusses positive impacts for molecular property prediction under distribution shifts, including solvent selection, molecular screening, and experimental prioritization. It also discusses indirect dual-use risks if molecular prediction tools are embedded into unsafe chemical design workflows, and states that the released assets are representation-learning and prediction tools rather than generative design tools.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [N/A].

Justification: The paper does not release high-risk generative models, scraped internet datasets, or models intended to synthesize harmful content. The planned release consists of property-prediction code, processed split files, configuration files, and reproduction instructions. The paper also encourages standard safety, licensing, and domain-review practices when applying the method.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes].

Justification: The paper cites the datasets, benchmarks, software libraries, and molecular featurization tools used in the experiments. Appendix J summarizes the main external assets, including RDKit, PyTorch, MoleculeNet/DeepChem, BigSolDB, SolProp, Leeds, the Bao et al. mixed-solvent dataset/CheMixHub task, and ZINC250k, together with their roles and license or access terms. The anonymized repository will include license files and attribution notes for reused assets.

## Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes].

Justification: The released research assets comprise the PRISM implementation and training and evaluation scripts. The public repository at https://github.com/cxy0206/PRISM provides usage instructions, environment requirements, and a code license.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A].

Justification: The paper does not involve crowdsourcing, user studies, surveys, or research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A].

Justification: The paper does not involve crowdsourcing, user studies, surveys, or research with human subjects, so IRB approval is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A].

Justification: The core method, experiments, and scientific claims do not use LLMs as an important, original, or non-standard methodological component. Any use of writing, editing, or formatting tools, if applicable, is limited to drafting or polishing and does not affect the method, experiments, results, or originality of the research.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.