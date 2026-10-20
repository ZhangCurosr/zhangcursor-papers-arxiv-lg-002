# ProtGlycanDock: Towards Accurate Protein-Glycan Docking with Tailored Dataset, Benchmark and Model

Minghao Xu<sup>1</sup> <sup>∗</sup> Jiaze Song<sup>1</sup> <sup>∗</sup> Jinyu Wang<sup>2</sup> Wentao Zhang<sup>1,3,4</sup> <sup>†</sup> <sup>\*</sup>equal contribution <sup>†</sup>corresponding author

<sup>1</sup>Peking University <sup>2</sup>University of Chinese Academy of Sciences <sup>3</sup>Zhongguancun Academy <sup>4</sup>Beijing Key Laboratory of Data Intelligence and Security (Peking University) contacts: minghao.xu@stu.pku.edu.cn, wentao.zhang@pku.edu.cn

## Abstract

Performing accurate protein-glycan docking is critical to understanding proteinglycan interactions which govern many biological processes. However, this important task is hampered by the lack of tailored datasets, benchmarks and models for protein-glycan docking. To fill this blank, we first curate a standard benchmark ProtGlycanDock for training and evaluating protein-glycan docking models under both protein and glycan generalization settings. On this benchmark, we evaluate ten models in three categories, physics-based docking methods, machine-learning-based docking models and molecular structure prediction models. To obtain more precise binding poses, we further propose two finetuning techniques to inject the knowledge of glycan conformations into a generalpurpose molecular structure prediction model. These two techniques respectively augment model inputs with glycan topological features and refine glycan binding pose with a glycan energy loss. Benchmark results show the superiority of AlphaFold 3 among existing methods, and the two proposed finetuning techniques can be well combined to achieve a state-of-the-art model for proteinglycan docking. Also, we perform stratified benchmark analysis and case studies to better understand the strengths and weaknesses of current models, providing insights for future model improvements. We maintain datasets at https:// huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock and source codes at https://github.com/kasawa1234/ProtGlycanDock.

## 1 Introduction

Protein-glycan interactions play pivotal roles in a wide spectrum of biological processes, including cell metabolism [22], immune response [40], pathogen recognition [13] and cellular adhesion [16]. Therefore, it is of great interest to solve the 3D structures of protein-glycan complexes, which lays the foundation for studying their interactions. However, the structural characterization of proteinglycan complexes remains experimentally challenging due to the inherent flexibility, branching diversity and weak binding affinity often exhibited by glycans [17]. Computational molecular docking offers a powerful complementary approach to predict protein-glycan complex structures [29].

Existing molecular docking [14, 39, 11, 9] and structure prediction [1, 23, 36, 37, 32] approaches are primarily designed to model protein-small molecule and protein-protein interactions. However, they give limited consideration to the unique physicochemical properties of glycans, such as their inherent flexibility and branching patterns. Moreover, existing protein-glycan complex benchmarks [31, 33] typically contain less than 100 samples and do not support model training.

To address this limitation, this work proposes the ProtGlycanDock benchmark supporting model training and evaluation for protein-glycan docking. To build the benchmark dataset, we collect high quality protein-glycan complex structures stored in the Protein Data Bank (PDB) [4], and we further use PDBFixer [15] and MODELLER [41] to repair incomplete structures and convert non-standard amino acid residues to their canonical counterparts. The benchmark dataset is split by holding out novel proteins and glycans respectively, thereby comprehensively assessing model performance under both protein and glycan generalization settings. On the benchmark, we evaluate a widely-used physics-based docking method, AutoDock Vina [39], its variant specific to protein-glycan docking, Vina-Carb [31], two machine-learning-based docking models, DiffDock [11] and SurfDock [9], and six general-purpose molecular structure prediction models, AlphaFold 3 [1], RoseTTAFold All-Atom [23], Chai-1 [37], Boltz-2 [32], Protenix [36] and Protenix with structural constraints (i.e., Protenix-constraint) [36].

To predict more precise protein-glycan complex structures, we finetune the Protenix and Protenixconstraint (two models with open-source training pipeline) on the training set of the ProtGlycanDock benchmark. During finetuning, we endow these general-purpose models with the knowledge of glycan structures from two aspects, i.e., input representations and training signals. For input represen tations, we augment the model with the global branching topology of glycans by encoding their 2D backbone structures and feeding into the model through an embedder module. For training signals, to ensure the physical plausibility of predicted glycan conformations, we incorporate a loss term based on the Carbohydrate Intrinsic (CHI) energy function [30], which is minimized along with the original structure prediction losses.

Benchmark results show the superior performance of AlphaFold 3 against other baseline models on protein-glycan docking. By using the proposed finetuning methods, the finetuned Protenixconstraint outperforms AlphaFold 3 with a clear margin. Ablation studies verify that the injection of glycan structural knowledge into input representation and training signal are two complementary finetuning techniques, where they can be well combined to achieve top-performing models. We also conduct stratified benchmark analysis and case studies to analyze the strengths and limitations of current models, so as to inspire future works on protein-glycan docking.

## 2 Related Work

Molecular docking and structure prediction. To get the complex structure of multiple molecules, two kinds of methods are commonly used, molecular docking and molecular structure prediction. Physics-based docking methods [14, 39, 31] predict protein-ligand interactions through an effective scoring function and search algorithm, while modern machine-learning-based docking models [11, 9] learn the spatial transformation of a ligand relative to a protein receptor.

In a separate line of research, AlphaFold 3 [1] and RoseTTAFold All-Atom [23] first achieved generalized molecular structure prediction for various biomolecules (proteins, DNAs, RNAs, small molecules, glycans and ions) and their complexes. Following their efforts, the later works sought to incorporate experimental structural constraints [37, 36] and predict binding affinity along with structures [32]. However, these models lack tailored design to capture the characteristics of glycan structures, and thus are suboptimal for protein-glycan docking. In this work, we mitigate this defect by customized model finetuning for protein-glycan docking.

Glycan machine learning. With the rapid growth of glycomics data [5, 27], machine learning shows great promise in predicting glycan functions. Early works used traditional machine learning approaches like support vector machines (SVMs) to learn patterns from mass spectrometry data [24], identify glycosylation sites on a protein [10, 26] and perform glycan clustering [45]. Recent works employed sequence-based models [35, 7, 6, 38] and graph neural networks (GNNs) [12, 3, 43] to encode the 2D backbone structure of a glycan and further predict its functions, including species origin, immunogenicity, type of glycosylation and binding affinity with a protein. GlycanAA [44] captured all-atom structures of a glycan with a hierarchical GNN, and it outperformed previous methods considering only glycan backbone structures on various function prediction tasks.

However, existing works mainly focused on predicting glycan functions while rarely studied glycan structure prediction, especially predicting protein-glycan docked structures which determine many critical biological processes like metabolism and immune response. In this work, we study proteinglycan docking in depth from the aspects of dataset, benchmark and model.

## 3 The ProtGlycanDock Benchmark

In this section, we first give an overview of the structural characteristics of glycans and proteinglycan complexes (Section 3.1), and next thoroughly introduce the procedure of building the benchmark dataset (Section 3.2).

## 3.1 Navigating Glycan Structures and Their Complexes with Proteins

In Figure 1, we display a typical proteinglycan complex in the PDB database, an endo-beta-1,4-galactosidase in complex with blood group A tetrasaccharide (PDB ID: 7Q1W) [42]. A glycan is composed of multiple monosaccharides $\{ m _ { i } \} _ { i = 1 } ^ { N }$ as its basic units, like the N-acetylglucosamine $m _ { 1 }$ , the galactose $m _ { 2 } .$ the fucose $m _ { 3 }$ and the N-acetylgalactosamine $m _ { 4 }$ in the glycan of Figure 1(b). Different monosaccharides are linked by glycosidic bonds to form a complete glycan structure. Each glycosidic bond can be represented by two dihedral angles $\phi _ { i j }$ and $\psi _ { i j }$ (Figure 1(b)), and the conformation of a glycan is primarily determined by the combination of all these dihedral angles on glycosidic bonds.

When binding with a protein, a glycan interacts with the amino acid residues in the binding pocket of the protein, as shown in Figure 1(c). Such interactions are driven by

![](images/23de0af5241b0ac03072cac2283e634f2ed6df3aeb7e452898f05b242985f76c.jpg)  
Figure 1: Structures of (a) a protein-glycan complex (PDB ID: 7Q1W), (b) the glycan in the complex, and (c) the binding pocket of the complex. Orange dashed lines denote hydrogen bonds between glycan and binding pocket amino acid residues.

various forces, such as hydrogen bonding, hydrophobic interactions and electrostatic interactions. Among them, hydrogen bonding is the primary determinant for glycan positioning in the binding pocket, where hydrogen donor-acceptor pairs between glycan atoms and protein side chain atoms are attracted directionally to lock the glycan into a specific orientation (Figure 1(c)). The binding pocket along with other protein domains form the complete protein-glycan complex (Figure 1(a)).

## 3.2 Building the Benchmark Dataset

Data collection. We first extract all the structures with at least one glycan entity from the PDB (CC0 1.0 license), for a total of 13,716 structures. Protein-glycan interactions within these structures can be distinguished as two types, i.e., covalent glycosylation and noncovalent binding. Glycosylation is a covalent modification of protein surface residues, which is out of the scope of protein-glycan binding studied in this work. Therefore, we keep only 2,150 noncovalent binding protein-glycan complex structures. To prevent some overly complicated structures from interfering with overall benchmark results, we remove the structures with more than 4 protein chains, resulting in the final 1,781 structures. In Appendix A, we provide the statistics of proteins and glycans in these structures.

Data fixing. By navigating the raw PDB structures, we find two typical problems hindering structure modeling: (1) the existence of non-standard amino acid residues, and (2) the incomplete protein structures. To address the first problem, we employ the PDBFixer [15] to substitute all non-standard amino acid residues occurred in the dataset with their structurally most similar counterpart within 20 common residues, followed by local structure relaxation through Amber energy minimization [19]. To repair incomplete protein structures, we perform targeted loop modeling with MODELLER [41], generating and selecting the best model based on the Discrete Optimized Protein Energy (DOPE) score to complement any missing residues. Notably, for all structures, the residues requiring fixing are beyond $6 \mathrm { \AA }$ away from the glycan, thereby leaving binding pockets fully unaffected. All 1,781 collected structures successfully pass these two fixing steps, forming the final dataset.

Dataset splitting. To comprehensively evaluate how well a model generalizes across unseen glycans and proteins, we build dedicated glycan-holdout and protein-holdout test sets respectively as below:

![](images/b672fd859ef7d0da41f8271e77b8bbd179f8925580995653588512cfc1921c88.jpg)  
Figure 2: Illustration of finetuning a molecular structure prediction model with glycan topology embedding (red part) and Carbohydrate Intrinsic (CHI) energy loss (purple part).

• Glycan-holdout test set: The whole dataset contains 332 distinct glycans. After sorting them by their earliest PDB deposition date, we hold out the 45 latest deposited glycans and their 134 corresponding protein–glycan samples to form the glycan-holdout test set. This test set evaluates the model’s generalizability to novel glycans identified in cutting-edge biological research.

• Protein-holdout test set: Given that the newest protein-glycan complexes have already been largely assigned to the glycan-holdout test set, and the remaining post-cutoff PDB entries (the latest training cutoff of baseline models: 2023-06-01) have close homologs or even the same proteins in earlier deposits, we use MMseqs2 [34] cluster-based split to avoid baseline memorization to the greatest extent. After clustering, we hold out 31 under-represented protein clusters (with 148 protein-glycan samples) for test, where 20 of the clusters have fewer than 5 PDB samples and the largest have only 8, making the evaluated proteins both dissimilar to our benchmark’s training data and hard to memorize from PDB-scale pre-training.

For the rest 280 protein clusters, we count the number of glycans that only occur in the proteinglycan samples of each cluster, and the 25 clusters with most such within-cluster unique glycans are selected. The 125 protein-glycan samples in these 25 clusters serve as the validation set. This validation set can evaluate model generalization to unseen glycans and proteins simultaneously, thus satisfying the criteria for model selection. Finally, the ProtGlycanDock benchmark is composed of 1,374 training, 125 validation, 134 glycan-holdout test, and 148 protein-holdout test samples.

## 4 Customized Finetuning for Protein-Glycan Docking

Given a pre-trained molecular structure prediction model, we aim to enhance its ability to predict protein-glycan complex structures. To achieve this, we finetune the model by incorporating glycan topological features into its input (Section 4.1) and employing a training loss that focuses on glycosidic bond rotation (Section 4.2), as shown in Figure 2.

## 4.1 Input Representation of Glycan Topology

Existing molecular structure prediction models [1, 37, 32, 36] commonly deem a glycan as a group of connected small molecules (each small molecule denotes a monosaccharide in the glycan), and they represent each small molecule by its reference conformer, as illustrated in the lower left corner of Figure 2. By using the features of reference conformers as input, the model can well capture the local atomic structure of each monosaccharide in a glycan. However, this method cannot capture the global topology of a glycan which defines the glycan’s overall backbone structure. To tackle this limitation, we inject the features of glycan topology into the model through single and pair representations, fitting the original design of mainstream models [1, 37, 32, 36]:

• Single representation of glycan topology: In single representation, we would like to incorporate the type information of different monosaccharides in a glycan topology. Specifically, we represent the type of each monosaccharide with a 37-dimensional one-hot vector and map it to the space of single representation with an embedding layer, where 37 types of monosaccharides occurred in the PDB database are considered. The embedding of a monosaccharide is added to the single representations of all its associated atoms.

• Pair representation of glycan topology: In pair representation, we would like to embed the graphical structure of a glycan formed by glycosidic bonds. In specific, we represent each glycosidic bond with an 11-dimensional one-hot vector and use an embedding layer to map the vector to the space of pair representation, where we consider 11 types of glycosidic bonds occurred in the PDB. The embedding of a glycosidic bond is added to its associated atom pair, and the pair representations of the atom pairs without glycosidic linkage are not modified.

The single and pair representations after glycan topology injection are fed into the model, and the model is finetuned to adapt these changed inputs.

## 4.2 Training Loss for Glycosidic Bond Geometry

When binding with a protein, a glycan tends to adopt a conformation with low energy [17]. As stated in Section 3.1, the dihedral angles ϕ and ψ of glycosidic bonds mainly determine the conformation of a glycan and further determine its energy. Therefore, in a predicted protein-glycan complex, we aim to minimize the glycan energy with respect to the dihedral angles of glycosidic bonds.

To achieve this, we adopt the Carbohydrate Intrinsic (CHI) energy function [30] to represent glycan energy with glycosidic dihedral angles. The CHI energy function approximates quantum mechanical rotational energies with Gaussian expansions, and the energy functions of both ϕ and ψ are respectively calculated for α- and β-linkages (two stereoisomeric configurations of glycosidic bonds):

ϕ in α-linkages:

$$
\begin{array} { r } { E ( \phi ) = 2 . 9 7 7 e ^ { - \frac { ( \phi + 1 9 9 . 4 9 ) ^ { 2 } } { 6 7 7 . 8 1 } } + 1 0 2 . 2 5 e ^ { - \frac { ( \phi - 1 7 0 . 6 ) ^ { 2 } } { 1 6 9 6 . 8 } } + 1 0 . 7 4 5 e ^ { - \frac { ( \phi + 1 0 5 . 3 1 ) ^ { 2 } } { 4 7 2 4 . 6 } } + 3 . 6 7 4 e ^ { - \frac { ( \phi - 6 . 2 0 1 ) ^ { 2 } } { 1 3 4 7 . 7 } } + } \\ { 2 . 0 6 1 e ^ { - \frac { ( \phi - 9 1 . 6 5 5 ) ^ { 2 } } { 1 5 0 0 } } + 6 . 1 9 4 e ^ { - \frac { ( \phi + 2 2 . 9 7 9 ) ^ { 2 } } { 2 1 2 2 . 3 } } - 2 . 1 1 2 e ^ { - \frac { ( \phi - 8 3 . 6 0 2 ) ^ { 2 } } { 1 2 5 4 . 1 } } - 9 8 . 0 0 1 e ^ { - \frac { ( \phi - 1 7 0 . 0 1 ) ^ { 2 } } { 1 5 0 8 . 7 } } , } \end{array}\tag{1}
$$

ϕ in β-linkages:

$$
\begin{array} { r l } & { E ( \phi ) = 4 5 0 . 5 4 e ^ { - \frac { ( \phi + 3 3 0 . 7 7 ) ^ { 2 } } { 4 4 9 . 5 } } + 2 3 . 7 1 2 e ^ { - \frac { ( \phi - 3 0 4 . 6 3 ) ^ { 2 } } { 8 3 7 5 . 2 } } + 5 . 9 3 5 e ^ { - \frac { ( \phi + 1 5 2 . 0 8 ) ^ { 2 } } { 6 0 4 9 . 8 } } + 2 2 . 4 6 7 e ^ { - \frac { ( \phi + 2 3 . 5 1 6 ) ^ { 2 } } { 6 0 6 9 . 9 } } + } \\ & { \phantom { \frac { 1 } { 2 } } 1 0 . 0 3 6 e ^ { - \frac { ( \phi - 1 2 0 . 9 6 ) ^ { 2 } } { 4 0 3 8 } } - 1 8 . 1 4 1 e ^ { - \frac { ( \phi + 2 4 . 2 6 8 ) ^ { 2 } } { 5 4 3 . 0 5 } } + 5 . 8 8 2 e ^ { - \frac { ( \phi - 1 9 . 6 3 2 ) ^ { 2 } } { 8 9 7 . 9 3 } } - 2 . 1 2 8 , } \end{array}\tag{2}
$$

ψ in α-linkages:

$$
\begin{array} { r l } & { E ( \psi ) = 4 . 6 2 4 e ^ { - \frac { ( \psi - 5 . 0 4 6 ) ^ { 2 } } { 5 0 0 5 . 8 } } + 4 . 6 1 4 e ^ { - \frac { ( \psi - 3 6 2 . 4 9 ) ^ { 2 } } { 2 0 0 0 . 6 } } + 4 . 9 4 2 e ^ { - \frac { ( \psi - 1 2 1 . 2 ) ^ { 2 } } { 2 0 9 3 . 8 } } + 0 . 4 0 3 e ^ { - \frac { ( \psi - 2 4 1 . 4 3 ) ^ { 2 } } { 4 5 6 . 8 3 } } + } \\ & { \phantom { e } 0 . 7 9 9 e ^ { - \frac { ( \psi - 6 8 . 4 2 5 ) ^ { 2 } } { 6 7 5 . 8 1 } } + 0 . 2 2 3 e ^ { - \frac { ( \psi - 1 9 2 . 9 3 ) ^ { 2 } } { 3 4 7 . 2 5 } } - 0 . 1 2 6 , } \end{array}\tag{3}
$$

ψ in β-linkages:

$$
\begin{array} { r l } & { E ( \psi ) = 4 . 4 6 8 e ^ { - \frac { ( \psi - 1 0 ^ { - 3 0 } ) ^ { 2 } } { 1 2 7 9 . 6 } } + 4 . 3 8 2 e ^ { - \frac { ( \psi - 3 5 7 . 7 7 ) ^ { 2 } } { 6 0 5 0 . 1 } } + 2 8 4 . 9 5 e ^ { - \frac { ( \psi - 1 4 6 . 6 4 ) ^ { 2 } } { 1 5 5 1 . 8 } } + 4 . 7 6 1 e ^ { - \frac { ( \psi - 2 0 . 6 8 ) ^ { 2 } } { 5 8 9 2 . 9 } } - } \\ & { \qquad 1 6 9 . 2 e ^ { - \frac { ( \psi - 1 4 7 . 3 7 ) ^ { 2 } } { 1 7 4 2 . 5 } } - 1 1 8 . 4 4 e ^ { - \frac { ( \psi - 1 4 6 . 0 6 ) ^ { 2 } } { 1 3 5 9 . 8 } } + 1 . 0 2 2 , } \end{array}\tag{4}
$$

where $- 1 8 0 ^ { \circ } < \phi < 1 8 0 ^ { \circ }$ and $0 ^ { \circ } < \psi < 3 6 0 ^ { \circ }$ . The CHI energy loss $L ^ { \mathrm { C H I } }$ is defined as the average of CHI energies on all K glycosidic bonds in a glycan:

$$
\boldsymbol { L } ^ { \mathrm { C H I } } = \frac { 1 } { 2 K } \sum _ { k = 1 } ^ { K } \big ( \boldsymbol { E } ( \phi _ { k } ) + \boldsymbol { E } ( \psi _ { k } ) \big ) .\tag{5}
$$

The CHI energy loss is minimized along with the original structure prediction loss $L ^ { \mathrm { p r e d } }$ as below:

$$
L = L ^ { \mathrm { p r e d } } + \gamma L ^ { \mathrm { C H I } } ,\tag{6}
$$

where $\gamma$ is the weight for CHI energy loss, and L denotes the total loss for model finetuning. In this way, the model learns to refine the glycan conformation in a predicted protein-glycan complex.

## 5 Experiments

## 5.1 Experimental Setups

Baseline models. We adopt two physics docking methods, AutoDock Vina [39] and Vina-Carb [31], two machine-learning docking models, DiffDock [11] and SurfDock [9], and six machine-learning molecular structure prediction models, RoseTTAFold All-Atom [23], Chai-1 [37], Boltz-2 [32], AlphaFold 3 [1], Protenix [36] and Protenix-constraint [36]. All models are evaluated under the blind-docking setting (no prior ground-truth protein receptor structure and binding site information) and inferred with seed 42, 43, 44, 45 and 46 to report error bars. The detailed setups of each model are stated in Appendix B.

Finetuned models. Besides the vanilla finetuning, we finetune the Protenix and Protenix-constraint with three proposed settings:

• + Glycan Topology Embedding: Only the model input is augmented by the proposed glycan topology embedding scheme.

• + CHI Loss: Only the loss function incorporates the proposed CHI energy loss with the loss weight $\gamma = 0 . 0 1$ . The choice of loss weight is based on the scale of CHI loss against original structure prediction loss. We give a sensitivity analysis on γ in Appendix C.

• + Glycan Topology Embedding & CHI Loss: Both model input and loss function are augmented as above.

Finetuning setups. Our finetuning configuration is based on the Finetuning Stage 2 setting provided by Protenix (training crop size: 640; with training the PAE head, distogram head and diffusion module). We set the EMA decay rate to 0.9999 and the learning rate to $1 \times 1 0 ^ { - 4 }$ . Under all settings, the model is finetuned for 15,000 steps with an Adam optimizer. For model selection, validation is performed every 100 training steps, and the checkpoint achieving the highest average ranking score over all validation samples is selected for test. All experiments are conducted on a local server equipped with 100 CPU cores and 8 NVIDIA HGX H20 GPUs (96 GB).

Evaluation metrics. Following previous practices [39, 1, 36, 9], we evaluate the performance of protein-glycan docking from two aspects, geometric correctness and physical plausibility. We adopt Ligand RMSD, Ring sRMSD, Complex RMSD and protein-glycan contact recovery rate to assess the geometric correctness of predicted protein-glycan complexes, report the number of predictions satisfying Ligand RMSD < 2A as geometrically correct samples, and report the number of predic- <sup>˚</sup> tions both satisfying Ligand $\mathrm { R M S D } < 2 \mathrm { \AA }$ and passing all PoseBusters tests [8] as the samples that are both geometrically correct and physically plausible. These metrics are defined as below:

• Ligand RMSD: After superposing the predicted protein onto the ground-truth protein, the Root-Mean-Square Deviation (RMSD) between the predicted glycan and the ground-truth glycan is computed as Ligand RMSD.

• Ring sRMSD: This metric specifically evaluates glycan conformation accuracy. It aligns all ring atoms of predicted glycan onto their counterparts in ground truth, and the alignment RMSD is defined as Ring sRMSD.

• Complex RMSD: The Complex RMSD is computed over all atoms after a global superposition of the predicted complex structure onto the ground-truth complex structure.

• Contact recovery rate: It calculates the ratio of the contacting residue-monosaccharide (glycan unit) pairs recovered in the predicted complex structure.

• PoseBusters tests: These tests evaluate ligand stereochemistry, bond lengths and angles, aromatic ring and double-bond planarity, internal steric clashes, and conformational energy. They also check for steric clashes and excessive volume overlap between the ligand and the protein in a complex.

## 5.2 Overall Benchmark Results

In Table 1, we report overall evaluation results on the protein-holdout and glycan-holdout test sets of the ProtGlycanDock benchmark. We report the mean and standard deviation of each metric over five inference seeds. According to these results, we highlight the following key findings:

• Machine-learning-based docking is superior to physics-based docking in protein-glycan docking tasks. DiffDock and SurfDock produces more geometrically valid and physically plausible docking samples than AutoDock Vina and Vina-Carb, and they also recover more residue-monosaccharide contacts on the protein-glycan interface. Thus, learning-based methods generalize better across the glycan ligands of varying sizes.

• Molecular structure prediction models significantly outperform docking models under the blind docking setting. On both test sets, all evaluated molecular structure prediction models except RoseTTAFold All-Atom achieve clearly more geometrically valid and physically plausible samples than four docking models, with lower RMSD metrics, higher contact recovery rate, and more samples passing all PoseBusters tests. Therefore, molecular structure prediction models are better suited to blind protein-glycan docking tasks.

Table 1: Benchmark results on ProtGlycanDock. Mean (standard deviation) of each metric is given. The best, second-best, and third-best results are denoted by bold, underline, and italic, respectively. Abbr., PB-valid: passing all PoseBusters tests, FT.: Finetuned; Topo.: Topology; Emb.: Embedding.
<table><tr><td rowspan="2">Model</td><td colspan="6">Protein-holdout test (148 samples)</td><td colspan="6">Glycan-holdout test (134 samples)</td></tr><tr><td>#Ligand RMSD &lt; 2Å↑</td><td>#Ligand RMSD &lt; 2Å &amp; PB-valid ↑</td><td>Ligand RMSD ↓</td><td>Ring sRMSD ↓</td><td>Complex RMSD ↓</td><td>Contact Recovery% ↑</td><td>#Ligand RMSD &lt; 2Å↑</td><td>#Ligand RMSD &lt; 2Å &amp; PB-valid ↑</td><td>Ligand RMSD ↓</td><td>Ring sRMSD ↓</td><td>Complex RMSD ↓</td><td>Contact Recovery% ↑</td></tr><tr><td colspan="9">Physics-based docking methods</td><td></td><td></td><td></td><td></td></tr><tr><td>AutoDock Vina Vina-Carb</td><td>1.2 (0.4)</td><td>0.6 (0.5)</td><td>23.05 (0.62)</td><td>2.60 (0.11) 2.23 (0.13)</td><td>13.87 (0.51) 11.65 (0.31)</td><td>17.2 (0.6) 18.7 (0.8)</td><td>1.8 (0.4) 3.8 (0.7)</td><td>1.6 (0.8) 2.6 (0.5)</td><td>24.47 (0.94) 22.95 (0.87)</td><td>1.98 (0.17)</td><td>7.63 (0.22)</td><td>17.5 (0.5)</td></tr><tr><td colspan="9">2.0 (0.6) 1.2 (0.4) 22.48 (0.53)</td><td>1.71 (0.23)</td><td>7.10 (0.29)</td><td>19.3 (1.1)</td></tr><tr><td>12.8 (0.4)</td><td></td><td></td><td></td><td>Machine-learning-based docking models</td><td></td><td></td><td>12.6 (0.8)</td><td>8.2 (0.4)</td><td>16.95 (0.49)</td><td></td><td></td><td></td></tr><tr><td colspan="9">8.4 (0.5) 17.22 (0.45) 1.68 (0.10) 9.17 (0.24) 24.9 (1.2) 8.55 (0.21)</td><td>1.49 (0.13) 1.42 (0.07)</td><td>5.24 (0.22) 4.61 (0.19)</td><td>25.8 (0.9) 27.1 (0.7)</td></tr><tr><td></td><td>15.2 (0.7)</td><td>10.6 (0.8)</td><td>16.31 (0.37)</td><td>1.42 (0.11)</td><td>Molecular structure prediction models</td><td>26.3 (0.7)</td><td>14.2 (0.4)</td><td>9.4 (0.5)</td><td>15.78 (0.42)</td><td></td><td></td><td></td></tr><tr><td colspan="9">2.97 (0.19) 12.75 (0.34) 3.6 (0.5)</td><td>1.94 (0.13)</td><td>7.09 (0.34)</td><td>20.5 (0.7)</td></tr><tr><td>RoseTTAFold All-Atom Chai-1</td><td>1.8 (0.4) 20.6 (1.0)</td><td>1.4 (0.5) 14.4 (0.5)</td><td>20.78 (0.47) 15.67 (0.42)</td><td>1.10 (0.12)</td><td>9.53 (0.25)</td><td>19.3 (0.8) 28.8 (0.7)</td><td>24.2 (0.7)</td><td>3.0 (0.9) 16.8 (0.4)</td><td>23.36 (0.42) 15.71 (0.38)</td><td>1.32 (0.13)</td><td>4.72 (0.26)</td><td>29.3 (0.5)</td></tr><tr><td>Boltz-2</td><td>19.6 (0.8)</td><td>13.6 (0.5)</td><td>15.76 (0.37)</td><td>1.62 (0.13)</td><td>8.94 (0.19)</td><td>27.2 (0.6)</td><td>25.0 (0.6)</td><td>16.2 (0.7)</td><td>14.95 (0.41)</td><td>1.27 (0.09)</td><td>4.54 (0.24)</td><td>27.7 (1.0)</td></tr><tr><td>AlphaFold 3</td><td>34.6 (0.5)</td><td>24.8 (0.4)</td><td>14.27 (0.54)</td><td>0.73 (0.05)</td><td>7.98 (0.23)</td><td>34.7 (1.4)</td><td>30.6 (1.0)</td><td>21.2 (0.7)</td><td>13.87 (0.65)</td><td>1.24 (0.11)</td><td>3.76 (0.28)</td><td>35.4 (1.1)</td></tr><tr><td>Protenix Protenix-constraint</td><td>26.8 (0.7) 30.0 (0.6)</td><td>18.2 (0.4) 22.2 (0.7)</td><td>17.73 (0.29) 17.54 (0.43)</td><td>1.12 (0.07)</td><td>8.83 (0.22) 9.17 (0.27)</td><td>30.6 (0.7) 32.8 (1.2)</td><td>25.6 (0.5) 28.4 (0.5)</td><td>19.2 (0.4) 20.6 (0.5)</td><td>14.75 (0.33) 14.63 (0.51)</td><td>1.27 (0.09)</td><td>4.59 (0.21)</td><td>31.1 (0.5)</td></tr><tr><td colspan="9">0.98 (0.06)</td><td>1.38 (0.13)</td><td>4.45 (0.20)</td><td>34.3 (1.2)</td></tr><tr><td colspan="10">Finetuned molecular structure prediction models</td><td></td><td></td><td></td></tr><tr><td>FT. Protenix FT. Protenix</td><td>31.4 (0.5)</td><td>21.8 (0.7)</td><td>16.86 (0.34)</td><td>0.95 (0.11)</td><td>8.12 (0.35)</td><td>33.4 (0.8)</td><td>31.6 (1.0) 32.6 (0.5)</td><td>22.0 (0.6) 23.0 (1.4)</td><td>13.52 (0.44) 13.08 (0.30)</td><td>1.22 (0.06)</td><td>4.15 (0.21)</td><td>33.8 (0.9)</td></tr><tr><td>+ TOPO. EMB. FT. Protenix</td><td>33.2 (0.4)</td><td>23.2 (0.7)</td><td>16.09 (0.37)</td><td>0.95 (0.06)</td><td>7.97 (0.26)</td><td>34.8 (0.7)</td><td></td><td></td><td></td><td>1.20 (0.05)</td><td>3.92 (0.24)</td><td>34.3 (1.0)</td></tr><tr><td>+ CHI LoSs FT. Protenix</td><td>34.6 (0.5)</td><td>24.0 (0.6)</td><td>15.57 (0.28)</td><td>0.93 (0.10)</td><td>7.66 (0.23)</td><td>35.9 (1.1)</td><td>33.6 (1.0)</td><td>23.2 (0.7)</td><td>12.85 (0.31)</td><td>1.17 (0.08)</td><td>3.79 (0.26)</td><td>34.5 (0.5)</td></tr><tr><td>+ ToPO. EMB. &amp; CHI Loss</td><td>36.8 (0.7)</td><td>25.6 (0.5)</td><td>15.12 (0.23)</td><td>0.85 (0.05)</td><td>7.50 (0.28)</td><td>38.0 (0.9)</td><td>34.8 (0.7)</td><td>24.4 (0.5)</td><td>12.52 (0.27)</td><td>1.15 (0.09)</td><td>3.73 (0.21)</td><td>37.2 (1.1)</td></tr><tr><td>FT. Protenix-constraint FT. Protenix-constraint</td><td>35.6 (0.5)</td><td>25.0 (0.6)</td><td>15.32 (0.47)</td><td>0.92 (0.09)</td><td>7.90 (0.33)</td><td>40.8 (1.2)</td><td>32.4 (0.5) 35.8 (0.4)</td><td>23.8 (0.4) 23.6 (0.8)</td><td>12.76 (0.41) 12.04 (0.30)</td><td>1.17 (0.07)</td><td>3.83 (0.25)</td><td>39.6 (1.2)</td></tr><tr><td>+ TOPO. EMB. FT. Protenix-constraint</td><td>37.2 (0.7)</td><td>24.8 (0.4)</td><td>13.96 (0.35)</td><td>0.90 (0.07)</td><td>8.02 (0.29)</td><td>41.7 (0.8)</td><td></td><td></td><td></td><td>1.13 (0.06)</td><td>3.61 (0.19)</td><td>40.9 (1.0)</td></tr><tr><td>+ CHI LoSS FT. Protenix-constraint</td><td>38.6 (0.5)</td><td>25.8 (0.4)</td><td>14.61 (0.53)</td><td>0.86 (0.05)</td><td>7.37 (0.26)</td><td>42.3 (0.7)</td><td>37.0 (0.6)</td><td>25.6 (0.8)</td><td>11.82 (0.33)</td><td>1.10 (0.05)</td><td>3.26 (0.16)</td><td>42.0 (0.8)</td></tr><tr><td>+ ToPO. EMB. &amp; CHI Loss</td><td>42.0 (0.6)</td><td>31.2 (0.4)</td><td>11.93 (0.46)</td><td>0.84 (0.06)</td><td>6.79 (0.25)</td><td>44.7 (0.8)</td><td>42.2 (0.7)</td><td>31.8 (0.7)</td><td>11.03 (0.57)</td><td>1.02 (0.07)</td><td>2.89 (0.20)</td><td>45.2 (1.3)</td></tr></table>

• AlphaFold 3 is the best-performing model without finetuning. AlphaFold 3 outperforms other non-finetuned models in terms of RMSD metrics, contact recovery rate, and the number of geometrically valid and physically plausible samples. Especially, on the protein-holdout test set, AlphaFold 3 achieves the lowest Ring sRMSD among all models including finetuned ones, illustrating its superiority in glycan binding pose prediction.

• The protein-glycan docking capability of a model can be well enhanced by finetuning on protein-glycan complex data. After finetuning on the training set of the ProtGlycanDock benchmark, Protenix and Protenix-constraint both produce more geometrically valid and physically plausible docking samples, and their RMSD metrics and contact recovery rate are also improved. This result verifies the effectiveness of domain-specific finetuning in boosting more accurate protein-glycan docking.

• Both glycan topology embedding and CHI loss can improve the effectiveness of finetuning, and they can be well combined to yield top-performing models. For both Protenix and Protenix-constraint, when incorporating either glycan topology embedding or CHI loss during finetuning, we can observe performance improvements in RMSD metrics, contact recovery rate, and the number of geometrically valid and physically plausible docking samples on both test sets. By jointly using these two techniques, the finetuned Protenix-constraint achieves the best performance on 11 out of 12 evaluation metrics. These results demonstrate that the two proposed finetuning techniques are mutually complementary, and their integration delivers superior protein-glycan docking performance.

## 5.3 Stratified Benchmark Analysis

In Table 2, we report the performance of six competitive models varied by glycan size, protein size and data deposition time. We stratify the results by glycan size on the glycan-holdout test set, by protein size on the protein-holdout test set, and by data deposition time on the protein-holdout test set. We analyze each stratification in the following parts.

Model performance varied by glycan size. We observe performance decay across all six models as glycan size increases, illustrating their limitations in predicting protein–glycan complexes containing large glycans. The Protenix-constraint finetuned with glycan topology embedding and CHI loss achieves best performance on all 3 glycan-stratified metrics. Especially, it narrows the performance gap between small and large glycans, where, in terms of the Ligand RMSD gap between the samples with more than 7 monosaccharides and the ones with 2-4 monosaccharides, it achieves a gap of 9.02 A against the gap of 19.88<sup>˚</sup> A achieved by the non-finetuned Protenix-constraint. Therefore, the<sup>˚</sup> proposed finetuning strategy can enhance model generalization to large glycans.

Table 2: The performance of six competitive models varied by glycan size, protein size and data deposition time. Mean (standard deviation) of each stratified Ligand RMSD metric is reported. The best result is denoted by bold. Abbr., Mono.: Monosaccharides; AA.: Amino Acids; FT.: Finetuned; Topo.: Topology; Emb.: Embedding.
<table><tr><td rowspan="2">Model</td><td colspan="3">Glycan-stratified Ligand RMSD ↓ on glycan-holdout test set</td><td colspan="3">Protein-stratified Ligand RMSD ↓ on protein-holdout test set</td><td colspan="3">Time-stratified Ligand RMSD ↓ on protein-holdout test set</td></tr><tr><td>2-4 Mono.</td><td>5-7 Mono.</td><td>&gt;7 Mono.</td><td>64-500 AA.</td><td>501-1000 AA.</td><td>&gt;1000 AA.</td><td>1990-2010</td><td>2011-2020</td><td>2021-2026</td></tr><tr><td>AlphaFold 3</td><td>11.45 (0.43)</td><td>21.73 (0.57)</td><td>31.92 (0.60)</td><td>9.13 (0.48)</td><td>13.18 (0.39)</td><td>18.25 (0.45)</td><td>12.17 (0.51)</td><td>12.90 (0.38)</td><td>13.97 (0.51)</td></tr><tr><td>Protenix-constraint</td><td>12.18 (0.47)</td><td>25.42 (0.63)</td><td>32.06 (0.64)</td><td>10.07 (0.35)</td><td>13.95 (0.58)</td><td>18.59 (0.43)</td><td>13.40 (0.61)</td><td>14.03 (0.46)</td><td>14.78 (0.42)</td></tr><tr><td>FT. Protenix-constraint</td><td>10.94 (0.38)</td><td>21.13 (0.47)</td><td>28.11 (0.43)</td><td>8.87 (0.32)</td><td>12.35 (0.40)</td><td>15.98 (0.51)</td><td>11.69 (0.46)</td><td>12.48 (0.39)</td><td>13.39 (0.54)</td></tr><tr><td>FT. Protenix-constraint + TOPO. EMB.</td><td>10.63 (0.36)</td><td>18.86 (0.48)</td><td>24.95 (0.55)</td><td>8.68 (0.33)</td><td>11.76 (0.46)</td><td>15.04 (0.42)</td><td>11.10 (0.37)</td><td>11.77 (0.41)</td><td>12.68 (0.49)</td></tr><tr><td>FT. Protenix-constraint + CHI Loss</td><td>10.82 (0.38)</td><td>16.78 (0.41)</td><td>22.70 (0.47)</td><td>8.73 (0.27)</td><td>11.53 (0.35)</td><td>14.70 (0.40)</td><td>10.94 (0.32)</td><td>11.59 (0.39)</td><td>12.59 (0.42)</td></tr><tr><td>FT. Protenix-constraint + ToPO. EMB. &amp; CHI Loss</td><td>10.00 (0.24)</td><td>14.34 (0.31)</td><td>19.02 (0.39)</td><td>8.19 (0.23)</td><td>10.57 (0.32)</td><td>13.18 (0.39)</td><td>9.89 (0.27)</td><td>10.51 (0.31)</td><td>11.55 (0.35)</td></tr></table>

Model performance varied by protein size. Performance decay also occurs across all six models as protein size increases, indicating their limitations in large-protein-involved protein-glycan docking. The Protenix-constraint finetuned with glycan topology embedding and CHI loss achieves the best results across all 3 protein-stratified metrics. Also, it reduces the performance discrepancy between small and large proteins. Specifically, when comparing the Ligand RMSD gap between the samples with more than 1000 amino acids and the ones with 64-500 amino acids, the best-performing finetuned Protenix-constraint yields a gap of 4.99 A, in contrast to the larger gap of 8.52<sup>˚</sup> A obtained<sup>˚</sup> by its non-finetuned counterpart. These results demonstrate that the proposed finetuning strategy effectively strengthens the model’s generalization capability toward large proteins.

Model performance varied by data deposition time. All six models exhibit a consistent, gradual increase in Ligand RMSD across these periods, indicating lower docking accuracy on more recently deposited structures. Nevertheless, the absolute differences remain modest, where the increase between the earliest and latest groups ranges from 1.38 to 1.80 A, remaining below 2<sup>˚</sup> A for all models.<sup>˚</sup> The relative ranking of all six models is also unchanged across the three periods, suggesting that their comparative performance is consistent across temporal subsets. Overall, these models exhibit robust generalization to recently identified protein–glycan complex systems, with unchanged performance rankings across deposition periods.

## 5.4 Case Studies

Cases of prediction improvements. In Figure 3, we visualize the predictions of AlphaFold 3 and the Protenix-constraint finetuned with the two proposed techniques for the test sample with PDB ID 2JEN, and more cases are provided in Appendix D. For this sample, AlphaFold 3 gets a Ligand RMSD of 4.348 A, and the finetuned Protenix-constraint gets a Ligand RMSD of 0.699<sup>˚</sup> A. The<sup>˚</sup> structural difference between the predictions of two models mainly lies in the glycosidic bond colored in red, where the predicted bond conformation by the finetuned Protenix-constraint is close to the ground truth, while the prediction of AlphaFold 3 is not even close. Due to this structural difference, the finetuned Protenix-constraint succeeds in finding an important hydrogen bond responsible for glycan localization in the binding pocket as reported by Gloster et al. [18] (denoted by an orange dashed line in Figure 3(b)), while AlphaFold 3 fails to identify this critical hydrogen bond. This improvement is consistent with the intended role of the CHI energy loss during finetuning, which penalizes energetically unfavorable glycosidic torsion angles and encourages conformationally plau sible glycan structures.

Failure cases. In Figure 4, we show two failure cases of the Protenix-constraint finetuned with the two proposed techniques, representing two typical failure modes, and more failure cases are provided in Appendix D. For the 3BC9 sample, the model fails to locate the correct glycan-binding site, placing the glycan in a region that does not correspond to the experimentally observed interface. For the 7DKU sample, the model predicts a deeply bound glycan conformation, while the groundtruth glycan should be more shallowly bound, since this glycan, i.e., Cellobiose, serves as a glycosyl donor and should leave sufficient binding space for its acceptor. Together, these cases reveal two complementary challenges: locating the correct binding site and positioning the glycan in a manner consistent with its biological role. They suggest that further improvements may require both more accurate modeling of protein-glycan recognition and the incorporation of functional context.

![](images/f3b71d2d5d09c886795edce5e00104e8a9396df9cd647125dfca413ae401ae9d.jpg)

![](images/954dd0e66b26a3d39ee570dfd6fb86234da13163a3e853262e880785d7553f2d.jpg)  
Figure 3: Protein-glycan complex structures predicted by AlphaFold 3 and the finetuned Protenixconstraint with glycan topology embedding and CHI loss. Ground-truth proteins and glycans are shown in gray, predicted proteins are shown in blue, the predicted glycans by AlphaFold 3 show in green, the predicted glycans by finetuned Protenix-constraint show in purple, orange dashed lines denote hydrogen bonds, and red bonds in glycans denote the glycosidic bonds leading to prediction conformational differences. Abbr., FT.: Finetuned; Topo.: Topology; Emb.: Embedding.

![](images/69a2868b7a5e7262140ae01700bbbe1ada06e50741fb7050549a33b4c334204b.jpg)

![](images/48576bcdaaa4b0238ecb8cb4f1ab3a07b01aa7ee94278888fb30da0e47108ef0.jpg)  
Figure 4: Failure cases of the finetuned Protenix-constraint with glycan topology embedding and CHI loss. Ground-truth proteins show in gray, ground-truth glycans show in green, predicted proteins show in blue, and predicted glycans show in purple.

## 6 Conclusions and Future Work

This work takes the initiative of boosting more accurate protein-glycan docking with a tailored benchmark and a set of customized models. We first curate the high-quality ProtGlycanDock benchmark to evaluate models’ generalizability to novel glycans and proteins. On the benchmark, we evaluate two physics-based docking methods, two machine-learning-based docking models and six molecular structure prediction models. We further propose two finetuning techniques to enhance the ability of general-purpose molecular structure prediction models on protein-glycan docking. Ac cording to benchmark results, AlphaFold 3 achieves superior performance over other existing models, and the two proposed finetuning techniques can be well combined to achieve top performance.

Limitations and future work. The current benchmark comprehensively encompasses existing experimental protein-glycan complex structures, while its data quantity is still not large enough. To further enlarge the dataset scale, we will try to synthesize large-scale protein-glycan complexes through cross-docking in the future. Moreover, the current benchmark focuses on the blind docking setting. In future work, we will extend it to protein-glycan docking with partially known protein receptor apo structures and binding sites.

## Acknowledgments

This work is supported by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM113), Beijing Municipal Natural Science Foundation (Grant No. L2603008), National Natural Science Foundation of China (92470121, 62402016), National Key R&D Program of China (2024YFA1014003), Zhongguancun Academy (C20250204, C20250602), Beijing Major Science and Technology Project (Z251100008125043, Z251100008425023), and High-performance Computing Platform of Peking University. This work is also supported by the Tencent Project Up Scholarship project.

## References

[1] Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, 630(8016):493–500, 2024.

[2] Rishal Aggarwal, Akash Gupta, Vineeth Chelur, CV Jawahar, and U Deva Priyakumar. Deeppocket: ligand binding site detection and segmentation using 3d convolutional neural networks. Journal ofChemical Information and Modeling, 62(21):5069–5079, 2021.

[3] Alhasan Alkuhlani, Walaa Gad, Mohamed Roushdy, and Abdel-Badeeh M Salem. Gnngly: Graph neural networks for glycan classification. IEEE Access, 2023.

[4] Helen M Berman, Tammy Battistuz, Talapady N Bhat, Wolfgang F Bluhm, Philip E Bourne, Kyle Burkhardt, Zukang Feng, Gary L Gilliland, Lisa Iype, Shri Jain, et al. The protein data bank. Biological Crystallography, 58(6):899–907, 2002.

[5] Daniel Bojar and Frederique Lisacek. Glycoinformatics in the artificial intelligence era. Chemical Reviews, 122(20):15971–15988, 2022.

[6] Daniel Bojar, Diogo M Camacho, and James J Collins. Using natural language processing to learn the grammar of glycans. bioRxiv, pages 2020–01, 2020.

[7] Daniel Bojar, Rani K Powers, Diogo M Camacho, and James J Collins. Sweetorigins: Extracting evolutionary information from glycans. bioRxiv, pages 2020–04, 2020.

[8] Martin Buttenschoen, Garrett M. Morris, and Charlotte M. Deane. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chemical Science, 15(9):3130–3139, 2024.

[9] Duanhua Cao, Mingan Chen, Runze Zhang, Zhaokun Wang, Manlin Huang, Jie Yu, Xinyu Jiang, Zhehuan Fan, Wei Zhang, Hao Zhou, et al. Surfdock is a surface-informed diffusion generative model for reliable and accurate protein–ligand complex prediction. Nature Methods, 22(2):310–322, 2025.

[10] Cornelia Caragea, Jivko Sinapov, Adrian Silvescu, Drena Dobbs, and Vasant Honavar. Glycosylation site prediction using ensembles of support vector machine classifiers. BMC bioinformatics, 8:1–13, 2007.

[11] Gabriele Corso, Hannes Stark, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. Diffdock:¨ Diffusion steps, twists, and turns for molecular docking. arXiv preprint arXiv:2210.01776, 2022.

[12] Bowen Dai, Daniel E Mattox, and Chris Bailey-Kellogg. Attention please: modeling global and local context in glycan structure-function relationships. bioRxiv, pages 2021–10, 2021.

[13] Roberto C Davicino, Ricardo J Elic¸abe, Mar´ıa S Di Genaro, and Gabriel A Rabinovich. Coupling pathogen recognition to innate immunity through glycan-dependent mechanisms. International immunopharmacology, 11(10):1457–1463, 2011.

[14] Cyril Dominguez, Rolf Boelens, and Alexandre MJJ Bonvin. Haddock: a protein- protein docking approach based on biochemical or biophysical information. Journal of the American Chemical Society, 125(7):1731–1737, 2003.

[15] Peter Eastman, Mark S Friedrichs, John D Chodera, Randall J Radmer, Christopher M Bruns, Joy P Ku, Kyle A Beauchamp, Thomas J Lane, Lee-Ping Wang, Diwakar Shukla, et al. Openmm 4: a reusable, extensible, hardware independent library for high performance molecular simulation. Journal ofchemical theory and computation, 9(1):461–469, 2013.

[16] MT Elola, C Wolfenstein-Todel, MF Troncoso, GR Vasta, and GA Rabinovich. Galectins: matricellular glycan-binding proteins linking cell adhesion, migration, and survival. Cellular and Molecular Life Sciences, 64(13):1679–1700, 2007.

[17] Ana Gimeno, Pablo Valverde, Ana Arda, and Jes´ us Jim´ enez-Barbero. Glycan structures and´ their interactions with proteins. a nmr view. Current opinion in structural biology, 62:22–30, 2020.

[18] Tracey M Gloster, Farid M Ibatullin, Katherine Macauley, Jens M Eklof, Shirley Roberts, Johan P Turkenburg, Mads E Bjørnvad, Per Lina Jørgensen, Steffen Danielsen, Katja S Johansen, et al. Characterization and three-dimensional structures of two distinct bacterial xyloglucanases from families gh5 and gh12. Journal of Biological Chemistry, 282(26):19177– 19189, 2007.

[19] Viktor Hornak, Robert Abel, Asim Okur, Bentley Strockbine, Adrian Roitberg, and Carlos Simmerling. Comparison of multiple amber force fields and development of improved protein backbone parameters. Proteins: Structure, Function, and Bioinformatics, 65(3):712–725, 2006.

[20] Paulina Kaplonek, Naeem Khan, Katrin Reppe, Benjamin Schumann, Madhu Emmadi, Mar ilda P Lisboa, Fei-Fei Xu, Adam DJ Calow, Sharavathi G Parameswarappa, Martin Witzenrath, et al. Improving vaccines against streptococcus pneumoniae using synthetic glycans. Proceed ings ofthe National Academy ofSciences, 115(52):13353–13358, 2018.

[21] Weston Kightlinger, Katherine F Warfel, Matthew P DeLisa, and Michael C Jewett. Synthetic glycobiology: parts, systems, and applications. ACS Synthetic Biology, 9(7):1534–1562, 2020.

[22] Nicole M Koropatkin, Elizabeth A Cameron, and Eric C Martens. How glycan metabolism shapes the human gut microbiota. Nature Reviews Microbiology, 10(5):323–335, 2012.

[23] Rohith Krishna, Jue Wang, Woody Ahern, Pascal Sturmfels, Preetham Venkatesh, Indrek Kalvet, Gyu Rie Lee, Felix S Morey-Burrows, Ivan Anishchenko, Ian R Humphreys, et al. Generalized biomolecular modeling and design with rosettafold all-atom. Science, 384(6693): eadl2528, 2024.

[24] Shotaro Kumozaki, Kengo Sato, and Yasubumi Sakakibara. A machine learning based approach to de novo sequencing of glycans from tandem mass spectrometry spectrum. IEEE/ACM transactions on computational biology and bioinformatics, 12(6):1267–1274, 2015.

[25] Greg Landrum. Rdkit documentation. Release, 1(1-79):4, 2013.

[26] Fuyi Li, Chen Li, Mingjun Wang, Geoffrey I Webb, Yang Zhang, James C Whisstock, and Jiangning Song. Glycomine: a machine learning-based approach for predicting n-, c-and olinked glycosylation in the human proteome. Bioinformatics, 31(9):1411–1419, 2015.

[27] Haining Li, Austin WT Chiang, and Nathan E Lewis. Artificial intelligence in the analysis of glycosylation data. Biotechnology Advances, 60:108008, 2022.

[28] Alex Morehead, Nabin Giri, Jian Liu, Pawan Neupane, and Jianlin Cheng. Assessing the potential of deep learning for protein–ligand docking. Nature Machine Intelligence, 8(1):32– 41, 2026.

[29] Ferran Nieto-Fabregat, Maria Pia Lenza, Angela Marseglia, Cristina Di Carluccio, Antonio Molinaro, Alba Silipo, and Roberta Marchetti. Computational toolbox for the analysis of protein–glycan interactions. Beilstein Journal ofOrganic Chemistry, 20(1):2084–2107, 2024.

[30] Anita K Nivedha, Spandana Makeneni, Bethany Lachele Foley, Matthew B Tessier, and Robert J Woods. Importance of ligand conformational energies in carbohydrate docking: Sorting the wheat from the chaff. Journal ofcomputational chemistry, 35(7):526–539, 2014.

[31] Anita K Nivedha, David F Thieker, Spandana Makeneni, Huimin Hu, and Robert J Woods. Vina-carb: improving glycosidic angles during carbohydrate docking. Journal of chemical theory and computation, 12(2):892–901, 2016.

[32] Saro Passaro, Gabriele Corso, Jeremy Wohlwend, Mateo Reveiz, Stephan Thaler, Vignesh Ram Somnath, Noah Getz, Tally Portnoi, Julien Roy, Hannes Stark, et al. Boltz-2: Towards accurate and efficient binding affinity prediction. BioRxiv, 2025.

[33] Anna Ranaudo, Marco Giulini, Angela Pelissou Ayuso, and Alexandre MJJ Bonvin. Modeling protein–glycan interactions with haddock. Journal of Chemical Information and Modeling, 64 (19):7816–7825, 2024.

[34] Martin Steinegger and Johannes Soding. Mmseqs2 enables sensitive protein sequence search-¨ ing for the analysis of massive data sets. Nature biotechnology, 35(11):1026–1028, 2017.

[35] Ghazaleh Taherzadeh, Abdollah Dehzangi, Maryam Golchin, Yaoqi Zhou, and Matthew P Campbell. Sprint-gly: predicting n-and o-linked glycosylation sites of human and mouse proteins by using sequence and predicted structural properties. Bioinformatics, 35(20):4140–4146, 2019.

[36] ByteDance AML AI4Science Team, Xinshi Chen, Yuxuan Zhang, Chan Lu, Wenzhi Ma, Jiaqi Guan, Chengyue Gong, Jincai Yang, Hanyu Zhang, Ke Zhang, et al. Protenix-advancing structure prediction through a comprehensive alphafold3 reproduction. BioRxiv, pages 2025–01, 2025.

[37] Chai Discovery team, Jacques Boitreaud, Jack Dent, Matthew McPartlon, Joshua Meier, Vinicius Reis, Alex Rogozhonikov, and Kevin Wu. Chai-1: Decoding the molecular interactions of life. BioRxiv, pages 2024–10, 2024.

[38] Luc Thomes, Rebekka Burkholz, and Daniel Bojar. Glycowork: A python package for glycan\` data science and machine learning. Glycobiology, 31(10):1240–1244, 2021.

[39] Oleg Trott and Arthur J Olson. Autodock vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. Journal of computational chemistry, 31(2):455–461, 2010.

[40] Yvette Van Kooyk and Gabriel A Rabinovich. Protein-glycan interactions in the control of innate and adaptive immune responses. Nature immunology, 9(6):593–601, 2008.

[41] Benjamin Webb and Andrej Sali. Comparative protein structure modeling using modeller. Current protocols in bioinformatics, 54(1):5–6, 2016.

[42] Haiyang Wu, Emmanuelle H Crost, C David Owen, Wouter Van Bakel, Ana Martinez Gascuena, Dimitrios Latousakis, Thomas Hicks, Samuel Walpole, Paulina A Urbanowicz, Didier Ndeh, et al. The human gut symbiont ruminococcus gnavus shows specificity to blood group a antigen during mucin glycan foraging: Implication for niche colonisation in the gastrointestinal tract. PLoS Biology, 19(12):e3001498, 2021.

[43] Minghao Xu, Yunteng Geng, Yihang Zhang, Ling Yang, Jian Tang, and Wentao Zhang. Glycanml: A multi-task and multi-structure benchmark for glycan machine learning. In International Conference on Learning Representations, 2025.

[44] Minghao Xu, Jiaze Song, Keming Wu, Xiangxin Zhou, Bin Cui, and Wentao Zhang. Modeling all-atom glycan structures via hierarchical message passing and multi-scale pre-training. In International Conference on Machine Learning. PMLR, 2025.

[45] Yoshihiro Yamanishi, Francis Bach, and Jean-Philippe Vert. Glycan classification with tree kernels. Bioinformatics, 23(10):1211–1216, 2007.

## A More Dataset Statistics

![](images/cad8d30c32da2a3be6a33fb997b5affd38d400ba45948ee0137f577e1e6d27c5.jpg)  
(a) Amino Acid Count Distribution

![](images/7306989af3e63221c833a5eb5cda8cd6a1eab5100dd194f4fc8fea7bd4913d5d.jpg)  
(b) Chain Count Distribution

![](images/359507433d3fa617caf32b556a88e690cf0a159d5084aa2f66f571af9b51e25a.jpg)  
(c) Glycan Monosaccharide Count Distribution  
Figure 5: Dataset statistics of protein and glycan sizes. We report the count distribution of amino acid, protein chain and glycan monosaccharide by histogram.

Table 3: Dataset statistics of glycan attributes. Abbr., Glc: Glucose; Man: Mannose; GlcNAc: N-Acetylglucosamine; Gal: Galactose; Xyl: Xylose; Ac: Acetylation; Sul: Sulfation; Phos: Phosphorylation; Me: Methylation.
<table><tr><td rowspan="2">Attribute</td><td colspan="2">Configuration</td><td colspan="4">Top-5 Stem Type</td><td colspan="2">Ring Size</td><td colspan="2">Anomeric Notation</td><td colspan="5">Top-5 Modification Group</td></tr><tr><td>D-config</td><td>L-config</td><td>Gle</td><td>Man</td><td>GlcNAc</td><td>Gal</td><td>Xyl</td><td>Pyranose</td><td>Furanose</td><td>Alpha Beta</td><td>No modify</td><td>Ac</td><td>Sul</td><td>Phos</td><td>Me</td></tr><tr><td>Percentage</td><td>98.3%</td><td>1.7%</td><td>31.6%</td><td>17.3%</td><td>15.7%</td><td>10.4%</td><td>7.2%</td><td>91.6%</td><td>8.4%</td><td>47.2%</td><td>52.8%</td><td>38.3%</td><td>24.0% 15.6%</td><td>8.9%</td><td>6.1%</td></tr></table>

Statistics of protein and glycan sizes. Figure 5 provides additional statistics of the proposed dataset. As shown in Figure 5(a), the distribution of amino acid counts per sample is right-skewed, with a mean of 694.7 and a median of 555.0, indicating substantial variability in protein size across samples.

As illustrated in Figure 5(b), regarding protein complex composition, the dataset is dominated by single-chain proteins. Specifically, 1,635 samples contain a single protein chain, while a small fraction involves multiple chains (118 samples with two chains, 12 samples with three chains, and 16 samples with four chains).

Figure 5(c) summarizes the distribution of glycan sizes measured by the number of monosaccharides. Most glycans are relatively small, with the majority consisting of 2 to 4 monosaccharides (902 samples with 2 monosaccharides, 417 samples with 3 monosaccharides, and 211 samples with 4 monosaccharides). The overall mean monosaccharide count is 3.05, while the median is 2.0, indicating a bias toward short glycan chains. Nevertheless, the dataset also includes a long tail of larger glycans, with sizes extending up to 17 monosaccharides, providing coverage of more complex glycan structures.

Statistics of glycan attributes. In Table 3, we report the data distribution on five core attributes of glycans. We observe that: (1) D-configuration dominates the dataset as in nature; (2) Five stem types occupy 82.2% of all data; (3) Pyranose appears much more frequently than furanose as in nature; (4) Alpha and Beta anomeric notations are balanced; (5) Unmodification and four modification groups occupy 92.9% of all data. Therefore, the dataset faithfully reflects naturally occurring glycan properties, featuring dominant natural configurations, concentrated mainstream backbone and modification types, and balanced anomeric distribution.

## B Setups of Baseline Models

We set up the 10 baseline models used in this work as below:

• AutoDock Vina (Apache license): We follow the official configuration of AutoDock Vina v1.2. Under the blind docking setting, following previous works [11, 28], the protein receptor structures are predicted by AlphaFold 3, glycan 3D conformations are generated from SMILES sequences using RDKit [25], and the binding pockets predicted by DeepPocket [2] are used to define search boxes.

• Vina-Carb (Apache license): We employ Vina-Carb 1.0, while keeping all other configurations and the docking pipeline identical to AutoDock Vina.

• DiffDock (MIT license): We use the official model checkpoints and its default blind-docking inference settings with AlphaFold 3-predicted receptor structures and RDKit-generated glycan conformations. The generated binding poses are ranked by the confidence model, with the topranked pose selected for evaluation.

• SurfDock (MIT license): We adopt the official model checkpoints and inference pipeline with AlphaFold 3-predicted receptor structures and RDKit-generated glycan conformations. Protein surface features are computed using the official preprocessing pipeline, and the top-ranked pose scored by SurfScore is selected for evaluation.

• RoseTTAFold All-Atom (BSD license): Following the official guideline, we input the model with the amino acid sequence of a protein and the 3D conformation of a glycan in an sdf file.

• AlphaFold 3 (CC BY-NC-SA 4.0 license): At the input, a protein is represented by its amino acid sequence, and a glycan is represented as a set of monosaccharides (denoted by CCD codes) with covalent bonds. We take the output sample with highest ranking score as the prediction.

• Chai-1 (Apache license): We follow the official setup of Chai-1 v0.6.1 with a protein sequence and a glycan SMILES sequence as inputs, outputting the structure with highest ranking score.

• Boltz-2 (MIT license): We follow Boltz-2 v2.2.1 to set up the model, and its inference pipeline is basically the same as AlphaFold 3.

• Protenix (Apache license): We adopt the Protenix v0.5.0 base checkpoint. The input configuration and output scheme of Protenix are identical to AlphaFold 3.

• Protenix-constraint (Apache license): We adopt the Protenix v0.5.3 constraint checkpoint. Besides the inputs of Protenix, Protenix-constraint uses the predicted binding pocket from Deep-Pocket as its pocket constraints. Its output scheme is the same as Protenix.

## C Sensitivity Analysis on CHI Loss Weight

Table 4: Sensitivity analysis on CHI loss weight γ. Mean (standard deviation) of each metric is reported. The best result is denoted by bold. Abbr., PB-valid: passing all PoseBusters tests, FT.: Finetuned; Ptx-C.: Protenix-constraint; CL.: CHI Loss.
<table><tr><td rowspan="2">Model</td><td colspan="6">Protein-holdout test (148 samples)</td><td colspan="6">Glycan-holdout test (134 samples)</td></tr><tr><td>#Ligand RMSD &lt; 2Å↑</td><td>#Ligand RMSD &lt; 2Å &amp; PB-valid ↑</td><td>Ligand RMSD ↓</td><td>Ring sRMSD ↓</td><td>Complex RMSD ↓</td><td>Contact Recovery%↑</td><td>#Ligand RMSD &lt; 2Å↑</td><td>#Ligand RMSD &lt; 2Å &amp; PB-valid ↑</td><td>Ligand RMSD ↓</td><td>Ring sRMSD ↓</td><td>Complex RMSD ↓</td><td>Contact Recovery%↑</td></tr><tr><td>FT. Ptx-C. + CL. (γ = 0.001)</td><td>36.2 (0.7)</td><td>24.2 (0.7)</td><td>15.54 (0.38)</td><td>0.96 (0.05)</td><td>8.17 (0.30)</td><td>39.5 (1.0)</td><td>35.8 (0.4)</td><td>24.0 (0.6)</td><td>12.98 (0.47)</td><td>1.22 (0.06)</td><td>3.94 (0.25)</td><td>39.1 (0.9)</td></tr><tr><td>FT. Ptx-C. + CL. (γ = 0.01)</td><td>38.6 (0.5)</td><td>25.8 (0.4)</td><td>14.61 (0.53)</td><td>0.86 (0.05)</td><td>7.37 (0.26)</td><td>42.3 (0.7)</td><td>37.0 (0.6)</td><td>25.6 (0.8)</td><td>11.82 (0.33)</td><td>1.10 (0.05)</td><td>3.26 (0.16)</td><td>42.0 (0.8)</td></tr><tr><td>FT. Ptx-C. + CL. (γ = 0.1)</td><td>35.4 (0.5)</td><td>23.6 (0.5)</td><td>16.07 (0.41)</td><td>0.98 (0.06)</td><td>8.53 (0.37)</td><td>39.1 (0.8)</td><td>34.8 (0.4)</td><td>23.2 (0.7)</td><td>13.25 (0.43)</td><td>1.28 (0.10)</td><td>4.27 (0.23)</td><td>38.3 (1.1)</td></tr><tr><td>FT. Ptx-C. + CL. (γ = 0.5)</td><td>31.0 (0.6)</td><td>20.6 (0.8)</td><td>16.79 (0.45)</td><td>1.03 (0.09)</td><td>9.44 (0.39)</td><td>37.6 (0.7)</td><td>31.6 (0.5)</td><td>20.8 (0.4)</td><td>13.75 (0.38)</td><td>1.32 (0.11)</td><td>4.43 (0.26)</td><td>37.5 (0.9)</td></tr></table>

In Table 4, we report the performance of the finetuned Protenix-constraint with CHI loss under the CHI loss weight 0.001, 0.01, 0.1 and 0.5. It can be observed that, on both test sets, the model with the CHI loss weight 0.01 performs best in terms of RMSD metrics, contact recovery rate, and the number of geometrically valid and physically plausible docking samples. This result is attributed to the fact that the CHI loss weight 0.01 makes the CHI loss and the original structure prediction loss at a similar scale, which well balances two tasks, glycan conformation refinement and complex structure prediction.

## D More Case Studies

More cases of prediction improvements. In Figure 6, we visualize two more predictions of AlphaFold 3 and the Protenix-constraint finetuned with the two proposed techniques. On the test samples with PDB ID 7NWF and 4XWN, the finetuned Protenix-constraint all achieves a lower Ligand RMSD than AlphaFold 3. By investigating the predicted structures, it can be observed that the glycan binding poses predicted by the finetuned Protenix-constraint is obviously closer to the ground truth than AlphaFold 3 predictions. These results again demonstrate the effectiveness of the proposed finetuning techniques in refining the glycan conformation in a protein binding pocket.

More failure cases. In Figure 7, we present three more failure cases of the Protenix-constraint finetuned with the two proposed techniques. In these three cases, the model respectively fails to identify the glycan binding site, docks the glycan overly deep inside the binding pocket, and predicts a wrong glycan binding pose. These represent three typical failure modes of the model, which we will prioritize addressing in our future work.

![](images/0ec8aeb0e78be6671f5abce67028b79267036101cb1cc2605b75e662361870a9.jpg)  
(a) Ligand RMSD for 7NWF: 2.734 Å  
FT. Protenix-constraint + Glycan Topo. Emb. & CHI Loss

![](images/f1ea2cd939604ba3a7106c8dd020585346136920c5c1ec42872dbee1dc3c40a5.jpg)  
(b) Ligand RMSD for 7NWF: 0.921 Å

![](images/6df82474825ab0b117bdd9e3d1e0b52499b5184c093df4486c3c584cda7f3991.jpg)  
(c) Ligand RMSD for 4XWN: 3.470 Å

![](images/a41ea9ac4877aeeaf00c0095e66c7c0dbcc628975fc823e0162ef495c52f9779.jpg)  
(d) Ligand RMSD for 4XWN: 1.299 Å

Figure 6: Visualization of predicted protein-glycan complex structures by AlphaFold 3 and the finetuned Protenix-constraint with glycan topology embedding and CHI loss. Ground-truth proteins and glycans are shown in gray, predicted proteins are shown in blue, the predicted glycans by AlphaFold 3 show in green, and the predicted glycans by finetuned Protenix-constraint show in purple. Abbr., FT.: Finetuned; Topo.: Topology; Emb.: Embedding.  
![](images/2e0cca95425d858d774ecf9f80a8ced05500bc14bcf4338b42f55707679795f2.jpg)  
(a) Ligand RMSD for 7BYV: 25.676 Å (wrong binding site)

![](images/aed03acd5b192809929343f641205d06c44b4bff4426ff69c94cb843b7445579.jpg)  
(b) Ligand RMSD for 7DKX: 6.448 Å (wrong binding depth)

![](images/d3912a3fceea5e3fbd4d892cc13f028bb72c51d437973b8857ac1df5f078c6e2.jpg)  
(c) Ligand RMSD for 5JIW: 4.184 Å (wrong binding pose)

Figure 7: Failure cases of the finetuned Protenix-constraint with glycan topology embedding and CHI loss. Ground-truth proteins show in gray, ground-truth glycans show in green, predicted proteins show in blue, and predicted glycans show in purple.

## E Broader Impacts

This work aims to build a tailored benchmark and develop a set of customized models for proteinglycan docking. The proposed models have the potential to help real-world applications related to protein-glycan interactions, such as synthetic biology [21] and vaccine design [20].

However, we should not ignore the potential risks brought by the proposed models, e.g., developing a novel microorganism disrupting local ecosystems. Therefore, we will strive to ensure the responsible deployment of the proposed models in real-world applications.