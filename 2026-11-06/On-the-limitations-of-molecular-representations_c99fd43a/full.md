# On the limitations of molecular representations

Acer Blake   
Department of Statistics   
University of Oxford   
Oxford, UK   
blake@stats.ox.ac.uk   
Charlotte M. Deane   
Department of Statistics   
University of Oxford   
Oxford, UK   
deane@stats.ox.ac.uk   
Fergus Imrie   
Department of Statistics   
University of Oxford   
Oxford, UK   
imrie@stats.ox.ac.uk

## Abstract

Molecular representations such as fingerprints and learned embeddings play a central role in cheminformatics and small-molecule machine learning. They serve as the primary input for a diverse array of tasks, including clustering, dataset splitting, and virtual screening, yet their evaluation is often limited to downstream model performance on narrow benchmark tasks. We carried out a systematic investigation of the quality of molecular representations, finding that none of the state-of-the-art molecular embedding models capture graph structure, 3D shape, or ADMET endpoint similarity better than fingerprints or physicochemical descriptors. However, classical fingerprints also revealed their own material limitations. For example, we show that depending on the representation, between 225M and ∼4B molecules have non-unique representations in the 10.8B Enamine REAL database, with implications for virtual screening. Our findings provide an insight into the current limitations of molecular representations, outlining future research directions for the development of more robust encodings of small molecules.

Small-molecule representations have been a primary research focus of the cheminformatics community since the inception of the field. Indeed, among the first questions facing a developer of a machine learning model for a task in small-molecule chemistry is what representation should be used as input. While representations abound, three primary input types predominate: string-based notations, molecular graphs, and vector-based encodings<sup>1</sup>.

String-based notations, such as the Simplified Molecular Input Line Entry System (SMILES)<sup>2</sup> format, encode small molecules as one-dimensional sequences of characters wherein the order of atom and bond types represents a particular traversal order of a two-dimensional molecular graph. SMILES is the standard for small-molecule datasets<sup>1</sup> and is additionally used as an input representation for many chemical language models (CLMs)<sup>3</sup>. Graph-based representations emphasize modelling the connectivity between atoms in two or three-dimensional molecular graphs and include representations such as adjacency matrices. Such representations are commonly used as inputs to graph neural networks (GNNs)<sup>4</sup>. Finally, vector-based representations are the primary representation for most small molecule machine-learning applications. Vector-based representations include physicochemical descriptors<sup>5</sup>, classical molecular fingerprints<sup>6,7,8</sup>, and learned molecular embeddings<sup>9</sup>. In all three paradigms, the aim is to encode the structural and physicochemical features of molecules as a fixed-length, multi-dimensional vector.

Fingerprinting algorithms typically encode molecular structures as sparse binary vectors, with common sizes being 1024 or 2048<sup>1</sup>. While fingerprinting algorithms vary with respect to how they encode 2D molecular graphs as a binary array, their high-level operation is the same: encode substructures as integers and hash these integers into an index that is used to mark the presence of the corresponding feature. Implicitly, 0-valued entries of the vector encode the absence of one or many features that map to the associated index. The most commonly used fingerprinting algorithms are extended connectivity fingerprints (ECFPs)<sup>6</sup>, functional class fingerprints (FCFPs)<sup>6</sup>, atom pair fingerprints<sup>8</sup>, and topological torsion fingerprints<sup>7</sup>.

In contrast to fingerprints, molecular embeddings aim to capture features from molecules that are learned by deep neural networks as part of a supervised<sup>10</sup> or self-supervised training objective<sup>11</sup>. Rather than explicitly crafting algorithms with inbuilt cheminformatic and graph-theoretic heuristics, the objective in molecular representation learning is to select loss functions that lead to an emergent representation of molecular structure and properties.

Much recent work has focused on benchmarking learned molecular representations against classical descriptors on downstream tasks such as property prediction<sup>12,13,14</sup>. Such studies have revealed that classical representations often match or outperform state-of-the-art molecular embedding models, while maintaining improved interpretability and computational eficiency<sup>12,13,14</sup>. Additionally, several studies have investigated the properties of molecular representations more directly, such as whether they capture stereoelectronic properties<sup>15</sup> and activity clifs<sup>16</sup>.

Absent in the literature is a systematic and comprehensive analysis of the quality of learned and classical representations across multiple axes of similarity and application areas. Consequently, many fundamental questions remain unanswered, such as how often fingerprints fail to uniquely encode structurally distinct molecules in ultra-large libraries, how well molecular representations can distinguish structurally close and distant molecules, to what extent similarity metrics capture three-dimensional shape, how closely representations place molecules with similar ADMET activity, and whether clustering molecules using current representations causes data leakage. We sought to address these questions by assessing a broad selection of representations across three axes of similarity: graph, shape, and chemical.

Graph similarity pertains to how well a representation captures the two-dimensional topology of a molecule, and whether it can distinguish any two given molecules. Vector-based measures of graph similarity are the foundation underlying ligand-based virtual screening<sup>1</sup>, database retrieval<sup>17</sup>, clustering<sup>18</sup>, and dataset splitting<sup>19,20,21</sup>.

Shape similarity relates to the degree of overlap in the three-dimensional conformation of molecules, including the distribution of pharmacophoric properties. Perception of molecular shape has been shown to be advantageous in ligand-based virtual screening<sup>22</sup>, particularly with respect to scafold hopping<sup>23</sup>.

Finally, chemical similarity captures the notion that two molecules may have similar activity for a given target or endpoint without necessarily appearing structurally similar by their molecular graphs<sup>24</sup>. In drug discovery, often what is desired is not a measure of shape or graph similarity, but rather a similarity metric that directly facilitates the retrieval of molecules with similar ADMET properties or which bind to the same target.

Here we evaluate multiple axes of similarity across four fingerprinting algorithms and nine diverse molecular embedding models, spanning CLMs, GNNs, and graph transformers (Table 1). We show that despite being a standard for virtual screening, molecular fingerprints display degeneracy rates over large chemical spaces between 2.2% and 40%. Further, we demonstrate that many learned molecular representations fail to capture diferences in graph topology, causing them to place highly structurally dissimilar molecules close in representation space, and highly similar molecules far apart. While we found that fingerprints generally better correlate with an NP-hard measure of graph similarity, we show that current ECFP-based clustering practices cause structural leakage, with implications for dataset splitting and the evaluation of machine-learning models trained on molecular data. Indeed, a general observation was that while fingerprints generally outperformed learned representations across each evaluation, neither class of representation can reliably perceive graph, shape or chemical similarity, with downstream consequences for molecular clustering, virtual screening, dataset splitting and property prediction.

## Results

## Fingerprints are degenerate in ultra-large libraries

Due to their fixed-length and discrete nature, molecular fingerprints display degeneracy, a property wherein structurally distinct molecules map to identical representations. Degeneracy has been reported in previous literature on fingerprints<sup>33,34</sup>. However, little follow-up has been conducted on the scale of this phenomenon. To date, the largest analysis of fingerprint degeneracy examined only one fingerprint type for a sample of 1.4 million molecules<sup>35</sup>. However, virtual screening eforts are now routinely conducted over ultra-large libraries<sup>36,37</sup>, with molecules numbering in the tens of billions. Such ‘active learning’ or ‘deep docking’ studies compute billions of fingerprints to accelerate structure-based approaches<sup>37,38</sup>. Absent in the literature is a systematic examination of the scale of fingerprint degeneracy in real-world virtualscreening libraries across fingerprint types, as well as a taxonomy of real molecular substitutions made by medicinal chemists that may lead to degenerate representations.

To address these questions, we generated fingerprints for the entirety of the Q4 2025 Enamine REAL database of 10.8 billion molecules. After pre-processing (Methods), 10.3 billion molecules remained. Each fingerprint was generated for every molecule twice: once with a chirality flag that enforces the inclusion of stereochemical information and

Table 1 | Molecular representations used in this study
<table><tr><td colspan="7">Molecular embedding models</td></tr><tr><td>Model MolBERT11</td><td>Model type Params CLM</td><td>87M</td><td>1.3M</td><td>Train set size Pre-training objective Masked language modelling, Multi-task loss</td><td>Loss function</td><td>Stereo-aware No</td></tr><tr><td></td><td>(Transformer)</td><td></td><td></td><td>SMILES equivalence, descriptor prediction</td><td></td><td></td></tr><tr><td>ChemBERTa25 CLM</td><td>(Transformer)</td><td>3.4M</td><td>77M</td><td>Masked language modelling</td><td>Categorical cross-entropy</td><td>No</td></tr><tr><td>MolFormer26 MolCLR27</td><td>CLM (Transformer) GNN</td><td>44.4M</td><td>1.1B</td><td>Masked language modelling Contrastive alignment</td><td>Multi-class cross-entropy Scaled</td><td>No</td></tr><tr><td></td><td></td><td></td><td></td><td>following atom masking, bond deletion, and subgraph contrastive loss removal</td><td>cross-entropy</td><td>Yes</td></tr><tr><td>CheMeleon28</td><td>GNN</td><td>9.3M</td><td>1M</td><td>Prediction of 1,613 molecular Masked mean descriptors</td><td>squared error</td><td>Yes</td></tr><tr><td>MiniMol10</td><td>GNN</td><td>10.3M</td><td>6M</td><td>Supervised regression and classification</td><td>Weighted MAE, BCE, HCE sum</td><td>No</td></tr><tr><td>UniMol29</td><td>Graph transformer</td><td>47.3M</td><td>209M</td><td>3D position recovery and masked atom prediction</td><td>Smooth L1 and cross-entropy</td><td>Partial</td></tr><tr><td>COATI30</td><td>Transformer- GNN hybrid</td><td>12.6M</td><td>140M</td><td>Contrastive alignment of SMILES and 3D molecular point clouds</td><td>Contrastive loss and autoregressive cross-entropy</td><td>No</td></tr><tr><td>MolE31</td><td>Graph transformer</td><td>86M</td><td>842M</td><td>Atom environment prediction and supervised fine-tuning on 1,310 ChEMBL readouts</td><td>Multitask binary No cross-entropy</td><td></td></tr><tr><td colspan="7">Molecular fingerprints</td></tr><tr><td>Fingerprint Atom Pair8</td><td>Type</td><td>Size</td><td></td><td>Environment Information encoded Pairs of atom environments</td><td>Implementation Stereo-aware RDKit atom-pair Optional</td><td></td></tr><tr><td>Topological</td><td>Topological</td><td>1024 bits</td><td>Linear paths</td><td>up to 30 bonds and their topological separation Quadruplets of four linearly</td><td>fingerprint</td><td></td></tr><tr><td>Torsion</td><td>Topological</td><td>1024 bits</td><td>Linear 4-atom sequences</td><td>bonded atoms and their atom environments</td><td>RDKit topological torsion fingerprint</td><td>Optional</td></tr><tr><td>ECFP46</td><td>Circular</td><td>1024 bits</td><td>Radius 2 substructures</td><td>Local atom-centred environments based on atomic connectivity</td><td>RDKit Morgan fingerprint</td><td>Optional</td></tr><tr><td>FCFP46</td><td>Circular</td><td>1024 bits</td><td>Radius 2 substructures</td><td>Local atom-centred environments based on pharmacophoric features</td><td>RDKit Morgan fingerprint with feature invariants</td><td>Optional</td></tr><tr><td colspan="7">Physicochemical descriptors</td></tr><tr><td>Descriptor RDKit32</td><td>Type Physico- chemical</td><td>Features 217</td><td>Input 2D molecular graph</td><td>Topology, charge-related properties,</td><td>RDKit CalcMolDescriptors</td><td>No</td></tr></table>

“Stereo-aware” indicates whether stereochemical information is explicitly encoded in the molecular representation or made available during pre-training. For fingerprint representations, "Environment" refers to the structural environment encoded by each fingerprint feature. The number following ECFP and FCFP fingerprints is equal to twice the radius set as a parameter.

once without.

Our analysis revealed that FCFP4 fingerprints were the most susceptible to degeneracy, with 40.5% and 32.7% of molecules being degenerate under this fingerprint for the achiral and chiral versions, respectively (Fig. 1a). This was followed by topological torsion (achiral: 15.4%, chiral: 4.3%), atom pair (achiral: 12.3%, chiral: 5.6%), and finally ECFP4 (achiral: 13.2%, chiral: 2.2%). Degenerate molecules include stereoisomers that neither parameterization of the fingerprints can distinguish<sup>39</sup>, as well as structurally distinct molecules. We find the rates of degeneracy to be surprisingly high, with even the lowest rate of 2.2% for chiral ECFP4 to be equivalent to 225 million molecules with at least one structurally distinct partner yielding an identical fingerprint.

An inverse relationship was observed between the size of degenerate groups of molecules and their frequency (Figs. 1b–e), with maximum degenerate group sizes of 3,974 (chiral: 2,196), 64 (chiral: 31), 64 (chiral: 24) and 136 (chiral: 69) for FCFP4, ECFP4, atom pair and topological torsion fingerprints, respectively. Group sizes are greater for achiral fingerprints due to the high number of stereoisomers that difer only by toggling stereobonds, which the chirality-aware fingerprints are able to distinguish more frequently.

Both ECFPs and FCFPs have been used in ultralarge-scale virtual screening of the Enamine REAL space, often with chirality disabled<sup>36,38,40</sup>. For example, one recent study screened 37 billion molecules from the Enamine REAL space using achiral FCFPs<sup>38</sup>. Despite the successes of these studies in reporting hits, many hundreds of millions to billions of molecules necessarily received identical predictions despite being structurally and physicochemically distinct, conflating not just diferences in afinity, but also binders and non-binders.

We provide a hand-crafted taxonomy of the core failure modes we observed in the sets of degenerate molecules that exist in ChEMBL<sup>41</sup> and Enamine. Fig. 2 highlights these for ECFP4, atom pair, and topological torsion fingerprints. Among the most prominent failure modes were carbon chain elongation, carbon chain shifting, ring expansion, substructure duplication, and substituent relocation. While less frequent, Fig. 2 shows that it is possible for all three fingerprints to be degenerate for the same molecular pairs. Thus, simply ensembling fingerprints in the hopes that this will generate enough features to distinguish all molecular pairs is not a panacea for degeneracy.

We provide constructions demonstrating how several of these modifications can be made to arbitrary molecules satisfying certain criteria to create degenerate analogues in Supplementary Figs. 1-8. As these examples are sourced from ChEMBL and Enamine, these constructions show that fingerprints are insensitive to molecular modifications that medicinal chemists regularly make. Further, these modifications can lead to material diferences in activity. We show three examples from ChEMBL in Fig. 3, where even simple changes in linker length can result in 180-fold changes in potency. Across approximately 40,000 groups of ECFP4-degenerate molecules with measurements in the same assay, approximately half showed a maximum activity diference of at least two-fold, and more than 5,000 displayed greater than a ten-fold change (Fig. 3d).

## No representation captures topology robustly

A primary objective of molecular representations is to capture the similarity between molecular graphs. It has previously been shown that Tanimoto similarity may provide radically diferent estimations of structural similarity between molecular pairs when compared with human intuition<sup>45</sup>. Moreover, a recent study by Kretschmer et al<sup>46</sup> provided examples of the diference between Tanimoto similarity and the weighted maximum common edge substructure (MCES) distance, an NP-hard graph similarity metric. We modified this metric for use as a measure of ground-truth graph similarity (Methods).

Denoted d<sub>MCES</sub>w ∈ [0, 1], our metric indicates the weighted cost of deleting bonds until two molecules are reduced to a shared bond structure, normalized by the worst case of deleting all bonds. Extended Data Fig. 1a provides an example calculation of the metric for two molecules. After calculating pairwise d<sub>MCES</sub>w values using the MCES dataset (Methods) computed by Kretschmer et al<sup>46</sup>, we quantified the Spearman correlation with Tanimoto and cosine similarities (Extended Data Figs. 1b–m) over the representations given in Table 1.

Atom pair fingerprints were the representation most strongly correlated with graph topology, with a Spearman correlation of -0.82, the next closest being topological torsion fingerprints with a Spearman correlation of -0.73. The ECFP4 representation was significantly less correlated with global graph shape, with a Spearman correlation of -0.63. MolFormer was the learned representation most correlated with global graph topology, with a correlation of -0.72. However, 5 of 9 learned molecular representations displayed correlations greater than -0.5, indicating weak alignment with global graph topology. Indeed, with a Spearman correlation of 0.1, MolE cosine similarity proved to be efectively uncorrelated with d<sub>MCES</sub>w<sub>norm</sub>. Marginal plots for the representations also reveal a diference

a  
![](images/d5ef26a98df9e8ac11d168550c10ad42c3bced9583ed37130aaeca70cf651f81.jpg)

![](images/1c5ed715bf9f221f56626302149e7b832768393d606fb519b2e6f910e5e418c4.jpg)

![](images/185f2bdc4f9abaa082ccbafa76bcd3160ad64357e7cc8d77e79f05d5a5d0f213.jpg)

d  
![](images/c85a744d4687844ad182cf9ecf6ca34e9a6f9d469d4223335e7fb1bcf182097f.jpg)  
Fig. 1 | Degeneracy rates for molecular fingerprints across the filtered 10.3 billion molecule Enamine REAL space. (a) Bar chart showing the degeneracy rates for each of the four tested fingerprint types. A molecule is considered to be degenerate if it participates in a group of two or more non-identical

e  
![](images/1a44a753741a768a23d565eccdded8ab4ec5fd191f161951160b83e1423d89c7.jpg)  
molecules that map to the same fingerprint. (b-e) Group sizes for sets of degenerate molecules across FCFP4, topological torsion, ECFP4, and atom pair fingerprints, respectively. Parameter settings for the fingerprints are given in Table 1.

![](images/4a30e3c6888bc1cdcc610a035dc864663d6c133a02884b6493a23ae4fe8d042c.jpg)  
the fingerprint type(s) for which degeneracy holds, while molecules with the same number have identical fingerprints. All examples are sourced from Enamine and ChEMBL. The Venn diagram at center indicates how many consistent and repeatable failure modes were found for each fingerprint type and combination.

Fig. 2 | Degeneracy failure modes across three fingerprint types. The figure depicts a nonexhaustive set of transformations that can be applied to preserve fingerprint identity across ECFP4, atom pair, topological torsion fingerprints, and every intersection between them. Each letter label denotes in distribution shape between molecular fingerprints and learned embeddings. Tanimoto similarities for the MCES dataset are right-skewed, while all of the learned representations are left-skewed, bimodal, collapsed, or normal, indicating a greater tendency toward ranking molecules as similar.

![](images/729b4cc9e0d7f669edcc0b874ef1a2b65811dc8f6642406d7003bdcc7a8f78e6.jpg)  
Fig. 3 | Activity diferences for degenerate molecules in ChEMBL. (a) δ-opioid receptor agonists indistinguishable by all fingerprints<sup>42</sup>, (b) Amgen TRPV1 antagonists indistinguishable by atom pair fingerprints<sup>43</sup>. (c) Antitrypanosomals indistin-  
guishable under ECFP4<sup>44</sup>. Each example pair is published within one paper and assayed under the same experimental conditions. (d) Number of chiral, 1024- bit ECFP4 degenerate groups in the same ChEMBL assay by maximum within-group fold change.

Fig. 4 provides examples of poor calibration of cosine and Tanimoto similarities with respect to the unnormalized weighted MCES distance, $d _ { M C E S } ^ { w } ,$ , for each of the 12 molecular representations. For the classical representations (Figs. 4j–l), these examples highlight the key flaw of binary fingerprints: repeated occurrence of the same substructures in one molecule with respect to a reference allows two molecules of very diferent sizes to appear highly similar. We find that across learned molecular representations (Figs. 4a–i), molecules with highly dissimilar scafolds are ranked as considerably more similar than molecules which difer by a handful of bonds. For example, Fig. 4d shows that CheMeleon may assign a cosine similarity of 0.994 to two molecules that difer in graph structure by a $d _ { M C E S } ^ { w }$ of more than 50, while two near-identical molecules with a distance of 4 are assigned a cosine similarity of 0.494. This behaviour suggests that current learned representations do not capture graph shape robustly, and for applications in which perception of graph shape is important, such as ligand-based virtual screening, these models are unlikely to be as suitable as classical fingerprints.

## Clustering ECFP4 fingerprints leads to leakage

A common task in small-molecule machine learning is clustering, a process which entails assigning structurally related molecules to groups, typically on the basis of their molecular fingerprints<sup>18,20,21</sup>. The most common application of this is in dataset splitting, wherein we desire to assign distinct clusters of molecules entirely to train or test, so as to minimise structural data leakage.

The most popular clustering method in smallmolecule machine learning is the Butina algorithm<sup>18</sup> a greedy optimization algorithm that iteratively creates clusters from the molecule with the greatest number of unassigned neighbours with distances under a given threshold. While the Butina algorithm can operate over any distance matrix, a widespread practice is to use Tanimoto similarities over ECFP4 fingerprints<sup>47,48</sup>. We therefore focused our analysis on this representation.

To investigate whether Butina clustering using ECFP4 Tanimoto similarity leads to structural data leakage, we clustered ∼ 20 thousand molecules from the MCES dataset using a range of Tanimoto thresholds from 0.4 to 0.8. Sampled from a range of molecular databases and datasets, the MCES dataset is ECFP4 chemically diverse, as shown in Fig. 5a.

Our analysis revealed that at both high and low similarity thresholds, distinct clusters often contain highly structurally similar molecules. Fig. 5h shows an example network of close MCES neighbours split across diferent clusters. The example uses a conservative Tanimoto similarity threshold of 0.4, which is widely used as a cutof for molecular novelty<sup>20,49</sup> and Butina clustering<sup>47,48</sup>. Each node represents a nonsingleton cluster, while edges indicate that at least one pair of molecules split between the two incident clusters are highly similar. A perfect clustering would show a completely disconnected network, indicating no examples of highly structurally similar molecules split between diferent clusters, whereas the example shows over 600 edges. Fig. 5n highlights several examples of highly-similar molecules that are split between distinct clusters by the algorithm. Due to the moderate correlation of ECFP4 Tanimoto similarity with global graph structure, it is not possible to use Tanimoto similarity to diagnose the issue. For example, each pair in Fig. 5n has a similarity below the threshold of 0.4, with one clearly highly structurally similar pair having a similarity of 0.276.

![](images/aa2d5e373a55aa9957bfbd2e7322d4b19bedfef5e0f93044eab3f80a8b1c2f7f.jpg)  
Fig. 4 | Disagreement between MCES and similarities defined over molecular representations. (a–l) Poor calibration of cosine similarities (a–i) and Tanimoto similarities (j–l) with respect to global molecular graph topology. Examples for each model are divided into two: the top molecular pair  
shows high representation similarity and high MCES distance, while the bottom pair shows low relative representation similarity and low MCES distance. Due to their extreme cosine similarity distributions, examples for the MolCLR (a) and COATI (b) models were rescaled using min-max percentile scaling.

![](images/51f482e037470fbeedc3baa2ff91f9cc7790572cbf12589f79105596516a9be4.jpg)  
Fig. 5 | Butina cluster leakage. (a) Pairwise Tanimoto similarity distributions for the MCES dataset. (b) Fraction of singleton clusters across Butina thresholds. (c) Fraction of the molecules contained in singleton clusters. (d) Fraction of molecules with $d _ { M C E S _ { n o r m } ^ { w } } ~ \leq ~ 0 . 0 5$ and $d _ { M C E S ^ { w } } ~ < ~ 1 0$ to other molecules in diferent clusters (edges). (e) Fraction of clusters with at least one edge incident to another cluster using the same threshold as in (d). (f) Cluster size distributions across a selected Butina Tanimoto threshold range. (g) Edges per cluster across diferent  
Butina Tanimoto thresholds. (h) Network showing 2,298 clusters using a Butina threshold of 0.4. Each edge represents a pair of molecules assigned to diferent clusters with $d _ { M C E S _ { n o r m } ^ { w } } \leq 0 . 0 5$ and $d _ { M C E S w } < 1 0$ Nodes are scaled by molecule count and coloured according to minimum incident $d _ { M C E S ^ { w } } . \ ( \mathbf { i } \mathbf { - m } )$ Average diference across each pair across physicochemical properties. (n) Example molecular pairs from (h). Each pair contains two molecules from diferent clusters with low MCES distance. $T _ { c }$ denotes the Tanimoto similarity coeficient.

As expected, we found that these structuresimilar analogues also have similar physicochemical properties. Figs. 5i–m show the distribution of average diferences in molecular weight, logP, TPSA, and hydrogen bond donors and acceptors for the pairs. For each distance threshold, the distribution of diferences in the properties of close analogues is modest.

While the number of these pairs decreases with a reduction in the similarity threshold used, this comes at the expense of greatly increasing the cluster sizes and heterogeneity. Figs. 5b–e show the efects on the fraction of singleton clusters, molecule membership to singleton clusters, molecule edge incidence, and cluster edge incidence as the cluster threshold varies.

Distinct from known cluster quality limitations of the Butina algorithm<sup>50,19</sup>, these findings partially explain the observation that models trained on datasets split using the Butina algorithm tend to overestimate virtual screening performance<sup>19</sup>. If Butina splitting is used for train test splits, many structurally similar molecules with near-identical physicochemical properties will be split between train and test sets. We note that many other splitting methods are reliant on clustering fingerprints, such as the $\mathrm { U M A P ^ { 1 9 } }$ split, $\mathrm { D a t a S A I L ^ { 2 1 } }$ , Bi $\mathrm { \dot { \ t B I R C H ^ { 5 1 } } }$ and $\mathrm { L o { - } H i ^ { 2 0 } }$ . It is likely that regardless of optimization objective, all of these methods will sufer the same failure mode as Butina splitting when used on current representations.

## Fingerprints track 3D shape more consistently

To assess the ability of molecular representations to capture three-dimensional shape, we investigated the extent to which 2D similarity scores capture shape and pharmacophoric (color) overlap (Methods).

The results broadly recapitulate the findings of the graph similarity experiments: with the exception of COATI in the case of shape score, fingerprints correlate more strongly with shape and color scores across their similarity range than every embedding (Extended Data Fig. 2a). Despite correlating poorly with 2D graph topology, the performance of

COATI is explained by its contrastive learning objective focused on aligning SMILES strings and 3D point clouds. However, COATI does not show material improvement in correlation strength for shape score $( \mathrm { S p e a r m a n } = 0 . 8 0 )$ over atom pair fingerprints $( \mathrm { S p e a r m a n } = 0 . 7 9 )$ , and displays weaker correlation with color score than all fingerprints.

The extent to which representations correlate with shape and color is highly dependent upon the sampling strategy. Using the rolling Spearman correlation within bins of molecular pairs, Extended Data Figs. 2b–m show that within narrow bands of similarity, none of the representations correlate strongly with shape and color. Rather, it is the general increase in shape and color scores across broad ranges of similarity that contributes to the apparent associations. Thus, use of current representations to rank close analogues for 3D similarity is unlikely to perform well, as suggested by recent work<sup>52</sup>.

## Descriptors better capture ADMET similarity

To evaluate the degree to which molecular representations place molecules with similar activity close together in representation space, we used the Biogen ADMET dataset, a high-quality industrial campaign where all assays were conducted within the same lab<sup>53</sup>. We excluded two representations from our analyses, namely MiniMol and MolE, as they have been trained on molecules with similar or identical structures to those in the Biogen dataset for related or even identical endpoints. Additionally, we added the RDKit physicochemical descriptor vector (PDV) containing 217 features as a further benchmark. For these 11 representations, we considered the median endpoint diference between molecules as the top closest molecular pairs to consider decreased.

All representations show a decrease in average endpoint diference as the number of top-most similar pairs to consider decreases, suggesting that all representations are at least somewhat useful in discriminating properties relevant to ADMET endpoint similarity (Extended Data Figs. 3a–d). While there is variability in the trajectories across the similarity thresholds, we observed that for all four of the endpoints, a simple RDKit physicochemical descriptor vector baseline produced the lowest diference in endpoint value for the top 1,000 pairs. This suggests that learned molecular representations are no better at placing molecules with similar endpoint values closer in representation space than standard cheminformatic descriptors. Indeed, Jaccard similarity matrices show that there is generally low agreement between molecular representations with respect to the most similar molecules (Extended Data Figs. 3e–h).

We conducted a follow-up analysis assessing the association between property landscape roughness and cross-validated model performance on the Biogen dataset. Expanding on previous findings<sup>54,13</sup>, we found that the RDKit PDV induces the smoothest landscapes for 3 of 4 tested endpoints, and that this tracks with lower median RMSE across models for those same endpoints (Extended Data Figs. 4a–c).

## Discussion

Used at almost every stage in the computational drug discovery pipeline, vector-based molecular representations play a central role in small-molecule machine learning. In this work, we demonstrate that both classical molecular fingerprints and learned molecular embeddings sufer from consistent failure modes with respect to their suitability for assessing similarity between pairs of molecules. Further, we show that failure to capture molecular similarity via these representations necessarily has a downstream impact on a broad range of applications in small-molecule machine learning, including virtual screening, clustering, and property prediction.

Our degeneracy analyses over the 10.8B Enamine REAL database revealed that hundreds of millions to billions of structurally distinct molecules are indistinguishable under common fingerprinting algorithms. This has implications for virtual screening of ultra-large libraries; if our models cannot distinguish molecules by their input features, it is not possible for them to predict distinct labels, such as binding afinity or docking scores.

Examination of the failure modes that lead to degeneracy revealed that modifications medicinal chemists regularly make are often imperceptible to all of the most commonly used fingerprinting algorithms. While degeneracy rates would be lower with count-based fingerprints<sup>34</sup>, it has been shown that degenerate representations can also be constructed for these fingerprints using combinatorial enumeration<sup>55</sup>.

Beyond degeneracy, we have shown that both learned and classical molecular representations struggle to capture global graph topology. Cosine similarity between embeddings generated by many state-of-theart learned molecular representation models correlated poorly with the normalized maximum common edge subgraph measure of global graph similarity, revealing that such models are not calibrated to capture the bond structure of molecules. Similarly, while fingerprint Tanimoto similarities generally correlated more strongly with global graph topology, examples abound of unintuitive similarity values generated by these algorithms.

Our findings reveal the uneven nature of the capabilities of molecular representations, highlighting the necessity of a new wave of small molecule encodings, as well as evaluation protocols not solely predicated on downstream predictive performance.

## Methods

## Enamine degeneracy analysis

The 10.8B Enamine REAL database was used to conduct degeneracy analyses for ECFP4, FCFP4, atom pair and topological torsion fingerprints. Files were downloaded from the Enamine website<sup>56</sup>. All molecules in the REAL database are compliant with Lipinski’s rule of five and Veber criteria: molecular weight ≤ 500, SlogP ≤ 5, hydrogen bond acceptors ≤ 10, hydrogen bond donors ≤ 5, rotatable bonds ≤ 10, and topological polar surface area ≤ 140. As provided, the REAL database was divided into 10 files, with molecules separated into heavy atom count intervals. Each file contains approximately 1B SMILES strings. SMILES strings are given in a CXSMILES format, which allows for the specification of additional molecular properties through the inclusion of metadata at the end of each string. One such feature is enhanced stereochemistry. Due to the inability of RDKit fingerprinting methods to utilize enhanced stereochemistry, we observed that these additional molecular annotations may lead to the inclusion of the same SMILES string multiple times in each file, trivially increasing the degeneracy rate. We therefore stripped all CXSMILES annotations and deduplicated the resulting SMILES strings within each file. This led to approximately 532 million molecules being removed from the dataset.

Naively calculating a pairwise Tanimoto similarity matrix over the entire pre-processed 10.3B REAL database would be computationally intractable, as the resulting matrix would contain ∼ 10<sup>20</sup> entries. Thus, to calculate degeneracy rates for each fingerprint type, we employed a four-stage process. Given a fingerprint type, we first generated fingerprints for the entire REAL database. We then hashed all fingerprints into one of 1,024 possible files. By selecting a hash function that approximately uniformly distributes the fingerprints, we ensured that each file contained roughly 10M fingerprints. Critically, given a deterministic hash function, all identical fingerprints are guaranteed to be in the same file. The final stages were to sort each file, count adjacent runs of identical fingerprints, and sum these across files to calculate

the total degeneracy rate.

Fingerprints were generated using RDKit via the rdFingerprintGenerator utility within the Chem module. All fingerprints were generated with a hashed vector length of 1024. ECFP and FCFP fingerprints were set with radius 2, while atom pair and topological torsion fingerprints were generated with default parameters.

## ChEMBL degeneracy analysis

We conducted our degeneracy analysis of ChEMBL using the ChEMBL37 SQLite download<sup>57</sup>. We extracted activity data only from assays with a non-empty journal field and a listed doi, and only compared the activity data of molecules listed within the same assay. Where a molecule had multiple readouts for the same assay, the median value was taken as representative. We excluded measurements with non-equality quantifiers, such as $\mathrm { I C } _ { 5 0 } < 2 0 \mathrm { n M }$ , and only retained activites reported with an equality relation. We did not de-duplicate degenerate groups. Thus, the counts given in Fig. 3d include cases in which a deposition associated with a paper performed multiple distinct assays for the same degenerate molecules. Similarly, distinct depositions may have assayed the same degenerate molecules under a diferent set of experimental conditions or for a diferent assay. These cases too are retained in the counts. 1024 bit ECFP4 fingerprints with chirality enabled were used.

## Graph similarity

We evaluated graph similarity using the weighted maximum common edge subgraph (MCES) distance as defined by Kretschmer et al<sup>46</sup>. Given two graphs $G _ { 1 } = ( V _ { 1 } , E _ { 1 } )$ and $G _ { 2 } = ( V _ { 2 } , E _ { 2 } )$ , their Maximum Common Edge Subgraph (MCES) is a graph $G _ { c } = $ $( V _ { c } , E _ { c } )$ , where

$$
V _ { c } \subseteq V _ { i } , \qquad E _ { c } \subseteq E _ { i } , \qquad i \in \{ 1 , 2 \} ,
$$

such that $| E _ { c } |$ is maximal. We define the MCES distance between the two graphs as

$$
d _ { \mathrm { M C E S } } ( G _ { 1 } , G _ { 2 } ) = | E _ { 1 } | + | E _ { 2 } | - 2 | E _ { c } | .
$$

Qualitatively, this can be thought of as the number of bonds that must be removed from both molecules to make them isomorphic. In particular, Kretschmer et al use a weighted version of the MCES, wherein weights are assigned to bonds to represent the magnitude of their contribution to the overall weight of the molecular graph. Single bonds are assigned a weight of 1, aromatic bonds a weight of 1.5, double bonds a weight of 2, and triple bonds a weight of 3.

Mathematically, let $G _ { 1 }$ and $G _ { 2 }$ be molecular graphs such that

$$
M _ { i } = ( G _ { i } , a _ { i } , b _ { i } ) , \qquad i \in \{ 1 , 2 \} ,
$$

where $a _ { i } ~ : ~ V _ { i } ~ \to ~ \Sigma$ denotes the atom type and $b _ { i } : E _ { i }  \{ 1 , 1 . 5 , 2 , 3 \}$ denotes the bond type. The weighted MCES distance is

$$
d _ { \mathrm { M C E S } ^ { w } } ( M _ { 1 } , M _ { 2 } ) = \sum _ { e \in E _ { 1 } } b _ { 1 } ( e ) + \sum _ { e \in E _ { 2 } } b _ { 2 } ( e ) - 2 \sum _ { e \in E _ { c } } \operatorname* { m i n } \{ b _ { 1 } ( e ) , b _ { 2 } ( e ) \}
$$

where the min operator is used to select the minimum bond weight in instances wherein bonds with diferent weights are matched. We note that as defined by Kretschmer et al<sup>46</sup>, the MCES distance between two molecules does not account for molecular size. While perhaps more qualitatively intuitive, thresholds based on this metric are not practical for assessing the similarity of molecules. This is because the metric does not account for the fact that a bond distance of d should be less consequential the greater the number of matched bonds the pair of molecules share. To address this issue, we normalize by the maximum weighted MCES distance possible between two molecules, namely the sum of their respective bond weights. This bounds the MCES distance between 0 and 1. Formally,

$$
d _ { \mathrm { M C E S } _ { \mathrm { n o r m } } ^ { \mathrm { w } } } ( G _ { 1 } , G _ { 2 } ) = \frac { d _ { \mathrm { M C E S } ^ { \mathrm { w } } ( G _ { 1 } , G _ { 2 } ) } } { \sum _ { e \in E _ { 1 } } b _ { 1 } ( e ) + \sum _ { e \in E _ { 2 } } b _ { 2 } ( e ) }
$$

## MCES Dataset

We used d<sub>MCES</sub>w distances pre-computed by Kretschmer et al to evaluate the ability of molecular representations to capture global topological similarity between molecules. As released, the MCES dataset consists of pairwise distances for 63,437 molecules. Since computing the MCES between two graphs is NP-hard, the fastest known algorithms may take exponential time in the worst case. Thus, as a compromise, Kretschmer et al provide exact distances up to a cutof of a d<sub>MCES</sub>w of 10. All distances less than 10 are exact values, whereas those greater than or equal to 10 are correct lower bounds on the d<sub>MCES</sub>w distance. Only exact values for pairs with a $\mathrm { d } _ { \mathrm { M C E S ^ { w } } } < 1 0$ were used for the correlation assessment.

Of the 63,437 molecules, 43,443 were sampled from ten commonly-used datasets, namely BACE<sup>58</sup>, BBBP<sup>59</sup>, ClinTox<sup>60</sup>, Delaney<sup>61</sup>, Lipo<sup>62</sup>, SIDER<sup>63</sup>, Tox21<sup>64</sup>, ToxCast<sup>65</sup>, SMRT<sup>66</sup>, and MS/MS<sup>46</sup>. The remaining 19,994 molecules are sampled from the following databases: Kyoto Encyclopedia of Genes and

Genomes<sup>67</sup> (KEGG), Chemical Entities of Biological Interest<sup>68</sup> (ChEBI), Human Metabolome Database<sup>69</sup> (HMDB), Yeast Metabolome Database<sup>70</sup> (YMDB), PlantCyc<sup>71</sup>, MetaCyc<sup>72</sup>, KNApSAcK<sup>73</sup>, Universal Natural Products Database<sup>74</sup> (UNPD), Mass spectrometry Contaminants Database (MaConDa)<sup>75</sup>, Hazardous Substances Data Bank (HSDB)<sup>76</sup>, Super Natural II<sup>77</sup>, COlleCtion of Open Natural prodUcTs (CO-CONUT)<sup>78</sup>, and NORMAN<sup>79</sup>.

Several of the embedding models used in this study impose constraints on the molecules they are able to encode. These constraints include connectivity, atom count, and the type of atoms supported. Moreover, while some models may impose fewer constraints on their inputs, their training datasets often do not include rare elements and disconnected fragments. Thus, to ensure a fair comparison in evaluating graph similarity, SMILES strings were preprocessed to be compliant with the most constrained model. Atom type filtering was performed to remove any molecule featuring any atom not in the set $\{ C , O , N , B r , C l , I , F , S \}$ . Since the MCES dataset contains large and inorganic structures, molecules with a molecular weight ≥ 900 Da were stripped, as were molecules containing no carbon atoms. Finally, molecules failing to featurize due to token limits were also stripped.

## Clustering

We quantified the frequency of adversarial examples of structurally similar molecules in clustering applications using Butina clustering. Some pairwise calculations in the MCES dataset are censored, as they were not computed. We therefore selected the subset of 19,994 molecules for which all pairwise calculations were computed. This enables the use of the Butina algorithm, which requires a complete pairwise distance matrix. We clustered the MCES dataset using the standard implementation of Butina clustering from the RDKit.ML.Cluster package. As ECFP4 is the field-default for clustering applications, we first featurized all molecules in the MCES dataset using this representation. Butina clustering was conducted with thresholds of 0.4, 0.5, 0.6, 0.7, and 0.8.

## Shape similarity

Using 30,000 molecules randomly sampled from ZINC<sup>80</sup>, we constructed pairwise similarity matrices for the 12 molecular representations under study. For each representation, we performed equal-width stratified random sampling across the observed similarity range, selecting up to 100 molecular pairs from each of 10 bins, yielding a maximum of 1,000 pairs per representation. For each sampled molecule, we attempted to generate 10 three-dimensional conformers using RDKit. For each molecular pair, we calculated the maximum shape and color similarity scores across all available conformer combinations, optimising the two scores independently using RDKit’s AlignMol function. Finally, we calculated Spearman rank correlations between each representation’s pairwise similarities and the corresponding shape and color similarity scores, excluding pairs for which valid scores could not be obtained.

## ADMET similarity

To assess whether property landscape roughness correlates with cross-validated model performance on the Biogen dataset, we performed 5-fold cross-validation using generic Bemis Murcko scafold splits. Property landscape roughness was computed for the full dataset across representations, and this score was correlated with regression performance (mean 5-fold RMSE) across each model type (kNN, random forest, SVR, and a gradient boosted regressor). Note that the 5-fold cross validation was performed without hyperparameter tuning, as our objective was to assess the association between property roughness and outof-the-box performance of molecular representations.

The ROGI-XD metric<sup>54</sup> was used to quantify property landscape roughness. ROGI-XD is a scalar metric that quantifies the roughness of a molecular property landscape by measuring how quickly property variation is lost as molecular space is progressively coarse-grained. Given molecular representations $x _ { i } ,$ property values $y _ { i } ,$ and a distance metric $d ( x _ { i } , x _ { j } )$ , molecules are grouped using complete-linkage hierarchical clustering. At each clustering stage, the property value of every molecule in a cluster $C _ { j }$ is replaced by the cluster mean:

$$
{ \bar { y } } _ { j } = { \frac { 1 } { | C _ { j } | } } \sum _ { i \in C _ { j } } y _ { i } .
$$

The standard deviation of the resulting coarsegrained property values, denoted by $\sigma _ { u }$ , is then calculated. The degree of coarse-graining is defined as

$$
u = 1 - \frac { \log N _ { \mathrm { c l u s t e r s } } } { \log N } ,
$$

where N is the total number of molecules. Thus, $u = 0$ corresponds to every molecule forming its own cluster, whereas u = 1 corresponds to all molecules being merged into a single cluster. The ROGI-XD score is defined as the area under the reduction-invariation curve:

$$
\int _ { 0 } ^ { 1 } 2 \left( \sigma _ { 0 } - \sigma _ { u } \right) d u
$$

where $\sigma _ { 0 }$ is the standard deviation of the original property values. A high ROGI-XD indicates that substantial property variation is removed when relatively similar molecules are merged, suggesting a rough landscape with frequent activity clifs, whereas a low value indicates that nearby molecules generally have similar properties and therefore form a smoother landscape. ROGI-XD is particularly suitable for comparing molecular representations because it measures coarse-graining through the fraction of the clustering hierarchy traversed rather than through raw distance thresholds. Raw distances can have very diferent scales and distributions across fingerprints, descriptor sets, and learned embeddings, especially when their dimensionalities difer. By using the normalized variable u, ROGI-XD reduces the efects of distance scaling, distance concentration, and representation-specific distance ranges, allowing a more meaningful comparison of how efectively each representation places molecules with similar property values near one another.

## Acknowledgements

This work was supported by the Biotechnology and Biological Sciences Research Council (BBSRC) and the Reuben Foundation.

## Ethics declaration

The authors declare no competing interests.

## Code and data availability

The code and data for this work are available at the following repository: https://github.com/oxpig/O n-the-limitations-of-molecular-representat ions.

## Author contributions

A.B, F.I, and C.M.D designed the study. A.B conducted the experiments and analysed the results. F.I and C.M.D supervised the project. All authors discussed the results and contributed to the interpretation. A.B wrote the manuscript with editorial input from all authors.

## References

[1] Laurianne David et al. “Molecular representations in AI-driven drug discovery: a review and practical guide”. In: Journal of Cheminformatics (2020). url: https://doi.org/10.1186/ s13321-020-00460-5.

[2] David Weininger. “SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules”. In: Journal of Chemical Information and Computer Sciences (1988). url: https://doi.org/10.1021/ ci00057a005.

[3] Hector Flores-Hernandez and Emmanuel Martinez-Ledesma. “A systematic review of deep learning chemical language models in recent era”. In: Journal of Cheminformatics (2024). url: https : / / doi . org / 10 . 1186 / s13321-024-00916-y.

[4] Odin Zhang et al. “Graph Neural Networks in Modern AI-Aided Drug Discovery”. In: Chemical Reviews (2025). url: https://doi.org/10. 1021/acs.chemrev.5c00461.

[5] Danishuddin and Asad U Khan. “Descriptors and their selection methods in QSAR analysis: paradigm for drug design”. In: Drug Discovery Today (2016). url: 10.1016/j.drudis.2016. 06.013.

[6] David Rogers and Mathew Hahn. “Extended-Connectivity Fingerprints”. In: Journal of Chemical Information and Modeling (2010). url: https://doi.org/10.1021/ci100050t.

[7] Ramaswamy Nilakantan et al. “Topological torsion: a new molecular descriptor for SAR applications. Comparison with other descriptors”. In: Journal of Chemical Information and Computer Sciences (1987). url: https://doi.org/10. 1021/ci00054a008.

[8] Raymond E. Carhart, Dennis H. Smith, and R. Venkataraghavan. “Atom pairs as molecular features in structure-activity studies: definition and applications”. In: Journal of Chemical Information and Computer Sciences (1985). url: https://doi.org/10.1021/ci00046a002.

[9] Thanh-Hoang Nguyen-Vo et al. “Molecular representations in bio-cheminformatics”. In: Memetic Computing (2024). url: https://doi. org/10.1007/s12293-024-00414-6.

[10] Kerstin Kläser et al. “MiniMol: A Parameter-Eficient Foundation Model for Molecular Learning”. In: Arxiv (ICML accessible and eficient foundation models for biological discovery workshop) (2024). url: https : / / doi . org / 10 . 48550/arXiv.2404.14986.

[11] Benedek Fabian et al. “Molecular representation learning with language models and domain-relevant auxiliary tasks”. In: Arxiv (NeurIPS machine learning for molecules workshop) (2020). url: https : / / doi . org / 10 . 48550/arXiv.2011.13230.

[12] Jianyuan Deng et al. “A systematic study of key elements underlying molecular property prediction”. In: Nature Communications (2023). url: https : / / doi . org / 10 . 1038 / s41467 - 023 - 41948-6.

[13] Yaëlle Fischer et al. “Deep Learning vs Classical Methods in Potency and ADME Prediction: Insights from a Computational Blind Challenge”. In: Journal of Chemical Information and Modeling (2025). url: https://doi.org/10.1021/ acs.jcim.5c01982.

[14] Dejun Jiang et al. “Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptorbased and graph-based models”. In: Journal of Cheminformatics (2021). url: https://doi. org/10.1186/s13321-020-00479-8.

[15] Rebekah Duke et al. “Evaluating Molecular Similarity Measures: Do Similarity Measures Reflect Electronic Structure Properties?” In: Journal of Chemical Information and Modeling (2025). url: https://doi.org/10.1021/acs.jcim. 5c00175.

[16] Derek van Tilborg, Alisa Alenicheva, and Francesca Grisoni. “Exposing the Limitations of Molecular Machine Learning with Activity Clifs”. In: Journal of Chemical Information and Modeling (2022). url: https://doi.org/10. 1021/acs.jcim.2c01073.

[17] Kenneth Lopez-Perez et al. “Molecular similarity: Theory, applications, and perspectives”. In: Artificial Intelligence Chemistry (2024). url: https://doi.org/10.1016/j.aichem.2024. 100077.

[18] Darko Butina. “Unsupervised Data Base Clustering Based on Daylight’s Fingerprint and Tanimoto Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets”. In: Journal of Chemical Information and Computer Sciences (1999). url: https://doi.org/10.1021/ ci9803381.

[19] Qianrong Guo, Saiveth Hernandez-Hernandez, and Pedro J. Ballester. “UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines”. In: Journal of Cheminformatics (2025). url: https : //doi.org/10.1186/s13321-025-01039-8.

[20] Simon Steshin. “Lo-Hi: Practical ML Drug Discovery Benchmark”. In: Advances in Neural Information Processing Systems 36. 2023. url: https://openreview.net/forum?id= H2Yb28qGLV.

[21] Roman Joeres, David B. Blumenthal, and Olga V. Kalinina. “Data splitting to avoid information leakage with DataSAIL”. In: Nature Communications (2025). url: https://doi.org/ 10.1038/s41467-025-58606-8.

[22] Georgia B. McGaughey et al. “Comparison of Topological, Shape, and Docking Methods in Virtual Screening”. In: Journal of Chemical Information and Modeling (2007). url: https: //doi.org/10.1021/ci700052x.

[23] Thomas S. Rush et al. “A Shape-Based 3-D Scafold Hopping Method and Its Application to a Bacterial Protein-Protein Interaction”. In: Journal of Medicinal Chemistry (2005). url: https://doi.org/10.1021/jm040163o.

[24] Gerald Maggiora et al. “Molecular Similarity in Medicinal Chemistry”. In: Journal of Medicinal Chemistry (2013). url: https://doi.org/10. 1021/jm401411z.

[25] Walid Ahmad et al. “ChemBERTa-2: Towards Chemical Foundation Models”. In: ArXiv (2022). url: https : / / doi . org / 10 . 48550 / arXiv . 2209.01712.

[26] Jerret Ross et al. “Large-scale chemical language representations capture molecular structure and properties”. In: Nature Machine Intelligence (2022). url: https://doi.org/10. 1038/s42256-022-00580-7.

[27] Yuyang Wang et al. “Molecular contrastive learning of representations via graph neural networks”. In: Nature Machine Intelligence (2022). url: https://doi.org/10.1038/s42256-022- 00447-x.

[28] Jackson W. Burns et al. “Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular Descriptors”. In: Journal of Chemical Information and Modeling (2026). url: https : / / doi . org / 10 . 1021 / acs . jcim . 6c01546.

[29] Gengmo Zhou et al. “Uni-Mol: A Universal 3D Molecular Representation Learning Framework”. In: ICLR (2023). url: https://doi.org/10. 26434/chemrxiv-2022-jjm0j-v4.

[30] Benjamin Kaufman et al. “COATI: Multimodal Contrastive Pretraining for Representing and Traversing Chemical Space”. In: Journal of Chemical Information and Modeling (2024). url: https : / / doi . org / 10 . 1021 / acs . jcim . 3c01753.

[31] Oscar Méndez-Lucio, Christos A. Nicolaou, and Berton Earnshaw. “MolE: a foundation model for molecular graphs using disentangled attention”. In: Nature Communications (2024). url: https : / / doi . org / 10 . 1038 / s41467 - 024 - 53751-y.

[32] RDKit. rdkit.Chem.Descriptors.CalcMolDescriptors. rs. url: https://www.rdkit.org/docs/source/ rdkit . Chem . Descriptors . html (visited on [4 08/20/2026).

[33] Cristian Bologa et al. “Descriptor collision and confusion: Toward the design of descriptors to mask chemical structures”. In: Journal of Computer-Aided Molecular Design (2005). url: https : / / doi . org / 10 . 1007 / s10822 - 005 - 9020-4.

[34] Florian Huber and Julian Pollmann. “Count your bits: fingerprint benchmarking to assess broad chemical space representation”. In: Journal of Cheminformatics (2026). url: https : //doi.org/10.1186/s13321-026-01245-y.

[35] Tuan Le et al. “Neuraldecipher – reverseengineering extended-connectivity fingerprints (ECFPs) to their molecular structures”. In: Chemical Science (2020). url: https://doi. org/10.1039/d0sc03115a.

[36] Andreas Luttens et al. “Rapid traversal of vast chemical space using machine learning-guided docking screens”. In: Nature Computational Science (2025). url: https://doi.org/10.1038/ s43588-025-00777-x.

[37] Francesco Gentile et al. “Artificial intelligence–enabled virtual screening of ultra-large chemical libraries with deep docking”. In: Nature Protocols (2022). url: https://doi.org/ 10.1038/s41596-021-00659-2.

[38] James Wellnitz et al. “Enabling Open Machine Learning of Deoxyribonucleic Acid-Encoded Library Selections to Accelerate the Discovery of Small Molecule Protein Binders”. In: Journal of Medicinal Chemistry (2025). url: https: //doi.org/10.1021/acs.jmedchem.5c01972.

[39] Gökhan Tahıl et al. “Stereoisomers Are Not Machine Learning’s Best Friends”. In: Journal of Chemical Information and Modeling (2024). url: https://doi.org/10.1021/acs.jcim. 4c00318.

[40] Domiziana Cecchini et al. “AI-enhanced adaptive virtual screening of large libraries for ligand discovery”. In: Nature Biotechnology (2026). url: https://doi.org/10.1038/s41587-026- 03217-x.

[41] Anna Gaulton et al. “ChEMBL: a large-scale bioactivity database for drug discovery”. In: Nu-

cleic Acids Research (2011). url: https://doi. org/10.1093/nar/gkr777.

[42] Giovanni Loriga et al. “Synthesis and biological evaluation of novel delta (δ) opioid receptor ligands with diazatricyclodecane skeletons”. In: European Journal of Medicinal Chemistry (2013). url: https://doi.org/10.1016/j. ejmech.2013.09.014.

Elizabeth M. Doherty et al. “Novel Vanilloid Receptor-1 Antagonists: 2. Structure-Activity Relationships of 4-Oxopyrimidines Leading to the Selection of a Clinical Candidate”. In: Journal of Medicinal Chemistry (2007). url: https: //doi.org/10.1021/jm070190p.

[44] Christophe Dardonville and Reto Brun. “Bisguanidine, Bis(2-aminoimidazoline), and Polyamine Derivatives as Potent and Selective Chemotherapeutic Agents against Trypanosoma brucei rhodesiense. Synthesis and in Vitro Evaluation”. In: Journal of Medicinal Chemistry (2004). url: https : //doi.org/10.1021/jm031024u.

[45] Darren R Flower. “On the Properties of Bit String-Based Measures of Chemical Similarity”. In: Journal of Chemical Information and Computer Sciences (1998). url: https://doi.org/ 10.1021/ci970437z.

[46] Fleming Kretschmer et al. “Coverage bias in small molecule machine learning”. In: Nature Communications (2025). url: https://doi. org/10.1038/s41467-024-55462-w.

[47] Thomas Blaschke et al. “Memory-assisted reinforcement learning for diverse molecular de novo design”. In: Journal of Cheminformatics (2020). url: https://doi.org/10.1186/s13321-020- 00473-0.

[48] Xiang Yu et al. “Target enhanced 2D similarity search by using explicit biological activity annotations and profiles”. In: Journal of Cheminformatics (2015). url: https://doi.org/10. 1186/s13321-015-0103-5.

[49] Jonathan M. Stokes et al. “A Deep Learning Approach to Antibiotic Discovery”. In: Cell (2020). url: https : / / doi . org / 10 . 1016 / j . cell . 2020.01.021.

[50] Saiveth Hernández-Hernández and Pedro J. Ballester. “On the Best Way to Cluster NCI-60 Molecules”. In: Biomolecules (2023). url: https://doi.org/10.3390/biom13030498.

[51] Kenneth López Pérez et al. “BitBIRCH: eficient clustering of large molecular libraries”. In: Digital Discovery (2025). url: https://doi. org/10.1039/d5dd00030k.

[52] Robin Winter, Julian Cremer, and Djork-Arné Clevert. “Learning Compressed Shape-Aware Molecular Representations for Virtual Screening”. In: Proceedings of the International Conference on Machine Learning. 2026.

[53] Cheng Fang et al. “Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective”. In: Journal of Chemical Information and Modeling (2023). url: https://doi.org/10.1021/acs.jcim. 3c00160.

[54] David E. Graf et al. “Evaluating the roughness of structure–property relationships using pretrained molecular representations”. In: Digital Discovery (2023). url: https://doi.org/10. 1039/d3dd00088e.

[55] Philippe Meyer et al. “Reverse engineering molecules from fingerprints through deterministic enumeration and generative models”. In: Journal of Cheminformatics (2025). url: https: //doi.org/10.1186/s13321-025-01074-5.

[56] Enamine. REAL Database. url: https : / / enamine . net / compound - collections / real-compounds/real-database (visited on 09/29/2026).

[57] ChEMBL. ChEMBL Downloads. url: https: //chembl.gitbook.io/chembl- interfacedocumentation / downloads (visited on 09/29/2026).

[58] Govindan Subramanian et al. “Computational Modeling of β-Secretase 1 (BACE-1) Inhibitors Using Ligand Based Approaches”. In: Journal of Chemical Information and Modeling (2016). url: https://doi.org/10.1021/acs.jcim. 6b00290.

[59] Ines Filipa Martins et al. “A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling”. In: Journal of Chemical Information and Modeling (2012). url: https://doi.org/ 10.1021/ci300124c.

[60] Paul A Novick et al. “SWEETLEAD: an in silico database of approved drugs, regulated chemicals, and herbal isolates for computer-aided drug discovery”. In: PLoS One (2013). url: https: //doi.org/10.1371/journal.pone.0079568.

[61] John S. Delaney. “ESOL: Estimating Aqueous Solubility Directly from Molecular Structure”. In: Journal of Chemical Information and Computer Sciences (2004). url: https://doi.org/ 10.1021/ci034243x.

[62] Mark Wenlock and Nicholas Tomkinson. Experimental in vitro DMPK and physicochemical data on a set of publicly disclosed compounds.

url: https : / / www . ebi . ac . uk / chembl / explore/document/CHEMBL3301361 (visited on 08/21/2026).

[63] Michael Kuhn et al. “The SIDER database of drugs and side efects”. In: Nucleic Acids Research (2016). url: https : / / doi . org / 10 . 1093/nar/gkv1075.

[64] Ruili Huang et al. “Tox21challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs”. In: Frontiers in environmental science (2016). url: https: //doi.org/10.3389/fenvs.2015.00085.

[65] Ann M. Richard et al. “ToxCast Chemical Landscape: Paving the Road to 21st Century Toxicology”. In: Chemical Research in Toxicology (2016). url: https://doi.org/10.1021/acs. chemrestox.6b00135.

[66] Xavier Domingo-Almenara et al. “The METLIN small molecule dataset for machine learningbased retention time prediction”. In: Nature Communications (2019). url: https : / / doi . org/10.1038/s41467-019-13680-7.

[67] Minoru Kanehisa et al. “KEGG: biological systems database as a model of the real world”. In: Nucleic Acids Research (2025). url: https: //doi.org/10.1093/nar/gkae909.

[68] Adnan Malik et al. “ChEBI: re-engineered for a sustainable future”. In: Nucleic Acids Research (2026). url: https://doi.org/10.1093/nar/ gkaf1271.

[69] David S Wishart et al. “HMDB 5.0: the Human Metabolome Database for 2022”. In: Nucleic Acids Research (2021). url: https://doi.org/ 10.1093/nar/gkab1062.

[70] Miguel Ramirez-Gaona et al. “YMDB 2.0: a significantly expanded version of the yeast metabolome database”. In: Nucleic Acids Research (2017). url: https : / / doi . org / 10 . 1093/nar/gkw1058.

[71] Charles Hawkins et al. “Plant Metabolic Network 16: expansion of underrepresented plant groups and experimentally supported enzyme data”. In: Nucleic Acids Research (2024). url: https://doi.org/10.1093/nar/gkae991.

[72] Ron Caspi et al. “The MetaCyc database of metabolic pathways and enzymes - a 2019 update”. In: Nucleic Acids Research (2020). url: https://doi.org/10.1093/nar/gkz862.

[73] Farit Mochamad Afendi et al. “KNApSAcK family databases: integrated metabolite-plant species databases for multifaceted plant research”. In: Plant and Cell Physiology (2011). url: https://doi.org/10.1093/pcp/pcr165.

[74] Jiangyong Gu et al. “Use of Natural Products as Chemical Library for Drug Discovery and Network Pharmacology”. In: PLOS ONE (2013). url: https : / / doi . org / 10 . 1371 / journal . pone.0062839.

[75] Ralf J. M. Weber et al. “MaConDa: a publicly accessible mass spectrometry contaminants database”. In: Bioinformatics (2012). url: https://doi.org/10.1093/bioinformatics/ bts527.

[76] George Charles Fonger et al. “The National Library of Medicine’s (NLM) Hazardous Substances Data Bank (HSDB): background, recent enhancements and future plans”. In: Toxicology (2014). url: https://doi.org/10.1016/j. tox.2014.09.003.

[77] Priyanka Banerjee et al. “Super Natural II–a database of natural products”. In: Nucleic Acids Research (2014). url: https://doi.org/10. 1093/nar/gku886.

[78] Venkata Chandrasekhar et al. “COCONUT 2.0: a comprehensive overhaul and curation of the collection of open natural products database”. In: Nucleic Acids Research (2024). url: https: //doi.org/10.1093/nar/gkae1063.

[79] Valeria Dulio and Jaroslav Slobodnik. “NOR-MAN—network of reference laboratories, research centres and related organisations for monitoring of emerging substances”. In: Environmental Science and Pollution Research (2009). doi: 10.1007/s11356-009-0129-1.

[80] John J. Irwin et al. “ZINC: A Free Tool to Discover Chemistry for Biology”. In: Journal of Chemical Information and Modeling (2012). url: https://doi.org/10.1021/ci3001277.

a  
![](images/d5ac0254b55150910c0cd7fe4385a9040c4091ef3972778cc63195d736ee4e92.jpg)

$$
d _ { \mathrm { M C E S } _ { \mathrm { n o r m } } ^ { w } } ( G _ { 1 } , G _ { 2 } ) = \frac { d _ { \mathrm { M C E S } ^ { w } } ( G _ { 1 } , G _ { 2 } ) } { \displaystyle \sum _ { e \in E _ { 1 } } b _ { 1 } ( e ) + \sum _ { e \in E _ { 2 } } b _ { 2 } ( e ) } = \frac { 9 } { 3 7 } \approx 0 . 2 4 3
$$

![](images/f0cfa5a8a72a376f97a664adbf4528a775d55836522f9a7fe0a2bfd259424471.jpg)

![](images/bddd1154ef459f825e9197665b7d3cd2a937c99bd152968a75075868c5c0ca6f.jpg)

![](images/d1d37fcdbcac673f3d0d65997e1edcdbef96487383c64061e72bb97d102d044b.jpg)

![](images/7d1bd0ade13a2c6835d16898f6cbb9a133a52bbf91f59084ec4e54152ac65cd5.jpg)

![](images/f842e3019425f320dea16da1d3fba2dda70129c4738cbe6aaa5fa933fc7a381a.jpg)

![](images/77552d1f3320ed49aab5528ca394bfe0ed58b470430fa734b5e054c257727edf.jpg)

![](images/6967bcaf776ef8fa625497a78802ab8bf54256b3cb6e516b643d6c608028a019.jpg)

![](images/f2c0947cc739b073d41f33f4073a03ebdcf8e5653fc54dc06ae2caaeb6534ddb.jpg)

![](images/5d41a0ad898be2a1d746ee678e996d907077e97d6f5a214255a1fa87db266b10.jpg)

![](images/85c70648817d64449637c2f60e6c9a407a725cb580368ebd784560919baf11e7.jpg)

![](images/10cbff2868cab9b20309c3e269e508d2928c4b4a8b0d7d6b6e292bc7ad2def42.jpg)

![](images/8ce987b3a6456d9d66e53199aea0830b2f6273ccc5bba453688dc20a0cd70a50.jpg)  
Extended Data Fig. 1 | Graph similarity evaluations. (a) An example of normalized weighted MCES distance between two molecules. (b-m) Hexbin relative density plots with marginal distributions showing the Spearman correlation between Tanimoto similarity  
(b-d) and cosine similarity (e-m) against normalized weighted MCES distance across 12 representations. After preprocessing, a total of 4,348,627 pairwise comparisons are plotted for each hexbin plot. All $d _ { M C E S ^ { w } }$ values $\mathrm { a r e } < 1 0$

![](images/f244b746ec260bd9ea33ec44e025b558e09d7de8f930eae5e11359d31462a098.jpg)

![](images/c9c3da086b978c1dae1496159178c7f8a33f283919ae26ee0d0194cb6a323c71.jpg)

![](images/3bd324a328921740b7ff9d679fdefdf43fd67e2ba168b11bc0ac401d819a3dd3.jpg)

![](images/6736c8ed4adce34aad02b296288ed71c1cbdadf1856540bac2b2ff57a9e9edc8.jpg)

![](images/e8952b2d5224ec2182b91d9078748a16afbfbff8e2ac320713b39dc89bb6b4ae.jpg)

![](images/597503f3f8f1978cf565fb9581e20cf52c237a7a3313b1f04d9fc43b182e9d37.jpg)

![](images/4b7aea4fc8d0ac48417bc69934050999c9803526b3cef9ff3e95d22e88eac10d.jpg)

![](images/d81b38e338543264c057de07f2e55fa33ac988f2ef6fbf223ba7fd3c404d79f0.jpg)

![](images/d079ff09816d03510b1b53a522128bbff9502fd45ab877126426dccf39d224fd.jpg)

![](images/ef8981c5bec171de3d209ce73935417b6c753217f09861bb1dfd27a5f4104f1b.jpg)

Extended Data Fig. 2 | Shape evaluations for molecular representations. (a) Bar chart showing the Spearman correlation between Tanimoto and cosine similarity against maximum pairwise conformational overlap score for randomly sampled molecules from ZINC. Overlap scores are given by RDKit shape and color score returned by the RDKit AlignMol API. For each representation, 1,000 total molecular pairs were uniformly sampled according to the given model’s similarity distribution. Each model’s distribution was divided into ten equal-width intervals and up to 100 pairs were randomly sampled per similarity bin. 10 conformers were generated per molecule and the maximum pairwise score was taken for each pair. (b-m) Rolling shape and color Spearman correlations for each of the 12 representations across binned (200- molecular pairs) intervals with a step size of 25 and shaded regions representing 95% confidence intervals estimated from 300 bootstrap resamples with replacement.

b  
c  
![](images/8b070dbfb5ddc332d823de17e528a65f3b2d351369e0f148774aeba4305c264c.jpg)

![](images/f7aea4b2d641065c083ebf5135ee943cbf2a477fd4d82e0e3b5f600f5c544053.jpg)

![](images/21ec7e95f99135dc203c61222648369fafd14f7bd3c9a391b3a3c44840137025.jpg)

![](images/5c3bf036a899d139d65761c553f658fa26847c5d9326341faa9fb0de81ee2596.jpg)

![](images/058e58ca5dcadc150f4f6eb573bd0bbbb61cbd9ce8f6ab98a8726b843693356a.jpg)

![](images/c801a44970a742946e474b408fb7d091f7be4091c902984bb91c7e070e7f6a42.jpg)

![](images/aa1354fe66882310f33a316ad3b52d940517152505d004f87e34b0839ff3244c.jpg)  
Extended Data Fig. 3 | Chemical similarity evaluations for molecular representations. (ad) ADMET endpoint similarity for the Biogen dataset across similarity thresholds. Each plot shows the median absolute endpoint diference for the top closest

![](images/6ff90707689c759eb419ba2648f81f6eb16c1e2828f97f6bc439e008e2e5ee5c.jpg)  
100,000, 10,000 and 1,000 molecular pairs by representation similarity. (e-h) Jaccard similarity triangles indicating the fraction of molecules that overlap in the top 1,000 of molecular pairs by representation similarity.

![](images/b3c2400254a10309f9bd1d62cca4f4f7bf0458760a5cf89eeed5fb1f93d35e0c.jpg)

Extended Data Fig. 4 | QSAR roughness and performance. (a) ROGI-XD roughness for the representations with the lowest ROGI-XD values. ROGI-XD is computed across all molecules for which endpoint measurements are available. Lower values indicate representations that induce a smoother landscape. (b) Box plots for mean 5-fold RMSE for the top 5 representations across each endpoint and model. (c) Pearson correlation between ROGI-XD roughness and cross-validated RMSE.

# Supplementary Information: On the limitations of molecular representations

Acer Blake   
Department of Statistics   
University of Oxford   
Oxford, UK   
blake@stats.ox.ac.uk   
Charlotte M. Deane   
Department of Statistics   
University of Oxford   
Oxford, UK   
deane@stats.ox.ac.uk   
Fergus Imrie   
Department of Statistics   
University of Oxford   
Oxford, UK   
imrie@stats.ox.ac.uk

1 Supplementary Figures

a  
![](images/30cd016fbbb92b15e0596c7fd467bd35270f7f7e54f4431d3674970dde666b54.jpg)

b  
![](images/b0654edafe241ab4a26a4c37f90df97053c60f6fd0edf74e528717c6ae5cd489.jpg)

c  
![](images/d606d08fccfe4c62505cda6650c55fc10dde0d6329c862a38fccb599d5f0f404.jpg)

d  
![](images/3543041facf75619e73b9a0f73f559533cfd839d3d9428410c3dee454d0e27fa.jpg)  
Supplementary Fig. 1 | Quinoline inversion construction for atom pair fingerprints. The figure shows how the nitrogen atom in a quinoline functional group (a) can be relocated to the same position on the opposite fused ring (b) while maintaining fingerprint identity. R groups here signify that provided the same structure is substituted for both placeholders in subfigures (a) and (b), the resulting molecule will have the same fingerprint. Numerical annotations represent the bond distance from the highlighted nitrogen to each atom. Atom pair fingerprints encode features using a three-tuple of the form $( a _ { 1 } , d , a _ { 2 } )$ where $a _ { 1 }$ and $a _ { 2 }$ are the atom environments of two distinct atoms and d is the bond distance between them. Atom environments are encoded using atomic number, number of π electrons, and degree. None of these values are changed by the relocation of the nitrogen. Further, because of the symmetry of the quinoline functional group, bond distances are matched by the relocation of the nitrogen. Subfigures (c) and (d) show an example pair of degenerate molecules satisfying the rule depicted.

![](images/72753167b676348d8c3642cf9abebd86fc0a4feefcc4f46a8b45bdaf0f25d4e6.jpg)

Supplementary Fig. 2 | Ring expansion for ECFP4 fingerprints. The figure shows how non-aromatic rings can be expanded while maintaining fingerprint identity for ECFP4 fingerprints. R groups here signify that provided the same structure is substituted for both placeholders in subfigures (a) and (b), the resulting molecule will have the same fingerprint. Note that there is no limit as to how large the rings can be made in the cases of subfigures (b) and (d). The maintenance of fingerprint identity can be attributed to the fact that no new radial substructures up to a bond distance of 2 are generated. Note that it is not necessary for rings to be cycloalkanes for ring expansion to be possible. It is possible to replace these with heterocycles in some instances.

a  
![](images/63c5fbeb23f66e6eea6cfe0b14c0c4b67affe517ba13cf71f5a6e830d7bf70bf.jpg)

b  
![](images/c35cf733faa390d5fc0eef648c0ff3fe56f923cd59d08dc446a872197373e693.jpg)

c  
![](images/5bb1169324bb2d2da0b17f33e4def2574c810ad42230724b8f940372ef8d2c7d.jpg)

d  
![](images/cb670d65c8055a3d0f6ac1c14c215bca300dc0a3b6e2fea98e48ccc796a04541.jpg)  
Supplementary Fig. 3 | Ring insertion for ECFP4 fingerprints. The figure shows how additional benzene groups can be inserted into a molecular structure while maintaining fingerprint identity. R groups here signify that provided the same structure is substituted for both structures in subfigures (a) and (b), the resulting molecule will have the same fingerprint. Note that there is no limit as to how many rings may be inserted into the structures of (b) and (d). The maintenance of fingerprint identity can be attributed to the fact that no new radial substructures up to a bond distance of 2 are generated. Ring insertion is a specific case of substructure insertion, wherein molecules degenerate under the ECFP4 fingerprint can be constructed by repeatedly appending repeated substructures to a reference molecule. Provided the attachment points for the repeated substructures do not introduce a new substructure, these molecules will maintain an identical fingerprint.

a  
![](images/74530d60bb484cc5143a33873663cc937b1806230b966218661e1a037c2c5c39.jpg)

b  
![](images/c3fc52a8b628c59d33acbd7793145e32a0d7618755824bcc517b98844603d1ff.jpg)

c  
![](images/3623c905040db7020860ff6d2acebb72d1c7487060ce4159319fd50d5a4b2249.jpg)

d  
![](images/dbafc47470ba7e3c653495de18fe6d215d305d2ff5c451469f13b9e8bd3d2446.jpg)  
Supplementary Fig. 4 | Macrocyclization for ECFP4 fingerprints. The figure shows how a 6- membered heterocycle with heteroatoms at the 1 and 4 positions and an attached side-chain can be resized into a 12-membered macrocycle while maintaining fingerprint identity. R groups here signify that provided the same structure is substituted for the side chains in subfigures (a) and (b), the resulting molecules (subfigures (c) and (d)) will have the same fingerprint. The atoms denoted η<sub>1</sub>, η<sub>2</sub> may be any heteroatoms satisfying valence rules. The maintenance of fingerprint identity can be attributed to the fact that the constructed 12-membered macrocycle maintains the same radius 2 substructures as the 6-membered heterocycle throughout. Indeed, provided the period of repeated heteroatoms and side-chains repeats with the same frequency, the heterocycle can be expanded infinitely.

a  
![](images/a4a56396fe29a37402165f63e2912d3eda27788098b50edb94fb991932bdb391.jpg)

b  
![](images/f536e0683e56ccdbc8409bb9e74865826f43d4a7ffd2523e4ebf06bbdf4ca899.jpg)

c  
![](images/808539c4d13d2ed7f660f03b1afcd0cea058fed468cb9f778273802facfae510.jpg)

d  
![](images/dbd39988201fad4d78df3ac150469e9ff7097339255cb37ab000f02bc0b6bd8e.jpg)  
Supplementary Fig. 5 | Ring swap for ECFP4 and topological torsion fingerprints. The figure shows how two substituted rings can be exchanged with respect to a shared skeleton while maintaining fingerprint identity. R groups here signify that provided the same structures are substituted for both $R _ { 1 }$ and $R _ { 2 }$ in figures (a) and (b), the resulting molecule will have the same fingerprint. The wavy double bonds indicate that the rings may be either aromatic or saturated. Further, the asterisks indicate that the given atoms may be heteroatoms. The maintenance of identical ECFP4 and topological torsion fingerprints can be attributed to the fact that no new radial substructures are introduced (accounting for ECFP4 identity) nor are any new length 4 sequences of atoms (accounting for topological torsion identity). Note that the construction is not limited to two substituted rings being exchanged. In fact, provided the rings follow the same structure as in (a) and (b) any number of rings can be swapped. Figures (c) and (d) provide a concrete example of ring swaps that maintain fingerprint identity.

a  
![](images/aa9220f2f3886013bf683275370385172c919ecff7279df5a5a5d9c65052b8cd.jpg)  
b

![](images/a04918abc140eef09f15dfbe51f5401046b98d00d48ea55c3dac7009241b0cf7.jpg)

c  
![](images/9cae804e49c06706ae7212fc7aeb5167a442ef1fd1eac64588b67a99b878aef1.jpg)

d  
![](images/a41fc05569461c9a66cf73cd5659888d09487236151373dac777dee033b9bf04.jpg)  
Supplementary Fig. 6 | Cage substitution for ECFP4, topological torsion and atom pair fingerprints. The figure depicts how diazatricyclodecane functional groups can be substituted for each other while maintaining fingerprint identity. Subfigure (a) depicts a 9,10-diazatricyclo $[ 4 . 2 . 1 . 1 ^ { 2 } , 5 \cdot$ ]decane ring, while (b) represents a 2,7-diazatricyclo[4 $\cdot 4 . 0 . 0 ^ { 3 } , { } ^ { 8 } ]$ decane ring. The R groups here signify that provided the same structures are substituted for both $R _ { 1 }$ and $R _ { 2 }$ in figures a and b, the resulting molecule will have the same fingerprint. Figures (c) and (d) provide a concrete example of two molecules sharing the same fingerprint with substituted diazatricyclodecane groups.

![](images/36145cc31a11e67bc0de2a3b011a3ee430d4651943404ca7138fb93f2712d503.jpg)  
Supplementary Fig. 7 | Linker extension for ECFP4 and topological torsion fingerprints. The figure shows how a carbon chain can be extended in length while maintaining fingerprint identity. R groups here signify that provided the same structure is substituted for both structures in figures (a) and (b), the resulting molecule will have the same fingerprint. Note that the parity of the length of the original linker may be required to be equivalent for degeneracy to hold. Figures (c) and (d) provide a concrete example of two molecules sharing the same fingerprint with carbon-chain linkers of varying lengths.

![](images/0551bcb878e798defb9a0a5b0cc19c98f822a835ff32836fbd09c742c58ecd8f.jpg)  
Supplementary Fig. 8 | Substituent swap and duplication for ECFP4 fingerprints. The figure shows how ring substituents can be relocated and duplicated on unsubstituted rings within the same molecule while maintaining fingerprint identity. R groups here signify that provided the same structure is substituted for the corresponding structures in subfigures (a) and (b), the resulting molecule will have the same fingerprint. Figures (c) and (d) provide two examples of substituent swaps and duplications for molecules that maintain fingerprint identity.