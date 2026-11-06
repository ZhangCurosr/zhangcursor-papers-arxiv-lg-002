# Towards Multiscale Molecular Representation Learning through Geometry-Based Fragmentation for Polymer Property Prediction

Makoto Takamoto,<sup>∗,†</sup> Chaitanya Sampara,<sup>‡</sup> Viktor Zaverkin,<sup>†,§</sup> Daniel

O˜noro-Rubio,<sup>†</sup> and Mathias Niepert<sup>†,¶</sup>

†NEC Laboratories Europe, Heidelberg, Germany

‡Heidelberg University, Heidelberg, Germany

¶University of Stuttgart, Institute for Artificial Intelligence, Stuttgart, Germany

§Present address: INM – Leibniz Institute for New Materials, Saarbr¨ucken, Germany;

Saarland University, Department of Computer Science, Saarbr¨ucken, Germany; German

Research Center for Artificial Intelligence (DFKI), Saarbr¨ucken, Germany

E-mail: makoto.takamoto@neclab.eu

## Abstract

It is well established that molecular substructures can contribute unequally to chemical and physical properties, with specific motifs or side chains often playing dominant roles. Fragment-based representations have therefore emerged as an efective strategy for incorporating chemically meaningful inductive biases, as exemplified by molecular fingerprints and fragment-based attention mechanisms. However, many existing fragmentation-based machine learning methods rely on predefined chemical rules, which may yield suboptimal fragmentation for previously unseen or chemically unfamiliar structures and often overlook the importance of multiscale molecular representations. To address these limitations, we formulate molecular fragmentation as a

radius-constrained optimization problem inspired by the k-center formulation, where the number of clusters is not fixed a priori but emerges from the specified radius, and propose a geometry-based fragmentation method that partitions arbitrary molecular or material structures using largely geometric information, without relying on predefined chemical fragmentation rules. The method provides flexible control over fragment size, enabling the construction of multiscale fragment representations for scale-adaptive modeling. Building on this framework, we develop a geometric fragment-based attention mechanism that integrates multiscale structural information and incorporate it into a multimodal machine learning model combining p-SMILES representations with three-dimensional (3D) conformational information. Experiments on a benchmark polymer dataset show that incorporating 3D conformational information improves prediction accuracy for several properties over models based exclusively on p-SMILES representations. Explainable artificial intelligence analyses further provide insights into the physical relevance of the learned representations, highlighting the potential of our approach to improve both the accuracy and interpretability of polymer informatics models.

## 1 Introduction

Polymer informatics, which leverages data-driven approaches to predict polymer properties, has advanced significantly in recent years, driven by developments in machine learning, highthroughput computation, and the growing availability of materials data. Early eforts, such as the Polymer Genome project, <sup>1</sup> established foundational datasets comprising hundreds of organic polymers with both computed (e.g., bandgaps and dielectric constants) and experimental properties (e.g., glass transition temperatures), enabling the development of multitask machine learning models. In these initial studies, polymer representations were typically constructed using handcrafted descriptors or fingerprints based on chemical intuition. For instance, neural networks trained on such features demonstrated reasonable predictive performance across multiple properties. <sup>1</sup> However, these approaches rely on expert-designed representations, which can be computationally expensive to generate and may limit their generalizability.

More recently, transformer-based language models have transformed molecular representation learning by directly encoding Simplified Molecular Input Line Entry System (SMILES) strings,<sup>2</sup> reducing the reliance on manual feature engineering. In polymer informatics, this paradigm is exemplified by models such as PolyBERT<sup>3</sup> and TransPolymer.<sup>4</sup> PolyBERT treats canonicalized polymer SMILES (p-SMILES) <sup>5</sup> as chemical sequences and employs a DeBERTa-based architecture <sup>6</sup> <sup>6</sup> pretrained on a large collection of synthetically generated polymers via BRICS fragmentation. <sup>7</sup> Similarly, TransPolymer utilizes a RoBERTa-based encoder<sup>8</sup> with a polymer-specific tokenizer and is pretrained on millions of polymer sequences derived from the PI1M dataset. <sup>5</sup> These approaches demonstrate the efectiveness of large-scale pretraining for learning data-driven molecular representations. Nevertheless, sequence-based representations primarily encode molecular structures as one-dimensional strings and may not explicitly capture the hierarchical and multiscale organization of molecular structures.

It is well established that the full molecular structure does not contribute equally to chemical and physical properties; rather, specific substructures, such as functional groups and side chains, may play dominant roles. Fragment-based representations have therefore emerged as an efective strategy for incorporating chemically meaningful inductive biases into molecular machine learning. Examples include molecular fingerprints and fragmentbased attention mechanisms, which explicitly characterize local and substructural environments.<sup>9–13</sup> Fragment-based approaches, such as FraGAT<sup>14</sup> for small molecules and PolymerGNN<sup>15</sup> for polymers, further demonstrate the potential of substructural information for improving molecular representation learning. Importantly, molecular properties can depend on structural patterns spanning multiple length scales, motivating the use of hierarchical representations that capture both local and global structural information.

Despite these advances, many existing fragmentation methods rely on predefined chemical rules or domain-specific knowledge. For example, BRICS fragmentation<sup>7</sup> identifies chemically meaningful bonds according to predefined decomposition rules. While such approaches can provide chemically interpretable fragments, their applicability may be limited when encountering previously unseen, structurally diverse, or chemically unfamiliar molecular and material systems. Moreover, rule-based fragmentation schemes generally provide limited flexibility in controlling fragment size and do not naturally support systematic construction of multiscale representations. These limitations motivate a more general fragmentation strategy that does not depend on predefined chemical knowledge and can be applied consistently across diverse molecular and material structures.

In this work, we address this challenge by introducing a geometry-based fragmentation method that formulates molecular fragmentation as a radius-constrained optimization problem inspired by the k-center formulation, where the number of clusters is not fixed a prior but emerges from the specified radius. The proposed method selects representative atoms and partitions a molecular or material structure according to geometric proximity, thereby defining fragments using geometric information rather than chemistry-specific fragmentation rules. By varying a pooling scale-length parameter, the characteristic fragment size can be explicitly controlled, enabling the systematic construction of multiscale fragment representations. This geometric formulation provides a general and flexible framework for partitioning arbitrary molecular structures <sup>1</sup> and ofers a natural route toward scale-adaptive molecular representations.

Building on this framework, we develop a geometric fragment-based attention mechanism and incorporate it into a multimodal machine learning model for polymer property prediction. The model combines p-SMILES representations with three-dimensional (3D) conformational information, while the proposed geometric fragmentation mechanism provides multiscale structural information to the 3D representation. In this setting, the multimodal polymer model serves as a showcase application of the proposed fragmentation method rather than as the primary objective of this work. Experiments on a benchmark polymer dataset show that incorporating 3D conformational information improves prediction accuracy for several properties compared with models based exclusively on p-SMILES representations, while the geometric fragment-based branch contributes complementary structural information.

Finally, we employ gradient-based explainable artificial intelligence (XAI) techniques using Captum<sup>16</sup> to examine the learned representations. The attribution analysis indicates that the model assigns substantial importance to chemically relevant functional groups, including carbonyl-, oxygen-, sulfur-, and nitrogen-containing moieties. These results provide evidence that the learned multimodal representations can capture physically meaningful structural information despite being defined without chemistry-specific fragmentation rules. Overall, this work establishes geometry-based fragmentation as a general, domain-independent approach for constructing controllable multiscale molecular representations and demonstrates its potential through a multimodal application to polymer informatics.

## 2 Materials and Methods

## 2.1 Geometric Fragmentation

While several molecular fragmentation methods have been proposed, including BRICS, many rely on predefined chemical rules and may therefore produce suboptimal fragmentation schemes when applied to previously unseen or chemically unfamiliar structures. Moreover, these approaches typically provide limited control over the characteristic fragment length scale, which is important for incorporating chemistry-informed inductive biases into molecular representations, as exemplified by recent multiscale molecular fingerprints. <sup>10,12</sup>

In this section, we introduce a geometry-based fragmentation method that partitions arbitrary molecular or material structures using largely geometric information, without requiring chemistry-specific knowledge. This provides an advantage over existing domaindependent approaches and enables the characteristic fragment size to be flexibly controlled. Consequently, the proposed method facilitates the construction of multiscale fragment representations for hierarchical and scale-adaptive modeling.

![](images/89a5b6164ba546f3d7efade5cf0b0aa239d87b1d5aaa7d164e0f28f80f38093a.jpg)  
Figure 1: Schematic illustration of the geometric fragment of a monomer, $[ { ^ { * } } ] \mathrm { C C } ( \mathrm { O } ) [ { ^ { * } } ]$ . (A) Example of a fragmentation result with $r _ { \mathrm { p o o l } } = 1 \mathrm { \AA }$ and $r _ { \mathrm { c u t o f f } } = 0 . 5 \mathrm { \AA }$ with an explicit description of the representative atoms, $r _ { \mathrm { p o o l } }$ and $r _ { \mathrm { c u t o f f } }$ . Here, $r _ { \mathrm { p o o l } }$ sets the spacing of the representative atoms (centers), whereas $r _ { \mathrm { c u t o f f } }$ is an atom-centered radius that determines fragment membership. Following the ASE neighbor-list convention, two atoms are neighbors when their rad $\mathrm { \ l u s } { - } r _ { \mathrm { c u t o f f } }$ spheres overlap, so the efective pairwise cutof is $2 \times \mathrm { c u t o f f }$ . (B) Illustration of fragmentation at multiple scales, with $r _ { \mathrm { p o o l } } = 1 , 2 , 3 \mathring \mathrm { A }$ and corresponding $r _ { \mathrm { c u t o f f } } = 0 . 5 , 1 . 0 , 1 . 5 \mathrm { \AA }$ , respectively.

Our geometric fragmentation method is based on the k-center problem, <sup>17–19</sup> which seeks a set of representative centers from a set of points while minimizing the maximum distance between any point and its nearest center. By treating the atoms of a molecule as points in geometric space, we select representative atoms as centers. In contrast to the classical formulation, we do not fix the number of centers k in advance. Instead, we fix a pooling radius $r _ { \mathrm { p o o l } }$ and seek a set of centers such that every atom lies within $r _ { \mathrm { p o o l } }$ of its nearest center. This is the radius-constrained form of the k-center problem: writing $d ( i , c )$ for the distance between atom i and center c,

$$
\operatorname* { m i n } _ { C } | C | \quad { \mathrm { s . t . } } \quad \operatorname* { m a x } _ { i } \operatorname* { m i n } _ { c \in C } d ( i , c ) \leq r _ { \mathrm { p o o l } } .\tag{1}
$$

We approximate its solution with the k-Center-Greedy strategy, <sup>20</sup> in which each new center

Algorithm 1 Geometric Fragmentation   
Require: Coordinates $\{ R _ { i } \}$ , atomic numbers $\{ Z _ { i } \}$ , pooling distance $r _ { \mathrm { p o o l } }$ , cutof distance   
$r _ { \mathrm { c u t o f f } }$   
Ensure: Disjoint fragments $\{ F _ { 1 } , \ldots , F _ { m } \}$ covering all atoms   
1: $\mathcal { H }  \{ i : Z _ { i } > 1 \}$ ▷ centers restricted to heavy atoms   
2: Choose an origin $O ; \ c _ { 0 } \gets \arg \operatorname* { m a x } _ { i \in \mathcal { H } } \| R _ { i } - O \| ^ { 2 } ; \ S \gets \{ c _ { 0 } \}$   
3: for each $i \in \mathcal { H }$ do   
4: $D _ { i }  \| R _ { i } - R _ { c _ { 0 } } \| ^ { 2 }$ ▷ squared distance to nearest center   
5: end for   
6: while ma $\mathrm { { \Large { s } } } _ { i \in { \mathcal { H } } } D _ { i } \geq r _ { \mathrm { { p o o l } } } ^ { 2 }$ do ▷ Stage 1: select centers via $r _ { \mathrm { p o o l } }$   
7: $j \gets \arg \operatorname* { m a x } _ { i \in \mathcal { H } } D _ { i } ; \mathcal { S } \gets \mathcal { S } \cup \{ j \}$   
8: for each $i \in \mathcal { H }$ do   
9: $D _ { i }  \operatorname* { m i n } ( D _ { i } , \ \lVert R _ { i } - R _ { j } \rVert ^ { 2 } )$ ▷ update distances   
10: end for   
11: end while   
12: for each center $s \in S$ do ▷ Stage 2: build fragments via $r _ { \mathrm { c u t o f f } }$   
13: $F _ { s } \gets \{ s \} \cup \{ j \neq s : \| R _ { s } - R _ { j } \| \leq 2 r _ { \mathrm { c u t o f f } } \}$   
14: end for   
15: for each atom i with $| { \cal S } _ { i } | > 1$ , where $S _ { i } = \left\{ s \in \mathcal { S } : i \in F _ { s } \right\}$ do   
16: $s ^ { * } \gets \arg \operatorname* { m i n } _ { s \in { \mathcal { S } } _ { i } } \| R _ { i } - R _ { s } \| ^ { 2 }$   
17: $F _ { s } \gets F _ { s } \setminus \{ i \} \quad \forall s \in S _ { i } \setminus \{ s ^ { * } \}$   
18: end for   
19: for each atom $i \not \in \bigcup _ { s \in S } F _ { s }$ do   
20: ${ \mathcal { F } } \gets { \mathcal { F } } \cup \{ \{ i \} \}$ ▷ one singleton per unassigned atom   
21: end for   
22: return $\mathcal { F }$   
is iteratively selected as   
$c _ { n + 1 } = \arg \operatorname* { m a x } _ { i } \operatorname* { m i n } _ { c \in C _ { n } } d ( i , c ) ,$ (2)   
thereby selecting the atom that is farthest from its nearest previously selected center. Centers   
are added until the covering radius max mi $\mathfrak { l } _ { c \in C } d ( i , c )$ falls below $r _ { \mathrm { p o o l } } ,$ , so that the pooling   
distance directly controls the number and spatial separation of the representative atoms,   
which are subsequently used to construct molecular fragments   
To adapt this construction to geometric fragmentation, we introduce two characteristic   
length scales that play distinct roles:   
• A pooling distance $r _ { \mathrm { p o o l } }$ , which sets the spacing between representative atoms (centers)   
and thereby the number and characteristic size of the fragments. It enters only the

center-selection stage, as the stopping criterion of the greedy farthest-point search.

• A cutof distance $r _ { \mathrm { c u t o f f } }$ , which defines, in the fragment-assignment stage, the atomcentered neighborhood used to populate each fragment around its center.

We adopt the atom-centered convention of standard neighbor-list implementations $( \mathrm { e . g . }$ 2 $\mathrm { A S E ^ { 2 1 } } )$ : each atom carries a sphere of radius $r _ { \mathrm { c u t o f f } }$ , and two atoms are counted as neighbors when their spheres overlap. The efective pairwise cutof is therefore $2 \times r _ { \mathrm { c u t o f f } }$ , and with the choice $r _ { \mathrm { c u t o f f } } = r _ { \mathrm { p o o l } } / 2$ used throughout this work, the efective fragment radius $2 \times r _ { \mathrm { c u t o f f } }$ coincides with the center spacing $r _ { \mathrm { p o o l } }$

The pooling distance $r _ { \mathrm { p o o l } }$ serves as the stopping criterion for the greedy center-selection procedure: the iteration terminates when the squared distance of the farthest atom from its nearest selected center falls below $r _ { \mathrm { p o o l } } ^ { 2 }$ . The cutof distance $r _ { \mathrm { c u t o f f } }$ then governs fragment membership: each representative atom is grouped with the atoms within its atom-centered neighborhood, atoms shared between neighborhoods are assigned to the nearest representative, and any atom lying beyond $2 \times r _ { \mathrm { c u t o f f } }$ of every representative forms its own singleton fragment. An algorithmic description of the proposed method is provided in Algorithm 1, while a step-by-step example is presented in the Supplemental Material. By varying $r _ { \mathrm { p o o l } } .$ fragments with diferent characteristic geometric length scales can be generated systematically (see also Figure 1). During the center-selection procedure, hydrogen atoms are excluded from consideration as representative centers because the primary molecular structure is generally defined by heavier atoms, such as carbon.

## 2.2 Geometric Fragment-Based Attention Module

It is well established that the full molecular structure does not always contribute equally to a chemical reaction; rather, specific substructures, such as side chains, may play a dominant role. Motivated by this observation, fragment-based representations have emerged as an efective strategy for incorporating chemically meaningful inductive biases, leading to approaches such as molecular fingerprints and fragment-based attention mechanisms.<sup>14,22</sup> However, many recent fragmentation-based machine learning methods primarily focus on local structural fragments and may overlook the importance of multiscale molecular representations; Although molecular fingerprints such as ECFP<sup>10</sup> and MAP4<sup>12</sup> partially capture structural information at diferent neighborhood scales, they are not specifically designed to construct flexible multiscale molecular representations. Building on this paradigm, we propose a geometric fragment-based attention mechanism that incorporates multiscale structural information while leveraging the scale flexibility of our proposed geometric fragmentation approach, inspired by recent advances in geometry-aware molecular fingerprints. <sup>12,23</sup>

![](images/69ff3c860e6e218da3548b2f10c41b1e433b90b1cf9efa76050132ed7616b8b0.jpg)  
Figure 2: A schematic illustration of the fragment-based attention architecture. The top pathway represents the global information flow, while the lower three pathways correspond to large-, medium-, and small-scale flows, respectively.

In the following, we describe the operation of the proposed geometric fragment-based attention module (see Figure 2). First, the proposed geometric fragmentation is applied to the 3D molecular conformations and atom-level embedding vectors are assigned to each fragment. In our framework, these embeddings are computed using a machine-learned interatomic potential (MLIP) model, that is, MACE-MP. <sup>24</sup> To capture multiscale efects, <sup>12</sup> we employ three fragmentation radii, $r _ { \mathrm { p o o l } } = 1 , 2 , 3 \mathrm { ~ \AA ~ }$ to resolve local atomic composition, localized functional groups, and extended topological frameworks, respectively <sup>2</sup>. For each scale, the fragment-level node embeddings are processed by a self-attention module, followed by a pooling operation that produces a single token per scale. As described in Figure 2, we also retain all the node embeddings to account for global structural information. Finally, these scale-specific tokens are averaged and aggregated into a one-dimensional representation via 1D global depthwise convolution, which is passed to the decoder of the fragmentation branch for downstream tasks.

## 2.3 Multimodal Embedding of Polymer Chemistry

![](images/f3b5332f63d2be4784c13131340df22894cc420d44c64b894c1f444119d96bb7.jpg)  
Figure 3: Overview of the multi-modal machine learning model architecture. (A) Schematic illustration of the overall model structure. The snow mark above 3D conf. Encoder denotes the model weight of the 3D conformation embedding (MACE-MP) is frozen. (B) Schematic illustration of the modality fusion module, where “Norm” denotes layer normalization and $\mathrm { ^ { 6 } A d d ^ { \prime } }$ indicates a residual connection. (C) Schematic illustration of the decoder module, where “SA Pooling” refers to self-attention pooling and $\mathrm { \Omega ^ { 6 6 } M L P ^ { 9 } }$ denotes a multilayer perceptron.

Although the proposed geometric fragmentation method operates solely on geometric information, its application to molecular structures requires access to three-dimensional (3D) conformations. This makes the method particularly suitable for integration with 3D structural encoders, such as machine-learning interatomic potentials (MLIPs). In contrast, many polymer property prediction datasets provide polymer structures primarily in the form of p-SMILES. Moreover, recent SMILES-based foundation models, such as TransPolymer and PolyBERT, have established strong baselines for polymer property prediction. Building on the proposed geometric fragmentation method, we develop a multimodal model that integrates p-SMILES representations with 3D conformational information. The architecture combines a p-SMILES encoder and a 3D molecular conformation encoder, followed by a modality-fusion module that integrates the complementary information from the two representations (see also Figure 3). To leverage established models for each modality, we employ PolyBERT<sup>3</sup> and TransPolymer<sup>4</sup> as p-SMILES encoders and MACE-MP<sup>24,25</sup> as the 3D conformation encoder. The atom-level representations obtained from MACE-MP are subsequently processed by a fragment encoder based on the proposed geometry-based fragmentation scheme, enabling the incorporation of multiscale structural information into the model.

To demonstrate the applicability of the proposed approach to polymer datasets containing only p-SMILES, we generate 3D molecular conformations from p-SMILES using RDKit and provide the resulting structures as additional inputs alongside the original p-SMILES representations. A detailed description of the procedure for generating 3D conformations from p-SMILES is provided in the Supplemental Material. This multimodal architecture serves primarily as a testbed for demonstrating the utility of the proposed geometric fragmentation method, while also enabling an assessment of the complementary predictive value of 3D conformational information beyond p-SMILES-based representations. Note that the weight parameters of the 3D conformation encoder, MACE-MP, are frozen during training to preserve its physically grounded representation of quantum-mechanical interactions, which

is essential for accurately capturing the chemical contributions of individual fragments and their impact on the target properties.

## 3 Results

## 3.1 Multiscale Analysis of Geometric Fragmentation

![](images/7e898c838dd51d73e24744eb677341d3fb2bfa8acfaa38f70ccd93ab4bf7d109.jpg)  
Figure 4: Fragmentation results of a monomer: $[ ^ { * } ] { \mathrm { C C O C } } ( = \mathrm { O } ) { \mathrm { N C C O C C O C C N C } } ( = \mathrm { O } ) { \mathrm { O } } [ ^ { * } ]$

Figure 4 illustrates the fragmentation results obtained using BRICS and the proposed geometric fragmentation method with $r _ { \mathrm { p o o l } } = 1$ , 3, and $5 [ \mathrm { \AA } ]$ for the monomer: [\*]CCOC(= $\mathrm { O ) N C C O C C O C C N C ( = O ) O [ ^ { * } ] }$ . For the geometric fragmentation method, the cutof length $r _ { \mathrm { c u t o f f } }$ is set to half of the pooling length. While BRICS produces relatively small fragments as a consequence of its rule-based decomposition, the proposed geometric fragmentation method generates fragments whose spatial extent is explicitly governed by the prescribed length scale $r _ { \mathrm { p o o l } }$ . This distinction highlights a key advantage of the proposed approach: the characteristic spatial scale of the resulting fragments can be directly controlled without relying on predefined chemical decomposition rules.

As is well-known,<sup>26,27</sup> molecular structures exhibit characteristic features across a broad range of spatial scales. Individual covalent bonds typically have lengths on the order of $1 { - } 2 \textup { \AA }$ , whereas chemically relevant local motifs and functional groups extend over several <sup>˚</sup>A. Longer side chains can span length scales approaching or exceeding 1 nm. These observations motivate a multiscale fragmentation strategy capable of capturing structural information across diferent spatial resolutions. By varying $r _ { \mathrm { p o o l } }$ , the proposed geometric fragmentation method provides a direct and flexible mechanism for constructing fragment representations at diferent characteristic length scales.

## 3.2 Quantitative analysis of multimodal efect

Table 1: Benchmark results on polymer datasets. The performance is measured using root mean-square error (RMSE). TP denotes TransPolymer, and PB denotes PolyBERT. The number written in bold font indicates the best performance for each dataset.
<table><tr><td rowspan="2">Model Name</td><td rowspan="2">Modality</td><td colspan="7">Test RMSE</td></tr><tr><td> $E _ { \mathrm { g c } } ( \mathrm { e V } )$ </td><td> $E _ { \mathrm { g b } } ( \mathrm { e V } )$ </td><td> $E _ { \mathrm { e a } } ( \mathrm { e V } )$ </td><td> $E _ { \mathrm { i } } ( \mathrm { e V } )$ </td><td> $X _ { \mathrm { c } } ( \% )$ </td><td> $\mathrm { E P S }$ </td><td> $n _ { \mathrm { c } }$ </td></tr><tr><td> $\overline { { { \mathrm { T P } \ ( \mathrm { s c r a t c h } ) ^ { a } } } }$ </td><td>SMILES</td><td> $\overline { { 0 . 5 8 ^ { \pm 0 . 0 1 } } }$ </td><td> $\overline { { 0 . 6 0 ^ { \pm 0 . 0 3 } } }$ </td><td> $\overline { { 0 . 3 6 ^ { \pm 0 . 0 3 } } }$ </td><td> $0 . 4 6 ^ { \pm 0 . 0 4 }$ </td><td> $1 8 . 9 ^ { \pm 0 . 5 1 }$ </td><td> $0 . 5 7 ^ { \pm 0 . 1 1 }$ </td><td> $0 . 0 9 9 ^ { \pm 0 . 0 1 5 }$ </td></tr><tr><td> $\mathrm { T P } ( \mathrm { p r e t r a i n e d } ) ^ { a }$ </td><td>SMILES</td><td> $0 . 4 4 ^ { \pm 0 . 0 1 }$ </td><td> $0 . 5 3 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 3 1 ^ { \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 3 9 ^ { \pm 0 . 0 7 } }$ </td><td> $1 7 . 5 ^ { \pm 0 . 6 9 }$ </td><td> $0 . 5 4 ^ { \pm 0 . 0 5 }$ </td><td> $0 . 0 9 8 ^ { \pm 0 . 0 1 7 }$ </td></tr><tr><td>PB+MACE</td><td>SMILES</td><td> $\overline { { 0 . 4 4 ^ { \pm 0 . 0 2 } } }$ </td><td> $0 . 4 8 ^ { \pm 0 . 0 5 }$ </td><td> $\overline { { { \bf 0 . 2 6 ^ { \pm 0 . 0 1 } } } }$ </td><td> $0 . 3 9 ^ { \pm 0 . 0 5 }$ </td><td> $\overline { { 1 7 . 1 ^ { \pm 0 . 5 9 } } }$ </td><td> $\overline { { 0 . 4 9 ^ { \pm 0 . 0 4 } } }$ </td><td> $\overline { { 0 . 0 8 7 ^ { \pm 0 . 0 1 3 } } }$ </td></tr><tr><td> $\mathrm { P B + M A C E }$ </td><td>3D</td><td> $0 . 4 8 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 5 0 ^ { \pm 0 . 0 5 }$ </td><td> $0 . 2 9 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 4 0 ^ { \pm 0 . 0 4 }$ </td><td> $1 7 . 7 ^ { \pm 0 . 9 4 }$ </td><td> $0 . 4 9 ^ { \pm 0 . 0 6 }$ </td><td> $0 . 0 8 5 ^ { \pm 0 . 0 1 6 }$ </td></tr><tr><td> $\mathrm { P B + M A C E }$ </td><td> $\mathrm { S M I L E S + 3 D }$ </td><td> $\mathbf { 0 . 4 2 ^ { \pm 0 . 0 1 } }$ </td><td> $0 . 4 6 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 2 7 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 3 9 ^ { \pm 0 . 0 5 }$ </td><td> $1 7 . 1 ^ { \pm 0 . 4 8 }$ </td><td> $\mathbf { 0 . 4 7 ^ { \pm 0 . 0 5 } }$ </td><td> $0 . 0 8 5 ^ { \pm 0 . 0 1 2 }$ </td></tr><tr><td> $\mathrm { T P + M A C E }$ </td><td>SMILES</td><td> $0 . 4 3 ^ { \pm 0 . 0 1 }$ </td><td> $0 . 4 7 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 2 8 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 3 9 ^ { \pm 0 . 0 5 }$ </td><td> $1 7 . 3 ^ { \pm 0 . 3 3 }$ </td><td> $0 . 5 1 ^ { \pm 0 . 0 6 }$ </td><td> $0 . 0 9 0 ^ { \pm 0 . 0 1 3 }$ </td></tr><tr><td> $\mathrm { T P + M A C E }$ </td><td>3D</td><td> $0 . 4 7 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 5 1 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 3 0 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 4 0 ^ { \pm 0 . 0 5 }$ </td><td> $1 8 . 2 ^ { \pm 0 . 5 7 }$ </td><td> $0 . 4 8 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 0 8 7 ^ { \pm 0 . 0 1 2 }$ </td></tr><tr><td> $\mathrm { T P + M A C E }$ </td><td> $\mathrm { S M I L E S + 3 D }$ </td><td> $\mathbf { 0 . 4 2 ^ { \pm 0 . 0 1 } }$ </td><td> $\mathbf { 0 . 4 5 ^ { \pm 0 . 0 5 } }$ </td><td> $0 . 2 8 ^ { \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 3 8 ^ { \pm 0 . 0 5 } }$ </td><td> $\mathbf { 1 6 . 6 ^ { \pm 0 . 4 3 } }$ </td><td> $0 . 5 0 ^ { \pm 0 . 0 5 }$ </td><td> $\mathbf { 0 . 0 8 3 ^ { \pm 0 . 0 1 3 } }$ </td></tr></table>

<sup>a</sup> For a fair comparison, we changed the TP model from double precision to single precision and re-performed the training using the same dataset as used for our models

In this section, we quantitatively evaluate the multimodal model incorporating the geometricfragmentation-based attention branch.

For this purpose, we use a widely adopted benchmark dataset for polymers <sup>1</sup> which provides p-SMILES representations together with a range of polymer properties, including the chain bandgap $( E _ { \mathrm { g c } } )$ , bulk bandgap $( E _ { \mathrm { g b } } )$ , electron afinity $\left( E _ { \mathrm { e a } } \right)$ , ionization energy $( E _ { \mathrm { i } } )$ crystallization tendency $( X _ { \mathrm { c } } )$ , dielectric constant (EPS), and refractive index $( n _ { \mathrm { c } } )$ , which are computed using density functional theory (DFT). As described in the Method section, 3D conformations are generated using RDKit.

As a baseline, we adopt TransPolymer,<sup>4</sup> a state-of-the-art model developed for this dataset. Within our multimodal framework, SMILES-based representations are encoded using either PolyBERT<sup>3</sup> or TransPolymer, <sup>4</sup> enabling a direct comparison between diferent sequence-based encoders. For the 3D structural modality, we employ the MACE-MP model<sup>24</sup> as a conformational encoder to obtain geometry-aware atomic representations. This setup allows us to systematically assess the individual contributions of each modality, as well as their combined efects, to predictive performance across diferent polymer properties. Following the evaluation protocol of TransPolymer, <sup>4</sup> we use five-fold cross-validation and evaluate model performance based on the average root mean square error (RMSE) on the validation sets. The models are pretrained on the PI1M dataset, <sup>5</sup> including TransPolymer and poly-BERT weights, augmented to five times its original size following the procedure described in.<sup>4</sup> Detailed experimental settings are provided in the Supplemental Material.

The benchmark results are presented in Table 1. Incorporating 3D conformational information improves predictive performance for several polymer properties, demonstrating the complementary value of geometric information beyond p-SMILES representations. Compared with the baseline, our multimodal model achieves an improvement of approximately 5–10% in predictive performance for several properties in comparison to the baseline. However, the contribution of the 3D modality is property-dependent; $E _ { \mathrm { g c } }$ and $X _ { \mathrm { c } }$ of TransPolymer exhibit improved performance when 3D structural information is incorporated in addition to ${ \mathrm { p } } { \mathrm { - } } { \mathrm { S M I L E S } }$ representations, whereas the remaining properties show an improvement in the averaged performance that fall within the observed fluctuations. We also note that no clear improvement is observed for $E _ { \mathrm { i } }$ . This may indicate that the available structural and sequence information provides limited additional predictive benefit for this property within the current dataset, imposing an empirical performance ceiling.

The efect of the diferent model components also depends on the model architecture and target property. We therefore investigate the contribution of the fragment-based attention branches through an ablation study, in which each branch is individually omitted during training<sup>3</sup>. The results are presented in Table 2. Overall, the model exhibits relatively weak dependence on any individual fragmentation branch, with performance diferences generally falling within the corresponding error bars. This suggests that the model can exploit complementary information across multiple structural representations rather than relying strongly on a single fragmentation scale. One notable exception is the PolyBERT model trained on $X _ { \mathrm { c } }$ without the fragmentation branch, for which a degradation in the averaged predictive performance is observed. This result suggests that fragment-level structural information may play a particularly important role in predicting this property.

Table 2: Ablation Study of the branch dependence on polymer datasets. The performance is measured using root mean-square error (RMSE). TP denotes TransPolymer, and PB denotes PolyBERT. The number written in bold font indicates a result showing a relatively notable diference from the other results, although the diference remains within one standard deviation.
<table><tr><td rowspan="2">Model Name Branch</td><td rowspan="2"></td><td colspan="7">Test RMSE</td></tr><tr><td> $E _ { \mathrm { g c } } ( \mathrm { e V } )$ </td><td> $E _ { \mathrm { g b } } ( \mathrm { e V } )$ </td><td> $E _ { \mathrm { e a } } ( \mathrm { e V } )$ </td><td> $E _ { \mathrm { i } } ( \mathrm { e V } )$ </td><td> $X _ { \mathrm { c } } ( \% )$ </td><td>EPS</td><td> $n _ { \mathrm { c } }$ </td></tr><tr><td>PB+MACE</td><td>pSMILES+3D+Fragment</td><td> $\overline { { 0 . 4 2 ^ { \pm 0 . 0 1 } } }$ </td><td> $0 . 4 6 ^ { \pm 0 . 0 4 }$ </td><td> $\overline { { 0 . 2 7 ^ { \pm 0 . 0 2 } } }$ </td><td> $\overline { { 0 . 3 9 ^ { \pm 0 . 0 5 } } }$ </td><td> $\overline { { 1 7 . 1 ^ { \pm 0 . 4 8 } } }$ </td><td> $0 . 4 7 ^ { \pm 0 . 0 5 }$ </td><td> $\overline { { 0 . 0 8 5 ^ { \pm 0 . 0 1 2 } } }$ </td></tr><tr><td>PB+MACE</td><td>3D+Fragment</td><td> $0 . 4 2 ^ { \pm 0 . 0 1 }$ </td><td> $0 . 4 8 ^ { \pm 0 . 0 5 }$ </td><td> $0 . 2 8 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 3 9 ^ { \pm 0 . 0 5 }$ </td><td> $1 7 . 1 ^ { \pm 0 . 4 7 }$ </td><td> $0 . 5 0 ^ { \pm 0 . 0 5 }$ </td><td> $0 . 0 8 4 ^ { \pm 0 . 0 1 2 }$ </td></tr><tr><td> $\mathrm { P B + M A C E }$ </td><td>pSMILES+Fragment</td><td> $0 . 4 2 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 4 6 ^ { \pm 0 . 0 5 }$ </td><td> $0 . 2 7 ^ { \pm 0 . 0 3 }$ </td><td> $0 . 3 8 ^ { \pm 0 . 0 5 }$ </td><td> $1 7 . 1 ^ { \pm 0 . 4 8 }$ </td><td> $0 . 5 1 ^ { \pm 0 . 0 5 }$ </td><td> $0 . 0 8 5 ^ { \pm 0 . 0 1 2 }$ </td></tr><tr><td>PB+MACE</td><td>pSMILES+3D</td><td> $0 . 4 1 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 4 5 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 2 7 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 3 8 ^ { \pm 0 . 0 5 }$ </td><td> $1 7 . 5 ^ { \pm 0 . 5 2 }$ </td><td> $0 . 4 8 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 0 8 1 ^ { \pm 0 . 0 1 3 }$ </td></tr><tr><td>TP+MACE</td><td>pSMILES+3D+Fragment</td><td> $0 . 4 2 ^ { \pm 0 . 0 1 }$ </td><td> $\overline { { 0 . 4 5 ^ { \pm 0 . 0 5 } } }$ </td><td> $\overline { { 0 . 2 8 ^ { \pm 0 . 0 2 } } }$ </td><td> $0 . 3 8 ^ { \pm 0 . 0 5 }$ </td><td> $\overline { { 1 6 . 6 ^ { \pm 0 . 4 3 } } }$ </td><td> $0 . 5 0 ^ { \pm 0 . 0 5 }$ </td><td> $\overline { { 0 . 0 8 3 ^ { \pm 0 . 0 1 3 } } }$ </td></tr><tr><td>TB+MACE</td><td>3D+Fragment</td><td> $0 . 4 2 ^ { \pm 0 . 0 1 }$ </td><td> $0 . 4 6 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 2 9 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 3 8 ^ { \pm 0 . 0 5 }$ </td><td> $1 7 . 1 ^ { \pm 0 . 5 1 }$ </td><td> $0 . 5 0 ^ { \pm 0 . 0 6 }$ </td><td> $0 . 0 8 3 ^ { \pm 0 . 0 1 2 }$ </td></tr><tr><td> $\mathrm { T B } { + } \mathrm { M A } \mathrm { C E }$ </td><td>pSMILES+Fragment</td><td> $0 . 4 2 ^ { \pm 0 . 0 1 }$ </td><td> $0 . 4 5 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 2 7 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 3 8 ^ { \pm 0 . 0 5 }$ </td><td> $1 6 . 9 ^ { \pm 0 . 8 0 }$ </td><td> $0 . 5 0 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 0 8 5 ^ { \pm 0 . 0 1 4 }$ </td></tr><tr><td> $\mathrm { T B } { + } \mathrm { M A } \mathrm { C E }$ </td><td>pSMILES+3D</td><td> $0 . 4 1 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 4 6 ^ { \pm 0 . 0 4 }$ </td><td> $0 . 2 7 ^ { \pm 0 . 0 2 }$ </td><td> $0 . 3 9 ^ { \pm 0 . 0 6 }$ </td><td> $1 6 . 8 ^ { \pm 0 . 6 5 }$ </td><td> $0 . 4 8 ^ { \pm 0 . 0 6 }$ </td><td> $0 . 0 8 3 ^ { \pm 0 . 0 1 1 }$ </td></tr></table>

Finally, we compare the embeddings produced by our geometric-fragmentation-based attention module with MAP4 fingerprints. <sup>12</sup> We use all p-SMILES representations in the polymer dataset <sup>1</sup> for this analysis. For our model, we use the weights fine-tuned on the $E _ { \mathrm { g c } }$ dataset using fold 1. We first compute the MAP4 fingerprints and the embeddings generated by our geometric fragmentation-based attention module, and then apply $\mathrm { t - S N E ^ { 2 8 } }$ to project both representations into a two-dimensional space. The results are shown in Figure 5. The embeddings produced by our fragment attention branch exhibit a relatively uniform and isotropic distribution of the input monomers, whereas the MAP4 fingerprints produce a more non-uniform distribution. The non-uniform distribution of MAP4 fingerprints may be advantageous for rapid molecular screening, which is one of the primary applications of MAP4. In contrast, the more uniform coverage observed in the learned fragment-based representation may provide a potentially useful inductive bias for downstream property prediction by reducing the concentration of representations in specific regions of the molecular space. However, this interpretation should be regarded as a qualitative hypothesis, as the two-dimensional t-SNE projections alone do not establish a direct relationship between representation uniformity and predictive performance. The color map indicates the number of heavy atoms in each monomer. In both representations, we observe a weak clustering efect associated with the number of heavy atoms, suggesting that molecular size remains a relevant factor in the learned representation.

![](images/47d93b75e47db634a4dd3ce3a6f1d8f57974e6a8556bd7e187faab5ce4de4f74.jpg)

![](images/d7e8472e62fa04878313e13fd590144bb3f16e511cd95fd031725cb831a1ce16.jpg)  
Figure 5: Scatter plot of latent vector embedded by $\mathrm { M A P 4 ^ { 1 2 } }$ and fragment-attention branch. The embedded vectors are mapped onto two-dimensional space with tSNE. The color map shows the heavy atom number in the monomer.

![](images/c7e9637ea25ef721cad9e447b18524099d4f9d1867bf5301127a84a4d7e1b484.jpg)  
Figure 6: A plot of importance-score of the input structure in the case of $[ ^ { * } ] \mathrm { C C C } ( = \mathrm { O } ) \mathrm { O } [ ^ { * } ]$ Only oxygen atoms are labeled, whereas carbon and hydrogen atoms are left unlabeled for clarity. The plotted importance-score is normalized by the maximum value of the original importance-score.

![](images/7ee8ebb4a2ade1a3822941f48c7324aee2cc59c804d802e5f2186604db387702.jpg)  
Figure 7: A plot of importance-score of the input structure in the case of [\*]CCCC-$\mathrm { C C C ( = O ) N [ ^ { * } ] }$ . Only oxygen and nitrogen atoms are labeled, whereas carbon and hydrogen atoms are left unlabeled for clarity. The plotted importance-score is normalized by the maximum value of the original importance-score.

## 3.3 Interpretation of model prediction via Explainable AI method

Interpretability is a critical aspect of machine learning in chemistry, where understanding the structural features underlying model predictions can provide insights into the learned representations. In this section, we investigate the types of structural information utilized by the ML model to assess whether its predictions are associated with chemically meaningful features. To this end, we compute feature attribution scores with respect to the input data using Captum,<sup>16</sup> which estimates feature importance based on gradient information<sup>4</sup>. By analyzing feature attributions across the diferent modalities, we examine whether the model consistently identifies chemically relevant functional groups and structural motifs in agreement with established chemical knowledge. Because many polymer properties are strongly influenced by specific functional groups, a chemically meaningful representation is expected to assign elevated attribution scores to atoms and substructures associated with these groups. This analysis provides additional evidence for the physical relevance of the learned representations and highlights the potential of multimodal approaches for interpretable polymer informatics.

As discussed above, molecular properties often exhibit strong dependencies on specific functional groups. In general, these relationships arise from the combined efects of electronic structure, such as donor and acceptor strength, dipole moment, polarizability, and threedimensional chain conformation. <sup>29–36</sup> A summary of potentially relevant functional groups and their expected relationships with the target properties is provided in Table 3. In the following analysis, we examine whether the ML model identifies these chemically relevant structural features directly from the data.

Figure 6 and Figure 7 present the importance scores for representative monomers with the following p-SMILES:

$$
\begin{array} { r } { [ * ] \mathrm { C C C } ( = \mathrm { O } ) \mathrm { O } [ * ] \quad \mathrm { a n d } \quad [ * ] \mathrm { C C C C C } ( = \mathrm { O } ) \mathrm { N } [ * ] . } \end{array}\tag{3}
$$

These p-SMILES correspond to structures containing several functional groups, including amine (NH<sub>2</sub>), carbonyl (-C=O-), hydroxyl (-OH), and alkyl $\mathrm { ( - R , \ e . g . , - C H _ { 3 } ) }$

We first consider the monomer $[ ^ { * } ] \mathrm { C C C } ( = \mathrm { O } ) \mathrm { O } [ ^ { * } ]$ , shown in Figure 6. In this structure, the ester linkage (-C(=)-O-) is located on the right-hand side, while the Alkyl groups (- $\mathrm { C } _ { n } \mathrm { H } _ { 2 n + 1 } \rangle$ appear on the left-hand side. Overall, both the SMILES-based and 3D models assign relatively high importance to these functional groups, though it also depends on the target properties. The SMILES model distributes attention across the entire $\mathrm { p { - } S M I L E S }$ string but exhibits enhanced attribution to tokens such as $^ { 6 6 } ( = ) ^ { 5 }$ and $^ { 6 6 } \mathrm { O } ^ { \ 3 }$ , indicating that it more or less identifies chemically relevant substructures. In contrast, the 3D model yields more localized importance patterns, with particularly high scores in regions surrounding oxygen atoms, again consistent with chemically meaningful features.

Next, we examine the monomer $[ { ^ { * } } ] { \mathrm { C C C C C C } } ( { = } \mathrm { O } ) \mathrm { N } [ { ^ { * } } ]$ , shown in Figure 7. In this case, the carbonyl $\scriptstyle \left( - \mathrm { C = O } \right)$ and amide linkage (-NH-) groups are located on the right-hand side of the structure. A similar trend is observed: both the SMILES-based and 3D models assign relatively high attribution scores to these functional groups, though depending on the target property. These results suggest that the model predictions are influenced, at least in part, by chemically meaningful structural motifs rather than being determined solely by global molecular features <sup>5</sup>.

Table 3: Summary of functional group that can efect on polymer electronic, structural, and optical properties.  
Functional group Structure Mechanism   
Carbonyl C=O Strong electron withdrawing, <sup>30,37</sup> planar conjugation enhances π-overlap<sup>38</sup>   
Fluoro C–F, CF<sub>3</sub> High electronegativity; <sup>30,37</sup> bulky groups disrupt packing <sup>35</sup>   
Ether C–O–C Breaks conjugation,<sup>38</sup> adds dipole<sup>34</sup>   
$\mathrm { { A l k y l } }$ $- \mathrm { C } _ { n } \mathrm { H } _ { 2 n + 1 }$ increases torsion/free volume<sup>31</sup>   
Hydroxyl –OH H-bonding increases packing<sup>36</sup> but breaks conjugation<sup>38</sup>   
Amine $\mathrm { - N H _ { 2 } }$ Electron-donating <sup>30,37</sup>   
Sulfonyl $- \mathrm { S O _ { 2 ^ { - } } }$ Strong dipole<sup>34</sup> + electron-withdrawing<sup>30,37</sup>

![](images/f53476a88b92125cecddb0850168113837317f10a06d63157a93b68e73c46d45.jpg)

![](images/db12fc7a74bb65dac506de3b2b283432704f7b3e0977cb6991b8a526052988ad.jpg)  
Figure 8: A statistics of importance-score of each dataset. $^ { \prime \prime } \mathrm { A v } ^ { \prime \prime }$ denotes the average of the all the connectivity and atoms in the input data. The importance-score of each sample is normalized by the each sample’s maximum value of the importance-score.

In Figure 8, we present the average importance scores of the atoms $[ \mathrm { = O , O , F , N , S ] ^ { 6 } }$ for the SMILES-based branch and [O, F, N, S] for the 3D conformation–based branch. Overall, the results indicate that the model attends preferentially to atoms associated with key functional groups. Notably, the SMILES importance score distribution suggests that the SMILES encoder emphasizes oxygen and sulfur related features while assigning comparatively less importance to fluorine. In contrast, the 3D encoder leverages a more diverse set of atomic features, with its attention patterns varying across datasets. Taken together, the attribution analyses indicate that both modality branches preferentially focus on chemically relevant heteroatoms and functional groups. This observation motivates future investigations into incorporating explicit functional-group information as an inductive bias.

## 4 Discussion

In this work, we proposed a geometry-based fragmentation method that reduces reliance on predefined chemistry-specific fragmentation schemes, such as BRICS, while enabling flexible control over fragment size and the extraction of structural information at multiple length scales. By formulating molecular fragmentation as a radius-constrained optimization problem inspired by the k-center formulation, where the number of clusters is not fixed a priori but emerges from the specified radius, our approach can be applied to arbitrary molecular or material structures using purely geometric information, without requiring predefined chemical rules. Building on this framework, we developed a geometric fragment-based attention module and incorporated it into a multimodal model that combines p-SMILES representations with three-dimensional (3D) conformational information. The multimodal framework further integrates representative pretrained models for the respective modalities, allowing complementary sequence-based and geometry-aware representations to be exploited for polymer property prediction.

Although we use geometry-based fragments in the present model, the proposed fragmentation framework is not restricted to this particular strategy. Other fragmentation approaches, including chemistry-based methods such as BRICS, could potentially be incorporated into the same framework. Combining geometry-based and rule-based fragmentation may provide complementary inductive biases by integrating chemically predefined structural information with geometry-driven representations at multiple length scales. Exploring such hybrid fragmentation strategies represents an interesting direction for future work.

We systematically investigated the efects of fragment-based attention and multimodal information using a benchmark p-SMILES dataset for polymer property prediction. Incorporating 3D conformational information improves predictive performance for several target properties, with the multimodal model achieving an improvement over the baseline for several properties. However, the contribution of each modality is property-dependent. In particular, 3D structural information improves predictions for properties such as $E _ { \mathrm { g c } }$ and $X _ { \mathrm { c } }$ of TransPolymer, whereas other properties exhibit greater sensitivity to the p-SMILES representation. The ablation analysis further reveals that the relative contributions of individual branches vary across datasets and target properties, with no single branch consistently outperforming the others. These results highlight the complementary nature of sequence-based and geometry-based representations and suggest that efective multimodal integration is important for capturing the diverse structural factors governing polymer properties.

To further investigate the structural information captured by the multimodal model, we performed explainable artificial intelligence (XAI) analysis using Captum, evaluating feature attributions with respect to the input representations. The results indicate that the model preferentially attends to chemically meaningful structural features. In both the SMILESbased and 3D representations, elevated attribution scores are observed for chemically relevant functional groups and heteroatoms, although the spatial patterns of attribution difer between the two modalities. The SMILES-based model distributes attribution more broadly across the input tokens, while the 3D model produces more localized attribution patterns around chemically relevant atomic regions. These observations suggest that the model learns representations that are associated with physically meaningful structural features and are broadly consistent with established chemical knowledge. However, the current attribution analysis does not resolve the directionality of these contributions; that is, it does not determine whether a given functional group increases or decreases a particular target property. Further work incorporating attribution methods capable of resolving feature-wise contribution directions could provide deeper insight into the learned structure–property relationships.

A potential limitation of the present approach concerns the accuracy and representativeness of the 3D conformations used as input. Although RDKit is widely used for molecular conformer generation in computational chemistry and machine learning research,<sup>39</sup> the conformations generated in our workflow are obtained using classical force-field-based optimization rather than quantum chemical methods such as density functional theory (DFT). Consequently, the generated structures may not accurately represent the energetically preferred conformations at the relevant level of theory. More importantly, flexible polymer structures can adopt a diverse ensemble of conformations, and multiple conformations may contribute to their observed macroscopic properties. <sup>39</sup> In the present work, we generate multiple conformers for each p-SMILES representation; however, only a single conformer is provided as the 3D input to the model at a given time, rather than simultaneously incorporating an ensemble of conformations as in. <sup>39</sup> Future work could therefore investigate higher-accuracy quantum-chemical conformers, explicit ensemble-based representations, or dynamically sampled conformations to better capture conformational variability and assess its impact on polymer property prediction.

Overall, our results demonstrate the potential of geometry-based fragmentation as a chemistry-independent strategy for constructing multiscale structural representations and integrating them with complementary molecular modalities. The combination of geometrybased fragmentation, multimodal learning, and XAI provides a promising framework for developing polymer informatics models that are both predictive and physically interpretable. More broadly, because the proposed fragmentation method relies only on geometric information, it may be applicable beyond polymers and molecular systems to more general molecular and material structures, providing a flexible foundation for future multiscale and modalityaware machine learning approaches in computational chemistry and materials science.

## 5 Data and code availability

The training data are available from the TransPolymer repository at https://github.com/ ChangwenXu98/TransPolymer. The PI1M dataset used for pretraining is publicly available at https://github.com/RUIMINMA1996/PI1M. The source code and implementation details supporting the findings of this study are available from the corresponding author upon reasonable request.

## Acknowledgements

Mathias Niepert acknowledges support from the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy (EXC 2075 – 390740016) and through grant 569019417. He also acknowledges support from the Stuttgart Center for

## Supporting information

The following files are included in the ACS Omega journal version of our manuscript.

• Supplemental Information.pdf: A more detailed model structure description, data preprocess, and training Setup; Additional details of geometry-based molecular fragmentation, including step-by-step example; Ablation studies of benchmark results; Functional-group distribution in fragments.

## References

(1) Kim, C.; Chandrasekaran, A.; Huan, T. D.; Das, D.; Ramprasad, R. Polymer genome: a data-powered polymer informatics platform for property predictions. The Journal of Physical Chemistry C 2018, 122, 17575–17585.

(2) Weininger, D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of chemical information and computer sciences 1988, 28, 31–36.

(3) Kuenneth, C.; Ramprasad, R. polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics. Nature communications 2023, 14, 4099.

(4) Xu, C.; Wang, Y.; Barati Farimani, A. TransPolymer: a Transformer-based language model for polymer property predictions. npj Computational Materials 2023, 9, 64.

(5) Ma, R.; Luo, T. PI1M: a benchmark database for polymer informatics. Journal of Chemical Information and Modeling 2020, 60, 4684–4690.

(6) He, P.; Liu, X.; Gao, J.; Chen, W. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654 2020,

(7) Degen, J.; Wegscheid-Gerlach, C.; Zaliani, A.; Rarey, M. On the art of compiling and using’drug-like’chemical fragment spaces. ChemMedChem 2008, 3, 1503.

(8) Liu, Y.; Ott, M.; Goyal, N.; Du, J.; Joshi, M.; Chen, D.; Levy, O.; Lewis, M.; Zettlemoyer, L.; Stoyanov, V. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 2019,

(9) Todeschini, R.; Consonni, V. Handbook of molecular descriptors; John Wiley & Sons, 2008.

(10) Rogers, D.; Hahn, M. Extended-connectivity fingerprints. Journal of chemical information and modeling 2010, 50, 742–754.

(11) Engel, T.; Gasteiger, J. Chemoinformatics: basic concepts and methods; John Wiley & Sons, 2018.

(12) Capecchi, A.; Probst, D.; Reymond, J.-L. One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome. Journal of cheminformatics 2020, 12, 43.

(13) Chen, L.; Pilania, G.; Batra, R.; Huan, T. D.; Kim, C.; Kuenneth, C.; Ramprasad, R. Polymer informatics: Current status and critical next steps. Materials Science and Engineering: R: Reports 2021, 144, 100595.

(14) Zhang, Z.; Guan, J.; Zhou, S. FraGAT: a fragment-oriented multi-scale graph attention model for molecular property prediction. Bioinformatics 2021, 37, 2981–2987.

(15) Queen, O.; McCarver, G. A.; Thatigotla, S.; Abolins, B. P.; Brown, C. L.; Maroulas, V.; Vogiatzis, K. D. Polymer graph neural networks for multitask property learning. npj Computational Materials 2023, 9, 90.

(16) Kokhlikyan, N.; Miglani, V.; Martin, M.; Wang, E.; Alsallakh, B.; Reynolds, J.; Melnikov, A.; Kliushkina, N.; Araya, C.; Yan, S.; others Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896 2020,

(17) Wolf, G. W. Facility location: concepts, models, algorithms and case studies. series: Contributions to management science. International Journal of Geographical Information Science 2011, 25, 331–333.

(18) Sener, O.; Savarese, S. Active Learning for Convolutional Neural Networks: A Core-Set Approach. International Conference on Learning Representations. 2018.

(19) Zaverkin, V.; Holzm¨uller, D.; Steinwart, I.; K¨astner, J. Exploring chemical and conformational spaces by batch mode deep active learning. Digital Discovery 2022, 1, 605–620.

(20) Gonzalez, T. F. Clustering to minimize the maximum intercluster distance. Theoretical Computer Science 1985, 38, 293–306.

(21) Larsen, A. H. et al. The atomic simulation environment—a Python library for working with atoms. J. Phys. Condens. Matter 2017, 29, 273002.

(22) Wollschl¨ager, T.; Kemper, N.; Hetzel, L.; Sommer, J.; G¨unnemann, S. Expressivity and Generalization: Fragment-Biases for Molecular GNNs. International Conference on Machine Learning. 2024; pp 53113–53139.

(23) Probst, D.; Reymond, J.-L. A probabilistic molecular fingerprint for big data settings. Journal of cheminformatics 2018, 10, 66.

(24) Batatia, I. et al. A foundation model for atomistic materials chemistry. The Journal of Chemical Physics 2025, 163, 184110.

(25) Batatia, I.; Kovacs, D. P.; Simm, G.; Ortner, C.; Cs´anyi, G. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. Advances in neural information processing systems 2022, 35, 11423–11436.

(26) Roberts, J. D.; Caserio, M. C. Basic principles of organic chemistry. 1965,

(27) McMurry, J.; Begley, T. P. The organic chemistry of biological pathways; Roberts and Company Publishers, 2005.

(28) Van der Maaten, L.; Hinton, G. Visualizing data using t-SNE. Journal of machine learning research 2008, 9 .

(29) Darling, S. B. Isolating the efect of torsional defects on mobility and band gap in conjugated polymers. The Journal of Physical Chemistry B 2008, 112, 8891–8895.

(30) Ansari, M. A.; Mohiuddin, S.; Kandemirli, F.; Malik, M. I. Synthesis and characterization of poly (3-hexylthiophene): improvement of regioregularity and energy band gap. RSC advances 2018, 8, 8319–8328.

(31) Wedler, S.; Bourdick, A.; Athanasopoulos, S.; Gekle, S.; Panzer, F.; McDowell, C.; Nguyen, T.-Q.; Bazan, G. C.; K¨ohler, A. What is the role of planarity and torsional freedom for aggregation in a π-conjugated donor–acceptor model oligomer? Journal of Materials Chemistry C 2020, 8, 4944–4955.

(32) Velasco Davoise, L.; Pe˜na Capilla, R.; D´ıez-Pascual, A. M. Assessment of the refractive index and extinction coeficient of graphene-poly (3-hexylthiophene) nanocomposites. Polymers 2022, 14, 1828.

(33) Mamand, D. M.; Hussen, S. A.; Aziz, S. B. Green approach to synthesis polymer composites based on chitosan with desired linear and non-linear optical characteristics. Scientific Reports 2025, 15, 3130.

(34) Mogbojuri, G.; Abtahi, S.; Hendeniya, N.; Chang, B. The Efects of Chain Conformation and Nanostructure on the Dielectric Properties of Polymers. Materials 2025, 18 .

(35) Wu, X.; Zhang, Z.; Wu, Q.; Ivry, Y.; Tan, D. Q. Dielectric performance of all-organic

polymer composites via incorporation of fluorinated molecules. Polymer 2025, 331, 128546.

(36) Aziz, I. A.; Tullii, G.; Antognazza, M. R.; Criado-Gonzalez, M. Poly (3-hexylthiophene) as a versatile semiconducting polymer for cutting-edge bioelectronics. Materials Horizons 2025, 12, 5570–5593.

(37) Peet, J.; Kim, J. Y.; Coates, N. E.; Ma, W. L.; Moses, D.; Heeger, A. J.; Bazan, G. C. Eficiency enhancement in low-bandgap polymer solar cells by processing with alkane dithiols. Nature materials 2007, 6, 497–500.

(38) Vu, K. B.; Vu, V. V.; Thi Thu, H. P.; Giang, H. N.; Tam, N. M.; Ngo, S. T. Conjugated polymers: A systematic investigation of their electronic and geometric properties using density functional theory and semi-empirical methods. Synthetic Metals 2018, 246, 128–136.

(39) Nguyen, D. M. H.; Lukashina, N.; Nguyen, T.; Le, A. T.; Nguyen, T.; Ho, N.; Peters, J.; Sonntag, D.; Zaverkin, V.; Niepert, M. Structure-Aware E (3)-Invariant Molecular Conformer Aggregation Networks. International Conference on Machine Learning. 2024; pp 37736–37760.