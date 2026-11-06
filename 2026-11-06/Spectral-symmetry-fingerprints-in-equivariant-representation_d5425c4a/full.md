# Spectral symmetry fingerprints in equivariant representations

Jiacheng Cen<sup>1,2,3</sup>, Anyi Li<sup>1,2,3</sup>, Ning Lin<sup>1,2,3</sup>, Zihe Wang<sup>1,2,3</sup>, Ji-Rong Wen<sup>1,2,3</sup>, and Wenbing Huang<sup>1,2,3,4,</sup> <sup>B</sup>

<sup>1</sup>Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China

<sup>2</sup>Beijing Key Laboratory of Research on Large Models and Intelligent Governance, Beijing, China

<sup>3</sup>Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE, Beijing, China <sup>4</sup>Beijing Academy of Artificial Intelligence, Beijing, China

<sup>B</sup>Correspondence should be addressed to: hwenbing@ruc.edu.cn

## ABSTRACT

Symmetry governs the organization of physical structures, while symmetry breaking reveals their transformations. In geometric deep learning, symmetry imposes rank constraints on equivariant representations, limiting the expressive capacity of equivariant graph neural networks (GNNs). Here we show that these constraints can instead be exploited: the singular-value spectra of equivariant representations provide a unified fingerprint of structural symmetry. We introduce spectral symmetry fingerprints (SymFP), which enable discrete point-group identification through rank-template matching and continuous symmetry analysis by quantifying spectral energy beyond target rank constraints. SymFP is orientation invariant and architecture agnostic, enabling a parameter-free GNN implementation. Its spectra can be reused across multiple target groups, and the resulting leakage scores satisfy subgroup monotonicity. Across synthetic structures, molecules, coordination complexes and large biological assemblies, SymFP accurately identifies point groups, tracks controlled symmetry breaking and reveals local and degree-resolved symmetry changes that can be obscured by global geometric measures. These results establish the spectral structure of equivariant representations as a general framework for characterizing symmetry across physical systems.

## 1 Introduction

Symmetry, defined as the invariance of a structure under specific transformations, provides a fun damental principle for understanding structural organization in the physical world. In chemistry and structural biology, characterizing symmetry ofers important insights into structural states, functional properties, and state transformations. Real-world structures, however, often exhibit only approximate symmetry due to thermal fluctuations, structural distortions, and external perturbations. Structural symmetry characterization therefore involves two complementary problems: discrete point-group identification, which assigns exactly symmetric structures to predefined point groups, and continuous symmetry analysis, which quantifies the extent to which a structure deviates from a target symmetry<sup>1–5</sup>.

Equivariant representations ofer a natural starting point for addressing these two tasks. These representations transform consistently with their inputs under rotations and reflections <sup>6</sup> and form the basis of equivariant graph neural networks (GNNs), which have achieved considerable success across scientific applications<sup>7–9</sup>. Specifically, equivariant representations of degree l are collected into a matrix $\tilde { \mathbf { V } } ^ { ( l ) } \in \mathbb { R } ^ { ( 2 l + 1 ) \times C }$ , whose C feature channels transform under the same degree-l representation of rotations and reflections. For symmetric structures, however, equivariance restricts these feature channels to subspaces fixed by the input point group, leading to a reduction in expressive power<sup>10–14</sup>. The dimensions of these subspaces determine the rank constraints on $\tilde { V } ^ { ( l ) }$ which vary across point groups and degrees <sup>13,14</sup>. Here we turn these rank constraints into a tool for characterizing symmetry.

To this end, we construct spectral symmetry fingerprints (SymFP) from the singular-value spectra of equivariant representations across degrees, providing a common basis for discrete pointgroup identification and continuous symmetry analysis. For the former, we define a rank template for each point group based on its fixed-subspace dimensions across degrees. The numerical rank sequence of an input structure is obtained from SymFP by counting singular values above a numerical threshold at each degree. Matching this numerical rank sequence against the candidate rank templates allows us to identify the point group. Degrees $l = 0 , \ldots , 9$ are suficient to distin guish the rank templates of all 32 crystallographic point groups. For the latter, deviations from a target symmetry can cause singular values that would otherwise be zero to become nonzero. To quantify these deviations, we introduce leakage scores (LS), which measure the fraction of spectral energy associated with singular values beyond the target fixed-subspace dimension at each degree. Degree-wise leakage scores further reveal how symmetry deviations are distributed across degrees. By averaging LS across selected degrees and applying a regularized negative-log transformation, we obtain pLS, with higher values indicating stronger agreement with the target rank constraints. As SymFP relies only on the algebraic structure of equivariant representations, it can be applied across diferent equivariant GNN architectures. To facilitate reuse and evaluation, we also develop a parameter-free (PF) equivariant GNN that generates representations directly from input structures using fixed radial bases and spherical harmonics.

By comparison, traditional approaches to structural symmetry characterization primarily rely on geometric distances or symmetry operations. Continuous symmetry measures (CSMs) quantify deviations from symmetry through the distance to the nearest symmetric configuration <sup>1,2</sup>. The continuous symmetry operation measure (CSoM) evaluates deviations under individual symmetry operations and combines them into a continuous score <sup>3,4</sup>. Engel’s method, here termed Engel score, uses spherical harmonics to represent structures and searches over target-group orientations to maximize the energy retained by projection onto the corresponding fixed subspaces<sup>5,15</sup>. In contrast, SymFP characterizes structural symmetry through the intrinsic spectral structure of equivariant representations. Its construction is independent of any target point group, which enters subsequent analysis only through its fixed-subspace dimensions. This avoids constructing symmetric reference structures or searching over target orientations. Moreover, the same spectra can be reused across diferent target groups without recomputing the representations or their singular values, enabling eficient analysis of multiple target symmetries. Beyond computational eficiency, SymFP also satisfies subgroup monotonicity: for a given structure, relaxing the symmetry requirements from a group to one of its subgroups cannot increase the leakage score. This property supports consistent analysis of symmetry breaking along group–subgroup hierarchies. We further connect CSoM style operation-wise deviations with Engel’s retained projection energy in representation space and establish their relationships with the spectral energy underlying SymFP through variational analysis.

We evaluate SymFP on synthetic structures and real-world systems across scales. Experiments on synthetic symmetric structures show that numerical rank sequences match the corresponding rank templates and demonstrate accurate discrete point-group identification across GNN architectures and training states, even with the parameter-free GNN. Controlled deformations further show that SymFP tracks symmetry breaking and distinguishes deformation modes even at matched displacement amplitudes. We then extend the analysis to three classes of real-world systems. In aromatic molecules, torsion scans show that SymFP retrieves energy extrema from geometric symmetry, outperforming CSoM and Engel score in overall retrieval performance, while pretraining further enhances spectral contrasts associated with target symmetries. In DOTA complexes, fixed-subspace dimensions correspond to crystal-field parameter counts, and spectral responses to capping-water rotation exhibit periodicity consistent with electronic-structure calculations. For Dy-DOTA-water, SymFP evaluates seven target groups approximately two orders of magnitude faster than the fastest tested Engel implementation and four orders of magnitude faster than CSoM. Finally, across four large biological assemblies, SymFP reveals opposing local symmetry changes masked by whole-assembly scores, regional diferences during conformational transitions, spatial variations in symmetry and the efects of missing assembly units. Together, these results show how SymFP brings discrete point-group identification and continuous symmetry analysis into a common framework, revealing structural symmetry across degrees and spatial regions.

## 2 Results

## 2.1 Spectral symmetry fingerprints

We construct spectral symmetry fingerprints (SymFP) from the singular-value spectra of degreewise representations produced by an equivariant GNN. Specifically, let $f ^ { ( l ) }$ denote the degree-l output map of an equivariant GNN<sup>16</sup>. For a geometric graph $\mathcal { G } \ ( e . g . , \mathrm { ~ a ~ }$ 3D molecule or protein structure), the C representation channels produced at each degree $l = 0 , \ldots , L$ are collected as $f ^ { ( l ) } ( \mathcal { G } ) = \tilde { \pmb { V } } ^ { ( l ) } ( \mathcal { G } ) = [ \tilde { \pmb { v } } _ { 1 } ^ { ( l ) } ( \mathcal { G } ) , \dots , \tilde { \pmb { v } } _ { C } ^ { ( l ) } ( \mathcal { G } ) ] \in \mathbb { R } ^ { ( 2 l + 1 ) \times C }$ , where each $\tilde { \pmb { v } } _ { c } ^ { ( l ) } ( { \mathcal G } ) \in \mathbb { R } ^ { 2 l + 1 }$ denotes the degree-l representation vector of channel c (Fig. 1a). By default, we use the same channel count $C > 2 L + 1$ across all degrees. In addition to being permutation-invariant with respect to nodes and edges, these representations are equivariant to transformations in $\mathrm { O ( 3 ) }$

$$
\tilde { \pmb { V } } ^ { ( l ) } ( \mathfrak { h } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { h } ) \tilde { \pmb { V } } ^ { ( l ) } ( \mathcal { G } ) , \qquad \forall \mathfrak { h } \in \mathrm { O } ( 3 ) ,\tag{1}
$$

where $\mathfrak { h } \in \mathrm { O } ( 3 )$ denotes a rotation or reflection, and $\rho ^ { ( l ) }$ is the corresponding degree-l representation<sup>17</sup>. We apply singular value decomposition (SVD) to each degree-l representation $\tilde { V } ^ { ( l ) } ( \mathcal { G } )$ and collect the ordered singular values $\sigma _ { 1 } ^ { ( l ) } \geq \cdot \cdot \cdot \geq \sigma _ { 2 l + 1 } ^ { ( l ) } \geq 0$ across degrees to define the spectral symmetry fingerprint (SymFP):

$$
\operatorname { S y m F P } ( \mathcal { G } ) : = ( \sigma _ { i } ^ { ( l ) } ) _ { 0 \leq l \leq L , \ 1 \leq i \leq 2 l + 1 } .\tag{2}
$$

By construction, SymFP preserves O(3)-invariance, enabling symmetry analysis without explicit orientation optimization or search. We consider two complementary tasks: discrete point-group identification for exactly symmetric structures and continuous symmetry analysis to quantify departures from target symmetry.

## Discrete point-group identification

If a graph G is symmetric under a group H, i.e., ${ \mathfrak { h } } \cdot { \mathcal { G } } = { \mathcal { G } }$ for every ${ \mathfrak { h } } \in { \mathfrak { H } }$ , equivariance constrains its degree-l representations to the corresponding fixed subspace of dimension $m _ { l } ( { \mathfrak { H } } )$ , implying rank $( \tilde { V } ^ { ( l ) } ( { \mathcal G } ) ) \le m _ { l } ( \mathfrak { H } )$ . The ranks of equivariant representations across degrees define the rank sequence:

$$
\mathrm { R a n k S e q } ( \mathcal { G } ) : = \left( \mathrm { r a n k } ( \tilde { V } ^ { ( l ) } ( \mathcal { G } ) ) \right) _ { l = 0 } ^ { L } \leq \left( m _ { l } ( \mathfrak { H } ) \right) _ { l = 0 } ^ { L } : \mathrm { R a n k T e m p l a t e } ( \mathfrak { H } ) .\tag{3}
$$

![](images/32832d4237a1170f9ff7957d0553195fcdfc5380b6d75cc396f1770c8cbde77f.jpg)

b  
![](images/1c1426ac6370f260bd5f93268d100f953a23448969e67839cf363fdce25bfa03.jpg)  
c

![](images/602a2388b1c2432fbf856beceb0e3f4143aea4e9537c7888bef0953e87ef7a44.jpg)  
Figure 1. Overview of SymFP. a, Pipeline for constructing SymFP from a geometric graph. The spectra support discrete point-group identification through rank-template matching and continuous symmetry analysis through leakage scores. b, Rank templates of the 32 crystallographic point groups. c, Subgroup relations and monotonicity of rank templates and leakage scores. In $\mathbf { b } , \mathbf { c } .$ , fill colour denotes the crystal system, with grey for blocks spanning multiple systems.

Here, $\leq$ is understood componentwise, and RankTemplate(H) denotes the rank template associ ated with H (Fig. 1b). In practice, we estimate RankSeq(G) by counting the singular values in SymFP above an appropriate numerical threshold, obtaining a numerical rank sequence (Eq. (26)). When the representations $\tilde { V } ^ { ( l ) } ( \mathcal { G } )$ span the fixed subspaces, RankSeq(G) = RankTemplate(H), so a unique match among the candidate rank templates then identifies the point group directly (Eq. (27)). Degrees $l = 0 , \ldots , 6$ already yield 30 distinct rank templates across the 32 crystallographic point groups, with only $\mathrm { D } _ { 6 } / \mathrm { D } _ { 6 h }$ and $\mathrm { O } / \mathrm { O } _ { h }$ sharing the same template. Extending the sequence to $l = 0 , \ldots , 9$ distinguishes all 32 groups (Fig. 1b; Cor. 4). Empirically, we find that the representations span the full fixed subspaces in most tested cases (Supplementary Table C.2). Moreover, for a fixed input and under suitable regularity conditions, our theoretical analysis shows that if equality in Eq. (3) holds at one parameter setting, it holds for almost every parameter choice.

## Continuous symmetry analysis

When a graph G departs from exact symmetry, its degree-wise representations may violate the corresponding rank-template constraints. Specifically, singular directions beyond the fixed-subspace dimension $m _ { l } ( { \mathfrak { H } } )$ can become active, resulting in $\bar { \sigma } _ { i } ^ { ( l ) } > 0$ for some $i > m _ { l } ( \mathfrak { H } )$ . The leakage score (LS) with respect to H quantifies the fraction of spectral energy contained in these spectral tail directions:

$$
\mathrm { L S } _ { \mathfrak { H } } ^ { ( l ) } ( \mathcal { G } ) = \frac { 1 } { Z _ { l } ( \mathcal { G } ) } \sum _ { i > m _ { l } ( \mathfrak { s } ) } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } , \qquad Z _ { l } ( \mathcal { G } ) = \sum _ { i } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } + \varepsilon _ { \mathrm { n o r m } } .\tag{4}
$$

The numerator measures the spectral tail energy beyond the target fixed-subspace dimension. By default, $Z _ { l } ( \mathcal G )$ normalizes this quantity by the total spectral energy at degree l, enabling comparison across degrees. Setting $Z _ { l } ( \mathcal { G } ) = 1$ instead recovers the unnormalized spectral tail energy. To aggregate responses over a selected set $\Lambda \subseteq \{ 0 , \ldots , L \}$ , we define the pooled leakage score and its negative-log transformation (pLS):

$$
\mathrm { L S } _ { \Lambda , { \mathfrak { s } } } ( { \mathcal { G } } ) = \frac { 1 } { | \Lambda | } \sum _ { l \in \Lambda } \mathrm { L S } _ { { \mathfrak { s } } } ^ { ( l ) } ( { \mathcal { G } } ) , \qquad \mathrm { p L S } _ { \Lambda , { \mathfrak { s } } } ( { \mathcal { G } } ) = - \log _ { 1 0 } ( \mathrm { L S } _ { \Lambda , { \mathfrak { s } } } ( { \mathcal { G } } ) + \varepsilon _ { \log } ) .\tag{5}
$$

We set $\varepsilon _ { \mathrm { n o r m } } = \varepsilon _ { \mathrm { l o g } } = 1 0 ^ { - 1 0 }$ . The logarithmic transformation improves sensitivity to small leakage values and yields a maximum pLS value of 10 when $\mathrm { L S } _ { \Lambda , { \mathfrak { H } } } ( { \mathcal { G } } ) = 0$ . Therefore, lower LS and higher pLS indicate stronger agreement with the rank-template constraints imposed by the target symmetry.

These target-dependent scores are derived from the same SymFP, which is intrinsic to the equivariant representation and constructed independently of any target point group. The target group afects the leakage score only through the fixed-subspace dimensions $m _ { l } ( { \mathfrak { H } } )$ , which determine the spectral tail at each degree. Consequently, a single SymFP supports symmetry analysis across multiple target groups without recomputing the underlying representations or singular values.

## Subgroup monotonicity

The SymFP framework also satisfies subgroup monotonicity in both tasks (Fig. 1c). A subgroup admits a fixed subspace of equal or larger dimension, yielding a componentwise equal or larger rank template. For the same input, degree set and pooling weights, it therefore yields an equal or lower leakage score and an equal or higher negative-log leakage score. Thus, whenever ${ \mathfrak { H } } \leq { \mathfrak { H } } ^ { \prime }$ , the following three monotonicity relations hold:

$$
\mathrm { R a n k T e m p l a t e } ( \mathfrak { H } ) \geq \mathrm { R a n k T e m p l a t e } ( \mathfrak { H } ^ { \prime } ) , \quad \mathrm { L S } _ { \Lambda , \mathfrak { H } } ( \mathcal { G } ) \leq \mathrm { L S } _ { \Lambda , \mathfrak { H } ^ { \prime } } ( \mathcal { G } ) , \quad \mathrm { p L S } _ { \Lambda , \mathfrak { H } } ( \mathcal { G } ) \geq \mathrm { p L S } _ { \Lambda , \delta ^ { \prime } } ( \mathcal { G } ) ,\tag{6}
$$

where the rank-template inequality is understood componentwise.

## Parameter-free equivariant GNN

SymFP relies only on the algebraic structure of equivariant representations and is therefore applicable to any equivariant GNN architecture. To facilitate reuse and evaluation, we provide a parameter-free (PF) equivariant GNN (Algorithm 1). It combines node and edge attributes, fixed radial bases and real spherical harmonics into node, edge and hybrid representations.

Algorithm 1: PF Construction at Fixed Degree l   
Data: Geometric graph $\mathcal { G } ( H , \vec { X } ; A ) = \mathcal { G } ( \nu ; \mathcal { E } )$ with node attributes $\pmb { h } _ { i } \in \mathbb { R } ^ { d _ { v } }$ , node   
coordinates $\vec { \pmb { x } } _ { i } \in \mathbb { R } ^ { 3 }$ and edge attributes $e _ { i j } \in \mathbb { R } ^ { d _ { e } }$ on $\mathcal { E } .$ Predefined RBFs $\{ \rho _ { n } \} _ { n = 1 } ^ { N _ { V } }$   
and $\{ \eta _ { m } \} _ { m = 1 } ^ { N \varepsilon } ,$ and radial normalization scales $Z _ { \nu }$ and $Z _ { \mathcal { E } }$   
Result: Node block $\tilde { V } _ { \nu } ^ { ( l ) } \in \mathbb { R } ^ { ( 2 l + 1 ) \times C \nu }$ , edge block $\tilde { V } _ { \mathcal { E } } ^ { ( l ) } \in \mathbb { R } ^ { ( 2 l + 1 ) \times C _ { \mathcal { E } } }$ , and hybrid block   
$\tilde { \pmb { V } } ^ { ( l ) } \in \mathbb { R } ^ { ( 2 l + 1 ) \times ( C _ { \mathcal { V } } + C _ { \mathcal { E } } ) }$   
// ----- Node Representations   
1 $C _ { \nu } = d _ { v } N _ { \nu }$   
2 for $i \in \mathcal V$ do   
3 $\pmb { a } _ { i } = [ \pmb { h } _ { i a } \rho _ { n } ( \| \pmb { \vec { x } } _ { i c } \| / ( Z _ { \mathcal { V } } + \varepsilon _ { 0 } ) ) ] _ { ( a , n ) } \in \mathbb { R } ^ { C _ { \mathcal { V } } }$ $/ / \quad Z _ { \mathcal { V } }$ : mean radius (default)   
4 end   
5 $\begin{array} { r } { \tilde { { \bf V } } _ { \nu } ^ { ( l ) } = \sum _ { i \in \mathcal { V } } Y ^ { ( l ) } ( \hat { { \bf x } } _ { i c } ) \otimes { \bf a } _ { i } \in \mathbb { R } ^ { ( 2 l + 1 ) \times C \nu } } \end{array}$ $/ / \textrm { \textit { Y } } ^ { ( l ) }$ : real spherical harmonics   
$/ /$ Edge Representations   
6 $C _ { \mathcal { E } } = ( 2 d _ { v } + d _ { e } ) N _ { \mathcal { E } }$   
7 for $( i , j ) \in \mathcal { E }$ do   
8 $\pmb q _ { i j } = [ h _ { i } , h _ { j } , \pmb e _ { i j } ] \in \mathbb { R } ^ { 2 d _ { v } + d _ { e } }$   
9 $\bar { \pmb { b } _ { i j } } = [ \pmb { q } _ { i j a } \eta _ { m } ( \| \bar { \pmb { x } } _ { i j } \| / ( Z _ { \mathcal { E } } + \varepsilon _ { 0 } ) ) ] _ { ( a , m ) } \in \mathbb { R } ^ { C _ { \mathcal { E } } } \quad / /$ Z<sub>E</sub>: mean edge length (default)   
10 end   
11 $\begin{array} { r } { \tilde { V } _ { \mathcal { E } } ^ { ( l ) } = \sum _ { ( i , j ) \in \mathcal { E } } Y ^ { ( l ) } ( \hat { { \pmb x } } _ { i j } ) \otimes { \pmb b } _ { i j } \in \mathbb { R } ^ { ( 2 l + 1 ) \times C \varepsilon } } \end{array}$ $/ / \textrm { \textit { Y } } ^ { ( l ) }$ : real spherical harmonics   
// ----- Hybrid Representations   
12 $\tilde { \mathbf { V } } ^ { ( l ) } = [ \tilde { V } _ { \gamma } ^ { ( l ) } , \tilde { V } _ { \varepsilon } ^ { ( l ) } ] \in \mathbb { R } ^ { ( 2 l + 1 ) \times ( C _ { \mathcal { V } } + C _ { \mathcal { E } } ) }$

## 2.2 Discrete point-group identification across equivariant GNNs

We first evaluate SymFP for discrete point-group identification across diferent equivariant GNN architectures, focusing on whether the observed rank sequences match the theoretical rank templates. Across architectures and training regimes, rank-template matching yields accurate discrete point-group identification, including at random initialization.

Specifically, we construct a synthetic dataset covering all 32 crystallographic point groups and compare PF with four learnable equivariant GNNs: HEGNN<sup>13</sup>, $\mathrm { T F N ^ { 1 8 } }$ , MACE<sup>19</sup> and Equiformer<sup>20</sup>, across diferent representation degrees. For the learnable models, we consider three parameter settings: random initialization, coordinate-denoising pretraining inspired by Uni-Mol<sup>21</sup>, and supervised training with point-group labels. We evaluate performance using two metrics. Point-group identification accuracy measures the fraction of structures assigned to their true point group by nearest-template matching under the $\ell _ { 1 }$ distance. Rank-template match rate measures the fraction of samples whose numerical rank sequences exactly match the true group’s rank template at every selected degree.

In Fig. 2a, point-group identification accuracy reaches or closely approaches the 30/32 templatebased limit at $l = 0 , \ldots , 6$ , with consistently high rank-template match rates across models and training states. Extending to $l = 0 , \ldots , 9$ raises accuracy to 99.06–100% across all models, consistent with the distinct rank templates of all 32 crystallographic point groups at these degrees (Supplementary Table C.3). The observed rank patterns are also reproducible across random seeds (Supplementary Table C.4). In Fig. 2b, PF and all randomly initialized GNNs attain the same identification accuracy at a shared relative rank tolerance, showing that a common numerical threshold sufices across models (Supplementary Table C.5). In the PF ablation (Fig. 2c), combining node and edge representations reaches the identification limit at $l = 0 , \ldots , 6$ , and varying the number or width of the radial bases leaves identification accuracy unchanged (Supplementary Table C.6). We therefore use the combined representation with 16 radial bases per branch in subsequent experiments.

![](images/7857a14a487a01cdd926afece4408a999d6deb3b18dcb9d0a86f14b51f43877d.jpg)

b  
![](images/977c3e3f5f44e82a9c9652873ce1b1ff7ade60974f44e8fae5c64901b2f97f56.jpg)

c  
![](images/be9e1c217533ef8d26d545fa5038e5cf6a744e8f492e5f81848a4afa5820ce14.jpg)  
Figure 2. Discrete point-group identification across equivariant GNNs. a, Point-group identification accuracy and rank-template match rate across GNN architectures and training states. The 30/32 reference line marks the identification accuracy limit at $l = 0 , \ldots , 6$ on this balanced benchmark. Owing to computational constraints, learnable GNNs are evaluated only at random initialization for $l = 0 , \ldots , 9 .$ . b, Sensitivity of point-group identification accuracy to the relative rank tolerance for randomly initialized models at $l = 0 , \ldots , 6 ;$ the shared tolerance $3 \times 1 0 ^ { - 5 }$ used across models is indicated. c, Ablations of the PF node and edge branches and radial-basis count and width at $l = 0 , \ldots , 6$ . N denotes the number of radial bases per branch, and $\Delta$ their centre spacing.

These empirical patterns align with the theoretical analysis presented later. For a fixed input and an architecture satisfying the stated regularity conditions, if the rank sequence matches the theoretical rank template at one parameter setting, then it does so for almost every parameter choice (Cors. 11 and 12). This result explains the consistency across random initializations and training states, while experiments show that accurate discrete point-group identification is also achieved across GNNs.

![](images/68dd09bd3c2f6804fa0fdc8a73fb78bf743a5f791e90b6973605b5b7863d28fd.jpg)

![](images/2e4b2c8414ca3a43516eb22404d13f847e90cdfab993794ebb7507c5b2388883.jpg)

## 2.3 Spectral responses track symmetry breaking and distinguish deformation modes

a  
b  
![](images/3b995b2d0055b8d3ef4d6fb55a5e789d22b62acdb9dd1414e18b372044938809.jpg)

![](images/4721b2ab439cad7d094951d7a553b30e1ae705ccbdd3b359a7be233123060e48.jpg)

![](images/1e1c81065d45ad83c431e473003b6139773dde8fcac73023dafe43eb462060b3.jpg)

c  
![](images/83f4bc38436df7fa8769e42bb5d3040bc6677325c807c433164e5418674e059e.jpg)

C6  
d  
![](images/6f39f6ff37573f90efc54b05a180df0bf3943ca0940068e8e77a9a729a8ee029.jpg)  
e

![](images/4db57d2fa3ffd85160f4f19fb3aa86bb8b408f052582203cf843d040d0ba87e3.jpg)

![](images/f922eca84ffbd05312da54b951d7c082d913c7d87687113d99a41ae78fd46843.jpg)

![](images/cb0a842e8b4cbae0628cb36cf3ffee2526b6667f003db26ff096e2854c6025f9.jpg)

![](images/2c516de64db6ad7dde303ce4d4fa169b8dac4e8f15a58754e609954ec86be54b.jpg)

![](images/07352ae086fa763ad678ba27284a2dfffafa1cc6a0b3e9cea6a64af54f4f5ef4.jpg)

![](images/bb96e5ddd5502c875795ee0ed473f19ed744cc0e3eae972963893f0f42c95d82.jpg)

![](images/d1059cb2bf61875bdf01d5b22a9ce87bc702c8abf9d14c362b71471c267e33bd.jpg)

Figure 3. SymFP tracks symmetry breaking and distinguishes deformation modes. All panels use PF at $l = 0 , \ldots , 6 . \ \mathbf { a } ,$ Symmetry breaking immediately activates new singular values, whose number matches the change in the rank template induced by the deformation. Structura illustrations show the three deformations, and curves show the corresponding singular-value trajectories at selected degrees. b, Along the subgroup chain, spectral tail energies relative to the target point groups rise in sequence and follow the theoretically predicted subgroup monotonicity.

Vertical dashed lines mark the symmetry-breaking transitions. c, Four deformation modes of the same $\mathrm { C _ { 6 } }$ structure with equal RMSD. d, Self-match RMSD between a deformed structure and its 60<sup>◦</sup>-rotated copy, computed using corresponding points and illustrated for the in-plane stretch. e, Self-match RMSD (top) and degree-resolved relative tail energy (bottom) as functions of deformation amplitude for the four modes in c. The self-match RMSD curves for the four modes largely overlap in pairs, whereas spectral onset degrees further distinguish their angular diferences. Relative tail energies below $1 0 ^ { - 9 }$ are displayed in a separate numerical-floor strip.

We further evaluate SymFP for continuous symmetry analysis using three controlled deformation experiments: single-step symmetry breaking, progressive symmetry breaking along a subgroup chain, and distinct deformation modes applied to the same structure. Together, these experiments demonstrate the sensitivity of SymFP to symmetry breaking, confirm subgroup monotonicity, and show how degree-resolved spectra distinguish diferent deformation modes.

In the single-step symmetry-breaking experiment (Fig. 3a), the number of activated singular values agrees exactly with the rank template predictions as symmetry is reduced. These singular values rise rapidly at the onset of symmetry breaking and then level of, demonstrating that the singular-value spectrum is highly sensitive to symmetry breaking and can continuously track its evolution. A similar response is observed in the progressive symmetry-breaking experiment along a subgroup chain (Fig. 3b), where the spectral tail energy for each target group rises sharply upon symmetry breaking before levelling of, while the relative magnitudes across groups verify subgroup monotonicity.

In the deformation-mode comparison (Fig. 3c), we apply four deformations with equal RMSD to the same structure: axial staircase, in-plane stretch, in-plane radial, and axial pucker. Although these deformations contain diferent Fourier components, their self-match RMSD responses largely collapse onto two curves (Fig. 3e). By contrast, the degree-resolved spectra reveal distinct patterns, with onset degrees of l = 1, 2, 3, and 3, respectively. These onset degrees agree with the lowest nonzero azimuthal orders of the corresponding deformation fields, consistent with the spherical-harmonic constraint $l \geq | m |$ . Thus, degree-resolved spectra can distinguish deformation modes that are dificult to distinguish using a single RMSD measure.

We repeat the subgroup-chain and deformation-mode experiments using four randomly initialized learnable equivariant GNNs (Extended Data Fig. 1). All models reproduce the subgrouptransition responses observed with PF and share the same onset degrees across deformation modes. The only exception is the axial-pucker response of HEGNN, which begins at l = 4 rather than $l = 3$ Introducing node types that preserve the original symmetry restores the onset to l = 3 (Supplementary Table D.9 and Supplementary Note D.1). These results verify the generality of the symmetry constraints across diferent GNN architectures, while showing that the specific spectral responses still depend on the model architecture and graph input.

## 2.4 Spectral symmetry scores retrieve molecular energy extrema

Having validated the theoretical properties of SymFP, we next explore its practical applications, starting with the retrieval of energy extrema among molecular conformations. This task is motivated by the observation that symmetric conformations often correspond to energy extrema <sup>22,23</sup>. We use a torsion-scan dataset of 86 aromatic molecules, each containing two planar groups connected by a rotatable bond<sup>24,25</sup>. Each molecule has 24 conformations sampled at diferent torsion angles, yielding 2,064 conformations with energy labels (Fig. 4a).

![](images/13a1ae565c6e9354d2593c962756ce9ae8e71d0e75fc8d9393c3fd6a5fd26554.jpg)  
b

![](images/088ef54c573c3873f896f03548cf33e3e5a75e280a6143fa966c88ae110dc374.jpg)

c  
![](images/a54e0c80fa6eb0bdc762d1e9949ddbb0e0679b0dc0cc7be925e131e8b67d8f89.jpg)

d  
![](images/8ef9be2e38a4822233035029595c92cd137b0f15601f0389909ae4ee4f1a64f3.jpg)  
e

![](images/bf89a0a05328196857819a03ea8af28f7ea1e310b523c0202b9d6469df131a3f.jpg)

f  
![](images/3a86fe9f38749e8aaa8678f84533706513d8aef74489947f946bef59544ff739.jpg)

![](images/7a613b952b635019acb4f8046432d113c15eb196f6aca5d815846b39ffa6009b.jpg)

![](images/5cbbd100a30a03416bc3199855e591f3533dd4ffb1c4b6012637701bb520e199.jpg)  
Figure 4. Symmetry and energy-extremum retrieval along biaryl torsions. Throughout,

SymFP uses $\mathrm { P F }$ representations with uniform pooling over natural-parity degrees l = 0, 2, 4, 6. a, Biphenyl torsion schematic and evaluated cohort: 86 molecules, 24 conformations each at $1 5 ^ { \circ }$ increments, with deposited coordinates and quantum-mechanical (QM) energies. b, Biphenyl conformations ranked by descending even-degree negative-log leakage score (pLS) for the $\mathrm { D _ { 2 h } }$ target. Coral downward and blue-grey upward triangles mark minima and maxima in the QM energy profile; grey dots mark other conformations. Shading marks the first eight conformations, and the arrow indicates increasing the selection budget. The four maxima rank 1–4 and the four minima rank 7–10, so the first ten of the 24 conformations recover all eight sampled energy extrema. $\mathbf { c } ,$ Energy-extremum recall against the selected fraction of conformations over 94 molecule–target pairs from 86 molecules. Extrema are defined relative to both periodic neighbours in the deposited energy profile. The inset enlarges the 10–25% selection range. Mean areas under the recall–budget curve (AUC) are 0.794, 0.747 and 0.787 for SymFP, CSoM and Engel score, respectively, compared with a random-ranking expectation of 0.5. At a 25% selection budget, the corresponding recall values are 68.8%, 66.9% and 66.3%. d, Mean AUC for minima and maxima separately. For SymFP, CSoM and Engel score, respectively, AUCs are 0.720, 0.647 and 0.735 for minima and 0.867, 0.847 and 0.839 for maxima. e, Per-molecule AUC diferences (SymFP minus baseline). Counts give molecule-level wins, numerical ties and losses; SymFP records more wins than losses against either baseline. Violins show kernel-density estimates restricted to the observed range; green points show means. In c–e, targets are averaged within molecules. Shading in c and error bars in d,e give 95% intervals from 5,000 molecule-bootstrap resamples, pointwise along the curves in c. f, Energy and score profiles for three molecules illustrate the relationships between symmetry scores and energy extrema across diferent methods. Triangles mark energy extrema as in b; grey lines align them across profiles. CSoM is sign-reversed, and each score profile is independently scaled to [0,1], so that higher values indicate greater target agreement.

Using PF representations, we compute SymFP and rank the conformations by the resulting pLS values in descending order for each nontrivial point group encountered along a molecule’s torsion scan (Fig. 4b). Retrieval performance is summarized by the area under the recall–budget curve (AUC), which plots the fraction of energy extrema recovered against the fraction of ranked conformations selected.

SymFP outperforms CSoM and Engel score<sup>5</sup> in overall energy-extremum retrieval (Fig. 4c), achieving the best performance for maxima while remaining competitive with Engel score for minima (Fig. 4d). Molecule-level comparisons show more wins than losses against both baselines (Fig. 4e), and individual energy and score profiles further illustrate how the methods respond diferently to energy extrema (Fig. 4f). To understand the retrieval diference between maxima and minima, we divide the extrema into four groups according to their type and whether their conformations possess the target symmetry. SymFP achieves the highest mean AUC in three of the four groups, falling slightly below Engel score only for minima without the target symmetry (Extended Data Table 1a). Moreover, 81.1% of maxima possess the target symmetry, compared with only 29.1% of minima (Extended Data Table 1b). This is consistent with biaryl systems such as biphenyl, where highly symmetric conformations can correspond to torsional barriers, while energy minima need not preserve the target symmetry due to competing interactions<sup>23</sup>. The stronger association between symmetry and energy maxima may help explain their better retrieval performance.

We also evaluate SymFP with learnable equivariant GNNs, comparing the four architectures introduced above at random initialization and after total-energy or Uni-Mol-style pretraining. At random initialization, all four GNNs achieve energy-extremum retrieval AUCs close to that of PF, suggesting that equivariant representations already contain symmetry-related structural information useful for this task. Both pretraining objectives enhance spectral contrast by increasing the mean pLS gap between energy maxima and minima (Extended Data Table 2a). They also enhance the contrast between conformations with and without the target symmetry, mainly because the mean pLS decreases more strongly for those without the target symmetry (Extended Data Table 2b). These results show that while the symmetry constraints captured by SymFP remain consistent across architectures and training states, learning can modulate the spectral organization of representations and enhance symmetry-related contrasts.

![](images/09bf4dd1724db0510d9a475298bc07017500c1f6a21ab54f18dbce618228a09e.jpg)

![](images/2b4e4cf427ce124c2a1e87346976ee67ab2085b63b0f4e43507b2ac9dd397649.jpg)

## 2.5 Degree-resolved spectra reveal crystal-field symmetry constraints

b  
c
<table><tr><td>target group</td><td>C4</td><td>S4</td><td>C4v</td><td>D4h</td></tr><tr><td>crystal-field parameters (k=1=2,4,6)</td><td>(1, 3, 3)</td><td>(1, 3,3)</td><td>(1,2,2)</td><td>(1,2,2)</td></tr><tr><td>odd-degree rank template 1=1,3,5</td><td>(1, 1, 3)</td><td>(0,2,2)</td><td>(1,1,2)</td><td>(0,0,0)</td></tr></table>

![](images/4b08b258044504bd0c69500c16c69aadf8837a716b25849fc65e637ddf102505.jpg)

![](images/76559957c65106609a64dec68222dcb2bd876ce2b2c4799f1bf3ba396166dcf6.jpg)

d  
![](images/165bf9f72a2e8161f09985384a90d5481ca472a61b8027913650c04e07cb0ee2.jpg)  
h

e  
![](images/d4469cd4916ee6aaf27661125bb88827aa6627728eed601dbaaab39028837ba4.jpg)  
g

f  
![](images/d9f9863e9549c3a39ff685026b3172f75de50f20f18dfa18a9ac82d53fd4c502.jpg)

![](images/d4932fdf9b70bf2fd1a7a2dba485d0a12fc0a828a222322ca9dc8a1c6a5066c9.jpg)

Figure 5. Crystal-field constraints and local orientation in lanthanide DOTA complexes (PF throughout). Even degrees $l = 2 , 4 , 6$ correspond to the crystal-field ranks, while odd degrees provide complementary geometric symmetry information. a, Water-free $[ \mathrm { D y ( D O T A ) } ] ^ { - }$ cage, representative of seven complexes with a crystallographic four-fold axis. b, Crystal-field parameter counts at $k = l = 2 ,$ 4, 6 and odd-degree rank templates at l = 1, 3, 5 for four tetragonal targets. Highlighted counts correspond to c. c, The same $l = k = 4$ Dy spectrum evaluated against $\mathrm { C _ { 4 } }$ and $\mathrm { C } _ { 4 \mathrm { v } }$ . Bars show normalized squared singular values; purple lines mark the target fixed-subspace dimensions. Green and red bars lie within and beyond the corresponding cutofs, respectively; the red contributions sum to the leakage score (LS). Changing the target from $\mathrm { C _ { 4 } }$ to $\mathrm { C } _ { 4 \mathrm { v } }$ moves the third component into the tail, increasing LS from below 10<sup>−11</sup> to 0.130. d, Degree-resolved LS for the seven water-free structures at even degrees $l = 2 , 4 , 6$ . LS remains at the numerical floor for $\mathrm { C _ { 4 } }$ and $\mathrm { S } _ { 4 } .$ , while the five-parameter targets raise LS at $l = 4 , 6$ but not at $l = 2$ . e, Odd-degree LS at $l = { 1 , 3 , 5 }$ distinguishes $\mathrm { C _ { 4 } }$ from $\mathrm { S } _ { 4 } .$ , despite their identical even-degree responses. f, Heavy-atom LS for the $\mathrm { C _ { 4 } }$ series, water-capped Dy

complex and distorted Tm analogue, against $\mathrm { C _ { 4 } }$ at $l = k = 2 , 4 , 6$ . Both the water-capped Dy complex and the distorted Tm analogue have higher LS than the crystallographic $\mathrm { C _ { 4 } }$ series. In d–f, boxes span the interquartile range across seven structures, lines mark medians, whiskers span the minimum–maximum range and diamonds mark means; Dy and Tm diamonds in f show individual values. $\mathbf { g } ,$ Rigid rotation of the capping water in $\mathrm { \Delta [ D y ( D O T A ) ( H _ { 2 } O ) ] } ^ { \mathrm { ~ } }$ about the Dy–O axis by $\alpha ,$ moving only its two protons. h, Negative-log leakage score (pLS), CSoM and Engel score<sup>5</sup> profiles for four targets along the water rotation. SymFP and Engel score show a dominant 90<sup>◦</sup> period for all four targets. CSoM is nearly constant for $\mathrm { C _ { 4 } }$ and $\mathrm { C _ { 2 } }$ , and has dominant periods of $9 0 ^ { \circ }$ and $4 5 ^ { \circ }$ for $\mathrm { C _ { 2 v } }$ and $\mathrm { C } _ { 4 \mathrm { v } }$ , respectively. Each trace has its own vertical scale. The $\mathrm { C _ { s } }$ result and harmonic content are given in Supplementary Table C.8.

We next investigate the relationship between geometric symmetry and crystal-field constraints in DOTA metal complexes, where the arrangement of ligands strongly influences the electronic properties of the central metal ion<sup>26–28</sup>.

We first consider seven water-free DOTA complexes with a crystallographic four-fold axis $( \mathrm { F i g . \ 5 a ) ^ { 2 9 } }$ . For the $4 f ^ { n }$ electronic configuration, the crystal-field terms responsible for splitting electronic energy levels occur at ranks $k = 2 , 4 , 6$ . At these ranks, the numbers of independent crystal-field parameters match the corresponding entries of the SymFP rank template, as both are determined by the same fixed-subspace dimensions (Fig. 5b). This correspondence is reflected in the degree-resolved spectra, where diferent target groups impose distinct constraints on the spectral tail (Fig. 5c). Across all seven complexes, LS remains at the numerical floor for $\mathrm { C _ { 4 } }$ and $\mathrm { S _ { 4 } }$ , while the more restrictive targets $\mathrm { C } _ { 4 \mathrm { v } }$ and $\mathrm { D } _ { 4 \mathrm { h } }$ yield higher LS at $l = 4 , 6$ , where their parameter counts difer (Fig. 5d). Although $\mathrm { C _ { 4 } }$ and $\mathrm { S _ { 4 } }$ have identical even-degree responses, their odd-degree leakage scores distinguish them (Fig. 5e). The water-capped Dy complex and distorted Tm analogue also show higher LS relative to $\mathrm { C _ { 4 } }$ , reflecting departures from ideal four-fold symmetry (Fig. 5f). Together, these results show that degree-resolved spectra capture the symmetry constraints underlying crystal-field parameters while revealing geometric symmetry information beyond the even-degree crystal-field description.

We further investigate local orientational changes within the coordination cage. The orientation of the capping water is known to afect the low-lying electronic states and magnetic anisotropy of DyDOTA<sup>27,28</sup>. By rotating the capping water about the Dy – O axis while keeping the cage fixed (Fig. 5g), we find that SymFP exhibits a dominant $9 0 ^ { \circ }$ periodicity across the four targets $\mathrm { C _ { 4 } , C _ { 4 v } , C _ { 2 } }$ and $\mathrm { C _ { 2 v } }$ , consistent with the electronic-property periodicity discussed by Nielsen and Sørensen <sup>4</sup>. Engel score shows the same periodicity, whereas CSoM captures it only for the $\mathrm { C _ { 2 v } }$ target<sup>4</sup> (Fig. 5h). This periodicity can be understood from the approximate four-fold symmetry of the coordination cage: a $9 0 °$ global rotation leaves the cage nearly unchanged while shifting the water orientation by $9 0 °$ . Together with the global rotation invariance of SymFP, this explains the similar spectral responses at water orientations separated by $9 0 °$

We also assess the computational eficiency of SymFP on the same system. By avoiding targetorientation searches and reusing spectra across targets, SymFP evaluates seven target groups for Dy-DOTA-water approximately two orders of magnitude faster than the fastest tested Engel implementation and four orders of magnitude faster than CSoM (Extended Data Table 3a).

![](images/195fbdfb5e2f7675da433f5664db3decdb8aec7afc65def1b5a5d9f0580f4cb7.jpg)

2.6 SymFP reveals symmetry variation in large protein assemblies a  
![](images/ac272d3e290525d30c9178fc2dc92af525150d8a8c4994b2cb1254b807aba065.jpg)  
Figure 6. Regional symmetry responses across six experimental p97 structures. Scores use PF at degrees $l = 0 , \ldots , 6$ and common backbone atom identities against a fixed $\mathrm { C _ { 6 } }$ target. Higher pLS indicates stronger agreement with the $\mathrm { C } _ { 6 }$ rank-template constraints. a, Six structures of p97 bound to its adaptor protein UBXD1, reconstructed with $\mathrm { C _ { 1 } }$ symmetry, with the closed structure (8FCL) as reference <sup>30</sup>. The H4-bound structure has density assigned to the UBXD1 H4 helix at D2. The lariat mutation is in UBXD1. Colours distinguish the six p97 chains. b, Regional analysis with 12 D1 and 14 D2 overlapping 40-residue windows, each combining corresponding segments from all six chains. c, The D1 and D2 example windows from

b contain the respective pore loops, with the surrounding domains shown in grey. Arrows mark local surface gaps between adjacent chains that widen or narrow relative to the closed reference. d, Changes in the negative-log leakage score (pLS) for each window in D1 (left) and D2 (right) relative to 8FCL, $\Delta \mathrm { p L S } = \mathrm { p L S } _ { \mathrm { c o m p a r i s o n } } - \mathrm { p L S } _ { \mathrm { r e f e r e n c e } } .$ The H4-bound structure shows both increases and decreases in regional D1 pLS despite its nearly unchanged whole-D1 score in a. The structure associated with the UBXD1 lariat mutant shows increased pLS in all D1 windows and decreased pLS in most D2 windows. The horizontal axis gives window starting residues. Dashed outlines identify the example windows in $\mathbf { b } , \mathbf { c } ;$ brackets labelled res < 40 mark windows with fewer than 40 common residues per chain. Numbered cells correspond to e. e, Cα-trace superpositions of the four numbered windows, with the reference in teal, the comparison in purple and the surrounding reference domain in grey. Each superposition uses a single rigid fit of the corresponding Cα atoms across all six chains. All views share the reference orientation and scale (scale bar, 20 <sup>˚</sup>A). Selection and scoring protocols are given in Supplementary Note C.6.1.

We further apply SymFP to large protein assemblies, where symmetry governs structural organization and may evolve during functionally relevant conformational rearrangements<sup>30,31</sup>. We investigate four systems spanning distinct architectures and scales: the AAA+ ATPase p97 (∼4K residues)<sup>30</sup>, influenza haemagglutinin (HA, ∼1K) <sup>32</sup>, the kainate receptor GluK2 $( \mathrm { \sim 3 K ) ^ { 3 1 } }$ , and the echovirus 18 (E18) capsid (>30K) <sup>33</sup>. We use the PF implementation of SymFP to characterize symmetry variation across structural regions, conformational states, and levels of assembly integrity. Symmetry is quantified by pLS, using backbone atoms for p97, HA, and GluK2 and Cα atoms for E18.

We first examine six experimental structures of p97 bound to its adaptor protein UBXD1, using the closed conformation as a reference to assess symmetry changes in the two hexameric rings, D1 and D2, relative to $\mathrm { C _ { 6 } \left( F i g . 6 a \right) }$ . To resolve regional diferences, we divide each ring into overlapping sequence windows containing corresponding segments from all six chains and compute pLS for each window (Fig. 6b). Both the open and meta states exhibit lower pLS than the closed reference across all sampled D1 and D2 windows. In contrast, the H4-bound conformation, with density assigned to the UBXD1 H4 helix at D2, has nearly the same whole-D1 score as the closed reference, despite showing both increases and decreases in regional pLS (Fig. 6c,d). The conformation associated with the UBXD1 lariat mutant exhibits a diferent pattern, with pLS increasing across all D1 windows but decreasing in most D2 windows. These opposing D1 and D2 responses persist across the tested graph neighbourhoods and window lengths, with shorter windows resolving finer local variations (Extended Data Fig. 2). Superpositions of selected windows further reveal the structural rearrangements underlying these regional responses (Fig. 6e). Together, these results show that regional symmetry profiles can reveal local rearrangements and opposing symmetry changes that whole-domain scores may obscure.

We extend the analysis to HA, GluK2 and the E18 capsid, covering diferent protein architectures and structural states. During the transition from HA state III to IV, the global $\mathrm { C _ { 3 } }$ pLS increases, whereas the HA1 head and HA2 stem respond in opposite directions, with pLS decreas ing in HA1 and increasing in HA2 (Extended Data Fig. 3). In GluK2, the $\mathrm { C _ { 4 } \ p L S }$ is generally higher near the transmembrane region than in the upper extracellular regions, both across spatial layers and at the domain level, reflecting diferences in subunit arrangement along the receptor axis (Extended Data Fig. 4). This pattern persists across the tested layer thicknesses. For the E18 capsid, the degree-six pLS relative to the icosahedral rotation group I decreases progressively from the intact structure to those lacking one, two or three pentamers (Extended Data Fig. 5).

Together, these examples show that SymFP captures symmetry variation across protein regions and conformational states, as well as changes associated with the integrity of large molecular assemblies.

## 3 Discussion

SymFP translates symmetry-imposed constraints on equivariant representations into structural information, providing a unified framework for discrete point-group identification and continuous symmetry analysis. These constraints arise from the algebraic structure of equivariant representations rather than any particular message-passing implementation, establishing a common basis for symmetry analysis across equivariant GNN architectures. The PF implementation constructs representations directly from geometry, enabling SymFP to serve as a readily accessible and reusable tool for structural symmetry characterization.

This formulation allows representation designs and equivariant architectures to be tailored to diferent systems and analysis objectives. At the analysis level, degree-dependent spectral responses distinguish diferent forms of structural change, while regional analysis localizes variations that may be obscured by whole-structure scores.

More broadly, SymFP provides a diferent perspective on symmetry analysis. Existing approaches typically rely on predefined target symmetries and quantify deviations from corresponding reference states. In contrast, SymFP reveals how structural symmetry is encoded in the spectra of equivariant representations, without requiring target-specific geometric references or orientation searches. This perspective provides a general framework for analysing symmetry beyond individual point groups and may extend to broader aspects of symmetry, including space groups, chirality, and helical symmetries. More importantly, it opens opportunities for studying how symmetry constraints reorganize during complex structural transformations, including phase transitions, defect evolution, and amorphization.

## 4 Methods

## 4.1 Preliminaries

## Geometric graphs and point-group actions

A geometric graph with N nodes is denoted by $\mathcal { G } ( H , \vec { X } ; A ) = \mathcal { G } ( \nu ; \mathcal { E } )$ , where $\mathcal { V } : = \{ 1 , \ldots , N \}$ is the node set, $\pmb { H } : = [ \pmb { h } _ { 1 } , \dots , \pmb { h } _ { N } ] \in \mathbb { R } ^ { d _ { v } \times N }$ contains node attributes, $\vec { \pmb { X } } : = [ \vec { \pmb { x } } _ { 1 } , \dots , \vec { \pmb { x } } _ { N } ] \in \mathbb { R } ^ { 3 \times N }$ contains 3D coordinates, and $\pmb { A } \in \mathbb { R } ^ { N \times N }$ is the adjacency matrix. Both node attributes and coordinates are stored column-wise. The directed edge set $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ is induced by A, with $( i , j ) \in \mathcal { E }$ if and only if $A _ { i j } \neq 0$ . For undirected neighbourhoods, both $( i , j )$ and $( j , i )$ are included. Edge attributes $e _ { i j }$ may be assigned to edges when needed.

As translations are removed by centring, we restrict attention to O(3) actions. For ${ \mathfrak { h } } \in { \mathrm { O } } ( 3 )$ we write h · G for its action on G, with coordinates transformed as $O _ { \mathfrak { h } } \vec { X }$ , where $S _ { \mathfrak { h } } \in \mathbb { R } ^ { 3 \times 3 }$ is the corresponding orthogonal matrix.

## Compatibility and stabilizers

Definition 1 (Graph Compatible with a Finite Point Group). Let ${ \mathfrak { H } } \leq \mathrm { O } ( 3 )$ be a finite subgroup. A geometric graph $\mathcal { G } ( H , \vec { X } ; A ) = \mathcal { G } ( \nu ; \mathcal { E } )$ is compatible with H if, for every ${ \mathfrak { h } } \in { \mathfrak { H } }$ with orthogonal action $O _ { \mathfrak { h } }$ , there exists a permutation matrix $P _ { \mathfrak { h } } \in S _ { N }$ such that

$$
O _ { \mathfrak { h } } \vec { X } = \vec { X } P _ { \mathfrak { h } } , \qquad \pmb { H } P _ { \mathfrak { h } } = \pmb { H } , \qquad \pmb { P } _ { \mathfrak { h } } ^ { \top } \pmb { A } P _ { \mathfrak { h } } = \pmb { A } ,\tag{7}
$$

and, when edge attributes are present, $\begin{array} { r } { e _ { i j } = e _ { \pi _ { \mathfrak { h } } ( i ) \pi _ { \mathfrak { h } } ( j ) } } \end{array}$ for all $( i , j ) \in \mathcal { E }$ , where $\pi _ { \mathfrak { h } }$ is the permutation associated with $P _ { \ u }$ . We then write h $\mathcal { A } = \mathcal { G }$ and denote the set of graphs compatible with H by G(H).

The equality ${ \mathfrak { h } } \cdot { \mathcal { G } } = { \mathcal { G } }$ is understood up to node relabelling that preserves attributes.

Definition 2 (Stabilizer). The stabilizer of a geometric graph G is $\operatorname { S t a b } ( { \mathcal { G } } ) : = \{ { \mathfrak { g } } \in \operatorname { O } ( 3 ) : { \mathfrak { g } } \cdot { \mathcal { G } } =$ G}. It is a subgroup of O(3) and, when finite, gives the exact point group of G.

Exact point-group identification statements below assume a finite stabilizer.

## Irreducible representations ofO(3)

Every ${ \mathfrak { h } } \in { \mathrm { O } } ( 3 )$ has a unique decomposition ${ \mathfrak { h } } = { \mathfrak { r } } { \mathfrak { m } }$ , where $\mathfrak { r } \in \mathrm { S O } ( 3 )$ and $\mathfrak { m } \in \mathrm { C } _ { \mathrm { i } }$ , with $\mathrm { C _ { i } } = \{ \mathfrak { e } , \mathfrak { i } \}$ denoting the two-element inversion group. Thus, $\mathrm { O ( 3 ) } \cong \mathrm { S O ( 3 ) } \times \mathrm { C } _ { \mathrm { i } }$ . For each degree $l \in  { \mathbb { N } } _ { 0 } =$ $\{ 0 , 1 , \ldots \}$ and parity $p \in \{ 1 , - 1 \}$ , define

$$
\rho ^ { ( l , p ) } ( \mathfrak { h } ) : = \sigma ^ { ( p ) } ( \mathfrak { m } ) D ^ { ( l ) } ( \mathfrak { r } ) ,\tag{8}
$$

where $\sigma ^ { ( p ) } ( \mathfrak { e } ) = 1 , \sigma ^ { ( p ) } ( \mathfrak { i } ) = p ;$ , and ${ \cal D } ^ { ( l ) } ( \mathbf { r } ) \in \mathbb { R } ^ { ( 2 l + 1 ) \times ( 2 l + 1 ) }$ is the real Wigner-D matrix of degree l, following the real spherical-harmonic basis and sign convention of the equivariant-operator library used in our implementation<sup>17</sup>. Throughout, we use natural-parity representations with $p _ { l } = ( - 1 ) ^ { l }$ and write $\rho ^ { ( l ) } : = \rho ^ { ( l , ( - 1 ) ^ { l } ) }$ . We denote the degree set $\{ 0 , \ldots , L \}$ by $\Lambda _ { L }$ . Degree-l real spherical harmonics transform under $\rho ^ { ( l ) }$ and satisfy ${ \cal Y } ^ { ( l \bar { ) } } ( - \vec { \bf x } ) = \bar { ( - 1 ) } ^ { l } { \cal Y } ^ { ( l ) } ( \vec { \bf x } ) ^ { 3 4 }$

## 4.2 Related work

## Existing symmetry measures

Classical continuous symmetry measures (CSMs) provide the standard geometry-level approach to approximate symmetry. In an abstract distance-based formulation, a CSM for a target point group H measures the distance from a geometric graph $\mathcal { G }$ to the nearest H-compatible realization,

$$
\operatorname { C S M } _ { 5 } ( \mathcal { G } ) : = \operatorname* { i n f } _ { \mathcal { G } ^ { \star } \in \mathbb { G } ( 5 ) } d ( \mathcal { G } , \mathcal { G } ^ { \star } ) ,\tag{9}
$$

where the infimum runs over the geometric graphs compatible with H in the sense of Def. $1 ^ { 1 , 2 }$ The comparison keeps the graph structure and attributes fixed while varying the coordinates under the allowed node correspondences. Subsequent work developed analytical formulas for selected symmetry families<sup>35</sup>, accelerated exact computation through structure-preserving permutations <sup>36</sup>, specialized the machinery to protein homomers <sup>37</sup>, extended it to large molecular structures through approximate algorithms<sup>38</sup>, and packaged the methodology into modern software for quantitative structural analysis <sup>39</sup>. Continuous shape measures (CShM) compare a coordination sphere with a reference polyhedron of the same vertex count <sup>40</sup>. More recently, the continuous symmetry operation measure (CSoM) replaces the nearest-symmetric reconstruction with an average of operationwise deviations over the target group<sup>3,4</sup>:

$$
\operatorname { S o M } _ { \mathfrak { h } } ( \mathcal { G } ) : = d ( \mathcal { G } , \mathfrak { h } \cdot \mathcal { G } ) , \qquad \operatorname { C S o M } _ { \mathfrak { s } } ( \mathcal { G } ) : = \frac { 1 } { | \mathfrak { H } \setminus \{ \mathfrak { e } \} | } \sum _ { \mathfrak { h } \in \mathfrak { s } \backslash \{ \mathfrak { e } \} } \operatorname { S o M } _ { \mathfrak { h } } ( \mathcal { G } ) .\tag{10}
$$

The CSoM average in Eq. (10) assumes $\mathfrak { H } \ne \{ \mathfrak { e } \}$ . CSoM evaluates target operations in the coordinate frame of the structure, requiring the target orientation to be supplied or determined.

The operation-based view has a harmonic analogue. Replacing the coordinate-space discrepancy by a squared residual between an equivariant harmonic embedding and its transformed copy gives a CSoM-style comparison in representation space. Averaging over the full target group yields twice the energy outside its oriented fixed subspaces.

Engel’s harmonic construction uses the complementary retained energy for point-group analysis<sup>5</sup>. The geometric graph enters through a normalized harmonic embedding $\tilde { \pmb { v } } ( \mathcal { G } ) : = ( \tilde { \pmb { v } } ^ { ( l ) } ( \mathcal { G } ) ) _ { l }$ the spherical-harmonic coeficients of the pooled distribution of bond directions on the sphere, with one coeficient vector at each retained degree and $\begin{array} { r } { \sum _ { l } \| \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } = 1 } \end{array}$ . A spatial realization of the target group then determines the corresponding projector: averaging $\rho ^ { ( l ) }$ over an oriented copy $\mathfrak { r } \mathfrak { H } \mathfrak { r } ^ { - 1 }$ of the target group,

$$
P _ { 5 } ^ { ( l ) } ( \mathfrak { r } ) : = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { r } \mathfrak { h } \mathfrak { r } ^ { - 1 } ) ,\tag{11}
$$

yields the orthogonal projector onto the corresponding fixed subspace<sup>41</sup>. Write $\alpha _ { 5 } ^ { ( l ) } ( \mathcal { G } ; \mathfrak { r } ) \ : =$ $\| \boldsymbol { P } _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } ) \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 }$ for the energy retained in this fixed subspace at degree l, and $\alpha _ { \mathfrak { H } }$ for the sum over the retained degrees. Averaging the squared harmonic residuals over the full target group, including the identity, gives

$$
R _ { 5 } ( \mathcal { G } ; \mathfrak { r } ) : = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \sum _ { l } \| \tilde { \pmb { v } } ^ { ( l ) } - \rho ^ { ( l ) } ( \mathfrak { r } \mathfrak { h } \mathfrak { r } ^ { - 1 } ) \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } = 2 ( 1 - \alpha _ { 5 } ( \mathcal { G } ; \mathfrak { r } ) ) .\tag{12}
$$

The derivation is given in Supplementary Note B.1.

The trace of ${ \breve { P } } _ { 5 } ^ { ( l ) } ( \mathfrak { r } )$ equals the dimension of the degree-l fixed subspace, denoted $m _ { l } ( { \mathfrak { H } } )$ . The normalization in Engel’s construction<sup>5</sup> compares retained energy against a baseline determined by the target. Under an equal-energy assumption over the harmonic coordinates, this baseline is

$$
\beta _ { 5 } ^ { ( l ) } : = \frac { \mathrm { t r } ( \pmb { P } _ { 5 } ^ { ( l ) } ( \mathbf { r } ) ) } { 2 l + 1 } , \qquad \beta _ { 5 } : = \frac { \sum _ { l } \mathrm { t r } ( \pmb { P } _ { 5 } ^ { ( l ) } ( \mathbf { r } ) ) } { \sum _ { l } ( 2 l + 1 ) } = \frac { \sum _ { l } m _ { l } ( \pmb { \mathfrak { H } } ) } { \sum _ { l } ( 2 l + 1 ) } .\tag{13}
$$

The traces, and hence $\beta _ { 5 }$ , are invariant under target reorientation. For $\beta _ { 5 } < 1$ , the normalized retained-energy score at a fixed orientation is

$$
\mathrm { E s } _ { \mathfrak { H } } ( \mathcal { G } ; \mathfrak { r } ) : = \frac { \alpha _ { \mathfrak { H } } ( \mathcal { G } ; \mathfrak { r } ) - \beta _ { \mathfrak { H } } } { 1 - \beta _ { \mathfrak { H } } } .\tag{14}
$$

Following the orientation search adopted in subsequent work by Engel and colleagues<sup>15</sup>, we use Engel score

$$
\mathrm { E s } _ { \mathfrak { H } } ( { \mathcal G } ) : = \operatorname* { m a x } _ { { \mathfrak { r } } \in \mathrm { S O } ( 3 ) } \mathrm { E s } _ { \mathfrak { H } } ( { \mathcal G } ; { \mathfrak { r } } ) ,\tag{15}
$$

with one spatial rotation shared across all retained degrees. Larger values indicate greater retained energy.

Alongside these general constructions, a number of methods are specialized to particular settings. A maximum-symmetry search returns the largest point group consistent with a predefined tolerance<sup>42</sup>, and related procedures assign a point group to a nearly symmetric molecule <sup>43,44</sup>. Others restrict the target instead: continuous measures have been derived for individual group families, notably $\mathrm { C } _ { \mathrm { n v } }$ and $\mathrm { D _ { n } } ^ { 4 5 }$ , or extended from point sets to vectors, matrices and operators<sup>46</sup>.

## From symmetry-induced constraints to spectral analysis

The constructions above compare coordinates with a symmetric realization or evaluate a spatial realization of the target operations. SymFP instead starts from the rank constraints that input symmetry imposes on equivariant representation blocks and uses the target’s fixed-subspace dimensions to read their singular-value spectra, without constructing a symmetric reference or selecting a target orientation.

Classical analysis identified the combinations of spherical harmonics fixed by individual point groups<sup>47</sup>. Cen et al.<sup>13</sup> and Lin et al.<sup>14</sup> studied how input symmetry can reduce the expressivity of equivariant representations through representation degeneracy and symmetry increase. Related studies have used equivariant networks to search for symmetry-breaking order parameters <sup>10</sup> and examined symmetry breaking through Curie-type constraints on equivariant maps <sup>11</sup>.

Operationally, we collect the fixed-subspace dimensions into rank templates and compare them with rank sequences obtained from the singular-value spectra of equivariant representation blocks. Rank-template matching uses the complete rank sequence for discrete point-group identification, while leakage scores provide an orientation-invariant, continuous measure of departures from targetimposed rank constraints.

## 4.3 Fixed-subspace derivation of rank templates

The rank templates follow from the dimensions of the fixed subspaces imposed by input symmetry on equivariant representations. Detailed proofs are provided in Supplementary Note A.

Theorem 3 (Fixed-subspace constraint). Suppose that $\tilde { V } ^ { ( l ) }$ is an O(3)-equivariant representation map satisfying Eq. (1). For any $\mathcal { G } \in \mathbb { G } ( \mathfrak { H } )$ , write $\mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) : = \left\{ \tilde { \pmb { v } } \in \mathbb { R } ^ { 2 \dot { l } + 1 } : \rho ^ { ( l ) } ( \mathfrak { h } ) \tilde { \pmb { v } } = \tilde { \pmb { v } } \ \forall \mathfrak { h } \in \mathfrak { H } \right\}$ for the fixed subspace and $\rho ^ { ( l ) } ( \mathfrak { H } ) : = 1 / | \mathfrak { H } | \cdot \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { h } )$ for the group average. Then

$$
\mathrm { c o l s p a n } ( \tilde { \mathbf { V } } ^ { ( l ) } ( \mathcal { G } ) ) \subseteq \mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) ,\tag{16}
$$

and consequently

$$
\operatorname { r a n k } ( { \tilde { V } } ^ { ( l ) } ( { \mathcal { G } } ) ) \leq m _ { l } ( { \mathfrak { H } } ) ,\tag{17}
$$

where $m _ { l } ( { \mathfrak { H } } )$ is the multiplicity of the trivial representation in $\rho ^ { ( l ) } | _ { \mathfrak { H } }$ , that is $m _ { l } ( \mathfrak { H } ) : = \dim \mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) =$ rank $( \rho ^ { ( l ) } ( \mathfrak { H } ) ) = \mathrm { t r } ( \rho ^ { ( l ) } ( \mathfrak { H } ) )$ .

The fixed-subspace dimension $m _ { l } ( { \mathfrak { H } } )$ can be computed from the character $\chi ^ { ( l ) } ( \mathfrak { h } ) : = \mathrm { t r } ( \rho ^ { ( l ) } ( \mathfrak { h } ) )$ of the degree-l representation:

$$
m _ { l } ( { \mathfrak { H } } ) = { \frac { 1 } { | { \mathfrak { H } } | } } \sum _ { { \mathfrak { h } } \in { \mathfrak { H } } } \chi ^ { ( l ) } ( { \mathfrak { h } } ) .\tag{18}
$$

Collecting these dimensions over a selected degree set $\Lambda \subset \mathbb { N } _ { 0 }$ gives the rank template

$$
\mathrm { R a n k T e m p l a t e } _ { \Lambda } ( \mathfrak { H } ) : = ( m _ { l } ( \mathfrak { H } ) ) _ { l \in \Lambda } .\tag{19}
$$

Under a change of coordinate frame, the fixed subspace transforms as

$$
\begin{array} { r } { \mathbb { F } ^ { ( l ) } ( { \mathfrak { g } } { \mathfrak { H } } { \mathfrak { g } } ^ { - 1 } ) = \rho ^ { ( l ) } ( { \mathfrak { g } } ) \mathbb { F } ^ { ( l ) } ( { \mathfrak { H } } ) , \qquad m _ { l } ( { \mathfrak { g } } { \mathfrak { H } } { \mathfrak { g } } ^ { - 1 } ) = m _ { l } ( { \mathfrak { H } } ) , } \end{array}\tag{20}
$$

for every finite point group ${ \mathfrak { H } } \leq \mathrm { O } ( 3 )$ , every ${ \mathfrak { g } } \in \mathrm { O } ( 3 )$ , and every degree l. Hence, the rank template is invariant under target reorientation. For the 32 crystallographic point groups, the number of distinct templates depends on the selected degrees.

Corollary 4 (Rank templates for the 32 crystallographic point groups). Let $\mathbb { C } _ { 3 2 }$ denote the 32 crystallographic point groups<sup>48</sup>. The map $\mathfrak { H } \mapsto \mathrm { R a n k T e m p l a t e } _ { \Lambda _ { 9 } } ( \mathfrak { H } )$ is injective on $\mathbb { C } _ { 3 2 }$ . The shorter degree set $\Lambda _ { 6 }$ gives 30 distinct templates, with only the pairs $\mathrm { D } _ { 6 } / \mathrm { D } _ { 6 \mathrm { h } }$ and $\mathrm { O } / \mathrm { O } _ { \mathrm { h } }$ sharing templates.

The corresponding multiplicities are listed in Supplementary Table A.1. Rank templates also satisfy the following ordering under subgroup inclusion.

Proposition 5 (Subgroup monotonicity). If $\mathfrak { H } _ { 1 } \le \mathfrak { H } _ { 2 } \le \mathrm { O } ( 3 )$ , then $\mathbb { F } ^ { ( l ) } ( \mathfrak { H } _ { 2 } ) \subseteq \mathbb { F } ^ { ( l ) } ( \mathfrak { H } _ { 1 } )$ and $m _ { l } ( { \mathfrak { H } } _ { 2 } ) \leq m _ { l } ( { \mathfrak { H } } _ { 1 } )$ for every l. Consequently, RankTemplate $_ { \Lambda } ( \mathfrak { H } _ { 2 } ) \leq$ RankTemplate (H<sub>1</sub>) componentwise for every finite degree set Λ.

## 4.4 Spectral formulation and numerical evaluation

## Spectral symmetry fingerprint

For a finite set of degrees Λ, consider the representation blocks $\tilde { \mathbf { V } } ^ { ( l ) } ( \mathcal { G } ) \in \mathbb { R } ^ { ( 2 l + 1 ) \times C }$ , each with $d _ { l } : = \operatorname* { m i n } \{ 2 l + 1 , C \}$ singular values. Under the default channel count, $d _ { l } = 2 l + 1$ . For each $l \in \Lambda$ 2 let

$$
\sigma _ { 1 } ^ { ( l ) } \ge \sigma _ { 2 } ^ { ( l ) } \ge \cdots \ge \sigma _ { d _ { l } } ^ { ( l ) } \ge 0\tag{21}
$$

be the ordered singular values of $\tilde { V } ^ { ( l ) }$ , with the graph argument suppressed when unambiguous and $\sigma _ { i } ^ { ( l ) } = 0$ for $i > d _ { l }$ . Write $\pmb { \sigma } ^ { ( l ) } : = ( \sigma _ { i } ^ { ( l ) } ) _ { i = 1 } ^ { d _ { l } }$ . The spectral symmetry fingerprint concatenates these spectra across degrees:

$$
\mathrm { S y m F P } _ { \Lambda } ( \mathcal { G } ) : = ( \sigma _ { i } ^ { ( l ) } ) _ { l \in \Lambda , 1 \leq i \leq d _ { l } } .\tag{22}
$$

SymFP is invariant under global reorientation (Prop. 7). For $\mathcal { G } \in \mathbb { G } ( \mathfrak { H } )$ , the fixed-subspace constraint (Thm. 3) imposes

$$
\mathrm { r a n k } ( \tilde { \pmb { V } } ^ { ( l ) } ) \leq m _ { l } ( \mathfrak { H } ) , \qquad \sigma _ { i } ^ { ( l ) } = 0 \quad \mathrm { f o r } m _ { l } ( \mathfrak { H } ) < i \leq d _ { l } .\tag{23}
$$

## Rank-template matching

We estimate numerical ranks using a graph-dependent threshold that combines absolute and relative tolerances:

$$
\tau ( \mathcal { G } ) : = \operatorname* { m a x } \{ \tau _ { \mathrm { a b s } } , \ \tau _ { \mathrm { r e l } } \sigma _ { 1 } ^ { ( 0 ) } ( \mathcal { G } ) \} ,\tag{24}
$$

where $\tau _ { \mathrm { a b s } }$ and $\tau _ { \mathrm { r e l } }$ are the respective tolerances, with the relative term scaled by the leading degree-zero singular value. The numerical rank at degree l is

$$
\widehat { m } _ { l , \tau } ( \mathcal { G } ) : = | \{ i \in \{ 1 , \dots , d _ { l } \} : \sigma _ { i } ^ { ( l ) } > \tau \} | .\tag{25}
$$

These ranks form the numerical rank sequence

$$
\widehat { \mathrm { R a n k S e q } _ { \Lambda , \tau } } ( \mathcal G ) : = ( \widehat { m } _ { l , \tau } ) _ { l \in \Lambda } .\tag{26}
$$

Given a candidate family C, classification minimizes the weighted $\ell _ { 1 }$ distance to the candidate rank templates:

$$
\widehat { \mathfrak { H } } _ { \Lambda , \tau } ^ { \mathrm { h a r d } } ( \mathcal { G } ) \in \arg \operatorname* { m i n } _ { \mathfrak { H } \in \mathbb { C } } \sum _ { l \in \Lambda } \alpha _ { l } \vert \widehat { m } _ { l , \tau } - m _ { l } ( \mathfrak { H } ) \vert ,\tag{27}
$$

with uniform weights $\alpha _ { l } = 1$ . The set of minimizers is invariant under global reorientation, since both the singular values and rank templates are orientation-invariant.

## Leakage scores

The degree-resolved leakage score in Eq. (4) is

$$
\mathrm { L S } _ { \mathfrak { H } } ^ { ( l ) } ( \mathcal { G } ) = \frac { 1 } { Z _ { l } ( \mathcal { G } ) } \sum _ { i > m _ { l } ( \mathfrak { s } ) } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } , \qquad Z _ { l } ( \mathcal { G } ) = \sum _ { i } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } + \varepsilon _ { \mathrm { n o r m } } .\tag{28}
$$

Pooling uniformly over degrees $l \in \Lambda$ gives

$$
\mathrm { L S } _ { \Lambda , \mathfrak { H } } ( \mathcal { G } ) = \frac { 1 } { | \Lambda | } \sum _ { l \in \Lambda } \mathrm { L S } _ { \mathfrak { H } } ^ { ( l ) } ( \mathcal { G } ) ,\tag{29}
$$

with the logarithmic score

$$
\mathrm { p L S } _ { \Lambda , \mathfrak { H } } ( \mathcal { G } ) = - \log _ { 1 0 } ( \mathrm { L S } _ { \Lambda , \mathfrak { H } } ( \mathcal { G } ) + \varepsilon _ { \log } ) .\tag{30}
$$

We set $\varepsilon _ { \mathrm { l o g } } = 1 0 ^ { - 1 0 }$ , yielding a maximum pLS of 10. A pLS of 3 corresponds to a mean spectral leakage below 0.1%, indicating near-satisfaction of the target rank constraints. However, a high pLS does not necessarily imply the presence of the target symmetry. For example, over degrees $l = 0 , \ldots , 9$ , structures with point groups $\mathrm { C _ { 2 } }$ and $\mathrm { T _ { h } }$ exhibit zero LS relative to $\mathrm { C _ { s } }$ and $\mathrm { D } _ { \mathrm { 6 h } }$ respectively, despite lacking these symmetries. This occurs because the ranks of their equivariant representations remain within the corresponding target fixed-subspace dimensions (Supplementary Table A.1). Discrete point-group identification based on rank-template matching is therefore more appropriate for this purpose.

More generally, pooling admits nonnegative weights $\beta _ { l }$ summing to one; all evaluations use uniform weights $\beta _ { l } = 1 / | \Lambda |$

## Normalization

When $m _ { l } ( { \mathfrak { H } } ) = 0$ , all block energy lies in the spectral tail, causing degree-wise normalized LS to approach unity when the block energy substantially exceeds $\varepsilon _ { \mathrm { n o r m } }$ . To avoid this saturation for centrosymmetric targets, the torsion analysis pools only even degrees. Alternative normalizations,

$$
Z _ { l } ( \mathcal { G } ) = 1 , \qquad \mathrm { o r } \qquad Z _ { l } ( \mathcal { G } ) = \sum _ { k \in \Lambda } \sum _ { i } ( \sigma _ { i } ^ { ( k ) } ) ^ { 2 } + \varepsilon _ { \mathrm { n o r m } } ,\tag{31}
$$

retain absolute block energies or relative cross-degree energies, respectively. Engel score similarly employs global energy normalization, made explicit by

$$
\mathrm { E s } _ { 5 } ( \mathcal { G } ) = 1 - \frac { 1 } { 1 - \beta _ { 5 } } \operatorname* { m i n } _ { \mathfrak { r } \in \mathrm { S O } ( 3 ) } \frac { \sum _ { l } \| ( \pmb { I } - \pmb { P } _ { 5 } ^ { ( l ) } ( \mathfrak { r } ) ) \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } } { \sum _ { l } \| \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } } .\tag{32}
$$

The subgroup ordering of fixed-subspace dimensions induces the following monotonicity of leakage scores under each of the normalization choices above.

Proposition 6 (Leakage monotonicity along subgroup chains). $I f \mathfrak { H } _ { 1 } \leq \mathfrak { H } _ { 2 } \leq \mathrm { O } ( 3 )$ , then for every geometric graph $\mathcal { G } _ { i }$

$$
\begin{array} { r } { \mathrm { L S } _ { \Lambda , \mathfrak { H } _ { 1 } } ( \mathcal { G } ) \leq \mathrm { L S } _ { \Lambda , \mathfrak { H } _ { 2 } } ( \mathcal { G } ) , \qquad \mathrm { p L S } _ { \Lambda , \mathfrak { H } _ { 1 } } ( \mathcal { G } ) \geq \mathrm { p L S } _ { \Lambda , \mathfrak { H } _ { 2 } } ( \mathcal { G } ) . } \end{array}\tag{33}
$$

This ordering need not hold when the pooled degrees depend on the target. Task-specific choices and limiting cases are given in Supplementary Notes A.2 and C.1.

## 4.5 Construction of the parameter-free equivariant GNN

In Algorithm 1, $\scriptstyle { \vec { \pmb { x } } } _ { c }$ denotes the graph centroid, with $\vec { \pmb { x } } _ { i c } : = \vec { \pmb { x } } _ { i } - \vec { \pmb { x } } _ { c }$ and $\vec { \pmb { x } } _ { i j } : = \vec { \pmb { x } } _ { j } - \vec { \pmb { x } } _ { i }$ representing centred node vectors and edge displacement vectors, respectively. Their unit directions are denoted by $\hat { \mathbf { x } } _ { i c }$ and $\hat { \mathbf { \mathscr { x } } } _ { i j }$ wherever defined. The radial scales $Z _ { \nu }$ and $Z _ { \mathcal { E } }$ are the mean centred-node radius and mean edge length, respectively, with a fixed $\varepsilon _ { 0 } > 0$ used to avoid zero denominators. The functions $\{ \rho _ { n } \}$ and $\{ \eta _ { m } \}$ are fixed Gaussian radial bases. For zero vectors, we set $Y ^ { ( l ) } ( { \mathbf 0 } ) = { \mathbf 0 } _ { 2 l + 1 }$ for $l > 0$ , while retaining the degree-zero scalar. If node attributes are absent, we use $d _ { v } = 1$ and $\begin{array} { r } { \pmb { h } _ { i } \equiv 1 ; } \end{array}$ if edge attributes are absent, we set $d _ { e } = 0$ . In all reported PF evaluations, node attributes $\mathbf { } _ { h _ { i } }$ are scalar atomic numbers, and edge attributes $e _ { i j } = \| \pmb { \vec { x } } _ { j } - \pmb { \vec { x } } _ { i } \|$ are edge lengths in the input coordinate units. Both attributes are scalar, with $d _ { v } = d _ { e } = 1$ , and neither is rescaled. Implementation constants are listed in Supplementary Note C.1.

In the edge block, keeping $\mathbf { } _ { h _ { i } }$ and $h _ { j }$ in separate channels can retain odd-degree information that would otherwise cancel when opposite edge directions are pooled with symmetric scalar weights<sup>5</sup>. For paired reverse edges, this cancellation is no longer enforced when the endpoint attributes difer.

For each degree l, the three PF blocks satisfy the equivariance relation in Eq. (1):

$$
\begin{array} { r } { \tilde { V } _ { \gamma } ^ { ( l ) } ( \mathfrak { g } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { g } ) \tilde { V } _ { \gamma } ^ { ( l ) } ( \mathcal { G } ) , \quad \tilde { V } _ { \mathcal { E } } ^ { ( l ) } ( \mathfrak { g } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { g } ) \tilde { V } _ { \mathcal { E } } ^ { ( l ) } ( \mathcal { G } ) , \quad \tilde { V } ^ { ( l ) } ( \mathfrak { g } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { g } ) \tilde { V } ^ { ( l ) } ( \mathcal { G } ) . } \end{array}\tag{34}
$$

Equivariance follows because centring commutes with orthogonal transformations, radial weights depend only on norms, and spherical harmonics transform by $\rho ^ { ( l ) }$ . Every stabilizer operation permutes the nodes and therefore preserves their centroid, so centring removes global translations without changing the symmetry constraints.

## 4.6 Theoretical properties and variational interpretation

## Generic rank realization and stability

For a fixed input and degree, an equivariant architecture attains its maximal attainable rank at almost every parameter setting, provided its representation block depends real-analytically on the parameters over a connected open domain (Prop. 10). In particular, full span at a single parameter setting guarantees full span almost everywhere, with the columns spanning the fixed subspace and the rank equaling its dimension (Cor. 11). If full span holds across the selected degrees, distinct rank templates identify the input’s point group within the candidate family, provided the numerical threshold is positive and smaller than every positive singular value. We assess rank attainment across learnable GNN architectures and training regimes, alongside PF, in the controlled benchmark (Supplementary Table C.2).

For graph families with fixed attributes and adjacency, the spectral descriptor inherits local Lipschitz continuity from the representation blocks (Thm. 13). Comparison with the target-symmetric subset then bounds spectral departure in terms of distance to that subset, linking the descriptor to a point-group-conditioned continuous symmetry measure (Eq. (65) and Cor. 15).

## Variational relation to operation-based symmetry measures

The spectral retained energy admits a variational characterization that connects SymFP to the retained-energy form of operation-based symmetry analysis introduced in § 4.2. Write $C ^ { ( l ) } : =$ $\tilde { V } ^ { ( l ) } ( \tilde { V } ^ { ( l ) } ) ^ { \top }$ and $k : = m _ { l } ( \mathfrak { H } )$ . SymFP retains the energy $\begin{array} { r } { \mathcal { A } _ { \mathfrak { H } } ^ { ( l ) , \mathrm { s y m f p } } : = \sum _ { i = 1 } ^ { k } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } } \end{array}$ , whose variational form follows from the Ky Fan maximum principle<sup>49</sup>:

$$
\mathcal { A } _ { \mathfrak { H } } ^ { ( l ) , \mathrm { s y m f p } } = \operatorname* { m a x } _ { P ^ { 2 } = P = P ^ { \top } \atop \operatorname { r a n k } ( P ) = k } \mathrm { t r } ( P C ^ { ( l ) } ) .\tag{35}
$$

For the same multi-channel block, a spatial realization of the target group determines a fixedsubspace projector. The retained energy is the trace of this projector against $C ^ { ( l ) }$ . For a single harmonic channel $\tilde { \mathbf { V } } ^ { ( l ) } = \tilde { \mathbf { v } } ^ { ( l ) }$ , this is the retained-energy term in Engel score at the specified orientation. To relate this projection-based retained energy to SymFP’s spectral variational form, we optimize the retained energy at each degree separately:

$$
\mathcal { A } _ { 5 } ^ { ( l ) , \mathrm { o p } } : = \operatorname* { m a x } _ { \mathfrak { r } \in \mathrm { S O } ( 3 ) } \mathrm { t r } \Big ( P _ { 5 } ^ { ( l ) } ( \mathfrak { r } ) C ^ { ( l ) } \Big ) ,\tag{36}
$$

where $P _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } )$ projects onto the fixed subspace of the spatially realized copy $\mathfrak { r } \mathfrak { H } \mathfrak { r } ^ { - 1 }$ . The two formulations share the variational objective $\mathrm { t r } ( P C ^ { ( l ) } )$ but difer in their feasible projector sets. SymFP ranges over all rank-k orthogonal projectors, while the operation-based formulation is restricted to projectors onto $\rho ^ { ( l ) } ( \mathfrak { r } ) \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ with $\mathfrak { r } \in \mathrm { S O } ( 3 )$ . Since every such projector has rank k, we obtain

$$
\mathcal { A } _ { \mathfrak { H } } ^ { ( l ) , \mathrm { o p } } \le \mathcal { A } _ { \mathfrak { H } } ^ { ( l ) , \mathrm { s y m f p } } .\tag{37}
$$

Equality holds in particular for exact-H-symmetric inputs, for which both optima equal $\| \tilde { V } ^ { ( l ) } \| _ { F } ^ { 2 }$ For distorted inputs, optimizing the target orientation selects a reference under the chosen projection criterion. The two formulations ofer distinct perspectives: agreement with target operations and target rank constraints. Once the representation blocks are computed, the same singularvalue spectra give the SymFP retained energies and leakage scores for every target through its fixed-subspace dimensions. The retained-energy identities and general equality conditions are de rived in Supplementary Note B.2.

## 4.7 Datasets and evaluation

## Synthetic point clouds

For each of the 32 crystallographic point groups, we generate 100 finite point clouds by combining point-group orbits and independently verify their stabilizers. Each group is split into training, validation, and test sets in an 80/10/10 ratio. Rank tolerances are selected on the validation set and held fixed across GNNs (Supplementary Note C.2).

## Controlled deformations

We induce single-step symmetry breaking by straining a unit octahedron or twisting an eclipsed triangular prism, and progressive symmetry breaking by applying successive linear maps to the octahedron along the subgroup chain in Fig. 3b. To compare deformation modes, we use a 30- point $\mathrm { C _ { 6 } }$ ring and normalize displacement fields to matched RMSD after removing rigid-body components.

We independently verify the stabilizer of each frame (Supplementary Note C.3) and identify spectral onset using degree-resolved relative tail energy, excluding round-of blocks (Supplementary Note D.1). For the HEGNN node-type control, we assign distinct types to the five node orbits, preserving the undeformed symmetry, and hold these assignments fixed across deformations (Sup plementary Note D.1).

## Biaryl torsion scans

We use deposited geometries and energies from OpenFF Rowley Biaryl $\mathrm { v } 1 . 0 ^ { 2 4 , 2 5 }$ , comprising 87 biaryls sampled at 15<sup>◦</sup> intervals. All atoms are retained in fully connected graphs. Nontrivial point groups identified by pymatgen<sup>50</sup> along each scan serve as target symmetries. Excluding molecule–target pairs that preserve the target symmetry throughout the scan yields 94 pairs from 86 molecules (Supplementary Note C.4).

For centrosymmetric targets, fixed subspaces vanish at odd natural-parity degrees, causing odd-degree LS to saturate near unity when block energies substantially exceed the regularization scale. We therefore pool degrees {0, 2, 4, 6} across all targets. Baseline rankings use Engel score or the negated CSoM deviation, with target orientations optimized for both methods as described in Supplementary Note C.4 and Supplementary Note C.4.2. AUCs are averaged first across targets within each molecule and then equally across molecules. For target-symmetry comparisons, extrema are classified by whether they possess the target symmetry, while all conformations remain in each ranking (Supplementary Note C.4.1).

Pretraining uses a fixed split of 69 training and 17 validation molecules. Total-energy pretraining minimizes mean absolute error on energies standardized using training-set statistics, while Uni-Mol-style pretraining <sup>21</sup> combines masked-atom prediction, coordinate recovery, and pair-distance prediction. We compare each pretrained model against its paired initialization and report pLS gap improvements separately for the full cohort and validation molecules (Supplementary Note C.4.3).

## Lanthanide DOTA complexes

We use seven water-free [Ln(DOTA)]<sup>–</sup> crystal structures (Y and Tb through Yb)<sup>29</sup>, together with water-capped Dy(DOTA) and distorted Tm(DOTA) structures from the CSoM package<sup>4</sup>. Metal-centred clusters are reconstructed excluding counterions and lattice solvent. Cross-family comparisons use heavy atoms, whereas illustrative spectral analyses retain hydrogens. For the capping-water trajectory, only the two water protons are rotated about the Dy–O axis, with the coordination cage held fixed. We evaluate degree-resolved LS for crystal-field analysis and pLS pooled over degrees 0, . . . , 6 for water rotation (Supplementary Note C.5).

## Biological assemblies

The p97, HA, and GluK2 analyses use backbone atoms with pLS pooled over degrees 0, . . . , 6.   
Each assembly or region is independently centred and represented by a nearest-neighbour graph.   
E18 uses sequence-matched Cα atoms and degree-six pLS.

For p97, we retain the six p97 chains from UBXD1-associated reconstructions<sup>30</sup>, match complete backbone atom identities across chains and states, and evaluate assemblies, domains, and corresponding sliding windows against $\mathrm { C _ { 6 } }$ (Supplementary Notes C.6.1 and C.6.2).

For HA<sup>32</sup>, shared residue windows define HA1, HA2, and their union, each evaluated against $\mathrm { C _ { 3 } }$ (Supplementary Note C.6.3). GluK2<sup>31</sup> is evaluated against $\mathrm { C _ { 4 } }$ by domain and by non-overlapping layers along the axis connecting the transmembrane and amino-terminal Cα centroids (Supplementary Note C.6.4). For E18 <sup>33</sup>, we retain all capsid-protein copies in each deposited assembly and evaluate the assembly against the proper icosahedral group I (Supplementary Note C.6.5).

## Computational cost

Timings include graph construction, representation computation, and spectral scoring, as well as region selection for the p97 window scan and orientation search for fitted baselines. For multiple target evaluations on Dy-DOTA-water, target-independent computations are reused within each method’s call (Supplementary Note C.7).

## Data availability

The experimental structures and energy data are obtained from the public sources listed below. The accompanying repository provides the synthetic point clouds and dataset splits, the molecular and biological coordinate selections, and numerical source data for the reported analyses.

Three families of synthetic point clouds appear in § 2.2 and 2.3: the 32-group benchmark, the controlled symmetry-breaking families and the matched-amplitude displacement fields. These datasets are generated deterministically using the constructions specified in Methods and Supple mentary Notes C.2 and C.3.

The torsion study of § 2.4 uses the relaxed scans of OpenFF Rowley Biaryl v1.0, obtained through QCArchive <sup>24</sup>, together with their deposited energies.

The study of lanthanide DOTA complexes in § 2.5 uses the water-free Ln(DOTA) series deposited as CCDC 2281329 and 2281615–2281620; it also uses the two approximate analogues and the capping-water trajectory, as redistributed with the CSoM package <sup>4</sup>.

The assemblies are Protein Data Bank entries. The p97 regional analysis in § 2.6 uses 8FCL, 8FCM, 8FCR, 8FCQ, 8FCO and 8FCT. For haemagglutinin in Supplementary Note D.3, the series is 6Y5G, 6Y5H, 6Y5I, 6Y5J and 6Y5K. The GluK2 domain and spatial-layer analyses use 5KUF; the echovirus 18 capsid comparison uses 6HBH, 6HBK, 6HHT and 6HBL. Atom selections, complete spectra and input-setting results accompany the analysis code. Computational benchmark records include the timing protocol, hardware settings, per-run measurements and input hashes.

## Code availability

The SymFP implementation, training code and evaluation scripts are released at https:// github.com/GLAD-RUC/SymFP. The point-group analysis and space-group expansion use pymatgen<sup>50</sup>; implementation details for the equivariant GNNs are provided in the accompanying repository. HEGNN uses the multi-channel extension developed here, which is included in the repository. CSoM is evaluated using the software released by its authors<sup>4</sup>; Engel score is our implementation of the published construction<sup>5,15</sup>

## Author contributions

J.C. proposed and developed the theoretical and algorithmic framework, designed and analysed the experiments, and drafted the manuscript under the supervision of W.H.; A.L. and N.L. contributed throughout these stages; Z.W. and J.-R.W. provided technical support; all authors reviewed and approved the final version.

## Competing interests

The authors declare no competing interests.

## References

1. Hagit Zabrodsky, Shmuel Peleg, and David Avnir. Continuous symmetry measures. Journal of the American Chemical Society, 114(20):7843–7851, 1992.

2. Hagit Zabrodsky, Shmuel Peleg, and David Avnir. Continuous symmetry measures. 2. symmetry groups and the tetrahedron. Journal of the American Chemical Society, 115(18):8278–8289, 1993.

3. Villads R. M. Nielsen, Boris Le Guennic, and Thomas Just Sørensen. Evaluation of point group symmetry in lanthanide(III) complexes: a new implementation of a continuous symmetry

operation measure with autonomous assignment of the principal axis. The Journal of Physical Chemistry A, 128:5740–5751, 2024.

4. Villads R. M. Nielsen and Thomas Just Sørensen. Determining molecular structure, coordi nation geometry, and molecular symmetry using a continuous symmetry operation measure software. Nature Communications, 16:11122, 2025.

5. Michael Engel. Point group analysis in particle simulation data. arXiv preprint arXiv:2106.14846, 2021.

6. Michael M Bronstein, Joan Bruna, Taco Cohen, and Petar Veliˇckovi´c. Geometric deep learning: grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

7. John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Z´ıdek, Anna Potapenko, Alex<sup>ˇ</sup> Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with AlphaFold. Nature, 596(7873):583–589, 2021.

8. Minkyung Baek, Frank DiMaio, Ivan Anishchenko, Justas Dauparas, Sergey Ovchinnikov, Gyu Rie Lee, Jue Wang, Qian Cong, Lisa N. Kinch, R. Dustin Schaefer, Claudia Mill´an, Hahnbeom Park, Carson Adams, Caleb R. Glassman, Andy DeGiovanni, Jose H. Pereira, Andria V. Rodrigues, Alberdina A. van Dijk, Ana C. Ebrecht, Diederik J. Opperman, Theo Sagmeister, Christoph Buhlheller, Tea Pavkov-Keller, Manoj K. Rathinaswamy, Udit Dalwadi, Calvin K. Yip, John E. Burke, K. Christopher Garcia, Nick V. Grishin, Paul D. Adams, Randy J. Read, and David Baker. Accurate prediction of protein structures and interactions using a three-track neural network. Science, 373(6557):871–876, 2021.

9. Claudio Zeni, Robert Pinsler, Daniel Z¨ugner, Andrew Fowler, Matthew Horton, Xiang Fu, Zilong Wang, Aliaksandra Shysheya, Jonathan Crabb´e, Shoko Ueda, Roberto Sordillo, Lixin Sun, Jake Smith, Bichlien Nguyen, Hannes Schulz, Sarah Lewis, Chin-Wei Huang, Ziheng Lu, Yichi Zhou, Han Yang, Hongxia Hao, Jielan Li, Chunlei Yang, Wenjie Li, Ryota Tomioka, and Tian Xie. A generative model for inorganic materials design. Nature, 639(8055):624–632, 2025.

10. Tess E Smidt, Mario Geiger, and Benjamin Kurt Miller. Finding symmetry breaking order parameters with Euclidean neural networks. Physical Review Research, 3(1):L012002, 2021.

11. S´ekou-Oumar Kaba and Siamak Ravanbakhsh. Symmetry breaking and equivariant neural networks. In NeurIPS 2023 Workshop on Symmetry and Geometry in Neural Representations, 2023.

12. Chaitanya K Joshi, Cristian Bodnar, Simon V Mathis, Taco Cohen, and Pietro Lio. On the expressive power of geometric graph neural networks. In International Conference on Machine Learning, pages 15330–15355. PMLR, 2023.

13. Jiacheng Cen, Anyi Li, Ning Lin, Yuxiang Ren, Zihe Wang, and Wenbing Huang. Are highdegree representations really unnecessary in equivariant graph neural networks? Advances in Neural Information Processing Systems, 37:26238–26266, 2024.

14. Ning Lin, Jiacheng Cen, Anyi Li, Wenbing Huang, and Hao Sun. Reducing symmetry increase in equivariant neural networks. In The Fourteenth International Conference on Learning Representations, 2026.

15. Chrameh Fru Mbah, Junwei Wang, Silvan Englisch, Praveen Bommineni, Nydia Roxana Varela-Rosales, Erdmann Spiecker, Nicolas Vogel, and Michael Engel. Early-stage bifurcation of crystallization in a sphere. Nature Communications, 14:5299, 2023.

16. Jiaqi Han, Jiacheng Cen, Liming Wu, Zongzhao Li, Xiangzhe Kong, Rui Jiao, Ziyang Yu, Tingyang Xu, Fandi Wu, Zihe Wang, Hongteng Xu, Zhewei Wei, Deli Zhao, Yang Liu, Yu Rong, and Wenbing Huang. A survey of geometric graph neural networks: data structures, models and applications. Frontiers of Computer Science, 19(11):1911375, 2025.

17. Mario Geiger and Tess Smidt. e3nn: Euclidean neural networks. arXiv preprint arXiv:2207.09453, 2022.

18. Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhof, and Patrick Riley. Tensor field networks: rotation- and translation-equivariant neural networks for 3D point clouds. arXiv preprint arXiv:1802.08219, 2018.

19. Ilyes Batatia, David P Kovacs, Gregor Simm, Christoph Ortner, and G´abor Cs´anyi. MACE: higher order equivariant message passing neural networks for fast and accurate force fields. Advances in Neural Information Processing Systems, 35:11423–11436, 2022.

20. Yi-Lun Liao and Tess Smidt. Equiformer: equivariant graph attention transformer for 3D atomistic graphs. In The Eleventh International Conference on Learning Representations, 2023.

21. Gengmo Zhou, Zhifeng Gao, Qiankun Ding, Hang Zheng, Hongteng Xu, Zhewei Wei, Linfeng Zhang, and Guolin Ke. Uni-Mol: a universal 3D molecular representation learning framework. In The Eleventh International Conference on Learning Representations, 2023.

22. Vojislava Pophristic and Lionel Goodman. Hyperconjugation not steric repulsion leads to the staggered structure of ethane. Nature, 411(6837):565–568, 2001.

23. Paul L. A. Popelier, Peter I. Maxwell, Joseph C. R. Thacker, and Ibon Alkorta. A relative energy gradient (REG) study of the planar and perpendicular torsional energy barriers in biphenyl. Theoretical Chemistry Accounts, 138(1):12, 2019.

24. Daniel G. A. Smith, Doaa Altarawy, Lori A. Burns, Matthew Welborn, Levi N. Naden, Logan Ward, Sam Ellis, Benjamin P. Pritchard, and T. Daniel Crawford. The MolSSI QCArchive project: an open-source platform to compute, organize, and share quantum chemistry data. WIREs Computational Molecular Science, 11(2):e1491, 2021.

25. Yudong Qiu, Daniel G. A. Smith, Chaya D. Stern, Mudong Feng, Hyesu Jang, and Lee-Ping Wang. Driving torsion scans with wavefront propagation. The Journal of Chemical Physics, 152(24):244116, 2020.

26. Christiane G¨orller-Walrand and Koen Binnemans. Rationalization of crystal-field parametrization. In Handbook on the Physics and Chemistry of Rare Earths, volume 23, pages 121–283. Elsevier, 1996.

27. Giuseppe Cucinotta, Mauro Perfetti, Javier Luzon, Mael Etienne, Pierre-Emmanuel Car, Andrea Caneschi, Guillaume Calvez, Kevin Bernot, and Roberta Sessoli. Magnetic anisotropy in a dysprosium/DOTA single-molecule magnet: beyond simple magneto-structural correlations. Angewandte Chemie International Edition, 51(7):1606–1610, 2012.

28. Matteo Briganti, Guglielmo Fernandez Garcia, Julie Jung, Roberta Sessoli, Boris Le Guennic, and Federico Totti. Covalency and magnetic anisotropy in lanthanide single molecule magnets: the DyDOTA archetype. Chemical Science, 10:7233–7245, 2019.

29. Anna Schannong Manvell, Rouven Pfleger, Niels Andreas Bonde, Matteo Briganti, Carlo Andrea Mattei, Theis Brock-Nannestad, Høgni Weihe, Annie K. Powell, Jacques Ollivier, Jesper Bendix, and Mauro Perfetti. LnDOTA puppeteering: removing the water molecule and imposing tetragonal symmetry. Chemical Science, 15:113–123, 2024.

30. Julian R. Braxton, Chad R. Altobelli, Maxwell R. Tucker, Eric Tse, Aye C. Thwin, Michelle R. Arkin, and Daniel R. Southworth. The p97/VCP adaptor UBXD1 drives AAA+ remodeling and ring opening through multi-domain tethered interactions. Nature Structural & Molecular Biology, 30:2009–2019, 2023.

31. Joel R. Meyerson, Sagar Chittori, Alan Merk, Prashant Rao, Tae Hee Han, Mihaela Serpe, Mark L. Mayer, and Sriram Subramaniam. Structural basis of kainate subtype glutamate receptor desensitization. Nature, 537:567–571, 2016.

32. Donald J. Benton, Steven J. Gamblin, Peter B. Rosenthal, and John J. Skehel. Structural transitions in influenza haemagglutinin at membrane fusion pH. Nature, 583:150–153, 2020.

33. David Buchta, Tibor F¨uzik, Dominik Hreb´ık, Yevgen Levdansky, Luk´aˇs Suken´ık, Liya Mukhamedova, Jana Moravcov´a, Robert V´acha, and Pavel Plevka. Enterovirus particles expel capsid pentamers to enable genome release. Nature Communications, 10:1138, 2019.

34. Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco S Cohen. 3D steerable CNNs: learning rotationally equivariant features in volumetric data. Advances in Neural Information Processing Systems, 31, 2018.

35. Mark Pinsky, Chaim Dryzun, David Casanova, Pere Alemany, and David Avnir. Analytical methods for calculating continuous symmetry measures and the chirality measure. Journal of Computational Chemistry, 29(16):2712–2721, 2008.

36. Gil Alon and Inbal Tuvi-Arad. Improved algorithms for symmetry analysis: structure preserving permutations. Journal of Mathematical Chemistry, 56(1):193–212, 2018.

37. Inbal Tuvi-Arad and Gil Alon. Improved algorithms for quantifying the near symmetry of proteins: complete side chains analysis. Journal of Cheminformatics, 11(1):39, 2019.

38. Gil Alon, Yuval Ben-Haim, and Inbal Tuvi-Arad. Continuous symmetry and chirality measures: approximate algorithms for large molecular structures. Journal of Cheminformatics, 15(1):106, 2023.

39. Inbal Tuvi-Arad, Yafa Shalit, and Gil Alon. CSM software: continuous symmetry and chirality measures for quantitative structural analysis. Journal of Chemical Information and Modeling, 64(14):5375–5380, 2024.

40. Miquel Llunell, David Casanova, Jordi Cirera, Pere Alemany, and Santiago Alvarez. SHAPE, version 2.1. Universitat de Barcelona, Barcelona, Spain, 2013.

41. M. J. Linehan and G. E. Stedman. Little groups of irreps of O(3), SO(3), and the infinite axial subgroups. Journal of Physics A: Mathematical and General, 34(34):6663–6688, 2001.

42. Tullio Pilati and Alessandra Forni. SYMMOL: a program to find the maximum symmetry group in an atom cluster, given a prefixed tolerance. Journal of Applied Crystallography, 31 (3):503–504, 1998.

43. R. Jefrey Largent, William F. Polik, and J. R. Schmidt. Symmetrizer: algorithmic determination of point groups in nearly symmetric molecules. Journal of Computational Chemistry, 33(19):1637–1642, 2012.

44. Peter J. Knowles. The determination of point groups from imprecise molecular geometries. Journal of Mathematical Chemistry, 60(1):161–171, 2022.

45. Mark Pinsky, Amir Zait, Maayan Bonjack, and David Avnir. Continuous symmetry analyses: C(nv) and D(n) measures of molecules, complexes, and proteins. Journal of Computational Chemistry, 34(1):2–9, 2013.

46. Chaim Dryzun and David Avnir. Generalization of the continuous symmetry measure: the symmetry of vectors, matrices, operators and functions. Physical Chemistry Chemical Physics, 11(42):9653–9666, 2009.

47. Burnett Meyer. On the symmetries of spherical harmonics. Canadian Journal of Mathematics, 6:135–157, 1954.

48. Mois I. Aroyo, editor. International Tables for Crystallography: space-group symmetry, volume A. Wiley, 6 edition, 2016. ISBN 978-0-470-97423-0.

49. Ky Fan. On a theorem of Weyl concerning eigenvalues of linear transformations I. Proceedings of the National Academy of Sciences, 35(11):652–655, 1949.

50. Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geofroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent L. Chevrier, Kristin A. Persson, and Gerbrand Ceder. Python Materials Genomics (pymatgen): a robust, open-source Python library for materials analysis. Computational Materials Science, 68:314–319, 2013.

a  
b  
![](images/7ba7abbf4f6936c63b5d765fbe0b83d29a0c87f2356368f1352e735e5d26ba0d.jpg)

Extended Data Fig. 1 | Spectral responses of PF and four learnable equivariant GNNs at random initialization. a, Progressive symmetry breaking along a subgroup chain. Curves show the spectral tail energy relative to each target group in $\mathrm { O _ { h } } \supset \mathrm { D _ { 4 h } } \supset \mathrm { D _ { 2 h } } \supset \mathrm { C _ { 2 h } } \supset \mathrm { C _ { i } }$ , normalized by the corresponding undeformed tail energy. Grey vertical lines indicate the boundaries between deformation segments. b, Deformation-mode comparison. Curves show the relative tail energy at $l = | m |$ as a function of deformation amplitude for four displacement fields with matched RMSD, where |m| denotes the lowest non-zero azimuthal order of each field. Blocks classified as round-of by Eq. (82) are shown as grey dashed traces with open markers. For HEGNN, the axial-pucker panel additionally shows the first energy-supported response at l = 4. The horizontal dashed line indicates the onset threshold in relative tail energy, whereas the round-of criterion is based on absolute block energy. Onset degrees are summarized in Supplementary Table D.9.

a  
![](images/d025cf6b7f115da14b80a26afb3f1db829681354dfbdb4d27ea00ef9be619003.jpg)

b  
![](images/61a950e49257f921d023060b9ebad14411d7720627f98c53be5646ebde637299.jpg)

![](images/f81ab5394c5349bc04154860019f7934e1731a7746b9ebc49856fac016978726.jpg)

c  
![](images/7dc809cdc0a648f730c83e6304f1ddc4504b3e99750dbc3ef998971572eb4cd0.jpg)

d  
![](images/382439f4df7125c5f9983c5dd84c0a1aab6bf057ab55a44f038932fb44eb8a9c.jpg)

![](images/0205601afa5a691434e83a63c150a3462113a7efa7c415ac796dc416673427ea.jpg)  
Extended Data Fig. 2 | Graph-neighbourhood and window-scale checks for p97. All scores use the hybrid PF, degrees $0 , \ldots , 6$ and target C . Diferences are relative to closed 8FCL. a, Local backbone around the Cα atom of chain A, residue 269, in 8FCL D1. Colours distinguish neighbours included at $k = 6$ and those added at $k = 2 4$ and 96, selected by three-dimensional distance. b, Whole-domain D1 (left) and D2 (right) changes in the negative-log leakage score (pLS) across neighbourhood sizes for the other five structures in Fig. 6; dashed lines mark $k = 2 4$ . c, Example D1 selections of 20, 40, 60 and 80 residues per chain, at a common centre. Coloured traces show the six corresponding segments against the complete D1 background, using the same view and scale. d, Regional D1 (left) and D2 (right) pLS changes at $k = 2 4$ , with four sequence-window lengths evaluated at common centres. Grey cells mark windows truncated at domain boundaries; each window retains the resolved common residues within its sequence interval. The opposing D1 and D2 responses associated with the UBXD1 lariat mutant persist across the neighbourhood sizes in b and window lengths in d, while shorter windows resolve more local variation. Selection and coverage details are given in Supplementary Note C.6.2.

![](images/94aed69dc77ea2f5c76febe0eee949f6b44d912358ceaad19c89967ba008c4ec.jpg)

Extended Data Fig. 3 | Regional symmetry responses in haemagglutinin. Five experimental structures <sup>32</sup>, comprising the neutral reference and acidified states I–IV (6Y5G–6Y5K), shown at a common scale. a, Global side views. $\mathbf { b } , \mathbf { c } ,$ , Top and side views highlighting HA1 (b) and HA2 (c), with the other region in grey. Colours identify protomers; darker and lighter shades distinguish HA1 and HA2. Values give the negative-log leakage score (pLS) against $\mathrm { C _ { 3 } }$ for each selection; ∆pLS is state IV minus state III. From state III to IV, global pLS increases, while HA1 pLS decreases and HA2 pLS increases, revealing opposing regional responses. All scores use the hybrid PF, degrees $0 , \ldots , 6$ and backbone graphs with $k = 2 4 \mathrm { : }$ traces show the corresponding Cα atoms. The reconstruction constraints are $\mathrm { C _ { 3 } , C _ { 3 } , C _ { 1 } , C _ { 1 } }$ and $\mathrm { C _ { 3 } }$ , respectively. Selection details are given in Supplementary Note C.6.3.

c  
b  
![](images/65e69f072ea3fdbad4eb2c9f548ed92e223b38147add8d64772b9c64108d73c7.jpg)

![](images/f2097f2cbef088067aff4d4aeea6e0a62138b6944a12ab61f2cd67f69de8552a.jpg)

![](images/ae7935896f6ee78ea4e978a26623c636974fcfc31c4a2e5391191c11d232f176.jpg)  
Extended Data Fig. 4 | Spatial symmetry organization in GluK2. ${ \mathbf { a } } ,$ Side view of the tetramer $( 5 \mathrm { K U F } ) ^ { 3 1 }$ , locating the amino-terminal (ATD), ligand-binding (LBD) and transmembrane (TMD) domains. b, Negative-log leakage scores (pLS) against $\mathrm { C _ { 4 } }$ for non-overlapping layers along the receptor axis, using thicknesses of 10–60 <sup>˚</sup>A. Blocks show the actual layer extents, aligned with ${ \mathbf { a } } ,$ with height measured from the lowest selected Cα position and each column’s shorter top remainder scored separately. The transmembrane region generally has higher pLS than the upper extracellular regions, a spatial pattern retained across the tested layer thicknesses. c, Top views and $\mathrm { C _ { 4 } \ p L S }$ of the three complete domain selections, shown at a common scale. Chains A–D retain the same colours across views. Scores use independently represented backbone selections with the hybrid PF, degrees $0 , \ldots , 6$ and $k = 2 4 ;$ traces show Cα atoms. The reconstruction imposes $\mathrm { C _ { 2 } }$ . Selection details are given in Supplementary Note C.6.4.

![](images/c0ea8d05b84395cfb856fd03cbcd9c1f87f177a04875a905f70b54f3a5dc6817.jpg)  
pLS = 6.56  
pLS = 1.65  
pLS = 1.15  
pLS = 0.85

Extended Data Fig. 5 | Icosahedral target agreement across echovirus 18 capsid structures. a, Deposited models of an intact activated E18 capsid and capsids lacking one, two or three pentamers <sup>33</sup>. Surfaces are envelopes of the sequence-matched Cα selections used for scoring. Colours distinguish adjacent pentamers, with corresponding positions assigned consistently after structural alignment. b, Cutaway views of the same models, with cavity-facing surfaces shaded grey. All views share a physical scale. The negative-log leakage scores (pLS) are evaluated at degree $l = 6$ for the proper icosahedral target group I, using the complete selected coordinates. From the intact capsid to structures lacking one, two or three pentamers, pLS decreases with the number of missing pentamers. Scores use the hybrid PF with $k = 2 4$ . The four structural classes were reconstructed with $\mathrm { I } , \mathrm { C } _ { 5 } , \mathrm { C } _ { 2 }$ and $\mathrm { C _ { 3 } }$ constraints, respectively. Selection details are given in Supplementary Note C.6.5.

<table><tr><td colspan="11">a Retrieval performance</td></tr><tr><td colspan="3"></td><td colspan="3">AUC</td><td colspan="3">Recall at 25% budget (%)</td><td colspan="2">pLS W/T/L</td></tr><tr><td>Group</td><td>Target symmetry</td><td></td><td>pLS</td><td>CSoM</td><td>Engel score</td><td>pLS</td><td>CSoM</td><td>Engel score</td><td>vs CSoM</td><td>vs Engel</td></tr><tr><td>Overall</td><td></td><td>一</td><td>0.794</td><td>0.747</td><td>0.787</td><td>68.8</td><td>66.9</td><td>66.3</td><td>41/36/9</td><td>23/53/10</td></tr><tr><td>All</td><td></td><td>w/</td><td>0.927</td><td>0.920</td><td>0.915</td><td>99.4</td><td>98.3</td><td>95.3</td><td>4/81/1</td><td>13/72/1</td></tr><tr><td>All</td><td></td><td>w/o</td><td>0.593</td><td>0.475</td><td>0.597</td><td>19.5</td><td>16.9</td><td>20.0</td><td>40/10/9</td><td>12/34/13</td></tr><tr><td>Maxima</td><td>w/</td><td></td><td>0.916 0.706</td><td>0.914</td><td>0.900</td><td>100.0</td><td>99.0</td><td>94.4</td><td>3/70/7</td><td>14/65/1</td></tr><tr><td>Maxima</td><td>w/o</td><td></td><td></td><td>0.601</td><td>0.627</td><td>38.9</td><td>25.0</td><td>33.3</td><td>13/6/8</td><td>11/10/6</td></tr><tr><td>Minima Minima</td><td>w/ w/o</td><td>0.926</td><td>0.908</td><td>0.924</td><td>93.8</td><td>93.1</td><td></td><td>93.8</td><td>9/29/2</td><td>4/31/5</td></tr><tr><td></td><td></td><td>0.567</td><td>0.454</td><td>0.591</td><td>17.9</td><td>16.5</td><td>19.8</td><td>33/13/7</td><td></td><td>7/29/17</td></tr><tr><td colspan="9">b Dataset composition</td></tr><tr><td>Group</td><td colspan="4">Target Molecules symmetry</td><td colspan="4">Molecule-target pairs</td><td colspan="2">Extrema, n (%)</td></tr><tr><td>Overall</td><td colspan="4">86</td><td colspan="4">94</td><td colspan="2">604 (100.0)</td></tr><tr><td>All</td><td colspan="4"></td><td colspan="4">94</td><td colspan="2">333 (55.1)</td></tr><tr><td>All</td><td colspan="4">w/ w/o</td><td colspan="4">61</td><td colspan="2">271 (44.9)</td></tr><tr><td>Maxima</td><td colspan="4">w/</td><td colspan="4">88</td><td colspan="2">245 (81.1)</td></tr><tr><td>Maxima</td><td colspan="4">w/o</td><td colspan="4">29</td><td colspan="2">57 (18.9)</td></tr><tr><td>Minima</td><td colspan="4">w/</td><td colspan="4">42</td><td colspan="2">88 (29.1)</td></tr><tr><td>Minima</td><td colspan="4">w/o</td><td colspan="4"></td><td colspan="2">214 (70.9)</td></tr></table>

Extended Data Table 1 | Energy-extremum retrieval and dataset composition by target symmetry. a, Mean AUC, recall at a 25% budget, and molecule-level AUC wins/ties/losses (W/T/L) for pLS against each baseline. Bold marks row-wise maxima for each metric. b, Extrema counts per molecule–target pair, with percentages within each extremum type. Target symmetry requires the point group assigned at tolerance 0.1 to contain the target. Rankings retain all 24 conformations; averaging follows Supplementary Note C.4.1.

Extended Data Table 2 | Efects of pretraining on pLS and energy-extremum retrieval. Changes are relative to paired random initializations. a, Mean maxima-minus-minima pLS gap. b, Mean target-present-minus-absent pLS gap, with target presence defined by point-group containment. Changes in raw pLS and AUC are averaged over targets within molecules, then equally over all 86 molecules. Improved molecules have increased gaps. Bold marks the largest gap increases and improvement proportions within each panel and pretraining group.
<table><tr><td colspan="7">a Energy extrema</td></tr><tr><td>GNN</td><td colspan="3">Change in pLS</td><td colspan="2">Improved molecules</td><td>∆AUC</td></tr><tr><td></td><td>Maxima</td><td>Minima</td><td>Gap</td><td>All (n = 86)</td><td>Validation (n = 17)</td><td></td></tr><tr><td colspan="7">Total-energy pretraining</td></tr><tr><td>HEGNN</td><td>-0.4718</td><td>-0.7146</td><td>+0.2428</td><td>54/86 (62.8%)</td><td>10/17 (58.8%)</td><td>+0.0061</td></tr><tr><td>TFN</td><td>-0.4048</td><td>-0.4192</td><td>+0.0143</td><td>45/86 (52.3%)</td><td>9/17 (52.9%)</td><td>-0.0088</td></tr><tr><td>MACE</td><td>-1.5250</td><td>-1.6448</td><td>+0.1198</td><td>56/86 (65.1%)</td><td>9/17 (52.9%)</td><td>+0.0006</td></tr><tr><td>Equiformer</td><td>-0.8038</td><td>-1.2372</td><td>+0.4334</td><td>67/86 (77.9%)</td><td>15/17 (88.2%)</td><td>-0.0069</td></tr><tr><td colspan="7">Uni-Mol-style pretraining</td></tr><tr><td>HEGNN</td><td>-1.9679</td><td>-1.9988</td><td>+0.0308</td><td>49/86 (57.0%)</td><td>9/17 (52.9%)</td><td>+0.0057</td></tr><tr><td>TFN</td><td>-0.7933</td><td>-0.9071</td><td>+0.1137</td><td>62/86 (72.1%)</td><td>10/17 (58.8%)</td><td>-0.0076</td></tr><tr><td>MACE</td><td>-2.0026</td><td>-2.1744</td><td>+0.1718</td><td>60/86 (69.8%)</td><td>10/17 (58.8%)</td><td>+0.0099</td></tr><tr><td>Equiformer</td><td>-0.0465</td><td>-0.5154</td><td>+0.4689</td><td>66/86 (76.7%)</td><td>15/17 (88.2%)</td><td>+0.0008</td></tr></table>

<table><tr><td>GNN</td><td colspan="3">Change in pLS</td><td colspan="2">Improved molecules</td></tr><tr><td></td><td>Target present</td><td>Target absent</td><td>Gap</td><td>All (n = 86)</td><td>Validation (n = 17)</td></tr><tr><td colspan="6">Total-energy pretraining</td></tr><tr><td>HEGNN</td><td>-0.3695</td><td>-0.9970</td><td>+0.6275</td><td>81/86 (94.2%)</td><td>17/17 (100.0%)</td></tr><tr><td>TFN</td><td>-0.3463</td><td>-0.4705</td><td>+0.1243</td><td>61/86 (70.9%)</td><td>11/17 (64.7%)</td></tr><tr><td>MACE</td><td>-1.4720</td><td>-1.7944</td><td>+0.3224</td><td>71/86 (82.6%)</td><td>15/17 (88.2%)</td></tr><tr><td>Equiformer</td><td>-0.8573</td><td>-1.2538</td><td>+0.3964</td><td>67/86 (77.9%)</td><td>14/17 (82.4%)</td></tr><tr><td colspan="6">Uni-Mol-style pretraining</td></tr><tr><td>HEGNN</td><td>-1.6033</td><td>-2.6896</td><td>+1.0864</td><td>81/86 (94.2%)</td><td>17/17 (100.0%)</td></tr><tr><td>TFN</td><td>-0.7853</td><td>-0.9514</td><td>+0.1661</td><td>64/86 (74.4%)</td><td>14/17 (82.4%)</td></tr><tr><td>MACE</td><td>-1.9376</td><td>-2.3959</td><td>+0.4583</td><td>76/86 (88.4%)</td><td>15/17 (88.2%)</td></tr><tr><td>Equiformer</td><td>-0.1030</td><td>-0.5324</td><td>+0.4294</td><td>69/86 (80.2%)</td><td>14/17 (82.4%)</td></tr></table>

## Extended Data Table 3 | Computational cost of structural symmetry analysis. a, Multi-target CPU timings for Dy-DOTA-water. b, CPU comparison on biological structures. c, SymFP CPU/GPU timings for whole structures and the p97 regional scan. CPU measurements use one thread. Times are medians except for single-trial biological coordinate fitting; † denotes a 7200 s timeout. Engel BFGS optimizes retained energy; BFGS + CSoM adds a coordinate-fitted start. Protocols and repetition counts are given in Supplementary Note C.7.

## a Dy-DOTA-water (30 heavy atoms): CPU total time (ms)

Multiple targets share computations within each call; baseline timings include orientation search.

<table><tr><td>Method</td><td>1 target  $\mathrm { C _ { 2 } }$ </td><td>2 targets  $\mathrm { C _ { 2 } , C _ { 4 } }$ </td><td>4 targets  $\mathrm { C _ { 2 } , C _ { 4 } , C _ { 6 } , D _ { 2 } }$ </td><td>7 targets  $\mathrm { C _ { 2 } , C _ { 4 } , C _ { 6 } , }$   $\mathrm { D _ { 2 } , D _ { 4 } , D _ { 6 } , O }$ </td></tr><tr><td>SymFP</td><td></td><td>6.59</td><td>6.87</td><td>7.21</td></tr><tr><td>CSoM</td><td>6.46 1575.40</td><td>5605.46</td><td>15901.44</td><td>96382.65</td></tr><tr><td>Engel, grid search</td><td>611.77</td><td>619.24</td><td>620.43</td><td>627.62</td></tr><tr><td>Engel, BFGS</td><td>297.45</td><td>499.16</td><td>975.75</td><td>1861.96</td></tr><tr><td>Engel, BFGS + CSoM</td><td>338.02</td><td>586.11</td><td>1454.12</td><td>7211.64</td></tr></table>

b Structural-analysis CPU total time (s)
<table><tr><td>Method</td><td>HA global (C3)</td><td>p97 common backbone  $\mathrm { ( C _ { 6 } ) }$ </td></tr><tr><td>SymFP</td><td>1.631</td><td>9.75</td></tr><tr><td>CSoM</td><td>≥ 7200†</td><td>≥ 7200†</td></tr><tr><td>Engel, grid search</td><td>16.175</td><td>244.378</td></tr><tr><td>Engel, BFGS</td><td>15.633</td><td>242.963</td></tr><tr><td>Engel, BFGS + CSoM</td><td>907.391</td><td>≥ 7200†</td></tr></table>

## c SymFP total time (s)

Atom and directed-edge ranges are per window. GPU totals include graph construction and data transfer.

<table><tr><td>Selection</td><td>Atoms</td><td>Edges</td><td>CPU</td><td>GPU</td></tr><tr><td>HA global</td><td>4,248</td><td>101,952</td><td>1.631</td><td>0.229</td></tr><tr><td>p97 common backbone</td><td>17,088</td><td>410,113</td><td>9.75</td><td>4.14</td></tr><tr><td>p97 regional scan (156 windows)</td><td>480-960</td><td>11,520–23,041</td><td>47.66</td><td>4.16</td></tr></table>

# Supplementary Information: Spectral symmetry fingerprints in equivariant representations

Jiacheng Cen<sup>1,2,3</sup>, Anyi Li<sup>1,2,3</sup>, Ning Lin<sup>1,2,3</sup>, Zihe Wang<sup>1,2,3</sup>, Ji-Rong Wen<sup>1,2,3</sup>, and Wenbing Huang<sup>1,2,3,4,</sup> <sup>B</sup>

<sup>1</sup>Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China

<sup>2</sup>Beijing Key Laboratory of Research on Large Models and Intelligent Governance, Beijing, China

<sup>3</sup>Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE, Beijing, China

<sup>4</sup>Beijing Academy of Artificial Intelligence, Beijing, China

<sup>B</sup>Correspondence should be addressed to: hwenbing@ruc.edu.cn

## Contents of Supplementary Information

A Theoretical analysis 39   
A.1 Point-group rank templates . 39   
A.2 Spectral invariance and leakage monotonicity 40   
A.3 Attainability and generic realization of equivariant representations 42   
A.4 Stability and continuous symmetry measures 44   
B Relation to existing symmetry measures 45   
B.1 The operation average in representation space 46   
B.2 Spectral and orientation-conditioned retained energy 46   
C Datasets and evaluation settings 47   
C.1 Shared representation and scoring settings . 47   
C.2 Synthetic point-group benchmark . 48   
C.3 Controlled deformation benchmarks . 50   
C.4 Biaryl torsion benchmark 51   
C.5 DOTA complexes . 54   
C.6 Biological assemblies 55   
C.7 Computational cost . 56   
D Supporting studies 57   
D.1 Spectral responses across equivariant GNNs 57   
D.2 Graph-neighbourhood and window-scale checks for p97 58   
D.3 Domain-resolved responses in haemagglutinin 58   
D.4 Spatial symmetry organization in GluK2 59   
D.5 Icosahedral target agreement in E18 capsids 59

## A Theoretical analysis

The following results establish the point-group rank templates, spectral invariance and subgroup ordering. We then relate full-span attainability to generic rank realization and quantify spectral departures under coordinate perturbations within a fixed graph family.

## A.1 Point-group rank templates

Theorem 3 (Fixed-subspace constraint). Suppose that $\tilde { V } ^ { ( l ) }$ is an $\mathrm { O ( 3 ) }$ -equivariant representation map satisfying Eq. (1). For any $\mathcal { G } \in \mathbb { G } ( \mathfrak { H } )$ , write $\mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) : = \left\{ \tilde { \pmb { v } } \in \mathbb { R } ^ { 2 \dot { l } + \dot { 1 } } : \rho ^ { ( l ) } ( \mathfrak { h } ) \tilde { \pmb { v } } = \tilde { \pmb { v } } \ \forall \mathfrak { h } \in \mathfrak { H } \right\}$ for the fixed subspace and $\rho ^ { ( l ) } ( \mathfrak { H } ) : = 1 / | \mathfrak { H } | \cdot \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { h } )$ for the group average. Then

$$
\mathrm { c o l s p a n } ( \tilde { \mathbf { V } } ^ { ( l ) } ( \mathcal { G } ) ) \subseteq \mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) ,\tag{16}
$$

and consequently

$$
\operatorname { r a n k } ( { \tilde { \mathbf { V } } } ^ { ( l ) } ( { \mathcal { G } } ) ) \leq m _ { l } ( { \mathfrak { H } } ) ,\tag{17}
$$

where $m _ { l } ( { \mathfrak { H } } )$ is the multiplicity of the trivial representation in $\rho ^ { ( l ) } | _ { \mathfrak { H } }$ , that is $m _ { l } ( \mathfrak { H } ) : = \dim \mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) =$ rank $( \rho ^ { ( l ) } ( { \bf \dot { \{ \sigma } }  ) ) = \mathrm { t r } ( \rho ^ { ( l ) } ( { \bf \dot { \{ \sigma } }  ) )$

Proof. For $\mathcal { G } \in \mathbb { G } ( \mathfrak { H } )$ , equivariance gives $\tilde { \cal V } ^ { ( l ) } ( \mathfrak { h } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { h } ) \tilde { \cal V } ^ { ( l ) } ( \mathcal { G } ) = \tilde { \cal V } ^ { ( l ) } ( \mathcal { G } )$ for every ${ \mathfrak { h } } \in { \mathfrak { H } }$ Define the group average $\rho ^ { ( l ) } ( \mathfrak { H } ) : = 1 / | \mathfrak { H } | \cdot \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { h } )$ . Averaging the equivariance identity yields

$$
\tilde { \cal V } ^ { ( l ) } ( { \mathcal G } ) = \rho ^ { ( l ) } ( \mathfrak { H } ) \tilde { \cal V } ^ { ( l ) } ( { \mathcal G } ) .\tag{38}
$$

Hence every column of $\tilde { V } ^ { ( l ) } ( \mathcal { G } )$ lies in Im $( \rho ^ { ( l ) } ( \mathfrak { H } ) )$ , so

$$
\mathrm { c o l s p a n } ( \tilde { \mathbf { V } } ^ { ( l ) } ( \mathcal { G } ) ) \subseteq \mathrm { I m } ( \rho ^ { ( l ) } ( \mathfrak { H } ) ) .\tag{39}
$$

It remains to identify this image with the fixed subspace. $\mathrm { I f } \ \tilde { \pmb { v } } ^ { ( l ) } = \rho ^ { ( l ) } ( \mathfrak { H } ) \tilde { \pmb { u } }$ for some $\tilde { \ b { u } } \in \mathbb { R } ^ { 2 l + 1 }$ 2 then for any ${ \mathfrak { h } } _ { 0 } \in { \mathfrak { H } }$ ,

$$
\rho ^ { ( l ) } ( \mathfrak { h } _ { 0 } ) \tilde { \pmb { v } } ^ { ( l ) } = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { h } _ { 0 } \mathfrak { h } ) \tilde { \pmb { u } } = \rho ^ { ( l ) } ( \mathfrak { H } ) \tilde { \pmb { u } } = \tilde { \pmb { v } } ^ { ( l ) } ,\tag{40}
$$

since ${ \mathfrak { h } } _ { 0 } { \mathfrak { H } } = { \mathfrak { H } }$ . Thus $\tilde { \pmb { v } } ^ { ( l ) } \in \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ . Conversely, if $\tilde { \pmb { v } } ^ { ( l ) } \in \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ , then

$$
\rho ^ { ( l ) } ( \mathfrak { H } ) \tilde { \pmb { v } } ^ { ( l ) } = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { h } ) \tilde { \pmb { v } } ^ { ( l ) } = \tilde { \pmb { v } } ^ { ( l ) } ,\tag{41}
$$

so $\tilde { \pmb { v } } ^ { ( l ) } \in \mathrm { I m } ( \rho ^ { ( l ) } ( \pmb { \mathscr { H } } ) )$

Therefore Im $( \rho ^ { ( l ) } ( \mathfrak { H } ) ) = \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ . The same averaging argument yields

$$
( \rho ^ { ( l ) } ( \mathfrak { H } ) ) ^ { 2 } = \rho ^ { ( l ) } ( \mathfrak { H } ) ,\tag{42}
$$

so $\rho ^ { ( l ) } ( \mathfrak { H } )$ is an idempotent projection. Moreover, each $\rho ^ { ( l ) } ( { \mathfrak { h } } )$ is orthogonal, hence

$$
( \rho ^ { ( l ) } ( \mathfrak { H } ) ) ^ { \top } = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { h } ) ^ { \top } = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { h } ^ { - 1 } ) = \rho ^ { ( l ) } ( \mathfrak { H } ) ,\tag{43}
$$

because ${ \mathfrak H }$ is closed under inversion. Thus $\rho ^ { ( l ) } ( \mathfrak { H } )$ is self-adjoint as well, so it is the orthogonal projector onto $\mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ . Hence all its eigenvalues are 0 or 1, and therefore

$$
\operatorname { r a n k } ( \rho ^ { ( l ) } ( \mathfrak { H } ) ) = \operatorname { t r } ( \rho ^ { ( l ) } ( \mathfrak { H } ) ) = \dim \mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) .\tag{44}
$$

The column-space inclusion and the rank inequality in Eqs. (16) and (17) follow immediately.

If a finite candidate family C has pairwise distinct rank templates over all natural-parity degrees, finitely many degrees sufice to separate it. Indeed, choose one distinguishing degree for each pair of groups; the finite union of these degrees gives a separating set Λ.

Let $\mathbb { C } _ { 3 2 }$ denote the 32 crystallographic point groups in Schoenflies notation<sup>1</sup>:

$$
\begin{array} { r } { \mathbb { C } _ { 3 2 } = \left\{ \begin{array} { l l } { \mathrm { C _ { 1 } , C _ { i } , C _ { 2 } , C _ { s } , C _ { 2 h } , D _ { 2 } , C _ { 2 v } , D _ { 2 h } , C _ { 4 } , S _ { 4 } , C _ { 4 h } , } } \\ { \mathrm { D _ { 4 } , C _ { 4 v } , D _ { 2 d } , D _ { 4 h } , C _ { 3 } , S _ { 6 } , D _ { 3 } , C _ { 3 v } , D _ { 3 d } , C _ { 6 } , C _ { 3 h } , } } \\ { \mathrm { C _ { 6 h } , D _ { 6 } , C _ { 6 v } , D _ { 3 h } , D _ { 6 h } , T , T _ { h } , O , T _ { d } , O _ { h } } } \end{array} \right\} . } \end{array}\tag{45}
$$

Corollary 4 (Rank templates for the 32 crystallographic point groups). Let $\mathbb { C } _ { 3 2 }$ denote the 32 crystallographic point groups<sup>1</sup>. The map ${ \mathfrak { H } } \mapsto$ RankTemplate (H) is injective on $\mathbb { C } _ { 3 2 }$ . The shorter degree set $\Lambda _ { 6 }$ gives 30 distinct templates, with only the pairs $\mathrm { D } _ { 6 } / \mathrm { D } _ { 6 \mathrm { h } }$ and $\mathrm { O } / \mathrm { O } _ { \mathrm { h } }$ sharing templates.

Proof. For every natural-parity degree l, the quantity $m _ { l } ( { \mathfrak { H } } )$ is the multiplicity of the trivial representation in $\rho ^ { ( l ) } | _ { \mathfrak { H } }$ , equivalently the character average $\operatorname { t r } ( \rho ^ { ( l ) } ( \mathfrak { H } ) )$ . Evaluating these multiplicities for the 32 crystallographic point groups gives the natural-parity rows of Supplementary Table A.1. Restricting to degrees up to 9 leaves the rows pairwise distinct, which proves injectivity for $\Lambda _ { 9 } ;$ restricting to degrees up to 6 gives 30 distinct rows, with the two coincidences stated, between $\mathrm { D } _ { 6 }$ and $\mathrm { D } _ { \mathrm { 6 h } }$ and between O and $\mathrm { O } _ { \mathrm { h } }$ □

Proposition 5 (Subgroup monotonicity). $I f \ { \mathfrak { H } } _ { 1 } \leq { \mathfrak { H } } _ { 2 } \leq \mathrm { O } ( 3 )$ , then $\mathbb { F } ^ { ( l ) } ( \mathfrak { H } _ { 2 } ) \subseteq \mathbb { F } ^ { ( l ) } ( \mathfrak { H } _ { 1 } )$ and $m _ { l } ( { \mathfrak { H } } _ { 2 } ) \leq m _ { l } ( { \mathfrak { H } } _ { 1 } )$ for every l. Consequently, RankTemplate (H ) ≤ RankTemplat $\mathrm { e } _ { \Lambda } ( \mathfrak { H } _ { 1 } )$ componentwise for every finite degree set Λ.

Proof. If $\tilde { \pmb { v } } ^ { ( l ) } \in \mathbb { F } ^ { ( l ) } ( { \mathfrak { H } } _ { 2 } )$ , then $\rho ^ { ( l ) } ( { \mathfrak { h } } ) { \tilde { \pmb { v } } } ^ { ( l ) } = { \tilde { \pmb { v } } } ^ { ( l ) }$ for every ${ \mathfrak { h } } \in { \mathfrak { H } } _ { 2 }$ , and hence in particular for every ${ \mathfrak { h } } \in { \mathfrak { H } } _ { 1 }$ . Thus $\tilde { \pmb { v } } ^ { ( l ) } \in \mathbb { F } ^ { ( l ) } ( { \mathfrak { H } } _ { 1 } )$ , which proves the subspace inclusion. The dimension and rank-template inequalities follow immediately. □

## A.2 Spectral invariance and leakage monotonicity

Proposition 6 (Leakage monotonicity along subgroup chains). $I f \mathfrak { H } _ { 1 } \leq \mathfrak { H } _ { 2 } \leq \mathrm { O } ( 3 )$ , then for every geometric graph $\mathcal { G }$

$$
\begin{array} { r } { \mathrm { L S } _ { \Lambda , \mathfrak { H } _ { 1 } } ( \mathcal { G } ) \leq \mathrm { L S } _ { \Lambda , \mathfrak { H } _ { 2 } } ( \mathcal { G } ) , \qquad \mathrm { p L S } _ { \Lambda , \mathfrak { H } _ { 1 } } ( \mathcal { G } ) \geq \mathrm { p L S } _ { \Lambda , \mathfrak { H } _ { 2 } } ( \mathcal { G } ) . } \end{array}\tag{33}
$$

Proof. Fix a geometric graph $\mathcal { G }$ and suppress the explicit argument (G) in the derived leakage quantities.

If $\mathfrak { H } _ { 1 } \le \mathfrak { H } _ { 2 }$ , then Prop. 5 gives

$$
m _ { l } ( { \mathfrak { H } } _ { 2 } ) \leq m _ { l } ( { \mathfrak { H } } _ { 1 } ) \qquad { \mathrm { f o r ~ e v e r y ~ } } l \in \Lambda .\tag{46}
$$

Hence the tail index set in Eq. (28) for ${ \mathfrak { H } } _ { 1 }$ is contained in the corresponding tail index set for ${ \mathfrak { H } } _ { 2 }$ so

$$
\sum _ { i > m _ { l } ( \mathfrak { H } _ { 1 } ) } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } \leq \sum _ { i > m _ { l } ( \mathfrak { H } _ { 2 } ) } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } .\tag{47}
$$

The denominator in Eq. (28) is the same for both target groups; therefore

$$
\mathrm { L S } _ { 5 _ { 1 } } ^ { ( l ) } \le \mathrm { L S } _ { 5 _ { 2 } } ^ { ( l ) } .\tag{48}
$$

Averaging over $l \in \Lambda$ as in Eq. (29) yields the first inequality in Eq. (33). Since $x \mapsto - \log _ { 1 0 } ( x + \varepsilon _ { \log } )$ is strictly decreasing on $[ 0 , \infty )$ , the second inequality for pLS follows immediately. □

Supplementary Table A.1. Fixed-subspace dimensions of crystallographic point groups. Values use natural-parity representations with $l = 0 , \ldots , 1 1$ . The final column gives the crystal-field parameter count $\begin{array} { r } { n _ { \mathrm { C F } } = \sum _ { k \in \{ 2 , 4 , 6 \} } m _ { k } } \end{array}$ (§ 2.5).
<table><tr><td rowspan="2">Schoenflies</td><td rowspan="2">Hermann-Mauguin</td><td colspan="10">Fixed-subspace dimension  $m _ { l }$  (natural parity)</td><td rowspan="2">11</td><td rowspan="2">nCF</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8 9</td><td>10</td></tr><tr><td> $\mathrm { C _ { 1 } }$ </td><td>1</td><td>1</td><td>3</td><td>5</td><td>7</td><td>9</td><td>11</td><td>13 15</td><td>17</td><td>19</td><td>21</td><td>23</td><td>27</td></tr><tr><td> $\mathrm { C _ { i } }$ </td><td>-1</td><td>1</td><td>0 5</td><td>0</td><td>9</td><td>0</td><td>13</td><td>0</td><td>17</td><td>0</td><td>21</td><td>0</td><td>27</td></tr><tr><td> $\mathrm { C _ { 2 } }$ </td><td>2</td><td>1</td><td>1 3</td><td>3</td><td>5</td><td>5</td><td>7</td><td>7</td><td>9</td><td>9</td><td>11</td><td>11</td><td>15</td></tr><tr><td> $\mathrm { C _ { s } }$ </td><td>m</td><td>1</td><td>2 3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>15</td></tr><tr><td> $\mathrm { C _ { 2 h } }$ </td><td> $2 / m$ </td><td>1</td><td>0 3</td><td>0</td><td>5</td><td>0</td><td>7</td><td>0</td><td>9</td><td>0</td><td>11</td><td>0</td><td>15</td></tr><tr><td> $\mathrm { D _ { 2 } }$ </td><td>222</td><td>1</td><td>0 2</td><td>1</td><td>3</td><td>2</td><td>4</td><td>3</td><td>5</td><td>4</td><td>6</td><td>5</td><td>9</td></tr><tr><td> $\mathrm { C _ { 2 v } }$ </td><td>mm2</td><td>1</td><td>1 2</td><td>2</td><td>3</td><td>3</td><td>4</td><td>4</td><td>5</td><td>5</td><td>6</td><td>6</td><td>9</td></tr><tr><td> $\mathrm { D _ { 2 h } }$ </td><td>mmm</td><td>1</td><td>0 2</td><td>0</td><td>3</td><td>0</td><td>4</td><td>0</td><td>5</td><td>0</td><td>6</td><td>0</td><td>9</td></tr><tr><td> $\mathrm { C _ { 4 } }$ </td><td>4</td><td>1</td><td>1 1</td><td>1</td><td>3</td><td>3</td><td>3</td><td>3</td><td>5</td><td>5</td><td>5</td><td>5</td><td>7</td></tr><tr><td> $\mathrm { S _ { 4 } }$ </td><td>-4</td><td>1 0</td><td>1</td><td>2</td><td>3</td><td>2</td><td>3</td><td>4</td><td>5</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td> $\mathrm { C _ { 4 h } }$ </td><td> $4 / m$ </td><td>1 0</td><td>1</td><td>0</td><td>3</td><td>0</td><td>3</td><td>0</td><td>5</td><td>0</td><td>5</td><td>0</td><td>7</td></tr><tr><td> $\mathrm { D } _ { 4 }$ </td><td>422</td><td>1 0</td><td>1</td><td>0</td><td>2</td><td>1</td><td>2</td><td>1</td><td>3</td><td>2</td><td>3</td><td>2</td><td>5</td></tr><tr><td> $\mathrm { C } _ { 4 \mathrm { v } }$ </td><td>4mm</td><td>1 1</td><td>1</td><td>1</td><td>2</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td><td>3</td><td>3</td><td>5</td></tr><tr><td> $\mathrm { D _ { 2 d } }$ </td><td>-42m</td><td>1 0</td><td>1</td><td>1</td><td>2</td><td>1</td><td>2</td><td>2</td><td>3</td><td>2</td><td>3</td><td>3</td><td>5</td></tr><tr><td> $\mathrm { D } _ { 4 \mathrm { h } }$ </td><td> $4 / m m m$ </td><td>1 0</td><td>1</td><td>0</td><td>2</td><td>0</td><td>2</td><td>0</td><td>3</td><td>0</td><td>3</td><td>0</td><td>5</td></tr><tr><td> $\mathrm { C _ { 3 } }$ </td><td>3</td><td>1</td><td>1 1</td><td>3</td><td>3</td><td>3</td><td>5</td><td>5</td><td>5</td><td>7</td><td>7</td><td>7</td><td>9</td></tr><tr><td> $\mathrm { S } _ { 6 }$ </td><td>-3</td><td>1</td><td>0 1</td><td>0</td><td>3</td><td>0</td><td>5</td><td>0</td><td>5</td><td>0</td><td>7</td><td>0</td><td>9</td></tr><tr><td> $\mathrm { D _ { 3 } }$ </td><td>32</td><td>1 0</td><td>1</td><td>1</td><td>2</td><td>1</td><td>3</td><td>2</td><td>3</td><td>3</td><td>4</td><td>3</td><td>6</td></tr><tr><td> $\mathrm { C } _ { \mathrm { 3 v } }$ </td><td>3m</td><td>1</td><td>1 1</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td><td>3</td><td>4</td><td>4</td><td>4</td><td>6</td></tr><tr><td> $\mathrm { D _ { 3 d } }$ </td><td> $^ { - 3 m }$ </td><td>1 0</td><td>1</td><td>0</td><td>2</td><td>0</td><td>3</td><td>0</td><td>3</td><td>0</td><td>4</td><td>0</td><td>6</td></tr><tr><td> $\mathrm { C _ { 6 } }$ </td><td>6</td><td>1</td><td>1 1</td><td>1</td><td>1</td><td>1</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>5</td></tr><tr><td> $\mathrm { C _ { 3 h } }$ </td><td>-6</td><td>1 0</td><td>1</td><td>2</td><td>1</td><td>2</td><td>3</td><td>2</td><td>3</td><td>4</td><td>3</td><td>4</td><td>5</td></tr><tr><td> $\mathrm { C } _ { \mathrm { G h } }$ </td><td> $6 / m$ </td><td>1 0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>3</td><td>0</td><td>3</td><td>0</td><td>3</td><td>0</td><td>5</td></tr><tr><td> $\mathrm { D } _ { 6 }$ </td><td>622</td><td>1 0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>2</td><td>1</td><td>2</td><td>1</td><td>2</td><td>1</td><td>4</td></tr><tr><td> $\mathrm { C } _ { 6 \mathrm { v } }$ </td><td>6mm</td><td>1 1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>4</td></tr><tr><td> $\mathrm { D } _ { \mathrm { 3 h } }$ </td><td> $- 6 m 2$ </td><td>1 0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>2</td><td>1</td><td>2</td><td>2</td><td>2</td><td>2</td><td>4</td></tr><tr><td> $\mathrm { D } _ { 6 \mathrm { h } }$ </td><td> $6 / m m m$ </td><td>1 0</td><td>1</td><td>0</td><td>1</td><td>0</td><td>2</td><td>0</td><td>2</td><td>0</td><td>2</td><td>0</td><td>4</td></tr><tr><td> $_ \mathrm { T }$ </td><td>23</td><td>1 0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>2</td><td>1</td><td>1</td><td>2</td><td>2</td><td>1</td><td>3</td></tr><tr><td> $\mathrm { T _ { h } }$ </td><td>m-3</td><td>1 0</td><td>0 0 0</td><td>0 0</td><td>1 1</td><td>0 0</td><td>2 1</td><td>0 0</td><td>1 1</td><td>0 1</td><td>2 1</td><td>0 0</td><td>3 2</td></tr><tr><td> $\mathrm { O }$   $\mathrm { T _ { d } }$ </td><td>432 -43m</td></table>

Proposition 7 (Orthogonal invariance of SymFP). Assume that for every $l \in \Lambda$ , every geometric graph G, and every ${ \mathfrak { g } } \in \mathrm { O } ( 3 )$

$$
\tilde { \mathbf { V } } ^ { ( l ) } ( \mathfrak { g } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { g } ) \tilde { \mathbf { V } } ^ { ( l ) } ( \mathcal { G } ) .\tag{49}
$$

Th $e n ,$ for every $l \in \Lambda$

$$
\pmb { \sigma } ^ { ( l ) } ( \mathfrak { g } \cdot \mathcal { G } ) = \pmb { \sigma } ^ { ( l ) } ( \mathcal { G } ) .\tag{50}
$$

Consequently, Sym $\mathrm { F P } _ { \Lambda } , ~ \mathrm { R a n k S e q } _ { \Lambda , \tau } , ~ \mathrm { L S } _ { \Lambda , 5 }$ , and $\mathrm { p L S } _ { \Lambda , \mathfrak { H } }$ are all invariant under the global O(3) action.

Proof. Assume Eq. (49). Since $\rho ^ { ( l ) } ( { \mathfrak { g } } )$ is orthogonal, left multiplication by $\rho ^ { ( l ) } ( { \mathfrak { g } } )$ does not change the singular values of $\tilde { V } ^ { ( l ) } ( \mathcal { G } )$ . Therefore Eq. (50) holds for every $l \in \Lambda$ . The remaining listed quantities are deterministic functions of these singular values, the invariant threshold in Eq. (24), and fixed template multiplicities $m _ { l } ( { \mathfrak { H } } )$ , so they inherit the same global $\mathrm { O ( 3 ) }$ invariance. □

Right multiplication of a representation block by an orthogonal channel matrix also preserves its singular values. A general non-isometric channel transformation can change the spectrum and its leakage scores, since these use the Euclidean metric of the chosen channel space.

Limiting cases of degree-resolved leakage. Two limiting cases clarify the behavior of degreeresolved leakage. When $m _ { l } ( { \mathfrak { H } } ) = 2 l + 1$ , the spectral tail is empty and $\mathrm { L S } _ { 5 } ^ { ( l ) } = 0$ regardless of normalization. Such degrees impose no rank constraint and contribute nothing to the pooled score. An example is the natural-parity scalar representation at $l = 0$ , for which $m _ { 0 } ( { \mathfrak { H } } ) = 1$ for every point group. Conversely, when $m _ { l } ( { \mathfrak { H } } ) = 0$ , the fixed subspace is trivial and all block energy lies in the spectral tail. Under the default block-energy normalization, $\mathrm { L S } _ { 5 } ^ { ( l ) }$ approaches unity as the block energy becomes large relative to $\varepsilon _ { \mathrm { n o r m } } .$

## A.3 Attainability and generic realization of equivariant representations

For fixed node count, centred geometric graphs modulo node relabelling form a metrizable space under the natural quotient topology, with a continuous $\mathrm { O ( 3 ) }$ action. The following extension-andaveraging argument<sup>2</sup> establishes pointwise attainability.

Proposition 8 (Pointwise attainability of the fixed subspace). Let a compact Hausdorf group K act continuously on a metric space G, with continuous output representation $\rho ^ { ( l ) } : \mathbb { A } \to { \mathrm { G L } } ( \mathbb { R } ^ { 2 l + 1 } )$ For each $\mathcal { G }$ with full stabilizer H, the continuous equivariant maps satisfy

$$
\{ f ( \mathcal { G } ) : f \in C _ { \& } ( \mathbb { G } , \mathbb { R } ^ { 2 l + 1 } ) \} = \mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) .\tag{51}
$$

Proof. Equivariance gives the forward inclusion. For $\tilde { \pmb { v } } \in \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ , define $\varphi ( \mathfrak { g } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { g } ) \tilde { v }$ on the orbit $\mathcal { O } _ { \mathcal { G } }$ . Since v˜ is fixed by H and ${ \mathcal { O } } _ { \mathcal { G } } \cong \mathbb { \mathcal { R } } / { \mathfrak { H } }$ , this map is well-defined and continuous. The orbit is compact and closed, so coordinatewise Tietze extension gives a continuous $F : \mathbb { G } \to \mathbb { R } ^ { 2 l + 1 }$ extending $\varphi .$ With normalized Haar measure $\nu ,$ , set

$$
f ( \mathcal { G } ^ { \prime } ) = \int _ { \mathbb { A } } \rho ^ { ( l ) } ( \mathfrak { g } ) ^ { - 1 } F ( \mathfrak { g } \cdot \mathcal { G } ^ { \prime } ) \mathrm { d } \nu ( \mathfrak { g } ) .\tag{52}
$$

Compactness and joint continuity imply continuity of $f .$ Substituting ${ \mathfrak { a } } = { \mathfrak { g } } ^ { \sharp }$ and using right invariance of ν gives $f ( \mathfrak { k } \cdot \mathcal { G } ^ { \prime } ) = \rho ^ { ( l ) } ( \mathfrak { k } ) f ( \mathcal { G } ^ { \prime } )$ . On the original orbit, $F ( { \mathfrak { g } } \cdot { \mathcal { G } } ) = \rho ^ { ( l ) } ( { \mathfrak { g } } ) { \tilde { v } }$ , hence $f ( \mathcal G ) = \tilde { v }$ □

Corollary 9 (Rank attainment by universal equivariant networks). Under these assumptions, suppose an equivariant network family with C output channels approximates all continuous equivariant matrix maps uniformly on compact sets. For each G with full stabilizer H and $C \geq m _ { l } ( \mathfrak { H } )$ , some network satisfies

$$
\operatorname { r a n k } { \tilde { V } } _ { \pmb { \theta } } ^ { ( l ) } ( { \mathcal { G } } ) = m _ { l } ( { \mathfrak { H } } ) .\tag{53}
$$

The network and its architecture may depend on $\mathcal { G }$

Proof. Set $m = m _ { l } ( { \mathfrak { H } } )$ ; the case $m = 0$ is immediate. By Prop. 8, realize an orthonormal basis of $\mathbb { F } ^ { ( l ) } ( { \mathfrak { H } } )$ as columns of a continuous equivariant map $\tilde { V } _ { * } ^ { ( l ) }$ , padding with zero columns. Then $\sigma _ { m } ( \tilde { V } _ { * } ^ { ( l ) } ( { \mathcal G } ) ) = 1$ . Approximation within $1 / 2$ in operator norm gives $\sigma _ { m } ( \tilde { V } _ { \theta } ^ { ( l ) } ( { \mathcal { G } } ) ) > 1 / 2$ by singularvalue perturbation. Thus the rank is at least $m ,$ and equivariance bounds it by $m$ □

From attainability to generic realization. For a fixed architecture, input and degree, one full-span parameter setting sufices to establish full span almost everywhere under real-analytic parameter dependence on a connected open domain. The following results formalize this consequence using the zero-set theorem for real-analytic functions<sup>3</sup>.

Fix a geometric graph $\mathcal { G }$ with finite stabilizer ${ \mathfrak { H } } = { \mathrm { S t a b } } ( { \mathcal { G } } )$ , a degree l, and an equivariant architecture. Let $\tilde { V } _ { \theta } ^ { ( l ) } ( { \mathcal { G } } ) \in \mathbb { R } ^ { ( 2 l + 1 ) \times C }$ denote its representation block, parameterized by $\pmb \theta \in \Omega \subset$ $\mathbb { R } ^ { P }$ . Assume that Ω is nonempty, open, and connected, and that the block entries depend real analytically on θ. Define the maximal attainable rank

$$
r _ { l } ^ { * } ( { \mathcal { G } } ) : = \operatorname* { m a x } _ { \theta \in \Omega } \operatorname { r a n k } { \tilde { V } } _ { \theta } ^ { ( l ) } ( { \mathcal { G } } ) .\tag{54}
$$

By Thm. 3, every column belongs to $\mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ , hence

$$
\operatorname { r a n k } \tilde { V } _ { \theta } ^ { ( l ) } ( \mathcal { G } ) \leq r _ { l } ^ { \ast } ( \mathcal { G } ) \leq \operatorname* { m i n } \{ C , m _ { l } ( \mathfrak { H } ) \} .\tag{55}
$$

Proposition 10 (Generic maximal-rank realization). Under these assumptions, rank $\tilde { V } _ { \theta } ^ { ( l ) } ( { \mathcal G } ) =$ $r _ { l } ^ { * } ( { \mathcal { G } } )$ for Lebesgue-almost every $\pmb \theta \in \Omega$ . Thus, for any probability distribution $\mu$ absolutely continuous with respect to Lebesgue measure on $\Omega$ ,

$$
\operatorname* { P r } _ { \theta \sim \mu } \Bigl [ \mathrm { r a n k } \tilde { V } _ { \theta } ^ { ( l ) } ( \mathcal { G } ) = r _ { l } ^ { \ast } ( \mathcal { G } ) \Bigr ] = 1 .\tag{56}
$$

Proof. Let $r = r _ { l } ^ { * } ( \mathcal { G } )$ . The case $r = 0$ is immediate. For $r > 0$ , choose $\pmb { \theta } _ { * } \in \Omega$ attaining rank r and row and column index sets $\mathbb { I } , \mathbb { J }$ of size r such that

$$
g ( \pmb \theta ) : = \operatorname* { d e t } \big ( \tilde { V } _ { \pmb \theta } ^ { ( l ) } ( \mathcal { G } ) [ \mathbb { I } , \mathbb { J } ] \big )\tag{57}
$$

satisfies $g ( \pmb { \theta } _ { * } ) \neq 0$ . Since $g$ is real analytic and not identically zero, its zero set has Lebesgue measure $\mathrm { z e r o ^ { 3 } }$ . Rank deficiency relative to r implies $g ( \pmb \theta ) = 0$ , while maximality precludes larger ranks. The probability statement follows by absolute continuity. □

Corollary 11 (Attainable full span is generic). Under the same assumptions, suppose $C \geq m _ { l } ( \mathfrak { H } )$ and full span is attained at some $\pmb { \theta } _ { * } \in \Omega , i . e .$ , rank $\tilde { V } _ { \pmb { \theta } _ { \ast } } ^ { ( l ) } ( { \mathcal G } ) = m _ { l } ( { \mathcal G } )$ . Then

$$
\mathrm { c o l s p a n } \big ( \tilde { V } _ { \theta } ^ { ( l ) } ( \mathcal { G } ) \big ) = \mathbb { F } ^ { ( l ) } ( \mathfrak { H } ) \qquad f o r \mu \ – a l m o s t \ e v e r y \ \pmb { \theta } \in \Omega .\tag{58}
$$

Proof. The attainable-rank constraint gives $r _ { l } ^ { * } ( { \mathcal { G } } ) = m _ { l } ( { \mathfrak { H } } )$ . The claim follows from Prop. 10 and the column-space inclusion. □

Corollary 12 (Simultaneous realization on a finite collection). Fix an architecture and parameter domain Ω, and let $\mathbb { K } = \left\{ ( \mathcal { G } _ { i } , l ) : 1 \le i \le N , \ l \in \Lambda _ { i } \right\}$ be finite. If the analytic assumptions hold and full span is attainable for every pair, then, with probability one under $\mu _ { ; }$

$$
\mathrm { c o l s p a n } \big ( \tilde { V } _ { \boldsymbol { \theta } } ^ { ( l ) } ( \mathcal { G } _ { i } ) \big ) = \mathbb { F } ^ { ( l ) } ( \mathrm { S t a b } ( \mathcal { G } _ { i } ) )\tag{59}
$$

simultaneously for all $( \mathcal { G } _ { i } , l ) \in \mathbb { K }$

Proof. Each pair has a null exceptional set by Cor. 11; their finite union remains null. □

The full-span witnesses may difer across pairs, yet simultaneous realization holds generically. The result requires neither independent channels nor Gaussian representations and applies to any fixed finite collection within an architecture satisfying the analytic assumptions.

Scope for the evaluated GNNs. We consider pre-SVD representation blocks at fixed inputs in exact arithmetic, holding initialization-fixed parameters and running statistics constant. HEGNN is analytic in its free parameters, whereas radial ReLU in TFN/MACE and guarded attention in Equiformer restrict generic full-span realization to connected analytic regions where full span is attainable. Replacing radial ReLU with SiLU yields analytic TFN/MACE variants without sacrificing attainability. Indeed, scaling the first afine map by t and the second weight matrix by t<sup>−1</sup> gives $t ^ { - 1 } \mathrm { S i L U } ( t z ) \to \mathrm { R e L U } ( z )$ as $t \to \infty$ . The resulting blocks converge to their ReLU counterparts, preserving every nonzero minor of order $m _ { l } ( { \mathfrak { H } } )$ for suficiently large finite t. Thus, Cor. 11 applies to these analytic variants over the unrestricted free-parameter space. PF is deterministic in this setting.

Numerical evidence of full-span attainability. The rank-template match rate measures agreement between numerical rank sequences and the true point-group templates across all retained degrees. In Supplementary Table C.4, TFN, MACE, and Equiformer recover the complete rank sequences for all 320 clean test inputs at $\Lambda _ { 6 }$ across three initializations, with TFN and MACE also achieving perfect recovery at $\Lambda _ { 9 }$ . These results provide numerical evidence of full-span attainability in the evaluated configurations, although rank recovery remains sensitive to the positive singular values and the numerical threshold.

## A.4 Stability and continuous symmetry measures

Let U be a family of geometric graphs with identical node attributes, edge attributes, and adjacency, difering only in their coordinates. Equip U with a permutation-invariant metric dist.

Theorem 13 (Local stability of SymFP on a fixed graph family). Fix a finite degree set Λ, a finite family of point groups C, and a graph family U as above. Suppose that, for each $l \in \Lambda$ , there exists $L _ { l } > 0$ such that

$$
\lVert \tilde { \mathbf { V } } ^ { ( l ) } ( \mathcal { G } ) - \tilde { \mathbf { V } } ^ { ( l ) } ( \mathcal { G } ^ { \prime } ) \rVert _ { F } \leq L _ { l } \mathrm { d i s t } ( \mathcal { G } , \mathcal { G } ^ { \prime } )\tag{60}
$$

for all $\mathcal { G } , \mathcal { G } ^ { \prime } \in \mathbb { U }$ . Then

$$
\| { \pmb \sigma } ^ { ( l ) } ( { \mathcal { G } } ) - { \pmb \sigma } ^ { ( l ) } ( { \mathcal { G } } ^ { \prime } ) \| _ { 2 } \leq L _ { l } \operatorname { d i s t } ( { \mathcal { G } } , { \mathcal { G } } ^ { \prime } )\tag{61}
$$

for every $l \in \Lambda$ . Consequently,

$$
\| \mathrm { S y m F P } _ { \Lambda } ( \mathcal { G } ) - \mathrm { S y m F P } _ { \Lambda } ( \mathcal { G } ^ { \prime } ) \| _ { \infty } \leq C _ { \Lambda } \operatorname { d i s t } ( \mathcal { G } , \mathcal { G } ^ { \prime } ) ,\tag{62}
$$

where $C _ { \Lambda } : = \operatorname* { m a x } _ { l \in \Lambda } L _ { l }$

Proof. Mirsky’s inequality gives

$$
\lVert \pmb { \sigma } ^ { ( l ) } ( \mathcal { G } ) - \pmb { \sigma } ^ { ( l ) } ( \mathcal { G } ^ { \prime } ) \rVert _ { 2 } \leq \lVert \tilde { \pmb { V } } ^ { ( l ) } ( \mathcal { G } ) - \tilde { \pmb { V } } ^ { ( l ) } ( \mathcal { G } ^ { \prime } ) \rVert _ { F } ,\tag{63}
$$

which, together with Eq. (60), proves Eq. (61). Taking the maximum over all singular-value coordinates yields Eq. (62). □

Definition 14 (Point-group-conditioned continuous symmetry measure). For a point group ${ \mathfrak { H } } \leq$ $\mathrm { O ( 3 ) }$ with $\mathbb { U } \cap \mathbb { G } ( { \mathfrak { H } } ) \neq \emptyset$ , define

$$
\operatorname { C S M } _ { 5 } ( \mathcal { G } ) : = \operatorname* { i n f } _ { \mathcal { G } ^ { \star } \in \operatorname { U } \cap \mathbb { G } ( 5 ) } \operatorname { d i s t } ( \mathcal { G } , \mathcal { G } ^ { \star } ) , \qquad \mathcal { G } \in \mathbb { U } .\tag{64}
$$

This measure quantifies distance to the H-compatible subset within U, following the nearestsymmetric-configuration construction of § 4.2. By Thm. 13, taking the infimum over this subset gives

$$
\operatorname* { i n f } _ { \mathcal { G } ^ { \star } \in \mathbb { U } \cap \mathbb { G } ( \mathfrak { H } ) } \| \mathrm { S y m F P } _ { \Lambda } ( \mathcal { G } ) - \mathrm { S y m F P } _ { \Lambda } ( \mathcal { G } ^ { \star } ) \| _ { \infty } \leq C _ { \Lambda } \mathrm { C S M } _ { \mathfrak { H } } ( \mathcal { G } ) .\tag{65}
$$

The fixed-subspace constraint further yields a quadratic bound on spectral leakage.

Corollary 15 (Leakage-score control via CSM). Under the assumptions of Thm. 13, consider the default block-energy normalization and nonnegative pooling weights $\beta _ { l }$ with $\begin{array} { r } { \sum _ { l \in \Lambda } \beta _ { l } = 1 } \end{array}$ . For any ${ \mathfrak { H } } \in \mathbb { C }$ satisfying $\mathbb { U } \cap \mathbb { G } ( { \mathfrak { H } } ) \neq \emptyset$ and any $\mathcal { G } \in \mathbb { U }$ ，

$$
\mathrm { L S } _ { \Lambda , 5 } ( \mathcal { G } ) \leq C _ { \Lambda } ^ { \mathrm { L S } } ( \mathcal { G } ) \mathrm { C S M } _ { 5 } ( \mathcal { G } ) ^ { 2 } ,\tag{66}
$$

where

$$
C _ { \Lambda } ^ { \mathrm { L S } } ( \mathcal { G } ) : = \sum _ { l \in \Lambda } \beta _ { l } \frac { L _ { l } ^ { 2 } } { \| \tilde { \pmb { V } } ^ { ( l ) } ( \mathcal { G } ) \| _ { F } ^ { 2 } + \varepsilon _ { \mathrm { n o r m } } } .\tag{67}
$$

Consequently,

$$
\begin{array} { r } { \mathrm { p L S } _ { \Lambda , \mathfrak { H } } ( \mathcal { G } ) \geq - \log _ { 1 0 } \bigl ( C _ { \Lambda } ^ { \mathrm { L S } } ( \mathcal { G } ) \mathrm { C S M } _ { \mathfrak { H } } ( \mathcal { G } ) ^ { 2 } + \varepsilon _ { \log } \bigr ) . } \end{array}\tag{68}
$$

Proof. For any $\mathcal { G } ^ { \star } \in \mathbb { U } \cap \mathbb { G } ( \mathfrak { H } )$ , the fixed-subspace constraint (Eq. (23)) gives $\sigma _ { i } ^ { ( l ) } ( \mathcal { G } ^ { \star } ) = 0$ for $i > m _ { l } ( \mathfrak { H } )$ . Hence, by Eq. (61),

$$
\begin{array} { r l } & { \displaystyle \sum _ { i > m _ { l } ( \mathfrak { H } ) } ( \sigma _ { i } ^ { ( l ) } ( \mathcal { G } ) ) ^ { 2 } \leq \| \pmb { \sigma } ^ { ( l ) } ( \mathcal { G } ) - \pmb { \sigma } ^ { ( l ) } ( \mathcal { G } ^ { \star } ) \| _ { 2 } ^ { 2 } } \\ & { \leq L _ { l } ^ { 2 } \operatorname { d i s t } ( \mathcal { G } , \mathcal { G } ^ { \star } ) ^ { 2 } . } \end{array}\tag{69}
$$

Dividing by the block-energy normalization and pooling with weights $\beta _ { l }$ yields

$$
\mathrm { L S } _ { \Lambda , { \mathfrak { H } } } ( { \mathcal { G } } ) \leq C _ { \Lambda } ^ { \mathrm { L S } } ( { \mathcal { G } } ) \operatorname { d i s t } ( { \mathcal { G } } , { \mathcal { G } } ^ { \star } ) ^ { 2 } .\tag{70}
$$

Taking the infimum over $\mathcal { G } ^ { \star } \in \mathbb { U } \cap \mathbb { G } ( \mathfrak { H } )$ proves Eq. (66). The pLS bound follows from the monotonicity of $- \log _ { 1 0 }$ □

The coeficient $C _ { \Lambda } ^ { \mathrm { L S } } ( { \mathcal { G } } )$ reflects the block Lipschitz constants and energies, with $\varepsilon _ { \mathrm { n o r m } }$ controlling low-energy blocks. Thus, spectral leakage is bounded quadratically by the distance to the targetsymmetric subset.

## B Relation to existing symmetry measures

The energy retained by SymFP has a variational characterization determined by the target fixedsubspace dimension. The same trace objective underlies operation-based retained energy, obtained from the average operation residual in harmonic representation space. We derive these relations to support § 4.6.

## B.1 The operation average in representation space

We derive the retained-energy identity in Eq. (12) from a CSoM-style operation average in harmonic representation space.

Following § 4.2, let $\tilde { \pmb { v } } ( \mathcal { G } ) = ( \tilde { \pmb { v } } ^ { ( l ) } ( \mathcal { G } ) ) _ { l }$ be a normalized harmonic embedding, with $\pmb { \tilde { v } } ^ { ( l ) } ( \mathcal { G } ) \in \mathbb { R } ^ { 2 l + 1 }$ transforming equivariantly:

$$
\tilde { \pmb { v } } ^ { ( l ) } ( \mathfrak { h } \cdot \mathcal { G } ) = \rho ^ { ( l ) } ( \mathfrak { h } ) \tilde { \pmb { v } } ^ { ( l ) } ( \mathcal { G } ) .\tag{71}
$$

For a fixed graph, we suppress the argument $\mathcal { G }$ and assume $\begin{array} { r } { \sum _ { l } \| \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } = 1 } \end{array}$ , with sums taken over the retained degrees. For an oriented target group $\mathfrak { r } \mathfrak { H } \mathfrak { r } ^ { - 1 }$ , define the average representation residual

$$
R _ { \mathfrak { H } } ( \mathcal { G } ; \mathfrak { r } ) : = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \sum _ { l } \| \tilde { \pmb { v } } ^ { ( l ) } - \rho ^ { ( l ) } ( \mathfrak { r } \mathfrak { h } \mathfrak { r } ^ { - 1 } ) \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } .\tag{72}
$$

The group average defines the orthogonal projector onto the rotated fixed subspace:

$$
P _ { 5 } ^ { ( l ) } ( \mathfrak { r } ) : = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { r } \mathfrak { h } \mathfrak { r } ^ { - 1 } ) .\tag{73}
$$

Expanding the squared norms and using the orthogonality of the representation matrices yields

$$
R _ { 5 } ( \mathcal { G } ; \mathbf { r } ) = 2 \sum _ { l } ( \| \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } - ( \tilde { \pmb { v } } ^ { ( l ) } ) ^ { \top } \pmb { P } _ { 5 } ^ { ( l ) } ( \mathbf { r } ) \tilde { \pmb { v } } ^ { ( l ) } ) = 2 \sum _ { l } \| ( \pmb { I } _ { 2 l + 1 } - \pmb { P } _ { 5 } ^ { ( l ) } ( \mathbf { r } ) ) \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } .\tag{74}
$$

Define the retained energy at degree l by

$$
\alpha _ { \mathfrak { H } } ^ { ( l ) } ( \mathcal { G } ; \mathfrak { r } ) : = \Vert P _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } ) \tilde { \pmb { v } } ^ { ( l ) } \Vert _ { 2 } ^ { 2 } = ( \tilde { \pmb { v } } ^ { ( l ) } ) ^ { \top } P _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } ) \tilde { \pmb { v } } ^ { ( l ) } ,\tag{75}
$$

and let $\begin{array} { r } { \alpha _ { \mathfrak H } ( \mathcal G ; \mathfrak c ) : = \sum _ { l } \alpha _ { \mathfrak H } ^ { ( l ) } ( \mathcal G ; \mathfrak c ) } \end{array}$ . By normalization,

$$
R _ { 5 } ( \mathcal { G } ; \mathfrak { r } ) = 2 \sum _ { l } ( \| \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } - \alpha _ { 5 } ^ { ( l ) } ( \mathcal { G } ; \mathfrak { r } ) ) = 2 ( 1 - \alpha _ { 5 } ( \mathcal { G } ; \mathfrak { r } ) ) .\tag{76}
$$

Thus, minimizing the operation-averaged residual in harmonic representation space is equivalent to maximizing retained energy in the target fixed subspaces.

## B.2 Spectral and orientation-conditioned retained energy

For the degree-l equivariant representation, let $\pmb { C } ^ { ( l ) } : = \tilde { \pmb { V } } ^ { ( l ) } ( \tilde { \pmb { V } } ^ { ( l ) } ) ^ { \top }$ and $k : = m _ { l } ( { \mathfrak { H } } )$ . SymFP retains the spectral energy $\begin{array} { r } { \mathcal A _ { \mathfrak H } ^ { ( l ) , \mathrm { s y m f p } } = \sum _ { i = 1 } ^ { k } ( \sigma _ { i } ^ { ( l ) } ) ^ { 2 } } \end{array}$ , whose variational characterization in Eq. (35) maximizes $\mathrm { t r } ( P C ^ { ( l ) } )$ over all rank-k orthogonal projectors. Operation-based retained energy optimizes the same trace objective over projectors induced by spatial orientations of the target group.

Reduction of the operation average to a retained energy. Let $P _ { 5 } ^ { ( l ) }$ be the orthogonal projector onto the fixed subspace $\mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ ). By Eq. (20), under a proper spatial rotation $\mathfrak { r } \in \mathrm { S O } ( 3 )$ the same target group in the rotated frame has projector

$$
\pmb { P } _ { 5 } ^ { ( l ) } ( \mathfrak { r } ) : = \rho ^ { ( l ) } ( \mathfrak { r } ) \pmb { P } _ { 5 } ^ { ( l ) } \rho ^ { ( l ) } ( \mathfrak { r } ) ^ { \top } = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \rho ^ { ( l ) } ( \mathfrak { r } \mathfrak { h } \mathfrak { r } ^ { - 1 } ) .\tag{77}
$$

Consider the average operation residual at degree l and orientation $\mathfrak { r } ,$

$$
R _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } ) : = \frac { 1 } { | \mathfrak { H } | } \sum _ { \mathfrak { h } \in \mathfrak { H } } \| \tilde { V } ^ { ( l ) } - \rho ^ { ( l ) } ( \mathfrak { r } \mathfrak { h } \mathfrak { r } ^ { - 1 } ) \tilde { V } ^ { ( l ) } \| _ { F } ^ { 2 } .\tag{78}
$$

Using the orthogonality of each matrix $\rho ^ { ( l ) } ( { \mathfrak { r } } \mathfrak { h } { \mathfrak { r } } ^ { - 1 } )$ , the group-average identity in Eq. (77), and the fact that $P _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } )$ is an orthogonal projector, we obtain

$$
R _ { 5 } ^ { ( l ) } ( \mathbf { r } ) = 2 \| ( I _ { 2 l + 1 } - P _ { 5 } ^ { ( l ) } ( \mathbf { r } ) ) \tilde { \mathbf { V } } ^ { ( l ) } \| _ { F } ^ { 2 } = 2 ( \| \tilde { \mathbf { V } } ^ { ( l ) } \| _ { F } ^ { 2 } - \mathrm { t r } ( P _ { 5 } ^ { ( l ) } ( \mathbf { r } ) C ^ { ( l ) } ) ) .\tag{79}
$$

At each orientation, the residual is twice the diference between the total energy at degree l and the retained energy,

$$
\begin{array} { r } { \mathcal A _ { \mathfrak H } ^ { ( l ) } ( \mathfrak r ) : = \| P _ { \mathfrak H } ^ { ( l ) } ( \mathfrak r ) \tilde { V } ^ { ( l ) } \| _ { F } ^ { 2 } = \mathrm { t r } ( P _ { \mathfrak H } ^ { ( l ) } ( \mathfrak r ) C ^ { ( l ) } ) , } \end{array}\tag{80}
$$

which extends the relation between operation averages and retained energy to all channels of the equivariant representation. For Engel’s single harmonic channel, $C = 1 , \tilde { V } ^ { ( l ) } = \tilde { v } ^ { ( l ) }$ and $C ^ { ( l ) } = \tilde { \pmb { v } } ^ { ( l ) } ( \tilde { \pmb { v } } ^ { ( l ) } ) ^ { \top }$ , so Eqs. (78) and (80) reduce to the single-channel operation residual and retained energy of the previous section. The latter is $\alpha _ { \mathfrak { H } } ^ { ( l ) } ( \mathcal { G } ; \mathfrak { r } ) ;$ ; summing over the retained degrees at a common orientation r gives the pooled retained energy <sup>4</sup>. Maximizing the normalized score over this shared orientation gives Eq. (15)<sup>5</sup>.

To relate this projection-based retained energy to $\mathrm { S y m F P ^ { \prime } s }$ spectral variational form degree by degree, we maximize over spatial orientations separately at each degree, giving the quantity $\begin{array} { r } { \check { \mathcal { A } } _ { \mathfrak { H } } ^ { ( l ) , \mathrm { o p } } : = \operatorname* { m a x } _ { \mathfrak { r } \in \mathrm { S O } ( 3 ) } \mathcal { A } _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } ) } \end{array}$ used in $\ S \ 4 . 6$

Feasible-set inclusion. SymFP optimizes over all rank-k orthogonal projectors, whereas the operation-based formulation restricts the search to projectors onto $\rho ^ { ( l ) } ( \mathfrak { r } ) \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ for $\mathfrak { r } \in \mathrm { S O } ( 3 )$ Every such projector has rank k, so the operation-based feasible set is contained in that of Eq. (35). Maximizing over r yields Eq. (37). The corresponding leakage inequality follows by subtracting both sides from $\| \tilde { V } ^ { ( l ) } \| _ { F } ^ { 2 }$ and normalizing.

Conditions for equality. Equality holds if a top-k left-singular subspace of $\tilde { V } ^ { ( l ) }$ coincides with $\rho ^ { ( l ) } ( \mathfrak { r } ) \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ for some $\mathfrak { r } \in \mathrm { S O } ( 3 )$ . In particular, if $\mathfrak { r } _ { 0 } ^ { - 1 } \cdot \mathcal { G } \in \mathbb { G } ( \mathfrak { H } )$ , Thm. 3 ensures that the column space of $\tilde { V } ^ { ( l ) }$ lies in $\rho ^ { ( l ) } ( \mathfrak { r } _ { 0 } ) \mathbb { F } ^ { ( l ) } ( \mathfrak { H } )$ , so both formulations retain the full energy $\| \tilde { V } ^ { ( l ) } \| _ { F } ^ { 2 }$ . Equality also holds at $l = 1$ , where rotations act transitively on subspaces of each admissible dimension in $\mathbb { R } ^ { 3 }$

Symmetry-equivalent orientations yield identical fixed-subspace projectors and retained energies. SymFP obtains the maximum over all rank-k projectors directly from the singular values, without selecting an orientation.

Subgroup ordering of approximate-symmetry quantities. For ${ \mathfrak { H } } _ { 1 } \leq { \mathfrak { H } } _ { 2 }$ and a fixed orientation r, the inclusion $\rho ^ { ( l ) } ( \mathfrak { t } ) \mathbb { F } ^ { ( l ) } ( \mathfrak { H } _ { 2 } ) \subseteq \rho ^ { ( l ) } ( \mathfrak { t } ) \mathbb { F } ^ { ( l ) } ( \mathfrak { H } _ { 1 } )$ implies monotonicity of the raw retained energies along subgroup chains. SymFP captures this ordering through the fixed-subspace dimensions, yielding the leakage monotonicity in Prop. 6 under common normalization and pooling. In contrast, classical CSoM normalizes sums of deviations over target-dependent operation sets, while Engel score involves the target-dependent factor $\beta _ { 5 }$ . Consequently, neither normalized score is guaranteed to preserve subgroup ordering, even at a fixed orientation.

## C Datasets and evaluation settings

## C.1 Shared representation and scoring settings

Unless otherwise specified, evaluations use natural-parity degrees ${ \Lambda } _ { 6 } = \{ 0 , \ldots , 6 \}$ , normalization regularizer $\varepsilon _ { \mathrm { n o r m } } = 1 0 ^ { - 1 0 }$ , and logarithmic regularizer $\varepsilon _ { \mathrm { l o g } } = 1 0 ^ { - 1 0 }$ (Eqs. (28) and (30)). Learnable GNNs use a fixed random initialization per configuration, except under explicit training or repeatedinitialization protocols. Leakage scores are pooled uniformly across degrees (§ 4.4). The p97, haemagglutinin, GluK2, and capping-water analyses use $\Lambda _ { 6 } ;$ torsion scans use $\{ 0 , 2 , 4 , 6 \}$ , and E18 uses degree six alone. Crystal-field and controlled-deformation settings are specified separately below.

PF settings. PF uses hybrid blocks with 16 Gaussian radial bases per branch (Algorithm 1). Their centres are uniformly spaced on [0, 1], with width $\sigma = 1 / 1 6$ . Node and edge attributes are atomic numbers and lengths, respectively, giving $d _ { v } = d _ { e } = 1$ . Single-species deformation inputs and E18 Cα inputs use $h _ { i } = 6$ . Attributes remain unscaled; $Z _ { \nu }$ and $Z _ { \mathcal { E } }$ normalize only the radial bases. Node, edge, and hybrid blocks have 16, 48, and 64 channels, respectively.

Radial denominators are stabilized with $\varepsilon _ { 0 } = 1 0 ^ { - 1 0 }$ . Nodes satisfying $\lVert \vec { \pmb { x } } _ { i c } \rVert \leq \delta ( Z _ { \mathcal { V } } + \varepsilon _ { 0 } )$ , with $\delta = 1 0 ^ { - 5 }$ , are treated as coincident with the centroid: their spherical-harmonic components vanish for $l > 0$ , while the scalar component is retained. The same convention applies to coincident edge endpoints.

## C.2 Synthetic point-group benchmark

We generate 100 finite point clouds for each of the 32 crystallographic point groups, with $8 0 / 1 0 / 1 0$ training/validation/test splits within each group. Each cloud combines orbits of generic seed points, using fewer orbits for higher-order groups. We retain only clouds whose Euclidean point group is exactly the target, excluding accidental higher symmetries.

We compare PF, HEGNN, TFN, MACE, and Equiformer. At $\Lambda _ { 6 } .$ , learnable GNNs use 16 channels per degree; at $\Lambda _ { 9 }$ , they use 24 channels per degree at random initialization, while PF uses 24 radial bases per branch. HEGNN employs a multi-channel extension of the original architecture<sup>6</sup> developed in this work.

Structures are classified by nearest rank-template matching under the $\ell _ { 1 }$ rule (Eq. (27)), with ties broken by the ordering in Supplementary Table $\mathrm { A . 1 }$ . Identification accuracy measures correct group assignments, whereas rank-template match rate requires agreement with the true group’s template at every retained degree. Shared templates count as matches for either group but permit only one identification label. At $\Lambda _ { 6 }$ , two indistinguishable pairs cap identification accuracy at 30/32 on the balanced benchmark.

Training regimes on the synthetic benchmark. The four learnable GNNs are evaluated at random initialization, after coordinate-denoising pretraining, and after supervised classification training. Both training regimes use the training split. For denoising, isotropic Gaussian noise is added to the coordinates, and the networks learn to predict displacements restoring the clean structures. The noise standard deviation is sampled from [0.02, 0.10] times the median nearest-neighbour distance during training and fixed at 0.05 times that distance for validation. Checkpoints are selected by minimum validation coordinate MAE.

For supervised training, pooled representations of clean structures are used to predict the 32 point-group labels. Models are trained with cross-entropy loss, with checkpoints selected by maximum validation accuracy. We perform one run per architecture in each training regime, keeping the architectures fixed across regimes.

Rank threshold. Numerical ranks use $\tau _ { \mathrm { a b s } } ~ = ~ 0$ and a relative tolerance scaled by the leading degree-zero singular value $\sigma _ { 1 } ^ { ( 0 ) } \left( \mathrm { E q . ~ ( 2 4 ) } \right)$ , providing a common threshold across degrees, including those with $m _ { l } ( { \mathfrak { H } } ) = 0$ . We calibrate the tolerance on clean validation structures by first maximizing the minimum identification accuracy across configurations, then the mean rank-template match rate. Across the thirteen $\Lambda _ { 6 }$ configurations, this yields $\tau _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 5 }$ (Supplementary Table C.7), which is held fixed for testing (Supplementary Tables C.2 and C.5).

Supplementary Table C.2. Discrete point-group identification across training regimes. Evaluations use natural-parity degrees $l = 0 , \ldots , 6$ and relative rank tolerance $\tau _ { \mathrm { r e l } } = 3 \times 1 0 ^ { - 5 }$
<table><tr><td>Training regime</td><td>GNN</td><td>Identification accuracy (%)</td><td>Template match rate (%)</td></tr><tr><td></td><td>PF</td><td>93.75</td><td>97.19</td></tr><tr><td rowspan="4">Random init.</td><td> $\mathrm { H E G N N } ^ { 6 }$ </td><td>93.75</td><td>92.50</td></tr><tr><td> $\mathrm { T F N } ^ { 7 }$ </td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathrm { M A C E ^ { 8 } }$ </td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathrm { E q u i f o r m e r } ^ { 9 }$ </td><td>93.75</td><td>100.00</td></tr><tr><td rowspan="4">Pretrained</td><td>HEGNN⁶</td><td>93.75</td><td>91.88</td></tr><tr><td> $\mathrm { T F N } ^ { 7 }$ </td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathrm { M A C E ^ { 8 } }$ </td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathrm { E q u i f o r m e r } ^ { 9 }$ </td><td>93.75</td><td>100.00</td></tr><tr><td rowspan="4">Supervised</td><td> $\mathrm { H E G N N } ^ { 6 }$ </td><td>93.44</td><td>94.38</td></tr><tr><td> $\mathrm { T F N } ^ { 7 }$ </td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathrm { M A C E ^ { 8 } }$ </td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathrm { E q u i f o r m e r } ^ { 9 }$ </td><td>93.75</td><td>100.00</td></tr></table>

Supplementary Table C.3. Discrete point-group identification using degrees $l = 0 , \ldots , 9 .$ Evaluations use natural-parity representations and validation-selected relative rank tolerance $\tau _ { \mathrm { r e l } } = 1 0 ^ { - 6 }$ (Supplementary Note C.2).

<table><tr><td>GNN</td><td>Identification accuracy (%)</td><td>Template match rate (%)</td></tr><tr><td>PF</td><td>99.38</td><td>93.75</td></tr><tr><td>HEGNN⁶</td><td>99.06</td><td>92.19</td></tr><tr><td>TFN7</td><td>100.00</td><td>100.00</td></tr><tr><td> $\mathrm { M A C E ^ { 8 } }$ </td><td>100.00</td><td>100.00</td></tr><tr><td>Equiformer9</td><td>99.38</td><td>99.38</td></tr></table>

For $\Lambda _ { 9 } .$ , we apply the same criterion across five GNNs, breaking any remaining ties in favour of the smaller tolerance. The search covers $\{ 1 , 3 \} \times 1 0 ^ { j }$ for $j = - 8 , \ldots , - 4$ , together with $1 0 ^ { - 3 }$ Calibration on 320 validation structures selects $\tau _ { \mathrm { r e l } } = 1 0 ^ { - 6 }$ , which is then applied to the 320 test structures. Each learnable GNN retains identical initialized weights across validation and testing (Supplementary Table C.3).

Random-initialization repeats. We evaluate all four learnable GNNs across three independent random initializations for each degree set (Supplementary Table C.4), holding data splits and architectures fixed. The validation-calibrated tolerances remain $3 \times 1 0 ^ { - 5 }$ for $\Lambda _ { 6 }$ and $1 0 ^ { - 6 }$ for $\Lambda _ { 9 }$ Results in Fig. 2 and Supplementary Tables C.2 and C.3 use a consistent reference initialization for each architecture.

PF ablation. The PF ablation uses $\Lambda _ { 6 }$ with relative rank tolerance $3 \times 1 0 ^ { - 5 }$ (Fig. 2c, Supplementary Table C.6). We compare node, edge, and hybrid branches using $N = 1 6$ radial bases with width $\sigma = 1 / 1 6$ . For the hybrid block, we then vary the basis count over $N \in \{ 8 , 1 6 , 3 2 \}$ with $\sigma = 1 / N$ followed by a width sweep at fixed $N = 1 6$ over $\sigma \in \{ \Delta / 2 , \Delta , 2 \Delta \}$ , where $\Delta = 1 / ( N - 1 )$ denotes

Supplementary Table C.4. Discrete point-group identification across random initializations.
<table><tr><td rowspan="2" colspan="2">L GNN</td><td colspan="4">Identification accuracy (%)</td><td colspan="4">Template match rate (%)</td></tr><tr><td>Seed 42</td><td>Seed 43</td><td>Seed 44</td><td> $\mathrm { M e a n } \pm \mathrm { s . d . }$ </td><td>Seed 42</td><td>Seed 43</td><td>Seed 44</td><td> ${ \mathrm { M e a n } } \pm { \mathrm { s . d . } }$ </td></tr><tr><td rowspan="4">6</td><td>HEGNN</td><td>93.75</td><td>93.44</td><td>93.13</td><td> $9 3 . 4 4 \pm 0 . 3 1$ </td><td>92.50</td><td>92.50</td><td>91.25</td><td> $9 2 . 0 8 \pm 0 . 7 2$ </td></tr><tr><td>TFN</td><td>93.75</td><td>93.75</td><td>93.75</td><td> $9 3 . 7 5 \pm 0 . 0 0$ </td><td>100.00</td><td>100.00</td><td>100.00</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>MACE</td><td>93.75</td><td>93.75</td><td>93.75</td><td> $9 3 . 7 5 \pm 0 . 0 0$ </td><td>100.00</td><td>100.00</td><td>100.00</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Equiformer</td><td>93.75</td><td>93.75</td><td>93.75</td><td> $9 3 . 7 5 \pm 0 . 0 0$ </td><td>100.00</td><td>100.00</td><td>100.00</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td rowspan="4">9</td><td>HEGNN</td><td>99.06</td><td>99.06</td><td>99.06</td><td> $9 9 . 0 6 \pm 0 . 0 0$ </td><td>92.19</td><td>93.13</td><td>92.81</td><td> $9 2 . 7 1 \pm 0 . 4 8$ </td></tr><tr><td>TFN</td><td>100.00</td><td>100.00</td><td>100.00</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td>100.00</td><td>100.00</td><td>100.00</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>MACE</td><td>100.00</td><td>100.00</td><td>100.00</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td>100.00</td><td>100.00</td><td>100.00</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Equiformer</td><td>99.38</td><td>99.38</td><td>99.38</td><td> $9 9 . 3 8 \pm 0 . 0 0$ </td><td>99.38</td><td>99.06</td><td>99.06</td><td> $9 9 . 1 7 \pm 0 . 1 8$ </td></tr></table>

the radial-centre spacing on [0, 1].

## C.3 Controlled deformation benchmarks

We evaluate PF on single-species point clouds, isolating geometric symmetry. Single-step deformations begin with a unit octahedron or an eclipsed triangular prism. Stretching the octahedron along z induces $\mathrm { O } _ { \mathrm { h } }  \mathrm { D } _ { 4 \mathrm { h } }$ , while straining along [111] induces $\mathrm { O } _ { \mathrm { h } }  \mathrm { D } _ { \mathrm { 3 d } }$ . Twisting the prism’s upper triangle breaks mirror symmetry, yielding $\mathrm { D } _ { 3 \mathrm { h } } \to \mathrm { D } _ { 3 }$ . Each deformation uses 41 quadratically spaced amplitudes, reaching 0.30 times the base radius for strains and $3 0 ^ { \circ }$ for twisting.

Progressive symmetry breaking applies successive linear maps to the octahedron along $\mathrm { O _ { h } } $ $\mathrm { D } _ { 4 \mathrm { h } }  \mathrm { D } _ { 2 \mathrm { h } }  \mathrm { C } _ { 2 \mathrm { h } }  \mathrm { C } _ { \mathrm { i } }$ . The trajectory comprises 61 frames, with 15 steps per segment and exact subgroup symmetries at the boundaries. All maps preserve inversion. Point-group analysis independently verifies each frame’s symmetry.

In Fig. 3a, singular values exceeding $1 0 ^ { - 4 }$ are considered active. We display the smallest initially active, largest initially inactive, and all newly active singular values. In Fig. 3b, the total spectral tail energy relative to H is $\begin{array} { r } { \sum _ { l \in \Lambda _ { 6 } } \sum _ { i > m _ { l } ( \mathfrak { H } ) } ( \bar { \sigma _ { i } ^ { ( l ) } } ) ^ { 2 } } \end{array}$

For matched-amplitude comparisons, we construct a 30-point ring with exact $\mathrm { C _ { 6 } }$ symmetry from six copies of a chiral five-point subunit, arranged at $6 0 ^ { \circ }$ intervals about z with radius 2.0. Four displacement fields probe distinct symmetry-breaking modes. Axial staircase assigns subunit k a displacement along z proportional to $k .$ , mixing azimuthal orders and reducing the symmetry to $\mathrm { C _ { 1 } }$ . In-plane stretch applies $( x , - y , 0 )$ to each point $( x , y , z )$ , producing azimuthal order $m = 2$ and symmetry $\mathrm { C _ { 2 } }$ . In-plane radial displacement alternates outward and inward shifts between subunits, producing $m = 3$ and $\mathrm { C _ { 3 } }$ . Axial pucker alternates subunit displacements along $\pm z$ likewise producing $m = 3$ and $\mathrm { C _ { 3 } }$

To match deformation amplitudes, we project out the six rigid-body modes (three translations and three infinitesimal rotations) and normalize each displacement field to unit per-atom RMSD. Scaling by $\alpha$ then produces RMSD α relative to the initial structure. Each mode uses 21 uniformly spaced amplitudes from 0 to 0.30, giving 84 frames. Fourier analysis across subunits uses corotating displacement vectors, with lowest nonzero orders $| m | = 1 , 2 , 3 , 3$ for staircase, stretch, inplane radial, and axial pucker, respectively. Point-group analysis verifies the initial and deformed structures.

Matched-amplitude evaluations use fully connected graphs over the 30 points with target $\mathrm { C } _ { 6 } .$ We measure degree-resolved relative tail energy (Eq. (28)), RMSD from the initial structure, and $\mathrm { C _ { 6 } }$ self-match RMSD. The latter compares each frame with its $6 0 ^ { \circ }$ rotation about the initial ring axis under the subunit correspondence $k \mapsto k + 1$ . Spectral onset is the smallest degree l at which relative tail energy exceeds $1 0 ^ { - 6 }$

Supplementary Table C.5. Sensitivity to rank tolerance. Group and Match denote point-group identification accuracy and rank-template match rate (%). Bold marks the validation-selected tolerance.
<table><tr><td rowspan="2"> $\tau _ { \mathrm { r e l } }$ </td><td colspan="2">PF</td><td colspan="2">HEGNN⁶</td><td colspan="2"> $\mathrm { T F N } ^ { 7 }$ </td><td colspan="2"> $\mathrm { M A C E ^ { 8 } }$ </td><td colspan="2">Equiformer9</td></tr><tr><td>Group</td><td>Match</td><td>Group</td><td>Match</td><td>Group</td><td>Match</td><td>Group</td><td>Match</td><td>Group</td><td>Match</td></tr><tr><td colspan="9">PF and random init.</td><td></td></tr><tr><td> $1 \times 1 0 ^ { - 7 }$ </td><td>87.50</td><td>77.81</td><td>93.75</td><td>98.75</td><td>93.75</td><td>99.69</td><td>93.75</td><td>100.00</td><td>28.44</td><td>15.62</td></tr><tr><td> $3 \times 1 0 ^ { - 7 }$ </td><td>93.75</td><td>93.44</td><td>93.75</td><td>99.06</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>86.56</td><td>84.38</td></tr><tr><td> $1 \times 1 0 ^ { - 6 }$ </td><td>93.75</td><td>97.19</td><td>93.75</td><td>97.81</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>99.69</td></tr><tr><td> $3 \times 1 0 ^ { - 6 }$ </td><td>93.75</td><td>97.19</td><td>93.75</td><td>96.88</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>99.69</td></tr><tr><td> $1 \times 1 0 ^ { - 5 }$ </td><td>93.75</td><td>97.19</td><td>93.75</td><td>94.06</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathbf { 3 \times 1 0 ^ { - 5 } }$ </td><td>93.75</td><td>97.19</td><td>93.75</td><td>92.50</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $1 \times 1 0 ^ { - 4 }$ </td><td>93.75</td><td>97.19</td><td>90.62</td><td>87.19</td><td>93.75</td><td>99.06</td><td>93.75</td><td>98.12</td><td>93.75</td><td>100.00</td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td>93.75</td><td>96.88</td><td>81.25</td><td>70.94</td><td>93.12</td><td>95.31</td><td>93.44</td><td>94.06</td><td>93.75</td><td>100.00</td></tr><tr><td> $1 \times 1 0 ^ { - 3 }$ </td><td>93.75</td><td>95.62</td><td>46.88</td><td>36.25</td><td>91.88</td><td>89.38</td><td>89.69</td><td>84.06</td><td>93.75</td><td>100.00</td></tr><tr><td colspan="9">Pretrained</td><td></td></tr><tr><td> $1 \times 1 0 ^ { - 7 }$ </td><td></td><td></td><td>93.75</td><td>98.75</td><td>92.81</td><td>88.75</td><td>24.06</td><td>15.31</td><td>25.31</td><td>15.31</td></tr><tr><td> $3 \times 1 0 ^ { - 7 }$ </td><td></td><td></td><td>93.75</td><td>99.06</td><td>93.75</td><td>99.69</td><td>84.38</td><td>78.44</td><td>88.44</td><td>90.00</td></tr><tr><td> $1 \times 1 0 ^ { - 6 }$ </td><td></td><td></td><td>93.75</td><td>97.81</td><td>93.75</td><td>100.00</td><td>93.75</td><td>99.69</td><td>93.75</td><td>99.38</td></tr><tr><td> $3 \times 1 0 ^ { - 6 }$ </td><td></td><td></td><td>93.75</td><td>96.56</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>99.69</td></tr><tr><td> $1 \times 1 0 ^ { - 5 }$ </td><td></td><td></td><td>93.75</td><td>93.12</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathbf { 3 \times 1 0 ^ { - 5 } }$ </td><td></td><td></td><td>93.75</td><td>91.88</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $1 \times 1 0 ^ { - 4 }$ </td><td></td><td></td><td>89.69</td><td>85.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td></td><td></td><td>78.75</td><td>64.38</td><td>93.75</td><td>98.75</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $1 \times 1 0 ^ { - 3 }$ </td><td></td><td></td><td>39.69</td><td>28.75</td><td>92.50</td><td>92.19</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td colspan="9">Supervised</td><td></td><td></td></tr><tr><td> $1 \times 1 0 ^ { - 7 }$ </td><td></td><td></td><td>3.12</td><td>3.12</td><td>92.81</td><td>97.19</td><td>52.81</td><td>39.69</td><td>38.44</td><td>21.56</td></tr><tr><td> $3 \times 1 0 ^ { - 7 }$ </td><td></td><td></td><td>4.69</td><td>2.81</td><td>93.44</td><td>99.06</td><td>91.88</td><td>97.19</td><td>89.69</td><td>91.88</td></tr><tr><td> $1 \times 1 0 ^ { - 6 }$ </td><td></td><td></td><td>39.69</td><td>20.00</td><td>93.75</td><td>99.69</td><td>93.44</td><td>99.38</td><td>93.12</td><td>98.75</td></tr><tr><td> $3 \times 1 0 ^ { - 6 }$ </td><td></td><td></td><td>86.88</td><td>82.19</td><td>93.75</td><td>99.69</td><td>93.75</td><td>99.69</td><td>93.75</td><td>99.69</td></tr><tr><td> $1 \times 1 0 ^ { - 5 }$ </td><td></td><td></td><td>92.81</td><td>93.75</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $\mathbf { 3 \times 1 0 ^ { - 5 } }$ </td><td></td><td></td><td>93.44</td><td>94.38</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $1 \times 1 0 ^ { - 4 }$ </td><td></td><td></td><td>92.81</td><td>91.88</td><td>93.75</td><td>99.69</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td></td><td></td><td>89.06</td><td>86.88</td><td>93.75</td><td>96.88</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr><tr><td> $1 \times 1 0 ^ { - 3 }$ </td><td></td><td></td><td>83.12</td><td>77.81</td><td>92.50</td><td>92.19</td><td>93.75</td><td>100.00</td><td>93.75</td><td>100.00</td></tr></table>

## C.4 Biaryl torsion benchmark

The torsion benchmark considers nine candidate point groups: $\mathrm { C _ { s } , C _ { 2 } , C _ { 2 v } , D _ { 2 } , D _ { 2 d } , D _ { 2 h } , C _ { i } , C _ { 2 h } }$ and $\mathrm { S _ { 4 } }$ . PF represents each conformation as a fully connected graph over all atoms. We analyse all 24 conformations per scan using pymatgen with tolerance 0.1, taking the distinct nontrivial point groups encountered as candidate targets. Of the 87 scans, 86 yield 96 molecule–target pairs.

Supplementary Table C.6. PF ablation on synthetic structures. N is the number of radial bases per branch, σ their Gaussian width, and $\Delta = 1 / ( N - 1 )$ their centre spacing on [0, 1]. Bold marks the default hybrid configuration.
<table><tr><td>Variant</td><td>Branch</td><td>N</td><td>σ</td><td>Identification accuracy (%)</td><td>Template match rate (%)</td></tr><tr><td>node default</td><td>node</td><td>16</td><td>0.0625</td><td>57.50</td><td>52.81</td></tr><tr><td>edge default</td><td>edge</td><td>16</td><td>0.0625</td><td>92.81</td><td>94.69</td></tr><tr><td>hybrid default</td><td>hybrid</td><td>16</td><td>0.0625</td><td>93.75</td><td>97.19</td></tr><tr><td>hybrid small N</td><td>hybrid</td><td>8</td><td>0.1250</td><td>93.75</td><td>96.25</td></tr><tr><td>hybrid large N</td><td>hybrid</td><td>32</td><td>0.03125</td><td>93.75</td><td>97.19</td></tr><tr><td>hybrid  $\sigma = \Delta / 2$ </td><td>hybrid</td><td>16</td><td>0.0333</td><td>93.75</td><td>96.88</td></tr><tr><td>hybrid  $\sigma = \Delta$ </td><td>hybrid</td><td>16</td><td>0.0667</td><td>93.75</td><td>97.19</td></tr><tr><td>hybrid  $\sigma = 2 \Delta$ </td><td>hybrid</td><td>16</td><td>0.1333</td><td>93.75</td><td>97.19</td></tr></table>

Supplementary Table C.7. Validation-based rank-tolerance selection. Group and Match denote point-group identification accuracy and rank-template match rate (%); Worst and Mean are taken across GNN and training-regime configurations.

<table><tr><td> $\tau _ { \mathrm { r e l } }$ </td><td>Worst Group (%)</td><td>Mean Group (%)</td><td>Mean Match (%)</td></tr><tr><td> $1 0 ^ { - 8 }$ </td><td>3.12</td><td>30.99</td><td>18.82</td></tr><tr><td> $3 \times 1 0 ^ { - 8 }$ </td><td>3.12</td><td>44.52</td><td>37.14</td></tr><tr><td> $1 0 ^ { - 7 }$ </td><td>3.12</td><td>62.64</td><td>59.40</td></tr><tr><td> $3 \times 1 0 ^ { - 7 }$ </td><td>4.06</td><td>84.47</td><td>87.40</td></tr><tr><td> $1 0 ^ { - 6 }$ </td><td>41.25</td><td>89.38</td><td>92.88</td></tr><tr><td> $3 \times 1 0 ^ { - 6 }$ </td><td>90.00</td><td>93.27</td><td>98.08</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>93.12</td><td>93.68</td><td>98.58</td></tr><tr><td> $\mathbf { 3 \times 1 0 ^ { - 5 } }$ </td><td>93.44</td><td>93.70</td><td>98.20</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>90.94</td><td>93.32</td><td>96.85</td></tr><tr><td> $3 \times 1 0 ^ { - 4 }$ </td><td>76.25</td><td>91.06</td><td>92.19</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>35.94</td><td>84.33</td><td>84.40</td></tr></table>

Excluding two pairs whose target symmetry is present throughout the scan leaves 94 pairs from 86 molecules.

For each conformation–target pair, CSoM minimizes the mean coordinate deviation over nonidentity group operations with uniform atom weights. We sample 10 axes and 2 angles, refine all 20 initial orientations using Powell optimization, and retain the minimum deviation. Engel maximizes retained energy through a multistart orientation search that includes the CSoM-optimal orientation (Supplementary Note C.4.2).

## C.4.1 Energy-extremum retrieval

All methods share the same conformations, target groups, and energy labels. SymFP ranks conformations by even-degree pLS, Engel by its orientation-maximized score, and CSoM by negated deviation. On the periodic $1 5 ^ { \circ }$ torsion grid, strict minima and maxima are conformations whose energies are respectively lower or higher than both neighbours.

For each molecule–target pair and selection budget $k = 0 , \ldots , 2 4$ , we retrieve the k highestranked conformations. Recall is the fraction of energy extrema retrieved, with exact score ties resolved by their expected hit count under uniform tie-breaking. Energy labels determine the extrema but not the rankings. Trapezoidal integration of recall against $k / 2 4$ gives the recall– budget AUC, with random-ranking expectation 0.5.

We average AUCs over targets within each molecule and then equally over molecules, separately for minima, maxima, and paired method diferences. Percentile 95% confidence intervals use 5,000 molecule-level bootstrap resamples, paired for method comparisons and without multiplecomparison adjustment. In paired-diference displays, per-molecule AUC diferences above $1 0 ^ { - 1 2 }$ count as wins, those below $- 1 0 ^ { - 1 2 }$ as losses, and the remainder as numerical ties. The example in Fig. 4b uses biphenyl record 21272387 with target $\mathrm { D _ { 2 h } }$

For Extended Data Table 1, extrema are stratified by energy type and target-symmetry presence, defined by whether the assigned point group contains the target, including equality. Rankings retain all 24 conformations, while recall counts only extrema in the specified class. Pairs without eligible extrema are excluded, and results are averaged over eligible targets within molecules and then equally over molecules.

Sensitivity to degree selection. For centrosymmetric targets, odd natural-parity degrees have trivial fixed subspaces, causing degree-resolved LS to saturate near unity when block energies substantially exceed the regularization scale. We therefore use a common even-degree set in the main analysis. This makes $\mathrm { D _ { 2 } }$ and $\mathrm { D _ { 2 h } }$ indistinguishable by SymFP, as their even-degree fixedsubspace dimensions coincide. As a sensitivity check, pooling over $\Lambda _ { 6 }$ with all other settings unchanged yields a mean recall–budget AUC of 0.771, compared with 0.794 for even-degree pooling.

## C.4.2 Engel orientation search

We evaluate Engel score (Eq. (15)) by minimizing the pooled projection residual over a common spatial rotation. For bond-direction coeficients $\tilde { \mathbf { { \mathfrak { v } } } } ^ { \bar { ( l ) } }$ at degrees $l = 1 , \ldots , 6$ , the objective is

$$
L _ { \mathfrak { H } } ^ { \mathrm { E } } ( \mathfrak { r } ) = \frac { \sum _ { l = 1 } ^ { 6 } \| \big ( \pmb { I } - \pmb { P } _ { \mathfrak { H } } ^ { ( l ) } ( \mathfrak { r } ) \big ) \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } } { \sum _ { l = 1 } ^ { 6 } \| \tilde { \pmb { v } } ^ { ( l ) } \| _ { 2 } ^ { 2 } } ,\tag{81}
$$

yielding the score $1 - L _ { \mathfrak { s } } ^ { \mathrm { E } } / ( 1 - \beta _ { \mathfrak { s } } )$

Rotations are parameterized by Y–X–Y Euler angles. BFGS refinement uses analytic gradients derived from representation generators, a gradient tolerance of $1 0 ^ { - 1 0 }$ , and at most 200 iterations per run. We evaluate the identity and 128 Haar-distributed rotations, then refine the best four among the identity and first 32 samples, the best twelve among all 129 orientations, and the CSoMfitted orientation. The minimum finite residual across sampled and refined orientations determines the score.

## C.4.3 Learnable GNNs and torsion pretraining

We evaluate HEGNN, TFN, MACE, and Equiformer on complete directed graphs without selfedges, retaining all atoms. Node features encode elements, edges carry interatomic distances, and coordinates are centred without rescaling. Within each architecture, random, energy-pretrained, and Uni-Mol-style variants share the same initialization. The molecule-level split comprises 69 training molecules (1,656 conformations) and 17 validation molecules (408 conformations).

For energy pretraining, mean-pooled scalar features predict deposited total energies in Hartree, standardized by the training-set mean and population standard deviation. The GNN and prediction head are jointly optimized using mean absolute error.

Uni-Mol-style pretraining selects approximately 15% of atoms per conformation, with at least one atom selected. Among selected atoms, 90% are masked, 5% are replaced by an element sampled uniformly from the training-set elements, and $5 \%$ remain unchanged. Masked and replaced atoms receive independent coordinate perturbations drawn uniformly from [−1, 1] <sup>˚</sup>A per component, after which edge distances are recomputed. Training corruption is resampled each epoch, whereas validation corruption is fixed across epochs and architectures.

The pretraining loss combines masked-atom cross-entropy, coordinate SmoothL1, and pairdistance SmoothL1 with weights 1, 5, and 10, respectively. Both SmoothL1 terms use $\beta = 1$ Losses are computed at selected atoms, with pair-distance predictions spanning each selected atom and all atoms in its conformation. Distance targets are standardized using training-set statistics. Energy labels are not used.

Checkpoints are selected by the lowest validation energy MAE or fixed-corruption validation loss, respectively. Prediction heads are discarded, and spectral evaluation uses clean conformations with the shared even-degree pLS protocol. For Extended Data Table 2, we report raw pLS and two contrasts: the maxima–minima gap, defined as mean pLS at energy maxima minus mean pLS at energy minima, and the target-symmetry gap, defined as mean pLS with the target symmetry present minus that with it absent. Target presence follows the point-group assignments at tolerance 0.1, including equality.

Gaps are averaged over targets within each molecule. Relative to paired random initializations, increases exceeding 10<sup>−10</sup> in maxima–minima gaps or 10<sup>−6</sup> in target-symmetry gaps count as improvements. Changes in mean pLS, gap, and retrieval AUC are averaged equally across all 86 molecules. Improvement counts are reported for the full cohort and separately for the 17 validation molecules.

## C.5 DOTA complexes

The water-free Ln(DOTA) series<sup>10</sup> comprises seven CCDC depositions: 2281620 (Y), 2281617 (Tb), 2281615 (Dy), 2281619 (Ho), 2281616 (Er), 2281618 (Tm), and 2281329 (Yb). All crystallize in space group P4bm, with the metal on the fourfold axis. Metal-centred clusters are reconstructed using space-group operations and bond tracing, with a 2.9 <sup>˚</sup>A metal–donor cutof and a covalent cutof of 1.30 times the sum of Cordero radii for other bonds. Counterions and lattice solvent are excluded, yielding 53-atom [Ln(DOTA)] <sup>–</sup> units of composition $\mathrm { L n O _ { 8 } N _ { 4 } C _ { 1 6 } H _ { 2 4 } }$

The approximate-symmetry structures $\mathrm { [ D y ( D O T A ) ( H _ { 2 } O ) ] }$ and Tm(DOTA) are taken from the CSoM package<sup>11</sup>. Their heavy-atom structures contain 30 and 29 atoms, respectively, with the metal centred at the origin. The normal to the least-squares plane through the four coordinated macrocycle nitrogens defines the approximate fourfold axis. Because these structures lack hydrogens, the tetragonal series is likewise evaluated without hydrogens in Fig. 5f.

We evaluate each structure against $\mathrm { C _ { 4 } , S _ { 4 } , C _ { 4 v } , D _ { 4 h } , D _ { 4 } , D _ { 2 d } , C _ { 2 v } , C _ { 2 } }$ , and $\mathrm { C _ { 1 } }$ . Crystal-field analysis reports degree-resolved leakage $\mathrm { L S } _ { 5 } ^ { ( k ) }$ at degree $l = k$ for crystal-field rank k (Eq. (28)).

The package also provides a water-rotation trajectory for $\mathrm { [ D y ( D O T A ) ( H _ { 2 } O ) ] ^ { - } }$ . Its 360 frames rotate the capping water about the $\mathrm { D y - O }$ axis in $1 ^ { \circ }$ increments, with the coordination cage fixed. For each score series, we subtract its mean and identify the dominant harmonic among the first 24 nonzero angular harmonics (Supplementary Table C.8). Engel score is evaluated at every frame for the four shared targets using the orientation search in Supplementary Note C.4.2.

Supplementary Table C.8. Dominant harmonics of capping-water rotation. Entries give the largest-amplitude harmonic among n = 1, . . . , 24 after mean subtraction. Stars mark noise-sensitive components of near-constant CSoM profiles; a dash denotes an unavailable target.
<table><tr><td>Target</td><td>SymFP</td><td>CSoM</td><td>Engel</td></tr><tr><td> $\mathrm { C _ { 4 } }$ </td><td>4α</td><td>16α*</td><td>4α</td></tr><tr><td> $\mathrm { C } _ { 4 \mathrm { v } }$ </td><td>4α</td><td>8α</td><td>4α</td></tr><tr><td> $\mathrm { C _ { 2 } }$ </td><td>4α</td><td>21α*</td><td>4α</td></tr><tr><td> $\mathrm { C _ { 2 v } }$ </td><td>4α</td><td>4α</td><td>4α</td></tr><tr><td> $\mathrm { C _ { s } }$ </td><td>8α</td><td>2α</td><td></td></tr></table>

## C.6 Biological assemblies

## C.6.1 Matched regional analysis ofp97

We analyse six p97–UBXD1 structures from Braxton et al.<sup>12</sup>: 8FCL, 8FCM, 8FCR, 8FCQ, 8FCO, and 8FCT, all reconstructed with $\mathrm { C _ { 1 } }$ symmetry. The closed state 8FCL serves as reference. Each assembly contains six $\mathrm { p 9 7 }$ chains identified by accession P55072. Atoms are matched across chains and structures by residue number, insertion code, residue identity, atom name, and element, retaining only residues with complete N, Cα, C, and O backbones. This yields 712 residues per chain and 17,088 atoms per assembly. D1 spans residues 209–460 and D2 residues 481–763; each region is independently centred, represented, and scored.

We scan D1 and D2 using 40-residue windows with a stride of 20, adding a domain-end-aligned window where necessary. The resulting 12 D1 and 14 D2 windows require at least 50% residue coverage and ten complete residues per chain. All D1 and eight D2 windows retain 40 common residues per chain; the six partially covered D2 windows are indicated in Fig. 6d. Graphs use k = 24 nearest neighbours, including distance ties within absolute and relative tolerances of 10<sup>−7</sup> and $1 0 ^ { - 6 }$ . Node and edge radial scales are computed separately for each region.

Local gap annotations in Fig. 6c quantify the minimum separation between exposed van der Waals surfaces of adjacent windows, using deposited heavy atoms, including side chains. With Biopython atomic radii, changes exceeding 0.5 <sup>˚</sup>A are annotated when their direction is consistent at both 960 and 240 surface points per atom.

## C.6.2 Graph-neighbourhood and window-scale checks

For the six matched p97 structures, we assess whole-domain D1 and D2 scores across neighbourhood sizes $k \in \{ 6 , 1 2 , 1 6 , 2 4 , 3 2 , 4 8 , 9 6 \}$ , retaining distance ties (Extended Data Fig. 2).

For window-scale comparisons, we combine window centres from 40- and 80-residue scans with stride 20 and domain-end alignment. At each centre, we construct windows of 20, 40, 60, and 80 residues, clipped to the domain and restricted to complete-backbone residues matched across all six chains. Each selection is independently centred and scored at $k = 2 4$ , with the closed-reference score subtracted at the same centre and width. Window lengths refer to sequence intervals before unresolved residues are excluded. Unlike the main scan, this analysis retains all centres regardless of coverage, including partially resolved D2 windows with as few as three common residues per chain.

## C.6.3 Haemagglutinin

We analyse the X-31 haemagglutinin ectodomain series <sup>13</sup>: the pH-8 reference 6Y5G and four successive pH-5 states I–IV (6Y5H, 6Y5I, 6Y5J, 6Y5K). Chains are identified by accession P03437, with HA1 chains A, C, and E and HA2 chains B, D, and F forming three protomers.

The reference and states I–IV were reconstructed under $\mathrm { C _ { 3 } , C _ { 3 } , C _ { 1 } , C _ { 1 } }$ , and $\mathrm { C _ { 3 } }$ symmetry constraints, respectively. Fixed residue windows define HA1 (residues 45–310), HA2 (38–125), and their union. These selections contain 4,248 backbone atoms per state, except state III (4,236). Restricting states III and IV to the common HA2 range 38–124 preserves the pLS increase (1.196 versus 1.203 with the original windows).

Each scope is independently centred and evaluated against $\mathrm { C _ { 3 } }$ using a fixed 24-nearestneighbour graph, without the distance-tie inclusion used for p97. For each state, the self-matching axis is normal to the plane through the three protomer centroids, computed from the combined HA1–HA2 Cα selection. HA1 and HA2 share this axis.

## C.6.4 GluK2

We use biological assembly 1 of the desensitized GluK2 tetramer 5KUF<sup>14</sup>, reconstructed with $\mathrm { C _ { 2 } }$ symmetry. In author residue numbering, the amino-terminal domain (ATD) spans residues 1– 384; the ligand-binding domain (LBD), 398–513 and 636–775; and the transmembrane domain (TMD), resolved helix cores 521–554, 599–629, and 785–809. Boundaries follow the source study and UniProt P42260, mapped to author numbering. Retaining only residues with complete N, $\mathrm { C } \alpha , \mathrm { C } ,$ , and O atoms matched across all four chains yields 6,128, 4,096, and 1,360 atoms in ATD, LBD, and TMD, respectively.

Each domain is independently centred and scored against $\mathrm { C _ { 4 } }$ using a 24-nearest-neighbour graph. For spatial profiling, we define the receptor axis from the TMD to the ATD Cα centroid and partition the structure into non-overlapping layers of width 10, 20, 30, 40, 50, or 60 <sup>˚</sup>A, starting at the lowest selected Cα and retaining any terminal remainder. Residues are assigned by Cα height within each chain, retaining all four backbone atoms. Layers are lower-inclusive and upperexclusive, except the final layer, which includes the highest Cα. Each layer is independently centred and represented; plotted heights are measured from the lowest selected Cα.

## C.6.5 Echovirus 18

We analyse biological assembly 1 of 6HBH, 6HBK, 6HHT, and $6 \mathrm { H B L ^ { 1 5 } }$ , representing the intact activated capsid and capsids lacking one, two, and three pentamers, respectively. The four classes were reconstructed under I, $\mathrm { C } _ { 5 } , \mathrm { C } _ { 2 } .$ , and $\mathrm { C _ { 3 } }$ symmetry constraints. Capsid-protein sequences are aligned to 6HBH, retaining reference positions and residue identities shared across all copies and models within each protein. All physically present copies are included, yielding 40,800, 37,400, 34,000, and 30,600 Cα atoms, respectively. E18 uses a 24-nearest-neighbour graph and degree-six pLS against the proper icosahedral group I, whose degree-six fixed subspace is one-dimensional.

## C.7 Computational cost

Controlled computational benchmark. Timings are measured on a workstation with an Intel Xeon Gold 5318Y CPU and an NVIDIA RTX 4090 GPU, using a single CPU thread for both CPU and GPU measurements. Inputs begin as coordinates and atomic attributes in host memory. SymFP timings include centring, graph construction, host-to-device transfers where applicable, representation computation, SVD of each degree block, and target scoring, with final scores returned to the host. GPU timing boundaries are synchronized. Model initialization, group tables, and one-time code generation are excluded.

Complete-input benchmarks use the 17,088 matched backbone atoms of closed p97 (8FCL) and the 4,248 atoms of the HA global selection (6Y5G), following their respective graph and centring conventions. Regional throughput measures complete scans of all 26 windows across six p97 structures (156 evaluations per repetition), including window selection and representation reconstruction. Each region contains 480–960 atoms.

CSoM uses the oficial implementation, with either the deposited target orientation or the multistart fitting procedure in Supplementary Note C.4. Engel retains all directed pair contributions.

For search-free evaluations, a calibration call determines the repetition count: calls of at most 2 s receive three warm-ups and ten measured repetitions; longer calls receive one warm-up and three repetitions. Medians are reported. Biological coordinate-fitting runs use a single trial without search warm-up, subject to a 7200 s limit. Configurations terminate at the first timeout, with incomplete runs reported as censored.

Multiple target groups. We benchmark the 30-heavy-atom Dy-DOTA-water structure against the first 1, 2, 4, or $7$ targets in the ordered sequence $\mathrm { C _ { 2 } , C _ { 4 } , C _ { 6 } , D _ { 2 } , D _ { 4 } , D _ { 6 } } .$ , O. Coordinates retain the deposited metal-centred frame, and PF uses a complete graph.

Within each timed call, SymFP shares representations and spectra across targets, while Engel reuses its harmonic descriptor. CSoM shares centring and, for fixed orientations, deviations under common operation matrices. Coordinate fitting and BFGS optimize targets separately, whereas grid search reuses rotated descriptors. All caching is confined to individual calls, with shared computations counted once. CSoM uses the Powell procedure in Supplementary Note C.4. DOTA CSoM timings are medians of three repetitions following a warm-up on the seven-target set, using one pinned CPU thread.

Engel search timing. Panels a and b of Extended Data Table 3 compare Engel grid search with BFGS refinement, with and without CSoM initialization. Both BFGS variants follow Supple mentary Note C.4.2, difering only in the additional coordinate-fitted start and its refinement. Timings include coordinate preprocessing, all-pair descriptor construction, optional coordinate fitting, retained-energy optimization, and final scoring. Group tables and fixed numerical kernels are prepared beforehand.

The direct grid search samples Y–X–Y Euler angles at $1 5 ^ { \circ }$ intervals, yielding $2 4 \times 1 3 \times 2 4 = 7$ ,488 triples, including repeated orientations at the poles, without local refinement. Timings include rotation construction, descriptor rotation, and projection, with rotated descriptors reused across targets. The grid and BFGS timings reflect their respective search protocols rather than matched optimization accuracy.

Biological grid-search and BFGS runs without CSoM initialization, together with all DOTA Engel searches, are reported as medians of three repetitions on one pinned CPU thread. Biological runs are prewarmed on the smaller DOTA input, while DOTA runs receive their own warm-ups. Biological BFGS runs with CSoM initialization use single trials.

## D Supporting studies

## D.1 Spectral responses across equivariant GNNs

We compare PF with four randomly initialized learnable GNNs on the controlled deformations and crystallographic complexes in § 2.3 and 2.5.

Degree-resolved responses across GNNs. Along the subgroup chain, all five GNNs detect each symmetry-breaking transition at the first deformed frame (Extended Data Fig. 1a). Because all deformations preserve inversion, leakage relative to $\mathrm { C _ { i } }$ remains at the numerical floor.

For matched-amplitude deformations, spectral onset is the lowest degree whose relative tail energy exceeds $1 0 ^ { - 6 }$ at maximum amplitude, excluding round-of blocks. A block is classified as

$$
E _ { l } < N \epsilon _ { \mathrm { f p 3 2 } } ^ { 2 } E _ { \mathrm { m a x } } ,\tag{82}
$$

where $E _ { l }$ is its energy, $E _ { \mathrm { m a x } }$ the maximum block energy for the same model and structure, and $\epsilon _ { \mathrm { f p 3 2 } }$ the single-precision machine epsilon. The node count N = 30 accounts for accumulated round-of in pooling.

PF, TFN, MACE, and Equiformer recover onset degrees l = 1, 2, 3, 3 for axial staircase, in-plane stretch, in-plane radial, and axial pucker, respectively (Supplementary Table D.9). For HEGNN, the degree-three axial-pucker block remains at round-of throughout the sweep, shifting the first energy-supported response to l = 4 (Extended Data Fig. 1b).

Supplementary Table D.9. Onset degrees across deformation modes. Onsets exclude round-of blocks (Eq. (82)). † The l = 3 block remains at round-of; the first energy-supported response occurs at l = 4.
<table><tr><td>GNN</td><td></td><td>Axial staircase (|m| = 1) In-plane stretch (|m| = 2) In-plane radial (|m| = 3)</td><td></td><td>Axial pucker (|m| = 3)</td></tr><tr><td>PF</td><td>1</td><td>2</td><td>3</td><td>3</td></tr><tr><td>HEGNN⁶</td><td>1</td><td>2</td><td>3</td><td>4†</td></tr><tr><td>TFN7</td><td>1</td><td>2</td><td>3</td><td>3</td></tr><tr><td>MACE8</td><td>1</td><td>2</td><td>3</td><td>3</td></tr><tr><td>Equiformer9</td><td>1</td><td>2</td><td>3</td><td>3</td></tr></table>

To assess the efect of node attributes, we assign distinct types to the five node orbits, constant within each $\mathrm { C _ { 6 } }$ orbit and fixed across deformations. This preserves the underlying symmetries and restores HEGNN’s axial-pucker onset to l = 3, yielding the expected onset sequence l = 1, 2, 3, 3 across all five GNNs.

Crystal-field responses across GNNs. For the seven crystallographic $\mathrm { C _ { 4 } }$ clusters, all five GNNs exhibit numerical-floor leakage against $\mathrm { C _ { 4 } }$ and identical even-degree scores for $\mathrm { C _ { 4 } }$ and $\mathrm { S _ { 4 } }$ . The five-parameter targets show increased leakage at $k = 4 , 6$ , whereas k = 2 remains at the floor because each target admits one independent parameter at that degree.

## D.2 Graph-neighbourhood and window-scale checks for p97

We assess sensitivity to graph neighbourhood size and window width using the matched p97 structures (Supplementary Note C.6.2). Relative to the closed reference, the UBXD1 lariat mutant consistently exhibits higher D1 and lower D2 whole-domain pLS across all tested neighbourhood sizes. Its elevated D1 and predominantly reduced D2 regional pLS persist across all four window widths (Extended Data Fig. 2). The other four structures likewise show lower whole-domain D2 pLS throughout the neighbourhood sweep. Smaller D1 diferences are more sensitive to neigh bourhood size: the open structure shifts from a slight increase at low k to a decrease at high k. Narrower windows resolve finer spatial variation, including sign changes in near-zero diferences.

## D.3 Domain-resolved responses in haemagglutinin

During the low-pH transition from HA state III to $\mathrm { I V ^ { 1 3 } }$ ， $\mathrm { C _ { 3 } }$ pLS decreases in HA1 but increases in HA2 and the combined HA1–HA2 region (Extended Data Fig. 3). Structural views reveal outward splaying of the heads and extension of the stem.

A 120<sup>◦</sup> cyclic self-match about the fitted threefold axis provides a complementary coordinatespace assessment (Supplementary Table D.10). From state III to IV, the self-match residual increases in HA1 and decreases slightly in HA2, consistent with their respective pLS changes. Within each state, mean pairwise Cα RMSD after Kabsch alignment of protomers remains below 0.04 <sup>˚</sup>A, indicating closely matched internal conformations across protomers.

Supplementary Table D.10. Domain-resolved self-match residuals in haemagglutinin. Residuals (<sup>˚</sup>A) use a cyclic $1 2 0 ^ { \circ }$ rotation about the shared protomer-centroid axis.
<table><tr><td>Domain</td><td>neutral  $\mathrm { p H }$ </td><td>state I</td><td>state II</td><td>state III</td><td>state IV</td></tr><tr><td>HA1 head</td><td>0.26</td><td>0.26</td><td>0.54</td><td>0.85</td><td>1.05</td></tr><tr><td>HA2 stem</td><td>0.23</td><td>0.22</td><td>0.54</td><td>0.51</td><td>0.49</td></tr></table>

## D.4 Spatial symmetry organization in GluK2

GluK2 exhibits a twofold arrangement in ATD and approximately fourfold arrangements in LBD and TMD<sup>14</sup>. Consistently, $\mathrm { C _ { 4 } }$ pLS is highest in TMD, followed by LBD and ATD (Extended Data Fig. 4c). Across layer thicknesses of 10–60 <sup>˚</sup>A, TMD-proximal layers generally exhibit higher pLS than upper extracellular layers, with local variations depending on layer boundaries and thickness (Extended Data Fig. 4b; protocol in Supplementary Note C.6.4).

## D.5 Icosahedral target agreement in E18 capsids

Against the proper icosahedral group I, degree-six pLS decreases from 6.56 in the intact activated E18 capsid to 1.65, 1.15, and 0.85 in structures lacking one, two, and three pentamers, respectively (Extended Data Fig. 5). Each class was reconstructed independently (Supplementary Note C.6.5).

## Supplementary References

1. Mois I. Aroyo, editor. International Tables for Crystallography: space-group symmetry, volume A. Wiley, 6 edition, 2016. ISBN 978-0-470-97423-0.

2. G. D. Mostow. Equivariant embeddings in euclidean space. Annals of Mathematics, 65(3): 432–446, 1957.

3. Boris S. Mityagin. The zero set of a real analytic function. Mathematical Notes, 107(3): 529–530, 2020.

4. Michael Engel. Point group analysis in particle simulation data. arXiv preprint arXiv:2106.14846, 2021.

5. Chrameh Fru Mbah, Junwei Wang, Silvan Englisch, Praveen Bommineni, Nydia Roxana Varela-Rosales, Erdmann Spiecker, Nicolas Vogel, and Michael Engel. Early-stage bifurcation of crystallization in a sphere. Nature Communications, 14:5299, 2023.

6. Jiacheng Cen, Anyi Li, Ning Lin, Yuxiang Ren, Zihe Wang, and Wenbing Huang. Are highdegree representations really unnecessary in equivariant graph neural networks? Advances in Neural Information Processing Systems, 37:26238–26266, 2024.

7. Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhof, and Patrick Riley. Tensor field networks: rotation- and translation-equivariant neural networks for 3D point clouds. arXiv preprint arXiv:1802.08219, 2018.

8. Ilyes Batatia, David P Kovacs, Gregor Simm, Christoph Ortner, and G´abor Cs´anyi. MACE: higher order equivariant message passing neural networks for fast and accurate force fields. Advances in Neural Information Processing Systems, 35:11423–11436, 2022.

9. Yi-Lun Liao and Tess Smidt. Equiformer: equivariant graph attention transformer for 3D atomistic graphs. In The Eleventh International Conference on Learning Representations, 2023.

10. Anna Schannong Manvell, Rouven Pfleger, Niels Andreas Bonde, Matteo Briganti, Carlo Andrea Mattei, Theis Brock-Nannestad, Høgni Weihe, Annie K. Powell, Jacques Ollivier, Jesper Bendix, and Mauro Perfetti. LnDOTA puppeteering: removing the water molecule and imposing tetragonal symmetry. Chemical Science, 15:113–123, 2024.

11. Villads R. M. Nielsen and Thomas Just Sørensen. Determining molecular structure, coordi nation geometry, and molecular symmetry using a continuous symmetry operation measure software. Nature Communications, 16:11122, 2025.

12. Julian R. Braxton, Chad R. Altobelli, Maxwell R. Tucker, Eric Tse, Aye C. Thwin, Michelle R. Arkin, and Daniel R. Southworth. The p97/VCP adaptor UBXD1 drives AAA+ remodeling and ring opening through multi-domain tethered interactions. Nature Structural & Molecular Biology, 30:2009–2019, 2023.

13. Donald J. Benton, Steven J. Gamblin, Peter B. Rosenthal, and John J. Skehel. Structural transitions in influenza haemagglutinin at membrane fusion pH. Nature, 583:150–153, 2020.

14. Joel R. Meyerson, Sagar Chittori, Alan Merk, Prashant Rao, Tae Hee Han, Mihaela Serpe, Mark L. Mayer, and Sriram Subramaniam. Structural basis of kainate subtype glutamate receptor desensitization. Nature, 537:567–571, 2016.

15. David Buchta, Tibor F¨uzik, Dominik Hreb´ık, Yevgen Levdansky, Luk´aˇs Suken´ık, Liya Mukhamedova, Jana Moravcov´a, Robert V´acha, and Pavel Plevka. Enterovirus particles expel capsid pentamers to enable genome release. Nature Communications, 10:1138, 2019.