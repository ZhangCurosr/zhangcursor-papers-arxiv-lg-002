# Interrelating Fruchterman-Reingold Graph Visualization and Agglomerative Clustering

Alexandre Benatti<sup>1</sup> and Luciano da F. Costa<sup>2</sup>

<sup>1</sup>Institute of Mathematics and Statistics - DCC University of S˜ao Paulo Rua do Mat˜ao, 1010, S˜ao Paulo, SP 05508-090 Brazil

<sup>2</sup>S˜ao Carlos Institute of Physics - DFCM University of S˜ao Paulo Av. Trabalhador S˜ao-Carlense, 400, S˜ao Carlos, SP 13566-590 Brazil (Prof. Senior)

11th Sep., 2026

## Abstract

Graph visualization methods and agglomerative clustering have been frequently considered in data analysis and pattern recognition. Because these approaches are interrelated and complementary, it is of particular interest to investigate their associations. In this work, we study the possible relationship between the Fruchterman-Reingold graph visualization method and four types of agglomerative clustering adopting single- and complete-linkage, average, and Ward’s linkage criteria. Three types of datasets have been considered in 2 and 10 dimensions, as well as the PCA projection of the latter to two dimensions. The results obtained suggest that the relationship between the methods considered did not vary much for the three types of data mentioned above. At the same time, the agglomerative methods tended to yield results that are mostly similar to each other, while presenting moderate similarity with the original data. The Fruchterman-Reingold visualization resulted similar to the original data, but exhibited relatively smaller similarity to the agglomerative methods.

## 1 Introduction

Graph visualization (e.g. [1, 2]) and agglomerative clustering (e.g. [3, 4]) have been frequently considered in data analysis and pattern recognition.

As implied by its own name, the main objective of graph visualization is to obtain visual representations of graphs and complex networks. At the same time, agglomerative clustering concerns typically non-supervised methods for estimating hierarchical relationships between data elements (characterized by respective measurements or features), which can be taken as a subsidy for searching for the existence of respective clusters. These relationships are obtained by successively merging data elements and subgroups according to some distance or similarity criterion. The results obtained by agglomerative clustering can often be represented as dendrograms, indicating how data elements are progressively joined into subclusters.

Although graphs and dendrograms can be considered distinct structures, they are interrelated and can be used in a complementary manner. In this work, the elements of a data set (or the leaves of a respective dendrogram) will be understood to be associated to nodes in a corresponding graph. While the modular structure of a low dimensional data set can be more directly appreciated from a respective graph visualization, dendrograms provide a representation of the complete hierarchical relationships between the data elements. In addition, the latter type of representation and analysis assumes particular importance in case the data elements are characterized by more than 2 or 3 measurements (dimensions), because in these cases graph visualization, by unavoidably implementing a projection onto a two-dimensional space, cannot typically preserve the adjacency among the elements and groups of elements.

Some level of congruence can be expected between graphs and dendrograms deriving from the same source. For instance, a data set involving 4 well-separated clusters should yield a visualization characterized by 4 associated groups of nodes that are more densely interconnected to each other than to nodes of diferent groups. Therefore, it would be expected that a respective dendrogram obtained using agglomerative clustering would correspond to 4 main branches that are well-separated from each other in the sense of having relatively long stems. At the same time, a graph visualization in which no discernible modules (or communities) can be identified should be accompanied by a dendrogram without well-separated branches.

Interestingly, it is possible to transform (e.g. [5]) between data sets, dendrograms, and graphs in several ways. For example, given a data set, the values of similarity indices or distances between the features of its elements yield a respective weighted and non-directed graph or network. At the same time, a graph can be transformed into a data set in several ways. For example, a weight matrix defining a respective graph can be obtained from the cophenetic distance (e.g. [6]) between the subgroups in a dendrogram. It is also possible to obtain comparisons between the features associated with each of the nodes in a graph, including complementary properties of nodes and/or respectively, topological measurements.

In the current work, we study a possible relationship between the often used Fruchterman-Reingold Graph Visualization methodology and four frequently adopted types of agglomerative clustering, namely those employing single- and complete-linkage, average, and Ward’s linkage criteria. As will be described, this visualization method can be more closely related, though by a little margin, to agglomerative clustering by using the Ward’s criterion. The results obtained from the four considered agglomerative methods were mostly similar, while the Fruchterman-Reingold visualization presented good compatibility with the original data, but smaller similarity with the four agglomerative approaches.

This work starts by presenting a brief review of the Fruchterman-Reingold visualization method and the considered agglomerative clustering approaches, and follows by presenting and discussing the comparison between them.

## 2 Fruchterman-Reingold Graph Visualization

The visualization of graphs can be approached in several ways, of which force-based algorithms (e.g. [1]) have been frequently considered. This type of approach is developed in analogy with a physical system involving several particles (the nodes) which interact with each other through attractive and repulsive forces. For example, the particles can be understood to have the same type of charge (e.g. negative or positive), so that they repel each other. At the same time, each pair of particles can have a spring of non-zero length associated, therefore quantifying the similarity (or proximity) between pairs of nodes.

In physical terms, these two types of interactions can be associated with respective energies (potential and spring). Given an initial configuration of nodes position, the system unfolds along time toward a minimum of energy, which may correspond to a local or global minimum. Temperature is often associated with the particle system, which controls the probability of the magnitude of state changes. Typically, the global minimum cannot be assumed to have been achieved in finite time.

The graph visualization approach known as Fruchterman-Reingold (e.g. [1]) considers interactions between nodes as described above, with the electric field part of the interaction implying the particles to become more and more distributed spatially, while the springs, with respective constants related to the weights of the respective edges, tend to keep them together. In practice, the visualizations obtained by the Fruchterman-Reingold method have particles that are close to each other in the graph placed together in the respective visualization. It is also important to keep in mind that diferent, though congruent, graph visualizations are typically obtained by repeated application of the Fruchterman-Reingold method on the same data.

## 3 Agglomerative Clustering

Agglomerative clustering methodologies (e.g. [7, 8, 9]) can be understood aiming to progressively merge the elements in a given set of data while considering a scale variable, here represented as s, which can be associated with distance (e.g. Euclidean) or similarity (e.g. Jaccard). In the case of distancebased approaches, starting with each data element being associated with a subgroup of size 1, the pair of nearest subgroups is identified and interconnected. This basic step proceeds until a single group remains, or a maximum distance is reached.

Several types of rules can be taken into account for merging subgroups. In the present work, we consider the single-, complete, average, and Ward’s merging criteria. Given two sets A and B, the single-linkage criterion results in the value of the smallest distance between all possible combinations of elements from A and elements from B. At the same time, the complete linkage criterion takes to largest of those distances. The average linkage yields the average of the pairwise distances between the elements taken from A and B. The Ward’s approach merges the two sets so as to minimize the resulting dispersion.

A hierarchy of relationships between the so resulting subgroups is obtained, which is typically visualized in terms of a respective dendrogram. Data elements which are less distant (or are more similar) tend to appear together in respective branches, the lengths of which reflect the diference between the respective elements. At the same time, the value of s where a merge takes quantifies the level of interrelationship (distance or similarity) between the two sets.

## 4 Methodology

In order to better understand possible relationships between Fruchterman-Reingold visualizations and agglomerative clustering, the methodology illustrated in the flow diagram in Figure 1 has been considered here.

![](images/b3d4601042e5d4ce1331072dcd5c36d11a1e1d18949fdf8bd714fc879e006e47.jpg)  
Figure 1: Flow diagram illustrating the approach adopted for relating Fructerman-Reingold graph visualization and four types of agglomerative clustering.

The first step is to obtain a dataset of N elements characterized by M respective measurements or features. The experiments in this work consider $M = 2$ and $M = 1 0$ , implying two and ten features understood to be associated with a metric space. Two-dimensional projections of the latter datasets area also taken into account. It should be observed that other configurational choices may lead to diferent types of results.

Because we are interested in considering relatively intricate data set structures characterized by modularity and hierarchy, synthetic data are generated hierarchically according to a respective statistical model. A set of $n _ { 1 }$ points is drawn with normal probability from a M−dimensional space, having average corresponding to the position of the reference point and standard deviation (circularly symmetric covariance matrices are assumed). Then, each of these points is replaced by a set of $N _ { 2 }$ new points drawn with normal probability $\sigma _ { 2 }$ , and so on.

We considered the pairwise Euclidean distance matrix of the generated data points (denoted by $M _ { D } )$ as the reference data. From this distance matrix, four diferent agglomerative hierarchical clustering linkage methods were evaluated: single, complete, average, and Ward. The resulting dendrograms were then converted into cophenetic distance matrices, $M _ { C }$ . In a cophenetic matrix, each entry represents the height at which two elements merge in a given dendrogram. Consequently, these cophenetic distance values depend on the linkage criterion adopted during the dendrogram construction.

Additionally, the distance matrix $M _ { D }$ was transformed into a similarity matrix S that was then used to construct a weighted similarity network. This was obtained by taking the reciprocal of the distances $( S = 1 / M _ { D } )$ . The Fruchterman–Reingold force-directed layout algorithm was applied to the similarity matrix to obtain a two-dimensional embedding of the data, and the pairwise Euclidean distances between the resulting positions were calculated to obtain an additional distance matrix (FR) that is used to compare with the other approaches considered.

The comparison among these six representations was performed using only the upper triangular elements of each distance matrix, excluding the diagonal. Each vector obtained was standardized independently by subtracting its mean and dividing by its standard deviation. Finally, the standardized vectors were compared pairwise using the coincidence similarity index $C ,$ (e.g. [10, 11]). This similarity index combines the Jaccard (e.g. [12, 13]) and interiority (or overlap, e.g. [14]) indices.

## 5 Results and Discussion

We begin by presenting an example of the comparison approach respectively to the data set in Figure $2 ( \mathrm { a } )$ , which is contained in a two-dimensional space. The colors indicate the groups of points which are progressively added during the adopted data generation methodology.

The dendrograms obtained from the data set in Figure $2 ( \mathrm { a } )$ by using agglomerative clustering with single-, complete-, average, and Ward’s linkage criteria are presented in Figure 3. The structures obtained for the complete-, average, and Ward’s approaches resulted mostly similar, while the dendrogram obtained by single-linkage can be observed to be more diferent.

![](images/64aeb40d8b5be588e4361f5fd172656be6ac673c59de56d5ff5f2036abab614f.jpg)  
(a)

![](images/1202e3d70e0cffac993ddbd07bd32885a7f5733853aee5745d48f35b51632233.jpg)  
(b)

Figure 2: Example of 2D dataset considered in our study (a) and respective visualization by using the Fruchterman-Reingold method on the reciprocal of the Euclidean distances.  
![](images/3d20af8e28ef22d312cb01592c478806fffecb2f7dc0ce85207fc732197dab3d.jpg)

![](images/092d7fbd50d69d84a1e36c34f39a7e631aa9a3eb84e20af2087f272eb74047f5.jpg)

![](images/8ed6fb2b7e59da2dcc54ce3f966dfa354596759f9a51767462f75e1b410e7429.jpg)

![](images/fe733c97a2181000f7fa3ed9b358ecfa6c5d01c8820307736a0a0a61f0d15708.jpg)  
Figure 3: Dendrograms obtained by the agglomerative clustering approaches adopting single- (a), complete- (b), average (c), and Ward’s (d) linkage criteria. Observe that diferent upper limits of the distance axes have been chosen for the sake of improved visualization.

Figure 4 shows the matrix of similarity values obtained by comparing each of the visualization and agglomerative approaches considered. In addition to confirming that the dendrogram obtained for the single-linkage methodology is less similar to those obtained by using the three other agglomerative approaches, this matrix also indicates that the original data is only moderately similar to all visualization and agglomerative methods considered in this work. At the same time, the Fruchterman-Reingold method yielded visualizations that are not particularly similar to any of the other approaches, which is discussed as follows.

Figure 2(b) presents the visualization of the dataset in (a) using the Fruchterman-Reingold approach. As can be readily perceived, though the adjacency between the groups represented in colors are mostly maintained, the distances between the nodes result not only in a more compact, but also

in a more uniform distribution.

![](images/4a4799b5f21ffe1f2e3bab5aac25b9d6eed5f140bd0a6d8e24c1fda855d673cd.jpg)  
Figure 4: Coincidence similarity matrix (D = 1) obtained for the dataset in $\mathrm { F i g . 2 ( a ) }$

The study of the relationship between the considered visualization and agglomerative approaches has been performed in terms of three experiments involving data generated as described in the previous section in two-dimensional and ten-dimensional spaces, as well as a two-dimensional projection of the latter. For each of these types of data, a total of 5,000 experiments have been performed to estimate the intrinsic relationships between the distance matrices resulting from Fruchterman-Reingold visualization and the four types of agglomerative clustering considered.

Figures 5, 6(a), and 6(b) illustrate the matrix of similarity obtained for the two- and ten-dimensional type of data, and the PCA projection of the latter, respectively.

![](images/599033bd29eeb0c58a2fa72ea75d55d84c7f007c994605fbebf8c962c95933cb.jpg)  
Figure 5: Average coincidence similarity matrix (D = 1) obtained for 5,000 datasets, considering two-dimensional data.

The original dataset has been found to be reasonably similar to all agglomerative methods. Although the four agglomerative methods were found to be interrelated, smaller similarity values were obtained between the Fruchterman-Reingold visualization and the four agglomerative methods. The Fruchterman-Reingold visualization was found to be particularly similar to the original dataset in the cases of the two-dimensional type of data and two-dimensional projection of the ten-dimensional type of data. The visualization resulted less similar to the type of ten-dimensional data because of the unavoidable projection that it implements cannot preserve the original adjacency between the data elements and groups of elements.

![](images/479459abbd877a151b983d449e7dbcb1b8091ac3c7e8f7fc6645ed8e46e79ca1.jpg)  
(a)

![](images/38eb392ca228acb76ca4bd837b9a6cc7f3790c579a6170520398ccb6786d17c3.jpg)  
(b)  
Figure 6: Average coincidence similarity matrix $( D \ = \ 1 )$ obtained for 5,000 tendimensional datasets (a) and for its two-dimensional PCA projection.

The relationship between the similarity matrix in the tables shown above is illustrated in terms of pairwise correlograms and Pearson correlation coeficients P in Figure 7. The results obtained indicated that these relationships are mostly compatible and congruent, meaning that the similarities between the approaches observed for the two-dimensional and ten-dimensional datasets as well as the PCA projection of the latter present similar properties.

![](images/3a8e490f441738c5553fd2ddf42227f539aefba65d99bdcf1f8fbe82047dd762.jpg)  
(a)

![](images/3dff4de4fb5522d6149570bf0a25572d59ccfd9b86f6ae8bf9bb85b3877ec3a1.jpg)  
(b)

![](images/8a6e78ea2dc0a8fdf9d53b1fb290b8d3db6e5c96b20b2364f260bb737279e2a9.jpg)  
(c)  
Figure 7: Correlograms relating the similarities between the visualization and agglomerative clustering obtained for three data sets considering two and ten dimensions, and the 2D PCA projection of the latter. The dashed line indicates the linear data regression.

Interestingly, the similarity relationships were in strong positive correlations, corroborating the above observations.

## 6 Concluding Remarks

Graph visualization and agglomerative clustering have often been used in several areas and from varying perspectives. Although conveying the data structure in distinct ways, some substantial level of relationship and coherence is still expected between the obtained graph visualization and dendrograms. That is of particular interest, because these two types of representation can not only better represent the original dataset, but also provide complementary information about the original data and the problem from which they were obtained.

In the present work, an approach was developed aimed at trying to relate the frequently used Fruchterman-Reingold graph visualization methodology and agglomerative clustering based on four linkage criteria. More specifically, an experimental approach has been applied in which graph visualizations and four types of dendrograms (considering single-, complete-, average, and Ward’s linkage criteria) are obtained from the same set of data elements synthesized according to a hierarchical statistical model. The visualization was then compared with the four dendrograms in terms of the distances between every pair of points.

The obtained results suggested, for the considered data sets and respective parametric configurations, that, although the Fruchterman-Reingold graph visualization methodology tended to be more similar to the original to the two-dimensional sets of data, its similarities with data obtained the four agglomerative approaches were smaller. The results obtained using Ward’s approach were found to be slightly more similar to the Fruchterman-Reingold visualization than with the dendrograms obtained for the three other linkage criteria. Interestingly, the relationships obtained were found to be mostly common to the three types of data considered.

The results obtained are of special interest because they provide subsidies for choosing among the four agglomerative clustering approaches while complementing the analysis of specific experiments and data sets by combining graph visualization and agglomerative clustering. It also provides indication, in the context of the adopted data and configurations, that the agglomerative methods preserve to a good extent the original organization of the data and that projection-based graph visualizations, such as that implemented by the Fruchterman-Reingold method, work better for lower dimensional data sets.

The approach, methodology, and results reported in the present work pave the way to several related further developments. Given that the results obtained refer to a specific model of synthesized data, other dimensions and types of data need to be taken into account in complementary experiments. It would also be of interest to consider other types of graph visualization, as well as clustering approaches.

## Acknowledgments

A. Benatti is grateful to FAPESP (grant 2025/26083-7 and 2022/15304- 4). Luciano da F. Costa thanks CNPq (grant no. 313505/2023-3) and FAPESP (grant 2022/15304-4).

## References

[1] T. M. J. Fruchterman and E. M. Reingold. Graph Drawing by Force-Directed Placement. Software: Practice and Experience, 21(11):1129–

1164, 1991.

[2] I. Herman, G. Melan¸con, and M. S. Marshall. Graph visualization and navigation in information visualization: A survey. IEEE Transactions on Visualization and Computer Graphics, 6(1):24–43, 2000.

[3] R. O. Duda, P. E. Hart, and D. G. Stork. Pattern Classification. Wiley-Interscience New York, 2nd edition, 2000.

[4] S. Theodoridis and K. Koutroumbas. Pattern Recognition. Elsevier, 2006.

[5] C. H. Comin, T. Peron, F. N. Silva, D. R. Amancio, F. A. Rodrigues, and L. da F. Costa. Complex systems: features, similarity and connectivity. Physics Reports, 861:1–41, 2020.

[6] R. R. Sokal and F. J. Rohlf. The Comparison of Dendrograms by Objective Methods. Taxon, 11(2):33–40, 1962.

[7] R. S. Hill. A Stopping Rule for Partitioning Dendrograms. Botanical Gazette, 141(3):321–324, 1980.

[8] M. Z. Rodriguez, C. H. Comin, D. Casanova, O. M. Bruno, D. R. Amancio, L. da F. Costa, and F A. Rodrigues. Clustering Algorithms: A comparative approach. PloS One, 14(1):e0210236, 2019.

[9] E. K. Tokuda, C. H. Comin, and L. da F. Costa. Revisiting agglomerative clustering. Physica A: Statistical Mechanics and its Applications, 585:126433, 2022.

[10] L. da F. Costa. On similarity. Physica A: Statistical Mechanics and its Applications, 599:127456, 2022.

[11] L. da F. Costa. Coincidence complex networks. Journal of Physics: Complexity, 3:015012, 2022.

[12] Wikipedia. Jaccard index, 2021. https://en.wikipedia.org/wiki/ Jaccard\_index. [Online; accessed 24-Sep-2026].

[13] L. da F. Costa. Further generalizations of the Jaccard index. https://www.researchgate.net/publication/355381945\_ Further\_Generalizations\_of\_the\_Jaccard\_Index, 2021.

[14] M. K. Vijaymeena and K. Kavitha. A survey on similarity measures in text mining. Machine Learning and Applications: An International Journal, 3(2):19–28, 2016.