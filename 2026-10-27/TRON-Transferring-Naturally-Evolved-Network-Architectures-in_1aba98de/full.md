# TRON: Transferring Naturally Evolved Network Architectures into Neuroevolution

Zacharopoulos, Konstantinos<sup>2,\*</sup>, Bekou, Alexandra<sup>4</sup>, Serradell, Jose M<sup>1</sup>, Paraskevopoulou Foteini<sup>4</sup>, Fournaris, Apostolos<sup>4</sup>, Arapakis, Ioannis<sup>3+</sup>, and Lao, Oscar<sup>1,\*,+</sup>

<sup>1</sup>IBE, Institute of Evolutionary Biology (CSIC-Universitat Pompeu Fabra), Barcelona, 08003, Spain

<sup>2</sup>Technical University of Crete, ECE Department,Chania, 732 00, Greece

<sup>3</sup>Barcelona Supercomputing Center, Barcelona, 08034, Spain

<sup>4</sup>R.C. ATHENA, Industrial Systems Institute, Patras, 265 04, Greece

<sup>5</sup>Computer Science Department, University of Turin, Turin, 10149, Italy

<sup>6</sup>ISI Foundation, Turin, 10126, Italy

constantine.zach97@gmail.com, oscar.lao@ibe.upf-csic.es

<sup>+</sup>both authors equally contributed

## ABSTRACT

Neural architecture search (NAS) remains a challenging optimization problem due to the vast and highly non-linear space of possible artificial neural network (ANN) topologies. Evolutionary approaches have demonstrated the ability to automatically generate ANN architectures; however, their effectiveness is often limited by the difficulty of preserving functional organization under mutation and recombination. In biological systems, complex networked phenotypes emerge from genetic information tha has remained evolvable over long evolutionary timescales, suggesting that naturally evolved biological networks may encode organizational principles that facilitate adaptation.

Here, we present TRON (TRansfer Of Natural networks), a neuroevolution framework that incorporates experimentally characterized protein-protein interaction (PPI) networks as architectural priors for ANN evolution. Rather than initializing search from randomly generated topologies, TRON exploits biologically derived interaction patterns as reusable structural building blocks. The framework combines classical evolutionary operators with mechanisms inspired by neuroevolution, including speciation, while introducing a species quarantine strategy designed to protect newly generated architectures during an initial maturation phase.

We evaluated TRON on three classification benchmarks (CORA, CITESEER, and PUBMED) and three regression benchmarks (DIABETES, FRIEDMAN, and CALIFORNIA), comparing its performance against established evolutionary and neuroevolutionary approaches. Across all classification datasets and most regression tasks, TRON achieved the highest predictive performance among the evaluated methods. Furthermore, the framework exhibited substantially lower variability across independent runs, indicating improved robustness and convergence stability during evolutionary search. While these gains were associated with increased computational requirements, the results consistently suggest that biologically derived interaction networks provide more favorable substrates for neuroevolution than randomly generated topologies.

Taken together, our findings support the hypothesis that protein interaction networks can serve as effective architectural priors for ANN evolution. More broadly, this work demonstrates the feasibility of transferring naturally evolved biological network architectures into artificial intelligence systems and suggests a new direction for improving evolvability in neural architecture search through the reuse of biological organizational principles.

Please note: Abbreviations should be introduced at the first mention in the main text — no abbreviations lists. Suggested structure of main text (not enforced) is provided below.

## Introduction

Artificial neural networks (ANNs) are a family of machine learning models inspired by the organization of biological nervous systems. The architecture of an ANN, defined by the arrangement of its neurons and connections, plays a central role in determining its representational capacity, generalization performance, and computational efficiency<sup>1</sup>. Identifying effective architectures, however, remains a challenging optimization problem due to the enormous number of possible network topologies.

To address this challenge, neural architecture search (NAS) methods have been developed to automate the design of ANN architectures<sup>2</sup>. Among these approaches, evolutionary algorithms (EAs) provide a particularly attractive framework because they naturally support the exploration of large, discrete, and non-differentiable search spaces. In neuroevolution, evolutionar principles such as mutation, recombination, and selection are applied directly to neural networks, enabling the optimization of network parameters, architectures, or both simultaneously<sup>3,</sup> <sup>4</sup>. Approaches such as NeuroEvolution of Augmenting Topologies (NEAT) demonstrated that evolutionary search can successfully discover increasingly complex neural architectures while maintaining architectural diversity through mechanisms such as speciation<sup>3</sup>.

Despite these successes, neuroevolution still faces important challenges. Small syntactic modifications to an encoded neural network may produce disproportionately large functional changes in the resulting architecture. As a consequence, mutation and recombination frequently disrupt useful organizational structures, limiting the efficiency of evolutionary search. This problem becomes increasingly severe as network complexity increases and represents a longstanding challenge for evolutionary representations of neural systems.

Biological evolution provides an intriguing source of inspiration for addressing this limitation. In living organisms, complex networked phenotypes emerge from genetic information that has remained evolvable over hundreds of millions of years of evolution. However, directly emulating the genetic encoding of biological nervous systems is difficult because the mapping between DNA sequence and mature neural circuitry involves multiple intermediate processes, including gene regulation, development, cellular differentiation, and environmental interactions.

An alternative perspective is to focus on biological networks whose organization is more directly related to their genetic encoding. Protein-protein interaction (PPI) networks represent one such example. Proteins are the primary functional products of genes, and their interactions depend largely on sequence-derived structural features. Consequently, the relationship between genotype and network organization is substantially more direct than for large-scale nervous systems.

Moreover, PPI networks frequently exhibit modular and hierarchical organization, properties that have long been proposed to contribute to robustness, functional specialization, and evolvability in biological systems<sup>5–7</sup>. Experimental and computational studies have shown that cellular functions are often organized into semi-independent modules whose components interact more strongly within modules than between them, facilitating the reuse and adaptation of functional units across different biological contexts<sup>5,</sup> <sup>8</sup>.

Importantly, large collections of experimentally characterized PPI networks are now available across diverse organisms. Advances in high-throughput experimental techniques, together with extensive curation efforts, have led to the reconstruction of protein interaction maps containing thousands of proteins and interactions in organisms ranging from yeast to humans<sup>9–11</sup>. These resources can be viewed as biological interaction architectures that have persisted under evolutionary pressures and therefore represent a rich source of naturally occurring connectivity patterns. While it remains unclear which specific properties contribute to their evolutionary persistence, the reuse of such structures may provide useful architectural priors for evolutionary search. Rather than initializing neuroevolution from random topologies, biologically derived interaction motifs may bias the search process toward more coherent regions of the architectural landscape and reduce the disruptive effects of mutation and recombination.

In this study, we propose TRON, a neuroevolution framework that incorporates experimentally characterized PPI networks as architectural substrates for artificial neural network evolution. Rather than searching exclusively from randomly generated topologies, TRON reuses biologically derived interaction patterns as candidate building blocks for ANN architectures. We hypothesize that these structures may provide a more evolvable substrate for neuroevolution by promoting the preservation of functional organization during mutation and recombination. To evaluate this hypothesis, we compare TRON against established evolutionary and neuroevolutionary approaches across a collection of classification and regression benchmarks and assess its impact on both predictive performance and evolutionary robustness.

## 1 Related Work

Gradient-based optimisation is highly effective for training the weights of a fixed network, but it is poorly suited to searching over architectures themselves, leaving design decisions such as depth, width and connectivity to be chosen manually. Differentiable NAS methods such as DARTS<sup>12</sup> avoid this by relaxing the architecture choice into a continuous mixture of candidate operations. This makes the search efficient, but it confines it to predefined cell-based search spaces and makes it prone to collapse towards degenerate architectures<sup>13</sup>. However, the resulting topologies are stacks of repeated, hand-designed cells. They carry no information about which structures stay functional when modified, a property that biological networks have acquired under long-term evolutionary pressure. Where biological networks have entered deep learning, they appear either as input data, as in graph convolutional networks<sup>14</sup>, whose architectures remain hand-designed and suffer from over-smoothing<sup>15</sup>, or as fixed wiring, as in knowledge-primed models such as P-NET<sup>16</sup> and KPNNs<sup>17</sup>, whose topologies are imposed rather than searched. Neither treats biological interaction patterns as reusable, evolvable building blocks.

Population-based metaheuristics offer a natural alternative. They require no gradients, and because they maintain a population of candidates rather than a single solution, they can explore several regions of the search space in parallel. Among these optimisers, Particle Swarm Optimization (PSO)<sup>18</sup> is one of the earliest and most widely adopted. PSO is a swarmintelligence metaheuristic algorithm inspired by collective cooperative behavior in social organisms, such as birds, fish and insects. In PSO, a swarm of particles is initialized as a set of candidate solutions to an objective function. At every iteration, each particle’s current position is updated according to its previous best position, as well as the global best position found by the swarm. Due to its relatively simple implementation and gradient-free nature, PSO has been applied to a number of optimization problems ranging from image processing to electrical power systems optimization. However, it is known to be susceptible to premature convergence due to insufficient population diversity, particularly when solving highly complex and multi-modal optimization problems, as well as suboptimal particle velocity tuning which can lead to stagnation around an optimal solution Fundamentally, PSO steers its search by a few elite positions instead of using the information held in the population as a whole.

Differential Evolution (DE)<sup>19</sup> removes this dependence on elite attractors by deriving search directions from the population itself. DE begins by randomly initializing a set of candidate solutions represented as real-valued vectors. At each iteration, new candidate solutions, referred to as trial vectors, are generated through mutation strategies that typically involve the weighted difference of randomly selected population members. The resulting donor vectors undergo recombination with existing solutions, creating trial vectors that are evaluated according to the corresponding objective function. Solutions are then selected based on their fitness, guiding the population toward the global optimum. Despite its robustness and applicability to high-dimensional optimization problems, DE remains sensitive to noisy objective functions as well as parameter tuning, particularly regarding the mutation factor, crossover rate and population size, leading to issues such as premature convergence or inefficient search space exploration.

The Covariance Matrix Adaptation Evolution Strategy (CMA-ES)<sup>20</sup> addresses this limitation by explicitly handling the dependencies between variables, which makes it one of the most robust metaheuristics for handling ill-conditioned and multimodal problems. CMA-ES optimizes a given objective function by iteratively adapting a multivariate normal mutation distribution. Initially, the mutation is modelled as a normal distribution with zero mean, characterized by a covariance matrix. At each iteration, candidate solutions are sampled from the current distribution and are evaluated according to the objective function. The mean, step size and covariance matrix of the distribution are subsequently updated according to the solutions with the highest fitness. This strategy allows the algorithm to progressively adapt its search distribution to the local geometry of the optimization landscape. Despite its strong local adaptation capabilities, CMA-ES remains vulnerable to converging to local optima on multimodal landscapes, a weakness that has been reduced by restart schemes with increasing population size, such as IPOP-CMA-ES<sup>21</sup>. Even so, CMA-ES shares a limitation with PSO and DE: all three operate on a fixed-length real-valued vector. They can optimise the weights or hyperparameters of a given network, but the topology must still be manually optimized.

Architecture search requires an encoding whose size can change during evolution. NeuroEvolution of Augmenting Topologies (NEAT)<sup>3</sup> introduced dynamic direct genetic encodings that allow the optimization of both neural network parameters and topology. Built on principles of complexifying evolution, NEAT initializes a population of relatively simple artificial neural networks (ANNs) and progressively increases their structural complexity through mutation and crossover. Each network is encoded as a flexible genome comprising node and connection genes, enabling the emergence of new graph components while maintaining direct genotype-phenotype mapping. Historical markings are assigned to structural mutations, providing a consistent record of their evolutionary origin and allowing corresponding genes to be aligned during crossover. This framework enables architecture search and weight optimization within a dynamically expanding search space, allowing increasingly complex topologies to evolve in response to fitness-driven selection.

Because NEAT involves direct mapping of a network’s elements to a genetic encoding, it is impractical for scaling to complex problems and modern deep architectures, such as convolutional neural networks. Indirect encodings address this limitation by evolving a compact rule that generates the network rather than the network itself. HyperNEAT<sup>22</sup> uses NEAT to evolve a Compositional Pattern-Producing Network (CPPN)<sup>23</sup>, which computes each connection weight from the spatial coordinates of the two neurons it joins. Because a single function generates the entire connectivity, structural motifs are reused across the network allowing large architectures to emerge from small genomes. This reuse, however, comes with a bias towards regularity, and HyperNEAT’s advantage diminishes on problems that require irregular connectivity<sup>24</sup>.

Existing NAS methods either optimize within predefined architectural search spaces, evolve network structures without explicitly incorporating biologically derived organizational patterns, or integrate biological knowledge as fixed architectural constraints. To address these limitations, we propose TRON, a neuroevolutionary framework that incorporates experimentally characterized PPI networks as architectural building blocks for artificial neural networks. TRON leverages biologically derived structural priors during ANN initialization, introducing connectivity patterns shaped by millions of years of biological evolution into the evolutionary search process. These architectures are subsequently optimized through mutation and fitness-based selection, allowing their connectivity to evolve in response to task-specific objectives. By combining the architectural flexibility of neuroevolution with the modular and hierarchical organization observed in biological networks, TRON investigates whether evolutionarily shaped connectivity patterns can provide a more robust substrate for neural architecture optimization. We evaluate this hypothesis by comparing TRON with established optimization and neuroevolutionary approaches across classification and regression benchmarks, assessing both predictive performance and evolutionary robustness.

## 2 Material and Methods

## 2.1 Datasets

The primary objective of this study is not to optimize performance for a specific application domain, but rather to evalu ate whether biologically derived architectural priors improve the effectiveness and robustness of neuroevolutionary search. Consequently, the experimental evaluation was designed to assess the generality of the proposed framework across multiple machine learning problems with different statistical properties. To this end, TRON was evaluated on a collection of widely used benchmark datasets comprising both classification and regression tasks.

Because TRON is intended to improve the evolutionary search process itself, rather than exploit task-specific biological information, benchmarks were intentionally selected from unrelated domains. This design minimizes the possibility that any performance improvements arise from a direct correspondence between the source biological pathways and the target prediction task. Furthermore, the selected tasks were intentionally unrelated to the biological origin of the PPI networks. Consequently, any observed improvements are more likely to reflect general properties of the evolutionary search process rather than domain-specific biological knowledge

For classification, we selected the CORA, CITESEER, and PUBMED datasets. These citation-network benchmarks have been extensively used for the evaluation of machine learning, graph mining, and graph neural network methods, making them among the most widely adopted benchmark datasets for node classification tasks<sup>14,</sup> <sup>25,</sup> <sup>26</sup>. CORA contains scientific publications categorized into seven research topics, CITESEER comprises six classes and exhibits greater sparsity and clas overlap, whereas PUBMED represents a larger biomedical citation network consisting of three classes. Together, these datasets provide classification problems of increasing scale and complexity and enable the evaluation of evolved neural architectures under different topological and statistical conditions.

For regression, we selected the DIABETES, FRIEDMAN, and CALIFORNIA datasets. These benchmarks are commonly employed for evaluating regression algorithms due to their diversity in sample size, dimensionality, noise characteristics, and underlying functional relationships<sup>27–29</sup>. The DIABETES dataset represents a real-world biomedical prediction problem involving physiological measurements, whereas the FRIEDMAN dataset is a synthetic benchmark specifically designed to assess the ability of machine learning models to capture nonlinear interactions in the presence of irrelevant variables. The CALIFORNIA housing dataset constitutes a larger real-world regression problem characterized by heterogeneous demographic and geographic predictors. Collectively, these benchmarks enable the assessment of evolved architectures across a broad range of continuous prediction scenarios.

The selected benchmarks collectively provide a broad evaluation framework spanning graph-base node classification, tabular classification, synthetic nonlinear regression, and real-world regression problems. This diversity is particularly important in the context of neuroevolution because the central hypothesis being tested concerns the evolvability of the architectural substrate itself rather than task-specific inductive biases. If protein-derived network topologies indeed provide a more favorable substrate for evolutionary optimization, performance improvements should emerge across multiple problem domains rather than being restricted to a single dataset type.

Furthermore, the selected datasets vary substantially in terms of dimensionality, complexity, and underlying data-generating processes. This heterogeneity makes them suitable for assessing not only predictive performance but also the robustness, stability, and generality of the evolutionary search process.

## 2.2 TRON Framework

## 2.2.1 Protein Graph Construction

The proposed framework constructs biologically informed graph structures from protein–protein interaction (PPI) datasets obtained from biological organisms. Each dataset is represented as an undirected interaction graph, in which proteins are represented as vertices and experimentally observed interactions are represented as edges. Formally,

$$
G = ( V , E ) ,
$$

where

$V = p _ { 1 } , p _ { 2 } , \dotsc , p _ { n }$ denotes the set of proteins;

$E \subseteq V \times V$ denotes the set of interaction edges.

The degree of each protein is defined as

$$
d ( p _ { i } ) = | N ( p _ { i } ) | ,
$$

where $N ( p _ { i } )$ denotes the neighborhood of protein $p _ { i }$

Following graph construction, the interaction network is represented using an adjacency matrix that captures pairwise connectivity between proteins. Let

$$
A \in \{ 0 , 1 \} ^ { | V | \times | V | }
$$

denote the adjacency matrix, where each entry is defined as

$$
A _ { i j } = { \left\{ \begin{array} { l l } { 1 , } & { ( p _ { i } , p _ { j } ) \in E , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

To guide biologically localized topology mutations, the framework constructs hierarchical neighborhood layers through recursive graph expansion. Given an initial protein $p ,$ , the zeroth layer is defined as

$$
L _ { 0 } ( p ) = \{ p \} .
$$

Subsequent layers are generated recursively as

$$
L _ { t } ( p ) = \left( \bigcup _ { u \in L _ { t - 1 } ( p ) } N ( u ) \right) \setminus \bigcup _ { i = 0 } ^ { t - 1 } L _ { i } ( p ) ,
$$

where $N ( u )$ denotes the neighborhood of protein u. This construction ensures that each protein is assigned to the layer corresponding to its shortest graph distance from the initial protein.

To obtain global mutation regions, neighborhood layers are aggregated across all proteins:

$$
P _ { t } = \bigcup _ { p \in V } L _ { t } ( p ) .
$$

These layer pools define biologically meaningful candidate regions for topology mutation. To maintain consistency between biological locality and neural architecture depth, separate layer pools are precomputed for different network depths. Consequently, deeper neural architectures can utilize correspondingly deeper graph neighborhoods during initialization and subsequent evolutionary mutations.

## 2.2.2 Graph Preprocessing

Prior to neural topology initialization, the protein interaction graphs are preprocessed to improve structural consistency, computational efficiency, and evolutionary stability. Given an initial graph $G ^ { ( 0 ) } \stackrel { \cdot } { = } ( V ^ { ( 0 ) } , E ^ { ( 0 ) } )$ ), iterative k-core filtering removes proteins with insufficient interaction connectivity. At iteration t, the node and edge sets are updated as

$$
V ^ { ( t + 1 ) } = p _ { i } \in V ^ { ( t ) } \mid d _ { G ^ { ( t ) } } ( p _ { i } ) \geq k ,\tag{1}
$$

and

$$
E ^ { ( t + 1 ) } = ( p _ { i } , p _ { j } ) \in E ^ { ( t ) } \mid p _ { i } , p _ { j } \in V ^ { ( t + 1 ) } .\tag{2}
$$

Here, $d _ { G ^ { ( t ) } } ( \boldsymbol { p } _ { i } )$ denotes the degree of protein $p _ { i }$ in graph $G ^ { ( t ) }$ . Filtering continues until convergence, i.e.,

$$
V ^ { ( t + 1 ) } = V ^ { ( t ) } ,
$$

yielding the filtered graph

$$
G ^ { \prime } = ( V ^ { \prime } , E ^ { \prime } ) ,
$$

which is subsequently used for neural architecture construction. This procedure removes weakly connected regions, produces denser interaction neighborhoods, and reduces the computational cost associated with topology exploration during evolution.

Following filtering, layer pools corresponding to different architecture depths are generated through iterative neighborhood expansion and cached for reuse during mutation. This avoids repeated graph traversal while preserving biologically localized topology modifications.

![](images/07c6ebe9549c3673308ea605f21c3abbc4ee73477ef389566c7b4bebff707f95.jpg)  
Figure 1. Protein Processing & Network Initialization

## 2.2.3 Protein-Guided Neural Initialization

Following graph preprocessing, the framework initializes neural architectures directly from the filtered protein interaction graph. Unlike conventional approaches that employ dense or randomly generated connectivity, the proposed method derives sparse neural topologies from biologically observed protein interactions. The initialization procedure consists of three stages:

1. protein-to-neuron assignment

2. connectivity initialization

3. sparse weight initialization

Let the neural architecture be defined as

$$
A = L _ { 0 } , L _ { 1 } , \dots , L _ { D } ,\tag{3}
$$

where D denotes the network depth and $L _ { i }$ denotes the set of neurons in layer i.

For each architecture depth, the framework selects the corresponding filtered interaction graph $G _ { D } = \left( V _ { D } , E _ { D } \right)$ and its precomputed layer pools. These structures define both the initial neural topology and the biologically constrained search space used during evolutionary optimization. The aforementioned process is illustrated in Figure 1

## 2.2.4 Protein-to-Neuron Mapping

Each neuron is assigned a protein sampled from the corresponding biological layer pool. Let

$$
P _ { i } \subseteq V _ { D }
$$

denote the set of proteins available for assignment to neural layer i. The mapping is defined as

$$
\phi : n _ { j } ^ { ( i ) } \mapsto p _ { k } ,\tag{4}
$$

where $n _ { j } ^ { ( i ) }$ denotes neuron j in neural layer i, and $p _ { k }$ denotes its assigned protein.

To encourage structurally informative assignments, protein sampling is weighted according to the neighborhood overlap with the subsequent layer pool:

$$
s ( p ) = ( | N ( p ) \cap P _ { i + 1 } | + \varepsilon ) ^ { 3 } ,\tag{5}
$$

where $N ( p )$ denotes the neighbors of protein $p , P _ { i + 1 }$ denotes the subsequent protein layer pool, and $\varepsilon > 0$ prevents zeroprobability sampling. The cubic weighting biases the sampling process toward proteins with a larger number of interactions with the subsequent layer, thereby increasing their probability of being selected for network construction.

For the output layer, proteins are selected from those directly connected to proteins assigned to the preceding hidden layer. This constraint preserves biological reachability across the neural architecture.

## 2.2.5 Connectivity Initialization

Following protein assignment, sparse inter-layer connectivity is constructed directly from the biological interaction graph. For two consecutive neural layers, let

$$
P _ { i } = p _ { 1 } ^ { ( i ) } , p _ { 2 } ^ { ( i ) } , \dots , p _ { n _ { i } } ^ { ( i ) } , \qquad P _ { i + 1 } = p _ { 1 } ^ { ( i + 1 ) } , p _ { 2 } ^ { ( i + 1 ) } , \dots , p _ { n _ { i + 1 } } ^ { ( i + 1 ) } .\tag{6}
$$

The connectivity matrix is defined as

$$
C ^ { ( i ) } \in \{ 0 , 1 \} ^ { n _ { i } \times n _ { i + 1 } } ,
$$

with entries

$$
C _ { u \nu } ^ { ( i ) } = \left\{ \begin{array} { l l } { { 1 , } } & { { ( p _ { u } ^ { ( i ) } , p _ { \nu } ^ { ( i + 1 ) } ) \in E _ { D } , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e } . } } \end{array} \right.\tag{7}
$$

Thus, two neurons are connected only when their assigned proteins interact within the biological network. To identify isolated neurons, the outgoing and incoming degrees are computed as

$$
d _ { \mathrm { o u t } } ( u ) = \sum _ { \nu } C _ { u \nu } ^ { ( i ) } , \qquad d _ { \mathrm { i n } } ( \nu ) = \sum _ { u } C _ { u \nu } ^ { ( i ) } .\tag{8}
$$

If a neuron has no incoming or outgoing connections, a corrective edge is introduced by selecting a biologically valid interaction from the corresponding protein neighborhood. This ensures that every neuron can participate in signal propagation while maintaining the biological connectivity constraints imposed by the interaction graph.

## 2.2.6 Weight Initialization

Weights are initialized only along biologically valid neural connections. For each connectivity matrix

$$
W ^ { ( i ) } \in \mathbb { R } ^ { n _ { i } \times n _ { i + 1 } } ,
$$

the corresponding weights are initialized as

$$
\begin{array} { r } { W _ { u \nu } ^ { ( i ) } = \left\{ \begin{array} { l l } { U ( - \alpha , \alpha ) , } & { C _ { u \nu } ^ { ( i ) } = 1 , } \\ { 0 , } & { C _ { u \nu } ^ { ( i ) } = 0 , } \end{array} \right. } \end{array}\tag{9}
$$

where $U ( - \alpha , \alpha )$ denotes a uniform distribution over the interval $[ - \alpha , \alpha ]$ . The initialization scale is defined as

$$
\alpha = \operatorname* { m i n } \left( \frac { 1 } { \sqrt { n _ { i } } } , 0 . 1 \right) .\tag{10}
$$

## 2.3 Evolutionary Framework

The proposed framework employs a hierarchical evolutionary strategy in which candidate neural architectures are organized according to their biological origin, structural characteristics, and evolutionary lineage. The evolutionary process combines species-level isolation, lineage-aware selection, parameter-level weight mutations, and protein-guided topology mutations. This organization allows the framework to explore multiple biologically constrained neural architectures concurrently while limiting premature convergence toward a single topology. The evolutionary lifecycle, specific stages, and computational cost per evolutionary stage are illustrated in Figure 2.

## 2.3.1 Species Definition

The framework employs hierarchical speciation to partition the evolutionary search space according to biological origin and neural architecture. Rather than evolving a homogeneous population, individuals are organized into biologically and structurally distinct species, enabling concurrent exploration of multiple topology configurations while preserving evolutionary diversity.

![](images/1b64ab4cf58511fc9bec7426c97c0413f7e2eac0625e1392e51d9144d82ebb7a.jpg)  
Figure 2. TRON Evolutionary Lifecycle

Biological Species At the highest level, species are defined according to the biological source of the protein–protein interaction (PPI) network. Let $\mathcal { S } _ { B }$ denote the set of biological species such that:

$$
\mathcal { S } _ { B } = S _ { 1 } , S _ { 2 } , . . . , S _ { m }\tag{11}
$$

where each biological species $S _ { k }$ corresponds to an organism-specific interaction graph

$$
G _ { k } = ( V _ { k } , E _ { k } ) ,\tag{12}
$$

with $V _ { k }$ and $E _ { k }$ denoting the corresponding proteins and interactions, respectively. Neural architectures initialized from $S _ { k }$ inherit their protein assignments and connectivity constraints from $G _ { k } .$ , thereby defining distinct biologically constrained topology search spaces.

Topology Species Within each biological species, individuals are further partitioned into topology species according to their architectural characteristics. Let

$$
T _ { 1 } ^ { ( k ) } , T _ { 2 } ^ { ( k ) } , \ldots , T _ { n _ { k } } ^ { ( k ) }\tag{13}
$$

denote the topology species associated with biological species $S _ { k }$ . Individuals are grouped according to properties such as network depth, layer dimensionality, and connectivity organization. This secondary level of speciation allows structurally distinct neural architectures to be explored concurrently while reducing direct competition between substantially different topology configurations.

Evolutionary Population and Lineage Tracking Each species maintains an evolutionary population,

$$
P _ { k } = m _ { 1 } , m _ { 2 } , \ldots , m _ { r } ,\tag{14}
$$

where $m _ { i }$ denotes an individual neural model belonging to species $S _ { k }$ . Each individual is assigned a lineage identifier based on its ancestral origin, allowing evolutionary histories to be tracked throughout optimization.

For a lineage root $l _ { i } ,$ its population size is defined as

$$
| m _ { j } \in P _ { k } \ | \mathrm { r o o t } ( m _ { j } ) = l _ { i } | .\tag{15}
$$

These lineage statistics are incorporated into survivor selection to limit the dominance of individual evolutionary lineages and preserve population diversity.

Species Adaptation Through Stagnation Detection Each species independently monitors optimization progress using its best fitness, denoted by $f _ { k } ^ { \mathrm { b e s t } } ( t )$ . Species $S _ { k }$ is considered stagnant when the number of generations since its last fitness improvement reaches a predefined threshold:

$$
t - t _ { k } ^ { \mathrm { i m p r o v e } } \geq \tau _ { \mathrm { s t a g } } ,\tag{16}
$$

where $t _ { k } ^ { \mathrm { i m p r o v e } }$ denotes the most recent generation in which the best fitness of species $S _ { k }$ improved, and $\tau _ { \mathrm { s t a g } }$ denotes the stagnation threshold.

When stagnation is detected, the species can trigger topology-level adaptation through protein-guided architectural mutations. This mechanism increases exploration of previously unexplored regions of the biologically constrained topology search space.

Quarantine Species To isolate individuals who have undergone topological changes through mutation from established populations, the framework employs temporary quarantine populations. Let

$$
Q _ { k } = q _ { 1 } , q _ { 2 } , . . . , q _ { n }\tag{17}
$$

denote the quarantine population associated with species $S _ { k }$ . Quarantine individuals are generated from high-performing members of the parent species and evolve independently for a predefined cooldown period,

$$
T _ { Q } = \tt q u a r a n t i n e \_ c o o l d o w n . \tt\tag{18}
$$

During this period, quarantine individuals are evaluated without immediately replacing members of the parent species. Following the cooldown period $T _ { Q } ,$ a quarantine individual may be reintegrated if its fitness exceeds that of an eligible member of the parent species:

$$
f ( q _ { i } ) > f ( p _ { j } ) ,\tag{19}
$$

where $q _ { i } \in Q _ { k }$ is a quarantine individual and $p _ { j } \in P _ { k }$ is an individual from the corresponding parent species.

The quarantine mechanism therefore permits large topological changes to be evaluated independently before they are allowed to influence the established evolutionary population.

## 2.3.2 Reproduction

During reproduction, a subset of each species’ members is selected to generate offspring for the subsequent generation. Two primary parameters control this process: the number of individuals eligible for reproduction and the fertility assigned to each selected individual.

• Member selection: determines the number of high-performing individuals from the current population that are eligible to reproduce. This may be implemented by selecting the top-k individuals according to fitness.

• Fitness: determines the number of offspring generated by each selected individual. Each offspring initially inherits the parent’s network parameters, connectivity, and protein assignments before undergoing mutation.

The configuration of these parameters determines the balance between exploitation and population diversity. Increasing the number of reproductive individuals increases the diversity of parental genome pool, whereas limiting fertility prevents excessive proliferation of closely related individuals and unnecessary growth in computational and memory requirements.

## 2.3.3 Mutations

The proposed framework optimizes biologically constrained neural architectures using gradient-free evolutionary search. Rather than relying on gradient-based optimization, candidate solutions evolve through repeated cycles of evaluation, selection, reproduction, and mutation. Individuals are organized into species and lineages, allowing structurally distinct architectures to evolve concurrently while maintaining population diversity.

The framework employs two complementary classes of mutation operators, summarized in Table 1.

Table 1. Mutation operators employed by the proposed framework.
<table><tr><td>Mutation Type</td><td>Purpose</td></tr><tr><td>Weight mutation</td><td>Perturbs existing connection weights.</td></tr><tr><td>Protein mutation</td><td>Modifies protein assignments and connectivity.</td></tr></table>

Weight mutations provide local parameter-level exploration, whereas protein mutations modify the underlying neural topology through changes in protein assignments. Mutation intensity is adaptively adjusted according to individual fitness and species-level stagnation, providing a mechanism for balancing local parameter refinement with larger-scale structural exploration.

## 2.3.4 Weight Mutations

Weight mutation combines several complementary operators acting at different structural scales. The mutation rate is first adjusted according to the fitness of the individual:

$$
\begin{array} { r } { \alpha ( f ) = \operatorname* { m a x } \left( e x p _ { m i n } , ( 1 - f ) ^ { 1 . 2 5 } \right) , } \end{array}\tag{20}
$$

where $e x p _ { m i n }$ denotes the minimum exploration and f denotes the model accuracy. This determines the effective mutation probability

$$
\mu = \operatorname* { m a x } \left( p _ { \operatorname* { m i n } } , p _ { \mathrm { b a s e } } \alpha ( f ) \right) .\tag{21}
$$

where $p _ { m i n }$ and $p _ { b a s e }$ denote the minumum allowed mutation rate and the base mutation rate respectively. The core weight mutation applies multiplicative and additive Gaussian perturbations:

$$
W \odot \exp ( \pmb { \varepsilon } _ { m } ) + \pmb { \varepsilon } _ { a } ,\tag{22}
$$

followed by bounded normalization:

$$
\frac { W ^ { \prime } } { 1 + | W ^ { \prime } | / r } ,\tag{23}
$$

where ⊙ denotes element-wise multiplication, r controls the effective weight range, and the perturbation terms are sampled according to

$$
\begin{array} { r } { \varepsilon _ { m } \sim \mathcal { N } ( 0 , \sigma _ { m } ^ { 2 } ) , \qquad \varepsilon _ { a } \sim \mathcal { N } \left( 0 , \sigma _ { a } ^ { 2 } ( | W | + 1 ) \right) . } \end{array}
$$

Here, $\sigma _ { m }$ and $\sigma _ { a }$ denote the standard deviations of the multiplicative and additive perturbations, respectively.

Higher-level mutation operators apply the same underlying perturbation mechanism at different structural scales. Their implementation is summarized in Table 2.

Table 2. Weight mutation operators.
<table><tr><td>Operator</td><td>Target</td><td>Probability</td><td>Operation</td></tr><tr><td>Element-wise</td><td>Individual weights</td><td>0.60</td><td> $\overline { { \mathcal { M } ( W _ { i j } ) } }$ </td></tr><tr><td>Neuron-wise</td><td>Incoming neuron weights</td><td>0.25</td><td> $\mathcal { M } ( \mathbf { w } _ { j } )$ </td></tr><tr><td>Block</td><td>Weight submatrix</td><td>0.15</td><td> $\mathcal { M } ( W _ { B } )$ </td></tr><tr><td>Layer-scale</td><td>Entire layer</td><td>0.05</td><td> $s W$ </td></tr></table>

The mutation operators are sampled independently during each generation, such that multiple operators may be applied to the same individual. Consequently, the probabilities in Table 3 represent independent activation probabilities rather than mutually exclusive categorical probabilities. The probability that none of the operators is activated is therefore

$$
\prod _ { k } ( 1 - P _ { k } ) .\tag{24}
$$

This formulation permits multiple mutation operators to act concurrently on different structural components of the same network.

## 2.3.5 Protein Mutations

Protein mutations perform topology-level adaptation by modifying the protein assignments that determine biologically constrained neural connectivity. Unlike weight mutations, protein mutations are activated when a species exhibits prolonged stagnation:

$$
t - t _ { k } ^ { \mathrm { i m p r o v e } } \geq \tau _ { \mathrm { s t a g } } ,\tag{25}
$$

where $\tau _ { \mathrm { s t a g } }$ denotes the species-level stagnation threshold.

When this condition is satisfied, a temporary quarantine population $Q _ { k } \subseteq S _ { k }$ is generated from high-performing individuals of the parent species. Each selected individual undergoes protein reassignment according to

$$
\phi ^ { ( l ) } \mapsto p _ { r } ,\tag{26}
$$

where $n _ { j } ^ { ( l ) }$ denotes neuron j in neural layer l and $p _ { r }$ denotes its assigned protein. A selected assignment is replaced according to

$$
\phi ^ { \prime } ( n _ { j } ^ { ( l ) } ) = p _ { r } ^ { \prime } , \qquad p _ { r } ^ { \prime } \in P _ { l } ,\tag{27}
$$

where P<sub>l</sub> denotes the protein layer pool corresponding to neural layer l.

The number of protein assignments subjected to mutation is determined by

$$
\operatorname* { m a x } \left( 1 , \left\lfloor N p _ { \mathrm { p r o t } } \right\rfloor \right) ,\tag{28}
$$

where N denotes the number of neurons and $p _ { \mathrm { p r o t } }$ denotes the adaptive protein mutation rate.

Following protein reassignment, neural connectivity is reconstructed from the updated protein interaction graph. The resulting architecture is then validated to ensure forward connectivity, output reachability, and consistency with the biological interaction constraints.

Mutated individuals remain in the quarantine population for the predefined cooldown period before potential reintegration into the parent species. Superior offspring can subsequently replace weaker members of the parent population, allowing substantial topological modifications to be evaluated without immediately disrupting established evolutionary lineages.

## 2.3.6 Selection and Optimization

Evolution proceeds through repeated cycles of evaluation, reproduction, mutation, and survivor selection. Each species evolves independently while maintaining lineage information and diversity-aware selection.

An individual neural architecture is represented as

$$
m _ { i } = ( \theta _ { i } , C _ { i } , P _ { i } ) ,\tag{29}
$$

where $\theta _ { i } , C _ { i }$ , and P<sub>i</sub> denote the network weights, connectivity structure, and protein assignments, respectively. Following evaluation, each individual receives a task-specific fitness score:

$$
f _ { i } = \mathcal { F } ( m _ { i } ) .\tag{30}
$$

Selection operates at two temporal scales. During ordinary generations,

$$
g \ \mathrm { m o d } \ \tau _ { \mathrm { g l o b a l } } \neq 0 ,\tag{31}
$$

offspring compete within their respective parental lineages. For lineage $L _ { j }$ , the highest-fitness individual is given by

$$
\arg \operatorname* { m a x } _ { m \in L _ { j } } f ( m ) .\tag{32}
$$

At periodic global selection intervals,

$$
g \ \mathrm { m o d } \ \tau _ { \mathrm { g l o b a l } } = 0 ,\tag{33}
$$

11/19

survivor selection is instead performed across the entire species population.

To reduce the potential for dominant evolutionary lineages to monopolize the population, global selection can use lineageadjusted fitness:

$$
f _ { i } \frac { 1 } { 1 + \alpha L _ { i } } ,\tag{34}
$$

where

$$
L _ { i } = \left| \left\{ m _ { j } \in \mathcal { P } ^ { ( t ) } : r _ { j } = r _ { i } \right\} \right|\tag{35}
$$

denotes the number of individuals in the current population that descend from the same lineage root $r _ { i } ,$ and α controls the strength of the lineage penalty.

The framework supports multiple global survivor-selection strategies, summarized in Table 3.

Table 3. Global survivor-selection strategies.
<table><tr><td>Strategy</td><td>Selection Rule</td><td>Characteristics</td></tr><tr><td>Elitism</td><td>Highest  $\widehat { f _ { i } }$ </td><td>Direct preservation of high-fitness individuals</td></tr><tr><td>Tournament</td><td> $m ^ { * } = \arg \operatorname* { m a x } _ { m \in \mathcal { T } } \hat { f } ( m )$ </td><td>Adjustable selection pressure through tournament size</td></tr><tr><td>Rank-based</td><td>Rank-dependent probability</td><td>Probabilistic selection based on population rank</td></tr></table>

For tournament selection, the tournament size may remain fixed or vary throughout evolution. For example, selection pressure can increase linearly according to

$$
k _ { \mathrm { m i n } } + ( k _ { \mathrm { m a x } } - k _ { \mathrm { m i n } } ) \frac { t } { T } ,\tag{36}
$$

or geometrically according to

$$
k _ { \mathrm { m i n } } \left( \frac { k _ { \mathrm { m a x } } } { k _ { \mathrm { m i n } } } \right) ^ { t / T } ,\tag{37}
$$

where T denotes the total number of generations. These schedules allow selection pressure to increase progressively during optimization.

Together, lineage-local replacement, lineage-aware fitness adjustment, and periodic global survivor selection provide complementary mechanisms for balancing exploitation of high-performing architectures with continued exploration of diverse topological configurations.

## 3 Results and Discussion

Since its introduction, NEAT has become one of the most influential neuroevolutionary frameworks by demonstrating that network topology and connection weights can be evolved simultaneously while preserving innovation through speciation<sup>3</sup>. The central insight of NEAT is that architectural innovation must be protected from direct competition with already established solutions. TRON adopts this principle but extends it in two important directions. First, the search process begins from biologically derived topologies rather than minimal or randomly generated architectures. Second, the introduction of species quarantine provides an additional mechanism through which newly generated topological innovations can undergo a period of maturation before entering the broader competitive population.

From this perspective, TRON differs conceptually from most existing neuroevolution approaches. Traditional methods primarily seek improved exploration through modifications of evolutionary operators, fitness functions, diversity maintenance procedures, or encoding schemes. By contrast, TRON focuses on the nature of the structures being evolved. Rather than asking how evolution can more effectively search arbitrary architectural spaces, TRON asks whether the space itself can be made more evolvable by seeding it with organizational motifs that have already been optimized through natural evolution. This distinction is closely related to the concept of evolvability, defined as the capacity of a system to generate heritable phenotypic variation that can subsequently be refined through selection<sup>30</sup>. In biological systems, evolvability is frequently associated with properties such as modularity, robustness, reuse of functional components, and hierarchical organization. Protein interaction networks exhibit many of these characteristics, having emerged through extensive evolutionary refinement. Consequently, rather than constructing candidate neural architectures from random topologies, TRON begins from connectivity motifs that have already persisted throughout biological evolution.

To determine whether these biologically derived architectural priors provide an advantage during evolutionary search, we evaluated TRON on a diverse collection of classification and regression benchmarks and compared its performance against several established evolutionary and neuroevolutionary approaches. Because the selected tasks are unrelated to the biological origin of the protein networks, any observed improvements are more likely to reflect general properties of the evolutionary search process rather than domain-specific biological knowledge.

Table 4. Algorithm performance on CORA
<table><tr><td rowspan="2">Method</td><td colspan="6">CORA</td></tr><tr><td>MIN</td><td>MAX</td><td>MEDIAN</td><td>MEAN</td><td>SD</td><td>SNR (dB)</td></tr><tr><td>CMA-ES</td><td>0.6679</td><td>0.8118</td><td>0.7634</td><td>0.7429</td><td>0.0554</td><td>22.5484</td></tr><tr><td>ES-MLP</td><td>0.6015</td><td>0.7989</td><td>0.7386</td><td>0.7336</td><td>0.0433</td><td>24.5794</td></tr><tr><td>DE</td><td>0.5119</td><td>0.5966</td><td>0.5527</td><td>0.5509</td><td>0.0264</td><td>26.3894</td></tr><tr><td>PSO</td><td>0.4207</td><td>0.5572</td><td>0.4841</td><td>0.4779</td><td>0.0415</td><td>21.2258</td></tr><tr><td>NEAT</td><td>0.4649</td><td>0.5793</td><td>0.5240</td><td>0.5189</td><td>0.0378</td><td>22.7518</td></tr><tr><td>TRON</td><td>0.8044</td><td>0.8357</td><td>0.8293</td><td>0.8262</td><td>0.0097</td><td>38.6063</td></tr></table>

Table 5. Algorithm performance on CITESEER
<table><tr><td rowspan="2">Method</td><td colspan="6">CITESEER</td></tr><tr><td>MIN</td><td>MAX</td><td>MEDIAN</td><td>MEAN</td><td>SD</td><td>SNR (dB)</td></tr><tr><td>CMA-ES</td><td>0.6698</td><td>0.7372</td><td>0.7057</td><td>0.7065</td><td>0.0205</td><td>30.7472</td></tr><tr><td>ES-MLP</td><td>0.5691</td><td>0.7282</td><td>0.7009</td><td>0.6785</td><td>0.0523</td><td>22.2610</td></tr><tr><td>DE</td><td>0.5210</td><td>0.6377</td><td>0.6055</td><td>0.5919</td><td>0.0388</td><td>23.6683</td></tr><tr><td>PSO</td><td>0.4565</td><td>0.5631</td><td>0.5248</td><td>0.5177</td><td>0.0340</td><td>23.6520</td></tr><tr><td>NEAT</td><td>0.5961</td><td>0.6396</td><td>0.6239</td><td>0.6216</td><td>0.0140</td><td>32.9477</td></tr><tr><td>TRON</td><td>0.7207</td><td>0.7357</td><td>0.7222</td><td>0.7250</td><td>0.0053</td><td>42.7212</td></tr></table>

Table 6. Algorithm performance on PUBMED
<table><tr><td rowspan="2">Method</td><td colspan="6">PUBMED</td></tr><tr><td>MIN</td><td>MAX</td><td>MEDIAN</td><td>MEAN</td><td>SD</td><td>SNR (dB)</td></tr><tr><td>CMA-ES</td><td>0.6798</td><td>0.8141</td><td>0.7868</td><td>0.7728</td><td>0.0476</td><td>24.2092</td></tr><tr><td>ES-MLP</td><td>0.6841</td><td>0.8126</td><td>0.7318</td><td>0.7396</td><td>0.0692</td><td>20.5778</td></tr><tr><td>DE</td><td>0.6526</td><td>0.7247</td><td>0.6747</td><td>0.6830</td><td>0.0228</td><td>29.5297</td></tr><tr><td>PSO</td><td>0.6385</td><td>0.6992</td><td>0.6668</td><td>0.6667</td><td>0.0221</td><td>29.5908</td></tr><tr><td>NEAT</td><td>0.6455</td><td>0.7784</td><td>0.7595</td><td>0.7508</td><td>0.0379</td><td>25.9377</td></tr><tr><td>TRON</td><td>0.8242</td><td>0.8395</td><td>0.8315</td><td>0.8313</td><td>0.0046</td><td>45.1400</td></tr></table>

Table 7. Algorithm performance on DIABETES
<table><tr><td rowspan="2">Method</td><td colspan="6">DIABETES</td></tr><tr><td>MIN</td><td>MAX</td><td>MEDIAN</td><td>MEAN</td><td>SD</td><td>SNR (dB)</td></tr><tr><td>CMA-ES</td><td>-0.2606</td><td>0.7915</td><td>0.2846</td><td>0.3037</td><td>0.3266</td><td>-0.6314</td></tr><tr><td>ES-MLP</td><td>-0.0553</td><td>0.8840</td><td>0.3336</td><td>0.3168</td><td>0.2904</td><td>0.7558</td></tr><tr><td>DE</td><td>-0.0886</td><td>0.8144</td><td>0.4316</td><td>0.3854</td><td>0.3135</td><td>1.7935</td></tr><tr><td>PSO</td><td>0.0989</td><td>0.8312</td><td>0.4307</td><td>0.4116</td><td>0.2361</td><td>4.8276</td></tr><tr><td>NEAT</td><td>0.2586</td><td>0.7855</td><td>0.4538</td><td>0.4046</td><td>0.2246</td><td>5.1123</td></tr><tr><td>TRON</td><td>0.5586</td><td>0.5676</td><td>0.5601</td><td>0.5614</td><td>0.0034</td><td>44.3559</td></tr></table>

Table 8. Algorithm performance on FRIEDMAN
<table><tr><td rowspan="2">Method</td><td colspan="6">FRIEDMAN</td></tr><tr><td>MIN</td><td>MAX</td><td>MEDIAN</td><td>MEAN</td><td>SD</td><td>SNR (dB)</td></tr><tr><td>CMA-ES</td><td>0.7449</td><td>0.8888</td><td>0.8189</td><td>0.8211</td><td>0.0507</td><td>24.1878</td></tr><tr><td>ES-MLP</td><td>0.7816</td><td>0.8965</td><td>0.8269</td><td>0.8242</td><td>0.0390</td><td>26.4994</td></tr><tr><td>DE</td><td>0.7305</td><td>0.8731</td><td>0.7965</td><td>0.7959</td><td>0.0437</td><td>25.2075</td></tr><tr><td>PSO</td><td>0.6698</td><td>0.8210</td><td>0.7411</td><td>0.7432</td><td>0.0528</td><td>22.9694</td></tr><tr><td>NEAT</td><td>0.6328</td><td>0.8067</td><td>0.7284</td><td>0.7341</td><td>0.0502</td><td>23.3010</td></tr><tr><td>TRON</td><td>0.8716</td><td>0.9205</td><td>0.8970</td><td>0.8962</td><td>0.0148</td><td>35.6429</td></tr></table>

Table 9. Algorithm performance on CALIFORNIA
<table><tr><td rowspan="2">Method</td><td colspan="6">CALIFORNIA</td></tr><tr><td>MIN</td><td>MAX</td><td>MEDIAN</td><td>MEAN</td><td>SD</td><td>SNR (dB)</td></tr><tr><td>CMA-ES</td><td>0.7121</td><td>0.7603</td><td>0.7291</td><td>0.7288</td><td>0.0184</td><td>31.9558</td></tr><tr><td>ES-MLP</td><td>0.6252</td><td>0.6936</td><td>0.6659</td><td>0.6656</td><td>0.0217</td><td>29.7351</td></tr><tr><td>DE</td><td>0.5572</td><td>0.7437</td><td>0.6511</td><td>0.6569</td><td>0.0495</td><td>22.4579</td></tr><tr><td>PSO</td><td>0.6068</td><td>0.7088</td><td>0.6643</td><td>0.6667</td><td>0.0355</td><td>25.4740</td></tr><tr><td>NEAT</td><td>0.4899</td><td>0.5962</td><td>0.5618</td><td>0.5488</td><td>0.0400</td><td>22.7471</td></tr><tr><td>TRON</td><td>0.7044</td><td>0.7527</td><td>0.7378</td><td>0.7336</td><td>0.0177</td><td>32.3497</td></tr></table>

## 3.1 Classification Performance

Classification experiments were conducted using the CORA, CITESEER, and PUBMED datasets. Across all three benchmarks, TRON consistently achieved the highest predictive performance among the evaluated evolutionary approaches (Tables 4, 5, and 6). The improvement was particularly pronounced for CORA and PUBMED, where TRON outperformed both classical evolutionary optimizers, including PSO, DE, and CMA-ES, and dedicated neuroevolutionary approaches such as NEAT. Similar advantages were observed for CITESEER despite the greater difficulty associated with the dataset.

These results suggest that biologically derived connectivity patterns can substantially improve the efficiency with which evolution explores the neural architecture search space. Rather than constructing architectures entirely through random variation, TRON begins from topological organizations that already exhibit coherent interaction patterns.

Although the specific mechanisms responsible for these improvements remain to be fully characterized, the results are consistent with the hypothesis that naturally evolved network organizations represent a more favorable substrate for evolutionary optimization than randomly generated topologies.

An additional and particularly noteworthy observation concerns the consistency of the evolutionary process. Across all classification datasets, TRON exhibited substantially lower variability across independent runs than competing approaches. Standard deviations were often several-fold lower than those observed for alternative methods, indicating that evolutionary trajectories converged toward high-performing solutions more reliably and reproducibly. This reduction in variability may be as important as the observed gains in average predictive performance. Many optimization algorithms can occasionally discover highly effective solutions through favorable stochastic trajectories. In contrast, consistently converging toward high-quality solutions across independent runs is substantially more difficult.

The results therefore suggest that the biological architectural priors employed by TRON constrain the search process toward more coherent regions of the architectural landscape, reducing sensitivity to random initialization and stochastic evolutionary events. More broadly, these observations support the idea that protein interaction networks may contribute not only useful initial topologies but also organizational properties that influence how effectively evolutionary operators act upon those topologies. Randomly generated architectures may be particularly vulnerable to the disruptive effects of mutation and recombination. In contrast, naturally evolved interaction networks may contain forms of modularity, robustness, or redundancy that allow beneficial structural features to persist across generations. This interpretation is consistent with previous work linking modularity and robustness to improved evolutionary adaptation. Modular systems tend to localize the effects of perturbations, thereby allowing innovations to accumulate without disrupting existing functionality<sup>31</sup>. Studies in evolutionary computation have similarly suggested that preserving functional building blocks can facilitate long-term adaptation by reducing the destructive effects of genetic variation. The protein interaction networks used in TRON may naturally provide such building blocks, allowing evolution to exploit organizational principles that have already demonstrated their effectiveness in living systems.

While the present study does not directly establish the mechanistic basis of these effects, the improved convergence stability observed throughout the classification benchmarks is consistent with this interpretation.

## 3.2 Regression Performance

To determine whether the observed benefits extend beyond classification problems, TRON was further evaluated on three regression benchmarks: DIABETES, FRIEDMAN, and CALIFORNIA.

On the DIABETES dataset, TRON achieved a substantial improvement over all competing methods (Table 7), exceeding the strongest baseline by approximately 36%. This result demonstrates that the advantages associated with biologically derived architectural priors are not restricted to discrete classification tasks and can generalize effectively to continuous prediction problems. As observed for the classification benchmarks, the performance gain was accompanied by remarkably low variability across runs, reinforcing the view that TRON promotes stable evolutionary optimization.

The framework also achieved the highest predictive performance on the synthetic FRIEDMAN benchmark (Table 8). Because FRIEDMAN was specifically designed to evaluate the ability of machine learning models to capture nonlinear interactions in complex search spaces, this result suggests that the benefits of protein-derived architectures are not limited to biological datasets. Instead, it indicates that the organizational principles embedded in naturally evolved interaction networks may provide generally useful inductive biases for evolutionary architecture search.

The CALIFORNIA housing benchmark produced a more nuanced outcome (Table 9). Although TRON remained among the highest-performing approaches, its advantage over the strongest competitors was considerably smaller than that observed on the previous datasets. This observation provides an important counterbalance to the otherwise consistent performance gains observed throughout the study. Rather than suggesting a fundamental limitation of the framework, it indicates that the utility of biological architectural priors is likely context dependent. Certain predictive problems may align more naturally with the organizational principles embodied within protein interaction networks, whereas others may derive more modest benefits from such structures.

Taken together, the regression experiments reveal a pattern similar to that observed for classification. Across datasets with distinct statistical characteristics, sample sizes, and underlying functional relationships, TRON either matched or exceeded the performance of all evaluated evolutionary competitors while maintaining superior robustness and reproducibility.

## 3.3 Biological Priors and Evolvability

A central hypothesis motivating this work is that protein interaction networks may represent architectural building blocks that are inherently more evolvable than randomly generated topologies. The results obtained across both classification and regression tasks are broadly consistent with this hypothesis.

Traditional neuroevolution methods primarily focus on improving the evolutionary process itself through modifications of mutation operators, crossover mechanisms, selection strategies, or diversity-preservation techniques. In contrast, TRON focuses on the substrate upon which these mechanisms operate. Rather than asking how evolution can more effectively search arbitrary architectural spaces, TRON asks whether the architectural space itself can be enriched with structures that have already undergone extensive evolutionary refinement. From this perspective, the superior performance and reduced variabilit exhibited by TRON may arise because protein interaction networks encode organizational properties that promote robustness to evolutionary perturbation. If useful functional structures are more likely to survive mutation and recombination, evolutionary adaptation can proceed more efficiently because beneficial modifications accumulate rather than being repeatedly disrupted. The markedly lower variance across independent runs is particularly consistent with this interpretation. It is important to note, however, that the present results do not fully disentangle the individual contributions of biological priors, speciation, and quarantine. Consequently, the observed improvements should not be attributed exclusively to any single component of the framework.

Future ablation studies will be necessary to quantify the specific contribution of each mechanism and to identify the precise factors responsible for the enhanced robustness observed throughout the experiments.

## 3.4 Computational Trade-Offs

The improvements in predictive performance and evolutionary stability were associated with increased computational requirements. As shown in Figures 3 and 4, TRON generally requires greater memory consumption than most of the competing approaches. This increase is expected because biologically derived interaction networks have larger memory requirements in general since they retain information about the proteins that have been assigned to the respective neurons. Furthermore, the quarantine mechanism introduces additional evaluations that increase the computational burden of the evolutionary process. On the other hand, regarding time complexity, TRON is amongst the top performers, especially when tested on large datasets like Pubmed and California. At the same time, TRON requires significantly fewer runs to verify its results, courtesy of its high SNR. From an engineering perspective, this trade-off reflects a deliberate design choice. The primary goal of TRON is to improve evolvability and search robustness rather than minimize computational cost. Consequently, the framework sacrifices some degree of efficiency in exchange for a richer architectural substrate and a more stable evolutionary search process.

Future work should investigate pathway compression strategies, sparse architectural representations, and distributed evolutionary approaches to improve scalability while retaining the benefits of biologically derived priors.

![](images/ba19f455e2fc08b4ebd08929574c0dc067462910a6e55e31756f466102ac9e9d.jpg)  
(a) Classification Tasks

![](images/b12a20771682babd648f8f46e0525187680326540c3fb7e7c1f130b5dc3ffd7e.jpg)  
(b) Regression Tasks  
Figure 3. Time complexity comparison for Evolutionary Frameworks quantified as CPU seconds

## 3.5 Limitations and Future Directions

Several limitations should be acknowledged. First, the present study focused exclusively on protein interaction networks and did not evaluate alternative biological network classes such as gene regulatory networks, signaling pathways, metabolic networks, or developmental programs. Second, the current experiments do not fully disentangle the contributions of biological priors, speciation, and quarantine. Third, although the results are consistent with the hypothesis that biological architectures improve evolvability, the specific structural properties responsible for this effect remain unknown.

Despite these limitations, the findings provide encouraging evidence that naturally evolved biological networks can serve as useful architectural priors for neuroevolution. In particular, the observed improvements in robustness and reproducibility suggest that biological organization may offer advantages extending beyond predictive performance alone.

Finally, the current implementation of TRON relies exclusively on experimentally characterized protein interaction networks. Recent advances in protein structure and interaction prediction, particularly through AlphaFold and related approaches, suggest the possibility of moving beyond known interaction maps toward predicted interaction landscapes derived directly from sequence information. Future generations of TRON may therefore exploit not only experimentally validated protein networks but also vast numbers of biologically plausible interactions inferred computationally. Under this perspective, genomes themselves become repositories of potential computational architectures, enabling neuroevolution to explore a substantially larger design space derived from the products of natural evolution.

![](images/44c43c397e98dd6646cac54c1d2191ca452edc1633b4a80b07a58833ce873d66.jpg)  
(a) Classification Tasks

![](images/c2bc9f5f8287dbfd415b3e0225c0a3eb7677be46eaaf89aef599372ed5646877.jpg)  
(b) Regression Tasks  
Figure 4. Memory Usage comparison for Evolutionary Frameworks

More broadly, these findings support the possibility that real biological networks may serve as reusable computational substrates for artificial intelligence and provide a foundation for progressively transferring increasingly complex forms of biological organization into artificial systems.

## Concluding Remarks

The present implementation of TRON relies on experimentally derived protein interaction networks to define the architectural substrate upon which neuroevolution operates. While this strategy already enables the direct reuse of connectivity patterns shaped through natural selection, it likely represents only a small fraction of the architectural information accessible from biological systems.

Recent advances in protein structure prediction have dramatically expanded our ability to infer molecular interactions directly from sequence information. In particular, AlphaFold and its multimer extensions have demonstrated the ability not only to predict protein structures with near-experimental accuracy, but also to identify and model protein–protein interactions from sequence alone<sup>32,</sup> <sup>33</sup>. Rather than restricting TRON to currently known protein interaction networks, future implementations may therefore generate architectural priors from predicted interaction landscapes derived directly from genomic sequences.

Such an approach would fundamentally expand the search space available to neuroevolution. Instead of reusing only experimentally characterized pathways, artificial neural architectures could be generated from millions of potential molecular interactions spanning diverse organisms, evolutionary lineages, and even synthetic sequences. In this scenario, evolution would no longer be limited to selecting among known biological networks, but could instead explore a vastly richer space of biologically plausible organizational motifs inferred computationally.

More speculatively, advances in protein interaction prediction may ultimately allow the direct translation of genomic information into adaptive computational architectures. Under this perspective, genomes become encodings of network organization from which artificial systems can derive architectural priors without requiring prior biological annotation. As protein structure and interaction prediction continue to improve, the boundary between biological sequence information and computational architecture design may progressively diminish, opening new opportunities for transferring the products of natural evolution into artificial intelligence systems.

## References

1. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436–44, DOI: 10.1038/nature14539 (2015).

2. Elsken, T., Metzen, J. H. & Hutter, F. Neural architecture search: A survey. Journal of Machine Learning Research 20, 1–21 (2019).

3. Stanley, K. O. & Miikkulainen, R. Evolving neural networks through augmenting topologies. Evolutionary Computation 10, 99–127, DOI: 10.1162/106365602320169811 (2002).

4. Floreano, D., Dürr, P. & Mattiussi, C. Neuroevolution: from architectures to learning. Evolutionary Intelligence 1, 47–62, DOI: 10.1007/s12065-007-0002-4 (2008).

5. Hartwell, L. H., Hopfield, J. J., Leibler, S. & Murray, A. W. From molecular to modular cell biology. Nature 402, C47–C52, DOI: 10.1038/35011540 (1999).

6. Barabási, A.-L. & Oltvai, Z. N. Network biology: understanding the cell’s functional organization. Nature Reviews Genetics 5, 101–113, DOI: 10.1038/nrg1272 (2004).

7. Wagner, G. P., Pavlicev, M. & Cheverud, J. M. The road to modularity. Nature Reviews Genetics 8, 921–931, DOI: 10.1038/nrg2267 (2007).

8. Spirin, V. & Mirny, L. Modular organization of cellular networks. Proceedings ofthe National Academy ofSciences 100, 12123–12128, DOI: 10.1073/pnas.2032324100 (2003).

9. Stelzl, U., Worm, U., Lalowski, M. et al. A human protein-protein interaction network: A resource for annotating the proteome. Cell 122, 957–968, DOI: 10.1016/j.cell.2005.08.029 (2005).

10. Rolland, T., Ta¸san, M., Charloteaux, B. et al. A proteome-scale map of the human interactome network. Cell 159, 1212–1226, DOI: 10.1016/j.cell.2014.10.050 (2014).

11. Luck, K., Kim, D., Lambourne, L. et al. A reference map of the human binary protein interactome. Nature 580, 402–408, DOI: 10.1038/s41586-020-2188-x (2020).

12. Liu, H., Simonyan, K. & Yang, Y. DARTS: Differentiable architecture search. In International Conference on Learning Representations (ICLR), DOI: https://doi.org/10.48550/arXiv.1806.09055 (2019).

13. Zela, A., Elsken, T., Saikia, T. et al. Understanding and robustifying differentiable architecture search. In International Conference on Learning Representations (ICLR) (2020).

14. Kipf, T. N. & Welling, M. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR) (2017).

15. Li, Q., Han, Z. & Wu, X.-M. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 32, 3538–3545, DOI: 10.1609/aaai.v32i1.11604 (2018).

16. Elmarakeby, H. A., Hwang, J., Arafeh, R. et al. Biologically informed deep neural network for prostate cancer discovery. Nature 598, 348–352, DOI: 10.1038/s41586-021-03922-4 (2021).

17. Fortelny, N. & Bock, C. Knowledge-primed neural networks enable biologically interpretable deep learning on single-cell sequencing data. Genome Biology 21, 190, DOI: 10.1186/s13059-020-02100-5 (2020).

18. Kennedy, J. & Eberhart, R. Particle swarm optimization. In Proceedings ofICNN’95 – International Conference on Neural Networks, vol. 4, 1942–1948, DOI: 10.1109/ICNN.1995.488968 (1995).

19. Storn, R. & Price, K. Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces. Journal ofGlobal Optimization 11, 341–359, DOI: 10.1023/A:1008202821328 (1997).

20. Hansen, N. & Ostermeier, A. Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation 9, 159–195, DOI: 10.1162/106365601750190398 (2001).

21. Auger, A. & Hansen, N. A restart CMA evolution strategy with increasing population size. In 2005 IEEE Congress on Evolutionary Computation (CEC), vol. 2, 1769–1776, DOI: 10.1109/CEC.2005.1554902 (2005).

22. Stanley, K. O., D’Ambrosio, D. B. & Gauci, J. A hypercube-based encoding for evolving large-scale neural networks. Artificial Life 15, 185–212, DOI: 10.1162/artl.2009.15.2.15202 (2009).

23. Stanley, K. O. Compositional pattern producing networks: A novel abstraction of development. Genetic Programming and Evolvable Machines 8, 131–162, DOI: 10.1007/s10710-007-9028-8 (2007).

24. Clune, J., Stanley, K. O., Pennock, R. T. et al. On the performance of indirect encoding across the continuum of regularity. IEEE Transactions on Evolutionary Computation 15, 346–367, DOI: 10.1109/TEVC.2010.2104157 (2011).

25. Sen, P., Namata, G., Bilgic, M. et al. Collective classification in network data. In AI Magazine, vol. 29, 93–106 (2008).

26. Yang, Z., Cohen, W. & Salakhutdinov, R. Revisiting semi-supervised learning with graph embeddings. ICML (2016).

27. Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. Least Angle Regression, vol. 32 (2004).

28. Friedman, J. H. Multivariate adaptive regression splines. The Annals ofStatistics 19, 1–67, DOI: 10.1214/aos/1176347963 (1991).

29. Pace, R. K. & Barry, R. Sparse spatial autoregressions. Statistics and Probability Letters 33, 291–297, DOI: 10.1016 S0167-7152(96)00140-X (1997).

30. Kirschner, M. & Gerhart, J. Evolvability. Proceedings of the National Academy of Sciences 95, 8420–8427, DOI: 10.1073/pnas.95.15.8420 (1998).

31. Wagner, A. Robustness and evolvability: A paradox resolved. Proceedings of the Royal Society B 275, 91–100, DOI: 10.1098/rspb.2007.1137 (2008).

32. Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate protein structure prediction with alphafold. Nature 596, 583–589, DOI: 10.1038/s41586-021-03819-2 (2021).

33. Bryant, P., Pozzati, G. & Elofsson, A. Improved prediction of protein–protein interactions using alphafold2. Nature Communications 13, 1265, DOI: 10.1038/s41467-022-28865-w (2022).