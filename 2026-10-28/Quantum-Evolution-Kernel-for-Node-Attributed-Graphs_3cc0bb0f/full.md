# Quantum Evolution Kernel for Node-Attributed Graphs

Manuel Padovani, Alessandra Di Pierro, Claudia Dafara

Department of Computer Science, University of Verona, Verona, Italy

{manuel.padovani, alessandra.dipierro, claudia.daffara}@univr.it

## Abstract

We present a novel approach to embed node-attributed graphs onto neutralatom quantum computers, and propose an application to digit recognition. Our approach extends PASQAL’s quantum evolution kernel (QEK) by introducing a node-level feature encoding based on three-level quantum states. To this purpose, we present a prototype showing how the resulting embedding can be leveraged to extract correlation-based information for classification tasks. The entire quantum process is simulated using Pasqal’s Pulser library, which inevitably imposes constraints on the size of the experimental datasets. We compare our method with classical baselines to evaluate its potential. We show that, for the digit recognition task, our method achieves a higher mean accuracy than the more commonly used Jensen-Shannon distance-based methods by utilizing the Wasserstein distance.

Keywords—quantum computation, neutral-atom quantum processor, graph machine learning, distance-based kernels, Wasserstein distance, node-attributed graphs

## 1 Introduction

Graph machine learning is a well-studied topic, given the large number of real-world problems that can be naturally represented as graphs. Classically, many kernels have been proposed to capture structural information, such as Weisfeiler-Lehman (WL) graph kernels [1], random walk kernels, shortest-path kernels, graphlet kernels, and many others [2]. However, these methods typically ignore continuous node features, which play an important role in node-attributed graphs [3], with applications ranging from molecular property prediction to handwritten character recognition.

To address this limitation, a major change occurred with the introduction of Graph Convolutional Networks (GCNs) [4], which generalize convolutional neural networks to graph data by naturally incorporating node-level features, encouraging deep learning techniques on graph-structured data, including handwritten character recognition [5, 6]. However, GCNs are theoretically bounded by the 1-WL isomorphism test, limiting their expressiveness in distinguishing non-isomorphic graphs [7]. Furthermore, GCNs sufer from over-smoothing as the number of layers increases [8], and training becomes computationally expensive as the graph size grows. Thus, exploring diferent paradigms is of paramount interest. In particular, a growing attention for analyzing graph data is shown in quantum machine learning, a machine learning paradigm that leverages quantum mechanics principles [9].

Several comprehensive reviews have surveyed this topic [10, 11], where both digital and analog approaches have been explored. Digital approaches typically leverage quantum neural networks (QNNs) implemented in gate-based quantum computers [12], also incorporating node-level attributes [13]. Analog quantum platforms, particularly neutralatom devices, have shown great potential for graph learning due to their natural mapping of graph structures onto atomic arrays [14, 15], especially in the context of quantum kernels. More recently, node-feature encoding strategies for analog quantum devices have also been investigated, including recent work by the Pasqal group [16].

Despite these recent advances, the incorporation of multiple continuous node attributes into analog quantum graph kernels remains a challenging and mostly unexplored problem. Nevertheless, they constitute a particularly efective approach for graph learning.

In this work, the focus is on quantum kernel methods [17], since they have shown promising performance in the low-data regime [18] and are particularly well suited for an analog computing paradigm with neutral-atom quantum computers in the NISQ era. Our primary objective is to assess the architectural feasibility of multi-attribute continuous encoding on analog hardware, providing a baseline for future large-scale implementations.

## 1.1 Contribution

We propose a quantum graph kernel method, based on node-level feature encoding. This method is designed for Pasqal’s Rydberg atom QPUs, whose flexible topology allows a natural mapping between graph nodes and physical atoms [19]. Following the Quantum Evolution Kernel (QEK) framework [14], the graph structure is encoded onto the atomic register via successive layers of quantum evolution, preserving topology through atom positioning and the Rydberg blockade mechanism.

The main contribution of this paper is the separate encoding of node-level features via local Raman laser pulses. This extends the Pasqal’s approach in [16], which only utilizes Rydberg laser pulses for encoding both the graph’s structure and the node-level features. In addition to the Rydberg state, we exploit the two hyperfine ground states of each atom, leading to a three-level (qutrit) system per atom. The long coherence times of the hyperfine states make them suitable for encoding node attributes, enriching each local Hilbert space to represent both local and structural features.

To define our kernel we use the Wasserstein distance on probability distributions, obtained by measuring correlation and occupation observables, instead of the most commonly used Jensen-Shannon metric, which we show to be less efective on the digit recognition task.

We evaluate our approach on the MNIST dataset, comparing against classical graph machine learning baselines [20].

## 2 Method

## 2.1 Pulses and Evolution

Before introducing the quantum evolution, we briefly describe the physics of neutral atom arrays as implemented in modern devices such as Pasqal’s platform [15]. A neutral atom used for quantum computation has three relevant states: two low-energy hyperfine ground states, $| g \rangle$ and $| h \rangle$ , and a high-energy Rydberg state, |r⟩, used to enable interactions between nearby atoms. Two atoms both in the Rydberg state experience a van der Waals repulsion scaling as $1 / R ^ { 6 }$ , R being the interatomic distance. This generates the Rydberg blockade efect: below a threshold distance $R _ { b }$ , the doubly-excited state $| r r \rangle$ becomes energetically disfavored, which is used to create entanglement between nearby atoms. For an array of atoms, the interaction Hamiltonian is

$$
H _ { \mathrm { i n t } } = \sum _ { i < j } \frac { C _ { 6 } } { R _ { i j } ^ { 6 } } \hat { n } _ { i } \hat { n } _ { j } ,
$$

where $\hat { n } _ { i } = | r \rangle _ { i } \langle r | , R _ { i j }$ is the interatomic distance between atoms i and $j ,$ and $C _ { 6 }$ depends on the atomic species and Rydberg level. This allows the encoding of any unit-disk graph<sup>1</sup> by positioning atoms such that connected nodes have $R _ { i j } \le R _ { b }$ and unconnected nodes have $R _ { i j } > R _ { b } ,$ so that the system’s evolution naturally encodes the graph structure.

Apart from the interaction term, we can act on the neutral atoms via two types of lasers: the Raman laser [15] which couples the two low-energy states, $| g \rangle$ and $| h \rangle$ ; the Rydberg laser which couples the low-energy state $| g \rangle$ with the Rydberg state $| r \rangle$ . The general Hamiltonian applied by each laser is:

$$
H = - \frac { \delta } { 2 } \sigma _ { z } + \frac { \Omega } { 2 } \left[ \cos ( \phi ) \sigma _ { x } + \sin ( \phi ) \sigma _ { y } \right] + H _ { \mathrm { i n t } } ,
$$

where there are three time-dependent parameters: the detuning (δ), the Rabi frequency (Ω), and the phase of the laser $\left( \phi \right)$ ; while the $\sigma _ { i }$ are the Pauli matrices. These parameters allow high flexibility in the possible transformations.

## 2.2 Mapping Node-Feature Graphs to Neutral Atoms

Given a graph $G = ( V , E )$ with node features organized as a matrix $X \in \mathbb { R } ^ { | V | \times d }$ , where d is the number of local features, we first encode the graph structure in the atoms’ positions using $H _ { \mathrm { i n t } }$ , as described above.

The second step is encoding the node features. Each node’s feature vector is mapped to a state obtained by applying rotations to the qubit corresponding to the $| g \rangle , | h \rangle$ levels of the neutral atom associated with the node. Specifically, we use the laser parameters $\left( \delta , \ \Omega , \ \phi \right)$ to implement rotations $R _ { X } ( \theta _ { 1 } )$ and $R _ { Y } ( \theta _ { 2 } )$ on the Bloch sphere, where $\theta _ { i }$ are functions of the feature values. Since features may lie in arbitrary ranges, we first normalize them to an appropriate interval $( \mathrm { e . g . , ~ } [ 0 , 2 \pi ] \mathrm { ~ o r ~ } [ 0 , \pi ] )$ suitable for the rotation gates.

If edge features are present, these can be encoded into the interatomic distances $R _ { i j }$ by linearly mapping a feature in $[ \boldsymbol { a } _ { \mathrm { m i n } } , \boldsymbol { a } _ { \mathrm { m a x } } ]$ to a distance in $[ d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } ]$ , where $d _ { \mathrm { m i n } }$ depends on the optical tweezers and $d _ { \operatorname* { m a x } } < R _ { b } ,$ so that edge features modulate the interaction strength via $C _ { 6 } / R _ { i j } ^ { 6 }$ . However, we do not evaluate the impact of this mechanism experimentally in this work.

In summary, graph structure is encoded via atom positions, node features via local laser rotations, and edge features via interatomic distances.

Enabling interactions requires exciting the system to the Rydberg states via Rydberg lasers, and since Rydberg coherence times are short, evolution times must remain brief.

The final encoding of each graph is obtained as

$$
\begin{array} { r l r } {  { | \Psi ( t ) \rangle = U _ { \mathrm { i n t } } U _ { \mathrm { l o c a l } } \ | g \rangle ^ { \otimes | V | } , \quad \mathrm { w h e r e } } } \\ & { } & { \displaystyle U _ { \mathrm { l o c a l } } = \prod _ { i = 1 } ^ { | V | } R _ { X } ^ { i } ( X _ { i , 1 } ) R _ { Y } ^ { i } ( X _ { i , 2 } ) \quad \mathrm { a n d } } \\ & { } & { \displaystyle U _ { \mathrm { i n t } } = \mathcal { T } \exp ( - i \int _ { 0 } ^ { t } ( H _ { \mathrm { i n t } } + H _ { \mathrm { R y d b e r g } } ( \tau ) ) d \tau ) . } \end{array}\tag{1}
$$

The rotations $R _ { X } ^ { i }$ and $R _ { Y } ^ { i }$ in $U _ { \mathrm { l o c a l } }$ are applied to the i-th neutral atom and are generated by Raman lasers; $H _ { \mathrm { R y d b e r g } }$ operator in $U _ { \mathrm { i n t } }$ represents the action of the Rydberg laser with constant Rabi frequency and constant detuning over the evolution time t, and $\tau$ is the time-ordering operator.<sup>2</sup> Since a single qubit encodes at most two real parameters, d node features yield $d / 2$ distinct quantum states per graph, each corresponding to a diferent feature pair: the first state encodes $( X _ { i , 1 } , X _ { i , 2 } )$ for all nodes, the second $( X _ { i , 3 } , X _ { i , 4 } )$ , and so on.

## 2.3 Distributions From Measurements

After obtaining the final state for each graph, we extract information by sampling the state multiple times. <sup>3</sup> Because typical neutral-atom hardware cannot distinguish all three atomic states directly, we use ground-Rydberg basis measurements, assigning bit 0 to either low-energy state and bit 1 to the Rydberg state.

Note that, since node features are encoded as population transferred between $| g \rangle$ and $| h \rangle$ while only the $| g \rangle$ component is coupled to $| r \rangle$ by the Rydberg laser, the local encoding efectively modulates the population available for Rydberg excitation: probability mass moved into $| h \rangle$ is protected from transitioning to $| r \rangle$ , thus reducing the atom’s contribution to the interaction-driven entanglement and the excitation readout.

Similarly to pooling in GCNs, this outcome is aggregated into a size-invariant, graphlevel representation.

We construct two distribution types: (i) for a bit string of length $| V |$ with Hamming weight n, we consider $f = n / | V | \in [ 0 , 1 ]$ and estimate $p ( f )$ as the fraction of sampled strings with that weight, binned into a discrete distribution; (ii) we consider correlations $\langle Z _ { i } Z _ { j } \rangle - \langle Z _ { i } \rangle \langle Z _ { j } \rangle$ , as in [16], similarly binned and normalized. Both distributions are scale-invariant with respect to $| V |$ . For each graph we thus obtain one excitation and one correlation distribution per encoded feature pair.

## 2.4 Distribution-based Quantum Kernel

As mentioned before, we aim to define a kernel based on the final data representation, namely distributions. Given a space S, a kernel is a symmetric, non-negative valued function $K : S \times S  \mathbb { R } ^ { + }$ that takes two objects and outputs a real number measuring their similarity.

In quantum machine learning, a quantum kernel is typically defined as the inner product $K ( | \phi \rangle , | \psi \rangle ) = | \langle \phi | \psi \rangle | ^ { 2 }$ between embedded states, maximized at 1 for identical states and minimized at 0 for orthogonal ones. This definition is not directly applicable here, since each graph is mapped onto a collection of states producing multiple distributions $\{ P _ { i } ^ { G } \} ;$ ; the kernel itself is computed classically from these distributions, with "quantum" referring to the fact that these originate from measurements on quantum states.

Several distance measures exist between probability distributions. Entropy-based distances such as the Jensen-Shannon distance (JS), a symmetrization of the Kullback-Leibler divergence whose square root is a true metric, are favored but cannot capture geometric diferences. $\mathrm { A }$ family of distances that preserve geometric information are the Wasserstein metrics [21,22]. The Wasserstein distance of order $p$ between two probability distributions $P , Q$ is defined as:

$$
W _ { p } ( P , Q ) = \biggl ( \operatorname* { i n f } _ { \mu \in \Pi ( P , Q ) } \int d ( x , y ) ^ { p } d \mu ( x , y ) \biggr ) ^ { \frac { 1 } { p } } ,\tag{2}
$$

where $\Pi ( P , Q )$ is the set of joint distributions with marginals $P , Q \ [ 2 2 ]$ . In our experiments, we will use $W _ { 1 }$ , also known as the Earth Mover’s Distance, which measures the minimum work to transform $P$ into $Q$ . Unlike JS, $W _ { 1 }$ requires no binning, avoiding discretization losses, at the cost of higher time complexity (O(m log m) vs. $O ( m )$ [23]), where m is the number of shots. This distance has already been used for the definition of graph kernels in the classical setting [24].

For a given distance $D ,$ we can define a kernel as:

$$
K ( P _ { k } ^ { G _ { 1 } } , P _ { k } ^ { G _ { 2 } } ) = \exp \left( - \gamma D ( P _ { k } ^ { G _ { 1 } } , P _ { k } ^ { G _ { 2 } } ) \right)\tag{3}
$$

where $\gamma > 0$ is a scale parameter. However, this is not automatically a positive semidefinite (PSD) kernel for any distance. In our setting, since the distributions are onedimensional, $W _ { 1 }$ is known to be conditionally negative definite, which guarantees that the kernel is PSD.

Since each graph produces multiple distributions $\{ P _ { k } ^ { G } \} _ { k = 1 } ^ { N _ { \mathrm { d i s t } } }$ , we combine the individual kernels using a weighted sum:

$$
K _ { \mathrm { s u m } } ( G _ { 1 } , G _ { 2 } ) = \sum _ { k = 1 } ^ { N _ { \mathrm { d i s t } } } \omega _ { k } \exp { \left( - \gamma D ( P _ { k } ^ { G _ { 1 } } , P _ { k } ^ { G _ { 2 } } ) \right) }\tag{4}
$$

where $\omega _ { k } \geq 0$ are optimizable weights such that $\textstyle \sum _ { k } \omega _ { k } = 1$ , allowing the model to favor the most informative distributions; $P _ { k } ^ { G _ { j } }$ is the k-th distribution associated with the $j -$ th graph, and $N _ { \mathrm { d i s t } } = d$ is the total number of distributions (equal to the number of node-level features $d )$ . Furthermore, we consider a second weighted combination:

$$
K _ { \mathrm { p r o d } } ( G _ { 1 } , G _ { 2 } ) = \exp \left( - \gamma \sum _ { k = 1 } ^ { N _ { \mathrm { d i s t } } } \omega _ { k } D ( P _ { k } ^ { G _ { 1 } } , P _ { k } ^ { G _ { 2 } } ) \right)\tag{5}
$$

which sums distances before exponentiating, making it a stricter, more sensitive similarity measure.

The resulting kernel is used within a Support Vector Classifier (SVC) to find the margin-maximizing hyperplane, employing a standard one-vs-one strategy for multi-class classification.

For each binary classifier, we jointly optimize $\gamma ,$ the regularization term $C .$ , and the weights $\omega _ { k }$ in (4) via nested cross-validation: an outer grid search over $\gamma ~ ( 5$ log-spaced values in [0.1, 100]), with $\omega _ { k }$ optimized inside via Bayesian optimization (Gaussian process surrogate, expected improvement, inner 3-fold CV selecting $C \in \{ 0 . 1 , 1 . 0 , 1 0 . 0 \} )$ , and a final outer-loop selection of $C \in \{ 0 . 1 , 1 . 0 , 1 0 . 0 , 1 0 0 . 0 \}$ . The triplet $( \gamma , \omega , C )$ maximizing macro-averaged F1 is retained. Note that the outer-loop C is the one reported, while inner-loop values are used only while optimizing ω.

## 2.5 Graph Embedding

The quantum process is carried out using Pasqal’s Pulser library [25], a Python library based on QuTip to specifically simulate neutral-atom quantum computers. The virtual device used in this work is pulser.DigitalAnalogDevice, which supports both Raman and Rydberg lasers. Furthermore, preprocessing must respect device constraints, notably the 2D unit-disk topology and a minimum interatomic distance of $4 ~ \mu \mathrm { m }$ (full constraints can be found in the Pasqal documentation). To embed the graph onto the quantum device, we find ideal atom positions using a standard force-directed spring model that balances attractive forces between connected nodes and repulsive forces among all pairs. Node positions are updated iteratively until an equilibrium configuration is reached. When edge features are present, the forces are calibrated so that the equilibrium target distance is close to $d _ { i j } ^ { \mathrm { t a r g e t } }$ , which is obtained, in the interval $[ 4 . 5 , 6 ] \mu m$ , via the linear mapping explained in Section 2.2 . Otherwise, all connected pairs use a constant target distance $d _ { \mathrm { m i n } }$ . The preprocessing ends by checking the quality of the transformation. We filter graphs based on three criteria: $| V | \le 9$ (computational reasons), $\mathrm { M S E } \le 1 . 5$ between target and achieved interatomic distances over connected pairs, and maximum distance from ${ \mathrm { c e n t e r } } \leq 5 0 . 0 \ \mu \mathrm { m }$ . Graphs that do not meet these criteria are discarded.

## 2.6 Experiments

We have prepared a dataset based on the MNIST dataset [26], consisting of 28x28 pixel images of handwritten digits across all 10 digit classes, by transforming each image into a graph.<sup>4</sup> This procedure is partially inspired by the preprocessing used in [5], which applies a graph machine learning model to handwritten Chinese characters.

The process can be briefly explained as follows (Fig. 1):

• Binarization: all pixels are mapped to 0 (black pixel) or 1 (white pixel);

• Skeletonization: the stroke width is reduced to a 1-pixel-wide stroke;

• Dense graph: each pixel is associated with a node, and links between nearby pixels are formed;

• Merge close nodes: nodes, which are at a distance (measured in pixels) below a given threshold, are merged together;

• Remove collinear nodes: consecutive edges that form an almost-flat angle are merged by removing the intermediate node. This step further reduces the number of nodes, thus reducing the computational overhead.

MNIST Digit-to-Graph Pipeline: From Raw Image to Spring-Relaxed Graph

![](images/1c438ecd2e4a98b5b838d6b936253b6bc21b100a5fac931fcc4ade3f736a55c1.jpg)  
Figure 1: The picture shows the pipeline from the raw image to the final graph encodable onto the quantum device.

After filtering, as described in section 2.5, the retained portion of the dataset is approximately 80% (48,018/60,000 training samples and 8,100/10,000 test samples). However, due to simulation constraints, only a small subset is used during training and evaluation.

Regarding node-level features, we compute four values for each node: the original pixel coordinates $( x _ { \mathrm { p i x e l } } , y _ { \mathrm { p i x e l } } )$ (linearly scaled to $[ 0 , \pi / 2 ] ^ { 2 }$ via a uniform rescaling that preserves the original aspect ratio), the deviation angle of consecutive edges, and the direction of the bisector of those edges. The deviation angle for a given pair of edges is defined as $| \pi - \theta |$ , where θ is the angle between the two edge vectors. The direction is the angle of the bisector vector between the two edges, normalized to the range $[ 0 , \pi )$ . For nodes with exactly two neighbors, we compute a single pair of features (deviation, direction) from the two incident edges. For nodes with three or four neighbors, we consider only the consecutive edge pairs with the largest angles: for three neighbors we take the single pair with the largest angle; for four neighbors we take the two pairs with the largest angles and sum their deviations and directions. For nodes with five or more neighbors, we take the three pairs with the largest angles. For nodes with zero or one neighbor, we set deviation to 0.9π and direction to 0.

The restriction of the coordinates to $[ 0 , \pi / 2 ]$ ensures that only a partial population transfer from $| g \rangle$ to $| h \rangle$ occurs. If the population were completely moved to $| h \rangle$ , that node would be disconnected from the Rydberg laser interaction (which only couples $| g \rangle$ to |r⟩), potentially destroying the topology information encoded in the final state.

The evolution time t in (1) is selected via Bayesian optimization using the macroaveraged F1-score as the objective. For each candidate t, we compute the quantum distributions, split the data into training (80%) and validation (20%) sets, train SVM classifiers, and evaluate the F1-score on the validation set. To keep the optimization feasible, this procedure is performed on a 200-sample subset of the MNIST training data. The optimization explores the range $t \in [ 2 5 , 1 0 0 0 ]$ ns with 30 iterations and 5 initial random points. The value of t that maximizes the validation F1-score is then selected for the final model. As shown in Fig. 2, the optimal value is t = 240.1 ns.

![](images/675c357a16987951ac7db6e1c887ccb20a6cb8d6fa2cb3ccdb50f0b9b86793ce.jpg)  
Figure 2: Bayesian optimization of the evolution parameter t using the validation F1- score (80% training, 20% validation) on a 200-sample subset of the MNIST dataset. The optimal value is $t = 2 4 0 . 1$ ns (highlighted). For small t, the system does not generate suficient entanglement, yielding low F1-score. For large t, long excitations cause oscillations between entangled states, leading to a sharp drop.

The quantum kernel model is compared to its version without node-level features to assess the contribution of local information, and we test diferent distances for the kernel (Wasserstein, Jensen-Shannon, Hellinger, and L2) to determine which one is more efective with our output distributions. For all metrics except Wasserstein, the occupation distributions are discretized into 10 bins (comparable to the typical number of nodes in our graphs, which is at most 9), while correlation distributions use 20 bins, as correlation values are concentrated near zero, especially for small evolution times, so finer binning helps increase the resolution. While all the quantum kernels are aggregated via the weighted sum in (4), for the Wasserstein metric we also implement the product aggregation in (5) to test the influence of the aggregation strategy.

We tested our method against classical baselines, such as Weisfeiler-Lehman (WL) graph kernels (implemented with GraKeL [27]). Note that these WL kernels use GraKeL’s normalization, which reduces but may not fully eliminate graph-size bias, while our quantum distributions are size-invariant by construction. We evaluated diferent node label configurations (no features, positional only, angle only, combined), and a graph convolutional network (GCN). For the positional and angle configurations, node labels are obtained by discretizing the corresponding features into bins, while the combined configuration uses both. The GCN is implemented with PyTorch Geometric [28] using two convolutional layers, batch normalization, and dropout to add regularization. Its input features are the same four node-level features used in our quantum kernel, but without the rescalings.

All models are evaluated using diferent training set sizes, with five random seeds for each size to assess robustness in the low-data regime, and on a fixed balanced test set of 15 samples per class, reporting accuracy as the primary metric.

## 3 Results

## 3.1 Comparison of Distance Metrics

In Fig. 3 we show the accuracy as a function of the training samples per class for the quantum kernel using diferent distance metrics. The kernel based on Wasserstein distance achieves the highest mean accuracy for all training sizes, with both aggregation strategies producing comparable results; we therefore use the sum aggregation in the following comparison with classical baselines. The Jensen-Shannon distance performs slightly worse, and increasing the number of bins to 30 (way higher than the typical number of nodes in our graphs) leads to a drop in performance compared to the 10- bin version. In contrast, the Hellinger and L2 distances perform poorly, with accuracy even decreasing as the training set size increases, suggesting that these metrics are not suitable for capturing the separation between distributions in our setting. Finally, the Wasserstein kernel without node-level features leads to better performance than Hellinger and L2, but performs worse than all other cases that include local features. Thus, our simple feature encoding actually leads to better results, and seems promising for more complex applications.

![](images/a3ec078fc146b9f94f8bd039038f5de41a1c879ddf7b664a8c43b312efc8b01f.jpg)  
Figure 3: Comparison of the performances of diferent distance metrics. Wasserstein achieves the highest accuracy. Shaded regions indicate ±1 standard deviation over 5 seeds.

## 3.2 Comparison with Classical Baselines

Fig. 4 compares our quantum kernel (Wasserstein distance with sum aggregation) against classical baselines. The GCN achieves the highest mean accuracy, as expected. Among the WL kernels, the combined positional-angle variant performs best, slightly exceeding the mean accuracy of our quantum kernel. However, our quantum kernel matches the positional WL variant and yields a higher mean performance than both the angle-only and featureless WL kernels. This confirms the relevance of node-level features, with positional information playing a more critical role for this task.

![](images/801285bcd6b6d2f5e850b04137b2a84aac2d8c6d74764a867a700fac4433f483.jpg)  
Figure 4: Comparison of our quantum kernel against classical baselines. The GCN achieves the highest accuracy, while our quantum kernel performs comparably to the positional WL variant and outperforms both angle-only and featureless WL kernels. Shaded regions indicate ±1 standard deviation over 5 seeds.

## 4 Related Work

Our work takes inspiration from Pasqal’s quantum evolution kernel (QEK) [14,19], which uses neutral-atom (analog) quantum computers to naturally embed graph structures. QEK applies successive layers of mixing and natural evolution to encode graphs, then constructs distributions from Rydberg excitations. Djellabi et al. [16] extended this to attributed graphs by encoding node features into local detuning fields, introducing the generalized-distance quantum-correlation kernel based on the local correlation matrix. Both approaches are based on two-level states and use only Rydberg lasers.

In contrast, our approach introduces local Raman laser pulses which couple the hyperfine ground states. This produces a three-level system for each atom, allowing for a richer Hilbert space, where node-features are encoded separately from the graph topology. We also use a diferent metric (Wasserstein), which allows us to obtain more accurate results compared to the Jensen-Shannon distance used in QEK.

## 5 Conclusion

In this paper, we proposed a quantum kernel method for node-attributed graphs embedded on neutral-atom quantum processors. We presented a simple prototypical circuit that encodes both structural and local feature information into the quantum evolution, and we showed how measurement distributions can be leveraged to define a kernel for classification.

Our experimental results on the MNIST dataset demonstrated that the Wasserstein distance is the most efective metric among those tested, as its ability to bypass binning ofers a practical advantage over the commonly used Jensen-Shannon distance. The quantum kernel achieved competitive performance with respect to selected classical graph kernels in a low-data regime, while the version without node-level features performed worse, confirming the importance of local information. The simplicity of our circuit, which uses a single encoding and executes the evolution layer for a very short time, did not allow us to surpass the mean performance of the WL combined classical kernel or the GCN. More refined designs with multiple, more elaborate encodings and longer evolution times could further improve performance, making this a promising direction to naturally encode node-attributed graphs. Interestingly, the optimizable weights $\omega _ { k }$ of our kernel could be interpreted as a measure of relevance for the distributions, potentially ofering insight into the choice of the graph’s features that are most useful for the classification task. Regarding scalability, the encoding time on real quantum hardware for each graph scales linearly with the number, $O ( | V | )$ , of nodes, since preparing the atomic register and addressing atoms with local Raman pulses takes time proportional to the graph size. Estimating the fraction of global Rydberg excitations requires a constant number, ${ \cal O } ( 1 / \epsilon ^ { 2 } )$ , of shots, and computing correlations requires a logarithmic time $O ( \log | V | / \epsilon ^ { 2 } )$ to maintain a target statistical error ϵ. While executing this pipeline on real analog hardware will successfully eliminate the exponential bottleneck of classical state-vector simulation, demonstrating a strict computational or learning advantage over classical graph methods remains an open challenge. Furthermore, reducing and managing the classical $O ( | V | ^ { 2 } )$ post-processing costs for the correlation matrix remains a critical focus for future large-scale implementations.

## Acknowledgment

We thank I. Sokolov and S. Acheche (PASQAL) for insightful discussions and support.

## References

[1] N. Shervashidze, P. Schweitzer, E. J. van Leeuwen, K. Mehlhorn, and K. M. Borgwardt, “Weisfeiler-lehman graph kernels,” Journal of Machine Learning Research, vol. 12, pp. 2539–2561, Sep. 2011.

[2] N. M. Kriege, F. D. Johansson, and C. Morris, “A survey on graph kernels,” Applied Network Science, vol. 5, no. 1, p. 6, 2020.

[3] Y. Ma and J. Tang, Deep Learning on Graphs. Cambridge University Press, 2021.

[4] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learn. Representations (ICLR), 2017.

[5] J. Gan, Y. Chen, B. Hu, J. Leng, W. Wang, and X. Gao, “Characters as graphs: Interpretable handwritten chinese character recognition via pyramid graph transformer,” Pattern Recognit., vol. 137, p. 109317, 2023.

[6] J. Gan, Y. Zhou, Y. Zhang, J. Leng, and X. Gao, “Structure-aware handwritten text recognition via graph-enhanced cross-modal mutual learning,” in Proc. IJCAI, 2025, pp. 5154–5162.

[7] C. Morris, G. Rattan, and P. Mutzel, “Weisfeiler and Leman go sparse: Towards scalable higher-order graph embeddings,” arXiv:1904.01543, 2020.

[8] J. Chen, X. Luo, Y. Yuan, and Z. Wang, “Enhancing graph convolutional networks with an eficient k-hop neighborhood approach,” Inf. Fusion, vol. 124, p. 103297, 2025.

[9] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, “Quantum machine learning,” Nature, vol. 549, no. 7671, pp. 195–202, 2017.

[10] S. Yu, C. Peng, Y. Wang, A. Shehzad, F. Xia, and E. R. Hancock, “Quantum graph learning: Frontiers and outlook,” arXiv:2302.00892, 2023.

[11] Y. Tang, J. Yan, and E. R. Hancock, “From Quantum Graph Computing to Quantum Graph Learning: A survey,” arXiv:2202.09506, 2022.

[12] X. Ai, Z. Zhang, L. Sun, J. Yan, and E. R. Hancock, “Towards Quantum Graph Neural Networks: An ego-graph learning approach,” arXiv:2201.05158, 2024.

[13] S. Y.-C. Chen, T.-C. Wei, C. Zhang, H. Yu, and S. Yoo, “Hybrid Quantum-Classical Graph Convolutional Network,” arXiv:2101.06189, 2021.

[14] L.-P. Henry, S. Thabet, C. Dalyac, and L. Henriet, “Quantum evolution kernel: Machine learning on graphs with programmable arrays of qubits,” Phys. Rev. A, vol. 104, no. 3, p. 032416, 2021.

[15] L. Henriet, L. Béguin, A. Signoles, T. Lahaye, A. Browaeys, G.-O. Reymond, and C. Jurczak, “Quantum computing with neutral atoms,” Quantum, vol. 4, p. 327, 2020.

[16] M. Djellabi, M. Hecker, and S. Acheche, “Attributed-graphs kernel implementation using local detuning of neutral-atoms Rydberg Hamiltonian,” arXiv:2509.09421, 2026.

[17] M. Schuld and N. Killoran, “Quantum machine learning in feature hilbert spaces,” Physical Review Letters, vol. 122, no. 4, p. 040504, 2019.

[18] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, “Supervised learning with quantum-enhanced feature spaces,” Nature, vol. 567, no. 7747, pp. 209–212, 2019.

[19] B. Albrecht, C. Dalyac, L. Leclerc, L. Ortiz-Gutiérrez, S. Thabet, M. D’Arcangelo, J. R. K. Cline, V. E. Elfving, L. Lassablière, H. Silvério, B. Ximenez, L.-P. Henry, A. Signoles, and L. Henriet, “Quantum feature maps for graph machine learning on a neutral atom quantum processor,” Phys. Rev. A, vol. 107, no. 4, p. 042615, 2023.

[20] W. L. Hamilton, Graph Representation Learning, ser. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool Publishers, 2020.

[21] Y. Rubner, C. Tomasi, and L. J. Guibas, “The earth mover’s distance as a metric for image retrieval,” Int. J. Comput. Vis., vol. 40, no. 2, pp. 99–121, 2000.

[22] C. R. Givens and R. M. Shortt, “A class of wasserstein metrics for probability distributions,” Michigan Mathematical Journal, vol. 31, no. 2, pp. 231–240, 1984.

[23] K. Nguyen, “Streaming Sliced Optimal Transport,” arXiv:2505.06835, 2026.

[24] L. Zhang, L. Yi, Y. Liu, C. Wang, and D. Zhou, “Motif Entropy Graph Kernel,” Pattern Recognit., vol. 140, p. 109544, 2023.

[25] H. Silvério, S. Grijalva, C. Dalyac, L. Leclerc, P. J. Karalekas, N. Shammah, M. Beji, L.-P. Henry, and L. Henriet, “Pulser: An open-source package for the design of pulse sequences in programmable neutral-atom arrays,” Quantum, vol. 6, p. 629, 2022.

[26] Y. LeCun, C. Cortes, and C. Burges, “Mnist handwritten digit database,” ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, vol. 2, 2010.

[27] G. Siglidis, G. Nikolentzos, S. Limnios, C. Giatsidis, K. Skianis, and M. Vazirgiannis, “Grakel: A graph kernel library in python,” pp. 1–5, 2020.

[28] M. Fey and J. E. Lenssen, “Fast graph representation learning with PyTorch Geometric,” arXiv:1903.02428, 2019.