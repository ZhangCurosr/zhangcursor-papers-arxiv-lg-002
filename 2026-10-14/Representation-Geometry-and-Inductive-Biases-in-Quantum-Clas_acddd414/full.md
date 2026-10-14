# Representation Geometry and Inductive Biases in Quantum Classifiers across Classical Visual Benchmarks

Yu Yamaoka<sup>1\*</sup>, Weng Ian Chan<sup>2</sup> and Kosuke Mitarai<sup>3</sup>

<sup>1\*</sup>Center for Quantum Information and Quantum Biology, The University of Osaka, 1-2 Machikaneyama, Toyonaka, 560-0043, Osaka, Japan.   
<sup>2</sup>SANKEN, The University of Osaka, 1-5 Yamadaoka, Suita, 565-0871, Osaka, Japan.

<sup>3</sup>Graduate School of Engineering Science, The University of Osaka, 1-3 Machikaneyama, Toyonaka, 560-8531, Osaka, Japan.

\*Corresponding author(s). E-mail(s): yamaoka.yuu.qiqb@osaka-u.ac.jp;

## Abstract

Quantum Machine Learning (QML) has a theoretical advantage over classical algorithms for specific synthetic data that is computationally hard. However, it remains unclear whether such theoretical advantages play a meaningful role for realistic classical data, such as natural images. Quantum structural properties induced by unitary operations or entanglement introduce inductive biases that difer from those of classical models, potentially leading to distinct learning behavior on real-world data. In a fully quantum learning pipeline, classical realworld data are first encoded into quantum states via a quantum feature mapping, analogous to a classical feature extractor, and a quantum classifier then classifies the resulting states. Quantum feature mappings generate representations that are dificult to realize classically, while also inducing representation-level biases that do not necessarily match the structure of realistic data. To enable a fair comparison at the classifier level, we focus on the classifier and use classical feature extractors that already provide powerful representations for natural images. By controlling the data scale and supervision strength, we analyze how the unitary structure of quantum classifiers afects generalization, and show that their efectiveness depends strongly on the geometry of the input representations. These results indicate that quantum classifiers benefit from classical features that are aligned with quantum feature spaces. Our analysis shows that kernel alignment identify representations compatible with quantum feature spaces, providing insights into translating classical models into quantum algorithms.

Keywords: Quantum machine learning, Quantum classifiers, Inductive biases, Representation geometry, Image classification

## 1 Introduction

Recent progress in machine learning relies on large models, high-resolution inputs, and large-scale datasets, which increases computational cost. This trend raises fundamental questions about the long-term sustainability of relying solely on classical digital hardware. To address this challenge, researchers begin exploring alternative computational paradigms. Quantum computing in machine learning is one potential approach. Unlike classical computation, which processes deterministic bit transitions, quantum computation manipulates superposed states and evolves them through unitary transformations, efectively operating on exponentially large state spaces in parallel. It thus provides a fundamentally diferent computational substrate for large-scale learning. Prior work on quantum machine learning (QML) demonstrates computational advantages of quantum models over classical learners on carefully constructed learning tasks, such as those derived from the discrete logarithm problem (Liu et al. 2021). In addition to such hardness-based advantages, QML studies have also highlighted distinctive inductive properties of quantum models, including enhanced expressivity (Abbas et al. 2021; Havlíček et al. 2019) and implicit regularization induced by unitary constraints (Jiang et al. 2023; Caro et al. 2022).

However, whether these theoretical advantages translate into practical improvements on natural image data remains an open question. Applying QML to such data is nontrivial, as classical inputs must be encoded into quantum states and learning procedures must be implemented using unitary quantum operations. These requirements impose structural constraints that difer fundamentally from those of classical architectures. As circuit depth increases, these constraints can also lead to well-known optimization problems, including barren plateaus (Larocca et al. 2025). Therefore, evaluating QML on realistic image data requires understanding what inductive biases quantum classifiers introduce and how these biases afect predictive performance and trainability. In this work, we systematically characterize the classifier-level inductive biases in vision classification across diverse datasets (Netzer et al. 2011; Yadav and Bottou 2019; Yang et al. 2023; Xiao et al. 2017; Coates et al. 2011; Krizhevsky 2009).

The inductive biases of quantum models depend on where quantum components are introduced in the learning pipeline: in the feature mapping stage, the classifier stage, or both (Havlíček et al. 2019; Abbas et al. 2021; Cong et al. 2019). While fully quantum models are an attractive long-term objective, a fair evaluation of quantum inductive bias requires controlled input conditions at the classifier stage and assessment on realistic data rather than synthetic datasets constructed for a quantum model. When both the feature extractor and classifier are quantum, the feature mapping may produce representations that are especially well matched to the quantum classifier, making it dificult to isolate classifier-level inductive bias because the observed behavior depends on both components. For natural images, strong self-supervised classical backbones already provide powerful visual representations (Caron et al. 2021; Oquab et al. 2023; Siméoni et al. 2025; He et al. 2016; Radford et al. 2021), while the advantage of quantum feature mappings over such backbones remains unclear. Therefore, we use the classical feature extractor and replace only the classifier to isolate the quantum model’s contribution for a fair evaluation on classical image tasks.

Our study tests two hypotheses. First, the unitary constraint in quantum circuits may serve as an implicit regularizer by enforcing transformations that preserve the norm of quantum states, thereby improving generalization under limited supervision. Second, the geometry of quantum state spaces may introduce inductive biases that appear more clearly in small-data or weakly supervised settings. We carefully control model capacity and circuit depth to ensure a fair and meaningful evaluation. In particular, we design quantum classifiers with parameter counts comparable to those of their classical counterparts while avoiding excessively deep circuits that are prone to optimization problems, such as barren plateaus. This controlled setup allows us to attribute observed diferences to inductive bias rather than to mismatched model capacity or unstable training dynamics. By isolating the quantum classifier from feature-representation efects, we identify when quantum models behave similarly to classical models and when unitary constraints help or hinder performance.

Our contributions are as follows:

• Theoretical perspective on quantum inductive bias. We formalize conditions under which quantum classifiers can operate comparably to classical models on natural image representations, and analyze how unitary constraints act as implicit regularization under limited supervision.

• Controlled empirical evaluation across supervision regimes. We systematically evaluate quantum classification heads across multiple classical feature extractors, dataset sizes, and weakly supervised settings. This allows us to disentangle efects arising from quantum expressivity and those induced by unitary structural constraints.

• Fair comparison under matched capacity and depth control. We match parameter counts between quantum and classical classifiers and restrict circuit depth to mitigate optimization pathologies such as barren plateaus, ensuring that observed diferences reflect inductive bias rather than model capacity or instability.

In modern vision systems dominated by strong self-supervised backbones, clarifying whether alternative computational paradigms provide complementary inductive biases is a first step toward practical quantum vision systems.

## 2 Related work

Quantum feature map and kernel Quantum feature maps are among the most widely studied approaches in quantum machine learning. Havlíček et al. (Havlíček et al. 2019) proposed embedding classical data into high-dimensional quantum states and performing classification via kernel methods. This idea was further formalized by Schuld and Killoran (Schuld and Killoran 2019), who showed that many quantum models can be interpreted as implicit kernel methods. More recently, Liu et al. (Liu et al. 2021) established a provable quantum advantage for a specific learning problem using quantum kernel estimation.

Variational quantum classifier Another major line of research focuses on variational quantum classifiers (VQCs), where parametrized quantum circuits are trained using classical optimizers. Early work by Farhi and Neven (Farhi and Neven 2018) demonstrated classification using shallow quantum circuits. Subsequent studies investigated the expressive power of variational quantum models, showing that quantum neural networks can represent complex decision boundaries in high-dimensional Hilbert spaces (Abbas et al. 2021).

Expressivity and trainability theory Theoretical studies have analyzed the representational capacity and trainability of quantum models. Huang et al. (Huang et al. 2021) established information-theoretic bounds on quantum advantage in learning. Caro et al. (Caro et al. 2022) investigated the generalization behavior of quantum models trained on small datasets. At the same time, several works have highlighted optimization challenges in variational quantum circuits, such as the barren plateau phenomenon, where gradients vanish exponentially with system size (Cerezo et al. 2022; Larocca et al. 2025).

Application to real-world vision data Applications of quantum machine learning to computer vision tasks have also been explored. Cong et al. (Cong et al. 2019) introduced quantum convolutional neural networks, while Henderson et al. (Henderson et al. 2020) proposed quanvolutional layers for image feature extraction. Hybrid classical–quantum pipelines combining classical feature extractors with quantum models have also been studied in transfer learning settings (Mari et al. 2020).

Despite these eforts, the empirical benefits of quantum components on natural image datasets remain unclear. Most existing studies rely on small-scale datasets or synthetic benchmarks, and it remains an open question whether quantum models provide practical advantages when combined with strong classical feature representations.

## 3 Method

## 3.1 Quantum circuit learning

## Quantum state in Bra-Ket Notation

In this paper, we use Dirac’s bra-ket notation to represent a quantum state, and the appendix explains this notation in detail. Any one-qubit pure quantum state is $\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle ( \alpha , \beta \in \mathbb { C } , \ \vert \alpha \vert ^ { 2 } + \vert \beta \vert ^ { 2 } = 1 )$ based on $| 0 \rangle = { \bigg [ } { \bigg ] } { \bigg ] } , \ | 1 \rangle = { \bigg [ } { \bigg ] }$ as a computational basis. The zero state of the n-qubit is ${ | 0 \rangle } ^ { \otimes n } = { | 0 \rangle } \otimes \cdots \otimes { | 0 \rangle }$ . The tensor product $\otimes$ represents the direct product of multi-qubit states.

## Unitary time evolution in a quantum circuit

A quantum circuit is a computational model based only on unitary matrices. We obtain the quantum state $\vert \bar { \psi } \rangle = U _ { L } \cdot \cdot \cdot U _ { 2 } U _ { 1 } \vert 0 \rangle ^ { \otimes n }$ though gates $U _ { 1 } U _ { 2 } \cdots$ · from zero state $\left| 0 \right. ^ { \otimes n }$

![](images/92e39a295ec609629bda73d18839aa8003e2a7f7634fce91969d9eca87a033ce.jpg)  
Fig. 1 Quantum classification pipeline using classical features. $U _ { e n t }$ is a fixed entangling unitary applied across the qubits.

## Encoding and learning

As shown in Fig. 1, quantum circuit encodes classical features $\mathbf { x } \in \mathbb { R } ^ { n }$ obtained thorugh a classical feature extractor (Radford et al. 2021; He et al. 2016; Caron et al. 2021; Oquab et al. 2023; Siméoni et al. 2025; Howard et al. 2017) and PCA (Pedregosa et al. 2011) to quantum circuit by angle encoding (Havlíček et al. 2019). We preserve the normalization of the amplitudes from the input data as a function of the angles ϕ and $\phi ^ { \prime }$ . This naturally introduces inverse trigonometric functions; the initial angle for the i-th qubit is $\theta _ { i } = \arcsin ( x _ { i } ) , \phi _ { i } = \operatorname { a r c c o s } ( x _ { i } ^ { 2 } )$

The trainable unitary matrix $W ( { \boldsymbol { \Theta } } )$ is appended after the feature encoding $U ( \mathbf { x } ) \left| 0 \right.$ The circuit $W ( { \boldsymbol { \Theta } } )$ is capable of adapting through training, regardless of the initial state |0⟩.

$$
U ( \mathbf { x } , \mathbf { \Theta } ) = W ( \mathbf { \Theta } \mathbf { \Theta } \mathbf { \Theta } U _ { \mathrm { e n c } } ( \mathbf { x } ) .
$$

The resulting parameterized quantum state $\left| \psi ( \mathbf { x } , \pmb \Theta ) \right. \mathrm { ~ i s ~ } U ( \mathbf { x } , \pmb \Theta ) \left| 0 \right. ^ { \otimes n }$

In the optimization step, we compute the gradient with respect to the circuit parameters. We describe the gradient computation in detail in the appendix.

## Generalization based on unitary

Because unitary operators satisfy $U ^ { \dagger } U = I$ , they preserve the norm of the quantum state. This property leads to the generalization bound stated below.

Theorem 1 (generalization vs. circuit depth) For an n-qubit quantum circuit consisting of T unitary operations, the generalization performance of the converged parameters θ<sup>ˆ</sup> after training has upper-bounded in terms of the dataset size N as follows (Caro et al. 2022):

$$
\begin{array} { r l } { \mathrm { g e n e r a l i z a t i o n } ( \hat { \Theta } ) } & { \leq \mathcal { O } \left( \sqrt { \frac { T \log T } { N } } \right) , } \end{array}\tag{1}
$$

For the same task, a quantum model use $T = \mathrm { p o l y } ( n )$ trainable gates, whereas a classical model requires $O ( 2 ^ { n } )$ parameters in the worst case to achieve comparable expressivity results in Sec. 4.1.

## 3.2 Kernel geometry induced by the quantum classifier

In our setting, the image representation x is fixed by a classical feature extractor. A parameterized quantum circuit then maps the classical feature vector $\mathbf { x } \in \mathbb { R } ^ { n }$ to a quantum state $\left| \bar { \psi } ( \mathbf { x } , \pmb \theta ) \right. = U ( \mathbf { x } , \pmb \theta ) \left| 0 \right. ^ { \otimes n }$

This mapping implicitly defines a feature representation in the Hilbert space. We characterize the representation through the quantum kernel.

$$
K _ { Q } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } ) = | \langle \psi ( \mathbf { x } _ { i } , \pmb { \theta } ) | \psi ( \mathbf { x } _ { j } , \pmb { \theta } ) \rangle | ^ { 2 } .
$$

This kernel corresponds to the inner product between quantum feature states. It measures the similarity between samples after the quantum feature transformation.

## Kernel Target Alignment

We evaluate the quality of the induced feature representation using Kernel Target Alignment (KTA). Given a kernel matrix K and a one-hot label matrix $Y \in \{ 0 , 1 \} ^ { N \times C }$ we define the label kernel as $T = Y Y ^ { \top }$ and the centering matrix as $\begin{array} { r } { H = I _ { N } - \frac { 1 } { N } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ Here, $N = \mathrm { m i n } ( 1 0 2 4 , N _ { \mathrm { t r a i n } } )$ denotes the number of training samples used to construct the kernel matrices and C is class number. KTA is defined as

$$
\operatorname { K T A } ( K , Y ) = { \frac { \langle H K H , H T H \rangle _ { F } } { \| H K H \| _ { F } \| H T H \| _ { F } } } .
$$

KTA measures how well the feature-induced similarity structure aligns with the class labels. Higher KTA indicates that samples from the same class become more similar in the feature space.

## Quantum vs. classical feature geometry

We compare the representation induced by the quantum circuit with that of the original classical features. Let $K _ { V }$ denote the kernel computed from the classical feature vectors and $K _ { Q }$ denote the kernel induced by the quantum states. We analyze the change in alignment.

$$
\Delta \mathrm { K T A } = \mathrm { K T A } _ { Q } - \mathrm { K T A } _ { V } .
$$

A positive ∆KTA indicates that the quantum-induced similarity structure is more aligned with the class labels than the original classical feature geometry. In Sec. 4, we

![](images/2d81624a17d558cba4a7c56a9ff237dcc04f78a5b8a0b85a920f9f6c616b1cbb.jpg)

![](images/14892fd9cf7992a507baf91426dd1e9ab1527cff20896ac3ed603d82d4492385.jpg)

![](images/dc3ac8f633e92b6005b9978727d1b19d56fc89ed61ca1423873690e0f327bbeb.jpg)

![](images/f7eeac8085f2d2915b8ceb932dcd91746385125d97d393d3b6073bbacb3b1b24.jpg)  
Fig. 2 Mean test accuracy with 95% confidence interval as error bars across diferent dataset sizes

![](images/eac84efd32eb1cb83f6751b8707ec5ad6fa66c5a41f6e41f4d284a71327ed5e1.jpg)

![](images/caa2b63ec0a61e4fe088bcaa1f6c72116c3fd08a14ba89a3a786c60006a0f957.jpg)

![](images/825b424da61060fba81c6b41eac0d0b705791dcf89400006a4d90ae9af81e46e.jpg)

![](images/503215f55f72636f01f8c359f92d2d063de8abbc577834004fe98488b610aabe.jpg)  
Fig. 3 Mean generalization with 95% confidence interval as error bars across diferent dataset sizes (lower is better). Generalization is defined as the diference between training and test accuracies. study how this change relates to classification accuracy across datasets and feature extractors results in Sec. 4.2 and Sec. 4.3.

## 4 Experiment and Result

## 4.1 Generalization by Unitary constraints

Objective. This experiment evaluates our first hypothesis introduced in Sec. 1: The unitary structure of quantum classifiers may act as an implicit regularizer and improve generalization under limited supervision. To test it, we adopt learning from label proportions (LLP), a weakly supervised setting in which the model observes only class proportions over groups of samples rather than individual instance labels. This setting reduces the amount of supervision while preserving the underlying image classification task, allowing us to examine whether quantum classifiers are more robust than classical classifiers when label information is limited. We define our LLP loss in the appendix. Experimental setting. We use the first four classes in each dataset due to the high computational cost of the quantum simulation. We set the bag size to ten, which defines the weakly supervised setting of learning from label proportions. We use qulacsgpu $( \mathrm { v 0 . 6 . 1 1 } ) ^ { 1 }$ as a quantum simulator and DINO $\mathrm { v 2 ^ { 2 } }$ ViT-S/14 model as the feature extractor. The classical classifier consists of three fully connected layers with the ReLU activation function (Nair and Hinton 2010). The number of learnable parameters is hidden layer size $h ,$ input size $n ,$ class number $C ,$ as $( n + 1 ) h + ( h + 1 ) h + ( h + 1 ) C$ from the hyperparameters. The number of parameters of the quantum circuit is 3 × n × depth, calculated from the depth of the three parametric gates RX–RZ–RX and the qubit number n. Under the condition of class number $C = 4$ , in order to make the number of parameters to be updated in quantum and classical close, we set $n = 4 , h = 1 5 , d e p t h = 3 2$ , obtaining: the number of learnable parameters θ<sup>ˆ</sup> is 379 for classical and 384 for quantum. We use 95% confidence intervals computed over fifty independently trained models initialized with diferent random seeds to account for variability arising from parameter initialization.

Result. Fig. 2 shows the test accuracy of the training dataset fraction across four benchmarks (CIFAR10 (Krizhevsky 2009), FashionMNIST (Xiao et al. 2017), MedMNIST-Blood (Yang et al. 2023), and SVHN (Netzer et al. 2011)). Fig. 3 reports the corresponding generalization gap, defined as |TestAcc − TrainAcc|, together with a reference scaling proportional to $\bar { 1 } / \sqrt { N }$ following Eq. 1.

Across all datasets, increasing the dataset fraction improves test accuracy for both the classical and quantum classifiers. However, their relative behavior difers significantly across data fractions. At low fractions, the classical classifier often achieves slightly higher raw test accuracy on CIFAR10 and FashionMNIST. On the other hand, the quantum classifier is marginally better on MedMNIST-Blood and SVHN, suggesting that the two approaches are overall competitive across datasets. Nevertheless, the quantum classifier consistently exhibits a smaller generalization gap (Fig. 3), indicating reduced overfitting. This result suggests that the unitary constraint acts as an implicit regularizer.

As the dataset size N increases, the generalization gap of the classical model decreases approximately, following the expected scaling $\scriptstyle { \mathcal { O } } \left( { \frac { 1 } { \sqrt { N } } } \right)$ as indicated by the dashed reference line in Fig. 3. On CIFAR10 and FashionMNIST, the classical head ultimately achieves higher asymptotic accuracy, while the quantum plateaus earlier, suggesting a capacity limitation induced by the unitary constraint. In contrast, on SVHN, quantum consistently outperforms the classical classifier across dataset fractions (Fig. 2), and exhibits lower variance and more stable generalization behavior.

While the generalization gap decreases with N for both models, the quantum classifier starts from a noticeably smaller gap, suggesting a smaller prefactor in the expected $\mathcal { O } ( 1 / \sqrt { N } )$ scaling. One possible explanation is that, for certain tasks, a quantum model can learn with $T = \mathrm { p o l y } ( n )$ trainable gates. In contrast, a classical model can need ${ \mathcal { O } } ( 2 ^ { n } )$ parameters in the worst case to match the same level of expressivity.

## 4.2 Cross-Dataset and Cross-Feature Comparison

Objective. This experiment evaluates our second hypothesis: the geometry induced by quantum state spaces interacts with the input representation, leading quantum classifiers to behave diferently across datasets and feature extractors. To examine this efect, we compare quantum and classical classifier heads under controlled training conditions while varying both the dataset and the feature extractor.

Experimental setting. We use seven feature extractors and ten datasets, and keep hyperparameters and training procedures identical to ensure a controlled comparison.

We set frac = 0.1 for CIFAR-10, FashionMNIST, MedMNIST-Derma, QMNIST, and SVHN, considering the computational cost of quantum simulation. We use frac = 0.5 for Food101 and STL-10 and frac = 1.0 for Flowers102, Oxford-IIIT Pet, and MedMNIST-Retina due to their smaller training-set sizes. We keep these fractions across quantum and classical models, and evaluate all models on the full test set of the corresponding four-class subset.

![](images/a8ffaf22dffc5d63dafc35b0f1e0fb1414379a2a30d8d70082d1a710c3c82367.jpg)  
Fig. 4 Test accuracy on cross-dataset and cross-feature comparison.

We consider four-class subsets for each dataset due to the computational cost of quantum simulation. The bag size is fixed at 1 in this setting, corresponding to fully supervised instance-level labels. For quantum training, we run 1000 optimization iterations with 10 number of random seeds. We use seven-dimensional features and qubit $n = 7$ , which is the largest feature dimension we could practically evaluate given the computational cost of quantum simulation. We also match the parameter counts of the quantum and classical classifiers as closely as possible. Feature extractors include CLIP, DINO variants, MobileNetV3-Large, ResNet50, and PCA-based features.

Result. Figure 4 shows the diference in test accuracy between the classical and quantum classifiers across datasets and feature extractors, where positive values indicate higher accuracy for the quantum classifier, whereas negative values indicate higher accuracy for the classical classifier. Overall, the classical classifier achieves higher accuracy in most settings. The gap is particularly large on Flowers102, where several backbones (e.g., DINO v1 and ResNet50) show diferences larger than 0.2. Moderate advantages for the classical model also appear on datasets such as FashionMNIST and QMNIST.

In contrast, the classical classifier achieves higher accuracy in most dataset–feature combinations. Quantum improvements occur only in a limited number of settings, including MedMNIST-Derma with DINOv2 and PCA features and MedMNIST-Retina with PCA features. The magnitude of these improvements remains modest, whereas several settings show substantially larger advantages for the classical classifier. These results indicate that replacing the classical classifier with a unitary quantum classifier does not consistently improve performance on standard image features. The results also suggest that the efectiveness of the quantum model depends strongly on the dataset and the choice of feature representation.

Relationship between geometry change and performance. Figure 5 plots the relationship between the change in kernel-target alignment (KTA) and the change in test accuracy between quantum and classical classifiers across all datasets and feature extractors. Each point corresponds to a dataset–backbone pair, and the axes represent $\Delta \mathrm { K T A } = \mathrm { K T A } _ { Q } - \mathrm { K T A } _ { V }$ and $\Delta \mathrm { A c c } = \mathrm { A c c } _ { Q } - \mathrm { A c c } _ { V }$ , respectively. We observe a clear positive correlation between these quantities (Pearson $r = 0 . 3 1 2 .$ , Spearman $\rho = 0 . 4 2 9 )$

![](images/364511eaac8e5caa689344044b9d84fe45287e8de82aab4741f519e0735e9b8c.jpg)  
Fig. 5 Relationship between the change in kernel-target alignment ∆KTA and the test accuracy diference ∆Acc. Each point corresponds to a dataset–backbone pair, with colors indicating datasets and marker shapes indicating backbones. A positive correlation (Pearson $r = 0 . 3 1 2$ , Spearman $\rho = 0 . 4 2 9 )$ indicates that performance diferences follow changes in representation alignment.

This trend indicates that performance diferences between quantum and classical classifiers are strongly related to the alignment between the induced representation geometry and the class labels. When the quantum classifier increases KTA relative to the classical classifier, the test accuracy tends to improve as well. Conversely, when the quantum representation reduces alignment with class labels, the performance gap becomes negative.

Importantly, the absolute KTA of the classical representation alone does not explain the performance gap (see Fig. 5). Instead, the change in alignment induced by the classifier is more significant. These results suggest that quantum classifiers do not simply improve weak feature geometries; rather, they reshape the feature space’s similarity structure. Performance gains arise when the induced quantum geometry better matches the data’s class structure.

![](images/9050e9077ae86ecb4439de83006d6636b4b4ff79e41dd4647c944366df8e0763.jpg)

![](images/d2d46d7313510cc409580a83c207bc6cd3dace6dee7a681623022f4800756e01.jpg)

![](images/15e4b497eee90306d179576b281ddd931598e7c02c853f53263f9d5ba8aed4df.jpg)

![](images/d885a369cf17c4f5784aa81f418f843b6e5a92663aa9d6186a815a91e6c3303f.jpg)

![](images/281b46312ac61d7d0237b35e7da7551a08a9ec7635d7c14a77bc312bccd5c13f.jpg)

![](images/a0a634cf6858b0955bee717f64108803ac82880cdfd3ee0caf73cab26dc1e527.jpg)

![](images/0ff94928e57ddfd1c767e7e72ea81b2a58f11891336bb0ca9f156199485e963d.jpg)

![](images/40bb30c5c3e6258d523bced0bdd6790ceded63d66c8fc810be043bba9704384d.jpg)

![](images/06504c84a2f7f702fddf42dc30af6b6261a56755234b69843d2ad2d3f4d43ce2.jpg)

![](images/a941137d5068dc542cf99f2e5bac2d3a15d9cd6d5aada0b0d568a35471a2c88f.jpg)  
Fig. 6 Test accuracy versus the number of trainable parameters for each dataset using CLIP-extracted image features and a quantum classifier head.

## 4.3 Parameter eficiency of quantum

We further study how test accuracy changes as the number of trainable parameters increases. Figure 6 shows the resulting accuracy curves. In these cases, the quantum model reaches competitive accuracy with fewer parameters than the classical model. This observation suggests that when the quantum feature representation aligns well with the label structure (high ∆KTA), the circuit can represent the task using a smaller number of trainable parameters. However, such favorable cases are rare. Among a total of 70 dataset–feature combinations we evaluated, only a small fraction exhibit this behavior. Overall, these observations indicate that quantum circuits do not uniformly improve performance, but they can provide parameter-eficient representations for specific feature geometries.

## 5 Limitation and Conclusion

## Limitations.

This work studies classifier-level quantum inductive bias in a controlled simulation setting. Because classical simulation of quantum circuits is computationally expensive, our experiments are limited to four-class subsets, reduced training fractions, and small circuits with n = 4 qubits at Fig. 2 or n = 7 at Fig. 4. The observed trends therefore characterize a low-dimensional regime and should be further validated with more qubits, more classes, and full datasets.

We also compress classical visual features, such as DINOv2 and CLIP embeddings, to four dimensions using PCA to match the quantum input size. This enables a controlled comparison between quantum and classical classifier heads under identical input dimensionality, but it introduces an information bottleneck and does not represent the performance of full-dimensional classical probes. Accordingly, our results should be interpreted as an analysis of quantum heads on compressed visual representations, rather than as evidence of general quantum advantage over modern classical vision models.

Finally, quantum classifiers are not universally superior in our experiments; classical MLPs outperform them in several settings. This highlights the importance of analyzing when quantum circuit structure is beneficial, rather than assuming a universal advantage. Future work should explore larger circuits, quantum-aware feature mappings, and evaluations beyond classical simulation.

## Conclusion.

Quantum computing is currently transitioning from the Noisy Intermediate-Scale Quantum (NISQ) era towards the development of fault-tolerant quantum computers (FTQC). Quantum machine learning is expected to benefit from the high computational capacity of FTQC systems and may ofer new opportunities to improve machine learning models.

In this work, we investigated the inductive biases of quantum classifiers for classical image features. We constructed a competitive quantum classifier with a comparable number of parameters to classical baselines and evaluated it across diverse datasets and feature extractors. Under controlled settings of data scale and supervision strength, the quantum model showed stable generalization behavior induced by unitary operators. Across 70 evaluated datasets–feature combinations, the quantum classifier achieved modest accuracy or parameter-eficiency advantages in only a small subset, while the classical classifier performed better in most settings. Furthermore, our analysis showed that kernel alignment provides a useful indicator for identifying representations compatible with quantum feature spaces. Quantum classifiers showed no clear advantage on conventional image classification tasks without an underlying source of classical computational hardness when suficient many trainable parameters and training data are available. However, quantum classifiers show advantages when the classical feature geometry aligns with quantum feature spaces. This observation ofers a practical guideline for selecting classical feature representations when translating classical models into quantum algorithms.

## 6 Acknowledgments

Y.Y. was supported by JST COI-NEXT (Grant No. JPMJPF2014) and JSPS KAKENHI (Grant No. 25K24367). W.C. was supported by JST (Grant Number JPMJFS2125). K.M. was supported by JST FOREST (Grant No. JPMJFR232Z) and JSPS KAKENHI (Grant Nos. 23H03819 and 24K16980).

## Author contributions statement

Y.Y. conceived the experiments. Y.Y., K.M. constructed an inference and learning pipeline and conducted the experiments. Y.Y., W.C. wrote the manuscript. All authors analyzed the results and reviewed the manuscript.

## Data availability

The datasets generated during the current study are available from the corresponding author on reasonable request. Source codes can be found in this Github(https:// anonymous.4open.science/r/Quantum\_Classifier-F176/), and updates will be made periodically.

## References

Abbas A, Sutter D, Zoufal C, et al (2021) The power of quantum neural networks. Nature Computational Science 1(6):403–409. https://doi.org/10.1038/ s43588-021-00084-1

Caro MC, Huang HY, Cerezo M, et al (2022) Generalization in quantum machine learning from few training data. Nature communications 13(1):4919

Caron M, Touvron H, Misra I, et al (2021) Emerging properties in self-supervised vision transformers. In: ICCV, pp 9650–9660

Cerezo M, Verdon G, Huang HY, et al (2022) Challenges and opportunities in quantum machine learning. Nature computational science 2(9):567–576

Coates A, Ng A, Lee H (2011) An analysis of single-layer networks in unsupervised feature learning. In: Gordon G, Dunson D, Dudík M (eds) Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, vol 15. PMLR, Fort Lauderdale, FL, USA, pp 215–223

Cong I, Choi S, Lukin MD (2019) Quantum convolutional neural networks. Nature Physics 15(12):1273–1278

Farhi E, Neven H (2018) Classification with quantum neural networks on near term processors. arXiv preprint arXiv:180206002

Havlíček V, Córcoles AD, Temme K, et al (2019) Supervised learning with quantumenhanced feature spaces. Nature 567(7747):209–212

He K, Zhang X, Ren S, et al (2016) Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 770–778

Henderson M, Shakya S, Pradhan S, et al (2020) Quanvolutional neural networks: powering image recognition with quantum circuits. Quantum Machine Intelligence 2(1):2

Howard AG, Zhu M, Chen B, et al (2017) Mobilenets: Eficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:170404861

Huang HY, Kueng R, Preskill J (2021) Information-theoretic bounds on quantum advantage in machine learning. Nature Communications 12(1):2631. https://doi.org/ 10.1038/s41467-021-22539-9

Jiang J, Zhao Y, Li R, et al (2023) Strong generalization in quantum neural networks. Quantum Information Processing 22(12)

Krizhevsky A (2009) Learning multiple layers of features from tiny images. Tech. rep., University of Toronto

Larocca M, Thanasilp S, Wang S, et al (2025) A review of barren plateaus in variational quantum computing. Nature Reviews Physics 7(4):174–189

Liu Y, Arunachalam S, Temme K (2021) A rigorous and robust quantum speed-up in supervised machine learning. Nature physics 17(9):1013–1017

Mari A, Bromley TR, Izaac J, et al (2020) Transfer learning in hybrid classical-quantum neural networks. Quantum 4:340

Nair V, Hinton GE (2010) Rectified linear units improve restricted boltzmann machines. ICML p 807–814

Netzer Y, Wang T, Coates A, et al (2011) Reading digits in natural images with unsupervised feature learning. In: Proceedings of the NIPS Workshop on Deep Learning and Unsupervised Feature Learning

Oquab M, Darcet T, Moutakanni T, et al (2023) Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:230407193

Pedregosa F, Varoquaux G, Gramfort A, et al (2011) Scikit-learn: Machine learning in python. The Journal of Machine Learning Research 12:2825–2830

Quadrianto N, Smola AJ, Caetano TS, et al (2008) Estimating labels from label proportions. In: ICML, pp 776–783

Radford A, Kim JW, Hallacy C, et al (2021) Learning transferable visual models from natural language supervision. In: International conference on machine learning, PmLR, pp 8748–8763

Schuld M, Killoran N (2019) Quantum machine learning in feature hilbert spaces. Physical review letters 122(4):040504

Siméoni O, Vo HV, Seitzer M, et al (2025) Dinov3. arXiv preprint arXiv:250810104

Suzuki Y, Kawase Y, Masumura Y, et al (2021) Qulacs: a fast and versatile quantum circuit simulator for research purpose. Quantum 5:559

Xiao H, Rasul K, Vollgraf R (2017) Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms

Yadav C, Bottou L (2019) Cold Case: The Lost MNIST Digits. In: NeurIPS. Curran Associates, Inc.

Yang J, Shi R, Wei D, et al (2023) MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification. Scientific Data 10(1):41

## Appendix A Appendix

## A.1 Quantum circuit

## Quantum state in Bra-Ket Notation

In this paper, we use Dirac’s bra-ket notation for the representation of a quantum state. Any one-qubit pure quantum state is $\left| \psi \right. = \alpha \left| 0 \right. + \beta \left| 1 \right. \left( \alpha , \beta \in \mathbb { C } , \ \left| \alpha \right| ^ { 2 } + \left| \beta \right| ^ { 2 } = 1 \right)$ based on $| 0 \rangle = { \bigg [ } { \overset { \mathbf { \sim } } { 0 } } { \bigg ] } , \ | 1 \rangle = { \bigg [ } { \overset { \mathbf { \sim } } { 1 } } { \bigg ] }$ as a computational basis.

The zero state of the n-qubit is ${ | 0 \rangle } ^ { \otimes n } = { | 0 \rangle } \otimes \cdots \otimes { | 0 \rangle }$ . The tensor product $\otimes$ represents the direct product of multi-qubit states.

## Unitary time evolution in a quantum circuit

A quantum circuit is a computational model based on only unitary matrices. We get quantum state $\left| \psi \right. = { \cal U } _ { L } \cdot \cdot \cdot \bar { { \cal U } } _ { 2 } { \cal U } _ { 1 } \left| 0 \right. ^ { \otimes n }$ though gates $U _ { 1 } U _ { 2 } \cdots$ · from zero state $| 0 \rangle ^ { \otimes { n } }$

## One-qubit gate

The Pauli matrices $X = { \left[ \begin{array} { l } { 0 } \\ { 1 } \end{array} \right] } , \ Y = { \left[ \begin{array} { l l } { 0 } & { - j } \\ { j } & { 0 } \end{array} \right] } , \ Z = { \left[ \begin{array} { l l } { 1 } & { 0 } \\ { 0 } & { - 1 } \end{array} \right] }$ are the fundamental operations. In this section, j denotes the imaginary unit. We define $R _ { X } , R _ { Y }$ , and $R _ { Z }$ gates as

$$
R _ { A } ( \theta ) = \exp \left( - \frac { j \theta } { 2 } A \right) \quad ( A \in \{ X , Y , Z \} ) .
$$

We encode the features obtained from the backbone into a quantum circuit by the operation $R _ { Y } ( \theta ) , R _ { Z } ( \phi )$ as below.

$$
R _ { Y } ( \theta ) | 0 \rangle = \cos { \frac { \theta } { 2 } } | 0 \rangle + \sin { \frac { \theta } { 2 } } | 1 \rangle , \qquad R _ { Z } ( \phi ) | 0 \rangle = e ^ { - j \phi / 2 } | 0 \rangle , ~ R _ { Z } ( \phi ) | 1 \rangle = e ^ { + j \phi / 2 } | 1 \rangle
$$

A global complex phase factor of the entire state does not afect observable quantities.

## Multi-qubits and entanglement

In multi-qubits, we generate entanglement state $| \psi \rangle \neq | \psi _ { 1 } \rangle \otimes \cdots \otimes | \psi _ { n } \rangle$ by control gate e.g. CNOT gate. Weight unitary with learning parameter $W ( { \boldsymbol { \Theta } } )$ includes entanglement.

## Measurement of quantum state

We use a Pauli $Z$ basis for measurement, 0 or 1 probability of the i-th qubit is as below by using the expectation value $\langle Z ^ { ( i ) } \rangle = \langle \psi | \hat { Z } ^ { ( i ) } | \psi \rangle \in [ - 1 , 1 ]$

$$
\operatorname* { P r } ( i \mathrm { { t h } { \ q u b i t } } = 1 ) = \frac { 1 - \langle Z ^ { ( i ) } \rangle } { 2 } , \quad \operatorname* { P r } ( i \mathrm { { t h } { \ q u b i t } } = 0 ) = \frac { 1 + \langle Z ^ { ( i ) } \rangle } { 2 }
$$

We use the expectation value of the loss function of the classification model.

![](images/6796f258e810216abfb8206fa1fff774af51cc17628d63a3156230d0e0e5696c.jpg)  
Fig. A Gate structure for encoding and weight matrix

## A.1.1 Feature Encoding

We explain angle encoding to a quantum circuit from features. Here, $\begin{array} { r l } { \mathbf { x } } & { { } = } \end{array}$ $( x _ { 1 } , . . . , x _ { d } ) ^ { \top } \in [ - 1 , 1 ] ^ { n }$ is input vector, qubit number is n. Since a single-qubit state is $\begin{array} { r } { | \psi \rangle = \cos \frac { \theta } { 2 } \left| 0 \right. + e ^ { j \bar { \phi } } \sin \frac { \theta } { 2 } \left| 1 \right. } \end{array}$ via Bloch sphere parameters. We preserve the normalization of the amplitudes from the input data as a function of the angles θ and $\phi .$ . This naturally introduces inverse trigonometric functions; the initial angle for the i-th qubit is

$$
\theta _ { i } = \arcsin ( x _ { i } ) , \qquad \phi _ { i } = \operatorname { a r c c o s } ( x _ { i } ^ { 2 } ) .
$$

We apply the unitary matrix $U _ { i } ( \mathbf { x } )$ to the i-th qubit as shown in Fig. A.

$$
U _ { i } ( { \bf x } ) = R _ { Z } ^ { ( i ) } ( \phi _ { i } ) R _ { Y } ^ { ( i ) } ( \theta _ { i } )
$$

In the implementation, operations act on the state from the $\mathrm { r i g h t }$ , so $R _ { Y }$ is applied before $R _ { Z } ;$ in matrix notation, the order is written as $R _ { Z } R _ { Y }$ . As the rotations are independent, the overall encoding circuit $U _ { \mathrm { e n c } } ( \mathbf { x } )$ is

$$
U _ { \mathrm { e n c } } ( { \bf x } ) = \bigotimes _ { i = 1 } ^ { n } U _ { i } ( { \bf x } ) = \bigotimes _ { i = 1 } ^ { n } \Big [ R _ { Z } ^ { ( i ) } \big ( \phi _ { i } \big ) R _ { Y } ^ { ( i ) } \big ( \theta _ { i } \big ) \Big ] .
$$

The mapping from the basis state |0⟩ is given by

$$
R _ { Y } ( \theta _ { i } ) \left| 0 \right. = \cos \frac { \theta _ { i } } { 2 } \left| 0 \right. + \sin \frac { \theta _ { i } } { 2 } \left| 1 \right. , R _ { Z } ( \phi _ { i } ) \left| 0 \right. = e ^ { - i \phi _ { i } / 2 } \left| 0 \right. , R _ { Z } ( \phi _ { i } ) \left| 1 \right. = e ^ { + i \phi _ { i } / 2 } \left| 1 \right. ,
$$

Therefore,

$$
U _ { i } ( { \bf x } ) \left| 0 \right. = e ^ { - i \phi _ { i } / 2 } \cos \frac { \theta _ { i } } { 2 } \left| 0 \right. + e ^ { + i \phi _ { i } / 2 } \sin \frac { \theta _ { i } } { 2 } \left| 1 \right. , \cdot ; \theta _ { i } = \arcsin ( x _ { i } ) , \phi _ { i } = \operatorname { a r c c o s } ( x _ { i } ^ { 2 } ) .
$$

The trainable unitary matrix $W ( { \boldsymbol { \theta } } )$ is appended after the feature encoding $U _ { i } ( \mathbf { x } ) \left| 0 \right.$ The circuit $W ( { \boldsymbol { \Theta } } )$ is capable of adapting through training, regardless of the initial state |0⟩.

$$
U ( \mathbf { x } , \mathbf { \Theta } ) = W ( \mathbf { \Theta } \mathbf { \Theta } \mathbf { \Theta } U _ { \mathrm { e n c } } ( \mathbf { x } ) .
$$

The resulting parameterized quantum state $| \psi ( \mathbf { x } , { \boldsymbol { \theta } } ) \rangle$ is

$$
\vert \psi ( \mathbf { x } , \mathbf { \Theta } \mathbf { \Theta } \mathbf { \Theta } ) \rangle ~ = ~ U ( \mathbf { x } , \mathbf { \Theta } \mathbf { \Theta } ) \vert 0 \rangle ^ { \otimes n } ~ = ~ W ( \mathbf { \Theta } ) \left( \bigotimes _ { i = 1 } ^ { n } R _ { Z } ^ { ( i ) } ( \operatorname { a r c c o s } ( x _ { i } ^ { 2 } ) \big ) ~ R _ { Y } ^ { ( i ) } ( \arcsin ( x _ { i } ) ) \right) \vert 0 \rangle ^ { \otimes n } .
$$

## A.1.2 Gradients per bag on Q-LLP

We explain how to compute the parameter gradients in a weakly supervised setting, which is learning from the label proportion (LLP) (Quadrianto et al. 2008). LLP allows the strength of the supervision signal to be adjusted. As shown in Fig. B, images are first processed by feature extraction, which are subsequently reduced to the qubit dimension n using Principal Component Analysis (PCA), resulting in an instance representation $\mathbf { x } \in \mathbb { R } ^ { n }$ . The multiple input instances in a bag $\mathbb { X } = \{ \mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { i } , . . . , \mathbf { x } _ { m } \}$ are encoded from the initial state |0⟩ using $R _ { Y }$ and $R _ { Z }$ gates via angle encoding, resulting in $U _ { e n c } ( \mathbf { x } ) \left| 0 \right. ^ { \otimes n }$

Each instance goes through the weight unitary matrix $W ( { \boldsymbol { \theta } } )$ , which consists of multiple layers of $R _ { X } , R _ { Z }$ with learnable parameters ${ \pmb \theta } = [ \theta _ { 1 } , . . . ] ( \theta \in \mathbb { R } )$ , and dense matrix gates for quantum superposition between each qubit. Then it obtains a quantum state as follows.

$$
\left. \psi ( \boldsymbol { \Theta } ) \right. = { W } ( \boldsymbol { \Theta } ) { U } ( \mathbf { x } ) \left. 0 \right. ^ { \otimes n }
$$

From the states $\left\{ \left| \psi _ { 1 } \right. , . . . , \left| \psi _ { i } \right. , . . . , \left| \psi _ { m } \right. \right\}$ of the bag, we construct the density matrix

$$
\hat { \rho } ( \boldsymbol { \Theta } ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left| \psi _ { i } ( \boldsymbol { \Theta } ) \right. \left. \psi _ { i } ( \boldsymbol { \Theta } ) \right|
$$

From the true $C .$ -class proportion $\mathbf { p } = ( p _ { 1 } , \ldots , p _ { q } )$ of the bag, the target density matrix is

$$
\rho : = \sum _ { k = 1 } ^ { C } p _ { k } \left| k \right. \left. k \right|
$$

Here $| k \rangle$ is the computational basis state corresponding to class k. From the viewpoint of reducing the number of qubits and computation cost, the class $k \in { 1 , \dots , C }$ is represented in binary with $n = \lceil \log _ { 2 } C \rceil$ qubits, and the basis state corresponding to that bitstring is associated with the class. For example, in the case of qubit number $n =$ 2 and class number C = 4, |00⟩ → class 1, |01⟩ → class 2, |10⟩ → class 3, |11⟩ → class 4.

We use the Hilbert-Schmidt inner product for fidelity $F ( \boldsymbol { \theta } )$ between the target density matrix $\rho$ and the density matrix $\hat { \rho } ( \boldsymbol { \Theta } )$ :

$$
\begin{array} { l } { { \displaystyle F ( \pmb \theta ) = \mathrm { T r } \big ( \rho \hat { \rho } ( \pmb \theta ) \big ) } \ ~ } \\ { { \displaystyle ~ = \mathrm { T r } \left( \rho \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | \psi _ { i } ( \pmb \theta ) \rangle \langle \psi _ { i } ( \pmb \theta ) | \right) } \ ~ } \end{array}
$$

![](images/62da64575dbc7732004f0956eb3a6e13ba7afa65de1cb7d0eb07cb098ae0b07d.jpg)  
Fig. B Overview of our Q-LLP.

$$
= { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \operatorname { T r } ( \rho \mid \psi _ { i } ( \Theta ) \rangle \langle \psi _ { i } ( \Theta ) | ) .
$$

By applying the properties of the trace $\operatorname { T r } ( A \ | \psi \rangle \langle \psi | ) = \langle \psi | \ A | \psi \rangle$ ，

$$
F ( \pmb \Theta ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left. \psi _ { i } ( \pmb \Theta ) \right| \rho \left| \psi _ { i } ( \pmb \Theta ) \right.
$$

The loss function per bag is defined as

$$
\mathcal { L } _ { \mathrm { b a g } } ( \Theta ) = - \log ( F ( \Theta ) + \varepsilon )
$$

where $\varepsilon$ is a small constant for numerical stability. The training goal is to minimize loss function: min $( \mathcal { L } _ { \mathrm { b a g } } ( \Theta ) )$ . The gradient of each parameter $\theta _ { j }$ is obtained as $\frac { \partial \mathcal { L } ( \pmb { \theta } ) } { \partial \theta _ { j } }$

$$
\frac { \partial \mathcal { L } _ { \mathrm { b a g } } ( \Theta ) } { \partial \theta _ { j } } = - \frac { 1 } { F ( \Theta ) + \varepsilon } \frac { \partial F ( \Theta ) } { \partial \theta _ { j } } .
$$

The derivative of fidelity is

$$
{ \frac { \partial F ( \pmb { \Theta } ) } { \partial \theta _ { j } } } = { \frac { \partial } { \partial \theta _ { j } } } { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \left. \psi _ { i } ( \pmb { \Theta } ) \right| \rho \left| \psi _ { i } ( \pmb { \Theta } ) \right. = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } { \frac { \partial } { \partial \theta _ { j } } } \left. \psi _ { i } ( \pmb { \Theta } ) \right| \rho \left| \psi _ { i } ( \pmb { \Theta } ) \right.
$$

From the derivative of a product(f · g)<sup>′</sup> = f<sup>′</sup> · g + f · g<sup>′</sup>

$$
\frac { \partial } { \partial \theta }  \psi | \rho | \psi  = \langle \frac { \partial } { \partial \theta } \psi | \rho | \psi  +  \psi | \rho \big | \frac { \partial } { \partial \theta } \psi 
$$

From ρ<sup>†</sup> = ρ ⇒ ρ = ρ ⇒ ⟨a| ρ |b⟩ = ⟨b| ρ<sup>†</sup> |a⟩ = ⟨b| ρ |a⟩

$$
{ \begin{array} { r l r } { { \frac { \partial } { \partial \theta } }  \psi  \rho  \psi  = \langle { \frac { \partial } { \partial \theta } } \psi  \rho  \psi  + { \overline { {  { \frac { \partial } { \partial \theta } } \psi  \rho  \psi  } } } } & { } \\ { = 2 \operatorname { R e }  { \frac { \partial } { \partial \theta } } \psi  \rho  \psi  } & { } & { \ \cdot \ \cdot \theta \in \mathbb { R } } \end{array} }
$$

Therefore,

![](images/dd421698f14b993bfd77d453965c10a6043c818d9e590877035d8ee4be62d8e8.jpg)

![](images/d49b60b5cdddbe4eb84b7cd3438dcf35e0b643838cea6e8f43101a843c786822.jpg)  
Fig. C Test accuracy as a function of parameter count for classical and quantum classifiers when the quantum feature map does not improve kernel–label alignment $( \Delta \mathrm { K T A } < 0 )$ . In these settings, the classical representation already provides a favorable feature geometry, and both models exhibit similar scaling behavior with respect to parameter count.

$$
{ \frac { \partial F ( \pmb { \theta } ) } { \partial \theta _ { j } } } = { \frac { 2 } { m } } \sum _ { i = 1 } ^ { m } \operatorname { R e } \langle { \frac { \partial } { \partial \theta _ { j } } } \psi ( \pmb { \theta } ) | \rho | \psi ( \pmb { \theta } ) \rangle
$$

By defining $\begin{array} { r } { g _ { j } = { \langle \frac { \partial } { \partial \theta _ { i } } \psi ( \pmb { \theta } ) | \rho | } \psi ( \pmb { \theta } ) \rangle } \end{array}$ as the gradient of the i-th instance $\mathbf { x } _ { i } .$ , we obtains the final gradient $\mathbf { g } _ { i } = [ g _ { 1 } , \ldots ]$ for $\boldsymbol { \Theta } = [ \theta _ { 1 } , \ldots ]$ . Bag size $m = 1$ means supervised learning. Note, however, that an increase in bag size is not monotonically related to an increase in classification dificulty.

## A.2 Parameter eficiency and KTA

We select two representative cases, Flower102×DINOv2 and Flower $1 0 2 \times \mathrm { C L I P } ,$ , and find that only a small subset shows strong positive alignment across the 70 evaluated combinations, and analyze how the parameter count afects the classification performance of classical and quantum classifiers. Figures C show the test accuracy as a function of the number of trainable parameters for two dataset–feature extractor pairs.

The subplot reports the diference in Kernel Target Alignment between quantum and classical feature representations,

$$
\Delta \mathrm { K T A } = \mathrm { K T A } _ { \mathrm { q u a n t u m } } - \mathrm { K T A } _ { \mathrm { c l a s s i c a l } } .
$$

Datasets with positive ∆KTA indicate that the quantum feature map improves the alignment between the kernel matrix and labels, suggesting that the induced Hilbert space provides a more separable representation.

Among the datasets with large ∆KTA, only Flowers102 with CLIP features shows a clear advantage of the quantum classifier at small parameter counts. This behavior suggests that these features are particularly compatible with the quantum feature space.

In contrast, the STL10 results exhibit large variance for the classical model. Across 30 random seeds, the optimization often remains near chance-level accuracy, indicating strong sensitivity to initialization.

Overall, these results suggest that the benefit of quantum classifiers depends on the compatibility between the input feature geometry and the quantum feature space.

![](images/9069d3e016c04f8b6dc9f4d589da38e008cdfc2ff1038a0f6330977d7e594abf.jpg)

![](images/c5df49007b1796d82c95e8a9bd13c1c1497cd541f6396b6124c7dd5e3081dc90.jpg)

![](images/9d2689c42173c1e1ed431cb3ffc5edaf5a36b5147a88031e78be48ff12b0143c.jpg)  
Fig. D Test accuracy (We report mean and standard deviation over random seeds. The setting is fixed to: CIFAR10×CLIP (Frac = 0.1), Flowers102×CLIP (Frac = 1.0), and Food101×CLIP (Frac $= 0 . 1 )$ .

When the representation aligns well with the quantum feature map, competitive performance can be achieved with fewer parameters.

## A.3 Sensitivity to bag size

Figure D shows how the bag size m afects the final test accuracy in LLP training. Overall, the classical model is relatively stable across m in CIFAR10 and Food101, while the quantum model tends to degrade as m increases, with larger variance at large bag sizes.

For CIFAR10 and Food101, the quantum model is competitive at small bag sizes $( m \in \{ 1 , 3 , 5 \} )$ ), but accuracy decreases for larger bags, suggesting that aggregating many instances into one supervision unit can weaken optimization signals for the current quantum parameterization. For Flowers102, both models drop at very large bag size (m = 20), indicating that excessive bag aggregation can harm optimization regardless of model family in this setting.

These results are consistent with the gradient formulation in Sec. A.1.2: increasing m averages instance-level contributions within each bag, which can reduce gradient magnitude and increase optimization dificulty when instance diversity inside bags is high. Practically, this suggests using small-to-moderate bag sizes as a robust default, and tuning m jointly with feature extractor and dataset characteristics.

## A.4 Computation resouces

All experiments were run on a Linux server equipped with 8 NVIDIA A100-SXM4 GPUs (40 GB memory each), two AMD EPYC 7742 64-core ${ \mathrm { C P U s } } ,$ and 1.0 TiB system memory. Feature extraction used CUDA, where available, with a batch size of 128 and 4 data-loading workers. The quantum classifier was executed using Qulacs on a CPU.

Training and inference for a quantum classifier take about 24.5 hours in a setting with 1 seed and 500 iterations.

## A.5 Licenses

We provide a summary of the subset and the license for the dataset used in our experiments.

<table><tr><td>Dataset/resource</td><td>Version/subset used</td><td>License status</td></tr><tr><td>CIFAR-10</td><td>4-class subsets</td><td>Non-Commercial Use</td></tr><tr><td>Fashion-MNIST</td><td>4-class subsets</td><td>MIT</td></tr><tr><td>SVHN</td><td>4-class subsets</td><td>Non-Commercial Use</td></tr><tr><td>STL-10</td><td>4-class subsets</td><td>Non-Commercial Use; images acquired from ImageNet</td></tr><tr><td>Oxford-IIIT Pet</td><td>4-class subsets</td><td>C BY-SA 4.0</td></tr><tr><td>Oxford Flowers 102</td><td>4-class subsets</td><td>Non-Commercial Use</td></tr><tr><td>Food-101</td><td>4-class subsets</td><td>Scientific fair use</td></tr><tr><td>MedMNIST</td><td>multiple subsets</td><td>CC BY 4.0; DermaMNIST is CC BY-NC 4.0</td></tr><tr><td>QMNIST</td><td>4-class subsets</td><td>BSD-style license</td></tr></table>

Table A Dataset license status summary.