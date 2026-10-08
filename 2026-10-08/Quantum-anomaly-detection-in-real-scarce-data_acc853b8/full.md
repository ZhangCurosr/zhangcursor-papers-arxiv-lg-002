# Quantum anomaly detection in real scarce data

Emanuele Casciaro<sup>1,</sup> <sup>\*</sup>, Fabio Mascherpa<sup>2</sup>, Alfonso Amendola<sup>2</sup>, and Filippo Caruso<sup>1</sup>

<sup>1</sup>Department of Physics and Astrophysics, University of Florence, Via Sansone, 1, Sesto Fiorentino, 50019, Italy <sup>2</sup>DICOX/C High Performance Computing Center of Excellence, DIT Digital & Information Technology, Eni S.p.A., Via Emilia 1, San Donato Milanese, 20097, Italy   
<sup>\*</sup>emanuele.casciaro@unifi.it

## ABSTRACT

Anomaly detection on small and unbalanced datasets remains very challenging in machine learning, although this scenario is common in several domains, including healthcare, cybersecurity, finance, and energy. Data augmentation and generative AI may mitigate training-data scarcity, but they often fall short because anomalies are, by definition, unpredictable, rare, and highly diverse events compared to high-probability normal data. Overfitting to pseudo-anomalies, model collapse, high-dimensional data, uninterpretable black-box models, and validation challenges are typical issues limiting their practica applicability. In this context, quantum machine learning may provide a promising and more sustainable avenue because it can enable more interpretable models with far fewer trainable parameters and smaller datasets, implementable on energy-efficient quantum hardware. Here, we propose a novel two-step hybrid classical–quantum architecture for sequential data and test it on a realistic scenario in the global energy-transition domain, i.e., automated anomaly detection in large-scale photovoltaic plants. The achieved generalization capability and competitive prediction accuracy may pave the way for new hybrid learn ing models able to exploit the continuously increasing power of cloud-available and more sustainable quantum accelerators integrated with more traditional energy-hungry High Performance Computing resources.

## Introduction

In the context of learning theory, anomaly detection (AD) addresses the problem of identifying whether a sample (or a time window) is normal or anomalous, whereas anomaly classification (AC) aims to assign an anomalous sample to a specific fault type. An anomaly is commonly defined as an observation that does not conform to the expected behavior of the system<sup>1</sup>. In real-world settings, AD/AC datasets are therefore highly unbalanced (anomalies are rare by definition), and crucially, anomaly labels are often scarce or expensive to obtain (i.e., requiring expert inspection), which can severely limit solutions based on labeled data. As the reliability of digital systems becomes a critical aspect, AD has become increasingly relevant in recent research, with notable applications in fields ranging from cybersecurity<sup>2,</sup> <sup>3</sup> to financial fraud detection<sup>4</sup>, IoT systems<sup>5</sup>, smart cities<sup>6</sup> and energy applications<sup>7</sup>, to name just a few. In industrial contexts, AD and AC systems find valuable use in predictive maintenance, one common application being in large-scale solar photovoltaic plants<sup>8</sup>. Predictive maintenance aims to prevent or reduce plant downtime due to major malfunctions by repairing or replacing defective components at the first signs of divergence from their expected behavior. Such anomalies in component operation may be undetectable or too costly to track through human inspection but easily recognized by sensors or imaging. This can be exploited by devising software capable of recognizing unexpected behavior from real-time sensor data or automatically inspecting panels for visible and non-visible defects<sup>9</sup> (i.e., by thermal imaging) and sending warnings to prompt a timely response when needed.

A more complex formulation often related to the problem of predictive maintenance involves dealing with sequential data, often provided by a large number of sensors. The time dimension present in the so-called time series anomaly detection (TS-AD) introduces additional challenges due to the dynamic nature of the system, which makes it more difficult to reliably define and consequently identify an anomaly. In TS-AD, unexpected behavior can be classified into three main categories<sup>10</sup>. Point anomalies, or outliers, are the simplest category. These are individual data points that are inherently anomalous, in the sense that they do not belong to the general data distribution. Contextual anomalies are more difficult to identify; a data point that would not be anomalous per se becomes so in relation to its specific neighborhood or context. Finally, collective anomalies are anomalies that concern a group of data points that are close to each other and collectively deviate from the underlying distribution. In most TS-AD applications, it is more interesting to capture the last two types of anomalies, which can only be identified by studying the correlations emerging among data points in the sequence over time.

In recent years, many machine learning algorithms have been proposed to solve the AD problem, leveraging assumptions about the data distribution to identify outliers. The isolation forest algorithm<sup>11</sup>, for instance, flags as anomalies the most isolated sampled points and is used in the predictive maintenance of IoT-aided PV systems<sup>12</sup>. Other methods flag as anomalies those instances that fail to associate with any other point in the dataset<sup>13</sup>, or learn a decision boundary based on previously seen sets comprising only non-anomalous data. The One-Class Support Vector Machine (SVM)<sup>14</sup>, for example, adapts the SVM formulation to the AD task and is widely used in monitoring large industrial plants<sup>15</sup>. However, all these methods are only capable of dealing with point anomalies and are not suitable for processing temporal information; in the context of TS-AD, the Autoregressive Integrated Moving Average (ARIMA) family of models<sup>16</sup> allows processing sequential information and is commonly used to monitor individual critical components (e.g., a turbogenerator<sup>17</sup>). Basic ARIMA models can only handle univariate sequences, but the related and more general Vector Autoregression family of models<sup>18</sup> enables working with sequences of vectors as well, extending the scope of the methods to more realistic settings, including those with complex, multivariate data.

Recent advances in ML have led to the development of deep learning methods, which employ large parameterized models to learn a latent representation of the data and subsequently use this information to learn the desired target function. A widely used pattern in standard unsupervised AD is the autoencoder architecture<sup>19</sup>, which learns to reconstruct non-anomalous data from a lower-dimensional representation and flags as anomalous any data point whose reconstruction error exceeds a certain threshold. This approach assumes that the dataset can be described by a set of latent variables, which are learned and used to fit a training set containing only sane data due to the scarcity of anomalous samples. Other techniques build upon a similar concept to exploit the scarcity of labeled anomalies; in<sup>20</sup>, a deep learning architecture is used to learn a latent representation of a dataset alongside a minimal hypersphere S that contains all points, signaling at test time every point projected by the model that does not belong to S as an anomaly, with promising results in industrial AD benchmarks on real-time production images<sup>21</sup>.

Over the years, several deep learning architectures have been proposed to deal with correlations within data sequences; the Convolutional Neural Network (CNN)<sup>22</sup> method learns several one-dimensional convolutional filters from the data, allowing the extraction of some information from the sequence, at the cost of fixing the size of the learned sequence; this lightweight model can be successfully applied in the monitoring of simple, non-interconnected systems with short-term memory, as in<sup>23</sup>, where a convolutional model is used in the predictive maintenance of mechanical equipment, using the data relative to their vibrations. The Recurrent Neural Network (RNN)<sup>24</sup> method, which is frequently used in AD in PV plants<sup>25</sup>, improves this aspect by storing a hidden state vector h, which is updated every time a new element is processed, allowing the encoding of sequences with indefinite length, at the cost of an exponentially vanishing gradient as the sequence grows, due to the repeated operations on the same weight matrix. The Long Short-Term Memory (LSTM) cell<sup>26</sup> enables a partial update of the recurrent cell, leading to the gradients vanishing more slowly and therefore larger treatable datasets; one of the most recent and popular solutions is the Transformer architecture<sup>27</sup>, which encodes every element of a sequence with arbitrary length using a non-linear mapping of the query, key and value" representation, and then computes a final embedding from them, taking into account every other element and its intersections within the sequence. This method is successfully demonstrated in<sup>28</sup> as a means to reliably isolate occurrences of anomalous energy output of PV systems caused by harsh climate conditions.

A common trend is the adoption of increasingly larger models as the size of the problem grows. Although this reflects the increasing complexity of the task at hand, this choice is not always optimal due to the new set of challenges that may arise when dealing with a high number of trainable parameters. These include a more expensive training procedure that requires more time and more data, and less predictable, more black-box behavior as the models size and complexity grow.

Quantum Machine Learning (QML) may offer a way around the numerical challenges presented by such problems by taking advantage of the unique characteristics of quantum algorithms to bypass some of the computational limits of classical ML methods<sup>29</sup>, including in the field of AD<sup>30</sup>. In particular, Quantum Neural Networks (QNN) have attracted much interest in the past few years. Similarly to its classical counterpart, a QNN uses a set of parameters to learn a quantum function, which is capable of expressing complex relationships not possible in classical ML by exploiting superposition and entanglement to express non-classical states<sup>31</sup>.

In this work, we try to harness the power of quantum computation to perform anomaly classification in a realistic energy transition problem, assessing the capabilities of QNNs to achieve desirable performance using smaller (in terms of parameters) models, leading to more efficient and less expensive training. The article is structured as follows: in Section , we showcase and discuss the results of a test application of our method over a realistic dataset, and in Section , the theoretical and empirical background of our work is provided, followed by the description of our hybrid training procedure.

## Results

Given that assessing the effectiveness of the models in realistic industrial settings is crucial in the development of AD techniques, the authors in<sup>32</sup> synthetically generated a dataset describing the simulation of a real solar panel power plant, inducing different kinds of faults during the simulation period, with the goal of matching a real-world scenario of deployment. In order to provide data as close as possible to a real use case, the simulation was performed using real instruments, and the errors introduced were derived from an artificially induced hardware failure. The simulation spanned a period of 16 days (Figure 1) with a sampling rate of 1 Hz, providing data gathered from two different strings of solar panels.

![](images/75056019b10175548d31157c4fefcc6b05dcd4bf53b9ea6f45b163bb06a217b9.jpg)  
Figure 1. Current level of the first string of panels sampled over the course of the experiments. The peaks represent daytime activity, while the inactive regions are associated with nighttime.

![](images/6f72b981e37094f9bcd390f815547d267dd5f0c151a4bf3b63803559eb7e84d5.jpg)

![](images/0b029dfe8306177bb98092748259b4e56ddd5a500d63ffe83382c45a3a0de123.jpg)

![](images/e53bc044db8d2b80d7e861c55cf284db5fe3ae03736c37643bb9dd413f250e32.jpg)

![](images/218a72c10168399272c8b0484b819113d2b1b3b30ad35117716e1914f8f0e7cc.jpg)  
Figure 2. Per-class feature distribution: due to the sane data being distributed all over the feature space, it is difficult to correctly identify the different types of fault instances.

Each data point describes the system status with the voltage and current values of both the strings of panels, alongside the irradiance and temperature of the system. Artificially introducing the anomalies inside the environment provides an automatic labeling of the data, which represents the kind of anomaly induced, which can be classified as the presence of a short circuit (label 1), degraded (label 2), open circuit (label 3) or shadowing (label 4), with non-faulty instances, which make up for the majority of the dataset (Figure 2), represented with label 0. The models are trained to minimize the weighted cross-entropy loss, where each class has weight $w _ { i } = 1 - r _ { i } ,$ and $r _ { i }$ is the ratio of occurrence of the class reported in Table 1. The standard metric for evaluating classification models is the accuracy, defined, given the true and false positive and negative prediction outputs TP, FP, TN and FN, as

$$
A c c = \frac { T P + T N } { T P + F N + F P + T N } ,
$$

which measures the fraction of correct predictions over all the data. However, assessing the performance of an AD model is not a trivial task: using naive metrics such as accuracy may lead to misleading information, since it does not take into account the unbalanced composition of the dataset, usually consisting of few anomalies and an abundance of regular data points, encouraging models to always flag points as non-anomalous, achieving high accuracy, while being clearly under-performing for the desired task. A more encouraged metric is the

$$
F _ { 1 } = \frac { 2 T P } { 2 T P + F N + F P }
$$

![](images/17620408b0c77d54e6d19fb7cc966605228f2ca222bdd44b6d46ce8100579fab.jpg)  
Figure 3. Different hybrid interactions analyzed: no interactions (left) is compared against a hybrid model where the quantum layer is employed only at the classification level (ours, center), or from the beginning of the training (right). The black and red lines represent the flow of the input during the first and second training phase.

score, which gives the same importance to both $\begin{array} { r } { p r e c i s i o n = \frac { T P } { T P + F P } } \end{array}$ and $\begin{array} { r } { r e c a l l = \frac { T P } { T P + F N } } \end{array}$ , rewarding different behaviors in classification. Although this metric takes into consideration the imbalance of the data, it does not depend on the number of true negatives, representing, even if to a reduced magnitude, a different bias in the evaluation. A more comprehensive metric is the balanced accuracy, which is the mean of the true positive rate and true negative rate, which indicates the probability of the model classifying a sample as each class given the ground truth.

$$
A c c _ { b } : = \frac { 1 } { 2 } \left( \frac { T P } { T P + F N } + \frac { T N } { T N + F P } \right)
$$

The problem, expressed in both the anomaly detection and classification formulation, is used to benchmark the two-step training architecture described in Section , which first learns a more compact representation of the data through a classical network and then uses a hybrid classifier, which shares components with the compression network, to perform prediction over the learned representation. The model performances are compared with two different architectures. The first is a fully classical architecture, constructed following<sup>33</sup> by replacing the quantum layer of the hybrid classifier with a fully connected layer with equivalent latent dimension size $N = 2 ^ { Q }$ , with Q being the size of the latent dimension of the learned representation and, consequently, the number of qubits used. Since the reconstruction objective function used for learning the representation is unsupervised, it is cheap to train and could potentially be considered fixed once trained with enough data. Achieving good performance with a quantum model implies the possibility of training a reliable supervised AD and AC model using few training samples, with a lower risk of overfitting compared to a classical-only solution. The other model follows<sup>33</sup> by using the QNN layer as an additional layer in the compression network, shifting the training of the quantum layer from the classification-only objective to also include the reconstruction task. At the cost of a more expensive initial training due to the presence of the quantum layer, the classification training becomes more straightforward, with only a small classification head to train. These variations, summarized in Figure 3, are done with the goal of assessing whether the quantum layers actually benefit the hybrid networks, and analyzing their contribution according to their position inside the architecture.

In the AD task, whose results are reported in Table 2, both the classical solution and our hybrid implementations perform similarly well, achieving near-perfect scores in all metrics, resulting in accurate predictions reported in Figure 4.

While the scores for our scheme and the fully classical case are similar, our hybrid classification layer employs far fewer parameters, reducing their number by a factor of ∼ 40, lowering both the risk of overfitting and the training cost. On the other hand, the application of the quantum layer at the encoder level breaks the autoencoder structure, resulting in a similarly small number of parameters but much poorer predictions. It is also worth noting that following the construction of the classical classifier described above, the parameter gain from the classical to the hybrid model would actually be much greater; however, since both classical and hybrid models achieve near-perfect scores, the key question to rank their performance becomes which model can solve the problem with fewer parameters and hence lower training costs. In this specific instance, the size of the classical classifier can be reduced from ≈ 267k to ≈ 4.4k by exploiting the structure of this problem without affecting the quality of the solutions. Therefore, in more complex tasks, assuming both classical and hybrid models retain comparable performances, we could have observed an exponential gain in parameter efficiency, up to a factor of ≈ 2.4k.

In the AC case, our hybrid model manages to achieve better test-time scores than the classical model, as reported in Table 3. Similarly to the AD case, it again outperforms the hybrid autoencoder–based approach, suggesting that quantum layers may yield different outcomes when placed in different positions in a hybrid neural network. In the per-class error breakdown reported in Table 4, we can observe that the hybrid model is less prone to incorrectly classifying sane data as anomalies, resulting in an overall reduction in false negatives. Even in the distinction between sane data (class 0) and data associated with the partial obscuration anomaly (class 4), which is hard to correctly identify, the hybrid model outperforms the classical counterpart. Similarly to the AD case, even though the hybrid model generally outperforms the classic model, the key difference in this case is the number of parameters employed in the classification head of the network: while the classic model requires $\mathcal { O } \left( 2 ^ { q } \right)$ parameters, the hybrid model achieves comparable inference capabilities by requiring only $\mathcal { O } \left( q \right)$ learnable quantum gates: this exponential reduction in the number of weights can result in a faster fine-tuning of the network and a lower risk of overfitting the training data, leading to more reliable models. It is notable that the number of quantum parameters required in both AD and AC cases is the same, the only difference being the size of the final layer adapting to the number of possible classes.

![](images/2819635b7985a72fc1501ddaab5f079e2066853df2c9a6436428ba63aae9528d.jpg)

![](images/90426e4a8c861f67edbf314c9d101513efb1b81e16b6480655002fe27d68ecb9.jpg)  
Figure 4. Anomaly detection over the last day of sampling. Both classic (left) and hybrid model (right) perform well, with the hybrid model avoiding some misclassification at the start of the sequence.

To assess the relevance and effectiveness of different circuit constructions, we tested our architecture using different ansätze, which are described in Table 5. From the results, summarized in Figure 5, it is possible to observe that circuit con struction is a major source of variability in classification performance: a simple circuit, given enough re-uploading blocks, outperforms more sophisticated circuits while keeping the number of required operations small, reducing noise in real-hardware applications and making the method suitable even for testing on currently available noisy intermediate-scale quantum (NISQ) devices.

## Methods

A generic quantum algorithm applies a series of operations $\mathcal { U } _ { i }$ to an initial reference state |0⟩, evolving it into the final state $\begin{array} { r } { \left| \psi \right. = \left( \prod _ { i } \mathcal { U } _ { i } \right) \left| 0 \right. } \end{array}$ , which is then measured to retrieve a classical value corresponding to the result of interest. In QNNs, the input data is encoded using a feature map, a parameterized operator that transforms the initial state into $| x \rangle = \mathcal { U } _ { \Phi ( x ) } \left| 0 \right.$ , and then processed using a trainable parameterized operator $V _ { \theta }$ , which adds degrees of freedom to the learned quantum function. Similarly to how it is possible to stack more perceptrons into a multi-layer perceptron<sup>34</sup>, which adds more expressive power to the network, one can repeat the feature map and variational layer, obtaining the final circuit $\begin{array} { r } { \mathcal { U } ( x ; \pmb { \theta } ) = \prod _ { i } V _ { \pmb { \theta } _ { i } } \mathcal { U } _ { \Phi ( x ) } } \end{array}$ , creating the data-reuploading scheme<sup>35</sup>. Most implementations use, as output of the model, the expectation value of an observable ${ \mathcal { O } } ,$ resulting in the output value

$$
y = \langle 0 | \mathcal { U } ( { \theta } ; x ) ^ { \dag } \mathcal { O } \mathcal { U } ( \theta ; x ) | 0 \rangle .
$$

This can then be used to compute a loss function, which evaluates the model and enables optimization of the parameters through classical algorithms, i.e., the gradient descent update rule. This scheme, similar to the classical case, has been successfully applied in different contexts<sup>36–38</sup>, showing promising results in several benchmarking problems. However, these methods are fairly limited in problem size due to the hardware required for handling larger problems, making them ill-suited for real-world tasks. Another limitation of QNNs is their ability to handle sequential information: feature maps capable of expressing positional information require supplementary qubits (Figure 6), making it impossible to operate with arbitrarily long sequences. The Quantum RNN model<sup>38</sup> tries to solve the problem by storing the state information in a fixed-size register every time a different time-step is encoded; however, for larger sequences the circuit becomes too deep, requiring mechanisms for qubit resets and introducing additional operational overhead, which hinders performance. In the Quantum $\yen 5 T M ^ { 39,40 }$ the state vector is classical, and different quantum circuits are used to decide the update of the cell, eliminating the scaling issue at the cost of reducing the contribution of quantum effects to the state evolution. A suitable solution is to rely on hybrid models that can combine the adaptability of classical architecture for processing sequential information into a fixed-size embedding, used by the quantum circuit, that processes it in a more complex state space. Hybrid models have applications in classification<sup>41,</sup> <sup>42</sup>, anomaly detection<sup>33,</sup> <sup>43</sup>, and forecasting<sup>44</sup> with competitive results compared to standard NN architectures, while also being able to reduce the number of qubits required for processing the data, and hence the amount of resources required.

![](images/ab82e7d6ef8f12849a5b71fc22a3c194d7293a60d2a11e681637706324bcbdec.jpg)  
Figure 5. $A c c _ { b }$ of each configuration of ansatz construction: without changing the measurement of the circuit (in our case $\langle Z _ { i } \rangle ) .$ , the family of states $| \psi ( x ; \theta ) \rangle$ associated with the circuit can have a great impact on the output of the layer, affecting the classification performances

![](images/1b113c3fb850622cc4aec3225477d553552fbc2a384ce65af5f169a2a852f4aa.jpg)  
Figure 6. Circuit representation of the feature map encoding a sequence of $d = 3$ dimensional data of length $T = 4 ,$ . Using different embedding strategies is possible to reduce the amount of qubits required for each element of the sequence, at the expense of a deeper circuit.

Inspired $\boldsymbol { \mathrm { b y } } ^ { 3 3 }$ , where the authors used a hybrid network to compress the data, which is then used to perform unsupervised AD using the isolation $f o r e s t ^ { 1 1 }$ algorithm, we employ a transformer-based autoencoder to learn a more compact representation of the sequence, made of an encoder $E : \mathbb { R } ^ { d }  \mathbb { R } ^ { p }$ , which compresses the sequence, and a decoder $D : \mathbb { R } ^ { p }  \mathbb { R } ^ { d }$ that recreates the original input and is trained over the mean squared error of the data reconstruction

$$
\mathcal { L } ( E , D , \vec { x } _ { t } ) = \sum _ { t = 1 } ^ { T } \left| \left| x _ { t } - D \left( E ( \vec { x } _ { t } ) _ { t } \right) \right| \right| ^ { 2 } .
$$

The autoencoder is based on the Transformer architecture, which uses the self-attention mechanism to compute an embedding for each element of the sequence based on its relationship with the other components: the attention mechanism is masked using a triangular matrix, in order to avoid non-causal interactions. The encoder is composed of several stacked blocks, which reduce the dimension of the data at each step, and the decoder retains a symmetric structure, restoring the initial dimensionality.

Once the encoder learns a suitable representation for the reconstruction task, the decoder is discarded, and the encoder is fine-tuned alongside a hybrid classifier H, comprised of a QNN layer alongside a shallow classical prediction. The classifica tion training minimizes the weighted cross-entropy loss

$$
\mathcal { L } ( E , H , \{ x _ { t } \} _ { t = 1 } ^ { T } , y ) = - \sum _ { k = 1 } ^ { K } \lambda _ { k } y ^ { ( k ) } \log \Big [ H \left( A g g \left( E \left( \{ x _ { t } \} _ { t = 1 } ^ { T } \right) \right) \right) \Big ] ^ { ( k ) } ,
$$

with $\lambda _ { k } > 0$ the weight of each class, used to counter the class imbalance, and $A g g : \mathbb { R } ^ { T \times p }  \mathbb { R } ^ { p }$ an aggregation function: different aggregators have been tested, including uniform and weighted mean $\begin{array} { r } { A g g \left( \left\{ x _ { t } \right\} _ { t } \right) = \sum _ { t } w _ { t } x _ { t } } \end{array}$ , with $\begin{array} { r } { w _ { t } = \frac { 1 } { T } } \end{array}$ in the case of uniform mean, otherwise a learnable parameter; however, the selected aggregator is a last-element selector $A g g \bar { ( } \{ x _ { t } \} _ { t } ) = x _ { T }$ due to the similar classification outcome and lightweight cost.

The classifier is a hybrid network comprising a trainable quantum circuit and a classical prediction head, which helps the quantum classifier adapt the output to the problem specifics without post-network processing and aids in abstracting the design of the measurement process from the dimensionality of the problem, delegating the task of reconstructing the right output from the measurement to a classical processor.

The quantum circuit, as depicted in Figure 7, is a QNN layer comprised of a feature map which encodes the data using the angle embedding scheme over the Pauli X axis, resulting in the state

$$
\left. x \right. = \otimes _ { k } \left( \cos \frac { x _ { k } } { 2 } \left. 0 \right. - i \sin \frac { x _ { k } } { 2 } \left. 1 \right. \right) ,
$$

where $\otimes _ { k }$ denotes the tensor product of all qubits, labeled by k. The quantum state evolves using a set of single-qubit rotations over the Pauli axes Y and X, with a set of controlled-X gates applied in a ring connectivity, which introduces entanglement in the circuit. The whole scheme is repeated R times according to the data-reuploading scheme, and at the end of the circuit the $\langle Z \rangle$ expectation value of each qubit is measured, ensuring an output o bounded in norm by $\| o \| _ { \infty } \leq 1$ . The prediction head is a linear $p r o b e ^ { 4 5 }$ , with the sole purpose of computing the logit vector from the expectations of the qubits, relaxing the circuit design from the constraint #classes $\leq \# q u b i t s$ , using the most information about the output state.

By using this two-step training procedure, summarized in Figure 8, it is possible to work efficiently with sequences using hybrid networks. Once we are able to extract a sufficient amount of fixed-size information from the sequence, we can use the quantum classifier to process it in a more expressive computational space, even in the case of real-world complex problems.

## Implementation Details

Both hybrid and classical solutions are implemented using $\mathrm { P y T o r c h } ^ { 4 6 }$ , with the aid of Pennylane<sup>47</sup>, used in the realization and training of the quantum circuits. All the solutions tested were trained for 10 epochs with the reconstruction task and 15 epochs over the classification objective. Training, validation and test split are obtained through sub-sampling of the whole dataset ensuring to evenly distribute the anomalies, concentrated towards the end of the dataset, across all the splits.

![](images/1939c6715ff65e5e91f67e61f959db8532b71df5061ad1689c924e803f77866b.jpg)

Figure 7. Ansatz used during the training: the data is encoded through the angle embedding $\left| x \right. = \otimes _ { i } R _ { X } ( x _ { i } ) \left| 0 \right.$ and processed using a variational layer.  
![](images/34f20d5539656160bf33985eec4d60685408c158ea075843975e3ceb82dd8b31.jpg)  
Figure 8. Training scheme of the proposed architecture: once the autoencoder learns a suitable representation of the data, the decoder is replaced by the hybrid classifier.

Table 1. Per-class distribution of each type of anomaly. The Sane state makes up for almost all the dataset size.
<table><tr><td colspan="6">Data incidence (%)</td></tr><tr><td>Sane 0.847</td><td>S. Circuit 0.004</td><td>Degradation 0.008</td><td>O. Circuit 0.004</td><td>Obscuration 0.137</td><td>All Anomalies 0.153</td></tr></table>

Table 2. Experiment results for the binary prediction task. Our proposed model and its classical counterpart perform similarly well, while the HAE model fails to reliably detect anomalies.
<table><tr><td>Model</td><td>Acc</td><td> $A c c _ { b }$ </td><td>F1</td><td>#Parameters</td></tr><tr><td>AE + QNN Classifier (ours)</td><td>0.9952</td><td>0.989</td><td>0.9955</td><td>82</td></tr><tr><td>Hybrid AE + Classical Classifier</td><td>0.7796</td><td>0.5</td><td>0.8617</td><td>18</td></tr><tr><td>AE + Classical Classifier</td><td>0.9956</td><td>0.9877</td><td>0.9959</td><td>≈44001</td></tr></table>

<sup>1</sup> Despite using fewer parameters than required by the model construction, the final score is not affected.

Table 3. Classification scores in the AC task of the proposed models: while the HAE model has fewer parameters in the classifier with respect to our model, it shares a similar parametrized structure in the encoder.
<table><tr><td>Model</td><td> $A c c$ </td><td> $A c c _ { b }$ </td><td>F1</td><td>#Parameters</td></tr><tr><td>AE + QNN Classifier (ours)</td><td>0.9357</td><td>0.9761</td><td>0.9418</td><td>109</td></tr><tr><td>Hybrid AE + Classical Classifier</td><td>0.8515</td><td>0.9293</td><td>0.8789</td><td>45</td></tr><tr><td>AE + Classical Classifier</td><td>0.8784</td><td>0.9549</td><td>0.9005</td><td>≈4400</td></tr></table>

Table 4. Per-class accuracy score of the best performing models: our solution is less prone to flagging sane instances as anomalous data, resulting in a low false-positive count.
<table><tr><td>Model</td><td>Sane</td><td>S. Circuit</td><td>Degradation</td><td>O. Circuit</td><td>Obscuration</td></tr><tr><td>AE + QNN Classifier (ours)</td><td>0.9481</td><td>0.9967</td><td>0.974</td><td>1</td><td>0.9653</td></tr><tr><td>AE + Classical Classifier</td><td>0.8881</td><td>0.9967</td><td>0.9846</td><td>1</td><td>0.9734</td></tr></table>

<table><tr><td>#</td><td>Strategy</td><td>Entanglement</td></tr><tr><td>1</td><td> $A E _ { X } ( x )$ </td><td>No entanglement</td></tr><tr><td>2</td><td> $A E _ { X } ( x ^ { 3 } ) A E _ { X } ( x ^ { 2 } ) A E _ { X } ( x )$ </td><td>No entanglement</td></tr><tr><td>3</td><td>Controlled  $A E _ { X } ( x )$ </td><td>Ring connectivity</td></tr><tr><td>4</td><td>Controlled  $A E _ { \vec { \sigma } } ( x )$ </td><td>Ring connectivity</td></tr></table>

<table><tr><td>#</td><td>Strategy</td><td>Entanglement</td></tr><tr><td>1</td><td> $R _ { X } R _ { Y }$ </td><td>CX Ring</td></tr><tr><td>2</td><td> $R _ { X } R _ { Y } H$ </td><td>CX Ring</td></tr><tr><td>3</td><td> $R _ { Z } R _ { X } R _ { Y } H$ </td><td>CX Ring</td></tr><tr><td>4</td><td> $R _ { \vec { \sigma } }$ </td><td>No entanglement</td></tr></table>

Table 5. Encoding and variational strategies used in the construction of the ansatz; each encoding is tested in combination with every variational layer proposed. $A E _ { a } ( \cdot )$ denotes the angle encoding with respect to the axis a, with ⃗σ indicating a general Pauli axis.

## Discussion

We presented an application of hybrid QNNs in the context of anomaly detection and classification, targeting the reliability of PV systems, showing that hybrid techniques can achieve competitive performance compared to classical models while using far (sometimes even exponentially) fewer parameters. To effectively use QNNs for AD on sequential data, we proposed a twostep training method that couples a QNN with an autoencoder, which is initially trained to learn a more compact and suitable representation of the sequence. This representation can then be used by the quantum layer to obtain the final prediction. By applying this method to a dataset mimicking real-world solar-plant faults, we achieve better performance than similar hybrid algorithms, with prediction accuracies comparable to classical approaches but significantly greater parameter efficiency in the classification head. This improvement carries over to the AC task, enabling the classification of highly unbalanced data using few parameters and encouraging better inference-time generalization capabilities, which are crucial for supporting the massive scaling required by the global energy transition. Our results also serve as a direct example of a near-term quantum application in anomaly detection, consolidating a framework that enables viable solutions with restricted computing resources. We can then deduce that quantum classification can benefit from classical learned preprocessing of the data, further improved by the shared-network structure between the compression and classification networks, which allows the hybrid model to modify the learned representation, shifting emphasis from the reconstruction task toward the classification objective.

## References

1. Chalapathy, R. & Chawla, S. Deep learning for anomaly detection: a survey (2019). ArXiv:1901.03407 [cs, stat].

2. Ten, C.-W., Hong, J. & Liu, C.-C. Anomaly detection for cybersecurity of the substations. IEEE Transactions on Smar Grid 2, 865–873, DOI: 10.1109/TSG.2011.2159406 (2011).

3. Hong, J., Liu, C.-C. & Govindarasu, M. Integrated anomaly detection for cyber security of the substations. IEEE Transactions on Smart Grid 5, 1643–1653, DOI: 10.1109/TSG.2013.2294473 (2014).

4. Ahmed, M., Mahmood, A. N. & Islam, M. R. A survey of anomaly detection techniques in financial domain. Futur. Gener. Comput. Syst. 55, 278–288, DOI: 10.1016/j.future.2015.01.001 (2016).

5. Said, A. M., Yahyaoui, A. & Abdellatif, T. Efficient anomaly detection for smart hospital iot systems. Sensors 21, 1026, DOI: 10.3390/s21041026 (2021).

6. Garcia-Font, V., Garrigues, C. & Rifà-Pous, H. A comparative study of anomaly detection techniques for smart city wireless sensor networks. Sensors 16, 868, DOI: 10.3390/s16060868 (2016). Publisher: Multidisciplinary Digital Publishing Institute.

7. Pérez, J. V., Chávez, M. R., Prieto, M. D. & Martínez, L. R. Recent progress of anomaly detection in energy applications: a systematic literature review. Anom. Detect. Complexities Appl. Methods, Complexities Appl. 3 (2025).

8. Zulfauzi, I. A., Dahlan, N. Y., Sintuya, H. & Setthapun, W. Anomaly detection using k-means and long-short term memory for predictive maintenance of large-scale solar (lss) photovoltaic plant. Energy Reports 9, 154–158, DOI: 10. 1016/j.egyr.2023.09.159 (2023).

9. Vlaminck, M., Heidbuchel, R., Philips, W. & Luong, H. Region-based cnn for anomaly detection in pv power plants using aerial imagery. Sensors 22, 1244, DOI: 10.3390/s22031244 (2022). Publisher: Multidisciplinary Digital Publishing Institute.

10. Sørbø, S. & Ruocco, M. Navigating the metric maze: a taxonomy of evaluation metrics for anomaly detection in time series. Data Min. Knowl. Discov. 38, 1027–1068, DOI: 10.1007/s10618-023-00988-8 (2024).

11. Liu, F. T., Ting, K. M. & Zhou, Z.-H. Isolation forest. In 2008 Eighth IEEE International Conference on Data Mining, 413–422, DOI: 10.1109/ICDM.2008.17 (2008).

12. Singh Thakur, V. & Raman, R. Efficient fault detection in renewable energy systems through iot and isolation forest algorithm. In 2023 International Conference on Innovative Computing, Intelligent Communication and Smart Electrical Systems (ICSES), 1–6, DOI: 10.1109/ICSES60034.2023.10465413 (2023).

13. Kwon, D. et al. A survey of deep learning-based network anomaly detection. Clust. Comput. 22, 949–961, DOI: 10.1007/ s10586-017-1117-8 (2019).

14. Manevitz, L. M. & Yousef, M. One-class svms for document classification. J. Mach. Learn. Res. 2, 139–154 (2002).

15. Ferreira, L., Pilastri, A., Romano, F. & Cortez, P. Using supervised and one-class automated machine learning for predictive maintenance. Appl. Soft Comput. 131, 109820, DOI: 10.1016/j.asoc.2022.109820 (2022).

16. Moschini, G., Houssou, R., Bovay, J. & Robert-Nicoud, S. Anomaly and fraud detection in credit card transactions using the arima model. Eng. Proc. 5, 56, DOI: 10.3390/engproc2021005056 (2021).

17. Kozitsin, V., Katser, I. & Lakontsev, D. Online forecasting and anomaly detection based on the arima model. Appl. Sci. 11, 3194, DOI: 10.3390/app11073194 (2021).

18. Zivot, E. & Wang, J. Vector autoregressive models for multivariate time series. In Zivot, E. & Wang, J. (eds.) Modeling Financial Time Series with S-Plus˝o, 369–413, DOI: 10.1007/978-0-387-21763-5\_11 (Springer, New York, NY, 2003).

19. Sakurada, M. & Yairi, T. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Proceed ings of the MLSDA 2014 2nd Workshop on Machine Learning for Sensory Data Analysis, 4–11, DOI: 10.1145/2689746. 2689747 (ACM, Gold Coast Australia QLD Australia, 2014).

20. Hojjati, H. & Armanfard, N. Dasvdd: deep autoencoding support vector data descriptor for anomaly detection. IEEE Transactions on Knowl. Data Eng. 36, 3739–3750, DOI: 10.1109/TKDE.2023.3328882 (2024). ArXiv:2106.05410 [cs].

21. Huang, W., Li, Y., Xu, Z., Yao, X. & Wan, R. Improved deep support vector data description model using feature patching for industrial anomaly detection. Sensors 25, 67, DOI: 10.3390/s25010067 (2025).

22. Kiranyaz, S., Ince, T., Abdeljaber, O., Avci, O. & Gabbouj, M. 1-d convolutional neural networks for signal processing applications. In ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 8360–8364, DOI: 10.1109/ICASSP.2019.8682194 (IEEE, Brighton, United Kingdom, 2019).

23. Apeiranthitis, S., Zacharia, P., Chatzopoulos, A. & Papoutsidakis, M. Predictive maintenance of machinery with rotating parts using convolutional neural networks. Electronics 13, 460, DOI: 10.3390/electronics13020460 (2024).

24. Schmidt, R. M. Recurrent neural networks (rnns): a gentle introduction and overview, DOI: 10.48550/arXiv.1912.05911 (2019). ArXiv:1912.05911 [cs].

25. Yi, C. et al. Anomaly detection of photovoltaic power generation based on quantile regression recurrent neural network. Electr. Power Syst. Res. 238, 111132, DOI: 10.1016/j.epsr.2024.111132 (2025).

26. Hochreiter, S. & Schmidhuber, J. Long short-term memory. Neural Comput. 9, 1735–1780, DOI: 10.1162/neco.1997.9.8. 1735 (1997).

27. Vaswani, A. et al. Attention is all you need. In Advances in Neural Information Processing Systems, vol. 30 (Curran Associates, Inc., 2017).

28. Wirawan, I. M., Wibawa, A. P. & Widiyanintyas, T. Photovoltaic energy anomaly detection using transformer based machine learning. Int. J. Robotics Control. Syst. 4, 1337–1352, DOI: 10.31763/ijrcs.v4i3.1260 (2024).

29. Schuld, M., Sinayskiy, I. & Petruccione, F. An introduction to quantum machine learning. Contemp. Phys. 56, 172–185, DOI: 10.1080/00107514.2014.964942 (2015).

30. Kyriienko, O. & Magnusson, E. B. Unsupervised quantum machine learning for fraud detection, DOI: 10.48550/arXiv. 2208.01203 (2022). ArXiv:2208.01203.

31. Dalzell, A. M. et al. Quantum algorithms: a survey of applications and end-to-end complexities (Cambridge University Press, 2025).

32. Lazzaretti, A. E. et al. A monitoring system for online fault detection and classification in photovoltaic plants. Sensors 20, DOI: 10.3390/s20174688 (2020).

33. Sakhnenko, A. et al. Hybrid classical-quantum autoencoder for anomaly detection. Quantum Mach. Intell. 4, DOI: 10.1007/s42484-022-00075-z (2022).

34. Goodfellow, I., Bengio, Y. & Courville, A. Deepfeedforward networks, chap. 6, 168–224 (The MIT Press, 2016).

35. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E. & Latorre, J. I. Data re-uploading for a universal quantum classifier. Quantum 4, 226, DOI: 10.22331/q-2020-02-06-226 (2020).

36. Tacchino, F. et al. Quantum implementation of an artificial feed-forward neural network. Quantum Sci. Technol. 5, 044010, DOI: 10.1088/2058-9565/abb8e4 (2020). ArXiv:1912.12486 [quant-ph].

37. Cong, I., Choi, S. & Lukin, M. D. Quantum convolutional neural networks. Nat. Phys. 15, 1273–1278, DOI: 10.1038/ s41567-019-0648-8 (2019). ArXiv:1810.03787 [quant-ph].

38. Li, Y. et al. Quantum recurrent neural networks for sequential learning. Neural Networks 166, 148–161, DOI: 10.1016/j. neunet.2023.07.003 (2023).

39. Chen, S. Y.-C., Yoo, S. & Fang, Y.-L. L. Quantum long short-term memory. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 8622–8626, DOI: 10.1109/ICASSP43922. 2022.9747369 (2022). ISSN: 2379-190X.

40. Zhou, Y., Xu, C. C., Song, M., Wong, Y. K. & Du, K. A novel quantum lstm network (2024). ArXiv:2406.08982 [quant-ph].

41. Arthur, D. & Date, P. A hybrid quantum-classical neural network architecture for binary classification, DOI: 10.48550/ arXiv.2201.01820 (2022). ArXiv:2201.01820 [cs].

42. Senokosov, A., Sedykh, A., Sagingalieva, A., Kyriacou, B. & Melnikov, A. Quantum machine learning for image classi fication. Mach. Learn. Sci. Technol. 5, 015040, DOI: 10.1088/2632-2153/ad2aef (2024).

43. Wang, M. et al. A quantum-classical hybrid solution for deep anomaly detection. Entropy 25, 427, DOI: 10.3390 e25030427 (2023). Number: 3 Publisher: Multidisciplinary Digital Publishing Institute.

44. Schetakis, N. et al. Data re-uploading in quantum machine learning for time series: application to traffic forecasting, DOI: 10.48550/arXiv.2501.12776 (2025). ArXiv:2501.12776 [quant-ph].

45. Alain, G. & Bengio, Y. Understanding intermediate layers using linear classifier probes, DOI: 10.48550/arXiv.1610.01644 (2018). ArXiv:1610.01644 [stat].

46. Paszke, A. et al. PyTorch: an imperative style, high-performance deep learning library. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, 721, 8026–8037 (Curran Associates Inc., Red Hook, NY, USA, 2019).

47. Bergholm, V. et al. Pennylane: Automatic differentiation of hybrid quantum-classical computations (2022). 1811.04968.

## Author Contributions

E.C. carried out the algorithm implementation and experiments. E.C. and F.C. conceived the methodology, while A.A. and F.C. proposed and supervised the project. All authors contributed to the discussion, analysis of the results and the writing of the manuscript.

## Additional information

## Data Availability

The dataset used in this work can be accessed at https://github.com/clayton-h-costa/pv\_fault\_dataset. The underlying code for this study is not publicly available but may be made available to qualified researchers on reasonable request from the corresponding author.

## Competing Interests

The authors declare no competing interests.

## Funding

This study received no funding.