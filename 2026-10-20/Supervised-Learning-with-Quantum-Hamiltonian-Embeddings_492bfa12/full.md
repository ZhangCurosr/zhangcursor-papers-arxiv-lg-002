# Supervised Learning with Quantum Hamiltonian Embeddings

Riccardo Molteni,<sup>1,</sup> <sup>2,</sup> ∗ Minjun Jeon,<sup>1,</sup> <sup>3,</sup> ∗ Paolo Braccia,<sup>1</sup> Abhinav Deshpande,<sup>1</sup>

Vedran Dunjko,<sup>2</sup> Liang Jiang,<sup>1,</sup> <sup>4</sup> Su-un Lee,<sup>1,</sup> <sup>4</sup> Swarnadeep Majumder,<sup>1</sup> Javier Robledo Moreno,<sup>1</sup> Parthasarathy Suryanarayanan,<sup>1</sup> and Kunal Sharma<sup>1,</sup> <sup>†</sup>

<sup>1</sup>IBM Research

<sup>2</sup>Applied Quantum Algorithms aQa<sup>L</sup> , Leiden University, The Netherlands <sup>3</sup>Department of Materials, University of Oxford, Parks Road, Oxford OX1 3PH, United Kingdom <sup>4</sup>The University of Chicago (Dated: October 20, 2026)

Machine learning is widely regarded as a promising application in which quantum computing could offer a computational advantage. However, general quantum machine learning models with provable guarantees on performance, trainability, and efficient inference remain limited. In this work, we introduce quantum Hamiltonian embeddings, a supervised learning framework in which a classical training dataset is encoded into a single effective, parameterized Hamiltonian. Training optimizes the parameters of this data-dependent Hamiltonian, whose approximate ground state defines the classification rule. Once the ground state has been approximated using a quantum algorithm, inference can be performed either classically using a classical representation of the learned state, or with a quantum representation of the state. We establish efficient methods for gradient estimation and prove favorable generalization guarantees. Furthermore, under standard cryptographic assumptions, we show that Hamiltonian embeddings can realize hypothesis functions beyond the reach of efficient classical algorithms. We then demonstrate the framework at scale on IBM Heron processors using a standard benchmark dataset. Together, these results establish Hamiltonian embeddings as a new framework for quantum machine learning, combining theoretical guarantees with a path for experimental investigations at scale on quantum hardware.

## CONTENTS

## I. Introduction

II. Hamiltonian embedding model 4   
A. Construction of the model 5   
1. Choice of the observable 5   
2. Choice of the unitary embedding 7   
3. Choice of the loss function 8   
4. Ground-state approximation 10   
B. Efficient gradient estimation and absence of barren plateaus 10   
III. Properties of the Hamiltonian embedding model 12   
A. Generalization bounds 12   
B. Classical inference and surrogate models 14   
1. Fourier surrogates 14   
2. Pauli surrogates 14   
3. Deep Clifford encodings 15   
4. Dequantizability of the model 16   
C. Encoding a classically hard problem in the Hamiltonian embedding model 17   
IV. Experiments 18   
A. The model instance used in the experiments 18   
B. Results 19   
V. Discussion 20   
Acknowledgments 21   
References 21   
A. Energy-based generalization bounds 23   
1. Lipschitz assumption 27   
B. Gradient evaluation and the Hellmann–Feynman relation 28   
1. Gradients with approximate ground states 29   
C. Absence of barren plateaus in the Hamiltonian embedding mode 29   
D. Universality of the Hamiltonian Embedding model 33   
E. Solving a classically hard problem with Hamiltonian embedding 35   
1. The discrete cube root problem 35   
2. Definition of the learning problem 36   
3. Classical hardness of the learning problem 37   
4. Quantum solution with Hamiltonian embedding 37   
F. A construction with highly entangled ground states 41   
1. Entangled graph states as ground states 41   
2. Entanglement properties of the ground state 42   
3. Numerical demonstration 42   
G. A provable separation between Pauli surrogates and Hamiltonian embedding 43   
1. Idealized embedding Hamiltonian and its ground state: perfect classification 45   
2. Embedding Hamiltonian with finite training data 46   
a. Empirical accuracy of Hamiltonian embedding 47   
b. Accuracy of classical surrogates 48   
3. Required bond dimension for good classification with Hamiltonian embedding 50   
H. Numerical setup 51   
1. Encoding circuit and embedding Hamiltonian 51   
2. Classical iterative solver 52   
3. Optimization and model selection 53   
I. Image classification benchmark 53   
1. Protocol 53   
2. Experimental Baselines 53   
J. Hardware experiments 54   
1. The hardware solver 54   
2. Model training insights 54   
K. A numerical counterexample to Pauli classical surrogate models 55   
1. Construction of the learning task 56   
2. Numerical experiment 58

## I. INTRODUCTION

The prospect of quantum computational advantage in learning tasks has driven significant interest in quantum machine learning (QML) [BWP<sup>+</sup>17, CC25]. For instance, in supervised learning, multiple architectures have been proposed [HCT<sup>+</sup>19, SP21, PCGL20, RML14, JGM<sup>+</sup>24] and provable advantages have been established in particularly constructed settings [LAT21, AG26, MGD26, ZZN<sup>+</sup>26, GGPD25].

A broader goal in QML is to develop general models that are efficiently trainable, admit generalization guarantees, cannot be efficiently simulated classically in general, while also being implementable at scale on quantum processors. However, the difficulty of simultaneously satisfying these requirements is particularly evident for parameterized quantum circuits [BLSF19, CAB<sup>+</sup>21, SBSW20]. Shallow-depth PQCs are well suited for current quantum processors but can often be simulated efficiently on classical computers [CLG<sup>+</sup>25, GGPD25], whereas deeper circuits may be difficult to optimize because of barren plateaus [MBS<sup>+</sup>18, CSV<sup>+</sup>21, LTW<sup>+</sup>25]. Similarly, beyond a few examples [GGC<sup>+</sup>24, HBK<sup>+</sup>25], identifying effective quantum kernels for more general datasets remains challenging.

In this work, we introduce Hamiltonian embedding, a classification algorithm that can be efficiently implemented on a quantum processor, in which the classifier is constructed by solving a physically motivated energy-minimization problem. The method encodes the entire training dataset into a single parameterized Hamiltonian. Each training example contributes a term obtained by dressing a fixed, label-dependent operator, namely by conjugating it with a parameterized unitary that encodes the corresponding input. Then training optimizes the encoding parameters to lower the ground-state energy of this data-dependent Hamiltonian, and an approximate ground state of the optimized Hamiltonian defines the classifier. In particular, for a new input, the trained encoding specifies an observable whose expectation value with respect to the learned ground state determines the predicted label. We identify tasks for which the physical constraints imposed by the learned quantum state provide a useful inductive bias, yielding a provable reduction in sample complexity relative to classically regularized surrogate models.

We then show that the Hamiltonian embedding framework has several properties desirable for QML. During optimization, a quantum algorithm prepares approximations to the required ground states. In the case where the encoding unitaries allow the expectation values of the dressed observables to be efficiently computed from a classical representation of these ground states, both the loss and its gradients with respect to the trainable parameters can be evaluated classically. In particular, we show that the gradients can be expressed as expectation values and evaluated using a parameter-shift rule.

Shallow local encoding unitaries provide one such setting, enabling classical inference after training. For a general family of these encodings, we also show the model avoids barren plateaus for a relevant subset of parameters. Importantly, in this regime the model is not known to be classically simulable, as it involves approximating ground states of local Hamiltonians. Although the shallow encoding may allow for classical surrogates, we exhibit a task on which a specific model instance provably achieves substantially higher test accuracy than a corresponding natural surrogate, at the same training-set size. Moreover, even with shallow encoding unitaries, we show that the Hamiltonian embedding framework can implement universal approximators for measurable binary target functions. Specifically, for any such function and any desired accuracy, we provide a construction using finitely many qubits such that the hypothesis function implemented by the model achieves an arbitrarily small average approximation error under the uniform input distribution.

Irrespective of the encoding depth, we also show the model admits favorable generalization guarantees for a wide class of encoding unitaries. If training yields a sufficiently low-energy approximate ground state, matrix Hoeffding inequalities and covering number arguments establish generalization to unseen inputs. We show that for fixed generalization error, the required dataset size scales linearly, up to logarithmic factors, with the number of qubits and variational parameters.

Finally, the model can realize hypothesis functions that lie beyond the reach of efficient classical algorithms. Although preparing ground states of local Hamiltonians is hard in the worst case, even for quantum computers, we identify structured instances that encode classically hard learning problems for which the relevant ground states can nevertheless be efficiently approximated on a quantum computer. Under standard cryptographic assumptions, these instances yield explicit learning problems solvable on a quantum processor within the Hamiltonian embedding framework, while remaining classically hard.

We note that restricting the encoding unitaries to those for which the relevant dressed-observable expectation values can be evaluated classically places Hamiltonian embedding within the shadow model framework introduced in [JGM<sup>+</sup>24]. A shadow QML model uses a quantum computer during training to obtain a quantum advice state, whose classical representation then supports entirely classical inference. Whereas Ref. [JGM<sup>+</sup>24] introduces shadow models as a general framework, our construction provides an explicit method to obtain the advice state, namely as the ground state of the embedding Hamiltonian.

Ground states of data-dependent Hamiltonians have also been used for classification in [YGH<sup>+</sup>25]. There, however, the encoding is fixed and the algorithm consists only of preparing a low-energy state of the resulting dataencoding Hamiltonian, without optimizing over a trainable family of Hamiltonians. The Hamiltonian embedding framework introduced here instead learns both the advice state and the Hamiltonian whose ground state realizes it. To assess whether the model can be trained to achieve low validation and test error, we study it in numerical experiments on a standard ML task and implement it on IBM Heron processors. The results show successful optimization and competitive predictive performance relative to the quantum and classical baselines considered.

The manuscript is organized as follows.

• In Section II, we introduce the Hamiltonian embedding model and provide a complete description of the training and prediction procedures. We also discuss the choice of the embedding unitaries and observables, derive an efficient gradient-estimation procedure, and establish the absence of barren plateaus for a relevant subset of parameters.

• The main theoretical results are in Section III. We derive analytical generalization bounds and study classical surrogates of the trained model. We also identify learning tasks for which the physical constraints imposed by the learned quantum ground state act as a useful inductive bias, yielding improved sample efficiency over classically regularized surrogate models. Finally, we exhibit a learning task that is efficiently solvable within the Hamiltonian embedding framework but classically hard under a standard cryptographic assumption.

• In Section ${ \mathrm { I V } } ,$ we present numerical results for the binary Fashion-MNIST classification task. We benchmark the Hamiltonian embedding framework on systems up to 50 qubits against quantum kernels studied in Ref. [HBM<sup>+</sup>21] and classical machine-learning baselines. Across this range, the model remains trainable and achieves competitive test performance. We also implement the training procedure on IBM quantum processors, using hardware-generated samples in the Sample-based Krylov Quantum Diagonalization ground-state solver $[ \mathrm { Y M I } ^ { + } 2 5 ,$ , MN25, SKI<sup>+</sup>25], and demonstrate successful training on a 50-qubit instance.

## II. HAMILTONIAN EMBEDDING MODEL

Although our framework can be readily extended to multiclass classification, here we address the problem of binary classification. In particular, we are given a training set $\mathcal { T } = \{ ( x _ { \ell } , y _ { \ell } ) \} _ { \ell = 1 } ^ { N } ,$ , where each input $x _ { \ell }$ is sampled from a distribution $\mathcal { D }$ over $\Omega \subseteq \mathbb { R } ^ { d }$ , and the corresponding label is generated by an unknown function $f : \Omega \to \{ 0 , 1 \}$ , such that $y _ { \ell } = f ( x _ { \ell } )$ . The goal of the learning algorithm is to infer an approximate map $\tilde { f }$ that, with high probability, agrees with the true mapping $f$ on previously unseen test data sampled from the same distribution $\tilde { \mathcal { D } } .$

Given a dataset $\tau$ , the model constructs a family of dataset-dependent hypothesis functions as follows. On an n-qubit Hilbert space, we define an encoding unitary $U ( x , \alpha )$ parameterized by both the input x and trainable parameters $\boldsymbol { \alpha } \in \mathbb { R } ^ { M }$ , together with a family of label-dependent operators $\{ O ( y ) \} _ { y } ,$ , assigning a measurement operator $O ( y )$ to each class label $y \in \{ 0 , 1 \}$ . Then, for each data point $z = ( x , y )$ , with ${ \dot { y } } = f ( x )$ , we define a corresponding Hamiltonian term as $\hat { H _ { z } ( \alpha ) } = U ^ { \dagger } ( x , \alpha ) O ( y ) U ( x , \alpha )$ . The data Hamiltonian on the whole dataset is defined as

$$
H _ { \mathcal { T } } ( \alpha ) = \frac { 1 } { N } \left( \sum _ { \ell = 1 } ^ { N } \Big ( I - U ^ { \dagger } ( x _ { \ell } , \alpha ) O ( y _ { \ell } ) U ( x _ { \ell } , \alpha ) \Big ) \right)\tag{1}
$$

$$
= I - \frac { 1 } { N } \sum _ { \ell } U ^ { \dagger } ( x _ { \ell } , \alpha ) O ( y _ { \ell } ) U ( x _ { \ell } , \alpha )\tag{2}
$$

$$
= I - \frac { 1 } { N } \sum _ { \ell } H _ { z _ { \ell } } ( \alpha ) .\tag{3}
$$

For each $\alpha ,$ we define $\vert \psi _ { T , \alpha } \rangle$ to be the ground state of $H _ { \mathcal { T } } ( \alpha )$ . The state $\vert \psi _ { T , \alpha } \rangle$ is then used to construct the corresponding hypothesis function implemented by the model. In particular, given a new input point $x ^ { \prime } ,$ , we apply the encoding unitary $U ( x ^ { \prime } , \alpha )$ to $| \psi _ { T , \alpha } \rangle$ and measure it with respect to $O ( y )$ , for $y \in \{ 0 , 1 \}$ . As we show in Section II $\mathbf { A } 1 _ { \ast }$ we can choose ${ \cal \dot { O } } ( 0 ) + { \cal O } ( 1 ) = \dot { \cal I } ,$ , so that the expectation values of these positive operators on the encoded state define a normalized probability distribution over the labels. Specifically, the probability $P _ { \psi _ { T , \alpha } , \alpha } ( x ^ { \prime } , y ) \in [ 0 , 1 ]$ assigned to label y for input $x ^ { \prime }$ is given by the following expression:

$$
P _ { \psi _ { T , \alpha } , \alpha } ( x ^ { \prime } , y ) = \mathrm { T r } \Big [ O ( y ) U ( x ^ { \prime } , \alpha ) | \psi _ { T , \alpha } \rangle \langle \psi _ { T , \alpha } | U ^ { \dagger } ( x ^ { \prime } , \alpha ) \Big ] .\tag{4}
$$

The corresponding hypothesis function assigns a label $\tilde { y } \in \{ 0 , 1 \}$ to the input $x ^ { \prime }$ as follows:

$$
\tilde { f } _ { \mathcal { T } , \alpha } ( x ^ { \prime } ) = \mathop { \arg \operatorname* { m a x } } _ { y \in \{ 0 , 1 \} } P _ { \psi _ { \mathcal { T } , \alpha } , \alpha } ( x ^ { \prime } , y ) .\tag{5}
$$

For a given training set $\tau ,$ , the resulting family of candidate hypotheses is therefore $\mathcal { H } _ { \mathcal { T } } = \{ \tilde { f } _ { \mathcal { T } , \alpha } : \alpha \in \mathbb { R } ^ { M } \}$ We note that the family of hypothesis functions depends on the training data through the ground states of $H _ { \mathcal { T } } ( \alpha )$ Moreover, $\mathcal { H } _ { \mathcal { T } }$ is a subset of the broader, data-independent family of hypothesis functions obtained by allowing arbitrary normalized n-qubit states in the same classifier construction.

The model selects a hypothesis from $\mathcal { H } _ { \mathcal { T } }$ by choosing α according to the following training procedure. Let us consider the case when $0 \preceq O ( y ) \preceq I$ for every $y \in \{ 0 , 1 \}$ , then $\begin{array} { r } { 0 \preceq \frac { 1 } { N } \sum _ { \ell } H _ { z _ { \ell } } ( \alpha ) \preceq I } \end{array}$ . This implies that the energy of $H _ { \mathcal { T } } ( \alpha )$ is always lower bounded by zero for any α. In particular, the energy can reach the minimum value zero if the ground state $\vert \psi _ { T , \alpha } \rangle$ of $H _ { \mathcal { T } } ( \alpha )$ is such that $\langle \psi _ { \mathcal { T } , \alpha } \vert U ^ { \dagger } ( x , \bar { \alpha } ) O ( y ) U ( x , \alpha ) \vert \psi _ { \mathcal { T } , \alpha } \bar { \rangle } = 1$ for every training point $\left( x _ { \ell } , y _ { \ell } \right)$ in $\tau$ The optimal parameter vector $\alpha ^ { \star }$ is then chosen to minimize the ground-state energy over the family of Hamiltonians $\{ H _ { T } ( \hat { \alpha } ) \} _ { \alpha }$ . As we see below, this guarantees the correct classification of the training data. Moreover, in Section III $\scriptstyle \mathrm { A , }$ we show that our model also satisfies favorable generalization bounds.

In particular, let us define the empirical energy cost function as:

$$
\widehat { C } _ { \mathcal { T } } ( \psi , \alpha ) \equiv \mathbb { E } _ { z \in \mathcal { T } } \left[ \ell _ { \mathrm { e n e r g y } } ( \psi , H _ { z } ( \alpha ) ) \right] = \langle \psi | H _ { \mathcal { T } } ( \alpha ) | \psi \rangle ,\tag{6}
$$

where $\ell _ { \mathrm { e n e r g y } }$ is the energy loss, defined as:

$$
\ell _ { \mathrm { e n e r g y } } ( \psi , H _ { z } ( \alpha ) ) = 1 - \langle \psi | H _ { z } ( \alpha ) | \psi \rangle .\tag{7}
$$

Then the training objective is to find the parameter value $\alpha ^ { \star }$ and the state $\psi _ { T , \alpha ^ { \star } }$ that minimize the cost function $\widehat { C } \tau ( \psi , \alpha )$ . We note that the algorithm may not reach the exact ground state of the training Hamiltonian $H _ { T } ( \alpha ^ { \star } )$ . In other words, the learned state $\psi _ { T , \alpha ^ { \smash { \prime } } }$ ⋆ may only be an approximate ground state. However, as we show in Section III $\mathbf { A } ,$ it is sufficient for the learned state to achieve a low energy with respect to the training Hamiltonian: a small empirical energy cost already guarantees good performance on unseen input points. In practice, this optimization is carried out through an alternating nested procedure. Starting from an initial value of $\alpha ,$ the algorithm first computes the corresponding ground state of $H _ { \mathcal { T } } ( \alpha )$ and then updates α to decrease the cost. Repeating this procedure yields a sequence of parameter values and associated ground states until convergence to a final pair $( \alpha ^ { \hat { \star } } , \psi _ { T , \alpha ^ { \star } } )$ . We detail the whole learning procedure in Algorithm 1. Although reaching zero energy loss guarantees correct classification on the training dataset, a decrease in the empirical energy cost ${ \widehat { C } } _ { T }$ as α is updated does not necessarily lead to a simultaneous increase in classification accuracy. We discuss this mismatch in Section II A 3, where we also consider alternative loss functions that more directly target the classification error.

Once training is complete, the learned parameters $\alpha ^ { \star }$ and ground state $\left| \psi _ { T , \alpha ^ { \star } } \right.$ define the hypothesis used to predict labels for unseen inputs. An important feature of the Hamiltonian embedding framework is that, for particular classes of encoding unitaries $U ( x , { \hat { \alpha } } )$ , these predictions can be evaluated entirely classically. As discussed in Section III B, this is achieved by constructing a classical surrogate $\sigma ( \psi _ { T , \alpha ^ { \star } } )$ of the learned quantum model from a finite number of measurements on the quantum device. The corresponding classical inference procedure is summarized in Algorithm 2.

## A. Construction of the model

The Hamiltonian embedding framework provides substantial freedom in how the encoding Hamiltonian is constructed, allowing problem-specific choices to be incorporated. In the following, we outline practical guidelines for selecting the label-dependent observable $O ( y )$ , the encoding unitary $U ( x , \alpha )$ , the choice of the loss function and the ground-state optimization method.

## 1. Choice of the observable

The label-dependent observable $O ( y )$ is measured at inference time to determine which label $y \in \{ 0 , 1 \}$ should be assigned to a given input x. Recall that the energy loss in Eq. (7) is minimized when, for every training point $\left( x _ { \ell } , y _ { \ell } \right)$ with $y _ { \ell } = f ( \bar { x _ { \ell } } )$ , the learned ground state $| \psi _ { T , \alpha ^ { \star } } \rangle$ satisfies

$$
\mathrm { T r } [ O ( y _ { \ell } ) U ( x _ { \ell } , \alpha ^ { \star } ) | \psi _ { \mathcal { T } , \alpha ^ { \star } } \rangle \langle \psi _ { \mathcal { T } , \alpha ^ { \star } } | U ^ { \dagger } ( x _ { \ell } , \alpha ^ { \star } ) ] = 1 .\tag{8}
$$

Algorithm 1 Supervised Training via Hamiltonian Embedding   
Input:   
1. Training set $\mathcal { T } = \{ ( x _ { \ell } , y _ { \ell } ) \} _ { \ell = 1 } ^ { N } , x _ { \ell } \in \mathbb { R } ^ { d }$ and $y _ { \ell } \in \{ 0 , 1 \}$   
2. Encoding unitary $U ( x , \alpha ) .$   
3. Desired training energy $\varepsilon \geq 0 ,$ learning rate $\eta > 0 ,$ maximum number of iterations $T _ { \mathrm { m a x } } .$   
Output: Trained parameters $\alpha ^ { \star }$ and final state $| \bar { \psi } _ { \mathcal { T } , \alpha ^ { \star } } \rangle$   
1: $\begin{array} { r } { O ( y ) \gets \frac { 1 } { 2 } \left[ I + ( - 1 ) ^ { y } A \right] } \end{array}$ ▷ Construct the label observable   
2: $\alpha _ { 0 } \gets \mathrm { U n i f } ( [ - 1 , 1 ] ^ { M } )$ ▷ Initialize the variational parameters   
3: $t  0$   
4: while $t < T _ { \mathrm { m a x } }$ do   
5: for $\ell = 1$ to N do   
6: $\boldsymbol { z } _ { \ell } \gets ( x _ { \ell } , y _ { \ell } )$   
7: $H _ { z _ { \ell } } ( \alpha _ { t } ) \gets U ^ { \dagger } ( x _ { \ell } , \alpha _ { t } ) O ( y _ { \ell } ) U ( x _ { \ell } , \alpha _ { t } )$ ▷ Embed the training point   
8: Store a classical representation of $\dot { H } _ { z _ { \ell } } ( \alpha _ { t } )$   
9: end for   
10: $\begin{array} { r } { H _ { \mathcal { T } } ( \alpha _ { t } ) \gets I - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } H _ { z _ { \ell } } ( \alpha _ { t } ) } \end{array}$ ▷ Construct the current embedding Hamiltonian   
11: $\left| \psi _ { T , \alpha _ { t } } \right.$ arg min $\langle \dot { \psi } | \dot { H } _ { T } ( \alpha _ { t } ) | \psi \rangle$ ▷ Find an approximate ground state   
|<sup>ψ</sup>⟩   
12: $E _ { t } \gets \langle \psi _ { T , \alpha _ { t } } | H _ { T } ( \alpha _ { t } ) | \psi _ { T , \alpha _ { t } } \rangle$ ▷ Evaluate the current training energy   
13: if $E _ { t } \leq \varepsilon$ then   
14: break   
15: end if   
16: for $\ell = 1$ to N do   
17: $m _ { \alpha _ { t } , \psi _ { T , \alpha _ { t } } } ( x _ { \ell } , y _ { \ell } ) \gets \langle \psi _ { T , \alpha _ { t } } | H _ { z _ { \ell } } ( \alpha _ { t } ) | \psi _ { T , \alpha _ { t } } \rangle$ ▷ Evaluate the correct label score   
18: end for   
19: Evaluate $\widehat { C } _ { T } ( \psi _ { T , \alpha _ { t } } , \alpha _ { t } )$ according to Eq. (6)   
20: for $j = 1$ T<sub>to M do</sub>   
21: $g _ { j } ( t ) \gets \frac { \partial } { \partial \alpha _ { j } } \widehat { C } _ { \mathcal { T } } \left( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha \right)$ ▷ Estimate the gradient via parameter shift,   
$\mid _ { \alpha = \alpha _ { t } }$ keeping the state fixed   
22: end for   
23: $g ( t ) \gets ( g _ { 1 } ( t ) , \ldots , g _ { M } ( t ) )$   
24: $\alpha _ { t + 1 }  \alpha _ { t } - \eta g ( t )$ ▷ Update the variational parameters   
25: $t \gets t + 1$   
26: end while   
27: $\alpha ^ { \star }  \alpha _ { t }$   
28: $\begin{array} { r } { H _ { \mathcal { T } } ( \alpha ^ { \star } ) ^ { \cdot }  I - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } H _ { z _ { \ell } } ( \alpha ^ { \star } ) } \end{array}$   
29: ψ <sub>,α</sub>⋆   arg min $\langle \psi | \dot { H } \tau ( \dot { \alpha } ^ { \star } ) | \psi \rangle$   
ψ   
30: return $\left( \alpha ^ { \star } , | \psi _ { T , \alpha ^ { \star } } \rangle \right)$

```latex
Algorithm 2 Classical inference via Hamiltonian embedding
Input:
1. New input point $x ^ { \prime } .$
2. Trained parameters $\alpha ^ { \star } .$
3. Classical representation $\sigma ( \psi _ { T , \alpha ^ { \star } } )$ of the learned state $\left| \psi _ { T , \alpha ^ { \star } } \right.$
4. Label-dependent observable $O ( y )$ for $y \in \{ 0 , 1 \}$
Output: Predicted label $\tilde { y } \in \{ 0 , 1 \}$
1: for $y \in \{ 0 , 1 \}$ do
2: $\widetilde { \cal O } ( x ^ { \prime } , y ) \longleftarrow U ^ { \dagger } ( x ^ { \prime } , \alpha ^ { \star } ) { \cal O } ( y ) U ( x ^ { \prime } , \alpha ^ { \star } )$ ▷ Construct observable in Eq. (39) for $x ^ { \prime }$ associated with label y
3: $\begin{array} { r } { \widetilde { O } ( x ^ { \prime } , y )  \sum _ { i } c _ { i } ( x ^ { \prime } , y , \alpha ^ { \star } ) P _ { i } } \end{array}$ ▷ Decompose $O ^ { \prime }$ into local Pauli operators
4: $\begin{array} { r } { p _ { y } ( x ^ { \prime } )  \sum _ { i } c _ { i } ( x ^ { \prime } , y , \alpha ^ { \star } ) \mathrm { T r } [ P _ { i } \sigma ( \psi _ { \tau , \alpha ^ { \star } } ) ] } \end{array}$ ▷ Estimate the label score using the classical representation $\dot { { \boldsymbol \sigma } } ( \psi _ { T , { \boldsymbol \alpha } ^ { \star } } )$
5: end for
<sup>6:</sup> <sup>y˜</sup> ← <sup>arg</sup> ${ \mathrm { m a x } } _ { y \in \{ 0 , 1 \} } p _ { y } ( x ^ { \prime } )$ ▷ Output the label with largest score
7: return $\tilde { y }$
```

This occurs when, for every training example $\scriptstyle x _ { \ell } ,$ the encoded state $U ( x _ { \ell } , \alpha ^ { \star } ) \left| \psi _ { T , \alpha ^ { \star } } \right.$ lies in the eigenvalue-1 eigenspace of the observable $O ( y )$ associated with the correct label $y \in \{ 0 , 1 \}$ We note that Eq. (8) provides a sufficient, but not necessary, condition for correct classification. Since prediction in Eq. (5) is performed by selecting the label associated with the largest expectation value, it is enough that the score assigned to the correct label exceeds that of the other label. To give these observables a direct probabilistic interpretation, we can impose the natural condition $O ( 0 ) + O ( 1 ) = I ,$ so that their expectation values in $\operatorname { E q . }$ (4) define a normalized probability distribution over the two classes. This condition is equivalent to parameterizing the two label observables in terms of a single Hermitian operator A as

$$
O ( y ) = \frac { I + ( - 1 ) ^ { y } A } { 2 } ,\tag{9}
$$

where $y \in \{ 0 , 1 \}$ labels the two classes. We take A to be normalized such that its spectrum lies in $[ - 1 , 1 ] .$ , with extremal eigenvalues $\lambda _ { \operatorname* { m i n } } ( A ) = - 1$ and $\lambda _ { \operatorname* { m a x } } ( A ) = 1$ . We note that an operator A with this normalization can always be obtained whenever the extremal eigenvalues $\lambda _ { \operatorname* { m i n } }$ and $\lambda _ { \mathrm { m a x } }$ of the original Hermitian operator are known, by applying the affine transformation $\begin{array} { r } { A \mapsto \frac { 2 \breve { A } - ( \lambda _ { \operatorname* { m a x } } + \lambda _ { \operatorname* { m i n } } ) I } { \lambda _ { \operatorname* { m a x } } - \lambda _ { \operatorname* { m i n } } } } \end{array}$ . The choice of A plays a central role in the construction of the model, as it determines how the two classes are represented in Hilbert space and can therefore be tailored to the learning task under consideration. We note however that the condition $O ( \bar { 0 } ) + O ( 1 ) = I$ is not necessary to define the model as labels can be assigned by comparing the expectation values of the two operators, without interpreting these values as probabilities. As we show in Appendix F and Appendix $G ,$ the choice of $A ,$ and $O ( y )$ in general, can influence properties of the learned ground state, such as its entanglement structure, and can thereby impact the classical simulability of the resulting model.

From a practical point of view, the choice of the operator A and the corresponding observable $O ( y )$ affects the form of the final embedding Hamiltonian. Since conjugation by a constant-depth unitary $U ( x , \alpha )$ increases the support of a local observable only by a constant factor, choosing A to be local ensures that the resulting embedding Hamiltonian remains local and with polynomially many terms.

In our numerical experiments, we choose the operator A to be a simple diagonal Hamiltonian of the form

$$
A = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } .\tag{10}
$$

With this convention, the model perfectly classifies the training set whenever, for each training example, the learned ground state $\left| \psi _ { T , \alpha ^ { \star } } \right.$ is mapped by the data-encoding unitary to a state in the eigenspace of $\check { O ( y ) }$ associated with the correct label. In the particular case of the choice in Eq. (10), a zero energy cost is obtained when, for each training point $( x _ { \ell } , y _ { \ell } ) \in \mathcal { T }$

$$
U ( x _ { \ell } , \alpha ^ { \star } ) \left| \psi _ { \mathcal { T } , \alpha ^ { \star } } \right. = \left\{ \begin{array} { l l } { | 0 ^ { n } \rangle } & { y _ { \ell } = 0 , } \\ { | 1 ^ { n } \rangle } & { y _ { \ell } = 1 . } \end{array} \right.\tag{11}
$$

As discussed above, the choice of A can influence the properties of the ground states learned by the model. For example, in Appendices F and $G ,$ we present alternative choices of $O ( y )$ that lead to highly entangled learned ground states.

## 2. Choice of the unitary embedding

As discussed above, the embedded Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ is determined by the choice of the parameterized encoding unitary $U ( x , \alpha )$ . This choice influences the expressiveness of the model, and different encoding strategies may be better suited to different learning tasks. Since training requires the ground state of the embedding Hamiltonian to be prepared at each optimization step, the corresponding quantum ground-state preparation procedure should be efficient. A necessary condition is that the embedding Hamiltonian admit an efficient classical description suitable as input to the quantum ground-state preparation algorithm. For example, when $O ( y )$ can be expressed as a sum of local Pauli operators, choosing $U ( x , \alpha )$ to have constant depth ensures that each data-dependent term $H _ { ( x , y ) } ( \alpha ) = U ^ { \dag } ( x , \alpha ) O ( y ) U ( x , \alpha )$ remains of bounded locality $O ( 1 )$ . In Appendix D, we show that the model achieves universality even with constant-depth encodings U and local $O ( y )$ , using a finite number of qubits whose scaling however need not be efficient. We stress, however, that a constant-depth encoding unitary is not strictly required by our framework. For instance, ground-state solvers based on quantum phase estimation only require the ability to efficiently implement controlled time evolution under the target Hamiltonian. Therefore, the encoding unitaries

$U ( x , \alpha )$ may have depth growing with the system size, provided that they admit an efficient circuit description from which the corresponding Hamiltonian evolution can be implemented efficiently. In particular, this description need not be restricted to a decomposition in terms of Pauli operators.

Another important consideration in the choice of the parameterized unitary $U ( x , \alpha )$ is that the cost function should admit efficient differentiation through the parameter-shift rule, as will be discussed in Section II B. For this to hold, the trainable parameters should enter the circuit through gates generated by operators with a finite, typically twopoint, spectrum, such as Pauli rotations of the form $e ^ { - i \smile P / 2 } \mathrm { w i t h } P ^ { 2 } = I .$

Keeping all of this in consideration, in the numerical experiments in this work we choose the following encoding. The unitary $U ( x , \alpha )$ is taken to be a constant-depth circuit composed of layers of bounded-locality Pauli rotations, whose angles depend on both the input $x \in \mathbb { R } ^ { \hat { d } }$ and the trainable parameters $\boldsymbol { \alpha } \in \mathbb { R } ^ { M }$ . Concretely, we consider circuits of the form:

$$
U ( \boldsymbol { x } , \alpha ) = \prod _ { l = 1 } ^ { K } \prod _ { g = 1 } ^ { G } \exp \left[ - \frac { i } { 2 } \left( \alpha _ { g , 0 } ^ { l } + \sum _ { r = 1 } ^ { d } \alpha _ { g , r } ^ { l } x ^ { r } \right) P _ { g } ^ { l } \right] ,\tag{12}
$$

where $K = O ( 1 )$ denotes the number of layers, $G = O ( n )$ the number of parameterized gates per layer, and $P _ { g } ^ { \ell }$ are bounded-locality Pauli operators. The products are taken according to the prescribed ordering of the gates in the circuit. In the numerical experiments, we instantiate Eq. (12) with $\breve { K } = 3$ with each layer applying, on every qubit $j ,$ an independently parameterized $Z _ { j } - Y _ { j } - Z _ { j }$ sequence of single-qubit rotations, followed by a nearest-neighbor $Z _ { j } Z _ { j + 1 }$ rotation. This architecture satisfies the locality requirement by construction: conjugating a one-local observable through a constant-depth local circuit only expands its support to a constant-size light cone. At the same time, because every trainable gate is a Pauli rotation, derivatives of the cost can be evaluated using the parameter-shift rule.

Finally, we note that the choice of the encoding unitary $U ( x , \alpha )$ , together with the observables $O ( y )$ , determines whether inference on new input points can be performed entirely classically. This is the case, for instance, when $U ( x , \alpha )$ has constant depth and $\hat { O ( y ) }$ is local. Efficient classical inference is not restricted to this setting, and certain classes of deeper encoding unitaries also admit efficient classical surrogates. We discuss these different regimes in Section III B. Furthermore, in the classically hard task discussed in Section III C, neither the embedded Hamiltonian nor the dressed observable $\widetilde { O } ( y )$ is local. Nonetheless, the construction still admits an efficient quantum procedure for preparing the relevant ground state, as well as a problem-specific classical procedure for representing and classically evaluating the corresponding hypothesis.

## 3. Choice of the loss function

The Hamiltonian embedding framework naturally defines the energy loss introduced in Eq. (7). For fixed training parameters α and state $\psi _ { T , \alpha } ,$ , its empirical average over the training set is precisely the energy of the corresponding embedding Hamiltonian $\dot { H } \tau ( \alpha )$ . This choice is particularly convenient, since minimizing the loss is directly connected to finding low-energy states of the embedded Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ . Moreover, reaching zero energy loss on a training example guarantees that the corresponding point is classified correctly. However, the energy loss is not identical to the classification error and, in particular, a decrease in energy is not necessarily accompanied by a simultaneous increase in classification accuracy. To see this, let us consider the prediction rule in Eq. (5). In order for a point x to be correctly classified, it is sufficient that the model finds a state $\psi _ { T , \alpha ^ { \star } }$ such that $P _ { \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } } ( x , y ) > 0 . 5$ if y is the correct label. This happens whenever the margin $m _ { \alpha ^ { \star } , \psi _ { T , \alpha ^ { \star } } } ( x , y ) = \langle \psi _ { T , \alpha ^ { \star } } | U ^ { \dagger } ( x , \alpha ^ { \star } ) { \cal O } ( y ) U ( x , \alpha ^ { \star } ) | \psi _ { T , \alpha ^ { \star } } \rangle$ is larger than 0.5. The training classification error is then defined as

$$
\widehat { R } _ { \mathcal { T } } ( \psi , \alpha ) = \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } \mathbb { 1 } \left[ m _ { \alpha , \psi } ( x _ { \ell } , y _ { \ell } ) \leq 0 . 5 \right] ,\tag{13}
$$

which only depends on whether each example lies on the correct side of the decision boundary. We recall that, at each optimization step $t ,$ the state $\psi _ { T , \alpha _ { t } }$ is held fixed while the encoding parameters α are updated. For a fixed state $\psi ,$ minimizing the energy loss $\ell _ { \mathrm { e n e r g y } }$ in $\mathrm { E q . ~ } ( 7 )$ favors parameters α that increase the margins $m _ { \alpha , \psi } ( x , y )$ of the training examples $( x , y ) \in \tau$ . However, since the energy loss $\ell _ { \mathrm { e n e r g y } }$ is linear in $m _ { \alpha , \psi } ( x , y )$ , it continues to reward increases in the margin even after a sample has already been classified correctly, i.e. when $m _ { \alpha , \psi } ( x , y ) >$ 0.5. Consequently, the model may continue to lower the energy by increasing the confidence assigned to points that are already correctly classified, without this necessarily leading to a corresponding reduction in the empirical classification error of Eq. (13).

Directly minimizing Eq. (13) is not well suited to gradient-based optimization, since the error $\hat { R } _ { { \mathcal T } }$ changes only when the score $m _ { \alpha , \psi } ( x _ { \ell } , y _ { \ell } )$ of a training example crosses the classification threshold, and therefore provides no useful gradient for most parameter values. As in classical supervised learning, one can instead optimize a continuous surrogate of the 0–1 loss that more directly reflects the classification objective. We introduce two alternative losses as possible extensions aimed more directly at reducing classification error, although the resulting parameter updates need not decrease the Hamiltonian energy.

For example, let us define a centered margin $\tilde { m } _ { \alpha , \psi } ( x , y ) = 2 m _ { \alpha , \psi } ( x , y ) - 1$ so that $\tilde { m } _ { \alpha , \psi } \in [ - 1 , 1 ]$ . We then consider the logistic loss

$$
\ell _ { \log } ( \tilde { m } ) = \log \left( 1 + e ^ { - \beta \tilde { m } } \right) ,\tag{14}
$$

where $\beta > 0$ controls the sharpness of the loss around the decision boundary, and the hinge loss

$$
\ell _ { \mathrm { h i n g e } } ^ { ( \gamma ) } ( \tilde { m } ) = \mathrm { m a x } \{ 0 , \gamma - \tilde { m } \} ,\tag{15}
$$

where $\gamma > 0$ specifies a desired classification margin. In contrast to the energy loss, the hinge loss assigns no additional penalty to examples whose margin already exceeds $\gamma ,$ , thereby focusing the optimization on misclassified points and on correctly classified points lying close to the decision boundary. Similarly, the logistic loss progressively suppresses the contribution of examples with large positive margins while assigning greater weight to misclassified examples and to points close to the decision boundary.

More generally, given a surrogate loss $\ell _ { \mathrm { c l s } } ,$ we define the corresponding empirical classification-oriented cost as

$$
\widehat { C } _ { \mathcal { T } } ^ { \mathrm { c l s } } ( \psi , \alpha ) = \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } \ell _ { \mathrm { c l s } } \left( \tilde { m } _ { \psi , \alpha } ( x _ { \ell } , y _ { \ell } ) \right) .\tag{16}
$$

Importantly, introducing this alternative cost does not modify the embedding Hamiltonian or the ground-state preparation step. At optimization iteration $t ,$ the state is still obtained as an approximate ground state of the embedding Hamiltonian,

$$
\left| \psi _ { \mathcal { T } , \alpha _ { t } } \right. \approx \arg \operatorname* { m i n } _ { \left| \psi \right. } \left. \psi \right| H _ { \mathcal { T } } ( \alpha _ { t } ) \left| \psi \right. .\tag{17}
$$

The resulting state is then kept fixed while the encoding parameters are updated using the gradient of the chosen loss,

$$
\begin{array} { r } { \alpha _ { t + 1 } = \alpha _ { t } - \eta \nabla _ { \alpha } \widehat { C } _ { \mathcal { T } } ^ { \mathrm { c l s } } \left( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha \right) \bigg | _ { \alpha = \alpha _ { t } } . } \end{array}\tag{18}
$$

We note that the Hamiltonian embedding algorithm in Algorithm 1 is recovered by choosing $\ell _ { \mathrm { c l s } } = \ell _ { \mathrm { e n e r g y } }$ . Furthermore, one may also interpolate between the two objectives through

$$
\widehat { C } _ { \tau } ^ { ( \lambda ) } = ( 1 - \lambda ) \widehat { C } _ { \mathcal { T } } + \lambda \widehat { C } _ { \mathcal { T } } ^ { \mathrm { c l s } } , \qquad \lambda \in [ 0 , 1 ] ,\tag{19}
$$

which allows the parameter optimization to balance the search for low-energy Hamiltonians with a loss that more directly targets classification performance.

The choice of the outer-loop loss therefore provides an additional degree of freedom in the Hamiltonian embedding framework: the embedding Hamiltonian determines the state used by the model, while the loss function determines how the parameters are optimized within the family of embedding Hamiltonians. Unless stated otherwise, the numerical results presented in Section IV are obtained using the canonical energy-based objective in Eq. $( 7 ) ,$ which we find to be sufficient to achieve good classification performance on the tasks considered. We also note that replacing the energy cost in Eq. (6) with either the hinge or logistic cost does not increase the complexity of gradient estimation. Indeed, for a generic differentiable surrogate loss $\ell _ { \mathrm { c l s } } ( m )$ , the derivative with respect to a variational parameter $\alpha _ { j }$ can be written using the chain rule as

$$
\frac { \partial } { \partial \alpha _ { j } } \ell _ { \mathrm { c l s } } \left( \widetilde { m } _ { \alpha , \psi } ( x , y ) \right) = \ell _ { \mathrm { c l s } } ^ { \prime } \left( \widetilde { m } _ { \alpha , \psi } ( x , y ) \right) \frac { \partial \widetilde { m } _ { \alpha , \psi } ( x , y ) } { \partial \alpha _ { j } } .\tag{20}
$$

Since $\widetilde { m } _ { \alpha , \psi } ( x , y ) = 2 \langle \psi | U ^ { \dagger } ( x , \alpha ) O ( y ) U ( x , \alpha ) | \psi \rangle - 1$ , we have $\begin{array} { r } { \frac { \partial \widetilde { m } _ { \alpha , \psi } ( x , y ) } { \partial \alpha _ { i } } = 2 \frac { \partial } { \partial \alpha _ { i } } \langle \psi | U ^ { \dagger } ( x , \alpha ) { \cal O } ( y ) U ( x , \alpha ) | \psi \rangle } \end{array}$

The first factor is a simple classical function of the margin, while the second corresponds to the derivative of the expectation value $m _ { \alpha , \psi } ( x , y ) = \langle \psi | U ^ { \dagger } ( x , \alpha ) O ( y ) U ( x , \alpha ) | \psi \rangle$

As discussed in Section II B, this derivative can be efficiently estimated through the parameter-shift rule. Therefore, the same quantum measurements used to estimate gradients of the energy cost can also be used to evaluate the gradients of the alternative surrogate losses, followed by a simple classical post-processing step.

## 4. Ground-state approximation

At the heart of the Hamiltonian embedding approach is the use of a quantum computer to prepare the ground state of the embedding Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ for varying parameter values α. Several ground-state preparation methods can be employed, and the most suitable choice depends on the structure of $\bar { H _ { \mathcal { T } } } ( \alpha )$ and on the capabilities of the available quantum device. In the general setting of local Hamiltonians, a standard approach to ground-state preparation is quantum phase estimation (QPE) [CEMM98, AL99]. Given an efficient implementation of the time evolution generated by $\dot { H } _ { \mathcal { T } } ( \alpha )$ and an initial state with non-negligible overlap with the ground state, QPE resolves the energy eigenvalues and projects the state onto the corresponding eigenspaces, producing the ground state with probability proportional to the initial overlap. When the Hamiltonian is gapped and the overlap is sufficiently large, QPE provides a systematic procedure for preparing the ground state with controllable precision.

The Sample-based Krylov Quantum Diagonalization (SKQD) algorithm offers an alternative that avoids phase estimation and also has rigorous guarantees in certain regimes [YMI<sup>+</sup>25]. In the SKQD algorithm, time-evolution circuits are applied at multiple time steps and measured in a chosen basis, such as the computational basis, producing a set of sampled bitstrings that defines a reduced subspace onto which the Hamiltonian is projected and classically diagonalized. This procedure is compatible with current quantum hardware and is robust against device noise, provided the sampled subspace retains sufficient support on the low-energy sector.

Both methods require an initial reference state with non-negligible overlap with the true ground state, ideally at least inverse-polynomial in the system size. In practice, the applicability of these ground-state solvers therefore depends on the embedding Hamiltonians remaining in a region of parameter space where reference states with sufficiently large ground-state overlap can be efficiently obtained. For structured Hamiltonian families, such reference states may be constructed by mean-field or tensor-network methods such as the density matrix renormalization group (DMRG) [Whi92, Sch05]. Identifying families of embedding Hamiltonians that simultaneously admit efficient quantum ground-state preparation while remaining difficult to simulate classically remains an important open question.

Finally, the ground state of the data Hamiltonian could also be approximated through adiabatic preparation. Assuming that a ground state of $H _ { \mathcal { T } } ( \alpha _ { 0 } )$ is available, one could obtain a ground state of $H _ { \mathcal { T } } ( \alpha _ { 1 } )$ entirely quantumly by following an adiabatic path in parameter space. Efficient preparation requires, in particular, that the spectral gap along this path remain bounded below by an inverse polynomial in the system size. Exploring choices of the embedding U and observables O for which such gapped paths can be guaranteed is an interesting direction for future work.

## B. Efficient gradient estimation and absence of barren plateaus

One of the main bottlenecks in training variational quantum ansatze, for example parameterized quantum circuits,¨ is the presence of barren plateaus in the optimization landscape [MBS<sup>+</sup>18, QWZ<sup>+</sup>23]. In this regime, gradients become exponentially small with the system size. Since gradients are estimated from measurement outcomes on a quantum computer, identifying the correct descent direction may then require exponentially many measurement shots. In the Hamiltonian embedding model, gradients can be efficiently evaluated through the parameter-shift rule. Furthermore, for a class of Pauli-rotation encodings satisfying the conditions stated below, we can ensure that a relevant subset of trainable parameters avoids barren plateaus. We also note that, in contrast to standard PQC settings, the use of a shallow encoding circuit does not by itself imply efficient classical simulability of the framework, as finding the ground state of the corresponding local data Hamiltonian may remain hard.

The gradient expression can be derived directly from the cost function in Eq. (6). As described in Algorithm 1, the optimization proceeds through a nested procedure. Given an initial parameter value $\alpha _ { 0 } ,$ , the algorithm first computes the ground state of the encoding Hamiltonian $H _ { T } ( \alpha _ { 0 } )$ , outputting $\psi _ { T , \alpha _ { 0 } }$ . The state is then kept fixed while the parameters are updated by minimizing the cost function $\widehat { C } ( \psi _ { T , \alpha _ { 0 } } , \alpha )$ . In particular, the algorithm searches for a new parameter value $\alpha _ { 1 }$ for which the energy of $H _ { \mathcal { T } } ( \alpha _ { 1 } )$ evaluated on $\psi _ { T , \alpha _ { 0 } }$ is minimized. As derived in Appendix B, this update is guided by the gradient of the cost $\widehat { C } ( \psi _ { T , \alpha _ { 0 } } , \alpha )$ in Eq. (6) with respect to parameter $\alpha _ { j }$ . Recalling the structure of $H _ { \mathcal { T } } ( \alpha )$ from Eq. (1), we obtain:

$$
\frac { \partial } { \partial \alpha _ { j } } \widehat { C } ( \psi _ { \mathcal { T } , \alpha _ { 0 } } , \alpha ) = - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } \left. \psi _ { \mathcal { T } , \alpha _ { 0 } } \middle | \frac { \partial H _ { z _ { \ell } } ( \alpha ) } { \partial \alpha _ { j } } \middle | \psi _ { \mathcal { T } , \alpha _ { 0 } } \right. ,\tag{21}
$$

where $N = | \mathcal { T } |$ is the size of the full training set. In the encoding unitary $U ( x , \alpha )$ in Eq. (12), each parameterized gate has a rotation angle that depends linearly on the input features. For notational simplicity, we omit the constant bias term in the following derivation. The bias parameters are treated identically by regarding them as coefficients of an additional constant feature $x ^ { 0 } = 1$ . For an input $x _ { i } = ( x _ { i } ^ { 1 } , \ldots , x _ { i } ^ { d } )$ , the angle of the j-th parameterized gate is

$$
\theta _ { j } ( x _ { i } , \alpha ) = \sum _ { k = 1 } ^ { d } \alpha _ { j , k } x _ { i } ^ { k }\tag{22}
$$

where $\alpha _ { j , k }$ is the trainable coefficient associated with the k-th feature in the j-th gate. We can therefore decompose the full encoding unitary around this gate as

$$
U ( x _ { i } , \alpha ) = U _ { L _ { j } } ( x _ { i } , \alpha ) U _ { j } \big ( \theta _ { j } ( x _ { i } , \alpha ) \big ) U _ { R _ { j } } ( x _ { i } , \alpha ) ,\tag{23}
$$

where $U _ { R _ { j } } ( { \boldsymbol { x } } _ { i } , { \boldsymbol { \alpha } } )$ contains the gates applied before the j-th parameterized gate, while $U _ { L _ { j } } ( { \boldsymbol { x } } _ { i } , { \boldsymbol { \alpha } } )$ contains those applied after it. Since $U _ { j }$ is a Pauli rotation gate, its dependence on $\theta _ { j } ,$ and consequently on each parameter $\alpha _ { j , k } ,$ , can be differentiated using the standard parameter-shift rule. Since differentiating the gate angle with respect to $\alpha _ { j , k }$ contributes a factor $x _ { i } ^ { k }$ , and assuming that $U _ { j }$ is a Pauli rotation gate, the parameter-shift rule gives

$$
\frac { \partial \widehat { C } } { \partial \alpha _ { j , k } } = - \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \boldsymbol { x } _ { i } ^ { k } \left[ \widehat { C } _ { i } ^ { ( j , + ) } - \widehat { C } _ { i } ^ { ( j , - ) } \right] ,\tag{24}
$$

where

$$
\widehat { C } _ { i } ^ { ( j , + ) } = \left. \psi _ { \tau , \alpha _ { 0 } } \right| U _ { R _ { j } } ^ { \dagger } U _ { j } ^ { \dagger } \left( \sum _ { k } \alpha _ { j , k } x _ { i } ^ { k } + \frac { \pi } { 2 } \right) U _ { L _ { j } } ^ { \dagger } O ( y _ { i } ) U _ { L _ { j } } U _ { j } \left( \sum _ { k } \alpha _ { j , k } x _ { i } ^ { k } + \frac { \pi } { 2 } \right) U _ { R _ { j } } \left| \psi _ { \tau , \alpha _ { 0 } } \right. ,\tag{25}
$$

$$
\widehat { C } _ { i } ^ { ( j , - ) } = \left. \psi _ { T , \alpha _ { 0 } } \right| U _ { R _ { j } } ^ { \dagger } U _ { j } ^ { \dagger } \left( \sum _ { k } \alpha _ { j , k } x _ { i } ^ { k } - \frac { \pi } { 2 } \right) U _ { L _ { j } } ^ { \dagger } O ( y _ { i } ) U _ { L _ { j } } U _ { j } \left( \sum _ { k } \alpha _ { j , k } x _ { i } ^ { k } - \frac { \pi } { 2 } \right) U _ { R _ { j } } \left| \psi _ { T , \alpha _ { 0 } } \right. .\tag{26}
$$

where $O ( y _ { i } )$ is the same labeled local observable as in Eq. (9). After the parameter-update step, the algorithm returns a new parameter value $\alpha _ { t + 1 }$ . It then constructs the updated Hamiltonian $H \bar { \tau ( \alpha _ { t + 1 } ) }$ and computes a new ground state $\psi _ { T , \alpha _ { t + 1 } }$ . This state is used to evaluate the updated cost $\widehat { C } _ { \mathcal { T } } ( \psi _ { T , \alpha _ { t + 1 } } , \alpha _ { t + 1 } )$ , and the optimization loop continues. A key point is that, during each parameter update, the current ground state is kept fixed. Thus, when computing the gradient with respect to $\alpha ,$ we do not differentiate through the dependence of $\psi _ { T , \alpha }$ on α. At iteration $t ,$ the algorithm first finds $\psi _ { T , \alpha _ { t } }$ for the Hamiltonian $H _ { \mathcal { T } } ( \alpha _ { t } )$ and then updates the parameters by minimizing the frozen-state cost $\widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha )$ This update rule is variationally well motivated. If the new parameters satisfy $\widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha _ { t + 1 } ) \leq \widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha _ { t } )$ , then, after recomputing the ground state $\psi _ { T , \alpha _ { t + 1 } . }$ , the energy cannot increase, i.e., $\widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t + 1 } } , \alpha _ { t + 1 } ) \leq \widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha _ { t } )$ . Moreover, although the ground state $\psi _ { T , \alpha }$ does depend implicitly on $\alpha ,$ the Hellmann-Feynman theorem implies that, at each iteration, the derivative of the ground-state energy at the current parameter value can be evaluated while keeping the corresponding ground state fixed, as in Eq. (21). Further details are provided in Appendix B, where we also analyze the case of approximate ground states.

As in the inference step, the gradient can also be estimated classically once an approximate ground state has been obtained from the quantum device. In particular, a classical-shadow representation of $\psi _ { T , \alpha _ { t } }$ can be used to estimate the expectation values of the two shifted observables

$$
O _ { i , j } ^ { \pm } = U _ { R _ { j } } ^ { \dag } U _ { j } ^ { \dag } \left( \sum _ { k } \alpha _ { j , k } x _ { i } ^ { k } \pm \frac { \pi } { 2 } \right) U _ { L _ { j } } ^ { \dag } O ( y _ { i } ) U _ { L _ { j } } U _ { j } \left( \sum _ { k } \alpha _ { j , k } x _ { i } ^ { k } \pm \frac { \pi } { 2 } \right) U _ { R _ { j } } .\tag{27}
$$

For constant depth $U ( x , \alpha )$ and local $O ( y ) ,$ even the dressed observables remain local, or at most low-weight, so their expectation values can be efficiently inferred from the local Pauli classical shadows. Classical shadows can estimate expectation values of local observables to inverse-polynomial precision with only polynomially many measurements.

In order to train the model efficiently, the cost-function gradients must remain sufficiently large to provide a useful optimization signal, and in particular should not vanish exponentially with the system size as in barren-plateau regimes. For a general class of shallow encodings, we show in Appendix C that, once the energy cost is strictly below $1 / 2$ by at least an inverse-polynomial amount, a situation that we observe in our numerical experiments and that corresponds to a non-trivial energy advantage, the gradient variance of the frozen-state cost admits an inversepolynomial lower bound for at least an inverse-polynomial fraction of parameters. In particular, in case of shallow encodings with linearly many parameters per layer, a constant energy advantage is sufficient to have a lower bound for a constant fraction of trainable parameters. This guarantees a useful training direction whenever the assumptions of the theorem are satisfied.

Theorem 1 (Absence of barren plateaus (informal)). Consider a Hamiltonian embedding model defined on n qubits with an observable $O ( y )$ as in Eq. (9) with $\begin{array} { r } { A = n ^ { - 1 } \sum _ { i = 1 } ^ { n } Z _ { i } , } \end{array}$ M trainable parameters $\boldsymbol { \alpha } \in \mathbb { R } ^ { M }$ , and an encoding $U ( x , \alpha )$ of depth K whosefinal layer applies one independently trainable $X - o r$ Y-rotation to each qubit, with rotation angles independent of the input x. At iteration t, keep the prepared state $\left| \psi _ { T , \alpha _ { t } } \right. .$ fixed and define

$$
\widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha ) = \langle \psi _ { \mathcal { T } , \alpha _ { t } } | H _ { \mathcal { T } } ( \alpha ) | \psi _ { \mathcal { T } , \alpha _ { t } } \rangle .\tag{28}
$$

$\begin{array} { r } { I f \widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha _ { t } ) \leq \frac { 1 } { 2 } - \gamma , } \end{array}$ for some $\gamma > 0$ , then at least $2 \gamma ^ { 2 } n$ final-layer parameters satisfy

$$
\operatorname { V a r } _ { \alpha _ { j } } \left[ \frac { \partial \widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha ) } { \partial \alpha _ { j } } \right] \geq \frac { \gamma ^ { 2 } } { 4 n ^ { 2 } } .\tag{29}
$$

Here, the variance is taken over the corresponding rotation angle uniformly distributed in [0, 2π], while the dataset, prepared state $\psi _ { T , \alpha _ { t } . }$ , and all other parameters remain fixed at their iteration-t values. Consequently, for encodings U $o f K = \Theta { \dot { ( } 1 ) }$ layer, $M = { \dot { K n } }$ parameters and $\gamma = \Theta ( 1 )$ , a constantfraction of trainable parameters has inverse-polynomial gradient variance.

The precise assumptions on the encoding U and the complete proof of the theorem are given in Appendix C.

## III. PROPERTIES OF THE HAMILTONIAN EMBEDDING MODEL

The Hamiltonian embedding model exhibits several desirable properties for a quantum machine learning model. First, we show that once trained, the model generalizes well to unseen inputs. Moreover, for certain families of encoding unitaries and label observables, the model can be evaluated entirely classically once the optimal ground state has been obtained during training. Finally, we identify a machine learning task believed to be classically hard, yet efficiently solvable on a quantum device using the Hamiltonian embedding framework.

## A. Generalization bounds

Let $\left| \psi _ { T , \alpha ^ { \star } } \right.$ be the ground state found by the algorithm after the training phase. The label for a new point $x ^ { \prime }$ is assigned by the hypothesis function $\widetilde { f } _ { T , \alpha } .$ ⋆ in Eq. (5). We define the total classification risk as the probability the model misclassifies a point x sampled from the input distribution over the whole input space $\Omega \subseteq \mathbb { R } ^ { d }$ :

$$
R ( \psi , \alpha ) = \underset { x \sim \mathcal { D } } { \mathrm { P r } } [ \widetilde { f } _ { T , \alpha } ( x ) \neq f ( x ) ]\tag{30}
$$

where we recall that f is the target labeling function and ${ \widetilde { f } } _ { T , \alpha }$ is the model implemented function for the Hamiltonian parameters $\alpha \in \Theta \subseteq { \mathbf { \bar { \mathbb { R } } } } ^ { M }$ and the corresponding reached ground state $\psi _ { T , \alpha }$ . We show that the total classification risk $\bar { \boldsymbol { R } } ( \psi , \alpha )$ can be controlled through the expected energy cost over the full input distribution, computed as $C ( \psi , \alpha ) =$ $\langle \psi | { H } _ { \mathcal { D } } ( \alpha ) | \psi \rangle$ , where $H _ { \mathcal { D } } ( \alpha ) : = \mathbb { E } _ { z \sim \mathcal { D } } [ I ^ { \mathbf { \bar { \alpha } } } - H _ { z } ( \alpha ) ]$ is the population Hamiltonian at parameter $\alpha .$ Specifically, $H _ { \mathcal { D } }$ represents the average encoding Hamiltonian associated with a random labeled input drawn from the full data distribution. In order to connect the total risk to the training procedure, we first bound the difference between the total cost energy and the empirical cost in Eq. (6):

$$
C ( \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } ) - \widehat C ( \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } ) = \left. \psi _ { \mathcal { T } , \alpha ^ { \star } } \middle | H _ { \mathcal { D } } ( \alpha ^ { \star } ) - H _ { \mathcal { T } } ( \alpha ^ { \star } ) \middle | \psi _ { \mathcal { T } , \alpha ^ { \star } } \right. .\tag{31}
$$

This difference is the energy generalization gap. The training algorithm minimizes the empirical energy cost $\widehat { C }$ and obtains Hamiltonian parameters $\alpha ^ { \star }$ and an associated ground state $\left| \psi _ { T , \alpha ^ { \star } } \right.$ . We then show that a small empirical cost together with a small generalization gap implies a small classification error on unseen input points. In particular, we find that the probability of observing a large deviation $\| H _ { \mathcal { D } } ( \alpha ^ { \star } ) - H _ { \mathcal { T } } ( \alpha ^ { \star } ) \| _ { \infty }$ can be bounded with a dependence that scales linearly with the number of qubits n and the number of trainable parameters $M ,$ up to logarithmic factors. The proof relies on a matrix Hoeffding inequality, which controls the energy deviation for a fixed parameter value, together with a covering-number argument, which extends the bound uniformly over the full parameter space. For the purpose of deriving a uniform generalization bound, we restrict the parameter search to a compact domain $\Theta \doteq [ \dot { - } 1 , 1 ] ^ { M }$ . This restriction is specific to the theoretical result below and is not required by the Hamiltonian embedding framework itself.

In particular, we obtain the following theorem.

Theorem 2 (Uniform energy-based generalization bound). Assume a Hamiltonian embedding model such that $0 ~ \preceq$ $H _ { z } ( \alpha ) \preceq I$ and that there exists a constant $L > 0$ such that $\| \dot { H } _ { z } ( \alpha ) - H _ { z } ( \beta ) \| _ { \infty } \leq L \| \alpha - \beta \| _ { \infty } f o r$ all points $z = ( x , y ) \in$ $\Omega \times \{ 0 , 1 \}$ and all $\alpha , \beta \in [ - 1 , 1 ] ^ { M }$ . Then, for every $ { \varepsilon } \in ( 0 , 4 L ]$ , with probability at least $1 - \delta$ over the draw of N-size training datasets,

$$
\operatorname* { s u p } _ { \alpha \in [ - 1 , 1 ] ^ { M } } \| H _ { T } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) \| _ { \infty } \leq \sqrt { \frac { 8 } { N } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 2 4 L } { \varepsilon } \right) \right] } + \frac { \varepsilon } { 4 } .\tag{32}
$$

where $D = 2 ^ { n }$ is the Hilbert space dimension.

The derived energy generalization gap bound can be used to control the population risk of the trained model. In particular, suppose that the training procedure reaches an empirical energy cost $\widehat { C } ( \psi _ { \mathscr { T } , \alpha ^ { \star } } , \alpha ^ { \star } )$ . Then, the number of training samples required to ensure that the population risk is upper bounded by twice the empirical energy cost, up to a constant additive error, scales linearly with the number of qubits and the number of trainable parameters, up to logarithmic factors. This is stated in the following corollary of Theorem 2.

Corollary 3 (Bound on the total risk). Assume the setting of Theorem 2. Let $\left( \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } \right)$ be the output of the training procedure. Let $\widehat { C } ( \psi _ { \tau , \alpha ^ { \star } } , \alpha ^ { \star } ) ~ = ~ \langle \psi _ { \tau , \alpha ^ { \star } } | H _ { \mathcal { T } } ( \alpha ^ { \star } ) | \psi _ { \tau , \alpha ^ { \star } } \rangle$ be the empirical energy reached by the trained model, and let $C ( \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } ) = \langle \psi _ { \mathcal { T } , \alpha ^ { \star } } | H _ { \mathcal { D } } ( \alpha ^ { \star } ) | \psi _ { \mathcal { T } , \alpha ^ { \star } } \rangle$ be its population energy. Moreover, let $R ( \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } ) = \operatorname* { P r } _ { x \sim \mathcal { D } } \left\lceil \tilde { f } _ { \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } } ( x ) \neq f ( x ) \right\rceil$ be the classification risk of the trained model. Then, with probability at least $1 - \delta$ over the draw of the training set $\tau ,$ , the classification risk of the trained model can be bounded by

$$
R ( \psi _ { \tau , \alpha ^ { \star } } , \alpha ^ { \star } ) \leq 2 \widehat { C } ( \psi _ { \tau , \alpha ^ { \star } } , \alpha ^ { \star } ) + \varepsilon ,\tag{33}
$$

provided that the number of training samples satisfies

$$
N \geq \frac { 1 2 8 } { \varepsilon ^ { 2 } } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 2 4 L } { \varepsilon } \right) \right] .\tag{34}
$$

For simplicity, we stated the bound for variational parameters $\alpha \in [ - 1 , 1 ] ^ { M }$ . We note that the same argument applies to any bounded parameter domain $\Theta = [ - \bar { R , } R ] ^ { M }$ , with the corresponding modification of the covering number. Finally, in Appendix A we note that many natural encoding unitaries $U ( x , \alpha )$ , including the one we propose in Section II A, satisfy the Lipschitz assumption on the terms $\check { H } _ { z } ( \alpha )$ required by Theorem $\breve { 2 } .$ Thus, a sample complexity linear in n and ${ \mathrm { ~  ~ \cal ~ M ~ } } ,$ for fixed $\epsilon , \delta ,$ up to logarithmic factors is enough to guarantee a classification error less than ε in the case of perfect training, i.e. when $\alpha ^ { \star }$ is such that $H _ { \mathcal { T } } ( \alpha ^ { \star } )$ has zero ground energy and the true ground state of $H \tau ( \alpha ^ { \star } )$ is found during the training.

We note that, if the ground states of the data Hamiltonian admit a polynomial-size circuit representation, generalization guarantees could also be obtained from existing QML bounds $\mathrm { [ C H C ^ { + } 2 2 ] }$ , whose sample complexity is controlled by the number of parameterized gates in the circuit. However, our bound does not depend on the circuit complexity required to prepare the ground state, but only on the number M of parameters entering the embedding-Hamiltonian family and on the number of qubits n. In particular we notice that, even when the ground state is efficiently preparable, the number of parameterized gates required by such a preparation circuit may be substantially larger than M.

## B. Classical inference and surrogate models

An interesting feature of the Hamiltonian embedding model is that, for specific choices of the encoding unitaries $U ( x , \alpha )$ , inference can be carried out entirely classically, thereby removing the need to access a quantum device at deployment. In these settings, the role of the quantum computer is confined to the training stage, where it is used to prepare a suitable advice state by minimizing the energy of the data-embedding Hamiltonian. Quantum models that allow for classical inference after quantum training were introduced in $[ [ \mathrm { G M } ^ { + } \dot { 2 } 4 ]$ under the name shadow models. In a shadow model, the quantum computer provides a quantum advice state obtained during training. We can then see the Hamiltonian embedding model as a special case of shadow model in which the advice state is obtained as the ground state of the data-dependent Hamiltonian. Once the trained ground state has been obtained, the structure of the encoding unitary $U ( x , \bar { \alpha } ^ { \star } )$ can be exploited to evaluate the prediction rule in Eq. (4) without further access to the quantum device. In this section, we present different classes of encoding unitaries $U ( x , \alpha )$ for which inference can be performed efficiently classically, together with the corresponding construction of classical surrogate models. Finally, in Section III B 4 we also identify settings in which the learned ground state acts as a useful inductive bias, enabling quantum Hamiltonian embedding to solve the task with fewer training examples than a classically regularized surrogate.

## 1. Fourier surrogates

Let us consider the case of a constant depth encoding unitary $U ( x , \alpha )$ . A first example of a classical surrogate for evaluating the trained model is based on the Fourier decomposition of the final circuit appearing in Eq. (4). Indeed, parameterized quantum circuits are known to admit a representation as generalized trigonometric polynomials [SSM21], whose set of frequencies is determined solely by the data-encoding gates. Specifically, let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ be the input points, then any parameterized quantum circuit of the kind

$$
f _ { \alpha } ( x ) = \langle \psi | U ^ { \dagger } ( x , \alpha ) M U ( x , \alpha ) | \psi \rangle\tag{35}
$$

can be equivalently expressed in the form

$$
f _ { \alpha } ( x ) = \sum _ { { \boldsymbol { \omega } } \in \Omega _ { \alpha } } c _ { \boldsymbol { \omega } } ( { \boldsymbol { \alpha } } ) e ^ { i { \boldsymbol { \omega } } \cdot { \boldsymbol { x } } } ,\tag{36}
$$

where $\Omega _ { \alpha }$ denotes the set of accessible frequencies and $c _ { \omega } ( \alpha ) \in \mathbb { C }$ are the corresponding Fourier coefficients. The set of accessible frequencies $\Omega _ { \alpha }$ only depends on the gates $U ( x , \alpha )$ used to encode the inputs x. In our model, where $U ( x , \alpha )$ is shallow and the measured observables are local, the number of distinct frequencies in $\Omega _ { \alpha }$ can be shown to scale polynomially with the input size.

In the case where the frequency spectrum $\Omega _ { \alpha }$ is polynomially large, recent results have shown that a classical representation of the circuit can be learned efficiently. In particular, we can use the results in [SEM23] to learn an approximation for the coefficients $c _ { \omega } ( \alpha )$ in Eq. (36) and construct a classical surrogate of the form:

$$
g _ { \alpha } ( x ) = \sum _ { \omega \in \Omega _ { \alpha } } \tilde { c } _ { \omega } ( \alpha ) e ^ { i \omega \cdot x } ,\tag{37}
$$

with the guarantee that

$$
\operatorname* { P r } \left[ \operatorname* { s u p } _ { x } | f _ { \alpha } ( x ) - g _ { \alpha } ( x ) | \leq \epsilon \right] \geq 1 - \delta ,\tag{38}
$$

using polynomially many queries in $1 / \delta$ and $1 / \epsilon$ of the original quantum circuit.

## 2. Pauli surrogates

For constant depth encoding $U ( x , \alpha )$ , we can define a different classical surrogate model by exploiting the locality of the observables considered at inference time. In particular, rather than learning the Fourier coefficients of the final circuit, we construct a classical representation of the trained model by expanding the dressed observable in a basis of local Pauli operators and estimating their expectation values on the learned ground state using classical shadows. When A in Eq. (9) is a sum of local Pauli operators, these estimates provide a classical description sufficient to evaluate the inference rule entirely through classical-shadow post-processing [HKP20]. In particular, once the optimized state $\left| \psi _ { T , \alpha ^ { \star } } \right.$ is obtained during training, the probability $P _ { \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } } ( x ^ { \prime } , \breve { y } )$ in Eq. (4) for a new input $x ^ { \prime }$ can be estimated from a classical-shadow representation of the learned ground state. Indeed, let us consider the dressed observable constructed by conjugating the original observable in Eq. (9) with the trained encoding unitary for a new input point $x ^ { \prime } { : }$

$$
\tilde { O } ( x ^ { \prime } , y ) = \frac { I + ( - 1 ) ^ { y } \tilde { A } } { 2 } , \qquad \tilde { A } ( x ) = U ^ { \dagger } ( x ^ { \prime } , \alpha ^ { \star } ) A U ( x ^ { \prime } , \alpha ^ { \star } ) .\tag{39}
$$

Since we choose A, and therefore $O ( y )$ , to be local, conjugating it by a constant-depth local encoding unitary $U ( x , \alpha )$ only spreads its support within a constant-size light cone. Therefore, the dressed observable $\widetilde { \cal O } ( x ^ { \prime } , y ) =$ $U ^ { \dagger } ( x ^ { \prime } , \alpha ^ { \star } ) \bar { O } ( y ) \bar { U } ( x ^ { \prime } , \alpha ^ { \star } )$ can then be efficiently decomposed into local Pauli operators. Indeed, if U is a constantdepth circuit composed of local gates, the backward light cone of each measured term contains at most $k = O ( 1 )$ qubits. Each such term can therefore be expanded into at most $4 ^ { k }$ Pauli operators. Since the observable contains $\bar { O } ( n )$ local terms, we can write

$$
\widetilde { A } ( x ^ { \prime } ) = \sum _ { i = 1 } ^ { K } c _ { i } ( x ^ { \prime } , \alpha ^ { \star } ) P _ { i } ,\tag{40}
$$

with $K = O ( n 4 ^ { k } )$ , where the $P _ { i }$ are Pauli observables of weight at most k. Moreover, the coefficients $c _ { i } ( x ^ { \prime } , \alpha ^ { \star } )$ can be computed efficiently classically by simulating the constant-size backward light cones. Once the approximate ground state $\left| \psi _ { T , \alpha ^ { \star } } \right.$ has been prepared quantumly, a classical shadow built from

$$
S = O \left( \frac { 3 ^ { k } \operatorname* { m a x } _ { x } \| c ( x , \alpha ^ { \star } ) \| _ { 1 } ^ { 2 } } { \epsilon ^ { 2 } } \log \frac { K } { \delta } \right)\tag{41}
$$

measurements suffices to estimate $P _ { \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } } ( x ^ { \prime } , y )$ to additive error $\epsilon ,$ uniformly over all $x ^ { \prime } \in \Omega$ , with probability at least 1 δ [HKP20].

## 3. Deep Clifford encodings

A constant depth $U ( x , \alpha )$ is not the only case in which the Hamiltonian embedding model allows for an entirely classical inference after training. As an example, let us consider an encoding unitary of the form $U ( x , \alpha ) =$ $U _ { 1 } ( \mathbf { \bar { x } } , \alpha ) V U _ { 2 } ( \alpha )$ , where $U _ { 2 } ( \alpha )$ is independent of the input x, V is a deep Clifford circuit, and $U _ { 1 } ( x , \alpha )$ is a shallowdepth unitary. Similarly to the Pauli-surrogate case, we can define the dressed observable obtained by conjugating $O ( y )$ with the shallow part of the encoding,

$$
\widetilde { O } _ { 1 } ( x , y ) = U _ { 1 } ^ { \dagger } ( x , \alpha ^ { \star } ) O ( y ) U _ { 1 } ( x , \alpha ^ { \star } ) = \sum _ { i = 1 } ^ { K } c _ { i } ( x , \alpha ^ { \star } ) P _ { i } ,\tag{42}
$$

where $K = { \mathrm { p o l y } } ( n )$ and the $P _ { i }$ are local Pauli observables. Since V is a Clifford circuit, conjugation by V maps every Pauli observable to another Pauli observable,

$$
V ^ { \dagger } P _ { i } V = Q _ { i } .\tag{43}
$$

Although the resulting operators $Q _ { i }$ may have support on a number of qubits growing with the system size, the number of terms in the decomposition remains polynomial. The prediction probability rule of Eq. (4) can therefore be written as

$$
P _ { \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } } ( x , y ) = \sum _ { i = 1 } ^ { K } c _ { i } ( x , \alpha ^ { \star } ) m _ { i } , \qquad m _ { i } = \langle \psi _ { \mathcal { T } , \alpha ^ { \star } } | U _ { 2 } ^ { \dagger } ( \alpha ^ { \star } ) Q _ { i } U _ { 2 } ( \alpha ^ { \star } ) | \psi _ { \mathcal { T } , \alpha ^ { \star } } \rangle .\tag{44}
$$

Crucially, since $U _ { 2 }$ is independent of $x ,$ the coefficients $m _ { i }$ need to be estimated only once after training. Since the operators $Q _ { i }$ may have large Pauli weight, product-Pauli classical shadows may not be efficient. However, as only polynomially many such observables appear, their expectation values can be measured directly on the quantum device and stored classically. Predictions for new inputs can then be obtained entirely classically by computing the coefficients $c _ { i } ( x , \alpha ^ { \star } )$ and evaluating the above sum.

## 4. Dequantizability of the model

The existence of efficient classical surrogates for the trained quantum model raises the natural question of whether the full learning procedure can itself be reproduced classically, potentially eliminating the need for quantum computation altogether. For example, let us consider the Pauli surrogate. Suppose that the dressed observable O<sup>˜</sup> admits a decomposition into polynomially many Pauli operators, as in Eq. (40). The prediction function of the trained quantum model can then be written as

$$
\tilde { f } _ { \alpha ^ { \star } } ( x ) = \sum _ { i } ^ { K } c _ { i } ( x , \alpha ^ { \star } ) m _ { i } ( \alpha ^ { \star } ) ,\tag{45}
$$

where $m _ { i } ( \alpha ^ { \star } ) = \langle \psi _ { T , \alpha ^ { \star } } | P _ { i } | \psi _ { T , \alpha ^ { \star } } \rangle$ are the expectation values of the Pauli operators $P _ { i }$ on the learned ground state, while $c _ { i } ( x , \alpha ^ { \star } )$ are the coefficients appearing in the Pauli decomposition of the dressed observable as in Eq (40).

We can then define a fully classical surrogate of the form

$$
g _ { \alpha } ( x ) = \sum _ { i } ^ { K } c _ { i } ( x , \alpha ) \tilde { m } _ { i } ( \alpha ) ,\tag{46}
$$

where the coefficients $\tilde { m } _ { i } ( \alpha ) \in \mathbb { R }$ are treated as trainable classical parameters and are not required to correspond to expectation values of Pauli operators on any quantum state.

In the quantum model of Eq. (45), the moments $m _ { i } ( \alpha ^ { \star } )$ are not independent parameters, but must be jointly compatible with a single physical quantum state. These compatibility constraints can in principle act as a useful inductive bias, although they do not by themselves imply either easier optimization or improved generalization relative to classical alternatives. To investigate this question, in Appendix $\dot { \boldsymbol { \mathrm { G } } }$ we construct a learning problem for which the benefit of these physical compatibility constraints can be established analytically, yielding a provable separation in the number of training examples required by Hamiltonian embedding and a corresponding classically regularized surrogate. We complement this result in Appendix K with a finite-size numerical example exhibiting the same qualitative behavior. Importantly, both tasks are classically tractable, and the purpose of these examples is to isolate the generalization advantage of the physical compatibility constraints imposed by the learned ground state rather than to demonstrate a computational quantum advantage.

A more constrained surrogate can be obtained when each dressed readout term $\tilde { A } _ { j } = U ( x , \alpha ) ^ { \dagger } P _ { j } U ( x , \alpha )$ in Eq. (40) is supported on a fixed region $L _ { j }$ of the n qubits, independent of x and $\alpha ,$ with $| L _ { j } | = \dot { O } ( \log n )$ . Writing the observable A in Eq. (9) as $\begin{array} { r } { A = \sum _ { j } ^ { K } a _ { j } P _ { j } } \end{array}$ , we define a classical surrogate as

$$
g _ { \sigma , \alpha } ( x ) = \sum _ { j } ^ { K } a _ { j } \mathrm { T r } \left[ \sigma _ { j } \tilde { A } _ { j } \right] ,\tag{47}
$$

where each $\sigma _ { j }$ is a density matrix on the fixed subsystem $L _ { j } ,$ with no global compatibility constraints, and $\tilde { A } _ { j } ( x , \alpha )$ has support on $L _ { j } .$ . At fixed $\alpha ,$ from the energy-loss definition in Eq. (6), the empirical energy becomes $\widehat { C } _ { \mathcal { T } } ( \sigma , \alpha ) =$ $\begin{array} { r } { \frac { 1 } { 2 } - \frac { 1 } { 2 } \sum _ { j = 1 } ^ { K } a _ { j } \operatorname { T r } [ \overset { , } { \sigma _ { j } } B _ { j } ( \alpha ) ] \quad } \end{array}$ , where $\begin{array} { r } { B _ { j } \big ( \alpha \big ) = N ^ { - 1 } \sum _ { \ell = 1 } ^ { N } s _ { \ell } \tilde { A } _ { j } \big ( x _ { \ell } , \alpha \big ) } \end{array}$ and $s _ { \ell } = ( - 1 ) ^ { y _ { \ell } }$ . Since the matrices $\sigma _ { j }$ can be optimized independently, energy minimization reduces to K eigenproblems by choosing each $\sigma _ { j }$ as a projector onto a top eigenvector of $a _ { j } B _ { j } ( \alpha )$ . These matrices have dimension at most $2 ^ { | L _ { j } | } = \mathrm { p o l y } ( n )$ , so the inner optimization runs in polynomial-time when the dressed terms can be computed efficiently and achieves an energy lower or equal to the ground-state energy of the original embedding Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ . Under $\begin{array} { r } { \sum _ { j = 1 } ^ { K } | a _ { j } | \leq 1 } \end{array}$ and uniform operatornorm Lipschitz bounds on the individual dressed terms $\tilde { A } ( x , \alpha )$ , the same proof strategy of Theorem 2 extends to this relaxation guaranteeing also the same generalization bounds. However, we note that this surrogate construction may not remain efficient when the encoding unitary $U ( x , \alpha )$ is such that the supports of each dressed term $\tilde { A } _ { j } ( x , \alpha )$ depend on x. In this case, the support of their union over all training points may involve all n qubits, making the dimension $2 ^ { | L _ { j } | }$ | of $\sigma _ { j }$ scale exponentially. For instance, the construction in Appendix G also exhibits such inputdependent supports, providing an example in which the efficiency guarantee for this surrogate no longer applies. Finally, we note that the polynomial-time training statements of the surrogates do not cover the outer optimization over α, which is not convex and could benefit from the inductive bias induced by the ground state of the embedding Hamiltonian.

In particular, a separate question is whether the full Hamiltonian embedding learning procedure can itself be efficiently reproduced classically. The existence of efficient classical surrogates for inference does not, in general, imply that the corresponding model can also be learned efficiently classically. In particular, one natural route beyond the shallow-surrogate regime is to use encodings whose dressed observables have growing light cones and do not admit a polynomial-sized, efficiently evaluable feature representation. Increasing the depth of U can therefore invalidate the particular Pauli- and Fourier-surrogate arguments used above. Depth alone, however, is not sufficient, since structured deep circuits may remain classically tractable and low-bond-dimension learned states may still permit efficient tensor-network evaluation. The relevant question is whether the task-dependent moments defining the classifier can be learned or computed classically, rather than circuit depth or entanglement alone. The construction in the following section provides an explicit setting in which the trained hypothesis is efficiently evaluable but learning it classically is nevertheless conjectured to be hard under cryptographic assumptions.

## C. Encoding a classically hard problem in the Hamiltonian embedding model

When introducing a new quantum machine learning method, it is natural to ask whether it can be applied to problems that lie beyond the reach of classical algorithms. In the Hamiltonian embedding framework, the training procedure requires finding the ground state of an encoding Hamiltonian. Since ground-state preparation is, in general, hard even for quantum computers, it is not immediately clear whether this approach can provide a rigorous quantum speedup. In this section we present a learning task that is believed to be classically intractable, yet can be expressed within our Hamiltonian-embedding framework and solved using a quantum computer. The learning problem is the binary classification task described in $[ \mathrm { J } \mathrm { G M } ^ { + } 2 4 ] .$ , where input $\bar { x ^ { \prime } } \in \dot { \mathbb { Z } } _ { N } = \{ 0 , 1 , . . . , \hat { N } - 1 \}$ are labeled in two classes $y = \{ \bar { 0 } , 1 \}$ by the following labeling function:

$$
y ( x ) = f _ { N , s } ( x ) = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } { \sqrt [ { 3 } ] { x } } \mod N \in \left[ s , s + { \frac { N - 1 } { 2 } } \right] , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{48}
$$

where $N = p q$ is a large n-bit integer product of the two primes p and q and the interval in Eq. (48) is interpreted cyclically in $\mathbb { Z } _ { N }$ . In particular, the labeling function partitions inputs into two classes according to whether their discrete cube root (DCR) modulo N lies in a specified half-interval, determined by s. Under the DCR assumption, computing discrete cube roots modulo large semiprimes N is believed to be classically intractable [KV94], and the associated learning problem is also believed to be classically hard [JGM<sup>+</sup>24]. Notably, the DCR function is a trapdoor function: there exists a key $d \in \mathbb { Z } _ { N }$ that allows one to evaluate the function efficiently. More precisely, for the DCR function, there exists a d such that $\sqrt [ 3 ] { x }$ mod $N = x ^ { d }$ mod N. Since modular exponentiation can be computed efficiently classically, evaluating the DCR function on different inputs x becomes easy once d is known. However, determining the key d requires factoring the integer $N ,$ which is believed to be computationally hard for classical algorithms for large integers N.

The Hamiltonian embedding framework can reproduce the DCR labeling function in Eq. (48) as its hypothesis function. More precisely, one can construct an encoding Hamiltonian whose ground state induces, through the prediction rule in Eq. (E25), a hypothesis $\widetilde { f }$ that well approximates the DCR labeling function. In particular, we consider the following encoding unitary $U ( x )$ defined over the Hilbert space $\mathcal { H } = \mathrm { s p a n } \{ | d , s \rangle : d , s \in \bar { \mathbb { Z } } _ { N } \} \otimes \mathbb { C } ^ { 2 }$ and the observable $O ( y )$

$$
U ( x ) | d , s  | b  = | d , s  | b \oplus g _ { d , s } ( x )  \quad \& \quad O ( y ) = I \otimes | y  \langle y | , \qquad x , d , s \in \mathbb { Z } _ { N } , y \in \{ 0 , 1 \} .\tag{49}
$$

where $g _ { d , s }$ is defined as:

$$
g _ { d , s } ( x ) = \left\{ 1 , \begin{array} { l } { { \mathrm { i f } x ^ { d } \mod N \in \left[ s , s + \frac { N - 1 } { 2 } \right] , } } \\ { { 0 , \mathrm { o t h e r w i s e } } } \end{array} \right.\tag{50}
$$

and it is classically computable as modular exponentiation can be computed efficiently. We then construct the embedding Hamiltonian $\hat { H _ { \mathrm { D C R } } }$ as

$$
H _ { \mathrm { D C R } } = \sum _ { \ell } H ( x _ { \ell } , y _ { \ell } ) .\tag{51}
$$

We note that the encoding Hamiltonian $H _ { \mathrm { D C R } }$ admits an efficient classical description, since the encoding unitary U in Eq. (49) can be implemented by a polynomial-size classical reversible circuit, so that each Hamiltonian term associated with a training point $( x _ { \ell } , \dot { y } _ { \ell } ) \dot { \in } \dot { T }$ can be efficiently specified as:

$$
H ( x _ { \ell } , y _ { \ell } ) = I - U ( x _ { \ell } ) ^ { \dagger } O ( y _ { \ell } ) U ( x _ { \ell } ) + I \otimes | 1 \rangle \langle 1 | .\tag{52}
$$

The ground space of $H _ { \mathrm { D C R } }$ contains states $| d , s , 0 \rangle$ associated with hypotheses $g _ { d , s }$ that are consistent with the training data. As shown in Appendix $\mathrm { E } ,$ a quantum algorithm can efficiently find a ground state $| d ^ { \star } , s ^ { \prime } , 0 \rangle$ whose corresponding hypothesis $g _ { d ^ { \star } , s ^ { \prime } }$ has small population error on the DCR learning task. In particular, following the construction in Section II, for a new point x the predicted label is the value of y that maximizes $\mathrm { T r } \big [ O ( \breve { y } ) U ( x ) | d ^ { \star } , s ^ { \prime } , 0 \rangle \langle d ^ { \star } , s ^ { \prime } , 0 | U ( x ) ^ { \dagger } \big ]$ . Therefore, we can prove the following theorem:

Theorem 4 (Encoding of a hard problem (informal)). There exists a classification problemfor which the Hamiltonian em beddingframework yields a classically efficiently describable data-embedding Hamiltonian whose ground states induce, through the prediction rule in Eq. (5), hypotheses with arbitrarily small classification error. Moreover, a quantum computer can $e { \dot { f } } { \dot { f } } -$ ciently find a corresponding ground state and construct the associated hypothesis. In contrast, assuming the classical hardness of computing discrete cube roots, no efficient classical learner can achieve the same guarantee with non-negligible probability.

The full specification of the result, together with the formal proof, can be found in Appendix E. Finally, we note a distinction between Hamiltonian embedding and some earlier examples of quantum models that efficiently solve learning tasks believed to be classically intractable [LAT21, AG26, MGD26, ZZN<sup>+</sup>26, GGPD25], concerning where the hard computation takes place. In these examples, the quantum models are explicitly constructed to incorporate the correct hard computation required to solve the specific task into their circuit architectures. In the discrete-logarithm-based constructions of [LAT21, GGPD25], for example, the encoding circuits incorporate discretelogarithm computations, while the remaining training optimization then becomes easy. In our construction, the hard computation instead takes place during data-dependent quantum optimization through ground-state preparation, while the encoding circuits themselves perform efficiently computable classical operations. Here ground-state preparation is treated as a subroutine. The encoding circuits therefore need not be designed a priori to perform a specific hard computation and, in fact, may also be classically tractable, as is the case in the classically hard learning task presented here.

## IV. EXPERIMENTS

We next test whether the optimization in Algorithm 1 remains feasible at nontrivial system sizes and whether the resulting low-energy states achieve low classification error on a standard benchmark dataset. In particular, we evaluate the Hamiltonian embedding algorithm on the binary classification task of Fashion-MNIST dress and shirt images, with inputs reduced by PCA to n features for an n-qubit model. We first train the model using a classical iterative solver as a proxy [KPM<sup>+</sup>26] to approximate the ground state at each iteration. This benchmark uses 100 random seeds for each of the 25 even system sizes $n = 2 , \breve { 4 } , \dotsc , 5 0$ , with each seed determining both the data split and the initial model parameters. We compare test accuracy against classical machine-learning baselines and the fidelity and projected quantum kernels in [HBM<sup>+</sup>21].

We then implement the full hybrid training procedure, using sample-based Krylov quantum diagonalization on IBM Heron processors for ground-state approximation. The selection criteria for the hardware runs are given below and in Appendix J.

## A. The model instance used in the experiments

All Hamiltonian embedding experiments use the same model architecture and training objective introduced in Section II, which we summarize here and discuss in detail in Appendix H. The encoding unitary is the constantdepth circuit in Eq. (12) with $K = 3$ layers. We choose $K = 3$ to explore nontrivial Hamiltonians while keeping the number of Pauli terms computationally tractable. Each layer applies three single-qubit rotations, generated by $\bar { Z } , Y ,$ and $Z ,$ to every qubit, followed by a ring of nearest-neighbor two-qubit $Z Z$ rotations. The circuit therefore contains 12n parameterized gates, whose trainable coefficients are $\alpha \in \mathbb { R } ^ { 1 2 n \times ( d + 1 ) }$ . For an input $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , the angle of gate $g$ is $\begin{array} { r } { \theta _ { g } ( x , \alpha ) = \sum _ { r = 1 } ^ { d } \alpha _ { g , r } x _ { r } + \alpha _ { g , 0 } } \end{array}$ . This is equivalent to the encoding in Eq. (12) with a constant feature appended to the input. We use the one-local readout $\begin{array} { r } { A = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ from Section II A. The outer optimization minimizes the energy loss defined in Eq. (7), which is sufficient to achieve good classification performance in our experiments.

Two features make the experiments feasible at these system sizes. First, the light-cone argument in Section II A shows that each data point contributes a linear combination of the same ${ \mathcal { O } } ( n )$ bounded-weight Pauli observables to the embedding Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ . We obtain these strings by propagating the readout observable through the circuit in the Heisenberg picture and combining identical terms. Second, as discussed in Section II B, the state is held fixed during each parameter update, so the gradient is given by the expectation value of $\partial _ { \alpha } H _ { \mathcal { T } } ( \alpha )$ in the current state.

![](images/595946323f83b75655be32b185363d877453b5a471e86ee0190870d7131876b9.jpg)

![](images/7c5edcbda7e3ae05532c251dc4271ee318ef726c0363beb876ad535b8957f03d.jpg)  
Figure 1. a) Test accuracy versus system size for various methods on a binary classification problem from the Fashion-MNIST dataset. The methods are: best-performing classical baseline (gray), classical iterative solver (orange) used in HE framework, quantum kernel based on a product-state embedding (blue), 1-reduced density matrix (1-RDM) projected quantum kernel with the same embedding [ $\mathrm { H B M } ^ { + } 2 1 ]$ (green), and the 1-RDM projected quantum kernel based on the E2 embedding of Ref. [HBM<sup>+</sup>21] (pink). b) Held-out test accuracy of the five runs with the highest validation accuracy at $n = 5 0 .$ . In this plot, we additionally show the accuracy obtained by the HE model trained on hardware (yellow).

We optimize the parameters using Adam [KB15] together with an $\ell _ { 2 }$ penalty. At each training epoch, we evaluate the model on the validation set. After training, we keep the variational parameters and associated learned quantum state from the epoch with the highest validation accuracy, and evaluate the resulting model once on the held-out test set.

We first study Hamiltonian embedding framework using a classical method for approximating ground states that we call “classical iterative solver” [KPM<sup>+</sup>26]. In these simulations, we approximate the ground state at each training iteration using an iterative solver based on repeated applications of $H _ { \mathcal { T } } ( \bar { \alpha } )$ , deterministically retaining the bitstrings with the highest probabilities, akin to a power iteration with truncation. We reserve the term $^ { \prime \prime } \mathrm { H E } ^ { \prime \prime }$ for the hardware runs using the real-time SKQD protocol.

For the hardware runs, we approximate the ground state at each training iteration using SKQD [YMI<sup>+</sup>25]. Starting from a reference state, we generate a sequence of Krylov states by Trotterized real-time evolution under the current training Hamiltonian and measure them in the computational basis. The sampled bitstrings span a reduced subspace onto which we project $H _ { \mathcal { T } } ( \alpha )$ and diagonalize it classically.

The classical iterative solver and SKQD may select different basis states and therefore produce different reduced subspaces. Both methods nevertheless project the full embedding Hamiltonian onto the selected subspace, so the lowest projected energy remains a variational upper bound on the true ground-state energy. The classical solver provides a useful reference for the optimization and classification performance of Hamiltonian embedding, although its performance need not predict that of SKQD.

## B. Results

We compare Hamiltonian embedding with the fidelity and projected quantum kernels studied in Ref. $[ \mathrm { H B M } ^ { + } 2 1 ]$ as well as with classical machine-learning methods. We use the same image-classification benchmark studied in Ref. [HBM<sup>+</sup>21], which is a binary classification of Fashion-MNIST images [XRV17] into the classes dress and shirt. We first use principal component analysis (PCA) to reduce the input dimension to the number of qubits n. For each run, a pool of 800 samples is divided into 448 training, 112 validation, and 240 held-out test examples. We first split the data $7 0 / 3 0$ into training and test sets, and then split the training set 80/20 into the final training and validation sets. At each $n ,$ all methods use the same random seeds and therefore the same data splits.

Figure 1(a) reports the mean test accuracy across the 100 runs at each system size, with error bars showing one standard deviation. We note that the best classical ML baseline, which corresponds to the support vector classifier with a radial basis function kernel, performs the best, reaching an accuracy of approximately 0.946. The classical iterative solver, which may be viewed as a proxy for the Hamiltonian embedding framework, achieves a higher mean test accuracy than the best reported quantum kernel baseline, with margins ranging from 0.018 to 0.067. The mean accuracy increases from 0.880 at $n = 2$ to approximately 0.924 at intermediate system sizes before gradually decreasing to 0.886 at $n = 5 0$ . The fidelity kernel based on the product-state embedding E1 from [HBM<sup>+</sup>21] decreases from 0.798 to 0.515, whereas the projected 1-RDM kernel based on the same embedding decreases more gradually from a maximum of 0.887 to 0.831.

We also study a projected quantum kernel constructed from the 1-RDMs of feature states generated using a simplified version of the E2 embedding from Refs. $[ \mathrm { H C T ^ { + } } 1 9 , \mathrm { H B M ^ { + } } 2 1 ]$ . These feature states are prepared using circuits inspired by the instantaneous quantum polynomial-time (IQP) model, which we call IQP-like circuits. To make their classical simulation tractable, we restrict the interactions to bands of neighboring qubits of width $w = 4 ,$ , allowing the circuits and resulting kernels to be evaluated using matrix product states. The test accuracy of this kernel decreases rapidly with n and levels off near 0.65, above that of the fidelity kernel at large n.

We next turn to the hardware results. In these runs, we prepare the SKQD Krylov states described above on IBM Heron processors and measure them in the computational basis. We then project $H _ { \mathcal { T } } ( \alpha _ { t } )$ in the sampled bitstring subspace and diagonalize it classically. We evaluate the gradients and perform the Adam updates classically, following Algorithm 1.

We select five hardware runs based on the results obtained using the classical iterative solver to select data splits and initial model parameters α that lead to high validation accuracy. As shown in Fig. 1(b), Hamiltonian embedding is competitive with the best classical baseline and outperforms the other quantum methods considered. All $n = 5 \bar { 0 }$ runs were performed on ibm pittsburgh, a 156-qubit IBM Heron r2 processor. The 50 data qubits were mapped to a linear chain chosen greedily to minimize two-qubit gate errors. Moreover, we apply dynamical decoupling and measurement twirling but no additional error mitigation.

These experiments demonstrate that Hamiltonian embedding models can be studied at scale on current quantum processors.

## V. DISCUSSION

In this work, we introduced Hamiltonian embedding, a QML framework for classification in which the entire training set is encoded in a single parameterized Hamiltonian. Training minimizes the ground-state energy or a more general loss function over the encoding parameters, with the loss function estimated by a quantum algorithm at each iteration. The ground state of the optimized Hamiltonian then defines a classifier for unseen inputs. For shallow encoding circuits, measurements of the prepared states yield classical representations that allow the loss and its gradients to be evaluated efficiently. Moreover, the representation of the final state supports inference without further quantum computation. Therefore, the role of quantum processors is restricted to approximating ground states during training. We further showed, analytically and numerically, that the physical compatibility constraints imposed by the learned state can provide a useful inductive bias for certain tasks. Finally, under standard cryptographic assumptions, Hamiltonian embedding can encode certain learning tasks that are intractable for efficient classical algorithms.

For generic learning tasks, however, it remains unclear when the inductive bias of Hamiltonian embedding offers an advantage over alternative classical or quantum methods. This question is particularly difficult for classical datasets, where symmetry-based arguments may not apply directly [GGC<sup>+</sup>24, RBN<sup>+</sup>22], and will likely require heuristic model design together with systematic, task-specific benchmarking. Practical use also requires the model to remain trainable and to generalize as the system size grows. For fixed generalization error, our analytical bounds show that the required number of training samples scales linearly with the number of qubits and trainable param eters, up to logarithmic factors. Our numerical results provide complementary evidence for trainability. On binary Fashion-MNIST, Hamiltonian embedding can be trained at scale, including implementations on IBM quantum processors, while maintaining competitive test performance as the system size increases. In the same benchmarks, the generalization performance of the quantum-kernel method of Ref. [HBM<sup>+</sup>21] deteriorates with system size, whereas Hamiltonian embedding also achieves higher accuracy than the projected quantum kernels designed to mitigate this behavior.

Hamiltonian embedding therefore offers a promising alternative to existing QML models. Several aspects of the framework remain unexplored. In particular, the freedom to choose both the encoding and the readout observable may enable constructions with improved trainability or expressivity. A central open question is which learning problems are naturally suited to Hamiltonian embedding and whether it can offer advantages over classical methods beyond cryptographically motivated constructions.

## ACKNOWLEDGMENTS

We acknowledge discussions with Aram Harrow, Will Kirby, Sarah Sheldon, Kevin Smith, and Minh Tran. R.M. and V.D. acknowledge the support of the Dutch Research Council (NWO) through the NWO/NWA project Divide and Quantum (Project No. 1389.20.241). V.D. also acknowledges the support of the European Union’s Horizon Europe program through the ERC CoG BeMAIQuantum (Grant No. 101124342). This work was also supported by the Dutch National Growth Fund (NGF) as part of the QDNL program.

LLM models were used to assist with numerical simulations and parts of the analytical work, including selected steps in the proofs. The authors critically reviewed and verified all incorporated suggestions and take full responsibility for the accuracy and integrity of this work.

[ACGS88] Werner Alexi, Benny Chor, Oded Goldreich, and Claus P. Schnorr. RSA and Rabin Functions: Certain Parts are as Hard as the Whole. SIAM Journal on Computing, 17(2):194–209, 1988. doi:10.1137/0217013.

[AG26] Eric R. Anschuetz and Xun Gao. Arbitrary Polynomial Separations in Trainable Quantum Machine Learning. Quantum, 10:1976, 2026. doi:10.22331/q-2026-01-20-1976

[AL99] Daniel S. Abrams and Seth Lloyd. Quantum Algorithm Providing Exponential Speed Increase for Finding Eigenvalues and Eigenvectors. Physical Review Letters, 83(24):5162–5165, 1999. doi:10.1103/PhysRevLett.83.5162.

[BLSF19] Marcello Benedetti, Erika Lloyd, Stefan Sack, and Mattia Fiorentini. Parameterized quantum circuits as machine learning models. Quantum Science and Technology, 4(4):043001, 2019. doi:10.1088/2058-9565/ab4eb5.

[BWP<sup>+</sup>17] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd. Quantum machine learning. Nature, 549(7671):195–202, 2017. doi:10.1038/nature23474.

[CAB<sup>+</sup>21] M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, and Patrick J. Coles. Variational quantum algorithms. Nature Reviews Physics, 3(9):625–644, 2021. doi:10.1038/s42254-021-00348-9

[CC25] Su Yeon Chang and M. Cerezo. A Primer on Quantum Machine Learning. (arXiv:2511.15969), 2025. arXiv:2511.15969.

[CEMM98] R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca. Quantum algorithms revisited. Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences, 454(1969):339–354, 1998. doi:10.1098/rspa.1998.0164.

[CHC<sup>+</sup>22] Matthias C. Caro, Hsin-Yuan Huang, M. Cerezo, Kunal Sharma, Andrew Sornborger, Lukasz Cincio, and Patrick J. Coles. Generalization in quantum machine learning from few training data. Nature Communications, 13(1):4919, 2022. doi: 10.1038/s41467-022-32550-3.

[CLG<sup>+</sup>25] M. Cerezo, Martin Larocca, Diego Garc´ıa-Mart´ın, N. L. Diaz, Paolo Braccia, Enrico Fontana, Manuel S. Rudolph, Pablo Bermejo, Aroosa Ijaz, Supanut Thanasilp, Eric R. Anschuetz, and Zoe Holmes. Does provable absence of barren plateaus¨ imply classical simulability? Nature Communications, 16(1):7907, 2025. doi:10.1038/s41467-025-63099-6.

[CSV<sup>+</sup>21] M. Cerezo, Akira Sone, Tyler Volkoff, Lukasz Cincio, and Patrick J. Coles. Cost Function Dependent Barren Plateaus in Shallow Parametrized Quantum Circuits. Nature Communications, 12(1):1791, 2021. doi:10.1038/s41467-021-21728-w.

[GGC<sup>+</sup>24] Jennifer R. Glick, Tanvi P. Gujarati, Antonio D. Corcoles, Youngseok Kim, Abhinav Kandala, Jay M. Gambetta, and´ Kristan Temme. Covariant quantum kernels for data with group structure. Nature Physics, 20(3):479–483, 2024. doi:10. 1038/s41567-023-02340-9.

[GGPD25] Elies Gil-Fuster, Casper Gyurik, Adrian Perez-Salinas, and Vedran Dunjko. On the relation between trainability and´ dequantization of variational quantum learning models. In The Thirteenth International Conference on Learning Representations, 2025. URL: https://openreview.net/forum?id=TdqaZbQvdi.

[HBK<sup>+</sup>25] Laura J. Henderson, Kerstin Beer, Salini Karuvade, Riddhi Gupta, Angela White, and Sally Shrapnel. Quantum advantage without exponential concentration: Trainable kernels for symmetry-structured data. 2025. arXiv:2509.14337.

[HBM<sup>+</sup>21] Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, and Jarrod R. McClean. Power of data in quantum machine learning. Nature Communications, 12(1):2631, 2021. doi:10.1038/ s41467-021-22539-9.

[HCT<sup>+</sup>19] Vojtech Havlicek, Antonio D. Corcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and´ Jay M. Gambetta. Supervised learning with quantum enhanced feature spaces. Nature, 567(7747):209–212, 2019. doi: 10.1038/s41586-019-0980-2.

[HKP20] Hsin-Yuan Huang, Richard Kueng, and John Preskill. Predicting many properties of a quantum system from very few measurements. Nature Physics, 16(10):1050–1057, 2020. doi:10.1038/s41567-020-0932-7.

[JGM<sup>+</sup>24] Sofiene Jerbi, Casper Gyurik, Simon C. Marshall, Riccardo Molteni, and Vedran Dunjko. Shadows of quantum machine learning. Nature Communications, 15(1):5676, 2024. doi:10.1038/s41467-024-49877-8.

[KB15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[KPM<sup>+</sup>26] William Kirby, Bibek Pokharel, Javier Robledo Moreno, Kevin C. Smith, Sergey Bravyi, Abhinav Deshpande, Constantinos Evangelinos, Bryce Fuller, James R. Garrison, Ben Jaderberg, Caleb Johnson, Petar Jurcevic, Su-un Lee, Simon Martiel, Mario Motta, Seetharami Seelam, Oles Shtanko, Kevin J. Sung, Minh Tran, Vinay Tripathi, Kazuhiro Seki, Kazuya Shinjo, Han Xu, Lukas Broers, Tomonori Shirakawa, Seiji Yunoki, Kunal Sharma, and Antonio Mezzacapo. Observation

of Improved Accuracy over Classical Sparse Ground-State Solvers using a Quantum Computer. (arXiv:2603.03496), 2026. arXiv:2603.03496.

[KV94] Michael J. Kearns and Umesh Vazirani. An Introduction to Computational Learning Theory. The MIT Press, 1994. doi: 10.7551/mitpress/3897.001.0001.

[LAT21] Yunchao Liu, Srinivasan Arunachalam, and Kristan Temme. A rigorous and robust quantum speed-up in supervised machine learning. Nature Physics, 17(9):1013–1017, 2021. doi:10.1038/s41567-021-01287-z.

[LTW<sup>+</sup>25] Mart´ın Larocca, Supanut Thanasilp, Samson Wang, Kunal Sharma, Jacob Biamonte, Patrick J. Coles, Lukasz Cincio, Jarrod R. McClean, Zoe Holmes, and M. Cerezo. Barren plateaus in variational quantum computing. ¨ Nature Reviews Physics, 7(4):174–189, 2025. doi:10.1038/s42254-025-00813-9.

[MBS<sup>+</sup>18] Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven. Barren plateaus in quantum neural network training landscapes. Nature Communications, 9(1):4812, 2018. doi:10.1038/s41467-018-07090-4.

[MGD26] Riccardo Molteni, Casper Gyurik, and Vedran Dunjko. Exponential quantum advantages in learning quantum observables from classical data. npj Quantum Information, 12(1):19, 2026. doi:10.1038/s41534-025-01162-2.

[MN25] Mathias Mikkelsen and Yuya O. Nakagawa. Quantum-selected configuration interaction with time-evolved state. Physical Review Research, 7(4):043043, 2025. doi:10.1103/75pv-hbrx.

[PCGL20] Adrian P´ erez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and Jos´ e I. Latorre. Data re-uploading for a universal´ quantum classifier. Quantum, 4:226, 2020. doi:10.22331/q-2020-02-06-226.

[QWZ<sup>+</sup>23] Han Qi, Lei Wang, Hongsheng Zhu, Abdullah Gani, and Changqing Gong. The barren plateaus of quantum neural networks: Review, taxonomy and trends. Quantum Information Processing, 22(12):435, 2023. doi:10.1007/ s11128-023-04188-7.

[RBN<sup>+</sup>22] Michael Ragone, Paolo Braccia, Quynh T. Nguyen, Louis Schatzki, Patrick J. Coles, Frederic Sauvage, Martin Larocca, and M. Cerezo. Representation Theory for Geometric Quantum Machine Learning. 2022. arXiv:2210.07980.

[RML14] Patrick Rebentrost, Masoud Mohseni, and Seth Lloyd. Quantum Support Vector Machine for Big Data Classification. Physical Review Letters, 113(13):130503, 2014. doi:10.1103/PhysRevLett.113.130503.

[SBSW20] Maria Schuld, Alex Bocharov, Krysta M. Svore, and Nathan Wiebe. Circuit-centric quantum classifiers. Physical Review A, 101(3):032308, 2020. doi:10.1103/PhysRevA.101.032308.

[Sch05] U. Schollwock. The density-matrix renormalization group. ¨ Reviews of Modern Physics, 77(1):259–315, 2005. doi:10.1103/ RevModPhys.77.259.

[SEM23] Franz J. Schreiber, Jens Eisert, and Johannes Jakob Meyer. Classical Surrogates for Quantum Learning Models. Physical Review Letters, 131(10):100803, 2023. doi:10.1103/PhysRevLett.131.100803.

[Sho97] Peter W. Shor. Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer. SIAM Journal on Computing, 26(5):1484–1509, 1997. doi:10.1137/S0097539795293172.

[SKI<sup>+</sup>25] Kenji Sugisaki, Shu Kanno, Toshinari Itoko, Rei Sakuma, and Naoki Yamamoto. Hamiltonian simulation-based quantum-selected configuration interaction for large-scale electronic structure calculations with a quantum computer. Physical Chemistry Chemical Physics, 27(38):20869–20884, 2025. doi:10.1039/d5cp02202a.

[SP21] Maria Schuld and Francesco Petruccione. Quantum Models as Kernel Methods, pages 217–245. Springer International Publishing, Cham, 2021. doi:10.1007/978-3-030-83098-4\_6.

[SSM21] Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. The effect of data encoding on the expressive power of varia tional quantum machine learning models. Physical Review A, 103(3):032430, 2021. doi:10.1103/PhysRevA.103.032430.

[Tro12] Joel A. Tropp. User-Friendly Tail Bounds for Sums of Random Matrices. Foundations of Computational Mathematics, 12(4):389–434, 2012. doi:10.1007/s10208-011-9099-z.

[Ver26] Roman Vershynin. High-Dimensional Probability. Cambridge University Press, New York, 1 edition, 2026.

[Whi92] Steven R. White. Density matrix formulation for quantum renormalization groups. Physical Review Letters, 69(19):2863– 2866, 1992. doi:10.1103/PhysRevLett.69.2863.

[XRV17] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms. 2017. arXiv:1708.07747.

[YGH<sup>+</sup>25] Qi Ye, Shuangyue Geng, Zizhao Han, Weikang Li, L.-M. Duan, and Dong-Ling Deng. Quantum automated learning with provable and explainable trainability. (arXiv:2502.05264), 2025. arXiv:2502.05264.

[YMI<sup>+</sup>25] Jeffery Yu, Javier Robledo Moreno, Joseph T. Iosue, Luke Bertels, Daniel Claudino, Bryce Fuller, Peter Groszkowski, Travis S. Humble, Petar Jurcevic, William Kirby, Thomas A. Maier, Mario Motta, Bibek Pokharel, Alireza Seif, Amir Shehata, Kevin J. Sung, Minh C. Tran, Vinay Tripathi, Antonio Mezzacapo, and Kunal Sharma. Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization. (arXiv:2501.09702), 2025. arXiv:2501.09702.

[ZZN<sup>+</sup>26] Haimeng Zhao, Alexander Zlokapa, Hartmut Neven, Ryan Babbush, John Preskill, Jarrod R. McClean, and Hsin-Yuan Huang. Exponential quantum advantage in processing massive classical data. (arXiv:2604.07639), 2026. arXiv:2604.07639.

## Appendix A: Energy-based generalization bounds

The bound on the energy generalization gap stated in Theorem 2 follows from two main technical ingredients. First, for any fixed value of α, we bound the probability that the deviation $\| H _ { \mathcal { D } } ( \alpha ) - H _ { \mathcal { T } } ( \alpha ) \|$ is large over the random draw of the training set  . Second, we extend this fixed-parameter concentration bound uniformly over the entire parameter space, thereby obtaining control of the deviation simultaneously for all α in the parameter space.

Consider a classification problem with training points $\mathcal { T } = \{ x _ { \ell } , y _ { \ell } \} _ { \ell = 1 } ^ { N }$ where x is sampled from an underlying distribution over an input space Ω and $y _ { \ell } \in \{ 0 , \bar { 1 } \}$ . Let $H _ { \mathcal { T } } ( \alpha )$ denote the training Hamiltonian defined in Eq. (1), constructed from an arbitrary encoding unitary $U ( x , \alpha )$ parameterized by $\alpha \in \Theta = \bar { [ - 1 , 1 ] ^ { M } }$ . For any labeled input $z = ( x , y )$ , we define $H _ { z } ^ { \prime } ( \alpha ) \ = \ I - \bar { H _ { z } } ( \alpha )$ , where $H _ { z } ( \alpha ) = U ^ { \dagger } ( x , \alpha ) O ( y ) U ( x , \alpha )$ We then define the population Hamiltonian as $H _ { \mathcal { D } } ( \alpha ) = \mathrm { \tilde { E } } _ { z \sim \mathcal { D } } [ H _ { z } ^ { \prime } ( \alpha ) ]$ , namely, the average encoding Hamiltonian associated with a random labeled input drawn from the data distribution. The key idea for the proof of Theorem 2 relies on controlling how much the empirical energy cost optimized during training can differ from the expected energy of the resulting trained state when averaged over the underlying data distribution. This amounts to controlling the deviation between the training Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ and the population Hamiltonian $H _ { \mathcal { D } } ( \alpha )$

Specifically, we first show that for any fixed $\alpha ,$ the following lemma holds.

Lemma 5. Let $\begin{array} { r } { H _ { \mathcal { T } } = I - \frac { 1 } { N } \sum _ { \ell } H _ { z _ { \ell } } ( \alpha ) } \end{array}$ be a Hamiltonian embedding a training set  as in Eq. (1), such that $0 \preceq H _ { z } ( \alpha ) \preceq I$ for all points $z = ( x , y ) \in \Omega \times \{ 0 , 1 \}$ and all $\alpha \in \Theta$ , and let $\begin{array} { r } { \dot { H } _ { \mathcal { D } } ( \alpha ) = \mathbb { E } _ { z \sim \mathcal { D } } [ H _ { z } ^ { \prime } ( \alpha ) ] } \end{array}$ be the population Hamiltonian, with $\dot { H _ { z } ^ { \prime } } ( \alpha ) = I - \dot { H } _ { z } \dot { ( \alpha ) }$ . Then, for every $\alpha \in \Theta$ , with probability at least $1 - \delta$ over the draw of N-size training datasets $\mathcal { T } = \{ ( x _ { \ell } , y _ { \ell } ) \} _ { \ell = 1 } ^ { N } ,$

$$
\| H _ { T } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) \| _ { \infty } \leq \sqrt { \frac { 8 } { N } \log \left( \frac { 2 D } { \delta } \right) } .\tag{A1}
$$

where $D = 2 ^ { n }$ is the Hilbert space dimension.

The proof of Lemma 5 relies on the matrix Hoeffding inequality. We report here the relative theorem, the proof can be found in [Tro12].

Theorem 6 (Matrix Hoeffding (Theorem 1.3 in [Tro12])). Consider a finite sequence $\{ X _ { k } \}$ of independent, random, selfadjoint matrices with dimension D, and let $\left\{ A _ { k } \right\}$ be a sequence offixed self-adjoint matrices. Assume that each random matrix satisfies

$$
{ \mathbb E } _ { k } [ X _ { k } ] = 0 \qquad a n d \qquad X _ { k } ^ { 2 } \preceq A _ { k } ^ { 2 } a l m o s t s u r e l y\tag{A2}
$$

Then, for all $t \geq 0$

$$
\operatorname* { P r } \left\{ \lambda _ { \operatorname* { m a x } } \left( \sum _ { k } X _ { k } \right) \geq t \right\} \leq D \cdot e ^ { - t ^ { 2 } / 8 \sigma ^ { 2 } } , w h e r e \sigma ^ { 2 } : = \left\| \sum _ { k } A _ { k } ^ { 2 } \right\| _ { \infty } .\tag{A3}
$$

In our proof of Lemma 5 we will actually need to bound the operator norm of the sum of random matrices, not just the largest eigenvalue. However, we can obtain a bound on the operator norm of a sum of random matrices $\textstyle \sum _ { k } X _ { k }$ by applying Theorem 6 twice: one to $\textstyle \sum _ { k } X _ { k }$ and one to $- \sum k X _ { k }$ as explained in the following corollary.

Corollary 7 (Matrix Hoeffding for operator norm (Corollary of Theorem 1.3 in [Tro12])). Consider a finite sequence $\{ X _ { k } \}$ ofindependent, random, self-adjoint matrices with dimension D, and let $\left\{ A _ { k } \right\}$ be a sequence offixed self-adjoint matrices. Assume that each random matrix satisfies

$$
{ \mathbb E } _ { k } [ X _ { k } ] = 0 \qquad a n d \qquad X _ { k } ^ { 2 } \preceq A _ { k } ^ { 2 } a l m o s t s u r e l y\tag{A4}
$$

Then, for all $t \geq 0$

$$
\operatorname* { P r } \left\{ \left\| \sum _ { k } X _ { k } \right\| _ { \infty } \geq t \right\} \leq 2 D \cdot e ^ { - t ^ { 2 } / 8 \sigma ^ { 2 } } , w h e r e \sigma ^ { 2 } : = \left\| \sum _ { k } A _ { k } ^ { 2 } \right\| _ { \infty } .\tag{A5}
$$

Proof. For a self-adjoint matrix A, the operator norm satisfies $\| A \| _ { \infty } \mathit { \Omega } = \operatorname* { m a x } \left\{ \lambda _ { \operatorname* { m a x } } ( A ) , \lambda _ { \operatorname* { m a x } } ( - A ) \right\}$ . Therefore, $\| A \| _ { \infty } \geq t$ is equivalent to

$$
\lambda _ { \operatorname* { m a x } } ( A ) \geq t \qquad \mathrm { o r } \qquad \lambda _ { \operatorname* { m a x } } ( - A ) \geq t .\tag{A6}
$$

Applying Tropp’s one-sided matrix Hoeffding bound to $\textstyle \sum _ { k } X _ { k }$ gives

$$
\operatorname* { P r } \left[ \lambda _ { \operatorname* { m a x } } \left( \sum _ { k } X _ { k } \right) \geq t \right] \leq D \mathrm { e x p } \left( - { \frac { t ^ { 2 } } { 8 \sigma ^ { 2 } } } \right) .\tag{A7}
$$

Applying the same bound $\begin{array} { r } { \mathrm { t o } - \sum _ { k } X _ { k } } \end{array}$ gives

$$
\operatorname* { P r } \left[ \lambda _ { \operatorname* { m a x } } \left( - \sum _ { k } X _ { k } \right) \geq t \right] \leq D \mathrm { e x p } \left( - { \frac { t ^ { 2 } } { 8 \sigma ^ { 2 } } } \right) .\tag{A8}
$$

Hence, by the union bound,

$$
\begin{array} { r l } & { \operatorname* { P r } \left[ \left\| \displaystyle \sum _ { k } X _ { k } \right\| _ { \infty } \geq t \right] \leq \operatorname* { P r } \left[ \lambda _ { \operatorname* { m a x } } \left( \displaystyle \sum _ { k } X _ { k } \right) \geq t \right] + \operatorname* { P r } \left[ \lambda _ { \operatorname* { m a x } } \left( - \displaystyle \sum _ { k } X _ { k } \right) \geq t \right] } \\ & { \qquad \leq 2 D \mathrm { e x p } \left( - \displaystyle \frac { t ^ { 2 } } { 8 \sigma ^ { 2 } } \right) . } \end{array}\tag{A9}
$$

We are now ready to give the complete proof of our Lemma 5

Proof of Lemma 5. We define $X _ { z } ( \alpha ) = H _ { z } ^ { \prime } ( \alpha ) - H _ { \mathcal { D } } ( \alpha )$ . Note that $\begin{array} { r } { H _ { \mathcal { T } } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) = \frac { 1 } { N } \sum _ { z \in \mathcal { T } } X _ { z } ( \alpha ) } \end{array}$ and that

$$
\begin{array} { r } { \mathbb { E } _ { z \sim \mathcal { D } } [ X _ { z } ( \alpha ) ] = \mathbb { E } _ { z \sim \mathcal { D } } [ H _ { z } ^ { \prime } ( \alpha ) ] - H _ { \mathcal { D } } ( \alpha ) = H _ { \mathcal { D } } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) = 0 . } \end{array}\tag{A10}
$$

Moreover, $- I \preceq X _ { z } \preceq I$ since we assumed $0 \preceq H _ { z } ^ { \prime } \preceq I$ for any point z and therefore also $0 \preceq H _ { d } \preceq I .$ . Hence, $\| X _ { z } \| _ { \infty } \leq 1$ for any data point z. We can then bound the difference in norm between $H _ { \mathcal { T } } ( \alpha )$ and $H _ { \mathcal { D } } ( \alpha )$ by

$$
\lVert H _ { \mathcal { T } } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) \rVert _ { \infty } \leq \left. \frac { 1 } { N } \sum _ { z \in \mathcal { T } } X _ { z } ( \alpha ) \right. _ { \infty }\tag{A11}
$$

In order to prove the lemma is then sufficient to bound the quantity $\begin{array} { r } { \left. \frac { 1 } { N } \sum _ { z \in \mathcal { T } } X _ { z } ( \alpha ) \right. _ { \infty } } \end{array}$ . Since the samples $z \in \mathcal { T }$ are drawn independently, the matrices $X _ { z } ( \alpha )$ are also independent for every fixed α. Moreover, under the assumption $0 \preceq H _ { z } ( \alpha ) \preceq I ,$ they satisfy $\| X _ { z } ( \alpha ) \| _ { \infty } \leq 1$ for all z. We can therefore apply the matrix Hoeffding inequality of Corollary 7. In particular, in our case $X _ { k } ^ { 2 } \preceq I$ and therefore $\sigma ^ { 2 } = \left\| \sum _ { k = 1 } ^ { N } I \right\| _ { \infty } = N$ to obtain:

$$
\operatorname* { P r } _ { T } \left[ \left\| \frac { 1 } { N } \sum _ { i = 1 } ^ { N } X _ { i } \right\| _ { \infty } \geq \varepsilon \right] \leq 2 D \mathrm { e x p } \left( - \frac { N \varepsilon ^ { 2 } } { 8 } \right) ,\tag{A12}
$$

where $D = 2 ^ { n }$ is the Hilbert space dimension. Finally, after setting the right hand to be $\delta ,$ we get

$$
\operatorname* { P r } _ { T } \left[ \left. \frac { 1 } { N } \sum _ { i = 1 } ^ { N } X _ { i } \right. _ { \infty } \leq \sqrt { \frac { 8 } { N } \log \left( \frac { 2 D } { \delta } \right) } \right] \geq 1 - \delta .\tag{A13}
$$

which, combined with Eq. (A11) proves the bound stated in Lemma 5.

Lemma 5 bounds the probability of having a large difference in operator norm between the training and population Hamiltonian for a fixed α. Importantly, this probability is taken over different random datasets . In general, the set of trained parameter $\alpha ^ { \star }$ will depend on the particular training set used in the training, therefore when averaging over different  this dependence must be taken into account. When the Hamiltonian embedding is Lipschitz continuous in α, the fixed-α bound can be extended uniformly over the whole parameter space. This is done by first applying the bound to all points of a finite covering net and then using the Lipschitz property to control the deviation at parameters outside the net. This is shown in the full proof of Theorem 2, that we restate here for convenience.

Theorem 2 (Uniform energy-based generalization bound). Assume a Hamiltonian embedding model such that $0 ~ \preceq$ $H _ { z } ( \alpha ) \preceq I$ and that there exists a constant $L > 0$ such that $\| H _ { z } ( \alpha ) - H _ { z } ( \beta ) \| _ { \infty } \leq L \| \alpha - \beta \| _ { \infty } f o r$ all points $z = ( x , y ) \in$ $\Omega \times \{ 0 , 1 \}$ and all $\alpha , \beta \in [ - 1 , 1 ] ^ { M }$ . Then, for every $ { \varepsilon } \in ( 0 , 4 L ]$ , with probability at least $1 - \delta$ over the draw of N-size training datasets,

$$
\operatorname* { s u p } _ { \alpha \in [ - 1 , 1 ] ^ { M } } \| H _ { T } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) \| _ { \infty } \leq \sqrt { \frac { 8 } { N } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 2 4 L } { \varepsilon } \right) \right] } + \frac { \varepsilon } { 4 } .\tag{32}
$$

where $D = 2 ^ { n }$ is the Hilbert space dimension.

Proof. Let the parameters take values in $\Theta = [ - 1 , 1 ] ^ { M }$ . Equip the parameter space with the $\ell _ { \infty }$ norm $\| \alpha - \beta \| _ { \infty } : =$ $\mathrm { m a x } _ { j = 1 , \dots , M } | \alpha _ { j } - \beta _ { j } |$ . Then $[ - 1 , 1 ] ^ { M }$ is exactly the radius-one $\ell _ { \infty }$ ball in $\mathbb { R } ^ { M }$ . Let $\eta \in ( 0 , 1 ]$ . An η-net $A _ { \eta } \subset \Theta$ is a finite set such that for every $\alpha \in \Theta ,$ , there exists $\beta \in \mathcal A _ { \eta }$ satisfying $\| \alpha - \beta \| _ { \infty } \leq \eta$ . It is known [Ver26] that the covering numbers of a ball $B _ { R }$ of radius $R > 0$ around a fixed point in $\mathbb { R } ^ { M }$ , for $0 \leq \eta \leq R$ is:

$$
\mathcal { N } ( B _ { R } , | | . | | , \eta ) = | \mathcal { A } _ { \eta } | \leq \biggl ( 1 + \frac { 2 R } { \eta } \biggr ) ^ { M } \leq \biggl ( \frac { 3 } { \eta } \biggr ) ^ { M } .\tag{A14}
$$

Where in the last step we consider $R = 1$ , as we are considering a ball of radius $R = 1$ around 0 (that corresponds to $\Theta = [ - 1 , 1 ] ^ { M } )$ . For each fixed $\beta \in \mathcal { A } _ { \eta } ,$ Eq. (A1) gives Prob $\begin{array} { r } { - [ \| H _ { \mathcal { T } } ( \beta ) - H _ { \mathcal { D } } ( \beta ) \| \ge t ] \le 2 D \mathrm { e x p } \left( - \frac { N t ^ { 2 } } { 8 } \right) } \end{array}$ . We now bound the probability that there is a large deviation for at least a point in the net. Define the bad event

$$
E _ { \beta } : = \left\{ \| H _ { T } ( \beta ) - H _ { \mathcal { D } } ( \beta ) \| \ge t \right\} .\tag{A15}
$$

The event that at least one net point is bad is $\textstyle \bigcup _ { \beta \in { \mathcal { A } } _ { \eta } } E _ { \beta }$ . The union bound does not require the events $E _ { \beta }$ to be independent. It gives

$$
\mathrm { P r o b } _ { \mathcal { T } } \left[ \operatorname* { s u p } _ { \beta \in \mathcal { A } _ { \eta } } \| H _ { \mathcal { T } } ( \beta ) - H _ { \mathcal { D } } ( \beta ) \| \ge t \right] \le \sum _ { \beta \in \mathcal { A } _ { \eta } } \mathrm { P r o b } _ { \mathcal { T } } ( E _ { \beta } )\tag{A16}
$$

$$
\leq 2 D | \mathcal { A } _ { \eta } | \mathbf { e x p } \biggl ( - \frac { N t ^ { 2 } } { 8 } \biggr )\tag{A17}
$$

$$
\leq 2 D \left( \frac { 3 } { \eta } \right) ^ { M } \exp \left( - \frac { N t ^ { 2 } } { 8 } \right) .\tag{A18}
$$

To make the right-hand side at most $\delta ,$ it is enough to choose

$$
t = \sqrt { \frac { 8 } { N } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 3 } { \eta } \right) \right] } .\tag{A19}
$$

Hence, with probability at least $1 - \delta ,$

$$
\operatorname* { s u p } _ { \beta \in \mathcal { A } _ { \eta } } \| H _ { \mathcal { T } } ( \beta ) - H _ { \mathcal { D } } ( \beta ) \| \leq \sqrt { \frac { 8 } { N } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 3 } { \eta } \right) \right] } .\tag{A20}
$$

The finite-net bound is not yet enough: the final trained parameter $\alpha ^ { \star } ( \tau )$ need not lie exactly on the grid of points in $\mathcal { A } _ { \eta } .$ . However, we can extend the bound to a supremum of all the parameter points in $\alpha \in \mathsf { \bar { \Gamma } } [ - 1 , 1 ] ^ { M }$ by assuming regularity of the Hamiltonian family. In particular, Lipschitz continuity implies the existence of a $L \ge 0$ such that for every data point z and every $\alpha , \beta \in [ - 1 , 1 ] ^ { M }$ it holds that $\| H _ { z } ( \alpha ) - \bar { H _ { z } } ( \beta ) \| _ { \infty } \leq L | \alpha - \beta | _ { \infty }$ . We notice that both the training Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ and the full population Hamiltonian inherit the same Lipschitz constant:

$$
\lVert H _ { T } ( \alpha ) - H _ { T } ( \beta ) \rVert \leq \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert H _ { z _ { i } } ( \alpha ) - H _ { z _ { i } } ( \beta ) \rVert\tag{A21}
$$

$$
\leq L \| \alpha - \beta \| _ { \infty } .\tag{A22}
$$

and, similarly,

$$
\| H _ { \mathcal { D } } ( \alpha ) - H _ { \mathcal { D } } ( \beta ) \| \le \mathbb { E } _ { z } [ \| H _ { z } ( \alpha ) - H _ { z } ( \beta ) \| ]\tag{A23}
$$

$$
\leq L \| \alpha - \beta \| _ { \infty } .\tag{A24}
$$

Define $\Delta _ { T } ( \alpha ) : = H _ { T } ( \alpha ) - H _ { \mathcal { D } } ( \alpha )$ . Combining Eqs. (A22) and (A24),

$$
\| \Delta _ { T } ( \alpha ) - \Delta _ { T } ( \beta ) \| \le \| H _ { T } ( \alpha ) - H _ { T } ( \beta ) \| + \| H _ { \mathcal { D } } ( \alpha ) - H _ { \mathcal { D } } ( \beta ) \|\tag{A25}
$$

$$
\leq 2 L \| \alpha - \beta \| _ { \infty } .\tag{A26}
$$

Now take an arbitrary $\alpha \in [ - 1 , 1 ] ^ { M }$ . Choose $\beta \in \mathcal A _ { \eta }$ such that $\| \alpha - \beta \| _ { \infty } \leq \eta$ . Then

$$
\| \Delta _ { T } ( \alpha ) \| \le \| \Delta _ { T } ( \beta ) \| + \| \Delta _ { T } ( \alpha ) - \Delta _ { T } ( \beta ) \|\tag{A27}
$$

$$
\leq \| \Delta _ { T } ( \beta ) \| + 2 L \eta .\tag{A28}
$$

Taking the supremum over all α gives

$$
\begin{array} { r } { \displaystyle \operatorname* { s u p } _ { \alpha \in [ - 1 , 1 ] ^ { M } } \| \Delta _ { T } ( \alpha ) \| \leq \displaystyle \operatorname* { s u p } _ { \beta \in \mathcal { A } _ { \eta } } \| \Delta _ { T } ( \beta ) \| + 2 L \eta . } \end{array}\tag{A29}
$$

We can now conclude the proof of the Theorem by inserting Eq. (A20) in Eq. (A29) in order to obtain:

$$
\operatorname* { s u p } _ { \alpha \in [ - 1 , 1 ] ^ { M } } \| H _ { \mathcal { T } } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) \| \leq \sqrt { \frac { 8 } { N } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 2 4 L } { \varepsilon } \right) \right] } + \frac { \varepsilon } { 4 } .\tag{A30}
$$

where we chose $\begin{array} { r } { \eta = \frac { \varepsilon } { 8 L } } \end{array}$ . This concludes the proof of Theorem 2.

Theorem 2 provides a bound on the energy generalization gap defined in Eq. (31), it holds for any α and therefore even for the $\alpha ^ { \bar { \star } }$ obtained after the training. The energy generalization gap can be directly used to bound the total risk R defined in Eq. (30) as we show in Corollary 3.

Corollary 3 (Bound on the total risk). Assume the setting of Theorem 2. Let $\left( \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } \right)$ be the output of the training procedure. Let $\widehat { C } ( \psi _ { \tau , \alpha ^ { \star } } , \alpha ^ { \star } ) ~ = ~ \langle \psi _ { \tau , \alpha ^ { \star } } | H _ { \mathcal { T } } ( \alpha ^ { \star } ) | \psi _ { \tau , \alpha ^ { \star } } \rangle$ be the empirical energy reached by the trained model, and let $C ( \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } ) = \langle \psi _ { \mathcal { T } , \alpha ^ { \star } } | H _ { \mathcal { D } } ( \alpha ^ { \star } ) | \psi _ { \mathcal { T } , \alpha ^ { \star } } \rangle$ be its population energy. Moreover, let $R ( \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } ) = \operatorname* { P r } _ { x \sim \mathcal { D } } \left| \tilde { f } _ { \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } } ( x ) \neq f ( x ) \right|$ be the classification risk of the trained model. Then, with probability at least $1 - \delta$ over the draw of the training set $\tau$ , the classification risk of the trained model can be bounded by

$$
R ( \psi _ { \tau , \alpha ^ { \star } } , \alpha ^ { \star } ) \leq 2 \widehat { C } ( \psi _ { \tau , \alpha ^ { \star } } , \alpha ^ { \star } ) + \varepsilon ,\tag{33}
$$

provided that the number of training samples satisfies

$$
N \geq \frac { 1 2 8 } { \varepsilon ^ { 2 } } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 2 4 L } { \varepsilon } \right) \right] .\tag{34}
$$

Proof. Theorem 2 bounds the energy generalization gap for any parameter $\alpha \in \Theta = [ - 1 , 1 ] ^ { M }$ and, in particular, for the $\alpha ^ { \star }$ obtained after the model is trained. In particular, for any $\epsilon > 0 ,$ , with probability larger than $\hat { 1 } - \delta$ over different training data $\tau _ { \iota }$ , it holds that

$$
\operatorname* { s u p } _ { \alpha \in [ - 1 , 1 ] ^ { M } } \| H _ { \mathcal { T } } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) \| \leq \sqrt { \frac { 8 } { N } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 2 4 L } { \varepsilon } \right) \right] } + \frac { \varepsilon } { 4 } .\tag{A31}
$$

Suppose now that the number of training samples satisfies

$$
N \geq \frac { 1 2 8 } { \varepsilon ^ { 2 } } \left[ \log \left( \frac { 2 D } { \delta } \right) + M \log \left( \frac { 2 4 L } { \varepsilon } \right) \right] ,\tag{A32}
$$

then Theorem 2 implies that, with probability $\begin{array} { r } { 1 - \delta , \mathsf { s u p } _ { \alpha \in [ - 1 , 1 ] ^ { M } } \| H _ { \mathcal { T } } ( \alpha ) - H _ { \mathcal { D } } ( \alpha ) \| \le \frac { \varepsilon } { 2 } } \end{array}$ . Recalling the definition of total cost, we obtain:

$$
C ( \psi _ { { \mathcal { T } } , \alpha ^ { \star } } , \alpha ^ { \star } ) = \langle \psi _ { { \mathcal { T } } , \alpha ^ { \star } } | H _ { \mathcal { D } } ( \alpha ^ { \star } ) | \psi _ { { \mathcal { T } } , \alpha ^ { \star } } \rangle
$$

$$
\leq \left. \psi _ { T , \alpha ^ { \star } } \right. H _ { T } ( \alpha ^ { \star } ) \left. \psi _ { T , \alpha ^ { \star } } \right. + \left. H _ { \mathcal { D } } ( \alpha ^ { \star } ) - H _ { T } ( \alpha ^ { \star } ) \right.\tag{A33}
$$

(A34)

$$
\leq \widehat { C } ( \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } ) + \frac { \varepsilon } { 2 } .\tag{A35}
$$

It now remains to relate the population energy to the total classification error R. Recall that, for a data point $\boldsymbol z = ( x , y )$ embedded through the Hamiltonian $H _ { z } ( \alpha )$ , the energy loss evaluated on the state ψ is defined in Eq. (7) as:

$$
\ell _ { \mathrm { e n e r g y } } ( \psi , H _ { z } ( \alpha ) ) = 1 - \langle \psi | H _ { z } ( \alpha ) | \psi \rangle .\tag{A36}
$$

Moreover, the model prediction rule for a point x is given by Eq. (5), which we also state here for convenience:

$$
\widetilde { f } _ { \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } } ( x ^ { \prime } ) = \underset { \widetilde { y } \in \{ 0 , 1 \} } { \mathrm { a r g m a x } } P _ { \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } } ( x ^ { \prime } , \widetilde { y } ) .\tag{A37}
$$

where $P _ { \psi _ { T , \alpha } , \alpha } ( x , y ) = \mathrm { T r } \left( O ( y ) U ( x , \alpha ) | \psi _ { \tau , \alpha } \rangle \langle \psi _ { \tau , \alpha } | U ^ { \dagger } ( x , \alpha ) \right)$ . We notice that if $( x , y )$ is misclassified, then the model assigns at least as much probability to the wrong label as to the true label, in other words

$$
\ell _ { \mathrm { e n e r g y } } ( \psi , H _ { z } ( \alpha ) ) = 1 - \left. \psi _ { \mathcal { T } , \alpha } \right| H _ { z } ( \alpha ) \left| \psi _ { \mathcal { T } , \alpha } \right. \geq \frac { 1 } { 2 } .\tag{A38}
$$

for misclassified point z. This directly implies that $\mathbf { 1 } \{ \tilde { f } _ { \alpha , \psi _ { \mathcal { T } , \alpha } } ( x ) ~ \neq ~ f ( x ) \} ~ \leq ~ 2 \ell _ { \mathrm { e n e r g y } } ( \psi , H _ { z } ( \alpha ) )$ for any point $z = ( x , y )$ . Therefore, taking the expectation over different point z for $x \sim \mathcal { D }$ and remembering that $R ( \psi , \alpha ) =$ $\mathrm { P r } _ { x \sim \mathcal { D } } [ \widetilde { f } _ { \psi _ { T , \alpha } , \alpha } ( x ) \neq f ( x ) ]$ , we obtain:

$$
R ( \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } ) \leq 2 C ( \psi _ { \mathcal { T } , \alpha ^ { \star } } , \alpha ^ { \star } )
$$

$$
\leq 2 { \widehat { C } } ( \psi _ { T , \alpha ^ { \star } } , \alpha ^ { \star } ) + \varepsilon .\tag{A39}
$$

(A40)

where in the second equation we used the bound on the population cost in Eq. (A33) given by the training cost plus the energy generalization gap assuming the sample size given in Eq. (A32).

## 1. Lipschitz assumption

In Theorem 2, we assumed that the parameterized Hamiltonians used to encode the data satisfy a Lipschitz continuity condition. Namely, we required that there exists a constant $L > 0$ such that $\| H _ { z } ( \alpha ) - H _ { z } ( \beta ) \| \leq \hat { L } \| \alpha - \beta \| _ { \infty }$ for any $\alpha , \beta \in [ - 1 , 1 ] ^ { M }$ and all the labeled input $z = ( x , y )$ . We now show that this is a natural assumption, and that it holds for common choices of encoding unitaries $U ( x , \alpha )$ used in the construction of $H _ { z } ( \alpha ) = U ^ { \dagger } ( x , \alpha ) O ( y ) U ( x , \alpha )$

Suppose that the data are encoded through rotation angles given by linear combinations of the input features, for example

$$
U ( \alpha , x ) = e ^ { - i \theta _ { \alpha } ( x ) G } , \qquad \theta _ { \alpha } ( x ) = \sum _ { j = 1 } ^ { M } \alpha _ { j } x _ { j } .\tag{A41}
$$

If $H _ { z } ( \alpha ) = U ( x , \alpha ) ^ { \dagger } O ( y ) U ( x , \alpha )$ with $\| O ( y ) \| \le 1$ , then differentiating with respect to $\alpha _ { j }$ gives a commutator-type expression. Since $\begin{array} { r } { \frac { \partial \theta _ { \alpha } ( x ) } { \partial \alpha _ { j } } = x _ { j } , } \end{array}$ , one obtains

$$
\left\| \frac { \partial H _ { z } ( \alpha ) } { \partial \alpha _ { j } } \right\| \leq 2 | x _ { j } | \| G \| .\tag{A42}
$$

If the features are uniformly bounded, $| x _ { j } | \leq B _ { x } ,$ then

$$
\left\| \frac { \partial H _ { z } ( \alpha ) } { \partial \alpha _ { j } } \right\| \le 2 B _ { x } \| G \| .\tag{A43}
$$

Therefore, moving coordinate by coordinate from α to $\beta ,$

$$
\| H _ { z } ( \alpha ) - H _ { z } ( \beta ) \| \leq 2 B _ { x } \| G \| \sum _ { j = 1 } ^ { M } | \alpha _ { j } - \beta _ { j } | .\tag{A44}
$$

In particular,

$$
\begin{array} { r } { \| H _ { z } ( \alpha ) - H _ { z } ( \beta ) \| \le 2 M B _ { x } \| G \| \| \alpha - \beta \| _ { \infty } . } \end{array}\tag{A45}
$$

Thus, for Pauli generators with $\| G \| = 1$ , one may take $L \leq 2 M B _ { x } .$ , up to convention-dependent factors such as the presence of $\mathtt { a } 1 / 2$ in the rotation generator.

## Appendix B: Gradient evaluation and the Hellmann–Feynman relation

Here we justify why, when evaluating the gradient of the cost function in Eq. (6), the ground state can be treated as fixed at the value corresponding to the current parameters, despite its implicit dependence on α. We formalize the result in Theorem 8.

Theorem 8 (Energy cost gradient). Assume that the data Hamiltonian $\begin{array} { r } { H _ { \mathcal { T } } ( \alpha ) = I - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } H _ { z _ { \ell } } ( \alpha ) } \end{array}$ , as defined in $E q . ( 1 )$ is continuously differentiable with a nondegenerate ground state. Define the energy cost as in $E q .$ (6)

$$
\widehat { C } _ { \mathcal { T } } ( \psi _ { \tau , \alpha } , \alpha ) = \langle \psi _ { \tau , \alpha } | H _ { \mathcal { T } } ( \alpha ) | \psi _ { \tau , \alpha } \rangle .\tag{B1}
$$

Then, for every parameter $\alpha _ { j } ,$

$$
\left. \frac { \partial } { \partial \alpha _ { j } } \widehat { C } _ { T } ( \psi _ { \tau , \alpha } , \alpha ) \right| _ { \alpha = \alpha _ { t } } = - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } \langle \psi _ { \tau , \alpha _ { t } } | \left. \frac { \partial H _ { z _ { \ell } } ( \alpha ) } { \partial \alpha _ { j } } \right| _ { \alpha = \alpha _ { t } } | \psi _ { \tau , \alpha _ { t } } \rangle .\tag{B2}
$$

Proof. Remember that the cost function as a function of the parameter α is defined as:

$$
{ \widehat C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha } , \alpha ) = \langle \psi _ { \mathcal { T } , \alpha } \vert H _ { \mathcal { T } } ( \alpha ) \vert \psi _ { \mathcal { T } , \alpha } \rangle ,\tag{B3}
$$

where $\vert \psi _ { T , \alpha } \rangle$ is a normalized ground state of $H _ { \mathcal { T } } ( \alpha )$ , satisfying $H _ { \mathcal { T } } ( \alpha ) \left| \psi _ { \mathcal { T } , \alpha } \right. = E _ { 0 } ( \alpha ) \left| \psi _ { \mathcal { T } , \alpha } \right.$ with $E _ { 0 } ( \alpha )$ the ground energy of the data Hamiltonian $H _ { \mathcal { T } } ( \alpha )$ at the point α.

Assuming that the ground state is non-degenerate and differentiable with respect to $\alpha ,$ differentiating $\widehat { C } \tau ( \psi _ { T , \alpha } , \alpha )$ with respect to $\alpha _ { j }$ gives

$$
\begin{array} { r l } & { \frac { \partial \widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha } , \alpha ) } { \partial \alpha _ { j } } = \left. \partial _ { j } \psi _ { \mathcal { T } , \alpha } \right| H _ { \mathcal { T } } ( \alpha ) \left| \psi _ { \mathcal { T } , \alpha } \right. + \left. \psi _ { \mathcal { T } , \alpha } \right| \frac { \partial H _ { \mathcal { T } } ( \alpha ) } { \partial \alpha _ { j } } \left| \psi _ { \mathcal { T } , \alpha } \right. } \\ & { \qquad + \left. \psi _ { \mathcal { T } , \alpha } \right| H _ { \mathcal { T } } ( \alpha ) \left| \partial _ { j } \psi _ { \mathcal { T } , \alpha } \right. , } \end{array}\tag{B4}
$$

where $\left| \partial _ { j } \psi _ { T , \alpha } \right. \equiv \partial \left| \psi _ { T , \alpha } \right. / \partial \alpha _ { j }$ . Using the eigenvalue equation, the first and third terms combine to give

$$
E _ { 0 } ( \alpha ) \left( \langle \partial _ { j } \psi _ { \mathcal { T } , \alpha } \vert \psi _ { \mathcal { T } , \alpha } \rangle + \langle \psi _ { \mathcal { T } , \alpha } \vert \partial _ { j } \psi _ { \mathcal { T } , \alpha } \rangle \right) .\tag{B5}
$$

Since $\vert \psi _ { T , \alpha } \rangle$ is normalized, differentiating $\langle \psi _ { T , \alpha } | \psi _ { T , \alpha } \rangle = 1$ yields $\langle \partial _ { j } \psi _ { \tau , \alpha } | \psi _ { \tau , \alpha } \rangle + \langle \psi _ { \tau , \alpha } | \partial _ { j } \psi _ { \tau , \alpha } \rangle = 0$ . Therefore,

$$
\frac { \partial \widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha } , \alpha ) } { \partial \alpha _ { j } } = \langle \psi _ { \mathcal { T } , \alpha } | \frac { \partial H _ { \mathcal { T } } ( \alpha ) } { \partial \alpha _ { j } } | \psi _ { \mathcal { T } , \alpha } \rangle ,\tag{B6}
$$

which is the Hellmann-Feynman theorem.

In particular, at iteration t of Algorithm 1, after computing the ground state $\vert \psi _ { T , \alpha _ { t } } \rangle$ of $H _ { \mathcal { T } } ( \alpha _ { t } )$ , Eq. (B6) gives

$$
\frac { \partial \widehat { C } _ { \mathcal { T } } ( \psi _ { \tau , \alpha } , \alpha ) } { \partial \alpha _ { j } } | _ { \alpha = \alpha _ { t } } =  \psi _ { \tau , \alpha _ { t } } |  \frac { \partial H _ { \mathcal { T } } ( \alpha ) } { \partial \alpha _ { j } } | _ { \alpha = \alpha _ { t } } | \psi _ { \tau , \alpha _ { t } }  ,\tag{B7}
$$

Hence, although the ground state depends implicitly on the parameters $\alpha ,$ its derivative does not need to be evaluated when computing the gradient of the ground-state energy at the current parameter value. Equivalently, the frozen-state cost $\widehat { C } _ { \mathcal { T } } ( \psi _ { \mathcal { T } , \alpha _ { t } } , \alpha )$ has the same gradient as the ground-state energy $E _ { 0 } ( \alpha )$ at $\alpha = \alpha _ { t }$

Finally, using $\begin{array} { r } { H _ { \mathcal { T } } ( \alpha ) = I - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } H _ { z _ { \ell } } ( \alpha ) } \end{array}$ , we obtain

$$
\left. \frac { \partial \widehat { C } _ { \mathcal { T } } ( \psi _ { \tau , \alpha } , \alpha ) } { \partial \alpha _ { j } } \right| _ { \alpha = \alpha _ { t } } = - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } \left. \psi _ { \tau , \alpha _ { t } } \right| \left. \frac { \partial H _ { z _ { \ell } } ( \alpha ) } { \partial \alpha _ { j } } \right| _ { \alpha = \alpha _ { t } } \left| \psi _ { \tau , \alpha _ { t } } \right. ,\tag{B8}
$$

which gives the gradient used in Eq. (21) in the main text.

## 1. Gradients with approximate ground states

In practice, the ground states in the optimization loop are obtained through quantum algorithms like QPE or SKQD. These may return only an approximate ground state $\left| \widetilde { \psi } _ { \alpha } \right.$ of $H _ { \mathcal { T } } ( \alpha )$ . Defining the corresponding approximate energy as

$$
\widetilde { E } ( \alpha ) = \left. \widetilde { \psi } _ { \alpha } \right| H _ { \mathcal { T } } ( \alpha ) \left| \widetilde { \psi } _ { \alpha } \right. ,\tag{B9}
$$

its exact derivative with respect to $\alpha _ { j }$ is

$$
\begin{array} { l } { \displaystyle \frac { \partial \widetilde { E } ( \alpha ) } { \partial \alpha _ { j } } = \left. \partial _ { j } \widetilde { \psi } _ { \alpha } \right| H _ { T } ( \alpha ) \left| \widetilde { \psi } _ { \alpha } \right. + \left. \widetilde { \psi } _ { \alpha } \right| \displaystyle \frac { \partial H _ { T } ( \alpha ) } { \partial \alpha _ { j } } \left| \widetilde { \psi } _ { \alpha } \right. } \\ { \displaystyle ~ + \left. \widetilde { \psi } _ { \alpha } \right| H _ { T } ( \alpha ) \left| \partial _ { j } \widetilde { \psi } _ { \alpha } \right. , } \end{array}\tag{B10}
$$

where $\left| \partial _ { j } \widetilde { \psi } _ { \alpha } \right. \equiv \partial \left| \widetilde { \psi } _ { \alpha } \right. / \partial \alpha _ { j }$ . Therefore, unless $\left| \widetilde { \psi } _ { \alpha } \right.$ is an exact eigenstate, the Hellmann-Feynman cancellation does not hold exactly for the derivative of $\widetilde E ( \alpha )$

Nevertheless, the gradient obtained by keeping the approximate ground state fixed provides a controlled approximation to the exact ground-state gradient. Let $\vert \breve { \psi } _ { \mathcal { T } , \alpha } \rangle$ denote the exact ground state of $H _ { \mathcal { T } } ( \alpha )$ and define its fidelity with the approximate state as $F ( \overset { \smile } { \alpha } ) = \vert \left. \psi _ { T , \alpha } \vert \widetilde { \psi } _ { \alpha } \right. \vert ^ { 2 }$ . Using Holder’s inequality together with the pure-state trace-¨ distance relation $\lVert | \widetilde { \psi } _ { \alpha } \rangle \langle \widetilde { \psi } _ { \alpha } | - | \psi _ { T , \alpha } \rangle \langle \psi _ { T , \alpha } | \rVert _ { 1 } = 2 \sqrt { 1 - F ( \alpha ) }$ , we obtain

$$
\left| \left. \widetilde { \psi } _ { \alpha } \right| \frac { \partial H _ { T } ( \alpha ) } { \partial \alpha _ { j } } \left| \widetilde { \psi } _ { \alpha } \right. - \left. \psi _ { T , \alpha } \right| \frac { \partial H _ { T } ( \alpha ) } { \partial \alpha _ { j } } \left| \psi _ { T , \alpha } \right. \right| \leq 2 \left\| \frac { \partial H _ { T } ( \alpha ) } { \partial \alpha _ { j } } \right\| \sqrt { 1 - F ( \alpha ) } .
$$

Using the Hellmann–Feynman relation in Eq. (B6), this gives

$$
\left. \left. \widetilde { \psi } _ { \alpha } \right. \frac { \partial H _ { T } ( \alpha ) } { \partial \alpha _ { j } } \left. \widetilde { \psi } _ { \alpha } \right. - \frac { \partial E _ { 0 } ( \alpha ) } { \partial \alpha _ { j } } \right. \leq 2 \left. \frac { \partial H _ { T } ( \alpha ) } { \partial \alpha _ { j } } \right. \sqrt { 1 - F ( \alpha ) } .\tag{B11}
$$

Hence, if the fidelity of the ground state found by the quantum solver is $F ( \alpha ) \geq 1 - \epsilon ,$ the error in the gradient obtained from the approximate ground state is at most $2 \left\| \partial H _ { T } ( \alpha ) / \partial \alpha _ { j } \right\| \sqrt { \epsilon }$ . Moreover, as discussed in Appendix C, for the standard Pauli encoding introduced in the main text, the operator norm $\left\| \partial H _ { T } ( \alpha ) / \partial \alpha _ { j } \right\|$ is bounded.

## Appendix C: Absence of barren plateaus in the Hamiltonian embedding model

One appealing feature of Hamiltonian embedding is that, for local encoding Hamiltonians, once the datasetencoding ground state has been obtained, inference reduces to estimating a local observable on the learned state. Together with enabling classical-shadow-based inference, this locality structure also allows us to establish trainability guarantees for the model. In particular, for a particular class of Pauli-rotation encoding $U ( x , \alpha )$ , we can theoretically prove that a nontrivial subset of the trainable parameters avoids barren plateaus. We present the proof for the simple observable $A = 1 / n \textstyle \sum _ { i } Z _ { i }$ and a family of shallow Hamiltonian embedding architectures. The same argument extends to observables that are sums of bounded-locality Pauli terms, provided that each Pauli term entering the bound anticommutes with at least one final-layer rotation generator acting on its support.

Theorem 9 (Cost-level absence of barren plateaus). Let $U ( x , \alpha )$ be a depth-K layered unitary of the form $U ( x , \alpha ) =$ $V ^ { K } ( \alpha ^ { K } ) U _ { \mathrm { e n c } } ^ { K } ( x ) \cdot \cdot \cdot V ^ { 1 } ( \alpha ^ { 1 } ) U _ { \mathrm { e n c } } ^ { 1 } ( x )$ , where $\alpha = ( \alpha ^ { 1 } , \dots , \alpha ^ { K } )$ contains M trainable parameters. Assume that $\begin{array} { r } { A = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ and the final trainable layer contains one independent $X - o r$ Y-rotation on each readout qubit, parameterized by the angles $\alpha _ { 1 } ^ { K } , \ldots , { \overset { \cdot } { \alpha } } _ { n } ^ { K }$ with $\alpha _ { i } ^ { K } \in [ 0 , 2 \pi ]$ . Let

$$
C _ { t } ( \alpha ) : = \widehat { C } ( \psi _ { \tau , \alpha _ { t } } , \alpha ) = \mathbb { E } _ { ( x , y ) \in \mathcal { T } } \left[ \ell _ { \mathrm { e n g } } \left( \psi _ { \tau , \alpha _ { t } } , H _ { ( x , y ) } ( \alpha ) \right) \right]\tag{C1}
$$

be the empirical energy cost at optimization step t, where the state $\psi _ { T , \alpha _ { t } }$ is kept fixed when differentiating with respect to α. Assume that, for some $\gamma > 0$

$$
C _ { t } ( \alpha _ { t } ) \leq \frac { 1 } { 2 } - \gamma .\tag{C2}
$$

Then the coordinate-averaged variance of the gradients of the cost function satisfies

$$
\frac { 1 } { M } \sum _ { \mu = 1 } ^ { M } \mathrm { V a r } _ { \alpha _ { \mu } } \left( \frac { \partial C _ { t } ( \alpha ) } { \partial \alpha _ { \mu } } \right) \geq \frac { \gamma ^ { 2 } } { 2 M n } ,\tag{C3}
$$

where $\operatorname { V a r } _ { \alpha _ { \mu } }$ denotes the coordinate-wise variance with respect to $\alpha _ { \mu } \sim \mathrm { U n i f } ( [ 0 , 2 \pi ] )$ while keeping all the other parameters fixed at their optimization-step values.

Proof. At optimization step t, we keep the state $\vert \psi _ { T , \alpha _ { t } } \rangle$ fixed and optimize only the circuit parameters appearing in the data-dependent Hamiltonian. For a single training example $\bar { z _ { \ell } } = ( x _ { \ell } , y _ { \ell } )$ , the fixed-state energy loss is defined as in Eq. (7):

$$
\begin{array} { r } { \ell _ { \mathrm { e n g } } ( \psi _ { \tau , \alpha _ { t } } , H _ { z _ { \ell } } ( \alpha ) ) = 1 - \left. \psi _ { \tau , \alpha _ { t } } \right. H _ { z _ { \ell } } ( \alpha ) \left. \psi _ { \tau , \alpha _ { t } } \right. , } \end{array}\tag{C4}
$$

where the state $\psi _ { T , \alpha _ { t } }$ is kept fixed when differentiating with respect to α. Let $\rho _ { t } ~ = ~ \vert \psi _ { T , \alpha _ { t } } \rangle ~ \langle \psi _ { T , \alpha _ { t } } \vert$ . Recalling the expression for $H _ { T } ( \alpha )$ , with $\begin{array} { r } { \dot { O } ( y ) = \frac { I + ( - 1 ) ^ { y } A } { 2 } } \end{array}$ and $\begin{array} { r } { A = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } Z _ { j } , } \end{array}$ we define

$$
\begin{array} { r } { h _ { j } ( x ; \boldsymbol { \alpha } ) : = \mathrm { T r } \left[ \rho _ { t } U ( x , \boldsymbol { \alpha } ) ^ { \dagger } Z _ { j } U ( x , \boldsymbol { \alpha } ) \right] . } \end{array}\tag{C5}
$$

Let us define $s ( x ) \ = \ ( - 1 ) ^ { y ( x ) }$ . Then the fixed-state single-sample energy loss is $\begin{array} { r }  \ell _ { \mathrm { e n g } } ( \psi _ { T , \alpha _ { t } } , H _ { ( x , y ) } ( \alpha ) ) \ = \ \frac { 1 } { 2 } \ \end{array}$ $\begin{array} { r } { \frac { s ( x ) } { 2 n } \sum _ { j = 1 } ^ { n } h _ { j } ( x ; \alpha ) } \end{array}$ , while the frozen total energy cost at optimization step t is

$$
C _ { t } ( \alpha ) = \frac { 1 } { 2 } - \frac { 1 } { 2 n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) h _ { j } ( x ; \alpha ) \right] .\tag{C6}
$$

<sup>The</sup> <sup>average</sup> <sup>is</sup> <sup>taken</sup> <sup>over</sup> <sup>the</sup> <sup>training</sup> <sup>set</sup> T <sup>.</sup> <sup>By</sup> <sup>the</sup> <sup>assumption</sup> <sup>in</sup> <sup>Eq.</sup> <sup>(C2),</sup> $\begin{array} { r } { C _ { t } ( \alpha _ { t } ) \leq \frac { 1 } { 2 } - \gamma } \end{array}$ . Equivalently,

$$
\frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) h _ { j } ( x ; \alpha _ { t } ) \right] \geq 2 \gamma .\tag{C7}
$$

We now isolate the dependence of the encoding unitary $U ( x , \alpha )$ on the final trainable layer. After fixing all previous layers, including both the data-encoding gates and the trainable gates in layers $1 , \ldots , \dot { K } - 1$ , we write ${ \hat { U } } ( x , \alpha ) =$ $\Breve { V _ { K } } ( \alpha ^ { K } ) \Breve { U _ { < K } } ( x , \Breve { \alpha ^ { < K } } )$ , where $U _ { < K } ( x , \alpha ^ { < K } )$ denotes the product of all layers before the final one, and $V _ { K } ( \alpha ^ { K } )$ is the last trainable layer. At optimization step t, the parameters $\alpha ^ { < K }$ are fixed at $\alpha _ { t } ^ { < K }$ , so we absorb them into the frozen unitary $W _ { t } ( x ) : = U _ { < K } ( x , \alpha _ { t } ^ { < K } )$ . Thus, for the purpose of analyzing the final layer at optimization step t, we write $U ( x , \alpha ^ { K } ) = V _ { K } ( \alpha ^ { K } ) W _ { t } ( x )$ . In particular, we take the final layer to consist of one trainable single-qubit rotation on each readout qubit, $\alpha ^ { K } = ( \alpha _ { 1 } ^ { K } , \cdot \cdot \cdot , \alpha _ { n } ^ { K } )$

By assumption, the rotation acting on qubit i is generated by a Pauli operator that does not commute with the measured observable $Z _ { i }$ . For example, if the final gate on qubit i is an $X _ { i } – r o t a t i o n$ , then $R _ { X _ { i } } ( \alpha _ { i } ^ { K } ) ^ { \dagger } Z _ { i } R _ { X _ { i } } ( \alpha _ { i } ^ { K } ) ~ =$ $Z _ { i } \cos \alpha _ { i } ^ { K } + Y _ { i } \sin \alpha _ { i } ^ { K }$ . Equivalently, after freezing $\rho _ { t } ,$ the dataset, all earlier parameters, and all final-layer parameters except $\alpha _ { i } ^ { K }$ , there exist real coefficients $a _ { i } ( x )$ and $b _ { i } ( x )$ such that we can rewrite $h _ { i }$ in Eq. (C5) as

$$
h _ { i } ( \boldsymbol { x } ; \alpha ) = a _ { i } ( \boldsymbol { x } ) \cos { \alpha _ { i } ^ { K } } + b _ { i } ( \boldsymbol { x } ) \sin { \alpha _ { i } ^ { K } } .\tag{C8}
$$

Note that the coefficients $a _ { i } ( x )$ and $b _ { i } ( x )$ are independent of $\alpha _ { i } ^ { K } .$

We now pass to the empirical cost by averaging $\ell _ { \mathrm { e n g } }$ over the input x. Since $\alpha _ { i } ^ { K }$ is independent of the data point $x ,$ we can define the labeled empirical coefficients $\overline { { A } } _ { i } : = \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) a _ { i } ( x ) \right]$ and $\overline { { B } } _ { i } : = \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) b _ { i } ( x ) \right]$ and obtain that:

$$
\mathbb { E } _ { \boldsymbol { x } \in \mathcal { T } } \left[ s ( \boldsymbol { x } ) h _ { i } ( \boldsymbol { x } ; \boldsymbol { \alpha } ) \right] = \overline { { A } } _ { i } \cos \alpha _ { i } ^ { K } + \overline { { B } } _ { i } \sin \alpha _ { i } ^ { K } .\tag{C9}
$$

Define the empirical amplitude $R _ { i } : = \sqrt { \overline { { A } } _ { i } ^ { 2 } + \overline { { B } } _ { i } ^ { 2 } }$ . By the amplitude-phase form of a sinusoid, there exists a phase $\varphi _ { i }$ such that

$$
\mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) h _ { i } ( x ; \alpha ) \right] = R _ { i } \cos \Big ( \alpha _ { i } ^ { K } + \varphi _ { i } \Big ) .\tag{C10}
$$

This implies that, for every value of $\alpha _ { i } ^ { K } .$

$$
| \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) h _ { i } ( x ; \alpha ) \right] | \le R _ { i } .\tag{C11}
$$

Evaluating Eq. (C11) at $\alpha = \alpha _ { t }$ we obtain:

$$
\begin{array} { r } { \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) h _ { i } ( x ; \alpha _ { t } ) \right] \leq | \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) h _ { i } ( x ; \alpha _ { t } ) \right] | \leq R _ { i } . } \end{array}\tag{C12}
$$

Therefore, $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } R _ { i } \geq 2 \gamma } \end{array}$ and, by Jensen’s inequality,

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \cal R } _ { i } ^ { 2 } \geq \left( { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \cal R } _ { i } \right) ^ { 2 } \geq 4 \gamma ^ { 2 } .\tag{C13}
$$

Equivalently, $\textstyle \sum _ { i = 1 } ^ { n } R _ { i } ^ { 2 } \geq 4 n \gamma ^ { 2 }$

Using Eq. (C9), the frozen empirical cost can be written as

$$
C _ { t } ( \alpha ^ { K } ) = \frac { 1 } { 2 } - \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } \left( \overline { { A } } _ { i } \cos \alpha _ { i } ^ { K } + \overline { { B } } _ { i } \sin \alpha _ { i } ^ { K } \right) ,\tag{C14}
$$

where all earlier-layer parameters are fixed at $\alpha _ { t } ^ { < K }$

The derivative of the frozen empirical cost with respect to the final-layer parameter $\alpha _ { i } ^ { K }$ can be computed using Eq. (C14)

$$
\frac { \partial C _ { t } ( \alpha ^ { K } ) } { \partial \alpha _ { i } ^ { K } } = \frac { 1 } { 2 n } \left( A _ { i } \sin \alpha _ { i } ^ { K } - B _ { i } \cos \alpha _ { i } ^ { K } \right) .\tag{C15}
$$

Following the standard barren-plateau analysis, we evaluate the gradient variance with respect to independently and uniformly distributed final-layer angles $\mathbf { \dot { \alpha } } _ { i } ^ { K } \sim \mathbf { U n i f } ( [ 0 , 2 \pi ] )$ , while keeping all other parameters fixed. Since

$$
\mathbb { E } _ { \alpha _ { i } ^ { K } } \left[ \frac { \partial C _ { t } } { \partial \alpha _ { i } ^ { K } } \right] = 0 ,\tag{C16}
$$

we obtain

$$
\begin{array} { r l } { \operatorname { V a r } _ { \alpha _ { i } ^ { K } } \displaystyle \left( \frac { \partial C _ { t } } { \partial \alpha _ { i } ^ { K } } \right) = \mathbb { E } _ { \alpha _ { i } ^ { K } } \left[ \left( \frac { \partial C _ { t } } { \partial \alpha _ { i } ^ { K } } \right) ^ { 2 } \right] } & { } \\ { = \displaystyle \frac { 1 } { 4 n ^ { 2 } } \mathbb { E } _ { \alpha _ { i } ^ { K } } \left[ \left( A _ { i } \sin \alpha _ { i } ^ { K } - B _ { i } \cos \alpha _ { i } ^ { K } \right) ^ { 2 } \right] } & { } \\ { = \displaystyle \frac { A _ { i } ^ { 2 } + B _ { i } ^ { 2 } } { 8 n ^ { 2 } } = \frac { R _ { i } ^ { 2 } } { 8 n ^ { 2 } } . } \end{array}\tag{C17}
$$

Summing over the n final-layer parameters and using $\operatorname { E q . }$ (C13), we therefore find

$$
\sum _ { i = 1 } ^ { n } \operatorname { V a r } _ { \alpha _ { i } ^ { K } } \left( \frac { \partial C _ { t } } { \partial \alpha _ { i } ^ { K } } \right) = \frac { 1 } { 8 n ^ { 2 } } \sum _ { i = 1 } ^ { n } R _ { i } ^ { 2 } \geq \frac { \gamma ^ { 2 } } { 2 n } .\tag{C18}
$$

Finally, let M denote the total number of trainable parameters in the circuit. Since the variance associated with every parameter direction is nonnegative, the average variance over all trainable parameters is lower bounded by the contribution of the n final-layer parameters alone:

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { M } \sum _ { \mu = 1 } ^ { M } \operatorname { V a r } _ { \alpha _ { \mu } } \left( \frac { \partial C _ { t } } { \partial \alpha _ { \mu } } \right) \geq \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { n } \operatorname { V a r } _ { \alpha _ { i } ^ { K } } \left( \frac { \partial C _ { t } } { \partial \alpha _ { i } ^ { K } } \right) } & { } \\ { \geq \displaystyle \frac { \gamma ^ { 2 } } { 2 M n } . } \end{array}\tag{C19}
$$

This proves $\operatorname { E q . } \left( \mathbf { C 3 } \right)$

If the ansatz has depth K and $O ( n )$ trainable parameters per layer, then $M = { \cal { O } } ( K n )$ . Hence, there exists a constant $c > 0$ such that

$$
{ \frac { 1 } { M } } \sum _ { \mu = 1 } ^ { M } \mathrm { V a r } _ { \alpha _ { \mu } } \left( { \frac { \partial C _ { t } } { \partial \alpha _ { \mu } } } \right) \geq { \frac { c \gamma ^ { 2 } } { K n ^ { 2 } } } .\tag{C20}
$$

We now show that the results in Theorem 9 are enough to guarantee that a non-trivial fraction of parameters in the model have gradient variance that remains inverse-polynomial in the system size, and therefore avoid barren plateaus.

Corollary 10 (Non-negligible fraction of parameters avoids barren plateaus.). Under the assumptions of Theorem 9, let

$$
S _ { \gamma } : = \left\{ i \in \{ 1 , \dots , n \} : \operatorname { V a r } _ { \alpha _ { i } ^ { K } } \left( { \frac { \partial C _ { t } } { \partial \alpha _ { i } ^ { K } } } \right) \geq { \frac { \gamma ^ { 2 } } { 4 n ^ { 2 } } } \right\} .\tag{C21}
$$

Then

$$
| S _ { \gamma } | \geq \frac { 2 \gamma ^ { 2 } } { 1 - 2 \gamma ^ { 2 } } n .\tag{C22}
$$

Hence, if the total number of trainable parameters satisfies $M = O ( K n )$ , an $\Omega ( \gamma ^ { 2 } / K )$ fraction of all trainable parameters has gradient variance at least $\dot { \gamma ^ { 2 } } / ( 4 n ^ { 2 } )$ . In particular, for $K = O ( 1 )$ number of layers and $\gamma = \Theta ( 1 )$ , a constant fraction of the trainable parameters has at least inverse-polynomial gradient variance.

Proof. From Eq. (C13) in Theorem 9, we have $\textstyle \sum _ { i = 1 } ^ { n } R _ { i } ^ { 2 } \geq 4 n \gamma ^ { 2 }$ . Moreover, $R _ { i } \leq 1$ for every i. Indeed, by Eq. (C10),

$$
R _ { i } = \operatorname* { m a x } _ { \alpha _ { i } ^ { K } } \left| \mathbb { E } _ { x \in \mathcal { T } } \left[ s ( x ) h _ { i } ( x ; \alpha ) \right] \right| ,\tag{C23}
$$

and since $s ( x ) \in \{ - 1 , + 1 \}$ and $h _ { i } ( \boldsymbol { x } ; \alpha ) = \mathrm { T r } [ \rho _ { t } U ( \boldsymbol { x } , \alpha ) ^ { \dagger } Z _ { i } U ( \boldsymbol { x } , \alpha ) ] \in [ - 1 , 1 ]$ , the absolute value of the expectation is at most one.

Define the set of indices i whose corresponding $R _ { i } ^ { 2 }$ is greater than $2 \gamma ^ { 2 }$

$$
\widetilde { S } _ { \gamma } : = \left\{ i \in \left\{ 1 , \dots , n \right\} : R _ { i } ^ { 2 } \geq 2 \gamma ^ { 2 } \right\} ,\tag{C24}
$$

and let $m : = | \widetilde { S } _ { \gamma } |$ . For the indices in $\widetilde { S } _ { \gamma }$ we use the bound $R _ { i } ^ { 2 } \leq 1$ derived above, while for the remaining $n - m$ indices we have $\hat { R } _ { i } ^ { 2 } < 2 \gamma ^ { 2 }$ . Therefore,

$$
\sum _ { i = 1 } ^ { n } R _ { i } ^ { 2 } \leq m + 2 \gamma ^ { 2 } ( n - m ) .\tag{C25}
$$

Combining this with Eq. (C13) gives $4 n \gamma ^ { 2 } \leq m + 2 \gamma ^ { 2 } { \left( n - m \right) }$ , and hence

$$
m ( 1 - 2 \gamma ^ { 2 } ) \geq 2 n \gamma ^ { 2 } .\tag{C26}
$$

Since the assumption $C _ { t } { \left( \alpha _ { t } \right) } \leq 1 / 2 - \gamma$ implies $0 < \gamma \leq 1 / 2$ , we have $1 - 2 \gamma ^ { 2 } > 0 ,$ , and thus the number of indices i whose $R _ { i } ^ { 2 }$ is greater than $2 \gamma ^ { 2 }$ is

$$
m \geq { \frac { 2 \gamma ^ { 2 } } { 1 - 2 \gamma ^ { 2 } } } n .\tag{C27}
$$

For every $i \in  { \widetilde { S } } _ { \gamma }$ , the same argument as in Theorem 9 allows us to bound the variance of the gradient for the parameter $\overset { \cdot } { \alpha _ { i } ^ { K } }$ using the quantity $R _ { i }$

$$
\operatorname { V a r } _ { \alpha _ { i } ^ { K } } \left( \frac { \partial C _ { t } } { \partial \alpha _ { i } ^ { K } } \right) = \frac { R _ { i } ^ { 2 } } { 8 n ^ { 2 } } \geq \frac { \gamma ^ { 2 } } { 4 n ^ { 2 } } .\tag{C28}
$$

Therefore $\widetilde { S } _ { \gamma } \subseteq S _ { \gamma } .$ , and hence

$$
| S _ { \gamma } | \geq | \widetilde { S } _ { \gamma } | \geq \frac { 2 \gamma ^ { 2 } } { 1 - 2 \gamma ^ { 2 } } n .\tag{C29}
$$

Finally, if $M = { \cal { O } } ( K n )$ , then

$$
\frac { | S _ { \gamma } | } { M } = \Omega \biggl ( \frac { \gamma ^ { 2 } } { K } \biggr ) .\tag{C30}
$$

Thus, for constant depth $K = O ( 1 )$ and $\gamma \geq 1 / \operatorname { p o l y } ( n )$ , an inverse-polynomial fraction of all trainable parameters has inverse-polynomial gradient variance. □

Our result concerns the cost evaluated for different α with the prepared ground state held fixed, as in line 21 of Algorithm 1. It shows that varying a non-negligible fraction of the parameters can produce cost differences of at least inverse-polynomial size. We note that, in general, the absence of barren plateaus alone does not guarantee successful training, as optimization may still encounter local minima or other stationary points.

## Appendix D: Universality of the Hamiltonian Embedding model

In this Appendix, we show that the Hamiltonian embedding model allows for universal construction. In particular, given a measurable target function $f : \Omega \subset \mathbb { R } ^ { d }  \{ 0 , 1 \}$ with inputs from $\Omega = [ - \pi L , \pi L ] _ { . } ^ { d }$ , where $L > 0$ is an arbitrary parameter determining the size of the input domain, we construct an HE model with a fixed product encoding $\check { U } ( \dot { \boldsymbol { x } } )$ whose ground-state classifier approximates f up to ε error under the uniform input distribution on Ω.

Theorem 11 (Universality with constant-depth encodings). Let $\Omega = [ - \pi L , \pi L ] ^ { d }$ be the input space, $\mu$ be the uniform distribution on Ω and let $\check { f } : \Omega \to \{ 0 , 1 \}$ be a measurable function. For every $\varepsilon , \delta \in ( \mathsf { 0 } , 1 )$ , there exist finite $n , N \in \mathbb { N }$ and a constant depth encoding $U ( x )$ such that given a training dataset $\mathcal { T } = \{ ( x _ { \ell } , \bar { f } ( x _ { \ell } ) ) \} _ { \ell = 1 } ^ { N }$ , with $x \sim \mu$ the following holds. Let $| \psi _ { T } \rangle$ be a ground state of

$$
H _ { \mathcal { T } } = I - \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } U ( x _ { \ell } ) ^ { \dagger } O ( f ( x _ { \ell } ) ) U ( x _ { \ell } ) ,\tag{D1}
$$

where $O ( y ) = ( I + ( - 1 ) ^ { y } A ) / 2$ with $\begin{array} { r } { A = n ^ { - 1 } \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ . Then, the HE classifier $\tilde { f } _ { \mathcal { T } }$ defined by the prediction rule in Eq. (5), using the fixed encoding ${ \dot { U } } ( x )$ and the ground state of H , satisfies with at least $1 - \delta$ probability over the training dataset,

$$
\operatorname* { P r } _ { x \sim \mu } [ { \tilde { f } } _ { T } ( x ) \neq f ( x ) ] \leq \varepsilon .\tag{D2}
$$

Proof. First, we write the target function as $s ( x ) = ( - 1 ) ^ { f ( x ) }$ and compute its Fourier coefficients $\begin{array} { r } { \widehat { s } _ { k } = \mathbb { E } _ { \mu } [ s ( x ) e ^ { - i k \cdot x / L } ] = } \end{array}$ $| \widehat { s } _ { k } | e ^ { i \phi _ { k } }$ , where $\mu$ is the uniform distribution over Ω. Since s is bounded and measurable, it belongs to $L ^ { 2 } ( \mu )$ . Completeness of the Fourier basis therefore ensures that, for every $ { \varepsilon } \in ( 0 , 1 )$ , a finite real Fourier sum p exists:

$$
p ( x ) = \sum _ { k \in K } a _ { k } \cos ( k \cdot x / L + \phi _ { k } ) ,\tag{D3}
$$

where $a _ { k } = 2 | \widehat { s } _ { k } |$ for $k \neq 0 , a _ { 0 } = | \widehat { s } _ { 0 } |$ for the constant term, and $\phi _ { k } = \mathrm { a r g } \widehat { s } _ { k }$ . Parseval’s identity then guarantees that with a finite K it holds that $\| s - p \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } \le \varepsilon / 4$ . Here K is the set of frequencies retained from the original Fourier decomposition of $s ,$ excluding those with zero Fourier coefficients and keeping one frequency from each nonzero pair $\{ k , - k \}$ . For example, we can take K to contain frequencies $k \in  { \mathbb { Z } ^ { d } }$ whose entries have absolute value at most a cutoff M, chosen such that $\| s - p \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } \le \varepsilon / 4$

We now construct the encoding U such that the HE model approximates the function p. Set $\begin{array} { r } { W = \sum _ { k \in K } a _ { k } } \end{array}$ . For each frequency $k \in K ,$ we choose integer multiplicities $m _ { k }$ such that $\begin{array} { r } { \sum _ { k \in K } m _ { k } = n } \end{array}$ and $| m _ { k } / n - a _ { k } / W | \leq 1 / n _ { \cdot }$ , where n is the number of qubits we use to encode the HE model. We then define the following HE model

$$
U ( x ) = \bigotimes _ { k \in { \cal K } } R _ { Y } ( k \cdot x / L ) ^ { \otimes m _ { k } } , \qquad A = \frac 1 n \sum _ { i = 1 } ^ { n } Z _ { i } , \qquad O ( y ) = \frac { I + ( - 1 ) ^ { y } { \cal A } } 2 .\tag{D4}
$$

Thus, each frequency $k \in K$ appearing in the truncated Fourier series p specifies a single-qubit rotation, and its Fourier magnitude determines the fraction of qubits encoding that frequency. The phases $\phi _ { k }$ will be assigned during training in the following way.

Let us consider a draw of independent inputs $x _ { \ell } \sim \mu$ and labeled by $y _ { \ell } = f ( x _ { \ell } )$ . Define the empirical correlations

$$
C _ { k } = \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } ( - 1 ) ^ { y _ { \ell } } e ^ { - i k \cdot x _ { \ell } / L } .\tag{D5}
$$

Notice that $C _ { k }$ approaches the Fourier coefficient $\widehat { s } _ { k }$ of the target s as $N \to \infty$ . The real and imaginary part of $C _ { k }$ are given by Euler’s formula:

$$
\operatorname { R e } ( C _ { k } ) = { \frac { 1 } { N } } \sum _ { \ell = 1 } ^ { N } ( - 1 ) ^ { y _ { \ell } } \cos ( k \cdot x _ { \ell } / L ) , \qquad \operatorname { I m } ( C _ { k } ) = - { \frac { 1 } { N } } \sum _ { \ell = 1 } ^ { N } ( - 1 ) ^ { y _ { \ell } } \sin ( k \cdot x _ { \ell } / L ) .\tag{D6}
$$

Since the encoding in Eq. (D4) acts independently on each qubit, the identity $R _ { Y } ( \theta ) ^ { \dagger } Z R _ { Y } ( \theta ) = \cos \theta Z$ sin θ X gives for each qubit i:

$$
U ( x ) ^ { \dagger } Z _ { i } U ( x ) = \cos ( k _ { i } \cdot x / L ) Z _ { i } - \sin ( k _ { i } \cdot x / L ) X _ { i } ,\tag{D7}
$$

where $k _ { i }$ is the frequency assigned to qubit i.

We can now write the embedded Hamiltonian in terms of $C _ { k }$ . Substituting these identities and the readout $O ( y )$ into the definition of H in Eq. (1) yields

$$
H _ { 7 } = I - { \frac { 1 } { N } } \sum _ { \ell = 1 } ^ { N } U ( x _ { \ell } ) ^ { \dagger } O ( y _ { \ell } ) U ( x _ { \ell } ) = { \frac { 1 } { 2 } } I - { \frac { 1 } { 2 n } } \sum _ { i = 1 } ^ { n } \left[ \mathrm { R e } ( C _ { k _ { i } } ) Z _ { i } + \mathrm { I m } ( C _ { k _ { i } } ) X _ { i } \right] .\tag{D8}
$$

By writing $C _ { k }$ in polar form, $C _ { k } = | C _ { k } | e ^ { i \theta _ { k } } .$ , it is easy to show that the ground state of the embedded Hamiltonian in Eq. (D8) is given by product state of one-qubit states $R _ { Y } ( \arg C _ { k _ { i } } ) | 0 \rangle$ , for the qubits encoding the frequency $k _ { i }$ . In fact, each one-qubit Hamiltonian term is real and symmetric in the computational basis, so a ground state can be chosen with real amplitudes. Since every normalized real qubit state can be written as $R _ { Y } ( \beta ) | 0 \rangle$ , it suffices to minimize the energy over this one-parameter family. Its Pauli expectations are $\left. Z \right. = \cos \beta$ and $\langle \dot { X } \rangle = \sin \beta ,$ so the corresponding energy term has expectation

$$
E _ { k } ( \beta ) = \frac { 1 } { 2 n } [ 1 - | C _ { k } | \left( \cos \theta _ { k } \cos \beta + \sin \theta _ { k } \sin \beta \right) ] = \frac { 1 } { 2 n } [ 1 - | C _ { k } | \cos ( \beta - \theta _ { k } ) ] .\tag{D9}
$$

This reaches the lowest eigenvalue, $E _ { k } = 1 / ( 2 n ) [ 1 - | C _ { k } | ]$ , when $\beta = \theta _ { k }$ . Therefore, the one-qubit ground state is $R _ { Y } ( \arg C _ { k } ) | 0 \rangle$ . Since the Hamiltonian terms act on separate qubits, their ground states can be combined into a product ground state. It is important to note that the empirical phase $\theta _ { k }$ is distinct from the target Fourier phase $\phi _ { k } = \mathrm { a r g } \widehat { s } _ { k }$ . However, as $C _ { k } \to { \widehat { s } } _ { k } .$ , the learned phase $\theta _ { k }$ approaches $\phi _ { k }$ modulo $2 \pi .$

We now show that the HE classifier obtained from the learned ground state approximates the target labels encoded by s. The full ground state is $| \psi _ { T } \rangle = \otimes _ { k \in { \cal K } } \left( R _ { Y } ( \arg C _ { k } ) | 0 \rangle \right) ^ { \otimes m _ { k } }$ , which implements the following prediction score function:

$$
g _ { \mathcal { T } } ( x ) : = \langle \psi _ { \mathcal { T } } | U ( x ) ^ { \dagger } A U ( x ) | \psi _ { \mathcal { T } } \rangle = \sum _ { k \in K } \frac { m _ { k } } { n } \cos ( k \cdot x / L + \arg C _ { k } ) .\tag{D10}
$$

Indeed, the encoding rotation and the ground-state rotation combine into $R _ { Y } ( k \cdot x / L + \arg C _ { k } )$ on each qubit, whose $Z$ expectation is the corresponding cosine. The model predicts label 0 when $g _ { T } ( x ) \geq 0$ and label 1 otherwise.

For fixed $n ,$ define the limiting score for $N \to \infty$

$$
g _ { \infty } ( x ) = \sum _ { k \in { \cal K } } \frac { m _ { k } } { n } \cos ( k \cdot x / L + \phi _ { k } ) .\tag{D11}
$$

Since each cosine has magnitude at most one,

$$
\| W g _ { \infty } - p \| _ { \infty } \leq \sum _ { k \in K } \left| \frac { W m _ { k } } { n } - a _ { k } \right| \leq \frac { W | K | } { n } .\tag{D12}
$$

We can then choose n such that this bound is at most $\sqrt { \varepsilon } / 4$ . Secondly, for a finite dataset, arg $C _ { k }$ will only be an approximation of $\phi _ { k } = \mathrm { a r g } \widehat { s } _ { k }$ . However, from the fact that $C _ { k } \to \widehat { s } _ { k }$ as $N \to \infty$ , we have that for any $\delta > 0$ , we can first choose n such that $W | K | / n \leq \sqrt { \varepsilon } / 4 ,$ , and then a finite N such that, with probability at least $1 - { \bar { \delta } } ,$

$$
\| W g _ { \mathcal { T } } - W g _ { \infty } \| _ { \infty } \leq \sqrt { \varepsilon } / 4 .\tag{D13}
$$

On this event, we can bound the average approximation error of the HE model with respect the target s as:

$$
\| s - W g _ { \mathcal { T } } \| _ { L ^ { 2 } ( \mu ) } \leq \| s - p \| _ { L ^ { 2 } ( \mu ) } + \| p - W g _ { \infty } \| _ { \infty } + \| W g _ { \infty } - W g _ { \mathcal { T } } \| _ { \infty }\tag{D14}
$$

$$
\leq { \frac { \sqrt { \varepsilon } } { 2 } } + { \frac { \sqrt { \varepsilon } } { 4 } } + { \frac { \sqrt { \varepsilon } } { 4 } } = { \sqrt { \varepsilon } } .\tag{D15}
$$

Since $W > 0$ , a mistaken prediction implies $| s ( x ) - W g _ { \mathcal { T } } ( x ) | \ge 1$ . Therefore, $\operatorname* { P r } _ { x \sim \mu } [ \tilde { f } _ { T } ( x ) \neq f ( x ) ] \leq \| s -$ $W g _ { T } \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } \leq \varepsilon .$

The construction guarantees universality with finite, target-dependent number of qubits n and training points N. We note that a polynomial scaling of n and N is not guaranteed for arbitrary target functions.

## Appendix E: Solving a classically hard problem with Hamiltonian embedding

## 1. The discrete cube root problem

We first recall the discrete cube root problem and the corresponding classical hardness assumption. Let p and q be two distinct large prime numbers of the form $3 k + 2$ and $3 k ^ { \prime } + 2 ,$ , respectively, and assume that they are represented by approximately the same number of bits. Let $N = p q$ be their product, which we take to be an n-bit integer, and denote by $\mathbb { Z } _ { N } = \mathbf { \bar { \{ 0 , \dots , } }  N - 1 \}$ the integers modulo N.

We consider the discrete cube function

$$
f _ { N } : \mathbb { Z } _ { N } \to \mathbb { Z } _ { N } , \qquad f _ { N } ( y ) = y ^ { 3 } \mod N .\tag{E1}
$$

For primes of the form specified above, this map is a bijection on $\mathbb { Z } _ { N } .$ , and therefore its inverse is well defined. We denote this inverse by

$$
g _ { N } ( x ) = f _ { N } ^ { - 1 } ( x ) = \sqrt [ 3 ] { x } \mod N .\tag{E2}
$$

Indeed, since $p = 3 k + 2$ and $q = 3 k ^ { \prime } + 2$ , one has gcd $( 3 , p - 1 ) = \operatorname* { g c d } ( 3 , q - 1 ) = 1$ . Hence cubing is invertible modulo both p and q, and therefore also modulo $N = p q$ by the Chinese remainder theorem.

Definition 12 (Discrete cube root problem). Let $p$ and q be two distinct primes of the form $3 k + 2$ and $3 k ^ { \prime } + 2 ,$ and let $N = p q$ be an n-bit integer. Given as input N and $x \in \mathbb { Z } _ { N }$ , the discrete cube root problem asks to output $y \in \mathbb { Z } _ { N }$ such that

$$
y ^ { 3 } = x \mod N .\tag{E3}
$$

The forward function $f _ { N }$ can be computed efficiently on a classical computer using modular exponentiation. By contrast, computing the inverse function $g _ { N }$ from only N and x is believed to be classically hard when the factorization of N is unknown. This is formalized by the following assumption, which is a special case of the RSA assumption with exponent $e = 3 .$

Definition 13 (Discrete cube root assumption). For any polynomial $P ( \cdot )$ , there is no classical algorithm A running in time $P ( n )$ such that, given as input an n-bit integer $\bar { N } = \dot { p } q .$ , where p and q are random primes of the form $3 k + \breve { 2 }$ and $3 k ^ { \prime } + \dot { 2 }$ , and a random $x \in \mathbb { Z } _ { N } ,$ , outputs a value y $\in \mathbb { Z } _ { N }$ satisfying

$$
y ^ { 3 } = x \mod N\tag{E4}
$$

with probability at least $1 / P ( n )$ . The probability is taken over the random choice of the primes $p , q ,$ , the input $x \in \mathbb { Z } _ { N } ,$ and the internal randomness of A.

The discrete cube function is also a trapdoor function. If the factors p and q of N are known, then one can compute Euler’s totient $\varphi ( N ) = ( p - 1 ) ( q - 1 )$ . Since $p = 3 k + 2$ and $q = 3 \dot { k ^ { \prime } } + 2$ , both $p - 1$ and $q - 1$ are congruent to 1 modulo $^ { 3 , }$ and therefore $\dot { \operatorname* { g c d } } ( \dot { 3 } , \dot { \varphi } ( N ) ) = 1$ . It follows that 3 has a multiplicative inverse modulo $\varphi ( N )$ . Hence, by Euclid’s algorithm, one can efficiently find an integer d such that $3 d = 1$ mod $\varphi ( N )$ , or equivalently $3 d = r \varphi ( N ) + \dot { 1 }$ for some integer r.

We now show why this value of d gives a secret key for inverting the discrete cube function. Let $x = f _ { N } ( y ) = y ^ { 3 }$ mod N for some $y \in \mathbb { Z } _ { N }$ . Then

$$
x ^ { d } = ( y ^ { 3 } ) ^ { d } = y ^ { 3 d } = y ^ { r \varphi ( N ) + 1 } \mod N .\tag{E5}
$$

Using $\varphi ( N ) = ( p - 1 ) ( q - 1 )$ ) together with Fermat’s little theorem modulo p and modulo q, one obtains $y ^ { r \varphi ( N ) + 1 } = y$ mod p and $y ^ { r \varphi ( N ) + 1 } = y$ mod q. Since p and q are distinct primes, congruence modulo both p and $q$ implies congruence modulo $N = p q$ . Therefore, $x ^ { d } = y$ mod N. Thus, for every $x \in \mathbb { Z } _ { N } .$ , the discrete cube root can be computed efficiently as $g _ { N } ( x ) = x ^ { d }$ mod N.

The integer d therefore acts as a trapdoor key: it can be computed efficiently from the factorization of $N ,$ but is believed to be inaccessible to efficient classical algorithms given only N. Hence, knowledge of the factorization makes the inverse efficiently computable, whereas without this trapdoor the problem is assumed to be classicall intractable.

## 2. Definition of the learning problem

Based on the DCR function in Eq. (E2), we define the following function that classifies the input $x \in \mathbb { Z } _ { N }$ of our learning problem.

For a fixed shift parameter $s \in \mathbb { Z } _ { N }$ , define $g _ { N , s } : \mathbb { Z } _ { N } \to \{ 0 , 1 \}$ , by

$$
y ( x ) = g _ { N , s } ( x ) = \left\{ \begin{array} { l l } { { 1 , } } & { { \mathrm { i f } g _ { N } ( x ) \in \left[ s , s + { \frac { N - 1 } { 2 } } \right] } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right.\tag{E6}
$$

where, as in Eq. 48, the interval is interpreted cyclically in $\mathbb { Z } _ { N }$

Equivalently, using the key d as in Eq. (E5),

$$
g _ { N , s } ^ { d } ( x ) = \mathbf { 1 } \left[ x ^ { d } { \bmod { N } } \in \left[ s , s + { \frac { N - 1 } { 2 } } \right] \right] .\tag{E7}
$$

Based on the labeling function above, we define the following learning problem. Let N be a large semiprime of the form $N = p q$ as above, we formally define the concept class $\mathcal { C } _ { N }$ as the family of all concepts function $g _ { N , s }$ in Eq. (E6):

$$
\mathcal { C } _ { N } = \{ g _ { N , s } \ | \ s \in \mathbb { Z } _ { N } \} .\tag{E8}
$$

Then, the learner is given training data labeled by one of the concept in $\mathcal { C } _ { N }$

$$
\mathcal { T } = \{ ( x _ { \ell } , y _ { \ell } ) \} _ { \ell = 1 } ^ { m } , \qquad x _ { \ell } \in \mathbb { Z } _ { N } , \qquad y _ { \ell } = g _ { N , s ^ { \star } } ^ { d ^ { \star } } ( x _ { \ell } ) \in \{ 0 , 1 \} ,\tag{E9}
$$

for some hidden target parameters $( d ^ { \star } , s ^ { \star } )$ , where $g _ { N , s } ^ { d }$ denotes the representation of $g _ { N , s }$ associated with the secret key d. Importantly, the learner does not know the true $( d ^ { \star } , s ^ { \star } )$ , it can only retrieve information about them through the training data. We also assume that the learner has knowledge of N from the specification of the concept class $\breve { \mathcal { C } } _ { N }$ or from the training data (for example we could trivially append it to all the training points $x _ { \ell } \in \mathbb { Z } _ { N }$ as $\left( x _ { \ell } , N , y _ { \ell } \right) )$ . The goal of the learning task is to implement a learning algorithm A that meets the following learning condition.

Definition 14 (Efficient PAC learning). Let $\mathcal { C } _ { N }$ be the DCR concept class. We say that $\mathcal { C } _ { N }$ is efficiently classically learnable if there exists a classical learning algorithm A with the following property. For every target concept $g _ { N , s } \in$ ${ \mathcal { C } } _ { N } ,$ , every probability distribution  over $\mathbb { Z } _ { N } ,$ and every $\varepsilon , \delta > 0 \nonumber$ , given a training set $\mathcal { T } = \{ \mathsf { \bar { \rho } } ( x _ { \ell } , g _ { N , s } ( x _ { \ell } ) ) \} _ { \ell = 1 } ^ { - m }$ with $x _ { \ell } \sim \mathcal { D }$ drawn independently, the algorithm outputs a hypothesis ${ \check { h } } : \mathbb { Z } _ { N } \to \{ 0 , 1 \} $ such that

$$
\Pr _ { x \sim \mathcal { D } } \left[ h ( x ) \neq g _ { N , s } ( x ) \right] \leq \varepsilon\tag{E10}
$$

with probability at least $1 - \delta$ over the random draw of the training set and the internal randomness of A. Moreover, A runs in time polynomial in $n , 1 / \varepsilon ,$ and $1 / \delta$

## 3. Classical hardness of the learning problem

The classical hardness result for the DCR learning task relies on the result of [ACGS88]. In particular, they show that even a weak predictor for suitable bits of the discrete cube root is sufficient to recover the full inverse function. Based on this, $[ [ \dot { \mathrm { G M } } ^ { + } 2 4 ]$ showed that no efficient classical learner can achieve accuracy significantly better than random guessing on the concept class defined from the discrete cube root function. We first recall the notion of an $\epsilon ( n )$ -oracle.

Definition 15 $( \epsilon ( n ) { \mathrm { - o r a c l e } } )$ . Let $O _ { N , s }$ be a probabilistic oracle such that, on input $x \in \mathbb { Z } _ { N } .$ , it outputs the label $g _ { N , s } ( x )$ correctly with probability $1 / 2 + \epsilon ( \dot { n } )$ over the random choice of x and the internal randomness of the oracle. We say that $O _ { N , s }$ is an ϵ(n)-oracle for $g _ { N , s }$

The result of Alexi et al. [ACGS88] implies that a weak oracle with inverse-polynomial advantage over random guessing is already sufficient to break the discrete cube root assumption.

Lemma 16 (Corollary (a) to Theorem 1 in [ACGS88]). For any concept $g _ { N , s }$ in the DCR concept class, given access to a $1 / \mathrm { p o l y } ( n )$ -oracle $O _ { N , s } f o r \ g _ { N , s } ,$ there exists a polynomial-time classical algorithm that uses $O _ { N , s }$ to compute the discrete cube root $g _ { N } ( x ) = \sqrt [ 3 ] { x }$ mod N for every $x \in \mathbb { Z } _ { N }$ with constant success probability.

Using this result, the authors in $[ [ \mathrm { G M } ^ { + } 2 4 ]$ obtain the following classical hardness statement for the corresponding learning problem.

Lemma 17 (Classical hardness of the DCR learning task). Assume the discrete cube root assumption. Then no polynomialtime classical learning algorithm can, with constant success probability over the random generation of the training data and its internal randomness, output a hypothesis h whose expected error on the DCR concept $g _ { N , s }$ is smaller than random guessing by an inverse-polynomial amount. More precisely,for the uniform distribution over $\mathbb { Z } _ { N }$ , no efficient classical learner can achieve

$$
\operatorname* { P r } _ { x \sim U ( \mathbb { Z } _ { N } ) } \left[ h ( x ) \neq g _ { N , s } ( x ) \right] \leq \frac { 1 } { 2 } - \frac { 1 } { \mathrm { p o l y } ( n ) } .\tag{E11}
$$

Indeed, if a classical learner satisfying the learning condition in Definition 14 existed, then the hypothesis it outputs would define a $1 / \mathrm { p o l y } ( n )$ -oracle for the label function $g _ { N , s }$ . By Lemma 16, this oracle could then be used to compute the full discrete cube root function $g _ { N } ( x )$ in polynomial time with constant success probability. This would contradict the discrete cube root assumption. Therefore, under this assumption, the DCR learning task cannot be solved efficiently by classical learning algorithms with non-negligible advantage over random guessing.

## 4. Quantum solution with Hamiltonian embedding

We now show how this problem can be encoded within the Hamiltonian-embedding framework. In particular, we construct an embedding Hamiltonian of the form $\begin{array} { r } { { H _ { T } } = \sum _ { \ell } H ( x _ { \ell } ) } \end{array}$ whose ground state can be used to correctly classify new inputs, in the same way as described in Section II. This allows us to show in the following lemma the quantum learnability of the DCR task within the Hamiltonian embedding model. Together with Lemma 17, this formally establishes the result anticipated in the main text as Theorem 4.

Lemma 18. Let $C _ { N }$ be the DCR concept class, and let $\mathcal { T } ~ = ~ \{ ( x _ { \ell } , y _ { \ell } ) \} _ { \ell = 1 } ^ { m }$ be the training data associated to a concept $g _ { N , s ^ { \star } } ^ { d ^ { \star } } \in C _ { N }$ with $x _ { \ell }$ sampled independently from the uniform distribution $\mathcal { U } ( \mathbb { Z } _ { N } )$ . For every $\epsilon , \delta > 0 ,$ , there exists $m =$ poly $\big ( n , 1 / \epsilon , \log ( 1 / \delta ) \big )$ , an encoding unitary $U ( x )$ , and a local label observable $O ( y )$ , such that, with probability at least $1 - \delta$ over the draw $o f \mathcal { T }$ , the ground state of the corresponding embedding Hamiltonian H induces a hypothesis $\tilde { f } : \overline { { \mathbb { Z } _ { N } } } \to \{ 0 , 1 \}$ as in Eq. (5) satisfying

$$
\begin{array} { r } { \operatorname* { P r } _ { x \sim \mathcal { U } ( \mathbb { Z } _ { N } ) } \Big [ \tilde { f } ( x ) \neq g _ { N , s ^ { \star } } ^ { d ^ { \star } } ( x ) \Big ] \leq \epsilon . } \end{array}\tag{E12}
$$

Moreover, such a ground state can be prepared efficiently by a quantum algorithm, and once obtained, the induced hypothesis $\tilde { f }$ can be efficiently evaluated classically.

Proof. The embedded Hamiltonian is defined on the Hilbert space $\mathcal { H } = \mathfrak { s p a n } \{ | d , s \rangle : d , s \in \mathbb { Z } _ { N } \} \otimes \mathbb { C } ^ { 2 }$ , where the last qubit serves as an ancilla with computational basis $\{ | 0 \rangle , | 1 \rangle \}$

For each input $x \in \mathbb { Z } _ { N } .$ , define the unitary

$$
U ( x ) \left| d , s \right. \left| b \right. = \left| d , s \right. \left| b \oplus g _ { N , s } ^ { d } ( x ) \right. .\tag{E13}
$$

We note that $g _ { N , s } ^ { d } ( x )$ can be evaluated efficiently classically, since its computation only requires modular exponentiation and a classical comparison. Hence, there exists a polynomial-size classical circuit computing $g _ { N , s } ^ { d } ( x )$ from $x , d ,$ and s. By reversibly implementing this computation, with at most polynomial overhead, one obtains a polynomialsize quantum circuit realizing the unitary $\grave { U ( x ) }$ . Define now the observable

$$
O ( y ) = I \otimes | y \rangle \langle y | , \qquad y \in \{ 0 , 1 \} .\tag{E14}
$$

We can now define the embedding Hamiltonian $H _ { \mathcal { T } }$ for the training set $\tau$ . For one training example $\left( x _ { \ell } , y _ { \ell } \right)$ define the Hamiltonian term

$$
H ( x _ { \ell } , y _ { \ell } ) = I - U ( x _ { \ell } ) ^ { \dagger } O ( y _ { \ell } ) U ( x _ { \ell } ) + I \otimes | 1 \rangle \langle 1 | .\tag{E15}
$$

The dataset Hamiltonian $H _ { \mathcal { T } }$ is then:

$$
H _ { \mathcal { T } } = \sum _ { \ell = 1 } ^ { m } H ( x _ { \ell } , y _ { \ell } ) .\tag{E16}
$$

We now show that the ground state of $H _ { \mathcal { T } }$ can be used to correctly predict the labels of unseen inputs. In particular, in analogy to Eq. (5), for a new point x the predicted label is the value of y that maximizes $\mathrm { T r } \big [ O ( y ) \overleftarrow { U } ( x ) \vert d ^ { \star } , s ^ { \star } , 0 \rangle \langle d ^ { \star } , s ^ { \star } , 0 \vert ^ { } U ( x ) ^ { \dagger } \big ]$ , with $| d ^ { \star } , s ^ { \star } , 0 \rangle$ ground state of $H _ { \mathcal { T } }$

We first show that the energy of each term $H ( x _ { \ell } , y _ { \ell } )$ is minimized by states of the form $| d , s , 0 \rangle$ with $( d , s )$ such that $g _ { N , s } ^ { d } ( x _ { \ell } ) = y _ { \ell }$ . Then, since the training labels are generated by the target concept $y _ { \ell } = g _ { N , s ^ { \star } } ^ { d ^ { \star } } ( x _ { \ell } )$ , the state $| d ^ { \star } , s ^ { \star } , 0 \rangle$ associated with the true parameters $d ^ { \star }$ and $s ^ { \star }$ minimizes every term in $H _ { T }$ and thus has zero energy.

Let us consider each term $H ( x _ { \ell } , y _ { \ell } )$ in Eq. (E15). We can expand $U ( x _ { \ell } ) ^ { \dagger } O ( y _ { \ell } ) U ( x _ { \ell } )$ as follows:

$$
U ( x _ { \ell } ) ^ { \dagger } O ( y _ { \ell } ) U ( x _ { \ell } ) = \sum _ { \substack { ( d , s ) : g _ { N , s } ^ { d } ( x _ { \ell } ) = y _ { \ell } } } | d , s \rangle \langle d , s | \otimes | 0 \rangle \langle 0 | + \sum _ { \substack { ( d , s ) : g _ { N , s } ^ { d } ( x _ { \ell } ) \neq y _ { \ell } } } | d , s \rangle \langle d , s | \otimes | 1 \rangle \langle 1 | .\tag{E17}
$$

Eq. (E17) follows from computing the action of $U ( x _ { \ell } ) ^ { \dagger } O ( y _ { \ell } ) U ( x _ { \ell } )$ on generic states $| d , s , b \rangle$ using the definitions of $U ( x )$ and $O ( y )$ in Eqs. (E13) and (E14):

$$
U ^ { \dagger } ( x _ { \ell } ) { \cal O } ( y _ { \ell } ) U ( x _ { \ell } ) \left| d , s , b \right. = \delta _ { y _ { \ell } , b \oplus g _ { N , s } ^ { d } ( x _ { \ell } ) } \left| d , s , b \right.\tag{E18}
$$

Therefore, by rewriting the Hamiltonian $H ( x _ { \ell } , y _ { \ell } )$ in Eq. (E15) using Eq. (E17) we obtain:

$$
H ( x _ { \ell } , y _ { \ell } ) = I - \sum _ { ( d , s ) : g _ { N , s } ^ { d } ( x _ { \ell } ) = y _ { \ell } } | d , s  \langle d , s | \otimes | 0 \rangle \langle 0 | - \sum _ { ( d , s ) : g _ { N , s } ^ { d } ( x _ { \ell } ) \neq y _ { \ell } } | d , s  \langle d , s | \otimes | 1 \rangle \langle 1 | + I \otimes | 1 \rangle \langle 1 |\tag{E19}
$$

We now evaluate the energy of a generic basis state $| d , s , b \rangle$ to determine which states minimize the Hamiltonian terms $H ( x _ { \ell } , y _ { \ell } )$

${ \mathrm { I f ~ } } b = 1 , { \mathrm { i . e . ~ } } | \psi \rangle = | d , s , 1 \rangle$ , then

$$
\langle \psi | H ( x _ { \ell } , y _ { \ell } ) | \psi \rangle = \left\{ { \begin{array} { l l } { 2 , } & { { \mathrm { i f ~ } } g _ { N , s } ^ { d } ( x _ { \ell } ) = y _ { \ell } , } \\ { 1 , } & { { \mathrm { i f ~ } } g _ { N , s } ^ { d } ( x _ { \ell } ) \neq y _ { \ell } . } \end{array} } \right.\tag{E20}
$$

${ \mathrm { I f ~ } } b = 0 , { \mathrm { i . e . ~ } } | \psi \rangle = | d , s , 0 \rangle$ , then

$$
\langle \psi | H ( x _ { \ell } , y _ { \ell } ) | \psi \rangle = \left\{ { 0 } , \ \mathrm { i f } \ g _ { N , s } ^ { d } ( x _ { \ell } ) = y _ { \ell } , \right.\tag{E21}
$$

Hence, each term $H ( x _ { \ell } , y _ { \ell } )$ in the dataset Hamiltonian $H _ { \mathcal { T } }$ is minimized precisely by states of the form $| d , s , 0 \rangle$ satisfying $g _ { N , s } ^ { d } ( x _ { \ell } ) = y _ { \ell }$

Since the total Hamiltonian is given by $\begin{array} { r } { H _ { \mathcal { T } } = \sum _ { \ell = 1 } ^ { m } H ( x _ { \ell } , y _ { \ell } ) } \end{array}$ , then for a generic basis state $| d , s , 0 \rangle$ one has

$$
\langle d , s , 0 | H _ { \mathcal { T } } | d , s , 0 \rangle = \sum _ { \ell = 1 } ^ { m } { \bf 1 } \Big [ g _ { N , s } ^ { d } ( x _ { \ell } ) \neq y _ { \ell } \Big ] .\tag{E22}
$$

Therefore, the energy of a state $| d , s , 0 \rangle$ is exactly the number of training examples misclassified by the hypothesis specified by $( d , s )$

$$
\langle d , s , 0 | H _ { T } | d , s , 0 \rangle = \Big | \{ \mathrm { t r a i n i n g ~ p o i n t s ~ m i s c l a s s i f i e d ~ b y ~ } g _ { N , s } ^ { d } \} \Big | .\tag{E23}
$$

It follows that a basis state $| d , s , 0 \rangle$ has zero energy if and only if the corresponding hypothesis $g _ { N , s } ^ { d }$ correctly classifies every training example. Since the target hypothesis $g _ { N , s ^ { \star } } ^ { d ^ { \star } }$ is consistent with the training data, the ground energy of $H _ { \mathcal { T } }$ is zero, and its ground space is spanned by the states $| d , s , 0 \rangle$ corresponding to hypotheses that are consistent with the training set.

We note that the exponent $d ^ { \star }$ can be recovered efficiently by a quantum algorithm by factoring N [Sho97]. Once $d ^ { \star }$ is known, the remaining parameter s can be learned efficiently from the training set by first computing the DCR of each input: $x _ { \ell } \longmapsto x _ { \ell } ^ { d ^ { \star } }$ mod $N = \sqrt [ 3 ] { x _ { \ell } }$ mod $N = g _ { N } ( x _ { \ell } )$ , with $g _ { N }$ the DCR function defined in Eq. (E2). The transformed dataset $\mathcal { \widetilde { T } } = \{ ( g _ { N } ( x _ { \ell } ) , y _ { \ell } ) \} _ { \ell = 1 } ^ { m }$ then reduces the problem to learning the shift of a one-dimensional interval classifier. Since the data are realizable, a shift $s ^ { \prime }$ that is consistent with all the training examples can be found efficiently. While several shifts may be consistent with a polynomial-size training set, sufficiently many training samples guarantee that a suitable consistent shift $s ^ { \prime }$ has small population error under the uniform distribution ${ \check { \mathcal { U } } } ( \mathbb { Z } _ { N } )$ In particular, [JGM<sup>+</sup>24] shows that, with $O ( \log ( 1 / \delta ) / \epsilon ^ { \prime } )$ uniformly sampled examples, one can identify a shift $\overset { \cdot } { s } { } ^ { \prime }$ satisfying $\begin{array} { r } { | s ^ { \star } - s ^ { \prime } | \leq \epsilon ^ { \prime } N } \end{array}$ with probability at least $1 - \delta ,$ where $| s ^ { \star } - s ^ { \prime } |$ denotes the cyclic distance between the two shifts in $\mathbb { Z } _ { N }$ . Let us fix $\epsilon ^ { \prime } = \epsilon / \hat { 2 }$ . This in turn guarantees

$$
\operatorname* { P r } _ { x \sim \mathcal { U } ( \mathbb { Z } _ { N } ) } \bigl [ g _ { N , s ^ { \prime } } ( x ) \neq g _ { N , s ^ { \star } } ( x ) \bigr ] \leq \epsilon .\tag{E24}
$$

Therefore, the quantum learning algorithm finds the state $| d ^ { \star } , s ^ { \prime } , 0 \rangle$ which has zero energy with respect to the data Hamiltonian $\bar { H _ { T } }$

Once a ground state $| d ^ { \star } , s ^ { \prime } , 0 \rangle$ is obtained, a new point x is classified by the same labeling rule as in Eq. (5). In particular, the probability associated to the label y for an input x is given by

$$
P _ { d ^ { \star } , s ^ { \prime } } ( x , y ) = \mathrm { T r } \Big [ O ( y ) U ( x ) | d ^ { \star } , s ^ { \prime } , 0 \rangle \langle d ^ { \star } , s ^ { \prime } , 0 | U ( x ) ^ { \dagger } \Big ] .\tag{E25}
$$

Since, by the construction of $U ( x )$ in Eq. (E13), $U ( x ) | d ^ { \star } , s ^ { \prime } , 0 \rangle = | d ^ { \star } , s ^ { \prime } , g _ { N , s ^ { \prime } } ^ { d ^ { \star } } ( x ) \rangle$ , it follows that

$$
P _ { d ^ { \star } , s ^ { \prime } } ( x , y ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } g _ { N , s ^ { \prime } } ^ { d ^ { \star } } ( x ) = y , } \\ { 0 , } & { \mathrm { i f ~ } g _ { N , s ^ { \prime } } ^ { d ^ { \star } } ( x ) \neq y . } \end{array} \right.\tag{E26}
$$

Hence, the prediction rule induced by the ground state $| d ^ { \star } , s ^ { \prime } , 0 \rangle$ is precisely the hypothesis $g _ { N , s ^ { \prime } } ^ { d ^ { \star } }$ . By Eq. (E24), this hypothesis has population error at most ϵ under $\mathcal { U } ( \mathbb { Z } _ { N } )$ , which proves the lemma.

Finally, we note that, once the state $| d ^ { \star } , s ^ { \prime } , 0 \rangle$ has been obtained quantumly, the prediction probability $P _ { d ^ { \star } , s ^ { \prime } } ( x , y )$ in Eq. (E25) can be evaluated entirely classically, since $U ( x )$ acting on $| d ^ { \star } , s ^ { \prime } , \bar { 0 } \rangle$ is efficiently computable by directly evaluating $g _ { N , s ^ { \prime } } ^ { d ^ { \star } } ( x )$ □

In Lemma 18, we constructed an embedding Hamiltonian $H _ { \mathcal { T } }$ whose ground space contains a state $| d ^ { \star } , s ^ { \prime } , 0 \rangle$ associated with a hypothesis $g _ { N , s ^ { \prime } } ^ { d ^ { \star } }$ that solves the DCR learning task, and we showed that such a state can be found efficiently by a quantum algorithm. In the following lemma, we show that, more generally, every ground state of $H _ { \mathcal { T } }$ induces a hypothesis with small population error on the DCR task.

Lemma 19. Let $\mathcal { H } _ { N } = \left\{ g _ { N , s } ^ { d } : d , s \in \mathbb { Z } _ { N } \right\}$ , be the hypothesis class of functions $g _ { N , s } ^ { d }$ as in Eq. (E7). Given the training set $\mathcal { T } = \{ ( x _ { \ell } , y _ { \ell } ) \} _ { \ell = 1 } ^ { m } ,$ with $y _ { \ell } = g _ { N , s ^ { \star } } ^ { d ^ { \star } } ( x _ { \ell } )$ , generated by drawing the inputs independently from the uniform distribution over $\mathbb { Z } _ { N } ,$ , let $H _ { \mathcal { T } }$ be the embedding Hamiltonian as in Eq. (E16).

For any $\epsilon , \delta > 0 ,$ , if

$$
m \geq \frac { 1 } { \epsilon } \left( 2 \log N + \log \frac { 1 } { \delta } \right) ,
$$

then, with probability at least $1 - \delta$ over the draw of the training set, every zero-energy basis state $| d , s , 0 \rangle$ of $H _ { \mathcal { T } }$ induces a hypothesis $g _ { N , s } ^ { d }$ satisfying

$$
\begin{array} { r } { \operatorname* { P r } _ { x \sim U ( \mathbb { Z } _ { N } ) } \left[ g _ { N , s } ^ { d } ( x ) \neq g _ { N , s ^ { \star } } ^ { d ^ { \star } } ( x ) \right] \leq \epsilon . } \end{array}
$$

In particular, the conclusion holds independently of whether $d = d ^ { \star }$

Proof. By Eq. (E23), for every basis state $| d , s , 0 \rangle$

$$
\langle d , s , 0 | H _ { \mathcal { T } } | d , s , 0 \rangle = \sum _ { \ell = 1 } ^ { m } { \bf 1 } \left[ g _ { N , s } ^ { d } ( x _ { \ell } ) \neq y _ { \ell } \right] .\tag{E27}
$$

Since the target hypothesis $g _ { N , s ^ { \star } } ^ { d ^ { \star } }$ is consistent with the training data, the ground energy of $H _ { \mathcal { T } }$ is zero. Therefore, any zero-energy basis state $| d , s , 0 \rangle$ corresponds to a hypothesis $g _ { N , s } ^ { d }$ that is consistent with all training examples.

Now consider a fixed hypothesis $h \in \mathcal { H } _ { N }$ whose population error with respect to the target hypothesis is larger than ϵ, namely

$$
R ( h ) : = \operatorname* { P r } _ { x \sim U ( \mathbb { Z } _ { N } ) } \left[ h ( x ) \neq g _ { N , s ^ { \star } } ^ { d ^ { \star } } ( x ) \right] > \epsilon .\tag{E28}
$$

The probability that h nevertheless agrees with all m independently drawn training examples is

$$
( 1 - R ( h ) ) ^ { m } \leq ( 1 - \epsilon ) ^ { m } \leq e ^ { - m \epsilon } .\tag{E29}
$$

Since $d , s \in \mathbb { Z } _ { N }$ , the hypothesis class satisfies $\vert \mathcal { H } _ { N } \vert \le N ^ { 2 }$

Hence, by a union bound,

$$
\operatorname* { P r } _ { \mathcal T } \Big [ \exists h \in \mathcal H _ { N } : R ( h ) > \epsilon \mathrm { ~ a n d ~ } \widehat { R } _ { \mathcal T } ( h ) = 0 \Big ] \leq N ^ { 2 } e ^ { - m \epsilon } .\tag{E30}
$$

If

$$
m \geq \frac { 1 } { \epsilon } \left( 2 \log N + \log \frac { 1 } { \delta } \right) ,\tag{E31}
$$

then the right-hand side is at most $\delta .$ Therefore, with probability at least $1 - \delta ,$ every hypothesis in $\mathcal { H } _ { N }$ that is consistent with the training set has population error at most $\epsilon .$

Since every zero-energy basis state of $H _ { \mathcal { T } }$ corresponds to such a consistent hypothesis, the claimed result follows.

We note that the same argument can be extended to arbitrary ground states that are superpositions of zero-energy basis states. Indeed, for a ground state $\begin{array} { r } { \left| \psi \right. = \sum _ { d , s } c _ { d , s } \left| d , s , 0 \right. } \end{array}$ , the prediction probabilities are convex combinations of those associated with the individual hypotheses $g _ { N , s } ^ { d } ,$ since the states $| d , s \rangle$ are mutually orthogonal. Consequently, if every zero-energy basis hypothesis has population error at most $\epsilon / 2 ,$ the classifier induced by $\left| \psi \right.$ has population error at most $\epsilon ,$ since a misclassification can occur only when at least half of the total weight is assigned to basis hypotheses that err on that input.

We note that this example establishes the existence of a hard task that can be solved within the Hamiltonian embedding framework, for which a quantum algorithm can efficiently prepare the ground state of the embedded Hamiltonian, while doing so classically is believed to be hard. This is sufficient to demonstrate an advantage for that specific task. For more general learning problems, however, we do not expect to be able to design a problem-specific algorithm for preparing the ground state of the embedded Hamiltonian. Instead, one would likely rely on more generic quantum ground-state preparation methods, such as QPE or SKQD.

Finally, for the ground state $| \bigstar , \dot { s } ^ { \prime } , 0 \rangle$ constructed by the quantum learning algorithm, inference can be performed entirely classically. Indeed, once the parameters $d ^ { \star }$ and $s ^ { \prime }$ are known, the corresponding hypothesis $g _ { N , s ^ { \prime } } ^ { \hat { d } ^ { \star } } ( x )$ can be evaluated efficiently using classical modular exponentiation and a classical comparison. This differs from the settings considered in Section III B, where the classical surrogates are constructed using task-agnostic procedures.

## Appendix F: A construction with highly entangled ground states

In Section II A 1, we discussed that the Hamiltonian embedding framework allows for different choices of the Her mitian operator A entering the observable $O ( y )$ in Eq. (9). The choice of the label observable $O ( y )$ can significantly affect the performance of the model, and different operators may be more suitable depending on the learning task under consideration. Although identifying which choices of $O ( y )$ are best suited to a given learning problem is itself an interesting question, in this Appendix and Appendix G we show that the choice of $O ( y )$ can influence properties of the ground state $\psi _ { T , \alpha ^ { \star } }$ found by the Hamiltonian embedding model, such as its degree of entanglement. In this Appendix, we provide a simple example of a label dependent observable $O ( y )$ for which the ground states obtained within the Hamiltonian embedding framework exhibit a high degree of entanglement. We note that, in this partic ular construction, the resulting states remain efficiently classically representable, as they are derived from Clifford states. More elaborate choices of $O ( y )$ could in principle give rise to ground states with more complex structure. The purpose of this example is therefore not to establish classical hardness results as we did in Section III C, but rather to illustrate how the choice of $O ( y )$ can directly influence the properties of the ground state found by the model.

## 1. Entangled graph states as ground states

In order to obtain entangled ground states, we start from the parent Hamiltonian of a graph state, which can exhibit a large amount of bipartite entanglement. Specifically, for n qubits, we consider a graph composed of two cycles of $n / \overset { \vartriangle } { \boldsymbol { 2 } }$ vertices each, connected by a perfect matching between corresponding vertices. To each vertex a we associate the graph-state stabilizer

$$
K _ { a } = X _ { a } \prod _ { q \in N ( a ) } Z _ { q } ,\tag{F1}
$$

where $N ( a )$ denotes the set of vertices adjacent to a. The corresponding parent Hamiltonian is constructed by introducing one local check for each qubit $\begin{array} { r } { h _ { a } = \frac { 1 } { 2 } \left( I - K _ { a } \right) } \end{array}$ , and defining $\begin{array} { r } { H = { \frac { 1 } { n } } \sum _ { a = 1 } ^ { n } h _ { a } } \end{array}$

Since $K _ { a }$ has eigenvalues 1, each check contributes energy 0 when $K _ { a } = + 1$ and energy 1 when $K _ { a } = - 1$ . The ground state is therefore the unique simultaneous +1 eigenstate of all the stabilizers $K _ { a } ,$ namely the graph state associated with the chosen graph.

In order to parameterize this construction with respect to the label y, we associate the two classes $y = 0$ and $y = 1$ with two different stabilizer syndromes. In particular, we choose the two syndromes $\beta ^ { ( 0 ) } = ( 0 , \ldots , 0 ) , \beta ^ { ( 1 ) } =$ $( 1 , 0 , 1 , 0 , \ldots , 1 , 0 )$ with both $\beta ^ { ( 0 ) } , \beta ^ { ( 1 ) } \in \{ 0 , 1 \} ^ { n }$ and define the label-dependent checks as

$$
h _ { a } ( y ) = \frac 1 2 \left[ I - ( - 1 ) ^ { \beta _ { a } ^ { ( y ) } } K _ { a } \right] ,\tag{F2}
$$

We can define then the label-dependent observable as the parent Hamiltonian of the graph states with the signature $\beta ^ { ( 0 ) }$ or $\beta ^ { ( 1 ) }$ according to the label y:

$$
O ( y ) = \frac { 1 } { n } \sum _ { a = 1 } ^ { n } h _ { a } ( y ) = \frac { 1 } { 2 } \left[ I - \frac { 1 } { n } \sum _ { a = 1 } ^ { n } ( - 1 ) ^ { \beta _ { a } ^ { ( y ) } } K _ { a } \right] .\tag{F3}
$$

For $y = 0 ,$ , the ground state satisfies $K _ { a } = + 1$ for every qubit $^ { a , }$ whereas for $y = 1$ it satisfies $K _ { a } = ( - 1 ) ^ { \beta _ { a } ^ { ( 1 ) } }$ corresponding to the alternating stabilizer syndrome.

To complete the construction of the Hamiltonian embedding model, we now dress the observable $O ( y )$ with a particular choice of encoding unitary $U ( x , \alpha )$ . In particular, for n qubits we choose the simple unitary

$$
U ( x , \alpha ) = \bigotimes _ { a = 1 } ^ { n } R _ { Z } \Big ( \alpha _ { a } ^ { Z } x _ { a } \Big ) R _ { Y } \Big ( \alpha _ { a } ^ { Y } x _ { a } \Big ) .\tag{F4}
$$

with $\alpha = \{ \alpha _ { a } ^ { Y } , \alpha _ { a } ^ { Z } \} _ { a = 1 } ^ { n } \in \mathbb { R } ^ { 2 n }$ denoting the set of trainable parameters. We notice that there are no entangling gates in $U ( x , \alpha )$ . This, however, does not preclude the embedding Hamiltonian from having entangled ground states, as we demonstrate below. For a datapoint $( x , y )$ the encoding Hamiltonian is then

$$
H _ { x , y } ( \alpha ) = \frac { 1 } { n } \sum _ { a = 1 } ^ { n } \frac { I - ( - 1 ) ^ { \beta _ { a } ^ { ( y ) } } \widetilde { K } _ { a } ( x , \alpha ) } { 2 } ,\tag{F5}
$$

where $\widetilde { K } _ { a } ( x , \alpha ) ~ = ~ U ( x , \alpha ) ^ { \dagger } K _ { a } U ( x , \alpha )$ . For a training set $\mathcal { T } ~ = ~ \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } ,$ , the total Hamiltonian is $H _ { \mathcal { T } } ( \alpha ) \ =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } H _ { x _ { i } , y _ { i } } ( \alpha ) } \end{array}$

## 2. Entanglement properties of the ground state

Our numerical experiments show that, for this construction, the ground states learned by the model exhibit a high degree of entanglement. This can be understood directly from the structure of the label Hamiltonians. The ground state $\left. G _ { \beta ^ { ( y ) } } \right.$ of the labeled observable $O ( y )$ in Eq. (F3) is the simultaneous eigenstate satisfying $K _ { a } \left| G _ { \beta } \right. =$ $\left( - 1 \right) ^ { \beta _ { a } } \left| G _ { \beta } \right.$ , a. For $\beta ^ { ( 0 ) }$ , this is the graph state

$$
| G _ { 0 } \rangle = \prod _ { ( i , j ) \in E } C Z _ { i j } | + \rangle ^ { \otimes n } ,\tag{F6}
$$

while for $\beta ^ { ( 1 ) }$ we have

$$
\left| G _ { 1 } \right. = \left( \prod _ { a : \beta _ { a } ^ { ( 1 ) } = 1 } Z _ { a } \right) \left| G _ { 0 } \right. .\tag{F7}
$$

The embedding Hamiltonian associated with the full training set is

$$
H _ { \mathcal { T } } ( \alpha ) = \frac { 1 } { | \mathcal { T } | } \sum _ { ( x _ { \ell } , y _ { \ell } ) \in \mathcal { T } } H _ { ( x _ { \ell } , y _ { \ell } ) } .\tag{F8}
$$

Its ground state $\left| \psi _ { T , \alpha } ^ { \star } \right.$ is in general determined by the competition between the different training terms and therefore does not coincide with the ground state of any individual term.

The case of zero energy loss is achieved when the learned parameters $\alpha ^ { \star }$ and ground state $\left| \psi _ { T , \alpha ^ { \star } } \right.$ satisfy

$$
U ( x _ { \ell } , \alpha ^ { \star } ) \left| \psi _ { T , \alpha ^ { \star } } \right. = \left| G _ { y _ { \ell } } \right. ,\tag{F9}
$$

for every training example $\left( x _ { \ell } , y _ { \ell } \right)$ . Since $U ( x , \alpha )$ is a product of single-qubit rotations, it is a local unitary and therefore preserves the bipartite entanglement spectrum. Hence, we can expect that the learned ground state $\dot { \vert \psi _ { T , \alpha ^ { \star } } \rangle }$ follows the same bipartite entanglement as the corresponding graph states $\left. G _ { 0 } \right.$ and $\left| G _ { 1 } \right.$

## 3. Numerical demonstration

To study the entanglement properties of the ground states reached by the constructed HE model, we consider the following constructed classification task. As input data $x ,$ we use images from the Fashion-MNIST dataset and apply PCA to reduce their dimensionality to n, matching the number of qubits in the model. Rather than using the original Fashion-MNIST class labels, we assign labels through a planted teacher model. Specifically, we fix a highly entangled graph-state ground state $\left| \psi _ { \mathrm { T } } \right.$ together with a set of teacher parameters $\alpha _ { \mathrm { T } } ,$ and we label each input x according to the class $\check { y } \in \{ 0 , 1 \}$ whose corresponding Hamiltonian term $H _ { x , y } ( \alpha _ { T } )$ in Eq. (F5) has the lower energy, i.e.

$$
y ( x ) = \mathop { \mathrm { a r g ~ m i n } } _ { y \in \{ 0 , 1 \} } \left. \psi _ { T } | H _ { x , y } ( \alpha _ { T } ) | \psi _ { T } \right.\tag{F10}
$$

We note that, in this Appendix, $O ( y )$ is defined as the label-dependent parent Hamiltonian, so that the state associated with the correct label minimizes its expectation value and prediction is performed via an arg min rule. This differs only by convention from the main text, where $O ( y )$ denotes the complementary score observable and prediction is therefore performed via an arg ma $\mathbf { \boldsymbol { x } } _ { y }$ rule. Moreover, the operators used here do not satisfy $O ( 0 ) + O ( \mathbf { \dot { 1 } } ) = I .$ Prediction selects the label with the lower expectation value, without interpreting these values as probabilities. The resulting labeled dataset is then used to train a Hamiltonian-embedding student model of the same form, without providing it with either the teacher state or the teacher parameters. We notice that the purpose of this experiment is to study the entanglement properties of ground states that can be learned within the Hamiltonian-embedding model, rather than the ability of the outer optimization to locate such solutions from arbitrary initializations. We therefore initialize the student parameters in the neighborhood of the planted teacher, $\alpha ^ { ( 0 ) } = \alpha _ { T } + \epsilon$ , with $\epsilon _ { j } \sim \mathcal { N } ( 0 , 0 . 2 5 ^ { 2 } )$ . In particular, we choose the teacher parameters $\alpha _ { T }$ by sampling each entry independently from a uniform distribution over $[ - \pi , \pi ]$ , and take as the teacher state $\left| { \psi _ { T } } \right.$ a highly entangled graph-syndrome state,

$$
\left. \psi _ { T } \right. = Z ^ { \beta _ { T } } \left. G _ { 0 } \right. ,\tag{F11}
$$

where $\left. G _ { 0 } \right.$ is the graph state introduced in Eq. (F6) and $\beta _ { T } \in \{ 0 , 1 \} ^ { n }$ is a fixed nontrivial stabilizer syndrome for each system size.

We use $N \ = \ 8 0 0$ training samples, together with 200 validation samples and 400 test samples for each run. For the outer parameter optimization, we use the energy-based analogue of the hinge loss introduced in Eq. (15). Since classification is performed by selecting the label with the lowest energy, we define the class-energy margin as $\Delta _ { \alpha } ( x , y ) = E _ { 1 - y } ( x ) - E _ { y } ( x )$ , and optimize $\ell _ { \mathrm { h i n g e } } ^ { ( \gamma ) } ( x , y ) = \mathrm { m a x } \{ 0 , \gamma - \Delta _ { \alpha } ( x , y ) \}$ , with $\gamma = 0 . 0 5$ . At each optimization step, the ground state of the corresponding training Hamiltonian is recomputed exactly and then kept fixed while updating the encoding parameters. After training, we study how accurately the final exact learned ground state can be represented by an MPS of bounded bond dimension. We fix the qubit ordering used in the simulation and convert the exact state into an ${ \mathrm { M P S } } ,$ truncating its Schmidt decompositions so that the maximum bond dimension is bounded by $\chi .$ For each value of $\chi ,$ we evaluate both the fidelity of the resulting truncated MPS with the exact learned state and the test accuracy obtained when this MPS is used in place of the exact state.

As shown in Fig. 2, low-bond-dimension MPS approximations in this fixed ordering have low fidelity with the learned ground state, and the corresponding classification accuracy is substantially reduced. Increasing χ progressively recovers both the exact state and the classification performance.

## Appendix G: A provable separation between Pauli surrogates and Hamiltonian embedding

In Section III $\scriptstyle \mathrm { \mathrm { B } , }$ we showed that, for certain choices of the encoding unitary $U ( x , \alpha )$ , the trained Hamiltonian embedding model can be evaluated classically once a suitable classical surrogate has been constructed from the final state $\psi _ { T , \alpha ^ { \star } }$ . This naturally raises the question of whether such a surrogate could instead be trained directly, thereby removing the need to obtain the quantum state altogether. In this Appendix, we present an example in which we analytically show that the physical constraints imposed by the Hamiltonian embedding model lead to better predictions on unseen data than the corresponding Pauli classical surrogate. Moreover, the efficiency guarantee for the classical surrogate based on reduced density matrices introduced in Eq. (47) does not extend to this construction. In order to prove a separation between the two models, we use properties of stabilizer codes and stabilizer states.

Similar to Appendix ${ \mathrm { F } } ,$ we consider a stabilizer code with stabilizers $K _ { a } , a \in [ n ]$ . Concretely, in our example, we have a graph $\bar { G }$ with stabilizers of the form $\begin{array} { r } { K _ { a } = X _ { a } \prod _ { q \in N _ { G } ( a ) } Z _ { a . } } \end{array}$ , where $N _ { G } ( a )$ is the set of neighboring vertices of a in the graph. This graph and the set of stabilizers is known and specifies the “concept class” of the labeling functions. In particular, each concept is associated with a distinct assignment specifying whether each $K _ { a }$ acts as a stabilizer or an antistabilizer of the ground state. Thus, there are $2 ^ { n }$ possible functions for a given graph, indexed by the syndrome vector $\beta$ whose coordinates $\beta _ { a }$ are +1 (-1) depending on whether $K _ { a }$ is a stabilizer (antistabilizer).

We now describe the function more explicitly. Let the data points have coordinates $x _ { i } = | x | _ { i } \zeta _ { i }$ with $| x | _ { i } \in [ 0 , 1 ]$ ] and $\zeta _ { i } = \mathrm { s i g n } ( x _ { i } )$ . For an input vector x, we form a query by considering which coordinates i have a magnitude above some cutoff $\xi .$ Define this to be the “active se $\mathrm { t } ^ { \prime \prime } S = \{ i : | x _ { i } | \geq \xi \}$ . We imagine the active set S to be a query to the function since the function depends only on the signs $\{ \zeta _ { i } , i \in S \}$ (and not on $\left| x _ { i } \right| )$ . For each such input, construct an “error vector” $\begin{array} { r } { { ' } W _ { \zeta } : = \prod _ { i \in S : \zeta _ { i } = - 1 } \sum _ { i } , } \end{array}$ supported on the collection of places where the query has a negative sign. Now define $\eta ( \zeta , a )$ to be +1 if the stabilizer $K _ { a }$ commutes with the error $W _ { \zeta }$ and 1 otherwise. The function to be learned is:

![](images/57e8b97d597911a1d58d3b49a67579d07c6d006a4e8e3823cb4dd09cfee2c37f.jpg)

![](images/dc19992dbe8f1a7915381aff4107deca8e355419fa64a508b4ae034548dac54e.jpg)  
Figure 2. Left: fidelity between the final exact learned ground state and its MPS approximation obtained by Schmidt truncation at maximum bond dimension $\chi ,$ using the fixed qubit ordering of the simulation. Right: test accuracy obtained by replacing the exact learned state with the corresponding truncated MPS state.

$$
\begin{array} { l } { { \displaystyle y _ { \beta } ( \boldsymbol { x } ) = \prod _ { a \in S } y _ { \beta ( \boldsymbol { x } ) , a } , \quad \mathrm { w h e r e } } } \\ { { \displaystyle y _ { \beta ( \boldsymbol { x } ) , a } = \beta _ { a } \eta ( \zeta , a ) . } } \end{array}\tag{G1}
$$

In our case where the stabilizers are of the form $\begin{array} { r } { K _ { a } = X _ { a } \prod _ { q \in N _ { G } ( a ) } Z _ { q , } } \end{array}$ , whether an error vector $W _ { \zeta }$ commutes with $K _ { a }$ or not is decided by whether $\zeta _ { a } = \pm 1$ . In other words, we have

$$
W _ { \zeta } ^ { \dagger } K _ { a } W _ { \zeta } = \zeta _ { a } K _ { a } ,\tag{G2}
$$

so in this case $\eta ( \zeta , a ) = \zeta _ { a } .$

Therefore, the function is

$$
\begin{array} { r l r } & { } & { y _ { \beta } ( x ) = \left( \underset { a \in S } { \prod } \beta _ { a } \right) \cdot \left( \underset { a \in S } { \prod } \zeta _ { a } \right) } \\ & { } & { = \left( \underset { a : | x | _ { a } \geq \xi } { \prod } \beta _ { a } \cdot \mathrm { s i g n } ( x _ { a } ) \right) , } \end{array}\tag{G3}
$$

(G4)

where we have made the dependence on x explicit by writing out S and $\zeta _ { a }$ in terms of x.

We define some more operators for notational convenience. Let $\begin{array} { r } { L _ { a } = - i Z _ { a } K _ { a } = Y _ { a } \prod _ { q \in N _ { G } ( a ) } Z _ { q } } \end{array}$ . This operator anticommutes both with $Z _ { a }$ and $K _ { a }$ . Also, define $\begin{array} { r } { L _ { S } \ = \ \prod _ { a \in S } L _ { a } } \end{array}$ and similarly $Z _ { S }$ and $\dot { K _ { S } }$ . Note that there is no ambiguity with respect to ordering in these definitions because all the different $L _ { a }$ operators (respectively, $Z _ { a }$ and $K _ { a } )$ commute with each other. For a given active set S, let $R _ { S }$ be a Clifford operator that applies the map

$$
R _ { S } ^ { \dagger } Z _ { 1 } R _ { S } = L _ { S } .\tag{G5}
$$

Now consider the unitary $\begin{array} { r } { U ( x ) = R _ { S } \prod _ { a \in S } \exp [ - i \alpha _ { a } x _ { a } Z _ { a } / 2 ] } \end{array}$ . The form of $U ( x )$ matches the definition in this work of a unitary that is efficiently constructible from the input x. Moreover, the gates that depend on the magnitudes of x

have the same form as those considered in the numerical sections of this work, namely Pauli rotations whose angles depend linearly on the input features, exp $\left[ - \frac { i } { 2 } ( \vec { \alpha } \cdot \vec { x } ) P \right]$ , where P denotes the corresponding Pauli generator.

Now

$$
U ( x ) ^ { \dagger } Z _ { 1 } U ( x ) = \prod _ { a \in S } \exp [ + i \alpha _ { a } x _ { a } Z _ { a } / 2 ] R _ { S } ^ { \dagger } Z _ { 1 } R _ { S } \prod _ { b \in S } \exp [ - i \alpha _ { b } x _ { b } Z _ { b } / 2 ]\tag{G6}
$$

$$
= \prod _ { a \in S } \exp [ + i \alpha _ { a } x _ { a } Z _ { a } / 2 ] L _ { S } \prod _ { b \in S } \exp [ - i \alpha _ { b } x _ { b } Z _ { b } / 2 ] .\tag{G7}
$$

Since $Z _ { a }$ commutes with all $L _ { b }$ except $L _ { a } ,$ we get

$$
U ( x ) ^ { \dagger } Z _ { 1 } U ( x ) = \prod _ { a \in S } \left( \cos { \frac { \alpha _ { a } x _ { a } } { 2 } } I + i \sin { \frac { \alpha _ { a } x _ { a } } { 2 } } Z _ { a } \right) L _ { a } \left( \cos { \frac { \alpha _ { a } x _ { a } } { 2 } } I - i \sin { \frac { \alpha _ { a } x _ { a } } { 2 } } Z _ { a } \right)\tag{G8}
$$

$$
= \prod _ { a \in S } ( \cos \left( \alpha _ { a } x _ { a } \right) L _ { a } + i \sin \left( \alpha _ { a } x _ { a } \right) Z _ { a } L _ { a } )\tag{G9}
$$

$$
= \prod _ { a \in S } ( \cos { ( \alpha _ { a } x _ { a } ) } L _ { a } + \sin { ( \alpha _ { a } x _ { a } ) } K _ { a } ) .\tag{G10}
$$

Recall $x _ { a } = | x _ { a } | \zeta _ { a } , \zeta _ { a } = \pm 1$ . Then

$$
U ( x ) ^ { \dagger } Z _ { 1 } U ( x ) = \prod _ { a \in S } \left( \cos ( \alpha _ { a } | x _ { a } | ) L _ { a } + \zeta _ { a } \sin ( \alpha _ { a } | x _ { a } | ) K _ { a } \right) .\tag{G11}
$$

We note that this construction also illustrates the limitation of the reduced-density-matrix surrogate in $\operatorname { E q . } \ ( 4 7 )$ in Section III B 4. Here, the readout is $A = Z _ { 1 }$ , and the support of the dressed observable $U ( x , \alpha ) ^ { \dagger } \bar { Z _ { 1 } } U ( x , \alpha )$ depends on the active set $S ( x )$ and contains its qubits. Consequently, a fixed region supporting all training observables must contain $\textstyle \bigcup _ { \ell = 1 } ^ { N } S ( x _ { \ell } )$ , the set of qubits active in at least one training example. Whenever the sampled active sets cover all n qubits, the fixed supporting region for all the training observables $U ( x ) ^ { \dagger } Z _ { 1 } U ( x )$ must contain the entire register. Therefore, the corresponding density matrix $\sigma _ { 1 }$ has dimension $2 ^ { n }$ . Thus, the polynomial-time argument based on explicitly diagonalizing matrices on small fixed regions does not apply. We remark that the accuracy separation proved below concerns the Pauli surrogate introduced in Eq. (46).

## 1. Idealized embedding Hamiltonian and its ground state: perfect classification

The Hamiltonian term corresponding to this data point is

$$
H ( x , y ) = \frac { I - y U ^ { \dagger } Z _ { 1 } U } { 2 }\tag{G12}
$$

$$
= \frac { I - y \prod _ { a } ( \cos { \left( \alpha _ { a } \vert x _ { a } \vert \right) } L _ { a } + \zeta _ { a } \sin { \left( \alpha _ { a } \vert x _ { a } \vert \right) } K _ { a } ) } { 2 }\tag{G13}
$$

$$
= \frac { I - \prod _ { a } \beta _ { a } \zeta _ { a } \left( \cos \left( \alpha _ { a } \vert x _ { a } \vert \right) L _ { a } + \zeta _ { a } \sin \left( \alpha _ { a } \vert x _ { a } \vert \right) K _ { a } \right) } { 2 } .\tag{G14}
$$

Rewrite this in terms of binary variables $z \in \{ 0 , 1 \} ^ { | S | }$ as

$$
H ( x , y ) = \frac { 1 } { 2 } \times \left[ I - \left( \prod _ { a \in S } \beta _ { a } \right) \sum _ { z \in \{ 0 , 1 \} ^ { | S | } } P _ { z } \right] ,\tag{G15}
$$

where $\begin{array} { r } { P _ { z } = \prod _ { a : z _ { a } = 0 } \zeta _ { a } \cos { \left( \alpha _ { a } | x _ { a } | \right) } L _ { a } \times \prod _ { a : z _ { a } = 1 } } \end{array}$ sin $\left( \alpha _ { a } | x _ { a } | \right) K _ { a }$ . If we average this over all data points from the distribution, then considering the different signs of $x _ { a }$ at the same magnitude, i.e. averaging over $\dot { \zeta _ { a } } = \pm 1$ while keeping $\left| x _ { a } \right|$ fixed: we get

$$
P _ { z } + P _ { z _ { \bar { a } } } = 0 \quad { \mathrm { i f ~ } } z _ { a } = 0 ,\tag{G16}
$$

where $z _ { \bar { a } }$ is the binary vector z with the bit at position a flipped: $z _ { \bar { a } _ { a } } = 1 - z _ { a }$ . The average over signs gives

$$
\sum _ { z \in \{ 0 , 1 \} ^ { n } } P _ { z } = \prod _ { a \in S } \sin { \left( \alpha _ { a } | x _ { a } | \right) } K _ { a } .\tag{G17}
$$

This means the embedding Hamiltonian averaged over different signs for the same $\left| x _ { a } \right|$ gives

$$
\begin{array} { l } { { \displaystyle H _ { \mathcal { D } } ( | x | ) = \frac { 1 } { 2 } \times \left[ I - \left( \prod _ { a \in S } \beta _ { a } \right) \prod _ { a \in S } \sin { ( \alpha _ { a } | x _ { a } | ) K _ { a } } \right] } } \\ { { \displaystyle ~ = \frac { I - \prod _ { a \in S } \bigl ( \beta _ { a } \sin { ( \alpha _ { a } | x _ { a } | ) K _ { a } } \bigr ) } { 2 } . } } \end{array}\tag{G18}
$$

(G19)

For the full embedding Hamiltonian, we should also average over the distribution of x for all coordinates, denoted $g ( x )$

$$
H _ { \mathcal { D } } = \int d ^ { n } x g ( x ) \frac { I - \prod _ { a \in S } \beta _ { a } ( \sin { ( \alpha _ { a } \vert x _ { a } \vert ) } K _ { a } ) } { 2 } .\tag{G20}
$$

This is a commuting Hamiltonian whose terms are products of $\beta _ { a } K _ { a }$ . The number of products that appear is precisely S , which is the sparsity of the input vector x above the cutoff $\xi .$ . If the distribution over the input data is supported on 1-sparse inputs $x ,$ then these stabilizers $\beta _ { a } K _ { a }$ are represented in the Hamiltonian. Since these stabilizers commute with each other, the full embedding Hamiltonian is frustration-free. Its ground space is the +1 eigenspace of the operators $\cup _ { S } \{ \prod _ { a \in S } \beta _ { a } K _ { a } \}$ . Note that if the distribution that the input data x is drawn from is not supported on all coordinates, the group generated by the stabilizer generators $\cup _ { S } \{ \prod _ { a \in S } \beta _ { a } K _ { a } \}$ may not be full rank.

Now consider classification. Assume that the ground state ψ of the embedding Hamiltonian is found by the ground state solver (which we treat as a black box in this Appendix). The expectation value of $U ( x ) ^ { \dagger } Z _ { 1 } U ( x )$ in this state is:

$$
\left. \psi \right| U ( x ) ^ { \dagger } Z _ { 1 } U ( x ) \left| \psi \right. = \left. \psi \right| \prod _ { a \in S } ( \cos \left( \alpha _ { a } | x _ { a } | \right) L _ { a } + \zeta _ { a } \sin \left( \alpha _ { a } | x _ { a } | \right) K _ { a } ) \left| \psi \right. .\tag{G21}
$$

We have $\left. \psi \right| L _ { a } \left| \psi \right. = 0$ because $L _ { a }$ anticommutes with $\beta _ { a } K _ { a . }$ , which is the stabilizer of $\left| \psi \right.$ . Therefore,

$$
\left. \psi \right. U ( x ) ^ { \dagger } Z _ { 1 } U ( x ) \left. \psi \right. = \left. \psi \right. \prod _ { a \in S } ( \cos \left( \alpha _ { a } \vert x _ { a } \right. ) L _ { a } + \zeta _ { a } \sin \left( \alpha _ { a } \vert x _ { a } \right. ) K _ { a } ) \left. \psi \right. = \prod _ { a \in S } \beta _ { a } \zeta _ { a } \sin \left( \alpha _ { a } \vert x _ { a } \right. ) .\tag{G22}
$$

The sign of this is $\Pi _ { a \in S } \beta _ { a } \zeta _ { a } \mathrm { s i g n } ( \alpha _ { a } )$ , which matches the function $y _ { \beta } ( x )$ whenever $\mathrm { s i g n } ( \alpha _ { a } ) = 1$ for all $a \in S$ Therefore, classification is perfect as long as $\alpha _ { a } > 0$

## 2. Embedding Hamiltonian with finite training data

In this subsection, we specify the distribution over inputs and prove that there exists a regime of number of training points such that the empirical Hamiltonian embedding model classifies the data nearly perfectly; in this same regime we can also prove an upper bound on the accuracy of the classical Pauli surrogates described in Section III B 2. This separation between the classical surrogate and the Hamiltonian embedding model proves that this model generalizes better when the model is suited to the task.

We choose the input training data to be drawn from the following distribution. First, we choose the size of the active set according to the distribution

$$
\operatorname* { P r } ( | S | = 2 ) = { \frac { 4 } { 7 } } , \qquad \operatorname* { P r } ( | S | = 3 ) = { \frac { 2 } { 7 } } , \qquad \operatorname* { P r } ( | S | = 4 ) = { \frac { 1 } { 7 } } .\tag{G23}
$$

Then, for a given size $| S | ,$ , we pick the active coordinates uniformly. Let a denote a location that is active, that is $a \in S .$ The input vector x at coordinate a has magnitude $| \boldsymbol { x } | _ { a } = r _ { a } ,$ , where $r _ { a }$ is a random number in [ξ, 1]. This number $r _ { a }$ is fixed for the duration of the experiment, that is, any other training point with a an active location will have $| \boldsymbol { x } | _ { a } = r _ { a }$

Now consider the empirical embedding Hamiltonian as opposed to the idealized one in Eq. (G20). The difference is that in practice we may not have queried every possible input, which is precisely the case where we expect the Hamiltonian embedding model to generalize better by virtue of its inductive bias. The empirical embedding Hamiltonian is

$$
H _ { T } ( \alpha ) = \frac { 1 } { N } \sum _ { \ell } H ( x _ { \ell } , y _ { \ell } ) = \sum _ { \ell } \frac { 1 } { 2 N } \times \left[ I - \left( \prod _ { a \in S ^ { \ell } } \beta _ { a } \right) \sum _ { z \in \{ 0 , 1 \} } P _ { z ^ { \ell } } ( \alpha ) \right] ,\tag{G24}
$$

where $\begin{array} { r } { P _ { z ^ { \ell } } ( \alpha ) = \prod _ { a : z _ { a } ^ { \ell } = 0 } \zeta _ { a } ^ { \ell } \cos { ( \alpha _ { a } \left| x _ { a } ^ { \ell } \right| ) } L _ { a } \times \prod _ { a : z _ { a } ^ { \ell } = 1 } \sin { ( \alpha _ { a } \left| x _ { a } ^ { \ell } \right| ) } K _ { a } } \end{array}$ and $\ell = 1 , 2 , \ldots$ . N denotes the different input training data points. Since, as mentioned before, for different data points we have the same magnitude $\left| x _ { a } ^ { \ell } \right| = r _ { a } \forall \ell ,$ we can rewrite this as:

$$
P _ { z ^ { \ell } } ( \alpha ) = \prod _ { a : z _ { a } ^ { \ell } = 0 } \zeta _ { a } ^ { \ell } \cos { ( \alpha _ { a } r _ { a } ) } L _ { a } \times \prod _ { a : z _ { a } ^ { \ell } = 1 } \sin { ( \alpha _ { a } r _ { a } ) } K _ { a } .\tag{G25}
$$

Notice that in this case, since we do not assume that the different signs $\zeta _ { a }$ queried at the same magnitude $r _ { a }$ cancel out, and thus the cosine term survives. This can potentially be problematic because $L _ { a }$ anticommutes with $K _ { a }$ and would make the Hamiltonian frustrated. Nevertheless, the empirical embedding Hamiltonian can have the same ground state $\left| \psi \right.$ as the full embedding Hamiltonian if the angles α can be trained. In this case, setting $\alpha _ { a } ^ { * } = \frac { \pi } { 2 r _ { a } }$ ensures that the cosine term gives 0 and thus only the all-sine term $z ^ { \ell } = 1 \ldots 1$ survives: $\begin{array} { r } { P _ { z ^ { \ell } } ( \alpha ) = \prod _ { a \in S ^ { \ell } } K _ { a } , } \end{array}$ , giving

$$
H _ { \mathcal { T } } ( \alpha ^ { * } ) = \sum _ { \ell } \frac { 1 } { 2 N } \times \left[ I - \left( \prod _ { a \in S ^ { \ell } } \beta _ { a } K _ { a } \right) \right] .\tag{G26}
$$

As we have seen before, the Hamiltonian has as ground space the +1 eigenspace of the stabilizer group $\cup _ { \ell } \{ \Pi _ { a \in S ^ { \ell } } \beta _ { a } K _ { a } \}$ with energy $\langle H _ { \mathcal { T } } \rangle = 0$ . As we have restricted ourselves to positive semidefinite Hamiltonians, this proves that $\alpha ^ { * }$ is an optimum.

## a. Empirical accuracy of Hamiltonian embedding

Consider the binary matrix A, whose ℓth row records the active set $S ^ { \ell }$ . We are interested in the event that A has rank n over $\mathbb { F } _ { 2 } .$ . This is equivalent to whether the sampled stabilizer products generate the full stabilizer group $\langle \beta _ { a } K _ { a } : a \in [ n ] \rangle$ . In this case, the empirical embedding Hamiltonian would have the unique ground state ψ . Note that merely requiring every coordinate a to appear in some sampled query is not sufficient for this spanning condition. On the other hand, if N is large enough that every possible input in the support of the distribution is observed, the classical surrogate can also classify perfectly by memorization.

Therefore, we now investigate the case where N is bounded. Specifically, we choose $N = 4 n \log n$ . Our analysis of accuracy depends on the probability that the stabilizer group $\scriptstyle { \dot { \bigcup _ { \ell } } } \{ \prod _ { a \in S ^ { \ell } } { \bf \dot { \beta } } _ { a } K _ { a } \}$ has full rank. Consider the matrix $A _ { N \times n }$ whose rows correspond to training data. For each data point $\ell ,$ we have $A _ { \ell , a } = 1$ if a is in the active set $S ^ { \ell } .$ , and 0 otherwise. Now A is not full rank if and only if there exists a nonzero binary vector v such that $A \cdot v = 0$ . Denote by T the set of locations i where $v _ { i } = 1$ and let $\mathbf { \sigma } _ { | T | } = t$ . Then

$$
A _ { \ell } \cdot v = 0 \Leftrightarrow \left| S ^ { \ell } \cap T \right| = 0 { \bmod { 2 } } .\tag{G27}
$$

Suppose we fix a v and consider whether for a random query ℓ from the distribution v is “detected” by $\ell ,$ i.e. whether $| \boldsymbol { S } \cap \boldsymbol { T } | = 1$ mod 2. The probability of this event, conditioned on $| S | = 2$ , is:

$$
\operatorname* { P r } [ | S \cap T | = 1 | | S | = 2 ] = { \frac { t \times ( n - t ) } { { \binom { n } { 2 } } } } .\tag{G28}
$$

Similarly, conditioning on $| S | = 3$ , we get the two cases $| S \cap T | = 1$ or 3, giving

$$
\operatorname* { P r } \Big [ | S \cap T | = 1 \bmod 2 \Big | | S | = 3 \Big ] = \frac { t \times { \binom { n - t } { 2 } } } { { \binom { n } { 3 } } } + \frac { { \binom { t } { 3 } } \times { \binom { n - t } { 0 } } } { { \binom { n } { 3 } } } .\tag{G29}
$$

Lastly, when $| S | = 4$ , we again have the possibilities $| S \cap T | = 1 \operatorname { o r } 3 ,$ with the probability

$$
\operatorname* { P r } \Big [ | S \cap T | = 1 \bmod 2 \Big | | S | = 4 \Big ] = \frac { t \times { \binom { n - t } { 3 } } } { { \binom { n } { 4 } } } + \frac { { \binom { t } { 3 } } \times { \binom { n - t } { 1 } } } { { \binom { n } { 4 } } } .\tag{G30}
$$

Denote the combined probability

$$
\begin{array} { r l } & { { r _ { t } } = \mathrm { P r } \left[ | { \mathcal { S } } \cap T | = 1 \mathrm { ~ m o d ~ } 2 \right] = \frac { 4 } { 7 } \mathrm { P r } \left[ | { \mathcal { S } } \cap T | = 1 \bigg | | { \mathcal { S } } | = 2 \right] + \frac { 2 } { 7 } \mathrm { P r } \left[ | { \mathcal { S } } \cap T | = 1 \mathrm { ~ m o d ~ } 2 \bigg | | { \mathcal { S } } | = 3 \right] } \\ & { \qquad + \frac { 1 } { 7 } \mathrm { P r } \left[ | { \mathcal { S } } \cap T | = 1 \mathrm { ~ m o d ~ } 2 \bigg | | { \mathcal { S } } | = 4 \right] , } \end{array}\tag{G31}
$$

where we have made use of the distribution over the active set sizes. The probability that no row detects v is given by $( 1 - r _ { t } ) ^ { N } \leq e ^ { - N r _ { t } }$ . By a union bound over all nonzero $\boldsymbol { v } \in \mathbb { F } _ { 2 } ^ { n }$

$$
\operatorname* { P r } [ \operatorname { r a n k } ( A ) < n ] \leq \sum _ { t = 1 } ^ { n } { \binom { n } { t } } e ^ { - N r _ { t } } .\tag{G32}
$$

We set $N = \lceil 4 n \log n \rceil$ and divide the sum into two regimes.

First, suppose $1 \leq t \leq n / 2$ . Retaining only the contribution from 2-sparse queries gives

$$
r _ { t } \geq { \frac { 4 } { 7 } } { \frac { t ( n - t ) } { { \binom { n } { 2 } } } }\tag{G33}
$$

$$
= { \frac { 8 t ( n - t ) } { 7 n ( n - 1 ) } } \geq { \frac { 4 t } { 7 n } } .\tag{G34}
$$

Consequently,

$$
N r _ { t } \geq \frac { 1 6 } { 7 } t \log n .\tag{G35}
$$

Using ${ \bf \Pi } _ { t } ^ { n } ) \le ( e n / t ) ^ { t }$ , the contribution of this regime to Eq. (G32) is at most

$$
\sum _ { t = 1 } ^ { \lfloor n / 2 \rfloor } { \binom { n } { t } } e ^ { - N r _ { t } } \leq \sum _ { t = 1 } ^ { \lfloor n / 2 \rfloor } \left( { \frac { e n } { t } } \right) ^ { t } n ^ { - 1 6 t / 7 }\tag{G36}
$$

$$
\leq \sum _ { t \geq 1 } \left( { \frac { e } { n ^ { 9 / 7 } } } \right) ^ { t } = O \left( n ^ { - 9 / 7 } \right) .\tag{G37}
$$

Next, suppose $n / 2 < t \leq n .$ A 3-sparse query lying entirely inside T has odd intersection with $T ,$ and hence

$$
r _ { t } \ge \frac { 2 } { 7 } \frac { { \binom { t } { 3 } } } { { \binom { n } { 3 } } } \ge \frac { 1 } { 5 6 }\tag{G38}
$$

for all sufficiently large n. The contribution of this regime is therefore at most

$$
2 ^ { n } e ^ { - N / 5 6 } \leq \exp \left( n \log 2 - { \frac { n \log n } { 1 4 } } \right) = o ( 1 ) .\tag{G39}
$$

Combining the two regimes gives

$$
\operatorname* { P r } [ \operatorname { r a n k } ( A ) < n ] = o ( 1 ) ,\tag{G40}
$$

so A has rank n with probability $1 - o ( 1 )$ . This means that the accuracy of the Hamiltonian embedding model, assuming that the ground-state solver is perfect, would be $1 - o ( 1 )$ .

## b. Accuracy of classical surrogates

For an active set S, define

$$
{ \cal P } _ { S , z } : = \prod _ { a \in S : z _ { a } = 0 } L _ { a } \prod _ { a \in S : z _ { a } = 1 } K _ { a } ,\tag{G41}
$$

$$
c _ { S , z } ( x , \alpha ) : = \prod _ { a \in S : z _ { a } = 0 } \cos ( \alpha _ { a } \vert x _ { a } \vert ) \prod _ { a \in S : z _ { a } = 1 } \zeta _ { a } \sin ( \alpha _ { a } \vert x _ { a } \vert ) ,\tag{G42}
$$

where $z \in \{ 0 , 1 \} ^ { | S | }$ . The inference observable and its Pauli surrogate are then

$$
U ( x , \alpha ) ^ { \dagger } Z _ { 1 } U ( x , \alpha ) = \sum _ { z \in \{ 0 , 1 \} ^ { | S | } } c _ { S , z } ( x , \alpha ) P _ { S , z } ,\tag{G43}
$$

$$
g _ { S } ( x , \alpha ) = \sum _ { z \in \{ 0 , 1 \} ^ { | S | } } c _ { S , z } ( x , \alpha ) m _ { S , z } .\tag{G44}
$$

Here, the expectation value of $P _ { S , z } ,$ namely $\left. \phi \right| P _ { S , z } \left| \phi \right.$ , is replaced by the coefficient $m _ { S , z } ,$ which is unconstrained. For the graph stabilizers considered here, $P _ { S , z }$ has an $X \mathrm { o r } Y$ on precisely the qubits in $S .$ Pauli observables belonging to different active sets are therefore distinct, so the free surrogate has a separate coefficient block for each possible query S.

For fixed $\alpha ,$ the loss function used in ridge regression is

$$
J ( \boldsymbol { m } ; \alpha ) = \frac { 1 } { 2 N } \sum _ { \ell = 1 } ^ { N } \left[ y _ { \ell } - \sum _ { z \in \{ 0 , 1 \} } c _ { S ^ { \ell } , z } ( x ^ { \ell } , \alpha ) m _ { S ^ { \ell } , z } \right] ^ { 2 } + \frac { \lambda } { 2 } \sum _ { S \in \mathcal { Q } } \sum _ { z \in \{ 0 , 1 \} ^ { | S | } } m _ { S , z } ^ { 2 } ,\tag{G45}
$$

where the second sum $S \in \mathcal { Q }$ is over all supports ${ \mathcal { Q } } = \{ S \subseteq [ n ] \}$ . If a support $S$ is absent from the training set, its coefficient block does not enter the data-fitting term. The unique ridge solution is therefore

$$
m _ { S , z } ^ { \star } ( \alpha ) = 0 \qquad \mathrm { f o r } \ a l l z , \quad S \not \in \mathcal { Q } _ { \mathcal { T } } ,\tag{G46}
$$

where $\mathcal { Q } _ { T }$ is the set of observed supports. This holds for every $\alpha ,$ and hence also when α and m are optimized jointly for the classical surrogate.

We can evaluate the coefficients explicitly when α is fixed to the Hamiltonian-embedding optimum constructed above, $\begin{array} { r } { \alpha _ { a } ^ { \star } r _ { a } = \frac { \pi } { 2 } } \end{array}$ . Writing $\begin{array} { r } { \beta _ { S } : = \Pi _ { a \in S } \mathbf { \bar { \beta } } _ { a } } \end{array}$ , we then have

$$
c _ { S , \mathbf { 1 } _ { S } } ( x , \alpha ^ { \star } ) = \prod _ { a \in S } \zeta _ { a } , \qquad c _ { S , z } ( x , \alpha ^ { \star } ) = 0 \quad ( z \neq \mathbf { 1 } _ { S } ) ,\tag{G47}
$$

and the label is $\begin{array} { r } { y _ { \beta } ( x ) = \beta _ { S } \prod _ { a \in S } \zeta _ { a } } \end{array}$

If $N _ { S }$ denotes the number of training examples with support $S ,$ the ridge solution is found by setting

$$
\begin{array} { r l r } {  { \frac { \partial J ( m ; \boldsymbol { \alpha } ) } { m _ { S , z } } = 0 } } \\ & { } & { \quad = - \frac { N _ { S } } { N } \beta _ { S } \delta _ { z , \mathbf { 1 } _ { S } } + \frac { N _ { S } } { N } m _ { S , z } \delta _ { z , \mathbf { 1 } _ { S } } + \lambda m _ { S , z } , } \end{array}\tag{G48}
$$

(G49)

giving

$$
m _ { S , z } ^ { \star } ( \alpha ^ { \star } ) = \left\{ \begin{array} { l l } { \displaystyle \beta _ { S } \frac { N _ { S } } { N _ { S } + \lambda N } , } & { \displaystyle N _ { S } > 0 \mathrm { a n d } z = \mathbf { 1 } _ { S } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{G50}
$$

Consequently, every observed support is classified correctly:

$$
y _ { \beta } ( x ) g _ { S } ( x , \alpha ^ { \star } ) = \frac { N _ { S } } { N _ { S } + \lambda N } > 0 , \qquad N _ { S } > 0 ,\tag{G51}
$$

whereas an unobserved support has zero score.

Let $M _ { k } ( \mathcal { T } )$ be the number of distinct k-element supports observed during training. A new query picked at random according to the distribution in Eq. (G23), will have been seen previously with probability

$$
q _ { \mathrm { s e e n } } ( \mathcal { T } ) = \sum _ { k = 2 } ^ { 4 } p _ { k } \frac { M _ { k } ( \mathcal { T } ) } { \binom { n } { k } } .\tag{G52}
$$

On a previously unobserved support the accuracy is $1 / 2$ . Therefore, for any fixed or independently optimized $\alpha ,$ the ridge classifier’s accuracy is

$$
A _ { \mathrm { r i d g e } } ( \mathcal { T } ; \alpha ) = \frac { 1 - q _ { \mathrm { s e e n } } ( \mathcal { T } ) } { 2 } + q _ { \mathrm { s e e n } } ( \mathcal { T } ) A _ { \mathrm { s e e n } } ( \mathcal { T } ; \alpha ) ,\tag{G53}
$$

where $A _ { \mathrm { s e e n } }$ is the classifier’s accuracy on previously seen points. This gives

$$
\left| A _ { \mathrm { r i d g e } } ( \mathcal { T } ; \alpha ) - \frac { 1 } { 2 } \right| \le \frac { 1 } { 2 } q _ { \mathrm { s e e n } } ( \mathcal { T } ) .\tag{G54}
$$

When α is the optimal Hamiltonian-embedding solution, Eq. (G50) gives $A _ { \mathrm { s e e n } } = 1$ , and hence

$$
A _ { \mathrm { r i d g e } } ( \mathcal { T } ; \alpha ^ { \star } ) = \frac { 1 } { 2 } + \frac { 1 } { 2 } q _ { \mathrm { s e e n } } ( \mathcal { T } ) .\tag{G55}
$$

Thus even if the surrogate optimizes α independently its accuracy remains bounded; moreover, fixing $\alpha = \alpha ^ { \star }$ shows that the bound is achievable.

Lastly, we bound $q _ { \mathrm { s e e n } } .$ . For a particular k-element support, the probability of appearing at least once in N independent training queries is

$$
1 - \left( 1 - \frac { p _ { k } } { { \binom { n } { k } } } \right) ^ { N } .\tag{G56}
$$

Averaging $\operatorname { E q } .$ . (G53) over the training set gives

$$
\mathbb { E } _ { \mathcal { T } } [ A _ { \mathrm { r i d g e } } ] \le \frac { 1 } { 2 } + \frac { 1 } { 2 } \sum _ { k = 2 } ^ { 4 } p _ { k } \left[ 1 - \left( 1 - \frac { p _ { k } } { { \binom { n } { k } } } \right) ^ { N } \right]\tag{G57}
$$

$$
\leq \frac { 1 } { 2 } + \frac { 1 } { 2 } \sum _ { k = 2 } ^ { 4 } \frac { N p _ { k } ^ { 2 } } { \binom { n } { k } }\tag{G58}
$$

$$
\leq { \frac { 1 } { 2 } } + { \frac { N } { 9 8 } } \left[ { \frac { 1 6 } { { \binom { n } { 2 } } } } + { \frac { 4 } { { \binom { n } { 3 } } } } + { \frac { 1 } { { \binom { n } { 4 } } } } \right] ,\tag{G59}
$$

where the first inequality is an equality at $\alpha ^ { \star }$ . In particular, for $N = \Theta ( n \log n )$

$$
A _ { \mathrm { r i d g e } } = { \frac { 1 } { 2 } } + O \left( { \frac { \log n } { n } } \right) .\tag{G60}
$$

The difference between the surrogate and the Hamiltonian embedding model is that the latter combines the observed stabilizer products: once their incidence vectors (i.e. the rows in $\bar { A } )$ have full rank, the underlying generator signs $\beta _ { a }$ are fixed and every support can be classified. The fact that a single state is consistent with all Pauli expectation values and also informs the expectation values of unseen Paulis by virtue of the Pauli algebra is what provides the inductive bias in this case, allowing Hamiltonian embedding to generalize better than the classical surrogate.

## 3. Required bond dimension for good classification with Hamiltonian embedding

In this subsection, we empirically study this previous example by instantiating a family of graphs and their corresponding codes. We randomly generate a simple, connected 3-regular graph, which fixes the stabilizer code family, at each system size. As mentioned earlier, the stabilizers are $\beta _ { i } X _ { i } \Breve { \Pi } _ { j \in N ( i ) } \Breve { Z _ { j } } ,$ , with the signs $\beta \in \pm 1$ that describe the function also drawn uniformly at random at every vertex. We set the cutoff $\xi = 0 . 6$ and draw the radius of every active coordinate $| x _ { i } | \sim \mathrm { U n i f } [ \bar { 0 } . 6 , 1 ]$ for every realization and fix them thereafter. The signs of the input coordinates are again chosen uniformly at random. This fixes the distribution over input vectors $\in \mathbb { R } ^ { n }$ . We take the same regime of $N = 4 n \log n ,$ with the same distribution over queries as Eq. (G23).

In order to train the Hamiltonian embedding model, we adopt a different numerical technique from the one considered in Section IV. Specifically, we apply imaginary time evolution on a classical MPS representation of the state. The MPS has a fixed ordering and no attempt was made at optimizing the ordering in order to better represent the ideal stabilizer ground state, since this strategy does not scale in general. After each application of imaginary time evolution, we compress the MPS up to a maximum bond dimension $\chi ,$ with a numerical cutoff of $1 0 ^ { - 8 }$ for the singular value decomposition (SVD). For the numerical optimization, we parameterize the rotation angles as

$$
\alpha _ { a } = \alpha _ { c } + \Delta _ { \alpha } u _ { a } , \qquad u _ { a } \in [ - 1 , 1 ] ,\tag{G61}
$$

where

$$
\alpha _ { c } = \frac { \pi } { 4 } ( 1 + \xi ^ { - 1 } ) , \qquad \Delta _ { \alpha } = \frac { \pi } { 4 } ( \xi ^ { - 1 } - 1 ) .\tag{G62}
$$

Thus $\begin{array} { r } { \alpha _ { a } ~ \in ~ \left[ \frac { \pi } { 2 } , \frac { \pi } { 2 \xi } \right] . } \end{array}$ , an interval containing the analytic optimum $\begin{array} { r } { \alpha _ { a } ^ { \star } ~ = ~ \frac { \pi } { 2 r _ { a } } } \end{array}$ for $r _ { a } ~ \in ~ [ \xi , 1 ]$ . We initialize $u _ { a } \sim$ $\mathrm { U n i f } [ - 0 . 7 5 , 0 . 7 5 ]$ and optimize u using Adam, projecting it onto $\lceil - 1 , 1 \rceil ^ { n }$ after each update. During this procedure, we optimize the energy of the parameterized Hamiltonian $H _ { T } ( \alpha )$ , keeping the state fixed.

![](images/34deecc8fd4b8b4cc8cfdb90d87da562dfe575ab87f5629e55ce76cab1a7e505.jpg)

![](images/fd4f778bd655d1f501c4b223d972b6d3d31d29512f88164c79456bfb2c368efd.jpg)

(c)  
![](images/1ad8e6db48bb89efdd3460907a51d451e6734b1111b34c4cb375f7337cd192c1.jpg)

(d)  
![](images/5274a1479b26df165572d4641849bbfbc57dbe40d2fe984ae1a79841461b31db.jpg)  
Figure 3. Test accuracy $A _ { \eta }$ as a function of bond dimension at system sizes $n \in \{ 1 2 , 2 4 , 3 6 , 4 8 , 6 0 \}$ , averaged over 10 trials, with (a) $\eta = 0 . 1 , ( { \bf b } ) \eta = \bar { 0 } . 0 1$ . (c) Minimum energy seen during training as a function of bond dimension. (d) The signed margin y(x) $\langle U ( x ) ^ { \dagger } Z _ { 1 } U ( x ) \rangle$ .

We illustrate the accuracy of the trained model as a function of the maximum bond dimension cap $\chi$ in Fig. 3. Writing $m ( x ) = y ( x ) \langle U ( x ) ^ { \dagger } Z _ { 1 } U ( x ) \rangle$ , each test example receives score 1 when $m ( x ) \geq \eta , 0$ when $m ( x ) \leq - \eta$ , and $1 / 2$ when $| m ( x ) | < \eta$ . The accuracy $A _ { \eta }$ is the average of these scores. We empirically find that increasing bond dimension is necessary in order to achieve the same accuracy at larger system sizes. Similarly, with increasing bond dimension, the energy and the observable’s margins also improve.

In Fig. 4, we also plot the empirical accuracy of the classical surrogates and compare with the one from Hamiltonian embedding. For the points labeled as coming from HE, we show the highest test score observed over all bond dimensions. We notice that the classical surrogates trained on the ridge loss show a gradual decay in accuracy with respect to system size n.

## Appendix H: Numerical setup

In this Appendix, we discuss the Hamiltonian embedding model used in all the experiments of Section IV, together with the classical simulation of the training loop. The protocols of the individual tasks are given in Appendices I and J.

## 1. Encoding circuit and embedding Hamiltonian

We use the encoding unitary of Eq. (12) with $K = 3$ layers. Layer ℓ applies to every qubit j the sequence of single-qubit Pauli rotations e $- i \dot { \theta } _ { j , 3 } ^ { \ell } Z _ { j } / 2 \stackrel {  } { e } e ^ { - i \theta _ { j , 2 } ^ { \ell } Y _ { j } / 2 } e ^ { - i \theta _ { j , 1 } ^ { \ell } Z _ { j } / 2 }$ , followed by a ring of nearest-neighbor two-qubit rotations $e ^ { - i \bar { \theta } _ { j , 4 } ^ { \ell } Z _ { j } \bar { Z } _ { j + 1 } / 2 }$ with periodic boundary conditions. All 12n angles are independently trainable, and each is a linear

![](images/bcc719b3736867a5064fa7ae8cebcb88326f443c260504401f99576fab32a3b4.jpg)  
Figure 4. Accuracy of surrogate models. The green curve shows the surrogate that optimizes the $m _ { S , z }$ from Eq. (G44), keeping $c _ { S , z } ( \alpha )$ fixed by taking $\alpha = \alpha ^ { * }$ obtained from the Hamiltonian embedding model. The orange curve optimizes over both $m _ { S , z }$ and α (which determines $c _ { S , z } ( \alpha ) )$ . Also shown in orange dots for comparison are the best (over bond dimensions χ) accuracies for the Hamiltonian embedding model for each trial.

function of the input feature and of the trainable weights,

$$
\theta _ { g } ( x , \alpha ) = \sum _ { r = 1 } ^ { d } \alpha _ { g , r } x _ { r } + \alpha _ { g , 0 } , \qquad g = 1 , \ldots , 1 2 n ,\tag{H1}
$$

so the trainable parameters are $\alpha \in \mathbb { R } ^ { 1 2 n \times ( d + 1 ) }$ and they are a total of $M = 1 2 n ( d + 1 )$ . Every trainable gate is generated by a Pauli operator with $P ^ { 2 } = I _ { \cdot }$ , so the cost admits the parameter-shift rule of Section II B. The two-qubit rotations are nearest-neighbor, so the circuit remains of constant depth with a light cone of constant size. Lastly, the label observable is set to $\begin{array} { r } { A = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ throughout.

We consider a constant-depth encoding unitary $U ( \alpha , x )$ such that the observable A can be evolved under $U ( \alpha , x )$ classically. In particular, using backpropagation, we store a list of which Pauli terms appear in the embedding Hamiltonian together with the functional dependence of their coefficients from the features and trainable weights.

## 2. Classical iterative solver

We describe the training procedure when using the classical iterative solver. At the first training step, we start from the reference state $| { \mathrm { r e f } } \rangle = | 0 ^ { n } \rangle$ and construct truncated approximations to the six vectors $\{ H _ { \mathcal { T } } ^ { k } ( \alpha _ { 0 } ) ~ \mathrm { | r e f \rangle } \} _ { k = 0 } ^ { 5 }$ obtained by repeatedly applying the data Hamiltonian evaluated at the initial parameters $\alpha _ { 0 }$ . After each application, we keep up to 8,000 computational-basis bitstrings with the largest amplitudes. We then diagonalize $H _ { \mathcal { T } } ( \bar { \alpha _ { 0 } } )$ in the subspace spanned by the collected computational-basis states, obtaining an approximate ground state. Let us refer to this procedure of applying the Hamiltonian, collecting bitstrings, and diagonalizing as one “round”. We perform three rounds in total at fixed $\alpha _ { 0 }$ , where the ground state from each round is used as the reference state for the next, and the diagonalization in each round is in the subspace spanned by all collected bitstrings from previous rounds. This gives us the approximate ground state for $H _ { T } ( \alpha _ { 0 } )$ . At subsequent training steps, we use an analogous procedure, taking the approximate ground state obtained at step t  1 as the reference state. Here, each step involves a single round with only one application of $H _ { \mathcal { T } } ( \alpha _ { t } )$ , yielding truncated approximations to the two Krylov states $\{ H _ { \mathcal { T } } ^ { k } ( \alpha _ { t } ) \ \bar { | } \mathrm { r e f } \rangle \} _ { k = 0 } ^ { 1 } .$ Furthermore, for optimization steps $\dot { t } > 0 ,$ the truncation limit is reduced from 8,000 to 3, s000 computational-basis components per state, retaining those with the largest amplitudes.

## 3. Optimization and model selection

The trainable parameters are initialized by drawing from independent Gaussian distributions $\alpha _ { g , r } \sim \mathcal { N } ( 0 , 0 . 3 ^ { 2 } )$ using a fixed seed for each run to ensure reproducibility. To limit overfitting, we add a squared-ℓ<sub>2</sub> penalty to the training objective,

$$
\widehat { C } _ { \mathcal { T } } ^ { \mathrm { r e g } } ( \psi , \alpha ) = \widehat { C } _ { \mathcal { T } } ( \psi , \alpha ) + \frac { \lambda } { 2 } \| \alpha \| _ { 2 } ^ { 2 } ,\tag{H2}
$$

where α denotes the full parameter vector and $\lambda = 0 . 0 0 5$ . At iteration $t ,$ we hold the state $\left| { \psi _ { t } } \right.$ used for gradient evaluation fixed and compute

$$
g _ { t } = \left. \langle \psi _ { t } | \nabla _ { \alpha } H _ { T } ( \alpha ) | \psi _ { t } \rangle \right| _ { \alpha = \alpha _ { t } } + \lambda \alpha _ { t } .\tag{H3}
$$

For the parameter update step, we use Adam with learning rate 0.05 and constants $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and $\epsilon = 1 0 ^ { - 8 }$ . Training is performed for 80 iterations for the image-classification benchmark described in the next section.

Model selection after training is performed by scoring the performance at each training epoch against a validation set that is split off the training data. In practice we hold out 20% of the latter, and the final reported model corresponds to the epoch with the highest validation accuracy. Classification uses the score $f ( x ) = \langle \hat { \psi | } U ^ { \dagger } ( x , \alpha ) A U ( x , \alpha ) \vert \psi \rangle$ with label 0 assigned when $f ( { \overset { \vartriangle } { x } } ) \geq 0$ and label 1 otherwise.

## Appendix I: Image classification benchmark

## 1. Protocol

The task considered is the binary classification of the Fashion-MNIST dataset [XRV17], restricted to the classes dress (label 3) and shirt (label 6), which is the setting used in Appendix L of Ref. [HBM<sup>+</sup>21]. Following that protocol, the two classes are pooled (14,000 images of 784 pixels each), the pool is reduced by principal component analysis to n components, each component is standardized to zero mean and unit variance, and $\hat { N } _ { \mathrm { p o o l } } = \mathrm { 8 0 0 }$ points are subsampled without replacement using the seed of the run. A stratified 70/30 train–test split of the 800 points is then drawn from the same seed, and a further stratified 80/20 split of the training part provides the inner validation set used for model selection. For every pair (qubit count, seed) the subsample and the splits are drawn once and shared by every method, so all comparisons are paired.

## 2. Experimental Baselines

The quantum baselines are those of Ref. [HBM<sup>+</sup>21], evaluated on the same splits and with the same hyperparameters. Three data embeddings $x \mapsto | \phi ( x ) \rangle$ are used: a product of single-qubit rotations, one per feature (E1); an IQP-type circuit in which two layers of Hadamards sandwich diagonal phases with linear terms $x _ { j } Z _ { j }$ and all-toall quadratic terms $x _ { j } x _ { j ^ { \prime } } Z _ { j } Z _ { j ^ { \prime } }$ (E2); and a Trotterized one-dimensional Heisenberg evolution for time $t = n / 3 ,$ with couplings set by the features, acting on a fixed Haar-random product state (E3). For each embedding we compute the fidelity quantum kernel $K _ { i j } = \mid  { \dot { \langle } { \phi ( x _ { i } ) } \vert } { \phi ( x _ { j } ) } \rangle  { \vert } ^ { 2 }$ and the projected quantum kernel built from the single-qubit reduced density matrices, $K _ { i j } = \exp [ - \gamma \sum _ { q } \| \rho _ { q } ( x _ { i } ) - \rho _ { q } ( x _ { j } ) \| _ { F } ^ { 2 } ]$ , together with its generalizations in which the reduced density matrices of all pairs (2-RDM) or all triples (3-RDM) of qubits enter the exponent. Each kernel is fed to a support vector machine with a precomputed Gram matrix. A penalty term C is scanned over an 18-point grid and the bandwidth over $\gamma \in \{ 0 . 2 5 , \hat { 0 } . 5 , 1 , 2 , \hat { 3 } , 4 , 5 , 2 0 \} / ( n \mathrm { V a r } ( X ) )$ (see Ref. $[ \mathrm { H B M ^ { + } } 2 1 ]$ for more details). Finally to compare against the performance of hardware runs (in Fig. 1 (b)), the best kernel results are selected based on their validation accuracy.

Simulating these baselines faithfully at 50 qubits requires embedding-specific methods, and not every embedding reaches every n. The E1 states are products of single-qubit states, so both kernels admit exact closed-form solutions. The E2 circuit generates volume-law entanglement through its all-to-all diagonal phases and admits no efficient tensor-network contraction, so it is simulated by exact statevector simulations, which bounds its reach by memory. The fidelity kernel is computed up to $n = 1 6 ,$ and the projected kernels up to $n = 2 4 ( n = 2 0$ at 2-RDM order). The E3 evolution also generates extensive entanglement, and hence its exact matrix-product-state bond dimension grows as $2 ^ { \left\lceil n / 2 \right\rceil }$ . We simulate it by time-evolving-block-decimation with 20 Trotter steps for the total time $t = n / 3$ , capping the bond dimension as $\chi = \operatorname* { m i n } ( 2 5 6 , 2 ^ { \lceil n / 2 \rceil } )$ ). With this choice, the MPS can represent exactly any quantum state up to $n \leq 1 6 ,$ and the only approximation error comes from the finite Trotter step, which we measured at $n = 1 6$ as a projected-kernel feature error of $1 . 6 \times 1 0 ^ { - 2 }$ . We do not evaluate E3 beyond $n = 1 6 ,$ where $\chi = 2 5 6$ would cease to be exact. We note that the classical hardness of simulating the E2 and E3 embeddings at large n does not, by itself, confer a learning advantage, as the corresponding kernels are precisely the ones that concentrate and fail on this task.

For the classical baseline, we consider support vector machines with radial basis function and linear kernels, random forests, gradient boosting, adaptive boosting, and a multilayer perceptron. We use the same n-dimensional features for all methods. The hyperparameters for these methods follow the same grids of [HBM<sup>+</sup>21]. For each model class, hyperparameters are tuned by three-fold cross-validation on the training split. The selected model is then evaluated once on the test split.

## Appendix J: Hardware experiments

## 1. The hardware solver

The hardware runs replace only the sampling stage of the solver of Appendix H 2 by a quantum device, whereas the construction of $H _ { T . }$ , the subspace union, the projection, the diagonalization, the gradient and the Adam update are performed classically on a laptop. At iteration t the current $\bar { H _ { T } } ( \alpha _ { t } )$ is converted into a Pauli operator and the circuits

$$
| \mathrm { r e f } \rangle \longmapsto \left( e ^ { - i \delta t \widetilde { H } _ { T } ( \alpha _ { t } ) } \right) ^ { k } | \mathrm { r e f } \rangle , \qquad k = 0 , \dots , k _ { \mathrm { m a x } } ,\tag{J1}
$$

are built by first-order Lie–Trotter product formulas, with the step $\delta t = \pi / ( 2 \sum _ { i } \left| c _ { i } \right| )$ of Appendix H 2, and measured in the computational basis. Here $\widetilde { H } _ { \mathcal { T } }$ is a pruned version of $H _ { \mathcal { T } }$ keeping only its $n _ { P }$ Pauli terms of largest coefficient magnitude (identity excluded). The pruned Hamiltonian $\widetilde { H } _ { \mathcal { T } }$ is determined as follows. First, Pauli terms with coefficients $\left| c _ { i } \right| < 1 \dot { 0 } ^ { - 3 } \operatorname* { m a x } _ { j } \left| c _ { j } \right|$ are dropped from the set of operators in the expansion. Next, we keep the $n _ { P } = 4 0 0$ Pauli terms of largest magnitude and assess whether the deepest transpiled circuit has at most 4,000 two-qubit gates. We iteratively reduce $n _ { P }$ by factors of 0.6 until this is the case. The size of this set of Pauli terms $n _ { P }$ is kept fixed for the remaining training epochs. The resulting value of $n _ { P }$ is 240 for the 50-qubit hardware run. For comparison, the full $H _ { \mathcal { T } }$ contains over 20,000 terms. Note that the pruning applies only to the evolution operator, while the projection uses the full $H \tau$

At the first iteration the reference is $\left| 0 ^ { n } \right.$ and the thirteen circuits $k \in \{ 0 , \ldots , 1 2 \}$ are measured with 25,000 shots each. At later iterations the references are the four highest-weight bitstrings of the previous ground state, the circuits $k \in \{ 1 , 2 , 3 \}$ are measured for each reference (12 circuits at $8 , 1 9 2$ shots), and the entire previous subspace is added to the union, so that the subspace can follow the training trajectory of the Hamiltonian.

Circuits are mapped to a chain of n physical qubits chosen from the calibration data at run time, to minimize the summed two-qubit-gate and readout errors along the chain by a greedy search over the coupling graph. These circuits are then executed with dynamical decoupling and measurement twirling enabled; no other error mitigation is used.

## 2. Model training insights

For every hardware run we compare the quantumly trained HE model to the classical iterative solver applied to the same data cell (Fig. 5). In all five runs, the accuracy of the hardware runs is greater, by +0.004 to +0.050 (mean +0.023).

Finally, to test whether the seed-to-seed variation reflects genuine differences in learned models or merely finite sample noise in the 240-example test set, we perform a cross-seed transfer analysis in simulation with the classical iterative solver. We evaluate twenty independently trained cells on all twenty held-out test splits $( \operatorname { F i g } . 6 )$ . This allows us to compute the seed-to-seed variation for the model initialization (indexed by rows) and the data splits (indexed by columns). The near-uniformity of the map shows that the accuracy barely changes when a model is applied to a different data sample. Both the native (diagonal) and transfer (off-diagonal) accuracies average 0.885. Thus, a model trained on one data sample performs just as well on an independent sample, confirming that HE learns transferable, sample-independent structure from the data.

![](images/ba85d0812b20ce6002d76dd41a0ddf57bc5bbb778050b8e1d725797844206abd.jpg)  
Figure 5. A comparison of the test accuracy from the Hamiltonian embedding model, when used in conjunction with a classical iterative solver (blue circles) and the SKQD method on hardware (orange squares) for the dataset of Fig. 1(b).

![](images/a764a93be2d55c7b5878589664ce7ba2e1bd14c02f50683ce7ce46aa191c6568.jpg)  
Figure 6. Cross-seed transfer at $n = 5 0$ . Each cell is the test accuracy of a model trained on one seed (row) and evaluated on another seed’s held-out data split (column); the red boxes mark the diagonal, where a model is tested on its own split.

## Appendix K: A numerical counterexample to Pauli classical surrogate models

In Appendix $G ,$ we presented a learning task for which the Hamiltonian embedding model provably requires fewer training examples than the corresponding classically regularized surrogate. Here, we present a different example as a proof of principle, numerically showing that retaining the quantum state during learning yields a substantial performance advantage over the corresponding Pauli-based classical surrogate.

Specifically, we compare the performance of a particular Hamiltonian embedding model, defined by a fixed choice of encoding unitary $\hat { U ( x , \alpha ) }$ and observable $O ( y )$ , with that of the corresponding Pauli classical surrogate. The latter takes the form

$$
g _ { \alpha } ( x ) = \sum _ { i } \phi _ { i } ( x , \alpha ) \widetilde { m } _ { i } ( \alpha ) ,\tag{K1}
$$

where the coefficients $\phi _ { i } ( x , \alpha )$ are defined through the Pauli decomposition

$$
U ^ { \dagger } ( x , \alpha ) A U ( x , \alpha ) = \sum _ { i = 1 } ^ { K } \phi _ { i } ( x , \alpha ) P _ { i } ,\tag{K2}
$$

where $\begin{array} { r } { O ( y ) = \frac { I + ( - 1 ) ^ { y } A } { 2 } } \end{array}$ . Here, $P _ { i }$ are the $K = { \mathrm { p o l y } } ( n )$ Pauli operators whose expectation values are measured by the Hamiltonian embedding model as explained in Section III B 2, while $\widetilde { m } _ { i } ( \alpha ) \in \mathbb { R }$ are unconstrained classical parameters replacing the physical quantum moments $m _ { i } ( \alpha ) = \langle \psi _ { T , \alpha } | P _ { i } | \psi _ { T , \alpha } \rangle$ for the quantum model in $\operatorname { E q . }$ (45). Thus, the two models use the same input-dependent parameterized Pauli feature map, but differ in that the coefficients of the Hamiltonian embedding model must arise as expectation values of a single physical quantum state, whereas those of the classical surrogate are optimized freely. Schematically,

$$
f _ { H E } ( x ) = \sum _ { i } \phi _ { i } ( x , \alpha ) m _ { i } ( \alpha ) ,
$$

$$
m _ { i } ( \alpha ) = \langle \psi _ { \mathcal { T } , \alpha } \vert P _ { i } \vert \psi _ { \mathcal { T } , \alpha } \rangle ,\tag{K3}
$$

$$
g _ { \mathrm { r i d g e } } ( x ) = \sum _ { i } \phi _ { i } ( x , \alpha ) { \tilde { m } } _ { i } ( \alpha ) ,
$$

$$
\tilde { m } _ { i } ( \boldsymbol { \alpha } ) \in \mathbb { R } .\tag{K4}
$$

For a fixed value of the trainable parameter vector $\alpha ,$ the coefficients of the classical surrogate are obtained by solving the ridge regression problem

$$
\tilde { m } ^ { \star } ( \alpha ) = \arg \operatorname* { m i n } _ { \tilde { m } } \left[ \frac 1 { 2 L } \sum _ { \ell = 1 } ^ { L } \left( ( - 1 ) ^ { y _ { \ell } } - \phi ( x _ { \ell } , \alpha ) ^ { T } \tilde { m } \right) ^ { 2 } + \frac { \lambda } { 2 } \| \tilde { m } \| _ { 2 } ^ { 2 } \right] .\tag{K5}
$$

Since the feature map $\phi ( x , \alpha )$ itself depends on $\alpha ,$ the optimal vector $\tilde { m } ^ { \star }$ depends implicitly on the choice of the representation parameters. The corresponding loss is then minimized with respect to the trainable parameters α.

## 1. Construction of the learning task

The task is constructed such that the requirement that the Pauli moments $m _ { i } ( \alpha )$ be jointly compatible with a physical quantum state provides a useful inductive bias in the low-data regime. To emphasize this effect, we consider a teacher–student setting in which the inputs x are labeled according to

$$
y = { \frac { 1 - s ( x ) } { 2 } } \qquad s ( x ) = \mathrm { s i g n } \Big [ \langle \psi ^ { \star } \mid U ^ { \dagger } ( x , \alpha ^ { \star } ) A U ( x , \alpha ^ { \star } ) \mid \psi ^ { \star } \rangle \Big ] ,\tag{K6}
$$

for a fixed teacher parameter $\alpha ^ { \star }$ and teacher state $| \psi ^ { \star } \rangle$ . We then use the same parameterized unitary $U ( x , \alpha )$ in the dataset Hamiltonian, with α initialized randomly. The goal of the Hamiltonian embedding model is therefore to learn parameters α and the corresponding ground state $\vert \bar { \psi } _ { \mathcal { T } , \alpha } \rangle$ such that the prediction rule in Eq. (5) reproduces the teacher labels.

In particular, we choose the following setting. We consider a system of $n = 2 M$ qubits, grouped into M two-qubit blocks $( 2 j - 1 , 2 j )$ for $j \in \{ 1 , . . , M \}$ . The teacher state $\left| \psi ^ { \star } \right.$ is the product of Bell states on each block

$$
\vert \psi ^ { \star } \rangle = \vert \Phi ^ { + } \rangle ^ { \otimes M } , \qquad \vert \Phi ^ { + } \rangle = { \frac { \vert 0 0 \rangle + \vert 1 1 \rangle } { \sqrt { 2 } } } .\tag{K7}
$$

The inputs x are chosen to be a tuple of four different variables $x = ( j , \theta , \beta , q )$ with different roles:

• j selects the active two-qubit block and it is sampled uniformly at random in $j \in \{ 1 , . . . , M \}$ ;

• θ is sampled from a two-pole Gaussian mixture centered at 0 and $\pi ,$

$$
\theta \sim \frac { 1 } { 2 } \mathcal { N } ( 0 , \sigma _ { \theta } ^ { 2 } ) + \frac { 1 } { 2 } \mathcal { N } ( \pi , \sigma _ { \theta } ^ { 2 } ) , \qquad \sigma _ { \theta } = 0 . 1 2 ,\tag{K8}
$$

where ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ denotes a normal distribution with mean $\mu$ and variance $\sigma ^ { 2 }$

$\beta \sim \mathrm { U n i f } [ - \pi , \pi ] ;$

$q = ( q _ { 0 } , q _ { 1 } , q _ { 2 } , q _ { 3 } )$ is a random unit vector such that $q _ { 0 } ^ { 2 } + q _ { 1 } ^ { 2 } + q _ { 2 } ^ { 2 } + q _ { 3 } ^ { 2 } = 1$ . It parametrizes a random single-qubit unitary $\dot { V } ( q ) \in \dot { S } U ( 2 )$ , which is applied to the active two-qubit block through $V ( q ) \otimes V ( q ) ^ { * }$

The label-dependent observable is defined as

$$
O ( y , j ) = \frac { I + ( - 1 ) ^ { y } A _ { j } } { 2 } , \qquad A = \frac { 2 } { 3 } X \otimes X - \frac { 1 } { 3 } Y \otimes Y .\tag{K9}
$$

and $A _ { j }$ denotes the operator A acting on the block $( 2 j - 1 , 2 j )$ and identity elsewhere.

The encoding unitary acts nontrivially only on the two-qubit block selected by the input index j. Specifically,

$$
U ( x = ( j , \theta , \beta , q ) , \alpha ) = I ^ { \otimes 2 ( j - 1 ) } \otimes U _ { \mathrm { l o c } } ( \theta , \beta , q ; \alpha _ { j } ) \otimes I ^ { \otimes 2 ( M - j ) } ,\tag{K10}
$$

where

$$
U _ { \mathrm { l o c } } ( \theta , \beta , q ; \alpha _ { j } ) = C ( \theta ) \left[ R _ { n ( \theta ) } \big ( \beta + \alpha _ { j } h ( \theta ) \big ) \otimes I \right] S ( q ) ,\tag{K11}
$$

with

$$
C ( \theta ) = R _ { Y } ( \theta / 2 ) \otimes R _ { Y } ( - \theta / 2 ) ,
$$

$$
\pmb { n } ( \theta ) = \left( \sin ( \theta / 2 ) , 0 , \cos ( \theta / 2 ) \right) ,\tag{K12}
$$

$$
h ( \theta ) = \sin ^ { 2 } ( \theta / 2 ) ,\tag{K13}
$$

$$
S ( q ) = V ( q ) \otimes V ( q ) ^ { * } ,\tag{K14}
$$

(K15)

and

$$
V ( q ) = q _ { 0 } I - i \left( q _ { 1 } X + q _ { 2 } Y + q _ { 3 } Z \right) .\tag{K16}
$$

We note that the unitary $S ( q ) = V ( q ) \otimes V ( q ) ^ { * }$ has the Bell state $| \Phi ^ { + } \rangle$ as eigenstate. In particular $\left( V \otimes V ^ { * } \right) \left| \Phi ^ { + } \right. =$ $| \Phi ^ { + } \rangle$ for any single-qubit unitary ${ \dot { V } } .$ Therefore q can rotate the Pauli representation seen by the learner while leaving the Bell teacher prediction unchanged. Finally, we also note that the local encoding $U _ { \mathrm { l o c } } ( \theta , \beta , q ; \alpha _ { j } )$ can be implemented entirely using standard single-qubit rotation gates. In particular, $R _ { n ( \theta ) } ( \varphi )$ can be decomposed as $R _ { Y } ( \theta / 2 ) R _ { Z } ( \varphi ) R _ { Y } ( - \theta / 2 )$ , while $V ( q )$ is itself a single-qubit rotation parameterized by the quaternion $q .$

We then label the data by the rule in $\operatorname { E q } .$ (K6), with the choice of $\alpha _ { j } ^ { \star } \ = \ 0$ for all $j \in \overline { { \{ 1 , . . . , M \} } }$ . The teacher corresponding expectation value is then

$$
\langle \Phi ^ { + } | U _ { \mathrm { l o c } } ( \theta , \beta , q ; \alpha ^ { \star } ) ^ { \dagger } A _ { j } U _ { \mathrm { l o c } } ( \theta , \beta , q ; \alpha ^ { \star } ) | \Phi ^ { + } \rangle = { \frac { 1 + 2 \cos \theta } { 3 } } \cos \beta .\tag{K17}
$$

We note that, although the observable $A _ { j }$ explicitly depends on the input index $j ,$ the same construction can be written with a fixed observable by absorbing this dependence into the encoding unitary. Choosing $A _ { 1 }$ as the fixed observable and letting $P _ { j }$ swap the pair $( 2 j - 1 , 2 j )$ with (1, 2), so that $P _ { j } ^ { \dagger } A _ { 1 } P _ { j } = A _ { j } .$ , we can define $\widetilde { U } ( x , \alpha ) ~ =$ $U _ { \mathrm { l o c } } ^ { ( 1 , 2 ) } ( \theta , \beta , q ; \alpha _ { j } ) P _ { j }$ , which gives $\widetilde { U } ^ { \dagger } ( x , \alpha ) A _ { 1 } \widetilde { U } ( x , \alpha ) = P _ { j } ^ { \dagger } U _ { \mathrm { l o c } } ^ { ( 1 , 2 ) \dagger } A _ { 1 } U _ { \mathrm { l o c } } ^ { ( 1 , 2 ) } P _ { j }$ and therefore reproduces the original $j -$ dependent construction.

From the labeling rule in Eq. (K6) and Eq. (K17), the labels are then given by Eq. (K6) with

$$
s = \operatorname { s i g n } [ ( 1 + 2 \cos \theta ) \cos \beta ] .\tag{K18}
$$

The nuisance variable q and the block index j do not appear in the labeling rule, but they make the learning problem for the classical surrogate more demanding. In particular, the index j determines which two-qubit block is probed by each example, with each block associated with its own vector of Pauli coefficients $\widetilde { m } _ { j }$ in the classical surrogate. On the other hand, $q$ changes the Pauli decomposition of the encoded observable. Indeed, conjugation by the random unitary $V ( q )$ rotates Pauli operators into different linear combinations of $X , Y ,$ and $Z ,$ so that different values of q make different examples probe different combinations of the Pauli coefficients. In the low-data regime, the training set may therefore be compatible with many choices of the free surrogate coefficients ${ \widetilde { m } } ,$ which can fit the observed data while generalizing differently to unseen inputs. In contrast, the Hamiltonian embedding model restricts the coefficients m to be jointly compatible with the Pauli moments of a physical quantum state. This provides an ad ditional inductive bias that favors coefficient vectors consistent with the teacher construction and, in our numerical experiment, leads to better generalization before the classical surrogate has received enough data to identify a wellgeneralizing solution.

![](images/b199be5dca27e4ea592cac7e4369094d4a49d3c5fbc4ffc9d91cf105c5bf86bb.jpg)  
Figure 7. Training examples needed to reach 90% test accuracy for Hamiltonian embedding and classical surrogate with $\lambda = 0 . 1$ The sweep uses 20 independent seeds. Central values are the first crossings of the mean test-accuracy curve across the 20 seeds, while the error bars are the standard deviations of the seed-wise first-crossing values. The bars therefore show seed-to-seed variability rather than a conventional confidence interval.

Table I. First tested training size L for which the mean test accuracy over all seeds reaches 90%. The λ sweep uses 20 independent seeds and 5000 test points per run.
<table><tr><td>Model / λ</td><td>n = 4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td></tr><tr><td>HE</td><td>56</td><td>72</td><td>96</td><td>140</td><td>144</td><td>168</td><td>192</td><td>216</td><td>240</td></tr><tr><td>10-8</td><td>160</td><td>216</td><td>320</td><td>400</td><td>432</td><td>560</td><td>640</td><td>648</td><td>800</td></tr><tr><td>10-7</td><td>160</td><td>216</td><td>320</td><td>400</td><td>432</td><td>560</td><td>640</td><td>648</td><td>800</td></tr><tr><td>10-6</td><td>160</td><td>216</td><td>320</td><td>400</td><td>432</td><td>560</td><td>640</td><td>648</td><td>800</td></tr><tr><td>10-5</td><td>160</td><td>216</td><td>320</td><td>400</td><td>432</td><td>560</td><td>640</td><td>648</td><td>800</td></tr><tr><td>10-4</td><td>160</td><td>216</td><td>320</td><td>400</td><td>432</td><td>560</td><td>640</td><td>648</td><td>800</td></tr><tr><td>10-3</td><td>160</td><td>216</td><td>320</td><td>400</td><td>432</td><td>560</td><td>640</td><td>648</td><td>800</td></tr><tr><td>10-2</td><td>128</td><td>216</td><td>288</td><td>360</td><td>432</td><td>448</td><td>576</td><td>648</td><td>720</td></tr><tr><td>10-1</td><td>128</td><td>216</td><td>256</td><td>320</td><td>384</td><td>448</td><td>512</td><td>576</td><td>640</td></tr><tr><td>1</td><td>144</td><td>240</td><td>256</td><td>360</td><td>432</td><td>448</td><td>640</td><td>576</td><td>640</td></tr></table>

## 2. Numerical experiment

We compare the Hamiltonian embedding model and the Pauli classical surrogate for systems ranging from 4 to 20 qubits, with $n \in \{ 4 , 6 , 8 , 1 0 , 1 2 , 1 4 , 1 6 , 1 8 , 2 0 \}$ , using 20 independent random seeds and 5000 fresh test examples per run. For each system size, we vary the number of training examples supplied to both models. Specifically, for $\mathbf { \hat { \boldsymbol { M } } } = n / 2$ two-qubit pairs, the total training-set size is chosen as $L \ = \ M s ,$ , where s is varied over a fixed grid of samples per pair. For each (n, L, seed), we also vary the strength λ of the regularization term in the ridge regression of Eq. (K5). We then evaluate the two models on the same training/test data and with the same initial random α.

As a common metric, for each model and system size we define $L _ { 9 0 }$ as the smallest training-set size L for which the mean test accuracy over the 20 random seeds reaches 90%. This quantity is plotted in Fig. 7. Because L is evaluated on a discrete grid, the reported values are first grid crossings. To quantify seed-to-seed variability, for each seed r we also define the first seed-wise crossing.

The error bars used in Fig. 7 are the standard deviation of these 20 seed-wise first-crossing values. The plotted central point is the smallest training-set size L for which the mean accuracy over the random seeds reaches 90%. Therefore the bars should be interpreted as a diagnostic of run-to-run variability, rather than as a confidence interval around the central point.

The complete results are shown in Table I. The first row is the Hamiltonian embedding model. The remaining

Table II. Hamiltonian embedding compared with the best ridge threshold observed over the tested λ grid. The final column gives the ridge-to-HE ratio averaged over system sizes.
<table><tr><td></td><td>4</td><td>6</td><td>8</td><td>10</td><td>12</td><td>14</td><td>16</td><td>18</td><td>20</td><td>mean ratio</td></tr><tr><td> $\overline { { L _ { 9 0 } ^ { \mathrm { H E } } } }$ </td><td>56</td><td>72</td><td>96</td><td>140</td><td>144</td><td>168</td><td>192</td><td>216</td><td>240</td><td></td></tr><tr><td> $L _ { 9 0 } ^ { \mathrm { r i d g e , b e s t } }$ </td><td>128</td><td>216</td><td>256</td><td>320</td><td>384</td><td>448</td><td>512</td><td>576</td><td>640</td><td>2.62×</td></tr></table>

rows give the classical surrogate $L _ { 9 0 }$ for each ridge regularization value.

The weakly regularized regime, from $1 0 ^ { - 8 }$ through $1 0 ^ { - 3 }$ , gives the same first-crossing thresholds on the tested grid. Moderate regularization improves ridge regression. The strongest improvement is typically around $\lambda = 1 0 ^ { - 1 }$ although the optimum is not identical at every system size.

To make the comparison maximally favorable to the classical surrogate, we also report the comparison between the $L 9 0$ value for the Hamiltonian embedding and the classical surrogate at the best regularization strength $\lambda ,$ for each system size, i.e. $\begin{array} { r } { L _ { 9 0 } ^ { \mathrm { r i d g e , b e s t } } ( n ) = \operatorname* { m i n } _ { \lambda \in \Lambda } L _ { 9 0 } ^ { \mathrm { r i d g e } } ( n , \lambda ) } \end{array}$ , where Λ is the tested grid. As the best performance for the classical surrogate is found at $\lambda = 0 . 1$ for almost all system sizes, we plot the comparison with the Hamiltonian embedding model in Fig. 7.

Finally, we emphasize that the labeling rule in $\operatorname { E q } .$ (K18) is deliberately simple, and a classical learner provided with additional structural information could easily solve the learning task. Nevertheless, the example serves to demonstrate that there exist settings in which the physical bias imposed by requiring the Pauli moments $m _ { i }$ to arise from a valid quantum state can improve the performance of the Hamiltonian embedding model relative to the corresponding classical surrogate.