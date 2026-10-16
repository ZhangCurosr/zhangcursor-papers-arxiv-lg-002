# The importance of tensor handling and implementation of physics-informed DeepONets

Nahum Banks Faculty of Engineering and Physical Sciences University of Southampton Southampton, UK nab1g24@soton.ac.uk

Gustavo A.M. de Almeida Faculty of Engineering and Physical Sciences University of Southampton Southampton, UK

Sergio Maldonado Faculty of Engineering and Physical Sciences University of Southampton Southampton, UK s.maldonado@soton.ac.uk

October 16, 2026

## Abstract

Physics-informed neural networks (PINNs) reduce or eliminate the need for large labelled datasets by enforcing governing equations during training. Physics-informed Deep Operator Networks (PI-DeepONets) extend this approach to learning solution operators, enabling rapid prediction across families of problems. Despite these advantages, we show that their application to higher-dimensional problems remains limited by the computational cost of training. To address this issue, we propose a modified PI-DeepONet architecture that reduces this cost by removing redundancy in how the Branch network processes input func tions across collocation points. The proposed formulation is mathematically equivalent to the original architecture but reduces input tensor sizes and the number of Branch network evaluations. We analyse the computational complexity of both architectures and derive theoretical scaling laws for training time with respect to the number of collocation points and input functions. The proposed architecture is evaluated on 1D, 2D, and 3D linear advection problems. In all cases, it achieves comparable predictive accuracy to the original PI-DeepONet while consistently reducing training time. In 2D experiments, a maximum speed-up of 3.75× is observed, with empirical training-time scaling agreeing with the predicted linear dependence on collocation points and input functions. Furthermore, training a 3D model on an NVIDIA A100 GPU is only feasible using the proposed architecture, as the original formulation exceeds memory limits. These results demonstrate that careful consideration of how PI-DeepONets are implemented can yield substantial gains, enhanc ing the practical performance of PI-DeepONets and enabling their application to otherwise intractable physics-informed learning problems.

Keywords Scientific machine learning, Physics-informed, Neural operator, Computational modelling

## 1 Introduction

Scientific Machine Learning (SciML) aims to make use of the vast amounts of prior knowledge that is often available about physical systems when training machine learning models [1, 2]. The seminal work by [3] introduced Physics-Informed Neural Networks (PINNs), which leverage automatic diferentiation to compute derivatives appearing in the governing equations. PINNs encode the underlying Partial Diferential

Equations (PDEs) governing a physical system, alongside the boundary and initial conditions directly into the loss function, reducing or even eliminating the requirements for labelled training data. Embedding physical constraints in the loss function greatly reduces the acceptable solution space without the need for labelled data [4]. In recent years, there has been an abundance of work applying PINNs, ranging from problems in fluid mechanics [5, 6, 7, 8, 9, 10, 11], molecular imaging [12], solid mechanics [13, 14, 15], and learning governing equations from data [16, 17, 15, 18], to cite only a few.

Despite their success, a critical limitation for the real-world application of PINNs for forward modelling is that they typically require retraining even for small changes to the domain, initial conditions, or — perhaps most importantly — boundary conditions. This stems from the fact that a conventional PINN takes only the independent variables of the PDE as input into the neural network, while the initial and boundary conditions are incorporated indirectly as soft constraints in the loss function. Consequently, the network is trained to output a solution corresponding to a fixed set of conditions, and does not naturally generalise to diferent initial or boundary conditions [19, 11]. As a result, the fast inference times of trained PINNs cannot be easily leveraged across diferent problem setups, limiting their practical utility in dynamic or rapidly changing scenarios [10].

A possible solution for this limitation is to use physics informed neural operators. Neural operators aim to learn mappings between infinite-dimensional function spaces [20]; for example, a mapping from boundary conditions to the PDE’s solution across the whole spatiotemporal domain. One notable neural operator framework is the Deep Operator Network (DeepONet) [21], and its physics-informed variant, the PI-DeepONet [22].

DeepONet is grounded in the universal operator approximation theorem, which states that a neural network with a single hidden layer can approximate any non-linear continuous operator G to arbitrary accuracy [23]. The learned operator approximation $\hat { G }$ maps an input function u to an output function $\hat { G } ( u )$ , which can then be evaluated at collocation/query points y to produce $\hat { G } ( u ) ( y )$ . [21] extended this framework to use deep neural networks in their unstacked DeepONet. In a DeepONet a single Branch network encodes the input function at m sensor points $\xi _ { j \in [ 1 , m ] }$ , while a single Trunk network encodes the collocation/query point $y = ( \mathbf { x } , t )$ . For example, the input layer of the Branch network could depict a time-dependent boundary condition, whereas the Trunk network functions similarly to a conventional PINN, taking points within the spatiotemporal domain as input. The outputs of these subnetworks are then combined via a dot product to approximate the output function evaluated at the query point. This can be represented by the equation:

$$
\hat { G } ( u ) ( y ) \approx \sum _ { k = 1 } ^ { s } b _ { k } ( u ( \xi _ { 1 } ) , u ( \xi _ { 2 } ) , . . . , u ( \xi _ { m } ) ) \tau _ { k } ( y ) ,\tag{1}
$$

where, $b _ { k }$ are the Branch network outputs, $\tau _ { k }$ are the Trunk network outputs, both of which are of length s. The subnetworks require no specific architecture, allowing flexible integration of diferent networks like feedforward, convolutional, etc.

Owing to the simple, feedforward architecture of the subnetworks in the DeepONet, it is straightforward to integrate physics-informed techniques into DeepONet designs. A PI-DeepONet is trained using a loss function formulated in the same way as in PINNs [22]. A schematic of a PI-DeepONet can be seen in Figure 1. Derivative terms in the physics loss are calculated by automatic diferentiation along the Trunk network, which encodes the collocation points of the output function space. The PI-DeepONet is especially attractive in scenarios where simulations are expensive, high-fidelity data are limited, and adherence to physical laws is critical for model reliability [24]. Such scenarios include real-time forecasting, uncertainty quantification, and digital twins. This is because the PI-DeepONet can be trained without labelled data, and can then eficiently generate predictions across a variety of problem configurations.

Overall, PI-DeepONet is one of the first physics-informed neural operator frameworks and has received substantial attention since its introduction [22], introducing a general approach that allows purely physics informed operator learning, but broader adoption has been limited compared to data-driven neural operators. This is partly due to the extra computational burden of using automatic diferentiation to calculate derivatives in the physics loss terms, which becomes more apparent for high-dimensional systems and when higher order derivatives are present. The high computational cost stems from the fact the number of Trunk eval uations scales exponentially with the number of dimensions $( \mathrm { i . e . } ~ N ^ { d } )$ , where N is the is the resolution of each coordinate axis and d is the number of spatiotemporal dimensions [25]. As a result, most research on PI-DeepONet to date has focused on low-dimensional problems, using hybrid approaches where physics constraints are combined with labelled data [26, 27, 28].

![](images/80f2dc021131a1121fa00f1081bfcea66169988e937920574e80ff409eb4d0af.jpg)  
Figure 1: Schematic diagram of a physics-informed Deep Operator Network (PI-DeepONet). u is the input function discretised at m sensor points, and y is the query location for the output function of dimensionality d. Derivatives of the output with respect to the input coordinates are calculated via automatic diferentiation; these operations are denoted by $\partial _ { x } ^ { ( o ) }$ and $\partial _ { t } ^ { ( o ) }$ , where o is the order of derivative. $\mathcal { L } _ { \mathrm { P D E } } , \mathcal { L } _ { \mathrm { B C } } , \mathcal { L } _ { \mathrm { I C } }$ , and ${ \mathcal { L } } _ { \mathrm { d a t a } }$ denote the PDE, boundary condition, initial condition, and labelled data loss terms, respectively.

Recent works have tried to overcome this hurdle by incorporating a separation-of-variables approach to help PINNs and PI-DeepONets scale to higher-dimensional problems [29, 25]. In the Separable PI-DeepONet, separate Trunk networks are employed to encode each coordinate axis (i.e. independent variable of the PDE). This PI-DeepONet structure reduces the number of Trunk network evaluations from $N ^ { d }$ to Nd, making it favourable for high-dimensional problems. However, this method achieves the best performance when the underlying physical system is compatible with a separation-of-variables approach, somewhat limiting the general applicability of the PI-DeepONet framework — one of its most compelling advantages. To the best of the authors’ knowledge, the work by [25] is the only so far that has aimed to reduce the computational expense of training PI-DeepONets.

This work investigates an alternative Accelerated Tensor Handling (ATenH) PI-DeepONet architecture which aims to increase its computational eficiency. This architecture takes inspiration from those suggested in [30], and applies it in the physics-informed setting.

The main objectives of this paper are summarised as follows:

1. To introduce an alternative PI-DeepONet architecture, hereafter referred to as ATenH, that reduces the computational cost of training and inference.

2. To derive the theoretical computational speed-ups achieved by ATenH relative to the original PI-DeepONet architecture (Vanilla).

3. To evaluate the performance of ATenH through a set of increasingly challenging numerical experiments.

4. To demonstrate that extending PI-DeepONets to higher-dimensional problems presents significant computational challenges, and that architectural improvements such as ATenH are essential for making some problems computationally tractable.

In doing so, we demonstrate that these architectural modifications enable PI-DeepONets to be applied to higher-dimensional PDE systems at greater spatial and temporal resolutions than was previously possible. All numerical results for the Vanilla and ATenH architectures are obtained using PyTorch, demonstrating that these eficiency gains can be realised within standard, widely used, deep learning frameworks.

The remainder of this paper is structured as follows. Section 2.1 outlines the PI-DeepONet framework for operator learning and identifies important issues responsible for their limited computational performance; Sections 2.2 and 2.3 detail the diferent PI-DeepONet architectures investigated in this research; Section 2.4 derives some theoretical scaling laws for the diferent architectures; Section 3 reports training time and accuracy of the Vanilla and ATenH PI-DeepONets for linear advection problems for progressively increasing spatial dimensions; Section 4 provides a discussion of key findings, and Section 5 provides concluding remarks.

## Notation

Table 1: Glossary of key variables used throughout this work.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>n</td><td>Number of input functions used during training</td></tr><tr><td> $p$ </td><td>Number of collocation points per input function</td></tr><tr><td> $m$ </td><td>Number of sensor (discretisation) points for input functions</td></tr><tr><td> $d$ </td><td>Dimension of the spatiotemporal domain</td></tr><tr><td> $s$   $\mathbf { u } ^ { ( j ) }$ </td><td>Latent dimension (output size of Branch/Trunk networks)</td></tr><tr><td> $\xi$ </td><td>jth input function Sensor point at which input functions are sampled</td></tr><tr><td> $\mathbf { y } = ( \mathbf { x } , t )$ </td><td></td></tr><tr><td></td><td>Collocation point in the spatiotemporal domain</td></tr><tr><td> $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { d - 1 } )$ </td><td>Spatial coordinate</td></tr><tr><td> $\mathbf { c } = \left( c _ { 1 } , \ldots , c _ { d - 1 } \right)$ </td><td>Advection velocity vector</td></tr><tr><td> $r _ { l }$ </td><td>Number of neurons in layer l (Branch network)</td></tr><tr><td> $e _ { l }$ </td><td>Number of neurons in layer l (Trunk network)</td></tr><tr><td> $L _ { B }$ </td><td>Number of layers in multilayer perceptron (Branch network)</td></tr><tr><td> $L _ { T }$ </td><td>Number of layers in multilayer perceptron (Trunk network)</td></tr><tr><td> $M$ </td><td>Number of Branch networks in PI-DeepONet</td></tr></table>

## 2 Methodology

## 2.1 General PI-DeepONet framework

The DeepONet is rendered physics-informed (PI-DeepONet) by encoding the PDEs, boundary and initial conditions directly into the loss term, which can take the following form:

$$
\mathcal { N } [ G ( u ) ( \mathbf { x } , t ) ] = 0 , \quad \mathbf { x } \in \Omega , \quad t \in ( 0 , T ]\tag{2}
$$

$$
\begin{array} { r } { \mathcal { B } [ G ( u ) ( \mathbf { x } , t ) ] = 0 , \quad \mathbf { x } \in \partial \Omega , \quad t \in ( 0 , T ] } \end{array}\tag{3}
$$

$$
\begin{array} { r } { G ( u ) ( \mathbf { x } , 0 ) = v _ { 0 } ( \mathbf { x } ) , \quad \mathbf { x } \in \Omega } \end{array}\tag{4}
$$

Here, $\mathcal { N } [ \cdot ]$ represents a general diferential operator, $B [ \cdot ]$ is a boundary operator, x is the spatial coordinate, t is the temporal coordinate, $v _ { 0 } ( \mathbf { x } )$ defines the initial condition, $\Omega \subset \bar { \mathbb { R } } ^ { d - 1 }$ denotes the spatial domain, and T denotes the upper limit of the temporal domain.

A composite loss function incorporates each of the conditions:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { { P D E } } } + \mathcal { L } _ { \mathrm { { B C } } } + \mathcal { L } _ { \mathrm { { I C } } } + \mathcal { L } _ { \mathrm { { d a t a } } } .\tag{5}
$$

Here, $\mathcal { L } _ { \mathrm { P D E } }$ enforces the diferential equation by minimising the residual of $\mathcal { N } [ G ( u ) ]$ at collocation points, $\left( \mathbf { x } _ { i } , t _ { i } \right)$ inside the spatiotemporal domain, $\mathcal { L } _ { \mathrm { B C } }$ penalises violations of boundary conditions at boundary collocation points, and $\mathcal { L } _ { \mathrm { I C } }$ enforces the initial condition at initial-time collocation points; ${ \mathcal { L } } _ { \mathrm { d a t a } }$ is an optional term that penalises the diference between the neural network’s prediction and available observation data. In all of our tests, ${ \mathcal { L } } _ { \mathrm { d a t a } }$ is not included, ensuring that training is labelled-data-free.

It is worth examining how input functions and collocation points are provided to the PI-DeepONet. The following architectural descriptions assume that the jth function $\mathbf { u } ^ { ( j ) }$ (where $1 \leq j \leq n )$ is discretised at m sensor points, and the corresponding output function is evaluated at $p ^ { ( j ) }$ diferent d−dimensional collocation points $\mathbf { y } _ { k \in [ 1 , p ] } ^ { ( j ) }$

## 2.2 Vanilla architecture

We first introduce the original PI-DeepONet architecture from [22]. In this architecture, the data triplet (Branch and Trunk network inputs, DeepONet output) for a specific input-output function mapping is given by the following expression:

$$
\begin{array} { r } { [ \mathbf { u } , \mathbf { y } , G ( \mathbf { u } ) ( \mathbf { y } ) ] = \left[ \left[ \begin{array} { c } { \mathbf { u } ^ { ( 1 ) } } \\ { \vdots } \\ { \mathbf { u } ^ { ( 1 ) } } \\ { \mathbf { u } ^ { ( 2 ) } } \\ { \vdots } \\ { \mathbf { u } ^ { ( n ) } } \end{array} \right] , \left[ \begin{array} { c } { \mathbf { y } _ { 1 } ^ { ( 1 ) } } \\ { \vdots } \\ { \mathbf { y } _ { p ^ { ( 1 ) } } ^ { ( 1 ) } } \\ { \mathbf { y } _ { 1 } ^ { ( 2 ) } } \\ { \vdots } \\ { \mathbf { y } _ { p ^ { ( n ) } } ^ { ( n ) } } \end{array} \right] , \left[ \begin{array} { c } { G ( \mathbf { u } ^ { ( 1 ) } ) ( \mathbf { y } _ { 1 } ^ { ( 1 ) } ) } \\ { \vdots } \\ { G ( \mathbf { u } ^ { ( 1 ) } ) ( \mathbf { y } _ { p ^ { ( 1 ) } } ^ { ( 1 ) } ) } \\ { G ( \mathbf { u } ^ { ( 2 ) } ) ( \mathbf { y } _ { 1 } ^ { ( 2 ) } ) } \\ { \vdots } \\ { G ( \mathbf { u } ^ { ( n ) } ) ( \mathbf { y } _ { p ^ { ( n ) } } ^ { ( n ) } ) } \end{array} \right] \right] . } \end{array}
$$

It is important to note that each input function $\mathbf { u } ^ { ( j ) }$ is repeated $p ^ { ( j ) }$ times to match the number of collocation points it is evaluated at, enabling a pointwise inner product between the Branch and Trunk network outputs. In the most general setting, $p ^ { ( j ) }$ may vary across input functions, allowing each input–output mapping to be evaluated on a distinct amount of collocation points. However, in the labelled-data-free setting considered here, it is natural to use the same number of collocation points across all input functions, such that $p ^ { ( j ) } = p$ for all j. As a result, the Branch network input shape is $( \boldsymbol { n } \times \boldsymbol { p } , m )$ , the Trunk network input shape is $( n \times p , d )$ , and the PI-DeepONet’s final output has shape $( n \times p , 1 ) ;$ ; that is, every output function (which is a mapping of its corresponding input function) evaluated at each of its corresponding collocation points. The Vanilla architecture provides the greatest flexibility, as any number of collocation points can be used with each input function. Pseudocode for the Vanilla architecture can be seen in Algorithm 1.

```latex
Algorithm 1 Vanilla PI-DeepONet Training
Require: Input functions $\{ \mathbf { u } ^ { ( j ) } \} _ { j = 1 } ^ { n } .$ , collocation points $\{ \mathbf { y } _ { k } \} _ { k = 1 } ^ { p }$
Ensure: Network outputs $\mathbf { G } ( \mathbf { u } ) \mathbf { \check { ( y ) } } \in \mathbb { R } ^ { n p }$
1: Repeat each input function $\mathbf { u } ^ { ( j ) }$ for all collocation points
repeat $( \mathbf { u } ^ { ( j ) } , p ) \stackrel { \cdot } {  } \mathbf { U } \in \mathbb { R } ^ { ( n p ) \times m }$
2: Tile collocation points for all input functions
$\mathrm { t i l e } ( \mathbf { y } _ { k } , n )  \mathbf { Y } \stackrel { ^ { \bullet } } { \in } \mathbb { R } ^ { ( n p ) \times d }$
3: Compute Branch network output
${ \cal B } ( { \bf U } )  {  } { \bf B } \in \mathbb { R } ^ { ( n p ) \times s }$
4: Compute Trunk network output
$\mathcal { T } ( \mathbf { Y } )  \mathbf { T } \in \mathbb { R } ^ { ( n p ) \times s }$
5: Compute final output via inner product
$\begin{array} { r } { G _ { i } = \sum _ { \ell = 1 } ^ { s } B _ { i \ell } T _ { i \ell } } \end{array}$
P6: Evaluate physics-informed loss
$\mathcal { L } = \mathcal { L } _ { \mathrm { P D E } } + \mathcal { L } _ { \mathrm { B C } } + \mathcal { L } _ { \mathrm { I C } }$
7: Update model parameters via gradient-based optimisation
Next, we introduce an alternative architecture, ATenH (Accelerated Tensor Handling), which contains a
modification that reduces the number of operations and the dataset sizes involved in a PI-DeepONet evalu
ation.
```

## 2.3 Accelerated Tensor Handling (ATenH) architecture

In the proposed alternative architecture, the number of Branch passes is reduced by a factor $p$ by re-using the output of a single Branch network pass for all corresponding collocation points. In particular, the input size of the Branch network becomes $( n , m )$ and the output of the Branch network becomes $( n , s )$ , which contrasts with $( n \times p , m ) \to ( n \times p , s )$ for the Vanilla architecture. Additionally, the input data for the Trunk network is reshaped as $( n , p , d ) \textrm { -- }$ rather than $( n \times p , d )$ as in the Vanilla architecture. This formulation requires a constant amount of collocation points across all input functions, a constraint not imposed by the Vanilla version. The data triplet for a specific input-output function mapping is given by:

$$
\begin{array} { r l } { [ \mathbf { u } , \mathbf { y } , G ( \mathbf { u } ) ( \mathbf { y } ) ] = } & { [ [ \begin{array} { c } { \mathbf { u } ^ { ( 1 ) } } \\ { \mathbf { u } ^ { ( 2 ) } } \\ { \vdots } \\ { \mathbf { u } ^ { ( n ) } } \end{array} ] , [ \begin{array} { c c c c } { \mathbf { y } _ { 1 } ^ { ( 1 ) } } & { \mathbf { y } _ { 2 } ^ { ( 1 ) } } & { \cdots } & { \mathbf { y } _ { p } ^ { ( 1 ) } } \\ { \mathbf { y } _ { 1 } ^ { ( 2 ) } } & { \mathbf { y } _ { 2 } ^ { ( 2 ) } } & { \cdots } & { \mathbf { y } _ { p } ^ { ( 2 ) } } \\ { \vdots } & { \vdots } & & { \vdots } \\ { \mathbf { y } _ { 1 } ^ { ( n ) } } & { \mathbf { y } _ { 2 } ^ { ( n ) } } & { \cdots } & { \mathbf { y } _ { p } ^ { ( n ) } } \end{array} ] , } \\ { [ G ( \mathbf { u } ^ { ( 1 ) } ) ( \mathbf { y } _ { 1 } ^ { ( 1 ) } ) } & { G ( \mathbf { u } ^ { ( 1 ) } ) ( \mathbf { y } _ { 2 } ^ { ( 1 ) } ) } & { \cdots } & { G ( \mathbf { u } ^ { ( 1 ) } ) ( \mathbf { y } _ { p } ^ { ( 1 ) } ) } \\ { G ( \mathbf { u } ^ { ( 2 ) } ) ( \mathbf { y } _ { 1 } ^ { ( 2 ) } ) } & { G ( \mathbf { u } ^ { ( 2 ) } ) ( \mathbf { y } _ { 2 } ^ { ( 2 ) } ) } & { \cdots } & { G ( \mathbf { u } ^ { ( 2 ) } ( \mathbf { y } _ { p } ^ { ( 2 ) } ) } \\ { \vdots } & { \vdots } & { \vdots } \\ { G ( \mathbf { u } ^ { ( n ) } ) ( \mathbf { y } _ { 1 } ^ { ( n ) } ) } & { G ( \mathbf { u } ^ { ( n ) } ) ( \mathbf { y } _ { 2 } ^ { ( n ) } ) } & { \cdots } &  G ( \mathbf { u } ^  ( n \end{array}
$$

It is important to note that the input function $\mathbf { u } ^ { ( j ) }$ is no longer repeated p times, as was the case with the Vanilla architecture. As a consequence, the inner product cannot be performed in the same manner due to the subnetworks’ output shapes no longer matching. Thus, the Branch network’s output must be broadcasted beforehand. This operation can be performed eficiently in many Python libraries using an einsum (Einstein summation) operation. This results in the PI-DeepONet output having the desired shape $( n , p , 1 )$ , which may be trivially converted to the shape $( n \times p , 1 )$ of the Vanilla architecture. Pseudocode for ATenH can be seen in Algorithm 2. A summary of the data shapes for the diferent architectures can be seen in Table 2.

Algorithm 2 ATenH (new proposal) PI-DeepONet Training   
Require: Input functions $\{ \mathbf { u } ^ { ( j ) } \} _ { j = 1 } ^ { n }$ , collocation points $\{ \mathbf { y } _ { k } \} _ { k = 1 } ^ { p }$   
Ensure: Network outputs $\mathbf { G } ( \mathbf { u } ) \dot { ( } \mathbf { y ) } \in \mathbb { R } ^ { n \times p }$   
1: No need to repeat each input function $\mathbf { u } ^ { ( j ) }$ for all collocation points, simply   
$\mathbf { u } ^ { ( j ) }  \mathbf { U } \in \mathbb { R } ^ { \mathbf { \bar { \boldsymbol { n } } } \times m }$   
2: Tile collocation points for all input functions   
$\operatorname { j i l e } ( \mathbf { y } _ { k } , n ) \to \mathbf { Y } \overset { \cdot } { \in } \mathbb { R } ^ { n \times p \times d }$   
3: Compute Branch network output   
${ \cal B } ( { \bf U } )  {  } { \bf B } \in \mathbb { R } ^ { n \times s }$   
4: Compute Trunk network output   
$\mathcal { T } ( \mathbf { Y } )  \mathbf { T } \in \mathbb { R } ^ { n \times p \times s }$   
5: Compute final output via dot product over latent dimension   
$\begin{array} { r } { G _ { j k } \stackrel { \cdot } { = } \sum _ { \ell = 1 } ^ { s } B _ { j \ell } \dot { T _ { j k \ell } } } \end{array}$   
P6: Evaluate physics-informed loss   
$\mathcal { L } = \mathcal { L } _ { \mathrm { P D E } } + \mathcal { L } _ { \mathrm { B C } } + \mathcal { L } _ { \mathrm { I C } }$   
7: Update model parameters via gradient-based optimisation

<table><tr><td>Architecture</td><td>Branch input</td><td>Trunk input</td><td>Branch output</td><td>Trunk output</td><td>Combination method</td></tr><tr><td>Vanilla</td><td> $( n p , m )$ </td><td> $( n p , d )$ </td><td> $( n p , s )$ </td><td> $( n p , s )$ </td><td>Inner product</td></tr><tr><td>ATenH</td><td> $( n , m )$ </td><td> $( n , p , d )$ </td><td> $( n , s )$ </td><td> $( n , p , s )$ </td><td>Inner product (with broadcasting)</td></tr></table>

Table 2: A comparison of data processing choices in the two PI-DeepONet architectures herein presented. Here, m is the number of discretisation points for the input functions (the number of sensor points), d is the physical dimensionality of the collocation points, n is the number of input functions, p is the number of collocation points per function, and s is the length of the output layer of the (Branch and Trunk) subnetworks

## 2.4 Theoretical scaling laws for a full training step

We now analyse the computational cost of the diferent architectures over a training step, including automatic diferentiation for first-order physics derivatives and gradients with respect to model parameters. The cost of computing the loss at the boundaries, where no physics derivatives are calculated (we only use Dirichlet boundary conditions), is not included, since the number of boundary collocation points is generally much less than the number of interior points used for calculating the physics loss.

## 2.4.1 Cost model

Throughout this section, computational cost is measured in units proportional to the number of input-weight products evaluated; i.e. scalar multiplications of the form $w _ { i } x _ { i } ,$ where $w _ { i }$ and $x _ { i }$ are weights and inputs, respectively. This choice highlights the dominant scaling behaviour.

For a neuron in layer l with $r _ { l - 1 }$ inputs, the activation is given by:

$$
\sigma \left( \sum _ { i = 1 } ^ { r _ { l - 1 } } w _ { i } x _ { i } + \beta \right) ,\tag{6}
$$

where $\sigma ( )$ is the activation function and $\beta$ is the neuron’s bias. Assuming the computational cost is dominated by the weighted sum and treating the cost of the activation function as a constant, evaluating this neuron processes $r _ { l - 1 }$ weighted inputs and therefore has a cost proportional to $r _ { l - 1 }$ . Bias additions and application of activation functions are pointwise scalar operations, which we therefore neglect.

A fully connected layer with $r _ { l - 1 }$ inputs and $r _ { l }$ neurons therefore has a cost proportional to:

$$
r _ { l - 1 } r _ { l } .\tag{7}
$$

Forward pass cost For an L-layer MLP, the total cost of a forward pass is

$$
C _ { \mathrm { M L P } } \sim \sum _ { l = 1 } ^ { L } r _ { l - 1 } r _ { l } ,\tag{8}
$$

where $r _ { 0 }$ denotes the size of the $\mathrm { M L P ^ { \prime } s }$ input layer.

This cost model is equivalent, up to a constant factor, to the conventional floating-point operation (FLOP) count. Each weighted input contributes one multiplication and one addition, which is commonly implemented as a fused multiply-add (FMA) instruction.

Dot product The DeepONet output is formed by the dot product

$$
\sum _ { i = 1 } ^ { s } b _ { i } \tau _ { i } ,\tag{9}
$$

which processes s weighted inputs, where $b _ { i }$ and $\tau _ { i }$ are the Branch and Trunk network outputs, respectively. Thus, in accordance with this cost model, a single dot product has a cost proportional to s. A dot product is performed for every input functions and collocation point pairing, meaning the total cost is

$$
C _ { \mathrm { d o t } } \sim n p s .\tag{10}
$$

Backward pass cost Reverse-mode automatic diferentiation for a standard MLP involves two dominant operations per layer:

1. propagation of the error signal via matrix-vector multiplication, and

2. computation of weight gradients via matrix multiplication.

Both operations process $r _ { l - 1 } r _ { l }$ weighted products, and a summation over inputs per layer. As a result, both operations have the same asymptotic cost when measured in units proportional to the number of weighted interactions, and each are roughly equal to the total of input-weight products performed during the forward pass. Therefore, the cost of computing gradients with respect to model parameters is approximately

$$
C _ { \mathrm { b a c k w a r d } } \approx 2 C _ { \mathrm { f o r w a r d } } .\tag{11}
$$

This approximation neglects element-wise operations (Hadamard products and activation function derivatives), which contribute lower cost.

When automatic diferentiation is used to compute physics derivatives, the computational graph is extended. The same backward-pass reasoning applies, but not every layer necessarily performs both dominant operations. In particular, when diferentiating with respect to input or coordinate variables (rather than model parameters), the computation of weight gradients is not required. These distinctions are accounted for explicitly below.

## 2.4.2 Vanilla architecture

The Vanilla architecture uses Branch network inputs of shape [np, m] and Trunk network inputs of shape [np, d].

The cost of a forward pass through the Branch network is

$$
C _ { \mathrm { b r a n c h } } \sim n p \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } ,\tag{12}
$$

and for the Trunk network,

$$
C _ { \mathrm { t r u n k } } \sim n p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } ,\tag{13}
$$

where $r _ { l }$ and $e _ { l }$ denote the number of neurons in the lth layer of the Branch and Trunk networks, respectively, and $L _ { B }$ and $L _ { T }$ denote the corresponding number of layers. The Branch network has input dimension $r _ { 0 } = m$ and the Trunk network has input dimension $e _ { 0 } = d .$

The dot product contributes

$$
C _ { \mathrm { d o t } } \sim n p s .\tag{14}
$$

Physics derivatives First derivatives treat the Branch network outputs as constants. Backpropagation therefore traverses only the Trunk network and the dot product. The cost of computing physics derivatives is

$$
C _ { u _ { x } } \sim n p \sum _ { l = 1 } ^ { L _ { T } } ( e _ { l - 1 } e _ { l } ) + n p s .\tag{15}
$$

Loss gradients The loss depends on both u and $u _ { x }$ . Backpropagation therefore traverses the entire computational graph. The Trunk network appears twice (forward and derivative paths), while the Branch network appears once. Applying the $2 \times$ backward rule gives

$$
C _ { \mathrm { l o s s } } \sim 2 \left( 2 n p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + M n p \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } \right) + 2 n p s .\tag{16}
$$

where $M$ is the number of Branch networks, which is relevant for problems with multiple input functions [31].

Total cost Summing all contributions yields

$$
\begin{array} { l } { { \displaystyle C _ { \mathrm { V } } \sim M n p \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } + 2 n p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + 2 \times \left( 2 n p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + M n p \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } \right) + 4 n p s , } } \\ { { \displaystyle C _ { \mathrm { V } } \sim 3 M n p \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } + 6 n p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + 4 n p s . } } \end{array}\tag{17}
$$

Factoring out the number of collocation points $p ,$ and the number of input functions, $n ,$

$$
C _ { \mathrm { V } } \sim n p \Big ( 3 M \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } + 6 \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + 4 s \Big ) .\tag{18}
$$

Using the simplifications introduced in Section 2.5, namely $r _ { 0 } = m , e _ { 0 } = d , L _ { B } = L _ { T } = L$ , and $e _ { l \geq 1 } =$ $r _ { l \ge 1 } = e$ , Equation 18 becomes

$$
C _ { \mathrm { V } } \sim n p \left( ( 3 M m + 6 d ) e + ( L - 1 ) ( 3 M + 6 ) e ^ { 2 } + 4 s \right) .\tag{19}
$$

## 2.4.3 ATenH architecture

The ATenH architecture uses Branch network inputs of shape [n, m] and Trunk network inputs of shape $[ n , p , d ]$ , such that

$$
C _ { \mathrm { b r a n c h } } \sim n \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } ,\tag{20}
$$

$$
C _ { \mathrm { t r u n k } } \sim n p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } ,
$$

$$
C _ { \mathrm { d o t } } \sim n p s .\tag{21}
$$

(22)

Physics derivatives and loss backpropagation follow the same structure as the Vanilla architecture, yielding

$$
C _ { \mathrm { A T e n H } } \sim 3 M n \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } + 6 n p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + 4 n p s .\tag{23}
$$

Factoring out the number of collocation points $p ,$ and the number of input functions, $n ,$

$$
C _ { \mathrm { A T e n H } } \sim n \left[ 3 M \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } + 2 p \left( 3 \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + 2 s \right) \right] .\tag{24}
$$

Using the simplifications introduced in Section 2.5, namely $r _ { 0 } = m , e _ { 0 } = d , L _ { B } = L _ { T } = L$ , and $e _ { l \geq 1 } =$ $r _ { l \ge 1 } = e$ , Equation 24 becomes

$$
C _ { \mathrm { A T e n H } } \sim n \left[ 3 M m e + 3 M ( L - 1 ) e ^ { 2 } + 2 p \left( 3 d e + 3 ( L - 1 ) e ^ { 2 } + 2 s \right) \right] .\tag{25}
$$

## 2.4.4 Theoretical Speed-up

The previous derivations allow us to estimate a theoretical speed-up factor, $\phi \sim C _ { \mathrm { V } } / C _ { \mathrm { A T e n H } }$ , between the optimised PI-DeepONet architecture (ATenH) and the original architecture (Vanilla).

$$
\phi \sim \frac { 3 M p \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } + 6 p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + 4 p s } { 3 M \sum _ { l = 1 } ^ { L _ { B } } r _ { l - 1 } r _ { l } + 6 p \sum _ { l = 1 } ^ { L _ { T } } e _ { l - 1 } e _ { l } + 4 p s }\tag{26}
$$

With the simplifications from Section 2.5, Equation 26 becomes

$$
\phi \sim \frac { 3 M m p e + 6 p d e + 3 p ( M + 2 ) ( L - 1 ) e ^ { 2 } + 4 p s } { 3 M m e + 6 p d e + 3 ( M + 2 p ) ( L - 1 ) e ^ { 2 } + 4 p s }\tag{27}
$$

The derived speed-up factor is independent of $n ,$ and reaches a constant (with all other parameters fixed) for a suficiently large p.

## 2.5 Numerical tests

The numerical experiments in the following section compare the performance of the diferent PI-DeepONet architectures, focusing on training time and how this scales with key hyper parameters and spatial dimensionality. Since the two architectures share the same mathematical operations – the diference being exclusively in data handling – the optimal parameters should be common to both. Ensuring a fair and consistent hyperparameter selection is therefore essential, especially when assessing scaling with problem dimensionality.

Tests are based on the solution of the linear advection equation in one, two and three spatial dimensions, which provides progressively more stringent computational and memory demands, which is governed by the following PDE:

$$
\frac { \partial f } { \partial t } ( \mathbf { x } , t ) + \mathbf { c } \cdot \nabla f ( \mathbf { x } , t ) = 0 , \quad \forall ( \mathbf { x } , t ) \in [ 0 , 1 ] ^ { d - 1 } \times ( 0 , 1 ] ,\tag{28}
$$

where $f$ is the advected field, $\mathbf { \rho } ( \mathbf { x } , t ) $ denotes the spatial and temporal coordinates, and $\mathbf { c } = ( c _ { 1 } , \hdots , c _ { d - 1 } )$ is the constant advection velocity in each spatial dimension.

The problems are constrained by the following initial and boundary conditions:

$$
f ( \mathbf { x } , 0 ) = 0 ,\tag{29}
$$

$$
\left. f ( \mathbf { x } , t ) \right| _ { x _ { i } = 0 } = A _ { i } \exp \left( - { \frac { ( t - 0 . 5 ) ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } } } \right) \prod _ { j = 1 \atop j \neq i } ^ { d - 1 } \exp \left( - { \frac { ( x _ { j } - 0 . 5 ) ^ { 2 } } { 2 \sigma _ { j } ^ { 2 } } } \right) , \quad i = 1 , \ldots , d - 1 ,\tag{30}
$$

where $A _ { i }$ denotes the amplitude associated with the boundary $x _ { i } = 0 , \sigma _ { t }$ is the temporal standard deviation of the Gaussian profile, and $\sigma _ { j }$ is the standard deviation in the jth spatial dimension.

The exact solutions can be expressed in a unified form using the method of characteristics. Define the backtracking times, $s _ { i }$

$$
{ \mathcal { S } } _ { i } = t - { \frac { x _ { i } } { c _ { i } } } , \quad i = 1 , \ldots , d - 1 ,\tag{31}
$$

which correspond to the time at which the characteristic passing through $( { \bf x } , t )$ intersects the boundary $x _ { i } = 0$ . The solution is then given by

$$
f ( \mathbf { x } , t ) = \left. \begin{array} { l l } { 0 , } & { \mathrm { i f } \operatorname* { m a x } _ { i } S _ { i } < 0 , } \\ { f _ { i ^ { * } } ( \mathbf { x } , t ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{32}
$$

where

$$
i ^ { * } = \arg \operatorname* { m a x } _ { 1 \leq i \leq d - 1 } S _ { i } ,\tag{33}
$$

and the contribution from each boundary $x _ { i } = 0$ is defined as

$$
f _ { i } ( { \bf x } , t ) = A _ { i } \exp \left( - \frac { ( S _ { i } - 0 . 5 ) ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } } \right) \prod _ { j = 1 } ^ { d - 1 } \exp \left( - \frac { \left( x _ { j } - 0 . 5 - c _ { j } \frac { x _ { i } } { c _ { i } } \right) ^ { 2 } } { 2 \sigma _ { j } ^ { 2 } } \right) , \quad i = 1 , \dots , d - 1 .\tag{34}
$$

The solution $f _ { i ^ { * } }$ is obtained by selecting the boundary contribution corresponding to the characteristic with the largest backtracking time.

We fix $c _ { i } = 1$ for all $i , \sigma _ { t } = \sigma _ { j } = 0 . 0 7$ , and allow $A _ { i }$ to vary independently, producing a range of boundary conditions the PI-DeepONet must generalise to. Specifically, $A _ { i }$ is randomly sampled from the uniform distribution [0.1, 1] yielding n distinct boundary condition realisations which are fed to the Branch network during training. Although the model is therefore restricted to Gaussian boundary conditions and cannot extrapolate to qualitatively diferent boundary-condition families, this is appropriate for the purpose of evaluating computational performance.

The number of training epochs for each problem was determined via preliminary tests which included validation-loss monitoring for the 1D and 2D advection problems: the validation loss was computed at every epoch (without contributing to weight updates) and compared with the training loss to detect overfitting and identify the onset of loss plateauing, indicating diminishing returns from further training. The choice of epochs was based on observing no overfitting, and achieving a low loss, as seen in Fig 2. The same number of epochs was used for the Vanilla and ATenH architectures to allow for a fair comparison. 50,000 epochs, 100,000 epochs, and 200,000 epochs were selected for 1D, 2D, and 3D linear advection problems, respectively.

For all numerical experiments, collocation points are sampled independently from a uniform distribution over the computational domain. The number of collocation points was chosen so that the average spacing between randomly sampled points remained constant across dimensions, according to $\begin{array} { r } { p = \left( \frac { \ell } { \Delta x } \right) ^ { d } } \end{array}$ , where $\ell = 1$ denotes the characteristic length of the domain in each dimension, and $\Delta x$   is the target average spacing between points. Accordingly, we set 10 collocation points per spatiotemporal dimension, since increasing this furthe (say, by a factor of 2) produced comparable accuracy while causing memory issues, thus preventing the Vanilla architecture from successfully completing beyond the 1D advection case. Failure to complete the relatively simple task of solving the 2D linear advection problem discretized with 20 points along each dimension sheds light onto the need for improved faster methods. In practice, separate collocation points must be sampled for evaluating the loss from the boundary conditions, the initial condition and physics loss. We denote their amounts as $p _ { b _ { x } }$ for $i = 1 , . . . , d - 1 , p _ { 0 }$ and $p _ { I }$ , respectively. This dimensional scaling yielded higher predictive accuracy in 2D and 3D than alternative sampling strategies tested.

![](images/39f9d89849ad9b1d5611c67024ced0150c963b46d9c7b172bfd7b9b02dd7bda2.jpg)

![](images/e7e97fd3b40cf8f376cb7a908470570703d166f1849aaf10c3430fb4baf3fbac.jpg)  
Figure 2: Training and validation loss profiles as a function of training epoch for 1D (left) and 2D (right) linear advection, trained using the ATenH PI-DeepONet. The training and validation loss curves remain closely aligned throughout training indicating no overfitting. The rate of decrease in the loss begins to plateau after approximately 50,000 epochs for 1D linear advection and 100,000 epochs for 2D linear advection

We use $n = 5 0 0$ training input functions and $n _ { \mathrm { t e s t } } = 5 0$ testing input functions, consistent with prior PI-DeepONet studies (e.g. 26, 32, 25), which is suficient to ensure generalisation. More complex problems may require larger datasets; for instance, [11] employ $\sim 1 0 ^ { 4 }$ input functions. Each input function was discretised using $m = 5 0 ^ { ( d - 1 ) }$ sensor points per (spatiotemporal) dimension. Although we consider Gaussian inputs with varying amplitudes, this resolution reflects more general settings where inputs may contain finer-scale variations while keeping training costs manageable.

In the 2D and 3D linear advection problems there are multiple input functions, corresponding to the inflow boundaries $x _ { i } ~ = ~ 0$ for $i = 1 , \ldots , d - 1$ Each boundary condition is parametrised by an independent amplitude $A _ { i } ,$ and therefore defines a distinct input function. To handle the multiple independent boundary condition inputs, we employ the Multiple-Input DeepONet (MIONet) structure [31], which uses separate Branch networks to process each input function. In all our experiments, both Branch networks adopt either the Vanilla or the ATenH architecture. The outputs from the separate Branch networks and the Trunk network are then combined via an inner product across all Branch networks and the Trunk network. As an illustrative example, in the 2D linear advection problem with spatial coordinates $( x _ { 1 } , x _ { 2 } )$ , there are two boundary inputs corresponding to $x _ { 1 } = 0$ and $x _ { 2 } = 0$ . For $n = 5 0 0$ samples, each boundary condition is discretised using 50 points per spatial and temporal dimension, yielding $\dot { m } = 5 0 ^ { 2 } = 2 5 0 0$ input degrees of freedom per boundary. The boundary and initial condition losses are evaluated at $1 0 ^ { 2 }$ collocation points each, while the interior physics loss is evaluated at $1 0 ^ { 3 }$ collocation points.

Subnetwork depths and widths were selected via grid searches on the 1D and 2D advection problems, which can be seen in Fig 3. To avoid overcomplicating the hyperparameter space, both subnetworks used the same width and depth. In both cases, four hidden layers provided the best compromise between computational cost and accuracy. Similarly, we select widths of 40 neurons per hidden layer for 1D and 60 for 2D. A ded icated grid search for 3D advection was not performed due to the prohibitive computational cost; instead, we heuristically inferred subnetwork widths from the lower-dimensional results, setting 80 neurons for the 3D problem. All hidden and output layers were kept at the same width to avoid unnecessarily complication of the hyperparameter space. A large proportion of computational resources was not devoted to extensive hyperparameter optimisation, as the primary objective of this study is to assess the computational performance of diferent architectures rather than to maximise predictive accuracy. The configurations used were suficient to reliably learn the solutions of all test problems, as demonstrated in the following section.

All models were trained in PyTorch using the Adam optimiser [33], with a learning rate of $1 0 ^ { - 4 }$ and tanh activation functions, on an NVIDIA A100 GPU.

To quantify the predictive accuracy of the architectures, we use the mean relative root mean squared error (MRRMSE), defined as

$$
\mathrm { M R R M S E } = \frac { 1 } { n _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t e s t } } } \frac { \| \hat { u } _ { i } - u _ { i } \| _ { 2 } } { \| u _ { i } \| _ { 2 } } \times 1 0 0 \% ,\tag{35}
$$

![](images/30c19163ce2dad14483beb4901a182a9d96e1fb8ababce5e99d38f8516fb2311.jpg)

![](images/f503ccd63c6f1b57711bfde5d9bc2f062b3e920a0f60f146852e795fb65fdd8a.jpg)  
Figure 3: Grid searches exploring the depth and width of the subnetworks for (left) 1D linear advection and (right) 2D linear advection. The same width and depth was used across Branch and Trunk networks to reduce the complexity of the hyperparameter space, alongside keeping the layer widths constant across hidden layers

where $u _ { i }$ denotes the exact solution associated with the ith input function, $\hat { u } _ { i }$ is the corresponding model prediction, and $\lVert \cdot \rVert _ { 2 }$ denotes the discrete $L ^ { 2 }$ norm over the spatiotemporal evaluation grid. MRRMSE is only computed in regions of the domain where $u _ { i } \neq 0$

## 3 Results

## 3.1 1D Linear Advection

We first compare the PI-DeepONet architectures on the 1D linear advection problem. Ten separate Vanilla and ATenH PI-DeepONets are trained over 50,000 epochs to solve this problem in order to report means and assess sensitivity to random parameter initialisation. The solutions produced by a Vanilla and an ATenH PI-DeepONet for a representative test input function are shown in Figure 4.

Table 3 reports the mean training time and MRRMSE for the 1D linear advection problem. The two architectures exhibit comparable predictive accuracy, with closely matching MRRMSE values and similar standard deviations. The ATenH architecture reduces the mean training time by 12.1% relative to the Vanilla architecture, corresponding to a 1.14× speed-up. For this case, the theoretical cost derived earlier (Eq. 27) predicts a speed-up scaling of 1.64×.

## 3.2 2D Linear Advection

In this example, the training of Vanilla and ATenH PI-DeepONets are compared for the 2D linear advection problem. The PI-DeepONets are trained for 100,000 full-batch epochs and the average of ten diferent realisations is reported. An example solution produced by the ATenH implementation of a PI-DeepONet is shown in Fig. 5, demonstrating good agreement with the ground truth. The prediction of the Vanilla architecture is omitted due to its close visual similarity to the prediction of the ATenH architecture.

Table 3 also reports the mean training time and MRRMSE for linear 2D advection. The two architectures exhibit comparable predictive accuracy, with closely matching MRRMSE values and overlapping standard deviations.

## 3.2.1 Computational Performance and Scaling for 2D Linear Advection

Table 3 reports the mean training time for linear 2D advection. The ATenH architecture reduces the mean training time by 71.4%, corresponding to a 3.49× speed-up. For this case, applying the theoretical cost derived earlier (Eq. 27) with the specified hyperparameters in Section 2.5 predicts a speed-up scaling of 7.46×.

![](images/3bdd8d1f7070f8885ca544a9b06568eb2986e13262a2114a7b02f2ce2f108f2d.jpg)

![](images/d69138aa7a8239fdafa0670edf5199286bf4fb72d8bdf7c46de9373047fa77ae.jpg)

![](images/b9186d227e696e6ce72d71dcaaa7499429f44066b0f4ec4433f8899bf7216f87.jpg)

![](images/77d1c900e769a878b5cb2268b2c563a5378270af0b54acee2773a839e856ab39.jpg)

![](images/ef6dad72f0adb71dc741891d9fefaa9d814bb7eba42f8ec597a6d4072374a5ca.jpg)  
Figure 4: Representative comparison of the exact solution and the predictions of the Vanilla and ATenH PI-DeepONet architectures for a test input function in 1D linear advection. The models are trained with 10 collocation points per dimension and 500 training input functions. (a) The Vanilla PI-DeepONet’s prediction. (b) The ATenH PI-DeepONet’s prediction. (c) The exact solution. (d) The absolute error for the Vanilla architecture. (e) The absolute error for the ATenH architecture

The diference in training computational time between both architectures is more pronounced in 2D relative to 1D. Therefore, we used the 2D problem to analyse the training time scaling for key hyperparameters. Fig. 6 shows the mean training time for both architectures for varying numbers of internal collocation points and input functions, respectively, along with the relative speed-up of the ATenH architecture compared to the Vanilla architecture. We use the internal, rather than total, collocation points due to the former being much greater in number than initial/boundary points in practice (specifically, $p _ { I } ^ { 2 / 3 } = p _ { b _ { x _ { 1 } } } = p _ { b _ { x _ { 2 } } } = p _ { 0 } )$ .

For both PI-DeepONet architectures, the training time increases linearly with the number of internal col location points or input functions. These linear scalings are consistent with the theoretical cost analysis presented in Section 2.4. In particular, Equations 19 and 25 show that the computational cost of a training step is proportional to the number of collocation points and the number of input functions, leading to an overall $\bar { \mathcal { O } } ( \bar { p } )$ and ${ \mathcal { O } } ( n )$ dependence. Moreover, the prefactors of these terms are smaller for the ATenH architecture compared to the Vanilla architecture, which explains the consistently lower training times observed for ATenH across all tested collocation point and input function counts.

Using the derived cost expressions in Equation 27 with the chosen hyperparameters yields the asymptotic theoretical speed-up factor $\phi  7 . 5$ as $p  \infty$ , while remaining independent of n. The value of 7.5 is specific to the chosen values of m, M, e, L, d, and s.

## 3.3 3D Linear Advection

In this example, the training of both PI-DeepONet architectures is compared for the 3D linear advection problem. The Vanilla and ATenH PI-DeepONets are trained for 300,000 full-batch epochs and the average of five realisations is reported. Representative exact and PI-DeepONet predicted solutions are shown in Figure 7.

Table 3 shows that only the ATenH architecture successfully handled this problem, with an average training time of $1 . 4 0 \times 1 0 ^ { 5 } \mathrm { s }$ (approximately 39 hours) and a mean predictive error of 5.438%. The Vanilla architecture failed during training due to insuficient memory, highlighting the need to review the implementation of algorithms such as DeepONet, particularly when using widely popular libraries like PyTorch. ATenh succeeds in training here, when the original implementation (Vanilla) fails, thanks to a less memory-intensive input tensor representation. Although an empirical comparison is not possible, Equation 27 predicts a theoretical speed-up factor of approximately $\phi \sim 5 8 0$ for the ATenH architecture relative to the Vanilla architecture.

![](images/5e19d411c871fb262560c42c93f0f811f7163ebd5f3ca9a813e1d49f147b528e.jpg)

![](images/3aff1937673a4884bc2e16b9c35a801d182e7ba00e414cba00430bcfc7a6946e.jpg)

![](images/3f18a80feb4d43063ecf788c8cdaefc928af64ad4ca75dd9119fa77dd9a97b65.jpg)

![](images/6a9cf0211e0a2bd69da738aba47d9cc0ece669a8080f1cda1ea8c9f186f82429.jpg)

![](images/c23d134da538383c7d5a36b714a9823f7671f129559ab194c632d3700819c4d7.jpg)

![](images/df9267422bdea70b059bd35c0c3e1660901d77ba136cabfbc8996bb79c203ab7.jpg)  
Figure 5: Comparison of the exact solution and the predictions of an ATenH PI-DeepONet for a representative combination of boundary inputs in 2D linear advection at two time instances. The Vanilla architecture yields qualitatively similar results and is therefore omitted for brevity. (a)–(c) Exact solution, prediction, and absolute error at t = 0.48. (d)–(f) Exact solution, prediction, and absolute error at t = 1.00

![](images/6be6a0b17f588f54d537ed8beb3b1ff857a84abd04edf7cb3a68a085ebf77bd4.jpg)

![](images/7d1870a887e406f0e7c124994f9c6fa5bac38933429f10330e95eb30d8d03c62.jpg)  
Figure 6: Mean training time of the Vanilla and ATenH architectures for 2D linear advection with (left) varying numbers of collocation points with a fixed number of input functions, and (right) varying numbers of input functions with a fixed number of collocation points. Solid lines indicate linear regression fits to the training times. Right y-axis shows the computed ratio of training times for both architectures (i.e. ATenH’s speed-up factor). The two largest collocation-point data points are excluded for the Vanilla architecture due to insuficient memory (left)

![](images/7d372a45c0c3a164daddaa2d19e98b25ac3338ffdb9749b43e367a54ff1cf78e.jpg)

![](images/35f4209e4fdd19e3c7ec5966e0c87934741ef62a673e66c87fbc461e08e5383e.jpg)

![](images/cfeb2a7e7a2ee1abeda2471bb6ee0689dc65b23badf81946510a64cd5b15337e.jpg)

![](images/99db9b1e73050aa1e031841d02629d9f84d4673040f27d4ef4b41dafa2c5cce3.jpg)

![](images/447ca21041d3c393bbb00b671505ca3045a0473c2c6126f0c600936a73944678.jpg)

![](images/8f2892dc5f0225b61df485dfd1a6e3159a1b9b3219a5cab04792a4b348ab7bc8.jpg)  
Figure 7: Comparison of the exact solution and the predictions of an ATenH PI-DeepONet for a representative combination of boundary inputs in 3D linear advection at $x _ { 3 } = 0 . 4 4$ and two time instances. The Vanilla architecture fails to train due to a memory error and is therefore omitted. (a)–(c) Exact solution, prediction, and absolute error at $t = 0 . 4 4 .$ . (d)–(f) Exact solution, prediction, and absolute error at $t = 1 . 0 0$

Table 3: Mean training time and MRRMSE scaling with spatial dimensionality in linear advection problems for the Vanilla and ATenH architectures. Standard deviations are not shown for the training times since they are negligible. There is no data for the Vanilla architecture in the 3D linear advection problem due to memory limitations during training. Also shown is the theoretical speed-up ϕ for ATenH when compared to the Vanilla PI-DeepONet architecture, for each problem with their corresponding hyperparameters, as specified in Section 2.5
<table><tr><td></td><td colspan="2">Vanilla</td><td colspan="2">ATenH</td><td>Theoretical speed-up</td></tr><tr><td>Spatial dimension- ality</td><td>Time (s)</td><td>MRRMSE (%)</td><td>Time (s)</td><td>MRRMSE (%)</td><td>φ</td></tr><tr><td>1D</td><td> $3 . 1 4 \times 1 0 ^ { 2 }$ </td><td> $0 . 4 7 1 \pm 0 . 2 9 3$ </td><td> $2 . 7 6 \times 1 0 ^ { 2 }$ </td><td> $0 . 3 0 8 \pm 0 . 0 7 8$ </td><td>1.64</td></tr><tr><td>2D</td><td> $1 . 1 8 \times 1 0 ^ { 4 }$ </td><td> $2 . 0 9 4 \pm 0 . 8 4 6$ </td><td> $3 . 3 6 \times 1 0 ^ { 3 }$ </td><td> $1 . 5 2 4 \pm 0 . 5 1 3$ </td><td>7.46</td></tr><tr><td>3D</td><td></td><td></td><td> $1 . 4 0 \times 1 0 ^ { 5 }$ </td><td> $5 . 4 3 8 \pm 1 . 1 0 8$ </td><td>580</td></tr></table>

## 4 Discussion

Since the introduction of PI-DeepONets, research has largely focused on improving predictive accuracy and extending the range of problems to which they can be applied. Comparatively less attention has been paid to reducing the computational cost of training especially when scaling PI-DeepONets to computationally demanding, higher-dimensional problems. This work addresses this issue by introducing an architectural modification that preserves the defining advantages of PI-DeepONets – training without labelled solution data, generalisation across varying initial and boundary conditions, and flexibility across PDE classes – while substantially reducing computational cost.

In this work, we propose a modified PI-DeepONet architecture that changes how input functions are processed within the Branch network. The key observation is that, in the Vanilla implementation of PI-DeepONets, the Branch network is evaluated repeatedly at every collocation point, even though the input function itself does not change. Addressing this redundancy yields our proposed alternative (ATenH), in which the Branch network is evaluated once per input function and its encoding is reused across all collocation points.

This modification comes with the constraint that all output functions must be evaluated at the same number of collocation points. In a data-driven setting this may be restrictive because the amount of labelled training data that is available will likely vary between samples. However, DeepONets are increasingly being adopted within physics-informed frameworks (PI-DeepONets), where this limitation does not apply – collocation points can be freely generated and chosen.

By reducing the dimensionality of the Branch network input tensors, the number of operations required by this subnetwork is substantially decreased. As the Branch network encodes the input functions, which may have a high number of degrees of freedom to adequately represent boundary or initial conditions, this reduction directly lowers the dominant computational cost. Importantly, the modified architecture performs the same underlying mathematical operations as the original one and, as a result, achieves comparable predictive accuracy while significantly reducing training time. For 1D linear advection, the average training time is 314s and 276s for the Vanilla and ATenH architectures, respectively, while for 2D linear advection the corresponding times are 11800s and 3360s. Most notably, the 3D linear advection problem exposed a fundamental practical limitation of the Vanilla PI-DeepONet implementation. It exceeded the available GPU memory and could not be trained on an NVIDIA A100 GPU, whereas the ATenH architecture completed training in approximately 39 hours and achieved an average relative error of 5.44%. Equation 27 also predicts a theoretical speed-up factor of 580× for this problem. The experimental results demonstrate that the proposed modification does more than reduce training time: by lowering memory requirements, it enables classes of higher-dimensional PI-DeepONet problems that are otherwise computationally infeasible on the same hardware. Consequently, extending PI-DeepONets to higher-dimensional problems is not simply a matter of applying the existing formulation to larger problems; careful architectural design is essential for maintaining computational tractability.

A theoretical cost model was developed to quantify the training-time scaling of both architectures with respect to the number of collocation points and the number of input functions. This model predicts linear scaling with both number of collocation points and input functions for both architectures, but with larger prefactors for the Vanilla architecture, implying steeper scaling. The empirical results for the 1D and 2D linear advection problems confirm this behaviour. The agreement between the theoretical scaling laws and the empirical measurements suggests that the observed performance gains arise primarily from diferences in the computational structure of the two architectures rather than implementation-specific efects. The model also predicts an asymptotic constant speed-up factor, which is reinforced by the experimental profiles seen in Figure 6; however, the observed speed-ups are smaller than the theoretical values (1.14× vs. 1.64× in 1D, and 3.49× vs. 7.46× in 2D). Despite the quantitative discrepancies, the consistency in qualitative scaling behaviour reinforces the observation that the theoretical model captures the dominant computational trends governing both architectures. Some discrepancy is expected, as the cost model accounts only for the number of summed products in forward and backward passes and therefore neglects implementation dependent overheads such as memory allocation, activation function evaluation, data movement, automatic diferentiation graph construction, and optimiser updates.

## 5 Conclusion

This work demonstrates that modifying the Branch network input handling in PI-DeepONets can substantially reduce computational cost without sacrificing predictive accuracy. While the theoretical analysis captures the qualitative scaling behaviour, the empirical results quantify substantial practical gains, particularly in higher-dimensional settings.

A key outcome is that the ATenH architecture enables problems that are infeasible under the Vanilla for mulation. In particular, while the Vanilla architecture failed on the 3D linear advection problem due to memory limitations, the ATenH architecture successfully completed training with a mean predictive error of 5.44%. This highlights the fact that implementation-level considerations can determine whether a problem is tractable in practice.

Overall, this work demonstrates that implementation choices can play a critical role in the practical performance of PI-DeepONets. ATenH provides significant improvements over the Vanilla PI-DeepONet formulation; however, it should not be viewed as a final design, but rather as evidence that careful consideration of how these models are implemented can yield substantial gains. This suggests that further improvements may be achievable through continued optimisation of existing architectures, particularly within widely used frameworks such as PyTorch. Furthermore, the agreement between theoretical and empirical scaling indi cates that these findings are not tied to a specific hardware or software configuration, but are expected to generalise to other implementations. Future work should explore the application of similar architectural improvements to tackle problems that have so far proved intractable for PI-DeepONets.

## Declarations

• Acknowledgements: The authors acknowledge the use of the IRIDIS High Performance Computing Facility, and associated support services at the University of Southampton, in the completion of this work.

• Funding: The first author was supported by UK Research and Innovation (UKRI) through a PhD studentship.

• Author contributions: NB: Conceptualisation, Methodology, Software, Visualisation, Analysis, Writing - Original Draft. GA and SM: Conceptualisation, Methodology, Supervision, Writing – Review & Editing. All authors discussed the results, contributed to the interpretation of the findings, and approved the final manuscript.

• Competing interests: The authors declare that they have no conflict of interest.

## References

[1] Nathan Baker, Frank Alexander, Timo Bremer, Aric Hagberg, Yannis Kevrekidis, Habib Najm, Manish Parashar, Abani Patra, James Sethian, Stefan Wild, Karen Willcox, and Steven Lee. Workshop Report on Basic Research Needs for Scientific Machine Learning: Core Technologies for Artificial Intelligence. Technical report, USDOE Ofice of Science (SC), Washington, D.C. (United States), February 2019.

[2] Christopher Rackauckas, Yingbo Ma, Julius Martensen, Collin Warner, Kirill Zubov, Rohit Supekar, Dominic Skinner, Ali Ramadhan, and Alan Edelman. Universal Diferential Equations for Scientific Machine Learning, November 2021.

[3] M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, February 2019.

[4] Klapa Antonion, Xiao Wang, Maziar Raissi, and Laurn Joshie. Machine Learning Through Physics– Informed Neural Networks: Progress and Challenges. Academic Journal of Science and Technology, 9(1):46–49, January 2024.

[5] Zhiping Mao, Ameya D. Jagtap, and George Em Karniadakis. Physics-informed neural networks for high-speed flows. Computer Methods in Applied Mechanics and Engineering, 360:112789, March 2020.

[6] Shengze Cai, Zhiping Mao, Zhicheng Wang, Minglang Yin, and George Em Karniadakis. Physicsinformed neural networks (PINNs) for fluid mechanics: a review. Acta Mechanica Sinica, 37(12):1727– 1738, December 2021.

[7] Taco de Wolf, Hugo Carrillo, Luis Martı, and Nayat Sanchez-Pi. Assessing Physics Informed Neural Networks In Ocean Modelling and Climate Change Applications. AI: Modeling Oceans and Climate Change Workshop at ICLR 2021, 2021.

[8] Qin Chen, Nan Wang, and Zhao Chen. Simultaneous mapping of nearshore bathymetry and waves based on physics-informed deep learning. Coastal Engineering, 183:104337, August 2023.

[9] Xin Qi, Gustavo A.M. De Almeida, and Sergio Maldonado. Physics-informed neural networks for solving flow problems modeled by the 2D Shallow Water Equations without labeled data. Journal of Hydrology, 636:131263, June 2024.

[10] James Donnelly, Alireza Daneshkhah, and Soroush Abolfathi. Physics-informed neural networks as surrogate models of hydrodynamic simulators. Science of The Total Environment, 912:168814, February 2024.

[11] Rüdiger Brecht, Elsa Cardoso-Bihlo, and Alex Bihlo. Physics-informed neural networks for tsunami inundation modeling. Journal of Computational Physics, 536, 2025.

[12] Bamidele O. Awojoyogbe and Michael O. Dada. Digital Molecular Magnetic Resonance Imaging. Series in BioEngineering. Springer Nature Singapore, Singapore, 2024.

[13] Somdatta Goswami, Cosmin Anitescu, Souvik Chakraborty, and Timon Rabczuk. Transfer learning enhanced physics informed neural network for phase-field modeling of fracture. Theoretical and Applied Fracture Mechanics, 106:102447, April 2020.

[14] Ehsan Haghighat, Danial Amini, and Ruben Juanes. Physics-informed neural network simulation of multiphase poroelasticity using stress-split sequential training. Computer Methods in Applied Mechanics and Engineering, 397:115141, July 2022.

[15] Chen Xu, Ba Trung Cao, Yong Yuan, and Günther Meschke. Transfer learning based physics-informed neural networks for solving inverse problems in engineering structures under diferent loading scenarios. Computer Methods in Applied Mechanics and Engineering, 405:115852, February 2023.

[16] Zhao Chen, Yang Liu, and Hao Sun. Physics-informed learning of governing equations from scarce data. Nature Communications, 12(1):6136, October 2021.

[17] Liu Yang, Xuhui Meng, and George Em Karniadakis. B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data. Journal of Computational Physics, 425:109913, January 2021.

[18] Daniel R. Gurevich, Matthew R. Golden, Patrick A.K. Reinbold, and Roman O. Grigoriev. Learning fluid physics from highly turbulent data using sparse physics-informed discovery of empirical relations (SPIDER). Journal of Fluid Mechanics, 996:A25, October 2024.

[19] Lu Lu, Xuhui Meng, Zhiping Mao, and George Em Karniadakis. DeepXDE: A Deep Learning Library for Solving Diferential Equations. SIAM Review, 63(1):208–228, January 2021.

[20] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural Operator: Learning Maps between Function Spaces with Applications to PDEs. Journal of Machine Learning Research, 24(89):1–97, 2023.

[21] Lu Lu, Pengzhan Jin, and George Em Karniadakis. DeepONet: Learning nonlinear operators for identifying diferential equations based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, March 2021.

[22] Sifan Wang, Hanwen Wang, and Paris Perdikaris. Learning the solution operator of parametric partial diferential equations with physics-informed DeepOnets. Science Advances, 7(40):eabi8605, 2021.

[23] Tianping Chen and Hong Chen. Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. IEEE Transactions on Neural Networks, 6(4):911–917, July 1995.

[24] Kamyar Azizzadenesheli, Nikola Kovachki, Zongyi Li, Miguel Liu-Schiafini, Jean Kossaifi, and Anima Anandkumar. Neural operators for accelerating scientific simulations and design. Nature Reviews Physics, 6(5):320–328, April 2024.

[25] Luis Mandl, Somdatta Goswami, Lena Lambers, and Tim Ricken. Separable physics-informed Deep-ONet: Breaking the curse of dimensionality in physics-informed machine learning. Computer Methods in Applied Mechanics and Engineering, 434:117586, February 2025.

[26] Somdatta Goswami, Aniruddha Bora, Yue Yu, and George Em Karniadakis. Physics-Informed Deep Neural Operator Networks, July 2022.

[27] Somdatta Goswami, Minglang Yin, Yue Yu, and George Em Karniadakis. A physics-informed variational DeepONet for predicting crack path in quasi-brittle materials. Computer Methods in Applied Mechanics and Engineering, 391:114587, March 2022.

[28] Mohammad Sadegh Eshaghi, Cosmin Anitescu, Manish Thombre, Yizheng Wang, Xiaoying Zhuang, and Timon Rabczuk. Variational Physics-informed Neural Operator (VINO) for solving partial diferential equations. Computer Methods in Applied Mechanics and Engineering, 437:117785, March 2025.

[29] Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, and Eunbyung Park. Separable Physics-Informed Neural Networks. In Advances in Neural Information Processing Systems, volume 36, pages 23761–23788, 2023.

[30] Lu Lu, Xuhui Meng, Shengze Cai, Zhiping Mao, Somdatta Goswami, Zhongqiang Zhang, and George Em Karniadakis. A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data. Computer Methods in Applied Mechanics and Engineering, 393:114778, April 2022.

[31] Pengzhan Jin, Shuai Meng, and Lu Lu. MIONet: Learning multiple-input operators via tensor product, February 2022.

[32] Robert Keppler, Rasmus Fensholt, and Julian Koch. Learning 2D Shallow Water Equations with Physics-Informed Neural Operator Networks, 2025.

[33] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2014.