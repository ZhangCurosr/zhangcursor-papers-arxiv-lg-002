# Revisiting Error Feedback for Model Quantization

Wai-Yiu Keung<sup>†∗</sup>, Chi Cui<sup>§∗</sup>, Wing-Kin Ma<sup>†</sup>, Jes´us Cid-Sueiro<sup>§</sup> and Hei Victor Cheng<sup>‡⋆</sup>

<sup>†</sup>The Chinese University of Hong Kong, Hong Kong SAR of China <sup>§</sup>Universidad Carlos III de Madrid, Spain <sup>‡</sup>Aarhus University, Denmark

## Abstract

We study causal error-feedback coeficients as design variables for model quantization. Modeling instantaneous quantization errors as independent, zero-mean variables with a common fixed variance, we formulated a metric-dependent trace problem for designing feedback coeficients. The formulation resembles fixed-order GPTQ/LDLQ feedback and supports Jacobian-aware designs that account for downstream error propagation. For a positivedefinite metric H, we prove that the joint rotation-feedback trace minimum is n det(H)<sup>1/n</sup> and present an attaining pair using the geometric mean decomposition. Experiments on synthetic networks, trained MLPs, and language models show benefits from downstreamsensitive feedback, but also reveal that trace optimality need not improve finite-level performance. Small QAT experiments further illustrate direct loss-driven feedback tuning with fixed or trainable network weights.

Keywords— Model quantization; post-training quantization; error-feedback quantization; Jacobianaware quantization.

## 1 Introduction

Post-training quantization (PTQ) reduces the precision of a trained network using limited calibration data, without end-to-end retraining [9, 10]. Earlier work on model compression combined pruning, quantization, and coding, while PTQ methods developed techniques that do not require full retraining [11, 4, 25]. AdaRound and BRECQ further established data-dependent rounding and block reconstruction for PTQ [24, 20]. PTQ performance depends not only on the quantization resolution, but also on how the errors introduced at individual weight coordinates afect subsequent quantization decisions and the network output. In a sequential quantizer, an error made at one coordinate can be fed into the inputs presented to the quantizer at later coordinates. We study the coeficients governing this causal error feedback as design variables of the quantizer.

Feedback and adaptive rounding already play an important role in PTQ. GPTQ and LDLQ use curvature-informed compensation, while QuIP combines adaptive rounding with orthogonal preprocessing [10, 6]. SmoothQuant and AWQ use activation information and rescaling in quantizer design [32, 21]. QuIP#, QuaRot, and SpinQuant further illustrate the use of structured, random, or learned transformations [30, 2, 22]. Other methods change the criterion used to assess quantization error: QEP accounts for errors propagated from earlier layers, GuidedQuant incorporates end-loss gradients, and YAQA uses an approximation to end-to-end curvature [1, 16, 31]. Qronos develops correction and difusion of quantization errors, while GPTQ-2D studies adaptive rounding under a two-sided quadratic metric [34, 7]. QuIP already formulates adaptive rounding with linear feedback and characterizes its choice under a layer-wise quadratic criterion. YAQA, in turn, develops adaptive rounding using an approximation to end-to-end curvature. Our question is diferent: within a fixed causal quantizer, how can its feedback coeficients be designed under diferent input-coordinate criteria, jointly with a change of representation, or directly from the quantized network’s loss?

Our starting point is a causal quantizer whose feedback coeficients form a unit lower-triangular matrix G. Under a fixed isotropic second-moment model for those errors, a specified input-coordinate metric yields a tractable trace design problem. The input covariance lead to the feedback structure associated with fixed-order $\mathrm { G P T Q / L D L Q }$ . Replacing that metric with a Jacobian-derived input-coordinate surrogate gives a downstream-sensitive feedback design. This surrogate provides a tractable way to incorporate downstream sensitivity into feedback design without directly optimizing network-output error.

We then allow G to be designed jointly with an orthogonal rotation R. For a specified positive-definite metric, we characterize the global optimum of the resulting fixed-statistics trace problem and construct an attaining rotation-feedback pair using the QRS/geometric mean decomposition [33, 13]. Unlike using a rotation solely as preprocessing, this result identifies the best rotation and causal feedback together for the trace objective. It provides a benchmark for asking whether a trace-optimal design also performs well after finite-level quantization.

Our main contributions are as follows:

• We formulate causal error feedback as an explicit quantizer-design variable under a general inputcoordinate metric. The resulting trace framework resembles covariance-based GPTQ/LDLQ feedback as a special case and enables downstream-sensitive designs through Jacobian-derived metrics.

• We characterize the global optimum of the joint rotation-feedback problem for any positive-definite metric, and construct a pair using the geometric mean decomposition attaining the lower bound. This result provides a fixed-statistics benchmark for how far rotation and causal feedback can reduce the trace surrogate.

• Our experiments expose the gap between optimizing error propagation and controlling the errors produced by a finite-level quantizer. Experiments show that objective-aware feedback can improve downstream fidelity, while the joint trace optimum need not yield better realized performance. Small QAT experiments further show that feedback coeficients can be optimized directly against the network loss.

## 2 Preliminaries

Consider an L-layer neural network parameterized by weights $\{ W _ { \ell } \} _ { \ell = 1 } ^ { L }$ . For an input data set $\boldsymbol { X }$ , the network output is

$$
\mathcal { F } _ { \{ W _ { \ell } \} } ( \boldsymbol { X } ) = \sigma _ { L } \big ( W _ { L } \sigma _ { L - 1 } ( W _ { L - 1 } \cdot \cdot \cdot \sigma _ { 1 } ( W _ { 1 } \boldsymbol { X } ) \cdot \cdot \cdot ) \big ) ,
$$

where $\sigma _ { \ell } ( \cdot )$ is the per-layer activation function, and $W _ { \ell }$ is the weight matrix of the network at layer ℓ. Here, $\dot { \boldsymbol W } _ { \ell } \in \mathbb R ^ { d _ { \ell } \times d _ { \ell - 1 } }$ and $\boldsymbol { X } _ { \ell } \in \mathbb { R } ^ { d _ { \ell - 1 } \times N }$ , where the columns of $X _ { \ell }$ contain the activation inputs to layer ℓ. The storage requirement of the neural network $\mathcal { F } _ { \{ W _ { \ell } \} } ( \cdot )$ increases as the width of each layer grows. To fully represent ${ \mathcal { F } } ,$ , one needs to store the matrices $\{ W _ { \ell } \} _ { \ell = 1 } ^ { L }$ in high-precision. In practical computing, the weight matrices are usually represented by 32-bit floating-point numbers (FP32). This incurs an implementation barrier for neural networks to be used in devices with limited memory and computational power, such as edge devices.

The goal of post-training quantization is to replace the full-precision matrices $W _ { \ell }$ with low-bit quantized matrices $\widehat { W } _ { \ell }$ , without retraining $\mathcal { F }$ with data labels. A natural scheme for such a layer-wise quantization task is to replace $W _ { \ell }$ by

$$
\widehat { \pmb { W } } _ { \ell } = \mathcal { Q } _ { M } ( \pmb { W } _ { \ell } ) = \pmb { W } _ { \ell } + \pmb { Q } _ { \ell } ,
$$

where ${ \mathcal { Q } } _ { M }$ is the quantization function, M is the number of quantization levels, and $Q _ { \ell }$ is the additive weight-error matrix. Depending on the magnitudes of the entries in $W _ { \ell } ,$ the quantization alphabet can be adjusted. This is also known as the round-to-nearest (RTN) method in the literature.

GPTQ/OBS-Style Quantization. The shortcoming of such a direct quantization scheme is that the quantization error $Q _ { \ell }$ can be large and its efect may dominate. Consider the activation input at a single layer when the direct quantization scheme is used. Denote $X _ { \ell }$ as the activated input at layer ℓ. The product

$$
{ \widehat { \cal W } } _ { \ell } X _ { \ell } = { \cal W } _ { \ell } X _ { \ell } + Q _ { \ell } X _ { \ell }
$$

describes the efect of quantization. This reveals an objective of minimizing $\| Q _ { \ell } X _ { \ell } \| _ { F } ^ { 2 }$ , where $X _ { \ell }$ is the data that cannot be modified, whereas $Q _ { \ell }$ depends on input to the quantizer $\mathcal { Q } _ { M } ( \cdot )$ . The Optimum Brain Surgery/Generalized Post-Training Quantization (OBS/GPTQ) family of quantizers leverages this insight [12, 10].

Consider the quantization of a single row in $W _ { \ell } ,$ represented by $\mathbf { \Delta } w ^ { \top }$ , with quantized vector $\widehat { \pmb { w } } .$ . The distortion at the layer output is

$$
\gamma ( \widehat { \pmb { w } } ; \pmb { w } ) = \left\| ( \widehat { \pmb { w } } - \pmb { w } ) ^ { \top } \pmb { X } _ { \ell } \right\| _ { 2 } ^ { 2 } = ( \widehat { \pmb { w } } - \pmb { w } ) ^ { \top } C _ { \ell } ( \widehat { \pmb { w } } - \pmb { w } ) .
$$

Here, $\pmb { C } _ { \ell } = \pmb { X } _ { \ell } \pmb { X } _ { \ell } ^ { \top } \in \mathbb { R } ^ { d _ { \ell - 1 } \times d _ { \ell - 1 } }$ is empirical second-order information associated with the activation input. The OBS/GPTQ quantizer exploits the correlation among the entries of w through $C _ { \ell } .$ Suppose that the i-th entry of w is quantized as ${ \widehat { w } } _ { i } = \mathcal { Q } _ { M } ( w _ { i } )$ , and denote its quantization displacement by $p _ { i } = \widehat { w } _ { i } - w _ { i }$ . Instead of leaving all the remaining entries unchanged, one can compensate for $p _ { i }$ by modifying the other unquantized entries of w. Let $\delta _ { i }$ denote the total change applied to w. The optimal compensation is obtained by solving

$$
\operatorname* { m i n i m i z e } _ { \delta _ { i } } \quad \delta _ { i } ^ { \top } C _ { \ell } \delta _ { i } ,
$$

$$
\mathrm { s u b j e c t ~ t o ~ } e _ { i } ^ { \top } \delta _ { i } = p _ { i } ,
$$

where $e _ { i }$ is the i-th standard basis vector. Assuming $C _ { \ell } \succ \mathbf { 0 }$ , the solution is

$$
\begin{array} { r } { \delta _ { i } ^ { \star } = \frac { p _ { i } } { [ C _ { \ell } ^ { - 1 } ] _ { i i } } C _ { \ell } ^ { - 1 } e _ { i } . } \end{array}
$$

In the first step, the corresponding full-vector compensation is

$$
\begin{array} { r } { \pmb { w }  \pmb { w } + \frac { Q _ { M } ( w _ { i } ) - w _ { i } } { [ \pmb { C } _ { \ell } ^ { - 1 } ] _ { i i } } \pmb { C } _ { \ell } ^ { - 1 } \pmb { e } _ { i } . } \end{array}
$$

The i-th entry of the updated vector is exactly $\mathcal { Q } _ { M } ( w _ { i } )$ . In sequential quantization, the same compensation principle is applied to the remaining unquantized coordinates using the corresponding restricted curvature. Applying this procedure sequentially to the entries of w gives an OBS-style quantizer. $\mathrm { G P T Q }$ uses the same second-order compensation principle, but organizes the sequential updates eficiently so that the rows of $W _ { \ell }$ can be processed in parallel and the weight coordinates can be quantized in blocks.

Transform-Based Quantization. The $\mathrm { O B S / G P T Q }$ quantization schemes adjust the pre-quantized weight matrix using information from the covariance $C _ { \ell }$ . When the covariance is poorly conditioned, some coordinate directions have a larger variance than others. In this case, the uniform quantizer wastes resolution on low-variance directions while being too coarse for high-variance ones. An orthogonal transformation applied before quantization can redistribute the variance more evenly across coordinates.

Let $\pmb { R } \in \mathbb { R } ^ { d \times d }$ be an orthogonal matrix satisfying $\pmb { R } ^ { \top } \pmb { R } = \pmb { I }$ . Recall the convention $\gamma = \| ( \widehat { \pmb w } - $ $w ) ^ { \top } X _ { \ell } \Vert _ { 2 } ^ { 2 }$ . The layer output for a single weight row satisfies

$$
\pmb { w } ^ { \top } \pmb { X } _ { \ell } = ( \pmb { R } \pmb { w } ) ^ { \top } ( \pmb { R } \pmb { X } _ { \ell } ) .
$$

Defining $\widetilde { X _ { \ell } } = R X _ { \ell }$ and $\widetilde { w } = R u$ preserves the output exactly. The rotated curvature is

$$
\widetilde { C } _ { \ell } = \widetilde { X } _ { \ell } \widetilde { X } _ { \ell } ^ { \top } = R C _ { \ell } R ^ { \top } ,
$$

and the OBS/GPTQ compensation rule may be applied in the rotated domain with $\widetilde { C } _ { \ell } ^ { - 1 }$ in place of $C _ { \ell } ^ { - 1 }$ . For the full weight matrix, the rotated weights are $\widetilde { W } _ { \ell } = W _ { \ell } R ^ { \top }$

The goal of R is to make the entries of $\widetilde { W _ { \ell } }$ and $\widetilde { X _ { \ell } }$ more uniformly distributed across coordinates, so that a fixed quantization step size introduces approximately equal distortion in every direction. A random orthogonal R can spread concentrated directions across the coordinate basis, while structured choices such as the randomized Hadamard transform provide a similar efect at lower computational cost [30].

Error Propagation. The layer-wise reconstruction objective measures the efect of a weight error at the output of the quantized layer. An error introduced at layer ℓ also passes through subsequent layers. We study its efect on the final network output.

For a calibration input $\mathbf { \nabla } _ { \mathbf { x } _ { t } . }$ , let ${ \bf { x } } _ { \ell , t }$ be the input to layer ℓ and $\begin{array} { r } { \boldsymbol { z } _ { \ell , t } = \boldsymbol { W } _ { \ell } \boldsymbol { x } _ { \ell , t } } \end{array}$ its preactivation. Write $\widehat { \pmb { W } } _ { \ell } = \pmb { W } _ { \ell } + \pmb { Q } _ { \ell }$ , where $Q _ { \ell }$ is the resulting weight-error matrix. Suppose that only layer ℓ is perturbed. The preceding layers are unchanged, so ${ \pmb x } _ { \ell , t }$ is fixed, and the preactivation changes exactly by

$$
\Delta z _ { \ell , t } = Q _ { \ell } \mathbf { x } _ { \ell , t } .
$$

Let ${ \mathbf { \mathit { f } } } _ { t } ~ = ~ \mathcal { F } _ { \{ W _ { k } \} _ { k = 1 } ^ { L } } \bigl ( \mathbf { \mathit { x } } _ { t } \bigr )$ be the full-precision network output. Denote by $h _ { \ell }$ the map from the preactivation at layer ℓ to the final output, with all subsequent weights fixed. Thus $\begin{array} { r } { \pmb { f } _ { t } = h _ { \ell } ( \pmb { z } _ { \ell , t } ) } \end{array}$ . Its Jacobian at the full-precision preactivation is

$$
\begin{array} { r } { \pmb { J } _ { \ell , t } = \frac { \partial h _ { \ell } ( \pmb { z } ) } { \partial \pmb { z } ^ { \top } } \bigg | _ { \pmb { z } = \pmb { z } _ { \ell , t } } \in \mathbb { R } ^ { d _ { L } \times d _ { \ell } } . } \end{array}
$$

A first-order expansion gives

$$
\widehat { \pmb f } _ { t } - \pmb f _ { t } \approx \pmb J _ { \ell , t } \pmb Q _ { \ell } \pmb x _ { \ell , t } .
$$

The preactivation change is exact under this single-layer perturbation, and the downstream efect is linearized. Averaging the predicted output changes over the calibration set gives

$$
\begin{array} { r } { D _ { \mathrm { J a c } , \ell } ( \pmb { Q } _ { \ell } ) : = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \| \pmb { J } _ { \ell , t } \pmb { Q } _ { \ell } \pmb { x } _ { \ell , t } \| _ { 2 } ^ { 2 } = \mathrm { v e c } ( \pmb { Q } _ { \ell } ) ^ { \top } \pmb { H } _ { \mathrm { J a c } , \ell } ^ { \mathrm { f u l l } } \mathrm { v e c } ( \pmb { Q } _ { \ell } ) , } \end{array}\tag{1}
$$

where, using a column-wise vectorization, we have

$$
\begin{array} { r } { \pmb { H } _ { \mathrm { J a c } , \ell } ^ { \mathrm { f u l l } } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \big ( \pmb { x } _ { \ell , t } \pmb { x } _ { \ell , t } ^ { \top } \otimes \pmb { J } _ { \ell , t } ^ { \top } \pmb { J } _ { \ell , t } \big ) . } \end{array}\tag{2}
$$

Here, $D _ { \mathrm { J a c } , \ell }$ measures the network-output distortion, and ${ \cal H } _ { \mathrm { J a c , \ell } } ^ { \mathrm { f u l l } }$ is the quadratic matrix on the vectorized weight error.

The above characterization tracks an error introduced at layer ℓ through the subsequent network. It difers from the upstream-error compensation studied by [1], which accounts for errors from previously quantized layers when reconstructing the current layer’s output.

## 3 Flexible Noise Shaping by a Causal Feedback Quantizer

In the above derivations, we have seen that the Jacobian-weighted distortion is a function of the quantization error. Directly optimizing this distortion is dificult because the instantaneous quantization errors depend on the inputs to the fixed uniform quantizer $\mathcal { Q } _ { M } ( \cdot )$ . In this section, we describe a causal feedback quantizer that uses error-feedback coeficients as design variables.

For convenience, we again consider the quantization of a single row of the weight matrix. We drop the t and ℓ indices momentarily. Given a frozen network weight $\pmb { w } \in \mathbb { R } ^ { d }$ , consider the preactivation output $\pmb { y } = \pmb { X } ^ { \top } \hat { \pmb { w } }$ , where $\pmb { X } \in \mathbb { R } ^ { d \times N }$ . A causal feedback quantizer generates the quantized weights according to

$$
\begin{array} { r } { b _ { i } = w _ { i } + \sum _ { j < i } g _ { i j } q _ { j } , } \end{array}\tag{3a}
$$

$$
\widehat { w } _ { i } = \mathcal { Q } _ { M } ( b _ { i } ) = b _ { i } + q _ { i } ,\tag{3b}
$$

$$
q _ { i } = \widehat { w } _ { i } - b _ { i } ,\tag{3c}
$$

where $g _ { i j } \in \mathbb { R }$ are the feedback coeficients, $b _ { i }$ is the instantaneous quantizer input, and $q _ { i }$ is the instantaneous quantization error introduced when quantizing $b _ { i }$ . In the first equation, the summation is taken over $j < i$ , that is, the errors introduced by quantizing prior indices. Coupling $i = 1 , \ldots , d ,$ the error-feedback quantizer has the input-output relationship

$$
\widehat { \pmb { w } } = \pmb { w } + G \pmb { q } ,\tag{4}
$$

where $G$ is a lower-triangular matrix with unit diagonal entries. Thus, q collects the instantaneous quantization errors, whereas $G q$ is the shaped quantization error.

Such error-feedback quantizers have historically been deployed in analog-to-digital and digital-toanalog converters $\left( \mathrm { A D C s } / \mathrm { D A C s } \right)$ , specifically in the form of Sigma-Delta $( \Sigma \Delta )$ modulation [3]. In that context, the goal is to shape quantization noise toward a high-frequency range so that the baseband signal has little distortion. In addition, Σ∆ modulation has been deployed in image halftoning in the form of error difusion [17] and has recently been revisited for low-bit signal transmission in wireless communication [27, 15]. In that context, the feedback coeficients can be adjusted to shape quantization noise away from target users. In this work, we pursue noise shaping in model quantization.

Looking at equation 3c, it is dificult to characterize $q _ { i }$ tractably without knowing the quantizer input $b _ { i } .$ In general, $q _ { i }$ depends on both $G$ and w, making the direct design of G dificult. We therefore make the following design assumption.

Assumption 1. The vector q of quantization errors is random with second-order moment $\mathbb { E } \{ q q ^ { \top } \} =$ $\sigma _ { q } ^ { 2 } I _ { \mathrm { ~ ; ~ } }$ independent of any other variables.

Consider the layer-wise distortion

$$
\gamma ( \pmb { G } ) = \| \pmb { X } ^ { \top } ( \pmb { \widehat { w } } - \pmb { w } ) \| _ { 2 } ^ { 2 } = \| \pmb { X } ^ { \top } \pmb { G } \pmb { q } \| _ { 2 } ^ { 2 } = \pmb { q } ^ { \top } \pmb { G } ^ { \top } C \pmb { G } \pmb { q } ,
$$

where $C = X X ^ { \top }$ . Under Assumption 1, the expected loss is $\mathbb { E } \{ \gamma ( G ) \} = \sigma _ { q } ^ { 2 } \mathrm { t r } ( G ^ { \top } C G )$ . When treated as fixed, it does not afect the optimal design of G. Hence, the design problem is posed as

$$
\begin{array} { r l } { \underset { \boldsymbol { G } } { \mathrm { m i n i m i z e } } } & { { } \mathrm { t r } ( \boldsymbol { G } ^ { \top } \boldsymbol { C } \boldsymbol { G } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { { } g _ { i j } = 0 , \quad j > i , \qquad g _ { i i } = 1 . } \end{array}\tag{5}
$$

Assuming $C \succ \mathbf { 0 }$ , possibly after diagonal regularization, equation 5 is strictly convex. One can leverage the KKT conditions to obtain the global minimizer of equation 5. Consider the Lagrangian

$$
\mathcal { L } = \operatorname { t r } ( \pmb { G } ^ { \top } C \pmb { G } ) + \operatorname { t r } \big ( \pmb { \Lambda } ^ { \top } ( \pmb { G } - \pmb { I } ) \big ) ,\tag{6}
$$

where Λ is an upper-triangular matrix collecting the Lagrange multipliers associated with the constraints on $g _ { i j }$ . The stationarity condition reads

$$
\nabla _ { G } \mathcal { L } = 2 C G + \Lambda = { \bf 0 } ,
$$

implying $C G = - { \textstyle \frac { 1 } { 2 } } \mathbf { \Lambda }$ . We see that CG is upper triangular. This implies that $G ^ { \top } ( C G )$ is also upper triangular. Moreover, $G ^ { \top } C G$ is symmetric. Combining these facts yields $G ^ { \top } C G : = D .$ where D is diagonal. Taking inverse on both sides gives

$$
( G ^ { \top } C G ) ^ { - 1 } = D ^ { - 1 } \quad \Longrightarrow \quad C ^ { - 1 } = G D ^ { - 1 } G ^ { \top } .
$$

Consider the symmetric LDL decomposition $C ^ { - 1 } = L \bar { D } L ^ { \top }$ , where L is unit lower triangular and $\bar { D }$ is diagonal. Comparing these equations, the optimizer of equation 5 is $G ^ { \star } = L$ . This resembles the fixed-order covariance-based feedback structure associated with $\mathrm { G P T Q / L D L Q } \ [ 6 , 5 , 8 ]$

The above description resembles the GPTQ-style feedback structure from the perspective of designing the coeficients of a causal error-feedback quantizer. This permits us to generalize the design criterion from layer-wise distortion to a general loss function $f ,$ i.e.,

$$
\begin{array} { r l } { \underset { \pmb { G } } { \mathrm { m i n i m i z e } } } & { { } f ( \pmb { G } ; \pmb { \Theta } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { { } g _ { i j } = 0 , \quad j > i , \qquad g _ { i i } = 1 , } \end{array}\tag{7}
$$

where Θ denotes network information that may extend beyond the data covariance $C .$

The network information Θ may include outputs of already-quantized upstream layers. Consequently, $f ( G ; \Theta )$ can also express mismatched-input reconstruction when those outputs are held fixed during layerwise design; see Appendix A.

As a design example, let $\bar { C } _ { \ell } = C _ { \ell } / N$ denote the sample-averaged input covariance. For weight row $r ,$ define the input-coordinate Jacobian metric

$$
\begin{array} { r } { \pmb { H } _ { \mathrm { J a c } , \ell , r } ^ { \mathrm { r o w } } : = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \| \pmb { J } _ { \ell , t } \pmb { e } _ { r } \| _ { 2 } ^ { 2 } \pmb { x } _ { \ell , t } \pmb { x } _ { \ell , t } ^ { \top } \in \mathbb { R } ^ { d _ { \ell - 1 } \times d _ { \ell - 1 } } , } \end{array}\tag{8}
$$

Table 1: Test-set output distortion over 100 independently generated networks. The fixed-α columns report mean distortion when the same $\alpha$ is used for every network; $\alpha = 0$ is the $\mathrm { { G P T Q - s t y l e } }$ baseline. In the final three columns, $\alpha$ is selected separately for each network using calibration data only. Reduction is relative to $\alpha = 0 .$ , and $\mathrm { W i n / T i e / L o s s }$ counts networks on which the selected method has lower, equal, or higher test distortion than that baseline. Boldface in the fixed-α columns marks the lowest mean in each row.
<table><tr><td rowspan="2">M |</td><td colspan="9">Fixed α: mean test distortion</td><td colspan="4">Per-network α selection</td></tr><tr><td>0</td><td>0.01</td><td>0.03</td><td>0.1</td><td>0.2</td><td>0.4</td><td>0.6</td><td>0.8</td><td></td><td>0.99 Mean distortion</td><td>Reduction</td><td></td><td>Win/Tie/Loss</td></tr><tr><td>2</td><td>1.985016</td><td>1.978903</td><td>2.024264</td><td>2.032677</td><td>2.045072</td><td>2.038097</td><td>2.036754</td><td>2.032965</td><td>2.046547</td><td></td><td>1.691480</td><td>14.79%</td><td>75/20/5</td></tr><tr><td>4</td><td>0.136493</td><td>0.137417</td><td>0.133098</td><td>0.133740</td><td>0.134332</td><td>0.133041</td><td></td><td>0.131872</td><td>0.130650</td><td>0.131327</td><td>0.112534</td><td>17.55%</td><td>75/21/4</td></tr><tr><td>8</td><td>0.022783</td><td>0.022621</td><td>0.022686</td><td>0.022051</td><td>0.021853</td><td>0.022372</td><td>0.022070</td><td></td><td>0.021908</td><td>0.022043</td><td>0.018986</td><td>16.67%</td><td>72/24/4</td></tr><tr><td>12</td><td>0.008864</td><td>0.008892</td><td>0.008792</td><td>0.008754</td><td>0.008883</td><td>0.008964</td><td>0.009062</td><td></td><td>0.008925</td><td>0.008988</td><td>0.007633</td><td>13.88%</td><td>73/20/7</td></tr><tr><td>16</td><td>0.004611</td><td>0.004536</td><td>0.004499</td><td>0.004568</td><td>0.004534</td><td>0.004555</td><td>0.004587</td><td></td><td>0.004654</td><td>0.004631</td><td>0.003855</td><td>16.39%</td><td>72/27/1</td></tr></table>

where $\boldsymbol { e } _ { r } \in \mathbb { R } ^ { d _ { \ell } }$ selects the r-th output coordinate. A row-wise feedback design may then use

$$
\begin{array} { r } { f _ { r } ( G ; \Theta ) = \mathrm { t r } \big [ G ^ { \top } ( \bar { C } _ { \ell } + \alpha s _ { \ell } H _ { \mathrm { J a c } , \ell , r } ^ { \mathrm { r o w } } ) G \big ] , } \end{array}\tag{9}
$$

where $\alpha \geq 0$ controls the tradeof and $s _ { \ell } > 0$ is a fixed layer-wise scale. When a feedback matrix is shared across rows, we use the metric

$$
\pmb { H } _ { \mathrm { J a c } , \ell } ^ { \mathrm { i s o } } : = \frac { 1 } { { d _ { \ell } } } \sum _ { r = 1 } ^ { d _ { \ell } } \pmb { H } _ { \mathrm { J a c } , \ell , r } ^ { \mathrm { r o w } } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } { \frac { \lVert \pmb { J } _ { \ell , t } \rVert _ { F } ^ { 2 } } { d _ { \ell } } } \pmb { x } _ { \ell , t } \pmb { x } _ { \ell , t } ^ { \top } .\tag{10}
$$

Numerical Example. We investigate whether the flexibility provided by $G$ in equation 7 brings a benefit to network quantization. We construct a five-layer fully connected network with widths $1 2 $ $1 6 \to 1 6 \to 1 2 \to 8 \to 3$ . The first four layers are followed by the tanh activation, and the output layer is linear. The full-precision weights are drawn independently from zero-mean Gaussian distributions, with variances scaled according to the layer input dimensions. The network input data are generated from a four-dimensional latent Gaussian variable followed by a fixed linear mixing matrix and additive Gaussian noise. We draw 256 samples for calibration and an independent set of 1024 samples for evaluation. All covariance and Jacobian matrices are computed using the full-precision network on the calibration set.

We consider $M \in \{ 2 , 4 , 8 , 1 2 , 1 6 \}$ levels of uniform quantization. For each weight row, the quantization alphabet is fixed before applying error feedback and shared by all choices of $\alpha .$ For each M, we sweep over $\alpha \in \{ 0 , 0 . 0 1 , 0 . 0 3 , 0 . 1 , 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 0 . 9 9 \}$ . The row-wise Jacobian term is scaled by a common layer-wise factor $s _ { \ell }$ before applying α. The output distortion is measured by the test MSE between the quantized and full-precision networks. The experiment is repeated over 100 independently generated networks and datasets.

The numerical results are presented in Table 1. When α is selected using calibration data, the Jacobian-aware design reduces the mean test distortion relative to the GPTQ-style baseline for each M, with relative reductions ranging from 13.88% to 17.55%. It also achieves lower test distortion on at least 72% of the random instances for every M. The complete sweep shows that the preferred value of α depends on the quantization resolution and that a fixed positive α does not necessarily improve upon $\alpha = 0$ . These results indicate a potential benefit from a flexibly designed feedback matrix. The tradeof factor α provides a degree of freedom to balance local layer-wise distortion and downstream error propagation.

## 4 Joint Design of Rotation and Error Feedback

Section 2 described how an orthogonal rotation can redistribute weight outliers before quantization. Here, we consider its distinct role in shaping quantization error jointly with the feedback matrix. Following the convention $\widetilde { \boldsymbol { w } } = R \boldsymbol { w }$ and $\widehat { \widetilde { \pmb { w } } } - \widetilde { \pmb { w } } = G q$ , the error in the original coordinates is $\widehat { \pmb { w } } - \pmb { w } = \pmb { R } ^ { \top } \pmb { G } \pmb { q }$ . For

a fixed positive-definite efective metric H, the resulting trace problem is

$$
\operatorname* { m i n } _ { \begin{array} { c } { R ^ { \top } R = I } \\ { G \mathrm { ~ u n i t ~ l o w e r ~ t r i a n g u l a r } } \end{array} } f ( G , R ; H ) : = \mathrm { t r } \big ( G ^ { \top } R H R ^ { \top } G \big ) .\tag{11}
$$

The metric may, for example, be the local covariance $C _ { \ell }$ or a specified Jacobian-aware input-coordinate metric. Rotation alone leaves the trace unchanged: $f ( I , R ; H ) = \operatorname { t r } ( H )$ . Its efect in equation 11 arises from changing the coordinates in which triangular feedback operates.

For a fixed $\scriptstyle { R , }$ the reverse LDL factorization is

$$
R H R ^ { \top } = U _ { R } D _ { R } U _ { R } ^ { \top } ,
$$

where $U _ { R }$ is unit upper triangular and $D _ { R }$ is diagonal. The feedback solution from Section 3 then gives

$$
G _ { R } ^ { \star } = { U _ { R } ^ { - } } ^ { \top } , \qquad \operatorname* { m i n } _ { G } f ( G , R ; H ) = \mathrm { t r } ( D _ { R } ) .
$$

Since det $( D _ { R } ) = \operatorname* { d e t } ( H )$ , the arithmetic-geometric mean inequality yields the lower bound n det $( H ) ^ { 1 / n }$

Proposition 1 (Joint rotation-feedback optimum). Let ${ \pmb { H } } \succ \mathbf { 0 }$ and $\eta = \operatorname* { d e t } ( \pmb { H } ) ^ { 1 / n }$ . Then

$$
\operatorname* { m i n } _ { \substack { \textit { \textbf { R } } ^ { \top } R = I } } f ( G , R ; \mathbf { \boldsymbol { H } } ) = n \eta .
$$

$T o$ construct an optimizer, take a geometric mean decomposition $\pmb { H } ^ { 1 / 2 } = \pmb { Q } \pmb { T } \pmb { V } ^ { \top }$ , where $Q , V$ are orthogonal and T is upper triangular with diagonal entries $\sqrt { \eta }$ . Then

$$
R ^ { \star } = Q ^ { \top } , \qquad G ^ { \star } = \sqrt { \eta } T ^ { - \top } .
$$

Indeed, $R ^ { \star } H ( R ^ { \star } ) ^ { \top } = T T ^ { \top }$ , and substitution gives $( G ^ { \star } ) ^ { \top } R ^ { \star } H ( R ^ { \star } ) ^ { \top } G ^ { \star } = \eta I$ , attaining the lower bound. Thus, rotation equalizes the feedback-adjusted sensitivities, while feedback removes their triangular coupling. Replacing H by $\pmb { H } + \lambda \pmb { I } \succ \mathbf { 0 }$ gives the corresponding damped design. The optimum concerns a fixed metric and fixed second moment of the instantaneous quantization errors. In the experiments below, the construction is applied independently to input-coordinate blocks, so its attained bound is blockwise rather than an unrestricted full-layer optimum.

## 4.1 Finite-level quantization: the role of scale

Proposition 1 minimizes $f ( G , R ; H )$ for a fixed metric and a common, fixed second moment of the instantaneous errors. At finite resolution, the error statistics depend on the quantization scale and the chosen rotation and feedback. For the min–max grids in our MLP and $\mathrm { G P T - 2 }$ experiments, a rotated weight row or block has step size

$$
\Delta ( \pmb { R } ) = \frac { \operatorname* { m a x } _ { i } ( \pmb { R } \pmb { w } ) _ { i } - \operatorname* { m i n } _ { i } ( \pmb { R } \pmb { w } ) _ { i } } { M - 1 } .
$$

Although fixed during feedback, this scale need not be shared across rotations. Under the idealized approximation of independent, uniform rounding errors on $[ - \Delta / 2 , \Delta / 2 ]$ without overload,

$$
\mathbb { E } [ D ] \approx \frac { \Delta ( R ) ^ { 2 } } { 1 2 } f ( G , R ; H ) ,
$$

where $D = ( \widehat { \pmb { w } } - \pmb { w } ) ^ { \top } \pmb { H } ( \widehat { \pmb { w } } - \pmb { w } )$ . Thus, a smaller trace can be ofset by a larger scale. Orthogonality preserves the weight norm, not its coordinate range, and the trace construction does not optimize this range.

Feedback can also move quantizer inputs outside the fitted interval, producing saturation and nonisotropic errors. Widening the interval to accommodate these inputs increases $\Delta$ at fixed $M ,$ trading resolution for range. These efects explain why the trace optimum is not a finite-level performance guarantee. Appendix C verifies trace attainment without consistent finite-level gains; Table 3 likewise shows that RHT can outperform joint design. These results do not isolate scale as the sole cause. The proposition is therefore a fixed-statistics noise-shaping benchmark, not an optimum over scale, center, and finite-grid errors.

Table 2: Post-training quantization of MNIST- and CIFAR-10-trained MLPs. Each entry reports test logit MSE / accuracy drop in percentage points, averaged over 100 trained checkpoints. Mean full-precision accuracy is 97.0539% on MNIST and 47.5550% on CIFAR-10. Lower is better for both metrics. Boldface marks the lowest unrounded mean within each dataset and M.
<table><tr><td>Dataset</td><td>Method</td><td>M = 2</td><td>M = 4</td><td>M = 8</td><td>M = 16</td></tr><tr><td rowspan="6">MNIST</td><td>RTN</td><td>8.606688/32.57</td><td>0.382275/0.92</td><td>0.056979/0.11</td><td>0.011988/0.03</td></tr><tr><td>Local EFQ</td><td>1.695767/5.12</td><td>0.084901/0.20</td><td>0.013702/0.04</td><td>0.002940/-0.0008</td></tr><tr><td>Local EFQ + RHT</td><td>3.921692/15.91</td><td>0.139418/0.27</td><td>0.023697/0.04</td><td>0.004970/0.01</td></tr><tr><td>Obj. EFQ</td><td>1.637242/4.83</td><td>0.082757/0.21</td><td>0.013351/0.03</td><td>0.002856/-0.0003</td></tr><tr><td>Obj. EFQ + RHT</td><td>3.839215/15.08</td><td>0.135396/0.25</td><td>0.023065/0.04</td><td>0.004850/0.01</td></tr><tr><td>Obj. joint R, G</td><td>8.672428/44.34</td><td>0.140144/0.49</td><td>0.014719/0.03</td><td>0.002961/0.00</td></tr><tr><td rowspan="6">CIFAR-10</td><td>RTN</td><td>5.123931/14.39</td><td>0.260287/2.13</td><td>0.046357/0.43</td><td>0.010024/0.10</td></tr><tr><td>Local EFQ</td><td>1.838424/13.18</td><td>0.105956/0.93</td><td>0.018626/0.14</td><td>0.004025/0.04</td></tr><tr><td>Local EFQ + RHT</td><td>2.955296/18.37</td><td>0.139829/1.24</td><td>0.024744/0.21</td><td>0.005300/0.06</td></tr><tr><td>Obj. EFQ</td><td>1.815878/13.05</td><td>0.104643/0.93</td><td>0.018366/0.14</td><td>0.003951/0.04</td></tr><tr><td>Obj. EFQ + RHT</td><td>2.896634/18.06</td><td>0.138125/1.21</td><td>0.024415/0.22</td><td>0.005216/0.05</td></tr><tr><td>Obj. joint R, G</td><td>3.448898/19.50</td><td>0.122654/1.07</td><td>0.020478/0.22</td><td>0.004366/0.04</td></tr></table>

## 5 Numerical Results

## 5.1 PTQ of Trained MLPs

We study weight-only post-training quantization of five-linear-layer tanh networks trained on MNIST and CIFAR-10 [19, 18]. The MNIST network has widths 784 → $1 2 8  1 2 8  6 4  3 2  1 0$ , and the CIFAR-10 network has widths $3 0 7 2  1 2 8  1 2 8  6 4  3 2  1 0$ . Images are flattened before entering the networks, and tanh activations follow the first four linear layers. We compare roundto-nearest quantization (RTN), covariance-based causal error feedback (EFQ), randomized Hadamard transforms (RHT) followed by feedback, objective-aware (based on equations 9 and 8 or 10), and joint rotation–feedback design (based on Proposition 1).

All methods use fixed quantization grids with $M \in \{ 2 , 4 , 8 , 1 6 \}$ levels. Input coordinates within each weight row are partitioned into consecutive blocks of at most 32, with rotation and feedback designed independently within each block. For objective-aware methods, α is selected among candidates in $\{ 0 , 0 . 0 3 , 0 . 1 , 0 . 3 , 1 \}$ using data disjoint from metric calibration and testing. The blockwise design metric combines the sample-averaged input covariance with a trace-normalized, row-isotropic Jacobian metric (equation 10) and diagonal damping (λ>0). Candidate selection uses the full first-order networkoutput distortion (equations 1 and 2) , whereas construction of each feedback and rotation design uses the row-isotropic trace surrogate (based on the diagonalization of equation 10). Further details appear in Appendix D.2. Table 2 reports test logit MSE relative to the full-precision model and accuracy drop in percentage points.

Objective-aware EFQ has the lowest mean test logit MSE at every tested resolution on both datasets, although its advantage over covariance-based EFQ narrows at larger M. Accuracy drop does not always follow the same ordering: covariance-based EFQ has a slightly lower mean accuracy drop at M = 4 on both datasets, despite its higher logit MSE. Joint rotation-feedback performs poorly at $M = 2$ on MNIST. A separate paired diagnostic in the appendix shows that joint design attains its blockwise fixedstatistics trace bound without necessarily reducing realized finite-level error. That diagnostic fixes α = 0 and is distinct from the selected-α comparisons reported here.

## 5.2 PTQ of Language Models

We evaluate feedback design on GPT-2 [26], Llama-2 [29], and Qwen3 [28] language models. Table 3 reports perplexity, next-token accuracy, and final-logit MSE at $M \in \{ 8 , 1 6 \}$ levels. “Local” denotes a projection-input metric, while “objective-aware” denotes a downstream-sensitive logit metric. Under each metric, we compare EFQ without rotation, RHT followed by EFQ, and joint rotation-feedback

Table 3: Post-training quantization on five language models. Each entry reports PPL / nexttoken accuracy $( \% ) ~ /$ final-logit MSE. Lower PPL and MSE and higher accuracy are better. Boldface marks the best of the six methods for each model and bit width. “Local” and “objective-aware” specify the design metric; EFQ, RHT+EFQ, and joint R, G specify the construction.
<table><tr><td>Model</td><td>M</td><td>Local EFQ</td><td>Local EFQ+RHT</td><td>Local joint R, G</td><td>Obj.-EFQ</td><td>Obj.-EFQ+RHT</td><td>Obj. joint R, G</td></tr><tr><td rowspan="2">GPT-2</td><td>8 16</td><td>118.54 /24.21 /257.89105.41 99.65 /25.50 /85.48 100.69</td><td>/24.60 221.42112.66 /25.30 /34.73</td><td>24.26 /186.16119.03 98.41 /26.14/31.10</td><td>24.45 /222.64105.38 99.60 7 /25.40/ /72.64</td><td>/24.40 /206.28106.94 99.95 /25.50 28.13</td><td>24.65 /182.59 98.25 / 26.09 /57.73</td></tr><tr><td>8</td><td>11.312 /50.22</td><td>9.120</td><td>/51.70</td><td>/50.81</td><td>8.704 / 53.73</td><td>9.338 / 52.55 /2.253</td></tr><tr><td rowspan="2">Llama-2-7B</td><td></td><td>/3.989 7.808 55.69 /0.791</td><td>/53.15 /2.077 7.450 /56.34 0.347</td><td>9.893 /2.794 7.563 /56.06 /0.498</td><td>11.002 /3.745 7.736 /55.84 /0.685</td><td>/1.739 7.431/ /56.44 /0.309</td><td>7.521 / /56.18 /0.453</td></tr><tr><td>16</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Llama-2-13B8</td><td></td><td>8.209 54.56 0.842</td><td>7.585 55.73 0.619</td><td>8.238 /54.52 /0.806</td><td>7.929 55.13 /0.765</td><td>7.501 /55.97 /0.579</td><td>7.989 /54.86 0.733</td></tr><tr><td>16</td><td>6.714 58.00 /0.201</td><td>6.611 58.25 /0.137</td><td>6.671 /58.03 /0.182</td><td>6.630 /58.15 /0.168</td><td>6.593 58.27 /0.126</td><td>6.675 /58.04 /0.168</td></tr><tr><td rowspan="2">Qwen3-8B</td><td>8</td><td>20.227 43.52 5.933</td><td>16.485 46.91 /2.583</td><td>16.857 46.45 /3.367</td><td>20.531 7 43.76</td><td>4.616 16.118 47.08 /2.536</td><td>16.839 46.50 3.008</td></tr><tr><td>16 14.140</td><td>49.53 1.156</td><td>13.469 /50.23 / 0.570</td><td>13.473 /50.12 /0.680</td><td>13.750 49.71 /0.952</td><td>13.312 /50.32 /0.554</td><td>13.449 /50.18 /0.659</td></tr><tr><td rowspan="2">Qwen3-14B</td><td></td><td></td><td>13.924 /49.27 /1.709</td><td>14.196 /48.94 / 1.992</td><td>16.561 / 46.63</td><td>49.44 /1.658</td><td></td></tr><tr><td>8 16 12.447</td><td>16.487 46.47 2.774 51.35 /0.750</td><td>11.984/51.88 /0.407</td><td>12.067 7/51.76 / 0.482</td><td>12.329 /51.37</td><td>/ 2.773 13.734 /0.716 11.912 51.93 /0.385</td><td>14.077 49.00 /1.947 12.045 / 51.84 / 0.475</td></tr></table>

design.

For GPT-2, we quantize the 48 attention and MLP projections in input-coordinate blocks of size 32, retaining full-precision embeddings. Objective-aware feedback uses stochastic downstream-sensitivity estimates; α is selected by validation logit MSE. The reported GPT-2 rows use 8064 scored positions and a separate evaluation protocol from the larger models. Its full-precision perplexity and next-token accuracy are 96.35 and 26.09%, respectively.

For Llama-2-7B, Llama-2-13B, Qwen3-8B, and Qwen3-14B, metric calibration uses seed 7 and 128 non-overlapping WikiText-2 [23] training sequences of length 512. Evaluation uses 340326 scored positions for each Llama model and 298424 for each Qwen model. Comparisons in the table are made within each model and bit width, not across the two evaluation protocols.

On the four larger models, objective-aware design lowers final-logit MSE and raises next-token accuracy in all 24 comparisons matched by model, bit width, and construction; it lowers perplexity in 21 comparisons. Objective-aware RHT+EFQ has the best three reported metrics among the six methods in each of settings considered. GPT-2 is less uniform: at three and four bits, its best perplexity, accuracy, and logit MSE do not all belong to the same method.

On the four larger models, RHT+EFQ outperforms joint rotation-feedback on all three reported metrics in the 16 comparisons. This finite-level ordering does not contradict joint design’s blockwise trace optimum: the objective does not account for rotation-dependent finite-grid errors.

## 5.3 Direct Feedback Tuning in QAT

The PTQ experiments above choose a design metric and compute the corresponding feedback matrix through the trace problem. We next consider a complementary route: updating the free entries of G against the loss of the hard-quantized network using a straight-through backward rule.

Synthetic regression. We consider a two-linear-layer regression network with dimensions 4 → 6 → 3 and a tanh hidden activation. One unit lower-triangular feedback matrix is shared across the weight rows of each layer. The data-loss objective is

$$
\begin{array} { r } { \displaystyle \underset { W , G } { \operatorname* { m i n } } ~ \frac { 1 } { N _ { \mathrm { t r } } d _ { L } } \sum _ { t = 1 } ^ { N _ { \mathrm { t r } } } \left\| \mathcal { F } _ { \boldsymbol { Q } _ { M } ( { W } ; { G } ) } ( \boldsymbol { x } _ { t } ) - \boldsymbol { y } _ { t } \right\| _ { 2 } ^ { 2 } . } \end{array}
$$

Here $\mathcal { Q } _ { M } ( W ; G )$ applies the causal quantizer in equation 3 separately to each weight row in inputcoordinate order. The forward pass uses hard nearest rounding on fixed per-row grids; the backward pass uses a straight-through surrogate derivative.

Across 100 independently generated datasets, all methods within a trial start from the same fitted full-precision checkpoint. Each trial has 160 training, 80 validation, and 256 test pairs, with $M \in \{ 2 , 4 , 8 \}$ Hard-quantized validation MSE selects the checkpoint. Table 4 reports test MSE against the generated targets; Appendix F gives the full training protocol.

Table 4: Mean test MSE over 100 paired synthetic regression trials; lower is better. Feedbackonly tuning holds the network fixed and starts from covariance-designed feedback. Both displayed QAT arms start from $G = I$ . Boldface marks the lowest mean among the quantized methods shown.
<table><tr><td>Method</td><td> $M = 2$ </td><td> $M = 4$ </td><td> $M = 8$ </td></tr><tr><td>Full precision</td><td>0.01391</td><td>0.01391</td><td>0.01391</td></tr><tr><td>RTN</td><td>0.63739</td><td>0.06576</td><td>0.02395</td></tr><tr><td>Local PTQ</td><td>0.48603</td><td>0.05473</td><td>0.02078</td></tr><tr><td>Feedback QAT(G)</td><td>0.28594</td><td>0.04055</td><td>0.01899</td></tr><tr><td>Weight-only QAT; G = I fixed</td><td>0.25458</td><td>0.04118</td><td>0.01914</td></tr><tr><td>Joint QAT;  $G _ { 0 } = { \cal I }$ </td><td>0.24524</td><td>0.04013</td><td>0.01784</td></tr></table>

Table 5: Held-out next-character perplexity on the Shakespeare character dataset, averaged over ten paired seeds. Perplexity is computed for each seed before averaging; lower is better. Evaluation uses the held-out second half of the validation file. Boldface marks the lowest mean among quantized methods.
<table><tr><td>Method</td><td> $M = 2$ </td><td> $M = 4$ </td><td> $M = 8$ </td><td> $M = 1 6$ </td></tr><tr><td>Init. Full precision</td><td>12.967</td><td>12.967</td><td>12.967</td><td>12.967</td></tr><tr><td>Cont. Full precision</td><td>12.374</td><td>12.374</td><td>12.374</td><td>12.374</td></tr><tr><td>RTN</td><td>18.282</td><td>13.046</td><td>12.986</td><td>12.970</td></tr><tr><td>Local PTQ</td><td>13.629</td><td>12.999</td><td>12.969</td><td>12.970</td></tr><tr><td>Feedback QAT (G)</td><td>13.074</td><td>12.842</td><td>12.865</td><td>12.910</td></tr><tr><td>Weight QAT (W)</td><td>12.561</td><td>12.404</td><td>12.390</td><td>12.382</td></tr><tr><td>Joint QAT (W, G)</td><td>12.528</td><td>12.392</td><td>12.381</td><td>12.380</td></tr></table>

Tuning G with the fitted network fixed lowers mean test MSE relative to covariance-feedback PTQ at each tested resolution. In the matched identity-initialized QAT comparison, joint training gives lower mean test MSE than weight-only QAT at all three resolutions. The complete comparison in Appendix F also includes covariance-initialized QAT arms.

Small character-level language model. We also evaluate feedback tuning on a character-level nanoGPT model trained on the Shakespeare character dataset [14]. It has 2 transformer blocks, 2 attention heads, and embedding width 64. We quantize 4 attention and MLP projection matrices per block in input-coordinate blocks of size 32.

For each of ten seeds, all arms start from the same full-precision checkpoint trained for 200 steps. We compare RTN, covariance-feedback PTQ, feedback-only tuning with network coeficients fixed, weightonly QAT with $G = { \cal I } ,$ and joint QAT initialized at G = I. The tuning arms receive 100 additional steps. An unquantized copy of the checkpoint also receives 100 additional steps as a training-budget reference.

Feedback-only tuning has lower mean held-out perplexity than fixed covariance feedback at each tested level. Joint QAT has a slightly lower unrounded mean than matched weight-only QAT, but the diferences are small; at M = 16, both means round to 12.38. Continued full-precision training has lower mean perplexity than either QAT arm at every level. This experiment demonstrates that flexible feedback coeficients can be trainable on language models.

## 6 Conclusions

This work formulated error feedback as a quantizer-design problem. We treated error feedback coeficients as design variables rather than a prescribed compensation rule. For a specified input-coordinate metric and a fixed isotropic second moment of the instantaneous quantization errors, this viewpoint yields covariance-based feedback as a reference design and a constructive optimum for joint design on rotation and feedback. Experiments show that adapting the feedback criterion can improve output fidelity, and feedback coeficients can also be jointly trained with model weights in a small QAT setting. They expose a limitation of the trace formulation: attaining its optimum does not necessarily reduce realized error under finite-level quantization. The resulting framework ofers a way to formulate and test feedback designs beyond a single reconstruction objective. A key direction for future work is to account for how feedback and rotation change the errors produced by the quantizer, and to examine whether learned feedback remains useful in larger pretrained models. In this sense, the contribution is both a characterization of what fixed-statistics design can achieve and a starting point for quantizer designs that reflect realized network behavior more closely.

## References

[1] Yamato Arai and Yuma Ichikawa. Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization. In Advances in Neural Information Processing Systems, volume 38, 2025.

[2] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. In Advances in Neural Information Processing Systems, volume 37, 2024.

[3] P.M. Aziz, H.V. Sorensen, and J. vn der Spiegel. An overview of Sigma-Delta converters. IEEE Signal Process. Magazine, 13(1):61–84, 1996.

[4] Ron Banner, Yury Nahshan, and Daniel Soudry. Post Training 4-bit Quantization of Convolutional Networks for Rapid-Deployment. In Advances in Neural Information Processing Systems, volume 32, 2019.

[5] Johann Birnick. The lattice geometry of neural network quantization: A short equivalence proof of gptq and babai's algorithm. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 101779–101787, 2026.

[6] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. QuIP: 2-Bit Quantization of Large Language Models With Guarantees. In Advances in Neural Information Processing Systems, volume 36, 2023.

[7] Jiale Chen, Torsten Hoefler, and Dan Alistarh. GPTQ-2D: Cubic-Time Two-Sided Adaptive Rounding, 2026.

[8] Jiale Chen, Yalda Shabanzadeh, Elvir Crnˇcevi´c, Torsten Hoefler, and Dan Alistarh. The geometry of LLM quantization: GPTQ as babai’s nearest plane algorithm. In The Fourteenth International Conference on Learning Representations, 2026.

[9] Elias Frantar and Dan Alistarh. Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning. In Advances in Neural Information Processing Systems, volume 35, 2022.

[10] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. In International Conference on Learning Representations, 2023.

[11] Song Han, Huizi Mao, and William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Hufman Coding. In International Conference on Learning Representations, 2016.

[12] Babak Hassibi and David G. Stork. Second Order Derivatives for Network Pruning: Optimal Brain Surgeon. In Advances in Neural Information Processing Systems, volume 5, pages 164–171, 1992.

[13] Yi Jiang, William W. Hager, and Jian Li. The Geometric Mean Decomposition. Linear Algebra and its Applications, 396:373–384, 2005.

[14] Andrej Karpathy. nanogpt. https://github.com/karpathy/nanoGPT, 2022.

[15] Wai-Yiu Keung and Wing-Kin Ma. Spatial sigma-delta modulation for coarsely quantized massive mimo downlink: Flexible designs by convex optimization. IEEE Open Journal of Signal Process., 5:520–538, 2024.

[16] Jinuk Kim, Marwa El Halabi, Wonpyo Park, Clemens Js Schaefer, Deokjae Lee, Yeonhong Park, Jae W. Lee, and Hyun Oh Song. GuidedQuant: Large language model quantization via exploiting end loss guidance. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 30011–30037. PMLR, 13–19 Jul 2025.

[17] T.D. Kite, B.L. Evans, A.C. Bovik, and T.L. Sculley. Digital halftoning as 2-D Delta-Sigma modulation. In Proc. of Int. Conf. on Image Process., volume 1, pages 799–802 vol.1, 1997.

[18] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[19] Yann LeCun, Corinna Cortes, and Christopher JC Burges. The mnist database of handwritten digits, 1998.

[20] Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction. In International Conference on Learning Representations, 2021.

[21] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. In Proceedings of Machine Learning and Systems, volume 6, 2024.

[22] Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. SpinQuant: LLM Quantiza tion with Learned Rotations. In International Conference on Learning Representations, 2025.

[23] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

[24] Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or Down? Adaptive Rounding for Post-Training Quantization. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7197–7206. PMLR, 2020.

[25] Markus Nagel, Mart van Baalen, Tijmen Blankevoort, and Max Welling. Data-Free Quantization Through Weight Equalization and Bias Correction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

[26] Alec Radford, Jef Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[27] Mingjie Shao, Wing-Kin Ma, Qiang Li, and A. Lee Swindlehurst. One-bit sigma-delta mimo precoding. IEEE Journal of Sel. Topics in Signal Process., 13(5):1046–1061, 2019.

[28] Qwen Team. Qwen3 technical report, 2025.

[29] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[30] Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks. In Proceedings of the 41st International Conference on Machine Learning, 2024.

[31] Albert Tseng, Zhaofeng Sun, and Christopher De Sa. Model-Preserving Adaptive Rounding. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

[32] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and Eficient Post-Training Quantization for Large Language Models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 38087–38099. PMLR, 2023.

[33] Jian-Kang Zhang, A. Kavcic, and Kon Max Wong. Equal-diagonal qr decomposition and its application to precoder design for successive-cancellation detection. IEEE Transactions on Information Theory, 51(1):154–172, 2005.

[34] Shihao Zhang, Haoyu Zhang, Ian Colbert, and Rayan Saab. Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization, 2025.

## A Related Compensation Objectives

The abstraction $f ( G _ { \ell } ; \Theta _ { \ell } )$ separates the feedback matrix being designed from the context held fixed during that design. Let $\hat { W } _ { \ell } ( G _ { \ell } )$ be the weights produced by our causal quantizer and define

$$
\begin{array} { r } { E _ { \ell } ( G _ { \ell } ) = \widehat { W } _ { \ell } ( G _ { \ell } ) - W _ { \ell } . } \end{array}
$$

For a layerwise design problem, $\Theta _ { \ell }$ may contain the original weights, quantization grid, calibration data, design metric, and the outputs of already-quantized upstream layers. In particular, let $X _ { \ell }$ and $\widetilde { X _ { \ell } }$ contain the full-precision and upstream-quantized inputs, respectively, as columns. Both are held fixed while designing $G _ { \ell }$

This notation permits a mismatched-input reconstruction criterion:

$$
f _ { \mathrm { M } } ( G _ { \ell } ; \Theta _ { \ell } ) = \Big \| { \cal W } _ { \ell } X _ { \ell } - \widehat { { \cal W } } _ { \ell } ( G _ { \ell } ) \widetilde { X } _ { \ell } \Big \| _ { F } ^ { 2 } = \Big \| { \cal W } _ { \ell } \big ( { \cal X } _ { \ell } - \widetilde { X } _ { \ell } \big ) - { \cal E } _ { \ell } \big ( G _ { \ell } \big ) \widetilde { X } _ { \ell } \Big \| _ { F } ^ { 2 } .
$$

The first term inside the second norm records the upstream-input mismatch; the second records the efect of the current layer’s quantized weights. Thus, upstream mismatch can be represented in $\Theta _ { \ell }$ without changing the design variable in Eq. (5). This criterion is generally not the homogeneous fixed-statistics trace objective solved by our rotation-feedback proposition.

For the comparison below, let $e _ { \ell , r } ( G _ { \ell } )$ be row r of ${ \bf \nabla } { \bf E } _ { \ell } ( { \bf G } _ { \ell } )$ , written as a column vector. To describe end-loss-gradient weighting, define

$$
\mathbf { \nabla } F _ { \ell , r } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \left( \frac { \partial \mathcal { L } _ { t } } { \partial z _ { \ell , r , t } } \right) ^ { 2 } \mathbf { x } _ { \ell , t } \mathbf { x } _ { \ell , t } ^ { \top } .
$$

This row-wise metric is distinct from the output-Jacobian metric used in our experiments.

The table compares objectives and compensation mechanisms, not identical feasible sets or algorithms. In particular, Qronos’s causal difusion should not be mistaken for the absence of triangular feedback. Its matrix denoted $G$ is a cross-Gram matrix between upstream-quantized and full-precision inputs, not our unit lower-triangular $G _ { \ell }$ . If feedback matrices across multiple layers are optimized jointly, then $\widetilde { X _ { \ell } }$ depends on earlier feedback choices and is no longer fixed layerwise context.

## B Proofs and Matrix-Design Details

Throughout this section, $\pmb { H } \in \mathbb { R } ^ { n \times n }$ is symmetric positive definite, $\ b { R } \in \mathbb { R } ^ { n \times n }$ is orthogonal, and $G \in \mathbb { R } ^ { n \times n }$ is unit lower triangular. The results concern the trace objective for a specified metric H.

Table 6: Related compensation criteria viewed through $f ( G _ { \ell } ; \Theta _ { \ell } )$
<table><tr><td>Method</td><td>Compensation criterion</td><td>Relationship to our feedback design</td></tr><tr><td>GPTQ / QuIP- Layer-output LDLQ [10, 6]</td><td> $\mathrm { t r } ( E _ { \ell } C _ { \ell } E _ { \ell } ^ { \top } )$ </td><td>quadratic The closest linear-feedback precedent. With the input covariance as metric, our fixed-isotropic-error design surrogate is proportional to  $\mathrm { t r } ( G _ { \ell } ^ { \top } C _ { \ell } G _ { \ell } )$ </td></tr><tr><td>GuidedQuant [16]</td><td>End-loss-gradient-weighted quadratics,  $\begin{array} { r } { \sum _ { r } e _ { \ell , r } ^ { \top } \pmb { F } _ { \ell , r } \pmb { e } _ { \ell , r } } \end{array}$ </td><td>row The criterion can be evaluated on  $e _ { \ell , r } ( G _ { \ell } )$  . Its weighting uses end-loss gra- dients rather than our output-Jacobian metric; this objective-level connection does not identify the two quantization al-</td></tr><tr><td>YAQA [31]</td><td>A structured quadratic approx- Its criterion can be restricted to imation to full-model output- vec distribution KL.</td><td> $\left( E _ { \ell } ( G _ { \ell } ) \right)$  Its adaptive rounding also uses output-side feedback, beyond a single shared input-coordinate  $G _ { \ell } .$ </td></tr><tr><td> $\mathrm { Q E P \ [ 1 ] }$ </td><td>Compensation for quantization er- Upstream-quantized states can be in- rors propagated from preceding lay- cluded in ers.</td><td> $\Theta _ { \ell } ,$  permitting an upstream- error-aware objective for  $G _ { \ell } .$  This is a restriction to our quantizer, not an equiv- alence with QEP&#x27;s procedure.</td></tr><tr><td>Qronos [34]</td><td>Mismatched-input layer reconstruc- Taking tion,  $\| { \mathbf { } W _ { \ell } } { X } _ { \ell } - \widehat { W } _ { \ell } \widetilde { { X } } _ { \ell } \| _ { F } ^ { 2 }$ </td><td> ${ \widehat { \mathbf { W } } } _ { \ell } \mathbf { \Phi } = { \widehat { \mathbf { W } } } _ { \ell } ( G _ { \ell } )$  gives  $f _ { \mathrm { M } }$  above. Qronos uses causal diffusion to future co- ordinates, but its full procedure also cor- rects for the upstream-input mismatch.</td></tr></table>

## B.1 Optimal feedback for a fixed rotation

## Fix R and set

$$
\pmb { A } = \pmb { R } \pmb { H } \pmb { R } ^ { \top } \succ \pmb { 0 } .
$$

We first consider

$$
\operatorname* { m i n } _ { G { \mathrm { ~ u n i t ~ l o w e r ~ t r i a n g u l a r } } } \mathrm { t r } ( G ^ { \top } A G ) .
$$

Because $A \ \succ \ 0$ , the objective is strictly convex in the free entries of G. At its unique minimizer, stationarity with respect to those entries requires AG to be upper triangular. Consequently, $G ^ { \top } A G$ is both upper triangular and symmetric, and is therefore diagonal.

Write the reverse LDL factorization as

$$
\mathbf { } A = U D U ^ { \top } ,
$$

where U is unit upper triangular and D is positive diagonal. The matrix $G ^ { \star } = U ^ { - \top }$ is unit lower triangular and satisfies

$$
( G ^ { \star } ) ^ { \top } A G ^ { \star } = D .
$$

It satisfies the stationarity condition and hence, by strict convexity, is the unique minimizer. In particular,

$$
\operatorname* { m i n } _ { \pmb { G } } \operatorname { t r } ( \pmb { G } ^ { \top } \pmb { A } \pmb { G } ) = \operatorname { t r } ( \pmb { D } ) .
$$

Equivalently, $\pmb { A } ^ { - 1 } = \pmb { G } ^ { \star } \pmb { D } ^ { - 1 } ( \pmb { G } ^ { \star } ) ^ { \top }$

## B.2 Proof of Proposition 1

For any feasible pair (G, R), define

$$
B = G ^ { \top } R H R ^ { \top } G .
$$

The matrix B is positive definite. Since G is unit lower triangular and R is orthogonal,

$$
\operatorname* { d e t } ( B ) = \operatorname* { d e t } ( G ) ^ { 2 } \operatorname* { d e t } ( R ) ^ { 2 } \operatorname* { d e t } ( H ) = \operatorname* { d e t } ( H ) .
$$

The arithmetic geometric mean inequality applied to the eigenvalues of B therefore gives

$$
\operatorname { t r } ( B ) \geq n \operatorname* { d e t } ( B ) ^ { 1 / n } = n \operatorname* { d e t } ( H ) ^ { 1 / n } .
$$

Let $\eta = \operatorname* { d e t } ( \pmb { H } ) ^ { 1 / n }$ . Take a geometric mean decomposition

$$
\pmb { H } ^ { 1 / 2 } = \pmb { Q } \pmb { T } \pmb { V } ^ { \top } ,
$$

where $Q , V$ are orthogonal and $_ { \mathbf { T } }$ is upper triangular with diagonal entries $\sqrt { \eta } \ \left[ 1 3 \right]$ . Set

$$
R ^ { \star } = Q ^ { \top } , \qquad G ^ { \star } = \sqrt { \eta } T ^ { - \top } .
$$

Since $\pmb { T } ^ { - \top }$ is lower triangular with diagonal entries $1 / \sqrt { \eta } , G ^ { \star }$ is unit lower triangular. Also,

$$
R ^ { \star } H ( R ^ { \star } ) ^ { \top } = T T ^ { \top } ,
$$

and thus

$$
( G ^ { \star } ) ^ { \top } R ^ { \star } H ( R ^ { \star } ) ^ { \top } G ^ { \star } = \eta I .
$$

Its trace is nη, attaining the lower bound.

## B.3 Scope of the blockwise construction

The proposition allows an unrestricted orthogonal rotation. In the experiments, the input coordinates are partitioned into blocks $I _ { 1 } , \ldots , I _ { K }$ , and rotation and feedback are restricted to be block diagonal. Let $n _ { b } = \left| I _ { b } \right|$ and let $\pmb { H } _ { b b } = \pmb { H } [ I _ { b } , I _ { b } ]$ . For this restricted problem, the trace separates across blocks. If every $\pmb { H } _ { b b }$ is positive definite, the blockwise optimum is

$$
T _ { \mathrm { b l o c k } } ^ { \star } = \sum _ { b = 1 } ^ { K } n _ { b } \operatorname* { d e t } ( H _ { b b } ) ^ { 1 / n _ { b } } .
$$

This is the bound attained by independent blockwise constructions. It is not, in general, the unrestricted full-coordinate bound n det $( \dot { \pmb { H } } ) ^ { 1 / n }$ . If the design uses damping, the reported trace and bound must both use the same damped metric.

## C Trace Optimality and Realized Quantization Error

We distinguish attainment of the block-restricted trace optimum from the behavior of the corresponding finite-level quantizer. For layer ℓ and input-coordinate block $b ,$ let

$$
\begin{array} { r } { T _ { \ell b } = \mathrm { t r } \big ( G _ { \ell b } ^ { \top } R _ { \ell b } H _ { \ell b } R _ { \ell b } ^ { \top } G _ { \ell b } \big ) , B _ { \ell b } = n _ { \ell b } \operatorname* { d e t } ( H _ { \ell b } ) ^ { 1 / n _ { \ell b } } , } \end{array}
$$

where $n _ { \ell b }$ is the block width. Since the same block design is applied to all $d _ { \ell }$ output rows of layer $\ell ,$ the whole-network trace ratio for each seed is

$$
\rho _ { \mathrm { t r a c e } } = \frac { \sum _ { \ell } d _ { \ell } \sum _ { b \in \ell } T _ { \ell b } } { \sum _ { \ell } d _ { \ell } \sum _ { b \in \ell } B _ { \ell b } } .
$$

A value of one verifies the restricted trace construction to numerical precision.

Table 7: Trace optimality versus realized finite-level error on MNIST at $\alpha = 0$ , averaged over 100 paired seeds. The trace ratio is computed for each seed from row-weighted block traces and determinant bounds, then averaged over seeds. Realized metric error is the aggregate blockwise quadratic error; test logit MSE is measured for the fully quantized network. Lower is better.
<table><tr><td>M</td><td>Method</td><td>Trace ratio</td><td>Realized metric error</td><td>Test logit MSE</td></tr><tr><td>2</td><td>Covariance feedback</td><td>1.98079</td><td>98.60880</td><td>1.69577</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>347.72978</td><td>9.21850</td></tr><tr><td>4</td><td>Covariance feedback</td><td>1.98079</td><td>9.73280</td><td>0.08490</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>22.36973</td><td>0.15830</td></tr><tr><td>8</td><td>Covariance feedback</td><td>1.98079</td><td>1.76087</td><td>0.01370</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>1.79542</td><td>0.01529</td></tr><tr><td>16</td><td>Covariance feedback</td><td>1.98079</td><td>0.38108</td><td>0.00294</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>0.33103</td><td>0.00303</td></tr></table>

Table 8: Trace optimality versus realized finite-level error on CIFAR-10 at $\alpha = 0$ , averaged over 100 paired seeds. Quantities and aggregation follow Table 7. Lower is better.
<table><tr><td>M</td><td>Method</td><td>Trace ratio</td><td>Realized metric error</td><td>Test logit MSE</td></tr><tr><td>2</td><td>Covariance feedback</td><td>1.13685</td><td>107.21706</td><td>1.83842</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>191.77801</td><td>3.52017</td></tr><tr><td>4</td><td>Covariance feedback</td><td>1.13685</td><td>10.39471</td><td>0.10596</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>12.26181</td><td>0.12758</td></tr><tr><td>8</td><td>Covariance feedback</td><td>1.13685</td><td>1.89431</td><td>0.01863</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>2.20850</td><td>0.02126</td></tr><tr><td>16</td><td>Covariance feedback</td><td>1.13685</td><td>0.41161</td><td>0.00403</td></tr><tr><td></td><td>Joint rotation-feedback</td><td>1.00000</td><td>0.47938</td><td>0.00453</td></tr></table>

For each output row, let ${ \mathbf { } q } _ { b , r }$ denote the instantaneous quantization-error vector generated within block b. Its shaped error in original input coordinates is $\mathbf { e } _ { b , r } = \mathbf { R } _ { b } ^ { \top } G _ { b } \mathbf { q } _ { b , r } .$ The reported realized metric error aggregates $e _ { b , r } ^ { \top } H _ { b b } e _ { b , r }$ over blocks and rows. This blockwise quadratic omits cross-block metric terms and is distinct from the test logit MSE of the fully quantized network. Table 7 reports these measurements under diferent set up.

Across 100 paired seeds, joint rotation-feedback attains the blockwise determinant bound to numerical precision on both datasets. This trace ordering does not generally carry over to finite-level performance. Covariance feedback has lower mean test logit MSE at every tested resolution on both MNIST and CIFAR-10. On MNIST at $M = 1 6 .$ , joint design has lower mean realized blockwise metric error, but its mean test logit MSE remains slightly higher. Thus, attainment of the trace bound, realized blockwise metric error, and end-to-end logit MSE remain distinct comparisons.

## D Post-Training Experimental Protocols

This appendix specifies the design and evaluation conventions for the post-training quantization experiments. Throughout, M denotes the number of quantization levels. For a weight row, $q _ { i } = \widehat { w } _ { i } - b _ { i }$ is the instantaneous quantization error, whereas $\widehat { \pmb { w } } - \pmb { w } = G \pmb { q }$ is the shaped weight error. Quantization grids are fixed before causal feedback is applied.

## D.1 Synthetic networks

The synthetic experiment uses a five-linear-layer network with widths $1 2  1 6  1 6  1 2  8  3 .$ The first four layers have tanh activations, and the output layer is linear. For each of 100 independently generated networks and datasets, we use 256 calibration samples and an independent set of 1024 evaluation samples. The weights are drawn from zero-mean Gaussian distributions with standard deviations proportional to the inverse square root of the layer input width. Inputs are generated from a four-dimensional latent Gaussian variable through a mixing matrix fixed within each trial, with additive Gaussian noise.

For output row r of layer ℓ, let $\bar { C } _ { \ell }$ and ${ \cal H } _ { \mathrm { J a c } , \ell , \prime } ^ { \mathrm { r o w } }$ be the sample-averaged input covariance and row-wise

Jacobian metric defined in Section 3. The same scale is applied to all rows of a layer:

$$
s _ { \ell } = \frac { \mathrm { t r } ( \bar { C } _ { \ell } ) } { \mathrm { m a x } \Bigl \{ d _ { \ell } ^ { - 1 } \sum _ { r = 1 } ^ { d _ { \ell } } \mathrm { t r } ( H _ { \mathrm { J a c } , \ell , r } ^ { \mathrm { r o w } } ) , 1 0 ^ { - 3 0 } \Bigr \} } .
$$

For each row, feedback is designed from

$$
\begin{array} { r l } & { \pmb { A } _ { \ell , r } ( \alpha ) = \bar { \pmb { C } } _ { \ell } + \alpha s _ { \ell } \pmb { H } _ { \mathrm { J a c } , \ell , r } ^ { \mathrm { r o w } } , } \\ & { \widetilde { \pmb { A } } _ { \ell , r } ( \alpha ) = \pmb { A } _ { \ell , r } ( \alpha ) + 1 0 ^ { - 5 } \frac { \mathrm { t r } \left( \pmb { A } _ { \ell , r } ( \alpha ) \right) } { d _ { \ell - 1 } } \pmb { I } . } \end{array}
$$

The second line is the trace-scaled ridge used before computing the feedback matrix.

For each weight row and M, the grid consists of M evenly spaced levels between the minimum and maximum entries of the original row. It is fixed across $\alpha \in \{ 0 , 0 . 0 1 , 0 . 0 3 , 0 . 1 , 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 0 . 9 9 \}$ Quantizer inputs outside the $\mathrm { g r i d }$ range map to an endpoint. The same 256 calibration samples are used to estimate the design metrics and to select α by output MSE relative to the full-precision network. The independent evaluation set is used only to report test output MSE. Thus, unlike the trained-network experiments below, this synthetic experiment does not use a disjoint α-selection set.

## D.2 Trained MNIST and CIFAR-10 networks

The MNIST network has widths $7 8 4 \to 1 2 8 \to 1 2 8 \to 6 4 \to 3 2 \to 1 0 ;$ the CIFAR-10 network has widths $3 0 7 2  1 2 8  1 2 8  6 4  3 2  1 0$ . Inputs are flattened, the first four linear layers have tanh activations, and the output layer is linear. The reported results are averaged over 100 independently trained checkpoints. In training, 2048 training samples are reserved for metric calibration and 512 training samples are reserved separately for α selection; the remaining training set are used to fit the checkpoint.

Input coordinates within each weight row are partitioned into consecutive blocks of at most 32. A separate feedback matrix is designed for each block. The RHT and joint methods also use a separate rotation for each block. Each block’s design is shared across the output rows of its layer.

For block $I _ { b } ,$ let $X _ { b }$ contain the corresponding calibration activations and define

$$
\begin{array} { r } { \begin{array} { r } { C _ { b } = \displaystyle \frac { 1 } { N _ { \mathrm { c a l } } } \boldsymbol { X } _ { b } \boldsymbol { X } _ { b } ^ { \top } , } \\ { H _ { \mathrm { J a c } , b } ^ { \mathrm { i s o } } = \displaystyle \frac { 1 } { N _ { \mathrm { c a l } } } \sum _ { t = 1 } ^ { N _ { \mathrm { c a l } } } \frac { \| \boldsymbol { J } _ { \ell , t } \| _ { F } ^ { 2 } } { d _ { \ell } } \boldsymbol { x } _ { \ell , t , b } \boldsymbol { x } _ { \ell , t , b } ^ { \top } . } \end{array} } \end{array}
$$

The efective design metric is constructed separately for each block:

$$
\begin{array} { r l } & { \pmb { A } _ { b } ( \alpha ) = \pmb { C } _ { b } + \alpha \frac { \mathrm { t r } ( \pmb { C } _ { b } ) } { \mathrm { m a x } \{ \mathrm { t r } ( \pmb { H } _ { \mathrm { J a c } , b } ^ { \mathrm { i s o } } ) , 1 0 ^ { - 3 0 } \} } \pmb { H } _ { \mathrm { J a c } , b } ^ { \mathrm { i s o } } , } \\ & { \widetilde { \pmb { H } } _ { b } ( \alpha ) = \pmb { A } _ { b } ( \alpha ) + \lambda \frac { \mathrm { t r } ( \pmb { A } _ { b } ( \alpha ) ) } { | \pmb { I } _ { b } | } \pmb { I } . } \end{array}
$$

We set $\lambda = 1 0 ^ { - 3 }$ . The covariance-based methods use $\alpha = 0 ;$ the Jacobian-aware methods select α among the candidates in {0, 0.03, 0.1, 0.3, 1}. The joint method applies its rotation-feedback construction to the damped block metric $\widetilde { H _ { b } } ( \alpha ) ;$ ; its determinant bound therefore refers to that metric.

Full-precision network Jacobians on the metric-calibration set are used to construct the candidates. Jacobians on the disjoint selection set are used to score them. Specifically, the selection criterion is the full first-order network-output quadratic

$$
D _ { \mathrm { s e l } } ^ { ( 1 ) } = \frac { 1 } { N _ { \mathrm { s e l } } d _ { L } } \sum _ { t = 1 } ^ { N _ { \mathrm { s e l } } } \left\| \sum _ { \ell = 1 } ^ { L } \pmb { J } _ { \ell , t } ( \widehat { \pmb { W } } _ { \ell } - \pmb { W } _ { \ell } ) \pmb { x } _ { \ell , t } \right\| _ { 2 } ^ { 2 } .
$$

This evaluates the resulting candidate weights; it is not the row-isotropic trace objective used to construct them. Final test logit MSE is measured from the fully quantized network. Accuracy drop is calculated relative to each checkpoint’s full-precision accuracy, in percentage points, before averaging over checkpoints.

Within each block, each rotated weight row receives an evenly spaced M-level grid spanning that row’s minimum and maximum before feedback. Inputs outside the grid range map to an endpoint. For an RHT block whose size is not a power of two, the implementation uses a random signed identity instead of a Hadamard transform.

## D.3 GPT-2 projection weights

The reported experiment uses full-precision embedding mode. We quantize four attention and MLP projection matrices in each of the 12 GPT-2 transformer blocks: attn.c attn, attn.c proj, mlp.c fc, and mlp.c proj. Thus, 48 projection matrices are quantized. Token and positional embeddings remain in full precision, as does the output head tied to the token embedding. Other parameters outside these projection matrices are not quantized. The projection weights are processed in input-coordinate blocks of size 32.

Metric calibration, α selection, and test evaluation use non-overlapping length-64 token windows from the WikiText-2 raw train, validation, and test splits, respectively. The script tokenizes nonempty text rows, inserts a separator between rows, and takes consecutive windows from the resulting token stream. Each window contributes 63 scored next-token positions.

For every calibration batch, the script draws one Rademacher vector over the vocabulary at each scored position, with entries scaled by the inverse square root of the vocabulary size. Backpropagating the resulting logit probe through the full-precision model gives a downstream gradient at each selected projection output. The squared gradient is averaged over that projection’s output coordinates to obtain one sensitivity value per scored token. The same probe supplies gradients for the selected projections in that calibration batch.

For each projection and input-coordinate block, the script accumulates a sample-averaged activation covariance $C _ { b }$ and a sensitivity-weighted activation covariance $\pmb { H } _ { b }$ . It then sets

$$
s _ { b } = \frac { \mathrm { t r } ( C _ { b } ) } { \mathrm { m a x } \{ \mathrm { t r } ( H _ { b } ) , 1 0 ^ { - 3 0 } \} } , \quad \quad A _ { b } ( \alpha ) = C _ { b } + \alpha s _ { b } H _ { b } ,
$$

and uses the damped metric

$$
\widetilde { \pmb { H } } _ { b } ( \alpha ) = \pmb { A } _ { b } ( \alpha ) + \lambda \frac { \mathrm { t r } ( \pmb { A } _ { b } ( \alpha ) ) } { | \boldsymbol { I } _ { b } | } \pmb { I } .
$$

We set $\lambda = 1 0 ^ { - 3 }$ . This is not the full downstream-Jacobian quadratic but a stochastic row-isotropic design surrogate. Feedback and, where applicable, rotation are designed independently within each block.

Each rotated weight row receives a fixed, evenly spaced M-level grid between its pre-feedback minimum and maximum. Quantizer inputs outside that range saturate at an endpoint. The Jacobian-aware methods select α by validation logit MSE against the full-precision model after quantizing all selected projection matrices. The covariance-feedback baseline uses fixed-order feedback rather than the full published GPTQ implementation.

Test perplexity is computed from next-token cross-entropy, and test logit MSE is measured relative to the full-precision model over the same scored positions. The reported evaluation uses 128 length-64 test windows, comprising 8064 scored next-token positions, in a single run. These values describe the evaluated windows rather than the entire WikiText-2 test corpus. We use 32 length-64 windows from the training split for metric calibration, 32 from the validation split for α selection, and 128 from the test split for evaluation. The reported run uses seed 0, input-coordinate blocks of size 32, and $M \in \{ 2 , 4 , 8 , 1 6 \}$ quantization levels. For the Jacobian-aware methods, candidate values are $\alpha \in \{ 0 , 0 . 0 3 , 0 . 1 , 0 . 2 , 0 . 8 , 0 . 9 9 \}$

## E Architecture-dependent treatment of downstream curvature.

The six baselines combine two distortion criteria, Local and Logit, with three coordinate-basis and feedback choices, GPTQ, RHT, and Joint. The interpretation of the Logit criterion difers between the LLM and CIFAR-10 CNN experiments.

Let $a _ { \ell , s }$ denote the vectorized preactivation of layer ℓ for calibration example s, and let $f _ { s }$ collect the final logits. Define

$$
J _ { \ell , s } = \frac { \partial f _ { s } } { \partial a _ { \ell , s } ^ { \top } } , \qquad C _ { \ell , s } = J _ { \ell , s } ^ { \top } J _ { \ell , s } .
$$

For a local perturbation $\delta \boldsymbol { a } _ { \ell , s } .$ , the first-order approximation to the squared logit perturbation is

$$
\begin{array} { r } { \| \delta f _ { s } \| _ { 2 } ^ { 2 } \approx \delta a _ { \ell , s } ^ { \top } C _ { \ell , s } \delta a _ { \ell , s } . } \end{array}
$$

In the LLM experiments, the downstream curvature is approximated by a diagonal matrix,

$$
C _ { \ell , s } \approx \mathrm { D i a g } ( \widehat { d } _ { \ell , s } ) .
$$

The diagonal sensitivities are obtained by deterministic operation-wise propagation. For an operation $y = \phi ( x )$ with Jacobian $B = \partial y / \partial x ^ { \top }$ , the propagation rule is

$$
\widehat { d } _ { x } = \mathrm { d i a g } \left( B ^ { \top } \operatorname { D i a g } ( \widehat { d } _ { y } ) B \right) = ( B \odot B ) ^ { \top } \widehat { d } _ { y } .
$$

This procedure discards intermediate of-diagonal curvature and cross-path covariance terms. Consequently, $\hat { d _ { \ell , s } }$ is generally an approximation to dia $\boldsymbol { \mathrm { \Sigma } } _ { 5 } ^ { \top } ( J _ { \ell , s } ^ { \top } J _ { \ell , s } )$ , rather than its exact evaluation.

The final RMSNorm and language-model head are treated analytically using the full vocabulary. This gives an exact initialization of the head sensitivity in the diferentiable real-arithmetic model. However, exact head initialization does not make the preceding diagonal propagation exact.

In contrast, the CIFAR-10 CNN experiments retain the full $J _ { \ell , s } ^ { \top } J _ { \ell , s }$ without diagonalization. Writing

$$
\delta \boldsymbol { a } _ { \ell , s } = \boldsymbol { B } _ { \ell , s } \delta \boldsymbol { \theta } _ { \ell } ,
$$

where $\delta \theta _ { \ell }$ stacks the perturbed parameters of layer $\ell ,$ the exact layerwise weight-space curvature is

$$
\mathcal { H } _ { \ell } ^ { \mathrm { e x a c t } } = \frac { 1 } { N C } \sum _ { s = 1 } ^ { N } B _ { \ell , s } ^ { \top } J _ { \ell , s } ^ { \top } J _ { \ell , s } B _ { \ell , s } , \qquad C = 1 0 .
$$

Equivalently, with the full layer-weight Jacobian

$$
\mathcal { T } _ { \ell , s } = \frac { \partial f _ { s } } { \partial \theta _ { \ell } ^ { \top } } = J _ { \ell , s } B _ { \ell , s } ,
$$

we have

$$
\mathcal { H } _ { \ell } ^ { \mathrm { e x a c t } } = \frac { 1 } { N C } \sum _ { s = 1 } ^ { N } \mathcal { T } _ { \ell , s } ^ { \top } \mathcal { T } _ { \ell , s } .
$$

The exact Jacobian factor is retained, including within-layer cross-filter and spatial-path interactions; no diagonal, Monte Carlo, or rank-truncation approximation is introduced in this curvature construction.

Here, “exact” refers to the Jacobian-based quadratic evaluated at the reference network. It does not imply an exact description of the nonlinear network loss after finite quantization, nor an unrestricted joint optimization over all network parameters.

## F Feedback Optimization in the Toy QAT Experiment

## F.1 Data, model, and quantization grids

The student is a two-layer regression network with dimensions $4  6  3$ and a tanh hidden activation. Each of 100 paired trials generates an independent teacher network, correlated four-dimensional inputs, and targets with independent additive Gaussian noise of standard deviation 0.03. Training, validation, and test sets contain 160, 80, and 256 input-target pairs. Within a trial, all methods share the data, full-precision student checkpoint, and quantization grids. The student is first fitted for 160 full-batch Adam steps at learning rate 0.025.

Both weight matrices are quantized at $M \in \{ 2 , 4 , 8 \}$ levels. Each row uses an evenly spaced grid between the minimum and maximum of that row in the fitted full-precision checkpoint. The grids are held fixed during subsequent training, even when shadow weights change. Nearest-grid rounding saturates at the grid endpoints.

## F.2 Causal quantizer and feedback initialization

A unit lower-triangular $G _ { \ell }$ is shared across the output rows of layer ℓ, with dimensions $4 \times 4$ and $6 \times 6$ for the two layers. For row $r ,$ the hard-quantized forward pass follows the input-coordinate order

$$
\begin{array} { r l } & { b _ { \ell , r , i } = w _ { \ell , r , i } + \displaystyle \sum _ { j < i } g _ { \ell , i j } q _ { \ell , r , j } , } \\ & { \widehat { w } _ { \ell , r , i } = \mathcal { Q } _ { M , \ell , r } \big ( b _ { \ell , r , i } \big ) , } \\ & { q _ { \ell , r , i } = \widehat { w } _ { \ell , r , i } - b _ { \ell , r , i } . } \end{array}
$$

Although feedback coeficients are shared across rows, each row has its own instantaneous-error sequence.

The covariance-feedback initialization is computed from the empirical input second moment at the full-precision checkpoint. For the second layer, the inputs are the full-precision hidden activations. The metric used for this initialization receives the ridge

$$
{ \pmb H } _ { \ell } \longleftarrow { \pmb H } _ { \ell } + 0 . 0 2 \frac { \mathrm { t r } ( { \pmb H } _ { \ell } ) } { n _ { \ell } } { \pmb I } .
$$

This ridge afects covariance-based initialization only. The unregularized QAT results below set the feedback-training penalty to zero; feedback entries are still clipped after updates.

## F.3 Compared methods and surrogate gradients

RTN uses identity feedback without training. Covariance-feedback PTQ uses the covariance-designed feedback with the full-precision network fixed. Feedback-only tuning starts from covariance feedback and updates only G, leaving weights fixed. Weight-only QAT starts from $G _ { \ell } = I$ , holds G fixed, and updates shadow weights. Joint QAT also starts from $G _ { \ell } = I$ and the same full-precision checkpoint, but updates the strictly lower-triangular feedback entries as well as weights. Thus, the two QAT arms in the selected main-text comparison have a common identity-feedback initialization. The full comparison in Table 9 also reports covariance-initialized QAT arms for context.

All tuning arms use 100 full-batch Adam steps. The learning rate is 0.015 for trainable weights and 0.009 for trainable feedback entries. The forward pass uses hard-quantized weights. In the backward pass, the scalar quantizer has the identity straight-through derivative, including at saturated inputs. Previously generated $q _ { \ell , r , j }$ values are treated as constants when diferentiating later quantizer inputs. Writing $S _ { \ell , r , i } = \partial \mathcal { L } / \partial \widehat { w } _ { \ell , r , i }$ for the gradient of the hard-quantized network’s mean-squared error, the implemented surrogate gradients are

$$
\widetilde { \frac { \partial \mathcal { L } } { \partial w _ { \ell , r , i } } } = S _ { \ell , r , i } , \qquad \widetilde { \frac { \partial \mathcal { L } } { \partial g _ { \ell , i j } } } = \sum _ { r } S _ { \ell , r , i } q _ { \ell , r , j } , \quad j < i .
$$

Here the loss is the mean of squared errors over samples and output coordinates. After each feedback update, the strictly lower-triangular entries are clipped to $[ - 2 , 2 ]$ , and the diagonal and upper triangle are reset to those of the identity. There is no G-training regularizer in these results.

Every ten steps, the hard-quantized network is evaluated on the validation set. The checkpoint with lowest validation MSE is retained, and its test MSE is computed on the independent test set. Test targets are not used to select checkpoints.

## F.4 Complete comparison

Feedback-only tuning lowers mean test MSE relative to covariance-feedback PTQ at each tested resolution while keeping the fitted network fixed. Among the identity-initialized QAT arms, joint training has lower mean test MSE than weight-only QAT at each tested resolution. The additional covarianceinitialized arms show that initialization also matters: joint training does not uniformly outperform QAT with covariance feedback held fixed.

Table 9: Mean test MSE over 100 paired synthetic regression trials without a feedback-training regularizer. Lower is better. Boldface indicates the lowest mean among quantized methods in each column. Feedback-only tuning starts from covariance feedback; the identity-initialized QAT arms start from $\pmb { G } = \pmb { I }$
<table><tr><td>Method</td><td> $M = 2$ </td><td> $M = 4$ </td><td> $M = 8$ </td></tr><tr><td>Full precision</td><td>0.01391</td><td>0.01391</td><td>0.01391</td></tr><tr><td>RTN</td><td>0.63739</td><td>0.06576</td><td>0.02395</td></tr><tr><td>Covariance-feedback PTQ</td><td>0.48603</td><td>0.05473</td><td>0.02078</td></tr><tr><td>Feedback-only tuning; network fixed</td><td>0.28594</td><td>0.04055</td><td>0.01899</td></tr><tr><td>Weight-only  $\mathrm { Q A T } ; G = I$  fixed</td><td>0.25458</td><td>0.04118</td><td>0.01914</td></tr><tr><td>Joint  ${ \mathrm { Q A T } } ; G _ { 0 } = I$ </td><td>0.24524</td><td>0.04013</td><td>0.01784</td></tr><tr><td>Weight-only  $\mathrm { Q A T } ;$  covariance G fixed</td><td>0.23674</td><td>0.03800</td><td>0.01763</td></tr><tr><td>Joint  $\mathrm { Q A T } ;$  covariance  $G _ { 0 }$ </td><td>0.23228</td><td>0.03844</td><td>0.01766</td></tr></table>

Table 10: MNIST test accuracy (%, mean ± sample standard deviation over three pretrained seeds). Bold marks the higher mean at each bit width. The two procedures difer in quantization grid and bias treatment.
<table><tr><td>Method</td><td></td><td></td><td>1 bit (M = 2) 2 bits (M = 4) 3 bits (M = 8) 4 bits (M = 16)</td><td></td></tr><tr><td>Local EFQ</td><td> $1 0 . 7 5 0 \pm 1 . 0 4 1$ </td><td> $9 5 . 0 6 3 \pm 0 . 2 8 5$ </td><td> $\mathbf { 9 7 . 1 6 7 \pm 0 . 2 9 9 }$ </td><td> $9 7 . 2 3 7 \pm 0 . 2 6 5$ </td></tr><tr><td>Obj.-EFQ (Logit-MSE), unconstrained</td><td> $\mathbf { 8 5 . 5 6 0 \pm 3 . 1 4 7 }$ </td><td> $\mathbf { 9 5 . 6 0 7 \pm 0 . 9 4 8 }$ </td><td> $9 7 . 1 6 0 \pm 0 . 2 1 0$ </td><td> $\mathbf { 9 7 . 2 5 7 \pm 0 . 3 3 7 }$ </td></tr></table>

## G Additional Experimental Analyses

These experiments complement the main-text comparisons under diferent architectures, grids, constraints, and evaluation protocols. EFQ denotes causal error-feedback quantization. Local EFQ uses a local input-coordinate criterion, whereas Obj.-EFQ uses the downstream criterion named in each experiment. EFQ+RHT applies randomized orthogonal preprocessing before feedback; joint R, G denotes the corresponding joint-design procedure. Metric names in parentheses specify the criterion used; downstream formulations difer across experiments. Accuracy is reported in percent throughout, and reported standard deviations are sample standard deviations.

## G.1 MNIST: Quantization-Procedure Comparison

Experimental setup. We quantize three pretrained ReLU MLPs with architecture $7 8 4  6 4 $ 32 → 10 and seeds {7, 8, 42} at $M \in \{ 2 , 4 , 8 , 1 6 \}$ levels, without retraining. For each seed and bit width, damping is selected on a disjoint 5,000-example training holdout; the selected quantizer is then rebuilt from the original model using the training data and evaluated on the 10,000-example test set. Local EFQ uses asymmetric afine grids and full-precision biases, whereas unconstrained Obj.-EFQ (Logit-MSE) uses symmetric Sigma–Delta grids and joint weight–bias quantization. Table 10 therefore compares complete quantization procedures rather than an isolated change of metric or constraint.

Results and analysis. Obj.-EFQ has higher mean accuracy at one and two bits, while the two procedures are close at three and four bits relative to the reported across-seed variation. The one-bit means difer substantially (85.560% versus 10.750%), but the simultaneous changes in grid construction and bias treatment prevent attributing this diference solely to the downstream criterion or removal of the no-overload constraint.

Table 11: Layerwise PTQ on CIFAR-10. Test accuracy (%) and logit MSE are reported as mean ± sample standard deviation over seeds {42, 33, 8}. All fused weights and biases are quantized. Bold marks the highest mean accuracy at each bit width; the FP32 reference attains 89. $. 7 7 \pm 0 . 1 4 \%$
<table><tr><td></td><td colspan="2">1 bit (M = 2)</td><td colspan="2">2 bits (M = 4)</td><td colspan="2">3 bits (M = 8)</td><td colspan="2">4 bits (M = 16)</td></tr><tr><td>Method</td><td>Acc. ↑</td><td>Logit MSE ↓</td><td>Acc. ↑</td><td>Logit MSE ↓</td><td>Acc. ↑</td><td>Logit MSE ↓</td><td>Acc. ↑</td><td>Logit MSE ↓</td></tr><tr><td>Local EFQ</td><td> $9 . 6 8 \pm 0 . 6 2$ </td><td> $6 5 5 6 \pm 1 7 8 0$ </td><td> $1 7 . 9 6 \pm 3 . 7 2$ </td><td> $3 7 . 7 0 \pm 2 . 9 2$ </td><td> $4 9 . 1 0 \pm 5 . 2 9$ </td><td> $2 7 . 3 8 \pm 4 . 2 2$ </td><td> $7 3 . 7 0 \pm 5 . 3 3$ </td><td> $1 1 . 5 8 \pm 2 . 4 6$ </td></tr><tr><td>Local EFQ+RHT</td><td> $9 . 9 9 \pm 0 . 1 4$ </td><td> $6 6 7 2 \pm 2 8 9 9$ </td><td> $1 7 . 6 3 \pm 2 . 6 3$ </td><td> $3 8 . 8 3 \pm 2 . 0 8$ </td><td> $5 2 . 3 6 \pm 6 . 3 8$ </td><td> $2 6 . 0 2 \pm 5 . 1 2$ </td><td>74.59 ± 5.12</td><td>10.91 ± 3.96</td></tr><tr><td>Local joint R, G</td><td> $9 . 5 3 \pm 0 . 8 9$ </td><td> $7 2 3 8 \pm 6 2 7 1$ </td><td> $1 7 . 9 4 \pm 1 . 4 8$ </td><td> $3 9 . 0 3 \pm 1 . 7 1 $ </td><td> $4 8 . 5 0 \pm 9 . 5 5$ </td><td>28.12 ± 9.82</td><td>75.18 ± 5.28</td><td>10.60 ± 3.73</td></tr><tr><td>Obj.-EFQ (diag)</td><td> $1 0 . 3 1 \pm 0 . 1 4$ </td><td> $2 1 0 3 \pm 1 4 1 6$ </td><td> ${ \bf 2 8 . 1 1 \pm 1 0 . 9 4 }$ </td><td> $4 0 . 5 2 \pm 1 0 . 5 2$ </td><td> $5 4 . 5 1 \pm 1 0 . 9 5$ </td><td> $2 3 . 5 1 \pm 5 . 2 0$ </td><td> $7 0 . 9 6 \pm 1 2 . 5 3$ </td><td> $1 3 . 5 4 \pm 8 . 2 5$ </td></tr><tr><td>Obj.-EFQ+RHT (diag)</td><td> ${ \bf 1 1 . 7 9 \pm 1 . 0 4 }$ </td><td> $1 9 7 2 \pm 6 6 0$ </td><td> $1 8 . 7 1 \pm 4 . 4 8$ </td><td> $4 2 . 1 8 \pm 3 . 8 8$ </td><td>45.86 ± 4.16</td><td>27.92 ± 2.58</td><td> $7 6 . 1 6 \pm 5 . 2 0$ </td><td> $9 . 8 6 \pm 3 . 6 4$ </td></tr><tr><td>Obj. joint R, G (diag)</td><td>10.53 ± 0.99</td><td> $3 5 0 8 \pm 1 4 8 1$ </td><td> $1 7 . 5 5 \pm 2 . 1 9$ </td><td>39.44 ± 5.07</td><td>46.16 ± 17.76</td><td>29.17 ± 10.04</td><td>74.83 ± 4.86</td><td> $9 . 7 5 \pm 2 . 8 0$ </td></tr><tr><td>Obj.-EFQ (exact)</td><td> $1 0 . 9 4 \pm 2 . 2 9$ </td><td> $2 1 4 9 \pm 2 0 7 7$ </td><td> $1 5 . 1 7 \pm 5 . 3 2$ </td><td>44.82 ± 9.77</td><td>54.55 ± 11.28</td><td>19.57 ± 3.92</td><td>82.40 ± 0.55</td><td>5.84 ± 0.43</td></tr><tr><td>Obj.-EFQ+RHT (exact)</td><td>8.90 ± 0.31</td><td> $3 1 6 3 \pm 9 7 5$ </td><td> $2 3 . 1 2 \pm 6 . 1 6$ </td><td>37.95 ± 4.35</td><td>55.73 ± 10.52</td><td>21.67 ± 1.90</td><td>81.90 ± 0.17</td><td> $6 . 0 9 \pm 0 . 6 2$ </td></tr><tr><td>Obj. joint R, G (exact)</td><td>10.23 ± 1.49</td><td>2558 ± 2271</td><td> $1 5 . 9 2 \pm 4 . 1 8$ </td><td>40.53 ± 0.47</td><td>55.28 ± 4.28</td><td>19.56 ± 2.06</td><td>81.00 ± 1.44</td><td>6.31 ± 0.42</td></tr></table>

## G.2 CIFAR-10: Layerwise Curvature and Rotation

Experimental setup. We evaluate layerwise PTQ on CIFAR-10 using its 50,000 training and 10,000 test images. For each seed in {42, 33, 8}, a stratified 5,000-image validation set is held out, and 512 calibration images are sampled without replacement from the remaining 45,000 training images. Inputs use per-channel normalization; teacher training additionally uses random crops with four-pixel padding and horizontal flips. The CNN has seven convolutions, each followed by batch normalization and ReLU, with 2 × 2 max pooling after convolutions 3, 5, and 7, global average pooling, and a 256 → 10 classifier. Folding batch normalization gives 292,826 afine parameters. Teachers are trained for 200 epochs with SGD, Nesterov momentum 0.9, weight decay $5 \times 1 0 ^ { - 4 }$ , batch size 256, learning rate 0.1, five-epoch linear warm-up, cosine decay, and BF16 mixed precision; checkpoint selection uses validation accuracy, then NLL. The FP32 teachers attain $8 9 . 7 7 \pm 0 . 1 4 \%$ test accuracy. All fused weights and biases are quantized without fine-tuning using per-output-row symmetric grids at $M \in \{ 2 , 4 , 8 , 1 6 \}$ ; each clipping radius is selected from $\{ 1 . 0 0 , 0 . 9 5 , \ldots , 0 . 5 0 \}$ times the row maximum by minimizing rounding MSE. We cross Local afine-output reconstruction, Obj.-diag exact downstream derivatives with cross-output curvature discarded, and Obj.-exact complete within-layer curvature $H = J ^ { \top } J / ( N C )$ with EFQ, EFQ+RHT, and joint R, G. Here N is the calibration size and C = 10; Obj.-exact retains cross-filter and spatial couplings but excludes cross-layer curvature. Joint design equalizes a loaded column-contracted metric before feedback. The GPTQ-style recurrence uses 128-coordinate computational blocks and rebuilds dual factors every 128 coordinates; two numerical retries with tenfold loading increments are allowed, but none is used. All 108 configurations are evaluated on the full training and test sets using dense FP32 reconstruction, recording accuracy, NLL, teacher-to-student KL, and per-class logit MSE. Table 11 reports three-seed test means and standard deviations.

Results and analysis. The clearest benefit of complete within-layer curvature appears at four bits: Obj.-EFQ (exact) attains 82.40% accuracy and logit MSE 5.84, exceeding Local EFQ by 8.70 percentage points and Obj.-EFQ (diag) by 11.44 points, although a 7.37-point gap to the teacher remains. At three bits, all exact variants attain approximately 55% accuracy; the RHT variant has the highest accuracy (55.73%), whereas joint R, G has the lowest logit MSE (19.56). Thus, output fidelity and classification accuracy need not rank methods identically. Rotation gains depend on the metric and precision: RHT helps Local EFQ at three and four bits but hurts Obj.-EFQ (diag) at two and three bits, and unrotated exact EFQ outperforms both rotated exact variants at four bits. Joint design is not uniformly best, consistent with its optimization of a column-contracted metric rather than the complete coupled objective. At one and two bits, even the best accuracies are only 11.79% and 28.11%; the mixed rankings are consistent with limited predictive power of calibration-point quadratic models under large perturbations, rather than a general advantage of local curvature.

Table 12: MODNet test MSE (mean ± sample standard deviation over three seeds; lower is better). Bold marks the lower mean among the two quantized procedures at each value of M.
<table><tr><td>Method</td><td>M = 2</td><td>M = 4</td><td>M = 8</td><td>M = 16</td></tr><tr><td>Local EFQ (GPTQ-style) 0.287104 ± 0.240392</td><td></td><td>0.015371 ± 0.002331</td><td>0.014062 ± 0.000881</td><td>0.013476 ± 0.001157</td></tr><tr><td>Obj.-EFQ (multilayer)</td><td> $0 . 2 9 1 3 9 0 \pm 0 . 2 2 6 7 1 8$ </td><td>0.016781 ± 0.004631</td><td>0.013747 ± 0.001241</td><td> $0 . 0 1 3 5 3 5 \pm 0 . 0 0 1 1 7 2$ </td></tr><tr><td>Full precision</td><td colspan="4">0.013615 ± 0.001164</td></tr></table>

## G.3 MODNet: Regression Comparison

Experimental setup. We evaluate three ReLU MLPs with architecture 1000 → 128 → 64 → 32 → 1 on the MODNet refractive-index task, using a fixed train/model-validation/quantization-validation/test split of 2614/402/402/604 samples. Afine weights and biases are quantized without fine-tuning at the values of M shown in Table 12. Damping is selected separately for each seed, method, and resolution using quantization-validation MSE; the test set is reserved for final evaluation. The table compares Local EFQ (GPTQ-style), Obj.-EFQ (multilayer), and full precision. The multilayer criterion here is not assumed to be identical to the row-isotropic Jacobian metric in the main-text MLP experiment.

Results and analysis. Local EFQ has lower mean test MSE at M = 2, 4, 16, whereas Obj.-EFQ has lower mean MSE at M = 8. At M = 16, both quantized means are close to the full-precision mean relative to the three-seed variation; their slightly smaller means do not establish an improvement over full precision. These conclusions concern only the displayed procedures, not the QEP variants from the broader experimental protocol.

## G.4 Qwen3-8B: Projection-Level Diagnostic

Experimental setup. This sequential-replay experiment compares Local EFQ, Local EFQ+RHT, and Local joint R, G on all 252 Qwen3-8B attention and MLP projections using whole-row asymmetric afine grids at one to four bits. Embeddings, normalization layers, and the language-model head remain in BF16. Calibration uses 128 WikiText-2 sequences of length 2048, with 32 additional validation sequences; evaluation uses 146 complete test sequences. For each projection, the original and quantized weights are applied to the same live replay input X, giving

$$
\mathrm { M S E _ { \mathrm { o u t } } } = \frac { \| \mathbf { X } \mathbf { W } ^ { \top } - \mathbf { X } \widehat { \mathbf { W } } ^ { \top } \| _ { F } ^ { 2 } } { N d _ { \mathrm { o u t } } } ,
$$

where N is the number of probe inputs. Replay inputs reflect preceding quantization decisions and can therefore difer across methods. Table 13 reports projection-level MSE, not final-logit MSE. Its sequence length, grid, and evaluation protocol difer from the main-text Qwen3-8B experiment, so the two BF16 perplexities are not directly comparable.

Results and analysis. Local EFQ+RHT has lower mean projection-output MSE than joint R, G at three and four bits and wins more projections. At two bits, joint R, G has lower mean MSE and more wins, yet its PPL is higher (2584.97 versus 833.66). At four bits, joint R, G instead has slightly lower PPL (10.0004 versus 10.0141) despite higher projection-output MSE. These diagnostics need not rank quantizers alike, especially when replay inputs are method-dependent. The one-bit results also require caution because the asymmetric two-level grid is not a symmetric sign quantizer.

## G.5 Qwen3-8B: Additional Metrics and Backends

Experimental setup. We extend the main-text Qwen3-8B experiment with Guided-shared and Model-Fisher-diag metrics and coordinate descent (CD), giving the 15 procedures in Table 14. Calibration uses 128 WikiText-2 sequences of length 512, and evaluation covers 298,424 scored test positions. Methods use the experiment’s asymmetric zero-inclusive finite grid and calibration-derived damping;

Table 13: Qwen3-8B sequential-replay diagnostic on WikiText-2 with length-2048 sequences. MSE statistics summarize direct projection-output error across 252 projections, not final-logit error. “Wins” counts the projections with the lowest MSE at each bit width. Bold marks the best quantized value in each column at that bit width.
<table><tr><td>Bits</td><td>Method</td><td></td><td>PPL ↓ Mean MSE↓</td><td>Median MSE↓</td><td>P95 MSE↓</td></tr><tr><td>1</td><td>Local EFQ</td><td>1.0993×108</td><td>39.3955</td><td>0.16201</td><td>42.4530</td></tr><tr><td>1</td><td>Local EFQ+RHT</td><td>3.8108×105</td><td>5.4685</td><td>0.13079</td><td>9.9605</td></tr><tr><td>1</td><td>Local joint R, G</td><td>1.0715×108</td><td>2.7440</td><td>0.39303</td><td>15.2230</td></tr><tr><td>2</td><td>Local EFQ</td><td>26488.66</td><td>1.9854</td><td>0.16360</td><td>11.3747</td></tr><tr><td>2</td><td>Local EFQ+RHT</td><td>833.66</td><td>0.85167</td><td>0.10521</td><td>4.8990</td></tr><tr><td>2</td><td>Local joint R, G</td><td>2584.97</td><td>0.80394</td><td>0.10230</td><td>4.8091</td></tr><tr><td>3</td><td>Local EFQ</td><td>14.1148</td><td>0.14179</td><td>0.01630</td><td>0.83881</td></tr><tr><td>3</td><td>Local EFQ+RHT</td><td>11.6523</td><td>0.05658</td><td>0.01005</td><td>0.33736</td></tr><tr><td>3</td><td>Local joint R, G</td><td>11.9912</td><td>0.06643</td><td>0.01185</td><td>0.42556</td></tr><tr><td>4</td><td>Local EFQ</td><td>10.3875</td><td>0.02881</td><td>0.003204</td><td>0.17395</td></tr><tr><td>4</td><td>Local EFQ+RHT</td><td>10.0141</td><td>0.01185</td><td>0.002070</td><td>0.06981</td></tr><tr><td>4</td><td>Local joint R, G</td><td>10.0004</td><td>0.01372</td><td>0.002380</td><td>0.08593</td></tr><tr><td>BF16</td><td></td><td>9.7245</td><td>0</td><td>0</td><td>0</td></tr></table>

results are reported at three and four bits. Local denotes the projection-input criterion, and Logit-diag is the objective-aware criterion used in the main-text LLM comparison. The corresponding Local and Logit-diag feedback results match the main text up to rounding. The BF16 reference has PPL 12.8705 and next-token accuracy 51.0395%.

Results and analysis. Obj.-EFQ+RHT (Logit-diag) has the lowest PPL and highest accuracy at both precisions; the other downstream-metric RHT variants are close at four bits. Within each metric family, joint R, G reduces PPL relative to unrotated EFQ but remains behind EFQ+RHT. These results compare implemented procedures and do not identify a unique mechanism for the ordering.

## G.6 Llama-2-7B: Additional Metrics and Backends

Experimental setup. We extend the main-text Llama-2-7B experiment with Guided-shared, Model-Fisher-diag, and CD variants, retaining the Local and Logit-diag results at their reported precision. Calibration uses 128 WikiText-2 sequences of length 512, and evaluation covers 340,326 scored next-token positions. Table 15 reports PPL, next-token accuracy, and final-logit MSE at three and four bits. The BF16 reference attains PPL 7.1782 and accuracy 57.046%.

Results and analysis. Obj.-EFQ+RHT (Logit-diag) is best on all three metrics at three bits. At four bits, Model-Fisher-diag with RHT gives the lowest PPL, Guided-shared with RHT the highest accuracy, and Logit-diag with RHT the lowest logit MSE; the diferences among these variants are small. Within each metric family, joint R, G improves PPL over unrotated EFQ but remains behind the corresponding RHT variant.

## G.7 Llama-2-13B: Cross-Dataset Evaluation on WikiText-2 and C4

Experimental setup. We evaluate weight-only post-training quantization of meta-llama/Llama-2-13b-hf on WikiText-2 and C4-English. The model contains 13.0B parameters, 40 decoder layers, hidden dimension 5120, intermediate dimension 13824, 40 attention heads, and a 32,000-token vocabulary. We quantize the query, key, value, output, gate, up, and down projections in every decoder layer, for 280

Table 14: Additional Qwen3-8B comparisons on WikiText-2. Each cell reports PPL / nexttoken accuracy (%); lower PPL and higher accuracy are better. Bold marks the best quantized value for each metric and bit width. Logit-diag is the objective-aware metric used in the maintext LLM comparison.
<table><tr><td>Method</td><td>3 bits (M = 8)</td><td>4 bits (M = 16)</td></tr><tr><td>Local EFQ</td><td>20.2267 43.5200</td><td>14.1402 49.5325</td></tr><tr><td>Local EFQ+RHT</td><td>16.4851 46.9051</td><td>13.4689 50.2322</td></tr><tr><td>Local joint R, G</td><td>16.8575 46.4504</td><td>13.4730 50.1226</td></tr><tr><td> $\mathrm { L o g i t - d i a g + C D }$ </td><td>27.1912 40.0718</td><td>14.1897 / 49.3405</td></tr><tr><td>Obj.-EFQ (Logit-diag)</td><td>20.5315 43.7609</td><td>13.7497 / 49.7075</td></tr><tr><td>Obj.-EFQ+RHT (Logit-diag)</td><td>16.1175 47.0803</td><td>13.3120 50.3200</td></tr><tr><td>Obj. joint R, G (Logit-diag)</td><td>16.8388 46.4963</td><td>13.4485 50.1796</td></tr><tr><td>Guided-shared + CD</td><td>30.9972 38.6722</td><td>14.3729 49.1643</td></tr><tr><td>Obj.-EFQ (Guided-shared)</td><td>22.2475 42.9865</td><td>13.7521 49.6947</td></tr><tr><td>Obj.-EFQ+RHT (Guided-shared)</td><td>16.2954 46.9124</td><td>13.3867 50.2862</td></tr><tr><td>Obj. joint R, G (Guided-shared)</td><td>17.1357 46.2406</td><td>13.4259 50.2269</td></tr><tr><td>Model-Fisher-diag + CD</td><td>27.8430 39.5451</td><td>14.2532 49.3405</td></tr><tr><td>Obj.-EFQ (Model-Fisher-diag)</td><td>20.6045 43.4550</td><td>13.7748 49.6471</td></tr><tr><td>Obj.-EFQ+RHT (Model-Fisher-diag)</td><td>16.1561 47.0629</td><td>13.3898 50.3193</td></tr><tr><td>Obj. joint R, G (Model-Fisher-diag)</td><td>16.9230 46.2831</td><td>13.4803 50.1347</td></tr><tr><td>BF16</td><td colspan="2">12.8705 51.0395</td></tr></table>

quantized linear projections in total, while retaining embeddings, normalization parameters, biases, and the language-model head in BF16. For both datasets, calibration uses 128 non-overlapping sequences of length 512; evaluation uses 666 non-overlapping length-512 windows, yielding 340,326 scored nexttoken predictions after excluding the first target of each window. WikiText-2 results use calibration seeds {7, 8, 42} and are reported as mean ± sample standard deviation, whereas the C4 archive contains one complete cohort, seed 7; incomplete C4 seed-8 and seed-42 runs are excluded. We compare six methods: Local $\mathrm { E F Q } ,$ , Local $\mathrm { E F Q + R H T }$ , Local joint $R , G , \mathrm { O b j . - E F Q } , \mathrm { O b j . - E F Q + R H T } .$ and Obj. joint R, G. The Local methods use the uncentered projection-input Gram matrix, whereas the Obj. methods use a downstream-aware diagonal logit metric. EFQ uses the reference $\mathrm { G P T Q }$ error-feedback recurrence with computational block size 128 and activation ordering disabled; RHT applies a full-width randomized Hadamard transform before EFQ; and joint $R , G$ jointly designs the rotation and causal feedback geometry for the regularized trace surrogate. We use static full-row asymmetric afine grids with $M \in \{ 4 , 8 , 1 6 \}$ levels, corresponding to $2 \mathrm { \cdot , 3 \mathrm { - } }$ , and 4-bit weights. For a projection metric H, damping is fixed from calibration statistics as

$$
A = ( 1 - \eta ) H + \left( \eta + \kappa _ { M } + 1 0 ^ { - 4 } \right) \bar { h } I , \qquad \bar { h } = \frac { \mathrm { t r } ( H ) } { d } ,
$$

where $\eta$ measures calibration-batch variability and $\kappa _ { M }$ is the quantizer-reliability term computed from the frozen reference weights. No grid or damping parameter is selected using evaluation performance. Model execution uses BF16, curvature statistics use FP32, and TF32 is disabled. We report perplexity (PPL), next-token accuracy, and full-vocabulary logit MSE relative to the BF16 teacher.

Results and analysis. The two datasets give a consistent ordering. Obj.-EFQ+RHT achieves the lowest PPL and logit MSE and the highest next-token accuracy at every reported bit width on both WikiText-2 and C4. The downstream-aware metric generally improves its Local counterpart when the backend is held fixed. At 3 bits, for example, replacing Local EFQ with Obj.-EFQ reduces WikiText-2 PPL from 8.178 to 7.939 and C4 PPL from 9.6109 to 9.3599; under RHT, the corresponding changes are 7.602 to 7.516 and 8.9591 to 8.8050. The gain is most consistent in logit MSE, which is directly aligned with the downstream-aware surrogate. Small exceptions remain at high precision: on C4 at 4 bits, Obj.-EFQ is marginally worse than Local EFQ on all three metrics, illustrating that a lower quadratic surrogate does not guarantee an identical ordering after finite-grid reconstruction. The benefit of objective awareness is especially pronounced at 2 bits when combined with RHT, although all 2-bit models remain far from their BF16 references.

Table 15: Additional Llama-2-7B comparisons on WikiText-2. Accuracy is in percent, and MSE is final-logit MSE relative to BF16. Bold marks the best quantized value for each metric and bit width.
<table><tr><td></td><td colspan="3">3 bits (M = 8)</td><td colspan="3">4 bits (M = 16)</td></tr><tr><td>Method</td><td>PPL↓</td><td></td><td>Acc. (%) ↑ Logit MSE ↓</td><td>PPL↓</td><td></td><td>Acc. (%) ↑ Logit MSE ↓</td></tr><tr><td>Local EFQ</td><td>11.3119</td><td>50.218</td><td>3.9890</td><td>7.8084</td><td>55.690</td><td>0.7909</td></tr><tr><td>Local EFQ+RHT</td><td>9.1196</td><td>53.145</td><td>2.0765</td><td>7.4502</td><td>56.337</td><td>0.3472</td></tr><tr><td>Local joint R, G</td><td>9.8926</td><td>51.701</td><td>2.7943</td><td>7.5634</td><td>56.059</td><td>0.4977</td></tr><tr><td>Logit-diag + CD</td><td>11.9827</td><td>49.148</td><td>4.7271</td><td>7.7136</td><td>55.861</td><td>0.7632</td></tr><tr><td>Obj.-EFQ (Logit-diag)</td><td>11.0021</td><td>50.807</td><td>3.7452</td><td>7.7356</td><td>55.841</td><td>0.6853</td></tr><tr><td>Obj.-EFQ+RHT (Logit-diag)</td><td>8.7035</td><td>53.731</td><td>1.7391</td><td>7.4306</td><td>56.444</td><td>0.3093</td></tr><tr><td>Obj. joint R, G (Logit-diag)</td><td>9.3382</td><td>52.553</td><td>2.2533</td><td>7.5213</td><td>56.177</td><td>0.4530</td></tr><tr><td>Guided-shared + CD</td><td>12.9219</td><td>48.092</td><td>5.5066</td><td>7.7680</td><td>55.712</td><td>0.8311</td></tr><tr><td>Obj.-EFQ (Guided-shared)</td><td>10.6300</td><td>51.043</td><td>3.5071</td><td>7.7163</td><td>55.899</td><td>0.6943</td></tr><tr><td>Obj.-EFQ+RHT (Guided-shared)</td><td>8.8047</td><td>53.540</td><td>1.8052</td><td>7.4340</td><td>56.469</td><td>0.3239</td></tr><tr><td>Obj. joint R, G (Guided-shared)</td><td>9.3419</td><td>52.533</td><td>2.4230</td><td>7.5340</td><td>56.206</td><td>0.4736</td></tr><tr><td>Model-Fisher-diag + CD</td><td>12.2602</td><td>48.832</td><td>4.8817</td><td>7.7682</td><td>55.770</td><td>0.7795</td></tr><tr><td>Obj.-EFQ (Model-Fisher-diag)</td><td>10.6735</td><td>50.915</td><td>3.4897</td><td>7.7080</td><td>55.904</td><td>0.6677</td></tr><tr><td>Obj.-EFQ+RHT (Model-Fisher-diag)</td><td>8.7383</td><td>53.669</td><td>1.7589</td><td>7.4276</td><td>56.454</td><td>0.3156</td></tr><tr><td>Obj. joint R, G (Model-Fisher-diag)</td><td>9.3606</td><td>52.472</td><td>2.4132</td><td>7.5223</td><td>56.182</td><td>0.4712</td></tr></table>

RHT also consistently outperforms joint R, G in the realized end-to-end metrics, despite joint R, G having the stronger theoretical guarantee for the fixed trace surrogate. This is not a contradiction. The joint construction minimizes $\operatorname { t r } ( G ^ { \top } R ^ { \top } A R G )$ for a fixed regularized metric under the assumed quantization-noise model, whereas the actual finite quantizer refits its afine grid after rotation. The rotation therefore changes the weight range, scale, clipping pattern, and error covariance, none of which is fixed by the trace optimum. These finite-grid efects can outweigh the reduction in the quadratic propagation term. The same RHT-over-Joint ordering on WikiText-2 and C4 suggests that the efect is not specific to one corpus. The three-seed WikiText-2 results also show that variability is concentrated at 2 bits, while the 3- and 4-bit rankings are comparatively stable. C4 should be interpreted as a single-seed cross-dataset replication until the remaining calibration seeds are completed. “‘

## G.8 Qwen3-14B: Additional Metrics and Backends

Experimental setup. We extend the main-text Qwen3-14B comparison with Guided-shared, Model-Fisher-diag, and CD variants. Calibration uses 128 WikiText-2 sequences of length 512 with seed 7; evaluation covers 298,424 scored next-token positions. Table 17 reports PPL, next-token accuracy, and final-logit MSE relative to BF16 at three and four bits. The Local and Logit-diag rows correspond to the main-text comparison.

Results and analysis. Obj.-EFQ+RHT (Logit-diag) has the lowest PPL and logit MSE and the highest accuracy at both precisions. Model-Fisher-diag with RHT is close in PPL at four bits. For every metric family, joint R, G improves PPL over unrotated EFQ but remains behind EFQ+RHT. This reproduces the main-text ordering for this model without isolating its cause; the two conditionally recovered joint runs retain the numerical qualification noted below the table.

Table 16: Llama-2-13B quantization results on WikiText-2 and C4. WikiText-2 values are mean ± sample standard deviation over calibration seeds {7, 8, 42}; C4 values are from the complete seed-7 cohort. Accuracy is reported in percent. Lower PPL and logit MSE are better; higher accuracy is better. Boldface marks the best quantized result at each bit width within each dataset. The BF16 references are 6.3937 PPL / 58.819% accuracy on WikiText-2 and 7.2441 PPL / 56.444% accuracy on C4.
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="3">WikiText-2</td><td colspan="3">C4</td></tr><tr><td>PPL↓</td><td>Acc. (%) ↑</td><td>Logit MSE ↓</td><td></td><td>PPL ↓ Acc. (%) ↑ Logit MSE ↓</td><td></td></tr><tr><td>Local EFQ</td><td>2</td><td> $1 4 8 7 4 . 2 \pm 1 7 3 6 . 0$ </td><td> $0 . 4 2 6 \pm 0 . 0 5 0$ </td><td> $8 . 3 8 7 \pm 0 . 3 2 3$ </td><td>12077.99</td><td>0.420</td><td>8.550</td></tr><tr><td>Local EFQ+RHT 2</td><td></td><td> $2 9 7 5 . 5 9 \pm 6 0 1 . 5 0$ </td><td> $4 . 8 3 3 \pm 1 . 2 9 1$ </td><td> $5 . 9 2 6 \pm 0 . 1 4 1$ </td><td>3212.88</td><td>4.467</td><td>6.938</td></tr><tr><td>Local joint  $R , G$ </td><td>2</td><td> $3 0 8 8 1 . 4 \pm 1 4 2 6 7 . 5$ </td><td> $0 . 0 9 0 \pm 0 . 0 4 3$ </td><td> $1 0 . 4 9 5 \pm 1 . 4 9 8$ </td><td>18447.59</td><td>0.442</td><td>10.308</td></tr><tr><td>Obj.-EFQ</td><td>2</td><td> $1 3 3 7 1 . 2 \pm 7 2 8 . 1$ </td><td> $0 . 4 0 8 \pm 0 . 0 2 8$ </td><td> $8 . 0 8 3 \pm 0 . 3 5 6$ </td><td>7538.99</td><td>1.294</td><td>7.388</td></tr><tr><td> $\mathrm { O b j . - E F Q + R H T }$ </td><td>2</td><td> $\mathbf { 3 4 3 . 5 8 \pm 3 2 . 2 2 }$ </td><td> $\mathbf { 1 5 . 7 0 9 \pm 0 . 5 5 4 }$ </td><td> $\mathbf { 4 . 4 7 8 \pm 0 . 0 3 5 }$ </td><td>486.20</td><td>12.647</td><td>4.854</td></tr><tr><td>Obj. joint  $R , G$ </td><td>2</td><td> $8 9 8 5 . 5 7 \pm 3 4 5 0 . 2 4$ </td><td> $1 . 4 3 2 \pm 0 . 5 5 6$ </td><td> $7 . 7 1 2 \pm 0 . 7 7 4$ </td><td>4724.31</td><td>2.783</td><td>6.969</td></tr><tr><td>Local EFQ</td><td>3</td><td> $8 . 1 7 8 \pm 0 . 0 2 7$ </td><td> $5 4 . 6 1 9 \pm 0 . 0 6 4$ </td><td> $0 . 8 3 7 2 \pm 0 . 0 0 3 8$ </td><td>9.6109</td><td>51.586</td><td>0.9608</td></tr><tr><td>Local EFQ+RHT 3</td><td></td><td> $7 . 6 0 2 \pm 0 . 0 1 5$ </td><td> $5 5 . 7 3 7 \pm 0 . 0 0 7$ </td><td> $0 . 6 1 9 5 \pm 0 . 0 0 7 6$ </td><td>8.9591</td><td>52.516</td><td>0.7339</td></tr><tr><td>Local joint R, G</td><td>3</td><td> $8 . 1 7 3 \pm 0 . 0 6 1$ </td><td> $5 4 . 5 5 8 \pm 0 . 0 6 8$ </td><td> $0 . 8 0 6 1 \pm 0 . 0 0 4 0$ </td><td>10.5715</td><td>49.903</td><td>1.0696</td></tr><tr><td>Obj.-EFQ</td><td>3</td><td> $7 . 9 3 9 \pm 0 . 0 0 9$ </td><td> $5 5 . 1 0 7 \pm 0 . 0 2 8$ </td><td> $0 . 7 6 6 4 \pm 0 . 0 0 1 0$ </td><td>9.3599</td><td>51.961</td><td>0.9045</td></tr><tr><td> $\mathrm { O b j . - E F Q + R H T }$ </td><td>3</td><td> $\mathbf { 7 . 5 1 6 \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 5 5 . 9 1 9 \pm 0 . 0 7 2 }$ </td><td> $\mathbf { 0 . 5 7 8 3 \pm 0 . 0 0 0 2 }$ </td><td>8.8050</td><td>52.766</td><td>0.6820</td></tr><tr><td>Obj. joint R, G</td><td>3</td><td> $8 . 0 0 3 \pm 0 . 0 4 0$ </td><td> $5 4 . 7 4 9 \pm 0 . 1 4 0$ </td><td> $0 . 7 2 8 9 \pm 0 . 0 0 4 1$ </td><td>10.2596</td><td>50.346</td><td>0.9038</td></tr><tr><td>Local EFQ</td><td>4</td><td> $6 . 6 7 8 \pm 0 . 0 3 3$ </td><td> $5 8 . 0 2 9 \pm 0 . 0 6 2$ </td><td> $0 . 1 8 5 5 \pm 0 . 0 1 3 7$ </td><td>7.6039</td><td>55.613</td><td>0.2117</td></tr><tr><td>Local EFQ+RHT 4</td><td></td><td> $6 . 6 0 3 \pm 0 . 0 0 7$ </td><td> $5 8 . 2 3 2 \pm 0 . 0 2 6$ </td><td> $0 . 1 3 6 0 \pm 0 . 0 0 1 2$ </td><td>7.5047</td><td>55.705</td><td>0.1615</td></tr><tr><td>Local joint  $R , G$ </td><td>4</td><td> $6 . 6 7 5 \pm 0 . 0 1 7$ </td><td> $5 7 . 9 9 2 \pm 0 . 0 3 2$ </td><td> $0 . 1 8 1 1 \pm 0 . 0 0 1 7$ </td><td>7.7087†</td><td>55.285</td><td>0.2471</td></tr><tr><td>Obj.-EFQ</td><td>4</td><td> $6 . 6 3 3 \pm 0 . 0 0 2$ </td><td> $5 8 . 1 3 2 \pm 0 . 0 1 3$ </td><td> $0 . 1 6 7 0 \pm 0 . 0 0 0 7$ </td><td>7.6132</td><td>55.560</td><td>0.2126</td></tr><tr><td>Obj.-EFQ+RHT</td><td>4</td><td> $\mathbf { 6 . 5 8 8 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 5 8 . 2 7 4 \pm 0 . 0 4 1 }$ </td><td> $\mathbf { 0 . 1 2 5 5 \pm 0 . 0 0 0 3 }$ </td><td>7.4854</td><td>55.787</td><td>0.1489</td></tr><tr><td>Obj. joint R, G</td><td>4</td><td> $6 . 6 6 7 \pm 0 . 0 0 8$ </td><td> $5 8 . 0 4 4 \pm 0 . 0 6 2$ </td><td> $0 . 1 6 7 3 \pm 0 . 0 0 1 1$ </td><td>7.6378</td><td>55.346</td><td>0.2084</td></tr></table>

<sup>†</sup>The C4 4-bit Local joint R, G run used the pre-specified relaxed-certificate recovery for two projections after the strict equal-pivot tolerance was not met. All reported end-to-end metrics remained finite. C4 seed 8 and seed 42 do not form complete six-method cohorts in the available artifacts and are therefore not included in aggregate statistics.

## G.9 C4: Independently Calibrated Language-Model Experiments

Experimental setup. We evaluate weight-only PTQ on C4-English using Llama-2-7B, Llama-2- 13B, and Qwen3-14B, with calibration independent of the WikiText-2 experiments. Their decoder-block counts are 32/40/40, hidden widths 4096/5120/5120, and feed-forward widths 11008/13824/17408. All 224/280/280 attention and MLP projections are quantized; embeddings, normalization parameters, and the output head retain reference precision. Each model uses 128 length-512 training sequences with calibration seed 7 and a fixed validation snapshot shared by all methods: 666 windows and 340,326 scored next-token predictions for each Llama model, and 584 windows and 298,424 predictions for Qwen3-14B, excluding the first target in each window. Llama-2-7B evaluation sampling uses seed 1701. The six methods cross Local uncentered input-Gram or $\mathrm { O b j }$ . diagonal downstream-logit metrics with EFQ, full-width $\mathrm { E F Q + R H T }$ , or full-dimensional joint $R , G . \ \mathrm { E F Q }$ uses the reference $\mathrm { G P T Q }$ recurrence with 128-column computational blocks and activation ordering disabled. Static asymmetric full-row afine grids use $M \in \{ 8 , 1 6 \}$ . For a $d \times d$ metric $H .$ , loading is $A = ( 1 - \eta ) H + ( \eta + \kappa _ { M } + 1 0 ^ { - 4 } ) \bar { h } I$ , with $\bar { h } = \mathrm { t r } ( H ) / d ,$ , calibration-batch estimate $\eta ,$ and $\kappa _ { M }$ frozen from reference weights; no internal extra damping or evaluation-based grid/damping selection is used. Forward evaluation uses BF16, metric statistics use FP32, and TF32 is disabled. Tables 18–20 report PPL, next-token accuracy, and fullvocabulary logit MSE relative to BF16. Only complete seed-7 cohorts are included; rejected or incomplete Llama-2-13B seed-8 and seed-42 cohorts are excluded, with no across-seed uncertainty claimed.

Table 17: Additional Qwen3-14B comparisons on WikiText-2. Accuracy is in percent, and MSE is final-logit MSE relative to BF16. Bold marks the best quantized value for each metric and bit width.
<table><tr><td rowspan="2">Method</td><td colspan="3">3 bits (M = 8)</td><td colspan="3">4 bits (M = 16)</td></tr><tr><td>PPL↓</td><td>Acc. (%) ↑ Logit MSE ↓</td><td></td><td>PPL↓</td><td></td><td>Acc. (%) ↑ Logit MSE ↓</td></tr><tr><td>Local EFQ</td><td>16.4868</td><td>46.469</td><td>2.7744</td><td>12.4473</td><td>51.347</td><td>0.7503</td></tr><tr><td>Local EFQ+RHT</td><td>13.9240</td><td>49.267</td><td>1.7087</td><td>11.9843</td><td>51.882</td><td>0.4070</td></tr><tr><td>Local joint R, G</td><td>14.1962</td><td>48.937</td><td>1.9917</td><td>12.0672</td><td>51.763</td><td>0.4822</td></tr><tr><td>Logit-diag + CD</td><td>22.2033</td><td>42.676</td><td>4.6914</td><td>12.5987</td><td>50.910</td><td>1.0003</td></tr><tr><td>Obj.-EFQ (Logit-diag)</td><td>16.5609</td><td>46.629</td><td>2.7727</td><td>12.3291</td><td>51.372</td><td>0.7162</td></tr><tr><td>Obj.-EFQ+RHT (Logit-diag)</td><td>13.7342</td><td>49.444</td><td>1.6576</td><td>11.9123</td><td>51.931</td><td>0.3854</td></tr><tr><td>Obj. joint R, G (Logit-diag)</td><td>14.0767</td><td>48.996</td><td>1.9473</td><td>12.0450</td><td>51.838</td><td>0.4753</td></tr><tr><td>Guided-shared + CD</td><td>23.9442</td><td>41.620</td><td>4.9713</td><td>12.6437</td><td>50.867</td><td>1.0728</td></tr><tr><td>Obj.-EFQ (Guided-shared)</td><td>16.4664</td><td>46.651</td><td>2.8788</td><td>12.3429</td><td>51.420</td><td>0.7545</td></tr><tr><td>Obj.-EFQ+RHT (Guided-shared)</td><td>13.9382</td><td>49.178</td><td>1.7625</td><td>11.9769</td><td>51.870</td><td>0.4155</td></tr><tr><td>Obj. joint R, G (Guided-shared)</td><td>14.3269</td><td>48.742</td><td>2.0438</td><td>12.0278</td><td>51.786</td><td>0.5047</td></tr><tr><td>Model-Fisher-diag + CD</td><td>22.2340</td><td>42.322</td><td>4.7415</td><td>12.6235</td><td>50.878</td><td>1.0490</td></tr><tr><td>Obj.-EFQ (Model-Fisher-diag)</td><td>16.5394</td><td>46.546</td><td>2.9141</td><td>12.3806</td><td>51.391</td><td>0.7628</td></tr><tr><td>Obj.-EFQ+RHT (Model-Fisher-diag)</td><td>13.8478</td><td>49.263</td><td>1.7129</td><td>11.9169</td><td>51.910</td><td>0.4053</td></tr><tr><td>Obj. joint R, G (Model-Fisher-diag)</td><td>14.1199</td><td>48.867</td><td>2.0193</td><td>12.0192</td><td>51.856</td><td>0.5045</td></tr></table>

Results and analysis. Obj.-EFQ+RHT is best on all three metrics for every model and both bit widths. Across the 18 matched objective-aware versus local comparisons, downstream weighting lowers PPL in 15, improves accuracy in 14, and lowers logit MSE in 16; all three metrics improve in every RHT pair. The exceptions preclude a universal claim: at three bits, Llama-2-13B Obj. joint R, G lowers MSE from 0.8936 to 0.7958 but raises PPL from 10.3036 to 10.4606, while Qwen3-14B Obj.- EFQ is worse than Local EFQ on all three metrics. RHT also outperforms joint R, G on all metrics in all 12 objective-matched comparisons. The theoretical joint optimum concerns $\operatorname { t r } ( G ^ { \top } R ^ { \top } A R G )$ for a fixed positive-definite metric and a common isotropic error second moment, not the complete finitegrid loss. Refitting grids after rotation can change scales, clipping, and error correlations, and diagonal downstream curvature need not predict finite weight perturbations. These efects could ofset a lower trace, but the reported measurements do not distinguish their contributions. The evidence supports objective-aware RHT as the strongest tested configuration, not a universal superiority of downstream metrics or trace-optimal rotations.

Table 18: C4 results for Llama-2-7B (calibration seed 7). The BF16 reference has PPL 8.2949, accuracy 53.960%, and zero logit MSE. Bold marks the best quantized value for each metric and bit width.
<table><tr><td rowspan="2">Method</td><td colspan="3">3 bits (M = 8)</td><td colspan="3">4 bits (M = 16)</td></tr><tr><td></td><td>PPL ↓ Acc. (%) ↑ Logit MSE↓</td><td></td><td></td><td>PPL ↓ Acc. (%) ↑ Logit MSE ↓</td><td></td></tr><tr><td>Local EFQ</td><td>12.5795</td><td>47.581</td><td>4.0825</td><td>8.9656</td><td>52.803</td><td>0.7504</td></tr><tr><td>Local EFQ+RHT</td><td>10.4755</td><td>49.963</td><td>2.0937</td><td>8.6259</td><td>53.226</td><td>0.3747</td></tr><tr><td>Local joint R, G</td><td>12.1531</td><td>47.710</td><td>3.3237</td><td>8.8598†</td><td>52.860</td><td>0.6428</td></tr><tr><td>Obj.-EFQ</td><td>12.7880</td><td>47.431</td><td>3.7904</td><td>8.9166</td><td>52.875</td><td>0.6589</td></tr><tr><td> $\mathrm { O b j . - E F Q + R H T }$ </td><td>10.2043</td><td>50.380</td><td>1.8171</td><td>8.5971</td><td>53.304</td><td>0.3356</td></tr><tr><td>Obj. joint R, G</td><td>11.6051</td><td>48.305</td><td>2.4838</td><td>8.7628</td><td>52.893</td><td>0.5361</td></tr></table>

Table 19: C4 results for Llama-2-13B (calibration seed 7). The BF16 reference has PPL 7.7063, accuracy 55.092%, and zero logit MSE. Bold marks the best quantized value for each metric and bit width.
<table><tr><td></td><td colspan="3">3 bits (M = 8)</td><td colspan="3">4 bits (M = 16)</td></tr><tr><td>Method</td><td></td><td>PPL ↓ Acc. (%) ↑ Logit MSE ↓</td><td></td><td></td><td>PPL ↓ Acc. (%) ↑ Logit MSE ↓</td><td></td></tr><tr><td>Local EFQ</td><td>9.7504</td><td>51.045</td><td>0.8677</td><td>8.0218</td><td>54.381</td><td>0.1844</td></tr><tr><td>Local EFQ+RHT</td><td>9.1778</td><td>51.953</td><td>0.6449</td><td>7.9397</td><td>54.517</td><td>0.1415</td></tr><tr><td>Local joint R, G</td><td>10.3036</td><td>49.964</td><td>0.8936</td><td>8.0735</td><td>54.237</td><td>0.2005</td></tr><tr><td>Obj.-EFQ</td><td>9.5984</td><td>51.345</td><td>0.8146</td><td>7.9984</td><td>54.427</td><td>0.1734</td></tr><tr><td>Obj.-EFQ+RHT</td><td>9.0501</td><td>52.146</td><td>0.6041</td><td>7.9281</td><td>54.534</td><td>0.1303</td></tr><tr><td>Obj. joint R, G</td><td>10.4606</td><td>49.643</td><td>0.7958</td><td>8.0559</td><td>54.227</td><td>0.1819</td></tr></table>

Table 20: C4 results for Qwen3-14B (calibration seed 7). The BF16 reference has PPL 15.4016, accuracy 44.9696%, and zero logit MSE. Bold marks the best quantized value for each metric and bit width.
<table><tr><td></td><td colspan="3">3 bits (M = 8)</td><td colspan="3">4 bits (M = 16)</td></tr><tr><td>Method</td><td>PPL↓</td><td>Acc. (%) ↑</td><td>Logit MSE ↓</td><td>PPL↓</td><td>Acc. (%) ↑</td><td>Logit MSE ↓</td></tr><tr><td>Local EFQ</td><td>20.4378</td><td>40.709</td><td>2.0484</td><td>16.1635</td><td>44.161</td><td>0.4624</td></tr><tr><td>Local EFQ+RHT</td><td>17.6068</td><td>42.681</td><td>1.1352</td><td>15.7867</td><td>44.484</td><td>0.2428</td></tr><tr><td>Local joint R, G</td><td>18.1779</td><td>42.232</td><td>1.4040</td><td>15.9107†</td><td>44.376</td><td>0.3178</td></tr><tr><td>Obj.-EFQ</td><td>20.6103</td><td>40.643</td><td>2.1770</td><td>16.1170</td><td>44.191</td><td>0.4535</td></tr><tr><td> $\mathrm { O b j . - E F Q + R H T }$ </td><td>17.5376</td><td>42.739</td><td>1.1176</td><td>15.7600</td><td>44.546</td><td>0.2307</td></tr><tr><td>Obj. joint R, G</td><td>18.0553</td><td>42.397</td><td>1.4362</td><td>15.8398†</td><td>44.463</td><td>0.3073</td></tr></table>