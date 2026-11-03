# MetaPOCS: Meta-Learned Multi-Exit Unfolding for Budget-Aware Seismic Data Interpolation

Baisong Jiang, Chunxia Zhang, Xiaoli Wei, Anxiang Di, Nanqi Zhao, Deng Xiong

Abstract—Seismic data interpolation must reconcile reconstruction fidelity with the computational budgets of acquisition quality control and subsequent imaging. Fixed-depth unfolding networks are optimized for a single operating point, whereas their intermediate estimates may be insufficiently trained for early deployment. We present MetaPOCS, a meta-learned multiexit unfolding framework for budget-aware seismic interpolation. Building on a spatial-spectral POCS reconstruction backbone, MetaPOCS supervises intermediate stages and preserves the acquired traces at every exit. A Meta-GF-based optimization procedure constructs exit-oriented parameter displacements from a common initialization and learns their fusion weights through an equal-exit outer objective. The weights vary across parameter tensors and respect the nested support of the reconstruction stages. We derive the resulting fusion hypergradient to characterize how the outer objective reallocates competing updates, and distinguish this mechanism from scalar loss reweighting. At inference, one checkpoint provides a set of reconstruction budgets through a latency-calibrated exit policy. The evaluation framework couples matched-cost interpolation comparisons with finite-update diagnostics and a fixed reverse-time-migration operator, connecting multi-exit optimization to waveform and imaging fidelity. MetaPOCS thus provides a unified formulation for studying the reconstruction-computation trade-off of seismic unfolding across deployment budgets.

Index Terms—Seismic data interpolation, deep unfolding, multi-exit networks, meta-learning, computational budget, reverse-time migration.

## I. INTRODUCTION

CQUIRING complete seismic data supports subsurface imaging and reservoir characterization. Terrain restrictions, receiver failures, and acquisition costs can leave irregularly sampled or missing traces. Interpolation seeks to recover the unobserved wavefield while preserving the information in acquired data. Its practical value depends on event kinematics and amplitude fidelity, as well as on computational resources available before subsequent processing.

Existing approaches can be organized into three classes: physical or mathematical prior-based model-driven methods, constrained data-driven methods, and hybrid methods integrating data and models. This classification separates assumptions about the wavefield from the mechanism used to learn or impose those assumptions.

Physical prior-based methods exploit propagation laws or coherent wavefield structure. Ronen [1] formulated waveequation trace interpolation, and Canning and Gardner [2] addressed regularly sampled data using wave-equation principles. Migration-based formulations incorporate propagation geometry into reconstruction [3], [4]; their usefulness depends on the propagation model and acquisition description. Prediction filtering provides another established route. Spitz [5] exploited frequency-space predictability, while nonstationary prediction filters [6] and streaming filters with varying smoothness [7] relaxed stationarity or computational requirements. Dynamic matching uses local event alignment to guide interpolation [8]. These methods encode meaningful seismic structure, but their assumptions become harder to satisfy across wide gaps and rapidly changing events.

Sparse-transform methods formulate reconstruction in a domain where complete data admit a compact representation [9]. Fourier sparse inversion and antileakage Fourier interpolation address incomplete spatial sampling [10]–[12]. Curvelet and seislet representations capture multiscale directional or slopealigned structure [13], [14], and POCS-type iterations combine transform-domain regularization with reinsertion of acquired traces [15]. Local Radon and apex-shifted hyperbolic Radon representations supply additional seismic dictionaries [16], [17]. Sampling operators for off-grid data extend this framework beyond simple trace deletion [18]. Transform choice, geometry, and regularization jointly determine which missing events can be recovered.

Constrained data-driven methods learn nonlinear representations from examples. Residual learning and texture-constrained reconstruction have been explored for seismic interpolation [19], [20], as have conditional adversarial models [21]. Spatial-Fourier losses and wavelet representations provide alternatives to purely pointwise supervision [22], [23]. Transformer and feature-fusion models introduce broader contextual interactions [24], [25]. Score-based and diffusion formulations address stochastic reconstruction [26], [27], while learned off-grid regularization explicitly incorporates the sampling operator [28]. Generalization to unseen acquisition patterns and preservation of measured amplitudes nevertheless require explicit evaluation.

Hybrid deep unfolding networks retain an iterative reconstruction structure while learning parts of the prior. CNNassisted POCS [29], POCS-Net [30], and SP-Net [31] illustrate this direction. Our related SSI-POCS work [32] develops a fixed-stage spatial-spectral prior for large-gap interpolation by coupling Restormer and fast Fourier convolution branches through a dual-domain fusion module (DFM). The present work adopts that reconstruction backbone and addresses a complementary problem: optimizing its intermediate reconstructions for deployment at different computational budgets. The spatial-spectral prior, data reinsertion, analysis-synthesis regularization, and frequency-domain supervision build on SSI-POCS. Here, each exit receives explicit time-space amplitude and frequency-domain supervision, aligning the reconstruction objective with the implicit dual-domain representation. The focus of the new formulation is the coordination and use of nested reconstruction exits.

The question considered here is how to train that reconstruction process for several deployment budgets within a single model. A separate fixed-depth model can be trained for each budget, but this increases training and checkpoint storage. Conversely, a network optimized only at its deepest stage need not provide a reliable shallow reconstruction. Deep supervision [33] and multi-exit networks [34] suggest supervising intermediate outputs. However, exits share early stages and can prefer different parameter updates. GradNorm [35] addresses unequal training rates, while gradient surgery [36] addresses conflicting directions. Meta-reweighting [37] learns weighting rules through an outer objective. Most directly, Meta-GF [38] learns how to fuse exit-specific updates in dynamicdepth networks. Meta-learning here concerns the training of reconstruction exits, not few-shot adaptation to a new survey.

For seismic unfolding, exit coordination is coupled to the acquisition operator. Each exit estimates the same amplitude field under the same measured-data constraint, but a shallow exit must complete that estimate with fewer refinement stages. Early-stage parameters consequently serve two roles: producing an immediately usable wavefield and preparing features for later iterations. Their preferred updates can differ even though the reconstruction target is shared. Moreover, the nested computation graph gives different exits access to different parameter tensors. These properties motivate an explicit treatment of parameter support and update interference, while the phase sensitivity of migration motivates evaluation beyond aggregate reconstruction error.

MetaPOCS combines supervised unfolding exits with parameter-tensor-wise fusion of exit-oriented training displacements. Following Meta-GF [38], the fusion weights are learned from an outer reconstruction objective. The formulation specializes this update to acquisition-consistent seismic outputs and makes the support of each exit explicit. An equal-exit outer objective retains all reconstruction budgets in the optimization target; inference then selects an available exit according to a calibrated latency budget. This separates the problem of training reliable intermediate wavefields from the choice of a deployment operating point.

The contributions are summarized as follows:

1) We formulate seismic interpolation as a family of supervised, acquisition-consistent unfolding exits. A single checkpoint supports several deployment budgets through a latency-calibrated prefix selection rule.

2) We adapt Meta-GF to the nested parameter support of the unfolding process and derive its tensor-wise fusion hypergradient. A scalar meta-supervision formulation and a finite-update analysis distinguish loss reweighting from the redistribution of exit-oriented optimization trajectories.

3) We establish an evaluation framework linking budgetdependent reconstruction, optimization interference, and migration-domain error. Matched-backbone and matched-computation controls separate training effects from the spatial-spectral prior inherited from SSI-POCS.

Section II describes the model and training. Section III specifies the evaluation protocol. Sections IV and V organize reconstruction and mechanism experiments. Section VI examines downstream imaging, followed by limitations and conclusions.

## II. BUDGET-AWARE MULTI-EXIT UNFOLDING

## A. Problem Formulation and Classical POCS Reconstruction

In field seismic exploration, adverse acquisition conditions, economic constraints, and environmental limitations often lead to incomplete spatial sampling. Let $x ~ \in ~ \mathbb { R } ^ { H \times W }$ denote the latent complete seismic gather (where H and W represent the temporal sampling points and spatial receiver traces, respectively), and let $M \in \{ 0 , 1 \} ^ { H \times W }$ represent the binary sampling mask, where 1 designates an acquired trace and 0 indicates an unrecorded position. The forward observation model for noise-free seismic data interpolation can be formulated as:

$$
y = M \odot x ,\tag{1}
$$

where $y \in \mathbb { R } ^ { H \times W }$ represents the observed incomplete seismic record, and  denotes the element-wise Hadamard product. Recovering x from y is intrinsically an ill-posed underdetermined inverse problem. To obtain a physically plausible solution, compressive sensing (CS) frameworks exploit the underlying transform-domain sparsity of complete wavefields, casting the reconstruction as a sparsity-constrained regularized optimization problem:

$$
\arg \operatorname* { m i n } _ { \boldsymbol { x } } \frac { 1 } { 2 } \| \boldsymbol { y } - \boldsymbol { M } \odot \boldsymbol { x } \| _ { 2 } ^ { 2 } + \lambda \| \boldsymbol { \Psi } \boldsymbol { x } \| _ { 1 } ,\tag{2}
$$

where the first term represents the data-fidelity constraint in the time-space (t-x) domain, the second term enforces sparsity regularization under a specific transform operator Ψ, and $\lambda > 0$ governs the trade-off between measurement fidelity and transform sparsity.

Under the framework of proximal gradient descent, such as the Iterative Shrinkage-Thresholding Algorithm (ISTA), the iterative solution for problem (2) at iteration k is given by:

$$
r ^ { k } = x ^ { k - 1 } - \tau M ^ { T } ( M \odot x ^ { k - 1 } - y ) ,\tag{3}
$$

$$
\begin{array} { r } { { \boldsymbol x } ^ { k } = \mathrm { p r o x } _ { \lambda \tau , \parallel \Psi \cdot \parallel _ { 1 } } ( { \boldsymbol r } ^ { k } ) = \Psi ^ { - 1 } S _ { \lambda \tau } ( \Psi { \boldsymbol r } ^ { k } ) , } \end{array}\tag{4}
$$

where τ denotes the gradient descent step size, prox(·) denotes the proximal operator, and $S _ { \lambda } ( z ) = \mathrm { s i g n } ( z ) \odot \mathrm { m a x } ( | z | - \lambda , 0 )$ is the element-wise soft-thresholding function. Because seismic trace selection satisfies $M ^ { T } M \ = \ M \odot M \ = \ M$ , setting the step size to $\tau = 1$ reduces ISTA to the classical Projection Onto Convex Sets (POCS) algorithm. Specifically, the POCS iterative update simplifies to alternating between transformdomain thresholding and spatial data reinsertion:

$$
\hat { x } ^ { k } = \Psi ^ { - 1 } S _ { \lambda _ { k } } ( \Psi x ^ { k } ) ,\tag{5}
$$

$$
x ^ { k + 1 } = y + ( 1 - M ) \odot { \hat { x } } ^ { k } ,\tag{6}
$$

where $x ^ { 0 } = y$ , and $\lambda _ { k }$ denotes the iteration-dependent threshold parameter.

Although classical POCS algorithms provide rigorous mathematical tractability and enforce exact data consistency on acquired traces, they exhibit notable operational drawbacks:

1) Rigid Prior Representations: Predefined analytical transforms Ψ (e.g., Fourier, Radon, or Curvelet transforms) are fundamentally linear and struggle to represent complex subsurface wavefields characterized by severe spatial aliasing, conflicting dips, and large contiguous trace gaps.

2) Heuristic Hyperparameter Tuning: Manual scheduling of threshold parameters $\lambda _ { k }$ often requires trial-anderror heuristics across different shot gathers, yielding sub-optimal convergence trajectories.

3) High Iteration Latency: Traditional POCS workflows demand dozens or even hundreds of iterations to achieve acceptable wavefield restoration, severely restricting computational throughput.

To circumvent these limitations, deep unfolding networks (DUNs) map iterative stages into trainable cascade architectures, replacing handcrafted transforms with parameterized neural networks [31], [32]. However, existing DUNs are typically optimized as static, fixed-depth models [32]. When deployed under strict computational budgets requiring early truncated execution, they suffer a severe performance cliff because intermediate states act primarily as transitional features rather than reliable wavefield estimates [32].

To resolve this conflict, we propose MetaPOCS, a metalearned multi-exit unfolding framework featuring: (1) Budget-Aware Multi-Exit Backbone: Intermediate stages act as independently supervised exits (Exit $t \in \{ 1 , \ldots , T \} )$ ensuring strict data consistency; (2) Support-Aware Meta Gradient Fusion: Tensor-wise meta-learning harmonizes conflicting parameter trajectories across shared early layers; (3) Calibrated Dynamic Deployment: A single checkpoint enables budget-driven prefix execution along a quality-latency Pareto frontier.

## B. Implicit Dual-Domain Prior Within Each Unrolled Stage

Classical POCS relies on a fixed, handcrafted transform Ψ in which complete seismic data are assumed to be sparse. In our deep unfolding framework, the rigid analysis and synthesis transforms are replaced by parameterized, learnable neural networks. Let $F _ { k }$ and $G _ { k }$ denote the forward and inverse transform networks at the k-th iteration stage, respectively.

As illustrated in Fig. 1, the internal data flow of a single unrolled stage strictly mimics the mathematical logic of the classical POCS algorithm. First, the input wavefield estimate $x ^ { k }$ is mapped into a sparse feature representation via the forward transform: $x _ { f } ^ { k } \ = \ F _ { k } ( x ^ { k } )$ . Subsequently, a soft-thresholding operator enforces sparsity on this learned representation:

$$
S _ { \lambda } ( x _ { f } ^ { k } ) = \mathrm { s i g n } ( x _ { f } ^ { k } ) \odot \operatorname* { m a x } ( | x _ { f } ^ { k } | - \lambda , 0 ) ,\tag{7}
$$

where $\lambda \geq 0$ is a learnable stage-specific threshold parameter. The sparsified features are then mapped back to the seismicamplitude domain via the inverse transform to generate the intermediate reconstruction: $\hat { x } ^ { k } \ = \ G _ { k } ( S _ { \lambda } ( x _ { \epsilon } ^ { k } ) )$ . Finally, the reinsert module explicitly imposes the hard data-fidelity constraint by replacing the reconstructed traces at the acquired positions with the original observations y, yielding the updated state for the next iteration:

$$
x ^ { k + 1 } = y + ( 1 - M ) \odot \hat { x } ^ { k } .\tag{8}
$$

While the macroscopic unrolling follows the POCS logic, the core innovation lies in the internal architecture of $F _ { k }$ and $G _ { k }$ . Unlike conventional single-domain unfolding methods that struggle to reconstruct large contiguous gaps due to restricted receptive fields, we design a heterogeneous dualbranch architecture. After an initial convolutional projection, the feature processing diverges into a Spatial Branch and a Spectral Branch. The Spatial Branch employs Restormer blocks (RTB) to capture local waveform structures and preserve the kinematic continuity of seismic events along the time-space lattice. Concurrently, the Spectral Branch utilizes Fast Fourier Convolution (FFC) to extract global contextual representations in the frequency domain, effectively bridging extensive missing regions that lack local structural information.

To fully exploit the complementarity between the local spatial details and the global spectral context, the features from both branches are aggregated through a Dual-domain Fusion Module (DFM). Instead of a naive feature concatenation, the DFM dynamically exchanges cross-domain context through bidirectional cross-channel attention and adaptive gated modulation, ensuring that the network suppresses incoherent artifacts while enhancing spectral energy concentration. Detailed operations of the internal cross-domain feature refinement and gated modulation mechanisms are referred to [32]. The modulated features are subsequently processed by a trailing RTB and convolution layer to produce the final branch output.

Ultimately, this structural design encodes an implicit dualdomain prior into the network weights. It avoids explicitly maintaining and alternately updating dual variables in two separate mathematical spaces; rather, the stage-wise composition of the learned transforms naturally regularizes the reconstruction by fusing multiscale spatial and spectral physical properties.

## C. Trained Exits and Budget-Aware Deployment

Let $D = \{ d _ { 1 } , \ldots , d _ { T } \}$ denote the set of trained exit depths, where T is the total number of supervised exits. Exit t (where $t \in \{ 1 , \ldots , T \} )$ directly outputs the intermediate wavefield estimate $\hat { x } _ { t } = x ^ { d _ { t } }$ without requiring any auxiliary reconstruction head. While all T stages are evaluated to compute the multiexit losses during training, only the required prefix up to the selected depth is executed during deployment.

![](images/4b7a9ad8b6f56352c6e8781dbb789e41118cb0a18cffdff45fe3200fdb44060c.jpg)  
Fig. 1. Detailed architecture of a single unrolled stage.

![](images/494277cc9a61362275241549c379deaec8247eb2bdee4f54a1b77d278362e1e3.jpg)  
Fig. 2. MetaPOCS structure. Each stage combines an implicit timespace/Fourier prior with exact data consistency (DC), and each trained exit receives explicit time-space and spectral supervision. Reconstruction stages inherit SSI-POCS; multi-exit coordination and deployment are the focus here. The lower block operates during training. Inference runs a budget-selected prefix, without computing all exits and subsequently discarding their cost.

To accommodate real-world deployment constraints for a given hardware device, numerical precision, and input geometry, we define a maximum allowable latency budget B. Let $\tau _ { t }$ represent the profiled inference latency and $q _ { t }$ denote the validation-set quality score for Exit t. The optimal deployment exit $t ^ { \star }$ is dynamically selected by maximizing the reconstruction quality subject to the latency constraint:

$$
t ^ { \star } ( B ) = \arg \operatorname* { m a x } _ { t : \tau _ { t } \leq B } q _ { t } .\tag{9}
$$

The feasible set in Eq. (9) comprises all exits whose profiled inference latency satisfies the budget $B ;$ an empty set indicates an unsupported budget constraint. This validation-driven selection policy is crucial because reconstruction accuracy in deep unfolding networks does not strictly increase monotonically with depth. By establishing this budget-to-exit mapping offline prior to actual testing, only the selected prefix is executed online. Consequently, the computational cost is strictly limited to the executed layers, providing a guaranteed, budget-aware seismic inference framework tailored for latency-sensitive applications.

## D. Explicit Dual-Domain Exit Supervision and Scalar Meta-Supervision

The implicit prior controls the representation available to the unfolded stages, whereas the training objective specifies which reconstruction errors that representation must reduce. We supervise every exit explicitly in two complementary views of the same seismic record: samplewise amplitudes on the time-space grid and the complex two-dimensional Fourier spectrum. The common exit objective is

![](images/3c5a5f52c64cb9c0683876479e8ef7ed8e45f8ccdec4a1626ca2001c6e0cfe5f.jpg)  
Fig. 3. MetaPOCS meta fusion.

$$
\mathcal { L } _ { k } = \mathcal { L } _ { 1 , k } + \eta _ { s } \mathcal { L } _ { \mathrm { s y m } , k } + \eta _ { f } \mathcal { L } _ { \mathrm { F F L } , k } .\tag{10}
$$

The time-space amplitude term is $\mathcal { L } _ { 1 , k } = ( H W ) ^ { - 1 } \| \hat { x } _ { k } - x \| _ { 1 }$ It directly penalizes samplewise waveform discrepancies and is less dominated by isolated large residuals than a squared error. With noiseless hard reinsertion, its residual is zero on measured samples, while its full-grid normalization retains a dependence on the missing fraction. All exits within a comparison use the same coefficients. Evaluation additionally normalizes errors by the number of missing samples, allowing comparisons across acquisition densities.

The transform term is a prefix mean

$$
\mathcal { L } _ { \mathrm { s y m } , k } = \frac { 1 } { d _ { k } } \sum _ { t = 1 } ^ { d _ { k } } \mathrm { M S E } \Big ( G _ { t } ( F _ { t } ( x ^ { t - 1 } ) ) , x ^ { t - 1 } \Big ) .\tag{11}
$$

The prefix average makes the strength of transform regularization comparable across exits. It encourages the unthresholded analysis-synthesis pair to approximate the identity along the unfolding trajectory. This term regularizes the learned transforms; the explicit dual-domain reconstruction constraints are supplied by the amplitude term and focal frequency loss [42]. The latter weights the complex Fourier residual:

$$
\mathcal { L } _ { \mathrm { F F L } , k } = \frac { 1 } { H W } \sum _ { u , \nu } \omega _ { k , u \nu } | \mathcal { F } ( \hat { x } _ { k } ) _ { u \nu } - \mathcal { F } ( x ) _ { u \nu } | ^ { 2 } .\tag{12}
$$

Writing $r _ { k , u \nu } = \mathcal { F } ( \hat { x } _ { k } ) _ { u \nu } - \mathcal { F } ( x ) _ { u \nu }$ , the default weighting is

$$
\omega _ { k , u \nu } = \mathrm { s t o p g r a d } \left( \frac { \left| r _ { k , u \nu } \right| } { \operatorname* { m a x } _ { u ^ { \prime } , \nu ^ { \prime } } \left| r _ { k , u ^ { \prime } \nu ^ { \prime } } \right| } \right) ,\tag{13}
$$

with $\omega ~ = ~ 0$ for a zero residual spectrum. The Fourier transform is orthonormal, and the spectral weights are held fixed during each loss differentiation. This weighting emphasizes frequencies with large reconstruction discrepancies rather than imposing a fixed high-frequency preference. Because the residual is complex valued, the objective is sensitive to both amplitude and phase discrepancies. For an orthonormal transform, an unweighted squared spectral residual equals its time-space counterpart by Parseval's identity. Here, residualdependent spectral weights and a time-space $L _ { 1 }$ term provide distinct error penalties. Thus, the architecture and objective play different roles: Restormer-FFC-DFM supplies an implicit dual-domain hypothesis class, while $\mathcal { L } _ { 1 , k }$ and $\mathcal { L } _ { \mathrm { F F L } , k }$ provide explicit error signals that select a reconstruction within that class. Their empirical complementarity is examined through the crossed prior-objective comparison in Table V.

This coupling is especially consequential in a shared multiexit network. Let $s ( j )$ be the stage containing parameter tensor $\theta _ { j }$ and let $\mathcal { K } _ { j } ~ = ~ \{ k ~ : ~ d _ { k } \geq s ( j ) \}$ collect the exits whose forward paths contain that tensor. Under equal or scalar supervision, its inner gradient is

$$
\nabla _ { \theta _ { j } } \mathcal { L } _ { \mathrm { i n } } = \sum _ { k \in \mathcal { K } _ { j } } \pi _ { k } \left( \nabla _ { \theta _ { j } } \mathcal { L } _ { 1 , k } + \eta _ { f } \nabla _ { \theta _ { j } } \mathcal { L } _ { \mathrm { F F L } , k } + \eta _ { s } \nabla _ { \theta _ { j } } \mathcal { L } _ { \mathrm { s y m } , k } \right) .\tag{14}
$$

Earlier stages receive explicit dual-domain signals from more exits, whereas a stage introduced after depth $d _ { k }$ receives no gradient from exit k. The time-space and spectral objectives may agree on some shared tensors and compete on others; different exit depths can introduce an additional source of disagreement. Scalar supervision can change the global contribution of an exit through $\pi _ { k }$ , but it scales all parameter tensors and all loss components of that exit together. This limitation motivates the supported, tensor-dependent update fusion developed in the next subsection.

Equal supervision minimizes $K ^ { - 1 } \sum _ { k } \mathcal { L } _ { k }$ . A scalar metasupervision control uses

$$
\pi _ { k } ( \alpha ) = \rho / K + ( 1 - \rho ) \operatorname { s o f t m a x } ( \alpha ) _ { k } ,\tag{15}
$$

$$
\boldsymbol { \theta } ^ { \star } ( \alpha ) \approx \mathrm { O p t i m i z e } _ { \theta } \sum _ { k } \pi _ { k } ( \alpha ) \mathcal { L } _ { k } ( \mathcal { D } _ { \mathrm { i n } } ; \boldsymbol { \theta } ) ,\tag{16}
$$

$$
\operatorname* { m i n } _ { \alpha } \phantom { \frac { 1 } { \alpha } } \frac { 1 } { K } \sum _ { k } \mathcal { L } _ { k } ( \mathcal { D } _ { \mathrm { m e t a } } ; \theta ^ { \star } ( \alpha ) ) .\tag{17}
$$

The lower bound $\rho / K$ maintains supervision at every exit; $\rho = 1$ recovers equal weights. The dependence of $\theta ^ { \star }$ on $\alpha$ is approximated through inner optimization using the multilevel differentiation framework Betty [43]. The outer objective assigns equal importance to all exits, so decreasing a training weight does not directly remove that exit from the meta objective. This scalar formulation changes the relative contribution of each exit uniformly across its parameters.

## E. Meta-Learned Fusion of Exit-Oriented Updates

The principal training variant adapts Meta-GF [38]. Each exit-oriented trajectory starts from the same parameters $\theta _ { 0 }$ and optimizer state, using

$$
\mathcal { T } _ { k } = \mathcal { L } _ { k } + \beta \sum _ { \ell \neq k } \mathcal { L } _ { \ell } , \qquad 0 \leq \beta \leq 1 .\tag{18}
$$

After the prescribed inner traversal, $\Delta _ { k , j } = \theta _ { k , j } - \theta _ { 0 , j }$ is the displacement of parameter tensor $j .$ Auxiliary terms train other exits during each trajectory. Displacements are cached and detached for the outer update.

Let $A _ { k , j }$ indicate whether tensor j belongs to exit $k ' s$ inference prefix. Define

$$
w _ { k , j } ( \phi ) = \frac { A _ { k , j } \sigma ( \phi _ { k , j } ) } { \sum _ { \ell = 1 } ^ { K } A _ { \ell , j } \sigma ( \phi _ { \ell , j } ) } ,\tag{19}
$$

$$
\theta _ { j } ^ { + } ( \phi ) = \theta _ { 0 , j } + \sum _ { k = 1 } ^ { K } w _ { k , j } ( \phi ) \operatorname { s t o p g r a d } ( \Delta _ { k , j } ) ,\tag{20}
$$

$$
\operatorname* { m i n } _ { \phi } \quad \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathcal { L } _ { k } ( \mathcal { D } _ { \mathrm { m e t a } } ; \theta ^ { + } ( \phi ) ) .\tag{21}
$$

Every backbone tensor has at least one supported exit. For independent stage parameters, $A _ { k , j } = 1$ precisely when tensor j belongs to a stage no deeper than $d _ { k } ;$ for stage-shared parameters, all exits are supported. The weights are nonnegative and sum to one over the supported exits of each tensor. Thus, $\theta _ { j } ^ { + }$ lies in the convex hull of the corresponding candidate tensor values, although neither the reconstruction network nor its loss is convex. Auxiliary objectives can move parameters outside the target exit's prefix, but the support mask excludes those displacements from that exit's fusion contribution.

After the outer updates, exponential smoothing gives $\phi $ $\mu \phi _ { \mathrm { o l d } } + ( 1 - \mu ) \phi$ , where $\mu \in [ 0 , 1 )$ . The corresponding fused parameters initialize the next round. Each displacement already includes the inner learning rate and optimizer effects, so Eq. (20) uses it directly. The outer derivative passes through the fused model and its weights while treating the candidate trajectories as fixed.

To expose the weighting mechanism, define $\begin{array} { r l } { \bar { \Delta } _ { j } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { k } w _ { k , j } \Delta _ { k , j } } \end{array}$ and let $h _ { j }$ be the gradient of the equal-exit outer objective with respect to $\theta _ { j } ^ { + }$ . For a supported exit $\ell ,$ differentiating the normalized sigmoid weights yields

$$
\frac { \partial \theta _ { j } ^ { + } } { \partial \phi _ { \ell , j } } = w _ { \ell , j } \big ( 1 - \sigma ( \phi _ { \ell , j } ) \big ) ( \Delta _ { \ell , j } - \bar { \Delta } _ { j } ) ,\tag{22}
$$

$$
\frac { \partial \mathcal { L } _ { \mathrm { o u t } } } { \partial \phi _ { \ell , j } } = w _ { \ell , j } \big ( 1 - \sigma ( \phi _ { \ell , j } ) \big ) \left. h _ { j } , \Delta _ { \ell , j } - \bar { \Delta } _ { j } \right. .\tag{23}
$$

The inner product is taken over all entries of tensor $j ;$ unsupported logits have zero derivative. Equation (23) compares an exit-oriented displacement with the currently fused displacement under the outer loss gradient. A candidate that is locally more favorable than the current mixture receives a negative logit gradient and hence an increased weight under gradient descent. The assessment is tensor dependent, unlike scalar loss reweighting.

This derivative characterizes local weight adaptation, rather than guaranteeing simultaneous descent of every exit. It also identifies a limiting case: when all candidate displacements coincide, the fusion hypergradient vanishes. At $\beta = 1$ , the task objectives are identical, and identical minibatch orders and stochastic states produce this collapse. Both observations motivate the finite-update diagnostics in Section V.

## F. Optimization Schedule and Computational Cost

Algorithm 1 describes a full fusion round. If an inner traversal contains $N _ { \mathrm { i n } }$ minibatches and the outer update uses $N _ { \mathrm { m e t a } }$ minibatches, the round performs $K N _ { \mathrm { i n } }$ task updates and $N _ { \mathrm { m e t a } }$ meta updates. Its training cost therefore exceeds that of one equal-supervision traversal. The experimental comparison distinguishes data exposure from elapsed GPU time. Fusion introduces KJ logits for J parameter tensors and storage for candidate displacements; these quantities are training overhead, since the fused reconstruction backbone alone is needed for inference.

A periodic schedule reduces the number of full rounds by refreshing the fusion weights every m outer epochs after a warm-up. Intermediate epochs either minimize the equal-exit mean or combine current task gradients with frozen fusion weights. The latter operates on instantaneous gradients rather than the cached multi-step displacements of Eq. (20), so the two schedules define separate optimization variants.

Algorithm 1 A full MetaPOCS fusion round   
Require: Parameters $\theta _ { 0 } .$ optimizer state $s _ { 0 } .$ logits $\phi ,$ exits $\varepsilon ,$   
inner/meta data.   
1: for $k = 1 , \ldots , K$ do   
2: Restore $( \theta , s ) = ( \theta _ { 0 } , s _ { 0 } )$   
3: Optimize $\mathcal { T } _ { k }$ for the prescribed traversal.   
4: Store detached displacements $\Delta _ { k } = \theta - \theta _ { 0 }$   
5: end for   
6: Restore the base model; save old logits $\phi _ { \mathrm { o l d } } .$   
7: for each permitted meta minibatch do   
8: Form $\theta ^ { + } \mathopen { } \mathclose \bgroup \left( \phi \aftergroup \egroup \right)$ by Eqs. (19)–(20).   
9: Update only $\phi$ using objective (21).   
10: end for   
11: Optionally set $\phi  \mu \phi _ { \mathrm { o l d } } + ( 1 - \mu ) \phi .$   
12: Install fused parameters $\theta  \theta ^ { + } ( \phi )$   
13: Record steps, wall time, and per-exit quality.

The meta batch distribution is denoted by $\mathcal { D } _ { \mathrm { m e t a } }$ . It can reuse the training distribution or be defined by disjoint acquisition units. These choices distinguish training-objective coordination from held-out meta supervision; checkpoint selection uses a separate monitoring role in the latter setting. Across fixed- and multi-exit controls, Eq. (11) defines the common regularization scale. A fixed-depth implementation returning the sum of stage losses therefore uses coefficient $\eta _ { s } / d .$ The original SSI-POCS recipe forms a separate reference for the relationship to the preceding work.

## III. EXPERIMENTAL SETUP

Numerical completion note: In this revision, \* marks an unverified numerical entry, and framed panels reserve space for experimental figures. The following sections define the comparison protocol and analysis; no result is inferred from a placeholder.

## A. Data, Splits, and Acquisition Geometry

The evaluation comprises synthetic SEG C3 data, North Sea field MAVO data, and controlled acoustic models for downstream imaging. SEG C3 represents structurally complex synthetic wavefields, whereas MAVO provides field acquisition characteristics. The corresponding patch dimensions are $1 2 8 \times 1 2 8$ and $2 5 6 \times 1 1 2$ . Table I defines the acquisitionlevel partitions and physical metadata, including a reserved external-survey test. Field reconstruction accuracy is evaluated by withholding recorded traces so that a reference remains available. Regions that were never recorded support qualitative interpretation but have no direct reconstruction-error reference.

The partitioning unit is a shot gather or a spatial acquisition block rather than an extracted patch. The protocol assigns each unit to one partition before patch extraction; spatial buffers separate strongly correlated neighboring blocks.

Training, held-out meta supervision, monitoring, and testing have distinct roles. The source/receiver identifiers, axis order, temporal crop, and physical sampling interval accompany each partition. Consequently, the number of independent acquisition units is distinct from the number of overlapping training patches.

Three sampling patterns characterize acquisition incompleteness: random trace removal, consecutive gaps, and periodic decimation. The primary missing fractions are $r \in$ $\{ 0 . 3 , 0 . 5 , 0 . 7 \}$ , with $r = 0 . 9$ defining a severe-undersampling condition. A test mask is defined on the full gather before patch extraction and remains identical across methods. Repeated mask realizations are nested within the same acquisition unit. Consecutive gaps are characterized by both their trace fraction and physical width where coordinates are available. In-distribution comparisons use matching training and test conditions; transfer experiments hold the trained model fixed while changing the mask, density, or survey.

The amplitude protocol uses a common reversible normalization determined from observed traces or a training-only calibration. For a scale $s _ { \mathrm { y } } > 0 .$ , the normalized input and target are $y / s _ { y }$ and $x / s _ { y } ,$ and the physical prediction is $s _ { y } \hat { x } _ { \mathrm { n o r m } } .$ The scale is independent of unobserved test amplitudes. Waveform and migration metrics operate in the restored amplitude range using identical preprocessing across methods. Tracewise gain normalization would alter relative amplitudes and is excluded from this comparison.

## B. Controls and Implementation Settings

The reference configuration contains $T = 8$ stages, a feature width of 16, and four attention heads. Stage parameters are independent, and $\mathcal { E } ~ = ~ \{ 2 , 3 , . . . , 8 \}$ gives seven supervised exits. The operating points $d ~ \in ~ \{ 2 , 4 , 6 , 8 \}$ summarize the budget range. The common reconstruction coefficients are $( \eta _ { s } , \eta _ { f } ) = ( 0 . 1 , 1 0 0 )$ ; the objective ablation removes the spectral term while retaining the same amplitude supervision and transform regularization. The optimization configuration uses Adam, initial inner and outer learning rates of $3 \times 1 0 ^ { - 4 }$ batch size four, $\beta = 0 . 4 .$ and $\mu = 0 . 9$ . The final training budget and hardware-dependent measurements remain represented by \*. Equal-exposure comparisons count task and meta updates separately, while equal-time comparisons use total measured GPU time.

The reconstruction comparison includes Curvelet-POCS, POCS-Net [30], SP-Net [31], and a Restormer-based endto-end reconstruction baseline [39]. The matched controls in Table II isolate training effects on the SSI-POCS backbone. The original SSI-POCS recipe is separated from the fixeddepth matched control because its loss coefficients differ. Uniform displacement fusion uses the same exit-oriented trajectories as MetaPOCS, replacing the learned weights with $\begin{array} { l l l } { w _ { k , j } } & { = } & { A _ { k , j } / \sum _ { \ell } A _ { \ell , j } } \end{array}$ . Unrestricted-support fusion instead permits every auxiliary-connected candidate to contribute to each tensor. MetaPOCS remains a Meta-GF-based method throughout these comparisons.

TABLE I  
DATASETS INFORMATION.
<table><tr><td>Dataset</td><td>Shots</td><td>Receivers per Shot</td><td>Sample Time  $( N _ { t } )$ </td><td>Time Sample Interval (∆t)</td><td>Spatial Sampling Interval (∆s)</td><td>Train/val/test</td></tr><tr><td>SEGC3</td><td>45</td><td> $2 0 1 \times 2 0 1$ </td><td>625</td><td>8 ms</td><td>20 m</td><td>25000/10000/10000</td></tr><tr><td>MAVO</td><td>1001</td><td>120</td><td>1500</td><td>4 ms</td><td>25 m</td><td>10000/4000/6000</td></tr><tr><td>F3</td><td>600</td><td>440</td><td>951</td><td>4 ms</td><td>25 m</td><td>8000/3000/6000</td></tr><tr><td>Model94</td><td>277</td><td>480</td><td>2000</td><td>4 ms</td><td>15 m</td><td>10000/3000/3500</td></tr></table>

TABLE II  
TRAINING AND DEPLOYMENT CONTROLS. MATCHED COMPARISONS SHARE THE RECONSTRUCTION BACKBONE, LOSS SCALING, ACQUISITION PARTITIONS, AND MASKS.
<table><tr><td>Control</td><td>Training / deployment</td><td>Isolated factor</td></tr><tr><td>SSI-POCS original Fixed-d matched</td><td>Original recipe, final exit Independent model at each</td><td>Relationship to the preceding reconstruction method Specialized training for each deployment budget</td></tr><tr><td></td><td> $d = { \bar { 2 } } , 4 , 6 , 8$ </td><td></td></tr><tr><td>Deep-only truncated</td><td>Depth-8 supervision; prefix inference</td><td>Intermediate reconstruction without direct supervision</td></tr><tr><td>Equal multi-exit Scalar meta</td><td>Average identical exit losses Eq. (17)</td><td>Direct supervision at several reconstruction depths Global redistribution of exit loss weights</td></tr><tr><td>PCGrad / GradNorm</td><td>Common exits and objectives</td><td>Gradient-direction or gradient-scale balancing</td></tr><tr><td>Uniform displacement</td><td>Same trajectories; uniform</td><td>Extra inner optimization with unlearned fusion</td></tr><tr><td>fusion Unrestricted-support</td><td>supported weights All auxiliary-connected</td><td>Effect of the prefix support restriction</td></tr><tr><td>fusion MetaPOCS</td><td>candidates participate Eqs. (18)–(21); one checkpoint</td><td></td></tr></table>

## C. Metrics, Timing, and Statistical Units

Let $\mathcal { U } = \{ i : M _ { i } = 0 \}$ and $e = { \hat { x } } - x .$ Primary metrics are

$$
\begin{array} { r } { \mathbf { M A E } _ { \mathcal { U } } = \lVert \boldsymbol { Q } \odot \boldsymbol { e } \rVert _ { 1 } / \lVert \mathcal { U } \rVert , } \end{array}\tag{24}
$$

$$
\mathrm { S N R } _ { \mathcal { U } } = 1 0 \log _ { 1 0 } \frac { \| Q \odot x \| _ { F } ^ { 2 } } { \| Q \odot e \| _ { F } ^ { 2 } } ,\tag{25}
$$

$$
E _ { \mathrm { D C } } = \frac { | | \boldsymbol { M } \odot ( \hat { { \boldsymbol { x } } } - { \boldsymbol { y } } ) | | _ { F } } { | | \boldsymbol { M } \odot { \boldsymbol { y } } | | _ { F } + \epsilon } .\tag{26}
$$

Full-gather SNR and SSIM [44] complement the missingregion metrics. SSIM is used only as an evaluation metric, with the same window and amplitude range across methods. The consistency error $E _ { \mathrm { D C } }$ measures preservation of acquired samples, whereas missing-region SNR measures interpolation. Windows with zero reference energy are excluded from SNR and retained in absolute-error summaries. Fixed reflection windows additionally characterize signed amplitude bias, waveform correlation, and arrival-time error. Spectral comparisons use a common orthonormal Fourier transform and physical frequency-wavenumber axes.

The replication protocol specifies three training seeds for the main comparisons, with five for the principal meta-versusequal contrast. Seed-to-seed dispersion describes optimization variability. Uncertainty over acquisition units is summarized by paired bootstrap confidence intervals, resampling shots or buffered blocks while retaining matched methods, masks, and exits. Thus, repeated masks and overlapping patches remain nested observations rather than additional independent geological samples. The principal effect is the paired change in missing-region SNR at matched inference cost; the deepestexit change quantifies any accompanying accuracy trade-off.

Inference profiling measures the selected prefix with batch size one, fixed input dimensions, and a common numerical precision. Device warm-up and synchronization precede timed measurements. Median and 95th-percentile latency describe central cost and runtime variability, respectively; peak allocated memory describes the executed prefix. Network time is separated from preprocessing and gather assembly. Operation counts identify the coverage of Fourier and attention kernels. Training GPU hours include all task trajectories, meta updates, and validation, while checkpoint storage measures the cost of retaining deployable models.

## IV. RECONSTRUCTION AND BUDGET EVALUATION

## A. Main Interpolation Comparison

Table III organizes the full-depth comparison by dataset and sampling pattern at 50% trace removal. Random removal retains distributed observations, whereas a consecutive gap removes the local support of events within an extended interval. Separating these conditions distinguishes interpolation between available traces from reconstruction across a missing region. Comparisons with conventional and end-toend methods characterize the overall reconstruction method; comparisons with fixed-8 and equal multi-exit training isolate the effect of supervising and coordinating intermediate exits.

To quantitatively validate the convergence efficiency and stability of the proposed meta-supervision mechanism, we evaluate the stage-wise reconstruction accuracy. As illustrated in Fig. 5, we plot the average PSNR evolution trajectories across 8 test shots at each unrolled iteration exit (T1 to T7).

It is evident that the Fixed-depth baseline exhibits sluggish convergence, as it lacks intermediate constraints. The Equal multi-exit strategy, while constrained at every stage, suffers from severe gradient conflicts during the early iterations (T1-T4), leading to suboptimal PSNR gains. Although the

![](images/856e316c5e6ddc844a85d33e14e3f25365e46feeec6317e3a28f159c727da8ab.jpg)  
(a) Missing

![](images/13f5bb08fdf87e8da5f8672384c0d9fa2108781ab7e08476f801f3cda5657559.jpg)  
(b) D3UNet (T1)

![](images/10aadc93de259a480f92d5a5c6df80daa98005a0451103b7bd4d527085ac8f3c.jpg)  
(c) D3UNet (T3)

![](images/9c4da5e511fa7c96ca1e353001d6534c316c04814a0e19d8c99cdf4f2c8592ca.jpg)  
(d) D3UNet (T5)

![](images/8c427a6871106c04d23beb1428b121aac44c09ce6b3486ac147651ecfbf238be.jpg)  
(e) D3UNet (T8 - Final)

![](images/9ad06b231c09ee058494a0203b430dbdb43871db2b52869fc69b359d6d787528.jpg)  
(f) Reference

![](images/92b3257600614648a0a147e1337e119a831968806f9d0b2aa6753eb191104ecd.jpg)  
(g) Missing

![](images/496e911559ed814be8f756ba80138cc52c7f5e08e62ed7f8162dc5360d80fd44.jpg)  
(h) Fixed-depth (T1)

![](images/c441891c28d0b98ee51534a75ef685a6734ef18ced13f446318192874bbe4212.jpg)  
(i) Fixed-depth (T3)

![](images/19f7bf6c98b9cc8ebf3cf08bddbf5ab165f1b96b2fa6a3a74c9bf155e02f3ca7.jpg)  
(j) Fixed-depth (T5)

![](images/d74527fa332419325f57ae705be02d8e8f7b60c0737abc574b449be3f99e4062.jpg)

![](images/06a8bd7c9642e097212c038c5b828184ae34b9185b55b954e8d7d5b8c674fe75.jpg)  
(1) Reference

![](images/4f104132bed093ccbde7a7fbefa0fc0f4ff95a12b20281f16da8647efc8368ac.jpg)  
(m) Missing

(k) Fixed-depth (T7 - Final)  
![](images/4d11d0f89e8972f14484b96fedbe8a80aeaa0d262c074b3b453aebeaf9d00819.jpg)  
(n) MetaPOCS (T1)

![](images/0ab75f04c19ab2e47c0835a8ab1adfd43f6d8b137db402e3a9788255df0833a3.jpg)  
(o) MetaPOCS (T3)

![](images/e7dd4824186d69bf19d9239ff05eb7cfe71c8e438fdcb5de9aa5231d2d51f69e.jpg)  
(p) MetaPOCS (T5)

![](images/08db4d1eed450bd6c377fc464b3be143e79d635c3879dbcad641c13b4f9a3392.jpg)  
nal)  
(q) MetaPOCS (T7 - Fi-

![](images/64cd5fe3294d2173cebf150153dba6127cfde060144643c072447dbd7e1a0f1d.jpg)  
(r) Reference  
Fig. 4. Panoramic view of physical iteration evolution across different unrolling mechanisms and supervision strategies (shot 474). The matrices visualize the step-wise wavefield reconstruction from initial iterations (T1, T3) to the final stage. Notably, traditional fixed-depth schemes (Rows 1-2) exhibit slow convergence, while the Equal multi-exit mechanism (Row 3) introduces severe gradient conflicts, blurring the intermediate wavefields (T3, T5). In contrast, the proposed MetaPOCS (Row 5), driven by dynamic bi-level meta-supervision, resolves early-stage feature entanglements rapidly, achieving unparalleled amplitude fidelity and structural continuity as early as T3, and yields the optimal final reconstruction.

Scalar meta-supervision mitigates this issue to some extent, it lacks spatial adaptivity. In striking contrast, the proposed MetaPOCS, driven by the bi-level dynamic meta-supervision, exhibits a remarkably steep and stable PSNR ascent. It not only resolves the early-stage feature entanglements but also achieves the highest upper bound of reconstruction accuracy at the final stage, corroborating the visual observations and proving the necessity of the proposed adaptive framework.

Relative to equal supervision, the paired full-depth changes for random and consecutive removal are (\*, \*) dB on SEG C3 and (\*, \*) dB on MAVO; their respective confidence intervals are (\*, \*) and (\*, \*). The original-recipe SSI-POCS comparison retains the relationship to the preceding method, while the matched fixed-8 comparison controls the loss normalization and reconstruction architecture.

The waveform comparison in Fig. 4r complements the aggregate scores. Event continuity, arrival-time alignment, and signed amplitude are evaluated on fixed windows selected independently of method performance. Signed residuals distinguish coherent reconstruction errors from diffuse artifacts, while the f-k representation characterizes spectral leakage and event slopes. These views also distinguish visually smooth interpolation from recovery of the measured wavefield structure.

## B. One Checkpoint Across Computational Budgets

Table IV evaluates the same checkpoint at depths 2, 4, 6, and 8. Independently trained fixed-depth models define budget-specialized references. Truncation of the deep-only model measures the quality of intermediate outputs without direct supervision. Equal and meta-trained exits share the same inference architecture, so differences between them reflect training rather than additional deployment modules.

The latency-quality frontier consists of operating points for which no alternative is both faster and more accurate. Equation (9) selects among these profiled outputs for a specified budget. The comparison considers both shallow-exit performance and the deepest-exit change; a gain at low cost can coexist with a loss at full depth. Training time and checkpoint storage separately quantify the cost of learning one multi-exit model or retaining a collection of specialized models.

TABLE III  
RECONSTRUCTION COMPARISON AT 50% TRACE REMOVAL. EACH ENTRY IS MISSING-REGION SNR (DB) / MAE; LARGER SNR AND SMALLER MAE ARE PREFERABLE
<table><tr><td>Method</td><td>SEG C3 random</td><td>SEG C3 consecutive</td><td>MAVO random</td><td>MAVO consecutive</td></tr><tr><td>Curvelet</td><td>20.131/2.366e-02</td><td>1.0578/2.220e-01</td><td>24.600/1.292e-02</td><td>1.0598/2.333e-01</td></tr><tr><td>POCS-Net</td><td>22.486/1.664e-02</td><td>17.810 /3.174e-02</td><td>26.550/9.237e-03</td><td>23.178/1.660e-02</td></tr><tr><td>SP-Net</td><td>22.281/1.236e-02</td><td>15.578/3.875e-02</td><td>26.446/9.444e-03</td><td>19.110 /2.298e-02</td></tr><tr><td>D3UNet</td><td>24.304/1.366e-02</td><td>18.527/2.825e-02</td><td>27.840/8.138e-03</td><td>24.241/1.317e-02</td></tr><tr><td>MetaPOCS</td><td>28.302/8.600e-03</td><td>20.114/2.470e-02</td><td>32.977/5.657e-03</td><td>26.533/1.336e-02</td></tr></table>

TABLE IV

RECONSTRUCTION QUALITY AND COMPUTATIONAL COST ACROSS EXIT DEPTHS. $Q _ { d }$ DENOTES MISSING-REGION SNR (DB). MEMORY AND LATENCY ARE MEASURED ONTHE SAME DEVICE AND INPUT GEOMETRY. INFERENCE LATENCY IS REPORTED AS THE MINIMUM AND MAXIMUM TIME CORRESPONDING TO $T = 1$ AND $T = 7 .$ RESPECTIVELY. TRAINING TIME INCLUDES META OPTIMIZATION.
<table><tr><td rowspan="2">Method</td><td colspan="4">Reconstruction Quality (dB)</td><td colspan="3">Memory (MB)</td><td rowspan="2">Inf. Time (ms)  $( T _ { 1 } ~ / ~ T _ { 7 } )$ </td></tr><tr><td> $Q _ { 1 }$ </td><td> $Q _ { 3 }$ </td><td> $Q _ { 5 }$ </td><td> $Q _ { 7 }$ </td><td>Allocated</td><td>Peak</td><td>Checkpoint</td></tr><tr><td>D3UNet</td><td>3.008</td><td>10.226</td><td>20.672</td><td>26.531</td><td>37.81</td><td>6630</td><td>9.27</td><td>3.783</td></tr><tr><td>Fixed-T</td><td>26.370</td><td>30.206</td><td>32.323</td><td>32.767</td><td>81.3</td><td>19124</td><td>5.90</td><td>1.567  / 8.033</td></tr><tr><td>Deep-only trunc.</td><td>14.275</td><td>19.420</td><td>23.066</td><td>32.202</td><td>81.3</td><td>19124</td><td>5.90</td><td>1.567  / 8.033</td></tr><tr><td>Equal-exit</td><td>24.937</td><td>30.765</td><td>31.630</td><td>31.919</td><td>72.60</td><td>17908</td><td>5.91</td><td>1.617 / 8.033</td></tr><tr><td>Exit-weight BLO</td><td>24.977</td><td>31.073</td><td>31.803</td><td>31.936</td><td>73.58</td><td>23676</td><td>5.91</td><td>1.617 / 8.033</td></tr><tr><td>MetaPOCS</td><td>26.034</td><td>31.968</td><td>32.818</td><td>32.977</td><td>68.45</td><td>17922</td><td>2.08</td><td>1.450 / 8.033</td></tr></table>

![](images/013ccdc7f5c92b935553ab3755553996c9341e778a4d2072aa76b1cc03517dc1.jpg)  
Fig. 5. Quantitative stage-wise PSNR evolution trajectories under different supervision mechanisms (averaged across 8 test shots). The proposed MetaPOCS effectively overcomes the gradient conflicts observed in the Equal multi-exit baseline during early stages (T1-T4) and consistently maintains the highest reconstruction accuracy, demonstrating the superiority of the dynamic bi-level meta-supervision strategy.

## V. OPTIMIZATION ANALYSIS AND ABLATION STUDIES

## A. Gradient Interference and Finite-Update Effects

On the same minibatch and at the same parameter state, let $g _ { k } = \nabla _ { \theta _ { S } } \mathcal { L } _ { k }$ , where $\theta _ { S }$ is the common parameter support of the compared exits. For nonzero gradients,

$$
c _ { k \ell } = \frac { g _ { k } ^ { \top } g _ { \ell } } { \| g _ { k } \| _ { 2 } \| g _ { \ell } \| _ { 2 } } , \quad C = \mathrm { m e a n } _ { k < \ell } { \bf 1 } [ c _ { k \ell } < 0 ] .\tag{27}
$$

The common support excludes parameters absent from either exit's computation graph. Layerwise values localize interfer-

ence within the shared prefix. The gradient norm ratio

$$
R _ { g } = { \frac { \operatorname* { m a x } _ { k } \| g _ { k } \| _ { 2 } } { \operatorname* { m i n } _ { k } \| g _ { k } \| _ { 2 } + \epsilon } }\tag{28}
$$

characterizes scale imbalance independently of directional conflict. Gradients below a fixed numerical tolerance are omitted from cosine averages and counted separately, since their directions are poorly defined.

For $\theta ^ { \prime } = \theta - \eta d ,$ the first-order loss change is $- \eta g _ { k } ^ { \top } d .$ A negative $g _ { k } ^ { \top } d$ therefore predicts a locally harmful update. Positive scalar reweighting preserves the pairwise cosine at a fixed state: cos $( a g _ { k } , b g _ { \ell } ) = \cos ( g _ { k } , g _ { \ell } )$ for $a , b > 0$ It changes the contribution of each gradient to the aggregate direction, whereas tensor-dependent weighting also changes the relative allocation across parameter groups. Training can subsequently move the model to states with different raw gradients. The analysis separates these immediate and trajectory-dependent effects.

Full fusion rounds operate on multi-step displacements. Their first-order diagnostic uses $d = - \Delta _ { \mathrm { f u s e d } }$ , while the finiteupdate effect on a fixed probe batch is

$$
D _ { k } = \mathcal { L } _ { k } \big ( \theta + \Delta _ { \mathrm { f u s e d } } ; \mathcal { D } _ { \mathrm { p r o b e } } \big ) - \mathcal { L } _ { k } \big ( \theta ; \mathcal { D } _ { \mathrm { p r o b e } } \big ) .\tag{29}
$$

The intervention applies learned, uniform, frozen, and shuffled weights to the same base parameters and cached candidate displacements. Shuffling exchanges weights only among supported exits of a tensor, preserving the support constraint. Stochastic layers and normalization statistics remain fixed during the probe. Consequently, changes in $D _ { k }$ arise from the fusion rule rather than different inner trajectories. The actual-update comparison is complemented by a commonnorm comparison to distinguish direction from step magnitude.

Table I relates optimization diagnostics to exit quality. The learned-minus-uniform change in harmful-update frequency is \*, and the paired shallow/deep reconstruction changes are $* / *$ dB. The uniform-fusion comparison holds the expensive inner trajectories fixed, isolating whether the outer objective identifies a more favorable mixture. Gradient-scale balance, finite-update harm, and reconstruction quality remain separate response variables: an improvement in one does not imply an improvement in the others.

![](images/cd3b700ee51bd57e3096d5f90eaa87819a5e141c115ab840f5e1011871ab29e2.jpg)  
(a) Reference

![](images/c48c965b7293406e5c3125b91965376b2f696ee2e79476674e076f8ed76630ac.jpg)  
(b) Curvelet

![](images/0ee4293294cbc4255e65f98f881ccc5993ecdcfa2b7989ef3b1c3850c1f0d420.jpg)  
(c) POCSNet

![](images/02f113b7709ce1ac121e354b056c486fc3e4aa839d779074114d49797ea49990.jpg)  
(d) Missing

![](images/187396f52d2320b1dad229155ecedccd47c184aee697f317934eb97a8b22048c.jpg)  
(e) SPNet

![](images/c21969f53f31d6e145c9a3c40ad633f442ce508d07dbbff9e0dcee5bf49282db.jpg)  
(f) D3UNet

![](images/cd53316d87425c5402bb9abb6e6c83c7dba7e3f4e0d85184120e9a57b45d942d.jpg)  
(g) Ours  
Fig. 6. Qualitative comparison of reconstruction results on the SEG C3 dataset under 50% random trace removal (shot 12\_4). We restore the seismic data to its original amplitude range.

Learning curves use both task-update count and elapsed GPU time as their horizontal axes. Time to a fixed validationquality threshold characterizes convergence efficiency; the proportion of runs reaching that threshold characterizes reliability. Similar training error with lower held-out error is instead consistent with a generalization effect. These distinctions connect the hypergradient in Eq. (23) to observable training behavior without requiring every pair of raw gradients to become aligned.

## B. Interaction With Reconstruction Priors and Losses

The factorial comparison in Table V separates the training rule from the reconstruction prior. Each backbone/loss configuration appears under both equal supervision and meta fusion. For quality Q, the interaction

$$
I = [ Q _ { \mathrm { m e t a , d u a l } } - Q _ { \mathrm { e q u a l , d u a l } } ] - [ Q _ { \mathrm { m e t a , s p a t i a l } } - Q _ { \mathrm { e q u a l , s p a t i a l } } ]\tag{30}
$$

measures how the meta-learning effect changes between a dual-domain and spatial-only prior. The paired confidence interval for I preserves the matched acquisition units and seeds of the four cells. Concatenation and DFM distinguish feature exchange from update fusion. Crossing spatial-only and dualdomain priors with amplitude-only and amplitude-frequency supervision isolates the roles of implicit representation and explicit reconstruction constraints; transform regularization remains fixed. The absolute scores identify the effect of adding

FFL to each prior, and the paired meta-minus-equal differences identify its interaction with the training rule. A plain POCS-Net backbone supplies an additional transfer condition. The effect of a prior alone and its interaction with meta training are consequently estimated separately.

## C. Meta-Optimization Sensitivity

The sensitivity analysis varies one factor around the reference configuration while retaining the same data partition and reconstruction objective. The auxiliary coefficient $\beta$ controls the distinction among exit-oriented tasks; $\mu$ controls the temporal smoothing of fusion logits. The refresh period changes both optimization dynamics and training cost. Fusion granularity and support are compared using common candidate trajectories, while meta-data-source comparisons retain a separate monitoring role. Scalar meta-supervision has its own floor parameter $\rho$ and is reported as a distinct training family.

## D. Acquisition Shift and Failure Cases

Table I separates in-distribution interpolation from acquisition and geological transfer. In mask-transfer experiments, a model trained with random removal is evaluated on consecutive or periodic removal without fine-tuning. Density transfer changes the missing fraction while retaining the same checkpoint. Wavelet and velocity-model transfer use independent synthetic conditions, and survey transfer changes the acquisition source. These tests concern missing traces on the documented grid; off-grid coordinate regularization is a different sampling problem.

Failure analysis focuses on gap centers, weak reflections, steep dips, and crossing events. At a fixed missing fraction, gap width changes the distance between an unobserved event and its available boundary information. Weak-event windows further separate amplitude recovery from scores dominated by strong reflectors. In noisy-input experiments, the hard consistency operator retains the actual noisy measurements; the resulting acquired-sample error is therefore evaluated independently from interpolation in missing traces.

![](images/ed3ffb9ef6f68208bf06a80bc80494fdae725cf7438d8986b43a0bfe0c07f76a.jpg)

![](images/92e1135e1de75993e9236285900c357f4dc3bf056bd3363091c672342b3ace34.jpg)  
(b) Curvelet

![](images/303d7163cf5bd7ca4a3da86228d0a5454762e84e3bf2caecd194057c640c40c3.jpg)  
(c) POCSNet

![](images/924e62aa667ef26a018c1c22115d41f3f6f796585b703e43bac4a83f2fc421d5.jpg)  
(d) Missing

![](images/0eb508f282cddedb873bcee830cf8a087ff2b5fd57d4eed9d5c02c10ebaac595.jpg)

![](images/89e4a6fc8f40b0121c1846797ddd8a2f1b0835b5ef3e3dbd4dcef906f60e4e5f.jpg)  
(f) D3UNet

![](images/0fc7ab914a464c02e708e5e56be170101ca50282ed901f40e012ec163ced1687.jpg)  
(g) Ours  
Fig. 7. Qualitative comparison of F-K spectrum reconstruction results on the SEG C3 dataset under 50% random trace removal (shot $1 2 \_ 4 )$ We restore the seismic data to its original amplitude range.

## VI. DOWNSTREAM REVERSE-TIME-MIGRATION VALIDATION

## A. Controlled Imaging Experiment

RTM provides a downstream test of whether interpolation preserves wavefield information relevant to subsurface imaging. The controlled setting uses two-dimensional acoustic data with known source and receiver coordinates. In a constantdensity medium, the source wavefield satisfies

$$
\frac { 1 } { \nu ^ { 2 } ( \mathbf { r } ) } \frac { \partial ^ { 2 } u _ { s } } { \partial t ^ { 2 } } - \nabla ^ { 2 } u _ { s } = q _ { s } ( \mathbf { r } , t ) ,\tag{31}
$$

where r is subsurface position, ν is velocity, and $q _ { s }$ is the source term. The numerical formulation follows finitedifference wave propagation and reverse-time imaging, for which Devito provides a reproducible implementation framework [45].

Layered, faulted, and laterally heterogeneous models define distinct structural conditions. Entire velocity models form the independent test units. Complete gathers are generated with the true model, whereas migration uses a smooth background velocity $\nu _ { b }$ shared by all interpolation methods. Grid spacing, source wavelet, boundary treatment, and temporal sampling determine the modeled bandwidth and numerical dispersion. The imaging comparison fixes these quantities, with discretization and background-velocity sensitivity treated as separate conditions.

The input groups in Table I are complete data, zero-filled incomplete data, conventional interpolation, SSI-POCS, equal multi-exit reconstruction, and MetaPOCS. Multi-exit inputs cover depths 2, 4, 6, and 8. Direct-arrival treatment, bandpass, migration aperture, and illumination compensation are common to all groups. Interpolated data enter migration after reversal of the reconstruction normalization, preserving their relative amplitudes.

Let $\boldsymbol { \mathcal { M } } _ { \nu _ { b } }$ be the fixed migration operator using background

![](images/f560aca3863a89b93ca204a44d57d8d58da9e653355d4f98b5f9a0bc3bfd67b7.jpg)  
(a) Reference

![](images/bf10cfd8a0d0238ff7226bf6ae13d504dd3cc3d76e4538297791c844a2a89991.jpg)  
(b) Curvelet

![](images/7485404ae581aac8899d028afc97962495b62a47a5b5a4797be792a4ca866450.jpg)  
(c) POCSNet

![](images/171a1abf87ff22672fa2d093fd6ba9746f6dc5c52d4f0ffab3a530bad29e8138.jpg)  
(d) Missing

![](images/6bf755ed1a5b8ab141f78c2a170eac2f9b6a6d934f8660fccfe913434c0f847b.jpg)  
(e) SPNet

![](images/d7ac57920bf2d821de0cdd569c92e17e4be3e47cd2840e024e8d470e756d8e82.jpg)  
(f) D3UNet

![](images/b624849edf472ae22d300c92d6d8ba35015bf8f285b39588301fee582fb1661f.jpg)  
(g) Ours  
Fig. 8. Qualitative comparison of reconstruction results on the MAVO dataset under 50% trace removal. (a) Random undersampling (shot 180). (b) Consecutive large gaps (shot 697). We restore the seismic data to its original amplitude range.

TABLE V  
INTERACTION BETWEEN IMPLICIT RECONSTRUCTION PRIORS, EXPLICIT RECONSTRUCTION OBJECTIVES, AND TRAINING. EQUAL AND META COLUMNS REPORT $Q _ { 2 } / Q _ { 8 }$ INDB; THE FINAL COLUMN GIVES THEIR PAIRED DIFFERENCE. TRANSFORM REGULARIZATION IS RETAINED IN EVERY MATCHED CONFIGURATION
<table><tr><td>Prior</td><td>Loss recipe</td><td>MAE↓</td><td>SSIM↑</td><td>SNR↑</td><td>Improvement</td></tr><tr><td>Spa-Spa, concat</td><td> $L _ { 1 }$ </td><td>8.063e-03</td><td>0.941</td><td>28.587</td><td>√</td></tr><tr><td>FFC–FFC, concat</td><td> $L _ { 1 }$ </td><td>8.267e-03</td><td>0.943</td><td>28.588</td><td>√</td></tr><tr><td>Spa-FFC, concat</td><td> $L _ { 1 }$ </td><td>8.239e-03</td><td>0.946</td><td>29.549</td><td>√</td></tr><tr><td>Spa–FFC, DFM</td><td> $L _ { 1 }$ </td><td>6.862e-03</td><td>0.956</td><td>30.831</td><td>√</td></tr><tr><td>Spa-FFC, DFM</td><td> $L _ { 1 } + \mathrm { F F L }$ </td><td>6.904e-03</td><td>0.957</td><td>31.049</td><td>√</td></tr><tr><td>Spa-FFC, DFM</td><td> $L _ { 1 } + \mathrm { s y m } + \mathrm { F F L }$ </td><td>6.427e-03</td><td>0.959</td><td>31.463</td><td>√</td></tr></table>

TABLE VI

GENERALIZATION COMPARISON OF DIFFERENT MODELS ON THE TEST SET OF THE MODEL94 DATASET (MISSING RATE: 50%).
<table><tr><td>Type</td><td>Model</td><td>MAE↓</td><td>SNR↑</td><td>SSIM↑</td><td>PSNR↑</td></tr><tr><td rowspan="3">SEGC3 to Model94: Random missing</td><td>D3UNet</td><td>0.0105</td><td>25.144</td><td>0.9501</td><td>31.712</td></tr><tr><td>Ours (T=7)</td><td>0.0081</td><td>26.788</td><td>0.9645</td><td>33.499</td></tr><tr><td>Ours (Best T=6)</td><td>0.0083</td><td>26.875</td><td>0.9645</td><td>33.586</td></tr><tr><td rowspan="3">MAVG to Model94: Random missing</td><td>D3UNet</td><td>0.0231</td><td>17.934</td><td>0.8136</td><td>24.502</td></tr><tr><td>Ours (T=7)</td><td>0.0149</td><td>21.057</td><td>0.9073</td><td>27.768</td></tr><tr><td>Ours (Best T=6)</td><td>0.0141</td><td>21.227</td><td>0.9073</td><td>27.938</td></tr><tr><td rowspan="3">MAVG to Model94: Consecutive missing</td><td>D3UNet</td><td>0.0593</td><td>13.331</td><td>0.7845</td><td>19.899</td></tr><tr><td>Ours (T=7)</td><td>0.0492</td><td>14.717</td><td>0.7936</td><td>21.428</td></tr><tr><td>Ours (Best T=2)</td><td>0.0480</td><td>14.891</td><td>0.7936</td><td>21.602</td></tr><tr><td rowspan="3">SEGC3 to Model94: Consecutive missing</td><td>D3UNet</td><td>0.0766</td><td>11.269</td><td>0.7638</td><td>17.837</td></tr><tr><td>Ours (T=7)</td><td>0.1306</td><td>6.545</td><td>0.7621</td><td>13.257</td></tr><tr><td>Ours (Best T=2)</td><td>0.0715</td><td>12.027</td><td>0.7621</td><td>18.739</td></tr></table>

velocity $\nu _ { b } \colon$

$$
I _ { \mathrm { r e f } } = M _ { \nu _ { b } } ( d _ { \mathrm { f u l l } } ) , \qquad I _ { a } = M _ { \nu _ { b } } ( \hat { d } _ { a } ) .\tag{32}
$$

For example, a correlation imaging condition has the form

$$
I _ { a } ( \mathbf { r } ) = \sum _ { s } \int _ { 0 } ^ { t _ { \operatorname* { m a x } } } u _ { s } ( \mathbf { r } , t ; \nu _ { b } ) b _ { s , a } ( \mathbf { r } , t ; \nu _ { b } ) d t ,\tag{33}
$$

where $b _ { s , a }$ is the receiver wavefield back-propagated from the reconstructed data. With a fixed source wavefield and linear receiver propagation, the image depends linearly on its data input. Accordingly, fixed linear preprocessing and migration give

$$
I _ { a } - I _ { \mathrm { r e f } } = M _ { \nu _ { b } } ( \hat { d } _ { a } - d _ { \mathrm { f u l l } } ) .\tag{34}
$$

The complete-data image is the band-limited reference associated with the same imaging operator. Data-space error and image-space error weight interpolation residuals differently: two residual fields with similar energy can produce different coherent artifacts after migration. Therefore, the downstream analysis evaluates imaging error directly rather than treating data SNR as a surrogate.

![](images/4b3ea1f9766eea5f147c8b807ed0b0c91146e4190b17d81954814468b8d0c4b3.jpg)  
(a) Reference

![](images/a0f9c00ebc19e0313fee1dc39e203a1f3bfc77a709d8ef51c3a1d09703734379.jpg)  
(b) Curvelet

![](images/4f18d7c93a97ec5a26424ebb56feb71119aa4ae5907113c9ff23815b4fad82d3.jpg)  
(c) POCSNet

![](images/4d017c518649f69d3aa9d01e87e8ac45d69641d8429692d8c56d2b08f3e2d3e1.jpg)  
(d) Missing

![](images/981969b9d96e1c0dd37342cf7b1bd445fe47473e02453590b0f1e9ef165a55e2.jpg)  
(e) SPNet

![](images/cfab29215c10bfd9d1f0266c2f3b2316a390162f0ec95b8e7f8a99045fce47ee.jpg)  
(f) D3UNet

![](images/7554feb773c42cdd77f2ee470326c4d7d7e7e930a74ca49156ca6755fa49a3b3.jpg)  
(g) Ours  
Fig. 9. Qualitative comparison of F-K spectrum reconstruction results on the MAVO dataset under 50% trace removal. (a) Random undersampling (shot 180). (b) Consecutive large gaps (shot 697). We restore the seismic data to its original amplitude range.

## B. Image-Domain Accuracy

The relative image error over a fixed subsurface window outside the absorbing boundaries is

$$
E _ { I } = \Vert I _ { a } - I _ { \mathrm { r e f } } \Vert _ { F } / \Vert I _ { \mathrm { r e f } } \Vert _ { F } .\tag{35}
$$

This measure is complemented by image correlation, reflectorposition error in meters, and artifact energy within prescribed nonreflector regions. The synthetic model supplies structural interface locations, while $I _ { \mathrm { r e f } }$ supplies the corresponding bandlimited image amplitudes. Paired uncertainty is computed across independent velocity models, retaining the shots and masks of each model within the same resampled unit.

At each reconstruction budget, Table I relates missing-trace SNR to $E _ { I }$ , reflector localization, and artifact energy. The paired MetaPOCS-minus-equal change in image error is \*, and the corresponding reflector-position change is \* m. Figure 4r provides the associated image residuals and fixed reflector profiles. Agreement between data- and image-domain changes supports downstream utility within this acoustic setting; disagreement identifies interpolation errors to which the imaging operator is particularly sensitive.

![](images/845bdc09d716759816037e4307073ac988dd89f507014db161c96cc4ddfce226.jpg)  
(a) Reference

![](images/ef11e5a1d1968ab8bde082b470c487202adce84d3d2ccdc6fe6d025e6bbda6fb.jpg)  
(b) Curvelet

![](images/fa0af7558b6313e5958c4d6549fcf3b5f92db6c3058a132ff7f77c174d5d7e2b.jpg)  
(c) POCSNet

![](images/1013c1e000600a5aedd8b1e9340ebc66536f46f2b846a66d53d1de712f0b2a30.jpg)  
(d) Missing

![](images/8786afd51f2d4d10f83a28148e5218734ded3ab6121964884aedacf3661f1977.jpg)  
(e) SPNet

![](images/effb8d02df6d2f5c7fe9fd9e6a93bdecb5dc69a3945bdcf0cfce63fa750ea923.jpg)  
(f) D3UNet

![](images/fc6a1330d68604b999d611c935d1b05ba74cccad6c23ca551c133233bfe44288.jpg)  
(g) Ours  
Fig. 10. Qualitative comparison of reconstruction results on the MAVO dataset under 50% trace removal. (a) Random undersampling (shot 180). (b) Consecutive large gaps (shot 488). We restore the seismic data to its original amplitude range.

Method Efficiency Analysis  
![](images/efde8733194df5fb1881a59bb9a2ab35697acf48eaddd658d309c69ed4121dfe.jpg)  
Fig. 11. Pareto efficiency analysis (PSNR vs. Parameters). The proposed method optimally occupies the top-left Pareto frontier, achieving the highest reconstruction fidelity with minimal parameter overhead.

## VII. DISCUSSION AND LIMITATIONS

MetaPOCS separates reconstruction representation, training allocation, and deployment cost. The spatial-spectral prior supplies the representation inherited from SSI-POCS. Supervised exits make several intermediate wavefields explicit optimization targets. Meta-GF-based fusion then allocates exitoriented updates across the shared parameter tensors, while the latency policy selects a deployment budget. This separation explains why a feature-fusion improvement, an optimization improvement, and a computational saving require different comparisons.

The fusion derivative also clarifies the role of the outer objective. A candidate update is assessed relative to the current mixture, using the sensitivity of the equal-exit reconstruction loss. This permits tensor-dependent preferences without requiring a globally preferred exit. Nevertheless, local hypergradient information does not imply global convergence or simultaneous improvement of every exit. The probe analysis quantifies finite-step effects, and the full learning curves establish whether those effects persist over optimization.

Several limitations remain intrinsic to the formulation. Full fusion rounds require additional inner trajectories, which may offset the savings obtained during deployment when few gathers are processed. Budget selection is device and input dependent, and it does not enforce monotonic reconstruction quality with depth. Hard reinsertion preserves measured amplitudes but also retains measurement noise; wide gaps remain underdetermined by acquisition constraints alone. Finally, a controlled acoustic imaging experiment characterizes one downstream operator and does not establish elastic or AVO fidelity. These limitations motivate more economical fusion schedules, uncertainty-aware budget selection, and extension to richer acquisition geometries.

## VIII. CONCLUSION

We have presented MetaPOCS, a multi-exit unfolding formulation for seismic interpolation under varying computational budgets. The framework combines the SSI-POCS reconstruction prior with supervised intermediate outputs and Meta-GF-based fusion of exit-oriented parameter displacements. Exact acquisition consistency is maintained at every exit, and a calibrated prefix policy supports several deployment operating points from one checkpoint. The derived fusion hypergradient explains how tensor-wise weights respond to candidate updates relative to their current mixture. The evaluation formulation connects this training mechanism to matched-budget waveform reconstruction and fixed-operator migration error. Together, these components establish a coherent basis for assessing the accuracy, optimization cost, and downstream value of budgetaware seismic unfolding.

## DATA AND CODE AVAILABILITY

The implementation, configurations, acquisition partitions, and evaluation scripts are associated with the DAP project. The public repository or archival identifier is \*. Dataset access identifiers and licenses are \*; checkpoint identifiers and the reproducible environment specification are \*.

## ACKNOWLEDGMENT

The authors acknowledge the seismic data providers. Dataset-specific attribution will be finalized after source and license verification. OpenAI Codex assisted with manuscript restructuring, language drafting, and experimental-plan preparation; the authors are responsible for verifying scientific content, references, and final results.

## REFERENCES

[1] J. RONEN, "Wave-equation trace interpolation," GEOPHYSICS, vol. 52, no. 7, pp. 973–984, JUL 1987.

[2] A. Canning and G. Gardner, “Regularizing 3-d data sets with dmo," GEOPHYSICS, vol. 61, no. 4, pp. 1103–1114, JUL-AUG 1996.

[3] S. Verma, S. Guo, and K. J. Marfurt, "Data conditioning of legacy seismic using migration-driven 5d interpolation,"INTERPRETATION-A JOURNAL OF SUBSURFACE CHARAČTERIZATION, vol. 4, no. 2, pp. SG31–SG40, MAY 2016.

[4] C. ShengChang and Z. HuaMin, "Least squares migration of seismic data with reflection wave equation," Chinese Journal of Geophysics, vol. 61, no. 4, pp. 1413–1420, 2018.

[5] S. Spitz, "Seismic trace interpolation in the fx domain," Geophysics, vol. 56, no. 6, pp. 785–794, 1991.

[6] Y. Chen, S. Fomel, H. Wang, and S. Zu, “5d dealiased seismic data interpolation using nonstationary prediction-error filter," GEOPHYSICS, vol. 86, no. 5, pp. V419–V429, SEP-OCT 2021.

[7] Y. Liu, G. Wu, and Z. Zheng, "Seismic data interpolation without iteration using a t-x-y streaming prediction filter with varying smoothness," GEOPHYSICS, vol. 87, no. 1, pp. V29–V38, JAN-FEB 2022.

[8] Y. Xu, S. Yu, L. Dong, and J. Ma, "Dealiased seismic data interpolation by dynamic matching," Geophysics, vol. 89, no. 5, pp. V361–V376, 2024.

[9] E. J. Candès and M. B. Wakin, “An introduction to compressive sampling," IEEE signal processing magazine, vol. 25, no. 2, pp. 21– 30, 2008.

[10] P. Zwartjes and M. Sacchi, “Fourier reconstruction of nonuniformly sampled, aliased seismic data," Geophysics, vol. 72, no. 1, pp. V21– V32, 2007.

[11] S. Xu, Y. Zhang, D. Pham, and G. Lambaré, "Antileakage fourier transform for seismic data regularization," Geophysics, vol. 70, no. 4, pp. V87–V95, 2005.

[12] S. Xu, Y. Zhang, and G. Lambaré, "Antileakage fourier transform for seismic data regularization in higher dimensions," Geophysics, vol. 75, no. 6, pp. WB113–WB120, 2010.

[13] E. Candes, L. Demanet, D. Donoho, and L. Ying, “"Fast discrete curvelet transforms," multiscale modeling & simulation, vol. 5, no. 3, pp. 861– 899, 2006.

[14] S. Fomel and Y. Liu, “Seislet transform and seislet frame," Geophysics, vol. 75, no. 3, pp. V25–V38, 2010.

[15] H. Zhang, X.-H. Chen, and L.-Y. Zhang, “3d simultaneous seismic data reconstruction and noise suppression based on the curvelet transform," Applied Geophysics, vol. 14, no. 1, pp. 87–95, 2017.

[16] J. Wang, M. Ng, and M. Perz, "Seismic data interpolation by greedy local radon transform," Geophysics, vol. 75, no. 6, pp. WB225–WB234, 2010.

[17] X. Gong, S. Wang, and L. Du, "Seismic data reconstruction using a sparsity-promoting apex shifted hyperbolic radon-curvelet transform," Studia Geophysica et Geodaetica, vol. 62, no. 3, pp. 450–465, 2018.

[18] S. Yu and J. Ma, “Simultaneous off-the-grid regularization and reconstruction for 3d seismic data by a new combined sampling operator," Geophysics, vol. 88, no. 4, pp. V291–V302, 2023.

[19] B. Wang, N. Zhang, W. Lu, P. Zhang, and J. Geng, "Seismic data interpolation using deep learning based residual networks," in 80th EAGE Conference and Exhibition 2018, vol. 2018, no. 1. European Association of Geoscientists & Engineers, 2018, pp. 1–5.

[20] W. Fang, L. Fu, M. Zhang, and Z. Li, "Seismic data interpolation based on u-net with texture loss," Geophysics, vol. 86, no. 1, pp. V41–V54, 2021.

[21] H. Kaur, N. Pham, and S. Fomel, “Seismic data interpolation using deep learning with generative adversarial networks," Geophysical Prospecting, vol. 69, no. 2, pp. 307–326, 2021.

[22] H. Park, J.-W. Lee, J. Hwang, and D.-J. Min, "Coarse-refine network with upsampling techniques and fourier loss for the reconstruction of missing seismic data," IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2022.

[23] N. Liu, L. Wu, J. Wang, H. Wu, J. Gao, and D. Wang, "Seismic data reconstruction via wavelet-based residual deep learning," IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–13, 2022.

[24] L. Gao, H. Shen, and F. Min, "Swin transformer for simultaneous denoising and interpolation of seismic data," Computers & Geosciences, vol. 183, p. 105510, 2024.

[25] Y. Mu, C. Wang, X. Geng, C. Zhang, J. Zhang, and J. Jia, "Seismic data reconstruction via an adaptive feature fusion network,"Engineering Applications of Artificial Intelligence, vol. 160, p. 111982, 2025.

[26] C. Meng, J. Gao, Y. Tian, H. Chen, W. Zhang, and R. Luo, "Stochastic solutions for simultaneous seismic data denoising and reconstruction via score-based generative models," IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[27] K. Li, J. Ma, Y. Dou, Y. Zhao, Z. Yang, and Z. Li, “Seismic denoising diffusion restoration model for seismic data processing,"Engineering Applications of Artificial Intelligence, vol. 156, p. 111068, 2025.

[28] T. Mo, J. Song, Z. Li, and B. Wang, "Deep learning-based off-the-grid seismic data reconstruction and regularization: Preliminary research," Geophysics, vol. 90, no. 1, pp. V1–V12, 2025.

[29] H. Zhang, X. Yang, and J. Ma, "Can learning from natural image denoising be used for seismic data interpolation?" Geophysics, vol. 85, no. 4, pp. WA115–WA136, 2020.

[30] Y. Chen, S. Yu, and J. Ma, “A projection-onto-convex-sets network for 3d seismic data interpolation," Geophysics, vol. 88, no. 3, pp. V249– V265, 2023.

[31] M. Wu, L. Fu, W. Fang, and J. Cao, “Sparse prior-net: A sparse priorbased deep network for seismic data interpolation," Geophysics, vol. 89, no. 1, pp. V37–V47, 2024.

[32] B. Jiang, C. Zhang, A. Di, X. Wei, X. Zhao, D. Xiong, and C. Meng, “Spatial-spectral interactive POCS network for large-gap seismic data interpolation," 2026, manuscript submitted to IEEE Geoscience and Remote Sensing Letters; unpublished.

[33] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu, "Deeplysupervised nets," in Proc. International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, vol. 38, 2015, pp. 562–570. [Online]. Available: https: //proceedings.mlr.press/v38/lee15a.html

[34] G. Huang, D. Chen, T. Li, F. Wu, L. van der Maaten, and K. Q. Weinberger, "Multi-scale dense networks for resource efficient image classification," in Proc. International Conference on Learning Representations, 2018. [Online]. Available: https://arxiv.org/abs/1703. 09844

[35] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, "GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks," in Proc. International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 794–803. [Online]. Available: https://proceedings.mlr.press/v80/chen18a.html

[36] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, "Gradient surgery for multi-task learning," in Advances in Neural Information Processing Systems, vol. 33, 2020. [Online]. Available: https://arxiv.org/abs/2001.06782

[37] M. Ren, W. Zeng, B. Yang, and R. Urtasun, "Learning to reweight examples for robust deep learning," in Proc. International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80, 2018, pp. 4334–4343. [Online]. Available: https://proceedings.mlr.press/v80/ren18a.html

[38] Y. Sun, J. Li, and X. Xu, "Meta-GF: Training dynamic-depth neural networks harmoniously," in Proc. European Conference on Computer Vision, 2022. [Online]. Available: https://www.ecva.net/papers/eccv\_ 2022/papers\_ECCV/html/6337\_ECCV\_2022\_paper.php

[39] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration," in 2022 IEEE/CVF CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR 2022), ser. IEEE Conference on Computer Vision and Pattern Recognition. IEEE; CVF; IEEE Comp Soc, 2022, pp. 5718–5729, iEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, JUN 18-24, 2022.

[40] L. Chi, B. Jiang, and Y. Mu, "Fast fourier convolution," Advances in Neural Information Processing Systems, vol. 33, pp. 4479–4488, 2020.

[41] Z. Wang, X. Gong, Z. Ma, G. Peng, and Y. Long, "Dual-domain deep unfolding network with implicit regularization for seismic data reconstruction," IEEE Transactions on Geoscience and Remote Sensing, vol. 64, 2026.

[42] L. Jiang, B. Dai, W. Wu, and C. C. Loy, "Focal frequency loss for image reconstruction and synthesis," in 2021 IEEE/CVF INTERNATIONAL CONFERENCE ON COMPUTER VISION (ICCV 2021). IEEE; CVF; IEEE Comp Soc, 2021, pp. 13 899–13 909, 18th IEEE/CVF International Conference on Computer Vision (ICCV), ELECTR NETWORK, OCT 11-17, 2021.

[43] S. K. Choe, W. Neiswanger, P. Xie, and E. Xing, “Betty: An automatic differentiation library for multilevel optimization," in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=LV\_MeMS38Q9

[44] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[45] M. Louboutin, M. Lange, F. Luporini, N. Kukreja, P. A. Witte, F. J. Herrmann, P. Velesko, and G. J. Gorman, "Devito (v3.1.0): an embedded domain-specific language for finite differences and geophysical exploration," Geoscientific Model Development, vol. 12, pp. 1165–1187, 2019. [Online]. Available: https://gmd.copernicus.org/ articles/12/1165/2019/