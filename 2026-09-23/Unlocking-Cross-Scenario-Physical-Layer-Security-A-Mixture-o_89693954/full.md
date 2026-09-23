# Unlocking Cross-Scenario Physical Layer Security: A Mixture-of-Experts Framework with Generative Diffusion Models

Xiao Tang, Tong Hui, Chao Shen, Yichen Wang, Qinghe Du, Li Sun, and Zhu Han

Abstract—The future 6G networks are expected to incorporate a proliferation of wireless services in diverse environments, which presents a significant challenge for information security. Conventionally optimization always requires recalculation and learning strategy often suffers poor generalization, which are thus incapable for the security provisioning with wide scenario coverage. In this paper, we propose an adaptive and robust learning framework that leverages a mixture-of-experts (MoE) architecture to achieve cross-scenario physical layer security guarantee. Specifically, we first select a few representative scenarios and establish the scenario-specific generative diffusion model (GDM)-based experts for secure transmission beamforming with artificial noise. The diffusion nature of experts learns the overall probability distribution of security strategy solution landscape and the Transformer-based denoising process enhances the ability to generalize across varying network configurations. Then, a lightweight gating network is constructed to identify the scenarios by engineering the channel features and select the most relevant experts. Finally, an attention-based combiner is introduced to synthesize the security proposals from the toprated experts to produce a high-fidelity security strategy to cover the unseen scenarios. Simulation results demonstrate that the proposed GDM-based MoE framework can accurately recognize the scenarios and properly select the experts, maintaining nearoptimal secrecy rates across a continuum of wireless scenarios and outperforming traditional single-model paradigms.

Index Terms—Physical layer security, cross-scenario, mixtureof-experts, generative diffusion model, Transformer

## I. INTRODUCTION

Information security is the fundamental and pervasive requirement for wireless communications, which is yet challenged by various malicious adversaries. The rapid development of 6G networks has brought a proliferation of wireless devices and services, which demand reliable and diverse security provisioning [1]. Despite the indispensability of upperlayer cryptography-based approaches, the overhead induced by the key management and distribution can be prohibitive for the vast amount of wireless devices which are usually of limited resources and capabilities [2]. Consequently, the physical layer security with keyless operations facilitates a powerful complement, enabling the lower-layer defense by exploiting the inherent randomness of the wireless medium [3]. With secrecy enhancement implemented in the air, the physical layer security is capable for ubiquitous security guarantee across different devices and applications. This advantage aligns with vision of future 6G to cover diverse services and scenarios, making physical layer security a fascinating solution towards a safeguarded 6G era [4].

For the extensive and continued research efforts on physical layer security, the majority are based on optimizations which are usually burdened with complex iterations and recalculations in changing circumstances [5]. Recently, the deep learning-based security designs, including convolutional neural networks, graph neural networks, etc., offer an alternative and promising path forward and thus are emerging rapidly [6]. Although these approaches feature prompt inferences with trained neural networks, they are fundamentally discriminative in nature that approximate a single-point estimation, making the model potentially brittle for unseen input [7]. Given the rapidly changing wireless environment, even slightly channel variations not encountered during training may induce significant security performance degradation [8]. Moreover, the complex relationship between the security strategy and performance leads to potentially vast and complicated space of the (near-)optimal solutions, which is rather difficult to approximate under the discriminative models [9].

Consequently, the recent work resorts to the generative models to overcome such deficiencies, where the generative diffusion model (GDM) is particularly prominent for its highquality and controllable generation capability [10], [11]. Unlike discriminative models that establish a direct and deterministic mapping, the GDMs learn the entire conditional probability distribution between the wireless environment and the optimal security strategy. In this regard, GDMs can capture the complex structure of the overall solution landscape, producing diverse and high-fidelity candidate solutions to tackle the random fading environments [12]. This capability is critical for physical layer security, where the system performance is challenged by the constant channel fluctuations and unpredictable adversary behavior. In this context, GDMs are promising to produce highly robust and flexible security countermeasures to achieve resilient and adaptive security protection.

When considered further, even a powerful generative model is significantly challenged when confronted with large dynamic of wireless networks, particularly for 6G communications with a variety of emerging scenarios demanding reliable security provisioning [13]. In this regard, despite the generative capability, a single GDM is still fundamentally limited by the training data statistics, as a model explicitly trained over a specified data set associated with one scenario can hardly maintain the performance when the scenario characteristics become different [14]. For example, the security loss is almost inevitable when a GDM-based security solution initially trained for urban settings is reused in suburban environments. To fill the gap between the inherent limitation of static-model issue and the cross-scenario applications in 6G, a higherlevel architecture is imperative to enable coverage to diverse scenarios with adaptive security guarantee [15].

Towards the orchestrated expertise of specialized models to achieve cross-scenario secure transmissions, we investigate the multi-user multiple-input single-output (MISO) communications in the presence of eavesdroppers under various scenarios, and propose a mixture-of-experts (MoE) framework with GDM-based security solutions. Specifically, the main contributions are summarized as follows:

• We consider the multi-user MISO physical layer secure communication system across a continuum of wireless scenarios, where the propagation environments are quantitatively characterized by the path loss exponent and Rician factor. Accordingly, we formulate the problem to develop a unified security solution that is effective across different scenarios, rather than conventional environmentspecific approaches.

• To enable scalable cross-scenario security provisioning, we propose a MoE-based architecture that orchestrates a pool of scenario-specialized security experts with scenario-aware adaptation. In this regard, the expert modeling adopts a conditional generative formulation to capture the solution landscape of secure beamforming, and a two-stage training paradigm is developed to combine stable supervised diffusion learning with secrecy-rateoriented fine-tuning.

• We introduce a lightweight router as the higher-level scenario recognizer, where a fixed-dimensional scenario feature is engineered from high-dimensional channel observations. The extracted features keep sensitive to propagation characteristics yet applicable under varying network configurations, based on which the top-rated experts are activated to establish the practical MoE workflow for cross-scenario operation.

• To effectively cover the unseen cross-scenarios, we further design an attention-based combiner to process the expert output. The attention mechanism naturally handles a variable number of activated experts, and a residual correction is learned to capture the non-linear mapping from the weighted expert proposals to a high-fidelity security strategy.

• Through extensive simulations, we validate that the proposed architecture maintains near-optimal secrecy rates across varying network settings and diverse scenarios.

We also demonstrate reliable scenario recognition by the router, and show that the attention-based residual synthesis consistently improves cross-scenario security performance compared with straightforward weighted expert averaging.

The remainder of this paper is organized as follows. Sec. II reviews the related work. Sec. III details the system model and problem formulation. Sec. IV presents the individual GDM experts for specified scenarios. Sec. V describes the design of the MoE framework, including the router and the combiner, on the basis of GDM experts. Sec. VI provides comprehensive simulation results, and Sec. VII concludes this paper.

## II. RELATED WORK

Attracted by the advantages of physical layer security, the research into this issue has been extensive, where primary methods involve the design of security-aware signals and transmission strategies for secrecy enhancement [5]. In this regard, the dominant techniques are the secure beamforming and artificial noise (AN) injection, where the former mainly aims to align with the legitimate direction and letter downgrades the unintended reception [16]. Recently, these classical approaches are jointly investigated with the emerging 6G techniques for further security enhancement, such as cooperative relaying [17], full-duplex transmissions [18], non-orthogonal multiple access [19], reconfigurable intelligent surfaces [20], programmable antenna systems [21], and aerial space diversity [22]. These security problems and their variants are typically formulated as optimizations, which are commonly solved by using different iterative techniques. Despite the capability to approach the optima, they are usually burdened with computation complexity that becomes significantly more severe with larger network size. Moreover, the same complexity of re-execution is required whenever the network condition becomes different, making them hardly suited for the escalated network dynamics along with 6G communications.

Recently, there have been increasing research efforts in learning-based physical layer security solutions, where a direct mapping between the channel condition and security strategy can be established in a data-driven manner [6]. In this respect, various models such as multi-layer perceptrons (MLPs) [23], convolutional neural networks [24], and graph neural networks [25] have been applied for secure transmission beamforming [26], physical layer authentication [27], secrecy key generation [28], etc. Although the advantage of these approaches with prompt inference for implementation can be rather attractive, their discriminative nature makes them potentially brittle to environmental variations, resulting in possible out-of-distribution non-robustness in practical applications.

A powerful alternative category of learning approach is based on the generative models, which approximate the overall distribution of data [29]. Early attempts have resorted to the generative adversarial networks [30] and variational autoencoders [31] for generative security designs. Recently, the GDMs have emerged as the frontier methods enabling high-quality strategy output, such as the generative secure Internet of Things communications [32], secure aerial collaborations [33], and secure wireless sensing [34]. In this regard, the application of GDMs for wireless secrecy is still unexplored and thus there remains great potential to further reveal. Moreover, one of the central challenge for any specific learning model is the domain generalization issue, as the essential to cover the various scenarios in the 6G networks [35]. Towards this issue, the MoE framework offers a scalable solution, and a few recent studies employ the MoE for semantic communications [36] and wireless resource allocation [37]. Despite the potential, the MoE architecture has not, to the best of our knowledge, been leveraged for physical layer security. Therefore, we in this work establish a MoE-based approach to achieve instantaneous environmental adaptation while taking the advantage of the GDM-powered security expertise, achieving resilient security provisioning across diverse wireless scenarios.

## III. SYSTEM MODEL AND PROBLEM FORMULATION

## A. System Model

We consider a downlink multi-user MISO system where a base station (BS), equipped with M antennas, serves K singleantenna legitimate users, denoted by the set $\mathcal { K } = \{ 1 , \ldots , K \}$ Meanwhile, each legitimate receiver is associated with one single-antenna eavesdropper that intends to wiretap the transmitted message. Practically, the associated eavesdropper can be interpreted as an effective worst-case wiretap receiver for the user (e.g., the most threatening one). The BS employs linear precoding to transmit the confidential data symbol $s _ { k }$ to user k, with $\mathbb { E } [ | s _ { k } | ^ { 2 } ] = 1 , \forall k \in K$ . For security enhancement, the BS also injects J columns of unit-power AN signals, denoted by $\varsigma _ { j }$ with $j \in \mathcal { I } = \{ 1 , \dots , J \}$ . Accordingly, the overall transmit signal vector $\pmb { x } \in \mathbb { C } ^ { M }$ is given by

$$
\pmb { x } = \sum _ { k \in \mathcal K } \pmb { u } _ { k } s _ { k } + \sum _ { j \in \mathcal J } \pmb { v } _ { j } \varsigma _ { j } ,\tag{1}
$$

where $\mathbf { u } _ { k } \in \mathbb { C } ^ { M }$ is the transmission beamforming vector for user $k ,$ and $\pmb { v } _ { j } \in \mathbb { C } ^ { M }$ is the j-th AN vector. Let $\pmb { h } _ { k } \in \mathbb { C } ^ { M }$ denote the channel gain between the BS and user k, and $\boldsymbol { h } _ { \mathrm { E } , \boldsymbol { k } } ~ \in ~ \mathbb { C } ^ { M }$ as the wiretap channel associated with the corresponding eavesdropper. The signal-to-interference-plusnoise ratio (SINR) for legitimate transmission of user k is

$$
\gamma _ { k } = \frac { \lvert h _ { k } ^ { \dag } \boldsymbol { u } _ { k } \rvert ^ { 2 } } { \displaystyle { \sum _ { k ^ { \prime } \in { \mathcal { K } } \backslash \{ k \} } \lvert h _ { k } ^ { \dag } \boldsymbol { u } _ { k ^ { \prime } } \rvert ^ { 2 } + \sum _ { j \in { \mathcal { T } } } \lvert h _ { k } ^ { \dag } \boldsymbol { v } _ { j } \rvert ^ { 2 } + \sigma _ { 0 } ^ { 2 } } } ,\tag{2}
$$

where $\sigma _ { 0 } ^ { 2 }$ is the power of the additive white Gaussian noise as assumed identical across all receivers. Similarly, the SINR for the eavesdropper intercepting the signal of user k is

$$
\gamma _ { \mathrm { E } , k } = \frac { \vert { h } _ { { \mathrm { E } } , k } ^ { \dagger } \pmb { u } _ { k } \vert ^ { 2 } } { \displaystyle \sum _ { k ^ { \prime } \in \mathcal { K } \backslash \{ k \} } \vert { h } _ { { \mathrm { E } } , k } ^ { \dagger } \pmb { u } _ { k ^ { \prime } } \vert ^ { 2 } + \sum _ { j \in \mathcal { J } } \vert { h } _ { { \mathrm { E } } , k } ^ { \dagger } \pmb { v } _ { j } \vert ^ { 2 } + \sigma _ { 0 } ^ { 2 } } .\tag{3}
$$

Then, the secrecy rate for user k is obtained as

$$
R _ { k } = ( \log ( 1 + \gamma _ { k } ) - \log ( 1 + \gamma _ { \mathtt { E } , k } ) ) ^ { + } ,\tag{4}
$$

where we assume unit bandwidth for the system transmissions and $( \cdot ) ^ { + } = \operatorname* { m a x } \left\{ 0 , \cdot \right\}$

![](images/f3abce7816c2ed90c53c6297cc18f28b837b2221c0fb63dca148c962661e6809.jpg)  
Fig. 1. Secure communications under diverse scenarios.

## B. Problem Formulation

Our objective is to maximize the sum secrecy rate of all users under a total transmit power constraint. Accordingly, the problem is formulated as

$$
\operatorname* { m a x } _ { \{ \substack { u _ { k } \} _ { k \in \mathcal { K } } , \{ v _ { j } \} _ { j \in \mathcal { I } } } } \quad \sum _ { k \in \mathcal { K } } R _ { k }\tag{5a}
$$

$$
\mathrm { s . t . } \qquad \sum _ { k \in { \mathcal K } } \left| { \pmb u } _ { k } \right| ^ { 2 } + \sum _ { j \in { \mathcal I } } \left| { \pmb v } _ { j } \right| ^ { 2 } \leq P _ { \operatorname* { m a x } } ,\tag{5b}
$$

where $P _ { \mathrm { m a x } }$ is the power budget of the BS. For notation simplicity, we define a composite vector w that concatenates all transmission and AN beamforming vectors as

$$
\begin{array} { r } { \pmb { w } = \left[ \pmb { u } _ { 1 } ^ { \top } , \ldots , \pmb { u } _ { K } ^ { \top } , \pmb { v } _ { 1 } ^ { \top } , \ldots , \pmb { v } _ { J } ^ { \top } \right] ^ { \top } , } \end{array}\tag{6}
$$

representing the complete security strategy. Also, we define a composite channel vector h that collects all channel state information over the network

$$
\pmb { h } = \left[ \pmb { h } _ { 1 } ^ { \top } , \ldots , \pmb { h } _ { K } ^ { \top } , \pmb { h } _ { \mathrm { E } , 1 } ^ { \top } , \ldots , \pmb { h } _ { \mathrm { E } , K } ^ { \top } \right] ^ { \top } .\tag{7}
$$

For the security optimization in (5), it essentially corresponds to a mapping from h to w, denoted as ${ w } = \mathcal { F } _ { \phi } ( h )$ such that the overall secrecy rate can be maximized for a given scenario ϕ. Although this problem has been well addressed in existing work from both optimization and learning perspectives, they are established upon the implicit assumption of a certain given network scenario. In contrast, we in this work intend to develop a unified security solution to the problem across different scenarios, as shown in Fig. 1.

Towards this goal, we first explicitly define the scenario, which is usually referred to as a high-level concept without quantitative characterization. Considering that different network scenarios induce different propagation, which is further revealed in the channel gains, we use these gains to define the scenario quantitatively. Accordingly, we in this work adopt the Rician channel model specified as

$$
\sqrt { \beta _ { 0 } d ^ { - \kappa _ { L } } } \left( \sqrt { \frac { \kappa _ { R } } { \kappa _ { R } + 1 } } { \pmb h } ^ { ( \mathrm { L o S } ) } + \sqrt { \frac { 1 } { \kappa _ { R } + 1 } } { \pmb h } ^ { ( \mathrm { N L o S } ) } \right) ,\tag{8}
$$

where $\beta _ { 0 }$ is the path loss at the reference distance, d is the distance between the transmitter and receiver, $\kappa _ { L }$ is the path loss exponent, and $\kappa _ { R }$ is the Rician factor determining the ratio of power in the line-of-sight (LoS) component ${ h ^ { ( \mathrm { L o S } ) } }$ to the non-line-of-sight (NLoS) component $\bar { h } ^ { ( \mathrm { N L } \mathrm { o S } ) }$ . The channel model in (8) is versatile to cover a wide variety of wireless environments. On this basis, we define the combination $\phi ~ = ~ ( \kappa _ { L } , \kappa _ { R } )$ as the numerical characteristics to describe the network scenarios. Accordingly, a few typical examples may include that, a tuple (4,1) largely corresponds to the dense urban environment, a tuple (3,1) is likely to be a NLoS suburban scenario, a tuple (2,10) often represents LoS rural or open filed communications, etc.

Under this formulation, the composite channel vector h is drawn from a distribution conditioned on the scenario, i.e., $h \sim p ( h | \phi )$ . The fundamental challenge under our formulation is no longer to target at a scenario-specific mapping ${ \mathcal { F } } _ { \phi } ,$ , but to design a universal framework $\mathcal { F }$ that is effective for a whole set of scenarios $\phi ~ \in ~ \Phi$ . The goal is to ensure that for any channel realization h across the wireless scenarios in $\Phi ,$ , the security strategy ${ \pmb w } = \mathcal { F } ( { \pmb h } )$ achieves high secrecy performance, transforming the problem from scenario-specific optimization to robust cross-scenario generalization.

## IV. GDM-BASED SECURITY EXPERT

As the basis to unlock the cross-scenario physical layer security framework over Φ, we first consider a few representative scenarios and establish the scenario-specific security solution, i.e., construct the mapping from the channel condition to the security beamformer as ${ w } = \mathscr { F } _ { \phi } ( h ; \Theta )$ , with given $\phi \in \Phi$ and Θ collecting the network parameters. The target output is a high-dimensional continuous security strategy, for which the underlying secrecy-rate maximization is highly non-convex. The non-convexity and random fading lead to diverse suboptima, motivating distribution learning rather than point estimation. Compared with alternative generative paradigms, conditional diffusion offers stable likelihood-based training, strong capability in modeling complex continuous distributions, and a controllable sampling procedure. Consequently, we train a specialized GDM-based security expert for each scenario, and the expert learns the conditional distribution and generate highquality security strategies for channel realizations under the current scenario.

## A. Conditional Diffusion Model

The security expert is essentially a conditional denoising diffusion model, which involves two stages, i.e., the forward process that gradually adds noise to the data, and reverse process that remove the noise to generate new data, as shown in Fig. 2. To facilitate the real-valued neural network operations, we first represent the complex-valued channel and beamforming vectors by separating the real and imaginary parts for new representations as

$$
\begin{array} { r } { \boldsymbol { z } = \left[ \Re ( \pmb { w } ) ^ { \top } , \ \Im ( \pmb { w } ) ^ { \top } \right] ^ { \top } , \quad \pmb { c } = \left[ \Re ( \pmb { h } ) ^ { \top } , \ \Im ( \pmb { h } ) ^ { \top } \right] ^ { \top } . } \end{array}\tag{9}
$$

Accordingly, the GDM is trained to use c as the condition or guidance to recover the intended security strategy as z.

We adopt the channel-conditioned denoising diffusion probability model (DDPM) as the backbone for the GDM expert, which includes $T$ denoising steps, denoted as $\boldsymbol { \mathcal { T } } =$ $\{ 1 , 2 , \cdots , T \}$ . Specifically, the forward diffusion process is a Markov chain that progressively adds Gaussian noise to the security strategy tensor over $T$ steps. We use a variance schedule $\{ \beta _ { t } \} _ { t \in \mathcal { T } }$ with $0 \leq \beta _ { t } \leq 1$ , and conduct the transition at step t as

![](images/e63523a4cf5909f79a55dbd60284cbe21ec43748e8a3c452867d740bbe150a1f.jpg)  
Fig. 2. Conditional denoising diffusion model for security beamforming.

$$
q ( z _ { t } | z _ { t - 1 } ) = \mathcal { N } \left( z _ { t } ; \sqrt { 1 - \beta _ { t } } z _ { t - 1 } , \beta _ { t } \pmb { I } \right) ,\tag{10}
$$

where I is the identity matrix, and $z _ { \mathrm { 0 } }$ is the (near-)optimal security strategy at the initial step. By cascaded iterations in (10), we arrive at a close-form sample ${ \boldsymbol { z } } _ { t }$ at any arbitrary step t form from the initialization as

$$
z _ { t } = \sqrt { \bar { \alpha } _ { t } } z _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ,\tag{11}
$$

where $\begin{array} { r } { \alpha _ { t } = 1 - \beta _ { t } , \bar { \alpha } _ { t } = \prod _ { t ^ { \prime } = 1 } ^ { t } \alpha _ { t ^ { \prime } } . } \end{array}$ , and $\epsilon \sim \mathcal { N } ( 0 , I )$ is the sampled noise. As the final step $T$ is approached, the distribution of $z _ { T }$ reduces to a standard isotropic Gaussian distribution.

The reverse process aims to reconstruct the (near-)optimal secure beamformer from noise. Starting from pure noise $z _ { T } \sim \mathcal { N } ( 0 , I )$ , a neural network is trained to predict the noise component ϵ that was added at each step $t \in \mathcal T$ . Denote the neural network as $\boldsymbol { \epsilon } _ { \Theta } ( z _ { t } , t , \boldsymbol { c } )$ , it takes the noisy tensor $z _ { t } .$ , the step $t ,$ and the channel condition c as input, and the output is used to assist the evaluation of expectation of denoising step. Specifically, the reverse transition is then given by

$$
p _ { \Theta } ( z _ { t - 1 } | z _ { t } , c ) = \mathcal { N } ( z _ { t - 1 } ; \mu _ { \Theta } ( z _ { t } , t , c ) , \sigma _ { t } ^ { 2 } I ) ,\tag{12}
$$

where the mean $\mu _ { \Theta }$ is a function of the predicted noise ϵ<sub>Θ</sub> from the neural network

$$
\mu _ { \Theta } ( z _ { t } , t , c ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( z _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \Theta } ( z _ { t } , t , c ) \right) .\tag{13}
$$

By iteratively applying this procedure for $T$ steps, the model starts from a pure noise $z _ { T } \sim \mathcal { N } ( \mathbf { 0 } , I )$ , and arrive the security strategy tensor $z _ { \mathrm { 0 } }$ . Then, we can recover the (near-)optimal beamformer by re-assembling the real and imaginary parts as

$$
{ \pmb w } ^ { \star } = \frac { \sqrt { P _ { \operatorname* { m a x } } } } { \| z _ { 0 } \| } \left( z _ { 0 } [ 1 : M ( K + J ) ] \right.\tag{14}
$$

as normalized by the transmit power budget, where ȷ is the imaginary unit.

![](images/5ad6a553293d3ce7f698282fa830c0e13ad3ed861d4ec90484565cacd3a1b5a8.jpg)  
Fig. 3. Transformer-based denoising step.

## B. Transformer-Based Denoising Process

The diffusion model presented above consists of T timestps, where each step requires a neural network to assist the denoising operation. In this respect, we adopt the Transformer-based structure, also known as diffusion Transformer (DiT), which is well-suited to the considered security problem, capturing the intricate influence among different users while conditioned on the network channel state. Moreover, the DiT has the advantages of permutation-invariant attention mechanism to handle variable-length input, and thus improves the generalization capability of the model.

Particularly, this security-aware DiT composes of I cascaded Transformer blocks, denoted as $\mathcal { Z } ~ = ~ \{ 1 , 2 , \ldots , I \}$ shown in Fig. 3 and elaborated below.

1) Input Embedding: For Transformer-based operations, the first step is to tokenize the input.

Consider the t-th denoising step, the input noisy strategy tensor ${ \boldsymbol { z } } _ { t }$ and the channel condition tensor c are treated as 2D grids. We use a patching strategy where each entry (corresponding to a specific antenna and beamforming vector) is linearly projected into a latent token, with a trainable convolutional layer. The token is further flattened and added with a sinusoidal positional encoding to retain the spatial information, leading to the tokenized input as $z _ { t , 0 } ^ { ( \mathrm { t k n } ) }$ and $c ^ { ( \mathrm { t k n } ) }$ Accordingly, the tokenization is specified as

$$
z _ { t , 0 } ^ { ( \mathrm { t k n } ) } = \mathsf { F L T } \left( \mathsf { C o n v 2 D } \left( z _ { t } \right) \right) + e ^ { ( \mathrm { p o s } ) } ,\tag{15}
$$

and

$$
\boldsymbol { c } ^ { \left( \mathrm { t k n } \right) } = \mathsf { F L T } \left( \mathsf { C o n v 2 D } \left( \boldsymbol { c } \right) \right) + \boldsymbol { e } ^ { \left( \mathrm { p o s } \right) } ,\tag{16}
$$

where the subscript 0 indicates the initial input for the cascaded I blocks, FLT(·) denotes the flattening operation, and $e ^ { ( \mathrm { p o s } ) }$ is the positional encoding shared at the noisy input and channel condition. The tokenization explicitly preserves the security strategy structure, which is critical for scaling the expert to different network configurations. Meanwhile, the positional embedding is designed to align token positions with user-antenna streams, providing a physically meaningful inductive bias that improves model generalization capability.

Moreover, the information on current step, number of users, and number of antennas, are embedded as the global conditioning vector. In this regard, the denoising operation becomes well aware of current step and the basic network configuration, such that better generalization can be supported. Particularly, the step t goes through the sinusoidal embedding, together with the scalar M and $K ,$ constitute a vector as the global conditioning embedding. Furthermore, a linear projection layer is employed to create a tuple of representations used later in the Transformer blocks, specified as

$$
\begin{array} { r l r } & { } & { \left\{ \gamma _ { t , i } ^ { ( \mathrm { s i f } ) } , \delta _ { t , i } ^ { ( \mathrm { s i f } ) } , g _ { t , i } ^ { ( \mathrm { s i f } ) } , \gamma _ { t , i } ^ { ( \mathrm { c r s } ) } , \delta _ { t , i } ^ { ( \mathrm { c r s } ) } , g _ { t , i } ^ { ( \mathrm { c r s } ) } , \gamma _ { t , i } ^ { ( \mathrm { f f n } ) } , \delta _ { t , i } ^ { ( \mathrm { f f n } ) } , g _ { t , i } ^ { ( \mathrm { f f n } ) } \right\} _ { i \in \mathcal { D } } } \\ & { } & { = \mathsf { L } | \mathsf { N } ( [ \mathsf { S i n E m b } ( t ) , M , K ] ) , } \end{array}\tag{17}
$$

where $\gamma _ { t , i } ^ { \mathrm { ( x ) } }$ and $\delta _ { t , i } ^ { ( \mathrm { x } ) }$ are the scale and shift for the adaptive layer normalization (AdaLN) operations, and $g _ { t , i } ^ { ( \mathrm { x } ) }$ indicates the gating coefficient, with $\mathbf { x } ~ \in ~ \{ \mathrm { s l f , c r s , f f n } \}$ denotes the selfattention, cross-attention, and feed forward network within the Transformer. As such, the information on current denoising step and global network setting can be conveyed into the denoising process, improves both fitness and generalization of the model.

2) DiT Block Structure: As noted before, the tokenized input goes through I DiT blocks, which mainly resort to the attention mechanism to integrating the channel state information to refine the security strategy. The main operations include:

• AdaLN: As the basic supporting operation within the Transformer, we adopt the adaptive LayerNorm, which exploits the global conditioning embedding induced coefficients. Particularly, for the input tensor y, adaptive LayerNorm is conducted as

$$
\begin{array} { r l r } & { } & { \mathsf { A d a L N } \left( y ; \gamma _ { t , i } ^ { \left( \mathrm { x } \right) } , \delta _ { t , i } ^ { \left( \mathrm { x } \right) } \right) = \gamma _ { t , i } ^ { \left( \mathrm { x } \right) } \odot \mathsf { L N } ( y ) + \delta _ { t , i } ^ { \left( \mathrm { x } \right) } , } \\ & { } & { \mathrm { x } \in \{ \mathrm { s l f } , \mathrm { c r s } , \mathrm { f f n } \} , } \end{array}\tag{18}
$$

as will be used later in the DiT, where LN(·) is the LayerNorm operation.

• Self-attention: As the tokenized input is modulated by the AdaLN, self attention calculation is conducted. This mechanism allows the model to capture the complexcoupled and long-range dependencies among the strategy elements of the beamforming and AN vectors. Then, the attention results are updated through a gated residual to update the token, given as

$$
\begin{array} { r } { z _ { t , i } ^ { ( \mathrm { t k n } ) } \gets z _ { t , i } ^ { ( \mathrm { t k n } ) } + g _ { t , i } ^ { ( \mathrm { s l f } ) } \mathsf { S l f A t t } ( \mathsf { A d a L N } ( z _ { t , i } ^ { ( \mathrm { t k n } ) } ; \gamma _ { t , i } ^ { ( \mathrm { s l f } ) } , \delta _ { t , i } ^ { ( \mathrm { s l f } ) } ) ) , } \end{array}\tag{19}
$$

where $\mathsf { S l f A t t } ( \cdot )$ denotes the self-attention calculation and the query, key, and value sequences are projections of input tokens.

• Cross-attention: Then, the tokenized channel state information is exploited as the guidance for the denoising process. Particularly, the cross-attention operation is conducted that, the output of self-attention serves as the query sequence, and the keys and values are based on the channel tokens, specified as

$$
\begin{array} { r l } & { z _ { t , i } ^ { ( \mathrm { t k n } ) } \gets z _ { t , i } ^ { ( \mathrm { t k n } ) } + g _ { t , i } ^ { ( \mathrm { c r s } ) } } \\ & { \quad \quad \quad \times \mathrm { C r s A t t } ( \mathsf { A d a L N } ( z _ { t , i } ^ { ( \mathrm { t k n } ) } ; \gamma _ { t , i } ^ { ( \mathrm { c r s } ) } , \delta _ { t , i } ^ { ( \mathrm { c r s } ) } ) , c ^ { ( \mathrm { t k n } ) } ) , } \end{array}\tag{20}
$$

where $\mathsf { C r s A t t } ( \cdot )$ denotes the cross-attention. Injecting channel information through conditional tokens and cross-attention enable the expert to learn a channel-aware solution landscape, which is necessary for robust secrecy strategy generation under heterogeneous propagation conditions.

• Feed-forward: Finally, the tokens pass through a position-wise feed-forward network, which consists of two linear layers. The gated residual is also applied for the output, represented as

$$
\begin{array} { r } { z _ { t , i } ^ { ( \mathrm { t k n } ) } \gets z _ { t , i } ^ { ( \mathrm { t k n } ) } + g _ { t , i } ^ { ( \mathrm { f f n } ) } { \sf F F N } ( \mathrm { A d a L N } ( z _ { t , i } ^ { ( \mathrm { t k n } ) } ; \gamma _ { t , i } ^ { ( \mathrm { f f n } ) } , \delta _ { t , i } ^ { ( \mathrm { f f n } ) } ) ) , } \end{array}\tag{21}
$$

where FFN(·) denotes the feed forward operation.

The DiT operations above are conducted I times within the I cascaded blocks, the channel condition is repeatedly injected to affect the strategy token such that the security strategy get continuously adjusted and refined.

3) Output Generation: After passing through all I DiT blocks, the final sequence of strategy tokens, i.e., $z _ { t , I } ^ { ( \mathrm { t k n } ) }$ is arrived. We then reconstruct the updated noisy strategy vector from the arrived token. To this end, we also adopt the AdaLN operation to modulate the token, which is then linearly projected to produce the predicted noise with the same dimension of the strategy vector. This finishes the overall noise prediction network $\epsilon _ { \Theta }$ . The noise prediction is further used by the sampling algorithm to compute the denoised mean $\mu _ { \Theta } ,$ leading to a less noisy strategy vector $z _ { t - 1 }$ , as the output of current denoising step.

## C. Loss Function and Training

The training of each GDM-based security expert follows a two-stage paradigm designed to combine the stability of supervised learning with the performance-oriented fine-tune.

In the first stage, we perform supervised pre-training. For each representative scenario $\phi ,$ we adopt the security optimization algorithm (e.g., the approach in [38]) to construct a dataset of channel realizations and their corresponding nearoptimal security strategies. The GDM expert is then trained to learn the conditional distribution $p ( w | h )$ by minimizing the standard diffusion model loss, as the mean squared error between the true and predicted noise, specified as

$$
\mathcal { L } ^ { \mathrm { ( p r e ) } } \left( \Theta \right) = \mathbb { E } _ { t , w , \epsilon } \left[ \| \epsilon - \epsilon _ { \Theta } ( z _ { t } , t , c ) \| ^ { 2 } \right] .\tag{22}
$$

This pre-training stage enables the model with a fundamental understanding of the solution space.

In the second stage, we fine-tune the pre-trained model to further maximize the sum secrecy rate. Particularly, for a given channel condition, we use the GDM expert to generate a security strategy. We use the negative of sum secrecy rate as the loss function, given as

$$
\mathcal { L } ^ { ( \mathrm { f u n } ) } ( \Theta ) = \mathbb { E } _ { h } \left\{ - \sum _ { k \in \mathcal { K } } R _ { k } ( h , \mathcal { F } _ { \phi } ( h ; \Theta ) ) \right\} ,\tag{23}
$$

such that the model parameters are then adjusted to further improve the secrecy rate.

## D. Complexity of GDM Experts

For the proposed GDM experts, each performs DDPM sampling with T denoising steps, where each step evaluates a DiT denoiser consisting of I cascaded Transformer blocks. The DiT block includes the operations of AdaLN, self-attention over the strategy tokens, cross-attention between the strategy and channel tokens, and position-wise feed-forwards. For the tokenization operations, the length of strategy token and channel token are $L ^ { ( \mathrm { z } ) } = M ( K { + } J )$ and $L ^ { \mathrm { ( c ) } } = 2 M K$ , respectively. Also, denote the token embedding dimension and the feedforward hidden width by as $d ^ { \mathrm { ( t k n ) } }$ and $d ^ { \mathrm { ( f f n ) } }$ , respectively, the module-specific complexity is analyzed as follows. First, the self-attention over $\bar { L } ^ { ( \mathrm { z } ) }$ strategy tokens has complexity $\mathcal { O } \left( { ( L ^ { \mathrm { ( z ) } } ) } ^ { 2 } d ^ { \mathrm { ( t k n ) } } + { L ^ { \mathrm { ( z ) } } ( d ^ { \mathrm { ( t k n ) } } ) } ^ { 2 } \right)$ , due to the attention weights and output projection calculation. Also, the cross-attention is of complexity $\mathcal { O } \left( L ^ { ( \mathrm { z } ) } L ^ { ( \mathrm { c } ) } d ^ { ( \mathrm { t k n } ) } + \left( L ^ { ( \mathrm { z } ) } + L ^ { ( \mathrm { c } ) } \right) ( d ^ { ( \mathrm { t k n } ) } ) ^ { 2 } \right)$ , due to the calculation between strategy tokens and channel tokens. Finally, the feed-forward network has complexity $\mathcal { O } \left( L ^ { \mathrm { ( z ) } } d ^ { \mathrm { ( t k n ) } } d ^ { \mathrm { ( f f n ) } } \right)$ . Therefore, the complexity of each GDM expert is given as

$$
\begin{array} { r l } & { \mathcal { O } \left( T I \left( ( L ^ { \mathrm { ( z ) } } ) ^ { 2 } d ^ { \mathrm { ( t k n ) } } + L ^ { \mathrm { ( z ) } } L ^ { \mathrm { ( c ) } } d ^ { \mathrm { ( t k n ) } } \right. \right. } \\ & { \qquad \left. \left. + ( 2 L ^ { \mathrm { ( z ) } } + L ^ { \mathrm { ( c ) } } ) ( d ^ { \mathrm { ( t k n ) } } ) ^ { 2 } + L ^ { \mathrm { ( z ) } } d ^ { \mathrm { ( t k n ) } } d ^ { \mathrm { ( f f n ) } } \right) \right) , } \end{array}\tag{24}
$$

which is further scaled with the number of selected experts when assembled within the overall MoE architecture.

## V. MOE-BASED CROSS-SCENARIO SECURITY SOLUTION

In the preceding section, we elaborated on the design of a scenario-specific GDM-based security expert. Although a single Transformer-based diffusion model possesses a certain degree of scalability, its security performance degrades significantly when the underlying wireless environment changes. To effectively provide security across a continuum of scenarios in Φ, we propose a MoE framework that leverages the pre-trained GDM-based models as its supporting experts.

![](images/a6e08e3a181756fa6fc776daa9b0a8ba3ddf9d747976559e54136d6373e7c259.jpg)  
Fig. 4. The MoE framework with GDM security experts.

To this end, we first select a set of representative wireless scenarios, indexed by $\begin{array} { r } {  { \mathcal { N } } ~ = ~ \{ 1 , 2 , \dots , N \} } \end{array}$ , with selected scenario denoted by $\Phi ^ { \mathrm { ( r e p ) } } ~ = ~ \{ \phi _ { n } \} _ { n \in \mathcal { N } }$ . For each specific scenario $\phi _ { n } ~ \in ~ \Phi ^ { \left( \mathrm { r e p } \right) }$ , we construct and train a dedicated GDM-based expert, possessing the learned security strategies, denoted by ${ \mathcal { F } } _ { \phi _ { n } }$ . The MoE framework is then established by orchestrating this collection of experts, in order to combine the specialized knowledge of these experts to achieve robust and high-performance security provisioning across the entire space of scenarios Φ.

Therefore, a gating network, or router is established as the core of the MoE framework, which intends to accurately identify the input channels information with its scenario properties. Then, the corresponding experts can be called such that their knowledge can be synergized to produce the scenario-aware security solution. The overall design is shown in Fig. 4 and the details are elaborated as follows.

## A. Scenario-Aware Feature Engineering

The responsibility of the router is to distinguish current network scenario and call the right experts. However, the scenario properties are embedded in the network channel information, which is of high-dimension and inherently random. Since our considered scenario are mainly characterized by the poth loss exponent and Rician factor, we introduce the feature engineering that are of low dimension yet remain sensitive to the physical parameters. Particularly, we construct a 6-dim feature, denoted by f, by calculating the statistical characteristics of the channel information.

The first category includes 3 features capturing the power distribution, which are sensitive to the path loss-related component, specified as

• Power Dynamic Range $( f _ { 1 } ) \colon$ The ratio of the maximum to minimum channel gain among the receivers, i.e., $\operatorname* { m a x } _ { k \in K } \left\{ \| h _ { k } \| ^ { 2 } , \| h _ { \mathrm { E } , k } \| ^ { 2 } \right\} / \overset { \cdot \cdot \cdot } { \underset { k \in K } { \operatorname* { m i n } } } \left\{ \| h _ { k } \| ^ { 2 } , \| h _ { \mathrm { E } , k } \| ^ { 2 } \right\}$ . A higher ratio leads to a more extreme power range.

• Variance of Log-Power $( f _ { 2 } )  \}$ The variance of the logscaled channel gain, Var log $( [ \Vert \boldsymbol h _ { k } \Vert ^ { 2 } , \Vert \boldsymbol h _ { \mathrm { E } , k } \Vert ^ { 2 } ] _ { k \in \mathcal { K } } ) )$ , which quantifies the power disparity across users.

• Average Path Gain $( f _ { 3 } ) { \mathrm { : } }$ : The mean received power across all users, Mean $\left( \left[ \Vert \pmb { h } _ { k } \Vert ^ { 2 } , \Vert \pmb { h } _ { \mathrm { E } , k } \Vert ^ { 2 } \right] _ { k \in \mathcal { K } } \right)$ , reflecting the overall channel power attenuation.

The second category has 3 features probing into the spatial structure of the channel condition, which is highly concerned with the LoS component of the channel, specified as

• Channel Condition Number $( f _ { 4 } ) \colon$ We concatenate the channel vectors at the users as a matrix, $\begin{array} { r l } { \pmb { H } } & { { } = } \end{array}$ $[ h _ { k } , h _ { \mathrm { E } } , _ { k } ] _ { k \in \mathcal { K } } .$ , and calculate the ratio of the largest to the smallest singular value of H, i.e., ${ \delta _ { \operatorname* { m a x } } ( \pmb { H } ) } / { \delta _ { \operatorname* { m i n } } ( \pmb { H } ) }$ where $\delta ( \cdot )$ denotes the singular value. A higher Rician factor often leads to a more ill-conditioned, lower-rank channel matrix.

• Ratio of Principal Singular Values $( f _ { 5 } ) \colon$ The ratio of the largest to the second largest singular value, $\delta _ { 1 } ( H ) / \delta _ { 2 } ( H )$ , which more directly measures the dominance of the strongest spatial path.

• Variance of Phase Angles (f<sub>6</sub>): The average phase variance across the antennas for each user’s channel vector, i.e., $\mathsf { M e a n } ( \mathsf { V a r } ( \mathsf { A r g } ( H ) ) )$ , where $\mathsf { A r g } ( \cdot )$ denotes the element-wise angle calculation and Var(·) is conducted over each column. In this respect, a stronger LoS component induces higher phase coherence and thus lower overall variance.

Through the engineered feature $f ,$ we obtain a compact yet comprehensive representation of the channel characteristics, which can be then used for scenario classification and expert selection. Moreover, the feature engineering enables a fixdimension representation regardless of the network channel dimensions, making the router directly applicable for networks with variable size configurations.

## B. Router Design and Training

The router, intending for scenario identification, can be implemented as a standard MLP for classification task. ${ \mathrm { A c } } -$ cordingly, the router takes the channel feature vector $f$ as input, and outputs a probability distribution over the N classes of representative scenarios.

The training of the router follows a standard supervised manner. To this end, we first generate a labeled dataset. Specifically, for each representative scenario $\phi _ { n } \in \Phi ^ { ( \mathrm { r e p } ) }$ , we generate a set of channel condition matrices following the scenario setting, i.e., generate the random network topology and calculate the channel gain based on the path loss exponent and Rician factor determined by the scenario. Then, we calculate the engineered channel feature, together with the groundtruth label associated with the representative scenario, is used for router training. The training is conducted to minimize the cross-entropy loss. The final layer of the router consists of $N _ { - }$ dim output with a Softmax activation function, which produces a gating probability N-dim vector $\textbf { \em q } = \mathbf { \Psi } [ q _ { n } ] _ { n \in \mathcal { N } }$ such that $\textstyle \sum _ { n \in { \mathcal { N } } } q _ { n } = 1$ , and $q _ { n }$ is the probability that scenario $\phi _ { n }$ is identified and expert ${ \mathcal { F } } _ { \phi _ { 1 } }$ gets selected.

## C. Expert Selection and Combination

As the router produce a probability distribution over the experts, the final MoE operation is to select the experts properly and determine the security beamforming strategy. This can be generally categorized into two cases. First, for scenarios that closely align with the representative ones, we can safely expect that one element of the gating probability vector is close to 1, which thus leads to a clear choice. In this case, we can simply select the corresponding expert for security strategy determination.

For the second case, also more likely to be encountered, there are unseen cross-scenarios, for which the gating probability also indicates multiple relevant experts. In this case, we adopt the “top-k” gating principle, where a set of $\bar { N }$ top-rated experts are selected, denoted as ${ \bar { \mathcal { N } } } \subseteq { \mathcal { N } } .$ . A direct approach is to normalize the gating probability within $\bar { \mathcal { N } }$ and use it as the weights to reach a linear combination, specified as

$$
\begin{array} { r } { \pmb { w } ^ { \star } = \displaystyle \sum _ { n \in \bar { \mathcal { N } } } \frac { q _ { n } } { \sum _ { n ^ { \prime } \in \bar { \mathcal { N } } } q _ { n ^ { \prime } } } \mathcal { F } _ { \phi _ { n } } ( \pmb { h } ) , } \end{array}\tag{25}
$$

which is further normalized with respect to the transmit power constraint as

$$
w ^ { \star }  w ^ { \star } \sqrt { \frac { P _ { \operatorname* { m a x } } } { \| w ^ { \star } \| ^ { 2 } } } .\tag{26}
$$

$\mathbf { A } s$ a further note, we can see that the first case with one selected expert corresponds to the “top-one” rule, as the special case to the general “top- $- k ^ { \dprime }$ settlement.

## D. Attention-Based Combiner Network

The previous combination approach based on the “top- $- k ^ { \dprime }$ rule, through direct and efficient, can be inherently limited, as it has the implicit assumption of linear interpolation. Given the complicated relationship between the considered scenario factors, i.e., path loss exponent and Rician factor, and the resultant channel gain, the assumption that the optimal solution lies on the straight line between the experts can be far from reasonable. Consequently, we resort to some more powerful mechanism to learn such non-linear combinations, towards more effective security enhancement.

To address this limitation and for improved security performance, we introduce a dedicated combiner network with attention mechanism as a residual learner. Instead of directly learning the final beamformer, the combiner is trained to predict a residual correction term, which is used along with the weighted expert output for final beamforming determination. In this respect, the combiner adjustment can be regarded as a further fine-tune to learn the non-linear mapping from the weighted experts to the optimum, while enabling a scalable model to handle variable numbers of experts.

1) Attention-Based Architecture: Basically, the number of selected experts through the gating network can be different, depending on how “cross” is the considered scenario, as compared with the representative scenarios. With this in mind, the combiner network is expected to synergize the output from different numbers of active experts. In this regard, a standard MLP with fixed-size input is incapable for such cases. We thus resort to the permutation-invariant attention mechanism to tackle the variable-length input.

Specifically, the input of combiner network includes:

• The scenario-aware feature vector, $f .$

• The beamformer proposal by the selected experts, $\{ w _ { n } \} _ { n \in \bar { \mathcal { N } } }$ , along with the selection probability, q.

Then, the input information is projected into the attention space and reinterpreted as query, key, and value. Here, we use the feature vector to induce the query through a linear network, given as $Q = \mathsf { L I N } ^ { ( \mathrm { q r y } ) } ( f )$ . Meanwhile, we use the weighted beamformer vector from the experts to derive the key and value, given as $\boldsymbol { K } _ { n } = \mathsf { L I N } ^ { ( \mathrm { k e y } ) } ( \bar { \boldsymbol { z } } _ { n } )$ and $V _ { n } = \mathsf { L I N } ^ { ( \mathrm { v a l } ) } ( z _ { n } )$ where $z _ { n }$ is the beamforming representation in real space as $z _ { n } = q _ { n } \cdot \left[ \Re ( { \pmb w } _ { n } ) ^ { \top } , \Im ( { \pmb w } _ { n } ) ^ { \top } \right] ^ { \top }$ , for all $n \in \bar { \mathcal { N } }$ . Note that the query is unique while the each expert output is projected to an independent key and value.

With the query, key, and value obtained, we can then conduct the attention-based aggregation such that the expert proposals can be synthesized in the form of

$$
o ^ { \mathrm { ( a t t ) } } = \sum _ { n \in \bar { \cal N } } \left( \mathsf { S o f t m a x } \left( \frac { Q K _ { n } ^ { \top } } { \sqrt { d ^ { ( \mathrm { a t t ) } } } } \right) V _ { n } \right) ,\tag{27}
$$

where $d ^ { \mathrm { ( a t t ) } }$ is the dimension of the latent attention space. As we see from (27), the attention mechanism allows the combiner to exploit the scenario feature (query) to weight the importance of the expert proposals, and gradually learns the non-linear relationship between the directly weighted beamformer and the optimum.

The context vector $\mathbf { \sigma } _ { o } ( \mathrm { a t t } )$ from the attention is then fed into a small MLP to generate the residual correction to the realvalued representation of the weighted beamforming, given as

$$
\begin{array} { r } { \Delta z = \mathsf { M L P } ^ { \mathrm { ( r d o ) } } \big ( o ^ { \mathrm { ( a t t ) } } \big ) . } \end{array}\tag{28}
$$

Here the MLP adopted is of fixed-dimension input and output, and the residual beamformer is reconstructed as $\begin{array} { r l } { \Delta w } & { { } = } \end{array}$ $[ \Delta z [ 1 : M ( K + J ) ] + \jmath \Delta z [ M ( K + J ) + 1 : 2 M ( K + J ) ] ]$ Then, we can use the weighted expert output along the residual beamformer for the final output, given as

$$
\begin{array} { r } { \pmb { w } ^ { \star } = \pmb { w } ^ { \circ } + \Delta \pmb { w } , } \end{array}\tag{29}
$$

where $\pmb { w } ^ { \circ }$ is obtained from the experts in the same form of (25), and $\boldsymbol { w } ^ { \star }$ is similarly normalized as (26) to obtain the final beamforming vector.

2) Combiner Network Training: For the overall MoE landscape, the combiner training accounts for the final stage, before which the GDM-based experts and gating network has already been trained with frozen neural network parameters. To fully reveal the potential for superior security performance, we adopt the unsupervised learning which also alleviates the burden of data generation and collection.

Specifically, for the cross-scenario with unseen settings of scenario parameters, we randomly generate the network topology and calculate the channel vector of the network. The channel vector gets through the router to get the probability distribution of the experts. Under given $ { \mathrm { \hat { \rho } } } _ { \mathrm { t o p } - k }  { \mathrm { \vec { ~ } } }$ rule, the corresponding top-rated experts are called to output their beamforming proposals. Finally, the channel feature and the expert output are fed into the combiner for the beamformer output. The combiner network is trained to maximize the sum secrecy rate as

$$
\mathcal { L } ^ { ( \mathrm { c m b } ) } ( \Xi ) = \mathbb { E } _ { h } \left\{ - \sum _ { k \in \mathcal { K } } R _ { k } ( \pmb { w } ; \Xi ) ) \right\} ,\tag{30}
$$

where w is the combiner output and $\Xi$ collects the parameters of the combiner network.

Overall, the residual learning provides a direct path to approach the optimal secure beamforming. This process trains the combiner to explicitly learn the non-linear correction to transform a simple linear interpolation towards a high-fidelity security strategy. Since the residual learning starts from an already strong baseline, the learning can be conducted in an efficient manner to cover the security requirement for the unseen cross-scenarios.

## E. Complexity of the Router and Combiner

Based on the preceding discussions, the router is a simple MLP operated over the engineered features. To obtain the feature vector, the matrix norm/mean/variance has complexity $\mathcal { O } ( M K )$ , while the singular value decomposition is of complexity $\mathcal { O } \left( \operatorname* { m i n } ( M , 2 K ) ^ { 2 } \operatorname* { m a x } ( M , 2 K ) \right)$ . Then the MLP-based router has complexity $\mathcal { O } \left( L ^ { \mathrm { ( r t r ) } } ( d ^ { \mathrm { ( r t r ) } } ) ^ { 2 } \right)$ with $L ^ { \mathrm { ( r t r ) } }$ and $d ^ { \mathrm { ( r t r ) } }$ being the number and dimension of the hidden layers, respectively. Accordingly, the router complexity is $\mathcal { O } \left( M K + \operatorname* { m i n } ( M , 2 K ) ^ { 2 } \operatorname* { m a x } ( M , 2 K ) + L ^ { \operatorname { ( r t r ) } } ( d ^ { \operatorname { ( r t r } ) } ) ^ { 2 } \right)$

For the attention-based combiner, the attention coefficients are calculated based on the engineered features and proposals from the selected experts, and thus the complexity is $\mathcal { O } \left( \bar { N } \cdot 2 M ( K + J ) d ^ { ( \mathrm { a t t } ) } \right)$ . Then the residual correction calculation through the MLP has complexity $\mathcal { O } \left( L ^ { \mathrm { ( r d o ) } } ( d ^ { \mathrm { ( r d o ) } } ) ^ { 2 } \right)$ with $L ^ { \mathrm { r d o } }$ layers of $d ^ { \mathrm { { r d o } } }$ -dim neurons. Therefore, the combiner complexity is $\mathcal { O } \left( \bar { N } \cdot 2 M ( K + J ) d ^ { { ( \mathrm { a t t } ) } } + L ^ { { ( \mathrm { r d o } ) } } ( d ^ { { ( \mathrm { r d o } ) } } ) ^ { 2 } \right)$

## VI. SIMULATION RESULTS

Simulation results are provided to evaluate our proposed framework for cross-scenario security performance. We consider a multi-user MISO system within an area of 500 m×500 m, where the BS is at the center and the legitimate users randomly distributed with an average distance of 100 m, and the eavesdroppers are randomly located with an average of 20 m away from its corresponding receiver. The base station has a maximum transmit power of 20 dBm, and the background noise power at the receivers is -110 dBm. The number of legitimate users (and thus eavesdroppers) is 4, and the number of BS antennas is 8. These settings are used as default unless otherwise indicated. For scenario setting, we consider a set of path loss exponents as {2,3,4}, and a set of Rician factors as {1,10}, which constitute totally 6 representative scenarios with different combination of path loss exponents and Rician factors.

For the neural networks, the proposed framework is trained on a dataset of 48,000 samples, with 8,000 samples generated for each of the 6 representative channel scenarios. Each GDMbased expert has 6 DiT blocks, trained for 2,000 epochs using the Adam optimizer with a learning rate of $5 \times 1 0 ^ { - 5 }$ The diffusion process uses 100 steps with $\beta _ { t }$ scheduled from 0.0001 to 0.02. The DiT attention modules have 8 heads and a token dimension of 512, while its feed-forward network uses two linear layers of 512→1536→512 with GELU activation. The gating network is an MLP with three linear layers of hidden dimension 512, trained for 10 epochs with a learning rate of $5 \times 1 0 ^ { - 4 }$ . The combiner has an attention dimension of 256 and a three-layer MLP with hidden dimension 512, trained for 200 epochs with a learning rate of $5 \times 1 0 ^ { - 5 }$

Since the overall MoE framework is complicated, to make the role of each component explicit, the results are organized as a component-wise evaluation pipeline. We first evaluates the GDM experts for the representative scenarios under varying system configurations. Then, the router is validated via feature separability and expert selection accuracy. Finally, the end-toend MoE performance across representative and unseen crossscenarios are corroborated considering different combiners with a sensitivity study on different expert activations.

![](images/019477070e43e98fb19c1bc23ac39a11801b216f38beffa89197c553da6ff538.jpg)  
Fig. 5. The training convergence of experts, router, and combiner.

## A. Convergence and Complexity Validation

We first illustrates the training convergence of the three main components in the proposed MoE framework in Fig. 5. The losses of the six GDM experts decrease steadily with training epochs and converge to low and stable values, indicating reliable learning of scenario-specialized generative security strategies. The router converges within a few epochs, where the loss drops rapidly and the classification accuracy quickly approaches saturation, validating the effectiveness of the proposed low-dimensional feature for scenario recognition. Moreover, the combiner loss decreases smoothly and stabilizes along with the trainings, demonstrating that the attentionbased residual fusion can be learned in a stable manner to refine the selected expert proposals. These convergence results collectively suggest that the overall framework is trainable with stable convergence behaviors across all modules.

To quantify the model complexity and inference latency, we evaluate the number of active parameters, computation FLOPs, and inference time, under different simulation settings, and the results are summarized in Table I. A key insight is that while the total framework has a comprehensive parameter set of 133.03 M, to ensure the coverage across diverse scenarios, the sparse gating mechanism ensures that the computational load remains relatively lightweight. Specifically, under Top-1 expert selection rule, the system activates only 23.23 M parameters, achieving an ultra-low inference latency of a few milliseconds. Even when scaling to Top-4 strategy, the latency remains millisecond level. Moreover, for a fixed number of selected experts, changing the network size leads to a moderate increase in FLOPs and latency due to the enlarged strategy and channel token lengths. Overall, the results indicate that, while the framework contains multiple experts, its online complexity is tunable and remains in the millisecond range for the considered settings, supporting the practicality of the proposed cross-scenario design.

TABLE I  
INFERENCE COMPLEXITY AND LATENCY PROFILE UNDER DIFFERENT NETWORK CONFIGURATIONS
<table><tr><td>Top-k</td><td>Netw. Config.</td><td>Act. Para.</td><td>FLOPs</td><td>Infer. Tm. (ms)</td></tr><tr><td rowspan="3">Top-1</td><td> $M = 8 , K = 2$ </td><td rowspan="3">23.23 M</td><td>1.00 G</td><td>5.926</td></tr><tr><td>M = 8, K = 4</td><td>1.72 G</td><td>6.424</td></tr><tr><td>M = 6, K = 4</td><td>1.29 G</td><td>6.043</td></tr><tr><td rowspan="3">Top-2</td><td> $M = 8 , K = 2$ </td><td rowspan="3">45.19 M</td><td>2.02 G</td><td>11.129</td></tr><tr><td>M = 8, K = 4</td><td>3.45 G</td><td>12.126</td></tr><tr><td>M = 6, K = 4</td><td>2.59 G</td><td>11.364</td></tr><tr><td rowspan="3">Top-4</td><td>M = 8, K = 2</td><td rowspan="3">89.11 M</td><td>4.02 G</td><td>21.536</td></tr><tr><td>M = 8, K = 4</td><td>6.90 G</td><td>23.532</td></tr><tr><td>M = 6, K = 4</td><td>5.15 G</td><td>22.006</td></tr><tr><td colspan="2"></td><td colspan="2">Model size: 0.14 M (Router) + 1.13 M (Combiner) + 6×21.96 M (GDM); Total 133.03 M</td><td></td></tr></table>

## B. Performance of GDM Experts

We show the performance of the single GDM experts, where for illustration purpose, we show two experts within the representative scenario set. In this respect, we evaluate the performance of experts and demonstrate the advantage of diffusion models. For performance comparison, we consider an optimization-based benchmark (OPT) and a conventional zeroforcing (ZF) secure-beamforming scheme. In this regard, ZF provides a lightweight conventional reference with a closedform spatial structure, while OPT provides a computationally more demanding per-instance optimization reference that directly targets the secrecy-rate objective. Moreover, consider the practical channel state information issue, we also evaluate the performance when the channel matrices are associated with 5% uncertainties.

Fig. 6 presents the performance of the GDM-based security expert trained for a specific wireless scenario characterized by light path loss and a strong Line-of-Sight (LoS) component $( \kappa _ { L } = 2 , \kappa _ { R } = 1 0 )$ . As illustrated across all three subfigures, the sum secrecy rate consistently improves with more system resources, such as a higher number of users, antennas, or greater transmit power. The key observation is the remarkable performance of our GDM expert, which operates nearly identically to the OPT benchmark under the assumption of perfect channel state information. This reveals that the GDM experts has the capability to learn the inherent structure of the optimal solution space. establishing the complex mapping from channel conditions to the optimal security beamforming strategy. Moreover, the GDM expert can be adaptive to the variations of network configurations of the considered factor, allowing significant generalization capability of the learning model.

More importantly, the results highlight the GDM’s robustness in the presence of imperfect channel information. When a 5% uncertainty is introduced (dashed lines), the GDM expert consistently outperforms the OPT method. This suggests that the GDM, through its training on diverse data, has learned a more robust strategy that is less susceptible to channel estimation errors compared to the optimization algorithm. The ZF method, experiences the most significant performance degradation under uncertainty. This makes the GDM model particularly applicable for practical wireless systems, as the channel information is inevitably associated with errors while the GDM-based design can more effectively maintain the reliable security performance.

![](images/d4033add5ce07f63f51f92581d7e89b56d271447b1d8b8b5b80d14db4cb9a610.jpg)

![](images/8e39671b06f4d84ac719772fe8e4e85f742725dc9463a836cdd5caff8eb1e989.jpg)

![](images/708a15f66f5d8dc57ff0bb396ded176a6e02186419b430a943f913bd88a63661.jpg)  
(a) Performance with respect to number of users. (b) Performance with respect to number of anten- (c) Performance with respect to transmit power. nas.

Fig. 6. Performance of GDM expert for wireless scenario with $\kappa _ { L } = 2 , \kappa _ { R } = 1 0$ (light attenuation with strong LoS, typically open-field rural).  
![](images/26015ab17ce021c8a03bebd3cdec4c31cfefded130ad35492fb201d18294df3c.jpg)

![](images/fa6669616704b46cd1e7280902ec8ab721129d51e256dc72434522414e6824b3.jpg)

![](images/6903dbec51a5e720e638271502f1afcc7681a15d24f58f9bc52b3147f7d792e2.jpg)  
(a) Performance with respect to number of users. (b) Performance with respect to number of anten- (c) Performance with respect to transmit power. nas.  
Fig. 7. Performance of GDM expert for wireless scenario with $\kappa _ { L } = 4 , \kappa _ { R } = 1$ (high attenuation with weak LoS, typically dense urban).

Fig. 7 evaluates the GDM expert specifically trained for an environment characterized by high path loss and a weak LoS component $( \kappa _ { L } = 4 , \kappa _ { R } = 1 )$ . The performance trends are consistent with the previous scenario, showing that the sum secrecy rate scales positively with the number of users, antennas, and transmit power, but the overall achieved secrecy rate is much lower due to the aggravated environment. Particularly, in this more challenging propagation environment, our GDM expert continues to demonstrate strong performance, closely approaching the optimization benchmark and dominating the ZF baseline. This confirms the model’s capability to learn effective security strategies even in harsh conditions.

When channel uncertainty is introduced, the GDM expert remains robust regardless of the uncertainties. Similarly in this harsh scenario, the GDM consistently outperforms the optimization benchmark under uncertainty, suggesting that the benefits of its learned robustness are reliable for different scenarios. The performance gap between the GDM and the optimal solution, especially when contrasted with the simpler ZF baseline, underscores the effectiveness and reliability of our proposed approach across different and challenging wireless environments.

## C. Performance of the Router

To evaluate the efficacy of the proposed MoE architecture, we first present a validation of the router, or the gating network, which serves as the core component for scenario identification and expert orchestration. The performance of the router is visualized in Fig. 8 and Fig. 9, which respectively demonstrate the separability of the engineered channel features and the classification accuracy of the router.

Fig. 8 provides a t-SNE visualization that projects the 6- dimensional engineered feature vectors, extracted from the channel matrices under the 6 representative scenarios, into a two-dimensional space for qualitative assessment. The results show 6 clearly delineated and well-separated clusters, where each cluster corresponds to one of the predefined wireless scenarios. This distinct clustering indicates that the proposed feature engineering is rather effective to captures the statistical properties of each scenario. This established separation is the foundation towards the scenario identification and expert association, which is further used to support the combination of expert output for final security strategy determination.

Meanwhile, Fig. 9 shows the performance of the router through a heatmap of the average expert selection probabilities. This matrix illustrates the probability of the gating network to select a particular expert (columns) when presented with input from a specific ground-truth scenario (rows). The matrix exhibits strong diagonal dominance, with probabilities for correct expert selection always exceeding 0.94. The nearzero values in the off-diagonal elements indicate a negligible rate of misclassification. The miss classified cases that happen rarely are also demonstrated in Fig. 8 with the misplaced markers and color identifications. As we can see in Fig. 8, the miss classifications are only encountered at the small intersections of different clusters. Nevertheless, the overall high classification accuracy implies that the router can reliably map the observed channel environment to the specialized experts, supporting the MoE framework to cover diverse scenarios.

![](images/11d8612c5c6849cc4473db2a3c7da3660a73be22fb7f5e55a70309bb785ad3a6.jpg)  
Fig. 8. The t-SNE visualization of the engineered channel features for the 6 representative scenarios.

## D. Cross-Scenario Security Performance

We now evaluate the end-to-end performance of the complete MoE framework, with the results presented in Fig. 10. This evaluation is designed to corroborate the core design goal of our architecture to effectively recognize and cover the unseen cross-scenarios. This requires, first, that the gating network can effectively select the appropriate expert for a given environment, and second, that the attention-based combiner can intelligently synthesize the expert output to achieve robust security across a wide varieties of unseen wireless scenarios. The results demonstrate the performance of using the top-1, top-2, and top-4 expert selection, both with our attention-based combiner and with a simple weighted-average combination baseline.

Fig. 10(a) presents the performance in the two representative scenarios upon which experts were explicitly trained. In these cases, the highest secrecy rate is achieved when only the single, top-rated expert is selected (Top-1). This result serves as a crucial sanity check, confirming that the gating network correctly identifies the single best expert for the known environments. When additional experts are included (Top-2, Top-4), the security performance can only get downgraded due to the intervention from inappropriate experts. Therefore, the model effectiveness is not necessarily monotonic with the number of selected experts, because expanded expert set may include less relevant expert and also increase the synthesis difficulty for the combiner. As a further note, we also show the results obtained through optimization, which is in consistency with those in Figs. 6 and 7 that the top-1 expert approximates the optimum.

![](images/d72b79ae8c4c65825bab140c1d2a7fdb59310aa7c7d4beaf2155205f21f2ed94.jpg)  
Fig. 9. Heatmap of the router with average expert selection probabilities.

The adaptability and robustness of the MoE architecture become more evident in the cross-scenarios as shown in Fig. 10(b), 10(c), and 10(d). These figures test the system in unseen environments that interpolate the characteristics of the representative scenarios. In contrast to the previous case, relying on a single expert (Top-1) here leads to certain inferior performance. The optimal strategy in these interpolated environments is not captured by any single expert but by activating multiple relevant experts (Top-2 and Top-4) to provide the candidate solutions to achieve a high secrecy rate. Moreover, to further quantify the contribution of the combiner module, the optimization-based reference is included. Evidently, the results show that the attention-based residual synthesis consistently approaches the optima, verifying that the proposed combiner effectively learns the non-linear mapping beyond linear interpolation. This observation is particularly pronounced when multiple experts are activated under the unseen cross scenarios, where the gap of weighted linear average to optima becomes evident, while the attention-based combiner reliably maintains near-optimal performance.

Therefore, the cross-scenario results verifies the reliable security guarantee in unknown environments, which relies on the proper expert selection by the router and the effective synthesis by the attention-based combiner. The approach of simply weighted averaging the outputs of the selected experts consistently yields poor results, as linear interpolation is insufficient to approximate the complex, non-linear solution space of secure beamforming. In contrast, the attentionbased combiner intelligently weighs the expert proposals and computes a residual correction, transforming the simple baseline into a high-fidelity security strategy. In this regard, the performance gain can be substantial, particularly when the scenario becomes more complex and more “cross” and thus requires the collaboration from more experts. Overall, these results validate the MoE framework can effectively orchestrate specialized knowledge to provide robust and superior security across diverse wireless landscapes.

As a further note, the current cross-scenario evaluation mainly considers propagation regimes within the coverage of the representative scenario set. Generalization to strongly extrapolated or fundamentally different channel environments is more challenging and is not guaranteed by the present framework, and extending the expert coverage and developing more explicit out-of-distribution adaptation mechanisms constitute important future directions. Additionally, the present evaluation focuses on the average sum secrecy rate in accordance with the secrecy-rate maximization objective. Nevertheless, the proposed framework can be conveniently extended to cover metrics like variance, secrecy outage, and lowertail performance without changing the basic GDM-based MoE architecture.

## VII. CONCLUSION

In this paper, we have proposed a novel and adaptive framework to address the challenge of physical layer security provisioning across diverse wireless scenarios. We have proposed a MoE architecture that orchestrates a committee of specialized security experts, and each expert is realized as a powerful GDM with a Transformer-based denoising network. We also have designed a gating network for expert selection and a combination network to intelligently synthesizes the expert outputs for security strategy determination. We have showed that the MoE framework effectively adapts to unseen environments, providing reliable, robust, and superior crossscenario physical layer security for future 6G networks.

## REFERENCES

[1] C.-X. Wang, X. You, X. Gao, X. Zhu, Z. Li, C. Zhang, H. Wang, Y. Huang, Y. Chen, H. Haas, J. S. Thompson, E. G. Larsson, M. D. Renzo, W. Tong, P. Zhu, X. Shen, H. V. Poor, and L. Hanzo, “On the road to 6G: Visions, requirements, key technologies, and testbeds,” IEEE Commun. Surveys Tuts., vol. 25, no. 2, pp. 905–974, 2023.

[2] T. N. Turnip, B. Andersen, and C. Vargas-Rosales, “Towards 6G authentication and key agreement protocol: A survey on hybrid post quantum cryptography,” IEEE Commun. Surveys Tuts., 2025, to appear.

[3] A. Chorti, A. N. Barreto, S. Kopsell, M. Zoli, M. Chafii, P. Sehier,¨ G. Fettweis, and H. V. Poor, “Context-aware security for 6g wireless: The role of physical layer security,” IEEE Commun. Stand. Mag., vol. 6, no. 1, pp. 102–108, 2022.

[4] G. Li, Y. Zhou, Y. Sun, S. Wen, S. Wang, A. Hu, and G. Wen, “Endogenous security techniques for safeguarding next-generation spectrum usage at the physical layer,” IEEE Netw., 2025, to appear.

[5] J. M. Hamamreh, H. M. Furqan, and H. Arslan, “Classifications and applications of physical layer security techniques for confidentiality: A comprehensive survey,” IEEE Commun. Surveys Tuts., vol. 21, no. 2, pp. 1773–1828, 2019.

[6] J. Du, H. Wang, C. Jiang, J. Simonjan, J. Wang, and M. Debbah, “Distributed AI-based secure communications in space-air-ground-sea integrated networks,” IEEE Commun. Mag., vol. 63, no. 7, pp. 48–55, 2025.

[7] X. Tang, K. Zhao, C. Shen, Q. Du, Y. Wang, D. Niyato, and Z. Han, “Deep graph reinforcement learning for UAV-enabled multi-user secure communications,” IEEE Trans. Mob. Comput., vol. 24, no. 9, pp. 8780– 8793, 2025.

[8] W. Yi, Y. Fu, J. Cao, L. Gan, L. Xiong, and H. Li, “Towards seamless 6G and AI/ML convergence: Architectural enhancements and security challenges,” IEEE Netw., 2025, to appear.

[9] T. Senevirathna, V. H. La, S. Marcha, B. Siniarski, M. Liyanage, and S. Wang, “A survey on XAI for 5G and beyond security: Technical aspects, challenges and research directions,” IEEE Commun. Surveys Tuts., vol. 27, no. 2, pp. 941–973, 2025.

[10] D. Fan, R. Meng, X. Xu, Y. Liu, G. Nan, C. Feng, S. Han, S. Gao, B. Xu, D. Niyato, T. Q. S. Quek, and P. Zhang, “Generative diffusion models for wireless networks: Fundamental, architecture, and state-of-the-art,” arXiv preprint arXiv:2507.16733, 2025.

[11] G. Sun, W. Xie, D. Niyato, H. Du, J. Kang, J. Wu, S. Sun, and P. Zhang, “Generative AI for advanced UAV networking,” IEEE Netw., 2024, to appear.

[12] Y. Yang, B. Zhang, D. Guo, H. Du, Z. Xiong, D. Niyato, and Z. Han, “Generative AI for secure and privacy-preserving mobile crowdsensing,” IEEE Wireless Commun., vol. 31, no. 6, pp. 29–38, 2024.

[13] W. Zhou, C.-X. Wang, C. Huang, Z. Li, Z. Qian, Z. Lv, and Y. Chen, “Channel scenario extensions, identifications, and adaptive modeling for 6G wireless communications,” IEEE Internet Things J., vol. 11, no. 5, pp. 7285–7308, 2024.

[14] M. Anas, A. Saiyeda, S. S. Sohail, E. Cambria, and A. Hussain, “Can generative AI models extract deeper sentiments as compared to traditional deep learning algorithms?” IEEE Intellig. Syst., vol. 39, no. 2, pp. 5–10, 2024.

[15] C. Zhao, H. Du, D. Niyato, J. Kang, Z. Xiong, D. I. Kim, X. Shen, and K. B. Letaief, “Enhancing physical layer communication security through generative AI with mixture of experts,” IEEE Wireless Commun., vol. 32, no. 3, pp. 176–184, 2025.

[16] C. Jin, Z. Chang, F. Hu, H.-H. Chen, and T. Ham¨ al¨ ainen, “Enhanced¨ physical layer security for full-duplex symbiotic radio with AN generation and forward noise suppression,” IEEE Trans. Commun., vol. 72, no. 7, pp. 3905–3918, 2024.

[17] Y. Zhang, S. Zhao, Y. Shen, X. Jiang, and N. Shiratori, “Enhancing the physical layer security of two-way relay systems with RIS and beamforming,” IEEE Trans. Inf. Forensics Sec., vol. 19, pp. 5696–5711, 2024.

[18] Y. Ju, W. Liu, M. Yang, L. Liu, Q. Pei, N. Zhang, K. Ota, M. Dong, and V. C. M. Leung, “Physical layer security in full-duplex millimeter wave communication systems,” IEEE Trans. Veh. Technol., vol. 73, no. 2, pp. 2816–2829, 2024.

[19] L. Lv, D. Xu, R. Qingyang Hu, Y. Ye, L. Yang, X. Lei, X. Wang, D. In Kim, and A. Nallanathan, “Safeguarding next-generation multiple access using physical layer security techniques: A tutorial,” Proc. IEEE, vol. 112, no. 9, pp. 1421–1466, 2024.

[20] X. Tang, Z. Ma, B. Li, C. Li, Q. Du, D. Niyato, and Z. Han, “Active RIS-aided anti-jamming wireless communications: A Stackelberg game perspective,” IEEE Trans. Commun., vol. 74, pp. 2612–2625, 2026.

[21] Z. Cheng, J. Si, Z. Li, P. Liu, Y. Huang, and N. Al-Dhahir, “Movable frequency diverse array for wireless communication security,” IEEE Trans. Commun., 2025, to appear.

[22] Z. Li, C. Su, Z. Su, H. Peng, Y. Wang, W. Chen, and Q. Wu, “Model predictive control enabled UAV trajectory optimization and secure resource allocation,” IEEE Trans. Commun., vol. 73, no. 11, pp. 12 652–12 665, 2025.

[23] T. M. Hoang, D. Liu, T. V. Luong, J. Zhang, and L. Hanzo, “Deep learning aided physical-layer security: The security versus reliability trade-off,” IEEE Trans. Cog. Commun. Netw., vol. 8, no. 2, pp. 442– 453, 2022.

[24] G. Oligeri, S. Sciancalepore, S. Raponi, and R. D. Pietro, “PAST-AI: Physical-layer authentication of satellite transmitters via deep learning,” IEEE Trans. Inf. Forensics Sec., vol. 18, pp. 274–289, 2023.

[25] X. Tang, K. Zhao, C. Shen, C. Lin, S. Liu, B. Wang, D. Niyato, and Z. Han, “Graph attention network-driven hierarchical learning for anti-jamming UAV communications,” IEEE Trans. Wireless Commun., vol. 25, pp. 5432–5445, 2026.

[26] Z. Song, Y. Lu, X. Chen, B. Ai, Z. Zhong, and D. Niyato, “A deep learning framework for physical-layer secure beamforming,” IEEE Trans. Veh. Technol., vol. 73, no. 12, pp. 19 844–19 849, 2024.

[27] J. Zhang, F. Ardizzon, M. Piana, G. Shen, and S. Tomasin, “Physical layer-based device fingerprinting for wireless security: From theory to practice,” IEEE Trans. Inf. Forensics Sec., vol. 20, pp. 5296–5325, 2025.

[28] D. Guo, J. Xiong, D. Ma, X. Liu, and J. Wei, “Physical layer secret key generation based on mutual information-driven autoencoder,” IEEE Trans. Wireless Commun., 2025, to appear.

[29] C. Zhao, H. Du, D. Niyato, J. Kang, Z. Xiong, D. I. Kim, X. Shen, and K. B. Letaief, “Generative AI for secure physical layer communications: A survey,” IEEE Trans. Cog. Commun. Netw., vol. 11, no. 1, pp. 3–26, 2025.

![](images/5dfb2cb7f52192f2f19ae1113c7b8aa0e49cb96f4f3211d7fadb03d4e46f6e7e.jpg)  
(a) Performance under representative scenarios.

![](images/8503962bc46e59a57f9d87b17349ca0ff119b6d92bb32342430d9e77ee7e3332.jpg)  
(b) Performance under cross-scenarios (interpolated Rician factors).

![](images/d4a71bb1691558b33528b24c1d265a79941c030973b6e0cc7d2bb189f798c6ab.jpg)  
(c) Performance under cross-scenarios (interpolated path loss exponents).

![](images/cd9d7cf8436e411876d88b89c0a7bb4ac57e2c2126b559930e2e299f804f52fd.jpg)  
(d) Performance under cross-scenarios (interpolated path loss exponents and Rician factors).

## Fig. 10. Performance comparison under diverse wireless scenarios.

[30] R. Lin, H. Qiu, J. Wang, Z. Zhang, L. Wu, and F. Shu, “Physical-layer security enhancement in energy-harvesting-based cognitive Internet of Things: A GAN-powered deep reinforcement learning approach,” IEEE Internet Things J., vol. 11, no. 3, pp. 4899–4913, 2024.

[31] R. Meng, X. Xu, B. Wang, H. Sun, S. Xia, S. Han, and P. Zhang, “Physical-layer authentication based on hierarchical variational autoencoder for industrial Internet of Things,” IEEE Internet Things J., vol. 10, no. 3, pp. 2528–2544, 2023.

[32] J. Zhang, Z. Liu, X. Feng, H. Yang, and S. Liang, “Enhanced secure beamforming for IRS-assisted IoT communication using a generativediffusion-model-enabled optimization approach,” IEEE Internet Things J., vol. 12, no. 10, pp. 13 398–13 414, 2025.

[33] C. Zhang, G. Sun, J. Li, Q. Wu, J. Wang, D. Niyato, and Y. Liu, “Multiobjective aerial collaborative secure communication optimization via generative diffusion model-enabled deep reinforcement learning,” IEEE Trans. Mob. Comput., vol. 24, no. 4, pp. 3041–3058, 2025.

[34] J. Wang, H. Du, Y. Liu, G. Sun, D. Niyato, S. Mao, D. In Kim, and X. Shen, “Generative AI based secure wireless sensing for ISAC networks,” IEEE Trans. Inf. Forensics Sec., vol. 20, pp. 5195–5210, 2025.

[35] X. Zhang, G. Li, J. Zhang, L. Peng, A. Hu, and X. Wang, “Enabling deep learning-based physical-layer secret key generation for FDD-OFDM systems in multi-environments,” IEEE Trans. Veh. Technol., vol. 73, no. 7, pp. 10 135–10 149, 2024.

[36] C. Liu, Y. Li, C. Chen, H. Kuang, X. Ma, X. Zou, J. Liu, Z. Lu, Z. Zhang, and X. Liu, “An adaptive and scalable framework for resourceefficient deployment of mixture of experts in LLM-based intelligent IoT networks,” IEEE Internet Things J., vol. 12, no. 13, pp. 22 746–22 757, 2025.

[37] J. He, X. Luo, J. Kang, H. Du, Z. Xiong, C. Chen, D. Niyato, and X. Shen, “Toward mixture-of-experts enabled trustworthy semantic communication for 6G networks,” IEEE Netw., 2024, to appear.

[38] E. Choi, M. Oh, J. Choi, J. Park, N. Lee, and N. Al-Dhahir, “Joint precoding and artificial noise design for MU-MIMO wiretap channels,” IEEE Trans. Commun., vol. 71, no. 3, pp. 1564–1578, 2023.