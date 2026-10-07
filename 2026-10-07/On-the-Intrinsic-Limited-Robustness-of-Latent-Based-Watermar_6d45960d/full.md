# On the Intrinsic Limited Robustness of Latent-Based Watermarking

Cheng-Han Yeh<sup>∗</sup> Kuan-chun Yu<sup>∗</sup> Cheng-Chang Tsai<sup>∗</sup> Chun-Shien Lu Institute of Information Science Academia Sinica Taipei, Taiwan (ROC) {jerryyeh,mikeyu1012,cctsai,lcs}@iis.sinica.edu.tw

## Abstract

Existing latent-based watermarking methods for diffusion models have overestimated their robustness to image distortions, including geometric transformations such as rotation, scaling, and translation (RST). Moreover, this paradigm of watermarking approaches may suffer from inherent limitations arising from the domain in which the watermark is embedded. In this paper, we provide the first theoretical analysis explaining why these methods lack invariance to perturbations. By relaxing the invariant relation, we derive a maximum perturbation bound that characterizes the relationship between pixel-space perturbations and their corresponding effects in latent space. In addition, we present the first analytical formulation that captures all components of practical detection mechanisms. Finally, we conduct experiments to validate the theoretical findings and the limitations of latent-based watermarking methods. Our theoretical and empirical results indicate that, under the current design paradigm, latent-based watermarking methods intrinsically exhibit limited robustness. We conclude by providing the analytical tool and design guidelines that future research could follow.

## 1 Introduction

## 1.1 Background

The dissemination of misinformation or disinformation severely causes financial loss and reputation damage. In particular, due to the maturation of diffusion models [6, 15, 37, 41] in deep learning, the generated (fake) images have been indistinguishable from the natural images, deteriorating the distribution of AI-generated contents (AIGCs) and further obscuring the facts. Several realistic cases indicate that if AIGCs cannot be efficiently detected, not only is intellectual property infringed, but also financial losses may occur, and reputation may be damaged. In addressing these concerns, governments and industries start to take precautionary measures internationally [5, 7, 27, 43]. These policies clearly suggest to employ digital watermarking technologies for AIGCs detection and tracing, revealing its importance.

## 1.2 Motivation

Watermarking methods [4, 10, 48] on AIGCs have been proposed, to name a few, but can only claim robustness against limited types of perturbations with a limited set of parameters. This obscures the potential risk, i.e., insufficient robustness, of these well-known watermarking schemes. Specifically, we are curious about the factors, including watermark patterns, the shift in latent distribution, etc, that affect the robustness claimed by these methods.

To answer this question, we establish a new formulation that combines the RST framework [38] and the diffusion model [19]. We prove that the inverse diffusion process lacks invariance and robustness to perturbations in the pixel space. Next, by relaxing the detection criterion, we extend this analysis and derive an upper bound that characterizes pixel-space perturbations induced by variations around the watermark. The results show that sufficiently shifting the mean of the standard normal distribution via watermark injection improves robustness to distortions in the corresponding images. In addition, we propose the first analytical form to characterize the components used in latent-based watermarking. This form also delineates the relationships between the detection mechanism, the setting of False Positive Rate (FPR), and different perturbations beyond RST.

In summary, our contributions are fivefold:

1. We are the first to bridge Vector Approximate Message Passing [33] with DDIM reverse process, deliver the proof to invert the reverse process, and give the closed form of the Lipschitz constant of DDIM reverse and inverse process.

2. We identify the factors that influence the robustness of watermarking on the latent representation, and explain the lack of invariance to perturbations.

3. We derive the upper bound of the perturbation, measured by the $\ell _ { p }$ -norm, for a watermarked image, and show that even if such an image is perturbed, it could still be recognized by the detection algorithm.

4. We provide the first analytical form to describe the robustness of a watermarking scheme, involving the design of watermark, perturbations, setting of False Positive Rate (FPR), and Lipschitz constant of perturbations.

5. Our findings, supported by both theoretical and experimental evidences, highlight a key limitation in that latent-based watermarking schemes may lack robustness under the current design paradigm.

## 2 Related Work

An increasing number of studies have been proposed for generative watermarking in the latent space of diffusion models, which require the inverse process of DDIM [41] to return to the time step at which the watermarks are embedded in that the image generation and watermarking processes are coupled together. The concept of generative watermarking or in-process watermarking<sup>2</sup> was first proposed in Tree-Ring [48].

Although Tree-Ring demonstrates somewhat robustness in verifying the existence of a watermark, its performance deteriorates significantly in watermark identification due to the inherent low-capacity property. To address this issue, RingID [4] proposes using a more meticulously designed pattern as the watermark, along with a multi-channel heterogeneous watermarking framework, to achieve superior robustness and identification capability. Unlike Tree-Ring and RingID, which use watermarks with designed patterns for specific purposes, Gaussian Shading [52] uses a binary sequence as its watermark. In addition, unlike Tree-Ring and RingID, which both cause the distribution of initial noises to deviate from the standard Gaussian distribution, Gaussian Shading and its variant [25] map the watermark to the initial noise sampled from the standard Gaussian distribution.

In contrast to the aforementioned works that embed watermarks into the initial noise, ROBIN [16] embeds the watermark into the latent at an intermediate time step, whereas Latent Watermark [26] embeds the watermark immediately before decoding. Finally, ZoDiac [54] employs a pre-trained stable diffusion [37] model to embed the watermark into the trainable latent space.

Another line of research explores content-dependent watermark (CDW). This concept was first introduced in [21] to counter watermark estimation attacks (WEAs), which are used in collusion and copy attacks [44]. Specifically, CDW comprises an informative watermark that carries information about the owner and an image/content hash that represents the cover carrier, i.e., content-dependent information [21]. For watermarking on AIGCs, Yang et al. [51] proposes averaging over a certain number of images to extract an additive pattern, which can be used for watermark forgery and removal.

To defend against this attack, Yang et al. [51] also suggests that content-adaptive watermarking methods, which employ the same underlying concept as CDW, demonstrate robust resistance.

## 3 Insufficient Robustness in Latent-Based Watermarking

In this section, we investigate the limited robustness of latent-based watermarking methods. First, we introduce the notations used in our analysis in Sec. 3.1. Second, we provide the analytical tool to explain the reverse and inverse processes of DDIM from the Vector Approximate Message Passing (VAMP) [33] and MMSE denoiser perspective in Sec. 3.2. Third, we examine the perturbation invariance of latent-based watermarking [4, 11, 16, 34, 48, 50, 52, 54, 56] in Sec. 3.3. Then, as mentioned in Sec. 1.2, by relaxing the invariant relation, we provide a framework describing the relationship between the perturbations on pixel space and the corresponding effects on latent space of diffusion models, and give insights regarding the tolerance $( i . e .$ , the maximum perturbation on the image) and the resistance $( i . e .$ , the upper bound of the Lipschitz constants of perturbations that a watermarking scheme can resist) of the latent-based watermarking methods against more severe perturbations in the pixel space in Sec. 3.4. Finally, we discuss the effects of thresholds and latent distribution shift of these methods in Sec. 3.5. For the background relevant to our analysis, please refer to Sec. A of Appendix.

## 3.1 Notation

Throughout this paper, $\mathbf { I } \in \mathbb { R } ^ { C \times H \times W }$ denotes an image, $\mathbf { x } _ { \mathrm { 0 } }$ represents the latent representation at time step $0 ,$ and $\mathbf { y }$ is the corresponding latent representation at time step $T ,$ where $\mathbf { x } _ { 0 } , \mathbf { y } \in \mathbb { R } ^ { n }$ . ∥ · ∥ represents the $\ell _ { p }$ norm, $p \geq 1$ , unless specifically stated. Id denotes the identity matrix. $D _ { \theta }$ is the minimum mean-squared error (MMSE) denoiser [28, 33], parameterized by $\theta .$ . We use $\epsilon _ { \theta }$ to represent the noise prediction network used in [41] and parameterized by θ. We denote the encoder and decoder used in [37] as $\mathcal { E } _ { \phi _ { \epsilon } }$ and $\mathcal { G } _ { \phi _ { g } } .$ , where $\mathbf { I } = \mathcal { G } _ { \phi _ { g } } ( \mathbf { x } _ { 0 } )$ and $\mathbf { x } _ { 0 } = \mathcal { E } _ { \phi _ { e } } ( \mathcal { G } _ { \phi _ { g } } ( \mathbf { x } _ { 0 } ) )$ . We use ${ \mathcal { R } } , { \mathcal { S } }$ , and $\mathcal { T }$ to denote the rotation, scaling, and translation operations, respectively. $\mathcal { F }$ and $\mathcal { F } _ { \mathcal { M } }$ represent the Fourier and Fourier–Mellin transforms, respectively. λ(A) collects eigenvalues of a square matrix A.

## 3.2 Formulation of DDIM Reverse and Inverse Process

Recently, diffusion models and their corresponding denoisers used in the reverse process have been studied in [19] to analyze the inductive biases of the network, under the assumption that the denoiser is a “piecewise linear, bias-free neural network”. Given a denoiser and a noisy input image, prior works [19, 29, 36] represent the denoised image as the product of the Jacobian matrix of the denoiser evaluated at the noisy input image and the noisy input image itself. Although these assumptions make the analysis more concise, they also restrict its applicability and practicality.

In our analysis, a $T \cdot$ -step reverse process $f _ { \theta }$ can be represented as an unrolling VAMP using the MMSE denoiser $D _ { \theta }$ without the above assumption. Specifically, we have the following Theorem.

Theorem 3.1. Given an MMSE denoiser $D _ { \theta }$ and an induced T-step DDIM reverse process $f _ { \theta } , f _ { \theta }$ is globally invertible and expressed as:

$$
\mathbf { y } = f _ { \theta } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } ) = f _ { \theta } ^ { - 1 } ( f _ { \theta } ( \mathbf { y } ) ) ,\tag{1}
$$

where $\hat { \mathbf { x } } _ { 0 }$ is the denoised representation given the latent representation y. At the same time,for all $\tilde { \mathbf { y } }$ in latent space and the corresponding denoised representation $\tilde { \mathbf { x } } _ { 0 } , \tilde { \mathbf { x } } _ { 0 } = f _ { \theta } ( \tilde { \mathbf { y } } ) = f _ { \theta } ( f _ { \theta } ^ { - 1 } ( \tilde { \mathbf { x } } _ { 0 } ) )$ .

The detailed derivations are in Sec. B and Sec. C of Appendix.

## 3.3 Limited Perturbation Invariance in Latent-based Watermark

We begin by analyzing the lack of invariance of the composition of RST transformations, which is formalized in the following theorem. The proofs of the following theorems are in Sec. C of Appendix.

Theorem 3.2. Given an invertible $f _ { \theta }$ defined in Eq. $( I ) ,$ , an RST-perturbed representation $\hat { \mathbf { x } } _ { 0 } ^ { R S T }$ , and its original counterpart $\hat { \mathbf { x } } _ { 0 }$ with $\hat { \mathbf { x } } _ { 0 } ^ { \check { R } \check { S } T } \neq \hat { \mathbf { x } } _ { 0 } ,$ , we have

$$
\begin{array} { r } { [ \mathcal { F _ { M } } \circ \mathcal { F } ] \mathbf { y } ^ { R S T } \neq [ \mathcal { F _ { M } } \circ \mathcal { F } ] \mathbf { y } , } \end{array}\tag{2}
$$

where $\mathbf { y } ^ { R S T }$ and y are the latent representations $o f \hat { \mathbf { x } } _ { 0 } ^ { R S T }$ and $\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } , } \end{array}$ , respectively. In other words, the combined process, i.e., the DDIM inversion process operating on an RST-transformed input, does not possess RST invariance, which results in the inequality ofthe transformed outputs in the joint Fourier and Fourier-Mellin domains.

Theorem 3.2 shows that even after transforming the latent representations, $\mathbf { y } ^ { R S T }$ and $\mathbf { y } ,$ into the Fourier domain, the results are not equal. To reach equality, it should satisfy the assumption called “fragility” in that no perturbation is allowed between either $\hat { \mathbf { x } } _ { 0 } ^ { R S T }$ and $\hat { \mathbf { x } } _ { 0 }$ or $\mathbf { y } ^ { R S T }$ and $\mathbf { y }$

Furthermore, we extend our analysis to perturbations beyond RST transformations and formalize the corresponding result in the following theorem.

Theorem 3.3. If the output of $f _ { \theta } ,$ defined in Theorem $3 . I ,$ is non-RST invariant, then the output is also not invariant to any perturbations.

## 3.4 Maximum Perturbation on Image and Resilience against Perturbations

Different from the fragility assumption made in Theorems 3.2 and 3.3, here we derive the upper bound of the maximum allowed perturbation on images for robustness purposes, given a certain tolerance in terms of distance in latent space. Specifically, all images within such a bound can go through DDIM inversion to obtain latent representations close to the unperturbed one under a certain distance metric.

Given the Lipschitz constants of DDIM reverse and inverse processes at each time step provided in Sec. B.2 of Appendix, we can derive the Lipschitz constants of T-step reverse and inverse processes.

Theorem 3.4. Let $f _ { \theta } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ be the generative process and $f _ { \theta } ^ { - 1 } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ be the inverse process. The global Lipschitz bounds with monotonically decreasing noise schedule $\left\{ \alpha _ { t } \right\} _ { t = 0 } ^ { T } a r e \colon$

$$
G e n e r a t i o n . \quad L _ { f _ { \theta } } \leq \frac { \sqrt { \alpha _ { 0 } } } { \sqrt { { \bar { \alpha } _ { T } } } } ,
$$

$$
I n \nu e r s i o n \mathrm { : } \quad L _ { f _ { \theta } ^ { - 1 } } \leq \frac { \sqrt { 1 - \bar { \alpha } _ { T } } } { \sqrt { 1 - \alpha _ { 0 } } } ,
$$

where $\textstyle { \bar { \alpha } } _ { t } : = \prod _ { s = 0 } ^ { t } \alpha _ { s }$ . The detailed proof is in Sec. C of Appendix.

Next, for the operations commonly used in the generation and detection of latent-based watermarks, we define the following operators: W injects a w in a latent representation y; M denotes a masking operation used to extract the watermark pattern; and $\mathcal { T } \in { S } _ { \mathcal { T } }$ represents an image transformation from a set of operator $S _ { T }$ , including geometric, non-geometric, and model-based perturbations under different parameter settings. Note that we treat different parameter settings of a perturbation as distinct operators in $S _ { T }$ . Additionally, we denote the Lipschitz constants of $\mathcal { F } ^ { - 1 } , \dot { \mathcal { F } } , \mathcal { W } , \mathcal { M } , \mathcal { E } _ { \phi _ { e } }$ $\mathcal { G } _ { \phi _ { g } }$ , and $\bar { \boldsymbol { \tau } } ,$ , as $L _ { \mathcal { F } ^ { - 1 } } , L _ { \mathcal { F } } , L _ { \mathcal { W } } , L _ { \mathcal { M } } , L _ { { \mathcal { E } } _ { \phi _ { e } } } , L _ { { \mathcal { G } } _ { \phi _ { g } } }$ , and $L _ { T }$ , respectively.

Now, we can define the image generation process from a watermarked latent. Given a latent y sampled from standard normal distribution, the watermarked image I can be generated as:

$$
\begin{array} { r } { \mathbf { I } = \mathbf { F } ( \mathbf { y } , \mathbf { w } ) \stackrel { \mathrm { d e f } } { = } \mathcal { G } _ { \phi _ { g } } ( f _ { \theta } ( \mathcal { F } ^ { - 1 } ( \mathbf { y } _ { \mathbf { w } } ) ) , } \\ { \mathbf { y } _ { \mathbf { w } } = \mathcal { W } ( \mathbf { y } , \mathbf { w } ) \stackrel { \mathrm { d e f } } { = } \mathcal { F } ( \mathbf { y } ) \odot \mathbf { 1 } _ { \mathcal { M } ^ { c } } + \mathbf { w } , } \end{array}\tag{3}
$$

where $\mathbf { 1 } _ { ( \cdot ) }$ is an indicator function, ⊙ is Hadamard product, $\mathbf { y } _ { \mathbf { w } }$ is the watermarked latent, and $\mathbf { w } = \mathbf { W } \mathbf { \operatorname { \odot } } \mathbf { 1 } _ { \mathcal { M } }$ for some $\mathbf { W } \in \mathbb { C } ^ { n }$ denoting the ground-truth watermark pattern. In Eq. (3), as W injects a pattern into $\mathbf { y } ,$ we formulate the Lipschitz constant of W with image size of $H \times W$ as $L _ { \mathcal { W } } = \sqrt { H W }$ . Similarly, we can formally define a mapping from a watermarked image I to the representation employed by the watermark detection algorithm as:

$$
\hat { \mathbf { w } } = \mathbf { F } ^ { - 1 } ( \mathcal { T } ( \mathbf { I } ) ) \stackrel { \mathrm { d e f } } { = } \mathcal { M } ( \mathcal { F } ( f _ { \theta } ^ { - 1 } ( \mathcal { E } _ { \phi _ { e } } ( \mathcal { T } ( \mathbf { I } ) ) ) ) ,\tag{4}
$$

where wˆ is the estimated watermark from the distorted image $\mathcal { T } ( \mathbf { I } )$

Due to numerical errors, asymmetric guidance scales, and value clipping during reverse and inverse processes, the invariant relation, as depicted in Theorem $3 . 3 ,$ is highly unachievable even if perturbations are not present. Therefore, we relax the invariant relation and define a scalar r to capture these practical effects. We examine these practical effects in Sec. 4.2.

Table 1: Comparison of different watermarking schemes with operations involved in the generation and detection processes. ∗ denotes the use of the noise layer for fine-tuning the network model or extra modules. † denotes the extra module for compensating for or correcting the errors caused by perturbations. $\triangle$ denotes the need for modifications of the reverse process. The “× number” indicates the number of modules used in the generation or detection process.
<table><tr><td>Watermarking Methods</td><td>MIW</td><td> $\mathcal { F } ^ { - 1 } \ d s / \mathcal { F }$ </td><td> $f _ { \theta } / f _ { \theta } ^ { - 1 }$ </td><td> $\mathcal { G } _ { \phi _ { g } } \ : / \mathcal { E } _ { \phi _ { e } }$ </td></tr><tr><td>Tree-Ring (TR) [48] RingID [4]</td><td>√1√  $\checkmark / \checkmark$ </td><td> $\eqslantless / \checkmark$ </td><td> $\overline { { \curvearrowright \subset \mathop { \egroup } } } \sqrt { \mathbin { \updownarrow } }$ </td><td> $\overline { { \curvearrowright \subset \mathop { \egroup } } }$ </td></tr><tr><td>HSQR/TR [20]</td><td> $\checkmark / \checkmark$ </td><td> $\checkmark / \checkmark$   $\checkmark / \checkmark$ </td><td> $\checkmark / \checkmark$   $\checkmark / \checkmark$ </td><td> $\checkmark / \checkmark$   $\checkmark / \checkmark$ </td></tr><tr><td>Gaussian Shading (GS) [52]</td><td> $\times / \checkmark$ </td><td> $\times / \times$ </td><td> $\checkmark / \checkmark$ </td><td> $\checkmark / \checkmark$ </td></tr><tr><td>PRC Watermark (PRCW) [12]</td><td></td><td></td><td></td><td> $\checkmark / \checkmark$ </td></tr><tr><td>WIND [2]</td><td> $\times / \checkmark$ </td><td> $\times / \times$ </td><td> $\checkmark / \checkmark$ </td><td></td></tr><tr><td></td><td> $\times / \checkmark$ </td><td> $\times / \times$ </td><td> $\checkmark / \checkmark$ </td><td> $\checkmark / \checkmark$ </td></tr><tr><td>TR-SynTag [8]</td><td> $\eqslantless / \checkmark$ </td><td> $\eqslantless / \checkmark$ </td><td> $\overline { { \textsf { V } / \textsf { V } ( \textsf { f } \times 2 ) } }$ </td><td> $\overline { { \textsf { V } / \textsf { V } ( \dagger ) } }$ </td></tr><tr><td>GS-SynTag [8]</td><td> $\times / \checkmark$ </td><td> $\times / \times$ </td><td> $\surd / \surd ( \div \times 2 )$ </td><td></td></tr><tr><td>TR-CoSDA [9]</td><td> $\checkmark / \checkmark$ </td><td> $\checkmark / \checkmark$ </td><td></td><td> $\surd / \surd ( \dagger )$ </td></tr><tr><td>GS-CoSDA [9]</td><td> $\times / \checkmark$ </td><td></td><td> $\surd / \surd ( * , \triangle )$ </td><td> $\checkmark / \checkmark$ </td></tr><tr><td>MaXsive [25]</td><td> $\times / \checkmark$ </td><td> $\times / \times$ </td><td> $\checkmark / \checkmark ( * , \bigtriangleup )$ </td><td> $\checkmark / \checkmark$ </td></tr><tr><td></td><td></td><td> $\times / \times$ </td><td> $\surd / \ V \ ( \dagger , \triangle )$ </td><td> $\checkmark / \checkmark$ </td></tr><tr><td>WMAdaptor [3]</td><td> $\times / \times$ </td><td> $\times / \times$ </td><td> $\times ~ / ~ \times$ </td><td> $\checkmark / \checkmark \left( * \right)$ </td></tr><tr><td>LaWa [35] Stable Signature [10]</td><td> $\times / \times$ </td><td> $\times / \times$ </td><td> $\times ~ / ~ \times$ </td><td> $\checkmark / \checkmark \left( * \right)$ </td></tr><tr><td>HiDDeN [55]</td><td> $\times / \times$ </td><td> $\times / \times$ </td><td> $\times ~ / ~ \times$ </td><td> $\checkmark / \check { \checkmark } \left( \ast \right)$ </td></tr><tr><td></td><td> $\times / \times$ </td><td> $\times / \times$ </td><td> $\times ~ / ~ \times$ </td><td> $\checkmark / \check { \check { \mathbf { \Xi } } } \big ( \dot { * } \big )$ </td></tr></table>

With these practical considerations, we define the ball of radius r centered at w as $B _ { r } ( \mathbf { w } ) = \{ \mathbf { w } ^ { \prime }$ $\| \mathbf { w } - \mathbf { w } ^ { \prime } \| \leq r \}$ . Our goal below is to demonstrate the relationship between the distortions within the ball $B _ { r } ( \mathbf { w } )$ in the latent space and the upper bound of corresponding perturbations in the pixel space.

Theorem 3.5. Given a watermarked image I generatedfrom w and y $\sim \mathcal { N } ( \mathbf { 0 } , \mathrm { I d } )$ , and a ball $B _ { r } ( \mathbf { w } )$ if the watermarked image I and its perturbed counterpart $\mathcal { T } ( \mathbf { I } )$ satisfy

$$
\forall \mathbf { I } , \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| \leq L _ { \mathbf { F } } r ,
$$

then for any watermarked images, $\mathbf { I } = \mathbf { F } ( \mathbf { y } , \mathbf { w } )$ and $\mathbf { I } ^ { \prime } = \mathbf { F } ( \mathbf { y } , \mathbf { w } ^ { \prime } )$ , with $\mathbf { w } ^ { \prime } \in \mathcal { B } _ { r } ( \mathbf { w } )$ , we have:

$$
\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| \leq L _ { \mathbf { F } ^ { - 1 } } ( 1 + L \tau ) L _ { \mathbf { F } } r ,\tag{5}
$$

where $\hat { \mathbf { w } } ^ { \prime }$ is the estimated watermark $o f \mathcal { T } ( \mathbf { I } ^ { \prime } )$ , obtained from Eq. (4), $L _ { \mathbf { F } } = L _ { \mathcal { G } _ { \phi _ { a } } } L _ { f _ { \theta } } L _ { \mathcal { F } ^ { - 1 } } L _ { \mathcal { W } }$ , and $L _ { \mathbf { F } ^ { - 1 } } = L _ { \mathcal { E } _ { \phi _ { e } } } L _ { f _ { \theta } ^ { - 1 } } L _ { \mathcal { F } } L _ { \mathcal { M } }$

The derivation is in Sec. C of Appendix. For $\hat { \mathbf { w } } ^ { \prime }$ to be identified as the embedded watermark under a given FPR and corresponding threshold R, we need $\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| \leq R$ , where $r < R$ . To clarify the difference between R and r, R describes the threshold of a given FPR that encloses the distance between the watermark w and any perturbed watermark $\hat { \mathbf { w } } ^ { \prime }$ , whereas r is the upper bound of the distance between any two unperturbed watermarks w and $\mathbf { w } ^ { \prime }$ , and the distance between w and its perturbed counterpart wˆ .

To connect R and Eq. (5) for analyzing the robustness of watermark w, we first look at the Lipschitz constants of each component used in the reverse and inverse processes. Actually, the Lipschitz constant $L _ { \mathcal { M } }$ of masking operation $\mathcal { M }$ is less than 1. However, from Theorem $3 . 4 .$ , we know that the reverse and inverse processes are expansive mappings, with both $L _ { f _ { \theta } }$ and $L _ { f _ { \theta } ^ { - 1 } }$ being larger than 1.

Finding 3.6. Suppose that an image is attacked by two perturbations, including $\tau _ { \mathrm { m i n } } .$ , which has the smallest Lipschitz constant $L _ { T _ { \mathrm { m i n } } }$ among $\scriptstyle { \mathcal { S } } _ { T }$ , and $\tau _ { \mathrm { m a x } } .$ , which has the largest Lipschitz constant $L _ { \mathcal { T } _ { \mathrm { m a x } } } > L _ { \mathcal { T } _ { \mathrm { m i n } } }$ among $S _ { T }$ . We then can derive from Eq. (5) to have:

$$
\begin{array} { r } { \vert \vert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \vert \vert \leq ( 1 + L _ { \mathcal { T } _ { \mathrm { m i n } } } ) L _ { \mathbf { F } ^ { - 1 } } L _ { \mathbf { F } } r \leq ( 1 + L _ { \mathcal { T } _ { \mathrm { m a x } } } ) L _ { \mathbf { F } ^ { - 1 } } L _ { \mathbf { F } } r . } \end{array}\tag{6}
$$

Finding 3.7. Eq. (6) shows that whether a perturbed image $\mathcal { T } ( \mathbf { I ^ { \prime } } )$ can be identified as a watermarked image depends on $\mathcal { M } , \mathcal { W }$ , and R. That is, given a diffusion model, M, w, and $R ,$ the watermark is robust against any perturbation $\mathcal { T } \in S _ { \mathcal { T } }$ whose Lipschitz constant is at most $\begin{array} { r } { \frac { R } { L _ { \mathbf { F } ^ { - 1 } } L _ { \mathbf { F } ^ { r } } } - 1 } \end{array}$ such that

$$
\begin{array} { r } { \| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| \leq ( 1 + L _ { { \mathcal { T } _ { \operatorname* { m i n } } } } ) L _ { \mathbf { F } ^ { - 1 } } L _ { \mathbf { F } } r \leq ( 1 + L _ { \mathcal { T } } ) L _ { \mathbf { F } ^ { - 1 } } L _ { \mathbf { F } } r \leq R . } \end{array}\tag{7}
$$

From Eq. (7), the impact of perturbations depends on the modules used in F and $\mathbf { F } ^ { - 1 }$ . In particular, expansive modules employed to extract the estimated watermark degrade watermarking robustness. Constraining these modules to be non-expansive in diffusion models, however, restricts the diversity of the generated samples, revealing an inherent trade-off in latent-based watermarking.

Remark 3.8. Latent-based watermarking uses expansive modules that amplify perturbations in the pixel space, whereas using non-expansive modules results in generated samples with limited diversity.

The ratio between R and r further limits robustness under a low FPR, which requires a smaller R, as discussed in Sec. 3.5 below. When $R \leq r$ , the detector inevitably classifies a non-perturbed watermarked image as unwatermarked based on the estimated watermark. This behavior highlights a fundamental limitation of latent-based watermarking in practical settings.

Remark 3.9. The factor r is always present due to inevitable errors in diffusion models. When the latent-based watermark is used in real-world applications, R cannot be smaller than r.

Design Guideline Based on our theoretical and empirical analyses, we fundamentally question the long-term viability of latent-based watermarking paradigm for AI-generated content. Our findings indicate that any robust watermarking scheme must navigate three inherent mathematical constraints— all of which latent-based methods structurally violate.

First, the reverse and inverse processes of diffusion models lack perturbation invariance, rendering their latent representations highly sensitive to even minor spatial distortions. Second, the expansive nature of these processes (with Lipschitz constants $L > 1 )$ mathematically guarantees that any pixelspace attack or numerical error will be significantly amplified during latent extraction. Consequently, both injection and detection must bypass such expansive modules. Third, latent-based methods face an inherent trade-off: injecting a more robust signal requires shifting the model’s sampling distribution, which consequently degrades generation quality and increases the FID score [14].

In light of these fundamental bottlenecks, the continued pursuit of latent-based watermarking appears mathematically counterproductive. A practical alternative is readily available: several existing post-hoc watermarking methods [13, 17, 42, 46, 49, 53] already meet these rigorous guidelines. By operating entirely outside the sensitive latent space and avoiding expansive diffusion modules, these existing post-hoc schemes naturally bypass the aforementioned vulnerabilities, ensuring high detection reliability while strictly adhering to prescribed fidelity constraints (e.g., PSNR or FID).

We provide a comparison of different watermarking methods that describe all the operations involved in the generation and detection processes in Table 1. In the upper block of Table 1, we include the methods that do not train and fine-tune any components, and embed a designed watermark pattern in the latent representation at time step T. The lower block indicates methods that introduce extra modules, modify the sampling process $f _ { \theta } ,$ or leverage the noise layer to improve the robustness of watermarking. However, none of the methods in Table 1 satisfy the aforementioned guidelines.

## 3.5 Effects of R and Distribution Shift in Latent Space

We can see that setting R is closely related to Eq. (5) and Eq. (7), which determines the robustness of the watermarking scheme against perturbations. Hence, we discuss the effect of R and its connection with the False Positive rate (FPR). Suppose we have two datasets: one with watermarked images $\mathcal { D } _ { w } \backslash$ the other with unwatermarked images D. According to [1], the detection threshold was set based on the ratio of the number of false positive examples to the number of samples in $\mathcal { D } .$

Specifically, since unwatermarked images are inverted by $\mathbf { F } _ { \theta } ^ { - 1 }$ into the latent space, this induces a distance between the estimated watermark of sample from $\mathrm { \Delta } { \bf \bar { \mathcal { D } } }$ and watermark w. As the number of samples in $\mathcal { D }$ increases, the distribution of distances between the unwatermarked latents and ground-truth watermark w converges to a non-central χ distribution. Hence, we can set the threshold R such that the ratio of area satisfying $\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| \leq \dot { R }$ to area under this non-central $\chi$ distribution equals the desired FPR. The condition $\left\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \right\| \leq R$ thus serves as the detection criterion.

Given R and based on Eq. (27) in Sec. C of Appendix, increasing the upper bound $L _ { \mathbf { F } } r$ around the image I, within which the bound covers more images and these images can be projected back to $B _ { r } ( \mathbf { w } )$ , provides a wider area to tolerate perturbations. Thus, according to Eq. (27), it is possible to increase the value of r or norm of w in the latent space, which correspond to relaxing the criterion $\lVert \mathbf { w } - \hat { \mathbf { w } } \rVert$ or increasing the strength of the injected w, respectively.

To reveal the importance of r, we vary it from 0 to $\infty ,$ , allowing us to explore similarity based on the distance between w and inversed latent representation $\mathbf { w } ^ { \prime }$ , as defined in Eq. (27). In fact, setting r is equivalent to defining the region in which latent representations lie within a distance r of w. If it is set to ∞, every latent representation is trivially considered close to w, which can result in high FPRs. Interestingly, this condition occurs when channels are masked by M. For example, Tree-Ring [48] and RingID [4] only consider the difference in one channel while discarding the rest of the channels. Hence, even if the rest of the channels differ significantly, as long as the difference in the selected dimension remains within the constraint, the data is still regarded as being watermarked. Please also refer to the visual explanations and the discussions of $B _ { r } ( \mathbf { w } )$ in Sec. D of Appendix. Conversely, if it equals 0, indicating $r = 0$ without allowing any tolerance, which depicts the “fragility” scenario discussed in Sec. 3.3.

On the other hand, by moving w to another position, where $\| \mathbf { w } \|$ increases greatly while not increasing r, this does not enlarge the maximum perturbation bound of I accordingly, yet corresponds to a larger allowed perturbation in the pixel space. Adding a shift on each latent representation not only moves the normal distribution $\mathcal { N } ( \mathbf { \dot { 0 } } , \mathbf { I } ) \mathrm { ~ t o ~ } \mathcal { N } ( \mu _ { \mathbf { w } } , \Sigma _ { \mathbf { w } } )$ ) but also moves a set of y’s to an area that induces a larger norm for every $\mathbf { y _ { w } } \in \mathcal { N } ( \mu _ { \mathbf { w } } , \pmb { \Sigma _ { w } } )$ , which creates a larger “buffer” between $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and $\mathcal { N } ( \mu _ { \mathbf { w } } , \pmb { \Sigma } _ { \mathbf { w } } )$ that accommodates perturbed latent representations. As for moving to near the normal distribution, which shrinks the “buffer,” there are higher chances that the latent representations are from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Hence, it is argued that the robustness of latent-based watermarking comes from shifting the normal distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ to $\mathcal { N } ( \mu _ { \mathbf { w } } , \pmb { \Sigma } _ { \mathbf { w } } )$ , as shown in Fig. 6 in Sec. E of Appendix. We also provide an alternative explanation of the detection mechanism from the perspective of forgery and removal attacks in Sec. F of Appendix.

## 4 Experiments

In Sec. 3, our theoretical analysis shows that the latent space of diffusion models is not invariant to perturbations in the pixel space. We provide Eqs. (6) and (7) in Sec. 3.4 to describe distortions in the latent space stemming from perturbations in the pixel space. To examine our theoretical findings, we provide empirical results using different perturbations and parameter settings on existing watermarking schemes. Our goal is to reveal why latent-based watermarking methods require auxiliary modules to withstand perturbations and identify the components that limit their robustness. Therefore, we selected some of the methods in the upper block of Table 1 as our evaluation targets. The implementation details are first described in Sec. 4.1. In Secs. 4.2, 4.3, and 4.4, we show that Eq. (7) provides explanations regarding the effects of perturbations and Lipschitz constant $L _ { T } ,$ , mask M, and watermark pattern w on the distance $\lVert \mathbf { w } - \bar { \hat { \mathbf { w } } } ^ { \prime } \rVert$ . Also, we can observe the roles of r and R mentioned in Secs. 3.4 and 3.5.

## 4.1 Implementation Details

All experiments were conducted in Docker containers, each equipped with an Intel<sup>®</sup> $\mathbf { X e o n } ^ { \mathbb { B } }$ Gold 6154 CPU and a single NVIDIA Tesla V100-SXM2-32GB GPU. The implementations used the Diffusers [45] package from Hugging Face<sup>3</sup>, along with the codebases from Tree-Ring [48], RingID [4], PRCW [12], and HSTR/QR [20]. We adopted Stable Diffusion v2-1 [37] from Diffusers. The prompt dataset is the stable-diffusion-prompts from Gustavosta<sup>4</sup>. The image distortions were implemented with Pillow<sup>5</sup>, torchvision [24] , and Stirmark [32].

To investigate how the strength of attacks affects the detection, each attack was conducted with five parameter settings. Additionally, we included two types of regeneration from WAVES [1]: Regen-VAE and Regen-DiffPure. See Sec. G of Appendix for detailed descriptions. Due to limited space, we present five perturbations, each with three different parameter settings. Please refer to Secs. H, I, and J of Appendix for complete experiments.

For experiment settings, we set the ring radius as 0 ∼ 10 for Tree-Ring [48] and 3 ∼ 14 for RingID [4]. The reverse process used 50 steps to generate watermarked and unwatermarked images with guidance scales set to 7.5. We followed the settings in [4, 12, 20, 48] and sampled 1,000 pairs of these images to compute the detection thresholds of 1e-2 in terms of FPR. Before watermark detection, we utilized the DDIM inversion [6, 41, 48] to execute the inverse process with the same 50 steps from images to latent representations with guidance scales set to 1.0, which followed the same settings in [16, 48].

![](images/03300f20c7c5c30c60e191bf362d086a3e2f3d3576f2d899d6a2f64237c997ce.jpg)  
Figure 1: Effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . From left to right, the perturbations were set with different parameters (Param). Watermark: Tree-Ring. Thresholds: 77.12 (1e-2 FPR).

## 4.2 Effects of perturbations on detection distance

In this experiment, we analyzed the implications of Eq. (7) in that the maximum perturbed distance $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ can be ordered by the magnitudes of applied perturbations. We used Tree-Ring [48] and HSQR [20] as representative examples, with the corresponding results presented in Figs. 1 and 2. For PRCW [12], see Figs. 10, 11, 12, and 13 in Sec. H of Appendix.

In these figures, the orange distribution denotes the distance between the watermarked latent and ground-truth watermark, the blue distribution corresponds to the distance between the unwatermarked latent and ground-truth watermark, and the red distribution represents the distances between the perturbed watermarked latent and ground-truth watermark. The red dotted lines indicate the detection threshold R at FPRs of 1e-2. Ideally, the orange distribution should be concentrated at 0. However, numerical errors, different guidance scales, stochastic sampling effects, and value clipping introduced during the reverse and inverse processes cause the estimated watermark to deviate from w, resulting in a nonzero shift of $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . These deviations, captured by the term r, are explicitly accounted for in our derivation in Sec. 3.4. We also present experiments of these deviations on different watermarking schemes in Table 2 and Fig. 14 in Sec. H of Appendix. It shows that as the number of time steps t increases, the mean value and maximum of a set of $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ are slightly reduced. This provides a remedy to reduce the numerical errors by the number of time steps t so that the orange distribution shifts towards 0 and leaves more distance for unknown perturbations.

It can be seen from Figs. 1 and 2 that increasing perturbation strength leads to progressively larger shifts of the red distributions. If the stricter threshold is adjusted from the red dotted line to the left of the red dotted line, several perturbations are observed to push the red distributions beyond the stricter detection thresholds. According to Eq. (7), a lower FPR setting, corresponding to a stricter threshold, excludes more perturbations than a looser threshold, thereby reducing the watermark robustness. However, in practice, a watermarking scheme should be robust to perturbations even under a lower FPR setting.

## 4.3 Effects of Masking M and Watermark w

We can observe from Figs. 1 and 2 that HSQR [20] is vulnerable to rotation but Tree-Ring [48] is not. The difference comes from the design of mask M and watermark w. According to HSQR [20], it utilizes a square mask and injects a QR-code-like pattern as the watermark. We hypothesize that the vulnerability comes from the desynchronization of the extracted watermark. However, even if the mask and pattern are circular and ring-like in Tree-Ring [48], we further verified Theorems 3.2 and 3.3 in Sec. 3.3. In Fig. 1, the perturbation was set to rotate an image by 5 degrees, but the red distribution shifts from the orange distribution. Hence, we can conclude that theoretically and empirically, the diffusion process, unlike the Fourier and Fourier–Mellin transforms, is not rotation-invariant. Moreover, according to these experiments, the diffusion process is not invariant to perturbation. In Sec. I of Appendix, according to our reasoning about $\| \mathbf { w } \|$ and the corresponding “buffer” mentioned in Sec. 3.5, we verify such an effect of moving a distribution toward the one with a larger norm by increasing the strength of injected watermarks.

![](images/d8ecb50bae8db73315952e7040edc3ad181a5d27b7ea7ad20d34a0f9077404db.jpg)  
Figure 2: Effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . From left to right, the perturbations were set with different parameters (Param). Watermark: HSQR. Thresholds: 68.52 (1e-2 FPR).

![](images/edb685767e8f86e3a7e88ab970be234f55f800f2e6145139bb01885853824c24.jpg)  
Figure 3: Experimental thresholds for different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ , each with different parameters on x-axis. The y-axis represents the potential threat that $\hat { \mathbf { w } } ^ { \prime }$ is getting far from w, which corresponds to Eq. (7). Watermarking method: Tree-Ring.

## 4.4 Effects of Lipschitz constant $L _ { T }$

As demonstrated in Eq. (5) and Eq. (7), the distance $\left\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \right\|$ is upper-bounded by a term proportional to $( 1 + L \tau ) r$ for a fixed F, provided that the condition $\| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| \le L _ { \mathbf { F } } r$ is satisfied. This leads us to investigate whether the size of $( 1 + L \tau ) \Vert \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \Vert$ is positively correlated with the separation of w from $\hat { \mathbf { w } } ^ { \prime }$ . The result for Tree-Ring is presented in Fig. 3 and those for other watermarking methods are shown in Figs. 22 and 23. Essentially, these patterns in these figures appear highly similar because the value $( 1 + \mathbf { \bar { \chi } } _ { L T } ) \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \|$ is independent of the watermark pattern. In comparison with Fig. 1, we observe that lower values on the y-axis in Fig. 3 correspond to smaller perturbation shifts, such as Blurring and high-quality factors of JPEG. Conversely, higher y-axis values typically indicate greater shifts, such as Cropping and certain Rotation cases. Interestingly, Jitter exhibits large standard deviations, suggesting that the shift varies significantly across different images. This effect corresponds to the dispersed distribution of the red bars in Jitter of Figs. 1 and 2. For thresholds computed with different watermarking schemes, please refer to Sec. J of Appendix.

## 5 Conclusion and Limitations

In this paper, we establish that latent-based watermarking methods lack invariance to perturbations. Through both theoretical analysis and empirical evaluation, we show that their performance is strongly influenced by the components used in the diffusion reverse and inverse processes. Furthermore, our theoretical and experimental findings demonstrate that existing latent-based watermarking schemes, utilizing DDIM or other expansive modules, are fundamentally limited in practical settings, particularly when operating under low FPR constraints. Taken together, these results highlight fundamental limitations of current latent-based watermarking approaches and suggest that the prevailing design paradigm may be insufficient to achieve robust watermarking in practice.

Our aim is not to hinder the development of latent-based watermarking methods, but to provide guidance for future research. We acknowledge that our analysis relies on assumptions about the MMSE denoiser and the invertibility of DDIM, which may not always hold in practice. Despite these limitations, the proposed framework offers a useful analytical perspective for identifying potential weaknesses in future watermarking designs.

## References

[1] Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, et al. Waves: Benchmarking the robustness of image watermarks. In International Conference on Machine Learning, pages 1456–1492. PMLR, 2024. 6, 7

[2] Kasra Arabi, Benjamin Feuer, R Teal Witter, Chinmay Hegde, and Niv Cohen. Hidden in the noise: Two-stage robust watermarking for images. arXiv preprint arXiv:2412.04653, 2024. 5

[3] Hai Ci, Yiren Song, Pei Yang, Jinheng Xie, and Mike Zheng Shou. Wmadapter: Adding watermark control to latent diffusion models. arXiv preprint arXiv:2406.08337, 2024. 5

[4] Hai Ci, Pei Yang, Yiren Song, and Mike Zheng Shou. Ringid: Rethinking tree-ring watermarking for enhanced multi-key identification. arXiv preprint arXiv:2404.14055, 2024. 1, 2, 3, 5, 7, 14, 22, 23, 26, 27, 36, 37

[5] Samantha Delouya. European union agrees to regulate potentially harmful effects of artificial intelligence, 2023. 1

[6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 1, 8, 14

[7] European Parliament. Generative ai and watermarking, 2023. 1

[8] Han Fang, Kejiang Chen, Zehua Ma, Jiajun Deng, Yicong Li, Weiming Zhang, and Ee-Chien Chang. Syntag: Enhancing the geometric robustness of inversion-based generative image watermarking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15416–15425, 2025. 5

[9] Han Fang, Kejiang Chen, Zijin Yang, Bosen Cui, Weiming Zhang, and Ee-Chien Chang. Cosda: Enhancing the robustness of inversion-based generative image watermarking framework. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 2888–2896, 2025. 5

[10] Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 22466–22477, 2023. 1, 5

[11] Zheng Gao, Yifan Yang, Xiaoyu Li, Xiaoyan Feng, Haoran Fan, Yang Song, and Jiaojiao Jiang. Slice: Semantic latent injection via compartmentalized embedding for image watermarking. arXiv preprint arXiv:2603.12749, 2026. 3, 14

[12] Sam Gunn, Xuandong Zhao, and Dawn Song. An undetectable watermark for generative image models. arXiv preprint arXiv:2410.07369, 2024. 5, 7, 8, 24, 25

[13] Mingze He, Hongxia Wang, Fei Zhang, and Yuyuan Xiang. Exploring accurate invariants on polar harmonic fourier moments in polar coordinates for robust image watermarking. IEEE Transactions on Multimedia, 26:5435–5449, 2024. 6

[14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6

[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1

[16] Huayang Huang, Yu Wu, and Qian Wang. Robin: Robust and invisible watermarks for diffusion models with adversarial optimization. Advances in Neural Information Processing Systems, 37:3937–3963, 2024. 2, 3, 8, 14

[17] Ying Huang, Hu Guan, Jie Liu, Shuwu Zhang, Baoning Niu, and Guixuan Zhang. Robust texture-aware local adaptive image watermarking with perceptual guarantee. IEEE Transactions on Circuits and Systems for Video Technology, 33(9):4660–4674, 2023. 6

[18] Zhaoyang Jia, Han Fang, and Weiming Zhang. Mbrs: Enhancing robustness of dnn-based watermarking by mini-batch of real and simulated jpeg compression. In Proceedings of the 29th ACM international conference on multimedia, pages 41–49, 2021. 2

[19] Zahra Kadkhodaie, Florentin Guth, Eero P Simoncelli, and Stéphane Mallat. Generalization in diffusion models arises from geometry-adaptive harmonic representation. arXiv preprint arXiv:2310.02557, 2023. 2, 3

[20] Sung Ju Lee and Nam Ik Cho. Semantic watermarking reinvented: Enhancing robustness and generation quality with fourier integrity. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18759–18769, 2025. 5, 7, 8, 26

[21] Chun-Shien Lu and Chao-Yong Hsu. Content-dependent anti-disclosure image watermark. In International Workshop on Digital Watermarking, pages 61–76. Springer, 2003. 2

[22] Chun-Shien Lu, Shih-Wei Sun, Chao-Yong Hsu, and Pao-Chi Chang. Media hash-dependent image watermarking resilient against both geometric attacks and estimation attacks based on false positiveoriented detection. IEEE Transactions on Multimedia, 8(4):668–685, 2006. 2, 14

[23] Xiyang Luo, Ruohan Zhan, Huiwen Chang, Feng Yang, and Peyman Milanfar. Distortion agnostic deep watermarking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13548–13557, 2020. 2

[24] TorchVision maintainers and contributors. Torchvision: Pytorch’s computer vision library. https: //github.com/pytorch/vision, 2016. 7

[25] Po-Yuan Mao, Cheng-Chang Tsai, and Chun-Shien Lu. Maxsive: High-capacity and robust training-free generative image watermarking in diffusion models. In Proceedings of the 33rd ACM International Conference on Multimedia, 2025. 2, 5

[26] Zheling Meng, Bo Peng, and Jing Dong. Latent watermark: Inject and detect watermarks in latent diffusion space. IEEE Transactions on Multimedia, 2025. 2

[27] Rachel Metz. The white house released an ‘ai bill of rights’, 2022. 1

[28] Christopher A Metzler, Arian Maleki, and Richard G Baraniuk. From denoising to compressed sensing. IEEE Transactions on Information Theory, 62(9):5117–5144, 2016. 3

[29] Sreyas Mohan, Zahra Kadkhodaie, Eero P Simoncelli, and Carlos Fernandez-Granda. Robust and interpretable blind image denoising via bias-free convolutional neural networks. arXiv preprint arXiv:1906.05478, 2019. 3

[30] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023. 14

[31] Fabien AP Petitcolas. Watermarking schemes evaluation. IEEE signal processing magazine, 17(5):58–64, 2000. 14, 27

[32] Fabien AP Petitcolas, Ross J Anderson, and Markus G Kuhn. Attacks on copyright marking systems. In International workshop on information hiding, pages 218–238. Springer, 1998. 7, 14, 26, 27

[33] Sundeep Rangan, Philip Schniter, and Alyson K Fletcher. Vector approximate message passing. IEEE Transactions on Information Theory, 65(10):6664–6684, 2019. 2, 3

[34] Sylvestre-Alvise Rebuffi, Tuan Tran, Valeriu Lacatusu, Pierre Fernandez, Tomáš Soucek, Nikola Jovanoviˇ c,´ Tom Sander, Hady Elsahar, and Alexandre Mourachko. Learning to watermark in the latent space of generative models. arXiv preprint arXiv:2601.16140, 2026. 3, 14

[35] Ahmad Rezaei, Mohammad Akbari, Saeed Ranjbar Alvar, Arezou Fatemi, and Yong Zhang. Lawa: Using latent space for in-generation image watermarking. In European Conference on Computer Vision, pages 118–136. Springer, 2024. 5

[36] Yaniv Romano, Michael Elad, and Peyman Milanfar. The little engine that could: Regularization by denoising (red). SIAM Journal on Imaging Sciences, 10(4):1804–1844, 2017. 3

[37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021, 2021. 1, 2, 3, 7

[38] Joseph JK Ò Ruanaidh and Thierry Pun. Rotation, scale and translation invariant spread spectrum digital image watermarking. Signal processing, 66(3):303–317, 1998. 2, 14, 19

[39] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 14

[40] Vassilios Solachidis and Loannis Pitas. Circularly symmetric watermark embedding in 2-d dft domain. IEEE transactions on image processing, 10(11):1741–1753, 2001. 2

[41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 1, 2, 3, 8, 14

[42] Tomáš Soucek, Pierre Fernandez, Hady Elsahar, Sylvestre-Alvise Rebuffi, Valeriu Lacatusu, Tuan Tran,ˇ Tom Sander, and Alexandre Mourachko. Pixel seal: Adversarial-only training for invisible image and video watermarking. arXiv preprint arXiv:2512.16874, 2025. 6

[43] The White House. Fact sheet: President biden issues executive order on safe, secure, and trustworthy artificial intelligence, 2023. 1

[44] Sviatoslav Voloshynovskiy, Shelby Pereira, Victor Iquise, and Thierry Pun. Attack modelling: towards a second generation watermarking benchmark. Signal processing, 81(6):1177–1214, 2001. 2, 14

[45] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers, 2022. 7

[46] Jiayan Wang, Jing Zhao, Li Li, Zichi Wang, Hanzhou Wu, and Deyang Wu. Robust blind video watermark ing based on ring tensor and bch coding. IEEE Internet of Things Journal, 11(24):40743–40756, 2024. 6

[47] Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Hezhen Hu, Hong Chen, and Houqiang Li. Dire for diffusion-generated image detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22445–22455, 2023. 14

[48] Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. Tree-rings watermarks: Invisible fingerprints for diffusion images. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 1, 2, 3, 5, 7, 8, 14, 22, 23, 27, 34, 35

[49] Deyang Wu, Xinpeng Zhang, Jiayan Wang, Li Li, and Guorui Feng. Novel robust video watermarking scheme based on concentric ring subband and visual cryptography with piecewise linear chaotic mapping. IEEE Transactions on Circuits and Systemsfor Video Technology, 34(10):10281–10298, 2024. 6

[50] JinFeng Xie, Peipeng Yu, Jianwei Fei, Xiaoyu Zhou, and Zhihua Xia. Safetr: Verifiable semantic treering watermark for diffusion model against forgery attacks. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 9505–9511, 2026. 3, 14

[51] Pei Yang, Hai Ci, Yiren Song, and Mike Zheng Shou. Can simple averaging defeat modern watermarks? In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2, 3, 23, 24

[52] Zijin Yang, Kai Zeng, Kejiang Chen, Han Fang, Weiming Zhang, and Nenghai Yu. Gaussian shading: Provable performance-lossless image watermarking for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12162–12171, 2024. 2, 3, 5, 14

[53] Fei Zhang, Hongxia Wang, Mingze He, and Jinghong Xia. Robust blind symmetry-based watermarking in the frequency domain against social network processing and desynchronization attacks. IEEE Transactions on Circuits and Systems for Video Technology, pages 1–1, 2024. 6

[54] Lijun Zhang, Xiao Liu, Antoni V Martin, Cindy X Bearfield, Yuriy Brun, and Hui Guan. Attack-resilient image watermarking using stable diffusion. Advances in Neural Information Processing Systems, 37: 38480–38507, 2024. 2, 3, 14

[55] Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks. In Proceedings ofthe European conference on computer vision (ECCV), pages 657–672, 2018. 2, 5

[56] Yifan Zhu, Yihan Wang, and Xiao-Shan Gao. Towards robust content watermarking against removal and forgery attacks. arXiv preprint arXiv:2604.06662, 2026. 3, 14

## Appendix

## A Background

We present the background knowledge relevant to our research.

## A.1 Rotation, Scale, and Translation Invariance

Geometric attacks (GAs), as one of the primary kinds of watermarking attacks [44], are greatly ignored nowadays in the evaluation of the robustness of generative watermarking. GAs aims to cause synchronization errors to disable watermark detection without removing embedded watermarks or degrading the quality of watermarked images. Rotation, scale, and translation are common global geometric distortions, which are included in Stirmark [31, 32]. In earlier studies, [38] presented a Fourier—Mellin-based approach to embed watermarks in resisting a combination of rotation and scaling transformations, whereas [22] proposed robust feature-based image watermarking with resistance to Stirmark benchmark.

## A.2 DDIM Inversion

Diffusion models [39] generate synthetic images through the reverse process, which progressively denoises the initial noise sampled from the standard Gaussian distribution. Since this reverse process is inherently not deterministic, it is virtually impossible to retrieve the initial noise that generates a given synthetic image. However, the introduction of DDIM [41] renders this possible. Song et al. [41] suggested that we can “reverse” the reverse process to obtain the initial noise for downstream applications that require it [4, 30, 47, 48]. Note that in [41], this “inverse process” was only mentioned, without any explicit formulation. The first explicit formulation of the inverse process appeared in [6], although it was not yet called the inverse process. Recently, in the literature on watermarking on AIGCs, the term “inverse process” was first introduced in Tree-Ring [48]. Since then, numerous methods exploiting the inverse process have been proposed [4, 11, 16, 34, 48, 50, 52, 54, 56].

## B The Lipschitz bounds of DDIM Reverse and Inverse Process (Prerequisite of Sec. C)

In this section, we provide a derivation that connects VAMP and the MMSE denoiser to the DDIM reverse process, and prove that such a process is invertible, a.k.a the inverse process.

## B.1 Theoretical Analysis: Spectral Stability of Diffusion via MMSE-AMP

In this section, we derive the spectral properties of the diffusion process by establishing an equivalence between the Denoising Diffusion Implicit Model (DDIM) and Vector Approximate Message Passing (VAMP). Our key insight is that DDIM acts as a normalized iterative denoiser, where the normalization constant induces an expansive regime essential for generation. By treating the diffusion step as a regularized MMSE estimation update, we derive rigorous spectral bounds for both the generative (reverse) and encoding (inverse) processes.

The DDIM sampling process generates a sequence $\mathbf x _ { T } ( i . e . , \mathbf y ) , \mathbf x _ { T - 1 } , . . . , \mathbf x _ { 0 }$ via the update rule:

$$
\mathbf { x } _ { t - 1 } = \sqrt { \bar { \alpha } _ { t - 1 } } \underbrace { \left( \frac { \mathbf { x } _ { t } - \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { \theta } ( \mathbf { x } _ { t } ) } { \sqrt { \bar { \alpha } _ { t } } } \right) } _ { \hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } ) } + \sqrt { 1 - \bar { \alpha } _ { t - 1 } } \epsilon _ { \theta } ( \mathbf { x } _ { t } ) ,\tag{8}
$$

where $\begin{array} { r } { \bar { \alpha } _ { t } : = \prod _ { s = 0 } ^ { t } \alpha _ { s } , \{ \alpha _ { t } \} _ { t = 0 } ^ { T } \subset ( 0 , 1 ] } \end{array}$ is the noise schedule, and $\epsilon _ { \theta } ( \cdot )$ is the learned denoiser parameterized by θ, and $\hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } )$ is the DDIM prediction at t.

## B.1.1 Connection: DDIM as Normalized AMP

To apply the VAMP theory, we must align the signal models. Standard VAMP assumes a signal-plusnoise model $\mathbf { y } = \mathbf { x } _ { 0 } + \xi , \xi \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \mathrm { I d } )$ . In contrast, the diffusion state $\mathbf { x } _ { t }$ scales the signal by

$\sqrt { \alpha _ { t } }$ . We define the Normalized AMP State $\mathbf { u } _ { t }$ as:

$$
\mathbf { u } _ { t } = \frac { \mathbf { x } _ { t } } { \sqrt { \bar { \alpha } _ { t } } } = \mathbf { x } _ { 0 } + \sqrt { \frac { 1 - \bar { \alpha } _ { t } } { \bar { \alpha } _ { t } } } \epsilon .
$$

This transforms $\mathbf { x } _ { t }$ into the standard form expected by an MMSE estimator. The DDIM prediction $\hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } )$ is equivalent to applying a standard MMSE denoiser $D _ { \theta } ( \cdot )$ to this normalized state:

$$
\hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t } ) = D _ { \theta } ( \mathbf { u } _ { t } ) = D _ { \theta } \left( \frac { \mathbf { x } _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \right) .\tag{9}
$$

To connect $\epsilon _ { \theta }$ in DDIM and $D _ { \theta }$ in VAMP, we have:

$$
\epsilon _ { \theta } ( \mathbf { z } ) = \frac { \mathbf { z } - \sqrt { \bar { \alpha } _ { t } } D _ { \theta } \left( \frac { \mathbf { z } } { \sqrt { \bar { \alpha } _ { t } } } \right) } { \sqrt { 1 - \bar { \alpha } _ { t } } } ,\tag{10}
$$

where z is the noisy representation of each time step t in DDIM.

## B.1.2 The Jacobian Analysis

We first reformulate the DDIM update rule in Eq. (8) to isolate the denoising operator by employing Eq. (10) to yield the MMSE-AMP Affine Form with $\mathbf { z } = \mathbf { x } _ { t }$ . The DDIM update, the reverse step, from $t  t - 1$ is given by the affine form:

$$
{ \bf x } _ { t - 1 } = \mu _ { t } { \bf x } _ { t } + \gamma _ { t } \hat { \bf x } _ { 0 } ( { \bf x } _ { t } ) = \mu _ { t } { \bf x } _ { t } + \gamma _ { t } D _ { \theta } \left( \frac { { \bf x } _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \right) ,\tag{11}
$$

where the coefficients are determined by the noise schedule:

$$
\begin{array} { r l } & { { \mu _ { t } } = \frac { { \sqrt { 1 - { { \bar { \alpha } } _ { t - 1 } } } } } { { \sqrt { 1 - { { \bar { \alpha } } _ { t } } } } } , \quad \mathrm { ( N o i s e } \mathrm { C a r r y o v e r ) } } \\ & { { \gamma _ { t } } = \sqrt { { { \bar { \alpha } } _ { t - 1 } } } - { \mu _ { t } } \sqrt { { { \bar { \alpha } } _ { t } } } . \quad \mathrm { ( S i g n a l I n j e c t i o n ) } } \end{array}
$$

To determine the stability of this update, we compute the Jacobian matrix $\begin{array} { r } { J _ { t } = \frac { \partial \mathbf { x } _ { t - 1 } } { \partial \mathbf { x } _ { t } } } \end{array}$ . Applying the chain rule to Eq. (11) introduces the critical scaling factor $1 / \sqrt { \bar { \alpha } _ { t } } \mathrm { : }$

$$
J _ { t } = \mu _ { t } \mathrm { I d } + \left. \frac { \gamma _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \nabla _ { \mathbf { u } } D _ { \theta } ( \mathbf { u } ) \right| _ { \mathbf { u } = \frac { \mathbf { x } _ { t } } { \sqrt { \bar { \alpha } _ { t } } } } .\tag{12}
$$

## B.2 Derivation of Spectral Bounds

We apply the Fundamental Property of MMSE Estimators used in VAMP theory: the MMSE denoiser is contractive, i.e., its eigenvalues are within [0, 1]. Thus, we can have the following proposition.

Proposition B.1 (Spectral Bounds of MMSE Denoiser). Assuming the denoiser $D _ { \theta } ( \cdot )$ approximates the optimal MMSE estimator under Gaussian noise, the eigenvalues of its Jacobian matrix satisfy $\lambda ( \nabla \bar { D } _ { \theta } ) \subset [ 0 , 1 ]$

Proof. Our goal is to show that $\nabla D _ { \theta }$ is positive semi-definite, and that $0 \preceq \nabla D _ { \theta } \preceq { \mathrm { I d } }$ . Given $\mathbf { u } \sim \mathcal { N } ( \mathbf { x } , \sigma ^ { 2 } \mathrm { I d } )$ , the denoiser $D = D _ { \theta }$ is defined as posterior expectation conditioned on u:

$$
D ( \mathbf { u } ) = \mathbb { E } [ \mathbf { x } | \mathbf { u } ] = \int \mathbf { x } p ( \mathbf { x } | \mathbf { u } ) d \mathbf { x } .
$$

Applying Tweedie’s formula,

$$
D ( \mathbf { u } ) = \mathbf { u } + \sigma ^ { 2 } \nabla \log p ( \mathbf { u } ) ,
$$

where $\begin{array} { r } { p ( \mathbf { u } ) : = \int p ( \mathbf { u } | \mathbf { x } ) p ( \mathbf { x } ) } \end{array}$ dx. So the Jacobian matrix is

$$
\nabla D ( \mathbf { u } ) = \operatorname { I d } + \sigma ^ { 2 } \nabla ^ { 2 } \log p ( \mathbf { u } ) .\tag{13}
$$

Using the formula (will be proved in Lemma B.2):

$$
\nabla ^ { 2 } \log { p ( \mathbf { u } ) } = \frac { 1 } { \sigma ^ { 4 } } \mathrm { C o v } ( \mathbf { x } | \mathbf { u } ) - \frac { 1 } { \sigma ^ { 2 } } \mathrm { I d } ,
$$

where $\operatorname { C o v } ( \mathbf { x } | \mathbf { u } ) : = \mathbb { E } [ \mathbf { x } \mathbf { x } ^ { \top } | \mathbf { u } ] - \mathbb { E } [ \mathbf { x } | \mathbf { u } ] \mathbb { E } [ \mathbf { x } | \mathbf { u } ] ^ { \top }$ , and taking it into Eq. (13) gives

$$
\nabla D ( \mathbf { u } ) = \operatorname { I d } + \sigma ^ { 2 } \left( { \frac { 1 } { \sigma ^ { 4 } } } \mathrm { { C o v } } ( \mathbf { x } | \mathbf { u } ) - { \frac { 1 } { \sigma ^ { 2 } } } \mathrm { { I d } } \right) = { \frac { 1 } { \sigma ^ { 2 } } } \mathrm { { C o v } } ( \mathbf { x } | \mathbf { u } ) .
$$

This immediately shows that $\nabla D \succeq 0$ since covariance matrices are positive semi-definite. To show that $\nabla D \preceq \mathrm { I d }$ , we observe that, $p ( \mathbf { u } )$ is log-concave, which means $\nabla ^ { 2 } \log p ( \mathbf { u } ) \preceq 0$ . Thus using Eq. (13) again, we obtain $\nabla D \preceq { \bar { \mathrm { I d } } }$ □

Lemma B.2. $\begin{array} { r } { \nabla ^ { 2 } \log p ( \mathbf { u } ) = \frac { 1 } { \sigma ^ { 4 } } \mathrm { C o v } ( \mathbf { x } | \mathbf { u } ) - \frac { 1 } { \sigma ^ { 2 } } \mathrm { I d } . } \end{array}$

Proof. Note that $p ( \mathbf { u } | \mathbf { x } ) = \mathcal { N } ( \mathbf { x } , \sigma ^ { 2 } \mathrm { I d } )$ , so we have the useful formula, the derivative of p.d.f. of Gaussian distribution:

$$
\nabla _ { \mathbf { u } } p ( \mathbf { u } | \mathbf { x } ) = - \frac { 1 } { \sigma ^ { 2 } } ( \mathbf { u } - \mathbf { x } ) p ( \mathbf { u } | \mathbf { x } ) .\tag{14}
$$

We begin with computing the Jacobian. From the chain rule, $\begin{array} { r } { \nabla \log p ( \mathbf { u } ) = \frac { \nabla p ( \mathbf { u } ) } { p ( \mathbf { u } ) } } \end{array}$ , and calculate the numerator:

$$
\begin{array} { r l } & { \nabla p ( \mathbf { u } ) = \nabla _ { \mathbf { u } } \int p ( \mathbf { u } | \mathbf { x } ) p ( \mathbf { x } ) d \mathbf { x } } \\ & { \qquad = \displaystyle \int \nabla _ { \mathbf { u } } p ( \mathbf { u } | \mathbf { x } ) p ( \mathbf { x } ) d \mathbf { x } } \\ & { \qquad = \displaystyle - \frac { 1 } { \sigma ^ { 2 } } \int ( \mathbf { u } - \mathbf { x } ) p ( \mathbf { u } | \mathbf { x } ) p ( \mathbf { x } ) d \mathbf { x } , } \end{array}\tag{15}
$$

where the last equality follows by Eq. (14). Thus, applying Bayes’ formula,

$$
\begin{array} { l } { { \nabla \log p ( { \mathbf u } ) = - { \displaystyle \frac { 1 } { p ( { \mathbf u } ) \sigma ^ { 2 } } } \int ( { \mathbf u } - { \mathbf x } ) p ( { \mathbf u } | { \mathbf x } ) p ( { \mathbf x } ) d { \mathbf x } } \ ~ } \\ { { \displaystyle ~ = - { \frac { 1 } { \sigma ^ { 2 } } } \int ( { \mathbf u } - { \mathbf x } ) { \frac { p ( { \mathbf u } | { \mathbf x } ) p ( { \mathbf x } ) } { p ( { \mathbf u } ) } } d { \mathbf x } } \ ~ } \\ { { \displaystyle ~ = - { \frac { 1 } { \sigma ^ { 2 } } } \int ( { \mathbf u } - { \mathbf x } ) p ( { \mathbf x } | { \mathbf u } ) d { \mathbf x } } \ ~ } \\ { { \displaystyle ~ = - { \frac { 1 } { \sigma ^ { 2 } } } \mathbb E [ { \mathbf u } - { \mathbf x } | { \mathbf u } ] } . } \end{array}\tag{16}
$$

Next, we compute the Hessian:

$$
\begin{array} { r l } & { \nabla ^ { 2 } \log p ( \mathbf { u } ) = \nabla \left( \frac { \nabla p ( \mathbf { u } ) } { p ( \mathbf { u } ) } \right) = \frac { p ( \mathbf { u } ) \nabla ^ { 2 } p ( \mathbf { u } ) - \nabla p ( \mathbf { u } ) ( \nabla p ( \mathbf { u } ) ) ^ { \top } } { p ( \mathbf { u } ) ^ { 2 } } } \\ & { \qquad = \frac { \nabla ^ { 2 } p ( \mathbf { u } ) } { p ( \mathbf { u } ) } - \left( \frac { \nabla p ( \mathbf { u } ) } { p ( \mathbf { u } ) } \right) \left( \frac { \nabla p ( \mathbf { u } ) } { p ( \mathbf { u } ) } \right) ^ { \top } } \\ & { \qquad = \frac { \nabla ^ { 2 } p ( \mathbf { u } ) } { p ( \mathbf { u } ) } - \frac { 1 } { \sigma ^ { 4 } } \mathbb { E } [ \mathbf { u } - \mathbf { x } | \mathbf { u } ] \mathbb { E } [ \mathbf { u } - \mathbf { x } | \mathbf { u } ] ^ { \top } , } \end{array}\tag{17}
$$

where the second term of the last equality follows by Eq. (16). To deal with the first term of Eq. (17), we need to calculate $\nabla ^ { 2 } p ( \mathbf { u } )$ . From Eqs. (14) and (15),

$$
\begin{array} { l } { { \nabla ^ { 2 } p ( { \bf u } ) = - { \frac { 1 } { \sigma ^ { 2 } } } \int \nabla _ { { \bf u } } [ ( { \bf u } - { \bf x } ) p ( { \bf u } | { \bf x } ) ] p ( { \bf x } ) d { \bf x } } \ ~ } \\ { { \displaystyle ~ = - { \frac { 1 } { \sigma ^ { 2 } } } \int \left[ { \bf L } \cdot p ( { \bf u } | { \bf x } ) + ( { \bf u } - { \bf x } ) ( \nabla _ { { \bf u } } p ( { \bf u } | { \bf x } ) ) ^ { \top } \right] p ( { \bf x } ) d { \bf x } } \ ~ } \\ { { \displaystyle ~ = - { \frac { 1 } { \sigma ^ { 2 } } } \int \left[ { \bf H } \cdot p ( { \bf u } | { \bf x } ) - { \frac { 1 } { \sigma ^ { 2 } } } ( { \bf u } - { \bf x } ) ( { \bf u } - { \bf x } ) ^ { \top } p ( { \bf u } | { \bf x } ) \right] p ( { \bf x } ) d { \bf x } } \ ~ } \\ { { \displaystyle ~ = - { \frac { 1 } { \sigma ^ { 2 } } } \mathrm { H d } \int p ( { \bf u } | { \bf x } ) p ( { \bf x } ) d { \bf x } + { \frac { 1 } { \sigma ^ { 4 } } } \int ( { \bf u } - { \bf x } ) ( { \bf u } - { \bf x } ) ^ { \top } p ( { \bf u } | { \bf x } ) p ( { \bf x } ) d { \bf x } } \ ~ } \\ { { \displaystyle ~ = - { \frac { 1 } { \sigma ^ { 2 } } } \mathrm { I d } \cdot p ( { \bf u } ) + { \frac { 1 } { \sigma ^ { 4 } } } \int ( { \bf u } - { \bf x } ) ( { \bf u } - { \bf x } ) ^ { \top } p ( { \bf u } | { \bf x } ) p ( { \bf x } ) d { \bf x } } . } \end{array}
$$

Therefore, using Bayes’ formula,

$$
\begin{array} { l } { \displaystyle \frac { \nabla ^ { 2 } p ( \mathbf { u } ) } { p ( \mathbf { u } ) } = - \frac { 1 } { \sigma ^ { 2 } } \mathrm { I d } + \frac { 1 } { \sigma ^ { 4 } } \int ( \mathbf { u } - \mathbf { x } ) ( \mathbf { u } - \mathbf { x } ) ^ { \top } p ( \mathbf { x } | \mathbf { u } ) d \mathbf { x } } \\ { \displaystyle \qquad = - \frac { 1 } { \sigma ^ { 2 } } \mathrm { I d } + \frac { 1 } { \sigma ^ { 4 } } \mathbb { E } [ ( \mathbf { u } - \mathbf { x } ) ( \mathbf { u } - \mathbf { x } ) ^ { \top } | \mathbf { u } ] . } \end{array}\tag{18}
$$

Finally, taking Eq. (18) into Eq. (17) gives

$$
\begin{array} { l } { { \nabla ^ { 2 } \log p ( { \bf u } ) = \displaystyle \frac { 1 } { \sigma ^ { 4 } } \left( { \mathbb { E } [ ( { \bf u } - { \bf x } ) ( { \bf u } - { \bf x } ) ^ { \top } | { \bf u } ] - \mathbb { E } [ { \bf u } - { \bf x } | { \bf u } ] \mathbb { E } [ { \bf u } - { \bf x } | { \bf u } ] ^ { \top } } \right) - \frac { 1 } { \sigma ^ { 2 } } \mathrm { I d } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { \sigma ^ { 4 } } \mathrm { C o v } ( { \bf u } - { \bf x } | { \bf u } ) - \frac { 1 } { \sigma ^ { 2 } } \mathrm { I d } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { \sigma ^ { 4 } } \mathrm { C o v } ( { \bf x } | { \bf u } ) - \frac { 1 } { \sigma ^ { 2 } } \mathrm { I d } } , } \end{array}
$$

so the Hessian is obtained. This completes the proof.

Using this property, the spectrum of the DDIM Jacobian matrix $J _ { t }$ in Eq. (12) is bounded by the linear combination of the identity and the denoiser spectrum. The spectrum of the reverse step is thus:

$$
\lambda ( J _ { t } ) = \left\{ \mu _ { t } + \frac { \gamma _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \lambda : \lambda \in \lambda ( \nabla D _ { \theta } ) \right\} .\tag{19}
$$

Also, we can verify whether our derived sampling step is invertible. First, since noise variance never hits zero during intermediate steps, $\mu _ { t } > 0$ . Also, for time step t $\mathbf { \sigma } _ { 0 } t - 1 , \bar { \alpha } _ { t - 1 } > \bar { \alpha } _ { t }$ , which implies $\gamma _ { t } > 0$ . For an MMSE denoiser $D _ { \theta } ( \cdot )$ , its Jacobian matrix is positive semi-definite since its eigenvalues are within [0, 1]. Therefore, the eigenvalues of Eq. (12), a.k.a Eq. (19), are strictly positive. As a result, Eq. (11) is non-singular.

## B.2.1 Generative Expansion (Reverse Process)

The maximum expansion is the supremum over the spectrum $\begin{array} { r } { L _ { \mathrm { g e n } } ^ { ( t ) } = \operatorname* { s u p } _ { \lambda _ { D } \in [ 0 , 1 ] } \left( \mu _ { t } + \frac { \gamma _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \lambda _ { D } \right) } \end{array}$ The Lipschitz constant of the generation step is the maximum singular value of $J _ { t }$ . This occurs along the signal direction, where the denoiser preserves structure $( \lambda _ { \operatorname* { m a x } } ( \nabla D _ { \theta } ) = 1 )$ . From Eq. (19), we can obtain:

$$
\begin{array} { r l r } {  { L _ { \mathrm { g e n } } ^ { ( t ) } \le \mu _ { t } + \frac { \gamma _ { t } } { \sqrt { \bar { \alpha } _ { t } } } = \mu _ { t } + \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } ( \sqrt { \bar { \alpha } _ { t - 1 } } - \mu _ { t } \sqrt { \bar { \alpha } _ { t } } ) } } \\ & { } & { ~ = \mu _ { t } + \frac { \sqrt { \bar { \alpha } _ { t - 1 } } } { \sqrt { \bar { \alpha } _ { t } } } - \mu _ { t } = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } } { \sqrt { \bar { \alpha } _ { t } } } . } \end{array}\tag{20}
$$

## B.2.2 Inversion Singularity

For the inverse process $x _ { t - 1 } \to x _ { t } .$ , we ask “where does the reverse process contract the input the most?” The maximum expansion of the inverse process is dictated by the infimum of the forward step $\begin{array} { r } { L _ { \mathrm { i n v } } ^ { ( t ) } = \left( \operatorname* { i n f } _ { \lambda _ { D } \in [ 0 , 1 ] } \left( \mu _ { t } + \frac { \gamma _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \lambda _ { D } \right) \right) ^ { - 1 } } \end{array}$ . During the reverse process, the denoiser suppresses the noisy signal the most $( \dot { \lambda } _ { \mathrm { m i n } } ( \nabla D _ { \theta } ) \dot { = } \mathrm { 0 ) }$ . Hence, the stability is governed by the noise subspace, where in this regime, the Jacobian matrix in Eq. (12) simplifies to $\breve { J _ { t } } = \mu _ { t } \mathrm { I d }$ . The inverse Lipschitz constant is therefore $1 / \mu _ { t } \colon$

$$
L _ { \mathrm { i n v } } ^ { ( t ) } \leq \frac { 1 } { \mu _ { t } } = \frac { \sqrt { 1 - \bar { \alpha } _ { t } } } { \sqrt { 1 - \bar { \alpha } _ { t - 1 } } } .\tag{21}
$$

## C Proofs of Theorems and Lemmas

In the following, we provide detailed proofs of Theorem 3.1, Theorem 3.2, Theorem 3.3, Theorem $3 . 4 ,$ and Theorem 3.5.

Proof of Theorem 3.1. In our specific MMSE-VAMP framework, the mapping naturally satisfies the "stronger conditions" required for global bijectivity. Specifically, the Jacobian of our update step is not merely non-singular; it is strictly positive definite globally. We prove with the following reasoning:

First, by Tweedie’s formula, the optimal MMSE denoiser under Gaussian noise is $D _ { \theta } ( \mathbf { x } ) = \mathbf { x } +$ $\sigma ^ { 2 } \nabla _ { \mathbf { x } } \log p ( \mathbf { x } )$ . Because it is the gradient of a scalar potential function, its Jacobian $\dot { \nabla } D _ { \boldsymbol { \theta } } ( \mathbf { x } )$ is symmetric. Furthermore, standard MMSE properties dictate that the eigenvalues of $\nabla D _ { \boldsymbol { \theta } } ( \mathbf { \dot { x } } )$ are bounded in [0, 1] everywhere in the domain. Therefore, $\nabla D _ { \boldsymbol { \theta } } ( \mathbf { x } )$ is a positive semi-definite (PSD) matrix globally.

Recall our derived Jacobian for a single DDIM step $( { \bf x } _ { t }  { \bf x } _ { t - 1 } )$

$$
J _ { t } = \mu _ { t } \mathbf { I } + \frac { \gamma _ { t } } { \sqrt { \bar { \alpha } _ { t } } } \nabla D _ { \theta } ( \mathbf { u } _ { t } )
$$

Because $\mu _ { t } > 0$ strictly (based on the diffusion noise schedule), $\gamma _ { t } > 0 .$ , and $\nabla D _ { \theta } \succeq 0$ globally, the entire Jacobian matrix $J _ { t }$ is strictly positive definite everywhere in $\mathbb { R } ^ { d }$ . Specifically, its minimum eigenvalue is bounded globally by $\mu _ { t } > 0$

Finally, according to the global inverse function theorem (and specifically the Gale-Nikaido theorem for mappings with positive principal minors, or standard results for strongly monotone operators), a continuously differentiable mapping $F : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ whose Jacobian is everywhere strictly positive definite is a global diffeomorphism (a smooth bijection). Because every individual step $t  t - 1$ is a global bijection, the full T-step composition is also globally invertible. □

Proof of Theorem 3.2. We use proof by contradiction. Suppose both $\hat { \mathbf { x } } _ { 0 } ^ { R S T }$ and $\hat { \mathbf { x } } _ { 0 }$ have the same latent representation, which means $\mathbf { y } ^ { R \pmb { \xi } _ { T } } = \mathbf { y }$ , given the invertible $f _ { \theta }$ and $\mathbf { I } ^ { R S T } = \mathcal { R } \mathcal { S } \mathcal { T } ( \mathbf { I } )$ , we can derive:

$$
\hat { \mathbf { x } } _ { 0 } ^ { R S T } = \mathcal { E } _ { \phi _ { e } } ( \mathbf { I } ^ { R S T } ) = f _ { \theta } ( \mathbf { y } ^ { R S T } ) ,
$$

which contradicts $\hat { \mathbf { x } } _ { 0 } ^ { R S T } \neq \hat { \mathbf { x } } _ { 0 }$ . Actually, the above statement is true only when $\mathcal { R S T } = \mathrm { I d }$ , that is to say, there are no $\mathcal { R } \mathcal { S } \dot { \mathcal { T } }$ perturbations on I.

Moreover, with the invertible $f _ { \theta } ,$ , we have:

$$
\begin{array} { r } { \mathbf { y } = f _ { \theta } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } ) , \quad \quad } \\ { \mathbf { y } ^ { R S T } = f _ { \theta } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } ^ { R S T } ) . } \end{array}
$$

If the assumption $\mathbf { y } ^ { R S T } = \mathbf { y }$ is imposed, we have:

$$
f _ { \theta } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } ^ { R S T } ) = f _ { \theta } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } ) ,\tag{22}
$$

which contradicts the definition of inverse function and $\mathcal { R } \mathcal { S } \mathcal { T } \neq \mathrm { I d }$ since $\hat { \mathbf { x } } _ { 0 } ^ { R S T } = \hat { \mathbf { x } } _ { 0 }$ and is true only when $\mathcal { R S T } = \mathrm { I d }$ , which means no perturbation is imposed on I. As a result, we can conclude that if $\hat { \mathbf { x } } _ { 0 } ^ { R S T } \neq \hat { \mathbf { x } } _ { 0 } \left( i . e . , \mathcal { R S T } \neq \mathrm { I d } \right)$ , then $\mathbf { y } ^ { R S T } \neq \mathbf { y }$

Now, given $\mathbf { I } ^ { R S T } = \mathcal { R } \mathcal { S } \mathcal { T } ( \mathbf { I } )$ and $\hat { \mathbf { x } } _ { 0 } ^ { R S T } \neq \hat { \mathbf { x } } _ { 0 }$ , we can derive by combining the DDIM inversion $f _ { \theta } ^ { - 1 }$ with $\mathcal { F }$ and $\mathcal { F } _ { \mathcal { M } }$ in the RST-invariant watermarking framework [38] as:

$$
\mathcal { I } _ { 1 } = [ \mathcal { F _ { M } } \circ \mathcal { F } ] f _ { \theta } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } ^ { R S T } ) = [ \mathcal { F _ { M } } \circ \mathcal { F } ] \mathbf { y } ^ { R S T } ,\tag{23}
$$

$$
\mathcal { T } _ { 2 } = [ \mathcal { F } _ { \mathcal { M } } \circ \mathcal { F } ] f _ { \theta } ^ { - 1 } ( \hat { \mathbf { x } } _ { 0 } ) = [ \mathcal { F } _ { \mathcal { M } } \circ \mathcal { F } ] \mathbf { y } .\tag{24}
$$

where $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ are outputs of ${ \mathcal { F } } _ { { \mathcal { M } } } \circ { \mathcal { F } }$ operations, respectively. To demonstrate that $\mathcal { T } _ { 1 } \neq \mathcal { T } _ { 2 }$ we leverage the inherent properties of DDIM inversion—and generative models more broadly. Specifically, even within the framework of a Probability Flow ODE, due to the expansive property of $f _ { \theta } ^ { - 1 } \left( i . e . \right.$ , large $L _ { f _ { \theta } ^ { - 1 } } )$ , the resulting latent representations y and $\mathbf { y } ^ { R S T }$ exhibit amplitude inconsistency. By treating the inconsistency as a random variable (see proof of Theorem 3.3), it follows that $\mathcal { T } _ { 1 } \neq \mathcal { T } _ { 2 }$ holds almost surely. In conclusion, when the DDIM inversion process operates with an RST-invariant framework, the combined process, $a . k . a$ DDIM inversion and RST-invariant framework, loses RST invariance. □

Proof of Theorem 3.3. In this proof, we will continue the discussion of the proof of Theorem 3.2, focusing on proving the expansion property of DDIM inverse process. Let $\mathbf x _ { t } , \mathbf y _ { t }$ be the trajectories of two distinct data points under the deterministic DDIM process, and let $\Delta _ { t } : = { \bf x } _ { t } - { \bf y } _ { t }$ denote the difference vector between the two particles at time t. Using the linearized approximation of the DDIM trajectory:

$$
\mathbf { x } _ { t } \approx \sqrt { \bar { \alpha } _ { t } } \mathbf { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ,
$$

For two points starting at $\mathbf { x } _ { \mathrm { 0 } }$ and $\mathbf { y } _ { 0 }$ with random initial difference $\begin{array} { r } { \Delta _ { 0 } = \mathbf { x } _ { 0 } - \mathbf { y } _ { 0 } ( i . e . } \end{array}$ , any perturbation), their inverted states at time t relate to the initial states as (with high probability):

$$
\| \Delta _ { t } \| \geq \frac { \sqrt { 1 - \bar { \alpha } _ { t } } } { \sqrt { 1 - \alpha _ { 0 } } } \| \Delta _ { 0 } \| .\tag{25}
$$

To show Eq. (25), we follow the rule of DDIM inversion, for each step,

$$
\begin{array} { r l } & { \mathbf { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \hat { \mathbf { x } } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon } \\ & { \quad = \sqrt { \bar { \alpha } _ { t } } \hat { \mathbf { x } } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \frac { \mathbf { x } _ { t - 1 } - \sqrt { \bar { \alpha } _ { t - 1 } } \hat { \mathbf { x } } _ { 0 } } { \sqrt { 1 - \bar { \alpha } _ { t - 1 } } } } \\ & { \quad = \sqrt { \displaystyle \frac { 1 - \bar { \alpha } _ { t } } { 1 - \bar { \alpha } _ { t - 1 } } } \mathbf { x } _ { t - 1 } + \left( \sqrt { \bar { \alpha } _ { t } } - \frac { \sqrt { ( 1 - \bar { \alpha } _ { t } ) \bar { \alpha } _ { t - 1 } } } { \sqrt { 1 - \bar { \alpha } _ { t - 1 } } } \right) \hat { \mathbf { x } } _ { 0 } , } \end{array}
$$

where $\hat { \mathbf { x } } _ { 0 } = \hat { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { t - 1 } )$ is the predicted $\mathbf { x } _ { \mathrm { 0 } }$ given $\mathbf { x } _ { t - 1 }$ . It follows that

$$
\begin{array} { r l } & { \frac { \partial \mathbf { x } _ { t } } { \partial \mathbf { x } _ { t - 1 } } = \sqrt { \frac { 1 - \bar { \alpha } _ { t } } { 1 - \bar { \alpha } _ { t - 1 } } } \mathrm { I d } + \left( \sqrt { \bar { \alpha } _ { t } } - \frac { \sqrt { ( 1 - \bar { \alpha } _ { t } ) } \bar { \alpha } _ { t - 1 } } { \sqrt { 1 - \bar { \alpha } _ { t - 1 } } } \right) \frac { \partial \hat { \mathbf { x } } _ { 0 } } { \partial \mathbf { x } _ { t - 1 } } } \\ & { \qquad = \underbrace { \sqrt { \frac { 1 - \bar { \alpha } _ { t } } { 1 - \bar { \alpha } _ { t - 1 } } } \mathrm { I d } } _ { = : \eta _ { t } } + \underbrace { \left( \sqrt { \frac { \bar { \alpha } _ { t } } { \bar { \alpha } _ { t - 1 } } } - \sqrt { \frac { 1 - \bar { \alpha } _ { t } } { 1 - \bar { \alpha } _ { t - 1 } } } \right) } _ { = : - \kappa _ { t } } \nabla D _ { \theta } ( \mathbf { u } _ { t } ) | _ { \mathbf { u } _ { t } = \mathbf { x } _ { t - 1 } / \sqrt { \bar { \alpha } _ { t - 1 } } } , } \\ & { \qquad = \eta _ { t } \left( \mathrm { I d } - \frac { \kappa _ { t } } { \eta _ { t } } \nabla D _ { \theta } ( \mathbf { u } _ { t } ) \right) , } \end{array}\tag{26}
$$

where Eq. (26) follows from Eq. (9), and $\eta _ { t } , \kappa _ { t } > 0$ denote the corresponding coefficients (see Fig. 4 for numerically computing $\eta _ { t } , \kappa _ { t }$ , and $\kappa _ { t } / \eta _ { t } )$ . We may multiply all the terms together and gives the Jacobian:

$$
\begin{array} { r l r } & { } & { J _ { 0  t } : = \displaystyle \frac { \partial \mathbf { x } _ { t } } { \partial \mathbf { x } _ { 0 } } = \prod _ { s = 1 } ^ { t } \frac { \partial \mathbf { x } _ { s } } { \partial \mathbf { x } _ { s - 1 } } = ( \prod _ { s = 1 } ^ { t } \eta _ { s } ) \prod _ { s = 1 } ^ { t } ( \mathrm { I d } - \frac { \kappa _ { s } } { \eta _ { s } } \nabla D _ { \theta } ( \mathbf { u } _ { s } ) ) } \\ & { } & { \displaystyle = \frac { \sqrt { 1 - { \bar { \alpha } } _ { t } } } { \sqrt { 1 - \alpha _ { 0 } } } ( \mathrm { I d } + \sum _ { m = 1 } ^ { t } R _ { 0  t } ^ { ( m ) } ) , } \end{array}
$$

where

$$
R _ { 0  t } ^ { ( m ) } : = ( - 1 ) ^ { m } \sum _ { \substack { 1 \leq i _ { 1 } < i _ { 2 } < \cdots < i _ { m } \leq t j = 1 } } \prod _ { j = 1 } ^ { m } \frac { \kappa _ { i _ { j } } } { \eta _ { i _ { j } } } \nabla D _ { \theta } ( \mathbf { u } _ { i _ { j } } ) .
$$

Since $\Delta _ { 0 }$ is a high-dimensional vector, and note that $0 \preceq \nabla D _ { \theta } ( \mathbf { u } _ { i _ { i } } ) \preceq { \mathrm { I d } }$ (Proposition B.1), the vector $R _ { 0  t } ^ { ( m ) } \Delta _ { 0 }$ , for each $m ,$ is perpendicular to $\Delta _ { 0 }$ with high probability. Therefore,

$$
\| \Delta _ { t } \| = \| J _ { 0 \to t } \Delta _ { 0 } \| = \frac { \sqrt { 1 - { \bar { \alpha } } _ { t } } } { \sqrt { 1 - \alpha _ { 0 } } } \left\| \Delta _ { 0 } + \sum _ { m = 1 } ^ { t } R _ { 0 \to t } ^ { ( m ) } \Delta _ { 0 } \right\| \geq \frac { \sqrt { 1 - { \bar { \alpha } } _ { t } } } { \sqrt { 1 - \alpha _ { 0 } } } \| \Delta _ { 0 } \|
$$

with high probability.

The inverse process maps data from a low-variance manifold $( \alpha _ { 0 } \approx 1 )$ to a high-variance Gaussian sphere $( \alpha _ { t }  0 )$ . The scaling factor $\sqrt { 1 - \bar { \alpha } _ { t } } / \sqrt { 1 - \alpha _ { 0 } }$ acts as an amplifier for any $\Delta _ { 0 } .$ . Consequently, the distance between two distinct particles is significantly magnified as t increases. □

![](images/b79dd9cfcc11f8018d888998ae071550fd4696d838953a055efe8e6c05d5a1ba.jpg)  
Figure 4: Coefficients $\eta _ { t } , \kappa _ { t } , \kappa _ { t } / \eta _ { t }$ of all time steps, under the linear noise schedule.

Proof of Theorem 3.4. Since $\bar { \alpha } _ { t - 1 } > \bar { \alpha } _ { t } , L _ { \mathrm { g e n } } ^ { ( t ) } > 1$ . Eq. (20) proves that generation is locally expansive. The total Lipschitz constant for the full trajectory telescopes perfectly:

$$
L _ { f _ { \theta } } \leq \prod _ { t = 1 } ^ { T } \frac { \sqrt { \bar { \alpha } _ { t - 1 } } } { \sqrt { \bar { \alpha } _ { t } } } = \frac { \sqrt { \alpha _ { 0 } } } { \sqrt { \bar { \alpha } _ { T } } } .
$$

Similarly, the global bound for inversion telescopes can be derived using Eq. (21) as:

$$
L _ { f _ { \theta } ^ { - 1 } } \leq \prod _ { t = 1 } ^ { T } \frac { \sqrt { 1 - \bar { \alpha } _ { t } } } { \sqrt { 1 - \bar { \alpha } _ { t - 1 } } } = \frac { \sqrt { 1 - \bar { \alpha } _ { T } } } { \sqrt { 1 - \alpha _ { 0 } } } .
$$

Before proving Theorem 3.5, we need two lemmas.

Lemma C.1. Given $L _ { \mathcal { G } } , L _ { f _ { \theta } } , L _ { \mathcal { W } } , L _ { \mathcal { F } ^ { - 1 } } , \mathbf { y } \sim \mathcal { N } ( \mathbf { 0 } , \mathrm { I d } )$ , w, and $r > 0 ,$ we denote the Lipschitz constant of F in Eq. (3) as $L _ { \mathbf { F } } = L _ { \mathcal { G } _ { \phi _ { a } } } L _ { f _ { \theta } } L _ { \mathcal { F } ^ { - 1 } } L _ { \mathcal { W } }$ . Furthermore, given $\mathbf { w } ^ { \prime } \in \mathcal { B } _ { r } ( \mathbf { w } )$ , we can obtain the maximum perturbation bound in the pixel space with $\mathbf { I } = \mathbf { F } ( \mathbf { y } , \mathbf { w } )$ and $\mathbf { I } ^ { \prime } = \mathbf { F } ( \mathbf { y } , \mathbf { w } ^ { \prime } )$ as:

$$
\| \mathbf { I } - \mathbf { I ^ { \prime } } \| \leq L _ { \mathbf { F } } r .\tag{27}
$$

Specifically, the inequality in Eq. (27) can be interpreted as the maximum discrepancy in the pixel space, given that the perturbed watermark $\mathbf { w } ^ { \prime } \in \bar { B _ { r } ( \mathbf { w } ) }$ in the latent space.

Proof of Lemma C.1. According to the results in Theorem 3.4, given the Lipschitz constants of W, $\mathcal { M } _ { : }$ , and $\mathcal { G } _ { \phi _ { g } }$ , and Eq. (3) and $L _ { \mathcal { W } } = \sqrt { H W }$ , we can have the following property:

$$
\| \mathbf { I } - \mathbf { I ^ { \prime } } \| = \| \mathbf { F } ( \mathbf { y } , \mathbf { w } ) - \mathbf { F } ( \mathbf { y } , \mathbf { w } ^ { \prime } ) \| \leq L _ { \mathbf { F } } \| \mathbf { w } - \mathbf { w } ^ { \prime } \| ,
$$

$$
\begin{array} { r } { L _ { \mathbf { F } } = L _ { \mathcal { G } _ { \phi _ { g } } } L _ { f _ { \theta } } L _ { \mathcal { F } ^ { - 1 } } L _ { \mathcal { W } } , } \end{array}\tag{28}
$$

and, with a radius $r$ in the latent space at point w, we define a ball centered at w as $\boldsymbol { B } = \left\{ \mathbf { w } ^ { \prime } \right.$ $\| \mathbf { w } ^ { \prime } - \mathbf { w } \| \leq r \}$ . Hence, the Eq. (28) can be simplified into:

$$
\| \mathbf { I } - \mathbf { I ^ { \prime } } \| \leq L _ { \mathbf { F } } r ,\tag{29}
$$

Lemma C.2. Given the Lipschitz constants $L _ { { \mathcal { E } } _ { \phi _ { e } } } , L _ { f _ { o } ^ { - 1 } } , L _ { \mathcal { F } }$ , and $L _ { \mathcal { M } } ,$ , we define the Lipschitz constant ofthe watermark estimation ofan image as $L _ { \mathbf { F } ^ { - 1 } } ^ { ' } = L \varepsilon _ { \phi _ { e } } L _ { f _ { a } ^ { - 1 } } L _ { \mathcal { F } } L _ { \mathcal { M } }$ . With Eq. (4), the ground-truth watermark w, and a distorted image T(I),for all $\mathbf { I } = \mathbf { F } ( \mathbf { \bar { y } } , \mathbf { w } )$ and $\mathbf { y } \sim \mathcal { N } ( \mathbf { 0 } , \mathrm { I d } )$ , the maximum perturbation that affects the estimated watermark wˆ can be derived as:

$$
\begin{array} { r } { \| \mathbf { w } - \hat { \mathbf { w } } \| \leq \| \mathbf { w } - \bar { \mathbf { w } } \| + L _ { \mathbf { F } ^ { - 1 } } \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| , } \end{array}\tag{30}
$$

where $\bar { \mathbf { w } } = \mathbf { F } ^ { - 1 } ( \mathbf { I } ) = \mathcal { M } ( \mathcal { F } ( f _ { \theta } ^ { - 1 } ( \mathcal { E } _ { \phi _ { e } } ( \mathbf { I } ) ) )$ indicates the estimated watermark from image I. If $\mathbf { F } ^ { - 1 } \mathbf { F } = \mathrm { I d } , E q . \ ( 3 0 )$ can be reduced into

$$
\| \mathbf { w } - \hat { \mathbf { w } } \| \leq L _ { \mathbf { F } ^ { - 1 } } \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| .\tag{31}
$$

ProofofLemma C.2. According to the results in Theorem 3.4, given the Lipschitz constants $L _ { \mathcal { E } _ { \phi _ { e } } }$ $L _ { f _ { \theta } ^ { - 1 } } , L _ { \mathcal { F } }$ , and $L _ { \mathcal { M } }$ , and Eq. (4), we can have the following property:

$$
\| \bar { \mathbf { w } } - \hat { \mathbf { w } } \| = \| \mathbf { F } ^ { - 1 } ( \mathbf { I } ) - \mathbf { F } ^ { - 1 } ( \mathcal { T } ( \mathbf { I } ) ) \| \leq L _ { \mathbf { F } ^ { - 1 } } \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| ,
$$

$$
L _ { \mathbf { F } ^ { - 1 } } = L \varepsilon _ { \phi _ { e } } L _ { f _ { \theta } ^ { - 1 } } L _ { \mathcal { F } } L _ { \mathcal { M } } .
$$

Using the triangular inequality, we have the result

$$
\left\| \mathbf { w } - { \hat { \mathbf { w } } } \right\| \leq \left\| \mathbf { w } - { \bar { \mathbf { w } } } \right\| + \left\| { \bar { \mathbf { w } } } - { \hat { \mathbf { w } } } \right\| \leq \left\| \mathbf { w } - { \bar { \mathbf { w } } } \right\| + L _ { \mathbf { F } ^ { - 1 } } \left\| \mathbf { I } - { \mathcal { T } } ( \mathbf { I } ) \right\| .\tag{32}
$$

Given the inverse process $f _ { \theta } ^ { - 1 }$ and the encoder $\mathcal { E } _ { \phi _ { e } } , \| \mathbf { w } - \bar { \mathbf { w } } \| = 0$ since $\mathbf { F } ^ { - 1 } \mathbf { F } = \mathrm { I d }$ , hence, Eq. (32) can be reduced into Eq. (31). □

Proof of Theorem 3.5. With the above lemmas, given the Lipschitz constant $L _ { T }$ and any two watermarked images I and I<sup>′</sup>, we have

$$
\Vert T ( \mathbf { I } ) - T ( \mathbf { I } ^ { \prime } ) \Vert \leq L \tau \Vert \mathbf { I } - \mathbf { I } ^ { \prime } \Vert .\tag{33}
$$

Using the triangle inequality and Eq. (33), we have

$$
\| \mathbf { I } - T ( \mathbf { I } ^ { \prime } ) \| \leq \| \mathbf { I } - T ( \mathbf { I } ) \| + \| T ( \mathbf { I } ) - T ( \mathbf { I } ^ { \prime } ) \| \leq \| \mathbf { I } - T ( \mathbf { I } ) \| + L _ { T } \| \mathbf { I } - \mathbf { I } ^ { \prime } \| .\tag{34}
$$

$$
\| \mathbf { I } - \mathcal { T } ( \mathbf { I } ^ { \prime } ) \| \leq \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| + L _ { \mathcal { T } } \| \mathbf { I } - \mathbf { I } ^ { \prime } \| \leq \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| + L _ { \mathcal { T } } L _ { \mathbf { F } } r .\tag{35}
$$

Next, given Lemma C.1, if all watermarked images are resistant to the perturbation $\tau$ on themselves, the following must be satisfied with Eq. (27):

$$
\forall \mathbf { I } , \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ) \| \leq L _ { \mathbf { F } } r .\tag{36}
$$

Hence, from Eqs. (35) and (36), we get

$$
\begin{array} { r } { \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ^ { \prime } ) \| \le L _ { \mathbf { F } } r + L _ { \mathcal { T } } L _ { \mathbf { F } } r = ( 1 + L _ { \mathcal { T } } ) L _ { \mathbf { F } } r . } \end{array}\tag{37}
$$

Define the following

$$
\bar { \mathbf { w } } = \mathbf { F } ^ { - 1 } ( \mathbf { I } ) , \quad \hat { \mathbf { w } } ^ { \prime } = \mathbf { F } ^ { - 1 } ( \mathcal { T } ( \mathbf { I } ^ { \prime } ) ) ,
$$

by the definition of Lipschitz constant $L _ { \mathbf { F } ^ { - 1 } }$ , we derive

$$
\| \bar { \mathbf { w } } - \hat { \mathbf { w } } ^ { \prime } \| = \| \mathbf { F } ^ { - 1 } ( \mathbf { I } ) - \mathbf { F } ^ { - 1 } ( \mathcal { T } ( \mathbf { I } ^ { \prime } ) ) \| \leq L _ { \mathbf { F } ^ { - 1 } } \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ^ { \prime } ) \| .\tag{38}
$$

Using the triangle inequality, we turn Eq. (38) into

$$
\begin{array} { r } { \| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| \leq \| \mathbf { w } - \bar { \mathbf { w } } \| + \| \bar { \mathbf { w } } - \hat { \mathbf { w } } ^ { \prime } \| \leq \| \mathbf { w } - \bar { \mathbf { w } } \| + L _ { \mathbf { F } ^ { - 1 } } \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ^ { \prime } ) \| . } \end{array}\tag{39}
$$

Given $f _ { \theta } ^ { - 1 }$ and $\mathcal { E } _ { \phi _ { e } } , \| \mathbf { w } - \bar { \mathbf { w } } \| = 0$ since F<sup>−1</sup>F = Id, hence, Eq. (39) can be reduced into

$$
\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| \leq L _ { \mathbf { F } ^ { - 1 } } \| \mathbf { I } - \mathcal { T } ( \mathbf { I } ^ { \prime } ) \| .\tag{40}
$$

Combining Eqs. (37) and (40), we finally have Eq. (5).

$$
\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| \leq L _ { \mathbf { F } ^ { - 1 } } ( 1 + L \tau ) L _ { \mathbf { F } } r
$$

## D B defined by Tolerance Constraints

![](images/88798d2eb7de1781d039481cdba051c8ffcb495cc6ebf509b42e52a3adce6fe3.jpg)  
Figure 5: Visual explanation of B. (a) The tolerance constraint is described in Sec. 3.4. (b) The constraint only considers the y<sub>1</sub>-axis (corresponding to watermark and detection on the Channel 3 in Tree-Ring [48]), leaving other dimensions unbounded. As long as the $\ell _ { 1 }$ distance between any points and y (centre) on the y<sub>1</sub>-axis is below a certain threshold $\tau$ (corresponding to falling within the two planes in the figure), it is considered to be in B. (c) This constraint considers the $y _ { 1 ^ { - } }$ and y<sub>2</sub>-axes, respectively (corresponding to the Channels 0 and 3 in RingID [4]), leaving the y<sub>3</sub>-axes unbounded. The constraint requires calculating the $\ell _ { 1 }$ distance between any point and $\mathbf { y }$ on $y _ { 1 ^ { - } }$ and y -axes, respectively, in a dimension-wise manner, and then takes the minimum of these two values. $\mathbf { A }$ point is considered to fall in B if this minimum value is below a certain threshold τ (corresponding to falling within the cross-shaped region in the figure).

In this section, we give an intuitive explanation of the high FPR under different tolerance constraints as described and illustrated in Fig. 5. Given a tolerance τ and the point $\mathbf { y } = ( y _ { 1 } , y _ { 2 } , y _ { 3 } ) \in \mathbb { R } ^ { 3 }$ , we define the following subspaces

$$
\begin{array} { r l } & { \mathcal { B } _ { a } = \{ \mathbf { y } ^ { \prime } | \| \mathbf { y } ^ { \prime } - \mathbf { y } \| _ { 2 } \leq \tau , \mathbf { y } ^ { \prime } \in \mathbb { R } ^ { 3 } \} , } \\ & { \mathcal { B } _ { b } = \{ \mathbf { y } ^ { \prime } | \| y _ { 1 } ^ { \prime } - y _ { 1 } \| _ { 1 } \leq \tau , \mathbf { y } ^ { \prime } \in \mathbb { R } ^ { 3 } \} , } \\ & { \mathcal { B } _ { b ^ { \prime } } = \{ \mathbf { y } ^ { \prime } | \| y _ { 2 } ^ { \prime } - y _ { 2 } \| _ { 1 } \leq \tau , \mathbf { y } ^ { \prime } \in \mathbb { R } ^ { 3 } \} , } \\ & { \mathcal { B } _ { c } = \{ \mathbf { y } ^ { \prime } | \operatorname* { m i n } ( \| y _ { 1 } ^ { \prime } - y _ { 1 } \| _ { 1 } , \| y _ { 2 } ^ { \prime } - y _ { 2 } \| _ { 1 } ) \leq \tau , \mathbf { y } ^ { \prime } \in \mathbb { R } ^ { 3 } \} . } \end{array}
$$

where $\begin{array} { r } { B _ { a } , B _ { b } . } \end{array}$ , and $B _ { c }$ correspond to Fig. 5(a), (b), and (c), respectively. We can easily observe that three different constraints can cover different volumes vol(·) in a three-dimensional space, denoted as $v o l ( B _ { a } )$ , vol $( B _ { b } )$ , and $v o l ( B _ { c } )$ . First, $B _ { a } \subset B _ { b } , B _ { b ^ { \prime } } , B _ { c }$ . Second, $v o l ( B _ { c } )$ can be seen as the union of two subspaces, $B _ { b }$ and $B _ { b ^ { \prime } }$ ′ . We could conclude that

$$
B _ { a } \subset B _ { b } \subset B _ { c } .\tag{41}
$$

Accordingly, it is easier for a data point $\mathbf { y } ^ { \prime }$ to be included in $\textstyle B _ { c } ,$ while it is harder to be included in $\scriptstyle { B _ { a } }$ since $B _ { c }$ covers the most volume. More importantly, to perturb/attack a data point in/out of subspaces $B _ { b }$ and $\textstyle B _ { c } .$ , the effort can be made by moving the point along the bounded axes, respectively. This results in a harder attack success rate for $\bar { B _ { a } } ^ { - }$ since all axes should be considered. Hence, we can conclude that, under the same tolerance $\tau ,$ , the constraint leads to different structures of $B ,$ , resulting in looser or tighter criteria to determine whether a point $\mathbf { y } ^ { \prime }$ lies within B.

## E Simulations of Distribution Shift

To demonstrate that relocation of $\mathbf { y } _ { \mathbf { w } }$ caused by watermarks moves the latent distributions away from those of normally generated images with the corresponding latent representations y close to 0, we examine this effect in this section for the simulation of Gaussian distribution in Fig. 6(a) and the corresponding $\chi$ distribution in Fig. 6(b) under different shifts in mean $\mu .$ . This gives a larger tolerance for perturbing the images and leads to better robustness of watermarking in the generative process, despite the possible high FPRs.

![](images/1d24893cb47bbdb29dfcb9c49bb9a88a5a03d46c86e4abf3b970fca88646f7da.jpg)  
(a)

![](images/d1fa0d619ed2cc05cd19f6845bff519122ca8b7f985466b001442960bd4cc4e9.jpg)  
(b)

Figure 6: (a) Distribution Shift in latent space. The norm of watermarked images will shift away from 0. By injecting a larger-norm watermark in latent representation, it is easier to distinguish whether an image is from a normal distribution. Also, the allowed perturbation will be larger since the latent representations of perturbed images will fall within the gap between the unwatermarked and watermarked latents. However, they are still away from the unwatermarked latent. (b) The distribution of the $\ell _ { 2 } { \mathrm { - n o r m } }$ of samples from each normal distribution. These distributions follow the $\chi$ distribution. The shaded areas represent the 0.3 and 99.7 percentiles for each distribution.  
![](images/ca989d16891664e1a51e11e44807e9f2d67b6aee052aeebe310b1e71d5c80edd.jpg)  
(a) Watermark: Tree-Ring

![](images/0608768a118587b5443ae1a3d27fa6588121c67a6ddd52ae15c1ad7493a7e397.jpg)

![](images/fae649f125d9904cdf6c8ddcdc5380da1f58dd0d07fa48bead22e0ccbcc5fb3e.jpg)

(b) Watermark: Tree-Ring  
![](images/7159750830c7630487a25c192c9ed24d7d8f2c7c79651ff07d4ef86d185e26d9.jpg)  
(c) Watermark: RingID  
(d) Watermark: RingID  
Figure 7: Histograms of distance. When comparing the latent of the attacked images to samples from the normal distribution $( e . g .$ , unwatermarked latent), we find that the attacked latent still maintains a certain distance from the unwatermarked latent, especially for the removal attack case. Here, “gtwm” means “ground-truth watermark”; “wm” means “watermarked images”; and “unwm” means “unwatermarked images”. (Best viewed in a color display)

## F Analysis of the Detection Distance through the Lens of Forgery and Removal Attacks

To provide another perspective, we have looked closely at the forgery and removal attacks on images to examine the effect of distribution shifts in latent representations in Fig. 7. The interpretation of the x-axis of Fig. 7 should not be confused with that of Fig. 6 in that the former describes the distance between the estimated watermark and the ground-truth watermark while the latter describes the mean value $\mu _ { \mathbf { w } }$ of watermarked distribution. Please also note that the interpretation of Fig. 7 should consider each histogram’s reference point and distance together, and that, in standard practice, we compute the distance between the inversed test latent and the ground truth watermark to decide if it is watermarked.

Specifically, in Fig. 7, we present the histograms of distance between two watermarks from two latent-based watermarking methods, Tree-Ring [48] and RingID [4], under the forgery and removal attacks [51]. We use the blue, orange, and green histograms to represent the distances between the latent representation of ground truth watermark and those of (1) unwatermarked images, (2) watermarked images, and (3) forgery/removal attacked images, respectively. As for the red histogram, it represents the distance between the latent representations of unwatermarked images and those of forgery/removal attacked images.

We can observe from Fig. 7(a) and Fig. 7(c) that to remove a watermark successfully, the ideal way is to push the watermarked images away from the ground truth watermark distribution, i.e., away from data that contain watermarks, which corresponds to the “green” histogram so that data being removed no longer stay in the ball. For forgery attack in Fig. 7(b) and Fig. 7(d), the attack adds an estimated watermark on other possibly unwatermarked images, which means that the attacker pushes unwatermarked image toward the ground truth watermark distribution that corresponds to the “green” histogram, indicating those data belong to the ball and hence are close to the watermark distribution.

Interestingly, the “red” histogram establishes that the attacked images (via forgery or removal) do not remain closer to the desired distribution (watermarked or unwatermarked images). This suggests key weaknesses in such methods: they allow excessive tolerance, exhibit small distribution shifts, and leak significant information (as shown in [51]), enabling attackers to estimate the watermark. Consequently, these methods are vulnerable to forgery and removal attacks. As long as the perturbations move data into or out of the target region, the attack needs not to be precisely aligned with—or directly oppose—the watermark direction. Unlike traditional non-learning-based watermarking techniques, which resist such attacks while preserving image quality, these generative watermarking methods trade robustness for flexibility, making the watermarks easier to be compromised.

## G Experimental Settings

In this section, we give a detailed description of the settings of perturbations. The perturbations and the settings of each perturbation is as follows:

– Blurring: blur with the kernel size set to 1, 2, 4, 6, 8.

– Color Jitter: increase the brightness of the image by setting the factor to 1, 2, 4, 6, 8.

– JPEG: compress the image with JPEG quality score set to 1, 5, 25, 40, 80.

– Rotation: rotate the image 5<sup>◦</sup>, 15<sup>◦</sup>, 30<sup>◦</sup>, 45<sup>◦</sup>, 75<sup>◦</sup> clock-wise.

– Cropping: crop 10%, 25%, 40%, 60%, 80% of the image (i.e., remaining factor 0.9, 0.75, 0.6, 0.4, 0.2, respectively).

– Elastic deformation: produces a see-through-water-like effect on the image, with (α, σ) = (500, 50), (600, 60), (750, 75), (850, 85), (1000, 100).

– Random perspective: perform random perspective on the image, with distortion scale set to 0.5, 0.6, 0.75, 0.85, 1.0.

– Shear: shear the image with the parameter set to 5, 7, 10, 15, 20.

– Regen-VAE: regenerate the image with VAE with the quality level set to 1, 2, 3, 4, 5.

– Regen-DiffPure: regenerate the image with Stable Diffusion v1-4 with the number of noise and denoise step set to 10, 20, 30, 40, 50.

For all cases of Tree-Ring, RingID, HSQR, HSTR, and PRCW, 100 watermarked images were processed through ten types of distortions and subsequently fed into the model for detection.

## H Complete Experiments of Tree-Ring, HSQR, and PRCW

We present the experiments related to the effects of perturbations on Tree-Ring, HSQR, and PRCW. Each watermarking scheme is attacked by seven perturbations with five different parameters to control the strength of the perturbation. Please refer to Figs. 8, 9, 10, 11, 12, and 13.

Following the official implementation of the PRCW method [12], first we initialize a key pair (encode\_key, decode\_key), and subsequently generate 100 watermarked images using encode\_key. We then apply seven types of image distortions listed in Sec. G of Appendix, each evaluated across five distinct intensity levels. Subsequently, decoding and detection are performed on the distorted images and their corresponding unattacked counterparts, using the decode\_key. The detection is determined by two binary criteria: (i) Decoding component: a bit-wise comparison between the recovered\_bits (from the input image) and the test\_bit (extracted by decode\_key), which requires an exact match to be considered valid, and (ii) Detection component: a threshold-based detection result. A final detection is deemed successful if either of these two criteria is satisfied.

![](images/a49a1aa9c40e35cbfea8a68294549d66553a247f5bca54dd3e7bf1083b556555.jpg)  
Figure 8: The effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . From left to right, the perturbations are set with different parameters (Param). Watermark: Tree-Ring. Thresholds: 77.12 (1e-2 FPR).

Table 2: Effects of different number of inference time steps on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . The number of inference time steps were set to 10, 30, 50 (default), 70, and 90. We present the mean and the maximum values of a set of $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$
<table><tr><td rowspan=1 colspan=1>WM Method</td><td rowspan=1 colspan=1>Tree-Ring</td><td rowspan=1 colspan=1>RingID</td><td rowspan=1 colspan=1>HSQR</td><td rowspan=1 colspan=1>HSTR</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>mean   max</td><td rowspan=1 colspan=1>mean   max</td><td rowspan=1 colspan=1>mean   max</td><td rowspan=1 colspan=1>mean   max</td></tr><tr><td rowspan=1 colspan=1>t = 10</td><td rowspan=1 colspan=1>55.36  62.38</td><td rowspan=1 colspan=1>29.34  45.37</td><td rowspan=1 colspan=1>43.17  48.91</td><td rowspan=1 colspan=1>19.94  25.93</td></tr><tr><td rowspan=1 colspan=1>t = 30</td><td rowspan=1 colspan=1>53.96  60.62</td><td rowspan=1 colspan=1>26.31  45.52</td><td rowspan=1 colspan=1>41.69  47.76</td><td rowspan=1 colspan=1>17.48  24.95</td></tr><tr><td rowspan=1 colspan=1>t = 50</td><td rowspan=1 colspan=1>53.72  60.53</td><td rowspan=1 colspan=1>25.63  45.91</td><td rowspan=1 colspan=1>41.33  47.53</td><td rowspan=1 colspan=1>16.94  24.86</td></tr><tr><td rowspan=1 colspan=1>t = 70</td><td rowspan=1 colspan=1>53.58  60.28</td><td rowspan=1 colspan=1>25.36  44.42</td><td rowspan=1 colspan=1>41.30  47.75</td><td rowspan=1 colspan=1>16.68  23.88</td></tr><tr><td rowspan=1 colspan=1>t = 90</td><td rowspan=1 colspan=1>53.52  60.16</td><td rowspan=1 colspan=1>25.03  43.66</td><td rowspan=1 colspan=1>41.19  47.75</td><td rowspan=1 colspan=1>16.55  23.54</td></tr></table>

To facilitate the visualization of the distance between perturbed watermarked images and their groundtruth watermarks, we employ the Hamming distance $\| \cdot \| _ { H }$ for the decoding component. For the detection component, we utilize the following discriminant from Algorithm 3 of PRCW [12]:

$$
\sum _ { \mathbf { w } \in \mathbf { P } } \log \left( \frac { 1 + \hat { s } _ { \mathbf { w } } } { 2 } \right) - \sqrt { C \log ( 1 / F ) } + \frac { 1 } { 2 } \sum _ { \mathbf { w } \in \mathbf { P } } \log \left( \frac { 1 - \hat { s } _ { \mathbf { w } } ^ { 2 } } { 4 } \right) \left\{ \begin{array} { l l } { \geq 0 } & { \Rightarrow \quad \mathrm { T r u e } , } \\ { < 0 } & { \Rightarrow \quad \mathrm { F a l s e } . } \end{array} \right.\tag{42}
$$

The corresponding results are illustrated in Figs. 10 and 11, respectively, under 1e-2 FPR. For 1e-5 FPR, please refer to Figs. 12 and 13, where blue dotted lines indicate the detection threshold.

We also present the numerical errors in Table 2. For visualizing the effects of numerical errors in DDIM inversion on $\left\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \right\|$ , we use histograms to demonstrate the shifting of $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ in Fig. 14.

![](images/5ca536feebd7be918c90c5c3a497389ddb50945c0af14191d8e34e0dffff5bb6.jpg)  
Figure 9: The effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . From left to right, the perturbations are set with different parameters (Param). Watermark: HSQR. Thresholds: 68.52 (1e-2 FPR).

## I Effects of $\left\| \mathbf { w } \right\|$ on Detection Distance

We examined the effect of ∥w∥ on watermark detection, as discussed in Lemma C.1 and Sec. 3.5. We used RingID [4] and HSTR [20] as representative examples, with the corresponding results presented in Figs. 15 and 16, and Figs. 20 and 21 in Sec. I.1 of Appendix. According to Eq. (27), we can expand the upper bound by increasing ∥w∥. The increase in ∥w∥ can be done by setting α in RingID [4] from 64 to 128, and multiplying the pattern injected in channel 0 by 2. Comparing Fig. 15 and Fig. 16, we see that the distance between the orange and blue distributions increases in Fig. 16. Such a gap can accommodate estimated watermarks that corresponding images are perturbed in the pixel space, and thereby create a buffer to withstand perturbations, especially when we use the threshold (red) of 1e-2 FPR. This modification can be done on HSTR. We also provide the experiments that we only set α to 128 without multiplying the pattern injected in channel 0 by 2. See Fig. 19 in Sec. I.1 of Appendix. For more thorough attacks, we used Stirmark [32] to examine the robustness of Tree-Ring and RingID in Sec. K of Appendix.

## I.1 Complete experiments of the effect of $\| \mathbf { w } \|$ on Detection Distance

In this section, we provide experiments on the effects of increasing ∥w∥ on robustness of watermark, which we consider RingID and HSTR as representative watermarking schemes. For notational simplicity, RingID-m-α denotes a configuration where the pattern on channel 0 is scaled by a factor of m (default: m = 1) and the pattern on channel 3 is governed by the parameter α (default: $\alpha = 6 4 )$ . Similarly, HSTR-m-m indicates that the patterns on both channels 0 and 3 are multiplied by m (default: m = 1). Please refer to Figs. 17, 18, 19, 20, and 21. For FID score results, please refer to Table 3.

## J Experiments of the Lipschitz Constant on thresholds

We present the estimated thresholds with different watermarking schemes. Please refer to Figs. 22 and 23.

![](images/c9475ab02f516a4652c4dbb094e7bb31a3a5bb04b2c0dfad6d453cd2a794b005.jpg)  
Figure 10: The effects of different perturbations on $\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| _ { H }$ . From left to right, the perturbations are set with different parameters (Param). Watermark: PRCW decoding component (1e-2 FPR). In this case, the number of bits is $\left\lceil \log _ { 2 } ( 1 / \mathrm { 1 0 ^ { - 2 } } ) \right\rceil = 7 .$

<table><tr><td>WM Methods</td><td>FID (↓)</td></tr><tr><td>RingID</td><td>45.31</td></tr><tr><td>RingID-2-128</td><td>428.68</td></tr><tr><td>HSTR</td><td>45.32</td></tr><tr><td>HSTR-2-2</td><td>187.77</td></tr></table>

Table 3: The FID score of watermark methods with different magnitude ∥w∥.

## K Performance of Tree-Ring & RingID on Stirmark

To check the robustness claimed by Tree-Ring [48] and RingID [4], we verified these methods via a well-known benchmark, called Stirmark [31, 32], which extensively includes both non-geometric image manipulations and geometric distortions, as shown in Tables 4, 5, 6, and 7. More information can be found in Stirmark [32].

In this experiment, we further examine the performance in terms of detection accuracy and TPR under more extensive perturbations with FPR set to 1e-2. Here, apart from the watermark detection accuracy in main text, the ACC represented in these tables is defined as

$$
\mathrm { A C C } = \operatorname* { m a x } \left\{ 1 - \frac { \mathrm { F P R } + \left( 1 - \mathrm { T P R } \right) } { 2 } \right\} ,
$$

which coincides with the usual definition of ACC since the numbers of watermarked and unwatermarked samples are equal.

![](images/900caed99a1bd06f673a91b6e43397b3334c5fa3c57cc142f665f6be22ec89f3.jpg)  
Figure 11: The effects of different perturbations by Eq. (42). From left to right, the perturbations are set with different parameters (Param). Watermark: PRCW detection component (1e-2 FPR).

![](images/b0dc4f6af77878c5383d4676f9012408124a01b58e0a77f994e1e9dc6c65d310.jpg)  
Figure 12: The effects of different perturbations on $\| \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \| _ { H }$ . From left to right, the perturbations are set with different parameters (Param). Watermark: PRCW decoding component (1e-5 FPR). In this case, the number of bits is $\left\lceil \log _ { 2 } ( 1 / 1 0 ^ { - 5 } ) \right\rceil = 1 7$

![](images/8a3e036fa016ce239ed757ac7017b54924ee63bbdd10356e2c53b9d3727386d0.jpg)

![](images/9f8aabeac19a69e97fd167aac2266364037627834d370f49052c8baf351d07ae.jpg)

![](images/30a920a6e07d36b8821807f6c725da16b4f7547e9e1d509f130e157fe431f960.jpg)

![](images/289bfb47d0f197e9755d34ae4471eaea2fb64b3278270517b72f797c31dda997.jpg)

![](images/e7d64ad61fb91a5f2e64a3e3d5e837d910d55a2574b745b656558927b778a370.jpg)

![](images/c4b84f6aca61098068306c89a3fa7468c50795bff8d28ca41a9cad996b957257.jpg)

![](images/66d39dc7f76bc7ee7bd1ea462d48065a7e35be94d5b65e8d2a11261c4fe0c03b.jpg)

![](images/a13b334c3b397079f3da204255f4da2827e525a49298bb713483fd64b3d847d3.jpg)

![](images/cb90913024a8f4a8c6141e06f4e6530979faefb17ad00bd384c98da209353e2f.jpg)

![](images/2dc3b5ed8ef0d4e3bd60d76312ceb83ebe9f910b0d2dbcff375820fca0a4c89b.jpg)

![](images/d675f1855c19dab68c3dd12d5ff4778ccfd103042377188d79f6389a14719162.jpg)

![](images/336fc99201e5371289a15ed04c74021bb171d334cb8e9a2c686ef62fd84aa31e.jpg)

![](images/6911d92938d054d21f2a19938dd069b69a8952050a660d074bdc39734be149ac.jpg)

![](images/377a975805c8489635d42e8a945db9dc40d81fcb63b837f6c0c5e18f19b49d50.jpg)

![](images/06c0d064b509a74fb0122a7b14c585acfa9b4321a6bddbb8f44118b4cf3ed29e.jpg)

![](images/8f149c72a3e0feae9c50f8898cc2fb099707c09bf1f535008bc633c8c43935be.jpg)

![](images/85a41a949f209a20ed8fa69c1ec0c2409d8af2c49f8653bbbf76952d7404fc93.jpg)

![](images/e2835ad6b56bf749db34708e505d46eb4c6fcddf573c1fe1e3992ecc04e30005.jpg)

![](images/412bd03e8c3f3d30b982f8c01de1ea70661470d4b880ab63aff50b832439311c.jpg)

<table><tr><td>50 25 0 -50</td><td>Param: 75 0 150 300</td></tr></table>

![](images/69e7c8ecaecff48b103e33f57a65d4059a034b7775e7db5e36c5152ac85b55f1.jpg)

![](images/0e1bd9767446e2c59749e4c41b4c2cd161485ca9fca08672e4c2e6d259689077.jpg)

![](images/ab5184a0eed74d96f04c018c8070f9f6b393eca3a2cca2a7381deddd408bca2f.jpg)

![](images/9d6377967f7cb12b127c5d4491999b066242efaf6551013d4fe36fd7f1f473bb.jpg)

![](images/38ffd4dcec2b05b1f35260ebf688067ddc1727234394bca17369141bd427fc29.jpg)

![](images/832b2a1d175a817016da12267609a917d0118320dc71100d44328faa5ca2b35e.jpg)

![](images/10617dba4d09cf6a3a76452da08f9330a089633ecfbf06285fda1e0e20dd18bd.jpg)

![](images/ab9f65c5391db3b7b41c59316b3364df3b0ef65d884179161fb95431038bd6c0.jpg)

<table><tr><td>50 25</td><td rowspan="16">0</td><td>Param: 1000.0</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>NL</td></tr><tr><td></td></tr><tr><td>-50 o</td><td>150 300</td></tr></table>

![](images/ca4d3d29b137b87de8ddea776abba4784ece83feaf1e0c50364b412f467e26c6.jpg)

![](images/15188a096a6b352cd1ccd408a6c7a78791d2f05c916f3fd232f6485582403d95.jpg)

![](images/a3ffe6d36b92b883f067d99680309923cabddac3a7ad0c62d60ec481436d4153.jpg)

![](images/c7eee9c0aad3da8923d3e7dcd3869028e5a121e07b20307ab516866de5ff406b.jpg)

![](images/d775060301eb1d322c0c094b286c36db88a248d78e519a924133a040f58bcd03.jpg)

![](images/d0d5ed69752b67c64d7f766e69e20689205f704d56957186fe5ad27c4675b957.jpg)

![](images/da2375f48a79268d19cd56a1d7e45de3eccfbae1afeebb9762af3b7fb14f9357.jpg)

![](images/5945eb92f0d7a21bb428d7a3300a5aa86e948481ea3147b201a0f3e5cc609a33.jpg)

![](images/ae498094a47279e7799c81ed9b05bb388a570b533a90c40704433e14d77e4b69.jpg)

![](images/606aa76c55a9513800e13df925b081e3b8e9f8acc22f8e2046156dbc6e87b292.jpg)

![](images/b754c0e9fd11e18de9ed9d86b3d9e0a385c518ebfd3801706244d2e9cc03665a.jpg)

![](images/906c01beee0594f81d73ae10c23d2e2fa2461211d55dd09a59858cbd161780da.jpg)

![](images/81c36ce0d1b5926dc1b3aeb3cb8c3c5e5ffdaa75601ae62646e40aad9433e53c.jpg)

![](images/a9360c6650f56e0a605b43b6bc04bacb89d3e87a0c05ff102e4b9ca77cb460a8.jpg)

![](images/be5b691edadbf5aeb80560cdd1170d70bb6b7aa500fdc9a3aa213711eb242923.jpg)

![](images/a280765d078a3d79af75afa9a97d7fd970a214fa1e109f5d6ae1fa9bc3d64479.jpg)

![](images/11e89e158f11e6106b9a10d2a8763ba2806c7e2682f88c808ef1bc8c63d9b254.jpg)

![](images/58444ee284a5c09b846cd8362242c068b58e0a9bbc1a6f293f163359d87e23c7.jpg)  
Figure 13: The effects of different perturbations by Eq. (42). From left to right, the perturbations are set with different parameters (Param). Watermark: PRCW detection component (1e-5 FPR).

![](images/2d024d4d98f561266e15eaa76c92d21aa41f1bf36dfecf8ffacbe2f834ed6819.jpg)

![](images/52b2743b4ddb2bd51890a47e177cb0323ff3e51094f50f0e1cfcbdc4c49b60a1.jpg)

![](images/77684c4e556432421433ef1106471a777734cc3ca96e0815854e2e6d7d684c7b.jpg)

![](images/e7b595f31da4bc68dec12ee9e16ca1c41cc50eabf010de9783d0162dfb535ad9.jpg)

![](images/c9abf7a813c141c5bb38a24d23e5f10ed3c6fdeaef5437e136e5bb5a4d470a67.jpg)

![](images/107d105f4c230b7f10793c61f9e36f27b47bf1014bb135987ecc9e9da7e68e86.jpg)

![](images/2c004d9f5eee367860a3a29cfed1bb2990b7b78bb601a3006d0e3a4b8c1cda0c.jpg)

![](images/a352bb36c2c3cf1e48d9ebfe2206ada18223b3f6b17b0fabec0d241ba92f4c2b.jpg)

![](images/37987b16e1ef4486b7ce80e235df02bbc360f0ed15ac5b7562873729465ad646.jpg)

![](images/1276a8c17f0ee8ce29854c09ae94500b42697b8f9e3e0c4fa46a24ca83cb5eca.jpg)

![](images/72e731fd04fff2ad21f34600915a9274975c54570b45233ef352ce9a674a184f.jpg)

![](images/4ecbe99f08e1feb468c5986fb39d25ad477ce55c00dbd6119805e676d4b401a5.jpg)

![](images/0eefd6056ec3ddeabd1b7dc65a6c2e1faa7cfc8e39b848a895c758b9500dcc61.jpg)

![](images/ff21f741c0bbd08d1022b049deebaf403812ef7bf07fcde0d215a4456303c445.jpg)

![](images/c8692def44f3d0a9aef27dea7aedc44b5de915082316f218285deeb49836746a.jpg)

![](images/35a71199b217dd4bd275f0e09ab86d3ee0e01586aaf2df2e7398881932bbd62f.jpg)

![](images/059cf4e1325cb1907eb5e6f0354844fb369bc294a6b14003dd8d8b1fe4e219ef.jpg)

![](images/e1fea9be9bba0940d9cd500437f3dba331ca813032da6a04e350506f60e158f8.jpg)

![](images/71c53460ac6a4cf1940a002341e3188e06e2f4ed30427ee618e43f5c80e672e2.jpg)  
Figure 14: Effects of different number of inference time steps on ∥w − wˆ <sup>′</sup>∥. From left to right, the number of inference time steps (Num Steps) were set to 10, 30, 50 (default), 70, and 90.

![](images/1dd003c5e419ed0726002d7f0f069dcbf6f18afee19cac96204575525ec06fad.jpg)

![](images/b24381f148d152c092a455da5bf7aad15c49280a60d34fc3da91bda3bb28537e.jpg)

![](images/dd42a8bd63efdc555cd3c36e33d3dbc045f917eeb078e89a7dd06e4fd9e0e73e.jpg)

![](images/1499a5290a1bb46d17c0949b0e25eb62a1fac9523ceaae862ad960a031b5e4e1.jpg)

![](images/901dc4fca500371e4e83f178486ca8ce03655e751dc53ca8cf39518ee2c8f358.jpg)

![](images/f11cac339eff8c3e8eb27b6de8c9cb5bc8d5aaab009e8a4aec264c7ae6aa7850.jpg)

![](images/d776eca0d8f52734d6e9149ecfdc779ba138461ca35e66fde903b2d619276ee7.jpg)

![](images/41cbe37a77a5b801077c90e0613178698851a62eb188cee96b9a3e14aebb3af0.jpg)

![](images/e026ad868ad1b2bd8b2563a860337364886766c93e92090027601759f459d10f.jpg)

![](images/1477c7784c4370fdd8832effb0e429d6e6d1f5185a08176e5e997a724605e41d.jpg)

![](images/44975ffbf97bc151b996f5490ed54c95f04b12e94115ab3e2cf17d9400a703e9.jpg)

![](images/a56370bca36a8f8f2921cf3523193b3b5221d395c752906e56b4ec7dbe461625.jpg)

![](images/7d01e213cf54f3c10be6fa3f27d7f922a3bffcbb833d52b31aee9fee3572441f.jpg)

![](images/f89a68b22dba8500809a80f2c3c6f87bbcd29a2382b753768282e09ee3857325.jpg)

![](images/c425cb52d85ecfcddbaa0dabe35a854e5d26163c0599063ed3c61d2cdb6f1ec6.jpg)

![](images/cfc206a316a6c1872d658746f7863464f9d3fa07ea4527c67698360e24ac9b0d.jpg)

![](images/2eb3e115de7539b68db809101a70de32d658b1708ad98e5f61586111d1a0b896.jpg)

![](images/51a3ca12471a00b4606e580b6a04228f525e8a5e1f02d04ee13a2f4ae58c817c.jpg)

![](images/a782ce59ad5877c9a975badab60056a24233afafeb4c8d6cb6a2907e17a42c4d.jpg)

![](images/f19198f32ab4cc213fabe7e4e70f716ea0f727edd0e8213fa52cfe51a2153ab8.jpg)

![](images/c6dce250a83fc1dd2240a9b7223a5c92fd20fc3ee40d0b9b4f251b96d11ca46b.jpg)

![](images/b9a93e6be33714b5046fe08f0595b7b397667ecc4b7559f1a54ba54be5c22366.jpg)

![](images/3c9375cd0a344ea209ddeca48969e0484710dde9ed53b08cbabdd2909d1baca5.jpg)

![](images/67185f2e8cb76bd380b14d4da60d3347f9cabd72569aa5f1df8fcedee85631d5.jpg)

![](images/94c5fd43a82de558bf1fdad10eea34c01ec8798c300c16dff5361def95f41677.jpg)  
Figure 15: Effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . Watermark: RingID. Thresholds: 73.0 (1e-2 FPR).

![](images/a19397b3a3a7f53769e467e5160d78e637e1386c893abf8f32ce2641d5b67de0.jpg)  
Figure 16: Effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . Watermark: RingID with channel 0 pattern multiplied by 2 and α = 128. Thresholds: 119.85 (1e-2 FPR).

![](images/96b03d5df91a5ab95835e7de1fb35b487fda199be34148d5cb9a36f4405fefce.jpg)

![](images/4a966080c38a3303cd458c9c46b6792bb463dfb207c21dd7b13bebca100077f3.jpg)

![](images/cfa17b1185afd9c958617de7f918592014212298cb38cc3a89e87ec073d7002f.jpg)

![](images/e982aac65b54ff14415f02c0b2fd3ddbd9cc07755b45bdf65dedb5c8d7092700.jpg)

![](images/5bba9f91928f98cf75b870095d8f9224b9f08da305ed202a3f3e6f46b5221f88.jpg)

![](images/546fd6f8e5e1a9369c5f107702563932e442372c0ccf4f3b0582d819aafc7790.jpg)

![](images/83e7faa2967d28d516cf53ae63b2b239598ba505bc93be69fd71de3d2c3fd4b7.jpg)

![](images/76fe67815e1cabd41ac8daefd6d6c4f50c4119d041c6daf4b1033bb7dd1ca87e.jpg)

![](images/a0e4367f16af467c9f5098cd15c23744079c080eab5776d5e4892d010e6127bf.jpg)

![](images/d37f98a18653620280ae69897b84cf39aa62e554571a5ae49c062f14f1bb2a61.jpg)

![](images/870ceb0584a59bc090e06c1c762fe5d4684caab539eab11e3079ef3f72cee8c1.jpg)

![](images/4860ef1e82d2657a7780823fc9e2c292ef81535bf5848fe76dbf950e9d252595.jpg)

![](images/2d33ce1a45d6987e1e269e9ba5bc3e80922b42ca3dfd2e095b541fb4db563ff3.jpg)

![](images/348b92e0367e12bcefda3e2744eeec958a79168692bb901ff880099877815b7f.jpg)

![](images/0ad2ac88ee231a3acc3b13a410e011a159a9b9fede6772c8e46289964c8bd716.jpg)

![](images/d97f34a57748b07a0e82723b3ad9c34ee9065fd3d1ae4189a8fea5c2a8432d60.jpg)

![](images/80c10287066e8ed2a48a553d4c5c93c9a1c2be7bfee040ac8170df666023dfa6.jpg)

![](images/d281f7c2890c3406e9a8370efab44601c47667a4d90d69b6fb1863fb50e282a7.jpg)

![](images/87041a814082dd50099e74195aa9b8c4c87540c6980fd2d6b27675ef613286cb.jpg)

![](images/45b8e07ea3307f987e13b77f4d101cee592f818cf0e4e7605dde8042aae7e0b4.jpg)

![](images/54b8f6e03724dcbd9621783f7e211c766d41d8a672d92b7541b1a55c054a9e6b.jpg)

![](images/b7484adf3e6fb59fb3941d8d08c36e33bb8cfd10e39fdef96224d0aa56ca9049.jpg)

![](images/3085c619d7e10ecd4457b2bc4bda0d6bc220360500efc191f7e6bf94e348cfcc.jpg)

![](images/7393e782c45da9ed4907ff1288237c68d8a5f2df6a0e2e002f28ec89c175fa9f.jpg)

![](images/5af419049e945e576ed8eda7bdd1635df583dfedf623050237a864fe6708c0fa.jpg)

![](images/5938d089d85864c176165853ebb36f2dd58cdc9e01f8422be9c0033f9eee4d62.jpg)

![](images/6a7f1ab9b60db0c07488fe8c58b201227ff374c652834921b696abc7e3d0f253.jpg)

![](images/5bd4c6de94448cbddabbb0d31a55ca982c88cfd53a7e1031e950496ca315d740.jpg)

![](images/cdcf3ac1d3fd6367a572cc36a8d47256a313809cf04cc0478356c56578925094.jpg)

![](images/587d326ee33001b02f1657ab989aac0cf9d2b52fef9b2c1ddd739fc996c8bdb3.jpg)

![](images/5a4246555293414426c7b8436232dcf1a9c4d4bccb219a49fcd7c11c1b1b42a5.jpg)

![](images/ce90e997730cfed2d776cc69f802712cbc1b70545dac9fc348519cc9cab3c606.jpg)

![](images/b5302b3db82ddd142ac19bc6e4040a117160c03a68636fce46ac714732acb58a.jpg)

![](images/75d240728619079cd1e897df28e31ef2fe1e906077d219a29cb329706f2bdc2a.jpg)

![](images/a376e98b2a259616205c2bea966e6c61749fcb47962bb2efd647a5e1907496de.jpg)

![](images/2834a12f88a3a2d6625ed76a236bb6ba3e36bea61463575e8dd6b572919a71dd.jpg)

![](images/4cac43e10c1325a3ce57a878f4cedfa134969d2dbf69a3e4fb239cbc315b3020.jpg)

![](images/d099a420c0aea66f85839f5e0b481271c6be93ca3abe55f2202d0bc08de37b96.jpg)

![](images/bf9a2e93d44f4c1e5c1d65cb365e3f39815994c1f557d67cb318799adc533393.jpg)

![](images/7f7d2c57e085f02ad13352de5ddecae00b81600a15547ae551d505072653b1b8.jpg)

![](images/3d36e179318e16c03964ae6220d815d3f6ff4060b11d380bbc51dadddf244a8a.jpg)

![](images/4538b87435cafbefa9fae3006fcefefd5263adc75e8d7f0a8219ba74bde83415.jpg)

![](images/c14d6a34d1dafb45748a8c24fa19d20469be51f4c8a09a6b9d4aa1b9ca3dbd9f.jpg)

![](images/bb200d508eb6eb9098e784ded4ce06f9de5406088deda80f272b187178b81c15.jpg)

![](images/a7f6267e3c3a7b7ca4ad6234237ccb603d4743f678c03f9108d1e3352f87d56a.jpg)

![](images/a78c4cd6448bdbc621326f1384c8045edf060659c1a79536d79e6ff6808593b5.jpg)

![](images/54a7ac5b82e4cdca9c62bd3b44c9644a67c9632af08cd9bfa992dc26a923e4e3.jpg)

![](images/923ae34ed6e3d7ad3fb6d67968eeb4c19d46447e6a470a153467e7e6572ef838.jpg)

![](images/b95877ac9154c09b262a2849fd03bd467570b03d8a0393f2aa9545ca842be4c7.jpg)

![](images/49beee2d19215dfc0863f8bbf607ddcc2ffdaa768f8777f935e6c292b9014eb4.jpg)  
Figure 17: The effects of different perturbations on ∥w − wˆ <sup>′</sup>∥. From left to right, the perturbations are set with different parameters (Param). Watermark: RingID. Thresholds: 73.0 (1e-2 FPR).

![](images/4b066b250a0df42bb5cb67b84c49058a12d7cd0382a49bea088c55e01384422e.jpg)  
Figure 18: The effects of different perturbations on ∥w − wˆ <sup>′</sup>∥. From left to right, the perturbations are set with different parameters (Param). Watermark: RingID with channel 0 pattern multiplied by 2 and α = 128 (RingID-2-128). Thresholds: 119.85 (1e-2 FPR).

![](images/5396d206cd25abd49db7c4671fb6ee8fe22af13770bfdc71c88c1eedf6b47c5e.jpg)

![](images/1b96aaf6e5c2077a19bcee0b716320171b4b58bf35c932856801946ec0cdf2db.jpg)

![](images/a9f0feeab367ad67dbdb0ffdc9ac40331e1e09eb0b94adca695ce37c4486be29.jpg)

![](images/5e74340b071c61d0eb3e8821c096f8fa7b06bafe0a6d873036af6fcd94843dc5.jpg)

![](images/de3c368629d69e80b686c29b02b8cba10d057e3a8df5f8d914601ba17d66e2d1.jpg)

![](images/69ea3e2903d51ddb3c406742e7097a43d1117425039dc50a2772d9ca5d0ecf77.jpg)

![](images/f90af937e643c042be6edfd3b4315434ec7e3252192519513c7f81f1fb24be66.jpg)

![](images/43526e42d48a8abc24ea8ccdf295e80e2eeaa232ee7927a3f254c9c9045afb70.jpg)

![](images/e264682afd0df8cf3e9c8a07b4b9c608b6961db15f9e1ebe0294320197228114.jpg)

![](images/69e75d3692c6fe538289a0d47989293f5b05b5b8e6fdd9bf44ecd41917879eda.jpg)

![](images/5e78f3c0485673146b2fd893ea751ae3e7c462b87619ee0c3fea036983a94d23.jpg)

![](images/3ba5de6fd9db5179b19e2a86f6e6fe3278a0af736873282f95a09b452b17931d.jpg)

![](images/d2f1b0ce6ef1cafaf366587613720bf2fcbaa249dc88429d8cd472eaf740bd97.jpg)

![](images/35002115c7b5bca76bd9755a488068159197f9ff44d1bd270b7d07329649f5c3.jpg)

![](images/81a49a95b9c5b4d2478fc8ff6e05a54ccb840e4364e357fb16f82c145fd8e5fe.jpg)

![](images/1fc37639f8b14034e50c568379034c6f1d2e52d40ea1a2ff430e370251809afb.jpg)

![](images/166b1ffeff69756e70e3c4bee6d57f3dbe94d2caebc0f53d50bbf1af113efc22.jpg)

![](images/79451b7f429c61ae541b5472c4ff922e90b6aa60a55542626be12b8aa783deb3.jpg)

![](images/293ad64ae2258d7b78d70dfe80b608d5a729b581b20f8144f575ffefab503229.jpg)

![](images/a516837cf181d474b99b87e3417a02e61e6630020f6ac023e2bfd3324ead079c.jpg)

![](images/55cb79658d7d8fece465e49c08f25266f6b91c4032d9f5fb14474637af76c102.jpg)

![](images/57d9684b03ffd66285c9f835b7fbdad03c6e0b64bb87bf14b56e5c81e7ad89fe.jpg)

![](images/0c56136a2f68d9eeae234a2c677396f0c2cacd726b7b8e2d29c6e65f5fe29f20.jpg)

![](images/762fd77ca2daa6980c685f7b6165c2a32d67dc33d80d3323f5a70c648a8be74a.jpg)

![](images/7479fca32b4ff1e766f41ae31b71679ee4bfca6e28478712262cb4d06cfb1d98.jpg)

![](images/442435d186310c72954c80681836cb992dd7f3290a8ec62963e0975c8ef977a9.jpg)

![](images/93866c603ca2e411e4a2f21b06d18a81127b94f0291d545f95889d88af5d2d1e.jpg)

![](images/15b0ea304f1e24d13fd33d233fd9786f4360c07a3bba14ec64f0587dd2a0c22f.jpg)

![](images/0452a22ce845d6402d0412fc6a6cbdd8579360df2bc4e0cd4c4aeedbb4bf1e0a.jpg)

![](images/07fc9516f0793f742616b2fd1f2d6df3180b5f53493f96e5f837344479aabf41.jpg)

![](images/54525d66d6946eb17fbf2bd4adde64504ec99be04ec32a1a026875c7cb03120f.jpg)

![](images/b60a60482609b787e5a11f1d2964588b549f95d490ea4ae979004950114aab14.jpg)

![](images/e871d5ac2f8d6c4a4d54ed0f2d28e96132ed6b08262e866da1ffb0237457532e.jpg)

![](images/10a55260c53cb3cb7391735b1846073702929e155d9d1781ef8c24addf27564e.jpg)

![](images/474b0b3c06d53a2e1367d55ccb912097541eb7cea8dee6ae66467a38b5caddc8.jpg)

![](images/2192945884b8ff8e4737bf5194d80a8df6f4d51eca3feefc769b4645fccee0c9.jpg)

![](images/785f34bb9465361ab71cba7cf3afc34c206996a5abf64eb9cc6825319b585cb6.jpg)

![](images/525e76a899321b485641e40453f811279bdbba8d06109a6da0e6068a77fb73c0.jpg)

![](images/9ca7447455dd75caf1f205c61bd7f4a8dca27445c81655b1a4cea872384ea676.jpg)

![](images/83eb6e4763ec0d244ed63375387a429c26913d8eef70fd947fa5f39e9ccde9b1.jpg)

![](images/a39600db653c87f81662f0fe6bb4a7ae43c4f42a3a67bbf93842b6a12a04c6bd.jpg)

![](images/f6c265520b67d4504a08bbd3ffdf0994e265211763ed109689f68d80dad2e4f0.jpg)

![](images/6f4fbbe45193b1f5a97a9460862949660560e65417f2a78139be567c8a6a22ba.jpg)

![](images/b45905eb779f36d332c4fe7a29e17505c95e1ca7e20302d3225d4e1efdc3c9d7.jpg)

![](images/24f86219026ccee91a260f2baa06c3f20eea70aaa3d0301fc40dc539a1854189.jpg)

![](images/bfd07e31be378fefff65a58f69de2c4fc54a7036d7b013061a8de869af869599.jpg)

![](images/05056d7b57b6ec90e8f49787d0d83f4170717701a18213b0d795814deafe7dc4.jpg)

![](images/64ce3a39e3d9bf050ebc0c99fbcbc37ba728405bfbdfd7a60e124342e5d3e76b.jpg)

![](images/6be4dc06100cab2ecfd04515ffd6b3b14ddb514c2614f276d045f4d9c8253a38.jpg)

![](images/c39d689db5a7b6725e6c3bee6552ad7cf91deb3003ab7a50680a32630d9c9f0b.jpg)  
Figure 19: The effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . From left to right, the perturbations are set with different parameters (Param). Watermark: RingID with $\alpha = 1 2 8$ (RingID-1-128). Thresholds: 73.51 (1e-2 FPR).

![](images/28df63d692d592ffb772478d350b2de44e5aac844ff65a78a387813d17a116de.jpg)

![](images/5931d5956ca5f0213b3fb3a73f2fb8c60fece9aabda16d72c788cd6ae06411d4.jpg)

![](images/526e52f255ab2e447a89cc26ace11ece4a218ceb72d002fb82c0f9280e95cc2e.jpg)

![](images/7dae4ff6a41f4a0301e284c0b90128d64d9c85ad2ed46ec1f83f3a5018733857.jpg)

![](images/2f963f30d44cc2893b4c8b850d91aa012fd9c718c0a611f891f78ea04fba66d0.jpg)

![](images/cf15be7b872fd7301890a3fcc0a9eafa94060da852c43420866a355fb738949c.jpg)

![](images/2aefa450ee3ceba051eb18d6acef9412ce95ae367e71a65062a0b240af691baa.jpg)

![](images/7b0d8688d0a975899c8091a50b607ae6f769b2c4da4f1409a5668d130b4ef72a.jpg)

![](images/b86054e3041873f424fad2f3e8ad326524e9cf6e6d66d60aa9434083f4a04aee.jpg)

![](images/9495bdefb747ca95d211dc586c0c3c6130c11c5435b70d9328891d0bc79ef9a4.jpg)

![](images/7ae3eabee2eb663a596b0b1b3e12cbb4ce63d1873fd5e09c0e12a11512bd6611.jpg)

![](images/e07796f46aca9a163096df2523bfe0c127efa4efa0ce755182afb3844978cd22.jpg)

![](images/af03d28fdab8f4afc55ef073add48aa0fa0afae18edaea6281aba04ac72d829d.jpg)

![](images/00072a5d2ec84d45f4c45efae274b2744e3d75cfab3aa4ff1810526d90c92218.jpg)

![](images/decb6be83ca2a81d7bee7788b37bbb5cbe3ad3f3fab0d122c9f2fc3fe9b92723.jpg)

![](images/32f728d75eca1c4ade4604d4d6e5ee6d19f8a3eb556fe58f4c8df1f587e09c07.jpg)

![](images/2743f8e1fc074bb7c6a0486233b23df82f8911362c5de06c8b9bada1c3a96c6c.jpg)

![](images/50871f430d4efc75618bafbde2b85050615bacc76e2f6bc78288577df7a09b30.jpg)

![](images/d9adfdf25d74f8c2984d9927c46bf125650a567b9fbba0477b18b6ff2d695673.jpg)

![](images/ba77dbc5130c273e344e0575f0debb17d241065494794ebb96d82ddb1bc94d83.jpg)

![](images/2882d275d4e86dbb95c1b4b0ad3d0c1ed732fba85550517a6ead920cf116a82c.jpg)

![](images/6db2ef5b352dec7f0d2b27451c63bac5af3366140e4a29258c4e7cb51b838432.jpg)

![](images/1eaf31e0bb2843d0e2fc88e2f54b138a4a3a023d3607bca0b53f0b334a21ee30.jpg)

![](images/749280dfb10714f06003b8f25a2e52a8259c2c881cd082116e1cb5a7bbdffa5e.jpg)

![](images/a69dea53ac7488c647ef335015a5645e5cebe3b10c452be382a309b7297fb508.jpg)

![](images/55ebccd0eb96c7e9f6a7e9d7a927129e7c5fa5c4b866cfca2483aad0c7b291d8.jpg)

![](images/af47647861a61a027868b2aa3d3475fd57df8bcb30feefd414bb4db9e2c655a3.jpg)

![](images/af9817bb79581f647d0a3d60c8ec395cfa160ea750b2704e5a42243682628bf3.jpg)

![](images/3296fe783a3ffe984566261e219c5297d5399bfb8c24aa5278d4fc82bdfef864.jpg)

![](images/dbfa11491bf74657414dbc05432e3ac8c4d8507c0f5651f0311751744c56bceb.jpg)

![](images/b305c115fa42efd3fd0e8f39be2aeb26135b753628d14f73d9b91763d5501126.jpg)

![](images/55b1209a7f9a4cf61a311efa35c1abdc79d74fa560f9ba601c1c5aa7fcfded0a.jpg)

![](images/306276f3490e556145f244bb60db138060d727ea3b656e777290f4119d1f5b88.jpg)

![](images/6aa74065b3c726e1b087d5d26875789b1cf7101938e1f7a9503a8fc525905deb.jpg)

![](images/89bdcc3c19f0d6a798f8d1d887997316efa722f7c569dd7456fdfabbd5e1634d.jpg)

![](images/98fc5b8f1a27df7413494964e02716c306a6b554c42a49fcee28cd971eddae48.jpg)

![](images/4848e6569c8902928857b55efe1d5a198075aecb97176fac35d06a401aeffe79.jpg)

![](images/193789f3e34cc729fde4399d4a6dbc4f0fa75588939e0eff86044d1ef386bb72.jpg)

![](images/e9b5de897a2a5240953000a1f3655edc1769add44047bf7f61c727f233c6874d.jpg)

![](images/d1a242521f252bc86f580db93a96c9ac1fb5eef9fcaa69e81232454cbc7ef60a.jpg)

![](images/6d964cbb3ef37c5987c2794f5741a4db39caa1041d987f3493427c47d1f1d2a1.jpg)

![](images/5bf7d45e438f2821d85dc0262225ff87591d1d14b45740ab701e6ba1fc82646b.jpg)

![](images/39f086974f2373f52a6b408712a1da6cb55b1f20c70abf3c5290737245d31815.jpg)

![](images/53912906af1b658e71b85899f43a22ab926958e2d68e2b3dd2a80948580dc8e8.jpg)

![](images/ff69067532c32a8b3c1587f79b0c83e9b2d1eef69b1ef0ed280ec102174b6dd5.jpg)

![](images/f6b7576bfd83eb33c12a8a02f31f148ed666c1685d99febabcd16c906ad6e90b.jpg)

![](images/0b4c009b85a21b3abb974fb6f7ad80db731ec85073649df4fc1437d0d85c5403.jpg)

![](images/31dc10b42d714bd1929828a9b39f8c855ce1e75277e5c564ee7322471e0aa288.jpg)

![](images/1c4b5b779f69e83d33e1485b36f28558f4a83839a5b9828dbca34a873facb1aa.jpg)

![](images/4ec4d8e86e170d99e7cbc05c1954533894b74363d784d4046c31d60fd5c9536e.jpg)  
Figure 20: The effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . From left to right, the perturbations are set with different parameters (Param). Watermark: HSTR. Thresholds: 50.39 (1e-2 FPR).

![](images/742c5a5b85fab98b38182a82cd5a1ebb8a8f0056bac60b3b186524e424813386.jpg)

![](images/3e4d29c3ab35a49a62b37422fabe69e24f268c5226e823474f80517f5aa65bdd.jpg)

![](images/b1eba64e5e69daf74f3b8af2720ef4174374d35cae4d87ef39e51b47474c1923.jpg)

![](images/8afa9b9e51c598b85241e591dc1ec5f27c68381c67268058b497a7bcdb0d86c2.jpg)

![](images/72568cad4a5ebaef02c41b0d635fb7d4df5206583c72ebc7d38ec5db89a35623.jpg)

![](images/8607a7a8308cadc2e31d5fb734567a92fb6a683cfc3d645090226a753fb71c26.jpg)

![](images/a36002b014f370fa4aaa0d9573b13749317c90acf4d897d46ce6ba834fbebb13.jpg)

![](images/98abca98fe75d3e71d5b31a2a294719a6bf8f7a7418b7838d5adc591a6b6fc18.jpg)

![](images/16de9a083246413ecdc0160e437b3a17d3bc31d11f5edb35cde7c41fe56d1b36.jpg)

![](images/fe6b4c5d832bf30864184951d33f52144e14e341c66db5f5308b32921364de71.jpg)

![](images/80f641924b73745a0fdba33025afaf8abf1131cd2e06bf12c7397ffa90e7467a.jpg)

![](images/a68bf18e8584ea032c40316e45cb3f3692bc8b7e34416c0b28930a7bbc98f223.jpg)

![](images/3e0a3a8d4062612158d3047e7d1283b11123aa6e69a58b84dc003ec309ddecb6.jpg)

![](images/5ea437863a3ad9336bd9d94596e3c1f6fe3ade1f3b24c3a56947231d09174a4a.jpg)

![](images/7b08f90891710a56072ecbd632e4ec24575d54e0097814b8032d4ef8d8a4342f.jpg)

![](images/55d4ff594772381fa759b60d3dfdf9930b4da456555479f84e8e685f55ebab5c.jpg)

![](images/7313c4a47944529b003440cefac326ac1ab707d4c595f77420b525095a86dca2.jpg)

![](images/edf31acdfab03ec6a98b48a2adf733805a1461e42b5f7f07be31e972c13ab8a1.jpg)

![](images/a67e5d4021695ef8b5195dd4ea66161f15e28f555b1b8f0c381280bcd794468b.jpg)

![](images/c01576cf70629f3d19bbeb5fc2310e169c33270980655b982da6adc3f59dd561.jpg)

![](images/88f52452d008d38c2d1efa34b555612127baaad42aea1428f7f7c34aadafff5e.jpg)

![](images/a72c5f2283b56151348da7be21c9e7ebfccb31c7edb70831ed69bf794387761e.jpg)

![](images/49c6136aeb8f1f762dbf2a0ff573d390ebf369e2163f9761d91cbd27ba072fea.jpg)

![](images/b649b16480b41c68867e00db5f489ad29c6605b7abd2f7f382390708ae250772.jpg)

![](images/4f94686ec298290968d160269c545987afc901ed10323f82058081cea0769674.jpg)

![](images/3919539c96c825a8508a6cbfc7c7adddb8afe2a116a984532b3789945b0fb5cc.jpg)

![](images/f45af648734572103c562dacb88d15543e682007f0bbf3b5ccf20423fc60753a.jpg)

![](images/e8ce4ec192298ab15eb56bd70b3c921f33875bce716c77a0802f5ec0004a4db3.jpg)

![](images/a3a8c309f691535d2f7c4223efeddb76e58430c9495db12342f0208c58aca1bb.jpg)

![](images/b6e0eeeb2deeb2136a8b4ea3c4bbfe82c155fdcb99408bebf3246c0bfd071369.jpg)

![](images/5da36eb2da14ee72416f73047a3fae912be26a2ce4d56406c5797fd1cd6b8793.jpg)

![](images/e05abfa2631886ed32b449e70dc6de21814bbdc8380151348e3c9315b0a5f29e.jpg)

![](images/28de06bf2dd11d1f00d66ffe23df72237981048245506dac28feb6330e6a0135.jpg)

![](images/0e826aa0ea08207d9322e2f2d431a3e6ab48830dcb6e59b334c803e13e1501b8.jpg)

![](images/44fe16088ad2567402f449b274a6060f62e86be78d0ea4a399477b20afb6244c.jpg)

![](images/1e73339d108cae344a6d53836e49c8cba7adb0514dce1efeaa972945a7c43d9c.jpg)

![](images/50ac5c5ba8377c460779004eb1ff0bc7fc7d3be5d487469afa26c459ef33d9e6.jpg)

![](images/0d03fff816a6a80a7f42d1ecfa27049f0bbc8116febeb4f7029a9f45784be087.jpg)

![](images/38d605a5cc10214d2d6ec1b6c4a8dd91dff02ccf1896a9928d69a6561f0987b5.jpg)

![](images/0a1d419c8becb653da2dd21678a828efd80641860d1ccae95372f8159ffdf59d.jpg)

![](images/0cd3529487a0cd6b409efbd50c5de321eaad5e012af40b5f5c152b1eb60e7603.jpg)

![](images/df2fc4ad289debe425126980e062a737784893f52f1c05eb9083772244ad8a54.jpg)

![](images/086dae52c4cf4f5e4d20849543254b9ae9a2ab9a45e1db69c5814a80476390c0.jpg)

![](images/d30464fff3b4ea8e90afe71e76957ba7ba235e6dc3987467b6f672cadac0e6f1.jpg)

![](images/42533f60aa1aaabd42fbf22100bb6858e8978618a6b6b9bb061340c2e316d40b.jpg)

![](images/1a37f7bad50aecd68293f1b468093ace60fba8235ec628f6652770f95b65a187.jpg)

![](images/401ca81c65a3336fd14a7072d0ad9c400b18494661785027196d7daf65e7c7f8.jpg)

![](images/51330707f222c6802139651efb753517e2213b30078ec486b85558dcd1818111.jpg)

![](images/614dad1ce2fbd02da31084637ff42757b1f8ca7866d281b9f2c2981bca2b8985.jpg)

![](images/89c0205831dd5c1da0a2fc3df50f3262f1d2c873563fac7ebb20fb3412fea935.jpg)

![](images/5040c1eac43f73db3774badb762e7d52b17f4f3d9ea9ffe88b534e0a7d6d2356.jpg)  
Figure 21: The effects of different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert$ . From left to right, the perturbations are set with different parameters (Param). Watermark: HSTR with channels 0 and 3 pattern multiplied by 2 (HSTR-2-2). Thresholds: 82.15 (1e-2 FPR).

![](images/cdf9297176ca63204728fad9a14bf7766bd23ea81890ba3715d5ed1d32187e2e.jpg)  
Figure 22: The experimental thresholds for different perturbations on ∥w − wˆ <sup>′</sup>∥, each with different parameters on x-axis. The value on y-axis represents the potential threat that wˆ <sup>′</sup> is getting far from w, which corresponds to Eq. (7). Watermark: RingID.  
Figure 23: The experimental thresholds for different perturbations on $\lVert \mathbf { w } - \hat { \mathbf { w } } ^ { \prime } \rVert .$ , each with different parameters on x-axis. The value on y-axis represents the potential threat that wˆ <sup>′</sup> is getting far from w, which corresponds to Eq. (7). Watermark: HSQR.

Table 4: Attacker: StirMark 3.1.79. Defense: Tree-Ring [48] (watermark: ring). Diffusion model: LDM. Number of testing images: watermarked: 1000 / unwatermarked: 1000. Generated image size: 3x512x512. Latent size: 4x64x64.
<table><tr><td>Attack</td><td>AUC</td><td>ACC</td><td>TPR@0.01FPR</td></tr><tr><td>Median filter 2x2</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Median filter 3x3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Median filter 4x4</td><td>1.0000</td><td>0.9995</td><td>0.9990</td></tr><tr><td>Gaussian filter 3x3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 90</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 80</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 70</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 60</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 50</td><td>1.0000</td><td>0.9995</td><td>1.0000</td></tr><tr><td>JPEG 40</td><td>1.0000</td><td>0.9975</td><td>0.9990</td></tr><tr><td>JPEG 30</td><td>0.9999</td><td>0.9970</td><td>0.9980</td></tr><tr><td>JPEG 20</td><td>0.9998</td><td>0.9945</td><td>0.9970</td></tr><tr><td>JPEG 10</td><td>0.9968</td><td>0.9760</td><td>0.9360</td></tr><tr><td>FMLR (Frequency Mode Laplacian Removal)</td><td>1.0000</td><td>0.9990</td><td>1.0000</td></tr><tr><td>Color reduce</td><td>1.0000</td><td>0.9995</td><td>1.0000</td></tr><tr><td>Sharpening 3x3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>1 column, 1 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>5 column, 1 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>1 column, 5 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>17 column, 5 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>5 column, 17 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Cropping 1% off</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Cropping 2% off</td><td>1.0000</td><td>0.9995</td><td>1.0000</td></tr><tr><td>Cropping 5% off</td><td>1.0000</td><td>0.9990</td><td>1.0000</td></tr><tr><td>Cropping 10% off</td><td>0.7880</td><td>0.7190</td><td>0.1070</td></tr><tr><td>Cropping 15% off</td><td>0.6690</td><td>0.6275</td><td>0.0620</td></tr><tr><td>Cropping 20% off</td><td>0.8747</td><td>0.7975</td><td>0.2470</td></tr><tr><td>Cropping 25% off</td><td>0.9265</td><td>0.8490</td><td>0.3330</td></tr><tr><td>Cropping 50% off</td><td>0.7645</td><td>0.7090</td><td>0.0470</td></tr><tr><td>Cropping 75% off</td><td>0.6154</td><td>0.5915</td><td>0.0280</td></tr></table>

Table 5: Attacker: StirMark 3.1.79. Defense: Tree-Ring [48] (watermark: ring). Diffusion model: LDM. Number of testing images: watermarked: 1000 / unwatermarked: 1000. Generated image size: 3x512x512. Latent size: 4x64x64.
<table><tr><td>Attack</td><td>AUC</td><td>ACC</td><td>TPR@0.01FPR</td></tr><tr><td>Linear (1.007, 0.010, 0.010, 1.012)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Linear (1.010, 0.013, 0.009, 1.011)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Linear (1.013, 0.008, 0.011, 1.008)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Aspect ratio change (0.80, 1.00)</td><td>0.6752</td><td>0.6300</td><td>0.0650</td></tr><tr><td>Aspect ratio change (0.90, 1.00)</td><td>0.9999</td><td>0.9955</td><td>0.9990</td></tr><tr><td>Aspect ratio change (1.00, 0.80)</td><td>0.9206</td><td>0.8400</td><td>0.4470</td></tr><tr><td>Aspect ratio change (1.00, 0.90)</td><td>1.0000</td><td>0.9995</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.00, 1.20)</td><td>0.8612</td><td>0.7755</td><td>0.2640</td></tr><tr><td>Aspect ratio change (1.00, 1.10)</td><td>1.0000</td><td>0.9990</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.10, 1.00)</td><td>1.0000</td><td>0.9990</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.20, 1.00)</td><td>0.9781</td><td>0.9270</td><td>0.7360</td></tr><tr><td>Rotation 1.00</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Rotation 2.00</td><td>1.0000</td><td>0.9985</td><td>1.0000</td></tr><tr><td>Rotation 5.00</td><td>0.9584</td><td>0.8880</td><td>0.5500</td></tr><tr><td>Rotation 10.00</td><td>0.8968</td><td>0.8150</td><td>0.2930</td></tr><tr><td>Rotation 15.00</td><td>0.9339</td><td>0.8560</td><td>0.4210</td></tr><tr><td>Rotation 30.00</td><td>0.8914</td><td>0.8100</td><td>0.2280</td></tr><tr><td>Rotation 45.00</td><td>0.8735</td><td>0.7865</td><td>0.1920</td></tr><tr><td>Rotation 90.00</td><td>0.9994</td><td>0.9935</td><td>0.9950</td></tr><tr><td>Flipping</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Rotation scale 1.00</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Rotation scale 10.00</td><td>0.8993</td><td>0.8180</td><td>0.3770</td></tr><tr><td>Rotation scale 15.00</td><td>0.9373</td><td>0.8620</td><td>0.4510</td></tr><tr><td>Rotation scale 30.00</td><td>0.8942</td><td>0.8155</td><td>0.2080</td></tr><tr><td>Rotation scale 45.00</td><td>0.8767</td><td>0.7885</td><td>0.2720</td></tr><tr><td>Rotation scale 90.00</td><td>0.9994</td><td>0.9935</td><td>0.9950</td></tr><tr><td>Shearing x-0% y-1%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-1% y-0%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-1% y-1%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-0% y-5%</td><td>1.0000</td><td>0.9970</td><td>1.0000</td></tr><tr><td>Shearing x-5% y-0%</td><td>1.0000</td><td>0.9965</td><td>0.9980</td></tr><tr><td>Shearing x-5% y-5%</td><td>0.9999</td><td>0.9950</td><td>0.9960</td></tr><tr><td>Random bending</td><td>0.9999</td><td>0.9985</td><td>0.9990</td></tr></table>

Table 6: Attacker: StirMark 3.1.79. Defense: RingID [4] (watermark: ring & noise). Diffusion model: LDM. Number of testing images: watermarked: 1000 / unwatermarked: 1000. Generated image size: 3x512x512. Latent size: 4x64x64.
<table><tr><td>Attack</td><td>AUC</td><td>ACC</td><td>TPR@0.01FPR</td></tr><tr><td>Median filter 2x2</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Median filter 3x3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Median filter 4x4</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Gaussian filter 3x3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 90</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 80</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 70</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 60</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 50</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 40</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 30</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 20</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>JPEG 10</td><td>1.0000</td><td>0.9995</td><td>1.0000</td></tr><tr><td>FMLR (Frequency Mode Laplacian Removal)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Color reduce</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Sharpening 3x3</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>1 column, 1 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>5 column, 1 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>1 column, 5 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>17 column, 5 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>5 column, 17 row removed</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Cropping 1% off</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Cropping 2% off</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Cropping 5% off</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Cropping 10% off</td><td>0.8307</td><td>0.7505</td><td>0.1030</td></tr><tr><td>Cropping 15% off</td><td>0.5457</td><td>0.5415</td><td>0.0120</td></tr><tr><td>Cropping 20% off</td><td>0.8277</td><td>0.7535</td><td>0.1070</td></tr><tr><td>Cropping 25% off</td><td>0.5457</td><td>0.5450</td><td>0.0120</td></tr><tr><td>Cropping 50% off</td><td>0.3713</td><td>0.5020</td><td>0.0070</td></tr><tr><td>Cropping 75% off</td><td>0.5412</td><td>0.5425</td><td>0.0150</td></tr></table>

Table 7: Attacker: StirMark 3.1.79. Defense: RingID [4] (watermark: ring & noise). Diffusion model: LDM. Number of testing images: watermarked: 1000 / unwatermarked: 1000. Generated image size: 3x512x512. Latent size: 4x64x64.
<table><tr><td>Attack</td><td>AUC</td><td>ACC</td><td>TPR@0.01FPR</td></tr><tr><td>Linear (1.007, 0.010, 0.010, 1.012)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Linear (1.010, 0.013, 0.009, 1.011)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Linear (1.013, 0.008, 0.011, 1.008)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Aspect ratio change (0.80, 1.00)</td><td>0.9996</td><td>0.9920</td><td>0.9920</td></tr><tr><td>Aspect ratio change (0.90, 1.00)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.00, 0.80)</td><td>0.9999</td><td>0.9935</td><td>0.9950</td></tr><tr><td>Aspect ratio change (1.00, 0.90)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.00, 1.20)</td><td>1.0000</td><td>0.9980</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.00, 1.10)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.10, 1.00)</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Aspect ratio change (1.20, 1.00)</td><td>1.0000</td><td>0.9980</td><td>0.9980</td></tr><tr><td>Rotation 1.00</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Rotation 2.00</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Rotation 5.00</td><td>0.9777</td><td>0.9260</td><td>0.6050</td></tr><tr><td>Rotation 10.00</td><td>0.6227</td><td>0.5955</td><td>0.0080</td></tr><tr><td>Rotation 15.00</td><td>0.8210</td><td>0.7435</td><td>0.1100</td></tr><tr><td>Rotation 30.00</td><td>0.4720</td><td>0.5060</td><td>0.0070</td></tr><tr><td>Rotation 45.00</td><td>0.3698</td><td>0.5005</td><td>0.0020</td></tr><tr><td>Rotation 90.00</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Flipping</td><td>1.0000</td><td>0.9995</td><td>1.0000</td></tr><tr><td>Rotation scale 1.00</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Rotation scale 10.00</td><td>0.6313</td><td>0.6040</td><td>0.0090</td></tr><tr><td>Rotation scale 15.00</td><td>0.8145</td><td>0.7415</td><td>0.1070</td></tr><tr><td>Rotation scale 30.00</td><td>0.4687</td><td>0.5080</td><td>0.0050</td></tr><tr><td>Rotation scale 45.00</td><td>0.3762</td><td>0.5000</td><td>0.0020</td></tr><tr><td>Rotation scale 90.00</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-0% y-1%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-1% y-0%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-1% y-1%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-0% y-5%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-5% y-0%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Shearing x-5% y-5%</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Random bending</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr></table>