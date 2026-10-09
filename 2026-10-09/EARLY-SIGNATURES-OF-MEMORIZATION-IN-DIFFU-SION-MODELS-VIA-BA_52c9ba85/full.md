# EARLY SIGNATURES OF MEMORIZATION IN DIFFU-SION MODELS VIA BASIN GEOMETRY AND CYCLIC DENOISING

Nikhil Verma<sup>1,∗∗</sup>, Siddharthan Dileep<sup>1,∗∗</sup>, Anoop Singh<sup>2</sup>

Srikanth Sastry<sup>3</sup>, Ramya Hebbalaguppe<sup>4</sup>, Sayan Ranu<sup>1,2</sup>, N. M. Anoop Krishnan<sup>1,5</sup> <sup>1</sup>Yardi School of Artificial Intelligence, Indian Institute of Technology Delhi

<sup>2</sup>Department of Computer Science and Engineering, Indian Institute of Technology Delhi

<sup>3</sup>Jawaharlal Nehru Centre for Advanced Scientific Research

<sup>4</sup> TCS Research Labs

<sup>5</sup>Department of Civil and Environmental Engineering, Indian Institute of Technology Delhi <sup>∗∗</sup>Equal contribution

## ABSTRACT

Diffusion models generalize early in training and later reproduce individual training samples. Standard tests detect memorization only once one-shot generation already produces near-copies, which leaves a released model unaudited until its outputs fail. We show that memorization is encoded in the geometry of the learned energy landscape before it appears in generated samples, a state we call latent memorization. Using score divergence and basin volume, we find that localized basins form around individual training samples and separate them from held-out samples before the first memorized sample appears, with an onset that follows the same O(n) scaling as the memorization time. We probe these basins with cyclic denoising, the repeated application of partial noising followed by denoising. Under the exact empirical score, we prove that cycling started near an isolated training sample recovers it and returns to it over any finite number of cycles with high probability, with a bound controlled by the cycling noise and the separation from competing samples. In trained models, cycling recovers training images from CelebA and CIFAR-10 checkpoints whose one-shot samples contain no copies, and at a CelebA checkpoint with 0.1% one-shot copies, 500 cycles raise the memorized fraction above 30%. Cycling also reveals degenerate attractors that match no single training image. They are prevalent early in training and fade as training proceeds, so residence in a basin does not by itself imply memorization. These findings hold on a Gaussian mixture, CelebA, and CIFAR-10 across optimizers, architectures, noise schedules, and training-set sizes. They also extend to offthe-shelf Stable Diffusion v1.4, where the cycled conditional–unconditional divergence gap separates memorized from non-memorized prompts with an AUC of 0.944 and a TPR of 0.866 at 1% FPR. Altogether, probing the learned landscape uncovers memorization that eludes one-shot generation. More broadly, what a diffusion model has memorized is a property of the geometry and stability of its learned distribution, and assessing it requires examining this structure rather than generated outputs alone.

## 1 INTRODUCTION

Diffusion models (DMs) generate high-quality and diverse samples by learning to reverse a gradual noising process (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2020b). The noising process is a stochastic differential equation (SDE) that transforms data into Gaussian noise, and its time reversal requires the score, the gradient of the log-density of the noised data. A neural network trained to approximate this score acts as a force field that guides samples from noise back to data. The training objective that enables generalization also admits memorization. For a finite training set, the loss-minimizing score is the score of the training data convolved with Gaussian noise, and exac sampling with this score drives every trajectory onto a training point as the noise vanishes (Gu et al., 2023). At this optimum, the model reproduces only its training data. Practical DMs do not reach this optimum immediately. They first generate novel samples and begin to reproduce individual training examples only after further training (Bonnaire et al., 2025). Such reproduction has been documented in large image DMs (Somepalli et al., 2023a;b), and generate-and-filter attacks extract training images from deployed models (Carlini et al., 2023). Memorization therefore has direct consequences for privacy, copyright, and the protection of sensitive data.

![](images/33ff07d8f642e2103dcf073f0a6a853d150bf07ad5791f569fee7dc5e09110ad.jpg)  
Figure 1: Memorization forms sharp, localized basins in the learned score field. (A,B) Normalized score divergence $\widehat { A } _ { 1 }$ for an eight-dimensional Gaussian mixture model (GMM), on a twobdimensional slice of the sample space obtained with the projection of Bihani et al. (2024) (App. G.6). (A) In the generalized regime, the field is smooth, with no prominent structure around training samples. (B) As memorization emerges, strongly negative regions form around training samples. (C) Schematic of cyclic denoising: at low noise, trajectories remain within a basin, whereas at sufficiently large noise they jump to neighbouring basins.

Detecting memorization in a released model is therefore a central problem for auditing DMs. A released model is a single checkpoint, usually without its training curves and often without its training set. The standard test generates samples and counts near-copies of training images, and this test registers memorization only once one-shot generation already reproduces the training set. A model may, however, encode individual training samples before any of them appears in its outputs. We cal this state latent memorization: the model has formed structure specific to individual training samples, yet one-shot generation produces no copies. Detecting it early, in the auditing sense, means detecting it from the model in hand, before copies appear in generated samples and without access to the training curve.

Related Work. The generalization-to-memorization transition is well-characterized along two axes, model capacity and training-time, for models whose training is analyzed. Along the first, DMs generate novel samples when their effective capacity is insufficient to reproduce the training set (Yoon et al., 2023). Along the second, Bonnaire et al. (2025) and Favero et al. (2025) identify a generalization time $\tau _ { \mathrm { g e n } } ,$ , at which high-quality novel samples appear, and a memorization time $\tau _ { \mathrm { m e m } } ,$ , beyond which samples reproduce training examples. Since $\tau _ { \mathrm { m e m } }$ grows linearly with the number of training samples $n ,$ a generalization window $[ \tau _ { \mathrm { g e n } } , \tau _ { \mathrm { m e m } } ]$ widens with n. Training and test losses separate before $\tau _ { \mathrm { m e m } } .$ , but this signal requires the training curve. Energy-based analyses connect DMs to dense associative memories and show that, as the dataset grows, the learned landscape passes from attractors centred on individual training examples, through spurious attractors, to generalized states with vanishing basin volume (Ambrogioni, 2023; Pham et al., 2025). Memorization is also local. Isolated examples in sparse regions of data space are memorized first (Merger & Goldt, 2026), and generated samples drift towards the training set while the held-out loss still improves (Garnier-Brun et al., 2026).

For trained models, geometric detectors measure the learned score through conditional– unconditional score differences, Hessian-based sharpness, directional anisotropy, or p-Laplacian estimates (Wen et al., 2024; Jeon et al., 2025; Asthana & Belagiannis, 2026; Brokman et al., 2025b;a). These detectors are validated on models and prompts whose memorization is already visible in generated samples. Cyclic denoising (CD), the repeated application of partial noising followed by denoising, offers a dynamical probe of the same landscape. Kang et al. (2026) characterize how such chains mix across the data manifold at fixed noise, and Sharma & Martiniani (2026) use CD to extract training images from fixed trained models (see App. B for further related work). Together, these works characterize memorization once it is visible in generated samples, or along axes that require access to training. There is limited understanding on how the geometry underlying memorization develops before copies appear in generated samples, and about what CD reveals in that regime. We ask whether memorization leaves geometric and dynamical signatures in the learned score before one-shot generation reproduces the training set, and to what extent these signatures can be read from the model alone.

Contributions. In this work, we show that memorization is encoded in the geometry of the learned score before it is expressed by single-shot generation. We study this through the time-dependent energy landscape $E _ { t } ( x ) = - \log p _ { t } ( x )$ learned by the model (Fig. 1). As training proceeds, this landscape develops localized basins around individual training samples. Within the generalization window, these basins capture little of the probability mass reached from pure noise, so one-shot sampling rarely lands in them. CD changes this picture. Each cycle perturbs a sample within a neighborhood set by the noise amplitude γ and denoises it back, effectively leading to stable regions of the landscape characterized by local minima, for instance, a training-sample basin. Thus, repeated cycling exposes basins that a single draw may miss. Our contributions are as follows.

• Geometric signatures of memorization. Using score divergence and a recovery-based basin volume, evaluated on training, held-out, and generated samples across training, we show that training-sample basins form before one-shot generation reproduces those samples. The onset of these signatures scales with $\tau / n ,$ as does τ<sub>mem</sub>.

• Latent memorization. Applying CD across checkpoints and noise amplitudes, we expose latent memorization inside the generalization window. CD raises the memorized fraction from at most 0.1% to over 30%. CD also reveals degenerate attractors, stable states that match no single training image, which are prevalent near $\tau _ { \mathrm { g e n } }$ and vanish as training proceeds.

• Theoretical proof. Under exact stochastic reversal with the empirical score, we derive a lower bound on the probability that CD repeatedly returns to a training point, controlled by the cycling noise and by the separation of that point from competing training points.

• Empirical evaluation. We establish these signatures on a Gaussian mixture model (GMM), CelebA, and CIFAR-10 across two optimizers, three architectures, two noise schedules, and a range of training-set sizes, and show that CD detects memorized prompts in off-the-shelf Stable Diffusion v1.4 (Rombach et al., 2022) without retraining.

## 2 METHOD

Sampling from a DM (Ho et al., 2020; Song et al., 2020b) simulates probability-flow or Langevin dynamics governed by the learned score field $s _ { \theta } ( x _ { t } , t )$ , where θ denotes the learned parameters (see App. E.1 for the preliminaries). This score defines an energy landscape through the time-dependent energy $E _ { t } ( x ) = - \log p _ { t } ( x )$ , since $s _ { t } ( x ) = - \nabla _ { x } E _ { t } ( x )$ . Because the score points toward increasing probability, it also points toward decreasing energy, and high-probability regions are low-energy regions. Each local minimum of $E _ { t }$ defines the centre of a basin, the set of points that flow toward it. In the memorization regime, individual training samples can form its own low-energy basins (Pham et al., 2025; Ambrogioni, 2023), which makes basin geometry a natural lens on memorization. In the remainder of the paper, we use this energy-landscape view to study memorization through the landscape learned by the model, $E _ { t , \theta } ( x ) = - \log p _ { t , \theta } ( \bar { x } )$

![](images/e4a79aeb9bfa9d04f15bdf305ca7795b6c8600c4ee45e409bfab18f7077c33f0.jpg)

![](images/deff4e2243816b0da0f17c7c4fbdac7b5a8c3a0c5a96ca56114a84c98a59b2ae.jpg)

![](images/5118cf81086d65c71b4e3b7391e96473260f8778bb0eedf4703a8a094351a67e.jpg)  
Figure 2: Geometric signatures across training on CelebA. (A) Training loss (solid), held-out loss (dashed), and FID (dotted, right axis) for $n = 1 0 2 4$ . (B) $\widehat { A } _ { 1 }$ at $\gamma = 0 . 0 0 1$ and $f _ { \mathrm { m e m } }$ (dotted, right axis) for $n = 5 1 2$ b(orange), 1024 (blue), and 2048 (green); training (solid), held-out (dashed), and generated (dash-dot) samples. Inset: divergence against $\tau / n$ . (C) Mean log basin volume for $n = 1 0 2 4$ over all training samples and an equal number of generated and held-out samples, with line styles as in (B); bands show one standard deviation across samples. Refer Appendix G for equivalent results on Adam Optimizer

## 2.1 GEOMETRIC SIGNATURES OF THE ENERGY LANDSCAPE

We characterize the learned landscape by two basin-level signatures, evaluated on training, held-out, and generated samples: score divergence $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } , t )$ and basin volume. The score divergence measures how sharply the landscape contracts around a point, and the basin volume measures the size of the neighborhood from which denoising returns to it. Evaluating them on training samples requires the training set, whereas evaluating them on generated and held-out samples requires only the model and reference data from the same distribution.

Score divergence and its estimation. Near a locally convex energy minimum, the learned score $s _ { \theta } ( x , t )$ can draw nearby points toward the basin centre. We characterize this net local contraction by the divergence $\nabla _ { x } \cdot \boldsymbol { s } _ { \theta } ( x , t )$ For an exact score, this divergence equals $\mathrm { t r } ( \nabla _ { x } ^ { 2 } \log p _ { t } ( x ) ) =$ $- \operatorname { t r } ( \nabla _ { x } ^ { 2 } E _ { t } ( { \bar { x } } ) )$ , the negative sum of the local energy curvatures. A more negative value near a basin centre therefore indicates stronger local concentration. If training forms localized basins around individual examples, the divergence near training samples should therefore fall below that near heldout samples before one-shot generation reproduces the training set. For the reported measurements we use the boundary Monte Carlo method of Brokman et al. (2025b). It computes the unscaled mean outward score flux on a ball of radius $R _ { \mathrm { p } }$ around the forward-perturbed sample $x _ { t }$ . We denote this reported estimate by $\pmb { A } ( \pmb { x } , \pmb { t } )$ and call it score divergence for convenience. It estimates the divergence averaged over the ball up to the known positive factor $R _ { \mathrm { p } } / d ,$ as detailed in App. G.3.4. Because this estimate depends on both the magnitude and the direction of the boundary scores, we isolate direction through the normalized score divergence $\hat { \boldsymbol { \mathcal { A } } } ( \boldsymbol { x } , t )$ , the mean outward alignment of bthe unit-normalized boundary scores. We use this name for convenience. $\widehat { A }$ lies in [−1, 1] when bthe sampled score norms are nonzero, and negative values indicate predominantly inward-pointing scores. Unless stated otherwise, we evaluate both statistics at the final denoising step $( t = 1 )$ and denote them by $\mathbf { A _ { 1 } }$ and $\hat { \ b { A } } _ { 1 }$

Basin volume. We estimate the basin volume with the DDIM recovery sweep of Pham et al. (2025). For a target sample xˆ, let $\gamma _ { c }$ be the critical noise amplitude (see Section 2.2) from which deterministic DDIM denoising recovers xˆ within a tolerance $\epsilon _ { \mathrm { r e c } }$ , and let $x _ { \gamma _ { c } }$ be the perturbed input at that amplitude. The distance $r _ { \mathrm { r e c } } ( \hat { x } ) = \| \hat { x } - x _ { \gamma _ { c } } \| _ { 2 }$ defines a recovery radius, and the basin volume is the volume of a d-dimensional ball with this radius, $\begin{array} { r } { V _ { \mathrm { b a s i n } } ( \hat { x } ) = \frac { \pi ^ { d / 2 } } { \Gamma ( d / 2 + 1 ) } r _ { \mathrm { r e c } } ( \hat { x } ) ^ { d } . \ V _ { \mathrm { b a s i n } } } \end{array}$ is thus an isotropic proxy for the recovery volume and does not equal the exact basin volume.

## 2.2 CYCLIC DENOISING AS AN ENERGY LANDSCAPE PROBE

The signatures above describe the landscape at individual points; CD probes how samples move through it. At cycle k, we noise the current sample to timestep t with the forward marginal $\mathcal { Q } _ { t }$ and denoise it back, $x ^ { ( k ) } \ { \overset { \ Q _ { t } } { \longrightarrow } } \ x _ { t } ^ { ( k ) } \ { \overset { p _ { \theta } } { \longrightarrow } } \ x ^ { ( k + 1 ) }$ <sup>)</sup>,where $p _ { \theta } ( x ^ { ( k + 1 ) } \mid x _ { t } ^ { ( k ) } )$ is the distribution induced by running the learned reverse process from t to 0. We parametrize each cycle by the normalized timestep $\gamma = t / T$ , which we call the cycling amplitude.Because noise schedules vary in how they allocate noise over time, evaluating them at the same normalized timestep $( \gamma )$ results in differing noise standard deviations $( \sigma _ { t } )$ (see App. E.2).

The cycling amplitude sets the scale of landscape exploration. Small $\gamma$ produces predominantly local motion, while large $\gamma$ produces stronger perturbations that make transitions between basins more likely. A trajectory therefore tends to remain in its current basin while each perturbation stays within the region from which denoising returns it to that basin, and it escapes more often as perturbations grow (Fig. 1C).To follow these dynamics, we track four quantities across cycles. The consecutive-cycle similarity $S _ { \mathrm { c y c } } ^ { ( k ) } = \cos ( \phi _ { \mathrm { L P I P S } } ( x ^ { ( k ) } ) , \phi _ { \mathrm { L P I P S } } ( x ^ { ( k - 1 ) } ) )$ , where $\phi _ { \mathrm { L P I P S } }$ denotes the LPIPS feature representation (Zhang et al., 2018), measures whether a trajectory has settled. We say that a trajectory resides in a basin from its residence onset, the first sustained crossing of 95% of the rise in $S _ { \mathrm { c y c } } ^ { ( k ) }$ from its early-cycle level to its late-cycle level, and we refer to the canonical image of that basin as its attractor. Beyond residence, we make this identification through the memorized fraction $f _ { \mathrm { m e m } }$ , which under CD is the fraction of trajectories whose state at cycle k satisfies the same nearest-neighbour criterion used for one-shot generation. When a trajectory resides in a basin whose attractor matches no single training image under this criterion, we call the attractor degenerate. Finally, we track $\mathbf { A _ { 1 } }$ and $\hat { \mathbf { A } } _ { 1 }$ along each trajectory to relate these dynamics to the local geometry of the basins it visits.

## 2.3 THEORETICAL ANALYSIS OF BASIN RESIDENCE UNDER CYCLIC DENOISING

Our experiments suggest that once a cycling trajectory reaches the basin of a memorized training image, subsequent cycles can repeatedly recover that image. To isolate this mechanism, we analyze CD under exact stochastic reversal of the empirical training distribution, following the exact empirical-score idealization(Biroli et al., 2024).

Assumption 1 (Exact empirical reversal). The clean distribution is $\begin{array} { r } { N ^ { - 1 } \sum _ { j = 1 } ^ { N } \delta _ { x _ { j } } } \end{array}$ for distinct training points $x _ { 1 } , \ldots , x _ { N } \in \mathbb { R } ^ { d }$ , with $N \geq 2 .$ . Each cycle noises its current state x to $U = x + { \sqrt { q } } Z ,$ where $Z \sim { \mathcal { N } } ( 0 , I )$ and the cycling variance $q > 0$ isfixed, then samples a clean endpoint using the exact stochastic reverse process. Forward and reverse randomness are independent across cycles.

The additive-noise form of Assumption 1 is a rescaling of the variance-preserving forward process used in our experiments. Noising to timestep t gives $\dot { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } x + \sqrt { 1 - \bar { \alpha } _ { t } } Z$ , and the rescaled state $U = x _ { t } / \sqrt { \bar { \alpha } _ { t } }$ equals $x + { \sqrt { q } } Z$ with $q = ( 1 - { \bar { \alpha } _ { t } } ) / { \bar { \alpha } _ { t } }$ . The cycling variance is therefore set by the cycling amplitude through $q ( \gamma ) = ( 1 - \bar { \alpha } _ { \gamma T } ) / \bar { \alpha } _ { \gamma T }$ , and it increases with $\gamma .$ . Throughout, residence refers to the clean cycle endpoints; intermediate noisy states and reverse trajectories need not remain near the target.

Theorem 1 (Recovery and cyclic residence). Under Assumption 1, fix a training point $x _ { i } ,$ , let $d _ { i j } =$ $\| x _ { i } - x _ { j } \|$ , and choose $\begin{array} { r } { 0 < R < \frac { 1 } { 4 } \operatorname* { m i n } _ { j \neq i } d _ { i j } } \end{array}$ . For $0 \leq r \leq R ,$ , define

$$
A _ { i } ( r ) = \operatorname* { m i n } \left\{ 1 , \frac { 1 } { 2 } \sum _ { j \neq i } \exp \left[ - \frac { d _ { i j } ( d _ { i j } - 4 r ) } { 8 q } \right] \right\} .
$$

Let $h _ { i } ( x )$ be the probability that one cycle starting at x ends at $x _ { i \cdot } I f \| x - x _ { i } \| \leq r ,$ then $h _ { i } ( x ) \geq$ $1 - A _ { i } ( r )$ . Consequently, for $X _ { 0 } = x$ with $\lVert x - \bar { x _ { i } } \rVert \leq R$ and any integer $M \geq 1$

$$
\mathbb { P } _ { x } \left( X _ { 1 } = \cdot \cdot \cdot = X _ { M } = x _ { i } \right) = h _ { i } ( x ) h _ { i } ( x _ { i } ) ^ { M - 1 } \geq \left[ 1 - A _ { i } ( R ) \right] \left[ 1 - A _ { i } ( 0 ) \right] ^ { M - 1 } .\tag{1}
$$

Implications. The theorem separates initial recovery from subsequent residence. Let $\begin{array} { r l } { \delta _ { i } } & { { } = } \end{array}$ $\operatorname* { m i n } _ { j \neq i } d _ { i j }$ . A cycle that starts within distance $R < \delta _ { i } \bar { / 4 }$ of $x _ { i }$ returns to $x _ { i }$ with high probability when the cycling variance $q$ is small relative to the separation margin $\delta _ { i } ( \delta _ { i } - 4 R )$ . Once recovered, each subsequent cycle starts at $x _ { i }$ itself, and since $A _ { i } ( \mathbf { \hat { 0 } } ) \leq A _ { i } ( R )$ , the return bound after recovery is at least as strong as the initial recovery bound. For any fixed number of cycles, the residence lower bound therefore approaches one as $q \to 0$ , which in the experiments corresponds to small cycling amplitude $\gamma .$ The same bound shows how geometry controls residence. A larger neighbourhood radius or greater cycling noise weakens it, whereas greater separation from competing training points strengthens it. At the opposite extreme, as $q  \infty$ , the endpoint distribution from any fixed starting state approaches the uniform distribution over the N training points, and the probability of M consecutive endpoints at $x _ { i }$ approaches $N ^ { - M }$ . The theorem thus provides an idealized mechanism for persistent residence. Once a trajectory reaches a suitable neighbourhood, cycling repeatedly recovers the same training image. This mechanism has a defined scope. The analysis assumes an exactly memorized score, and it describes residence after a trajectory reaches the neighbourhood of a training point, not how the trajectory first reaches that neighbourhood from a distant initialization (see App. F for background, the complete proof, and empirical checks).

## 3 EXPERIMENTS

In this section, we evaluate the geometric signatures and cyclic-denoising dynamics across training regimes defined relative to $\tau _ { \mathrm { g e n } }$ and $\tau _ { \mathrm { m e m } } .$ , on three datasets with several training configurations and on an off-the-shelf Stable Diffusion model, and establish the following.

• Geometric signatures of memorization (§ 3.2). Score divergence and basin volume separate training and generated samples from held-out samples before one-shot generation reproduces any training sample. The onset of this separation scales with $\tau / n$ , as does τ<sub>mem</sub>.

• Latent memorization (§ 3.3). Within the generalization window, CD increases the memorized fraction from at most 0.1% to over 30%. CD also reveals degenerate attractors, which are prevalent near $\tau _ { \mathrm { g e n } }$ and vanish as training proceeds.

• Stable Diffusion (§ 3.4). Without retraining, CD recovers training images from initial generations that are not copies, and it increases the detection power of $\Delta \mathcal { A } _ { 1 }$ at strict false-positive rates.

## 3.1 EXPERIMENTAL SETUP

Datasets. We train from scratch on three datasets (synthetic, CelebA, and CIFAR-10) and evaluate a pretrained model (Stable Diffusion) on a prompt collection. The synthetic dataset is an eight-dimensional, equally weighted GMM, $\begin{array} { r } { P _ { 0 } = \hat { \Omega } \mathcal { N } ( \hat { \mathbf { 1 } } _ { d } , I _ { d } ) + \frac { 1 } { 2 } \mathcal { N } ( - \mathbf { 1 } _ { d } , \hat { I } _ { d } ) } \end{array}$ , whose population score is available analytically, with training sets of size $n \in \{ 1 2 8$ , 256, 512, 1024, 2048, 4096}. For CelebA (Liu et al., 2015), we centre-crop and resize images to $3 2 \times 3 2$ grayscale and use subsets of size n ∈ {512, 1024, 2048, 4096}. For CIFAR-10 (Krizhevsky, 2009), we use images at their native $3 2 \times 3 2$ colour resolution and subsets of size $n \in \{ 2 0 4 8 , 4 0 9 6 , 8 1 9 2 \}$ . For Stable Diffusion, we select memorized, partially memorized, non-memorized and plain english prompts from Webster (2023), Wen et al. (2024) and Hong et al. (2024).

Models and training. For the GMM, we train residual multilayer perceptrons (MLPs) with fullbatch stochastic gradient descent (SGD). For CelebA, we train attention-based U-Nets with SGD or Adam. The GMM and CelebA models use a linear noise schedule from $\beta _ { 1 } = 1 0 ^ { - 4 } \mathrm { t o } \beta _ { T } =$ $2 \times 1 0 ^ { - 2 }$ over $T = 1$ ,000 diffusion timesteps. For CIFAR-10, we train unconditional pixel-space diffusion transformers (DiT-S/2) (Peebles & Xie, 2022) with Adam, using a cosine noise schedule (Nichol & Dhariwal, 2021) with offset $s = 0 . 0 0 8 \mathrm { o v e r } T = 5 0 0$ diffusion timesteps. We continue training until $f _ { \mathrm { m e m } }$ is high, so that the saved checkpoints span pre-generalization, generalization, and memorization. We evaluate Stable Diffusion v1.4 off the shelf, without retraining or fine-tuning (see App. H for full training and evaluation details).

Cyclic-denoising protocol. We apply CD at three representative checkpoints: the checkpoint with the best generation quality before overfitting, a checkpoint after overfitting but before $\tau _ { \mathrm { m e m } }$ , and a checkpoint after $\tau _ { \mathrm { m e m } } .$ . At each, we freeze the model and initialize trajectories either from its training images or from n fresh DDIM generations, matching the training-set size. For a cycling amplitude γ, we set $t _ { \gamma } = \mathrm { r o u n d } [ \gamma ( T - \bar { 1 } ) ]$ ], noise each image to $t _ { \gamma }$ , and denoise it back to $t = 0$ with the DDPM ancestral sampler, drawing fresh forward noise at every cycle. Each output initializes the next cycle, and the initial images form cycle zero. Trajectories are independent; batching serves only computation. For Stable Diffusion, we apply CD in latent space.

Evaluation metrics. Geometry. At each checkpoint, we generate samples with DDIM (Song et al., 2020a) and evaluate $\mathcal { A } _ { 1 } , \widehat { \mathcal { A } } _ { 1 }$ , and basin volume on training, generated, and held-out samples (Secbtion 2.1). We also report the training and held-out losses at the fixed low-noise level $\gamma = 0 . 0 1$

Memorization and sample quality. Following Yoon et al. (2023); Gu et al. (2023), we classify a generated sample as memorized when the ratio of its Euclidean distances to its nearest and secondnearest training neighbours is below $\kappa = 1 / 3$ , and $f _ { \mathrm { m e m } }$ is the percentage of generated samples so classified. Error bars denote one bootstrap standard deviation of the mean over 1,000 resamples of the generated set. Sample quality is measured by the Frechet inception distance (´ FID) (Heusel et al., 2017) against disjoint held-out sets for CelebA and CIFAR-10, and by a Monte Carlo estimate of $D _ { \mathrm { K L } } ( P _ { \theta } | | \mathbf { \bar { P } _ { 0 } } )$ from 10,000 generated samples for the GMM.

Cyclic dynamics. We track $S _ { \mathrm { c y c } } ^ { ( k ) } , A _ { 1 } , \widehat { A } _ { 1 }$ , and $f _ { \mathrm { m e m } }$ as functions of the cycle index k and the cycling amplitude γ (§ 2.2).

Stable Diffusion. We use two copy-detection quantities built on the $\ell _ { 2 }$ -normalized SSCD descriptor ϕ (Pizzi et al., 2022). The similarity to training data, $\operatorname { \mathrm { 3 S C D } } ( x , p ) = \operatorname* { m a x } _ { g \in G ( p ) } \langle \phi ( x ) , \phi ( { \bar { g } } ) \rangle$ compares a sample with the set $G ( \boldsymbol { p } )$ of training images matched to prompt p. The seed agreement, $\begin{array} { r } { \mathbf { S } \mathbf { A } ( p ) = \binom { S } { 2 } ^ { - 1 } \sum _ { i < j } \langle \phi ( x ^ { ( i ) } ) , \phi ( x ^ { ( j ) } ) \rangle } \end{array}$ ⟩, is the mean pairwise similarity of the $S = 4$ generations of $p$ under different seeds and measures basin collapse without reference to any training image. We also track the conditional–unconditional gap $\pmb { \Delta } \bar { \pmb { A _ { 1 } } } = \mathcal { A } _ { 1 } ( \pmb { x } ; c _ { p } ) - \mathcal { A } _ { 1 } ( \pmb { x } ; \epsilon )$ , where $c _ { p }$ is the CLIP text embedding of $p$ and ϕ that of the empty prompt, which forms the unconditional branch of classifier-free guidance. Subtracting the unconditional branch removes the curvature the model has at x independently of any prompt (see App. G.3 for definitions and implementations of all metrics).

## 3.2 EVOLUTION OF GEOMETRIC SIGNATURES ACROSS TRAINING

Figure 2 shows the losses, the mean normalized score divergence $\widehat { A } _ { 1 }$ on training, held-out, and generated samples, the mean log basin volume, and $f _ { \mathrm { m e m } }$ bacross training on CelebA at $n = 1 0 2 4$

Divergence separates before one-shot memorization. The held-out loss reaches its minimum at $\tau = 4 5 , 0 0 0$ and then rises while the training loss continues to fall, which marks the overfitting transition (Fig. 2A). The divergence of training and held-out samples coincides until $\tau \approx 4 5 , 0 0 0$ and separates at $\tau _ { \mathrm { d i v } } = 6 5 , 0 0 0$ (Fig. 2B). The first non-zero $f _ { \mathrm { m e m } }$ appears only at $\tau _ { \mathrm { m e m } } = 9 0 , 0 0 0$ 25,000 steps after $\tau _ { \mathrm { d i v } }$ , and $f _ { \mathrm { m e m } }$ exceeds 1% only at $\tau = 1 5 5 , 0 0 0$ . Between $\tau _ { \mathrm { d i v } }$ and $\tau _ { \mathrm { m e m } } .$ , the model is in the latent-memorization regime: the score field contracts around training samples, yet one-shot generation reproduces none of them.

Dependence on noise level and dataset size. The separation appears at small noise levels. Across the $\gamma$ sweep at every checkpoint (Fig. 20), the divergence stays in a narrow band near zero in the generalized regime, whereas in the memorized regime it is strongly negative at small $\gamma$ and returns toward zero as $\gamma$ grows. Small $\gamma$ resolves structure on the scale of individual basins and large $\gamma$ averages over many, so we report $\gamma = 0 . 0 0 1$ The timing depends on the dataset size. Both the separation and the rise in $f _ { \mathrm { m e m } }$ occur later for larger $n ,$ and replotting against $\tau / n$ collapses the onsets for $n \in \{ 5 1 2$ , 1024, 2048} (Fig. 2B, inset). The geometric signature thus follows the $O ( n )$ scaling established for $\tau _ { \mathrm { m e m } }$ (Bonnaire et al., 2025).

Basin volume and generated samples. Basin volume separates in the same direction (Fig. 2C). Its mean grows for training and generated samples and decreases for held-out samples. Its standard deviation decreases for training and generated samples and stays roughly constant for held-out samples, which indicates that basins form increasingly uniformly around training examples. From $\tau = 5 { , } 0 0 0$ onward, the divergence of generated samples follows that of training samples with a mean absolute difference of 0.015, and the two distributions are almost identical across training configurations (App. G). Generated samples can therefore replace the training set in the contrast with held-out samples, which makes the signature readable from the model and held-out reference data alone. Appendix G.4 uses this property for membership inference from score divergence.

## 3.3 CYCLIC DENOISING DYNAMICS

We now test whether the basins that form in the latent-memorization regime are reachable through CD, on CelebA $( n = 4 0 9 6$ , Adam) and CIFAR-10 $\left( n = 8 1 9 2 \right) \left( { \mathrm { F i g } } . 3 \right)$

CD recovers training images that one-shot generation misses. At CelebA $\tau = 1 0 { , } 0 0 0$ and CIFAR-$1 0 \tau = 8 3 , 0 0 0$ , one-shot samples exhibit $f _ { \mathrm { m e m } } = 0 \%$ , yet $f _ { \mathrm { m e m } }$ rises with cycling (Fig. 3B,E). At Celeb $\mathrm { ~ \ i ~ } \tau = 2 0 , 0 0 0 .$ where one-shot $f _ { \mathrm { m e m } }$ is $0 . 1 \% .$ , cycling at $\gamma = 0 . 1 0$ raises $f _ { \mathrm { m e m } }$ above 30% by cycle 500. Continued exploration brings more trajectories into basins of individual training samples, although only a subset of training samples form such basins.

The outcome depends on the cycling amplitude and the training stage. In several CelebA settings, $f _ { \mathrm { m e m } }$ plateaus within 500 cycles (Fig. 3B). At that $\gamma ,$ , few additional trajectories find detectable training-sample basins within the observed horizon, although exploration need not have stopped. $\mathrm { A t } \mathrm { C I F A R - } 1 \bar { 0 } \tau = 2 0 0 , 0 0 0$ and $\gamma = 0 . 3 0 , f _ { \mathrm { m e m } }$ instead keeps rising through cycle 500 (Fig. 3E). At CelebA $\gamma = 0 . 4 0$ , neither $f _ { \mathrm { m e m } }$ nor $S _ { \mathrm { c y c } } ^ { ( k ) }$ rises appreciably, even at $\tau = 2 0 , 0 0 0$ (Fig. 3B,C). Perturbations at this amplitude exceed the range over which the accessible basins remain stable, and no residence occurs. This trend matches Thm. 1, where larger cycling variance weakens the residence bound.

Geometry changes before identification. Individual trajectories follow the same order of events observed across training (App. Fig. 14). Their images approach particular training examples over hundreds of cycles (A,D). The drop in $\boldsymbol { A } _ { 1 }$ often begins before the trajectory is persistently classified as memorized (B,E), and $S _ { \mathrm { c y c } } ^ { ( k ) }$ then reaches a high plateau (C,F), consistent with continued residence. Score divergence also decreases across cycles at the population level (App. G). Within a trajectory, as within training, the geometry changes first and identification with a training sample follows.

![](images/4803f4099c8ed1e512e9d9ca111089dadf9990934c1eb48c90d9cff0b0ed99c4.jpg)  
Figure 3: Cyclic denoising exposes latent memorization inside the generalization window. CelebA $( n = 4 0 9 6 .$ , Adam; A–C) and CIFAR-10 $( n = 8 1 9 2 ; \mathbf { D - F } )$ at checkpoints from $\tau _ { \mathrm { g e n } }$ to $\tau _ { \mathrm { { m e m } } } . \mathbf { \Gamma } ( \mathbf { A } , \mathbf { D } )$ Generated samples at $c _ { 0 }$ and c<sub>500</sub> $( \gamma = 0 . 1 0 )$ with their nearest training images. (B,E) $f _ { \mathrm { m e m } }$ and $( \mathbf { C } , \mathbf { F } ) \ S _ { \mathrm { c y c } } ^ { ( k ) }$ across cycles for several $\gamma .$

Residence alone does not establish memorization. At the earlier $\tau _ { g e n }$ CelebA and CIFAR-10 checkpoints, $S _ { \mathrm { c y c } } ^ { ( k ) }$ approaches 0.9 at some amplitudes while $f _ { \mathrm { m e m } }$ stays near zero (Fig. $^ { 3 \mathrm { B } , \mathrm { C } , \mathrm { E } , \mathrm { F } ) }$ . These trajectories reside in degenerate attractors. On CIFAR-10 at $\gamma = 0 . 0 5$ , degenerate attractors hold 50.9% of trajectories at $\tau _ { \mathrm { g e n } } = 4 9 , 5 0 0 , 1 6 . 0 \%$ at the generalization-window checkpoint $\tau = 8 3 , 0 0 0 .$ and 0.98% by $\tau = 2 0 0 { , } 0 0 0$ Their appearance changes with training (Fig. 4). The $\tau _ { \mathrm { g e n } }$ states include stripes, nearly monochromatic images, and mixed-colour patterns, whereas some states at $\tau = 8 3$ ,000 contain recognizable objects while matching no single training image. In both cases, $S _ { \mathrm { c y c } } ^ { ( k ) }$ rises above 0.9 and saturates, often alongside a decrease in $\mathcal { A } _ { 1 }$ . Sharma & Martiniani (2026) also observe such regions in the landscape and refer to these images as trivial points. Residence in these attractors is sensitive to the cycling amplitude: at $\tau _ { \mathrm { g e n } }$ , no trajectory passes the residence screen at $\gamma = 0 . 2 0 \mathrm { o r } 0 . 3 0 .$ . Appendix G reports further results on the same.

These results separate two properties that CD measures. $S _ { \mathrm { c y c } } ^ { ( k ) }$ and $\mathcal { A } _ { 1 }$ detect residence in a basin and require only the model, whereas identifying that basin with a training sample requires the $f _ { \mathrm { m e m } }$ test and the training set. Inside the generalization window, the share of degenerate attractors falls while $f _ { \mathrm { m e m } }$ under cycling rises, so residence increasingly occurs in training-sample basins that one-shot generation does not reach. How degenerate basins arise and lose accessibility as training proceeds remains an open question.

## 3.4 EXPERIMENTS ON STABLE DIFFUSION

We next test whether these observations extend to off-the-shelf Stable Diffusion v1.4, using prompts from Webster (2023), Wen et al. (2024) and Hong et al. (2024).

Individual chains. The six chains in App. Fig. 21A each isolate one factor. The hippo chain reaches its training image at $c _ { 1 }$ and stays there through $c _ { 2 0 0 }$ . The two laura chains share a prompt, a seed, and a $\iota \ : c \ : _ { 0 }$ image with $\mathrm { S S C D 0 . 2 0 3 }$ to the training image; the chain at $\gamma = 0 . 1$ 1 never arrives, whereas the chain at $\gamma = 0 . 9$ settles from $c _ { 3 5 }$ . The two air-con chains share a prompt and γ and differ only in seed: one enters its basin early, leaves it near $c _ { 1 0 0 }$ , and returns, whereas the other arrives only at $c _ { 5 0 }$ . The label tape chain arrives at $c _ { 7 0 }$ from a $c _ { 0 }$ image that bears no resemblance to its training image. Arrival thus depends on the cycling amplitude and the seed, can occur long after cycling begins, can be transient, and does not require the initial image to resemble the training image. The laura and label tape chains show latent memorization in a large-scale model: the initial generation is not a copy, and cycling recovers the training image. In all six chains, $\operatorname { S S C D } ( x , p )$ and $\Delta \mathcal { A } _ { 1 }$ change together at arrival and departure (App. Fig. 21B,C).

D τ = 83, 000 | f = 0.0% | FID=98.3  
A τ = 49, 500 | f<sub>mem</sub> = 0.0% | FID=54.0  
![](images/8bc620fcbf251fa42686ae3c9f86deb5591c76d468c9823aa32f2f1a2140a711.jpg)

![](images/f16e88e1009cd5f167e24a03e71a2e7f9843a52d6963371b3ca1ac55fc442be4.jpg)

![](images/ab81b094b403a03fed72b8ef6e1b41f9b8c55742e242af1a350d4b24fa5ab922.jpg)

![](images/e4a9ebf6c05e964e31453550bd4ef7e2eeaeed5c36e71d7284db6a889e97415c.jpg)

![](images/b6c467d4b91f4c02a1883d61b02d79e29f09078126405018c7f23e2b26dbd679.jpg)  
Figure 4: Cyclic denoising reveals degenerate attractors. CIFAR-10 $( n = 8 1 9 2 , \gamma = 0 . 0 5 )$ at $\tau _ { \mathrm { g e n } } = 4 9 , 5 0 0$ (A–C) and $\tau = 8 3 , 0 0 0$ (D–F). (A,D) Three trajectories from $c _ { 0 }$ to c<sub>500</sub> with the nearest training image to the final state; no attractor matches a single training image. $^ { ( \mathbf { B } , \mathbf { E } ) \mathcal { A } _ { 1 } }$ and $( \mathbf { C } , \mathbf { F } ) S _ { \mathrm { c y c } } ^ { ( k ) }$ across cycles. Markers show the residence onset, which does not indicate memorization.

Population behaviour. Memorized prompts show rising $\operatorname { S A } ( p )$ and increasingly negative $\Delta \mathcal { A } _ { 1 }$ under cycling, whereas non-memorized prompts change little in either (App. Fig. 21E). After 200 cycles, $\Delta \mathcal { A } _ { 1 }$ is graded by the degree of memorization and stays near zero for non-memorized and plain-English controls (App. Fig. 21D).

Detection. On 500 memorized and 2,000 non-memorized prompts from (Wen et al., 2024), cycling at $\gamma = 0 . 9$ raises the AUC of $\Delta \mathcal { A } _ { 1 }$ from 0.811 at $c _ { 0 }$ to 0.944 at $c _ { 6 0 }$ , and its TPR at 1% FPR from 0.526 to 0.866 (App. I). The conditional–unconditional subtraction matters at strict false-positive rates: the conditional divergence alone reaches a similar AUC of 0.948 but a TPR at 1% FPR of only 0.194. On the same prompts and seeds, the single-shot detector of Wen et al. (2024) achieves a higher AUC of 0.990 and a comparable TPR at 1% FPR of 0.838. The comparison depends on the negative source. Against LAION negatives, which match the training distribution, $\Delta \mathcal { A } _ { 1 }$ reaches a TPR at 1% FPR of 0.820 against 0.740 for the single-shot detector, whereas the single-shot detector remains stronger against COCO, Lexica, and random-token negatives (App. I). CD thus transfers to a model trained at scale without retraining. It reveals when and how a trajectory reaches a memorized image, and it adds detection power at strict false-positive rates on in-distribution negatives.

## 4 CONCLUSION

We studied how memorization develops in diffusion models through the geometry of the learned energy landscape. Localized basins form around individual training examples before one-shot generation produces copies. Score divergence and basin volume separate training from held-out samples in this latent-memorization regime, and their onset follows the same $O ( n )$ scaling as $\tau _ { \mathrm { m e m } }$ . Cyclic denoising reaches the training-sample basins that one-shot sampling misses and recovers training images from models whose one-shot samples contain none. Residence under cycling alone does not imply memorization: degenerate attractors are prevalent near $\tau _ { \mathrm { g e n } }$ and fade as training proceeds. Under the exact empirical score, a well-separated training point retains cycling trajectories with a probability controlled by the cycling noise and its separation from competing points. These behaviours hold across a Gaussian mixture, CelebA, CIFAR-10, and off-the-shelf Stable Diffusion v1.4. Memorization is encoded in the landscape geometry before it is expressed in generated samples, and probing this geometry offers a route to auditing models whose outputs appear free of copies. Limitations and future works arising from the present work are discussed in App. C.

## AI USE STATEMENT

All AI-assisted suggestions, analyses, and manuscript changes were reviewed by the authors. The authors independently verified the reported experimental results and mathematical claims and take full responsibility for the final technical content, analyses, and conclusions of the paper. We used the Stanford Agentic Reviewer (https://paperreview.ai) as author-side review tools to obtain preliminary feedback and identify points requiring further clarification or validation.

## ETHICS STATEMENT

This work studies the emergence of memorization in diffusion models through the geometry of the learned energy landscape and the dynamics of cyclic denoising. Understanding these dynamics can inform future work on privacy auditing, training data protection, and mitigating unintended memorization in generative models. Our experiments use publicly available datasets and models in controlled research settings. The analysis is intended to improve understanding of memorization, thereby supporting the development of safer and more privacy-preserving generative models.

## REPRODUCIBILITY STATEMENT

The code will be made publicly available upon acceptance of the paper. The appendix details the full experimental protocol, including hyperparameters, model and training details, sampling configurations, data preprocessing and splits, and metric definitions and computation, together with the implementation details needed to reproduce the principal results.

## REFERENCES

Luca Ambrogioni. In search of dispersed memories: Generative diffusion models are associative memory networks, 2023.

Rohan Asthana and Vasileios Belagiannis. Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability, 2026.

Vaibhav Bihani, Srikanth Sastry, Sayan Ranu, and N M Anoop Krishnan. Low-dimensional projections for visualizing energy landscapes of atomic systems. In AIfor Accelerated Materials Design - Vienna 2024, 2024. URL https://openreview.net/forum?id=x3ryxZgHgu.

Giulio Biroli, Tony Bonnaire, Valentin De Bortoli, and Marc Mezard. Dynamical regimes of dif-´ fusion models. Nature Communications, 15(1):9957, November 2024. ISSN 2041-1723. doi: 10.1038/s41467-024-54281-3.

Tony Bonnaire, Raphael Urfin, Giulio Biroli, and Marc M¨ ezard. Why Diffusion Models Don’t´ Memorize: The Role of Implicit Dynamical Regularization in Training. 2025. doi: 10.48550/ ARXIV.2505.17638.

Jonathan Brokman, Itay Gershon, Omer Hofman, Guy Gilboa, and Roman Vainshtein. Tracking memorization geometry throughout the diffusion model generative process. In NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations, 2025a.

Jonathan Brokman, Amit Giloni, Omer Hofman, Roman Vainshtein, Hisashi Kojima, and Guy Gilboa. Identifying Memorization of Diffusion Models Through p-Laplace Analysis. In Tatiana A. Bubba, Romina Gaburro, Silvia Gazzola, Kostas Papafitsoros, Marcelo Pereyra, and Carola-Bibiane Schonlieb (eds.),¨ Scale Space and Variational Methods in Computer Vision, volume 15667, pp. 295–307. Springer Nature Switzerland, Cham, 2025b. ISBN 978-3-031-92365-4 978-3-031-92366-1. doi: 10.1007/978-3-031-92366-1 23.

Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer,\` Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In Proceedings of the 32nd USENIX Conference on Security Symposium, Sec ’23, USA, 2023. USENIX Association. ISBN 978-1-939133-37-3.

Alessandro Favero, Antonio Sclocchi, and Matthieu Wyart. Bigger Isn’t Always Memorizing: Early Stopping Overparameterized Diffusion Models, 2025.

Jerome Garnier-Brun, Luca Biggio, Davide Beltrame, Marc Mezard, and Luca Saglietti. Biased ´ Generalization in Diffusion Models, 2026.

Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, and Ye Wang. On Memorization in Diffusion Models, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Neural Information Processing Systems, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models, 2020.

Chunsan Hong, Tae-Hyun Oh, and Minhyuk Sung. MemBench: Memorized Image Trigger Prompt Dataset for Diffusion Models, 2024.

Dongjae Jeon, Dueun Kim, and Albert No. Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes. In Proceedings of the 42nd International Conference on Machine Learning, pp. 27091–27112. PMLR, 2025.

Hyunmo Kang, Noam Itzhak Levi, Corinna Elena Wegner, Daniel J. Korchinski, and Matthieu Wyart. Sampling Data with Chains of Forward-Backward Diffusion Steps, 2026.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In 2015 IEEE International Conference on Computer Vision (ICCV), pp. 3730–3738, December 2015. doi: 10.1109/ICCV.2015.425.

Claudia Merger and Sebastian Goldt. Local Coverage Governs Memorization in Diffusion Models, 2026.

Alex Nichol and Prafulla Dhariwal. Improved Denoising Diffusion Probabilistic Models, 2021.

William S. Peebles and Saining Xie. Scalable diffusion models with transformers. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4172–4182, 2022.

Bao Pham, Gabriel Raya, Matteo Negri, Mohammed J. Zaki, Luca Ambrogioni, and Dmitry Krotov. Memorization to Generalization: Emergence of Diffusion Models from Associative Memory, 2025.

Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, and Matthijs Douze. A selfsupervised descriptor for image copy detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14532–14542, June 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10674–10685, 2022. doi: 10.1109/ CVPR52688.2022.01042.

Rishabh Sharma and Stefano Martiniani. Cyclic Denoising Reveals Ultrastable Memories in Diffusion Models, 2026.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 2256–2265, Lille, France, July 2015. PMLR.

Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? Investigating data replication in diffusion models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6048–6058, 2023a. doi: 10.1109/CVPR52729.2023.00586.

Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and Mitigating Copying in Diffusion Models. In Advances in Neural Information Processing Systems 36, pp. 47783–47803, New Orleans, Louisiana, USA, 2023b. Neural Information Processing Systems Foundation, Inc. (NeurIPS). ISBN 978-1-7138-9911-2. doi: 10.52202/075280-2071.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models, 2020a.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-Based Generative Modeling through Stochastic Differential Equations, 2020b.

Ryan Webster. A reproducible extraction of training images from diffusion models, 2023. URL https://arxiv.org/abs/2305.08694.

Yuxin Wen, Yuchen Liu, Chen Chen, and Lingjuan Lyu. Detecting, Explaining, and Mitigating Memorization in Diffusion Models, 2024.

TaeHo Yoon, Joo Young Choi, Sehyun Kwon, and Ernest K. Ryu. Diffusion probabilistic models generalize when they fail to memorize. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023.

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

## APPENDIX CONTENTS

A Notation 13   
B Extended Related Works 13   
C Limitations and future works 15   
D Broader Impact 15   
E Some Preliminaries 16   
E.1 Diffusion Models 16   
E.2 γ to $\sigma _ { t }$ 16   
F Theoretical Insight 17   
F.1 Background . 17   
F.2 Proof of Theorem 1 18   
F.3 Experimental Verification . 19   
G Additional Results 20   
G.1 Results on the GMM 20   
G.2 Cyclic Denoising 23   
G.3 Metrics 30   
G.4 Membership Inference from Score Divergence . 33   
G.5 Other Train Results . 35   
G.6 Visualizing the Score-Divergence Landscape 36   
G.7 How the Score Function Evolves Across Training Regimes 36   
H Finer Experiment Details 38   
I Detection on Stable Diffusion 40   
I.1 Data . 40   
I.2 Uncertainty 40   
I.3 Results by cycle . 41   
I.4 Comparison . 41   
I.5 Raw divergence . 42   
I.6 By negative source 42

## A NOTATION

Table 1 summarizes the notation used throughout the paper.

## B EXTENDED RELATED WORKS

Memorisation in Diffusion Models. Bonnaire et al. (2025) distinguish a generalization time $\tau _ { \mathrm { g e n } } ,$ when high-quality novel samples first appear, from a later memorization time $\tau _ { \mathrm { m e m } } .$ , when generated samples begin reproducing the training set. They find that both loss-level overfitting and visible memorization occur on $O ( n )$ training timescales, where n is the number of training samples. These events do not coincide: training and test losses separate while the memorization fraction $f _ { \mathrm { m e m } }$ remains near zero. Favero et al. (2025) report similar memorization-time scaling across image and language diffusion models and describe generalization and memorization as competing training timescales. Both works motivate early stopping.

Connecting diffusion models with dense associative memories, Pham et al. (2025) vary dataset size while holding the model and training procedure fixed. Small datasets yield attractors associated with individual training examples; intermediate sizes yield spurious attractors; and larger datasets yield generalized states on flatter manifolds. Their geometric measurements distinguish these regimes: memorized states have the largest measured basin volumes, spurious states have smaller but nonzero basins, and generalized states have nearly vanishing basin volume. Their comparison varies the data regime, whereas our analysis follows basin geometry along the optimization axis of a fixed model and dataset.

Table 1: Notation used throughout the paper.
<table><tr><td>Notation</td><td>Meaning</td><td colspan="2">Notation</td><td>Meaning</td></tr><tr><td colspan="5">Diffusion model</td></tr><tr><td> $x _ { 0 }$ </td><td>Clean data sample.</td><td> $\mathbf { \Psi } _ { T } ^ { x _ { t } }$ </td><td></td><td>Noisy sample at diffusion timestep t.</td></tr><tr><td>t</td><td>Diffusion timestep.</td><td></td><td></td><td>Total number of diffusion timesteps.</td></tr><tr><td> $\beta _ { t }$ </td><td>Forward-process variance schedule.</td><td> $\alpha _ { t } ~ = ~ 1 -$ </td><td></td><td>Signal-retention coefficient.</td></tr><tr><td> $\bar { \alpha } _ { t }$ </td><td>Cumulative signal-retention coefficient.</td><td>βt €</td><td>2</td><td>Gaussian noise in the forward process.</td></tr><tr><td>二  $\Pi _ { s = 1 } ^ { t } \alpha _ { s }$ </td><td></td><td> $\mathcal { N } ( 0 , I )$ </td><td></td><td></td></tr><tr><td> $\bar { \epsilon } _ { \theta } ( x _ { t } , t )$ </td><td>Network prediction of injected noise.</td><td> $q ( \dot { x } _ { t } \mid \dot { x } _ { 0 } )$ </td><td></td><td>Forward diffusion distribution.</td></tr><tr><td> $p _ { \theta } \left( x _ { t - 1 } \right)$ </td><td>Learned reverse transition.</td><td> $p _ { t } ( x )$ </td><td></td><td>Data density at diffusion time t.</td></tr><tr><td> $x _ { t } )$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\begin{array} { r l } { s _ { t } ( x ) } & { { } = } \end{array}$ </td><td>Exact score function.</td><td> $s _ { \theta } ( x , t )$ </td><td></td><td>Learned score approximation.</td></tr><tr><td> $\nabla _ { x } \log p _ { t } ( x )$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\begin{array} { r l } { E _ { t } ( x ) } & { { } = } \end{array}$   $- \log p _ { t } ( x )$ </td><td>Time-dependent energy.</td><td> $d$ </td><td></td><td>Dimensionality of the data space.</td></tr><tr><td colspan="5">Geometric quantities</td></tr><tr><td> $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } , t )$ </td><td>Score divergence: reported unscaled mean outward</td><td> $\widehat { s } _ { \theta } ( x , t ) \ =$ </td><td></td><td>Unit-normalized learned score, where defined.</td></tr><tr><td> $\widehat { A } ( x , t )$ </td><td>learned-score flux. Normalized score divergence: mean outward align-</td><td> $s _ { \theta } ( x , t ) / \| s _ { \theta } ( x , t ) \| _ { 2 }$   $R _ { \mathrm { p } }$ </td><td></td><td>2 Physical radius of the score-probing sphere.</td></tr><tr><td></td><td>ment of unit-normalized boundary scores.</td><td></td><td></td><td></td></tr><tr><td> $e _ { j } \sim$ </td><td>Random outward unit probe direction.</td><td> $J$ </td><td></td><td>Number of boundary probes  $( J = 6 4 ) .$ </td></tr><tr><td> $\bar { \mathrm { U n i f } } ( \mathbb { S } ^ { d - 1 } )$   $V _ { \mathrm { b a s i n } }$ </td><td></td><td>R</td><td></td><td>Radius of the theoretical neighborhood  $B _ { R } .$ </td></tr><tr><td colspan="5">Effective basin/recovery volume.</td></tr><tr><td></td><td>Training and memorization</td><td></td><td></td><td></td></tr><tr><td>T Tmem</td><td>Training/optimization step. Onset of visible memorization.</td><td> $\tau _ { \mathrm { g e n } }$   $f _ { \mathrm { m e m } }$ </td><td></td><td>Onset of the generalization regime. Fraction of generated samples classified as memo-</td></tr><tr><td>n</td><td>Training-set size.</td><td>κ</td><td></td><td>rized. Threshold in the nearest-neighbor memorization cri-</td></tr><tr><td colspan="5"></td></tr><tr><td>Cyclic denoising</td><td></td><td> $x ^ { ( k ) }$ </td><td></td><td></td></tr><tr><td> $\gamma = t / T$   $\dot { U } _ { t } ( x ^ { \prime } \mid x )$ </td><td>Normalized cyclic-denoising noise amplitude. One-cycle transition kernel.</td><td> $k \mathrm { o r } \mathrm { } c$ </td><td></td><td>Sample after cycle k. Cyclic-denoising iteration index.</td></tr><tr><td colspan="5">Theoretical analysis</td></tr><tr><td> $x _ { i } , x _ { j }$ </td><td>Individual training samples.</td><td> $N$ </td><td></td><td>Number of samples in the empirical distribution.</td></tr><tr><td>q</td><td>Noise variance in  $\mathbf { \bar { \boldsymbol { U } } } = \mathbf { \dot { \boldsymbol { x } } } _ { i } + \mathbf { \nabla } \sqrt { q } \boldsymbol { Z } .$ </td><td> $Z$ </td><td>2</td><td>Gaussian noise in the theoretical analysis.</td></tr><tr><td> $U$ </td><td>Noisy observation of a training sample.</td><td> $\mathcal { N } ( 0 , I )$   $w _ { i } ( u )$ </td><td></td><td>Posterior probability associated with  $x _ { i } .$ </td></tr><tr><td> $\begin{array} { r l r l } { d _ { i j } } & { { } } & { = } & { { } } \end{array}$ </td><td>Distance between training samples  $x _ { i } , x _ { j } .$ </td><td> $B _ { R }$ </td><td></td><td>Radius-R neighborhood around  $x _ { i } .$ </td></tr><tr><td> $\| \dot { x _ { j } } - x _ { i } \|$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\bar { A _ { i } ( \boldsymbol { r } ) }$ </td><td>Upper-bound term for failure to recover</td><td> $h _ { i } ( x )$ </td><td></td><td>Probability that one cycle returns to  $x _ { i } .$ </td></tr><tr><td> $\dot { M }$ </td><td>Number of cycles in the persistence bound.</td><td>r</td><td></td><td>Distance/radius around training sample  $x _ { i } .$ </td></tr><tr><td colspan="5">Evaluation metrics</td></tr><tr><td>FID</td><td>Fréchet Inception Distance.</td><td></td><td> $D _ { \mathrm { K L } } ( P _ { \boldsymbol \theta } \| P _ { 0 } \mathbf { K } \mathbf { L }$ </td><td>divergence used for GMM evaluation.</td></tr><tr><td>SSCD</td><td>Copy-detection similarity.</td><td> $\Delta \hat { A }$ </td><td></td><td>Conditional-unconditional geometric-statistic differ- ence.</td></tr></table>

Geometric detection of memorization. Several methods detect memorization through the geometry of trained diffusion models. For conditional text-to-image models, Wen et al. (2024) use the norm of the difference between conditional and unconditional scores to identify memorized prompts, including at the first denoising step. Building on this measure, Jeon et al. (2025) relate the score difference to sharpness of the log-probability density and introduce a Hessian-based measure of sharpness at the start of generation. Asthana & Belagiannis (2026) show that the low-noise landscape becomes anisotropic, so a scalar score-difference norm can miss directional structure. They combine the norm with the angular alignment between the classifier-free guidance vector and the unconditional score. Brokman et al. (2025b;a) construct Monte Carlo estimators of the p-Laplacian from the learned score to identify memorized samples. These methods evaluate trained models; our analysis asks how geometric signatures develop during training.

Cyclic Denoising. Geometric measurements describe the local structure of the learned landscape, but not how trajectories move within or between its regions. Repeated partial noising and denois ing probes this dynamical structure directly. Kang et al. (2026) iterate forward–backward diffusion steps to form a U-Turn Markov chain (UTMC) and study its dynamics across cycles at fixed U-Turn noise levels. In their synthetic hierarchical model, low-noise UTMCs restrict exploration and cannot connect disconnected components of the data manifold, leading to non-ergodic dynamics. Larger noise levels enable transitions between components and restore ergodicity. At low noise, high-level features retain memory of the initial sample longer than low-level features; this ordering reverses at larger noise levels. They observe the same qualitative behavior in natural-language and image experiments. Sharma & Martiniani (2026) study repeated forward–reverse steps as cyclic denoising (CD), using it as an unconditional extraction attack and a probe of the learned landscape. At low corruption, trajectories can enter trivial fixed points or short periodic cycles. At larger corruption, they rearrange, cross between basins, and become trapped in long-lived structured attractors, many of which reproduce training images. They use high similarity between successive cycle outputs as an indication of basin residence and uncover images close to potential training examples. They demonstrate this behavior and the resulting extraction on Stable Diffusion v1.4 and on a denoising diffusion probabilistic model (DDPM) trained on CIFAR. Together, these works offer distinct views of CD: Kang et al. (2026) study landscape exploration and derive theoretical results on ergodicity and feature relaxation in a synthetic hierarchical model, while Sharma & Martiniani (2026) demonstrate its utility in training data extraction from fixed trained models. What remains missing is an account of how CD behavior changes as memorization develops during training, when it can extract memorized images, and the dynamics underlying its behavior. We investigate these questions across training and provide idealized theoretical analysis for CD.

## C LIMITATIONS AND FUTURE WORKS

Some of the major limitations of the work are discussed below. These could be addressed as part of future works.

• Our study of how memorization develops during training covers small datasets at $3 2 \times 3 2$ resolution with models trained from scratch. Stable Diffusion, although studied, is only evaluated as a single released checkpoint. Whether latent memorization forms on the same timescales, and remains detectable, in large conditional models trained on web-scale data is untested.

• Similarly, residence under cyclic denoising identifies basins but not their origin. Linking a basin to a training sample still requires the training set or a set of candidate images, and degenerate attractors show that residence alone can be misleading.

• The geometric probes carry their own constraints. In high dimensions, score divergence is estimated from finite boundary samples at a chosen probe scale and noise level, and the signal fades at larger noise. The estimates therefore depend on these choices.

• CD needs tens to hundreds of cycles per trajectory, which makes it far more expensive than single-shot detectors. Its benefit on Stable Diffusion is limited to strict false-positive rates on in-distribution negatives.

• The theory assumes the exact empirical score, the fully memorized limit. It explains why trajectories stay near a training point once they arrive, but not how they arrive from distant starts, and not how residence behaves under the learned score in the latent regime.

• Finally, the same probe that audits a model can extract training data from it, as our Stable Diffusion experiments show. Deploying cyclic denoising for auditing therefore requires the access controls discussed in Appendix D.

## D BROADER IMPACT

Understanding how memorization emerges in diffusion models has implications for both trustworthy generative modeling and responsible model deployment. Our results suggest that memorization can leave detectable geometric and dynamical signatures before it becomes obvious through ordinary generation, creating opportunities for earlier auditing of models for privacy leakage, training-data reproduction, and copyright-sensitive behavior. The proposed landscape-based diagnostics and cyclicdenoising probes may therefore help practitioners compare training configurations, identify models or checkpoints that are beginning to encode sample-specific memories, and motivate memorizationaware stopping or mitigation strategies. At the same time, because cyclic denoising can expose training-associated samples that standard generation may not readily reveal, the same techniques could potentially be used for model extraction or privacy attacks. We therefore view these methods primarily as diagnostic tools for understanding and auditing diffusion models, and encourage their use alongside appropriate access controls and responsible evaluation practices. The broader message is that model reliability should be assessed not only from generated outputs, but also from the hidden geometric structure and dynamical stability of memories encoded in the learned distribution.

## E SOME PRELIMINARIES

## E.1 DIFFUSION MODELS

Diffusion models (DMs) are generative models that learn to transform a simple noise distribution into a complex data distribution by reversing a fixed, progressive corruption process (Song et al., 2020a). Given a clean data sample $x _ { 0 } ~ \sim ~ p _ { \mathrm { d a t a } }$ and a predefined variance schedule $\beta _ { 1 } , \ldots , \beta _ { T }$ the forward process injects Gaussian noise over T discrete timesteps. Defining $\alpha _ { t } = 1 - \beta _ { t }$ and $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } \alpha _ { s } } \end{array}$ , the forward marginal is

$$
Q ( x _ { t } \mid x _ { 0 } ) = \mathcal { N } \big ( x _ { t } ; \sqrt { \bar { \alpha } _ { t } } x _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) I \big ) .\tag{2}
$$

This lets a noisy sample at any timestep t be written in closed form as

$$
\begin{array} { r } { x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \epsilon \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{3}
$$

To generate new samples, DMs learn a reverse process that gradually denoises a sample initialized from the prior $x _ { T } \sim \mathcal { N } ( 0 , I )$ . A network ϵ with parameters θ is trained to predict the injected noise via the simplified denoising objective (Ho et al., 2020):

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { x _ { 0 } , \epsilon , t } \left[ \Vert \epsilon - \epsilon _ { \theta } ( x _ { t } , t ) \Vert ^ { 2 } \right] .\tag{4}
$$

This objective is mathematically equivalent to denoising score matching (Song et al., 2020b): the score of the perturbed distribution, $s _ { t } ( x _ { t } ) = \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } )$ , is recovered directly from the model’s noise prediction,

$$
s _ { \theta } ( x _ { t } , t ) = - \frac { \epsilon _ { \theta } ( x _ { t } , t ) } { \sqrt { 1 - \bar { \alpha } _ { t } } } .\tag{5}
$$

Sampling from a diffusion model can therefore be understood as simulating a probability-flow or Langevin dynamics governed by the learned score field $s _ { \theta } ( x _ { t } , t )$

## E.2 γ TO σ<sub>t</sub>

![](images/b1316f6ad1397c0403190ab30ea77fce346b6feb72766bb53e70c98bae86b79b.jpg)  
Figure 5: Comparison of the noise standard deviation $( \sigma _ { t } ~ = ~ \sqrt { 1 - { \bar { \alpha } } _ { t } } )$ versus the normalized timestep $( \gamma = t / T )$ for linear and cosine noise schedulers. The cosine scheduler maintains lower noise magnitude for a significantly broader range of timesteps.

## F THEORETICAL INSIGHT

## F.1 BACKGROUND

Why should cyclic denoising reveal memorized samples? Our experiments suggest that, as memorization develops, individual training examples become increasingly stable under repeated noise–denoise cycles. This raises a natural theoretical question: when does a training sample exhibit prolonged residence under cyclic denoising, and how does this residence depend on the noise level and separation from other training examples? To isolate this mechanism, we analyze the exact empirical-score limit (Biroli et al., 2024), in which the clean distribution is the empirical training distribution and denoising performs exact stochastic reversal.

Exact empirical-score limit. Let $\mathcal { D } = \{ x _ { 1 } , . . . , x _ { N } \} \subset \mathbb { R } ^ { d }$ consist of distinct training points, with $N \geq { \bar { 2 } } .$ , and let

$$
p _ { 0 } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \delta _ { x _ { j } } .
$$

Fix a cycling noise variance $q > 0$ . In additive-noise coordinates, a cycle starting from $x$ first produces

$$
U = x + { \sqrt { q } } Z , \qquad Z \sim { \mathcal { N } } ( 0 , I ) .
$$

We assume that denoising uses the exact empirical score and samples from the corresponding exact reverse endpoint distribution. Bayes’ rule gives

$$
\mathbb { P } ( X ^ { \prime } = x _ { i } \mid U = u ) = w _ { i } ( u ) , \qquad w _ { i } ( u ) = \frac { \exp [ - \| u - x _ { i } \| ^ { 2 } / ( 2 q ) ] } { \sum _ { j = 1 } ^ { N } \exp [ - \| u - x _ { j } \| ^ { 2 } / ( 2 q ) ] } .
$$

Each cycle uses fresh independent forward noise and reverse-sampling randomness. Starting from any initial state, the first complete cycle produces a training point; subsequent clean endpoints form a time-homogeneous Markov chain on $\mathcal { D }$

Isolation controls recovery and residence. Recovery depends on the noisy observation’s relative compatibility with the target and all competing training points. The relevant quantities are the intersample distances $d _ { i j } = \| x _ { i } - x _ { j } \|$ , the cycling variance $q ,$ and the initial distance from the target. Starting sufficiently close to $x _ { i }$ , small noise relative to its separation from competitors makes recovery reliable. The theorem quantifies this through a sum over all competitors, accounting for both their distances and their number. Increasing the noise weakens the resulting recovery and residence bounds.

What the theorem establishes. For an initialization within a sufficiently small neighborhood of $x _ { i }$ , the theorem lower-bounds the probability of recovering $x _ { i }$ in one complete cycle. Conditional on recovery, subsequent cycles start at $x _ { i }$ itself, allowing a bound on repeated return to the same training point. For a fixed dataset and a neighborhood satisfying the theorem’s separation condition, sufficiently small noise yields high-probability residence over any prescribed finite horizon. Residence concerns clean endpoints; intermediate noisy states and reverse trajectories need not remain in the neighborhood. The theorem does not establish permanent trapping or discovery of the neighborhood from distant initializations. The empirical validation in this appendix evaluates the theorem’s quantitative prediction and lower bound using exact empirical posterior sampling on a GMM dataset, then examines approximate recovery and residence near training images using fully memorized and low-memorization checkpoints of a U-Net diffusion model trained on CelebA. The learned-model evaluation assesses the practical relevance of the mechanism using a reconstruction distance tolerance; the exact guarantee applies under the assumptions stated above.

Formally, define

$$
h _ { i } ( x ) = \mathbb { E } _ { Z } [ w _ { i } ( x + { \sqrt { q } } Z ) ] ,
$$

the probability that one complete cycle starting from x recovers $x _ { i }$ . We now bound this probability and the resulting finite-cycle residence.

## F.2 PROOF OF THEOREM 1

Forward process. The variance-preserving forward SDE

$$
d Y _ { t } = - Y _ { t } d t + \sqrt { 2 } d B _ { t } , \qquad Y _ { 0 } = x ,
$$

where $B _ { t }$ is standard Brownian motion, has transition law

$$
Y _ { t } \stackrel { d } { = } e ^ { - t } x + \sqrt { 1 - e ^ { - 2 t } } Z , \qquad Z \sim { \mathcal { N } } ( 0 , I ) .
$$

For $t > 0$ , setting $q = e ^ { 2 t } - 1$ and $U = e ^ { t } Y _ { t }$ gives

$$
\ U \triangleq x + { \sqrt { q } } Z .
$$

Thus the additive-noise formulation is a rescaling of this forward process. We fix $q > 0$ across cycles.

Assumption 1 . The clean law is $\begin{array} { r } { N ^ { - 1 } \sum _ { j = 1 } ^ { N } \delta _ { x _ { j } } } \end{array}$ , with distinct $x _ { 1 } , \ldots , x _ { N }$ and $N \geq 2$ . Denoising is its exact stochastic reverse diffusion to time zero, using the exact empirical score. Each cycle uses fresh independent forward noise and reverse-sampling randomness.

For $q > 0$ , define

$$
w _ { i } ( u ) = \frac { e ^ { - \| u - x _ { i } \| ^ { 2 } / ( 2 q ) } } { \sum _ { j } e ^ { - \| u - x _ { j } \| ^ { 2 } / ( 2 q ) } } , \qquad E _ { q } ( u ) = - q \log \sum _ { j } e ^ { - \| u - x _ { j } \| ^ { 2 } / ( 2 q ) } .\tag{6}
$$

Here $E _ { q }$ is the scaled negative log-density, up to an additive constant, and

$$
- \nabla E _ { q } ( u ) = \sum _ { j } w _ { j } ( u ) x _ { j } - u .
$$

Bayes’ rule gives $\mathbb { P } ( X ^ { \prime } = x _ { i } \mid U = u ) = w _ { i } ( u )$ . This endpoint law describes posterior sampling, rather than the posterior mean or an arbitrary finite-step reverse sampler.

Theorem 1 (restated). Fix $x _ { i }$ , write $d _ { i j } = \| x _ { j } - x _ { i } \|$ , and choose

$$
0 < 4 R < \operatorname* { m i n } _ { j \neq i } d _ { i j } .
$$

Let $B _ { R } = \{ x : \| x - x _ { i } \| \leq R \}$ . For $0 \leq r \leq R$ , define

$$
A _ { i } ( r ) = \operatorname* { m i n } \left\{ 1 , \frac { 1 } { 2 } \sum _ { j \neq i } \exp \left[ - \frac { d _ { i j } ( d _ { i j } - 4 r ) } { 8 q } \right] \right\} , \qquad h _ { i } ( x ) = \mathbb { E } _ { Z } [ w _ { i } ( x + \sqrt { q } Z ) ] .
$$

Then $h _ { i } ( x ) \geq 1 - A _ { i } ( r )$ whenever $\| x - x _ { i } \| \leq r .$ . Moreover, for $X _ { 0 } = x \in B _ { R }$ and every integer $M \geq 1$

$$
\mathbb { P } _ { x } \left( X _ { 1 } = \cdot \cdot \cdot = X _ { M } = x _ { i } \right) = h _ { i } ( x ) h _ { i } ( x _ { i } ) ^ { M - 1 } \geq \left[ 1 - A _ { i } ( R ) \right] \left[ 1 - A _ { i } ( 0 ) \right] ^ { M - 1 } .
$$

Proof. We first bound the probability of recovering a competing training point, then use the Markov property to obtain the residence bound.

(i) A two-point bound. Fix $j \neq i$ and write

$$
L _ { k } ( u ) = e ^ { - \| u - x _ { k } \| ^ { 2 } / ( 2 q ) } .
$$

The denominator of $w _ { j } ( u )$ contains the two distinct positive terms $L _ { i } ( u )$ and $L _ { j } ( u )$ , so

$$
w _ { j } ( u ) \leq \frac { L _ { j } ( u ) } { L _ { i } ( u ) + L _ { j } ( u ) } .
$$

Writing $\tau = L _ { j } ( u ) / L _ { i } ( u ) > 0$ and using $1 + \tau \geq 2 \sqrt { \tau }$ gives

$$
w _ { j } ( u ) \leq \frac { \tau } { 1 + \tau } \leq \frac { 1 } { 2 } \sqrt { \tau } = \frac { 1 } { 2 } \sqrt { \frac { L _ { j } ( u ) } { L _ { i } ( u ) } } .\tag{7}
$$

(ii) Recovery after Gaussian noising. Expanding the squared norms yields

$$
\| u - x _ { j } \| ^ { 2 } - \| u - x _ { i } \| ^ { 2 } = d _ { i j } ^ { 2 } - 2 \langle x _ { j } - x _ { i } , u - x _ { i } \rangle .
$$

Consequently, for $U = x + { \sqrt { q } } Z ,$

$$
\sqrt { \frac { L _ { j } ( U ) } { L _ { i } ( U ) } } = \exp \left[ - \frac { d _ { i j } ^ { 2 } } { 4 q } + \frac { \langle x _ { j } - x _ { i } , x - x _ { i } \rangle } { 2 q } + \frac { \langle x _ { j } - x _ { i } , Z \rangle } { 2 \sqrt { q } } \right] .
$$

The Gaussian identity $\mathbb { E } [ e ^ { \langle v , Z \rangle } ] = e ^ { \| v \| ^ { 2 } / 2 }$ therefore gives

$$
\mathbb { E } \sqrt { \frac { L _ { j } ( U ) } { L _ { i } ( U ) } } = \exp \left[ - \frac { d _ { i j } ^ { 2 } } { 8 q } + \frac { \langle x _ { j } - x _ { i } , x - x _ { i } \rangle } { 2 q } \right] .\tag{8}
$$

If $\| x - x _ { i } \| \leq r$ , Cauchy–Schwarz implies $\langle x _ { j } - x _ { i } , x - x _ { i } \rangle \leq d _ { i j } r$ , hence

$$
\mathbb { E } { \sqrt { \frac { L _ { j } ( U ) } { L _ { i } ( U ) } } } \leq \exp \left[ - { \frac { d _ { i j } ( d _ { i j } - 4 r ) } { 8 q } } \right] .
$$

Combining this with equation 7 and summing over $j \neq i$ gives

$$
1 - h _ { i } ( x ) = \mathbb { E } \left[ \sum _ { j \neq i } w _ { j } ( U ) \right] \leq \frac { 1 } { 2 } \sum _ { j \neq i } \exp \left[ - \frac { d _ { i j } ( d _ { i j } - 4 r ) } { 8 q } \right] .
$$

Since also $1 - h _ { i } ( x ) \leq 1$ , we conclude that

$$
1 - h _ { i } ( x ) \leq \operatorname* { m i n } \left\{ 1 , \frac { 1 } { 2 } \sum _ { j \neq i } \exp \left[ - \frac { d _ { i j } ( d _ { i j } - 4 r ) } { 8 q } \right] \right\} = A _ { i } ( r ) .
$$

This proves the one-cycle recovery bound.

(iii) Cyclic residence. Fresh independent randomness in each cycle gives the transition law

$$
\mathbb { P } ( X _ { k + 1 } = x _ { i } \mid X _ { k } = y ) = h _ { i } ( y ) .
$$

Although $X _ { 0 } ~ = ~ x$ may lie outside the training set, every $X _ { k }$ with $k \geq 1$ belongs to D. By the Markov property,

$$
\mathbb { P } _ { x } \left( X _ { 1 } = \cdot \cdot \cdot = X _ { M } = x _ { i } \right) = h _ { i } ( x ) \prod _ { k = 2 } ^ { M } h _ { i } ( x _ { i } ) = h _ { i } ( x ) h _ { i } ( x _ { i } ) ^ { M - 1 } ,
$$

where the product is empty when $M = 1$ . Applying part (ii) at radius R to the first factor and at radius 0 to the subsequent factors gives

$$
\mathbb { P } _ { x } \big ( X _ { 1 } = \cdot \cdot \cdot = X _ { M } = x _ { i } \big ) \geq \left[ 1 - A _ { i } ( R ) \right] \left[ 1 - A _ { i } ( 0 ) \right] ^ { M - 1 } .
$$

For $M = 1$ , the second factor is interpreted as 1.

## F.3 EXPERIMENTAL VERIFICATION

We check whether the quantitative prediction of Theorem 1 matches exact empirical cycling, and whether a learned model exhibits corresponding recovery and residence near training images. Defining, $\begin{array} { r } { \Delta _ { i } = \operatorname* { m i n } _ { j \neq i } \| x _ { j } - x _ { i } \| } \end{array}$ . Nearby initializations are displaced by $0 . 2 \Delta$ toward the target’s nearest training neighbor, satisfying the theorem’s condition $4 R _ { i } \ < \ \Delta _ { i }$ with $R _ { i } \ = \ 0 . 2 \Delta _ { i }$ . The experimental cycling level γ specifies a fraction of the forward schedule; its corresponding additive variance is $q = ( 1 - { \bar { \alpha } _ { t } } ) / { \bar { \alpha } _ { t } }$

Testing the quantitative prediction. We use an eight-dimensional bimodal GMM dataset containing 2048 training points and select 20 targets spanning nearest-neighbor separations. Each exact cycle adds Gaussian noise and samples a training point from the posterior weights $w _ { j } ( U )$ defined above. For each target and noise level, we estimate one-cycle return probabilities using 20,000 trials. Separately, for each target and noise level, we simulate 2048 independent trajectories from nearby initializations for up to 500 cycles, counting residence only when every clean endpoint equals the original target.

This allows an independent comparison between measured residence, the theorem’s product prediction $h _ { i } ( x ) h _ { i } ( x _ { i } ) ^ { \mathbf { \bar { M } - 1 } }$ estimated from one-cycle trials, and its lower bound $[ 1 - \bar { A } _ { i } ( R _ { i } ) ] [ 1 -$ $A _ { i } ( 0 ) ] ^ { M - 1 } . \mathrm { \ ' } \mathrm { A t \ ' } \gamma = 0 . 0 5$ , measured 500-cycle residence averages $6 6 . 4 8 \%$ , closely matching the product estimate of 66.01%, while the mean lower bound is 45.25%. $\mathrm { A t } \gamma = 0 . 1 0$ , these values are 9.72%, 9.57%, and 3.06%, respectively. All averages are taken over target-wise probabilities. Thus, the experiment supports the quantitative prediction and shows that the bound gives a nontrivial guarantee of residence in the tested low-noise regime.

Recovery and residence in a learned model. We next compare two checkpoints of the same U-Net(Adam Optimizer) trained on 512 CelebA images: an early checkpoint at $\tau = 2 4 0 0$ and a late checkpoint at $\tau \ : = \ : 1 3 0 0 0 0$ , with logged memorization fractions of 0.69% and $1 0 0 . 0 0 \%$ respectively. We report the comparison at $\gamma = 0 . 0 5$ , using the same 20 separation-ranked targets at both checkpoints and 16 trajectories per target from nearby initializations. Outputs are passed directly into subsequent cycles without projection onto the training set.

For the late checkpoint, mean relative reconstruction distance $\| X _ { k } - x _ { i } \| / \Delta _ { i }$ falls from 0.20 initially to 0.069 after one cycle and remains approximately 0.070 after 100 cycles. For the early checkpoint, it instead increases to 0.314 after one cycle and 1.468 after 100 cycles. Thus, on average over the tested initializations, the late checkpoint rapidly recovers proximity to the target and maintains a small reconstruction error, whereas the early checkpoint’s error grows. To distinguish accurate reconstruction from merely retaining the same nearest training image, we measure uninterrupted residence within a specified reconstruction tolerance. At the prespecified sensitivity tolerance of $0 . 1 0 \Delta _ { i } ,$ , 86.88% of late-checkpoint trajectories remain within the target neighborhood at every endpoint through 100 cycles, compared with no observed survivors for the early checkpoint. At the stricter primary tolerance of $0 . 0 5 \Delta _ { i }$ , late-checkpoint residence is 2.81%. The contrast between these tolerances also highlights the remaining reconstruction error where the late checkpoint’s mean distance from the target is approximately $0 . { \bar { 0 7 } } \Delta _ { i }$ after both one and 100 cycles.

Together, the experiments support the recovery and residence mechanism of Theorem 1. Exact empirical sampling corroborates its quantitative prediction. Under the same perturbations and cycling noise, the fully memorized checkpoint recovers proximity to training targets and sustains it, whereas the checkpoint with low memorization moves away. This contrast supports the interpretation of memorized training images as locally stable basins under cyclic denoising. The dependence on reconstruction tolerance reflects approximate recovery in the learned model, while the theorem provides a quantitative guarantee under exact empirical reversal.

## G ADDITIONAL RESULTS

## G.1 RESULTS ON THE GMM

The GMM has a known population score and low dimension $( d = 8 )$ , so every quantity in the paper can be computed on it cheaply and, for the landscape plots, exactly. We use it to check that the signatures observed on CelebA and CIFAR-10 also appear in this controlled setting. All models are residual MLPs trained with full-batch SGD (App. H).

Geometric signatures across training. Figure 6 repeats the analysis of Fig. 2 on the GMM. The held-out loss and $D _ { \mathrm { K L } } ( P _ { \boldsymbol { \theta } } \Vert P _ { 0 } )$ locate the overfitting transition (Fig. 6A). The normalized score divergence $\widehat { A } _ { 1 }$ is close to zero and identical for training, held-out, and generated samples in the bgeneralized regime, and the training and generated curves then fall well below the held-out curve (Fig. 6B). On the GMM, the lag between this separation and the rise in $f _ { \mathrm { m e m } }$ is smaller than on CelebA. As on CelebA, both occur later for larger n, and replotting against $\tau / n$ aligns the onsets (inset). Basin volume separates in the same direction: it grows by several orders of magnitude for training and generated samples and does not grow for held-out samples (Fig. 6C). The generated curves follow the training curves in both signatures.

![](images/dcb88b5e38550acdf878894acf6af8b47710372c74ff9edfa9b42b2d5a4676cd.jpg)

![](images/42d3e14d446ed4e5330a95100aacbe32894616159dd6d6515b6e87bcd3c16b04.jpg)

![](images/3c6c82d5bd8e549b80aa074ac6196b8e68d5301c85bf116467bc902552976c41.jpg)  
Figure 6: Geometric signatures across training on the GMM. (A) Train loss (solid) and test loss (dashed) at $\gamma = 0 . 0 1$ on the left axis, and $D _ { \mathrm { K L } } ( { \bar { P } } _ { \theta } \Vert P _ { 0 } )$ (dotted, log scale) on the right axis, versus training step τ, for $n \ = \ 2 0 4 8$ (B) Mean normalized score divergence $\widehat { A } _ { 1 }$ on the left axis and $f _ { \mathrm { m e m } }$ (dotted) on the right axis, versus $\tau .$ b. Line style denotes the sample group: train (solid), test (dashed), generated (dash-dot). Colour denotes the training-set size: $n = 1 0 2 4$ (orange), $n = 2 0 4 8$ (blue), $n \ : = \ : 4 0 9 6$ (green). Inset: the same divergence curves against $\tau / n$ . (C) Mean log basin volume versus τ for $n = 2 0 4 8$ , with the line styles of (B); bands show one standard deviation across samples.

Cyclic denoising from generated and training samples. Figure 7 applies CD to the $n = 5 1 2$ model at four checkpoints, from the generalized regime $( \tau = 4 2 , 1 0 7 ,$ , one-shot $f _ { \mathrm { m e m } } = 0 . 0 1 \% )$ to the memorized regime $( \tau = 8 6 4 , 0 4 4 , 8 5 . 4 7 \% )$ . We start 512 trajectories from fresh generations and 512 from the training points, and cycle them at eleven amplitudes between $\gamma = 0 . 0 0 1$ and $\gamma = 0 . 3 5$

Cycling exposes memorization that one-shot sampling misses. $\mathrm { A t } ~ \tau = 9 7 { , } 4 6 6$ , one-shot $f _ { \mathrm { m e m } }$ is 4.3%, and 100 cycles at $\gamma \in [ 0 . 0 2 , 0 . 0 6 ]$ raise it to about 20% (Fig. 7A). $\mathrm { A t } \tau = 1 6 1 , 2 6 8 .$ , cycling at $\gamma = 0 . 0 2$ raises it from 45% to 94%. At large amplitudes $( \gamma \ge 0 . 1 6 )$ $f _ { \mathrm { m e m } }$ stays near its one-shot value, so, as on CelebA, basins become accessible only when the perturbation is small enough. In the generalized regime no basin is reached at any amplitude, and even trajectories started exactly on training points leave them: their $f _ { \mathrm { m e m } }$ falls from 100% to 0% within ten cycles at every $\gamma \geq 0 . 0 4$ (Fig. 7C). Training points are therefore not stable states of this model. At later checkpoints, the trajectories started from training points first lose some memorized states and then return to the same level as those started from generated samples, so both kinds of starting points reach the same set of basins.

The cosine similarity to the starting sample shows how far trajectories travel (Fig. 7B, D). At every checkpoint it decays faster at larger γ, as expected from stronger perturbations. With training, trajectories at small $\gamma$ stay closer to their start: after 100 cycles at $\gamma = 0 . 0 2 .$ , the similarity is 0.75 at $\tau = 4 2 , 1 0 7$ and 1.00 at $\tau = 8 6 4 . 0 4 4$ . This matches the residence predicted by Theorem 1 for small cycling variance. $\mathrm { A t } \gamma = 0 . 3 5$ the similarity falls to about zero at every checkpoint, yet in the memorized regime $f _ { \mathrm { m e m } }$ remains near 85%. The trajectories therefore keep landing on training points but no longer on the one they started from. This is the large $^ - q$ limit of Theorem 1, in which each cycle ends at an essentially random training point.

Trajectories on the score-divergence landscape. Figure 8 follows a single trajectory on the landscape. Because the GMM score network is cheap to differentiate, we compute $\boldsymbol { A } _ { 1 }$ exactly as the trace of the Jacobian of the learned score at $t = 1$ , on a two-dimensional slice built as in App. G.6: a $2 0 0 \times 2 0 0$ grid $x ( \alpha , \beta ) = x ^ { \star } + \alpha { \bf d } _ { 1 } + \beta { \bf d } _ { 2 }$ . Here the origin $x ^ { \star }$ is the midpoint of the two GMM components, ${ \bf d } _ { 1 }$ is the direction $\mu / \lVert \boldsymbol { \mu } \rVert$ joining the two component means, and $\mathbf { d } _ { 2 }$ is the leading principal direction of the training data orthogonal to ${ \bf d } _ { 1 }$ . This fixes the plane so that it contains both components at every checkpoint and is the same across rows. Each cycle state is projected onto the plane. The landscape changes in both scale and shape during training. In the generalized regime, $\mathcal { A } _ { 1 }$ is smooth and small in magnitude, with a ridge between the two GMM components. After memorization sets in, its magnitude grows by two orders of magnitude and it breaks into many small regions of strongly negative divergence, the localized basins around individual training points. The trajectories follow the behaviour of Fig. 7. At small γ they stay near their start, and at $\tau = 8 6 4 { , } 0 4 4$ with $\gamma = 0 . 1 0$ the trajectory moves into one such region and stays there. At large γ they jump between distant regions of the plane.

![](images/0fbcb6a09dd9f4dc4a795f9a24905079ed7c834ff2ad6ebac6ea07e39a203eae.jpg)  
Figure 7: Cyclic denoising on the GMM across training. GMM, $n = 5 1 2$ . Columns are checkpoints of increasing $\tau ,$ with their one-shot $f _ { \mathrm { m e m } }$ . Each panel shows 512 trajectories per amplitude, coloured by $\gamma ( 0 . 0 \bar { 0 } 1 \leq \gamma \leq 0 . 3 5 )$ . (A) $f _ { \mathrm { m e m } }$ versus cycle for trajectories started from generated samples. (B) Mean cosine similarity between each state and its starting sample, computed on the raw 8-dimensional vectors, for the same trajectories. (C, D) The same quantities for trajectories started from the training points. Cycling raises $f _ { \mathrm { m e m } }$ above its one-shot value at small $\gamma$ once basins have formed, while at $\tau = 4 2 , 1 0 7$ even training points are not retained.

![](images/58a9c97ea0cc38c319a66f2c2501be24d2d2ffa21fb1a2891279cc63278d2580.jpg)  
Figure 8: Cyclic-denoising trajectories on the GMM score-divergence landscape. GMM, $n = 5 1 2$ . Rows are checkpoints of increasing τ, with their one-shot $f _ { \mathrm { m e m } } ;$ ; columns are cycling amplitudes $\gamma .$ The background is the exact score divergence $\boldsymbol { A } _ { 1 }$ of the learned score on a twodimensional plane spanned by the direction joining the two component means and the leading orthogonal principal direction of the training data; red marks negative (contracting) values. Each row has its own colour scale, since the range grows from about ±60 to ±6,000 during training. The magenta path joins the projected states of one trajectory every 5 cycles over 200 cycles, from the start (green dot) to the end (yellow triangle). Because states are projected onto the plane, the path shows direction of motion rather than exact position on the background.

## G.2 CYCLIC DENOISING

Here, we provide futher results of CD dynamics.

Degenerate attractors Figures 9–12 illustrate saturated colour blocks, stripe-like textures, nearly monochromatic fields, and loss of facial detail. We refer to these as degenerate attractor candidates. We identify non-memorized residence by thresholding high consecutive similarity, together with the absence of memorization at every logged cycle. Consistent with the population analysis in Fig. 3, residence alone does not establish memorization. On CIFAR-10 at $\gamma = 0 . 0 5$ , the fraction satisfying this screen decreases from 50.87% at $\tau _ { \mathrm { g e n } } = 4 9 , 5 0 0 \ t 0 \ 1 6 . 0 0 \%$ at the generalizationwindow checkpoint $\tau = 8 3 , 0 0 0$ , and to 0.98% at $\tau = 2 0 0 { , } 0 0 0$ . This corresponds to a decrease of 49.89 percentage points, or approximately a 98.1% relative reduction. At $\tau _ { \mathrm { g e n } } = 4 9 , 5 0 0$ , the fraction of CIFAR-10 trajectories passing the same residence and non-memorization screen falls from 50.87% a $\gamma = 0 . 0 5$ to 2.54% $\mathrm { a t } \gamma = 0 . 1 0$ , with none passing at $\gamma = 0 . 2 0$ or 0.30. Alongside increasing $f _ { \mathrm { m e m } }$ under cycling, this is consistent with residence shifting towards training-sample basins as training progresses. Qualitatively similar degeneration is visible on CelebA, where selected trajectories darken or become washed out, losing facial detail under repeated CD (Fig. 12).

Their appearance ranges from stripes and nearly monochromatic fields to configurations retaining recognizable objects. The CelebA examples similarly illustrate that residence can accompany substantial loss of facial detail without satisfying the memorization criterion; they do not establish a monotonic decline across training.

Throughout, $f _ { \mathrm { m e m } }$ measures memorization rather than residence. Where both are reported, we distinguish $f _ { \mathrm { m e m } }$ for one-shot generation from $f _ { \mathrm { m e m } }$ after 500 CD cycles. These are population measurements, separate from the displayed non-memorized trajectories. Together, the results distinguish residence, characterized by $S _ { \mathrm { c y c } } ^ { ( k ) }$ and $\mathcal { A } _ { 1 }$ , from identification with a training sample through $f _ { \mathrm { m e m } }$ . The declining share of degenerate attractors on CIFAR-10 accompanies increasing memorization under cycling, while the mechanism by which these basins arise and lose accessibility remains open. For clarity, we distinguish the logged ordinary-generation checkpoint estimate, $f _ { \mathrm { m e m } } ^ { \mathrm { g e n } } ,$ , from the logged full-generated-cohort estimate after 500 cycles, $f _ { \mathrm { m e m } } ^ { \mathrm { C D } } ( 5 0 0 )$ . The percentages below are the recorded bootstrap mean estimates; they are not prevalence estimates for the visually selected examples.

Saturated geometric configurations. $\mathrm { A t } ~ \tau = 4 9 , 5 0 0 , \mathrm { C I F A R  – 1 0 }$ trajectories evolve from recognizable images into saturated colour blocks and simplified geometric configurations (Figure 9). Here, $f _ { \mathrm { m e m } } ^ { \mathrm { g e n } } = 0 \%$ and $f _ { \mathrm { m e m } } ^ { \mathrm { C D } } ( 5 0 0 ) = 0 \%$ . Thus, substantial visual degeneration occurs without triggering the pixel-distance memorization criterion. These six examples were selected by their highest late similarity among qualifying trajectories.

![](images/a3b39f9bfe5af66b59acc856f648e2b0bd81d36095c1442b88976bb97408837d.jpg)  
Figure 9: Saturated geometric degeneration on CIFAR-10. $\tau = 4 9 , 5 0 0 , \gamma = 0 . 0 5 ; f _ { \mathrm { m e m } } ^ { \mathrm { g e n } } = 0 \%$ and full-cohort $f _ { \mathrm { m e m } } ^ { \mathrm { C D } } ( 5 0 0 ) = 0 \%$ . Repeated CD replaces recognizable content with saturated bands and colour blocks. The final column shows the nearest training image to c500.

Fading structure and nearly monochromatic endpoints. ${ \mathrm { A t } } \tau = 8 3 { , } 0 0 0 { \mathrm { . } }$ , the selected CIFAR-10 trajectories lose object detail and end in low-contrast blue fields (Figure 10). The checkpoint has $f _ { \mathrm { m e m } } ^ { \mathrm { \bar { g e n } } } = 0 \pm 0 \%$ , while the full generated cohort reaches $f _ { \mathrm { m e m } } ^ { \mathrm { C D } } ( 5 0 0 ) = 3 . 2 \%$ . The displayed trajectories remain outside the memorization criterion. With high late cycles mean similarity should not be interpreted as proof that the final appearance has stabilized.

Stripe-like textures and colour fields. Additional trajectories at $\tau = 4 9 { , } 5 0 0$ develop repeated stripes, colour bands, or nearly uniform fields (Figure 11). Both $f _ { \mathrm { m e m } } ^ { \mathrm { g e n } }$ and $f _ { \mathrm { m e m } } ^ { \mathrm { C D } } ( 5 0 0 )$ are $0 \%$ . The intermediate states reveal the emergence of these patterns and subsequent changes in colour and contrast.

![](images/b3188134c7df6cc268997278aaffff2bf339e979c80223587ba4ceeec9974d03.jpg)  
Figure 10: Fading structure on CIFAR-10. $\tau = 8 3 , 0 0 0 , \gamma = 0 . 0 5 ; f _ { \mathrm { m e m } } ^ { \mathrm { g e n } } = 0 \%$ and full-cohort $f _ { \mathrm { m e m } } ^ { \mathrm { S D } } ( 5 0 0 ) = 3 . 1 \overset { \smile } { 9 } 2 1 \%$ . Six appearance-selected trajectories lose semantic structure and approach nearly monochromatic blue configurations.

![](images/5fb5ae92c52413156850085f5bfacaf0ee1f4e12c1bb4723245a4823d2997f77.jpg)  
Figure 11: Stripe-like and monochromatic degeneration on CIFAR-10. $\tau = 4 9 , 5 0 0 , \gamma = 0 . 0 5 ;$ $f _ { \mathrm { m e m } } ^ { \mathrm { g e n } } = 0 \%$ and full-cohort $f _ { \mathrm { m e m } } ^ { \mathrm { C D } } ( 5 0 0 ) = 0 \%$ . Trajectories develop repeated spatial bands or nearly monochromatic fields.

CelebA across training checkpoints. Figure 12 combines two examples from each of three checkpoints at $\gamma = 0 . 0 5 . \mathrm { \ A t \ } \tau = 5 . 2 0 0$ (top pair), faces darken and lose detail; both memorization estimates are 0%. $\mathrm { A t } \tau = 1 0 \small { , } 0 0 0$ (middle pair), the selected faces become bright and washed out, with $f _ { \mathrm { m e m } } ^ { \mathrm { g e n } } = 0 \%$ and full-cohort $\dot { f } _ { \mathrm { m e m } } ^ { \mathrm { C D } } ( 5 0 \dot { 0 } ) = 9 . 8 \%$ $\mathrm { A t } \tau = 2 0 \small { , } 0 0 0$ (bottom pair), the selected outputs retain coherent facial structure, while the corresponding estimates are 0.1308% and 21.1202%. All six displayed trajectories remain non-memorized under the stated rule at the logged cycles.

![](images/5d61360f5b4fadd00810cf88a6f2761f22cb8ec30bda9b982c612544ba1e3890.jpg)  
Figure 12: CelebA CD trajectories across training checkpoints, $\gamma = 0 . 0 5 .$ . Top, middle and bottom pairs correspond to $\tau = 5 { , } 2 0 0$ , 10,000 and 20,000, respectively. Their logged $f _ { \mathrm { m e m } } ^ { \mathrm { g e n } }$ values are 0%, 0% and 0.1308%; full-cohort $f _ { \mathrm { m e m } } ^ { \mathrm { C D } }$ (500) values are 0%, 9.8% and 21.12%. $\mathrm { A t } \tau = 5 2 0 0$ the images move towards mere monochromatic darkness and remain stable there. For $\tau = 1 0 0 0 0$ and $\tau = 2 0 0 0 0$ more facial structure emerges for these non memorized basins.

![](images/50816599ff6dbf4722b9bcf76e991c93899b13c8f304a7e27369bb11d42abaff.jpg)  
Figure 13: Cyclic denoising from training samples. Memorization fraction $f _ { \mathrm { m e m } }$ across 500 cycles for trajectories initialized from training images in (A) CelebA $( n = 4 0 9 6$ , Adam, linear noise schedule) and (B) CIFAR-10 $( n = 8 1 9 2 $ , Adam, cosine noise schedule). Although $f _ { \mathrm { m e m } }$ starts at 100%, it drops sharply under cycling. At $\tau _ { \mathrm { g e n } } ,$ it remains near zero, whereas later checkpoints retain or recover a larger fraction, depending on γ. Classification is against the full training set and does not require returning to the initial training image. Shaded bands combine the reported bootstrap standard errors with variation within each cycle bin. The memory of the initial train samples can be lost under cycling, basin residence only remains in a subset of train samples that become attractors(subject to cycling amplitude). Cycling moves trajectories to other stronger attractor basins.

<sub>A</sub> =20,000 | $\scriptstyle f _ { \mathrm { m e m } } = 0 . 1 \% \ 1$ FID=48.2  
![](images/809cba8687505a10d030e6d4a8f8897e0807eeb717b792b4b7617a4888a8bc35.jpg)  
<sub>D</sub> =83,000 | $\scriptstyle f _ { \mathrm { m e m } } = 0 . 0 \% \mid$ FID=98.3

![](images/783e34231b82a91b8d13fdb05d28b0027c76b1a77090fd7398c2c157dcfbfa9f.jpg)

C  
![](images/5f4c64395cfe93b523a2cfb6fa006dd707f2d69860671ae0c511c9a5c75f19fc.jpg)

![](images/e8040db5d23e605102e694791dc4f7c8f19ad3f311536e7dbac12ba618829437.jpg)

![](images/90ced844af51f3d990af57cf00f8c626ea5ee7f643babaed52155473856a8a8d.jpg)  
Figure 14: Trajectories discover and reside in basins corresponding to training samples. (A) CelebA trajectories $( n = 4 0 9 6$ , Adam, $\gamma = 0 . 1 0 )$ from $c _ { 0 }$ to $c _ { 5 0 0 }$ , with the nearest training image to each final state. The marked onset is the first cycle at which a trajectory satisfies the $f _ { \mathrm { m e m } }$ criterion and remains classified as memorized for the rest of the run. (B) Score divergence $\mathcal { A } _ { 1 }$ decreases as the trajectories approach these states; the decline can begin before the marked onset. (C) Consecutive-state LPIPS cosine similarity $S _ { \mathrm { c y c } } ^ { ( k ) }$ increases and reaches a high plateau, consistent with residence after the approach. (D–F) The corresponding CIFAR-10 trajectories $( n = 8 1 9 2$ Adam, $\gamma = 0 . 0 5 )$ and measurements. Colors identify trajectories across panels. Dotted lines and markers show their onsets. Divergence and similarity are displayed with a centered 50-cycle moving average. Annotations above A and D give the training step $\tau ,$ one-shot $f _ { \mathrm { m e m } }$ , and FID.

800  
400  
Cycle 0  
600  
1000  
![](images/fdf4eff36a69b78ff621472871f79a527a331a717bb80df63e6c4600168c1656.jpg)  
Figure 15: Basin residence and jumping under repeated cycling at $\tau = 4 0 , 0 0 0$ (run: CelebA32, UNet, $n = 4 0 9 6$ , Adam). Both sets begin in the basin associated with the same memorised training example. $\mathrm { A t } \gamma = 0 . 1$ (top), trajectories remain in this basin. $\mathrm { A t } \gamma = 0 . 2$ (bottom), stronger perturbations induce basin jumping, followed by sustained residence in another attractor, consistent with greater stability under cycling.

## G.3 METRICS

## G.3.1 MEMORIZATION FRACTION

Following Yoon et al. (2023); Gu et al. (2023); Bonnaire et al. (2025), we classify a sample as memorized using its relative proximity to its two nearest training neighbours. For a sample $x ,$ let $x _ { ( 1 ) }$ and $x _ { ( 2 ) }$ denote its nearest and second-nearest training neighbours under Euclidean distance. Define

$$
r ( x ) = { \frac { \| x - x _ { ( 1 ) } \| _ { 2 } } { \| x - x _ { ( 2 ) } \| _ { 2 } } } , \qquad m ( x ) = 1 [ r ( x ) < \kappa ] , \qquad \kappa = { \frac { 1 } { 3 } } .\tag{9}
$$

For an evaluated collection of $M$ samples, the memorization fraction is the percentage satisfying this criterion:

$$
f _ { \mathrm { m e m } } = \frac { 1 0 0 } { M } \sum _ { i = 1 } ^ { M } m ( x _ { i } ) .\tag{10}
$$

A small ratio indicates that a sample is substantially closer to one training example than to its nearest competitor. The criterion does not require exact equality with a training example.

For one-shot generation, we evaluate generated samples. During cyclic denoising, we evaluate the trajectory states at each cycle. For trajectories initialized from training samples, memorization is measured against the full training set, not only the starting image. An increase in $f _ { \mathrm { m e m } }$ therefore indicates that more trajectories reach states satisfying the training-sample memorization criterion; it does not measure the number of distinct training samples reached.

We estimate uncertainty using 1,000 bootstrap resamples of the evaluated collection, each containing M samples drawn with replacement. The standard deviation of the bootstrap fractions provides one standard error on $f _ { \mathrm { m e m } }$

## G.3.2 CYCLIC SIMILARITY

We measure perceptual similarity between consecutive saved trajectory states using the VGG backbone of LPIPS. Images undergo the LPIPS input preprocessing before feature extraction. Let $F _ { \ell } ( x )$ denote the feature map at layer $\ell .$ We normalize its feature vectors across channels at each spatial location, then flatten and concatenate the normalized maps:

$$
\widetilde { F } _ { \ell , h , w } ( x ) = \frac { F _ { \ell , h , w } ( x ) } { \| F _ { \ell , h , w } ( x ) \| _ { 2 } + \epsilon } , \qquad \Phi ( x ) = \mathrm { c o n c a t } _ { \ell } \left[ \mathrm { v e c } \big ( \widetilde { F } _ { \ell } ( x ) \big ) \right] ,\tag{11}
$$

where $\epsilon > 0$ ensures numerical stability. For trajectory $i ,$ let $x _ { i } ^ { ( k ) }$ denote its state at the kth saved observation. We define

$$
S _ { \mathrm { c y c } , i } ^ { ( k ) } = \frac { \left. \Phi ( x _ { i } ^ { ( k ) } ) , \Phi ( x _ { i } ^ { ( k - 1 ) } ) \right. } { \| \Phi ( x _ { i } ^ { ( k ) } ) \| _ { 2 } \| \Phi ( x _ { i } ^ { ( k - 1 ) } ) \| _ { 2 } } , \qquad S _ { \mathrm { c y c } } ^ { ( k ) } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } S _ { \mathrm { c y c } , i } ^ { ( k ) } .\tag{12}
$$

This statistic is cosine similarity in the normalized LPIPS-VGG feature space. Values close to one indicate small perceptual changes between saved states. An increase followed by a sustained high plateau is consistent with trajectories settling into basins that remain stable under the cycling amplitude. This alone does not establish residence in a training-sample basin, since degenerate attractors can also produce high similarity. We therefore interpret $S _ { \mathrm { c y c } } ^ { ( \bar { k } ) }$ together with $f _ { \mathrm { m e m } }$ and score divergence. For visualizations, we apply a centered moving average.

## G.3.3 BASIN VOLUME

We estimate the basin volume of a target sample $\hat { x } \in \mathbb { R } ^ { d }$ using a deterministic DDIM recovery sweep adapted from Pham et al. (2025). The sweep identifies the largest tested noise amplitude at which forward-perturbed copies of xˆ recover with sufficiently high probability.

Let $Q _ { \gamma } ( \hat { x } ; \xi )$ denote the noise scheduler’s forward perturbation of $\hat { x }$ at amplitude $\gamma ,$ using an independent draw $\xi \sim \mathcal { N } ( 0 , I _ { d } )$ . Let $\mathcal { D } _ { \theta } ^ { \mathrm { D D I M } } ( x _ { \gamma } ; \gamma )$ denote deterministic DDIM denoising of $x _ { \gamma }$ from that amplitude. We test $\gamma \in \{ 0 . 0 5 , 0 . 1 0 , \ldots , 1 . 0 0 \}$ in increasing order. At each amplitude, we independently perturb $M = 1 0$ copies of xˆ, drawing fresh noise for every copy and every amplitude.

```tcl
Algorithm 1: Recovery-based basin-volume estimate
Input: Target xˆ; forward noising operation $Q ;$ DDIM denoiser $\overline { { \mathcal { D } _ { \theta } ^ { \mathrm { D D I M } } } }$ ; recovery distance
$D _ { \mathrm { r e c } } ;$ tolerance $\epsilon _ { \mathrm { r e c } }$
Output: $\gamma _ { c } , r _ { \mathrm { r e c } } ( \hat { x } ) , V _ { \mathrm { b a s i n } } ( \hat { x } )$
1 $M \gets 1 0 ;$
2 $p _ { \mathrm { s t o p } } \gets 0 . 9 ;$
3 $\gamma _ { c } \gets 0 ;$
4 for $\gamma \in \{ 0 . 0 5 , 0 . 1 0 , \ldots , 1 . 0 0 \}$ do
5 for $i \gets 1$ to M do
6 Sample $\xi _ { i } \sim \mathcal { N } ( 0 , I _ { d } ) ;$
7 $x _ { i } \gets Q _ { \gamma } ( \hat { x } ; \xi _ { i } ) ;$
8 $\begin{array} { r } { \tilde { x } _ { i }  \mathcal { D } _ { \theta } ^ { \mathrm { D D I M } } ( x _ { i } ; \gamma ) ; } \end{array}$
9 $R _ { i }  \mathbf { 1 } \check { [ } D _ { \mathrm { r e c } } ( \tilde { x } _ { i } , \hat { x } ) \leq \epsilon _ { \mathrm { r e c } } ] ;$
10 $\begin{array} { r } { p _ { \mathrm { r e c } }  \frac { 1 } { M } \sum _ { i = 1 } ^ { M } R _ { i } ; } \end{array}$
11 if $p _ { \mathrm { r e c } } < p _ { \mathrm { s t o p } }$ then
12 break;
13 $\gamma _ { c } \gets \gamma ;$
14 if $\gamma _ { c } = 0$ then
15 $x _ { \gamma _ { c } } \gets \hat { x } ;$
16 else
17 Sample fresh $\xi ^ { \star } \sim \mathcal { N } ( 0 , I _ { d } ) ;$
18 $\_ x _ { \gamma _ { c } } \gets Q _ { \gamma _ { c } } ( \hat { x } ; \xi ^ { \star } ) ;$
19 $r _ { \mathrm { r e c } } ( \hat { x } ) \gets \| \hat { x } - x _ { \gamma _ { c } } \| _ { 2 } ;$
20 $\begin{array} { r } { V _ { \mathrm { b a s i n } } ( \hat { x } ) \gets \frac { \pi ^ { d / 2 } } { \Gamma ( d / 2 + 1 ) } \big [ r _ { \mathrm { r e c } } ( \hat { x } ) \big ] ^ { d } ; } \end{array}$
21 return $\gamma _ { c } , r _ { \mathrm { r e c } } ( \hat { x } ) , V _ { \mathrm { b a s i n } } ( \hat { x } ) ;$
```

For copy i, write

$$
\begin{array} { r } { x _ { \gamma } ^ { ( i ) } = Q _ { \gamma } ( \hat { x } ; \xi _ { \gamma } ^ { ( i ) } ) , \qquad \tilde { x } _ { \gamma } ^ { ( i ) } = \mathcal { D } _ { \theta } ^ { \mathrm { D D I M } } \big ( x _ { \gamma } ^ { ( i ) } ; \gamma \big ) . } \end{array}\tag{13}
$$

Its recovery indicator and the empirical recovery probability are

$$
R _ { \gamma } ^ { ( i ) } = \mathbf { 1 } \Big [ D _ { \mathrm { r e c } } \big ( \tilde { x } _ { \gamma } ^ { ( i ) } , \hat { x } \big ) \leq \epsilon _ { \mathrm { r e c } } \Big ] , \qquad p _ { \mathrm { r e c } } ( \gamma ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } R _ { \gamma } ^ { ( i ) } .\tag{14}
$$

For CelebA and CIFAR, $D _ { \mathrm { r e c } }$ is LPIPS with a VGG backbone and $\epsilon _ { \mathrm { r e c } } = 0 . 0 5$ . Before evaluating LPIPS, we invert the dataset normalization, clip pixel values to $[ 0 , 1 ]$ , and map them to $[ - 1 , 1 ]$ The grayscale CelebA images, centre-cropped to ${ \bar { 3 } } 2 \times 3 2$ , are replicated across three channels for LPIPS; CIFAR images are already RGB. For the GMM experiments, $D _ { \mathrm { r e c } }$ is Euclidean distance and $\epsilon _ { \mathrm { r e c } } = 0 . 4$

The sweep stops at the first amplitude for which $p _ { \mathrm { r e c } } ( \gamma ) < 0 . 9$ . We define the critical noise ampli tude $\gamma _ { c }$ as the last preceding tested amplitude for which $p _ { \mathrm { r e c } } ( \gamma _ { c } ) \geq 0 . 9$ . Thus, at least nine of the ten copies must recover for an amplitude to pass. We set $\gamma _ { c } = 0$ if the first tested amplitude fails, and $\gamma _ { c } = 1$ if every tested amplitude passes.

Once $\gamma _ { c }$ is determined, we draw fresh noise $\xi ^ { \star } \sim \mathcal { N } ( 0 , I _ { d } )$ and form a new perturbation $x _ { \gamma _ { c } } =$ $Q _ { \gamma _ { c } } ( \hat { x } ; \xi ^ { \star } )$ , with $x _ { 0 } = { \hat { x } }$ . The ten copies in the recovery sweep determine $\gamma _ { c } ;$ they are not used to calculate the radius. We define

$$
r _ { \mathrm { r e c } } ( \hat { x } ) = \bigl \| \hat { x } - x _ { \gamma _ { c } } \bigr \| _ { 2 } , \qquad V _ { \mathrm { b a s i n } } ( \hat { x } ) = \frac { \pi ^ { d / 2 } } { \Gamma ( d / 2 + 1 ) } \bigl [ r _ { \mathrm { r e c } } ( \hat { x } ) \bigr ] ^ { d } .\tag{15}
$$

Here, $\Gamma ( \cdot )$ is the Euler gamma function. The fresh perturbation $x _ { \gamma _ { c } }$ is used to measure displacement and is not separately tested for recovery. $V _ { \mathrm { b a s i n } }$ is an isotropic proxy of basin volume based on this estimated recovery amplitude.

## G.3.4 SCORE DIVERGENCE AND ITS ESTIMATION

The pointwise divergence of the learned score field, where defined, is $\nabla _ { x } \cdot s _ { \theta } ( x , t )$ . We obtain the learned score from the model’s noise prediction $\epsilon _ { \theta }$ as

$$
s _ { \theta } ( x , t ) = - \frac { \epsilon _ { \theta } ( x , t ) } { \sqrt { 1 - \bar { \alpha } _ { t } } } ,\tag{16}
$$

where $\bar { \alpha } _ { t }$ is the cumulative signal-retention coefficient of the forward noise schedule. All quantities below are evaluated in the data coordinates used by the diffusion model.

Evaluation point and boundary radius. For each original sample $x _ { 0 }$ , whether training, held-out, or generated, we draw one forward perturbation

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I _ { d } ) .\tag{17}
$$

The probing sphere is centred at $x _ { t }$ . Unless stated otherwise, we use $t = 1$ , the final denoising timestep. We calibrate the physical probing radius from all $n _ { \mathrm { t r a i n } }$ training samples:

$$
R _ { \mathrm { p } } = 0 . 0 7 5 \ \mathrm { m e d i a n } _ { 1 \leq i \leq n _ { \mathrm { t r a i n } } } \left( \operatorname* { m i n } _ { 1 \leq k \leq n _ { \mathrm { t r a i n } } } \| x _ { i } - x _ { k } \| _ { 2 } \right) .\tag{18}
$$

The nearest-neighbour distances are computed between unperturbed training samples in the model’s data coordinates. Thus the sphere probes the score field at 7.5% of the median training nearestneighbour distance. The implementation passes a radius parameter equal to $R _ { \mathrm { p } } / \sqrt { d }$ and multiplies it by $\sqrt { d }$ when forming boundary points; their physical distance from $x _ { t }$ is $R _ { \mathrm { p } }$ . The median nearestneighbour distance provides a data-dependent measure of typical training-sample separation in the model’s data coordinates. Setting $R _ { \mathrm { p } }$ to 7.5% of this distance probes the score field at a smaller spatial scale than typical sample separation. A finite radius also retains a measurable outward-flux signal: under local smoothness, the mean flux is $( R _ { \mathrm { p } } / d ) \nabla \cdot s _ { \theta } ( x _ { t } , t ) + o ( R _ { \mathrm { p } } )$ as $R _ { \mathrm { p } }  0$ . We use the same calibrated radius for training, held-out, and generated samples throughout an experiment, so their fluxes are compared at a common spatial scale.

Reported score divergence. Following the boundary method of Brokman et al. (2025b), we draw $J = 6 4$ independent Gaussian vectors $g _ { j } \sim \mathcal { N } ( 0 , I _ { d } )$ and construct

$$
e _ { j } = { \frac { g _ { j } } { \| g _ { j } \| _ { 2 } } } , \qquad y _ { j } = x _ { t } + R _ { \mathrm { p } } e _ { j } .\tag{19}
$$

Each $e _ { j }$ is uniformly distributed on the unit sphere and is the outward unit normal at $y _ { j }$ . We report the unscaled mean outward learned-score flux

$$
A ( x _ { 0 } , t ) = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } s _ { \theta } ( y _ { j } , t ) \cdot e _ { j } .\tag{20}
$$

We call this reported statistic score divergence for convenience and write $\mathcal { A } _ { 1 } ( x _ { 0 } ) = \mathcal { A } ( x _ { 0 } , 1 )$ . The argument $x _ { 0 }$ identifies the original sample; the score evaluations are made on a sphere around its forward-perturbed version $x _ { t }$

The relationship to mathematical divergence follows when the score field satisfies the conditions for the divergence theorem. Let $B _ { R _ { \mathrm { D } } } ( x _ { t } )$ be the ball of radius $R _ { \mathrm { p } }$ centred at $x _ { t }$ . The sphere-area-toball-volume ratio is $| \partial B _ { R _ { \mathrm { p } } } | / | B _ { R _ { \mathrm { p } } } ^ { \star } | = d / R _ { \mathrm { p } }$ , giving

$$
\begin{array} { r l r } {  { \frac { 1 } { | B _ { R _ { \mathrm { p } } } | } \int _ { B _ { R _ { \mathrm { p } } } ( x _ { t } ) } \nabla _ { z } \cdot s _ { \theta } ( z , t ) \mathrm { d } z } } \\ & { } & { = \frac { d } { R _ { \mathrm { p } } } \mathbb { E } _ { e \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } ) } [ s _ { \theta } ( x _ { t } + R _ { \mathrm { p } } e , t ) \cdot e ] . } \end{array}\tag{21}
$$

Conditional on $x _ { t } ,$ Eq. equation 20 is a Monte Carlo estimate of the expectation on the right. Multiplying it by $d / R _ { \mathrm { p } }$ would give a Monte Carlo estimate of the divergence averaged over the ball. We report the unscaled value. Because this factor is positive and fixed within an experiment, omitting it preserves the signs and ordering of samples compared within that experiment. Numerical magnitudes from experiments with different dimensions or radii do not have the same scaling.

Algorithm 2: Monte Carlo computation of the reported score statistics   
Input: Sample $x _ { 0 } \in \mathbb { R } ^ { d } ;$ noise predictor $\epsilon _ { \theta } ;$ schedule coefficient ${ { \bar { \alpha } } _ { t } } ;$ calibrated radius $R _ { \mathrm { p } } { \mathrm { : } }$   
timestep t   
Output: $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } _ { 0 } , t )$ and $\widehat { A } ( x _ { 0 } , t )$   
1 $J \gets 6 4 ;$   
2 Sample $\epsilon \sim \mathcal { N } ( 0 , I _ { d } ) ;$   
3 $x _ { t } \gets \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon ;$   
4 $F \gets 0 ; C \gets 0 ;$   
5 for $j  1$ to J do   
6 Sample $g _ { \mathscr { L } } \sim \mathcal { N } ( 0 , I _ { d } ) ;$   
7 $e _ { j }  g _ { j } / \| g _ { \underline { { j } } } \| _ { 2 } ;$   
8 $y _ { j } \gets x _ { t } + R _ { \mathrm { p } } e _ { j } ;$   
9 $s _ { j } \gets - \epsilon _ { \theta } ( y _ { j } , t ) / \sqrt { 1 - \bar { \alpha } _ { t } } ;$   
10 $\bar { F }  F + s _ { j } \cdot e _ { j } ;$   
11 $C \gets C + ( \overline { { s } } _ { j } \cdot \overline { { e } } _ { j } ) / \| s _ { j } \| _ { 2 } ;$   
12 $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } _ { 0 } , t ) \gets \boldsymbol { F } / J ;$   
13 $\widehat { A } ( x _ { 0 } , t ) \gets C / J ;$   
b14 return $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } _ { 0 } , t ) , \widehat { \boldsymbol { A } } ( \boldsymbol { x } _ { 0 } , t ) ;$

Normalized score divergence. Where $s _ { \theta } ( y _ { j } , t ) \neq 0$ , the unit-normalized learned score is

$$
\widehat { s } _ { \theta } ( y _ { j } , t ) = \frac { s _ { \theta } ( y _ { j } , t ) } { \vert \vert s _ { \theta } ( y _ { j } , t ) \vert \vert _ { 2 } } .\tag{22}
$$

We report its mean outward alignment,

$$
\widehat { A } ( x _ { 0 } , t ) = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \widehat { s } _ { \theta } ( y _ { j } , t ) \cdot e _ { j } .\tag{23}
$$

We call this statistic normalized score divergence for convenience and write $\widehat { A } _ { 1 } ( x _ { 0 } ) = \widehat { A } ( x _ { 0 } , 1 )$ Each summand lies in $[ - 1 , 1 ]$ b bwhere defined. Negative values indicate that the boundary scores point predominantly toward the centre of the sphere.

For $n _ { \mathrm { t r a i n } } = 2 0 4 8$ , the physical probing radii are $R _ { \mathrm { p } } = 0 . 1 0 2 6 2 9$ for GMM, 0.406240 for CelebA, and 0.769862 for CIFAR-10.

## G.4 MEMBERSHIP INFERENCE FROM SCORE DIVERGENCE

In Section 3.2 we saw that, as a model memorizes, the normalized score divergence $\widehat { A } _ { 1 }$ around trainbing samples becomes markedly more negative than around held-out test samples, while generated samples behave almost exactly like training samples. This difference is large enough to decide, for a new point, whether it belonged to the training set. Here we describe how we turn this geometric signal into a simple membership inference test and report its performance across training.

The membership score. For each candidate point x we use the normalized score divergence $\widehat { A } _ { 1 } ( x )$ , evaluated at a single small noise amplitude at the final denoising step. Recall that $\widehat { A } _ { 1 }$ b bthe mean outward alignment of the unit-normalized score over a small sphere around x (a mean of cosines, bounded in $[ - 1 , 1 ] ) ;$ it is strongly negative where the score field points inward, i.e. in the sharp basins that form around memorized training samples. We use the same boundary estimator for both datasets: we draw directions on a small sphere around x, evaluate the learned score at those boundary points, normalize each score to unit length, and average its alignment with the outward normal. For the GMM the score is the exact learned score at $d = 8 ;$ for CelebA it is the learned UNet score at $d = 1 0 2 4$ . We flip the sign so that a higher score means “more likely a member” and use this value directly as the membership score.

No training labels required. The key property we exploit is that generated samples are statisti cally indistinguishable from training samples under this score (Figure 16). Generated samples can therefore stand in for members whenever a reference distribution is needed, so the procedure never requires access to the training set or its labels.

How we evaluate. We score an equal number of training samples (true members) and held-out test samples (true non-members). We report the area under the ROC curve (AUC), which is thresholdfree and measures how well the score separates the two groups on its own. We additionally report the true positive rate (TPR) and false positive rate (FPR) at a moderate 5% false-positive budget, and the TPR at a strict 1% false-positive budget (TPR@1%FPR); all operating points are read directly off the ROC curve, so no threshold needs to be chosen.

Results. Figure 16 shows the distribution of the membership score for training, test, and generated samples at increasing training steps, for both GMM (top) and CelebA (bottom). Early in training the three distributions overlap almost completely and no rule can separate members from non-members. As training enters the memorized regime, the training and generated distributions shift together toward higher scores while the test distribution stays in place, opening a clear margin. Tables 2 and 3 quantify this: the AUC rises from close to chance in the generalized regime to 1.0 in the memorized regime, and the TPR at 1% FPR follows the same trend. The attack becomes effective at the same training step at which the geometric signatures of Section 3.2 first separate—well before $f _ { \mathrm { m e m } }$ reflects memorization in the generated output.

![](images/26edf6fa0dd311c6cfdd9c293df1bb299e60953b8c883b60ea6c7418f0b5da1b.jpg)

![](images/7043f61050faf8e65b60df35e4388290f971488999768243fe37edb389c6f72e.jpg)

![](images/cd36a2a717f9315c5989196557cb91a400800f3ec114a7569daf226a6468f33b.jpg)

![](images/75d9a926cc021a38edc993cc881ef9db918ede059d00705829fdd994ea826f08.jpg)

![](images/f97bb9ca3f3c45dd747119c8c4710ccb54a0c89a86f40cd85457d4530567578a.jpg)

![](images/bdb3f925402ff7bd3d0d78afe6c3872e650087b6392fe69a11758f372789f7fb.jpg)

![](images/2f4ddcf717acef452463ce8224ca107f734e7c8794203dbdfe954a350a0c48cc.jpg)

![](images/6ffa08a69a2d01e9be00b637706924ceab33b777f6f29bb9d16ab8bde759c23b.jpg)  
Figure 16: Distribution of the membership score (the sign-flipped normalized score divergence $\widehat { A } _ { 1 } )$ bfor training (blue), held-out test (red), and generated (green) samples, at increasing training steps τ (left to right), for GMM (top row) and CelebA (bottom row). The three distributions overlap in the generalized regime and separate—with training and generated moving together, away from test—as the model memorizes.

Table 2: Membership inference from the normalized score divergence $\widehat { A } _ { 1 }$ at a single noise level, on bCelebA, without any access to training labels. AUC is threshold-free. TPR and FPR are reported at a moderate 5% false-positive budget; TPR@1%FPR is the true-positive rate at a strict $1 \%$ budget. All operating points are read directly off the ROC curve.
<table><tr><td>T</td><td>AUC</td><td>TPR</td><td>FPR</td><td>TPR@1%FPR</td></tr><tr><td>60,000</td><td>0.522</td><td>0.875</td><td>0.835</td><td>0.035</td></tr><tr><td>115,000</td><td>0.887</td><td>0.910</td><td>0.325</td><td>0.315</td></tr><tr><td>200,000</td><td>1.000</td><td>0.925</td><td>0.000</td><td>0.995</td></tr><tr><td>430,000</td><td>1.000</td><td>0.980</td><td>0.000</td><td>1.000</td></tr></table>

![](images/d3deb4681334134a80165a422424a34d2873d55b4949aba44cbe673e35634035.jpg)  
Figure 17: CelebA width-64 U-Net trained with Adam, number of training samples $\cdot \ n \ =$ 512, 1024, 2048, 4096. A: FID and memorization fraction, $f _ { \mathrm { m e m } } ,$ , versus training step τ. B: The normalized score divergence $\hat { A } _ { 1 }$ for train and held-out test samples versus $\tau ;$ bands show ±1 standard deviation across samples. Consistent lag between train-test $\hat { A } _ { 1 }$ split and onset of $f _ { m e m } \mathrm { C } \mathrm { : }$ The same $\hat { A } _ { 1 }$ curves and $f _ { \mathrm { m e m } }$ versus normalized training steps $\tau / n$ . D: For the $n = 4 0 9 6$ run, mean log basin-volume proxy for training and test images versus τ, with ±1 standard deviation.

Table 3: Membership inference from the normalized score divergence $\widehat { A } _ { 1 }$ at a single noise level, on the GMM. Conventions are identical to Table 2.
<table><tr><td>T</td><td>AUC</td><td>TPR</td><td>FPR</td><td>TPR@1%FPR</td></tr><tr><td>4,750</td><td>0.531</td><td>0.066</td><td>0.047</td><td>0.012</td></tr><tr><td>42,107</td><td>0.510</td><td>0.047</td><td>0.045</td><td>0.012</td></tr><tr><td>97,466</td><td>0.909</td><td>0.607</td><td>0.049</td><td>0.490</td></tr><tr><td>315,604</td><td>1.000</td><td>1.000</td><td>0.000</td><td>1.000</td></tr></table>

## G.5 OTHER TRAIN RESULTS

![](images/3893dbbae4430bf6d48c728a2ee4f7ebf684a2664731752006ce649f7ce5487a.jpg)  
Figure 18: FID and memorization ratio of CIFAR-10 DiT-S runs with Adam optimizer and Cosine Noise Scheduler with n=2048, 4096,8192.

## G.6 VISUALIZING THE SCORE-DIVERGENCE LANDSCAPE

Figure 1(a–b) visualizes the score-divergence landscape on a two-dimensional slice through the high-dimensional sample space. Because the score field lives in d dimensions $( d = 8$ for the GMM, $d = 1 0 2 4$ for CelebA), it cannot be plotted directly, so we follow the low-dimensional projection approach of Bihani et al. (2024), originally developed for visualizing the energy landscapes of atomic systems.

The idea is to pick two informative directions in the sample space and evaluate the landscape on a grid spanned by them. We choose these directions by principal component analysis, so that the plane we plot captures as much of the relevant variation as possible rather than an arbitrary slice. Concretely, we proceed in four steps:

1. Sample points. We collect a set of points from the region of interest—training samples together with samples along their denoising trajectories—which trace out the part of the landscape we want to see.

2. Choose an origin and two directions. We pick one point as the origin $x ^ { \star }$ , form the displacement of every other sampled point from $x ^ { \star }$ , and stack these displacements into a matrix. Its top two principal components ${ \bf d } _ { 1 }$ and $\mathbf { d } _ { 2 }$ (the directions of largest variance) define the plane we plot.

3. Build a grid. We form a grid of points $x ( \alpha , \beta ) = x ^ { \star } + \alpha { \bf d } _ { 1 } + \beta { \bf d } _ { 2 }$ by varying the two interpolating coefficients α and $\beta$ over a fixed range.

4. Evaluate the landscape. At every grid point we evaluate the score divergence $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } , t )$ and plot it as the height of the surface. Negative values (sharp, inward-pointing score) appear as wells, and near-zero values appear as flat regions.

Choosing the plane by principal components ensures that the slice reflects genuine structure in the landscape rather than an uninformative random cut: the top two components account for most of the variation among the sampled points, so the plotted plane passes through the directions along which the landscape actually changes. The resulting surface makes the difference between regimes directly visible—broad and featureless in the generalized regime, and punctuated by sharp wells around individual training samples in the memorized regime.

## G.7 HOW THE SCORE FUNCTION EVOLVES ACROSS TRAINING REGIMES

Energy-landscape geometry vs. cyclic-denoising noise amplitude across training regimes  
![](images/2a12b4c59d470a2f93d6012783b82dbf9cb65df97c23081f27e835e74e51a00e.jpg)  
Figure 19: Visualised for the GMM dataset - As γ decreases from left to right, the two-basin structure sharpens from a single merged attractor into two well-separated basins. The generalization window (row 2) already shows some sign of attractor formation, but this sharpening becomes more pronounced and localized only in the memorized regime (row 3), compared to the generalized regime (row 1) – directly quantifying its effect on the cyclic denoising dynamics.

![](images/d8471e7d3feb416313ea7381bd367617bf0a02334c3b0bca60bbdbed8d8cf70a.jpg)  
Figure 20: Normalized score divergence versus noise amplitude $\gamma$ across all training checkpoints, for CelebA (a) and the GMM (b). Each curve is one checkpoint, coloured by training regime: blue for the generalized regime and orange for the memorized regime, with lighter to darker shades as the training step τ increases within each regime. Line style distinguishes train, test, and generated samples. In the generalized regime (blue) the curves are shallow and cluster together across all $\gamma .$ . In the memorized regime (orange) the normalized divergence is strongly negative at small $\gamma$ and rises toward zero as $\gamma$ increases, with the effect deepening as τ grows.

## H FINER EXPERIMENT DETAILS

We train models from scratch on a synthetic Gaussian mixture, CelebA, and CIFAR-10, continuing training until the memorization fraction $f _ { \mathrm { m e m } }$ is high. We evaluate checkpoints spanning pre-generalization, generalization, and memorization. For all three datasets, checkpoint sample generation uses denoising diffusion implicit model (DDIM) sampling.

Synthetic data. We use an equally weighted two-component Gaussian mixture in $d = 8$ dimensions, $\begin{array} { r } { P _ { 0 } = \frac { 1 } { 2 } \mathcal { N } ( \pmb { \mu } , I _ { d } ) + \frac { 1 } { 2 } \mathcal { N } ( \dot { - } \pmb { \mu } , \dot { I } _ { d } ) } \end{array}$ , with $\pmb { \mu } = \mathbf { 1 } _ { d }$ . The population score is available analytically. The score network is a residual multilayer perceptron (MLP) with three residual blocks and sinusoidal time embeddings. The standard model we used has 102,024 parameters. Training-set sizes are $n \in \{ 1 2 8 , 2 5 6 , 5 1 2 , 1 0 2 4 , 2 0 4 8 , 4 0 9 6 \}$ . Models are trained using full-batch stochastic gradient descent (SGD) with learning rate $6 \times 1 0 ^ { - 3 }$ and momentum 0.95, for up to $4 \times 1 0 ^ { 6 }$ optimization steps. We use a linear noise schedule from $\beta _ { 1 } = 1 0 ^ { - 4 } \mathrm { t o } \beta _ { T } = 2 \times 1 0 ^ { - 2 } \mathrm { o v e r } T = 1 , 0 0 0$ diffusion timesteps. Generation uses DDIM with 200 denoising steps.

CelebA. We centre-crop and downsample CelebA (Liu et al., 2015) to $3 2 \times 3 2$ grayscale images, giving $d = 1 , 0 2 4$ . We apply no data augmentation. The score network is a denoising diffusion probabilistic model (DDPM) U-Net with three resolution levels, channel multipliers {1, 2, 4}, and attention at the two coarsest resolutions. The standard model we used has 16 million parameters.

We train separate SGD and Adam models on subsets of size $n \in \{ 5 1 2 , 1 0 2 4 , 2 0 4 8 , 4 0 9 6 \}$ . The SGD models use learning rate 0.01, momentum 0.95, and batch size $B = \operatorname* { m i n } ( n , 5 1 2 )$ , with no exponential moving average (EMA) or learning-rate warm-up. They are trained for at least $2 \times 1 0 ^ { 6 }$ optimization steps. The Adam models use learning rate $1 \dot { 0 } ^ { - 4 } , ( \dot { \beta } _ { 1 } ^ { \mathrm { A d a m } } , \beta _ { 2 } ^ { \mathrm { A d a m } } ) = ( 0 . 9 , 0 . 9 9 9 )$ ), batch size 512, and no weight decay. They are trained for $2 \times 1 0 ^ { 6 }$ optimization steps.

All models use a linear noise schedule from $\beta _ { 1 } = 1 0 ^ { - 4 }$ to $\beta _ { T } = 2 \times 1 0 ^ { - 2 }$ over T = 1,000 diffusion timesteps. Evaluation uses the same disjoint held-out set of 10,000 images across all runs. At each checkpoint, generation uses DDIM with 500 denoising steps.

CIFAR-10. We use CIFAR-10 (Krizhevsky, 2009) at its native $3 2 \times 3 2$ RGB resolution, giving $d = 3 { , } 0 7 2$ . We apply no data augmentation. We train unconditional pixel-space diffusion transformers (DiT-S/2) (Peebles & Xie, 2022). The architecture uses patch size 2, hidden width 384, 12 transformer blocks, 6 attention heads, and an MLP ratio of 4, giving approximately 32.6 million parameters.

We train separate models on subsets of size $n \in \{ 2 0 4 8 , 4 0 9 6 , 8 1 9 2 \}$ . All models use Adam with learning rate $1 0 ^ { - 4 } , ( \beta _ { 1 } ^ { \mathrm { A d a m } } , \beta _ { 2 } ^ { \mathrm { A d a m } } ) = ( 0 . 9 , 0 . 9 9 9 )$ , batch size 512, and no weight decay. They are trained for $2 \times 1 0 ^ { 6 }$ optimization steps. We use a cosine noise schedule (Nichol & Dhariwal, 2021) with offset $s = 0 . 0 0 8$ over $T = 5 0 0$ diffusion timesteps.

Evaluation uses the same set of 7,952 CIFAR-10 test images across all runs. The remaining 2,048 test images are reserved for loss evaluation. At each checkpoint, we generate 10,000 samples using DDIM with 100 denoising steps.

Stable Diffusion v1.4. We evaluate off-the-shelf Stable Diffusion v1.4 (Rombach et al., 2022) without retraining or fine-tuning, in its 4 × 64 × 64 latent space $( d = 1 6 , 3 8 4 )$ , with the model’s own scaled-linear noise schedule from $\beta _ { 1 } = 8 . 5 \times 1 0 ^ { - 4 } \mathrm { t o } \beta _ { T } = 1 . 2 \times 1 0 ^ { - 2 }$ over $T = 1 { , } 0 0 0$ timesteps. Each chain is initialized from a 50-step DDIM generation at guidance scale 7.5; every subsequent cycle perturbs the latent to noise amplitude $\gamma$ and denoises it back with 10 DDIM steps. Chains run for 200 cycles with 4 seeds per prompt, and seeds are fixed across γ, so that rows differing only in γ share a $c _ { 0 }$ image.

Prompts come from three sources. The individual chains of Fig. 21A–C use memorized prompts from (Hong et al., 2024), which pairs trigger prompts with the LAION images they reproduce; the six prompts used are listed verbatim in Table 4. The population panels Fig. 21D–E use additional plain english prompts from (Hong et al., 2024) and prompts from (Webster, 2023): In total 71 completely memorized (MV) and all 113 non-memorized (N) prompts, together with a fixed random subset of 150 of the 339 partially memorized (TV) prompts. Plain-English prompts, for which no training-image match exists, are the first 150 of the 499 non-memorized prompts of MemBench. Where a reference training image is available we measure SSCD; this replaces $f _ { \mathrm { m e m } } ,$ , which requires access to the training set. Seed agreement SA needs no reference image and is computed for every prompt.

<table><tr><td>row</td><td>prompt</td><td>idx</td><td> $\gamma$ </td><td>seed</td></tr><tr><td>hippo</td><td>Mothers influence on her young hippo</td><td>1262</td><td>0.9</td><td>2</td></tr><tr><td>laura</td><td>insights with laura unknown</td><td>67</td><td>0.1</td><td>2</td></tr><tr><td>laura</td><td>insights with laura unknown</td><td>67</td><td>0.9</td><td>2</td></tr><tr><td>air-con</td><td>air conditioners set for</td><td>215</td><td>0.9</td><td>1</td></tr><tr><td>air-con</td><td>air conditioners set for</td><td>215</td><td>0.9</td><td>0</td></tr><tr><td>label tape</td><td>set of nine brother tze 231 offlight white  $\mathtt { p - t o u c h } \mathtt { l a b e l }$ </td><td>662</td><td>0.9</td><td>2</td></tr></table>

Table 4: The six chains of Fig. 21A, verbatim. idx is the row index in the MemBench SD v1.4 prompt table (Hong et al., 2024), which also supplies the matched training image. The two laura rows share a prompt, and seed differing only in $\gamma ;$ the two air-con rows share a prompt and γ and differ only in seed.

Trajectory-level measurements use a dense grid of 31 cycles between $c _ { 0 }$ and $c _ { 2 0 0 }$ , with both SSCD and $\boldsymbol { A } _ { 1 }$ evaluated at every one. Population measurements use 15 cycles for SSCD and SA and, because the divergence probe dominates the cost, the subgrid {0, 10, 50, 100, 200} for $\mathcal { A } _ { 1 } ;$ the γ sweep covers 15 amplitudes from 0.005 to 0.9.

We estimate the unnormalized score divergence A with the forward-pass boundary-flux estimator $\left( p { = } 2 \right)$ of Brokman et al. (2025b), probing at $y = \sqrt { \bar { \alpha } _ { t } } x$ on a sphere of radius $r \stackrel { \cdot } { = } 1 0 ^ { - 2 } \sigma _ { t }$ with 16 directions arranged as 8 antithetic pairs. We evaluate at $t = 1$ and write $\mathcal { A } _ { 1 }$ . Conditional and unconditional branches use the same probe points and directions, and we report their difference $\Delta \mathcal { A } _ { 1 }$

For the detection benchmark of Appendix I we instead cycle at $\gamma = 0 . 9$ for 100 cycles with checkpoints at $c \in \{ 0 , 1 0 , 2 0 , 4 0 , 6 0 , 8 0 , 1 0 0 \}$ , over 500 memorized prompts from Wen et al. (2024) as positives and 2,000 negatives drawn evenly from LAION, COCO-2017 val, Lexica, and random CLIP-token strings, with 4 seeds each and the same probe settings.

Computational resources. Experiments were run on NVIDIA A100 GPUs, 80 GB GPU memory. UNet and DiT models were trained using data parallelism across two GPUs per run. Cyclic denoising used one GPU per run; trajectories evolved independently and were processed in batches. The cyclic experiments spanned multiple checkpoints and noise amplitudes on CelebA, CIFAR-10, and the GMM, with runs of up to 5,000 cycles.

## I DETECTION ON STABLE DIFFUSION

A  
![](images/4e9d2b5c454e3d71d97f8a0a4aef1fc3bcf1b1cc4e0412862e9f112b0cb48b6c.jpg)

![](images/98c813012c2bd5b85fa37d178f5196c2d3443a9d02623f8d070f987b0e45848a.jpg)

![](images/36d53a6ccec2a9ff487fd8115248dde6a5a944affec45efd8f3e459e7f7548cf.jpg)  
Figure 21: Cyclic denoising on off-the-shelf Stable Diffusion v1.4. Each cycle noises to amplitude $\gamma$ and denoises back; $c _ { k }$ denotes the sample after k cycles. (A) Six chains, one per row, with the matched LAION training image in the $\pm \mathtt { r a i n }$ column; each row label gives the prompt, the setting that varies, and the cycle at which the chain reaches its training image. The two laura rows share a prompt, a seed and a $c _ { 0 }$ image and differ only in $\gamma ;$ the air-con s1 row leaves its basin and re-enters it. (B,C) The same six chains per cycle: SSCD to the training image and $\Delta { \mathcal { A } } _ { 1 } .$ , on a shared axis. (D) $\Delta \mathcal { A } _ { 1 }$ after 200 cycles against $\gamma ,$ , over 71 completely memorised, 150 partially memorised and 113 non-memorised prompts (Webster, 2023) and 150 plain-English prompts (Hong et al., 2024). (E) Population means at $\gamma = 0 . 1$ over the 71 completely memorised and 113 nonmemorised prompts, 4 seeds each: seed agreement (solid, left) and $\Delta \mathcal { A } _ { 1 }$ (dashed, right); bands are ±SEM over prompts.

## I.1 DATA

For below experiments and comparison with (Wen et al., 2024) we use the following prompts Positives: 500 memorised prompts from Wen et al. (2024) Negatives: 2000 prompts, 500 each from LAION, COCO-2017 val, Lexica, and random CLIP-token strings. Cycling at $\gamma = 0 . 9$ with 10 DDIM steps per cycle; checkpoints at c = 0, 10, 20, 40, 60, 80, 100.

## I.2 UNCERTAINTY

Intervals are from a stratified percentile bootstrap over prompts: positives and negatives resampled independently with replacement, preserving the 500/2000 split, $\mathit { \dot { B } } = 4 0 0 0$ resamples, 95% per-

centile intervals. The same resample indices are reused across cycles and across statistics, so all differences quoted are paired.

## I.3 RESULTS BY CYCLE

<table><tr><td>cycle</td><td>AUC (95% CI)</td><td>TPR@1%FPR (95% CI)</td></tr><tr><td>0</td><td>0.811 [0.787, 0.836]</td><td>0.526 [0.482, 0.572]</td></tr><tr><td>10</td><td>0.890 [0.864, 0.914]</td><td>0.804 [0.766, 0.838]</td></tr><tr><td>20</td><td>0.935 [0.916, 0.953]</td><td>0.834 [0.804, 0.872]</td></tr><tr><td>40</td><td>0.931 [0.910, 0.950]</td><td>0.858 [0.824, 0.892]</td></tr><tr><td>60</td><td>0.944 [0.926, 0.961]</td><td>0.866 [0.830, 0.900]</td></tr><tr><td>80</td><td>0.934 [0.914, 0.953]</td><td>0.856 [0.820, 0.888]</td></tr><tr><td>100</td><td>0.940 [0.922, 0.957]</td><td>0.852 [0.812, 0.884]</td></tr></table>

Table 5: $\Delta A$ at $\gamma \ = \ 0 . 9 .$ . Paired difference between $c = ~ 6 0$ and $c = 4 0 \colon + 0 . 0 1 6 ,$ , 95% CI $[ - 0 . 0 1 0 , + 0 . 0 4 1 ]$

![](images/55c58f8cb7ad84a537f5fd14d87c5ddc48d9b5cf32970a27a70f7b00b7c4964c.jpg)

![](images/1e79798ca23eb79b43c36a0a417a1855b8abc57470771210d54665f60ccd2212.jpg)  
Figure 22: AUC and TPR@1%FPR against cycle count, with bootstrap bands.

## I.4 COMPARISON

All rows use the same prompts and the same seeds.

<table><tr><td>statistic</td><td>AUC (95% CI)</td><td>TPR@1%FPR (95% CI)</td><td>AP</td></tr><tr><td>∆A (ours)</td><td>0.944 [0.926, 0.961]</td><td>0.866 [0.830, 0.900]</td><td>0.933</td></tr><tr><td> $A _ { \mathrm { c o n d } } ~ \mathrm { a l o n e }$ </td><td>0.948 [0.939, 0.957]</td><td>0.194 [0.142, 0.246]</td><td>0.782</td></tr><tr><td> $A _ { \mathrm { u n c o n d } }$  alone</td><td>0.899 [0.884, 0.914]</td><td>0.066 [0.030, 0.094]</td><td>0.623</td></tr><tr><td>Wen SDN (single shot)</td><td>0.990 [0.986, 0.993]</td><td>0.838 [0.786, 0.882]</td><td>0.970</td></tr></table>

Table 6: ${ \mathrm { A t ~ } } c = 6 0 , \gamma = 0 . 9 .$ Paired differences, ∆A minus SDN: AUC −0.0456, 95% CI $[ - 0 . 0 6 3 3 , - 0 . 0 2 8 7 ]$ ; TPR@1%FPR +0.0265, 95% CI $\left[ - 0 . 0 2 6 0 , + 0 . 0 8 4 1 \right]$

![](images/7f04d942d76826b1db16e3a82af86ddf004e41defdb2e5e9b8b080125d4e68fc.jpg)

![](images/5510dfb045ba22cc436f317d17282dbfab31cee8b0bb119b8a11ec3243a21f93.jpg)  
Figure 23: ROC (left) and precision–recall (right) at $c = 6 0$ . Class balance is 1:4; the dotted line in the right panel marks the positive rate.

## I.5 RAW DIVERGENCE

![](images/040dbe63d81b0b5ce251c4089ee3172d48391e971998c7f6df0c544cdb49e318.jpg)  
Figure 24: Raw $\Delta A$ against cycle, population means with 95% intervals. Memorised prompts move from $- 1 . 2 4 \times 1 0 ^ { 5 }$ at $\dot { c } = 0 \dot { \mathrm { ~ t ~ o ~ } } - \dot { 3 } . \dot { 1 } 2 \times 1 0 ^ { 5 }$ at $c = 6 0 ;$ non-memorised prompts remain within $0 . 0 3 \times 1 0 ^ { 5 }$ of zero.

## I.6 BY NEGATIVE SOURCE

<table><tr><td colspan="2"></td><td colspan="2"> $\Delta A , c = 6 0$ </td><td colspan="2">Wen SDN, single shot</td></tr><tr><td>negatives</td><td>n</td><td>AUC</td><td>TPR@1%FPR</td><td>AUC</td><td>TPR@1%FPR</td></tr><tr><td>laion</td><td>500</td><td>0.942</td><td>0.820</td><td>0.967</td><td>0.740</td></tr><tr><td>coco</td><td>500</td><td>0.945</td><td>0.884</td><td>0.996</td><td>0.950</td></tr><tr><td>lexica</td><td>500</td><td>0.945</td><td>0.892</td><td>0.997</td><td>0.976</td></tr><tr><td>random</td><td>500</td><td>0.944</td><td>0.876</td><td>1.000</td><td>0.998</td></tr></table>

Table 7: Each row scores the 500 positives against one negative source.