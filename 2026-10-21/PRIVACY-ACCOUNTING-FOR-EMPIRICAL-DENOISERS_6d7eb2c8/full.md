# PRIVACY ACCOUNTING FOR EMPIRICAL DENOISERS

William Cheng University of British Columbia willc135@cs.ubc.ca

Dongjun Kim <sup>∗</sup> Stanford University dongjun@stanford.edu

Mijung Park University of British Columbia mijungp@cs.ubc.ca

## ABSTRACT

Diffusion models have emerged as the foundation of modern generative systems, yet their high memorization capacity raises privacy concerns. While differentially private (DP) training provides formal guarantees, it comes at a cost in sample quality. In this work, we study a complementary question: how much privacy loss is incurred during the sampling process itself? Directly analyzing modern neural denoisers is challenging because their sensitivity to a single training example is generally intractable. To overcome this challenge, we construct and analyze a clipped privacy-accountable empirical denoiser, a training-free estimator of the Bayes-optimal denoiser, whose per-step sensitivity can be computed explicitly, allowing each denoising step to be a Gaussian mechanism. This perspective enables privacy accounting under Gaussian DP when each training example contributes to each denoising step, and under f-DP trade-off functions when each generated sample uses a single randomly subsampled block of the training data. We report every (ε, δ) guarantee as a certified upper bound from a privacy loss distribution accountant, and use a cumulant-based Edgeworth approximation as a fast diagnostic, characterizing where it is reliable and where it becomes anti-conservative. We also analyze the privacy loss along the denoising trajectory and show that it grows steeply near the critical window in which semantic content is formed. This observation motivates hybrid samplers that use the private empirical denoiser through the formative steps where it remains a faithful proxy for the neural denoiser, then switch to a public neural denoiser for the remaining refinement. Our results on CIFAR-10 and CelebA-64 show that the privacy of diffusion models depends not only on the training procedure, but also on sampling-time design choices such as subsampling, stopping time, and hybrid denoising.

## 1 INTRODUCTION

Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020) are now central to modern generative systems, but their ability to reproduce training examples raises serious privacy concerns (Wu et al., 2023; Carlini et al., 2023; Tang et al., 2024; Hu & Pang, 2023; Duan et al., 2023; Matsumoto et al., 2023). Post-hoc filters and memorization-aware model changes can reduce visible leakage, but they do not give attack-agnostic privacy guarantees. Differentially private (DP) training gives such guarantees, but it answers a different question: it bounds what the weights reveal, while users and regulators see only generated samples. Sampling-time leakage is a separate surface that persists even when the denoiser is trained non-privately, whether or not DP training is practical at scale. This raise a complementary question: how much privacy loss is incurred by the sampling process itself?

Answering this requires treating the sampler, rather than the training algorithm, as the mechanism to be analyzed. The obstacle is that a neural denoiser has intractable data sensitivity, so its denoising steps admit no explicit privacy accounting. To make the problem analyzable, we privatize an empirical denoiser (Niedoba et al., 2025) whose sensitivity can be computed directly. Since each reverse diffusion step already injects Gaussian noise, the empirical denoising update can be interpreted as a Gaussian mechanism, allowing us to track the sampling trajectory with Gaussian Differential Privacy (GDP) (Dong et al., 2022).

However, the GDP composition using the full batch of training data yields a poor privacy level. Denoising with subsampled data at each denoising step can improve the privacy level. However, this hurts the diffusion sampling, as every newly drawn subset of data injects extra stochastic drift into the sampling trajectory, which prevents coherent semantics from forming across the trajectory. To this end, we propose output-wise subsampling, subsample a block from the dataset once and reuse it for the entire denoising trajectory to produce one sample. This stabilizes the denoising direction and achieves a decent privacy level.

When converting f-DP to $( \epsilon , \delta ) \ / – \mathrm { D P } ,$ , we encounter unreliable conversion, especially at small $\delta ,$ when using the CLT-type $f \mathrm { - D P }$ accounting that assumes the CDF of the standardized test statistic under the composition is simply Gaussian. We report all guarantees as certified upper bounds from a PLD accountant (Doroshenko et al., 2022), and study a cumulant-based Edgeworth approximation (Zheng et al., 2020) as a fast estimate and measure where it is accurate and where it understates the certified value.

Our contributions are summarized below:

1. We privatize a clipped empirical denoiser with explicit per-step sensitivity of $2 C / | \mathcal { D } |$ Diffusion sampling then becomes a composition of Gaussian mechanisms, with no model trained on the private data.

2. We derive privacy guarantees for full-batch and output-wise subsampled diffusion sampling using $\mathrm { G D P }$ and f-DP trade-off functions, and report certified $( \epsilon , \bar { \delta } )$ upper bounds from a PLD accountant. We use the Edgeworth approximation as a diagnostic where cumulant checks pass, and characterize where it fails. Subsampling amplification applies once per reused block, rather than once per step.

3. We show that the per-step privacy cost grows along the trajectory relative to the critical window, measure where the empirical denoiser is a window-local semantic proxy for a neural denoiser, and use both to place the private-to-public handoff, evaluated on CIFAR-10 and CelebA-64.

## 2 BACKGROUND

## 2.1 DENOISING DIFFUSION MODELS

Diffusion models are built upon the forward and reverse processes (Ho et al., 2020; Song et al., 2020). The forward perturbation describes how the noise is gradually corrupting data sample $\mathbf { x } _ { \mathrm { 0 } }$ defined as ${ \bf x } _ { t } = \alpha _ { t } { \bf x } _ { 0 } + \sigma _ { t } \epsilon$ , where ϵ is a standard Gaussian variable, $\alpha _ { t }$ and $\sigma _ { t }$ are prescribed time-dependent mean and variance coefficients, respectively. For simplicity, we describe the diffusion process from EDM (Karras et al., 2022), but our framework is adaptable to any diffusion processes such as DDPM (Ho et al., 2020) or Flow Matching (FM) (Lipman et al., 2022). In EDM, the forward diffusion process is simply defined by ${ \bf x } _ { t } = { \bf x } _ { 0 } + t \epsilon$ , i.e., $\alpha _ { t } \equiv 1$ and $\sigma _ { t } = t$ , and the reverse denoising process is computed by

$$
\mathbf { x } _ { t - \Delta t } = \left( 1 - 2 \frac { \Delta t } { t } \right) \mathbf { x } _ { t } + 2 \frac { \Delta t } { t } \mathbb { E } [ \mathbf { x } | \mathbf { x } _ { t } ] + \sqrt { 2 t \Delta t } \epsilon .\tag{1}
$$

The term $\mathbb { E } [ \mathbf { x } | \mathbf { x } _ { t } ]$ , called a Bayes optimal denoiser, is the conditional expectation of the clean data given the noisy observation $\mathbf { x } _ { t }$ . In practice, a neural denoiser $D ( \mathbf { x } _ { t } , t ; \mathbf { \theta } \cdot \mathbf { \tilde { \phi } } ) )$ approximates this quantity, where $\pmb \theta ( \mathcal D )$ are parameters learned from dataset D. By rolling out eq. 1 up to time zero, diffusion model gradually synthesizes a sample.

## 2.2 DIFFERENTIAL PRIVACY AND GAUSSIAN MECHANISM

Differential Privacy (DP) provides a formal framework to limit the influence of any single record on the output of a randomized algorithm. Intuitively, whether or not an individual’s data is present, the

distribution of outputs should remain (almost) unchanged. This view naturally extends to diffusion sampling, which is itself a randomized mechanism.

A fundamental building block is the Gaussian mechanism: given a function $h : \mathcal { D } \mapsto \mathbb { R } ^ { p }$ , the mechanism releases

$$
\mathcal { M } _ { \mathrm { G a u s s } } ( \mathcal { D } ; h ) = h ( \mathcal { D } ) + { \pmb n } , \qquad { \pmb n } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \Delta _ { h } ^ { 2 } { \bf I } ) ,\tag{2}
$$

where $\Delta _ { h }$ is the global sensitivity—the maximum change in $h$ between neighboring datasets. By calibrating the noise to $\Delta _ { h }$ , the Gaussian mechanism guarantees DP (see Appendix H.1 for formal definitions).

For analyzing multi-step procedures such as diffusion sampling, we adopt Gaussian Differential Privacy (GDP) (Dong et al., 2022). GDP tightly characterizes the privacy loss of Gaussian mechanisms and, crucially, admits an exact composition rule:

$$
\mu = \sqrt { \sum _ { i } \mu _ { i } ^ { 2 } } ,\tag{3}
$$

where $\mu _ { i }$ is the GDP parameter of the i-th step. For example, in the case of eq. 2 with $_ n \sim$ ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } \Delta _ { h } ^ { 2 } \mathbf { I } )$ , the Gaussian mechanism corresponds to $\mu _ { i } – \mathbf { G D P }$ with $\textstyle \mu _ { i } = { \frac { 1 } { \sigma } }$ . This makes GDP especially well-suited for diffusion processes, which inject Gaussian noise at every denoising step.

Two further facts are important for our analysis. First, privacy amplification by subsampling can further reduce the effective $\mu .$ . Second, GDP parameters can be translated into $( \epsilon , \bar { \delta } )$ guarantees when needed for regulatory reporting; we provide details in Appendix H.2. Together, these tools allow us to connect the denoising process directly to established DP accounting methods.

## 2.3 PRIVACY LOSS DISTRIBUTIONS

$( \epsilon , \delta )$ -DP is equivalently characterized by the hockey-stick divergence: a mechanism is $( \epsilon , \delta )$ -DP iff ${ \bf \ddot { \mathnormal { H } } } _ { e ^ { \epsilon } } ( M ( \mathcal { D } ) \parallel \mathbf { \dot { \mathnormal { M } } } ( \mathcal { D } ^ { \prime } ) ) \dot { \leq \delta }$ for all neighboring $\mathcal { D } , \dot { \mathcal { D } } ^ { \prime }$ , where $\begin{array} { r } { \bar { H _ { \gamma } } ( P \| Q ) = \int [ P ( x ) - \gamma \dot { Q } ( x ) ] . } \end{array}$ <sub>+</sub> dx. The privacy loss distribution (PLD) is the law of the log-likelihood ratio log $\frac { P ( x ) } { Q ( x ) }$ under $P ;$ the hockey-stick divergence, and hence $\delta ( \epsilon )$ , is a functional of it, and the PLD of a composition is the convolution of the individual PLDs. Numerical PLD accountants evaluate this convolution on a discretized support (Gopi et al., 2021; Doroshenko et al., 2022). Discretization can be performed pessimistically, overestimating $H _ { e ^ { \epsilon } }$ for every ϵ and hence yielding a valid $( \epsilon , \delta )$ upper bound, or optimistically, underestimating it. We use the pessimistic estimate for all reported guarantees, and the optimistic estimate to quantify how much of the reported ϵ is discretization slack rather than true privacy loss. This distinction is what we mean by certified throughout: the reported value is an upper bound by construction, not an approximation of unknown sign.

## 3 PRIVACY ANALYSIS OF DENOISING PROCESS WITH EMPIRICAL DENOISER

## 3.1 SINGLE-STEP DENOISING IS A GDP

The Bayes denoiser in eq. 1 is the conditional expectation $\begin{array} { r l } { \mathbb { E } [ { \bf x } _ { 0 } \vert { \bf x } _ { t } ] } & { { } = } \end{array}$ $\begin{array} { r } { p _ { t } ( \mathbf { x } _ { t } ) ^ { - 1 } \int \mathbf { x } _ { 0 } p ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) p ( \mathbf { x } _ { 0 } ) d \mathbf { x } _ { 0 } } \end{array}$ , where $p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { t } ; \bar { \mathbf { x } _ { 0 } } , t ^ { 2 } I )$ is the closed-form EDM forward transition density and $\begin{array} { r } { p _ { t } ( \mathbf { x } _ { t } ) = \int p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) p ( \mathbf { x } _ { 0 } ) d \mathbf { x } _ { 0 } } \end{array}$ is the marginal of the noisy state. Real-world samplers approximate this Bayes denoiser with a neural denoiser $D ( \mathbf { x } _ { t } , t ; \pmb { \theta } ( \mathcal { D } ) )$ but the sensitivity of $\mathbf { \bar { \rho } } _ { D }$ to changing one training example is not tractable, so it admits no direct privacy accounting. We therefore analyze the empirical denoiser, which replaces $p ( \mathbf { x } _ { 0 } )$ with the empirical measure over $\mathcal { D }$ and uses the empirical marginal $\begin{array} { r } { \widehat { p _ { t } } ( \mathbf { x } _ { t } ) = \frac { 1 } { | \mathcal { D } | } \sum _ { \mathbf { x } _ { 0 } \in \mathcal { D } } \hat { p ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } ) } } \end{array}$ as its normalizer. Unlike the transition kernel, $\widehat { p } _ { t }$ is data-dependent and cannot enter a private mechanism without its own sensitivity accounting.

$$
\hat { \mathbb { E } } [ { \mathbf { x } } | { \mathbf { x } } _ { t } ; \mathcal { D } ] = \frac { 1 } { \widehat { p _ { t } } ( { \mathbf { x } } _ { t } ) | \mathcal { D } | } \sum _ { { \mathbf { x } } _ { 0 } \sim \mathcal { D } } { \mathbf { x } } _ { 0 } p ( { \mathbf { x } } _ { t } | { \mathbf { x } } _ { 0 } ) .\tag{4}
$$

![](images/9d3e30b46b08e561b13c10ae611c46b9e4d9853fc2f1a9e8c453d19dd20a7e14.jpg)

Figure 1: Cosine similarity between the denoising directions, i.e., Cos $( \mathbf { x } _ { t } - \hat { \mathbb { E } } , \mathbf { x } _ { t } - D )$ , by empirical and neural denoisers. Here, $\mathbf { x } _ { t } - \dot { \mathbb { E } }$ denotes the actual denoising direction induced by the update rule in eq. 1. Results are reported along the log-SNR axis (Kingma et al., 2021), where $\mathrm { S N R } = { 1 } / { 4 t } ^ { \overline { { 2 } } }$ , providing a unified noise-scale representation across diffusion formulations $( \mathrm { e . g . }$ ., EDM, DDPM, FM).

Substituting eq. 4 into eq. 1 gives

$$
\begin{array} { r } { \mathbf { x } _ { t - \Delta t } = \Big ( 1 - \frac { 2 \Delta t } { t } \Big ) \mathbf { x } _ { t } + \frac { 2 \Delta t } { t } \underbrace { \frac { 1 } { p _ { t } ( \mathbf { x } _ { t } ) | \mathcal { D } | } \sum _ { \mathbf { x } _ { 0 } \in \mathcal { D } } \mathbf { x } _ { 0 } p ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } ) } _ { \mathrm { d a t a - d e p e n d e n t } } + \sqrt { 2 t \Delta t } \epsilon , } \end{array}\tag{5}
$$

so the only term depending on $\mathcal { D }$ is the empirical average, and the noise $\sqrt { 2 t \Delta t } e$ ϵ already injected by the sampler can serve as the noise of a Gaussian mechanism once this average has bounded sensitivity. Let $\mathbf { v } ( \mathbf { x } _ { 0 } ; \mathbf { x } _ { t } ) = \mathbf { x } _ { 0 } p ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } )$ and let $\bar { \mathbf { v } } ( \mathbf { x } _ { 0 } ; \mathbf { x } _ { t } ) = \mathbf { v } ( \mathbf { x } _ { 0 } ; \mathbf { x } _ { t } ) / \operatorname* { m a x } ( 1 , \| \mathbf { v } ( \bar { \mathbf { x } _ { 0 } } ; \mathbf { x } _ { t } ) \| / C )$ be its clipped version, so that $\| { \bar { \mathbf { v } } } \| \leq C$ . Computing $\widehat { p } _ { t } ( \mathbf { x } _ { t } )$ privately would require an additional data-dependent pass, so we replace it with a public constant $\beta ( \beta = 1$ in experiments). This yields a deliberately unnormalized empirical mechanism. $\beta = 1$ is a design choice, not a claim that $\widehat { p _ { t } } ( \mathbf x _ { t } ) \approx 1$ , and $\beta$ is a genuine privacy–utility parameter that we ablate in Appendix E. Substituting the clipped average into eq. 1 gives

$$
\begin{array} { r } { { \bf x } _ { t - \Delta t } = \left( 1 - \frac { 2 \Delta t } { t } \right) { \bf x } _ { t } + \frac { 2 \Delta t } { t } \frac { 1 } { \beta | \mathcal { D } | } \sum _ { { \bf x } _ { 0 } \in \mathcal { D } } \bar { \bf v } ( { \bf x } _ { 0 } ; { \bf x } _ { t } ) + \sqrt { 2 t \Delta t } \epsilon , } \end{array}
$$

and the public invertible rescaling by $\beta t / ( 2 \Delta t )$ shows that the data-dependent part of one step is the Gaussian mechanism

$$
\mathcal { M } _ { \mathrm { G a u s s } } ( \mathcal { D } ; { \bf x } _ { t } ) = \frac { 1 } { | \mathcal { D } | } \sum _ { { \bf x } _ { 0 } \in \mathcal { D } } \bar { \bf v } ( { \bf x } _ { 0 } ; { \bf x } _ { t } ) + \beta \sqrt { \frac { t ^ { 3 } } { 2 \Delta t } } \epsilon .\tag{6}
$$

For neighboring datasets, where $\mathcal { D } ^ { \prime }$ adds or removes one record of $\mathcal { D } _ { : }$ the clipped average has sensitivity at most $2 C / | \mathcal { D } |$ , so eq. 6 is $\mu _ { t } – \mathbf { G D P }$ with $\begin{array} { r } { \mu _ { t } = \frac { 2 C } { \beta | \mathcal { D } | } \sqrt { 2 \Delta t / t ^ { 3 } } } \end{array}$ . This bound is conservative in the sense that it treats the empirical denoising direction as the private release at every step and accounts for the sampling noise already injected by the diffusion process.

## 3.2 MULTI-STEP DENOISING IS ALSO A GDP

For discretized timesteps $t _ { i }$ with increments $\Delta t _ { i }$ , the i-th empirical denoising step has $\begin{array} { r l } { \mu _ { t _ { i } } } & { { } = } \end{array}$ $\frac { 2 C } { \beta | \mathcal { D } | } \sqrt { 2 \Delta t _ { i } / t _ { i } ^ { 3 } }$ . By exact GDP composition, denoising from $t _ { m }$ to $t _ { n }$ is $\mu { \mathrm { - G D P } }$ with $\mu ~ =$ $( \sum _ { i = n } ^ { m } \mu _ { t _ { i } } ^ { 2 } ) ^ { 1 / 2 }$ . The parameter $\mu$ has a direct testing interpretation: larger $\mu$ means it is easier to distinguish neighboring datasets from the generated output. Privacy loss is therefore far from uniform along the trajectory. Under the EDM schedule $\Delta t _ { i } \approx c t _ { i } ^ { 1 - 1 / \rho } , \mathrm { { s o } } \mu _ { t _ { i } } \propto t _ { i } ^ { - 1 - 1 / ( 2 \rho ) }$ , which is roughly $1 \bar { / } t _ { i }$ for $\rho = 7$ (Figure 5). The last low-noise steps therefore dominate the sum. This suggests a hybrid sampler: run the private empirical denoiser through the early and middle steps, then hand off to a public denoiser, whose steps are post-processing and add no privacy cost (Section 6).

## 3.3 BRIDGING EMPIRICAL AND NEURAL DENOISERS

A central limitation is that real diffusion samplers use neural denoisers, while our formal accounting uses an empirical denoiser. Directly accounting for neural denoisers remains open because the sensitivity of a trained network to one training example is not tractable. We therefore use the empirical denoiser as an analyzable proxy and check where this proxy is informative.

First, Figure 1 shows that empirical and neural denoising directions are nearly aligned on CIFAR-10 in the earlier stages of the denoising process. Second, recent work on critical windows (Li & Chen,

Table 1: Endpoint distance (1 − cos) after substituting the empirical denoiser inside one window (swap) and after re-drawing only that window’s noise with the neural denoiser kept (ref.). rel. = swap/ref.
<table><tr><td></td><td></td><td colspan="3">CLIP (Radford et al., 2021)</td><td colspan="3">DINO (Oquab et al., 2024)</td></tr><tr><td>Window</td><td>t range</td><td>swap</td><td>ref.</td><td>rel.</td><td>swap</td><td>ref.</td><td>rel.</td></tr><tr><td>high-noise</td><td>19.15-7.47</td><td>0.0097</td><td>0.0917</td><td>0.11×</td><td>0.0717</td><td>0.5792</td><td>0.12×</td></tr><tr><td>mid-noise</td><td>6.09-1.98</td><td>0.0772</td><td>0.1000</td><td>0.77×</td><td>0.5175</td><td>0.6203</td><td>0.83×</td></tr><tr><td>low-noise</td><td>1.54–0.38</td><td>0.1127</td><td>0.0687</td><td>1.64×</td><td>0.6910</td><td>0.4431</td><td>1.56×</td></tr><tr><td>min-noise</td><td>0.14-0.018</td><td>0.2147</td><td>0.0241</td><td>8.90×</td><td>0.8758</td><td>0.1376</td><td>6.37×</td></tr></table>

2024; Biroli et al., 2024; Sclocchi et al., 2024; Georgiev et al., 2023; Raya & Ambrogioni, 2023) suggests that coarse semantic identity is determined early and then remains stable, so the formative part of the trajectory overlaps with the regime where the two denoisers are most similar. Together, these checks do not prove a neural-denoiser DP guarantee, but they justify using the empirical accountant to study where sampling leakage occurs.

Cosine similarity captures only the denoising direction; the per-step magnitude gap $\| \widehat { \mathbb { E } } - D \| ^ { 2 }$ is reported in Appendix F, where it peaks inside the mid-noise window, but as Table 1 shows, does not survive to the released sample. We therefore restrict the proxy claim to directional agreement over the regime our private block occupies.

Trajectory-level agreement. The per-step divergence above bounds how far the two kernels can differ at a single step, but not how much of that difference survives to the released sample. We therefore substitute the empirical denoiser for the neural denoiser within one window at a time, with the initial noise and all stochastic increments held fixed, and measure the embedding distance (1−cos) of the resulting sample to the all-neural reference trajectory. As a descriptive scale, we also keep the neural denoiser and re-draw only the noise increments inside that window. Table 1 reports both distances over 2048 paired trajectories, and Appendix F adds absolute similarities, paired-bootstrap CIs, PSNR, SSIM, and label agreement (Efron & Tibshirani, 1993; Wang et al., 2004). In the highand mid-noise windows substitution moves the endpoint less than re-drawing that window’s noise (mid-noise: 0.77× CLIP, 0.83× DINO, reference minus swap distance +0.0228 in CLIP, 95% CI $\mathrm { \Delta \mathrm { [ + 0 . 0 2 0 7 , + 0 . 0 2 4 8 ] } } \mathrm { \Delta }$ ). In the mid-noise window endpoint similarity stays high in CLIP (0.923) but falls in DINO (0.483), so coarse semantics are preserved while instance-level structure is not. In the low-noise window the ordering reverses (CLIP difference −0.0441, 95% CI [−0.0459, −0.0422]), consistent with the $t ^ { - 3 }$ growth of the per-step divergence. The min-noise window is always handled by the public denoiser, while the low-noise window is private in our 65% configuration (Section 6.3).

Taken together, these measurements support a claim: through the mid-noise window, about the first half of generation, the empirical denoiser is a window-local semantic proxy for the neural denoiser. They do not establish a $\mathrm { D } \dot { \mathrm { P } }$ guarantee for a neural denoiser by itself, which remains open.

## 4 PRIVACY ANALYSIS OF DENOISING PROCESS WITH SUBSAMPLED DATA

Section 3 analyzes the empirical denoiser when every record in D contributes to every private denoising step. This full-batch accountant is tight, but it is expensive in terms of privacy: each generated output repeatedly touches every record, and privacy loss grows quickly when many outputs are released. Subsampling reduces this exposure because a record that is not selected contributes no privacy loss. Throughout, each generated output draws a minibatch $\mathcal { S } \subseteq \mathcal { D }$ of fixed size B uniformly without replacement, so a fixed record is included with probability $q _ { B } = B / n$ , and neighboring datasets differ by adding or removing one record. Because $| { \cal S } | = B$ is fixed, an added record that is selected takes the place of another record of $s ,$ so within the minibatch the change is a replacement (Zhu et al., 2022; Lebeda et al., 2025). The clipped minibatch average therefore has sensitivity $2 C / B$ , and each step is $\mu _ { t } – \mathbf { G D P }$ with $\begin{array} { r } { \mu _ { t } = \frac { 2 C } { \beta B } \sqrt { 2 \Delta t / t ^ { 3 } } } \end{array}$ conditional on inclusion. The subsampled Gaussian mechanism is $\begin{array} { r } { \mathcal { M } _ { \mathrm { S u b G a u s s } } ( \mathcal { D } ; \boldsymbol { x } _ { t } ) = | \mathcal { S } | ^ { - 1 } \sum _ { \boldsymbol { x } _ { 0 } \in \mathcal { S } } \bar { \mathbf { v } } ( \boldsymbol { x } _ { 0 } ; \boldsymbol { x } _ { t } ) + \beta \sqrt { t ^ { 3 } / ( 2 \Delta t ) } \epsilon } \end{array}$

![](images/51a873c1892a7f021d8701b885aa5200883babf7a4ab471d088256aafd1e3a97.jpg)  
Figure 2: Output-wise vs. per-step subsampling, $B = 2 5 ,$ , identical initial noise. Each row is one sample; columns show decreasing noise levels. Under output-wise subsampling (a) the subsample is reused, so each row progressively denoises the same emerging image. Under per-step subsampling (b) a fresh subsample is drawn each step, pulling the trajectory toward a different subset: semantic content changes between columns and never settles. This instability motivates block reuse.

The important design choice is how often we resample S. Resampling at every denoising step for every output gives more independent amplification events, but it also changes the empirical denoising direction at every step. Small minibatches, which is better for privacy, inject extra stochastic drift into the reverse trajectory: the sample is repeatedly pulled toward different private subsets, and late low-noise steps are especially sensitive to this instability, as shown in Figure 2. This can harm sample quality even though the privacy accountant looks attractive. We therefore employ output-wise subsampling: sample once for a generated output, reuse them through an empirical denoising block, then account for amplification once at the block level. This keeps the denoising direction stable while still limiting how often any record is accessed.

We state both accountants because they correspond to different mechanisms. Applying per-step amplification to output-wise subsampling would be incorrect: privacy amplification is tied to independent subsampling events, not to the number of denoising steps. If the same minibatch is reused over a block B, the included record accumulates the full block-level Gaussian parameter $\textstyle { \bar { \mu } } B = ( \sum _ { t \in B } \mu _ { t } ^ { 2 } ) ^ { 1 / 2 }$ before amplification is applied. If a fresh minibatch is drawn at every step, each $\mu _ { t }$ is amplified separately.

For full-batch Gaussian mechanisms, GDP composition is exact. With subsampling, each block’s trade-off curve is a mixture rather than a Gaussian curve, so exact GDP composition no longer applies, and we compose numerically with a pessimistic PLD accountant (Section 2.3). The GDP central limit theorem (Dong et al., 2022) fits poorly here because the $\mu _ { t }$ are highly heterogeneous, and its Berry-Esseen error is additive and too coarse for small δ (Table 11). As a fast estimate, we also compute a degree-two Edgeworth approximation (Zheng et al., 2020), which adds skewness and kurtosis corrections from the first four cumulants (eq. 25, Appendix D). It is an approximation and matches the certified PLD value to within 1% for $\varepsilon \leq 2$ but understates it for larger ε, by up to 15% in our runs (Appendix G).

Proposition 4.1 (Output-wise subsampling). Consider a sampler that, for each of $N _ { \mathrm { o u t } }$ generated outputs, performs private subsampling once and reuses themfor an empirical denoising block $B ;$ all later public denoising is post-processing. Let q<sub>B</sub> be the block-level inclusion probability of a fixed record and $\textstyle { \bar { \mu } } B = ( \sum _ { t \in B } ^ { } \mu _ { t } ^ { 2 } ) ^ { 1 / 2 }$ . Define $h _ { B } = q _ { B } G _ { \bar { \mu } _ { B } } + ( 1 - q _ { B } )$ Id and

$$
g _ { \mathrm { o u t } } = \bigotimes _ { j = 1 } ^ { N _ { \mathrm { o u t } } } h _ { B } , \qquad f _ { \mathrm { o u t } } = \operatorname* { m i n } \{ g _ { \mathrm { o u t } } , g _ { \mathrm { o u t } } ^ { - 1 } \} ^ { * * } .\tag{7}
$$

Then the joint release ofthe $N _ { \mathrm { o u t } }$ generated outputs is $f _ { \mathrm { o u t } } { - } D P .$ Equivalently, for every $\varepsilon \geq 0$ it is $( \varepsilon , \delta _ { \mathrm { o u t } } ( \varepsilon ) ) – D P$ with

$$
\delta _ { \mathrm { o u t } } ( \varepsilon ) = \operatorname* { m a x } _ { \substack { \mathrm { d i r } \in \{ P  Q , Q  P \} } } [ P r _ { Q } ( S _ { \mathrm { o u t } } ^ { \mathrm { d i r } } > \varepsilon ) - e ^ { \varepsilon } P r _ { P } ( S _ { \mathrm { o u t } } ^ { \mathrm { d i r } } > \varepsilon ) ] _ { + } ,\tag{8}
$$

where $\begin{array} { r } { S _ { \mathrm { o u t } } = \sum _ { j = 1 } ^ { N _ { \mathrm { o u t } } } L _ { \mathcal { B } } ( Z _ { j } ) , L _ { \mathcal { B } } ( z ) = \log ( 1 - q _ { \mathcal { B } } + q _ { \mathcal { B } } \exp ( \bar { \mu } _ { \mathcal { B } } z - \bar { \mu } _ { \mathcal { B } } ^ { 2 } / 2 ) ) } \end{array}$ , and $\otimes$ indicates a tensor product. The probabilities are taken under the two neighboring-data hypotheses. We evaluate $\delta _ { \mathrm { o u t } } ( \varepsilon )$ with a pessimistic PLD accountant, which gives the certified values we report, and use the Edgeworth expansion only as an estimate (Appendix G).

Algorithm 1 instantiates this: the subsample is drawn once per output (line 3) and reused across the block, so amplification is applied once to $\bar { \mu } _ { B }$ rather than per step. In practice, each output enters the PLD accountant (Section 2.3) as one subsampled Gaussian mechanism with noise multiplier $1 / \bar { \mu } _ { B }$ and sampling probability q<sub>B</sub>, whose two neighboring directions give exactly $h _ { B }$ and $h _ { B } ^ { - 1 }$ . Composing $N _ { \mathrm { o u t } }$ copies gives $\delta _ { \mathrm { o u t } } ( \varepsilon )$ , and Appendix G lists every step and setting. The proof is in Appendix C, which also gives the accountant for per-step resampling (Proposition C.1). That accounting is valid only when the implementation truly resamples at every step, so it does not apply to our sampler.

Output-wise subsampling is not privacy-free: if $\textstyle \sum _ { t } \mu _ { t } ^ { 2 }$ is large, the influence of an included record accumulates before amplification. This motivates our hybrid denoiser, which switches to a public neural denoiser after the empirical denoising block so that the remaining generation is post-processing. We use the Edgeworth approximation only when the composed privacy loss is in a stable finite-sample regime; Appendix D states these diagnostics.

## 5 RELATED WORK

Differentially private training has been the most direct approach to reduce memorization in diffusion models. DP training of diffusion models builds on DP-SGD (Abadi et al., 2016; Dockhorn et al., 2023; Ghalebikesabi et al., 2023; Liu et al., 2024) and protects the weights at a cost in sample quality. Private sampling has mostly been studied for Bayesian posterior inference, where the samples are model parameters rather than data (Welling & Teh, 2011; Wang et al., 2015; Li et al., 2019; Räisä et al., 2021; Zhang & Zhang, 2023). Bertazzi et al. (2025) show that privacy loss in SGLD can concentrate in the final step, but their analysis needs strongly convex, Lipschitz objectives, which our empirical denoiser does not satisfy.

Sampling-time privacy for diffusion models. Closest to our setting, Sehwag et al. (2023) also analyze privacy at generation rather than training time, clip denoiser outputs, treat each step as a Gaussian mechanism, and delegate late denoising to a public model. Our analysis differs in three respects. First, their per-step sensitivity $C / k$ arises from a sample-and-aggregate ensemble of k non-private models trained on disjoint shards, so the sensitivity floor is set by the ensemble size and private training is still required; we obtain sensitivity 2C/|D| by clipping individual contributions to a single empirical average, with no training at all. Second, they adopt a per-sample private-prediction threat model and name composition across multiple released samples as open; we give an explicit joint-release guarantee over $N _ { \mathrm { o u t } }$ outputs under standard DP, using subsampled f-DP with block-level amplification. Third, we localize privacy loss to the critical window and place the handoff at the measured crossover rather than at a fixed schedule fraction, validating the choice with a membershipinference proxy. Wei et al. (2023) study inherent privacy properties of discrete denoising diffusion and likewise find that leakage concentrates at particular stages of generation. Their mechanism is the discrete transition kernel, while ours is the continuous Gaussian reverse SDE, where the noise already injected by the sampler serves as the mechanism noise. Our results are therefore a continuous-state counterpart to their staged view, and our claims are scoped accordingly.

Empirical denoisers. Niedoba et al. (2025) characterize the empirical denoiser in the non-private setting and analyze how trained neural denoisers depart from it. We build on this object directly: our contribution is the clipped, privatized variant and its sensitivity analysis, and their non-private characterization is complementary evidence for the proxy argument in Section 3.3.

## 6 EXPERIMENTS

We study (i) full-batch versus subsampled accounting, (ii) where privacy loss accumulates relative to the critical window, and (iii) the privacy-utility trade-off of the hybrid sampler.

Implementation We implement our modified reverse denoising process in PyTorch (Paszke et al., 2019) and categorize our experiments into two regimes: (i) full-batch, where the entire dataset is used in each denoising step, and (ii) output-wise subsampled SDEdit (Meng et al., 2021), where a minibatch of private examples is subsampled once per generated output and reused for an empirical denoising block. We apply the private empirical denoising steps until time $t _ { 0 } ,$ then add Gaussian noise with scale $( \sigma _ { t _ { 1 } } ^ { 2 } - \sigma _ { t _ { 0 } } ^ { 2 } ) ^ { 1 / 2 }$ to move the sample back to an SDEdit restart level $t _ { 1 } > t _ { 0 }$ , and finally complete generation with a public neural denoiser from $t _ { 1 }$ to 0. Only the empirical subsampled block is privacy-accounted. Re-noising to a higher noise level lets the public neural denoiser correct artifacts introduced by the empirical step while incurring no additional privacy loss by post-processing.

![](images/7a89b3377f9dd95845fa8eec8bd5de54ec08cfadceb8997129baecff00f9e69c.jpg)  
ε<sub>u</sub> (certified PLD)

![](images/484490e03d70cf2c2bab80f72540828e65c1892c2291c20f2df2239215101824.jpg)  
ε<sub>u</sub> (certified PLD)

![](images/5419a472be320c16efc4e3fe471f6057390a1c463906742ca9e0c7f461550720.jpg)  
Figure 3: Privacy-utility trade-offs of the output-wise subsampled hybrid sampler on CIFAR-10. We report FID against the certified PLD upper bound $\varepsilon _ { u } ~ ( \delta = 1 0 ^ { - 5 }$ , values from Table 8) while varying (a) the handoff time $t _ { 0 } , ( \mathbf { b } )$ the number of public re-noising steps from $t _ { 0 }$ to $t _ { 1 }$ , and (c) the subsampling rate $q = B / n$

We use the EDM schedule (Karras et al., 2022) with $\sigma _ { \operatorname* { m i n } } = 0 . 0 0 2 , \sigma _ { \operatorname* { m a x } } = 8 0 , \rho = 7 , N = 5 0$ steps for full-batch runs and $N = 3 9$ for the hybrid pipeline, and the public normalization $\beta = 1$ (Appendices E and J).

Privacy Analysis Given the noise schedule and sensitivity, we calculate the corresponding $\mu { \mathrm { - G D P } }$ for each diffusion step (3). For the full-batch case, we derive the overall $\mu { \mathrm { - G D P } }$ using the n-fold composition of $\mu _ { i } – \mathbf { G D P }$ mechanisms. For the output-wise subsampled SDEdit hybrid, we compute privacy using the certified PLD accounting library (Doroshenko et al., 2022) with grid spacing $\stackrel { \cdot } { h } = \stackrel { \cdot } { 1 } 0 ^ { - 5 }$ and $\mathrm { 1 o g \_ m a s s \_ t r u n c a t i o n \_ b o u n d = - 5 0 , }$ , and the Edgeworth approximation from Section $^ { 4 , }$ which approximates the per-output privacy-loss distribution under both neighbouring data hypotheses using its first four cumulants. The subsequent re-noising and public neural denoising stages incur no additional privacy loss.

Datasets and Evaluation On CIFAR-10 (Krizhevsky et al., 2009) $( n = 5 0 , 0 0 0 )$ the public denoiser is trained on ImageNet $3 2 \times 3 2$ (Russakovsky et al., 2015), and on CelebA 64 × 64 (Liu et al., 2015) $( n = 1 6 2 , 7 7 0 )$ it is trained on FFHQ (Karras et al., 2019). We use $\delta = 1 0 ^ { - 5 } \approx 1 / \tilde { n }$ for CIFAR-10 and $1 0 ^ { - 6 }$ for CelebA-64 to match prior DP generative models, and report the stricter $\delta = 1 / n ^ { 2 }$ in Appendix K. Utility is FID against the training set with 50,000 samples, reported as mean ± standard deviation over three generation seeds where runs are repeated.

## 6.1 PRIVACY ANALYSIS OF DIFFUSION-BASED SAMPLING WITH FULL TRAINING DATA

When every record is used at every step, a single sample already costs ε ≈ 200 when denoising down to $\sigma _ { \mathrm { m i n } } = 0 . 1$ , which is needed for a recognizable sample, and at $\varepsilon \approx 1 \ : ( \sigma _ { \mathrm { m i n } } = 5 )$ the sample is indistinguishable from noise (Figure 4, Appendix I). Since each denoising step operates on the full dataset, the per-sample privacy loss is already high, and composing multiple samples results in a rapidly growing total privacy budget under GDP. Subsampling can alleviate this issue through privacy amplification, making it more suitable for large-scale sample generation.

## 6.2 HYBRID DENOISING WITH OUTPUT-WISE SUBSAMPLING

Figure 3 highlights three patterns. First, the empirical denoising stop timestep $t _ { 0 }$ controls a clear privacy-utility tradeoff: stopping later at $t _ { 0 } = 7 0 \%$ yields the best FID, while stopping earlier at $t _ { 0 } = 5 5 \%$ yields the strongest privacy guarantee. Second, the best number of public re-noising steps depends on the privacy budget. With larger private blocks (65–70%, certified $\varepsilon \geq 6 )$ three re-noising steps give the best FID, while with the smallest block $( 5 5 \% , \varepsilon \le 0 . 3 )$ more steps help (FID 45.7 with three steps and 30.4 with nine, $B = 2 5$ ). Re-noising steps are public, so they change utility but not ε.

Table 2: Comparison of DP generative models on CIFAR-10 and CelebA-64. Our method performs a short private editing block using an output-wise subsampled empirical denoiser, then hands off to a publicly pretrained neural denoiser (ImageNet for CIFAR-10, FFHQ for CelebA-64) to complete sampling. FID is computed for $N _ { o u t } = 5 0$ , 000 samples. Lower FID is better.
<table><tr><td colspan="3">CIFAR-10</td><td colspan="3">CelebA-64</td></tr><tr><td>Method</td><td>ε</td><td>FID</td><td>Method</td><td>ε</td><td>FID</td></tr><tr><td>DP-Diffusion (Ghalebikesabi et al., 2023)</td><td>10</td><td>9.8</td><td>DP-LDM (Liu et al., 2024)</td><td>10</td><td>14.3</td></tr><tr><td>DP-LDM (Liu et al., 2024)</td><td>10</td><td>8.4</td><td>DP-MEPF (Harder et al., 2023)</td><td>10</td><td>17.4</td></tr><tr><td>DP-API (Lin et al., 2023)</td><td>0.67</td><td>7.87</td><td>DP-DM (Dockhorn et al., 2023)</td><td>10</td><td>78.3</td></tr><tr><td>DP-MEPF (Harder et al., 2023)</td><td>10</td><td>29.1</td><td>DP-Promise (Wang et al., 2024)</td><td>10</td><td>25.3</td></tr><tr><td>DP-DM (Dockhorn et al., 2023)</td><td>10</td><td>97.7</td><td></td><td></td><td></td></tr><tr><td>DP-Promise (Wang et al., 2024)</td><td>10</td><td>17.9</td><td></td><td></td><td></td></tr><tr><td>Ours</td><td> $\operatorname { P L D } \left[ 1 1 . 1 4 , 1 1 . 3 8 \right]$ </td><td> $2 7 . 1 \pm 0 . 1 5$ </td><td>Ours</td><td>PLD [6.53, 6.77]  $\varepsilon _ { e w } = 5$ </td><td> $2 5 . 6 7 \pm 0 . 0 3$ </td></tr></table>

Third, decreasing the subsampling rate $q$ lowers ε, as expected from privacy amplification. In our setting it also improves FID, likely because smaller q weakens the high-variance empirical denoiser before the public neural denoiser recovers global structure after re-noising.

To contextualize the effect of our privacy mechanisms, Table 2 compares the FID (Heusel et al., 2017) of our hybrid sampler with DP generative models on CIFAR-10 and CelebA-64, and Appendix J reports all hybrid configurations.

## 6.3 PRIVACY IN RELATION TO THE CRITICAL WINDOWS

We extend our privacy analyses to quantify privacy levels based on where we stop relative to the critical window of the denoising process. Following Georgiev et al. (2023), we locate the critical window with a zero-shot CLIP classifier (Radford et al., 2021) applied along the trajectory $( \mathrm { F i g u r e } 6 ( \mathrm { a } ) )$ . The probability that the sample is classified as a ship is about 0.1 at $t = 0 . 2 7 , 0 . 3$ at $t = 0 . 0 7$ , and close to 1 at t = 0.02 (70%, 80%, and 88% of the N = 50 steps). Stopping the private block earlier gives a smaller $\mu$ but a noisier sample, and stopping later gives the reverse (Figure 7).

We choose the handoff point so that, over the private block, the empirical denoiser remains a faithful stand-in for the neural denoiser. The empirical denoiser is a window-local semantic proxy for the neural denoiser through the mid-noise window, which ends at t = 1.98, about half of the $N = 3 9$ steps (Table 1), and directional agreement stays at cos $\geq 0 . 9 8$ through about 55% of generation (Figure 1, Table 4). A 55% handoff therefore keeps the private block essentially inside the proxy regime. Our 65% handoff $( t _ { 0 } \approx 0 . 5 2 )$ still comes before the CLIP-visible critical window (≈ 70– 88%), but its private block covers most of the low-noise window, where substitution moves the endpoint more than re-drawing that window’s noise (1.64× CLIP, Table 1). The proxy interpretation is therefore weaker for the 65% block, while the certified privacy guarantee holds for both, because it is a statement about the sampler we run. Stopping earlier tightens the certified bound at a utility cost, and stopping later extends the private block into the low-noise steps that dominate the privacy cost (Appendix J).

## 6.4 PRIVACY IMPLICATIONS OF NEURAL AND EMPIRICAL DENOISERS IN MEMBERSHIP INFERENCE ATTACKS

We compare the membership-inference success rate (ASR) implied by our ε for the subsampled empirical sampler (Appendix L) with the ASR that the attack of Duan et al. (2023) achieves on a neural denoiser trained on all of CIFAR-10, stopping at the same t and using the CIFAR-10 test set as non-members (Table 13, Appendix L). The attack is weak at every t (ASR at most 0.510) and stays below the bound for $q = 0 . 5$ at every t. $\mathrm { A t } q \le 0 . 0 1$ it slightly exceeds the bound at $t = T / 2$ $( 0 . { \dot { 5 } } 0 7$ vs. at most 0.503), which is expected because the neural denoiser uses every record while the subsampled sampler includes each record with probability q.

## 7 CONCLUSION AND DISCUSSION

We give certified sampling-time privacy guarantees for continuous diffusion models by privatizing a clipped empirical denoiser. Framing each denoising step as a Gaussian mechanism, we account with GDP and f-DP, report certified (ϵ, δ) bounds from a PLD accountant, and characterize where Edgeworth approximation is and is not reliable. Our results show that leakage is highly nonuniform across timesteps, concentrating in critical windows where semantic structure emerges, while subsampling significantly amplifies privacy guarantees. These findings suggest that privacy in diffusion models is not solely determined by training but can be shaped by sampling strategies such as early stopping and hybrid use of public denoisers.

Our certificate covers the sampler we run. Samplers that query a neural denoiser trained on the private data at every step, and discrete diffusion, where the transition kernel would be the mechanism, remain open. Our sampler could also serve as a DP data-augmentation mechanism $\mathcal { A } ( \mathcal { D } )$ : a model trained only on $\mathcal A ( \mathcal D )$ and public data is $( \varepsilon , \delta ) – \mathrm { D P }$ by post-processing, with no DP training algorithm. Since privacy loss concentrates at low noise, training on A(D) for $t \geq t _ { 0 }$ and on public data for $t < t _ { 0 }$ would spend the budget only on the formative regime.

## AI USE STATEMENT

In this work, we used generative AI tools as writing, reviewing, and coding assistants. Specifically, we used them to improve grammar and word choice, to draft and revise parts of the text, to get feedback on drafts, such as consistency checks of numbers, claims, and references and suggestions for additional diagnostics. We also used them to search for and summarize related work, which we verified against the original sources, and to help write and debug code for the accounting, evaluation, and scripts, which we reviewed and tested before use. Additionally, we used AI to proofread our math equations and review our mathematical proofs. We did not use generative AI tools to generate research ideas or synthetic data, or to produce any reported number. Dataset cleaning, translation, and qualitative data analysis are not applicable to this work. We have reviewed all AI-assisted work and take full responsibility for the final content of this paper, including text, claims, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work studies privacy leakage from diffusion sampling and uses only public benchmark datasets (CIFAR-10, CelebA, ImageNet, FFHQ). Our guarantees cover the sampler in Algorithm 1 and should not be read as guarantees for other diffusion samplers (Appendix M).

## REPRODUCIBILITY STATEMENT

The code is available in Appendix A, including the PLD accounting, Edgeworth, and FID scripts behind every table and figure. Algorithm 1 specifies the sampler, Section 3 and Appendix E give the sensitivity and the public normalization, and Appendix C gives full proofs. Section 6 and Appendix G list the accountant settings (sampling model, adjacency, $q , N _ { \mathrm { o u t } } , \delta ,$ , grid spacing, truncation, and precision). Appendix J lists every hyperparameter and reports all 95 CIFAR-10 and 112 CelebA-64 configurations (Tables 8 and 9) and the repeated-seed runs.

## REFERENCES

Martin Abadi, Andy Chu, Ian J. Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, CCS ’16, pp. 308–318, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450341394. doi: 10.1145/2976749. 2978318.

Andrea Bertazzi, Tim Johnston, Gareth O. Roberts, and Alain Oliviero Durmus. Differential privacy guarantees of markov chain monte carlo algorithms. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=lLnAua5poB.

Giulio Biroli, Tony Bonnaire, Valentin de Bortoli, and Marc Mézard. Dynamical regimes of diffusion models. 2024. URL https://arxiv.org/abs/2402.18491. arXiv preprint arXiv:2402.18491.

Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In Proceedings ofthe 32nd USENIX Conference on Security Symposium, SEC ’23, USA, 2023. USENIX Association. ISBN 978-1-939133-37-3.

Tim Dockhorn, Tianshi Cao, Arash Vahdat, and Karsten Kreis. Differentially private diffusion models. Transactions on Machine Learning Research, 2023. URL https://openreview. net/forum?id=ZPpQk7FJXF.

Jinshuo Dong, Aaron Roth, and Weijie J. Su. Gaussian differential privacy. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1):3–37, 02 2022. ISSN 1369-7412. doi: 10.1111/rssb.12454. URL https://doi.org/10.1111/rssb.12454.

Vadym Doroshenko, Badih Ghazi, Pritish Kamath, Ravi Kumar, and Pasin Manurangsi. Connect the dots: Tighter discrete approximations of privacy loss distributions. arXiv preprint arXiv:2207.04380, 2022.

Jinhao Duan, Fei Kong, Shiqi Wang, Xiaoshuang Shi, and Kaidi Xu. Are diffusion models vulnerable to membership inference attacks? In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 8717–8730. PMLR, Jul 2023. URL https://proceedings.mlr.press/v202/duan23b.html.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference (TCC), pp. 265–284, 2006.

Bradley Efron and Robert J. Tibshirani. An Introduction to the Bootstrap, volume 57 of Monographs on Statistics and Applied Probability. Chapman & Hall, New York, 1993.

Kristian Georgiev, Joshua Vendrow, Hadi Salman, Sung Min Park, and Aleksander Madry. The journey, not the destination: How data guides diffusion models. In Arxiv preprint arXiv:2312.06205, 2023.

Sahra Ghalebikesabi, Leonard Berrada, Sven Gowal, Ira Ktena, Robert Stanforth, Jamie Hayes, Soham De, Samuel L. Smith, Olivia Wiles, and Borja Balle. Differentially private diffusion models generate useful synthetic images, 2023.

Sivakanth Gopi, Yin Tat Lee, and Lukas Wutschitz. Numerical composition of differential privacy. Advances in Neural Information Processing Systems, 34:11631–11642, 2021.

Frederik Harder, Milad Jalali, Danica J. Sutherland, and Mijung Park. Pre-trained perceptual features improve differentially private image generation. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=R6W7zkMz0P.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.

Hailong Hu and Jun Pang. Membership inference of diffusion models, 2023.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4401–4410, 2019.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In advances in neural information processing systems, volume 35, pp. 26565–26577, 2022.

Shiva Prasad Kasiviswanathan, Homin K. Lee, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. What can we learn privately? SIAM Journal on Computing, 40(3):793–826, 2011. doi: 10.1137/090756090. URL https://doi.org/10.1137/090756090.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. Advances in neural information processing systems, 34:21696–21707, 2021.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, Toronto, ON, Canada, 2009.

Christian Janos Lebeda, Matthew Regehr, Gautam Kamath, and Thomas Steinke. Avoiding pitfalls for privacy accounting of subsampled mechanisms under composition. In IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 996–1006, 2025. doi: 10.1109/SaTML64287. 2025.00060.

Bai Li, Changyou Chen, Hao Liu, and Lawrence Carin. On connecting stochastic gradient mcmc and differential privacy. In Kamalika Chaudhuri and Masashi Sugiyama (eds.), Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pp. 557–566. PMLR, 16–18 Apr 2019. URL https://proceedings.mlr.press/v89/li19a.html.

Marvin Li and Sitan Chen. Critical windows: non-asymptotic theory for feature emergence in diffusion models. 2024. URL https://arxiv.org/abs/2403.01633. arXiv preprint arXiv:2403.01633.

Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, Harsha Nori, and Sergey Yekhanin. Differentially private synthetic data via foundation model apis 1: Images, 2023.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Michael F Liu, Saiyue Lyu, Margarita Vinaroz, and Mijung Park. Differentially private latent diffusion models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=AkdQ266kHj.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings ofInternational Conference on Computer Vision (ICCV), December 2015.

Andrew Lowy, Zhuohang Li, Jing Liu, Toshiaki Koike-Akino, Kieran Parsons, and Ye Wang. Why does differential privacy with large ε defend against practical membership inference attacks? In AAAI Workshop on Privacy-Preserving Artificial Intelligence, Feb 2024. URL https://www. merl.com/publications/TR2024-009. MERL Technical Report TR2024-009 / arXiv preprint arXiv:2402.09540.

Tomoya Matsumoto, Takayuki Miura, and Naoto Yanai. Membership inference attacks against diffusion models. In 2023 IEEE Security and Privacy Workshops (SPW), pp. 77–83, 2023. doi: 10.1109/SPW59333.2023.00013.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

Matthew Niedoba, Berend Zwartsenberg, Kevin Murphy, and Frank Wood. Towards a mechanistic explanation of diffusion model generalization. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of PMLR, pp. 46389–46411, 2025.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Ossi Räisä, Antti Koskela, and Antti Honkela. Differentially private hamiltonian monte carlo. In NeurIPS 2021 Workshop Privacy in Machine Learning, 2021. URL https://openreview. net/forum?id=oqNqsnOVwlK.

Gabriel Raya and Luca Ambrogioni. Spontaneous symmetry breaking in generative diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal ofcomputer vision, 115:211–252, 2015.

Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. 2024. URL https://arxiv.org/abs/2402.16991. arXiv preprint arXiv:2402.16991.

Vikash Sehwag, Ashwinee Panda, Ashwini Pokle, Xinyu Tang, Saeed Mahloujifar, Mung Chiang, J Zico Kolter, and Prateek Mittal. Differentially private generation of high fidelity samples from diffusion models. In ICML 2023 Workshop on Deployable Generative AI, 2023. URL https://openreview.net/forum?id=vuVGcl0ed1.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. deep unsupervised learning using nonequilibrium thermodynamics. In proceedings ofthe 32nd international conference on machine learning, pp. 2256–2265, 2015.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Shuai Tang, Zhiwei Steven Wu, Sergul Aydore, Michael Kearns, and Aaron Roth. Membership inference attacks on diffusion models via quantile regression, 2024.

Aleksei Triastcyn and Boi Faltings. Bayesian differential privacy for machine learning. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 9583–9592. PMLR, Jul 2020. URL https://proceedings.mlr. press/v119/triastcyn20a.html.

Haichen Wang, Shuchao Pang, Zhigang Lu, Yihang Rao, Yongbin Zhou, and Minhui Xue. dppromise: Differentially private diffusion probabilistic models for image synthesis. In 33rd USENIX Security Symposium (USENIX Security 24), Philadelphia, PA, August 2024. USENIX Association. URL https://www.usenix.org/conference/usenixsecurity24/ presentation/wang-haichen.

Yu-Xiang Wang, Stephen Fienberg, and Alex Smola. Privacy for free: Posterior sampling and stochastic gradient monte carlo. In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 2493–2502, Lille, France, 07–09 Jul 2015. PMLR. URL https:// proceedings.mlr.press/v37/wangg15.html.

Yu-Xiang Wang, Borja Balle, and Shiva Prasad Kasiviswanathan. Subsampled rényi differential privacy and analytical moments accountant. In AISTATS, 2019.

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004.

Rongzhe Wei, Eleonora Kreaciˇ c, Haoyu Wang, Haoteng Yin, Eli Chien, Vamsi K Potluru, and Pan ´ Li. On the inherent privacy properties of discrete denoising diffusion models. arXiv preprint arXiv:2310.15524, 2023.

Max Welling and Yee Whye Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th International Conference on International Conference on Machine Learning, ICML’11, pp. 681–688, Madison, WI, USA, 2011. Omnipress. ISBN 9781450306195.

Yixin Wu, Ning Yu, Zheng Li, Michael Backes, and Yang Zhang. Membership inference attacks against text-to-image generation models, 2023. URL https://openreview.net/forum? id=J41IW8Z7mE.

Wanrong Zhang and Ruqi Zhang. DP-fast MH: Private, fast, and accurate Metropolis-Hastings for large-scale Bayesian inference. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 41847–41860. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/ zhang23aw.html.

Qinqing Zheng, Jinshuo Dong, Qi Long, and Weijie Su. Sharp composition bounds for gaussian differential privacy via edgeworth expansion. In International Conference on Machine Learning, pp. 11420–11435. PMLR, 2020.

Yuqing Zhu, Jinshuo Dong, and Yu-Xiang Wang. Optimal accounting of differential privacy via characteristic function. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.

## A CODE AVAILABILITY

The public code is available at https://github.com/whalebeavercat/ DPDiffusionSampling/.

## B LIPSCHITZ-BASED SENSITIVITY ANALYSIS FOR NEURAL DENOISERS

One might wonder whether considering Lipschitz-based sensitivity analysis for neural denoisers would help validate the empirical denoiser bounds. We describe our thoughts on this below.

Let $\mathcal { L } \quad : \quad X ^ { n } \to \Theta$ be the score network learning algorithm which maps a dataset $S \in$ $X ^ { n }$ to parameters $\theta \in \Theta$ . Let us define the sensitivity of the score network as $\begin{array} { r l } { \Delta _ { s } } & { { } = } \end{array}$ $\begin{array} { r } { \operatorname* { s u p } _ { x \in X } \left( \operatorname* { s u p } _ { S \in X ^ { n } , i \in [ n ] } \| s _ { \mathcal { L } ( S ) } ( x ) - s _ { \mathcal { L } ( S \backslash \{ i \} ) } ( x ) \| _ { 2 } \right) } \end{array}$ . If we could show that $\Delta _ { s }$ is bounded, then the update step in sampling can be made differentially private with appropriate noise scales as follows: $\tilde { x } _ { t } \gets \tilde { x } _ { t - 1 } + \dot { \overline { { \gamma } } } _ { i } s _ { \theta } \big ( \tilde { x } _ { t - 1 } , \tilde { \sigma } _ { i } \big ) + N ( 0 , \sigma _ { t } ^ { 2 } \Delta _ { s } ^ { 2 } I )$ . The question is how to bound the sensitivity. Does the Lipschitz network help here?

Suppose the score network is a 1-Lipschitz network, satisfying $\| s _ { \mathcal { L } ( S ) } ( x ) - s _ { \mathcal { L } ( S ) } ( x ^ { \prime } ) \| _ { 2 } \leq \| x - \relax$ $x ^ { \prime } \| _ { 2 }$ and $\| s _ { \mathcal { L } ( S \setminus \{ i \} ) } ( x ) - s _ { \mathcal { L } ( S \setminus \{ i \} ) } ( x ^ { \prime } ) \| _ { 2 } \le \| x - x ^ { \prime } \| _ { 2 }$ . Using this we can bound the sensitivity, $\| s _ { \mathcal { L } ( S ) } ( x ) - s _ { \mathcal { L } ( S \setminus \{ i \} ) } ( x ) \| _ { 2 } \le \| s _ { \mathcal { L } ( S ) } ( x ) - s _ { \mathcal { L } ( S ) } ( x ^ { \prime } ) \| _ { 2 } + \| s _ { \mathcal { L } ( S ) } ( x ^ { \prime } ) - s _ { \mathcal { L } ( S \setminus \{ i \} ) } ( x ) \| _ { 2 } \le \| x - s _ { \mathcal { L } ( S ) } ( x ) \| _ { 2 } .$ $x ^ { \prime } \| _ { 2 } + \| s _ { \mathcal { L } ( S ) } ( x ^ { \prime } ) - s _ { \mathcal { L } ( S \backslash \{ i \} ) } ( x ) \| _ { 2 }$ . So here even if $\| s _ { \mathcal { L } ( \mathcal { S } ) } ( x ^ { \prime } ) - s _ { \mathcal { L } ( S \setminus \{ i \} ) } ( x ) \| _ { 2 }$ is very small, we have ma $\operatorname { x } _ { x , x ^ { \prime } } \| x - x ^ { \prime } \| _ { 2 }$ on our upperbound.

Without having any knowledge other than all $x , x ^ { \prime }$ are pre-processed such that any of them is normbounded by $L ,$ the upperbound to this is ma $\begin{array} { r } { \mathrm { x } _ { x , x ^ { \prime } } \| x - x ^ { \prime } \| _ { 2 } \leq \operatorname* { m a x } _ { x } \| x \| _ { 2 } \leq 2 \dot { L } } \end{array}$ . In this case, the amount of noise we have to add in each denoising step is on the order of the entire input domain, and such an amount of noise will destroy the signal from the score net completely. Hence, this approach is not usable in practice.

## C PROOFS FOR SUBSAMPLING PROPOSITIONS

We prove Proposition 4.1 and Proposition C.1 (per-step subsampling, stated below) under the sampling model stated in Section 4. The proofs are directional first, comparing two neighboring datasets $\mathcal { D }$ and $\mathcal { D } ^ { \prime }$ that differ by adding or removing one record, and then symmetrized by taking both neighboring-data directions.

## C.1 PROOF OF PROPOSITION 4.1

Fix one generated output and an empirical denoising block $B ,$ and let ${ \mathcal { D } } ^ { \prime } = { \mathcal { D } } \cup \{ x \}$ , so that $q _ { B } = B / n$ with $n = | \bar { \mathcal { D } } ^ { \prime } |$ . A uniform B-subset of $\mathcal { D } ^ { \prime }$ excludes x with probability $1 - q _ { B }$ , and it is then a uniform B-subset of D, so the two neighboring datasets induce the same output distribution and the trade-off curve is Id. With probability $q _ { B }$ , it is ${ \bar { S } } ^ { \prime } \cup \{ x \}$ , where $S ^ { \prime }$ is a uniform $( B - 1 )$ )-subset of D. We pair it with $S ^ { \prime } \cup \{ y \}$ , where $y$ is uniform in $\mathcal { D } \setminus \mathcal { S } ^ { \prime }$ , which is a uniform B-subset of $\mathcal { D } .$ These two minibatches differ by one replacement, so each step has parameter $\mu _ { t }$ (Section 4), and by GDP composition the block is $\bar { \mu } _ { B } – \mathbf { G D P }$ for every such pair, where

$$
\bar { \mu } _ { B } = \Big ( \sum _ { t \in B } \mu _ { t } ^ { 2 } \Big ) ^ { 1 / 2 } .\tag{9}
$$

The hockey-stick divergence is jointly convex, so this bound survives averaging over $( S ^ { \prime } , y )$ , and the canonical binary experiment for the included-record event is $P _ { B } = \mathcal { N } ( 0 , \bar { 1 } )$ versus $\mathcal { N } ( \bar { \mu } _ { B } , 1 )$ ), with trade-off curve $G _ { \bar { \mu } s }$ . Thus the one-output directional trade-off curve is

$$
h _ { B } = q _ { B } G _ { \bar { \mu } _ { B } } + ( 1 - q _ { B } ) \operatorname { I d } .\tag{10}
$$

The $N _ { \mathrm { o u t } }$ generated outputs use independent row-subsampling events, so $f \mathrm { - D P }$ composition gives the directional composed trade-off

$$
g _ { \mathrm { o u t } } = \bigotimes _ { j = 1 } ^ { N _ { \mathrm { o u t } } } h _ { B } .\tag{11}
$$

Since differential privacy must hold for both neighboring-data directions, we symmetrize the directional curve and take the lower convex envelope, yielding

$$
f _ { \mathrm { o u t } } = \mathrm { m i n } \{ g _ { \mathrm { o u t } } , g _ { \mathrm { o u t } } ^ { - 1 } \} ^ { * * } .\tag{12}
$$

Therefore the joint release is $f _ { \mathrm { o u t } } { \mathrm { - D P } } .$ . Any public neural denoising or deterministic post-processing after the private block is post-processing of the same private release and cannot weaken the guarantee.

It remains to derive the $( \varepsilon , \delta )$ profile. In the above directional experiment, the privacy-loss random variable for one output can be written as

$$
L _ { \mathcal { B } } ( z ) = \log \left( 1 - q _ { \mathcal { B } } + q _ { \mathcal { B } } \exp \left( \bar { \mu } _ { \mathcal { B } } z - \frac { \bar { \mu } _ { \mathcal { B } } ^ { 2 } } { 2 } \right) \right) ,\tag{13}
$$

and the composed privacy loss is $\begin{array} { r } { S _ { \mathrm { o u t } } = \sum _ { j = 1 } ^ { N _ { \mathrm { o u t } } } L _ { B } ( Z _ { j } ) } \end{array}$ . The hockey-stick divergence characterization of approximate DP gives

$$
\delta _ { \mathrm { o u t } } ^ { P \to Q } ( \varepsilon ) = { \bf P r } _ { Q } ( S _ { \mathrm { o u t } } > \varepsilon ) - e ^ { \varepsilon } { \bf P r } _ { P } ( S _ { \mathrm { o u t } } > \varepsilon ) .\tag{14}
$$

Applying the same argument in the reverse neighboring direction and taking the positive part gives the stated $\delta _ { \mathrm { o u t } } ( \varepsilon )$ .

## C.2 PER-STEP SUBSAMPLING

Proposition C.1 (Per-step subsampling). Consider instead a sampler that draws an independent minibatch of fixed size $B _ { t }$ at every private denoising step. For step t, let $q _ { t } = B _ { t } / n$ be the inclusion probability and $\mu _ { t }$ the Gaussian parameter conditional on inclusion, and set $h _ { t } = { \dot { q } } _ { t } G _ { \mu _ { t } } + ( 1 - q _ { t } )$ Id. For $N _ { \mathrm { o u t } }$ outputs and T private steps, define

$$
g _ { \mathrm { s t e p } } = \bigotimes _ { j = 1 } ^ { N _ { \mathrm { o u t } } } \bigotimes _ { t = 1 } ^ { T } h _ { t } , \qquad f _ { \mathrm { s t e p } } = \mathrm { m i n } \{ g _ { \mathrm { s t e p } } , g _ { \mathrm { s t e p } } ^ { - 1 } \} ^ { * * } .\tag{15}
$$

Then the joint release is $f _ { \mathrm { s t e p } } { - } D P .$ Hence, for every $\varepsilon \geq 0$ it is $( \varepsilon , \delta _ { \mathrm { s t e p } } ( \varepsilon ) )$ -DP, where $\delta _ { \mathrm { s t e p } } ( \varepsilon ) =$ I $\begin{array} { r } { \mathrm { n a x } _ { \mathrm { d i r } } [ P r _ { Q } ( S _ { \mathrm { s t e p } } ^ { \mathrm { d i r } } \ > \ \varepsilon ) - e ^ { \varepsilon } P r _ { P } ( S _ { \mathrm { s t e p } } ^ { \mathrm { d i r } } \ > \ \varepsilon ) ] _ { + } , \ S _ { \mathrm { s t e p } } = \ \sum _ { j = 1 } ^ { N _ { \mathrm { o u t } } } \sum _ { t = 1 } ^ { T } L _ { t } ( Z _ { j , t } ) , } \end{array}$ , and $L _ { t } ( z ) =$ $\log ( 1 - q _ { t } + q _ { t } \exp ( \mu _ { t } z - \mu _ { t } ^ { 2 } / 2 ) )$

The two propositions describe different privacy-loss distributions. In general, $\begin{array} { r l } { S _ { \mathrm { s t e p } } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { j , t } L _ { t } ( \dot { Z _ { j , t } } ) \dot { \neq } \sum _ { j } L _ { \mathcal { B } } ( Z _ { j } ) = S _ { \mathrm { o u t } } , \mathrm { s o } f _ { \mathrm { s t e p } } \dot { \neq } f _ { \mathrm { o u t } } } \end{array}$ . Per-step amplification is valid only when the implementation truly resamples at every step. For our output-wise subsampled sampler, Proposition 4.1 is the correct accountant: amplification is applied once to the whole empirical block.

## C.3 PROOF OF PROPOSITION C.1

For per-step subsampling, the differing record is included at step t with probability $q _ { t }$ , independently across steps. As in Appendix C.1, each minibatch of $\mathcal { D } ^ { \prime }$ containing x is paired with a minibatch of D that differs from it in one record, now with $B _ { t } , q _ { t }$ and $\mu _ { t }$ in place of $B , q _ { B }$ and $\bar { \mu } _ { B }$ . Conditional on inclusion at step $t ,$ the step is a Gaussian mechanism with GDP parameter $\mu _ { t } ,$ , so the canonical included-record trade-off curve is $G _ { \mu _ { t } }$ . If the record is not included, the two neighboring datasets induce the same distribution for that step and the trade-off curve is Id. Therefore the one-step directional trade-off curve is

$$
h _ { t } = q _ { t } G _ { \mu _ { t } } + ( 1 - q _ { t } ) \operatorname { I d } .\tag{16}
$$

Because minibatches are redrawn independently across timesteps and generated outputs, the directional trade-off curve for $N _ { \mathrm { o u t } }$ outputs and T private steps is the tensor-product composition

$$
g _ { \mathrm { s t e p } } = \bigotimes _ { j = 1 } ^ { N _ { \mathrm { o u t } } } \bigotimes _ { t = 1 } ^ { T } h _ { t } .\tag{17}
$$

Symmetrizing over the two neighboring-data directions and convexifying gives

$$
f _ { \mathrm { s t e p } } = \mathrm { m i n } \{ g _ { \mathrm { s t e p } } , g _ { \mathrm { s t e p } } ^ { - 1 } \} ^ { * * } ,\tag{18}
$$

so the joint release is $f _ { \mathrm { s t e p } } { \mathrm { - D P } } .$

The corresponding likelihood-ratio representation is obtained by summing the independent per-step privacy losses. For one step,

$$
L _ { t } ( z ) = \log \left( 1 - q _ { t } + q _ { t } \exp \left( \mu _ { t } z - \frac { \mu _ { t } ^ { 2 } } { 2 } \right) \right) ,\tag{19}
$$

and hence

$$
S _ { \mathrm { s t e p } } = \sum _ { j = 1 } ^ { N _ { \mathrm { o u t } } } \sum _ { t = 1 } ^ { T } L _ { t } ( Z _ { j , t } ) .\tag{20}
$$

Using the same hockey-stick divergence conversion as above gives

$$
\delta _ { \mathrm { s t e p } } ( \varepsilon ) = \operatorname* { m a x } _ { \mathrm { d i r } } \left[ \operatorname* { P r } _ { Q } ( S _ { \mathrm { s t e p } } ^ { \mathrm { d i r } } > \varepsilon ) - e ^ { \varepsilon } \operatorname* { P r } _ { P } ( S _ { \mathrm { s t e p } } ^ { \mathrm { d i r } } > \varepsilon ) \right] _ { + } ,\tag{21}
$$

which is the claimed $( \varepsilon , \delta )$ guarantee.

## D DIAGNOSTICS FOR EDGEWORTH APPROXIMATION

The standard Edgeworth approximation for subsampled Gaussian mechanisms is often stated for the setting where a fresh subsample is drawn at every private step. Our output-wise subsampling mechanism uses the same mixture privacy-loss model, but after grouping timesteps by independent subsampling events. Thus each “step” in the diagnostic below should be read as a block B with sampling probability $q _ { B }$ and Gaussian parameter

$$
\bar { \mu } _ { B } = \left( \sum _ { t \in \mathcal { B } } \mu _ { t } ^ { 2 } \right) ^ { 1 / 2 } .\tag{22}
$$

For a block, recall the privacy-loss variable

$$
L _ { \mathcal { B } } ( z ) = \log \left( 1 - q s + q s \exp \left( \bar { \mu } _ { \mathcal { B } } z - \frac { \bar { \mu } _ { \mathcal { B } } ^ { 2 } } { 2 } \right) \right) ,\tag{23}
$$

evaluated under both $P _ { B } = \mathcal { N } ( 0 , 1 )$ and $Q _ { B } = ( 1 - q _ { B } ) \mathcal { N } ( 0 , 1 ) + q _ { B } \mathcal { N } ( \bar { \mu } _ { B } , 1 )$ . Let $\kappa _ { r , B } ^ { D }$ denote the r-th cumulant of $L _ { B }$ under $D \in \{ P , Q \}$ . Across generated samples and independent blocks, cumulants add:

$$
M _ { D } = \sum _ { \mathcal { B } } \kappa _ { 1 , \mathcal { B } } ^ { D } , \qquad V _ { D } = \sum _ { \mathcal { B } } \kappa _ { 2 , \mathcal { B } } ^ { D } , \qquad K _ { 3 , D } = \sum _ { \mathcal { B } } \kappa _ { 3 , \mathcal { B } } ^ { D } , \qquad K _ { 4 , D } = \sum _ { \mathcal { B } } \kappa _ { 4 , \mathcal { B } } ^ { D } .\tag{24}
$$

Edgeworth approximation. Let S denote a composed privacy-loss variable with mean M, variance V, third cumulant $K _ { 3 }$ , and fourth cumulant $K _ { 4 }$ (that is, $M _ { D } , \mathrm { \tilde { \it { V } } } _ { D } , K _ { 3 , D } , K _ { 4 , D }$ under $D \in \{ P , Q \} )$ . The degree-two Edgeworth approximation to the CDF of $( S - M ) / \sqrt { V }$ is

$$
F _ { \mathrm { E W } } ( z ) = \Phi ( z ) - \varphi ( z ) \left[ \frac { K _ { 3 } } { 6 V ^ { 3 / 2 } } ( z ^ { 2 } - 1 ) + \frac { K _ { 4 } } { 2 4 V ^ { 2 } } ( z ^ { 3 } - 3 z ) + \frac { K _ { 3 } ^ { 2 } } { 7 2 V ^ { 3 } } ( z ^ { 5 } - 1 0 z ^ { 3 } + 1 5 z ) \right] ,\tag{25}
$$

where Φ and $\varphi$ are the standard normal CDF and density. We apply this approximation separately under the two neighboring-data hypotheses, since the two privacy-loss CDFs need not match.

What the cumulants represent. For a random variable X, the cumulant generating function is

$$
K _ { X } ( s ) = \log \mathbb { E } [ e ^ { s X } ] ,\tag{26}
$$

and the r-th cumulant is the r-th derivative of $K _ { X }$ at $s = 0$ . In our accountant, $X = L _ { B }$ under either $P$ or $Q$ . The first four cumulants have the familiar interpretations

$$
\kappa _ { 1 } = \mathbb { E } [ X ] ,
$$

$$
\kappa _ { 2 } = \operatorname { V a r } ( X ) ,\tag{27}
$$

(28)

$$
\kappa _ { 3 } = \mathbb { E } [ ( X - \mathbb { E } X ) ^ { 3 } ] ,\tag{29}
$$

$$
\kappa _ { 4 } = \mathbb { E } [ ( X - \mathbb { E } X ) ^ { 4 } ] - 3 \mathrm { V a r } ( X ) ^ { 2 } .\tag{30}
$$

Thus $\kappa _ { 1 }$ and $\kappa _ { 2 }$ give the mean and variance of the privacy loss, while $\kappa _ { 3 }$ and $\kappa _ { 4 }$ quantify the leading skewness and kurtosis corrections to the Gaussian approximation.

No lower bound on the sampling rate alone. A small sampling probability $q _ { B }$ does not by itself invalidate the Edgeworth approximation. For fixed finite ${ \bar { \mu } } _ { B } .$ , the variance and higher cumulants shrink with powers of $q _ { B } ,$ , and these powers largely cancel after standardization. The relevant question is not whether $q _ { B }$ is large, but whether the composed privacy-loss sum has enough effective variance and is not dominated by a small number of blocks.

Variance and dominance checks. First, $V _ { P }$ and $V _ { Q }$ should be positive and numerically stable. If either variance is extremely small, the mechanism may be close to perfect privacy, but the standardized Edgeworth CDF and inverse-CDF calculations become ill-conditioned. Second, no block should dominate the variance:

$$
\operatorname* { m a x } _ { B } \frac { \kappa _ { 2 , B } ^ { P } } { V _ { P } } \ll 1 , \qquad \operatorname* { m a x } _ { B } \frac { \kappa _ { 2 , B } ^ { Q } } { V _ { Q } } \ll 1 .\tag{31}
$$

This condition is particularly important for diffusion schedules, where late low-noise timesteps can dominate $\textstyle \sum _ { t } \mu _ { t } ^ { 2 }$ if they are grouped into a large private block.

Standardized cumulants. The Edgeworth correction should be small at the quantiles used for the reported $( \varepsilon , \delta )$ guarantee. We monitor the standardized third and fourth cumulants

$$
{ \rho _ { 3 , P } } = { \frac { { \vert { K _ { 3 , P } } \vert } } { { V _ { P } ^ { 3 / 2 } } } } , \qquad { \rho _ { 4 , P } } = \frac { { \vert { K _ { 4 , P } } \vert } } { { V _ { P } ^ { 2 } } } , \qquad { \rho _ { 3 , Q } } = \frac { { \vert { K _ { 3 , Q } } \vert } } { { V _ { Q } ^ { 3 / 2 } } } , \qquad { \rho _ { 4 , Q } } = \frac { { \vert { K _ { 4 , Q } } \vert } } { { V _ { Q } ^ { 2 } } } .\tag{32}
$$

Small values near the center of the distribution are not sufficient by themselves: degree-two Edgeworth terms multiply these quantities by polynomials in the standardized quantile z, so far-tail evaluations can still be unreliable.

Tail and CDF checks. Very small $\delta$ values may require evaluating the approximation close to the boundary of the CDF, where finite-sample Edgeworth approximations are less stable. We therefore check that the approximated CDFs are monotone, remain in [0, 1] before substantial clipping or projection, and produce a nonnegative $\delta ( \varepsilon )$ curve in the reported range. We compute both neighboringdata directions and use the larger value. The Edgeworth value is never a certified bound. These diagnostics only decide whether the estimate is stable enough to show next to the certified PLD value, and configurations that fail them are marked invalid (Table 9).

Practical implications of output-wise subsampling. Suppose the dataset has size $n = | \mathcal { D } |$ and each generated sample draws a minibatch of size $B$ under the sampling model of Section 4, reused for an empirical denoising block. For a fixed training record, the block-level inclusion probability is

$$
q _ { B } = B / n .\tag{33}
$$

For our CIFAR-10 setting with $n = 5 0 { , } 0 0 0$ and $B = 2 5$ , this gives $q _ { B } = 5 \times 1 0 ^ { - 4 }$ . If we release $N _ { \mathrm { o u t } } = 5 0 \small { , } 0 0 0$ generated samples, then a fixed record is included in about

$$
N _ { \mathrm { o u t } } q _ { B } = 2 5\tag{34}
$$

generated trajectories on average. The privacy-relevant point is that each inclusion can affect the entire empirical block. Therefore the output-wise subsampled accountant is not

$$
N _ { \mathrm { o u t } } T \ \mathrm { s u b s a m p l e d \ G a u s s i a n \ m e c h a n i s m s \ w i t h \ p a r a m e t e r s } \ ( q _ { t } , \mu _ { t } ) ,\tag{35}
$$

which would correspond to independently resampling at every denoising step. It is instead

$$
N _ { \mathrm { o u t ~ } } { \mathrm { s u b s a m p l e d ~ b l o c k ~ m e c h a n i s m s ~ w i t h ~ p a r a m e t e r s ~ } } \left( q _ { B } , \left( \sum _ { t \in B } \mu _ { t } ^ { 2 } \right) ^ { 1 / 2 } \right)\tag{36}
$$

This is the practical reason to subsample per output once and keep the empirical block short: outputwise subsampling sharply limits how often any record is accessed, but once a record is selected its influence accumulates across all empirical denoising steps in that block.

## E EFFECT OF THE PUBLIC NORMALIZATION CONSTANT $\beta$

For EDM, the forward transition kernel is $p ( x _ { t } \mid x _ { 0 } ) = \mathcal { N } ( x _ { t } ; x _ { 0 } , t ^ { 2 } I )$ , so the marginal at each time is $\begin{array} { r } { \widehat { p } _ { t } ( x _ { t } ) = \frac { 1 } { | \mathcal { D } | } \sum _ { x _ { 0 } \in \mathcal { D } } p ( x _ { t } \mid x _ { 0 } ) } \end{array}$ . The exact empirical posterior mean divides the weighted average by this data-dependent normalizer:

$$
\widehat { E } \big [ x _ { 0 } \mid x _ { t } \big ] = \frac { 1 } { \widehat { p _ { t } } ( x _ { t } ) \mid \mathcal { D } \mid } \sum _ { x _ { 0 } \in \mathcal { D } } x _ { 0 } p ( x _ { t } \mid x _ { 0 } ) .\tag{37}
$$

Privately estimating and using this normalizer would require accounting for an additional datadependent statistic. Our implemented mechanism instead fixes a public constant (hyperparameter) $\beta$ and uses

$$
\widehat { E } _ { \beta } [ x _ { 0 } \mid x _ { t } ] = \frac { 1 } { \beta \left| \mathscr { D } \right| } \sum _ { x _ { 0 } \in \mathscr { D } } x _ { 0 } p ( x _ { t } \mid x _ { 0 } ) .\tag{38}
$$

Consequently, $\beta = 1$ defines a deliberately unnormalized empirical mechanism; it is not an estimate that the density $\widehat { p } _ { t } ( x _ { t } )$ numerically equals one. With public $\beta ,$ this remains a well-defined randomized mechanism with a rescaled empirical drift, and the privacy guarantee applies exactly to the implemented mechanism. Its privacy and utility both depend on the public scaling through

$$
\mu _ { t } ( \beta ) = \frac { 2 C } { \beta \left| \mathcal { D } \right| } \sqrt { \frac { 2 \Delta t } { t ^ { 3 } } } ,\tag{39}
$$

Algorithm 1 Output-wise subsampled empirical denoising sampler   
Require: D with $n = | \mathcal { D } | .$ , subsample size B, clip norm $C ,$ EDM schedule with switch time $t _ { 0 }$ and   
restart level $t _ { 1 } > t _ { 0 } .$ , public denoiser $D _ { \mathrm { p u b } }$ (data disjoint from $\mathcal { D } )$ , outputs $N _ { \mathrm { o u t } }$   
Ensure: $N _ { \mathrm { o u t } }$ samples with an $f _ { \mathrm { o u t } } – \mathrm { D P }$ guarantee on $\mathcal { D } .$   
1: $q _ { B }  B / n ; \bar { B }  \{ t : t _ { 0 } < t \leq t _ { \operatorname* { m a x } } \}$ ▷ block inclusion prob.; private block   
2: for $j = \dot { 1 }$ to $N _ { \mathrm { o u t } }$ do   
3: draw $S _ { j } \subseteq { \mathcal { D } } , | S _ { j } | = B ,$ , and reuse for all of B; $x \gets \sigma _ { \mathrm { m a x } } \epsilon$ ▷ subsample once per output   
4: for $t \in B$ in decreasing order do   
5: $\begin{array} { r } { \hat { \mathbb { E } }  \frac { 1 } { \beta | S _ { j } | } \sum _ { x _ { 0 } \in S _ { j } } \bar { v } ( x _ { 0 } ; x ) , \quad \bar { v } = v / \operatorname* { m a x } ( 1 , \| v \| / C ) , \quad v = x _ { 0 } p _ { t } ( x \mid x _ { 0 } ) } \end{array}$ ▷ clipped   
direction   
6: $\begin{array} { r } { x \gets \big ( 1 - \frac { 2 \Delta t } { t } \big ) x + \frac { 2 \Delta t } { t } \hat { \mathbb { E } } + \sqrt { 2 t \Delta t } \epsilon } \end{array}$ ▷ Gaussian mechanism, $\mu _ { t }$ by eq. 6   
7: end for   
8: $x  x + ( \sigma _ { t _ { 1 } } ^ { 2 } - \sigma _ { t _ { 0 } } ^ { 2 } ) ^ { 1 / 2 } \epsilon ; ~ x ^ { ( j ) }  D _ { \mathrm { p u b } }$ integrated from $t _ { 1 }$ to 0 ▷ re-noise; public   
refinement (post-processing)   
9: end for   
10: $\begin{array} { r } { \bar { \mu } _ { B }  ( \sum _ { t \in B } \mu _ { t } ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ ; amplify once by q<sub>B</sub> and compose over $N _ { \mathrm { o u t } }$ outputs (Prop. 4.1) to obtain   
f<sub>out</sub>.   
11: return $\{ x ^ { ( j ) } \} _ { j = 1 } ^ { N _ { \mathrm { o u t } } }$ with certificate $f _ { \mathrm { o u t } }$

where larger $\beta$ reduces both the private drift and $\mu _ { t } .$ , while smaller $\beta$ strengthens both.

Table 3 reports $\beta$ directly (the experimental code uses the inverse, $1 / \beta )$ . The setup is CIFAR-10, the 65% empirical block, batch size 50, three-step SDEdit restart, $\delta = \mathrm { \dot { 1 } 0 ^ { - 5 } }$ , and $N _ { \mathrm { o u t } } = 5 0 \small { , } 0 0 0$

Table 3: Effect of the public normalization $\beta$ on privacy and utility.
<table><tr><td> $\beta$ </td><td>Edgeworth estimate</td><td>Certified PLD upper bound</td><td>FID</td></tr><tr><td>4</td><td>0.438</td><td>0.438</td><td>132.89</td></tr><tr><td>2</td><td>1.218</td><td>1.217</td><td>62.81</td></tr><tr><td>1</td><td>10.777</td><td>11.381</td><td>27.13</td></tr><tr><td>0.5</td><td>256.624</td><td>259.761</td><td>53.54</td></tr><tr><td>0.25</td><td>unstable</td><td>unstable</td><td>81.43</td></tr></table>

This ablation shows that the public normalization is a genuine privacy–utility parameter rather than a privacy-neutral approximation: a weak empirical drift gives strong privacy but poor utility, whereas an overly strong drift worsens the certified privacy value and destabilizes generation.

## F PER-STEP MAGNITUDE GAP BETWEEN EMPIRICAL AND NEURAL DENOISERS

The relevant diagnostic in the main text is Figure 1 (directional cosine similarity), which we supplement here with the mean-squared error (MSE) between the neural and empirical denoiser outputs along the trajectory.

Cosine and MSE measure different quantities: MSE is $\| \widehat { \mathbb { E } } - D \| ^ { 2 } .$ , while Figure 1 reports the cosine between the residual directions $\boldsymbol { r } _ { E } = \boldsymbol { x } _ { t } - \widehat { \mathbb { E } }$ and $\boldsymbol { r } _ { N } = \boldsymbol { x } _ { t } - \boldsymbol { D }$ , related by $\| \widehat { \mathbb { E } } - D \| ^ { 2 }$ = $\| r _ { E } \| ^ { 2 } + \| r _ { N } \| ^ { 2 } - 2 \| r _ { E } \| \| r _ { N } \| \cos ( r _ { E } , r _ { N } )$ . The cosine can be near one while the magnitudes differ (large MSE), and at low noise both residual norms shrink, making the cosine numerically sensitive. Cosine alone therefore cannot establish closeness; we report MSE alongside it and narrow the proxy claim to directional agreement.

Absolute MSE is not comparable across rows: both residual norms shrink as $t  0 ,$ so the small values at low noise reflect this scale collapse rather than closer agreement, and the only cross-row claim we make is the directional one.

The magnitude gap is largest in the mid-noise window (the 46–55% rows), yet Table 1 shows that the difference surviving to the released sample there stays below the noise-resampling reference: the

Table 4: Directional cosine similarity (Fig. 1) and per-step $\mathbf { M S E } \parallel \widehat { \mathbb { E } } - D \parallel ^ { 2 }$ between empirical and neural denoiser outputs, along the sampling trajectory. Rows are labelled by the denoiser active at that noise level under the hybrid sampler, and switch marks the empirical→public handoff for the $5 5 / 6 5 / 7 0 \%$ configurations. Rows use the $N = 5 0$ EDM grid $( \mathrm { p e r c e n t } = \mathrm { s t e p } / N )$ . On the $N = 3 9$ grid of the hybrid runs these handoffs fall near $t = 1 . 5 , \bar { 0 . 5 2 } .$ , and 0.28. MSE is mean ± standard error over 3 seeds.
<table><tr><td>t</td><td>% gen.</td><td>log SNR</td><td> $\cos ( x _ { t } - { \widehat { \mathbb { E } } } , x _ { t } - D )$ </td><td>MSE</td><td>denoiser</td></tr><tr><td>80.0</td><td>0%</td><td>-10.2</td><td>1.0000</td><td> $3 . 2 4 \pm 0 . 0 3$ </td><td>emp.</td></tr><tr><td>10.2</td><td>32%</td><td>-6.0</td><td>1.0000</td><td> $1 . 8 4 \pm 0 . 0 7$ </td><td>emp.</td></tr><tr><td>3.29</td><td>46%</td><td>-3.8</td><td>0.9965</td><td> $1 4 . 6 3 \pm 0 . 2 0$ </td><td>emp.</td></tr><tr><td>1.29</td><td>55%</td><td>-1.9</td><td>0.9895</td><td> $1 0 . 1 3 \pm 0 . 0 9$ </td><td>switch 55%</td></tr><tr><td>0.55</td><td>65%</td><td>-0.2</td><td>0.9766</td><td> $6 . 4 0 \pm 0 . 0 5$ </td><td>switch 65%</td></tr><tr><td>0.27</td><td>70%</td><td>+1.3</td><td>0.9568</td><td> $4 . 2 3 \pm 0 . 0 4$ </td><td>switch 70%</td></tr><tr><td>0.09</td><td>78%</td><td>+3.4</td><td>0.9011</td><td> $2 . 1 3 \pm 0 . 0 2$ </td><td>pub.</td></tr><tr><td>0.02</td><td>88%</td><td>+6.7</td><td>0.7517</td><td> $0 . 6 2 \pm 0 . 0 0 4$ </td><td>pub.</td></tr><tr><td>0.002</td><td>98%</td><td>+11.0</td><td>0.3901</td><td> $0 . 1 0 \pm 0 . 0 0 0 3$ </td><td>pub.</td></tr></table>

private block sets the denoising direction, and the subsequent re-noising and public refinement (Sec. 6) repair magnitude at no additional privacy cost. Table 5 gives the absolute endpoint agreement behind Table 1, and Table 6 gives image-space agreement for both the substituted and the noise-resampled paths.

Table 5: Endpoint agreement between the all-neural reference and the path with the empirical denoiser substituted in one window (2048 paired trajectories, $N = 3 9 )$ . CLIP and DINO are cosine similarities (1− the swap distance of Table 1), with 95% bootstrap CIs in brackets. ∆CLIP is noise-control distance minus swap distance, with 95% paired-bootstrap CIs. Image-space metrics are in Table 6.
<table><tr><td>Window</td><td>t range</td><td>CLIP</td><td>DINO</td><td>∆CLIP [95% CI]</td></tr><tr><td>high-noise</td><td>19.15-7.47</td><td>0.990 [0.989, 0.991]</td><td>0.928 [0.922, 0.934]</td><td>+0.0821  $\left( + 0 . 0 8 0 1 , + 0 . 0 8 4 0 \right]$ </td></tr><tr><td>mid-noise</td><td>6.09-1.98</td><td>0.923 [0.921, 0.925]</td><td>0.483 [0.474, 0.491]</td><td>+0.0228 [+0.0207, +0.0248]</td></tr><tr><td>low-noise</td><td>1.54–0.38</td><td>0.887 [0.886, 0.889]</td><td>0.309 [0.303, 0.315]</td><td>-0.0441 [-0.0459, -0.0422]</td></tr><tr><td>min-noise</td><td> $0 . 1 4 \substack { - 0 . 0 1 8 }$ </td><td>0.785 [0.783, 0.788]</td><td>0.124 [0.119, 0.129]</td><td>-0.1906 [-0.1930, -0.1881]</td></tr><tr><td>all steps</td><td>80-0.002</td><td>0.794 [0.792, 0.796]</td><td>0.121 [0.117, 0.126]</td><td></td></tr></table>

Table 6: Image-space agreement between the all-neural endpoint and each variant, from the same starting noise (2048 trajectories, $N = 3 9$ , windows of 6 steps). swap: empirical denoiser substituted inside the window. ref.: neural denoiser kept and only that window’s noise re-drawn. Higher means closer to the all-neural endpoint for all three metrics. Brackets are 95% paired bootstrap CIs.
<table><tr><td>Window (steps)</td><td>PSNR (dB)</td><td>SSIM</td><td>Label agreement</td></tr><tr><td>high-noise (9–14)</td><td>swap 33.19 [32.88, 33.50] ref. 11.77 [11.67, 11.86]</td><td>0.981 [0.979, 0.983] 0.634 [0.629, 0.639]</td><td>0.867 [0.853, 0.882] 0.412 [0.391, 0.433]</td></tr><tr><td>mid-noise (15–20)</td><td>swap 14.74 [14.63, 14.84] ref. 10.40 [10.36, 10.45]</td><td>0.731 [0.726, 0.736] 0.466 [0.463, 0.469]</td><td>0.462 [0.440, 0.483] 0.390 [0.368, 0.411]</td></tr><tr><td>low-noise (21–26)</td><td>swap 12.23 [12.19, 12.27] ref. 11.98 [11.94, 12.03]</td><td>0.412 [0.410, 0.415] 0.415 [0.412, 0.419]</td><td>0.255 [0.236, 0.274] 0.603 [0.582, 0.624]</td></tr><tr><td>min-noise (29–34)</td><td>swap 11.39 [11.36, 11.42] ref. 22.69 [22.67, 22.72]</td><td>0.115 [0.112, 0.118] 0.917 [0.916, 0.918]</td><td>0.152 [0.136, 0.167] 0.793 [0.776, 0.811]</td></tr><tr><td>all steps (0–38)</td><td>swap 10.19 [10.14, 10.23]</td><td>0.087 [0.085, 0.089]</td><td>0.171 [0.155, 0.188]</td></tr></table>

## G COMPARISON BETWEEN PLD ACCOUNTANT AND EDGEWORTH APPROXIMATION

## G.1 ACCOUNTING PROCEDURE

Shared steps. (1) Compute $\mu _ { t }$ for every private step and $\textstyle { \bar { \mu } } B = ( \sum _ { t \in B } \mu _ { t } ^ { 2 } ) ^ { 1 / 2 }$ (Table 7 lists $\bar { \mu } _ { B }$ for each handoff). (2) Form the one-output pair $P _ { \mathcal B } = \mathcal N ( 0 , 1 ) , Q _ { \mathcal B } = ( \widetilde 1 - q _ { \mathcal B } ) \mathcal N ( 0 , 1 ) + q _ { \mathcal B } \mathcal N ( \bar { \mu } _ { \mathcal B } , 1 )$ and its privacy loss $L _ { B }$ (Proposition $4 . 1 )$ , in both directions $( Q _ { B }$ against $P _ { B }$ and $P _ { B }$ against $Q _ { B } )$ For a pair $( A , A ^ { \prime } )$ with privacy loss $\ell = \log ( \mathrm { d } A / \mathrm { d } A ^ { \prime } )$ , the hockey-stick divergence is $H _ { e ^ { \varepsilon } } ( A \| A ^ { \prime } ) =$ $\mathbb { E } _ { A } [ ( \bar { 1 } - e ^ { \dot { \varepsilon } - \ell } ) _ { + } ]$ , so $\delta ( \varepsilon )$ depends on the pair only through the distribution of ℓ under A, its PLD.

PLD accountant (certified, reported as $\varepsilon _ { u } )$ . (3) For each direction, discretize the PLD of one output pessimistically on a grid of width $h = 1 0 ^ { - 5 }$ , so that the hockey-stick divergence of the discrete PLD upper-bounds the true one at every ε. Tail mass below $e ^ { - 5 0 }$ is truncated and added to δ. (4) Compose the $N _ { \mathrm { o u t } }$ independent outputs by convolving the discretized PLD with itself $N _ { \mathrm { o u t } }$ times, computed with the FFT. Any mass truncated during composition is also added to δ. The result is a pessimistic PLD of $S _ { \mathrm { o u t } } . ~ ( 5 )$ For each direction evaluate $\begin{array} { r } { \dot { \delta } ( \varepsilon ) = \sum _ { s } \omega ( s ) ( 1 - e ^ { \varepsilon - s } ) _ { + } } \end{array}$ <sub>+</sub> plus the truncated mass, where ω is the composed discrete PLD, take the larger of the two directions, and find the smallest ε with $\delta _ { \mathrm { o u t } } ( \varepsilon ) \leq \delta$ . This is $\varepsilon _ { u } .$ . (6) Repeating (3)–(5) with every value rounded down gives $\varepsilon _ { \ell } ,$ , which is not a bound and only shows how much of $\varepsilon _ { u }$ is discretization slack. We use the implementation of Doroshenko et al. (2022) (Google dp\_accounting) to automatically calculate these values given $\bar { \mu } _ { B }$ and subsampling rate.

Edgeworth approximation (estimate). (3<sup>′</sup>) For $D \in \{ P , Q \}$ , compute the first four cumulants $\kappa _ { r , B } ^ { D }$ of $L _ { B }$ under $D _ { B }$ by numerical integration over z. (4<sup>′</sup>) Cumulants add over independent outputs, so $S _ { \mathrm { o u t } }$ has $M _ { D } = N _ { \mathrm { o u t } } \kappa _ { 1 , B } ^ { D } , V _ { D } = N _ { \mathrm { o u t } } \kappa _ { 2 , B } ^ { D } , K _ { 3 , D } = N _ { \mathrm { o u t } } \kappa _ { 3 , B } ^ { D } .$ , and $K _ { 4 , D } = N _ { \mathrm { o u t } } \kappa _ { 4 , B } ^ { D }$ (Appendix D). (5<sup>′</sup>) Approximate Pr<sub>D</sub> $\left( S _ { \mathrm { o u t } } > \varepsilon \right)$ ≈ $1 - F _ { \mathrm { E W } } \big ( ( \varepsilon - M _ { D } ) / \sqrt { V _ { D } } \big )$ with eq. 25, insert both probabilities into $\delta _ { \mathrm { o u t } } ( \varepsilon )$ of Proposition 4.1 for each direction, take the larger, and solve $\delta _ { \mathrm { o u t } } ( \bar { \varepsilon } ) = \delta$ by bisection. (6<sup>′</sup>) Check the diagnostics of Appendix D. The value is an estimate: the truncated series has an error of unknown sign and there is no discretization or truncation control. The CLT column of Table 11 instead replaces the composed trade-off curve by a single Gaussian curve $G _ { \mu } ,$ , with $\mu$ from the summed first two moments of the per-output privacy loss (Dong et al., 2022).

## G.2 COMPARISON

The degree-two Edgeworth approximation used in the paper is a numerical approximation, not a one-sided certified DP upper bound. Although Propositions 4.1 and C.1 exactly characterize the relevant block/per-step privacy-loss distributions, replacing their CDFs by a truncated Edgeworth expansion introduces an approximation error whose sign is not controlled in our finite heterogeneous setting.

To provide a formal guarantee, we additionally composed the mechanism actually implemented— within a private block, conditional per-step Gaussian privacy accumulates to $\bar { \mu } _ { B } = ( \bar { \sum } _ { t \in B } \mu _ { t } ^ { 2 } ) ^ { 1 / 2 }$ subsampling amplification is applied once to the block, and the result is composed over $N _ { \mathrm { o u t } }$ released samples—using PLD (Doroshenko et al., 2022) and RDP (Wang et al., 2019) accountants. All accountants use the sampling model of Section 4 $( q = B / n$ , add/remove adjacency), block parameter ${ \bar { \mu } } _ { B } .$ , and the same output count. RDP is a conservative certified relaxation; the PLD method provides a rigorous upper bound and a lower bound that bracket the optimal privacy value after controlling discretization and truncation error. At $\delta = 1 0 ^ { - 5 } , N _ { \mathrm { o u t } } = 5 0 , \bar { 0 } 0 0 , B \bar { = } 2 5$ for CIFAR-10, PLD grid spacing $h = 1 0 ^ { - 5 }$ and log\_mass\_truncation\_bound = −50:

The comparison reveals two regimes in these evaluated configurations. For $\varepsilon \lesssim 2 ,$ the Edgeworth estimate lies within 1% of the certified PLD upper bound (55% and 62% blocks). For $\varepsilon \gtrsim 5$ , it is anti-conservative: at the 65% and 67% blocks, the Edgeworth estimate falls below even the optimistic PLD lower estimate (5.33 vs. [6.05, 6.29] and 22.98 vs. [25.49, 25.70]), understating the certified value by about 10–15%.

Table 7: Privacy accounting comparison for different empirical block sizes.
<table><tr><td>Empirical block (%)</td><td> $\bar { \mu } _ { B }$ </td><td>Edgeworth estimate  $\varepsilon _ { \mathrm { e w } }$ </td><td>PLD lower bound</td><td>PLD upper bound</td><td>RDP upper bound</td></tr><tr><td>55</td><td>0.69</td><td>0.2987</td><td>0.04880</td><td>0.2988</td><td>0.36</td></tr><tr><td>62</td><td>1.56</td><td>1.68</td><td>1.44</td><td>1.69</td><td>2.63</td></tr><tr><td>65</td><td>2.09</td><td>5.33</td><td>6.05</td><td>6.29</td><td>7.81</td></tr><tr><td>67</td><td>2.84</td><td>23.0</td><td>25.49</td><td>25.7</td><td>48.8</td></tr></table>

## H DIFFERENTIAL PRIVACY AND GAUSSIAN DIFFERENTIAL PRIVACY

## H.1 DIFFERENTIAL PRIVACY

The concept of differential privacy (DP) (Abadi et al., 2016; Dwork et al., 2006) provides a formal guarantee that a model behaves almost identically regardless of whether any particular data sample is included in the training set. In other words, DP bounds the extent to which a single sample can influence the model’s behavior. More precisely, a mechanism M, which may be a training algorithm such as Stochastic Gradient Descent (SGD) or, more generally, any randomized algorithm including diffusion sampling, is said to be ϵ-DP if the privacy loss is bounded by ϵ: log $\begin{array} { r } { \frac { \mathbf { \tilde { \rho } } ^ { \prime } P r [ \mathcal { M } ( \mathcal { D } ) = o ] } { P r [ \mathcal { M } ( \mathcal { D } ^ { \prime } ) = o ] } \leq \check { \epsilon } . } \end{array}$ where D and $\mathcal { D } ^ { \prime }$ differ in the record of a single individual; and $\mathcal { M } ( \mathcal { D } ) = o$ denotes that applying the mechanism M on a dataset D ends up converging to the output event o.

For instance, if M is a training optimizer such as SGD, then $\mathcal { M } ( \mathcal { D } )$ is the trained model and $P r ( \mathcal { M } ( \mathcal { D } ) = o )$ depicts the probability the model o being the trained one. Intuitively, if deleting a single data point in $\mathcal { D }$ to obtain $\mathcal { D } ^ { \prime }$ does not substantially change the probability of obtaining the same output model $^ { O , }$ then the training algorithm can be regarded as privacy-preserving<sup>1</sup>. Put more simply, if the probability the model converges to the same parameters is relatively equivalent regardless of whether a particular sample is present, then that sample cannot be said to meaningfully affect training. More generally, M is (ε, δ)-DP if for all neighboring $\mathcal { D } , \mathcal { D } ^ { \prime }$ and every measurable set of outputs $O _ { ; }$

$$
\operatorname* { P r } [ \mathcal { M } ( \mathcal { D } ) \in O ] \leq e ^ { \varepsilon } \mathrm { P r } [ \mathcal { M } ( \mathcal { D } ^ { \prime } ) \in O ] + \delta .\tag{40}
$$

The privacy-loss condition $\begin{array} { r } { \mathrm { P r } \big [ \log \frac { \mathrm { P r } [ \mathcal { M } ( \mathcal { D } ) = o ] } { \mathrm { P r } [ \mathcal { M } ( \mathcal { D ^ { \prime } } ) = o ] } \geq \varepsilon \big ] \leq \delta } \end{array}$ implies eq. 40 but is not equivalent to it.

A natural building block for designing private (either training or sampling) algorithms is the Gaussian mechanism. This is particularly relevant for our setting, since each step in diffusion sampling injects Gaussian noise. Formally, given a function $h : \mathcal { D } \mapsto \mathbb { R } ^ { p }$ , the Gaussian mechanism releases

$$
\mathcal { M } _ { \mathrm { G a u s s } } ( \mathcal { D } ; h ) = h ( \mathcal { D } ) + n ,\tag{41}
$$

where $\pmb { n } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \Delta _ { h } ^ { 2 } \mathbf { I } )$ . Here, $\Delta _ { h }$ denotes the global sensitivity—the maximum $L _ { 2 }$ difference between outputs on two neighboring datasets—and σ controls the noise level as a function of $( \epsilon , \delta )$ By adding noise calibrated to sensitivity, the Gaussian mechanism ensures that the presence or absence of any single sample only has a limited influence on the output. This mechanism underlies differentially private training methods such as DP-SGD (Abadi et al., 2016), and, as we will show, it also directly connects to the diffusion sampling process we analyze in this work.

As an interesting fact in differential privacy, it is widely known that subsampling amplifies privacy (Kasiviswanathan et al., 2011). For example, if we flip a coin to decide whether each entry of the training data is in or out to form a smaller dataset, we would expect the mechanism using the smaller dataset to be twice as private as the original mechanism using the full dataset. With a flipping probability of 50%, every individual benefits from perfect privacy if the individual is not included in the smaller dataset. To benefit from the privacy amplification effect, a subsampled Gaussian mechanism (applying the Gaussian mechanism on randomly subsampled data) is also widely used (Abadi et al., 2016; Wang et al., 2019). We will also study the effect of subsampling on the privacy loss incurring during the diffusion sampling process.

## H.2 GAUSSIAN DIFFERENTIAL PRIVACY

While the Gaussian mechanism provides a natural foundation for analyzing privacy, the classical (ϵ, δ)-DP framework yields only loose bounds when these mechanisms are composed, often resulting in substantially looser guarantees than the actual privacy loss. To address this, we adopt the more refined f-DP framework (Dong et al., 2022), which characterizes privacy through hypothesis testing and provides an exact trade-off curve (function), rather than the looser worst-case bounds of $( \epsilon , \delta ) \ / { \cdot } \mathrm { D } \bar { \mathrm { P } } .$

A particularly important special case is Gaussian Differential Privacy (GDP), which describes the privacy profile of the Gaussian mechanism exactly using a single parameter $\mu .$ For example, in the case of $\mathrm { e q . }$ 41 with $\pmb { n } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \Delta _ { h } ^ { 2 } \mathbf { I } )$ , the Gaussian mechanism corresponds to $\scriptstyle { \frac { 1 } { \sigma } } - \mathrm { G D P } .$ . For a Gaussian mechanism, we can define a trade-off function $T$ using the standard normal CDF Φ given by,

$$
G _ { \mu } : = T ( { \mathcal { N } } ( 0 , 1 ) , { \mathcal { N } } ( \mu , 1 ) ) . \quad G _ { \mu } ( \alpha ) : = \Phi ( \Phi ^ { - 1 } ( 1 - \alpha ) - \mu ) .\tag{42}
$$

This makes GDP especially well aligned with diffusion sampling, where Gaussian perturbations are intrinsic to every denoising step. Moreover, GDP is closed under composition (Corollary 2 of Dong et al. (2022)): the cumulative privacy loss of n mechanisms with parameters $\mu _ { 1 } , \ldots , \mu _ { n }$ is given tightly by

$$
\mu = \sqrt { \sum _ { i = 1 } ^ { n } \mu _ { i } ^ { 2 } } .\tag{43}
$$

Thus, GDP not only matches the Gaussian mechanism exactly but also enables precise accounting of privacy loss across many Gaussian steps that constitute diffusion sampling. $\mathbf { A \mu } \mu \mathbf { - G D P }$ mechanism is $\bar { ( } \varepsilon , \delta ( \varepsilon ) ) \mathrm { - D P }$ for every $\varepsilon \geq 0$ with the closed form (Dong et al., 2022)

$$
\delta ( \varepsilon ) = \Phi \left( - \frac { \varepsilon } { \mu } + \frac { \mu } { 2 } \right) - e ^ { \varepsilon } \Phi \left( - \frac { \varepsilon } { \mu } - \frac { \mu } { 2 } \right) ,\tag{44}
$$

which we use for all full-batch conversions.

For subsampled Gaussian mechanisms, the trade-off function in eq. 42 needs to be modified to incorporate the randomness coming from the subsampling process and the GDP composition in eq. 43 is no longer valid. In this case, one can benefit from using the central limit theorem, which tells us that composing many f-DP mechanisms converges to GDP, as formally stated in Theorem 3.5 of Dong et al. (2022).

## I FULL-BATCH SAMPLING

In the full-batch regime every record contributes to every denoising step, so the per-step parameters $\mu _ { t }$ compose exactly by eq. 3 and convert to $( \varepsilon , \delta )$ by eq. 44. We vary the minimum noise level $\sigma _ { \mathrm { m i n } }$ to understand its impact on privacy guarantees. Figure 5 shows how the per-step $\mu$ grows as the noise level falls.

Lower $\sigma _ { \mathrm { m i n } }$ corresponds to less privacy. Figure 4 shows how the choice of $\sigma _ { \mathrm { m i n } }$ affects the privacy-utility trade-off. As $\sigma _ { \mathrm { m i n } }$ decreases (getting closer to the training data), the GDP analysis gives worse $( \varepsilon , \delta ) – \mathrm { D P }$ guarantees.

The full-batch empirical denoiser does not give good utility for its privacy cost. $\mathbf { A } \mathfrak { t } \varepsilon \approx 1$ (corresponding to $\sigma _ { \operatorname* { m i n } } = 5 )$ , the generated sample is hard to distinguish from pure noise (Figure 4, right). For a recognizable sample we need $\sigma _ { \mathrm { m i n } } = 0 . 1$ , which corresponds to ε ≈ 200.

Scalability of full-batch sampling. Even at low $\sigma _ { \mathrm { m i n } } .$ , full-batch diffusion sampling does not scale well with the number of generated samples. Since each denoising step operates on the full dataset, the per-sample privacy loss is already high, and composing multiple samples results in a rapidly growing total privacy budget under GDP. Subsampling alleviates this through privacy amplification (Section 4).

![](images/27efbca0aae7254ef914730da7320bf25f707fed24c46192c72958820e18641b.jpg)

![](images/1ce53899754a9eabfbb423ce162127ac018d19028ef84e6eca6b62fc5edf2599.jpg)

![](images/9a21e17543eb97981715a85ae4fe43a21b55f3d49de3d526ceced9db8e924dab.jpg)

Figure 4: GDP Analysis of Diffusion-Based Sampling (full batch) and Corresponding Diffusion Process Visualization. The noise scheduling constants are $\sigma _ { m a x } = 8 0 , N \bar { = } 5 0 , \bar { \beta } = 1$ , and $\rho = 7$ . The $( \epsilon , \delta ) \ / – \mathrm { D P }$ guarantees account for a single sample. Left. GDP conversion to $( \epsilon , \delta ) \ / – \mathrm { D P }$ for varying $\sigma _ { \mathrm { m i n } }$ . Middle. Type I and Type II Error under full-batch GDP with varying $\sigma _ { m i n }$ . Right. Visualization of Diffusion Process with Varying $\sigma _ { \mathrm { m i n } }$ . Only six timesteps are shown for figure clarity.  
![](images/50992a999aa4e9c8689710ea4ca032637af6e83ab9eef1ebb3052a4b1b47fda9.jpg)  
Figure 5: $\mu { \mathrm { - G D P } }$ for each time step with a full batch of data.

## I.1 PRIVACY LOSS IN RELATION TO CRITICAL WINDOWS

Figure 6 shows the CLIP-based detection of the critical window used in Section $6 . 3 ,$ and Figure 7 shows (ϵ, δ)-DP traces (left) and Type I and II error profiles (right) under GDP with varying stopping time points in the reverse process.

## J EVALUATION OF OUR HYBRID EMPIRICAL DENOISER AND NEURAL DENOISER ON CIFAR-10 AND CELEBA

The CIFAR-10 experiment uses an ImageNet-pretrained diffusion model for the neural denoiser, and we adopt the ρ-power schedule of EDM (Karras et al., 2022). The scheduler is parameterized by the minimum and maximum noise levels $( \sigma _ { m i n } , \sigma _ { m a x } )$ , the noise decay rate $\rho ,$ and the number of denoising steps N. We experiment with $\sigma _ { \operatorname* { m i n } } = 0 . 0 0 2 , \sigma _ { \operatorname* { m a x } } = 8 0 , \rho = 7 , N = 3 9$ , and the public normalization $\beta = 1$ (Appendix E).

We conduct a hyperparameter search over the number of re-noising steps $T \in \{ 1 , 3 , 5 , 7 , 9 \}$ empirical generation proportion prop $\in \ \{ 5 5 , 6 2 , 6 5 , 6 7 , 7 0 \} \%$ , and subsample batch size $B \in$ {25, 50, 100, 200}, giving 95 configurations (the 70%, $B = \hat { 2 } 0 0$ setting is not reported due to overflowing privacy calculations), with results in Table 8. For SDEdit with output-wise subsampling of size B, we run empirical denoising for the first $p r o p \%$ of the generation process, up to $t _ { 0 }$ . Afterwards, we re-noise by adding Gaussian noise with scale $( \sigma _ { t _ { 1 } } ^ { 2 } - \sigma _ { t _ { 0 } } ^ { 2 } ) ^ { 1 / 2 }$ . We continue denoising with the public Imagenet denoiser starting at $t _ { 1 }$ until the end. We generate 50,000 samples for every run and compute the FID against CIFAR10.

![](images/643d74889620010821e63a6fc0b5d3aa3e3f6cb6eb20735205b99bf5cc2e0bb7.jpg)  
(a)

![](images/f4b6905234e446ae3da545288156086745bdf6764546ca076028dbced0d90dbd.jpg)  
(b)

![](images/00af842f3362467e1fc6bd685cc9e1adeb360ca308607e3b99e9ad53b8a0cc63.jpg)  
(c)  
Figure 6: Detection of critical window. (a) zero-shot CLIP model detects the probability of an image being generated belonging to the ship class. (b) Privacy costs at three time points. (c) Generated images denoised at varying timesteps.

$$
\sigma _ { \mathrm { m i n } } = 0 . 0 0 2
$$

![](images/1932080a784f9b97c178295d4ffbc86b77340d90440974a5c727bcbc822a5659.jpg)

![](images/df174d6044802b623874dfde47ccd7fb50ef26e3c61fd6b8cced0983abf5012a.jpg)  
Figure 7: (ϵ, δ)-DP traces (Left) and Type I and II error profiles (Right) under GDP with varying stopping time points in the reverse process

Experiments were conducted on NVIDIA L40S GPUs with 48GB memory. Since our method primarily performs private sampling rather than full diffusion training, compute requirements were moderate. CIFAR-10 experiments typically required approximately 2.5 GPU-hours per run.

For CelebA (Liu et al., 2015) at $6 4 \times 6 4 ( n = 1 6 2 , 7 7 0 , \delta = 1 0 ^ { - 6 } )$ , we use FFHQ (Karras et al., 2019), trained without access to the private CelebA training split, as the disjoint public source for the neural denoiser, and keep the same EDM scheduler $( \sigma _ { m i n } = 0 . 0 0 2 , \sigma _ { m a x } = 8 0 , \rho = 7 , N = 3 9 )$ and public normalization $\beta = 1$ . We sweep the same base grid of re-noising steps $T \in \{ 1 , 3 , 5 , 7 , 9 \}$ empirical generation proportion prop $\bar { \in } \ \{ 5 5 , 6 2 , 6 5 , 6 7 , 7 0 \} \%$ , and subsample batch size $B \in$ {25, 50, 100, 200}, and at $T = 3$ additionally refine B (to {60, 70} for $p r o p \in \{ 6 2 , 6 5 \} \%$ and to $\{ 8 0 , 9 0 , 1 2 0 , 1 4 0 \}$ for $p r o p \in \{ 6 5 , 6 7 \} \% )$ , giving 112 configurations in total, with results reported in Table 9. The pipeline is otherwise identical to the CIFAR-10 setting: we run empirical denoising for the first $p r o p \%$ of generation up to $t _ { 0 } ,$ re-noise by adding Gaussian noise with scale $( \sigma _ { t _ { 1 } } ^ { 2 } - \sigma _ { t _ { 0 } } ^ { 2 } ) ^ { 1 / 2 }$ and complete generation with the public FFHQ denoiser from $t _ { 1 }$ until the end. We generate 50,000 samples for every run and compute the FID against CelebA. The Edgeworth diagnostic $\varepsilon _ { s y m }$ is available only for the refined $T \stackrel { - } { = } 3$ rows and is marked invalid where the reuse-validity flag is false or a cumulant ratio $\rho \geq 1 \left( \mathrm { e . g . } , p r o p = 6 5 \% , B = 6 0 \right.$ , where $\rho _ { 4 } ^ { Q } = 1 . 1 9 )$ . Table 10 repeats the $B = 1 0 0 , T = 3$ handoff sweep over three generation seeds.

Table 8: Hyperparameter search for output-wise subsample SDEdit on CIFAR-10. We sweep the number of re-noising steps $T \in \{ 1 , 3 , 5 , 7 , 9 \}$ , the proportion of empirical (denoiser) generation $p \in \{ 5 5 , 6 2 , 6 5 , 6 7 , 7 0 \} \%$ , and the subsample batch size $B \in \{ 2 5 , 5 0 , \bar { 1 } 0 0 , 2 0 0 \}$ ; 95 configurations in total. FID measures sample quality (lower is better). PLD $[ \mathinner { \varepsilon _ { \ell } \mathopen { , } \varepsilon _ { u } } ]$ is the PLD interval under the sampling model of Section 4 (optimistic lower / pessimistic upper estimate, Google dp\_accounting). Only $\varepsilon _ { u }$ is a certified bound. $\varepsilon _ { e w }$ is the second degree Edgeworth approximation (lower is better).
<table><tr><td rowspan=1 colspan=13>Noise Steps  Prop. Empirical (%)  Subsample Batch Size  FID↓     $\begin{array} { r } { \mathbf { P L D } \left[ \varepsilon _ { \ell } , \varepsilon _ { u } \right] \downarrow } \end{array}$      $\varepsilon _ { e w \ \downarrow }$ </td></tr><tr><td rowspan=2 colspan=10>1                 55                      25            147.4355                      50            143.57</td><td rowspan=1 colspan=2>[0.049, 0.299]</td><td rowspan=1 colspan=1>0.299</td></tr><tr><td rowspan=1 colspan=2>143.57</td><td rowspan=1 colspan=2>[0.386, 0.636</td><td rowspan=1 colspan=1>0.636</td></tr><tr><td rowspan=2 colspan=8>55                      10055                     200</td><td rowspan=1 colspan=2>141.23</td><td rowspan=1 colspan=2>[1.111, 1.361</td><td rowspan=1 colspan=1>1.360</td></tr><tr><td rowspan=1 colspan=2>139.74</td><td rowspan=1 colspan=2>[2.702, 2.952]</td><td rowspan=1 colspan=1>2.946</td></tr><tr><td rowspan=6 colspan=8>62                      2562                      5062                      10062                     20065                      2565                      50</td><td rowspan=1 colspan=2>78.79</td><td rowspan=1 colspan=2>[1.442, 1.691]</td><td rowspan=1 colspan=1>1.683</td></tr><tr><td rowspan=1 colspan=2>78.23</td><td rowspan=1 colspan=2>[3.172, 3.422]</td><td rowspan=1 colspan=1>3.513</td></tr><tr><td rowspan=1 colspan=2>78.73</td><td rowspan=1 colspan=2>[7.009, 7.259]</td><td rowspan=1 colspan=1>7.515</td></tr><tr><td rowspan=1 colspan=2>80.37    [1</td><td rowspan=1 colspan=2>6.009, 16.259]</td><td rowspan=1 colspan=1>16.706</td></tr><tr><td rowspan=2 colspan=6></td><td rowspan=1 colspan=2>78.53    [</td><td rowspan=1 colspan=2>6.052, 6.289]</td><td rowspan=1 colspan=1>5.331</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=4>77.65    [11.136, 11.381]</td><td rowspan=1 colspan=1>10.777</td></tr><tr><td rowspan=1 colspan=2></td><td></td><td></td><td rowspan=1 colspan=4>65                      100</td><td rowspan=1 colspan=4>78.24    [22.530, 22.780]</td><td rowspan=1 colspan=1>22.540</td></tr><tr><td rowspan=1 colspan=2></td><td></td><td></td><td rowspan=1 colspan=4>65                     200</td><td rowspan=1 colspan=3>79.77    [49.332</td><td rowspan=1 colspan=1>, 49.582</td><td rowspan=1 colspan=1>49.625</td></tr><tr><td rowspan=3 colspan=8>50100</td><td rowspan=2 colspan=5></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>80.59</td><td rowspan=1 colspan=1>[46.196,</td><td rowspan=1 colspan=1>46.438]</td><td rowspan=1 colspan=1>44.481</td></tr><tr><td rowspan=1 colspan=1>81.94</td><td rowspan=1 colspan=3>[90.634, 90.898]</td><td rowspan=1 colspan=1>89.733</td></tr><tr><td rowspan=1 colspan=8>67                     200</td><td rowspan=1 colspan=1>84.24</td><td rowspan=1 colspan=2>[190.673,</td><td rowspan=1 colspan=1>190.915</td><td rowspan=1 colspan=1>]   190.198</td></tr><tr><td rowspan=1 colspan=8>70                      25</td><td rowspan=1 colspan=1>74.98</td><td rowspan=1 colspan=2>[108.154,</td><td rowspan=1 colspan=1>108.444</td><td rowspan=1 colspan=1>]   104.035</td></tr><tr><td rowspan=2 colspan=8>70                      5070                     100</td><td rowspan=1 colspan=1>76.42</td><td rowspan=1 colspan=2>[194.318,</td><td rowspan=1 colspan=1>194.64</td><td rowspan=1 colspan=1>5   191.700</td></tr><tr><td rowspan=1 colspan=3>79.52   [369.696,</td><td rowspan=1 colspan=1>370.025</td><td rowspan=1 colspan=1>]   368.188</td></tr><tr><td rowspan=1 colspan=8>3                 55                      25</td><td rowspan=1 colspan=4>45.67     [0.049, 0.299]</td><td rowspan=1 colspan=1>0.299</td></tr><tr><td rowspan=1 colspan=8>55                      50</td><td rowspan=1 colspan=1>46.09</td><td></td><td rowspan=1 colspan=1>[0.386,</td><td rowspan=1 colspan=1>0.636]</td><td rowspan=1 colspan=1>0.636</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=3>55                      100</td><td rowspan=1 colspan=1>47.69</td><td></td><td rowspan=1 colspan=1>[1.111,</td><td rowspan=1 colspan=1>1.361]</td><td rowspan=1 colspan=1>1.360</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=3>55                     200</td><td rowspan=1 colspan=1>49.92</td><td></td><td rowspan=1 colspan=1>[2.702,</td><td rowspan=1 colspan=1>2.952]</td><td rowspan=1 colspan=1>2.946</td></tr><tr><td rowspan=2 colspan=6>6262</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>29.66</td><td></td><td rowspan=1 colspan=1>1.442,</td><td rowspan=1 colspan=1>1.691</td><td rowspan=1 colspan=1>1.683</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=1>31.51</td><td></td><td rowspan=1 colspan=1>[3.172,</td><td rowspan=1 colspan=1>3.422]</td><td rowspan=1 colspan=1>3.513</td></tr><tr><td rowspan=1 colspan=2></td><td></td><td rowspan=1 colspan=3>62</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>34.59</td><td></td><td rowspan=1 colspan=1>[7.009,</td><td rowspan=1 colspan=1>7.259]</td><td rowspan=1 colspan=1>7.515</td></tr><tr><td rowspan=1 colspan=6>62</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38.46</td><td rowspan=1 colspan=2>[16.009,</td><td rowspan=1 colspan=1>16.259]</td><td rowspan=1 colspan=1>16.706</td></tr><tr><td rowspan=1 colspan=6>65</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=2>[6.052,</td><td rowspan=1 colspan=1>6.289]</td><td rowspan=1 colspan=1>5.331</td></tr><tr><td rowspan=1 colspan=6>65</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27.13</td><td rowspan=1 colspan=2>[11.136,</td><td rowspan=1 colspan=1>11.381]</td><td rowspan=1 colspan=1>10.777</td></tr><tr><td rowspan=1 colspan=6>65</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>31.01</td><td rowspan=1 colspan=2>22.530,</td><td rowspan=1 colspan=1>22.780</td><td rowspan=1 colspan=1>22.540</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>65</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>35.12</td><td rowspan=1 colspan=2>[49.332,</td><td rowspan=1 colspan=1>49.582]</td><td rowspan=1 colspan=1>49.625</td></tr><tr><td rowspan=1 colspan=6>67</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>20.49</td><td rowspan=1 colspan=2>25.486,</td><td rowspan=1 colspan=1>25.696</td><td rowspan=1 colspan=1>22.982</td></tr><tr><td rowspan=1 colspan=6>67</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>23.27</td><td rowspan=1 colspan=2>46.196,</td><td rowspan=1 colspan=1>46.438</td><td rowspan=1 colspan=1>44.481</td></tr><tr><td rowspan=1 colspan=6>67</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27.48</td><td rowspan=1 colspan=2>[90.634,</td><td rowspan=1 colspan=1>90.898]</td><td rowspan=1 colspan=1>89.733</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>32.59</td><td rowspan=1 colspan=2>190.673</td><td rowspan=1 colspan=1>190.91</td><td rowspan=1 colspan=1>190.198</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16.38</td><td rowspan=1 colspan=2>108.154,</td><td rowspan=1 colspan=1>108.44</td><td rowspan=1 colspan=1>4   104.035</td></tr><tr><td rowspan=2 colspan=5></td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>19.46</td><td rowspan=1 colspan=2>194.318,</td><td rowspan=1 colspan=1>194.64</td><td rowspan=1 colspan=1>5   191.700</td></tr><tr><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>24.28   [</td><td rowspan=1 colspan=2>369.696,</td><td rowspan=1 colspan=1>370.025</td><td rowspan=1 colspan=1>]   368.188</td></tr><tr><td rowspan=1 colspan=5>5</td><td rowspan=1 colspan=2>55                      25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>33.16     [0.049,</td><td rowspan=1 colspan=1>0.299]</td><td rowspan=1 colspan=1>0.299</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>34.12</td><td rowspan=1 colspan=1>[0.386,</td><td rowspan=1 colspan=1>0.636]</td><td rowspan=1 colspan=1>0.636</td></tr><tr><td rowspan=1 colspan=5></td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>36.19</td><td rowspan=1 colspan=1>[1.111</td><td rowspan=1 colspan=1>, 1.361</td><td rowspan=1 colspan=1>1.360</td></tr><tr><td rowspan=1 colspan=6>55</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38.89</td><td></td><td rowspan=1 colspan=1>[2.702</td><td rowspan=1 colspan=1>, 2.952</td><td rowspan=1 colspan=1>2.946</td></tr><tr><td rowspan=6 colspan=8>6262                      5062                      10062                     20065                      2565                      50</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>29.39</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>30.94</td><td rowspan=1 colspan=2>[3.172,</td><td rowspan=1 colspan=1>3.422]</td><td rowspan=1 colspan=1>3.513</td></tr><tr><td rowspan=1 colspan=1>33.94</td><td rowspan=1 colspan=2>[7.009,</td><td rowspan=1 colspan=1>7.259]</td><td rowspan=1 colspan=1>7.515</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>37.44</td><td rowspan=1 colspan=2>[16.009,</td><td rowspan=1 colspan=1>16.259]</td><td rowspan=1 colspan=1>16.706</td></tr><tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27.35</td><td rowspan=1 colspan=2>[6.052,</td><td rowspan=1 colspan=1>6.289]</td><td rowspan=1 colspan=1>5.331</td></tr><tr><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>29.12</td><td rowspan=1 colspan=2>[11.136,</td><td rowspan=1 colspan=1>11.381]</td><td rowspan=1 colspan=1>10.777</td></tr><tr><td rowspan=1 colspan=6>65</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>32.55</td><td rowspan=1 colspan=2>[22.530,</td><td rowspan=1 colspan=1>22.780]</td><td rowspan=1 colspan=1>22.540</td></tr><tr><td rowspan=1 colspan=6>65</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>36.30</td><td rowspan=1 colspan=2>[49.332,</td><td rowspan=1 colspan=1>49.582]</td><td rowspan=1 colspan=1>49.625</td></tr><tr><td rowspan=2 colspan=6>6767</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>24.68</td><td rowspan=1 colspan=2>25.486,</td><td rowspan=1 colspan=1>25.696</td><td rowspan=1 colspan=1>22.982</td></tr><tr><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=1>26.94</td><td rowspan=1 colspan=2>46.196,</td><td rowspan=1 colspan=1>46.438</td><td rowspan=1 colspan=1>44.481</td></tr><tr><td rowspan=1 colspan=6>67</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>30.70</td><td rowspan=1 colspan=2>[90.634,</td><td rowspan=1 colspan=1>90.898]</td><td rowspan=1 colspan=1>89.733</td></tr><tr><td rowspan=2 colspan=6>6770</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>34.91   [</td><td rowspan=1 colspan=2>190.673,</td><td rowspan=1 colspan=1>190.915</td><td rowspan=1 colspan=1>]   190.198</td></tr><tr><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>22.65</td><td rowspan=1 colspan=2>108.154,</td><td rowspan=1 colspan=1>108.44</td><td rowspan=1 colspan=1>4   104.035</td></tr><tr><td rowspan=2 colspan=8>7070                     100</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>25.21</td><td rowspan=1 colspan=1>[194.318</td></tr><tr><td rowspan=1 colspan=1>29.07</td><td rowspan=1 colspan=3>369.696, 370.02</td><td rowspan=1 colspan=1>5   368.188</td></tr></table>

Continued on next page

Table 8 continued
<table><tr><td>Noise Steps</td><td>Prop. Empirical (%)</td><td>Subsample Batch Size</td><td>FID↓</td><td> $\begin{array} { r } { \mathbf { P L D } \left[ \varepsilon _ { \ell } , \varepsilon _ { u } \right] \downarrow } \end{array}$ </td><td> $\varepsilon _ { e w \ \downarrow }$ </td></tr><tr><td>7</td><td>55</td><td>25</td><td>30.99</td><td>[0.049, 0.299]</td><td>0.299</td></tr><tr><td></td><td>55</td><td>50</td><td>31.97</td><td>[0.386, 0.636]</td><td>0.636</td></tr><tr><td></td><td>55</td><td>100</td><td>34.01</td><td>[1.111, 1.361]</td><td>1.360</td></tr><tr><td></td><td>55</td><td>200</td><td>36.38</td><td>[2.702, 2.952]</td><td>2.946</td></tr><tr><td></td><td>62</td><td>25</td><td>30.54</td><td>[1.442, 1.691]</td><td>1.683</td></tr><tr><td></td><td>62</td><td>50</td><td>31.81</td><td>[3.172, 3.422]</td><td>3.513</td></tr><tr><td></td><td>62</td><td>100</td><td>34.34</td><td>[7.009, 7.259]</td><td>7.515</td></tr><tr><td></td><td>62</td><td>200</td><td>37.34</td><td>[16.009, 16.259]</td><td>16.706</td></tr><tr><td></td><td>65</td><td>25</td><td>29.70</td><td>[6.052, 6.289]</td><td>5.331</td></tr><tr><td></td><td>65</td><td>50</td><td>31.26</td><td>[11.136, 11.381]</td><td>10.777</td></tr><tr><td></td><td>65</td><td>100</td><td>34.02</td><td>[22.530, 22.780]</td><td>22.540</td></tr><tr><td></td><td>65</td><td>200</td><td>37.18</td><td>49.332, 49.582</td><td>49.625</td></tr><tr><td></td><td>67</td><td>25</td><td>28.23</td><td>25.486, 25.696</td><td>22.982</td></tr><tr><td></td><td>67</td><td>50</td><td>29.86</td><td>[46.196, 46.438]</td><td>44.481</td></tr><tr><td></td><td>67</td><td>100</td><td>33.13</td><td>[90.634, 90.898]</td><td>89.733</td></tr><tr><td></td><td>67</td><td>200</td><td>36.41</td><td>[190.673, 190.915</td><td>190.198</td></tr><tr><td></td><td>70</td><td>25</td><td>26.32</td><td>108.154, 108.444</td><td>104.035</td></tr><tr><td></td><td>70</td><td>50</td><td>28.51</td><td>[194.318, 194.645]</td><td>191.700</td></tr><tr><td>9</td><td>70</td><td>100</td><td>31.76</td><td>[369.696, 370.025]</td><td>368.188</td></tr><tr><td></td><td>55</td><td>25</td><td>30.43</td><td>[0.049, 0.299]</td><td>0.299</td></tr><tr><td></td><td>55</td><td>50</td><td>31.41</td><td>[0.386, 0.636]</td><td>0.636</td></tr><tr><td></td><td>55</td><td>100</td><td>33.14</td><td>[1.111, 1.361]</td><td>1.360</td></tr><tr><td></td><td>55</td><td>200</td><td>35.07</td><td>[2.702, 2.952]</td><td>2.946</td></tr><tr><td></td><td>62</td><td>25</td><td>30.71</td><td>[1.442, 1.691]</td><td>1.683</td></tr><tr><td></td><td>62</td><td>50</td><td>31.79</td><td>[3.172, 3.422]</td><td>3.513</td></tr><tr><td></td><td>62</td><td>100</td><td>33.94</td><td>[7.009, 7.259]</td><td>7.515</td></tr><tr><td></td><td>62</td><td>200</td><td>36.49</td><td>[16.009, 16.259]</td><td>16.706</td></tr><tr><td></td><td>65</td><td>25</td><td>30.30</td><td>[6.052, 6.289]</td><td>5.331</td></tr><tr><td></td><td>65</td><td>50</td><td>31.80</td><td>[11.136, 11.381]</td><td>10.777</td></tr><tr><td></td><td>65</td><td>100</td><td>34.00</td><td>[22.530, 22.780]</td><td>22.540</td></tr><tr><td></td><td>65</td><td>200</td><td>36.74</td><td>[49.332, 49.582</td><td>49.625</td></tr><tr><td></td><td>67</td><td>25</td><td>29.97</td><td>[25.486, 25.696]</td><td>22.982</td></tr><tr><td></td><td>67</td><td>50</td><td>31.31</td><td>[46.196, 46.438]</td><td>44.481</td></tr><tr><td></td><td>67</td><td>100</td><td>33.87</td><td>[90.634, 90.898]</td><td>89.733</td></tr><tr><td></td><td>67</td><td>200</td><td>36.69</td><td>[190.673, 190.915]</td><td>190.198</td></tr><tr><td></td><td>70</td><td>25</td><td>29.25</td><td>[108.154, 108.444</td><td>104.035</td></tr><tr><td></td><td>70</td><td>50</td><td>31.11 33.78</td><td>[194.318, 194.645]</td><td>191.700 368.188</td></tr><tr><td></td><td>70</td><td>100</td><td></td><td>[369.696, 370.025]</td><td></td></tr></table>

Table 9: Hyperparameter search for output-wise subsample SDEdit on CelebA. The base grid sweeps re-noising steps $T \in \{ 1 , 3 , 5 , 7 , 9 \}$ , proportion of empirical (denoiser) generation $p \in$ $\{ 5 5 , 6 2 , 6 5 , 6 7 , 7 0 \} \%$ , and subsample batch size $\bar { B } \in \{ 2 5 , 5 0 , 1 \bar { 0 } 0 , 2 0 0 \} ; \mathrm { a t } T = 3$ we additionally refine B (to {60, 70} for $p \in \{ 6 2 , 6 5 \} \%$ and to {80, 90, 120, 140} for $\bar { p } \in \{ 6 5 , 6 7 \} \% )$ , giving 112 configurations. FID measures sample quality (lower is better). $\mathrm { P L D } \left[ \varepsilon _ { \ell } , \varepsilon _ { u } \right]$ is the PLD interval under the sampling model of Section 4 (optimistic lower / pessimistic upper estimate, Google dp\_accounting). $\mathrm { O n l y } ~ \varepsilon _ { u }$ is a certified bound. $\varepsilon _ { e w }$ is the (diagnostic) Edgeworth symmetric privacy loss; it is marked invalid where the reuse validity flag is false or a cumulant ratio $\rho \geq 1$ (diagnostics available only for the refined T = 3 rows).
<table><tr><td>Noise Steps</td><td>Prop. Empirical (%)</td><td>Subsample Batch Size</td><td>FID↓</td><td> $\begin{array} { r } { \mathbf { P L D } \left[ \varepsilon _ { \ell } , \varepsilon _ { u } \right] \downarrow } \end{array}$ </td><td> $\varepsilon _ { e w \ \downarrow }$ </td></tr><tr><td rowspan="7">1</td><td>55</td><td>25</td><td>116.96</td><td>[0.000, 0.099]</td><td>0.099</td></tr><tr><td>55</td><td>50</td><td>113.07</td><td>[0.000, 0.208]</td><td>0.208</td></tr><tr><td>55</td><td>100</td><td>109.95</td><td>[0.185, 0.435]</td><td>0.435</td></tr><tr><td>55</td><td>200</td><td>107.90</td><td>[0.666, 0.916]</td><td>0.915</td></tr><tr><td>62</td><td>25</td><td>45.45</td><td>[0.516, 0.767]</td><td>0.563</td></tr><tr><td>62</td><td>50</td><td>45.23</td><td>[1.179, 1.430]</td><td>1.150</td></tr><tr><td>62 62</td><td>100</td><td>46.24 47.05</td><td>[2.308, 2.558]</td><td>2.361</td></tr><tr><td></td><td>65</td><td>200 25</td><td>37.32</td><td>[4.654, 4.904] [3.623, 3.891]</td><td>4.912 invalid  $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 3 2 )$ </td></tr></table>

Table 9 continued
<table><tr><td rowspan=1 colspan=9>Noise Steps  Prop. Empirical (%)  Subsample Batch Size  FID↓    PLD $\left[ \varepsilon _ { \ell } , \varepsilon _ { u } \right] \downarrow$       $\varepsilon _ { e w \ \downarrow }$ </td></tr><tr><td rowspan=1 colspan=9>65                      50             38.20    [5.249, 5.500]      $i n \nu a l i d ( \rho _ { 4 } ^ { Q } { = } 1 . 4 8 )$ </td></tr><tr><td rowspan=1 colspan=5>65                     100             39.87</td><td rowspan=1 colspan=4>[8.482, 8.737]     7.147</td></tr><tr><td rowspan=1 colspan=5>65                     200             41.71</td><td rowspan=1 colspan=4>[15.522, 15.769]    14.552</td></tr><tr><td rowspan=1 colspan=5>67                      25             30.23</td><td rowspan=1 colspan=3>[13.360, 13.657]</td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 4 6 )$ </td></tr><tr><td rowspan=1 colspan=5>67                      50             32.07</td><td rowspan=1 colspan=3>[20.517, 20.746]</td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 1 . 2 8 )$ </td></tr><tr><td rowspan=1 colspan=5>67                     100             34.65</td><td rowspan=1 colspan=3>[34.120, 34.392]</td><td rowspan=1 colspan=1>30.085</td></tr><tr><td rowspan=1 colspan=5>67                     200             37.48</td><td rowspan=1 colspan=2>[61.908, 62.120]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.631</td></tr><tr><td rowspan=1 colspan=4>70                      25</td><td rowspan=1 colspan=1>24.58</td><td rowspan=1 colspan=2>[52.702, 53.019]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { P } { = } 2 . 2 6 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>70                      50</td><td rowspan=1 colspan=1>27.01</td><td rowspan=1 colspan=2>[83.289, 83.470]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { P } \mathrm { = } 1 . 1 5 )$ </td></tr><tr><td rowspan=1 colspan=4>70                     100</td><td rowspan=1 colspan=1>30.48  [1</td><td rowspan=1 colspan=3>40.112, 140.393]</td><td rowspan=1 colspan=1>132.911</td></tr><tr><td rowspan=1 colspan=9>70                     200             33.82  [251.533, 251.653]  245.964</td></tr><tr><td rowspan=2 colspan=4>3                 55                      2555                      50</td><td rowspan=1 colspan=1>35.16</td><td rowspan=1 colspan=4>[0.000, 0.099]     0.099</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>35.78</td><td rowspan=1 colspan=3>[0.000, 0.208]</td><td rowspan=1 colspan=1>0.208</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>37.42</td><td rowspan=1 colspan=3>[0.185, 0.435]</td><td rowspan=1 colspan=1>0.435</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>39.30</td><td rowspan=1 colspan=3>[0.666, 0.916]</td><td rowspan=1 colspan=1>0.915</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>25.56</td><td rowspan=1 colspan=3>[0.516, 0.767]</td><td rowspan=1 colspan=1>0.563</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>62                      50</td><td rowspan=1 colspan=1>27.57</td><td rowspan=1 colspan=3>[1.179, 1.430]</td><td rowspan=1 colspan=1>1.150</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>62                      60</td><td rowspan=1 colspan=1>28.36</td><td rowspan=1 colspan=3>[1.415, 1.666]</td><td rowspan=1 colspan=1>1.388</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>62                      70</td><td rowspan=1 colspan=1>28.85</td><td rowspan=1 colspan=3>[1.644, 1.893]</td><td rowspan=1 colspan=1>1.628</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>62                     100</td><td rowspan=1 colspan=1>30.31</td><td rowspan=1 colspan=3>[2.308, 2.558]</td><td rowspan=1 colspan=1>2.361</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>62                     200</td><td rowspan=1 colspan=1>33.01</td><td rowspan=1 colspan=3>[4.654, 4.904]</td><td rowspan=1 colspan=1>4.912</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>65                      25</td><td rowspan=1 colspan=1>21.95</td><td rowspan=1 colspan=3>[3.623, 3.891]</td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 3 2 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=1>24.34</td><td rowspan=1 colspan=1>[5.249,</td><td rowspan=1 colspan=2>5.500]</td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 1 . 4 8 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>60</td><td rowspan=1 colspan=1>25.01</td><td rowspan=1 colspan=1>[5.886,</td><td rowspan=1 colspan=2>6.129]</td><td rowspan=1 colspan=1>in valid (p =1.19)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>70</td><td rowspan=1 colspan=1>25.67</td><td rowspan=1 colspan=1>[6.531,</td><td rowspan=1 colspan=2>6.767]</td><td rowspan=1 colspan=1>4.999</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>80</td><td rowspan=1 colspan=1>26.30</td><td rowspan=1 colspan=1>[7.167,</td><td rowspan=1 colspan=2>7.415]</td><td rowspan=1 colspan=1>5.712</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>90</td><td rowspan=1 colspan=1>26.81</td><td rowspan=1 colspan=1>[7.818,</td><td rowspan=1 colspan=2>8.072]</td><td rowspan=1 colspan=1>6.428</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>27.25</td><td rowspan=1 colspan=1>[8.482,</td><td rowspan=1 colspan=2>8.737]</td><td rowspan=1 colspan=1>7.147</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>120</td><td rowspan=1 colspan=1>27.99</td><td rowspan=1 colspan=1>[9.839, 1</td><td rowspan=1 colspan=1>0.090]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8.596</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>140</td><td rowspan=1 colspan=1>28.72</td><td rowspan=1 colspan=1>[11.228,</td><td rowspan=1 colspan=1>11.472]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10.060</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>30.05</td><td rowspan=1 colspan=1>[15.522,</td><td rowspan=1 colspan=1>15.769]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>14.552</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>17.94</td><td rowspan=1 colspan=1>[13.360,</td><td rowspan=1 colspan=1>13.657]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 4 6 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=1>20.39</td><td rowspan=1 colspan=1>[20.517,</td><td rowspan=1 colspan=1>20.746]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 1 . 2 8 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>22.50</td><td rowspan=1 colspan=1>[28.670,</td><td rowspan=1 colspan=1>28.950]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>24.433</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>23.16</td><td rowspan=1 colspan=1>[31.387,</td><td rowspan=1 colspan=1>31.670]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27.260</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>23.67</td><td rowspan=1 colspan=1>[34.120,</td><td rowspan=1 colspan=1>34.392</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>30.085</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=2>120</td><td rowspan=1 colspan=1>24.63</td><td rowspan=1 colspan=1>[39.619,</td><td rowspan=1 colspan=1>39.854]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>35.739</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>140</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>25.35</td><td rowspan=1 colspan=1>[45.108,</td><td rowspan=1 colspan=1>45.350]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>41.411</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>26.93</td><td rowspan=1 colspan=1>[61.908,</td><td rowspan=1 colspan=1>62.120</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>58.631</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>14.70   [</td><td rowspan=1 colspan=1>52.702,</td><td rowspan=1 colspan=1>53.019]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { P } \mathrm { { = } } 2 . 2 6 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>17.44</td><td rowspan=1 colspan=1>[83.289,</td><td rowspan=1 colspan=1>83.470]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>invalid (ρP =1.15)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>20.97  [1</td><td rowspan=1 colspan=1>40.112,</td><td rowspan=1 colspan=2>140.393]</td><td rowspan=1 colspan=1>132.911</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>70</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>24.54  [2</td><td rowspan=1 colspan=3>51.533, 251.653]</td><td rowspan=1 colspan=1>245.964</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>28.81</td><td rowspan=1 colspan=3>[0.000, 0.099]</td><td rowspan=1 colspan=1>0.099</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>29.63</td><td rowspan=1 colspan=1>[0.000,</td><td rowspan=1 colspan=2>0.208]</td><td rowspan=1 colspan=1>0.208</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>31.20</td><td rowspan=1 colspan=1>[0.185,</td><td rowspan=1 colspan=2>0.435]</td><td rowspan=1 colspan=1>0.435</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>32.72</td><td rowspan=1 colspan=1>[0.666,</td><td rowspan=1 colspan=2>0.916]</td><td rowspan=1 colspan=1>0.915</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>26.17</td><td rowspan=1 colspan=1>[0.516,</td><td rowspan=1 colspan=2>0.767]</td><td rowspan=1 colspan=1>0.563</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=1>28.28</td><td rowspan=1 colspan=1>[1.179,</td><td rowspan=1 colspan=2>1.430]</td><td rowspan=1 colspan=1>1.150</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>31.21</td><td rowspan=1 colspan=1>[2.308,</td><td rowspan=1 colspan=2>2.558]</td><td rowspan=1 colspan=1>2.361</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>62</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>33.94</td><td rowspan=1 colspan=3>[4.654, 4.904]</td><td rowspan=1 colspan=1>4.912</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>24.37</td><td rowspan=1 colspan=2>[3.623, 3.891</td><td rowspan=1 colspan=1>]</td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 3 2 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>50</td><td rowspan=1 colspan=1>26.86</td><td rowspan=1 colspan=2>[5.249, 5.500]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 1 . 4 8 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>100</td><td rowspan=1 colspan=1>29.66</td><td rowspan=1 colspan=2>[8.482, 8.737]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.147</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=2>200</td><td rowspan=1 colspan=1>32.50</td><td rowspan=1 colspan=2>[15.522, 15.769</td><td rowspan=1 colspan=1>]</td><td rowspan=1 colspan=1>14.552</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>67</td><td rowspan=1 colspan=2>25</td><td rowspan=1 colspan=1>22.17</td><td rowspan=1 colspan=2>[13.360, 13.657</td><td rowspan=1 colspan=2>]    invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 4 6 )$ </td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>67                      50</td><td rowspan=1 colspan=1>24.53</td><td rowspan=1 colspan=2>[20.517, 20.746]</td><td rowspan=1 colspan=2>invalid $( \rho _ { 4 } ^ { Q } \mathrm { = } 1 . 2 8 )$ </td></tr></table>

Continued on next page

Table 9 continued
<table><tr><td>Noise Steps</td><td>Prop. Empirical (%)</td><td>Subsample Batch Size</td><td>FID↓</td><td>PLD  $\left[ \varepsilon _ { \ell } , \varepsilon _ { u } \right] \downarrow$ </td><td> $\varepsilon _ { e w \ \downarrow }$ </td></tr><tr><td></td><td>67</td><td>100</td><td>27.70</td><td>[34.120, 34.392]</td><td>30.085</td></tr><tr><td></td><td>67</td><td>200</td><td>30.67</td><td>[61.908, 62.120]</td><td>58.631</td></tr><tr><td></td><td>70</td><td>25</td><td>20.05</td><td>[52.702, 53.019]</td><td>invalid  $( \rho _ { 4 } ^ { P } \mathrm { { = } } 2 . 2 6 )$ </td></tr><tr><td></td><td>70</td><td>50</td><td>22.64</td><td>[83.289, 83.470]</td><td>invalid  $( \rho _ { 4 } ^ { P } \mathrm { = } 1 . 1 5 )$ </td></tr><tr><td></td><td>70</td><td>100</td><td>25.88</td><td>[140.112, 140.393]</td><td>132.911</td></tr><tr><td>7</td><td>70</td><td>200</td><td>29.24</td><td>[251.533, 251.653]</td><td>245.964</td></tr><tr><td></td><td>55</td><td>25</td><td>29.25</td><td>[0.000, 0.099]</td><td>0.099</td></tr><tr><td></td><td>55</td><td>50</td><td>29.67</td><td>[0.000, 0.208]</td><td>0.208</td></tr><tr><td></td><td>55</td><td>100</td><td>30.73</td><td>[0.185, 0.435]</td><td>0.435</td></tr><tr><td></td><td>55</td><td>200</td><td>31.74</td><td>[0.666, 0.916]</td><td>0.915</td></tr><tr><td></td><td>62</td><td>25</td><td>26.42</td><td>[0.516, 0.767]</td><td>0.563</td></tr><tr><td></td><td>62</td><td>50</td><td>28.15</td><td>[1.179, 1.430]</td><td>1.150</td></tr><tr><td></td><td>62</td><td>100</td><td>30.47</td><td>[2.308, 2.558]</td><td>2.361</td></tr><tr><td></td><td>62</td><td>200</td><td>32.79</td><td>[4.654, 4.904]</td><td>4.912</td></tr><tr><td></td><td>65</td><td>25</td><td>25.73</td><td>[3.623, 3.891]</td><td>invalid  $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 3 2 )$ </td></tr><tr><td></td><td>65</td><td>50</td><td>27.96</td><td>[5.249, 5.500]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { Q } { = } 1 . 4 8 )$ </td></tr><tr><td></td><td>65</td><td>100</td><td>30.43</td><td>[8.482, 8.737]</td><td>7.147</td></tr><tr><td></td><td>65</td><td>200</td><td>33.00</td><td>[15.522, 15.769]</td><td>14.552</td></tr><tr><td></td><td>67</td><td>25</td><td>25.02</td><td>[13.360, 13.657]</td><td>invalid  $( \rho _ { 4 } ^ { Q } \mathrm { = } 3 . 4 6 )$ </td></tr><tr><td></td><td>67</td><td>50</td><td>27.37</td><td>[20.517, 20.746]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { Q } { = } 1 . 2 8 )$ </td></tr><tr><td></td><td>67</td><td>100</td><td>30.17</td><td>[34.120, 34.392]</td><td>30.085</td></tr><tr><td></td><td>67</td><td>200</td><td>32.95</td><td>[61.908, 62.120]</td><td>58.631</td></tr><tr><td></td><td>70</td><td>25</td><td>23.51</td><td>[52.702, 53.019]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { P } { = } 2 . 2 6 )$ </td></tr><tr><td></td><td>70</td><td>50</td><td>25.97</td><td>[83.289, 83.470]</td><td> $i n \nu a l i d ~ ( \rho _ { 4 } ^ { P } { = } 1 . 1 5 )$ </td></tr><tr><td></td><td>70</td><td>100</td><td>29.19</td><td>[140.112, 140.393]</td><td>132.911</td></tr><tr><td>9</td><td>70</td><td>200</td><td>32.23</td><td>[251.533, 251.653]</td><td>245.964</td></tr><tr><td></td><td>55</td><td>25</td><td>31.27</td><td>[0.000, 0.099]</td><td>0.099</td></tr><tr><td></td><td>55</td><td>50</td><td>31.25</td><td>[0.000, 0.208]</td><td>0.208</td></tr><tr><td></td><td>55</td><td>100</td><td>31.69</td><td>[0.185, 0.435]</td><td>0.435</td></tr><tr><td></td><td>55</td><td>200</td><td>32.19</td><td>[0.666, 0.916]</td><td>0.915</td></tr><tr><td></td><td>62</td><td>25</td><td>27.50</td><td>[0.516, 0.767]</td><td>0.563</td></tr><tr><td></td><td>62</td><td>50</td><td>28.47</td><td>[1.179, 1.430]</td><td>1.150</td></tr><tr><td></td><td>62</td><td>100</td><td>29.87</td><td>[2.308, 2.558]</td><td>2.361</td></tr><tr><td></td><td>62</td><td>200</td><td>31.47</td><td>[4.654, 4.904]</td><td>4.912</td></tr><tr><td></td><td>65</td><td>25</td><td>26.39</td><td>[3.623, 3.891]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { Q } { = } 3 . 3 2 )$ </td></tr><tr><td></td><td>65</td><td>50</td><td>27.89</td><td>[5.249, 5.500]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { Q } { = } 1 . 4 8 )$ </td></tr><tr><td></td><td>65</td><td>100</td><td>29.73</td><td>[8.482, 8.737]</td><td>7.147</td></tr><tr><td></td><td>65</td><td>200</td><td>31.58</td><td>[15.522, 15.769]</td><td>14.552</td></tr><tr><td></td><td>67</td><td>25</td><td>26.09</td><td>[13.360, 13.657]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { Q } { = } 3 . 4 6 )$ </td></tr><tr><td></td><td>67</td><td>50</td><td>27.78</td><td>[20.517, 20.746]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { Q } { = } 1 . 2 8 )$ </td></tr><tr><td></td><td>67</td><td>100</td><td>29.93</td><td>[34.120, 34.392]</td><td>30.085</td></tr><tr><td></td><td>67</td><td>200</td><td>32.34</td><td>[61.908, 62.120]</td><td>58.631</td></tr><tr><td></td><td>70</td><td>25</td><td>25.30</td><td>[52.702, 53.019]</td><td> $i n \nu a l i d ( \rho _ { 4 } ^ { P } { = } 2 . 2 6 )$ </td></tr><tr><td></td><td>70</td><td>50</td><td>27.54</td><td>[83.289, 83.470]</td><td> $i n \nu a l i d ~ ( \rho _ { 4 } ^ { P } { = } 1 . 1 5 )$ </td></tr><tr><td></td><td>70</td><td>100</td><td>30.32</td><td>[140.112, 140.393</td><td>132.911</td></tr><tr><td></td><td>70</td><td>200</td><td>32.90</td><td>[251.533, 251.653]</td><td>245.964</td></tr></table>

Table 10: CelebA-64, $B = 1 0 0$ , three public re-noising steps, $\delta = 1 0 ^ { - 6 } , N _ { \mathrm { o u t } } = 5 0 , 0 0 0$ . FID is mean ± standard deviation over three generation seeds.
<table><tr><td>Handoff</td><td>Edgeworth estimate</td><td>PLD lower</td><td>Certified PLD upper</td><td>FID</td></tr><tr><td>55%</td><td>0.43</td><td>0.19</td><td>0.44</td><td> $3 7 . 3 9 \pm 0 . 0 5$ </td></tr><tr><td>62%</td><td>2.36</td><td>2.31</td><td>2.56</td><td> $3 0 . 4 5 \pm 0 . 1 7$ </td></tr><tr><td>65%</td><td>7.14</td><td>8.48</td><td>8.74</td><td> $2 7 . 3 1 \pm 0 . 0 7$ </td></tr><tr><td>67%</td><td>30.09</td><td>34.12</td><td>34.39</td><td> $2 3 . 7 5 \pm 0 . 0 8$ </td></tr><tr><td>70%</td><td>132.91</td><td>140.11</td><td>140.39</td><td> $2 0 . 9 8 \pm 0 . 0 5$ </td></tr></table>

<table><tr><td>Handoff (%)</td><td>Edgeworth estimate</td><td>CLT estimate</td><td>Certified PLD upper</td></tr><tr><td>55</td><td>0.299</td><td>0.296</td><td>0.299</td></tr><tr><td>62</td><td>1.683</td><td>1.388</td><td>1.691</td></tr><tr><td>65</td><td>5.331</td><td>4.313</td><td>6.289</td></tr><tr><td>67</td><td>22.982</td><td>45.101</td><td>25.696</td></tr><tr><td>70</td><td>104.035</td><td>25,905.494</td><td>108.444</td></tr></table>

Table 11: Edgeworth and CLT estimates of ε against the certified PLD upper bound (CIFAR-10, $B = 2 5$ , three re-noising steps, $\delta = 1 0 ^ { - 5 } )$ . Neither approximation is a bound. The CLT estimate understates the certified value by 18% and 31% at the 62% and 65% blocks and overstates it by more than 200 times at the 70% block. The Edgeworth estimate stays within 1% for $\varepsilon \lesssim 2$ and understates it by 15%, 11%, and 4% at the 65%, 67%, and 70% blocks.

## K SELECTION OF THE PRIVACY PARAMETERS ε AND δ

For a fixed sampler, Propositions 4.1 and C.1 determine a complete privacy profile $\delta _ { \mathcal { M } } ( \varepsilon )$ : a practitioner picks a target $\delta _ { \mathrm { t a r g e t } }$ and reports the smallest ε with $\delta _ { \mathcal { M } } ( \varepsilon ) \leq \delta _ { \mathrm { t a r g e t } } .$ , so δ is not tuned only through $\varepsilon -$ the pair describes one fixed mechanism. Arbitrarily small ε and δ cannot coexist at fixed utility; improving both requires changing the mechanism (lower sampling rate, fewer private steps or released outputs, smaller clipping bound, more noise, or an earlier handoff).

We used $\delta = 1 0 ^ { - 5 } \approx 1 / n$ for CIFAR-10 to match common DP generative-model comparisons;   
$1 / n ^ { 2 } = 4 \times 1 0 ^ { - 1 0 }$ is indeed stricter, and we report both in Table 12.

Table 12: Certified RDP ε for CIFAR-10 output-wise subsampled blocks at $\delta = 1 0 ^ { - 5 }$ and at the stricter ${ \delta = 4 \times 1 0 ^ { - 1 0 } = 1 / n ^ { 2 } ( B = 2 5 , \tilde { N _ { \mathrm { { o u t } } } } \tilde { = } 5 0 , 0 0 0 ) }$
<table><tr><td>Configuration</td><td> $\mathsf { R D P } \varepsilon , \delta = 1 0 ^ { - 5 }$ </td><td> $\mathsf { R D P } \varepsilon , \delta = 4 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td> $5 5 \% , B = 2 5$ </td><td>0.36</td><td>0.71</td></tr><tr><td> $6 2 \% , B = 2 5$ </td><td>2.63</td><td>5.16</td></tr><tr><td> $6 5 \% , B = 2 5$ </td><td>7.81</td><td>12.87</td></tr></table>

## L COMPUTING THEORETICAL ASR

Following Triastcyn & Faltings (2020), we calculate the theoretical ASR as follows. Suppose an adversary observed an ϵ-DP release o and tried to distinguish between two candidate datasets $\mathcal { D } , \mathcal { D } ^ { \prime }$ with a uniform prior $p ( \mathcal { D } ) = p ( \mathcal { D } ^ { \prime } ) = 0 . 5$ . Without loss of generality, let D be the correct dataset, then the adversary will infer the probability p(D|o) as $\begin{array} { r } { p ( \mathcal { D } | o ) = \frac { p ( o | \mathcal { D } ) p ( \mathcal { D } ) } { p ( o | \mathcal { D } ) p ( \mathcal { D } ) + p ( o | \mathcal { D } ^ { \prime } ) p ( \mathcal { D } ^ { \prime } ) } \leq } \end{array}$ $\begin{array} { r } { \frac { p ( o | \mathcal { D } ) } { p ( o | \mathcal { D } ) + \exp ( - \epsilon ) p ( o | \mathcal { D } ) } = \frac { 1 } { 1 + \exp ( - \epsilon ) } } \end{array}$ . For $( \varepsilon , \delta ) – \mathrm { D P }$ the bound becomes $\frac { e ^ { \varepsilon } + \delta } { 1 + e ^ { \varepsilon } }$ , which at $\delta = 1 0 ^ { - 5 }$ changes no entry of Table 13 at the reported precision. So, for $\epsilon = 1$ the upperbound on the membership inference success probability is $7 3 . 1 \%$ , for $\epsilon = 5 \AA$ , it is 99.33%, and for $\epsilon = 1 0$ , it is 99.995%. Furthermore, the success probability of MIAs typically similar or lower than the theoretical success rate as shown in Lowy et al. (2024).

Table 13: Membership inference. Theoretical: ASR implied by our ε for the subsampled empirical sampler at rate q. Numerical: ASR of the attack of Duan et al. (2023) on a neural denoiser trained on all of CIFAR-10. Both stop at time t of a $T = 1 0 0 0$ step discretization. The neural denoiser is not subsampled, so the theoretical rows are not a bound on it.
<table><tr><td>t</td><td>T</td><td> $\overline { { \frac { 3 } { 4 } T } }$ </td><td> $\overline { { \frac { 1 } { 2 } T } }$ </td><td> $\scriptstyle { \frac { 1 } { 4 } } T$ </td></tr><tr><td>Cosine similarity (Fig. 1)</td><td>1.000</td><td>0.999</td><td>0.956</td><td>0.845</td></tr><tr><td>Theoretical ASR at q = 0.5</td><td>0.5019</td><td>0.5204</td><td>0.6909</td><td>0.9999</td></tr><tr><td>Theoretical ASR at q = 0.01</td><td>0.5000</td><td>0.5002</td><td>0.5031</td><td>0.6927</td></tr><tr><td>Theoretical ASR at q = 0.005</td><td>0.5000</td><td>0.5001</td><td>0.5012</td><td>0.6371</td></tr><tr><td>Numerical ASR</td><td> $0 . 5 0 0 5 \pm 0 . 0 0 2 5$ </td><td> $0 . 5 0 0 1 \pm 0 . 0 0 1 1$ </td><td> $0 . 5 0 7 2 \pm 0 . 0 0 0 7$ </td><td> $0 . 5 0 9 6 \pm 0 . 0 0 2 6$ </td></tr></table>

## M BROADER IMPACT STATEMENT

This work studies privacy leakage that can arise during diffusion-based sampling and provides tools for quantifying how individual training records may influence generated outputs. Its primary positive impact is to support more privacy-aware deployment of generative models, especially in settings where training data may contain sensitive or proprietary information. By shifting attention from only differentially private training to sampling-time privacy accounting, the proposed framework may help practitioners design generative pipelines that better balance privacy and utility, for example through subsampling, early stopping, or hybrid use of public denoisers. This can make formal privacy analysis more accessible for large-scale diffusion systems where fully private training remains costly or impractical.

At the same time, the work has limitations and possible risks. The formal guarantees are derived for an empirical denoiser with tractable sensitivity, while modern deployed systems typically use neural denoisers whose exact data sensitivity remains difficult to characterize. As a result, the analysis should not be interpreted as a complete privacy certificate for arbitrary diffusion models. If misused or overstated, sampling-time privacy estimates could give a false sense of security. Moreover, improving the privacy–utility tradeoff of generative models may also make it easier to deploy high-quality synthetic data systems in sensitive domains, where failures could still expose private information or amplify biases present in the underlying data.

We therefore view this work as a step toward, rather than a substitute for, end-to-end privacy evaluation. Responsible use should combine sampling-time accounting with empirical privacy audits, membership-inference evaluation, dataset governance, and clear reporting of assumptions. More broadly, our analysis highlights that privacy in generative modeling is shaped not only by training, but also by how samples are produced, released, and reused.