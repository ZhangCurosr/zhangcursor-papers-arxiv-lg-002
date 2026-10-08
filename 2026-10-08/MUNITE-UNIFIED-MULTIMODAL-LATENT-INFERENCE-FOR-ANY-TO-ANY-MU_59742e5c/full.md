# MUNITE: UNIFIED MULTIMODAL LATENT INFERENCE FOR ANY-TO-ANY MULTIMODAL GENERATION

Kyeongmin Yeo Minhyuk Sung

KAIST

{aaaaa, mhsung}@kaist.ac.kr

## ABSTRACT

We introduce MUNITE, a latent-variable framework for flexible any-to-any multimodal generation that treats encoding and latent generation as the same inference problem under different amounts of observed evidence. Given any subset of modalities, MUNITE models the conditional distribution over the latent representation associated with the complete observation. Full observation recovers deterministic encoding, no observation recovers the latent marginal, and intermediate subsets define conditional latent inference, all within a single conditional flow model. A shared latent sample captures variation that must remain consistent across generated targets, while modality-specific generative decoders model the remaining uncertainty independently. To learn these conditional distributions from incomplete training examples, we extend conditional flow matching through self-distillation: predictions conditioned on richer available observations supervise the same model conditioned on smaller subsets at the same intermediate latent state. When the richer-evidence trajectory follows the exact conditional flow, this provides the same expected learning signal as full-target denoising. Across PolyMNIST-D-Q, FFHQ64, and image–text–audio, MUNITE achieves competitive or better generation quality and source–target alignment, with higher joint-generation coherence. In particular, it attains the highest coherence in all one-to-many and unconditional image–text–audio comparisons, showing the effectiveness of unified latent inference across diverse multimodal settings. Project page: munite-proj.github.io.

## 1 INTRODUCTION

Amid rapid progress in large foundation models for multimodal generation, an important complementary direction is to develop flexible any-to-any multimodal generative models without requiring a large language model as the multimodal backbone. Many practical applications may depend on only a subset of the capabilities offered by general-purpose language models, often without advanced text reasoning, coding, or mathematical problem solving, and in some cases with little or no text processing. Such applications call for models specialized to the modalities at hand that still support arbitrary combinations of observed and generated modalities.

A natural alternative is to build multimodal generation around a shared latent space. Latent space modeling has long been successful for images and videos (Rombach et al., 2022; Blattmann et al., 2023), has recently been explored for language generation through diffusion models (Lovelace et al., 2023), and is now emerging as a promising basis for any-to-any multimodal generation (Yeo et al., 2026; Cha et al., 2026). Recent approaches further blur the boundary between representation learning and generation by jointly learning modality encoders, decoders, and the latent generative model, rather than fixing the latent representation before training the generator (Heek et al., 2026; Yeo et al., 2026). This suggests a broader possibility: instead of treating multimodal encoding and latent generation as separate components, can they be viewed as instances of the same underlying modeling problem?

We answer this question by unifying these roles within a single conditional model. Our key observation is that full-observation encoding, partial-observation latent inference, and unconditional latent generation correspond to different amounts of evidence about the same latent representation (Figure 1). Given an arbitrary subset of modalities, our model learns the conditional distribution over latent representations of complete observations. When all modalities are observed, this distribution collapses to a point mass, recovering deterministic multimodal encoding. When no modality is observed, it becomes the marginal latent distribution, recovering unconditional generation. Every intermediate subset corresponds to conditional latent inference, so encoding, conditional inference, and unconditional generation form a spectrum indexed by the observed modality subset. This view generalizes the Generative Encoder of UNITE (Duggal et al., 2026), whose single network covers only the two endpoints of full and absent observation, to every level of partial observation. The inference model is coupled with modality-specific generative decoders through a shared latent: one latent sample captures the variation that must be shared across generated targets, while the decoders model the remaining modality-specific uncertainty.

This formulation also creates a natural mechanism for self-distillation across observation levels. A partially observed training example does not provide the latent representation of its complete observation as a direct training target. Instead, the model conditioned on a richer set of available modalities provides a clean-latent target for the same model conditioned on a smaller subset, evaluated at the same intermediate flow state. Un der the assumptions developed in Section 2.3, we show that this self-distillation objective yields the same expected gradient as denoising against the unavailable full-observation latent. This allows partially observed examples to contribute directly to learning the desired conditional distribution.

We evaluate MUNITE on PolyMNIST-D-Q,

![](images/d425d532e159b937d22a81c98498811e86b6ca1469c1d67953a0cbd91f5a1623.jpg)  
Figure 1: Latent inference and decoding. Left: latent distributions under different observation subsets. Right: factorized decoding from a latent value.

FFHQ64, and image–text–audio generation, which together span categorical attributes, dense visual modalities, and heterogeneous text, image, and audio, with training data that mix fully and partially observed examples. MUNITE attains the best unconditional fidelity and coherence on PolyMNIST-D Q and FFHQ64 and the highest coherence in every one-to-many and unconditional image–text–audio comparison, while remaining competitive on one-to-one and many-to-one generation. These results show that a single conditional latent-inference model can support the full range of observation regimes in any-to-any generation, improving joint coherence while retaining the benefits of modality-specific generative decoders.

## 2 METHOD

MUNITE learns any-to-any generation from a mixture of fully and partially observed multimodal data. We first introduce a single latent inference model that, for any observed subset of modalities, captures the conditional distribution over latent representations of complete observations. We pair it with modality-specific generative decoders, so that the shared latent carries cross-modal dependencies and the decoders model the remaining uncertainty (Section 2.1). We learn the latent representation through cross-modal reconstruction (Section 2.2) and its evidence-conditioned distributions through self-distillation, which extends conditional flow matching to partially observed examples (Section 2.3). An auxiliary contrastive loss further encourages alignment across observation subsets (Section 2.4).

## 2.1 SHARED-LATENT GENERATIVE FRAMEWORK

Let $\mathcal { M } = \{ 1 , \dots , M \}$ index the modalities and $X ^ { \mathcal { M } } \sim P _ { \mathrm { d a t a } }$ denote a complete observation. We learn a latent representation W of complete observations together with a single model E that operates on an arbitrary observed subset ${ \mathcal { S } } \subseteq { \mathcal { M } } .$ . A partial observation $x ^ { s }$ generally admits multiple possible completions $\dot { x } ^ { \mathcal { M } }$ , which in turn induce a conditional distribution $Q \overline { { \varepsilon } } ( \cdot \mid x ^ { S } )$ over their latent representations.

At the two extremes, the same model E reduces to familiar operations. When $\textstyle S = { \mathcal { M } } , { \mathcal { E } }$ acts as a deterministic encoder, yielding a unique latent representation $\mathbf { \bar { \Sigma } } { w } = \mathcal { E } ( \mathbf { \Sigma } { x } ^ { \mathcal { M } } ) \in \mathbb { R } ^ { D }$ . Accordingly, the conditional distribution collapses to $Q \varepsilon ( \cdot \mid x ^ { \mathcal { M } } ) = \delta _ { w }$ . When $s = \emptyset$ , the model instead generates from the latent marginal $Q \varepsilon ( \mathrm { d } w )$ . For an intermediate subset $\emptyset \subsetneq S \subsetneq M$ , it models $Q \varepsilon ( \mathrm { d } w \mid x ^ { S } )$ (Figure 1, left). We parameterize $\dot { \varepsilon }$ as a conditional flow model that transports a standard Gaussian to the corresponding distribution $Q \varepsilon ( \cdot \mid x ^ { S } )$ for any observed subset S.

A generative-encoder perspective. Motivated by the success of latent diffusion models (Rombach et al., 2022), Unified Latents (Heek et al., 2026) proposed jointly learning latent representations of the data and a generative model of their distribution. UNITE (Duggal et al., 2026) took this idea further by introducing a Generative Encoder that uses the same model to encode an observed image and, when no image is observed, to generate samples from the latent distribution. Multimodal data add a natural intermediate regime in which only a subset of modalities is observed, and MUNITE extends the generative-encoder view to it by modeling the conditional distribution of the full-observation latent given an arbitrary observed subset. Full-observation encoding and unconditional latent generation then become the endpoints ${ \mathcal { S } } = { \mathcal { M } }$ and $s = \emptyset$ of a family of latent-inference problems $\{ Q _ { \mathcal { E } } ( \cdot \mid x ^ { S } ) \} _ { \mathcal { S } \subseteq \mathcal { M } }$

To generate multimodal observations, we pair the latent inference model $\mathcal { E }$ with modality-specific generative decoders $\{ \mathcal { D } _ { \psi _ { m } } \} _ { m \in \mathcal { M } }$ . As in multimodal VAEs (Wu & Goodman, 2018; Shi et al., 2019), the decoders generate independently conditioned on a shared latent. For this factorization to preserve the dependencies in the data, we require $W$ to be coherence-sufficient (Yeo et al., 2026):

$$
P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { \mathcal { M } } \mid W = w ) = \prod _ { m \in \mathcal { M } } P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { m } \mid W = w ) .\tag{1}
$$

The latent thus accounts for cross-modal dependence, leaving the remaining modality-specific variation to the decoders. For disjoint input and target sets $s , \tau \subseteq \mathcal { M }$ , marginalizing over the latent gives

$$
P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { \mathcal { T } } \mid x ^ { S } ) = \int \prod _ { m \in { \mathcal { T } } } P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { m } \mid W = w ) Q _ { \mathcal { E } } ( \mathrm { d } w \mid x ^ { S } ) .\tag{2}
$$

We train each decoder to model the conditional distribution of its modality given W. At inference, we use E to sample one latent w conditioned on $x ^ { S }$ and generate the target modalities in parallel with $\{ \mathcal { D } _ { \psi _ { m } } ( w ) \} _ { m \in \mathcal { T } }$

## 2.2 LEARNING THE SHARED REPRESENTATION

We learn the shared representation by reconstructing observed modalities through their generative decoders. We write $\dot { \mathcal { E } _ { \theta } } ( w _ { t } , t ; x ^ { S } )$ for the clean-latent prediction of the model E with parameters θ. For a training example $x ^ { A }$ , let the nonempty set ${ \mathcal { A } } \subseteq { \mathcal { M } }$ denote its available modalities. For reconstruction, we sample a target modality $m \in { \mathcal { A } }$ and a conditioning subset ${ \mathcal { S } } \subseteq A$ as described in Appendix C.2. Given a Gaussian query $\epsilon \sim \mathcal { N } ( 0 , I _ { D } )$ at $t = 0$ , the reconstruction objective is

$$
\mathcal { L } _ { \mathrm { r e c } } = \mathbb { E } _ { ( A , x ^ { A } ) , \mathcal { S } , m , \epsilon } \left[ \ell _ { m } \left( \mathcal { D } _ { \psi _ { m } } \big ( \mathcal { E } _ { \theta } ( \epsilon , 0 ; x ^ { S } ) \big ) , x ^ { m } \right) \right] .\tag{3}
$$

Here $\ell _ { m }$ denotes the training objective of decoder $m$ , such as cross-entropy for text and labels or a denoising objective for diffusion and flow decoders. With ${ \mathcal { S } } = { \mathcal { M } }$ , this trains each decoder on the full-observation latents required by Equation 2, while smaller subsets further stabilize the learning of the shared representation.

To favor information useful across modalities, we use target-detached self-reconstruction (Yeo et al., 2026). When the target is also an input $( m \in S )$ , it still contributes to the latent, but the reconstruction gradients through its token stream are stopped. Consequently, the target’s own reconstruction loss cannot train its token stream to pass target-private information to the latent. We describe the implementation in Section 2.5.

## 2.3 CONDITIONAL FLOW MATCHING WITH PARTIAL OBSERVATIONS

We train ${ \mathcal { E } } _ { \theta }$ to model $Q \varepsilon _ { \theta } ( \cdot \mid x ^ { S } )$ by conditional flow matching along a linear interpolant between standard Gaussian noise and clean latents. Since the clean target can be computed only from a complete example, we extend the objective to partial observations through self-distillation.

![](images/74e491b1f6cc9195dba4d79b9342cc1848bbb6515995bf79cfcd7b75687fa811.jpg)  
Figure 2: MUNITE training objectives. (a) Reconstruction learns a shared representation, using target detaching to discourage copying modality-private information. (b) Self-distillation uses detached predictions conditioned on more modalities to supervise subset-conditioned predictions at the same noisy latent state, with both branches using the same model. (c) An auxiliary contrastive loss encourages semantic alignment between complementary modality subsets using in-batch negatives.

Fully observed examples. The full-observation prediction provides the clean target $w =$ $\mathcal { E } _ { \theta } ( \dot { \epsilon } , 0 ; x ^ { \mathcal { M } } )$ , where $\epsilon \sim \mathcal { N } ( 0 , I _ { D } )$ . We use the same Gaussian sample as the initial flow state, setting $w _ { 0 } = \epsilon$ , and sample $t \sim \mathcal { U } [ 0 , 1 )$ . The interpolant and the velocity field implied by the clean prediction are

$$
w _ { t } = ( 1 - t ) w _ { 0 } + t \mathrm { s g } ( w ) , \qquad v _ { \theta } ( w _ { t } , t ; x ^ { S } ) = \frac { \mathcal { E } _ { \theta } ( w _ { t } , t ; x ^ { S } ) - w _ { t } } { 1 - t } ,\tag{4}
$$

where sg denotes stop-gradient. Writing $W , W _ { 0 } , W _ { t }$ for the corresponding random variables, the conditional flow matching objective (Lipman et al., 2023) becomes

$$
\mathcal { L } _ { \mathrm { C F M } } = \mathbb { E } \big [ \| v _ { \theta } ( W _ { t } , t ; X ^ { \mathcal { S } } ) - ( \mathrm { s g } ( W ) - W _ { 0 } ) \| _ { 2 } ^ { 2 } \big ] = \mathbb { E } \left[ \frac { \| \mathcal { E } _ { \theta } ( W _ { t } , t ; X ^ { \mathcal { S } } ) - \mathrm { s g } ( W ) \| _ { 2 } ^ { 2 } } { ( 1 - t ) ^ { 2 } } \right] .\tag{5}
$$

Clean-latent prediction allows encoding and denoising to share their output space, as in JiT and UNITE (Li & He, 2026; Duggal et al., 2026). For optimization stability, we drop the factor $( 1 - t ) ^ { - 2 }$ and train with the unweighted denoising loss

$$
\begin{array} { r } { \widetilde { \mathcal { L } } _ { \mathrm { C F M } } = \mathbb { E } \big [ \| \mathcal { E } _ { \theta } ( W _ { t } , t ; X ^ { \mathcal { S } } ) - \mathrm { s g } ( W ) \| _ { 2 } ^ { 2 } \big ] . } \end{array}
$$

At a fixed encoder and time t, both weightings have the population-optimal prediction (Lipman et al., 2023)

$$
\mu _ { S } ( w _ { t } , t ; x ^ { S } ) : = \mathbb { E } [ W \mid W _ { t } = w _ { t } , X ^ { S } = x ^ { S } ] .\tag{6}
$$

With full observations, this reduces to $\mathcal { E } ( x ^ { \mathcal { M } } )$ , recovering encoding as a one-step prediction (Section 2.1). As in UNITE, the Gaussian query keeps the input format common to all observation levels; since the reconstruction target does not depend on it, the trained models effectively ignore it (Appendix A.1).

Partially observed examples. When ${ \mathcal { A } } \subsetneq { \mathcal { M } }$ , we cannot compute $\mathcal { E } ( x ^ { \mathcal { M } } )$ . The available modalities nevertheless constrain its conditional distribution. For nested subsets ${ \mathcal { S } } \subsetneq { \mathcal { A } } ,$ the tower property gives, at each fixed t,

$$
\mathbb { E } \big [ \mu _ { A } ( W _ { t } , t ; X ^ { A } ) \mid W _ { t } , X ^ { S } \big ] = \mu _ { S } ( W _ { t } , t ; X ^ { S } ) .\tag{7}
$$

A clean prediction using $x ^ { A }$ can therefore supervise the prediction using $x ^ { S }$ , provided their common noisy state follows the full-latent intermediate distribution. We approximate it by integrating ${ \mathcal { E } } _ { \theta }$ conditioned on all available modalities up to the sampled time, using N Euler steps with $\bar { N } = \bar { 4 }$ for training efficiency:

$$
\frac { \mathrm { d } w _ { u } ^ { A } } { \mathrm { d } u } = v _ { \theta } ( w _ { u } ^ { A } , u ; x ^ { A } ) , \qquad w _ { 0 } ^ { A } = w _ { 0 } , \qquad 0 \leq u \leq t .\tag{8}
$$

At the resulting state, the same model serves as the teacher, conditioned on $x ^ { A }$ , and as the student, conditioned on $x ^ { s }$ , giving the unweighted distillation objective

$$
\begin{array} { r } { \widetilde { \mathcal { L } } _ { \mathrm { d i s t } } = \mathbb { E } \left[ \left| \left| \mathcal { E } _ { \theta } ( \mathrm { s g } ( W _ { t } ^ { \mathcal { A } } ) , t ; X ^ { \mathcal { S } } ) - \mathrm { s g } \left( \mathcal { E } _ { \theta } ( W _ { t } ^ { \mathcal { A } } , t ; X ^ { \mathcal { A } } ) \right) \right| \right| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{9}
$$

![](images/261db37dcc4f00ac026e4d1ef71c5144066af3741933d790f9cae0ee3f3863f0.jpg)  
Stop target K/V gradients during reconstruction  
Figure 3: Latent inference architecture. A shared Transformer performs latent inference under arbitrary observation subsets by aggregating modality information in latent registers. Keeping modality streams separate allows reconstruction gradients to be stopped on the target’s key/value paths to the registers (slashes), while retaining its information in the forward pass.

For a complete example, the teacher target is its encoding w and the state is the analytic interpolant in Equation 4, recovering $\widetilde { \mathcal { L } } _ { \mathrm { C F M } }$

Proposition 1 (Gradient equivalence). Let ${ \mathcal { S } } \subsetneq { \mathcal { A } } ,$ , let $x ^ { A }$ be drawn from $P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { A } )$ , and treat the target latent distribution and teacher as fixed. If the teacher predicts $\mu _ { \mathcal { A } }$ and its trajectory reproduces the conditional time marginals of $W _ { t } ,$ then $\widetilde { \mathcal { L } } _ { \mathrm { d i s t } }$ and $\widetilde { \mathcal { L } } _ { \mathrm { C F M } }$ yield identical expected gradients through the subset-conditioned prediction.

The proof follows from the orthogonality of conditional expectation (Appendix B). While the equivalence holds at a fixed representation with an exact teacher, in practice the current model serves as the teacher and its conditional flow is integrated numerically.

## 2.4 CONTRASTIVE ALIGNMENT ACROSS OBSERVATION SUBSETS

Reconstruction and denoising train $\mathcal { E }$ to retain cross-modal information and to model its conditional distributions, but neither directly shapes the latent geometry to reflect semantic similarity. We therefore add an auxiliary contrastive objective that promotes semantic alignment across observation subsets. For a batch of $\dot { B }$ examples with available set ${ \mathcal { A } } ,$ choose a nonempty proper subset ${ \mathcal { S } } \subsetneq { \mathcal { A } }$ and its complement ${ \mathcal { C } } = { \mathcal { A } } \backslash { \mathcal { S } }$ . Let $r _ { i } ^ { S }$ and $r _ { i } ^ { \mathcal { C } }$ be the flattened, $\ell _ { 2 } \cdot$ -normalized outputs of $\mathcal { E } _ { \theta } ( \epsilon _ { i } ^ { S } , 0 ; \mathcal { x } _ { i } ^ { S } )$ and $\mathcal { E } _ { \theta } ( \epsilon _ { i } ^ { \mathcal { C } } , 0 ; x _ { i } ^ { \mathcal { C } } )$ , using independent Gaussian queries. With temperature $\tau _ { c } > 0$ , the directional InfoNCE loss (van den Oord et al., 2018) is

$$
\mathcal { L } _ { \mathcal { S }  \mathcal { C } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( ( r _ { i } ^ { \mathcal { S } } ) ^ { \top } r _ { i } ^ { \mathcal { C } } / \tau _ { c } ) } { \sum _ { j = 1 } ^ { B } \exp ( ( r _ { i } ^ { \mathcal { S } } ) ^ { \top } r _ { j } ^ { \mathcal { C } } / \tau _ { c } ) } .\tag{10}
$$

We average the two directions, $\mathcal { L } _ { \mathrm { c o n } } = ( \mathcal { L } _ { S  c } + \mathcal { L } _ { C  S } ) / 2$ . As in Section 2.2, these one-step predictions from partial observations are not samples of the target latent distribution, but because they share the backbone parameters, aligning them stabilizes the internal representation of E.

## 2.5 MODEL ARCHITECTURE

We implement E as a shared Transformer that predicts a clean latent from a latent state and an arbitrary subset of observed modalities. Figure 3 illustrates the architecture. Observed modalities are represented as separate token streams, while K latent registers receive the Gaussian query or noisy latent state and aggregate information from the observed streams at every block. The final register states are projected to the clean latent prediction.

Within each block, modality streams attend only within themselves, whereas the latent registers attend to all observed streams and to one another, following the latent-attention construction of Perceiver (Jaegle et al., 2021). This separation prevents direct mixing between modality streams and allows their features to be reused across denoising steps.

For reconstruction target m, we detach its projected keys and values on every modality-to-register attention edge. The forward computation is unchanged, but reconstruction gradients can no longer flow from the registers into the target stream. Since modality streams do not attend to one another, no other modality offers an indirect path. This realizes the target-detached reconstruction of Section 2.2 without target-specific encoders. Appendix C.1 gives further implementation details.

## 2.6 TRAINING AND INFERENCE

We train E and the modality-specific generative decoders with

$$
\mathcal { L } = \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { d i s t } } \widetilde { \mathcal { L } } _ { \mathrm { d i s t } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } .\tag{11}
$$

The reconstruction loss updates E and the decoder of the sampled target, while the distillation and contrastive losses update only E. Algorithm 1 in Appendix C.2 gives the full training procedure.

At inference, one latent drawn by integrating v<sub>θ</sub> from Gaussian noise is shared by all target decoders, as the factorization in Equation 2 requires.

## 2.7 RELATION TO MULTIMODAL LATENT-VARIABLE MODELS

Among multimodal VAEs, which likewise decode modalities independently from a shared latent (Wu & Goodman, 2018; Shi et al., 2019), JNF (Senellart & Allassonnière, 2025) is particularly close to our formulation: it first learns a shared latent with a joint VAE trained on complete observations, then fits a separate normalizing-flow posterior for each modality to samples from the joint encoder. Writing these posteriors as $q _ { m } \bar { ( } z \mid x ^ { m } )$ , inference from a subset S is based on their product,

$$
q _ { \mathrm { J N F } } ( z \mid x ^ { S } ) \propto p ( z ) ^ { 1 - | S | } \prod _ { m \in S } q _ { m } ( z \mid x ^ { m } ) ,\tag{12}
$$

from which JNF samples using HMC.

MUNITE instead directly models $Q \varepsilon ( \mathrm { d } w ~ \mid ~ x ^ { S } )$ , the conditional distribution of the same fullobservation latent for every S, without composing modality-specific posteriors. Whereas JNF builds on a joint encoder trained on complete observations, MUNITE can also learn these conditionals from partially observed examples: the prediction under richer available evidence serves as a self-distillation target, justified by Equation 7 and Proposition 1.

MUNI (Yeo et al., 2026) shares our use of factorized generative decoders, target-detached reconstruction, and a single latent sample for all targets, but draws this sample from an aggregated posterior $q _ { \phi , S } ( z \mid x ^ { S } )$ for an observed subset. The central condition of its predictive-sufficiency criterion is

$$
I ( X ^ { \cal S } ; X ^ { m } \mid Z _ { \cal S } ) = 0 \qquad \forall m \not \in { \cal S } ,\tag{13}
$$

which guarantees target-wise sufficiency but does not imply conditional independence among multiple missing targets. MUNITE instead samples $W \sim Q _ { \mathcal { E } } ( \cdot \mid \bar { x } ^ { \mathcal { S } } )$ , where the full-observation latent $\bar { W }$ is trained to satisfy Equation 1. The missing variation shared across targets is therefore sampled once in W, and marginalizing this latent yields the conditional joint in Equation 2. We formalize this distinction in Appendix A.3.

## 3 RELATED WORK

Since Section 2.1 covers latent diffusion, Unified Latents, and UNITE (Rombach et al., 2022; Heek et al., 2026; Duggal et al., 2026), we focus here on multimodal generation.

Any-to-any generation. Any-to-any generation extends cross-modal synthesis beyond fixed modality pairs to arbitrary combinations of observed and target modalities (Tang et al., 2023; Bachmann et al., 2024). Masked models such as 4M and 4M-21 learn these mappings by predicting targetmodality tokens from observed-modality tokens with a shared Transformer (Mizrahi et al., 2023; Bachmann et al., 2024). Joint flow models instead couple modality trajectories during sampling, using continuous representations in OmniFlow and discrete tokens in NExT-OMNI (Li et al., 2025; Luo et al., 2026); our CFM and DFM baselines implement these two approaches in a controlled setting (Section 4). CoDi similarly couples modality-specific denoisers through cross-attention to aligned representations of the other modalities’ noisy latents (Tang et al., 2023).

Shared-latent generative models. Multimodal VAEs model multiple modalities through a shared latent variable and modality-specific encoders and decoders (Wu & Goodman, 2018; Shi et al., 2019; Sutter et al., 2021; Palumbo et al., 2023; Vo & Valera, 2026). Building on this factorization, MUNI jointly trains modality-specific encoders, expressive generative decoders, and a shared flow prior, and introduces coherence sufficiency, predictive sufficiency, and minimality as criteria for the learned multimodal latent (Yeo et al., 2026). FlowBind takes a different latent-space approach, using bidirectional flows to connect modality representations through a common latent distribution learned from partially paired data (Cha et al., 2026). MUNITE instead directly models the conditional distribution of a coherence-sufficient full-observation latent under arbitrary observed subsets, replacing the diagonal-Gaussian parameterization of subset posteriors with conditional transport.

## 4 EXPERIMENTS

We evaluate one-to-one, many-to-one, one-to-many, and many-to-many generation, as well as unconditional co-generation. Our central question is whether jointly generated targets agree on content that their inputs leave unspecified. We adapt the PolyMNIST–digit–quadrant (PolyMNIST-D-Q) and image–text–audio benchmarks of MUNI (Yeo et al., 2026) and introduce a new FFHQ64 benchmark with three image-like and two categorical modalities.

## 4.1 EXPERIMENTAL SETUP

We compare MUNITE with MUNI (Yeo et al., 2026) and joint continuous and discrete flow-matching baselines, CFM and DFM (Lipman et al., 2023; Gat et al., 2024). We train MUNI with its released training and sampling pipelines and retain its modality-specific decoder architectures in MUNITE. To compare shared-latent and joint generation, CFM and DFM implement the continuous and discrete flow-matching approaches used in OmniFlow and NExT-OMNI, respectively (Li et al., 2025; Luo et al., 2026). Both baselines use the same Transformer blocks as MUNITE but apply joint attention across all modality tokens. Since they do not use generative decoders, we increase the backbone depth to approximately match MUNITE’s total number of trainable parameters. Because fully Transformerbased joint models may converge more slowly than latent models with generative decoders, we also train CFM and DFM for twice as many epochs as MUNITE (Appendix D.2). CFM represents categorical modalities with one-hot vectors, whereas DFM discretizes continuous modalities with separately trained, frozen VQ-VAEs (van den Oord et al., 2017). All four methods share the same training data, evaluation protocol, and frozen evaluators.

For the image–text–audio task, we additionally compare against FlowBind (Cha et al., 2026) and pretrained CoDi and OmniFlow (Tang et al., 2023; Li et al., 2025). Because CoDi and OmniFlow are pretrained on substantially larger datasets, we use their released models without retraining, take their single-target and unconditional scores from MUNI (Yeo et al., 2026), and evaluate their one-to-many coherence ourselves (Appendix E.3). The released FlowBind checkpoint is trained on less data than our benchmark, so we retrain it on our training data with its official pipeline.

The training data mix fully and partially observed examples (Table 4 in Appendix D). For PolyMNIST-D-Q and FFHQ64, we mask one or two of the three non-categorical modalities in three quarters of the training examples and keep the categorical labels. For image–text–audio, the corpora used by MUNI and FlowBind contain only modality pairs, so we add caption–image–audio triplets from E-MM1 (Broadbent et al., 2025). For evaluation, we use 10,000 complete PolyMNIST tuples, 5,000 complete FFHQ tuples, and the image–text–audio evaluation sets of MUNI (Yeo et al., 2026). Appendix F shows qualitative results.

Table 1: Results on PolyMNIST-D-Q (top) and FFHQ64 (bottom). Acc. and Coh. denote accuracy and coherence, d and q denote digit and quadrant, and scores for m are averaged over the three views. FD uses verifier features, and Normal Err. is 1 − cos. Best results are in bold.
<table><tr><td>Route</td><td>Input → output</td><td>Metric</td><td>MUNI</td><td>CFM</td><td></td><td>DFM MUNITE</td></tr><tr><td colspan="7">PolyMNIST-D-Q</td></tr><tr><td rowspan="4">Uncond. All five modalities</td><td rowspan="4"></td><td>FD↓</td><td>3.0407</td><td>71.6626</td><td>19.9587</td><td>1.5230</td></tr><tr><td>Digit Coh. (all) ↑</td><td>0.5799</td><td>0.0237</td><td>0.2973</td><td>0.9826</td></tr><tr><td>Quad. Coh. (ail) ↑</td><td>0.9041</td><td>0.9902</td><td>0.8222</td><td>1.0000</td></tr><tr><td>Digit Acc. ↑</td><td>0.9843</td><td>0.2402</td><td></td><td></td></tr><tr><td rowspan="2">1 → 1</td><td>d → mi q → mi</td><td>Quad. Acc. ↑</td><td>1.0000</td><td>0.9936</td><td>0.7224 0.9372</td><td>0.9941 1.0000</td></tr><tr><td>N → 1 (d, q) → mi</td><td>Both-label Acc. ↑</td><td>0.9925</td><td>0.2200</td><td>0.7325</td><td></td></tr><tr><td rowspan="2">1 → N</td><td> $d  ( m _ { 0 } , m _ { 1 } , m _ { 2 } )$ </td><td></td><td></td><td></td><td></td><td>0.9943</td></tr><tr><td> $q  ( m _ { 0 } , m _ { 1 } , m _ { 2 } )$ </td><td>Quad. Coh. ↑ Digit Coh. ↑</td><td>0.2528 0.1074</td><td>0.9879 0.1597</td><td>0.8798 0.4612</td><td>1.0000 0.9904</td></tr><tr><td colspan="2"></td><td>FFHQ64</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">Uncond. All five modalities</td><td rowspan="5"></td><td>RGB FID ↓</td><td>34.618</td><td>122.617</td><td>55.840</td><td>29.918</td></tr><tr><td>Age Coh. ↑</td><td>0.5630</td><td>0.3068</td><td>0.4546</td><td>0.5692</td></tr><tr><td>Gender Coh. ↑</td><td>0.9596</td><td>0.7328</td><td>0.8658</td><td>0.9610</td></tr><tr><td>RGB-Seg. Coh. ↑</td><td>0.7762</td><td>0.7357</td><td>0.8084</td><td>0.8896</td></tr><tr><td>RGB-Normal Err. ↓</td><td>0.0700</td><td>0.0603</td><td>0.0495</td><td>0.0491</td></tr><tr><td rowspan="2">1 → 1</td><td rowspan="2">Age → RGB Gender → RGB</td><td>Age Acc. ↑</td><td>0.5204</td><td>0.2506</td><td>0.4002</td><td>0.6274</td></tr><tr><td>Gender Acc. ↑</td><td>0.9684</td><td>0.6944</td><td>0.8518</td><td>0.9790</td></tr><tr><td>N → 1</td><td>(Age, Gender) → RGB</td><td>Both Acc. ↑</td><td>0.5100</td><td>0.1842</td><td>0.3690</td><td>0.5898</td></tr><tr><td rowspan="2">M → N</td><td rowspan="2">(Age, Gender) → (RGB, Seg., Normals)</td><td>RGB-Seg. Coh. ↑</td><td>0.6038</td><td>0.7441</td><td>0.8142</td><td>0.8888</td></tr><tr><td>RGB-Normal Err. ↓</td><td>0.1106</td><td>0.0572</td><td>0.0492</td><td>0.0500</td></tr></table>

## 4.2 POLYMNIST-D-Q

The three 64 × 64 RGB views share digit identity and quadrant but differ in handwriting and background. Conditioning on either label leaves the other attribute unobserved, so joint generation requires the three views to agree on that missing attribute. For unconditional co-generation, coherence measures agreement among the three images and the jointly generated labels. Following MUNI (Yeo et al., 2026), we compute digit and quadrant accuracy and coherence using a frozen SegFormer verifier (Appendix D.4), while Fréchet distance (FD) in verifier-feature space measures image fidelity.

MUNITE retains strong single-image label accuracy while substantially improving agreement on the unobserved attribute (Table 1, top). Under quadrant conditioning, MUNI and MUNITE both reach 1.0000 quadrant accuracy for single images, yet the digit coherence of their jointly generated views is 0.1074 and 0.9904, respectively. Under digit conditioning, quadrant coherence likewise rises from 0.2528 to 1.0000. MUNI’s scores are near the chance agreement of independent views, 0.10 and 0.25, so accurate single-view conditionals alone do not make the views agree (Section 2.7). Each MUNITE view still follows the data distribution of the missing attribute, and the views agree only by chance when decoded from separate latent samples (Appendix E.1).

MUNITE also achieves the lowest FD and the highest unconditional coherence for both labels. CFM and DFM attain high quadrant agreement on some routes, yet lag behind both latent models in digit accuracy and image fidelity despite matched trainable parameters and twice as many training epochs. Their joint backbones must model the handwriting and background of each view together with the shared attributes, which both latent models leave to image-specific decoders.

## 4.3 FFHQ64

We construct a five-modality task from FFHQ (Karras et al., 2019), combining 64 × 64 RGB faces with the age, gender, and segmentation annotations of FFHQ-Aging (Or-El et al., 2020) and pseudolabeled surface normals. We evaluate RGB generation from either or both categorical attributes, joint generation of RGB, segmentation, and normals from both, and unconditional co-generation of all five modalities. Attribute accuracy measures agreement with the observed labels, and coherence measures agreement among jointly generated modalities. RGB FID measures image fidelity, and normal error the geometric consistency between generated normals and RGB.

On FFHQ64, the gains extend from shared labels to dense cross-modal structure (Table 1, bottom). When conditioning on age and gender, MUNITE increases RGB–segmentation coherence from 0.6038 to 0.8888 and reduces normal error from 0.1106 to 0.0500 relative to MUNI. DFM attains a slightly lower normal error (0.0492 vs. 0.0500), but MUNITE attains much higher segmentation coherence (0.8888 vs. 0.8142). MUNITE also has the lowest unconditional RGB FID and the strongest unconditional coherence on all four cross-modal comparisons. CFM and DFM remain substantially weaker in RGB fidelity, as in PolyMNIST-D-Q.

## 4.4 IMAGE–TEXT–AUDIO

We follow the image–text–audio benchmark used by FlowBind and MUNI (Cha et al., 2026; Yeo et al., 2026), modeling fixed 768-dimensional features and evaluating the rendered images, captions, and audio. For one-to-one generation, we use FID, FAD, and CIDEr for fidelity and CLIP, CLAP, and Audio-Image Similarity (AIS) for cross-modal alignment. For many-to-one generation, we measure the alignment of the generated target with each source modality, and for one-to-many and unconditional co-generation, the alignment among jointly generated modalities, which we report as coherence.

Table 2: Image–text–audio coherence between jointly generated modalities. T, I, and A denote text, image, and audio. OmniFlow’s unconditional scores are from Yeo et al. (2026, Table 4), and dashes denote unsupported unconditional generation. Higher is better, and best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">1 → N coherence</td><td colspan="3">Unconditional coherence</td></tr><tr><td> $T \to ( I , A )$  AIS</td><td> $I  ( T , A )$  CLAP</td><td> $A  ( T , I )$  CLIP</td><td>CLIP (T, I)</td><td>CLAP (T, A)</td><td>AIS (I, A)</td></tr><tr><td>CoDi</td><td>63.866</td><td>7.418</td><td>23.414</td><td></td><td></td><td></td></tr><tr><td>OmniFlow</td><td>77.027</td><td>14.781</td><td>22.697</td><td>21.17</td><td>14.23</td><td>50.95</td></tr><tr><td>MUNI</td><td>81.558</td><td>32.237</td><td>25.850</td><td>26.949</td><td>28.467</td><td>81.989</td></tr><tr><td>CFM</td><td>74.702</td><td>21.695</td><td>24.034</td><td>23.622</td><td>19.896</td><td>74.208</td></tr><tr><td>DFM</td><td>73.635</td><td>20.951</td><td>24.037</td><td>23.740</td><td>19.503</td><td>71.942</td></tr><tr><td>FlowBind</td><td>79.819</td><td>25.812</td><td>24.853</td><td></td><td></td><td></td></tr><tr><td>MUNITE</td><td>83.336</td><td>34.762</td><td>26.334</td><td>27.398</td><td>30.220</td><td>87.045</td></tr></table>

MUNITE achieves the highest coherence in all three one-to-many directions and all three unconditional comparisons (Table 2). The improvement holds across text, image, and audio, indicating that the shared latent preserves dependencies among generated targets when their shared variation is not determined by the input. Ablating target detaching, contrastive alignment, or partial-data distillation leaves this one-to-many advantage over MUNI intact, whereas unconditional coherence relies most on target detaching and partial-data distillation (Appendix E.3.2). On the one-to-one and many-to-one routes, MUNITE is competitive with the compared baselines, including the externally pretrained models (Appendix E.3.1).

## 5 CONCLUSION AND FUTURE WORK

MUNITE unifies encoding and any-to-any multimodal generation as inference over a full-observation latent. Self-distillation extends the learning of its conditional distributions to incomplete training examples, and modality-specific decoders generate from a shared latent sample. Across the evaluated benchmarks, MUNITE improves coherence among jointly generated modalities while maintaining competitive single-target generation quality.

A natural next step is multimodal world modeling. The compact latent learned by MUNITE could serve as a shared world-state representation that can be inferred from arbitrary subsets of observations. Learning dynamics directly in this latent space would enable state estimation and future prediction under varying multimodal observations, while modality-specific decoders could render the predicted state into any desired modality.

## ETHICS STATEMENT

Our experiments include face and audiovisual generation, which may reproduce biases present in the underlying datasets and can be misused to create deceptive synthetic media. In the FFHQ64 benchmark, age and gender are used only as dataset annotations for controlled evaluation and should not be interpreted as validated personal attributes. Generated examples should therefore be interpreted as synthetic outputs rather than representations of real individuals or verified attributes.

## REPRODUCIBILITY STATEMENT

Section 2 specifies the learning objectives, gradient routing, and inference procedure, and Appendix B proves Proposition 1. Appendix C describes the architectures, the sampling of training subsets, and the training update in Algorithm 1. Appendix D details how each benchmark and its observation groups are constructed, together with model sizes, optimization and inference budgets, and the evaluators. Appendix E defines all metrics and gives the sources of the external baseline results.

## REFERENCES

Roman Bachmann, Oguzhan Fatih Kar, David Mizrahi, Ali Garjani, Mingfei Gao, David Griffiths,˘ Jiaming Hu, Afshin Dehghan, and Amir Zamir. 4M-21: An any-to-any vision model for tens of tasks and modalities. In NeurIPS, 2024.

Black Forest Labs. FLUX. https://github.com/black-forest-labs/flux, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable Video Diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Jim Broadbent, Felix Cohen, Frederik Hvilshøj, Eric Landau, and Eren Sasoglu. EBind: A practical approach to space binding. arXiv preprint arXiv:2511.14229, 2025.

Yeonwoo Cha, Semin Kim, Jinhyeon Kwon, and Seunghoon Hong. FlowBind: Efficient any-to-any generation with bidirectional flows. In ICLR, 2026.

Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. VGGSound: A large-scale audio-visual dataset. In ICASSP, 2020.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.

Shivam Duggal, Xingjian Bai, Zongze Wu, Richard Zhang, Eli Shechtman, Antonio Torralba, Phillip Isola, and William T. Freeman. End-to-end training for unified tokenization and latent denoising. arXiv preprint arXiv:2603.22283, 2026.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan de Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. In WACV, 2025.

Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. In NeurIPS, 2024.

Gemma Team. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Jonathan Heek, Emiel Hoogeboom, Thomas Mensink, and Tim Salimans. Unified Latents (UL): How to train your latents. arXiv preprint arXiv:2602.17270, 2026.

Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, R. Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron J. Weiss, and Kevin Wilson. CNN architectures for large-scale audio classification. In ICASSP, 2017.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In NeurIPS, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022.

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In ICML, 2021.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019.

Kevin Kilgour, Mauricio Zuluaga, Dominik Roblek, and Matthew Sharifi. Fréchet audio distance: A reference-free metric for evaluating music enhancement algorithms. In Interspeech, 2019.

Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. AudioCaps: Generating captions for audios in the wild. In NAACL, 2019.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

Khaled Koutini, Jan Schlüter, Hamid Eghbal-zadeh, and Gerhard Widmer. Efficient training of audio transformers with patchout. In Interspeech, 2022.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11):2278–2324, 1998.

Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Zichun Liao, Yusuke Kato, Kazuki Kozuka, and Aditya Grover. OmniFlow: Any-to-any generation with multi-modal rectified flows. In CVPR, 2025.

Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In CVPR, 2026.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D. Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. In ICML, 2023.

Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q. Weinberger. Latent diffusion for language generation. In NeurIPS, 2023.

Run Luo, Xiaobo Xia, Lu Wang, Longze Chen, Renke Shan, Jing Luo, Min Yang, and Tat-Seng Chua. NExT-OMNI: Towards any-to-any omnimodal foundation models with discrete flow matching. In ICLR, 2026.

David Mizrahi, Roman Bachmann, Oguzhan Fatih Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan,˘ and Amir Zamir. 4M: Massively multimodal masked modeling. In NeurIPS, 2023.

Roy Or-El, Soumyadip Sengupta, Ohad Fried, Eli Shechtman, and Ira Kemelmacher-Shlizerman. Lifespan age transformation synthesis. In ECCV, 2020.

Emanuele Palumbo, Imant Daunhawer, and Julia E. Vogt. MMVAE+: Enhancing the generative quality of multimodal VAEs without compromises. In ICLR, 2023.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In AAAI, 2018.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022.

Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Juyeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divyashree Sreepathihalli, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Qin Yin, Yunhsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, and Mojtaba Seyedhosseini. EmbeddingGemma: Powerful and lightweight text representations. arXiv preprint arXiv:2509.20354, 2025.

Christoph Schuhmann, Andreas Köpf, Theo Coombes, Richard Vencu, Benjamin Trom, and Romain Beaumont. LAION COCO: 600M synthetic captions from LAION2B-en. https://laion.ai/blog/ laion-coco/, 2022.

Agathe Senellart and Stéphanie Allassonnière. Bridging the inference gap in multimodal variational autoencoders. Journal ofData Science, Statistics, and Visualisation, 5(9), 2025.

Yuge Shi, N. Siddharth, Brooks Paige, and Philip H. S. Torr. Variational mixture-of-experts autoencoders for multi-modal deep generative models. In NeurIPS, 2019.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021a.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021b.

Stability AI. Stable unCLIP 2.1. https://huggingface.co/docs/diffusers/api/pipelines/stable\_unclip, 2023.

Thomas M. Sutter, Imant Daunhawer, and Julia E. Vogt. Generalized multimodal ELBO. In ICLR, 2021.

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the Inception architecture for computer vision. In CVPR, 2016.

Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion. In NeurIPS, 2023.

Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In CVPR, 2015.

Huyen Thuc Khanh Vo and Isabel Valera. Hellinger multimodal variational autoencoders. In AISTATS, 2026.

Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, and Juan Pablo Bello. Wav2CLIP: Learning robust audio representations from CLIP. In ICASSP, 2022.

Mike Wu and Noah Goodman. Multimodal generative models for scalable weakly-supervised learning. In NeurIPS, 2018.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP, 2023.

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. Seg-Former: Simple and efficient design for semantic segmentation with transformers. In NeurIPS, 2021.

Kyeongmin Yeo, Yunhong Min, and Minhyuk Sung. MUNI: Multimodal unified latent diffusion for coherent any-to-any generation. arXiv preprint arXiv:2606.16408, 2026.

Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions ofthe Associationfor Computational Linguistics, 2:67–78, 2014.

Yijie Zhang, Yiyang Shen, and Weiran Wang. Disentanglement of variations with multimodal generative modeling. In ICLR, 2026.

## APPENDIX

## A FULL-OBSERVATION LATENTS AND CONDITIONAL GENERATION

This section expands on the formulation of Section 2.1. We first characterize the latent conditionals induced by the full-observation encoder, then derive the conditional joint in Equation 2 and contrast it with the target-wise criterion of MUNI.

## A.1 THE DISTRIBUTION INDUCED BY FULL OBSERVATIONS

For a fixed representation, $W = { \mathcal { E } } ( X ^ { { \mathcal { M } } } )$ , so the conditional distributions in Section 2.1 are pushforwards of the data conditionals through the full-observation encoder:

$$
Q \varepsilon ( \cdot \mid x ^ { S } ) = { \mathcal { E } } _ { \# } \left[ P _ { \mathrm { d a t a } } ( \cdot \mid x ^ { S } ) \right] .\tag{14}
$$

With full observation this is $\delta _ { \mathcal { E } ( x ^ { \mathcal { M } } ) }$ , and empty conditioning gives the marginal distribution of $W$

Query dependence. As described in Section 2.3, the shared model also receives a Gaussian query under full observation, so the full-observation latent $\mathcal { E } _ { \theta } ( \epsilon , 0 ; x ^ { \mathcal { M } } )$ need not be exactly deterministic. We therefore measure its dependence on the query by the variance ratio

$$
r = \frac { \mathbb { E } _ { x ^ { \mathcal { M } } } \mathrm { t r } \mathrm { C o v } _ { \epsilon } \left[ \mathcal { E } _ { \theta } ( \epsilon , 0 ; x ^ { \mathcal { M } } ) \right] } { \mathrm { t r } \mathrm { C o v } _ { x ^ { \mathcal { M } } } \left[ \mathbb { E } _ { \epsilon } \mathcal { E } _ { \theta } ( \epsilon , 0 ; x ^ { \mathcal { M } } ) \right] } ,
$$

estimated from eight queries for each of 1,000 held-out complete observations (the 975 evaluation triplets for image–text–audio). The trained models give $r ^ { \cdot } = \ : 2 . 6 \times 1 0 ^ { - 5 }$ on PolyMNIST-D-Q, $7 . 9 \times 1 0 ^ { - 5 }$ on FFHQ64, and $6 . 5 \times 1 0 ^ { - 4 }$ on image–text–audio, so the full-observation latent is deterministic for practical purposes.

## A.2 DERIVATION OF THE CONDITIONAL JOINT

The coherence-sufficiency condition in Equation 1 makes all modalities conditionally independent given $W$ . Hence, for disjoint input and target sets S and $\tau .$

$$
P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { T } \mid W = w , x ^ { S } ) = \prod _ { m \in { \cal T } } P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { m } \mid W = w ) ,
$$

and integrating over the latent conditional in Equation 14 gives

$$
\begin{array} { l } { P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { \mathcal { T } } \mid x ^ { S } ) = \displaystyle \int P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { \mathcal { T } } \mid W = w , x ^ { S } ) Q _ { \mathcal { E } } ( \mathrm { d } w \mid x ^ { S } ) } \\ { \displaystyle \quad = \int \prod _ { m \in { \mathcal T } } P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { m } \mid W = w ) Q _ { \mathcal { E } } ( \mathrm { d } w \mid x ^ { S } ) , } \end{array}\tag{15}
$$

which is Equation 2. One sample of w must therefore be shared by all target decoders. Sampling a separate latent for each target would instead produce $\begin{array} { r } { \prod _ { m \in \mathcal { T } } \dot { P _ { \mathrm { d a t a } } } ( \mathrm { d } x ^ { \breve { m } } \mid x ^ { S } ) } \end{array}$ , discarding the dependence that remains after observing the inputs. Taking $s = \emptyset$ covers unconditional co-generation.

## A.3 PREDICTIVE SUFFICIENCY AND JOINT COHERENCE

MUNI trains the latent of an observed subset toward predictive sufficiency (Yeo et al., 2026),

$$
\operatorname { T C } ( X ^ { \mathcal { S } } \mid Z _ { \mathcal { S } } ) = 0 , \qquad I ( X ^ { \mathcal { S } } ; X ^ { m } \mid Z _ { \mathcal { S } } ) = 0 \quad \mathrm { f o r ~ a l l } \ m \not \in \mathcal { S } .\tag{16}
$$

The first condition, on the total correlation (TC), concerns dependence within the observed subset, and the second preserves its information about each missing modality. Neither requires the missing targets to be independent given $Z _ { S }$ , as factorized decoding would. A latent can therefore satisfy Equation 16 while factorized decoding misses the joint conditional.

Consider independent uniform bits U and B with $X ^ { 1 } = U$ and $X ^ { 2 } = X ^ { 3 } = \left( U , B \right)$ , and let ${ \mathcal { S } } = \{ 1 \}$ The latent $\bar { Z _ { S } } = U$ satisfies Equation $^ { 1 6 , }$ and pairing $U$ with an independent uniform bit reproduces each conditional $P _ { \mathrm { d a t a } } ( \mathrm { d } x ^ { m } \mid x ^ { 1 } )$ exactly. When the two targets are decoded independently in this way, however, they coincide only with probability $1 / 2$ , whereas $X ^ { 2 }$ and $X ^ { 3 }$ always coincide in the data. This holds even though $I ( \bar { X } ^ { 1 } ; ( X ^ { \dot { 2 } } , X ^ { 3 } ) \mid \bar { Z _ { S } } ) \stackrel { \prime } { = } 0$ , so preserving the source information about the joint target does not justify independent decoding; the latent must also resolve the dependence between the targets. The full-observation latent $W = \mathsf { \bar { ( } } U , B )$ satisfies Equation 1, and its conditional distribution

$$
\begin{array} { r } { Q _ { \mathcal { E } } ( { } \cdot { } | x ^ { 1 } = u ) = \frac { 1 } { 2 } \delta _ { ( u , 0 ) } + \frac { 1 } { 2 } \delta _ { ( u , 1 ) } } \end{array}
$$

fixes the shared bit once, before either target is decoded, so Equation 15 recovers the correct joint conditional. MUNI imposes coherence sufficiency on the encodings used for prior learning, whereas MUNITE requires it of the full-observation latent whose conditional distribution is inferred for every input subset.

## B DERIVATION OF PARTIAL-DATA SELF-DISTILLATION

We prove Proposition 1 for the unweighted objectives used in training. Throughout, the representation and the teacher are fixed, both objectives use the same distribution over times and conditioning subsets, and each partially observed example is a complete draw from $P _ { \mathrm { d a t a } }$ with its unavailable modalities removed independently of the data. The synthetic masking of PolyMNIST-D-Q and FFHQ64 matches this setting, whereas on image–text–audio the available modalities follow the source corpus.

## B.1 THE TEACHER STATE

Fix $t \in \mathsf { [ 0 , 1 ) }$ . Let $W _ { 0 } \sim \mathcal { N } ( 0 , I _ { D } )$ be independent of the complete observation and $W _ { t } \ =$ $( 1 - t ) W _ { 0 } ^ { \mathbf { \bar { \alpha } } } + t W$ . Since $W - W _ { t } = ( \dot { 1 } - t ) ( W \mathbf { \bar { \mathbf { \alpha } } } W _ { 0 } )$ , the clean-latent prediction $\mu _ { \mathcal { A } }$ determines the velocity

$$
{ \frac { \mu _ { \mathcal { A } } ( w _ { t } , t ; x ^ { A } ) - w _ { t } } { 1 - t } } = \mathbb { E } [ W - W _ { 0 } \mid W _ { t } = w _ { t } , X ^ { \mathcal { A } } = x ^ { \mathcal { A } } ] ,
$$

which is the marginal velocity of the interpolation given $X ^ { A }$ . By Theorem 1 of Lipman et al. (2023), the ODE driven by this velocity from $W _ { 0 }$ has the conditional time marginals of $W _ { t } .$ , so an exact teacher gives

$$
( W _ { t } ^ { \boldsymbol { A } } , \boldsymbol { X } ^ { \boldsymbol { A } } ) \stackrel { d } { = } ( W _ { t } , \boldsymbol { X } ^ { \boldsymbol { A } } ) .\tag{17}
$$

A partial example thus supplies a correctly distributed evaluation state without revealing its full latent. Only these marginals must match; individual teacher trajectories may differ from the straight interpolation paths.

## B.2 PROOF OF PROPOSITION 1

Proof. Fix t and write $f _ { \theta } = \mathcal { E } _ { \theta } ( W _ { t } , t ; X ^ { S } )$ and $\mu _ { \mathcal { A } } = \mu _ { \mathcal { A } } ( W _ { t } , t ; X ^ { \mathcal { A } } )$ . Since ${ \mathcal { S } } \subsetneq { \mathcal { A } }$ , both are functions of $( W _ { t } , X ^ { \mathcal { A } } )$ , and $\mathbb { E } [ W - \mu _ { \mathcal { A } } \ | \ W _ { t } , X ^ { \mathcal { A } } ] = 0$ . The cross term in

$$
\begin{array} { r } { \mathbb { E } \| f _ { \theta } - W \| _ { 2 } ^ { 2 } = \mathbb { E } \| f _ { \theta } - \mu _ { A } \| _ { 2 } ^ { 2 } + \mathbb { E } \| W - \mu _ { A } \| _ { 2 } ^ { 2 } + 2 \mathbb { E } \langle f _ { \theta } - \mu _ { A } , \mu _ { A } - W \rangle } \end{array}
$$

therefore vanishes. By Equation 17, the first term on the right is the distillation loss at time t with teacher prediction $\mu _ { \mathcal { A } }$ . Averaging over t and the subsets gives

$$
\begin{array} { r } { \mathcal { \widetilde L } _ { \mathrm { C F M } } - \mathcal { \widetilde L } _ { \mathrm { d i s t } } = \mathbb E \| W - \mu _ { A } ( W _ { t } , t ; X ^ { A } ) \| _ { 2 } ^ { 2 } , } \end{array}\tag{18}
$$

which does not depend on $\theta ,$ so the two objectives have the same gradient with respect to θ. By the tower property, their common minimizer over functions of $( W _ { t } , \bar { X } ^ { S } )$ is $\mathbb { E } [ \mu _ { A } \mid \dot { W _ { t } } , X ^ { S } ] = \dot { \mathbb { E } [ W \mid }$ $W _ { t } , \boldsymbol { X } ^ { \dot { \boldsymbol { S } } } ] \stackrel { \cdot } { = } \mu _ { S } ( W _ { t } , t ; \boldsymbol { X } ^ { \boldsymbol { S } } )$ □

## B.3 FULL OBSERVATIONS AND SUBSET AVERAGING

When ${ \mathcal { A } } = { \mathcal { M } }$ , the teacher target is $\mathcal { E } ( x ^ { \mathcal { M } } )$ , and the state is the analytic interpolant in Equation 4. Under deterministic encoding, the residual in Equation 18 vanishes, and the target is independent of the Gaussian query, so that query can also serve as $W _ { 0 }$

The implementation averages the loss over one subset of each size (Algorithm 1) instead of drawing a single size per update. By linearity of expectation, this stratified average equals the objective under uniformly sampled sizes, so the derivation applies to it unchanged.

## C ARCHITECTURE AND TRAINING DETAILS

## C.1 ARCHITECTURE

Images are divided into $8 \times 8$ patches as in ViT (Dosovitskiy et al., 2021), giving 64 tokens per $6 4 \times 6 4$ image, and each categorical modality uses a learnable embedding table. For image–text–audio, every modality is given as a precomputed 768-dimensional feature vector, following MUNI and FlowBind, and a linear layer maps it to eight tokens.

The final register states predict K latent tokens of dimension d, with total dimension $D = K d$ . We use $K \times d = 4 \times 8$ for PolyMNIST-D-Q, $6 4 \times 8$ for FFHQ64, and $1 \times 7 6 8$ for image–text–audio.

Table 3 lists the widths and depths of the Transformer backbones, which use eight attention heads and FiLM timestep conditioning (Perez et al., 2018) where applicable. Table 7 compares the total trainable parameters, including the generative decoders.

Table 3: Transformer backbones.
<table><tr><td colspan="4">Model Width Depth Time cond.</td></tr><tr><td>PolyMNIST-D-Q MUNITE</td><td>256</td><td>7 16</td><td>FiLM FiLM</td></tr><tr><td>CFM DFM</td><td>256 256</td><td>24</td><td>N/A</td></tr><tr><td colspan="4">FFHQ64</td></tr><tr><td>MUNITE</td><td>512</td><td>11</td><td>FiLM</td></tr><tr><td>CFM</td><td>512</td><td>22</td><td>FiLM</td></tr><tr><td>DFM</td><td>512</td><td>33</td><td>N/A</td></tr><tr><td colspan="4">Image-text-audio</td></tr><tr><td>MUNITE</td><td>768</td><td>6</td><td>FiLM</td></tr><tr><td>CFM</td><td>768</td><td>26</td><td>FiLM</td></tr><tr><td>DFM</td><td>768</td><td>37</td><td>N/A</td></tr></table>

MUNI and MUNITE use the same generative-decoder families (Yeo et al., 2026). On PolyMNIST-D-Q, the three images share an NCSN++ decoder (Song et al., 2021b) of base width 64, and the labels use two-layer heads of width 256. On FFHQ64, the dense modalities use a spatial NCSN++ decoder of base width 128, and the labels use two-layer heads of width 512. Following common practice for image diffusion models (Esser et al., 2024), both methods train these image decoders with logit-normal timestep sampling. For image–text–audio, we adopt MUNI’s decoders and their training details without modification. Image and audio features are decoded by a width-768, depth-10 flow and a width-768, depth-6 feedforward network, respectively. Text is decoded by the Gemma-3-1B (Gemma Team, 2025) feature-to-text decoder fine-tuned by FlowBind (Cha et al., 2026). A feedforward network maps the shared latent to its prefix, and the decoder itself is adapted only through LoRA (Hu et al., 2022).

## C.2 TRAINING PROCEDURE

Each minibatch contains examples with the same set of available modalities A, from which all conditioning subsets are drawn. Let $n = | { \mathcal { A } } |$ . For reconstruction, a target m is sampled uniformly from A, and a conditioning subset $s \subseteq { \mathcal { A } }$ is drawn by first sampling its size uniformly from $\{ 1 , \ldots , n \}$ . On image–text–audio, this rule is mixed equally with one that draws S from $\mathcal { A } \backslash \{ m \}$ with size uniform on $\{ 0 , \ldots , n - 1 \}$ , to increase the share of cross-modal reconstruction. Contrastive splits draw the size of S uniformly from $\{ 1 , \ldots , n - 1 \}$ . Algorithm 1 summarizes one update. We set $\lambda _ { \mathrm { r e c } } = 1 , \lambda _ { \mathrm { d i s t } } = 2 , \lambda _ { \mathrm { c o n } } = 0 . \dot { 1 }$ , and $\tau _ { c } = 0 . 1$

## C.3 BASELINES

MUNI aggregates Gaussian modality encoders by a product of experts without a prior expert and samples this posterior directly for conditional generation (Yeo et al., 2026).

CFM and DFM use the Transformer blocks, width, and number of attention heads of MUNITE (Table 3), with more blocks so that their trainable parameters approximately match those of MUNITE, including its generative decoders (Table 7). DFM has no timestep conditioning and therefore needs more blocks than CFM. CFM represents images and features as continuous patches or vectors and labels as one-hot vectors, and learns a straight flow from Gaussian noise (Lipman et al., 2023). DFM represents each label by a single token and each image or feature by frozen VQ-VAE codes, with 1,024-entry codebooks, 64 codes per image, and 32 codes per image–text–audio feature, and is trained with absorbing-mask corruption and clean-token prediction (Gat et al., 2024). During training, the numbers of source and target modalities, $s \geq 0$ and $r \geq 1$ with $s + r \leq n$ , are sampled uniformly, and the available modalities are randomly assigned to the source, target, and unused sets.

Algorithm 1 MUNITE training step   
Require: minibatch $x ^ { A } .$ , teacher Euler steps N, learning rate $\eta$   
1: Sample m ∈ A, S ⊆ A, and $\epsilon \sim \mathcal { N } ( \bar { 0 } , I _ { D } )$   
2: $\mathcal { L } _ { \mathrm { r e c } } \gets \ell _ { m } \big ( \mathcal { D } _ { \psi _ { m } } ( \mathcal { E } _ { \theta } ( \epsilon , 0 ; x ^ { S } ) ) , x ^ { m } \big )$ ▷ target-detached   
3: Sample w<sub>0</sub> $\therefore \mathcal { N } ( 0 , I _ { D } )$ and $t \sim \mathcal { U } [ 0 , 1 )$   
4: $\mathbf { i f } \ { \hat { A } } = { \mathcal { M } }$ then   
5: $\bar { w }  \mathcal { E } _ { \theta } ( w _ { 0 } , 0 ; x ^ { \mathcal { M } } ) ; w _ { t }  ( 1 - t ) w _ { 0 } + t \bar { w } ; \mathcal { K }  \{ 0 , \dots , M \}$   
6: else   
7: $w _ { t } \gets w _ { 0 }$   
8: for $j = 0 , \ldots , N - 1$ do   
9: $\begin{array} { r } { \dot { w } _ { t } \gets w _ { t } + \frac { t } { N } v _ { \theta } \big ( w _ { t } , \frac { j t } { N } ; x ^ { A } \big ) } \end{array}$   
10: end for   
11: $\bar { w } \gets \xi _ { \theta } ( w _ { t } , t ; x ^ { A } ) ; \ : { \mathcal { K } } \gets \{ 0 , \dots , | A | - 1 \}$   
12: end if   
13: $\tilde { \mathcal { L } } _ { \mathrm { d i s t } }  0$   
14: for $k \in \mathcal { K }$ do   
15: Sample $S _ { k } \subseteq { \mathcal { A } }$ with $| S _ { k } | = k$   
16: $\begin{array} { r } { \tilde { \mathcal { L } } _ { \mathrm { d i s t } }  \tilde { \mathcal { L } } _ { \mathrm { d i s t } } + \frac { 1 } { \vert \mathcal { K } \vert } \| \dot { \mathcal { E } } _ { \boldsymbol { \theta } } ( \mathrm { s g } ( \boldsymbol { w } _ { t } ) , t ; \boldsymbol { x } ^ { \mathcal { S } _ { k } } ) - \mathrm { s g } ( \bar { \boldsymbol { w } } ) \| _ { 2 } ^ { 2 } } \end{array}$   
17: end for   
18: $\mathcal { L } _ { \mathrm { c o n } }  0$   
19: $\mathbf { i f } \left| { \mathcal { A } } \right| > 1$ then   
20: Sample nonempty $\mathcal { S } \subsetneq \mathcal { A } ; \mathcal { C }  \mathcal { A } \backslash \mathcal { S }$   
21: $\mathcal { L } _ { \mathrm { c o n } } \dot { \gets } \frac { 1 } { 2 } ( \mathcal { L } \dot { s _ {  } } \dot { c } + \dot { \mathcal { L } } c _ {  } s )$   
22: end if   
23: ${ \mathcal { L } } \gets \lambda _ { \mathrm { { r e c } } } { \mathcal { L } } _ { \mathrm { { r e c } } } + \lambda _ { \mathrm { { d i s t } } } { \widetilde { \mathcal { L } } } _ { \mathrm { { d i s t } } } + \lambda _ { \mathrm { { c o n } } } { \mathcal { L } } _ { \mathrm { { c o n } } }$   
24: $( \theta , \psi ) \gets ( \theta , \psi ) - \eta \nabla _ { \theta , \psi } \mathcal { L }$

## D BENCHMARK CONSTRUCTION AND COMPARISON PROTOCOL

## D.1 DATASETS

Table 4: Training observations grouped by available modality subset. For PolyMNIST-D-Q and FFHQ64, categorical labels are observed in all groups. T, I, and A denote text, image, and audio, respectively.
<table><tr><td colspan="3">PolyMNIST  $( + d , q )$ </td></tr><tr><td>Observed subset</td><td>Rows</td><td> $\%$ </td></tr><tr><td> $m _ { 0 } , m _ { 1 } , m _ { 2 }$ </td><td>55,000</td><td>25.0</td></tr><tr><td> $m _ { 0 } , m _ { 1 }$ </td><td>18,333</td><td>8.3</td></tr><tr><td> $m _ { 0 } , m _ { 2 }$ </td><td>18,334</td><td>8.3</td></tr><tr><td> $m _ { 1 } , m _ { 2 }$ </td><td>18,333</td><td>8.3</td></tr><tr><td> $m _ { 0 }$ </td><td>36,667</td><td>16.7</td></tr><tr><td> $m _ { 1 }$ </td><td>36,666</td><td>16.7</td></tr><tr><td> $m _ { 2 }$ </td><td>36,667</td><td>16.7</td></tr><tr><td>Total</td><td>220,000</td><td>100</td></tr></table>

<table><tr><td colspan="3">FFHQ64 (+ Age, Gender)</td></tr><tr><td>Observed subset</td><td>Rows</td><td> $\%$ </td></tr><tr><td>RGB, Seg., Norm. RGB, Seg.</td><td>16,250 5,417</td><td>25.0 8.3</td></tr><tr><td>RGB, Norm.</td><td>5,416</td><td>8.3</td></tr><tr><td>Seg., Norm.</td><td>5,417</td><td>8.3</td></tr><tr><td>RGB</td><td>10,833</td><td>16.7</td></tr><tr><td>Seg.</td><td>10,834</td><td>16.7</td></tr><tr><td>Norm.</td><td>10,833</td><td>16.7</td></tr><tr><td>Total</td><td>65,000</td><td>100</td></tr></table>

<table><tr><td colspan="3">Image-Text-Audio</td></tr><tr><td>Observed subset</td><td>Rows</td><td>%</td></tr><tr><td> $T , I$ </td><td>272,703 183,449</td><td>37.8 25.5</td></tr><tr><td> $I , A$   ${ \dot { T } } , A$ </td><td>91,028</td><td>12.6</td></tr><tr><td> $T , I , A$ </td><td>173,576</td><td>24.1</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Total</td><td>720,756</td><td>100</td></tr></table>

PolyMNIST-D-Q, released by Yeo et al. (2026) as PolyMNIST-Quadrant-Labels and built on PolyMNIST-Quadrant (Zhang et al., 2026), contains three $6 4 \times 6 4$ RGB views with their digit (D) and quadrant (Q) labels. Each view composites a resized MNIST digit (LeCun et al., 1998) onto a modality-specific background by color inversion. The first 220,000 of the 230,000 tuples are used for training, and the last 10,000 are held out for evaluation. Because digits are drawn from both MNIST splits, the same digit image may appear in training and evaluation tuples.

Table 5: Image–text–audio training data.
<table><tr><td>Corpus</td><td>Modalities</td><td>Rows</td></tr><tr><td>LAION-COCO aesthetic subset</td><td>T, I</td><td>241,873</td></tr><tr><td>VGGSound</td><td>I,A</td><td>183,449</td></tr><tr><td>AudioCaps</td><td>T, A</td><td>91,028</td></tr><tr><td>Flickr30k</td><td>T,I</td><td>30,830</td></tr><tr><td>E-MM1</td><td> $T , I , A$ </td><td>173,576</td></tr><tr><td>Total</td><td></td><td>720,756</td></tr></table>

Table 6: Image–text–audio evaluation sets.
<table><tr><td>Routes</td><td>Population</td><td>Count</td></tr><tr><td> $T  I$ </td><td>COCO</td><td>30,000</td></tr><tr><td> $T  A$ </td><td>AudioCaps</td><td>975</td></tr><tr><td> $I  A$ </td><td>VGGSound</td><td>15,446</td></tr><tr><td> $2  1 , 1  2$ </td><td>AudioCaps-based triplets</td><td>975</td></tr><tr><td>Unconditional Joint samples</td><td></td><td>1,000</td></tr></table>

Table 7: Trainable parameters in millions, including learned encoders, priors, and decoders. Frozen foundation models and the frozen VQ-VAEs of DFM, which have 13.27M parameters on PolyMNIST-D-Q and FFHQ64, are excluded.
<table><tr><td>Task</td><td>FlowBind</td><td>MUNI</td><td>CFM</td><td>DFM</td><td>MUNITE</td></tr><tr><td>PolyMNIST-D-Q</td><td></td><td>32.09</td><td>20.09</td><td>20.57</td><td>19.33</td></tr><tr><td>FFHQ64</td><td></td><td>130.99</td><td>105.11</td><td>107.27</td><td>106.64</td></tr><tr><td>Image-text-audio</td><td>567.97</td><td>328.90</td><td>281.21</td><td>281.16</td><td>278.72</td></tr></table>

FFHQ64 is constructed from FFHQ (Karras et al., 2019) and contains RGB, 19-class segmentation, surface normals, age in ten bins (0–2, 3–6, 7–9, 10–14, 15–19, 20–29, 30–39, 40–49, 50–69, and 70+ years), and binary gender. Age, gender, and segmentation labels are taken from FFHQ-Aging (Or-El et al., 2020), and surface normals are pseudo-labels from Marigold E2E-FT (Garcia et al., 2025). The first 65,000 aligned faces are used for training, and the last 5,000 are held out for evaluation.

The image–text–audio corpus (Table 5) contains 720,756 paired or triplet observations from the LAION-COCO aesthetic subset (Schuhmann et al., 2022), VGGSound (Chen et al., 2020), Audio-Caps (Kim et al., 2019), Flickr30k (Young et al., 2014), and E-MM1 (Broadbent et al., 2025). The observed modalities of each example follow the original pairing of its corpus rather than synthetic masking. All models operate on fixed, normalized 768-dimensional features from EmbeddingGemma-300M (Schechter Vera et al., 2025) for text, CLIP ViT-L/14 (Radford et al., 2021) for images, and LAION CLAP HTSAT-base (Wu et al., 2023) for audio, whose features are mapped to 768 dimensions by the audio autoencoder released with FlowBind (Cha et al., 2026). For the three-modality observa tions, we use the caption–image–audio triplets of E-MM1 rated Good Match in its human-annotated 1M split, excluding any triplet that overlaps our evaluation sets or the other training corpora.

## D.2 MODEL SIZES AND TRAINING BUDGETS

In Table 7, CFM and DFM approximately match the trainable parameters of MUNITE, whereas MUNI and FlowBind keep the larger architectures of their released implementations.

All methods are trained with Adam (Kingma & Ba, 2015) under the training budgets in Table 8. MUNI, MUNITE, CFM, and DFM are evaluated with EMA weights, with decay 0.999 on PolyMNIST-D-Q and 0.9995 on FFHQ64 and image–text–audio, whereas FlowBind follows its official pipeline without EMA.

## D.3 INFERENCE BUDGETS

In Table 9, CFM is given as many network evaluations as the latent flow and decoder of MUNITE combined, counting two forward passes per step with classifier-free guidance (Ho & Salimans, 2022). DFM samples one token at a time in random order, which avoids the factorization error of predicting several tokens in parallel. FlowBind (Cha et al., 2026) infers a latent from each source and averages them when several sources are given; it has no unconditional sampler.

Table 8: Training budgets. Batch sizes are global, and all methods use a constant learning rate. FlowBind’s epoch count assumes 702 updates per epoch.
<table><tr><td>Method</td><td>Epochs</td><td>Updates</td><td>Batch</td><td>LR</td></tr><tr><td>PolyMNIST-D-Q</td><td></td><td></td><td></td><td></td></tr><tr><td>MUNI, MUNITE</td><td>100</td><td>85,600</td><td>256</td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>CFM, DFM</td><td>200</td><td>171,200</td><td>256</td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>FFHQ64</td><td></td><td></td><td></td><td></td></tr><tr><td>MUNI, MUNITE</td><td>500</td><td>252,000</td><td>128</td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>CFM, DFM</td><td>1,000</td><td>504,000</td><td>128</td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Image-text-audio</td><td></td><td></td><td></td><td></td></tr><tr><td>MUNITE</td><td>100</td><td>70,200</td><td>1,024</td><td> $4 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>MUNI, CFM, DFM</td><td>200</td><td>140,400</td><td>1,024</td><td> $4 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>FlowBind</td><td>≈ 285</td><td>200,000</td><td>1,024</td><td> $1 . 0 \times 1 0 ^ { - 4 }$ </td></tr></table>

Table 9: Inference and decoding methods. Euler step counts equal the number of function evaluations, CFG denotes classifier-free guidance, and AR denotes autoregressive sampling. Rendering of image–text–audio features is not included.
<table><tr><td>Method</td><td>Conditional</td><td>Unconditional</td><td>Decoding</td></tr><tr><td colspan="4">PolyMNIST-D-Q</td></tr><tr><td>MUNITE</td><td>Euler 50</td><td>Euler 50</td><td>Image Euler 50, CFG 1.5</td></tr><tr><td>MUNI</td><td>Posterior sampling</td><td>Euler 50</td><td>Image Euler 50, CFG 1.5</td></tr><tr><td>CFM</td><td>Euler 150</td><td>Euler 150</td><td></td></tr><tr><td>DFM</td><td>Random-order AR</td><td>Random-order AR</td><td>VQ decoder</td></tr><tr><td colspan="4">FFHQ64</td></tr><tr><td>MUNITE</td><td>Euler 50</td><td>Euler 50</td><td>Image Euler 50, CFG 1.5</td></tr><tr><td>MUNI</td><td>Posterior sampling</td><td>Euler 50</td><td>Image Euler 50, CFG 1.5</td></tr><tr><td>CFM</td><td>Euler 150</td><td>Euler 150</td><td></td></tr><tr><td>DFM</td><td>Random-order AR</td><td>Random-order AR</td><td>VQ decoder</td></tr><tr><td colspan="4">Image-text-audio</td></tr><tr><td>MUNITE</td><td>Euler 25</td><td>Euler 25</td><td>Image-feature Euler 25, CFG 4</td></tr><tr><td>MUNI</td><td>Posterior sampling</td><td>Euler 25</td><td>Image-feature Euler 25, CFG 4</td></tr><tr><td>CFM</td><td>Euler 75</td><td>Euler 75</td><td></td></tr><tr><td>DFM</td><td>Random-order AR</td><td>Random-order AR</td><td>VQ decoder</td></tr><tr><td>FlowBind</td><td>Euler 10 per source</td><td></td><td>Euler 10 per target</td></tr></table>

For the models trained on our benchmark, generated image and audio features are rendered by Stable unCLIP small (Stability AI, 2023), an unCLIP-style decoder (Ramesh et al., 2022), and by AudioLDM-m-full (Liu et al., 2023), respectively, with 20 denoising steps as in the benchmark. Because CFM and DFM have no modality-specific decoders, they generate normalized text features as FlowBind does, and all three decode them with the frozen FlowBind text decoder, on which the text decoders of MUNI and MUNITE are built (Appendix C.1).

## D.4 VERIFIERS

Following MUNI (Yeo et al., 2026), accuracy and coherence on PolyMNIST-D-Q and FFHQ64 are measured with verifiers that predict the labels and dense maps of generated images. Each verifier fine tunes a pretrained SegFormer backbone (Xie et al., 2021), MiT-B0 for PolyMNIST-D-Q and MiT-B5 for FFHQ64, on the classification, segmentation, and normal-estimation tasks of its benchmark. The verifiers are trained on all 230,000 PolyMNIST-D-Q tuples and all 70,000 FFHQ faces, including the held-out ones, and are kept frozen when scoring all methods.

## E EVALUATION METRICS AND ADDITIONAL RESULTS

## E.1 POLYMNIST-D-Q

PolyMNIST-D-Q is evaluated on the 10,000 held-out tuples and on 10,000 unconditional samples.   
FD uses verifier features instead of Inception features and is averaged over the three views.

For label-to-image generation, accuracy compares the verifier’s prediction for each generated view with the input label, and the $( d , q ) \to m _ { i }$ route counts a view as correct only if both labels match. Unconditional coherence is the fraction of samples in which the verifier assigns the generated label to all three views. One-to-many coherence is the agreement between pairs of generated views on the attribute that the input leaves out, the quadrant for $d  ( m _ { 0 } , m _ { 1 } , m _ { 2 } )$ and the digit for $q  ( m _ { 0 } , m _ { 1 } , m _ { 2 } )$ , averaged over the three pairs and all samples.

Pairwise agreement alone does not show that the missing attribute is sampled correctly, since a model that always generates the same quadrant or digit for a given input would also agree fully. Table 10 therefore also compares the missing label of each generated view with its data conditional, which is uniform because every digit–quadrant pair occurs equally often. MUNITE stays close to the finite-sample reference for both attributes, so its agreement does not come from a collapsed prediction. For the missing quadrant, MUNI is equally close to uniform, yet its views agree only at chance level: the quadrant of each view is correctly distributed, but the three views choose their quadrants independently. When $m _ { 1 }$ and $m _ { 2 }$ are decoded from separately drawn latents (Indep.), MUNITE’s agreement also falls to chance, while the accuracy of each view on the input label remains 0.995 and 1.000 for digit and quadrant inputs. The agreement among views is therefore carried by the shared latent sample.

## E.2 FFHQ64

FFHQ64 is evaluated on the 5,000 held-out tuples and on 5,000 unconditional samples, and RGB fidelity is measured by FID (Heusel et al., 2017) with InceptionV3 (Szegedy et al., 2016) features. Age and gender accuracy compare the verifier’s predictions for a generated face with the input labels, requiring both to match on the two-label route, and unconditional age and gender coherence compare them with the generated labels. Generated dense maps are scored against the verifier’s predictions for the generated face. RGB–segmentation coherence is the pixel accuracy over the 19 classes, and normal error is one minus the mean per-pixel cosine similarity. For (Age, Gender) → (RGB, Seg., Normals), the inputs provide no reference maps, so these scores compare the generated targets with one another.

## E.3 IMAGE–TEXT–AUDIO

The evaluation follows the routes and evaluation sets of FlowBind and MUNI (Cha et al., 2026; Yeo et al., 2026), listed in Table 6. The 975 joint-evaluation triplets contain AudioCaps (Kim et al., 2019) captions and real audio with FLUX.1-generated images (Black Forest Labs, 2024), so the three modalities are not co-recorded. CIDEr is computed with one reference caption per COCO (Lin et al., 2014) or AudioCaps item.

Table 10: Missing attribute in PolyMNIST-D-Q one-to-many generation. TV is the total variation distance between the missing-label distribution of a generated view and the uniform data conditional, averaged over input labels and views; Ref. is its expected value for the same number of samples drawn from that conditional. Indep. is MUNITE with $m _ { 1 }$ and $m _ { 2 }$ decoded from separately drawn latents, keeping $m _ { 0 }$ and all decoder noise fixed. Chance is the coherence of independent uniform views.
<table><tr><td></td><td colspan="3">TV↓</td><td colspan="4">Coherence ↑</td></tr><tr><td>Route</td><td>MUNI</td><td>MUNITE</td><td>Ref.</td><td>MUNI</td><td>MUNITE</td><td>Indep.</td><td>Chance</td></tr><tr><td> $d  ( m _ { 0 } , m _ { 1 } , m _ { 2 } )$ </td><td>0.029</td><td>0.029</td><td>0.022</td><td>0.2528</td><td>1.0000</td><td>0.2500</td><td>0.25</td></tr><tr><td> $q  ( m _ { 0 } , m _ { 1 } , m _ { 2 } )$ </td><td>0.109</td><td>0.034</td><td>0.024</td><td>0.1074</td><td>0.9904</td><td>0.1014</td><td>0.10</td></tr></table>

Fidelity is measured by FID (Heusel et al., 2017) with InceptionV3 (Szegedy et al., 2016) features, FAD (Kilgour et al., 2019) with VGGish (Hershey et al., 2017) embeddings, and CIDEr (Vedantam et al., 2015). Source–target alignment is measured by CLIP ViT-B/32 (Radford et al., 2021) for text– image and LAION larger-CLAP (Wu et al., 2023) for text–audio, both different from the encoders used to extract the training features. CLIP and CLAP similarities and AIS are reported on a scale of 100.

AIS measures how highly a generated sample ranks its paired counterpart within a reference pool, using Wav2CLIP (Wu et al., 2022) audio and CLIP ViT-B/32 (Radford et al., 2021) image embeddings as in FlowBind and MUNI (Cha et al., 2026; Yeo et al., 2026). For N generated samples with normalized embeddings $g _ { i }$ , the normalized embeddings $c _ { i }$ of their counterparts, and a pool R of counterpart embeddings spanning the evaluation set,

$$
\mathrm { A I S } = \frac { 1 0 0 } { N | R | } \sum _ { i = 1 } ^ { N } \sum _ { r \in R } \mathbf { 1 } [ \langle g _ { i } , r \rangle < \langle g _ { i } , c _ { i } \rangle ] .\tag{19}
$$

For text-conditioned joint image–audio generation, each generated image is paired with its generated audio, and the pool contains the generated audios.

External pretrained models. CoDi (Tang et al., 2023) and OmniFlow (Li et al., 2025) are evaluated as released pretrained models. We measure their one-to-many coherence on the same 975 triplets and take their other scores from MUNI (Yeo et al., 2026), where they are reported to two decimal places. For text-conditioned joint image–audio generation, CoDi uses its default DDIM sampler (Song et al., 2021a) with 50 steps and guidance 7.5, and OmniFlow uses 28 steps with guidance 5.

## E.3.1 SINGLE-TARGET GENERATION

Tables 11–13 report the one-to-one and many-to-one scores summarized in Section 4.4. MUNITE has the best score on three of the six one-to-one fidelity measures (Table 11). Relative to MUNI, it has higher text–audio and lower text–image alignment in both one-to-one and many-to-one generation (Tables 12 and 13). Since its single outputs are not uniformly better, the coherence gains in Table 2 are not explained by single-output quality.

Table 11: Image–text–audio one-to-one fidelity on the evaluation sets of Table 6, grouped by target modality. CoDi and OmniFlow scores are taken from Yeo et al. (2026, Table 8). Bold marks the best score in each column.
<table><tr><td></td><td colspan="2">Image: FID ↓</td><td colspan="2">Audio: FAD ↓</td><td colspan="2">Text: CIDEr ↑</td></tr><tr><td>Method</td><td>T → I</td><td>A → I</td><td>T → A</td><td>I → A</td><td>I → T</td><td>A → T</td></tr><tr><td>CoDi</td><td>24.97</td><td>53.60</td><td>8.67</td><td>13.92</td><td>11.97</td><td>7.70</td></tr><tr><td>OmniFlow</td><td>20.52</td><td>97.56</td><td>4.08</td><td>4.87</td><td>26.60</td><td>31.42</td></tr><tr><td>MUNI</td><td>15.615</td><td>22.786</td><td>3.507</td><td>2.246</td><td>44.400</td><td>51.794</td></tr><tr><td>CFM</td><td>13.455</td><td>28.971</td><td>3.564</td><td>2.105</td><td>13.085</td><td>19.729</td></tr><tr><td>DFM</td><td>14.400</td><td>28.374</td><td>3.670</td><td>1.991</td><td>11.263</td><td>20.909</td></tr><tr><td>FlowBind</td><td>14.826</td><td>24.270</td><td>3.477</td><td>2.166</td><td>17.211</td><td>30.905</td></tr><tr><td>MUNITE</td><td>15.939</td><td>21.511</td><td>3.059</td><td>2.135</td><td>37.409</td><td>57.592</td></tr></table>

## E.3.2 COMPONENT ABLATIONS

Table 14 ablates the training components of MUNITE on image–text–audio. No target detach lets reconstruction gradients flow through the target’s key/value paths, no contrastive removes ${ \mathcal { L } } _ { \mathrm { c o n } } .$ and full-only distillation applies the distillation loss only to fully observed examples, so partially observed examples no longer supervise the latent conditionals.

Every variant keeps one-to-many coherence within two points of the full model and above MUNI on all three routes (Table 2), so the one-to-many advantage does not depend on any single auxiliary component. Unconditional coherence depends more on these components. Target detaching matters most: without it, CLAP and AIS coherence drop by 5.6 and 9.6 points, and all three unconditional scores fall below those of MUNI. Restricting distillation to fully observed examples lowers CLAP and AIS coherence by 2.8 and 4.6 points, since partially observed examples then no longer shape the latent distribution sampled in unconditional generation.

Table 12: Image–text–audio one-to-one alignment. CoDi and OmniFlow scores are taken from Yeo et al. (2026, Table 9). Higher is better, and bold marks the best score in each column.
<table><tr><td rowspan="2">Method</td><td colspan="2">Text-image: CLIP</td><td colspan="2">Text-audio: CLAP</td><td colspan="2">Image-audio: AIS</td></tr><tr><td> $T  I$ </td><td> $I \to T$ </td><td> $T  A$ </td><td> $A  T$ </td><td> $I  A$ </td><td>A → I</td></tr><tr><td>CoDi</td><td>29.70</td><td>26.05</td><td>16.72</td><td>32.72</td><td>59.45</td><td>84.47</td></tr><tr><td>OmniFlow</td><td>30.76</td><td>27.48</td><td>30.07</td><td>44.74</td><td>76.52</td><td>64.22</td></tr><tr><td>MUNI</td><td>29.718</td><td>28.373</td><td>30.596</td><td>40.102</td><td>90.390</td><td>94.773</td></tr><tr><td>CFM</td><td>27.710</td><td>24.918</td><td>25.127</td><td>32.094</td><td>82.888</td><td>80.842</td></tr><tr><td>DFM</td><td>27.321</td><td>24.353</td><td>24.883</td><td>32.262</td><td>79.903</td><td>79.933</td></tr><tr><td>FlowBind</td><td>29.202</td><td>26.004</td><td>28.011</td><td>35.393</td><td>87.621</td><td>92.445</td></tr><tr><td>MUNITE</td><td>29.007</td><td>27.526</td><td>30.695</td><td>40.335</td><td>90.655</td><td>94.254</td></tr></table>

Table 13: Image–text–audio many-to-one alignment on 975 triplets, with each generated target scored against both source modalities. CoDi and OmniFlow scores are taken from Yeo et al. (2026, Table 3). Higher is better, and bold marks the best score in each column.
<table><tr><td rowspan="2">Method</td><td colspan="2"> $( I , A ) \to T$ </td><td colspan="2"> $( T , A ) \to I$ </td><td colspan="2"> $( T , I ) \to A$ </td></tr><tr><td>CLIP</td><td>CLAP</td><td>CLIP</td><td>AIS</td><td>CLAP</td><td>AIS</td></tr><tr><td>CoDi</td><td>24.05</td><td>33.72</td><td>24.98</td><td>85.52</td><td>11.06</td><td>65.31</td></tr><tr><td>OmniFlow</td><td>24.73</td><td>36.26</td><td>26.41</td><td>81.51</td><td>13.50</td><td>63.55</td></tr><tr><td>MUNI</td><td>27.922</td><td>36.695</td><td>27.056</td><td>85.168</td><td>30.524</td><td>82.621</td></tr><tr><td>CFM</td><td>25.221</td><td>31.889</td><td>25.712</td><td>80.305</td><td>24.907</td><td>76.098</td></tr><tr><td>DFM</td><td>24.855</td><td>33.259</td><td>25.398</td><td>80.347</td><td>25.141</td><td>75.846</td></tr><tr><td>FlowBind</td><td>25.541</td><td>31.624</td><td>25.841</td><td>88.089</td><td>28.748</td><td>81.878</td></tr><tr><td>MUNITE</td><td>27.289</td><td>38.969</td><td>26.721</td><td>86.745</td><td>30.653</td><td>82.677</td></tr></table>

For single-target generation, removing target detaching or the contrastive loss degrades fidelity on all six one-to-one routes, most visibly image-to-audio FAD and audio-to-text CIDEr without the contrastive loss. Full-only distillation gives mixed results, slightly improving fidelity on the four routes that generate or condition on images and degrading text-to-audio FAD and audio-to-text CIDEr, so partial-data distillation mainly benefits coherence.

Table 14: Component ablations on image–text–audio. Control is the full MUNITE model of Tables 2 and 11–13. Each variant removes one component and otherwise keeps the control’s architecture, optimization, and evaluation protocol. The 1 → N and unconditional rows report coherence among jointly generated modalities. Best results in each row are in bold.
<table><tr><td>Route Input → output</td><td>Metric</td><td></td><td>No target Control</td><td>detach contrastive</td><td>No</td><td>Full-only distillation</td></tr><tr><td>1 → 1</td><td>T → I</td><td>FID↓</td><td>15.939</td><td>16.700</td><td>17.308</td><td>15.230</td></tr><tr><td rowspan="10"></td><td></td><td>CLIP↑</td><td>29.007</td><td>28.543</td><td>27.750</td><td>29.593</td></tr><tr><td>I → T</td><td>CIDEr ↑</td><td>37.409</td><td>35.414</td><td>35.741</td><td>39.667</td></tr><tr><td></td><td>CLIP↑</td><td>27.526</td><td>27.261</td><td>26.919</td><td>27.882</td></tr><tr><td>T → A</td><td>FAD↓</td><td>3.059</td><td>3.145</td><td>3.321</td><td>3.234</td></tr><tr><td></td><td>CLAP↑</td><td>30.695</td><td>30.458</td><td>31.461</td><td>30.540</td></tr><tr><td>A → T</td><td>CIDEr ↑</td><td>57.592</td><td>53.462</td><td>51.927</td><td>53.302</td></tr><tr><td></td><td>CLAP↑</td><td>40.335</td><td>40.670</td><td>39.588</td><td>40.678</td></tr><tr><td>I → A</td><td>FAD↓</td><td>2.135</td><td>2.146</td><td>2.622</td><td>2.094</td></tr><tr><td></td><td>AIS↑</td><td>90.655</td><td>90.768</td><td>91.952</td><td>90.157</td></tr><tr><td>A → I</td><td>FID↓</td><td>21.511</td><td>21.923</td><td>22.466</td><td>20.823</td></tr><tr><td rowspan="6">N → 1</td><td></td><td>AIS↑</td><td>94.254</td><td>93.932</td><td>93.565</td><td>94.260</td></tr><tr><td>(I, A) → T</td><td>CLIP↑</td><td>27.289</td><td>27.191</td><td>27.148</td><td>27.441</td></tr><tr><td></td><td>CLAP↑</td><td>38.969</td><td>38.400</td><td>36.322</td><td>38.517</td></tr><tr><td>(T, A) → I</td><td>CLIP↑</td><td>26.721</td><td>26.670</td><td>26.581</td><td>26.799</td></tr><tr><td>AIS↑</td><td></td><td>86.745</td><td>85.898</td><td>85.624</td><td>85.127</td></tr><tr><td>(T, I) → A</td><td>CLAP↑</td><td>30.653</td><td>30.511</td><td>31.420</td><td>30.804</td></tr><tr><td rowspan="3">1 → N</td><td></td><td>AIS↑</td><td>82.677</td><td>82.461</td><td>83.407</td><td>83.055</td></tr><tr><td>T → (I, A)</td><td>AIS↑</td><td>83.336</td><td>84.174</td><td>83.771</td><td>82.487</td></tr><tr><td>I → (T, A)</td><td>CLAP↑</td><td>34.762</td><td>34.844</td><td>32.852</td><td>34.392</td></tr><tr><td></td><td>A → (T, I)</td><td>CLIP↑</td><td>26.334</td><td>26.261</td><td>26.024</td><td>26.287</td></tr><tr><td rowspan="3">Uncond.</td><td>All three</td><td>CLIP (T, I) ↑</td><td>27.398</td><td>26.713</td><td>26.639</td><td>27.539</td></tr><tr><td></td><td>CLAP (T, A) ↑</td><td>30.220</td><td>24.657</td><td>29.370</td><td>27.379</td></tr><tr><td>AIS (I, A) ↑</td><td></td><td>87.045</td><td>77.451</td><td>85.760</td><td>82.429</td></tr></table>

## F QUALITATIVE RESULTS

Qualitative examples complement the quantitative results of Section 4, particularly for joint generation, where they show whether the generated outputs agree on content that the input leaves unspecified. The examples are selected to illustrate each route, and the outputs of routes whose only output is audio are provided on the project page.

## F.1 POLYMNIST-D-Q

In single-target generation (Figure 4), MUNI and MUNITE render the conditioned labels in every example, whereas CFM often produces illegible digits and DFM frequently omits the digit or renders a different one, in line with their accuracies in Table 1.

In unconditional co-generation (Figure 5), the three views and two labels of a sample must agree with one another. All four samples of MUNITE agree, whereas MUNI’s views sometimes disagree, showing different digits or placing one digit in different quadrants.

In the one-to-many routes (Figure 6), the views must also agree on the attribute that the input leaves unspecified. MUNITE places a given digit in the same quadrant in all three views and draws the same digit for a given quadrant. MUNI’s views are each consistent with the input but draw the missing attribute separately, so the digit 1 appears in three different quadrants and the lower-left quadrant holds the digits 0, 5, and 4. This is the gap between target-wise sufficiency and joint coherence discussed in Section 2.7.

<sup>(a)</sup> <sup>Digit</sup> → <sup>m</sup>i  
![](images/257fc40d3a83a36949c3798ee79247d601273a4ebecb74bb25a39ab677fac3f0.jpg)  
Figure 4: PolyMNIST-D-Q single-target generation. Icons show the observed labels: a digit, the conditioned quadrant shaded in a 2 × 2 grid, or the digit placed in that quadrant. Each column generates the view listed under its input.

![](images/173bf68825dc9a76a325983bd7385d56b2a601a042f7005c057065922e519982.jpg)  
Figure 5: Unconditional PolyMNIST-D-Q co-generation. Each group is one joint sample of the three views and both labels, and its icon shows the generated digit in the generated quadrant.

![](images/3bcf0484fc49a2ffbb800278c4d9f3aa9043fa3505f18f7ccc135a79340ba469.jpg)  
Figure 6: PolyMNIST-D-Q one-to-many generation. Each row generates all three views from a digit (a) or a quadrant (b), so the views must agree on the unobserved attribute.

## F.2 FFHQ64

(a) Age  RGB  
(b) Gender RGB  
![](images/1cb31514a436b5424cc2af16ed18a04a69d24662bf0df96b3a61585e60f1c242.jpg)  
Figure 7: FFHQ64 RGB generation from age (a), gender (b), or both attributes (c).

In RGB generation from age and gender (Figure 7), MUNITE’s faces follow the conditioned age from young children to older adults and match the conditioned gender. MUNI’s faces also follow the attributes but are occasionally distorted, CFM’s show strong artifacts, and DFM’s are noticeably blurred.

In unconditional co-generation of all five modalities (Figure 8), MUNITE’s generated labels agree with the apparent age and gender of its faces, and its segmentation and normals align with the RGB image. The man in the 50–69 sample, for instance, wears glasses in the RGB image, and the segmentation marks them as well.

In the many-to-many route (Figure 9), the segmentation and normals must describe the generated face. MUNITE’s maps follow the head pose, hairline, and facial layout of its RGB image in every example. MUNI’s maps sometimes describe a different face, as for the male 40–49 input, whose segmentation omits the glasses in the RGB image.

![](images/7208769bc3897a7544ce77d47a37ce2b2446f4ee009920d62c343bfb11161d07.jpg)  
Figure 8: Unconditional FFHQ64 co-generation of all five modalities. The generated gender and age labels of each joint sample appear below its maps.

![](images/c494ed782cec963231876706c9075029d2ba5928f654651583e61b00dfc15c22.jpg)  
Figure 9: FFHQ64 many-to-many generation of RGB, segmentation, and surface normals from age and gender. The three maps of each group come from one joint sample.

## F.3 IMAGE–TEXT–AUDIO

Audio clips are provided on the project page under the IDs printed below their waveforms. The bracketed descriptions are only a reading aid and summarize the top AudioSet labels predicted by PaSST (Koutini et al., 2022).

In one-to-one and many-to-one generation (Figures 10 and 11), MUNITE’s outputs follow the specific content of their inputs. Its image for the roaring lions in A02 shows a lion where FlowBind shows a tiger, and its caption for the stop-sign image reads the sign. With two inputs, its captions draw on both. For the faucet image paired with running water and speech (A10), MUNITE mentions both the water and the speaker, whereas MUNI mentions only the water.

For joint generation, we show samples of MUNITE (Figures 12 and 13). An unconditional sample has no input, so whatever its outputs share comes from the latent sample. The three outputs of each column describe one scene, such as a coastal town at sunset with the sound of waves (A13). For the painting of a woman at a table, the image shows cups and a plate and the audio contains dishes and cutlery (A16), a detail that the caption leaves out.

In the one-to-many routes, the two outputs are decoded independently from one latent sample, so any detail they share beyond the input must also be carried by that sample. For the ducks and speech in A34, both outputs turn the speech into a crowd, with the caption describing people talking near quacking ducks and the image showing a crowd lining a canal full of ducks. Given the horse under lightning, the caption mentions the lightning and the audio adds thunder and rain (A29).

![](images/c001c369b28026cb51e7efb09af93634f1140a08553f67265a60d33cb80a77c6.jpg)  
Figure 10: Image–text–audio one-to-one and many-to-one generation of images. Audio inputs are shown as waveforms.

<table><tr><td>Input</td><td>MUNI</td><td>CFM</td><td>DFM</td><td>FlowBind</td><td>MUNITE</td></tr><tr><td>(a) Image → text</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>A street light is covered in snow and has a green dome.</td><td>Many people in the picture are crossing the street light.</td><td>Night in the city, the night lights on snow covered mountains.</td><td>Side shot of the street with green rain.</td><td>traffic lights on the side of a road in winter</td></tr><tr><td></td><td>A street sign that says New York City Hall.</td><td>A long-sleeved photo shows the city with its large, stopped traffic.</td><td>Silverware on the side scene where a guy has of a road with lights. stopped all the traffic</td><td>Closeup picture of a signs</td><td>A sign on the side of a street that says, &quot;Stop!&quot;</td></tr><tr><td></td><td>Three teddy bears and two cups on a table.</td><td>Children, with plastic knee braces, are playing with a table of toddlers.</td><td>Purple roses are a great gesture for a female.</td><td>Silverware and tea bags are on the table with several teddy bears.</td><td>teddy bears sitting at a table with tea cups</td></tr><tr><td>(b) Audio → text</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>[speech, motorboat, wind] A07</td><td>wind blows and a man and child speak</td><td>wind rustling, a child and an adult man speak in the distance</td><td>Asian men are talking, wind rushes, a child a man and a child talk and man speak, a child after loud</td><td>talks</td><td>a man and a child talk while wind blows</td></tr><tr><td></td><td>a man speaks as ducks quack</td><td>wind and rustling, people speaking, and in the background with</td><td>windmills and people wind blows, and a man</td><td>speaks, and ducks</td><td>ducks quack while a man talks</td></tr><tr><td>[speech, ducks quacking] A08</td><td>a train horn blows as it passes</td><td>ducks emergency sirens on a truck, with a loud</td><td>ducks wind, a train running,</td><td>quack wind rustling, a train running on something,</td><td>a train is blowing its</td></tr><tr><td>[train passing, horn] A09 (c) (Image, audio) → text</td><td></td><td>opening of a train</td><td>a horn blows</td><td>a horn</td><td>horn as it passes by</td></tr><tr><td></td><td>water running from a</td><td>wind rushes, a man talking, a water splash speaks running water</td><td>wind rustles, a man</td><td>heavy machinery, a man speaks, water</td><td>water running while a</td></tr><tr><td>[running water, speech] A10</td><td>faucet</td><td></td><td></td><td>running</td><td>man speaks</td></tr><tr><td></td><td>bells ring out repeatedly</td><td>adult females speak, bells chime</td><td>tapping sounds, and some ringing of bells</td><td>animals are barking, bells ring out playing a and bells ring.</td><td>song</td></tr><tr><td>[glockenspiel melody] A11</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>a woman is giving a</td></tr><tr><td>[woman speaking] A12</td><td>A woman sitting at a table with a microphone.</td><td>female speaking in front of a microphone</td><td>birds loudly chirping, a girl speaks</td><td>turquoise chairs, a woman speaking,</td><td>speech</td></tr></table>

Figure 11: Image–text–audio one-to-one and many-to-one generation of text. Audio inputs are shown as waveforms.

![](images/f3565c47df3539e68a977c241daef33a1fbc1e0eea1447f5904393b442e74048.jpg)  
Text  
The sun sets over a town on the coast.  
Image  
[dogs barking] A15  
Audio  
Image  
“vibrationsfrom a sewing machine”  
Audio

![](images/094f52c6460a1474bef18d6a8e9df55f3f83a8c948dd234a82e0fb53b5f97bea.jpg)  
a wolf howling with light rustling in the background

![](images/5185187a6ba8a4b8e3628e3afe08fe2f8aaf1ed88aa98d6545fb621612a75748.jpg)  
[ocean waves, wind] A13  
Two dogs that are looking at each other.  
[canine howling] A14

![](images/90061e0d240f7768c1c889d9e3e61023c7f8a68037c830d786c8d87ca0fe46b3.jpg)

Audio

A painting of a woman sitting at a table.

![](images/6022f031c7399744e7ce40460c928b36fbe7b95043a652e450a36b3ad9b4dd12.jpg)  
A bird sitting on top of a tree branch.

![](images/05b9255b9ec7553d1dd3993691c57459f4d79f6e7b83c7a901d7168c144e17a2.jpg)  
[dishes and cutlery] A16  
[wingsflapping] A17  
The foggy forest is full of trees and bushes.

Text

Image

![](images/ab0477f528aeef1e9c87a8027d66ac5df70f9855f44c6dcf6ce100f3a69a3b8c.jpg)  
[owl hooting] A18  
An old photo shows the snow covered landscape.

![](images/5d08a5d87c34116048777fe54218cccf1da171094ab40db8ca3a3a6a87774315.jpg)  
[speech, wind] A19  
Two men in hats and jackets are talking.

![](images/d0ed14fb0f8a6c1b41afaf3621af57f02868fdc9e2bbb10d52c09806ffe9074f.jpg)  
[speech, hoofbeats] A20  
An old painting shows people standing outside a building.

![](images/7796fcd1d2236612d18b5cb49ebae82dce872340f4ea1630173df2f1985f27b8.jpg)  
[speech, hoofbeats] A2  
Two young boys in suits standing next to each other.

![](images/81e7802709949c2b30ccd366c0947c09c2f9dcdbb26db9ba8c4aad39c868fff0.jpg)  
[speech,footsteps] A22

Figure 12: Unconditional image–text–audio co-generation by MUNITE. Each column decodes one latent sample into text, image, and audio.

## (a) Text  (image, audio)

Input

![](images/c74cfcc7ab2b86782fae6b4549b782b6247ddf056370e4af074966fcaf062747.jpg)  
[sewing machine] A23  
“a fire siren with people talking”

![](images/c7ba2120442c6270ff93d93132c37e6b198d7ab0b58f018694fce222c83a3938.jpg)  
[fire-truck siren, speech] A24  
“a woman speaksfollowed by birds chirping”

![](images/ac80a00896286746406541c9a91cc917ea8ca27d2d8106fe61f5494b5ef7fc63.jpg)  
[birds chirping, speech] A25  
“an aircraft engine is taking off”

![](images/ea19d1814e8d22b24565bbf11e4fecc90c5eca7aa8c04426662a0fdb2abdae4c.jpg)  
[aircraft engine] A26  
“a series of bell chimes and ringing”

![](images/5ef325917b9c058375b9e7dfc278595e07ca45072d8f065e4ff31c13f15ad9af.jpg)  
[church bells ringing] A27  
(b) Image (text, audio)  
Input

![](images/122adfc95decfd2e1494ed7b988eec99a712d74aa2380dd989e06a3016dfa425.jpg)  
a bell rings continuously  
[bell ringing] A28

![](images/0d6a3bc12160479b9c352663cfcda7625b99f9f55c9d19c5b072ed9beb4020d1.jpg)  
A horse is standing in the grass with lightning behind it.  
[thunder and rain] A29

![](images/1d238ba3d0e8a96f610a6010ec281c33a39444a1ca9f05beace240fbe91b5f3f.jpg)

![](images/a37cbbf73f008a4818323f29ab29ac51f6799ed2ce8bfe69f495949abb84c3d8.jpg)

![](images/61c80d83da7b82891b1b5861751c41fc4ab263d64463fb55b2f13e4b9013a405.jpg)  
[clock ticking] A30  
a person is typing on a computer keyboard  
[keyboard typing] A3

![](images/8987927333d14945c1921f9f3487e3f78e77f352851ff7c64de96c7d5c0bb451.jpg)  
A bathroom with an empty toilet and white walls.  
[toilet flushing] A32  
(c) Audio  (text, image)

Input

![](images/c77d92054cdcd6289cd86eab101cfcca59252c94916606067936adf966e71c41.jpg)  
[church bells ringing] A33

church bells ringing in the Text distance

![](images/e7356a552fd0821d7fb0510745642e803703592d8df351ab5fcee2a27311801c.jpg)  
Image

![](images/1430b53017add767001ce03c5e7956723156707df41591059c2ec1717d79f118.jpg)  
[ducks quacking, speech] A34 a group of ducks quacking as a crowd of people talk in the background

![](images/bc42b9bfb48b060526644d1a49b6209aefd9f1e43740dea71f094899832a5a7c.jpg)  
[speech, applause] A35

![](images/73f0f99f0e50b52cf9b982e12fdd727893e0d5ce3a414eefb6bede313613493f.jpg)  
a man gives a speech then an audience gives applause

![](images/5b4889f584af5843b3d4f7c21eeecca038177fd74167990aaf2371e4430e9f84.jpg)

![](images/a6ea0ec8d5caacc3aa89489ac1e078ac685a7a5b47d4b3ca2e02ebaca5dcf8f3.jpg)  
[speech, guitar] A36 a man speaks while a guitar is played followed by someone briefly talking

![](images/8c3cee264dd3e7cf08da3746e7725fe53807d6f88a28f463e9e85775e4d522fc.jpg)  
[frogs croaking] A37

![](images/2022a2e426034a08e4b88a0141a6b29fab62eb451f8b519e35751f4db5020b6f.jpg)  
frogs croak and crickets chirp in the background

![](images/ef4d1e09757445f05e391eec1458f9375ab1b56c05ccfad531d53eb7db5890c5.jpg)

Figure 13: One-to-many image–text–audio generation by MUNITE. In each column, the two outputs are decoded independently from one latent sample inferred from the input.