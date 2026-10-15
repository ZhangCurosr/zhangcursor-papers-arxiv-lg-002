# Predicting Unseen Scientific Interactions: Conditional Flow Models Hallucinate Non-Additive Effects

Sam Gijsen<sup>∗</sup>

Moritz Seiler

Kerstin Ritter

Hertie Institute for AI in Brain Health, University of Tübingen, Germany Tübingen AI Center, University of Tübingen, Germany Charité – Universitätsmedizin Berlin, Department of Psychiatry and Psychotherapy, Germany German Center for Mental Health (DZPG), partner site Tübingen, Germany

## Abstract

Conditional generative models are increasingly used for scientific prediction beyond the direct training support. Yet scientific questions often concern whether one intervention changes the effect of another. Such interactions can be of primary interest while explaining little of the total variation, allowing models to appear accurate overall without recovering the relevant scientific effect. It remains unclear how these models learn interactions and when these successfully extrapolate to unseen conditions. We therefore study interaction recovery in flow-matching models using a physiological time-series simulator, drug dose-response data, combinatorial cell perturbations, and molecular conformer generations, where we separate recovery from hallucinated interaction structure. In the controlled setting, individual effects are learned much earlier than interactions and models hallucinate interactions where none exist. Across settings, we find qualitatively different generalization problems, with successful prediction depending on whether condition structure continues from observed to held-out targets. When it does, interactions are recoverable. If not, models can generate plausible responses while missing the interaction. We furthermore find that hallucinations increase as targets become less identifiable, and that successful interaction recovery in output space can fail under nonlinear readouts. Finally, we show that aggregate fidelity metrics can conceal whether models learn and extrapolate interactions.

Conditional generative models such as diffusion and flow matching models have gained significant attention for generating novel biological samples and for predicting biological responses under unseen experimental conditions. Prominent use cases include single-cell transcriptomics [He et al., 2026, Yu et al., 2026], neuroscience [Gijsen et al., 2026], and protein design [Watson et al., 2023, Ingraham et al., 2023].

To be scientifically valuable, models must be able to predict how experimental conditions interact, rather than only generating broadly plausible responses. For example, administering two drugs together may be antagonistic, independent, or synergistic, and redundant pairs require predicting that a drug has a clear effect on its own but not when combined with another. Importantly, however, the bulk of overall variability across samples is often dominated by effects of individual factors. Conversely, interactions often account only for a small fraction, despite regularly being the primary object of scientific interest. For instance, in single-cell perturbation experiments, the interaction between cell-type and perturbation determines less of the variance than either factor by itself, yet it is often the central question [Schäfer et al., 2026].

As aggregate fidelity metrics (such as MSE or FID) are heavily influenced by those stronger effects of individual factors, they may tell us relatively little about how interactions are learned during training or whether an interaction learned from observed combinations can be extrapolated to unseen ones. This is critical for scientific prediction and discovery, as this implies risks where models may appear converged while still learning the interaction, or appear reasonably accurate while predicting the effect that motivated an experiment incorrectly. Furthermore, interactions are not only a property of the system but also of the data representation. A nonlinear readout of a sample can imply a different interaction, and therefore interaction recovery in output space does not transfer to a quantity which often can be more scientifically relevant.

The difficulty of this generalization problem directly depends on the available training data. A held-out target may be identifiable via smoothness of the condition space, via graph-relational knowledge, or underdetermined if only marginal support is provided. The latter problem, often called compositional generalisation [Wiedemer et al., 2023], is fundamentally different in the presence of interactions, as a training support covering each factor alone no longer determines the response in the held-out region of condition space [Brady et al., 2025]. Simply put, observing two factors separately does not determine how they interact [Lachapelle et al., 2023, Dong and Ma, 2022]. In this case, the missing relationship must come from observed joint conditions, inductive biases in the model, or meta-knowledge [Shojaee et al., 2025].

Here, we study how conditional flow models learn and extrapolate non-additive interactions in four systems that differ in the manner in which the training support constrains the held-out targets. Specifically, 1) a photoplethysmography (PPG) simulator with ground truth and tunable interaction strength, 2) drug dose-response screens where observed doses surround the held-out conditions, 3) combinatorial perturbations in single cells where components of a held-out pair are optionally excluded from observed pairs, and 4) radii of gyration over generated conformers given distinct molecular supports.

## We summarize our contributions as follows:

• Diagnostics for interaction prediction. We decompose predicted interactions into interaction recovery and orthogonal hallucination, with an identity to interaction error.

• Hallucination of non-additivity. In a simulator with a controlled interaction strength, we find flow models to generate interactions even in fully additive regimes. We additionally find stronger hallucinations for less constrained targets across settings.

• Continuation rather than distance determines extrapolation. We show that held-out regions of a shared condition space at equal distance from the training support can be recovered almost completely or barely at all, depending on the continuation of the condition structure off the training support.

• Aggregate metrics conceal failures to predict interactions. We observe distinct learning trajectories for interactions compared to main effects, that low endpoint errors can accompany failed interaction recovery, and that recovery in output space can fail after a scientific readout.

## 1 Related work

Predicting perturbation and combination responses. Deep models are now commonly used to predict responses to unseen chemical and genetic perturbations and their combinations [Roohani et al., 2024, Klein et al., 2025, Preuer et al., 2018]. However, recent work shows that such models can fail to beat additive or mean baselines for double perturbations [Ahlmann-Eltze et al., 2025], and that common metrics are inflated by systematic variation [Viñas Torné et al., 2025]. As additive baselines predict no interaction, we use these as a reference for scoring interaction errors. We complement these existing evaluations by decomposing interaction errors and relating them to the identifiability of held-out condition regions.

Compositional generalization and inductive biases of generative models. Compositional general ization has often been studied in settings where the joint response is determined by its components [Wiedemer et al., 2023, Okawa et al., 2023, Liu et al., 2022, Lippl and Stachenfeld, 2024], which is an assumption that the presence of interactions violate. Prior work additionally links the order in which concepts are learned to their signal strength [Rahaman et al., 2019, Park et al., 2024], which invites the question how low-energy interactions are learned. Diffusion models have also been shown to produce unseen samples by interpolating between modes [Aithal et al., 2024], with the interaction hallucinations we study here constituting an analogous failure over condition space rather than sample space.

![](images/a1f84aa295869778bba6bf2a4ff48656f025461629a672071c274ad623fe3bf1.jpg)  
Figure 1: Aggregate generation quality does not guarantee accurate predicted interactions. Two conditions are sampled with equal distance off the training support but in different directions (1). Both samples score well on normalized endpoint error $( \mathcal E _ { \mathrm { e n d } } )$ , far below a trivial baseline (2). But decomposing the prediction into the component along the true interaction (α) and the orthogonal, unrelated component (β) distinguishes the samples into interaction recovery (✓) and attenuated interaction with hallucinated structure $\left( \approx \right) \left( 3 \right)$ . Finally, applying a readout function g can considerably change the results $( \checkmark \xrightarrow { g _ { \downarrow } } \times )$ , with less recovery and yet more hallucinated structure in the readout space compared to output space.

Interactions depend on representations. Whether factors interact depends on the data representation in which we measure, which has been long established in epidemiology, where it is described as the interaction scale [Rothman et al., 1980, VanderWeele and Knol, 2014]. We extend this to the evaluation of generative models, showing that interaction recovery can differ between the output space and under a scientific readout.

## 2 Methods

## 2.1 Setup

We start by providing a brief general framework. Given two factors A and B taking values a and b (categorical or continuous), flow matching models predict response distributions $P _ { a , b }$ over $\mathbb { R } ^ { d }$ with conditional means $\mu ( a , b ) = \mathbb { E } [ Y \mid a , \mathbf { \bar { b } } ]$ . The factors are additive if $\mu ( a , b ) = g ( a ) + h ( b )$ for functions g and h and the interaction here is the part of the response not captured by such decomposition. For reference levels $( a _ { 0 } , b _ { 0 } )$ (Section 2.3), we define the additive interaction contrast as

$$
I ( a , b ; a _ { 0 } , b _ { 0 } ) = \mu ( a , b ) - \mu ( a , b _ { 0 } ) - \mu ( a _ { 0 } , b ) + \mu ( a _ { 0 } , b _ { 0 } ) \in \mathbb { R } ^ { d } ,
$$

which vanishes for all $( a , b )$ if and only if $\mu$ is additive. In the canonical $2 \times 2$ design with a 0 baseline, individual conditions A and $B ,$ and their joint condition $A B$ , this reduces to

$$
I _ { A B } = \mu _ { A B } - \mu _ { A } - \mu _ { B } + \mu _ { 0 } .
$$

For continuous factors without a natural reference level, there is no scientifically meaningful choice for $( a _ { 0 } , b _ { 0 } )$ . We therefore use functional ANOVA (Section 2.2), which avoids choosing a reference. Namely, its interaction at each condition is simply the average of the contrast across all possible reference points in the grid. The interaction is contrasted with the total effect $\mu ( a , b ) - \mu ( a _ { 0 } , b _ { 0 } )$ which is what most aggregate evaluations are sensitive to.

Interactions depend on the representation. Scientific disciplines are often concerned with representations in a semantic space (rather than the raw output space), which then form the basis for drawing conclusions. For a fixed, deterministic readout function $g ~ : ~ \mathbb { R } ^ { d } ~  ~ \mathbb { R } ^ { k }$ $I _ { A B } ^ { g } = g ( \mu _ { A B } ) \stackrel {  } { - } g ( \mu _ { A } ) - g ( \mu _ { B } ) + g ( \mu _ { 0 } )$ . If g is nonlinear, then interaction recovery in the output space does not imply recovering it after applying the readout. Therefore, $\| \hat { I } - I \|$ may be considerably smaller than $\| { \hat { I } } ^ { g } - I ^ { g } \|$ , in which case evaluating model predictions only in the output space can be incomplete or misleading.

Relating endpoint errors to interaction errors. Let $\hat { \mu } _ { c }$ denote the predicted mean response under condition c and $e _ { c } = \hat { \mu } _ { c } - \mu _ { c }$ the endpoint error. By substituting into the definition of fIAB $I _ { A B }$ we get:

$$
\hat { I } - I = e _ { A B } - e _ { A } - e _ { B } + e _ { 0 } ,
$$

where $\hat { I }$ denotes the implied interaction based on the model predictions. These four errors have no guarantee of cancellation meaning that a small joint error does not bound the interaction error. Conversely, any errors shared across conditions may cancel out and thus small interaction errors do not indicate that individual conditional means are accurate.

Due to this, we distinguish between ’anchored’ and fully generated estimates. The former relies on reference conditions while the latter relies fully on the model:

$$
\hat { I } ^ { \mathrm { a n c h } } = \hat { \mu } _ { A B } - \mu _ { A } - \mu _ { B } + \mu _ { 0 } , \hat { I } ^ { \mathrm { f u l l } } = \hat { \mu } _ { A B } - \hat { \mu } _ { A } - \hat { \mu } _ { B } + \hat { \mu } _ { 0 } .
$$

The anchored error corresponds to $e _ { A B }$ and isolates model error for unseen conditions, which is analogous to experimental settings in which only single factors (but not their joint) have been measured. $\hat { I } ^ { \mathrm { f u l l } }$ instead corresponds to using the model for the whole experiment.

Distinguishing hallucination and other interaction failure modes. To expose distinct failure modes, we decompose the prediction into two components: one along the true interaction and along an orthogonal component:

$$
\hat { I } = \alpha I + I _ { \perp } , \qquad \alpha = \frac { \left. \hat { I } , I \right. } { \| I \| ^ { 2 } } , \qquad I _ { \perp } = \hat { I } - \alpha I , \qquad \beta = \frac { \| I _ { \perp } \| } { \| I \| } ,
$$

where $I _ { \bot }$ is the part of the model prediction orthogonal to the true interaction. α then is the signed gain along the true interaction with β denoting the orthogonal relative energy. Because the components are orthogonal this gives the following exact identity:

$$
\mathcal { E } _ { I } ^ { 2 } = \frac { \Vert \hat { I } - I \Vert ^ { 2 } } { \Vert I \Vert ^ { 2 } } = ( 1 - \alpha ) ^ { 2 } + \beta ^ { 2 } .
$$

The zero-interaction predictor is $( \alpha , \beta ) = ( 0 , 0 )$ with $\mathcal { E } _ { I } = 1$ . Thus, α indicates how much of the true interaction the model recovered and $\beta$ how much the model additionally hallucinated. $0 < \alpha < 1$ with a small $\beta$ can be interpreted as an attenuated predicted interaction, $\alpha < 0$ a reversal, $\alpha > 1$ an exaggeration, and $\alpha \approx 0$ with a large $\beta$ indicating error of which the structure is unrelated to the truth (Figure 1). We note that with $m = \lVert \hat { I } \rVert / \lVert I \rVert$ and $\theta$ denoting the angle between the predicted and true interaction, we get $\alpha = m$ cos θ and $\beta = m$ sin $\theta .$ These are consequently recoverable from commonly used quantities rather than novel metrics.

Identification and generalization regimes. Many conditional-mean functions agree on the training support while yielding distinct interactions for held-out targets. Consequently, $I _ { A B }$ is not identified by the training support alone, with inductive biases determining the prediction. Hence our experiments can be seen as evaluating this bias.

Crucially, this distinguishes an interaction setting from the additive compositional generalization problem. Namely, with interactions, the widely studied L-shaped support for grid-based contexts [Wiedemer et al., 2023] therefore requires function extrapolation that the observed marginals do not determine. We distinguish between generalization regimes based on what partially enables identification:

• Smoothness identified: the held-out region of condition space is surrounded by joint observations.

• Relationally identified: factors A and B are each observed in other combinations.

• Underdetermined: regime in which only marginals have been observed and thereby a training support which does not provide any constraint on interactions on the held-out region.

Table 1: Metrics overview.
<table><tr><td>Metric</td><td>Definition</td><td>1 corresponds to</td></tr><tr><td> $\mathcal { E } _ { \mathrm { e n d } }$ </td><td> $\mathrm { R M S } ( \hat { Y } - Y ) / \mathrm { R M S } ( Y - Y _ { 0 } )$ </td><td>predicting the baseline</td></tr><tr><td> $\mathcal { E } _ { \mathrm { a d d } }$ </td><td> $\mathrm { R M S } ( \hat { \tilde { Y } } - \tilde { Y } ) / \mathrm { R M S } ( Y - Y _ { 0 } )$ </td><td>an error the size of the total response</td></tr><tr><td> $\mathcal { E } _ { I }$ </td><td> $\| \hat { I } - I \| / \| I \|$ </td><td>predicting zero interaction</td></tr><tr><td> $\alpha , \beta$ </td><td>Section 2.1</td><td></td></tr></table>

## 2.2 Metrics

We normalize errors such that a value of 1 corresponds to a trivial predictor. Specifically, we use the response scale ${ \mathsf { R M S } } ( Y - Y _ { 0 } )$ for response errors and the interaction scale ${ \mathsf { R M S } } ( I )$ for interaction errors (Table 1). $Y _ { 0 }$ is a domain-appropriate control or reference response.

For $\mathrm { a 2 } \times 2 \mathrm { d e s i g n } , I _ { A B }$ is the AB response minus the additive part $\mu _ { A } + \mu _ { B } - \mu _ { 0 }$ . Our simulator uses a large 2D grid of conditions and therefore has many condition levels per factor. We therefore obtain the same decomposition using functional ANOVA over this grid, with $Y = \tilde { Y } + I$ , where $\tilde { Y }$ is the grand mean plus the two main-effect surfaces and I the residual non-additive component. We also apply this to the model’s generated grid $\hat { Y }$ , giving $\hat { Y } = \hat { \tilde { Y } } + \hat { I } .$ , where $\hat { \tilde { Y } }$ is the additive part of the model’s prediction and $\hat { I }$ its interaction. All grid cells are used to compute the grand and main-effect means. $\mathcal { E } _ { I } , \alpha$ , and $\beta$ are computed afterwards on the held-out region of condition space.

## 2.3 Domain-specific interaction targets

The additive contrast is realized differently across our four studied systems (described in Section 2.5):

• Physiological time series: given continuous factors without a natural reference level, we use the fANOVA decomposition directly in the waveform representation, such that $I _ { A B }$ is itself a time series.

• Physiological readout: a nonlinear physiological readout $g$ is applied to the generated waveforms before the fANOVA decomposition, yielding $I _ { A B } ^ { g } { } _ { \ }$

• Single-cell perturbation: we use unperturbed controls as the reference, with $I _ { A B }$ computed from condition-level mean expression vectors, thus yielding a vector over genes.

• Molecular conformers: as graphs with different atom counts cannot be averaged in coordinate space, we define the interaction on a readout: the mean heavy-atom radius of gyration over conformers; as reference, we use the unedited parent molecule

• Drug dose–response: the zero dose of the combined drug serves as the reference, and we define interaction relative to Bliss independence rather than additively (see below).

Additivity is not a sensible null for drug combinations as fractional effects are bounded to [0, 1], meaning simple addition of two drug effects can violate these bounds. Consequently, Bliss independence is commonly used in pharmacology, which assumes that two drugs act independently [Bliss, 1939]. Concretely, we define $E _ { A }$ and $E _ { B }$ as normalized $( \mathrm { t o } \ [ 0 , 1 ] )$ fractional drug effects such as inhibition, where 0 indicates no effect and 1 fully inhibiting. When drugs act independently, their combined effect is

$$
E _ { A B } ^ { \mathrm { B l i s s } } = E _ { A } + E _ { B } - E _ { A } E _ { B } .
$$

Equivalently, Bliss independence means multiplication of surviving fractions, $1 - E _ { A B } = ( 1 -$ $E _ { A } ) ( 1 - E _ { B } )$ , which is additive following the readout $g ( E ) = \log ( 1 - E )$ and another example of interactions being representation-dependent (Section 2.1). Next, we can define the interaction as the deviation from this independent effect:

$$
I _ { A B } ^ { \mathrm { B l i s s } } = E _ { A B } - E _ { A B } ^ { \mathrm { B l i s s } } .
$$

Positive values then indicate drug-drug synergy seeing as inhibition is larger than expected and negative values vice versa indicate antagonism.

![](images/0a95d43153a981c7640004909af5bc2d513f77517e5401e5a4acdbbe352e8c95.jpg)  
observed held out removed  
Figure 2: Held-out designs across the four systems. Please see Section 2.5 for details.

## 2.4 Models

We train conditional flow matching models following Lipman et al. [2023] with minor variations across domains which we summarize in Appendix Table 5. In the standard setting, we learn a velocity field $v _ { \theta } ( x _ { \tau } , \tau , c )$ that transports $p _ { 0 } = \mathcal { N } ( 0 , 1 )$ to the conditional data distribution $p _ { 1 } ( \cdot | c )$ along the probability path $x _ { \tau } = ( 1 - \tau ) z _ { 0 } + \tau z _ { 1 }$ where $\tau \in [ 0 , 1 ]$ indicates the flow matching time. Models are trained using an MSE loss and, unless stated otherwise, sampled by numerical integration of the learned flow. Table 5 also includes conditioning, optimization, model architectures, and prediction targets. For molecular conformers, we use a published equivariant flow-matching model conditioned on molecular graphs [Hassan et al., 2024].

We compare three ways of obtaining a prediction for the joint condition AB:

• Joint model is trained and evaluated with conditioning directly on AB and explicitly represents interaction.

• Composed composes the learned conditional velocity fields

$$
v _ { A B } ^ { \mathrm { c o m p } } ( x , t ) = v _ { A } ( x , t ) + v _ { B } ( x , t ) - v _ { 0 } ( x , t )
$$

at every sampling step and is a widely-used compositional approach for unseen combinations Liu et al. [2022], Du et al. [2023]. Importantly, $\hat { I } ^ { \mathrm { f u l l } }$ is not necessarily zero as the composed and single-factor fields induce different nonlinear sampling trajectories.

• Shared-model AB relies on same network as used for the Composed evaluation, but instead directly queries AB. It thereby functions as a control which allows us to separate the effect of the shared training from the effect of the compositional step.

## 2.5 Datasets

We study four systems, which cover different aforementioned identification regimes while having well defined interaction targets (Figure 2). The physiological simulator of photoplethysmogram (PPG; widely used in wearables) maps melanin and blood-volume fractions to 16-channel PPG waveforms with 64 timesteps and provides exact simulator ground truths and a tunable interaction strength. Specifically, we use fANOVA to decompose the simulated grid as $Y = { \tilde { Y } } + I$ (Section 2.2) and define $Y _ { \gamma } = \tilde { Y } + \gamma I .$ Consequently, γ modulates the interaction while leaving both main effects and the grand mean unchanged. Then, γ = 1 refers to the original grid, with $\gamma = 0 . 5$ denoting a halved interaction strength and $\gamma = 0$ a purely additive setting. We construct a simplified version of PPGen [Behrmann et al., 2026], where tissue composition determines optical properties in a multilayer skin model. For each channel (ch), we apply a standard normalized pulse-amplitude (AC/DC) readout [Allen, 2007, Coutrot et al., 2021]:

$$
\left( \frac { \mathrm { A C } } { \mathrm { D C } } \right) _ { c h } ( Y ) = \frac { \operatorname* { m a x } _ { t } Y _ { c h , t } - \operatorname* { m i n } _ { t } Y _ { c h , t } } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } Y _ { c h , t } } .
$$

![](images/015864548355c9b382c7feb6b7a5af7188546dcd890c1eb15d84d5a02a06bb42.jpg)

![](images/5dc71aec9b63096b5dfbb75edf085347175ccdfb2afe2b9e534e7f240a1f3cec.jpg)

![](images/8171cbd93e6b4f877a305c67504caf483ae6e118433844e62e6bde14a88de29a.jpg)  
Figure 3: Learning dynamics for PPG targets in the training set (ID) and unseen targets (OOD) for separate regions of condition space. We show mean dynamics of the joint model across three random seeds, with faint lines indicating single-seed runs.

Briefly, AC is the heartbeat-synchronous change, mainly attributed to pulsatile blood volume. Meanwhile, DC is the baseline optical signal from skin, melanin, and other slowly varying effects. The readout therefore asks the physical question of how large the pulsatile optical change is relative to the baseline reflectance, and how this jointly depends on melanin and dermal blood volume.

We include three further datasets enabling investigations in measured data under smoothness-identified and relationally-identified holdouts. Specifically, the dose-response dataset consists of longitudinal yeast growth responses (96-dimensional) to pairs of drugs administered on 8 × 8 dose grids [Cokol et al., 2011]. We hold-out and predict 2 × 2 and 3 × 3 blocks for seen drug pairs. The Norman CRISPRa Perturb-seq dataset contains single-cell responses to single- and paired gene perturbations [Norman et al., 2019]. For the held-out double perturbations, other combinations of the targets are either included in the training set (’graph-supported’ regime) or excluded (’cold-start’ regime). We predict in a 500-dimensional PCA space. Finally, GEOM provides ensembles of 3D conformers for drug-like molecules [Axelrod and Gomez-Bombarelli, 2022]. We use chemical squares in which two edits are applied to a shared molecular core, enabling prediction of conformers for the double-edited molecule. We keep held-out targets fixed while varying the training support: parent molecules plus structurally similar molecules (’neighborhood’), ’parents only’, or neither (’distant’). As edits are commonly observed on hundreds of other molecular cores, relational support is constant while loca smoothness support varies. We provide more details on these datasets in Appendix C.

For the empirical data, we report the anchored estimate to approximate more realistic scientific workflows. For PPG, only generated estimates are applicable. Interaction magnitudes vary across the systems, with interaction-to-total norm ratios of 5.6% (PPG), 23.2% (dose-response; median assay), 46.1% (perturbation; median pair), and 47.4% (conformer; median family). This corresponds to 0.3%, 5%, 21%, and 22% of response energy respectively. For $Y _ { 0 } ,$ we use the lowest melanin/blood volume value, untreated responses, unperturbed conditions, and unedited parent molecule.

## 3 Results

## 3.1 Aggregate endpoint error conceals interaction learning dynamics

For the synthetic PPG dataset, we hold-out two corners of the condition space (low-low (LL) and high-high (HH)), referring to the melanin and blood volume values) and investigate errors across training (Figure 3). Whereas additive and endpoint errors appear to have largely converged by 10K optimizer steps, the interaction error has not. Specifically, for seen targets, the model needs at least an order of magnitude longer training. For unseen targets, we observe two distinct dynamics: when LL is held-out, interaction error stays high and plateau’s by 50K steps. In contrast, for unseen HH targets, interaction error is approximately 7 times smaller by 20K steps,

We also find such decoupling between trajectories of endpoint and interaction errors in measured data, albeit in the opposite direction. For held-out Norman pairs, $\mathcal { E } _ { I }$ begins to increase after ∼30K steps while $\mathcal { E } _ { \mathrm { e n d } }$ continues to decrease until ∼50K steps (Appendix Figure 7). Consequently, checkpoint selection using endpoint error selects models whose interaction predictions are already degrading. In the well-constrained setting of dose-response prediction, no clear decoupling is observed.

![](images/e4c8231392e774891388b883792143390dc4071535378438fd1792df16386a93.jpg)  
Figure 4: Interaction errors for unseen PPG targets. The left panel shows errors in the output (waveform) representation and readout (AC/DC) representation, while the right panel shows a magnified view of the (same) waveform results. The interaction strength γ was modulated $1  0$ and is indicated by a dark → bright marker color gradient. Mean over training and inference seeds.

## 3.2 Distinguishing interaction-prediction failures

Distinct recoverability of a shared condition space. We further investigate these bifurcated results by inspecting model interaction recovery and hallucination (Figure 4). We observe stark differences, with joint models largely recovering the interaction for HH but not LL targets (α ≈ 1.00 and $\alpha \approx 0 . 3 1$ at $\gamma = 1$ , respectively). Hallucinated structure is also higher for LL $( \beta \stackrel { - } { \approx } 0 . 0 8$ vs 0.43). For this LL corner, we provide the produced maps of interaction in Appendix Figure 6, showing that composed and shared-model AB baseline significantly underestimate interaction amplitude.

These results showcase that the extend to which interactions can be extrapolated based on a joint training support can strongly depend on the modeled domain. The two corners are geometrically matched but not physically equivalent. We find that a simple log-additive continuation rule selected based on the training support transfers to the held out HH corner. LL, in contrast, approaches a physical boundary where it is not (Appendix D). This maps to the aforementioned identifiability problem.

Modulating interaction strengths. The PPG simulator allowed us to do a controlled investigation of different interaction strengths $( \gamma )$ . However, at $\gamma = 0$ , we have $I = 0$ and thus $\mathcal { E } _ { I }$ and α are not applicable. As such, to quantify what interaction a model predicts in the absence of a true interaction to recover, we compare against the full-strength $\gamma = 1$ truth $I ^ { ( 1 ) }$ , with $\alpha ^ { ( 1 ) } = \langle \hat { I } , I ^ { ( 1 ) } \rangle / \| I ^ { ( 1 ) } \| ^ { 2 }$ and $\beta ^ { ( 1 ) } = \| \hat { I } - \alpha ^ { ( 1 ) } I ^ { ( 1 ) } \| / \| I ^ { ( 1 ) } |$ . Ideally, a model shows $\alpha ^ { ( 1 ) } \approx \gamma \mathrm { a n d } \beta ^ { ( 1 ) } \approx 0$

We observe that reducing interaction strength leads to monotonic decreases in $\alpha ^ { ( 1 ) }$ , indicating models do somewhat calibrate their magnitudes based on the training support. For LL, hallucinated structure similarly decreases. Importantly, at $\gamma = 0 , \beta ^ { ( 1 ) } > 0$ for all models. Thus, when the true interaction has been fully removed, models still generate structure, with joint $\beta _ { L L } ^ { ( 1 ) } \approx 0 . 1 9$ and $\beta _ { H H } ^ { ( 1 ) } \approx 0 . 0 8$ This suggests that models’ inductive bias produces a baseline level of non-additivity.

Strong predictions in output space can mask failed recovery in readout space. While the joint model predicts interaction for unseen HH targets well in waveform space, results can differ significantly after a physiological readout, with $\alpha ^ { w f } \overset { - } { \approx } 1 . 0 0 \to \alpha ^ { A C / D C } \approx \overset { - } { 0 . 5 0 }$ and $\beta ^ { w f } \approx 0 . 0 7 $ $\beta ^ { \mathrm { { \overline { { A } } C / } } D C } \approx 1 . 9 0$

## 3.3 Interaction errors across regimes

Finally, we study interaction recovery and hallucination across generalization regimes and varying training supports. For PPG, whereas previously we held out 25% of the condition grid, we now extend this to 50% and 75% and train only for 100K steps due to computational costs.

![](images/ff9a566a17f86d9fe2ffa2e40070787645decab5b8efb4d090f0c4d000f3bda4.jpg)  
Figure 5: Interaction recovery and hallucination for unseen targets across systems for joint models. Contour lines indicate $\mathcal { E } _ { I }$ and are not applicable to $\gamma = 0$

Table 2: Normalized endpoint and relative interaction error of joint models, with 1 indicating prediction of a control response or zero interaction respectively. Mean and SDs across training seeds. For PPG, we always use 100K step checkpoints for consistency.
<table><tr><td>System / holdout</td><td> $\underline { { \mathcal { E } _ { \mathrm { e n d } } } }$ </td><td> $\overline { { \mathcal { E } _ { I } } }$ </td></tr><tr><td>PPG LL, 25%</td><td> $\overline { { 0 . 1 0 4 \pm 0 . 0 0 6 } }$ </td><td> $\overline { { 0 . 7 9 8 \pm 0 . 0 6 5 } }$ </td></tr><tr><td>PPG LL, 50%</td><td> $0 . 1 4 4 \pm 0 . 0 0 0 4$ </td><td>0.960 ± 0.009</td></tr><tr><td>PPG LL, 75%</td><td> $0 . 1 6 5 \pm 0 . 0 0 7$ </td><td>0.992 ± 0.031</td></tr><tr><td>PPG HH, 25%</td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $0 . 1 2 5 \pm 0 . 0 3 5$ </td></tr><tr><td>PPG HH, 50%</td><td> $0 . 0 1 5 \pm 0 . 0 0 1$ </td><td> $0 . 2 0 5 \pm 0 . 0 2 1$ </td></tr><tr><td>PPG HH, 75%</td><td> $0 . 0 2 7 \pm 0 . 0 0 2$ </td><td> $0 . 2 5 1 \pm 0 . 0 2 6$ </td></tr><tr><td>Dose-response  $\overline { { 2 \times 2 } }$ </td><td> $\overline { { 0 . 0 4 5 \pm 0 . 0 0 2 } }$ </td><td> $\overline { { 0 . 2 1 6 \pm 0 . 0 1 6 } }$ </td></tr><tr><td> $\mathrm { D o s e - r e s p o n s e 3 \times 3 }$ </td><td> $0 . 0 5 7 \pm 0 . 0 0 3$ </td><td> $0 . 3 0 0 \pm 0 . 0 0 7$ </td></tr><tr><td>Perturb. supported</td><td> $\overline { { 0 . 5 8 2 \pm 0 . 0 2 7 } }$ </td><td> $\overline { { 0 . 9 1 9 \pm 0 . 0 1 8 } }$ </td></tr><tr><td>Perturb. cold-start</td><td> $0 . 5 6 1 \pm 0 . 0 2 1$ </td><td> $1 . 0 5 2 \pm 0 . 0 4 4$ </td></tr><tr><td>Conformers neighb.</td><td> $\overline { { 0 . 4 3 6 \pm 0 . 0 0 3 } }$ </td><td> $\overline { { 0 . 7 0 4 \pm 0 . 0 0 5 } }$ </td></tr><tr><td>Conformers parents</td><td> $0 . 4 6 7 \pm 0 . 0 1 5$ </td><td> $0 . 7 5 4 \pm 0 . 0 2 4$ </td></tr><tr><td>Conformers distant</td><td> $0 . 5 1 3 \pm 0 . 0 0 3$ </td><td> $0 . 8 2 8 \pm 0 . 0 0 5$ </td></tr></table>

In most systems, as identification gets harder, we observe reduced interaction recovery (α), with distinct delta magnitudes across systems. For PPG $( \gamma = 1 )$ LL this nearly entirely removes weak recovery $( 0 . 3 0  0 . 0 5 )$ , indicating the importance of locality for extrapolation. Meanwhile, HH is recoverable even far from the training support (1.03). For dose-response, we observe only a very weak drop $( 0 . 9 8  0 . 9 7 )$ , with sufficient smoothness constraint persisting to the larger held-out block. We confirm this by fitting a standard (deterministic) regression baseline and find nearly identical recovery $( 0 . 9 8  0 . 9 7 )$ . For perturbations, recovery is again reduced $( 0 . 6 6  0 . 4 1 )$ , highlighting the importance of other observed combinations. We observe an exception for molecular conformers, with less local support leading to increases in $\alpha ( 0 . 6 9  0 . 7 0  0 . 7 6 )$ . We find this results from exaggerated differences between scaffold families, while within-family recovery remains attenuated (further details in Appendix E).

Interestingly, we observe increased hallucination across systems: PPG HH $\beta 0 . 1 2  0 . 2 5$ , doseresponse $0 . 1 8  0 . 2 4$ , perturbations $0 . 8 0  0 . 8 6$ , and conformers $0 . 6 3  0 . 6 9  0 . 7 9$ . This suggests hallucinated structure not to be a fixed property of the model and that they do not merely add a baseline level of hallucinated structure, but that this scales as targets get less identifiable.

Comparing endpoint and interaction errors instead, reveals an important further insight (Table 2). Namely, for PPG LL, enlarging the held-out region of condition space maintains non-trivial endpoint prediction $( \mathcal { E } _ { \mathrm { e n d } } \ll 1 )$ , while making the interaction prediction no better than predicting no interaction at all $( \mathcal { E } _ { I } \approx 1 )$ ).

## 4 Discussion

We have investigated how flow models learn interactions and generate them for unseen targets using error decompositions. We have found multiple interesting dynamics: aggregate endpoint errors can conceal interaction errors, scientifically-relevant readouts can materially change interaction prediction scores, and found that both recovery and hallucination depend on identifiability regimes.

Our findings suggest that scientific evaluation should test interaction recovery directly in the representation space which is used to draw conclusions. With respect to data, individual factors appear insufficient; either collecting observations or appropriate structural assumptions are necessary to constrain unseen interactions. Computing interaction recovery and hallucination in addition to aggregate errors can help distinguish succesful extrapolation from plausible predictions which miss the scientific effects of interest.

Limitations of the presented work include our restriction to two-factor interactions and a reliance on a mean-based (rather than distributional) contrast.

## Acknowledgments and Disclosure of Funding

This research was funded by Gemeinnützigen Hertie-Stiftung and the Deutsche Forschungsgemeinschaft (DFG) through FOR 5187 (project number 442075332). Additional support was provided by the Machine Excellence Cluster and DFG through the Germany’s Excellence Strategy (EXC 2064 - project number 390727645) and the following projects: CRC 1404 (project number 414984028), TRR 265 (project number 402170461), and RU 5363 (project number 459422098). Finally, we are grateful to all dataset providers for access.

## References

Constantin Ahlmann-Eltze, Wolfgang Huber, and Simon Anders. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. Nature Methods, 22(8): 1657–1661, 2025.

Sumukh K Aithal, Pratyush Maini, Zachary C Lipton, and J Zico Kolter. Understanding hallucinations in diffusion models through mode interpolation. Advances in neural information processing systems, 37:134614–134644, 2024.

John Allen. Photoplethysmography and its application in clinical physiological measurement. Physiological measurement, 28(3):R1–R39, 2007.

Simon Axelrod and Rafael Gomez-Bombarelli. Geom, energy-annotated molecular conformations for property prediction and molecular generation. Scientific data, 9(1):185, 2022.

Jens Behrmann, Maria R Cervera, Antoine Wehenkel, Andrew C Miller, Albert Cerussi, Pranay Jain, Vivek Venugopal, Shijie Yan, Guillermo Sapiro, Luca Pegolotti, et al. Inferring optical tissue properties from photoplethysmography using hybrid amortized inference. Machine Learning: Health, 2(1):015012, 2026.

Chester I. Bliss. The toxicity of poisons applied jointly. Annals ofApplied Biology, 26(3):585–615, 1939. doi: 10.1111/j.1744-7348.1939.tb06990.x.

Jack Brady, Julius von Kügelgen, Sébastien Lachapelle, Simon Buchholz, Thomas Kipf, and Wieland Brendel. Interaction asymmetry: A general principle for learning composable abstractions. In International Conference on Learning Representations, volume 2025, pages 46606–46660, 2025.

Murat Cokol, Hon Nian Chua, Murat Tasan, Beste Mutlu, Zohar B. Weinstein, Yo Suzuki, Mehmet E. Nergiz, Michael Costanzo, Anastasia Baryshnikova, Guri Giaever, Corey Nislow, Chad L. Myers, Brenda J. Andrews, Charles Boone, and Frederick P. Roth. Systematic exploration of synergistic drug pairs. Molecular Systems Biology, 7:544, 2011. doi: 10.1038/msb.2011.71.

Maxime Coutrot, Emmanuel Dudoignon, Jona Joachim, Etienne Gayat, Fabrice Vallee, and François Depret. Perfusion index: physical principles, physiological meanings and clinical implications in anaesthesia and critical care. Anaesthesia Critical Care & Pain Medicine, 40(6):100964, 2021.

Kefan Dong and Tengyu Ma. First steps toward understanding the extrapolation of nonlinear models to unseen domains. arXiv preprint arXiv:2211.11719, 2022.

Yilun Du, Conor Durkan, Robin Strudel, Joshua B. Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and MCMC. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 8489–8510. PMLR, 2023.

Sam Gijsen, Michał Łukomski, Marc-André Schulz, and Kerstin Ritter. Flow matching with in-context priors for out-of-distribution brain dynamics, 2026.

Majdi Hassan, Nikhil Shenoy, Jungyoon Lee, Hannes Stärk, Stephan Thaler, and Dominique Beaini. Et-flow: Equivariant flow-matching for molecular conformer generation. Advances in neural information processing systems, 37:128798–128824, 2024.

Siyu He, Yuefei Zhu, Daniel Naveed Tavakol, Haotian Ye, Yeh-Hsing Lao, Zixian Zhu, Cong Xu, Shradha Chauhan, Guy Garty, Raju Tomer, et al. Squidiff: predicting cellular development and responses to perturbations using a diffusion model. Nature methods, 23(1):65–77, 2026.

John B. Ingraham, Max Baranov, Zak Costello, Karl W. Barber, Wujie Wang, Ahmed Ismail, Vincent Frappier, Dana M. Lord, Christopher Ng-Thow-Hing, Erik R. Van Vlack, Shan Tie, Vincent Xue, Sarah C. Cowles, Alan Leung, João V. Rodrigues, Claudio L. Morales-Perez, Alex M. Ayoub, Robin Green, Katherine Puentes, Frank Oplinger, Nishant V. Panwar, Fritz Obermeyer, Adam R. Root, Andrew L. Beam, Frank J. Poelwijk, and Gevorg Grigoryan. Illuminating protein space with a programmable generative model. Nature, 623(7989):1070–1078, 2023. doi: 10.1038/ s41586-023-06728-8.

Dominik Klein, Jonas Simon Fleck, Daniil Bobrovskiy, Lea Zimmermann, Sören Becker, Alessandro Palma, Leander Dony, Alejandro Tejada-Lapuerta, Guillaume Huguet, Hsiu-Chuan Lin, et al. Cellflow enables generative single-cell phenotype modeling with flow matching. bioRxiv, pages 2025–04, 2025.

Sébastien Lachapelle, Divyat Mahajan, Ioannis Mitliagkas, and Simon Lacoste-Julien. Additive decoders for latent variables identification and cartesian-product extrapolation. Advances in Neural Information Processing Systems, 36:25112–25150, 2023.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Samuel Lippl and Kim Stachenfeld. When does compositional structure yield compositional generalization? a kernel theory. arXiv preprint arXiv:2405.16391, 2024.

Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B. Tenenbaum. Compositional visual generation with composable diffusion models. In Computer Vision – ECCV 2022, volume 13677 of Lecture Notes in Computer Science, pages 423–439. Springer, 2022. doi: 10.1007/ 978-3-031-19790-1\_26.

Thomas M. Norman, Max A. Horlbeck, Joseph M. Replogle, Alex Y. Ge, Albert Xu, Marco Jost, Luke A. Gilbert, and Jonathan S. Weissman. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. Science, 365(6455):786–793, 2019. doi: 10.1126/science. aax4438.

Maya Okawa, Ekdeep S. Lubana, Robert P. Dick, and Hidenori Tanaka. Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task. In Advances in Neural In formation Processing Systems, volume 36, pages 50173–50195, 2023. doi: 10.52202/075280-2182.

Core F Park, Maya Okawa, Andrew Lee, Hidenori Tanaka, and Ekdeep S Lubana. Emergence of hidden capabilities: Exploring learning dynamics in concept space. Advances in Neural Information Processing Systems, 37:84698–84729, 2024.

Kristina Preuer, Richard PI Lewis, Sepp Hochreiter, Andreas Bender, Krishna C Bulusu, and Günter Klambauer. Deepsynergy: predicting anti-cancer drug synergy with deep learning. Bioinformatics, 34(9):1538–1546, 2018.

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.

Yusuf Roohani, Kexin Huang, and Jure Leskovec. Predicting transcriptional outcomes of novel multigene perturbations with gears. Nature biotechnology, 42(6):927–935, 2024.

Kenneth J. Rothman, Sander Greenland, and Alexander M. Walker. Concepts of interaction. American Journal of Epidemiology, 112(4):467–470, 1980. doi: 10.1093/oxfordjournals.aje.a113015.

Philipp S. L. Schäfer, Kendall A. Reid, Zach Boldyga, Ekin D. Aksu, Hugo Hakem, and Julio Saez-Rodriguez. Towards principled evaluation of single-cell perturbation prediction models. bioRxiv, 2026. doi: 10.64898/2026.07.23.740433.

Parshin Shojaee, Kazem Meidani, Shashank Gupta, Amir Barati Farimani, and Chandan Reddy. Llmsr: Scientific equation discovery via programming with large language models. In International Conference on Learning Representations, volume 2025, pages 16054–16085, 2025.

Tyler J VanderWeele and Mirjam J Knol. A tutorial on interaction. Epidemiologic methods, 3(1): 33–72, 2014.

Ramon Viñas Torné, Maciej Wiatrak, Zoe Piran, Shuyang Fan, Liangze Jiang, Sarah A Teichmann, Mor Nitzan, and Maria Brbic. Systema: a framework for evaluating genetic perturbation response´ prediction beyond systematic variation. Nature Biotechnology, pages 1–10, 2025.

Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E. Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, Nikita Hanikel, Samuel J. Pellock, Alexis Courbet, William Sheffler, Jue Wang, Preetham Venkatesh, Isaac Sappington, Susana Vázquez Torres, Anna Lauko, Valentin De Bortoli, Emile Mathieu, Sergey Ovchinnikov, Regina Barzilay, Tommi S. Jaakkola, Frank DiMaio, Minkyung Baek, and David Baker. De novo design of protein structure and function with RFdiffusion. Nature, 620(7976):1089–1100, 2023. doi: 10.1038/s41586-023-06415-8.

Thaddäus Wiedemer, Prasanna Mayilvahanan, Matthias Bethge, and Wieland Brendel. Compositional generalization from first principles. In Advances in Neural Information Processing Systems, volume 36, pages 6941–6960, 2023. doi: 10.52202/075280-0304.

Chenglei Yu, Chuanrui Wang, Bangyan Liao, and Tailin Wu. scDFM: Distributional flow matching model for robust single-cell perturbation prediction, 2026. ICLR 2026 poster.

## A Interaction Maps

![](images/bfc608fa8fd2543f6c66ecbf8c3dbb2cbcebe1d17c0b62d5be783fa0bd16fda1.jpg)

![](images/c530fb4fc7bb97e2ec0b030769c6bb756ec63ccd30b5a903ce997143905dffbd.jpg)

![](images/67e56b5ed3118f94dac2770ac4c3f20614f6a10b27177854fb8982d337ac1162.jpg)

![](images/487c2321a79a8879b8dba260dbae14efc0adb2b8a775a9523a4106d3480cad37.jpg)  
Figure 6: Maps showing the non-additive interaction projected onto the primary truth interaction mode. Colors indicate the signed interaction amplitude, normalized by the $\gamma = 1$ low-low truth RMS. The dotted square indicates the held-out region of condition space (corresponding to PPG LL 25%). To aid visualisation, we clip the colormap to [0.5%, 99.5%] of the maximum amplitude.

## B Additional Learning Dynamics

![](images/ddb568c5bd9a6fb6c889a6308d7ce177c2aeee18b224ccc7ac69c5bedd4d7e57.jpg)

![](images/81e30245dda26b6ba7e4dcf373e08f8e6526812fe88d448dc879c955185e8b0c.jpg)

![](images/edf9bbec6437ece2ecb9dd514f110151bbe4a9e29dc456597c97be7dd7c3f89c.jpg)  
Figure 7: Held-out evaluation across the training trajectory for the Cokol et al. [2011] dose-response setting and for the Norman et al. [2019] perturbation data. While the former shows synchronous dynamics for aggregate and non-additive metrics due to a smoothness-constrained held-out region of condition space, the latter shows decoupling. Specifically, whereas $\mathcal { E } _ { I }$ begins to increase after 30K steps, ${ \mathcal { E } } _ { e n d }$ continues to decrease up to 50K steps. The shading indicates checkpoint selection based on validation endpoint error, highlighting that using endpoint error for model selection can be suboptimal for interaction prediction.

## C Dataset details

## C.1 PPG simulator.

The PPG data are generated using deterministic replay of a fixed Monte Carlo photon-path bank using a three-layer skin model (simplified from Behrmann et al. [2026]). A path bank contains $1 0 ^ { 8 }$ photon histories for 35 scattering settings with linear interpolation between adjacent settings, and Beer-Lambert reweighting applying absorption. This makes the simulator deterministic conditional on the path bank. We generate 64 time points for four source-detector distances (3-6 mm) across four wavelengths (525, 660, 850, 940 nm).

The condition space is made up of epidermal melanin (0.25–14%) and dermal blood volume (BV2; 0.1–3.9%). Meanwhile, arterial oxygen saturation is fixed at 80% and subcutaneous blood volume at 4.05%. The temporal blood-volume modulation follows a normalized arterial luminal-area waveform (from the PWDB database; age-45). Dermal and subcutaneous blood volumes are multiplied by $1 + 0 . 0 2 w ( t )$ with $w ( t ) \in [ 0 , \bar { 1 } ]$

First, we simulate a uniform 101 × 101 Mel-BV2 grid. We apply balanced functional ANOVA to the clean $\gamma = 1$ reference grid, writing $Y _ { 1 } = { \tilde { Y } } + I , $ , where $\tilde { Y }$ contains the grand mean and the Mel and BV2 main-effect functions, and I is the residual interaction. The controlled interaction-strength grids are then defined as $Y _ { \gamma } = \tilde { Y } + \gamma I .$ . As such, $\gamma = 1$ is the original grid, with $\gamma < 1$ reducing its interaction while preserving the mean and main effects. $1 0 ^ { 6 }$ Mel-BV2 pairs are drawn independently and uniformly across the condition space. We obtain the PPG waveform for every pair by bilinearly interpolating each detector-wavelength-time coordinate of the reference grid.

Table 3: PPG simulator configuration.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Tissue geometry</td><td>Epidermis/dermis/subcutaneous thicknesses: 0.20/1.50/18.30 mm; 0.05- mm isotropic voxels;  $1 8 \times 1 8 \times 2 0 – m m$  domain.</td></tr><tr><td>Source and detectors</td><td>Normally incident uniform disk source, radius 1.6 mm; four spherical detectors of radius 0.25 mm at 3, 4, 5, and 6 mm.</td></tr><tr><td>Optical constants</td><td>Refractive index  $n = 1 . 4 ;$  anisotropy  $g = 0 . 9 ;$  shared reduced-scattering law  $\mu _ { s } ^ { \prime } ( \lambda ) = 6 . 2 5 ( \lambda / 1 0 0 0 \mathrm { n m } ) ^ { - 1 \hat { . } \hat { 4 } } \mathrm { c m } ^ { - 1 } .$ </td></tr><tr><td>Absorbers</td><td>Eumelanin in epidermis; water and lipid in all layers; collagen in dermis; oxygenated/deoxygenated hemoglobin in dermis and subcutaneous tissue.</td></tr><tr><td>Photon-path bank</td><td>35 scattering settings;  $1 0 ^ { 8 }$  photons per setting; 5-ns MCX time window; fixed paths reused for absorption replay.</td></tr></table>

Splitting. We use nested low-low or high-high rectangles to create increasingly larger hold-out regions of condition space: $2 5 \%  5 0 \%  7 5 \%$ . These contain 735k/15k, 490k/10k, and 245k/5k training/validation samples, respectively. For final evaluation, we use a consistent held-out corner of condition space such that the test region does not change as the training support shrinks.

## C.2 Drug-response data

The Cokol et al. [2011] yeast screenings consist of 200 drug-pair assays across 33 drugs, with 175 different-drug assays and 25 same-drug controls. Each assay is an $8 \times 8$ dose grid, with every cell consisting of a 96D growth curve. This results from 15 minute measurements across 24 hours, yielding 12800 total trajectories.

Splitting. For each assay we hold out a fixed set of blocks (rather than cross-validation). The $2 \times 2$ hold-out yields [11577, 759, 464] train/validation/test trajectories, with $3 \times 3$ sample sizes being [11077, 769, 954]. The same 106 assays are used to compare across the two block sizes. Untreated, single-drug, and same-drug conditions are included in both training sets.

## C.3 Single-cell perturbation data

The Norman et al. [2019] K562 CRISPRa Perturb-seq dataset consists of 91205 cells and 5045 genes, including controls, single perturbations, and 131 double perturbations which involves 73 genes. The double-perturbation conditions contain 49-1030 cells (median=257). Models use a 500D PCA representation fitted to 20000 control and single-perturbation cells. For evaluation, we decode predictions to the 5045-gene log-expression space.

Splitting. We use cross-validation with four outer folds holding out each double perturbation once, while all controls and single perturbations are always included in the training set. Train/validation/test sample sizes range between [78539-80391, 2628-3770, 7348-9992] cells. Of the 131 test pairs, 113 are ’graph-supported’ meaning that both genes are included in other training pairs. For the remaining 18 ’cold-start’ pairs, at least one of the two genes has no observed training pairs. The four folds are repeated across three training seeds.

## C.4 Molecular conformer data

GEOM contains conformer ensembles of drug-like molecules [Axelrod and Gomez-Bombarelli, 2022]. We use a pool of 26830 molecules with at least one retained conformer. The target distribution for a molecule is uniform over its 30 conformers which have the highest CREST weights (following Hassan et al. [2024]), sampled randomly during training. As a readout, we use the heavy-atom radius of gyration $( R _ { g } , \mathring \mathrm { A } )$ over conformers. For generated estimates, we average $R _ { g }$ across 128 draws per molecule.

Chemical squares. Molecules are decomposes into a core with two attachment sites, each carrying one substituent. A ’square’ is made up of a parent molecule (00), two single edits each swapping the substituent at one site (10, 01), and the double edit (11). From a set of 3319 squares, we retain those for which all corners share a scaffold and have $\geq 1 0$ conformers. We select 30 scaffold families with five squares per family, which yields 150 test queries with 384 unique parent molecules.

Splits. The ’neighborhood’ of the test squares is specified as the union of the 30 scaffold families and all pool molecules with Morgan fingerprint (radius 2, 2048 bits, chirality-aware) Tanimoto similarity $\geq 0 . 6$ to a square corner (4504 molecules). Each of the three arms contain 22076 training molecules of which 17572 are shared. The Neighborhood arm adds all 4504 neighborhood molecules (including the 384 parents), while Parents only adds the 384 parents and 4120 additional distant controls, and finally Distant adds only 4504 distant controls such that no parent or neighborhood molecule is seen. Controls are matched to the removed molecules across multiple graph properties (heavy atoms, rings, bonds, rotatable bonds, heteroatoms, formal charge, aromatic atoms; all absolute mean differences $\leq 0 . 1 1 )$ . The same validation and target sets were used across the three arms.

## C.5 Evaluation

Prior to computing endpoint and interaction metrics, we average across sampling seeds within each trained model. For PPG, we average across four sampling seeds, for drug-response data we average eight predictions, and for single-cell perturbation data we average eight batches of 24 generated cells. We report variation across training seeds, with PPG using three seeds, drug-response data uses four seeds, single-cell perturbation data using three, and conformers using three.

## D Why do the PPG hold-out corners differ?

To understand why flow models trained on PPG data extrapolated well into the held-out high-high (HH) corner, but not into the geometrically matched low-low (LL) corner, we performed a small check on spatial continuation. Specifically, we use the training support to select from a small set of simple continuation rules. This procedure selected a raw-log-additive rule which constructed HH highly accurately (RMSE= 0.007, interaction cosine=1.00). While the same rule was selected when LL was held out, it failed to extrapolate (RMSE=0.771). This suggests that the response dynamics in the held-out HH corner continue smoothly from observed support, while the LL region has interaction structure which cannot be accurately inferred from the L-shaped training support.

## E Extra results for molecular conformers

Paired differences. As all arms predict the same queries, we can compare them via paired differences which are more precise than arm-specific means. As we move across the arms Neighborhood → Parents only → Distant reduces $\beta , \mathcal { E } _ { I } ,$ , adn ${ \mathcal E } _ { \mathrm { e n d } }$ (family-bootstrap intervals exclude zero). Compared to Distant, the more supported arms also have lower α: removing nearby support increases aligned amplitude, although this increase is outweighed by the misaligned error (β).

Table 4: Anchored GEOM results. Arm values are mean ± SD over three training seeds. Paired differences are mean ± seed SD of per-seed differences with 95% scaffold-family bootstrap intervals in brackets.
<table><tr><td></td><td>α</td><td> $\beta$ </td><td> $\mathcal { E } _ { I }$ </td><td> $\mathcal { E } _ { \mathrm { e n d } }$ </td></tr><tr><td>Neighborhood</td><td> $0 . 6 8 8 \pm 0 . 0 1 4$ </td><td> $0 . 6 3 1 \pm 0 . 0 1 0$ </td><td> $0 . 7 0 4 \pm 0 . 0 0 5$ </td><td> $0 . 4 3 6 \pm 0 . 0 0 3$ </td></tr><tr><td>Parents only</td><td> $0 . 6 9 6 \pm 0 . 0 1 8$ </td><td> $0 . 6 8 9 \pm 0 . 0 1 8$ </td><td> $0 . 7 5 4 \pm 0 . 0 2 4$ </td><td> $0 . 4 6 7 \pm 0 . 0 1 5$ </td></tr><tr><td>Distant</td><td> $0 . 7 5 8 \pm 0 . 0 2 3$ </td><td> $0 . 7 9 1 \pm 0 . 0 0 3$ </td><td> $0 . 8 2 8 \pm 0 . 0 0 5$ </td><td> $0 . 5 1 3 \pm 0 . 0 0 3$ </td></tr><tr><td>Neighb. — parents</td><td> $- 0 . 0 0 7 \pm 0 . 0 1 8$ </td><td> $- 0 . 0 5 9 \pm 0 . 0 2 7$ </td><td> $- 0 . 0 5 0 \pm 0 . 0 2 9$ </td><td> $- 0 . 0 3 1 \pm 0 . 0 1 8$ </td></tr><tr><td></td><td> $[ - 0 . 0 6 2 , 0 . 0 3 1 ]$ </td><td> $[ - 0 . 1 2 2 , - 0 . 0 0 8 ]$ </td><td> $[ - 0 . 1 0 7 , - 0 . 0 1 3 ]$ </td><td> $[ - 0 . 0 6 6 , - 0 . 0 0 8 ]$ </td></tr><tr><td>Parents — distant</td><td> $- 0 . 0 6 2 \pm 0 . 0 2 5$ </td><td> $- 0 . 1 0 2 \pm 0 . 0 2 1$ </td><td> $- 0 . 0 7 4 \pm 0 . 0 2 5$ </td><td> $- 0 . 0 4 6 \pm 0 . 0 1 5$ </td></tr><tr><td></td><td> $\left[ - 0 . 0 9 9 , - 0 . 0 3 1 \right]$ </td><td> $[ - 0 . 1 8 1 , - 0 . 0 5 1 ]$ </td><td> $\left[ - 0 . 1 6 6 , - 0 . 0 1 8 \right]$ </td><td> $[ - 0 . 1 0 3 , - 0 . 0 1 1 ]$ </td></tr><tr><td>Neighb. — distant</td><td> $- 0 . 0 6 9 \pm 0 . 0 0 9$ </td><td> $- 0 . 1 6 1 \pm 0 . 0 0 8$ </td><td> $- 0 . 1 2 4 \pm 0 . 0 0 7$ </td><td> $- 0 . 0 7 7 \pm 0 . 0 0 4$ </td></tr><tr><td></td><td> $[ - 0 . 1 4 7 , - 0 . 0 0 7 ]$ </td><td> $[ - 0 . 2 8 1 , - 0 . 0 7 4 ]$ </td><td> $\left[ - 0 . 2 5 1 , - 0 . 0 4 4 \right]$ </td><td> $[ - 0 . 1 5 5 , - 0 . 0 2 7 ]$ </td></tr></table>

Why does α increase with less support? To understand why, we decompose $\alpha - 1 = [ \overline { { I } } \overline { { e } } +$ $\operatorname { C o v } ( I , e ) ] / { \overline { { I ^ { 2 } } } }$ , with endpoint error $e = \hat { \mu } _ { 1 1 } - \mu _ { 1 1 }$ , and split the covariance into two terms: one within- and one between-family terms over the 30 scaffold families. Of the increase in α resulting from Distant compared to Neighborhood training, a global offset contributes −0.001, withinfamily terms +0.009, and between-family terms +0.062 (≈ 90%). The amplitude of family-average interactions increase from 0.87 (neighborhood) and 0.92 (parents only) overshooting to 1.14 (distant), while the within-family amplitude remains attenuated at 0.64, 0.63, and 0.65. Correlations between predicted and true interactions decrease (overall: 0.74, 0.72, 0.70, within-family: 0.70, 0.69, 0.67). This indicates that moving to a Distant training setup therefore mostly rescales coarse, family-level interactions rather than boosting recovery of square-specific interactions. These results once more indicate caution is needed with single metrics and the utility of additionally inspecting $\beta .$

## F Implementation details

Table 5: Implementation details for the four experimental settings. For molecular conformers, we follow [Hassan et al., 2024].
<table><tr><td>Detail</td><td>PPG</td><td>Dose-response</td><td>Perturbation</td><td>Conformers</td></tr><tr><td>Output Architecture</td><td>16 × 64 waveform Conv-patch DiT</td><td>96-point trajectory Symmetric MLP</td><td>PCA-500 expression Residual MLP</td><td>3D coordinates, all atoms ET-Flow equivariant</td></tr><tr><td></td><td></td><td></td><td></td><td>transformer (O(3), post-hoc parity)</td></tr><tr><td>Width / depth</td><td>128 / 4</td><td>256 / 3</td><td>512 /4</td><td>160 / 20</td></tr><tr><td>Attention heads</td><td>4</td><td>N/A</td><td>N/A</td><td>8</td></tr><tr><td>Parameters</td><td>0.86M (both)</td><td>0.51M each</td><td>3.03M</td><td>8.34M</td></tr><tr><td>Conditioning</td><td>2 factors; + presence bits (shared)</td><td>Drug-token set</td><td>Gene-token set</td><td>Molecular graph (atom/bond features)</td></tr><tr><td>Condition set</td><td>AB (Joint); all four (shared)</td><td>0–2 drugs</td><td>0–2 genes</td><td>One graph per molecule</td></tr><tr><td>Flow source</td><td>Gaussian noise</td><td>Gaussian noise</td><td>Measured control (OT-paired)</td><td>Harmonic prior, σ = 0.1, RMSD-aligned</td></tr><tr><td>Prediction target</td><td>Endpoint x1</td><td>Endpoint x1</td><td>Velocity x1 − x0</td><td>Velocity</td></tr><tr><td>Training loss</td><td>MSE; 4-mask MSE (shared)</td><td>Endpoint MSE</td><td>Velocity MSE</td><td>Velocity MSE</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>2 × 10−4</td><td>2 × 10−4</td><td>2 × 10−4</td><td>5 × 10−4 (peak)</td></tr><tr><td>Weight decay LR schedule</td><td>10⁻8</td><td>10⁻4</td><td>10⁻4</td><td>10⁻8</td></tr><tr><td></td><td>5k warm-up + cosine</td><td>Constant</td><td>Constant</td><td>1k warm-up + cosine to 10-6</td></tr><tr><td>Gradient clipping</td><td>None</td><td>5</td><td>1</td><td>Adaptive (ET-Flow gradient-norm queue)</td></tr><tr><td>EMA decay</td><td>0.999</td><td>0.999</td><td>0.999</td><td>None</td></tr><tr><td>Batch size</td><td>512</td><td>512</td><td>512</td><td>32 molecules (16 × 2 accumulation)</td></tr><tr><td>Training budget</td><td>1M steps</td><td>20k steps Best covered-validation</td><td>200k steps / fold Best inner-validation EMA</td><td>200 epochs (138k steps) Final (epoch 200)</td></tr><tr><td>Checkpoint</td><td>Final EMA</td><td>EMA</td><td></td><td></td></tr><tr><td>Primary inference</td><td>Euler</td><td>t = 0 endpoint</td><td>Heun</td><td>Euler</td></tr><tr><td>Inference steps</td><td>128</td><td>1</td><td>20</td><td>50</td></tr><tr><td>Joint prediction</td><td>Direct AB query</td><td>Direct AB query</td><td>Direct AB query</td><td>Direct query of the 11 graph</td></tr><tr><td>Composed prediction</td><td>xA + xB − xρ</td><td>N/A</td><td>vA + vB − v∅</td><td>N/A</td></tr></table>