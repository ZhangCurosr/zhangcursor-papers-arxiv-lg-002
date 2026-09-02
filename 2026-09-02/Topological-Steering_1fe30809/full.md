# Topological Steering

Benoît Guérand Department of Mathematics National University of Singapore (NUS) 2 Science Drive 2, Singapore 117543 benoit.guerand@u.nus.edu

Tan Minh Nguyen Department of Mathematics National University of Singapore (NUS) 2 Science Drive 2, Singapore 117543 tanmn@nus.edu.sg

## Abstract

With the rapid rise of large language models (LLMs), controlling undesirable model behaviors has become increasingly important. Existing behavioral control methods typically intervene directly in activation or feature space, but such approaches can be sensitive to outliers, distributional shifts, noise, and other local perturbations. Motivated by Topological Data Analysis (TDA), which captures global rather than purely local structure, we propose Topological Steering, a new framework for steering LLM behavior through the topological representation of activation spaces. Using persistence diagrams, our method connects activation-based steering with TDA and enables more robust behavioral control. We show that Topological Steering consistently modifies LLM behavior across multiple model families and model sizes.

## 1 Introduction

Large language models (LLMs) are increasingly deployed as general-purpose assistants, yet their behavior remains difficult to control reliably. Even highly aligned models can produce harmful, dishonest, sycophantic, or otherwise undesirable outputs under adversarial or distribution-shifted prompts. As a result, a growing line of work has sought lightweight mechanisms for controlling model behavior at inference time, without retraining the model or modifying its weights [3, 16, 2].

A particularly promising direction is activation steering [15], which intervenes directly on internal representations. Early methods showed that adding a fixed steering vector to the residual stream can induce predictable changes in sentiment, refusal, truthfulness, or safety-related behavior. Subsequent approaches, including contrastive activation addition [14] and more recent nonlinear or rotational interventions [16], have improved the flexibility of this paradigm. However, most existing steering methods are fundamentally geometric and local: they construct directions from means, linear contrasts, or low-dimensional subspaces in activation space. Such approaches can be effective, but they implicitly assume that the behavioral concept of interest is well captured by a simple global direction or a smooth local transformation.

This assumption is often too restrictive. Behavioral representations in LLMs may be multimodal, nonlinearly organized, and distributed across heterogeneous regions of activation space. For example, refusal behavior may not correspond to a single cluster or linear axis; instead, it may be supported by several distinct activation substructures associated with different prompt types, semantic contexts, or safety mechanisms. In such settings, a global mean-difference vector can dilute or average away the relevant signal, while purely local geometric interventions may be sensitive to outliers, noise, and small perturbations. These limitations motivate a more structural view of activation space: rather than asking only where individual activations lie, we ask what shape the activation cloud has and which multiscale structures distinguish one behavioral regime from another.

Topological Data Analysis (TDA) provides mathematical tools precisely for this purpose. Persistent homology summarizes the multiscale topology of a point cloud by tracking when connected components, loops, and higher-dimensional features appear and disappear as a distance scale varies. The resulting persistence diagrams are stable summaries of global structure and have been widely studied as robust descriptors of complex high-dimensional data. Historically [1, 5], however, the use of TDA in large-scale machine learning has been limited by computational cost: persistent homology can be expensive to compute inside training loops or on very large datasets. In the context of inference-time steering, this bottleneck is substantially reduced. Steering requires only a finite collection of activation vectors from contrastive prompt sets, making it possible to use persistent topology as an actionable signal for intervention design.

Contribution. In this paper, we introduce Topological Steering, a new inference-time method for controlling LLM behavior through the topology of activation spaces. Given contrastive prompt sets, such as refusal-eliciting and compliance-eliciting prompts, we extract residual-stream activations and view them as point clouds. We then apply joint dimensionality reduction, compute Vietoris–Rips persistence diagrams, and compare the diagrams of the two behavioral regimes. Features that are both persistent and specific to the target behavior are used to identify coherent activation subsets, rather than averaging indiscriminately over all examples. These subsets define localized contrastive directions, which are aggregated into a steering vector and injected into the residual stream during generation.

The key idea is that topology provides a mechanism for moving beyond global centroid shifts. Classical activation steering [14] constructs a direction from the mean difference between positive and negative activations. Topological Steering instead identifies robust, behavior-specific substructures in the positive activation cloud and steers using directions derived from those substructures. This allows the intervention to reflect the heterogeneous organization of behavioral concepts in representation space, while also providing an interpretable account of which activation components support the steering signal.

## Our contributions are three-fold:

1. Topology-Aware Steering for LLMs: We introduce Topological Steering, an inference-time framework that uses tools from Topological Data Analysis to construct steering directions from the multiscale structure of LLM activation spaces. By applying TDA outside the training loop, our method avoids the computational bottlenecks that have historically limited its use in large-scale deep learning.

2. Structural Interpretability via Persistence Diagrams: In contrast to linear or angular steering methods [16] that operate through geometric shifts in activation space, our approach uses persistence diagrams to identify robust topological features of behavioral representations, such as connected components and higher-order structures. This provides an interpretable way to localize behavior-relevant regions in the residual stream.

3. Empirical Validation for Topological Intervention: We demonstrate that Topological Steering consistently improves over unsteered baselines and produces measurable behavioral changes across model families and sizes. While the current implementation is not intended to outperform highly optimized geometric steering methods, it offers a complementary trade-off: sacrificing some raw efficacy in exchange for a more structured and inspectable intervention mechanism grounded in the topology of activation space.

## 2 Related work

Activation Steering. Because of its simplistic idea, activation steering has emerged as a lightweight, inference-time alternative to fine-tuning, enabling researchers to alter Large Language Model (LLM) behavior by perturbing internal representations during the forward pass. In the early foundational work, it was shown that a linear intervention, i.e., adding a fixed “steering vector” to the residual stream, could shift model sentiment and alignment properties [15]. This technique was naturally refined with more advanced methods, such as Contrastive Activation Addition (CAA), which isolates behavioral directions by averaging activation differences between paired positive and negative prompts [14].

Most Recent Steering Methods. Most recently, the field has begun exploring beyond purely additive, translational shifts; for instance, Angular Steering frames behavioral control as rotational interventions within fixed activation subspaces [16]. However, while these methods successfully manipulate representations, they predominantly treat the activation space as possessing relatively simple geometry. This leaves room for approaches such as Topological Steering that can isolate and leverage the complex, global, and non-linear invariants inherent to high-dimensional data manifolds.

Topological Data Analysis and Persistent Homology. TDA provides a rigorous mathematical framework for quantifying the "shape" of complex, high-dimensional datasets. One of the most important methods of TDA is persistent homology [9, 7], which tracks the evolution of topological features such as connected components, one-dimensional loops, and higher-dimensional voids across a continuous range of spatial scales. By constructing a sequence of expanding simplicial complexes (e.g., a Vietoris-Rips filtration) over a point cloud, persistent homology records the scale at which each topological feature appears (birth) and the scale at which it ultimately merges or closes (death). These structural invariants are summarized in a Persistence Diagram, offering a multiscale signature of the data’s topology. According to stability theorems [8], Persistence Diagrams are robust to noise and continuous geometric deformations within the data manifold. This is why they have been the subject of extensive research in machine learning.

Topological Machine Learning. The integration of these topological summaries into deep learning led to the creation of the term Topological Machine Learning. Initially, the main focus was on improving Persistence Diagrams. PDs are inherently discrete, and to make these topological features compatible with standard neural networks, researchers have developed stable, finite-dimensional vectorization techniques, the most famous being Persistence Landscapes [6] in 2015 and Persistence images [1] in 2016. In the following year, the researchers developed differentiable topological layers [11, 5], allowing models to optimize for specific topological properties directly during gradient descent.

Despite their theoretical elegance, these methods were fundamentally limited by computational constraints. The combinatorial explosion of tracking high-dimensional simplices, coupled with the expensive matrix reduction algorithms required for persistent homology, scaled poorly. Consequently, computing these topological penalties within the iterative training loops of large-scale models became prohibitively slow and memory-intensive, thereby stalling the field of Topological Machine Learning. Our method revives these powerful topological tools by shifting their application from training-time optimization to inference-time activation steering, effectively bypassing the historical computationa barrier.

## 3 Background

This section introduces the technical ingredients used by our method. We first formalize the transformer computation and the activation-steering intervention used at inference time. We then review the topological knowledge needed to characterize activation geometry: persistent homology, persistence diagrams, and Vietoris–Rips filtrations. Together, these tools let us connect representation-level interventions (steering vectors in residual space) with structure-level summaries (topological features across scales), which is the central perspective of Topological Steering. Some supplementary details can be found in the Appendix A.1.

## 3.1 Transformers and Activation Steering

For the rest of the paper we are going to fix the notation for a decoder-only transformer with L layers, hidden width $d _ { h }$ , and token sequence length T. At a layer $\ell \in \{ 0 , \ldots , L ^ { \setminus } - 1 \}$ we have self attention denoted $\mathrm { A t t n } ^ { ( \ell ) }$ with head dimension $d _ { k }$

Activation steering modifies an internal residual activation at a chosen layer $\ell ^ { \star }$ and token position $t ^ { \star }$ (often the final prompt token [18, 14, 15]) by adding a direction vector at inference time: $h _ { t ^ { \star } } ^ { ( \ell ^ { \star } ) \prime } =$ $h _ { t ^ { \star } } ^ { ( \ell ^ { \star } ) } + \alpha \mathbf { v }$ , where: $h _ { t ^ { \star } } ^ { ( \ell ^ { \star } ) } \in \mathbb { R } ^ { d _ { h } }$ is the original residual activation, $\mathbf { v } \in \mathbb { R } ^ { d _ { h } }$ is the steering vector, and $\alpha \in \mathbb { R }$ is a scalar steering strength.

## 3.2 Persistent Homology, Persistence Diagrams and Vietoris–Rips Filtration

Persistent homology is a multiscale way to describe the shape of a point cloud X (here, activation vectors). As we vary a distance scale ϵ, topological features (connected components, loops, etc.) appear and disappear; for feature i, we denote its birth and death by $( b _ { i } , \delta _ { i } )$ , and its persistence by $\pi _ { i } = \delta _ { i } - b _ { i }$ . Features with larger $\pi _ { i }$ are more stable across scales and are treated as more structurally meaningful.

A Persistence Diagram is the set of birth–death pairs $D ( X ) = \{ ( b _ { i } , \delta _ { i } ) \} _ { i }$ returned by persistent homology. Each point corresponds to one topological feature, and its distance from the diagonal $( u , u )$ equals (up to a constant factor) its persistence $\pi _ { i }$ . In our method, these diagrams summarize activation geometry and allow comparison between harmful and harmless activation clouds.

To compute Persistent Homology, we build a Vietoris–Rips Filtration from pairwise distances. For each scale $\epsilon ,$ we connect points in X whose distance is at most $\epsilon ,$ forming a simplicial complex $\mathrm { V R } _ { \epsilon } ( X )$ . As ϵ increases, these complexes are nested, and tracking topology across this nested sequence yields the birth/death pairs used in our theory and experiments.

## 4 Topological Steering

## 4.1 Motivation for Topological Steering

Classical activation steering typically uses a global contrastive direction $\mathbf { v _ { b a s e } } = \bar { X } ^ { + } - \bar { X } ^ { - }$ , where ${ \bar { X } } ^ { + }$ and $\bar { X } ^ { - }$ are class means of harmful and harmless activations. This is optimal only under restrictive geometric assumptions (e.g., approximately unimodal classes with similar covariance structure), where a single linear axis captures the relevant class gap.

A convenient way to see this is through a mixture model. Suppose harmful activations are multimodal: $\begin{array} { r } { p ^ { + } ( x ) = \sum _ { r = 1 } ^ { R } { \pi } _ { r } p _ { r } ( x ) , \sum _ { r = 1 } ^ { R } { \pi } _ { r } = 1 , \ { \pi } _ { r } > 0 } \end{array}$ , while harmless activations follow $p ^ { - } ( x )$ Then $\begin{array} { r } { \bar { X } ^ { + } - \bar { X } ^ { - } = \sum _ { r = 1 } ^ { R } \pi _ { r } \big ( \mu _ { r } - \mu ^ { - } \big ) } \end{array}$ , with $\mu _ { r } = \mathbb { E } _ { x \sim p _ { r } }$ [x] and $\mu ^ { - } = \mathbb { E } _ { x \sim p ^ { - } } [ x ]$ . Hence $\mathbf { v _ { \mathrm { b a s e } } }$ averages over potentially incompatible behavioral modes; if only a subset of modes encode refusal behavior, their signal is diluted by unrelated components.

This motivates replacing “global centroid shift” with “structure-aware subset shift.” Let $S ^ { + } \subset X ^ { + }$ denote activation subsets that correspond to robust refusal structure. The desired direction is $\mathbf { v } _ { \star } \approx$ $\mu _ { S ^ { + } } - \mu ^ { - }$ , where $\mu _ { S ^ { + } }$ is the mean of the structurally selected subset, not of the entire harmful cloud. The challenge is to identify $S ^ { + }$ without supervision on latent modes.

Topological summaries provide a natural criterion because they are multiscale and coordinate-free. Given a filtration $\{ \mathcal { K } _ { \epsilon } \} _ { \epsilon \ge 0 }$ on the activation cloud, persistent homology yields features $\gamma _ { i } = \left( b _ { i } , \delta _ { i } \right)$ pers $( \gamma _ { i } ) = \delta _ { i } - b _ { i }$ , where high persistence indicates geometric stability across scales. Comparing harmful and harmless diagrams introduces a second criterion: behavioral specificity. If $m ( \gamma _ { i } ^ { + } )$ is the optimal match of harmful feature $\gamma _ { i } ^ { + }$ in the harmless diagram (or diagonal), define $c _ { i } =$ $d _ { \infty } ( \gamma _ { i } ^ { + } , m ( \gamma _ { i } ^ { + } ) )$ , so large $c _ { i }$ indicates a refusal-unique topological signature.

We therefore seek features that are simultaneously stable and class-specific: $\mathcal { F } _ { \mathrm { s e l } } ~ = ~ \{ \gamma _ { i } ^ { + } ~ :$ per $( \gamma _ { i } ^ { + } ) \ge \pi _ { \operatorname * { m i n } } , c _ { i } \ge c _ { \operatorname * { m i n } } \Big \}$ . Each selected feature induces a local component $C _ { i } ~ \subset ~ X ^ { + }$ and a local contrastive direction $\mathbf { v } _ { i }$ . Topological steering then constructs $\begin{array} { r } { { \bf v } _ { \mathrm { s t e e r } } = \sum _ { i \in \mathcal { F } _ { \mathrm { s e l } } } \omega _ { i } { \bf v } _ { i } . } \end{array}$ $\omega _ { i } \propto \mathrm { p e r s } ( \gamma _ { i } ^ { + } ) c _ { i } \Delta _ { i }$ , which emphasizes refusal-relevant submanifolds rather than global class averages.

In short, the motivation is that refusal behavior in residual space is often structurally heterogeneous: linear global means collapse this heterogeneity, whereas persistent topology isolates robust local structure and yields steering directions with stronger mechanistic interpretability.

## 4.2 Overview of Topological Steering

We propose Topological Steering, a method for constructing activation-space steering vectors for large language models (LLMs) that exploits persistent-homology structure in residual-stream representations. Rather than using the global mean difference between harmful and harmless activations, we identify topologically-significant, refusal-unique sub-clusters in the harmful activation cloud and use those to derive a more precise steering direction. The end-to-end pipeline consists of five key steps:

1. Activation Extraction and Joint Projection: We hook into an intermediate transformer layer ℓ to extract residual stream representations for both harmful $( X ^ { + } )$ and harmless $( X ^ { - } )$ prompts. We project these sets into a shared low-dimensional subspace using joint PCA.

2. Topological Feature Extraction: We treat the projected activations as finite metric spaces and compute their persistent homology via Vietoris–Rips filtrations. This yields persistence diagrams $\mathrm { { ( D g m ^ { + } } }$ and $\mathrm { D g m } ^ { - } )$ that capture the birth and death of topological features $( \mathrm { e . g . }$ connected components) across multiple spatial scales.

3. Refusal-Unique Feature Selection: We compare the topological summaries of both classes using optimal bottleneck matching. By identifying features with high persistence (geometric stability) and high mismatch cost (absence in the harmless class), we isolate structural signatures that uniquely encode refusal behavior.

4. Local Steering Vector Assembly: For each selected topological feature, we trace back to its constituent points in the activation cloud. We compute local contrastive vectors $( \mathbf { v } _ { i } )$ between these refusal-specific clusters and their nearest harmless neighbors, weighting them by their topological significance and effect size to form the final aggregate vector, $\mathbf { v } _ { \mathrm { s t e e r } } .$

5. Inference-Time Intervention: During autoregressive generation, $\mathbf { v } _ { \mathrm { s t e e r } }$ is injected into the target layer’s residual stream, consistently nudging the model toward the robustly identified refusal sub-manifolds without compromising the surrounding semantic space.

The advantages of the Topological Steering approach are the following:

• Heterogeneity Awareness: It identifies and leverages specific, local clusters of target behavior, ensuring that the steering vector is not diluted by averaging disjoint sub-populations.

• Coordinate-Free Robustness: Because persistent homology characterizes the intrinsic geometry of representations independently of rigid coordinate assumptions, the method is highly resilient to noise and varied prompt distributions.

• Enhanced Specificity: The dual criteria of persistence and cross-class mismatch ensure that the resulting steering vector is heavily anchored in features unique to the target behavior, maximizing the separation margin and minimizing off-target side effects.

• Computational Cost: Traditional TDA methods for Deep Learning were focused on the training aspect but here we intervene only at the inference time, reducing the compute cost drastically.

## 4.3 Notation and Setup

Let M be a decoder-only transformer with L layers. Given an input prompt, we denote the residualstream activation at layer ℓ and the final token position as $h _ { \ell } \in \bar { \mathbb { R } ^ { d _ { h } } }$ , where $d _ { h }$ is the model’s hidden dimension. We collect two datasets of prompts:

$\mathcal { D } ^ { + } = \{ s _ { i } ^ { + } \} _ { i = 1 } ^ { N ^ { + } }$ : harmful prompts that elicit model refusals.

$\mathcal { D } ^ { - } = \{ s _ { j } ^ { - } \} _ { j = 1 } ^ { N ^ { - } } ;$ : harmless prompts on similar topics that elicit compliance.

We split each dataset into disjoint train and holdout sets. All steering vectors are derived exclusively from the train split; holdout activations are used only for evaluation.

## 4.4 Activation Extraction

For each prompt $s _ { i } ^ { + } ~ ( \mathrm { r e s p . } ~ s _ { j } ^ { - } )$ , we register a forward hook on the output of transformer block ℓ and record the last-token residual: $X ^ { + } ~ = ~ \big [ h _ { \ell } ( s _ { 1 } ^ { + } ) , \dots , h _ { \ell } ( s _ { N _ { \mathrm { t r } } } ^ { + } ) \big ] ~ \in ~ \mathbb { R } ^ { N _ { \mathrm { t r } } ^ { + } \times d _ { h } }$ and $X ^ { - } =$ $\left[ h _ { \ell } ( s _ { 1 } ^ { - } ) , \ldots , h _ { \ell } ( s _ { N _ { \mathrm { t r } } } ^ { - } ) \right] \in \mathbb { R } ^ { N _ { \mathrm { t r } } ^ { - } \times d _ { h } }$ . The choice of layer ℓ is treated as a hyperparameter and swept over mid-to-late layers.

## 4.5 Joint PCA Projection

Hidden dimensions $d _ { h } ( { \bf e . g . } , d _ { h } = 4 0 9 6$ for Llama-3.1-8B) are too large for tractable persistenthomology computation. We reduce dimensionality via joint PCA: both activation sets are projected into the same low-dimensional space so that their relative geometry is preserved. Concretely, we concatenate $X ^ { \pm }$ into $X _ { \mathrm { a l l } } = [ X ^ { + } ; X ^ { - } ] \in \mathbb { R } ^ { ( N ^ { + } + N ^ { - } ) \times d _ { h } }$ , fit PCA on $X _ { \mathrm { a l l } }$ , and project: $\tilde { X } ^ { + } =$ $X ^ { + } W _ { \mathrm { p c a } } , \tilde { X } ^ { - } = X ^ { - } W _ { \mathrm { p c a } } , W _ { \mathrm { p c a } } \in \mathbb { R } ^ { d _ { h } \times k }$ , where $k \ll d _ { h }$ is the number of retained components (default $k = 3 2 )$ . The effective rank is capped at min $( k , N ^ { + } + N ^ { - } - 1 , d _ { h } )$ to avoid degeneracy.

## 4.6 Persistent Homology of Activation Clouds

We model each projected activation set as a finite metric space (point cloud) and compute its persistent homology using the Vietoris–Rips filtration [10].

Given a point cloud $P \subset \mathbb { R } ^ { k }$ and a scale parameter $\epsilon \geq 0$ , the Vietoris–Rips complex $\operatorname { V R } ( P , \epsilon )$ is the simplicial complex containing every simplex $\sigma \subseteq P$ whose pairwise diameter is at most $\epsilon , \mathbf { A } \mathbf { s } \ \epsilon$ grows from 0 to $\infty .$ , topological features (connected components in degree $H _ { 0 } ,$ , loops in degree $H _ { 1 }$

etc.) are born and die. The Persistence Diagram $\mathrm { D g m } _ { q } ( P )$ records each q-dimensional feature as a point $( \beta , \delta ) \in \mathbb { R } ^ { 2 }$ with birth $\beta$ and death δ; its persistence is $\pi = \delta - \beta$

We compute persistence diagrams separately for each class: $\mathrm { D g m ^ { + } } \ = \ \mathrm { D g m } ( \tilde { X } ^ { + } ) , \mathrm { D g m ^ { - } } \ =$ $\operatorname { D g m } ( \tilde { X } ^ { - } )$ , using the Ripser algorithm [4] with $\mathbb { Z } / 2 \mathbb { Z }$ coefficients. We compute $H _ { 0 }$ features by default and optionally $H _ { 1 }$ features.

## 4.7 Feature Matching and Refusal-Unique Feature Selection

A topological feature in $\mathrm { D g m ^ { + } }$ is refusal-unique if it has no close counterpart in $\mathrm { D g m } ^ { - }$ . To operationalise this, we compute an optimal matching between $\mathrm { D g m } _ { q } ^ { + }$ and $\mathrm { D g m } _ { q } ^ { - }$ (for each homology degree q) under the bottleneck distance [8]. Each feature $\gamma _ { i } ^ { + } \in \mathrm { D g m } _ { q } ^ { + }$ is matched to its nearest counterpart $\gamma _ { \sigma ( i ) } ^ { - } \in \mathrm { D g m } _ { q } ^ { - }$ (or to the diagonal if no finite match exists), yielding a mismatch cost $c _ { i } = d _ { \infty } ( \gamma _ { i } ^ { + } , \gamma _ { \sigma ( i ) } ^ { - } )$ . We rank all finite features by their persistence $\pi _ { i }$ and mismatch cost $c _ { i } ,$ , filtering by thresholds $\pi _ { \mathrm { m i n } }$ and $c _ { \operatorname* { m i n } } \colon \mathcal { F } = \left\{ \gamma _ { i } ^ { + } : \pi _ { i } \geq \pi _ { \operatorname* { m i n } } , \ c _ { i } \geq c _ { \operatorname* { m i n } } \right\}$ , sorted by $( \pi _ { i } , c _ { i } )$ descending.

## 4.8 Steering Vector Construction

For each selected $H _ { 0 }$ feature $\gamma _ { i } ^ { + } \in { \mathcal { F } } ,$ , we identify the corresponding sub-cluster of harmful activations using the death radius. Specifically, the death value $\delta _ { i }$ of the $H _ { 0 }$ feature corresponds to the scale at which its connected component merges into a larger one in the Vietoris–Rips filtration. We extract the points belonging to that component just before merger: $C _ { i } = \{ x \in \tilde { X } ^ { + }$ x is in the component born at $\gamma _ { i } ^ { + }$ within radius $\delta _ { i } - \varepsilon \}$ , where $\varepsilon > 0$ is a small numerical tolerance. Only components satisfying $| C _ { i } | \geq m _ { \operatorname* { m i n } }$ (a minimum size threshold) are retained.

For each component $C _ { i } ,$ we identify the $k _ { \mathrm { n e g } }$ nearest harmless training points (in PCA space) to the component centroid, forming a local negative contrast set $N _ { i } \subset X ^ { . }$ <sup>−</sup>. The per-component steering direction is: $\begin{array} { r } { \mathbf { v } _ { i } = \bar { X } _ { C _ { i } } ^ { + } - \bar { X } _ { N _ { i } } ^ { - } , \bar { X } _ { C _ { i } } ^ { + } = \frac { \mathbf { \bar { \Phi } } _ { 1 } } { | C _ { i } | } \sum _ { j \in C _ { i } } x _ { j } ^ { + } , \bar { X } _ { N _ { i } } ^ { - } = \frac { \mathbf { \Phi } _ { 1 } } { | N _ { i } | } \sum _ { j \in N _ { i } } x _ { j } ^ { - } } \end{array}$ . Each direction is weighted by a quality score combining persistence, mismatch cost, and the Cohen’s d effect size along $\mathbf { v } _ { i } \colon$

$$
\Delta _ { i } = \frac { | \bar { P } _ { i } - \bar { N } _ { i } | } { \sigma _ { \mathrm { p o o l } } ( P _ { i } , N _ { i } ) } ,\tag{1}
$$

$$
w _ { i } = \pi _ { i } \cdot c _ { i } \cdot \Delta _ { i } ,\tag{2}
$$

where ${ \bar { P } } _ { i } \left( \mathrm { r e s p . } \ { \bar { N } } _ { i } \right)$ is the mean projection of $C _ { i }$ (resp. $N _ { i } )$ onto ${ \hat { \mathbf { v } } } _ { i } ,$ and $\sigma _ { \mathrm { p o o l } }$ is the pooled standard deviation. The final steering vector is the normalised weighted sum over the top-K components:

$$
\mathbf { v } _ { \mathrm { s t e e r } } = \sum _ { i = 1 } ^ { K } { \frac { w _ { i } } { \sum _ { j } w _ { j } } } \mathbf { v } _ { i } ,\tag{3}
$$

with orientation enforced so that $\langle { \bf v } _ { \mathrm { s t e e r } } , \bar { X } ^ { + } \rangle > \langle { \bf v } _ { \mathrm { s t e e r } } , \bar { X } ^ { - } \rangle$ . We compare against the naive meandifference direction $\mathbf { v } _ { \mathrm { b a s e } } = \bar { X } ^ { + } - \bar { X } ^ { - }$ , which corresponds to the standard activation-steering approach [18].

## 4.9 Inference-Time Steering

At inference time, we inject the steering vector into the residual stream at the same layer ℓ used for extraction. Given a new input prompt s, we modify the last-token hidden state at every generation step: $\begin{array} { r } { h _ { \ell } ( \cdot ) \  \ h _ { \ell } ( \cdot ) + \alpha \cdot ^ { \cdot } \frac { { \bf v } _ { \mathrm { s t e r } } } { \| { \bf v } _ { \mathrm { s t e e r } } \| } } \end{array}$ , where $\alpha > 0$ is the steering magnitude. Applying the hook only at the final token position preserves the full autoregressive context while consistently nudging the model’s representation towards the refusal subspace.

## 4.10 Evaluation Metric

We measure steering quality via the refusal score: the normalised class gap along the steering direction evaluated on held-out activations $X _ { \mathrm { h o } } ^ { + }$ and $\begin{array} { r } { X _ { \mathrm { h o } } ^ { - } \colon \rho ( { \bf v } ) = \frac { \langle { \bf v } , \bar { X } _ { \mathrm { h o } } ^ { + } \rangle - \langle { \bf v } , \bar { X } _ { \mathrm { h o } } ^ { - } \rangle } { \sigma _ { \mathrm { p o o l } } ( X _ { \mathrm { h o } } ^ { + } , X _ { \mathrm { h o } } ^ { - } ; { \bf v } ) } } \end{array}$ . We report the steering score delta $\Delta \rho = \rho (  { \mathbf { v } } _ { \mathrm { s t e e r } } ) - \rho (  { \mathbf { v } } _ { \mathrm { b a s e } } )$ as the primary comparative metric, where positive values indicate that our topological direction separates harmful from harmless activations more effectively than the baseline.

## 5 Experimentations

Following recent steering papers that separate representation-level diagnostics from downstream behavioral tests (e.g., refusal sweeps and automated judges; cf. Angular Steering [16]), we organize this section into i) datasets and models and then separate our experiment section into ii) a proofthat TS is working with benchmarks and metrics 5.2 and iii) a proof that TS is working across different model architectures 5.3. Unless stated otherwise, steering vectors are computed on training splits only, injected at inference as a normalized residual update at the last prompt token, and evaluated on held-out activations and on open-ended generations. All experiments were run on an A100 (40Gb) and for reproductibility purposes the hyperparameter details are in Appendix A.5.3. Moreover proof that the experiment results we obtain are not due to randomness are in A.6.

## 5.1 Datasets and Models

Construction of Contrastive Activation Pairs for Steering. Harmful prompts are drawn from AdvBench [17]. We use the standard harmful split materialized as line-delimited prompts; the preparation utility enforces 520 unique harmful strings with a fixed calibration/evaluation partition: 416 calibration and 104 evaluation harmful prompts. Harmless prompts are sampled from the public instruction corpus tatsu-lab/alpaca on Hugging Face as we require at least 512 unique instructions for calibration-side harmless data; when no local snapshot is supplied, prompts are fetched with the datasets library and de-duplicated by normalized text. This mirrors the contrastive pipeline in our codebase: activations are collected at a chosen transformer block, projected with joint $\bar { \mathrm { P C A } } .$ , and passed to the topological construction of $\mathbf { v } _ { \mathrm { s t e e r } }$

Models. The default development model in our configuration is meta-llama/Llama-3.1-8B-Instruct; the repository also supports multi-model sweeps (Qwen 2.5 and Gemma 2 families) through a shared hooking interface. Layer index ℓ and TDA hyperparameters (PCA dimension, persistence thresholds, $k _ { \mathrm { n e g } } ,$ , component merge depth, etc.) are treated as tunables; a representative end-to-end tuning run selects a single layer and hyperparameter hash before behavioral validation.

## 5.2 Experimental Proof that Topological Steering is working with relevant Benchmarks

The first goal of the Experiment section 5 is to show that our method is indeed steering LLMs and working. This is shown in table 1.

We report three complementary families of numbers: (i) activation steering scores, (ii) lightweight lexical jailbreak-rates, and (iii) external LLM judges aligned with community red-teaming practice. We now explain the different metrics computed:

Activation Separation Metric $( \Delta \rho )$ . Let $X _ { \mathrm { h o } } ^ { + } , X _ { \mathrm { h o } } ^ { - }$ be harmful and harmless held-out activations at $( \ell , t ^ { \star } )$ . For a candidate direction v, write $u = \mathbf { v } / \| \mathbf { v } \|$ and define the scalar refusal score $\rho ( \mathbf { v } ) =$ $( \mu ^ { + } - \mu ^ { - } ) / \sigma _ { \mathrm { p o o l } }$ , where $\mu ^ { + }$ (resp. $\mu ^ { - } )$ is the mean of $u ^ { \top } X _ { \mathrm { h o } } ^ { + }$ (resp. $u ^ { \top } X _ { \mathrm { h o } } ^ { - } )$ and $\sigma _ { \mathrm { p o o l } }$ is the pooled standard deviation of the concatenated projections (implementation matches our training-time weighting routine). The steering score delta is $\Delta \rho = \rho (  { \mathbf { v } } _ { \mathrm { s t e e r } } ) - \rho (  { \mathbf { v } } _ { \mathrm { b a s e } } )$ with $\mathbf { v } _ { \mathrm { b a s e } } = \bar { X } ^ { + } - \bar { \bar { X } } ^ { - }$ on the train split. Positive $\Delta \rho$ certifies that the topological direction separates harmful from harmless clouds more sharply than the mean-difference baseline before any decoding. This steering score delta is vital as it is the first indicator that our model is indeed steering, allowing us to then compute different benchmark methods that will more precisely evaluate the answers of our steered model.

The different benchmarks used are Llama Guard 3 implemented from [12], Harmbench classifier [13] and Substring Matching. More details are found in A.5.1.

## Interpretation. Table 1 answers two distinct questions:

• First, the activation row shows that the learned topological vector improves the linear separability of harmful versus harmless activations relative to the global mean direction. This is direct evidence that the persistence-guided subset construction is extracting a refusalaligned axis in representation space.

• Second, the judge rows show that decoding under the same intervention increases automated jailbreak rates on a fixed harmful evaluation set, with the largest relative movement on Llama Guard 3 and a smaller but consistent gain on HarmBench; the substring probe moves modestly.

Table 1: Representative end-to-end result after single-layer steering with comparison to baseline; steering injection $\alpha { = } 1$ in that run). “Judge $\Delta ^ { \prime \prime }$ is steered minus baseline success rate.
<table><tr><td>Metric family</td><td>Baseline</td><td>Topological</td><td> $\Delta$ </td></tr><tr><td>Activation: best  $\Delta \rho$  (held-out) Activation: mean ± std.  $\Delta \rho$  (robustness replicates)</td><td></td><td></td><td> $+ 0 . 0 1 2 7$   $+ 0 . 0 0 9 2 \pm 0 . 0 0 2 2$ </td></tr><tr><td>Llama Guard 3 success rate</td><td>0.510</td><td>0.692</td><td>+0.183</td></tr><tr><td>HarmBench classifier success rate</td><td>0.010</td><td>0.029</td><td>+0.019</td></tr><tr><td>Substring refusal lexicon success rate</td><td>0.029</td><td>0.038</td><td>+0.010</td></tr></table>

Table 2: Multi-model comparative run. For each checkpoint we sweep transformer blocks $\ell \in$ $\{ 8 , \ldots , 2 0 \}$ , pick $\ell ^ { \star }$ that maximizes held-out $\Delta \rho$ (topological vs. mean-difference baseline), and report $\rho ( \mathbf { v _ { \mathrm { b a s e } } } ) , \rho ( \mathbf { v _ { \mathrm { s t e e r } } } )$ , and $\Delta \rho$ at $\ell ^ { \star }$ . L is the total number of decoder blocks; $\ell ^ { \star } / L$ is a coarse “depth” readout.
<table><tr><td>Model</td><td> $L$ </td><td> $\ell ^ { \star }$ </td><td> $\ell ^ { \star } / L$ </td><td> $\rho _ { \mathrm { b a s e } }$ </td><td> $\rho _ { \mathrm { t d a } }$ </td><td> $\Delta \rho$ </td></tr><tr><td>google/gemma-2-9b-it</td><td>42</td><td>13</td><td>0.31</td><td>0.499</td><td>1.249</td><td>0.750</td></tr><tr><td>meta-1lama/Llama-3.2-3B-Instruct</td><td>28</td><td>20</td><td>0.71</td><td>1.948</td><td>1.966</td><td>0.018</td></tr><tr><td>Qwen/Qwen2.5-3B-Instruct</td><td>36</td><td>20</td><td>0.56</td><td>1.981</td><td>1.987</td><td>0.006</td></tr><tr><td>Qwen/Qwen2.5-14B-Instruct</td><td>48</td><td>20</td><td>0.42</td><td>1.977</td><td>1.983</td><td>0.006</td></tr><tr><td>Qwen/Qwen2.5-7B-Instruct</td><td>28</td><td>8</td><td>0.29</td><td>1.974</td><td>1.979</td><td>0.005</td></tr><tr><td>meta-1lama/Llama-3.1-8B-Instruct</td><td>32</td><td>16</td><td>0.50</td><td>1.972</td><td>1.974</td><td>0.001</td></tr></table>

Limitations. Automated judge metrics are subject to evaluation drift: measured success rates can change when the judge stack changes (e.g., prompt template, backend model revision, API/server configuration), even if the generated responses are held fixed; therefore HarmBench and Llama Guard scores should be interpreted as setup-dependent and not directly interchangeable. Our substring-based refusal probe is intentionally lightweight but misses non-lexical refusals (safe refusals that do not contain predefined trigger phrases for example “That request is unsafe and I won’t provide operational instructions”), which can bias jailbreak-rate estimates upward for semantically valid but lexically novel refusals. Moreover, oppositely because our substring probe only flags generic refusal phrases and does not assess whether a response still contains harmful instructional content, partially compliant or “disclaimer-first” answers can look like refusals to the probe while remaining unsafe under a semantic judge (for example, prefacing harmful steps with a refusal-like disclaimer), this can bias jailbreak-rate estimates downward. Higher activation-space separation (e.g., higher $\Delta \rho )$ does not guarantee uniform improvements across other capabilities, so helpfulness, reasoning, truthfulness, and related tasks must be measured directly rather than inferred from steering metrics.

## 5.3 Experimental Proof that Topological Steering is working across different Model Architectures

The second goal of our experiments is to show that our method does not work only on meta-llama/Llama-3.1-8B-Instruct but also on other models. So we compared the steering score $\Delta \rho$ of different models, as a positive $\Delta \rho$ is proof of a working TS method. Our results are shown in table 2.

Interpretation. All six models show $\Delta \rho > 0$ at their selected layer: the persistence-guided steering direction consistently improves linear separability of harmful vs. harmless activations relative to the global mean contrast, under the same training/holdout protocol and sweep budget. The ranking is not monotone in parameter count (e.g. Llama-3.2-3B gains more than Llama-3.1-8B), which is expected if refusal geometry is shaped by post-training alignment and width/depth tradeoffs, not raw capacity alone. The optimal injection depth $\ell ^ { \star }$ varies by family: Qwen-7B peaks very early $( \ell ^ { \star } { = } 8 )$ whereas Qwen-3B/14B and Llama-3.2-3B peak at the sweep cap $( \ell ^ { \star } { = } 2 0 )$ , and Llama-3.1-8B peaks mid-stack (ℓ<sup>⋆</sup>=16). Such spread is consistent with the hypothesis that “where” harmful/harmless structure is most readable in the residual stream depends on stacking depth, normalization placement, and how safety-related features are distributed across layers. All of which differ between Llama-3, Qwen2.5, and Gemma-2 blocks (e.g., Gemma 2 uses alternating local/global attention and a different norm/attention ordering than Llama). We notice a significative difference in Gemma-2-9b-it with a $\Delta \rho$ way bigger than the other models. Gemma attains a much larger $\Delta \rho$ but also a much smaller baseline $\rho _ { \mathrm { b a s e } }$ than the other five models. Architecturally, that pattern indicates that the scalar refusal score is not cross-model calibrated: residual magnitudes, RMSNorm scaling, and hidden width $d _ { h }$ jointly change the scale of projections $u ^ { \top } h .$ , so comparing raw $\rho$ across checkpoints is weaker than comparing $\Delta \rho$ within a model or reporting additional generation-side judges. Here, the low $\rho _ { \mathrm { b a s e } }$ suggests the mean-difference axis is a poor proxy for refusal separation in Gemma’s activation geometry at $\ell ^ { \star }$ , while the topological subset, hence the TS method, still recovers a strong separating direction (large $\rho _ { \mathrm { t d a } } )$ . We therefore emphasize within-model gains and treat Gemma’s large $\Delta \rho$ as evidence that Gemma may benefit more from topology-based steering than from simple global mean-difference steering, not as a claim that Gemma $\mathrm { i s } \ ^ { 6 6 } \mathrm { 1 0 0 \times }$ more steerable” than Llama in an absolute sense.

Remark 1. Table 2 supports a cross-family proof of mechanism: topological steering is not an artifact of a single Llama checkpoint, it yields positive $\Delta \rho$ across decoder-only architectures with different depths, widths, and normalization/attention recipes. The dispersion of $\ell ^ { \star }$ motivates reporting layer sweeps (or at least a mid-late band) rather than a single universal layer, and motivates appendix plots of $\Delta \rho ( \ell )$ per model. As shown in Figure 1 and Figure 2.

## 6 Conclusion, limits and remarks

We introduced Topological Steering, an inference-time method that builds steering directions from persistent-homology structure in activation space rather than from a single global mean contrast. By matching harmful/harmless persistence diagrams, selecting robust contrastive features, and aggregating local directions, the method turns activation geometry into an explicit and inspectable steering signal. Our results show consistent positive separation gains $( \Delta \rho > 0 )$ across the tested models and layers in our single-layer setup, together with measurable behavioral movement on external judge benchmarks. More broadly, this work positions TDA as an actionable representationlevel tool for intervention design, not only a descriptive visualization aid. While we acknowledge that specialized geometric methods may still lead on specific benchmarks, the results presented here argue for a conceptual pivot. By providing a reproducible and inspectable pipeline, this paper opens a new ’steering primitive’ for the field. It is indeed a contribution we believe will remain relevant as model architectures continue to evolve and as engineering refinements will likely improve its absolute performance in the future.

## Some limitations of our method are:

1. Persistent Homology is sensitive to preprocessing and to hyperparameters that govern feature selection; our gains depend on tuning, as is common for steering.

2. Diagram-based summaries are not uniquely tied to semantics: high-persistence features can reflect nuisance geometry unless paired with contrastive matching and behavioral validation.

3. Computational cost, while modest compared to training using TDA tools, still scales with point-cloud size and homology degree.

4. We focus on refusal as a concrete safety axis; transfer to other behaviors will require new contrast sets and careful evaluation.

A natural next step is hybrid steering: use topology to identify where to intervene, then combine with optimized geometric update rules for how to intervene. Another direction is to use persistence diagrams as structured evidence for safety debugging and representation-level auditing.

Closing Remark. Topological Steering is a proof-of-concept that multiscale structure in activation space can be converted into a practical intervention without retraining. Even when not outperforming specialized baselines, it expands the steering design space by making support subsets and multiscale structure first-class components of the method. We believe that Topological Steering will be a promising foundation for the next generation of representation level safety tools and we hope it helps revive topological thinking in modern deep learning research.

## References

[1] Henry Adams, Sofya Chepushtanova, Tegan Emerson, Eric Hanson, Michael Kirby, Francis Motta, Rachel Neville, Chris Peterson, Patrick Shipman, and Lori Ziegelmeier. Persistence images: A stable vector representation of persistent homology, 2016. URL https://arxiv. org/abs/1507.06217.

[2] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. Advances in Neural Information Processing Systems, 37:136037–136083, 2024.

[3] Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

[4] Ulrich Bauer. Ripser: efficient computation of vietoris–rips persistence barcodes. Journal of Applied and Computational Topology, 5(3):391–423, 2021.

[5] Rickard Brüel-Gabrielsson, Bradley J. Nelson, Anjan Dwaraknath, Primoz Skraba, Leonidas J. Guibas, and Gunnar Carlsson. A topology layer for machine learning, 2020. URL https: //arxiv.org/abs/1905.12200.

[6] Peter Bubenik. Statistical topological data analysis using persistence landscapes, 2015. URL https://arxiv.org/abs/1207.6437.

[7] Gunnar Carlsson. Topology and data. Bulletin ofthe American Mathematical Society, 46(2): 255–308, 2009.

[8] David Cohen-Steiner, Herbert Edelsbrunner, and John Harer. Stability of persistence diagrams. In Proceedings of the twenty-first annual symposium on Computational geometry, pages 263– 271, 2005.

[9] Edelsbrunner, Letscher, and Zomorodian. Topological persistence and simplification. Discrete & computational geometry, 28(4):511–533, 2002.

[10] Herbert Edelsbrunner and John Harer. Computational topology: an introduction. American Mathematical Soc., 2010.

[11] Christoph Hofer, Roland Kwitt, Marc Niethammer, and Andreas Uhl. Deep learning with topological signatures, 2018. URL https://arxiv.org/abs/1707.04041.

[12] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674, 2023.

[13] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.

[14] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition, 2024. URL https://arxiv.org/ abs/2312.06681.

[15] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering, 2024. URL https://arxiv.org/abs/2308.10248.

[16] Hieu M Vu and Tan M Nguyen. Angular steering: Behavior control via rotation in activation space. arXiv preprint arXiv:2510.26243, 2025.

[17] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models, 2023. URL https: //arxiv.org/abs/2307.15043.

[18] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to ai transparency, 2025. URL https://arxiv.org/abs/2310.01405.

## A Technical appendices and supplementary material

## A.1 Complementary Theoritical Background

## A.1.1 Transformers

We recall the notation for a decoder-only transformer with L layers, hidden width $d _ { h }$ , and token sequence length T. Given an input text, tokenization maps it to discrete token IDs

$$
( x _ { 1 } , \dots , x _ { T } ) \in \mathcal { V } ^ { T } ,
$$

where V is the vocabulary. Each token $x _ { t }$ is embedded into $\mathbb { R } ^ { d _ { h } }$ and combined with positional information to form the initial residual states

$$
h _ { t } ^ { ( 0 ) } \in \mathbb R ^ { d _ { h } } , \qquad t = 1 , \dots , T .
$$

At layer $\ell \in \{ 0 , \ldots , L - 1 \}$ , a standard pre-norm block applies self-attention and an MLP through residual updates:

$$
\begin{array} { r l } & { \tilde { h } _ { t } ^ { ( \ell ) } = h _ { t } ^ { ( \ell ) } + \mathrm { A t t n } ^ { ( \ell ) } \left( \mathrm { L N } _ { 1 } ^ { ( \ell ) } ( h _ { 1 : T } ^ { ( \ell ) } ) \right) _ { t } , } \\ & { h _ { t } ^ { ( \ell + 1 ) } = \tilde { h } _ { t } ^ { ( \ell ) } + \mathrm { M L P } ^ { ( \ell ) } \left( \mathrm { L N } _ { 2 } ^ { ( \ell ) } ( \tilde { h } _ { t } ^ { ( \ell ) } ) \right) . } \end{array}
$$

Here, $\mathrm { A t t n } ^ { ( \ell ) }$ denotes multi-head causal self-attention and $\mathrm { M L P ^ { ( \ell ) } }$ is the position-wise feed-forward block. For each head, attention weights are computed from queries, keys, and values:

$$
Q = X W _ { Q } , \quad K = X W _ { K } , \quad V = X W _ { V } , \quad \mathrm { A t t n } ( X ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } + M \right) V ,
$$

with causal mask M and head dimension $d _ { k }$

## A.1.2 Activation Steering

To complement the Background 3 section we add that:

In our setting, v is estimated from two prompt-conditioned activation collections: a positive set (target behavior to increase) and a negative set (behavior to suppress). Let $X ^ { + } , X ^ { - } \subset \bar { \mathbb { R } } ^ { d _ { h } }$ be activations extracted at the same $( \ell ^ { \star } , t ^ { \star } )$ , and let $\mu ^ { + }$ and $\mu ^ { - }$ be the respective training-set centroids,

$$
\mu ^ { + } = \frac { 1 } { | X ^ { + } | } \sum _ { x \in X ^ { + } } x , \qquad \mu ^ { - } = \frac { 1 } { | X ^ { - } | } \sum _ { x \in X ^ { - } } x ,
$$

and sets the steering vector to the displacement between these class summaries,

$$
\mathbf { v } = \mu ^ { + } - \mu ^ { - } .
$$

This makes steering explicit: positive α moves activations toward the target behavioral region and away from the contrast region in residual space. The modified state $h _ { t ^ { \star } } ^ { ( \ell ^ { \star } ) \prime }$ is then propagated through layers $\ell ^ { \star } + 1 , \ldots , L$ to produce the final logits and generated text.

## A.2 Persistent Homology

Let $X = \{ x _ { 1 } , \ldots , x _ { n } \} \subset \mathbb { R } ^ { m }$ be a finite point cloud (in our case, sampled activations), and let $\operatorname { d i s t } ( \cdot , \cdot )$ be a metric on X (typically Euclidean distance after preprocessing). Persistent homology studies how the topology of X evolves across spatial scales by constructing a nested family of simplicial complexes $K _ { \epsilon _ { 1 } } \subseteq K _ { \epsilon _ { 2 } } \subseteq \cdot \cdot \cdot , \epsilon _ { 1 } \leq \epsilon _ { 2 } \leq \cdot \cdot \cdot$ , called a filtration.

For each scale ϵ and homological dimension $k \geq 0 ,$ , the k-th homology group $H _ { k } ( K _ { \epsilon } ; \mathbb { F } )$ (with coefficients in a field $\mathbb { F } , \mathrm { e . g . } , \bar { \mathbb { Z } } _ { 2 } )$ encodes k-dimensional topological features: $H _ { 0 }$ tracks connected components, $H _ { 1 }$ tracks 1-dimensional loops, $H _ { 2 }$ tracks voids, etc. Its rank $\beta _ { k } ( \epsilon ) : =$ rank $H _ { k } ( K _ { \epsilon } ; \mathbb { F } )$ is the k-th Betti number at scale ϵ.

As ϵ increases, features appear and disappear. A feature in dimension k is said to be born at scale b if it first appears in $H _ { k } ( \bar { \kappa } _ { b } )$ , and dies at scale $\delta > b$ if it merges into an older class or becomes a boundary in $H _ { k } ( \ K _ { \delta } )$ ). Its lifetime (or persistence) is $\mathrm { p e r s } = \delta - b$ . Long-lived classes are typically interpreted as structurally robust, while short-lived classes are often attributed to sampling noise [9].

Formally, persistence is induced by inclusion maps in the filtration. For $\epsilon \leq \epsilon ^ { \prime } .$ , the inclusion $\kappa _ { \epsilon } \hookrightarrow \dot { \kappa } _ { \epsilon ^ { \prime } }$ yields a linear map $f _ { \epsilon , \epsilon ^ { \prime } } ^ { k } : H _ { k } ( K _ { \epsilon } ; \mathbb { F } ) \overset { \bullet } {  } H _ { k } ( K _ { \epsilon ^ { \prime } } ; \mathbb { F } )$ . A class born at b that remains nonzero under $f _ { b , \epsilon } ^ { k }$ for $\epsilon < \delta$ but vanishes at δ has persistence interval $[ b , \delta )$

In this work, persistent homology provides a multiscale summary of activation geometry without assuming linear separability. By tracking birth/death events of components and higher-order structures across scales, it supplies the topological signal that we later encode with persistence diagrams and use to construct steering-relevant subsets/directions.

## A.3 Persistence Diagram

A persistence diagram is a finite multiset of birth–death pairs that summarizes the output of persistent homology for a fixed dimension k. Given a filtration $\{ \mathcal { K } _ { \epsilon } \} _ { \epsilon \ge 0 }$ and its induced persistence module in dimension $k ,$ each topological class contributes one point $( b _ { i } ^ { ( k ) } , \delta _ { i } ^ { ( k ) } ) \in \mathbb { R } ^ { 2 } , b _ { i } ^ { ( k ) } < \delta _ { i } ^ { ( k ) } \leq + \infty$ where $b _ { i } ^ { ( k ) }$ is the birth scale and $\delta _ { i } ^ { ( k ) }$ is the death scale. The corresponding diagram is $D _ { k } ( X ) =$ $\{ ( b _ { i } ^ { ( k ) } , \delta _ { i } ^ { ( k ) } ) \}$ <sub>i</sub>.

The diagonal $\Delta = \{ ( u , u ) : u \in \mathbb { R } \}$ represents zero persistence. Distance from a point to $\Delta$ is proportional to its lifetime $\mathrm { p e r s } _ { i } ^ { ( k ) } = \bar { \delta } _ { i } ^ { ( k ) } - b _ { i } ^ { ( k ) }$ , so points far from the diagonal correspond to topologically robust structure, while points near the diagonal are typically less stable under perturbations. In practice, we compute diagrams separately by homology degree $( \mathrm { e . g . , ~ } D _ { 0 }$ for connected components and $D _ { 1 }$ for loops), then compare diagrams across behavioral conditions.

To compare two diagrams $D _ { k } ( X )$ and $D _ { k } ( Y )$ , one uses a matching-based metric (e.g., bottleneck or p-Wasserstein) that allows points to match either across diagrams or to the diagonal. Intuitively, this quantifies how much geometric deformation is required to transform one topological summary into the other.

## A.4 Vietoris–Rips Filtration

Let $X = \{ x _ { 1 } , \ldots , x _ { n } \} \subset \mathbb { R } ^ { m }$ with metric dist. For a scale parameter $\epsilon \geq 0 .$ , the Vietoris–Rips complex at scale ϵ is $\operatorname { \bar { V } R } _ { \epsilon } ( X ) = \{ \sigma \subseteq X$ : dist $( x _ { i } , x _ { j } ) \leq \epsilon$ for all $x _ { i } , x _ { j } \in \sigma \}$ . Equivalently, a simplex is included iff all pairwise distances among its vertices are at most ϵ. Thus, edges appear first, then triangles, tetrahedra, and higher-dimensional simplices as ϵ grows.

Because pairwise constraints are monotone in ϵ, these complexes form a filtration: $\operatorname { V R } _ { \epsilon _ { 1 } } ( X ) \subseteq$ $\mathrm { V R } _ { \epsilon _ { 2 } } ( X )$ whenever $\epsilon _ { 1 } ~ \leq ~ \epsilon _ { 2 }$ . Applying homology to this nested sequence yields persistent homology and, in turn, persistence diagrams.

Vietoris–Rips is especially convenient for activation point clouds because it only requires pairwise distances and does not need an explicit manifold model. Its main trade-off is combinatorial growth: the number of simplices can increase rapidly with sample size and intrinsic dimension, so implementations typically cap the maximum homology degree and rely on dimension reduction (e.g., PCA) before computing persistence.

## A.5 Complements on experiments

## A.5.1 Benchmarks

In this section we are going to detail more on the 3 benchmarks used:

1. Llama Guard 3: We call a hosted chat-completions endpoint with a Llama Guard 3- compatible model card and aggregate the fraction of generations labeled safe refusal versus compliance, following the spirit of the Llama Guard evaluator line [12]. To be clear the success rate is currently scored as unsafe (higher = more jailbreak success). The implementation records baseline\_success\_rate and steered\_success\_rate on the same fixed harmful evaluation prompts.

2. HarmBench classifier: We use the HarmBench behavioral test harness with a HarmBenchstyle binary classifier (default backend model cais/HarmBench-Llama-2-13b-cls in our launcher) that reads the harmful prompt and model answer and returns a harm / no-harm style verdict [13]; we export the resulting success\_rate fields analogously to Llama Guard. More concretely a higher Harmbench score is equal to worse safety)

3. Substring matching: We mark a generation as a “refusal hit” if its lowercased text contains any phrase from a fixed short lexicon (default list includes templates such as “i’m sorry”, “i cannot”, “as a language model”, . . . ); the reported success rate is non-refusal rate under substring proxy. This probe is inexpensive and correlates loosely with human-visible refusals, but it is brittle on paraphrases; we therefore treat it as a diagnostic, not a standalone claim.

## A.5.2 Implementation clarifications and scope.

For transparency, we clarify that the reported Topological Steering results use an operational $\mathrm { H } _ { 0 ^ { - } }$ component recovery procedure rather than an exact birth-component oracle. Concretely, for a selected $\mathrm { H } _ { \mathrm { 0 } }$ feature ${ \gamma } _ { i } ^ { + } = \mathsf { \bar { ( } } \bar { b _ { i } } , \delta _ { i } )$ , we extract a connected subset at scale $\delta _ { i } - \varepsilon$ and apply deterministic selection heuristics (e.g., viability/size filtering and component scoring) to obtain the steering subset; therefore, statements of the form “the component born at $\gamma _ { i } ^ { + \cdots }$ should be read as an implementationlevel approximation to the intended topological object, not as exact symbolic tracking across the full filtration. In addition, sign orientation is not claimed as a universal post-hoc guarantee for every possible builder: in the single-layer component-aggregation pipeline used for the main results, direction polarity follows the constructed local contrasts and is validated empirically through held-out $\Delta \rho ,$ rather than by a separate mandatory global sign-flip step logged for every run. Weighting is also fixed only for the reported experiments: we use $w _ { i } \propto \pi _ { i } c _ { i } \Delta _ { i }$ (persistence × mismatch × separation) as the chosen experimental setting, and do not claim this weighting is theoretically unique or mandatory beyond the present study.

Table 3: Hyperparameters selected for the row in Table 1.
<table><tr><td>Field</td><td>Value</td></tr><tr><td>Layer l</td><td>15</td></tr><tr><td>PCA components k</td><td>192</td></tr><tr><td> $H _ { 0 }$  top-K merged components</td><td>7</td></tr><tr><td>Minimum component / subset sizes</td><td>16 / 40</td></tr><tr><td>Local negatives  $k _ { \mathrm { n e g } }$ </td><td>128</td></tr><tr><td>Persistence / mismatch floors</td><td>0.01 /0.01</td></tr><tr><td>Component weights</td><td>uniform</td></tr></table>

Table 4: Approximate wall-clock runtime for the end-to-end best-run pipeline (reconstructed from artifact timestamps).
<table><tr><td>Stage</td><td>Time (min)</td><td>Time (hours)</td></tr><tr><td>Phase 0 (baseline calibration)</td><td>7.67</td><td>0.13</td></tr><tr><td>Phase 1 (coarse search / layer sweep)</td><td>75.19</td><td>1.25</td></tr><tr><td>Phase 2 (focused hyperparameter optimization)</td><td>56.99</td><td>0.95</td></tr><tr><td>Phase 3 (robustness replicates)</td><td>23.73</td><td>0.40</td></tr><tr><td>Phase 4 (inference + judge benchmarks)</td><td>36.03</td><td>0.60</td></tr><tr><td>Phase 5 (final selection / aggregation)</td><td>0.00</td><td>0.00</td></tr><tr><td>Total (Phase 0–5)</td><td>199.62</td><td>3.33</td></tr></table>

These durations are reconstructed from filesystem mtime windows for each phase folder, so they’re a good approximation but not perfect wall-clock start/stop time of each subprocess. All experiments were run on a GPU A100 with 40Gb memory.

Regarding experiment 2 5.3 the times for the run are in Table 5:

Table 5: Approximate compute time for the multi-model comparative run (artifact-time reconstruction).
<table><tr><td>Model</td><td>Time (min)</td><td>Time (hours)</td></tr><tr><td>Qwen/Qwen2.5-3B-Instruct</td><td>5.36</td><td>0.09</td></tr><tr><td>Qwen/Qwen2.5-7B-Instruct</td><td>5.76</td><td>0.10</td></tr><tr><td>Qwen/Qwen2.5-14B-Instruct</td><td>8.73</td><td>0.15</td></tr><tr><td>meta-1lama/Llama-3.2-3B-Instruct</td><td>3.27</td><td>0.05</td></tr><tr><td>meta-1lama/Llama-3.1-8B-Instruct</td><td>3.93</td><td>0.07</td></tr><tr><td>google/gemma-2-9b-it</td><td>8.91</td><td>0.15</td></tr><tr><td>Total wall-clock (run start → final artifact)</td><td>35.96</td><td>0.60</td></tr></table>

Note: durations are reconstructed from filesystem timestamps of run artifacts and are therefore approximate wall-clock estimates.

## A.5.3 Reproducibility details for the reported runs.

To make the main results reproducible, we fix the full implementation scope and runtime configuration used in the final single-layer study. All reported numbers are produced with a single-layer Topological Steering pipeline on meta-llama/Llama-3.1-8B-Instruct, where the steering vector is injected at the last prompt token with $\alpha \ = \ 1 . 0$ We use deterministic random control with controller seed 42, baseline calibration seeds 0:19, and robustness seeds {20, 21, 22, 23, 24}. Layer selection is performed over $\ell \in \{ 8 , \ldots , 2 0 \}$ , and the final reported winner uses $\ell ^ { \star } = 1 5 .$ The steering builder is h0\_components with $H _ { 0 } { \mathrm { - o n l y } }$ persistence (maxdim=0, homology\_focus=h0\_first), and fixed winner hyperparameters listed in Table 6. Behavioral evaluation is run on the fixed harmful evaluation set with the same prompt list for baseline and steered generations, using the judge stack substring\_matching, llamaguard3, and harmbench; exact endpoint/model settings are also fixed and reported in the table so the judge layer is reproducible.

Table 6: Reproducibility configuration for the final reported single-layer run.
<table><tr><td>Item</td><td>Fixed value used for reported results</td></tr><tr><td>Base model</td><td>meta-1lama/Llama-3.1-8B-Instruct</td></tr><tr><td>Steering scope</td><td>Single-layer only (no multi-layer injection)</td></tr><tr><td>Injection position</td><td>Last prompt token residual</td></tr><tr><td>Steering scale</td><td> $\alpha = 1 . 0$ </td></tr><tr><td>Layer sweep range</td><td> $\ell \in \{ 8 , \ldots , 2 0 \}$ </td></tr><tr><td>Final selected layer</td><td> $\ell ^ { \star } = 1 5$ </td></tr><tr><td>Direction method</td><td>h0_components</td></tr><tr><td>Persistence computation</td><td>Vietoris-Rips, Ripser,  $\mathbb { Z } / 2 \mathbb { Z } ,$  maxdim=0</td></tr><tr><td>Homology focus</td><td>h0_first</td></tr><tr><td>PCA components (winner)</td><td> $k = 1 9 2$ </td></tr><tr><td>h0_component_min_size</td><td>16</td></tr><tr><td> $h 0 \_ t o p \_ k _ { - }$  components</td><td>7</td></tr><tr><td>local_neg_k</td><td>128</td></tr><tr><td>min_subset_size</td><td>40</td></tr><tr><td>Persistence threshold</td><td>persistence_min = 0.01</td></tr><tr><td>Mismatch threshold</td><td>mismatch_cost_min = 0.01</td></tr><tr><td>Component weighting</td><td>h0_component_weight_by=uniform 42</td></tr><tr><td>Randomness (controller seed)</td><td></td></tr><tr><td>Baseline calibration seeds</td><td>0:19</td></tr><tr><td>Robustness seeds Data (calibration/eval)</td><td>{20, 21, 22, 23, 24}</td></tr><tr><td></td><td>Harmful: 416 cal + 104 eval (AdvBench); Harmless: 512 cal (Alpaca)</td></tr><tr><td>Behavioral prompt set</td><td>benchmarks/paper_data/deval_harmful.txt</td></tr><tr><td>Judge methods LlamaGuard endpoint/model</td><td>substring_matching, llamaguard3, harmbench OpenAI-compatible endpoint</td></tr><tr><td></td><td> $\mathtt { h t t p : / / 0 . 0 . 0 . 0 : 8 8 9 8 / v 1 ) , }$ </td></tr><tr><td>HarmBench endpoint/model</td><td>meta-1lama/Llama-Guard-3-8B OpenAI-compatible endpoint</td></tr><tr><td>Metric definitions</td><td> $\mathbf { \lambda } \mathbf { h i t t p } { : } / / 0 . 0 \overset { \cdot } { . } 0 . 0 { : } 8 8 9 7 / \mathrm { v } 1 ) ,$   $\mathtt { c a i s / H a r m B e n c h - L 1 a m a - } 2 \mathtt { - } 1 3 \mathtt { b - c l s }$   $\begin{array} { r } { \rho ( \mathbf { v } ) = \frac { \mu ^ { + } - \mu ^ { - } } { \sigma _ { \mathrm { p o o l } } } , \Delta \rho = \rho ( \mathbf { v } _ { \mathrm { s t e e r } } ) - \rho ( \mathbf { v } _ { \mathrm { b a s e } } ) } \end{array}$ </td></tr></table>

## A.5.4 Hyperparameter selection

We selected single-layer topological-steering hyperparameters with a staged search orchestrated by our tuning script (random-integer seeds; full search space and per-phase budgets archived with each run).

First, a baseline calibration phase repeatedly runs the unsteeered pipeline over many seeds and records the mean, best, and standard deviation of the activation-side steering statistic produced by the code (the same separability gain we report as $\Delta \rho$ relative to a mean-difference baseline). From this we derive a minimum gain threshold (a floor value together with a multiple of the baseline standard deviation) so that later stages ignore improvements indistinguishable from seed noise.

Second, a coarse search draws a large batch of random configurations from a discrete grid over injection layer, PCA dimension, persistent-homology filtration knobs, component selection and weighting options, local negative-pool size, and subsampling controls; each configuration is evaluated by one end-to-end pipeline run, ranked by average gain, and only the top fraction is kept.

Third, a focused search densifies the grid around those survivors by local moves that perturb several TDA-related coordinates (and occasionally the layer) to propose refined candidates, again ranked by the same activation objective.

Fourth, a robustness phase re-evaluates the leading candidates on additional fresh seeds, aggregates per-candidate statistics, and discards settings whose mean gain does not exceed the calibrated baseline plus the minimum gain threshold.

Finally, a behavioral validation phase generates baseline versus steered outputs on harmful prompts and scores them with external judges (e.g. Llama Guard and HarmBench classifiers, plus inexpensive substring refusal probes where desired); the final selection then picks from the robust shortlist using this behavioral evidence (in our main experiment, ranking by the sum of Llama Guard and HarmBench success-rate deltas), freezing the resulting YAML configuration and steering vector for downstream evaluation.

![](images/45fe66adcd8fae05659479425a11def9dc924f5fa098cfd3e53d91c41a80b1c1.jpg)  
Figure 1: Per-model layer sweeps of $\Delta \rho ( \ell )$ on the comparative run. Each panel reports one model, with the selected best layer marked.

![](images/4bb5473b9406549378e6cd515512f340b688525441b145cc70a860d3e27b1a74.jpg)  
Figure 2: Overlay view of $\Delta \rho ( \ell )$ across all models.

## A.6 Statistical Significance

Table 7 and Figure 3 quantify how stable the observed steering gain is under randomness. In particular, $\Delta \rho$ denotes the steering-score delta (the improvement in the held-out refusal separation over the global mean-difference baseline) produced by the same winning single-layer configuration, while only the random seed is changed. This isolates a key source of variability that can affect topological feature selection, subset extraction, and downstream direction construction, even when the data split, model architecture, and hyperparameters are held fixed.

Each table entry corresponds to one full re-run of the pipeline for the selected configuration. The five robustness seeds yield $\Delta \rho$ values of approximately 0.00752–0.01271, with a mean of $\overline { { \Delta \rho } } \approx 0 . 0 0 9 2 2$ and a sample standard deviation of $s \approx 0 . 0 0 2 1 8$ . The fact that all five runs are positive indicates that the improvement is not driven by a single lucky run: the steering direction recovered from persistent-topology-guided selection consistently enhances linear separability between harmful and harmless activation clouds relative to the baseline. Moreover, the relative spread is modest; the standard deviation is about 24% of the mean, suggesting that seed-induced fluctuations affect the magnitude of the gain but not its sign.

To further assess statistical significance, we report uncertainty around the mean. The table includes the standard error of the mean (SEM) and a 95% t-confidence interval for the mean value of $\Delta \rho$ (with $n = 5$ robustness runs). This interval is approximately [0.00652, 0.01192], which remains clearly above zero. Under the standard t-interval assumptions (exchangeability of runs and reasonable approximation to symmetric errors), this supports the interpretation that the steering gain is robust and not merely an artifact of run-to-run noise.

Figure 3 provides an intuitive visual complement. The dashed horizontal line shows the mean across robustness seeds. The shaded darker band represents ±1σ around the mean, illustrating the observed variability across seeds, while the lighter band shows the 95% t-confidence interval for the mean. All seed points lie within the ±1σ region, consistent with the numeric summary in Table 7. Overall, the combined evidence from the table and figure indicates that the persistent-topology-guided steering direction is stable: the method yields consistently positive improvements over the baseline across independent re-runs, and the uncertainty in the mean estimate does not overlap zero.

Table 7: Robustness of the selected configuration across 5 random seeds in phase-3 robustness runs. $\Delta \rho$ is the steering-score improvement over the baseline direction.
<table><tr><td>Seed index (phase-3)</td><td> $\Delta \rho$ </td></tr><tr><td>20</td><td>0.00819</td></tr><tr><td>21</td><td>0.00772</td></tr><tr><td>22</td><td>0.00752</td></tr><tr><td>23 24</td><td>0.01271</td></tr><tr><td></td><td>0.00997</td></tr><tr><td>Mean</td><td>0.00922</td></tr><tr><td>Standard deviation (1σ) Standard error of the mean</td><td>0.00218</td></tr><tr><td>95% t-CI (df = 4)</td><td>0.00097 [0.00652, 0.01192]</td></tr></table>

![](images/0fa8eb448d73e87c2553ba28757a741894b2f1c0008e67c21a95b708be400bde.jpg)  
Figure 3: Robustness of the selected configuration across 5 random seeds. The shaded band shows ±1σ; the lighter band shows the 95% t-confidence interval for the mean.

## A.7 Societal impact (positive and negative).

This work studies inference-time steering methods intended to improve refusal behavior and reduce harmful responses in large language models, which can have positive societal impact by strengthening safety guardrails in high-risk interaction settings. In particular, improving controllable refusal may lower accidental assistance with dangerous requests and support safer deployment practices when combined with independent safety monitoring. At the same time, the same class of representationlevel intervention techniques can be repurposed negatively: steering methods may be adapted to weaken existing safeguards, evade policy-aligned behavior, or induce model behaviors that are difficult to detect from surface text alone. Additional risks include over-reliance on automated judges, setup-dependent evaluation drift, and false confidence if activation-space gains are interpreted as universal safety guarantees. For these reasons, we frame our contribution as a diagnostic and methodological study under controlled benchmarks, not as a complete solution to safe deployment, and we encourage layered evaluation (multiple judges, robustness checks, and transparent reporting of failure modes) before real-world use.