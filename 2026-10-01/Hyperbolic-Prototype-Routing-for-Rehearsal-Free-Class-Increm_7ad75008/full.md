# Hyperbolic Prototype Routing for Rehearsal-Free Class-Incremental Learning

HongWei Zhao<sup>1</sup>, Rui Liu<sup>1</sup>, Yong Chen<sup>†2</sup>

<sup>1</sup>Beihang University, Beijing, China

<sup>2</sup>Beijing University of Posts and Telecommunications, Beijing, China

{zhaohongwei, lr}@buaa.edu.cn, alphawolf.chen@gmail.com

Abstract—Class-Incremental Learning (CIL) aims to continually learn new classes while preserving prior knowledge. Parameter-efficient fine-tuning (PEFT) with pre-trained models (PTMs) has recently shown promise, enabling CIL with minimal parameter updates. However, existing approaches still suffer from catastrophic forgetting due to cumulative interference across tasks and suboptimal module-sample matching during inference. We propose Hyperbolic Prototype Routing (HyPro), a rehearsalfree framework that exploits hyperbolic geometry for continual learning. First, we allocate a dedicated LoRA-Expert module for each incremental task, enabling isolated representation learning and eliminating cross-task interference. Second, to enhance module-sample matching, we introduce a HyPro mechanism that projects features onto a Poincare ball and performs geodesic´ nearest-prototype matching, yielding exponentially larger decision margins and more reliable task-level discrimination. Extensive experiments on standard CIL and Few-Shot CIL (FSCIL) benchmarks show that HyPro consistently improves average and final-stage accuracy over strong baselines. Code is available at: https://github.com/Geeks-Z/ICME-HyPro-main.

Index Terms—Class-Incremental Learning, Hyperbolic Geometry, Catastrophic Forgetting

## I. INTRODUCTION

In open-world environments, data often arrives as a continuous stream of novel categories, a paradigm formalized as Class-Incremental Learning (CIL). Traditional machine learning models trained on such sequential data suffer from catastrophic forgetting [1], where acquiring new knowledge severely disrupts previously learned representations, leading to significant performance degradation. Incremental learning addresses this challenge by balancing the acquisition of new information with the retention of existing knowledge—a trade off termed the stability-plasticity dilemma [2].

Pre-trained models (PTMs), with their strong generalization capabilities [3] derived from large-scale datasets, offer a promising foundation for CIL. However, fine-tuning all parameters of a PTM in continual learning scenarios risks undermining its inherent generalization. Recent advances mitigate this by freezing the PTM backbone and integrating parameterefficient fine-tuning (PEFT) modules [4], such as prompts [5], [6], adapters [7], and LoRA [8], [9]. These methods enable task-specific adaptation with minimal trainable parameters, preserving generalization while reducing forgetting.

Despite progress, two critical challenges persist:

1. Stability–plasticity imbalance under cumulative interference:

• Shared prompt pools [5] are prone to overwriting earlier tasks’ knowledge when exposed to shifting data distribu tions.

• LoRA-based strategies [8], [9], though effective in constraining updates to mitigate forgetting, inadvertently restrict the plasticity needed for new task adaptation.

• Fusion-based methods [8] attempt to balance old and new knowledge but often degrade the fidelity of both due to forced trade-offs in shared parameters.

2. Inference-stage module-sample mismatches:

• Fixed PTM selection mechanisms [5], [10] struggle under substantial domain shifts, resulting in suboptimal activations and degraded predictions.

• With volume growth, high-dimensional features concentrate into narrow cones (“Cone Effect” [11]), causing prototype crowding and ambiguous boundaries as tasks accumulate, which leads to inference-stage module—sample mismatches.

These issues motivate our core research question: Can we jointly optimize stability-plasticity and module retrieval to mitigate catastrophic forgetting?

To this end, we propose HyPro, a novel rehearsal-free framework for PEFT-based CIL. HyPro comprises two synergistic components operating in distinct stages. First, we dynamically allocate dedicated LoRA-Experts for each new incremental task. Unlike shared or sequentially fine-tuned modules, each LoRA-Expert exclusively encodes task-specific knowledge, with only the current expert being trainable and all prior experts frozen. These lightweight experts are strategically integrated into Transformer architectures—specifically within multilayer perceptrons (MLPs) and multi-head attention (MHA) layers—to efficiently capture discriminative taskspecific features. Second, to overcome the geometric limitations of Euclidean feature space, we project features into a hyperbolic manifold and perform geodesic prototype routing for module-sample. By leveraging the exponentially expanding capacity of hyperbolic space, our approach ensures large interclass distances and improves sample-to-module alignment.

Our principal contributions are threefold:

• We introduce dynamic LoRA-Expert allocation into PEFT-based CIL, allowing each incremental task to obtain its own dedicated expert without predefining the total number of experts. This design ensures strict parameter isolation across tasks and enhances the model’s plasticity to new knowledge.

• We propose a hyperbolic prototype routing mechanism that embeds features into a non-Euclidean manifold for module selection. By leveraging the exponentially expanding capacity of hyperbolic space to preserve large pairwise margins between class prototypes, this design overcomes the limitations of conventional Euclidean feature representations during inference.

• We provide rigorous empirical validation across six challenging benchmarks, establishing new state-of-the-art results in diverse incremental learning scenarios. Furthermore, we demonstrate architectural flexibility through HyPro-MLP, a variant employing alternative LoRA-Expert placement that maintains competitive performance while offering implementation alternatives.

## II. RELATED WORK

## A. Class Incremental Learning

Class-Incremental Learning requires models to continually learn new classes while retaining prior knowledge. Existing methods can be broadly categorized as follows [5]:

Parameter regularization-based methods [12] mitigate catastrophic forgetting by constraining important weights. However, they often underperform on complex datasets or in more challenging incremental scenarios [13].

Rehearsal-based methods utilize data replay to reinforce old knowledge, either by storing raw images [13] or intermediate features [14]. While effective, these methods are limited by buffer sizes and potential privacy concerns.

Dynamic network-based methods [15] allocate new subnetworks for each incremental task while freezing previous ones, thus effectively balancing stability and plasticity. However, they introduce substantial memory overhead due to continually expanding parameters, and many still require old samples for cross-network fusion.

## B. PEFT-Based CIL

Recent advances in PEFT have gained prominence for enabling efficient training and inference with minimal parameters while preserving the strong generalization capabilities of PTMs. Integrating PEFT into PTM-based CIL offers a promising approach to enhancing continual learning performance.

Prompt-based methods like L2P [5] introduced a shared prompt pool for sequential learning. S-Prompts [10] learned domain-specific prompts retrieved by K-NN, while CODA-Prompt [6] proposed an end-to-end decomposed attention framework without rehearsal.

LoRA-based methods, such as InfLoRA [8], project updates into a gradient-orthogonal subspace to avoid interference, requiring extensive data storage. MoE-Adapters [16] employed a dynamic activate-freeze strategy for inter-task collaboration, but its pre-defined number of LoRAs limited adaptation to subsequent incremental tasks. SD-LoRA [9] decoupled gradient direction and magnitude, preserving earlytask directions while adapting to new tasks via shrinking, at the cost of reduced plasticity.

## C. Geometric Deep Learning and Hyperbolic Embeddings

Geometric deep learning studies representation learning beyond Euclidean vector spaces, leveraging non-Euclidean structures such as graphs and Riemannian manifolds. Among them, hyperbolic embeddings are particularly effective for data with latent hierarchical or tree-like organization, because hyperbolic space provides exponentially expanding volume with radius, enabling compact yet highly separable representations.

Poincare embeddings´ [17] demonstrated that hyperbolic space can represent hierarchical relations with substantially lower distortion than Euclidean embeddings. Subsequently, hyperbolic neural networks generalized common operations to the manifold using tools such as exponential/log maps and Mobius addition, enabling end-to-end learning in hyperbolic¨ space [18], [19]. In vision, hyperbolic representations have also been shown to be beneficial for capturing fine-grained relations and improving discrimination when features exhibit implicit hierarchy [19]. These properties motivate us to exploit hyperbolic geometry to enlarge inter-class margins for more reliable prototype-based routing in continual learning.

## III. PRELIMINARIES

Problem Definition: We study class-incremental learning over a stream of tasks $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T }$ , where $\mathcal { D } _ { t } = \{ ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { n _ { t } }$ and task label spaces are disjoint $( \mathcal { V } _ { t } \cap \mathcal { V } _ { t ^ { \prime } } = \varnothing$ for $t \neq t ^ { \prime } )$ Under the rehearsal-free setting [5], [6], training on task t uses only $\mathcal { D } _ { t }$ to learn $f _ { \Theta } ( \mathbf { x } ) = \mathbf { W } _ { c l s } ^ { \top } \phi ( \mathbf { x } )$ by minimizing

$$
\mathbf { L } ( \mathcal { D } _ { t } ) = \frac { 1 } { | \mathcal { D } _ { t } | } \sum _ { ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \in \mathcal { D } _ { t } } \mathbf { L } ( f _ { \Theta } ( \mathbf { x } _ { i } ) , \mathbf { y } _ { i } ) ,\tag{1}
$$

and evaluation after task t is performed on all observed classes $\textstyle \mathbf { Y } _ { t } = \bigcup _ { k = 1 } ^ { t } \mathcal { Y } _ { k }$ . The embedding function $\phi \left( \cdot \right)$ refers to the final [CLS] token in Vision Transformer (ViT) architecture [20] and $\mathbf { W } _ { c l s }$ represents the classifier parameters.

Low-Rank Adaptation. LoRA was introduced to efficiently fine-tune large PTMs by injecting low-rank updates into weight matrices [21]. Given a pre-trained weight matrix $\mathbf { W } \in \mathbf { R } ^ { d \times d }$ , LoRA learns an additive low-rank decomposition:

$$
\begin{array} { r } { \mathbf { W } + \Delta \mathbf { W } = \mathbf { W } + \mathbf { B } \mathbf { A } , } \end{array}\tag{2}
$$

where $\mathbf { A } \in \mathbf { R } ^ { r \times d } , \mathbf { B } \in \mathbf { R } ^ { d \times r }$ and the rank $r \ll d .$ This reduces trainable parameters while maintaining expressiveness. LoRA enables cost-effective, scalable fine-tuning, making it suitable for continual learning where efficiency and avoiding forgetting are critical.

Hyperbolic Poincare Ball Model.´ We adopt the $d -$ dimensional Poincare ball model of constant sectional curva-´ $\mathrm { t u r e } \mathrm { ~ - } c \left( c > 0 \right)$

$$
\mathbb { D } _ { c } ^ { d } = \{ { \bf x } \in \mathbb { R } ^ { d } \vert c \Vert { \bf x } \Vert ^ { 2 } < 1 \} ,\tag{3}
$$

![](images/7ed558265b97060c141996db5578563cb5118fadf82afb6a8c7aee99d4e9bd2b.jpg)  
Fig. 1: Illustration of HyPro. In the t-th incremental task, a new LoRA-Expert $\mathbf { E } _ { t }$ (with parameters ${ \bf A } _ { t }$ and $\mathbf { B } _ { t } )$ is trained to capture task-specific features. Domain-specific features from the router ${ \bf E } ^ { r o u t e r }$ are projected into the hyperbolic space (Poincare ball). During inference, the nearest prototype guides LoRA-Experts selection for each input sample.´

i.e., the open Euclidean ball of radius $1 / { \sqrt { c } } .$ The Riemannian metric is conformal to the Euclidean metric $g ^ { E }$ :

$$
g _ { \bf x } ^ { \mathbb { D } } = \lambda _ { \bf x } ^ { 2 } g ^ { E } , \qquad \lambda _ { \bf x } = \frac { 2 } { 1 - c \| { \bf x } \| ^ { 2 } } .\tag{4}
$$

The geodesic distance between $\mathbf { x } , \mathbf { y } \in \mathbb { D } _ { c } ^ { d }$ admits the closed form

$$
d _ { \mathbb { D } } ( \mathbf { x } , \mathbf { y } ) ~ = ~ \frac { 1 } { \sqrt { c } } \operatorname { a r c o s h } \left( 1 + 2 c \frac { \| \mathbf { x } - \mathbf { y } \| ^ { 2 } } { ( 1 - c \| \mathbf { x } \| ^ { 2 } ) ( 1 - c \| \mathbf { y } \| ^ { 2 } ) } \right) .\tag{5}
$$

Note that the argument of arcosh is $\geq ~ 1$ and that as $\| \mathbf { x } \|  1 / \sqrt { c }$ the conformal factor $\lambda _ { \mathbf { x } }  \infty ,$ , so distances to the boundary diverge. Consequently, hyperbolic space exhibits exponential volume growth near the boundary; this geometric trait makes the Poincare ball suitable for embedding´ hierarchical or tree-like structures, since a rapidly growing representational capacity is available compared to Euclidean space of the same nominal dimensionality.

## IV. THE PROPOSED METHOD

Following HiDe-Prompt [22], we decompose CIL prediction into two probabilistic components: Module-Identity Inference (MII) and Within-Task Prediction (WTP). By Bayes’ theorem, the probability of a sample x belonging to class $j$ in task i is:

$$
P ( \mathbf x \in \mathcal X _ { i , j } | \mathcal D , \Theta ) = \underbrace { P ( \mathbf x \in \mathcal X _ { i , j } | \mathbf x \in \mathcal X _ { i } , \mathcal D , \Theta ) } _ { \mathrm { W I P } } \underbrace { P ( \mathbf x \in \mathcal X _ { i } | \mathcal D , \Theta ) } _ { \mathrm { M I I } }\tag{6}
$$

Eq. (6) implies that overall performance hinges on optimizing both WTP and MII. However, existing methods often falter in these areas: iterative updates or fusion degrade WTP via interference [5], [8], while reliance on fixed PTM features limits MII under domain shifts [10]. To overcome these limitations, we propose HyPro, which explicitly targets both components: Dynamic LoRA-Experts ensure isolated, highfidelity WTP, while Hyperbolic Prototype Routing leverages geometric properties to maximize MII accuracy. An overview is shown in Fig. 1.

## A. Dynamic LoRA-Experts

We investigate the integration of LoRA-Expert into the ViT. ViT first splits an input image into fixed-size patches, which are then linearly projected and augmented with positional embeddings before being processed by the Transformer encoder. The encoder comprises multi-head self-attention (MHA) layers and multilayer perceptron (MLP) blocks. To adaptively capture task-specific features in incremental learning, we dynamically incorporate a LoRA-Expert module at each incremental stage. This module can be attached either as a parallel branch to the MLP (MLP-Expert) or to the query $( \mathbf { W } _ { q } ) _ { : }$ , key $( \mathbf { W } _ { k } )$ or value $( \mathbf { W } _ { v } )$ projections in the MHA mechanism $\mathbf { \Gamma } ( \mathbf { Q K V } .$ Expert). When applied to both $\mathbf { W } _ { q }$ and $\mathbf { W } _ { v } ,$ the configuration is termed (QV-Expert). Formally, letting $\phi (  { \mathbf { x } } ;  { \mathbf { E } _ { t } } )$ denote the embedding with LoRA-Expert $\mathbf { E } _ { t }$ , we write the MLP and MHA updates as:

$$
\phi ( \mathbf { x } ; \mathbf { E } _ { t } ^ { \mathrm { M L P } } ) = \mathbf { e } + \mathrm { M L P } ( \mathbf { e } ) + \mathbf { E } _ { t } ^ { \mathrm { M L P } } ( \mathbf { e } ) ,\tag{7}
$$

$$
\phi ( \mathbf { x } ; \mathbf { E } _ { t } ^ { \mathrm { Q K V } } ) = \mathrm { A t t n } \big ( \mathbf { u } _ { Q } + \mathbf { E } _ { t } ^ { Q } ( \mathbf { e } ) , \mathbf { u } _ { K } + \mathbf { E } _ { t } ^ { K } ( \mathbf { e } ) , \mathbf { u } _ { V } + \mathbf { E } _ { t } ^ { V } ( \mathbf { e } ) \big ) ,\tag{8}
$$

where e and u are the input and output of the original module, and $\mathbf { E } _ { t }$ denotes the output of the t-th LoRA-Expert. Here, the superscripts $( \mathbf { e . g . } , \mathbf { E } _ { t } ^ { \mathrm { M L P } } , \mathbf { E } _ { t } ^ { Q } , \mathbf { E } _ { t } ^ { K } , \mathbf { E } _ { t } ^ { V } )$ indicate different finetuning locations within the ViT block. The attention operator is

$$
\operatorname { A t t n } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = \operatorname { s o f t m a x } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } } \right) \mathbf { V } ,\tag{9}
$$

and the multi-head formulation is omitted for brevity.

Each LoRA-Expert $\mathbf { E } _ { t }$ introduces task-specific parameters ${ \bf A } _ { t }$ and $\mathbf { B } _ { t } ,$ computing $\mathbf { E } _ { t } ( \mathbf { e } ) = \mathbf { B } _ { t } \mathbf { A } _ { t } \mathbf { e }$ . While the backbone remains shared, each task maintains a distinct expert. We initialize $\mathbf { B } _ { t }$ to zero and ${ \bf A } _ { t }$ via Kaiming initialization [23] for the first task; subsequent experts inherit weights from the preceding task to accelerate convergence. Crucially, during training on task t, all prior experts $( \mathbf { E } _ { 1 \ldots t - 1 } )$ are frozen. This isolation strategy ensures that new knowledge is acquired without interfering with previously learned representations, effectively mitigating catastrophic forgetting with minimal parameter overhead.

## B. Hyperbolic Prototype Routing

While the frozen PTM backbone offers robust generalization, it inherently lacks the plasticity to capture task-specific discriminative features, particularly under significant domain shifts. Relying solely on PTM representations for module selection often leads to suboptimal expert retrieval. To bridge this gap, we introduce a Hyperbolic Prototype Routing mechanism that synergizes a learnable domain adapter with the exponential capacity of the Poincare manifold.´

1) Manifold Projection and Learnable Router: We employ a dedicated router module, ${ \bf E } ^ { r o u t e r }$ , which shares the architecture of LoRA-Experts but is continuously updated across all tasks. For an input x, the router extracts a domain-specific feature $\textbf { z } \in \mathbb { R } ^ { d }$ with $\phi ( \mathbf { x } ; \mathbf { E } ^ { r o u t e r } )$ . To mitigate the “Cone Effect” in high-dimensional Euclidean space, we project z onto the d-dimensional Poincare ball ´ $\mathbb { D } _ { c } ^ { d }$ with curvature −c. Using the exponential map at the origin, exp<sup>c</sup><sub>0</sub> : $T _ { 0 } \mathbb { D } _ { c } ^ { d }  \mathbb { D } _ { c } ^ { d }$ , the Euclidean feature is mapped to the hyperbolic embedding h:

$$
\mathbf { h } = \exp _ { 0 } ^ { c } ( \mathbf { z } ) = \operatorname { t a n h } \left( \sqrt { c } \left\| \mathbf { z } \right\| \right) \frac { \mathbf { z } } { \sqrt { c } \left\| \mathbf { z } \right\| } .\tag{10}
$$

The tanh nonlinearity naturally compresses the embedding magnitude, pushing highly discriminative features toward the boundary of $\mathbb { D } _ { c } ^ { d }$ . In this boundary region, the hyperbolic volume expands exponentially, providing ample geometric capacity to separate accumulating task prototypes without interference.

The router network $\mathbf { E } ^ { r o u t e r }$ is trained after each new task’s LoRA-Expert is learned. To construct the optimization targets, we first extract domain-specific prototypes P from the trained LoRA-Expert of the current task. For a task t with dataset $\mathcal { D } _ { t }$ the prototype for class i is formulated as:

$$
\mathbf { P } _ { i } = \frac { 1 } { N _ { i } } \sum _ { j = 1 } ^ { | \mathcal { D } _ { t } | } \mathbb { I } ( y _ { j } = i ) \phi ( \mathbf { x } _ { j } ; \mathbf { E } _ { t } ) .\tag{11}
$$

Let $\begin{array} { r } { \mathbf { h } _ { p } = \exp _ { 0 } ^ { c } ( \mathbf { P } _ { i } ) } \end{array}$ denote the hyperbolic prototype of the ground-truth class i.

To ensure the router accurately assigns samples to these prototypes while retaining knowledge of previous tasks, we minimize a hybrid loss defined directly on the manifold:

$$
\operatorname* { m i n } _ { \mathbf { E } _ { t } ^ { r o u t e r } } \mathrm { H } _ { r o u t e r } = \alpha d _ { \mathbb { D } } \bigl ( \mathbf { h } _ { q } , \mathbf { h } _ { p } \bigr ) + ( 1 - \alpha ) d _ { \mathbb { D } } \bigl ( \mathbf { h } _ { q } , \mathbf { h } _ { o l d } \bigr ) ,\tag{12}
$$

where $\mathbf { h } _ { q }$ is the router’s output for input x, and $\alpha \in$ is a trade-off hyperparameter. The first term (plasticity) aligns the router’s output with the current task’s expert features. The second term (stability) acts as hyperbolic knowledge distillation by aligning the current embedding $\mathbf { h } _ { q }$ with the router output from the previous stage, $\mathbf { h } _ { o l d } = \mathbf { \bar { \rho } } \exp _ { 0 } ^ { c } \left( \phi ( \mathbf { x } ; \mathbf { E } _ { \mathrm { r o u t e r } } ^ { t - 1 } ) \right)$ thereby preventing decision boundary drift for prior tasks. The computation of $d _ { \mathbb { D } }$ follows Eq. (5).

2) Expert Selection: During inference, the query embedding $\mathbf { h } _ { q }$ is matched against the stored prototype using the Riemannian geodesic distance:

$$
i = \operatorname { a r g m i n } ( d _ { \mathbb { D } } ( \mathbf { h } _ { q } , \mathbf { h } _ { p } ) ) .\tag{13}
$$

The LoRA-Expert corresponding to the nearest prototype is then activated for final prediction. Crucially, as prototypes approach the boundary, the denominator $( 1 - c \| \mathbf h \| ^ { 2 } ) \to 0 .$ causing inter-class distances to diverge. This creates large “safety margins” between tasks, minimizing routing confusion.

## C. Optimization Objective for HyPro

The optimization process of HyPro consists of two parts: one is the learning of the dynamic LoRA-Expert, and its optimization objective function can be formulated as:

$$
\operatorname* { m i n } _ { \mathbf { W } _ { c l s } , \mathbf { E } _ { i } } \mathrm { L } \left( \mathbf { W } _ { c l s } ^ { \top } \phi \left( \mathbf { x } ; \mathbf { E } _ { i } \right) , \mathbf { y } \right) ,\tag{14}
$$

where L is the cross-entropy loss. In the second stage, the router is optimized via Eq. (12).

During inference, we adopt the class prototypes extracted by the LoRA-Experts as the classifier weights, $\mathbf { W } _ { c l s } = \mathbf { P }$ and use a cosine classifier for final prediction:

$$
\mathrm { f } ( \mathbf { x } | \mathbf { E } _ { i } ) = ( \frac { \mathbf { W } _ { c l s } } { \| \mathbf { W } _ { c l s } \| _ { 2 } } ) ^ { \top } ( \frac { \phi ( \mathbf { x } ; \mathbf { E } _ { i } ) } { \| \phi ( \mathbf { x } ; \mathbf { E } _ { i } ) \| _ { 2 } } ) ,\tag{15}
$$

where $\mathbf { E } _ { i }$ denotes the selected LoRA-Expert by Eq. (13) for the input x.

## V. EXPERIMENTS

## A. Experimental Settings

Datasets: We evaluate HyPro on two tasks: CIL and Few-Shot Class-Incremental Learning (FSCIL) [24]. For CIL, we follow standard protocols [9] and test on five benchmarks: CIFAR100 [25], CUB200 [26](ImageNet-R [27], Omnibenchmark [28], and VTAB [29] are provided in the Supplementary), with classes evenly divided into T incremental tasks.

For FSCIL, we adopt the settings of PriViLege [30] and ASP [31] on CUB200 (100-base 10-way 5-shot) and CI-FAR100 (60-base 5-way 5-shot), where “100-base” indicates that the first task contains 100 classes with sufficient training samples, “10-way 5-shot” means that each subsequent task introduces 10 novel classes, each with only 5 examples.

TABLE I: Performance comparison of selected CIL methods, all built on the same pre-trained backbone (ViT-B/16-IN21K).
<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR100 (T=10)</td><td colspan="2">CUB200 (T=10)</td></tr><tr><td> $\boldsymbol { \mathcal { A } } _ { L }$ </td><td>A</td><td> $\boldsymbol { \mathcal { A } } _ { L }$ </td><td>A</td></tr><tr><td>Full Fine-Tuning</td><td>66.26</td><td>76.94</td><td>55.29</td><td>70.30</td></tr><tr><td>SimpleCIL [7]</td><td>81.27</td><td>87.13</td><td>82.28</td><td>91.85</td></tr><tr><td>L2P [5]</td><td>84.82</td><td>89.78</td><td>71.98</td><td>81.80</td></tr><tr><td>CODA-Prompt [6]</td><td>86.69</td><td>91.31</td><td>75.45</td><td>84.65</td></tr><tr><td>InfLoRA [8]</td><td>86.43</td><td>91.80</td><td>70.07</td><td>81.71</td></tr><tr><td>SD-LoRA [9]</td><td>87.62</td><td>92.10</td><td>72.69</td><td>83.17</td></tr><tr><td>MoE-Adapters [16]</td><td>77.96</td><td>85.19</td><td>52.62</td><td>65.67</td></tr><tr><td>HyPro-MLP</td><td>89.13</td><td>93.23</td><td>87.79</td><td>92.11</td></tr><tr><td>HyPro-QV</td><td>89.68</td><td>93.62</td><td>88.13</td><td>92.20</td></tr></table>

TABLE II: Performance comparison of selected FSCIL methods, all built on the same pre-trained backbone (ViT-B/16- IN21K).
<table><tr><td rowspan="2">Method</td><td colspan="3">CUB200 (T=11)</td><td colspan="3">CIFAR100 (T=9)</td></tr><tr><td> $\mathcal { A } _ { \mathrm { B a s e } }$ </td><td> $\boldsymbol { A } _ { L }$ </td><td> $\bar { A }$ </td><td> $A _ { \mathrm { B a s e } }$ </td><td> $A _ { L }$ </td><td> $\bar { A }$ </td></tr><tr><td>L2P [5]</td><td>91.50</td><td>50.04</td><td>66.70</td><td>93.43</td><td>55.75</td><td>71.81</td></tr><tr><td>CODA-Prompt [6]</td><td>91.50</td><td>53.65</td><td>69.30</td><td>94.05</td><td>57.10</td><td>73.11</td></tr><tr><td>InfLoRA [8]</td><td>92.45</td><td>45.18</td><td>66.27</td><td>94.92</td><td>57.41</td><td>74.28</td></tr><tr><td>SD-LoRA [9]</td><td>91.92</td><td>56.28</td><td>70.87</td><td>94.60</td><td>73.51</td><td>78.42</td></tr><tr><td>CPE-CLIP [32]</td><td>80.21</td><td>63.32</td><td>69.37</td><td>88.32</td><td>79.99</td><td>83.38</td></tr><tr><td>ASP [31]</td><td>87.14</td><td>82.86</td><td>83.46</td><td>91.77</td><td>86.04</td><td>88.54</td></tr><tr><td>PriViLege [30]</td><td>82.21</td><td>75.08</td><td>77.50</td><td>90.88</td><td>86.06</td><td>88.08</td></tr><tr><td>HyPro-MLP</td><td>92.68</td><td>76.84</td><td>84.92</td><td>93.75</td><td>88.96</td><td>91.16</td></tr><tr><td>HyPro-QV</td><td>93.35</td><td>66.50</td><td>82.48</td><td>93.80</td><td>88.08</td><td>90.54</td></tr></table>

Evaluation metrics: For CIL, we evaluate performance using two standard metrics: the average accuracy over all incremental tasks $\begin{array} { r } { \bar { \mathcal { A } } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \mathcal { A C C } _ { i } } \end{array}$ and the accuracy on the last task $A _ { L } \ [ 8 ]$ . Here, we denote the Top-1 average accuracy after training on the i-th task as ${ \mathcal { A } } { \mathcal { C } } { \mathcal { C } } _ { i }$ . For FSCIL, we report the first task accuracy $\mathcal { A } _ { \mathrm { B a s e } }$ , last task accuracy $\boldsymbol { A } _ { L } ,$ , and overall average accuracy $\bar { \mathcal { A } } .$

Architecture and Training details: We adopt ViT-B/16- IN21K [20] as our pre-trained backbone, initialized with weights from ImageNet-21K. For optimization, we use SGD with an initial learning rate of 0.02, decayed via a cosine annealing schedule. The LoRA-Expert modules are trained for 20 epochs with a batch size of 48, while the router undergoes 5 epochs of training under the same batch size configuration. LoRA decomposition employs a rank of $r { = } 4 ,$ curvature $c = 0 . 5$ , and the plasticity feature distillation coefficient α is set to 0.1. To ensure reproducibility, all experiments are conducted on an NVIDIA A800 GPU using identical data splits and pre-trained backbones. Reported results are averaged over three independent runs to account for variability.

For more experimental details, please see the Supplementary and the forthcoming code release on GitHub.

TABLE III: Ablation studies on CIL and FSCIL tasks. The first dataset corresponds to CIL, and the second to FSCIL. For each metric, the left/right values represent performance with MLP-LoRA and QV-LoRA fine-tuning, respectively. HPR: Hyperbolic Prototype Routing.
<table><tr><td rowspan="3">Ablated Components</td><td colspan="2">ImageNet-R (T=5)</td><td colspan="2">CIFAR100 (T=9)</td></tr><tr><td> $A _ { L }$ </td><td>A</td><td> $A _ { L }$ </td><td> $\bar { A }$ </td></tr><tr><td>w/o LoRA Dynamically</td><td>61.17 / 72.37</td><td>74.31  / 80.14</td><td>73.03 / 79.73</td><td>80.39 / 86.59</td></tr><tr><td>w/o HPR</td><td>69.53 / 73.13</td><td>77.75 / 80.42</td><td>81.13 / 84.09</td><td>86.97 / 88.67</td></tr><tr><td>HyPro-MLP / QV</td><td>77.00 / 78.10</td><td>82.44 / 83.05</td><td>88.96  / 88.08</td><td>91.16 / 90.54</td></tr></table>

## B. Benchmark Comparison

Class Incremental Learning: We evaluate HyPro against SOTA methods across five benchmarks. As shown in Table I, HyPro consistently achieves superior performance on CIFAR100 and CUB200, while results on ImageNet-R, Omnibenchmark, and VTAB (detailed in the Supplementary) further confirm its efficacy. Specifically, HyPro outperforms leading LoRA-based methods (InfLoRA and SD-LoRA) by margins of 1.5%–2.0% on CIFAR100 and 15%–18% on CUB200 in terms of $\boldsymbol { \mathcal { A } } _ { L }$ . Notably, both HyPro-MLP and HyPro-QV variants exhibit robust performance, maintaining a clear advantage over competing approaches.

Few-Shot Class-Incremental Learning: We further evaluate HyPro on few-shot class-incremental learning. As shown in Table II, HyPro consistently delivers superior accuracy, establishing new state-of-the-art results on multiple benchmarks. In particular, it achieves the highest last accuracy $( \mathcal { A } _ { L } )$ and average accuracy (A<sup>¯</sup>) on CUB200 and CIFAR100. For instance, on CUB200, HyPro-MLP achieves an average accuracy of 84.92%, outperforming the prior best method, ASP, by a margin of 1.46%. On CIFAR100, HyPro-MLP reaches 91.16%, surpassing ASP by 2.62%.

## C. Ablation Study

Different Components: We conduct ablation studies to assess the contribution of each component in HyPro (Table III). The variant w/o LoRA Dynamically uses a single LoRA for all tasks, resulting in a significant performance drop under large domain shifts—indicating severe forgetting and task interference. In contrast, assigning a dedicated LoRA per task better preserves stability–plasticity trade-offs across incremental steps. This underscores the value of task-specific LoRAs in PEFT-based continual learning. Removing the hyperbolic prototype routing (w/o HPR) and using frozen class prototypes as keys also degrades performance, confirming the router’s critical role in effective module–sample matching.

## D. Further Analysis

Impact of Matching Strategies: To validate the effectiveness of our routing mechanism, we compare the proposed Hyperbolic Prototype Routing against two baseline strategies: (1) K-Nearest Neighbors (KNN, K=3), which retrieves experts based on raw feature similarity, and (2) Euclidean Prototype, which employs a learnable router identical to ours but operates within Euclidean space using cosine similarity. As presented in Table IV, HyPro significantly outperforms all baselines. The superior performance over the Euclidean counterpart confirms that projecting features onto the Poincare manifold effectively´ alleviates the “Cone Effect”, providing larger decision margins for more accurate expert retrieval.

TABLE IV: Router average accuracy comparison of different module-sample matching strategies on CIL tasks. All methods are based on the same pre-trained backbone (ViT-B/16- IN21K).
<table><tr><td>Method</td><td>CIFAR100 (T=10)</td><td>CUB200 (T=10)</td><td>ImageNet-R (T=5)</td></tr><tr><td>KNN</td><td>86.80</td><td>90.40</td><td>75.24</td></tr><tr><td>Prototype</td><td>89.60</td><td>91.10</td><td>77.04</td></tr><tr><td>HyPro-MLP</td><td>93.71</td><td>93.40</td><td>88.48</td></tr><tr><td>HyPro-QV</td><td>94.13</td><td>93.32</td><td>88.61</td></tr></table>

## VI. CONCLUSION

We propose HyPro, a novel framework that mitigates catastrophic forgetting by embedding task-specific features into a Poincare ball manifold. By exploiting the exponential capac-´ ity of hyperbolic space, HyPro resolves the “Cone Effect” prevalent in Euclidean PEFT methods. Extensive experiments confirm that HyPro establishes new state-of-the-art results across CIL and FSCIL benchmarks. Future work will explore hyperbolic operations for inter-layer feature fusion.

## REFERENCES

[1] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks: The sequential learning problem.” Academic Press, 1989, vol. 24, pp. 109–165.

[2] S. T. Grossberg, Studies of mind and brain: Neural principles of learning, perception, development, cognition, and motor control. Springer Science & Business Media, 2012, vol. 70.

[3] X. Han, Z. Zhang, N. Ding, Y. Gu, X. Liu, Y. Huo, J. Qiu, Y. Yao, A. Zhang, L. Zhang et al., “Pre-trained models: Past, present and future,” AI Open, vol. 2, pp. 225–250, 2021.

[4] Y. Xin, S. Luo, H. Zhou, J. Du, X. Liu, Y. Fan, Q. Li, and Y. Du, “Parameter-efficient fine-tuning for pre-trained vision models: A survey,” CoRR, vol. abs/2402.02242, 2024.

[5] Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, G. Su, V. Perot, J. Dy, and T. Pfister, “Learning to prompt for continual learning,” in 2022 CVPR, 2022, pp. 139–149.

[6] J. S. Smith, L. Karlinsky, V. Gutta, P. Cascante-Bonilla, D. Kim, A. Arbelle, R. Panda, R. Feris, and Z. Kira, “Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning,” in 2023 CVPR, 2023, pp. 11 909–11 919.

[7] D.-W. Zhou, Z.-W. Cai, H.-J. Ye, D.-C. Zhan, and Z. Liu, “Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need,” IJCV, pp. 1–21, 2024.

[8] Y.-S. Liang and W.-J. Li, “Inflora: Interference-free low-rank adaptation for continual learning,” in 2024 CVPR, 2024, pp. 23 638–23 647.

[9] Y. Wu, H. Piao, L.-K. Huang, R. Wang, W. Li, H. Pfister, D. Meng, K. Ma, and Y. Wei, “Sd-lora: Scalable decoupled low-rank adaptation for class incremental learning,” in 2025 ICLR, 2025.

[10] Y. Wang, Z. Huang, and X. Hong, “S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning,” in Advances in Neural Information Processing Systems 35, 2022.

[11] J. Gao, D. He, X. Tan, T. Qin, L. Wang, and T. Liu, “Representation degeneration problem in training natural language generation models,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=SkEYojRqtm

[12] R. Aljundi, K. Kelchtermans, and T. Tuytelaars, “Task-free continual learning,” in 2019 CVPR, 2019, pp. 11 246–11 255.

[13] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl: Incremental classifier and representation learning,” in 2017 CVPR, 2017, pp. 5533–5542.

[14] L. Yu, B. Twardowski, X. Liu, L. Herranz, K. Wang, Y. Cheng, S. Jui, and J. van de Weijer, “Semantic drift compensation for class-incremental learning,” in 2020 CVPR, 2020, pp. 6980–6989.

[15] F. Wang, D. Zhou, H. Ye, and D. Zhan, “FOSTER: feature boosting and compression for class-incremental learning,” in 2022 ECCV, ser. Lecture Notes in Computer Science, S. Avidan, G. J. Brostow, M. Cisse, G. M.´ Farinella, and T. Hassner, Eds., vol. 13685. Springer, 2022, pp. 398– 414.

[16] J. Yu, Y. Zhuge, L. Zhang, P. Hu, D. Wang, H. Lu, and Y. He, “Boosting continual learning of vision-language models via mixture-of-experts adapters,” in 2024 CVPR, 2024, pp. 23 219–23 230.

[17] M. Nickel and D. Kiela, “Poincare embeddings for learning hierarchical´ representations,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, p. 6341–6350.

[18] O.-E. Ganea, G. Becigneul, and T. Hofmann, “Hyperbolic neural net-´ works,” in Proceedings of the 32nd International Conference on Neural Information Processing Systems, ser. NIPS’18. Red Hook, NY, USA: Curran Associates Inc., 2018, p. 5350–5360.

[19] V. Khrulkov, L. Mirvakhabova, E. Ustinova, I. Oseledets, and V. Lempitsky, “Hyperbolic image embeddings,” in 2020 CVPR, June 2020.

[20] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in 2021 ICLR, 2021.

[21] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in 2022 ICLR, 2022.

[22] L. Wang, J. Xie, X. Zhang, M. Huang, H. Su, and J. Zhu, “Hierarchical decomposition of prompt-based continual learning: Rethinking obscured sub-optimality,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., 2023.

[23] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers: Surpassing human-level performance on imagenet classification,” in 2015 ICCV, 2015, pp. 1026–1034.

[24] X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong, “Few-shot class-incremental learning,” in 2020 CVPR, 2020, pp. 12 180–12 189.

[25] A. Krizhevsky and G. Hinton, “Learning multiple layers of features from tiny images,” 2009.

[26] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The caltech-ucsd birds-200-2011 dataset,” 2011.

[27] D. Hendrycks, S. Basart, N. Mu, S. Kadavath, F. Wang, E. Dorundo, R. Desai, T. Zhu, S. Parajuli, M. Guo et al., “The many faces of robustness: A critical analysis of out-of-distribution generalization,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 8340–8349.

[28] Y. Zhang, Z. Yin, J. Shao, and Z. Liu, “Benchmarking omni-vision representation through the lens of visual realms,” in European Conference on Computer Vision. Springer, 2022, pp. 594–611.

[29] X. Zhai, J. Puigcerver, A. Kolesnikov, P. Ruyssen, C. Riquelme, M. Lucic, J. Djolonga, A. S. Pinto, M. Neumann, A. Dosovitskiy et al., “A large-scale study of representation learning with the visual task adaptation benchmark,” arXiv preprint arXiv:1910.04867, 2019.

[30] K.-H. Park, K. Song, and G.-M. Park, “Pre-trained vision and language transformers are few-shot incremental learners,” in 2024 CVPR, 2024, pp. 23 881–23 890.

[31] C. Liu, Z. Wang, T. Xiong, R. Chen, Y. Wu, J. Guo, and H. Huang, “Few-shot class incremental learning with attention-aware self-adaptive prompt,” in 2024 ECCV. Cham: Springer Nature Switzerland, 2024, pp. 1–18.

[32] M. D’Alessandro, A. Alonso, E. Calabres, and M. Galar, “Multimodal´ parameter-efficient few-shot class incremental learning,” in 2023 IC-CVW, 2023, pp. 3385–3395.