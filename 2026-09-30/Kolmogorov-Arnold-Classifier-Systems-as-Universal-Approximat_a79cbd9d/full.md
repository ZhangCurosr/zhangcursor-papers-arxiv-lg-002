# Kolmogorov-Arnold Classifier Systems as Universal Approximators

Hiroki Shiraishi , Hisao Ishibuchi , Fellow, IEEE, and Masaya Nakata , Member, IEEE

Abstract—As the input dimension � grows, rule-based machine learning, such as Learning Classifier Systems (LCSs), faces a fundamental scalability bottleneck for function approximation: both rule count and parameter count grow exponentially with �. Traditional LCSs partition the �-dimensional input space directly, requiring O(�<sup>�</sup>) rules for adequate coverage, where � is the per-variable resolution. This article breaks from this paradigm by reorganizing rules dimension-wise, guided by the Kolmogorov-Arnold representation theorem: any continuous �- dimensional function can be expressed as a finite superposition of one-dimensional functions. The proposed Kolmogorov-Arnold Classifier System (KACS) decomposes the target function into one-dimensional subproblems and assigns a dedicated ruleset to each, reducing the worst-case rule count from O(�<sup>�</sup>) to O(��<sup>2</sup>) and replacing �-dimensional local models with one-dimensional models requiring only two parameters per rule, independent of �. We also provide the first constructive proof that an LCS, namely KACS, is a universal approximator for continuous functions on compact domains. Evaluated against a direct �-dimensional input space partitioning approach under otherwise identical conditions, KACS achieves competitive accuracy in many settings while using only 2% to 40% of the parameters. Our implementation is available at https://github.com/YNU-NakataLab/KACS.

Index Terms—Learning classifier systems, evolutionary rulebased machine learning, Kolmogorov-Arnold representation theorem, universal approximation, dimension-wise decomposition

## I. INTRODUCTION

UNCTION approximation, also known as regression, is the problem of constructing an approximation ${ \hat { f } } : X \to \mathbb { R }$   
from a set of training samples $\mathcal { D } _ { \mathrm { t r } } ~ = ~ \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { \mathrm { \tilde { N } _ { \mathrm { t r } } } }$ , where   
$\mathbf { x } _ { i } ~ \in ~ { \mathcal { X } } ~ \subseteq ~ \mathbb { R } ^ { n }$ and $y _ { i } ~ \in ~ \mathbb { R }$ . The goal is to approximate   
an unknown target function � such that ${ \hat { f } } ( \mathbf { x } ) ~ \approx ~ f ( \mathbf { x } )$ for   
all x ∈ X. Ideally, <sup>ˆ</sup>� generalizes well to unseen inputs [1],   
[2]. Reliable function approximation is a core requirement   
in many real-world applications [3], where a model supports   
not only data fitting but also decision making, monitoring,   
and control [4], [5]. In such settings, approximation accuracy,

robustness, and multi-dimensional scalability directly affect overall performance and safety [3].

From an algorithmic viewpoint, function approximation becomes challenging when the target relationship $y = f ( \mathbf { x } )$ is complex. In particular, two difficulties are frequently observed:

• Heterogeneity: Different regions of the input space may exhibit different local behaviors, such as sharp changes, discontinuities, or region-specific nonlinearities [6], [7]. In such cases, a single global model can be difficult to train and may require high complexity to represent all local patterns simultaneously [8].

• Multiple input variables: Approximation becomes harder as the number of input variables increases $( \mathrm { i } . \mathrm { e } . , n \ge 2 )$ [1]. Even when � is continuous, multi-dimensional spaces require many samples for adequate coverage [9], and the model class must scale without an explosive increase in parameters [2].

To address heterogeneity, local modeling and divide-andconquer strategies have been widely studied (e.g., regression trees [10], Takagi-Sugeno fuzzy systems [11], and mixture-ofexperts models [7]). Among these approaches, Learning Classifier Systems (LCSs) [12] are a representative family. LCSs are an evolutionary rule-based machine learning paradigm that represents a model as a population of IF-THEN rules [13]. Each rule covers a local region (IF-part or antecedent) and provides a local classification or approximation model (THENpart or consequent); the final output is obtained by aggregating the predictions of matching rules. Because of this rule-based, adaptive local modeling nature, LCSs have been used as automated tools for designing classifiers and approximators [14]– [16]. For function approximation, to the best of our knowledge, all LCSs directly handle rules defined over the �-dimensional input space. For example, XCSF [17], the most widely studied Michigan-style LCS, combines evolutionary search for rule-antecedents with gradient-based online updates of ruleconsequents. SupRB [18], a recently proposed Pittsburgh-style LCS, alternates between an evolutionary rule discovery phase and a batch supervised fitting of rule-consequents. Despite these architectural differences, both systems represent rules over the full �-dimensional input space.

While LCSs are effective in heterogeneous problems due to their local modeling nature [12], they still face a major scalability issue in multi-dimensional settings [19]–[22]. From a partitioning perspective, if each of the � input variables requires � effective subdivisions to achieve good accuracy, the number of local regions (and thus rules) can grow as O(�<sup>�</sup>) [19]. This growth makes both rule discovery and parameter learning increasingly costly [19], [23]. Moreover, when the rule-consequent is parameterized in an �-dimensional form (e.g., linear regression), each rule must estimate multiple parameters, which can be difficult when each local region contains limited data [1].

Based on these considerations, this article proposes a Kolmogorov-Arnold Classifier System (KACS), an LCS designed to address both heterogeneity and multi-dimensional scalability. KACS is guided by the Kolmogorov-Arnold (KA) representation theorem [24], [25]. The KA theorem states that any continuous function of � variables can be represented as a finite superposition of one-dimensional functions [25], [26]:

$$
f ( \mathbf { x } ) = \sum _ { q = 1 } ^ { 2 n + 1 } \Phi _ { q } \left( \sum _ { p = 1 } ^ { n } \psi _ { q , p } ( x _ { p } ) \right) ,\tag{1}
$$

where $\psi _ { q , p }$ are inner one-dimensional functions and $\Phi _ { q }$ are outer one-dimensional functions.

KACS follows this KA structure and converts the original �-dimensional approximation task into learning a collection of one-dimensional functions: the inner functions $\{ \psi _ { q , p } \}$ and the outer functions $\{ \Phi _ { q } \}$ . Instead of evolving an �-dimensional ruleset, KACS focuses on learning one-dimensional rulesets for these functions, which are then combined according to (1). Each one-dimensional ruleset is used to represent an inner or outer function. Thus, the total number of required rulesets is $n ( 2 n + 1 ) + ( 2 n + 1 ) = 2 n ^ { 2 } + 3 n + 1$ . If each ruleset has � rules, the total number of rules is $2 m n ^ { 2 } + 3 m n + m$ . This design has two advantages. First, under a fixed per-variable resolution (i.e., roughly � effective subdivisions per variable), the number of required rules is reduced from $O ( m ^ { n } )$ to $O ( m n ^ { 2 } )$ because KACS allocates rules to one-dimensional intervals rather than to �-dimensional regions (cf. Fig. 1) <sup>1</sup>. Second, KACS replaces �-dimensional rule-consequent models with one-dimensional models, reducing the number of parameters per local model and making parameter estimation more reliable from the same training data [1]. Beyond these practical advantages, this article establishes the theoretical expressive power of KACS by proving that it is a universal approximator for continuous functions on compact domains (cf. Theorem 2) [27]. Specifically, we prove that for any continuous target function � and any desired accuracy level, there always exists a finite KACS model $\hat { f }$ that can approximate � within that accuracy.

Our contributions are as follows:

1) We propose KACS, a new paradigm for rule-based machine learning in which rules are organized dimensionwise rather than by partitioning only the input space (cf. Section IV). Unlike conventional LCSs (e.g., XCSF) that perform divide-and-conquer solely in the �-dimensional input space, KACS decomposes the problem into onedimensional subproblems via the KA theorem, allocates dedicated rulesets to each subproblem, and composes them as a finite superposition. This reduces the worstcase rule count from $O ( m ^ { n } )$ to $O ( m n ^ { 2 } )$

2) We prove that KACS is a universal approximator for continuous functions on compact domains. To our knowledge, this is the first universal approximation proof for any LCS (cf. Section V).

3) We empirically validate the proposed paradigm by holding all design choices constant except rule dimensionality. Specifically, we benchmark KACS against XCSF on artificial and real-world datasets (cf. Section VI). This controlled comparison evaluates KA-based rule organization under otherwise comparable LCS settings, and the results show consistent reductions in both approximation error and model complexity.

It should be noted that compactness and transparency are important practical properties of LCSs [18]. Predictive accuracy is not the sole objective of regression; when the target function is known, fidelity to its underlying structure may also be important. This article focuses on local modeling and scalability rather than on a comparative evaluation of global interpretability. Although individual KACS rules are one-dimensional, we do not claim that the composed KACS model is more globally interpretable than XCSF solely because it uses fewer rules or parameters. A systematic evaluation of global interpretability remains future work.

The rest of this article is organized as follows. Section II describes XCSF and the KA theorem. Section III reviews related work. Section IV presents KACS. Section V establishes the universal approximation theorem of KACS. Section VI reports experimental results, discussion, and analysis. Finally, Section VII concludes this article.

## II. PRELIMINARIES

## A. XCSF Classifier System

XCSF is one of the most successful Michigan-style LCSs for function approximation. It incrementally optimizes a population $\mathcal { P }$ of IF-THEN rules online, updating and evolving rules as new samples arrive. These rules collectively partition the input space, and each rule covers a local region with a linear prediction model. We next provide a brief overview of XCSF. For further details on XCSF, see [15], [17]; for complementary perspectives on LCSs, see [21], [22], [28].

1) Rule Representation: An �-dimensional XCSF rule<sup>2</sup> ��<sub>�</sub> is written as

$$
c l _ { k } : \mathrm { I F } \mathrm { \bf ~ x } \in C _ { k } = \prod _ { i = 1 } ^ { n } \left[ l _ { k , i } , u _ { k , i } \right] \mathrm { \bf ~ T H E N } P _ { k } ( \mathrm { \bf x } ) = \mathrm { \bf w } _ { k } ^ { \top } \mathrm { \bf x } ^ { \prime } ,\tag{2}
$$

where the antecedent $C _ { k }$ is an �-dimensional hyperrectangle and the consequent $P _ { k }$ is a linear model with weight vector $\mathbf { w } _ { k } ~ \in ~ \mathbb { R } ^ { n + 1 }$ and the augmented input vector $\begin{array} { r l } { \mathbf { x } ^ { \prime } } & { { } = } \end{array}$ $( 1 , x _ { 1 } , . . . , x _ { n } ) ^ { \top } [ 1 7 ]$ . Alternative antecedent representations, including hyperellipsoids, are discussed in Section III-A. Each rule $c l _ { k }$ maintains seven bookkeeping parameters: (i) a fitness $F _ { k } \in ( 0 , 1 ]$ , which represents the relative usefulness of the rule; (ii) a prediction error $\epsilon _ { k } \ \geq \ 0 .$ , which measures the absolute prediction error of the rule; (iii) an accuracy $\kappa _ { k } ~ \in ~ ( 0 , 1 ]$ which is computed from $\epsilon _ { k }$ and quantifies the rule accuracy; (iv) a match set size estimate $\mathrm { m s } _ { k } \geq 1$ , which estimates the average size of the match set in which the rule appears; (v) an experience $\exp _ { k } \in \mathbb { N } _ { 0 } .$ , which counts how many updates the rule has received; (vi) a numerosity $\mathrm { n u m } _ { k } \in \mathbb { N } ,$ , which records how many copies of the rule are represented by this rule; and (vii) a time stamp $\mathrm { t s } _ { k } \in \mathbb { N } _ { 0 }$ , which records when a steady-state genetic algorithm (GA) was last applied to the rule [29], [30].

2) Algorithm: XCSF processes one training sample $\left( \mathbf { x } _ { i } , y _ { i } \right)$ per iteration. First, it forms the match set $\mathcal { M } = \{ c l _ { k } \in \mathcal { P }$ | ${ \bf x } _ { i } \in C _ { k } \} . \mathrm { ~ } ^ { 3 }$ If M is empty, a covering operator generates a new rule whose hyperrectangular antecedent is centered near $\mathbf { X } _ { i }$ with random width controlled by a half-width parameter $r _ { 0 } ;$ each dimension is independently set to the full domain $[ 0 , 1 ]$ with probability $P _ { \# }$ [31].

Each rule $c l _ { k } \in { \mathcal { M } }$ then computes its prediction $\hat { y } _ { k } = \mathbf { w } _ { k } ^ { \top } \mathbf { x } ^ { \prime }$ and updates its weight vector using a gradient-based method (e.g., the Widrow-Hoff delta rule [32], Adam [33]), based on the rule-specific error $y _ { i } - { \hat { y } } _ { k }$ . We use Adam in XCSF to enable a fair comparison with KACS, which also uses Adam for weight updates. The prediction error $\epsilon _ { k }$ is updated as a running mean of $| y _ { i } \mathrm { ~ - ~ } \hat { y } _ { k } | .$ , and the accuracy $\kappa _ { k }$ is computed from $\epsilon _ { k }$ relative to a threshold $\epsilon _ { 0 }$ . The fitness $F _ { k }$ is updated within M in proportion to accuracy and numerosity, so accurate rules accumulate high fitness over time [29]. The system output is the fitness-weighted average ${ \hat { f } } ( \mathbf { x } _ { i } ) \mathbf { \alpha } = { \hat { \mathbf { \alpha } } }$ $\textstyle \sum _ { k \in { \mathcal { M } } } P _ { k } ( \mathbf { x } _ { i } ) F _ { k } / \sum _ { k \in { \mathcal { M } } } F _ { k } [ 1 7 ]$

The GA is applied to M when the average time since the last GA application across rules in M exceeds $\theta _ { \mathrm { G A } }$ [30]. Two parents (i.e., two existing rules in M) are chosen by tournament selection; two offspring (i.e., two new rules) are created by uniform antecedent crossover and bound mutation. The created offspring are added to the current population $\mathcal { P }$ unless a parent subsumes them: a parent subsumes an offspring when it is both more general (antecedent contains the offspring’s) and sufficiently accurate and experienced [34]. When the total numerosity exceeds the population size limit �, rules with low fitness-adjusted deletion votes are deleted by roulette-wheel selection [29], [30].

## B. Kolmogorov-Arnold Representation Theorem

The Kolmogorov-Arnold (KA) representation theorem [24], [25] gives a fundamental result about the structure of multivariate continuous functions.

Theorem 1 (Kolmogorov-Arnold Representation Theorem). Let $n \geq 2$ and $\mathcal { K } = [ 0 , 1 ] ^ { n }$ . For any continuous function $f \in$ �(K), there exist continuous one-dimensional functions $\psi _ { q , p } :$ $[ 0 , 1 ] \to { \mathbb { R } }$ and $\Phi _ { q } : \mathbb { R }  \mathbb { R }$ such that

$$
f ( \mathbf { x } ) = \sum _ { q = 1 } ^ { 2 n + 1 } \Phi _ { q } \left( \sum _ { p = 1 } ^ { n } \psi _ { q , p } ( x _ { p } ) \right) ,\tag{1}
$$

where $\psi _ { q , p }$ are called inner functions and $\Phi _ { q }$ are called outer functions.

Although Theorem 1 formally defines each outer function as $\Phi _ { q } : \mathbb { R }  \mathbb { R }$ , in practice $\Phi _ { q }$ is evaluated only at values taken by $z _ { q } ( \mathbf { x } ) = \textstyle \sum _ { p = 1 } ^ { n } \psi _ { q , p } ( x _ { p } )$ as x ranges over K. Hence, $\Phi _ { q }$ is only required on $\mathcal { K } _ { z } ^ { ( q ) } : = z _ { q } ( \mathcal { K } ) \subseteq \mathbb { R }$ . This image is a compact interval: it is compact because the continuous image of a compact set is compact [35], and it is an interval because the continuous image of the connected set K is connected [35], and every compact connected subset of R is a closed bounded interval [35].

Theorem 1 states that any �-dimensional continuous function can be represented exactly using only one-dimensional functions. Specifically, it requires $n ( 2 n + 1 )$ inner functions and $2 n + 1$ outer functions, giving $2 n ^ { 2 } + 3 n + 1$ one-dimensional functions in total. This is quadratic in $n ,$ a much more favorable scaling than the exponential growth that appears in direct �-dimensional partitioning [19].

Two remarks are worth noting for the use of this theorem in KACS. First, the domain of $f$ can always be mapped to $[ 0 , 1 ] ^ { n }$ by a linear rescaling without loss of generality. Second, the theorem guarantees the existence of the representation, but does not specify a unique set of $\psi _ { q , p }$ and $\Phi _ { q }$ [26]; in practice, these functions need to be learned from data [36].

## III. RELATED WORK

## A. Multi-Dimensional Scalability in LCSs

The multi-dimensional scalability problem in LCSs, including XCSF [17], is well known. To approximate a target function with � effective subdivisions per variable in � dimensions, XCSF needs to maintain $O ( m ^ { n } )$ rules in the worst case [19]. Several directions have been explored to ease this burden.

A first direction focuses on the antecedent representation (for a recent comprehensive review, cf. [37]). The standard hyperrectangular representation [38], [39] is axis-aligned, which forces the system to use many small rules when the true subproblem boundaries are tilted or curved [40]. Replacing hyperrectangles with hyperellipsoids via kernel functions reduces the rule count for curved boundaries [41]. Neural network-based conditions can represent even more flexible boundaries but raise the cost of rule evaluation and evolutionary search [42]. Code-fragment-based conditions, inspired by genetic programming, handle multi-dimensional problems where only a few variables are relevant [43], thereby yielding shorter and more interpretable conditions. Such representations are practically effective when real-world data have low effective dimensionality. The worst case instead corresponds to a target with strong joint dependence on all variables and fine-scale variation throughout the input space; for such problems, achieving a fixed per-variable resolution can still require exponentially many local models.

A second direction applies dimensionality reduction before passing the input to an LCS. Behdad et al. [44] showed that combining an LCS with principal component analysis (PCA) substantially reduces the required population size while maintaining accuracy on multi-dimensional classification tasks. Several approaches [45]–[47] have also been proposed to reduce the dimensionality of multi-dimensional inputs using autoencoders. A shared limitation is that preserving high accuracy in the compressed representation is difficult, and any compression error can propagate into an LCS.

A third direction uses feature selection to reduce the effective input dimension before or during learning. Urbanowicz and Moore [48] introduced ExSTraCS 2.0, which improves scalability by incorporating expert knowledge scores based on TuRF [49] into the covering and mutation operators so that rules are biased toward more relevant features. ExSTraCS 2.0 was motivated in part by biomedical classification settings, where a multi-dimensional input may contain many irrelevant genetic attributes with only a small subset being predictive. This approach works well when many input variables are genuinely irrelevant to the target output. However, in function approximation, variables may interact and contribute jointly to the target output [50]. In such cases, feature selection cannot reduce the exponential rule count increase [19], [51], and the multi-dimensional scalability problem remains.

Unlike these approaches, KACS reorganizes rules according to the KA theorem (i.e., Theorem 1), reducing the worst-case rule count from $O ( m ^ { n } )$ to $O ( m n ^ { 2 } )$ without preprocessing.

## B. Kolmogorov-Arnold Networks and Variants

A representative recent use of the KA theorem in machine learning is the Kolmogorov-Arnold Network (KAN) [36]. KAN brings the KA theorem into the neural network paradigm by replacing the fixed activation functions of a multi-layer perceptron (MLP) [27] with learnable univariate B-spline functions on the edges (i.e., connections between neurons).

Since then, several other KA-inspired variants have been proposed. Wav-KAN [52] replaces B-spline activation functions with wavelets to enable efficient capture of both highfrequency and low-frequency components through multiresolution analysis. Ensemble-KAN [53] builds multiple KANs, each using a different subset of input features, and aggregates their predictions. Chebyshev KAN [54] uses Chebyshev polynomials for improved parameter efficiency. Temporal-KAN [55] and Federated-KAN [56] extend the framework to sequential and federated learning settings, respectively.

Within the LCS community, X-KAN [14] exploits the divide-and-conquer nature of LCSs to optimize multiple local KAN models through evolutionary search. By placing a KAN model in the consequent of each �-dimensional XCSF rule, X-KAN achieves effective approximation of discontinuous functions and functions with high local complexity. These two types of functions are difficult for a single global KAN.

As explained in the brief literature review above, KAN and its variants apply the KA theorem to neural architecture design, and X-KAN uses it to design local models in rule-consequents. Our KACS in this article takes a different approach: it first applies the KA theorem to the rule organization itself within a rule-learning framework. That is, our approach in KACS and the above-mentioned approaches in existing studies operate at structurally different levels of KA application (i.e., neural architecture, rule-consequents, and rule organization).

## C. Universal Approximation in Rule-Based Systems

Universal approximation has been studied across a range of model classes. For neural networks, Cybenko [27] and Hornik et al. [57] showed that a single hidden layer of sigmoidal units is sufficient to approximate any continuous function on a compact set. Wang and Mendel [58] proved an analogous result for fuzzy systems using the Stone-Weierstrass theorem [59]. Ying [60] extended this to a broader class of fuzzy inference systems covering product, minimum, and other common tnorms. Kosko [61] gave a separate proof for additive fuzzy systems, and Castro [62] further showed that fuzzy logic controllers with arbitrary membership functions and a broad class of inference engines are universal approximators.

A practically important question is how the required number of rules scales with dimension �. Standard fuzzy systems with a uniform grid still need $O ( m ^ { n } )$ rules, the same exponential cost as XCSF [19], [51]. Wang [63] showed that a hierarchical fuzzy system (HFS), which consists of several low-dimensional fuzzy systems connected in sequence, is a universal approximator whose rule count grows only linearly with �. This result is important because it shows that the exponential rule count is not an essential requirement for universal approximation.

KACS reaches a similar conclusion via the KA decomposition rather than a manually designed hierarchy, such as HFS; it achieves $O ( m n ^ { 2 } )$ rules in a mathematically grounded two-layer structure. Furthermore, to our knowledge, no prior LCS has been proved to be a universal approximator. In contrast, this article provides the first constructive universal approximation proof for an LCS, namely KACS.

## IV. KOLMOGOROV-ARNOLD CLASSIFIER SYSTEM

KACS has two main properties. First, it organizes the rule population into one-dimensional inner and outer submodels according to the KA structure. Second, it learns rule consequents through system-level backpropagation across the KA structure. The following subsections detail these characteristics.

## A. Overview

Fig. 1 schematically illustrates XCSF and KACS. KACS is a Michigan-style LCS that inherits the structural framework of XCSF, including the covering mechanism, fitness-based prediction aggregation, GA, subsumption, and hyperparameter settings. By organizing rules via the KA decomposition (Theorem 1), KACS not only suppresses the exponential growth in the number of rules with dimension �, but also reduces the difficulty of optimizing consequents by learning one-dimensional local models with only two parameters per rule, independent of �. Like XCSF, KACS optimizes its rules online, updating and evolving them when a new sample arrives.

As visualized in Fig. 1, the KA decomposition in (1) reduces learning � to learning �(2�+1) inner functions $\{ \psi _ { q , p } : [ 0 , 1 ] \to$ R} and $( 2 n + 1 )$ outer functions $\{ \Phi _ { q } : \mathcal { K } _ { z } ^ { ( q ) }  \mathbb { R } \}$ , all onedimensional. KACS maintains a dedicated ruleset, called a submodel, for each one-dimensional function. All submodels live in a single population ${ \mathcal { P } } ,$ partitioned into:

$n ( 2 n + 1 )$ inner submodels $\mathcal { P } _ { q , p } ^ { \psi }$ , each approximating $\psi _ { q , p }$ using rules over the input domain [0, 1];

$( 2 n + 1 )$ outer submodels $\mathcal { P } _ { q } ^ { \Phi }$ , each approximating $\Phi _ { q }$ using rules over the intermediate domain $\mathcal { K } _ { z } ^ { \left( q \right) }$

![](images/9c8eb739462fd1200237b4f222e839497d9b01c972c92a58df43ce2429c3bbc4.jpg)  
Fig. 1. Conceptual comparison of XCSF and KACS for � = 2 (2� + 1 = 5 channels). The left panel illustrates 2-D rule regions used by XCSF, where each rule covers a rectangle in the $( x _ { 1 } , x _ { 2 } )$ plane and maps x directly to ˆ�, yielding $O ( m ^ { n } )$ rules with � + 1 consequent parameters per rule. The right panel illustrates the KACS architecture, where prediction is decomposed into one-dimensional inner submodels $\psi _ { q , p } ( x _ { P } )$ , intermediate sums $\hat { z } _ { q } ,$ and one-dimensional outer submodels $\Phi _ { q } ( \hat { z } _ { q } )$ , which are finally summed to obtain ˆ�, yielding $O ( m n ^ { 2 } )$ rules with two consequent parameters per rule.

Given an input x, KACS computes ˆ� through a two-stage feedforward pass. It then backpropagates the system-level loss through this structure to update the weights of all active rules jointly. This is a key departure from XCSF, which updates each rule independently using its own prediction error.

Although each inner submodel processes one variable, KACS captures epistasis through the potentially nonlinear outer function $\Phi _ { q }$ applied to $\begin{array} { r } { z _ { q } = \sum _ { p } \psi _ { q , p } ( x _ { p } ) } \end{array}$ , which couples the dimensions. This is not bi-level optimization: active inner and outer consequents are jointly updated from the systemlevel loss, allowing gradients through $\Phi _ { q }$ to coordinate the inner submodels. Theorem 2 guarantees representation of continuous interactions in principle, although finite capacity and imperfect optimization may limit their learning. Algorithm 1 summarizes one complete training iteration of KACS; the following subsections describe each component in detail.

## B. Rule Representation and Population

Every rule $c l _ { k } \in \mathcal { S }$ , regardless of which submodel it belongs to, has the same one-dimensional format:

��<sub>�</sub> : IF � ∈ �<sub>�</sub> = [�<sub>�</sub>, �<sub>�</sub>] THEN �<sub>�</sub> (�) = �<sub>�,0</sub> + �<sub>�,1</sub> �, (3) where $C _ { k } \subset \mathbb { R }$ is a compact interval and $\mathbf { w } _ { k } = ( \boldsymbol { w } _ { k , 0 } , \boldsymbol { w } _ { k , 1 } ) ^ { \top }$ ∈ $\mathbb { R } ^ { 2 }$ . For inner rules $c l _ { k } \in \mathcal { P } _ { q , p } ^ { \psi } , z = x _ { \mathcal { p } } \in [ 0 , 1 ]$ . For outer rules $c l _ { k } \in \mathcal { P } _ { q } ^ { \Phi } , z = \hat { z } _ { q } \in \mathcal { K } _ { z } ^ { ( q ) }$

A finite set of such local linear rules can represent a piecewise-linear function. This property is central to the universal approximation proof in Section V-B, but does not constrain KACS operation, permitting overlapping rules.

The weight vector always has exactly two elements regardless of $n ,$ in contrast to $n + 1$ elements per rule in XCSF. Each rule $c l _ { k }$ maintains the same seven bookkeeping parameters as in XCSF: fitness $F _ { k }$ , prediction error $\epsilon _ { k }$ , accuracy $\kappa _ { k }$ , match set size estimate ms<sub>�</sub>, experience $\exp _ { k }$ , numerosity num<sub>�</sub>, and GA time stamp ts<sub>�</sub>.

A rule $c l _ { k }$ belongs to exactly one submodel, identified by its type (� or Φ) and index: $( q , p )$ for inner submodels (channel index � and input-dimension index $p )$ , or � for outer submodels (channel index). The population size limit � constrains the total numerosity, i.e., $\begin{array} { r } { \sum _ { c l _ { k } \in \mathcal { P } } \operatorname { n u m } _ { k } \le N } \end{array}$

## C. Feedforward Prediction and Covering

1) Stage 1: Inner Function Evaluation: For each pair $( q , p )$ where � is the channel index $( q \in \{ 1 , \ldots , 2 n + 1 \} )$ and $\boldsymbol { p }$ is the input-dimension index $( p \in \{ 1 , \ldots , n \} )$ , form the inner match set for a given input x:

$$
M _ { q , p } ^ { \psi } = \big \{ c l _ { k } \in \mathcal { P } _ { q , p } ^ { \psi } \mid x _ { p } \in C _ { k } \big \} .\tag{4}
$$

If $M _ { q , p } ^ { \psi } = \emptyset .$ , generate a covering rule according to (10), as described later in Section IV-C3. Compute the fitness-weighted prediction of $\psi _ { q , p } \mathrm { : }$

$$
\hat { \psi } _ { q , p } ( x _ { p } ) = \frac { \sum _ { c l _ { k } \in \mathcal { M } _ { q , p } ^ { \psi } } P _ { k } ( x _ { p } ) \cdot F _ { k } } { \sum _ { c l _ { k } \in \mathcal { M } _ { q , p } ^ { \psi } } F _ { k } } .\tag{5}
$$

Sum over $\boldsymbol { p }$ to obtain the intermediate value for channel $q \colon$

$$
\hat { z } _ { q } = \sum _ { p = 1 } ^ { n } \hat { \psi } _ { q , p } ( x _ { p } ) , \qquad q = 1 , \ldots , 2 n + 1 .\tag{6}
$$

2) Stage $2 { : }$ Outer Function Evaluation: For each channel index $q ,$ form the outer match set for the obtained intermediate value $\hat { z } _ { q } \mathrm { : \Omega }$

$$
\mathcal { M } _ { q } ^ { \Phi } = \bigl \{ c l _ { k } \in \mathcal { P } _ { q } ^ { \Phi } \ | \ \hat { z } _ { q } \in C _ { k } \bigr \} .\tag{7}
$$

```tcl
Algorithm 1 One training iteration of KACS
Require: Sample $( { \bf x } , y )$ , population P, iteration �
Ensure: Updated P
Step 1: Feedforward ⊲ Section IV-C
1: for $q = 1$ to 2� + 1 do
2: for $p = 1$ to � do
3: Form ${ \cal M } _ { q , p } ^ { \psi } \ ( 4 ) ;$ cover if empty (10)
4: Compute $\hat { \psi } _ { q , p } \mathopen { } \mathclose \bgroup \left( x _ { p } \aftergroup \egroup \right) \mathclose \bgroup \left( 5 \aftergroup \egroup \right)$
5: end for
6: Compute $\hat { z } _ { q }$ as $\begin{array} { r } { \hat { z } _ { q } \gets \sum _ { p } \hat { \psi } _ { q , p } ( x _ { p } ) } \end{array}$ (6)
7: Form $\mathcal { M } _ { q } ^ { \Phi }$ (7); cover if empty (11)
8: Compute $\hat { \Phi } _ { q } ( \hat { z } _ { q } ) \ ( 8 )$
9: end for
10: Compute ˆ� as $\begin{array} { r } { \hat { y } \gets \sum _ { q } \hat { \Phi } _ { q } ( \hat { z } _ { q } ) \ ( 9 ) } \end{array}$
Step 2: Backpropagation ⊲ Section IV-D
11: Form $M _ { \mathrm { a c t } }$ as $\begin{array} { r } { \hat { \mathcal { M } } _ { \mathrm { a c t } } \overset {  } { = } ( \bar { \bigcup } _ { q , p } \mathcal { M } _ { q , p } ^ { \psi } ) \cup ( \bigcup _ { q } \mathcal { M } _ { q } ^ { \Phi } ) } \end{array}$ (12)
12: for each ${ c l } _ { k } \in \mathcal { M } _ { \mathrm { a c t } }$ do
13: Compute g<sub>�</sub> via (13) or (14)
14: Update $\mathbf { w } _ { k }$ via Adam (15)–(16)
15: end for
Step 3: Parameter update ⊲ Section IV-E
16: for each ${ c l } _ { k } \in \mathcal { M } _ { \mathrm { a c t } }$ do
17: Update exp<sub>�</sub> as exp<sub>�</sub> ← exp<sub>�</sub> + 1
18: Update $\epsilon _ { k } , \kappa _ { k } , F _ { k } .$ , ms<sub>�</sub> within submodel M via (17)–(20)
19: end for
Step 4: GA and subsumption ⊲ Section IV-F
20: for each $\underline { { M } } \in \{ { \cal M } _ { q , p } ^ { \psi } \} \cup \{ { \cal M } _ { q } ^ { \bar { \Phi } } \}$ do
21: if $\begin{array} { r } { t - \sum _ { c l _ { k } \in { \mathcal { M } } } } \end{array}$ num $\boldsymbol { \mathrm { \ell } } _ { \boldsymbol { k } } \cdot \boldsymbol { \mathrm { t s } } _ { \boldsymbol { k } } / \sum _ { c l _ { \boldsymbol { k } } \in \mathcal { M } }$ num<sub>�</sub> $> \theta _ { \mathrm { G A } }$ then
22: Update ts<sub>�</sub> as $\mathrm { t s } _ { k } \gets \dot { t } \overline { { \mathrm { f o r } { } c l _ { k } } } \in \mathcal { M }$
23: Select $c l _ { p _ { 1 } } , c l _ { p _ { 2 } }$ by tournament
24: Copy to $c \bar { l } _ { o _ { 1 } } , c \bar { l } _ { o _ { 2 } } ;$ apply crossover and mutation
25: Initialize offspring parameters
26: Attempt subsumption by parents; add survivors to $\mathcal { P }$
27: while $\bar { \sum _ { k } }$ num<sub>�</sub> > � do
28: Delete by roulette-wheel on (21)
29: end while
30: end if
31: end for
```

If ${ \cal M } _ { q } ^ { \Phi } \ = \ 0$ , generate a covering rule according to (11), as described later in Section IV-C3. Compute the fitness-weighted prediction of $\Phi _ { q } .$

$$
\hat { \Phi } _ { q } ( \hat { z } _ { q } ) = \frac { \sum _ { c l _ { k } \in { \cal M } _ { q } ^ { \Phi } } P _ { k } ( \hat { z } _ { q } ) \cdot F _ { k } } { \sum _ { c l _ { k } \in { \cal M } _ { q } ^ { \Phi } } F _ { k } } .\tag{8}
$$

Sum over $q$ to obtain the final system prediction:

$$
\hat { y } = \hat { f } (  { \mathbf { x } } ) = \sum _ { q = 1 } ^ { 2 n + 1 } \hat { \Phi } _ { q } ( \hat { z } _ { q } ) .\tag{9}
$$

Stage 2 requires the intermediate values $\hat { z } _ { q }$ in (6). Thus, KACS first completes Stage 1 for all $( q , p )$ pairs and then proceeds to Stage 2. Within each stage, the submodels are independent and can be evaluated in any order (and also in parallel).<sup>4</sup>

3) Covering: When a submodel match set is empty, a new rule $c l _ { \mathrm { c o v } }$ is generated to ensure every input is always matched.

For inner submodel $\mathcal { P } _ { q , p } ^ { \psi }$ with uncovered $x _ { p } \in [ 0 , 1 ]$

$$
( l _ { \mathrm { c o v } } , \ u _ { \mathrm { c o v } } ) = \big ( \operatorname* { m a x } ( 0 , x _ { p } - \mathcal { U } ( 0 , r _ { 0 } ] ) , \ \operatorname* { m i n } ( 1 , x _ { p } + \mathcal { U } ( 0 , r _ { 0 } ] ) \big ) ,\tag{10}
$$

where $r _ { 0 } \in ( 0 , 1 ]$ is the maximum covering half-width. With probability $P _ { \# } \in [ 0 , 1 ]$ , a maximally general rule is generated instead, by setting $( l _ { \mathrm { c o v } } , u _ { \mathrm { c o v } } ) = ( 0 , 1 )$ as in XCSF.

For outer submodel $\mathcal { P } _ { q } ^ { \Phi }$ with uncovered $\hat { z } _ { q } \in \mathcal { K } _ { z } ^ { \left( q \right) } \subseteq \mathbb { R }$ , the maximally general option is inapplicable because $\mathcal { K } _ { z } ^ { \left( q \right) }$ is not known a priori. The antecedent is set to:

$$
\begin{array} { r } { \left( l _ { \mathrm { c o v } } , \mathrm { ~ } u _ { \mathrm { c o v } } \right) = \left( \widehat { z } _ { q } - \mathcal { U } ( 0 , r _ { 0 } ] , \widehat { z } _ { q } + \mathcal { U } ( 0 , r _ { 0 } ] \right) . } \end{array}\tag{11}
$$

In both cases, the new rule is initialized as follows: the consequent weight vector $\mathbf { w } _ { \mathrm { c o v } }$ is sampled i.i.d. from $\mathcal { U } [ - 1 , 1 ]$ for each element, $\epsilon _ { \mathrm { c o v } } = 0 , F _ { \mathrm { c o v } } = 0 . 0 1 , \mathrm { m s _ { \mathrm { c o v } } = 1 , \exp _ { \mathrm { c o v } } = 0 } .$ $\mathrm { \ n u m } _ { \mathrm { c o v } } = 1$ , with its time stamp $\mathrm { \ t s \mathrm { _ { c o v } } }$ set to the current iteration �, and is added to both its submodel and $\mathcal { P } _ { \mathcal { S } }$

## D. Consequent Update by Backpropagation

After computing ˆ�, KACS updates the weight vectors of all active rules by backpropagating the system-level loss through (9). Define the loss as $\begin{array} { r } { \mathcal { L } = \frac { 1 } { 2 } ( y - \hat { y } ) ^ { 2 } } \end{array}$ and the set of all active rules as

$$
M _ { \mathrm { a c t } } = \left( \bigcup _ { q , p } { M _ { q , p } ^ { \psi } } \right) \cup \left( \bigcup _ { q } { M _ { q } ^ { \Phi } } \right) .\tag{12}
$$

1) Gradient for Outer Rules: For $d _ { k } \in \mathcal { M } _ { q } ^ { \Phi }$ , differentiating L through (9) and (8) gives:

$$
\begin{array} { r l r } & { } & { \frac { \partial \mathcal { L } } { \partial \mathbf { w } _ { k } } = \frac { \partial \mathcal { L } } { \partial \hat { y } } \cdot \frac { \partial \hat { y } } { \partial \hat { \Phi } _ { q } ( \hat { z } _ { q } ) } \cdot \frac { \partial \hat { \Phi } _ { q } ( \hat { z } _ { q } ) } { \partial P _ { k } ( \hat { z } _ { q } ) } \cdot \frac { \partial P _ { k } ( \hat { z } _ { q } ) } { \partial \mathbf { w } _ { k } } } \\ & { } & { = - ( y - \hat { y } ) \cdot \frac { F _ { k } } { \sum _ { c l _ { k ^ { \prime } } \in \mathcal { M } _ { q } ^ { \Phi } } F _ { k ^ { \prime } } } \cdot \hat { \mathbf { z } } _ { q } ^ { \prime } , ~ } \end{array}\tag{13}
$$

where $\hat { \mathbf { z } } _ { q } ^ { \prime } = ( 1 , \hat { z } _ { q } ) ^ { \intercal }$

2) Gradient for Inner Rules: For $\boldsymbol { c l } _ { k } \in \boldsymbol { \mathcal { M } } _ { q , p } ^ { \psi } ,$ the gradient must pass through the outer stage because $\hat { z } _ { q }$ enters (8). Applying the chain rule through $( 9 ) { \longrightarrow } ( 8 ) { \longrightarrow } ~ ( 6 ) { \stackrel { - } { \to } } ( 5 ) \colon$

$$
\begin{array} { r l r } {  { \frac { \partial \mathcal { L } } { \partial \mathbf { w } _ { k } } = \frac { \partial \mathcal { L } } { \partial \hat { y } } \cdot \frac { \partial \hat { y } } { \partial \hat { \Phi } _ { q } ( \hat { z } _ { q } ) } \cdot \frac { \partial \hat { \Phi } _ { q } ( \hat { z } _ { q } ) } { \partial \hat { z } _ { q } } } } \\ & { } & { \cdot \frac { \partial \hat { z } _ { q } } { \partial \hat { \psi } _ { q , p } ( x _ { p } ) } \cdot \frac { \partial \hat { \psi } _ { q , p } ( x _ { p } ) } { \partial P _ { k } ( x _ { p } ) } \cdot \frac { \partial P _ { k } ( x _ { p } ) } { \partial \mathbf { w } _ { k } } } \\ & { } & { = - ( y - \hat { y } ) \cdot \frac { \sum _ { c l _ { j } \in M _ { q } ^ { \oplus } } w _ { j , 1 } \cdot F _ { j } } { \sum _ { c l _ { j } \in M _ { q } ^ { \oplus } } F _ { j } } \cdot \frac { F _ { k } } { \sum _ { c l _ { k ^ { \prime } } \in M _ { q , p } ^ { \ell } } F _ { k ^ { \prime } } } \cdot \mathbf { x } _ { p } ^ { \prime } , } \end{array}\tag{14}
$$

where $\mathbf { x } _ { p } ^ { \prime } = ( 1 , x _ { p } ) ^ { \top }$

3) Adam Update: Each rule $c l _ { k } \in \mathcal { M } _ { \mathrm { a c t } }$ maintains its own first and second moment vectors $\mathbf { m } _ { k } , \mathbf { v } _ { k } \in \mathbb { R } ^ { 2 }$ , both initialized to 0 when the rule is first created. Here, $t _ { k }$ denotes a rulelocal Adam time step, defined as $t _ { k } = \exp _ { k } + 1$ at the moment of updating $c l _ { k }$ . We use +1 because newly created rules (by covering or the GA) have $\boldsymbol { \mathrm { e x p } } _ { k } = 0$ before their first gradient update. At iteration �, after computing $\mathbf { g } _ { k } = \partial \mathcal { L } / \partial \mathbf { w } _ { k }$

$$
\mathbf { m } _ { k } \gets \beta _ { 1 } \mathbf { m } _ { k } + \left( 1 - \beta _ { 1 } \right) \mathbf { g } _ { k } , \ : \mathbf { v } _ { k } \gets \beta _ { 2 } \mathbf { v } _ { k } + \left( 1 - \beta _ { 2 } \right) \mathbf { g } _ { k } ^ { 2 } ,\tag{15}
$$

$$
\mathbf { w } _ { k }  \mathbf { w } _ { k } - \eta \frac { \mathbf { m } _ { k } / ( 1 - \beta _ { 1 } ^ { t _ { k } } ) } { \sqrt { \mathbf { v } _ { k } / ( 1 - \beta _ { 2 } ^ { t _ { k } } ) } + \varepsilon _ { \mathrm { A d a m } } } ,\tag{16}
$$

where squaring and the square root in (15)–(16) are elementwise. We use the default values recommended in [33]: $\eta = 0 . 0 0 1 , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and $\varepsilon _ { \mathrm { A d a m } } = 1 0 ^ { - 8 }$

## E. Parameter Update

After the gradient update, the following bookkeeping parameters are updated for all $c l _ { k } \in \mathcal { M } _ { \mathrm { a c t } }$

First, the experience is updated as $\mathrm { e x p } _ { k } \gets \mathrm { e x p } _ { k } + 1$

Next, the prediction error is updated. Unlike XCSF, which uses each rule’s individual error $| y - \hat { y } _ { k } |$ , KACS uses the system-level error for all rules because each rule contributes to ˆ� as part of a larger structure, and its individual output is not directly comparable to �:

$$
\epsilon _ { k } \gets \epsilon _ { k } + \beta \big ( | y - \hat { y } | - \epsilon _ { k } \big ) ,\tag{17}
$$

with learning rate $\beta \in ( 0 , 1 ]$

Then, the accuracy is calculated as:

$$
\kappa _ { k } = \left\{ { 1 , \atop \alpha ( \epsilon _ { k } / \epsilon _ { 0 } ) ^ { - \nu } , } \right. \mathrm { i f } \epsilon _ { k } < \epsilon _ { 0 } ,\tag{18}
$$

with error threshold $\epsilon _ { 0 } > 0 .$ , accuracy fall-off rate $\alpha \in [ 0 , 1 ]$ accuracy exponent $\nu > 0 .$

After that, the fitness is updated within the submodel match set $\boldsymbol { \mathcal { M } } \in \{ \boldsymbol { \mathcal { M } } _ { q , p } ^ { \psi } \} \cup \{ \boldsymbol { \mathcal { M } } _ { q } ^ { \Phi } \}$ , not across all active rules. For $c l _ { k } \in { \mathcal { M } } ;$

$$
F _ { k } \gets F _ { k } + \beta \left( \frac { \kappa _ { k } \cdot \mathsf { n u m } _ { k } } { \sum _ { c l _ { k ^ { \prime } } \in \mathcal { M } } \kappa _ { k ^ { \prime } } \cdot \mathsf { n u m } _ { k ^ { \prime } } } - F _ { k } \right) .\tag{19}
$$

This per-submodel update ensures that the fitness of a rule reflects its accuracy relative to other rules competing to approximate the same component function.

Finally, the match set size estimate is updated as:

$$
\begin{array} { r } { \operatorname* { m s } _ { k } \gets \operatorname* { m s } _ { k } + \beta \left( \sum _ { c l _ { k ^ { \prime } } \in \mathcal { M } } \mathrm { n u m } _ { k ^ { \prime } } - \operatorname* { m s } _ { k } \right) . } \end{array}\tag{20}
$$

## F. Genetic Algorithm and Subsumption

For each active submodel match set $M .$ , the GA is activated when $\begin{array} { r } { t \mathrm { ~ - ~ } \sum _ { c l _ { k } \in { \mathcal { M } } } } \end{array}$ num<sub>�</sub> $\cdot \mathrm { \Delta t s } _ { k } / \sum _ { c l _ { k } \in \mathcal { M } }$ num<sub>�</sub> $> ~ \theta _ { \mathrm { G A } }$ . When triggered, the time stamp is updated as ${ \mathrm t s } _ { k } \gets t$ for all $c l _ { k } \in \mathcal { M }$ In the GA, two parents $c l _ { p _ { 1 } }$ and $c l _ { p _ { 2 } }$ are selected from M by tournament selection with tournament size �. Two offspring $c l _ { o _ { 1 } }$ and $c l _ { o _ { 2 } }$ are initialized as copies of $c l _ { p _ { 1 } }$ and $c l _ { p _ { 2 } }$ . With probability $\chi ,$ crossover is applied: $l _ { o _ { 1 } }$ and $l _ { o _ { 2 } }$ are swapped with probability 0.5, and the same is done for $u _ { o _ { 1 } }$ and $u _ { o _ { 2 } }$ If crossover yields an invalid interval (i.e., $l _ { o _ { i } } ~ > ~ u _ { o _ { i } }$ for $i \in \{ 1 , 2 \} )$ , then $l _ { o _ { i } }$ and ${ { u } _ { o } } _ { i }$ are swapped so that $l _ { o _ { i } } \leq u _ { o _ { i } }$ . With probability � per bound, mutation is applied: add ${ \mathcal { U } } [ - m _ { 0 } ,$ �<sub>0</sub>] to each of $l _ { o _ { i } }$ and ${ { u } _ { o } } _ { i }$ for $i ~ \in ~ \{ 1 , 2 \}$ , where $m _ { 0 } ~ > ~ 0$ is the mutation magnitude. For inner rules, both bounds are clipped to [0, 1]. For outer rules, no clipping is applied. After crossover and mutation, for $i \in \{ 1 , 2 \}$ , set $\epsilon _ { o _ { i } } \gets \frac { 1 } { 2 } ( \epsilon _ { { p } _ { 1 } } + \epsilon _ { { p } _ { 2 } } )$ $\begin{array} { r } { F _ { o _ { i } } \gets 0 . 1 \times \frac { 1 } { 2 } ( F _ { p _ { 1 } } + F _ { p _ { 2 } } ) , \exp _ { o _ { i } } = 0 } \end{array}$ , and num $\boldsymbol { 1 } _ { o _ { i } } = 1$

Before adding an offspring $c l _ { o } ~ \in ~ \{ c l _ { o _ { 1 } } , c l _ { o _ { 2 } } \}$ to $\mathcal { P }$ , each parent $c l _ { p }$ attempts to subsume it. Subsumption occurs when all the following three conditions are satisfied: (i) $[ l _ { o } , u _ { o } ] \subseteq$ $[ l _ { \dot { p } } , u _ { \dot { p } } ] ; \ ( \mathrm { i i } ) \ \epsilon _ { \dot { p } } \ < \ \epsilon _ { 0 } ;$ and (iii) $\exp _ { p } \ > \ \theta _ { \mathrm { s u b } }$ . If subsumed, num $\mathsf { l } _ { p } \gets \mathsf { n u m } _ { p } + \mathsf { n u m } _ { o }$ and $c l _ { o }$ is discarded.

After inserting surviving offspring, while $\textstyle \sum _ { c l _ { k } \in { \mathcal { P } } }$ num<sub>�</sub> $>$ �, one rule is deleted by roulette-wheel selection with vote

$$
d _ { k } = \left\{ \begin{array} { l l } { \operatorname* { m s } _ { k } \cdot \mathrm { n u m } _ { k } \cdot \bar { F } / F _ { k } , } & { \exp _ { k } > \theta _ { \mathrm { d e l } } \mathrm { ~ a n d ~ } F _ { k } < \delta \bar { F } , } \\ { \operatorname* { m s } _ { k } \cdot \mathrm { n u m } _ { k } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{21}
$$

where $\begin{array} { r } { \bar { F } = \sum _ { k \in \mathcal { P } } F _ { k } / N , \theta _ { \mathrm { d e l } } > 0 } \end{array}$ is the deletion threshold, and $\delta \in ( 0 , 1 ]$ . The numerosity of the selected rule is decreased by one; any rule with $\boldsymbol { \mathrm { n u m } } _ { k } = 0$ is removed from $\mathcal { P }$

## V. UNIVERSAL APPROXIMATION THEOREM OF KACS

## A. Assumptions and Main Theorem

The proof in this section concerns the expressive power of KACS, not the behavior of its learning algorithm. The following two assumptions make this scope precise.

Assumption 1 (Target function). The target function � is a real-valued continuous function defined on the compact set $\mathcal { K } = [ 0 , 1 ] ^ { n } , \ : i . e . , f \in C ( \mathcal { K } )$

Assumption 2 (Existence (not learnability)). This assumption concerns representational existence rather than algorithmic learnability. The proof is an existence theorem for the expressive power of KACS. We assume that KACS can represent and store any rule $c l _ { k }$ with any compact interval $C _ { k } = [ l _ { k } , u _ { k } ]$ as its antecedent, any linear function $P _ { k } ( z ) = w _ { k , 0 } + w _ { k , 1 } z \ a s$ its consequent, and any fitness value $F _ { k } \in ( 0 , 1 ]$ . This is the same standard assumption adopted in universal approximation proofs for neural networks [27] and fuzzy systems [60], where arbitrary weights or membership functions are assumed to be expressible. It does not guarantee that the online updates and GA find such rules.

Under these assumptions, the main theorem is stated as follows.

Theorem 2 (Universal Approximation of KACS). Let $\mathcal { F } _ { \mathrm { K A C S } }$ be the set of all functions <sup>ˆ</sup>� expressible by KACS models on $\mathcal { K } = [ 0 , 1 ] ^ { n }$ as defined in Section IV. Then F is dense in �(K). That is, for any $f \in C ( { \mathcal { K } } )$ and any $\varepsilon > 0 ,$ , there exists $\hat { f } \in \mathcal { F } _ { \mathrm { K A C S } }$ such that

$$
\underset { { \mathbf { x } } \in \mathcal { K } } { \operatorname* { s u p } } | f ( \mathbf { x } ) - \hat { f } ( \mathbf { x } ) | < \varepsilon .\tag{22}
$$

The proof uses the following two-step strategy.

1) Step 1 (Section V-B): We prove that a KACS ruleset operating on a one-dimensional compact interval, referred to as a 1D KACS submodel, is itself a universal approximator for $C ( [ a , b ] )$ ).

2) Step 2 (Section V-C): We apply the KA representation theorem (Theorem 1) to extend the one-dimensional result to � dimensions, thereby proving Theorem 2.

## B. Step 1: 1D KACS Submodel as a Universal Approximator

A 1D KACS submodel $\hat { g }$ is a KACS ruleset defined on a one-dimensional compact interval ${ \cal T } = [ a , b ] \subset \mathbb { R }$ . Each rule $c l _ { k } \in \mathcal { S }$ has the form given in (3), and the output for a given $z \in \mathcal { I }$ is

$$
\hat { g } ( z ) = \frac { \sum _ { c l _ { k } \in { \cal M } ( z ) } P _ { k } ( z ) \cdot F _ { k } } { \sum _ { c l _ { k } \in { \cal M } ( z ) } F _ { k } } , \quad { \cal M } ( z ) = \{ c l _ { k } \in \mathcal { P } \mid z \in C _ { k } \} ,\tag{23}
$$

which matches (5) and (8) in the one-dimensional case. Let $ { G _ { \mathrm { K A C S } } } ^ { \mathrm { 1 D } }$ denote the set of all functions expressible by 1D KACS submodels on I.

Theorem 3 (1D Universal Approximation). $ { G _ { \mathrm { K A C S } } } ^ { \mathrm { 1 D } }$ is dense in $C ( { \cal { T } } )$ . That is, for any $g \in C ( \mathcal { T } )$ and any $\varepsilon > 0 ,$ , there exists $\hat { g } \in \mathcal { G } _ { \mathrm { K A C S } } ^ { \mathrm { 1 D } }$ such that $\begin{array} { r } { \operatorname* { s u p } _ { z \in \mathcal { T } } | g ( z ) - \hat { g } ( z ) | < \varepsilon . } \end{array}$

As noted in Section IV-B, the proof uses piecewise-linear approximation as a bridge from local linear rules to arbitrary continuous functions.

Theorem 4 (Piecewise Linear Approximation [35], [64]). For any $g ~ \in ~ C ( \mathcal { I } )$ and any $\varepsilon \ > \ 0 ,$ there exists a continuous piecewise linear function $h \in \mathsf { C P L } ( \mathcal { I } )$ such that $\begin{array} { r } { \operatorname* { s u p } _ { z \in \mathcal { I } } | g ( z ) - h ( z ) | < \varepsilon , } \end{array}$ , where $\operatorname { C P L } ( \boldsymbol { \mathcal { T } } )$ is the set ofcontinuous functions on I that are linear on each subinterval of a finite partition of I.

Theorem 4 reduces the proof of Theorem 3 to the following lemma.

Lemma 1. For any $h \in \mathrm { C P L } ( \mathcal { I } )$ and any $\varepsilon > 0 ,$ , there exists $\hat { g } \in \mathcal { G } _ { \mathrm { K A C S } } ^ { \mathrm { 1 D } }$ such that $\begin{array} { r } { \operatorname* { s u p } _ { z \in \mathcal { T } } | h ( z ) - \hat { g } ( z ) | < \varepsilon / 2 . } \end{array}$

Proof. Since $h \in \mathrm { C P L } ( \mathcal { I } )$ , there is a finite partition $a = z _ { 0 } <$ $z _ { 1 } < \dots < z _ { M } = b$ of I into � subintervals $R _ { j } = [ z _ { j - 1 } , z _ { j } ]$ such that ℎ restricts to a linear function $h _ { j } ( z ) = a _ { j } z + b _ { j }$ on each $R _ { j }$

For each subinterval $R _ { j } \ ( j = 1 , \ldots , M )$ , we place a reference rule $c l _ { j } ^ { * }$ in $\mathcal { P }$ with antecedent $C _ { j } ^ { * } ~ = ~ R _ { j }$ and consequent $P _ { i } ^ { * } ( z ) ~ \stackrel { } { = } ~ h _ { j } ( z )$ . Since $h _ { j }$ is linear, this satisfies (3) under Assumption 2. The ruleset $\mathcal { P }$ may additionally contain any finite number of rules $\{ c l _ { k } \} _ { k \neq j }$ with arbitrary consequents, provided their antecedents are compact intervals and their fitnesses satisfy $F _ { k } \in ( 0 , 1 ]$

Fix any $z \in \mathcal { I }$ . Since $z \in R _ { j }$ for some $j ,$ the reference rule $c l _ { j } ^ { * }$ is guaranteed to match. If � lies on a breakpoint $z _ { j } ,$ , continuity of ℎ gives $h _ { j } ( z _ { j } ) = h _ { j + 1 } ( z _ { j } )$ , so either reference rule yields the same value and the argument is unaffected. Applying the triangle inequality:

$$
| h ( z ) - \hat { g } ( z ) | \leq \underbrace { | h _ { j } ( z ) - P _ { j } ^ { * } ( z ) | } _ { \mathrm { ( A ) } } + \underbrace { | P _ { j } ^ { * } ( z ) - \hat { g } ( z ) | } _ { \mathrm { ( B ) } } .\tag{24}
$$

Here, (A) denotes the reference rule error, which measures the local approximation error of the selected reference rule $c l _ { j } ^ { * }$ on the interval $R _ { j } \colon ( \mathbf { B } )$ denotes the aggregated error, which captures the additional error induced by the fitness-weighted aggregation with the other matching rules $\{ c l _ { k } \}$ . We show that both (A) and (B) can be made less than $\varepsilon / 4$ , so their sum is less than $\varepsilon / 2$

(A) Reference Rule Error. By construction $P _ { j } ^ { * } ( z ) = h _ { j } ( z )$ so

$$
( \mathrm { A } ) = | h _ { j } ( z ) - P _ { j } ^ { * } ( z ) | = 0 < \frac { \varepsilon } { 4 } .\tag{25}
$$

(B) Aggregated Error. Substituting (23) and separating the reference rule $c l _ { j } ^ { * } .$

$$
\begin{array} { r l r } {  { \big \vert P _ { j } ^ { * } ( z ) - \hat { g } ( z ) \big \vert =  \frac { \sum _ { c l _ { k } \in \mathcal { M } ( z ) } \big ( P _ { j } ^ { * } ( z ) - P _ { k } ( z ) \big ) \cdot F _ { k } } { \sum _ { c l _ { k } \in \mathcal { M } ( z ) } F _ { k } }  } } \\ & { } & { \leq \frac { \sum _ { c l _ { k } \in \mathcal { M } ( z ) \backslash \{ c l _ { j } ^ { * } \} } \big \vert P _ { j } ^ { * } ( z ) - P _ { k } ( z ) \vert \cdot F _ { k } } { F _ { j } ^ { * } + \sum _ { c l _ { k } \in \mathcal { M } ( z ) \backslash \{ c l _ { j } ^ { * } \} } F _ { k } } . } \end{array}\tag{26}
$$

Since all consequents are linear on the compact set ${ \boldsymbol { \mathit { I } } } ,$ the quantity $\begin{array} { r } { \Lambda = \operatorname* { m a x } _ { 1 \leq j \leq M } \operatorname* { m a x } _ { 1 \leq k \leq | \mathcal { P } | } \operatorname* { s u p } _ { z \in { \cal T } } | P _ { i } ^ { * } ( z ) - P _ { k } ( z ) | } \end{array}$ is finite. Set $F _ { i } ^ { * } = F _ { \mathrm { r e f } } \in ( 0 , 1 ]$ and $F _ { k } = \delta \in ( 0 , \dot { 1 } ]$ for all $k \neq j$ Since $| \mathcal { M } ( z ) \setminus \{ c l _ { j } ^ { * } \} | \leq | \mathcal { P } | - 1$ , (26) gives

$$
| P _ { j } ^ { * } ( z ) - \hat { g } ( z ) | \leq \Lambda \cdot \frac { ( | \mathcal { P } | - 1 ) \delta } { F _ { \mathrm { r e f } } + ( | \mathcal { P } | - 1 ) \delta } = \sigma ( \gamma ) ,\tag{27}
$$

where $\begin{array} { r l r } { \gamma } & { { } = } & { \delta / F _ { \mathrm { r e f } } } \end{array}$ and $\begin{array} { r c l } { \sigma ( \gamma ) } & { = } & { \Lambda \ \cdot \ \frac { \ d ( | \mathcal { P } | - 1 ) \gamma } { \ d 1 + \ d ( | \mathcal { P } | - 1 ) \gamma } } \end{array}$ . Since $\begin{array} { r } { \operatorname* { l i m } _ { \gamma \to 0 ^ { + } } \sigma ( \gamma ) = 0 } \end{array}$ , the $\varepsilon { - } \delta$ definition of the limit guarantees that, for any $E \ > \ 0 ,$ , there exists $\Delta \ > \ 0$ such that $0 ~ <$ $\gamma < \Delta \Rightarrow \sigma ( \gamma ) < E .$ . In particular, setting $E = \varepsilon / 4$ yields $0 < \gamma < \Delta \Rightarrow \sigma ( \gamma ) < \varepsilon / 4$ . Choosing $\delta > 0$ sufficiently small and $F _ { \mathrm { r e f } }$ sufficiently close to 1 so that $\gamma = \delta / F _ { \mathrm { r e f } } < \Delta$ yields

$$
( \mathbf { B } ) = | P _ { j } ^ { * } ( z ) - \hat { g } ( z ) | \leq \sigma ( \gamma ) < \frac { \varepsilon } { 4 } .\tag{28}
$$

Combining (A) and (B). Substituting (25) and (28) into (24) gives

$$
\operatorname* { s u p } _ { z \in \mathcal { T } } \left| h ( z ) - \hat { g } ( z ) \right| < \frac { \varepsilon } { 4 } + \frac { \varepsilon } { 4 } = \frac { \varepsilon } { 2 } .\tag{29}
$$

This completes the proof of Lemma 1.

Proof of Theorem 3. Let $g \in C ( \mathcal { T } )$ and $\varepsilon > 0$ . By Theorem 4 with $\varepsilon / 2$ , there exists $h \in \mathrm { C P L } ( \mathcal { I } )$ with $\begin{array} { r } { \operatorname* { u p } _ { z \in J } | g ( z ) - h ( z ) | < } \end{array}$ $\varepsilon / 2$ . By Lemma 1 with �, there exists $\hat { g } ~ \in ~ \mathcal { G } _ { \mathrm { K A C S } } ^ { \mathrm { 1 D } }$ with $\begin{array} { r } { \operatorname* { s u p } _ { z \in \mathcal { T } } | h ( z ) - \hat { g } ( z ) | < \varepsilon / 2 } \end{array}$ . The triangle inequality gives

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { z \in { \cal T } } | g ( z ) - \hat { g } ( z ) | \leq \operatorname* { s u p } _ { z \in { \cal T } } | g ( z ) - h ( z ) | + \operatorname* { s u p } _ { z \in { \cal T } } | h ( z ) - \hat { g } ( z ) | } \\ { \displaystyle ~ < \frac { \varepsilon } { 2 } + \frac { \varepsilon } { 2 } = \varepsilon . \qquad \quad ( } \end{array}\tag{30}
$$

This completes the proof of Theorem 3.

## C. Step 2: Extension to � Dimensions

Proof of Theorem 2. Let $f \in C ( { \mathcal { K } } )$ and $\varepsilon > 0$ . By Theorem 1, there exist continuous functions $\psi _ { q , p } : [ 0 , 1 ] \to \mathbb { R }$ and $\Phi _ { q } : \mathcal { K } _ { z } ^ { ( q ) } \ \to \ \mathbb { R }$ such that $f ( { \bf x } ) = \sum _ { q = 1 } ^ { 2 n + 1 } \Phi _ { q } ( z _ { q } )$ , where $\begin{array} { r } { z _ { q } = \sum _ { p = 1 } ^ { n } \psi _ { q , p } ( x _ { p } ) } \end{array}$ and $\begin{array} { r } { \mathcal { K } _ { z } ^ { ( q ) } = \big \{ \sum _ { p = 1 } ^ { n } \psi _ { q , p } \overset { \cdot } { ( } x _ { p } ) ~ | ~ \mathbf { x } \in \mathcal { K } \big \} \subset \mathbb { R } } \end{array}$ is a compact interval.   
Define $\begin{array} { r } { \hat { f } (  { \mathbf { x } } ) = \sum _ { q = 1 } ^ { 2 n + 1 } \hat { \Phi } _ { q } ( \hat { z } _ { q } ) } \end{array}$ , where $\begin{array} { r } { \hat { z } _ { q } = \sum _ { p } \hat { \psi } _ { q , p } ( x _ { p } ) } \end{array}$ . For all $\mathbf { x } \in \mathcal { K }$

$$
\begin{array} { r l } & { \displaystyle | f ( \mathbf x ) - \hat { f } ( \mathbf x ) | = \left| \sum _ { q = 1 } ^ { 2 n + 1 } \Phi _ { q } ( z _ { q } ) - \sum _ { q = 1 } ^ { 2 n + 1 } \hat { \Phi } _ { q } ( \hat { z } _ { q } ) \right| } \\ & { \qquad \le \displaystyle \sum _ { q = 1 } ^ { 2 n + 1 } \left[ \underbrace { | \Phi _ { q } ( z _ { q } ) - \Phi _ { q } ( \hat { z } _ { q } ) | } _ { \mathrm { ( C ) } } + \underbrace { | \Phi _ { q } ( \hat { z } _ { q } ) - \hat { \Phi } _ { q } ( \hat { z } _ { q } ) | } _ { \mathrm { ( D ) } } \right] . } \end{array}\tag{31}
$$

Here, (C) is the inner approximation error: the error in $\Phi _ { q } \mathrm { ^ { * } s }$ input caused by replacing the true intermediate value $z _ { q }$ with its approximation $\hat { z } _ { q } ; ( \mathrm { D } )$ is the outer approximation error: the error from approximating $\Phi _ { q }$ itself by the 1D KACS submodel $\hat { \Phi } _ { q } .$ . We show that both (C) and (D) are bounded by $\varepsilon / ( 2 ( 2 n +$ 1)) for each �, so the total is at most �.

(C) Inner Approximation Error. KACS computes intermediate values $\begin{array} { r } { z _ { q } = \sum _ { p } { \psi _ { q , p } ( x _ { p } ) } } \end{array}$ and feeds them into the outer function $\Phi _ { q }$ . Because each $\psi _ { q , p }$ is replaced by an approximation $\hat { \psi } _ { q , p } ^ { \phantom { \dagger } } ,$ , the resulting intermediate value $\begin{array} { r } { \hat { z } _ { q } = \sum _ { p } \hat { \psi } _ { q , p } ( x _ { p } ) } \end{array}$ may deviate from $z _ { q }$ and could fall outside the range $\mathcal { K } _ { z } ^ { \left( q \right) }$ on which $\Phi _ { q }$ was originally analyzed. Error (C) quantifies the impact of this deviation: $| \Phi _ { q } ( z _ { q } ) - \Phi _ { q } ( \hat { z } _ { q } ) |$ . We now bound this error.

Fix any distance $r > 0$ as the enlargement width around $\mathcal { K } _ { z } ^ { \left( q \right) }$ , and define the enlarged compact interval

$$
\tilde { \mathcal { K } } _ { z } ^ { ( q ) } = \big \{ t \in \mathbb { R } \ | \ \mathrm { d i s t } ( t , \mathcal { K } _ { z } ^ { ( q ) } ) \leq r \big \} ,\tag{32}
$$

where dist $\begin{array} { r } { ( t , S ) = \operatorname* { i n f } _ { s \in S } | t - s | } \end{array}$ denotes the distance from the point � to the set � [35], [64]. Concretely, if $\mathcal { K } _ { z } ^ { ( q ) } = [ a , b ]$ , then $\ddot { \mathcal { K } } _ { z } ^ { ( q ) } = [ a - r , b + r ]$ , which is again a compact interval [35]. This enlarged set acts as a safety buffer: any $\hat { z } _ { q }$ that stays within distance � of the true value $z _ { q }$ is guaranteed to remain inside $\tilde { \mathcal { K } } _ { z } ^ { ( q ) }$ , where $\Phi _ { q }$ is still well-defined.

Since $\Phi _ { q }$ is continuous on the compact set $\tilde { \mathcal { K } } _ { z } ^ { ( q ) }$ , it is uniformly continuous there by the Heine-Cantor theorem [35]. Precisely, for the target accuracy $\delta _ { \Phi } = \varepsilon / ( 2 ( 2 n + 1 ) )$ , there exists ${ \delta } _ { z } ^ { * } > 0$ such that

$$
t _ { 1 } , t _ { 2 } \in \tilde { \mathcal { K } } _ { z } ^ { ( q ) } , \quad | t _ { 1 } - t _ { 2 } | < \delta _ { z } ^ { * } \implies | \Phi _ { q } ( t _ { 1 } ) - \Phi _ { q } ( t _ { 2 } ) | < \delta _ { \Phi } .\tag{33}
$$

Define $\delta _ { z } = \operatorname* { m i n } ( \delta _ { z } ^ { * } , r )$ . Because $\delta _ { z } ~ \leq ~ \delta _ { z } ^ { * }$ , condition (33) remains valid for any pair with $| t _ { 1 } - t _ { 2 } | < \delta _ { z }$ . Because $\delta _ { z } \leq r ,$ any deviation of $\hat { z } _ { q }$ from $z _ { q }$ smaller than $\delta _ { z }$ is also smaller than �, keeping $\hat { z } _ { q }$ inside $\tilde { \mathcal { K } } _ { z } ^ { ( \bar { q } ) }$ (verified in the next paragraph). Distributing the budget equally among � inner functions, set $\delta _ { \psi } = \delta _ { z } / n$ . By Theorem 3, for each pair $( q , p )$ there exists a 1D submodel $\hat { \psi } _ { q , p }$ with

$$
\operatorname* { s u p } _ { x _ { p } \in [ 0 , 1 ] } | \psi _ { q , p } ( x _ { p } ) - \hat { \psi } _ { q , p } ( x _ { p } ) | < \delta _ { \psi } .\tag{34}
$$

Applying the triangle inequality gives

$$
| z _ { q } - \hat { z } _ { q } | \leq \sum _ { p = 1 } ^ { n } | \psi _ { q , p } ( x _ { p } ) - \hat { \psi } _ { q , p } ( x _ { p } ) | < n \cdot \delta _ { \psi } = \delta _ { z } \leq r .\tag{35}
$$

Since $z _ { q } \in \mathcal { K } _ { z } ^ { ( q ) } \ \subseteq \tilde { \mathcal { K } } _ { z } ^ { ( q ) }$ and the deviation is less than $r ,$ it follows that $\hat { z } _ { q } \in \tilde { \mathcal { K } } _ { z } ^ { ( q ) }$

Both $z _ { q }$ and $\hat { z } _ { q }$ lie in $\tilde { \mathcal { K } } _ { z } ^ { ( q ) }$ , and $| z _ { q } - \hat { z } _ { q } | ~ < ~ \delta _ { z } ~ \leq ~ \delta _ { z } ^ { * }$ Applying (33) with $t _ { 1 } = z _ { q }$ and $t _ { 2 } = \hat { z } _ { q }$ yields

$$
( \mathbf { C } ) = \vert \Phi _ { q } ( z _ { q } ) - \Phi _ { q } ( \hat { z } _ { q } ) \vert < \delta _ { \Phi } = \frac { \varepsilon } { 2 ( 2 n + 1 ) } .\tag{36}
$$

(D) Outer Approximation Error. Since $\Phi _ { q } ~ \in ~ C ( \tilde { \mathcal { K } } _ { z } ^ { ( q ) } )$ , Theorem 3 applied on the compact interval $\tilde { \mathcal { K } } _ { z } ^ { ( q ) }$ gives a 1D submodel $\hat { \Phi } _ { q }$ such that

$$
\operatorname* { s u p } _ { t \in \tilde { \mathcal { K } } _ { z } ^ { ( q ) } } | \Phi _ { q } ( t ) - \hat { \Phi } _ { q } ( t ) | < \delta _ { \Phi } = \frac { \varepsilon } { 2 ( 2 n + 1 ) } .\tag{37}
$$

Since $\hat { z } _ { q } ( \mathbf { x } ) ~ \in ~ \tilde { \mathcal { K } } _ { z } ^ { ( q ) }$ for all $\textbf { x } \in \ \mathcal { K }$ (established above), evaluating at $t = \hat { z } _ { q } ( \mathbf { x } )$ yields

$$
( \mathrm { D } ) = | \Phi _ { q } ( \hat { z } _ { q } ) - \hat { \Phi } _ { q } ( \hat { z } _ { q } ) | < \delta _ { \Phi } = \frac { \varepsilon } { 2 ( 2 n + 1 ) } .\tag{38}
$$

Total Error. Substituting into (31),

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { K } } | f ( \mathbf { x } ) - \hat { f } ( \mathbf { x } ) | < \displaystyle \sum _ { q = 1 } ^ { 2 n + 1 } \left[ \frac { \varepsilon } { 2 ( 2 n + 1 ) } + \frac { \varepsilon } { 2 ( 2 n + 1 ) } \right] } \\ { \displaystyle = \sum _ { q = 1 } ^ { 2 n + 1 } \frac { \varepsilon } { 2 n + 1 } = \varepsilon . } \end{array}\tag{39}
$$

This completes the proof of Theorem $2 . ^ { 5 }$

Theorem 2 also holds for consequent models that represent linear functions exactly or approximate them arbitrarily well; see Section S2 of the supplementary material.

## VI. EXPERIMENTS

## A. Experimental Setup

1) Methods, Metrics, and Objective: We compare two LCSs: (i) XCSF, which uses �-dimensional rules with linear consequents, and (ii) KACS, which uses one-dimensional rules with linear consequents via KA-based decomposition.

We evaluate each method using three metrics: model accuracy, model complexity, and the accuracy-complexity trade-off.

• Model Accuracy. We track MAE (mean absolute error) on training and testing data.

• Model Complexity. We count the number of macro-rules $| { \mathcal { P } } |$ and the total number of parameters � in P.

• Accuracy-Complexity Trade-off. We use AIC (Akaike Information Criterion) to capture the balance between accuracy and complexity. Under the Gaussian error assumption, AIC is computed as [65]:

$$
\mathrm { A I C } = N _ { \mathrm { t r } } \ln \left( \sum _ { i = 1 } ^ { N _ { \mathrm { t r } } } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } \big / N _ { \mathrm { t r } } \right) + 2 ( k + 1 ) ,\tag{40}
$$

where $N _ { \mathrm { t r } }$ is the number of training samples, � is the number of model parameters, and the extra +1 accounts for the estimated noise variance $\hat { \sigma } ^ { 2 }$

The parameter count �, also used in AIC, is defined as the total number of consequent weight parameters across all rules, i.e., $k = | \mathcal { P } | \times ( d + 1 )$ , where � is the input dimension of each rule (� = � for XCSF; � = 1 for KACS). Hence, for fixed �, � scales linearly with |P | for both methods.

The objective of this comparison is to evaluate KA-based rule organization, which replaces �-dimensional XCSF rules with one-dimensional KACS rulesets while keeping the surrounding LCS framework as comparable as possible.

For this reason, XCSF is used as the primary baseline: it is the most established LCS for function approximation and enables a controlled comparison in which the key difference is rule dimensionality rather than unrelated architectural choices. Accordingly, we do not include KAN [36] in the main comparison, because it is a neural network with B-spline activations and is structurally distinct from LCSs. We also do not include

TABLE I  
BENCHMARK PROBLEMS USED
<table><tr><td>Abbr.</td><td>Name</td><td>#Samples</td><td>n</td></tr><tr><td> $f _ { 1 }$ </td><td>Rastrigin Function</td><td>1000</td><td>10</td></tr><tr><td> $f _ { 2 }$ </td><td>Rosenbrock Function</td><td>1000</td><td>10</td></tr><tr><td>f3</td><td>Cross Function</td><td>1000</td><td>10</td></tr><tr><td> $f _ { 4 }$ </td><td>Styblinski-Tang Function</td><td>1000</td><td>10</td></tr><tr><td> $\mathsf { A S N }$ </td><td>Airfoil Self-Noise</td><td>1503</td><td>5</td></tr><tr><td>CCPP</td><td>Combined Cycle Power Plant</td><td>9568</td><td>4</td></tr><tr><td>CS</td><td>Concrete Strength</td><td>1030</td><td>8</td></tr><tr><td>EEC</td><td>Energy Efficiency Cooling</td><td>768</td><td>8</td></tr></table>

X-KAN [14] in the main comparison, because its KAN-based consequents substantially change the consequent model class and tuning space. We therefore interpret the experimental results as evidence for the effect of dimensional reorganization within the LCS framework, not as a universal ranking against all KA-inspired models. A systematic comparison with KAN and X-KAN is left for future work. For a comparison with SupRB [18], see Section S3 of the supplementary material.

2) Benchmark Problems: Table I summarizes four synthetic test functions and four real-world regression datasets used.

The synthetic functions are the multimodal Rastrigin function [66], the valley-shaped Rosenbrock function [67], the ridge-like Cross function [68], and the nonconvex Styblinski-Tang function [68], all evaluated in $n = 1 0$ dimensions with 1000 uniformly sampled points over $[ 0 , 1 ] ^ { n }$ . Fig. 2 illustrates each function for $n = 2 ,$ and the corresponding mathematical formulations are provided in Section S4 of the supplementary material. The real-world datasets are taken from [18], and include two highly nonlinear problems (ASN and CS) and two nearly linear problems (CCPP and EEC).

3) Hyperparameter Settings and Protocol: We use the same hyperparameter settings for both XCSF and KACS. A detailed description of the hyperparameters is provided in Section S5 of the supplementary material. Unless otherwise stated, the main hyperparameters are set as follows: $N = 6 4 0 0 , \ \epsilon _ { 0 } =$ $0 . 0 1 , \ \beta = 0 . 2 , \ \alpha = 1 , \ \nu = 1 , \ \delta = 0 . 1 , \ m _ { 0 } = 0 . 1 , \ r _ { 0 } = 1 . 0 ,$ $\{ \theta _ { \mathrm { d e l } } , \theta _ { \mathrm { s u b } } , \theta _ { \mathrm { G A } } \} = 5 0 , \chi = 0 . 8 , \mu = 0 . 0 4 , \tau = 0 . 4$ , where most values are typical defaults (e.g., [14], [15], [68]). For synthetic test functions, we set $P _ { \# } ~ = ~ 0$ [68] so that covering always generates local rules, which quickly tile the uniformly sampled input space. For real-world datasets, we set $P _ { \# } = 0 . 8 ~ [ 3 1 ]$ so that covering starts from globally-covering rules. This setting is more stable under highly non-uniform input distributions, which are frequently observed in real-world situations.

We run each method for 100,000 learning iterations and report results averaged over 30 independent runs. For each run, we use Monte Carlo cross-validation with 90% training data and 10% testing data, following [15]. Unlike online functionapproximation protocols that continue learning during testing after a warm-up, we evaluate unseen test samples with learning disabled. For each split, min–max scaling parameters are estimated from the training data only and then applied to the test data, mapping inputs to [0, 1]<sup>�</sup> and targets to [−1, 1]. Thus, the reported MAEs are on the same normalized target scale across all problems. We use the Wilcoxon signed-rank test with significance level $\alpha = 0 . 0 5$ . No post-training rule-compaction [14] or condensation [41] was applied. All experiments were conducted using our Julia implementation, which is publicly available at https://github.com/YNU-NakataLab/KACS.

## B. Main Results and Discussion

Table II reports the averages over 30 runs of model accuracy (training MAE and testing MAE), model complexity (number of rules and parameters), and their trade-off (AIC) at the end of learning for XCSF and KACS. Section S6 of the supplementary material provides the corresponding standard deviations, paired Wilcoxon �-values, and effect sizes for every benchmark and metric.

From Table II, we can see that there is no statistically significant difference in training MAE between XCSF and KACS $( p = 0 . 3 8 3 )$ . In contrast, KACS is significantly better than XCSF in testing MAE, number of rules, number of parameters, and AIC $( p < 0 . 0 5 )$ .

To assess the sensitivity of our conclusions to the evaluation protocol, we additionally compared XCSF and KACS using a protocol based on Heider’s evaluation design [18], with target standardization, MSE, a 75/25 train–test split, and eight Monte Carlo splits combined with eight random seeds per split. Under this alternative protocol, KACS achieved significantly lower training and testing MSE than XCSF on all eight benchmarks, while using fewer parameters and yielding lower AIC in every case. The details are provided in Section S7 of the supplementary material.

1) Model Accuracy: Fig. 3 shows the testing MAE learning curves across all four synthetic functions. The training and testing MAE learning curves across all eight problems are provided in Sections S8-A and S8-B of the supplementary material, respectively. From Table II and Fig. 3, KACS tends to achieve lower testing MAE than XCSF as training progresses, except on near-linear problems where both methods perform comparably. This indicates that reducing the problem to a set of one-dimensional subproblems allows KACS to improve prediction accuracy more efficiently than XCSF, which optimizes rules directly in the full �-dimensional space.

The magnitude of this performance gap depends on the problem characteristics, which are summarized below:

• All four artificial functions $f _ { 1 } { - } f _ { 4 }$ are strongly nonlinear 10-dimensional problems. $f _ { 1 }$ has strong local complexity in the form of multimodality. $f _ { 2 }$ shares this local complexity in the form of a steep curved valley and also exhibits interactions between adjacent dimensions. $f _ { 3 }$ has interactions across all dimensions. $f _ { 4 }$ is challenging because function values rise steeply near the edges as dimensionality increases [68].

• Among the real-world datasets, ASN is the hardest to predict due to its strong nonlinearity [18]. CS is also nonlinear, but somewhat easier to predict [18]. CCPP and EEC are relatively straightforward problems with stronger linearity, where even simple linear models can achieve adequate prediction [18].

On all four artificial functions, KACS achieved significantly lower testing MAE than XCSF. $f _ { 1 }$ is a notable case: XCSF recorded significantly lower training MAE, yet worse testing MAE than KACS. Since XCSF used roughly 17 times more

<sup>�</sup>2  
<sup>�</sup>1  
![](images/ca2b9b0ae98c3e1471d7dfe83f2d24e115ac63ef1b2880a3d1fed854726efea5.jpg)  
<sup>�</sup>1  
(a) �<sub>1</sub>: Rastrigin Function

![](images/220f83830404d6394ff8a354188d295d8e2429c83203187255e54ba8f2c0b821.jpg)  
<sup>�</sup>1  
(b) �<sub>2</sub>: Rosenbrock Function

![](images/3facb2f1cabc5ad5f04d56c5a3d93c4ab2b43d9202496066cf8cb5824fbecace.jpg)  
(c) �<sub>3</sub>: Cross Function

![](images/c5d5fce92d9911cea3c9f7d1f552257d68c1db74680757a9fac06222f9363b3b.jpg)  
<sup>�</sup>1  
(d) �<sub>4</sub>: Styblinski-Tang Function  
Fig. 2. Surface plots of the four synthetic test functions (� = 2) shown for visualization; throughout the experiments we fix � = 10.  
TABLE II

SUMMARY OF MAIN RESULTS. GREEN INDICATES THE BEST VALUE. RANK IS THE AVERAGE RANK ACROSS PROBLEMS. FOR EACH PROBLEM, SYMBOLS +/−/∼ DENOTE SIGNIFICANTLY BETTER/WORSE/SIMILAR PERFORMANCE COMPARED WITH KACS, BASED ON A PAIRED WILCOXON SIGNED-RANK TEST OVER 30 RUNS USING IDENTICAL DATA SPLITS. THE �-VALUES IN THE FINAL ROW ARE FROM ACROSS-PROBLEM PAIRED WILCOXON SIGNED-RANK TESTS APPLIED TO THE 30-RUN MEAN VALUES FOR THE EIGHT BENCHMARK PROBLEMS. ARROWS ↑ /↓ INDICATE RANK IMPROVEMENT/DECLINE RELATIVE TO KACS. STATISTICAL SIGNIFICANCE IS AT � = 0.05 (†)

<table><tr><td rowspan="3"></td><td colspan="4">MODEL ACCURACY</td><td colspan="4">MODEL COMPLEXITY</td><td colspan="2">TRADE-OFF</td></tr><tr><td colspan="2">Training MAE</td><td colspan="2">Testing MAE</td><td colspan="2">#Rules (|P|)</td><td colspan="2">#Parameters (k)</td><td colspan="2">AIC</td></tr><tr><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td></tr><tr><td>fi</td><td>0.2033 +</td><td>0.2194</td><td>0.2642 1</td><td>0.2351</td><td>4162-</td><td>1341</td><td>45780-</td><td>2681</td><td>89210-</td><td>3024</td></tr><tr><td>f2</td><td>0.1124~</td><td>0.1157</td><td>0.1564-</td><td>0.1253</td><td>4087-</td><td>1462</td><td>44960-</td><td>2924</td><td>86580-</td><td>2342</td></tr><tr><td>f3</td><td>0.2510 -</td><td>0.2097</td><td>0.3246-</td><td>0.2329</td><td>4130-</td><td>1429</td><td>45430-</td><td>2857</td><td>88890 -</td><td>3324</td></tr><tr><td>f4</td><td>0.1680-</td><td>0.1334</td><td>0.2303 -</td><td>0.1430</td><td>4170-</td><td>1414</td><td>45870-</td><td>2829</td><td>89080-</td><td>2412</td></tr><tr><td>ASN</td><td>0.1434-</td><td>0.1094</td><td>0.1478-</td><td>0.1144</td><td>2224 –</td><td>1417</td><td>13350-</td><td>2834</td><td>22270 –</td><td>371.1</td></tr><tr><td>CCPP</td><td>0.0930~</td><td>0.0918</td><td>0.0929~</td><td>0.0926</td><td>1953~</td><td>1924</td><td>9764-</td><td>3848</td><td>-17360-</td><td>-29070</td></tr><tr><td>CS</td><td>0.1299～</td><td>0.1213</td><td>0.1350~</td><td>0.1288</td><td>2985 -</td><td>1425</td><td>26870 -</td><td>2851</td><td>50450-</td><td>2237</td></tr><tr><td>EEC</td><td>0.0820~</td><td>0.0967</td><td>0.0847~</td><td>0.1020</td><td>2925 -</td><td>1012</td><td>26330-</td><td>2023</td><td>49850-</td><td>1125</td></tr><tr><td>Rank</td><td>1.62↓</td><td>1.38</td><td>1.88↓†</td><td>1.12</td><td>2.00↓†</td><td>1.00</td><td>2.00↓†</td><td>1.00</td><td>2.00↓†</td><td>1.00</td></tr><tr><td>+/-1~</td><td>1/3/4</td><td></td><td>0/5/3</td><td></td><td>0/7/1</td><td>=</td><td>0/8/0</td><td></td><td>0/8/0</td><td></td></tr><tr><td>p-value</td><td>0.383</td><td></td><td>0.0391</td><td></td><td>0.00781</td><td></td><td>0.00781</td><td></td><td>0.00781</td><td></td></tr></table>

![](images/e0037c62b2e1ea7974f088817041e35243fd7142b2781619df4d870288a1d607.jpg)  
(a) � : Rastrigin Function

![](images/a017c16abbbe5f28b1270d53fbf389e1c4b940687e416e72d5ceeb4d34686684.jpg)  
(b) � : Rosenbrock Function

(c) �<sub>3</sub>: Cross Function  
![](images/9b65520eff96cc51cb9b6ff96815d1fede4f52d042f50b48af07b589f4bd93bb.jpg)

![](images/46c97178df24a535fa6d94b15bf89d546a081a592b3e7fa4a8dce8297f1aea74.jpg)  
(d) � : Styblinski-Tang Function  
Fig. 3. Testing MAE learning curves for all four synthetic functions. Curves show the mean over 30 runs, and shaded regions denote 95% confidence intervals

parameters on this problem (45780 for XCSF vs. 2681 for KACS), this pattern points to overfitting driven by multidimensional rules on a multimodal function. A similar pattern appears on $f _ { 2 } { \mathrm { : } }$ the gap between training and testing MAE is much wider for XCSF (0.1124 → 0.1564, Δ = 0.044) than for KACS $( 0 . 1 1 5 7  0 . 1 2 5 3$ $\Delta \ : = \ : 0 . 0 1 0 )$ , suggesting that steep curved valleys cause mild overfitting for �-dimensional rules. On $f _ { 3 } ,$ , where interactions span all dimensions simultaneously, KACS outperformed XCSF in both training and testing MAE. XCSF must use �-dimensional rules to capture such interactions implicitly, whereas KACS represents them through nonlinear outer functions that aggregate all dimensions. On $f _ { 4 } ,$ the difficulty lies in approximating steep edge regions in 10- dimensional space, and KACS again showed a clear advantage.

Among the real-world datasets, KACS achieved significantly lower testing MAE than XCSF only on ASN. For CS and CCPP, no statistically significant differences were found. For EEC, KACS showed higher testing MAE than XCSF (0.1020 vs. 0.0847), but the difference was not statistically significant. For CCPP and EEC, �-dimensional linear rules are already sufficient (testing MAE below 0.1). In such nearlinear problems, the KA decomposition introduces unnecessary structural overhead: KACS must coordinate �(2� + 1) inner submodels and compose their outputs through 2� + 1 outer submodels to recover a function that XCSF can represent directly with multiple linear rules. This additional indirection accumulates approximation error in the outer stage, which outweighs the benefits of dimensional decomposition.

In Fig. 3, KACS started with higher testing MAE than XCSF in the early phase of training (0–20000 iterations) across all problems. This is because KACS needs to initialize and tune many one-dimensional submodels at once, which produces inaccurate predictions early on and temporarily inflates the overall error. The GA-driven search is also distributed across all $2 n ^ { 2 } + 3 n + 1$ submodels at the start, which slows convergence of the final output compared to XCSF.

The above findings can be summarized as follows:

• The accuracy advantage of KACS grows with problem nonlinearity, particularly when cross-dimensional interactions or sharp local structures are present, and fades when the problem is well-captured by linear approximations.

• In all cases, a warm-up period is needed at the start of KACS training, as coordinating many one-dimensional submodels from scratch inevitably raises early-stage error.

Finally, we evaluate rotated, translated, and scaled variants of $f _ { 3 }$ to test sensitivity to ridge orientation, location, and scale. As reported in Section S9 of the supplementary material, KACS achieves significantly lower testing MAE than XCSF for all three variants, indicating that its advantage is not limited to the original axis-aligned structure of $f _ { 3 }$

2) Model Complexity and Accuracy-Complexity Trade-off: One of the most significant results in this article is the large difference in model complexity. As shown in Table II, KACS achieved comparable or higher prediction accuracy using only roughly 6–40% of the parameters required by XCSF across all eight problems. The parameter-count learning curves are provided in Section S8-C of the supplementary material.

This gap reflects a fundamental difference in how each method constructs its rules. XCSF covers the input space with local linear models, each defined over the full �-dimensional space. As � grows, defining meaningful local regions and fitting reliable linear models within them both become harder, which tends to produce redundant or poorly fitted rules, driving up model size and the risk of overfitting (cf. Section VI-B1). KACS avoids this by decomposing the problem into onedimensional subproblems, allowing it to represent multivariate functions compactly with far fewer parameters.

KACS achieved significantly lower AIC than XCSF on all eight problems, indicating that its models are statistically more efficient. On CS, where testing MAE is similar for both methods, the AIC advantage of KACS comes entirely from its smaller parameter count. On EEC, where KACS records slightly higher testing MAE than XCSF, KACS still achieves far better AIC, showing that its reduction in model complexity more than compensates for the small loss in accuracy.

## C. Scalability Analysis

This subsection analyzes the scalability of XCSF and KACS. We use $f _ { 3 }$ (Cross function) as the representative benchmark because it is the only synthetic function in our set that involves interactions across all dimensions, making it directly sensitive to changes in �.

1) Experimental Design: Unless otherwise noted, hyperparameter settings and training iterations follow the setup in Section VI-A. We report testing MAE and two complexity measures, |P| and �, as means with 95% confidence intervals.

2) Results on Sample-Size Scaling: We vary $N _ { S } \in \mathsf { \Gamma }$ {125, 250, 500, 1000, 2000, 4000, 8000} while fixing $n ~ = ~ 1 0$ Fig. 4 summarizes how both methods respond to more data.

The detailed tabular results are provided in Section S10-A of the supplementary material.

Increasing $N _ { S }$ improves accuracy for both methods, but the gain is larger for KACS. XCSF testing MAE decreases from 0.416 $( N _ { S } = 1 2 5 )$ to 0.312 $( N _ { S } = 8 0 0 0 )$ , while KACS decreases from 0.380 to 0.216 and stabilizes after $N _ { S } \approx 2 0 0 0 .$ Complexity grows much more slowly for KACS: XCSF rules and parameters increase from 3164 to 4724 and from 34800 to 51960, respectively, whereas KACS stays around 1242– 1430 rules and 2484–2860 parameters. Thus, more data mainly improves KACS accuracy without inducing comparable complexity growth.

3) Results on Dimensional Scaling: We vary � ∈ {2, 4, 6, 8, 10, 12, 14, 16, 18, 20} while fixing $N _ { S } = 1 0 0 0 . { \mathrm { ~ F i g . ~ } } 5$ summarizes how both methods respond to increasing dimensionality. The detailed tabular results are provided in Section S10-B of the supplementary material.

As � increases, XCSF testing MAE rises from 0.158 at $n = 2$ to about 0.27–0.33 at $n = 4 – 2 0 ;$ ; KACS significantly outperforms XCSF from � = 2 to � = 16 (e.g., 0.233 vs. 0.325 at � = 10), but degrades sharply at higher dimensions (0.384 at � = 18, 1.25 at � = 20). Complexity trends are clearer: XCSF rules grow monotonically (1501 to 5283), and its parameter count grows from 4504 to 110900. KACS remains compact, with 1186–1955 rules and 2372–3909 parameters across most settings, with only a notable increase at $n \geq 1 8$

4) Discussion: Figs. 4 and 5 together support two claims. First, KACS provides a better accuracy-complexity trade-off in the practical regime $( n \leq 1 4$ and all tested sample sizes), consistently using far fewer rules and parameters than XCSF. Second, under the default $N = 6 4 0 0$ , KACS shows a clear performance drop in higher dimensions $( n = 1 6 – 2 0 )$ : accuracy deteriorates sharply even though the model remains compact.

The failure at higher � stems from two related capacity bottlenecks. First, KACS decomposes the �-dimensional problem into $( 2 n + 1 ) ( n + 1 ) \ = \ 2 n ^ { 2 } + 3 n + 1$ one-dimensional subproblems, so the number of macro-rules per subproblem shrinks as � grows. At $n = 2 0 .$ , for instance, this gives 861 subproblems, leaving at most $\lfloor 6 4 0 0 / 8 6 1 \rfloor \approx 7$ macro-rules per subproblem on average, and typically fewer because individual rules can carry numerosity greater than one. Second, and more critically, the inner and outer submodels have fundamentally different input domains. The inner submodel input domain is fixed at [0, 1] regardless of $n ,$ whereas the outer submodel input $\begin{array} { r } { \hat { z } _ { q } = \sum _ { p = 1 } ^ { n } \hat { \psi } _ { q , p } ( x _ { p } ) } \end{array}$ is a sum of � terms. Since inner rule weights are initialized from $\mathcal { U } [ - 1 , 1 ]$ , the output of each $\hat { \psi } _ { q , p }$ can be negative, and the outer input domain can be written as $\begin{array} { r } { \hat { \mathcal { K } } _ { z } ^ { ( q ) } : = \{ \hat { z } _ { q } | \hat { z } _ { q } = \sum _ { p = 1 } ^ { n } \hat { \psi } _ { q , p } ( x _ { p } ) , x _ { p } \in [ 0 , 1 ] \} \approx [ - n , n ] . } \end{array}$ so its domain width scales as $| \hat { \mathcal { K } } _ { z } ^ { ( q ) } | ~ \approx ~ 2 n$ . Consequently, outer submodels require $O ( n )$ times more rules than inner submodels to achieve the same coverage density. When � is set without accounting for this difference, outer submodels are severely under-resourced at large $n ,$ triggering a coverdelete $c y c l e ^ { 6 }$ in which newly generated covering rules are immediately deleted to maintain the population within � [69].

�  
![](images/57aa04019362f36dbbc5afed6df7d44649956bbcd926ec81400d01ba22282f4e.jpg)  
(a) Testing MAE vs. �<sub>�</sub>

![](images/83939376e5eb681015d40b3622396bdcb4d0e3a54dd97ed2d421a421620c5c4e.jpg)  
(b) #Rules vs. �<sub>�</sub>

![](images/0e30dab7d68ec5a5e4bc980e933485705be4eb7f553ec2f5e8da50c3317e9f0f.jpg)  
(c) #Parameters vs. �<sub>�</sub>  
Fig. 4. Sample-size scaling results for XCSF and KACS on $f _ { 3 } ,$ where �<sub>�</sub> ∈ {125, 250, 500, 1000, 2000, 4000, 8000} and $n = 1 0 .$

![](images/df5058de837597049632165d3c6af6507560df3652d199ab5c58c643876af8f0.jpg)  
(a) Testing MAE vs. �

![](images/6d723e2facb8e4a6d24b5979b1539c791eb53038f8706e1b60ddd3da268aece0.jpg)  
(b) #Rules vs. �

![](images/d6bb9d6488d19bd88b567f9d4d9e285b00b7553c02a4f2d704adf7b7c1d9699e.jpg)  
(c) #Parameters vs. �  
Fig. 5. Dimensional scaling results for XCSF and KACS on $f _ { 3 } ,$ where � ∈ {2, 4, 6, 8, 10, 12, 14, 16, 18, 20} and $N _ { S } = 1 0 0 0 .$

![](images/d510e0034424250bb809b6e621fd6f467fe9b8794c404291ea3e91d37d3524af.jpg)

![](images/ce97aa8b6cf6f345241b40b9656d1536a5376c10a963625f3b2cafff47553b32.jpg)

![](images/ebeec0d50de47eb97ed9607778179d6fa8c2ecd1ad68d56c0899434ef12db2e9.jpg)  
(a) Testing MAE vs. �  
(b) #Rules vs. �  
(c) #Parameters vs. �  
Fig. 6. Dimensional scaling results for XCSF and KACS on $f _ { 3 } ,$ where $n \in$ {16, 18, 20}, � ∈ {12672, 15984, 19680}, and $N _ { S } = 1 0 0 0 .$

A guideline for � should therefore distinguish between inner and outer coverage requirements. Let $k _ { \mathrm { m i n } }$ denote the minimum number of macro-rules per unit of input domain width. Inner submodels, each with domain [0, 1] of width 1, require $k _ { \mathrm { m i n } }$ rules. Outer submodels, with domain $\approx [ - n , n ]$ of width 2�, require $k _ { \mathrm { m i n } } { \cdot } 2 I$ � rules. With $n ( 2 n { + } 1 )$ inner submodels and (2�+1) outer submodels, the total rule budget must satisfy

$$
N \ \geq \ k _ { \operatorname* { m i n } } \cdot n ( 2 n + 1 ) + k _ { \operatorname* { m i n } } \cdot 2 n \cdot ( 2 n + 1 ) \ = \ 3 k _ { \operatorname* { m i n } } \cdot n ( 2 n + 1 ) .\tag{41}
$$

We estimate $k _ { \mathrm { m i n } }$ from the last stable dimension $n = 1 4 \colon$

$$
k _ { \mathrm { m i n } } ~ \approx ~ { \frac { N } { 3 \cdot n ( 2 n + 1 ) } } { \Bigg | } _ { n = 1 4 } = { \frac { 6 4 0 0 } { 3 \times 1 4 \times 2 9 } } \approx 5 . 3 .
$$

Rounding up and adding a modest safety margin to account for problem-to-problem variation in the outer function range, we adopt $k _ { \operatorname* { m i n } } = 8$ , which gives ${ \cal N } \ge 3 { \times } 8 { \times } n ( 2 n { + } 1 ) = 2 4 n ( 2 n { + } 1 )$ predicting $N \ge 1 2 6 7 2$ , 15984, 19680 for $n ~ = ~ 1 6 , 1 8 , 2 0 .$ , respectively. To verify this, we ran additional experiments with these � values at the corresponding dimensions. As shown in Fig. 6, under this scaled budget, the severe degradation at $n \in \{ 1 6 , 1 8 , 2 0 \}$ no longer appears. KACS attains significantly lower testing MAE than XCSF at all three dimensions (16:

0.2109 vs. 0.2768, 18: 0.2115 vs. 0.2768, 20: 0.1970 vs. 0.2833). KACS also remains much more compact, using significantly fewer macro-rules $( | { \mathcal { P } } | ;$ 2633, 3306, 4017 vs. 9520, 12310, 15690) and significantly fewer parameters (�: 5266, 6612, 8033 vs. 161800, 233900, 329600). In particular, at $n \ : = \ : 2 0 $ , KACS achieves lower testing MAE than XCSF using only about 2% as many parameters. This confirms that the higher-dimensional failure is a capacity issue attributable to under-resourced outer submodels, rather than an inherent limitation of KACS.

Note that $k _ { \mathrm { m i n } }$ is not a universal constant: its value depends on the actual range of the outer submodel inputs, which varies with the problem and the learned inner functions. The estimate $k _ { \mathrm { m i n } } \approx 5 . 3$ was derived empirically from a single benchmark $( f _ { 3 } ) ,$ , so (41) should be treated as a practical starting point rather than a tight bound; its transferability to other functions and real-world data remains to be validated.

## VII. CONCLUSION

In this article, we proposed KACS, the first LCS that reorganizes rules dimension-wise according to the Kolmogorov-Arnold (KA) representation theorem. Instead of evolving rules directly in the �-dimensional input space, KACS decomposes the target function into one-dimensional subproblems and assigns a dedicated ruleset to each, optimized through evolutionary algorithms and gradient-based methods. This reduces the worst-case rule count from $O ( m ^ { n } )$ to $O ( m n ^ { 2 } )$ where � is the per-variable resolution. We also proved that KACS is a universal approximator for continuous functions on compact domains, which is the first such proof for any LCS. Experiments showed that KACS provides a parameterefficient alternative for function approximation, with competitive accuracy across many tested settings.

These results suggest that KA-guided dimension-wise decomposition is a broadly applicable design principle. Any local modeling framework that partitions the input space directly, such as fuzzy systems and other rule-based systems, may benefit from reorganizing its components into one-dimensional subproblems. KACS provides a concrete example of this principle and shows that such reorganization improves both accuracy and model compactness without sacrificing expressive power, as guaranteed by the universal approximation proof.

Several limitations remain. First, initializing many submodels simultaneously requires a warm-up period, during which prediction error temporarily exceeds that of XCSF. Second, reliable coverage requires a population budget of $N = O ( n ^ { 2 } )$ a smaller budget triggers a cover-delete cycle that degrades performance. Third, the universal approximation proof guarantees only the existence of an accurate ruleset, not that the learning algorithm will converge to it in practice.

Future work should address these issues in order of practical impact. An adaptive mechanism that monitors submodel coverage and adjusts � dynamically would resolve both the warm-up and budget problems. On the theoretical side, deriving convergence rates for the joint GA and gradientbased optimization would connect the existence proof to algorithmic behavior. At a broader level, extending dimensionwise decomposition to other local modeling frameworks and identifying the conditions under which it outperforms direct partitioning is a direction that reaches beyond LCSs, and KACS provides a concrete foundation for that investigation. Within this direction, relaxing the strict KA structure may further improve approximation efficiency. For example, one can vary the number of inner and outer models, or stack multiple KA layers as in KAN [36]. Whether universal approximation is preserved under these modifications remains an open question. Future studies will also visualize the learned rules and submodels, evaluate structural fidelity by comparing learned models with known low-dimensional target functions, and compare KACS with KAN-based models [14], genetic programming variants [70], and neural- and code-fragmentbased LCSs [42], [43] across broader problem suites.

## REFERENCES

[1] C. M. Bishop and N. M. Nasrabadi, Pattern recognition and machine learning. Springer, 2006, vol. 4, no. 4.

[2] T. Hastie, R. Tibshirani, and J. H. Friedman, The elements of statistical learning: data mining, inference, and prediction. Springer, 2009, vol. 2.

[3] I. Goodfellow, Y. Bengio, and A. Courville, Deep learning. MIT press Cambridge, 2016, vol. 1, no. 2.

[4] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT press, 2018.

[5] K. J. Astr<sup>˚</sup> om, “Adaptive control,” in¨ Mathematical System Theory: The Influence of RE Kalman. Springer, 1995, pp. 437–450.

[6] S. Vijayakumar, A. D’souza, and S. Schaal, “Incremental online learning in high dimensions,” Neural computation, vol. 17, no. 12, pp. 2602– 2634, 2005.

[7] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive mixtures of local experts,” Neural Comput., vol. 3, no. 1, pp. 79–87, 1991.

[8] H. D. Nguyen and F. Chamroukhi, “Practical and theoretical aspects of mixture-of-experts modeling: An overview,” Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, vol. 8, no. 4, p. e1246, 2018.

[9] A. Hinrichs, E. Novak, and H. Wozniakowski, “The curse of dimen-´ sionality for monotone and convex functions of many variables,” arXiv preprint arXiv:1011.3680, 2010.

[10] L. Breiman, J. Friedman, R. A. Olshen, and C. J. Stone, Classification and regression trees. Chapman and Hall/CRC, 2017.

[11] T. Takagi and M. Sugeno, “Fuzzy identification of systems and its applications to modeling and control,” IEEE Trans. Syst., Man, Cybern., no. 1, pp. 116–132, 1985.

[12] R. J. Urbanowicz and W. N. Browne, Introduction to Learning Classifier Systems, 1st ed. Springer Publishing Company, Incorporated, 2017.

[13] A. Siddique, M. Heider, M. Iqbal, and H. Shiraishi, “A survey on learning classifier systems from 2022 to 2024,” in Proc. Genet. Evol. Comput. Conf. Companion, 2024, pp. 1797–1806.

[14] H. Shiraishi, H. Ishibuchi, and M. Nakata, “X-KAN: Optimizing local kolmogorov-arnold networks via evolutionary rule-based machine learning,” in Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, 8 2025, pp. 8930–8938, Main Track.

[15] R. J. Preen, S. W. Wilson, and L. Bull, “Autoencoding with a classifier system,” IEEE Trans. Evol. Comput., vol. 25, no. 6, pp. 1079–1090, 2021.

[16] H. H. Dam, H. A. Abbass, C. Lokan, and X. Yao, “Neural-based learning classifier systems,” IEEE Trans. Knowl. Data Eng., vol. 20, no. 1, pp. 26–39, 2007.

[17] S. W. Wilson, “Classifiers that approximate functions,” Natural Comput., vol. 1, no. 2, pp. 211–234, 04 2002.

[18] M. Heider, H. Stegherr, R. Sraj, D. Patzel, J. Wurth, and J. H¨ ahner,¨ “SupRB in the context of rule-based machine learning methods: A comparative study,” Appl. Soft Comput., p. 110706, 2023.

[19] E. Debie and K. Shafi, “Implications of the curse of dimensionality for supervised learning classifier systems: theoretical and empirical analyses,” Pattern Anal. Appl., vol. 22, no. 2, pp. 519–536, 05 2019.

[20] R. J. Urbanowicz and J. H. Moore, “Learning classifier systems: a complete introduction, review, and roadmap,” J. Artif. Evol. Appl., vol. 2009, 2009.

[21] M. Heider, D. Patzel, H. Stegherr, and J. H¨ ahner, “A metaheuristic per-¨ spective on learning classifier systems,” in Metaheuristics for Machine Learning: New Advances and Tools. Springer, 2022, pp. 73–98.

[22] J. Drugowitsch, Design and Analysis of Learning Classifier Systems: A Probabilistic Approach. Springer Science & Business Media, 2008, vol. 139.

[23] P. O. Stalph, X. Llora, D. E. Goldberg, and M. V. Butz, “Resource\` management and scalability of the XCSF learning classifier system,” Theoretical Computer Science, vol. 425, pp. 126–141, 2012.

[24] A. N. Kolmogorov, On the representation of continuous functions of several variables by superpositions of continuous functions of a smaller number of variables. American Mathematical Society, 1961.

[25] V. I. Arnold, “On functions of three variables,” Collected Works: Representations of Functions, Celestial Mechanics and KAM Theory, 1957–1965, pp. 5–8, 2009.

[26] J. Schmidt-Hieber, “The kolmogorov–arnold representation theorem revisited,” Neural networks, vol. 137, pp. 119–126, 2021.

[27] G. Cybenko, “Approximation by superpositions of a sigmoidal function,” Math. Control Signals Syst., vol. 2, no. 4, pp. 303–314, 1989.

[28] P. L. Lanzi and D. Loiacono, “A proposal for a leaner narrative of learning classifier systems,” in Proceedings of the Genetic and Evolutionary Computation Conference Companion, 2025, pp. 2261–2268.

[29] S. W. Wilson, “Classifier fitness based on accuracy,” Evol. Comput., vol. 3, no. 2, p. 149–175, jun 1995.

[30] M. V. Butz and S. W. Wilson, “An algorithmic description of XCS,” Soft Comput., vol. 6, no. 3-4, pp. 144–153, 2002.

[31] M. Nakata and W. N. Browne, “Learning optimality theory for accuracybased learning classifier systems,” IEEE Trans. Evol. Comput., vol. 25, no. 1, pp. 61–74, Feb 2021.

[32] B. Widrow and M. E. Hoff, “Adaptive switching circuits,” Stanford Univ Ca Stanford Electronics Labs, Tech. Rep., 1960.

[33] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization. international conference on learning representations,” 2015.

[34] S. W. Wilson, “Generalization in the XCS classifier system,” Proc. Genetic Programming 1998, 1998.

[35] W. Rudin, “Principles of mathematical analysis,” 3rd ed., 1976.

[36] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Y. Hou, and M. Tegmark, “KAN: Kolmogorov–arnold networks,” in The Thirteenth International Conference on Learning Representations, 2025.

[37] H. Shiraishi, Y. Hayamizu, T. Hashiyama, K. Takadama, H. Ishibuchi, and M. Nakata, “Adapting rule representation with four-parameter beta distribution for learning classifier systems,” IEEE Transactions on Evolutionary Computation, vol. 30, no. 1, pp. 363–377, 2026.

[38] S. W. Wilson, “Get real! XCS with continuous-valued inputs,” in Int. Worksh. Learn. Classif. Syst. Springer, 1999, pp. 209–219.

[39] ——, “Mining oblique data with XCS,” in Int. Worksh. Learn. Classif. Syst., 2000, pp. 158–174.

[40] H. Shiraishi, Y. Hayamizu, H. Sato, and K. Takadama, “Can the same rule representation change its matching area? enhancing representation in XCS for continuous space by probability distribution in multiple dimension,” in Proc. Genet. Evol. Comput. Conf., 2022, p. 431–439.

[41] M. V. Butz, P. L. Lanzi, and S. W. Wilson, “Function approximation with XCS: Hyperellipsoidal conditions, recursive least squares, and compaction,” IEEE Trans. Evol. Comput., vol. 12, no. 3, pp. 355–376, June 2008.

[42] L. Bull and T. O’Hara, “Accuracy-based neuro and neuro-fuzzy classifier systems,” in Proc. 4th Annu. Conf. Genet. Evol. Comput., 2002, p. 905–911.

[43] M. H. Arif, J. Li, and M. Iqbal, “Solving social media text classification problems using code fragment-based XCSR,” in 2017 IEEE 29th Int. Conf. Tools Artif. Intell. (ICTAI), Nov 2017, pp. 485–492.

[44] M. Behdad, T. French, L. Barone, and M. Bennamoun, “PCA for improving the performance of XCSR in classification of high-dimensional problems,” in Proc. 13th Annu. Conf. Companion Genet. Evol. Comput., 2011, pp. 361–368.

[45] H. Shiraishi, M. Tadokoro, Y. Hayamizu, Y. Fukumoto, H. Sato, and K. Takadama, “Increasing accuracy and interpretability of highdimensional rules for learning classifier system,” in 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021, pp. 311–318.

[46] N. Yatsu, H. Shiraishi, H. Sato, and K. Takadama, “Exploring highdimensional rules indirectly via latent space through a dimensionality reduction for XCS,” in Proc. Genet. Evol. Comput. Conf., 2023, pp. 606–614.

[47] C. Schonberner, A. Mackensen, and S. Tomforde, “Dimensionality¨ reduction for enabling visual reinforcement learning with a classifier system,” in Proc. Genet. Evol. Comput. Conf. Companion, 2025, pp. 2269–2278.

[48] R. J. Urbanowicz and J. H. Moore, “ExSTraCS 2.0: description and evaluation of a scalable learning classifier system,” Evol. Intell., vol. 8, no. 2, pp. 89–116, 2015.

[49] J. H. Moore and B. C. White, “Tuning relieff for genome-wide genetic analysis,” in Proc. Eur. Conf. Evol. Comput., Mach. Learn. Data Mining Bioinformatics. Springer, 2007, pp. 166–175.

[50] J. H. Friedman and B. E. Popescu, “Predictive learning via rule ensembles,” Ann. Appl. Stat., vol. 2, no. 3, pp. 916–954, 2008.

[51] R. Bellman, “Dynamic programming,” science, vol. 153, no. 3731, pp. 34–37, 1966.

[52] Z. Bozorgasl and H. Chen, “Wav-kan: Wavelet kolmogorov-arnold networks,” 2024. [Online]. Available: https://arxiv.org/abs/2405.12832

[53] G. De Franceschi et al., “Ensemble-kan: Leveraging kolmogorov arnold networks to discriminate individuals with psychiatric disorders from controls,” in Proc. Int. Workshop Appl. Med. AI, 2024, pp. 186–197.

[54] S. SS, K. AR, A. KP et al., “Chebyshev polynomial-based kolmogorovarnold networks: An efficient architecture for nonlinear function approximation,” arXiv preprint arXiv:2405.07200, 2024.

[55] R. Genet and H. Inzirillo, “Tkan: Temporal kolmogorov-arnold networks,” 2025. [Online]. Available: https://arxiv.org/abs/2405.07344

[56] E. Zeydan, C. J. Vaca-Rubio, L. Blanco, R. Pereira, M. Caus, and A. Aydeger, “F-kans: Federated kolmogorov-arnold networks,” in Proc. 2025 IEEE 22nd Consum. Commun. Netw. Conf. IEEE, 2025, pp. 1–6.

[57] K. Hornik, M. Stinchcombe, and H. White, “Multilayer feedforward networks are universal approximators,” Neural networks, vol. 2, no. 5, pp. 359–366, 1989.

[58] L.-X. Wang, J. M. Mendel et al., “Fuzzy basis functions, universal approximation, and orthogonal least-squares learning,” IEEE transactions on Neural Networks, vol. 3, no. 5, pp. 807–814, 1992.

[59] M. H. Stone, “The generalized weierstrass approximation theorem,” Mathematics Magazine, vol. 21, no. 5, pp. 237–254, 1948.

[60] H. Ying, “Sufficient conditions on general fuzzy systems as function approximators,” Automatica, vol. 30, no. 3, pp. 521–525, 1994.

[61] B. Kosko, “Fuzzy systems as universal approximators,” IEEE Transactions on Computers, vol. 43, no. 11, pp. 1329–1333, 2002.

[62] J. L. Castro, “Fuzzy logic controllers are universal approximators,” IEEE Trans. Syst., Man, Cybern., vol. 25, no. 4, pp. 629–635, 2002.

[63] L.-X. Wang, “Universal approximation by hierarchical fuzzy systems,” Fuzzy sets and systems, vol. 93, no. 2, pp. 223–230, 1998.

[64] G. B. Folland, Real analysis: modern techniques and their applications. John Wiley & Sons, 1999.

[65] K. P. Burnham and D. R. Anderson, Model selection and multimodel inference: a practical information-theoretic approach. Springer, 2002.

[66] H. Muhlenbein, M. Schomisch, and J. Born, “The parallel genetic¨ algorithm as function optimizer,” Parallel computing, vol. 17, no. 6-7, pp. 619–632, 1991.

[67] M. Jamil and X. S. Yang, “A literature survey of benchmark functions for global optimisation problems,” Int. J. Math. Model. Numer. Optim., vol. 4, no. 2, p. 150, 2013.

[68] A. Stein, S. Menssen, and J. Hahner, “What about interpolation? a radial¨ basis function approach to classifier prediction modeling in XCSF,” in Proc. Genet. Evol. Comput. Conf., 2018, pp. 537–544.

[69] M. Butz, T. Kovacs, P. Lanzi, and S. Wilson, “Toward a theory of generalization and learning in XCS,” IEEE Trans. Evol. Comput., vol. 8, no. 1, pp. 28–46, Feb 2004.

[70] J. R. Koza, “Genetic programming as a means for programming computers by natural selection,” Stat. Comput., vol. 4, no. 2, pp. 87–112, 1994.

![](images/534df19d39f5e4a0e9be15e6a352b1b2b1904c46f5101b5c91e23fa1fba561e1.jpg)

Hiroki Shiraishi received the B.E. and M.E. degrees in informatics from the University of Electro-Communications, Tokyo, Japan, in 2021 and 2023, respectively, and the Ph.D. degree in information systems from Yokohama National University, Yokohama, Japan, in 2026.

Since 2026, he has been a Researcher with the Digital Healthcare Research Department, Hitachi, Ltd., Tokyo, Japan. His research interests include fuzzy systems, neuroevolution, learning classifier systems, and their clinical applications in pathology.

Dr. Shiraishi received the Best Paper Award at GECCO 2022 and was nominated for the Best Paper Award at GECCO 2023. He served as Chair of the International Workshop on Evolutionary Rule-Based Machine Learning at GECCO from 2024 to 2026.

![](images/e74a3df6adb7329ef841c6bc2c4b7cb37270ebd37a26e043c6238cf05e49e7f6.jpg)

Hisao Ishibuchi (Fellow, IEEE) received the B.S. and M.S. degrees from Kyoto University in 1985 and 1987, respectively, and the Ph.D. degree from Osaka Prefecture University in 1992.

He was with Osaka Prefecture University from 1987 to 2017. Since 2017, he has been a Chair Professor at Southern University of Science and Technology, China. His research interests include the design, analysis, comparison and application of evolutionary multi-objective optimization algorithms.

Dr. Ishibuchi was the IEEE CIS Vice-President

for Technical Activities in 2010-2013, the Editor-in-Chief of the IEEE COMPUTATIONAL INTELLIGENCE MAGAZINE in 2014-2019, and an AdCom member of the IEEE CIS in 2014-2019 and 2021-2026. Currently, he is General Chair of EMO 2027.

![](images/72a0be0b72227dc3c017f980fde17f218f574c4c062b4891ddc35c4de3cf632f.jpg)

Masaya Nakata (Member, IEEE) received the Ph.D. degree in informatics from the University of Electro-Communications, Tokyo, Japan, in 2016.

He is an Associate Professor with the Faculty of Engineering, Yokohama National University, Yokohama, Japan. Since 2019, he has focused his research on surrogate-assisted evolutionary algorithms. His research has also included evolutionary machine learning, data mining, and the theoretical analysis of learning classifier systems.

Dr. Nakata chaired the International Workshop on

Learning Classifier Systems in 2015, 2016, and 2018–2020 in GECCO.

# Supplementary Material of “Kolmogorov-Arnold Classifier Systems as Universal Approximators”

Hiroki Shiraishi , Hisao Ishibuchi , Fellow, IEEE, and Masaya Nakata , Member, IEEE

## CONTENTS

S1 Inference Complexity Analysis 17   
S1-A XCSF 17   
S1-B KACS 17   
S1-C Comparison of Inference Complexity 17   
S1-D Asymptotic Rule-Count Interpretation 18   
S2 Generalization of the KACS Universal Approximation Theorem to Other Consequent Models 18   
S3 Comparison with SupRB: A State-of-the-Art Pittsburgh-Style LCS 19   
S3-A Experimental Protocol and Comparability 19   
S3-B Prediction Accuracy 19   
S3-C Rule Complexity . 20   
S3-D Interpretability of the Learned Models 20   
S4 Formulations of Synthetic Functions 21   
S5 A Description of Hyperparameters 22   
S6 Detailed Statistical Results 22   
S7 Robustness under an Alternative Evaluation Protocol 24   
S7-A Experimental Protocol 24   
S7-B Main Results . 24   
S7-C Statistical Results and Interpretation 25   
S8 Additional Learning Curves 27   
S8-A Training MAE Learning Curves 27   
S8-B Testing MAE Learning Curves 27   
S8-C Parameter-Count Learning Curves 28   
S9 Effects of Rotation, Translation, and Scaling on � 29   
S9-A Problem Definition 29   
S9-B Experimental Results 29   
S9-C Discussion 30   
S10 Scalability Analysis — Detailed Results 31   
S10-A Sample-Size Scaling 31   
S10-B Dimensional Scaling . 34   
S11 Covering Dynamics and Cover-Delete Behavior 36

## References

## S1. INFERENCE COMPLEXITY ANALYSIS

This section compares the worst-case inference complexity of XCSF and KACS. The analysis concerns prediction only. It excludes all training operations, including covering, parameter updates, genetic search, subsumption, and deletion.

Let $R _ { X }$ denote the number of macro-rules in the final XCSF population, i.e.,

$$
R _ { X } = | { \mathcal { P } } | .\tag{1}
$$

Let $R _ { \mathrm { K } }$ denote the total number of macro-rules in all KACS inner and outer submodels:

$$
R _ { \mathrm { K } } = \vert \mathcal { P } \vert = \sum _ { q = 1 } ^ { 2 n + 1 } \sum _ { p = 1 } ^ { n } \left. \mathcal { P } _ { q , p } ^ { \psi } \right. + \sum _ { q = 1 } ^ { 2 n + 1 } \left. \mathcal { P } _ { q } ^ { \Phi } \right. .\tag{2}
$$

Here, a macro-rule is counted once regardless of its numerosity.

## A. XCSF

For an input $\mathbf { x } \in \mathbb { R } ^ { n }$ , XCSF forms a match set by testing whether x satisfies the interval condition of each rule. Since an XCSF rule has an �-dimensional condition, matching one rule requires $O ( n )$ comparisons in the worst case. Consequently, constructing the match set requires

$$
O ( n R _ { \mathrm { X } } )\tag{3}
$$

operations.

Each matched rule has an $( n + 1 )$ -parameter linear consequent. Its prediction therefore requires $O ( n )$ arithmetic operations. Because the number of matched rules is at most $R _ { \mathrm { X } } .$ , computing and aggregating the rule predictions also requires

$$
O ( n R _ { \mathrm { X } } )\tag{4}
$$

operations in the worst case. Therefore, the total worst-case inference complexity of XCSF is

$$
T _ { \mathrm { X C S F } } = O ( n R _ { \mathrm { X } } ) .\tag{5}
$$

## B. KACS

KACS evaluates $n ( 2 n + 1 )$ inner submodels and $2 n + 1$ outer submodels. Thus, it contains

$$
n ( 2 n + 1 ) + ( 2 n + 1 ) = ( n + 1 ) ( 2 n + 1 ) = 2 n ^ { 2 } + 3 n + 1\tag{6}
$$

submodels.

Each KACS rule has a one-dimensional interval condition and a two-parameter linear consequent. Matching and evaluating an individual rule therefore require $O ( 1 )$ operations. Scanning and evaluating all inner and outer rule populations require

$$
{ \cal O } ( R _ { \mathrm { K } } ) .\tag{7}
$$

KACS additionally computes the intermediate sums

$$
\hat { z } _ { q } = \sum _ { p = 1 } ^ { n } \hat { \psi } _ { q , p } ( x _ { p } ) , \qquad q = 1 , \ldots , 2 n + 1 ,\tag{8}
$$

which requires $n ( 2 n + 1 )$ additions. It subsequently aggregates the $2 n + 1$ outer outputs to obtain ˆ�. The total channel-wise aggregation cost is therefore

$$
O ( n ( 2 n + 1 ) ) = O \left( n ^ { 2 } \right) .\tag{9}
$$

Combining rule evaluation and channel-wise aggregation yields

$$
T _ { \mathrm { K A C S } } = O ( R _ { \mathrm { K } } + n ( 2 n + 1 ) ) = O \Bigl ( R _ { \mathrm { K } } + n ^ { 2 } \Bigr ) .\tag{10}
$$

## C. Comparison of Inference Complexity

For the final learned populations, the worst-case inference complexities are

$$
T _ { \mathrm { X C S F } } = O \left( n R _ { \mathrm { X } } \right) , \qquad T _ { \mathrm { K A C S } } = O \left( R _ { \mathrm { K } } + n ^ { 2 } \right) .\tag{11}
$$

Thus, KACS removes the multiplicative factor � from the cost of matching and evaluating an individual rule. However, it evaluates inner and outer submodels across multiple channels and consequently incurs a channel-wise aggregation cost of $O ( n ^ { 2 } )$ . Hence, this analysis does not imply that KACS is always faster in wall-clock inference time. The practical runtime also depends on $R _ { \mathrm { X } } , R _ { \mathrm { K } }$ , the allocation of KACS rules among submodels, and implementation-specific factors.

## D. Asymptotic Rule-Count Interpretation

The preceding analysis is expressed in terms of the final numbers of macro-rules, $R _ { X }$ and $R _ { \mathrm { K } }$ . To relate these quantities to dimensionality, consider an idealized setting in which � effective intervals are required per input dimension for XCSF and, on average, � rules are required per one-dimensional KACS submodel.

Under this assumption, XCSF requires

$$
R _ { X } = O ( m ^ { n } )\tag{12}
$$

rules to cover an �-dimensional input space. KACS has $n ( 2 n + 1 )$ inner submodels and 2� + 1 outer submodels. Hence, its total number of rules is

$$
R _ { \mathrm { K } } = O ( m [ n ( 2 n + 1 ) + ( 2 n + 1 ) ] ) = O ( m ( n + 1 ) ( 2 n + 1 ) ) = O ( m n ^ { 2 } ) .\tag{13}
$$

Substituting these estimates into Eqs. (5) and (10) gives

$$
T _ { \mathrm { X C S F } } = O ( n m ^ { n } ) , \qquad T _ { \mathrm { K A C S } } = O ( m n ^ { 2 } ) .\tag{14}
$$

These expressions describe asymptotic scaling under the stated idealized rule-allocation assumption. They do not represent the numbers of rules learned in a finite experiment, which depend on the population budget �, the data distribution, and the learning dynamics.

## S2. GENERALIZATION OF THE KACS UNIVERSAL APPROXIMATION THEOREM TO OTHER CONSEQUENT MODELS

Although the proof in Section V of the main article is presented with a linear consequent $P _ { k } ( z ) = w _ { k , 0 } + w _ { k , 1 } z$ as in (3), the same argument extends to a broader class of consequent models. In particular, KACS retains the universal approximation property even when the consequent is chosen as (i) polynomial or B-spline models that can represent linear functions exactly [1], or (ii) universal approximators such as MLP, RBFN [2], and KAN.

This follows because the construction in Lemma 1 (see also (25)) of the main article only requires the consequent of each reference rule $c l _ { j } ^ { * } .$ , denoted by $P _ { j } ^ { * } ( z )$ , to approximate the target linear function $h _ { j } ( z )$ on the corresponding region $R _ { j }$ with error at most $\varepsilon / 4$ . A linear consequent can represent $h _ { j }$ exactly, and polynomial or B-spline consequents can do so as well because they include linear functions as a special case. If a universal approximator (MLP, RBFN, KAN) is used as the consequent, it can approximate $h _ { j }$ within $\varepsilon / 4$ , which is sufficient for the same error bound to carry through.

Therefore, the universal approximation capability of KACS is not limited to linear consequents; it generalizes to more flexible consequent models that can represent linear functions either exactly or to arbitrary accuracy.

## S3. COMPARISON WITH SUPRB: A STATE-OF-THE-ART PITTSBURGH-STYLE LCS

## A. Experimental Protocol and Comparability

We conducted an additional comparison with SupRB, a Pittsburgh-style LCS that separately optimizes rule discovery and final rule-set composition [3], [4]. The comparison includes the four real-world benchmarks used in the main article and two additional benchmarks considered by Heider [4]: the Physicochemical Properties of Protein Tertiary Structure (PPPTS) dataset with � = 9 and the Parkinsons Telemonitoring (PT) dataset with $n = 1 8 .$ . The latter two extend the dimensionality of the real-world evaluation beyond the original maximum of � = 8. They should be interpreted as higher-dimensional relative to the original benchmark set, rather than as extremely high-dimensional data.

For our XCSF and KACS experiments, we followed Heider’s evaluation protocol [4]: eight Monte Carlo data splits were combined with eight random seeds per split, yielding 64 runs for each method and dataset; 25% of the samples were held out for testing; and the target was standardized. XCSF and KACS used the same splits and seeds. The input features were scaled to [0, 1] instead of the [−1, 1] range used by Heider because the present KACS formulation assumes inputs in [0, 1]. Our XCSF and KACS populations were evaluated without post-training compaction. For PT (� = 18) only, we set the population budget to � = 15984 for both XCSF and KACS, following the practical guideline in Section VI-C4 of the main article and retaining an equal budget for the controlled comparison. For all other datasets, we used $N = 6 4 0 0$ , as in Section VI-A3 of the main article.

The literature values require several qualifications. Heider’s reported XCSF results for ASN, CCPP, CS, and EEC use dataset-specific hyperparameter tuning, recursive-least-squares consequent updates, and post-training population compaction; XCSF results were not reported for PPPTS or PT. For SupRB, we use the evolution-strategy rule-discovery configuration, denoted SupRB-ES, because this is the reported configuration for which results are available on all six datasets. Its rule count is the number of rules in the final selected solution. The reported XCSF and SupRB-ES values were obtained in separate experimental campaigns and are therefore included as literature reference points, not as paired observations from our runs. Consequently, the tables are used for descriptive comparison only; we do not report cross-study statistical tests or average ranks.

## B. Prediction Accuracy

Table S1 compares testing MSE in the standardized target space. Values for all methods are reported as the mean and standard deviation over 64 runs.

On the four benchmarks used in the main article, KACS obtains lower MSE than our uncompacted XCSF in every case: 0.14 vs. 0.32 on ASN, 0.06 vs. 0.07 on CCPP, 0.14 vs. 0.21 on CS, and 0.06 vs. 0.08 on EEC. Its MSE is also close to the literature reference values on ASN, CCPP, and CS, whereas the reported XCSF and SupRB-ES results are lower on EEC. These observations indicate that the favorable KACS–XCSF comparison is retained under the 64-run, standardized-target protocol, but they do not establish superiority over independently reported methods.

The two additional datasets give a more mixed result. On PPPTS, XCSF and KACS both obtain an MSE of 0.74, compared with 0.63 for SupRB-ES. On PT, our XCSF obtains the lowest MSE (0.09), followed by KACS (0.21) and SupRB-ES (0.31). Thus, the higher-dimensional real-world experiments do not support a claim that KACS uniformly provides the best predictive accuracy. Instead, they show that its relative performance depends on the dataset: KACS remains competitive on PT but does not improve upon XCSF on PPPTS.

Descriptively, KACS records lower mean testing MSE than the reported SupRB-ES values on ASN, CCPP, CS, and PT, whereas SupRB-ES records lower values on EEC and PPPTS. These results suggest that KACS can achieve predictive performance competitive with SupRB-ES. However, because the SupRB-ES values were obtained in independent experiments, this comparison should not be interpreted as evidence of statistical superiority.

TABLE S1  
TESTING MSE ON THE SIX REAL-WORLD REGRESSION BENCHMARKS. VALUES ARE MEAN ± STANDARD DEVIATION OVER 64 RUNS. OUR XCSF AND KACS RESULTS WERE OBTAINED WITHOUT COMPACTION. THE LITERATURE VALUES ARE REPORTED BY HEIDER; SUPRB-ES USES EVOLUTION-STRATEGY RULE DISCOVERY. N/A INDICATES THAT NO CORRESPONDING XCSF RESULT WAS REPORTED. THE CROSS-STUDY RESULTS ARE PRESENTED DESCRIPTIVELY, WITHOUT STATISTICAL TESTS OR AVERAGE RANKS
<table><tr><td rowspan="2">Benchmark #Samples</td><td rowspan="2">n</td><td rowspan="2"></td><td colspan="2">Ours (no compaction)</td><td colspan="2">Reported by [4]</td></tr><tr><td>XCSF</td><td>KACS</td><td></td><td>XCSF (after compaction) SupRB-ES (final solution)</td></tr><tr><td>ASN</td><td>1503</td><td>5</td><td> $0 . 3 2 \pm 0 . 0 8$ </td><td> $0 . 1 4 \pm \ : 0 . 0 2$ </td><td> $\overline { { 0 . 1 2 \pm 0 . 1 6 } }$ </td><td> $0 . 1 5 \pm 0 . 0 2$ </td></tr><tr><td>CCPP</td><td>9568</td><td>4</td><td> $0 . 0 7 \pm 0 . 0 0$ </td><td> $0 . 0 6 \pm 0 . 0 1$ </td><td> $0 . 0 6 \pm 0 . 0 0$ </td><td> $0 . 0 7 \pm 0 . 0 0$ </td></tr><tr><td>CS</td><td>1030</td><td>8</td><td> $0 . 2 1 \pm 0 . 0 5$ </td><td> $0 . 1 4 \pm \ : 0 . 0 3$ </td><td> $0 . 1 7 \pm 0 . 1 3$ </td><td> $0 . 1 5 \pm 0 . 0 4$ </td></tr><tr><td>EEC</td><td>768</td><td>8</td><td> $0 . 0 8 \pm 0 . 0 2$ </td><td> $0 . 0 6 \pm \ : 0 . 0 2$ </td><td> $0 . 0 2 \pm 0 . 0 2$ </td><td> $0 . 0 4 \pm \ : 0 . 0 3$ </td></tr><tr><td>PPPTS</td><td>45739</td><td>9</td><td> $0 . 7 4 \pm 0 . 0 1$ </td><td> $0 . 7 4 \pm \ : 0 . 0 4$ </td><td>N/A</td><td> $0 . 6 3 \pm 0 . 0 2$ </td></tr><tr><td>PT</td><td>5875</td><td>18</td><td> $0 . 0 9 \pm 0 . 0 2$ </td><td> $0 . 2 1 \pm 0 . 0 7$ </td><td>N/A</td><td> $0 . 3 1 \pm 0 . 0 3$ </td></tr></table>

TABLE S2  
NUMBERS OF RULES ON THE SIX REAL-WORLD REGRESSION BENCHMARKS. NOTATION FOLLOWS TABLE S1
<table><tr><td rowspan="2">#Samples</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">Ours (no compaction)</td><td colspan="2">Reported by [4]</td></tr><tr><td>XCSF</td><td>KACS</td><td>XCSF (after compaction)</td><td>SupRB-ES (final solution)</td></tr><tr><td>ASN</td><td>1503</td><td>5</td><td> $2 0 0 0 \pm 4 1 0 . 8$ </td><td> $\overline { { 1 4 3 4 \pm 4 0 . 3 8 } }$ </td><td> $\overline { { 1 6 1 7 . 5 8 \pm 4 1 3 . 1 2 } }$ </td><td> $\overline { { 4 3 . 3 \pm 2 . 8 } }$ </td></tr><tr><td>CCPP</td><td>9568</td><td>4</td><td> $1 8 9 8 \pm 2 9 7 . 1$ </td><td> $1 9 2 7 \pm 4 5 . 2 6$ </td><td> $1 9 2 2 . 7 1 \pm 3 9 0 . 7$ </td><td> $5 . 5 \pm 0 . 8$ </td></tr><tr><td>CS</td><td>1030</td><td>8</td><td> $2 8 3 7 \pm 4 3 4 . 5$ </td><td> $1 4 1 6 \pm 4 2 . 3 0$ </td><td> $4 8 1 . 6 8 \pm 3 3 6 . 3 3$ </td><td> $4 2 . 1 \pm 2 . 9$ </td></tr><tr><td>EEC</td><td>768</td><td>8</td><td> $2 9 8 5 \pm 3 9 1 . 4$ </td><td> $1 0 1 4 \pm 3 5 . 3 4$ </td><td> $7 0 7 . 9 4 \pm 2 8 2 . 1 6$ </td><td> $2 0 . 1 \pm 2 . 2$ </td></tr><tr><td>PPPTS</td><td>45739</td><td>9</td><td> $3 0 0 1 \pm 5 7 8 . 4$ </td><td> $1 4 0 5 \pm 3 5 . 9 6$ </td><td>N/A</td><td> $4 5 . 7 \pm 3 . 6$ </td></tr><tr><td>PT</td><td>5875</td><td>18</td><td> $4 4 2 2 \pm 9 9 9 . 9$ </td><td> $3 5 6 4 \pm 8 9 . 6 4$ </td><td>N/A</td><td> $3 8 . 8 \pm 3 . 3$ </td></tr></table>

## C. Rule Complexity

Table S2 reports the corresponding numbers of rules.

KACS uses fewer rules than our uncompacted XCSF on five of the six datasets; CCPP is the sole exception, where the counts are similar (1927 vs. 1898). The reduction is particularly pronounced on CS, EEC, and PPPTS. Nevertheless, the results also qualify the compactness claim made from the controlled KACS–XCSF comparison. The compacted XCSF models reported by Heider use fewer rules than KACS on CS and EEC, and SupRB-ES uses substantially fewer rules on every dataset, with only 5.5–45.7 rules on average. KACS should therefore be described as compact relative to the uncompacted XCSF configuration evaluated in our controlled experiments, not as the most compact LCS among the methods considered here.

Raw rule counts also represent different model structures. SupRB-ES selects a small final set of multidimensional rules, whereas KACS distributes one-dimensional rules among $( n + 1 ) ( 2 n + 1 ) = 2 n ^ { 2 } + 3 n + 1$ inner and outer submodels. This fixed decomposition imposes a structural cost because every submodel must maintain input coverage. Moreover, Heider did not report a directly comparable total parameter count for SupRB-ES. We therefore cannot infer a precise parameter-coun ordering from Table S2, although its much smaller final rule sets clearly demonstrate greater rule-set compactness.

## D. Interpretability of the Learned Models

A KACS population containing approximately 1000 or more macro-rules should not be regarded as a directly human-readable rule list. Its structure is nevertheless different from that of a flat XCSF population. Across the six datasets, KACS distributes its rules among 45 submodels for CCPP, 66 for ASN, 153 for CS and EEC, 190 for PPPTS, and 703 for PT. The observed totals therefore correspond to approximately 5.1–42.8 one-dimensional rules per submodel. Each submodel can be plotted and inspected as a one-dimensional local function, which provides structured inspectability that is not apparent from the total rule count alone.

This decomposition does not, however, guarantee global interpretability. A complete KACS prediction depends on the composition of all inner and outer submodels across 2� + 1 channels. Understanding an individual one-dimensional submodel is considerably easier than reading thousands of multidimensional XCSF rules, but understanding the complete mapping still requires tracing many interacting local functions. In this sense, KACS is better characterized as a structured and visualizable model than as a globally transparent model.

SupRB-ES addresses interpretability more directly by selecting a small final set of multidimensional rules. Its 5.5–45.7-rule solutions are far closer to a model that can be inspected rule by rule than either the XCSF or KACS populations reported here. Accordingly, we do not claim that KACS is more globally interpretable than SupRB. The present results support parameter efficiency and structured inspectability relative to uncompacted XCSF, while SupRB-ES provides substantially greater final rule-set compactness. A formal comparison of interpretability—including visualization of learned KACS submodels, fidelit of local explanations, human-subject evaluation, and the effects of pruning or compaction—is left for future work.

## S4. FORMULATIONS OF SYNTHETIC FUNCTIONS

This section summarizes the synthetic benchmark functions used in the main article. Fig. S1 illustrates each function for $n = 2 .$ . The four synthetic benchmark functions are:

• Rastrigin function $( f _ { 1 }$ , cf. Fig. S1a): a highly multimodal landscape with many local minima, defined as:

$$
f _ { 1 } ( \mathbf { x } ) = 1 0 n + \sum _ { i = 1 } ^ { n } \left( x _ { i } ^ { 2 } - 1 0 \cos \left( 2 \pi x _ { i } \right) \right) .\tag{15}
$$

• Rosenbrock function (�<sub>2</sub>, cf. Fig. S1b): a narrow curved valley with strong interactions between adjacent variables, defined as:

$$
f _ { 2 } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { n - 1 } \left[ 1 0 0 \left( x _ { i + 1 } - x _ { i } ^ { 2 } \right) ^ { 2 } + ( x _ { i } - 1 ) ^ { 2 } \right] .\tag{16}
$$

• Cross function $( f _ { 3 } ,$ cf. Fig. S1c): axis-aligned ridge-like components along each coordinate axis and a central Gaussian peak, with interactions across all dimensions, defined as:

$$
a = \frac { 1 } { \lfloor n / 2 \rfloor } \sum _ { i = 1 } ^ { \lfloor n / 2 \rfloor } x _ { i } , b = \frac { 1 } { \lceil n / 2 \rceil } \sum _ { i = \lfloor n / 2 \rfloor + 1 } ^ { n } x _ { i } ,\tag{17}
$$

$$
f _ { 3 } ( { \bf x } ) = \mathrm { m a x } \left( \exp ( - 1 0 a ^ { 2 } ) , \exp ( - 5 0 b ^ { 2 } ) , 1 . 2 5 \exp \Bigl ( - 5 ( a ^ { 2 } + b ^ { 2 } ) \Bigr ) \right) .\tag{18}
$$

• Styblinski-Tang function $( f _ { 4 } .$ , cf. Fig. S1d): a separable nonconvex polynomial surface with multiple local minima per dimension, defined as:

$$
f _ { 4 } ( \mathbf { x } ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { n } \left( x _ { i } ^ { 4 } - 1 6 x _ { i } ^ { 2 } + 5 x _ { i } \right) .\tag{19}
$$

These functions collectively cover multimodality (�<sub>1</sub>), inter-variable curvature $( f _ { 2 } )$ , anisotropic structure $( f _ { 3 } )$ , and separable nonconvexity (� ).

![](images/29f311676886c6d41c75cd4bfa338a3fdc69afa0c071ee85ee35f79130ba83bd.jpg)  
(a) �<sub>1</sub>: Rastrigin Function

![](images/aee5c1878b35d615102262f723b0997fe3cc0b80c7db38b7fb2002339e59acb9.jpg)  
(b) �<sub>2</sub>: Rosenbrock Function

![](images/2f714a33577af4374f99dd92f72c82e1ad7af8a5ce0f7778a7d6a40e65c278a2.jpg)  
(c) �<sub>3</sub>: Cross Function

![](images/63e9702735295c06f92b27db1fd68bcdd41092071eeec55166bc1ca31b00419a.jpg)  
(d) �<sub>4</sub>: Styblinski-Tang Function  
Fig. S1. Surface plots of the four synthetic test functions $( n = 2 )$ shown for visualization (same as Fig. 2 of the main article).

## S5. A DESCRIPTION OF HYPERPARAMETERS

The following hyperparameters are commonly used in both XCSF and KACS.

• �: The maximum population size $\mathcal { P }$ (in micro-rules).

• � : The error threshold; rules with prediction error below $\epsilon _ { 0 }$ are considered accurate. A suitable tolerance can improve robustness to noise-induced error fluctuations and avoid unnecessary specialization, whereas an excessively large value can reduce approximation accuracy [5].

• $\beta \colon$ The learning rate for updating prediction error, fitness, and match set size estimates.

• �: The fall-off coefficient in the accuracy calculation; controls the accuracy assigned when the prediction error is at or above $\epsilon _ { 0 }$

• �: The power parameter for the accuracy calculation; controls the steepness of the accuracy fall-off.

• �: The fraction of the mean fitness used as the threshold for rule deletion.

• �<sub>0</sub>: The maximum mutation range for interval parameters.

• $r _ { 0 } \colon$ The maximum spread for the covering operator.

• $\chi \colon$ The crossover probability in the GA.

• $\mu { : }$ The mutation probability per allele in the GA.

• �: The tournament size ratio for parent selection.

$\theta _ { \mathrm { G A } } { \mathrm { : } }$ The GA application threshold.

$\theta _ { \mathrm { d e l } } \colon$ The experience threshold for rule deletion.

$\theta _ { \mathrm { s u b } } \colon$ The experience threshold for subsumption.

• $P \# \colon$ The probability of generating a [0, 1] interval during covering.

• doSubsumption: A Boolean flag to enable GA subsumption.

## S6. DETAILED STATISTICAL RESULTS

This section presents run-level statistics that complement Table II in the main article. For each benchmark and metric, Table S3 reports the mean and standard deviation over the same 30 train–test splits, together with the two-sided Wilcoxon signed-rank �-value and matched-pairs rank-biserial correlation. The effect size is computed from the paired differences XCSF−KACS; positive values favor KACS and negative values favor XCSF because lower values are preferable for every metric.

KACS shows its most consistent advantage in model complexity. Its parameter count and AIC are lower on all eight benchmarks, with an effect size of 1 in every comparison. Its rule count is also lower with an effect size of 1 on seven benchmarks; on CCPP, the rule counts are similar and the effect is small (0.0796). This pattern reflects the KA decomposition, which replaces full �-dimensional local models with one-dimensional submodels.

Accuracy depends more strongly on problem characteristics. For testing MAE, KACS shows positive effects on the four nonlinear synthetic functions (0.7462–1) and ASN (0.9484), supporting its advantage on problems with pronounced nonlinearity or cross-dimensional structure. On $f _ { 1 }$ , the effect favors XCSF for training MAE (−0.5226) but KACS for testing MAE (0.7677), which, together with the large parameter-count difference, is consistent with greater overfitting by XCSF. The testing-MAE effects are smaller on CCPP (0.2430) and CS (0.2172), while EEC favors XCSF (−0.3462). For problems that are adequately represented by simpler linear models, the coordination of KACS’s inner and outer submodels can introduce additional approximation overhead, although KACS remains substantially more compact.

The standard deviations show that stability is metric dependent. KACS has lower standard deviations in rule count, parameter count, and AIC on all benchmarks. For testing MAE, however, KACS is less variable on ASN (0.0149 vs. 0.0222) and CS (0.0169 vs. 0.0231), but more variable on the synthetic functions, CCPP, and EEC. In the small-MAE cases, the standard deviations are 0.0029 for XCSF and 0.0073 for KACS on CCPP, and 0.0223 and 0.0314, respectively, on EEC. Thus, KACS provides consistently stable model complexity but does not uniformly reduce variability in prediction accuracy. All MAE statistics use the normalized target scale of the main experiments.

TABLE S3  
DETAILED PAIRED COMPARISON OF XCSF AND KACS OVER 30 MATCHED RUNS FOR EACH BENCHMARK AND METRIC. VALUES ARE REPORTED AS THE MEAN ± STANDARD DEVIATION. GREEN CELLS INDICATE THE LOWER MEAN FOR EACH BENCHMARK AND METRIC. THE WILCOXON COLUMN REPORTS THE TWO-SIDED WILCOXON SIGNED-RANK �-VALUE COMPUTED FROM THE 30 PAIRED RUNS. THE EFFECT SIZE IS THE MATCHED-PAIRS RANK-BISERIAL CORRELATION COMPUTED FROM THE PAIRED DIFFERENCES (XCSF MINUS KACS); POSITIVE VALUES INDICATE THAT KACS TENDED TO PRODUCE LOWER VALUES, WHEREAS NEGATIVE VALUES INDICATE THAT XCSF TENDED TO PRODUCE LOWER VALUES
<table><tr><td>Benchmark</td><td>Metric</td><td>XCSF mean ± SD</td><td>KACS mean ± SD</td><td>Wilcoxon p</td><td>Effect size</td></tr><tr><td>f1</td><td>Training MAE</td><td> $\overline { { 0 . 2 0 3 3 \pm 0 . 0 1 2 2 } }$ </td><td> $\overline { { 0 . 2 1 9 4 \pm 0 . 0 2 9 1 } }$ </td><td>0.0113</td><td>-0.5226</td></tr><tr><td>fi</td><td>Testing MAE</td><td> $0 . 2 6 4 2 \pm 0 . 0 2 2 0$ </td><td> $0 . 2 3 5 1 \pm 0 . 0 2 5 0$ </td><td>8.86E-05</td><td>0.7677</td></tr><tr><td>fi</td><td>#Rules</td><td> $4 1 6 2 \pm 5 7 . 5 9$ </td><td> $1 3 4 1 \pm 3 5 . 0 7$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>f</td><td>#Parameters</td><td> $4 5 7 8 1 \pm 6 3 3 . 5$ </td><td> $2 6 8 1 \pm 7 0 . 1 3$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>fi</td><td>AIC</td><td> $8 9 2 0 7 \pm 1 2 6 6$ </td><td> $3 0 2 4 \pm 2 6 2 . 6$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>£</td><td>Training MAE</td><td> $0 . 1 1 2 4 \pm 0 . 0 1 1 9$ </td><td> $\overline { { 0 . 1 1 5 7 \pm 0 . 0 2 8 0 } }$ </td><td>0.5838</td><td>0.1183</td></tr><tr><td></td><td>Testing MAE</td><td> $0 . 1 5 6 4 \pm 0 . 0 1 9 5$ </td><td> $0 . 1 2 5 3 \pm 0 . 0 2 7 4$ </td><td>0.0001529</td><td>0.7462</td></tr><tr><td> $f _ { 2 }$ </td><td>#Rules</td><td> $4 0 8 7 \pm 5 6 . 8 8$ </td><td> $1 4 6 2 \pm 3 7 . 6 3$ </td><td>1.82E-06</td><td>1</td></tr><tr><td> $f _ { 2 }$ </td><td>#Parameters</td><td> $4 4 9 6 0 \pm 6 2 5 . 6$ </td><td> $2 9 2 4 \pm 7 5 . 2 7$ </td><td>1.82E-06</td><td>1</td></tr><tr><td> $f _ { 2 }$ </td><td>AIC</td><td> $8 6 5 7 9 \pm 1 1 9 3$ </td><td> $2 3 4 2 \pm 3 1 4 . 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>f3</td><td>Training MAE</td><td> $\overline { { 0 . 2 5 1 0 \pm 0 . 0 1 6 7 } }$ </td><td> $0 . 2 0 9 7 \pm 0 . 0 3 9 5$ </td><td>4.42E-06</td><td>0.8667</td></tr><tr><td> $f _ { 3 }$ </td><td>Testing MAE</td><td> $0 . 3 2 4 6 \pm 0 . 0 2 5 0$ </td><td> $0 . 2 3 2 9 \pm 0 . 0 5 0 8$ </td><td>3.73E-09</td><td>0.9957</td></tr><tr><td> $f _ { 3 }$ </td><td>#Rules</td><td> $4 1 3 0 \pm 7 4 . 0 8$ </td><td> $1 4 2 9 \pm 3 6 . 9 4$ </td><td>1.82E-06</td><td>1</td></tr><tr><td> $f _ { 3 }$ </td><td>#Parameters</td><td> $4 5 4 2 6 \pm 8 1 4 . 9$ </td><td> $2 8 5 7 \pm 7 3 . 8 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td> $f _ { 3 }$ </td><td>AIC</td><td> $8 8 8 9 2 \pm 1 5 8 0$ </td><td> $3 3 2 4 \pm 3 4 9 . 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td></td><td>Training MAE</td><td> $\overline { { 0 . 1 6 8 0 \pm 0 . 0 1 5 0 } }$ </td><td> $0 . 1 3 3 4 \pm 0 . 0 3 0 1$ </td><td>2.08E-05</td><td>0.8194</td></tr><tr><td></td><td>Testing MAE</td><td> $0 . 2 3 0 3 \pm 0 . 0 1 8 3$ </td><td> $0 . 1 4 3 0 \pm 0 . 0 2 9 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>4 4 </td><td>#Rules</td><td> $4 1 7 0 \pm 6 1 . 1 1$ </td><td> $1 4 1 4 \pm 3 5 . 2 7$ </td><td>1.82E-06</td><td>1</td></tr><tr><td></td><td>#Parameters</td><td> $4 5 8 7 3 \pm 6 7 2 . 2$ </td><td> $2 8 2 9 \pm 7 0 . 5 4$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>f4</td><td>AIC</td><td> $8 9 0 7 8 \pm 1 3 0 0$ </td><td> $2 4 1 2 \pm 4 2 5 . 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>ASN</td><td>Training MAE</td><td>0.1434 ± 0.0203</td><td> $\overline { { 0 . 1 0 9 4 \pm 0 . 0 1 3 9 } }$ </td><td>2.05E-07</td><td>0.9398</td></tr><tr><td>ASN</td><td>Testing MAE</td><td> $0 . 1 4 7 8 \pm 0 . 0 2 2 2$ </td><td> $0 . 1 1 4 4 \pm 0 . 0 1 4 9$ </td><td>1.30E-07</td><td>0.9484</td></tr><tr><td>ASN</td><td>#Rules</td><td>2224 ± 356.7</td><td> $1 4 1 7 \pm 4 6 . 2 6$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>ASN</td><td>#Parameters</td><td> $1 3 3 4 5 \pm 2 1 4 0$ </td><td> $2 8 3 4 \pm 9 2 . 5 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>ASN</td><td>AIC</td><td> $2 2 2 7 5 \pm 3 9 5 3$ </td><td> $3 7 1 . 1 \pm 3 4 4 . 2$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>CCPP</td><td>Training MAE</td><td> $\overline { { 0 . 0 9 3 0 \pm 0 . 0 0 1 6 } }$ </td><td> $\overline { { 0 . 0 9 1 8 \pm 0 . 0 0 6 2 } }$ </td><td>0.1294</td><td>0.3204</td></tr><tr><td>CCPP</td><td>Testing MAE</td><td> $0 . 0 9 2 9 \pm 0 . 0 0 2 9$ </td><td> $0 . 0 9 2 6 \pm 0 . 0 0 7 3$ </td><td>0.2534</td><td>0.243</td></tr><tr><td>CCPP</td><td>#Rules</td><td> $1 9 5 3 \pm 2 9 2 . 8$ </td><td> $1 9 2 4 \pm 4 8 . 1 4$ </td><td>0.7112</td><td>0.0796</td></tr><tr><td>CCPP</td><td>#Parameters</td><td> $9 7 6 4 \pm 1 4 6 4$ </td><td> $3 8 4 8 \pm 9 6 . 2 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>CCPP</td><td>AIC</td><td> $- 1 7 3 5 8 \pm 2 8 2 3$ </td><td> $- 2 9 0 6 5 \pm 1 1 9 4$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>CS</td><td>Training MAE</td><td> $\overline { { 0 . 1 2 9 9 \pm 0 . 0 1 9 8 } }$ </td><td> $0 . 1 2 1 3 \pm 0 . 0 1 3 7$ </td><td>0.1191</td><td>0.329</td></tr><tr><td>CS</td><td>Testing MAE</td><td> $0 . 1 3 5 0 \pm 0 . 0 2 3 1$ </td><td> $0 . 1 2 8 8 \pm 0 . 0 1 6 9$ </td><td>0.3085</td><td>0.2172</td></tr><tr><td>CS</td><td>#Rules</td><td> $2 9 8 5 \pm 3 9 3 . 1$ </td><td> $1 4 2 5 \pm 3 4 . 5 8$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>CS</td><td>#Parameters</td><td> $2 6 8 6 8 \pm 3 5 3 8$ </td><td> $2 8 5 1 \pm 6 9 . 1 6$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>CS</td><td>AIC</td><td> $5 0 4 5 3 \pm 6 8 6 0$ </td><td> $2 2 3 7 \pm 2 2 9 . 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>EEC</td><td>Training MAE</td><td> $0 . 0 8 2 0 \pm 0 . 0 2 0 2$ </td><td> $\overline { { 0 . 0 9 6 7 \pm 0 . 0 3 2 0 } }$ </td><td>0.1048</td><td>-0.3419</td></tr><tr><td>EEC</td><td>Testing MAE</td><td> $0 . 0 8 4 7 \pm 0 . 0 2 2 3$ </td><td> $0 . 1 0 2 0 \pm 0 . 0 3 1 4$ </td><td>0.1004</td><td>-0.3462</td></tr><tr><td>EEC</td><td>#Rules</td><td> $2 9 2 5 \pm 3 7 3 . 6$ </td><td> $1 0 1 2 \pm 3 1 . 5 1$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>EEC</td><td>Parameters</td><td> $2 6 3 2 6 \pm 3 3 6 3$ </td><td> $2 0 2 3 \pm 6 3 . 0 2$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>EEC</td><td>AIC</td><td> $4 9 8 4 9 \pm 6 5 6 7$ </td><td> $1 1 2 5 \pm 3 0 2 . 7$ </td><td>1.86E-09</td><td>1</td></tr></table>

## S7. ROBUSTNESS UNDER AN ALTERNATIVE EVALUATION PROTOCOL

## A. Experimental Protocol

To examine whether the controlled XCSF–KACS comparison depends on the evaluation protocol used in the main article, we repeated the experiments on all eight benchmark problems using a protocol based on Heider’s evaluation design [3], [4]. The main experiments use 30 Monte Carlo splits with 90% of the samples for training, min–max-normalized targets, and MAE. In this alternative protocol, we instead used eight Monte Carlo splits and eight random seeds per split, yielding 64 matched runs for each method and problem. For each split, 75% of the samples were used for training and 25% for testing, the target was standardized, and prediction accuracy was evaluated using MSE.

The target standardization parameters were estimated from the training split and then applied to the corresponding test split. The synthetic inputs retained their prescribed [0, 1]<sup>�</sup> domains. For the real-world datasets, input-scaling parameters were estimated from the training split only and used to map the inputs to $[ 0 , 1 ] ^ { n } ;$ ; the same transformations were then applied to the test split. This differs from Heider’s input range of $[ - 1 , 1 ] ^ { n }$ because the present KACS formulation assumes inputs in [0, 1]<sup>�</sup>. Thus, this experiment follows the principal elements of Heider’s evaluation design but is not an exact reproduction of all preprocessing choices

XCSF and KACS used identical data splits and random seeds. Their hyperparameters, population budget $( N = 6 4 0 0 )$ learning duration, and absence of post-training compaction were otherwise unchanged from the main experiments. Learning was disabled when evaluating the held-out test samples.

## B. Main Results

Table S4 summarizes prediction accuracy, model complexity, and AIC under the alternative protocol. KACS obtains lower mean training and testing MSE than XCSF on all eight benchmark problems. The advantage is especially pronounced on the four synthetic functions, ASN, and CS, while the difference is smaller but remains statistically significant on CCPP and EEC. The across-problem paired Wilcoxon signed-rank test applied to the 64-run mean for each benchmark also favors KACS for both training and testing MSE $( p = 0 . 0 0 7 8 1 3 )$ ).

KACS also uses fewer macro-rules on seven of the eight problems. CCPP is the only exception: XCSF uses 1898 rules on average and KACS uses 1927, with no significant difference between them. Despite these similar rule counts, KACS uses substantially fewer parameters on CCPP because every KACS rule has only two consequent parameters. Across all eight problems, KACS uses only approximately 5.9%–40.6% as many parameters as XCSF and obtains a lower AIC. The across-problem tests favor KACS for rule count $( p = 0 . 0 1 5 6 3 )$ , parameter count $( p = 0 . 0 0 7 8 1 3 )$ , and AIC $( p = 0 . 0 0 7 8 1 3 )$ .

TABLE S4  
SUMMARY OF THE XCSF–KACS COMPARISON UNDER THE ALTERNATIVE EVALUATION PROTOCOL. VALUES ARE MEANS OVER 64 MATCHED RUNS CONSISTING OF EIGHT MONTE CARLO SPLITS AND EIGHT RANDOM SEEDS PER SPLIT. TRAINING AND TESTING ERRORS ARE MSE IN THE STANDARDIZED TARGET SPACE. GREEN CELLS INDICATE THE LOWER MEAN. FOR EACH PROBLEM, SYMBOLS +/−/∼ DENOTE THAT XCSF IS SIGNIFICANTLY BETTER/WORSE/SIMILAR TO KACS BASED ON A TWO-SIDED PAIRED WILCOXON SIGNED-RANK TEST. RANK IS THE AVERAGE RANK ACROSS THE EIGHT PROBLEMS. THE �-VALUES IN THE FINAL ROW ARE FROM ACROSS-PROBLEM PAIRED WILCOXON SIGNED-RANK TESTS APPLIED TO THE 64-RUN MEAN VALUES. STATISTICAL SIGNIFICANCE IS AT � = 0.05 (†)
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=4>MODEL ACCURACYTraining MSE       Testing MSE</td><td rowspan=1 colspan=4>MODEL COMPLEXITY#Rules (|P|)      #Parameters (k)</td><td rowspan=2 colspan=2>TRADE-OFFAICXCSF   KACS</td></tr><tr><td rowspan=1 colspan=2>XCSF   KACS</td><td rowspan=1 colspan=2>XCSF   KACS</td><td rowspan=1 colspan=2>XCSF  KACS</td><td rowspan=1 colspan=2>XCSF   KACS</td></tr><tr><td rowspan=1 colspan=1>f1</td><td rowspan=1 colspan=1>0.6028-</td><td rowspan=1 colspan=1>0.5683</td><td rowspan=1 colspan=1>0.8825 -</td><td rowspan=1 colspan=1>0.6655</td><td rowspan=1 colspan=1>4198-</td><td rowspan=1 colspan=1>1356</td><td rowspan=1 colspan=2>46175-   2712</td><td rowspan=1 colspan=1>91970-</td><td rowspan=1 colspan=1>4999</td></tr><tr><td rowspan=1 colspan=1>f2</td><td rowspan=1 colspan=1>0.4405-</td><td rowspan=1 colspan=1>0.2622</td><td rowspan=1 colspan=1>0.7426-</td><td rowspan=1 colspan=1>0.3226</td><td rowspan=1 colspan=1>4178-</td><td rowspan=1 colspan=1>1485</td><td rowspan=1 colspan=1>45954-</td><td rowspan=1 colspan=1>2971</td><td rowspan=1 colspan=1>91292 -</td><td rowspan=1 colspan=1>4919</td></tr><tr><td rowspan=1 colspan=1> $f _ { 3 }$ </td><td rowspan=1 colspan=1>0.5968-</td><td rowspan=1 colspan=1>0.2870</td><td rowspan=1 colspan=1>0.8707-</td><td rowspan=1 colspan=1>0.3654</td><td rowspan=1 colspan=1>4174-</td><td rowspan=1 colspan=1>1444</td><td rowspan=1 colspan=1>45919-</td><td rowspan=1 colspan=1>2888</td><td rowspan=1 colspan=1>91450-</td><td rowspan=1 colspan=1>4797</td></tr><tr><td rowspan=1 colspan=1> $f _ { 4 }$ </td><td rowspan=1 colspan=1>0.4797-</td><td rowspan=1 colspan=1>0.2223</td><td rowspan=1 colspan=1>0.7784-</td><td rowspan=1 colspan=1>0.2795</td><td rowspan=1 colspan=1>4188-</td><td rowspan=1 colspan=1>1420</td><td rowspan=1 colspan=1>46069-</td><td rowspan=1 colspan=1>2839</td><td rowspan=1 colspan=1>91583 -</td><td rowspan=1 colspan=1>4529</td></tr><tr><td rowspan=1 colspan=1>ASN</td><td rowspan=1 colspan=1>0.3203-</td><td rowspan=1 colspan=1>0.1285</td><td rowspan=1 colspan=1>0.3232 -</td><td rowspan=1 colspan=1>0.1456</td><td rowspan=1 colspan=1>2000-</td><td rowspan=1 colspan=1>1434</td><td rowspan=1 colspan=1>12000-</td><td rowspan=1 colspan=1>2868</td><td rowspan=1 colspan=1>22688-</td><td rowspan=1 colspan=1>3416</td></tr><tr><td rowspan=1 colspan=1>CCPP</td><td rowspan=1 colspan=1>0.0680-</td><td rowspan=1 colspan=1>0.0631</td><td rowspan=1 colspan=1>0.0699 -</td><td rowspan=1 colspan=1>0.0653</td><td rowspan=1 colspan=1>1898~</td><td rowspan=1 colspan=1>1927</td><td rowspan=1 colspan=1>9490-</td><td rowspan=1 colspan=1>3854</td><td rowspan=1 colspan=1>-310.4 -</td><td rowspan=1 colspan=1>-12171</td></tr><tr><td rowspan=1 colspan=1>CS</td><td rowspan=1 colspan=1>0.1955一</td><td rowspan=1 colspan=1>0.1223</td><td rowspan=1 colspan=1>0.2098-</td><td rowspan=1 colspan=1>0.1491</td><td rowspan=1 colspan=1>2837 -</td><td rowspan=1 colspan=1>1416</td><td rowspan=1 colspan=1>25535 -</td><td rowspan=1 colspan=1>2831</td><td rowspan=1 colspan=1>49788-</td><td rowspan=1 colspan=1>4034</td></tr><tr><td rowspan=1 colspan=1>EEC</td><td rowspan=1 colspan=1>0.0770-</td><td rowspan=1 colspan=1>0.0510</td><td rowspan=1 colspan=1>0.0842一</td><td rowspan=1 colspan=1>0.0593</td><td rowspan=1 colspan=1>2985 -</td><td rowspan=1 colspan=1>1014</td><td rowspan=1 colspan=2>26867-   2028</td><td rowspan=1 colspan=1>52228 -</td><td rowspan=1 colspan=1>2319</td></tr><tr><td rowspan=1 colspan=1>Rank</td><td rowspan=1 colspan=1>2.00↓†</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>2.00↓†</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.88↓†</td><td rowspan=1 colspan=1>1.12</td><td rowspan=1 colspan=2>2.00↓†    1.00</td><td rowspan=1 colspan=1>2.00↓†</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>+/-1~</td><td rowspan=1 colspan=1>0/8/0</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>0/8/0</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>0/7/1</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=2>0/8/0       -</td><td rowspan=1 colspan=1>0/8/0</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>p-value</td><td rowspan=1 colspan=1>0.007813</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.007813</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.01563</td><td rowspan=1 colspan=2>0.007813</td><td rowspan=1 colspan=2>0.007813</td></tr></table>

## C. Statistical Results and Interpretation

Table S5 provides the means, standard deviations, paired Wilcoxon �-values, and matched-pairs rank-biserial correlations for all benchmark–metric combinations. All 16 training- and testing-MSE comparisons favor KACS significantly $( p < 0 . 0 0 1 )$ . The corresponding effect sizes are positive and range from 0.5452 to 1 for training MSE and from 0.7635 to 1 for testing MSE. The parameter-count and AIC comparisons have an effect size of 1 on every benchmark. The rule-count comparisons also have an effect size of 1 except on CCPP, where the difference is nonsignificant $\begin{array} { r } { ( p = 0 . 4 4 5 8 ; } \end{array}$ effect size = −0.1101).

Compared with the MAE-based results in the main article, the alternative protocol produces a more uniform accuracy advantage for KACS. Because KACS also obtains lower training MSE on every benchmark, this pattern cannot be explained solely by improved generalization from a comparable training fit; under this protocol, KACS fits both the training and test data more accurately according to squared error. These results strengthen the evidence that the favorable KACS–XCSF comparison is robust to a different target transformation, error metric, train–test ratio, and repetition design.

Nevertheless, the larger performance difference cannot be attributed to any single protocol component because target standardization, MSE, and the train–test ratio were changed simultaneously. In particular, MSE places greater weight on large residuals than MAE, and the smaller training fraction changes the amount of data available for fitting local models. Moreover, absolute MSE and AIC values should not be compared directly with those in the main experiments because the target scale and training-set size differ. We therefore treat this experiment as a robustness analysis rather than a replacemen for the controlled main evaluation.

TABLE S5  
DETAILED PAIRED COMPARISON OF XCSF AND KACS UNDER THE ALTERNATIVE EVALUATION PROTOCOL. VALUES ARE REPORTED AS THE MEAN ± STANDARD DEVIATION OVER 64 MATCHED RUNS. OTHER NOTATION FOLLOWS TABLE S3
<table><tr><td>Benchmark</td><td>Metric</td><td>XCSF mean ± SD</td><td>KACS mean ± SD</td><td>Wilcoxon  $\mathcal { P }$ </td><td>Effect size</td></tr><tr><td> $f _ { 1 }$ </td><td>Training MSE</td><td> $\overline { { 0 . 6 0 2 8 \pm 0 . 0 5 3 3 } }$ </td><td> $0 . 5 6 8 3 \pm 0 . 0 6 4 0$ </td><td>0.0001516</td><td>0.5452</td></tr><tr><td> $f _ { 1 }$ </td><td>Testing MSE</td><td> $0 . 8 8 2 5 \pm 0 . 0 8 6 6$ </td><td> $0 . 6 6 5 5 \pm 0 . 0 8 6 6$ </td><td>5.52E-12</td><td>0.9913</td></tr><tr><td> $f _ { 1 }$ </td><td>#Rules</td><td> $4 1 9 8 \pm 6 2 . 8 5$ </td><td> $1 3 5 6 \pm 4 1 . 1 4$ </td><td>3.60E-12</td><td>1</td></tr><tr><td> $f _ { 1 }$ </td><td>#Parameters</td><td> $4 6 1 7 5 \pm 6 9 1 . 4$ </td><td> $2 7 1 2 \pm 8 2 . 2 8$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 1 }$ </td><td>AIC</td><td> $9 1 9 7 0 \pm 1 3 7 0$ </td><td> $4 9 9 9 \pm 1 7 9 . 8$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 2 }$ </td><td>Training MSE</td><td> $\overline { { 0 . 4 4 0 5 \pm 0 . 0 4 4 0 } }$ </td><td> $0 . 2 6 2 2 \pm 0 . 0 7 1 4$ </td><td>7.17E-11</td><td>0.9375</td></tr><tr><td> $f _ { 2 }$ </td><td>Testing MSE</td><td> $0 . 7 4 2 6 \pm 0 . 1 2 2 2$ </td><td> $0 . 3 2 2 6 \pm 0 . 0 8 6 4$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 2 }$ </td><td>#Rules</td><td> $4 1 7 8 \pm 6 7 . 6 2$ </td><td> $1 4 8 5 \pm 3 1 . 2 4$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 2 }$ </td><td>#Parameters</td><td> $4 5 9 5 4 \pm 7 4 3 . 9$ </td><td> $2 9 7 1 \pm 6 2 . 4 9$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 2 }$ </td><td>AIC</td><td> $9 1 2 9 2 \pm 1 4 6 3$ </td><td> $4 9 1 9 \pm 2 0 7 . 2$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 3 }$ </td><td>Training MSE</td><td> $\overline { { 0 . 5 9 6 8 \pm 0 . 0 5 5 0 } }$ </td><td>0.2870 ± 0.0984</td><td>3.79E-12</td><td>0.999</td></tr><tr><td> $f _ { 3 }$ </td><td>Testing MSE</td><td> $0 . 8 7 0 7 \pm 0 . 0 7 6 5$ </td><td> $0 . 3 6 5 4 \pm 0 . 1 2 0 1$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 3 }$ </td><td>#Rules</td><td> $4 1 7 4 \pm 5 1 . 8 9$ </td><td> $1 4 4 4 \pm 3 5 . 8 7$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 3 }$ </td><td>#Parameters</td><td> $4 5 9 1 9 \pm 5 7 0 . 8$ </td><td> $2 8 8 8 \pm 7 1 . 7 4 $ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 3 }$ </td><td>AIC</td><td> $9 1 4 5 0 \pm 1 1 2 1$ </td><td> $4 7 9 7 \pm 3 0 7 . 1$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 4 }$ </td><td>Training MSE</td><td> $0 . 4 7 9 7 \pm 0 . 0 6 1 1$ </td><td> $0 . 2 2 2 3 \pm 0 . 0 6 4 9$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 4 }$ </td><td>Testing MSE</td><td> $0 . 7 7 8 4 \pm 0 . 0 7 0 2$ </td><td> $0 . 2 7 9 5 \pm 0 . 0 7 8 3$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 4 }$ </td><td>#Rules</td><td> $4 1 8 8 \pm 6 1 . 5 9$ </td><td> $1 4 2 0 \pm 3 2 . 7 7$ </td><td>3.60E-12</td><td>1</td></tr><tr><td> $f _ { 4 }$ </td><td>#Parameters</td><td> $4 6 0 6 9 \pm 6 7 7 . 5$ </td><td> $2 8 3 9 \pm 6 5 . 5 4$ </td><td>3.61E-12</td><td>1</td></tr><tr><td> $f _ { 4 }$ </td><td>AIC</td><td> $9 1 5 8 3 \pm 1 3 3 6$ </td><td> $4 5 2 9 \pm 2 1 7 . 0$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>ASN</td><td>Training MSE</td><td>0.3203 ± 0.0720</td><td> $0 . 1 2 8 5 \pm 0 . 0 1 8 2$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>ASN</td><td>Testing MSE</td><td> $0 . 3 2 3 2 \pm 0 . 0 7 9 2$ </td><td> $0 . 1 4 5 6 \pm 0 . 0 2 4 3$ </td><td>3.97E-12</td><td>0.9981</td></tr><tr><td>ASN</td><td>#Rules</td><td>2000 ± 410.8</td><td>1434±40.38</td><td>3.61E-12</td><td>1</td></tr><tr><td>ASN</td><td>#Parameters</td><td>12000 ± 2465</td><td>2868±80.77</td><td>3.61E-12</td><td>1</td></tr><tr><td>ASN</td><td>AIC</td><td>22688 ± 4671</td><td>3416±191.6</td><td>3.61E-12</td><td>1</td></tr><tr><td>CCPP</td><td>Training MSE</td><td> $\overline { { 0 . 0 6 8 0 \pm 0 . 0 0 1 6 } }$ </td><td> $\overline { { 0 . 0 6 3 1 \pm 0 . 0 0 9 0 } }$ </td><td>1.24E-08</td><td>0.8192</td></tr><tr><td>CCPP</td><td>Testing MSE</td><td> $0 . 0 6 9 9 \pm 0 . 0 0 4 2$ </td><td> $0 . 0 6 5 3 \pm 0 . 0 0 8 6$ </td><td>3.13E-08</td><td>0.7962</td></tr><tr><td>CCPP</td><td>#Rules</td><td> $1 8 9 8 \pm 2 9 7 . 1$ </td><td> $1 9 2 7 \pm 4 5 . 2 6$ </td><td>0.4458</td><td>-0.1101</td></tr><tr><td>CCPP</td><td>#Parameters</td><td> $9 4 9 0 \pm 1 4 8 5$ </td><td> $3 8 5 4 \pm 9 0 . 5 1$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>CCPP</td><td>AIC</td><td> $- 3 1 0 . 4 \pm 2 9 1 1$ </td><td> $- 1 2 1 7 1 \pm 8 1 6 . 7$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>CS</td><td>Training MSE</td><td> $0 . 1 9 5 5 \pm 0 . 0 4 7 1$ </td><td> $0 . 1 2 2 3 \pm 0 . 0 1 8 6$ </td><td>7.50E-11</td><td>0.9365</td></tr><tr><td>CS</td><td>Testing MSE</td><td> $0 . 2 0 9 8 \pm 0 . 0 4 6 1$ </td><td> $0 . 1 4 9 1 \pm 0 . 0 2 7 1$ </td><td>5.10E-10</td><td>0.8942</td></tr><tr><td>CS</td><td>#Rules</td><td> $2 8 3 7 \pm 4 3 4 . 5$ </td><td> $1 4 1 6 \pm 4 2 . 3 0$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>CS</td><td>#Parameters</td><td> $2 5 5 3 5 \pm 3 9 1 0$ </td><td> $2 8 3 1 \pm 8 4 . 6 0$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>CS</td><td>AIC</td><td> $4 9 7 8 8 \pm 7 6 6 3$ </td><td> $4 0 3 4 \pm 1 9 7 . 8$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>EEC</td><td>Training MSE</td><td> $\overline { { 0 . 0 7 7 0 \pm 0 . 0 2 1 4 } }$ </td><td> $0 . 0 5 1 0 \pm 0 . 0 1 5 5$ </td><td>2.59E-08</td><td>0.801</td></tr><tr><td>EEC</td><td>Testing MSE</td><td> $0 . 0 8 4 2 \pm 0 . 0 2 3 6$ </td><td> $0 . 0 5 9 3 \pm 0 . 0 1 9 6$ </td><td>1.12E-07</td><td>0.7635</td></tr><tr><td>EEC</td><td>#Rules</td><td> $2 9 8 5 \pm 3 9 1 . 4$ </td><td> $1 0 1 4 \pm 3 5 . 3 4$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>EEC</td><td>#Parameters</td><td> $2 6 8 6 7 \pm 3 5 2 3$ </td><td> $2 0 2 8 \pm 7 0 . 6 8$ </td><td>3.61E-12</td><td>1</td></tr><tr><td>EEC</td><td>AIC</td><td> $5 2 2 2 8 \pm 6 8 9 7$ </td><td> $2 3 1 9 \pm 2 3 1 . 5$ </td><td>3.61E-12</td><td>1</td></tr></table>

## S8. ADDITIONAL LEARNING CURVES

## A. Training MAE Learning Curves

Fig. S2 presents the training MAE curves for all eight benchmark problems.

![](images/707d9a9c77636e723193c8889f309d4965a09f2e59961acdd08420f74df84cff.jpg)  
(a) �<sub>1</sub>: Rastrigin Function

![](images/581a1cce7ccf660fac9e29e7dcf34cfde0c78b3dcb1cbf5d61e3054786aca947.jpg)  
(b) �<sub>2</sub>: Rosenbrock Function

![](images/d111a27ad51183d9626b815097dd82ef72bd1e9800322de5334e0049f84095a3.jpg)  
(c) �<sub>3</sub>: Cross Function

![](images/27b2e4861a185ed336f082b440e11f38bd5ce869941652d14a18df142c1849f6.jpg)  
(d) �<sub>4</sub>: Styblinski-Tang Function

![](images/7a3e56b7067af747561c14abf647f834d450c24a3e3dc5daccb64bd0e6172abe.jpg)  
(e) ASN

![](images/1f394f3c3f7103377fe96dfcf0a489c19a3ebd955a62b7e62f9abc7c33f76e5f.jpg)  
(f) CCPP

![](images/6e029cafe51b9f422ac2090caae9abc1964337e19dc1152a9eb5ba737685bbb4.jpg)  
(g) CS

![](images/1f0586359ff32008b10ad0884d01b5f5dbccd1a0bee64ecd8db482c9897efcff.jpg)  
(h) EEC  
Fig. S2. Training MAE learning curves for all eight benchmark problems. Curves show the mean over 30 runs, and shaded regions denote 95% confidence intervals.

## B. Testing MAE Learning Curves

Fig. S3 presents the testing MAE curves for all eight benchmark problems. Note that Figs. S3a–S3d reproduce Figs. 3a–3d from the main article.

![](images/47907aee5173e8aa79b8933b604be0c0c8e8f0211141f79b80d12136d964265a.jpg)  
(a) �<sub>1</sub>: Rastrigin Function

![](images/47b836d6ed9d3e55dbed9b366bf64cef0c04d0b12a9bb84a4dd9de6fb738130c.jpg)  
(b) �<sub>2</sub>: Rosenbrock Function

![](images/f8ea229b24e0d135c8e21d2e891288480384aecb534c7e6f34c67c505b31184a.jpg)  
(c) �<sub>3</sub>: Cross Function

![](images/508d2fb0e06ce4a67ecb25fe903dfbb565e1a2001df162f3e11331f0cdae0ad5.jpg)  
(d) �<sub>4</sub>: Styblinski-Tang Function

![](images/7ad03aab3d951f979ff5053f0c06c68c36a9254b8682c53353f57e9e30e841a6.jpg)  
(e) ASN

![](images/268a10eed791a3f306184f33090283f4a944fdd1c2086835908e3747680247d0.jpg)  
(f) CCPP

![](images/86933e431d8d146d003f74e9cd386cc766e37f4982dd4e8709216ad52800a676.jpg)  
(g) CS

![](images/61d92d4f87e9d938634a0e2e3ae851860cebf04780a939d0f5edd3a8f82f6887.jpg)  
(h) EEC  
Fig. S3. Testing MAE learning curves for all eight benchmark problems. Curves show the mean over 30 runs, and shaded regions denote 95% confidence intervals.

## C. Parameter-Count Learning Curves

Fig. S4 summarizes the parameter-count trajectories during training. Compared with MAE reduction, parameter growth remains controlled, suggesting that KACS improves predictive accuracy without relying on uncontrolled model-size expansion. This compactness trend is observed consistently over all benchmark problems.

![](images/01fd9f7438c58561a616c3281e333ce0e061db8ec6115297a13893c3e3e8a8f6.jpg)  
(a) �<sub>1</sub>: Rastrigin Function

![](images/8d630d1e47880c31767007c2d9af055b28f7f906ba6016034000c6598d16989a.jpg)  
(b) � : Rosenbrock Function

![](images/d0b00c6cd35e04d84e04753a653d1828b0bc06051087a1c47cd9b78f7742b0a5.jpg)  
(c) �<sub>3</sub>: Cross Function

![](images/30ea7169279d22ff530e7b8a1246961b1593e23af55d9fbc0488509f9a05c845.jpg)  
(d) �<sub>4</sub>: Styblinski-Tang Function

![](images/f06ab5181cf35305c0a6397a02bd3797666b86cc661d1f41161eff466cdbcbc1.jpg)  
(e) ASN

![](images/7ec19aeec5c23fd583ceeaca5bb7acbbe8c435ac4880f7d10061c6dc81addc05.jpg)  
(f) CCPP

![](images/89d99c7d7f5971f8e61fc5016e9dd9d3f8f52213e70b58adfa6093d4157381dd.jpg)  
(g) CS

![](images/d973b4d584d0bca2a1efdb712574e51d0ccc482c5ceae897df5738934875e8a4.jpg)  
(h) EEC  
Fig. S4. Parameter-count learning curves for all eight benchmark problems. Curves show the mean over 30 runs, and shaded regions denote 95% confidence intervals.

## S9. EFFECTS OF ROTATION, TRANSLATION, AND SCALING ON $f _ { 3 }$

The original $f _ { 3 }$ (cf. Fig. S5a) has ridges aligned with the coordinate axes, which would be a favorable structure for both KACS and XCSF. To test whether the advantage of KACS depends on this alignment, we evaluate both methods on a rotated variant of $f _ { 3 }$ in which the ridges are tilted by $4 5 ^ { \circ }$ relative to the coordinate axes. To complement this primary rotation analysis, we also define translated and scaled variants of $f _ { 3 }$ below.

## A. Problem Definition

Define two aggregate variables

$$
a = \frac { 1 } { \lfloor n / 2 \rfloor } \sum _ { i = 1 } ^ { \lfloor n / 2 \rfloor } x _ { i } , \qquad b = \frac { 1 } { \lceil n / 2 \rceil } \sum _ { i = \lfloor n / 2 \rfloor + 1 } ^ { n } x _ { i } .\tag{20}
$$

The $4 5 ^ { \circ }$ -rotated coordinates are

$$
a _ { \mathrm { r o t } } = { \frac { a + b } { \sqrt { 2 } } } , \qquad b _ { \mathrm { r o t } } = { \frac { - a + b } { \sqrt { 2 } } } .\tag{21}
$$

The rotated $f _ { 3 }$ is then defined as

$$
f _ { 3 } ^ { \mathrm { r o t } } ( { \bf x } ) = \mathrm { m a x } \left( \exp ( - 1 0 a _ { \mathrm { r o t } } ^ { 2 } ) , ~ \exp ( - 5 0 b _ { \mathrm { r o t } } ^ { 2 } ) , ~ 1 . 2 5 ~ \exp \left( - 5 ( a ^ { 2 } + b ^ { 2 } ) \right) \right) .\tag{22}
$$

The third term 1.25 exp $\left( - 5 ( a ^ { 2 } + b ^ { 2 } ) \right)$ does not change under rotation. Fig. S5b illustrates the rotated $f _ { 3 }$ for $n = 2$

The translated variant shifts every input coordinate by 0.20, i.e.,

$$
z _ { i } ^ { \mathrm { t r } } = x _ { i } - 0 . 2 0 , \qquad i = 1 , . . . , n ,\tag{23}
$$

and

$$
a _ { \mathrm { t r } } = \frac { 1 } { \lfloor n / 2 \rfloor } \sum _ { i = 1 } ^ { \lfloor n / 2 \rfloor } z _ { i } ^ { \mathrm { t r } } , \qquad b _ { \mathrm { t r } } = \frac { 1 } { \lceil n / 2 \rceil } \sum _ { i = \lfloor n / 2 \rfloor + 1 } ^ { n } z _ { i } ^ { \mathrm { t r } } .\tag{24}
$$

The translated $f _ { 3 }$ is

$$
f _ { 3 } ^ { \mathrm { t r } } ( \mathbf { x } ) = \operatorname* { m a x } \left( \exp ( - 1 0 a _ { \mathrm { t r } } ^ { 2 } ) , \ \exp ( - 5 0 b _ { \mathrm { t r } } ^ { 2 } ) , \ 1 . 2 5 \exp \left[ - 5 \left( a _ { \mathrm { t r } } ^ { 2 } + b _ { \mathrm { t r } } ^ { 2 } \right) \right] \right) .\tag{25}
$$

Fig. S5c illustrates the translated $f _ { 3 }$ for $n = 2$

The scaled variant scales every input coordinate by a factor of 1.25, i.e.,

$$
z _ { i } ^ { \mathrm { s c } } = 1 . 2 5 x _ { i } , \qquad i = 1 , \ldots , n ,\tag{26}
$$

and

$$
a _ { \mathrm { s c } } = \frac { 1 } { \lfloor n / 2 \rfloor } \sum _ { i = 1 } ^ { \lfloor n / 2 \rfloor } z _ { i } ^ { \mathrm { s c } } , \qquad b _ { \mathrm { s c } } = \frac { 1 } { \lceil n / 2 \rceil } \sum _ { i = \lfloor n / 2 \rfloor + 1 } ^ { n } z _ { i } ^ { \mathrm { s c } } .\tag{27}
$$

The scaled $f _ { 3 }$ is

$$
f _ { 3 } ^ { \mathrm { s c } } ( \mathbf { x } ) = \mathrm { m a x } \left( \exp ( - 1 0 a _ { \mathrm { s c } } ^ { 2 } ) , \ \exp ( - 5 0 b _ { \mathrm { s c } } ^ { 2 } ) , \ 1 . 2 5 \exp \left[ - 5 \left( a _ { \mathrm { s c } } ^ { 2 } + b _ { \mathrm { s c } } ^ { 2 } \right) \right] \right) .\tag{28}
$$

Fig. S5d illustrates the scaled $f _ { 3 }$ for $n = 2$

All other experimental conditions follow the main article: $n = 1 0 , N _ { S } = 1 0 0 0 .$ , and 30 independent runs.

## B. Experimental Results

Figs. S5e and S5f show the testing MAE learning curves for the original and rotated $f _ { 3 } ,$ respectively. On both problems, KACS achieves significantly lower testing MAE than XCSF at the end of training. After rotation, KACS accuracy improves slightly (0.2329 → 0.2184) while XCSF accuracy degrades $( 0 . 3 2 4 6  0 . 3 4 3 0 )$ , widening the gap between the two methods from 0.0917 to 0.1246.

Figs. S5g and S5h show the corresponding results for the translated and scaled variants. On the translated $f _ { 3 } ,$ , the final testing MAE is 0.2901 for XCSF and 0.2478 for KACS, whereas on the scaled $f _ { 3 } ,$ , it is 0.3911 and 0.2488, respectively. KACS therefore retains significantly lower testing MAE under both transformations. Relative to the original $f _ { 3 } ,$ the accuracy gap narrows from 0.0917 to 0.0423 after translation but widens to 0.1423 after scaling.

## C. Discussion

XCSF uses axis-aligned rectangles to cover the input space. These rectangles fit well when the ridges of $f _ { 3 }$ run along the coordinate axes, but become less efficient when the ridges are tilted, requiring more rules to cover the diagonal structure. This is why XCSF accuracy degrades after rotation (testing MAE: 0.3246 → 0.3430).

KACS decomposes the target function into sums of one-dimensional components following the KA theorem (cf. (1) in the main article). This additive structure is compatible with both the original and rotated ridges of �<sub>3</sub>, so the rotation does not disadvantage KACS. As a result, the accuracy gap between the two methods widens after rotation (0.0917 → 0.1246), showing that the advantage of KACS does not depend on axis-aligned function structure.

Translation preserves the orientation and width of the ridges while moving their location by 0.20 and therefore introduces no orientation mismatch. Although the accuracy gap becomes smaller, KACS remains significantly more accurate, indicating that its advantage is not restricted to a function centered at the origin.

Scaling the inputs by 1.25 contracts the ridge widths to 80% of their original values, requiring finer spatial resolution. The larger degradation of XCSF is consistent with the need for finer multidimensional partitions, whereas KACS remain comparatively stable through its one-dimensional submodels.

Together with the rotation results, these findings show that the advantage of KACS persists across changes in the orientation, location, and spatial scale of $f _ { 3 }$

![](images/8fa6c597bbbcb415793b766b1da68cdbee91a8cccdebfa0dde1b6f7c92b263e1.jpg)  
(a) �<sub>3</sub>

![](images/ab3f5c1270a782ae030733a2d927f4ba698c6ba8fa347a963451578abb08dc86.jpg)  
(b) Rotated �<sub>3</sub>

![](images/27fb25c6d7d35046d0c05397d6e41457af08b715c10e4944c88ee66de3cbd05c.jpg)  
(c) Translated �<sub>3</sub>

![](images/c5dc037abaf7c9e5afec85193affcc3d7e528c2a5a4913d25fa90aadfc2bdfb4.jpg)  
(d) Scaled �<sub>3</sub>

![](images/db1e8fe3805f994165104eae22c23f29626684535cbb5209d1774d2e93a8e105.jpg)  
(e) Testing MAE on $f _ { 3 }$

![](images/bb6e3c8c502b9cfa5acc81306e39bdc977ffd3e54558a981ee5fcbc07d343822.jpg)  
(f) Testing MAE on Rotated �<sub>3</sub>

![](images/255c56bdff443882a188ff175a8cef31c24d988be855571cf7182614b306acd0.jpg)  
(g) Testing MAE on Translated �<sub>3</sub>

![](images/241f577d17ceef35d726ec05089d3902b2821c8fe82860b691cf4df6d57068f2.jpg)  
(h) Testing MAE on Scaled �<sub>3</sub>  
Fig. S5. Original, rotated, translated, and scaled �<sub>3</sub> settings. (a)–(d): 3D landscapes of the original function and its rotated, translated by 0.20, and scaled by 1.25 variants. (e)–(h): corresponding testing MAE curves.

## S10. SCALABILITY ANALYSIS — DETAILED RESULTS

## A. Sample-Size Scaling

Table S6 reports detailed results under varying sample sizes $N _ { S } \in \{ 1 2 5 , 2 5 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 , 4 0 0 0 , 8 0 0 0 \}$ with $n = 1 0$ fixed. Table S7 reports the corresponding run-level variability, paired Wilcoxon �-values, and effect sizes for each sample size and metric.

Several patterns are evident. In training MAE, XCSF achieves significantly lower values than KACS at small sample sizes $( N _ { S } \leq 2 5 0 )$ , indicating that �-dimensional rules overfit when data are scarce. This gap narrows and reverses as $N _ { S }$ increases: from $N _ { S } = 1 0 0 0$ onward, KACS achieves significantly lower training MAE, suggesting that one-dimensional submodels become more effective as more data are available per subproblem. In testing MAE, KACS consistently achieves significantly lower values across all sample sizes $( p = 0 . 0 1 5 6 )$ , confirming that the generalization advantage of KA-based decomposition holds regardless of data volume.

Fig. S6 provides an anytime view of testing MAE for all sample sizes. XCSF improves rapidly during the early iterations but subsequently changes only gradually, reaching an apparent plateau for most settings and showing increasing testing error at $N _ { S } = 2 5 0$ . In contrast, KACS requires a longer warm-up but continues to improve and approaches a lower testing-MAE level. For $N _ { S } \geq 5 0 0 .$ , the XCSF curves vary little during the latter half of training while a clear accuracy gap remains. At $N _ { S } = 1 2 5$ , both curves remain relatively variable and KACS is still improving at the final iteration; nevertheless, XCSF has already stabilized at a higher error level.

Model complexity shows a clear structural difference. XCSF rules and parameters grow monotonically with $N _ { S }$ (3164 to 4724 rules; 34800 to 51960 parameters), whereas KACS remains compact throughout (1242–1430 rules; 2484–2860 parameters). The AIC advantage of KACS widens substantially as $N _ { S }$ increases: at $N _ { S } = 8 0 0 0$ , KACS achieves an AIC of −13090 compared with 90170 for XCSF, reflecting both higher accuracy and dramatically fewer parameters at large data scales.

TABLE S6  
SUMMARY OF SAMPLE-SIZE SCALABILITY ANALYSIS. GREEN INDICATES THE BEST VALUE. RANK IS THE AVERAGE RANK ACROSS PROBLEMS. SYMBOLS +/−/∼ DENOTE SIGNIFICANTLY BETTER/WORSE/SIMILAR PERFORMANCE COMPARED WITH KACS BY THE WILCOXON SIGNED-RANK TEST. ARROWS $\uparrow / \downarrow$ INDICATE RANK IMPROVEMENT/DECLINE RELATIVE TO KACS. STATISTICAL SIGNIFICANCE IS AT � = 0.05 (†)
<table><tr><td rowspan="2"></td><td colspan="4">MODEL ACCURACY</td><td colspan="4">MODEL COMPLEXITY</td><td colspan="2">TRADE-OFF</td></tr><tr><td>Training MAE</td><td></td><td colspan="2">Testing MAE</td><td colspan="2">#Rules (|P|)</td><td colspan="2">#Parameters (k)</td><td colspan="2">AIC</td></tr><tr><td> $N _ { S }$ </td><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td><td>XCSF</td><td>KACS</td></tr><tr><td>125</td><td>0.0124 +</td><td>0.2077</td><td>0.4156~</td><td>0.3798</td><td>3164-</td><td>1242</td><td>34802</td><td>2484</td><td>68860-</td><td>4652</td></tr><tr><td>250</td><td>0.0893 +</td><td>0.207</td><td>0.3964-</td><td>0.3308</td><td>3585 -</td><td>1333</td><td>39436-</td><td>2666</td><td>78067-</td><td>4722</td></tr><tr><td>500</td><td>0.1805~</td><td>0.1903</td><td>0.3484-</td><td>0.2246</td><td>3924-</td><td>1353</td><td>43166-</td><td>2707</td><td>85158-</td><td>4137</td></tr><tr><td>1000</td><td>0.2510 -</td><td>0.2097</td><td>0.3246-</td><td>0.2329</td><td>4130-</td><td>1429</td><td>45426-</td><td>2857</td><td>88892 -</td><td>3324</td></tr><tr><td>2000</td><td>0.2824-</td><td>0.2080</td><td>0.3095-</td><td>0.2177</td><td>4327-</td><td>1441</td><td>47597-</td><td>2882</td><td>91538-</td><td>886.6</td></tr><tr><td>4000</td><td>0.3027-</td><td>0.2098</td><td>0.3129–</td><td>0.2131</td><td>4521-</td><td>1406</td><td>49731-</td><td>2812</td><td>92463-</td><td>-3964</td></tr><tr><td>8000</td><td>0.3075-</td><td>0.2146</td><td>0.3117-</td><td>0.2159</td><td>4724-</td><td>1430</td><td>51964 -</td><td>2860</td><td>90165 -</td><td>-13089</td></tr><tr><td>Rank</td><td>1.57↓</td><td>1.43</td><td>2.00↓†</td><td>1.00</td><td>2.00↓†</td><td>1.00</td><td>2.00↓†</td><td>1.00</td><td>2.00↓†</td><td>1.00</td></tr><tr><td> $+ / - / \sim$ </td><td>2/4/1</td><td>1</td><td>0/6/1</td><td>-</td><td>0/7/0</td><td>-</td><td>0/7/0</td><td>-</td><td>0/7/0</td><td>-</td></tr><tr><td>p-value</td><td>1</td><td></td><td>0.0156</td><td></td><td>0.0156</td><td></td><td>0.0156</td><td></td><td>0.0156</td><td></td></tr></table>

TABLE S7  
DETAILED PAIRED COMPARISON OF XCSF AND KACS OVER 30 MATCHED RUNS FOR EACH SAMPLE SIZE AND METRIC. NOTATION FOLLOWS TABLE S3
<table><tr><td> $N _ { S }$ </td><td>Metric</td><td>XCSF mean ± SD</td><td>KACS mean ± SD</td><td>Wilcoxon p</td><td>Effect size</td></tr><tr><td>125</td><td>Training MAE</td><td> $0 . 0 1 2 4 \pm 0 . 0 0 8 0$ </td><td> $\overline { { 0 . 2 0 7 7 \pm 0 . 0 7 7 5 } }$ </td><td>1.86E-09</td><td>-1</td></tr><tr><td>125</td><td>Testing MAE</td><td> $0 . 4 1 5 6 \pm 0 . 0 8 1 4$ </td><td> $0 . 3 7 9 8 \pm 0 . 0 9 5 3$ </td><td>0.1241</td><td>0.3247</td></tr><tr><td>125</td><td>#Rules</td><td> $3 1 6 4 \pm 8 2 . 0 6$ </td><td> $1 2 4 2 \pm 4 5 . 3 6$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>125</td><td>#Parameters</td><td> $3 4 8 0 2 \pm 9 0 2 . 7$ </td><td> $2 4 8 4 \pm 9 0 . 7 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>125</td><td>AIC</td><td> $6 8 8 6 0 \pm 1 8 5 7$ </td><td> $4 6 5 2 \pm 1 7 6 . 2$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>250</td><td>Training MAE</td><td> $0 . 0 8 9 3 \pm 0 . 0 2 9 7$ </td><td> $0 . 2 0 7 0 \pm 0 . 0 4 6 4$ </td><td>1.86E-09</td><td>-1</td></tr><tr><td>250</td><td>Testing MAE</td><td> $0 . 3 9 6 4 \pm 0 . 0 7 1 7$ </td><td> $0 . 3 3 0 8 \pm 0 . 0 7 6 0$ </td><td>0.001232</td><td>0.6516</td></tr><tr><td>250</td><td>#Rules</td><td> $3 5 8 5 \pm 6 9 . 4 5$ </td><td> $1 3 3 3 \pm 4 2 . 3 4$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>250</td><td>#Parameters</td><td> $3 9 4 3 6 \pm 7 6 3 . 9$ </td><td> $2 6 6 6 \pm 8 4 . 6 8$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>250</td><td>AIC</td><td> $7 8 0 6 7 \pm 1 5 9 3$ </td><td> $4 7 2 2 \pm 1 9 6 . 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>500</td><td>Training MAE</td><td> $\overline { { 0 . 1 8 0 5 \pm 0 . 0 2 6 5 } }$ </td><td> $0 . 1 9 0 3 \pm 0 . 0 3 7 8$ </td><td>0.1772</td><td>-0.286</td></tr><tr><td>500</td><td>Testing MAE</td><td> $0 . 3 4 8 4 \pm 0 . 0 4 7 1$ </td><td> $0 . 2 2 4 6 \pm 0 . 0 5 1 5$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>500</td><td>#Rules</td><td> $3 9 2 4 \pm 6 3 . 1 4$ </td><td> $1 3 5 3 \pm 3 9 . 3 7$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>500</td><td>#Parameters</td><td> $4 3 1 6 6 \pm 6 9 4 . 5$ </td><td> $2 7 0 7 \pm 7 8 . 7 4$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>500</td><td>AIC</td><td> $8 5 1 5 8 \pm 1 3 8 7$ </td><td> $4 1 3 7 \pm 2 1 3 . 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>1000</td><td>Training MAE</td><td> $\overline { { 0 . 2 5 1 0 \pm 0 . 0 1 6 7 } }$ </td><td> $\overline { { 0 . 2 0 9 7 \pm 0 . 0 3 9 5 } }$ </td><td>4.42E-06</td><td>0.8667</td></tr><tr><td>1000</td><td>Testing MAE</td><td> $0 . 3 2 4 6 \pm 0 . 0 2 5 0$ </td><td> $0 . 2 3 2 9 \pm 0 . 0 5 0 8$ </td><td>3.73E-09</td><td>0.9957</td></tr><tr><td>1000</td><td>#Rules</td><td> $4 1 3 0 \pm 7 4 . 0 8$ </td><td> $1 4 2 9 \pm 3 6 . 9 4$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>1000</td><td>#Parameters</td><td> $4 5 4 2 6 \pm 8 1 4 . 9$ </td><td> $2 8 5 7 \pm 7 3 . 8 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>1000</td><td>AIC</td><td> $8 8 8 9 2 \pm 1 5 8 0$ </td><td> $3 3 2 4 \pm 3 4 9 . 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>2000</td><td>Training MAE</td><td> $\overline { { 0 . 2 8 2 4 \pm 0 . 0 0 8 6 } }$ </td><td> $0 . 2 0 8 0 \pm 0 . 0 6 0 9$ </td><td>1.22E-05</td><td>0.8366</td></tr><tr><td>2000</td><td>Testing MAE</td><td> $0 . 3 0 9 5 \pm 0 . 0 1 6 0$ </td><td> $0 . 2 1 7 7 \pm 0 . 0 6 6 3$ </td><td>3.79E-06</td><td>0.871</td></tr><tr><td>2000</td><td>#Rules</td><td> $4 3 2 7 \pm 6 9 . 4 3$ </td><td> $1 4 4 1 \pm 3 5 . 9 3$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>2000</td><td>#Parameters</td><td> $4 7 5 9 7 \pm 7 6 3 . 8$ </td><td> $2 8 8 2 \pm 7 1 . 8 5$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>2000</td><td>AIC</td><td> $9 1 5 3 8 \pm 1 5 2 1$ </td><td> $8 8 6 . 6 \pm 8 1 7 . 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>4000</td><td>Training MAE</td><td> $\overline { { 0 . 3 0 2 7 \pm 0 . 0 0 7 0 } }$ </td><td> $0 . 2 0 9 8 \pm 0 . 0 4 8 3$ </td><td>1.86E-08</td><td>0.9785</td></tr><tr><td>4000</td><td>Testing MAE</td><td> $0 . 3 1 2 9 \pm 0 . 0 1 4 7$ </td><td> $0 . 2 1 3 1 \pm 0 . 0 4 6 9$ </td><td>3.73E-09</td><td>0.9957</td></tr><tr><td>4000</td><td>#Rules</td><td> $4 5 2 1 \pm 5 0 . 5 7$ </td><td> $1 4 0 6 \pm 3 0 . 4 2$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>4000</td><td>#Parameters</td><td> $4 9 7 3 1 \pm 5 5 6 . 3$ </td><td> $2 8 1 2 \pm 6 0 . 8 5$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>4000</td><td>AIC</td><td> $9 2 4 6 3 \pm 1 0 3 3$ </td><td> $- 3 9 6 4 \pm 1 4 4 5$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>8000</td><td>Training MAE</td><td> $\overline { { 0 . 3 0 7 5 \pm 0 . 0 0 5 9 } }$ </td><td> $0 . 2 1 4 6 \pm 0 . 0 4 6 8$ </td><td>1.30E-08</td><td>0.9828</td></tr><tr><td>8000</td><td>Testing MAE</td><td> $0 . 3 1 1 7 \pm 0 . 0 0 9 9$ </td><td> $0 . 2 1 5 9 \pm 0 . 0 4 9 1$ </td><td>1.86E-08</td><td>0.9785</td></tr><tr><td>8000</td><td>#Rules</td><td> $4 7 2 4 \pm 5 7 . 7 2$ </td><td> $1 4 3 0 \pm 3 2 . 9 3$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>8000</td><td>#Parameters</td><td> $5 1 9 6 4 \pm 6 3 4 . 9$ </td><td> $2 8 6 0 \pm 6 5 . 8 6$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>8000</td><td>AIC</td><td> $9 0 1 6 5 \pm 1 2 5 5$ </td><td> $- 1 3 0 8 9 \pm 2 8 5 3$ </td><td>1.86E-09</td><td>1</td></tr></table>

![](images/89bd87867da9ec6ed8ef74593d56225d0d7ffe3b4cbc517ff9fd462f42f772dd.jpg)  
(a) $N s = 1 2 5$

![](images/87c3e44df794e27a60924035d77e53b9decc1363680e3ed27af12780df1752b5.jpg)  
(b) $N s = 2 5 0$

![](images/97ff065f5fe926c21439e70b1daac3999fea268e6e564e40f524fc8c681980db.jpg)  
(c) $N _ { S } = 5 0 0$

![](images/d025a5912b1dadfba6e4d47473c18ec53d4ca2d7b71408bc1227b232f359db48.jpg)  
(d) $N _ { S } = 1 0 0 0$

![](images/3c4bf8cc14e78a4e02bb5294a14c1465757408e7e5b715dae4d0154aa43697e4.jpg)  
(e) $N _ { S } = 2 0 0 0$

![](images/656022f9eedd8d10159b7432aed2324b264dd8ae61229a1d54267f473d20298d.jpg)  
(f) $N _ { S } = 4 0 0 0$

![](images/efaf14205255fd900a988c84c6d3a6316fcec3bfde2a754b8ff94cefeceec925.jpg)  
(g) $N _ { S } = 8 0 0 0$  
Fig. S6. Testing MAE learning curves of XCSF and KACS on $f _ { 3 }$ under varying sample sizes, with the input dimension fixed at $n = 1 0 .$

## B. Dimensional Scaling

Table S8 reports detailed results as the input dimension � varies from 2 to 20 with $N _ { S } = 1 0 0 0$ fixed. Table S9 reports the corresponding run-level variability, paired Wilcoxon �-values, and effect sizes for each dimension and metric.

In the practical regime $( n \leq 1 6 )$ , KACS achieves significantly lower testing MAE than XCSF at all dimensions. The accuracy gap is largest at ${ n = 2 \ ( 0 . 0 8 8 3 \ \mathrm { v s . } \ 0 . 1 5 8 2 ) }$ and remains consistent through � = 14 (0.2021 vs. 0.3066). Notably, XCSF testing MAE does not grow monotonically with �: it peaks around $n = 6 ~ ( 0 . 3 2 9 2 )$ and slightly decreases at higher dimensions, while KACS testing MAE remains stable between 0.20 and 0.25 up to � = 14.

At $n \geq 1 8$ under the default budget $N = 6 4 0 0 .$ , KACS degrades sharply: testing MAE rises to 0.3842 at $n = 1 8$ and 1.2500 at $n = 2 0$ , at which point XCSF achieves significantly better accuracy. As discussed in Section VI-C4 of the main article, this failure is caused by the cover-delete cycle that occurs when � is too small relative to the $( 2 n ^ { 2 } + 3 n + 1 )$ subproblems. Using the scaled budgets $N = 1 5 9 8 4$ and $N = 1 9 6 8 0$ for $n = 1 8$ and $n = 2 0 .$ , respectively, KACS recovers and again achieves significantly lower testing MAE than XCSF (0.2115 vs. 0.2768 at $n = 1 8 ; 0 . 1 9 7 0$ vs. 0.2833 at $n = 2 0 )$ , confirming that the degradation is a capacity issue rather than an inherent limitation of KACS

Model complexity confirms the structural efficiency of KACS throughout. XCSF parameters grow monotonically from 4504 $( n = 2 )$ to 110,900 (� = 20), whereas KACS parameters remain between 2372 and 3909 across all dimensions, including the high-dimensional failure cases. The AIC advantage of KACS holds for � ≤ 16 (e.g., 2120 vs. 138,200 at $n = 1 4 )$ and reverses only partially at $n \geq 1 8$ under $N = 6 4 0 0$ , where KACS accuracy collapses despite its compact parameter count. At $n = 2 .$ KACS rule count (1955) slightly exceeds that of XCSF (1501), the only exception across all conditions, reflecting the fixed submodel structure of KACS at low dimensions

TABLE S8  
SUMMARY OF DIMENSIONALITY SCALABILITY ANALYSIS. NOTATION FOLLOWS TABLE S6
<table><tr><td rowspan=2 colspan=1>n</td><td rowspan=1 colspan=4>MODEL ACCURACYTraining MAE       Testing MAE</td><td rowspan=1 colspan=4>MODEL COMPLEXITY#Rules (|P|)      #Parameters (k)</td><td rowspan=2 colspan=2>TRADE-OFFAICXCSF   KACS</td></tr><tr><td rowspan=1 colspan=2>XCSF   KACS</td><td rowspan=1 colspan=2>XCSF   KACS</td><td rowspan=1 colspan=2>XCSF  KACS</td><td rowspan=1 colspan=2>XCSF   KACS</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.1459 -</td><td rowspan=1 colspan=1>0.0831</td><td rowspan=1 colspan=1>0.1582 -</td><td rowspan=1 colspan=1>0.0883</td><td rowspan=1 colspan=1>1501 +</td><td rowspan=1 colspan=1>1955</td><td rowspan=1 colspan=2>4504 -   3909</td><td rowspan=1 colspan=2>6251-    3792</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.2210～</td><td rowspan=1 colspan=1>0.2145</td><td rowspan=1 colspan=1>0.2656 -</td><td rowspan=1 colspan=1>0.2345</td><td rowspan=1 colspan=1>2829 -</td><td rowspan=1 colspan=1>1862</td><td rowspan=1 colspan=1>14145-</td><td rowspan=1 colspan=1>3725</td><td rowspan=1 colspan=2>26047-   5125</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.2632 -</td><td rowspan=1 colspan=1>0.2174</td><td rowspan=1 colspan=1>0.3292 -</td><td rowspan=1 colspan=1>0.2417</td><td rowspan=1 colspan=1>3423-</td><td rowspan=1 colspan=1>1723</td><td rowspan=1 colspan=1>23960-</td><td rowspan=1 colspan=1>3447</td><td rowspan=1 colspan=2>45884-   4575</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.2525-</td><td rowspan=1 colspan=1>0.2076</td><td rowspan=1 colspan=1>0.3269 -</td><td rowspan=1 colspan=1>0.2208</td><td rowspan=1 colspan=1>3837-</td><td rowspan=1 colspan=1>1540</td><td rowspan=1 colspan=1>34529-</td><td rowspan=1 colspan=1>3079</td><td rowspan=1 colspan=2>67024-   3765</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.2510 –</td><td rowspan=1 colspan=1>0.2097</td><td rowspan=1 colspan=1>0.3246-</td><td rowspan=1 colspan=1>0.2329</td><td rowspan=1 colspan=1>4130-</td><td rowspan=1 colspan=1>1429</td><td rowspan=1 colspan=1>45426-</td><td rowspan=1 colspan=1>2857</td><td rowspan=1 colspan=2>88892 -   3324</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>0.2355 –</td><td rowspan=1 colspan=1>0.2117</td><td rowspan=1 colspan=1>0.3224-</td><td rowspan=1 colspan=1>0.2252</td><td rowspan=1 colspan=1>4439 -</td><td rowspan=1 colspan=1>1311</td><td rowspan=1 colspan=1>57703-</td><td rowspan=1 colspan=1>2623</td><td rowspan=1 colspan=1>113331 -</td><td rowspan=1 colspan=1>2787</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0.2398一</td><td rowspan=1 colspan=1>0.1870</td><td rowspan=1 colspan=1>0.3066-</td><td rowspan=1 colspan=1>0.2021</td><td rowspan=1 colspan=1>4674-</td><td rowspan=1 colspan=1>1186</td><td rowspan=1 colspan=1>70114-</td><td rowspan=1 colspan=1>2372</td><td rowspan=1 colspan=1>138194-</td><td rowspan=1 colspan=1>2120</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.2234~</td><td rowspan=1 colspan=1>0.2359</td><td rowspan=1 colspan=1>0.2770-</td><td rowspan=1 colspan=1>0.2513</td><td rowspan=1 colspan=1>4890-</td><td rowspan=1 colspan=1>1209</td><td rowspan=1 colspan=1>83137-</td><td rowspan=1 colspan=1>2418</td><td rowspan=1 colspan=1>164127-</td><td rowspan=1 colspan=1>2713</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>0.2388 +</td><td rowspan=1 colspan=1>0.3712</td><td rowspan=1 colspan=1>0.2749 +</td><td rowspan=1 colspan=1>0.3842</td><td rowspan=1 colspan=1>5067-</td><td rowspan=1 colspan=1>1400</td><td rowspan=1 colspan=1>96268-</td><td rowspan=1 colspan=1>2801</td><td rowspan=1 colspan=1>190490-</td><td rowspan=1 colspan=1>4335</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>0.2395 +</td><td rowspan=1 colspan=1>1.2015</td><td rowspan=1 colspan=1>0.2706 +</td><td rowspan=1 colspan=1>1.2497</td><td rowspan=1 colspan=1>5283 -</td><td rowspan=1 colspan=1>1862</td><td rowspan=1 colspan=1>110941-</td><td rowspan=1 colspan=1>3725</td><td rowspan=1 colspan=1>219841-</td><td rowspan=1 colspan=1>8259</td></tr><tr><td rowspan=1 colspan=1>Rank</td><td rowspan=1 colspan=1>1.70↓</td><td rowspan=1 colspan=1>1.30</td><td rowspan=1 colspan=1>1.80↓</td><td rowspan=1 colspan=1>1.20</td><td rowspan=1 colspan=1>1.90↓†</td><td rowspan=1 colspan=1>1.10</td><td rowspan=1 colspan=1>2.00↓†</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>2.00↓†</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1> $+ / - / \sim$ </td><td rowspan=1 colspan=1>2/6/2</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>2/8/0</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>1/9/0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0/10/0</td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=2>0/10/00.00195</td></tr><tr><td rowspan=1 colspan=1>p-value</td><td rowspan=1 colspan=2>0.557</td><td rowspan=1 colspan=1>0.432</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.00391</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.00195</td></tr></table>

TABLE S9  
DETAILED PAIRED COMPARISON OF XCSF AND KACS OVER 30 MATCHED RUNS FOR EACH DIMENSION AND METRIC. NOTATION FOLLOWS TABLE S3
<table><tr><td>n</td><td>Metric</td><td> $\mathrm { X C S F ~ m e a n } \pm \mathrm { { S D } }$ </td><td> $\mathrm { K A C S \ m e a n \pm S D }$ </td><td>Wilcoxon p</td><td>Effect size</td></tr><tr><td>2</td><td>Training MAE</td><td> $\overline { { 0 . 1 4 5 9 \pm 0 . 0 5 2 8 } }$ </td><td> $0 . 0 8 3 1 \pm 0 . 0 2 0 3$ </td><td> $\overline { { 8 . 3 3 \mathrm { E } \mathrm { - } 0 7 } }$ </td><td>0.9097</td></tr><tr><td>2</td><td>Testing MAE</td><td> $0 . 1 5 8 2 \pm 0 . 0 6 3 7$ </td><td> $0 . 0 8 8 3 \pm 0 . 0 2 2 3$ </td><td>6.92E-06</td><td>0.8538</td></tr><tr><td>2</td><td>#Rules</td><td> $1 5 0 1 \pm 2 2 0 . 6$ </td><td> $1 9 5 5 \pm 9 6 . 3 6$ </td><td>1.82E-06</td><td>-1</td></tr><tr><td>2</td><td>#Parameters</td><td> $4 5 0 4 \pm 6 6 1 . 9$ </td><td> $3 9 0 9 \pm 1 9 2 . 7$ </td><td>4.41E-05</td><td>0.7935</td></tr><tr><td>2</td><td>AIC</td><td> $6 2 5 1 \pm 9 9 5 . 4$ </td><td> $3 7 9 2 \pm 4 9 7 . 9$ </td><td>9.31E-09</td><td>0.9871</td></tr><tr><td>4</td><td>Training MAE</td><td> $\overline { { 0 . 2 2 1 0 \pm 0 . 0 2 0 9 } }$ </td><td> $0 . 2 1 4 5 \pm 0 . 0 2 2 6$ </td><td>0.4045</td><td>0.1785</td></tr><tr><td>4</td><td>Testing MAE</td><td> $0 . 2 6 5 6 \pm 0 . 0 3 1 0$ </td><td> $0 . 2 3 4 5 \pm 0 . 0 3 0 1$ </td><td>0.0001529</td><td>0.7462</td></tr><tr><td>4</td><td>#Rules</td><td> $2 8 2 9 \pm 8 4 . 7 9$ </td><td> $1 8 6 2 \pm 4 2 . 2 4$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>4</td><td>#Parameters</td><td> $1 4 1 4 5 \pm 4 2 3 . 9$ </td><td> $3 7 2 5 \pm 8 4 . 4 8$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>4</td><td>AIC</td><td> $2 6 0 4 7 \pm 7 6 7 . 7$ </td><td> $5 1 2 5 \pm 2 2 2 . 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>6</td><td>Training MAE</td><td> $\overline { { 0 . 2 6 3 2 \pm 0 . 0 1 9 4 } }$ </td><td> $0 . 2 1 7 4 \pm 0 . 0 2 8 8$ </td><td>1.64E-07</td><td>0.9441</td></tr><tr><td>6</td><td>Testing MAE</td><td> $0 . 3 2 9 2 \pm 0 . 0 2 2 8$ </td><td> $0 . 2 4 1 7 \pm 0 . 0 3 4 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>6</td><td>#Rules</td><td> $3 4 2 3 \pm 7 8 . 4 7$ </td><td> $1 7 2 3 \pm 3 4 . 2 4$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>6</td><td>#Parameters</td><td> $2 3 9 6 0 \pm 5 4 9 . 3$ </td><td> $3 4 4 7 \pm 6 8 . 4 8$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>6</td><td>AIC</td><td> $4 5 8 8 4 \pm 1 0 4 5$ </td><td> $4 5 7 5 \pm 2 8 2 . 3$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>8</td><td>Training MAE</td><td> $\overline { { 0 . 2 5 2 5 \pm 0 . 0 1 6 9 } }$ </td><td> $0 . 2 0 7 6 \pm 0 . 0 3 7 2$ </td><td>9.22E-06</td><td>0.8452</td></tr><tr><td>8</td><td>Testing MAE</td><td> $0 . 3 2 6 9 \pm 0 . 0 2 4 2$ </td><td> $0 . 2 2 0 8 \pm 0 . 0 3 5 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>8</td><td>#Rules</td><td> $3 8 3 7 \pm 7 3 . 0 9$ </td><td> $1 5 4 0 \pm 4 1 . 6 8$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>8</td><td>#Parameters</td><td> $3 4 5 2 9 \pm 6 5 7 . 8$ </td><td> $3 0 7 9 \pm 8 3 . 3 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>8</td><td>AIC</td><td> $6 7 0 2 4 \pm 1 2 8 6$ </td><td> $3 7 6 5 \pm 3 0 0 . 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>10</td><td>Training MAE</td><td> $\overline { { 0 . 2 5 1 0 \pm 0 . 0 1 6 7 } }$ </td><td> $\overline { { 0 . 2 0 9 7 \pm 0 . 0 3 9 5 } }$ </td><td>4.42E-06</td><td>0.8667</td></tr><tr><td>10</td><td>Testing MAE</td><td> $0 . 3 2 4 6 \pm 0 . 0 2 5 0$ </td><td> $0 . 2 3 2 9 \pm 0 . 0 5 0 8$ </td><td>3.73E-09</td><td>0.9957</td></tr><tr><td>10</td><td>#Rules</td><td> $4 1 3 0 \pm 7 4 . 0 8$ </td><td> $1 4 2 9 \pm 3 6 . 9 4$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>10</td><td>#Parameters</td><td> $4 5 4 2 6 \pm 8 1 4 . 9$ </td><td> $2 8 5 7 \pm 7 3 . 8 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>10</td><td>AIC</td><td> $8 8 8 9 2 \pm 1 5 8 0$ </td><td> $3 3 2 4 \pm 3 4 9 . 7$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>12</td><td>Training MAE</td><td> $\overline { { 0 . 2 3 5 5 \pm 0 . 0 1 8 5 } }$ </td><td> $\overline { { 0 . 2 1 1 7 \pm 0 . 0 9 2 8 } }$ </td><td>0.01454</td><td>0.5054</td></tr><tr><td>12</td><td>Testing MAE</td><td> $0 . 3 2 2 4 \pm 0 . 0 2 2 2$ </td><td> $0 . 2 2 5 2 \pm 0 . 0 9 2 7$ </td><td>4.42E-06</td><td>0.8667</td></tr><tr><td>12</td><td>#Rules</td><td> $4 4 3 9 \pm 4 4 . 6 1$ </td><td> $1 3 1 1 \pm 3 7 . 3 0$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>12</td><td>#Parameters</td><td> $5 7 7 0 3 \pm 5 8 0 . 0$ </td><td> $2 6 2 3 \pm 7 4 . 6 0$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>12</td><td>AIC</td><td> $1 1 3 3 3 1 \pm 1 1 5 6$ </td><td> $2 7 8 7 \pm 5 9 3 . 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>14</td><td>Training MAE</td><td> $\overline { { 0 . 2 3 9 8 \pm 0 . 0 1 7 5 } }$ </td><td> $0 . 1 8 7 0 \pm 0 . 0 5 3 0$ </td><td>3.24E-06</td><td>0.8753</td></tr><tr><td>14</td><td>Testing MAE</td><td> $0 . 3 0 6 6 \pm 0 . 0 2 0 6$ </td><td> $0 . 2 0 2 1 \pm 0 . 0 5 9 5$ </td><td>1.86E-08</td><td>0.9785</td></tr><tr><td>14</td><td>#Rules</td><td> $4 6 7 4 \pm 4 8 . 4 8$ </td><td> $1 1 8 6 \pm 3 3 . 2 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>14</td><td>#Parameters</td><td> $7 0 1 1 4 \pm 7 2 7 . 2$ </td><td> $2 3 7 2 \pm 6 6 . 5 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>14</td><td>AIC</td><td> $1 3 8 1 9 4 \pm 1 4 3 7$ </td><td> $2 1 2 0 \pm 5 5 1 . 8$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>16</td><td>Training MAE</td><td> $0 . 2 2 3 4 \pm 0 . 0 1 4 9$ </td><td> $\overline { { 0 . 2 3 5 9 \pm 0 . 0 3 6 5 } }$ </td><td>0.0961</td><td>-0.3505</td></tr><tr><td>16</td><td>Testing MAE</td><td> $0 . 2 7 7 0 \pm 0 . 0 2 5 6$ </td><td> $0 . 2 5 1 3 \pm 0 . 0 4 2 2$ </td><td>0.01454</td><td>0.5054</td></tr><tr><td>16</td><td>#Rules</td><td> $4 8 9 0 \pm 5 4 . 4 3$ </td><td> $1 2 0 9 \pm 3 1 . 9 1$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>16</td><td>#Parameters</td><td> $8 3 1 3 7 \pm 9 2 5 . 3$ </td><td> $2 4 1 8 \pm 6 3 . 8 3$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>16</td><td>AIC</td><td> $1 6 4 1 2 7 \pm 1 8 2 8$ </td><td> $2 7 1 3 \pm 2 7 7 . 5$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>18</td><td>Training MAE</td><td> $0 . 2 3 8 8 \pm 0 . 0 1 2 4$ </td><td> $\overline { { 0 . 3 7 1 2 \pm 0 . 0 5 3 7 } }$ </td><td>1.86E-09</td><td>-1</td></tr><tr><td>18</td><td>Testing MAE</td><td> $0 . 2 7 4 9 \pm 0 . 0 2 1 9$ </td><td> $0 . 3 8 4 2 \pm 0 . 0 6 4 3$ </td><td>3.73E-09</td><td>-0.9957</td></tr><tr><td>18</td><td>#Rules</td><td> $5 0 6 7 \pm 5 1 . 7 0$ </td><td> $1 4 0 0 \pm 4 2 . 8 7$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>18</td><td>#Parameters</td><td> $9 6 2 6 8 \pm 9 8 2 . 3$ </td><td> $2 8 0 1 \pm 8 5 . 7 4$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>18</td><td>AIC</td><td> $1 9 0 4 9 0 \pm 1 9 4 6$ </td><td> $4 3 3 5 \pm 3 8 0 . 9$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>20</td><td>Training MAE</td><td> $\overline { { 0 . 2 3 9 5 \pm 0 . 0 1 5 6 } }$ </td><td> $\overline { { 1 . 2 0 1 5 \pm 0 . 1 6 3 2 } }$ </td><td>1.86E-09</td><td>-1</td></tr><tr><td>20</td><td>Testing MAE</td><td> $0 . 2 7 0 6 \pm 0 . 0 3 2 2$ </td><td> $1 . 2 4 9 7 \pm 0 . 2 0 8 0$ </td><td>1.86E-09</td><td>-1</td></tr><tr><td>20</td><td>#Rules</td><td> $5 2 8 3 \pm 7 8 . 4 5$ </td><td> $1 8 6 2 \pm 3 7 . 5 7$ </td><td>1.82E-06</td><td>1</td></tr><tr><td>20</td><td>#Parameters</td><td> $1 1 0 9 4 1 \pm 1 6 4 7$ </td><td> $3 7 2 5 \pm 7 5 . 1 4$ </td><td>1.86E-09</td><td>1</td></tr><tr><td>20</td><td>AIC</td><td> $2 1 9 8 4 1 \pm 3 3 6 9$ </td><td> $8 2 5 9 \pm 3 2 2 . 6$ </td><td>1.86E-09</td><td>1</td></tr></table>

## S11. COVERING DYNAMICS AND COVER-DELETE BEHAVIOR

To examine whether XCSF and KACS repeatedly lose input coverage during training, we recorded the number of match sets that invoked covering. For a recording interval of � iterations, let $C _ { \mathrm { { X } } } , C _ { \mathrm { { K } } } , C _ { \mathrm { { i n } } } .$ , and $C _ { \mathrm { o u t } }$ denote the numbers of covering events in XCSF, all KACS submodels, the KACS inner submodels, and the KACS outer submodels, respectively. We calculate

$$
r _ { \mathrm { X C S F } } = { \frac { C _ { \mathrm { X } } } { L } } ,
$$

$$
r _ { \mathrm { K A C S } } = { \frac { C _ { \mathrm { K } } } { L ( 2 n ^ { 2 } + 3 n + 1 ) } } ,\tag{29}
$$

$$
r _ { \mathrm { i n n e r } } = \frac { C _ { \mathrm { i n } } } { L n ( 2 n + 1 ) } ,
$$

$$
r _ { \mathrm { o u t e r } } = { \frac { C _ { \mathrm { o u t } } } { L ( 2 n + 1 ) } } .\tag{30}
$$

Thus, each rate is the fraction of match-set evaluations that require covering, which accounts for the different numbers of match sets evaluated by XCSF and KACS.

Fig. S7 shows that both methods require frequent covering during initialization. Thereafter, the XCSF covering rate rapidly approaches zero and remains negligible on all eight problems. Thus, under the standard experimental settings, XCSF does not regularly exhibit recurring coverage gaps that would indicate a cover-delete cycle. The overall KACS covering rate also decreases sharply, but remains nonzero throughout training. This persistence is most pronounced on the four synthetic problems and CS, while it is smaller on ASN, CCPP, and EEC.

The near-zero XCSF covering rate is consistent with its high rule density within a single population. As reported in Table II of the main article, XCSF retains 1953–4170 macro-rules, all of which are tested when constructing its single match set. In contrast, KACS distributes 1012–1924 macro-rules among $2 n ^ { 2 } + 3 n + 1$ submodels. For example, the $n = 1 0$ synthetic problems have 231 submodels and 1341–1462 KACS macro-rules, corresponding to only about 5.8–6.3 macro-rules per submodel on average. This substantially lower per-submodel rule density makes a KACS match set more likely to become empty. Thi interpretation concerns antecedent coverage and rule allocation rather than the total number of consequent parameters, because covering is triggered by an empty match set.

![](images/20957a279d3d40a5c8a8ae98e8ded214ddcfced6b1b5c8b405c966324c80065e.jpg)  
(a) �<sub>1</sub>: Rastrigin Function

![](images/79ca9a51c487b8bbbd1ff8f5d31ea29f1efa2df0989807bcae1b8b689cd35107.jpg)  
(b) � : Rosenbrock Function

![](images/5f981938ae79e3e354107760494fd82448c99d7386a439860a4bdb3668438d6c.jpg)  
(c) �<sub>3</sub>: Cross Function

![](images/b76b7d7f76401a95494da0a47f85a4a48802b35a6161a5b249e98a1a761b641a.jpg)  
(d) �<sub>4</sub>: Styblinski-Tang Function

![](images/b61c2e4f4d264559350c133da7264a4ad500dd33cddba0368642b0b063807432.jpg)  
(e) ASN

![](images/9b81805b01e3795d8d38458a2ead023ac80312208737aa47ce248ffbf8fd933e.jpg)  
(f) CCPP

![](images/49f2be895ce8a30ae68187a543505e4d9c7133746ac7dadca2be77a05a5d1637.jpg)  
(g) CS

![](images/f85c52f08f5812f246662ea0abebe7d224c0af73e7511722622d4588d92f4dd6.jpg)  
(h) EEC  
Fig. S7. Covering-rate trajectories of XCSF and KACS during training on the eight benchmark problems. Rates are normalized by the number of match-set evaluations. Curves show the mean over 30 runs, and shaded regions denote 95% confidence intervals.

Fig. S8 localizes this behavior within the KACS architecture. The covering rate of the inner submodels approaches zero after initialization on every problem, following a pattern similar to XCSF. In contrast, the outer submodels continue to invoke covering throughout training, although the magnitude is problem dependent. This difference is consistent with the distinct input domains of the two submodel types: inner submodels always receive inputs in [0, 1], whereas the outer inputs $\hat { z } _ { q }$ change as the inner functions are learned and can span a substantially wider range. Consequently, recurrent coverage gaps are concentrated in the outer submodels rather than being a general property of all KACS rulesets.

Persistent late-stage covering indicates that stable outer-submodel coverage is not always maintained and is therefore consistent with recurrent cover-delete behavior. Moreover, this behavior does not cause a general accuracy failure under the standard settings, where KACS remains competitive despite the nonzero outer covering rate. On the other hand, persistent outer-submodel covering becomes harmful when the population budget is too small to maintain adequate coverage, as observed in the high-dimensional experiments analyzed in Section VI-C4 of the main article. In summary, XCSF and the KACS inner submodels maintain stable coverage after initialization, whereas the dynamically changing outer domains make the KACS outer submodels more susceptible to recurrent covering.

![](images/ec6ce77feef32cf948e5c1bfe0703be1150bd3d3a07b0d64229ccfb305656e73.jpg)  
(a) �<sub>1</sub>: Rastrigin Function

![](images/52ff4f0bbaaeb3cd0e21996e20b58f1a7ee5984dfed9ac9788ee965e63cb2336.jpg)  
(b) �<sub>2</sub>: Rosenbrock Function

![](images/6a40efaea3effe90b689847147d5112e0a9d35f4b2a99ac5e2eb41622641b6a5.jpg)  
(c) �<sub>3</sub>: Cross Function

![](images/0c14c3fe2e8ca51d4249ea208c5b47d59353753af4f0dbb60e5fa91013fa8d58.jpg)  
(d) �<sub>4</sub>: Styblinski-Tang Function

![](images/0efd3108b2fe63ff9760ea3ffe56f2f38783613ee4efbeaec63456d8c3ef5157.jpg)  
(e) ASN

![](images/9d43460bd90f23a728b08b100f8dfa03de12fb7f9f7caa7afb6ac4dfd6a5f6f2.jpg)  
(f) CCPP

![](images/b5e8637c2ae34a8c731876e3ebf947543e0e95aad8c3a657cdb54cb537a2bd01.jpg)  
(g) CS

![](images/c6bfc3f7e308e179b48978468c007fec65c936ce3a2191de965ca562d63f9947.jpg)  
(h) EEC  
Fig. S8. Covering-rate trajectories of the inner and outer KACS submodels during training on the eight benchmark problems. Each rate is normalized b the number of match-set evaluations for the corresponding submodel type. Curves show the mean over 30 runs, and shaded regions denote 95% confidenc intervals.

## REFERENCES

[1] C. De Boor, A practical guide to splines. springer New York, 1978, vol. 27.

[3] M. Heider, H. Stegherr, R. Sraj, D. Patzel, J. Wurth, and J. H¨ ahner, “SupRB in the context of rule-based machine learning methods: A comparative study,”¨ Appl. Soft Comput., p. 110706, 2023.

[2] J. Park and I. W. Sandberg, “Universal approximation using radial-basis-function networks,” Neural Comput., vol. 3, no. 2, pp. 246–257, 1991.

[4] M. Heider, “Disentangling the model selection tasks for improved explainability in a rule-based machine learning system,” Ph.D. dissertation, Universitat¨ Augsburg, 2025.

[5] R. J. Urbanowicz and W. N. Browne, Introduction to Learning Classifier Systems, 1st ed. Springer Publishing Company, Incorporated, 2017.