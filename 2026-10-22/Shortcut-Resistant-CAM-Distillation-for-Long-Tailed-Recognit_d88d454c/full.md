# Shortcut-Resistant CAM Distillation for Long-Tailed Recognition

Wenhai Wan <sup>1</sup> <sup>2</sup> Teng Zhang <sup>1</sup> <sup>2</sup> Shao-Yuan Li <sup>3</sup> <sup>4</sup> <sup>5</sup> <sup>6</sup> Xinrui Wang <sup>3</sup> <sup>4</sup> <sup>7</sup> Qiang-Sheng Hua <sup>1</sup> <sup>2</sup> Songcan Chen <sup>3</sup> <sup>4</sup>

## Abstract

Real-world datasets often follow a long-tailed distribution, making generalization to tail classes difficult. We revisit this problem through the lens of shortcut learning, where models prefer the easiest predictive cues (e.g., background or textures) over object-centric semantics, especially under scarce and biased supervision. We find that this tendency is amplified for tail classes: limited examples often share similar contexts, making non-semantic signals highly correlated and thus tempting shortcuts, whereas head classes with diverse appearances and environments encourage more stable object-focused representations. Motivated by this observation, we propose Shortcut-Resistant CAM Distillation (SRCD), a plug-and-play framework that transfers objectfocused explanations from head to tail classes. SRCD operates in the Class Activation Map (CAM) space, where a CAM provides a classspecific spatial evidence map for a prediction. SRCD aggregates CAMs from a small set of headclass candidates into a shortcut-resistant teacher using an energy-model weighting based on coherence and concentration, and distills it to the tail-class CAM. We provide a theoretical analysis that quantifies shortcut reliance as shortcutregion evidence mass in CAM space and shows that SRCD suppresses tail shortcuts. Extensive experiments on long-tailed benchmarks consistently improve strong baselines. The code is available at https://github.com/Haifeng3/SRCD.

![](images/7a704f0399c82c15cda24724add327c9c7539d99dc7ff30ffdf85016ed56f737.jpg)  
Figure 1. Shortcut learning in long-tailed recognition visualized by CAMs. For a head-class example (Tench), the model attends to the object region, while for a tail-class example (Speedboat), attention drifts to spurious cues (e.g., ocean background), leading to shortcut-dominated predictions and degraded tail generalization.

## 1. Introduction

In recent years, deep neural networks have become the dominant paradigm for visual recognition, enabling strong performance across a wide range of tasks such as image classification (He et al., 2016), medical image analysis (Ronneberger et al., 2015), and object detection (Ren et al., 2015). However, much of this progress implicitly relies on large-scale and relatively balanced training data, which rarely matches the distributions encountered in real-world deployments. In practice, data often follows a long-tailed distribution (Zhang et al., 2023), where a few frequent categories dominate the dataset while most categories contain only a limited number of examples. Learning effectively under such imbalanced supervision has therefore become a central challenge in modern computer vision (Zhang et al., 2025).

Despite steady progress (Zhang et al., 2025), tail-class generalization remains brittle: gains on tail categories are often modest, and performance can degrade sharply when tail supervision is scarce and biased. Fundamentally, the dominant challenge in long-tailed recognition is still tail data scarcity: with only a few labeled instances, predictors are hard to learn and easy to overfit. Building on this, we make a finergrained observation: when examples are few, tail data is also more likely to be context-homogeneous (e.g., captured in similar environments), which can make non-semantic cues appear unusually reliable during training. Consequently, even improved optimization or calibration may not prevent the model from settling on undesirable decision rules for tail classes.

A useful lens is to view learning as selecting one explanation among many that fit the training labels equally well (Lee et al., 2023). Under limited supervision, models often favor the easiest explanations, giving rise to shortcut learning (Geirhos et al., 2020; Hermann et al., 2024): instead of discovering object-centric semantics, they may rely on nonsemantic yet predictive signals such as background context or characteristic textures. In generic spurious-correlation settings (Ye et al., 2024), such shortcuts are typically attributed to distribution bias in the data. In long-tailed recognition, however, shortcut reliance can be further amplified by scarcity: scarce supervision encourages overfitting to accidental correlations, and the accompanying context homogeneity makes these spurious cues appear pseudo-stable within the training set. This scarcity–bias interaction makes shortcuts particularly tempting for minimizing training loss, yet especially harmful for tail generalization.

This mechanism creates a systematic head–tail gap. Head categories contain diverse instances and environments, weakening the reliability of spurious cues and encouraging the model to focus on stable object evidence. In contrast, tail categories often exhibit strong contextual bias across their few examples, making it easier for the model to “explain” them using shortcuts. To make this behavior visible, we use Class Activation Maps (CAMs) (Zhou et al., 2016), which provide a spatial attribution by highlighting image regions that contribute most to a class prediction. As shown in Figure 1, for a head-class example (Tench), the CAM concentrates on the object region, indicating meaningful object cues. For a tail-class example (Speedboat), scarce and biased supervision makes the ocean background a strong correlate; learning such context is substantially easier than learning the object itself. Consequently, attention drifts toward shortcut cues, leading to shortcut-dominated recognition and systematically poor tail generalization.

Motivated by these observations, we propose Shortcut-Resistant CAM Distillation (SRCD). Different from prior head-to-tail transfer (Li et al., 2024) that propagates knowledge in the feature/logit space, we intervene at the evidence level to encourage learning object-centric cues that are essential for tail recognition. Intuitively, since head classes are trained with abundant and diverse examples, their CAMs are more likely to reflect stable object evidence rather than spurious shortcuts. SRCD constructs a shortcut-resistant teacher for each tail sample by aggregating CAMs from a small set of predicted head-class candidates, and distills it to the tail-class CAM. To ensure reliability, SRCD assigns each candidate a lightweight weight based on coherence (Laplacian smoothness) and concentration (Hoyer sparsity), discouraging fragmented or overly diffuse explanations. SRCD is plug-and-play and can be seamlessly combined with existing long-tailed learning methods to further reduce shortcut reliance and improve tail generalization.

Our contributions are three-fold:

• We revisit long-tailed recognition from a shortcutlearning perspective and identify a systematic head– tail gap where tail classes are more prone to shortcutdominated attention.

• We propose SRCD, a simple and compatible CAM distillation strategy that transfers object-focused explanations from head to tail classes.

• We provide theoretical support and extensive experiments showing that SRCD effectively suppresses shortcut reliance and consistently improves performance on long-tailed benchmarks.

## 2. Related Work

Long-tailed Recognition. Long-tailed recognition aims to improve underrepresented tail classes while maintaining head accuracy. To alleviate tail scarcity, a promising direction is head-to-tail transfer that injects diversity into tail categories, e.g., by synthesizing tail-like samples/features (M2m (Kim et al., 2020), RSG (Wang et al., 2021)) or transferring distributional statistics and contexts to enrich tail representations (FASA (Zang et al., 2021), LTA (Yin et al., 2019), CMO (Park et al., 2022), H2T (Li et al., 2024)). Different from these approaches that mainly operate in the sample/feature/statistics space, our method intervenes at the evidence level, encouraging tail predictions to rely on object-centric cues rather than shortcuts. For completeness, we provide more related work and discussion in the appendix A.1.

Shortcut Learning. Shortcut learning (Geirhos et al., 2020) describes a common failure mode where deep networks rely on spurious but predictive cues (e.g., background (Beery et al., 2018) context or textures (Geirhos et al., 2018)) instead of semantic evidence. Prior work has studied such shortcuts in standard recognition settings (Hermann et al., 2024) and proposed interventions via augmentation, bias mitigation (Sagawa et al., 2019). However, the role of shortcut learning under long-tailed distributions remains largely unexplored: tail classes suffer from scarce and biased supervision, making shortcuts unusually consistent and easy to fit, yet existing long-tailed methods rarely address this issue explicitly. We provide more related work in Appendix A.2.

## 3. Methodology

## 3.1. Problem Setup

Let X be the input space and $\mathcal { Y } = \{ 1 , 2 , \ldots , C \}$ the label space. We focus on long-tailed recognition, where the training set $\pmb { S } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ is drawn from an imbalanced distribution $\mathcal { D } ,$ , with $x _ { i } \in \mathcal X$ and $y _ { i } \in \mathcal { V }$ . The total number of instances is n. For each class $y \in \mathcal { V }$ , let $n _ { y }$ denote the number of training examples for that class, and define the empirical class prior as $\pi _ { y } = n _ { y } / n$ . We assume that classes are sorted by frequency such that $n _ { 1 } \geq n _ { 2 } \geq \dots \geq n _ { C }$ with $n _ { 1 } \gg n _ { C }$ , capturing the head-to-tail imbalance. Following previous work (Zhang et al., 2021b), we use a frequency threshold $n _ { \mathrm { t h r e s } } ~ = ~ 1 0 0$ to partition classes into head and tail groups: $\mathcal { V } _ { \mathrm { h e a d } } ~ = ~ \left\{ y ~ \mid ~ n _ { y } ~ \geq ~ n _ { \mathrm { t h r e s } } \right\}$ and $\mathcal { V } _ { \mathrm { t a i l } } ~ = ~ \{ y ~ \vert ~ n _ { y } ~ < ~ n _ { \mathrm { t h r e s } } \}$ . Note that the above binary partition is used for applying SRCD during training. For split-wise evaluation, we follow the common reporting protocol where Head, Med, and Tail denote classes with more than 100, 20–100, and fewer than 20 training samples, respectively.

Following common long-tailed evaluation protocols (Cao et al., 2019), we consider a balanced test distribution $\mathcal { D } _ { b a l }$ that assigns equal probability to each class. We further assume that D and $\mathcal { D } _ { b a l }$ share the same class-conditional distributions, i.e., $\mathcal { D } ( x \mid y ) = \mathcal { D } _ { b a l } ( x \mid y )$ , for $\forall y \in \mathcal { V }$ , and only differ in their class priors.

Let $f ( \cdot ; \theta )$ be a classifier parameterized by $\theta \in \mathbb { R } ^ { k }$ , and let $\ell ( \theta ; x , y )$ denote the per-example loss. The empirical risk on S is

$$
L _ { S } ( \theta ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( \theta ; x _ { i } , y _ { i } ) .
$$

At the distribution level, the expected risks under D and $\mathcal { D } _ { b a l }$ are given by $L _ { \mathcal { D } } ( \theta ) ~ = ~ \mathbb { E } _ { ( x , y ) \sim \mathcal { D } } [ \ell ( \theta ; x , y ) ]$ , and $L _ { \mathcal { D } _ { b a l } } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { b a l } } [ \ell ( \theta ; x , y ) ]$ , respectively.

Our goal is to learn parameters θ from the long-tailed training set S such that the classifier achieves low risk on the balanced test distribution, i.e., minimizing $L _ { \mathcal { D } _ { b a l } } ( \theta )$ , which reflects strong performance across both head and tail classes.

## 3.2. Revisiting Class Activation Maps

To analyze shortcut learning in long-tailed recognition, we need a mechanism to identify where a model focuses its class-specific evidence. Standard metrics fall short, as models can predict correctly while relying on spurious cues. Since shortcut learning manifests in spatial attention, feature-level interpretability is crucial.

Class Activation Maps (CAMs) (Zhou et al., 2016) provide a principled way to expose such behavior by producing classspecific spatial explanations for a given input. For an image x and class $c ,$ CAM highlights regions that contribute most to the prediction, making it a suitable tool for diagnosing whether the model attends to meaningful object evidence or shortcut cues.

Formally, let $\{ A _ { k } ( x ) \} _ { k = 1 } ^ { K }$ denote the feature maps from a convolutional layer. The class activation map for class c is defined as

$$
M _ { c } ( \boldsymbol { x } ) = \phi \left( \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { c } ( \boldsymbol { x } ) A _ { k } ( \boldsymbol { x } ) \right) \in \mathbb { R } ^ { H \times W } ,
$$

where $\alpha _ { k } ^ { c } ( x )$ measures the contribution of the k-th feature map to class c, and $\phi ( \cdot )$ is a non-linear operation (e.g., ReLU). In practice, $\alpha _ { k } ^ { c } ( x )$ can be obtained using standard CAM (Zhou et al., 2016), which require no architectural modifications and generalize well across models.

To facilitate quantitative analysis, we interpret CAMs as spatial evidence distributions. Denoting by $M _ { c } ( x ) _ { \imath }$ the activation at spatial location $u \in \{ 1 , \dots , H W \}$ , we normalize the CAM as

$$
\bar { M } _ { c } ( x ) = \frac { M _ { c } ( x ) } { \sum _ { u } M _ { c } ( x ) _ { u } + \varepsilon } ,\tag{1}
$$

The normalized map $\bar { M } _ { c } ( x )$ represents the fraction of classspecific evidence allocated to each location. This formulation enables direct comparison of attention patterns across classes and examples, and serves as the foundation for our shortcut diagnosis and distillation framework.

## 3.3. Diagnosing Shortcut Learning in Long-Tailed Recognition

To empirically diagnose shortcut learning in long-tailed recognition, we leverage CAMs as a spatial indicator of what evidence a model relies on. Our key hypothesis is that models trained on a balanced dataset are more likely to learn object-centric semantics, since abundant and diverse supervision reduces the chance of overfitting to spurious correlations. Therefore, CAMs produced by a balanced model can serve as a reasonable reference for semantic explanations, while deviations from this reference may reflect shortcut-dominated attention. A more detailed discussion on why CAM alignment with the balanced reference is a reasonable proxy for shortcut reliance (together with limitations) is deferred to Appendix B.1.

Concretely, we train a reference model on the full balanced CIFAR100 (Krizhevsky et al., 2009) and another model on its long-tailed counterpart CIFAR100-LT (Cao et al., 2019). For the same input example x with label $y ,$ we compute the class-specific CAMs from both models, denoted as $M _ { y } ^ { b a l } ( x )$ and $M _ { y } ^ { l t } ( x )$ , respectively, and measure their alignment via cosine similarity:

![](images/a35583db1d4070b2b80e15966b569668261ec9338793fced6b0bf53144de492d.jpg)  
Figure 2. CAM similarity analysis. LT–Balanced (GT) denotes the cosine similarity between CAMs produced by the long-tailed model and the balanced model for the same examples under their ground-truth classes. LT–Balanced (Head-Proxy) computes CAMs for tail examples using the most probable head-class predictions from the long-tailed model and compares them to the balanced-model CAMs. Shaded regions indicate standard deviation.

$$
\mathrm { S i m } ( x , y ) = \cos \bigl ( \mathrm { v e c } ( \bar { M } _ { y } ^ { l t } ( x ) ) , \mathrm { v e c } ( \bar { M } _ { y } ^ { b a l } ( x ) ) \bigr ) ,\tag{2}
$$

where M<sup>¯</sup> indicates normalization as in Eq. (1), and vec(·) vectorizes the spatial map. Intuitively, a high similarity suggests that the long-tailed model attends to regions consistent with the balanced reference (i.e., less shortcut reliance), whereas a low similarity indicates that the long-tailed model may exploit alternative, non-semantic cues to explain the same class prediction.

Figure 2 reports the CAM cosine similarity across classes, sorted from head to tail according to training frequency. We observe that for head classes, the long-tailed model produces CAMs highly consistent with the balanced reference, indicating that shortcut reliance is relatively mild when sufficient supervision is available. In contrast, the similarity drops substantially on tail classes, revealing a systematic shift in spatial evidence under scarce training data. This supports the intuition that limited tail supervision makes spurious correlations easier to fit, leading to shortcut-dominated explanations and degraded tail generalization.

Moreover, we examine whether tail examples can benefit from object-centric explanations learned by head classes. For each tail example x, we select the most confident predicted head class <sup>ˆ</sup>k and compute its CAM $M _ { \widehat { k } } ^ { l t } ( x )$ . We then compare this head-proxy CAM to the balanced reference CAM $M _ { y } ^ { b a l } ( x )$ of the ground-truth class. As shown in Figure 2, the head-proxy CAM exhibits notably higher similarity on tail classes than the tail-class CAM itself. This suggests that borrowing CAM cues from a likely head class can provide a more stable, less shortcut-prone spatial explanation for tail examples, motivating our shortcut-resistant CAM distillation framework.

## 3.4. Shortcut-Resistant CAM Distillation

In this section, we present Shortcut-Resistant CAM Distillation (SRCD), a training strategy that explicitly suppresses shortcut-dominated explanations for tail classes by distilling reliable spatial evidence from head classes. SRCD is motivated by our empirical findings that (i) tail-class CAMs learned under long-tailed supervision deviate substantially from those obtained under balanced training, and (ii) CAMs computed using predicted head classes provide more stable and semantically aligned explanations for tail examples.

Motivation. Under long-tailed distributions, tail classes suffer from limited and biased supervision, making it easy for models to rely on spurious but predictive cues to explain these classes. Such shortcut reliance manifests as fragmented or overly diffuse spatial attention. In contrast, head classes are trained with abundant and diverse examples, which reduces the consistency of spurious correlations and encourages the model to learn more robust visual evidence. SRCD leverages this asymmetry by constructing a shortcutresistant teacher explanation from head-class CAMs and transferring it to tail classes through spatial distillation.

Head-Class Proxy Selection. Given a training example (x, y) with $y \in \mathcal { V } _ { \mathrm { t a i l } }$ , we first compute the predictive distribution $p ( c \mid x )$ . From the head-class set $\mathcal { D } _ { \mathrm { h e a d } }$ , we select the top-K most probable classes:

$$
\mathcal { H } ( x ) = \arg \operatorname* { m a x } _ { c \in \mathcal { V } _ { \mathrm { h e a d } } } p ( c \mid x ) .
$$

Each $c \in \mathcal { H } ( x )$ defines a head-class proxy for $x ,$ and induces a candidate explanation $\bar { M } _ { c } ( x )$ As shown in Section 3.3, such head-proxy CAMs are empirically less shortcut-prone than tail-class CAMs computed using the ground-truth label. This suggests that head proxies can provide more stable spatial evidence for tail examples, even without enforcing strict semantic matching or explicit partlevel decomposition. We provide further discussion on this design in Appendix B.3.

Quantifying Shortcut Resistance of CAMs. Not all head-proxy CAMs are equally reliable. Some may still be noisy, fragmented, or overly diffuse. To identify explanations that are resistant to shortcut reliance, we evaluate each candidate CAM using two complementary structural properties.

![](images/46756bbca5fc4c53082f66afaf116ce4f441b57edc76b8e215f2b5de98c3bae8.jpg)  
Figure 3. SRCD pipeline. We generate class-conditioned CAMs from feature maps and classifier weights, form a shortcut-resistant teacher CAM by energy-weighting a set of head-proxy CAMs (with stop-gradient used only in this proxy branch), and distill the teacher to the ground-truth class CAM via $\mathcal { L } _ { \mathrm { S R C D } }$ to reduce shortcut/background attention alongside ${ \mathcal L } _ { \mathrm { c l s } }$

(1) Spatial Coherence Energy. Shortcut-driven explanations often manifest as scattered activations over multiple disconnected regions, reflecting reliance on incidental cues rather than consistent visual evidence. To penalize such behavior, we measure the spatial coherence of a CAM using a Laplacian-based energy: $\mathcal { E } _ { \mathrm { h } } ( M ) = M ^ { \top } L M$ , where $\overset { \smile } { M } \in \mathbb { R } ^ { \grave { H } W }$ is the vectorized CAM and L is the graph Laplacian defined on the $H \times W$ grid. Specifically, $L = D - G $ where G encodes 4-neighbor pixel adjacency and D is the corresponding degree matrix. Lower coherence energy indicates smoother and more contiguous spatial evidence, which is less characteristic of shortcut-driven attention.

(2) Evidence Concentration Score. Shortcut learning may also result in evidence being diffusely spread across large regions, particularly when background cues dominate the prediction. To discourage such behavior, we measure how concentrated the evidence is using a normalized sparsity score:

$$
\mathcal { E } _ { \mathrm { n } } ( M ) = \frac { \sqrt { N } - \| M \| _ { 1 } / \| M \| _ { 2 } } { \sqrt { N } - 1 } ,
$$

where $N = H W , \| M \| _ { 1 }$ and $\| M \| _ { 2 }$ are the $L _ { 1 }$ and $L _ { 2 }$ norms of the vectorized CAM. For nonnegative CAMs, $\| M \| _ { 1 } / \| M \| _ { 2 }$ increases when evidence spreads over many locations and decreases when evidence concentrates, making $\mathcal { E } _ { \mathrm { n } } ( M ) \in [ 0 , 1 ]$ a normalized concentration measure.

Together, ${ \mathcal E } _ { \mathrm { h } }$ and ${ \mathcal { E } } _ { \mathrm { n } }$ capture complementary aspects of shortcut resistance: spatial continuity and evidence compactness. For a more detailed discussion on the rationale and limitations of $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ , see Appendix B.2.

Energy-Based Teacher Aggregation. We aggregate headproxy CAMs into a single shortcut-resistant teacher by assigning each candidate an energy-based weight:

$$
w _ { c } ( x ) = \frac { \exp \left( \frac { - \mathcal { E } _ { \mathrm { h } } ( \bar { M } _ { c } ( x ) ) + \mathcal { E } _ { \mathrm { n } } ( \bar { M } _ { c } ( x ) ) } { \tau } \right) } { \sum _ { c ^ { \prime } \in \mathcal { H } ( x ) } \exp \left( \frac { - \mathcal { E } _ { \mathrm { h } } ( \bar { M } _ { c ^ { \prime } } ( x ) ) + \mathcal { E } _ { \mathrm { n } } ( \bar { M } _ { c ^ { \prime } } ( x ) ) } { \tau } \right) } .
$$

This formulation favors CAMs that are both spatially coherent and concentrated, while suppressing candidates that exhibit fragmented or diffuse patterns.

The resulting teacher CAM is defined as

$$
\bar { M } _ { \mathrm { T } } ( x ) = \sum _ { c \in \mathcal { H } ( x ) } w _ { c } ( x ) \bar { M } _ { c } ( x ) .
$$

Adaptive Distillation to Tail Classes. Let $\bar { M } _ { y } ( x )$ denote the normalized CAM for the tail class y and ${ \bar { M } } _ { \mathrm { T } } ( x )$ the shortcut-resistant teacher CAM. We align tail explanations with the teacher using an $L _ { 2 }$ distillation loss weighted by a class-dependent factor:

$$
\mathcal { L } _ { \mathrm { S R C D } } ( x , y ) = \gamma _ { y } \big | \big | \bar { M } _ { y } ( x ) - \mathrm { s g } \left[ \bar { M } _ { \mathrm { T } } ( x ) \right] \big | \big | _ { 2 } ^ { 2 } ,\tag{3}
$$

where $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operation, and $\gamma _ { y }$ is a class-dependent coefficient that increases as class y becomes rarer, reflecting the fact that tail classes are more prone to shortcut learning and thus require stronger regularization. In our implementation, we set $\gamma _ { y } ~ = ~ n _ { \mathrm { t h r e s } } / n _ { y }$ , with $n _ { y }$ the number of training examples for class y and $n _ { \mathrm { t h r e s } }$ the head–tail threshold. This choice reflects the intuition that tail classes with fewer examples are more prone to shortcut learning and therefore require stronger regularization.

Overall Objective. We optimize

$$
\mathcal { L } ( x , y ) = { \mathcal { L } } _ { \mathrm { c l s } } ( x , y ) + \mathbb { I } [ y \in \mathcal { V } _ { \mathrm { t a i l } } ] \ \beta \mathcal { L } _ { \mathrm { S R C D } } ( x , y ) ,
$$

where ${ \mathcal { L } } _ { \mathrm { c l s } }$ can be any long-tailed re-balancing loss $( \mathrm { e . g . }$ LDAM (Cao et al., 2019)) and $\beta$ controls the distillation strength. SRCD is architecture-agnostic and can be plugged into existing long-tailed methods. Since SRCD relies on reliable head-class explanations, we enable it only in the late training stage, which also keeps the overall training lightweight. Additional schedule ablations are deferred to Appendix D.6.

## 4. Theoretical Analysis

We provide a shortcut-centric analysis of SRCD in the CAM space. Our key idea is to measure shortcut learning as the attention mass assigned to shortcut/background regions, and show that SRCD reduces it in a principled way: (i) an energy-weighted teacher is constructed to favor coherent and concentrated (thus object-centric) evidence, and (ii) $L _ { 2 }$ CAM distillation induces a contraction dynamics that monotonically transports shortcut-region attention mass from tail CAMs toward this teacher. Together, these results explain why SRCD is effective, while we defer detailed proofs and extended discussions to the appendix C.

Let $\bar { M } _ { y } ( x ) \ \in \ \mathbb { R } ^ { H \times W }$ denote the normalized CAM for class $y ,$ where $\bar { M } _ { y } ( x ) \ge 0$ and $\begin{array} { r } { \sum _ { u \in \Omega } \bar { M } _ { y } ( x ) _ { u } = 1 } \end{array}$ . We formalize shortcut reliance as the CAM mass on a shortcut/background region $\Omega _ { \mathrm { s c } } \subset \Omega$

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ) = \sum _ { u \in \Omega _ { \mathrm { s c } } } \bar { M } _ { u } \in [ 0 , 1 ] .\tag{4}
$$

Energy-weighted teacher as a principled utility maximizer. SRCD constructs a shortcut-resistant teacher CAM by aggregating head-proxy candidates $\{ \bar { M } _ { c } ( x ) \} _ { c \in \mathcal { H } ( x ) }$ with the following weights:

$$
\begin{array} { l } { \displaystyle \bar { M } _ { T } ( x ) = \sum _ { c \in \mathcal { H } ( x ) } w _ { c } ( x ) \bar { M } _ { c } ( x ) , } \\ { \displaystyle w _ { c } ( x ) = \frac { \exp \bigl ( \mathcal { U } ( \bar { M } _ { c } ( x ) ) \bigr ) } { \sum _ { c ^ { \prime } \in \mathcal { H } ( x ) } \exp \bigl ( \mathcal { U } \bigl ( \bar { M } _ { c ^ { \prime } } ( x ) \bigr ) \bigr ) } , } \end{array}\tag{5}
$$

where the utility is defined by

$$
\mathcal { U } ( \bar { M } ) = - \mathcal { E } _ { h } ( \bar { M } ) + \mathcal { E } _ { n } ( \bar { M } ) ,
$$

with $\mathcal { E } _ { h }$ encouraging spatial coherence and ${ \mathcal { E } } _ { n }$ encouraging evidence concentration.

Theorem 4.1 (Entropy-regularized utility maximization). Let $\mathcal { U } _ { c } = \mathcal { U } ( \bar { M } _ { c } ( x ) ) f o r c \in \mathcal { H } ( x )$ . The softmax weights in (5) are the unique maximizer ofthe strictly concave objective

$$
\operatorname* { m a x } _ { w \in \Delta } \ \sum _ { c } w _ { c } \mathcal { U } _ { c } \ - \ \tau \sum _ { c } w _ { c } \log w _ { c } ,\tag{6}
$$

where $\begin{array} { r } { \Delta = \{ w : w _ { c } \geq 0 , \sum _ { c } w _ { c } = 1 \} } \end{array}$ and $\tau > 0$ controls the smoothness of aggregation.

Eq. (6) shows that teacher construction is principled: it prefers CAMs with low coherence energy $\mathcal { E } _ { h }$ (more spatially consistent evidence) and high concentration score ${ \mathcal { E } } _ { n }$ (more object-centric evidence), while the entropy term prevents overly brittle selection.

$L _ { 2 }$ CAM distillation induces shortcut-mass transport. SRCD distills the teacher into the tail-class CAM using Eq (3) where $\gamma _ { y }$ increases with tail scarcity.

Theorem 4.2 (Monotone shortcut-mass contraction). Consider the gradient flow that minimizes the $L _ { 2 }$ distillation energy:

$$
\frac { d } { d t } \bar { M } ( t ) \ : = \ : - \nabla _ { \bar { M } } \big \| \bar { M } ( t ) - \bar { M } _ { T } \big \| _ { 2 } ^ { 2 } .\tag{7}
$$

Then $\bar { M } ( t )$ admits a closed-form solution $\bar { M } ( t ) = \bar { M } _ { T } +$ $e ^ { - 2 t } \big ( \bar { M } ( 0 ) - \bar { M } _ { T } \big )$ , and

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( t ) ) = \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) + e ^ { - 2 t } \Big ( \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( 0 ) ) - \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) \Big ) . } \end{array}\tag{8}
$$

In particular, $i f \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( 0 ) ) > \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } )$ , then $\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( t ) )$ decreases monotonically and converges exponentially to $\bar { \mathcal { A } } _ { \mathrm { s c } } ( \bar { M } _ { T } )$

Eq. (8) shows that $L _ { 2 }$ distillation makes $\bar { M } ( t )$ exponentially contract to $\bar { M _ { T } }$ . Since $A _ { \mathrm { s c } } ( \cdot )$ is a linear mass on $\Omega _ { \mathrm { s c } } .$ , the shortcut attention is transported toward $\bar { \mathcal { A } } _ { \mathrm { s c } } ( \bar { M } _ { T } )$ at the same rate; thus if the teacher is more object-centric, shortcut reliance decreases monotonically.

SRCD suppresses tail shortcuts. If the energy-weighted teacher is shortcut-resistant (i.e., $\bar { \mathcal { A } } _ { \mathrm { s c } } ( \bar { M } _ { T } )$ is small), then minimizing Eq (3) drives the tail CAM $\bar { M } _ { y } ( x )$ toward $\bar { M } _ { T } ( x )$ and consequently reduces shortcut reliance $\bar { \mathcal { A } } _ { \mathrm { s c } } ( \bar { M } _ { y } ( x ) )$ toward $\mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ( x ) )$ .

## 5. Experiments

## 5.1. Experiment Protocols

Datasets. We evaluate our method on five widely adopted long-tailed recognition benchmarks: CIFAR-10-LT and

Table 1. Training-from-scratch results on CIFAR-10-LT and CIFAR-100-LT under imbalance ratios $\mathbf { I R } \in \{ 1 0 0 , 5 0 , 1 0 \}$ (Top-1 accuracy, %). +SRCD denotes integrating SRCD into each baseline; superscripts show absolute changes (↑ /↓ for gain/drop).
<table><tr><td>Method</td><td colspan="3">CIFAR100 IR=100 IR=50 IR=10</td><td colspan="3">CIFAR10 IR=50 IR=10</td></tr><tr><td>BCL (Zhu et al., 2022)</td><td>51.76</td><td>56.51</td><td>67.90</td><td> $\scriptstyle \mathrm { I R } = 1 0 0$  83.61</td><td>86.13</td><td>90.10</td></tr><tr><td>+SRCD GCL (Li et al., 2022b)</td><td> $5 3 . 6 4 \mathrm { \uparrow } 1 . 8 8 $  46.50</td><td> $5 6 . 8 1 \mathrm { \Omega } \mathrm { \Omega } \mathrm { \Omega } 0 . 3 0$  51.72</td><td> $6 8 . 2 2 \uparrow 0 . 3 2$  61.79</td><td> $8 2 . 9 7 \downarrow 0 . 6 4 $  80.56</td><td> $8 7 . 8 7 \mathrm { \uparrow } 1 . 7 4 $  84.74</td><td> $9 1 . 2 3 \substack { \uparrow 1 . 1 3 }$  89.65</td></tr><tr><td>+SRCD CE+FCC (Li et al., 2023)</td><td> $4 7 . 0 7 \substack { \uparrow 0 . 5 7 }$  40.20</td><td> $5 2 . 3 2 \mathrm { \uparrow } 0 . 6 0$  45.93</td><td> $6 0 . 2 3 \downarrow 1 . 5 6$  57.80</td><td> $8 1 . 1 1 \substack { \uparrow 0 . 5 5 }$  73.80</td><td> $8 5 . 6 9 \substack { \uparrow 0 . 9 5 }$  79.57</td><td> $8 8 . 9 0 \downarrow 0 . 7 5 $  87.75</td></tr><tr><td>+SRCD GLMC (Du et al., 2023)</td><td> $4 2 . 3 8 \uparrow 2 . 1 8$  53.91</td><td> $4 7 . 1 7 \substack { \uparrow 1 . 2 4 }$  58.87</td><td> $5 8 . 2 9 \substack { \uparrow 0 . 4 9 }$  68.07</td><td> $7 5 . 2 9 \substack { \uparrow 1 . 4 9 }$  83.68</td><td> $8 0 . 4 6 \mathrm { \cdot } 0 . 8 9$  86.90</td><td> $8 8 . 0 5 \mathrm { \cdot } 0 . 3 0$  91.16</td></tr><tr><td>+SRCD CE+ H2T (Li et al., 2024)</td><td> $5 5 . 2 6 \substack { \mathrm { \uparrow } 1 . 3 5 }$  42.27</td><td> $6 0 . 4 3 \substack { \uparrow 1 . 5 6 }$  47.58</td><td> $6 9 . 0 1 \substack { \uparrow 0 . 9 4 }$  58.24</td><td> $8 3 . 9 7 \mathrm { \Omega } _ { \mathrm { { f 0 . 2 9 } } }$  79.91</td><td> $8 7 . 7 2 \substack { \uparrow 0 . 8 2 }$  82.80</td><td> $9 0 . 7 8 \downarrow 0 . 3 8$  88.77</td></tr><tr><td>+SRCD BNS (Lin &amp; Yuan, 2025)</td><td> $4 4 . 5 3 \substack { \uparrow 2 . 2 6 }$  52.40</td><td> $4 6 . 9 7 \scriptstyle \downarrow 0 . 6 1$  55.90</td><td> $6 0 . 0 7 \substack { \uparrow 1 . 8 3 }$  62.60</td><td> $8 0 . 4 3 \substack { \uparrow 0 . 5 2 }$  76.40</td><td> $8 3 . 2 9 \mathrm { \uparrow 0 . 4 9 }$  81.90</td><td> $8 9 . 2 1 \substack { \uparrow 0 . 4 4 }$  83.70</td></tr><tr><td>+SRCD Focal-SAM (Li et al., 2025)</td><td> $5 3 . 8 8 \mathrm { \uparrow } 1 . 4 8 $ </td><td> $5 6 . 1 2 \mathrm { \Omega } \mathrm { \Omega } \mathrm { \Omega } 0 . 2 2$ </td><td> $6 3 . 0 4 \mathrm { \uparrow } 0 . 4 4$ </td><td> $7 7 . 3 6 \mathrm { \cdot } 0 . 9 6$ </td><td> $8 1 . 6 6 \downarrow 0 . 2 4$ </td><td> $8 4 . 8 5 \substack { \uparrow 1 . 1 5 }$ </td></tr><tr><td>+SRCD</td><td>50.70  $5 2 . 5 4 \mathrm { \uparrow } 1 . 8 4$ </td><td>54.50  $5 5 . 2 9 \mathrm { \Omega } _ { \mathrm { T 0 . 7 9 } }$ </td><td>63.80  $6 4 . 7 1 \substack { \mathrm { \sim } 0 . 9 1 }$ </td><td>82.90  $8 4 . 2 9 \mathrm { \uparrow } 1 . 3 9$ </td><td>85.50  $8 6 . 0 4 \mathrm { \uparrow } 0 . 5 4$ </td><td>90.50  $9 0 . 7 7 \substack { \uparrow 0 . 2 7 }$ </td></tr></table>

CIFAR-100-LT (Cao et al., 2019), ImageNet-LT (Liu et al., 2019), iNaturalist 2018 (Van Horn et al., 2018), and Places-LT (Liu et al., 2019). We defer details to Appendix D.1.

![](images/df24f887fd41adfe1355f2198c0daa844e346abca2ef04284a32765f37c5690b.jpg)

![](images/6910f64ec8c9528df65d06d8a2c1cb3ddd8b1d0b1cd265ecc50c565132902e1a.jpg)  
(a) CIFAR100-LT (IR=100)

Evaluation Settings and Competitors. We evaluate our method under two practical settings: training from scratch and fine-tuning foundation models. For training from scratch, we compare against representative long-tailed learning baselines, including BCL (Zhu et al., 2022), GCL (Li et al., 2022b), FCC (Li et al., 2023), GLMC (Du et al., 2023), H2T (Li et al., 2024), BNS (Lin & Yuan, 2025), and Focal-SAM (Li et al., 2025). For fine-tuning foundation models, we follow the standard CLIP adaptation protocol and compare with recent methods such as Decoder (Wang et al., 2024b), PECEL (Ru et al., 2024) and LIFT (Shi et al., 2024). Notably, our method is a plug-and-play component that can be seamlessly combined with arbitrary long-tailed learning strategies across both settings.

(b) CIFAR100-LT (IR=50)

Figure 4. Ablation study of SRCD w.r.t. β.

Implementation Details. We report results under two settings: trainingfrom scratch andfine-tuningfoundation models. For the scratch setting, we adopt ResNet-32 (He et al., 2016) on CIFAR-LT and train for 200 epochs, while using ResNet-50 (He et al., 2016) on ImageNet-LT (Liu et al., 2019) and iNaturalist (Van Horn et al., 2018) with the same training schedule. For foundation model fine-tuning, we follow the protocol in LIFT (Shi et al., 2024) and finetune the image encoder of CLIP (Radford et al., 2021) with a ViT-B/16 (Dosovitskiy, 2020) backbone for 20 epochs. Additional training details are provided in the Appendix.

## 5.2. Results with training from scratch

We report training from scratch results on CIFAR10/100- LT (Table 1) and large-scale long-tailed benchmarks (upper block of Table 2), where SRCD is integrated into a wide range of representative baselines. Overall, SRCD delivers consistent gains across imbalance ratios and datasets, indicating that explanation-level shortcut suppression is complementary to existing long-tailed objectives rather than method-specific. On CIFAR100-LT, adding SRCD improves most baselines under $\mathrm { I R } \in \{ 1 0 0 , 5 0 , 1 0 \}$ , with larger gains typically observed in the more challenging regimes, while CIFAR10-LT shows similarly stable improvements across skews. This trend scales to ImageNet-LT and iNaturalist: SRCD consistently increases the overall accuracy for all evaluated methods and yields particularly strong improvements on Medium/Tail splits, which aligns with our motivation that data-scarce classes are more prone to relying on shortcut/background cues. Additional split-wise results are provided in Appendix D.3, offering a more detailed analysis of SRCD’s effect on tail-class performance.

Table 2. Results on ImageNet-LT and iNaturalist, reporting Top-1 accuracy (%) on Head/Med/Tail splits and All. The upper block is training from scratch, and the lower block is fine-tuning foundation models. +SRCD denotes integrating SRCD into each baseline; superscripts show absolute changes $( \uparrow / \downarrow$ for gain/drop).
<table><tr><td rowspan="2">Method</td><td colspan="4">ImageNet-LT</td><td colspan="4">iNaturalist</td></tr><tr><td>Med</td><td></td><td>Tail</td><td></td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td colspan="9">Training from scratch</td></tr><tr><td>BCL +SRCD</td><td>65.83  $6 6 . 2 9 \mathrm { \uparrow 0 . 4 6 }$ </td><td>53.16  $5 4 . 4 7 \substack { \uparrow 1 . 3 1 }$ </td><td>36.27  $3 9 . 5 6 \substack { \uparrow 3 . 2 9 }$ </td><td>55.67  $5 7 . 0 0 \mathrm { \uparrow } 1 . 3 3$ </td><td>73.47  $7 4 . 2 6 \mathrm { \cdot } 0 . 7 9$ </td><td>71.99  $7 2 . 6 5 \substack { \mathrm { \uparrow 0 . 6 6 } }$ </td><td>68.82  $7 0 . 3 3 \substack { \uparrow 1 . 5 1 }$ </td><td>70.80  $7 1 . 8 9 \mathrm { ‰ }$ </td></tr><tr><td>GCL +SRCD</td><td>65.04  $6 4 . 8 3 \scriptstyle \downarrow 0 . 2 1$ </td><td>52.77  $5 3 . 6 2 \mathrm { \uparrow 0 . 8 5 }$ </td><td>35.69  $3 7 . 1 0 \mathrm { \uparrow } 1 . 4 1$ </td><td>55.51  $5 5 . 6 9 \mathrm { \uparrow 0 . 1 8 }$ </td><td>73.75  $7 4 . 2 8 \mathrm { \uparrow 0 } . 5 3$ </td><td>72.28  $7 4 . 0 2 \substack { \uparrow 1 . 7 4 }$ </td><td>68.34  $7 0 . 3 1 \substack { \mathrm { : 1 . 9 7 } }$ </td><td>70.76  $7 2 . 5 8 \substack { \mathrm { \uparrow 1 . 8 2 } }$ </td></tr><tr><td>CE+FCC +SRCD</td><td>65.52  $6 6 . 1 5 \substack { \mathrm { \uparrow 0 . 6 3 } }$ </td><td>52.18  $5 3 . 2 9 \mathrm { \uparrow 1 . 1 1 }$ </td><td>33.83  $3 6 . 4 8 \substack { \uparrow 2 . 6 5 }$ </td><td>54.68  $5 5 . 9 5 \substack { \mathrm { \uparrow } 1 . 2 7 }$ </td><td>72.78  $7 2 . 7 9 \mathrm { \div 0 . 0 1 }$ </td><td>71.09  $7 0 . 8 4 \downarrow 0 . 2 5$ </td><td>67.41  $6 8 . 8 7 \substack { \uparrow 1 . 4 6 }$ </td><td>69.70  $7 0 . 2 6 \mathrm { { \scriptstyle \uparrow 0 . 5 6 } }$ </td></tr><tr><td>GLMC +SRCD</td><td>69.42  $7 0 . 4 3 \substack { \uparrow 1 . 0 1 }$ </td><td>52.49  $5 2 . 1 7 _ { \downarrow 0 . 3 2 }$ </td><td>30.41  $3 4 . 2 2 \substack { \uparrow 3 . 8 1 }$ </td><td>56.17  $5 6 . 7 6 \mathrm { ‰}$ </td><td>76.04  $7 6 . 7 7 \substack { \mathrm { \uparrow 0 . 7 3 } }$ </td><td>73.69  $7 3 . 2 1 \downarrow 0 . 4 8$ </td><td>71.77  $7 3 . 0 6 \mathrm { \cdot } 1 . 2 9$ </td><td>73.14  $7 3 . 5 2 \mathrm { \uparrow 0 } . 3 8 $ </td></tr><tr><td>Focal-SAM +SRCD</td><td>63.90  $6 3 . 9 9 \mathrm { \uparrow 0 . 0 9 }$ </td><td>52.20  $5 2 . 7 1 \substack { \mathrm { \sim } 0 . 5 1 }$ </td><td>34.40  $3 6 . 6 7 \substack { \uparrow 2 . 2 7 }$ </td><td>54.30  $5 4 . 8 7 \substack { \uparrow 0 . 5 7 }$ </td><td>68.40  $6 9 . 0 3 \substack { \uparrow 0 . 6 3 }$ </td><td>72.00  $7 3 . 1 4 \uparrow 1 . 1 4$ </td><td>72.50  $7 2 . 4 3 \downarrow 0 . 0 7$ </td><td>71.80  $7 2 . 4 3 \substack { \mathrm { \uparrow 0 . 6 3 } }$ </td></tr><tr><td colspan="9">Fine-tuning foundation model</td></tr><tr><td>Decoder +SRCD</td><td>76.0  $7 6 . 3 \substack { \mathrm { : } 0 . 3 }$ </td><td>72.1  $7 3 . 4 \uparrow 1 . 3$ </td><td>69.3  $7 1 . 2 \substack { \mathrm { \uparrow 1 . 9 } }$ </td><td>73.2  $7 4 . 2 \substack { \mathrm { \uparrow } 1 . 0 }$ </td><td>55.1  $5 8 . 7 \substack { \cdot 3 . 6 }$ </td><td>58.6</td><td>61.1</td><td>59.2</td></tr><tr><td>PECEL</td><td>81.3</td><td>77.6</td><td>73.0</td><td>78.3</td><td>76.8</td><td> $6 1 . 7 \substack { \uparrow 3 . 1 }$  80.6</td><td> $6 1 . 4 \uparrow 0 . 3$  81.3</td><td> $6 1 . 3 \substack { \uparrow 2 . 1 }$  80.5</td></tr><tr><td>+SRCD LIFT</td><td> $8 1 . 8 \substack { \mathrm { \uparrow 0 . 5 } }$  79.7</td><td> $7 7 . 5 \downarrow 0 . 1 $ </td><td> $7 4 . 6 \substack { \mathrm { \uparrow 1 . 6 } }$ </td><td> $7 8 . 8 \mathrm { \uparrow } 0 . 5$  77.1</td><td> $7 7 . 4 \substack { \mathrm { \uparrow 0 . 6 } }$  74.1</td><td> $8 1 . 2 \substack { \mathrm { \uparrow 0 . 6 } }$  79.4</td><td> $8 2 . 3 \substack { \mathrm { \uparrow 1 . 0 } }$  81.5</td><td> $8 1 . 2 \substack { \mathrm { \uparrow 0 . 7 } }$  79.7</td></tr></table>

(a) Effect of K  
![](images/5506ada69cdc3e50c7849a0dac36a2964cd80946564482c31d403f0698583e8c.jpg)

![](images/b5f0e6694836873e870303b0c95f9a622ddeca76cf187864d84951cded687cf4.jpg)  
(b) Averaging vs. Ours teacher

Table 3. Ablation study on the teacher-weighting cues $\mathcal { E } _ { h }$ and $\mathcal { E } _ { n }$ in SRCD, evaluated on ImageNet-LT with the FCC.
<table><tr><td>Coherence  $\mathcal { E } _ { h }$ </td><td>Concentration  ${ \mathcal { E } } _ { n }$ </td><td> $\operatorname { A c c . }$ </td></tr><tr><td>√</td><td>X</td><td>54.02</td></tr><tr><td>x</td><td>√</td><td>53.64</td></tr><tr><td>√</td><td>√</td><td>55.95</td></tr></table>

Figure 5. Ablation study of K and energy-based weighting.

## 5.3. Results with fine-tuning foundation models

We further evaluate SRCD under thefoundation modelfinetuning setting (lower block of Table 2), where we plug SRCD into strong long-tailed fine-tuning baselines without modifying architectures. Additional Places-LT results are provided in Appendix D.3, further demonstrating the generality of SRCD on large-scale long-tailed benchmarks. Across all evaluated methods, SRCD consistently improves the overall accuracy on both ImageNet-LT and iNaturalist, while delivering clear gains on Medium/Tail splits, which are typically the most sensitive to shortcut reliance. In particular, SRCD provides substantial tail improvements when paired with competitive fine-tuning pipelines (e.g., PECEL and LIFT), and the gains are achieved alongside stable or improved Head performance, suggesting that SRCD enhances non-head generalization without trading off head-class accuracy. These results support SRCD as a lightweight, plugand-play regularizer that remains effective when adapting foundation models to long-tailed distributions.

## 5.4. Ablation Study

Impact of β. We vary $\beta$ to control the distillation strength (Figure 4). Notably, when β exceeds 0.6, we observe a significant and stable improvement in accuracy, indicating that stronger distillation has a regularizing effect on tailclass attention.

Impact of K. We ablate the number of head-class proxies K used to build the teacher (Figure 5 (a)). Using $K = 3$ already provides most of the gains, and further increasing

K leads to stable but marginal improvements.

Impact of $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ . We toggle the coherence cue $\mathcal { E } _ { h }$ and concentration cue ${ \mathcal { E } } _ { n }$ in teacher weighting (Table 3). Using both cues achieves the best performance.

Comparison with Averaging Strategy. We compare our energy-based aggregation with simple averaging for teacher CAM construction (Figure 5 (b)) using Eq (2). Energybased weighting yields higher CAM similarity to the balanced reference model, indicating that the resulting teacher is more shortcut-resistant.

Comparison with Center Mask Prior. We compare SRCD with a center-mask baseline to test whether its gain comes from a trivial geometric prior. Replacing the head-proxy teacher with a fixed center mask leads to degraded performance, especially on the Tail split, while SRCD consistently improves the baseline. This shows that SRCD benefits from adaptive shortcut-resistant evidence transfer rather than simple center-biased attention. Full results are provided in Table 9.

Additional Analyses. We further report the computational overhead and provide qualitative CAM visualizations to better illustrate SRCD’s behavior, and these results are deferred to the Appendix.

## 6. Conclusion

We introduce SRCD, a method to reduce shortcut learning in long-tailed recognition. By distilling reliable spatial evidence from head classes, SRCD aligns tail-class explanations with more stable object features. Our experiments show that SRCD improves tail-class performance while maintaining accuracy on head classes, offering a robust regularizer for long-tailed learning tasks.

## Acknowledgements

This work was supported by the National Science and Technology Major Project (2024YFB3311401), the National Natural Science Foundation of China (62576168), Open Project Funds for the Joint Laboratory of Spatial Intelligent Perception and Large Model Application (SIPLMA-2024-YB-05), the Natural Science Foundation of Wuhan (2023010201020229), and the Fundamental Research Funds for the Central Universities (NO.NJ2023032).

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Agarwal, V., Shetty, R., and Fritz, M. Towards causal vqa: Revealing and reducing spurious correlations by invariant and covariant semantic editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9690–9698, 2020.

Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-Paz, D. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Beery, S., Van Horn, G., and Perona, P. Recognition in terra incognita. In Proceedings of the European conference on computer vision (ECCV), pp. 456–473, 2018.

Cao, K., Wei, C., Gaidon, A., Arechiga, N., and Ma, T. Learning imbalanced datasets with label-distributionaware margin loss. In Advances in Neural Information Processing Systems, 2019.

Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. Smote: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16:321–357, 2002.

Cui, J., Zhong, Z., Liu, S., Yu, B., and Jia, J. Parametric contrastive learning. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 715– 724, 2021.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Du, F., Yang, P., Jia, Q., Nan, F., Chen, X., and Yang, Y. Global and local mixture consistency cumulative learning for long-tailed visual recognitions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15814–15823, 2023.

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., and Brendel, W. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In International conference on learning representations, 2018.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F. A. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

Hermann, K., Mobahi, H., FEL, T., and Mozer, M. C. On the foundations of shortcut learning. In The Twelfth International Conference on Learning Representations, 2024.

Hou, C., Zhang, J., Wang, H., and Zhou, T. Subclassbalancing contrastive learning for long-tailed recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5395–5407, 2023.

Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., and Kalantidis, Y. Decoupling representation and classifier for long-tailed recognition. arXiv preprint arXiv:1910.09217, 2019.

Kim, J., Jeong, J., and Shin, J. M2m: Imbalanced classification via major-to-minor translation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 13896–13905, 2020.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

Lee, Y., Yao, H., and Finn, C. Diversify and disambiguate: Out-of-distribution robustness via disagreement. In The Eleventh International Conference on Learning Representations, 2023.

Li, J., Meng, Z., Shi, D., Song, R., Diao, X., Wang, J., and Xu, H. Fcc: Feature clusters compression for long-tailed visual recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24080–24089, 2023.

Li, M., Cheung, Y.-M., and Hu, Z. Key point sensitive loss for long-tailed visual recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4812– 4825, 2022a.

Li, M., Cheung, Y.-m., and Lu, Y. Long-tailed visual recognition via gaussian clouded logit adjustment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6929–6938, 2022b.

Li, M., Zhikai, H., Lu, Y., Lan, W., Cheung, Y.-m., and Huang, H. Feature fusion from head to tail for long-tailed visual recognition. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 13581–13589, 2024.

Li, S., Gong, K., Liu, C. H., Wang, Y., Qiao, F., and Cheng, X. Metasaug: Meta semantic augmentation for longtailed visual recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5212–5221, 2021.

Li, S., Xu, Q., Yang, Z., Wang, Z., Zhang, L., Cao, X., and Huang, Q. Focal-SAM: Focal sharpness-aware minimization for long-tailed classification. In Forty-second International Conference on Machine Learning, 2025.

Lin, F. and Yuan, X. Long-tailed recognition via information-preservable two-stage learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Liu, Z., Miao, Z., Zhan, X., Wang, J., Gong, B., and Yu, S. X. Large-scale long-tailed recognition in an open world. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2537–2546, 2019.

Luo, J., Hong, F., Hu, Q., Cao, X., Liu, F., and Yao, J. Long-tailed recognition with model rebalancing. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., and Kumar, S. Long-tail learning via logit adjustment. In International Conference on Learning Representations, 2021.

Park, S., Lim, J., Jeon, Y., and Choi, J. Y. Influence-balanced loss for imbalanced visual classification. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 735–744, 2021.

Park, S., Hong, Y., Heo, B., Yun, S., and Choi, J. Y. The majority can help the minority: Context-rich minority oversampling for long-tailed classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6887–6896, 2022.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Ren, S., He, K., Girshick, R., and Sun, J. Faster r-cnn: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28, 2015.

Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pp. 234–241. Springer, 2015.

Ru, L., Guo, X., Yu, L., Zhang, Y., Lao, J., Wang, J., Chen, J., Li, Y., and Yang, M. Parameter-efficient complementary expert learning for long-tailed visual recognition. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pp. 5393–5402, 2024.

Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.

Shi, J.-X., Wei, T., Zhou, Z., Shao, J.-J., Han, X.-Y., and Li, Y.-F. Long-tail learning with foundation model: Heavy fine-tuning hurts. In Forty-first International Conference on Machine Learning, 2024.

Sun, S., Lu, H., Li, J., Xie, Y., Li, T., Yang, X., Zhang, L., and Yan, J. Rethinking classifier re-training in longtailed recognition: Label over-smooth can balance. In The Thirteenth International Conference on Learning Representations, 2025.

Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Perona, P., and Belongie, S. The inaturalist species classification and detection dataset. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 8769–8778, 2018.

Wan, W., Wang, X., Xie, M.-K., Li, S.-Y., Huang, S.-J., and Chen, S. Unlocking the power of open set: A new perspective for open-set noisy label learning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 38, pp. 15438–15446, 2024.

Wang, J., Lukasiewicz, T., Hu, X., Cai, J., and Xu, Z. Rsg: A simple but effective module for learning imbalanced datasets. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 3784–3793, 2021.

Wang, T., Li, Y., Kang, B., Li, J., Liew, J., Tang, S., Hoi, S., and Feng, J. The devil is in classification: A simple framework for long-tail instance segmentation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16, pp. 728–744. Springer, 2020.

Wang, X., Geng, C., Wan, W., Li, S.-Y., and Chen, S. Forgetting, ignorance or myopia: Revisiting key challenges in online continual learning. Advances in Neural Information Processing Systems, 37:58341–58375, 2024a.

Wang, Y., Yu, Z., Wang, J., Heng, Q., Chen, H., Ye, W., Xie, R., Xie, X., and Zhang, S. Exploring vision-language models for imbalanced learning. International Journal of Computer Vision, 132(1):224–237, 2024b.

Xiao, K., Engstrom, L., Ilyas, A., and Madry, A. Noise or signal: The role of image backgrounds in object recognition. arXiv preprint arXiv:2006.09994, 2020.

Xinrui, W., Wan, W., Geng, C., Li, S.-Y., and Chen, S. Beyond myopia: Learning from positive and unlabeled data through holistic predictive trends. Advances in Neural Information Processing Systems, 36:67589–67602, 2023.

Xuan, S. and Zhang, S. Decoupled contrastive learning for long-tailed recognition. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pp. 6396– 6403, 2024.

Ye, W., Zheng, G., Cao, X., Ma, Y., and Zhang, A. Spurious correlations in machine learning: A survey. arXiv e-prints, pp. arXiv–2402, 2024.

Yin, X., Yu, X., Sohn, K., Liu, X., and Chandraker, M. Feature transfer learning for face recognition with underrepresented data. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 5704–5713, 2019.

Zang, Y., Huang, C., and Loy, C. C. Fasa: Feature augmentation and sampling adaptation for long-tailed instance segmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3457–3466, 2021.

Zhang, C., Almpanidis, G., Fan, G., Deng, B., Zhang, Y., Liu, J., Kamel, A., Soda, P., and Gama, J. A systematic review on long-tailed learning. IEEE Transactions on Neural Networks and Learning Systems, 2025.

Zhang, S., Li, Z., Yan, S., He, X., and Sun, J. Distribution alignment: A unified framework for long-tail visual recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2361–2370, 2021a.

Zhang, Y., Hooi, B., Hong, L., and Feng, J. Test-agnostic long-tailed recognition by test-time aggregating diverse experts with self-supervision. arXiv e-prints, pp. arXiv– 2107, 2021b.

Zhang, Y., Kang, B., Hooi, B., Yan, S., and Feng, J. Deep long-tailed learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 45(9):10795– 10816, 2023.

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A. Learning deep features for discriminative localization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2921–2929, 2016.

Zhu, J., Wang, Z., Chen, J., Chen, Y.-P. P., and Jiang, Y.- G. Balanced contrastive learning for long-tailed visual recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6908– 6917, 2022.

## A. Related Works

## A.1. Long-tailed Recognition and Head-to-Tail Transfer

Long-tailed recognition aims to improve the performance of underrepresented tail classes while maintaining accuracy on head classes. Early solutions mainly resorted to resampling (Chawla et al., 2002; Wang et al., 2020; Zang et al., 2021) or re-weighting (Luo et al., 2025; Park et al., 2021) to rebalance the class distribution during training, e.g., by increasing the effective contribution of tail instances or reducing the dominance of head classes in gradient updates. While such re-balancing is simple and often effective, it may introduce optimization instability (e.g., repeated exposure to a small set o tail samples) and can potentially degrade head performance when the re-balancing is aggressive.

Data augmentation provides another route to alleviate tail scarcity by expanding the diversity of rare categories. A line of work transfers semantic knowledge from head classes to enrich tail representations (Kim et al., 2020; Wang et al., 2021), whereas other methods synthesize new examples or features based on prior information and model feedback (Zang et al., 2021; Li et al., 2021). These approaches generally aim to compensate for the missing intra-class variation of tail classes, either by generating tail-like samples/features or by expanding the tail distribution in a controlled manner.

To improve prediction calibration under imbalance, logit adjustment methods shift the decision boundary according to class frequency, encouraging more balanced margins and reducing the bias toward head classes (Li et al., 2022a; Menon et al., 2021; Zhang et al., 2021a). Since many re-balancing strategies can trade head performance for tail gains, recent advances have further explored decoupled training pipelines that separate representation learning from classifier re-balancing (Kang et al., 2019; Lin & Yuan, 2025; Sun et al., 2025). In parallel, contrastive learning variants have been adopted to learn more robust and separable representations under severe imbalance by reshaping instance- and class-level similarity structures (Cui et al., 2021; Zhu et al., 2022; Hou et al., 2023; Xuan & Zhang, 2024).

Head-to-tail transfer for alleviating tail scarcity. A promising direction to address data scarcity in long-tailed learning is to transfer diverse information from head classes to augment tail representations. M2m (Kim et al., 2020) pioneers a generative major-to-minor translation framework based on adversarial learning. It learns a generator that maps abundant majority samples into the feature space of minority classes, effectively upsampling tail data with high-fidelity synthetic counterparts and enabling training on a more balanced set without collecting new real-world samples.

Moving beyond raw image generation, RSG (Wang et al., 2021) augments tail classes at the feature level via rare style generation. It designs a learned displacement field to extract intra-class variations from head classes and applies them to tai prototypes, thereby synthesizing diverse tail features by perturbing tail centers with rich head-induced variations.

Focusing on feature statistics, FASA (Zang et al., 2021) combines feature augmentation with adaptive resampling. It models each class as a Gaussian cloud and leverages statistics transferred from head data to estimate the missing variance of tail classes. Augmented features are then sampled dynamically (often guided by training status or model performance) to maintain balanced optimization throughout training.

LTA (Yin et al., 2019) further emphasizes second-order statistics, particularly covariance transfer from head to tail. By assuming that semantically related classes share similar directions of intra-class variation, LTA inflates point-like tai representations into fuller distributions, allowing the classifier to anticipate unseen variations and smoothing decision boundaries for rare categories.

In the realm of data mixing, CMO (Park et al., 2022) proposes a CutMix-based context-rich minority oversampling strategy. It samples diverse background contexts from head images and pastes them onto tail objects to form composite training examples, transferring environmental diversity to tail classes and reducing overfitting to limited tail backgrounds.

Most recently, H2T (Li et al., 2024) develops a head-to-tail feature fusion mechanism to improve the transferability of semantics. It identifies transferable semantic statistics (e.g., prototypes and variance) from head classes and fuses them with tail features to supplement missing diversity, effectively enriching tail representations without relying on heavy generative models.

## A.2. Shortcut Learning and Spurious Correlations

Definition and mechanisms. Deep neural networks, despite strong in-distribution performance, often exhibit shortcut learning (Geirhos et al., 2020), i.e., relying on superficial yet predictive cues that fit the training distribution but fail under distribution shift. This behavior is closely related to the simplicity bias of networks trained with SGD: models tend to adopt low-complexity features that minimize the training loss (e.g., color, texture, background) rather than more semantic but harder-to-learn evidence. As a result, standard empirical risk minimization can produce models that are correct for non-robust reasons, leading to poor OOD generalization.

Manifestations in vision. In visual recognition, shortcut learning commonly manifests as context and texture biases. Background/context bias arises when models spuriously correlate the environment with the label; for example, models trained on “cows in pastures” may fail on “cows on beaches” (Beery et al., 2018), indicating an over-reliance on background signals instead of object evidence. Related analyses further quantify this dependency by measuring performance drops when foreground and background statistics are altered or separated (Xiao et al., 2020). Texture bias reflects another prevalent shortcut: convolutional networks tend to prefer local texture cues over global shape information (Geirhos et al., 2018), making predictions brittle under style transfer and domain shift.

Mitigation strategies. A broad set of methods has been proposed to mitigate spurious correlations and improve robustness. Invariant learning aims to extract features that remain stable across environments; Invariant Risk Minimization (IRM) enforces that the same classifier is optimal across multiple training environments, thereby discouraging reliance on environment-specific spurious cues (Arjovsky et al., 2019). Robust optimization approaches, such as Distributionally Robust Optimization (DRO), improve worst-group performance by explicitly optimizing for challenging sub-populations where shortcuts break (Sagawa et al., 2019). Causal-inspired interventions attempt to break the dependence between shortcuts and labels by manipulating the data generation process; counterfactual augmentation removes or alters spurious attributes (Agarwal et al., 2020), while causal attention mechanisms encourage models to attend to causal evidence instead of contextual confounders.

Beyond conventional shortcut learning, recent studies have also discussed myopic learning behaviors in other settings. For example, online continual learning may suffer from model myopia, where models rely on locally convenient features rather than learning globally robust representations (Wang et al., 2024a). Similar concerns have been raised in positiveunlabeled learning, where holistic predictive trends are used to mitigate decisions based on limited or isolated supervision signals (Xinrui et al., 2023). Although these settings differ from long-tailed recognition, they share a common intuition: limited or locally biased supervision can make easy but suboptimal cues overly attractive. Our work studies this issue from an evidence-level perspective, showing that tail-class scarcity can induce shortcut-dominated CAMs and that SRCD suppresses such spatial shortcut reliance.

## B. More Discussion

## B.1. Additional Discussion on CAM-Based Shortcut Diagnosis

Here, we provide additional rationale for using CAM similarity to a balanced reference model as a proxy for shortcut reliance under long-tailed training, and clarifies the underlying limitations.

Why a balanced model is a reasonable semantic reference. Our diagnosis is based on the hypothesis that balanced training tends to produce explanations that are closer to class-relevant semantics. The key reason is supervision diversity: when each class has abundant and varied samples, spurious correlations (e.g., recurring contexts or textures) are less consistent across examples, and thus less reliable for minimizing the empirical risk. As a result, the model is pressured to rely on evidence that persists across diverse conditions, which is more likely to correspond to stable class-related cues. In contrast, long-tailed supervision reduces diversity for tail classes: a small number of examples often share similar acquisition conditions or contexts, making spurious cues highly predictive and therefore attractive shortcuts. Under this viewpoint, the balanced model provides a practical reference for “what evidence is likely to be used when shortcuts are less rewarded”.

Why deviations in CAMs can indicate shortcut reliance. CAMs localize spatial evidence that contributes to a class score, providing an interpretable proxy for the model’s reliance on different regions. For the same input $( x , y )$ , if the long-tailed model learns a shortcut for class y, its evidence distribution may shift toward shortcut-correlated regions that are less stable or less semantically tied to the object. This shift results in a measurable discrepancy between the long-tailed CAM and the balanced reference CAM. Therefore, low cosine similarity between $\bar { M } _ { y } ^ { l t } ( x )$ and $\bar { M } _ { y } ^ { b a l } ( x )$ is consistent with shortcut-dominated attention, whereas high similarity suggests the long-tailed model relies on evidence closer to the balanced reference.

Role of normalization and cosine similarity. We normalize CAMs to reduce scale effects (e.g., different magnitudes induced by logit scale or activation strength), and treat them as spatial evidence distributions. Cosine similarity then measures alignment in evidence patterns rather than absolute magnitude, making the metric more robust to global rescaling. Moreover, cosine similarity is sensitive to systematic spatial shifts: when evidence relocates from one region type to another (e.g., from object-supporting regions to shortcut-correlated regions), the vector directions diverge and similarity decreases.

What this metric captures (and what it does not). Our metric is designed to capture relative changes in spatial evidence allocation induced by long-tailed training. It is not intended to claim that the balanced CAM is a perfect “ground-truth” explanation, nor that every deviation is necessarily a shortcut. For example, architecture capacity, optimization noise, or class confusion can also affect CAM patterns. Nevertheless, in aggregate across samples and frequency groups, we expect long-tailed training to induce systematic deviations concentrated on tail classes if shortcut reliance is the dominant mechanism, which is precisely what we observe.

Limitations. Our CAM-similarity diagnosis is a proxy for shortcut reliance and has several limitations. First, the balanced reference CAM is not a ground-truth explanation: even under balanced training, CAMs may still be imperfect and can highlight discriminative parts rather than the full object, so deviations should be interpreted as relative shifts in evidence allocation rather than definitive proof of shortcuts. Second, low similarity does not uniquely imply shortcut learning—it may also arise from class confusion, optimization noise, or different local minima, especially for visually similar classes where the same image can admit multiple plausible evidence patterns. Third, CAM quality depends on the chosen architecture and CAM variant; for some backbones, CAMs can be spatially coarse or saturate due to feature-map resolution, which may blur fine-grained differences and weaken the interpretability of similarity. Finally, the metric assumes comparable training recipes and model capacity across the balanced and long-tailed runs; large differences in augmentation, regularization, or convergence can introduce confounding factors, making CAM discrepancies harder to attribute solely to long-tailed supervision. Despite these caveats, we find the trend is highly consistent across frequency groups, supporting its usefulness as an empirical indicator of shortcut-dominated attention in the long-tailed regime.

Why this proxy is still useful for SRCD. Importantly, our method does not require access to balanced training at deployment; the balanced reference is used only to diagnose the shortcut phenomenon. SRCD itself operates without any balanced reference model. The diagnostic result mainly motivates the use of head-class proxies: since head classes in long-tailed training enjoy richer supervision (closer to the balanced regime), their CAMs can serve as a practical source of shortcut-resistant evidence for tail classes.

## B.2. Additional Discussion on the Rationale of $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$

Here, we provide a detailed rationale for why the spatial coherence energy $\mathcal { E } _ { h }$ and the evidence concentration score ${ \mathcal { E } } _ { n }$ are suitable for detecting and mitigating shortcut learning in long-tailed recognition tasks, and discuss the assumptions and limitations of these measures.

Why spatial coherence is important. In long-tailed datasets, tail-class examples often share a similar contextua background, such as the same background textures, environments, or co-occurring objects, which can lead to shortcut learning. In contrast, head-class examples tend to be more diverse, with different object views, lighting conditions, and backgrounds, forcing the model to focus on the object itself. The spatial coherence energy $\mathcal { E } _ { h }$ penalizes fragmented and disconnected activations, which are commonly seen in shortcut-driven explanations. Such fragmented patterns are often a result of relying on incidental background cues rather than stable, object-centric features.

By using Laplacian-based energy, $\mathcal { E } _ { h }$ effectively captures the smoothness of the CAM, enforcing that evidence should be spatially coherent. This is particularly critical in long-tailed scenarios, where the risk of shortcut learning is higher, and background regions may dominate the tail-class CAMs. As a result, $\mathcal { E } _ { h }$ encourages the model to focus on contiguous, stable regions that are more likely to correspond to meaningful object representations.

The role of evidence concentration. Tail classes in long-tailed datasets often have fewer examples, and these examples may be dominated by background context rather than the object itself. In these cases, the model may learn to focus on background cues (e.g., ocean or sky) instead of the actual object. The evidence concentration score, ${ \mathcal { E } } _ { n }$ , addresses this issue by quantifying how concentrated the evidence is in a small region. A higher concentration score means the model is focusing its attention on a smaller, more discriminative region, which is typically associated with better object-centric explanations.

For example, in datasets like ImageNet, most objects are typically centered and well-defined, and background cues are less important. In such cases, the concentration score helps the model focus on the object itself, thus avoiding distractions from the background. By contrast, in long-tailed datasets where tail-class examples often share similar background features, the concentration score helps to suppress the model’s reliance on large, diffuse background regions.

Why these metrics work together. While $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ capture different failure modes of shortcut learning, they complement each other in controlling both fragmentation and diffuse attention. A CAM may be compact but fragmented (e.g., a few isolated peaks in different locations), which is penalized by $\mathcal { E } _ { h }$ but not necessarily by ${ \mathcal { E } } _ { n }$ . On the other hand, a CAM may be smooth but overly spread across a large area (e.g., background activation covering a large portion of the image), which is penalized by ${ \mathcal { E } } _ { n }$ but may not yield a large coherence energy.

By combining both metrics, SRCD ensures that the final teacher CAM is both coherent and concentrated, making it more likely to reflect object-centric features rather than background or texture cues that may be prevalent in tail classes. These complementary aspects make SRCD a more reliable method for regularizing tail-class predictions.

Dataset characteristics and generalization. The effectiveness of $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ is particularly evident in datasets where background cues are prevalent, especially in long-tailed distributions. In datasets like ImageNet (Liu et al., 2019), where most classes are centered on an object and background context is less important, the combination of these metrics guides the model to focus on the object itself rather than on irrelevant background areas. In long-tailed data, however, the object-centric representations in tail classes are weaker, and background cues dominate. By using these metrics to prioritize head-class-like behavior, SRCD helps suppress shortcut reliance in tail classes.

Furthermore, we observe that these metrics generalize well to various long-tailed benchmarks. The assumption is that head classes, with more diverse supervision, have learned object-focused features. Therefore, SRCD leverages the better generalization of head-class explanations and transfers them to the tail classes, where there is a higher risk of shortcut learning.

Limitations of the approach. Despite their usefulness, $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ are not perfect proxies for object-centric semantics. There are a few cases where the metrics may penalize valid evidence:

• Large objects or irregular shapes: For some objects, the relevant evidence may naturally be spread over large or irregular regions (e.g., a large animal or vehicle), which might be penalized by ${ \mathcal { E } } _ { n }$ for being diffuse, even though it is semantically important.

• Disconnected object parts: Some objects may have parts that are naturally disconnected (e.g., thin structures), which could result in a fragmented CAM, and thus a higher $\mathcal { E } _ { h }$ . In such cases, the metric may incorrectly penalize valid object evidence.

• Fine-grained categories: In fine-grained classification tasks, where classes are very similar (e.g., different species of birds), $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ may not always perfectly distinguish between legitimate object features and spurious correlations, requiring further fine-tuning of the criteria.

Practical considerations. We compute both $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ on normalized CAMs $\bar { M }$ to avoid scale effects that could be introduced by different activation magnitudes across classes. This normalization ensures that the metrics focus on the spatia distribution of attention rather than absolute scale. The Laplacian matrix L is computed once per input resolution and reused throughout training, which makes the computation efficient. Additionally, the 4-neighbor grid used in the Laplacian has been found to work well in most datasets, but alternative choices, such as 8-neighbor grids, could be explored for more complex tasks.

Conclusion. In summary, $\mathcal { E } _ { h }$ and ${ \mathcal { E } } _ { n }$ are designed to capture different aspects of shortcut reliance in long-tailed learning. By enforcing spatial coherence and evidence concentration, SRCD effectively suppresses shortcut-driven behavior in tail classes. These metrics are not perfect but provide a reliable mechanism to regularize tail-class predictions, especially in datasets where background cues dominate. Despite certain limitations, SRCD remains a robust tool for improving the generalization of models trained on long-tailed datasets.

## B.3. Discussion on Head-Proxy CAM Supervision

SRCD uses head-proxy CAMs as a shortcut-resistant supervision signal for tail classes. We emphasize that these proxies are not intended to provide perfect semantic matching or complete part-level supervision for the target tail object. In particular, we do not assume that different head proxies must cover mutually complementary object parts, nor do we require each proxy CAM to strictly align with the full target object. Even for head classes with abundant training samples, obtaining complete and semantically correct part-level CAMs is difficult. Therefore, enforcing such a requirement on tail examples, where supervision is scarce and often biased, would be overly strong.

The goal of SRCD is different from explicit part decomposition. Rather than constructing a pixel-level or part-leve ground-truth explanation, SRCD aims to build a teacher CAM that is more stable and less affected by shortcut cues than the original tail-class CAM. This distinction is important: a useful teacher does not need to perfectly cover every object part. It can still provide effective supervision as long as it is less shortcut-prone than the tail CAM learned from scarce and biased data. In this sense, SRCD uses head-proxy CAMs as a regularization signal to encourage tail explanations to move away from background or context shortcuts.

Semantic mismatch between head proxies and tail classes may also occur. SRCD does not assume that every selected proxy is reliable. Instead, it aggregates multiple candidate CAMs using energy-based weights based on spatial coherence and evidence concentration. This design suppresses fragmented, noisy, or overly diffuse proxy CAMs and favors candidates that provide more stable spatial evidence. As a result, the teacher is not determined by a single potentially mismatched proxy, but by a weighted combination of candidates that are structurally more consistent with shortcut-resistant attention.

To further examine the effect of semantic mismatch, we conduct an additional analysis on iNaturalist using its taxonomy labels, such as Actinopterygii, Amphibia, Animalia, Arachnida, Aves, Insecta, Mammalia, and Plantae. Specifically, we identify a mismatched subset of tail samples whose selected head proxies are taxonomically unrelated to the target tail class. With BCL as the baseline, 32.13% of tail samples fall into this mismatched subset. As shown in Table 4, SRCD stil improves the accuracy on these samples from 67.25 to 69.88. This result suggests that SRCD does not strictly rely on exact semantic matching between the head proxy and the tail target. Even when the selected proxy is taxonomically unrelated, its CAM can still provide useful shortcut-resistant spatial evidence, as long as it is less shortcut-prone than the original tail-class CAM.

Table 4. Accuracy (%) on the semantically mismatched subset of iNaturalist tail samples, where the selected head proxies are taxonomically unrelated to the target tail classes.
<table><tr><td>Method</td><td>Acc.</td></tr><tr><td>BCL</td><td>67.25</td></tr><tr><td>BCL + SRCD</td><td>69.88</td></tr></table>

One possible alternative is to explicitly enforce diversity across proxies so that different candidates attend to different object parts. However, this would require additional assumptions and computational costs, such as a larger top-K candidate set, fine-grained pixel-level comparisons among CAMs, and extra diversity or structural constraints. More importantly, without dense annotations, it is difficult to verify whether the resulting regions are truly correct, complete, and complementary. This difficulty is even more pronounced for irregular objects, such as insects, where object parts can be small, deformable, or visually ambiguous.

Therefore, SRCD should be understood as an evidence-level shortcut suppression method rather than a part-level decomposition method. Its objective is to provide tail classes with a less shortcut-prone CAM teacher, not a perfect semantic or part-level explanation. The empirical improvements across datasets and baselines suggest that this form of shortcut-resistant CAM supervision is sufficient to improve tail-class generalization.

This design is also related to recent observations in open-set noisy-label learning that out-of-set or non-target samples are not necessarily purely harmful; when properly exploited, they can provide useful auxiliary information for improving class separation (Wan et al., 2024). Similarly, SRCD does not require head proxies to be semantically identical to the tail label, but uses them as candidate spatial explanations whose reliability is further filtered by energy-based weighting.

## B.4. Long-Tailed-Specific Design of SRCD

SRCD is designed specifically for the long-tailed setting. The key challenge is not only class imbalance itself, but also the asymmetric shortcut vulnerability induced by such imbalance. Tail classes, under scarce and biased supervision, are more likely to develop shortcut-dominated attention, whereas head classes tend to learn more stable object-centric evidence due to richer and more diverse training samples.

Motivated by this long-tailed-specific observation, SRCD uses head-side explanations to regularize tail-class attention during training. Therefore, the role of CAM distillation in SRCD is not generic. Instead, the distillation is constructed around the head-tail structure: the teacher is built from head-class proxies, the correction is applied specifically to tail classes, and the regularization strength is scaled by class rarity. In this sense, SRCD is a long-tailed-motivated mechanism for correcting shortcut-dominated tail attention, rather than a generic CAM distillation strategy directly applied to long-tailed data.

## C. Proofs and Extended Discussion

## C.1. Notation and Preliminaries

Normalized CAM. For an input image x and class $y ,$ let $\bar { M } _ { y } ( x ) \in \mathbb { R } ^ { H \times W }$ denote the normalized CAM:

$$
\bar { M } _ { y } ( x ) _ { u } \ge 0 , \qquad \sum _ { u \in \Omega } \bar { M } _ { y } ( x ) _ { u } = 1 ,\tag{9}
$$

where Ω indexes the spatial grid (with $| \Omega | = N = H W )$ . Hence, $\bar { M } _ { y } ( x )$ can be viewed as a discrete probability distribution over spatial locations.

Shortcut/background region. We assume the spatial domain Ω can be partitioned into an object-related region $\Omega _ { \mathrm { o b j } }$ and a shortcut/background region $\Omega _ { \mathrm { s c } }$

$$
\Omega = \Omega _ { \mathrm { o b j } } \cup \Omega _ { \mathrm { s c } } , \qquad \Omega _ { \mathrm { o b j } } \cap \Omega _ { \mathrm { s c } } = \emptyset .\tag{10}
$$

This partition is conceptual: the analysis does not require explicit masks during training; it only serves to formalize the shortcut mechanism.

Shortcut-region attention mass. We quantify shortcut reliance of any normalized CAM $\bar { M }$ as:

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ) = \sum _ { u \in \Omega _ { \mathrm { s c } } } \bar { M } _ { u } \in [ 0 , 1 ] .\tag{11}
$$

Smaller $A _ { \mathrm { s c } }$ indicates more object-centric explanations; larger $A _ { \mathrm { s c } }$ indicates stronger shortcut/background attention.

Teacher construction. Let $\mathcal { H } ( x )$ denote the set of head-proxy candidates used to form the teacher CAM. SRCD defines:

$$
\bar { M } _ { T } ( x ) = \sum _ { c \in \mathcal { H } ( x ) } w _ { c } ( x ) \bar { M } _ { c } ( x ) , \qquad w _ { c } ( x ) = \frac { \exp \bigl ( \mathcal { U } ( \bar { M } _ { c } ( x ) ) / \tau \bigr ) } { \sum _ { c ^ { \prime } \in \mathcal { H } ( x ) } \exp \bigl ( \mathcal { U } ( \bar { M } _ { c ^ { \prime } } ( x ) ) / \tau \bigr ) } ,\tag{12}
$$

where $\tau > 0$ is a temperature and the utility is

$$
\mathcal { U } ( \bar { M } ) = - \mathcal { E } _ { h } ( \bar { M } ) + \mathcal { E } _ { n } ( \bar { M } ) .\tag{13}
$$

Intuitively, $\mathcal { E } _ { h }$ penalizes spatial fragmentation (favoring coherent evidence), while ${ \mathcal { E } } _ { n }$ rewards evidence concentration (favoring object-centric evidence).

CAM distillation loss. Given a tail sample $( x , y )$ , SRCD distills the teacher explanation into the tail-class CAM:

$$
\mathcal { L } _ { \mathrm { S R C D } } ( x , y ) = \gamma _ { y } \big | \big | \bar { M } _ { y } ( x ) - \bar { M } _ { T } ( x ) \big | \big | _ { 2 } ^ { 2 } ,\tag{14}
$$

where $\gamma _ { y }$ increases with tail scarcity to emphasize shortcut mitigation for low-data classes.

## C.2. Proof of Theorem 4.1 (Entropy-Regularized Utility Maximization)

We restate the theorem in a self-contained form.

Theorem C.1 (Entropy-regularized utility maximization). Let $\{ \mathcal { U } _ { c } \} _ { c = 1 } ^ { K }$ be utilities and $\Delta \ = \ \{ w \ \in \ \mathbb { R } ^ { K } \ : \ w _ { c } \ \geq$ $0 , \textstyle \sum _ { c = 1 } ^ { K } w _ { c } = 1 \big \}$ . For $\tau > 0 ,$ , the optimization

$$
\operatorname* { m a x } _ { w \in \Delta } \ J ( w ) = \sum _ { c = 1 } ^ { K } w _ { c } \mathcal { U } _ { c } - \tau \sum _ { c = 1 } ^ { K } w _ { c } \log w _ { c }\tag{15}
$$

has a unique maximizer

$$
w _ { c } ^ { \star } = \frac { \exp ( \mathcal { U } _ { c } / \tau ) } { \sum _ { j = 1 } ^ { K } \exp ( \mathcal { U } _ { j } / \tau ) } .\tag{16}
$$

Step 1: Strict concavity and uniqueness. The first term $\sum _ { c } w _ { c } \mathcal { U } _ { c }$ is linear in w. The entropy term − P w<sub>c</sub> log $w _ { c }$ is strictly concave over the interior of $\Delta$ (and concave over the whole simplex). Thus, $J ( w )$ is strictly concave on $\Delta .$ , implying a unique global maximizer.

Step 2: Lagrangian and KKT conditions. Introduce a Lagrange multiplier $\lambda$ for the simplex constraint:

$$
\mathcal { L } ( w , \lambda ) = \sum _ { c = 1 } ^ { K } w _ { c } \mathcal { U } _ { c } - \tau \sum _ { c = 1 } ^ { K } w _ { c } \log w _ { c } + \lambda \Big ( \sum _ { c = 1 } ^ { K } w _ { c } - 1 \Big ) .\tag{17}
$$

For an interior optimum $( w _ { c } > 0 )$ , the stationarity condition is:

$$
\frac { \partial \mathcal { L } } { \partial w _ { c } } = \mathcal { U } _ { c } - \tau ( \log w _ { c } + 1 ) + \lambda = 0 .\tag{18}
$$

Rearranging yields:

$$
\log w _ { c } = \frac { \mathcal { U } _ { c } + \lambda - \tau } { \tau } \quad \Longrightarrow \quad w _ { c } = \exp \Bigl ( \frac { \mathcal { U } _ { c } } { \tau } \Bigr ) \cdot \exp \Bigl ( \frac { \lambda - \tau } { \tau } \Bigr ) .\tag{19}
$$

Let $\begin{array} { r } { Z = \sum _ { j = 1 } ^ { K } \exp ( \mathcal { U } _ { j } / \tau ) } \end{array}$ . Enforcing $\textstyle \sum _ { c } w _ { c } = 1$ gives:

$$
\exp \Bigl ( \frac { \lambda - \tau } { \tau } \Bigr ) = \frac { 1 } { Z } \Longrightarrow w _ { c } ^ { \star } = \frac { \exp ( \mathcal { U } _ { c } / \tau ) } { Z } .\tag{20}
$$

This proves (16).

Interpretation (why this is principled). Eq. (15) formalizes a standard “exploitation–exploration” trade-off:

$\sum _ { c } w _ { c } \mathcal { U } _ { c }$ pushes the teacher to select candidates with high utility (low fragmentation $\mathcal { E } _ { h }$ and high concentration $\mathcal { E } _ { n } )$

• The entropy penalty $- \sum c$ log $w _ { c }$ prevents brittle hard selection, improving robustness when multiple candidates have similar quality.

• The temperature τ controls smoothness: $\tau  0$ approaches hard max selection; larger τ approaches uniform averaging.

Useful equivalent form (free-energy). Define the log-sum-exp value:

$$
\Phi ( \mathcal { U } ) = \tau \log \sum _ { c = 1 } ^ { K } \exp ( \mathcal { U } _ { c } / \tau ) .\tag{21}
$$

One can show:

$$
\operatorname* { m a x } _ { w \in \Delta } \Big ( \sum _ { c } w _ { c } \mathcal { U } _ { c } - \tau \sum _ { c } w _ { c } \log w _ { c } \Big ) = \Phi ( \mathcal { U } ) ,\tag{22}
$$

and $\boldsymbol { w } ^ { \star } = \nabla _ { \boldsymbol { u } } \Phi ( \boldsymbol { u } )$ equals the softmax weights. This view is useful when discussing stability and smoothness properties of the teacher.

## C.3. Proof of Theorem 4.2 (Distillation Dynamics)

We restate Theorem 4.2 and provide a detailed derivation.

Theorem C.2 (Monotone shortcut-mass contraction). Consider the gradient flow minimizing $F ( \bar { M } ) = \| \bar { M } - \bar { M } _ { T }$ ∥<sup>2</sup><sub>2</sub>:

$$
\frac { d } { d t } \bar { M } ( t ) = - \nabla _ { \bar { M } } F ( \bar { M } ( t ) ) .\tag{23}
$$

Then the solution is $\bar { M } ( t ) = \bar { M } _ { T } + e ^ { - 2 t } ( \bar { M } ( 0 ) - \bar { M } _ { T } )$ , and

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( t ) ) = \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) + e ^ { - 2 t } \big ( \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( 0 ) ) - \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) \big ) . } \end{array}\tag{24}
$$

$I f \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( 0 ) ) > \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } )$ then $\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( t ) )$ decreases monotonically to $\bar { \mathcal { A } } _ { \mathrm { s c } } ( \bar { M } _ { T } )$

Step 1: Compute the gradient. Let $F ( \bar { M } ) = \| \bar { M } - \bar { M } _ { T } \| _ { 2 } ^ { 2 } = ( \bar { M } - \bar { M } _ { T } ) ^ { \top } ( \bar { M } - \bar { M } _ { T } )$ . Then:

$$
\nabla _ { \bar { \cal M } } F ( \bar { \cal M } ) = 2 ( \bar { \cal M } - \bar { \cal M } _ { T } ) .\tag{25}
$$

Step 2: Solve the ODE. Plug into (23):

$$
\frac { d } { d t } \bar { M } ( t ) = - 2 ( \bar { M } ( t ) - \bar { M } _ { T } ) .\tag{26}
$$

This is a linear ODE. Let $\Delta ( t ) = \bar { M } ( t ) - \bar { M } _ { T }$ . Then:

$$
\frac { d } { d t } \Delta ( t ) = - 2 \Delta ( t ) \quad \Longrightarrow \quad \Delta ( t ) = e ^ { - 2 t } \Delta ( 0 ) = e ^ { - 2 t } ( \bar { M } ( 0 ) - \bar { M } _ { T } ) ,\tag{27}
$$

hence

$$
\bar { M } ( t ) = \bar { M } _ { T } + e ^ { - 2 t } ( \bar { M } ( 0 ) - \bar { M } _ { T } ) .\tag{28}
$$

Step 3: Shortcut-mass transport follows from linearity. Define the indicator vector $\mathbf { 1 } _ { \Omega _ { \mathrm { s c } } }$ (with 1 on $\Omega _ { \mathrm { s c } }$ and 0 elsewhere). Then:

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ) = \langle \mathbf { 1 } _ { \Omega _ { \mathrm { s c } } } , \bar { M } \rangle .\tag{29}
$$

Taking inner products of (28) with $\mathbf { 1 } _ { \Omega _ { \mathrm { s c } } }$ yields:

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( t ) ) = \langle \mathbf { 1 } _ { \Omega _ { \mathrm { s c } } } , \bar { M } _ { T } \rangle + e ^ { - 2 t } \langle \mathbf { 1 } _ { \Omega _ { \mathrm { s c } } } , \bar { M } ( 0 ) - \bar { M } _ { T } \rangle\tag{30}
$$

$$
= \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) + e ^ { - 2 t } \big ( \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( 0 ) ) - \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) \big ) ,\tag{31}
$$

which proves (24). Monotonicity follows immediately if $\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ( 0 ) ) > \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } )$

Discrete-time analogue (what SGD is doing). If we take a simple gradient step with step size $\eta \in ( 0 , 1 )$

$$
\bar { M } ^ { ( t + 1 ) } = \bar { M } ^ { ( t ) } - \eta \nabla _ { \bar { M } } \| \bar { M } ^ { ( t ) } - \bar { M } _ { T } \| _ { 2 } ^ { 2 } = ( 1 - 2 \eta ) \bar { M } ^ { ( t ) } + 2 \eta \bar { M } _ { T } .\tag{32}
$$

Thus, each step is an explicit interpolation toward $\bar { M _ { T } }$ . Applying $A _ { \mathrm { s c } } ( \cdot )$ gives:

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ^ { ( t + 1 ) } ) = ( 1 - 2 \eta ) \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ^ { ( t ) } ) + 2 \eta \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) , } \end{array}\tag{33}
$$

so shortcut mass decreases whenever the teacher has smaller shortcut mass. This provides an optimization-level explanation beyond a static inequality bound.

## C.4. Why the Teacher is Shortcut-Resistant: Utility Alignment and Consequences

Theorems above show that distillation will transport shortcut mass toward the teacher. We now justify why the proposed teacher is expected to be shortcut-resistant.

## C.4.1. A WEAK ALIGNMENT ASSUMPTION

We adopt a mild and interpretable condition: the shortcut mass is negatively aligned with utility.

Assumption C.3 (Utility–shortcut alignment). There exist constants $\alpha \in \mathbb { R }$ and $\beta > 0$ such that for all candidates $c \in { \mathcal { H } } ( x )$

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { c } ( x ) ) \leq \alpha - \beta \mathcal { U } ( \bar { M } _ { c } ( x ) ) .\tag{34}
$$

Why this assumption is reasonable. The utility $\mathcal { U } = - \mathcal { E } _ { h } + \mathcal { E } _ { n }$ favors coherent and concentrated evidence:

• Background shortcuts typically appear as scattered activations across many regions (high fragmentation), yielding larger $\mathcal { E } _ { h }$ and lower utility.

• Shortcut-driven CAMs are often diffuse (low concentration), yielding smaller ${ \mathcal { E } } _ { n }$ and lower utility.

Thus, higher U tends to correlate with lower background mass $\boldsymbol { A } _ { \mathrm { s c } }$

## C.4.2. TEACHER SHORTCUT MASS BOUND UNDER ALIGNMENT

Since $\begin{array} { r } { \bar { M } _ { T } = \sum _ { c } w _ { c } \bar { M } _ { c } } \end{array}$ and $A _ { \mathrm { s c } }$ is linear:

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) = \sum _ { c } w _ { c } \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { c } ) .\tag{35}
$$

Applying Assumption C.3:

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) \leq \sum _ { c } w _ { c } ( \alpha - \beta \mathcal { U } _ { c } ) = \alpha - \beta \sum _ { c } w _ { c } \mathcal { U } _ { c } ,\tag{36}
$$

where $\mathcal { U } _ { c } = \mathcal { U } ( \bar { M } _ { c } )$

Key implication. Eq. (36) shows the teacher shortcut mass is upper-bounded by negative expected utility. Therefore, any mechanism that increases $\sum _ { c } w _ { c } \mathcal { U } _ { c }$ will reduce (or tightly control) the teacher shortcut mass.

## C.4.3. SOFTMAX WEIGHTS MAXIMIZE EXPECTED UTILITY (WITH ROBUSTNESS)

By Theorem 4.1, the softmax weights are the unique maximizer of

$$
\sum _ { c } w _ { c } \mathcal { U } _ { c } - \tau \sum _ { c } w _ { c } \log w _ { c } .
$$

This yields two desirable properties:

• Low-shortcut preference: candidates with larger $\mathcal { U } _ { c }$ receive exponentially larger weights $w _ { c }$

• Robustness: the entropy term avoids collapsing onto a single candidate when utilities are close or noisy.

Combining with (36), the proposed teacher aggregation is systematically biased toward lower shortcut mass.

## C.4.4. FROM TEACHER QUALITY TO TAIL SHORTCUT SUPPRESSION

If the aggregated teacher satisfies

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) < \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { y } ) ,\tag{37}
$$

then Theorem 4.2 implies that distillation decreases the tail shortcut mass monotonically (and exponentially under the gradient-flow view). This establishes the end-to-end shortcut mitigation mechanism:

utility-biased teacher ⇒ teacher is more object-centric $\Rightarrow \quad L _ { 2 }$ distillation transports mass away from shortcuts.

## C.5. Technical Details of $\mathcal { E } _ { h }$ : Graph Laplacian Coherence Energy

We provide a detailed explanation of $\mathcal { E } _ { h } ( \bar { M } ) = \bar { M } ^ { \top } L \bar { M }$ and why it captures spatial coherence.

## C.5.1. WHAT IS L AND WHY IT IS USED

Let $G = ( V , { \mathcal { E } } )$ be the 4-neighbor grid graph over pixels, where $V = \Omega$ and $( i , j ) \in \mathcal { E }$ if pixels i and j are adjacent. Let $G \in \mathbb { R } ^ { N \times N }$ be the adjacency matrix $( G _ { i j } = 1 \mathrm { i f } ( i , j ) \in \mathcal { E }$ , else 0), and $D$ be the degree matrix $\begin{array} { r } { ( D _ { i i } = \sum _ { j } G _ { i j } ) } \end{array}$ . The (combinatorial) graph Laplacian is

$$
L = D - G .\tag{38}
$$

Dirichlet energy interpretation. For any vector $z \in \mathbb { R } ^ { N }$

$$
z ^ { \top } L z = \frac { 1 } { 2 } \sum _ { i , j } G _ { i j } ( z _ { i } - z _ { j } ) ^ { 2 } = \sum _ { ( i , j ) \in \mathcal E } ( z _ { i } - z _ { j } ) ^ { 2 } .\tag{39}
$$

Thus, $z ^ { \top } L z$ penalizes local discontinuities across neighboring pixels.

Why it captures “shortcut fragmentation”. CAMs driven by shortcut/background cues often show scattered, highfrequency activations across diverse regions. Such patterns have large neighbor differences $( \bar { M } _ { i } - \bar { M } _ { j } ) ^ { 2 }$ , increasing $\bar { M } ^ { \top } L \bar { M }$ In contrast, object-centric evidence tends to form connected blobs with smooth interiors and fewer abrupt changes, yielding smaller coherence energy.

## C.5.2. SPECTRAL VIEW: HIGH-FREQUENCY SUPPRESSION

Since L is symmetric positive semidefinite, it admits ${ \cal L } = U \Lambda U ^ { \top }$ , where $\boldsymbol { \Lambda } = \operatorname { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { N } )$ with $0 = \lambda _ { 1 } \leq \cdots \leq \lambda _ { N }$ Then:

$$
\bar { M } ^ { \top } L \bar { M } = \sum _ { k = 1 } ^ { N } \lambda _ { k } \langle u _ { k } , \bar { M } \rangle ^ { 2 } .\tag{40}
$$

Large eigenvalues correspond to high-frequency graph components. Therefore, minimizing $\mathcal { E } _ { h }$ suppresses high-frequency oscillations in CAMs—precisely the type of spatial noise often induced by shortcuts.

## C.5.3. DISCUSSION: COHERENCE ENERGY VS. “PURE CONNECTIVITY”

A purely connectivity-based metric $( \mathrm { e . g . }$ , number of connected components after thresholding) is non-differentiable and sensitive to thresholds. In contrast, $\bar { M } ^ { \top } L \bar { M }$ is:

• Differentiable and stable: smoothly changes under small perturbations.

• Scale-aware: penalizes fine-grained fragmentation without requiring binarization.

• Computationally cheap: computable via local neighbor differences on the grid.

These properties make $\mathcal { E } _ { h }$ a practical surrogate for shortcut-related fragmentation in CAM space.

## C.6. Technical Details of ${ \mathcal { E } } _ { n }$ : Evidence Concentration Score

We discuss why ${ \mathcal { E } } _ { n }$ is a suitable measure for discouraging diffuse, background-spread evidence.

## C.6.1. A NORMALIZED CONCENTRATION SCORE

Let $N = | \Omega |$ and $\bar { M } \geq 0$ with $\| \bar { M } \| _ { 1 } = 1$ (by normalization). Consider the Hoyer-style normalized score:

$$
\mathcal { E } _ { n } ( \bar { M } ) = \frac { \sqrt { N } - \| \bar { M } \| _ { 1 } / \| \bar { M } \| _ { 2 } } { \sqrt { N } - 1 } = \frac { \sqrt { N } - 1 / \| \bar { M } \| _ { 2 } } { \sqrt { N } - 1 } ,\tag{41}
$$

which lies in [0, 1].

Extreme cases (sanity check).

• Uniform/diffuse CAM: $\begin{array} { r } { \bar { M } _ { u } = \frac { 1 } { N } } \end{array}$ for all u. Then $\| \bar { M } \| _ { 2 } = 1 / \sqrt { N }$ and $\mathcal { E } _ { n } ( \bar { M } ) = 0$

• One-hot CAM: all mass on one pixel. Then $\| \bar { M } \| _ { 2 } = 1$ and $\mathcal { E } _ { n } ( \bar { M } ) = 1$

Thus, larger ${ \mathcal { E } } _ { n }$ corresponds to more concentrated evidence.

Why concentration discourages shortcuts. Shortcut reliance often manifests as evidence spread across large background regions (e.g., texture, context), increasing diffusion and reducing peakiness. Rewarding concentration therefore reduces the probability mass assigned to broad, non-object background areas.

## C.6.2. SCALE MATCHING WITH $\mathcal { E } _ { h }$

In practice, $\mathcal { E } _ { h }$ can have a larger numerical range than $\mathcal { E } _ { n } \in [ 0 , 1 ]$ . Using $\mathcal { U } = - \mathcal { E } _ { h } + \mathcal { E } _ { n }$ implicitly assumes the two terms are roughly comparable. A common and robust alternative is:

$$
\mathcal { U } ( \bar { M } ) = - \lambda _ { h } \mathcal { E } _ { h } ( \bar { M } ) + \lambda _ { n } \mathcal { E } _ { n } ( \bar { M } ) ,\tag{42}
$$

with $\lambda _ { h } , \lambda _ { n } > 0$ . This preserves the analysis and typically improves stability of teacher selection.

## C.7. Additional Bounds: Shortcut Mass Gap Controlled by Distillation Distance

Although our main text focuses on the transport dynamics, it is also useful to provide a static inequality bound.

Lemma C.4 (Shortcut-mass gap bound). For any normalized CAMs $\bar { M } , \bar { M } _ { T }$

$$
\left| \mathcal { A } _ { \mathrm { s c } } ( \bar { M } ) - \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) \right| \leq \sqrt { | \Omega _ { \mathrm { s c } } | } \ \| \bar { M } - \bar { M } _ { T } \| _ { 2 } .\tag{43}
$$

Proof. Let $\Delta = \bar { M } - \bar { M _ { T } }$ . Then

$$
\mathcal { A } _ { \mathrm { s c } } ( \bar { M } ) - \mathcal { A } _ { \mathrm { s c } } ( \bar { M } _ { T } ) = \sum _ { u \in \Omega _ { \mathrm { s c } } } \Delta _ { u } .
$$

By Cauchy–Schwarz:

$$
\Big | \sum _ { u \in \Omega _ { \mathrm { s c } } } \Delta _ { u } \Big | \leq \sqrt { | \Omega _ { \mathrm { s c } } | } \cdot \sqrt { \sum _ { u \in \Omega _ { \mathrm { s c } } } \Delta _ { u } ^ { 2 } } \leq \sqrt { | \Omega _ { \mathrm { s c } } | } \ \| \Delta \| _ { 2 } .
$$

Discussion. Lemma C.4 directly connects the optimization target $\lVert \bar { M } - \bar { M } _ { T } \rVert _ { 2 } ^ { 2 }$ to shortcut mass discrepancy. It complements the transport view by providing a simple worst-case guarantee even without analyzing learning dynamics.

## C.8. Why SRCD Should be Enabled After Head Classes Become Reliable

SRCD distillation is only beneficial if the teacher is indeed shortcut-resistant. This requirement motivates a late-start (warm-up) schedule.

Early-stage risk. In early epochs, head-class CAMs may not be fully semantic and can still attend to spurious background cues. In this regime, utilities $\mathcal { U } ( \bar { M } _ { c } )$ are noisy, and teacher aggregation may fail to reduce shortcut mass, potentially amplifying shortcuts through distillation.

Late-stage advantage. After the model learns strong head semantics, head-proxy CAMs become more coherent and concentrated: $\mathcal { E } _ { h }$ decreases and ${ \mathcal { E } } _ { n }$ increases, hence U becomes reliably higher for object-centric evidence. Teacher construction then becomes shortcut-resistant, and Theorem 4.2 guarantees monotone shortcut-mass reduction.

Compatibility with fine-tuning. This naturally aligns with long-tailed training protocols that apply re-balancing in a second stage. SRCD can be introduced during the fine-tuning phase as a plug-and-play explanation regularizer on top of any re-balancing objective, without modifying architectures.

## C.9. Extended Discussion and Practical Takeaways

(1) What exactly is “shortcut” in our theory? Shortcut learning is modeled as excessive CAM mass on $\Omega _ { \mathrm { s c } }$ . This focuses the theory on where the model draws evidence, rather than only how confident it is. It matches the empirical failure mode where tail classes rely on correlated backgrounds.

(2) Why CAM distillation (not feature distillation) is the right lever? Feature alignment may still permit shortcut reliance if background cues dominate internal representations. By contrast, CAM alignment targets the spatial evidence directly, enforcing object-centric explanations. The transport identity (24) makes this effect explicit

(3) Why combine coherence and concentration? Either term alone is insufficient:

• Low $\mathcal { E } _ { h }$ alone can be achieved by smooth but diffuse CAMs covering background.

• High ${ \mathcal { E } } _ { n }$ alone can be achieved by peaky but fragmented evidence.

Using $\mathcal { U } = - \mathcal { E } _ { h } + \mathcal { E } _ { n }$ favors CAMs that are simultaneously blob-like (coherent) and object-localized (concentrated), which better matches semantic evidence and discourages shortcut backgrounds.

(4) Temperature τ and stability. Smaller τ performs near-hard selection of the top-utility candidate, which may be brittle when utilities are noisy. Larger τ averages more candidates, improving stability but weakening selectivity. Entropyregularized utility maximization provides a clean interpretation for choosing τ.

(5) Why SRCD is especially effective for tail classes. Tail supervision is data-scarce, so learned explanations are more likely to drift toward shortcuts. SRCD introduces an additional signal—a shortcut-resistant teacher—and Theorem 4.2 shows this signal induces monotone shortcut-mass reduction. The tail-weight $\gamma _ { y }$ further amplifies this correction.

(6) What does the theory not assume? We do not assume availability of segmentation masks or explicit knowledge o $\Omega _ { \mathrm { s c } }$ during training. The region split is only for analysis, while the learning signal arises from coherence and concentration energies, which are computed directly from CAMs.

(7) Summary. The appendix provides a complete shortcut-centric explanation of SRCD: teacher construction is a principled entropy-regularized utility maximizer (Section C.2), and $L _ { 2 }$ CAM distillation induces an explicit contraction/transport dynamics that monotonically reduces shortcut attention mass toward a more object-centric teacher (Section C.3–C.4).

## D. Experimental Details

## D.1. Detailed Dataset Description

We leverage five benchmark datasets — CIFAR100-LT (Cao et al., 2019), ImageNet-LT (Liu et al., 2019), iNaturalist 2018 (Van Horn et al., 2018), and Places-LT (Liu et al., 2019) — to mimic real - world long - tailed class distributions. These datasets are characterized by severe class imbalance, as noted in previous studies (Zhang et al., 2023). Table 5 summarizes the data statistics of these datasets. Notably, CIFAR100-LT features three variants, each with a distinct imbalance ratio. The imbalance ratio is defined as $\frac { \operatorname* { m a x } { n _ { j } } } { \operatorname* { m i n } { n _ { j } } }$ , where $n _ { j }$ represents the number of data points in class j.

Table 5. Statistics of datasets.
<table><tr><td>Dataset</td><td># classes</td><td># training data</td><td># test data</td><td>imbalance ratio</td></tr><tr><td>CIFAR10-LT (Cao et al., 2019)</td><td>10</td><td>50,000</td><td>10,000</td><td>{100, 50,10}</td></tr><tr><td>CIFAR100-LT (Cao et al., 2019)</td><td>100</td><td>50,000</td><td>10,000</td><td>{100, 50,10}</td></tr><tr><td>ImageNet-LT (Liu et al., 2019)</td><td>1,000</td><td>115,846</td><td>50,000</td><td>256</td></tr><tr><td>iNaturalist 2018 (Van Horn et al., 2018)</td><td>8,142</td><td>437,513</td><td>24,426</td><td>500</td></tr><tr><td>Places-LT (Liu et al., 2019)</td><td>365</td><td>62,500</td><td>36,500</td><td>996</td></tr></table>

CIFAR10/100-LT. CIFAR10/100-LT are long-tailed variants of CIFAR-10 and CIFAR-100 (Krizhevsky et al., 2009). CIFAR-10 contains 50,000 training images and 10,000 validation images of size $( 3 2 \times 3 2 )$ from 10 classes, while CIFAR 100 contains 50,000 training images and 10,000 validation images of size $( 3 2 \times 3 2 )$ from 100 classes. Following (Cao et al., 2019), we adopt the same long-tailed data construction for fair comparison. The imbalance ratio (IR) is defined as $\rho = N _ { \mathrm { m a x } } / N _ { \mathrm { m i n } }$ , where $N _ { \mathrm { m a x } }$ and $N _ { \mathrm { m i n } }$ denote the numbers of samples in the most frequent and least frequent classes, respectively. Unless otherwise specified, we evaluate on $\rho \in \{ 1 0 0 , 5 0 , 1 0 \}$ for both CIFAR10-LT and CIFAR100-LT.

ImageNet-LT. ImageNet-LT is a long-tailed version of ImageNet-2012 (Deng et al., 2009) introduced by (Liu et al., 2019). Following (Liu et al., 2019), we use the same construction by sampling training images according to a Pareto distribution with power $\gamma = 6 .$ ImageNet-LT contains 115.8K images from 1,000 categories, with an imbalance factor $\rho = N _ { \mathrm { m a x } } / N _ { \mathrm { m i n } } = 1 2 8 0 / 5$

iNaturalist 2018. iNaturalist 2018 (Van Horn et al., 2018) is a large-scale real-world long-tailed recognition benchmark with an extremely imbalanced class distribution. It contains 437.5K training images and 24.4K validation images from 8,142 categories. Its fine-grained categories further increase the recognition difficulty.

## D.2. Competitors

For a fair and controlled comparison, we apply our method on top of each baseline without changing its original architecture. All experiments follow the same training pipeline as the corresponding baseline, and we keep all optimization and training hyperparameters (e.g., learning rate schedule, optimizer, epochs, batch size) identical across methods. All experiments are conducted on four NVIDIA A100 GPUs. Moreover, SRCD is enabled only after a warm-up stage and is applied in the late training phase, when head-class representations become sufficiently reliable.

Unless otherwise specified, we fix the SRCD loss weight to $\beta = 0 . 6$ and set $K = 3$ in all experiments.

In our experiments, we benchmark our proposed method against a comprehensive suite of state-of-the-art algorithms. We categorize these baselines into two prevailing training paradigms: training from scratch on standard long-tailed benchmarks and fine-tuning foundation models.

## D.2.1. TRAINING FROM SCRATCH

For the standard training-from-scratch setting, we compare with the following representative methods, ordered by their release timeline and technical approach:

• BCL (Balanced Contrastive Learning) (Zhu et al., 2022): BCL identifies that standard contrastive learning is inherently biased towards head classes due to the dominance of majority samples in the mini-batch. To mitigate this, it introduces a class-averaging mechanism that equalizes the gradient contributions of different classes. Furthermore, it employs a balanced temperature parameter to dynamically adjust the difficulty of negative samples, ensuring that the learned feature space remains discriminative for both head and tail categories.

• GCL (Gaussian Clouded Logit Adjustment) (Li et al., 2022b): GCL approaches the long-tailed problem from the perspective of uncertainty estimation. It observes that tail classes suffer from high epistemic uncertainty, which manifests as vague decision boundaries. By modeling the logits of each class as a Gaussian distribution (a ”cloud”) rather than a deterministic point, GCL enforces a calibration constraint that pushes the decision boundaries of tail classes outward, effectively compensating for the lack of samples with a probabilistic margin.

• FCC (Feature Clusters Compression) (Li et al., 2023): FCC focuses on optimizing the geometric structure of the feature space. It argues that head classes, with their abundant samples, tend to form expansive clusters with large intra-class variance, which ”intrude” into the limited feature space available for tail classes. FCC proposes a cluster compression mechanism to explicitly constrain the variance of head classes, thereby reserving more embedding space for tail categories and reducing feature confusion.

• GLMC (Global and Local Mixture Consistency) (Du et al., 2023): GLMC combines the benefits of cumulative learning with consistency regularization. It utilizes a global branch to maintain a historical context of the class distribution (addressing the forgetting issue in streaming data) and a local branch to enforce consistency via Mixup augmentation. By aligning the predictions of these two branches, GLMC ensures that the model learns robust representations that are invariant to the sampling bias inherent in long-tailed data.

• H2T (Head-to-Tail Feature Fusion) (Li et al., 2024): H2T exploits the high-quality semantic representations learned from head classes to aid the recognition of tail classes. It designs a feature fusion module that transfers diverse semantic statistics (e.g., variance and prototypes) from head categories to augment the sparse representations of tail categories. This ”Head-to-Tail” transfer effectively hallucinates diverse samples for the tail, enriching their feature distribution without requiring additional real data.

• BNS (Information-Preservable Two-Stage Learning) (Lin & Yuan, 2025): BNS revisits the classical two-stage decoupling paradigm (learning representation then classifier). It critiques standard decoupling methods for freezing the backbone in the second stage, which causes a significant loss of spatial and semantic information. BNS introduces a ”Batch-No-Shift” module and an information-preservable objective, allowing the model to adjust the classifier while maintaining the integrity of the learned feature representation, thus bridging the gap between representation learning and classifier alignment.

• Focal-SAM (Focal Sharpness-Aware Minimization) (Li et al., 2025): Focal-SAM brings advanced optimization techniques into long-tailed recognition by adapting Sharpness-Aware Minimization (SAM). Recognizing that tail classes often converge to sharp, unstable minima in the loss landscape, Focal-SAM introduces a focal re-weighting strategy to the SAM objective. This forces the optimizer to prioritize finding flatter, more generalizable minima specifically for rare categories, thereby improving their robustness against test-time distribution shifts.

## D.2.2. FINE-TUNING FOUNDATION MODELS

For the setting of adapting Vision-Language Models (e.g., CLIP (Radford et al., 2021)) to long-tailed downstream tasks, we follow the standard protocol and compare with the following recent approaches:

• Decoder (VL-LTR) (Wang et al., 2024b): This method explores the integration of vision-language models into imbalanced learning. It appends a lightweight decoder structure to the fixed CLIP image encoder, which is trained to align visual features with their corresponding text embeddings. By explicitly leveraging the rich, pre-trained semantic priors from the text modality, the Decoder acts as a bridge that guides the visual classifier to recognize tail concepts that are visually under-represented but semantically rich.

• PECEL (Parameter-Efficient Complementary Expert Learning) (Ru et al., 2024): PECEL addresses the efficiency and adaptability trade-off by combining Mixture-of-Experts (MoE) with parameter-efficient fine-tuning. It creates a set of complementary experts (e.g., via LoRA or Adapters) that specialize in different data regimes (head vs. tail). A gating mechanism dynamically routes samples to the most appropriate expert, allowing the model to handle extreme class imbalances without the massive computational cost of full model fine-tuning.

• LIFT (Long-Tail Learning with Foundation Model) (Shi et al., 2024): LIFT provides a critical analysis of fine-tuning dynamics, revealing that ”heavy” fine-tuning of foundation models on long-tailed data often leads to feature distortion and catastrophic forgetting of general knowledge. Instead, LIFT proposes a lightweight adaptation strategy that involves freezing most backbone parameters and tuning only a minimal set of bias terms or prompts. This preserves the robust open-world generalization capability of the pre-trained model while adapting it to the target distribution.

Applicability. It is important to emphasize that our proposed method operates orthogonally to these baselines. Whether it is an optimization-based approach like Focal-SAM, a representation learning method like BCL, or a foundation model adapter like LIFT, our method can be seamlessly integrated as a plug-and-play component to further boost performance across all these diverse settings.

## D.3. Additional Results

Tables 6 and 7 report split-wise results on CIFAR100-LT under $\scriptstyle \mathrm { I R } = 1 0 0$ and IR=50, respectively. SRCD consistently improves the Tail split across all evaluated baselines in both imbalance settings, showing that it mainly benefits data-scarce classes. Under the more challenging $\scriptstyle { \mathrm { I R } = 1 0 0 }$ setting, SRCD improves Head, Medium, Tail, and overall accuracy for all baselines, with the most pronounced gains observed on the Tail split. Under IR=50, SRCD still improves Tail accuracy for every baseline; although CE+H2T shows a slight drop in overall accuracy, its Tail performance is still improved. Overall, these results further demonstrate that SRCD effectively enhances tail-class generalization, especially under severe long-tailed imbalance.

Table 6. Split-wise results on CIFAR100-LT under IR=100, reporting Top-1 accuracy (%) on Head/Med/Tail splits and All. +SRCD denotes integrating SRCD into each baseline; superscripts show absolute changes (↑ /↓ for gain/drop).
<table><tr><td>Method</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>GLMC +SRCD</td><td>63.39  $6 3 . 8 8 \mathrm { \uparrow 0 . 4 9 }$ </td><td>60.95  $6 2 . 3 7 \substack { \uparrow 1 . 4 2 }$ </td><td>34.63  $3 6 . 9 0 \substack { \uparrow 2 . 2 7 }$ </td><td>53.91  $5 5 . 2 6 \substack  \mathrm { \ t \} 1 . 3 5 }$ </td></tr><tr><td>CE+H2T</td><td>50.85</td><td>48.91</td><td>24.53</td><td>42.27</td></tr><tr><td>+SRCD BNS</td><td> $5 1 . 9 3 \substack { \uparrow 1 . 0 8 }$  62.40</td><td> $5 1 . 1 7 \substack { \uparrow 2 . 2 6 }$  59.70</td><td> $2 8 . 1 6 \substack { \uparrow 3 . 6 3 }$  31.90</td><td> $4 4 . 5 3 \substack { \uparrow 2 . 2 6 }$  52.40</td></tr><tr><td>+SRCD</td><td> $6 2 . 9 3 \mathrm { \uparrow 0 } . 5 3$ </td><td> $6 0 . 5 6 \substack { \mathrm { \uparrow 0 . 8 6 } }$ </td><td> $3 5 . 5 2 \substack { \uparrow 3 . 6 2 }$ </td><td> $5 3 . 8 8 \mathrm { \uparrow } 1 . 4 8 $ </td></tr><tr><td>Focal-SAM</td><td>63.90</td><td>53.00</td><td>32.50</td><td>50.70</td></tr><tr><td>+SRCD</td><td> $6 4 . 6 6 \mathrm { \cdot } 0 . 7 6$ </td><td> $5 4 . 2 7 \substack { \uparrow 1 . 2 7 }$ </td><td> $3 6 . 4 0 \substack { \uparrow 3 . 9 0 }$ </td><td> $5 2 . 5 4 \mathrm { \uparrow } 1 . 8 4$ </td></tr></table>

Table 7. Split-wise results on CIFAR100-LT under IR=50. reporting Top-1 accuracy (%) on Head/Med/Tail splits and All. +SRCD denotes integrating SRCD into each baseline; superscripts show absolute changes (↑ /↓ for gain/drop).
<table><tr><td>Method</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>GLMC +SRCD</td><td>64.73  $6 5 . 5 1 \substack { \uparrow 0 . 7 8 }$ </td><td>62.10  $6 3 . 5 3 \substack { \uparrow 1 . 4 3 }$ </td><td>38.19  $4 1 . 8 6 \substack { \uparrow 3 . 6 7 }$ </td><td>58.87  $6 0 . 4 3 \substack { \uparrow 1 . 5 6 }$ </td></tr><tr><td>CE+H2T</td><td>51.42</td><td>50.96</td><td>29.93</td><td>47.58</td></tr><tr><td>+SRCD BNS</td><td> $5 1 . 4 7 \substack { \uparrow 0 . 0 5 }$ </td><td> $4 9 . 8 3 \downarrow 1 . 1 3$ </td><td> $3 0 . 2 1 \mathrm { \Omega } \mathrm { \Omega } \mathrm { \Omega } 0 . 2 8$ </td><td> $4 6 . 9 7 \scriptstyle \downarrow 0 . 6 1$ </td></tr><tr><td>+SRCD</td><td>61.30  $6 1 . 2 0 \downarrow 0 . 1 0$ </td><td>58.70  $5 8 . 9 4 \mathrm { \uparrow } 0 . 2 4 $ </td><td>37.20  $3 8 . 1 7 \substack { \uparrow 0 . 9 7 }$ </td><td>55.90  $5 6 . 1 2 \mathrm { \Omega } \mathrm { \Omega } \mathrm { \Omega } 0 . 2 2$ </td></tr><tr><td>Focal-SAM</td><td></td><td></td><td></td><td></td></tr><tr><td>+SRCD</td><td>60.20  $6 0 . 2 4 \uparrow 0 . 0 4$ </td><td>58.00  $5 8 . 3 2 \mathrm { \uparrow } 0 . 3 2$ </td><td>36.20  $3 7 . 1 5 \substack { \mathrm { ( 0 . 9 5 } }$ </td><td>54.50  $5 5 . 2 9 \mathrm { \Omega } _ { \mathrm { T 0 . 7 9 } }$ </td></tr></table>

Table 8 further verifies the effectiveness of SRCD on Places-LT, a challenging long-tailed scene recognition benchmark. In the training-from-scratch setting, SRCD improves the overall accuracy of all evaluated baselines, with particularly clear gains on the Tail split. For example, SRCD improves the Tail accuracy of BCL, GLMC, GCL, and H2T by 1.32, 1.84, 0.58, and 2.96 percentage points, respectively, indicating that SRCD effectively improves tail-class recognition where shortcut reliance is more severe. In the foundation-model fine-tuning setting, SRCD also brings consistent gains, improving the overall accuracy of PECEL, Decoder, and LIFT by 0.4, 0.7, and 1.0 points, respectively. Their Tail accuracy is also improved by 0.9, 1.5, and 2.3 points. Overall, the Places-LT results demonstrate that SRCD generalizes beyond object-centric long-tailed recognition and remains effective for long-tailed scene recognition.

Table 8. Results on Places-LT, reporting Top-1 accuracy (%) on Head/Med/Tail splits and All. The upper block is training from scratch, and the lower block is fine-tuning foundation models. +SRCD denotes integrating SRCD into each baseline; superscripts show absolute changes (↑ /↓ for gain/drop).
<table><tr><td>Method</td><td colspan="3">Places-LT</td></tr><tr><td></td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td colspan="5">Training from scratch</td></tr><tr><td>BCL</td><td>41.98</td><td>42.36</td><td>34.91</td><td>40.77</td></tr><tr><td>+SRCD</td><td>41.85↓0.13</td><td>42.79↑0.43</td><td>36.23↑1.32</td><td>41.18↑0.41</td></tr><tr><td>GLMC</td><td>42.09</td><td>42.88</td><td>35.27</td><td>41.12</td></tr><tr><td>+SRCD</td><td>42.28↑0.19</td><td>43.05↑0.17</td><td>37.11↑1.84</td><td>41.62↑0.50</td></tr><tr><td>GCL +SRCD</td><td>38.64</td><td>42.59</td><td>38.44</td><td>40.30</td></tr><tr><td>H2T</td><td>38.93↑0.29</td><td>42.74↑0.15</td><td>39.02↑0.58</td><td>40.65↑0.35</td></tr><tr><td>+SRCD</td><td>41.96 42.20↑0.24</td><td>42.87  $4 2 . 6 4 \downarrow 0 . 2 3$ </td><td>35.33  $3 8 . 2 9 \mathrm { \uparrow } 2 . 9 6 $ </td><td>40.95 41.64↑0.69</td></tr><tr><td colspan="5">Fine-tuning foundation model</td></tr><tr><td>Decoder</td><td>46.3</td><td>46.8</td><td>45.8</td><td>46.4</td></tr><tr><td>+SRCD</td><td> $4 6 . 5 \substack { \uparrow 0 . 2 }$ </td><td> $4 7 . 6 \mathrm { \uparrow } 0 . 8 $ </td><td> $4 7 . 3 \substack { \uparrow 1 . 5 }$ </td><td>47.1↑0.7</td></tr><tr><td>PECEL</td><td>50.1</td><td>51.2</td><td>48.3</td><td>50.2</td></tr><tr><td>+SRCD</td><td>50.3↑0.2</td><td>51.4↑0.2</td><td>49.2↑0.9</td><td>50.6↑0.4</td></tr><tr><td>LIFT +SRCD</td><td>51.3 51.6↑0.3</td><td>52.2 53.0↑0.8</td><td>50.5 52.8↑2.3</td><td>51.5 52.5↑1.0</td></tr></table>

## D.4. CAM Extraction Details for Fine-tuning Foundation Models

In the fine-tuning setting, we adapt a pre-trained vision(-language) foundation model (e.g., CLIP) to long-tailed recognition while explicitly supervising its spatial evidence via CAM. We use CAM (weight-based evidence) rather than Grad-CAM: the resulting map is computed from class weights and token/feature activations, without backpropagating gradients through the backbone for visualization. This choice makes the CAM branch lightweight, stable, and easy to integrate with parameter efficient tuning (Adapters/LoRA), and it also matches the objective of constraining where the model looks under long-tailed bias.

## D.4.1. NOTATION AND GENERAL FORM

Let the image encoder output a spatial feature tensor. For CNN-style encoders, this is a feature map $A \in \mathbb { R } ^ { C \times H \times W }$ . For ViTstyle encoders, we treat patch tokens as a spatial grid $Z \in \mathbb { R } ^ { N \times \mathbf { \bar { D } } }$ with $N = H _ { p } W _ { p }$ , and reshape it into $F \in \mathbb { R } ^ { D \times H _ { p } \times W _ { p } }$ Let the classifier (or linear probe) for class c have weight vector $w _ { c }$ . The CAM for class c is computed in a unified form:

$$
M ^ { c } ( \cdot ) = \phi \Bigl ( \bigl < w _ { c } , \ s p a t i a l \mathrm { f e a t u r e s } ( \cdot ) \bigr > \Bigr ) ,\tag{44}
$$

where $\langle \cdot , \cdot \rangle$ denotes channel-wise inner product at each spatial location, and $\phi ( \cdot )$ is a post-processing operator (e.g., ReLU + normalization + upsampling).

We always generate the CAM conditioned on the target class (ground-truth label) t, i.e., we use $M ^ { t }$ as the supervision target in the SRCD branch.

## D.4.2. VIT-BASED ENCODERS: TOKEN-CAM (CAM ON PATCH TOKENS)

Many CLIP variants use a ViT image encoder. In this case, the encoder produces a sequence of tokens after the last transformer block:

$$
\tilde { Z } = [ z _ { \mathrm { c l s } } , z _ { 1 } , \dots , z _ { N } ] \in \mathbb { R } ^ { ( N + 1 ) \times D } ,
$$

where z is the class token and $z _ { \mathrm { c l s } }$ $\{ z _ { i } \} _ { i = } ^ { N }$ are patch tokens. We build a spatial tensor from patch tokens by discarding $z _ { \mathrm { c l s } }$ and reshaping:

$$
F \in \mathbb { R } ^ { D \times H _ { p } \times W _ { p } } , \qquad F ( : , u , v ) = z _ { i ( u , v ) } .
$$

Then the Token-CAM for class c is defined as:

$$
M ^ { c } ( u , v ) = \phi \bigl ( w _ { c } ^ { \top } F ( : , u , v ) \bigr ) .\tag{45}
$$

Practical choices that matter:

• Which token features to use. We use the output tokens after the final transformer block and its normalization (e.g., the last LayerNorm), because this is the feature space directly consumed by the classifier/projection head.

• Spatial resolution. For ViT-B/16, $\begin{array} { r } { ( H _ { p } , W _ { p } ) = ( \frac { H } { 1 6 } , \frac { W } { 1 6 } ) } \end{array}$ . The raw CAM is at patch resolution and is upsampled to the input image resolution using bilinear interpolation.

• Classifier weights. If we fine-tune a linear classifier on top of the frozen/pre-trained encoder, $w _ { c }$ comes from this classifier head. If we use parameter-efficient modules (Adapters/LoRA) inside the encoder, $w _ { c }$ is still the classifier head, while $F$ reflects the adapted token features.

Post-processing $\phi$ for ViT Token-CAM. We use a simple and stable $\phi$ to make CAM values comparable across samples:

$$
\hat { M } ^ { c } = \operatorname* { m a x } ( M ^ { c } , 0 ) \quad \mathrm { ( R e L U ) }\tag{46}
$$

$$
\tilde { M } ^ { c } = \frac { \hat { M } ^ { c } - \operatorname* { m i n } ( \hat { M } ^ { c } ) } { \operatorname* { m a x } ( \hat { M } ^ { c } ) - \operatorname* { m i n } ( \hat { M } ^ { c } ) + \epsilon } \quad \mathrm { ( m i n \mathrm { - } m a x \ n o r m ) }\tag{47}
$$

$$
M _ { \uparrow } ^ { c } = \mathrm { U p s a m p l e } ( \tilde { M } ^ { c } , ~ H \times W ) ~ ( \mathrm { b i l i n e a r } ) .\tag{48}
$$

This yields a bounded map $M _ { \uparrow } ^ { c } \in [ 0 , 1 ] ^ { H \times W }$ used by the SRCD loss.

## D.4.3. CNN-BASED ENCODERS: STANDARD CAM ON FEATURE MAPS

If the image encoder is CNN-style (e.g., ResNet-based CLIP), we extract the last convolutional feature map $A \in \mathbb { R } ^ { C \times H _ { f } \times W _ { f } }$ before global pooling. With a linear classifier weight $w _ { c } \in \mathbb { R } ^ { C }$ , the standard CAM is:

$$
{ \cal M } ^ { c } ( x , y ) = \phi \Big ( \sum _ { k = 1 } ^ { C } w _ { c , k } A _ { k } ( x , y ) \Big ) .\tag{49}
$$

The same ϕ (ReLU + min-max + bilinear upsampling) is applied to map it back to the image resolution.

## D.4.4. CLIP-SPECIFIC CONSIDERATIONS (VISION-LANGUAGE ENCODERS)

When using CLIP, there are two common classifier constructions:

• Linear classifier on visual features. We take the visual embedding $v \left( \mathrm { e . g } \right.$ ., pooled $z _ { \mathrm { c l s } }$ or pooled CNN feature) and train a classification head. CAM uses the head weights $w _ { c }$ and spatial features from the visual backbone (Token-CAM for ViT or CAM for CNN).

• Text-prototype classifier (zero-shot style). If logits are computed by similarity between image embedding v and text embeddings $\{ t _ { c } \}$ , then a direct CAM needs an explicit linear weight $w _ { c } .$ In our fine-tuning setting, we use a learnable classification head for long-tailed adaptation, so $w _ { c }$ is well-defined and CAM is straightforward.

This design ensures CAM is aligned with the downstream classifier used in evaluation.

## D.4.5. STOP-GRADIENT ON THE CAM BRANCH

In fine-tuning, especially with parameter-efficient tuning, we avoid destabilizing the main optimization by blocking gradients through the CAM construction path. Concretely, we compute CAM from detached spatial features:

$$
{ \boldsymbol { F } } \gets \operatorname* { d e t a c h } ( { \boldsymbol { F } } ) \quad \mathrm { o r } \quad { \boldsymbol { A } } \gets \operatorname* { d e t a c h } ( { \boldsymbol { A } } ) ,
$$

and use it only for the SRCD loss computation. This prevents the CAM supervision from directly reshaping backbone features in a way that harms classification calibration, while still allowing SRCD to act as a consistent auxiliary signal through the designated parameters (e.g., SRCD-specific projection ϕ, adapters, or a small CAM head if used).

## D.4.6. MULTI-CAM COLLECTION FOR SRCD TEACHER CONSTRUCTION (FINE-TUNING)

In the fine-tuning regime, we often exploit the stronger head-class priors of foundation models to build a robust teacher CAM. For each training sample, we form K head-proxy CAMs (with K fixed across all experiments; default $K = 3 )$ ):

1. obtain candidate proxy classes $\{ c _ { 1 } , \ldots , c _ { K } \}$ (e.g., top-K predictions among head classes or a predefined head-class pool);

2. compute their Token-CAM/CAM maps $\{ M ^ { c _ { i } } \} _ { i = } ^ { K }$ using Eq. (45) or Eq. (49);

3. aggregate them into a teacher CAM via an energy-based weighting scheme (details in Sec. 3.4).

This produces a teacher map that is typically more object-centric and less sensitive to long-tailed noise, which is crucial when tail classes have weak or diffuse evidence.

## D.4.7. IMPLEMENTATION DETAILS (HOOKS, SHAPES, AND EFFICIENCY)

We summarize practical details that are important for a correct and efficient implementation:

• Feature extraction point. For ViT, hook the output after the last transformer block (and its final LayerNorm if present) to get patch tokens. For CNN, hook the last convolutional stage before global pooling.

• Shape handling. For ViT, exclude the class token and reshape N tokens into $( H _ { p } , W _ { p } )$ . For non-square inputs, keep the true grid size derived from the patch embedding layer.

• Normalization stability. Use ϵ in min-max normalization to avoid division-by-zero on flat maps (common for early training).

• Upsampling. Bilinear upsampling is sufficient; we keep it deterministic for reproducibility.

• Memory/computation. CAM is computed from a single linear projection $w _ { c } ^ { \top } F ( : , u , v )$ at each location; the overhead is negligible compared to a forward pass. Blocking gradients further reduces autograd memory.

## D.4.8. DEFAULT HYPERPARAMETERS IN FINE-TUNING

Unless otherwise stated, we fix the SRCD loss weight to $\beta = 0 . 6$ and set the number of head-proxy CAMs to $K = 3$ for all fine-tuning experiments, keeping the rest of the training protocol (optimizer, learning rate schedule, epochs, batch size, and baseline-specific settings) unchanged for fair comparison.

## D.5. Comparison with a Center-Mask Prior

We further examine whether the gain of SRCD can be explained by a trivial geometric prior. Specifically, we construct a center-mask baseline on CIFAR100-LT under IR=100 using BNS as the baseline. This variant replaces the adaptive head-proxy teacher with a fixed mask centered in the image and uses it as the distillation target.

As shown in Table 9, the center-mask baseline performs worse than BNS, especially on the Tail split. This suggests that the gain of SRCD cannot be explained by simply encouraging attention toward the image center. A fixed center prior may suppress both object evidence and shortcut cues, and it can fail to focus on the actual object region when objects are

Table 9. Comparison with a center-mask baseline on CIFAR100-LT under IR=100. The center-mask baseline replaces the adaptive head-proxy teacher with a fixed centered mask.
<table><tr><td>Method</td><td>Head</td><td>Med</td><td>Tail</td><td>All</td></tr><tr><td>BNS</td><td>62.40</td><td>59.70</td><td>31.90</td><td>52.40</td></tr><tr><td> $\mathrm { B N S } + \mathrm { C e n t e r \ : M a s k }$ </td><td> $6 1 . 7 6 \downarrow 0 . 6 4$ </td><td> $5 8 . 2 7 \downarrow 1 . 4 3$ </td><td> $2 8 . 6 8 \downarrow 3 . 2 2$ </td><td> $5 0 . 6 1 \downarrow 1 . 7 9$ </td></tr><tr><td> $\mathrm { B N S } + \mathrm { S R C D }$ </td><td> $6 2 . 9 3 \mathrm { \uparrow } 0 . 5 3$ </td><td> $6 0 . 5 6 \substack { \mathrm { \uparrow 0 . 8 6 } }$ </td><td> $3 5 . 5 2 \uparrow 3 . 6 2$ </td><td> $5 3 . 8 8 \mathrm { \cdot } 1 . 4 8 $ </td></tr></table>

off-center or have diverse layouts. In contrast, SRCD constructs an adaptive teacher from head-proxy CAMs and uses energy-based weighting to favor spatially coherent and concentrated evidence, which is crucial for improving tail-class recognition.

## D.6. Computational Overhead

Overhead results. We quantify the training overhead by comparing each baseline before and after integrating SRCD under an identical setup (same architecture, optimizer, learning-rate schedule, batch size, and total epochs). Table 10 reports the wall-clock time per iteration, total training time, throughput (images/sec). Under the current evaluated setting, adding SRCD incurs only a small increase in iteration time, while the overall training time remains close to the original baseline.

Measurement protocol. We measure (i) wall-clock time per iteration, (ii) total training time, and (iii) throughput (images/sec). All measurements are conducted on the same hardware/software environment and are averaged over multiple runs after a warm-up period.

Table 10. Training overhead on ImageNet with ResNet-50 (method GLMC). Reported Total time is the amortized wall-clock time over the full training run.
<table><tr><td>Method</td><td>Time/iter (ms)</td><td>Total time (h)</td><td>Throughput (img/s)</td><td>∆ Time/iter</td></tr><tr><td>GLMC</td><td>142</td><td>19.7</td><td>1842</td><td></td></tr><tr><td> $\mathrm { G L M C } + \mathrm { S R C D }$ </td><td>153</td><td>22.6</td><td>1707</td><td>+7.7%</td></tr></table>

Why the overhead is small. The efficiency mainly comes from four design choices.

Late-stage activation: SRCD is enabled only in the late training stage (e.g., the last portion of epochs), so most iterations are identical to the baseline and the amortized overhead is reduced.

Reusing forward features: CAMs are computed from spatial representations already produced in the standard forward pass (e.g., the last CNN feature map or ViT patch tokens), requiring no extra backbone forward passes; CAM generation is a class-weighted linear projection plus a lightweight ϕ(·) (ReLU/normalization/upsampling).

No attribution backpropagation: SRCD uses weight-based CAM with stop-gradient on the CAM branch, avoiding Grad-CAM-style gradient computation w.r.t. intermediate activations and thus avoiding an extra backward graph.

Small constant factor (K): teacher construction uses a small fixed number of head-proxy CAMs (default K = 3); each proxy CAM only adds a few linear projections on the reused feature map.

## D.7. Visualization of CAM Evidence

We visualize the spatial evidence during training to better understand how SRCD reshapes attention patterns under long-tailed bias. We provide two complementary views: (i) the diversity of head-proxy CAM candidates versus the tail ground-truth CAM, and (ii) the CAM quality change of the same backbone before and after integrating SRCD. These visualizations reveal two recurring phenomena: proxy CAMs are diverse and partially complementary, and SRCD consistently reduces shortcut-driven background attention while strengthening object-centric evidence.

Head-proxy CAMs vs. tail ground-truth CAM. Figure 6 presents multiple tail-class examples, where each row contains three head-proxy CAM candidates and the corresponding tail ground-truth CAM. Although each proxy is induced by a head-biased hypothesis, high-quality candidates can still localize object-centric regions that overlap with the tail ground-truth evidence. In contrast, low-quality proxies frequently respond to scattered patterns or context/background cues, reflecting shortcut-driven correlations that are amplified by long-tailed training.

A key observation is that proxy CAMs are not identical: different proxies often emphasize different discriminative parts of the object (e.g., complementary regions), while their spurious responses are less consistent across proxies. This diversity is beneficial when aggregated properly. Our energy-based teacher construction explicitly leverages this property by favoring proxy CAMs that are spatially coherent and evidence-concentrated. As a result, the aggregated teacher tends to (i) combine complementary object evidence from multiple proxies into a more complete attention map, and (ii) suppress proxy-specific noise and background activations that do not persist across candidates. Empirically, this yields a teacher that is visually closer to the tail ground-truth CAM and is more stable across iterations, providing a robust supervision target for CAM alignment under severe class imbalance.

![](images/22a11b6bc3e5d97d9a9d5330ed529d5c4a4b5a006a2644af6f7f228c099f8eef.jpg)  
Figure 6. CAM visualizations on multiple tail-class examples. For each example (row), the first three columns show head-proxy CAM candidates, and the last column shows the tail ground-truth class CAM. High-quality proxies tend to capture object-centric evidence consistent with the ground truth, while low-quality ones highlight scattered or background-dominated regions.

Effect of SRCD on CAM quality. Figure 7 compares CAMs obtained from the same model without and with SRCD across multiple examples. Without SRCD, the model often exhibits shortcut-driven evidence: activations spread to background/context regions, and the attention mass is less spatially structured. This behavior is particularly pronounced for tail classes, where limited data encourages reliance on easy context cues rather than object evidence.

After integrating SRCD, CAMs become more concentrated on the true object with improved spatial coherence. In addition to sharper localization, we observe two qualitative improvements that are critical under long-tailed bias. First, SRCD reduces the dominance of single spurious cues by redistributing attention toward multiple object-relevant regions, resulting in more complete evidence coverage. Second, SRCD improves the consistency of attention across examples: the model relies on more stable object evidence instead of fluctuating background patterns. These observations are consistent with our design: SRCD uses a robust teacher (aggregated from diverse head-proxy candidates) to provide shortcut-resistant supervision, thereby pulling the student CAM away from background-biased modes and aligning it with object-centric evidence. Overall, the visualizations support that SRCD promotes more faithful evidence alignment and stronger shortcut resistance, especially for tail classes where spurious correlations are most harmful.

without SRCD  
![](images/d51398429c2c25b74ff18a8a945f123591ebd85666e0c201f4ef04ba20eff3ee.jpg)

with SRCD  
![](images/c37b8feae2aa1a0e19c8572a9c422d2a4d47b99c44001e9a38a91606d9279429.jpg)

![](images/79c7e34ec157267b56cb212f363b91f38ad9810e735b1760a77e31a7b80283e6.jpg)

![](images/99591c53d04f805fb15fa8a552500d35cc2963af2c933485c3a2e86207f6ea51.jpg)

![](images/3f28eee723f2b3a191081ecc5499fb4ff297264b0537ec382c8aa50b16ccd472.jpg)

![](images/70cc546877ce32a277fe8ca6b7a211054869df01b25ab614e7afefefd098d77e.jpg)  
Figure 7. Effect of SRCD on CAM localization across multiple examples. Left: without SRCD. Right: with SRCD. SRCD promotes more object-centric and spatially coherent activations while reducing background/context responses, indicating stronger shortcut resistance.

Table 11. Mask-based shortcut analysis on the long-tailed VOC 2012 subset. FG/BG Ratio measures the CAM evidence mass inside the foreground mask relative to the background region. A larger ratio indicates weaker shortcut reliance.
<table><tr><td>Method</td><td>FG/BG Ratio ↑</td><td>Acc. ↑</td></tr><tr><td>LIFT</td><td>0.85</td><td>78.37</td></tr><tr><td> $\mathrm { L I F T } + \mathrm { S R C D }$ </td><td>1.56</td><td>83.16</td></tr></table>

## D.8. Direct Evidence for Shortcut Suppression on VOC

Recognition accuracy alone may not fully reveal whether a method actually suppresses shortcut reliance. To provide more direct evidence, we conduct a mask-based analysis on Pascal VOC 2012, whose semantic segmentation masks enable foreground-background evaluation. Since the original VOC 2012 dataset is multi-label, we curate a single-label subset and reorganize it into a long-tailed distribution consistent with our setting, with an imbalance ratio of 80.

We use LIFT as the baseline and report the Foreground-to-Background Ratio (FG/BG Ratio), which measures how CAM attention is distributed inside versus outside the segmentation mask:

$$
\mathrm { F G / B G } = \frac { \sum _ { u \in \Omega _ { \mathrm { f g } } } \bar { M } ( x ) _ { u } } { \sum _ { u \in \Omega _ { \mathrm { b g } } } \bar { M } ( x ) _ { u } + \epsilon } ,
$$

where $\Omega _ { \mathrm { f g } }$ and $\Omega _ { \mathrm { b g } }$ denote the foreground and background regions given by the segmentation mask, respectively. A larger FG/BG ratio indicates that the model assigns more evidence to object regions and relies less on background shortcuts.

As shown in Table 11, SRCD substantially increases the FG/BG Ratio from 0.85 to 1.56, while also improving accuracy from 78.37 to 83.16. This provides direct evidence that SRCD shifts attention toward object regions and away from background or shortcut regions, rather than merely improving recognition accuracy.

We further provide VOC CAM visualizations in Figure 8. Compared with the baseline, SRCD produces CAMs that are more concentrated on foreground object regions and less dominated by background context. These qualitative results are consistent with the mask-based FG/BG analysis and further support that SRCD alleviates shortcut-dominated attention in practice.

![](images/32e98e0fee88d8585a86a92e7c8714444e0eeefc9e2e26ddbdae2a8ab6acfa12.jpg)  
Figure 8. CAM visualizations on the long-tailed VOC 2012 subset. Compared with the baseline, SRCD shifts attention from background or shortcut regions toward foreground object regions, which is consistent with the mask-based FG/BG analysis.

## E. Algorithm

Algorithm 1 Shortcut-Resistant CAM Distillation (SRCD)   
Require: Mini-batch B, model $f _ { \theta } ,$ head classes $\mathcal { D } _ { \mathrm { h e a d } }$ , tail classes $\mathcal { D } _ { \mathrm { t a i l } } .$ , proxy number $K .$ , temperature τ, weight $\beta .$   
1: Forward $f _ { \theta }$ to obtain logits $z _ { i }$ and feature maps $A _ { i }$   
2: Compute classification loss ${ \mathcal { L } } _ { \mathrm { c l s } } .$   
3: Initialize $\mathcal { L } _ { \mathrm { S R C D } }  0 .$   
4: for each $( x _ { i } , y _ { i } ) \in B$ with $y _ { i } \in \mathcal { V } _ { \mathrm { t a i l } }$ do   
5: Select top-K predicted head classes:   
$\begin{array} { r } { \mathcal { H } ( x _ { i } ) = \mathrm { T o p K } _ { c \in \mathcal { V } _ { \mathrm { h e a d } } } p _ { i } ( c ) . } \end{array}$   
6: Compute the student tail CAM without stop-gradient:   
$\bar { M } _ { y _ { i } } ( x _ { i } ) = \mathrm { N o r m } ( \mathrm { C A M } ( A _ { i } , y _ { i } ) )$   
7: Compute detached head-proxy CAMs:   
$\bar { M } _ { c } ^ { \mathrm { s g } } ( x _ { i } ) = \mathrm { s g } [ \mathrm { N o r m } ( \mathrm { C A M } ( A _ { i } , c ) ) ] , \quad c \in \mathcal { H } ( x _ { i } )$   
8: Compute proxy scores and weights:   
$s _ { c } = - \mathscr { E } _ { \mathrm { h } } ( \bar { M } _ { c } ^ { \mathrm { s g } } ) + \mathscr { E } _ { \mathrm { n } } ( \bar { M } _ { c } ^ { \mathrm { s g } } ) , \quad w _ { c } = \frac { \exp ( s _ { c } / \tau ) } { \sum _ { c ^ { \prime } \in \mathcal { H } ( x _ { i } ) } \exp ( s _ { c ^ { \prime } } / \tau ) } .$   
9: Construct the detached teacher CAM:   
$\bar { M } _ { T } ( x _ { i } ) = \mathrm { s g } \left[ \sum _ { c \in \mathcal { H } ( x _ { i } ) } w _ { c } \bar { M } _ { c } ^ { \mathrm { s g } } ( x _ { i } ) \right]$   
10: Accumulate SRCD loss:   
$\mathcal { L } _ { \mathrm { S R C D } }  \mathcal { L } _ { \mathrm { S R C D } } + \gamma _ { y _ { i } }  \bar { M } _ { y _ { i } } ( x _ { i } ) - \bar { M } _ { T } ( x _ { i } )  _ { 2 } ^ { 2 } .$   
11: end for   
12: Optimize:   
$\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \beta \mathcal { L } _ { \mathrm { S R C D } } . } \end{array}$   
13: Update θ by back-propagating $\mathcal { L } .$