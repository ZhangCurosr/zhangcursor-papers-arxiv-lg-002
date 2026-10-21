# Understanding Data Augmentation under Label Noise

Hong Zheng<sup>∗</sup>   
School of Computing and Artificial Intelligence   
Southwest Jiaotong University   
Chengdu, 611756, China

hirozheng@my.swjtu.edu.cn

FTENG fteng@swjtu.edu.cn

Fei Teng   
School of Computing and Artificial Intelligence   
and Engineering Research Center of Sustainable Urban Intelligent Transportation   
Southwest Jiaotong University   
and Ministry of Education   
Chengdu, 611756, China

## Abstract

Data augmentation (DA) has become a fundamental component of modern machine learning systems and is widely regarded as an efective strategy for improving robustness and generalization. Despite its empirical success, the theoretical impact of DA under label noise remains poorly understood. In this work, we first theoretically explain when DA is benefi cial and when it becomes detrimental, and provide suficient conditions for distinguishing between these two regimes under both the i.i.d. learning setting and the more challeng ing domain generalization scenario. Then, we reveal a counterintuitive phenomenon: DA essentially acts as an uncertainty amplifier in the learning process, decreasing model stability and generalization performance. This finding is supported by the proposed learning and generalization bounds. Our analyses do not depend on specific DA techniques and require only mild assumptions, indicating that the results are applicable to a broad range of machine learning scenarios. Namely, we provide a unified theoretical foundation for understanding, analyzing, and designing DA methods in the presence of label noise.

Keywords: Data Augmentation, Label Noise, Machine Learning, Domain Generalization

## 1 Introduction

Data augmentation (DA) techniques are widely applied in machine learning and are typically considered as regularization terms, aiming to increase the robustness of models for prediction (Chen et al., 2020). However, it is often unclear exactly how, why, and when DA methods work for given data (Lin et al., 2024). In particular, for real-world applications, label noise is common, and its presence harms the learning process (Pereira et al., 2022). Under such data conditions, answering the above questions is essential for understanding the underlying mechanisms of machine learning and for facilitating practical applications.

Pereira et al. (2022) conducted a series of experiments on DA methods under noisylabel conditions for empirical understanding. They stated that the appropriate selection of DA can drastically improve model robustness to label noise and argued that the best augmentation strategy appears to be dataset-dependent. However, several aspects of their results require more careful scrutiny. For example, when the noise types are symmetric or asymmetric, DA methods can substantially improve the evaluation performance even when the noise rate in the training set is as high as 0.8. In contrast, when the noise type is semantic noise, the improvement is not significant, and in some cases, performance even degrades. Note that a noise rate of 0.8 is typically considered highly challenging for machine learning, since successful learning is generally expected only when the label noise rate is less than 0.5. Nevertheless, their experiments clearly showed that DA methods do not always improve performance; in most cases, they degrade it.

In this work, we present theoretical analyses to understand DA methods under noisylabel conditions in both i.i.d. and Domain Generalization (DG) learning settings. Specifically, we theoretically explain when DA is beneficial and when it becomes detrimental, and provide suficient conditions for distinguishing between these two regimes. This exactly provides a theoretical explanation for the experimental observations in (Pereira et al., 2022). We then extend these results to the more challenging DG scenario, although the obtained conditions are more restrictive than those in the i.i.d. setting, which in turn reflects the inherent challenges of DG. Additionally, we reveal a counterintuitive phenomenon: DA essentially acts as an uncertainty amplifier in the learning process, decreasing model stability and generalization performance. This finding is supported by our proposed learning and generalization bounds, as DA methods make these bounds looser.

In conclusion, the main contributions of our work are as follows:

(1) We establish a theoretical framework for analyzing DA under label noise in both i.i.d. and independent and non-identically distributed (i.n.d.) settings. Within this framework, we derive suficient conditions under which DA fails. We use simple and general models to represent various types of label noise and DA methods. The introduced assumptions are mild in most machine learning scenarios, and the lemmas used in the proofs are simple and classical. Moreover, the presented theorems establish a theoretical foundation for developing future DA methods under noisy-label settings.

(2) We reveal that DA increases uncertainty in both learning and generalization, regardless of the existence of label noise. It is well known that machine learning is commonly studied under the framework of probably approximately correct (PAC) learning theory. This implies that the learning process is inherently uncertain, and the use of DA methods may further amplify this uncertainty, potentially leading to significant risks in high-risk application domains, such as medical diagnosis, autonomous driving, and financial decision-making.

## 2 Related Works

In this section, we briefly discuss the development of DA methods and related empirical and theoretical analyses, providing an initial glance at studies related to DA. For more details, please refer to the cited references.

Initially, DA methods are introduced in the field of computer vision (Baird, 1992), and currently many DA methods are still mainly focused on computer vision tasks, such as image flipping, rotation, Mixup series (Zhang et al., 2018; Yao et al., 2022), etc. Some methods based on generative adversarial networks (GANs) are also introduced, such as DA-GANs (Antoniou et al., 2017). Moreover, nowadays, methods based on large models are also presented (Zhou et al., 2024). GAN-based and large-model-based methods typically assume that the generated $x _ { i } ^ { \prime }$ is independent of $x _ { i }$ for $i \in [ n ]$ , where n denotes the number of data samples under the i.i.d. assumption. Thus, they typically consider that their methods expand the amount of data for learning. However, considering a non-independent assumption is more natural (Chen et al., 2020). Thus, in our work, we also consider the nonindependent setting and assume that $\boldsymbol { x } _ { i } ^ { \prime }$ is independent of $x _ { j }$ for any $i \neq j \in [ n ]$ . Namely, as long as the augmented inputs are independent across diferent training samples, classical learning-theoretic analysis remains applicable. Therefore, we do not regard DA methods as increasing the efective number of training samples. Additionally, there are also some studies that focus on DA methods under the learning scenario of domain generalization or out-of-domain settings, such as (Gao et al., 2023; Tiwary et al., 2025).

Along the development of DA, learning with noisy labels has gained increasing attention, as most real-world applications sufer from such data conditions (Natarajan et al., 2013b; Han et al., 2025). Consequently, some studies also consider using DA methods for learning robust models under label noise, such as (Han et al., 2018; Zhang et al., 2020). Moreover, some studies suggest that Mixup-based methods can benefit learning under noisy data (Hwang et al., 2024). One particular work (Pereira et al., 2022) has recently attracted our attention; although published several years ago, it systematically conducted a series of experiments on DA methods under label noise to empirically investigate their behavior. However, the study did not provide theoretical explanations for its findings, and several experimental results lacked suficient justification.

Theoretical analysis in machine learning is also important for its development, as it aims to explain learning phenomena in a direct manner, namely through mathematical analysis. Similarly, several studies have attempted to provide theoretical explanations for DA methods. For example, existing studies have sought to explain the efectiveness of DA using diferent theoretical frameworks, including Markov processes (Dao et al., 2019), kernel methods (Donhauser et al., 2021), and nonlinear regularization (LeJeune et al., 2019). Chen et al. (2020) showed that DA is equivalent to an averaging operation over the orbits of a certain group that approximately preserves the data distribution. Lin et al. (2024) found that, on the one hand, DA improves generalization, while on the other hand, it is less forgiving for over-parameterized regression models. These findings suggest that the efectiveness of DA strongly depends on the task, model, and augmentation type. However, the above studies generally do not account for the presence of label noise.

## 3 A Classical Learning Perspective

## 3.1 Setup

Basic Data Assumption. We first clarify the basic data structure for subsequent analysis. Under the classical supervised learning setting, assuming X and Y being respectively an input and output space, the clean dataset without labeling noise is denoted as

$$
D ^ { c } = \{ ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { n } , y _ { n } ) \} ,
$$

where each pair $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ is independently drawn from an unknown joint distribution $P ( x , y ) , { \mathrm { i . e . , ~ } } ( x , y ) \sim _ { i . i . d . } P ( x , y )$ . The dataset $D ^ { c }$ of size n is used as the training set for learning models. Correspondingly, a test dataset is assumed to be drawn from the same

distribution $P ( x , y )$ . Due to the i.i.d. assumption, the test dataset is typically omitted from the theoretical analysis. Therefore, we do not introduce specific notation for the test data.

In this work, we mainly focus on the noisy label condition for data. Thus, we assume the training data

$$
D = \{ ( x _ { 1 } , \tilde { y _ { 1 } } ) , \dots , ( x _ { n } , \tilde { y _ { n } } ) \} .
$$

Here, we assume that $\tilde { y } \sim P ( \tilde { y } | x , y )$ . For this assumption, we provide several explanations. This labeling noise model $P ( \tilde { y } | x , y )$ is, in fact, a master model that can represent any noisy model depending on how it is constrained. For example, for a binary classification problem, if we consider the symmetric uniform noise model $P ( \tilde { y } \neq y | y ) = \eta$ and $P ( \tilde { y } = y | y ) = 1 - \eta .$ where $\eta \in [ 0 , 1 ]$ , we have $P ( \tilde { y } | x , y ) = P ( \tilde { y } | y )$ . If we consider the Massart noise model $P ( \tilde { y } \neq y | x ) = \eta ( x )$ , where $\eta ( x ) \leq \eta _ { m a x } \leq 1 / 2$ , we have $P ( \tilde { y } | x , y ) = P ( \tilde { y } | x )$ . Moreover, the instance-dependent noise model $P ( \tilde { y } | x , y ) = P ( \tilde { y } \neq y | x , y )$ can also be considered.

Based on the assumed labeling noise model, we obtain a new data distribution $P ( x , \tilde { y } ) =$ $\begin{array} { r } { \sum _ { y } P ( x , y ) P ( \tilde { y } | x , y ) } \end{array}$ ; if considering the continuous case, $\begin{array} { r } { P ( x , \tilde { y } ) = \int P ( x , y ) P ( \tilde { y } | x , y ) d y } \end{array}$ . In fact, we can consider the operator $P ( \tilde { y } | x , y )$ as performing an irreversible transformation on the original distribution $P ( x , y )$ , which is a convolutional operator.

Data Augmentation. Having introduced the noisy labeled data model and its associated distribution, we now consider the data augmentation (DA) model. In practice, two types of DA are commonly used: online and ofline DA. Online DA refers to applying augmentation during the training process, whereas ofline DA refers to augmenting the training dataset prior to training.

In this work, we ONLY focus on the online setting in our theoretical analysis, as the ofline setting may violate the i.i.d. assumption. For example, for any $i \in [ n ]$ , let $x _ { i } ^ { \prime } = T ( x _ { i } )$ , where $T ( \cdot \mid x )$ is a stochastic transformation kernel. It is clear that $x _ { i } ^ { \prime }$ depends on $x _ { i }$ introducing dependence between the original and augmented samples. In the online setting, it is suficient to assume that $x _ { i } ^ { \prime }$ is independent of $x _ { j }$ for all $j \neq i \in [ n ]$ . Consequently, there is no need to consider the number of times that $T$ is applied, as repeated applications cannot increase the number of independently generated samples. That is, after applying $T _ { i }$ , the number of data samples in $D$ remains $n .$

We now model DA as a stochastic transformation kernel $T ( \cdot | x )$ , i.e.,

$$
x ^ { \prime } \sim T ( \cdot | x ) .
$$

Note that this $T$ can de also considered as a mixture of $m$ kernels, such as $T ( x ^ { \prime } | x ) =$ $\begin{array} { r } { \sum _ { j } ^ { m } \pi _ { j } t _ { j } ( x ^ { \prime } | x ) } \end{array}$ , where $\textstyle \sum _ { j } \pi _ { j } = 1$ . Since this consideration does not afect the analysis, we omit further discussion of it. Now, based on the DA kernel $T$ , we obtain the following distribution

$$
P _ { D A } ( x ^ { \prime } , \tilde { y } ) = \int T ( x ^ { \prime } | x ) P ( x , \tilde { y } ) d x .
$$

Typically, this distribution is considered a pushforward distribution in most of the literature. In our analysis, we additionally treat it as a fixed distribution obtained by applying $T$ to $P ( x , \tilde { y } )$ , where $T$ can also be regarded as a non-orthogonal convolutional kernel. In most training scenarios, $T$ is not 1-1, that is, there does not exist an inverse mapping $T ^ { - 1 }$ such that $x = T ^ { - 1 } ( x ^ { \prime } )$ . Consequently, unless T is 1-1, $P ( x ^ { \prime } , \tilde { y } ) \neq P ( x , \tilde { y } )$ , which supports the validity of our consideration. Additionally, we assume that for P-almost every $( x ^ { \prime } , \tilde { y } )$ , it holds that $P _ { D A } ( x ^ { \prime } , \tilde { y } ) > 0$ . This assumption implies that $P ( x , \tilde { y } ) \ll P ( x ^ { \prime } , \tilde { y } )$ , allowing the Kullback–Leibler (KL) divergence to be used to measure their discrepancy in the following analysis. If one considers this assumption to be strong, the Jensen–Shannon (JS) divergence can also be used instead.

The kernel T does not afect the label ˜y. If we consider DA methods such as Mixup (Zhang et al., 2018), which introduce new labels, this $T$ is no longer suficient. Thus, we introduce the following model

$$
( x ^ { \prime } , \tilde { y } ^ { \prime } ) \sim T ^ { \prime } ( \cdot | x , \tilde { y } ) ,
$$

and correspondingly the distribution is

$$
P _ { D A } ( x ^ { \prime } , \tilde { y } ^ { \prime } ) = \iint T ^ { \prime } ( x ^ { \prime } , \tilde { y } ^ { \prime } | x , \tilde { y } ) P ( x , \tilde { y } ) d x d \tilde { y } .
$$

Similarly, we assume that $P _ { D A } ( x ^ { \prime } , \tilde { y } ^ { \prime } ) > 0$ for P-almost every $( x ^ { \prime } , \tilde { y } ^ { \prime } )$ . For notational convenience, we subsequently use $P ^ { c } , P , P _ { D A }$ , and $P _ { D A } ^ { \prime }$ to represent $P ( x , y ) , P ( x , \tilde { y } ) , P _ { D A } ( x ^ { \prime } , \tilde { y } )$ and $P _ { D A } ( x ^ { \prime } , \tilde { y } ^ { \prime } )$ , respectively.

Others. We consider a hypothesis space H equipped with a measurable structure, and let $h \in { \mathcal { H } } : \mathcal { X } \to \mathcal { Y }$ denote an arbitrary hypothesis. To quantify the prediction accuracy of h, we introduce a cost function $c : \mathcal { V } \times \mathcal { V } \to \mathbb { R } _ { > 0 }$ . Then, the loss of h with respect to any data pair $( x , y )$ is defined as

$$
\ell ( h , ( x , y ) ) = c ( h ( x ) , y ) .
$$

Based on this loss, we then introduce a definition of a property of the loss function as follows.

Definition 1 (σ-admissible loss function (Bousquet and Elisseef, 2002)) A loss function ℓ defined on $\mathcal { H } \times \{ \mathcal { X } \times \mathcal { Y } \}$ is σ-admissible with respect to H if the associated cost function c is convex with respect to its first argument and the following condition holds

$$
\forall y _ { 1 } , y _ { 2 } \in \mathcal { D } , \forall y \in \mathcal { V } , | c ( y _ { 1 } , y ) - c ( y _ { 2 } , y ) | \leq \sigma \left| y _ { 1 } - y _ { 2 } \right| ,
$$

where $\mathcal { D } = \{ y : \exists h \in \mathcal { H } , \exists x \in \mathcal { X } , h ( x ) = y \}$ is the domain of the first argument of c.

Remarks: (a) This definition, in fact, implies that ℓ is σ-Lipschitz. (b) $\| \nabla \ell ( \cdot ) \| _ { \infty } = \sigma$

For measuring the discrepancy between two distributions, if $P \ll Q$ , we use the following divergence.

Definition 2 (Kullback–Leibler (KL) divergence) Let P and Q be two probability measures on the same measurable space (Z, Ω), and assume that P is absolutely continuous with respect to $Q , i . e . , P \ll Q$ . The KL-divergence from P to $Q$ is defined as

$$
K L ( P | | Q ) : = \int _ { \mathcal Z } l o g \left( \frac { d P } { d Q } ( z ) \right) d P ( z ) .
$$

In the discrete case:

$$
\begin{array} { r } { K L ( P | | Q ) : = \sum _ { z \in \mathcal { Z } } P ( z ) l o g \displaystyle \frac { P ( z ) } { Q ( z ) } . } \end{array}
$$

Remarks: ${ \mathrm { ( a ) ~ } } 0 \cdot \log ( 0 / 0 ) = 0 . \quad { \mathrm { ( b ) ~ } } \exists a : Q ( a ) = 0 , P ( a ) > 0 \Rightarrow K L ( P | | Q ) = \infty$ $K L ( P | | Q ) \geq 0 ; K L ( P | | Q ) = 0 ~ i f f ~ P = Q . ~ ( \mathrm { d } ) ~ K L ( P | | Q ) \neq K L ( Q | | P )$

(c)

We additionally introduce the Wasserstein-1 distance, which characterizes distributional discrepancy through geometric discrepancy. In certain cases, this characterization simplifies the analysis. However, when the conditions required for such a transformation are restrictive, we still employ the KL divergence as the discrepancy measure.

Definition 3 (Wasserstein-1(W<sub>1</sub>) Distance) Let $( \mathcal { Z } , d )$ be a metric space, and let P and $Q$ be probability measures on Z. The $W _ { 1 }$ distance between P and $Q$ is defined as

$$
W _ { 1 } ( P , Q ) : = \operatorname* { i n f } _ { \gamma \in \Pi ( P , Q ) } \mathbb { E } _ { ( z _ { 1 } , z _ { 2 } ) \sim \gamma } [ d ( z _ { 1 } , z _ { 2 } ) ] ,
$$

where $\Pi ( P , Q )$ denotes the set of all coupling of P and Q, and each $\gamma \in \prod ( P , Q )$ is a joint distribution whose marginals are P and Q, respectively.

Remarks: (a) $W _ { 1 } ( P , Q ) \geq 0 ; W _ { 1 } ( P , Q ) = 0 \ i f f \ P = Q .$ (b) $W _ { 1 } ( P , Q ) = W _ { 1 } ( Q , P )$ . (c) $W _ { 1 } ( P , Q ) \leq W _ { 1 } ( P , M ) + W _ { 1 } ( M , Q )$ for any distribution M. (d) $W _ { 1 } ( P , Q ) \le D i a$ , where Dia is the diameter of the union of the supports.

## 3.2 A Warm-up Analysis

In this subsection, we provide several interesting results based on a strong but simple assumption. We begin by imposing the following assumption on D.

Assumption 4 (Separability of D) Assume that D contains η-noisy labels such that $n =$ $n _ { \eta } + n _ { c }$ and $n _ { \eta } \leq n _ { c }$ . Here, $n _ { \eta }$ denotes the number of noisy samples.

Remarks: Here, we assume that the dataset D can be explicitly partitioned into clean and noisy subsets. In practice, however, this assumption is relatively strong. Nevertheless, this simplified yet strong assumption enables several interesting analyses. Here, η is not associated with any specific noise model, and the $n _ { \eta }$ noisy samples may arise from arbitrary sources; η only serves as a criterion for separating noisy data. Note that we strictly prohibit splitting D according to η after applying DA methods.

First, we present our analysis based on optimization theory and define the expected risk as follows:

$$
R ( h ) = \mathbb { E } _ { ( x , y ) \sim P } [ \ell ( h , ( x , y ) ) ] .
$$

For optimization theory, we are typically interested in the following learning objective:

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim P } [ \ell ( h _ { \theta } , ( x , y ) ) ] ,
$$

where θ denotes the associated parameters for h. Hereafter, we omit the subscript $\theta$ in $h _ { \theta }$ i.e., we treat h as parameterized. Then, under diferent conditions of the training sets, we have

$$
\mathcal { L } ^ { c } ( \theta ) = \mathbb { E } _ { ( x , y ) } [ \ell ( h , ( x , y ) ) ] = \mathbb { E } _ { ( x , y ) \sim P ^ { c } } [ \ell ( h , ( x , y ) ) ]
$$

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { ( x , \tilde { y } ) } [ \ell ( h , ( x , \tilde { y } ) ) ] = \mathbb { E } _ { ( x , y ) \sim P } [ \ell ( h , ( x , y ) ) ]
$$

$$
\mathcal { L } _ { D A } ( \theta ) = \mathbb { E } _ { ( x , \tilde { y } ) } \mathbb { E } _ { x ^ { \prime } \sim T ( \cdot | x ) } [ \ell ( h , ( x ^ { \prime } , \tilde { y } ) ) ] = \mathbb { E } _ { ( x , y ) \sim P _ { D A } } [ \ell ( h , ( x , y ) ) ]
$$

$$
\mathcal { L } _ { D A } ^ { \prime } ( \theta ) = \mathbb { E } _ { ( x , \tilde { y } ) } \mathbb { E } _ { ( x ^ { \prime } , \tilde { y } ^ { \prime } ) \sim T ^ { \prime } ( \cdot | x , \tilde { y } ) } [ \ell ( h , ( x ^ { \prime } , \tilde { y } ^ { \prime } ) ) ] = \mathbb { E } _ { ( x , y ) \sim P _ { D A } ^ { \prime } } [ \ell ( h , ( x , y ) ) ]
$$

Based on these learning objectives, we consider the classical optimization process, i.e., Stochastic Gradient Descent (SGD), and introduce the following assumption for illustration.

Assumption 5 (Gradient Descent Method) Assume that the learning objective function $\mathcal { L } ( \boldsymbol { \theta } )$ is smooth, and that the parameters θ are optimized using the gradient descent method, which is defined as

$$
\theta _ { k + 1 } = \theta _ { k } - \alpha _ { k } \nabla { \mathcal { L } } ( \theta _ { k } ) ,
$$

where k denotes the epoch index and $\alpha _ { k }$ is the learning rate.

Remarks: (a) As we consider that ℓ is σ-admissible, the smoothness requirement in this assumption can be easily satisfied. (b) Although we denote the subscript k in $\alpha _ { k }$ , unless learning rate decay methods are considered, $\alpha _ { k }$ remains unchanged.

Why do we consider SGD as the basic optimization method? The reason is that most optimization algorithms used in modern machine learning and deep learning remain gradientbased, including SGD with momentum, Adam, AdamW, and their variants. Therefore, under this assumption, our analysis covers the vast majority of optimization scenarios encountered in practice. As a result, it cannot be considered a strong assumption unless one adopts optimization methods that are independent of loss gradients.

Analyzing optimization convergence rates is a major topic in optimization theory. Thus, based on the assumptions of SGD, we present the following theorem on the convergence of the learning objective $\mathcal { L } ( \boldsymbol { \theta } )$ using dataset D, together with the noise tolerance rate of SGD optimization.

Theorem 6 (Noise Tolerance under SGD Optimization) For any $h \in { \mathcal { H } }$ , assume that the cost function c is convex and satisfies Definition 1, and let $\theta ^ { * c } \in a r g m i n _ { h _ { \theta } \in \mathcal { H } } \mathcal { L } ^ { c } ( \theta )$ Under the SGD optimization scheme in Assumption 5, training on the dataset D with noise rate η yields an averaged iterate $\theta _ { K }$ satisfying

$$
\mathbb { E } [ \mathcal { L } ( \bar { \theta } _ { K } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) ] \leq \mathcal { O } \left( \frac { 1 } { \sqrt { K } } + \eta \sigma \Delta _ { \| \tilde { y } - y \| } \right) ,
$$

which implies a noise tolerance characterized by the optimization error $\varepsilon > 0$ 2

$$
\eta < \frac { \varepsilon } { \sigma \Delta _ { \parallel \tilde { y } - y \parallel } } .\tag{1}
$$

Here, $\Delta _ { \parallel \tilde { y } - y \parallel } = \mathbb { E } _ { y } \mathbb { E } _ { \tilde { y } } \lVert \tilde { y } - y \rVert$

Remarks: (a) The convergence rate is sublinear. ησ∆ characterizes the optimization bias but does not afect the overall convergence rate. (b) The loss functions are assumed to be convex rather than strongly convex. (c) $\Delta _ { \parallel \tilde { y } - y \parallel }$ denotes the label bias induced by noisy labels. (d) Inequality (1) can, in fact, be written as $\eta \Delta < \varepsilon / \sigma$ . (e) The proofs of this and the following theoretical results are provided in Section $\mathrm { A . 2 }$

Note that here we consider the optimization objective $| { \mathcal { L } } ( \theta ) - { \mathcal { L } } ^ { c } ( \theta ^ { * c } ) |$ instead of $| { \mathcal { L } } ( \theta ) -$ $\mathcal { L } ( \theta ^ { \ast } ) |$ . This choice allows us to explicitly characterize the impact of noisy labels on the optimization process. In contrast, optimizing $| { \mathcal { L } } ( \theta ) - { \mathcal { L } } ( \theta ^ { * } ) |$ leads to the classical optimiza tion setting, where the analysis is independent of the presence of noisy labels and therefore cannot capture their efect.

The above theorem yields several interesting results. For convex optimization problems based on $D ^ { c }$ , it is well known that the convergence rate is sublinear. The theorem shows that this convergence rate remains unchanged even in the presence of label noise. In particular, the convergence rate is still sublinear, and the only diference is the appearance of an optimization bias term induced by the noise rate η and the noise bias $\Delta _ { \parallel \tilde { y } - y \parallel }$ . When $\eta = 0$ which implies $\Delta _ { \parallel \tilde { y } - y \parallel } = 0$ , the result reduces to the classical convergence guarantee. As remarked above, this optimization bias does not afect the convergence rate, suggesting that noisy labels do not significantly increase the learning cost. Nevertheless, the resulting solution may deviate from $\theta ^ { * c }$ by $\eta \Delta$ . The most interesting result is that we observe a noise tolerance for SGD methods, i.e., Inequality (1). This indicates that the deviation of the resulting solution from $\theta ^ { * c }$ is upper bounded by $\varepsilon / \sigma , \mathrm { i . e . , } \eta \Delta < \varepsilon / \sigma$ , where $\sigma$ is linked to the properties of the loss function ℓ. If we consider a large $\varepsilon ,$ the optimization can tolerate a larger η, and vice versa.

Now, we clearly acknowledge how η afects the optimization process under the training set $D .$ . Next, we observe how $\eta$ afects the optimization process under the training set D when applying $T$ or $T ^ { \prime }$ . We first present the following theorem.

Theorem 7 (Noise Tolerance under SGD Optimization with DA) Under the setting of Theorem 6 and the $W _ { 1 }$ distance defined in Definition $^ { 1 \downarrow , }$ training on the dataset D with the DA kernel $T ( \cdot | x )$ under noise rate η produces an averaged iterate $\theta _ { K }$ satisfying

$$
\mathbb { E } [ \mathcal { L } _ { D A } ( \bar { \theta } _ { K } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) ] \leq \mathcal { O } \left( \frac { 1 } { \sqrt { K } } + \sigma \Delta _ { \| x ^ { \prime } - x \| } + \eta \sigma \Delta _ { \| \tilde { y } - y \| } \right) ,
$$

which implies a noise tolerance characterized by the optimization error $\varepsilon > 0$ 2

$$
\eta < \frac { \varepsilon - \sigma \Delta _ { \parallel x ^ { \prime } - x \parallel } } { \sigma \Delta _ { \parallel \tilde { y } - y \parallel } } .\tag{2}
$$

Here, $\Delta _ { \parallel x ^ { \prime } - x \parallel } = \mathbb { E } _ { x } \mathbb { E } _ { x ^ { \prime } } \lVert x ^ { \prime } - x \rVert$

Remarks: (a) The convergence rate remains sublinear. The optimization bias is increased by an additional term $\sigma \Delta _ { \parallel x ^ { \prime } - x \parallel }$ , yet this does not afect the overall convergence rate. (b) The loss functions are still assumed to be convex. (c) $\Delta _ { \parallel x ^ { \prime } - x \parallel }$ denotes the data bias induced by applying T, i.e., DA methods. (d) Inequality (2) can similarly be rewritten as $\eta \Delta _ { \parallel \tilde { y } - y \parallel } <$ $( \varepsilon - \Delta _ { \parallel x ^ { \prime } - x \parallel } ) / \sigma . \mathrm { ( e ) }$ If $\Delta _ { \parallel x ^ { \prime } - x \parallel } = 0$ , then Inequality (2) reduces to Inequality (1). Similarly, we consider the the optimization objective $| \mathcal { L } _ { D A } ( \theta ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) |$ instead of $| \mathcal { L } _ { D A } ( \theta ) - \mathcal { L } _ { D A } ( \theta _ { D A } ^ { * } ) |$ . For this consideration, we align the objective in Theorem 6, i.e., $\left| \mathcal { L } ( \theta ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) \right|$ . This allows us to directly characterize the impact of introducing DA methods in optimization, as the convergence target is the same $\mathcal { L } ^ { c } ( \theta ^ { * c } )$ . Moreover, we still need to analyze the impact of noisy labels on optimization.

The above theorem still yields several interesting results. For the convergence rate, after applying $T ,$ it remains sublinear, which indicates that using DA methods cannot accelerate nor slow down convergence. However, it introduces an additional optimization bias term induced by the data bias, i.e., $\Delta _ { \parallel x ^ { \prime } - x \parallel } = 0$ . Interestingly, we find that when $\eta = 0$ , we obtain the convergence rate for $D ^ { c }$ after applying T. This result indicates that, regardless of whether the training data contains label noise, DA methods always introduce an optimization bias, which may cause the resulting solution to deviate from the original optimal one. As remarked above, when $\Delta _ { \parallel x ^ { \prime } - x \parallel } = 0 .$ , we obtain the same result as in Theorem 6. However, for DA methods, this data bias is unavoidable since their basic operation is to introduce transformations that alter the original information in x, rather than correcting or preserving it. Although the convergence rate remains unchanged, the noise tolerance is reduced, i.e., $\eta \Delta _ { \parallel \tilde { y } - y \parallel } < ( \varepsilon - \Delta _ { \parallel x ^ { \prime } - x \parallel } ) / \sigma$ , provided that $\Delta _ { \parallel x ^ { \prime } - x \parallel } > 0$

This data bias is problematic for the optimization because we cannot guarantee an upper bound on $\Delta _ { \parallel x ^ { \prime } - x \parallel }$ , implying that the magnitude of this bias term can become arbitrarily large. Consequently, an arbitrarily large data bias may lead to $\eta < 0$ , which indicates that SGD can no longer tolerate label noise. As a result, under noisy labels conditions, any estimate of the deviation between the solution of the learning objective and the optimal solution becomes meaningless. In other words, we cannot guarantee a meaningful connection between the model trained with DA methods and the optimal model. Therefore, one condition should be satisfied, i.e., $\varepsilon \geq \Delta _ { \parallel x ^ { \prime } - x \parallel }$ . Under this condition, we observe that the noise tolerance decreases. This is interesting, as before using the DA method, the optimization could tolerate a relatively large amount of noise; however, now the optimization can tolerate less noise. Consequently, the deviation between the solution of $\mathcal { L } _ { D A } ( \theta )$ and the optimal solution must increase, as $\sigma \Delta _ { \parallel x ^ { \prime } - x \parallel } + \eta \sigma \Delta _ { \parallel \tilde { y } - y \parallel }$ . Now, if we consider using $T ^ { \prime }$ , do the above results still hold? We present the following corollary for further analysis.

Corollary 8 Under the setting ofTheorem 7 and the KL divergence defined in Definition ${ \mathcal { Q } } ,$ additionally assume that ℓ is upper bounded by M. Then, training on the dataset D with the DA kernel $T ^ { \prime } ( \cdot | ( x , \tilde { y } ) )$ under noise rate η yields a noise tolerance characterized $b y$ the optimization error ε

$$
\eta < \frac { \varepsilon - M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) } } { \sigma \Delta _ { \parallel \tilde { y } - y \parallel } } .
$$

Remarks: (a) The above noise tolerance is similar to the one obtained in Theorem 7. (b) Considering the behavior of the risk as $K L  \infty$ , this bound may become meaningless. (c) If $K L = 0$ , then the above tolerance reduces to Inequality (1).

The above result is similar to that obtained from Theorem $^ { 7 , }$ but $\Delta _ { \parallel x ^ { \prime } - x \parallel }$ is replaced by $M \sqrt { K L } ,$ as $T ^ { \prime }$ transforms the labels as well. This implies that, when using $T ^ { \prime }$ , the condition becomes more restrictive, as we need $\varepsilon \ge M \sqrt { K L }$ . Then, if $\sqrt { K L } \geq \Delta _ { \parallel x ^ { \prime } - x \parallel }$ , we argue that the noise tolerance becomes smaller when using $T ^ { \prime }$ . In contrast, if $\sqrt { K L } < \Delta _ { \parallel x ^ { \prime } - x \parallel }$ we consider that using $T$ obtains a smaller noise tolerance. However, we do not need to compare the specific relationship between them, as both $T$ and $T ^ { \prime }$ decrease the noise tolerance compared to the result when DA methods are not incorporated.

One may ask whether assuming that the cost function c is µ-strongly convex can alleviate the reduction in noise tolerance. The answer is negative, since $\eta$ becomes coupled with additional terms, rather than depending solely on $\Delta$ . This further complicates the analysis of noise tolerance. The detailed derivation is omitted here and left to the reader.

In conclusion, based on the above theoretical results, we find that DA methods always introduce the optimization bias, which agin with the analysis in (Lin et al., 2024). However, one normally introduce DA methods aim to increase the robust for the trained models. Then, we give our sight on the learning consistency theory analysis.

We consider the empirical risk minimization (ERM) objective defined as ${ \hat { R } } ( h ) \ =$ $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \ell ( h ( x _ { i } ) , y _ { i } )$ Correspondingly, under diferent data conditions, we denote the empirical risks by $\hat { R } ^ { c } ( h ) , \hat { R } ( h ) , \hat { R } _ { D A } ( h )$ , and $\hat { R } _ { D A } ^ { \prime } ( h )$ , i.e.,

$$
\begin{array} { r l } & { \displaystyle R ^ { c } ( h ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( h , ( x _ { i } , y _ { i } ) ) } \\ & { \displaystyle \hat { R } ( h ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( h , ( x _ { i } , \tilde { y } _ { i } ) ) } \\ & { \displaystyle \hat { R } _ { D A } ( h ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( h , ( x _ { i } ^ { \prime } , \tilde { y } _ { i } ) ) \quad x _ { i } ^ { \prime } \sim T ( \cdot \mid x _ { i } ) } \\ & { \displaystyle \hat { R } _ { D A } ^ { \prime } ( h ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( h , ( x _ { i } ^ { \prime } , \tilde { y } _ { i } ^ { \prime } ) ) \quad ( x _ { i } ^ { \prime } , \tilde { y } _ { i } ^ { \prime } ) \sim T ^ { \prime } ( \cdot \mid x _ { i } , \tilde { y } _ { i } ^ { \prime } ) . } \end{array}
$$

The corresponding expected risks are denoted by $R ^ { c } ( h ) , R ( h ) , R _ { D A } ( h )$ , and $R _ { D A } ^ { \prime } ( h )$ , and their specific forms (see related ${  { \mathcal { L } } } ( \theta ) )$ are defined at the beginning of this subsection. Note that we do not consider using $T$ or $T ^ { \prime }$ to increase the number of samples, since, under the online DA setting, regardless of how many times $T$ or $T ^ { \prime }$ is applied, the generated samples are still treated as a single sample in ERM. If we instead consider using $T$ or $T ^ { \prime }$ to increase the sample size, the resulting samples would generally violate the i.i.d. assumption.

Now, given the empirical risks and the expected risks, according to classical learning theory, we typically need to analyze the learning consistency, i.e., to determine the gap between the empirical risks and the expected risks. We therefore present the following result.

Theorem 9 (Learning Consistency) Given the training dataset D with η-noisy labels and the corresponding definitions of the risk functions, assume that $\ell \in [ 0 , M ]$ . Then, for any $h \in \mathcal H$ and any $\delta \in ( 0 , 1 )$ , the following inequality holds with probability at least $1 - \delta$

$$
\hat { R } ( h ) - R ( h ) \leq 2 \eta M \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n _ { \eta } } } + 2 ( 1 - \eta ) M \sqrt { \frac { \ln ( 2 / \delta ) } { 2 n _ { c } } } .
$$

Here, $n = n _ { \eta } + n _ { c }$ denotes the total number of data samples in $D$

Remarks: (a) The main order of this upper bound is O(min $\{ 1 / \sqrt { n _ { \eta } } , 1 / \sqrt { n _ { c } } \} )$ . (b) If $\eta = 0$ this bound reduces to the well-known learning bound in the literature, $\mathrm { i . e . , } \varepsilon \leq \mathcal { O } ( 1 / \sqrt { n } )$

Note that we still assume Assumption 4 holds for D. Thus, the above learning bound can be regarded as a combination of the learning bounds for data with noise and data without noise. This bound, in fact, tells us that $\mathrm { i f } \ \eta$ is small, the learning gap between $\hat { R } ( h )$ and $R ( h )$ tends to approach the clean data side. In contrast, if $\eta$ is large, the learning gap clings to the noisy data side. This result aligns with the above results, which indicates that in real-world applications, it is acceptable to have some label noise in data when training models, unless the noise rate is large enough to afect learning consistency. Interestingly, based on the above bound, we derive the following inequality for estimating η.

Corollary 10 (Noise Bound) Based on Theorem 9, for any $\varepsilon > 0$ , we derive a noise bound:

$$
\frac { \varepsilon - B } { A - B } \leq \eta \leq \frac { B - \varepsilon } { B - A } ,\tag{3}
$$

where $A = \sqrt { ( 2 M ) ^ { 2 } \ln ( 1 / \delta ) / 2 n _ { \eta } }$ and $B = \sqrt { ( 2 M ) ^ { 2 } \ln ( 1 / \delta ) / 2 n _ { c } }$

Remarks: (a) If $A \gg B$ , i.e., $n _ { c } \to \infty$ , then $\eta \approx \varepsilon / A$ . (b) if $A \ll B , { \mathrm { i . e . , ~ } } n _ { \eta } \to \infty$ , then $\eta \approx ( B - \varepsilon ) / B , { \mathrm { i . e . , } } \eta \approx 1$

The above estimation for η seems to be meaningless, as in Assumption 4, we assume that the data can be clearly partitioned into clean and noisy parts. This means that $\eta = n _ { \eta } / n$ and when $n _ { \eta } \ \to \ n ,$ we have $\eta = 1$ . In fact, from the perspective of (Natarajan et al., 2013a), the above result can also be regarded as a measure of noise tolerance, i.e., its role is similar to that of Inequality (1). Then, for a small ε, we probably need η to be confined within the above interval. Here, we explain this result from another perspective. This perspective implies that the efect of noisy labels on learning consistency is both suficient and necessary with respect to the noise rate. In simple words, given a noise rate, we can estimate the magnitude of the learning gap. Conversely, given an acceptable learning gap, we can estimate the corresponding noise rate.

Then, we need to verify whether the relationship between the efect of noisy labels on learning consistency and the noise rate still holds after applying $T$ and $T ^ { \prime }$ to D. We therefore present the following theorem.

Theorem 11 (Learning Consistency with DA) Given the training dataset D with $\eta -$ noisy labels, the DA kernel $T ( \cdot \mid x )$ , the corresponding risk function definitions, and the $W _ { 1 }$ distance defined in Definition $^ { 1 \downarrow , }$ assume that $\ell \in [ 0 , M ]$ and that the cost function c is convex and satisfies Definition 1. Then, for any $h \in \mathcal H$ and any $\delta \in ( 0 , 1 )$ , the following inequality holds with probability at least $1 - \delta$ 2

$$
\begin{array} { r l } & { \hat { R } _ { D A } ( h ) - R ( h ) \leq \sigma \Delta _ { \| x ^ { \prime } - x \| } + 4 M \sqrt { \frac { \ln ( 3 / \delta ) } { 2 n } } } \\ & { \qquad + 2 \eta M \sqrt { \frac { \ln ( 3 / \delta ) } { 2 n _ { \eta } } } + 2 ( 1 - \eta ) M \sqrt { \frac { \ln ( 3 / \delta ) } { 2 n _ { c } } } . } \end{array}
$$

Here, $n = n _ { \eta } + n _ { c }$ represents the number of data samples in D and $\Delta _ { \parallel x ^ { \prime } - x \parallel } = \mathbb { E } _ { x } \mathbb { E } _ { x ^ { \prime } } \lVert x ^ { \prime } - x \rVert$

Remarks: (a) The main order of this upper bound is $\mathcal { O } ( 1 / \sqrt { n } + \operatorname* { m i n } \{ 1 / \sqrt { n _ { \eta } } , 1 / \sqrt { n _ { c } } \} )$ (b) There is a bias term $\Delta _ { \parallel x ^ { \prime } - x \parallel }$ , which may lead to a loose or even vacuous bound, even when $n \to \infty$ . (c) If $\eta = 0$ , the bias term still exists, and therefore continues to afect the tightness of the bound.

From the above learning bound, we observe that there is an additional bias term on the right-hand side (RHS) compared with the bound in Theorem 9. This bias term also appears in Inequality 2. The result indicates that the data bias induced by DA methods afects not only the optimization process but also the learning consistency. As remarked above, when this bias term is large, the learning bound may become loose or even vacuous. Moreover, we observe that the main order of the bound changes, namely, the convergence rate deteriorates, since $\mathcal { O } ( 1 / \sqrt { n } ) \leq \mathcal { O } ( 1 / \sqrt { n } + \operatorname* { m i n } \{ 1 / \sqrt { n _ { \eta } } , 1 / \sqrt { n _ { c } } \} )$ . This implies that although the optimization convergence rate remains unchanged after applying DA methods, the convergence rate of the learning bound deteriorates. This observation is interesting because it suggests that even when the bias term is suficiently small, the above bound is looser than that in Theorem 9, resulting in a more unstable learning gap, i.e., $\varepsilon \ \leq$ $\varepsilon _ { D A }$ . In other words, the use of DA methods may render learning consistency uncertain. Nevertheless, the learning gap can still be estimated based on the noise rate. We then state the following result for the application of $T ^ { \prime }$

Corollary 12 Given the training dataset D with η-noisy labels, the DA kernel $T ^ { \prime } ( \cdot | ( x , \tilde { y } ) )$ , the corresponding risk function definitions, and the KL divergence defined in Definition 2, assume that $\ell \in [ 0 , M ]$ and that the cost function c is convex and satisfies Definition 1. $T h e n ,$ for any $h \in \mathcal H$ and any $\delta \in ( 0 , 1 )$ , the following inequality holds with probability at least $1 - \delta$

$$
\begin{array} { r l } & { \hat { R } _ { D A } ^ { \prime } ( h ) - R ( h ) \leq M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) } + 4 M \sqrt { \displaystyle \frac { \ln ( 3 / \delta ) } { 2 n } } } \\ & { \qquad + 2 \eta M \sqrt { \displaystyle \frac { \ln ( 3 / \delta ) } { 2 n _ { \eta } } } + 2 ( 1 - \eta ) M \sqrt { \displaystyle \frac { \ln ( 3 / \delta ) } { 2 n _ { c } } } . } \end{array}
$$

Here, $n = n _ { \eta } + n _ { c }$ represents the number of data samples in $D$ .

Remarks: (a) The main order of this upper bound is still $\mathcal { O } ( 1 / \sqrt { n } + \operatorname* { m i n } \{ 1 / \sqrt { n _ { \eta } } , 1 / \sqrt { n _ { c } } \} )$ (b) There is a bias term $\sqrt { K L }$ , which may lead to a loose or even vacuous bound, even when $n \to \infty$ . (c) If $\eta = 0$ , the bias term still exists, and therefore continues to afect the tightness of the bound.

Similarly, we have $\varepsilon \le \varepsilon _ { D A } ^ { \prime }$ , and the learning gap can still be estimated. The only diference, compared with the bound in Theorem 11, is the $\sqrt { K L }$ term. Therefore, we omit further discussion of this bound. Based on the above results, we present the following bound for estimating η.

Corollary 13 (Noise Bound with DA) Based on Theorem 11 or Corollary 12, for any $\varepsilon > 0$ , we derive a noise bound:

$$
\frac { \varepsilon - ( B + C + \Delta ) } { A - B } \leq \eta \leq \frac { ( B + C + \Delta ) - \varepsilon } { B - A } ,\tag{4}
$$

where $A = \sqrt { ( 2 M ) ^ { 2 } \ln ( 1 / \delta ) / 2 n _ { \eta } } , B = \sqrt { ( 2 M ) ^ { 2 } \ln ( 1 / \delta ) / 2 n _ { c l } }$ , and $C = \sqrt { ( 4 M ) ^ { 2 } \ln ( 1 / \delta ) / 2 n }$ $\Delta = \sigma \Delta _ { \parallel x ^ { \prime } - x \parallel } \ o r \ \Delta = M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) }$

Remarks: (a) If $A \gg B ,$ , i.e., $n _ { c } \to \infty , n \to \infty$ , then $\eta \approx ( \varepsilon - \Delta ) / A$ . (b) if $A \ll B ,$ , i.e., $n _ { \eta }  \infty , n  \infty$ , then $\eta \approx ( B + \Delta - \varepsilon ) / B$

Here, we observe that the suficient and necessary relationship between the efect of noisy labels on learning consistency and the noise rate no longer holds. This breakdown is induced by $\Delta .$ . We then provide the following figure to illustrate the results in (3) and (4), where $\eta _ { D A }$ denotes the quantity defined in (4). First, we assume that $\varepsilon \ge ( B + C + \Delta )$ ; then, we can still estimate the interval for the noise rate as follows.

![](images/3133fd66a80391502aea5d8e238643c8d3fe6e8904f04b6f08a5d14cdc9e764a.jpg)

This indicates that DA methods also expand the estimation range of $\eta ,$ i.e., the bound for noise becomes looser. Under this assumption, the suficient and necessary relationship stil holds. However, when this assumption does not hold, we have the following result, where the estimation of $\eta$ is totally meaningless. Correspondingly, the suficient and necessary relationship breaks.

![](images/9a47a0e582ac8601f7b69d17bff9539552db5b96589cb5dd29161c7043f33c44.jpg)

In conclusion, applying DA methods not only decreases the noise tolerance for SGD optimization but also makes the learning consistency uncertain. Notes that the above learning bounds are Hoefding-type bounds. If one employs Bernstein-type inequalities (Bernstein, 1946) to obtain tighter bounds, or uses VC-dimension-based bounds (Vapnik and Chervonenkis, 2015) to analyze the classification case, similar results can be derived; the proof is left to the reader.

The above analysis results finally provide the insight that the noise rate η may difer from the noise rate after applying DA methods, i.e., $\eta \neq \eta _ { D A }$ . A natural question is whether $\eta \le \eta _ { D A } ~ \mathrm { o r } ~ \eta > \eta _ { D A }$ , and under what conditions each case holds. In the following, we investigate these questions.

In the beginning of this subsection, we introduce a strong assumption, i.e., Assumption 4. Based on this assumption, the analysis appears to be limited to symmetric label noise, i.e., $P ( \tilde { y } \mid y )$ , thereby excluding scenarios with semantic noise, i.e., instance-dependent noise $P ( \tilde { y } \mid x , y )$ . In the subsequent analysis, we no longer rely on this assumption. Here, we first explain why we no longer rely on this assumption. The analysis derived under this assumption is applicable only under limited conditions, which restricts its applicability to more general cases. As a result, the corresponding conclusions provide limited insight into real-world learning problems. Nevertheless, even under such a strong assumption, we can still conduct a further investigation of the above question. Interested readers may find it worthwhile to explore this direction over a cup of cofee.

## 3.3 Analysis

In this subsection, we present the formal analysis without relying on the strong assumption introduced in the previous subsection. Instead, we consider several standard settings that are commonly satisfied in real-world learning scenarios.

We firstly define the following risk functions: the relative expected risk for any $h _ { 1 } , h _ { 2 } \in$ H,

$$
\gamma ( h _ { 1 } , h _ { 2 } ) = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) } [ \ell ( h _ { 1 } , ( \boldsymbol { x } , \boldsymbol { y } ) ) - \ell ( h _ { 2 } , ( \boldsymbol { x } , \boldsymbol { y } ) ) ] ,
$$

and the relative empirical risk for any $h _ { 1 } , h _ { 2 } \in \mathcal { H }$

$$
\hat { \gamma } ( h _ { 1 } , h _ { 2 } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } [ \ell ( h _ { 1 } , ( x _ { i } , y _ { i } ) ) - \ell ( h _ { 2 } , ( x _ { i } , y _ { i } ) ) ] .
$$

Similarly, under diferent data conditions, we define $\gamma ^ { c } , \ \gamma _ { D A } ,$ , and $\gamma _ { D A } ^ { \prime }$ , consistent with the previous settings. Note that we do not introduce the absolute value operation in the substitution for the two losses, so the resulting relative risks may take negative values and must be accounted for accordingly. We then introduce a measure of the discrepancy between any two hypotheses $h , h ^ { \prime } \in \mathcal { H }$ as follows.

Definition 14 Let H be a hypothesis space consisting of measurable functions $h : \mathcal { X } \to \mathcal { Y }$ where X and Y are measurable spaces, and µ be a probability measure on X. Assume $h , h ^ { \prime } \in \mathcal { H }$ are µ-integrable. Define a distance between any two hypotheses $h , h ^ { \prime } \in \mathcal { H }$ by the $\mathbb { L } _ { 1 } ( \mu )  – n o r m \colon$

$$
d _ { \mu } ( h , h ^ { \prime } ) : = \left\| h - h ^ { \prime } \right\| _ { \mathbb { L } _ { 1 } ( \mu ) } = \mathbb { E } _ { x \sim \mu } [ \left| h ( x ) - h ^ { \prime } ( x ) \right| ] .
$$

We consider that this distance is uniformly bounded, $i . e .$ ,

$$
0 \leq d _ { \mu } ( h , h ^ { \prime } ) \leq \beta , \quad \forall h , h ^ { \prime } \in \mathcal { H } ,
$$

for some finite constant $\beta > 0$

Remarks: (a) By definition of the distance, $d _ { \mu } ( h , h ^ { \prime } ) = 0$ if and only if $h ( x ) = h ^ { \prime } ( x ) \ \mu \cdot$ almost everywhere on X. (b) Note that we consider the distance is upper bounded only to exclude the case of infinity. (c) Hereafter, we denote $d _ { x }$ by $d _ { \mu } .$ , as this notation preserves the dependence on the underlying measure µ without ambiguity.

We define $h ^ { \ast } \in a r g m i n _ { h \in \mathcal { H } } R ( h )$ and $h ^ { \ast c } \in a r g m i n _ { h \in \mathcal { H } } R ^ { c } ( h ) . \ h _ { D A } ^ { \ast } \in a r g m i n _ { h \in \mathcal { H } } R _ { D A } ( h )$ and $\begin{array} { r } { h _ { D A } ^ { * c } \ \in \ a r g m i n _ { h \in \mathcal { H } } \mathbb { E } _ { ( x , y ) } \mathbb { E } _ { x ^ { \prime } \sim T ( \cdot \vert x ) } [ \ell ( h , ( x ^ { \prime } , y ) ) ] } \end{array}$ , or, $h _ { D A } ^ { * } ~ \in$ argmi $n _ { h \in \mathcal { H } } R _ { D A } ^ { \prime } ( h )$ and $h _ { D A } ^ { * c } \in a r g m i n _ { h \in \mathcal { H } } \mathbb { E } _ { ( x , y ) } \mathbb { E } _ { ( x ^ { \prime } , y ^ { \prime } ) \sim T ( \cdot | x , y ) } [ \ell ( h , ( x ^ { \prime } , y ^ { \prime } ) ) ]$ . Here, we treat $h _ { D A } ^ { * c }$ as a reference to $h ^ { * c }$ . Here, we assume that applying the same $T$ or $T ^ { \prime }$ to $D ^ { c }$ yields an optimal hypothesis that difers from $h ^ { * c }$ , since $T$ or $T ^ { \prime }$ is not 1-1. Now, based on the definition of optimal hypotheses, we introduce the following notations:

$$
\begin{array} { c } { \varepsilon _ { \eta } : = d _ { x } ( h ^ { \ast } , h ^ { \ast c } ) , } \\ { \varepsilon _ { \eta _ { D A } } : = d _ { x ^ { \prime } } ( h _ { D A } ^ { \ast } , h _ { D A } ^ { \ast c } ) } \end{array}
$$

These notations indicate that the noise errors are induced by noisy labels and can represent all kinds of noisy conditions. Note that these noise errors are proportional to the noise rates rather than equal to them, i.e., $\varepsilon _ { \eta } \propto \eta$ . Consequently, larger noise errors correspond to higher noise rates, and vice versa. Here, the validity of the above definition is independent of whether $x \sim P$ or $x \sim P ^ { c }$ , and whether $x ^ { \prime } \sim P _ { D A }$ or $x ^ { \prime } \sim P _ { D A } ^ { c }$

Now, having introduced all the necessary notations, we present the following theorem.

Theorem 15 Given any training set, the definition of the relative risks, and assuming $\ell \in [ 0 , M ]$ , for any $h _ { 1 } , h _ { 2 } \in \mathcal { H }$ , the following inequality holds with probability at least $1 - \delta .$

$$
\hat { \gamma } ( h _ { 1 } , h _ { 2 } ) - \gamma ( h _ { 1 } , h _ { 2 } ) \leq 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 n } } .
$$

Remarks: (a) The main order of this bound is $\mathcal { O } ( 1 / \sqrt { n } )$ . (b) Here, $h _ { 1 }$ and $h _ { 2 }$ can be any hypotheses in H. For the non-trivial case, we let $h _ { 1 }$ and $h _ { 2 }$ be diferent. If $h _ { 1 }$ is equal to $h _ { 2 }$ , this leads to a trivial case, as the RHS is always non-negative.

The above bound is also a Hoefding-type bound, and its tightness is guaranteed as $n  \infty$ . This indicates that the relationship between empirical and expected risks can still be analyzed via high-probability bounds. Based on this observation, we then consider replacing $h _ { 1 }$ and $h _ { 2 }$ with specific hypotheses and derive the following corollary.

Corollary 16 Based on Theorem 15, considering the training set D with η-noisy labels and the DA kernel $T ,$ for any $h \in \mathcal H$ , we have the following inequalities:

$$
\hat { \gamma } ( h , h ^ { * } ) - \gamma ( h , h ^ { * c } ) \leq \sigma \varepsilon _ { \eta } + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 n } } ,
$$

and

$$
\hat { \gamma } _ { D A } ( h , h _ { D A } ^ { * } ) - \gamma _ { D A } ( h , h _ { D A } ^ { * c } ) \leq \sigma \varepsilon _ { \eta _ { D A } } + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 n } } .
$$

Here, $h ^ { \ast c } , \ h _ { D A } ^ { \ast c } , \ \varepsilon _ { \eta }$ , and $\varepsilon _ { \eta _ { D A } }$ are defined above. When considering $T ^ { \prime } , \hat { \gamma } _ { D A } ^ { \prime } ( h , h _ { D A } ^ { * } ) \textrm { -- }$ $\gamma _ { D A } ^ { \prime } ( h , h _ { D A } ^ { * c } )$ yields the same result as that obtained for T.

Remarks: (a) The main orders of the above bounds are ${ \mathcal { O } } ( 1 / { \sqrt { n } } )$ . (b) There are bias terms on the RHS of the above bounds, namely $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ , which may lead to loose or vacuous bounds even as $n \to \infty$

The above corollary provides an insight suggesting that, for any learned h under $D ,$ with or without DA methods, the relative risk gaps are bounded by the noise errors. Here, we consider the gap between $\hat { \gamma } ( h , h ^ { * } )$ and $\gamma ( h , h ^ { * c } )$ in order to examine the efect of noisy labels. This setting is consistent with Theorem 9, and is derived under milder assumptions. Through this corollary, we also observe that the only diference between these two bounds lies in $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ . From the previous analysis, we have shown that under a strong assumption, they are diferent. However, once this assumption is no longer imposed, it is unclear whether such a discrepancy still persists. Once this discrepancy is identified and the relative magnitudes of $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ are determined, we can explicitly characterize the tightness of the above two bounds, identify which bound is tighter and which is looser, and consequently draw conclusions regarding the efect of DA methods on the relative risk gap. In simple terms, when $\varepsilon _ { \eta _ { D A } } \geq \varepsilon _ { \eta }$ , the risk gap increases, implying that DA is not beneficial for learning, and vice versa. Based on the classical triangle inequality, we then establish the following result.

Theorem 17 By the definitions of $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ , the inequality follows:

$$
| \varepsilon _ { \eta } - \varepsilon _ { \eta _ { D A } } | \leq d _ { x , x ^ { \prime } } ( h ^ { * } , h _ { D A } ^ { * } ) + d _ { x , x ^ { \prime } } ( h ^ { * c } , h _ { D A } ^ { * c } )\tag{5}
$$

Here, $d _ { x , x ^ { \prime } } ( h ^ { * } , h _ { D A } ^ { * } ) : = \mathbb { E } _ { x , x ^ { \prime } } [ | h ^ { * } ( x ) - h _ { D A } ^ { * } ( x ^ { \prime } ) | ]$ and $d _ { x , x ^ { \prime } } ( h ^ { * c } , h _ { D A } ^ { * c } ) \ : = \ \mathbb { E } _ { x , x ^ { \prime } } [ | h ^ { * c } ( x ) \ -$ $h _ { D A } ^ { * c } ( x ^ { \prime } ) | ]$

Remarks: $\mathrm { ( a ) } \varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ must be diferent unless $d _ { x , x ^ { \prime } } ( h ^ { \ast } , h _ { D A } ^ { \ast } )  0$ and $d _ { x , x ^ { \prime } } ( h ^ { * c } , h _ { D A } ^ { * c } ) $ 0. However, this special case cannot occur because neither T nor $T ^ { \prime }$ is 1-1. (b) A trivial case is $\varepsilon _ { \eta } = \varepsilon _ { \eta _ { D A } } = 0$

The above result establishes that, provided T and $T ^ { \prime }$ are not 1-1 transformations, $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ are generally diferent. However, it does not characterize their ordering. To determine whether $\varepsilon _ { \eta } \ge \varepsilon _ { \eta _ { D A } } ~ \mathrm { o r } ~ \varepsilon _ { \eta _ { D A } } \ge \varepsilon _ { \eta }$ , we introduce the following conditions. First, a suficient condition under which $\varepsilon _ { \eta _ { D A } } \geq \varepsilon _ { \eta }$ holds is given as follows.

Theorem 18 (Trivial Suficient Condition) Let $\epsilon = \mathbb { E } _ { x ^ { \prime } , \tilde { y } } [ \ell ( h _ { D A } ^ { \ast } , ( x ^ { \prime } , \tilde { y } ) ) ]$ be a suficiently small and $d _ { x } ( h ^ { \ast } , h ^ { \ast c } ) \leq \eta _ { m a x }$ . Then, we have $\varepsilon _ { \eta _ { D A } } \geq \varepsilon _ { \eta }$ if the following condition holds:

$$
\operatorname* { m a x } \left\{ \alpha \Delta _ { \parallel \tilde { y } - y \parallel _ { D A } } , \frac { \Delta _ { \parallel x ^ { \prime } - x \parallel } } { r ( x ) } , \frac { 1 } { 2 } \left( \alpha \Delta _ { \parallel \tilde { y } - y \parallel _ { D A } } + \frac { \Delta _ { \parallel x ^ { \prime } - x \parallel } } { r ( x ) } \right) \right\} \geq \frac { \sigma \eta _ { m a x } + \epsilon } { M }\tag{6}
$$

Here, $\alpha > 0$ is a constant, $\tilde { y }$ can be replaced by $\tilde { y } ^ { \prime } { } _ { i }$ , if considering $T ^ { \prime } . \ r ( x )$ denotes a margin condition such that $h ( x ^ { \prime } ) \neq h ( x ) = y$ whenever $| x ^ { \prime } - x | > r ( x )$ for any $h _ { . }$ , which characterizes the failure condition of prediction.

Remarks: (a) $\Delta _ { \parallel \tilde { y } - y \parallel _ { D A } }$ represents the label bias in D after applying T. (b) $\Delta _ { \parallel x ^ { \prime } - x \parallel } / r ( x )$ represents the ratio of data bias in D after applying $T$ to $r ( x ) . \quad \mathrm { ( c ) }$ More rigorously, $d _ { x } ( h ^ { \ast } , h ^ { \ast c } ) \leq \alpha \eta _ { m a x } ,$ as $d _ { x } ( h ^ { \ast } , h ^ { \ast c } ) \propto P [ h ^ { \ast } ( x ) \neq h ^ { \ast c } ( x ) ]$ . (d) ϵ is a small constant due to $h _ { D A } ^ { * } \in a r g m i n R _ { D A } ( h )$

Here, this condition is regarded as trivial, as it follows immediately from $\varepsilon _ { \eta _ { D A } } \geq \eta _ { m a x } \geq$ $\varepsilon _ { \eta } .$ This implies a suficient condition under which the event $\varepsilon _ { \eta _ { D A } } \geq \eta _ { m a x }$ occurs. Although this condition is rather trivial, it leads to several interesting observations. (a) If $\Delta _ { \parallel \tilde { y } - y \parallel _ { D A } } \geq$ RHS, it implies that the DA method amplifies the spread of noisy labels, leading to $\varepsilon _ { \eta _ { D A } } \geq$ $\varepsilon _ { \eta } .$ . For synthetic or semi-synthetic datasets, this quantity can be computed directly. Indeed, in such settings, both $\eta$ and $\eta _ { D A }$ are available and can be explicitly estimated. However, in real-world applications, these quantities are generally inaccessible, making this condition dificult to verify in practice. Therefore, we do not further pursue this condition, as well as the mixture condition. (b) One condition that remains computable during training, even in real-world applications, is $\Delta _ { \parallel x ^ { \prime } - x \parallel } / r ( x )$ . Here, $r ( x )$ denotes the prediction margin of $h ,$ , which is typically treated as a constant and can be estimated empirically through repeated experiments. If $\Delta _ { \parallel x ^ { \prime } - x \parallel } / r ( x ) \geq R H S$ , the condition directly indicates that the noisy labels are propagated by the DA methods. Moreover, it suggests avoiding the inclusion of min $| | h ( x ^ { \prime } ) - h ( x ) |$ during the training, as

$$
\operatorname* { l i m } _ { r ( x ) \to 0 } { \frac { \Delta _ { \| x ^ { \prime } - x \| } } { r ( x ) } } = \infty .
$$

On the contrary, one could consider the chain $\varepsilon _ { \eta } \geq \eta _ { m a x } \geq \varepsilon _ { \eta _ { D A } }$ to derive a condition. However, this is unnecessary, since in machine learning settings we do not typically use original data whose label noise exceeds the maximum threshold. Moreover, when this condition does not hold, the relative magnitudes of $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ remain indeterminate. Therefore, we instead state the following non-trivial condition.

Theorem 19 (Non-Trivial Suficient Condition) Assume that $h ( x ) ~ \in ~ [ 0 , v ] , \forall x ~ \in$ $\mathcal { X } , \forall h \in \mathcal { H }$ . Then, for any $h \in \mathcal H$ 2

$$
d _ { x } ( h , h ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) \geq \mathcal { K } \Rightarrow \varepsilon _ { \eta } \geq \varepsilon _ { \eta _ { D A } } ,\tag{7}
$$

$$
d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x } ( h , h ^ { * } ) \geq \mathcal { K } \Rightarrow \varepsilon _ { \eta _ { D A } } \geq \varepsilon _ { \eta } .\tag{8}
$$

Here, K = E<sub>x</sub>h(x) + E<sub>x</sub>′h(x<sup>′</sup>) + 2υ.

Remarks: (a) The boundedness assumption on h is mild in classification settings, whereas it is stronger in regression settings. When extending to regression, it is typically necessary to post-process the outputs of h, for example via a sigmoid transformation. Further details will be provided later. (b) Note that $d _ { x } ( h , h ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) \leq K \neq d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x } ( h , h ^ { * } ) \geq K$ and $d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x } ( h , h ^ { * } ) \leq K \neq d _ { x } ( h , h ^ { * } ) - d _ { x ^ { \prime } } ( \widetilde { h } , h _ { D A } ^ { * } ) \geq { \mathscr K } . { \mathrm { ~ ( c ) } }$ The above conditions are relatively loose; further discussion is deferred to Section 5.

Based on the above, it is possible to estimate whether label noise is propagated by DA methods during training or after evaluation. We therefore introduce the following approximation methods.

Let $h ^ { * } ( x ) = \tilde { y }$ and $h _ { D A } ^ { * } ( x ^ { \prime } ) = \tilde { y }$ , or $h _ { D A } ^ { * } ( x ^ { \prime } ) = \tilde { y } ^ { \prime }$ when considering $T ^ { \prime }$ . Then, letting $\hat { h } \in a r g m i n \hat { \phi } ( h )$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { \tilde { y } } } [ \ell ( \hat { h } , ( \boldsymbol { x } , \boldsymbol { \tilde { y } } ) ) ] - \mathbb { E } _ { \boldsymbol { x ^ { \prime } } , \boldsymbol { \tilde { y } } } [ \ell ( \hat { h } , ( \boldsymbol { x ^ { \prime } } , \boldsymbol { \tilde { y } } ) ) ] \geq \mathcal { K } , } \\ & { \mathbb { E } _ { \boldsymbol { x ^ { \prime } } , \boldsymbol { \tilde { y } } } [ \ell ( \hat { h } , ( \boldsymbol { x ^ { \prime } } , \boldsymbol { \tilde { y } } ) ) ] - \mathbb { E } _ { \boldsymbol { x } , \boldsymbol { \tilde { y } } } [ \ell ( \hat { h } , ( \boldsymbol { x } , \boldsymbol { \tilde { y } } ) ) ] \geq \mathcal { K } , } \end{array}
$$

where $\mathcal { K } = \mathbb { E } _ { x } \hat { h } ( x ) + \mathbb { E } _ { x ^ { \prime } } \hat { h } ( x ^ { \prime } ) + 2 v$ . Here, $\hat { \phi } ( h )$ denotes either $\hat { R } ( h ) , \hat { R } _ { D A } ( h )$ , or $\hat { R } _ { D A } ^ { \prime } ( h )$ . In practice, the expectation $\mathbb { E } [ \cdot ]$ can be approximated by the corresponding empirical average.

Note that both the above conditions and the trivial condition have limitations. When neither condition holds, the relative magnitudes of $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ remain unclear. This issue will be further discussed in Section 5.

Finally, we provide the following additional analyses before concluding this section.

Theorem 20 Given the training set D with η-noisy labels, the DA kernel T or $T ^ { \prime }$ , the corresponding risk function definitions, and the $W _ { 1 }$ distance and the KL divergence. Assume that $\ell \in [ 0 , M ]$ that is σ-admissible. Then for any $h \in \mathcal H$ , the discrepancy between $R ( h )$ and $R _ { D A } ( h )$ satisfies, with probability at least $1 - \delta$

$$
R ( h ) - R _ { D A } ( h ) \leq 2 \sigma \Delta _ { \| \tilde { y } - y \| } + \sigma \Delta _ { \| x ^ { \prime } - x \| } + C M \sqrt { \frac { \ln ( 7 / \delta ) } { 2 n } } ,
$$

or the discrepancy between $R ( h )$ and $R _ { D A } ^ { \prime } ( h )$ satisfies:

$$
R ( h ) - R _ { D A } ^ { \prime } ( h ) \leq 2 \sigma \Delta _ { \| \tilde { y } - y \| } + M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) } + C M \sqrt { \frac { \ln ( 7 / \delta ) } { 2 n } } .
$$

Here, $C > 0$ is a constant, $\Delta _ { \parallel x ^ { \prime } - x \parallel } = \mathbb { E } _ { x } \mathbb { E } _ { x ^ { \prime } } \lVert x ^ { \prime } - x \rVert$ , and $\Delta _ { \parallel \tilde { y } - y \parallel } = \mathbb { E } _ { y } \mathbb { E } _ { \tilde { y } } \lVert \tilde { y } - y \rVert$

Remarks: (a) The main order of the above bounds is $\mathcal { O } ( 1 / \sqrt { n } )$ . (b) There are two bias terms in each bound, which may harm the tightness of the bounds.

The above bounds characterize the diference between $R ( h )$ and $R _ { D A } ( h )$ for any $h \in \mathcal H$ We observe that, even when $\Delta _ { \parallel x ^ { \prime } - x \parallel } \to 0$ , the bound remains loose in the presence of label bias. Moreover, since $\Delta _ { \parallel \tilde { y } - y \parallel }$ is unknown in real-world settings, estimating the gap between $R ( h )$ and $R _ { D A } ( h )$ is challenging. If $\Delta _ { \parallel \tilde { y } - y \parallel } = 0$ , i.e., there is no label noise, then as $\Delta _ { | x ^ { \prime } - x | }  0$ , we have that $R ( h )$ and $R _ { D A } ( h )$ become asymptotically equivalent. Based on this observation, we derive the following corollary.

Corollary 21 Under the settings of Theorem 20 and the definitions of $h ^ { * }$ and $h _ { D A } ^ { * }$ , the following inequality holds with probability at least $1 - 2 \delta$

$$
R ( h ^ { * } ) - R _ { D A } ( h _ { D A } ^ { * } ) \leq \sigma d _ { x ^ { \prime } , x } ( h ^ { * } , h _ { D A } ^ { * } ) + 2 \left\{ 2 \sigma \Delta _ { \| \tilde { y } - y \| } + \sigma \Delta _ { \| x ^ { \prime } - x \| } + C M \sqrt { \frac { \ln ( 7 / \delta ) } { 2 n } } \right\} ,
$$

or

$$
R ( h ^ { * } ) - R _ { D A } ^ { \prime } ( h _ { D A } ^ { * } ) \leq \sigma d _ { x ^ { \prime } , x } + 2 \left\{ 2 \sigma \Delta _ { \| \tilde { y } - y \| } + M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) } + C M \sqrt { \frac { \ln ( 7 / \delta ) } { 2 n } } \right\} .
$$

$$
H e r e , d _ { x ^ { \prime } , x } ( h ^ { \ast } , h _ { D A } ^ { \ast } ) : = \mathbb { E } _ { x ^ { \prime } , x } [ | h ^ { \ast } ( x ^ { \prime } ) - h _ { D A } ^ { \ast } ( x ) | ] .
$$

Remarks: (a) The main order of the above bounds is still ${ \mathcal { O } } ( 1 / { \sqrt { n } } )$ . (b) An additional bias term $d _ { x ^ { \prime } , x }$ is introduced in the above bounds.

The above results indicate the discrepancy between $R ( h ^ { * } )$ and $R ( h _ { D A } ^ { * } )$ . Note that these bounds do not indicate which risk is larger; rather, they only quantify the gap between the two risks. These bounds reveal that multiple factors can contribute to enlarging this gap. Through these bounds, we can draw another conclusion: using DA methods further increases the uncertainty of uncertain learning procedures. How can this conclusion be understood? Machine learning procedures are essentially unstable, and introducing DA significantly increases this uncertainty. We argue that although DA may bring improvements in evaluation performance (see the experiments by Pereira et al. (2022)), the risk introduced by DA itself is larger than its benefits.

## 4 A Domain Generalization Perspective

## 4.1 Setup

Basic Data Assumption. We clarify the basic data structure for subsequent analysis. Under the domain generalization (DG) setting, let X and Y be the input and output spaces, respectively. The clean dataset without labeling noise is denoted as

$$
\mathcal { E } ^ { c } : = \{ D _ { 1 } ^ { c } , \ldots , D _ { E } ^ { c } \} = \{ \{ ( x _ { i } , y _ { i } ) _ { 1 } \} _ { i = 1 } ^ { n _ { 1 } } , \ldots , \{ ( x _ { i } , y _ { i } ) _ { E } \} _ { i = 1 } ^ { n _ { E } } \} .
$$

Here, each pair $( x ^ { e } , y ^ { e } ) \in \mathcal { X } \times \mathcal { Y }$ , under any environment $e \in [ E ]$ , is independently and identically drawn from an unknown joint distribution $P ( x ^ { e } , y ^ { e } ) , { \mathrm { i . e . , } } ( x ^ { e } , y ^ { e } ) \sim _ { i . i . d . } P ( x ^ { e } , y ^ { e } )$ where $P ( x ^ { e } , y ^ { e } )$ is denoted by $P _ { e }$ for simplicity. In the DG setting, we assume that $P _ { e } \neq$ $P _ { e ^ { \prime } } , \forall e \neq e ^ { \prime } \in [ E ]$ . Thus, the data are independently and non-identically distributed $( \mathrm { i . n . d . } )$ The dataset $\mathcal { E } ^ { c }$ with size $E \sum _ { i } ^ { E } n _ { i }$ is used as the training set for learning models, i.e., the source set. Correspondingly, a test dataset $\mathcal { E } _ { t } ^ { c }$ , the target set, is assumed to be drawn from related yet diferent distributions from $\mathcal E ^ { c } { } _ { : }$ , i.e., $P _ { e } \neq P _ { e _ { t } } , \forall e \in [ E ] , \forall e _ { t } \in [ E _ { t } ]$ , where $E _ { t }$ denotes the number of test domains or environments in $\mathcal { E } _ { t } ^ { c }$ . Moreover, no information about the test set is available during training.

Typically, we consider the distribution set for $\mathcal { E } ^ { c }$ , denoted by $\mathbf { P } ^ { c } = \{ P _ { 1 } ^ { c } , \ldots , P _ { E } ^ { c } \}$ , and the distribution set for $\mathcal { E } _ { t } ^ { c }$ , denoted by $\mathbf { P _ { t } } ^ { c } = \{ P _ { 1 } { } ^ { c } , \dots , P _ { E _ { t } } { } ^ { c } \}$ . Most importantly, the source and target distribution sets satisfy $\mathbf { P } ^ { c } \cap \mathbf { P _ { t } } ^ { c } = \emptyset$ , with both $\mathbf { P } ^ { c } \subset \mathcal { P } ^ { c }$ and $\mathbf { P _ { t } } ^ { c } \subset \mathcal { P } ^ { c }$

Then, according to the above, the noisy-label source dataset is denoted by

$$
\mathcal E : = \{ D _ { 1 } , \dotsc , D _ { E } \} = \{ \{ ( x _ { i } , \tilde { y } _ { i } ) _ { 1 } \} _ { i = 1 } ^ { n _ { 1 } } , \dotsc , \{ ( x _ { i } , \tilde { y } _ { i } ) _ { E } \} _ { i = 1 } ^ { n _ { E } } \} .
$$

Similarly, $\mathcal { E } _ { t }$ denotes the noisy-label target dataset, while P and $\mathbf { P _ { t } }$ denote the corresponding distribution sets. Most importantly, $\mathbf { P } \cap \mathbf { P _ { t } } = \emptyset$ , with both $\mathbf { P } \subset \mathcal { P }$ and $\mathbf { P _ { t } } \subset \mathcal { P }$

Data Augmentation. The DA kernel remains the same as that in the previous section, namely $T ( \cdot | x )$ . Since the analysis for the kernel $T ^ { \prime }$ is analogous to that for $T ,$ with the only diference being that the term $\sigma \Delta$ replaces $M \sqrt { K L }$ , we restrict our attention to $T$ in this section. Then, we consider the source distribution sets $\mathbf { P } _ { D A }$ , satisfying the same properties as the above distribution sets. Note that DA is not applied to the target domains. Therefore, the corresponding distribution set for the target domains remains $\mathbf { P _ { t } }$

Others. The loss function remains $\ell ,$ which is assumed to be bounded by M and $\sigma -$ admissible. In the previous section, we considered the KL divergence to measure the diference between two distributions under the assumption of absolute continuity. However, in DG settings, this assumption does not necessarily hold. Therefore, we consider alternative measures as follows.

Definition 22 (Jensen-Shannon (JS) Divergence) Let P and Q be two probability distribution, and let $M = ( P { + } Q ) / 2$ be their mixture distribution. The JS divergence is defined as

$$
J S ( P | | Q ) = \frac { 1 } { 2 } K L ( P | | M ) + \frac { 1 } { 2 } K L ( Q | | M ) .
$$

Additionally, the JS distance is defined as the square root of the JS divergence:

$$
d _ { J S } ( P , Q ) = \sqrt { J S ( P | | Q ) } ,
$$

which is a true metric on the space of probability distributions.

Remarks: (a) $d _ { J S } ( P , Q ) \geq 0 ; d _ { J S } ( P , Q ) = 0 \ i f f \ P = Q . \ ( \mathrm { b } ) \ d _ { J S } ( P , Q ) = d _ { J S } ( Q , P )$ . (c) $d _ { J S } ( P , Q ) \leq d _ { J S } ( P , M ) + d _ { J S } ( M , Q )$ for any distribution M. $( \mathrm { d } ) \ d _ { J S } ( P , Q ) \leq { \sqrt { \ln 2 } }$ when the JS divergence is defined using the natural logarithm.

Moreover, we introduce a generalized JS divergence, since DG typically involves multiple domains.

Definition 23 (Generalized JS Divergence) Let $\{ P _ { 1 } , \ldots , P _ { k } \}$ be probability distributions (discrete or continuous), and let $\pi = ( \pi _ { 1 } , \cdots , \pi _ { k } )$ be the set of weights such that $\pi _ { i } > 0$ and $\textstyle \sum _ { i = 1 } ^ { k } \pi = 1$ . The generalized JS divergence is defined as

$$
J S D _ { \pi } ( P _ { 1 } , \ldots , P _ { k } ) = \sum _ { i = 1 } ^ { k } \pi _ { i } K L ( P _ { i } | | { \cal M } ) ,
$$

where M is the weighted mixture distribution: $\begin{array} { r } { M = \sum _ { i = 1 } ^ { k } { \pi } _ { i } P _ { i } } \end{array}$

Remarks: (a) We typically consider uniform weights $\pi _ { i } = 1 / k$ , in which case $J S D ( P _ { 1 } , \ldots , P _ { k } ) =$ $( 1 / k ) \sum K L ( P _ { i } | | M )$ . (b) Definition 22 corresponds to the special case $k = 2$

## 4.2 Analysis

We denote the expected risk and empirical risk by $\mathcal { R } ( h ) ~ = ~ \mathbb { E } _ { P \sim \mathbf { P } } R _ { P } ( h )$ and $\hat { \mathcal { R } } ( h ) \mathop { = }$ $( 1 / E ) \textstyle \sum _ { s } ^ { E } \hat { R } _ { s } ( h )$ , respectively. We then consider the following relative risk functions: the relative expected risk for any $h _ { 1 } , h _ { 2 } \in \mathcal { H }$

$$
\Gamma ( h _ { 1 } , h _ { 2 } ) = \mathbb { E } _ { P \sim \mathbf { P } } [ \gamma _ { P } ( h _ { 1 } , h _ { 2 } ) ] = \frac { 1 } { E } \sum _ { s = 1 } ^ { E } \gamma _ { P s } ( h _ { 1 } , h _ { 2 } ) ,
$$

and the relative empirical risk for any $h _ { 1 } , h _ { 2 } \in \mathcal { H }$

$$
\hat { \Gamma } ( h _ { 1 } , h _ { 2 } ) = \frac { 1 } { E \bar { n } } \sum _ { s = 1 } ^ { E } \hat { \gamma } _ { s } ( h _ { 1 } , h _ { 2 } ) .
$$

Similarly, under diferent data conditions, we define $\gamma ^ { c }$ and $\gamma _ { D A } .$ , consistent with the previous settings. Here, $\begin{array} { r } { \mathbb { E } _ { P \sim \mathbf { P } } [ \cdot ] \ = \ ( 1 / E ) \sum _ { s = 1 } ^ { E } [ \cdot ] } \end{array}$ , since $P _ { i } \neq P _ { j } , \forall i \neq j \in [ E ]$ . Note that defining $\hat { \gamma } ( h _ { 1 } , h _ { 2 } ) ~ = ~ ( 1 / \bar { n } ) \sum [ \cdot ]$ is convenient for the subsequent analysis without loss of generality. If this is regarded as a strong assumption, one may instead consider $\begin{array} { r } { \hat { \Gamma } ( h _ { 1 } , h _ { 2 } ) = ( 1 / E ) \sum _ { s = 1 } ^ { E } \hat { \gamma } _ { s } ( h _ { 1 } , h _ { 2 } ) } \end{array}$

In the previous section, we defined $h ^ { * } \in a r g m i n _ { h \in \mathcal { H } } R ( h ) , \ h ^ { * c } \in a r g m i n _ { h \in \mathcal { H } } R ^ { c } ( h )$ $h _ { D A } ^ { * } \in a r g m i n _ { h \in { \mathcal { H } } } R _ { D A } ( h )$ , and $h _ { D A } ^ { * c } \in$ $\boldsymbol { \Xi } \ \textit { a r g m i n } _ { h \in \mathcal { H } } \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) } \mathbb { E } _ { \boldsymbol { x } ^ { \prime } \sim T ( \cdot | \boldsymbol { x } ) } [ \ell ( h , ( \boldsymbol { x } ^ { \prime } , \boldsymbol { y } ) ) ]$ . Then, under the multi-domain setting, we assume the corresponding sets of optimal hypotheses: $\{ h _ { 1 } ^ { * } , \ldots , h _ { E } ^ { * } \} , \{ h _ { 1 } ^ { * c } , \ldots , h _ { E } ^ { * c } \} , \{ h _ { D A 1 } ^ { * } , \ldots , h _ { D A E } ^ { * } \}$ , and $\{ h _ { D A 1 } ^ { * c } , \ldots , h _ { D A E } ^ { * c } \}$ . This assumption is milder than assuming that there exists $h ^ { \ast } \in$ argmin $\mathcal { R } ( h )$ . One important reason is that the latter requires $\cap _ { s = 1 } ^ { E } a r g m i n R _ { s } ( h ) \neq \emptyset$ , whereas assuming the existence of domain-wise optimal hypotheses $\{ h _ { 1 } ^ { * } , \ldots , h _ { E } ^ { * } \}$ only requires argmin $R _ { s } ( h ) \neq \emptyset , \forall s \in [ E ]$ . In other words, the domain-wise optimal hypothesis assumption removes the cross-domain consistency requirement that all domains share a common minimizer. Consequently, it remains applicable even under substantial distribution shifts, where a shared optimal hypothesis may not exist.

Following previous studies (Massart and N´ed´elec, 2006; Wang et al., 2024), we adopt the following assumption on the distribution family introduced in (Zheng and Teng, 2026).

Assumption 24 Assume a hypothesis space $\mathcal { H } \subseteq \{ h : \mathcal { X } \to \mathcal { Y } \}$ , and let $\mathcal { P } = \{ P \in \mathcal { P } ( \mathcal { H } ) \}$ denote a family of data distributions, where $\mathcal { P } ( \mathcal { H } ) \subseteq \mathcal { P } ( \mathcal { X } \times \mathcal { Y } )$

Remarks: (a) This assumption essentially implies that the distribution family $\mathcal { P }$ is induced by functions in H, while the marginal distribution $p ( x )$ is assumed to be fixed. Consequently, each distribution $P \in \mathcal { P }$ can be identified with a hypothesis $h \in { \mathcal { H } } , \operatorname { i . e . } , P _ { h } = p ( x ) h ( y | x )$ (b) In learning theory, $\mathcal { P } ( \mathcal { H } )$ is commonly used to denote the class of data distributions under which there exists a perfect predictor that lies exactly within the hypothesis space H. In other words, for any distribution $P \in \mathcal { P } ( \mathcal { H } )$ , there exists some $h ^ { \ast } \in \mathcal { H }$ such that $h ^ { \ast } ( x )$ perfectly predicts $y .$

We further assume that any distribution set P defined in the previous subsection satisfies Assumption 24. Consequently, for example, $\mathbf { P } ^ { c } = \{ P _ { 1 } ^ { c } , \ldots , P _ { E } ^ { c } \}$ can equivalently be written as $\mathbf { P } ^ { c } = \{ P _ { h _ { E } ^ { * c } } ^ { c } , \ldots , P _ { h _ { E } ^ { * c } } ^ { c } \}$ . Here, we remind the reader that the original $P _ { 1 } ^ { c }$ denotes a joint distribution over X and Y.

Similar to the previous section, we define the following noise-induced error terms:

$$
\begin{array} { l } { { \varepsilon _ { \eta } = \displaystyle \frac { 1 } { E } \sum _ { i = 1 } ^ { E } d { _ { P _ { i } } } ( h _ { i } ^ { * } , h _ { i } ^ { * c } ) } } \\ { { \varepsilon _ { \eta _ { D A } } = \displaystyle \frac { 1 } { E } \sum _ { s = 1 } ^ { E } d { _ { P _ { D A } } } ( h _ { D A s } ^ { * } , h _ { D A s } ^ { * c } ) . } } \end{array}
$$

Note that neither $P _ { i }$ nor $P _ { D A i }$ is specified explicitly, as the above definitions hold for arbitrary domains sampled from P and $\mathbf { P } ^ { c }$ , or from $\mathbf { P } _ { D A }$ and $\mathbf { P } _ { D A } ^ { c }$ . Moreover, we do not use s to index the domains of $\varepsilon _ { \eta } ,$ since we need to account for noise-induced errors in both source and target domains. Specifically, for the source domains, we denote the corresponding noise-induced errors by $\varepsilon _ { \eta } ,$ where $P _ { i } \sim { \bf P }$ and the domain index is s. For the target domains, we denote the corresponding noise-induced errors by $\varepsilon _ { \eta _ { t } }$ , where $P _ { i } \sim \mathbf { P _ { t } }$ and the domain index is t. For $\varepsilon _ { \eta _ { D A } }$ , we only consider source domains, since DA is applied exclusively to source domains and not to target domains. Consequently, for target domains with $P _ { i } \sim \mathbf { P _ { t } }$ , the corresponding noise-induced error remains denoted by $\varepsilon _ { \eta _ { t } }$

Firstly, we present the following result in terms of relative risks, similar to the previous section.

Theorem 25 Given any multi-domain training set, the definition of the relative risks, and assuming $\ell \in [ 0 , M ]$ that is σ-admissible, for any $h _ { 1 } , h _ { 2 } \in \mathcal { H }$ , the following inequality holds with probability at least $1 - \delta$

$$
\hat { \Gamma } ( h _ { 1 } , h _ { 2 } ) - \Gamma ( h 1 , h _ { 2 } ) \leq 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 E \bar { n } } } .
$$

Remarks: (a) The main order of this bound is $\mathcal { O } ( 1 / \sqrt { E \bar { n } } )$ . (b) Here, $h _ { 1 }$ and $h _ { 2 }$ can be any hypotheses in H. For the non-trivial case, we let $h _ { 1 }$ and $h _ { 2 }$ be diferent. If $h _ { 1 }$ is equal to $h _ { 2 }$ , this leads to a trivial case, as the right-hand side (RHS) is always non-negative.

Through the above bound, we observe that the tightness depends on $E ,$ rather than solely on n as in Theorem 15. This implies that as $E  \infty$ , the bound becomes tighter, which is consistent with conclusions in recent DG studies. Based on this result, we now present the following bounds.

Corollary 26 Based on Theorem ${ \it 2 5 , }$ considering the training set $\mathcal { E }$ with η-noisy labels and the DA kernel T, for any $h \in \mathcal H$ , we have the following inequalities:

$$
\frac { 1 } { E \bar { n } } \sum _ { s = 1 } ^ { E } \hat { \gamma } _ { s } ( h , h _ { s } ^ { * } ) - \frac { 1 } { E } \sum _ { s = 1 } ^ { E } \gamma _ { s } ( h , h _ { s } ^ { * c } ) \leq \sigma \varepsilon _ { \eta } + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 E \bar { n } } } ,
$$

and

$$
\frac { 1 } { E \bar { n } } \sum _ { s = 1 } ^ { E } \hat { \gamma } _ { s } ( h , h _ { D A s } ^ { * } ) - \frac { 1 } { E } \sum _ { s = 1 } ^ { E } \gamma _ { s } ( h , h _ { D A s } ^ { * c } ) \leq \sigma \varepsilon _ { \eta _ { D A } } + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 E \bar { n } } } .
$$

Here, for any $s \in [ E ] , h ^ { * c } { } _ { s } , h _ { D A _ { s } } ^ { * c } , \varepsilon _ { \eta }$ , and $\varepsilon _ { \eta _ { D A } }$ are defined above.

Remarks: (a) The main orders of the above bounds are $\mathcal { O } ( 1 / \sqrt { E \bar { n } } )$ . (b) There are bias terms on the RHS of the above bounds, namely $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } } .$ , which may lead to loose or vacuous bounds even as $E \bar { n } \to \infty$

We also observe that, when comparing the two bounds, their diference additionally depends on the gap between $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ . Consequently, the following result holds.

Theorem 27 By the definitions of $\varepsilon _ { \eta }$ and $\varepsilon _ { \eta _ { D A } }$ , the inequality follows:

$$
\left| \varepsilon _ { \eta } - \varepsilon _ { \eta _ { D A } } \right| \leq \frac { 1 } { E } \sum _ { s = 1 } ^ { E } \left( d _ { x , x ^ { \prime } } ( h _ { s } ^ { * } , h _ { D A s } ^ { * } ) + d _ { x , x ^ { \prime } } ( h _ { s } ^ { * c } , h _ { D A s } ^ { * c } ) \right)\tag{9}
$$

Here, $d _ { x , x ^ { \prime } } ( h _ { s } ^ { * } , h _ { D A s } ^ { * } ) : = \mathbb { E } _ { x , x ^ { \prime } } [ | h _ { s } ^ { * } ( x ) - h _ { D A s } ^ { * } ( x ^ { \prime } ) | ]$ and $d _ { x , x ^ { \prime } } ( h _ { s } ^ { \ast c } , h _ { D A s } ^ { \ast c } ) : = \mathbb { E } _ { x , x ^ { \prime } } [ | h _ { s } ^ { \ast c } ( x ) - $ $h _ { D A _ { s } } ^ { * c } ( x ^ { \prime } ) \vert ]$

Remarks: (a) Despite considering multiple-domain settings, observations similar to those from Inequality (5) can still be obtained. (b) The trivial condition is stronger than the i.i.d. condition, since for any domain $s \in [ E ] , d _ { P _ { s } }$ and $d _ { P _ { D A _ { s } } }$ are required to be 0.

According to the above result, Theorems 18 and 19 can also be extended to the distribution-shift setting. The proof is analogous and is therefore omitted. We directly state the following conditions for analyzing whether the noisy labels are propagated,

$$
\begin{array} { c } { { \displaystyle \frac { 1 } { E } \sum _ { s } ^ { E } \frac { \Delta _ { \parallel x ^ { \prime } - x \parallel } } { r ( x ) } \geq \frac { \sigma \eta _ { m a x } + \epsilon } { M } , } } \\ { { \displaystyle \frac { 1 } { E } \sum _ { s } ^ { E } d _ { x ^ { \prime } } ( h , h _ { D A s } ^ { * } ) - d _ { x } ( h , h _ { s } ^ { * } ) \geq \frac { 1 } { E } \sum _ { s } ^ { E } \mathcal { K } _ { s } , } } \end{array}
$$

where $\kappa _ { s }$ represents the K (see the previous section) within each domain s. Now, based on the above results, we conclude that the risks induced by DA under i.i.d. data persist under distribution shifts.

In fact, from the perspective of DG, a primary concern is the generalization error between the source and target domains. The above analyses are conducted solely on the source domains, which limits their applicability to DG settings. Therefore, to provide a more complete understanding of DA under DG, we further analyze the source-target generalization error. Consequently, we derive the following generalization bounds.

Theorem 28 (Generalization Bound) Given the training set E and the testing set $\mathcal { E } _ { t }$ with η-noisy labels, and assuming $\ell \in [ 0 , M ]$ that is σ-admissible, the generalization gap between the relative risks on $\mathcal { E }$ and $\mathcal { E } _ { t }$ satisfies the following bound with probability at least $1 - \delta _ { i }$ , for any $h \in { \mathcal { H } } .$

$$
\frac { 1 } { E _ { t } } \sum _ { t } \gamma ( h , h _ { t } ^ { * } ) \leq \frac { 1 } { E \bar { n } } \sum _ { s } \hat { \gamma } ( h , h _ { s } ^ { * } ) + \sigma \left( \varepsilon _ { \eta t } + \varepsilon _ { \eta } \right) + C \left( \lambda ^ { * c } + \lambda \right) + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 E \bar { n } } } .
$$

Here, $\begin{array} { r } { \varepsilon _ { \eta _ { t } } = \frac { 1 } { E } \sum _ { t = 1 } ^ { E _ { t } } d _ { P _ { i } } ( h _ { t } ^ { * } , h _ { t } ^ { * c } ) , C = ( 2 \sqrt { 2 } M ) / E E _ { t } \ , \lambda = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { h _ { t } ^ { * } } , P _ { h _ { s } ^ { * } } ) } \end{array}$ , and $\lambda ^ { * c } =$ $\begin{array} { r } { \sum _ { t } \sum _ { s } d _ { J S } ( P _ { h _ { t } ^ { * c } } , P _ { h _ { s } ^ { * c } } ) } \end{array}$

Remarks: (a) The main orders of the above bound is $\mathcal { O } ( 1 / \sqrt { E \bar { n } } )$ . When all ε and λ terms are assumed to be small, the bound is tight provided that $E \bar { n }  \infty$ . (b) By considering the JS divergence (JS distance) instead of the KL divergence to measure the discrepancy between distributions, the assumption that $P _ { i } \ll P _ { j } , \forall i \neq j \in [ E ]$ , is no longer required. Moreover, $d _ { J S }$ is bounded. (c) ˆγ denotes the empirical risk, indicating that an ERM minimizer can make this term relatively small. (d) $\lambda ^ { * c }$ and λ characterize the challenges of DG and are analogous to the assumptions in (Ben-David et al., 2010).

According to the above bound, we observe that the term $\varepsilon _ { \eta _ { t } }$ appears on the RHS. This is natural, as we do not assume that the target domains are free of label noise. Therefore, the sum of $\varepsilon _ { \eta _ { t } }$ and $\varepsilon _ { \eta }$ contributes to the label-noise bias in the bound, which exhibits a significant diference from the i.i.d. data scenarios. Moreover, it is important to consider whether $\lambda ^ { * c }$ and λ are suficiently small. If either quantity is large, then, following the arguments in Ben-David et al. (2010), generalization to the target domain becomes dificult; conversely, smaller values are more favorable for generalization. These uncontrollable factors (ε and λ) on the RHS indicate that generalization under noisy data is inherently challenging. In particular, they are generally unavailable during training. We now argue that the above bound characterizes a more realistic and general DG setting, as label noise is often unavoidable in practical data collection processes. Based on this bound, we further derive the corresponding bound when applying T, as follows.

Corollary 29 Given the training set E and the test set $\mathcal { E } _ { t }$ with η-noisy labels, the DA kernel T, and assuming $\ell \in [ 0 , M ]$ that is σ-admissible, with probability at least $1 - \delta ,$ , the generalization gap between the relative risks on E after applying T and on $\mathcal { E } _ { t }$ is bounded as follows for any $h \in { \mathcal { H } }$

$$
\begin{array} { r } { \displaystyle \frac { 1 } { E _ { t } } \sum _ { t } \gamma ( h , h _ { t } ^ { * } ) \leq \frac { 1 } { E \bar { n } } \sum _ { s } \hat { \gamma } ( h , h _ { D A s } ^ { * } ) + \sigma \left( \varepsilon _ { \eta _ { t } } + \varepsilon _ { \eta _ { D A } } \right) } \\ { \displaystyle + C \left( \lambda _ { D A } ^ { * c } + \lambda _ { D A } \right) + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 E \bar { n } } } . } \end{array}
$$

Here, $\begin{array} { r } { \lambda _ { D A } = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { h _ { t } ^ { * } } , P _ { h _ { D A _ { s } } ^ { * } } ) } \end{array}$ and $\begin{array} { r } { \lambda _ { D A } ^ { * c } = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { h _ { t } ^ { * c } } , P _ { h _ { D A _ { s } } ^ { * c } } ) } \end{array}$

Remarks: (a) The same conclusions can be drawn, as discussed in Theorem 28. ((b) Since T is applied only to the source domains, the notations associated with the target domain remain the same as those in Theorem 28. (c) Assuming that $\lambda _ { D A }$ and $\lambda _ { D A } ^ { * c }$ are small is no longer suitable, which requires further discussion.

Here, $\lambda _ { D A }$ and $\lambda _ { D A } ^ { * c }$ are the new problems introduced by the DA methods. Compared with λ and $\lambda ^ { * c }$ , which typically represent the challenges introduced by the data themselves, $\lambda _ { D A }$ and $\lambda _ { D A } ^ { * c }$ are now linked to the learning strategies, bringing more uncertainty to the generalization problem. Is it necessary to derive conditions under which $\lambda _ { D A } + \lambda _ { D A } ^ { * c } \ge \lambda + \lambda ^ { * c }$ holds? The answer is negative, since such a derivation introduces additional unavailable terms, such as $h _ { t } ^ { * } , h _ { t } ^ { * c } , \forall t \ \in \ [ E _ { t } ]$ , which would require additional assumptions, or even stronger assumptions, for the analysis. Then, here we just need to know that applying DA methods further increases the uncertainty associated with learning and generalization. Consequently, we argue that DA methods should not be applied in the presence of noisy data for DG. Additionally, we establish the following result to support this argument.

Theorem 30 Given any training set E and the corresponding distribution set P, applying the DA kernel T to $\mathcal { E }$ yields the following inequality:

$$
J S D ( P _ { D A 1 } , \ldots , P _ { D A E } ) \le J S D ( P _ { 1 } , \ldots , P _ { E } )\tag{10}
$$

with equality if T is 1-1.

Remarks: (a) Applying $T$ to the data in $\mathcal { E }$ leads to a decrease in information, as $J S D =$ $I ( \cdot ; x )$ , where I represents mutual information. (b) If T is 1-1, there must exist $T ^ { - 1 }$ such that $x = T ^ { - 1 } ( x ^ { \prime } )$

From another perspective, applying T essentially pulls the data distributions across source domains closer together. Note that we do not specifically assume that the data contain noisy labels, which implies that the above result is applicable to any data condition. This implies that DA methods reduce the distribution discrepancy, which may be regarded as an advantage. However, the following result illustrates the potential risk associated with this efect.

Theorem 31 Let $\mathbf { P } = \{ P _ { 1 } , \ldots , P _ { E } \}$ be a finite set of probability distributions satisfying Assumption $\mathcal { Z } _ { \mathcal { H } ; }$ , and let $J S D ( P _ { 1 } , \dots , P _ { E } ) = K$ . Then, for any $h \in \mathcal H$ , and any epsilon $> 0$ it holds that

$$
\frac { 1 } { E } \sum _ { i } ^ { E } P [ d _ { P _ { i } } ( h , h _ { i } ^ { * } ) \geq \epsilon ] \geq 1 - \frac { K + \log ( 2 ) } { \log ( E ) } .
$$

Here, $h _ { i } ^ { * } \in a r g m i n _ { h \in \mathcal { H } } R _ { i } ( h ) , \forall i \in [ E ]$

Remarks: (a) There is a critical condition given by $K = \log ( E / 2 )$ . This condition implies that whenever $K \leq \log ( E / 2 )$ , the above lower bound characterizes a regime in which model estimation is intrinsically dificult. In contrast, if $K > \log ( E / 2 )$ , the lower bound becomes vacuous, indicating that the estimation dificulty characterized by the bound no longer exists. (b) $J S D ( P _ { 1 } , \ldots , P _ { E } ) \le \log ( E )$ . (c) As $E  \infty$ , a small value of K implies that the left-hand side (LHS) approaches 1, which represents the most challenging scenario for estimation.

The above lower bound is a variation of the well-known Fano’s lemma, and it clearly implies the risk associated with reducing the distribution discrepancy. Then, from Theorem 30, we have $K _ { D A } \le K$ , which results in $L H S _ { D A } \ge L H S$ . One particular case is that $K _ { D A } \ge \log ( E / 2 )$ , which implies that reducing the distribution discrepancy is acceptable. One may then ask what the essential efect of $L H S _ { D A } \ge L H S$ is. We present the following result to answer this question.

Theorem 32 Given a training set E and a testing set $\mathcal { E } _ { t } ,$ , and assuming $\ell \in [ 0 , M ]$ that is σ-admissible, the generalization gap between the relative risks on $\mathcal { E }$ and $\mathcal { E } _ { t }$ satisfies the following bound, for any $h \in { \mathcal { H } } .$

$$
\frac { 1 } { E _ { t } } \sum _ { t } \gamma ( h , h _ { t } ^ { * } ) \leq \sigma \frac { 1 } { E } \sum _ { s } d _ { P _ { s } } ( h , h _ { s } ^ { * } ) + 2 C \lambda .
$$

Here, $h _ { s } ^ { * } \in a r g m i n _ { h \in \mathcal { H } } R _ { s } ( h ) , \forall s \in [ E ]$ and $h _ { t } ^ { * } \in a r g m i n _ { h \in \mathcal { H } } R _ { t } ( h ) , \forall t \in [ E _ { t } ]$

Remarks: This bound is not of high-probability type, and its tightness is strongly dependent on the values of $d _ { P _ { s } }$ for all $s \in [ E ]$ . Even when λ is suficiently small, large values of $d _ { P _ { s } }$ can render the bound loose or even vacuous.

Since $P \propto d _ { P }$ , when $L H S _ { D A } \ge L H S$ , it follows that $d _ { P _ { D A _ { s } } } \geq d _ { P _ { s } }$ , which implies that the above bound becomes looser when applying T. Moreover, since we do not explicitly consider noise conditions, the above result holds under arbitrary data scenarios. In other words, regardless of label noise, applying DA methods can make the above bound looser, leading to less stable generalization. We now complete the analysis and summarize as follows. DA methods not only increase uncertainty in the learning process but also reduce the stability of generalization. The risk of using DA is greater in DG than in i.i.d. settings.

## 5 Discussion

(1) σ-admissible and M. One may note that in many results, we only provide the notations σ and M to represent the Lipschitz property and the uniform bound of the loss function. This is because we do not specify the exact loss formulation or the specific task. In this section, we present the following examples for classification and regression tasks, and the corresponding lemma can be found on Page 515 in (Bousquet and Elisseef, 2002). $\mathrm { I f ~ } \mathcal { Y } \in \{ - 1 , 1 \}$ , and considering the following loss function

$$
\ell ( f , ( x , y ) ) = ( 1 - y f ( x ) ) _ { + } = \left\{ { 1 - y f ( x ) } , \begin{array} { l l } { i f { 1 - y f ( x ) \geq 0 } } \\ { 0 , } \end{array}  , \right.
$$

we have $\sigma = 1$ and $M = 1 . { \mathrm { ~ I f ~ } } \mathcal { V } \in \{ 0 , B \}$ and considering the following loss function

$$
\ell ( f , ( x , y ) ) = ( f ( x ) - y ) ^ { 2 } ,
$$

we have $\sigma = 2 B$ and $M = B$ . Note that, for special learning objectives beyond ERM, such as Structure Risk Minimization (SRM) and Invariant Risk Minimization (IRM), M is associated with the corresponding hyper-parameters of the objective.

(2) Non-trivial Suficient Condition. We first discuss the assumption that $h ( x ) \in [ 0 , \upsilon ]$ As remarked previously, this assumption is mild for classification tasks, since most classification tasks can be formulated as binary classification problems. Under this formulation, we have $h ( x ) \in [ 0 , 1 ]$ . Considering regression tasks, however, this assumption is somewhat strong, since the outputs may be unbounded, $\mathrm { i . e . , ~ } h ( x ) \in [ - \infty , + \infty ]$ . Nevertheless, there exist methods for mapping these outputs into a finite interval without losing their original meaning by shrinking their amplitudes. For example, we can introduce the following operator, the sigmoid transformation

$$
s ( y ) = \frac { 1 } { 1 + e ^ { - y } } .
$$

Through this transformation, it follows that $h ( x ) \in [ 0 , 1 ]$

Then, we discuss why the presented conditions are considered loose. By examining the proof, one can observe that our original goal is to guarantee that $d _ { x } ( h , h ^ { * } ) ~ -$ $d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) \geq d _ { x } ( h , h ^ { * c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } )$ . However, in practical applications, $h ^ { * c }$ and $h _ { D A } ^ { * c }$ are unavailable. Therefore, we seek an upper bound D for $d _ { x } ( h , h ^ { \ast c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { \ast c } )$ , i.e., $d _ { x } ( h , h ^ { \ast } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { \ast } ) \geq D \geq d _ { x } ( h , h ^ { \ast c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { \ast c } )$ . Nevertheless, since $h ^ { * c }$ and $h _ { D A } ^ { * c }$ are inaccessible, we further derive a relatively loose upper bound K. Consequently, the resulting suficient conditions are also loose. If a tighter upper bound for $d _ { x } ( h , h ^ { \ast c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { \ast c } )$ can be derived, the corresponding suficient conditions can be tightened accordingly. We leave this as future work.

(3) λ. According to the bounds provided in Section 4, we denote λ as a measure of the challenge of Domain Generalization. This interpretation was first introduced by Ben-David et al. (2010), who derived the following upper bound for domain adaptation:

$$
\epsilon _ { T } ( h ) \leq \epsilon _ { S } ( h ) + C d _ { \mathcal { H } \Delta \mathcal { H } } ( S , T ) + \mathcal { O } ( 1 / \sqrt { m } ) + \lambda .
$$

Here, $\epsilon _ { T } ( h )$ and $\epsilon _ { S } ( h )$ represent the expected errors on the target domain T and source domain $S , \ d _ { \mathcal { H } \Delta \mathcal { H } }$ denotes the H-divergence, and $\lambda = \epsilon _ { T } ( h ^ { * } ) + \epsilon _ { S } ( h ^ { * } )$ . Then, they argued that when λ is large, one cannot expect to learn a good target classifier by minimizing the source error. Essentially, $\begin{array} { r } { \lambda = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { h _ { t } ^ { * } } , P _ { h _ { s } ^ { * } } ) } \end{array}$ is similar to theirs, as both depend on the assumed optimal hypotheses. However, they assume that the source and target domains share a common optimal hypothesis $h ^ { * }$ , which is a strong assumption in DG. In contrast, our analysis adopts a milder assumption. Therefore, as long as λ is suficiently small, one can learn a well-generalized $\hat { h }$ across all domains.

(4) $J S D ( \mathbf { P _ { D A } } )$ v.s. $J S D ( \mathbf { P } )$ . In Section 4, we present Inequality (10) as a simple characterization of the discrepancy between P and $\mathbf { P } _ { D A }$ . This is because the inequality can be derived without imposing any additional assumptions and follows directly from information-theoretic arguments. In fact, under the additional assumption that $v : =$ $\begin{array} { r } { \operatorname* { i n f } _ { x _ { i } \in s u p p P _ { i } , x _ { j } \in s u p p P _ { j } , \forall i \neq j \in [ E ] } \| x _ { i } - x _ { j } \| , \ v \ > \ 0 } \end{array}$ , and by defining $\mathbb { E } [ \Delta ] : = \mathbb { E } _ { P \sim \mathbf { P } } [ \Delta _ { \parallel x ^ { \prime } - x \parallel } ]$ we further have

$$
J S D ( P _ { D A 1 } , \ldots , P _ { D A E } ) \approx J S D ( P _ { 1 } , \ldots , P _ { E } ) \cdot ( 1 - \left( { \frac { \mathbb { E } [ \Delta ] } { v } } \right) ^ { 2 } ) .
$$

This relationship more clearly illustrates the efects of DA methods. If $\mathbb { E } [ \Delta ] / v  0$ , or if no DA method is applied, the information remains unchanged. If $0 < \mathbb { E } [ \Delta ] / v < 1$ , then we have $J S D _ { D A } < J S D$ . If $\mathbb { E } [ \Delta ] / v  \infty$ , we have $J S D _ { D A } = 0$ . In simple terms, DA changes the original support sets of the distributions and mixes them into indistinguishable states when $\mathbb { E } [ \Delta ] / v  \infty$ . Although this relationship facilitates the analysis, the assumption on v is restrictive, as it characterizes the interactions among all distribution support sets. In practice, such an assumption may not be satisfied.

## 6 Conclusion

According to analyses, we conclude that the potential risk of applying DA methods under label noise may outweigh their potential benefits. Although DA methods may yield $\varepsilon _ { \eta _ { D A } } \leq$ $\varepsilon _ { \eta }$ with a certain probability, and $\Delta _ { \parallel x ^ { \prime } - x \parallel }$ can be controlled during training, the resulting uncertainty cannot be eliminated. Note that we do not advocate minimizing $| h ( x ^ { \prime } ) - h ( x ) |$ or $| | h ( x ) - y | - | h ( x ^ { \prime } ) - \tilde { y } | |$ during training. In other words, our objective is not to control $r ( x )$ , which is primarily constrained by the predictive capability of the model itself.

Finally, we discuss a limitation of our work, namely the lack of empirical validation of the proposed suficient conditions under both the i.i.d. setting and the DG scenario.

At first glance, this limitation may hinder the practical applicability of our conditions in real-world settings. However, our analyses in the i.i.d. setting are motivated by a previous study (Pereira et al., 2022) that conducted a series of empirical validations. Our theorems provide a complete explanation for all the findings reported in that work under the i.i.d. setting. As for the DG scenario, we leave the empirical investigation to interested readers, as conducting such experiments would require substantial engineering efort and extensive computational resources beyond those currently available to us.

## Acknowledgements

## Appendix A. Lemmas & Proofs

In this appendix, we provide the lemmas and related proofs that support our main theorems.

## A.1 Lemmas

Lemma 33 (Expectation diference bound via KL) Let P and Q be two probability distribution such that $P \ll Q$ . Let $f : \mathcal { Z } \to \mathbb { R }$ be a measurable function satisfying

$$
\| f \| _ { \infty } : = \operatorname* { s u p } _ { z \in { \mathcal { Z } } } | f ( z ) | \leq M \leq \infty .
$$

Then

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \leq M { \sqrt { 2 K L ( P | | Q ) } } .
$$

Proof The result follows from standard inequalities in information theory. First, by a standard total variation bound,

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \leq 2 \left\| f \right\| _ { \infty } d _ { T V } ( P , Q ) .
$$

Applying Pinsker’s inequality, we obtain the desired result.

Here, we clarify how this lemma can be used to derive some results in our paper. Let $z = ( x , y )$ and consider $\ell ( h , z )$ as a function $f ( z )$ , since fixing h makes ℓ independent of the underlying distributions P and $Q .$ Then, assuming $\| \ell \| _ { \infty } \leq M$ , we have

$$
\mathbb { E } _ { P } [ | \ell ( h , z _ { 1 } ) | ] - \mathbb { E } _ { Q } [ | \ell ( h , z _ { 2 } ) | ] \leq M \sqrt { 2 K L ( P | | Q ) } .
$$

Here, $| a | - | b | \leq | a - b |$ . Note that if h afects the distributions, we obtain

$$
\mathbb { E } _ { P } [ | \ell ( h _ { 1 } , z _ { 1 } ) | ] - \mathbb { E } _ { Q } [ | \ell ( h _ { 2 } , z _ { 2 } ) | ] \le M { \sqrt { 2 K L ( P _ { h _ { 1 } } | | Q _ { h _ { 2 } } ) } } .
$$

Lemma 34 (Expectation diference bound via $W _ { 1 } )$ Let $f : \mathcal { Z } \to \mathbb { R }$ be an L-Lipschitz function, i.e.,

$$
\forall z _ { 1 } , z _ { 2 } \in \mathcal { Z } , | f ( z _ { 1 } ) - f ( z _ { 2 } ) | \le L d ( z _ { 1 } , z _ { 2 } ) .
$$

Then for any probability measures $P , Q$ on Z,

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \leq L W _ { 1 } ( P , Q ) .
$$

Proof Let $\gamma \in \prod ( P , Q )$ be any coupling. Then

$$
\begin{array} { r } { \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] = \mathbb { E } _ { ( z _ { 1 } , z _ { 2 } ) \sim \gamma } [ f ( z _ { 1 } ) - f ( z _ { 2 } ) ] . } \end{array}
$$

By the L-Lipschitz property of $f ,$ we have $| f ( z _ { 1 } ) - f ( z _ { 2 } ) | \leq L d ( z _ { 1 } , z _ { 2 } )$ , which yields

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \leq L \mathbb { E } _ { \gamma } [ d ( z _ { 1 } , z _ { 2 } ) ] .
$$

Taking the infimum over all couplings completes the proof.

Here, we clarify how this lemma can be used to derive some results in our paper. Let $z = ( x , y )$ and regard $\ell ( h , z )$ as a function $f ( z )$ , since fixing h makes ℓ independent of the underlying distributions $P$ and $Q .$ . Then, we have

$$
\mathbb { E } _ { P } [ | \ell ( h , z _ { 1 } ) | ] - \mathbb { E } _ { Q } [ | \ell ( h , z _ { 2 } ) | ] \le \sigma \mathbb { E } _ { \gamma } d ( z _ { 1 } , z _ { 2 } ) .
$$

Note that when h influences the distributions, we instead consider the KL divergence or JS divergence. ■

Lemma 35 (Hoefding Inequality (Hoefding, 1963)) Let $Z _ { 1 } , \ldots , Z _ { n }$ be n i.i.d random variables with $Z _ { i } \in [ a , b ]$ for each $i \in [ n ]$ . Then for any $\varepsilon > 0$ , we have

$$
P \left\{ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Z _ { i } - \mathbb { E } [ Z ] \ge \varepsilon \right\} \le e x p \left\{ - \frac { 2 n \varepsilon ^ { 2 } } { ( b - a ) ^ { 2 } } \right\} .
$$

Proof This is a classical inequality, and thus we omit the proof here. We only illustrate how it is applied in our cases. We consider $\ell ( h , z _ { i } )$ as the i-th random variable and assume that $\ell \in [ 0 , M ]$ . Then, based on this inequality, we obtain

$$
P \left\{ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell ( h , z _ { i } ) - \mathbb { E } _ { z \sim P } [ \ell ( h , z ) ] \geq \varepsilon \right\} \leq e x p \left\{ - \frac { 2 n \varepsilon ^ { 2 } } { M ^ { 2 } } \right\}
$$

Lemma 36 (Hoefding Inequality under Non-i.i.d. Settings) Let $Z _ { 1 } , \ldots , Z _ { n }$ be n independent random variables with $Z _ { i } ~ \in ~ [ a _ { i } , b _ { i } ]$ for each $i \in [ n ]$ . Define $\begin{array} { r } { S _ { n } = \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ Then for any $\varepsilon > 0$

$$
P \left\{ S _ { n } - \mathbb { E } [ S _ { n } ] \ge \varepsilon \right\} \le e x p \left\{ - \frac { 2 \varepsilon ^ { 2 } } { \sum _ { i = 1 } ^ { n } ( b _ { i } - a _ { i } ) ^ { 2 } } \right\} .
$$

Proof This is a classical inequality, and thus we omit the proof here. We only illustrate how it is applied in our cases. We consider $Z _ { i } = \ell ( h , z _ { i } )$ as the i-th random variable and $\begin{array} { r } { S _ { n } = \sum _ { i = 1 } ^ { n } Z _ { i } } \end{array}$ , and assume that $Z _ { i } \in [ 0 , M _ { i } ]$ . Then, based on this inequality, we have

$$
P \left\{ S _ { n } - \mathbb { E } [ S _ { n } ] \geq \varepsilon \right\} \leq e x p \left\{ - \frac { 2 \varepsilon ^ { 2 } } { \sum _ { i = 1 } ^ { n } M _ { i } ^ { 2 } } \right\} .
$$

Note that no assumption is made that the variables are i.i.d.

Lemma 37 (Expectation Bound via JS divergence) For any function f satisfying the condition in Lemma 33, the diference between its expectations under P and Q is upper bounded by:

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \le 2 M \sqrt { 2 J S ( P | | Q ) } ,
$$

or

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \leq 2 \sqrt { 2 } M d _ { J S } ( P , Q ) .
$$

Proof The result follows from standard inequalities in information theory. First, by a standard total variation bound,

$$
| \mathbb { E } _ { P } [ f ] - \mathbb { E } _ { Q } [ f ] | \leq 2 \left\| f \right\| _ { \infty } d _ { T V } ( P , Q ) .
$$

Applying the Pinsker’s inequality, we have

$$
\begin{array} { l } { \displaystyle \left( \frac 1 2 d _ { T V } ( P , Q ) \right) ^ { 2 } \le \frac 1 2 K L ( P | | M ) } \\ { \displaystyle \left( \frac 1 2 d _ { T V } ( P , Q ) \right) ^ { 2 } \le \frac 1 2 K L ( Q | | M ) . } \end{array}
$$

Here, $M = ( P + Q ) / 2 , d _ { T V } ( P , M ) = ( 1 / 2 ) d _ { T V } ( P , Q )$ , and $d _ { T V } ( Q , M ) = ( 1 / 2 ) d _ { T V } ( P , Q )$ Then, according to Definition 22, $\mathrm { i . e . , } J S ( P ; Q ) = 1 / 2 K L ( P ; M ) + 1 / 2 K L ( Q ; M )$ , we obtain

$$
2 \left( \frac { 1 } { 2 } d _ { T V } ( P , Q ) \right) ^ { 2 } \leq \frac { 1 } { 2 } J S ( P | | Q ) .
$$

Here, we clarify how this lemma can be used to derive some results in our paper. Let $z = ( x , y )$ and consider $\ell ( h , z )$ as a function $f ( z )$ , since fixing h makes ℓ independent of the underlying distributions P and $Q .$ . Then, assuming $\| \ell \| _ { \infty } \leq M$ , we have

$$
\mathbb { E } _ { P } [ | \ell ( h , z _ { 1 } ) | ] - \mathbb { E } _ { Q } [ | \ell ( h , z _ { 2 } ) | ] \le 2 M \sqrt { 2 J S ( P | | Q ) } .
$$

Note that if h afects the distributions, we obtain

$$
\mathbb { E } _ { P } [ | \ell ( h _ { 1 } , z _ { 1 } ) | ] - \mathbb { E } _ { Q } [ | \ell ( h _ { 2 } , z _ { 2 } ) | ] \le 2 M \sqrt { 2 J S ( P _ { h _ { 1 } } | | Q _ { h _ { 2 } } ) } .
$$

Lemma 38 (Properties of Divergences (Polyanskiy and Wu, 2014)) $\begin{array} { r l r l } { ( 1 ) } & { { } x ^ { \prime } } & { } & { { } = } \end{array}$ $f ( x ) \ \Rightarrow \ K L ( P _ { x } | | Q _ { x } ) \ \geq \ K L ( P _ { x ^ { \prime } } | | Q _ { x ^ { \prime } } )$ , with equality if f is 1-1. (2) $x ^ { \prime } \ = \ f ( x ) \ \Rightarrow$ $K L ( P _ { ( x , y ) } | | Q _ { ( x , y ) } ) \geq K L ( P _ { ( x ^ { \prime } , y ) } | | Q _ { ( x ^ { \prime } , y ) } )$ , with equality if f is 1-1.

Proof Lemma (1) is a direct consequence of Theorem 2.2 in (Polyanskiy and Wu, 2014) (Page 15), whose proof is omitted therein. This lemma is a classical result in information theory, indicating that whenever the input is processed by a non-injective function, some

information is inevitably lost, leading to a reduction in the KL divergence. We now proceed to prove Lemma (2).

Let $z = ( x , y )$ , and define a deterministic mapping g by $g ( x , y ) = ( f ( x ) , y )$ . Denote $z ^ { \prime } = g ( z ) = ( x ^ { \prime } , y )$ . Then, by the data processing inequality, it follows that $K L ( P _ { z } | | Q _ { z } ) \geq$ $K L ( P _ { z ^ { \prime } } | | Q _ { z ^ { \prime } } )$

Lemma 39 (Fano’s Lemma) Let S be a finite set of probabilities $S = \{ P _ { 1 } , \ldots , P _ { m } \}$ such that $J S D ( P _ { 1 } , \ldots , P _ { m } ) = K$ . Then for any estimate ϕ with values in $\{ 1 , \ldots , m \}$ we have

$$
m ^ { - 1 } \sum _ { i = 1 } ^ { m } P [ \phi \neq i ] \geq 1 - \frac { K + \log ( 2 ) } { \log ( m ) } .
$$

Proof This is also a classical Fano-type lemma in statistical learning theory (see the proof in $( \mathrm { B i r g e } , 1 9 8 6 ) )$ . It characterizes a fundamental dificulty in statistical estimation: as the JSD between distributions decreases, the estimation error necessarily increases.

In our work, we consider $\mathcal { P } = \{ P \in \mathcal { P } ( \mathcal { H } ) \}$ . Therefore, this lemma can be naturally extended to model estimation. Details are provided in the proofs of the corresponding theorems. ■

## A.2 Proofs

Proof of Theorem 6. Since $\mathcal { L } ( \theta ) = \mathbb { E } _ { ( x , \tilde { y } ) } [ \ell ( h , ( x , \tilde { y } ) ) ] = \mathbb { E } _ { ( x , y ) } \mathbb { E } _ { \tilde { y } } [ \ell ( h , ( x , \tilde { y } ) ) ]$ , we have

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \mathcal { L } ^ { c } ( \theta ) + \mathbb { E } _ { ( x , y ) } \mathbb { E } _ { \tilde { y } } [ \ell ( h , ( x , \tilde { y } ) ) - \ell ( h , ( x , y ) ) ] } \\ & { \quad \quad = \mathcal { L } ^ { c } ( \theta ) + \eta \mathbb { E } _ { ( x , y ) } \mathbb { E } _ { \tilde { y } \neq y } [ \ell ( h , ( x , \tilde { y } ) ) - \ell ( h , ( x , y ) ) ] } \\ & { \quad \quad = \mathcal { L } ^ { c } ( \theta ) + \eta \Delta ( \theta ) } \end{array}
$$

where $\Delta ( \theta ) = \mathbb { E } _ { ( x , y ) } \mathbb { E } _ { \tilde { y } \neq y } [ \ell ( h , ( x , \tilde { y } ) ) - \ell ( h , ( x , y ) ) ]$

Let $g _ { k } = \nabla \mathcal { L } ( \theta _ { k } )$ and $\nabla \mathcal { L } ( \theta _ { t } ) = \nabla \mathcal { L } ^ { c } ( \theta _ { k } ) + \eta \nabla \Delta ( \theta _ { k } )$ . Here, we consider $\nabla _ { \theta } { \mathcal { L } } ( \theta ) =$ $\mathbb { E } _ { ( x , y ) \sim P } [ \nabla _ { \theta } \ell ( h , ( x , y ) ) ]$ , and do not necessarily to explicitly Jacobian matrix $J _ { \theta } ( x ) \ =$ $\nabla _ { \boldsymbol { \theta } } h _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ , as it is not essential for the following derivations. Since the stochastic gradient $g _ { k }$ is an unbiased estimator of the gradient for noisy samples, i.e., $\mathbb { E } [ g _ { k } \ | \ \theta _ { k } ] = \nabla \mathcal { L } ( \theta _ { k } )$ , we have the following second moment

$$
\begin{array} { r } { \mathbb { E } \| g _ { k } \| ^ { 2 } \leq 2 \mathbb { E } \| \nabla \mathcal { L } ^ { c } \| ^ { 2 } + 2 \eta ^ { 2 } \mathbb { E } \| \nabla \Delta \| ^ { 2 } . } \end{array}
$$

By Definition 1, we have $\| \nabla { \mathcal { L } } ^ { c } \| ^ { 2 } \leq \sigma ^ { 2 }$ and obtain $\begin{array} { r } { \mathbb { E } \| g _ { k } \| ^ { 2 } \leq 2 ( \sigma ^ { 2 } + \eta ^ { 2 } \mathbb { E } \| \nabla \Delta \| ^ { 2 } ) : = G ^ { 2 } ( \eta ) } \end{array}$

Under Assumption 5, let $\theta ^ { * }$ denote an optimal solution for $\mathcal { L } ( \boldsymbol { \theta } )$ . Then, one step of SGD yields

$$
\lVert \theta _ { k + 1 } - \theta ^ { * } \rVert = \lVert \theta _ { k } - \theta ^ { * } - \alpha _ { k } g _ { k } \rVert .
$$

Squaring both sides, we obtain

$$
\begin{array} { r l } & { \| \theta _ { k + 1 } - \theta ^ { * } \| ^ { 2 } = \| \theta _ { k } - \theta ^ { * } - \alpha _ { k } g _ { k } \| ^ { 2 } } \\ & { \qquad = \| \theta _ { k } - \theta ^ { * } \| ^ { 2 } - 2 \alpha _ { k } \langle g _ { k } , \theta _ { k } - \theta ^ { * } \rangle + \alpha _ { k } ^ { 2 } g _ { k } ^ { 2 } . } \end{array}
$$

Taking expectation and using the convexity of $\ell ,$ we have

$$
\begin{array} { r l } & { \mathbb { E } [ \| \theta _ { k + 1 } - \theta ^ { * } \| ^ { 2 } \ | \ \theta _ { k } ] \leq \mathbb { E } [ \| \theta _ { k } - \theta ^ { * } \| ^ { 2 } ] - 2 \alpha _ { k } \mathbb { E } [ \mathcal { L } ( \theta _ { k } ) - \mathcal { L } ( \theta ^ { * } ) ] + \alpha _ { k } ^ { 2 } \mathbb { E } \| g _ { k } \| ^ { 2 } } \\ & { 2 \alpha _ { k } \mathbb { E } [ \mathcal { L } ( \theta _ { k } ) - \mathcal { L } ( \theta ^ { * } ) ] \leq \mathbb { E } [ \| \theta _ { k } - \theta ^ { * } \| ^ { 2 } ] - \mathbb { E } [ \| \theta _ { k + 1 } - \theta ^ { * } \| ^ { 2 } ] + \alpha _ { k } ^ { 2 } G ^ { 2 } ( \eta ) . } \end{array}
$$

Then, by a telescoping argument, we obtain

$$
2 \sum _ { k = 1 } ^ { K } \alpha _ { k } \mathbb { E } [ \mathcal { L } ( \theta _ { k } ) - \mathcal { L } ( \theta ^ { * } ) ] \leq \Theta ^ { 2 } + G ^ { 2 } ( \eta ) \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { 2 } ,
$$

where $\Theta = \lVert \theta _ { 0 } - \theta ^ { * } \rVert$ . Let $\bar { \theta } _ { K } = \sum { \alpha \theta _ { k } / \sum }$ α and $\alpha = \Theta / ( G ( \eta ) \sqrt { K } )$ , we finally have

$$
\mathbb { E } [ \mathcal { L } ( \bar { \theta } _ { K } ) - \mathcal { L } ( \theta ^ { * } ) ] \leq \frac { \Theta G ( \eta ) } { \sqrt { K } } .
$$

Assume that $\theta ^ { \ast c } \in$ argmin ${ \mathcal { L } } ^ { c } ( \theta )$ , we have

$$
\begin{array} { r l } & { \mathcal { L } ( { \theta ^ { * } } ) - \mathcal { L } ^ { c } ( { \theta ^ { * c } } ) = \eta \Delta ( { \theta ^ { * } } ) + [ \mathcal { L } ^ { c } ( { \theta ^ { * } } ) - \mathcal { L } ^ { c } ( { \theta ^ { * c } } ) ] } \\ & { \qquad \leq \eta \Delta ( { \theta ^ { * } } ) } \\ & { \qquad \leq \eta \sigma \mathbb { E } _ { y , \tilde { y } } \| \tilde { y } - y \| , } \end{array}
$$

where the last inequality holds for Definition 1. Let $\Delta _ { \parallel \tilde { y } - y \parallel } = \mathbb { E } _ { y , \tilde { y } } \lVert \tilde { y } - y \rVert$ , we have

$$
\begin{array} { r l } & { \qquad \mathcal { L } ( \theta ^ { * } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) \leq \eta \sigma \Delta _ { \| \tilde { y } - y \| } } \\ & { \qquad \mathcal { L } ( \bar { \theta } _ { K } ) + \mathcal { L } ( \theta ^ { * } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) \leq \mathcal { L } ( \bar { \theta } _ { K } ) + \eta \sigma \Delta _ { \| \tilde { y } - y \| } } \\ & { \qquad \mathcal { L } ( \bar { \theta } _ { K } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) \leq \mathcal { L } ( \bar { \theta } _ { K } ) - \mathcal { L } ( \theta ^ { * } ) + \eta \sigma \Delta _ { \| \tilde { y } - y \| } . } \end{array}
$$

Combining the previous results, we have

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } ( \bar { \theta } _ { K } ) - \mathcal { L } ^ { c } ( { \theta } ^ { * c } ) ] \leq \mathbb { E } [ \mathcal { L } ( \bar { \theta } _ { K } ) - \mathcal { L } ( { \theta } ^ { * } ) ] + \eta \sigma \Delta _ { \| \tilde { y } - y \| } } \\ & { \mathbb { E } [ \mathcal { L } ( \bar { \theta } _ { K } ) - \mathcal { L } ^ { c } ( { \theta } ^ { * c } ) ] \leq \displaystyle \frac { { \Theta } G ( \eta ) } { \sqrt { K } } + \eta \sigma \Delta _ { \| \tilde { y } - y \| } . } \end{array}
$$

To achieve an ε-optimal solution, it sufices that

$$
K \ge \left( \frac { \Theta G ( \eta ) } { \varepsilon - \eta \sigma \Delta _ { \| \tilde { y } - y \| } } \right) ^ { 2 } ,
$$

which derives η.

Proof of Theorem 7. Since $\mathcal { L } _ { D A } ( \theta ) = \mathbb { E } _ { ( x , \tilde { y } ) } \mathbb { E } _ { x ^ { \prime } } [ \ell ( h , ( x ^ { \prime } , \tilde { y } ) ) ]$ , we have

$$
\begin{array} { r l } & { \mathcal { L } _ { D A } ( \theta ) = \mathcal { L } ( \theta ) + \mathbb { E } _ { ( x , \tilde { y } ) } \mathbb { E } _ { x ^ { \prime } } [ \ell ( h , ( x ^ { \prime } , \tilde { y } ) ) - \ell ( h , ( x , \tilde { y } ) ) ] } \\ & { \quad \quad \quad = \mathcal { L } ( \theta ) + \Delta _ { D A } ( \theta ) , } \end{array}
$$

where $\Delta _ { D A } ( \theta ) = \mathbb { E } _ { ( x , \tilde { y } ) } \mathbb { E } _ { x ^ { \prime } } [ \ell ( h , ( x ^ { \prime } , \tilde { y } ) ) - \ell ( h , ( x , \tilde { y } ) ) ]$

Let $g _ { k } = \nabla \mathcal { L } _ { D A } ( \theta _ { k } )$ and $\nabla \mathcal { L } _ { D A } ( \theta _ { t } ) = \nabla \mathcal { L } ( \theta _ { k } ) + \nabla \Delta _ { D A } ( \theta _ { k } )$ . Similarly, we have

$$
\begin{array} { r } { \mathbb { E } \| g _ { k } \| ^ { 2 } \leq 2 \mathbb { E } \| \nabla \mathcal { L } \| ^ { 2 } + 2 \mathbb { E } \| \nabla \Delta _ { D A } \| ^ { 2 } . } \end{array}
$$

By Definition 1, we have $\| \nabla { \mathcal { L } } \| ^ { 2 } \leq \sigma ^ { 2 }$ and $\begin{array} { r } { \mathbb { E } \| g _ { k } \| ^ { 2 } \leq 2 ( \sigma ^ { 2 } + \mathbb { E } \| \nabla \Delta _ { D A } \| ^ { 2 } ) : = G _ { D A } ^ { 2 } . } \end{array}$

Under Assumption 5 and by following the previous steps in the proof of Theorem 6, we finally obtain

$$
\mathbb { E } [ \mathcal { L } _ { D A } ( \bar { \theta } _ { K } ) - \mathcal { L } _ { D A } ( \theta _ { D A } ^ { * } ) ] \leq \frac { \Theta G _ { D A } } { \sqrt { K } } ,
$$

where $\theta _ { D A } ^ { * }$ is an optimal solution for $\mathcal { L } _ { D A } ( \theta )$ . Consider that $\theta ^ { * } \in a r g m i n \mathcal { L } ( \theta )$ and $\theta ^ { \ast c } \in$ argminL<sup>c</sup>(θ), we have

$$
\begin{array} { r l } & { \mathcal { L } _ { D A } ( \theta _ { D A } ^ { * } ) - \mathcal { L } ( \theta ^ { * } ) = \Delta ( \theta _ { D A } ^ { * } ) + [ \mathcal { L } ( \theta _ { D A } ^ { * } ) - \mathcal { L } ( \theta ^ { * } ) ] } \\ & { \qquad \leq \Delta ( \theta _ { D A } ^ { * } ) } \\ & { \qquad \leq \sigma \mathbb { E } _ { x , x ^ { \prime } } \lVert x ^ { \prime } - x \rVert , } \end{array}
$$

where the last inequality holds for Lemma 34. Let $\Delta _ { \parallel x ^ { \prime } - x \parallel } = \mathbb { E } _ { x , x ^ { \prime } } \lVert x ^ { \prime } - x \rVert$ , we have

$$
\begin{array} { r l } & { \qquad \mathcal { L } _ { D A } ( \theta _ { D A } ^ { * } ) - \mathcal { L } ( \theta ^ { * } ) \leq \sigma \Delta _ { \| x ^ { \prime } - x \| } } \\ & { ( \mathcal { L } _ { D A } ( \theta _ { D A } ^ { * } ) - \mathcal { L } ( \theta ^ { * } ) ) + ( \mathcal { L } ( \theta ^ { * } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) ) \leq \sigma \Delta _ { \| x ^ { \prime } - x \| } + \eta \sigma \Delta _ { \| \tilde { y } - y \| } } \\ & { \qquad \mathcal { L } _ { D A } ( \theta _ { D A } ^ { * } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) \leq \sigma \Delta _ { \| x ^ { \prime } - x \| } + \eta \sigma \Delta _ { \| \tilde { y } - y \| } } \\ & { \qquad \mathcal { L } _ { D A } ( \bar { \theta } _ { K } ) + \mathcal { L } _ { D A } ( \theta _ { D A } ^ { * } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) \leq \mathcal { L } _ { D A } ( \bar { \theta } _ { K } ) + \sigma \Delta _ { \| x ^ { \prime } - x \| } + \eta \sigma \Delta _ { \| \tilde { y } - y \| } } \\ & { \qquad \mathcal { L } _ { D A } ( \bar { \theta } _ { K } ) - \mathcal { L } ^ { c } ( \theta ^ { * c } ) \leq \mathcal { L } _ { D A } ( \bar { \theta } _ { K } ) - \mathcal { L } _ { D A } ^ { * } + \sigma \Delta _ { \| x ^ { \prime } - x \| } + \eta \sigma \Delta _ { \| \tilde { y } - y \| } . } \end{array}
$$

Combining the previous results, we have

$$
\mathbb { E } [ \mathcal { L } _ { D A } ( \bar { \theta } _ { K } ) - \mathcal { L } ^ { c } ( { \theta } ^ { * c } ) ] \leq \frac { \Theta G _ { D A } } { \sqrt { K } } + \sigma \Delta _ { \| { x } ^ { \prime } - { x } \| } + \eta \sigma \Delta _ { \| \tilde { y } - { y } \| } .
$$

To achieve an ε-optimal solution, it sufices that

$$
K \geq \left( \frac { \Theta G _ { D A } } { \varepsilon - \eta \sigma \Delta _ { \| \tilde { y } - y \| } - \sigma \Delta _ { \| x ^ { \prime } - x \| } } \right) ^ { 2 } ,
$$

which derives η.

Proof of Corollary 8. The proof process follows that of Theorem 7. The main diference is that we additionally need to consider $\Delta _ { D A } ^ { \prime } : = \mathbb { E } _ { x , \tilde { y } } \mathbb { E } _ { ( x ^ { \prime } , \tilde { y } ^ { \prime } ) } [ \ell ( h , ( x ^ { \prime } , \tilde { y } ^ { \prime } ) ) - \ell ( h , ( x , \tilde { y } ) ) ]$ , since we have $\mathcal { L } _ { D A } ^ { \prime } ( \theta ) = \mathcal { L } ( \theta ) + \mathcal { L } _ { D A } ^ { \prime } ( \theta ) - \mathcal { L } ( \theta )$ . Assuming $\ell \leq M$ , by Lemma 33, we obtain

$$
\Delta _ { D A } ^ { \prime } \leq M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) } ,
$$

which completes the proof.

Proof of Theorem 9. We first establish the following result:

$$
\begin{array} { r l } & { \hat { R } ( h ) - R ( h ) = \eta ( \hat { R } ( h ) - R ( h ) ) _ { \tilde { y } \neq y } + ( 1 - \eta ) ( \hat { R } ( h ) - R ( h ) ) _ { \tilde { y } = y } } \\ & { \qquad = \eta ( \hat { R } ( h ) - R ( h ) _ { \tilde { y } \neq y } + ( 1 - \eta ) ( \hat { R } ^ { c } ( h ) - R ^ { c } ( h ) ) . } \end{array}
$$

Since we assume that $\ell \leq M .$ , it follows that $\ell ( h , ( x _ { i } , \tilde { y } _ { i } ) ) - \mathbb { E } _ { ( x , \tilde { y } ) } [ \ell ] \in [ - M , M ]$ . Then, by Lemma 35, with probability at least $1 - ( \delta / 2 )$ , we have

$$
\hat { R } ( h ) - R ( h ) _ { \tilde { y } \neq y } \leq \sqrt { \frac { ( 2 M ) ^ { 2 } \ln ( 2 / \delta ) } { 2 n _ { \eta } } } ,
$$

and

$$
\hat { R } ^ { c } ( h ) - R ^ { c } ( h ) ) \leq \sqrt { \frac { ( 2 M ) ^ { 2 } \ln ( 2 / \delta ) } { 2 n _ { c } } } ,
$$

which completes the proof.

Proof of Theorem 11. We first establish the following result:

$$
\hat { R } _ { D A } ( h ) - R ( h ) = [ \hat { R } _ { D A } ( h ) - \hat { R } ( h ) ] + [ \hat { R } ( h ) - R ( h ) ] .
$$

The second term has been addressed previously; thus, we focus on the first term. Let $Z _ { i } = [ \ell ( h , ( x _ { i } ^ { \prime } , \tilde { y } _ { i } ) ) - \ell ( h , ( x _ { i } , \tilde { y } _ { i } ) ) ] - [ \mathbb { E } _ { ( x ^ { \prime } , \tilde { y } ) } - \mathbb { E } _ { ( x , \tilde { y } ) } ]$ , then it follows that $Z _ { i } \in [ - 2 M , 2 M ]$ Similarly, by Lemma 35, with probability at least $\dot { 1 } - ( \delta / 3 )$ , we have

$$
\begin{array} { r l } & { [ \hat { R } _ { D A } ( h ) - \hat { R } ( h ) ] - [ R _ { D A } ( h ) - R ( h ) ] \leq \sqrt { \frac { ( ( 4 M ) ^ { 2 } \ln ( 3 / \delta ) } { 2 n } } . } \\ & { \qquad \leq [ R _ { D A } ( h ) - R ( h ) ] + \sqrt { \frac { ( 4 M ) ^ { 2 } \ln ( 3 / \delta ) } { 2 n } } } \\ & { \qquad \leq \sigma \Delta _ { \| x ^ { \prime } - x \| } + \sqrt { \frac { ( 4 M ) ^ { 2 } \ln ( 3 / \delta ) } { 2 n } } , } \end{array}
$$

where the last inequality follows from Lemma 34, completing the proof.

Proof of Corollary 12. Based on the proof of Corollary 8, we have

$$
R _ { D A } ^ { \prime } ( h ) - R ( h ) \leq M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) } ,
$$

which completes the proof by following the same procedure as in the proof of Theorem 11.

Proof of Corollary 10 and Corollary 13. Considering the events ${ \hat { R } } ( h ) - R ( h ) \geq \varepsilon$ $\hat { R } _ { D A } ( h ) - R ( h ) \geq \varepsilon$ , and $\hat { R } _ { D A } ^ { \prime } ( h ) - R ( h ) \geq \varepsilon$ , the desired results follow directly from the

corresponding theorems. For example, $\varepsilon \le \eta A + ( 1 - \eta ) B = \eta A - \eta B + B$ leads $\begin{array} { r } { \eta \ge \frac { \varepsilon - B } { A - B } } \end{array}$ and $\begin{array} { r } { \eta \le \frac { B - \varepsilon } { B - A } } \end{array}$

Proof of Theorem 15. Since ℓ is bounded by M, let $Z _ { i } = \hat { \gamma } ( h _ { 1 } , h _ { 2 } ) _ { i } - \gamma ( h _ { 1 } , h _ { 2 } )$ , we obtain $Z _ { i } \in [ - 2 M , 2 M ]$ . Then, by Lemma 35, for any $\varepsilon > 0$ , we have

$$
P \left\{ { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } Z _ { i } - \mathbb { E } [ Z ] \geq \varepsilon \right\} \leq e x p \left\{ - { \frac { 2 n \varepsilon ^ { 2 } } { ( 4 M ) ^ { 2 } } } \right\} .
$$

Proof of Corollary 16. Based on Theorem 15, diferent types of inequalities can be derived. Here, we only illustrate how to decompose the relative expected risks. Specifically, we consider the following decomposition:

$$
\begin{array} { r l } & { \mathbb { E } [ \ell ( h _ { 1 } , z ) - \ell ( h _ { 2 } , z ) ] = \mathbb { E } [ \ell ( h _ { 1 } , z ) + \ell ( h , z ) - \ell ( h , z ) - \ell ( h _ { 2 } , z ) ] } \\ & { \qquad = \mathbb { E } [ \ell ( h _ { 1 } , z ) - \ell ( h , z ) ] + \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { 2 } , z ) ] . } \end{array}
$$

Then, based on Definition 1 and Definition 14, we rewire $\mathbb { E } [ \ell ( h , z ) - \ell ( h _ { 2 } , z ) ]$ to $\sigma d ( h , h _ { 2 } )$ 2 completing the proof. ■

Proof of Theorem 17. Let $a = h ^ { * } ( x ) , b = h ^ { * c } ( x ) , c = h _ { D A } ^ { * } ( x ^ { \prime } )$ , and $d = h _ { D A } ^ { * c } ( x ^ { \prime } )$ . Then, applying the inequality

$$
| | a - b | - | c - d | | \leq | a - c | + | b - d | ,
$$

the proof is completed by taking expectations on both sides.

Proof of Theorem 18. Since $\epsilon = \mathbb { E } _ { x ^ { \prime } , \tilde { y } } [ \ell ( h _ { D A } ^ { * } , ( x ^ { \prime } , \tilde { y } ) ) ]$ , it holds that

$$
\begin{array} { r l r } & { } & { { \mathbb { E } } _ { x ^ { \prime } , \tilde { y } } [ \left| \ell ( h _ { D A } ^ { \ast } , ( x ^ { \prime } , \tilde { y } ) ) - \ell ( h _ { D A } ^ { \ast c } , ( x ^ { \prime } , \tilde { y } ) ) \right| ] = { \mathbb { E } } _ { x ^ { \prime } , \tilde { y } } [ \left| \ell ( h _ { D A } ^ { \ast c } , ( x ^ { \prime } , \tilde { y } ) ) - \epsilon \right| ] } \\ & { } & { = { \mathbb { E } } _ { x ^ { \prime } , \tilde { y } } [ \left| \ell ( h _ { D A } ^ { \ast c } , ( x ^ { \prime } , \tilde { y } ) ) \right| ] - \epsilon } \end{array}
$$

Moreover, since $\ell \in [ 0 , M ]$ , the expected risk can be transformed into the error rate as

$$
\mathbb { E } _ { \boldsymbol { x } ^ { \prime } , \tilde { \boldsymbol { y } } } [ \big | \ell ( h _ { D A } ^ { * c } , ( \boldsymbol { x } ^ { \prime } , \tilde { \boldsymbol { y } } ) ) \big | ] = M \cdot P _ { D A } ( h _ { D A } ^ { * c } ( \boldsymbol { x } ^ { \prime } ) \neq \tilde { \boldsymbol { y } } ) ,
$$

and let $P _ { n } = P _ { D A } ( h _ { D A } ^ { * c } ( x ^ { \prime } ) \ne \tilde { y } )$ and $h = h _ { D A } ^ { * c }$ . Then, according to the Law of Total Probability, we can decompose $P _ { n }$ as follows while ignoring higher-order infinitesimal terms:

$$
P _ { n } \approx \underbrace { P ( \tilde { y } \neq y ) P ( h ( x ^ { \prime } ) = y | \tilde { y } \neq y ) } _ { \mathrm { e r r o r ~ b y ~ n o i s e } } + \underbrace { P ( \tilde { y } = y ) P ( h ( x ^ { \prime } ) \neq y | \tilde { y } = y ) } _ { \mathrm { e r r o r ~ b y ~ D A } } .
$$

Since h is the optimal solution trained on cleaned data with DA, $P ( h ( x ^ { \prime } ) = y | \tilde { y } \neq y ) = 1$ while $P ( h ( x ^ { \prime } ) \neq y | \tilde { y } = y )$ is mainly determined by the prediction behavior of h, namely that

h fails to correctly classify the input $x ^ { \prime }$ . This implies that the feature perturbation $\left| x ^ { \prime } - x \right|$ crosses the decision margin boundary of $h ,$ , namely, $| x ^ { \prime } - x | > r ( x )$ , and we have

$$
P ( h ( x ^ { \prime } ) \neq y | \tilde { y } = y ) = P _ { r ( x ) } ( \| x ^ { \prime } - x \| \ge r ( x ) ) .
$$

Then, we obtain

$$
P _ { n } = P ( \tilde { y } \ne y ) + P ( \tilde { y } = y ) P _ { r ( x ) } .
$$

Let $Z = | x ^ { \prime } - x |$ . By Markov’s inequality, we have

$$
P ( Z \geq r ( x ) ) \leq \frac { \mathbb { E } [ Z ] } { r ( x ) } .
$$

We reformulate all results as follows:

$$
M P _ { n } - \varepsilon \approx P ( \tilde { y } \neq y ) + P ( \tilde { y } = y ) P _ { r ( x ) } .
$$

Based on the above, we know that

$$
P _ { n } \ge \operatorname* { m a x } \left\{ P ( \tilde { y } \ne y ) , P _ { r ( x ) } , \frac { 1 } { 2 } \left( P ( \tilde { y } \ne y ) + P _ { r ( x ) } \right) \right\} .
$$

Moreover, $P ( \tilde { y } \neq y ) \propto \mathbb { E } _ { x ^ { \prime } , \tilde { y } , y } [ | \tilde { y } - y | ]$ , and let $P _ { r ( x ) } = \mathbb { E } [ | x ^ { \prime } - x | ] / r ( x )$ , we have

$$
P _ { n } \geq \operatorname* { m a x } \left\{ \underbrace { \alpha \mathbb { E } _ { x ^ { \prime } , \tilde { y } , y } [ | \tilde { y } - y | ] } _ { A } , \underbrace { \frac { \mathbb { E } [ | x ^ { \prime } - x | ] } { r ( x ) } } _ { B } , \underbrace { \frac { 1 } { 2 } \left( \alpha \mathbb { E } _ { x ^ { \prime } , \tilde { y } , y } [ | \tilde { y } - y | ] + \frac { \mathbb { E } [ | x ^ { \prime } - x | ] } { r ( x ) } \right) } _ { C } \right\} .
$$

Here, $\mathbb { E } _ { x ^ { \prime } , \tilde { y } , y }$ and E denote the expectations with respect to $P _ { D A }$ , and α represents a constant discrepancy between $P$ and $\mathbb { E }$ . Then, based on the condition (6) and $d _ { x } ( h ^ { \ast } , h ^ { \ast c } ) \leq \eta _ { m a x } ,$ we have

$$
\operatorname* { m a x } \{ A , B , C \} \geq \frac { \sigma \eta _ { m a x } - \epsilon } { M } .
$$

Namely,

$$
M \operatorname* { m a x } \{ A , B , C \} - \epsilon \geq \sigma \eta _ { m a x } .
$$

Since ℓ is σ-admissible, it follows that $\sigma d _ { x ^ { \prime } } ( h _ { D A } ^ { * } , h _ { D A } ^ { * c } ) \geq \mathbb { E } _ { x ^ { \prime } , \tilde { y } } [ | \ell ( h _ { D A } ^ { * } , \cdot ) - \ell ( h _ { D A } ^ { * c } , \cdot ) | ]$ . Thus, we have

$$
\begin{array} { r } { \sigma d _ { x ^ { \prime } } ( h _ { D A } ^ { * } , h _ { D A } ^ { * c } ) \geq M \cdot P _ { n } - \epsilon \geq M \operatorname* { m a x } \{ A , B , C \} - \epsilon \geq \sigma \eta _ { m a x } \geq \sigma d _ { x } ( h ^ { * } , h ^ { * c } ) , } \end{array}
$$

which completes the proof.

Proof of Theorem 19. By the triangle inequality, for any $h \in \mathcal H$ , we have

$$
\begin{array} { c } { { - d _ { x } ( h ^ { * } , h ^ { * c } ) \geq - d _ { x } ( h , h ^ { * } ) - d _ { x } ( h , h ^ { * c } ) } } \\ { { { } } } \\ { { - d _ { x ^ { \prime } } ( h _ { D A } ^ { * } , h _ { D A } ^ { * c } ) \geq - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) } } \end{array}
$$

First, we prove that $d _ { x } ( h , h ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) \geq \mathcal { K } \Rightarrow \varepsilon _ { \eta } \geq \varepsilon _ { \eta _ { D A } }$ . From the above, we have

$$
\begin{array} { r l } & { d _ { x } ( h ^ { * } , h ^ { * c } ) - d _ { x ^ { \prime } } ( h _ { D A } ^ { * } , h _ { D A } ^ { * c } ) \geq d _ { x } ( h ^ { * } , h ^ { * c } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) } \\ & { \qquad = d _ { x } ( h ^ { * } , h ^ { * c } ) + d _ { x } ( h ^ { * c } , h ) - d _ { x } ( h ^ { * c } , h ) } \\ & { \qquad - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) } \\ & { \qquad \geq d _ { x } ( h ^ { * } , h ) - d _ { x } ( h ^ { * c } , h ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) } \\ & { \qquad = d _ { x } ( h , h ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - \left[ d _ { x } ( h , h ^ { * c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) \right] } \end{array}
$$

By the definition of $d ,$ we have

$$
\begin{array} { r l } & { d _ { x } ( h , h ^ { * c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) = \mathbb { E } _ { x } [ | h ( x ) - h ^ { * c } ( x ) | ] + \mathbb { E } _ { x ^ { \prime } } [ | h ( x ^ { \prime } ) - h _ { D A } ^ { * c } ( x ^ { \prime } ) | ] } \\ & { \qquad \leq \mathbb { E } _ { x } [ | h ( x ) | ] + \mathbb { E } _ { x } [ | h ^ { * c } ( x ) | ] + \mathbb { E } _ { x ^ { \prime } } [ | h ( x ^ { \prime } ) | ] + \mathbb { E } _ { x ^ { \prime } } [ | h _ { D A } ^ { * c } ( x ^ { \prime } ) | ] } \\ & { \qquad \leq \mathbb { E } _ { x } [ | h ( x ) | ] + \mathbb { E } _ { x ^ { \prime } } [ | h ( x ^ { \prime } ) | ] + 2 v . } \end{array}
$$

Here, the inequalities hold due to the triangle inequality and the assumption that $h ( x ) \in$ [0, υ]. Since $d _ { x } ( h , h ^ { \ast } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { \ast } ) \geq \mathcal { K }$ , we obtain

$$
d _ { x } ( h , h ^ { * } ) - d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - [ d _ { x } ( h , h ^ { * c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) ] \geq 0 ,
$$

that is $\varepsilon _ { \eta } - \varepsilon _ { \eta _ { D A } } \geq 0$

Second, we prove that $d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x } ( h , h ^ { * } ) \geq \mathcal { K } \Rightarrow \varepsilon _ { \eta _ { D A } } \geq \varepsilon _ { \eta } .$ . Similarly, we have

$$
\begin{array} { r l } & { d _ { x ^ { \prime } } ( h _ { D A } ^ { * } , h _ { D A } ^ { * c } ) - d _ { x } ( h ^ { * } , h ^ { * c } ) \geq d _ { x ^ { \prime } } ( h _ { D A } ^ { * } , h _ { D A } ^ { * c } ) - d _ { x } ( h , h ^ { * } ) - d _ { x } ( h , h ^ { * c } ) } \\ & { \qquad \geq d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x } ( h , h ^ { * } ) - [ d _ { x } ( h , h ^ { * c } ) + d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * c } ) ] } \end{array}
$$

Based on the above results and $d _ { x ^ { \prime } } ( h , h _ { D A } ^ { * } ) - d _ { x } ( h , h ^ { * } ) \geq \mathcal { K }$ , the proof is complete. 厂

Proof of Theorem 20. Consider the following decomposition. For any $h \in \mathcal H$

$$
\begin{array} { r l } & { R ( h ) - R ^ { c } ( h ) = R ( h ) + \hat { R } ( h ) - \hat { R } ( h ) + \hat { R } ^ { c } ( h ) - \hat { R } ^ { c } ( h ) - R ^ { c } ( h ) } \\ & { \qquad = [ R ( h ) - \hat { R } ( h ) ] + [ \hat { R } ^ { c } ( h ) - R ^ { c } ( h ) ] + [ \hat { R } ( h ) - \hat { R } ^ { c } ( h ) ] . } \end{array}
$$

Then, by Lemma 35, with probability at least $1 - ( \delta / 7 )$ , we have

$$
R ( h ) - R ^ { c } ( h ) \leq 4 M \sqrt { \frac { \ln ( 7 / \delta ) } { 2 n } } + [ \hat { R } ( h ) - \hat { R } ^ { c } ( h ) ]
$$

Let $Z _ { i } = [ \hat { R } ( h ) - \hat { R } ^ { c } ( h ) ] _ { i } - [ R ( h ) - R ^ { c } ( h ) ]$ , then it follows that $Z _ { i } \in [ - 2 M , 2 M ]$ . Hereafter, we simplify the expression of $Z _ { i }$ stated in the previous proof, while preserving the same meaning. Similarly, by lemma 35, with probability at least $1 - ( \delta / 7 )$ , we have

$$
\begin{array} { l } { \displaystyle \hat { R } ( h ) - \hat { R } ^ { c } ( h ) \leq [ R ( h ) - R ^ { c } ( h ) ] + \sqrt { \frac { ( 4 M ) ^ { 2 } \ln ( 7 / \delta ) } { 2 n } } } \\ { \displaystyle \qquad \leq \sigma \Delta _ { \| \tilde { y } - y \| } + 4 M \sqrt { \frac { ( \ln ( 7 / \delta ) } { 2 n } } , } \end{array}
$$

where the last inequality follows from Lemma 34. Then, we have

$$
R ( h ) - R ^ { c } ( h ) \leq \sigma \Delta _ { \| \tilde { y } - y \| } + 8 M \sqrt { \frac { ( \ln ( 7 / \delta ) } { 2 n } } .
$$

Consider the following decomposition. For any $h \in \mathcal H$ 2

$$
\begin{array} { r l } & { R ^ { c } ( h ) - R _ { D A } ( h ) = R ^ { c } ( h ) + \hat { R } ^ { c } ( h ) - \hat { R } ^ { c } ( h ) - \hat { R } ( h ) + [ \hat { R } ( h ) - R _ { D A } ( h ) ] } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \end{array}
$$

Then, by Lemma 35 and the above, with probability at least $1 - ( \delta / 7 )$ , we have

$$
R ^ { c } ( h ) - R _ { D A } ( h ) \leq \sigma \Delta _ { \| \tilde { y } - y \| } + 8 M \sqrt { \frac { \ln ( 7 / \delta ) } { 2 n } } + [ \hat { R } ( h ) - \hat { R } _ { D A } ( h ) ] .
$$

Let $Z _ { i } \ = \ [ \hat { R } ( h ) - \hat { R } _ { D A } ( h ) ] _ { i } - [ R ( h ) - R _ { D A } ( h ) ]$ , then it follows that $Z _ { i } ~ \in ~ [ - 2 M , 2 M ]$ Similarly, by lemma 35, with probability at least $1 - ( \delta / 7 )$ , we have

$$
\begin{array} { r l r } {  { \hat { R } _ { D A } ( h ) - \hat { R } ( h ) \le [ R _ { D A } ( h ) - R ( h ) ] + \sqrt { \frac { ( 4 M ) ^ { 2 } \ln ( 7 / \delta ) } { 2 n } } } } \\ & { } & \\ & { } & { \le \sigma \Delta _ { \| x ^ { \prime } - x \| } + \sqrt { \frac { ( 4 M ) ^ { 2 } \ln ( 7 / \delta ) } { 2 n } } , } \end{array}
$$

where the last inequality follows from Lemma 34. Then we have

$$
R ^ { c } ( h ) - R _ { D A } ( h ) \leq \sigma \Delta _ { \| \tilde { y } - y \| } + \sigma \Delta _ { \| x ^ { \prime } - x \| } + 1 2 M \sqrt { \frac { \ln ( 7 / \delta ) } { 2 n } }
$$

If considering $R _ { D A } ^ { \prime } ( h ) , \sigma \Delta _ { \parallel x ^ { \prime } - x \parallel }$ can be replaced by $M \sqrt { 2 K L ( P | | P _ { D A } ^ { \prime } ) }$ . Finally, let $[ R ( h ) - R ^ { c } ( h ) ] + [ R ^ { c } ( h ) - \ddot { R } _ { D A } ( h ) ]$ , completing the proof. ■

Proof of Corollary 21. According to Theorem 20, we have

$$
\begin{array} { r l } & { R ( h ^ { * } ) - R _ { D A } ( h ^ { * } ) \leq \Omega , } \\ & { R ( h _ { D A } ^ { * } ) - R _ { D A } ( h _ { D A } ^ { * } ) \leq \Omega , } \end{array}
$$

where Ω denotes the RHS of the inequality in Theorem 20. Moreover, by Definition 1, it follows that $| R _ { D A } ( h ^ { * } ) - R ( h _ { D A } ^ { * } ) | \ \leq \ \sigma d _ { x ^ { \prime } , x } .$ The same result can also be derived by considering $R _ { D A } ^ { \prime } ( h )$ . Then, incorporating the above inequality completes the proof.

Proof of Theorem 25. Since ℓ is bounded by M, let $Z _ { ( s , i ) } = \hat { \gamma } _ { ( s , i ) } ( h _ { 1 } , h _ { 2 } ) - \gamma _ { s } ( h _ { 1 } , h _ { 2 } )$ we obtain that $Z _ { ( s , i ) } \in [ - 2 M , 2 M ]$ , where s and i denote the domain and sample indices, respectively. Then, by Lemma 36, for any $\varepsilon > 0$ , we have

$$
P \left\{ \frac { 1 } { E \bar { n } } \sum _ { s = 1 } ^ { E } \sum _ { i = 1 } ^ { \bar { n } } \hat { \gamma } _ { ( s , i ) } ( h _ { 1 } , h _ { 2 } ) - \frac { 1 } { E } \sum _ { s = 1 } ^ { E } \gamma _ { s } ( h _ { 1 } , h _ { 2 } ) \geq \varepsilon \right\} \leq e x p \left\{ - \frac { 2 ( E \bar { n } \varepsilon ) ^ { 2 } } { E \bar { n } ( 4 M ) ^ { 2 } } \right\} .
$$

Here, $\hat { \gamma } _ { ( s , i ) } ( h _ { 1 } , h _ { 2 } ) \ : = \ : \ell ( h _ { 1 } , ( x _ { i } , y _ { i } ) _ { s } ) \ : - \ : \ell ( h _ { 2 } , ( x _ { i } , y _ { i } ) _ { s } )$ and $\gamma _ { s } ( h _ { 1 } , h _ { 2 } ) = \mathbb { E } _ { P _ { s } } [ \ell ( h _ { 1 } , ( x , y ) _ { s } ) -$ $\ell ( h _ { 2 } , ( x , y ) _ { s } ) ]$ ■

Proof of Corollary 26. Based on Theorem 25, diferent types of inequalities can be derived. Similarly, we only illustrate how to decompose the relative expected risks, which as follows:

$$
\begin{array} { r l } & { \mathbb { E } [ \ell ( h _ { 1 } , z ) - \ell ( h _ { 2 } , z ) ] = \mathbb { E } [ \ell ( h _ { 1 } , z ) + \ell ( h , z ) - \ell ( h , z ) - \ell ( h _ { 2 } , z ) ] } \\ & { \qquad = \mathbb { E } [ \ell ( h _ { 1 } , z ) - \ell ( h , z ) ] + \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { 2 } , z ) ] . } \end{array}
$$

Considering the multi-domain settings, we have

$$
\sum \gamma _ { s } ( h , h _ { s } ) = \sum \gamma _ { s } ( h , h _ { s } ^ { \prime } ) + \sum \gamma _ { s } ( h _ { s } ^ { \prime } , h _ { s } ) .
$$

Note that, since h is arbitrary, we allow $h _ { s }$ and $h _ { s } ^ { \prime }$ to vary across domains. Then, based on Definition 1 and Definition 14, we rewire ${ \mathbb E } [ \ell ( h , z ) - \ell ( h _ { 2 } , z ) ]$ to $\sigma d ( h , h _ { 2 } )$ , completing the proof.

Proof of Theorem 27. The proof follows that of Theorem 17. Extending the argument to the multiple-domain setting completes the proof. ■

Proof of Theorem 28. According to Lemma 37, for any $P _ { i }$ and $Q _ { j }$ , we have

$$
\left| \mathbb { E } _ { P _ { i } } [ \ell ] - \mathbb { E } _ { Q _ { j } } [ \ell ] \right| \le 2 \sqrt { 2 } M d _ { J S } ( P _ { i } , Q _ { j } ) .
$$

Then, extending this inequality to P and $\mathbf { P _ { t } } .$ , we obtain

$$
\left| \frac { 1 } { E _ { t } } \sum _ { t = 1 } ^ { E _ { t } } \mathbb { E } _ { P _ { t } } [ \ell ] - \frac { 1 } { E } \sum _ { s = 1 } ^ { E } \mathbb { E } _ { P _ { s } } [ \ell ] \right| \leq C \sum _ { t = 1 } ^ { E _ { t } } \sum _ { s = 1 } ^ { E } d _ { J S } ( P _ { t } , P _ { s } ) ,
$$

where $C = ( 2 \sqrt { 2 } M ) / E E _ { t }$ . Let $\begin{array} { r } { \lambda = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { t } , P _ { s } ) } \end{array}$ and

$$
a = \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h , z ) - \ell ( h _ { t } ^ { * c } , z ) ]
$$

$$
b = \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h _ { t } ^ { * c } , z ) ]
$$

$$
c = \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h , z ) - \ell ( h _ { s } ^ { * c } , z ) ]
$$

$$
d = \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h _ { s } ^ { * c } , z ) ] .
$$

Applying the above inequality yields

$$
| ( a + b ) - ( c + d ) | = | ( a - c ) - ( d - b ) | \leq C \lambda .
$$

By the triangle inequality, we have

$$
\begin{array} { c } { { | a - c | - | d - b | \le | ( a - c ) - ( d - b ) | \le C \lambda } } \\ { { | a - c | \le | d - b | + C \lambda } } \end{array}
$$

Similarly, it follows from Lemma 37 that

$$
| d - b | = \left| \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h _ { s } ^ { * c } , z ) ] - \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h _ { t } ^ { * c } , z ) ] \right| \le C \lambda ^ { * c } ,
$$

where $\begin{array} { r } { \lambda ^ { * c } = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { h _ { t } ^ { * c } } , P _ { h _ { s } ^ { * c } } ) } \end{array}$ . This implies that

$$
| a - c | \leq C \lambda ^ { * c } + C \lambda .
$$

We now examine $| a - c |$ and consider

$$
\begin{array} { r l } & { \displaystyle \mathfrak { a } = \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } [ \ell ( h _ { t } ^ { * } , z ) + \ell ( h _ { t } ^ { * } , z ) - \ell ( h _ { t } ^ { * } , z ) - \ell ( h _ { t } ^ { * } , z ) ] } \\ & { \displaystyle = \underbrace { \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { t } ^ { * } , z ) ] } _ { \mathfrak { a } _ { 1 } } - \underbrace { \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } [ \ell ( h _ { t } ^ { * \mathfrak { c } } , z ) - \ell ( h _ { t } ^ { * } , z ) ] } _ { \mathfrak { a } _ { 2 } } } \\ & { \displaystyle = - \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) + \ell ( h _ { s } ^ { * } , z ) - \ell ( h _ { s } ^ { * } , z ) - \ell ( h _ { s } ^ { * } , z ) ] } \\ & { \displaystyle = \underbrace { \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { s } ^ { * } , z ) ] } _ { \mathfrak { c } _ { 1 } } - \underbrace { \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h _ { s } ^ { * \mathfrak { c } } , z ) - \ell ( h _ { s } ^ { * } , z ) ] } _ { \mathfrak { c } _ { 2 } } . } \end{array}
$$

Similarly, applying the triangle inequality $| a - c | \geq | a _ { 1 } - a _ { 2 } | - | c _ { 1 } - c _ { 2 } |$ , yields

$$
| a _ { 1 } | \leq | a _ { 2 } | + | c _ { 1 } | + | c _ { 2 } | + C \lambda ^ { * c } + C \lambda .
$$

Now, substituting $a _ { 2 } = \sigma \varepsilon _ { \eta _ { t } }$ and $c _ { 2 } = \sigma \varepsilon _ { \eta }$ yields

$$
\frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { t } ^ { * } , z ) ] \leq \sigma \varepsilon _ { \eta _ { t } } + \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { s } ^ { * } , z ) ] + \sigma \varepsilon _ { \eta } + C \lambda ^ { * c } + C \lambda .
$$

Now, by Lemma 36, the following inequality holds with probability at least $1 - \delta \colon$

$$
\frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { s } ^ { * } , z ) ] \leq \frac { 1 } { E \bar { n } } \sum _ { s } \hat { \gamma } ( h , h _ { s } ^ { * } ) + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 E \bar { n } } } ,
$$

which completes the proof.

Proof of Corollary 29. Based on the proof of Theorem 28, let

$$
a = \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h , z ) - \ell ( h _ { t } ^ { * c } , z ) ]
$$

$$
b = \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h _ { t } ^ { * c } , z ) ]
$$

$$
c = \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h , z ) - \ell ( h _ { D A s } ^ { * c } , z ) ]
$$

$$
d = \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h _ { D A s } ^ { * c } , z ) ] .
$$

Then, it similarly follows from Lemma 37 that

$$
\left| d - b \right| = \left| \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h _ { D A s } ^ { * c } , z ) ] - \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h _ { t } ^ { * c } , z ) ] \right| \leq C \lambda _ { D A } ^ { * c } ,
$$

where $\begin{array} { r } { \lambda _ { D A } ^ { * c } = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { h _ { t } ^ { * c } } , P _ { h _ { D A s } ^ { * c } } ) } \end{array}$ . Then, we consider

$$
\begin{array} { l } { \displaystyle { c = \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) + \ell ( h _ { D A s } ^ { * } , z ) - \ell ( h _ { D A s } ^ { * } , z ) - \ell ( h _ { D A s } ^ { * c } , z ) ] . } } \\ { \displaystyle { = \underbrace { \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { D A s } ^ { * } , z ) ] } _ { c _ { 1 } } - \underbrace { \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h _ { D A s } ^ { * c } , z ) - \ell ( h _ { D A s } ^ { * } , z ) ] } _ { c 2 } . } } \end{array}
$$

Now, substituting $c _ { 2 } = \sigma \varepsilon _ { \eta _ { D A } }$ yields

$$
\begin{array} { r l r } {  { \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { t } ^ { * } , z ) ] \le \sigma \varepsilon _ { \eta _ { t } } + \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { D A s } ^ { * } , z ) ] } } \\ & { } & \\ & { } & { + \sigma \varepsilon _ { \eta _ { D A } } + C \lambda _ { D A } ^ { * c } + C \lambda _ { D A } , } \end{array}
$$

where, $\begin{array} { r } { \lambda _ { D A } = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { t } , P _ { D A s } ) } \end{array}$ . Now, by Lemma 36, the following inequality holds with probability at least $1 - \delta \colon$

$$
\frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { D A s } ^ { \ast } , z ) ] \leq \frac { 1 } { E \bar { n } } \sum _ { s } \hat { \gamma } ( h , h _ { D A s } ^ { \ast } ) + 4 M \sqrt { \frac { \ln ( 1 / \delta ) } { 2 E \bar { n } } } ,
$$

which completes the proof.

Proof of Theorem 30. By Lemma 38, for any $i \neq j \in [ E ]$ , we have

$$
K L ( P _ { D A i } | | P _ { D A j } ) \leq K L ( P _ { i } | | P _ { j } ) .
$$

Let $M _ { D A } = ( P _ { D A _ { i } } + P _ { D A _ { j } } ) / 2$ and $M = ( P _ { i } + P _ { j } ) / 2$ . Then, still applying Lemma 38, we obtain

$$
\begin{array} { r } { K L ( P _ { D A i } | | M _ { D A } ) \leq K L ( P _ { i } | | M ) . } \\ { K L ( P _ { D A j } | | M _ { D A } ) \leq K L ( P _ { j } | | M ) , } \end{array}
$$

which implies that

$$
\begin{array} { c } { { K L ( P _ { D A _ { i } } | | M _ { D A } ) + K L ( P _ { D A _ { j } } | | M _ { D A } ) \le K L ( P _ { i } | | M ) + K L ( P _ { j } | | M ) } } \\ { { { \displaystyle \frac { 1 } { 2 } \left( K L ( P _ { D A _ { i } } | | M _ { D A } ) + K L ( P _ { D A _ { j } } | | M _ { D A } ) \right) \le \frac { 1 } { 2 } \left( K L ( P _ { i } | | M ) + K L ( P _ { j } | | M ) \right) . } } } \end{array}
$$

Then, by Definition 23, we have

$$
\begin{array} { c } { { \displaystyle J S D ( P _ { D A 1 } , \ldots , P _ { D A E } ) = \frac { 1 } { E } \sum _ { i = 1 } ^ { E } K L ( P _ { D A i } | | { \cal M } _ { D A } ) } } \\ { { \displaystyle J S D ( P _ { 1 } , \ldots , P _ { E } ) = \frac { 1 } { E } \sum _ { i = 1 } ^ { E } K L ( P _ { i } | | { \cal M } ) , } } \end{array}
$$

where $M _ { D A } = ( 1 / E ) \sum P _ { D A i }$ and $M = ( 1 / E ) \sum P _ { i }$ , completing the proof.

Proof of Theorem 31. By identifying $\{ P _ { 1 } , \ldots , P _ { E } \}$ with $\{ P _ { h _ { 1 } ^ { * } } , \ldots , P _ { h _ { E } ^ { * } } \}$ , estimating $\mathbf { 1 } _ { \{ \phi ( X ) \in P _ { i } \} } , \forall i \in [ E ]$ , is equivalent to estimating $\mathbf { 1 } _ { \{ \phi ( X ) \in P _ { h _ { i } ^ { * } } \} } , \forall i \in [ \bar { E } ]$ . Then, according to Definition $2 2 , \phi \neq h _ { i } ^ { * }$ implies

$$
\phi ( X ) \neq h _ { i } ^ { * } ( X ) \Leftrightarrow d ( \phi , h _ { i } ^ { * } ) \geq \epsilon .
$$

Then, by applying Lemma 39, the proof is complete. Note that we do not consider the case where the $n _ { i }$ samples are i.i.d. observations drawn from domain i. Incorporating this setting would further refine the lower bound through its dependence on $n _ { i }$ . However, omitting this consideration does not afect the main insights of our analysis. ■

Proof of Theorem 32. According to Lemma 37, for any $P _ { i }$ and $Q _ { j }$ , we have

$$
\left| \mathbb { E } _ { P _ { i } } [ \ell ] - \mathbb { E } _ { Q _ { j } } [ \ell ] \right| \le 2 \sqrt { 2 } M d _ { J S } ( P _ { i } , Q _ { j } ) .
$$

Then, extending this inequality to P and $\mathbf { P _ { t } } ,$ we obtain

$$
\left| \frac { 1 } { E _ { t } } \sum _ { t = 1 } ^ { E _ { t } } \mathbb { E } _ { P _ { t } } [ \ell ] - \frac { 1 } { E } \sum _ { s = 1 } ^ { E } \mathbb { E } _ { P _ { s } } [ \ell ] \right| \leq C \sum _ { t = 1 } ^ { E _ { t } } \sum _ { s = 1 } ^ { E } d _ { J S } ( P _ { t } , P _ { s } ) ,
$$

where $C = ( 2 \sqrt { 2 } M ) / E E _ { t }$ . Let $\begin{array} { r } { \lambda = \sum _ { t } \sum _ { s } d _ { J S } ( P _ { t } , P _ { s } ) } \end{array}$ and

$$
a = \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h , z ) - \ell ( h _ { t } ^ { * } , z ) ]
$$

$$
b = \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h _ { t } ^ { * } , z ) ]
$$

$$
c = \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h , z ) - \ell ( h _ { s } ^ { * } , z ) ]
$$

$$
d = \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h _ { s } ^ { * } , z ) ] .
$$

Applying the above inequality and the triangle inequality yields that

$$
| a - c | \leq | d - b | + C \lambda .
$$

By lemma 37, we also have

$$
| d - b | = \left| \frac { 1 } { E } \sum _ { s } \mathbb { E } _ { z \sim P _ { s } } [ \ell ( h _ { s } ^ { * } , z ) ] - \frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } _ { z \sim P _ { t } } [ \ell ( h _ { t } ^ { * } , z ) ] \right| \le C \lambda .
$$

This follows that

$$
| a - c | \leq 2 C \lambda ,
$$

that is

$$
\frac { 1 } { E _ { t } } \sum _ { t } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { t } ^ { \ast } , z ) ] \leq \frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { s } ^ { \ast } , z ) ] + 2 C \lambda .
$$

Then, by Definition 1, it follows that

$$
\frac { 1 } { E } \sum _ { s } \mathbb { E } [ \ell ( h , z ) - \ell ( h _ { s } ^ { \ast } , z ) ] \leq \sigma \frac { 1 } { E } \sum _ { s } d _ { P _ { s } } ( h , h _ { s } ^ { \ast } ) ,
$$

completing the proof.

## References

A. Antoniou, A. Storkey, and H. Edwards. Data augmentation generative adversarial networks. arXiv preprint arXiv:1711.04340, 2017.

H. S. Baird. Document image defect models. In Structured Document Image Analysis, pages 546–556. Springer, 1992.

S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan. A theory of learning from diferent domains. Machine learning, 79(1):151–175, 2010.

S. Bernstein. The theory of probabilities, 1946.

L. Birg´e. On estimating a density using hellinger distance and some other strange facts. Probability theory and related fields, 71(2):271–291, 1986.

O. Bousquet and A. Elisseef. Stability and generalization. Journal of machine learning research, 2(Mar):499–526, 2002.

S. Chen, E. Dobriban, and J. H. Lee. A group-theoretic framework for data augmentation. Journal of Machine Learning Research, 21(245):1–71, 2020.

T. Dao, A. Gu, A. Ratner, V. Smith, C. De Sa, and C. R´e. A kernel theory of modern data augmentation. In International conference on machine learning, pages 1528–1537. PMLR, 2019.

K. Donhauser, M. Wu, and F. Yang. How rotational invariance of common kernels prevents generalization in high dimensions. In International Conference on Machine Learning, pages 2804–2814. PMLR, 2021.

I. Gao, S. Sagawa, P. W. Koh, T. Hashimoto, and P. Liang. Out-of-domain robustness via targeted augmentations. In International conference on machine learning, pages 10800– 10834. PMLR, 2023.

A. Han, W. Huang, Z. Zhou, G. Niu, W. Chen, J. Yan, A. Takeda, and T. Suzuki. On the role of label noise in the feature learning process. In International Conference on Machine Learning, pages 21733–21773. PMLR, 2025.

B. Han, Q. Yao, X. Yu, G. Niu, M. Xu, W. Hu, I. Tsang, and M. Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. Advances in neural information processing systems, 31, 2018.

W. Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American statistical association, 58(301):13–30, 1963.

S.-H. Hwang, M. Kim, and S. E. Whang. Rc-mixup: A data augmentation strategy against noisy data for regression tasks. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1155–1165, 2024.

D. LeJeune, R. Balestriero, H. Javadi, and R. G. Baraniuk. Implicit rugosity regularization via data augmentation. arXiv preprint arXiv:1905.11639, 2019.

C.-H. Lin, C. Kaushik, E. L. Dyer, and V. Muthukumar. The good, the bad and the ugly sides of data augmentation: An implicit spectral regularization perspective. Journal of Machine Learning Research, 25(91):1–85, 2024.

P. Massart and E. N´ed´elec. Risk bounds for statistical learning. <sup>´</sup> The Annals of Statistics, 34(5):2326–2366, 2006.

N. Natarajan, I. S. Dhillon, P. K. Ravikumar, and A. Tewari. Learning with noisy labels. Advances in neural information processing systems, 26, 2013a.

N. Natarajan, I. S. Dhillon, P. K. Ravikumar, and A. Tewari. Learning with noisy labels. Advances in neural information processing systems, 26, 2013b.

E. Pereira, G. Carneiro, and F. R. Cordeiro. A study on the impact of data augmentation for training convolutional neural networks in the presence of noisy labels. In 2022 35th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), volume 1, pages 25–30. IEEE, 2022.

Y. Polyanskiy and Y. Wu. Lecture notes on information theory. Lecture Notes for ECE563 (UIUC) and, 6(2012-2016):7, 2014.

P. Tiwary, K. Bhattacharyya, et al. Langdaug: Langevin data augmentation for multi-source domain generalization in medical image segmentation. arXiv preprint arXiv:2505.19659, 2025.

V. N. Vapnik and A. Y. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. In Measures of complexity: festschrift for alexey chervonenkis, pages 11–30. Springer, 2015.

Y. Wang, Y. Wu, and H. Zhang. Lost domain generalization is a natural consequence of lack of training domains. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 15689–15697, 2024.

H. Yao, Y. Wang, L. Zhang, J. Y. Zou, and C. Finn. C-mixup: Improving generalization in regression. Advances in neural information processing systems, 35:3361–3376, 2022.

H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018.

Z. Zhang, H. Zhang, S. O. Arik, H. Lee, and T. Pfister. Distilling efective supervision from severe label noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9294–9303, 2020.

H. Zheng and F. Teng. Distribution shift is key to learning invariant prediction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, page 28812–28820, Mar. 2026.

Y. Zhou, C. Guo, X. Wang, Y. Chang, and Y. Wu. A survey on data augmentation in large model era. arXiv preprint arXiv:2401.15422, 2024.