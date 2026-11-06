# Neural Collapse on Imbalanced Datasets and Open-Set Scenario

Qiufu Li liqiufu@szu.edu.cn Huibin Xiao hb xiao@outlook.com Linlin Shen llshen@szu.edu.cn

School of Artificial Intelligence; National Engineering Laboratory for Big Data System Computing Technology;

Guangdong Provincial Key Laboratory of Intelligent Information Processing; Intelligent Computing Center,

Shenzhen University, Shenzhen 518060, China

Yongdong Huang

huang yongdong@163.com

Institute of Image Understanding Research, North Minzu University, Yinchuan 750021, China

## Abstract

Neural collapse (NC) describes the elegant geometric structure that sample features and classifier vectors exhibit when a classification model’s loss reaches its minimum. Class imbalance, however, generally distorts standard NC. In this paper, we theoretically reformulate NC on imbalanced datasets and in open-set settings. Our analysis is based on an optimization problem over class-wise averaged losses, rather than the global averaged losses used in prior NC studies. We prove that when class-wise averaged losses attain their min ima, standard NC is achieved on balanced or imbalanced datasets, for both cross-entropy (CE) and binary cross-entropy (BCE) losses. We further extend the analysis to an openset-inspired scenario, where some classes contain no training samples yet are still assigned classifier vectors in the losses. We find that in this scenario, when class-wise averaged losses reach their minima, standard NC still emerges on the known classes with training samples, while the classifier vectors of unknown classes without training samples overlap with each other and become orthogonal to those of the known classes. Finally, we analyze Softmaxand Sigmoid-based contrastive losses re-balanced by inverse sample frequency, proving that at their minima on balanced or imbalanced datasets, sample features of diferent classes likewise form standard NC.

Keywords: neural collapse, class-wise averaged loss, imbalanced datasets, open-set scenario, contrastive learning

## 1 Introduction

Neural collapse (NC) (Papyan et al., 2020; Hong and Ling, 2026) refers to the elegant geometric structure formed by the classifier vectors and sample feature vectors at the terminal phase of training a classification model. Under NC, feature vectors from the same class converge to the class center and align with the corresponding classifier vector, while the class centers are maximally separated and arranged as an equiangular tight frame (ETF). The NC geometry theoretically corresponds to maximizing within-class compactness and between-class separability of the learned features. On balanced datasets, extensive empirical and in-depth theoretical researches have shown that NC occurs when cross-entropy (CE)

loss (Lu and Steinerberger, 2022) or binary cross-entropy (BCE) loss (Li et al., 2025b) reach their minimum values in training classification models.

In real-world scenarios, sample distributions are typically class-imbalanced. When training classification models on imbalanced datasets, the training process and the final models performance are easily dominated by majority classes containing large numbers of training samples, making it dificult for the final models to learn well discriminative features and classifier vectors for minority classes containing few training samples (Alshammari et al., 2022; Kang et al., 2020). NC provides guidance for empirical classification studies on imbalanced datasets. To enhance the feature properties and classification accuracy for minority classes of imbalanced datasets, in studies by Yang et al. (2022); Kasarla et al. (2022); Liu et al. (2023a), the classifier vectors were fixed before the model training so that they form an ETF with balanced between-class separability, and a learnable rotation matrix was employed in the work of Peifeng et al. (2023) during the model training to rotate the ETF classifier and align it with the final sample features. However, the theoretical studies shows that the sample features and classifier vectors of unconstrained feature models (UFM) (Fang et al., 2021), trained on imbalanced datasets, cannot attain the standard NC and therefore cannot reach the ETF with maximal and balanced within-class compactness and between-class separability.

On imbalanced datasets that contain only two distinct class scales, when the imbalance factor exceeding a finite threshold, the CE losses’ minimizers lead to minority collapse, which means that the classifier vectors of minority classes will collapse into a single one (Fang et al., 2021); the sharp value of the imbalance ratio threshold was given by Hong and Ling (2024). For contrastive learning on imbalanced datasets, it also results in similar minority collapse (Yi et al., 2025). On general imbalanced datasets, when CE or MSE loss of classification models is minimized, the features from the same class still collapse to their class center, while the norms and directions of the class-center vectors and classifier vectors for diferent classes depend on the number of samples in each class, exhibiting uneven between-class separability and resulting in asymmetric geometric structures, such as simplex-encodedlabels interpolation (SELI) geometry (Thrampoulidis et al., 2022), (δ, R)-SELI geometry (Behnia et al., 2023), and general orthogonal frame (GOF) (Dang et al., 2023), et al.

Some useful techniques can balance the between-class separability of features learned on imbalanced datasets. If a ReLU activation was integrated into the final layer of the model, then when the supervised contrastive loss (Kini et al., 2024) or CE loss (Dang et al., 2024) attain its minima, feature vectors from diferent classes become orthogonal to each other. Two NC regularization terms were proposed by Liu et al. (2023a) to equalize feature properties, and empirical results show that integrating them into the CE or MSE loss enables obtaining sample features whose distribution exhibits a symmetric structure on imbalanced datasets. A generalized classification rule was proposed for classification on imbalanced datasets, which assigns multiple classifier vectors to diferent classes, and with this rule, the sample features and classifier vectors form NC to multiple centers (NCMC) when the MSE loss reaches its minimum (Yan et al., 2024).

Most existing NC studies are based on the globally averaged losses, which have shown that the class imbalance distorts the standard NC at the extrema of the losses. More recently, in the work of Wang et al. (2026), the authors solved for the optimal reweighted loss with Tikhonov regularizer on imbalanced datasets and provided a closed-form solution for the optimal weighting factors that form the ideal NC. Compatible with their conclusion, in this paper we directly demonstrate that by re-balancing the losses using the number of samples in each class and optimizing the resulted class-wise averaged losses, the standard NC can be achieved on imbalanced datasets, and even in open-set scenarios.

## 2 Preliminaries

Suppose $\textstyle { \mathcal { D } } = \bigcup _ { k = 1 } ^ { K } { \mathcal { D } } _ { k }$ be a dataset captured from K classes, where its subset $\mathcal { D } _ { k }$ contains $n _ { k } = | \mathcal { D } _ { k } |$ samples from class k, for $\forall k \in [ K ] = \{ 1 , 2 , \cdot \cdot \cdot , K \}$ . For any sample $\pmb { X } \in \mathcal { D }$ , a model M extracts its feature vector $\pmb { x } = \mathcal { M } ( \pmb { X } ) \in \mathbb { R } ^ { d } ;$ then $\begin{array} { r } { \mathcal { F } = \bigcup _ { k = 1 } ^ { K } \mathcal { F } _ { k } = \bigcup _ { k = 1 } ^ { K } \{ \pmb { x } = } \end{array}$ $\mathcal { M } ( \pmb { X } ) : \pmb { X } \in \mathcal { D } _ { k } \}$ denotes the feature set. For convenience, the feature vector is normalized into unit ball with $\| \pmb { x } \| = 1$

## 2.1 Classification

To classify a sample X, a linear classifier with a weight matrix $\ b { W } = [ \ b { w } _ { 1 } , \cdots , \ b { w } _ { K } ] ^ { \top } \in \mathbb { R } ^ { K \times d }$ and a bias vector $\pmb { b } = [ b _ { 1 } , \cdots , b _ { K } ] ^ { \top } \in \mathbb { R } ^ { K }$ , is first applied on its feature vector $\pmb { x } = \mathcal { M } ( \pmb { X } )$ and transforms it into a decision vector $\begin{array} { r } { \boldsymbol { z } = \gamma W \boldsymbol { x } - \boldsymbol { b } = [ \gamma \langle \pmb { w } _ { k } , \pmb { x } \rangle - b _ { k } ] _ { k = 1 } ^ { K } } \end{array}$ , where $\gamma$ is a scale factor, $\| \pmb { w } _ { k } \| = 1$ for $\forall k \in [ K ]$ , and $\langle \cdot , \cdot \rangle$ is operation of inner product for any two vectors. Then, X is classified into class $\hat { k } = \arg \operatorname* { m a x } _ { j } \{ \gamma \langle { \pmb w } _ { j } , { \pmb x } \rangle - b _ { j } \} _ { j = 1 } ^ { K }$ , where $\boldsymbol { w } _ { k } \in \mathbb { R } ^ { d }$ is classifier vector for class k. To correctly classify a sample $X ^ { ( k ) } \in \mathcal { D } _ { k }$ from class $k ,$ it requires

$$
\gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \geq \operatorname* { m a x } \big \{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } : j \in [ K ] , j \neq k \big \} ,\tag{1}
$$

where $\pmb { x } ^ { ( k ) } = \mathcal { M } ( \pmb { X } ^ { ( k ) } )$ is feature vector of the sample $X ^ { ( k ) }$

In training the classification model, using Softmax, the decision vector $z ^ { ( k ) } ~ =$ $[ \gamma \langle { \pmb w } _ { k } , { \pmb x } ^ { ( k ) } \rangle - b _ { k } ] _ { k = 1 } ^ { K }$ of $\pmb { X } ^ { ( k ) } \in \mathcal { D } _ { k }$ is transformed into probabilities that the sample belong to each class, and then the probabilities are applied to calculate a loss using cross-entropy (CE),

$$
L _ { \mathrm { c e } } ( \mathbf { \boldsymbol { X } } ^ { ( k ) } ) = - \log \bigg ( \frac { \exp \big ( \gamma \langle \boldsymbol { w } _ { k } , \boldsymbol { x } ^ { ( k ) } \rangle - b _ { k } \big ) } { \sum _ { j = 1 } ^ { K } \exp \big ( \gamma \langle \boldsymbol { w } _ { j } , \boldsymbol { x } ^ { ( k ) } \rangle - b _ { j } \big ) } \bigg ) ,\tag{2}
$$

which has been widely used in deep learning.

## 2.2 Feature learning

In deep learning, Softmax-based contrastive losses, such as SupCon Khosla et al. (2020) and InfoNCE He et al. (2020), are commonly used to train feature extraction models. However, for feature learning, it expects good within-class compactness and between-class separability among the sample feature vectors. Intuitively, for feature vectors of class k, taking its classifier vector ${ \pmb w } _ { k }$ as an anchor, the high compactness requires

$$
\gamma \langle { \pmb w } _ { k } , { \pmb x } ^ { ( k ) } \rangle > b _ { k } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } ,\tag{3}
$$

for the arbitrary reasonable threshold $b _ { k }$ . Using Sigmoid, one can calculate the probability $p _ { k } ( { \pmb x } ^ { ( k ) } ) = \sigma ( \gamma \langle { \pmb w } _ { k } , { \pmb x } ^ { ( k ) } \rangle - b _ { k } )$ that the compactness condition (3) holds, and binary crossentropy (BCE) could be used to calculate a loss, − log $( p _ { k } ( \pmb { x } ^ { ( k ) } ) )$ , for the model training.

Similarly, for high separability between class k and class $j ,$ taking ${ \pmb w } _ { j }$ as the anchor, it wishes

$$
\gamma \langle { \pmb w } _ { j } , { \pmb x } ^ { ( k ) } \rangle < b _ { j } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } ,\tag{4}
$$

for the arbitrary reasonable threshold $b _ { j }$ . Then, one can calculate the probability $p _ { j } ( \pmb { x } ^ { ( k ) } ) =$ $1 - \sigma ( \gamma \langle { \pmb w } _ { j } , { \pmb x } ^ { ( k ) } \rangle - b _ { j } )$ that the separability condition (4) holds, which could be used to calculate a BCE loss, − log $\left( p _ { j } ( \pmb { x } ^ { ( k ) } ) \right)$ , for model training.

Totally, for the feature learning, a multi-class BCE loss for the sample $X ^ { ( k ) }$ is

$$
L _ { \mathrm { b c e } } ( X ^ { ( k ) } ) = \log \big ( 1 + \mathrm { e } ^ { - \gamma \langle w _ { k } , x ^ { ( k ) } \rangle + b _ { k } } \big ) + \sum _ { j = 1 , j \ne k } ^ { K } \log \big ( 1 + \mathrm { e } ^ { \gamma \langle w _ { j } , x ^ { ( k ) } \rangle - b _ { j } } \big ) ,\tag{5}
$$

which has been widely used in multi-label classification, and we here reveal its relationship to the feature learning.

## 2.3 NC on balanced datasets

On the whole dataset ${ \mathcal { D } } ,$ using unconstrained feature model or layer-peeled model Fang et al. (2021), the existing works analyze the minimization of loss functions,

$$
\hat { f } _ { \mu } ( \boldsymbol { W } , b , \mathcal { F } ) = \frac { 1 } { N } \sum _ { k = 1 } ^ { K } \sum _ { \boldsymbol { X } ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mu } ( \boldsymbol { X } ^ { ( k ) } ) ,\tag{6}
$$

where $\mu \in \{ \mathrm { c e } , \mathrm { b c e } \}$ and $\begin{array} { r } { N = \sum _ { k = 1 } ^ { K } n _ { k } } \end{array}$ . On the balanced datasets with $n = n _ { k } = | \mathcal Ḋ D Ḍ _ { k } |$ for $\forall k \in [ K ]$ , the current works have proved that the CE Lu and Steinerberger (2022) and BCE Li et al. (2025b) losses lead to standard neural collapse (NC) when reaching their minima in training models. The standard NC means that,

• within the same class k for $\forall k \in [ K ]$ , the sample feature vectors $\mathcal { F } _ { k } = \{ \pmb { x } = \mathcal { M } ( \pmb { X } )$ $\pmb { X } \in \mathcal { D } _ { k } \}$ collapse to their class center $\begin{array} { r } { \bar { \mathbf { x } } _ { k } = \frac { \bar { 1 } } { n _ { k } } \displaystyle \sum _ { \mathbf { x } \in \mathcal { F } _ { k } } \mathbf { \Delta x } , { \mathrm { i . e . , } } ~ \mathbf { x } = \bar { \mathbf { x } } _ { k } } \end{array}$ for $\forall x \in \mathcal { F } _ { k }$

• for any class $k ,$ , its class center align with the classifier vector, i.e., $\bar { \mathbfit { x } } _ { k } ~ = ~ \mathbfit { w } _ { k }$ for $\forall k \in [ K ]$ ，

• between the classes, the angles of diferent class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ are equal and maximized, forming an ETF, i.e., $\bar { \pmb x } _ { k } ^ { \top } \bar { \pmb x } _ { j } = - \frac { 1 } { K - 1 }$ for $\forall k \neq j \in [ K ]$

When the losses reach their minima, the classifier biases become equal, i.e., $b _ { j } = b _ { k } = b ^ { \star } \in \mathbb { R }$ Then, the standard NC means that, for any sample, it satisfies classification condition (1) with maximum possible likelihood; meanwhile, across the entire dataset, the withinclass compactness of sample features and the between-class separability are both uniformly maximized. In other words, the classification and feature extraction models trained with CE or BCE possess the optimal theoretical expectations.

## 3 Main Results

## 3.1 Neural Collapse on Imbalanced Datasets

On imbalanced datasets, when the globally averaged losses defined in Eq. (6) reach their minimum values, it is found that the standard NC cannot be achieved. In this case, although the feature vectors of samples within the same class can still collapse to the class center, the class center is not necessarily aligned with its classifier vector, and the angle between classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ and that between class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ vary with the number of samples contained in the corresponding classes. In practice, the training of models and their final performance are often dominated by the majority classes containing large numbers of training samples. Although various strategies have been proposed to rebalance their training and performance across classes, none of them can theoretically reformulate the standard NC on imbalanced datasets.

It can be observed that the losses in Eq. (6) are global mean of the CE and BCE losses on the entire dataset D. In this paper, we define class-wise averaged losses,

$$
f _ { \mu } ( W , b , \mathcal { F } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mu } ( X ^ { ( k ) } ) ,\tag{7}
$$

which first computes the averaged loss for each class and then calculates the total averaged loss over the K classes. We prove that, on an imbalanced dataset, the class-wise averaged CE or BCE loss lead to standard NC when reaching their minimum values.

Theorem 1 On a dataset D containing K classes, each class contains at least one sample, i.e., $n _ { k } = | \mathcal D _ { k } | > 0$ for $\forall k \in [ K ]$ . Suppose that the feature vectors extracted by the model M have dimensionality $d \geq K - 1$ . Then, when the losses $f _ { \mu } ( W , b , { \mathcal { F } } )$ defined in $E q .$ . (7) achieving their minima, they satisfy

$$
\pmb { x } ^ { ( k ) } = \pmb { w } _ { k } , \quad \forall \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] ,\tag{8}
$$

$$
\langle { \pmb w } _ { k } , { \pmb w } _ { j } \rangle = - \frac { 1 } { K - 1 } , \quad \forall k \neq j \in [ K ] ,\tag{9}
$$

$$
b _ { k } = b _ { j } = b ^ { \star } , \quad \forall k , j \in [ K ] ,\tag{10}
$$

while, for $\mu =$ bce, b<sup>⋆</sup> is the unique real solution of

$$
\mathrm { e } ^ { \frac { \gamma } { K - 1 } } \mathrm { e } ^ { 2 b } - ( K - 2 ) \mathrm { e } ^ { b } - ( K - 1 ) \mathrm { e } ^ { \gamma } = 0 .\tag{11}
$$

and for $\mu = \mathrm { c e } , b ^ { \star } \in \mathbb { R }$ could be arbitrary number.

Proof The proofs are provided in Appendix, i.e., Theorems S.4 (BCE) and S.6 (CE) .

Eq. (8) implies that the feature vectors of samples of the k-th class not only collapse to their class center but also align with their classifier vector, ${ \pmb w } _ { k } ;$ ; Eq. (9) signifies that both the class centers and the classifier vectors form ETF. In total, according to Theorem 1, training classification models on dataset D, balanced or imbalanced, minimizing the class-wise averaged losses induces standard NC.

For the CE and BCE, the diference between their minimizations lies in the classifier biases. At a minimum point of CE, the classifier biases can take any value as long as the K biases are equal; therefore, setting the classifier biases to 0 in the CE still yields the standard NC. In contrast, at the minimum point of BCE, the classifier biases have only a unique value, which means that the classifier biases play a substantive role in training models using BCE. Furthermore, when removing the classifier biases in BCE, the final classifier vector ${ \pmb w } _ { k }$ and class center $\bar { \mathbfit { x } } _ { k }$ cannot be aligned, failing to form a standard NC, except for the case K = 2 where its theoretical optimal bias is zero.

## 3.1.1 The NC-optimal Re-balancing Strategy

When training models on imbalanced datasets, practitioners commonly use re-balancing strategies such as re-sampling and re-weighting to achieve good performance in practice. Theorem 1 implies that re-weighting the loss using the inverse frequency, $\frac { 1 } { n _ { k } }$ , is theoretically the NC-optimal re-balancing strategy.

In practice, logit adjustment (LA) has been a commonly used re-balancing strategy in model training on imbalanced datasets. For each sample X, LA typically modulates the diference between its positive and negative logits (i.e., the decision scores), or adjust its logits using the sample number corresponding to the class to which the classifier vectors belong, such as

$$
L _ { \mathrm { c e - l a } } ( \boldsymbol { X } ^ { ( k ) } ) = - \log \bigg ( \frac { \mathrm { e } ^ { ( \gamma w _ { k } ^ { \top } \boldsymbol { x } ^ { ( k ) } - b _ { k } ) / \delta _ { k k } } } { \sum _ { j = 1 } ^ { K } \mathrm { e } ^ { ( \gamma w _ { j } ^ { \top } \boldsymbol { x } ^ { ( k ) } - b _ { j } ) / \delta _ { k j } } } \bigg ) ,
$$

$$
L _ { \mathrm { b c e - l a } } ( \pmb X ^ { ( k ) } ) = \log \Big ( 1 + \mathrm { e } ^ { ( - \gamma w _ { k } ^ { \top } \pmb x ^ { ( k ) } + b _ { k } ) / \delta _ { k k } } \Big ) + \sum _ { j = 1 , j \neq k } ^ { K } \log \Big ( 1 + \mathrm { e } ^ { ( \gamma w _ { j } ^ { \top } \pmb x ^ { ( k ) } - b _ { j } ) / \delta _ { k j } } \Big ) ,
$$

where $\delta _ { k j }$ is typically a quantity related to $n _ { k }$ and $n _ { j }$ .

The LA strategy operates at the individual sample level, adjusting the similarities between the feature vector and the imbalanced classifier vectors within the loss function. When the LA loss reaches its minimum, the feature and classifier vectors will form a certain geometric structure, but there is a failure to prove that they would form a standard NC.

## 3.2 Neural Collapse in Open-set Scenarios

In Fang et al. (2021), the authors proved that minority collapse occurs when the imbalance factor (IF) tends to infinity. In practice, since the number of samples in majority classes cannot be infinite, an IF approaching infinity implies that the minority classes contain zero samples, shifting the classification task into open-set recognition (OSR). OSR refers to a classification task in which the test or validation set contains samples from classes that do not appear in the training set, meaning that some classes have zero training samples when optimizing the classification loss. In the loss function for training OSR model, assigning classifier placeholders (Zhou et al., 2021) or dummy classifiers (Liu et al., 2023b) for unknown classes can improve OSR performance. NC structure was also used to develop OSR method Li et al. (2025a). In an open-set setting, this paper investigates the geometry of sample features and classifier vectors at the loss minimum.

In our open-set setting, we assume that the first $K _ { 1 }$ classes in the dataset D are known classes containing non-zero training samples, with $1 < K _ { 1 } < K$ , while the remaining $K - K _ { 1 }$ classes are unknown classes containing zero training samples. For each class $\ell \in [ K ]$ , regardless of that it is known or unknown class, we assign a classifier vector ${ \pmb w } _ { \ell }$ and bias $b _ { \ell }$ to it in training models. Then, for any sample $X ^ { ( k ) }$ from a known class $k \in [ K _ { 1 } ]$ its CE and BCE losses are still expressed by Eqs. (2) and (5).

Theorem 2 For dataset $\textstyle { \mathcal { D } } = \bigcup _ { k = 1 } ^ { K } { \mathcal { D } } _ { k }$ with $1 < K _ { 1 } < K$ , assume that its last $K - K _ { 1 }$ classes are empty, i.e., $n _ { k } = | \mathcal Ḋ D Ḍ _ { k } | > 0$ for $\forall k \in [ K _ { 1 } ]$ and $n _ { k } = | \mathcal D _ { k } | = 0$ for $K _ { 1 } < k \le K$ Suppose $d \geq K _ { 1 }$ and denote $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { k } } \end{array}$ with $\begin{array} { r } { \pmb { \bar { x } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { \pmb { x } \in \mathcal { F } _ { k } } \pmb { x } } \end{array}$ . Then, for BCE loss

$$
f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } ) \triangleq \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \frac { 1 } { n _ { k } } \sum _ { \boldsymbol { X } ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mathrm { b c e } } ( \boldsymbol { X } ^ { ( k ) } ) ,\tag{12}
$$

it satisfies

$$
\begin{array} { r l } & { f _ { \mathrm { b c e - o s } } ( W , \boldsymbol { b } , \mathcal { F } ) \geq \log \big ( 1 + \mathrm { e } ^ { b ^ { \star } - \gamma \sqrt { 1 - \| \widetilde { x } \| ^ { 2 } } } \big ) + ( K - K _ { 1 } ) \log \big ( 1 + \mathrm { e } ^ { - \gamma \| \widetilde { x } \| - b ^ { \dagger } } \big ) } \\ & { \qquad + ( K _ { 1 } - 1 ) \log \big ( 1 + \mathrm { e } ^ { - \frac { \gamma } { K _ { 1 } - 1 } \sqrt { 1 - \| \widetilde { x } \| ^ { 2 } } - b ^ { \star } } \big ) , } \end{array}\tag{13}
$$

where $b ^ { \dagger } \in \mathbb { R }$ and $b ^ { \star }$ is the unique real solution of

$$
\mathrm { e } ^ { \frac { \gamma } { K _ { 1 } - 1 } \sqrt { 1 - \| \tilde { { \boldsymbol x } } \| ^ { 2 } } } \mathrm { e } ^ { 2 b } - ( K _ { 1 } - 2 ) \mathrm { e } ^ { b } - ( K _ { 1 } - 1 ) \mathrm { e } ^ { \gamma \sqrt { 1 - \| \tilde { { \boldsymbol x } } \| ^ { 2 } } } = 0 .
$$

When $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ reaching the lower bound in Eq. (13),

• (collapse) for $\forall k \in [ K _ { 1 } ]$

$$
{ \pmb x } ^ { ( k ) } = \bar { { \pmb x } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in \mathcal { F } _ { k } } { \pmb x } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } ,
$$

• (equiangular) for $k , j \in [ K _ { 1 } ]$ with $j \neq k$ 2

$$
{ \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } = - \frac { 1 } { K _ { 1 } - 1 } , \quad \bar { \pmb x } _ { k } ^ { \top } \bar { \pmb x } _ { j } = - \frac { 1 } { K _ { 1 } - 1 } + \frac { K _ { 1 } } { K _ { 1 } - 1 } \| \tilde { \pmb x } \| ^ { 2 } ,
$$

• (isogonal) for $\forall k , j \in [ K _ { 1 } ]$ with $j \neq k$

$$
w _ { k } ^ { \top } \bar { \mathbf { x } } _ { k } = \sqrt { 1 - \| \widetilde { \mathbf { x } } \| ^ { 2 } } , \quad w _ { k } ^ { \top } \bar { \mathbf { x } } _ { j } = - \frac { 1 } { K _ { 1 } - 1 } \sqrt { 1 - \| \widetilde { \mathbf { x } } \| ^ { 2 } } , \quad w _ { k } ^ { \top } \widetilde { \mathbf { x } } = 0 ,
$$

• (minority collapse) for $K _ { 1 } < j \le K$ ， $i f \| { \tilde { \mathbf { x } } } \| \neq 0$

$$
{ w _ { j } } = - \frac { { \tilde { x } } } { \| \tilde { { \boldsymbol { x } } } \| } ,
$$

and for $K _ { 1 } < j \le K$ with $\forall k \in [ K _ { 1 } ]$

$$
\begin{array} { r } { \pmb { w } _ { j } ^ { \top } \bar { \pmb { x } } _ { k } = - \| \tilde { \pmb { x } } \| , \quad \pmb { w } _ { j } ^ { \top } \pmb { w } _ { k } = 0 , } \end{array}
$$

and for $\forall k \in [ K _ { 1 } ] , K _ { 1 } < j \leq K$

$$
b _ { k } = b ^ { \star } , \quad b _ { j } = b ^ { \dagger } .
$$

Furthermore, when $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ approaching its minimum value, $b ^ { \dagger }  + \infty$ and $\| \tilde { \pmb x } \|  0$ □

Proof The proof is given in Appendix, i.e., Theorem S.8.

Theorem 3 For dataset $\textstyle { \mathcal { D } } = \bigcup _ { k = 1 } ^ { K } { \mathcal { D } } _ { k }$ with $1 < K _ { 1 } < K$ , assume that its last $K - K _ { 1 }$ classes are empty, i.e., $n _ { k } = | \mathcal D _ { k } | > 0$ for $\forall k \in [ K _ { 1 } ]$ and $n _ { k } = | \mathcal D _ { k } | = 0$ for $K _ { 1 } < k \le K$ Suppose $d \geq K _ { 1 }$ and denote $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { k } } \end{array}$ with $\begin{array} { r } { \pmb { \bar { x } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { \pmb { x } \in \mathcal { F } _ { k } } \pmb { x } } \end{array}$ . Then, for CE loss

$$
f _ { \mathrm { c e - o s } } ( { W } , b , \mathcal { F } ) \triangleq \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } } L _ { \mathrm { c e } } ( X ^ { ( k ) } ) ,\tag{14}
$$

it satisfies

$$
f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } ) \geq \log ( 1 + c _ { 1 } + c _ { 2 } ) ,\tag{15}
$$

where

$$
\begin{array} { r l } & { c _ { 1 } \ = \left( K _ { 1 } - 1 \right) \exp { \bigg ( \displaystyle \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { { \boldsymbol x } } \| ^ { 2 } - 1 \big ) \bigg ) } } \\ & { c _ { 2 } \ = K _ { 1 } \displaystyle \frac { 1 - q ^ { \star } } { q ^ { \star } } \exp { \bigg ( \displaystyle \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { { \boldsymbol x } } \| ^ { 2 } - 1 \big ) \bigg ) } } \\ & { c _ { 3 } \ = \displaystyle \frac { 1 } { q ^ { \star } } \sqrt { g ( q ^ { \star } ) } \exp { \bigg ( \displaystyle \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { { \boldsymbol x } } \| ^ { 2 } - 1 \big ) \bigg ) } , } \end{array}
$$

and $g ( q ) = 1 - 2 q \| \tilde { \pmb { x } } \| ^ { 2 } + q ^ { 2 } \| \tilde { \pmb { x } } \| ^ { 2 }$ with a certain $q ^ { \star } \in ( 0 , 1 )$ . When $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ reaching the lower bound in Eq. (15),

• (collapse) for $\forall k \in [ K _ { 1 } ]$

$$
{ \pmb x } ^ { ( k ) } = \bar { { \pmb x } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in \mathcal { F } _ { k } } { \pmb x } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } ,
$$

• (equiangular) for ∀k, $j \in [ K _ { 1 } ]$ with $j \neq k$

$$
{ \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } = - \frac { 1 } { K _ { 1 } - 1 } + \frac { 1 } { K _ { 1 } } \frac { 1 } { K _ { 1 } - 1 } \big ( \frac { c _ { 2 } } { c _ { 3 } } \big ) ^ { 2 } \| \tilde { { \pmb x } } \| ^ { 2 } ,
$$

$$
\bar { \pmb { x } } _ { k } ^ { \top } \bar { \pmb { x } } _ { j } = - \frac { 1 } { K _ { 1 } - 1 } + \frac { K _ { 1 } } { K _ { 1 } - 1 } \| \tilde { \pmb { x } } \| ^ { 2 } ,
$$

• (isogonal) for $\forall k \in [ K _ { 1 } ]$ and $\forall j \in [ K _ { 1 } ]$ with $j \neq k$ 2

$$
{ \boldsymbol w } _ { k } ^ { \top } \bar { { \boldsymbol { x } } } _ { k } = \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \right) \frac { 1 } { c _ { 3 } } - \frac { c _ { 1 } } { K _ { 1 } - 1 } \frac { 1 } { c _ { 3 } } \| \tilde { { \boldsymbol { x } } } \| ^ { 2 } , \quad { \boldsymbol w } _ { k } ^ { \top } \tilde { { \boldsymbol { x } } } = \frac { c _ { 2 } } { c _ { 3 } } \frac { \| \tilde { { \boldsymbol { x } } } \| ^ { 2 } } { K _ { 1 } } ,
$$

$$
w _ { k } ^ { \top } \bar { x } _ { j } = \frac { 1 } { c _ { 3 } } \Bigg ( \frac { 1 } { K _ { 1 } - 1 } \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { K _ { 1 } } { K _ { 1 } - 1 } \frac { c _ { 2 } } { K _ { 1 } } \Bigg ) \| \tilde { x } \| ^ { 2 } - \frac { 1 } { c _ { 3 } } \frac { 1 } { K _ { 1 } - 1 } \Bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \Bigg ) ,\tag{16}
$$

• (minority collapse) for $K _ { 1 } < j \le K , \ i f \| \tilde { \pmb { x } } \| \neq 0$

$$
{ \pmb w } _ { j } = - \frac { \tilde { \pmb x } } { \lVert \tilde { \pmb x } \rVert } ,
$$

and for $K _ { 1 } < j \le K$ with $\forall k \in [ K _ { 1 } ]$

$$
\mathbf { \boldsymbol { w } } _ { j } ^ { \top } \bar { \mathbf { \boldsymbol { x } } } _ { k } = - \| \tilde { \mathbf { \boldsymbol { x } } } \| , \quad \mathbf { \boldsymbol { w } } _ { j } ^ { \top } \mathbf { \boldsymbol { w } } _ { k } = - \frac { c _ { 2 } } { c _ { 3 } } \frac { \| \tilde { \mathbf { \boldsymbol { x } } } \| } { K _ { 1 } } ,
$$

and for $\forall k \in [ K _ { 1 } ] , K _ { 1 } < j \leq K , \exists b ^ { \star } , b ^ { \dagger } \in \mathbb { R }$ , such that

$$
b _ { k } = b ^ { \star } , \quad b _ { j } = b ^ { \dagger } .
$$

Furthermore, when $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ approaching its minimum value, $b ^ { \dagger } - b ^ { \star } \to + \infty .$ ， $q ^ { \star } \to 1$ , and $\| \tilde { \pmb x } \|  0$ □

Proof The complete proof is given in Appendix, i.e., Theorem S.11.

According to Theorems 2 and 3, in the open-set scenarios, before the BCE and CE losses approach their global minima, they respectively reach the lower bound in Eqs. (13) and (15), at which point their classifier vectors and feature vectors have already formed relatively concise geometric structures. In the geometric structures, similar to standard $\mathrm { N C } ,$ sample features $\{ \pmb { x } = \mathcal { M } ( \pmb { X } ) : \pmb { X } \in \mathcal { D } _ { k } \}$ of a certain known class k collapse to the class center $\begin{array} { r } { \bar { \mathbfit { x } } _ { k } ; } \end{array}$ ; the diference is that although the class means $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K _ { 1 } }$ and classifier vectors $\{ \pmb { w } _ { k } \} _ { k = 1 } ^ { K _ { 1 } }$ of $K _ { 1 }$ known classes form equiangular frames respectively, they do not necessarily have the maximum pairwise angles, arccos $[ - 1 / ( K _ { 1 } - 1 ) ]$ . In the geometric structures, the classifier vectors $\{ \pmb { w } _ { j } : K _ { 1 } < j \leq K \}$ of the unknown class coincide with each other and are opposite to the direction of the global feature mean x˜. Simultaneously, the classifier biases of known and unknown classes converge to specific values, $b ^ { \star }$ and $b ^ { \dagger }$

In the structures, the classifier vectors $\{ w _ { k } \} _ { k = 1 } ^ { K _ { 1 } }$ of diferent known classes under the BCE have a uniform and largest angular separation, whereas those under the CE are uniform but with a smaller angular separation. Between the known and unknown classes, their classifier vectors, $\{ \pmb { w } _ { k } \} _ { k = 1 } ^ { K _ { 1 } }$ and $\{ \pmb { w } _ { j } : K _ { 1 } < j \leq K \}$ , under the BCE are orthogonal, while they under the CE have a larger angular separation

As BCE and CE respectively approach their minima in the open-set scenario, the global feature mean $\tilde { \mathbf { x } }$ tend toward the zero vector. At this point, standard NC still forms over the $K _ { 1 }$ known classes. Theoretically, the classifier vectors $\{ w _ { j } : K _ { 1 } < j \leq K \}$ of unknown classes always coincide with each other and remain perpendicular to the classifier vectors $\{ w _ { k } \} _ { k = 1 } ^ { K _ { 1 } }$ of known classes under the BCE, while under the CE they gradually approach perpendicularity to $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K _ { 1 } }$ . Meanwhile, as the losses approach their minima, $\{ w _ { j } : K _ { 1 } < j \le K \}$ consistently remains in the direction opposite to the global feature mean $\tilde { \mathbf { x } }$ in theoretical, i.e., $\begin{array} { r } { { \pmb w } _ { j } = - \frac { { \tilde { \pmb x } } } { \| { \tilde { \pmb x } } \| } } \end{array}$ ; however, in practice, as ∥x˜∥ approaches 0, any random perturbation can dramatically change the direction of x˜, thereby altering its angle with ${ \pmb w } _ { j }$

## 3.3 Extensions to Contrastive Losses

In contrastive learning, the common losses used for training feature extraction models include Softmax-based losses such as SupCon Khosla et al. (2020) and N-pair loss Sohn (2016), as well as Sigmoid-based losses such as MS loss Wang et al. (2019) and SimPLE Wen et al. (2023). The Softmax-based contrastive loss originates from noise contrastive estimation (Oord et al., 2018). In this paper, for $\forall ~ \mathbf { \boldsymbol { X } } ^ { ( k ) }$ from class $k , \forall k \in [ K ]$ , we define its Softmax-based contrastive loss as

$$
L _ { \mathrm { s o f t } } ( \boldsymbol { X } ^ { ( k ) } ) = - \frac { 1 } { n _ { k } } \sum _ { x _ { \mathrm { p } } \in \mathcal { F } _ { k } } \log \Bigg ( \frac { \mathrm { e } ^ { \gamma x _ { \mathrm { p } } ^ { \top } \boldsymbol { x } ^ { ( k ) } } } { \sum _ { j = 1 } ^ { K } \frac { 1 } { n _ { j } } \sum _ { \boldsymbol { x } \in \mathcal { F } _ { j } } \mathrm { e } ^ { \gamma \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { ( k ) } } } \Bigg ) ,\tag{17}
$$

where $\pmb { x } ^ { ( k ) } = \mathcal { M } ( \pmb { X } ^ { ( k ) } )$ and we apply the numbers of positive and negative sample pairs to re-balance the model training. After we completed the proof of Theorem 4 and began drafting this paper, we found that Zhu et al. Zhu et al. (2022) established on each batch of samples a re-balanced Softmax-based contrastive loss in the batch algorithm, similar to Eq. (17), and they proved that it can induce standard NC.

Similar to the BCE loss in Eq. (5), one can define Sigmoid-based contrastive loss from the first principle of feature learning. For any sample $\mathbf { \boldsymbol { X } } ^ { ( k ) }$ from class k, with $\boldsymbol { x } ^ { ( k ) } \ : = \ :$ $\mathcal { M } ( \mathbf { \boldsymbol { X } } ^ { ( k ) } )$ , it is expected that the feature similarity $\bar { \mathbf { \Delta } } \bar { \mathbf { \Delta } } _ { \mathbf { \Delta } \mathbf { p } } ^ { \top } \bar { \mathbf { \Delta } } \mathbf { x } ^ { ( k ) }$ with feature vector $\mathbf { \Delta x _ { \mathrm { p } } }$ of any positive sample from the same class exceeds any reasonable threshold $t _ { k } = b _ { k } / \gamma$ , while the feature similarity $\pmb { x } _ { \mathrm { n } } ^ { \top } \pmb { x } ^ { ( k ) }$ between it and ${ \pmb x } _ { \mathrm { n } }$ of any negative sample from any other class, such as $j$ with $j \neq k ,$ , is below any reasonable threshold $t _ { j } = b _ { j } / \gamma$ . Then, we apply Sigmoid to transform $\gamma \pmb { x } _ { \mathrm { p } } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k }$ and $\gamma \pmb { x } _ { \mathrm { n } } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j }$ into probabilities that meet the above expectations, and take BCE to construct the Sigmoid-based contrastive loss for model training. In total, for any sample $X ^ { ( k ) }$ from class $k .$

$$
L _ { \mathrm { s i g m } } ( \mathbf { X } ^ { ( k ) } ) = \frac { 1 } { n _ { k } } \sum _ { \mathbf { x } _ { \mathrm { p } } \in \mathcal { F } _ { k } } \log \big ( 1 + \mathrm { e } ^ { - \gamma \mathbf { x } _ { \mathrm { p } } ^ { \top } \mathbf { x } ^ { ( k ) } + b _ { k } } \big ) + \lambda \sum _ { j = 1 , j \neq k } ^ { K } \frac { 1 } { n _ { j } } \sum _ { \mathbf { x } _ { \mathrm { n } } \in \mathcal { F } _ { j } } \log \big ( 1 + \mathrm { e } ^ { \gamma \mathbf { x } _ { \mathrm { n } } ^ { \top } \mathbf { x } ^ { ( k ) } - b _ { j } } \big ) ,\tag{18}
$$

where $1 / n _ { k }$ and $1 / n _ { j }$ are used to eliminate the impact of the diferent numbers of positive sample pairs and negative ones of sample $X ^ { ( k ) }$ on the balanced training, and $\lambda > 0$ is used to adjust the learning rate of the feature similarities of negative sample pairs that reflecting between class separability.

Theorem 4 Suppose that each class in dataset D contains at least two samples, and that the feature dimension is greater than the number of classes, i.e., $n _ { k } = | \mathcal { D } _ { k } | > 2 ~ f o r ~ \forall k \in [ K ]$ and $d \geq K - 1$ . On the dataset D, define losses

$$
f _ { \mu } ( \mathcal { F } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mu } ( X ^ { ( k ) } ) ,\tag{19}
$$

where $\mu \in \{ \mathrm { s o f t } , \mathrm { s i g m } \}$ . Then, at its minimum point,

$$
{ \pmb x } ^ { ( k ) } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in \mathcal { F } _ { k } } { \pmb x } = \bar { { \pmb x } } _ { k } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] ,\tag{20}
$$

$$
\langle \bar { \pmb { x } } _ { k } , \bar { \pmb { x } } _ { j } \rangle = - \frac { 1 } { K - 1 } , \quad \forall k \neq j \in [ K ] ,\tag{21}
$$

and for the Sigmoid-based loss, it also satisfies

$$
b _ { k } = b ^ { \star } , \quad \forall k \in [ K ] ,\tag{22}
$$

where $b ^ { \star }$ is the unique solution of

$$
\mathrm { e } ^ { \frac { \gamma } { K - 1 } } \mathrm { e } ^ { 2 b } - \big [ \lambda ( K - 1 ) - 1 \big ] \mathrm { e } ^ { b } - \lambda ( K - 1 ) \mathrm { e } ^ { \gamma } = 0 .\tag{23}
$$

Proof The proofs are provided in Appendix, i.e., Theorems S.16 (Sigmoid) and S.17 (Softmax). ■

Similar to Eq. (7) in Theorem 1, which re-balances the model training using the inverse frequency of each class, Eq. (19) in Theorem 4 re-balances the model training by using the reciprocal of the number of sample pairs $( n _ { k } , n _ { j } )$ that can be formed by any two classes, such as classes k and j.

As shown in Eq. (18), in the Sigmoid-based contrastive loss, a weighting parameter λ can be flexibly employed to balance the learning rates of feature similarities of positive and negative sample pairs that reflecting within-class compactness and between-class separability. In contrast, the diferent feature similarities in Eq. (17) are coupled together through Softmax, making it dificult to flexibly adjust their learning intensity in the Softmax-based contrastive loss.

Furthermore, although the biases in the Sigmoid-based contrastive loss still play a substantive role during the feature learning, it can be proved that if the they are removed, the main conclusion of Theorem 4 remains valid.

## 4 Experiments

To verify the theorems, we applied the CE and BCE losses to train models such as ResNet and ViT on CIFAR10 and CIFAR100 (Krizhevsky and Hinton, 2009), respectively. The detailed experimental settings can be found in the supplementary.

<table><tr><td rowspan="2">Loss</td><td rowspan="2">Dataset</td><td rowspan="2">IF</td><td colspan="4">ResNet18</td></tr><tr><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w x } } } }$ </td><td>b</td></tr><tr><td rowspan="8">CE</td><td rowspan="5"></td><td>1</td><td> $\overline { { . 9 7 9 9 \pm . 0 0 2 9 } }$ </td><td> $\overline { { . 9 8 9 5 \pm . 0 0 1 5 } }$ </td><td> $\overline { { . 1 0 9 9 \pm . 0 0 7 9 } }$ </td><td> $\overline { { . 0 0 0 0 \pm . 0 2 2 1 } }$ </td></tr><tr><td>10</td><td> $. 9 7 7 8 \pm . 0 0 5 6$ </td><td> $. 9 8 8 6 \pm . 0 0 2 8$ </td><td> $- . 1 0 9 8 \pm . 0 0 8 1$ </td><td> $. 0 0 0 0 \pm . 0 1 3 1$ </td></tr><tr><td>CIFAR10 50</td><td> $. 9 7 0 9 \pm . 0 1 1 3$ </td><td> $. 9 8 5 0 \pm . 0 0 5 6$ </td><td> $- . 1 0 9 3 \pm . 0 1 0 5$ </td><td> $. 0 0 0 0 \pm . 0 2 4 6$ </td></tr><tr><td>100</td><td> $. 9 6 9 7 \pm . 0 1 3 4$ </td><td> $. 9 8 4 3 \pm . 0 0 6 7$ </td><td> $- . 1 0 9 2 \pm . 0 1 1 0$ </td><td> $. 0 0 0 0 \pm . 0 3 4 7$ </td></tr><tr><td>NC theor. value</td><td>1.0</td><td>1.0</td><td> $\bar { - 1 } \bar { / 9 } \overset { - } { \approx } - \bar { . } \bar { 1 } \bar { 1 } \bar { 1 }$ </td><td></td></tr><tr><td rowspan="4"></td><td>1</td><td> $\overline { { . 9 9 4 1 \pm . 0 0 1 8 } }$ </td><td> $\overline { { . 9 9 6 7 \pm . 0 0 1 0 } }$ </td><td> $- . 0 1 0 1 \pm . 0 3 0 6$   $\overline { { . 0 0 0 0 \pm . 0 0 4 1 } }$ </td></tr><tr><td>10 CIFAR100</td><td> $. 9 9 1 4 \pm . 0 0 3 3$ </td><td> $. 9 9 5 3 \pm . 0 0 1 7$   $- . 0 1 0 0 \pm . 0 7 8 7$ </td><td></td><td> $. 0 0 0 0 \pm . 0 1 4 8$ </td></tr><tr><td>50</td><td> $. 9 8 9 8 \pm . 0 0 6 3$ </td><td> $. 9 9 4 5 \pm . 0 0 3 2$ </td><td> $- . 0 1 0 0 \pm . 0 7 2 8$ </td><td> $. 0 0 0 0 \pm . 0 2 4 0$ </td></tr><tr><td>100</td><td> $. 9 8 8 8 \pm . 0 0 8 1$ </td><td> $. 9 9 4 0 \pm . 0 0 4 1$ </td><td> $- . 0 1 0 0 \pm . 0 7 3 7$ </td><td> $. 0 0 0 0 \pm . 0 2 5 9$ </td></tr><tr><td rowspan="11">BCE</td><td>NC theor. value</td><td>1.0</td><td></td><td>1.0</td><td> $\bar { 1 / 9 9 } \bar { \approx } - \bar { . 0 1 0 1 }$ </td><td></td></tr><tr><td rowspan="4"></td><td>1</td><td> $\overline { { . 9 9 9 3 \pm . 0 0 0 1 } }$ </td><td> $\overline { { . 9 9 9 6 \pm . 0 0 0 1 } }$ </td><td> $- . 1 1 1 1 \pm . 0 0 3 1$ </td><td> $\overline { { 4 . 6 6 9 \pm . 0 0 0 9 } }$ </td></tr><tr><td>10</td><td> $. 9 9 8 9 \pm . 0 0 0 5$ </td><td> $. 9 9 9 4 \pm . 0 0 0 2$ </td><td> $- . 1 1 1 0 \pm . 0 0 2 8$ </td><td> $4 . 6 6 8 \pm . 0 0 4 0$ </td></tr><tr><td>CIFAR10 50</td><td> $. 9 9 6 0 \pm . 0 0 2 0$ </td><td> $. 9 9 7 9 \pm . 0 0 1 0$ </td><td> $- . 1 1 0 9 \pm . 0 0 3 6$ </td><td> $4 . 6 6 3 \pm . 0 1 0 1$ </td></tr><tr><td>100</td><td> $. 9 9 6 7 \pm . 0 0 1 8$ </td><td> $. 9 9 8 3 \pm . 0 0 0 9$ </td><td> $- . 1 1 0 9 \pm . 0 0 3 4$ </td><td> $4 . 6 6 4 \pm . 0 0 7 3$ </td></tr><tr><td rowspan="3"></td><td>NC theor. value</td><td> $^ { - } \bar { 1 . 0 } ^ { - }$ </td><td> $^ - = - \overline { { 1 . 0 } } ^ { -- }$ </td><td> $\bar { - 1 } \bar { / 9 } \bar { \approx } - . \bar { 1 } \bar { 1 } \bar { 1 1 } \bar { 1 }$ </td><td> $^ - \bar { 4 } . \bar { 6 } 6 \bar { 9 } \bar { 8 } ^ { \mathrm { ~ -- ~ } }$ </td></tr><tr><td>1</td><td> $\overline { { . 9 9 9 3 \pm . 0 0 0 7 } }$ </td><td> $\overline { { . 9 9 9 4 \pm . 0 0 0 3 } }$ </td><td> $- . 0 1 0 1 \pm . 0 0 2 5$ </td><td> $\overline { { 6 . 3 3 0 \pm . 0 0 0 9 } }$ </td></tr><tr><td>10</td><td> $. 9 9 8 9 \pm . 0 0 0 4$ </td><td> $. 9 9 9 3 \pm . 0 0 0 2$ </td><td> $- . 0 1 0 1 \pm . 0 0 2 1$ </td><td> $6 . 3 2 8 \pm . 0 0 4 2$ </td></tr><tr><td rowspan="2"></td><td>CIFAR100 50 100</td><td> $. 9 9 7 8 \pm . 0 0 1 4$   $. 9 9 7 1 \pm . 0 0 2 1$ </td><td> $. 9 9 8 7 \pm . 0 0 0 7$ </td><td> $- . 0 1 0 1 \pm . 0 1 5 4$ </td><td> $6 . 3 2 9 \pm . 0 0 6 6$ </td></tr><tr><td>NC theor. value</td><td>1.0</td><td> $. 9 9 8 3 \pm . 0 0 1 1$  1.0</td><td> $- . 0 1 0 1 \pm . 0 2 4 6$   $- 1 / 9 9 \stackrel { - } { \approx } - . 0 \bar { 1 } \bar { 0 } \bar { 1 }$ </td><td> $6 . 3 3 2 \pm . 0 0 7 5$  6.3437</td></tr></table>

Table 1: Statistical results $( \mathrm { m e a n } \pm \mathrm { s t d } )$ for positive/negative similarities $( s _ { \mathrm { p o s } } / s _ { \mathrm { n e g } } )$ and biases (b) of diferent models trained by CE and BCE on $\mathrm { C I F A R 1 0 – L T }$ and CIFAR100-LT with $\mathrm { I F } = 1 , 1 0 , 5 0 .$ , and 100. The CE and BCE are class-wise averaged using Eq. (7). In the table, when the absolute value of a number if less than 1, we omit the ‘0’ in its integer part to save space.

In each experiment, after the model training, we collect the following sets of diferent types of values,

$$
\begin{array} { r } { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } = \big \{ x ^ { \top } x _ { * } : \forall x , x _ { * } \in \mathcal { F } k , \forall k \in [ K ] \big \} , \quad s _ { \mathrm { n e g } } ^ { \mathrm { x x } } = \big \{ \bar { x } _ { k } ^ { \top } \bar { x } _ { j } : \forall k \ne j \in [ K ] \big \} , } \end{array}
$$

$$
\begin{array} { r } { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } = \big \{ w _ { k } ^ { \top } \bar { x } _ { k } : \forall k \in [ K ] \big \} , \quad s _ { \mathrm { n e g } } ^ { \mathrm { w x } } = \big \{ w _ { j } ^ { \top } \bar { x } _ { k } : \forall j \neq k \in [ K ] \big \} , } \end{array}
$$

$$
s _ { \mathrm { n e g } } ^ { \mathrm { w w } } = \{ \pmb { w } _ { j } ^ { \top } \pmb { w } _ { k } : \forall j \neq k \in [ K ] \} .
$$

According to Theorem 1, when the losses reaching their minima, the numbers in $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ and $s _ { \mathrm { p o s } } ^ { \mathrm { w x } }$ will converge to 1, while that in $s _ { \mathrm { n e g } } ^ { \mathrm { x x } } , s _ { \mathrm { n e g } } ^ { \mathrm { w x } }$ , and $s _ { \mathrm { n e g } } ^ { \mathrm { w w } }$ will converge to $- \frac { 1 } { K - 1 }$

In open-set scenario, we collect the above sets on the first $K _ { 1 }$ non-empty classes, and the numbers will converge to specific values related to $K _ { 1 }$ and ∥x˜∥, given by Theorems 2 and 3. Besides that, for the last $K - K _ { 1 }$ empty unknown classes, we collect the sets as

$$
s _ { \mathrm { e m p } } ^ { \mathrm { w w } } = \big \{ \pmb { w } _ { j } ^ { \top } \pmb { w } _ { j ^ { \prime } } : K _ { 1 } < j \neq j ^ { \prime } \leq K \big \} ,
$$

$$
s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } } = \big \{ \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } : k \in [ K _ { 1 } ] , K _ { 1 } < j \le K \big \} ,
$$

<table><tr><td rowspan="2">Loss</td><td rowspan="2">Dataset</td><td rowspan="2">IF</td><td colspan="4">ViT-Tiny</td></tr><tr><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w x } } } }$ </td><td>b</td></tr><tr><td rowspan="9">CE</td><td rowspan="5">CIFAR10</td><td>1</td><td> $. 9 9 9 7 \pm . 0 0 0 0$ </td><td>.9999 ± .0000</td><td> $. 1 1 1 1 \pm . 0 0 0 4$ </td><td> $\overline { { . 0 0 1 1 \pm . 0 0 1 2 } }$ </td></tr><tr><td>10</td><td> $. 9 9 9 3 \pm . 0 0 0 1$ </td><td>.9997 ± .0001</td><td> $- . 1 1 1 1 \pm . 0 0 0 3$ </td><td> $- . 0 0 0 4 \pm . 0 0 1 3$ </td></tr><tr><td>50</td><td> $. 9 9 9 2 \pm . 0 0 0 1$ </td><td>.9996 ± .0000</td><td>−.1111 ±.0004</td><td> $- . 0 0 0 2 \pm . 0 0 1 2$ </td></tr><tr><td>100</td><td> $. 9 9 8 5 \pm . 0 0 0 2$ </td><td>.9993 ± .0001</td><td>−.1110 ± .0003</td><td> $- . 0 0 0 8 \pm . 0 0 0 8$ </td></tr><tr><td>NC theor. value</td><td>1.0</td><td>1.0</td><td> $\bar { - 1 } \bar { / 9 } \overset { - } { \approx } - \bar { . 1 } \bar { 1 } \bar { 1 1 }$ </td><td></td></tr><tr><td rowspan="4">CIFAR100</td><td>1</td><td> $\overline { { . 9 8 9 3 \pm . 0 0 5 2 } }$ </td><td>.9942 ± .0026</td><td> $- . 0 1 0 1 \pm . 0 0 7 0$ </td><td>.0208 ± .0041</td></tr><tr><td>10</td><td>.9999 ± .0000</td><td>1.000 ± .0000</td><td> $- . 0 1 0 1 \pm . 0 0 0 5$ </td><td> $. 0 1 4 5 \pm . 0 0 1 3$ </td></tr><tr><td>50</td><td>.9999 ± .0000</td><td>.9999 ± .0000</td><td>−.0101 ± .0007</td><td> $. 0 1 4 5 \pm . 0 0 1 4$ </td></tr><tr><td>100</td><td>.9968 ± .0007</td><td>.9984 ± .0004</td><td>−.0101 ± .0067</td><td>.0142 ± .0072</td></tr><tr><td rowspan="11">BCE</td><td rowspan="2">NC theor. value</td><td></td><td>1.0</td><td>1.0</td><td>-1/99≈−.0101</td><td></td></tr><tr><td>1</td><td>.9999 ± .0000</td><td>.9999 ± .0000</td><td> $- . 1 1 1 1 \pm . 0 0 1 6$ </td><td>4.670 ± .0003</td></tr><tr><td rowspan="4"></td><td>10</td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td>−.1111 ± .0012</td><td> $4 . 6 7 0 \pm . 0 0 0 2$ </td></tr><tr><td>CIFAR10 50</td><td>.9996 ± .0002</td><td>.9998 ± .0001</td><td>−.1111 ±.0007</td><td>4.675 ± .0092</td></tr><tr><td>100</td><td>.9945 ± .0046</td><td>.9954 ± .0038</td><td>−.1109 ± .0162</td><td>4.658 ± .0507</td></tr><tr><td>NC theor. value</td><td>1.0</td><td>1.0</td><td>−1/9≈−.1111</td><td>4.6698</td></tr><tr><td rowspan="4"></td><td>1</td><td>.9963 ± .0036</td><td>.9980 ± .0019</td><td> $- . 0 1 0 1 \pm . 0 0 3 7$ </td><td>6.337 ± .0017</td></tr><tr><td>10</td><td> $. 9 9 9 9 \pm . 0 0 0 0$ </td><td>.9999 ± .0000</td><td>−.0101 ± .0010</td><td> $6 . 3 4 4 \pm . 0 0 1 1$ </td></tr><tr><td>CIFAR100 50</td><td>.9996 ± .0001</td><td>.9998 ± .0000</td><td>−.0101 ± .0008</td><td> $6 . 3 4 2 \pm . 0 0 1 8$ </td></tr><tr><td>100</td><td>.9983 ± .0010</td><td>.9992 ± .0005</td><td>−.0101 ± .0088</td><td> $6 . 3 2 5 \pm . 0 0 7 9$ </td></tr><tr><td></td><td>NC theor. value</td><td>1.0</td><td>1.0</td><td>-1/99≈-.0101</td><td>6.3437</td></tr></table>

$$
s _ { \mathrm { e m p } } ^ { \mathrm { w x } } = \big \{ { w } _ { j } ^ { \top } \tilde { { \pmb x } } / \| \tilde { { \pmb x } } \| : K _ { 1 } < j \leq K \big \} ,
$$

and in these three sets, the numbers will respectively converge to 1, 0, and −1 when minority collapse occurs.

## 4.1 On Imbalanced Datasets

We first verify Theorem 1. Table 1 shows the statistical results of $s _ { \mathrm { p o s } } ^ { \mathrm { x x } } , \ s _ { \mathrm { p o s } } ^ { \mathrm { w x } } , \ s _ { \mathrm { n e g } } ^ { \mathrm { w x } } .$ , and that of biases $\{ b _ { j } \} _ { j = 1 } ^ { K }$ for ResNet18 and ViT-Tiny trained on training data of CIFAR10-LT and CIFAR100-LT (Cao et al., 2019) with diferent imbalanced factors, $\mathrm { I F } \in \{ 1 , 1 0 , 5 0 , 1 0 0 \}$ where the model training is driven using the class-wise (CW) averaged CE and BCE defined in Eq. (7). In the experiments, we trained the models for 800 epochs with a batch size of 128. The optimizer and learning rate setting are detailed in Table S.1. No weight decay was applied. The classifier biases were initialized to 0.0, and the scale factor γ was set to 8.0.

According to Theorem 1, when the CW losses reaching their global minima, the features of samples of the same class align with their corresponding classifier vectors, while diferent classes form an ETF. From Table 1, one can find that the results are highly consistent with the theoretical analysis, regardless of whether the CE or BCE is used to train the ResNet or ViT. On CIFAR10-LT, except for the data of ResNet18 trained with CE, which deviates slightly more from their theoretical values, the means of $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ and $s _ { \mathrm { p o s } } ^ { \mathrm { w x } }$ for the other models are basically all around 0.99, very close to the theoretical value of 1, while the means of the $s _ { \mathrm { n e g } } ^ { \mathrm { w x } }$ are all around −0.11, close to the theoretical value of $- 1 / 9 ;$ moreover, these all data have relatively small standard deviations, even $\mathrm { I F } = 1 0 0$ . Similar observations can also be made on CIFAR100-LT. These results indicate that, regardless of the imbalance ratio, class number, or the model architecture, training a model with the CW averaged loss ultimately causes the sample features and classifier vectors of diferent classes to form an ETF, reformulating standard NC on imbalanced datasets.

Furthermore, the results of final classifier biases $\{ b _ { k } \} _ { k = 1 } ^ { K }$ in Table 1 illustrate the distinction between the CE and BCE losses. For the CE loss, Theorem 1 only requires the biases of diferent classes to converge to the same value, which can be any real number, with no unique theoretical value. In the experiments, since the initial values were set to zero mean, after training the models using CE, their all classifier biases still concentrated around 0.0. In contrast, when BCE loss reaches a minimum, there is a unique theoretical classifier bias. As shown in Table 1, the theoretical bias values on CIFAR10 and CIFAR100 are 4.6698 and 6.3437, respectively. The final biases obtained in practical training concentrate near these two values and exhibit small standard deviations across diferent models and datasets with diferent IFs.

![](images/13130c275554e9a1f51b7fdc083ed116fe405c25c1c3d12a24e09a9a7650ebc9.jpg)

![](images/985f5cb7fff1c2e9ca414da9795d00debc3e3f081ada0422bf11986eac3ca0f4.jpg)  
Figure 1: Distribution of positive and negative scaled similarities $( \gamma \pmb { w } _ { k } ^ { \top } \pmb { \bar { x } } _ { j } )$ and biases of ResNet18 trained by class-wise (CW) and globally (G.) averaged CE and BCE.

To demonstrate the theoretical advantages of the class-wise (CW) averaged losses defined in Eq. (7), we also train the models using the globally (G.) averaged losses defined in Eq. (6), with $\gamma = 8 . 0$ . Using violin plots, Fig. 1 shows the distributions of positive/negative scaled similarities $( \gamma \pmb { w } ^ { \top } \pmb { x } )$ and biases of ResNet18 trained by the diferent losses on CIFAR10-LT with $\mathrm { I F } = 1 , 1 0 , 5 0$ , and 100. When IF= 1, the CW loss and the G. loss are equivalent, and their final results are both close to NC solution. However, as IF increases, there is a significant diference between the results of these two types of losses. With diferent IFs, the three metrics obtained by the CW averaged loss are all concentrated around the NC theoretical values. In contrast, for the G. averaged losses, as IF increases, although the positive similarities are still mainly concentrated around the theoretical value, the negative similarities and classifier biases become increasingly dispersed and increasingly dificult to converge. These results demonstrate the theoretical advantages of the CW averaged losses over the G. averaged loss, i.e., on imbalanced datasets, the former can achieve standard

![](images/a3f6a48d53a6e593c1defe5b7c3f3ef20f671b738522b963b800832982de3b3e.jpg)

![](images/6c38886e4957ac8a7a50a95502d4ce08a91793c5f863d459cc10a23eb00d34fe.jpg)

![](images/017856e1a05be88a6f3c92945b01f794a14bbb0be4f8242091ff4ffbedad930d.jpg)

(a) Similarity matrices learned by class-wise (CW) averaged BCE loss defined in Eq. 7.  
![](images/7df10424139eb0a55130a0019ca5a3de6385b7424c100b603624fb502deeaafd.jpg)  
(b) Similarity matrices learned by global (G.) averaged BCE loss defined in Eq. 6.  
Figure 2: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ of ResNet18 trained by diferent BCE losses on CIFAR10-LT with $\mathrm { I F } = 1 0 0$

NC, theoretically maximizing the between-class compactness of sample features and, in a balanced manner, maximizing the between-class separability between diferent sample features and classifier vectors of diferent classes, whereas the latter cannot.

Fig. 2 shows the similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ of ResNet18 trained by CW and G. averaged BCE on CIFAR10-LT with $\mathrm { I F } = 1 0 0$ For the CW loss, in its diferent similarity matrices, the of-diagonal elements that reflect the between-class separabilities are uniformly smaller than −0.1, close to the NC theoretical value $\mathrm { o f - } 1 / 9$ , and lie within a similarity interval with a relative length of less than 0.024. These matrices indicate that the CW loss achieves balanced between-class separabilities. In contrast, for the G. loss, the between-class geometric structures among sample features and classifier vectors are significantly distorted. The of-diagonal elements of all three matrices exhibits obvious diferences, lying within a similarity interval of about 0.4 in length; meanwhile, these of-diagonal values gradually increase from about −0.3 in the upper-left corner to about 0.1 in the lower-right corner, which is clearly correlated with the number of samples contained in the classes from head to tail. These results illustrate that, when training model on imbalanced datasets by using the globally (G.) averaged loss, the training is significantly afected by the sample number, and the between-class separabilities exhibit obvious imbalanced efect across the diferent classes, whereas the class-wise (CW) averaged loss could efectively suppresses this imbalance.

<table><tr><td rowspan="2">D</td><td rowspan="2"> $K _ { 1 }$ </td><td colspan="5">ResNet18 trained using CE</td></tr><tr><td> $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ </td><td> $\underline { { s } } _ { \mathrm { n e g } } ^ { \mathrm { x x } }$ </td><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\underline { { s } } _ { \mathrm { n e g } } ^ { \mathrm { w x } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w w } } } }$ </td></tr><tr><td rowspan="5">CIA10</td><td>9</td><td> $. 9 9 3 8 \pm . 0 0 0 5$ </td><td> $. 1 2 4 1 \pm . 0 0 5 1$ </td><td> $. 9 9 6 8 \pm . 0 0 0 2$ </td><td> $\overline { { . 1 2 4 6 \pm . 0 0 3 4 } }$ </td><td> $. 1 2 5 0 \pm . 0 0 0 4$ </td></tr><tr><td>7</td><td> $. 9 5 5 6 \pm . 0 0 6 7$ </td><td> $- . 1 2 2 1 \pm . 0 1 7 5$ </td><td> $. 9 6 4 9 \pm . 0 0 2 2$ </td><td> $- . 1 5 5 8 \pm . 0 1 3 7$ </td><td> $- . 1 6 5 9 \pm . 0 1 0 2$ </td></tr><tr><td>5</td><td> $. 9 5 3 2 \pm . 0 0 6 0$ </td><td> $- . 1 1 5 5 \pm . 0 2 2 0$ </td><td> $. 9 4 2 9 \pm . 0 0 6 3$ </td><td> $- . 2 0 9 5 \pm . 0 1 8 9$ </td><td> $- . 2 4 4 2 \pm . 0 1 1 1$ </td></tr><tr><td>3</td><td> $. 9 3 8 5 \pm . 0 0 7 3$ </td><td> $- . 1 4 7 6 \pm . 0 1 4 5$ </td><td> $. 9 2 7 1 \pm . 0 0 5 0$ </td><td> $- . 3 1 5 5 \pm . 0 1 8 8$ </td><td> $- . 4 3 0 7 \pm . 0 1 6 2$ </td></tr><tr><td>90</td><td> $\overline { { . 9 9 4 8 \pm . 0 0 1 0 } }$ </td><td> $- . 0 1 0 8 \pm . 0 5 4 1$ </td><td> $\overline { { . 9 9 7 1 \pm . 0 0 0 6 } }$ </td><td> $- . 0 1 1 2 \pm . 0 5 4 1$ </td><td> $- . 0 1 1 2 \pm . 0 5 4 1$ </td></tr><tr><td>CI10</td><td>70</td><td> $. 9 9 2 2 \pm . 0 0 1 1$ </td><td> $- . 0 1 3 1 \pm . 0 4 8 5$ </td><td> $. 9 9 5 3 \pm . 0 0 0 6$ </td><td> $- . 0 1 4 4 \pm . 0 4 8 6$ </td><td> $- . 0 1 4 5 \pm . 0 4 8 7$ </td></tr><tr><td></td><td>50</td><td> $. 9 9 4 4 \pm . 0 0 0 6$ </td><td> $- . 0 1 6 4 \pm . 0 2 1 0$ </td><td> $. 9 9 5 4 \pm . 0 0 0 3$ </td><td> $- . 0 2 0 0 \pm . 0 2 0 8$ </td><td> $- . 0 2 0 4 \pm . 0 2 0 7$ </td></tr><tr><td></td><td>30</td><td> $. 9 8 7 8 \pm . 0 0 1 7$ </td><td> $- . 0 2 1 1 \pm . 0 0 8 5$ </td><td> $. 9 8 9 1 \pm . 0 0 0 9$ </td><td> $- . 0 3 2 2 \pm . 0 0 6 2$ </td><td> $- . 0 3 4 2 \pm . 0 0 3 3$ </td></tr><tr><td rowspan="3">D</td><td rowspan="3"> $K _ { 1 }$ </td><td colspan="4">ResNet18 trained using BCE</td><td></td></tr><tr><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w w } } } }$ </td></tr><tr><td colspan="3"> $. 9 9 3 5 \pm . 0 0 3 2$ </td><td></td><td> $. 1 2 4 5 \pm . 0 1 1 3$ </td><td> $. 1 2 5 0 \pm . 0 0 2 8$ </td></tr><tr><td rowspan="4">CI10</td><td>9 7</td><td> $. 9 9 5 4 \pm . 0 0 0 3$ </td><td> $. 1 2 3 0 \pm . 0 1 5 2$ </td><td> $. 9 9 5 8 \pm . 0 0 1 5$ </td><td></td><td></td></tr><tr><td></td><td></td><td> $- . 1 6 5 3 \pm . 0 0 3 9$ </td><td> $. 9 9 7 4 \pm . 0 0 0 2$ </td><td> $- . 1 6 6 2 \pm . 0 0 2 6$ </td><td> $\cdot . 1 6 6 7 \pm . 0 0 0 2$ </td></tr><tr><td>5</td><td> $. 9 9 3 6 \pm . 0 0 0 9$ </td><td> $- . 2 3 2 3 \pm . 0 1 3 7$ </td><td> $. 9 9 0 3 \pm . 0 0 0 3$ </td><td> $- . 2 4 7 6 \pm . 0 0 8 7$ </td><td> $- . 2 5 0 0 \pm . 0 0 0 2$ </td></tr><tr><td>3</td><td> $. 9 8 5 9 \pm . 0 0 1 2$ </td><td> $- . 2 7 1 3 \pm . 0 1 2 5$ </td><td> $. 9 1 2 4 \pm . 0 0 1 9$ </td><td> $- . 4 1 0 7 \pm . 0 0 9 1$ </td><td> $- . 4 9 9 9 \pm . 0 0 0 2$ </td></tr><tr><td rowspan="4">CI10</td><td>90</td><td> $\overline { { . 9 9 9 0 \pm . 0 0 0 6 } }$ </td><td> $- . 0 1 0 9 \pm . 0 0 3 1$ </td><td> $\overline { { . 9 9 9 3 \pm . 0 0 0 3 } }$ </td><td> $- . 0 1 1 2 \pm . 0 0 2 3$ </td><td> $\overline { { \cdot . 0 1 1 2 \pm . 0 0 1 2 } }$ </td></tr><tr><td>70</td><td> $. 9 9 9 0 \pm . 0 0 0 3$ </td><td> $- . 0 1 3 7 \pm . 0 0 3 2$ </td><td> $. 9 9 9 1 \pm . 0 0 0 1$ </td><td> $- . 0 1 4 5 \pm . 0 0 2 3$ </td><td> $- . 0 1 4 5 \pm . 0 0 0 7$ </td></tr><tr><td>50</td><td> $. 9 9 9 4 \pm . 0 0 0 3$ </td><td> $- . 0 1 8 4 \pm . 0 0 3 7$ </td><td> $. 9 9 8 7 \pm . 0 0 0 1$ </td><td> $- . 0 2 0 4 \pm . 0 0 2 6$ </td><td> $- . 0 2 0 4 \pm . 0 0 0 5$ </td></tr><tr><td>30</td><td> $. 9 9 8 5 \pm . 0 0 0 2$ </td><td> ${ . 0 0 9 8 \pm . 0 0 3 8 }$ </td><td> $. 9 8 7 1 \pm . 0 0 0 2$ </td><td> $. 0 3 4 2 \pm . 0 0 2 7$ </td><td> $. 0 3 4 5 \pm . 0 0 0 6$ </td></tr></table>

Table 2: Statistical results (mean ± std) for diferent metrics of ResNet18 trained by CE and BCE on CIFAR10 and CIFAR100 with $K - K _ { 1 }$ empty classes. The CE and BCE are class-wise averaged using Eqs. (12) and (14).

## 4.2 On Open-Set Scenario

We further verify Theorems 2 and 3. On CIFAR10 $( K = 1 0 )$ and CIFAR100 $( K = 1 0 0 )$ we retain all samples in the first $K _ { 1 }$ classes while empty the last $K - K _ { 1 }$ classes. When $K _ { 1 } = 9 , 7 , 5 , 3$ on CIFAR10 and $K _ { 1 } = 9 0 , 7 0 , 5 0 , 3 0$ on CIFAR100, we respectively train ResNet18 using the CE and BCE losses defined in Eqs. (12) and (14). In the experiments, the losses were class-wise (CW) averaged over the $K _ { 1 }$ known classes, and the models were trained for 1000 epochs. The other training settings are detailed in Table S.1.

According to the theorems, as the losses decrease, these numbers in $s _ { \mathrm { p o s } } ^ { \mathrm { x x } } , s _ { \mathrm { n e g } } ^ { \mathrm { x x } } , s _ { \mathrm { p o s } } ^ { \mathrm { w x } } , s _ { \mathrm { n e g } } ^ { \mathrm { w x } } ,$ $s _ { \mathrm { n e g } } ^ { \mathrm { w w } } ,$ and $s _ { \mathrm { o p e n } } ^ { \mathrm { w x } }$ will converge to specific values related to $K _ { 1 }$ and the norm of the global feature center, $\begin{array} { r } { \| \tilde { \pmb { x } } \| = \big \| \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { k } \big \| } \end{array}$ . Table 2 shows their mean and standard deviations.

Except for the model trained on CIFAR10 using CE, the features extracted by other models all converge to their class centers, as evidenced by the means of their $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ being around 0.99 and having very small standard deviations. For these models, when $\| \dot { \widetilde { \pmb { x } } } \|$ is less than 0.025, the final sample features and classifier vectors closely approximate the ETF structure corresponding to a number of classes equal to $K _ { 1 }$ . For example, consider the model trained using BCE on CIFAR100 with $K _ { 1 } = 9 0$ . For all the known classes, the norm of their feature center x˜ is 0.0186. Correspondingly, the means of $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ and $s _ { \mathrm { p o s } } ^ { \mathrm { w x } }$ are 0.9990 and 0.9993, respectively, with very small standard deviations, indicating that the sample features within the same class essentially collapse and align with their classifier vector. Meanwhile, the means of $s _ { \mathrm { n e g } } ^ { \mathrm { x x } } , s _ { \mathrm { n e g } } ^ { \mathrm { w x } }$ and $s _ { \mathrm { n e g } } ^ { \mathrm { w w } }$ are −0.0109, −0.0112 and −0.0112, which are relatively close $\mathrm { t o \ t o \ t { \ t i } }$ , indicating that the class centers and classifier vectors basically form an ETF structure that maximizes equiangular separation.

<table><tr><td rowspan="3">D</td><td rowspan="3"> $K _ { 1 }$ </td><td colspan="4">ResNet18 trained using CE</td></tr><tr><td> $s _ { \mathrm { e m p } } ^ { \mathrm { w x } }$ </td><td> $\underline { s } _ { \mathrm { e m p } } ^ { \mathrm { w w } }$ </td><td> $\underline { s } _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } }$ </td><td> $\overline { { | | \tilde { \mathbf { x } } | | } }$ </td></tr><tr><td></td><td></td><td> $. 0 0 0 0 \pm . 0 0 2 8$ </td><td>0.0114</td></tr><tr><td rowspan="4">CI10</td><td>9 7</td><td> $. 1 6 6 8 \pm . 0 0 0 0$ </td><td> $. 8 4 3 1 \pm . 0 1 7 1$ </td><td> $- . 0 2 2 0 \pm . 0 1 7 0$ </td><td>0.1784</td></tr><tr><td></td><td> $- . 9 2 6 1 \pm . 0 0 8 3$ </td><td> $. 7 3 8 3 \pm . 0 2 1 4$ </td><td> $- . 0 5 8 3 \pm . 0 1 0 3$ </td><td></td></tr><tr><td>5 3</td><td> $- . 8 6 6 3 \pm . 0 0 9 2$ </td><td> $. 5 5 8 1 \pm . 0 2 3 4$ </td><td> $- . 1 6 0 3 \pm . 0 2 8 1$ </td><td>0.3135 0.4631</td></tr><tr><td>90</td><td> $- . 7 5 6 9 \pm . 0 0 6 8$   $- . 2 5 4 1 \pm . 0 1 5 6$ </td><td> $\overline { { . 4 8 3 6 \pm . 0 3 0 9 } }$ </td><td> $\overline { { \cdot . 0 0 0 1 \pm . 0 1 5 8 } }$ </td><td>0.0204</td></tr><tr><td>CI10</td><td>70 50 30</td><td> $- . 7 0 4 7 \pm . 0 1 5 6$   $- . 8 2 6 9 \pm . 0 1 3 3$   $. 7 9 7 7 \pm . 0 1 7 0$ </td><td> $. 6 8 2 2 \pm . 0 2 4 7$   $. 7 6 3 0 \pm . 0 2 0 2$   $. 6 4 8 9 \pm . 0 2 5 1$ </td><td> $- . 0 0 1 3 \pm . 0 1 1 5$   $- . 0 0 3 9 \pm . 0 0 8 2$   $. 0 1 3 6 \pm . 0 0 8 9$ </td><td>0.0359 0.0620 0.1121</td></tr><tr><td rowspan="2"> $\mathcal { D }$ </td><td rowspan="2"> $K _ { 1 }$ </td><td></td><td>ResNet18 trained using BCE</td><td></td><td></td></tr><tr><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } } } }$ </td><td>||x|</td></tr><tr><td rowspan="4">CI10</td><td rowspan="4">9 7 5</td><td> $- . 5 3 9 0 \pm . 0 0 0 0$ </td><td></td><td> $. 0 0 0 4 \pm . 0 0 3 2$ </td><td></td></tr><tr><td></td><td></td><td></td><td>0.0328</td></tr><tr><td> $- . 9 5 4 5 \pm . 0 0 0 6$ </td><td> $. 9 9 9 1 \pm . 0 0 0 3$ </td><td> $. 0 0 0 0 \pm . 0 0 1 6$ </td><td>0.0216</td></tr><tr><td> $. 9 9 0 5 \pm . 0 0 0 3$ </td><td> $. 9 9 9 4 \pm . 0 0 0 2$ </td><td> $. 0 0 0 2 \pm . 0 0 1 2$ </td><td>0.1133</td></tr><tr><td rowspan="4">CI10</td><td>3 90</td><td> $- . 9 9 9 0 \pm . 0 0 0 3$ </td><td> $. 9 9 9 0 \pm . 0 0 0 4$   $\overline { { 1 . 0 0 0 \pm . 0 0 0 0 } }$ </td><td> $. 0 0 7 8 \pm . 0 0 2 6$   $\overline { { . 0 0 0 0 \pm . 0 0 1 3 } }$ </td><td>0.3845 0.0186</td></tr><tr><td>70</td><td> $- . 2 1 6 6 \pm . 0 0 0 0$   $- . 7 8 3 0 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 0 0 \pm . 0 0 0 3$ </td><td>0.0279</td></tr><tr><td>50</td><td> $- . 9 2 5 3 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 0 0 \pm . 0 0 0 3$ </td><td>0.0444</td></tr><tr><td>30</td><td> $- . 9 9 6 3 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 0 9 \pm . 0 0 0 7$ </td><td>0.1544</td></tr></table>

In contrast, when ∥x˜∥ is relatively large, although features within each class also collapse, the class centers do not align well with the classifier vectors, and their structure difers substantially form the ETF, while remaining very close to the structure specified by the theorems. For example, $\| \tilde { \mathbfcal { x } } \| = 0 . 1 5 4 4$ for the model trained using BCE on CIFAR100 with $K _ { 1 } = 3 0$ . Correspondingly, the mean of $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ is 0.9985, whereas that of $s _ { \mathrm { p o s } } ^ { \mathrm { w x } }$ is 0.9871, indicating that within-class collapse is achieved but the collapsed features do not fully align with their classifier vectors. Then mean of $s _ { \mathrm { n e g } } ^ { \mathrm { w w } }$ is −0.0345, which is equal to − $- 1 / 2 9$ , indicating that the classifier vectors essentially form an ETF. However, the mean of $s _ { \mathrm { n e g } } ^ { \mathrm { x x } }$ is −0.0098, which difers substantially from $- 1 / 2 9$ . Nevertheless, it should be noted that the means of $s _ { \mathrm { p o s } } ^ { \mathrm { w x } }$ and $s _ { \mathrm { n e g } } ^ { \mathrm { x x } }$ are very close to the theoretical values, $\sqrt { 1 - \| \tilde { \pmb { x } } \| ^ { 2 } } \approx 0 . 9 8 8 0$ and $\begin{array} { r } { - \frac { 1 } { K _ { 1 } - 1 } + \frac { \mathbf { \tilde { \rho } } _ { K _ { 1 } } } { K _ { 1 } - 1 } \| \tilde { \pmb x } \| ^ { 2 } \approx - 0 . 0 0 9 8 . } \end{array}$ , given by Theorem 2.

In Table 2, for models trained with BCE, it is relatively easily to verify that their metrics largely conform to the conclusions of Theorem 2. In contrast, for models trained with $\mathrm { C E } ,$ verifying compliance with Theorem 3 requires first numerically solving $c _ { 1 } , c _ { 2 }$ , and $c _ { 3 } ,$ which is slightly more dificult. One can observe that, compared to the models trained by CE, that trained by BCE achieve smaller standard deviations across various metrics, indicating that after suficient training, the BCE-trained models yield classifiers and sample features with superior geometric structures. Furthermore, from Table 2, it can also be observed that when $K _ { 1 }$ is closer to K, after the same training, the model’s ∥x˜∥ is smaller, and all metrics are closer to NC values, which is consistent with our intuitive understanding.

<table><tr><td rowspan="2">epoch</td><td colspan="5">ViT-Tiny trained by BCE</td></tr><tr><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w w } } } }$ </td></tr><tr><td>50</td><td> $\overline { { . 9 7 2 6 \pm . 0 2 5 6 } }$ </td><td> $\overline { { . 4 9 4 7 \pm . 0 0 7 9 } }$ </td><td> $. 0 7 8 3 \pm . 0 1 9 9$ </td><td> $. 4 6 0 8 \pm . 0 0 7 6$ </td><td> $3 8 9 9 \pm . 0 0 6 6$ </td></tr><tr><td>100</td><td> $. 9 9 0 2 \pm . 0 0 4 5$ </td><td> $. 4 0 9 6 \pm . 0 0 4 1$ </td><td> $. 2 3 0 5 \pm . 0 0 5 3$ </td><td> $- . 3 9 2 9 \pm . 0 0 4 4$ </td><td> $. 3 3 0 2 \pm . 0 0 4 4$ </td></tr><tr><td>200</td><td> $. 9 9 4 9 \pm . 0 0 5 2$ </td><td> $. 2 6 0 1 \pm . 0 0 2 6$ </td><td> $. 5 0 0 1 \pm . 0 0 4 7$ </td><td> $- . 2 6 2 9 \pm . 0 0 3 2$ </td><td> $. 2 0 7 4 \pm . 0 0 3 4$ </td></tr><tr><td>300</td><td> $. 9 9 3 7 \pm . 0 0 3 8$ </td><td> $. 1 2 0 3 \pm . 0 0 3 7$ </td><td> $. 7 5 7 3 \pm . 0 0 4 0$ </td><td> $- . 1 3 6 8 \pm . 0 0 4 1$ </td><td> $. 0 8 4 4 \pm . 0 0 3 6$ </td></tr><tr><td>400</td><td> $. 9 9 5 4 \pm . 0 0 5 9$ </td><td> $. 0 0 5 2 \pm . 0 0 3 5$ </td><td> $. 9 6 8 8 \pm . 0 0 3 7$ </td><td> $- . 0 3 2 9 \pm . 0 0 3 4$ </td><td> $- . 0 1 3 6 \pm . 0 0 2 6$ </td></tr><tr><td>600</td><td> $. 9 9 3 9 \pm . 0 1 0 7$ </td><td> $- . 0 2 0 0 \pm . 0 0 4 5$ </td><td> $. 9 9 6 7 \pm . 0 0 5 6$ </td><td> $- . 0 2 0 4 \pm . 0 0 4 3$ </td><td> $- . 0 2 0 4 \pm . 0 0 3 4$ </td></tr><tr><td rowspan="2">epoch</td><td colspan="4">ViT-Tiny trained by CE</td><td rowspan="2"></td></tr><tr><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w w } } } }$ </td></tr><tr><td>50</td><td> $\overline { { . 9 0 9 2 \pm . 0 2 1 6 } }$ </td><td> $\overline { { . 0 0 5 7 \pm . 0 3 5 5 } }$ </td><td> $\overline { { . 9 4 2 2 \pm . 0 1 2 5 } }$ </td><td> $- . 0 1 4 7 \pm . 0 3 7 2$ </td><td> $- . 0 1 9 5 \pm . 0 3 8 2$ </td></tr><tr><td>100</td><td> $. 9 8 8 0 \pm . 0 0 1 4$ </td><td> $- . 0 1 0 1 \pm . 0 0 6 9$ </td><td> $. 9 9 0 0 \pm . 0 0 0 8$ </td><td> $- . 0 1 9 1 \pm . 0 0 6 9$ </td><td> $- . 0 2 0 3 \pm . 0 0 6 8$ </td></tr><tr><td>200</td><td> $. 9 8 8 2 \pm . 0 0 0 8$ </td><td> $- . 0 1 9 3 \pm . 0 0 6 0$ </td><td> $. 9 9 3 7 \pm . 0 0 0 4$ </td><td> $- . 0 2 0 2 \pm . 0 0 6 3$ </td><td> $- . 0 2 0 4 \pm . 0 0 6 5$ </td></tr><tr><td>300</td><td> $. 9 6 0 4 \pm . 0 1 8 6$ </td><td> $- . 0 1 8 5 \pm . 0 1 2 7$ </td><td> $. 9 7 8 3 \pm . 0 1 0 0$ </td><td> $- . 0 2 0 0 \pm . 0 1 2 9$ </td><td> $- . 0 2 0 4 \pm . 0 1 0 4$ </td></tr><tr><td>400</td><td> $. 9 7 4 3 \pm . 0 1 0 3$ </td><td> $- . 0 1 8 4 \pm . 0 1 3 2$ </td><td> $. 9 8 5 5 \pm . 0 0 5 6$ </td><td> $- . 0 2 0 1 \pm . 0 1 2 5$ </td><td> $- . 0 2 0 4 \pm . 0 0 9 6$ </td></tr><tr><td>600</td><td> $. 9 9 9 7 \pm . 0 0 0 3$ </td><td> $- . 0 2 0 4 \pm . 0 0 1 0$ </td><td> $. 9 9 9 8 \pm . 0 0 0 1$ </td><td> $- . 0 2 0 4 \pm . 0 0 1 2$ </td><td> $- . 0 2 0 4 \pm . 0 0 1 4$ </td></tr></table>

<table><tr><td rowspan="2">epoch</td><td colspan="4">ViT-Tiny trained by BCE</td></tr><tr><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { \| \tilde { \pmb { x } } \| } }$ </td></tr><tr><td>50</td><td> $\overline { { . 9 9 9 5 \pm . 0 0 0 1 } }$ </td><td> $\overline { { . 9 9 9 7 \pm . 0 0 0 2 } }$ </td><td> $\overline { { . 6 3 4 0 \pm . 0 0 3 0 } }$ </td><td>0.7101</td></tr><tr><td>100</td><td> $- . 9 9 9 9 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 5 8 6 2 \pm . 0 0 4 0$ </td><td>0.6490</td></tr><tr><td>200</td><td> $- . 9 9 9 9 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 4 7 2 5 \pm . 0 0 4 6$ </td><td>0.5243</td></tr><tr><td>300</td><td> ${ } . 9 9 9 8 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 3 2 0 5 \pm . 0 0 5 5$ </td><td>0.3712</td></tr><tr><td>400</td><td> $- . 9 9 9 2 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 8 1 5 \pm . 0 0 6 2$ </td><td>0.1581</td></tr><tr><td>600</td><td> $- . 6 1 8 1 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 0 2 \pm . 0 0 1 0$ </td><td>0.0160</td></tr><tr><td rowspan="2">epoch</td><td colspan="4">ViT-Tiny trained by CE</td></tr><tr><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { \| \tilde { \pmb { x } } \| } }$ </td></tr><tr><td>50</td><td> $\overline { { . 9 5 1 3 \pm . 0 0 1 2 } }$ </td><td> $\overline { { . 9 9 8 3 \pm . 0 0 1 6 } }$ </td><td> $\overline { { . 0 3 0 2 \pm . 0 1 2 1 } }$ </td><td>0.1542</td></tr><tr><td>100</td><td> $- . 9 9 9 6 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 0 1 0 5 \pm . 0 0 2 6$ </td><td>0.0991</td></tr><tr><td>200</td><td> $- . 9 9 7 7 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 0 0 1 1 \pm . 0 0 2 1$ </td><td>0.0289</td></tr><tr><td>300</td><td> $- . 2 5 0 4 \pm . 0 0 0 4$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 0 0 0 2 \pm . 0 0 9 6$ </td><td>0.0322</td></tr><tr><td>400</td><td> $- . 0 4 9 2 \pm . 0 0 0 4$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 0 1 \pm . 0 0 7 7$ </td><td>0.0378</td></tr><tr><td>600</td><td> $- . 0 4 4 2 \pm . 0 0 0 3$ </td><td> $. 9 9 9 9 \pm . 0 0 0 1$ </td><td> $. 0 0 0 0 \pm . 0 0 0 7$ </td><td>0.0013</td></tr></table>

Table 3: Statistical results $( \mathrm { m e a n } \pm \mathrm { s t d } )$ for diferent metrics of ViT-Tiny trained on CI-FAR100 with $K _ { 1 } = 5 0 .$

In Table 2, one can observe that for classifier vectors $\{ w _ { j } : K _ { 1 } < j \le K \}$ of empty classes, the means of $s _ { \mathrm { e m p } } ^ { \mathrm { w w } }$ and $s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } }$ are respectively close to 1 and 0, with very small standard deviations. These results indicate that $\{ \pmb { w } _ { j } : K _ { 1 } < j \leq K \}$ have basically collapsed to a single vector and are orthogonal to $\{ \pmb { w } _ { k } : k \in [ K _ { 1 } ] \}$ , the classifier vectors of nonempty known classes. With Theorems 2 and 3, the theoretical value of the metrics of $s _ { \mathrm { e m p } } ^ { \mathrm { w x } } \ \mathrm { i s \ - 1 }$ ; however, in Table 2, the $s _ { \mathrm { e m p } } ^ { \mathrm { w x } }$ of most models deviates markedly from −1. We conjecture that it is because when $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { k } } \end{array}$ is very close to $\mathbf { 0 } ,$ even minor angular variations between class centers $\{ \bar { \pmb { x } } _ { k } : k \overset { \cdot } { \in } \left[ K _ { 1 } \right] \}$ can substantially alter the direction of ${ \tilde { \mathbf { x } } } ,$ and consequently change its similarity with the classifier vectors of the empty unknown classes, i.e., $\pmb { w } _ { j } ^ { \top } \tilde { \pmb { x } } / \| \tilde { \pmb { x } } \| , K _ { 1 } < j \leq K _ { 1 }$

In Table 3, we present the statistical results of ViT-Tiny trained on CIFAR100 with $K _ { 1 } = 5 0$ using BCE and CE. Before the model training, the randomly initialized classifier vectors are almost mutually orthogonal. However, after training begins, the classifier vectors $\{ \pmb { w } _ { j } : K _ { 1 } < j \leq K \}$ of empty unknown classes quickly coincide with one another, as shown by the statistical results for $s _ { \mathrm { e m p } } ^ { \mathrm { w w } }$ , which are already very close to $1 \pm 0$ at epoch = 50. For $K _ { 1 } < j \le K$ , there are no samples from this unknown class attracting its classifier vector ${ \pmb w } _ { j }$ , while the classifier vectors of diferent unknown classes are subjected to exactly the same repulsive force; therefore, in a bounded space, they will ultimately coincide. Meanwhile, they become opposite in direction to the global feature averaged vector $\tilde { \mathbf { x } }$ , as shown by the statistical results for $s _ { \mathrm { e m p } } ^ { \mathrm { w x } }$ , which rapidly approach $- 1 \pm 0$ within the first 100 epochs. For $\{ \pmb { w } _ { j } : K _ { 1 } < j \leq K \}$ , they cannot indirectly receive repulsive forces from the classifier vectors $\{ w _ { k } : k \in [ K _ { 1 } ] \}$ of non-empty known classes through their own samples, and can only receive repulsive forces from all sample features of other classes, so ultimately they become opposite to the direction of the resultant repulsive force, x˜. However, as $\tilde { \mathbf { x } }$ gradually tends toward 0, for the model trained with BCE, the mean of $s _ { \mathrm { e m p } } ^ { \mathrm { w x } }$ at the 600th epoch is −0.6181, whereas for that trained with CE, at the 300th, 400th, and 600th epochs the means of $s _ { \mathrm { e m p } } ^ { \mathrm { w x } }$ change to −0.2504, −0.0492, and −0.0442, respectively, indicating that the angle between x˜ and $\{ \pmb { w } _ { j } : K _ { 1 } < j \leq K \}$ changes drastically, which is consistent with our earlier conjecture.

In the final stage of training, the structure of classifier vectors $\{ w _ { k } : k \in [ K _ { 1 } ] \}$ and the sample features of non-empty known classes are very close to an ETF. The numbers in $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ , which reflect the within-class collapse, and $s _ { \mathrm { p o s } } ^ { \mathrm { w x } }$ , which reflect the alignment between classifier vectors $\{ w _ { k } : k \in [ K _ { 1 } ] \}$ and class centers $\{ \bar { \pmb { x } } _ { k } : k \in [ K _ { 1 } ] \}$ , are both close to a mean of 1 and a standard deviation of 0. For $s _ { \mathrm { n e g } } ^ { \mathrm { x x } }$ and $s _ { \mathrm { n e g } } ^ { \mathrm { w w } } .$ , which reflect the betweenclass separation of sample features and classifier vectors, their means are both close to $- 1 / 4 9 \approx - 0 . 0 2 0 4$ , with relatively small standard deviations. These results indicate that, regardless of whether the models are trained with CE or BCE, NC is essentially achieved on the non-empty classes.

## 4.3 Contrastive Losses on Imbalanced Datasets

Finally, we verify Theorem 4 for contrastive learning on balanced and imbalanced datasets. In experiments for the Sigmoid-based contrastive loss, we set $\begin{array} { r } { \lambda = \frac { 1 } { K - 1 } ; } \end{array}$ ; then, $\begin{array} { r } { b ^ { \star } = \frac { \gamma } { 2 } \Big ( 1 - } \end{array}$ $\scriptstyle { \frac { 1 } { K - 1 } } \int$

Table 4 reports the statistical results of ResNet18 and ViT-Tiny trained using the rebalanced Softmax-based and Sigmoid-based losses on CIFAR10-LT and CIFAR100-LT with $\mathrm { I F } = 1$ , 10, 50, and 100. In these experiments, across settings with diferent numbers of classes and IFs, the mean of the final $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ consistently stays close to 1, while the mean of $s _ { \mathrm { n e g } } ^ { \mathrm { x x } }$ remains close to the corresponding theoretical extreme values, i.e., −1/9 on CIFAR10-

<table><tr><td rowspan="2">Loss</td><td rowspan="2">Dataset IF</td><td colspan="3">ResNet18</td></tr><tr><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } } }$ </td><td> $\frac { s \mathrm { x x } } { s \mathrm { n e g } }$ </td><td>b</td></tr><tr><td rowspan="8">Softmax-based</td><td>1</td><td> $\overline { { 0 . 9 9 9 6 \pm 0 . 0 0 0 1 } }$ </td><td> $\overline { { - 0 . 1 1 1 1 \pm 0 . 0 0 3 6 } }$ </td><td> $\overline { { 0 . 0 0 0 0 \pm 0 . 0 0 0 2 } }$ </td></tr><tr><td>10</td><td> $0 . 9 9 9 6 \pm 0 . 0 0 0 2$ </td><td> $- 0 . 1 1 1 0 \pm 0 . 0 0 3 1$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 0 4$ </td></tr><tr><td>CIFAR10 50</td><td> $0 . 9 9 9 6 \pm 0 . 0 0 0 2$ </td><td> $- 0 . 1 1 1 1 \pm 0 . 0 0 4 0$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 2 4$ </td></tr><tr><td>100</td><td> $0 . 9 9 9 5 \pm 0 . 0 0 0 3$ </td><td> $- 0 . 1 1 0 4 \pm 0 . 0 0 5 4$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 4 4$ </td></tr><tr><td>NC theo. value</td><td>1.0</td><td> $\bar { - 1 } \bar { / 9 } \overset { - } { \approx } - \bar { 0 } . \bar { 1 } \bar { 1 1 } \bar { 1 }$ </td><td></td></tr><tr><td>1</td><td> $\overline { { 0 . 9 9 9 6 \pm 0 . 0 0 0 5 } }$ </td><td> $- 0 . 0 1 0 1 \pm 0 . 0 0 2 9$ </td><td> $\overline { { 0 . 0 0 0 0 \pm 0 . 0 0 0 3 } }$ </td></tr><tr><td>10</td><td> $0 . 9 9 9 5 \pm 0 . 0 0 0 6$ </td><td> $- 0 . 0 1 0 1 \pm 0 . 0 0 3 1$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 0 3 4$ </td></tr><tr><td>CIFAR100 50</td><td> $0 . 9 9 9 6 \pm 0 . 0 0 0 3$ </td><td> $- 0 . 0 1 0 0 \pm 0 . 0 0 2 9$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 1 2 2$ </td></tr><tr><td rowspan="8"></td><td>100</td><td> $0 . 9 9 9 4 \pm 0 . 0 0 0 6$ </td><td> $- 0 . 0 0 9 9 \pm 0 . 0 0 3 3$ </td><td> $0 . 0 0 0 0 \pm 0 . 0 2 3 2$ </td></tr><tr><td>NC theo. value</td><td>1.0</td><td> $\bar { - 1 / 9 9 } \bar { \approx } - \bar { 0 . 0 1 } 0 \bar { 1 }$ </td><td></td></tr><tr><td>1</td><td> $\overline { { 0 . 9 9 9 8 \pm 0 . 0 0 0 2 } }$ </td><td> $- 0 . 1 1 1 1 \pm 0 . 0 0 3 5$ </td><td> $\overline { { 1 . 7 7 7 2 \pm 0 . 0 0 0 4 } }$ </td></tr><tr><td>10</td><td> $0 . 9 9 9 6 \pm 0 . 0 0 0 2$ </td><td> $- 0 . 1 1 1 0 \pm 0 . 0 0 4 3$ </td><td> $1 . 7 7 6 9 \pm 0 . 0 0 0 5$ </td></tr><tr><td>CIFAR10 50</td><td> $0 . 9 9 9 5 \pm 0 . 0 0 0 4$ </td><td> $- 0 . 1 1 0 8 \pm 0 . 0 0 6 1$ </td><td> $1 . 7 7 7 0 \pm 0 . 0 0 0 9$ </td></tr><tr><td>100</td><td></td><td> $- 0 . 1 1 0 6 \pm 0 . 0 0 6 1$ </td><td> $1 . 7 7 7 0 \pm 0 . 0 0 0 7$ </td></tr><tr><td>NC theo. value</td><td> $\begin{array} { c } { { 0 . 9 9 9 5 \pm 0 . 0 0 0 3 } } \\ { { - \ - \ - \ - \ \bar { 1 } . \bar { 0 } \ - \ - \ - \ } } \end{array}$ </td><td> $\bar { - 1 } \bar { / 9 } \overset { - } { \approx } \bar { - 0 } . \bar { 1 } \bar { 1 1 } \bar { 1 }$ </td><td>1.7778</td></tr><tr><td>1</td><td> $\overline { { 0 . 9 9 9 9 \pm 0 . 0 0 0 2 } }$ </td><td> $\overline { { - 0 . 0 1 0 1 \pm 0 . 0 2 9 0 } }$ </td><td> $\overline { { 1 . 9 8 0 9 \pm 0 . 0 0 2 0 } }$ </td></tr><tr><td rowspan="4"></td><td>10</td><td> $0 . 9 9 9 8 \pm 0 . 0 0 0 4$ </td><td></td><td> $1 . 9 7 8 8 \pm 0 . 0 0 3 0$ </td></tr><tr><td>CIFAR100 50</td><td> $0 . 9 9 9 7 \pm 0 . 0 0 0 8$ </td><td> $- 0 . 0 1 0 1 \pm 0 . 0 1 4 5$ </td><td></td></tr><tr><td>100</td><td> $0 . 9 9 9 5 \pm 0 . 0 0 1 1$ </td><td> $- 0 . 0 1 0 1 \pm 0 . 0 5 1 2$   $- 0 . 0 1 0 2 \pm 0 . 0 2 4 7$ </td><td> $1 . 9 8 4 1 \pm 0 . 0 0 2 6$ </td></tr><tr><td>NC theo. value</td><td>1.0</td><td> $= \bar { 1 } / \bar { 9 } 9 \overset { - } { \approx } \bar { - 0 . 0 1 } \bar { 0 1 }$ </td><td> $1 . 9 7 7 2 \pm 0 . 0 1 3 5$  1.9798</td></tr></table>

Table 4: Statistical results $( \mathrm { m e a n } \pm \mathrm { s t d } )$ of diferent models trained by Softmax-based and Sigmoid-based losses on CIFAR10-LT and CIFAR100-LT with IF = 1, 10, 50, and 100.

LT and −1/99 on CIFAR100-LT. Meanwhile, the standard deviations of these two kinds of quantities are relatively small. These results indicate that the re-balanced Softmax-based and Sigmoid-based loss functions using sample inverse frequencies efectively suppress the impact of class imbalance, and drive the learned features toward within-class collapse and uniformly separated class centers, as predicted by Theorem 4. Furthermore, for the Sigmoidbased loss, the learned biases also remain close to their theoretical values, approximately 1.7778 on CIFAR10-LT and 1.9798 on CIFAR100-LT, supporting the theoretical analysis.

Compared with the Softmax-based loss, the Sigmoid-based loss exhibits relatively larger standard deviations in $s _ { \mathrm { n e g } } ^ { \mathrm { x x } } ,$ particularly on CIFAR100-LT, although the means of $s _ { \mathrm { n e g } } ^ { \mathrm { x x } }$ remain close to the theoretical values. As the negative terms in the rebalanced Sigmoidbased loss are scaled by $\frac { 1 } { K - 1 }$ , when the number of classes, K, is large, their contribution to contrastive learning is smaller and between-class separation may not fully converge within a finite training time. We recommend adding a weighting parameter before the negative terms when using the Sigmoid-based loss in practice.

<table><tr><td rowspan="2">Loss</td><td rowspan="2">Dataset</td><td rowspan="2">IF</td><td colspan="3">ViT-Tiny</td></tr><tr><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { x x } } } }$ </td><td> $\frac { s \mathrm { x x } } { \mathrm { n e g } }$ </td><td>b</td></tr><tr><td rowspan="8">Softmax-based</td><td></td><td>1</td><td>0.9998 ± 0.0001</td><td>−0.1111 ± 0.0036</td><td>0.0002 ± 0.0002</td></tr><tr><td>10</td><td></td><td>0.9997 ± 0.0001</td><td>−0.1111 ± 0.0040</td><td>0.0000 ± 0.0007</td></tr><tr><td>CIFAR10</td><td>50</td><td>0.9996 ± 0.0002</td><td>−0.1110 ± 0.0057</td><td>0.0000 ± 0.0024</td></tr><tr><td></td><td>100</td><td>0.9997 ± 0.0001</td><td>−0.1109 ± 0.0053</td><td>0.0000 ± 0.0046</td></tr><tr><td>NC theo. value</td><td></td><td>1.0</td><td>−1/9≈−0.1111</td><td></td></tr><tr><td></td><td>1</td><td>0.9998 ± 0.0001</td><td>−0.0101 ± 0.0026</td><td>0.0000 ± 0.0000</td></tr><tr><td>10</td><td></td><td>0.9998 ± 0.0001</td><td>−0.0101 ± 0.0037</td><td>0.0000 ± 0.0034</td></tr><tr><td>CIFAR100 50</td><td></td><td>0.9997 ± 0.0001</td><td>−0.0100 ± 0.0300</td><td>0.0000 ± 0.0132</td></tr><tr><td rowspan="8"></td><td>100</td><td></td><td>0.9997 ± 0.0001</td><td>−0.0100 ± 0.0321</td><td>0.0000 ± 0.0241</td></tr><tr><td>NC theo. value</td><td></td><td>1.0</td><td>-1/99≈−0.0101</td><td></td></tr><tr><td></td><td>1</td><td>0.9996 ± 0.0001</td><td>−0.1111 ± 0.0045</td><td>1.7790 ± 0.0020</td></tr><tr><td></td><td>10</td><td>0.9999 ± 0.0000</td><td>−0.1111 ± 0.0020</td><td>1.7777 ± 0.0003</td></tr><tr><td>CIFAR10</td><td>50</td><td>0.9999 ± 0.0000</td><td>−0.1111 ± 0.0021</td><td>1.7776 ± 0.0002</td></tr><tr><td>100</td><td></td><td>0.9999 ± 0.0000</td><td>−0.1110 ± 0.0015</td><td>1.7776 ± 0.0005</td></tr><tr><td>NC theo. value</td><td></td><td>1.0</td><td>−1/9≈ −0.1111</td><td>1.7778</td></tr><tr><td>1</td><td>0.9999 ± 0.0000</td><td></td><td>−0.0101 ± 0.0245</td><td>1.9827 ± 0.0028</td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10</td><td>0.9996 ± 0.0002</td><td>−0.0101 ± 0.0655</td><td></td><td>1.9894 ± 0.0043</td></tr><tr><td>CIFAR100 50</td><td>0.9997 ± 0.0001</td><td></td><td>−0.0101 ± 0.0550</td><td>1.9835 ± 0.0045</td></tr><tr><td>100</td><td>0.9997 ± 0.0001</td><td></td><td>−0.0101 ± 0.0616</td><td>1.9841 ± 0.0052</td></tr><tr><td></td><td>NC theo. value</td><td></td><td>1.0</td><td>-1/99≈−0.0101</td><td>1.9798</td></tr></table>

## 5 Conclusion

This paper reformulates neural collapse (NC) in imbalanced datasets and open-set scenarios. Unlike the prior works that use globally averaged losses, we analyze the minimizers of classwise (CW) averaged losses re-balanced by sample frequency of each class. When the CW averaged losses reach their minima, the sample features and the classifier vectors form a standard NC, on both balanced and imbalanced datasets. Furthermore, we study in an open-set scenario that includes empty classes, the geometric structure of features and classifier vectors under extreme conditions. In the open-set scenario, when the CW averaged losses achieve their minima, a standard NC still formed on the known classes that contain samples. For the unknown classes that contain no samples, the classifier vectors collapse into a single vector, which is orthogonal to the classifier vectors of the known classes. Finally, for contrastive learning on imbalanced datasets, the losses re-balanced by the inverse sample frequency can also achieve standard NC.

Data Availability. The original CIFAR10 and CIFAR100 datasets used in this paper are publicly available, see https://cave.cs.toronto.edu/kriz/cifar.html for details.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Natural Science Foundation of China (Grant Nos. 62576217 and 62576216), Guangdong Provincial Key Laboratory (Grant No. 2023B1212060076), and Natural Science Foundation of Ningxia Province (Grant No. 2025AAC020002). This work was also supported by the Intelligent Computing Center of Shenzhen University.

## Appendix: Supplementary Material

This supplementary material provides additional experimental settings and results, as well as complete proofs of the theoretical results presented in the main paper.

## S.1 Experimental Settings and More Results

We summarize the experimental settings for supervised classification and supervised contrastive training, and provide additional results for ResNet18 and ViT-Tiny on CIFAR10 and CIFAR100.

## S.1.1 Experimental Settings

We conduct experiments using ResNet18 and ViT-Tiny on CIFAR10 and CIFAR100 (Krizhevsky and Hinton, 2009) to verify the theorems. Table S.1 summarizes the detailed training settings for classification, open-set scenarios, and supervised contrastive learning. All models are trained from scratch. The feature dimensions are d = 512 for ResNet18 and d = 192 for ViT-Tiny. No additional projection head or terminal ReLU activation is used. All input images are channel-wise normalized, without weight decay, feature regularization, warmup, label smoothing, or other data augmentation. For optimization, SGD is with a momentum of 0.9, and AdamW is with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$

<table><tr><td colspan="2" rowspan="2"></td><td colspan="2">Classification Training</td><td rowspan="2">Open-set Scenario</td><td colspan="2">Supervised Contrastive Learning</td></tr><tr><td>ResNet18 ViT-Tiny</td><td>ResNet18</td><td>ViT-Tiny</td><td>ResNet18 ViT-Tiny</td></tr><tr><td>epochs</td><td>800</td><td>800</td><td>1000</td><td>1000</td><td>5000</td><td>5000</td></tr><tr><td>H-əter optimizer</td><td></td><td>SGD AdamW</td><td>SGD</td><td>AdamW</td><td>SGD</td><td>AdamW</td></tr><tr><td>batch size</td><td>128</td><td>128</td><td>128</td><td>128</td><td>8192</td><td>8192</td></tr><tr><td>learning rate</td><td>0.01</td><td>0.0001</td><td>0.01</td><td>0.0001</td><td>0.5</td><td>0.0002</td></tr><tr><td>LR schedule</td><td>step</td><td>step</td><td>step</td><td>step</td><td>step</td><td>cosine</td></tr><tr><td>scale factor</td><td> $\gamma = 8$ </td><td> $\gamma = 8$ </td><td> $\gamma = 8$ </td><td> $\gamma = 8$ </td><td> $\gamma = 4$ </td><td>γ = 4</td></tr><tr><td>weight decay λ</td><td>x</td><td>X</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>warmup epochs</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Normalization</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table S.1: Detailed experimental settings.

<table><tr><td rowspan="2">Dataset</td><td colspan="3"> $n _ { \mathrm { m a x } }$  for classification</td><td colspan="3"> $n _ { \mathrm { m a x } }$  for contrastive learning</td></tr><tr><td> $\mathrm { I F } = 1 0 $ </td><td> $\mathrm { { I F } = 5 0 }$ </td><td> $\mathrm { I F } = 1 0 0$ </td><td> $\mathrm { I F } = 1 0 $ </td><td> $\mathrm { { I F } = 5 0 }$ </td><td> $\mathrm { I F } = 1 0 0$ </td></tr><tr><td>CIFAR10-LT</td><td>5000</td><td>5000</td><td>5000</td><td>2005</td><td>2927</td><td>3302</td></tr><tr><td>CIFAR100-LT</td><td>500</td><td>500</td><td>500</td><td>210</td><td>325</td><td>378</td></tr></table>

Table S.2: The number of samples, $n _ { \mathrm { m a x } }$ , of the largest class in CIFAR10-LT and CIFAR100-LT for classification and contrastive learning.

We achieve CIFAR10-LT and CIFAR100-LT (Cao et al., 2019) by sampling from the original CIFAR10 and CIFAR100, such that the number of samples in class k is

$$
n _ { k } = \left\lfloor n _ { \operatorname* { m a x } } * \mathrm { I F } ^ { - ( k - 1 ) / ( K - 1 ) } \right\rfloor , \quad \mathrm { I F } \in \{ 1 , 1 0 , 5 0 , 1 0 0 \} , \quad k \in [ K ] , \quad K \in \{ 1 0 , 1 0 0 \} .\tag{S.1}
$$

For classification experiments, $n _ { \mathrm { m a x } } = 5 0 0 0$ on CIFAR10 and $n _ { \mathrm { m a x } } = 5 0 0$ on CIFAR100, and $\mathrm { I F } = 1$ corresponds to the original balanced datasets. For supervised contrastive learning, we construct long-tailed datasets using the same sampling rules but set $n _ { \mathrm { m a x } }$ to a smaller value to obtain smaller long-tailed datasets, which accelerates the convergence of the models in contrastive learning on the imbalanced datasets.

## S.1.2 More Results on Open-set Scenario

Table S.3 reports the results of ViT-Tiny trained on CIFAR10 and CIFAR100 in open-set scenarios. In total, ViT-Tiny exhibits the same behaviors predicted by Theorems 2 and 3, demonstrating that the theoretical conclusions are not specific to the CNN architecture.

For the known classes, $k \in [ K _ { 1 } ]$ , their sample features exhibit strong within-class collapse, as indicated by $s _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ being close to 1 in most configurations. Meanwhile, their classifier vectors approximately form an ETF, with $s _ { \mathrm { n e g } } ^ { \mathrm { w w } }$ closely matching the theoretical value $- 1 / ( K _ { 1 } - 1 )$ . For example, on CIFAR100 with $K _ { 1 } = 3 0$ , the mean and standard deviation of $s _ { \mathrm { n e g } } ^ { \mathrm { w w } }$ are $- 0 . 0 3 4 5 { \pm } 0 . 0 0 1 0$ for the model trained by CE, and they are $- 0 . 0 3 4 5 { \pm } 0 . 0 0 0 5$ for the model trained by BCE, both closely matching the theoretical value $- 1 / 2 9 \approx - 0 . 0 3 4 5$

The classifier vectors, $\{ \pmb { w } _ { j } : K _ { 1 } < j \leq K \}$ , for the empty unknown classes also exhibit the predicted collapse behavior. $s _ { \mathrm { e m p } } ^ { \mathrm { w w } }$ is consistently close to 1, indicating that the classifier vectors of the empty classes become nearly identical. Meanwhile, $s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } }$ remains close to 0, showing that these collapsed empty-class classifier vectors are approximately orthogonal to the classifier vectors of the known classes. When $\| \tilde { \pmb x } \|$ is close to zero, the similarities between the empty-class classifier vectors $\{ w _ { j } : K _ { 1 } < j \le K \}$ and $\tilde { \mathbf { x } }$ are correspondingly small. For example, for the model trained by CE on CIFAR100 with $K _ { 1 } = 5 0 , \| \tilde { \mathbfit { x } } \| =$ 0.0013 and $s _ { \mathrm { e m p } } ^ { \mathrm { w x } } = - 0 . 0 4 4 2 \pm 0 . 0 0 0 3$ , which does not conform to the conclusion of the theorem. We believe this stems from the fact that when the magnitude of x˜ is too small, its direction is prone to being afected by interference. In contrast, when the magnitude of x˜ is large enough, the empty-class classifier vectors become strongly aligned with the direction predicted by the theory. For the model trained by BCE on CIFAR100 with $K _ { 1 } = 3 0$ $\| \tilde { \pmb { x } } \| = 0 . 0 8 0 8$ and $s _ { \mathrm { e m p } } ^ { \mathrm { w x } } = - 1 . 0 0 0 \pm 0 . 0 0 0 0$

In total, the experimental results of ViT-Tiny consistently reproduce the similar geometric structures observed in the experiments of ResNet18, including within-class collapse, ETF-like geometry among known-class classifiers, collapse of empty-class classifiers, and their near orthogonality to the known-class classifiers.

## S.1.3 Statistical Results for Models Trained on Imbalanced Datasets

Figs. S.1–S.4 visually present statistical results for ResNet18 and ViT-Tiny trained on diferent imbalanced datasets. Consistent with the conclusions of the theorem, for models trained with CW averaged losses, their negative scaled similarity $\gamma \pmb { w } _ { k } ^ { \top } \bar { \pmb { x } } _ { j }$ is close to the NC value $- \gamma / ( K - 1 )$ in most cases, especially for the models trained with BCE, where this remains true even as the IF increases. In contrast, for models trained with G. averaged losses, the negative feature similarity becomes increasingly more dispersed as the IF increases. For the models trained by diferent losses, their positive similarity remains close to γ in most settings. For the BCE loss, the learned bias also tends toward its theoretical value.

<table><tr><td rowspan="2">D</td><td rowspan="2"> $K _ { 1 }$ </td><td colspan="5">ViT-Tiny trained using CE</td></tr><tr><td> $\frac { s \mathrm { x x } } { \mathrm { p o s } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { x x } } } }$ </td><td> $\overline { { s _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w w } } } }$ </td></tr><tr><td></td><td>9</td><td> $\overline { { 1 . 0 0 0 \pm . 0 0 0 0 } }$ </td><td> $\overline { { . 1 2 5 0 \pm . 0 0 0 1 } }$ </td><td> $\overline { { 1 . 0 0 0 \pm . 0 0 0 0 } }$ </td><td> $\overline { { . 1 2 5 0 \pm . 0 0 1 9 } }$ </td><td> $\overline { { . 1 2 5 0 \pm . 0 0 2 9 } }$ </td></tr><tr><td></td><td>7</td><td> $. 9 9 0 9 \pm . 0 0 2 3$ </td><td> $- . 1 6 4 6 \pm . 0 2 0 7$ </td><td> $. 9 9 4 8 \pm . 0 0 1 4$ </td><td> $- . 1 6 5 8 \pm . 0 1 7 0$ </td><td> $- . 1 6 6 6 \pm . 0 1 1 7$ </td></tr><tr><td>CI10</td><td>5</td><td> $. 9 9 9 6 \pm . 0 0 0 0$ </td><td> $- . 2 4 9 9 \pm . 0 0 0 4$ </td><td> $. 9 9 9 8 \pm . 0 0 0 0$ </td><td> $- . 2 4 9 9 \pm . 0 0 2 0$ </td><td> $- . 2 5 0 0 \pm . 0 0 3 1$ </td></tr><tr><td></td><td>3</td><td> $. 9 9 7 9 \pm . 0 0 0 2$ </td><td> $- . 4 9 7 7 \pm . 0 0 2 0$ </td><td> $. 9 9 8 4 \pm . 0 0 0 1$ </td><td> $- . 4 9 9 2 \pm . 0 0 2 9$ </td><td> $- . 4 9 9 8 \pm . 0 0 4 6$ </td></tr><tr><td>CI10</td><td>90</td><td> $9 8 8 1 \pm . 0 0 6 0$ </td><td> $- . 0 1 0 7 \pm . 0 0 6 7$ </td><td> $\overline { { . 9 9 3 5 \pm . 0 0 3 2 } }$ </td><td> $- . 0 1 1 2 \pm . 0 0 6 7$ </td><td> $- . 0 1 1 2 \pm . 0 0 6 0$ </td></tr><tr><td></td><td>70</td><td> $. 9 9 7 2 \pm . 0 0 0 5$ </td><td> $- . 0 1 4 5 \pm . 0 0 4 5$ </td><td> $. 9 9 8 6 \pm . 0 0 0 2$ </td><td> $- . 0 1 4 5 \pm . 0 0 4 5$ </td><td> $- . 0 1 4 5 \pm . 0 0 4 4$ </td></tr><tr><td></td><td>50</td><td> $. 9 9 9 7 \pm . 0 0 0 3$ </td><td> $- . 0 2 0 4 \pm . 0 0 1 0$ </td><td> $. 9 9 9 8 \pm . 0 0 0 1$ </td><td> $- . 0 2 0 4 \pm . 0 0 1 2$ </td><td> $- . 0 2 0 4 \pm . 0 0 1 4$ </td></tr><tr><td></td><td>30</td><td> $1 . 0 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 0 3 4 5 \pm . 0 0 0 2$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 0 3 4 5 \pm . 0 0 0 7$ </td><td> $- . 0 3 4 5 \pm . 0 0 1 0$ </td></tr><tr><td>D</td><td></td><td colspan="5">ViT-Tiny trained using BCE</td></tr><tr><td rowspan="4">CIAA10</td><td> $K _ { 1 }$ </td><td> $\underline { s } _ { \mathrm { p o s } } ^ { \mathrm { x x } }$ </td><td> $\underline { s } _ { \mathrm { n e g } } ^ { \mathrm { x x } }$ </td><td> $\overline { { \underline { { s } } _ { \mathrm { p o s } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { n e g } } ^ { \mathrm { w w } } } }$ </td></tr><tr><td>9</td><td> $. 9 9 9 8 \pm . 0 0 0 0$ </td><td> $. 1 2 5 0 \pm . 0 0 0 2$ </td><td> $. 9 9 9 9 \pm . 0 0 0 0$ </td><td> $. 1 2 5 0 \pm . 0 0 0 8$ </td><td> $. 1 2 5 0 \pm . 0 0 1 2$ </td></tr><tr><td>7</td><td> $. 9 9 5 1 \pm . 0 0 2 0$ </td><td> $- . 1 6 5 7 \pm . 0 0 3 7$ </td><td> $. 9 9 7 4 \pm . 0 0 1 1$ </td><td> $- . 1 6 6 3 \pm . 0 0 3 9$ </td><td> $- . 1 6 6 6 \pm . 0 0 2 3$ </td></tr><tr><td>5</td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 2 5 0 0 \pm . 0 0 0 3$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 2 5 0 0 \pm . 0 0 1 7$ </td><td> $- . 2 5 0 0 \pm . 0 0 2 8$ </td></tr><tr><td rowspan="4">CI10</td><td>3</td><td> $. 9 9 9 9 \pm . 0 0 0 0$ </td><td> $- . 4 9 9 2 \pm . 0 0 0 3$ </td><td> $. 9 9 9 7 \pm . 0 0 0 0$ </td><td> $- . 4 9 9 9 \pm . 0 0 0 3$ </td><td> $- . 5 0 0 0 \pm . 0 0 0 5$ </td></tr><tr><td>90</td><td> $\overline { { . 9 9 2 5 \pm . 0 0 5 3 } }$ </td><td> $- . 0 1 1 0 \pm . 0 0 3 5$ </td><td> $\overline { { . 9 9 6 1 \pm . 0 0 2 7 } }$ </td><td> $- . 0 1 1 2 \pm . 0 0 3 4$ </td><td> $- . 0 1 1 2 \pm . 0 0 2 8$ </td></tr><tr><td>70</td><td> $. 9 9 9 7 \pm . 0 0 0 2$ </td><td> $- . 0 1 4 5 \pm . 0 0 0 5$ </td><td> $. 9 9 9 8 \pm . 0 0 0 1$ </td><td> $- . 0 1 4 5 \pm . 0 0 0 7$ </td><td> $- . 0 1 4 5 \pm . 0 0 0 8$ </td></tr><tr><td>50</td><td> $. 9 9 9 8 \pm . 0 0 0 2$ </td><td> $- . 0 2 0 4 \pm . 0 0 0 3$ </td><td> $. 9 9 9 9 \pm . 0 0 0 1$ </td><td> $- . 0 2 0 4 \pm . 0 0 0 8$ </td><td> $- . 0 2 0 4 \pm . 0 0 1 1$ </td></tr><tr><td></td><td>30</td><td> $. 9 9 9 9 \pm . 0 0 0 0$ </td><td> $- . 0 2 7 7 \pm . 0 0 0 1$ </td><td> $. 9 9 6 6 \pm . 0 0 0 1$ </td><td> $\cdot . 0 3 4 5 \pm . 0 0 0 4$ </td><td> $. . 0 3 4 5 \pm . 0 0 0 5$ </td></tr></table>

Table S.3: Statistical results (mean ± std) for diferent metrics of ViT-Tiny trained by CE and BCE on CIFAR10 and CIFAR100 with $K - K _ { 1 }$ empty classes. The CE and BCE are class-wise averaged using Eqs. (12) and (14).

## S.1.4 Similarity Matrices for Models Trained on Imbalanced Datasets

Figs. S.5-S.13 present the similarity matrices of feature centers and classifier vectors for ResNet18 trained with CW averaged losses and G. averaged losses. For models trained with CW averaged losses, the of-diagonal values in the similarity matrices are predominantly close to the NC value $- 1 / ( K - 1 )$ , consistent with the conclusion of the theorem. In contrast, for models trained with G. averaged losses, the between-class separation becomes obviously diverse, manifested as a gradual increase in the of-diagonal values of the matrix from the upper-left to the lower-right corner, exhibiting a clear correlation with the sample frequencies from the head classes to the tail classes. These results demonstrate that on imbalanced datasets, the CW. averaged losses maintain balanced between-class separability, whereas the G. averaged losses do not.

<table><tr><td rowspan="2">D</td><td rowspan="2"> $K _ { 1 }$ </td><td colspan="4">ViT-Tiny trained using CE</td></tr><tr><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w x } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { | | \tilde { \pmb { x } } | | } }$ </td></tr><tr><td></td><td>9</td><td> $\overline { { . 1 1 7 7 \pm . 0 0 0 0 } }$ </td><td></td><td> $\overline { { . 0 0 0 1 \pm . 0 0 1 5 } }$ </td><td>0.0002</td></tr><tr><td rowspan="4">CI10</td><td>7</td><td> $- . 7 7 1 2 \pm . 0 0 4 3$ </td><td> $. 9 9 9 7 \pm . 0 0 0 2$ </td><td> $- . 0 0 0 4 \pm . 0 0 7 2$ </td><td>0.0217</td></tr><tr><td>5</td><td> $- . 9 9 6 5 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 0 0 \pm . 0 0 0 9$ </td><td>0.0057</td></tr><tr><td>3</td><td> $- . 9 1 1 3 \pm . 0 0 1 6$ </td><td> $. 9 9 9 9 \pm . 0 0 0 1$ </td><td> $. 0 0 0 0 \pm . 0 0 5 3$ </td><td>0.0292</td></tr><tr><td>90</td><td> $- . 0 0 1 1 \pm . 0 0 0 0$ </td><td> $\overline { { . 9 9 9 7 \pm . 0 0 0 4 } }$ </td><td> $\overline { { . 0 0 0 0 \pm . 0 0 4 8 } }$ </td><td>0.0211</td></tr><tr><td rowspan="4">CI10</td><td>70</td><td> $- . 0 1 5 0 \pm . 0 0 1 6$ </td><td> $. 9 9 9 9 \pm . 0 0 0 2$ </td><td> $. 0 0 0 0 \pm . 0 0 3 0$ </td><td>0.0012</td></tr><tr><td>50</td><td> $. 0 4 4 2 \pm . 0 0 0 3$ </td><td> $. 9 9 9 9 \pm . 0 0 0 1$ </td><td> $. 0 0 0 0 \pm . 0 0 0 7$ </td><td>0.0013</td></tr><tr><td>30</td><td> $- . 9 9 2 9 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td> $. 0 0 0 0 \pm . 0 0 0 5$ </td><td>0.0039</td></tr><tr><td rowspan="2">D</td><td> $K _ { 1 }$   $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w x } } } }$ </td><td></td><td>ViT-Tiny trained using BCE</td><td></td><td></td></tr><tr><td></td><td></td><td> $\overline { { s _ { \mathrm { e m p } } ^ { \mathrm { w w } } } }$ </td><td> $\overline { { s _ { \mathrm { e m p 2 } } ^ { \mathrm { w w } } } }$ </td><td>||x||</td></tr><tr><td rowspan="5">CI10</td><td>9 7</td><td> $- . 0 3 7 0 \pm . 0 0 0 0$ </td><td></td><td> $\overline { { . 0 0 0 1 \pm . 0 0 1 0 } }$ </td><td>0.0004</td></tr><tr><td></td><td> $. 7 5 5 4 \pm . 0 0 0 1$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 7 1 \pm . 0 0 2 4$ </td><td>0.0107</td></tr><tr><td>5</td><td> $. 0 5 5 1 \pm . 0 0 0 1$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $- . 0 0 0 1 \pm . 0 0 0 2$ </td><td>0.0003</td></tr><tr><td>3</td><td> $- . 9 9 9 8 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 1 8 \pm . 0 0 3 8$ </td><td>0.0220</td></tr><tr><td>90</td><td> $- . 0 0 3 2 \pm . 0 0 0 0$ </td><td> $\overline { { 1 . 0 0 0 \pm . 0 0 0 0 } }$ </td><td> $\overline { { . 0 0 0 1 \pm . 0 0 1 5 } }$ </td><td>0.0104</td></tr><tr><td>CI10</td><td>70</td><td> $- . 0 1 2 0 \pm . 0 0 0 1$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $. 0 0 0 0 \pm . 0 0 0 3$ </td><td></td></tr><tr><td></td><td></td><td></td><td> $1 . 0 0 0 \pm . 0 0 0 0$ </td><td></td><td>0.0007</td></tr><tr><td></td><td>50</td><td> $- . 5 9 9 2 \pm . 0 0 0 0$ </td><td></td><td> $. 0 0 0 0 \pm . 0 0 0 4$ </td><td>0.0005</td></tr><tr><td></td><td>30</td><td> $- 1 . 0 0 0 \pm . 0 0 0 0$ </td><td> $1 . 0 0 0 \pm . 0 0 0 0$  Table S 3 (continued)</td><td> $. 0 0 1 1 \pm . 0 0 1 2$ </td><td>0.0808</td></tr></table>

Table S.3 (continued)

![](images/b5a17dd661fb74547431712a1c86ebb4c9be35f52a1a4a40ba71e46efeb693d1.jpg)  
(a) BCE

![](images/aeb477ee07577cdb1893ec8572d1566cce070c0b29f0e74f23748ee95fb3c6d9.jpg)  
(b) CE  
Figure S.1: Distributions of positive and negative scaled similarities and biases for ResNet18 trained on CIFAR10-LT. $\mathrm { ^ { 6 6 } C W ^ { 5 9 } }$ and $\mathrm { ^ { 6 6 } G } . ^ { \mathrm { 9 9 } }$ denote class-wise and globally averaged losses, respectively.

![](images/406cf6e37e081908f7bd5b5890889ffdd45d747a45979eafaa5b26bdd26ba990.jpg)  
(a) BCE

![](images/fb362d2636e0720adf525c8cda7a80924471e100b821c239bb97eda1b5ad05b6.jpg)  
(b) CE  
Figure S.2: Distributions of positive and negative scaled similarities and biases for ResNet18 trained on CIFAR100-LT. $\mathrm { ^ { 6 6 } C W } ^ { \mathrm { 9 9 } }$ and $^ { 6 6 } \mathrm { G } . ^ { \mathfrak { n } }$ denote class-wise and globally averaged losses, respectively.

![](images/652a369322c9c8d0a6673bcff62936850cfbc5db29d1298219326a9d88233724.jpg)  
(a) BCE

![](images/b1c6fd8b00b5535e7cb94a102e314470640d9d76e9d9cde5dbb5f57cfda7a727.jpg)  
(b) CE  
Figure S.3: Distributions of positive and negative scaled similarities and biases for ViT-Tiny trained on CIFAR10-LT. “CW” and “G.” denote class-wise and globally averaged losses, respectively.

![](images/65540ac99e472a2c7015e06cbcd7363a878561115129b648def0e9131dfacaa5.jpg)  
(a) BCE

![](images/8088cb7dec6a643b904db0f4a423392e3bf4105ad07fbcf9e6172bd216222fdd.jpg)  
(b) CE  
Figure S.4: Distributions of positive and negative scaled similarities and biases for ViT-Tiny trained on CIFAR100-LT. “CW” and $\mathrm { ^ { 6 6 } G } . ^ { \mathfrak { n } }$ denote class-wise and globally averaged losses, respectively.

![](images/8ea77a988b96cdd308716c8b7dc6a5c5218fa5bba9156f2c035115c094f892f3.jpg)

![](images/9faf580d0dddea11707f726bbde8ca466a54ef5f57e410a431353d52abac6de5.jpg)

![](images/a9521c867e53c70c2549bc0cc34c1388bea504e32e3d3202b9751a9144845085.jpg)  
(a) CW averaged BCE

![](images/025c8702b09557fdab610a14c52a5e5f7b58dfbe031bba45fab3e9b95203eac9.jpg)

![](images/2c6c1b8c7682eb291455910c64e932b987d17dd290d794cd2b00be4b18abfd26.jpg)  
(b) G. averaged BCE

![](images/6ae5a459698c18a47166621fee2fe41fe51fc55600a4d9f129db5918243cfdb5.jpg)  
Figure S.5: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ of ResNet18 trained with BCE on CIFAR10-LT with $\mathrm { I F } { = } 1 0 .$

![](images/608f1ee2739d8668fb424a090be7fd25b7f6d2c4c2eacf6235b9ca8c3720d70c.jpg)

![](images/7031f6e12028e4b4cc0cfdf8b77f8e29aaa0c2c2e917d64ff0db53de012eef22.jpg)  
(a) CW averaged BCE

![](images/d13701d9d85ea9de8fce69db2300b53d543a51119c0c08ddef9add52a01d66a0.jpg)

![](images/6a0b51a122c370c3304f3a688be4e80b22cb6e1a31216996c6659be47a11a3a3.jpg)

![](images/d258e13176f305a6173e2cc1538d480171f29fa6e7be3624fb38615abbe1c6a9.jpg)  
(b) G. averaged BCE

![](images/fc923f5663d9b338bfbf083ba8f645bddcf368852d35df2fb5284931caca580d.jpg)

Figure S.6: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ of ResNet18 trained with BCE on CIFAR10-LT with $\mathrm { { I F } = 5 0 }$  
![](images/04152ae1d90423c94b6cb262b7717ded94c6724cd880ef0d1e27cd2030004f72.jpg)

![](images/e6faee83c91336dfc8a8636c84ab5f58f97a307d0837eb50812e7a41bfeaa655.jpg)

![](images/ceeb5053b1eb288edf269accfe597bfb86f2367ce03e1080646b2862189ab274.jpg)  
(a) CW averaged BCE

![](images/fddef5fcef5b8a3dd32221a6e77af3d27d06ffaecb5d241a2d3adf501c66d208.jpg)

![](images/1315b1158047185ae0e524a56b221789f3eb5432ea311023eda15fea8994f9d7.jpg)  
(b) G. averaged BCE

![](images/df20cf9807ef351beedf41976e67a5a863c7d20b19158cdd92eac6340271a7ab.jpg)  
Figure S.7: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ ResNet18 trained with BCE on CIFAR10-LT with $\mathrm { { I F } = 1 0 0 }$

![](images/693ab19e5d8a929e4a23abd430f3792b4d4c0b48ed8228b734abee1eb63a64e3.jpg)

![](images/e49e4523eeab49e01d41d95227bda56028834693f108edb91ea110c0fa68f67a.jpg)  
(a) CW averaged CE

![](images/5c172dbf7f984dd303bba4a1ba18a2365ea7a8d5b38a5d8779bc298ef001879b.jpg)

![](images/93b4d2e2b87ee1a92869109200b3f79afc7043d87322769705e0956cd8a61f31.jpg)

![](images/02b026140579eb1cf53d4f2b3232b7da4b001920e7bd20a9fabd9dd03548b696.jpg)  
(b) G. averaged CE

![](images/76dc300c1ef50e6cdf184c2f81c49705e39934a81d3d64ec054eb78eefd22109.jpg)

Figure S.8: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ of ResNet18 trained with CE on CIFAR10-LT with $\mathrm { I F } { = } 1 0 .$  
![](images/5aed8377f4f32f22b0e9f3b573cfcf093234d1a0b4d3301716d4841e993168ad.jpg)

![](images/63124f06eb09285627e2633fb1cbbde497d223ddfa84536752b20e32e69ae07f.jpg)

![](images/76d8daba6f50635699703b080ab105024af212abfb7fa48ac1694746aa9a48a2.jpg)  
(a) CW averaged CE

![](images/30d4d2bf1db149988229aa86e8e339776f6a1fb6ce8ed10131f0289267048524.jpg)

![](images/aff499903b66ede60c438d44dd63d8e6217d9725616b317fc8f76633e6202800.jpg)  
(b) G. averaged CE

![](images/3d16719e039b181b84386c02b4ffb4ada6531207f27c84ea9b3e736f5d662b49.jpg)  
Figure S.9: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ ResNet18 trained with CE on CIFAR10-LT with $\mathrm { { I F } = 5 0 }$

![](images/b9d5da647b7cfeef18a3186cc923cf55d3376d316fad73787492a94e751ffc9f.jpg)

![](images/af1e163a15aa755bffa9e9c32170e46fe24fc292d5dab18afe6a66a06a3500d5.jpg)  
(a) CW averaged CE

![](images/ed9de6742bd7f346fe2eeaa37da9b1d8d9f20ce68ecfa1dc25d58af8e4bb4460.jpg)

![](images/c701fc05ce0ea0e22bbeb783b35d7ea665271c6c3a8cc36422c0a48cb88135ef.jpg)

![](images/40cdaf3a2967569720ab58bf5505d064898cdc518821abbd2f1b1e4880b5b32d.jpg)  
(b) G. averaged CE

![](images/1e49b3b970c2f3feb123e8cd1d24736d401aa42f29867e2a44187d2d9c13d89e.jpg)

Figure S.10: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ of ResNet18 trained with CE on CIFAR10-LT with $\mathrm { { I F } = 1 0 0 }$  
![](images/1b1b1bc8ff8425ee240fe086c3b162e509b36bd3ef18c7ce5c57e4d1e5739ece.jpg)

![](images/a93c7d7d86d359513fa9afe825c68812f83aad53cb062f7b09be94b6e764b7d9.jpg)

![](images/278c10b65799972f735883c6c49687656091ed5145a087e163b3fe032e213ce3.jpg)  
(a) CW averaged BCE

![](images/3b0f3f9e4f952a3cb9e91fe4f189a67bebc0dbe0b940a9ed8d7c245f9dd05c2a.jpg)

![](images/26be57b8b2cff4995474e15394c2f98ded19c6813fa411da7b5b9a11a0525f3e.jpg)  
(b) G. averaged BCE

![](images/7ced511c99d29aa4119acce81e4224104a2c445772d47dbef3048281f825b370.jpg)  
Figure S.11: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ ResNet18 trained with BCE on CIFAR100-LT with $\mathrm { I F } { = } 1 0$

![](images/336511c79e02e3dde6c75c0bc7dde00c700c8a44f097506fdb0c12e2f7c02a2d.jpg)

![](images/bf1aca7b3856a68b1a122f6e221f7a41e05b9a6fb2c0c8bf9ea8459df3f1ec69.jpg)  
(a) CW averaged BCE

![](images/67b995614073cd99a95c8b1c995a7bf12db13859213f62205ce45d51c3ddf056.jpg)

![](images/2610fcbb91094ad2c1fadc85245e0e4b3e50660b1e7c2808889984704773ce97.jpg)

![](images/515ac9942e66576b0689288ce153303c8a7d17c66ad98938df4571b61b392e1b.jpg)  
(b) G. averaged BCE

![](images/7049dfe9769cefea21d5db7e64014b9f5242da840b5419df0d4a84a6daf4b21e.jpg)

Figure S.12: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ of ResNet18 trained with BCE on CIFAR100-LT with $\mathrm { I F } { = } 5 0$  
![](images/e75104fa2e9e9060c0fa32d8ae907c3ac4485730887bb0ceaf156e2f7de56d1c.jpg)

![](images/46fdee1b1ff2ab3ef1469aebfd4334f8afa4f777f40a84b9d8432d50649adff7.jpg)

![](images/5a7e54e0d182c3e5805e73f6f4ef8c3374f4dd4ac17a8fb47f1762bbc5363d03.jpg)  
(a) CW averaged BCE

![](images/bbfbbb91936f952cab396f2af592b4f6d7a85c1724228bfd717fe4f1938801ce.jpg)

![](images/36ae23db3421a081653065af03c1b7661c41a60d448529fc9d2c223cd7b8503b.jpg)  
(b) G. averaged BCE

![](images/de9a3a36ad1f1911791ef0c4f5e0c4afe6e46c8b0569ac4ec2435314d581153e.jpg)  
Figure S.13: Similarity matrices for class centers $\{ \bar { \pmb { x } } _ { k } \} _ { k = 1 } ^ { K }$ and classifier vectors $\{ \boldsymbol { w } _ { k } \} _ { k = 1 } ^ { K }$ ResNet18 trained with BCE on CIFAR100-LT with $\mathrm { { I F } = 1 0 0 }$

## S.2 Proofs

We provide the proof of the theorems in our main paper, along with the necessary lemmas.

## S.2.1 Basics

We first formally present the definition of canonical K-simplex ETF in d-dimension space, and two lemmas required by the proofs.

Definition S.1 (K-simplex ETF) A K-simplex ETF is a collection of vectors in $\mathbb { R } ^ { d }$ specified by the columns of matrix

$$
M = \sqrt { \frac { K } { K - 1 } } P \left( \pmb { I } _ { K } - \frac { 1 } { K } \pmb { 1 } _ { K } \pmb { 1 } _ { K } ^ { \top } \right) ,\tag{S.2}
$$

where $\pmb { I } _ { K } \in \mathbb { R } ^ { K \times K }$ is the identity matrix and $\mathbf { 1 } _ { K } \in \mathbb { R } ^ { K }$ is the ones vector, and $\textbf { \textit { P } } \in$ $\mathbb { R } ^ { d \times K } ( d \geq K )$ is a partial-orthogonal matrix such that $P ^ { \top } P = I _ { K }$

Note that the matrix M satisfies:

$$
\boldsymbol { M } ^ { \top } \boldsymbol { M } = \frac { K } { K - 1 } \left( \pmb { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \right) .\tag{S.3}
$$

Lemma S.2 (Young’s Inequality) Let $p , q$ be positive numbers satisfying $\begin{array} { r } { \frac { 1 } { p } + \frac { 1 } { q } = 1 } \end{array}$ . Then for any $a , b \in \mathbb { R }$ , we have

$$
| a b | \leq \frac { | a | ^ { p } } { p } + \frac { | b | ^ { q } } { q } ,\tag{S.4}
$$

where the equality holds if and only if $| a | ^ { p } = | b | ^ { q }$ . The case for $p = q = 2$ is just the AM-GM inequality, $| a b | \leq \frac { 1 } { 2 } ( a ^ { 2 } + b ^ { 2 } )$ , and the equality holds if and only $i f \left| a \right| = \left| b \right|$

Lemma S.3 (Jensen’s Inequality) Let f be a convex function on an interval I. Then

$$
f ( \lambda _ { 1 } x _ { 1 } + \cdot \cdot \cdot + \lambda _ { n } x _ { n } ) \leq \lambda _ { 1 } f ( x _ { 1 } ) + \cdot \cdot \cdot + \lambda _ { n } f ( x _ { n } ) ,\tag{S.5}
$$

where

$$
x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \in I , \quad \lambda _ { 1 } , \lambda _ { 2 } , \cdots , \lambda _ { n } \in [ 0 , 1 ] , \quad \sum _ { i = 1 } ^ { n } \lambda _ { i } = 1 .\tag{S.6}
$$

When f is strictly convex, let

$$
I _ { 1 } = \{ k : \lambda _ { k } \in ( 0 , 1 ] \} , \ a n d \ I _ { 2 } = \{ k : \lambda _ { k } = 0 \} .\tag{S.7}
$$

The equality holds if and only if all $x _ { k } , k \in I _ { 1 }$ are equal. In practice, if the function f likes exp(x), log(x), or $\log ( 1 + \mathrm { e } ^ { x } )$ , Jensen’s Inequality can be written as

$$
\exp \left( \frac { \sum _ { i = 1 } ^ { n } x _ { i } } { n } \right) \leq \frac { \sum _ { i = 1 } ^ { n } \exp ( x _ { i } ) } { n } ,\tag{S.8}
$$

$$
\log \left( \frac { \sum _ { i = 1 } ^ { n } x _ { i } } { n } \right) \geq \frac { \sum _ { i = 1 } ^ { n } \log ( x _ { i } ) } { n } ,\tag{S.9}
$$

$$
\log \left( 1 + \exp \left( \frac { \sum _ { i = 1 } ^ { n } x _ { i } } { n } \right) \right) \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log \left( 1 + \exp ( x _ { i } ) \right) .\tag{S.10}
$$

## S.2.2 Proof of Theorem 1

Theorem 1 consists of two parts, namely, the class-wise averaged CE and BCE losses can lead to neural collapse (NC) on imbalanced datasets. We prove them in turn.

Theorem S.4 (BCE) Suppose that each class in dataset D contains at least one sample, and that the feature dimension is greater than the number of classes, i.e., $n _ { k } = | \mathcal D _ { k } | > 0$ for $\forall k \in [ K ]$ and $d \geq K - 1$ . Then, for given scale factor γ and class number K, any global minimizer of

$$
f _ { \mathrm { b c e } } ( W , b , \mathcal { F } ) \triangleq \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mathrm { b c e } } ( X ^ { ( k ) } )\tag{S.11}
$$

satisfies

$$
\pmb { x } ^ { ( k ) } = \pmb { w } _ { k } , \quad \forall \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] ,\tag{S.12}
$$

$$
\langle { \pmb w } _ { k } , { \pmb w } _ { j } \rangle = - \frac { 1 } { K - 1 } , \quad \forall k \neq j \in [ K ] ,\tag{S.13}
$$

$$
b _ { k } = b _ { j } = b ^ { \star } , \quad \forall k , j \in [ K ] ,\tag{S.14}
$$

where $b ^ { \star }$ is the unique real solution of

$$
\mathrm { e } ^ { \frac { \gamma } { K - 1 } } \mathrm { e } ^ { 2 b } - ( K - 2 ) \mathrm { e } ^ { b } - ( K - 1 ) \mathrm { e } ^ { \gamma } = 0 .\tag{S.15}
$$

Proof

According to Lemma S.5, for any $c _ { 1 } , c _ { 2 } > 0$ and any $\pmb { X } ^ { ( k ) } \in \mathcal { D } _ { k }$ with $\forall k \in [ K ]$ 2

$$
L _ { \mathrm { b c e } } ( \mathbf { \boldsymbol { X } } ^ { ( k ) } ) \geq \frac { 1 } { 1 + c _ { 1 } } \Big ( - \gamma \mathbf { \boldsymbol { w } } _ { k } ^ { \top } \mathbf { \boldsymbol { x } } ^ { ( k ) } + b _ { k } \Big ) + \frac { 1 } { 1 + c _ { 2 } } \sum _ { j = 1 \atop j \neq k } ^ { K } \Big ( \gamma \mathbf { \boldsymbol { w } } _ { j } ^ { \top } \mathbf { \boldsymbol { x } } ^ { ( k ) } - b _ { j } \Big ) + C ,
$$

where

$$
C = \log ( 1 + c _ { 1 } ) - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K - 1 ) \Big ( \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \Big ) ,
$$

and the inequality becomes an equality if and only if

$$
c _ { 1 } = \exp { \Big ( \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { k } \Big ) } , \quad c _ { 2 } = \exp { \Big ( b _ { j } - \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } \Big ) } , \quad \forall j \neq k \in [ K ] .
$$

We take the same $c _ { 1 }$ and $c _ { 2 }$ for all $X ^ { ( k ) }$ with $\forall k \in [ K ]$ , then

$$
\begin{array} { r l } & { f _ { \mathrm { b c e } } ( W , b , \mathcal { F } ) = \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mathrm { b c e } } \big ( \pmb { X } ^ { ( k ) } \big ) } \\ & { \geq \displaystyle \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \left[ \frac { 1 } { 1 + c _ { 1 } } \Big ( - \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } + b _ { k } \Big ) + \frac { 1 } { 1 + c _ { 2 } } \sum _ { j = 1 } ^ { K } \Big ( \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } \Big ) \right] + C } \end{array}\tag{S.16}
$$

$$
\begin{array} { r l }    \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \sum _ { k = 1 } ^ { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \sum _ { k = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { k = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^ { n } \sum _ { l = 1 } ^  \end{array}\tag{17}
$$

where $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } } \end{array}$ x¯<sub>j</sub> and $\begin{array} { r } { \bar { \pmb { x } } _ { j } = \frac { 1 } { n _ { j } } \sum _ { \pmb { x } ^ { ( j ) } \in \mathcal { F } _ { j } } \pmb { x } ^ { ( j ) } , \forall j \in [ K ] } \end{array}$

For the function $\hat { f } _ { \mathrm { b c e } } ( W , b , \mathcal { F } )$ , its global minima are located at its stationary points in terms of $b _ { k }$ for $\forall k \in [ K ]$ , then

$$
0 = \frac { \partial \hat { f } _ { \mathrm { b c e } } } { \partial b _ { k } } = - \frac { 1 } { K } \bigg ( \frac { K - 1 } { 1 + c _ { 2 } } - \frac { 1 } { 1 + c _ { 1 } } \bigg ) .\tag{S.18}
$$

Therefore,

$$
\hat { f } _ { \mathrm { b c e } } ( W , b , \mathcal { F } ) \geq \frac { \gamma } { 1 + c _ { 2 } } \sum _ { k = 1 } ^ { K } w _ { k } ^ { \top } \big ( \tilde { { \pmb x } } - \bar { { \pmb x } } _ { k } \big ) + C .
$$

Furthermore, according to the AM-GM inequality (see Lemma S.2), we have

$$
u ^ { \top } v \geq - \Big ( \frac { c } { 2 } \| u \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c } \| v \| _ { 2 } ^ { 2 } \Big ) , \quad \forall u , v \in \mathbb { R } ^ { d } , \ \forall c > 0 ,
$$

which becomes an equality when $\mathbf { \nabla } \pmb { v } = - c \pmb { u }$ . Then,

$$
\begin{array} { l } { { \displaystyle f _ { \mathrm { b e c } } ( W , b , \mathcal { F } ) \geq \hat { f } _ { \mathrm { b e c } } ( W , b , \mathcal { F } ) \geq \frac { \gamma } { 1 + c _ { 2 } } \sum _ { k = 1 } ^ { K } w _ { k } ^ { \top } \big ( \bar { x } - \bar { x } _ { k } \big ) + C } } \\ { { \displaystyle \geq - \frac { \gamma } { 1 + c _ { 2 } } \left( \frac { c _ { 3 } } { 2 } \sum _ { k = 1 } ^ { K } | w _ { k } | | _ { 2 } ^ { 2 } + \frac { 1 } { 2 c _ { 3 } } \sum _ { k = 1 } ^ { K } | \bar { x } - \bar { x } _ { k } | | _ { 2 } ^ { 2 } \right) + C } } \\ { { \displaystyle = - \frac { \gamma } { 1 + c _ { 2 } } \left[ \frac { c _ { 3 } } { 2 } \sum _ { k = 1 } ^ { K } | w _ { k } | | _ { 2 } ^ { 2 } + \frac { 1 } { 2 c _ { 3 } } \left( \sum _ { k = 1 } ^ { K } | \bar { x } _ { k } | | _ { 2 } ^ { 2 } - K | | \bar { x } | | _ { 2 } ^ { 2 } \right) \right] + C } } \\ { { \displaystyle \geq - \frac { \gamma } { 1 + c _ { 2 } } \left( \frac { c _ { 3 } } { 2 } K + \frac { 1 } { 2 c _ { 3 } } K \right) + \frac { \gamma } { 1 + c _ { 2 } } \frac { 1 } { 2 c _ { 3 } } K | | \bar { x } | | _ { 2 } ^ { 2 } + C } } \\ { { \displaystyle \geq - \frac { \gamma } { 1 + c _ { 2 } } \left( \frac { c _ { 3 } } { 2 } K + \frac { 1 } { 2 c _ { 3 } } K \right) + C , } } \end{array}\tag{S.19}
$$

(S.20)

(S.21)

where inequality (S.19) becomes an equality if and only if

$$
c _ { 3 } { \pmb w } _ { k } = { \bar { \pmb x } } _ { k } - { \tilde { \pmb x } } , \quad \forall k \in [ K ] , \forall c _ { 3 } > 0 ,\tag{S.22}
$$

inequality (S.20) becomes an equality if and only if

$$
\begin{array} { c l } & { 1 = \displaystyle \| \bar { \boldsymbol { \mathbf { x } } } _ { k } \| = \left\| \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \pmb { x } ^ { ( k ) } \right\| \leq \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \| \pmb { x } ^ { ( k ) } \| = 1 } \\ & { \Leftrightarrow \quad \pmb { x } ^ { ( k ) } = \bar { \boldsymbol { x } } _ { k } , \forall \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \quad \forall k \in [ K ] , } \end{array}\tag{S.23}
$$

and, the inequality (S.21) becomes an equality if and only if

$$
\tilde { \mathbf { x } } = \mathbf { 0 } ,\tag{S.24}
$$

implying

$$
\begin{array} { r l } & { c _ { 3 } { \mathbf { \overline { { w } } } } _ { k } = \bar { \mathbf { x } } _ { k } \quad \Longrightarrow \quad \| c _ { 3 } { \mathbf { \overline { { w } } } } _ { k } \| _ { 2 } ^ { 2 } = \| \bar { \mathbf { x } } _ { k } \| _ { 2 } ^ { 2 } \quad \stackrel { S \cdot 2 3 } { \longrightarrow } \quad c _ { 3 } ^ { 2 } = 1 } \\ & { \quad \implies c _ { 3 } = 1 \mathrm { a n d } \mathbf { \overline { { w } } } _ { k } = { \mathbf { x } } ^ { ( k ) } , \quad \forall { \mathbf { x } } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] . } \end{array}
$$

Combining Eqs. (S.21) and (S.25), we get

$$
f _ { \mathrm { b c e } } ( W , \pmb { b } , \mathcal { F } ) \geq - \frac { \gamma } { 1 + c _ { 2 } } \bigg ( \frac { c _ { 3 } } { 2 } K + \frac { 1 } { 2 c _ { 3 } } K \bigg ) + C = - \frac { \gamma K } { 1 + c _ { 2 } } + C ,\tag{S.25}
$$

which becomes an equality if and only if $E q s$ . (S.39), (S.40), (S.22), (S.23), and (S.24) hold.

When Eqs. (S.22), (S.23), and (S.24) hold, one can deduce that

$$
c _ { 1 } = \exp { \Big ( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \Big ) } = \exp { \Big ( \gamma \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } \Big ) } , \quad \forall k \in [ K ] ,\tag{S.26}
$$

$$
c _ { 2 } = \exp { \Big ( b _ { j } - \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } \Big ) } = \exp { \Big ( b _ { j } - \gamma { \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } \Big ) } , \quad \forall j \neq k \in [ K ] .\tag{S.27}
$$

Since the $c _ { 1 } , c _ { 2 } > 0$ are the same for diferent $j \neq k \in [ K ]$ , we have

$$
\gamma \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } = \gamma \| \pmb { w } _ { j } \| _ { 2 } ^ { 2 } - b _ { j } , \quad \forall k , j \in [ K ] ,\tag{S.28}
$$

$$
\gamma w _ { k } ^ { \top } w _ { j } - b _ { j } = \gamma w _ { k } ^ { \top } w _ { \ell } - b _ { \ell } , \quad \forall j \neq \ell \in [ K ] , \ \forall k \in [ K ] .\tag{S.29}
$$

In the normalized feature space, $E q .$ . (S.28) implies

$$
\begin{array} { r } { \gamma \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } = \gamma \| \pmb { w } _ { j } \| _ { 2 } ^ { 2 } - b _ { j } \quad \Rightarrow \quad \gamma - b _ { k } = \gamma - b _ { j } \quad \Leftrightarrow \quad b _ { k } = b _ { j } , \quad \forall k , j \in [ K ] . } \end{array}\tag{S.30}
$$

Therefore, at the minimum points of $f _ { \mathrm { b c e } }$ , one can write ${ \pmb b } = b ^ { \star } { \bf 1 }$ for some $b ^ { \star } \in \mathbb { R }$ . Then, from Eq. (S.29),

$$
\begin{array} { r } { \gamma \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = \gamma \pmb { w } _ { k } ^ { \top } \pmb { w } _ { \ell } \quad \Rightarrow \quad \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = \pmb { w } _ { k } ^ { \top } \pmb { w } _ { \ell } , \quad \forall j \neq \ell \in [ K ] , \quad \forall k \in [ K ] , } \end{array}\tag{S.31}
$$

and from $E q s$ . (S.22), (S.23), and (S.24) hold, one deduces that $\textstyle \sum _ { k = 1 } ^ { K } { \pmb { w } _ { k } = \mathbf { 0 } }$ implying

$$
\begin{array} { r l } & { 1 = \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } = - \displaystyle \sum _ { j = 1 \atop j \neq k } ^ { K } \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \left( K - 1 \right) \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } } \\ { \Rightarrow } & { \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \displaystyle \frac { 1 } { K - 1 } , \quad \forall j \neq k \in [ K ] . } \end{array}\tag{S.32}
$$

Plugging $E q s$ . (S.30), (S.32) into Eqs. (S.26), (S.27),

$$
c _ { 1 } = \exp ( \gamma - b ^ { \star } ) \quad \mathrm { a n d } \quad c _ { 2 } = \exp \bigg ( b ^ { \star } + \frac { \gamma } { K - 1 } \bigg ) .\tag{S.33}
$$

When $f _ { \mathrm { b c e } }$ defined in $E q$ . (S.11) achieves its minimum value, it theoretically satisfies

$$
\begin{array} { c } { \displaystyle 0 = \frac { \partial f _ { \mathrm { b c e } } } { \partial b _ { k } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \bigg ( 1 - \frac { 1 } { 1 + \exp \big ( - \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } + b _ { k } \big ) } \bigg ) } \\ { \displaystyle - \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \frac { 1 } { n _ { j } } \sum _ { \pmb { x } ^ { ( j ) } \in \mathcal { F } _ { j } } \sum _ { k = 1 } ^ { K } \bigg ( 1 - \frac { 1 } { 1 + \exp \big ( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( j ) } - b _ { k } \big ) } \bigg ) , \quad \forall k \in [ K ] , } \end{array}\tag{S.34}
$$

thus, combining Eqs. (S.23) and (S.32),

$$
\begin{array} { r l } & { 0 = 1 - \displaystyle \frac { 1 } { 1 + \exp \big ( - \gamma + b _ { k } \big ) } - ( K - 1 ) \bigg ( 1 - \frac { 1 } { 1 + \exp \big ( - \gamma \frac { 1 } { K - 1 } - b _ { k } \big ) } \bigg ) } \\ { \Leftrightarrow } & { ~ \frac { 1 } { 1 + \exp \big ( \gamma - b _ { k } \big ) } = \frac { K - 1 } { 1 + \exp \big ( \frac { \gamma } { K - 1 } + b _ { k } \big ) } } \\ { \Leftrightarrow } & { ~ \mathrm { e } ^ { \frac { \gamma } { K - 1 } } \mathrm { e } ^ { 2 b _ { k } } - ( K - 2 ) \mathrm { e } ^ { b _ { k } } - ( K - 1 ) \mathrm { e } ^ { \gamma } = 0 , \quad \forall k \in [ K ] , } \end{array}\tag{S.35}
$$

which satisfies Eq. (S.18).

In total,

$$
\begin{array} { l } { { f _ { \mathrm { b c e } } ( W , { \pmb b } , \mathcal { F } ) \geq \displaystyle - \frac { \gamma K } { 1 + c _ { 2 } } - \frac { \gamma K } { 1 + c _ { 2 } } + \log ( 1 + c _ { 1 } ) + C } } \\ { { \displaystyle ~ - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K - 1 ) \bigg ( \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \bigg ) } } \\ { { \displaystyle ~ s . 3 3 \underline { { s } } . 1 8 ~ \log \big ( 1 + \mathrm { e } ^ { b ^ { \star } - \gamma } \big ) + ( K - 1 ) \log \Big ( 1 + \mathrm { e } ^ { - b ^ { \star } - \frac { \gamma } { K - 1 } } \Big ) , } } \end{array}\tag{S.36}
$$

and $f _ { \mathrm { b c e } }$ achieves this constant lower boundary if and only if Eqs. (S.39), (S.40), (S.22), (S.23), (S.24), and (S.35) hold. It is easy deduce Eqs. (S.12), (S.13), and (S.14) from these conditions, and vice versa, which completes the proof. 7

Lemma S.5 For any $X ^ { ( k ) } \in \mathcal { D } _ { k } , \forall k \in [ K ]$ , with any $c _ { 1 } , c _ { 2 } > 0$ and $\gamma > 0$ , the BCE loss $L _ { \mathrm { b c e } } ( \mathbf { \mathbf { } } X ^ { ( k ) } )$ defined in Eq. (5) is lower bounded,

$$
L _ { \mathrm { b c e } } ( \mathbf { \mathbf { \mathbf { X } } } ^ { ( k ) } ) \geq \frac { 1 } { 1 + c _ { 1 } } \Big ( - \gamma \mathbf { \mathbf { \mathbf { w } } } _ { k } ^ { \top } \mathbf { \mathbf { \mathbf { x } } } ^ { ( k ) } + b _ { k } \Big ) + \frac { 1 } { 1 + c _ { 2 } } \sum _ { j = 1 \atop j \neq k } ^ { K } \Big ( \gamma \mathbf { \mathbf { \mathbf { w } } } _ { j } ^ { \top } \mathbf { \mathbf { \mathbf { x } } } ^ { ( k ) } - b _ { j } \Big ) + C ,\tag{S.37}
$$

where

$$
C = \log ( 1 + c _ { 1 } ) - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K - 1 ) \Big ( \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \Big ) .\tag{S.38}
$$

The inequality becomes an equality if and only if

$$
c _ { 1 } = \exp { \left( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \right) } ,\tag{S.39}
$$

$$
c _ { 2 } = \exp { \Big ( } b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } \Big ) , \quad \forall j \neq k \in [ K ] .\tag{S.40}
$$

Proof By the concavities of $\log ( x )$ , and according to Lemma S.3, the BCE loss $L _ { \mathrm { b c e } } ( \mathbf { \mathbf { } } X ^ { ( k ) } )$ for any $\pmb { X } ^ { ( k ) } \in \mathcal { D }$ can be lower bounded by

$$
\begin{array} { r l } { \sum _ { i = 1 } ^ { n }  \mathbf { X } ^ { i + 1 } - \mathbf { X } ^ { i }  _ { 1 } - \frac { 1 } { n + 2 } \alpha \alpha \Big ( \frac { 1 } { n + 3 } ( \sum _ { i = 1 } ^ { n } - \frac { 1 } { n + 3 } ) \ln { \frac { 1 } { 2 } } + \frac { 1 } { 3 } \ln { \frac { 1 } { 2 } } + \frac { 1 } { 3 } \ln { \frac { 1 } { 2 } } + \frac { 1 } { 3 } \ln { \frac { 1 } { 2 } } + \alpha \ln { \frac { 1 } { 2 } } \alpha ^ { i + 1 } - \beta \ln { \frac { 1 } { 2 } } \alpha ^ { i }  } \\ { + } & { \frac { 1 } { n + 3 } ( \frac { 1 } { n + 3 } - \frac { 1 } { n + 3 } ) \ln { \frac { 1 } { 2 } } + \frac { 1 } { 3 } \ln { \frac { 1 } { 2 } } ( \sum _ { i = 1 } ^ { n } - 3 \ln { \frac { 1 } { 2 } } \alpha ^ { i } ) } \\ & { + \frac { 1 } { n + 3 } \ln { \frac { 1 } { 2 } } \exp ( \frac { \alpha } { 1 } \frac { 1 } { n + 3 } + \frac { 1 } { 1 + 3 } \ln { \frac { 1 } { 2 } } + \frac { 1 } { 3 } \ln { \frac { 1 } { 2 } } \alpha ^ { i } ) \exp ( \frac { \alpha - 3 } { 1 + 3 } ) \Big \rangle } \\ { \geq } & { \frac { 1 } { n + 3 } \exp \Bigg ( - \frac { 1 } { n + 3 } \ln { \frac { 1 } { 2 } } ( \frac { 1 } { n + 3 } ) - 1 ) \ln { \frac { 1 } { 2 } } + \frac { 1 } { n + 3 } \exp ( \frac { \alpha - 3 } { 1 + 3 } \ln { \frac { 1 } { 2 } } \alpha ^ { i } - \beta \ln { \frac { 1 } { 2 } } ) } \\ { \geq } &  \frac { 1 } { n + 3 } \exp \Bigg ( - \frac { 1 } { n + 3 }  \end{array}\tag{S.41}
$$

The inequality becomes an equality if and only if

$$
\begin{array} { r l } & { \cfrac { 1 + c _ { 1 } } { c _ { 1 } } = ( 1 + c _ { 1 } ) \exp \left( - \gamma w _ { k } ^ { \top } { \boldsymbol x } ^ { ( k ) } + b _ { k } \right) \mathrm { ~ o r ~ } c _ { 1 } = 0 \mathrm { ~ o r ~ } c _ { 1 } = + \infty , \mathrm { ~ a n d ~ } } \\ & { \cfrac { 1 + c _ { 2 } } { c _ { 2 } } = ( 1 + c _ { 2 } ) \exp \left( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \right) \mathrm { ~ o r ~ } c _ { 2 } = 0 \mathrm { ~ o r ~ } c _ { 2 } = + \infty , \forall j \not = k \in [ K ] . } \end{array}
$$

These equalities are trivial when $c _ { 1 } = 0$ or $c _ { 1 } = + \infty$ and $c _ { 2 } = 0$ or $c _ { 2 } = + \infty$ . Therefore, if and only if

$$
\begin{array} { r l } & { c _ { 1 } = \exp \left( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \right) , \quad \mathrm { a n d } } \\ & { c _ { 2 } = \exp \left( b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } \right) , \quad \forall j \neq k \in [ K ] , } \end{array}
$$

Eq. (S.37) becomes an equality, which completes the proof.

Theorem S.6 (CE) Suppose that each class in dataset D contains at least one sample, and that the feature dimension is greater than the number of classes, i.e., $n _ { k } = | \mathcal D _ { k } | > 0$ for $\forall k \in [ K ]$ and $d \geq K - 1$ . Then, at any minimum point of loss

$$
f _ { \mathrm { c e } } ( W , b , \mathcal { F } ) \triangleq \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } } L _ { \mathrm { c e } } ( X ^ { ( k ) } )\tag{S.42}
$$

satisfies

$$
\pmb { x } ^ { ( k ) } = \pmb { w } _ { k } , \quad \forall \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] ,\tag{S.43}
$$

$$
\langle { \pmb w } _ { k } , { \pmb w } _ { j } \rangle = - \frac { 1 } { K - 1 } , \quad \forall k \neq j \in [ K ] ,\tag{S.44}
$$

$$
b _ { k } = b _ { j } = b ^ { \star } , \quad \forall k , j \in [ K ] ,\tag{S.45}
$$

while $b ^ { \star } \in \mathbb { R }$ could be arbitrary number.

Proof According to Lemma S.7, for any $c _ { 1 } > 0$ and any $\pmb { X } ^ { ( k ) } \in \mathcal { D } _ { k }$ with $\forall k \in [ K ]$

$$
L _ { \mathrm { c e } } ( \mathbf { \boldsymbol { X } } ^ { ( k ) } ) \ge \frac { \sum _ { j = 1 } ^ { K } \left( \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } \right) - K ( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } ) } { \left( 1 + c _ { 1 } \right) ( K - 1 ) } + C ,
$$

where

$$
C = \frac { c _ { 1 } } { 1 + c _ { 1 } } \log \left( \frac { 1 + c _ { 1 } } { c _ { 1 } } \right) + \frac { 1 } { 1 + c _ { 1 } } \log \left[ \bigl ( 1 + c _ { 1 } \bigr ) ( K - 1 ) \right] ,
$$

and inequality becomes an equality if and only if

$$
\begin{array} { r l } & { c _ { 1 } = \left[ \left( K - 1 \right) \exp \left( \frac { \sum _ { j = 1 } ^ { K } \left( \gamma w _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } \right) - K \left( \gamma w _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \right) } { K - 1 } \right) \right] ^ { - 1 } } \\ & { \gamma w _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } = \gamma w _ { \ell } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { \ell } , \quad \forall j , \ell \neq k \in [ K ] . } \end{array}
$$

and

We take the same $c _ { 1 }$ for all $\pmb { X } ^ { ( k ) } \in \mathcal { D }$ , then one can deduce the following lower bound for the function $f _ { \mathrm { c e } }$

$$
\begin{array} { r l } & { \quad ( 1 + c ) ( N - 1 ) ( \int _ { \mathbb { R } } ( { \boldsymbol W } , { \boldsymbol b } , { \boldsymbol \mathcal F } ) - { \boldsymbol C } ) } \\ & { = ( 1 + c _ { 1 } ) ( K - 1 ) \Big [ \displaystyle \frac { K } { K - 1 } \frac { 1 } { \bar { n } _ { k } } \sum _ { n _ { k } } L _ { w _ { k } } ( { \boldsymbol X } ^ { ( k ) } ) - C \Big ] } \\ & { \ge \displaystyle \frac { 1 } { K } \sum _ { \underline { { K } } = 1 } ^ { K } \frac { 1 } { \bar { n } _ { k } } \sum _ { \underline { { W } } ^ { ( k ) } \in \mathcal { K } _ { \mathbb { F } } } \left[ \displaystyle \sum _ { j = 1 } ^ { K } \left( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \right) - K \left( \gamma w _ { k } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { k } \right) \right] } \\ & { = \displaystyle \frac { 1 } { K } \sum _ { \underline { { K } } = 1 } ^ { K } \frac { 1 } { \bar { n } _ { k } } \sum _ { 1 = \underline { { K } } } ^ { K } \sum _ { j = 1 } ^ { K } \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - \displaystyle \sum _ { k = 1 } ^ { K } \frac { 1 } { \bar { n } _ { k } } \sum _ { \underline { { K } } ^ { ( k ) } \in \mathcal { K } _ { \mathbb { F } } } \gamma \displaystyle \mathsf { w } _ { k } ^ { \top } { \boldsymbol x } ^ { ( k ) } - \underbrace { \frac { K } { K } \sum _ { \underline { { K } } = 1 , \underline { { K } } } ^ { K } b _ { j } + \sum _ { \underline { { K } } = 1 } ^ { K } b _ { k } } _ { ( 0 ) } } \\ &  = \displaystyle \frac { 1 } { K } \sum _ { k = 1 + 2 } ^ { K } \gamma w _ { j } ^ { \top } \bar { x } _ { k } - \displaystyle \sum _ { k = 1 } ^ { K } \gamma w _ { k } ^ { \top }  \end{array}\tag{.46}
$$

where $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \bar { \pmb { x } } _ { j } } \end{array}$ and $\begin{array} { r } { \bar { \pmb { x } } _ { j } = \frac { 1 } { n _ { i } } \sum _ { \pmb { x } ^ { ( j ) } \in \mathcal { F } _ { j } } \pmb { x } ^ { ( j ) } , \forall j \in [ K ] } \end{array}$

According to the AM-GM inequality in Lemma S.2,

$$
\pmb { u } ^ { \top } \pmb { v } \leq \frac { c } { 2 } \| \pmb { u } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c } \| \pmb { v } \| _ { 2 } ^ { 2 } , \ \forall \pmb { u } , \pmb { v } \in \mathbb { R } ^ { d } , \ \forall c > 0 ,
$$

which becomes an equality when ${ \pmb v } = c { \pmb u }$ . Then, based on $E q .$ (S.46),

$$
\begin{array} { r l r } {  { ( 1 + c _ { 1 } ) ( K - 1 ) \big [ f _ { \mathrm { c e } } ( W , b , \mathcal { F } ) - C \big ] \geq \gamma \sum _ { k = 1 } ^ { K } w _ { k } \big ( \tilde { \mathbf { x } } - \bar { \mathbf { x } } _ { k } \big ) } } \\ & { } & \\ & { \geq - \gamma \frac { c _ { 2 } } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } \| { \mathbf { w } } _ { k } \| _ { 2 } ^ { 2 } - \frac { \gamma } { 2 c _ { 2 } } \displaystyle \sum _ { k = 1 } ^ { K } \| \tilde { \mathbf { x } } - \bar { \mathbf { x } } _ { k } \| _ { 2 } ^ { 2 } } \\ & { } & \\ & { = - K \gamma \frac { c _ { 2 } } { 2 } - \frac { \gamma } { 2 c _ { 2 } } \bigg ( \displaystyle \sum _ { k = 1 } ^ { K } \| \bar { \mathbf { x } } _ { k } \| ^ { 2 } - K \| \tilde { \mathbf { x } } \| ^ { 2 } \bigg ) , } \end{array}
$$

where the second inequality becomes an equality if and only if

$$
c _ { 2 } { \pmb w } _ { k } = - { \tilde { \pmb x } } + { \bar { \pmb x } } _ { k } , \quad \forall k \in [ K ] , \forall c _ { 2 } > 0 .\tag{S.47}
$$

Therefore,

$$
\begin{array} { r l } { f _ { \infty } ( W , b , \mathcal { F } ) \geq - \frac { c _ { 2 } \gamma K } { 2 ( 1 + c _ { 1 } ) ( K - 1 ) } - \frac { \gamma } { 2 c _ { 2 } ( 1 + c _ { 1 } ) ( K - 1 ) } \bigg ( \underset { K = 1 } { \overset { \kappa } { \underset { k = 1 } { K } } } \| x _ { k } \| ^ { 2 } - K \| \bar { x } \| ^ { 2 } \bigg ) + C } & { } \\ { \geq - \frac { c _ { 2 } \gamma K } { 2 ( 1 + c _ { 1 } ) ( K - 1 ) } - \frac { \gamma } { 2 c _ { 2 } ( 1 + c _ { 1 } ) ( K - 1 ) } \underset { K = 1 } { \overset { \kappa } { \underset { k = 1 } { K } } } \| x _ { k } \| ^ { 2 } + C } & { } \\ { = - \frac { c _ { 2 } \gamma K } { 2 ( 1 + c _ { 1 } ) ( K - 1 ) } - \frac { \gamma } { 2 c _ { 2 } ( 1 + c _ { 1 } ) ( K - 1 ) } \underset { k = 1 } { \overset { \kappa } { \underset { k = 1 } { K } } } \left\| \frac { 1 } { n _ { k } } \underset { z ^ { \infty } \in \mathcal { F } _ { \mathcal { F } _ { \mathcal { F } _ { \mathcal { F } _ { \mathcal { F } } } } } } x ^ { ( k ) } \right\| ^ { 2 } + C } & { } \\ { \geq - \frac { c _ { 2 } \gamma K } { 2 ( 1 + c _ { 1 } ) ( K - 1 ) } - \frac { \gamma } { 2 c _ { 2 } ( 1 + c _ { 1 } ) ( K - 1 ) } \underset { k = 1 } { \overset { \kappa } { \sum } } \frac { 1 } { n _ { k } } \underset { \alpha ^ { ( 0 ) } \leq \mathcal { F } _ { \mathcal { F } _ { \mathcal { F } } } } \| \alpha ^ { ( k ) } \| ^ { 2 } + C } & { } \\  = - \frac { c _ { 2 } \gamma K } { 2 ( 1 + c _ { 1 } ) ( K - 1 ) } - \frac { K \gamma } { 2 c _ { 2 } ( 1 + c _ { 1 } ) ( K - 1 ) } + C \end{array}\tag{48}
$$

where the second inequality becomes an equality if and only if

$$
\tilde { \mathbf { x } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \bar { \mathbf { x } } _ { j } = \mathbf { 0 } ,\tag{S.49}
$$

and the third inequality becomes an equality if and only if

$$
\begin{array} { r } { \pmb { x } ^ { ( k ) } = \pmb { x } _ { * } ^ { ( k ) } , \quad \forall \pmb { x } ^ { ( k ) } , \pmb { x } _ { * } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] . } \end{array}\tag{S.50}
$$

From Eqs. (S.47), (S.49), and (S.50), we get

$$
\begin{array} { r l r l r l } & { c _ { 2 } { \boldsymbol w } _ { k } = \bar { \boldsymbol x } _ { k } \ } & { \Rightarrow } & { \| c _ { 2 } { \boldsymbol w } _ { k } \| _ { 2 } ^ { 2 } = \| \bar { \boldsymbol x } _ { k } \| _ { 2 } ^ { 2 } } & { \Rightarrow } & { ( c _ { 2 } ) ^ { 2 } = 1 } & { \ \Rightarrow } & { c _ { 2 } = 1 , \ \mathrm { a n d } } \\ & { { \boldsymbol w } _ { k } = { \boldsymbol x } ^ { ( k ) } , \ \forall { \boldsymbol x } ^ { ( k ) } \in \mathcal { F } _ { k } , \ \forall k \in [ K ] . } \end{array}
$$

Plugging Eq. (S.51) into (S.48), we have

$$
f _ { \mathrm { c e } } ( W , b , \mathcal { F } ) \geq - \frac { \gamma K } { 2 ( 1 + c _ { 1 } ) ( K - 1 ) } - \frac { K \gamma } { 2 ( 1 + c _ { 1 } ) ( K - 1 ) } + C = - \frac { \gamma K } { ( 1 + c _ { 1 } ) ( K - 1 ) } + C .
$$

Furthermore, $E q s$ . (S.47), (S.49), and (S.50) imply

$$
\sum _ { k = 1 } ^ { K } w _ { k } = \mathbf { 0 } .\tag{S.51}
$$

Then,

$$
\sum _ { j = 1 } ^ { K } \left( \gamma w _ { j } ^ { \top } \mathbf { x } ^ { ( k ) } - b _ { j } \right) = \gamma \bigg ( \sum _ { j = 1 } ^ { K } w _ { j } ^ { \top } \bigg ) \mathbf { x } ^ { ( k ) } - \sum _ { j = 1 } ^ { K } b _ { j } = 0 - \sum _ { j = 1 } ^ { K } b _ { j } = - K \bar { b } ,\tag{S.52}
$$

where $\begin{array} { r } { \bar { b } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } b _ { j } } \end{array}$ , and

$$
K \big ( \gamma w _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \big ) = K \big ( \gamma \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } \big ) = K \gamma - K b _ { k } .\tag{S.53}
$$

Combining Eqs. (S.59), (S.60), (S.52), and (S.53), we have

$$
\begin{array} { l } { \displaystyle { c _ { 1 } = \left[ \left( K - 1 \right) \exp \left( \frac { \sum _ { j = 1 } ^ { K } \left( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \right) - K \left( \gamma w _ { k } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { k } \right) } { K - 1 } \right) \right] ^ { - 1 } } } \\ { = \left[ \left( K - 1 \right) \exp \left( \frac { - K \bar { b } - \left( K \gamma - K b _ { k } \right) } { K - 1 } \right) \right] ^ { - 1 } = \left[ \left( K - 1 \right) \exp \left( \frac { K } { K - 1 } \big ( b _ { k } - \bar { b } - \gamma \big ) \right) \right] ^ { - 1 } . } \end{array}\tag{S.54}
$$

Since the $c _ { 1 } > 0$ can be arbitrary number, we choose the same $c _ { 1 }$ for any $k , j \in [ K ]$ Then, we have

$$
b _ { k } - \bar { b } - \gamma = b _ { j } - \bar { b } - \gamma \quad \Rightarrow \quad b _ { k } = b _ { j } , \forall j \neq k \in [ K ] .\tag{S.55}
$$

Therefore, we can write ${ \pmb b } = b ^ { \star } { \bf 1 } _ { K }$ . Plugging the results in $E q$ . (S.55) into (S.54), we have

$$
c _ { 1 } = \bigg [ ( K - 1 ) \exp \bigg ( \frac { K } { K - 1 } \big ( b _ { k } - \bar { b } - \gamma \big ) \bigg ) \bigg ] ^ { - 1 } = \bigg [ ( K - 1 ) \exp \bigg ( - \frac { \gamma K } { K - 1 } \bigg ) \bigg ] ^ { - 1 } .
$$

Moreover, plugging Eqs. (S.51) and (S.55) into $E q .$ (S.53),

$$
\gamma w _ { j } ^ { \top } x ^ { ( k ) } - b _ { j } = \gamma w _ { \ell } ^ { \top } x ^ { ( k ) } - b _ { \ell } \quad \Rightarrow \quad w _ { j } ^ { \top } x ^ { ( k ) } = w _ { \ell } ^ { \top } x ^ { ( k ) } \quad \Rightarrow \quad w _ { j } ^ { \top } w _ { k } = w _ { \ell } ^ { \top } w _ { k } .
$$

Combining with $E q$ . (S.51),

$$
1 = \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } = - \sum _ { j = 1 } ^ { K } \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \left( K - 1 \right) \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } \Rightarrow \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \frac { 1 } { K - 1 } , \forall j \neq k \in [ K ] .\tag{S.56}
$$

In total,

$$
f _ { \mathrm { c e } } ( W , \pmb { b } , \mathcal { F } ) \geq - \frac { \gamma K } { ( 1 + c _ { 1 } ) ( K - 1 } + C = \log \Big ( 1 + ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma K } { K - 1 } } \Big ) ,
$$

where this constant lower boundary is achieved if and only if $E q s .$ (S.59), (S.60), (S.47), (S.49), and (S.50) hold. One can deduce Eqs. (S.43), (S.44), and (S.45) from these conditions, and vice versa, which completes the proof. 7

Lemma S.7 For any $\pmb { X } ^ { ( k ) } \in \mathcal { D }$ with any $c _ { 1 } > 0$ and $\gamma > 0$ , the CE loss $L _ { \mathrm { c e } } ( \mathbf { X } ^ { ( k ) } )$ defined in Eq. (2) is lower bounded $b y$

$$
L _ { \mathrm { c e } } ( \mathbf { \boldsymbol { X } } ^ { ( k ) } ) \ge \frac { \sum _ { j = 1 } ^ { K } \left( \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } \right) - K ( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } ) } { \left( 1 + c _ { 1 } \right) ( K - 1 ) } + C ,\tag{S.57}
$$

where

$$
C = \frac { c _ { 1 } } { 1 + c _ { 1 } } \log \left( \frac { 1 + c _ { 1 } } { c _ { 1 } } \right) + \frac { 1 } { 1 + c _ { 1 } } \log \left[ \bigl ( 1 + c _ { 1 } \bigr ) ( K - 1 ) \right] .\tag{S.58}
$$

The inequality becomes an equality if and only if

$$
c _ { 1 } = \Bigg [ ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } \left( \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } \right) - K \left( \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { k } \right) } { K - 1 } \right) \Bigg ] ^ { - 1 } ,\tag{S.59}
$$

and

$$
\gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } = \gamma { \pmb w } _ { \ell } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { \ell } , \quad \forall j , \ell \neq k \in [ K ] .\tag{S.60}
$$

Proof According to Jensen’s inequality in Lemma S.3, the CE loss $L _ { \mathrm { c e } } ( \mathbf { X } ^ { ( k ) } )$ can be lower bounded,

$$
\begin{array} { r l } { \displaystyle L _ { \infty } ( \mathbf { X } ^ { ( k ) } ) = \log \left( 1 + \sum _ { \frac { j \in \mathbb { N } } { j = 1 } } ^ { K } \frac { \exp \big ( \gamma w _ { k } ^ { \top } x ^ { ( k ) } - b _ { j } \big ) } { \exp \big ( \gamma w _ { k } ^ { \top } x ^ { ( k ) } - b _ { k } \big ) } \right) } & { } \\ { = \log \left( 1 + ( K - 1 ) \sum _ { \frac { j \in \mathbb { N } } { j = 1 } } ^ { K } \frac { \exp \big ( \gamma w _ { k } ^ { \top } x ^ { ( k ) } - b _ { j } \big ) } { ( K - 1 ) \exp \big ( \gamma w _ { k } ^ { \top } x ^ { ( k ) } - b _ { k } \big ) } \right) } & { } \\ { \geq \log \left( 1 + ( K - 1 ) \exp \left( \sum _ { \frac { j \in \mathbb { N } } { j = 1 } } ^ { K } \frac { \gamma w _ { j } ^ { \top } x ^ { ( k ) } - b _ { j } - \gamma w _ { k } ^ { \top } x ^ { ( k ) } + b _ { k } } { K - 1 } \right) \right) } & { } \\ { = \log \left( 1 + ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } x ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } x ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right) , } & { } \end{array}\tag{S.61}
$$

which achieves the equality if and only if $\gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } = \gamma \pmb { w } _ { \ell } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { \ell }$ for all $j , \ell \neq k \in [ K ]$ Second, by the concavity of the function log $( 1 + \mathrm { e } ^ { x } )$ , for any $c _ { 1 } > 0$ , we get

$$
\begin{array} { r l } & { L _ { m , \mu } ( X ^ { ( k ) } ) } \\ & { \geq \ln ( 1 + ( K - 1 ) \exp ( \frac { \sum _ { j = 1 } ^ { K } ( \sqrt { w } \pi \sum _ { j = 1 } ^ { T } x ^ { ( k ) } - b _ { j } ) - \bar { K } ( \gamma w \pi ) \frac { T } { x } x ^ { ( k ) } - b _ { j } ) } { K - 1 } ) } \\ & { = \ln ( \begin{array} { l l l } { \alpha } & { 1 + \alpha _ { 1 } } & \\ { 1 + \alpha _ { 1 } } & { \alpha _ { 1 } } & { \alpha _ { 1 } } \end{array} ) \exp ( \sum _ { j = 1 } ^ { K } \sqrt { \alpha ( w ) } \sum _ { \sigma = 1 } ^ { K ( K - 1 ) } b _ { j } ^ { ( K ) } - b _ { j } ) - \bar { K } ( \gamma w \pi ) \frac { \bar { K } ( \bar { K } ) - b _ { j } } { K } ) \Bigg ) } \\ & { \leq \frac { C _ { 1 } } { 1 + \alpha _ { 1 } } \ln ( \frac { 1 + \alpha _ { 1 } } { \alpha _ { 1 } } ) } \\ & { \qquad - \frac { C _ { 1 } } { 1 + \alpha _ { 1 } } \ln ( \frac { 1 + \alpha _ { 1 } } { \alpha _ { 1 } } ) } \\ & { \qquad - \frac { C _ { 1 } } { 1 + \alpha _ { 1 } } \log ( \frac { 1 + \alpha _ { 1 } } { \alpha _ { 1 } } ) ( 1 + \alpha _ { 1 } ) \exp ( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w \frac { T } { 2 } x ^ { ( k ) } - b _ { j } ) } { K - 1 } - \bar { K } ( \gamma w \pi ) \frac { T } { x } x ^ { ( k ) } - b _ { j } ) ) } \\ & { = \quad 1 \ \sum _ { j = 1 } ^ { K } \sqrt { \alpha ( w ) } x ^ { ( k ) } - \bar { K } ( \gamma w \pi ) x ^ { ( k ) } - b _ { j } \cdot } \\ &  \qquad \quad - \frac  C _  1 \end{array}
$$

The last inequality achieves its equality if and only if

$$
\frac { 1 + c _ { 1 } } { c _ { 1 } } = ( 1 + c _ { 1 } ) ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right)
$$

Actually, when $c _ { 1 } = 0$ or $c _ { 1 } = + \infty$ , the equality is trivial. Therefore, we have

$$
c _ { 1 } = \Bigg [ ( K - 1 ) \exp \Bigg ( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { k } ) } { K - 1 } \Bigg ) \Bigg ] ^ { - 1 } ,
$$

as desired.

## S.2.3 Theorems in Open-set Scenario

We here present the proof Theorems 2 and 3.

Theorem S.8 (BCE) Suppose that the first $K _ { 1 }$ classes in dataset $\begin{array} { r } { \mathcal { D } = \bigcup _ { k = 1 } ^ { K } \mathcal { D } _ { k } } \end{array}$ contains at least one sample, $1 < K _ { 1 } < K$ , while the other classes are empty, and that the feature dimension is greater than the number of classes, i $\therefore e . , n _ { k } = | { \mathcal { D } } _ { k } | > 0 f o r \forall k \in [ K _ { 1 } ] , n _ { k } =$ $| \mathcal { D } _ { k } | = 0$ for $K _ { 1 } < k \le K$ , and $d \geq K _ { 1 }$ . Denote $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { k } } \end{array}$ with $\begin{array} { r } { \pmb { \bar { x } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { \pmb { x } \in \mathcal { F } _ { k } } \pmb { x } } \end{array}$ On the dataset D, define BCE loss

$$
f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } ) \triangleq \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \frac { 1 } { n _ { k } } \sum _ { \boldsymbol { X } ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mathrm { b c e } } ( \boldsymbol { X } ^ { ( k ) } ) ,\tag{S.62}
$$

where

$$
L _ { \mathrm { b c e } } ( \pmb X ^ { ( k ) } ) = \log \big ( 1 + \mathrm e ^ { - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } \big ) + \sum _ { j = 1 , j \neq k } ^ { K } \log \big ( 1 + \mathrm e ^ { \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } } \big ) ,
$$

with $\pmb { x } ^ { ( k ) } = \mathcal { M } ( \pmb { X } ^ { ( k ) } ) ~ f o r ~ \forall \pmb { X } ^ { ( k ) } \in \mathcal { D } _ { k } \subset \mathcal { D }$

Then, the BCE loss $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ satisfies

$$
\begin{array} { r l } & { f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } ) \geq \log \big ( 1 + \mathrm { e } ^ { b ^ { \star } - \gamma \sqrt { 1 - \| \bar { x } \| ^ { 2 } } } \big ) + ( K _ { 1 } - 1 ) \log \big ( 1 + \mathrm { e } ^ { - \frac { \gamma } { K _ { 1 } - 1 } \sqrt { 1 - \| \bar { x } \| ^ { 2 } } - b ^ { \star } } \big ) } \\ & { \phantom { f _ { \mathrm { b c e - o s } } } + ( K - K _ { 1 } ) \log \big ( 1 + \mathrm { e } ^ { - \gamma \| \bar { x } \| - b ^ { \dagger } } \big ) , } \end{array}\tag{S.63}
$$

where $b ^ { \star }$ is the unique real solution of

$$
\mathrm { e } ^ { \frac { \gamma } { K _ { 1 } - 1 } \sqrt { 1 - \| \tilde { { \boldsymbol { x } } } \| ^ { 2 } } } \mathrm { e } ^ { 2 b } - ( K _ { 1 } - 2 ) \mathrm { e } ^ { b } - ( K _ { 1 } - 1 ) \mathrm { e } ^ { \gamma \sqrt { 1 - \| \tilde { { \boldsymbol { x } } } \| ^ { 2 } } } = 0 ,
$$

and $b ^ { \dagger } \in \mathbb { R }$ . When the BCE loss $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ reaching its lower bound in $E q .$ (S.63), it satisfies

• (collapse)

$$
{ \pmb x } ^ { ( k ) } = \bar { { \pmb x } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in \mathcal { F } _ { k } } { \pmb x } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K _ { 1 } ] ;
$$

• (equiangular)

$$
\pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \frac { 1 } { K _ { 1 } - 1 } , \quad \forall j \neq k \in [ K _ { 1 } ] ,
$$

$$
\bar { \pmb { x } } _ { k } ^ { \top } \bar { \pmb { x } } _ { j } = - \frac { 1 } { K _ { 1 } - 1 } + \frac { K _ { 1 } } { K _ { 1 } - 1 } \| \tilde { \pmb { x } } \| ^ { 2 } , \quad \forall j \neq k \in [ K _ { 1 } ] ;
$$

• (isogonal)

$$
\begin{array} { r } { \pmb { w } _ { k } ^ { \top } \bar { \pmb { x } } _ { k } = \sqrt { 1 - \| \tilde { \pmb { x } } \| ^ { 2 } } , \quad \forall k \in [ K _ { 1 } ] , } \end{array}
$$

$$
\pmb { w } _ { j } ^ { \top } \bar { \pmb { x } } _ { k } = - \frac { 1 } { K _ { 1 } - 1 } \sqrt { 1 - \| \tilde { \pmb { x } } \| ^ { 2 } } , \quad \forall j \neq k \in [ K _ { 1 } ] ,
$$

$$
\pmb { w } _ { k } ^ { \top } \tilde { \pmb { x } } = 0 , \quad \forall k \in [ K _ { 1 } ] ;
$$

• (minority collapse) for $\forall j = K _ { 1 } + 1 , K _ { 1 } + 2 , \cdot \cdot \cdot , K , \ i f \ \| \tilde { \pmb x } \| \neq 0 ,$

$$
{ \pmb w } _ { j } = - \frac { \tilde { \pmb x } } { \lVert \tilde { \pmb x } \rVert } ,
$$

and for $\forall k \in [ K _ { 1 } ] , \ \forall j = K _ { 1 } + 1 , K _ { 1 } + 2 , \cdot \cdot \cdot , K _ { \cdot }$

$$
\begin{array} { r } { \pmb { w } _ { j } ^ { \top } \bar { \pmb { x } } _ { k } = - \| \tilde { \pmb { x } } \| , } \\ { \pmb { w } _ { j } ^ { \top } \pmb { w } _ { k } = 0 , } \end{array}
$$

and

$$
\begin{array} { l } { { b _ { k } = b ^ { \star } , \quad \forall k \in [ K _ { 1 } ] , } } \\ { { b _ { j } = b ^ { \dagger } , \quad \forall j = K _ { 1 } + 1 , K _ { 1 } + 2 , \cdot \cdot \cdot , K . } } \end{array}
$$

As the BCE loss $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ approaching its minimum value, $b ^ { \dagger } \to + \infty \ a n d \ \| \tilde { \pmb x } \| \to 0$

Proof

According to Lemma S.10, for any $c _ { 1 } , c _ { 2 } , c _ { 3 } > 0$ and any $\pmb { X } ^ { ( k ) } \in \mathcal { D } _ { k }$ with $\forall k \in [ K _ { 1 } ]$ 2

$$
\begin{array} { r l } & { { \cal L } _ { \mathrm { b c e } } ( { \pmb X } ^ { ( k ) } ) \geq \displaystyle \frac { 1 } { 1 + c _ { 1 } } \big ( - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } \big ) + \frac { 1 } { 1 + c _ { 2 } } \displaystyle \sum _ { j \neq k } ^ { K _ { 1 } } \big ( \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } \big ) } \\ & { \qquad + \displaystyle \frac { 1 } { 1 + c _ { 3 } } \displaystyle \sum _ { j = K _ { 1 } + 1 } ^ { K } \big ( \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } \big ) + C , } \end{array}
$$

where

$$
\begin{array} { l } { { \displaystyle C = \log ( 1 + c _ { 1 } ) - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K _ { 1 } - 1 ) \biggl ( \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \biggr ) } } \\ { { \displaystyle ~ + ( K - K _ { 1 } ) \biggl ( \log ( 1 + c _ { 3 } ) - \frac { c _ { 3 } } { 1 + c _ { 3 } } \log ( c _ { 3 } ) \biggr ) , } } \end{array}
$$

and the inequality becomes an equality if and only if

$$
c _ { 1 } = \exp \big ( \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { k } \big ) , \quad 1 \leq k \leq K _ { 1 } ,
$$

$$
c _ { 2 } = \exp { \left( b _ { j } - \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } \right) } , \quad 1 \leq j \neq k \leq K _ { 1 } ,
$$

$$
c _ { 3 } = \exp { \big ( } b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } \big ) , \quad 1 \leq k \leq K _ { 1 } , \ K _ { 1 } < j \leq K .
$$

We take the same $c _ { 1 } , c _ { 2 }$ , and $c _ { 3 }$ for all $X ^ { ( k ) } , \forall k \in [ K _ { 1 } ]$ , then

$$
\begin{array} { r l } & { \quad _ { 1 } = - \sinh x _ { 2 } , } \\ & { \quad _ { 2 } = \sinh x _ { 3 } , } \\ & { \quad _ { 3 } = \sinh x _ { 4 } , } \\ & { \quad _ { 4 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 4 } , } \\ & { \quad _ { 4 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 4 } , } \\ & { \quad _ { 5 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 5 } , } \\ & { \quad _ { 6 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 6 } , } \\ & { \quad _ { 7 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 7 } , } \\ & { \quad _ { 8 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 7 } , } \\ & { \quad _ { 8 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 7 } , } \\ & { \quad _ { 8 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 9 } , } \\ & { \quad _ { 8 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 1 1 } , } \\ & { \quad _ { 1 0 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 1 2 } , } \\ & { \quad _ { 1 0 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 1 3 } , } \\ & { \quad _ { 1 0 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 1 3 } , } \\ & { \quad _ { 1 0 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 1 4 } , } \\ & { \quad _ { 1 0 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 1 4 } , } \\ & { \quad _ { 1 0 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 1 5 } , } \\ & { \quad _ { 2 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 2 4 } , } \\ & { \quad _ { 2 } = \frac { \sqrt { 3 } } { 2 } \sinh x _ { 2 4 } , } \\ &  \quad _ { 3 } = \frac { \sqrt { 3 } }  \end{array}
$$

$$
\begin{array} { r l } & { \quad - \displaystyle \frac { 1 } { K _ { 1 } } \displaystyle \frac { 1 } { 1 + c _ { 3 } } \sum _ { k = 1 } ^ { K _ { 1 } } \displaystyle \frac { 1 } { n _ { k } } \sum _ { \substack { x ^ { ( k ) } \in \mathcal { F } _ { k } } } \displaystyle \sum _ { b = K _ { 1 } + 1 } ^ { K } b _ { j } + C } \\ & { = \displaystyle \frac { 1 } { 1 + c _ { 2 } } \sum _ { k = 1 } ^ { K _ { 1 } } \gamma w _ { k } ^ { \top } \big ( \bar { x } - \bar { x } _ { k } \big ) + \displaystyle \frac { 1 } { K _ { 1 } } \Big ( \displaystyle \frac { K _ { 1 } - 1 } { 1 + c _ { 2 } } - \displaystyle \frac { 1 } { 1 + c _ { 1 } } \Big ) \displaystyle \sum _ { k = 1 } ^ { K _ { 1 } } \big ( \gamma w _ { k } ^ { \top } \bar { x } _ { k } - b _ { k } \big ) } \\ & { \qquad + \displaystyle \frac { 1 } { 1 + c _ { 3 } } \displaystyle \sum _ { j = K _ { 1 } + 1 } ^ { K } \big ( \gamma w _ { j } ^ { \top } \bar { x } - b _ { j } \big ) + C } \\ & { \triangleq \hat { f } _ { \mathrm { b e e s e } } ^ { ( 1 ) } ( W , b , \mathcal { F } ) , } \end{array}
$$

where $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K _ { 1 } } \sum _ { j = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { j } } \end{array}$ and $\begin{array} { r } { \pmb { \bar { x } } _ { j } = \frac { 1 } { n _ { j } } \sum _ { \pmb { x } ^ { ( j ) } \in \mathcal { F } _ { j } } \pmb { x } ^ { ( j ) } , \forall j \in [ K _ { 1 } ] , } \end{array}$

For the function $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 1 ) } ( W , b , \mathcal { F } )$ , its global minima are located at its stationary points in terms of $b _ { k }$ for $\forall k \in [ K _ { 1 } ]$ ，

$$
0 = \frac { \partial \hat { f } _ { \mathrm { b c e - o s } } } { \partial b _ { k } } = - \frac { 1 } { K _ { 1 } } \bigg ( \frac { K _ { 1 } - 1 } { 1 + c _ { 2 } } - \frac { 1 } { 1 + c _ { 1 } } \bigg ) .\tag{S.64}
$$

Therefore,

$$
\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 1 ) } ( { \pmb W } , { \pmb b } , { \mathscr F } ) \geq \frac { 1 } { 1 + c _ { 2 } } \sum _ { k = 1 } ^ { K _ { 1 } } \gamma { \pmb w } _ { k } ^ { \top } \big ( \tilde { { \pmb x } } - \bar { { \pmb x } } _ { k } \big ) + \frac { 1 } { 1 + c _ { 3 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } \gamma { \pmb w } _ { j } ^ { \top } \hat { { \pmb x } } - \frac { 1 } { 1 + c _ { 3 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } b _ { j } + C .
$$

Furthermore, according to the AM-GM inequality (see Lemma S.2),

$$
u ^ { \top } v \geq - \Big ( \frac { c } { 2 } \| u \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c } \| v \| _ { 2 } ^ { 2 } \Big ) , \quad \forall u , v \in \mathbb { R } ^ { d } , \ \forall c > 0 ,
$$

which becomes an equality when $\mathbf { \nabla } \pmb { v } = - c \pmb { u }$ . Then,

$$
\begin{array} { r l } & { \quad \hat { f } _ { \mathrm { R e s e } } ( M , \mathbf { b } , F ) \geq \frac { \hat { f } _ { 1 } ^ { ( 1 ) } } { \hat { f } _ { 1 \mathrm { S e } } ^ { \tau } } \mathrm { d } W , } \\ & { \quad \hat { \mathbf { s } } _ { 1 } ^ { - } \frac { 1 } { 1 + \alpha _ { 2 } } \displaystyle \sum _ { i = 1 } ^ { M } \frac { K _ { i } } { \gamma \alpha \kappa _ { i } } \mathrm { d } \overline { { s } } _ { 1 } ^ { \tau } + \overline { { \kappa } } _ { 2 } \displaystyle \sum _ { i = 1 } ^ { M } \frac { K _ { i } } { \gamma \alpha \kappa _ { i } + 1 } \mathrm { d } w _ { i } \hat { \mathbf { x } } - \frac { 1 } { 1 + \alpha _ { 3 } } \displaystyle \sum _ { i = 1 } ^ { M } \frac { K _ { i } } { \gamma \alpha \kappa _ { i } + 1 } \mathrm { d } \overline { { s } } _ { 1 } ^ { \tau } } \\ & { \quad \hat { \mathbf { s } } - \frac { \overline { { g } } } { 1 + \tau _ { 2 } } ( \frac { \kappa _ { i } } { 2 } \displaystyle \sum _ { i = 1 } ^ { M } \| w _ { i } | ^ { 2 } + \frac { 1 } { 2 \alpha _ { 4 } } \displaystyle \sum _ { k = 1 } ^ { M } \| \bar { x } - w _ { k } \| ^ { 2 } ) + \frac { 1 } { 1 + \alpha _ { 3 } } \displaystyle \sum _ { i = 1 } ^ { M } \frac { K } { 1 + \alpha _ { 4 } } \mathrm { d } w _ { i } ^ { \tau } \bar { \mathbf { x } } } \\ & { \quad \quad - \frac { 1 } { 1 + \alpha _ { 3 } } \displaystyle \sum _ { i = 1 + 1 } ^ { M } b _ { i } + \mathcal { C } } \\ &  \quad \quad \geq - \frac { \overline { { g } } } { 1 + \alpha _ { 2 } } ( \displaystyle \sum _ { i = 1 } ^ { M } \| w _ { i } \| ^ { 2 } + \frac { 1 } { 2 \alpha _ { 4 } } \displaystyle \sum _ { k = 1 } \end{array}\tag{S.65}
$$

(S.66)

$$
\begin{array} { r l } & { = - \cfrac { \gamma } { 1 + c \varrho _ { 2 } } \left[ \cfrac { c _ { 4 } } { 2 } \cfrac { K } { \kappa - 1 } \| { \boldsymbol u } _ { \bar { x } ^ { k } } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c _ { 4 } } \bigg ( \cfrac { K 1 } { k - 1 } \| { \boldsymbol x } _ { k } \| _ { 2 } ^ { 2 } - K _ { 1 } \| { \boldsymbol \bar { x } } \| _ { 2 } ^ { 2 } \bigg ) \right] } \\ & { \phantom { = } - \gamma \frac { K - K _ { 1 } } { 1 + c _ { 3 } } \| { \boldsymbol \bar \kappa } \| - \frac { 1 } { 1 + c _ { 3 } } \cfrac { K } { \rho \kappa k _ { 1 } + 1 } } \\ & { \geq - \frac { \gamma } { 1 + c _ { 2 } } \bigg ( \cfrac { c _ { 4 } } { 2 } K _ { 1 } + \frac { 1 } { 2 c _ { 4 } } K _ { 1 } \bigg ) + \frac { \gamma } { 1 + c _ { 2 } } \frac { 1 } { 2 c _ { 4 } } K _ { 1 } \| { \boldsymbol x } \| _ { 2 } ^ { 2 } } \\ & { \phantom { = } - \gamma \frac { K - K _ { 1 } } { 1 + c _ { 3 } } \| { \boldsymbol \bar \kappa } \| - \frac { 1 } { 1 + c _ { 3 } } \cfrac { K } { \rho \kappa _ { 1 } + 1 } + { C } } \\ & { = - \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \bigg ( \cfrac { c _ { 4 } } { 2 } + \frac { 1 } { 2 c _ { 4 } } \bigg ) + \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \frac { 1 } { 2 c _ { 4 } } \| { \boldsymbol \kappa } \| ^ { 2 } - \gamma \frac { K - K _ { 1 } } { 1 + c _ { 3 } } \| { \boldsymbol \bar \kappa } \| - \frac { 1 } { 1 + c _ { 3 } } \cfrac { K } { \rho \kappa _ { 1 } + 1 } + { C } , } \end{array}\tag{S.67}
$$

where inequality (S.65) becomes an equality if and only if

$$
c _ { 4 } \pmb { w } _ { k } = \bar { \pmb { x } } _ { k } - \tilde { \pmb { x } } , \quad \forall k \in [ K ] , \forall c _ { 4 } > 0 ,\tag{S.68}
$$

inequality (S.66) becomes an equality if and only if

$$
c _ { 5 } { \pmb w } _ { j } = - { \tilde { \pmb x } } , \quad K _ { 1 } + 1 \leq j \leq K , \forall c _ { 5 } > 0 ,\tag{S.69}
$$

and inequality (S.67) becomes an equality if and only if

$$
\begin{array} { l } { 1 = \| \bar { \pmb { x } } _ { k } \| = \left\| \displaystyle \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \pmb { x } ^ { ( k ) } \right\| \leq \displaystyle \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \| \pmb { x } ^ { ( k ) } \| = 1 } \\ { \Leftrightarrow \pmb { x } ^ { ( k ) } = \bar { \pmb { x } } _ { k } , \forall \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \quad \forall k \in [ K _ { 1 } ] . } \end{array}\tag{S.70}
$$

With Eq. (S.69), $\| \tilde { \pmb { x } } \| = c _ { 5 }$ , then

$$
\begin{array} { r l } { { } } & { { \mathrm {  ~ { \cal ~ f } ~ } _ { \mathrm { b c e s o s } } ( W , b , \mathcal { F } ) } } \\ { { } } & { { \geq \ - \ \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \biggl ( \frac { c _ { 4 } } { 2 } + \frac { 1 } { 2 c _ { 4 } } \biggr ) + \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \frac { 1 } { 2 c _ { 4 } } \| \tilde { x } \| ^ { 2 } - \frac { \gamma ( K - K _ { 1 } ) } { 1 + c _ { 3 } } \| \tilde { x } \| - \frac { 1 } { 1 + c _ { 3 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } b _ { j } + C } } \\ { { } } & { { = \ - \ \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \biggl ( \frac { c _ { 4 } } { 2 } + \frac { 1 } { 2 c _ { 4 } } \biggr ) + \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \frac { c _ { 5 } ^ { 2 } } { 2 c _ { 4 } } - \frac { \gamma ( K - K _ { 1 } ) } { 1 + c _ { 3 } } c _ { 5 } - \frac { 1 } { 1 + c _ { 3 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } b _ { j } + C . \eqno { ( \mathrm { S M } ) } } } \end{array}\tag{.71}
$$

Finally, Lemma S.9 completes the proof.

Lemma S.9 Suppose that the first $K _ { 1 }$ classes in dataset D contains at least one sample, $K _ { 1 } < K$ , while the other classes are empty, and that the feature dimension is greater than the number of classes, i.e., $n _ { k } = | \mathcal { D } _ { k } | > 0 ~ f o r ~ \forall k \in [ K _ { 1 } ] , ~ n _ { k } = | \mathcal { D } _ { k } | = 0 ~ f o r ~ K _ { 1 } < k \le K$

and $d \geq K _ { 1 }$ . Then, for the BCE loss $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ defined in Eq. (S.62),

$$
\begin{array} { r l } & { f _ { \mathrm { b c e } } ( W , b , \mathcal { F } ) \geq \log \left( 1 + \mathrm { e } ^ { b ^ { \star } - \gamma \sqrt { 1 - \| \bar { x } \| ^ { 2 } } } \right) + \left( K _ { 1 } - 1 \right) \log \left( 1 + \mathrm { e } ^ { - \frac { \gamma } { K _ { 1 } - 1 } \sqrt { 1 - \| \bar { x } \| ^ { 2 } } - b ^ { \star } } \right) } \\ & { \phantom { f _ { \mathrm { b c e } } ( } + \left( K - K _ { 1 } \right) \log \left( 1 + \mathrm { e } ^ { - \gamma \| \bar { x } \| - b ^ { \dagger } } \right) , } \end{array}\tag{S.72}
$$

where $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { k } } \end{array}$ with $\begin{array} { r } { \bar { \mathbf { x } } _ { k } \ = \ \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \pmb { x } ^ { ( k ) } , \ b ^ { \dagger } \ \in \ \mathbb { R } } \end{array}$ , and $b ^ { \star }$ is the unique real solution of

$$
\begin{array} { r } { \mathrm { e } ^ { \frac { \gamma \sqrt { 1 - \| { \tilde { \mathbf { x } } } \| ^ { 2 } } } { K _ { 1 } - 1 } } \mathrm { e } ^ { 2 b } - ( K _ { 1 } - 2 ) \mathrm { e } ^ { b } - ( K _ { 1 } - 1 ) \mathrm { e } ^ { \gamma \sqrt { 1 - \| { \tilde { \mathbf { x } } } \| ^ { 2 } } } = 0 . } \end{array}\tag{S.73}
$$

When $f _ { \mathrm { b c e } } ( W , b , \mathcal { F } )$ approaching its minimum value, $b ^ { \dagger }  + \infty$ and $\| \tilde { \pmb x } \|  0$

Proof According to Theorem S.8, the BCE loss $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ satisfies

$$
\begin{array} { l } { f _ { \mathrm { b c e - o s } } ( { \boldsymbol W } , { \boldsymbol b } , \mathcal { F } ) \ge - \displaystyle \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \bigg ( \frac { c _ { 4 } } { 2 } + \frac { 1 } { 2 c _ { 4 } } \bigg ) + \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \frac { c _ { 5 } ^ { 2 } } { 2 c _ { 4 } } - \frac { \gamma ( K - K _ { 1 } ) } { 1 + c _ { 3 } } c _ { 5 } } \\ { \displaystyle - \frac { 1 } { 1 + c _ { 3 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } b _ { j } + C , } \end{array}\tag{S.74}
$$

and the loss achieves this constant lower bound if and only if Eqs. (S.94), (S.95), (S.96), (S.64), (S.68), (S.69), and (S.70) hold.

From Eq. (S.68), for $\forall c _ { 4 } > 0$

$$
c _ { 4 } \sum _ { k = 1 } ^ { K _ { 1 } } w _ { k } = \sum _ { k = 1 } ^ { K _ { 1 } } ( { \bar { x } } _ { k } - { \tilde { x } } ) = \sum _ { k = 1 } ^ { K _ { 1 } } { \bar { x } } _ { k } - K _ { 1 } { \tilde { x } } = \sum _ { k = 1 } ^ { K _ { 1 } } { \bar { x } } _ { k } - { \frac { K _ { 1 } } { K _ { 1 } } } \sum _ { k = 1 } ^ { K _ { 1 } } { \bar { x } } _ { k } = \mathbf { 0 } \Leftrightarrow \sum _ { k = 1 } ^ { K _ { 1 } } w _ { k } = \mathbf { 0 } .\tag{S.75}
$$

Then, from Eq. (S.68),

$$
\begin{array} { l c l } { { c _ { 1 } w _ { k } = x _ { k } - \bar { x } } } & { { \Rightarrow } } & { { c _ { 4 } = c _ { 4 } w _ { k } ^ { \top } w _ { k } - w _ { k } ^ { \top } x _ { k } } } \\ { { } } & { { } } & { { \displaystyle \leq } } & { { \begin{array} { c } { { K _ { 1 } c _ { 4 } = \displaystyle \sum _ { k = 1 } ^ { K } w _ { k } ^ { \top } \bar { x } _ { k } - ( \displaystyle \sum _ { k = 1 } ^ { K } w _ { k } ) ^ { \top } \bar { x } } } \\ { { } } \\ { { } } \\ { } \\ { { \displaystyle \leq } } \\ { {  \begin{array} { c } { { S _ { \lfloor \bar { k } \rfloor } ^ { \top } } } \\ { { } } \\ { { } } \\ { { \vdots } w _ { k } ^ { \top } x _ { k } - \bar { x } } \end{array}  } } \\ { { c _ { 1 } w _ { k } = x _ { k } - \bar { x } } } & { { \to } } & { { c _ { 4 } w _ { k } ^ { \top } \bar { x } _ { k } - \bar { x } _ { k } ^ { \top } x _ { k } } } \end{array} } } \\ { { } } & { { } } &   \begin{array} { r l } { { \displaystyle \leq _ { \bar { k } \leq 1 } ^ { \top } w _ { \bar { k } } ^ { \top } = { \binom { K } { \bar { k } \leq 1 } } } } & { { \displaystyle \sum _ { k = 1 } ^ { K } \bar { x } _ { \bar { k } } - ( \displaystyle \sum _ { k = 1 } ^ { K } \bar { x } _ { \bar { k } } ^ { \top } \bar { x } _ { k } - K _ { 1 } ) ^ { \top } \bar { x } _ { \bar { k } } } } \\ { { } } & { { } } \\ { { \displaystyle \leq } } &   \begin{array} { c }   \displaystyle \leq _ { \bar { k } \leq 1 } ^ { \top } \bar { x } _ { \bar { k } } - ( \displaystyle - \sum _ { k = 1 } ^ { K } \bar  \end{array} \end{array} \end{array}\tag{S.76}
$$

(S.77)

$$
\begin{array} { r l r } & { \overset { S : \eta _ { 0 } , S , T 7 } { = } \quad 1 - c _ { 4 } ^ { 2 } = \| \tilde { x } \| ^ { 2 } = \| \bar { x } _ { k } \| ^ { 2 } + c _ { 4 } ^ { 2 } \| w _ { k } \| ^ { 2 } - 2 c _ { 4 } w _ { k } ^ { \top } \bar { x } _ { k } = 1 + c _ { 4 } ^ { 2 } - 2 c _ { 4 } w _ { k } ^ { \top } \bar { x } _ { k } } \\ & { \Rightarrow \quad w _ { k } ^ { \top } \bar { x } _ { k } = c _ { 4 } = \sqrt { 1 - \| \tilde { x } \| ^ { 2 } } , \quad \forall k \in [ K ] , } & { ( \mathrm { S . 7 8 } ) } \\ & { c _ { 4 } w _ { k } = \bar { x } _ { k } - \tilde { x } } & { \Rightarrow \quad c _ { 4 } = c _ { 4 } w _ { k } ^ { \top } w _ { k } = w _ { k } ^ { \top } \bar { x } _ { k } - w _ { k } ^ { \top } \tilde { x } } \\ & { \overset { S : \eta \mathrm { s } } { \Rightarrow } \quad w _ { k } ^ { \top } \tilde { x } = 0 , \quad \forall k \in [ K ] . } & { ( \mathrm { S . 7 9 } ) } \end{array}
$$

Meanwhile,

$$
\begin{array} { r l r } & { c _ { 4 } \boldsymbol { w } _ { k } = \boldsymbol { \bar { x } } _ { k } - \tilde { \boldsymbol { x } } \quad \Rightarrow \quad c _ { 4 } \boldsymbol { w } _ { k } ^ { \top } \boldsymbol { w } _ { j } = \boldsymbol { w } _ { j } ^ { \top } \boldsymbol { \bar { x } } _ { k } - \boldsymbol { w } _ { j } ^ { \top } \tilde { \boldsymbol { x } } } & \\ { \Rightarrow \quad c _ { 4 } \gamma \boldsymbol { w } _ { k } ^ { \top } \boldsymbol { w } _ { j } = \gamma \boldsymbol { w } _ { j } ^ { \top } \bar { \boldsymbol { x } } _ { k } - b _ { j } - \gamma \boldsymbol { w } _ { j } ^ { \top } \tilde { \boldsymbol { x } } + b _ { j } = \gamma \boldsymbol { w } _ { j } ^ { \top } \bar { \boldsymbol { x } } _ { k } - b _ { j } - \frac 1 { K _ { 1 } } \displaystyle \sum _ { k = 1 } ^ { K _ { 1 } } \left( \gamma \boldsymbol { w } _ { j } ^ { \top } \bar { \boldsymbol { x } } _ { k } - b _ { j } \right) } & \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times 9 4 , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times _ { 1 } } \\ { \Rightarrow \quad \boldsymbol { w } _ { k } ^ { \top } \boldsymbol { w } _ { j } = - \frac { c _ { 1 } ^ { \prime } + c _ { 2 } ^ { \prime } } { c _ { 4 } \gamma K _ { 1 } } , \quad \forall j \neq k \in [ K _ { 1 } ] , } & & { ( \mathrm { S } . } \end{array}\tag{.80}
$$

where $c _ { 1 } ^ { \prime } = \log ( c _ { 1 } ) , c _ { 2 } ^ { \prime } = \log ( c _ { 2 } )$ . From Eq. (S.75)

$$
\begin{array} { r l r } { \displaystyle \sum _ { j = 1 } ^ { K _ { 1 } } w _ { j } = 0 \quad \Rightarrow \quad w _ { k } ^ { \top } \displaystyle \sum _ { j = 1 } ^ { K _ { 1 } } w _ { j } = 0 \quad \Rightarrow \quad 1 = - \displaystyle \sum _ { j \neq k } ^ { K _ { 1 } } w _ { k } ^ { \top } w _ { j } \stackrel { S \cdot 8 0 } { = } ( K _ { 1 } - 1 ) \frac { c _ { 1 } ^ { \prime } + c _ { 2 } ^ { \prime } } { c _ { 4 } \gamma K _ { 1 } } } \\ { \displaystyle } & { \Rightarrow \quad \frac { c _ { 1 } ^ { \prime } + c _ { 2 } ^ { \prime } } { c _ { 4 } \gamma K _ { 1 } } = \displaystyle \frac { 1 } { K _ { 1 } - 1 } \quad \Rightarrow \quad w _ { k } ^ { \top } w _ { j } = - \displaystyle \frac { 1 } { K _ { 1 } - 1 } , \forall j \neq k \in [ K _ { 1 } ] . } \end{array}\tag{S.81}
$$

Then,

$$
\begin{array} { r l } & { 2 - 2 \bar { x } _ { k } ^ { \top } \bar { x } _ { j } = \| \bar { x } _ { k } - \bar { x } _ { j } \| ^ { 2 } \overset { S , 6 8 } { = } \| c _ { 4 } w _ { k } - c _ { 4 } w _ { j } \| ^ { 2 } } \\ & { \qquad = c _ { 4 } ^ { 2 } \big ( \| w _ { k } \| ^ { 2 } + \| w _ { j } \| ^ { 2 } - 2 w _ { k } ^ { \top } w _ { j } \big ) \overset { S , 8 1 } { = } 2 c _ { 4 } ^ { 2 } \frac { K _ { 1 } } { K _ { 1 } - 1 } } \\ { \Rightarrow } & { \bar { x } _ { k } ^ { \top } \bar { x } _ { j } = 1 - \displaystyle \frac { K _ { 1 } } { K _ { 1 } - 1 } c _ { 4 } ^ { 2 } = 1 - \displaystyle \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( 1 - \| \bar { x } \| ^ { 2 } \big ) } \\ & { \qquad = - \displaystyle \frac { 1 } { K _ { 1 } - 1 } + \displaystyle \frac { K _ { 1 } } { K _ { 1 } - 1 } \| \tilde { x } \| ^ { 2 } , \quad \forall j \neq k \in [ K _ { 1 } ] , } \end{array}\tag{S.82}
$$

and

$$
\begin{array} { r l } & { c _ { 4 } \boldsymbol { w } _ { k } ^ { \top } \bar { \boldsymbol { x } } _ { j } \stackrel { S , 6 8 } { = } \bar { \boldsymbol { x } } _ { k } ^ { \top } \bar { \boldsymbol { x } } _ { j } - \tilde { \boldsymbol { x } } ^ { \top } \bar { \boldsymbol { x } } _ { j } = \bar { \boldsymbol { x } } _ { k } ^ { \top } \bar { \boldsymbol { x } } _ { j } - \displaystyle \frac { 1 } { K _ { 1 } } \sum _ { \ell = 1 } ^ { K _ { 1 } } \bar { \boldsymbol { x } } _ { \ell } ^ { \top } \bar { \boldsymbol { x } } _ { j } } \\ & { \qquad \stackrel { S , 8 2 } { = } - \displaystyle \frac { 1 } { K _ { 1 } } + \left( 1 - \frac { K _ { 1 } - 1 } { K _ { 1 } } \right) \left( 1 - \frac { K _ { 1 } } { K _ { 1 } - 1 } c _ { 4 } ^ { 2 } \right) = - \frac { 1 } { K _ { 1 } - 1 } c _ { 4 } ^ { 2 } } \\ { \Rightarrow } & { \boldsymbol { w } _ { k } ^ { \top } \bar { \boldsymbol { x } } _ { j } = - \displaystyle \frac { 1 } { K _ { 1 } - 1 } c _ { 4 } = - \frac { \sqrt { 1 - \| \tilde { \boldsymbol { x } } \| ^ { 2 } } } { K _ { 1 } - 1 } , \quad \forall j \neq k \in [ K _ { 1 } ] . } \end{array}\tag{S.83}
$$

Now, we try to express the lower bound of $f _ { \mathrm { b c e } }$ using $_ { c _ { 4 } ; }$ to this end, it is necessary to derive the relationship between $c _ { 4 }$ with $c _ { 1 } , c _ { 2 } , c _ { 3 } ,$ , and $c _ { 5 }$ . In first, from Eq. (S.64),

$$
\begin{array} { r l } & { 0 = \cfrac { K _ { 1 } - 1 } { 1 + c _ { 2 } } - \cfrac { 1 } { 1 + c _ { 1 } } \overset { S \cdot 9 4 , S \cdot 9 5 , S \cdot 7 0 } { \Rightarrow } ( K _ { 1 } - 2 ) + ( K _ { 1 } - 1 ) \mathrm { e } ^ { \gamma w _ { k } ^ { \top } \bar { x } _ { k } - b _ { k } } = \mathrm { e } ^ { b _ { k } - \gamma w _ { k } ^ { \top } \bar { x } _ { j } } \overset { S \cdot 7 8 , S \cdot 8 3 } { S \cdot 1 8 } } \\ & { \qquad \Rightarrow \quad ( K _ { 1 } - 2 ) + ( K _ { 1 } - 1 ) \mathrm { e } ^ { \gamma c _ { 4 } - b _ { k } } = \mathrm { e } ^ { b _ { k } + \frac { \gamma } { K _ { 1 } - 1 } c _ { 4 } } , } \end{array}
$$

then,

$$
b _ { k } = b ^ { \star } , \quad \forall k \in [ K _ { 1 } ] ,\tag{S.84}
$$

where $b ^ { \star }$ is the unique real solution of

$$
\mathrm { e } ^ { \frac { \gamma c _ { 4 } } { K _ { 1 } - 1 } } \mathrm { e } ^ { 2 b } - ( K _ { 1 } - 2 ) \mathrm { e } ^ { b } - ( K _ { 1 } - 1 ) \mathrm { e } ^ { \gamma c _ { 4 } } = 0 ,
$$

i.e., the solution of Eq. (S.73). Then,

$$
\begin{array} { r l } & { c _ { 1 } = \exp \big ( \gamma \pmb { w } _ { k } ^ { \top } \bar { \pmb { x } } _ { k } - b _ { k } \big ) = \exp \big ( \gamma c _ { 4 } - b ^ { \star } \big ) , } \\ & { c _ { 2 } = \exp \big ( b _ { k } - \gamma \pmb { w } _ { k } ^ { \top } \bar { \pmb { x } } _ { j } \big ) = \exp \big ( b ^ { \star } + \frac { \gamma c _ { 4 } } { K _ { 1 } - 1 } \big ) . } \end{array}\tag{S.85}
$$

(S.86)

From Eq. (S.69),

$$
c _ { 5 } { \pmb w } _ { j } = - \tilde { \pmb x } \quad \Rightarrow \quad c _ { 5 } ^ { 2 } \| { \pmb w } _ { j } \| ^ { 2 } = \| \tilde { \pmb x } \| ^ { 2 } \quad \Rightarrow \quad c _ { 5 } ^ { 2 } \stackrel { S . 7 7 } { = } 1 - c _ { 4 } ^ { 2 } ,\tag{S.87}
$$

$$
\begin{array} { r l } { c _ { 5 } w _ { j } = - \bar { x } } & { \Rightarrow \quad c _ { 5 } w _ { j } ^ { \top } \bar { x } _ { k } = - \bar { x } ^ { \top } \bar { x } _ { k } = - \frac { 1 } { K _ { 1 } } \displaystyle \sum _ { \ell = 1 } ^ { K _ { 1 } } \frac { 1 } { c _ { \ell } + 1 } x _ { \ell } ^ { \top } \bar { x } _ { k } } \\ & { \qquad \quad \overset { s \leq 2 } - \displaystyle \frac { 1 } { K _ { 1 } } \Big [ 1 + ( K _ { 1 } - 1 ) \Big ( 1 - \frac { K _ { 1 } } { K _ { 1 } - 1 } c _ { \ell } ^ { 2 } \Big ) \Big ] } \\ & { \qquad \quad - \displaystyle \frac { 1 } { K _ { 1 } } \big ( K _ { 1 } - K _ { 1 } c _ { \ell } ^ { 2 } \big ) } \\ { \quad \quad \Rightarrow \quad w _ { j } ^ { \top } \bar { x } _ { k } = - \frac { 1 - c _ { \ell } ^ { 2 } } { c _ { 5 } } \frac { s \leq u _ { j } } { \omega _ { j } } - \sqrt { 1 - c _ { \ell , j } ^ { 2 } } , \quad K _ { 1 } < j \leq K , \forall k \in [ K _ { 1 } ] } \\ { \quad \Rightarrow \quad c _ { 3 } = \exp \big ( b _ { j } - \gamma w _ { j } ^ { \top } \bar { x } _ { k } \big ) = \exp \big ( b _ { j } + \gamma \sqrt { 1 - c _ { \ell } ^ { 2 } } \big ) , \quad K _ { 1 } < j } \\ { \quad \quad \leq \quad K , \forall k \in [ K _ { 1 } ] } \\ { \quad \quad \Rightarrow \quad b _ { j } = b _ { j } = b _ { j } ^ { \top } , \quad \forall K _ { 1 } \leq j \neq \ell \leq K . } \end{array}\tag{S.88}
$$

(S.89)

(S.90)

Therefore,

$$
\begin{array} { r l } { { } } & { { f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } ) } } \\ { { } } & { { \geq \ - \ \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \bigg ( \frac { c _ { 4 } } { 2 } + \frac { 1 } { 2 c _ { 4 } } \bigg ) + \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \frac { c _ { 5 } ^ { 2 } } { 2 c _ { 4 } } - \frac { \gamma \left( K - K _ { 1 } \right) } { 1 + c _ { 3 } } c _ { 5 } - \frac { 1 } { 1 + c _ { 3 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } b _ { j } + C } } \\ { { } } & { { = \ - \ \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \bigg ( \frac { c _ { 4 } } { 2 } + \frac { 1 } { 2 c _ { 4 } } \bigg ) + \frac { \gamma K _ { 1 } } { 1 + c _ { 2 } } \frac { 1 - c _ { 4 } ^ { 2 } } { 2 c _ { 4 } } - \frac { \gamma \left( K - K _ { 1 } \right) } { 1 + c _ { 3 } } c _ { 5 } - \frac { 1 } { 1 + c _ { 3 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } b _ { j } + C } } \end{array}
$$

$$
\begin{array} { r l } & { = - \frac { \gamma K _ { 1 } } { 1 + \alpha _ { 2 } } c _ { 4 } - \frac { \gamma ( K - K _ { 1 } ) } { 1 + \alpha _ { 3 } } c _ { 5 } - \frac { 1 } { 1 + c _ { 3 } } \frac { K } { \gamma - K _ { 1 } + 1 } b _ { 2 } + C } \\ & { = \log ( 1 + \frac { 1 } { c _ { 4 } } ) + ( K _ { 1 } - 1 ) \log ( 1 + \frac { 1 } { c _ { 2 } } ) + ( K - K _ { 1 } ) \log ( 1 + \frac { 1 } { c _ { 3 } } ) } \\ & { \qquad + ( \gamma c _ { 4 } - b ^ { * } ) \underbrace { ( \frac { 1 } { 1 + c _ { 1 } } - \frac { K _ { 1 } - 1 } { 1 + c _ { 2 } } ) } _ { \mathfrak { D } } - \frac { 1 } { 1 + c _ { 3 } } \frac { K } { \gamma - K _ { 1 } + 1 } b _ { 2 } } \\ & { = \log ( 1 + e ^ { \beta ^ { 2 } - \infty _ { 2 } ^ { 2 } } ) + ( K _ { 1 } - 1 ) \log ( 1 + c ^ { - \frac { \kappa \alpha _ { 4 } } { \kappa _ { 1 } - 1 } - \kappa ^ { 2 } } ) } \\ & { \qquad + ( K - K _ { 1 } ) \log ( 1 + c ^ { - \gamma \sqrt { 1 - c _ { 4 } } } ) } \\ & { \triangleq \widetilde { f } _ { 0 , \infty } ^ { \dagger } \exp (  {  \frac { \kappa } { c } , b ^ { * }  } . } \end{array}
$$

For the function $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 2 ) } ( \| \tilde { \pmb x } \| , b ^ { \star } , b ^ { \dagger } )$

$$
\frac { \partial \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 2 ) } ( \lVert \tilde { { \boldsymbol x } } \rVert , b ^ { \star } , b ^ { \dagger } ) } { \partial b ^ { \dagger } } = - \frac { \mathrm { e } ^ { - \gamma \lVert \tilde { { \boldsymbol x } } \rVert - b ^ { \dagger } } } { 1 + \mathrm { e } ^ { - \gamma \lVert \tilde { { \boldsymbol x } } \rVert - b ^ { \dagger } } } < 0 \quad \forall b ^ { \dagger } \in \mathbb { R } ,
$$

then, when $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 2 ) } ( \| \tilde { \pmb x } \| , b ^ { \star } , b ^ { \dagger } )$ approaching its minimum value,

$$
b ^ { \dagger }  + \infty ,\tag{S.91}
$$

and

$$
\begin{array} { r l } & { \quad f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } ) \geq \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 2 ) } \big ( \| \tilde { { \boldsymbol x } } \| , b ^ { \star } , b ^ { \dagger } \big ) } \\ & { \geq \log \big ( 1 + \mathrm { e } ^ { b ^ { \star } - \gamma \sqrt { 1 - \| \tilde { { \boldsymbol x } } \| ^ { 2 } } } \big ) + ( K _ { 1 } - 1 ) \log \bigg ( 1 + \mathrm { e } ^ { - \frac { \gamma \sqrt { 1 - \| \tilde { { \boldsymbol x } } \| ^ { 2 } } } { K _ { 1 } - 1 } - b ^ { \star } } \bigg ) } \\ & { \triangleq \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } \big ( \| \tilde { { \boldsymbol x } } \| , b ^ { \star } \big ) . } \end{array}\tag{S.92}
$$

For $\begin{array} { r } { \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } \big ( \lVert \tilde { { \boldsymbol x } } \rVert , b ^ { \star } \big ) \mathrm { ~ o n ~ } \mathbb { D } _ { 1 } = \big \{ ( \lVert \tilde { { \boldsymbol x } } \rVert , b ^ { \star } ) : \lVert \tilde { { \boldsymbol x } } \rVert \in [ 0 , 1 ] , b ^ { \star } \big . } \end{array}$ satisfies $E q . \ ( S . 7 3 ) \}$ , as $\mathbb { D } _ { 1 } \subset$ $\mathbb { D } _ { 2 } = \left\{ ( \lVert \tilde { \mathbf { x } } \rVert , \tilde { b } ^ { \star } ) : \lVert \tilde { \mathbf { x } } \rVert \in [ 0 , 1 ] , b ^ { \star } \in \mathbb { R } \right\}$

$$
\operatorname* { m i n } _ { \mathbb { D } _ { 1 } } \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } \big ( \lVert \tilde { \mathbf { x } } \rVert , b ^ { \star } \big ) \geq \operatorname* { m i n } _ { \mathbb { D } _ { 2 } } \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } \big ( \lVert \tilde { \mathbf { x } } \rVert , b ^ { \star } \big ) .
$$

For $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } ( \| \tilde { \pmb { x } } \| , b ^ { \star } )$ on $\mathbb { D } _ { 2 } ,$ it is easily to deduce that $\frac { \partial \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } \Big ( \| \tilde { \pmb { x } } \| , b ^ { \star } \Big ) } { \partial \| \tilde { \pmb { x } } \| } ~ > ~ 0 ;$ therefore on $\mathbb { D } _ { 2 } .$ , when $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } ( \| \tilde { \pmb { x } } \| , b ^ { \star } )$ reaches its minimum value, $\| \tilde { \pmb { x } } \| = 0$ . In terms of $b ^ { \star }$ on $\mathbb { D } _ { 2 }$ when $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } ( \| \tilde { \pmb x } \| , b ^ { \star } )$ reaches its minimum value, $\frac { \partial \hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } \Big ( \lVert \tilde { \mathbf { x } } \rVert , b ^ { \star } \Big ) } { \partial b ^ { \star } } = 0$ , which aligns with $\operatorname { E q }$ (S.73), indicating that the minimum points of $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } ( \| \tilde { \pmb x } \| , b ^ { \star } )$ are contained in ${ \mathbb D } _ { 1 }$ and when $\hat { f } _ { \mathrm { b c e - o s } } ^ { ( 3 ) } ( \| \tilde { \pmb x } \| , b ^ { \star } )$ reaching its minimum value on $\mathbb { D } _ { 1 } , \| \tilde { \pmb { x } } \| = 0$

In total, when $f _ { \mathrm { b c e - o s } } ( W , b , \mathcal { F } )$ approaching its minimum value, $b ^ { \dagger }  + \infty$ and $\| \tilde { \pmb { x } } \|  0 .$ which completes the proof.

Lemma S.10 For any $\pmb { X } ^ { ( k ) } \in \mathcal { D }$ with any $c _ { 1 } , c _ { 2 } > 0$ and $\gamma > 0$ , the BCE loss $L _ { \mathrm { b c e } } ( \mathbf { \mathbf { } } X ^ { ( k ) } )$ defined in Eq. (5) is lower bounded by

$$
\begin{array} { c } { { \displaystyle { \cal L } _ { \mathrm { b c e } } ( { \pmb X } ^ { ( k ) } ) \geq \frac { 1 } { 1 + c _ { 1 } } \big ( - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } \big ) + \frac { 1 } { 1 + c _ { 2 } } \displaystyle \sum _ { j \neq k } ^ { K _ { 1 } } \big ( \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } \big ) } } \\ { { + \displaystyle \frac { 1 } { 1 + c _ { 3 } } \displaystyle \sum _ { j = K _ { 1 } + 1 } ^ { K } \big ( \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } \big ) + C , } } \end{array}\tag{S.93}
$$

where

$$
\begin{array} { l } { { \displaystyle C = \log ( 1 + c _ { 1 } ) - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K _ { 1 } - 1 ) \biggl ( \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \biggr ) } } \\ { { \displaystyle ~ + ( K - K _ { 1 } ) \biggl ( \log ( 1 + c _ { 3 } ) - \frac { c _ { 3 } } { 1 + c _ { 3 } } \log ( c _ { 3 } ) \biggr ) . } } \end{array}
$$

The inequality becomes an equality if and only if

$$
c _ { 1 } = \exp { \left( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \right) } , \quad 1 \leq k \leq K _ { 1 }\tag{S.94}
$$

$$
c _ { 2 } = \exp { \left( b _ { j } - \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } \right) } , \quad 1 \leq j \neq k \leq K _ { 1 } .\tag{S.95}
$$

$$
c _ { 3 } = \exp { \big ( } b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } \big ) , \quad 1 \leq k \leq K _ { 1 } , \ K _ { 1 } < j \leq K .\tag{S.96}
$$

Proof By the concavities of log(x), and according to Lemma S.3, the BCE loss $L _ { \mathrm { b c e } } ( \mathbf { \mathbf { } } X ^ { ( k ) } )$ for any $\pmb { X } ^ { ( k ) } \in \mathcal { D }$ can be lower bounded by

$$
\begin{array} { r l } { \hat { u } _ { \mathrm { i s } } ( X ^ { \theta _ { i } } ) } & { = \operatorname* { i n f } _ { 0 } \Big ( - \infty _ { i } ( 1 - \mathrm { w i t h } \cdot u ^ { \theta _ { i } + \theta _ { i } } ) - \frac { \partial ^ { 2 } \mathrm { K } } { \partial x ^ { \theta _ { i } } } \Big ( \frac { 1 } { 1 + \mathrm { s e } } \nabla ( x ^ { \theta _ { i } } , x ^ { \theta _ { i } } - h _ { i } ) \Big ) } \\ & { \qquad + \underbrace { \frac { \partial ^ { 2 } \mathrm { K } } { \partial x ^ { \theta _ { i } } } } _ { \mathrm { \normalfont { \textit { S - 1 } } } } \frac { 1 } { \mathrm { i } \mathrm { s e } } \Big ( \mathrm { l } \big ( 1 - \mathrm { w i t h } \cdot u ^ { \theta _ { i } + \theta _ { i } } \big ) - \frac { \partial ^ { 2 } \mathrm { K } } { \partial x ^ { \theta _ { i } } } \Big ( \frac { 1 } { 1 + \mathrm { s e } } \big ) } \\ & { \qquad + \underbrace { \frac { \partial ^ { 2 } \mathrm { K } } { \partial x ^ { \theta _ { i } } } } _ { \mathrm { \normalfont { \textit { S - 1 } } } } \frac { 1 } { \mathrm { i } \mathrm { s e } } \frac { \mathrm { i } ( 1 - \mathrm { s e } ) } { \mathrm { i } \mathrm { s e } } \frac { \mathrm { i } ( 1 - \mathrm { s e } ) } { \mathrm { i } \mathrm { s } e } \frac { \mathrm { i } ( 1 - \mathrm { s e } ) } { \mathrm { i } \mathrm { s } e } \frac { \mathrm { i } ( 1 - \mathrm { s e } ) } { \mathrm { i } \mathrm { s } e } \frac { \mathrm { i } ( 1 - \mathrm { s e } ) } { \mathrm { i } \mathrm { s } e } \frac { \mathrm { i } ( 1 - \mathrm { s e } ) } { \mathrm { i } \mathrm { s } e } } \\ &  \qquad + \underbrace { \frac { \partial ^ { 2 } \mathrm { K } } { \partial x ^ { \theta _ { i } } } } _ { \mathrm { \normalfont { \textit { S - 1 } } } } \frac { 1 }  \mathrm  \end{array}
$$

$$
\begin{array} { r l } & { \quad + \displaystyle \frac { 1 } { 1 + c _ { 3 } } \displaystyle \sum _ { j = K _ { 1 } + 1 } ^ { K } \log \Big [ ( 1 + c _ { 3 } ) \exp \big ( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \big ) \Big ] } \\ & { = \displaystyle \frac { 1 } { 1 + c _ { 1 } } \Big ( - \gamma w _ { k } ^ { \top } { \boldsymbol x } ^ { ( k ) } + b _ { k } \Big ) + \frac { 1 } { 1 + c _ { 2 } } \displaystyle \sum _ { j = \mathbb { k } } ^ { K _ { 1 } } \Big ( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \Big ) } \\ & { \quad \quad + \frac { 1 } { 1 + c _ { 3 } } \displaystyle \sum _ { j = K _ { 1 } + 1 } ^ { K } \big ( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \big ) + C . } \end{array}\tag{S.97}
$$

where

$$
\begin{array} { l } { { \displaystyle C = \log ( 1 + c _ { 1 } ) - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K _ { 1 } - 1 ) \biggl ( \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \biggr ) } } \\ { { \displaystyle ~ + ( K - K _ { 1 } ) \biggl ( \log ( 1 + c _ { 3 } ) - \frac { c _ { 3 } } { 1 + c _ { 3 } } \log ( c _ { 3 } ) \biggr ) . } } \end{array}
$$

The inequality becomes an equality if and only if

$$
\begin{array} { l } { \displaystyle \frac { 1 + c _ { 1 } } { c _ { 1 } } = ( 1 + c _ { 1 } ) \exp \left( - \gamma w _ { k } ^ { \top } { \boldsymbol x } ^ { ( k ) } + b _ { k } \right) \mathrm { ~ o r ~ } c _ { 1 } = 0 \mathrm { ~ o r ~ } c _ { 1 } = + \infty , 1 \leq k \leq K _ { 1 } , } \\ { \displaystyle \frac { 1 + c _ { 2 } } { c _ { 2 } } = ( 1 + c _ { 2 } ) \exp \left( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \right) \mathrm { ~ o r ~ } c _ { 2 } = 0 \mathrm { ~ o r ~ } c _ { 2 } = + \infty , 1 \leq j \neq k \leq K _ { 1 } , } \\ { \displaystyle \frac { 1 + c _ { 3 } } { c _ { 3 } } = ( 1 + c _ { 3 } ) \exp \left( \gamma w _ { j } ^ { \top } { \boldsymbol x } ^ { ( k ) } - b _ { j } \right) \mathrm { ~ o r ~ } c _ { 3 } = 0 \mathrm { ~ o r ~ } c _ { 3 } = + \infty , 1 \leq k \leq K _ { 1 } , K _ { 1 } + 1 \leq j \leq K . } \end{array}
$$

These equalities are trivial when $c _ { 1 } = 0 , c _ { 1 } = + \infty , c _ { 2 } = 0 , c _ { 2 } = + \infty , c _ { 3 } = 0 , \mathrm { o r } c _ { 3 } = + \infty .$ Therefore, if and only if

$$
\begin{array} { r l } & { c _ { 1 } = \exp { \big ( \gamma \pmb { w } _ { k } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { k } \big ) } , \quad 1 \le k \le K _ { 1 } , } \\ & { c _ { 2 } = \exp { \big ( b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } \big ) } , \quad 1 \le j \neq k \le K _ { 1 } , } \\ & { c _ { 3 } = \exp { \big ( b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } \big ) } , \quad 1 \le k \le K _ { 1 } , \ K _ { 1 } < j \le K , } \end{array}
$$

Eq. (S.93) becomes an equality, which completes the proof.

Theorem S.11 (CE) Suppose that the first $K _ { 1 }$ classes in dataset D contains at least one sample, $K _ { 1 } ~ < ~ K$ , while the other classes are empty, and that the feature dimension is greater than the number of classes, $i . e . , n _ { k } = \left| \mathcal { D } _ { k } \right| > 0$ for $\forall k \in [ K _ { 1 } ] , n _ { k } = | \mathcal { D } _ { k } | = 0$ for $K _ { 1 } < k \le K$ , and $d \geq K _ { 1 }$ . Denote $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \bar { \pmb { x } } _ { k } } \end{array}$ with $\begin{array} { r } { \bar { \pmb x } _ { k } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in { \mathscr F } _ { k } } { \pmb x } } \end{array}$ . On the dataset $\mathcal { D } , \ d e f i n e$

$$
f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } ) \triangleq \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mathrm { c e } } ( X ^ { ( k ) } ) ,\tag{S.98}
$$

where

$$
L _ { \mathrm { c e } } ( \mathbf { \boldsymbol { X } } ^ { ( k ) } ) = - \log \Big ( \frac { \exp \big ( \gamma \mathbf { \boldsymbol { w } } _ { k } ^ { \top } \mathbf { \boldsymbol { x } } ^ { ( k ) } - b _ { k } \big ) } { \sum _ { j = 1 } ^ { K } \exp \big ( \gamma \mathbf { \boldsymbol { w } } _ { j } ^ { \top } \mathbf { \boldsymbol { x } } ^ { ( k ) } - b _ { j } \big ) } \Big ) ,\tag{S.99}
$$

with $\pmb { x } ^ { ( k ) } = \mathcal { M } ( \pmb { X } ^ { ( k ) } )$ for $\boldsymbol { X } ^ { ( k ) } \in \mathcal { D } _ { k } \subset \mathcal { D }$

Then, the CE loss $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ satisfies

$$
f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } ) \geq \log ( 1 + c _ { 1 } + c _ { 2 } ) ,\tag{S.100}
$$

where

$$
c _ { 1 } \ = ( K _ { 1 } - 1 ) \exp \bigg ( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { { \boldsymbol x } } \| ^ { 2 } - 1 \big ) \bigg )
$$

$$
c _ { 2 } \ = K _ { 1 } \frac { 1 - q ^ { \star } } { q ^ { \star } } \exp \left( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { { \boldsymbol x } } \| ^ { 2 } - 1 \big ) \right)
$$

$$
c _ { 3 } ~ = \frac { 1 } { q ^ { \star } } \sqrt { g ( q ^ { \star } ) } \exp { \left( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { x } \| ^ { 2 } - 1 \big ) \right) } ,
$$

and $g ( q ) = 1 - 2 q \lvert \lvert \tilde { \mathbf { x } } \rvert \rvert ^ { 2 } + q ^ { 2 } \lvert \lvert \tilde { \mathbf { x } } \rvert \rvert ^ { 2 }$ with a certain $q ^ { \star } \in ( 0 , 1 )$ . When the CE loss $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ reaching its lower bound in $E q$ . (S.100), it satisfies

• (collapse)

$$
\pmb { x } ^ { ( k ) } = \bar { \pmb { x } } _ { k } = \frac { 1 } { n _ { k } } \sum _ { \pmb { x } \in \mathcal { F } _ { k } } \pmb { x } , \quad \forall \pmb { x } ^ { ( k ) } , \pmb { x } _ { * } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K _ { 1 } ] ,
$$

• (equiangular)

$$
\bar { x } _ { k } ^ { \top } \bar { x } _ { j } = 1 - \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( 1 - \| \tilde { x } \| ^ { 2 } \big ) = - \frac { 1 } { K _ { 1 } - 1 } + \frac { K _ { 1 } } { K _ { 1 } - 1 } \| \tilde { { \boldsymbol x } } \| ^ { 2 } , \quad \forall j \neq k \in [ K _ { 1 } ] ,
$$

$$
{ \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } = \frac { 1 } { K _ { 1 } } \frac { 1 } { K _ { 1 } - 1 } \bigg ( \frac { c _ { 2 } } { c _ { 3 } } \bigg ) ^ { 2 } \| { \pmb x } \| ^ { 2 } - \frac { 1 } { K _ { 1 } - 1 } , \quad \forall j \neq k \in [ K _ { 1 } ] ,
$$

• (isogonal)

$$
{ \pmb w } _ { k } ^ { \top } \bar { { \pmb x } } _ { k } = \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \right) \frac { 1 } { c _ { 3 } } - \frac { c _ { 1 } } { K _ { 1 } - 1 } \frac { 1 } { c _ { 3 } } \| \tilde { { \pmb x } } \| ^ { 2 } , \quad \forall k \in [ K _ { 1 } ] ,
$$

$$
\begin{array} { r l } & { { \pmb w } _ { k } ^ { \top } \bar { \pmb x } _ { j } = \displaystyle \frac { 1 } { c _ { 3 } } \bigg ( \displaystyle \frac { 1 } { K _ { 1 } - 1 } \displaystyle \frac { c _ { 1 } } { K _ { 1 } - 1 } + \displaystyle \frac { K _ { 1 } } { K _ { 1 } - 1 } \displaystyle \frac { c _ { 2 } } { K _ { 1 } } \bigg ) \| \tilde { \pmb x } \| ^ { 2 } } \\ & { \quad \quad \quad - \displaystyle \frac { 1 } { c _ { 3 } } \displaystyle \frac { 1 } { K _ { 1 } - 1 } \bigg ( \displaystyle \frac { c _ { 1 } } { K _ { 1 } - 1 } + \displaystyle \frac { c _ { 2 } } { K _ { 1 } } \bigg ) , \quad \forall j \neq k \in [ K _ { 1 } ] , } \end{array}
$$

$$
{ \pmb w } _ { k } ^ { \top } \tilde { { \pmb x } } = \frac { c _ { 2 } } { c _ { 3 } } \frac { \| \tilde { { \pmb x } } \| ^ { 2 } } { K _ { 1 } } , \quad \forall k \in [ K _ { 1 } ] ,\tag{S.101}
$$

• (minority collapse) for $\forall j = K _ { 1 } + 1 , K _ { 1 } + 2 , \cdot \cdot \cdot , K$ , if $\| \tilde { \pmb { x } } \| \neq 0$

$$
{ \pmb w } _ { j } = - \frac { \tilde { \pmb x } } { \lVert \tilde { \pmb x } \rVert } , \quad ,
$$

and for $\forall k \in [ K _ { 1 } ] , \ \forall j = K _ { 1 } + 1 , K _ { 1 } + 2 , \cdot \cdot \cdot , K ,$

$$
\begin{array} { r } { \pmb { w } _ { j } ^ { \top } \bar { \pmb { x } } _ { k } = - \| \tilde { \pmb { x } } \| , } \end{array}
$$

$$
{ \pmb w } _ { j } ^ { \top } { \pmb w } _ { k } = - \frac { c _ { 2 } } { c _ { 3 } } \frac { \| \tilde { \pmb x } \| } { K _ { 1 } } ,
$$

and $\exists b ^ { \star } , b ^ { \dagger }$ , such that

$$
\begin{array} { r l } & { b _ { k } = b ^ { \star } \in \mathbb { R } , \quad \forall k \in [ K _ { 1 } ] , } \\ & { b _ { j } = b ^ { \dagger } \in \mathbb { R } , \quad K _ { 1 } < j \leq K . } \end{array}
$$

As the CE loss $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ approaching its minimum value, $b ^ { \dagger } - b ^ { \star } \to + \infty , q ^ { \star } \to 1$ ， and $\| \tilde { \pmb x } \|  0$

Proof According to Lemma S.7, for any $c _ { 1 } , c _ { 2 } > 0$ and any $\pmb { X } ^ { ( k ) } \in \mathcal { D } _ { k }$ with $\forall k \in [ K ]$

$$
\begin{array} { r l } & { { \cal L } _ { \mathrm { c e } } ( { \pmb X } ^ { ( k ) } ) \geq \displaystyle \frac { c _ { 1 } } { 1 + c _ { 1 } + c _ { 2 } } \sum _ { j = 1 } ^ { K _ { 1 } } \frac { \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K _ { 1 } - 1 } } \\ & { \qquad + \displaystyle \frac { c _ { 2 } } { 1 + c _ { 1 } + c _ { 2 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } \frac { \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K - K _ { 1 } } + C , } \end{array}
$$

where

$$
\begin{array} { l } { \displaystyle { C = \frac { 1 } { 1 + c _ { 1 } + c _ { 2 } } \log \big ( 1 + c _ { 1 } + c _ { 2 } \big ) + \frac { c _ { 1 } } { 1 + c _ { 1 } + c _ { 2 } } \log \bigg ( \frac { 1 + c _ { 1 } + c _ { 2 } } { c _ { 1 } } ( K _ { 1 } - 1 ) \bigg ) } } \\ { \displaystyle { + \frac { c _ { 2 } } { 1 + c _ { 1 } + c _ { 2 } } \log \bigg ( \frac { 1 + c _ { 1 } + c _ { 2 } } { c _ { 2 } } ( K - K _ { 1 } ) \bigg ) , } } \end{array}
$$

and the inequality becomes an equality if and only if Eqs. S.131, S.132, S.133, and S.134 hold.

We take the same $c _ { 1 } , c _ { 2 }$ for all $\pmb { X } ^ { ( k ) } \in \mathcal { D }$ , then one can deduce the following lower bound for the function $f _ { \mathrm { c e - } }$ os

$$
\begin{array} { r l } & { \quad _ { \le , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm , \pm } } \\ &  \quad \le \frac { 1 } { \sqrt { \pi } } \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { \pi } } { \sqrt { \pi } } [ \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt } { \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } { \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } { \sqrt \pi } \frac { } { \sqrt } { \sqrt \pi } \frac { \sqrt { } \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } \frac { \sqrt { } \sqrt { \pi } } \frac { \sqrt { } \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } \frac { \sqrt { \pi } } { \sqrt } \frac  \sqrt \end{array}
$$

where $\begin{array} { r } { \tilde { \pmb { x } } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \bar { \pmb { x } } _ { j } } \end{array}$ and $\begin{array} { r } { \pmb { \bar { x } } _ { j } = \frac { 1 } { n _ { i } } \sum _ { \pmb { x } ^ { ( j ) } \in \mathcal { F } _ { j } } \pmb { x } ^ { ( j ) } , \forall j \in [ K _ { 1 } ] } \end{array}$

(S.102)

According to the AM-GM inequality (see Lemma S.2), we have

$$
u ^ { \top } v \geq - \Big ( \frac { c } { 2 } \| u \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c } \| v \| _ { 2 } ^ { 2 } \Big ) , \quad \forall u , v \in \mathbb { R } ^ { d } , \ \forall c > 0 ,
$$

which becomes an equality if and only if $\mathbf { \nabla } \pmb { v } = - c \pmb { u }$ , and

$$
\begin{array} { r } { \pmb { u } ^ { \top } \pmb { v } \geq - \| \pmb { u } \| \| \pmb { v } \| = - c \| \pmb { v } \| ^ { 2 } , \quad \forall \pmb { u } , \pmb { v } \in \mathbb { R } ^ { d } , \forall c > 0 , } \end{array}
$$

which becomes an equality if and only if $\mathbf { } u = - c \mathbf { } v$ . Then,

$$
\begin{array} { r l }  \sum _ { j = 1 } ^ { N } \operatorname* { m a x } _ { j = 1 } ^ { N } \sum _ { \ell = 0 } ^ { N } \sum _ { \ell = 1 } ^ { N } \sum _ { \ell = 1 } ^ { N } \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _ { \ell = 1 } ^ { N } \Big | \sum _  \ell  \end{array}
$$

where the second inequality becomes an equality if and only if

$$
\begin{array} { l } { - c _ { 3 } w _ { k } \ = \displaystyle \frac { c _ { 1 } } { K _ { 1 } - 1 } \tilde { \pmb { x } } - \bigg ( \displaystyle \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \bigg ) \bar { \pmb { x } } _ { k } , \quad \forall c _ { 3 } > 0 , \ \forall k \in [ K _ { 1 } ] , } \\ { - c _ { 4 } w _ { j } \ = \pmb { \tilde { x } } , \quad \forall c _ { 4 } > 0 , \ K _ { 1 } < j \leq K , } \end{array}\tag{S.103}
$$

(S.104)

and the third inequality becomes an equality if and only if

$$
1 = \| { \bar { \boldsymbol { x } } } _ { k } \| = \left\| { \frac { 1 } { n _ { k } } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \pmb { x } ^ { ( k ) } \right\| \leq { \frac { 1 } { n _ { k } } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \| \pmb { x } ^ { ( k ) } \| = 1 = \| \pmb { x } ^ { ( k ) } \|
$$

$$
\Leftrightarrow \quad \pmb { x } ^ { ( k ) } = \bar { \pmb { x } } _ { k } , \quad \forall \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K _ { 1 } ] .\tag{S.105}
$$

As $\| \pmb { w } _ { j } \| = 1$ , from Eq. (S.104), one can deduce

$$
c _ { 4 } = \| \tilde { \pmb { x } } \| .\tag{S.106}
$$

Then,

$$
\begin{array} { r l } & { f _ { e x e } \big ( W , b , \mathcal { F } \big ) } \\ & { \geq - \frac { \gamma } { 1 + \epsilon _ { 1 } + \epsilon _ { 2 } } \Bigg [ \Bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } \Bigg ) ^ { 2 } K _ { 1 } \| \epsilon \| ^ { 2 } - 2 \Bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } \Bigg ) \Bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \Bigg ) K _ { 1 } \| \bar { x } \| ^ { 2 } } \\ & { \qquad + \Bigg ( \frac { \epsilon _ { \Delta } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \Bigg ) ^ { 2 } K _ { 1 } \Bigg ] - \frac { c _ { 2 } \gamma } { 1 + \epsilon _ { 1 } + c _ { 2 } } \| \bar { x } \| } \\ & { \qquad + \frac { \sigma } { 1 + c _ { 1 } + \epsilon _ { 2 } } \Bigg ( \frac { 1 } { K _ { 1 } } \underset { k = 1 } { \overset { \Sigma _ { b } } { \sum } } - \frac { 1 } { K _ { 1 } - K _ { 1 } } \underset { j = K _ { 1 } , 1 } { \overset { \Sigma _ { b } } { \sum } } \Bigg ) + C } \\ & { = - \frac { \gamma } { 1 + c _ { 1 } + \epsilon _ { 2 } } \Bigg \{ \Bigg [ \frac { K _ { 1 } } { c _ { 3 } } \bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } \bigg ) ^ { 2 } \underset { ( * ) } { \overset { \mathrm { 2 } } { \sum } } \frac { 2 K _ { 1 } } { K _ { 1 } - 1 } \underset { k = 1 } { \overset { \Sigma _ { b } } { \sum } } \Bigg ] \| \bar { x } \| ^ { 2 } - c _ { 2 } \| \bar { x } \| \Bigg \} + C } \\ &  \qquad - \frac { \gamma } { 1 + c _ { 1 } + \epsilon _ { 2 } } \epsilon _ { S } \epsilon _ { S } \bigg ( \frac { c _ { 1 } }  K _ { 1 }  \end{array}
$$

In summary, $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ achieves the lower bound if and only if Eqs. (S.131, S.132, S.133, S.134) and Eqs. (S.103, S.104, S.105) hold.

Finally, Lemma S.12 completes the proof.

Lemma S.12 Suppose that the first $K _ { 1 }$ classes in dataset D contains at least one sample, $K _ { 1 } < K$ , while the other classes are empty, and that the feature dimension is greater than the number of classes, i.e., $n _ { k } = | \mathcal D _ { k } | > 0$ for $\forall k \in [ K _ { 1 } ] , n _ { k } = | \mathcal { D } _ { k } | = 0$ for $K _ { 1 } < k \le K$ and $d \geq K _ { 1 }$ . Then, for the CE loss $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ defined in Eq. (S.98),

$$
\begin{array} { r } { f _ { \mathrm { c e } } ( W , b , \mathcal { F } ) \geq \log \big ( 1 + c _ { 1 } + c _ { 2 } \big ) , } \end{array}\tag{S.107}
$$

where

$$
c _ { 1 } \ = ( K _ { 1 } - 1 ) \exp \bigg ( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { { \boldsymbol x } } \| ^ { 2 } - 1 \big ) \bigg )
$$

$$
c _ { 2 } \ = K _ { 1 } \frac { 1 - q ^ { \star } } { q ^ { \star } } \exp \left( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { { \boldsymbol x } } \| ^ { 2 } - 1 \big ) \right)
$$

$$
c _ { 3 } ~ = \frac { 1 } { q ^ { \star } } \sqrt { g ( q ^ { \star } ) } \exp { \left( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( \| \tilde { x } \| ^ { 2 } - 1 \big ) \right) } ,
$$

and $g ( q ) = 1 - 2 q \| \tilde { \pmb { x } } \| ^ { 2 } + q ^ { 2 } \| \tilde { \pmb { x } } \| ^ { 2 }$ with a certain $q ^ { \star } \in ( 0 , 1 )$

When $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ approaching its minimum value, $q ^ { \star } \to 1 \ a n d \ \| \tilde { \mathbf { x } } \| \to 0$

Proof According to Theorem S.11, the CE loss $f _ { \mathrm { c e } } ( W , b , \mathcal { F } )$ satisfies

$$
\begin{array} { l } { \displaystyle f _ { \mathrm { c e - o s } } ( { \pmb W } , { \pmb b } , { \mathcal F } ) \geq - \frac { \gamma } { 1 + c _ { 1 } + c _ { 2 } } \frac { c _ { 2 } } { 2 } \| { \tilde { \pmb x } } \| - \frac { \gamma } { 1 + c _ { 1 } + c _ { 2 } } \frac { K _ { 1 } } { c _ { 3 } } \bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \bigg ) ^ { 2 } } \\ { \displaystyle + \frac { c _ { 2 } } { 1 + c _ { 1 } + c _ { 2 } } \bigg ( \frac { 1 } { K _ { 1 } } \sum _ { k = 1 } ^ { K _ { 1 } } b _ { k } - \frac { 1 } { K - K _ { 1 } } \sum _ { j = K _ { 1 } + 1 } ^ { K } b _ { j } \bigg ) + C , } \end{array}\tag{S.108}
$$

and the loss achieves this lower boundary if and only if Eqs. (S.131, S.132, S.133, S.134) and Eqs. (S.103, S.104, S.105) hold.

From Eq. (S.103),

$$
\begin{array} { r l } { \gamma _ { 1 } = \gamma _ { 2 } - \frac { \lambda _ { 1 } } { \mu _ { 2 } \mu _ { 3 } } - 1 \frac { \lambda _ { 2 } } { \mu _ { 3 } \mu _ { 4 } } \frac { \lambda _ { 2 } } { \mu _ { 5 } \mu _ { 5 } } } & { = } & { 0 } \\ { \gamma _ { 2 } = \frac { \lambda _ { 1 } } { \mu _ { 5 } } - \frac { \lambda _ { 2 } } { \mu _ { 6 } \mu _ { 5 } } - \frac { \lambda _ { 3 } } { \mu _ { 6 } \mu _ { 6 } } - \frac { \lambda _ { 3 } } { \mu _ { 6 } \mu _ { 7 } } } & { = } & { 0 } \\ { \gamma _ { 3 } = \frac { \lambda _ { 1 } } { \mu _ { 5 } } - \frac { \lambda _ { 2 } } { \mu _ { 6 } \mu _ { 5 } } - \frac { \lambda _ { 3 } } { \mu _ { 6 } \mu _ { 7 } } - \frac { \lambda _ { 3 } } { \mu _ { 6 } \mu _ { 6 } } } & { = } & { 0 } \\ { - \frac { \lambda _ { 2 } } { \mu _ { 6 } \mu _ { 7 } } - 1 \frac { \lambda _ { 3 } } { \mu _ { 6 } \mu _ { 7 } } - \frac { \lambda _ { 3 } } { \mu _ { 6 } \mu _ { 8 } } } & { = } & { 0 } \\ { \gamma _ { 4 } = \frac { \lambda _ { 1 } } { \mu _ { 7 } } - \frac { \lambda _ { 2 } } { \mu _ { 7 } \mu _ { 7 } } - 1 \frac { \lambda _ { 3 } } { \mu _ { 8 } \mu _ { 8 } } } & { = } & { 0 } \\ { \gamma _ { 5 } = \frac { \lambda _ { 2 } } { \mu _ { 8 } } - \frac { \lambda _ { 3 } } { \mu _ { 8 } \mu _ { 7 } } - \frac { \lambda _ { 3 } } { \mu _ { 8 } \mu _ { 8 } } } & { = } & { 0 } \\  \gamma _ { 6 } = \frac { \lambda _ { 3 } } { \mu _ { 8 } } - \frac  \lambda _ \end{array}\tag{S.109}
$$

(S.110)

(S.111)

$$
\begin{array} { r l r } & { } & { \stackrel { S . 1 1 1 } { = } 2 \bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \bigg ) ^ { 2 } - 2 \bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \bigg ) \frac { c _ { 1 } } { K _ { 1 } - 1 } \| \tilde { x } \| ^ { 2 } } \\ & { \Rightarrow } & { w _ { k } ^ { \top } \bar { x } _ { k } = \bigg ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \bigg ) \frac { 1 } { c _ { 3 } } - \frac { c _ { 1 } } { K _ { 1 } - 1 } \frac { 1 } { c _ { 3 } } \| \tilde { x } \| ^ { 2 } , \quad \forall k \in [ K _ { 1 } ] . } \end{array}\tag{S.112}
$$

Furthermore,

$$
\begin{array} { r l } & { \quad \le - \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } } \\ & { \quad + \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } } \\ & { \quad \le - \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } } \\ & { \quad \le - \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } } \\ & { \quad \le - \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } } \\ & { \quad \le - \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } } \\ &  \quad \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac { \sqrt { 3 } } { 2 } \le \frac  \end{array}\tag{S.113}
$$

(S.114)

(S.115)

(S.116)

According to Eqs. (S.104,S.106),

$$
{ \pmb w } _ { j } ^ { \top } \bar { \pmb x } _ { k } = - \| \tilde { \pmb x } \| , \quad \forall k \in [ K _ { 1 } ] , \ K _ { 1 } < j \leq K ;\tag{S.117}
$$

according to Eqs. (S.131, S.116), we denote

$$
\begin{array} { r } { b ^ { \star } = b _ { k } = b _ { j } , \quad \forall j \neq k \in [ K _ { 1 } ] ; } \end{array}\tag{S.118}
$$

according to Eqs. (S.132, S.117), we denote

$$
b ^ { \dagger } = b _ { j } = b _ { \ell } , \quad \forall K _ { 1 } < j , \ \ell \leq K ;\tag{S.119}
$$

according to Eqs. (S.116, S.115),

$$
\log \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } \right) = \frac { \gamma } { c _ { 3 } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \right) \big ( \| \tilde { \pmb x } \| ^ { 2 } - 1 \big ) ;\tag{S.120}
$$

according to Eqs. (S.117,S.112),

$$
\log \left( \frac { c _ { 2 } } { K - K _ { 1 } } \right) = - \gamma \| \tilde { x } \| - \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \right) \frac { \gamma } { c _ { 3 } } + \frac { c _ { 1 } } { K _ { 1 } - 1 } \frac { \gamma } { c _ { 3 } } \| \tilde { x } \| ^ { 2 } + b ^ { \star } - b ^ { \dagger } .\tag{S.121}
$$

In total, when the CE loss $f _ { \mathrm { c e - o s } } ( W , b , \mathcal { F } )$ reaching its lower bound in Eq. (S.108), $c _ { 1 } , c _ { 2 } , c _ { 3 }$ satisfy Eqs. (S.111, S.120, S.121). According to Lemma S.14, for any given $\gamma > 0$ $K > K _ { 1 } > 1 , b ^ { \star } , b ^ { \dagger } \in \mathbb { R }$ , and $\| \tilde { \pmb { x } } \| \in ( 0 , 1 )$ , there exists a unique set of $c _ { 1 } , c _ { 2 } , c _ { 3 }$ . Then, they can be expressed as functions of $x = \| \tilde { \pmb { x } } \| , \delta _ { b } = b ^ { \star } - b ^ { \dagger }$ and $\gamma , K , K _ { 1 }$ . Therefore,

$$
\begin{array} { r l } & { \frac { \partial } { \partial x } \widetilde { \mathbf { f } } _ { \mathbf { f } } ( \mathbf { f } , \mathbf { f } ) } \\ & { = \frac { 1 } { 2 } \frac { \partial } { \partial x } [ ( \frac { \mathbf { f } _ { \mathbf { f } } } { \mathbf { f } _ { \mathbf { f } } } ( \frac { \mathbf { f } _ { \mathbf { f } } } { \mathbf { f } _ { \mathbf { f } } } ) ) ^ { 2 } - \frac { \mathbf { f } _ { \mathbf { f } } } { \mathbf { f } _ { \mathbf { f } } } \frac { \partial } { \mathbf { f } _ { \mathbf { f } } } \frac { \partial } { \mathbf { f } _ { \mathbf { f } } } \frac { \partial } { \mathbf { f } _ { \mathbf { f } } }  \mathbf { f } _ { \mathbf { f } } ^ { \prime } - z _ { \parallel }  ] - C } \\ & { \qquad \times \frac { 1 } { 2 } ( [ \frac { \mathbf { f } _ { \mathbf { f } } } { \mathbf { f } _ { \mathbf { f } } } ( \frac { \mathbf { f } _ { \mathbf { f } } } { \mathbf { f } _ { \mathbf { f } } } ) ] ^ { 2 } ) ^ { 2 } } \\ & { \qquad \times \frac { 1 } { 2 } \frac { \partial } { \mathbf { f } _ { \mathbf { f } } } ( \mathbf { f } , \mathbf { f } ) \cdot \mathbf { f } _ { \mathbf { f } } ( \mathbf { f } ) ^ { 2 } , \qquad \mathrm { i f ~ } \frac { \partial } { \mathbf { f } _ { \mathbf { f } } } ( \mathbf { f } , \mathbf { f } ) \cdot \frac { \mathbf { f } _ { \mathbf { f } } } { \mathbf { f } _ { \mathbf { f } } ( \mathbf { f } _ { \mathbf { f } } ) } \cdot \mathbf { f } _ { \mathbf { f } } ( \mathbf { f } , \mathbf { f } ) } \\ &  = \frac { 1 } { 1 + \mathbf { f } _ { \mathbf { f } } } \frac { \partial } { \mathbf { f } _ { \mathbf { f } } } \{ [ \mathbf { f } , \mathbf { f } ] \cdot \mathbf  \end{array}
$$

$$
\begin{array} { r l } & { \quad + \log \left( 1 + c _ { 1 } + c _ { 2 } \right) - \frac { c _ { 1 } } { 1 + c _ { 1 } + c _ { 2 } } \log \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } \right) - \frac { c _ { 2 } } { 1 + c _ { 1 } + c _ { 2 } } \log \left( \frac { c _ { 2 } } { K - K _ { 1 } } \right) } \\ & { = \log \left( 1 + c _ { 1 } + c _ { 2 } \right) = \log \left( 1 + c _ { 1 } \big ( x , \delta _ { b } ; \gamma , K , K _ { 1 } \big ) + c _ { 2 } \big ( x , \delta _ { b } ; \gamma , K , K _ { 1 } \big ) \right) } \\ & { \overset { \Delta } { = } \int _ { \mathrm { c e - o s } } ^ { ( 1 ) } ( x , \delta _ { b } ; \gamma , K , K _ { 1 } ) . } \end{array}
$$

According to Lemma S.13, $\frac { \partial \hat { f } _ { \mathrm { c e - o s } } ^ { ( 1 ) } ( x , \delta _ { b } ; \gamma , K , K _ { 1 } ) } { \partial \delta _ { b } } ~ > ~ 0$ , therefore, as $\hat { f } _ { \mathrm { c e - o s } } ^ { ( 1 ) } ( x , \delta _ { b } ; \gamma , K , K _ { 1 } )$ approaches its minimum,

$$
\delta _ { b } = b ^ { \star } - b ^ { \dagger } \to - \infty .
$$

According to Eq. (S.125), $q ^ { \star } \ \to \ 1$ , as $\delta _ { b } \quad \to \quad - \infty$ , and then according to $\operatorname { E q } .$ (S.126,S.127,S.128),

$$
c _ { 1 } ~  ( K _ { 1 } - 1 ) \exp \Big ( - \gamma \frac { K _ { 1 } } { K _ { 1 } - 1 } \sqrt { 1 - \| \tilde { { \pmb x } } \| ^ { 2 } } \Big ) ,
$$

$$
c _ { 2 } \ \to 0 ,
$$

$$
c _ { 3 } ~  \sqrt { 1 - \| \tilde { { \pmb x } } \| ^ { 2 } } \exp \Big ( - \gamma \frac { K _ { 1 } } { K _ { 1 } - 1 } \sqrt { 1 - \| \tilde { { \pmb x } } \| ^ { 2 } } \Big ) .
$$

Therefore,

$$
\begin{array} { r l } { f _ { \mathrm { c e - o s } } ( W , \boldsymbol { b } , \mathcal { F } ) } & { \geq \hat { f } _ { \mathrm { c e - o s } } ^ { ( 1 ) } ( \boldsymbol { x } , \delta _ { \boldsymbol { b } } ; \gamma , K , K _ { 1 } ) = \log \big ( 1 + c _ { 1 } + c _ { 2 } \big ) } \\ & { \geq \log \big ( 1 + c _ { 1 } \big ) = \log \Big ( 1 + ( K _ { 1 } - 1 ) \exp \Big ( - \gamma \frac { K _ { 1 } } { K _ { 1 } - 1 } \sqrt { 1 - \| \tilde { \boldsymbol { x } } \| ^ { 2 } } \Big ) \Big ) } \\ & { \triangleq \hat { f } _ { \mathrm { c e - o s } } ^ { ( 2 ) } ( \| \tilde { \boldsymbol { x } } \| ; \gamma , K _ { 1 } ) . } \end{array}
$$

One can easily deduce that $\frac { \partial \hat { f } _ { \mathrm { c e - o s } } ^ { ( 2 ) } ( \lVert \tilde { \pmb x } \rVert ; \gamma , K _ { 1 } ) } { \partial \lVert \tilde { \pmb x } \rVert } > 0$ , then when $\hat { f } _ { \mathrm { c e - o s } } ^ { ( 2 ) } ( \| \tilde { \pmb { x } } \| ; \gamma , K _ { 1 } )$ approaching its minimum value, $\| \tilde { \pmb x } \|  0$

Lemma S.13 Suppose $\gamma > 0 , K > K _ { 1 } > 1 , \delta _ { b } \in \mathbb { R } , x \in ( 0 , 1 )$ , and

$$
c _ { 3 } ^ { 2 } = \left( { \frac { c _ { 1 } } { K _ { 1 } - 1 } } + { \frac { c _ { 2 } } { K _ { 1 } } } \right) ^ { 2 } - \left[ \left( { \frac { c _ { 1 } } { K _ { 1 } - 1 } } + { \frac { c _ { 2 } } { K _ { 1 } } } \right) { \frac { c _ { 1 } } { K _ { 1 } - 1 } } + { \frac { c _ { 1 } } { K _ { 1 } - 1 } } { \frac { c _ { 2 } } { K _ { 1 } } } \right] x ^ { 2 }
$$

$$
\log \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } \right) = \frac { \gamma } { c _ { 3 } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \Biggl ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \Biggr ) \left( x ^ { 2 } - 1 \right)
$$

$$
\log { \left( \frac { c _ { 2 } } { K - K _ { 1 } } \right) } = - \gamma x - \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \right) \frac { \gamma } { c _ { 3 } } + \frac { c _ { 1 } } { K _ { 1 } - 1 } \frac { \gamma } { c _ { 3 } } x ^ { 2 } + \delta _ { b } .
$$

Then, for the function $\begin{array} { r } { \hat { f } _ { c e - o s } ^ { ( 1 ) } ( x , \delta _ { b } ; \gamma , K , K _ { 1 } ) = \log ( 1 + c _ { 1 } + c _ { 2 } ) } \end{array}$

$$
\frac { \partial \hat { f } _ { c e - o s } ^ { ( 1 ) } ( x , \delta _ { b } ; \gamma , K , K _ { 1 } ) } { \partial \delta _ { b } } = \frac { 1 } { 1 + c _ { 1 } + c _ { 2 } } \bigg ( \frac { \partial c _ { 1 } } { \partial \delta _ { b } } + \frac { \partial c _ { 2 } } { \partial \delta _ { b } } \bigg ) > 0 .
$$

Proof Denote $\begin{array} { r } { u = \frac { c _ { 1 } } { K _ { 1 } - 1 } , v = \frac { c _ { 2 } } { K _ { 1 } } } \end{array}$ and $\begin{array} { r } { s = u + v , q = \frac { u } { u + v } \in ( 0 , 1 ) } \end{array}$ . According to Lemma S.14,

$$
c _ { 3 } = s { \sqrt { g ( q , x ) } } ,
$$

$$
c _ { 1 } = ( K _ { 1 } - 1 ) \exp \big ( - \phi ( q , x ) \big ) > 0 ,
$$

$$
c _ { 2 } = K _ { 1 } \frac { 1 - q } { q } \exp \big ( - \phi ( q , x ) \big ) > 0 ,
$$

where

$$
g ( q , x ) = 1 - 2 x ^ { 2 } q + x ^ { 2 } q ^ { 2 } ,
$$

$$
\phi ( q , x ) = \gamma \frac { K _ { 1 } } { K _ { 1 } - 1 } \frac { 1 - x ^ { 2 } } { \sqrt { g ( q , x ) } } .
$$

Let

$$
\begin{array} { r l } { \delta _ { b } } & { = \log \left( \cfrac { c _ { 2 } } { K - K _ { 1 } } \right) + \gamma x + \left( \cfrac { c _ { 1 } } { K _ { 1 } - 1 } + \cfrac { c _ { 2 } } { K _ { 1 } } \right) \cfrac { \gamma } { c _ { 3 } } - \cfrac { c _ { 1 } } { K _ { 1 } - 1 } \cfrac { \gamma } { c _ { 3 } } x ^ { 2 } } \\ & { = \log \left( \cfrac { c _ { 2 } } { K _ { 1 } } \right) + \log \left( \cfrac { K _ { 1 } } { K - K _ { 1 } } \right) + \gamma x + s \frac { \gamma } { s \sqrt { g ( q , x ) } } - u \frac { \gamma } { s \sqrt { g ( q , x ) } } x ^ { 2 } } \\ & { = \log \left( \cfrac { 1 - q } { q } \right) + \log \left( \cfrac { K _ { 1 } } { K - K _ { 1 } } \right) + \gamma x - \gamma T ( q , x ) } \\ & { \triangleq B ( q , x ) , } \end{array}
$$

where

$$
T ( q , x ) = \frac { 1 } { \sqrt { g ( q , x ) } } \biggl [ \frac { K _ { 1 } } { K _ { 1 } - 1 } ( 1 - x ^ { 2 } ) - 1 + q x ^ { 2 } \biggr ] .
$$

According to Lemma S.14, given $\gamma , K , K _ { 1 } , q$ is a function in terms of $x$ and $\delta _ { b }$ . Denote it as $q ( x , \delta _ { b } )$ . Then,

$$
\frac { \partial B ( q , x ) } { \partial q } \frac { \partial q ( x , \delta _ { b } ) } { \partial \delta _ { b } } = 1 .
$$

One can deduce that

$$
\frac { \partial B ( q , x ) } { \partial q } = - \frac { 1 } { 1 - q } - \frac { 1 } { q } - \gamma \frac { \partial T ( q , x ) } { \partial q } = - \frac { 1 } { 1 - q } - \frac { 1 } { q } - \gamma \frac { x ^ { 2 } ( 1 - x ^ { 2 } ) ( K _ { 1 } - q ) } { ( K _ { 1 } - 1 ) \left( g ( q , x ) ^ { 3 / 2 } \right) } < 0 ;
$$

therefore, $\frac { \partial q ( x , \delta _ { b } ) } { \partial \delta _ { b } } < 0$ . As

$$
\frac { \partial \phi ( q , x ) } { \partial q } = \frac { 1 } { \left( g ( q , x ) \right) ^ { 3 / 2 } } \Biggl ( \gamma \frac { K _ { 1 } } { K _ { 1 } - 1 } ( 1 - x ^ { 2 } ) x ^ { 2 } ( 1 - q ) \Biggr ) > 0 ,
$$

and

$$
\begin{array} { l c l } { \displaystyle { c _ { 1 } = \big ( K _ { 1 } - 1 \big ) \mathrm { e } ^ { - \phi ( q , x ) } } } & { \displaystyle { \Rightarrow } } & { \log ( c _ { 1 } ) = \log ( K _ { 1 } - 1 ) - \phi ( q , x ) } \\ { \displaystyle } & { \displaystyle { \Rightarrow } } & { \displaystyle { \frac { \partial c _ { 1 } } { \partial \delta _ { b } } \frac { 1 } { c _ { 1 } } = - \frac { \partial \phi ( q , x ) } { \partial q } \frac { \partial q } { \partial \delta _ { b } } } } \\ { \displaystyle } & { \displaystyle { \Rightarrow } } & { \displaystyle { \frac { \partial c _ { 1 } } { \partial \delta _ { b } } = - c _ { 1 } \frac { \partial \phi ( q , x ) } { \partial q } \frac { \partial q } { \partial \delta _ { b } } > 0 , } } \\ { \displaystyle { c _ { 2 } = K _ { 1 } \frac { 1 - q } { q } \mathrm { e } ^ { - \phi ( q , x ) } \big ) } } & { \displaystyle { \Rightarrow } } & { \displaystyle { \frac { \partial c _ { 2 } } { \partial \delta _ { b } } \frac { 1 } { c _ { 2 } } = \Big ( - \frac { 1 } { 1 - q } - \frac { 1 } { q } - \frac { \partial \phi ( q , x ) } { \partial q } \Big ) \frac { \partial q } { \partial \delta _ { b } } } } \\ { \displaystyle } & { \displaystyle { \Rightarrow } } & { \displaystyle { \frac { \partial c _ { 2 } } { \partial \delta _ { b } } = - c _ { 2 } \Big ( \frac { 1 } { 1 - q } + \frac { 1 } { q } + \frac { \partial \phi ( q , x ) } { \partial q } \Big ) \frac { \partial q } { \partial \delta _ { b } } > 0 , } } \end{array}
$$

one can deduce that

$$
\frac { \partial \hat { f } _ { \mathrm { c e - o s } } ^ { ( 1 ) } ( x , \delta _ { b } ; \gamma , K , K _ { 1 } ) } { \partial \delta _ { b } } = \frac { 1 } { 1 + c _ { 1 } + c _ { 2 } } \bigg ( \frac { \partial c _ { 1 } } { \partial \delta _ { b } } + \frac { \partial c _ { 2 } } { \partial \delta _ { b } } \bigg ) > 0 \mathrm { , }
$$

which completes the proof.

Lemma S.14 Given $\gamma > 0 , K > K _ { 1 } > 1 , \delta _ { b } \in \mathbb { R }$ , and $x \in ( 0 , 1 )$ , Eqs. (S.111, S.120, S.121), i.e.,

$$
c _ { 3 } ^ { 2 } = \left( { \frac { c _ { 1 } } { K _ { 1 } - 1 } } + { \frac { c _ { 2 } } { K _ { 1 } } } \right) ^ { 2 } - \left[ \left( { \frac { c _ { 1 } } { K _ { 1 } - 1 } } + { \frac { c _ { 2 } } { K _ { 1 } } } \right) { \frac { c _ { 1 } } { K _ { 1 } - 1 } } + { \frac { c _ { 1 } } { K _ { 1 } - 1 } } { \frac { c _ { 2 } } { K _ { 1 } } } \right] x ^ { 2 }
$$

$$
\log \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } \right) = \frac { \gamma } { c _ { 3 } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \Biggl ( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \Biggr ) \left( x ^ { 2 } - 1 \right)
$$

$$
\log { \left( \frac { c _ { 2 } } { K - K _ { 1 } } \right) } = - \gamma x - \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \right) \frac { \gamma } { c _ { 3 } } + \frac { c _ { 1 } } { K _ { 1 } - 1 } \frac { \gamma } { c _ { 3 } } x ^ { 2 } + \delta _ { b } ,
$$

determine a unique set of values for $c _ { 1 } , c _ { 2 } , c _ { 3 }$

Proof Denote

$$
u = \frac { c _ { 1 } } { K _ { 1 } - 1 } , \quad v = \frac { c _ { 2 } } { K _ { 1 } } ,
$$

then $u > 0 , v > 0 , \delta _ { b } \in \mathbb { R } , x \in ( 0 , 1 )$ and the set of equations evolves into

$$
c _ { 3 } ^ { 2 } = \left( u + v \right) ^ { 2 } - \left( u ^ { 2 } + 2 u v \right) x ^ { 2 } ,\tag{S.122}
$$

$$
\log { \left( u \right) } = \frac { \gamma } { c _ { 3 } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( u + v \big ) \big ( x ^ { 2 } - 1 \big ) ,\tag{S.123}
$$

$$
\log \left( \frac { K _ { 1 } } { K - K _ { 1 } } v \right) = - \gamma x - \frac { \gamma } { c _ { 3 } } \big ( u + v \big ) + \frac { \gamma } { c _ { 3 } } u x ^ { 2 } + \delta _ { b } .\tag{S.124}
$$

Let

$$
q = \frac { u } { u + v } \in ( 0 , 1 ) , \quad s = u + v \in \mathbb { R } ,
$$

then

$$
u = q s , \quad v = ( 1 - q ) s = \frac { 1 - q } { q } u ,
$$

and for $c _ { 3 }$ , according to Eq. (S.122),

$$
c _ { 3 } ^ { 2 } = s ^ { 2 } \big ( 1 - 2 x ^ { 2 } q + x ^ { 2 } q ^ { 2 } \big ) .
$$

Denote

$$
g ( q ) = 1 - 2 x ^ { 2 } q + x ^ { 2 } q ^ { 2 } ,
$$

then as $x \in ( 0 , 1 ) , q \in ( 0 , 1 )$ , one can deduce that $g ( q ) > 0$ . Therefore,

$$
c _ { 3 } = s { \sqrt { g ( q ) } } .
$$

According to Eq. (S.123),

$$
\begin{array} { c l c r } { { \log \left( u \right) = \displaystyle \frac { \gamma } { s \sqrt { g ( q ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } s \big ( x ^ { 2 } - 1 \big ) = \displaystyle \frac { \gamma } { \sqrt { g ( q ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( x ^ { 2 } - 1 \big ) } } \\ { { \Rightarrow } } & { { u = \displaystyle \exp \bigg ( \displaystyle \frac { \gamma } { \sqrt { g ( q ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( x ^ { 2 } - 1 \big ) \bigg ) . } } \end{array}
$$

According to Eq. (S.124)

$$
\begin{array} { l } { \displaystyle \log \bigg ( \frac { K _ { 1 } } { K - K _ { 1 } } v \bigg ) = \delta _ { b } - \gamma x - \frac { \gamma } { c _ { 3 } } \big ( s - u x ^ { 2 } \big ) } \\ { = \delta _ { b } - \gamma x - \frac { \gamma } { s \sqrt { g ( q ) } } \big ( s - q s x ^ { 2 } \big ) } \\ { = \delta _ { b } - \gamma x - \frac { \gamma } { \sqrt { g ( q ) } } \big ( 1 - q x ^ { 2 } \big ) . } \end{array}
$$

$$
\begin{array} { r } { \mathrm { A s } \ v = ( 1 - q ) s = \frac { 1 - q } { q } u , } \end{array}
$$

$$
\begin{array} { r l } & { \log \Big ( \displaystyle \frac { K _ { 1 } } { K - K _ { 1 } } \frac { 1 - q } { q } u \Big ) = \delta _ { b } - \gamma x - \displaystyle \frac { \gamma } { \sqrt { g ( q ) } } \big ( 1 - q x ^ { 2 } \big ) } \\ { \Rightarrow } & { \underbrace { \log \frac { 1 - q } { q } } _ { \alpha ( q ) } = \underbrace { \delta _ { b } - \gamma x - \displaystyle \frac { \gamma } { \sqrt { g ( q ) } } \bigg ( \frac { K _ { 1 } } { K _ { 1 } - 1 } ( x ^ { 2 } - 1 ) + 1 - q x ^ { 2 } \bigg ) - \log \Big ( \frac { K _ { 1 } } { K - K _ { 1 } } \Big ) } _ { \beta ( q ) } . } \end{array}\tag{S.125}
$$

When $q \to 0 ^ { + }$ 2

$$
\alpha ( q )  + \infty , \quad \beta ( q )  \delta _ { b } - \gamma x - \gamma \biggl ( \frac { K _ { 1 } } { K _ { 1 } - 1 } \bigl ( x ^ { 2 } - 1 \bigr ) + 1 \biggr ) - \log \biggl ( \frac { K _ { 1 } } { K - K _ { 1 } } \biggr ) ,
$$

and when $q \to 1$ <sup>−</sup>,

$$
\alpha ( q )  - \infty , \quad \beta ( q )  \delta _ { b } - \gamma x + \gamma \frac { \sqrt { 1 - x ^ { 2 } } } { K _ { 1 } - 1 } - \log { ( \frac { K _ { 1 } } { K - K _ { 1 } } ) } ,
$$

which indicate that $\alpha ( q )$ and $\beta ( q )$ must intersect. Meanwhile

$$
\begin{array} { r l } & { \frac { \mathrm { d } \alpha ( q ) } { \mathrm { d } q } = - \cfrac { 1 } { 1 - q } \cfrac { 1 } { q } < 0 , \quad \forall q \in ( 0 , 1 ) , } \\ & { \cfrac { \mathrm { d } \beta ( q ) } { \mathrm { d } q } = - \gamma \cfrac { - x ^ { 2 } \sqrt { g ( q ) } - \Big ( \cfrac { K _ { 1 } } { K _ { 1 } - 1 } \big ( x ^ { 2 } - 1 \big ) + 1 - q x ^ { 2 } \Big ) \mathrm { d } \big ( \sqrt { g ( q ) } \big ) / \mathrm { d } q } { g ( q ) } } \\ & { \quad \quad \quad = \cfrac { \gamma x ^ { 2 } ( 1 - x ^ { 2 } ) ( K _ { 1 } - q ) } { \big ( K _ { 1 } - 1 \big ) \big ( g ( q ) \big ) ^ { 3 / 2 } } > 0 , \quad \forall q \in ( 0 , 1 ) . } \end{array}
$$

Therefore, there exists only one $q ^ { \star } \in ( 0 , 1 )$ such that $\alpha ( q ^ { \star } ) = \beta ( q ^ { \star } )$ , and

$$
u = \exp \Bigg ( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( x ^ { 2 } - 1 \big ) \Bigg ) \ \Rightarrow c _ { 1 } = \big ( K _ { 1 } - 1 \big ) \exp \Bigg ( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( x ^ { 2 } - 1 \big ) \Bigg ) ,\tag{S.126}
$$

$$
v = { \frac { 1 - q ^ { \star } } { q ^ { \star } } } u \ \Rightarrow c _ { 2 } = K _ { 1 } { \frac { 1 - q ^ { \star } } { q ^ { \star } } } \exp \Bigg ( { \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } } { \frac { K _ { 1 } } { K _ { 1 } - 1 } } \big ( x ^ { 2 } - 1 \big ) \Bigg ) ,\tag{S.127}
$$

$$
\begin{array} { l } { { \displaystyle c _ { 3 } = \left( \frac { c _ { 1 } } { K _ { 1 } - 1 } + \frac { c _ { 2 } } { K _ { 1 } } \right) \sqrt { g ( q ^ { \star } ) } } } \\ { { \displaystyle = \frac { 1 } { q ^ { \star } } \sqrt { g ( q ^ { \star } ) } \exp \left( \frac { \gamma } { \sqrt { g ( q ^ { \star } ) } } \frac { K _ { 1 } } { K _ { 1 } - 1 } \big ( x ^ { 2 } - 1 \big ) \right) , } } \end{array}\tag{S.128}
$$

which completes the proof.

Lemma S.15 For any $\pmb { X } ^ { ( k ) } \in \mathcal { D }$ with any $c _ { 1 } , c _ { 2 } > 0$ and $\gamma > 0$ , the CE loss $L _ { \mathrm { c e } } ( \mathbf { X } ^ { ( k ) } )$ defined in $E q . \ ( 2 )$ is lower bounded by

$$
\begin{array} { r l } & { L _ { \mathrm { c e } } \big ( { \pmb X } ^ { ( k ) } \big ) \geq \frac { c _ { 1 } } { 1 + c _ { 1 } + c _ { 2 } } \displaystyle \sum _ { j \neq k } ^ { K _ { 1 } } \frac { \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K _ { 1 } - 1 } } \\ & { \qquad + \frac { c _ { 2 } } { 1 + c _ { 1 } + c _ { 2 } } \displaystyle \sum _ { j = K _ { 1 } + 1 } ^ { K } \frac { \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K - K _ { 1 } } + C , } \end{array}\tag{S.129}
$$

where

$$
\begin{array} { l } { { \displaystyle { C = \frac { 1 } { 1 + c _ { 1 } + c _ { 2 } } \log \left( 1 + c _ { 1 } + c _ { 2 } \right) + \frac { c _ { 1 } } { 1 + c _ { 1 } + c _ { 2 } } \log \left( \frac { 1 + c _ { 1 } + c _ { 2 } } { c _ { 1 } } ( K _ { 1 } - 1 ) \right) } } } \\ { { \displaystyle { ~ + \frac { c _ { 2 } } { 1 + c _ { 1 } + c _ { 2 } } \log \left( \frac { 1 + c _ { 1 } + c _ { 2 } } { c _ { 2 } } ( K - K _ { 1 } ) \right) } . } } \end{array}\tag{S.130}
$$

The inequality becomes an equality if and only if

$$
\gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } = \gamma \pmb { w } _ { \ell } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { \ell } , \quad \forall j , \ell \neq k \in [ K _ { 1 } ] ,\tag{S.131}
$$

$$
\gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } = \gamma { \pmb w } _ { \ell } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { \ell } , \quad \forall k \in [ K _ { 1 } ] , ~ K _ { 1 } < j , \ell \leq K , ,\tag{S.132}
$$

and

$$
c _ { 1 } = ( K _ { 1 } - 1 ) \exp \Bigg ( \sum _ { j = 1 } ^ { K _ { 1 } } \frac { \gamma { \pmb w } _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K - 1 } \Bigg ) ,\tag{S.133}
$$

$$
c _ { 2 } = ( K - K _ { 1 } ) \exp \Bigg ( \sum _ { j = K _ { 1 } + 1 } ^ { K } \frac { \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K - K _ { 1 } } \Bigg ) .\tag{S.134}
$$

Proof According to Jensen’s inequality in Lemma S.3, the CE loss $L _ { \mathrm { c e } } ( \mathbf { X } ^ { ( k ) } )$ can be lower bounded,

$$
\begin{array} { r l } &  \mathbb { E } _ { \sigma } ( X ^ { \varepsilon } ) = \frac { 1 } { N } \left( 1 + \frac { N } { 2 \sqrt { N } } \frac { \varepsilon } { N } \right) ^ { 2 } \left( 1 - \frac { N } { 2 \sqrt { N } } \frac { \varepsilon } { N } \right) ^ { 2 } \frac { \varepsilon } { N } \Bigg ( \varepsilon \sigma _ { \varepsilon } ^ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^ { \varepsilon } \sigma _ { \varepsilon } ^  \varepsilon \end{array}
$$

where the first inequality becomes an equality if and only if

$$
\gamma \pmb { w } _ { j } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { j } = \gamma \pmb { w } _ { \ell } ^ { \top } \pmb { x } ^ { ( k ) } - b _ { \ell } , \quad \forall j , \ell \neq k \in [ K _ { 1 } ] ,
$$

and the second inequality becomes an equality if and only if

$$
\gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } = \gamma { \pmb w } _ { \ell } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { \ell } , \quad \forall k \in [ K _ { 1 } ] , K _ { 1 } < j , \ell \leq K .
$$

Then, by the concavity of the function $\log ( 1 + \mathrm { e } ^ { x } )$ , for any $c _ { 1 } , c _ { 2 } > 0$ , we get

$$
\begin{array} { r l } & { \begin{array} { r l } & { \langle \Delta \xi \Delta \xi ^ { ( 1 ) } \rangle } \\ & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \\ & { \vdots } \\ & { \langle \Delta \xi ^ { ( 1 ) } \rangle } \end{array} } \\ & { \begin{array} { r l } { \langle \Delta \xi ^ { ( 1 ) } \rangle } & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 2 ) } \Delta \xi ^ { ( 2 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 2 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \\ & { \vdots } \\ { \langle \Delta \xi ^ { ( 1 ) } \rangle } & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \end{array} } \\ & { \begin{array} { r l } { \langle \Delta \xi ^ { ( 1 ) } \rangle } & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \\ & { \vdots } \\ { \langle \Delta \xi ^ { ( 1 ) } \rangle } & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \end{array} } \\ & { \begin{array} { r l } { \langle \Delta \xi ^ { ( 1 ) } \rangle } & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \\ & { \vdots } \\ { \langle \Delta \xi ^ { ( 1 ) } \rangle } & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \end{array} } \\ &  \begin{array} { r l } { \langle \Delta \xi ^ { ( 1 ) } \rangle } & { = \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } \\ & { \vdots } \\ { \langle \Delta \xi ^ { ( 1 ) } \Delta \xi ^ { ( 1 ) } \rangle } &  = \langle \Delta \xi ^ { ( 1 ) } \ \end{array} \end{array}
$$

where

$$
\begin{array} { l } { { C = \displaystyle \frac 1 { 1 + c _ { 1 } + c _ { 2 } } \log \left( 1 + c _ { 1 } + c _ { 2 } \right) + \displaystyle \frac { c _ { 1 } } { 1 + c _ { 1 } + c _ { 2 } } \log \left( \displaystyle \frac { 1 + c _ { 1 } + c _ { 2 } } { c _ { 1 } } ( K _ { 1 } - 1 ) \right) } } \\ { { \displaystyle \phantom { \frac { c _ { 2 } } { c _ { 1 } + c _ { 2 } } \log \left( 1 + c _ { 1 } + c _ { 2 } \right) + \frac { c _ { 1 } } { c _ { 2 } } \log \left( 1 + c _ { 1 } + c _ { 2 } \right) } . } } \end{array}\tag{S.135}
$$

The second inequality achieves its equality if and only if

$$
\frac { 1 + c _ { 1 } + c _ { 2 } } { 1 } = \frac { 1 + c _ { 1 } + c _ { 2 } } { c _ { 1 } } ( K _ { 1 } - 1 ) \exp \Bigg ( \sum _ { j = 1 } ^ { K _ { 1 } } \frac { \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K _ { 1 } - 1 } \Bigg )
$$

$$
= \frac { 1 + c _ { 1 } + c _ { 2 } } { c _ { 2 } } ( K - K _ { 1 } ) \exp \Bigg ( \sum _ { j = K _ { 1 } + 1 } ^ { K } \frac { \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma { \pmb w } _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K - K _ { 1 } } \Bigg ) ,
$$

which is equivalent to

$$
\begin{array} { r l } & { c _ { 1 } = ( K _ { 1 } - 1 ) \exp \bigg ( \displaystyle \sum _ { j \ne k } ^ { K _ { 1 } } \frac { \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K - 1 } \bigg ) \quad \mathrm { a n d } } \\ & { c _ { 2 } = ( K - K _ { 1 } ) \exp \bigg ( \displaystyle \sum _ { j = K _ { 1 } + 1 } ^ { K } \frac { \gamma w _ { j } ^ { \top } { \pmb x } ^ { ( k ) } - b _ { j } - \gamma w _ { k } ^ { \top } { \pmb x } ^ { ( k ) } + b _ { k } } { K - K _ { 1 } } \bigg ) , } \end{array}
$$

as desired.

## S.2.4 Theorems in Contrastive Learning

Theorem S.16 Suppose that each class in dataset D contains at least two samples, and that the feature dimension is greater than the number of classes, i.e., $n _ { k } = | \mathcal D _ { k } | > 0$ for $\forall k \in [ K ]$ and $d \geq K - 1$ . On the dataset D, define losses

$$
f _ { \mathrm { s i g m } } ( \mathcal { F } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mathrm { s i g m } } ( X ^ { ( k ) } ) ,\tag{S.136}
$$

where

$$
L _ { \mathrm { s i g m } } ( \mathbf { X } ^ { ( k ) } ) = \frac { 1 } { n _ { k } } \sum _ { \mathbf { x } _ { p } \in \mathcal { F } _ { k } } \log \left( 1 + \mathrm { e } ^ { - \gamma \mathbf { x } _ { p } ^ { \top } \mathbf { x } ^ { ( k ) } + b _ { k } } \right) + \lambda \sum _ { j = 1 , j \neq k } ^ { K } \frac { 1 } { n _ { j } } \sum _ { \mathbf { x } _ { n } \in \mathcal { F } _ { j } } \log \left( 1 + \mathrm { e } ^ { \gamma \mathbf { x } _ { n } ^ { \top } \mathbf { x } ^ { ( k ) } - b _ { j } } \right) ,\tag{S.137}
$$

where $\lambda > 0$ . Then, at its minimum point,

$$
{ \pmb x } ^ { ( k ) } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in \mathcal { F } _ { k } } { \pmb x } = \bar { { \pmb x } } _ { k } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] ,\tag{S.138}
$$

$$
\langle \bar { \pmb { x } } _ { k } , \bar { \pmb { x } } _ { j } \rangle = - \frac { 1 } { K - 1 } , \quad \forall k \neq j \in [ K ] ,\tag{S.139}
$$

$$
b _ { k } = b ^ { \star } , \quad \forall k \in [ K ] ,\tag{S.140}
$$

where $b ^ { \star }$ is the unique solution of

$$
\mathrm { e } ^ { \frac { \gamma } { K - 1 } } \mathrm { e } ^ { 2 b } - \big [ \lambda ( K - 1 ) - 1 \big ] \mathrm { e } ^ { b } - \lambda ( K - 1 ) \mathrm { e } ^ { \gamma } = 0 .
$$

Proof The Sigmoid-based contrastive loss can be divided into two parts,

$$
\begin{array} { r l } & { \int _ { \mathrm { s i g m } } ( \mathcal { F } ) = \frac { 1 } { K } \underbrace { \displaystyle \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { x ^ { [ i ] } \in D _ { k } } } _ { x ^ { [ i ] } \in D _ { k } } L _ { \mathrm { s i g m } } ( X ^ { ( k ) } ) } \\ & { \qquad = \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { x ^ { [ i ] } \in \mathcal { F } _ { k } } \frac { 1 } { n _ { k } } \sum _ { x ^ { [ i ] } \in \mathcal { F } _ { k } } \log \big ( 1 + \mathrm { c } ^ { - \gamma x ^ { [ i ] } \pi ^ { [ i ] } + b _ { k } } \big ) } _ { f _ { \mathrm { s i g m } } ^ { ( \mathrm { s i r s } ) } } } \\ & { \qquad + \lambda \cdot \underbrace { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { x ^ { [ i ] } \in \mathcal { F } _ { k } } \sum _ { x ^ { [ i ] } \ne k } ^ { K } \frac { 1 } { n _ { j } } \sum _ { x ^ { [ i ] } \in \mathcal { F } _ { j } } \log \big ( 1 + \mathrm { c } ^ { \gamma x ^ { [ i ] } \pi ^ { [ i ] } - b _ { j } } \big ) } _ { f _ { \mathrm { s i g m } } ^ { ( \mathrm { s i r s } ) } } . } \end{array}
$$

According to the convexity of $\kappa ( x ) = \log ( 1 + \mathrm { e } ^ { - x } )$ ),

$$
\begin{array} { r l } { \displaystyle \int _ { \mathrm { R a g n } } ^ { ( 1 \mathrm { m a x } ) } } & { = \frac { 1 } { K } \displaystyle \sum _ { i = 1 } ^ { K } \frac { 1 } { \pi } \sum _ { k _ { x } < i > k _ { x } } \frac { 1 } { \pi k _ { x } } \displaystyle \sum _ { k \neq p \leq K } \log \Big ( 1 + \mathrm { e } ^ { - \gamma \alpha _ { p } ^ { \mathrm { T } } \alpha ^ { \mathrm { T } } ( k ) } | k _ { p } \Big ) } \\ & { = - \displaystyle \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \frac { 1 } { \pi k _ { x } } \displaystyle \sum _ { k \neq p < K } \frac { 1 } { \pi k _ { x } } \displaystyle \sum _ { k \neq p \leq K } \Big ( \gamma \ x _ { p } ^ { \mathrm { T } } \ \alpha ^ { ( k ) } - b _ { k } \Big ) } \\ & { \geq \displaystyle \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \kappa \left( \displaystyle \frac { 1 } { \pi k _ { x } } \displaystyle \sum _ { k \neq p \leq K } \displaystyle \sum _ { s \neq p \leq K } \Big ( \gamma \ x _ { p } ^ { \mathrm { T } } \ \alpha ^ { ( k ) } - b _ { k } \Big ) \right) } \\ & { \geq \displaystyle \kappa \Big ( \displaystyle \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \frac { 1 } { \pi k _ { x } } \displaystyle \sum _ { s \neq s \neq p \leq K } \ \sum _ { s \neq p \leq K } \Big ( \gamma \ x _ { p } ^ { \mathrm { T } } \ \alpha ^ { ( k ) } - b _ { k } \Big ) \Big ) } \\ & { = \displaystyle \kappa \Big ( \displaystyle \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \frac { 1 } { \pi k _ { x } } \displaystyle \sum _ { s \neq p \leq K } \ \sum _ { s \neq p \leq K } \Big ( \gamma \ x _ { p } ^ { \mathrm { T } } \alpha ^ { ( k ) } - b _ { k } \Big ) \Big ) } \\ &  = \displaystyle \kappa \Big ( \displaystyle \frac { 1 } { K } \displaystyle \sum _ { k = 1 } ^ { K } \displaystyle \sum \end{array}\tag{S.141}
$$

where the first inequality becomes an equality if and only if

$$
\gamma x _ { \mathrm { p } } ^ { \top } x ^ { ( k ) } - b _ { k } = \gamma \hat { x } _ { \mathrm { p } } ^ { \top } \hat { x } ^ { ( k ) } - b _ { k } , \quad \mathrm { f o r } \quad \forall x _ { \mathrm { p } } , \hat { x } _ { \mathrm { p } } , { x } ^ { ( k ) } , \hat { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] ,\tag{S.142}
$$

and the second inequality becomes an equality if and only if

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { n _ { k } ^ { 2 } } \sum _ { x ^ { ( k ) } \in \mathcal { F } _ { k } } \sum _ { x _ { \mathrm { p } } \in \mathcal { F } _ { k } } \left( \gamma x _ { \mathrm { p } } ^ { \top } x ^ { ( k ) } - b _ { k } \right) = \frac { 1 } { n _ { \ell } ^ { 2 } } \sum _ { x ^ { ( \ell ) } \in \mathcal { F } _ { \ell } } \sum _ { x _ { \mathrm { p } } \in \mathcal { F } _ { \ell } } \left( \gamma x _ { \mathrm { p } } ^ { \top } x ^ { ( \ell ) } - b _ { \ell } \right) } \\ & { \gamma x _ { \mathrm { p } } ^ { \top } x ^ { ( k ) } - b _ { k } = \gamma \hat { x } _ { \mathrm { p } } ^ { \top } x ^ { ( \ell ) } - b _ { \ell } , \quad \forall x _ { \mathrm { p } } , x ^ { ( k ) } \in \mathcal { F } _ { k } , \forall \hat { x } _ { \mathrm { p } } , x ^ { ( \ell ) } \in \mathcal { F } _ { \ell } , \forall k , \ell \in [ K ] . } \end{array}\tag{S.143}
$$

Similarly, for the second term, according to the convexity of $\kappa ( x ) = \log ( 1 + \mathrm { e } ^ { - x } )$

$$
\begin{array} { r l } { I _ { \le \le n } ^ { \mathrm { ( r i m e t ) } } } & { = \displaystyle \frac { 1 } { K } \sum _ { 1 \le 1 } ^ { K } \displaystyle \sum _ { u _ { K } \le \| u _ { K } ^ { \mathrm { L } } \| _ { \le \sigma _ { K } } \le 1 , j \le M } \displaystyle \sum _ { u _ { K } \le \| u _ { K } ^ { \mathrm { L } } \| _ { \le \sigma _ { K } } \le 1 } ^ { K } \log \big ( 1 + \mathrm { e } ^ { \gamma \pi _ { K } ^ { \mathrm { L } } \pi ^ { ( k ) } - \delta _ { k } } \big ) } \\ & { = \displaystyle \frac { 1 } { K } \sum _ { k = 1 - \lfloor \gamma \delta \rfloor = 1 } ^ { K } \displaystyle \sum _ { u _ { K } ^ { \mathrm { L } } = 1 , j \le K } ^ { K } \displaystyle \sum _ { u _ { K } ^ { \mathrm { L } } \ge u _ { K } ^ { \mathrm { L } } \le \varepsilon _ { p } ^ { \varepsilon } } ^ { K } \displaystyle \sum _ { ( u _ { K } ^ { \mathrm { L } } - 1 ) \le 1 } ^ { K } \nu \big ( \delta _ { p } - \gamma \overline { { x } } _ { u } ^ { \mathrm { T } } \pi ^ { ( k ) } \big ) } \\ & { \ge \displaystyle \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \displaystyle \sum _ { j = 1 } ^ { K } \nu \displaystyle \sum _ { u _ { K } ^ { \mathrm { L } } \le 1 } ^ { K } \nu \displaystyle \sum _ { u _ { K } ^ { \mathrm { L } } \ge 1 \le \gamma \delta _ { p } ^ { \varepsilon } } ^ { K } \qquad \sum _ { ( b _ { K } ^ { \mathrm { L } } - 1 ) \le 1 } ^ { K } \nu \displaystyle \sum _ { u _ { K } ^ { \mathrm { L } } \le \varepsilon _ { p } ^ { \varepsilon } } ^ { K } \big ( \delta _ { p } - \gamma x _ { u } ^ { \mathrm { T } } \pi ^ { ( k ) } \big ) \Big ) } \\ &  \ge ( K - 1 ) \kappa \bigg ( \displaystyle \frac { 1 } { K ( K - 1 ) } \displaystyle \sum _ { k = 1 - \lfloor \gamma \delta \rfloor = 1 }  \end{array}
$$

$$
\begin{array} { r l r } & { = \langle \mathcal { K } - 1 \rangle \epsilon \frac { \langle \hat { \mathcal { K } } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } } & { \sum _ { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } \frac { \langle \mathcal { K } \rangle } { \langle \mathcal { K } \rangle } } \\ & { } & { + \frac { \mathcal { K } } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } } \\ &  \leq \langle \mathcal { K } - 1 \rangle \epsilon \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } } \frac { \langle \mathcal { K } \rangle } { \mathcal { K } \mathcal { K } } \frac  \langle  \end{array}\tag{4}
$$

where the first inequality is derived from the convexity of $\kappa ( x )$ and it becomes an equality if and only if

$$
b _ { j } - \gamma \mathbf { x } _ { \mathbf { n } } ^ { \top } \mathbf { x } ^ { ( k ) } = b _ { j } - \gamma \hat { x } _ { \mathbf { n } } ^ { \top } \hat { x } ^ { ( k ) } , \quad \forall \mathbf { x } ^ { ( k ) } , \hat { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall \mathbf { x } _ { \mathbf { n } } , \hat { x } _ { \mathbf { n } } \in \mathcal { F } _ { j } , \forall k \neq j \in [ K ] ,\tag{S.145}
$$

the second inequality is derived from the convexity of $\kappa ( x )$ and it becomes an equality if and only if

$$
\begin{array} { r l } & { \cfrac { 1 } { n _ { k } n _ { j } } \displaystyle \sum _ { \mathbf { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \sum _ { \mathbf { x } _ { \mathrm { n } } \in \mathcal { F } _ { j } } \big ( b _ { j } - \gamma \mathbf { x } _ { \mathrm { n } } ^ { \top } \mathbf { x } ^ { ( k ) } \big ) = \cfrac { 1 } { n _ { k ^ { \prime } } n _ { j ^ { \prime } } } \displaystyle \sum _ { \mathbf { x } ^ { ( k ^ { \prime } ) } \in \mathcal { F } _ { k ^ { \prime } } } \sum _ { \mathbf { x } _ { \mathrm { n } } \in \mathcal { F } _ { j ^ { \prime } } } \big ( b _ { j ^ { \prime } } - \gamma \mathbf { x } _ { \mathrm { n } } ^ { \top } \mathbf { x } ^ { ( k ^ { \prime } ) } \big ) } \\ { \overset { S \cdot 1 4 5 } { \longleftrightarrow } } & { b _ { j } - \gamma \mathbf { x } _ { \mathrm { n } } ^ { \top } \mathbf { x } ^ { ( k ) } = b _ { j ^ { \prime } } - \gamma \hat { \mathbf { x } } _ { \mathrm { n } } ^ { \top } \hat { \mathbf { x } } ^ { ( k ^ { \prime } ) } , } \\ & { \qquad \forall \mathbf { x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall \mathbf { x } _ { \mathrm { n } } \in \mathcal { F } _ { j } , \forall \hat { \mathbf { x } } ^ { ( k ^ { \prime } ) } \in \mathcal { F } _ { k ^ { \prime } } , \forall \hat { \mathbf { x } } _ { \mathrm { n } } \in \mathcal { F } _ { j ^ { \prime } } , \forall k \neq j , k ^ { \prime } \neq j ^ { \prime } \in [ K ] , } \end{array}\tag{S.146}
$$

the third inequality is derived from the monotonic decreasing of $\kappa ( x )$ and it becomes an equality if and only if

$$
\begin{array} { r } { \pmb { x } ^ { ( k ) } = \pmb { x } _ { \mathrm { p } } , \quad \mathrm { f o r } \forall \pmb { x } ^ { ( k ) } , \pmb { x } _ { \mathrm { p } } \in \mathcal { F } _ { k } , \forall k \in [ K ] , } \end{array}\tag{S.147}
$$

and the fourth inequality is derived from the monotonic decreasing of $\kappa ( x )$ and it becomes an equality if and only if

$$
\tilde { \mathbf {  { x } } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bar { \mathbf {  { x } } } _ { k } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { \mathbf {  { x } } ^ { ( k ) } \in \mathcal { F } _ { k } } = \mathbf { 0 } .\tag{S.148}
$$

Combining Eqs. (S.143,S.147), one can deduce that, $\exists b \in \mathbb { R }$

$$
b _ { k } = b , \quad \forall k \in [ K ] .\tag{S.149}
$$

Based on the above analysis, if and only if Eqs. (S.142, S.143, S.145, S.146, S.147, S.148),

$$
\begin{array} { r l } & { \int _ { \mathrm { s i g m } } ( \mathcal { F } ) = \displaystyle { J } \frac { ( \mathrm { i n f r a n } ) } { \mathrm { s i g m } } + \lambda \int _ { \mathrm { s i g m } } ^ { ( \mathrm { i n t r a } ) } } \\ & { \geq \kappa \bigg ( \displaystyle { \frac { \kappa } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } ^ { 2 } } \sum _ { w _ { k } ^ { \prime } \in \mathcal { F } _ { k } } \sum _ { v \in \mathcal { F } _ { k } } \gamma x _ { \mathrm { p } } ^ { \mathrm { r } } x _ { \mathrm { p } } ^ { ( k ) } - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } b _ { k } \bigg ) + \lambda ( K - 1 ) \kappa \bigg ( \displaystyle { \frac { 1 } { K } \sum _ { k = 1 } ^ { I } b _ { k } + \frac { \gamma } { K - 1 } } \bigg ) } } \\ & { = \kappa \bigg ( \displaystyle { \frac { \kappa } { K } \sum _ { k = 1 } ^ { K } \gamma - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } b _ { k } } \bigg ) + \lambda ( K - 1 ) \kappa \bigg ( \displaystyle { \frac { 1 } { K } \sum _ { k = 1 } ^ { I } b _ { k } + \frac { \gamma } { K - 1 } } \bigg ) } \\ & { = \kappa \big ( \gamma - b \big ) + \lambda ( K - 1 ) \kappa \bigg ( b + \displaystyle { \frac { \gamma } { K - 1 } } \bigg ) } \\ & { = \log \bigg ( 1 + \exp \big ( b - \gamma \big ) \bigg ) + \lambda ( K - 1 ) \log \bigg ( 1 + \exp \big ( - b - \displaystyle { \frac { \gamma } { K - 1 } } \big ) \bigg ) \triangleq \hat { f } _ { \mathrm { s i g m } } ( b ) . \quad ( \mathrm { S } . 1 } \end{array}\tag{50}
$$

At the minimum points of $\hat { f } _ { \mathrm { s i g m } } ( b )$ 2

$$
\begin{array} { l l l } { 0 = \displaystyle \frac { \partial \hat { f } _ { \mathrm { s i g m } } ( b ) } { \partial b } = \displaystyle \frac { \exp \left( b - \gamma \right) } { 1 + \exp \left( b - \gamma \right) } - \lambda ( K - 1 ) \displaystyle \frac { \exp \big ( - b - \frac { \gamma } { K - 1 } \big ) } { 1 + \exp \big ( - b - \frac { \gamma } { K - 1 } \big ) } } \\ { \displaystyle } & { = \displaystyle \frac { 1 } { 1 + \exp \big ( \gamma - b \big ) } - \frac { \lambda ( K - 1 ) } { 1 + \exp \big ( b + \frac { \gamma } { K - 1 } \big ) } } \\ { \iff \mathrm { e } ^ { \frac { \gamma } { K - 1 } } \mathrm { e } ^ { 2 b } - \big [ \lambda ( K - 1 ) - 1 \big ] \mathrm { e } ^ { b } - \lambda ( K - 1 ) \mathrm { e } ^ { \gamma } = 0 . } \end{array}\tag{S.151}
$$

From Eq. (S.147), one can deduce that

$$
{ \pmb x } ^ { ( k ) } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in \mathcal { F } _ { k } } { \pmb x } = { \bar { \pmb x } } _ { k } , \quad \forall { \pmb x } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] .
$$

Similar to the proof of Theorem S.4, from Eq. (S.148), one can deduce

$$
\langle \bar { \pmb { x } } _ { k } , \bar { \pmb { x } } _ { j } \rangle = - \frac { 1 } { K - 1 } , \quad \forall k \neq j \in [ K ] .
$$

It completes the proof.

Finally, if set $b _ { k } = 0$ for $\forall k \in [ K ]$ in (S.137), Eqs. (S.147, S.148) can still be derived from the minimization of $f _ { \mathrm { s i g m } } ^ { \mathrm { ( i n t e r ) } }$ , and in this case it sufices for each class in D to contain at least one sample. Therefore, if remove the biases in the Sigmoid-based contrastive loss, the main conclusions of this theorem, i.e., Eqs. (S.138) and (S.139) still hold when reaching its minimum value, provided that there are no empty classes in $\mathcal { D } .$

Theorem S.17 Suppose that each class in dataset D contains at least two samples, and that the feature dimension is greater than the number of classes, i.e., $n _ { k } = | \mathcal D _ { k } | > 0$ for $\forall k \in [ K ]$ and $d > K$ . On the dataset D, define losses

$$
f _ { \mathrm { s o f t } } ( \mathcal { F } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { X ^ { ( k ) } \in \mathcal { D } _ { k } } L _ { \mathrm { s o f t } } ( X ^ { ( k ) } ) ,\tag{S.152}
$$

where

$$
L _ { \mathrm { s o f t } } ( \boldsymbol { X } ^ { ( k ) } ) = - \frac { 1 } { n _ { k } } \sum _ { \boldsymbol { x } _ { p } \in \mathcal { F } _ { k } } \log \Bigg ( \frac { \mathrm { e } ^ { \gamma \boldsymbol { x } _ { p } ^ { \top } \boldsymbol { x } ^ { ( k ) } } } { \sum _ { j = 1 } ^ { K } \frac { 1 } { n _ { j } } \sum _ { \boldsymbol { x } \in \mathcal { F } _ { j } } \mathrm { e } ^ { \gamma \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { ( k ) } } } \Bigg ) .\tag{S.153}
$$

Then, at its minimum point,

$$
\begin{array} { l } { { \displaystyle { \pmb x } ^ { ( k ) } = \frac { 1 } { n _ { k } } \sum _ { { \pmb x } \in { \mathscr F } _ { k } } { \pmb x } = { \bar { \pmb x } } _ { k } } , \quad \forall { \pmb x } ^ { ( k ) } \in { \mathscr F } _ { k } , \forall k \in [ K ] } ,  \\ { { \displaystyle \langle { \bar { \pmb x } } _ { k } , { \bar { \pmb x } } _ { j } \rangle = - \frac { 1 } { K - 1 } } , \quad \forall k \not = j \in [ K ] . } \end{array}\tag{S.154}
$$

(S.155)

Proof According to the convexities of $\mathrm { e } ^ { x }$ and log x,

$$
\begin{array} { r l } &  \begin{array} { r l } &  \mathrm { i } \cdot \sum _ { j = 1 } ^ { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \sum _ { k = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { k = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^ { N } \sum _ { l = 1 } ^  \end{array} \end{array}
$$

$$
\begin{array} { r l } & { = \frac { 1 } { K } \displaystyle \sum _ { i = 1 } ^ { K } \frac { 1 } { w _ { i } w _ { i } w _ { j } w _ { k } } \{ \frac { e _ { 1 } } { 1 - e _ { 1 } } \frac { 1 + e _ { 1 } } { c _ { 1 } }  } \\ & { \qquad +  \frac { 1 + e _ { 1 } } { 1 + e _ { 1 } } ( K - i ) \mathrm { e m } [ \frac { 1 } { K - 1 } \displaystyle \sum _ { i = 1 } ^ { K } ( \frac { 1 } { s _ { i } } \displaystyle \sum _ { j \in \mathcal { S } _ { i } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { i } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { i } } ^ { ( k ) } - \frac { 1 } { m _ { i } } \displaystyle \sum _ { i \in \mathcal { S } _ { i } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { i } } ^ { ( k ) } ) ] \} } \\ & { \geq \frac { 1 } { K } \displaystyle \sum _ { i = 1 } ^ { K } \sum _ { i = 1 } ^ { K } \sum _ { i \in \mathcal { S } _ { i } } \{ \frac { e _ { 1 } } { 1 + e _ { 1 } } \log [ \frac { 1 } { \Theta _ { i } } ( \displaystyle \sum _ { \ell = 1 } ^ { 1 } )  } \\ &  \qquad -  \frac { 1 } { 1 + e _ { 1 } } \displaystyle \sum _ { i = 1 } ^ { K } ( \frac { 1 } { 1 - e _ { 1 } } ) \exp [ \frac { 1 } { K - 1 } \displaystyle \sum _ { i = 1 } ^ { K } ( \frac { 1 } { e _ { 1 } } \displaystyle \sum _ { j \in \mathcal { S } _ { i } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { i } } \mathbb { T } _ { \bar { j } } \mathbb { T } _ { \bar { i } } ^ { ( k ) } - \frac  \end{array}
$$

where the first inequality is derived from the convexity of $\mathrm { e } ^ { x }$ and it becomes an equality if and only if

$$
\gamma { \pmb x } ^ { \top } { \pmb x } ^ { ( k ) } = \gamma \hat { { \pmb x } } ^ { \top } { \pmb x } ^ { ( k ) } , \quad \forall { \pmb x } , \hat { { \pmb x } } \in { \mathcal F } _ { j } , \forall { \pmb x } ^ { ( k ) } \in { \mathcal F } _ { k } , \forall j , k \in [ K ] ,\tag{S.156}
$$

the second inequality is derived from the convexity of $\mathrm { e } ^ { x }$ and it becomes an equality if and only if

[S.156] ⇔

$$
\begin{array} { r l r } & { \displaystyle \frac { 1 } { n _ { j } } \sum _ { \boldsymbol { x } \in \mathcal { F } _ { j } } \gamma \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { ( k ) } - \frac { 1 } { n _ { k } } \sum _ { x _ { \mathrm { p } } \in \mathcal { F } _ { k } } \gamma \boldsymbol { x } _ { \mathrm { p } } ^ { \top } \boldsymbol { x } ^ { ( k ) } = \frac { 1 } { n _ { j ^ { \prime } } } \sum _ { \boldsymbol { x } \in \mathcal { F } _ { j ^ { \prime } } } \gamma \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { ( k ) } - \frac { 1 } { n _ { k } } \sum _ { x _ { \mathrm { p } } \in \mathcal { F } _ { k } } \gamma \boldsymbol { x } _ { \mathrm { p } } ^ { \top } \boldsymbol { x } ^ { ( k ) } } & \\ & { \gamma \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { ( k ) } - \gamma \boldsymbol { x } _ { \mathrm { p } } ^ { \top } \boldsymbol { x } ^ { ( k ) } = \gamma \hat { \boldsymbol { x } } ^ { \top } \hat { \boldsymbol { x } } ^ { ( k ) } - \gamma \hat { \boldsymbol { x } } _ { \mathrm { p } } ^ { \top } \hat { \boldsymbol { x } } ^ { ( k ) } , \quad \forall \boldsymbol { x } \in \mathcal { F } _ { j } , \hat { \boldsymbol { x } } \in \mathcal { F } _ { j ^ { \prime } } , } & \\ & { \forall x _ { \mathrm { p } } , \boldsymbol { x } ^ { ( k ) } , \hat { x } _ { \mathrm { p } } , \hat { \boldsymbol { x } } ^ { ( k ) } \in \mathcal { F } _ { k } , \forall k \in [ K ] , \forall j , j ^ { \prime } \neq k \in [ K ] , } & { \mathrm { ( S . l . } \forall k ] . } \end{array}\tag{7}
$$

and the third inequality is derived from the convexity of log x and it becomes an equality if and only if

⇔

$$
\begin{array} { l } { \displaystyle \frac { 1 + c _ { 1 } } { c _ { 1 } } = ( 1 + c _ { 1 } ) ( K - 1 ) \exp \left[ \frac { 1 } { K - 1 } \sum _ { j = 1 , j \neq k } ^ { K } \left( \frac { 1 } { n _ { j } } \sum _ { \boldsymbol { x } \in \mathcal { F } _ { j } } \gamma \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { ( k ) } - \frac { 1 } { n _ { k } } \sum _ { \boldsymbol { x } _ { \mathrm { p } } \in \mathcal { F } _ { k } } \gamma \boldsymbol { x } _ { \mathrm { p } } ^ { \top } \boldsymbol { x } ^ { ( k ) } \right) \right] } \\ { \displaystyle c _ { 1 } = \left\{ ( K - 1 ) \exp \left[ \frac { 1 } { K - 1 } \sum _ { j = 1 , j \neq k } ^ { K } \left( \frac { 1 } { n _ { j } } \sum _ { \boldsymbol { x } \in \mathcal { F } _ { j } } \gamma \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { ( k ) } - \frac { 1 } { n _ { k } } \sum _ { \boldsymbol { x } _ { \mathrm { p } } \in \mathcal { F } _ { k } } \gamma \boldsymbol { x } _ { \mathrm { p } } ^ { \top } \boldsymbol { x } ^ { ( k ) } \right) \right] \right\} ^ { - 1 } . } \end{array}\tag{S.158}
$$

Furthermore,

$$
\begin{array} { r l r l } { d u _ { \infty } ( T ) \mathcal { E } \equiv \frac { \lambda } { T } \displaystyle { \sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \frac { 1 } { \lambda } } \sum _ { \ell = 1 } ^ { n } \frac { \sum _ { \ell = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } } { \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } } ( \frac { 1 } { \lambda } \sum _ { \ell = 1 } ^ { n } z _ { \ell } ) ^ { 2 } \mathcal { E } ^ { \lambda } \equiv \frac { \lambda } { T } \displaystyle { \sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } } \mathcal { E } ^ { \lambda } ) + \mathcal { E } } \\ & { } & { - \frac { \lambda } { T } \displaystyle { \sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } } \sum _ { \ell = 1 } ^ { n } ( \frac { 1 } { \lambda } \sum _ { \ell = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } z _ { \ell } ) ^ { 2 } \mathcal { E } ^ { \lambda } \equiv \frac { \lambda } { T } \displaystyle { E _ { i } ^ { \ell } } \sum _ { \ell = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } \mathcal { E } ^ { \lambda } \Bigg ) + \mathcal { E } } \\ & { } & { - \frac { \lambda } { T } \displaystyle { \sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } } \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } \mathcal { E } ^ { \lambda } \equiv \frac { \lambda } { T } \displaystyle { E _ { i } ^ { \ell } } \sum _ { \ell = 1 } ^ { n } \mu _ { \ell } ^ { \ell } } \\ & { }  \end{array}
$$

where $\begin{array} { r } { \tilde { \mathbfit { x } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bar { \mathbfit { x } } _ { k } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { n _ { k } } \sum _ { \pmb { x } ^ { ( k ) } \in \mathcal { F } _ { k } } \mathbfit { x } ^ { ( k ) } } \end{array}$ , the second inequality becomes an equality if and only if

$$
\begin{array} { r } { \pmb { x } _ { \mathrm { p } } = \pmb { x } ^ { ( k ) } , \quad \forall \pmb { x } , \pmb { x } _ { \mathrm { p } } \in \mathcal { F } _ { k } , \ \forall k \in [ K ] , } \end{array}\tag{S.160}
$$

and the third inequality becomes an equality if and only if

$$
\tilde { \pmb { x } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bar { \pmb { x } } _ { k } = \mathbf { 0 } .\tag{S.161}
$$

From Eq. (S.160), one can deduce Eq. (S.154), from Eq. (S.161), one can deduce Eq. (S.155), and they are compatible with Eq. (S.156) and (S.157), which completes the proof.

## References

Shaden Alshammari, Yu-Xiong Wang, Deva Ramanan, and Shu Kong. Long-tailed recognition via weight balancing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6897–6907, 2022.

Tina Behnia, Ganesh Ramachandra Kini, Vala Vakilian, and Christos Thrampoulidis. On the implicit geometry of cross-entropy parameterizations for label-imbalanced data. In International Conference on Artificial Intelligence and Statistics, pages 10815–10838, 2023. URL https://proceedings.mlr.press/v206/behnia23a.html.

Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, and Tengyu Ma. Learning imbalanced datasets with label-distribution-aware margin loss. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/file/ 621461af90cadfdaf0e8d4cc25129f91-Paper.pdf.

Hien Dang, Tho Tran Huu, Stanley J. Osher, Hung Tran-The, Nhat Ho, and Tan Minh Nguyen. Neural collapse in deep linear networks: From balanced to imbalanced data. In Proceedings of the 40th International Conference on Machine Learning, pages 6873–6947, 2023. URL https://proceedings.mlr.press/v202/dang23b.html.

Hien Dang, Tho Tran Huu, Tan Minh Nguyen, and Nhat Ho. Neural collapse for crossentropy class-imbalanced learning with unconstrained reLU features model. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/ forum?id=YBetKvUlF7.

Cong Fang, Hangfeng He, Qi Long, and Weijie J Su. Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced training. Proceedings of the National Academy of Sciences, 118(43):e2103091118, 2021.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9729–9738, 2020.

Wanli Hong and Shuyang Ling. Neural collapse for unconstrained feature model under crossentropy loss with imbalanced data. Journal of Machine Learning Research, 25(192):1–48, 2024.

Wanli Hong and Shuyang Ling. Beyond unconstrained features: Neural collapse for shallow neural networks with general data. Journal of Machine Learning Research, 27(82):1–46, 2026. URL https://www.jmlr.org/papers/v27/24-1429.html.

Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng, and Yannis Kalantidis. Decoupling representation and classifier for long-tailed recognition. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=r1gRTCVFvB.

Tejaswi Kasarla, Gertjan Burghouts, Max van Spengler, Elise van der Pol, Rita Cucchiara, and Pascal Mettes. Maximum class separation as inductive bias in one matrix. In Advances in Neural Information Processing Systems, volume 35, pages 19553–19566. Curran Associates, Inc., 2022. doi: 10.52202/068431-1421.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673, 2020.

Ganesh Ramachandra Kini, Vala Vakilian, Tina Behnia, Jaidev Gill, and Christos Thrampoulidis. Symmetric neural-collapse representations with supervised contrastive loss: The impact of reLU and batching. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=AyXIDfvYg8.

A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Handbook of Systemic Autoimmune Diseases, 2009.

Chaohua Li, Enhao Zhang, Chuanxing Geng, and Songcan Chen. Unlocking better closedset alignment based on neural collapse for open-set recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 18227–18235, 2025a.

Qiufu Li, Huibin Xiao, and Linlin Shen. BCE vs. CE in deep feature learning. In Forty-second International Conference on Machine Learning, 2025b. URL https: //openreview.net/forum?id=aoLFIUlyPE.

Xuantong Liu, Jianfeng Zhang, Tianyang Hu, He Cao, Yuan Yao, and Lujia Pan. Inducing neural collapse in deep long-tailed learning. In International Conference on Artificial Intelligence and Statistics, pages 11534–11544. PMLR, 2023a.

Zhengfa Liu, Guang Chen, Zhijun Li, Yu Kang, Sanqing Qu, and Changjun Jiang. Psdc: A prototype-based shared-dummy classifier model for open-set domain adaptation. IEEE Transactions on Cybernetics, 53(11):7353–7366, 2023b.

Jianfeng Lu and Stefan Steinerberger. Neural collapse under cross-entropy loss. Applied and Computational Harmonic Analysis, 59:224–241, 2022.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Vardan Papyan, Xiao Y Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020.

Gao Peifeng, Qianqian Xu, Peisong Wen, Zhiyong Yang, Huiyang Shao, and Qingming Huang. Feature directions matter: Long-tailed learning via rotated balanced representation. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 27542–27563. PMLR, 23–29 Jul 2023.

Kihyuk Sohn. Improved deep metric learning with multi-class n-pair loss objective. Advances in neural information processing systems, 29, 2016.

Christos Thrampoulidis, Ganesh Ramachandra Kini, Vala Vakilian, and Tina Behnia. Imbalance trouble: Revisiting neural-collapse geometry. In NeurIPS, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html.

Jinping Wang, Zixin Tong, Zhiwu Xie, and Zhiqiang Gao. Rethinking loss reweighting for imbalance learning as an inverse problem: A neural collapse point of view. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 130142–130161. PMLR, 06–11 Jul 2026.

Xun Wang, Xintong Han, Weilin Huang, Dengke Dong, and Matthew R Scott. Multisimilarity loss with general pair weighting for deep metric learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5022–5030, 2019.

Yandong Wen, Weiyang Liu, Yao Feng, Bhiksha Raj, Rita Singh, Adrian Weller, Michael J Black, and Bernhard Sch¨olkopf. Pairwise similarity learning is SimPLE. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5308–5318, 2023.

Hongren Yan, Yuhua Qian, Furong Peng, Jiachen Luo, Zheqing Zhu, and Feijiang Li. Neural collapse to multiple centers for imbalanced data. In Advances in Neural Information Processing Systems, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/ hash/78a2ed8e7da1829d64fb3561adbd706b-Abstract-Conference.html.

Yibo Yang, Shixiang Chen, Xiangtai Li, Liang Xie, Zhouchen Lin, and Dacheng Tao. Inducing neural collapse in imbalanced learning: Do we really need a learnable classifier at the end of deep neural network? Advances in neural information processing systems, 35: 37991–38002, 2022.

Lingjie Yi, Jiachen Yao, Weimin Lyu, Haibin Ling, Raphael Douady, and Chao Chen. Geometry of long-tailed representation learning: Rebalancing features for skewed distributions. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=GySIAKEwtZ.

Da-Wei Zhou, Han-Jia Ye, and De-Chuan Zhan. Learning placeholders for open-set recognition. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4401–4410, 2021. URL https://openaccess.thecvf.com/content/ CVPR2021/html/Zhou\_Learning\_Placeholders\_for\_Open-Set\_Recognition\_CVPR\_ 2021\_paper.html.

Jianggang Zhu, Zheng Wang, Jingjing Chen, Yi-Ping Phoebe Chen, and Yu-Gang Jiang. Balanced contrastive learning for long-tailed visual recognition. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6898–6907. IEEE, 2022.