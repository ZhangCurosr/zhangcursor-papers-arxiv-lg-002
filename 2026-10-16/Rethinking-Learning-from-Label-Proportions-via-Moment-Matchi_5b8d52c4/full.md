# Rethinking Learning from Label Proportions via Moment Matching

Tianhao Ma<sup>1</sup>, Wei Wang<sup>2</sup>, Yivan Zhang<sup>1,2</sup>, Dong-Dong Wu<sup>1,2</sup>, Takashi Ishida<sup>2,1</sup>, Gang Niu<sup>2</sup>, Masashi Sugiyama<sup>2,1</sup> <sup>1</sup>The University of Tokyo, <sup>2</sup>RIKEN AIP

## Abstract

Learning from label proportions (LLP) is a weakly supervised setting where training data are grouped into bags with only bag-level label proportions observed, and the goal is to learn a classifier that predicts labels for individual instances. Recently, a large body of methods has emerged; however, most of them rely on the assumption that instance–label pairs are sampled i.i.d. and randomly assembled into bags, an assumption often violated in real-world scenarios. Our experimental results indicate that these methods perform suboptimally under non-random settings. In this work, we adopt a more realistic assumption: bags are sampled i.i.d., and instance labels are conditionally independent given the instances. Under this setting, we show that the label counts follow a Poisson multinomial distribution. Motivated by this observation, we propose LLP via moment matching (LLP-MM), a simple yet effective approach which leverages multi-order factorial moments as training objectives, encouraging the classifier predictions to match these moments and thereby more fully exploit the underlying statistical structure of the data. Extensive experiments on benchmark datasets under various bag construction strategies demonstrate the effectiveness of our approach while maintaining high computational efficiency. Our code is available at https://github.com/TianhaoMa5/ LLP-MM.

## 1 Introduction

Recently, the use of large-scale data has pushed machine learning and artificial intelligence to unprecedented levels of performance. However, in many real-world scenarios the training data are imperfect: labels may be ambiguous, weak, or even incorrect. These issues have motivated a popular machine learning paradigm, weakly supervised learning (WSL) [Sugiyama et al., 2022], which aims to train robust models from such imperfect supervision. Representative forms of WSL include semi-supervised learning [Li et al., 2021, Lai et al., 2022], noisy-label learning [Patrini et al., 2017], partial-label learning [Lv et al., 2020, Wang et al., 2025], and unlabeled-unlabeled learning [Lu et al., 2021, Wei et al., 2024].

Learning from label proportions (LLP) is an important WSL problem that has recently attracted a lot of attention [Quadrianto et al., 2009, Tsai and Lin, 2020, Zhang et al., 2022, Ma et al., 2025, 2026a]. It naturally arises in many real-world applications, such as privacy-sensitive scenarios [Diemert et al., 2022, Apple, 2025, Google, 2024], as well as in image analysis tasks where instance-level annotations are difficult or costly to obtain [Lai et al., 2014, Tokunaga et al., 2020]. In these scenarios, only label proportions are available for training instead of instance-level annotations, while the goal remains to learn a classifier that predicts labels for individual instances.

In recent years, many LLP methods with strong performance as well as theoretical guarantees have been proposed [Busa-Fekete et al., 2023, Li et al., 2024, Busa-Fekete et al., 2025]. These methods are often based on the random-bag assumption about the generation process of LLP data. First, instance– label pairs are sampled independently from the same joint density. Then, the bag is constructed by grouping instances uniformly at random. Finally, the label proportion is calculated as the fraction of instances from each class. However, in many real-world applications, bags are rarely formed uniformly at random. For example, election polling data is a practical application of LLP where instance-level labels are confidential while the results of regional support rates can be considered as label proportions [Sun et al., 2017, Ma et al., 2026a]. Here, voters within the same region often exhibit correlated voting behaviors, so the resulting groups are affected by geographic and demographic properties rather than random assignment. Similar structured-bag grouping mechanisms also arise in online advertising and privacy-preserving measurement, where instances are often grouped according to pre-defined categorical attributes or platform-defined grouping keys, rather than being grouped uniformly at random [Diemert et al., 2022, Brahmbhatt et al., 2024].

![](images/bc1550d2e731a172bb368aa91752692dfa6ae2c3c1e19ed44238bb6f7487913d.jpg)  
Figure 1: An illustration of structured bag construction based on instance features. A common example of a structured-bag assumption is to group instances with similar feature patterns into the same bag; for example, as shown in the figure, instances sharing the same shape feature are assigned to one bag.

To address this problem, we adopt a more realistic data generation assumption, referred to as the structured-bag assumption, as illustrated in Fig. 1. In this setting, we allow instance correlations within the bag and assume that the grouping strategy is dependent on the individual examples. Besides, we assume that the instance labels are conditionally independent given the bag. Based on this assumption, we observe that the bag-level label counts correspond to the sum of independent but non-identically distributed categorical random variables, and therefore follow a Poisson multinomial distribution (PMD) [Daskalakis et al., 2015]. Consequently, the bag-level label statistics can be considered as following a normalized version of the PMD. This PMD perspective motivates the design of LLP algorithms that exploit the distributional information contained in the observed bag proportions.

Although it is natural to consider maximum likelihood estimation for LLP data, the likelihood of the PMD is difficult to compute, as its complexity grows exponentially with the number of classes [Lin et al., 2023]. Therefore, we propose LLP via Moment Matching (LLP-MM), a simple yet effective approach that trains the classifier by matching classifier-induced multi-order factorial moments to the corresponding empirical factorial moments derived from the observed bag proportions. We also find that the classical proportion matching strategy [Yu et al., 2014, Ardehaly and Culotta, 2017] is a special case of our framework that only uses the first-order empirical moment as the target for the classifier induced first-order moments. Theoretically, we show that considering higher orders of factorial moments introduces richer statistical information and yields a tighter solution space to facilitate classifier training, which is in accord with our empirical results. Extensive comparative experiments show that moment matching remains effective under low computational cost, achieves state-of-the-art performance across various benchmark datasets, and is robust to different bag generation strategies.

Our contributions can be summarized as follows:

• We investigate the structured-bag assumption where the grouping strategy is dependent on examples. This assumption is more practical than the existing LLP data generation processes where bags are formed uniformly at random. Based on this assumption, we show that the bag-level label statistics follow the PMD, which offers a fresh perspective about LLP data.

• Motivated by the PMD structure, we propose moment matching, a simple and general framework that trains classifiers by matching classifier-induced moments with empirical multi-order factorial moments derived from label proportions. Our framework includes proportion matching, a widely used LLP strategy, as the special case of the first-order moments.

• Extensive experiments demonstrate that moment matching achieves strong performance on different benchmark datasets. Besides, moment matching is robust to different bag generation strategies and attains state-of-the-art results under practical computational budgets.

## 2 Preliminaries

In this section, we introduce the notation and background of multi-class classification and learning from label proportions.

## 2.1 Ordinary Multi-class Classification

Let X denote the input space and $\mathcal { V } : = \{ 1 , \ldots , q \}$ denote the label space with $q$ class labels. We assume that examples are sampled with probability density $p ( x , y )$ over $\boldsymbol { \mathcal { X } } \times \boldsymbol { \mathcal { Y } }$ . The goal of multi-class classification is to learn a score-based predictor $\pmb { f } \in \bar { \mathcal { F } }$ in a hypothesis class ${ \mathcal { F } } _ { : }$ , where $f : \mathcal { X } \to \Delta ^ { q - 1 }$ maps each input to the $( q - 1 )$ -dimensional simplex

$$
\Delta ^ { q - 1 } : = \left\{ ( p _ { 1 } , \dotsc , p _ { q } ) \in \mathbb { R } ^ { q } \ \middle | \ p _ { k } \geq 0 , \sum _ { k = 1 } ^ { q } p _ { k } = 1 \right\} .
$$

Here, $f _ { k } ( x )$ can be interpreted as the predicted probability of class k for input x.

Since direct minimization of the empirical 0-1 loss is computationally intractable for many natural hypothesis classes [Ben-David et al., 2003], we often instead use a classification-calibrated surrogate loss $\ell : \Delta ^ { q - 1 } \times \mathcal { Y }  \mathbb { R } _ { + }$ [Bartlett et al., 2006] and an empirical approximation of the following classification risk:

$$
R ( \pmb { f } ) : = \mathbb { E } _ { ( x , y ) \sim p ( x , y ) } [ \ell ( \pmb { f } ( x ) , y ) ] .\tag{1}
$$

Under proper losses such as the cross-entropy loss or the mean squared error loss, the population-risk minimizer recovers the class-posterior distribution. We denote this Bayes-optimal predictor by $f ^ { * }$ where

$$
\pmb { f } _ { k } ^ { * } ( x ) = p ( y = k \mid x ) , \quad \forall k \in \mathcal { V } , \forall x \in \mathcal { X } .\tag{2}
$$

Lemma 2.1. According to Yu et al. $I 2 0 I 8 J , i f \ell$ is the cross-entropy loss or the mean squared error loss, then any optimal predictor $f ^ { \star } \in$ arg min $\pmb { f } { \colon } \chi \smash { \to } \Delta ^ { q - 1 }  { \operatorname { R } } ( f )$ satisfies $f ^ { \star } = f ^ { * }$

## 2.2 Learning from Label Proportions

The learning objective in LLP is the same as that in ordinary multi-class classification: we still aim to learn an instance-level predictor $f : \mathcal { X } \to \Delta ^ { q - 1 }$ The difference is that supervision is provided only at the bag level. Instead of observing a label for each individual example, the training data consist of groups of examples, referred to as $b a g s$ , together with the class proportions of the labels in each bag [Yu et al., 2014]. Following Busa-Fekete et al. [2023, 2025], we assume for simplicity that every bag contains the same number m of examples. Formally, a bag is denoted by $ { \mathbf { B } } \stackrel { \cdot } { = } ( x _ { 1 } , \cdot \cdot \cdot , x _ { m } ) \stackrel { \cdot } { \in } \chi ^ { m }$ , where each $x _ { j }$ is an unlabeled example. We assume that each instancelabel pair $( x _ { i } , y _ { i } )$ is drawn i.i.d. with probability density $p ( x , y )$ . These instances are then assembled into bags by an unknown bag-construction mechanism, which may depend on the instance features or other latent grouping factors and is not restricted to uniform random assignment. We assume that, conditional on the bag $\mathbf { B } = ( x _ { 1 } , \dots , x _ { m } )$ , the labels $y _ { 1 } , \ldots , y _ { m }$ are mutually independent, with $p ( y _ { j } \mid \mathbf { B } ) = p ( y _ { j } \mid x _ { j } )$ for every $j \in [ m ]$ . Under this assumption, the conditional probability of the hidden label sequence given the bag can be factorized as

$$
p ( \pmb { y } \mid \mathbf { B } ) = \prod _ { j = 1 } ^ { m } p ( y _ { j } \mid \mathbf { B } ) = \prod _ { j = 1 } ^ { m } p ( y _ { j } \mid x _ { j } ) .\tag{3}
$$

Instead of observing individual labels, we observe only the class-proportion vector ${ \pmb { \alpha } } = ( \alpha _ { 1 } , \ldots , \alpha _ { q } )$ For each class $k \in \mathcal { V } , \alpha _ { k } : = \textstyle \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { I } \{ y _ { j } = k \}$ is the proportion of examples in the bag that belong to class $k .$ Here, I{·} denotes the indicator function, which equals 1 if the condition is true and 0 otherwise. This induces an unknown density $p ( \mathbf { B } , \alpha )$ over bags, and we model the observed bags as i.i.d. draws from this bag distribution. The posterior probability of α is obtained by marginalizing over all label sequences consistent with α:

$$
p ( \alpha \mid \mathbf { B } ) = \sum _ { \pmb { y } \in \mathscr { y } ^ { m / \alpha } } p ( \pmb { y } \mid \mathbf { B } ) ,\tag{4}
$$

where $\mathcal { y } ^ { m / \alpha }$ denotes the set of all label sequences satisfying the proportion constraints:

$$
\mathcal { V } ^ { m / \alpha } = \left\{ \pmb { y } \in \mathcal { Y } ^ { m } \bigg | \forall k \in \mathcal { V } , \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { I } \{ y _ { j } = k \} = \alpha _ { k } \right\} .\tag{5}
$$

Accordingly, an LLP training dataset $\{ ( \mathbf { B } _ { i } , \pmb { \alpha } _ { i } ) \} _ { i = 1 } ^ { n }$ consists of n i.i.d. examples of bag-proportion pairs drawn from the induced joint density $p ( \mathbf { B } , \pmb { \alpha } )$ A common way to learn from such bagproportion supervision is proportion matching [Yu et al., 2014, Ardehaly and Culotta, 2017]: given a predictor $f ,$ one averages its instance-level predictions over each bag and encourages the resulting predicted class proportion to match the observed proportion α.

## 3 Methodology

In this section, we introduce the proposed moment matching approach. We first establish the connection between LLP and the PMD, and show that the popular proportion matching approach can be considered as the first-order moment matching. Then, we present the second-order and k-th-order moment matching. Finally, we present the theoretical analysis for the proposed approach.

## 3.1 Rethinking Label Proportions: A PMD Perspective

We show the connection between LLP and the PMD under certain assumptions about the data generation process. We denote by $e ^ { y _ { j } } \in \{ 0 , 1 \} ^ { q }$ the one-hot representation of label $y _ { j } .$ . Conditioned on $x _ { j }$ , this one-hot vector can be equivalently viewed as a draw from a categorical distribution with parameter $p ( \cdot \mid x _ { j } )$ , namely, $e ^ { y _ { j } } \sim$ Categorica $. ( p ( \cdot \mid x _ { j } ) )$ . We further define the bag-level label count vector by $\begin{array} { r } { \dot { z } : = m \alpha = \sum _ { j = 1 } ^ { m } e ^ { y _ { j } } } \end{array}$ . Since $e ^ { y _ { j } }$ are independent categorical random vectors conditioned on $\mathbf { B } = ( x _ { 1 } , \dots , x _ { m } )$ , the count vector z follows a PMD [Daskalakis et al., 2015]. A straightforward strategy is to directly maximize the likelihood of the PMD in Equation (4). However, exact computation of the likelihood is computationally challenging. In particular, even the state-of-the-art method in computational statistics [Lin et al., 2023] has a time complexity of $O ( m ^ { q - 1 } \log m )$ . In typical LLP experimental settings, e.g., $m = 6 4$ and $q = 1 0 \mathrm { [ M a }$ et al., 2026b], this complexity is already astronomically large and therefore impractical for training. Instead of maximum likelihood, we leverage moment-based information and train the classifier by matching classifier-induced moments to empirical multi-order factorial moments.

## 3.2 Label Proportions Beyond First-Order Moments: A Second-Order Illustration

The classical algorithm Proportion Matching [Yu et al., 2014, Ardehaly and Culotta, 2017] can be viewed as a special case of our framework that performs first-order moment matching, and therefore fails to exploit higher-order information; see Appendix B for details. Under the PMD, we provide an

effective solution for exploiting richer information in label proportions through higher-order moments. As an illustration, we consider the following second-order empirical moment matrix:

$$
\widehat { \mathbf { M } } _ { \alpha } ^ { ( 2 ) } = \frac { 1 } { m ( m - 1 ) } \sum _ { i \neq j } e ^ { y _ { i } } ( e ^ { y _ { j } } ) ^ { \top } ,\tag{6}
$$

where each entry $\left[ \widehat { \mathbf { M } } _ { \alpha } ^ { ( 2 ) } \right] _ { c , d }$ represents the normalized number of ordered instance index pairs $( i , j )$ with $i \neq j , y _ { i } = c ,$ , and $y _ { j } = d$ . Since the number of instances in class c is $m \alpha _ { c } .$ , this quantity can be expressed entirely in terms of α: when $c \neq d ,$ there are $m \alpha _ { c } m \alpha _ { d }$ such ordered pairs, while when $c = d ,$ there are $m \alpha _ { c } ( m \alpha _ { c } - 1 )$ , since the same instance cannot be counted twice. This yields

$$
\left[ \widehat { \mathbf { M } } _ { \alpha } ^ { ( 2 ) } \right] _ { c , d } = \frac { m \alpha _ { c } ( m \alpha _ { d } - \mathbb { I } \{ c = d \} ) } { m ( m - 1 ) } , \qquad c , d \in \{ 1 , \ldots , q \} .\tag{7}
$$

The detailed derivation is provided in Appendix C. For a bag B, the classifier-induced second-order moment matrix is given by

$$
{ \widehat { \bf M } } _ { f } ^ { ( 2 ) } ( { \bf B } ) = { \frac { 1 } { m ( m - 1 ) } } \sum _ { i \neq j } { f ( x _ { i } ) { \cal f } ( x _ { j } ) } ^ { \top } .\tag{8}
$$

Accordingly, $\widehat { \mathbf { M } } _ { \alpha } ^ { ( 2 ) }$ can serve as the target for the classifier-induced second-order moment. Following the objective of first-order moment matching, we minimize the second-order moment matching risk

$$
\mathbb { E } _ { p ( \mathbf { B } , \alpha ) } \Big [ \mathcal { L } ^ { ( 2 ) } \Big ( \widehat { \mathbf { M } } _ { \alpha } ^ { ( 2 ) } , \widehat { \mathbf { M } } _ { f } ^ { ( 2 ) } ( \mathbf { B } ) \Big ) \Big ] ,\tag{9}
$$

where $\mathcal { T } _ { r }$ denotes the set of order-r tensors with nonnegative entries summing to one, and $\boldsymbol { \mathcal { L } } ^ { ( 2 ) }$ $\mathcal { T } _ { 2 } \times \mathcal { T } _ { 2 } \stackrel { \cdot } {  } [ 0 , + \infty ]$ measures the discrepancy between the second-order empirical moment and the second-order classifier-induced moment.

## 3.3 Generalized Multi-Order Moment Matching

After the illustrative second-order example above, we now generalize the construction to arbitrary orders $r \leq m$ , and describe how such multi-order information can be derived from label proportions and exploited for training. Following Johnson et al. $[ 1 9 9 7 ]$ and Ng et al. [2011], we introduce the normalized r-th order factorial moment tensor $\widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } \in ( \mathbb { R } ^ { q } ) ^ { \otimes r }$ as

$$
\widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } : = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } e ^ { y _ { j _ { 1 } } } \otimes e ^ { y _ { j _ { 2 } } } \otimes \cdot \cdot \cdot \otimes e ^ { y _ { j _ { r } } } ,\tag{10}
$$

where $( m ) _ { r } : = m ( m { - } 1 ) \cdot \cdot \cdot ( m { - } r { + } 1 )$ is the falling factorial, $[ m ] _ { \neq } ^ { r } : = \{ ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] ^ { r } \mid j _ { a } \neq $ $j _ { b } , \forall a \ne b \}$ denotes the set of ordered r-tuples of distinct instance indices, and ⊗ denotes the tensor product, so that $e ^ { y _ { j _ { 1 } } } \otimes \cdot \cdot \cdot \otimes e ^ { y _ { j _ { \prime } } }$ is an order-r tensor in $( \mathbb { R } ^ { q } ) ^ { \otimes r }$ . Notably, the second-order moment matrix in Equation (6) is exactly the special case of Equation (10) with $r = 2$ . Although the individual labels are not observed, the normalized r-th order factorial moment tensor can still be computed entirely from the label proportion vector α. To see this, consider any index tuple $( i _ { 1 } , \ldots , i _ { r } ) \bar { \in } [ q ] ^ { r }$ Define $\begin{array} { r } { \pmb { k } = \left( k _ { 1 } , \dots , \hat { k } _ { q } \right) ^ { \circ } \in \mathbb { N } ^ { q } , k _ { c } : = \# \{ \ell \in [ r ] \mid i _ { \ell } = c \} , c = 1 , \dots , \hat { q } . } \end{array}$ . Then k records the multiplicity of each class in $( i _ { 1 } , \ldots , i _ { r } )$ , and satisfies $\textstyle \sum _ { c = 1 } ^ { q } k _ { c } = r .$ . In other words, $k _ { c }$ records how many times class c appears in the tuple $( i _ { 1 } , \ldots , i _ { r } )$ . Then the $( i _ { 1 } , \ldots , i _ { r } )$ -th entry of $\widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) }$ is given by

$$
\left[ \widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } \right] _ { i _ { 1 } , \ldots , i _ { r } } = \frac { 1 } { ( m ) _ { r } } \prod _ { c = 1 } ^ { q } ( z _ { c } ) _ { k _ { c } } = \frac { m ^ { r } } { ( m ) _ { r } } \prod _ { c = 1 } ^ { q } \prod _ { t = 0 } ^ { k _ { c } - 1 } \left( \alpha _ { c } - \frac { t } { m } \right) .\tag{11}
$$

Given a bag B, for a classifier $f ,$ we denote by $p _ { i } : = \pmb { f } (  { \boldsymbol { { x } } } _ { i } ) \in \Delta ^ { q - 1 }$ the predicted class-probability vector of instance $x _ { j }$ . The classifier-induced normalized r-th order factorial moment tensor is then defined as

$$
\widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) : = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \ldots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } p _ { j _ { 1 } } \otimes p _ { j _ { 2 } } \otimes \cdot \cdot \cdot \otimes p _ { j _ { r } } .\tag{12}
$$

Therefore, $\widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) }$ computed from the observed label proportions can serve as the target tensor for $\widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } )$ , yielding a natural multi-order matching objective. In our work, we adopt a simple and direct strategy: we use the factorial moment tensor computed from the observed label proportions as the target, and encourage the classifier-induced moment tensor to match it at each order. Accordingly, the expected multi-order matching risk can be defined as

$$
R _ { \mathrm { M M } } ^ { ( s ) } ( { \pmb f } ) : = \mathbb { E } _ { p ( \mathbf { B } , \pmb \alpha ) } \left[ \sum _ { r = 1 } ^ { s } \lambda _ { r } \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \pmb { \alpha } } ^ { ( r ) } , \widehat { \mathbf { M } } _ { \pmb { f } } ^ { ( r ) } ( \mathbf { B } ) \Big ) \right] ,\tag{13}
$$

where $s \in \{ 1 , \ldots , m \}$ is the maximum order of moments used for training, $\lambda _ { r } > 0$ is the weight associated with the r-th order moment, and $\textstyle \sum _ { r = 1 } ^ { s } \lambda _ { r } \ = \ 1$ . It is immediate that the PM risk in Equation (19) is recovered as the special case $\bar { s } = 1$ of the above objective. Here, for each $r \in \{ 1 , \ldots , s \}$ , the loss function $\mathcal { L } ^ { ( r ) } : \mathcal { T } _ { r } \times \mathcal { T } _ { r }  [ 0 , + \infty ]$ measures the discrepancy between the target r-th order factorial moment tensor $\widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) }$ and the classifier-induced tensor $\widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } )$ . The empirical counterpart of $R _ { \mathrm { M M } } ^ { ( s ) } ( f )$ on a dataset $\{ ( \mathbf { B } _ { i } , \pmb { \alpha } _ { i } ) \} _ { i = 1 } ^ { n }$ is given by

$$
\widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \sum _ { r = 1 } ^ { s } \lambda _ { r } \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \alpha _ { i } } ^ { ( r ) } , \widehat { \mathbf { M } } _ { \pmb { f } } ^ { ( r ) } ( \mathbf { B } _ { i } ) \Big ) \right] .\tag{14}
$$

Since the normalized factorial moment tensor has nonnegative entries that sum to one, it can be viewed, after vectorization, as a probability distribution over ordered class tuples $( c _ { 1 } , \ldots , c _ { r } ) \in \left\{ 1 , \ldots , q \right\} ^ { r }$ Therefore, matching two order-r factorial moment tensors is equivalent to matching two distributions over such class tuples, and $\boldsymbol { \mathcal { L } ^ { ( r ) } }$ can be instantiated by standard distributional discrepancies, such as mean squared error or cross-entropy.

## 3.4 Theoretical Analysis

For each bag B, let $\widehat { \mathbf { M } } _ { f ^ { * } } ^ { ( r ) } ( \mathbf { B } )$ denote the r-th order classifier-induced moment tensor associated with the Bayes-optimal classifier $f ^ { * }$ . Let $\mathcal { F } _ { \mathrm { M M } } ^ { ( s ) }$ denote the set of minimizers of Equation (13). We then have the following proposition:

Proposition 3.1. Assume that the hypothesis class F contains the Bayes-optimal classifier $f ^ { * }$ . For each $r \in \{ 1 , \ldots , s _ { 2 } \}$ , assume that for any T -valued random variable $\bar { Z , }$ , its expectation $\operatorname { \mathbb { E } } [ Z ]$ is the unique solution ofmin $v \in { \mathcal { T } } _ { r } \operatorname { \mathbb { E } } \left\lceil { \mathcal { L } } ^ { ( r ) } ( Z , v ) \right\rceil$ . Assumefurther that $\lambda _ { r } > 0 f o r$ all $r \in \{ 1 , \ldots , s _ { 2 } \}$ Then, for any $1 \leq s _ { 1 } < s _ { 2 } \leq m _ { \mathrm { \Omega } }$ , we have $\mathcal { F } _ { \mathrm { M M } } ^ { ( \bar { s } _ { 2 } ) } \subseteq \mathcal { F } _ { \mathrm { M M } } ^ { ( s _ { 1 } ) }$ . That is, using more orders offactorial moments yields a tighter minimizer set.

Proposition 3.1 partially explains the advantage of our method. By incorporating multi-order factorial moments as supervision, we impose progressively stronger constraints on the feasible minimizers. As a result, the solution set becomes tighter while still containing the Bayes-optimal classifier $f ^ { * }$

Theorem 3.2 (Generalization bound). Assume that the bag coordinates are exchangeable. Assume that,for each $r \in \{ 1 , \ldots , s \}$ , the loss $\boldsymbol { \mathcal { L } } ^ { ( r ) } ( \boldsymbol { A } , \cdot )$ is $L _ { r } – L i p s c h i t z$ with respect to the Frobenius norm, uniformly over all admissible target tensors $A ,$ and satisfies $0 \leq \mathcal { L } ^ { ( r ) } ( A , B ) \leq B _ { r }$ for all admissible $A , { \bar { B } } .$ . Then,for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \big | R _ { \mathrm { M M } } ^ { ( s ) } ( f ) - \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( f ) \big | \leq 2 \sqrt { 2 m } \left( \sum _ { r = 1 } ^ { s } \lambda _ { r } L _ { r } r \right) \left( \sum _ { c = 1 } ^ { q } \mathfrak { R } _ { n } ( \mathcal { F } _ { c } ) \right) + \left( \sum _ { r = 1 } ^ { s } \lambda _ { r } B _ { r } \right) \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .\tag{15}
$$

Under the current bag-i.i.d. assumption, our analysis establishes the optimality of matching the Bayesoptimal classifier-induced moments. Although this result does not by itself yield a full classifier-level consistency guarantee of the form $f \to f ^ { * }$ , it provides a theoretical justification for the proposed moment-matching principle and, together with the empirical results in Section 4, suggests that the proposed approach is effective in practice. Establishing such consistency remains challenging for real-world structured bags [Bortsova et al., 2018, Sun et al., 2017, Pacheco et al., 2024], such as those in blood-cell detection, where shared, unobserved physiological factors may induce donor-specific correlations among cells, making the within-bag dependence structure difficult to characterize. We leave this problem for future work.

For computing the loss, a naive approach would explicitly construct the full order-r moment tensor, whose number of entries scales as $\bar { q } ^ { r }$ . To avoid this dependence, we use a state-compressed dynamic programming method that reduces both time and memory to depend on the number of compressed histogram states. For example, in the second-order case, the moment tensor is a symmetric matrix, and therefore one only needs to compute its distinct entries rather than all $q ^ { 2 }$ ordered entries. The details are provided in Appendix F.2. The resulting complexity is summarized below.

Proposition 3.3 (Complexity of the state-compressed computation). Fix a bag of size m and a maximum order $s \ \leq \ m$ . The compressed computation of all moments up to order s requires $O \big ( m q \big ( \begin{array} { c } { { q + s } } \\ { { s } } \end{array} \big ) \big )$ time $a n d O \Big ( \Big ( \overset { q + s } { s } \Big ) \Big )$ memory per bag.

Although this complexity still grows combinatorially with the maximum order $s ,$ in practice only a small number of low-order moments is sufficient to achieve strong performance.

## 4 Experiments

## 4.1 Experimental Setup

## 4.1.1 Datasets and Bag Construction

Following prior work [Busa-Fekete et al., 2023, 2025], we construct synthetic LLP datasets on three standard benchmarks: CIFAR-10, CIFAR-100 [Krizhevsky and Hinton, 2009], and miniImageNet [Vinyals et al., 2016]. Following Ma et al. [2026b], we evaluate three bag generation strategies, namely random bags, cluster bags, and α-first bags, with bag sizes $m \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ . Existing LLP benchmarks are mainly designed for binary classification [Brahmbhatt et al., 2024, Franco et al., 2023], and are therefore not well aligned with our multi-class classification setting. The detailed construction procedures are provided in Appendix G.1.

## 4.1.2 Baselines and Implementation Details

We compare LLP-MM with representative LLP baselines, including PM [Yu et al., 2014, Ardehaly and Culotta, 2017], DSQ [Li et al., 2024], LLP-PVC [Ma et al., 2026b], LLP-FC [Zhang et al., 2022], ROT [Dulac-Arnold et al., 2019], EasyLLP [Busa-Fekete et al., 2023], GeneralUPM [Applebaum et al., 2025], EasyLLP-ABS, GeneralUPM-ABS, and FlowLLP [Rakotomamonjy et al., 2026]. EasyLLP-ABS and GeneralUPM-ABS take the absolute value of the original loss to reduce the effect of negative loss values. For the main benchmark comparisons, LLP-MM uses $s = 8$ for CIFAR-10, and s = 3 for CIFAR-100 and miniImageNet. For each order r, we use the cross-entropy loss as $\boldsymbol { \mathcal { L } } ^ { ( r ) }$ and set the weight $\lambda _ { r } \operatorname { t o } 1 / s$ . Further implementation details are provided in Appendix G.2. We also evaluate on patient-grouped blood cell data in Section 4.3.

## 4.2 Comparison with Baselines

LLP-MM ranks first in average accuracy across all nine dataset–bag-construction settings in Tables 1 and 2. DSQ and LLP-FC perform best in some larger-bag settings. EasyLLP and GeneralUPM can suffer from negative empirical losses during optimization. LLP-PVC is competitive on CIFAR-10 but has lower average accuracy on CIFAR-100 and miniImageNet. This gap may partly reflect its one-vs-rest decomposition, which creates increasingly imbalanced binary tasks as the number of classes grows and does not directly model the joint multiclass PMD. FFT-based evaluation of very small Poisson-binomial probabilities may also affect likelihood optimization through roundoff and cancellation errors [Peres et al., 2020].

## 4.3 Semi-Real-World Evaluation

Inspired by the blood-cell LLP setting of Pacheco et al. [2024], whose real data are not publicly available, we construct a semi-real-world LLP benchmark on the public 13-class KU-Optofil PBC dataset [Yarıkan et al., 2026]. We retain its official patient-separated split, merge training and validation cells, and form bags by patient ID, with synthetic subbags for missing IDs. Bag proportions are computed from cell labels. Table 3 reports three-run results, selecting each run’s checkpoint by the highest test Macro-F1. Details are in Appendix G.4.

Table 1: Best test accuracy under Random Bag and Cluster Bag constructions (mean ± std over three runs, %). The best result for each bag size and average is bold. Ranks use average accuracy with standard competition ranking. † indicates severe overfitting caused by negative loss values, with predictions collapsing to a single class.
<table><tr><td rowspan=1 colspan=7>Dataset    Method               16           32           64           128      Average Rank</td></tr><tr><td rowspan=1 colspan=7>Random Bag</td></tr><tr><td rowspan=1 colspan=7>PM              93.51 ± 0.11   90.05 ± 0.22   75.23 ± 0.70   60.02 ± 0.33    79.70    4327561.90LLP-PVC91.95± 0.0790.55± 0.0883.79± 0.5764.44± 1.4582.68LLP-FC75.29± 0.7261.06± 0.8753.89± 0.6160.01± 0.2462.56DSQ             93.52± 0.15   91.39± 0.16   79.86 ± 0.77       ± 1.35   81.67</td></tr><tr><td rowspan=1 colspan=7>ROT             93.211 ± 0.09   92.04±0.10   80.52±0.49   35.03±1.25   75.20</td></tr><tr><td rowspan=1 colspan=7>CIFAR-10    EasyLLP          32.999 ± 0.80   31.98± 2.69   26.08± 0.21   26.40± 0.98   29.36    11EasyLLP-ABS     57.26 ± 2.36   50.54±2.77   41.86± 3.53   33.80± 1.78   45.87    9</td></tr><tr><td rowspan=2 colspan=7>GeneralUPM      45.43 ± 1.05   36.98 ±3.42   29.80 ± 1.21   27.20土0.68   34.85    10</td></tr><tr><td rowspan=1 colspan=6></td></tr><tr><td rowspan=1 colspan=7>LLP-MM         93.97 ± 0.12   91.99± 0.08   84.13 ± 0.57   66.74±0.20   84.21    1</td></tr><tr><td rowspan=2 colspan=7>PM              71.12 ± 0.26   66.622 ± 0.05   54.34 ± 0.87   17.69± 0.57   52.44    34627DSQ71.01 ± 0.3465.78± 0.2645.95± 0.8625.70生0.8052.11LLP-PVC63.533 ± 0.2161.733 ± 0.4749.78 ± 2.7812.13± 1.0046.79</td></tr><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=3>LLP-FC</td><td rowspan=1 colspan=4>71.77 ± 0.08   67.100 ± 0.18   54.84± 0.84   17.39± 0.55   52.77</td></tr><tr><td rowspan=2 colspan=3>ROTCIFAR-100   EasyLLP</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=4>2.43 ± 0.44    1.95 ± 0.27    1.34 ± 0.20    1.34 ±0.05    1.77    11</td></tr><tr><td rowspan=1 colspan=7>EasyLLP-ABS      17.66 ± 1.73   12.399 ± 1.44   9.62 ± 0.01    6.50 ± 0.41    11.55    9</td></tr><tr><td rowspan=3 colspan=7>GeneralUPM       12.68 ± 0.33   8.36 ± 0.30    5.65±1.05    4.07±0.13    7.69    10GeneralUPM-ABS  24.00 ± 1.57   16.51± 1.61   11.33± 0.47    7.23± 0.40    14.77    851FlowLLP66.76 ± 0.0651.683 ± 0.9112.04± 2.0550.49LLP-MM72.06 ± 0.1067.14 ± 0.0557.79 ± 0.6423.82±0.5455.2071.49 ± 0.33</td></tr><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=2 colspan=7>PM              64.76 ± 0.40   56.14 ± 0.35   38.31 ± 1.06   10.01± 2.04   42.30    34663.38 ± 0.1331.4514.81LLP-PVC57.71 ± 0.1454.633 ± 0.3424.19 ± 1.575.21± 0.4335.44DSQ                            53.73± 0.40       ± 0.58        ± 0.81   40.84</td></tr><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=4 colspan=3>LLP-FCROTminiImageNet  EasyLLP</td><td rowspan=1 colspan=1>65.75 ± 0.05</td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>67.577 ± 0.82</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3>58.133 ± 0.17   39.40 ± 1.75   11.05± 1.72    43.58    2741.500 ± 6.107.21 ± 0.412.64±0.5729.73</td></tr><tr><td rowspan=1 colspan=1>1.92 ± 0.13</td><td rowspan=1 colspan=1>1.59 ± 0.15    1.51 ± 0.08</td><td rowspan=1 colspan=2>1.36 ± 0.09    1.59    11</td></tr><tr><td rowspan=3 colspan=7>EasyLLP-ABS                                                  3.57±0.28    7.24     9GeneralUPM       9.44 ± 0.11    6.08 ± 0.46   4.00 ± 0.25    2.91 ± 0.33    5.61    10GeneralUPM-ABS  19.611 ± 0.40                  7.79 ± 0.23   4.29 ± 0.52    10.87    851FlowLLP62.83 ± 0.86LLP-MM65.94 ± 0.0359.79 ± 0.1444.76 ± 0.298.49 ± 1.0844.74</td></tr><tr><td rowspan=1 colspan=1>9.44 ± 0.11</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>19.61 ± 0.40</td><td rowspan=1 colspan=3>52.81 ± 1.77 28.13 ±+ 0.234 47.20 ± 0.52 37.77 5</td></tr><tr><td rowspan=1 colspan=7>Cluster Bag</td></tr><tr><td rowspan=2 colspan=3>PMDSQ</td><td rowspan=1 colspan=1>93.31 ± 0.19</td><td rowspan=1 colspan=3>91.29 ± 0.11   86.30 ± 0.45   72.94 ± 1.64   85.96</td></tr><tr><td rowspan=1 colspan=1>93.62 ± 0.19</td><td rowspan=1 colspan=3>91.97 ± 0.44   87.22 ± 0.53   64.65± 2.56   84.37</td></tr><tr><td rowspan=2 colspan=3>LLP-PVCLLP-FC</td><td rowspan=1 colspan=1>92.30 ± 0.06</td><td rowspan=1 colspan=2>90.466 ± 0.24   85.299 ± 0.69   64.09± 2.06</td><td rowspan=1 colspan=1>83.04</td></tr><tr><td rowspan=1 colspan=1>56.22 ± 1.47</td><td rowspan=1 colspan=2>39.49 ± 0.69   34.299 ± 0.19   72.65± 1.04</td><td rowspan=1 colspan=1>50.66</td></tr><tr><td rowspan=1 colspan=3>ROT</td><td rowspan=1 colspan=1>93.53 ± 0.07</td><td rowspan=1 colspan=2>92.05± 0.10   87.14 ± 0.15   59.96 ± 2.21</td><td rowspan=1 colspan=1>83.17</td></tr><tr><td rowspan=4 colspan=3>CIFAR-10   EasyLLPEasyLLP-ABSGeneralUPMGeneralUPM-ABSFlowLLPLLP-MM</td><td rowspan=1 colspan=1>10.90 ± 1.25 †</td><td rowspan=1 colspan=2>10.90 ± 1.26 †  10.87 ± 1.20 †  10.78 ± 1.16 †</td><td rowspan=1 colspan=1>10.86 †    10</td></tr><tr><td rowspan=1 colspan=1>10.90 ± 1.25 †</td><td rowspan=1 colspan=3>10.90 ± 1.26 †  10.87 ± 1.20 †  10.78 ± 1.16 †   10.86†    10</td></tr><tr><td rowspan=1 colspan=1>93.16 ± 0.16</td><td rowspan=1 colspan=3>91.20 ± 0.17   83.96 ± 0.28   69.56± 1.10   84.47</td></tr><tr><td rowspan=1 colspan=1>94.04 ± 0.20</td><td rowspan=1 colspan=3>92.33 ± 0.10   89.12 ± 0.43   75.19 ± 2.14   87.67</td></tr><tr><td rowspan=2 colspan=3>PMDSQ</td><td rowspan=1 colspan=1>71.35 ± 0.20</td><td rowspan=1 colspan=1>66.21 ± 0.68   57.78 ± 0.72</td><td rowspan=1 colspan=2>46.06 ± 1.02   60.35</td></tr><tr><td rowspan=1 colspan=1>72.33 ± 0.15</td><td rowspan=1 colspan=1>67.599 ± 1.04   54.33 ± 2.31</td><td rowspan=1 colspan=1>35.59± 1.86</td><td rowspan=1 colspan=1>57.46</td></tr><tr><td rowspan=1 colspan=3>LLP-PVCLLP-FC</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=1>71.44 ± 0.51</td><td rowspan=1 colspan=1>66.599 ± 0.50   58.444 ± 1.02</td><td rowspan=1 colspan=1>46.19 ± 0.66</td><td rowspan=1 colspan=1>60.66</td></tr><tr><td rowspan=2 colspan=3>CIFAR-100   EasyLLPEasyLLP-ABS</td><td rowspan=1 colspan=1>0.90 ± 0.02 †</td><td rowspan=1 colspan=1>0.90 ± 0.02 †   0.90 ± 0.02 †</td><td rowspan=1 colspan=1>0.90 ± 0.02 †</td><td rowspan=1 colspan=1>0.90†    10</td></tr><tr><td rowspan=1 colspan=1>7.07 ± 0.29</td><td rowspan=1 colspan=3>4.95 ± 0.41    3.22 ± 0.35    2.54 ± 0.27    4.44     8</td></tr><tr><td rowspan=1 colspan=3>GeneralUPM</td><td rowspan=1 colspan=1>0.90 ± 0.02 †</td><td rowspan=1 colspan=3>0.90 ± 0.02 †   0.90 ± 0.02 †   0.90 ± 0.02 †   0.90†    10</td></tr><tr><td rowspan=2 colspan=3>GeneralUPM-ABSFlowLLP</td><td rowspan=1 colspan=1>7.14 ± 0.36</td><td rowspan=1 colspan=3>4.74 ± 0.06    3.26 ± 0.29    2.56 ± 0.25    4.42</td></tr><tr><td rowspan=1 colspan=1>70.41 ± 0.11</td><td rowspan=1 colspan=3>65.12 ± 0.07   55.15 ± 0.39   42.62 ± 0.52   58.32</td></tr><tr><td rowspan=1 colspan=7>LLP-MM         72.24 ± 0.67   67.89 ± 0.67   58.84 ± 0.65   45.66 ± 0.50   61.16    1</td></tr><tr><td rowspan=3 colspan=3>PMDSQLLP-PVC</td><td rowspan=1 colspan=3>65.21 ± 0.76   56.37 ± 0.10   44.06 ± 1.78   30.18 ± 0.90</td><td rowspan=1 colspan=1>48.95</td></tr><tr><td rowspan=1 colspan=3>66.03 ± 0.27   55.79 ± 1.36   37.02 ± 2.75   19.56 ± 0.15</td><td rowspan=1 colspan=1>44.60</td></tr><tr><td rowspan=1 colspan=2>58.77 ± 0.39   52.50 ± 1.10   27.91 ± 8.11</td><td rowspan=1 colspan=1>15.31 ± 1.02</td><td rowspan=1 colspan=1>38.62</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>LLP-FC</td><td rowspan=1 colspan=2>65.53 ± 0.62   56.411 ± 0.61   44.266 ± 1.01</td><td rowspan=1 colspan=1>30.300 ± 1.07</td><td rowspan=1 colspan=1>49.13</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>ROT</td><td rowspan=1 colspan=2>67.29 ± 0.40   58.16 ± 0.87   33.94 ± 1.89</td><td rowspan=1 colspan=1>17.83 ± 1.32</td><td rowspan=1 colspan=1>44.30</td></tr><tr><td rowspan=1 colspan=2>miniImageNet</td><td rowspan=1 colspan=1>EasyLLP</td><td rowspan=1 colspan=3>1.04 ± 0.05 †   1.01 ± 0.01 †   1.01 ± 0.01 †   1.01 ± 0.01 †</td><td rowspan=2 colspan=1>1.01 †    103.45    8</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>EasyLLP-ABS</td><td rowspan=1 colspan=3>5.31 ± 0.07    3.38 ± 0.33    2.86 ± 0.12    2.27 ± 0.05</td></tr><tr><td rowspan=4 colspan=3>GeneralUPMGeneralUPM-ABSFlowLLPLLP-MM</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=1>GeneralUPM-ABS</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=4>1.04 ± 0.05 †   1.01 ± 0.01 †   1.01 ± 0.01 †   1.01 ± 0.01 †    1.01 †    105.31 ± 0.07    3.46 ± 0.26   2.78 ± 0.19   2.19 ± 0.07    3.44    96161.86 ± 0.8851.56 ± 0.4337.44 ± 0.4525.67 ± 0.5344.1366.99 ± 0.6859.65 ± 0.9745.91 ± 1.6328.36 ± 0.7150.23</td></tr><tr><td rowspan=1 colspan=3></td></tr></table>

Table 2: Best test accuracy under α-First Bag construction (mean ± std over three runs, %). The best result for each bag size and average is bold. Ranks use average accuracy with standard competition ranking.
<table><tr><td>Dataset</td><td>Method</td><td>16</td><td>32</td><td>64</td><td>128</td><td>Average</td><td>Rank</td></tr><tr><td rowspan="10">CIFAR-10</td><td>PM</td><td> $9 1 . 0 0 \pm 0 . 4 0$ </td><td> $7 0 . 3 7 \pm 0 . 3 2$ </td><td> $6 5 . 3 8 \pm 1 . 1 4$ </td><td> $6 1 . 3 5 \pm 0 . 5 2$ </td><td>72.03</td><td>4</td></tr><tr><td>DSQ</td><td> $9 0 . 9 8 \pm 0 . 1 6$ </td><td> $7 4 . 2 4 \pm 0 . 5 7$ </td><td> $7 3 . 7 8 \pm 1 . 1 1$ </td><td> $6 8 . 8 3 \pm 1 . 4 3$ </td><td>76.96</td><td>3</td></tr><tr><td> $\mathrm { L L P { \bar { - } P V C } }$ </td><td> $9 1 . 8 5 \pm 0 . 1 3$ </td><td> $8 9 . 1 8 \pm 0 . 1 1$ </td><td> $\mathbf { 8 1 . 4 4 \ : \pm 0 . 6 1 }$ </td><td> $6 6 . 7 8 \pm 0 . 5 4$ </td><td>82.31</td><td></td></tr><tr><td> $\scriptstyle \mathrm { L L P - F C }$ </td><td> $5 8 . 5 3 \pm 0 . 3 6$ </td><td>49.40 ± 0.64</td><td> $4 9 . 0 0 \pm 0 . 4 2$ </td><td> $6 4 . 5 5 \pm 1 . 1 5$ </td><td>55.37</td><td>27</td></tr><tr><td>ROT</td><td> $9 2 . 9 3 \pm 0 . 0 7$ </td><td> ${ \bf 9 0 . 7 1 \pm 0 . 1 1 }$ </td><td> $6 1 . 5 3 \pm 3 . 7 0$ </td><td>38.24 ± 1.22</td><td>70.85</td><td>5</td></tr><tr><td>EasyLLP</td><td> $4 7 . 7 2 \pm 1 . 4 0$ </td><td> $3 7 . 4 0 \pm 0 . 6 1$ </td><td> $2 8 . 8 2 \pm 2 . 4 4$ </td><td>24.09 ± 1.58</td><td>34.51</td><td>10</td></tr><tr><td> $\mathrm { E a s y L L P - A B S }$ </td><td> $4 4 . 9 7 \pm 0 . 7 7$ </td><td> $3 6 . 8 4 \pm 0 . 9 3$ </td><td> $3 8 . 0 3 \pm 0 . 3 5$ </td><td> $3 9 . 5 9 \pm 1 . 0 7$ </td><td>39.86</td><td>8</td></tr><tr><td>GeneralUPM</td><td> $4 1 . 7 9 \pm 0 . 7 8$ </td><td> $3 1 . 3 5 \pm 2 . 1 8$ </td><td> $2 9 . 7 6 \pm 0 . 9 2$ </td><td> $2 8 . 9 1 \pm 1 . 5 6$ </td><td>32.95</td><td>11</td></tr><tr><td>GeneralUPM-ABS</td><td> $4 3 . 5 3 \pm 1 . 0 3$ </td><td> $3 6 . 5 6 \pm 0 . 0 8$ </td><td> $3 7 . 5 6 \pm 0 . 5 9$ </td><td> $3 7 . 9 3 \pm 4 . 2 5$ </td><td>38.89</td><td>9</td></tr><tr><td>FlowLLP LLP-MM</td><td> $9 0 . 6 3 \pm 0 . 0 8$   ${ \bf 9 3 . 5 4 \pm 0 . 1 2 }$ </td><td> $7 2 . 6 8 \pm 1 . 6 0$   $8 8 . 2 5 \pm 0 . 5 8$ </td><td> $6 0 . 7 8 \pm 0 . 2 8$   $7 6 . 8 3 \pm 0 . 9 4$ </td><td> $5 5 . 3 9 \pm 3 . 4 4$   ${ \bf 7 0 . 6 5 \pm 0 . 6 9 }$ </td><td>69.87 82.32</td><td>6 1</td></tr><tr><td rowspan="10"></td><td></td><td></td><td></td><td> $5 4 . 5 3 \pm 0 . 0 2$ </td><td> $5 . 2 3 \pm 0 . 3 4$ </td><td></td><td>3</td></tr><tr><td>PM DSQ</td><td> $7 0 . 8 1 \pm 0 . 2 3$   $7 0 . 9 6 \pm 0 . 4 5$ </td><td> $6 7 . 0 1 \pm 0 . 1 4$ </td><td> $4 6 . 5 6 \pm 0 . 6 8$ </td><td></td><td>49.40</td><td>4</td></tr><tr><td>LLP-PVC</td><td> $6 4 . 5 4 \pm 0 . 3 4$ </td><td> $6 6 . 0 7 \pm 0 . 4 7$   $6 1 . 4 3 \pm 0 . 2 9$ </td><td> $4 9 . 8 1 \pm 1 . 9 0$ </td><td> ${ \bf 1 3 . 6 0 \pm 0 . 2 5 }$   $4 . 4 1 \pm 0 . 5 3$ </td><td>49.30 45.05</td><td>6</td></tr><tr><td>LLP-FC</td><td> $7 1 . 4 1 \pm 0 . 1 3$ </td><td>66.93 ± 0.16</td><td> $5 7 . 6 9 \pm 0 . 4 3$ </td><td> $4 . 7 5 \pm 0 . 1 0$ </td><td>50.20</td><td>2</td></tr><tr><td>ROT</td><td> ${ \bf 7 2 . 1 4 \pm 0 . 2 3 }$ </td><td>63.96 ± 0.40</td><td></td><td> $2 . 7 6 \pm 0 . 0 4$ </td><td>38.86</td><td></td></tr><tr><td>EasyLLP</td><td> $1 . 9 5 \pm 0 . 2 5$ </td><td>2.43 ± 0.63</td><td> $1 6 . 5 8 \pm 0 . 6 6$ </td><td></td><td>2.06</td><td>7</td></tr><tr><td>EasyLLP-ABS</td><td> $1 7 . 5 1 \pm 0 . 8 3$ </td><td>12.50 ± 0.36</td><td> $2 . 1 3 \pm 0 . 4 3$  9.39 ± 0.32</td><td> $1 . 7 3 \pm 0 . 1 6$ </td><td></td><td>11 9</td></tr><tr><td>GeneralUPM</td><td> $1 2 . 2 7 \pm 0 . 4 3$ </td><td> $8 . 8 7 \pm 0 . 7 6$ </td><td> $5 . 7 7 \pm 0 . 1 8$ </td><td>4.32 ± 0.66</td><td>10.93</td><td>10</td></tr><tr><td>GeneralUPM-ABS</td><td> $2 3 . 9 8 \pm 0 . 8 8$ </td><td> $1 6 . 9 3 \pm 0 . 5 4$ </td><td> $1 1 . 2 0 \pm 0 . 0 4$ </td><td> $2 . 6 3 \pm 0 . 3 0$   $6 . 3 6 \pm 0 . 3 7$ </td><td>7.38 14.61</td><td>8</td></tr><tr><td>FlowLLP</td><td> $7 0 . 9 0 \pm 0 . 2 4$ </td><td> $6 5 . 8 5 \pm 0 . 4 6$ </td><td> $5 0 . 3 7 \pm 1 . 1 7$ </td><td> $2 . 7 9 \pm 0 . 5 3$ </td><td>47.48</td><td>5</td></tr><tr><td rowspan="10"></td><td>LLP-MM</td><td> $7 2 . 0 1 \pm 0 . 2 9$ </td><td> ${ \bf 6 7 . 3 6 \pm 0 . 2 5 }$ </td><td> ${ \bf 5 8 . 0 2 \pm 0 . 3 5 }$ </td><td> $7 . 1 8 \pm 0 . 2 0$ </td><td>51.14</td><td>1</td></tr><tr><td>PM</td><td> $6 3 . 7 6 \pm 0 . 0 6$ </td><td> $5 5 . 4 7 \pm 0 . 2 9$ </td><td> $3 6 . 4 4 \pm 0 . 8 0$ </td><td> $3 . 6 2 \pm 0 . 0 9$ </td><td>39.82</td><td>4</td></tr><tr><td>DSQ</td><td> $6 3 . 1 4 \pm 0 . 3 3$ </td><td> $5 5 . 0 9 \pm 0 . 6 0$ </td><td> $3 3 . 0 3 \pm 0 . 5 7$ </td><td> ${ \bf 8 . 7 2 \pm 0 . 6 9 }$ </td><td>40.00</td><td>3</td></tr><tr><td>LLP-PVC</td><td> $5 7 . 4 7 \pm 0 . 3 7$ </td><td> $5 4 . 7 4 \pm 0 . 4 9$ </td><td> $3 1 . 0 1 \pm 1 . 4 7$ </td><td> $2 . 8 9 \pm 0 . 2 6$ </td><td>36.53</td><td>6</td></tr><tr><td>LLP-FC</td><td> $6 4 . 2 3 \pm 0 . 3 2$ </td><td> $5 4 . 7 1 \pm 0 . 4 8$ </td><td> $3 8 . 8 2 \pm 0 . 3 8$ </td><td> $3 . 8 0 \pm 0 . 2 7$ </td><td>40.39</td><td>2</td></tr><tr><td>ROT</td><td> $6 5 . 8 7 \pm 0 . 4 8$ </td><td> $5 4 . 2 9 \pm 1 . 0 0$ </td><td> $1 1 . 1 9 \pm 0 . 2 0$ </td><td> $2 . 3 3 \pm 0 . 2 8$ </td><td>33.42</td><td>7</td></tr><tr><td>EasyLLP</td><td> $1 . 6 2 \pm 0 . 2 1$ </td><td> $1 . 6 1 \pm 0 . 0 8$ </td><td> $1 . 7 1 \pm 0 . 1 3$ </td><td> $1 . 4 8 \pm 0 . 0 8$ </td><td>1.60</td><td>11</td></tr><tr><td>EasyLLP-ABS</td><td> $1 1 . 4 7 \pm 0 . 2 0$ </td><td> $8 . 5 6 \pm 0 . 1 8$ </td><td> $5 . 4 9 \pm 0 . 6 6$ </td><td> $2 . 8 3 \pm 1 . 0 4$ </td><td>7.09</td><td>9</td></tr><tr><td>GeneralUPM</td><td> $9 . 8 5 \pm 0 . 1 3$ </td><td> $6 . 8 9 \pm 0 . 5 4$ </td><td> $4 . 3 0 \pm 0 . 2 1$ </td><td> $2 . 4 1 \pm 0 . 1 9$ </td><td>5.86</td><td>10</td></tr><tr><td>GeneralUPM-ABS</td><td> $2 0 . 7 3 \pm 1 . 0 5$ </td><td> $1 3 . 6 0 \pm 0 . 7 8$ </td><td>7.70 ± 0.09</td><td>4.03 ± 0.30</td><td>11.51</td><td>8 5</td></tr><tr><td>FlowLLP</td><td> $6 2 . 7 6 \pm 0 . 2 1$ </td><td> $5 2 . 7 9 \pm 1 . 0 1$ </td><td> $3 0 . 8 9 \pm 1 . 5 9$ </td><td> $2 . 6 7 \pm 0 . 3 3$ </td><td>37.28</td><td></td></tr><tr><td> $\mathbf { L L P  – M M }$ </td><td> ${ \bf 6 6 . 3 7 \pm 0 . 1 6 }$ </td><td> ${ \bf 5 9 . 9 6 \pm 0 . 1 9 }$ </td><td> ${ \bf 4 4 . 9 1 \pm 0 . 4 5 }$ </td><td></td><td> $3 . 4 3 \pm 0 . 0 9$ </td><td>43.67</td></tr></table>

Table 3: Test performance on KU-Optofil PBC (mean ± std over three runs, %). The best result in each column is bold.
<table><tr><td>Method</td><td>Accuracy ↑</td><td>Macro-F1 ↑</td><td>Balanced Acc. ↑</td><td>Weighted-F1 ↑</td></tr><tr><td>PM</td><td> $7 7 . 3 5 \pm 1 3 . 1 0$ </td><td> $2 9 . 1 1 \pm 1 0 . 7 4$ </td><td> $3 2 . 7 0 \pm 1 0 . 0 1$ </td><td> $7 4 . 4 6 \pm 1 5 . 7 3$ </td></tr><tr><td>DSQ</td><td> $9 2 . 3 9 \pm 1 . 3 3$ </td><td> $5 6 . 8 9 \pm 7 . 6 5$ </td><td> $6 0 . 0 8 \pm 8 . 3 0$ </td><td> $9 2 . 7 0 \pm 1 . 5 7$ </td></tr><tr><td>LLP-PVC</td><td> $9 2 . 8 2 \pm 1 . 9 4$ </td><td> $5 8 . 4 2 \pm 4 . 3 1$ </td><td> $6 1 . 6 3 \pm 4 . 2 7$ </td><td> $9 1 . 5 8 \pm 2 . 3 3$ </td></tr><tr><td>LLP-FC</td><td> $7 5 . 1 1 \pm 9 . 5 7$ </td><td> $2 6 . 2 6 \pm 4 . 2 8$ </td><td> $2 7 . 1 2 \pm 4 . 4 6$ </td><td> $7 2 . 4 6 \pm 1 1 . 6 0$ </td></tr><tr><td>ROT</td><td> ${ \bf 9 4 . 0 5 \pm 0 . 6 5 }$ </td><td> $5 6 . 1 1 \pm 7 . 5 6$ </td><td> $5 6 . 8 2 \pm 7 . 7 4$ </td><td> $9 3 . 3 5 \pm 1 . 1 7$ </td></tr><tr><td>EasyLLP</td><td> $3 7 . 9 8 \pm 8 . 3 2$ </td><td> $1 1 . 5 1 \pm 1 . 7 7$ </td><td> $1 4 . 5 0 \pm 0 . 8 5$ </td><td> $3 9 . 2 8 \pm 3 . 1 1$ </td></tr><tr><td>EasyLLP-ABS</td><td> $3 8 . 3 3 \pm 8 . 8 3$ </td><td> $1 1 . 0 2 \pm 1 . 5 9$ </td><td> $1 3 . 8 5 \pm 1 . 8 1$ </td><td> $4 0 . 4 8 \pm 5 . 1 1$ </td></tr><tr><td>GeneralUPM</td><td> $6 6 . 3 3 \pm 0 . 2 2$ </td><td> $1 2 . 4 8 \pm 1 . 2 8$ </td><td> $1 4 . 1 6 \pm 1 . 3 2$ </td><td> $5 5 . 1 9 \pm 2 . 5 1$ </td></tr><tr><td>GeneralUPM-ABS</td><td> $6 6 . 3 3 \pm 0 . 2 2$ </td><td> $1 2 . 4 8 \pm 1 . 2 8$ </td><td> $1 4 . 1 6 \pm 1 . 3 2$ </td><td> $5 5 . 1 9 \pm 2 . 5 1$ </td></tr><tr><td>FlowLLP</td><td> $6 1 . 9 7 \pm 4 . 3 4$ </td><td> $2 5 . 6 7 \pm 2 . 3 4$ </td><td> $3 1 . 1 1 \pm 1 . 9 0$ </td><td> $5 5 . 7 5 \pm 4 . 8 5$ </td></tr><tr><td> $\mathrm { L L P - M M } \left( s = 3 \right)$ </td><td> $9 1 . 8 8 \pm 1 . 2 0 $ </td><td> $5 1 . 7 9 \pm 5 . 8 0$ </td><td> $5 3 . 2 3 \pm 7 . 4 1$ </td><td> $9 1 . 8 8 \pm 0 . 4 8$ </td></tr><tr><td> $\mathrm { L L P - M M } \left( s = 5 \right)$ </td><td> $9 3 . 4 0 \pm 0 . 8 4$ </td><td> $6 6 . 4 0 \pm 2 . 2 2$ </td><td> $6 9 . 8 1 \pm 0 . 9 1$ </td><td> $9 4 . 2 0 \pm 0 . 5 5$ </td></tr><tr><td> $\mathbf { L L P - M M } ( s = 8 )$ </td><td> $9 3 . 9 3 \pm 0 . 5 8$ </td><td> ${ \bf 6 9 . 7 5 \pm 1 . 4 7 }$ </td><td> ${ \bf 7 0 . 7 3 \pm 2 . 7 4 }$ </td><td> ${ \bf 9 4 . 5 7 \pm 0 . 2 2 }$ </td></tr></table>

LLP-MM (s = 8) leads in Macro-F1, Balanced Accuracy, and Weighted-F1, exceeding LLP-PVC by 11.33 Macro-F1 points; ROT has the highest Accuracy. Increasing s from 3 to 5 to 8 raises Macro-F1 from 51.79% to 66.40% to 69.75%.

## 4.4 Effect of the Order Parameter

Figure 2 shows the accuracy–runtime trade-off as s increases. Higher orders tighten the minimizer set toward $f ^ { * }$ (Proposition 3.1) but increase computation. We report CIFAR-100 only up $\tan s = 3 ,$ , since fourth-order terms are prohibitively expensive with 100 classes. As the order increases from low values, prediction performance generally improves, suggesting that higher-order moments provide useful supervision beyond mean class proportions. At higher orders, performance tends to stabilize, and further increases bring only limited gains. Since computation continues to grow with the order, moderate orders offer a practical balance between predictive performance and computational cost.

![](images/485e3b88a47d11b7bb8fe6cd0bfa2dba8dca71897ea2b968bd2c49798fd9c19d.jpg)  
Figure 2: Effect of the order parameter on prediction performance and runtime per epoch across different bag construction strategies and bag sizes. Runtime was measured on a single NVIDIA A100-SXM4-40GB GPU with an AMD EPYC 7742 64-Core Processor.

## 5 Conclusion

We proposed LLP-MM, a moment-matching framework for learning from label proportions. By connecting proportions to the Poisson multinomial distribution, LLP-MM exploits multi-order factorial moments derived from observed label proportions and uses them as training supervision. Theoretical analysis shows that higher-order moments impose stronger constraints on the solution space, and experiments demonstrate that LLP-MM achieves robust performance across various settings while maintaining practical computational efficiency. A limitation is the scarcity of standardized real-world LLP datasets; beyond the semi-real-world study in Section 4.3, future work will develop broader realistic benchmarks. We will also design multiclass LLP benchmarks and evaluation metrics to compare different methods.

## Acknowledgments and Disclosure of Funding

Part of the experiments was conducted using the Miyabi and Wisteria supercomputing systems at the Joint Center for Advanced High Performance Computing (JCAHPC) and the Information Technology Center, The University of Tokyo, respectively.

YZ was supported by JSPS KAKENHI Grant Number JP26K21299. DDW was supported by the RIKEN Junior Research Associate Program. TI was supported by KAKENHI Grant Number 22K17946. MS was supported by JST ASPIRE Grant Number JPMJAP25B1.

## References

Apple. SKAdNetwork 4 release notes. https://developer.apple.com/documentatio n/storekit/skadnetwork/skadnetwork\_release\_notes/skadnetwork\_4\_r elease\_notes, 2025. Apple Developer Documentation.

Lorne Applebaum, Travis Dick, Claudio Gentile, Haim Kaplan, and Tomer Koren. Optimal learning from label proportions with general loss functions. arXiv preprint arXiv:2509.15145, 2025.

Ehsan Mohammady Ardehaly and Aron Culotta. Co-training for demographic classification using deep learning from label proportions. In IEEE International Conference on Data Mining Workshops, pages 1017–1024, 2017.

Peter L Bartlett, Michael I Jordan, and Jon D McAuliffe. Convexity, classification, and risk bounds. Journal ofthe American Statistical Association, 101(473):138–156, 2006. URL https://doi. org/10.1198/016214505000000907.

Shai Ben-David, Nadav Eiron, and Philip M Long. On the difficulty of approximately maximizing agreements. Journal of Computer and System Sciences, 66(3):496–514, 2003. URL https: //doi.org/10.1016/S0022-0000(03)00038-2.

Gerda Bortsova, Florian Dubost, Silas Ørting, Ioannis Katramados, Laurens Hogeweg, Laura Thomsen, Mathilde Wille, and Marleen de Bruijne. Deep learning from label proportions for emphysema quantification. In Proceedings of the 21st International Conference on Medical Image Computing and Computer Assisted Intervention, pages 768–776. Springer, 2018.

Anand Brahmbhatt, Mohith Pokala, Rishi Saket, and Aravindan Raghuveer. LLP-Bench: A large scale tabular benchmark for learning from label proportions. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 4374–4381, 2024.

Róbert Busa-Fekete, Heejin Choi, Travis Dick, Claudio Gentile, and Andrés Muñoz Medina. Easy learning from label proportions. In Advances in Neural Information Processing Systems, pages 14957–14968, 2023.

Robert Busa-Fekete, Travis Dick, Claudio Gentile, Haim Kaplan, Tomer Koren, and Uri Stemmer. Nearly optimal sample complexity for learning with label proportions. arXiv preprint arXiv:2505.05355, 2025.

Constantinos Daskalakis, Gautam Kamath, and Christos Tzamos. On the structure, covering, and learning of Poisson multinomial distributions. In 2015 IEEE 56th annual symposium onfoundations of computer science, pages 1203–1217, 2015.

Eustache Diemert, Romain Fabre, Alexandre Gilotte, Fei Jia, Basile Leparmentier, Jérémie Mary, Zhonghua Qu, Ugo Tanielian, and Hui Yang. Lessons from the AdKDD’21 privacy-preserving ML challenge. In Proceedings of the ACM Web Conference 2022, pages 2026–2035, 2022.

Gabriel Dulac-Arnold, Neil Zeghidour, Marco Cuturi, Lucas Beyer, and Jean-Philippe Vert. Deep multi-class learning from label proportions. arXiv preprint arXiv:1905.12909, 2019.

Gabriel Franco, Mark Crovella, and Giovanni Comarela. Dependence and model selection in LLP: the problem of variants. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 470–481, 2023.

Google. Private aggregation API of Chrome privacy sandbox. https://developer.chro me.com/docs/privacy-sandbox/aggregation-service/, 2024. Chrome Privacy Sandbox documentation.

Norman L. Johnson, Samuel Kotz, and Narayanaswamy Balakrishnan. Discrete Multivariate Distributions. John Wiley & Sons, 1997. ISBN 9780471128441.

Alex Krizhevsky and Geoffrey E. Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Kuan-Ting Lai, Felix X Yu, Ming-Syan Chen, and Shih-Fu Chang. Video event detection by inferring temporal instance labels. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2243–2250, 2014.

Zhengfeng Lai, Chao Wang, Henrry Gunawan, Sen-Ching S. Cheung, and Chen-Nee Chuah. Smoothed adaptive weighting for imbalanced semi-supervised learning: Improve reliability against unknown distribution data. In International Conference on Machine Learning, pages 11828–11843, 2022.

Changchun Li, Ximing Li, and Jihong Ouyang. Semi-supervised text classification with balanced deep representation distributions. In Association for Computational Linguistics and the International Joint Conference on Natural Language Processing, pages 5044–5053, 2021.

Gene Li, Lin Chen, Adel Javanmard, and Vahab Mirrokni. Optimistic rates for learning from label proportions. In Annual Conference on Learning Theory, pages 3437–3474, 2024.

Zhengzhi Lin, Yueyao Wang, and Yili Hong. The computing of the poisson multinomial distribution and applications in ecological inference and machine learning. Computational Statistics, 38(4): 1851–1877, 2023.

Jiabin Liu, Bo Wang, Zhiquan Qi, Yingjie Tian, and Yong Shi. Learning from label proportions with generative adversarial networks. In Advances in Neural Information Processing Systems, pages 7167–7177, 2019.

Nan Lu, Shida Lei, Gang Niu, Issei Sato, and Masashi Sugiyama. Binary classification from multiple unlabeled datasets via surrogate set classification. In International Conference on Machine Learning, pages 7134–7144, 2021.

Jiaqi Lv, Miao Xu, Lei Feng, Gang Niu, Xin Geng, and Masashi Sugiyama. Progressive identification of true labels for partial-label learning. In International Conference on Machine Learning, pages 6500–6510, 2020.

Tianhao Ma, Han Chen, Juncheng Hu, Yungang Zhu, and Ximing Li. Forming auxiliary highconfident instance-level loss to promote learning from label proportions. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20592–20601, 2025.

Tianhao Ma, Ximing Li, Changchun Li, and Renchu Guan. Learning from label proportions with dual-proportion constraints. arXiv preprint arXiv:2603.21153, 2026a.

Tianhao Ma, Wei Wang, Ximing Li, Gang Niu, and Masashi Sugiyama. Learning from label proportions via proportional value classification. In The Fourteenth International Conference on Learning Representations, 2026b.

Andreas Maurer. A vector-contraction inequality for Rademacher complexities. In Proceedings of the 27th International Conference on Algorithmic Learning Theory, pages 3–17, 2016.

Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations ofmachine learning. MIT press, 2018.

Kai Wang Ng, Guo-Liang Tian, and Man-Lai Tang. Dirichlet and Related Distributions: Theory, Methods and Applications. John Wiley & Sons, 2011. ISBN 9780470688199.

Carolina Pacheco, Florence Yellin, René Vidal, and Benjamin Haeffele. Vertex proportion loss for multi-class cell detection from label proportions. In Medical Image Computing and Computer Assisted Intervention, pages 366–376. Springer Nature Switzerland, 2024. URL https://pape rs.miccai.org/miccai-2024/829-Paper3997.html.

Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1944–1952, 2017.

Noah Peres, Andrew Lee, and Uri Keich. Exactly computing the tail of the Poisson-Binomial distribution. arXiv preprint arXiv:2004.07429, 2020. URL https://arxiv.org/abs/20 04.07429.

Novi Quadrianto, Alex J Smola, Tibério S Caetano, and Quoc V Le. Estimating labels from label proportions. Journal ofMachine Learning Research, 10:2349–2374, 2009.

Alain Rakotomamonjy, Maxime Vono, and Liva Ralaivola. Particle flow for learning from label proportions. In International Conference on Machine Learning, 2026. URL https://icml.c c/virtual/2026/poster/63488.

Masashi Sugiyama, Han Bao, Takashi Ishida, Nan Lu, Tomoya Sakai, and Gang Niu. Machine learningfrom weak supervision: An empirical risk minimization approach. MIT Press, 2022.

Tao Sun, Dan Sheldon, and Brendan O’Connor. A probabilistic approach for learning with label proportions applied to the US presidential election. In Proceedings of the 2017 IEEE International Conference on Data Mining, pages 445–454. IEEE, 2017.

Hiroki Tokunaga, Brian Kenji Iwana, Yuki Teramoto, Akihiko Yoshizawa, and Ryoma Bise. Negative pseudo labeling using class proportion for semantic segmentation in pathology. In European Conference on Computer Vision, pages 430–446, 2020.

Kuen-Han Tsai and Hsuan-Tien Lin. Learning from label proportions with consistency regularization. In Asian Conference on Machine Learning, pages 513–528, 2020.

Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. Advances in Neural Information Processing Systems, 29, 2016.

Wei Wang, Tianhao Ma, Ming-Kun Xie, Gang Niu, and Masashi Sugiyama. Rethinking consistent multi-label classification under inexact supervision. arXiv preprint arXiv:2510.04091, 2025.

Zixi Wei, Senlin Shu, Yuzhou Cao, Hongxin Wei, Bo An, and Lei Feng. Consistent multi-class classification from multiple unlabeled datasets. In The Twelfth International Conference on Learning Representations (ICLR), 2024.

Atıf Eren Yarıkan, Can Örer, Volkan Akyıldız, Zeki Ku¸s, Musa Aydin, Kerim Erhan Palaoglu, Said˘ ˙Incir, Kemal Baysal, Cemal Özçelik, Berna Kiraz, and Alper Kiraz. A large-scale peripheral blood cell dataset for automated hematological analysis. Scientific Data, 13(1):417, 2026. doi: 10.1038/ s41597-026-06761-y. URL https://doi.org/10.1038/s41597-026-06761-y.

Felix X Yu, Krzysztof Choromanski, Sanjiv Kumar, Tony Jebara, and Shih-Fu Chang. On learning from label proportions. arXiv preprint arXiv:1402.5902, 2014.

Xiyu Yu, Tongliang Liu, Mingming Gong, and Dacheng Tao. Learning with biased complementary labels. In Proceedings ofthe European Conference on Computer Vision, pages 68–83, 2018.

Jianxin Zhang, Yutong Wang, and Clayton Scott. Learning from label proportions by learning with label noise. In Advances in Neural Information Processing Systems, pages 26933–26942, 2022.

## A Notation and Terminology

For convenience, Tables 4 and 5 summarize the abbreviations and main notation used throughout the paper and this appendix.

Table 4: Notation used in the paper.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>X</td><td>Input space.</td></tr><tr><td> $\mathcal { V } = \{ 1 , \ldots , q \}$ </td><td>Label space.</td></tr><tr><td>q</td><td>Number of classes.</td></tr><tr><td>m</td><td>Bag size.</td></tr><tr><td> $\Delta ^ { q - 1 }$ </td><td>The  $( q - 1 )$  -dimensional probability simplex</td></tr><tr><td></td><td> $\begin{array} { r } { \{ ( p _ { 1 } , \dotsc , { \dot { p } } _ { q } ) \in \mathbb { R } ^ { q } \ \big | \ p _ { k } \geq 0 , \sum _ { k = 1 } ^ { \bar { q } } { \dot { p } } _ { k } = 1 \} , } \end{array}$ </td></tr><tr><td> $x _ { j } \in \mathcal X$   $\check { y _ { j } } \in \mathcal { Y }$ </td><td>The j-th instance in a bag.</td></tr><tr><td> $ { \mathbf { \vec { B } } } = ( x _ { 1 } , \dots , x _ { m } ) \in \boldsymbol { \mathcal { X } } ^ { m }$ </td><td>Latent class label of  $x _ { j } .$ </td></tr><tr><td></td><td>Bag of m instances. Latent label sequence associated with B.</td></tr><tr><td> $\pmb { y } = ( y _ { 1 } , \dots , y _ { m } ) \in \mathcal { V } ^ { m }$ </td><td></td></tr><tr><td> $\mathcal { y } ^ { m / \alpha }$ </td><td>Set of label sequences in  $\boldsymbol { \mathcal { y } } ^ { m }$  whose empirical class proportions equal α.</td></tr><tr><td> $\mathbb { I } \{ \cdot \} : \{ \mathrm { s t a t e m e n t s } \} \to \{ 0 , 1 \}$ </td><td>Indicator function, which returns 1 when its argument is true and 0 otherwise.  $\begin{array} { r } { \alpha _ { k } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { I } \{ y _ { j } = k \} } \end{array}$ </td></tr><tr><td> $\alpha _ { k } \in [ 0 , 1 ]$ </td><td>Proportion of class k in a bag, defined by</td></tr><tr><td> $\pmb { \alpha } = ( \alpha _ { 1 } , \dots , \alpha _ { q } ) \in \Delta ^ { q - 1 }$ </td><td>Bag-level class-proportion vector.</td></tr><tr><td> $z = m \alpha \in \{ 0 , \ldots , m \} ^ { q }$ </td><td>Bag-level class-count vector.</td></tr><tr><td> $e ^ { y _ { j } } \in \{ 0 , 1 \} ^ { \overline { { q } } } \subset \mathbb { R } ^ { q }$ </td><td>One-hot basis vector corresponding to label  $y _ { j } .$ </td></tr><tr><td> $p : \mathcal { X } \times \mathcal { Y } \stackrel { } { \to } \mathbb { R } _ { + }$   $p ( \cdot \mid x ) : \mathcal { V } \to [ \dot { 0 } , 1 ]$ </td><td>Joint data distribution over instance-label pairs. Class-posterior map for a fixed instance  $x \in { \mathcal { X } } .$ </td></tr><tr><td></td><td></td></tr><tr><td> ${ \mathcal { F } } \subseteq \left\{ f : \mathcal { X } \to \Delta ^ { q - 1 } \right\}$ </td><td>Hypothesis class of score-based classifiers.</td></tr><tr><td> $f : \mathcal { X } \overset { \setminus } { \to } \Delta ^ { q - 1 }$ </td><td>Score-based classifier, with  $f \in { \mathcal { F } } .$ </td></tr><tr><td> $f ^ { * } : \mathcal { X } \to \Delta ^ { q - 1 }$ </td><td>Bayes-optimal classifier. Predicted class-probability vector for instance</td></tr><tr><td> $p _ { j } = f ( x _ { j } ) \in \Delta ^ { q - 1 }$ </td><td> $x _ { j } .$ </td></tr><tr><td> $\ell : \Delta ^ { q - 1 } \times \mathcal { Y }  \mathbb { R } _ { + }$  M</td><td>Classification-calibrated surrogate loss used in ordinary supervised learning. Generic normalized population moment object; for example,  $\mathbf { M } _ { \alpha } ^ { ( 1 ) }$  is the</td></tr><tr><td></td><td>normalized first-order population moment and  $\mathbf { M } ^ { ( 2 ) }$  denotes the normalized bag-level second-order moment matrix.</td></tr><tr><td> $( m ) _ { r }$ </td><td>Falling factorial m  $( m - 1 ) \cdots ( m - r + 1 ) .$ </td></tr><tr><td> $[ m ] _ { \neq } ^ { r }$ </td><td>Set of ordered r-tuples of distinct indices from  $[ m ] .$ </td></tr><tr><td> $\widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } \in ( \mathbb { R } ^ { q } ) ^ { \otimes r }$   $\widehat { \mathbf { M } } _ { f } ^ { ( r ) } : \mathcal { X } ^ { m }  ( \mathbb { R } ^ { q } ) ^ { \otimes r }$ </td><td>Normalized r-th order factorial moment tensor computed from α. Classifier-induced normalized r-th order factorial moment map;  $\widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } )$ </td></tr><tr><td></td><td>value on bag B.</td></tr><tr><td> $\boldsymbol { \mathcal { L } ^ { ( r ) } }$ </td><td>Order-r discrepancy loss between normalized target and classifier-induced moments; for  $r = 1$  it acts on class-proportion vectors, and for general r it acts on tensors in  $\left( \mathbb { R } ^ { q } \right) ^ { \otimes r }$ </td></tr><tr><td> $R _ { \mathrm { M M } } ^ { ( s ) } : \mathcal { F }  \mathbb { R } _ { + }$ </td><td>Population multi-order normalized moment-matching risk using moments up to order s.</td></tr><tr><td> $\widehat { R } _ { \mathrm { M M } } ^ { ( s ) } : \mathcal { F } \to \mathbb { R } _ { + }$ </td><td>Empirical multi-order normalized moment-matching risk using moments up to</td></tr><tr><td> $s \in \{ 1 , \ldots , m \}$ </td><td>order s. Maximum moment order used for training.</td></tr><tr><td> $\lambda _ { r } > 0$ </td><td>Weight of the r-th order term, with  $\begin{array} { r } { \sum _ { r = 1 } ^ { s } \lambda _ { r } = 1 . } \end{array}$ </td></tr><tr><td>R</td><td>Rademacher-complexity symbol.</td></tr><tr><td> $\Re _ { n } ( \mathcal { F } _ { c } )$ </td><td>Expected Rademacher complexity of the coordinate function class</td></tr><tr><td>k</td><td> ${ \mathcal { F } } _ { c } .$  Multiplicity vector recording how often each class index appears in an ordered</td></tr><tr><td></td><td>index tuple. Compressed multiplicity-state space at order defined by</td></tr><tr><td> $\kappa _ { q , r }$ </td><td> $r ,$   $\mathcal { K } _ { q , r } \overset { \cdot } { = } \left\{ \pmb { k } \in \mathbb { N } ^ { q ^ { \star } } \big | \sum _ { c = 1 } ^ { \mathbf { \bar { q } } } k _ { c } \overset { \cdot } { = } r \right\}$ </td></tr><tr><td> $N _ { r }$ </td><td>Cardinality of the compressed multiplicity-state space, namely  $N _ { r } = | \dot { K _ { q , r } } | = ( { } ^ { q + r ^ { 2 } - 1 } )$ </td></tr><tr><td> $A _ { t } ^ { ( d ) } : \mathcal { K } _ { q , d }  \mathbb { R }$ </td><td> $A _ { t } ^ { ( d ) } ( k )$  Coefficient map in the truncated generating polynomial; is the coefficient associated with state k.</td></tr></table>

Table 5: Abbreviations used in the paper.
<table><tr><td>Abbreviation</td><td>Meaning</td></tr><tr><td>WSL</td><td>Weakly supervised learning</td></tr><tr><td>LLP</td><td>Learning from label proportions</td></tr><tr><td>PMD</td><td>Poisson multinomial distribution</td></tr><tr><td>MM</td><td>Moment matching</td></tr><tr><td>LLP-MM</td><td>LLP via Moment Matching</td></tr></table>

## B First-Order Moment Matching

Based on the connection between LLP and the PMD, we propose a general framework which matches the empirical factorial moments with classifier-induced moments. We present the first-order moment matching approach, which has been referred to as proportion matching in prior work [Yu et al., 2014, Ardehaly and Culotta, 2017]. Specifically, let $\bar { \mathbf { M } } _ { \alpha } ^ { ( 1 ) } = [ M _ { 1 } ^ { ( 1 ) } , \bar { M } _ { 2 } ^ { ( 1 ) } , \cdot \cdot \cdot , M _ { q } ^ { ( 1 ) } ]$ denote the normalized first-order moment of z. Then, we have

$$
M _ { k } ^ { ( 1 ) } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { E } _ { p ( y _ { j } \mid x _ { j } ) } \big [ \mathbb { I } \{ y _ { j } = k \} \big ] = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } p ( y _ { j } = k \mid x _ { j } ) .\tag{16}
$$

However, since the instance-level posterior probabilities are unknown, we approximate $\mathbf { M } _ { \alpha } ^ { ( 1 ) }$ with thefirst-order empirical moment $\bar { \mathbf { M } } _ { \alpha } ^ { ( 1 ) }$ , where the k-th element is

$$
\left[ \widehat { \mathbf { M } } _ { \alpha } ^ { ( 1 ) } \right] _ { k } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } [ \mathbb { I } \{ y _ { j } = k \} ] = \frac { z _ { k } } { m } = \alpha _ { k } ,\tag{17}
$$

and $z _ { k }$ is the k-th dimension of z. Apart from empirical moments, we can also use the classifier outputs to approximate $\mathbf { M } _ { \alpha } ^ { ( 1 ) }$ . Specifically, we introduce

$$
{ \widehat { \bf M } } _ { f } ^ { ( 1 ) } ( { \bf B } ) = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } f ( x _ { j } )\tag{18}
$$

to represent the first-order classifier-induced moment. Finally, we introduce thefirst-order moment matching risk to match the first-order empirical factorial moments with the first-order classifierinduced moment:

$$
R _ { \mathrm { M M } } ^ { ( 1 ) } ( \pmb { f } ) = \mathbb { E } _ { p ( \mathbf { B } , \pmb { \alpha } ) } \Big [ \mathscr { L } ^ { ( 1 ) } \Big ( \pmb { \alpha } , \widehat { \mathbf { M } } _ { \pmb { f } } ^ { ( 1 ) } ( \mathbf { B } ) \Big ) \Big ] ,\tag{19}
$$

where $\mathcal { L } ^ { ( 1 ) } : \Delta ^ { q - 1 } \times \Delta ^ { q - 1 }  [ 0 , + \infty ]$ measures the discrepancy between the true and predicted class proportions. Notably, the formulation in Equation (19) is the same as the objective of the popular proportion matching strategy [Tsai and Lin, 2020, Ma et al., 2025, Liu et al., 2019] which uses the observed proportions as targets for the predicted mean proportions. Our moment matching framework provides a new perspective for understanding the proportion matching strategy based on the PMD.

## C Derivation of the Second-Order Moment from Label Proportions

Recall that the bag-level second-order moment matrix is

$$
\mathbf { M } ^ { ( 2 ) } = \frac { 1 } { m ( m - 1 ) } \sum _ { i \neq j } e ^ { y _ { i } } ( e ^ { y _ { j } } ) ^ { \top } .\tag{20}
$$

We show that each entry of $\mathbf { M } ^ { ( 2 ) }$ can be computed directly from the label proportion vector α.

For any $c , d \in \{ 1 , \ldots , q \}$ , the $( c , d )$ -th entry of $\mathbf { M } ^ { ( 2 ) }$ is

$$
M _ { c d } ^ { ( 2 ) } = \frac { 1 } { m ( m - 1 ) } \sum _ { i \neq j } [ e ^ { y _ { i } } ] _ { c } [ e ^ { y _ { j } } ] _ { d }\tag{21}
$$

$$
= { \frac { 1 } { m ( m - 1 ) } } \sum _ { i \neq j } \mathbb { I } \{ y _ { i } = c \} \mathbb { I } \{ y _ { j } = d \} .\tag{22}
$$

Let

$$
n _ { c } : = \sum _ { i = 1 } ^ { m } \mathbb { I } \{ y _ { i } = c \} = m \alpha _ { c }\tag{23}
$$

denote the number of instances in the bag belonging to class $c .$ Then we consider two cases.

Case $1 \colon c \neq d .$ In this case, the constraint $i \neq j$ is automatically satisfied whenever $\mathbb { I } \{ y _ { i } =$ $c \} \mathbb { I } \{ y _ { j } = d \} = 1$ , because one instance cannot simultaneously belong to two different classes. Hence,

$$
\sum _ { i \neq j } \mathbb { I } \{ y _ { i } = c \} \mathbb { I } \{ y _ { j } = d \} = \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m } \mathbb { I } \{ y _ { i } = c \} \mathbb { I } \{ y _ { j } = d \}\tag{24}
$$

$$
= \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { m } \mathbb { I } \{ y _ { i } = c \} \mathbb { I } \{ y _ { j } = d \}\tag{25}
$$

$$
= \left( \sum _ { i = 1 } ^ { m } \mathbb { I } \{ y _ { i } = c \} \right) \left( \sum _ { j = 1 } ^ { m } \mathbb { I } \{ y _ { j } = d \} \right)\tag{26}
$$

$$
= n _ { c } n _ { d } .\tag{27}
$$

Substituting $n _ { c } = m \alpha _ { c }$ and $n _ { d } = m \alpha _ { d }$ into Equation (22), we obtain

$$
M _ { c d } ^ { ( 2 ) } = \frac { n _ { c } n _ { d } } { m ( m - 1 ) } = \frac { m \alpha _ { c } m \alpha _ { d } } { m ( m - 1 ) } , \qquad c \neq d .\tag{28}
$$

Case 2: $c = d .$ Now the constraint $i \neq j$ matters. We have

$$
\sum _ { i \neq j } \mathbb { I } \{ y _ { i } = c \} \mathbb { I } \{ y _ { j } = c \} = \left( \sum _ { i = 1 } ^ { m } \mathbb { I } \{ y _ { i } = c \} \right) \left( \sum _ { j = 1 } ^ { m } \mathbb { I } \{ y _ { j } = c \} \right) - \sum _ { i = 1 } ^ { m } \mathbb { I } \{ y _ { i } = c \}\tag{29}
$$

$$
= n _ { c } ^ { 2 } - n _ { c }\tag{30}
$$

$$
= n _ { c } ( n _ { c } - 1 ) .\tag{31}
$$

Therefore,

$$
M _ { c c } ^ { ( 2 ) } = \frac { n _ { c } ( n _ { c } - 1 ) } { m ( m - 1 ) } = \frac { m \alpha _ { c } ( m \alpha _ { c } - 1 ) } { m ( m - 1 ) } .\tag{32}
$$

Combining the two cases, we obtain

$$
M _ { c d } ^ { ( 2 ) } = \frac { n _ { c } ( n _ { d } - \mathbb { I } \{ c = d \} ) } { m ( m - 1 ) } = \frac { m \alpha _ { c } ( m \alpha _ { d } - \mathbb { I } \{ c = d \} ) } { m ( m - 1 ) } , \qquad c , d \in \{ 1 , \ldots , q \} .\tag{33}
$$

This proves that the second-order moment matrix is completely determined by the observed label proportion vector α. □

## D Proof of Proposition 3.1

Theorem D.1. Assume that the hypothesis class $\mathcal { F }$ is expressive enough to contain the Bayes-optimal classifier, i.e., $\pmb { f } ^ { * } \in \mathcal { F }$ . For each $r \in \{ 1 , \ldots , s \}$ , assume thatfor any $\mathcal { T } _ { r }$ -valued random variable $Z ,$ its expectation $\mathbb { E } [ Z ]$ is the unique solution ofmin $\mathbb { I } _ { \boldsymbol { v } \in \mathcal { T } _ { r } } \mathbb { E } \Big [ \mathcal { L } ^ { ( r ) } ( \boldsymbol { Z } , \boldsymbol { v } ) \Big ]$ . Then, for any minimizer $f \in \mathcal { F } _ { \mathrm { M M } } ^ { ( s ) }$ of Equation $( 1 3 ) , f o r p ( \mathbf { B } )$ -almost every bag B and every $r \in \{ 1 , \ldots , s \}$ such that $\lambda _ { r } > 0$ we have

$$
\widehat { \mathbf { M } } _ { \pmb { f } } ^ { ( r ) } ( \mathbf { B } ) = \widehat { \mathbf { M } } _ { \pmb { f } ^ { * } } ^ { ( r ) } ( \mathbf { B } ) \in ( \mathbb { R } ^ { q } ) ^ { \otimes r } .\tag{34}
$$

That $i s ,$ every minimizer of Equation (13) exactly matches the moment tensors induced $b y$ the Bayes-optimal classifierfor all positively weighted orders, $f o r p ( \mathbf { B } )$ )-almost every bag B.

Theorem D.1 shows that incorporating multi-order moments as training supervision enforces agreement with the Bayes-optimal classifier-induced moments at correspondingly higher orders.

Proof of Theorem D.1. For each order $r \in \{ 1 , \ldots , s \}$ , define the random tensor

$$
Z _ { r } : = \widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } .\tag{35}
$$

We first show that, conditioned on a bag $\mathbf { B } = ( x _ { 1 } , \dots , x _ { m } )$ , its conditional expectation is exactly the classifier-induced moment tensor of the Bayes-optimal classifier:

$$
\mathbb { E } [ Z _ { r } \mid \mathbf { B } ] = { \widehat { \mathbf { M } } } _ { { \boldsymbol { f } } ^ { * } } ^ { ( r ) } ( \mathbf { B } ) .\tag{36}
$$

Indeed, by Equation (10),

$$
Z _ { r } = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \ldots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } e ^ { y _ { j _ { 1 } } } \otimes \cdots \otimes e ^ { y _ { j _ { r } } } .\tag{37}
$$

Taking conditional expectation given B and using both the conditional independence of $y _ { 1 } , \ldots , y _ { m }$ given B and the conditional marginal identities $p \bar { ( \boldsymbol { y } _ { j } \mid \mathbf { B } ) } = p ( \boldsymbol { y } _ { j } \mid \boldsymbol { x } _ { j } )$ assumed in Equation (3), we obtain

$$
\mathbb { E } [ e ^ { y _ { j _ { 1 } } } \otimes \cdot \cdot \cdot \otimes e ^ { y _ { j _ { r } } } \mid \mathbf { B } ] = \bigotimes _ { \ell = 1 } ^ { r } \mathbb { E } [ e ^ { y _ { j _ { \ell } } } \mid x _ { j _ { \ell } } ] .\tag{38}
$$

By definition of the Bayes-optimal classifier,

$$
\mathbb { E } [ e ^ { y _ { j } } \mid x _ { j } ] = p ( \cdot \mid x _ { j } ) = f ^ { * } ( x _ { j } ) .\tag{39}
$$

Hence

$$
\mathbb { E } [ e ^ { y _ { j _ { 1 } } } \otimes \cdot \cdot \cdot \otimes e ^ { y _ { j _ { r } } } \mid \mathbf { B } ] = \pmb { f } ^ { * } ( x _ { j _ { 1 } } ) \otimes \cdot \cdot \cdot \otimes \pmb { f } ^ { * } ( x _ { j _ { r } } ) ,\tag{40}
$$

and therefore

$$
\underline { { \mathbb { E } } } [ Z _ { r } \mid \mathbf { B } ] = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } { f ^ { * } ( x _ { j _ { 1 } } ) \otimes \dots \otimes f ^ { * } ( x _ { j _ { r } } ) } = \widehat { \mathbf { M } } _ { { f ^ { * } } } ^ { ( r ) } ( \mathbf { B } ) ,\tag{41}
$$

which proves Equation (36).

Now let $f \in \mathcal { F } _ { \mathrm { M M } } ^ { ( s ) }$ be any minimizer of Equation (13). For each r, define

$$
V _ { r } ( { \bf B } ) : = \widehat { \bf M } _ { f } ^ { ( r ) } ( { \bf B } ) .\tag{42}
$$

Then the multi-order risk can be written as

$$
R _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) = \mathbb { E } _ { \mathbf { B } } \left[ \sum _ { r = 1 } ^ { s } \lambda _ { r } \mathbb { E } \Big [ \mathcal { L } ^ { ( r ) } ( Z _ { r } , V _ { r } ( \mathbf { B } ) ) \mid \mathbf { B } \Big ] \right] .\tag{43}
$$

Fix r, and define

$$
g _ { r } ( v , \mathbf { B } ) : = \mathbb { E } \left\lceil \mathcal { L } ^ { ( r ) } ( Z _ { r } , v ) \mid \mathbf { B } \right\rceil .\tag{44}
$$

Applying the loss assumption of the theorem to the conditional distribution of $Z _ { r }$ given B, the unique minimizer of $v \mapsto g _ { r } ( v , \mathbf { B } )$ is the conditional expectation $\mathbb { E } [ Z _ { r } \mid \mathbf { B } ]$ . By Equation (36), this unique minimizer is exactly $\widehat { \mathbf { M } } _ { f ^ { * } } ^ { ( r ) } ( \mathbf { B } )$ ). Therefore, for every $v \in \mathcal { T } _ { r }$

$$
g _ { r } ( v , \mathbf { B } ) \geq g _ { r } \Big ( \widehat { \mathbf { M } } _ { \pmb { f } ^ { * } } ^ { ( r ) } ( \mathbf { B } ) , \mathbf { B } \Big ) ,\tag{45}
$$

with equality if and only if

$$
v = \widehat { \mathbf { M } } _ { f ^ { * } } ^ { ( r ) } ( \mathbf { B } ) .\tag{46}
$$

Since $f$ is a minimizer of Equation (13) and $f ^ { * }$ is feasible, we also have

$$
R _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) \leq R _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ^ { \ast } ) .\tag{47}
$$

On the other hand, by the pointwise lower bound above,

$$
R _ { \mathrm { M M } } ^ { ( s ) } ( \mathbf { f } ) - R _ { \mathrm { M M } } ^ { ( s ) } ( \mathbf { f } ^ { * } ) = \mathbb { E } _ { \mathbf { B } } \left[ \sum _ { r = 1 } ^ { s } \lambda _ { r } \left[ g _ { r } ( V _ { r } ( \mathbf { B } ) , \mathbf { B } ) - g _ { r } \Big ( \widehat { \mathbf { M } } _ { \mathbf { f } ^ { * } } ^ { ( r ) } ( \mathbf { B } ) , \mathbf { B } \Big ) \right] \right]\tag{48}
$$

(49)

Therefore,

$$
\mathbb { E } _ { \mathbf { B } } \left[ \sum _ { r = 1 } ^ { s } \lambda _ { r } \left[ g _ { r } ( V _ { r } ( \mathbf { B } ) , \mathbf { B } ) - g _ { r } \Big ( \widehat { \mathbf { M } } _ { { f ^ { * } } } ^ { ( r ) } ( \mathbf { B } ) , \mathbf { B } \Big ) \right] \right] = 0 .\tag{50}
$$

The integrand is nonnegative almost surely, so it must vanish almost surely. Hence, for every r with $\lambda _ { r } > 0$

$$
g _ { r } ( V _ { r } ( \mathbf { B } ) , \mathbf { B } ) = g _ { r } \left( \widehat { \mathbf { M } } _ { { f ^ { * } } } ^ { ( r ) } ( \mathbf { B } ) , \mathbf { B } \right)\tag{51}
$$

for $p ( \mathbf { B } )$ -almost every bag B. By the uniqueness of the minimizer of $g _ { r } ( \cdot , \mathbf { B } )$ , this implies

$$
V _ { r } ( \mathbf { B } ) = \widehat { \mathbf { M } } _ { f ^ { * } } ^ { ( r ) } ( \mathbf { B } )\tag{52}
$$

for $p ( \mathbf { B } )$ -almost every bag B and every r such that $\lambda _ { r } > 0 .$ . Equivalently,

$$
\widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) = \widehat { \mathbf { M } } _ { f ^ { * } } ^ { ( r ) } ( \mathbf { B } ) , \qquad \mathrm { f o r } p ( \mathbf { B } ) \mathrm { - a l m o s t } \mathrm { e v e r y } \mathrm { b a g } \mathbf { B } \mathrm { ~ a n d ~ e v e r y } r \mathrm { s u c h } \mathrm { t h a t } \lambda _ { r } > 0 .\tag{53}
$$

This proves that any minimizer $f$ induces exactly the same moment tensors as $f ^ { * }$ for all orders with positive weight. □

ProofofProposition 3.1. By Theorem D.1, any $f \in \mathcal { F } _ { \mathrm { M M } } ^ { ( s _ { 2 } ) }$ matches the Bayes-optimal classifierinduced moment tensors for every order $r \in \{ 1 , \ldots , s _ { 2 } \}$ , since $\lambda _ { r } > 0$ for all such r. In particular, it matches these moment tensors for every order $r \in \{ 1 , \ldots , s _ { 1 } \}$ . Therefore, $f$ also satisfies the moment-matching conditions required for membership in $\mathcal { F } _ { \mathrm { M M } } ^ { ( s _ { 1 } ) }$ . Hence,

$$
\mathcal { F } _ { \mathrm { M M } } ^ { ( s _ { 2 } ) } \subseteq \mathcal { F } _ { \mathrm { M M } } ^ { ( s _ { 1 } ) } .
$$

## E Generalization Analysis

Definition E.1 (Loss-composed hypothesis class and its Rademacher complexity). For each order $r \in \{ 1 , \ldots , s \}$ , define the bag-level hypothesis class

$$
\mathcal { G } ^ { ( r ) } : = \{ ( \mathbf { B } , \pmb { \alpha } ) \mapsto \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } , \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) \Big ) : \ f \in \mathcal { F } \} .\tag{54}
$$

Its expected Rademacher complexity is defined as

$$
\Re _ { n } ^ { ( r ) } ( \mathcal { G } ^ { ( r ) } ) : = \mathbb { E } _ { \mathcal { D } } \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \mathcal { L } ^ { ( r ) } \left( \widehat { \mathbf { M } } _ { \alpha _ { i } } ^ { ( r ) } , \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } _ { i } ) \right) \right] ,\tag{55}
$$

where $\mathbfcal { D } = \{ ( \mathbf { B } _ { i } , \pmb { \alpha } _ { i } ) \} _ { i = 1 } ^ { n }$ consists of n i.i.d. samples from $p ( \mathbf { B } , \pmb { \alpha } )$ , and $\sigma _ { 1 } , \ldots , \sigma _ { n }$ are i.i.d.   
Rademacher variables.

Theorem E.2 (Generic generalization bound via loss-composed classes). Assume that for each $r \in \{ 1 , \ldots , s \}$ ,

$$
0 \leq \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } , \widehat { \mathbf { M } } _ { \pmb { f } } ^ { ( r ) } ( \mathbf { B } ) \Big ) \leq B _ { r }\tag{56}
$$

for all $( \mathbf { B } , \alpha )$ and all $\pmb { f } \in \mathcal { F }$ . Then, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \lvert R _ { \mathrm { M M } } ^ { ( s ) } ( f ) - \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( f ) \rvert \leq 2 \sum _ { r = 1 } ^ { s } \lambda _ { r } \Re _ { n } ^ { ( r ) } ( \mathcal { G } ^ { ( r ) } ) + \left( \sum _ { r = 1 } ^ { s } \lambda _ { r } B _ { r } \right) \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .\tag{57}
$$

Proof. Define the aggregated loss class

$$
\mathcal { H } : = \{ ( \mathbf { B } , \pmb { \alpha } ) \mapsto \sum _ { r = 1 } ^ { s } \lambda _ { r } \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \pmb { \alpha } } ^ { ( r ) } , \widehat { \mathbf { M } } _ { \pmb { f } } ^ { ( r ) } ( \mathbf { B } ) \Big ) : \ \pmb { f } \in \mathcal { F } \} .\tag{58}
$$

Then

$$
R _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) = \mathbb { E } [ h _ { \pmb { f } } ( \mathbf { B } , \pmb { \alpha } ) ] , \qquad \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } h _ { \pmb { f } } ( \mathbf { B } _ { i } , \pmb { \alpha } _ { i } ) ,\tag{59}
$$

where $h _ { f } \in \mathcal { H }$

Define the two one-sided deviations

$$
\begin{array} { r } { U ( \mathcal { D } ) : = \underset { \pmb { f } \in \mathcal { F } } { \operatorname* { s u p } } \Big ( R _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) - \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) \Big ) , } \\ { V ( \mathcal { D } ) : = \underset { \pmb { f } \in \mathcal { F } } { \operatorname* { s u p } } \Big ( \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) - R _ { \mathrm { M M } } ^ { ( s ) } ( \pmb { f } ) \Big ) . } \end{array}
$$

Applying the standard one-sided symmetrization inequality [Mohri et al., 2018] separately in the two directions gives

$$
\begin{array} { r } { \mathbb { E } _ { \mathcal { D } } U ( \mathcal { D } ) \leq 2 \Re _ { n } ( \mathcal { H } ) , \qquad \mathbb { E } _ { \mathcal { D } } V ( \mathcal { D } ) \leq 2 \Re _ { n } ( \mathcal { H } ) , } \end{array}\tag{60}
$$

where

$$
\Re _ { n } ( \mathcal { H } ) = \mathbb { E } _ { \mathcal { D } } \mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { s u p } _ { { \pmb { f } } \in \mathcal { F } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } h _ { { \pmb { f } } } ( \mathbf { B } _ { i } , \pmb { \alpha } _ { i } ) \right] .\tag{61}
$$

The reverse direction uses the symmetry of the Rademacher signs: replacing $\sigma \operatorname { b y } - \sigma$ leaves their distribution unchanged. Thus both inequalities use the same complexity defined above, without an absolute value inside the supremum. Using the definition of $h _ { f }$ and the subadditivity of the supremum,

$$
\begin{array} { r l } & { \Re _ { n } ( \mathcal { H } ) = \mathbb { E } _ { \mathcal { D } } \mathbb { E } _ { \sigma } \Bigg [ \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \sum _ { r = 1 } ^ { s } \lambda _ { r } \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \alpha _ { i } } ^ { ( r ) } , \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } _ { i } ) \Big ) \Bigg ] } \\ & { \qquad \leq \displaystyle \sum _ { r = 1 } ^ { s } \lambda _ { r } \mathbb { E } _ { \mathcal { D } } \mathbb { E } _ { \sigma } \Bigg [ \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \alpha _ { i } } ^ { ( r ) } , \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } _ { i } ) \Big ) \Bigg ] } \\ & { \qquad = \displaystyle \sum _ { r = 1 } ^ { s } \lambda _ { r } \Re _ { n } ^ { ( r ) } ( \mathcal { G } ^ { ( r ) } ) . } \end{array}
$$

Hence,

$$
\operatorname* { m a x } \{ \mathbb { E } _ { \mathcal { D } } U ( \mathcal { D } ) , \mathbb { E } _ { \mathcal { D } } V ( \mathcal { D } ) \} \leq 2 \sum _ { r = 1 } ^ { s } \lambda _ { r } \Re _ { n } ^ { ( r ) } ( \mathcal { G } ^ { ( r ) } ) .\tag{62}
$$

Now let

$$
C : = \sum _ { r = 1 } ^ { s } \lambda _ { r } B _ { r } .\tag{63}
$$

For every $\pmb { f } \in \mathcal { F }$ , the aggregated loss

$$
h _ { f } ( \mathbf { B } , \alpha ) = \sum _ { r = 1 } ^ { s } \lambda _ { r } \mathscr { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } , \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) \Big )\tag{64}
$$

satisfies $0 \leq h _ { f } \leq C$ . Therefore, if D and $\mathcal { D } ^ { \prime }$ differ in one bag-proportion pair $( \mathbf { B } _ { i } , \pmb { \alpha } _ { i } )$ , then

$$
\bigl | U ( \mathscr { D } ) - U ( \mathscr { D } ^ { \prime } ) \bigr | \leq \frac { C } { n } , \qquad \bigl | V ( \mathscr { D } ) - V ( \mathscr { D } ^ { \prime } ) \bigr | \leq \frac { C } { n } .\tag{65}
$$

The independent sample units are the n entire bag-proportion pairs. Applying McDiarmid’s inequality [Mohri et al., 2018] separately to $U ( \mathcal D )$ and $V \bar { ( \mathcal { D } ) }$ , with failure probability $\delta / 2$ for each, gives

$$
\operatorname* { P r } \left( W ( \mathcal { D } ) > \mathbb { E } _ { \mathcal { D } } W ( \mathcal { D } ) + C \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } \right) \leq \frac { \delta } { 2 } , \qquad W \in \{ U , V \} .
$$

By a union bound and the two symmetrization inequalities, with probability at least $1 - \delta ,$ , both directions simultaneously satisfy

$$
\operatorname* { m a x } \{ U ( \mathcal D ) , V ( \mathcal D ) \} \leq 2 \Re _ { n } ( \mathcal H ) + C \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .\tag{66}
$$

Finally, using

$$
\operatorname* { s u p } _ { { \pmb f } \in \mathcal { F } } \left| R _ { \mathrm { M M } } ^ { ( s ) } ( { \pmb f } ) - \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( { \pmb f } ) \right| = \operatorname* { m a x } \{ U ( \mathcal { D } ) , V ( \mathcal { D } ) \} ,
$$

together with the subadditivity bound for $\Re _ { n } ( \mathcal { H } )$ and the definition of C, proves Equation (57).

Definition E.3 (Coordinate-wise Rademacher complexity of $\mathcal { F } )$ . For each position $j \in [ m ]$ and class $c \in [ q ]$ , define the scalar-valued class

$$
{ \mathcal { F } } _ { j , c } : = \{ \mathbf { B } = ( x _ { 1 } , \ldots , x _ { m } ) \mapsto [ { \pmb f } ( x _ { j } ) ] _ { c } : { \pmb f } \in { \mathcal { F } } \} .\tag{67}
$$

Its expected Rademacher complexity is

$$
\Re _ { n } ( \mathcal { F } _ { j , c } ) : = \mathbb { E } _ { \left\{ \mathbf { B } _ { i } \right\} _ { i = 1 } ^ { n } } \mathbb { E } _ { \left\{ \sigma _ { i } \right\} _ { i = 1 } ^ { n } } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left[ \pmb { f } ( x _ { i j } ) \right] _ { c } \right] .\tag{68}
$$

Lemma E.4 (Lipschitz property of the moment map). For any $r \in \{ 1 , \ldots , m \}$ , any bag ${ \textbf { B } } =$ $( x _ { 1 } , \ldots , x _ { m } )$ , and any $\boldsymbol { f } , \boldsymbol { f } ^ { \prime } \in \mathcal { F } ,$ , we have

$$
\| \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) - \widehat { \mathbf { M } } _ { f ^ { \prime } } ^ { ( r ) } ( \mathbf { B } ) \| _ { F } \leq \frac { r } { m } \sum _ { j = 1 } ^ { m } \| f ( x _ { j } ) - f ^ { \prime } ( x _ { j } ) \| _ { 2 } \leq \frac { r } { m } \sum _ { j = 1 } ^ { m } \sum _ { c = 1 } ^ { q } \bigl | [ f ( x _ { j } ) ] _ { c } - [ f ^ { \prime } ( x _ { j } ) ] _ { c } \bigr | ,\tag{69}
$$

where $\left\| \cdot \right\| _ { F }$ denotes the Frobenius norm ofan order-r tensor, i.e.,for $T = ( T _ { i _ { 1 } , \dots , i _ { r } } ) \in \mathbb { R } ^ { q \times \cdots \times q }$

$$
\Vert T \Vert _ { F } : = \left( \sum _ { i _ { 1 } , \dots , i _ { r } } T _ { i _ { 1 } , \dots , i _ { r } } ^ { 2 } \right) ^ { 1 / 2 } .\tag{70}
$$

Proof. Let $u _ { j } : = f ( x _ { j } ) \in \Delta ^ { q - 1 }$ and $v _ { j } : = f ^ { \prime } ( x _ { j } ) \in \Delta ^ { q - 1 }$ . For any ordered r-tuple $( j _ { 1 } , \ldots , j _ { r } ) \in$ $[ m ] _ { \neq } ^ { r }$ , write

$$
U : = u _ { j _ { 1 } } \otimes \cdot \cdot \cdot \otimes u _ { j _ { r } } , \qquad V : = v _ { j _ { 1 } } \otimes \cdot \cdot \cdot \otimes v _ { j _ { r } } .\tag{71}
$$

A telescoping expansion gives

$$
U - V = \sum _ { t = 1 } ^ { r } v _ { j _ { 1 } } \otimes \cdot \cdot \cdot \otimes v _ { j _ { t - 1 } } \otimes ( u _ { j _ { t } } - v _ { j _ { t } } ) \otimes u _ { j _ { t + 1 } } \otimes \cdot \cdot \cdot \otimes u _ { j _ { r } } .\tag{72}
$$

Using

$$
\| a _ { 1 } \otimes \cdot \cdot \cdot \otimes a _ { r } \| _ { F } = \prod _ { \ell = 1 } ^ { r } \| a _ { \ell } \| _ { 2 }\tag{73}
$$

and the fact that $\| u _ { j } \| _ { 2 } \leq \| u _ { j } \| _ { 1 } = 1$ and $\| v _ { j } \| _ { 2 } \leq \| v _ { j } \| _ { 1 } = 1$ for all $j ,$ we obtain

$$
\| U - V \| _ { F } \le \sum _ { t = 1 } ^ { r } \| u _ { j _ { t } } - v _ { j _ { t } } \| _ { 2 } .\tag{74}
$$

Hence

$$
\begin{array} { l } { \| \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) - \widehat { \mathbf { M } } _ { f ^ { \prime } } ^ { ( r ) } ( \mathbf { B } ) \| _ { F } \leq \displaystyle \frac { 1 } { ( m ) _ { r } } \displaystyle \sum _ { ( j _ { 1 } , \ldots , j _ { r } ) \in [ m ] _ { \sharp } ^ { r } } \displaystyle \sum _ { t = 1 } ^ { r } \| u _ { j _ { t } } - v _ { j _ { t } } \| _ { 2 } } \\ { = \displaystyle \frac { r ( m - 1 ) _ { r - 1 } } { ( m ) _ { r } } \displaystyle \sum _ { j = 1 } ^ { m } \| u _ { j } - v _ { j } \| _ { 2 } } \\ { = \displaystyle \frac { r } { m } \displaystyle \sum _ { j = 1 } ^ { m } \| u _ { j } - v _ { j } \| _ { 2 } . } \end{array}
$$

The second inequality in Equation (69) follows from $\begin{array} { r } { \| { \boldsymbol { a } } \| _ { 2 } \leq \| { \boldsymbol { a } } \| _ { 1 } = \sum _ { c = 1 } ^ { q } | a _ { c } | . } \end{array}$

Lemma E.5 (Coordinate-wise contraction). Let $T \subset ( \mathbb { R } ^ { d } ) ^ { n }$ . For each $i \in [ n ]$ , let $\Phi _ { i } : \mathbb { R } ^ { d }  \mathbb { R }$ satisfy $\Phi _ { i } ( \mathbf { 0 } ) = 0$ and

$$
| \Phi _ { i } ( u ) - \Phi _ { i } ( v ) | \leq L \| u - v \| _ { 1 } \qquad f o r a l l u , v \in \mathbb { R } ^ { d } .\tag{75}
$$

Then

$$
\mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { s u p } _ { t \in T } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \Phi _ { i } ( t _ { i } ) \right] \leq \sqrt { 2 d } L \sum _ { \ell = 1 } ^ { d } \mathbb { E } _ { \pmb { \tau } ^ { ( \ell ) } } \left[ \operatorname* { s u p } _ { t \in T } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } ^ { ( \ell ) } t _ { i , \ell } \right] .\tag{76}
$$

Proof. Since $\| x \| _ { 1 } \leq { \sqrt { d } } \| x \| _ { 2 }$ for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , the assumption implies

$$
| \Phi _ { i } ( u ) - \Phi _ { i } ( v ) | \leq L \sqrt { d } \| u - v \| _ { 2 } \qquad \mathrm { f o r ~ a l l ~ } u , v \in \mathbb { R } ^ { d } .\tag{77}
$$

Hence each $\Phi _ { i }$ is $L \sqrt { d } \mathrm { - } \mathrm { I }$ ipschitz with respect to the Euclidean norm.

Applying the vector contraction inequality [Maurer, 2016] with respect to $\| \cdot \| _ { 2 }$ yields

$$
\mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { t \in T } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \Phi _ { i } ( t _ { i } ) \right] \leq \sqrt { 2 d } L \mathbb { E } _ { \epsilon } \left[ \operatorname* { s u p } _ { t \in T } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { d } \epsilon _ { i , \ell } t _ { i , \ell } \right] ,\tag{78}
$$

where $\epsilon _ { i , \ell }$ are i.i.d. Rademacher variables.

Finally, by subadditivity of the supremum,

$$
\mathbb { E } _ { \epsilon } \left[ \operatorname* { s u p } _ { t \in T } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { \ell = 1 } ^ { d } \epsilon _ { i , \ell } t _ { i , \ell } \right] \leq \sum _ { \ell = 1 } ^ { d } \mathbb { E } _ { \epsilon } \left[ \operatorname* { s u p } _ { t \in T } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \epsilon _ { i , \ell } t _ { i , \ell } \right]\tag{79}
$$

$$
= \sum _ { \ell = 1 } ^ { d } \mathbb { E } _ { \tau ^ { ( \ell ) } } \left[ \operatorname* { s u p } _ { t \in T } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \tau _ { i } ^ { ( \ell ) } t _ { i , \ell } \right] .\tag{80}
$$

This proves the claim.

Proposition E.6 (Reduction to the base class ${ \mathcal { F } } ) .$ . Assume that for each $r \in \{ 1 , \ldots , s \}$ , the loss

$$
B \mapsto \mathcal { L } ^ { ( r ) } ( A , B )\tag{81}
$$

is $L _ { r }$ -Lipschitz with respect to the Frobenius norm, uniformly over all admissible target tensors A. Then

$$
\Re _ { n } ^ { ( r ) } ( { \mathcal { G } } ^ { ( r ) } ) \leq \sqrt { 2 m } \frac { L _ { r } r } { m } \sum _ { j = 1 } ^ { m } \sum _ { c = 1 } ^ { q } \Re _ { n } ( { \mathcal { F } } _ { j , c } ) .\tag{82}
$$

Proof. Fix a sample $\{ ( \mathbf { B } _ { i } , \pmb { \alpha } _ { i } ) \} _ { i = 1 } ^ { n }$ . For a bag $\mathbf { B } = ( x _ { 1 } , \dots , x _ { m } )$ and a classifier $f ,$ write

$$
\pmb { u } _ { j } = \pmb { f } ( x _ { j } ) \in \Delta ^ { q - 1 } , \qquad \pmb { u } = ( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { m } ) \in ( \mathbb { R } ^ { q } ) ^ { m } .
$$

We first show that the classifier-induced order-r factorial moment tensor is $r / \sqrt { m } \mathrm { - I }$ ipschitz as a function of u with respect to the Euclidean norm. For $u , v \in ( \Delta ^ { q - 1 } ) ^ { m }$ , define

$$
\widehat { \mathbf { M } } ^ { ( r ) } ( \pmb { u } ) = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \pmb { u } _ { j _ { 1 } } \otimes \cdot \cdot \cdot \otimes \pmb { u } _ { j _ { r } } ,
$$

where $[ m ] _ { \neq } ^ { r }$ denotes the set of ordered r-tuples of distinct indices. By the triangle inequality and a telescoping expansion of tensor products,

$$
\begin{array} { l } { \displaystyle \Big \| \widehat { \mathbf { M } } ^ { ( r ) } ( \pmb { u } ) - \widehat { \mathbf { M } } ^ { ( r ) } ( \pmb { v } ) \Big \| _ { F } \leq \frac { 1 } { ( m ) _ { r } } \displaystyle \sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \Big \| \pmb { u } _ { j _ { 1 } } \otimes \dots \otimes \pmb { u } _ { j _ { r } } - \pmb { v } _ { j _ { 1 } } \otimes \dots \otimes \pmb { v } _ { j _ { r } } \Big \| _ { F } } \\ { \leq \frac { 1 } { ( m ) _ { r } } \displaystyle \sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \displaystyle \sum _ { \ell = 1 } ^ { r } \Big \| \pmb { u } _ { j _ { \ell } } - \pmb { v } _ { j _ { \ell } } \Big \| _ { 2 } . } \end{array}\tag{83}
$$

The second inequality uses the fact that $\| \pmb { u } _ { j } \| _ { 2 } \le 1$ and $\| \pmb { v } _ { j } \| _ { 2 } \le 1$ for probability vectors. For each fixed index $j \in [ m ]$ and each position $\bar { \ell } \in \left. 1 , \ldots , r \right.$ , the number of ordered tuples $( j _ { 1 } , \ldots , j _ { r } ) \in$ $[ m ] _ { \neq } ^ { r }$ satisfying $j _ { \ell } = j$ is $( m - 1 ) _ { r - 1 }$ . Therefore,

$$
\begin{array} { r l r } {  { \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \sum _ { \ell = 1 } ^ { r } \| { \boldsymbol u } _ { j _ { \ell } } - { \boldsymbol v } _ { j _ { \ell } } \| _ { 2 } = \frac { r ( m - 1 ) _ { r - 1 } } { ( m ) _ { r } } \sum _ { j = 1 } ^ { m } \| { \boldsymbol u } _ { j } - { \boldsymbol v } _ { j } \| _ { 2 } } } \\ & { } & { = \frac { r } { m } \sum _ { j = 1 } ^ { m } \| { \boldsymbol u } _ { j } - { \boldsymbol v } _ { j } \| _ { 2 } . } \end{array}\tag{84}
$$

Applying Cauchy–Schwarz over the m bag instances gives

$$
\sum _ { j = 1 } ^ { m } \left\| { \pmb u } _ { j } - { \pmb v } _ { j } \right\| _ { 2 } \leq \sqrt { m } \left( \sum _ { j = 1 } ^ { m } \left\| { \pmb u } _ { j } - { \pmb v } _ { j } \right\| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } = \sqrt { m } \left\| { \pmb u } - { \pmb v } \right\| _ { 2 } .\tag{85}
$$

Combining this with Equations (83) and (84), we obtain

$$
\left\| \widehat { \mathbf { M } } ^ { ( r ) } ( \pmb { u } ) - \widehat { \mathbf { M } } ^ { ( r ) } ( \pmb { v } ) \right\| _ { F } \leq \frac { r } { \sqrt { m } } \| \pmb { u } - \pmb { v } \| _ { 2 } .\tag{86}
$$

Now define, for each training bag i,

$$
\Phi _ { i } ^ { ( r ) } ( \pmb { u } ) = \mathcal { L } ^ { ( r ) } \Big ( \widehat { \mathbf { M } } _ { \alpha _ { i } } ^ { ( r ) } , \widehat { \mathbf { M } } ^ { ( r ) } ( \pmb { u } ) \Big ) .
$$

By the assumed $L _ { r }$ -Lipschitzness of $\boldsymbol { \mathcal { L } ^ { ( r ) } }$ in its second argument and by Equation (86), each $\Phi _ { i } ^ { ( r ) }$ is $\dot { L _ { r } } r / \sqrt { m } { - \mathrm { I } }$ Lipschitz with respect to the Euclidean norm on $\left( \mathbb { R } ^ { q } \right) ^ { m }$ . Therefore, by Maurer’s vector contraction inequality [Maurer, 2016],

$$
\begin{array} { l } { \displaystyle \mathfrak { N } _ { n } ^ { ( r ) } ( \mathcal { G } ^ { ( r ) } ) \leq \sqrt { 2 } \frac { L _ { r } r } { \sqrt { m } } \mathbb { E } _ { \sigma } \left[ \displaystyle \operatorname* { s u p } _ { f \in \mathcal { F } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } \sum _ { c = 1 } ^ { q } \sigma _ { i , j , c } f _ { c } ( x _ { i , j } ) \right] } \\ { \leq \sqrt { 2 } \frac { L _ { r } r } { \sqrt { m } } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { c = 1 } ^ { q } \mathfrak { N } _ { n } ( \mathcal { F } _ { j , c } ) } \\ { = \sqrt { 2 m } \frac { L _ { r } r } { m } \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { c = 1 } ^ { q } \mathfrak { N } _ { n } ( \mathcal { F } _ { j , c } ) . } \end{array}\tag{87}
$$

This proves the desired bound.

Corollary E.7 (Generalization bound in terms of F). Under the assumptions of Theorem E.2 and Proposition E.6, for any $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta ,$

$$
\begin{array} { r l } & { \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \left| R _ { \mathrm { M M } } ^ { ( s ) } ( f ) - \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( f ) \right| } \\ & { \leq \frac { 2 \sqrt { 2 m } } { m } \left( \displaystyle \sum _ { r = 1 } ^ { s } \lambda _ { r } L _ { r } r \right) \left( \displaystyle \sum _ { j = 1 } ^ { m } \displaystyle \sum _ { c = 1 } ^ { q } \Re _ { n } ( \mathcal { F } _ { j , c } ) \right) } \\ & { \quad + \left( \displaystyle \sum _ { r = 1 } ^ { s } \lambda _ { r } B _ { r } \right) \sqrt { \displaystyle \frac { \log ( 2 / \delta ) } { 2 n } } . } \end{array}\tag{88}
$$

Proof. Substitute the bound in Proposition E.6 into Theorem E.2.

Remark E.8 (Exchangeable bag coordinates). Under our bag-generation mechanism, the instances within each bag need not be independent. If the construction is permutation invariant, or if each bag is randomly permuted afterward, its coordinates are exchangeable. Thus, for every fixed $j \in [ m ] , x _ { j }$ has the same marginal distribution, although the coordinates may remain dependent. Hence, for each class $c \in [ q ]$ ], the classes $\mathcal { F } _ { j , c }$ have the same expected Rademacher complexity:

$$
\Re _ { n } ( \mathcal { F } _ { 1 , c } ) = \cdot \cdot \cdot = \Re _ { n } ( \mathcal { F } _ { m , c } ) .\tag{89}
$$

Consequently, the complexity term in Corollary E.7 can be simplified.

Definition E.9 (Instance-level Rademacher complexity). For each class $c \in [ q ]$ , define the scalarvalued instance-level class

$$
{ \mathcal { F } } _ { c } : = \{ x \mapsto [ { \pmb f } ( x ) ] _ { c } \mid { \pmb f } \in { \mathcal { F } } \} .\tag{90}
$$

Its expected Rademacher complexity is

$$
\mathfrak { R } _ { n } ( \mathcal { F } _ { c } ) : = \mathbb { E } _ { \{ x _ { i } \} _ { i = 1 } ^ { n } } \mathbb { E } _ { \{ \sigma _ { i } \} _ { i = 1 } ^ { n } } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left[ \pmb { f } ( \boldsymbol { x } _ { i } ) \right] _ { c } \right] ,\tag{91}
$$

where $x _ { 1 } , \ldots , x _ { n }$ are i.i.d. draws from the common marginal distribution of any bag coordinate.

Corollary E.10 (Simplified generalization bound under exchangeability). Under the assumptions of Corollary $E . 7$ and the exchangeability condition in Remark $E . { \bar { 8 } } ,$ for any $\delta \in ( 0 , 1 )$ , with probability at least 1 − δ,

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \Big | R _ { \mathrm { M M } } ^ { ( s ) } ( f ) - \widehat { R } _ { \mathrm { M M } } ^ { ( s ) } ( f ) \Big | \leq 2 \sqrt { 2 m } \left( \sum _ { r = 1 } ^ { s } \lambda _ { r } L _ { r } r \right) \left( \sum _ { c = 1 } ^ { q } \Re _ { n } ( \mathcal { F } _ { c } ) \right) + \left( \sum _ { r = 1 } ^ { s } \lambda _ { r } B _ { r } \right) \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .\tag{92}
$$

## F Computation Analysis

## F.1 Coefficient-to-Tensor Correspondence

In this appendix, we justify the correspondence between the compressed coefficients of the multivariate generating polynomial and the entries of the classifier-induced factorial moment tensor.

For a bag $\mathbf { B } = ( x _ { 1 } , \dots , x _ { m } )$ , recall that the classifier-induced normalized r-th order factorial moment tensor is

$$
\widehat { \mathbf { M } } _ { \pmb { f } } ^ { ( r ) } ( \mathbf { B } ) = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } p _ { j _ { 1 } } \otimes \cdots \otimes p _ { j _ { r } } ,\tag{93}
$$

where $p _ { j } = f ( x _ { j } ) \in \Delta ^ { q - 1 }$ . Its $( i _ { 1 } , \ldots , i _ { r } )$ -th entry is therefore

$$
[ \widehat { \bf M } _ { f } ^ { ( r ) } ( { \bf B } ) ] _ { i _ { 1 } , \ldots , i _ { r } } = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \ldots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \prod _ { \ell = 1 } ^ { r } p _ { j _ { \ell } , i _ { \ell } } .\tag{94}
$$

Now define the multivariate generating polynomial

$$
G _ { m } ( u _ { 1 } , \dots , u _ { q } ) : = \prod _ { j = 1 } ^ { m } \left( 1 + \sum _ { c = 1 } ^ { q } p _ { j , c } u _ { c } \right) .\tag{95}
$$

For $\boldsymbol { k } = ( k _ { 1 } , \ldots , k _ { q } ) \in \mathbb { N } ^ { q }$ with $\begin{array} { r } { | k | = \sum _ { c = 1 } ^ { q } k _ { c } = r . } \end{array}$ , let

$$
A _ { m } ^ { ( r ) } ( k ) : = [ \boldsymbol { u } ^ { k } ] G _ { m } ( u _ { 1 } , \ldots , u _ { q } ) , \qquad \boldsymbol { u } ^ { k } : = \prod _ { c = 1 } ^ { q } u _ { c } ^ { k _ { c } } .\tag{96}
$$

We will show that $A _ { m } ^ { ( r ) } ( k )$ determines all tensor entries whose multiplicity vector is $k$

Proposition F.1 (Coefficient-to-tensor correspondence). Fix an ordered index tuple $( i _ { 1 } , \ldots , i _ { r } ) \in$ $[ q ] ^ { r }$ , and let

$$
k _ { c } : = \# \{ \ell \in [ r ] \mid i _ { \ell } = c \} , \qquad c = 1 , \ldots , q ,\tag{97}
$$

be its multiplicity vector. Then

$$
[ \widehat { \bf M } _ { f } ^ { ( r ) } ( { \bf B } ) ] _ { i _ { 1 } , \ldots , i _ { r } } = \frac { \prod _ { c = 1 } ^ { q } k _ { c } ! } { ( m ) _ { r } } A _ { m } ^ { ( r ) } ( k ) .\tag{98}
$$

Consequently, all tensor entries with the same multiplicity vector are equal, and the compressed coefficients $\{ A _ { m } ^ { ( r ) } ( k ) \} _ { k \in \mathcal { K } _ { q , r } }$ completely determine $\widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } )$ .

Proof. We first expand the generating polynomial in Equation (95). In each factor

$$
1 + \sum _ { c = 1 } ^ { q } p _ { j , c } u _ { c } ,\tag{99}
$$

we either choose the constant term 1, meaning that instance $j$ is not selected, or one of the terms $\mathit { p } _ { j , c } u _ { c } .$ , meaning that instance $j$ is selected and contributes one occurrence of class c.

Therefore, the coefficient of $u ^ { k }$ in $G _ { m }$ is obtained by summing over all ways to choose r distinct instances and assign to them class labels whose class-count profile is exactly k. Formally,

$$
A _ { m } ^ { ( r ) } ( k ) = \sum _ { \stackrel { S \subseteq [ m ] } { | S | = r } } \sum _ { \substack { \phi : S \to [ q ] : | \phi ^ { - 1 } ( c ) | = k _ { c } \forall c } } \prod _ { j \in S } p _ { j , \phi ( j ) } .\tag{100}
$$

Now fix the ordered tensor index tuple $( i _ { 1 } , \ldots , i _ { r } )$ . Let

$$
\mathfrak { S } ( i _ { 1 } , \ldots , i _ { r } )\tag{101}
$$

denote the set of distinct permutations of the multiset $( i _ { 1 } , \ldots , i _ { r } )$ . Its cardinality is

$$
| \mathfrak { S } ( i _ { 1 } , \dots , i _ { r } ) | = \frac { r ! } { \prod _ { c = 1 } ^ { q } k _ { c } ! } .\tag{102}
$$

For any ordered tuple of distinct instance indices $( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r }$ , define

$$
\Psi ( j _ { 1 } , \dots , j _ { r } ) : = \prod _ { \ell = 1 } ^ { r } p _ { j _ { \ell } , i _ { \ell } } .\tag{103}
$$

Then Equation (94) becomes

$$
[ \widehat { \bf M } _ { f } ^ { ( r ) } ( { \bf B } ) ] _ { i _ { 1 } , \ldots , i _ { r } } = \frac { 1 } { ( m ) _ { r } } \sum _ { ( j _ { 1 } , \ldots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \Psi ( j _ { 1 } , \ldots , j _ { r } ) .\tag{104}
$$

Next, we compare Equation (100) and Equation (104). Take any subset $S = \{ j _ { 1 } , \dots , j _ { r } \} \subseteq [ m ]$ of size r. To contribute to $A _ { m } ^ { ( r ) } ( k )$ , we assign class labels to the elements of S so that class c appears exactly $k _ { c }$ times. On the other hand, to contribute to $[ \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) ] _ { i _ { 1 } , \ldots , i _ { r } }$ , we must choose an ordered tuple of distinct instances $( j _ { 1 } , \ldots , j _ { r } )$ and match them to the fixed ordered label pattern $( i _ { 1 } , \ldots , i _ { r } )$ For a fixed subset $S$ and a fixed assignment of labels with multiplicity vector $k ,$ there are exactly $\textstyle \prod _ { c = 1 } ^ { q } k _ { c } !$ ordered tuples $( j _ { 1 } , \ldots , j _ { r } )$ that induce the same monomial contribution in Equation (94). Indeed, within each class $c ,$ the $k _ { c }$ positions carrying label c may be permuted arbitrarily among themselves without changing the underlying unlabeled assignment counted in Equation (100). Thus, each term counted once in $A _ { m } ^ { ( r ) } ( k )$ is counted exactly $\Pi _ { c = 1 } ^ { q } k _ { c } !$ times in the sum defining

$$
\sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \Psi ( j _ { 1 } , \dots , j _ { r } ) .\tag{105}
$$

Therefore,

$$
\sum _ { ( j _ { 1 } , \dots , j _ { r } ) \in [ m ] _ { \neq } ^ { r } } \prod _ { \ell = 1 } ^ { r } p _ { j _ { \ell } , i _ { \ell } } = \left( \prod _ { c = 1 } ^ { q } k _ { c } ! \right) A _ { m } ^ { ( r ) } ( k ) .\tag{106}
$$

Substituting Equation (106) into Equation (94) yields

$$
[ \widehat { \bf M } _ { f } ^ { ( r ) } ( { \bf B } ) ] _ { i _ { 1 } , \ldots , i _ { r } } = \frac { \prod _ { c = 1 } ^ { q } k _ { c } ! } { ( m ) _ { r } } A _ { m } ^ { ( r ) } ( k ) ,\tag{107}
$$

which proves Equation (98).

Finally, the right-hand side depends on $( i _ { 1 } , \ldots , i _ { r } )$ only through its multiplicity vector $k .$ Hence all tensor entries with the same multiplicity vector are equal, and the collection of compressed coefficients $\{ A _ { m } ^ { ( r ) } ( k ) \} _ { k \in \mathcal { K } _ { q , \ i } }$ completely determines the full tensor. This completes the proof.

## F.2 Time and Space Complexity Analysis

A direct evaluation of the order-r loss in Equation (13) requires materializing an order-r tensor with $q ^ { r }$ entries, which is computationally prohibitive for large q or r. To reduce both time and memory

costs, we exploit the permutation symmetry of the factorial moment tensors. Both $[ \widehat { \mathbf { M } } _ { \alpha } ^ { ( r ) } ] _ { i _ { 1 } , \dots , i _ { r } }$ and $[ \widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } ) ] _ { i _ { 1 } , \ldots , i _ { \tau } }$ depend only on $k ,$ rather than on the ordering of $( i _ { 1 } , \ldots , i _ { r } )$ . Hence, the full $\boldsymbol q ^ { r }$ ordered tensor entries can be grouped into the compressed state space

$$
\mathcal { K } _ { q , r } : = \Bigg \{ k \in \mathbb { N } ^ { q } \Bigg | \sum _ { c = 1 } ^ { q } k _ { c } = r \Bigg \} , \qquad N _ { r } : = \big | \mathcal { K } _ { q , r } \big | = \binom { q + r - 1 } { r } .\tag{108}
$$

The second-order case provides the simplest illustration: both $\widehat { \mathbf { M } } _ { \alpha } ^ { ( 2 ) }$ and $\widehat { \mathbf { M } } _ { f } ^ { ( 2 ) } ( \mathbf { B } )$ are symmetric $q \times q$ matrices, so the full $q ^ { 2 }$ entries reduce to only $N _ { 2 } = { \binom { q + 1 } { 2 } }$ distinct states. More generally, at order $r ,$ this reduces the computation from the full tensor size $\boldsymbol { q } ^ { r }$ to the compressed state size $N _ { r }$

To compute the classifier-induced states efficiently, we introduce the truncated multivariate generating polynomial

$$
G _ { t } ( u _ { 1 } , \dots , u _ { q } ) : = \prod _ { j = 1 } ^ { t } \left( 1 + \sum _ { c = 1 } ^ { q } p _ { j , c } u _ { c } \right) , \qquad t = 0 , 1 , \dots , m .\tag{109}
$$

For a state $k \in \mathcal { K } _ { q , d } .$ define

$$
A _ { t } ^ { ( d ) } ( \pmb { k } ) : = [ u ^ { \pmb { k } } ] G _ { t } ( u _ { 1 } , \dots , u _ { q } ) ,\tag{110}
$$

where $\begin{array} { r } { u ^ { k } : = \prod _ { c = 1 } ^ { q } u _ { c } ^ { k _ { c } } } \end{array}$ and $[ u ^ { k } ] G _ { t }$ denotes the coefficient of the monomial $u ^ { k }$ in $G _ { t }$ . Thus, $A _ { t } ^ { ( d ) } ( k )$ is the aggregated coefficient of all order-d terms whose class-count pattern is k after processing the first t instances.

These coefficients are directly related to the classifier-induced moment tensor. Indeed, for any ordered index tuple $( i _ { 1 } , \ldots , i _ { r } ) \in [ \dot { q ] } ^ { r }$ with multiplicity vector $k ,$ we have

$$
[ \widehat { \bf M } _ { f } ^ { ( r ) } ( { \bf B } ) ] _ { i _ { 1 } , . . . , i _ { r } } = \frac { \prod _ { c = 1 } ^ { q } k _ { c } ! } { ( m ) _ { r } } A _ { m } ^ { ( r ) } ( { \pmb k } ) .\tag{111}
$$

The derivation of Equation (111) is given in Appendix F.1. Hence, the compressed coefficients $\{ A _ { m } ^ { ( r ) } ( k ) \} _ { k \in \mathcal { K } _ { q , + } }$ completely determine the full tensor $\widehat { \mathbf { M } } _ { f } ^ { ( r ) } ( \mathbf { B } )$ . Since

$$
G _ { t } ( u ) = G _ { t - 1 } ( u ) \biggl ( 1 + \sum _ { c = 1 } ^ { q } p _ { t , c } u _ { c } \biggr ) ,\tag{112}
$$

the coefficients satisfy the recurrence

$$
A _ { t } ^ { ( d ) } ( \pmb { k } ) = A _ { t - 1 } ^ { ( d ) } ( \pmb { k } ) + \sum _ { c : \pmb { k } _ { c } > 0 } p _ { t , c } A _ { t - 1 } ^ { ( d - 1 ) } ( \pmb { k } - e ^ { c } ) ,\tag{113}
$$

with initialization

$$
A _ { 0 } ^ { ( 0 ) } ( 0 ) = 1 , \qquad A _ { 0 } ^ { ( d ) } ( k ) = 0 \quad \mathrm { f o r ~ a l l } \ d > 0 .\tag{114}
$$

Starting from this initialization, Equation (113) recursively computes all coefficients $A _ { t } ^ { ( d ) } ( k )$ for $t = 1 , \ldots , m$ and $d = 0 , \ldots , s ,$ , and in particular yields the desired coefficients $A _ { m } ^ { ( r ) } ( k )$ for all $r \leq s$ Proposition 3.3 (Complexity of the state-compressed computation). Fix a bag of size m and a maximum order $s \ \leq \ m$ . The compressed computation of all moments up to order s requires $O \big ( m q \big ( \begin{array} { c } { { q + s } } \\ { { s } } \end{array} \big ) \big )$ time and $O \big ( \big ( { } _ { s } ^ { q + s } \big ) \big )$  memory per bag.

Proof. For the target side, at a fixed order $r ,$ the compressed representation contains exactly $N _ { r } =$ $\binom { q + r - 1 } { r }$ states. By Equation (11), the value of each state is available in closed form from the multiplicity vector k and the proportion vector α. Computing one such value requires multiplying the contributions of all q classes, and thus costs $O ( q )$ time. Therefore, the total cost of computing the compressed target states at order r is $O ( q N _ { r } )$ , and summing over $r = 1 , \ldots , s$ gives

$$
O \Bigg ( q \sum _ { r = 1 } ^ { s } N _ { r } \Bigg ) = O \Bigg ( q \Bigg ( \begin{array} { c } { { q + s } } \\ { { s } } \end{array} \Bigg ) \Bigg ) .\tag{115}
$$

If all compressed target states up to order s are stored, the required memory is

$$
O \Bigg ( \sum _ { r = 1 } ^ { s } N _ { r } \Bigg ) = O \Bigg ( \Bigg ( { q + s } \Bigg ) \Bigg ) .\tag{116}
$$

For the classifier-induced side, we use the convention $N _ { 0 } = 1$ for the unique degree-0 state. At degree r there are exactly $N _ { r }$ compressed states. In the recurrence Equation (113), when processing one instance, each state of degree r can be extended to at most $q$ states of degree $r + 1$ by adding one unit to one coordinate. Hence the update cost per instance is

$$
O \left( q \sum _ { r = 0 } ^ { s - 1 } N _ { r } \right) ,\tag{117}
$$

and summing over all m instances yields

$$
O \Bigg ( m q \sum _ { r = 0 } ^ { s - 1 } N _ { r } \Bigg ) = O \Bigg ( m q \binom { q + s - 1 } { s - 1 } \Bigg ) .\tag{118}
$$

The corresponding memory cost is the size of the truncated coefficient table over degrees $0 , \ldots , s ,$ namely

$$
O \Bigg ( \sum _ { r = 0 } ^ { s } N _ { r } \Bigg ) = O \Bigg ( \Bigg ( { q + s } \Bigg ) \Bigg ) .\tag{119}
$$

Combining the target-side and classifier-induced costs gives the stated total time and memory bounds. Finally, since

$$
{ \binom { q + s - 1 } { s - 1 } } \leq { \binom { q + s } { s } } ,\tag{120}
$$

the total time can be simplified to

$$
O \biggl ( m q \biggl ( \begin{array} { c } { { q + s } } \\ { { s } } \end{array} \biggr ) \biggr ) .\tag{121}
$$

This completes the proof.

## G Additional Experimental Settings

## G.1 Datasets and Bag Construction

Following Ma et al. [2026b], we use three bag construction strategies: (i) Random Bag: randomly shuffle instances and partition them into bags; (ii) Cluster Bag: sample from feature clusters using Dirichlet mixing weights with total concentration 1 and a base measure proportional to cluster sizes; (iii) α-First Bag: sample class proportions from a symmetric Dirichlet distribution with concentration 10 per class, then draw instances accordingly. For all bag generation strategies, the bag size m varies over powers of two, i.e., $m \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ }.

## G.2 Baselines and Implementation Details

Baselines. We compare our method with several representative LLP baselines. PM [Yu et al., 2014, Ardehaly and Culotta, 2017] trains the model by matching the predicted class mean with the observed label proportions. DSQ [Li et al., 2024] builds upon PM by adopting a squared loss formulation with an additional debiasing term. LLP-PVC [Ma et al., 2026b] formulates LLP under a one-vs-rest framework and performs classification on bag-level proportions. LLP-FC [Zhang et al., 2022] approaches LLP from the perspective of label noise learning. ROT [Dulac-Arnold et al., 2019] leverages optimal transport to generate pseudo-labels that satisfy proportion constraints. EasyLLP [Busa-Fekete et al., 2023] derives an unbiased estimator by rewriting the loss function. GeneralUPM [Applebaum et al., 2025] also constructs an unbiased estimator but achieves lower variance. EasyLLP-ABS and GeneralUPM-ABS take the absolute value of the loss function to mitigate the overfitting effect caused by negative loss values. FlowLLP [Rakotomamonjy et al., 2026] learns latent particles by particle flow and uses them as classifier anchors.

Implementation details. We use ResNet-18 for CIFAR-10 and CIFAR-100 and ResNet-34 for miniImageNet, with modified stems and standard data augmentation. Training lasts 500 epochs with 1024 samples per update. The classifier optimizer is SGD with momentum 0.9 and weight decay 0.0005. The initial learning rate is 0.005 for LLP-PVC and 0.05 for the other methods, with linear warmup over 8% of updates followed by quarter-cosine decay. DSQ uses mean squared error; the other methods use cross-entropy as the surrogate loss. LLP-MM uses s = 8 for CIFAR-10 and $s = 3$ for CIFAR-100 and miniImageNet, with cross-entropy moment losses and weights $\lambda _ { r } = 1 / s$ LLP-FC uses approximate noisy priors, forward-correction weight 1, and entropy weight 0. FlowLLP uses a 50-dimensional latent space, 1000 anchors per class, and anchor-loss weight 0.1, following its official implementation [Rakotomamonjy et al., 2026]. Its particles are optimized for 3000 steps using Adam with a learning rate of 0.001. We report the highest finite test accuracy recorded in each run as mean ± sample standard deviation over three runs.

## G.3 Compatibility with GAN-Based Representation Learning

LLP-GAN [Liu et al., 2019] combines GAN-based representation learning with first-order proportion matching. Replacing its proportion loss with our order-eight moment-matching objective improves mean accuracy in all 12 CIFAR-10 settings in Table 6, with gains of up to 13.37 percentage points. However, compared with the standalone results, MM+GAN underperforms LLP-MM in most settings, while LLP-GAN does not consistently outperform PM. Both GAN-based variants generally exhibit larger standard deviations than their standalone counterparts, especially with large α-First bags. This variability may be related to instability in adversarial training. Implementation details follow.

Table 6: Best test accuracy on CIFAR-10 with GAN-based representation learning (mean ± std over three runs, %). The best result in each paired comparison is bold.
<table><tr><td>Bag</td><td>Method</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td>Random</td><td>LLP-GAN MM+GAN</td><td> $9 2 . 9 6 \pm 0 . 0 9$   ${ \bf 9 3 . 5 6 \pm 0 . 1 9 }$ </td><td> $8 8 . 9 6 \pm 0 . 3 5$   ${ \bf 9 1 . 1 8 \pm 0 . 2 2 }$ </td><td> $7 8 . 0 7 \pm 1 . 3 1$   ${ \bf 8 0 . 6 9 \pm 1 . 1 7 }$ </td><td> $5 8 . 1 4 \pm 2 . 9 4$   ${ \bf 6 7 . 4 1 \pm 1 . 4 1 }$ </td></tr><tr><td>Cluster</td><td>LLP-GAN MM+GAN</td><td> $9 2 . 8 1 \pm 0 . 1 5$   ${ \bf 9 3 . 6 2 \pm 0 . 0 5 }$ </td><td> $9 0 . 3 7 \pm 0 . 1 6$   ${ \bf 9 1 . 7 9 \pm 0 . 1 6 }$ </td><td> $8 5 . 1 6 \pm 0 . 7 7$   ${ \bf 8 7 . 1 7 \pm 0 . 3 8 }$ </td><td> $7 6 . 2 7 \pm 0 . 9 1$   ${ \bf 7 6 . 5 3 \pm 1 . 9 3 }$ </td></tr><tr><td>α-First</td><td>LLP-GAN MM+GAN</td><td> $9 1 . 0 4 \pm 0 . 5 9$   ${ \bf 9 3 . 3 9 \pm 0 . 0 8 }$ </td><td> $7 4 . 7 5 \pm 0 . 5 5$   $\mathbf { 8 8 . 1 2 \pm 1 . 0 9 }$ </td><td> $6 9 . 5 2 \pm 4 . 1 2$   $\pm { \bf 7 7 . 0 9 } \pm { \bf 4 . 1 7 }$ </td><td> $6 6 . 2 5 \pm 2 . 2 3$   ${ \bf 7 3 . 1 3 \pm 5 . 6 2 }$ </td></tr></table>

Implementation details. For the CIFAR-10 comparison in Table 6, LLP-GAN and MM+GAN share the same modified-stem ResNet-18 discriminator, generator, and bag assignments. Both are trained for 500 epochs with 1024 real images per update and unit weights for the bag and adversarial losses. The discriminator uses SGD with a learning rate of 0.05, momentum 0.9, and weight decay 0.0005, with five warmup epochs followed by standard cosine decay; the generator uses Adam with a learning rate of 0.0003, $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 5 , 0 . 9 9 9 )$ , and 100-dimensional noise. MM+GAN replaces only the proportion loss with equally weighted cross-entropy moment losses up to order $s = 8 .$ , retaining the adversarial and generator feature-matching objectives. We report the highest test accuracy recorded during training as mean ± sample standard deviation over three runs.

## G.4 Semi-Real-World Implementation Details

Bag construction. Cells from each known patient form one bag. We retain all 219 known-patient training bags and partition the 3,204 cells with unknown patient IDs into 26 bags of 123–124 cells using a fixed shuffle, independently of cell labels. The resulting 245 training bags have sizes 1–207, with a mean of 104.98; all 56 test bags are unchanged. Bag proportions are computed after grouping, and only proportions are used for training.

Training. We use ImageNet-pretrained ResNet-18 with its standard stem. Images use $2 2 4 \times 2 2 4$ inputs and ImageNet normalization, with horizontal and vertical flips during training. The classifier optimizer is Adam with a learning rate of 0.001. Training lasts 100 epochs, including five warmup epochs. Each update uses four complete bags, or five for GeneralUPM and GeneralUPM-ABS, with forward chunks of 32 images. LLP-MM uses orders $s \in \{ 3 , 5 , 8 \}$ ; orders exceeding the bag size are omitted and the remaining weights are renormalized. FlowLLP uses 50 pretraining and 50 joint-training epochs. Method-specific loss settings follow Appendix G.2.

Evaluation. We select the checkpoint with the highest test Macro-F1 in each run and evaluate all metrics at that checkpoint. Results are reported as mean ± sample standard deviation over three runs.

Metrics. Accuracy is the fraction of correctly classified cells, and Balanced Accuracy is the mean class-wise recall. Macro-F1 averages class-wise F1 scores equally, whereas Weighted-F1 weights them by class frequency in the test set.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes].

Justification: The abstract and introduction accurately summarize the paper’s methodological, theoretical, and empirical contributions without overstating the scope beyond the stated LLP setting and assumptions.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Sections 2.2, 3.4, and 5 for assumptions and limitations.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: Section 3.4 states the results; Appendices D-F provide the proofs.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 4 and Appendix G describe the datasets, baselines, evaluation protocols, model and training setup, and hyperparameter choices needed to reproduce the main experimental results.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes].

Justification: The experiments use publicly available datasets. The public repository linked in the abstract provides code, environment setup, data preparation instructions, and scripts for reproducing the main experimental results.

## Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips. cc/public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes].

Justification: Sections 4.1 and 4.3 and Appendix G specify the data splits, bag construction, optimizers, hyperparameters, training schedules, and evaluation protocols used in the experiments.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes].

Justification: Tables 1-3 and 6 report mean and standard deviation across three runs, with the summary statistic identified in each caption. Appendix G describes the corresponding training and evaluation protocols.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No].

Justification: Figure 2 reports hardware and per-epoch runtime, and Appendix G gives training schedules. Complete resource usage for each experiment and the total compute are not reported.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The paper does not delve into the broader societal impacts.

## Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper poses no such risks.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [No]

Justification: Assets are cited, but their licenses are not fully documented.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The public LLP-MM repository linked in the abstract includes a README with setup, data preparation, and run instructions, together with an MIT license.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments or research with human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: The paper does not involve crowdsourcing or research with human subjects, so IRB approval or equivalent review is not applicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: The core method development in this research does not involve LLMs as an important, original, or non-standard component.

Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.