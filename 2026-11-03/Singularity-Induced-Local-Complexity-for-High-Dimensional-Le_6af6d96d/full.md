# Singularity-Induced Local Complexity for High-Dimensional Learning Models

Atsuyoshi Muta<sup>1,2\*</sup> and Masaaki Imaizumi<sup>1,2,3</sup>

<sup>1\*</sup>The University of Tokyo, Tokyo, Japan. <sup>2</sup>RIKEN Center for Advanced Intelligence Project, Tokyo, Japan. <sup>3</sup>Kyoto University, Kyoto, Japan.

\*Corresponding author(s). E-mail(s): atsuyoshimuta@g.ecc.u-tokyo.ac.jp; Contributing authors: imaizumi@g.ecc.u-tokyo.ac.jp;

## Abstract

Many overparameterized models that have achieved remarkable success are singular models. That is, the set of optimal parameters consists of multiple points, or the Hessian of the generalization error at such parameters is degenerate, reflecting redundancy in the model parametrization. Therefore, understanding the efective dimension and intrinsic model complexity, beyond the apparent dimension of the parameter space, is essential for studying the generalization ability of such overparameterized models. In this work, we focus on the real log canonical threshold (RLCT) and its localized counterpart, the local RLCT, which are fundamental quantities for characterizing the generalization of singular models. We clarify the theoretical relationship between the local RLCT and existing two key metrics related to generalization under some localization assumption: (I) the rank of the Hessian matrix of the generalization error and (II) the covering number of a family of loss functions. This result provides novel insights into efective dimension and intrinsic complexity of overparameterized models.

Keywords: Overparameterized models, local model complexity, singular learning theory, real log canonical threshold, covering number, generalization bound

## 1 Introduction

## 1.1 Background

In machine learning, many learning models like neural networks are overparameterized, which have more parameters than samples, and are known to achieve good generalization performance on a wide range of tasks [1–3]. In contrast, classical learning theory suggests that an excess number of parameters can lead learning models to overfitting and prevents improvement in predictive performance [4, 5]. Therefore, the theoretical mechanisms underlying the generalization of overparameterized models is not yet fully understood, and this is an open problem [6, 7].

To tackle the problem, the singular learning theory [8–11] has been utilized to understand overparameterized models through a notion of model singularity. Here, model singularity means that a learning model has either of the two conditions [12]: (i) there exist multiple parameters that minimize the generalization error (non-identifiability), or (ii) the generalization error has a degenerate Hessian at its minimizers (non-regularity). The singular learning theory shows that, for example, a performacne of Bayesian methods for overparameterized models with model singularity is described by a key quantity named real log canonical threshold (RLCT) [10, 13, 14]. In particular, [13] pointed out that the RLCT behaves as an efective dimension in Bayesian inference and showed that it plays an essential role in generalization. To broaden its applicability, [9] introduced the local RLCT, also called the local learning coeficient (LLC), as a localized version of the RLCT, and empirically demonstrated its relationship with the generalization performance of overparameterized models [8, 9, 15, 16].

Despite the development, the studies of the singular learning theory still have some limitations. First, the relationship between the local RLCT and generalization performance has so far been established empirically, and theoretical guarantee of them has not yet been established. Furthermore, the theoretical validation for the (nonlocal) RLCT is limited to the Bayesian setting, so its applicability to a broad range of learning algorithms has been restricted.

## 1.2 Our study

In this work, we establish theoretical connections between the local RLCT and certain notions of efective dimension and complexity that are related to good generalization. We derive upper bounds using the local RLCT for two key metrics in the evaluation of learning models under suitable localization conditions: (I) the rank of the Hessian matrix of the generalization error and (II) the covering number of a family of loss functions by a learning model. These metrics are associated with efective dimension [17, 18] and model complexity [5, 6], respectively, and have both been studied as potential tools for explaining the generalization ability of overparameterized models. These results provide a theoretical explanation for the observed relationship between the local RLCT and the generalization error [9].

Our contributions can be summarized as follows.

• Hessian rank bound using the local RLCT (Theorem 1): We show that the local RLCT provides an upper bound on the rank of the Hessian of the generalization error at its local minimizers. Let $R ( w )$ denote the generalization error at a parameter $w \in W$ , and $w _ { * } \in W$ be a local minimizer of R. Then, by local RLCT $\lambda ( w _ { * } )$ at $w _ { * }$ satisfies

$$
\operatorname { r a n k } \nabla ^ { 2 } R ( w _ { * } ) \leq 2 \lambda ( w _ { * } ) .
$$

This result establishes a theoretical connection between existing measure of efective dimension and the local RLCT. It has been pointed out that the rank of the Hessian of the generalization error can be interpreted as a local efective dimension [17–20]. Indeed, this inequality implies that a smaller local RLCT also leads to a smaller Hessian rank.

• Local covering number bound (Theorem 2): We derive an upper bound in terms of the local RLCT on the local covering number around a grobal minimizer of generalization error. Informally, we establish a upper bound of the following form: let $w _ { 0 }$ be a parameter that is a global minimizer of the generalization error, then there exist $C > 0$ and $\rho > 0$ such that for any $\tau > \rho$

$$
N \left( \mathcal { F } _ { w _ { 0 } } , L ^ { \infty } ( P ) , \tau \right) \leq \left( 1 + \frac { C } { \tau - \rho } \right) ^ { 2 \lambda ( w _ { 0 } ) } .
$$

Here, $\mathcal { F } _ { w _ { 0 } }$ is the function space induced by parameters in a neighborhood of w<sub>0</sub> and $\lambda ( w _ { 0 } )$ is the local RLCT. Note that standard covering number bounds are typically considered for all $\tau > 0$ , whereas in this work we only consider the range $\tau > \rho$ for some constant $\rho > 0$ . This is because we are interested in a local bound on the generalization error (see Corollary 5). This upper bound does not explicitly depend on the number of parameters, whereas classical bounds typically do [5, 6], which can prevent them from providing sharp bound in overparameterized models.

• Connecting theoretical insights from Bayesian inference to other inference methods: Our results provide theoretical support for interpreting the local RLCT as a measure of efective dimension or complexity. [13] pointed out that the RLCT behaves like an efective dimension in Bayesian inference. Moreover, [9] empirically showed a similar interpretation of the local RLCT (LLC) can hold even for models trained by empirical risk minimization rather than Bayesian inference. Our result provides a theoretical explanation for this interpretation.

We present two technical contributions of our proof to achieve the main results: First, for the Hessian rank bound, the key idea is to combine the splitting lemma with the invariance of the RLCT under difeomorphisms. The splitting lemma separates the quadratic contribution to the Hessian from the higher-order contribution, and combining this separation with the monotonicity (Proposition 10) and invariance (Proposition 12) of the local RLCT under difeomorphisms bounds the rank of the Hessian. Second, for the covering number bound, we consider the covering number after embedding the parameters into a function space through the loss function. In this case, the local covering number is mainly determined by the second-order contribution of the Hessian near the zeros of the loss function, then we can obtain an upper bound on the covering number.

## 1.3 Related Works

There is a broad range of studies that focus on the geometry of the parameter space of learning models, including singularity, spanning from classical works to those developed after the rise of deep learning. Here, we consider (i) classical influential studies on singularity, (ii) studies that investigate the generalization error of deep learning using flatness or sharpness, and (iii) studies that leverage singularity to analyze generalization error in deep learning. The present work is positioned within (iii).

(i) Classical studies on singularities include [10, 21–24]. In [22], which investigates multilayer-perceptron learning dynamics, singularities are shown to have a strong efect on the dynamics. The works [10, 14] established key results on the asymptotic generalization error of singular models under Bayesian inference, where algebraic-geometric singularities of the zero set of the generalization error determine the Bayesian generalization error.

(ii) Pioneering studies that investigate the generalization error of deep learning using flatness or sharpness are [25]. Later, [26] reported that large-batch training tends to converge to sharp minima and generalizes worse, suggesting a close link between flatness and generalization. However, [27] showed that such flatness can be changed arbitrarily by reparameterization, so Hessian-eigenvalue-based measures are not invariant. In contrast, the local real log canonical threshold used here is invariant under any difeomorphic change of variables and can be viewed as a new notion of flatness.

(iii) Representative studies that leverage singularity to analyze generalization error in deep learning include [8, 9, 11, 28]. The work [8] pointed out that singular learning theory has not been efectively incorporated into the theory of deep learning, and aimed to propose potential applications. [9] attempts a more concrete application and argues that the local learning coeficient is an appropriate complexity measure for deep models, focusing on empirical validation. In contrast, this study proves a theoretical inequality relating the Hessian rank and the local real log canonical threshold, and derives a theoretical upper bound. [16] pointed out a connection between the LLC (local RLCT) and the Hessian rank briefly in their settings in the footnote of Appendix D.3. In this work, however, we give a rigorous formulation of this relationship and apply it to clarify its relationship with existing complexity measures. [11] computes the real log canonical threshold for linear neural networks with identity activation. Since [11, 28] focus on Bayesian generalization error, their goals difer from ours; we also contribute by using the real log canonical threshold as a complexity measure.

## 1.4 Notation

Let $k \in \mathbb { N }$ and $\langle \cdot , \cdot \rangle$ is the standard inner product on $\mathbb { R } ^ { k }$ , and $\| \cdot \|$ is the standard norm on $\mathbb { R } ^ { k }$ . For a point $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { k }$ and radius $\delta > 0$ , the open ball $\{ x \in \mathbb { R } ^ { k } : \| x - x _ { 0 } \| < \delta \}$ is denoted by $B _ { \delta } ( x _ { 0 } )$ . We write $B _ { \delta } ^ { k } ( x _ { 0 } )$ when we wish to make the dimension k explicit. Let $\| \cdot \| _ { F }$ denote the Frobenius norm for a real matrix. For symmetric matrices A and $B ,$ we write $A \preceq B$ if $B - A$ is positive semidefinite. $I _ { d }$ denotes the $d \times d$ identity matrix with $d \in \mathbb { N } .$ For $i , j \in \mathbb { N } , I _ { i \times j }$ denotes the $i \times j$ matrix whose (k, k)-entries are 1 for $1 \leq k \leq$ min $\{ i , j \}$ and whose other entries are 0. Similarly, $0 _ { i \times j }$ denotes the $i \times j$ zero matrix.

Let $\mathcal { M } \subset \mathbb { R } ^ { k }$ be a Borel set and let µ be a Borel measure on M. For $p \in [ 1 , \infty ] , \| \cdot \| _ { p }$ denotes the norm of $L ^ { p } ( \mu ) = L ^ { p } ( \mathcal { M } , \mu )$ . Let $P$ be a Borel probability measure on ${ \mathcal { M } } ,$ and let $X , X ^ { \prime } \sim P$ be random vectors. We write Cov $( X , \bar { X } ^ { \prime } ) : = \mathbb { E } \big [ ( X - \mathbb { E } [ X ] ) ( X ^ { \prime } -$ $\mathbb { E } [ X ^ { \prime } ] ) ^ { \top } ]$ and $\operatorname { C o v } ( X ) : = \operatorname { C o v } ( X , X )$

## 2 Preliminaries

## 2.1 Setup: Learning from Data

We introduce the standard learning problem. Let $d \in \mathbb { N }$ denote the number of parameters, and let the parameter space $W \subset \mathbb { R } ^ { d }$ be a bounded open set. Moreover, let $m _ { 1 } , m _ { 2 } \in \mathbb { N }$ denote the input and output dimensions, respectively. Let $\mathcal { X } \subset \mathbb { R } ^ { m _ { 1 } }$ and $\mathcal { V } \subset \mathbb { R } ^ { m _ { 2 } }$ be compact input and output spaces, respectively. Let $\mathcal { Z } = \mathcal { X } \times \mathcal { Y } \subset \mathbb { R } ^ { m }$ be the data space with $m = m _ { 1 } + m _ { 2 }$

Let $\phi : \mathcal { X } \times W \to \mathcal { Y }$ be measurable function, and define the parameterized model class ${ \mathcal { F } } = \{ \phi _ { w } : { \mathcal { X } } \to { \mathcal { Y } } \mid w \in W \}$ , where $\phi _ { w } ( x ) = \phi ( x , w )$ . Let $L : \mathcal { y } \times \mathcal { y }  [ 0 , \infty )$ be a nonnegative measurable function. For $z = ( x , y ) \in \mathcal { Z }$ and $w \in W$ , we define the loss function associated with the model $\phi _ { w } \mathrm { b y } \ell ( z , w ) = L \big ( y , \phi _ { w } ( x ) \big )$

In this problem, we select parameters based on observed data points and evaluate their performance. Let $Z = ( X , Y ) \sim P$ be a random vector, and let $Z _ { 1 } , \ldots , Z _ { n }$ be i.i.d. copies of $Z .$ . Then we denote the dataset by $S = ( Z _ { 1 } , \ldots , Z _ { n } ) $ . The empirical error and the generalization error are defined as

$$
R _ { S } ( w ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \ell ( Z _ { i } , w ) , \qquad R ( w ) = \operatorname { \mathbb { E } } _ { Z \sim P } [ \ell ( Z , w ) ] \qquad ( w \in W ) .
$$

In the typical learning problem, a model parameter w is selected based on this empirical error (for example, the minimizer of the empirical error $R _ { S } ( w ) )$ , and its performance is then evaluated using the generalization error $R ( w ) \ [ 5 , 6 , 2 9 ]$ . Since the generalization error cannot be directly observed from the dataset, it is essential to control a gap between the empirical error and the generalization error [5, 29].

## Assumption and Notation

We introduce basic assumptions in this work.

Assumption 1 (Realizability) Let $\mathcal { G }$ denote the class of all measurable functions from X to $\mathcal { N } .$ . There exists $w _ { 0 } \in W$ such that

$$
R ( w _ { 0 } ) = R ^ { \star } : = \operatorname* { i n f } _ { g \in { \mathcal { G } } } \mathbb { E } \big [ L ( Y , g ( X ) ) \big ] .
$$

Assumption 2 $\ell ( z , w )$ is analytic on $\mathcal { O } _ { \mathcal { Z } } \times W$ for some open set $\mathcal { O } _ { \mathcal { Z } }$ containing $\mathcal { Z } .$

Assumption 1 is often assumed in machine laerning and coincides with the realizability in prior work [30, 31] (see Remark 1). Assumption 2 is also natural as discussed in Appendix C. With this assumption, we can state that the generalization error $R ( w )$ is analytic on W (see Proposition 7 in Section A).

As basic notation, we write $w _ { 0 } \in W$ as a global minimzier of the generalization error $R ( w )$ . We also define a set of global minimizers of the generalization error as

$$
W _ { 0 } = \mathop { \mathrm { a r g } \operatorname* { m i n } } _ { w \in W } R ( w ) = \{ w \in W : R ( w ) = R ^ { \star } \} .
$$

Further, we write $w _ { * } \in W$ as a local minimzier of $R ( w )$

Remark 1 In standard settings, $R ( w ) - R ^ { \star } = 0$ if and only if the KL divergence from the true distribution to the model at w is zero; this holds, for example, for multiclass classification with the log loss and for regression with the squared loss under Gaussian noise [32, Examples 1 and 2]. Hence, Assumption 1 is equivalent to the realizability condition in $\it { 1 3 0 , ~ 3 1 }$ in these settings.

## 2.2 Our Interest: Local Complexity Measures of Learning

The focus of this study is to evaluate the local complexity measures. Specifically, we consider the problem of evaluating the following metrics: (I) a Hessian matrix of the generalization error at local minimizers, and (II) a covering number of models in a neighborhood of the global minimizers.

Below, we define complexity measures of $R ( w )$ at the corresponding points.

## I. Hessian Rank

The first measure of interest is the rank of the Hessian of the generalization error $R ( w )$ . In this work, we focus on its value at local minimizers $w _ { * } \in W$ :

$$
\operatorname { r a n k } { \nabla } ^ { 2 } R ( w _ { * } ) .\tag{1}
$$

We evaluate the Hessian rank for two reasons: (i) it is an intuitively reasonable measure of efective dimension, (ii) it is closely related to classical quantities that have been proposed as efective dimension measures. Indeed, for (i), the Hessian rank is the number of eigenvectors associated with positive eigenvalues, which is the number of directions along which substantial changes can occur in a neighborhood of w [17, 18]. Based on this property, previous studies bounded the Hessian rank of deep linear fully connected networks, and deep linear convolutional networks [17]. Moreover, for (ii), the Hessian rank is closely related to classical efective dimension that have also been studied in overparameterized models. First, it is an upper bound of MacKay’s efective dimension $\begin{array} { r } { [ 1 7 , 1 9 , 2 0 ] ; \gamma ( \varepsilon ) = \sum _ { i = 1 } ^ { d } \frac { \lambda _ { i } } { \lambda _ { i } + \varepsilon } , \varepsilon > 0 } \end{array}$ , where $\lambda _ { i }$ denotes the i-th eigenvalue of the Hessian. Since $\gamma ( \varepsilon ) \leq$ rank $\nabla ^ { 2 } R ( w _ { * } )$ holds, the Hessian rank is an upper bound on MacKay’s efective dimension. Second, this Hessian rank connected to the Fisher information rank at global minimizers, which has been pointed out as a measure of efective dimension [33–35]. That is, when the negative log-likelihood loss is considered, the Fisher information matrix $F ( w _ { 0 } )$ satisfies $F ( w _ { 0 } ) = \nabla ^ { 2 } R ( w _ { 0 } )$ at $w _ { 0 } \in W _ { 0 }$

## II. Local Covering Number

The second measure of interest is a local covering number of a neighborhood of $w _ { 0 } \in$ $W _ { 0 }$ . In preparation, the covering number is defined as follows:

Definition 1 (Covering Number) Let $( \mathbb { D } , \rho )$ be a metric space. A finite set $\{ x _ { 1 } , \dotsc , x _ { N } \} \subset \mathbb { D }$ for some $N \in \mathbb { N }$ is called τ-cover of D with respect to the metric $\rho { \mathrm { ~ i f ~ } } \mathbb { D } \subset \bigcup _ { i = 1 } ^ { N } \{ x \in \mathbb { D } \}$ $\rho ( x , x _ { j } ) \leq \tau \}$ holds. The covering number $N \left( \mathbb { D } , \rho , \tau \right)$ is the minimum cardinality among all τ-covers of D with respect to the metric $\rho .$

Covering numbers are a standard tool in the analysis of the generalization error $[ 5 ,$ 6, 36, 37]. The importance of this quantity lies in its ability to control the gap between the empirical error and the generalization error [36, 37]; see also Proposition 15.

With the notion, our interest is a covering number of a local set of models $\mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) = \{ \ell ( \cdot , w ) : w \in B _ { \delta } ( w _ { 0 } ) \}$ with $\delta > 0$ 0:

$$
N \left( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { \infty } ( P ) , \tau \right) .\tag{2}
$$

Of particular interest is the case in which $\mathbb { D } = \mathcal { F } _ { \ell } ( A )$ and d is the distance induced by the norm $\| \cdot \| _ { \infty }$ . Since the metric is clear from the context, we simply write $L ^ { \infty } ( P )$ instead of $\| \cdot \| _ { \infty }$

The covering number of such local sets is important for analyzing modern machine learning methods, such as deep learning. Since learning algorithms restrict parameters to specific local sets, the covering number of these sets is examined to characterize generalization error [38–40].

## 2.3 Local Real Log Canonical Threshold (Local RLCT)

We introduce the notion of the real log canonial threthold (RLCT) and the local real log canonial threthold (local RLCT) as basic concepts for this study.

## RLCT

We first introduce the RLCT, which is necessary to define the local RLCT.

Definition 2 (Real Log Canonical Threshold; RLCT) Let R be analytic on W, as ensured by Assumption 2. Let $C \subset W$ be a compact set, and $w _ { \mathrm { m i n } } = \arg \operatorname* { m i n } _ { w \in C } R ( w )$ . We define its $\zeta$ function of $R ( w ) - R ( w _ { \mathrm { m i n } } )$ by

$$
\zeta ( s ) = \int _ { C } \left( R ( w ) - R ( w _ { \mathrm { m i n } } ) \right) ^ { s } d w \quad ( \mathrm { R e } ( s ) > 0 , s \in \mathbb { C } ) ,
$$

Then $\zeta$ admits an analytic continuation to a meromorphic function on $\mathbb { C } .$ and all of its poles lie on the negative real axis [41] [42, Corollary 5.2] [14, Theorem 2.4].

![](images/15178a4eeeb98f70d2c0d22d618137a727253d8bddf7730e8e729e2d102e48a6.jpg)  
(a) The definition of RLCT.

![](images/c4aee525b355b11d696dc0e64f1e16f8fe2bf98f994347f55ae2bade3ebed5f5.jpg)  
(b) The image of ζ(s).  
Fig. 1 (a) The definition of RLCT. The RLCT is defined as the absolute value of the largest pole of the ζ function. The $\zeta$ function is initially defined for $s \in \mathbb { C }$ with $\operatorname { R e } ( s ) > 0 .$ . and it can then be extended to the entire complex plane by analytic continuation [41]. Then their poles (Remark 2) arise on the negative real axis [14, 42]. It was found that this mathematical quantity is deeply related to statistical models [14]. (b) The image of $\zeta ( s )$ . At each pole, |ζ(s)| diverges to infinity.

We denote by $\lambda ( R , C )$ the absolute value of the largest pole (see Remark 2) of $\zeta ,$ and call it the real log canonical threshold (RLCT) of R on C. When it is clear from the context, we write $\lambda ( R , C )$ simply as $\lambda \left( \mathrm { F i g . 1 } \right)$ .

Remark 2 (Poles of complex functions) Poles are notions from complex analysis. Let $s \in \mathbb { C }$ and $f ( s )$ be a complex valued function defined on a neighborhood of $s _ { 0 } \in \mathbb { C }$ . We say that s<sub>0</sub> is a pole of f if there exist an open neighborhood U of s<sub>0</sub> and a natural number m $> 0$ such that

$$
f ( s ) = \frac { g ( s ) } { ( s - s _ { 0 } ) ^ { m } } \quad ( s \in U \setminus \{ s _ { 0 } \} ) ,
$$

where a complex function g(s) that is diferentiable on U with $g ( s _ { 0 } ) \neq 0$

The RLCT was first introduced in singular learning theory to analyze a key quantity characterizing singular models [24, 43–45]. In the original formulation, the RLCT $\lambda ( R , W )$ was used to analyze the generalization error of Bayesian inference, where W is the whole (compact) parameter space [43], which suggested that the RLCT behaves as a certain notion of efective complexity.

## Local RLCT

We introduce the local RLCT at local minimizer $w _ { * } \in W$ of the generalization error. Applications of the RLCT to generalization error analysis have primarily focused on Bayesian inference, while its localized version, the local RLCT (LLC), has recently been shown to play an important role also for models trained by empirical risk minimization [8, 9, 15].

Definition 3 (Local RLCT) Let $w _ { * } \in W$ be a local minimizer of the generalization error R. The limit lim ${ \iota } _ { \delta \to 0 } \lambda \left( R , \overline { { B _ { \delta } } } ( w _ { * } ) \right)$ is called the local RLCT at w<sub>∗</sub>, and is denoted by $\lambda ( R , w _ { * } )$ When R is clear from the context, we write it simply as $\lambda ( w _ { * } )$ .

Note that the limit lim $\iota _ { \delta  0 } \lambda ( R , \overline { { B _ { \delta } } } ( w _ { * } ) )$ always exists (Appendix $\mathrm { E } ,$ Lemma 9).

## 3 Main Results

We present main results of this study: we first bound the rank of the Hessian (1), and we then derive an upper bound on the covering number (2).

## 3.1 Hessian Rank Bound

We first show that the local RLCT gives an upper bound on the rank of the Hessian at the point. A underlying idea was previously sketched in [16, Appendix D.3]; here, we formulate the result rigorously.

Main Theorem 1 (The Hessian Rank Bound) Under Assumption 2, for any $w _ { * } \in W$ that is a local minimizer of R, we have

$$
\operatorname { r a n k } \nabla ^ { 2 } R ( w _ { * } ) \leq 2 \lambda ( w _ { * } ) .
$$

This result shows that the local RLCT is an upper bound on one measure of efective dimension at any local minimizer of the generalization error. Since the Hessian rank can be regarded as a measure of efective dimension as discussed in Section 2.2, and a small local RLCT necessarily implies a small Hessian rank.

Furthermore, we can confirm that a small local RLCT is related to better generalization through a reduction in the efective dimension. A more detailed discussion is provided in Theorem 2 and its application, Corollary 5.

## Proof Outline

To prove this theorem, we introduce a fundamental lemma from the singularity theory Its proof is given in Appendix B.1.

Lemma 3 (Splitting lemma) Let R be a nonnegative $C ^ { 3 }$ function on $W _ { i }$ , and $r =$ rank $\nabla ^ { 2 } R ( w _ { * } )$ . Then for any local minimizer $w _ { * } \in W$ there exists a local $C ^ { 1 }$ difeomorphism Φ around w<sub>∗</sub> such that Φ $: \mathbb { R } ^ { \sim } \times \mathbb { R } ^ { d - r }  \mathbb { R } ^ { d } ; ( u , v ) \mapsto \quad$ w with w<sub>∗</sub> corresponding to $( u _ { * } , v _ { * } )$ and

$$
R \circ \Phi ( u , v ) = \| u \| ^ { 2 } + \operatorname { R e s } ( v ) .
$$

Here, Res is a nonnegative $C ^ { 2 }$ function and satisfies

$$
\operatorname { R e s } ( v _ { * } ) = R ( w _ { * } ) , \qquad \nabla \operatorname { R e s } ( v _ { * } ) = 0 , \qquad \nabla ^ { 2 } \operatorname { R e s } ( v _ { * } ) = 0 .
$$

The splitting lemma states that there exists a local change of coordinates that separates the function into a canonical quadratic form and a residual higher-order terms. This allows us to separate the directions along which R can vary substantially near a local minimum and to investigate their relationship with the local RLCT.

Note that, when R is analytic, the coordinate transformation can be chosen analytic, and analytic version is used in the proof of Main Theorem 1.

Proof of Main Theorem 1 For simplicity, we may assume $R ( w _ { * } ) = 0$ without loss of generality. By Lemma 3, there exist neighborhoods $\mathcal { O } _ { u , v } , \mathcal { O } _ { w }$ of the origin and an analytic difeomorphism $\Phi : \mathcal { O } _ { u , v }  \mathcal { O } _ { w }$ such that $w = \Phi ( u , v )$ for $( u , v ) \in \mathcal { O } _ { u , v }$ and

$$
R \circ \Phi ( u , v ) = \| u \| ^ { 2 } + \operatorname { R e s } ( v ) .
$$

By the definition of the local RLCT, we may take $\delta > 0$ small enough so that $B _ { \delta } ( w _ { * } ) \subset \mathcal { O } _ { w }$ Hence, by the invariance of the RLCT under difeomorphisms (Proposition 12), we have

$$
\begin{array} { r l } & { \lambda \left( R , \overline { { B _ { \delta } } } ( w _ { * } ) \right) = \lambda \left( R \circ \Phi , \Phi ^ { - 1 } \left( \overline { { B _ { \delta } } } ( w _ { * } ) \right) \right) } \\ & { \qquad = \lambda \left( \| u \| ^ { 2 } + \mathrm { R e s } ( v ) , \Phi ^ { - 1 } \left( \overline { { B _ { \delta } } } ( w _ { * } ) \right) \right) } \\ & { \qquad \geq \lambda \left( \| u \| ^ { 2 } , \Phi ^ { - 1 } \left( \overline { { B _ { \delta } } } ( w _ { * } ) \right) \right) } \\ & { \qquad = \frac { r } { 2 } . } \end{array}
$$

where the inequality uses Proposition 10 and the fact that $\mathrm { R e s } ( v ) \geq 0$ . The last equality follows from Proposition 14. □

## 3.2 Local Covering Bound

We next state the main result, the bound of the covering number in terms of the local RLCT. Here, we clarify loss functions we consider. We provide some examples after presenting the assumption.

Assumption 4 The loss function ℓ satisfies the following condition: there exists a constant $\alpha > 0$ such that for any w<sub>0</sub> ∈ W<sub>0</sub>

$$
\operatorname { C o v } \big ( \nabla _ { \boldsymbol { w } } \ell ( Z , { \boldsymbol { w } } _ { 0 } ) \big ) \preceq \alpha \nabla ^ { 2 } R ( { \boldsymbol { w } } _ { 0 } ) .
$$

The negative log-likelihood loss is a standard example satisfying this condition, and thus multiclass classification with cross-entropy loss and squared loss regression with Gaussian noise satisfy this assumption [46, Section 4.3.4] [47, Section 5.5]. Since $\operatorname { C o v } ( \nabla _ { w } \ell ( Z , w ) )$ coincides with the Fisher information matrix at every $w _ { 0 } ~ \in ~ W _ { 0 }$ note that $\operatorname { C o v } \left( \nabla _ { w } \ell ( Z , w _ { 0 } ) \right) = \nabla ^ { 2 } R ( w _ { 0 } )$ holds [12, 14]. Thus, this assumption holds with $\alpha = 1$ for the negative log-likelihood loss. Another important example is the squared loss regression with bounded outputs. This setup also satisfies this condition (Appendix D).

We give an upper bound of a covering number of a functional set $\mathcal { F } _ { \ell } \big ( B _ { \delta } ( w _ { 0 } ) \big )$ around the global minima $w _ { 0 } \in W$ . Here, we provide an upper bound for covering numbers with radius $\tau > \rho _ { \delta }$ that exceed a certain threshold $\rho _ { \delta } > 0$ . Although this upper bound is restrictive than the conventional upper bound for covering numbers with positive $\tau > 0$ , this setting is necessary for evaluation using local RLCT.

Main Theorem 2 (The Local Covering Bound) Under Assumption 1, 2, 4, for any $w _ { 0 } \in W _ { 0 }$ there exist constants $L , C > 0$ and $\delta > 0$ such that, with $\rho _ { \delta } = C \delta ^ { 2 }$ , the following holds for every $\tau > \rho _ { \delta } .$

$$
N \left( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { \infty } ( P ) , \tau \right) \leq \left( 1 + \frac { 2 L } { \tau - \rho _ { \delta } } \right) ^ { 2 \lambda ( w _ { 0 } ) } .
$$

This result provides several insights: (i) the small local RLCT reduces the localized complexity of the model around the global minimizer $w _ { 0 }$ , (ii) this upper bound improves the previous upper bound for covering numbers based on the parameter dimension d [5, 6],

## Proof

We provide the complete proof here. Main Theorem 1 makes a key contribution for this proof.

Proof of Main Theorem 2 Let $r =$ rank $\nabla _ { w } ^ { 2 } R ( w _ { 0 } )$ . The case $r = 0$ is trivial, so assume $r > 0$ By the Taylor expansion around $w _ { 0 }$

$$
R ( w ) = \frac { 1 } { 2 } \left. \nabla ^ { 2 } R ( w _ { 0 } ) ( w - w _ { 0 } ) , ( w - w _ { 0 } ) \right. + o ( \| w - w _ { 0 } \| ^ { 2 } ) .
$$

Then for any $\varepsilon > 0$ there exists $\delta > 0$ such that

$$
\left| R ( w ) - \frac { 1 } { 2 } \left. \nabla ^ { 2 } R ( w _ { 0 } ) ( w - w _ { 0 } ) , ( w - w _ { 0 } ) \right. \right| < \frac { \varepsilon } { 4 }
$$

holds on $B _ { \delta } ( w _ { 0 } )$ . This implies $\begin{array} { r } { \frac { 1 } { 2 } \left. \nabla ^ { 2 } R ( w _ { 0 } ) ( w - w _ { 0 } ) , ( w - w _ { 0 } ) \right. < R ( w ) + \frac { \varepsilon } { 4 } } \end{array}$ . Here, let Ψ be a linear transformation that diagonalizes $\nabla ^ { 2 } R ( w _ { 0 } )$ , and let $u \in \mathbb { R } ^ { r }$ denote the directions corresponding to positive eigenvalues, and let $\boldsymbol { h } \in \mathbb { R } ^ { d - r }$ denote the directions corresponding to zero eigenvalues. Suppose that $w _ { 0 }$ corresponds to $( u _ { 0 } , v _ { 0 } )$ under this transformation. Then

$$
\frac { 1 } { 2 } \sum _ { i = 1 } ^ { r } \lambda _ { i } ^ { + } ( w _ { 0 } ) \big ( u _ { i } - u _ { 0 , i } \big ) ^ { 2 } = \frac { 1 } { 2 } \left. \nabla ^ { 2 } R ( w _ { 0 } ) ( w - w _ { 0 } ) , w - w _ { 0 } \right. < R ( w ) + \frac { \varepsilon } { 4 } ,
$$

where $\lambda _ { i } ^ { + } ( w _ { 0 } )$ denotes the i-th positive eigenvalue. Since we may take δ smaller if necessary, we can assume that $\begin{array} { r } { R ( w ) < \frac { \varepsilon } { 4 } } \end{array}$ . Then we have

$$
\frac { 1 } { 2 } \sum _ { i = 1 } ^ { r } \lambda _ { i } ^ { + } ( w _ { 0 } ) ( u _ { i } - u _ { 0 , i } ) ^ { 2 } < \frac { \varepsilon } { 2 } .
$$

This shows that, in the sense of the metric induced by the positive definite matrix determined by the eigendirections corresponding to the nondegenerate eigenvalues of $\nabla ^ { 2 } R ( w _ { 0 } )$ , the radius is less than $\sqrt { \varepsilon }$

Then by Lemma $^ { 8 , }$ there exist $C > 0$ such that

$$
\begin{array} { r } { \big \| \ell ( \cdot , \Psi ( u _ { 1 } , v _ { 1 } ) ) - \ell ( \cdot , \Psi ( u _ { 2 } , v _ { 2 } ) ) \big \| _ { \infty } \leq L \| u _ { 1 } - u _ { 2 } \| + C \delta ^ { 2 } . } \end{array}
$$

Hence, let $B _ { \sqrt { \varepsilon } } ^ { r } ( u _ { 0 } )$ denote the r-dimensional ball centered at $u _ { 0 }$ , where $( u _ { 0 } , v _ { 0 } ) = \Psi ^ { - 1 } ( w _ { 0 } )$ with radius $\sqrt { \varepsilon }$ . Then, letting $\rho _ { \delta } = C \delta ^ { 2 }$ , for any $\tau > \rho _ { \delta }$

$$
\begin{array} { r l r } {  { N ( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { \infty } ( P ) , \tau ) \leq N ( B _ { \sqrt { \varepsilon } } ^ { r } ( u _ { 0 } ) , \| \cdot \| , \frac { \tau - \rho _ { \delta } } { L } ) } } \\ & { } & { \leq ( 1 + \sqrt { \frac { \varepsilon } { \lambda _ { \operatorname* { m i n } } ^ { + } ( w _ { 0 } ) } } \frac { 2 L } { \tau - \rho _ { \delta } } ) ^ { r } , } \end{array}
$$

where $\lambda _ { \operatorname* { m i n } } ^ { + } ( w _ { 0 } )$ denotes the smallest eigenvalue. The second inequality uses the fact that for the distance $d _ { G } ( u , u ^ { \prime } ) = \sqrt { \langle G ( u - u ^ { \prime } ) , u - u ^ { \prime } \rangle }$ induced by a positive definite matrix $G , { \mathrm { i f } }$ $\lambda _ { \mathrm { m i n } }$ denotes the minimum eigenvalue of $G ,$ then $\lambda _ { \operatorname* { m i n } } \langle u - u ^ { \prime } , u - u ^ { \prime } \rangle \leq \langle G ( u - u ^ { \prime } ) , u - u ^ { \prime } \rangle$ holds. That is, for the usual distance $d ( u , u ^ { \prime } ) = \lVert u - u ^ { \prime } \rVert$

$$
d ( u , u ^ { \prime } ) \leq \frac { 1 } { \sqrt { \lambda _ { \operatorname* { m i n } } } } d _ { G } ( u , u ^ { \prime } )
$$

holds, and hence $B _ { \sqrt { \varepsilon } } ^ { r } ( u _ { 0 } )$ is contained in the ball of radius $\sqrt { \varepsilon / \lambda _ { \operatorname* { m i n } } ^ { + } ( w _ { 0 } ) }$ under the usual distance.

Since Main Theorem 1 gives $r \leq 2 \lambda ( w _ { 0 } )$ , the exponent r can be replaced by $2 \lambda ( w _ { 0 } )$ Finally, since ε was arbitrary, setting $\varepsilon = \lambda _ { \mathrm { m i n } } ^ { + } ( w _ { 0 } )$ completes the proof. □

## 4 Application

We discuss the implications of the preceding results for generalization error.

## 4.1 Local Generalization Error Bounds and Its Implications

We derive local generalization bounds and discuss the following implication: the generalization bound can vary depending on the local RLCT. We first establish the following corollary, whose proof is given in Appendix B.3.

Corollary 5 Under Assumption 1, 2, 4, for every $w _ { 0 } \in W _ { 0 }$ there exist $\delta > 0$ and $L , B , C > 0$ such that, for any $\eta > 0$ , the following holds with probability at least $\begin{array} { r } { 1 - \exp \left( - \frac { n \eta ^ { 2 } } { 2 B ^ { 2 } } \right) } \end{array}$

$$
R ( w ) \leq R _ { S } ( w ) + 9 6 \sqrt { \frac { 2 \lambda ( w _ { 0 } ) L B } { n } } + C \delta ^ { 2 } + \eta , \qquad ( w \in B _ { \delta } ( w _ { 0 } ) ) ,
$$

where B is a constant with $\ell ( z , w ) \leq B \ o n \ \mathcal { Z } \times B _ { \delta } ( w _ { 0 } )$

This result shows that the generalization error has the smaller upper bound in neighborhoods of global minimizers with smaller local RLCTs.

Furthermore, since this result holds for every $w \in B _ { \delta } ( w _ { 0 } )$ , the generalization bound applies to a learned parameter whenever it lies in this neighborhood, even if it depends on the data. We present a natural result with a learned parameter wˆ.

Corollary 6 Let $w _ { 0 } \in W _ { 0 }$ and $\delta > 0$ be as in Corollary 5. Suppose Assumption $1 , \ 2 , \ 4$ holds with the same notation as in Corollary 5. Moreover, let $\hat { w } = \hat { w } ( S )$ be a W-valued random variable depending on the dataset S and asuume that wˆ $\in B _ { \delta } ( w _ { 0 } )$ holds. Then, we have the following bound with probability at least $\begin{array} { r } { 1 - \exp \left( - \frac { n \eta ^ { 2 } } { 2 B ^ { 2 } } \right) } \end{array}$

$$
R ( \hat { w } ) \leq R _ { S } ( \hat { w } ) + 9 6 \sqrt { \frac { 2 \lambda ( w _ { 0 } ) L B } { n } } + C \delta ^ { 2 } + \eta .
$$

This result shows that Corollary 5 also applies to a data-dependent parameter wˆ provided that $\hat { w } \in B _ { \delta } ( w _ { 0 } )$ . That is, the generalization error bound is smaller when the learned parameter lies in a neighborhood of a global minimizer with a smaller local RLCT.

This result is consistent with several empirical observations. [9] indicates that, among solutions selected by empirical risk minimization, those with smaller local RLCTs tend to generalize better. Moreover, [48] indicates that larger local Bayesian evidence, which corresponds to a smaller local RLCT, tends to yield better generalization (see Section 3 and Appendix A). Our result provides theoretical support for these empirical observations in the special case where the learned parameter lies near a global minimizer.

## 4.2 Local Generalization Error Analysis for Reduced-Rank Regression

We derive concrete upper bounds on the generalization error for reduced-rank regression. These results show that even among global minimizers, diferent local RLCTs can lead to diferent generalization bounds.

## Problem Setup

We introduce the following reduced-rank regression model. Fix an integer h satisfying $1 \leq h < \operatorname* { m i n } \{ m _ { 1 } , m _ { 2 } \}$ , and the reduced-rank regression model with an input $x \in$ $[ - 1 , 1 ] ^ { m _ { 1 } }$ is given by

$$
\phi ( x , w ) = U V x ,
$$

where $U \in \mathbb { R } ^ { m _ { 2 } \times h } , V \in \mathbb { R } ^ { h \times m _ { 1 } } , w = ( U , V )$ , and the parameter space is $W =$ $( - 2 , 2 ) ^ { m _ { 2 } \times h } \times ( - 2 , 2 ) ^ { h \times m _ { 1 } }$ . The number of parameters is $d = h ( m _ { 1 } + m _ { 2 } )$

We consider a noiseless regression problem with the reduced-rank regression model. Let (X, Y) be an input-output pair, where X takes values in $[ - 1 , 1 ] ^ { m _ { 1 } }$ and $Y$ takes values in a compact subset of $\mathbb { R } ^ { m _ { 2 } }$ , and it follows the data-generating model

$$
Y = T ^ { \star } X , \quad T ^ { \star } = \binom { I _ { r } \ 0 } { 0 \ 0 } \in \mathbb { R } ^ { m _ { 2 } \times m _ { 1 } } ,
$$

with $r \in \mathbb { N }$ . Moreover, assume that $\mathbb { E } [ X X ^ { \top } ]$ is positive definite. The loss function is the squared loss $\ell ( ( X , Y ) , w ) = \| Y - \phi ( X , w ) \| ^ { 2 }$ . Then, the generalization error is given by $R ( w ) = \mathbb { E } [ \| Y - \phi ( X , w ) \| ^ { 2 } ] = \mathbb { E } [ \| T ^ { \star } X - U V X \| ^ { 2 } ]$

## Derivation of the Local RLCT

We investigate local RLCTs for the following three global minimizers $w _ { i } = ( U _ { i } , V _ { i } ) \in$ $\mathbb { R } ^ { m _ { 2 } \times h } \times \mathbb { R } ^ { \bar { h } \times m _ { 1 } } , i = 1 , 2 , 3$ , that is, those that satisfy $U _ { i } V _ { i } = T ^ { \star }$ . While there are more global minizers, we select $w _ { i } , i = 1 , 2 , 3$ with representative structures from among them. Specifically, we consider

$$
\begin{array} { l } { { ( U _ { 1 } , V _ { 1 } ) = \left( \left( \begin{array} { l } { { I _ { h } } } \\ { { 0 } } \end{array} \right) , \left( \begin{array} { l } { { I _ { r } \ 0 } } \\ { { 0 \ 0 } } \end{array} \right) \right) , ( U _ { 2 } , V _ { 2 } ) = \left( \left( \begin{array} { l l } { { I _ { r } \ 0 } } \\ { { 0 \ 0 } } \end{array} \right) , \left( I _ { h } \ 0 \right) \right) , } } \\ { { ( U _ { 3 } , V _ { 3 } ) = \left( \left( \begin{array} { l l } { { I _ { r } \ 0 } } \\ { { 0 \ 0 } } \end{array} \right) , \left( \begin{array} { l l } { { I _ { r } \ 0 } } \\ { { 0 \ 0 } } \end{array} \right) \right) , } } \end{array}
$$

and then we have $R ( w _ { i } ) = 0 ( i = 1 , 2 , 3 )$ . Their local RLCTs are computed as

$$
\begin{array} { c } { { \lambda ( w _ { 1 } ) = \displaystyle \frac { h m _ { 1 } + r ( m _ { 2 } - h ) } { 2 } , \qquad \lambda ( w _ { 2 } ) = \displaystyle \frac { h m _ { 2 } + r ( m _ { 1 } - h ) } { 2 } , } } \\ { { \lambda ( w _ { 3 } ) = \displaystyle \frac { 1 } { 2 } \left\{ r ( m _ { 1 } + m _ { 2 } - r ) + \kappa \right\} , } } \end{array}\tag{3}
$$

where $\begin{array} { r } { \kappa \ = \ \operatorname* { m i n } _ { 0 \leq j \leq h - r , j \in { \mathbb Z } } \left\{ j ( m _ { 2 } - r ) + ( m _ { 1 } - r - j ) ( h - r - j ) \right\} } \end{array}$ . This derivation demonstrates that diferent global minimziers have diferent local RLCTs. The detailed derivation is given in Appendix F.1. In Section 5, we empirically verify that the inequality in the main theorem holds.

## Derivation of the Local Generalization Error

Finally, we derive the bound on the local generalization error of the reduced rank regression problem. By selecting several constants in Lemma 8, Corollary 5 provides that there exist $\delta > 0$ and the following bounds

$$
\begin{array} { r l } & { R ( w ) \leq R _ { S } ( w ) + 9 6 \sqrt { \frac { \left( h m _ { 1 } + r ( m _ { 2 } - h ) \right) L B } { n } } + C \delta ^ { 2 } + \eta \quad \left( w \in B _ { \delta } ( w _ { 1 } ) \right) , } \\ & { R ( w ) \leq R _ { S } ( w ) + 9 6 \sqrt { \frac { \left( h m _ { 2 } + r ( m _ { 1 } - h ) \right) L B } { n } } + C \delta ^ { 2 } + \eta \quad \left( w \in B _ { \delta } ( w _ { 2 } ) \right) , } \\ & { R ( w ) \leq R _ { S } ( w ) + 9 6 \sqrt { \frac { \left( r ( m _ { 1 } + m _ { 2 } - r ) + \kappa \right) L B } { n } } + C \delta ^ { 2 } + \eta \quad \left( w \in B _ { \delta } ( w _ { 3 } ) \right) , } \end{array}
$$

A key insight is that diferent local geometries yield diferent bounds, whereas existing studies give the same bound for every parameter. The complexity term in [49, Corollary 6] is controlled by a parameter-independent quantity, which is independent of local poperties with a learned parameter. In contrast, our bound varies with the local RLCT at each global minimizer, giving tighter values near minimizers with fewer efective degrees of freedom. Thus, we obtain a bound that adapts to the local geometry at each parameter.

## 4.3 Local Generalization Error Analysis for Linear Neural Network

We derive upper bounds on the generalization error for a linear neural network.

## Problem Setup

We consider the following linear neural network model. We define

$$
\phi ( x , w ) = W _ { \mathrm { o u t } } W _ { \mathrm { h i d } } W _ { \mathrm { i n } } x ,
$$

where $W _ { \mathrm { i n } } \in \mathbb { R } ^ { h _ { 1 } \times m _ { 1 } } , W _ { \mathrm { h i d } } \in \mathbb { R } ^ { h _ { 2 } \times h _ { 1 } } , W _ { \mathrm { o u t } } \in \mathbb { R } ^ { m _ { 2 } \times h _ { 2 } } \mathrm { ~ a n d ~ } w = ( W _ { \mathrm { i n } } , W _ { \mathrm { h i d } } , W _ { \mathrm { o u t } } )$ Then the parameter space is $W = ( - 2 , 2 ) ^ { h _ { 1 } \times m _ { 1 } } \times ( - 2 , 2 ) ^ { h _ { 2 } \times h _ { 1 } } \times ( - 2 , 2 ) ^ { m _ { 2 } \times h _ { 2 } }$ . For simplicity, we asuume $m _ { 1 } > h _ { 1 } > h _ { 2 } > m _ { 2 }$

We consider a noiseless regression problem with the realizable true function $Y = 0$ almost surely. This setting is introduced only to simplify the calculation of the local RLCT; essentially the same proof works when X and Y are correlated, as in reducedrank regression, with a more involved calculation. Moreover, assume that $\mathbb { E } [ X X ^ { \top } ]$ is positive definite. The loss function is the squared loss $\ell ( ( X , Y ) , w ) = \| \phi ( \dot { X _ { , } } w ) \| ^ { 2 }$ Then, the generalization error is given by $R ( w ) = \mathbb { E } [ \| \phi ( X , w ) \| ^ { 2 } ]$

## Derivation of the Local RLCT

We investigate local RLCTs for the following three global minimizers $w _ { i } \in \mathbb { R } ^ { h _ { 2 } \times m _ { 1 } } \times$ $\mathbb { R } ^ { h _ { 2 } \times m _ { 1 } } \times \mathbb { R } ^ { m _ { 2 } \times h _ { 2 } } , i = 1 , 2 , 3$ . As in the previous section, while there are more global minizers, we select the parameters $w _ { i } , i = 1 , 2 , 3$ with representative structures from among them. Specifically, let $w _ { 1 } , w _ { 2 } , w _ { 3 }$ be the parameters defined by

$$
\begin{array} { r } { w _ { 1 } = ( I _ { h _ { 1 } \times m _ { 1 } } , I _ { h _ { 2 } \times h _ { 1 } } , 0 _ { m _ { 2 } \times h _ { 2 } } ) , } \\ { w _ { 2 } = ( I _ { h _ { 1 } \times m _ { 1 } } , 0 _ { h _ { 2 } \times h _ { 1 } } , I _ { m _ { 2 } \times h _ { 2 } } ) , } \\ { w _ { 3 } = ( 0 _ { h _ { 1 } \times m _ { 1 } } , I _ { h _ { 2 } \times h _ { 1 } } , I _ { m _ { 2 } \times h _ { 2 } } ) , } \end{array}\tag{4}
$$

and then we have $R ( w _ { i } ) = 0 ( i = 1 , 2 , 3 )$ . Then

$$
\lambda ( w _ { 1 } ) = { \frac { m _ { 2 } h _ { 2 } } { 2 } } , \qquad \lambda ( w _ { 2 } ) = { \frac { m _ { 2 } h _ { 1 } } { 2 } } , \qquad \lambda ( w _ { 3 } ) = { \frac { m _ { 2 } m _ { 1 } } { 2 } } .\tag{5}
$$

The detailed derivation is given in Appendix F.2. In Section 5, we empirically verify that the inequality in the main theorem holds.

## Derivation of the Local Generalization Error

We derive the bound on the local generalization error. By Corollary 5, there exist $\delta > 0$ and such that

$$
\begin{array} { r l } { \displaystyle R ( w ) \leq R _ { S } ( w ) + 9 6 \sqrt { \frac { m _ { 2 } h _ { 2 } L B } { n } } + C \delta ^ { 2 } + \eta } & { { } \left( w \in B _ { \delta } ( w _ { 1 } ) \right) , } \\ { \displaystyle R ( w ) \leq R _ { S } ( w ) + 9 6 \sqrt { \frac { m _ { 2 } h _ { 1 } L B } { n } } + C \delta ^ { 2 } + \eta } & { { } \left( w \in B _ { \delta } ( w _ { 2 } ) \right) , } \\ { \displaystyle R ( w ) \leq R _ { S } ( w ) + 9 6 \sqrt { \frac { m _ { 2 } m _ { 1 } L B } { n } } + C \delta ^ { 2 } + \eta } & { { } \left( w \in B _ { \delta } ( w _ { 3 } ) \right) . } \end{array}
$$

Our bound adapts to the local geometry and is invariant under smooth reparameterizations unlike norm-based bounds. Norm-based bounds [38, 40, 50], a central theory for analyzing the generalization error of neural networks, depend on weight norms; however, it has been pointed out that these norms can change under functionpreserving reparameterizations [27]. In contrast, our bound is invariant under (smooth) reparameterizations. Indeed, since local RLCTs are invariant under difeomorphisms (Proposition 12), the same bound holds if δ is re-chosen appropriately. Because our bound is local, it is not directly comparable to these bounds; nevertheless, it depends only on the intrinsic local geometry.

## 5 Experiment

In this section, we empirically verify Main Theorem 1.

Reduced Rank Regression  
![](images/e26d5bf10e4dde8335b3f05f57dcba40d72c431b9a5d55e5a5e3791b3d2cbfe6.jpg)  
Fig. 2 Hessian Eigenvalues vs. Local RLCT: rank $\nabla ^ { 2 } R ( w _ { i } ) \le 2 \lambda ( w _ { i } ) ~ ( i = 1 , 2 , 3 )$ holds at all displayed parameter values. The red dots indicate the values of the eigenvalues, and the orange vertical line marks $2 \lambda ( w _ { i } )$ . The rank of the Hessian equals the number of its positive eigenvalues at global minimizers. The parameters and their corresponding local RLCTs are given in (3) for reducedrank regression, and in (4) for the linear neural network.

## 5.1 Verification of the Hessian Rank Bound in Simple Models

We verify Main Theorem 1 experimentally for the concrete examples. Specifically, for models for which the local RLCT is known as Section 4.2 and 4.3, we numerically compare the rank of the Hessian with the local RLCT. For the reduced-rank regression setup (Section 4.2), let $r = 1 , h = 2 , m _ { 1 } = 3$ and $m _ { 2 } = 1 0$ in (3), then we have $\begin{array} { r } { \lambda ( w _ { 1 } ) = 7 , \lambda ( w _ { 2 } ) = \frac { 2 1 } { 2 } , } \end{array}$ , and $\lambda ( w _ { 3 } ) = 7$ . For the linear neural network setup (Section 4.3), let $m _ { 1 } = 5 , h _ { 1 } = 3 , h _ { 2 } = 2$ and $m _ { 2 } = 1$ in (5), then $\begin{array} { r } { \lambda ( w _ { 1 } ) = \frac 1 2 , \lambda ( w _ { 2 } ) = \frac 3 2 } \end{array}$ and $\begin{array} { r } { \lambda ( w _ { 3 } ) = \frac { 5 } { 2 } } \end{array}$ . In both experiments, we assume $\mathbb { E } [ X X ^ { \top } ] = I _ { m _ { 1 } }$ , which satisfies the positive definiteness assumption on $\mathbb { E } [ X X ^ { \top } ]$ in Sections 4.2 and 4.3. Then, the generalization error is given by $R ( w ) = \mathbf { \bar { \| } } U V - T ^ { \star } \| _ { F } ^ { 2 }$ for reduced-rank regression and $R ( w ) = \| W _ { \mathrm { o u t } } W _ { \mathrm { h i d } } \bar { W } _ { \mathrm { i n } } \| _ { F } ^ { 2 }$ for the linear neural network. All eigenvalue computations were performed in double precision (float64). Note that no samples need in this setting.

Fig. 2 shows that the rank of the Hessian is well bounded by $2 \lambda ( w _ { i } ) ~ ( i = 1 , 2 , 3 )$ In reduced rank regression, the local RLCT at $w _ { 3 }$ difers from the rank of the Hessian. This is because the local RLCT also captures contributions of order higher than two [16], which the Hessian cannot detect.

## 5.2 More General Networks

We conduct experiments for cases in which the local RLCT is dificult to compute theoretically. The local RLCT can be estimated by the method proposed in [9] whose code is available from the GitHub repository given in the paper.

As a basic setup, we consider a GELU or ReLU neural network $\phi _ { w } : \mathbb { R } ^ { 3 }  \mathbb { R }$ . The number of hidden layers and width indicated in the graph. We set the true function to $Y = 0$ , which allows us to identify parameter that realize it.

To study the local RLCT, we select the global minimizers as follows. We construct five parameters $w _ { i } , i = 1 , \ldots 5$ , each representing the zero function. Roughly speaking, we set the outputs of all units in the last layer to zero. Specifically, each hidden unit has the form $w _ { \mathrm { o u t } } \sigma ( w _ { \mathrm { i n } } ^ { \top } x )$ , where σ is an activation function with $\sigma ( 0 ) = 0 ;$ x is the input or the output of the previous layer, $w _ { \mathrm { i n } }$ is a vector of the same dimension as x, and $w _ { \mathrm { o u t } }$ is a scalar. If $w _ { \mathrm { o u t } } = 0 .$ , or $w _ { \mathrm { i n } } = 0 .$ , then ${ w _ { \mathrm { o u t } } } \sigma ( w _ { \mathrm { i n } } ^ { \top } x ) = 0$ . Assume that the last hidden layer has width at least 5. For $i = 1 , \ldots , 5$ , the parameter $w _ { i }$ is defined so that the first $j - 1$ units of the last hidden layer have $w _ { \mathrm { i n } } = 0$ and the remaining units have $w _ { \mathrm { o u t } } = 0$ . All remaining parameters are assumed to be drawn randomly from a Gaussian distribution. Then, all of these parameters represent the zero function.

We need to estimate the the local RLCTs, since they are not theoretically clarified. To estimate the local RLCT, we use the estimation method based on [9]. We draw $N = 1 0 2 4$ independent inputs $X _ { k } \sim N ( 0 , I _ { 3 } )$ and use $\begin{array} { r } { R _ { N } ( w ) = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \ell ( X _ { k } , w ) } \end{array}$ as a Monte Carlo approximation of $R ( w )$ . Note that $R _ { N }$ is used for estimating the generalization error, not for training. At each $w _ { i }$ , we compute the eigenvalues of $\nabla ^ { 2 } R _ { N } ( w _ { i } )$ and estimate the local RLCT using the WBIC-type estimator [9]

$$
\widehat { \lambda } ( w _ { i } ) = N \beta \left\{ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } R _ { N } ( w _ { i } ^ { ( t ) } ) - R _ { N } ( w _ { i } ) \right\} , \qquad \beta = \frac { 1 } { \log N } ,
$$

where $w _ { i } ^ { ( t ) }$ denotes the t-th sample retained after burn-in from the SGLD chain initialized at $w _ { i }$ , and $T = 1 7 5 0$ is the number of retained samples. We use 2500 steps with step size $3 \times 1 0 ^ { - 4 }$ , Gaussian prior scale $\gamma = 1$ , and batch size N, discarding the first 750 steps.

We plot the Hessian eigenvalues in descending order and indicate $2 \widehat { \lambda } ( w _ { i } )$ on the eigenvalue-index axis. This comparison examines how the Hessian eigenvalues and the estimated local RLCT characterize neighborhoods of diferent parameters representing the same function.

## 5.2.1 GELU Network

We conduct experiments using GELU networks (Fig. 3). Despite some fluctuations, the results suggest that the local RLCT roughly upper-bounds the number of large eigenvalues.

The advantage of this method is that it can directly estimate an upper bound on the rank of the Hessian. As shown in the Fig. 3, it is dificult to determine the exact rank of the Hessian because its eigenvalues decay gradually. In fact, in numerical experiments, the Hessian often has many positive eigenvalues that are very close to zero [51, 52]. On the other hand, since $2 \lambda ( w _ { i } ) ~ ( i = 1 , \ldots , 5 )$ is guaranteed to be an upper bound on the rank of the Hessian, it can provide a criterion for deciding which eigenvalues may be treated as zero and ignored.

GELU Network (Number of hidden layers: 1, Hidden layer widths: 5)  
![](images/7446b4d89ab97aa434ae306b2c993bf7194600c8bb58e90e0d31d5e3af6615f2.jpg)

GELU Network (Number of hidden layers: 1, Hidden layer widths: 10)  
![](images/bf9333360c5c313e8fcaa251face13c4b98648b1080cec165f4bd7a443f0c478.jpg)

GELU Network (Number of hidden layers: 5, Hidden layer widths: (3,4,5,6,7))  
![](images/00fff06f921ef41ffb4014931c53464b614dc1370a43f73b1a932811ad8f10b9.jpg)  
Fig. 3 Hessian Eigenvalues vs. Local RLCT (GELU Network): This plot shows the eigenvalues of the Hessian and the estimated local RLCTs λ<sup>ˆ</sup><sub>SGLD</sub>(w<sub>i</sub>) (orange lines) at five parameters w<sub>i</sub> $( i = 1 , \ldots , 5 )$ , which are chosen from a GELU network so that the output function is 0. The number of dominant eigenvalues is roughly bounded above by $2 \lambda ( w _ { i } ) ( i = 1 , \dots , 5 )$

## 5.2.2 ReLU Network

We investigate what happens when the assumption 2 is not satisfied. The loss function must be analytic for Main Theorem 1, however, one of the typical activation functions ReLU is not analytic, since it is not smooth at the origin. The experimental results do not clearly support the claim (Fig. 4).

## 6 Conclusion

In this paper, we investigated the relationship between the local RLCT and the generalization error through local covering numbers. We focused on the RLCT and its localized counterpart, the local RLCT, and showed that the latter can serve as a measure of local complexity even for models trained by empirical risk minimization, extending the role of RLCT beyond the characterization of Bayesian generalization behavior.

ReLU Network (Number of hidden layers: 1, Hidden layer widths: 5)  
![](images/c060b456a0cd690d62502a4702e06104ab859067bb0eb2ae47a84110cf949177.jpg)

ReLU Network (Number of hidden layers: 1, Hidden layer widths: 10)  
![](images/667a5b81c6119668af960a9f03357e750b5a037d7b3e130d85e812e74b509887.jpg)

ReLU Network (Number of hidden layers: 5, Hidden layer widths: (3,4,5,6,7))  
![](images/ad4db339fc68f18d68d5ecccc8f34c2f89786476b53f781558311266d9b92e8e.jpg)  
Fig. 4 Hessian Eigenvalues vs. Local RLCT (ReLU Network): This plot shows the eigenvalues of the Hessian and the estimated local RLCTs $\hat { \lambda } _ { \mathrm { S G L D } } ( w _ { i } )$ (orange lines) at five parameters w<sub>i</sub> $( i = 1 , \ldots , 5 )$ , which are chosen from a ReLU network so that the output function is zero. Since ReLU is not analytic, the experimental results do not clearly support the claim of the Main Theorem.

The main contribution of this work is twofold. (i) we proved that the local RLCT provides an upper bound on half the Hessian rank at local minimizers of the generalization error. (ii) we derived a local covering number bound whose exponent is governed by the local RLCT. As an application of this result, we obtained a local uniform generalization gap bound. These results suggest that the local RLCT partially captures the efective dimension or intrinsic model complexity.

There remain several directions for future work. One important direction is to extend the analysis beyond neighborhoods of global minimizers of the generalization error and to develop analogous complexity bounds at any local minimizers. Another direction is to treat non-analytic models, especially ReLU networks, which are typical model in deep learning.

Acknowledgements. Atsuyoshi Muta was supported by RIKEN Junior Research Associate Program. Masaaki Imaizumi was supported by JSPS KAKENHI (Grant

No. 24K02904), JST CREST (Grant No. JPMJCR21D2), JST FOREST (Grant No.   
JPMJFR216I), and JST BOOST (Grant No. JPMJBY24A9).

## Additional Notation in the Appendix

The notation and setup introduced in Sections 1.4 and 2.1 are also used throughout this appendix. The empirical measure $P _ { n }$ is defined by $\begin{array} { r } { P _ { n } ( A ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } _ { \{ Z _ { i } \in A \} } } \end{array}$ for all Borel measurable subset $A \subset { \mathcal { Z } }$ and $\mathbf { 1 } _ { \{ Z _ { i } \in A \} }$ equals 1 if $Z _ { i } \in A$ and 0 otherwise.

For $\boldsymbol { x } \in \mathbb { R } ^ { k }$ and $S \subset \mathbb { R } ^ { k }$ , we define the distance between the point x and the set $s$ by $d ( x , S ) = \operatorname* { i n f } _ { y \in S } \| x - y \|$ . For a diferentiable function $g : \mathbb { R } ^ { k _ { 1 } } \to \mathbb { R } ^ { k _ { 2 } } ( k _ { 1 } , k _ { 2 } \in \mathbb { N } )$ we denote by $\bar { J _ { q } } ( x ) \in \mathbb { R } ^ { k _ { 1 } \times k _ { 2 } }$ its Jacobian matrix at $x ,$ whose $( i , j ) \ – \mathrm { t h }$ entry is given by $\begin{array} { r } { ( J _ { g } ( x ) ) _ { i j } = \frac { \partial g _ { i } ( x ) } { \partial x _ { i } } } \end{array}$

Let $\mathbb { N } _ { 0 } : = \{ 0 \} \cup \mathbb { N } .$ . For $r \in \mathbb { N } .$ , an element $h = ( h _ { 1 } , \ldots , h _ { r } ) \in \mathbb { N } _ { 0 } ^ { r }$ is called a multi-index of length r. For $x = ( x _ { 1 } , \ldots , x _ { r } ) \in \mathbb { R } ^ { r }$ , we define $x ^ { \hat { h } } = x _ { 1 } ^ { \hat { h _ { 1 } } } \cdot \cdot \cdot x _ { r } ^ { h _ { r } }$ . By a slight abuse of notation, we use the same notation for the gradient operator $\nabla =$ $( \partial / \partial x _ { 1 } , \ldots , \partial / \partial x _ { r } )$ , and set $\nabla ^ { h } = ( \partial / \partial x _ { 1 } ) ^ { h _ { 1 } } \cdot \cdot \cdot ( \partial / \partial x _ { r } ) ^ { h _ { r } }$ . When it is necessary to specify the variable of diferentiation, we write $\nabla _ { x } ;$ otherwise, ∇ denotes diferentiation with respect to all variables. We define the order of $h \in \mathbb { N } _ { 0 } ^ { r }$ by $\begin{array} { r } { \| h \| _ { 1 } : = \sum _ { j = 1 } ^ { r } | h _ { j } | = } \end{array}$ $h _ { 1 } + \cdots + h _ { r }$ , i.e. the $L ^ { 1 } { \mathrm { - n o r m } }$ of $h : \{ 1 , \ldots , r \} \to \mathbb { N } _ { 0 }$ for the counting measure.

## Appendix A Analyticity of Generalization Error

With Assumption 2, we obtain the following statement.

Proposition 7 Under Assumption 2, $R ( w )$ is analytic on $W$

Proof We show that $R ( w )$ is analytic. Let $K = m + d .$ By Assumption 2, for any $p = ( \bar { z } , \bar { w } ) \in$ $\mathcal { O } _ { \mathcal { Z } } \times W$ , there exists $r > 0$ such that ℓ admits the following power series expansion on $B _ { r } ^ { K } ( p ) { \mathrm { : } }$

$$
\ell ( z , w ) = \sum _ { \alpha \in \mathbb { N } _ { 0 } ^ { K } } \frac { \nabla ^ { \alpha } \ell ( \bar { z } , \bar { w } ) } { \alpha ! } h ^ { \alpha } \qquad ( h \in B _ { r } ^ { K } ( p ) ) .
$$

Here and below, $h = ( z - \bar { z } , w - \bar { w } )$ . We write a multi-index ${ \boldsymbol { \alpha } } \in  { \mathbb { N } } _ { 0 } ^ { K }$ as $\alpha = ( \gamma , \beta )$ , where $\gamma \in \mathbb { N } _ { 0 } ^ { m }$ corresponds to the z-variables and ${ \boldsymbol { \beta } } \in  { \mathbb { N } } _ { 0 } ^ { d }$ corresponds to the w-variables.

Fix $\bar { w } \in W$ . Since ${ \mathcal Z } \times \{ \bar { w } \}$ is compact, we may cover ${ \mathcal Z } \times \{ \bar { w } \}$ by finitely many balls on which the above power series expansions converge absolutely and uniformly on smaller balls. Hence, there exist $z _ { 1 } , \dotsc , z _ { N } \in \mathcal { Z }$ and $\rho > 0$ such that, for every $( z , w ) \in \mathcal { Z } \times B _ { \rho } ^ { d } ( \bar { w } )$ there exists $i \in \{ 1 , \ldots , N \}$ for which z belongs to the corresponding neighborhood of $z _ { i }$ , and the associated power series expansion converges absolutely and uniformly.

$$
\begin{array} { r l } & { \ell ( z , w ) = \displaystyle \sum _ { \gamma \in \mathbb { N } _ { 0 } ^ { m } } \sum _ { \beta \in \mathbb { N } _ { 0 } ^ { d } } \frac { \nabla _ { z } ^ { \gamma } \nabla _ { w } ^ { \beta } \ell ( z _ { i } , \bar { w } ) } { \gamma ! \beta ! } ( z - z _ { i } ) ^ { \gamma } ( w - \bar { w } ) ^ { \beta } , } \\ & { \quad \quad \quad = \displaystyle \sum _ { \beta \in \mathbb { N } _ { 0 } ^ { d } } \frac { 1 } { \beta ! } \left( \displaystyle \sum _ { \gamma \in \mathbb { N } _ { 0 } ^ { m } } \frac { \nabla _ { z } ^ { \gamma } \nabla _ { w } ^ { \beta } \ell ( z _ { i } , \bar { w } ) } { \gamma ! } ( z - z _ { i } ) ^ { \gamma } \right) ( w - \bar { w } ) ^ { \beta } } \end{array}
$$

$$
\quad = \sum _ { \beta \in \mathbb { N } _ { 0 } ^ { d } } { \frac { \nabla _ { w } ^ { \beta } \ell ( z , { \bar { w } } ) } { \beta ! } } ( w - { \bar { w } } ) ^ { \beta } .
$$

Hence, for every w $\in B _ { \rho } ^ { d } ( \bar { w } )$ , we may interchange the summation and the integral to get

$$
\begin{array} { r l } {  { R ( w ) = \int _ { \mathcal { Z } } \ell ( z , w ) d P ( z ) } } \\ & { = \int _ { \mathcal { Z } } \sum _ { \hat { g } \in \mathbb { R } _ { 0 } ^ { d } } \frac { \nabla \cdot w } { \beta ! } ( w - \bar { w } ) ^ { \beta } d P ( z ) } \\ & { = \int _ { \mathcal { Z } } \displaystyle \sum _ { \underline { { \xi } } = 0 } \sum _ { | 1 | \hat { g } | _ { 1 } = 1 } \frac { \nabla \cdot \hat { g } _ { w } ^ { \beta } ( \bar { \xi } , \bar { w } ) } { \beta ! } ( w - \bar { w } ) ^ { \beta } d P ( z ) } \\ & { = \displaystyle \operatorname* { l i m } _ { j \to \infty } \int _ { \mathbb { Z } } \sum _ { \underline { { \xi } } = 0 } ^ { J } \sum _ { | 1 | \hat { g } | _ { 1 } = 1 } \frac { \nabla \cdot \hat { g } _ { w } ^ { \beta } ( \bar { \xi } , w ) } { \beta ! } ( w - \bar { w } ) ^ { \beta } d P ( z ) } \\ & { = \displaystyle \sum _ { \hat { g } \in \mathbb { R } _ { 0 } ^ { d } } ( \int _ { \mathbb { Z } } \frac { \nabla \cdot \hat { g } _ { w } ^ { \beta } ( \bar { \xi } , \bar { w } ) } { \beta ! } d P ( z ) ) ( w - w ) ^ { \beta } . } \end{array}
$$

The exchange of the limit and the integral is justified by the absolute uniform convergence of the power series and Lebesgue’s dominated convergence theorem. The right-hand side is a power series in $w - \bar { w }$ with positive radius of convergence. Therefore R is real analytic in a neighborhood of ¯w. Since $\bar { w } \in W$ was arbitrary, $R$ is real analytic on $W$ . This proves the proposition. □

## Appendix B Full Proof

## B.1 Proof of Lemma 3

Proof Since $R ( w ) - R ( w _ { * } ) \geq 0$ in an appropriate neighborhood of w<sub>∗</sub>, we have $\nabla R ( w _ { * } ) = 0$ and $\nabla ^ { 2 } R ( w _ { * } )$ is positive semidefinite. Since rank $\nabla ^ { 2 } R ( w _ { * } ) = r$ , after a linear change of coordinates we may write $\boldsymbol { w } = ( u , v ) \in \mathbb { R } ^ { r } \times \mathbb { R } ^ { d - r }$ , so that

$$
\nabla ^ { 2 } R ( u _ { * } , v _ { * } ) = \binom { \Lambda _ { + } \ 0 } { 0 \ 0 } ,
$$

where $\Lambda _ { + }$ is a symmetric positive definite $r \times r$ matrix. Since

$$
\nabla _ { u } R ( u _ { * } , v _ { * } ) = 0 , \qquad \nabla _ { u } ^ { 2 } R ( u _ { * } , v _ { * } ) = \Lambda _ { + } ,
$$

and $\Lambda _ { + }$ is invertible, the implicit function theorem implies that there exists a $C ^ { 2 }$ map $\sigma ( v )$ defined around $v _ { * } \in \mathbb { R } ^ { d - r }$ such that $\sigma ( v _ { * } ) = u _ { * }$ and

$$
\nabla _ { u } R ( \sigma ( v ) , v ) = 0 .
$$

Define $\widetilde { R } ( u , v ) = R ( u + \sigma ( v ) , v )$ . Then $\nabla _ { \boldsymbol { u } } \widetilde { R } ( 0 , \boldsymbol { v } ) = \nabla _ { \boldsymbol { u } } R ( \sigma ( \boldsymbol { v } ) , \boldsymbol { v } ) = 0$ . Therefore, by Taylor’s formula in the u-variables around $u = 0$

$$
\widetilde { R } ( u , v ) = \widetilde { R } ( 0 , v ) + \left. B ( u , v ) u , u \right. ,
$$

where

$$
B ( u , v ) : = \int _ { 0 } ^ { 1 } ( 1 - t ) \nabla _ { u } ^ { 2 } \widetilde { R } ( t u , v ) d t .
$$

The matrix $B ( u , v )$ is symmetric and continuous, and $B ( 0 , v _ { 0 } ) ~ = ~ \frac { 1 } { 2 } \Lambda _ { + }$ . Hence, $B ( u , v )$ remains symmetric positive definite in a suficiently small neighborhood of the $( 0 , v _ { * } )$ . Let $C ( u , v ) = B ( u , v ) ^ { 1 / 2 }$ , and ${ \bar { u } } = C ( u , v ) u$ , then we have

$$
\widetilde { R } ( u , v ) = \widetilde { R } ( 0 , v ) + \| \bar { u } \| ^ { 2 }
$$

around $( 0 , v _ { * } )$ . Define $\operatorname { R e s } ( v ) = \widetilde { R } ( 0 , v ) = R ( \sigma ( v ) , v )$ . Renaming ¯u as u, we get

$$
R \circ \Phi ( u , v ) = \left\| u \right\| ^ { 2 } + \mathrm { R e s } ( v ) ,
$$

for some $C ^ { 1 }$ difeomorphism Φ around w<sub>∗</sub>.

It remains to check $\operatorname { R e s } ( v _ { * } ) = R ( w _ { * } ) , \nabla \mathrm { R e s } ( v _ { * } ) = 0$ and $\nabla ^ { 2 } \mathrm { R e s } ( v _ { * } ) = 0$ . Since σ $\boldsymbol { \cdot } ( v _ { * } ) =$ $u _ { * } .$ , we have Res $( v _ { * } ) = R ( u _ { * } , v _ { * } ) = R ( w _ { * } )$ . Moreover, diferentiating $\nabla _ { u } R ( \sigma ( v ) , v ) = 0$ at v<sub>0</sub> gives

$$
\begin{array} { r } { \Lambda _ { + } \nabla _ { v } \sigma ( v _ { * } ) + \nabla _ { v } \nabla _ { u } R ( u _ { * } , v _ { * } ) = 0 . } \end{array}
$$

Since $\nabla _ { v } \nabla _ { u } R ( u _ { * } , v _ { * } ) \ = \ 0$ and $\Lambda _ { + }$ is invertible, $\nabla _ { v } \sigma ( v _ { * } ) ~ = ~ 0$ . Therefore, noting that $\nabla R ( w _ { * } ) = 0$ implies $\nabla _ { u } R ( u _ { * } , v _ { * } ) = 0$ and $\nabla _ { v } R ( u _ { * } , v _ { * } ) = 0$

$$
\nabla \mathrm { R e s } ( v _ { * } ) = \nabla _ { u } R ( u _ { * } , v _ { * } ) \nabla _ { v } \sigma ( v _ { * } ) + \nabla _ { v } R ( u _ { * } , v _ { * } ) = 0 ,
$$

$$
\nabla ^ { 2 } \mathrm { R e s } ( \boldsymbol { v } _ { * } ) = \nabla _ { \boldsymbol { v } } \sigma ( \boldsymbol { v } _ { * } ) ^ { \top } \nabla _ { \boldsymbol { u } } ^ { 2 } R ( \boldsymbol { u } _ { * } , \boldsymbol { v } _ { * } ) \nabla _ { \boldsymbol { v } } \sigma ( \boldsymbol { v } _ { * } ) + \nabla _ { \boldsymbol { v } } ^ { 2 } R ( \boldsymbol { u } _ { * } , \boldsymbol { v } _ { * } ) = 0 ,
$$

Finally, since $\operatorname { R e s } ( v ) = R ( \sigma ( v ) , v )$ and $R \ge R ( w _ { * } )$ near w<sub>0</sub>, we have $\operatorname { R e s } ( v ) \geq R ( w _ { * } )$ This completes the proof. □

## B.2 Statement and Proof of Lemma 8

Before stating the lemma, we introduce some notations. Fix an arbitrary $w _ { 0 } \in W _ { 0 }$ and let r = rank $\nabla ^ { 2 } R ( w _ { 0 } )$ . There exist matrices $Q _ { + } \in \mathbb { R } ^ { d \times r }$ and $Q _ { 0 } \in \mathbb { R } ^ { \dot { d } \times ( d - r ) }$ such that the matrix $Q \ = \ ( Q _ { + } \ Q _ { 0 } )$ diagonalizes $\nabla ^ { 2 } R ( w _ { 0 } )$ . That is, Q is an orthogonal matrix, and

$$
\begin{array} { r } { Q ^ { \top } \nabla ^ { 2 } R ( w _ { 0 } ) Q = \left( \begin{array} { c c } { \Lambda _ { + } } & { 0 } \\ { 0 } & { 0 } \end{array} \right) , } \end{array}
$$

where $\Lambda _ { + } = \mathrm { d i a g } \big ( \lambda _ { 1 } ^ { + } ( w _ { 0 } ) , \dots , \lambda _ { r } ^ { + } ( w _ { 0 } ) \big )$ is the diagonal matrix whose diagonal entries are the positive eigenvalues of $\nabla ^ { 2 } R ( w _ { 0 } )$

For $u \in \mathbb { R } ^ { r }$ and $v \in \mathbb { R } ^ { d - r }$ , define the afine orthogonal coordinate map

$$
\Psi ( u , v ) = w _ { 0 } + Q _ { + } u + Q _ { 0 } v .\tag{B1}
$$

Then $\Psi ( 0 , 0 ) = w _ { 0 }$

Lemma 8 (Loss Diference Bound) For any $w _ { 0 } \in W _ { 0 }$ , let Ψ denote the coordinate transformation defined in (B1). Then there exist constants $L , C > 0$ and $\delta > 0$ such that $f o r$ any $( u _ { 1 } , v _ { 1 } ) , ( u _ { 2 } , v _ { 2 } ) \in B _ { \delta } ( ( 0 , 0 ) )$

$$
\begin{array} { r } { \big \| \ell ( \cdot , \Psi ( u _ { 1 } , v _ { 1 } ) ) - \ell ( \cdot , \Psi ( u _ { 2 } , v _ { 2 } ) ) \big \| _ { \infty } \leq L \| u _ { 1 } - u _ { 2 } \| + C \delta \| v _ { 1 } - v _ { 2 } \| . } \end{array}
$$

In particular, noting that $\lVert v _ { 1 } - v _ { 2 } \rVert \leq 2 \delta ,$

$$
\left\| \ell ( \cdot , \Psi ( u _ { 1 } , v _ { 1 } ) ) - \ell ( \cdot , \Psi ( u _ { 2 } , v _ { 2 } ) ) \right\| _ { \infty } \leq L \| u _ { 1 } - u _ { 2 } \| + 2 C \delta ^ { 2 } .
$$

Proof The main idea of the proof is to use the coordinate transformation Ψ to obtain direction wise bounds. By the fundamental theorem of calculus, we have

$$
\begin{array} { l } { \ell ( \cdot , \Psi ( u _ { 1 } , v _ { 1 } ) ) - \ell ( \cdot , \Psi ( u _ { 2 } , v _ { 2 } ) ) = \displaystyle \int _ { 0 } ^ { 1 } \frac { d } { d t } \ell ( z , \Psi ( u _ { t } , v _ { t } ) ) d t } \\ { \displaystyle \quad = \int _ { 0 } ^ { 1 } \left\{ \langle \nabla _ { u } \ell ( z , \Psi ( u _ { t } , v _ { t } ) ) , u _ { 1 } - u _ { 2 } \rangle + \langle \nabla _ { v } \ell ( z , \Psi ( u _ { t } , v _ { t } ) ) , v _ { 1 } - v _ { 2 } \rangle \right\} d t , } \end{array}
$$

where $( u _ { t } , v _ { t } ) = t ( u _ { 1 } , v _ { 1 } ) + ( 1 - t ) ( u _ { 2 } , v _ { 2 } ) \ ( 0 \leq t \leq 1 )$ . Then we show that there exist suitable constants $L , C > 0$ and $\delta > 0$ such that the following bound holds:

$$
\begin{array} { r l } & { \displaystyle \| \ell ( \cdot , \Psi ( u _ { 1 } , v _ { 1 } ) ) - \ell ( \cdot , \Psi ( u _ { 2 } , v _ { 2 } ) ) \| _ { \infty } } \\ & { \leq \int _ { 0 } ^ { 1 } \| \nabla _ { u } \ell ( \cdot , \Psi ( u _ { t } , v _ { t } ) ) \| _ { \infty } \| u _ { 1 } - u _ { 2 } \| d t + \int _ { 0 } ^ { 1 } \| \nabla _ { v } \ell ( \cdot , \Psi ( u _ { t } , v _ { t } ) ) \| _ { \infty } \| v _ { 1 } - v _ { 2 } \| d t } \\ & { \leq L \| u _ { 1 } - u _ { 2 } \| + C \delta \| v _ { 1 } - v _ { 2 } \| . } \end{array}
$$

To this end, it remains to prove the following two inequalities.

$$
( \mathrm { i } ) \| \nabla _ { u } \ell ( \cdot , \Psi ( u _ { t } , v _ { t } ) ) \| _ { \infty } \leq L ( 0 \leq t \leq 1 ) ,
$$

(ii) $\begin{array} { r } { \| \nabla _ { v } \ell ( \cdot , \Psi ( u _ { t } , v _ { t } ) ) \| _ { \infty } \leq C \delta ( 0 \leq t \leq 1 ) . } \end{array}$

For (i), noting that $\nabla _ { u } \ell ( z , \Psi ( u , v ) ) \ = \ Q _ { + } ^ { \top } \nabla _ { w } \ell ( z , w )$ , we bound $\| Q _ { + } ^ { \top } \nabla _ { w } \ell ( z , w ) \|$ . By Taylor’s theorem, there exists $\delta _ { 1 } > 0$ such that for all $w \in B _ { \delta _ { 1 } } ( w _ { 0 } )$

$$
\begin{array} { r l } & { \displaystyle \left\| Q _ { + } ^ { \top } \nabla _ { w } \ell ( z , w ) \right\| \leq \left\| Q _ { + } ^ { \top } \nabla _ { w } \ell ( z , w _ { 0 } ) \right\| + \displaystyle \int _ { 0 } ^ { 1 } \left\| Q _ { + } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( z , w _ { 0 } + t ( w - w _ { 0 } ) ) ( w - w _ { 0 } ) \right\| d t , } \\ & { \qquad \leq \left\| Q _ { + } ^ { \top } \nabla _ { w } \ell ( z , w _ { 0 } ) \right\| + \displaystyle \operatorname* { s u p } _ { w \in B _ { \delta _ { 1 } } ( w _ { 0 } ) } \left\| Q _ { + } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( z , w ) \right\| \delta _ { 1 } } \\ & { \qquad \leq \left\| Q _ { + } ^ { \top } \nabla _ { w } \ell ( z , w _ { 0 } ) \right\| + 1 . } \end{array}
$$

Thus we define $L = \left. Q _ { + } ^ { \top } \nabla w \ell ( \cdot , w _ { 0 } ) \right. _ { \infty } + 1$ . This yields the desired conclusion.

For (ii), noting that $\nabla _ { \boldsymbol { v } } \ell ( \boldsymbol { z } , \Psi ( \boldsymbol { u } , \boldsymbol { v } ) ) \ = \ Q _ { 0 } ^ { \top } \nabla _ { \boldsymbol { w } } \ell ( \boldsymbol { z } , \boldsymbol { w } )$ , we bound $\| Q _ { 0 } ^ { \top } \nabla _ { w } \ell ( z , w ) \|$ . By Taylor’s theorem, we have

$$
\begin{array} { r l } & { \displaystyle Q _ { 0 } ^ { \top } \nabla _ { w } \ell ( z , w ) = Q _ { 0 } ^ { \top } \nabla _ { w } \ell ( z , w _ { 0 } ) + Q _ { 0 } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( z , w _ { 0 } ) ( w - w _ { 0 } ) } \\ & { \qquad + Q _ { 0 } ^ { \top } \displaystyle \sum _ { | \beta | = 2 } \frac { 1 } { \beta ! } \nabla _ { w } ^ { \beta } ( \nabla _ { w } \ell ) ( z , w ^ { \circ } ) ( w - w _ { 0 } ) ^ { \beta } , } \end{array}\tag{B2}
$$

where $w ^ { \circ } = w - \theta ( w - w _ { 0 } )$ for some $\theta \in ( 0 , 1 )$

Then for $v \in \ker \nabla ^ { 2 } R ( w _ { 0 } )$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { Z \sim P } \big [ \| v ^ { \top } \nabla _ { w } \ell ( Z , w _ { 0 } ) \| ^ { 2 } \big ] = \mathbb { E } _ { Z \sim P } \big [ \langle \nabla _ { w } \ell ( Z , w _ { 0 } ) \nabla _ { w } \ell ( Z , w _ { 0 } ) ^ { \top } v , v \rangle \big ] , } \\ & { \qquad = \langle \mathbb { E } _ { Z \sim P } \big [ \nabla _ { w } \ell ( Z , w _ { 0 } ) \nabla _ { w } \ell ( Z , w _ { 0 } ) ^ { \top } \big ] v , v \rangle , } \\ & { \qquad = \langle \mathrm { C o v } ( \nabla _ { w } \ell ( Z , w _ { 0 } ) ) v , v \rangle \le \alpha \langle \nabla ^ { 2 } R ( w _ { 0 } ) v , v \rangle = 0 , } \end{array}
$$

where the inequality follows from Assumption 4. Therefore, $v ^ { \top } \nabla _ { w } \ell ( Z , w _ { 0 } ) = 0 \ \mathrm { a . s . }$ , and by continuity of the loss function, $v ^ { \top } \nabla _ { w } \ell ( z , w _ { 0 } ) = 0$ for every $z \in { \mathcal { Z } }$ . Since the columns of $Q _ { 0 }$ form an orthonormal basis of ker $\nabla ^ { 2 } R ( w _ { 0 } )$ , we obtain $Q _ { 0 } ^ { \top } \nabla _ { w } \ell ( z , w _ { 0 } ) = 0$ for every $z \in { \mathcal { Z } }$ Consequently, by (B2), we obtain

$$
Q _ { 0 } ^ { \top } \nabla w \ell ( z , w ) = Q _ { 0 } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( z , w _ { 0 } ) ( w - w _ { 0 } ) + Q _ { 0 } ^ { \top } \sum _ { | \beta | = 2 } \frac { 1 } { \beta ! } \nabla _ { w } ^ { \beta } ( \nabla _ { w } \ell ) ( z , w ^ { 0 } ) ( w - w _ { 0 } ) ^ { \beta } ,
$$

Then, there exist ${ \delta } _ { 2 } > 0$ such that for all $w \in B _ { \delta _ { 2 } } ( w _ { 0 } )$

$$
\begin{array} { r l } & { \left\| Q _ { 0 } ^ { \top } \nabla _ { w } \ell ( \cdot , w ) \right\| \leq \left\| Q _ { 0 } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( z , w _ { 0 } ) \right\| \left\| w - w _ { 0 } \right\| + \left\| Q _ { 0 } ^ { \top } \displaystyle \sum _ { | \beta | = 2 } \frac { 1 } { \beta ! } \nabla _ { w } ^ { \beta } ( \nabla _ { w } \ell ) ( z , w ^ { \circ } ) \right\| \left\| w - w _ { 0 } \right\| ^ { 2 } } \\ & { \qquad \leq \left( \left\| Q _ { 0 } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( z , w _ { 0 } ) \right\| + \left\| Q _ { 0 } ^ { \top } \displaystyle \sum _ { | \beta | = 2 } \frac { 1 } { \beta ! } \nabla _ { w } ^ { \beta } ( \nabla _ { w } \ell ) ( z , w ^ { \circ } ) \right\| \left\| w - w _ { 0 } \right\| \right) \left\| w - w _ { 0 } \right\| } \\ & { \qquad \leq \left( \left\| Q _ { 0 } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( z , w _ { 0 } ) \right\| + 1 \right) \delta _ { 2 } } \\ & { \qquad < C \delta _ { 2 } . } \end{array}
$$

where ${ \cal C } = 2 \left\| \boldsymbol { Q } _ { 0 } ^ { \top } \nabla _ { w } ^ { 2 } \ell ( \cdot , w _ { 0 } ) \right\| _ { \infty } + 1$ . This yields the desired conclusion.

Finally, setting $\delta = \operatorname* { m i n } \{ \delta _ { 1 } , \delta _ { 2 } \}$ gives the desired conclusion.

## B.3 Proof of Corollary 5

Proof We use the truncated form of truncated Dudley integral (Proposition 17).

We first define an upper bound B on the loss function. By taking δ small if necessary, we may assume that $\overline { { B _ { \delta } } } ( w _ { 0 } ) \subset W$ . Then, since $\overline { { B _ { \delta } } } ( w _ { 0 } )$ is compact, we can define $B : =$ $\operatorname* { m a x } _ { ( z , w ) \in \mathcal { Z } \times \overline { { B _ { \delta } } } ( w _ { 0 } ) } \ell ( z , w )$ . That is, $\ell ( z , w ) \leq B$ holds for all $( z , w ) \in \mathcal { Z } \times B _ { \delta } ( w _ { 0 } )$

For every α > 0 with $\alpha < B$

$$
\mathrm { R a d } ( B _ { \delta } ( w _ { 0 } ) ) \leq 4 \alpha + \frac { 1 2 } { \sqrt { n } } \int _ { \alpha } ^ { 2 B } \sqrt { \log N \big ( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { 2 } ( P _ { n } ) , \tau \big ) } d \tau .
$$

Let $r =$ rank $\nabla _ { w } ^ { 2 } R ( w _ { 0 } )$ , and first consider the case $r > 0$ . By Main Theorem 2, there exists a constant $\rho _ { \delta } > 0$ such that, for every $\tau > \rho _ { \delta }$

$$
\begin{array} { l } { \displaystyle { N \big ( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { 2 } ( P _ { n } ) , \tau \big ) \le N \big ( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { \infty } ( P ) , \tau \big ) } } \\ { \displaystyle { \le \left( 1 + \frac { 2 L } { \tau - \rho _ { \delta } } \right) ^ { 2 \lambda ( w _ { 0 } ) } } . } \end{array}
$$

Set $\alpha : = 2 \rho _ { \delta }$ . Then $\tau > \rho _ { \delta }$ for all $\tau \in [ \alpha , 2 B ]$ , so the covering bound above is applicable on the interval of integration. Therefore,

$$
\int _ { \alpha } ^ { 2 B } \sqrt { \log N \big ( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { 2 } ( P _ { n } ) , \tau \big ) } d \tau \leq \sqrt { 2 \lambda ( w _ { 0 } ) } \int _ { \alpha } ^ { 2 B } \sqrt { \log \left( 1 + \frac { 2 L } { \tau - \rho _ { \delta } } \right) } d \tau .
$$

By the change of variables $s = \tau - \rho _ { \delta }$ , we get

$$
\sqrt { 2 \lambda ( w _ { 0 } ) } \int _ { \alpha } ^ { 2 B } \sqrt { \log \left( 1 + \frac { 2 L } { \tau - \rho _ { \delta } } \right) } d \tau \leq \sqrt { 2 \lambda ( w _ { 0 } ) } \int _ { 0 } ^ { 2 B } \sqrt { \log \left( 1 + \frac { 2 L } { s } \right) } d s .
$$

Using the elementary inequality $\begin{array} { r } { \int _ { 0 } ^ { a } \sqrt { \log { \left( 1 + \frac { c } { t } \right) } } d t \leq 2 \sqrt { c a } } \end{array}$ for $a > 0 , \ c > 0$ , we obtain

$$
\int _ { \alpha } ^ { 2 B } \sqrt { \log N \big ( \mathcal { F } _ { \ell } ( B _ { \delta } ( w _ { 0 } ) ) , L ^ { 2 } ( P _ { n } ) , \tau \big ) } d \tau \leq 4 \sqrt { 2 \lambda ( w _ { 0 } ) L B } .
$$

Thus the truncated Dudley bound gives

$$
\mathrm { R a d } ( B _ { \delta } ( w _ { 0 } ) ) \leq 4 \alpha + 4 8 \sqrt { \frac { 2 \lambda ( w _ { 0 } ) L B } { n } } .
$$

Since $\alpha = 2 \rho _ { \delta } = 2 C \delta ^ { 2 }$ , we have

$$
\operatorname { R a d } ( A ) \leq 8 C \delta ^ { 2 } + 4 8 { \sqrt { \frac { 2 \lambda ( w _ { 0 } ) L B } { n } } } .
$$

By Proposition 15, for every $\eta > 0 ,$ , with probability at least $\begin{array} { r } { 1 - \exp \left( - \frac { n \eta ^ { 2 } } { 2 B ^ { 2 } } \right) } \end{array}$ , we have

$$
\operatorname* { s u p } _ { w \in B _ { \delta } ( w _ { 0 } ) } | R ( w ) - R _ { S } ( w ) | \leq 2 \operatorname { R a d } ( B _ { \delta } ( w _ { 0 } ) ) + \eta .
$$

Therefore,

$$
\operatorname* { s u p } _ { w \in B _ { \delta } ( w _ { 0 } ) } | R ( w ) - R _ { S } ( w ) | \leq 1 6 C \delta ^ { 2 } + 9 6 \sqrt { \frac { 2 \lambda ( w _ { 0 } ) L B } { n } } + \eta .
$$

Finally, by writing 16C as $C _ { i }$ we obtain

$$
\operatorname* { s u p } _ { w \in B _ { \delta } ( w _ { 0 } ) } | R ( w ) - R _ { S } ( w ) | \leq 9 6 \sqrt { \frac { 2 \lambda ( w _ { 0 } ) L B } { n } } + C \delta ^ { 2 } + \eta .
$$

If $r = 0 ,$ , then the covering bound in Main Theorem 2 is interpreted with the right-hand side equal to 1. Hence the entropy integral over $[ \alpha , 2 B ]$ is zero, and the same argument gives

$$
\operatorname* { s u p } _ { w \in B _ { \delta } ( w _ { 0 } ) } | R ( w ) - R _ { S } ( w ) | \leq C \delta ^ { 2 } + \eta ,
$$

which is stronger than the desired bound. This completes the proof.

## Appendix C On the Validity of Assumption 2

We provide a justification for Assumption 2. In the following two examples, we show that the loss function of representative tasks satisfies Assumption 2 when the learning model is analytic. Here, a learning model $f ( x , w )$ is said to be analytic if there exists an open set ${ \mathcal { O } } _ { \mathcal { X } }$ containing X and f is analytic on $\mathcal { O } _ { \mathcal { X } } \times W$

Example 1 (Regression) In regression problems, we verify that the squared loss regression satisfies Assumption 2. The model outputs

$$
f ( x , w ) = \left( f _ { 1 } ( x , w ) , \ldots , f _ { m _ { 2 } } ( x , w ) \right) \in \mathbb { R } ^ { m _ { 2 } } ,
$$

are analytic on $\mathcal { O } _ { \mathcal { X } } \times W$ . The loss function is defined by

$$
\ell ( ( x , y ) , w ) = \frac { 1 } { 2 } \| f ( x , w ) - y \| ^ { 2 } .
$$

R is clearly analytic on $( \mathcal { O } _ { \mathcal { X } } \times \mathcal { O } _ { \mathcal { Y } } ) \times W$ for any open set $\mathcal { O } _ { \mathcal { V } }$ containing Y.

Example 2 (Classification) In classification problems, we define the output space $\mathcal { V } = \{ p \in$ $[ 0 , 1 ] ^ { m _ { 2 } } : \textstyle \sum _ { i = 1 } ^ { m _ { 2 } } p _ { i } = 1 \}$ . The observed labels are one-hot vectors in $\{ e _ { 1 } , \ldots , e _ { K } \} \subset \mathcal { V } ,$ where $e _ { i } ~ ( i = 1 , \ldots , K )$ denotes the vector whose i-th component is 1 and all other components are 0. We can then choose the open set $\mathcal { O } _ { \mathcal { Y } }$ containing Y by $\mathcal { O } _ { y } = ( - \varepsilon , 1 + \varepsilon ) ^ { m _ { 2 } }$ for suficiently small $\varepsilon > 0$ . Consider a classification model whose outputs are

$$
f ( x , w ) = ( f _ { 1 } ( x , w ) , \ldots , f _ { m _ { 2 } } ( x , w ) ) \in \mathcal { V } ,
$$

where $f _ { i } ( x , w ) \ ( i = 1 , \ldots , m _ { 2 } )$ are analytic on $\mathcal { O } _ { \mathcal { X } } \times W$ . Then the cross-entropy loss can be written as

$$
\ell ( ( x , y ) , w ) = - \sum _ { i = 1 } ^ { m _ { 2 } } y _ { i } \log f _ { i } ( x , w ) ,
$$

and this is analytic on $( \mathcal { O } _ { \mathcal { X } } \times \mathcal { O } _ { \mathcal { Y } } ) \times W$ . This shows Assumption 2 is satisfied on $( { \mathcal { O } } _ { \mathcal { X } } \ \times$ $\mathcal { O } _ { \mathcal { Y } } ) \times W$

Many models are analytic with respect to both the parameters and the samples. For example, in neural networks, classical activation functions such as tanh and the sigmoid function are analytic. Although the widely used ReLU is not analytic, GELU, which was proposed as an analytic alternative, has become widely used [53, 54].

## Appendix D Regarding Assumption 4

In addition to the negative log-likelihood loss, squared loss regression with bounded noise is another example that satisfies Assumption 4.

Example 3 (Squared loss regression with bounded outputs) Let the output space be ${ \mathcal { V } } =$ $[ - M , M ]$ for some $M > 0$ . Let $f _ { w ^ { \star } } : \mathcal { X }  \mathcal { Y }$ be the true function, and

$$
Y = f _ { w ^ { \star } } ( X ) + \varepsilon , \qquad \mathbb { E } [ \varepsilon \mid X ] = 0 ,
$$

where the noise variable ε satisfies $f w \star ( X ) + \varepsilon \in [ - M , M ]$ . We consider the squared loss

$$
\ell ( ( x , y ) , w ) = \frac { 1 } { 2 } \big ( y - f _ { w } ( x ) \big ) ^ { 2 } .
$$

Proof that this example satisfies Assumption $\it 4 . \mathrm { \textrm { B y } }$ the assumption on the noise, we have $| \varepsilon | = | Y - f _ { w ^ { \star } } ( X ) | \leq 2 M$ . Noting that $\begin{array} { r } { \mathbb { E } [ \nabla _ { w } \ell ( Z , w _ { 0 } ) ] = \mathbb { E } [ - \varepsilon \nabla _ { w } f _ { w _ { 0 } } ( X ) ] = 0 } \end{array}$ holds for every $w _ { 0 } \in W _ { 0 }$ , we have

$$
\begin{array} { r l } & { \nabla ^ { 2 } R ( w _ { 0 } ) = \mathbb { E } \left[ \nabla _ { w } f _ { w _ { 0 } } ( X ) \nabla _ { w } f _ { w _ { 0 } } ( X ) ^ { \top } \right] - \mathbb { E } \left[ \varepsilon \nabla _ { w } ^ { 2 } f _ { w _ { 0 } } ( X ) \right] } \\ & { \qquad = \mathbb { E } \left[ \nabla _ { w } f _ { w _ { 0 } } ( X ) \nabla _ { w } f _ { w _ { 0 } } ( X ) ^ { \top } \right] , } \end{array}
$$

and,

$$
\begin{array} { r l } & { \mathrm { C o v } \left( \nabla _ { \boldsymbol { w } } \ell ( Z , { \boldsymbol { w } } _ { 0 } ) \right) = \mathbb { E } \left[ \nabla _ { \boldsymbol { w } } \ell ( Z , { \boldsymbol { w } } _ { 0 } ) \nabla _ { \boldsymbol { w } } \ell ( Z , { \boldsymbol { w } } _ { 0 } ) ^ { \top } \right] } \\ & { \quad \quad \quad = \mathbb { E } \left[ \varepsilon ^ { 2 } \nabla _ { \boldsymbol { w } } f _ { { \boldsymbol { w } } _ { 0 } } ( X ) \nabla _ { \boldsymbol { w } } f _ { { \boldsymbol { w } } _ { 0 } } ( X ) ^ { \top } \right] } \\ & { \quad \quad \quad \preceq 4 { \boldsymbol { M } } ^ { 2 } \mathbb { E } \left[ \nabla _ { \boldsymbol { w } } f _ { { \boldsymbol { w } } _ { 0 } } ( X ) \nabla _ { \boldsymbol { w } } f _ { { \boldsymbol { w } } _ { 0 } } ( X ) ^ { \top } \right] . } \end{array}
$$

Therefore, we have

$$
\operatorname { C o v } \big ( \nabla _ { \boldsymbol { w } } \ell ( Z , { \boldsymbol { w } } _ { 0 } ) \big ) \preceq 4 M ^ { 2 } \nabla ^ { 2 } R ( { \boldsymbol { w } } _ { 0 } ) .
$$

Thus Assumption 4 holds with $\alpha = 4 M ^ { 2 }$

## Appendix E Basic Properties in Singular Learning Theory

## E.1 Existence of the Local RLCT

In this section, we establish the existence of the limit required to define the local RLCT.

Lemma 9 For any local minimizer $w _ { * } \in W$ of R, there exists $\delta > 0$ such that

$$
\lambda \big ( R , \overline { { B _ { \delta } } } ( w _ { * } ) \big ) = \lambda \big ( R , \overline { { B _ { \delta ^ { \prime } } } } ( w _ { * } ) \big ) \quad f o r \ a l l \ 0 < \delta ^ { \prime } \leq \delta .
$$

Proof Let $f ( w ) = ( R ( w ) - R ( w _ { * } ) ) ^ { 2 }$ , then by Hironaka’s resolution of singularities $[ 1 4 , 5 5 ]$ there exist a real analytic manifold M and a real analytic map $g : { \mathfrak { M } } \to W$ satisfying the following conditions: Denoting ${ \mathfrak { M } } _ { 0 } = \{ u \in { \mathfrak { M } } : f \circ g ( u ) = 0 \}$ , then for any $u \in \mathfrak { M } _ { 0 }$

$$
\begin{array} { c } { { f \circ g ( u ) = u ^ { 2 k _ { \alpha } } , } } \\ { { | J _ { g } ( u ) | = b _ { \alpha } ( u ) u ^ { h _ { \alpha } } , } } \end{array}
$$

where $k _ { \alpha } , h _ { \alpha } ~ \in ~ \mathbb { N } _ { 0 } ^ { d }$ and $b _ { \alpha } ( u )$ is a positive real analytic function. In this case, using a coordinate neighborhood system $\{ U _ { \alpha } \}$ on M, the ζ function on $C$ satisfies

$$
\begin{array} { l } { \displaystyle \zeta ( s ) = \int _ { \overline { { B _ { \delta } } } ( w _ { * } ) } ( R ( w ) - R ( w _ { * } ) ) ^ { 2 s } d w , } \\ { \displaystyle \quad = \int _ { g ^ { - 1 } \left( \overline { { B _ { \delta } } } ( w _ { * } ) \right) } f \circ g ( u ) ^ { s } | J _ { g } ( u ) | d u , } \\ { \displaystyle \quad = \sum _ { \alpha } \int _ { U _ { \alpha } \cap g ^ { - 1 } \left( \overline { { B _ { \delta } } } ( w _ { * } ) \right) } u ^ { 2 k _ { \alpha } s + h _ { \alpha } } b _ { \alpha } ( u ) d u . } \end{array}
$$

Therefore, the only candidates for the poles of ζ are $\left\{ - \frac { h _ { \alpha _ { i } } + 1 } { 2 k _ { \alpha _ { i } } } : i = 1 , \cdot \cdot \cdot , d \right\}$ . By the inclusion of the integration domains, if $\delta ^ { \prime } < \delta$ then $\lambda \left( R , \overline { { B _ { \delta } } } ( w _ { * } ) \right) \leq \lambda \left( R , \overline { { B _ { \delta ^ { \prime } } } } ( w _ { * } ) \right)$ , thus $\lambda ( \overline { { B _ { \delta } } } ( w _ { * } ) )$ is nonincreasing in δ. Moreover, since the candidates of RLCT $\textstyle \left\{ { \frac { h _ { \alpha i } + 1 } { 2 k _ { \alpha i } } } : i = 1 , \cdots , d \right\}$ is finite, we obtain the claim. □

## E.2 Some Important Properties of RLCT and Local RLCT

In this subsection, we state some important properties of the RLCT, which also hold for the local RLCT.

First, the RLCT satisfies a monotonicity property with respect to functions. This fact plays a central role in the proof of the main result.

Proposition 10 (Monotonicity) Let $f , g : O \to \mathbb { R }$ be nonnegative real analytic functions defined on an open set $O \subset \mathbb { R } ^ { d } . \ I f \ f \ \leq \ g$ , that is, $f ( x ) \leq g ( x )$ holds at every point $x \in O _ { \mathrm { { i } } }$ then for any compact set $C \subset O$ ，

$$
\lambda ( f , C ) \leq \lambda ( g , C ) .
$$

Proof Let s be a negative real number. Then

$$
\int _ { C } g ( w ) ^ { s } d w \leq \int _ { C } f ( w ) ^ { s } d w ,
$$

and hence, if the right-hand side converges, then the left-hand side must also converge. By the definition of the real log canonical threshold, the right-hand side converges for any s such that $s > - \lambda ( f , C )$ . Therefore, we must have $- \lambda ( g , C ) \le - \lambda ( f , C )$ . That is, we obtain $\lambda ( f , C ) \leq \lambda ( g , C )$ □

The following proposition is useful for computing local RLCTs in practice.

Proposition 11 (Invariance under Multiplicative Equivalence) Let $f , g : O \to \mathbb { R }$ be nonnegative real analytic functions defined on an open set $O \subset \mathbb { R } ^ { d }$ . If there exist constans $c , c ^ { \prime } > 0$ such that $c g \leq f \leq c ^ { \prime } g$ , then

$$
\lambda ( f , C ) = \lambda ( g , C ) .
$$

Proof This follows from Proposition 10 and the invariance of the convergence of the $\zeta$ function under multiplication by a positive constant. □

Moreover, the RLCT is invariant under difeomorphic transformations of the parameter space.

Proposition 12 (Difeomorphism invariance) The RLCT is invariant under difeomorphisms. That $i s , f o r$ a nonnegative real analytic function $f : O $ R defined on an open set $O \subset \mathbb { R } ^ { d }$ , when there exist an open set $U \subset \mathbb { R } ^ { d }$ and a difeomorphism $\Phi : U  O$

$$
\lambda ( f , C ) = \lambda ( f \circ \Phi , \Phi ^ { - 1 } ( C ) ) ,
$$

holds.

Proof Define the ζ-function associated with f and $C$ by

$$
\begin{array} { l } { \displaystyle \zeta _ { 1 } ( s ) = \int _ { C } f ( w ) ^ { s } d w , } \\ { \displaystyle \zeta _ { 2 } ( s ) = \int _ { \Phi ^ { - 1 } ( C ) } f ( \Phi ( u ) ) ^ { s } d u . } \end{array}
$$

It sufices to show that $\zeta _ { 1 }$ and $\zeta _ { 2 }$ have the same convergence/divergence behavior on the real axis. In what follows, let $s \in \mathbb { R }$

By a change of variables in the integral,

$$
\zeta _ { 1 } ( s ) = \int _ { \Phi ^ { - 1 } ( C ) } f ( \Phi ( u ) ) ^ { s } | J _ { \Phi } ( u ) | d u ,
$$

holds. Since Φ is a difeomorphism and C is compact, $\Phi ^ { - 1 } ( C )$ is also compact. Hence there exist constants $c , c ^ { \prime } > 0$ such that $c \leq | J _ { \Phi } ( u ) | \leq c ^ { \prime }$ holds. Therefore,

$$
c \zeta _ { 2 } ( s ) \leq \zeta _ { 1 } ( s ) \leq c ^ { \prime } \zeta _ { 2 } ( s )
$$

holds. Consequently, $\zeta _ { 1 }$ and $\zeta _ { 2 }$ have the same convergence/divergence behavior on the real axis, and thus we must have $\lambda ( f , C ) = \lambda ( f \circ \Phi , \Phi ^ { - 1 } ( C ) )$ □

Next, we show that the RLCT is determined only by a neighborhood of the zero set of the function.

Proposition 13 (Removal of nonzero points) Let f be a nonnegative real analytic function $f : O \to \mathbb { R }$ defined on an open set $O \subset \mathbb { R } ^ { d }$ whose zero set $Z = \{ x \in O : f ( x ) = 0 \}$ is compact. For any compact set $C \subset O$ containing $Z ,$ we have

$$
\lambda ( f , C ) = \lambda ( f , T _ { \delta } ( Z ) ) ,
$$

for all suficiently small $\delta > 0$ such that $T _ { \delta } ( Z ) \subset C _ { \delta }$ , where $T _ { \delta } ( Z ) = \{ x \in O : d ( x , Z ) \leq \delta \}$

Proof We have

$$
\zeta ( s ) = \int _ { C } f ( u ) ^ { s } d u = \int _ { T _ { \delta } ( Z ) } f ^ { s } d u + \int _ { C \backslash T _ { \delta } ( Z ) } f ^ { s } d u .
$$

Since the second integral has no poles, the poles of the zeta function are determined only by the first integral. Therefore, we obtain

$$
\lambda ( f , C ) = \lambda ( f , T _ { \delta } ( Z ) ) .
$$

Finally, we compute the RLCT of a function f whose Hessian is non-degenerate.

Proposition 14 (RLCT of a sum of squares) Let $f : \mathbb { R } ^ { d } $ R be given by

$$
f ( u ) = \left\| u \right\| ^ { 2 } = u _ { 1 } ^ { 2 } + \cdot \cdot \cdot + u _ { d } ^ { 2 } .
$$

and C contains the origin in its interior. Then we have

$$
\lambda ( f , C ) = { \frac { d } { 2 } } .
$$

Proof Since f vanishes only at the origin, for any $\delta > 0$ such that $B _ { \delta } ( 0 ) \subset C$ , we have $\lambda ( f , C ) = \lambda ( f , B _ { \delta } ( 0 ) )$ (Proposition 13). Consider the local zeta function near the origin:

$$
\zeta ( s ) = \int _ { B _ { \delta } ( 0 ) } f ( u ) ^ { s } d u = \int _ { B _ { \delta } ( 0 ) } | u | ^ { 2 s } d u .
$$

Using polar coordinates $u \ : = \ : r \omega$ with $r \geq 0$ and $\omega \in S ^ { d - 1 }$ , the $( d - 1 )$ -dimensional unit sphere, we have $d u = r ^ { d - 1 } d r d \omega$ . Therefore,

$$
\zeta ( s ) = \operatorname { v o l } ( S ^ { d - 1 } ) \int _ { 0 } ^ { \delta } r ^ { 2 s + d - 1 } d r = \operatorname { v o l } ( S ^ { d - 1 } ) { \frac { \delta ^ { 2 s + d } } { 2 s + d } } ,
$$

where vol $( S ^ { d - 1 } )$ denotes the usual volume of $S ^ { d - 1 }$ . Since the rightmost pole on the right-hand side is $- \frac { d } { 2 }$ , we have

$$
\lambda ( f , C ) = \frac { d } { 2 } .
$$

## Appendix F Computation of the Local RLCT

We explicitly compute the local RLCT claimed in Subsection 4.2 and 4.3. The computational method difers depending on the point under consideration.

## F.1 Local RLCTs of Reduced Rank Regression

The computation in (3) is as follows.

## F.1.1 Derivation of $\lambda ( w _ { 1 } )$

Recall that $w _ { 1 } = ( U _ { 1 } , V _ { 1 } )$ is given by

$$
U _ { 1 } = { \binom { I _ { h } } { 0 } } \in \mathbb { R } ^ { m _ { 2 } \times h } , \qquad V _ { 1 } = { \binom { I _ { r } \ 0 } { 0 \ 0 } } \in \mathbb { R } ^ { h \times m _ { 1 } } ,
$$

Let $U ~ \in ~ \mathbb { R } ^ { m _ { 2 } \times h }$ and $V ~ \in ~ \mathbb { R } ^ { h \times m _ { 1 } }$ be matrices in neighborhoods of $U _ { 1 }$ and $V _ { 1 }$ respectively. Since $\mathbb { E } [ X X ^ { \top } ]$ is positive definite, there exist constants $c , c ^ { \prime } > 0$ such that

$$
c \| U V - T ^ { \star } \| _ { F } ^ { 2 } \leq R ( w ) \leq c ^ { \prime } \| U V - T ^ { \star } \| _ { F } ^ { 2 } .
$$

Hence, the local RLCT of $R$ at $w _ { 1 }$ is equal to that of $\| U V - T ^ { \star } \| _ { F } ^ { 2 }$ by Proposition 11. Furthermore, we simplify the expression by applying a difeomorphism. Write

$$
U = \left( \begin{array} { c } { { \mathcal { U } _ { 1 } } } \\ { { \mathcal { U } _ { 2 } } } \end{array} \right) , \qquad T ^ { \star } = \left( \begin{array} { c } { { T _ { 0 } } } \\ { { 0 } } \end{array} \right) ,
$$

where $\mathcal { U } _ { 1 } \in \mathbb { R } ^ { h \times h }$ and

$$
T _ { 0 } = { \binom { I _ { r } \ 0 } { 0 \ 0 } } \in \mathbb { R } ^ { h \times m _ { 1 } } .
$$

Since $\mathcal { U } _ { 1 } = I _ { h }$ at $w _ { 1 }$ , the matrix $\mathcal { U } _ { 1 }$ is invertible in a suficiently small neighborhood of $w _ { 1 }$ . We can therefore write

$$
U V - T ^ { \star } = \binom { \mathcal { U } _ { 1 } V - T _ { 0 } } { \mathcal { U } _ { 2 } V } = \binom { \mathcal { U } _ { 1 } V - T _ { 0 } } { \mathcal { U } _ { 2 } \mathcal { U } _ { 1 } ^ { - 1 } ( \mathcal { U } _ { 1 } V - T _ { 0 } ) + \mathcal { U } _ { 2 } \mathcal { U } _ { 1 } ^ { - 1 } T _ { 0 } } .
$$

Then we introduce the difeomorphic coordinate transformation $\Phi ~ : ~ ( U , V ) \mapsto \mapsto $ $( { \mathcal { U } } _ { 1 } , G , H )$ , where

$$
G = \mathcal { U } _ { 1 } V - T _ { 0 } , \qquad H = \mathcal { U } _ { 2 } \mathcal { U } _ { 1 } ^ { - 1 } .
$$

Under this transformation,

$$
U V - T ^ { \star } = \binom { G } { H G + H T _ { 0 } } .
$$

Letting $R = \left( \begin{array} { l l } { { I _ { h } } } & { { 0 } } \\ { { - H \ I _ { m _ { 2 } - h } } } \end{array} \right)$ , we have $U V - T ^ { \star } = R ^ { - 1 } \left( { { G \atop H T _ { 0 } } } \right)$ . Since Φ is a difeomorphism and left multiplication by $R ^ { - 1 }$ preserves the Frobenius norm up to constant factors, Propositions 12 and 11 imply that $\| U V - T ^ { \star } \| _ { F } ^ { 2 }$ and $\| G \| _ { F } ^ { 2 } + \| H T _ { 0 } \| _ { F } ^ { 2 }$ have the same local RLCT. By Proposition 14, we obtain

$$
\lambda ( w _ { 1 } ) = \frac { h m _ { 1 } + r ( m _ { 2 } - h ) } { 2 } .
$$

## F.1.2 Derivation of $\lambda ( w _ { 2 } )$

Recall that $w _ { 2 } = ( U _ { 2 } , V _ { 2 } )$ is

$$
U _ { 2 } = { \binom { I _ { r } \ 0 } { 0 \ 0 } } \in \mathbb { R } ^ { m _ { 2 } \times h } , \qquad V _ { 2 } = \left( I _ { h } \ 0 \right) \in \mathbb { R } ^ { h \times m _ { 1 } } .
$$

Writing

$$
V = \left( \mathcal { V } _ { 1 } \ \mathcal { V } _ { 2 } \right) , \qquad T ^ { \star } = \left( T _ { 0 } \ 0 \right) ,
$$

where $\boldsymbol { \nu } _ { 1 } \in \mathbb { R } ^ { h \times h }$ and

$$
\begin{array} { r } { T _ { 0 } = \binom { I _ { r } \ 0 } { 0 \ 0 } \in \mathbb { R } ^ { m _ { 2 } \times h } , } \end{array}
$$

the proof is essentially the same as that for $\lambda ( w _ { 1 } )$ . We obtain

$$
\lambda ( w _ { 2 } ) = \frac { h m _ { 2 } + r ( m _ { 1 } - h ) } { 2 } .
$$

## F.1.3 Derivation of $\lambda ( w _ { 3 } )$

Recall that $w _ { 3 } = ( U _ { 3 } , V _ { 3 } )$ is given by

$$
U _ { 3 } = { \binom { I _ { r } \ 0 } { 0 \ 0 } } \in \mathbb { R } ^ { m _ { 2 } \times h } , \qquad V _ { 3 } = \left( I _ { r } \ 0 \right) \in \mathbb { R } ^ { h \times m _ { 1 } } .
$$

The main idea is to use Main Theorem Case (2) of [44]. By Lemma $5 \ \mathrm { ( a ) }$ of [44], the local RLCT of R is equal to that of $\| S \| ^ { 2 } + \| \dot { \mathcal { U } } \mathcal { V } \| _ { F } ^ { 2 }$ , where $S \in \mathbb { R } ^ { D _ { r } } , \bar { D } _ { r } = \bar { r } ( \bar { m } _ { 1 } +$ $m _ { 2 } - r ) , \mathcal { U } \in \mathbb { R } ^ { ( h - r ) \times ( m _ { 1 } - r ) }$ and $\mathcal { V } \in \mathbb { R } ^ { ( m _ { 2 } - r ) \times ( h - r ) }$ . Since $\| ( t \mathcal { U } ) ( t \mathcal { V } ) \| _ { F } ^ { 2 } = t ^ { 4 } \| \mathcal { U } \mathcal { V } \| _ { F } ^ { 2 }$ for $t \in \mathbb { R }$ , the RLCT is equal to the local RLCT. Indeed, letting $\varepsilon > 0$ and $p$ be the number of parameters of $( u , \nu )$

$$
\int _ { \| ( \boldsymbol { u } , \boldsymbol { \nu } ) \| < \varepsilon } \| ( t \mathcal { U } ) ( t \mathcal { V } ) \| _ { F } ^ { 2 } s d \mathcal { U } d \mathcal { V } = \varepsilon ^ { p + 4 s } \int _ { \| ( \boldsymbol { A } , \boldsymbol { B } ) \| < 1 } \| \mathcal { U } \mathcal { V } \| _ { F } ^ { 2 } s d \boldsymbol { A } d \boldsymbol { B } ,
$$

and $\varepsilon ^ { p + 4 s }$ does not change the poles. Thus, we can use Main Theorem Case (2) of [44] for the local RLCT of $\| \mathcal { U V } \| _ { F } ^ { 2 }$ , and we have

$$
\lambda ( w _ { 3 } ) = { \frac { 1 } { 2 } } \left[ r ( m _ { 1 } + m _ { 2 } - r ) + \operatorname* { m i n } _ { 0 \leq j \leq h - r , j \in { \mathbb Z } } \left\{ j ( m _ { 2 } - r ) + ( m _ { 1 } - r - j ) ( h - r - j ) \right\} \right] .
$$

## F.2 Local RLCTs of Linear Neural Networks

The computation in (5) is as follows. The computational method difers depending on the point under consideration.

## F.2.1 Derivation of $\lambda ( w _ { 1 } )$

Recall that $w _ { 1 } = \left( I _ { h _ { 1 } \times m _ { 1 } } , I _ { h _ { 2 } \times h _ { 1 } } , 0 _ { m _ { 2 } \times h _ { 2 } } \right)$ . Write

$$
W _ { \mathrm { i n } } = I _ { h _ { 1 } \times m _ { 1 } } + E , \qquad W _ { \mathrm { h i d } } = I _ { h _ { 2 } \times h _ { 1 } } + F , \qquad W _ { \mathrm { o u t } } = G ,
$$

in the neighborhood of $w _ { 1 }$ where $E \in \mathbb { R } ^ { h _ { 1 } \times m _ { 1 } } , F \in \mathbb { R } ^ { h _ { 2 } \times h _ { 1 } } , G \in \mathbb { R } ^ { m _ { 2 } \times h _ { 2 } }$ . Then $w _ { 1 }$ corresponds to $E = F = G = 0$ , and $W _ { \mathrm { o u t } } W _ { \mathrm { h i d } } W _ { \mathrm { i n } } = G \left( I _ { h _ { 2 } \times h _ { 1 } } + F \right) \left( I _ { h _ { 1 } \times m _ { 1 } } + E \right)$

Then the generalization error is $R ( w ) = \mathbb { E } \left| | G \left( I _ { h _ { 2 } \times h _ { 1 } } + F \right) \left( I _ { h _ { 1 } \times m _ { 1 } } + E \right) X | ^ { 2 } \right|$ Let $M = ( I _ { h _ { 2 } \times h _ { 1 } } + F ) ( I _ { h _ { 1 } \times m _ { 1 } } + E )$ , and note that $M \mathbb { E } [ X X ^ { \top } ] M ^ { \top }$ is positive definite in a suficiently small neighborhood of $( E , F ) = ( 0 , 0 )$ . Therefore there exist constants $c , c ^ { \prime } > 0$ such that

$$
c \| G \| _ { F } ^ { 2 } \leq R ( w ) \leq c ^ { \prime } \| G \| _ { F } ^ { 2 } .
$$

Thus, by Proposition $1 1 , R ( w )$ has the same local RLCT at $w _ { 1 } \ \mathrm { a s } \ \| G \| _ { F } ^ { 2 }$

Since $G \in \mathbb { R } ^ { m _ { 2 } \times h _ { 2 } }$ consists of $m _ { 2 } h _ { 2 }$ parameters, by Proposition 14 we have

$$
\lambda ( w _ { 1 } ) = \frac { m _ { 2 } h _ { 2 } } { 2 } .
$$

## F.2.2 Derivation of $\lambda ( w _ { 2 } )$

Recall that $w _ { 2 } = \left( I _ { h _ { 1 } \times m _ { 1 } } , 0 _ { h _ { 2 } \times h _ { 1 } } , I _ { m _ { 2 } \times h _ { 2 } } \right)$ . Write

$$
\begin{array} { r } { W _ { \mathrm { i n } } = I _ { h _ { 1 } \times m _ { 1 } } + E , \qquad W _ { \mathrm { h i d } } = F , \qquad W _ { \mathrm { o u t } } = I _ { m _ { 2 } \times h _ { 2 } } + G , } \end{array}
$$

in the neighborhood of $w _ { 2 }$ , where $E \in \mathbb { R } ^ { h _ { 1 } \times m _ { 1 } } , \ F \in \mathbb { R } ^ { h _ { 2 } \times h _ { 1 } }$ , and $G \in \mathbb { R } ^ { m _ { 2 } \times h _ { 2 } }$ Then $w _ { 2 }$ corresponds to $E = F = G = 0$ , and the generalization error is $R ( w ) =$ $\mathbb { E } \left[ | ( I _ { m _ { 2 } \times h _ { 2 } } + G ) F \left( I _ { h _ { 1 } \times m _ { 1 } } + E \right) X | ^ { 2 } \right]$

First, we absorb the dependence on G into a local coordinate transformation. Since m<sub>2</sub> $\leq h _ { 2 }$ , write

$$
I _ { m _ { 2 } \times h _ { 2 } } + G = \left( I _ { m _ { 2 } } + G _ { 1 } \ G _ { 2 } \right) , \qquad F = { \binom { F _ { 1 } } { F _ { 2 } } } ,
$$

where $G _ { 1 } ~ \in ~ \mathbb { R } ^ { m _ { 2 } \times m _ { 2 } } , G _ { 2 } ~ \in ~ \mathbb { R } ^ { m _ { 2 } \times ( h _ { 2 } - m _ { 2 } ) } , F _ { 1 } ~ \in ~ \mathbb { R } ^ { m _ { 2 } \times h _ { 1 } } , F _ { 2 } ~ \in ~ \mathbb { R } ^ { ( h _ { 2 } - m _ { 2 } ) \times h _ { 1 } }$ . Then $\left( I _ { m _ { 2 } \times h _ { 2 } } + G \right) F = \left( I _ { m _ { 2 } } + G _ { 1 } \right) F _ { 1 } + G _ { 2 } F _ { 2 }$ . Define $\widetilde { F } : = ( I _ { m _ { 2 } } + G _ { 1 } ) F _ { 1 } + G _ { 2 } F _ { 2 } \in$ R<sup>m2×h1</sup>. Since $I _ { m _ { 2 } } + G _ { 1 }$ is invertible in a suficiently small neighborhood of $G _ { 1 } ~ = ~ 0$ , the transformation $( F _ { 1 } , F _ { 2 } , G ) \mapsto ( { \widetilde { F } } , F _ { 2 } , G )$ is invertible. Therefore,

$$
R ( w ) = \mathbb { E } \left[ \left| \widetilde { F } \left( I _ { h _ { 1 } \times m _ { 1 } } + E \right) X \right| ^ { 2 } \right] .
$$

Moreover, $I _ { h _ { 1 } \times m _ { 1 } } + E$ has full row rank $h _ { 1 }$ in a suficiently small neighborhood of $E ~ = ~ 0$ . Therefore, since $\left( I _ { h _ { 1 } \times m _ { 1 } } + E \right) \mathbb { E } [ X X ^ { \top } ] \left( I _ { h _ { 1 } \times m _ { 1 } } + E \right) ^ { \top }$ is positive definite,

there exist constants $c , c ^ { \prime } > 0$ such that

$$
c \| \widetilde { F } \| _ { F } ^ { 2 } \le R ( E , F , G ) \le c ^ { \prime } \| \widetilde { F } \| _ { F } ^ { 2 } .
$$

Thus, by Proposition 11, $R ( w )$ has the same local RLCT at $w _ { 2 }$ as $\| \widetilde { F } \| _ { F } ^ { 2 }$ . Since $\widetilde { F } \in$ $\mathbb { R } ^ { m _ { 2 } \times h _ { 1 } }$ consists of $m _ { 2 } h _ { 1 }$ <sub>1</sub> parameters, by Proposition 14 we have

$$
\lambda ( w _ { 2 } ) = \frac { m _ { 2 } h _ { 1 } } { 2 } .
$$

## F.2.3 Derivation of $\lambda ( w _ { 3 } )$

Recall that $w _ { 3 } = ( 0 _ { h _ { 1 } \times m _ { 1 } } , I _ { h _ { 2 } \times h _ { 1 } } , I _ { m _ { 2 } \times h _ { 2 } } )$ . Write

$$
W _ { \mathrm { i n } } = E , \qquad W _ { \mathrm { h i d } } = I _ { h _ { 2 } \times h _ { 1 } } + F , \qquad W _ { \mathrm { o u t } } = I _ { m _ { 2 } \times h _ { 2 } } + G ,
$$

in the neighborhood of $w _ { 3 } .$ , where $E \in \mathbb { R } ^ { h _ { 1 } \times m _ { 1 } } , \ F \in \mathbb { R } ^ { h _ { 2 } \times h _ { 1 } }$ <sup>1</sup> and $G \in \mathbb { R } ^ { m _ { 2 } \times h _ { 2 } }$ Then $w _ { 3 }$ corresponds to $E = F = G = 0$ , and the generalization error is $R ( w ) =$ $\begin{array} { r } { \mathbb { E } \left[ | ( I _ { m _ { 2 } \times h _ { 2 } } + G ) \left( I _ { h _ { 2 } \times h _ { 1 } } + F \right) E X | ^ { 2 } \right] } \end{array}$

Let $\begin{array} { r } { M ( F , G ) ~ = ~ \left( I _ { m _ { 2 } \times h _ { 2 } } + G \right) \left( \bar { I } _ { h _ { 2 } \times h _ { 1 } } + F \right) ~ \in ~ \mathbb { R } ^ { m _ { 2 } \times h _ { 1 } } } \end{array}$ . Noting that $M ( 0 , 0 ) =$ $I _ { m _ { 2 } \times h _ { 1 } }$ and $m _ { 2 } \leq h _ { 1 }$ , let

$$
M ( F , G ) = \left( M _ { 1 } ( F , G ) \ M _ { 2 } ( F , G ) \right) , \qquad E = { \binom { E _ { 1 } } { E _ { 2 } } } ,
$$

where $M _ { 1 } ( F , G ) \ \in \ \mathbb { R } ^ { m _ { 2 } \times m _ { 2 } } , M _ { 2 } ( F , G ) \ \in \ \mathbb { R } ^ { m _ { 2 } \times ( h _ { 1 } - m _ { 2 } ) }$ and $E _ { 1 } ~ \in ~ \mathbb { R } ^ { m _ { 2 } \times m _ { 1 } } , E _ { 2 } ~ \in$ $\mathbb { R } ^ { ( h _ { 1 } - m _ { 2 } ) \times \dot { m } _ { 1 } }$ . Since $M _ { 1 } ( 0 , 0 ) = I _ { m _ { 2 } } , M _ { 1 } ( F , G )$ is invertible in a suficiently small neighborhood of $( F , G ) = ( 0 , 0 )$ . Accordingly, define $\widetilde { E } : = { \cal M } _ { 1 } ( { \cal F } , { \cal G } ) E _ { 1 } + { \cal M } _ { 2 } ( { \cal F } , { \cal G } ) E _ { 2 } =$ $M ( F , G ) E \in$ R<sup>m2×m1</sup>. Then the transformation $( E _ { 1 } , E _ { 2 } , F , G ) \ \mapsto \ ( \widetilde E , E _ { 2 } , F , G )$ is invertible. Therefore, there exist constants $c , c ^ { \prime } > 0$ such that

$$
c \| \widetilde { E } \| _ { F } ^ { 2 } \leq R ( w ) \leq c ^ { \prime } \| \widetilde { E } \| _ { F } ^ { 2 } .
$$

Thus, by Proposition 11, $R ( w )$ has the same local RLCT at $w _ { 3 }$ as $\| \widetilde { E } \| _ { F } ^ { 2 }$ . Since $\widetilde { E } \in$ $\mathbb { R } ^ { m _ { 2 } \times m _ { 1 } }$ consists of $m _ { 2 } m _ { 1 }$ parameters, by Proposition 14 we have

$$
\lambda ( w _ { 3 } ) = \frac { m _ { 2 } m _ { 1 } } { 2 } .
$$

## Appendix G High-Dimensional Statistical Techniques

By using techniques from high-dimensional statistics, we can derive high-probability upper bounds for stochastic processes. In this chapter, we introduce some basic propositions.

Proposition 15 Let $A \subset W$ and $B = \operatorname* { s u p } _ { w \in A } \| \ell ( \cdot , w ) \| _ { \infty }$ . For any $\eta > 0$

$$
\mathbb { P } \left( \operatorname* { s u p } _ { w \in A } | R ( w ) - R _ { S } ( w ) | \geq 2 \mathrm { { R a d } } ( A ) + \eta \right) \leq \exp \left( - \frac { n \eta ^ { 2 } } { 2 B ^ { 2 } } \right) ,
$$

holds, where

$$
\operatorname { R a d } ( A ) : = \mathbb { E } _ { Z } \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { w \in A } \left| { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \sigma _ { i } \ell ( Z _ { i } , w ) \right| \right] ,
$$

and each component of $\sigma = ( \sigma _ { 1 } , \ldots , \sigma _ { n } )$ is an independent Rademacher random variable, i.e.,

$$
\mathbb { P } ( \sigma _ { i } = 1 ) = \mathbb { P } ( \sigma _ { i } = - 1 ) = \frac { 1 } { 2 } .
$$

Proof This follows from the standard Rademacher complexity bound; see, e.g., [36, Theorem 4.10]. □

Proposition 16 (Massart’s Lemma) Let H be a finite subset $o f \mathbb { R } ^ { n }$ , and $D = \operatorname* { m a x } _ { h \in { \mathcal { H } } } \| h \| _ { 2 }$ ， then

$$
\mathbb { E } _ { \sigma } \left[ \operatorname* { m a x } _ { h \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } h _ { i } \right] \le \frac { D } { n } \sqrt { 2 \log | \mathcal { H } | } ,
$$

where $\sigma = ( \sigma _ { 1 } , \ldots , \sigma _ { n } )$ is a vector of independent Rademacher variables and $h _ { i }$ is the $_ { i - t h }$ element of $h \in \mathcal H$

Proof If $D = 0 \ \mathrm { o r } \ | \mathcal { H } | = 1$ , the claim is immediate. Hence assume $D > 0$ and $| \mathcal { H } | \geq 2$ . For any $\lambda > 0$ , we have

$$
\begin{array} { r } { \mathbb { E } _ { \sigma } \left[ \underset { h \in \mathcal { H } } { \operatorname* { m a x } } \frac { 1 } { n } \underset { i = 1 } { \overset { n } { \sum } } \sigma _ { i } h _ { i } \right] = \frac { 1 } { \lambda n } \mathbb { E } _ { \sigma } \left[ \log \exp \left( \lambda \underset { h \in \mathcal { H } } { \operatorname* { m a x } } \langle \sigma , h \rangle \right) \right] } \\ { = \displaystyle \frac { 1 } { \lambda n } \mathbb { E } _ { \sigma } \left[ \log \underset { h \in \mathcal { H } } { \operatorname* { m a x } } \exp \left( \lambda \langle \sigma , h \rangle \right) \right] } \\ { \leq \displaystyle \frac { 1 } { \lambda n } \mathbb { E } _ { \sigma } \left[ \log \underset { h \in \mathcal { H } } { \sum } \exp \left( \lambda \langle \sigma , h \rangle \right) \right] } \\ { \leq \displaystyle \frac { 1 } { \lambda n } \log \underset { \textnormal { v e \mathcal { H } } } { \sum } \mathbb { E } _ { \sigma } \left[ \exp \left( \lambda \langle \sigma , h \rangle \right) \right] , } \end{array}
$$

where the last step follows from Jensen’s inequality.

For each fixed $h \in { \mathcal { H } }$

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \sigma } \left[ \exp \left( \lambda \langle \sigma , h \rangle \right) \right] = \prod _ { i = 1 } ^ { n } \mathbb { E } _ { \sigma _ { i } } \left[ \exp \left( \lambda \sigma _ { i } h _ { i } \right) \right] } \\ { \displaystyle \quad \leq \prod _ { i = 1 } ^ { n } \exp \left( \frac { \lambda ^ { 2 } v _ { i } ^ { 2 } } { 2 } \right) } \\ { \displaystyle \qquad = \exp \left( \frac { \lambda ^ { 2 } \| v \| ^ { 2 } } { 2 } \right) } \end{array}
$$

$$
\leq \exp \left( \frac { \lambda ^ { 2 } D ^ { 2 } } { 2 } \right) ,
$$

where first inequality follows from the fact that Rademacher variables are sub-Gaussian. Choosing $\lambda = \sqrt { 2 \log | \mathcal { H } | } / D$ and performing a straightforward calculation, we obtain

$$
\mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { h \in \mathcal { H } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } h _ { i } \right] \le \frac { D } { n } \sqrt { 2 \log | \mathcal { H } | } .
$$

□

Proposition 17 (Truncated Dudley Integral) Let $\mathcal { F }$ be a function set. Assume that $0 \in { \mathcal { F } } _ { \mathrm { : } }$ and put $B : = \operatorname* { s u p } _ { f \in { \mathcal { F } } } \| f \| _ { \infty }$ . Then, for every $0 < \alpha < 2 B _ { \mathrm { { i } } }$

$$
\mathrm { R a d } ( \mathcal { F } ) : = \mathbb { E } _ { Z } \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } f ( Z _ { i } ) \right| \right] \leq 4 \alpha + \frac { 1 2 } { \sqrt { n } } \int _ { \alpha } ^ { 2 B } \sqrt { \log N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \tau \big ) } d \tau .
$$

Proof Since $0 \in \mathcal { F }$ and $\| f \| _ { \infty } \leq B$ for all $f \in { \mathcal { F } } _ { \mathrm { : } }$ , we have $\| f \| _ { L ^ { 2 } ( P _ { n } ) } \leq 2 B$ . Thus the singleton {0} is a 2B-net of ${ \mathcal F } .$

Let $\{ \delta _ { j } \} _ { j \ge 0 }$ be a decreasing dyadic sequence with $\delta _ { 0 } ~ = ~ 2 B , \delta _ { i } ~ = ~ 2 ^ { - j } \delta _ { 0 }$ . Choose K such that $\alpha ^ { - } \bar { < } \delta _ { K + 1 } \le 2 \alpha$ . For each $j = 0 , \ldots , K$ , let $T _ { j }$ be a δ<sub>j</sub>-net of $\mathcal { F }$ with $| T _ { j } | =$ $N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \delta _ { j } \big )$ . We take $T _ { 0 } ~ = ~ \{ 0 \}$ . For each $f \in { \mathcal { F } }$ , choose π<sub>j</sub> $f \in T _ { j }$ such that $\parallel f -$ $\pi _ { j } f \| _ { L ^ { 2 } ( P _ { n } ) } \leq \delta _ { j }$ . Then

$$
f = \pi _ { 0 } f + \sum _ { j = 1 } ^ { K } ( \pi _ { j } f - \pi _ { j - 1 } f ) + ( f - \pi _ { K } f ) .
$$

Since $\pi _ { 0 } f = 0$ , we obtain

$$
\begin{array} { r l } & { \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \left| \frac { 1 } { n } \underset { i = 1 } { \overset { n } { \sum } } \sigma _ { i } f ( Z _ { i } ) \right| \leq \displaystyle \sum _ { j = 1 } ^ { K } \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left( \pi _ { j } f ( Z _ { i } ) - \pi _ { j - 1 } f ( Z _ { i } ) \right) \right| + \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } \left( f ( Z _ { i } ) - \pi _ { K } f ( Z _ { i } ) \right) \right| } \\ & { \qquad \leq \displaystyle \sum _ { j = 1 } ^ { K } \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ( \pi _ { j } f ( Z _ { i } ) - \pi _ { j - 1 } f ( Z _ { i } ) ) \right| + \delta _ { K } . } \end{array}
$$

where we used Schwarz inequality in the last step. Taking $\mathbb { E } _ { \sigma }$ on both sides, we have

$$
\mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } f ( Z _ { i } ) \right| \right] \leq \sum _ { j = 1 } ^ { K } \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ( \pi _ { j } f ( Z _ { i } ) - \pi _ { j - 1 } f ( Z _ { i } ) ) \right| \right] + \delta _ { K } .
$$

By Proposition 16, we obtain

$$
\begin{array} { r l } { \displaystyle \mathbb E _ { \sigma } \left[ \displaystyle \operatorname* { s u p } _ { f \in \mathcal F } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ( \pi _ { j } f ( Z _ { i } ) - \pi _ { j - 1 } f ( Z _ { i } ) ) \right| \right] \le \displaystyle \frac { \operatorname* { s u p } _ { f \in \mathcal F } \| \pi _ { j } f - \pi _ { j - 1 } f \| _ { 2 } } { n } \sqrt { 2 \log | T _ { j } | | T _ { j - 1 } | } } & { } \\ { \displaystyle \le \frac { 2 \operatorname* { s u p } _ { f \in \mathcal F } \| \pi _ { j } f - \pi _ { j - 1 } f \| _ { 2 } } { n } \sqrt { \log | T _ { j } | } . } & { } \end{array}
$$

Moreover we have,

$$
\operatorname* { s u p } _ { f \in \mathcal { F } _ { R } } \| \pi _ { j } f - \pi _ { j - 1 } f \| _ { 2 } \leq \operatorname* { s u p } _ { f \in \mathcal { F } _ { R } } \| \pi _ { j } f - f \| _ { 2 } + \operatorname* { s u p } _ { f \in \mathcal { F } _ { R } } \| f - \pi _ { j - 1 } f \| _ { 2 } ,
$$

$$
\begin{array} { l } { \leq \sqrt { n } \delta _ { j } + \sqrt { n } \delta _ { j - 1 } } \\ { \leq \sqrt { n } \delta _ { j } + 2 \sqrt { n } \delta _ { j } } \\ { \leq 3 \sqrt { n } \delta _ { j } . } \end{array}
$$

Thus

$$
\mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } ( \pi _ { j } f ( Z _ { i } ) - \pi _ { j - 1 } f ( Z _ { i } ) ) \right| \right] \le \frac { 6 \delta _ { j } } { \sqrt { n } } \sqrt { \log | T _ { j } | } .
$$

Since $N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \tau \big )$ is nonincreasing in $\tau ,$ we have

$$
\begin{array} { r l } & { \displaystyle \int _ { \delta _ { j + 1 } } ^ { \delta _ { j } } \sqrt { \log N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \tau \big ) } d \tau \geq ( \delta _ { j } - \delta _ { j + 1 } ) \sqrt { \log N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \delta _ { j } \big ) } } \\ & { \qquad \geq \frac { \delta _ { j } } { 2 } \sqrt { \log N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \delta _ { j } \big ) } . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { \mathbb { E } _ { \sigma } \left[ \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } f ( Z _ { i } ) \right| \right] \leq \delta _ { K } + \frac { 6 } { \sqrt { n } } \sum _ { j = 1 } ^ { K } \delta _ { j } \sqrt { \log N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \delta _ { j } \big ) } } \\ { \leq 4 \alpha + \frac { 1 2 } { \sqrt { n } } \int _ { \alpha } ^ { 2 B } \sqrt { \log N \big ( \mathcal { F } , L ^ { 2 } ( P _ { n } ) , \tau \big ) } d \tau , } \end{array}
$$

noting that $\alpha < \delta _ { K + 1 } \leq 2 \alpha$ . Taking expectation over $Z = ( Z _ { 1 } , \ldots , Z _ { n } )$ gives the desired bound. □

## References

[1] Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. Communications of the ACM 60(6), 84–90 (2017)

[2] Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)

[3] Hinton, G., Deng, L., Yu, D., Dahl, G.E., Mohamed, A.-r., Jaitly, N., Senior, A., Vanhoucke, V., Nguyen, P., Sainath, T.N., et al.: Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal processing magazine 29(6), 82–97 (2012)

[4] Vapnik, V.N.: Statistical Learning Theory. Adaptive and Learning Systems for Signal Processing, Communications, and Control. John Wiley & Sons, New York (1998)

[5] Bartlett, P.L., Mendelson, S.: Rademacher and gaussian complexities: Risk bounds and structural results. Journal of machine learning research 3(Nov), 463–482 (2002)

[6] Zhang, C., Bengio, S., Hardt, M., Recht, B., Vinyals, O.: Understanding deep learning requires rethinking generalization. In: International Conference on Learning Representations (2017)

[7] Nagarajan, V., Kolter, J.Z.: Uniform convergence may be unable to explain generalization in deep learning. In: Advances in Neural Information Processing Systems, vol. 32 (2019)

[8] Wei, S., Murfet, D., Gong, M., Li, H., Gell-Redman, J., Quella, T.: Deep learning is singular, and that’s good. IEEE Transactions on Neural Networks and Learning Systems (2022)

[9] Lau, E., Furman, Z., Wang, G., Murfet, D., Wei, S.: The local learning coefficient: A singularity-aware complexity measure. In: Proceedings of The 28th International Conference on Artificial Intelligence and Statistics (2025)

[10] Watanabe, S.: Algebraic analysis for nonidentifiable learning machines. Neural Computation 13(4), 899–933 (2001)

[11] Aoyagi, M.: Consideration on the learning eficiency of multiple-layered neural networks with linear units. Neural Netw. 172(106132), 106132 (2024)

[12] Watanabe, S.: A widely applicable bayesian information criterion. The Journal of Machine Learning Research 14(1), 867–897 (2013)

[13] Watanabe, S.: Algebraic geometrical methods for hierarchical learning machines. Neural Netw. 14(8), 1049–1060 (2001)

[14] Watanabe, S.: Algebraic Geometry and Statistical Learning Theory. Cambridge Monographs on Applied and Computational Mathematics, vol. 25. Cambridge University Press, Cambridge (2009)

[15] Hoogland, J., Wang, G., Farrugia-Roberts, M., Carroll, L., Wei, S., Murfet, D.: Loss landscape degeneracy and stagewise development in transformers. Transactions on Machine Learning Research (2025)

[16] Wang, G., Hoogland, J., Wingerden, S., Furman, Z., Murfet, D.: Diferentiation and specialization of attention heads via the refined local learning coeficient. In: International Conference on Learning Representations, vol. 2025, pp. 101037– 101082 (2025)

[17] Singh, S.P., Bachmann, G., Hofmann, T.: Analytic insights into structure and rank of neural network hessian maps. In: Advances in Neural Information Processing Systems, vol. 34, pp. 23914–23927 (2021)

[18] Singh, S.P., Hofmann, T., Sch¨olkopf, B.: The hessian perspective into the nature of convolutional neural networks. In: Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 31930–31968 (2023)

[19] MacKay, D.J.C.: Bayesian interpolation. Neural Comput. 4(3), 415–447 (1992)

[20] Maddox, W.J., Benton, G., Wilson, A.G.: Rethinking parameter counting in deep models: Efective dimensionality revisited. arXiv [cs.LG] (2020) [cs.LG]

[21] Amari, S.-i., Ozeki, T., Park, H.: Learning and inference in hierarchical models with singularities. Systems and Computers in Japan 34(7), 34–42 (2003)

[22] Amari, S.-i., Park, H., Ozeki, T.: Singularities afect dynamics of learning in neuromanifolds. Neural computation 18(5), 1007–1065 (2006)

[23] Fukumizu, K.: Generalization error of linear neural networks in unidentifiable cases. In: International Conference on Algorithmic Learning Theory, pp. 51–62 (1999). Springer

[24] Watanabe, S., Opper, M.: Asymptotic equivalence of bayes cross validation and widely applicable information criterion in singular learning theory. Journal of machine learning research 11(12) (2010)

[25] Hochreiter, S., Schmidhuber, J.: Flat minima. Neural computation 9(1), 1–42 (1997)

[26] Keskar, N.S., Mudigere, D., Nocedal, J., Smelyanskiy, M., Tang, P.T.P.: On large-batch training for deep learning: Generalization gap and sharp minima. In: International Conference on Learning Representations (2017)

[27] Dinh, L., Pascanu, R., Bengio, S., Bengio, Y.: Sharp minima can generalize for deep nets. ICML 70, 1019–1028 (2017)

[28] Nagayasu, S., Watanabe, S.: Free energy of bayesian convolutional neural network with skip connection. arXiv [cs.LG], 927–942 (2023) [cs.LG]

[29] Bousquet, O., Elisseef, A.: Stability and generalization. J. Mach. Learn. Res. 2(Mar), 499–526 (2002)

[30] Watanabe, S.: Algebraic analysis for non-regular learning machines. In: Advances in Neural Information Processing Systems, vol. 12, pp. 356–362 (1999)

[31] Watanabe, S.: Equations of states in singular statistical estimation. Neural Networks 23(1), 20–34 (2010)

[32] Lahlou, S., Jain, M., Nekoei, H., Butoi, V.I., Bertin, P., Rector-Brooks, J., Korablyov, M., Bengio, Y.: DEUP: Direct epistemic uncertainty prediction. Transactions on Machine Learning Research (2023). Expert Certification

[33] Berezniuk, O., Figalli, A., Ghigliazza, R., Musaelian, K.: A scale-dependent notion of efective dimension. arXiv [stat.ML] (2020) [stat.ML]

[34] Datres, M., Leonardi, G.P., Figalli, A., Sutter, D.: A two-scale complexity measure for deep learning models. In: Advances in Neural Information Processing Systems,

vol. 37, pp. 23013–23039 (2024)

[35] Sun, K., Nielsen, F.: A geometric modeling of occam’s razor in deep learning. Information Geometry 8(S1), 233–273 (2025)

[36] Wainwright, M.J.: High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics, vol. 48. Cambridge University Press, Cambridge (2019)

[37] Gin´e, E., Nickl, R.: Mathematical Foundations of Infinite-Dimensional Statistical Models, Revised paperback edition edn. Cambridge Series in Statistical and Probabilistic Mathematics, vol. 40. Cambridge University Press, Cambridge (2021)

[38] Neyshabur, B., Tomioka, R., Srebro, N.: Norm-based capacity control in neural networks. In: Conference on Learning Theory, pp. 1376–1401 (2015). PMLR

[39] Bartlett, P., Foster, D.J., Telgarsky, M.: Spectrally-normalized margin bounds for neural networks. Neural Inf Process Syst 30, 6240–6249 (2017)

[40] Golowich, N., Rakhlin, A., Shamir, O.: Size-independent sample complexity of neural networks. In: Bubeck, S., Perchet, V., Rigollet, P. (eds.) Proceedings of the 31st Conference On Learning Theory (2018)

[41] Atiyah, M.F.: Resolution of singularities and division of distributions. Commun. Pure Appl. Math. 23(2), 145–150 (1970)

[42] Kashiwara, M.: B-functions and holonomic systems. Inventiones mathematicae 38(1), 33–53 (1976)

[43] Watanabe, S.: Algebraic analysis for nonidentifiable learning machines. Neural Comput. 13(4), 899–933 (2001)

[44] Aoyagi, M., Watanabe, S.: Stochastic complexities of reduced rank regression in bayesian estimation. Neural Networks 18(7), 924–933 (2005)

[45] Watanabe, K., Watanabe, S.: Stochastic complexities of gaussian mixtures in variational bayesian approximation. J. Mach. Learn. Res. 7(22), 625–644 (2006)

[46] Bishop, C.M.: Pattern Recognition and Machine Learning by Christopher M. Bishop. Springer, New York, NY (2006)

[47] Goodfellow, I., Bengio, Y., Courville, A.: Deep Learning. MIT Press, Cambridge, MA (2016)

[48] Smith, S.L., Le, Q.V.: A Bayesian perspective on generalization and stochastic gradient descent. In: International Conference on Learning Representations (2018)

[49] Bunea, F., She, Y., Wegkamp, M.H.: Optimal selection of reduced rank estimators of high-dimensional matrices. The Annals of Statistics 39(2), 1282–1309 (2011)

[50] Ledent, A., Alves, R., Lei, Y.: Generalization bounds for rank-sparse neural networks. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)

[51] Xie, Z., Tang, Q.-Y., Sun, M., Li, P.: On the overlooked structure of stochastic gradients. In: Advances in Neural Information Processing Systems, vol. 36, pp. 66257–66276 (2023)

[52] Karakida, R., Akaho, S., Amari, S.-I.: Pathological spectra of the fisher information metric and its variants in deep neural networks. Neural Comput. 33(8), 2274–2307 (2021)

[53] Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al.: Improving language understanding by generative pre-training (2018)

[54] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding, pp. 4171–4186. Association for Computational Linguistics, Minneapolis, Minnesota (2019)

[55] Hironaka, H.: Resolution of singularities of an algebraic variety over a field of characteristic zero: Ii. Annals of Mathematics 79(2), 205–326 (1964)